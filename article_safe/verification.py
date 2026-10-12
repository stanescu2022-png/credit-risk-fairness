"""Verify a complete independent rerun and preserve a machine-readable receipt.

Usage: python -m article_safe.verification --reference <id> --replicate <id>
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from article_safe.artifacts import existing_run, verify_preparation, verify_stage, write_json
from article_safe.data import check_evaluation_artifacts
from article_safe.inference import paired_delong, group_row_weights
from article_safe.selection import choose_forward_prefix
from article_safe.protocol import ROOT, sha256
from src.baseline import DATA_PATH, load_split
from src.performance_evaluation import performance_metrics

# Runtime is not a scientific result. All other compared fields remain checked.
JSON_RUNTIME_FIELDS = {'completed_at_utc', 'time_utc', 'seconds', 'started_at_utc'}


def strip_runtime(value):
    if isinstance(value, dict):
        return {key: strip_runtime(item) for key, item in value.items() if key not in JSON_RUNTIME_FIELDS}
    if isinstance(value, list):
        return [strip_runtime(item) for item in value]
    return value


def assert_json_equal(first, second, location='root'):
    if isinstance(first, dict):
        assert isinstance(second, dict) and first.keys() == second.keys(), location
        for key in first:
            assert_json_equal(first[key], second[key], f'{location}/{key}')
    elif isinstance(first, list):
        assert isinstance(second, list) and len(first) == len(second), location
        for index, (a, b) in enumerate(zip(first, second)):
            assert_json_equal(a, b, f'{location}/{index}')
    elif isinstance(first, float):
        np.testing.assert_allclose(first, second, rtol=1e-12, atol=1e-12, err_msg=location)
    else:
        assert first == second, (location, first, second)


def numerical_rerun(reference, replicate):
    checked = []
    for stage in ('selection', 'evaluation'):
        paths_a = {path.relative_to(reference/stage).as_posix() for path in (reference/stage).rglob('*')
                   if path.is_file() and path.suffix in ('.csv', '.npz', '.json') and path.name != 'stage_manifest.json'}
        paths_b = {path.relative_to(replicate/stage).as_posix() for path in (replicate/stage).rglob('*')
                   if path.is_file() and path.suffix in ('.csv', '.npz', '.json') and path.name != 'stage_manifest.json'}
        assert paths_a == paths_b, (stage, paths_a ^ paths_b)
        for relative in sorted(paths_a):
            a, b = reference/stage/relative, replicate/stage/relative
            omitted = []
            if a.suffix == '.csv':
                frame_a = pd.read_csv(a, float_precision='round_trip')
                frame_b = pd.read_csv(b, float_precision='round_trip')
                omitted = [column for column in frame_a if column == 'fit_seconds']
                pd.testing.assert_frame_equal(frame_a.drop(columns=omitted), frame_b.drop(columns=omitted),
                                              check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
            elif a.suffix == '.npz':
                with np.load(a, allow_pickle=False) as arr_a, np.load(b, allow_pickle=False) as arr_b:
                    assert set(arr_a.files) == set(arr_b.files), relative
                    for key in arr_a.files:
                        if np.issubdtype(arr_a[key].dtype, np.number):
                            np.testing.assert_allclose(arr_a[key], arr_b[key], atol=1e-12, rtol=1e-12, equal_nan=True)
                        else:
                            np.testing.assert_array_equal(arr_a[key], arr_b[key])
            else:
                assert_json_equal(strip_runtime(json.loads(a.read_text())), strip_runtime(json.loads(b.read_text())), relative)
                omitted = sorted(JSON_RUNTIME_FIELDS)
            checked.append({'artifact': f'{stage}/{relative}', 'passed': True,
                            'byte_identical': sha256(a) == sha256(b), 'ignored_runtime_fields': omitted})
    for relative in ('protocol.json', 'development_assignments.csv', 'data_audit.json'):
        assert sha256(reference/relative) == sha256(replicate/relative), relative
        checked.append({'artifact': relative, 'passed': True, 'byte_identical': True})
    return checked


def independent_saved_checks(run, protocol):
    choices = json.loads((run/'selection/selected_subsets.json').read_text())
    validation = pd.read_csv(run/'selection/gbm_validation_predictions.csv', float_precision='round_trip')
    path = pd.read_csv(run/'selection/gbm_prefix_path.csv', float_precision='round_trip')
    steps = []
    for k in range(2, 23):
        result = paired_delong(validation.default, validation[f'k_{k}'], validation[f'k_{k-1}'])
        saved = path.loc[path.k == k].iloc[0]
        np.testing.assert_allclose([result.difference, result.p_value], [saved.gain, saved.p_value], atol=1e-14)
        steps.append({'k': k, 'gain': result.difference, 'p_value': result.p_value, 'status': result.status})
    assert choose_forward_prefix(steps)['selected_k'] == choices['gbm']['selected_k'] == 3
    assert choices['lr_mle']['status'] == 'not_estimable'
    parts = load_split(pd.read_csv(DATA_PATH))
    y, metadata = parts[3].to_numpy(), parts[5]
    predictions = pd.read_csv(run/'evaluation/holdout_predictions.csv', float_precision='round_trip')
    metrics = pd.read_csv(run/'evaluation/holdout_metrics.csv', float_precision='round_trip').set_index('model')
    assert set(predictions.model) == {'gbm_full', 'gbm_selected', 'lr_frozen_regularized'}
    for role in ('lr_mle_full', 'lr_mle_selected'):
        assert metrics.loc[role, 'status'] == 'not_estimable'
        assert pd.isna(metrics.loc[role, 'ROC-AUC'])
    probabilities = {}
    for model, frame in predictions.groupby('model', sort=False):
        np.testing.assert_array_equal(frame.ID, metadata.ID)
        np.testing.assert_array_equal(frame.SEX, metadata.SEX)
        np.testing.assert_array_equal(frame.actual, y)
        probability = frame.probability_default.to_numpy()
        np.testing.assert_array_equal(frame.predicted, probability > .50)
        for quantity, value in performance_metrics(y, probability).items():
            np.testing.assert_allclose(value, metrics.loc[model, quantity], atol=1e-12)
        probabilities[model] = probability
    with np.load(run/'evaluation/bootstrap_replicates.npz') as saved:
        models, features = saved['explanation_models'].tolist(), saved['features'].tolist()
        phi = {model: np.load(run/f'evaluation/{model}_shap.npz')['original'] for model in models}
        rng = np.random.default_rng(42)
        sex = metadata.SEX.to_numpy()
        for draw in range(10):
            weights = group_row_weights(metadata.duplicate_group_id.to_numpy(), rng)
            delta = roc_auc_score(y, probabilities['gbm_selected'], sample_weight=weights) - roc_auc_score(y, probabilities['gbm_full'], sample_weight=weights)
            np.testing.assert_allclose(delta, saved['auc_differences'][draw, 3], atol=1e-12)
            for i, model in enumerate(models):
                for j in range(len(features)):
                    for offset, values in ((0, phi[model]), (len(models)*len(features), np.abs(phi[model]))):
                        gap = np.average(values[sex == 2, j], weights=weights[sex == 2]) - np.average(values[sex == 1, j], weights=weights[sex == 1])
                        np.testing.assert_allclose(gap, saved['explanation_gaps'][draw, offset+i*len(features)+j], atol=1e-12)
        interval = pd.read_csv(run/'evaluation/auc_difference_intervals.csv').iloc[3]
        np.testing.assert_allclose([interval.ci_lower, interval.ci_upper], np.quantile(saved['auc_differences'][:, 3], [.025, .975]), atol=1e-12)
    events = json.loads((run/'evaluation/execution_events.json').read_text())
    boundary = next(i for i, event in enumerate(events) if event['event'] == 'holdout_evaluation_started')
    assert all(i < boundary for i, event in enumerate(events) if event['event'] == 'final_refit_finished')
    manifest = json.loads((run/'evaluation/stage_manifest.json').read_text())
    assert manifest['selected_subsets_sha256'] == sha256(run/'selection/selected_subsets.json')
    assert json.loads((run/'evaluation/full_gbm_reproduction.json').read_text())['passed']
    profiles = pd.read_csv(run/'report/model_profiles.csv', float_precision='round_trip').set_index('model')
    assert profiles.index.tolist() == protocol['final_roles'] + [protocol['estimator_reference']]
    for column in profiles:
        pd.testing.assert_series_equal(profiles[column], metrics.loc[profiles.index, column], check_names=False)
    for name in ('gbm_selection_path', 'gbm_full_vs_selected', 'group_contributions'):
        assert all((run/f'report/figures/{name}.{suffix}').stat().st_size > 1000 for suffix in ('png', 'svg'))
    return {'saved_selection_replayed': True, 'holdout_metrics_recomputed': True,
            'first_10_bootstrap_draws_independently_checked': True, 'final_fits_precede_holdout': True,
            'missing_LR_roles_visible_in_report': True, 'full_GBM_matches_frozen_reference': True,
            'report_profiles_match_saved_metrics': True}


def test_suite(relative, destination):
    arguments = [sys.executable, '-m', 'unittest', 'discover', '-s', relative, '-v']
    result = subprocess.run(arguments, cwd=ROOT, text=True, capture_output=True, timeout=600)
    output = result.stdout + result.stderr
    (destination / ('article_tests.log' if relative.startswith('article_safe') else 'existing_tests.log')).write_text(output, encoding='utf-8')
    if result.returncode != 0:
        raise RuntimeError(f'Tests failed: {relative}\n{output[-4000:]}')
    match = re.search(r'Ran (\d+) tests?', output)
    if not match:
        raise RuntimeError('Cannot recover executed test count.')
    count = int(match.group(1))
    print(f'{relative}: {count} tests passed', flush=True)
    return {'command': arguments, 'tests_run': count, 'exit_code': result.returncode}


def verify(reference_id, replicate_id):
    reference, replicate = existing_run(reference_id), existing_run(replicate_id)
    if reference == replicate:
        raise ValueError('Numerical reproduction requires an independently executed fresh run.')
    destination = ROOT/'article_safe/milestones/checkpoint_04'
    if destination.exists():
        raise FileExistsError('A final verification destination already exists; do not overwrite it.')
    for run in (reference, replicate):
        protocol = verify_preparation(run)
        for stage in ('selection', 'evaluation', 'report'):
            verify_stage(run/stage, protocol)
    assert_json_equal(json.loads((reference/'protocol.json').read_text()), json.loads((replicate/'protocol.json').read_text()))
    before = check_evaluation_artifacts()
    checked = numerical_rerun(reference, replicate)
    independent = independent_saved_checks(reference, protocol)
    independent_saved_checks(replicate, protocol)
    original = json.loads((ROOT/'article_safe/milestones/checkpoint_01/preexisting_project_inventory.json').read_text())
    changed = [relative for relative, digest in original.items() if not (ROOT/relative).is_file() or sha256(ROOT/relative) != digest]
    assert set(changed) <= {'PROJECT_CONTEXT.md'}, changed
    destination.mkdir(parents=True)
    isolated = test_suite('article_safe/tests', destination)
    existing = test_suite('tests', destination)
    after = check_evaluation_artifacts()
    assert before == after
    receipt = {'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'client_date': '2026-10-10',
               'status': 'implementation_complete_with_explicit_unestimable_LR_roles',
               'reference_run': reference_id, 'independent_rerun': replicate_id,
               'numerical_tolerance': {'rtol': 1e-12, 'atol': 1e-12},
               'rerun_artifacts': checked, 'rerun_artifact_count': len(checked),
               'independent_checks': independent, 'isolated_tests': isolated, 'existing_tests': existing,
               'prior_evaluation_hash_checks': after['checks'], 'unexpected_preexisting_changes': [],
               'pooling_performed': False, 'regularized_LR_substituted_for_article_tests': False,
               'scientific_claim': 'GBM reduction trade-off; no framework superiority established',
               'verification_source_sha256': sha256(Path(__file__)),
               'report_manifest_sha256': sha256(reference/'report/stage_manifest.json')}
    write_json(destination/'verification.json', receipt)
    print(json.dumps({key: receipt[key] for key in ('status', 'rerun_artifact_count', 'prior_evaluation_hash_checks', 'pooling_performed')}, indent=2))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--replicate', required=True)
    args = parser.parse_args()
    verify(args.reference, args.replicate)


if __name__ == '__main__':
    main()