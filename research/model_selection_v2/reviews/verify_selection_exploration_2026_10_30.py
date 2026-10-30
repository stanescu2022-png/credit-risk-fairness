"""Independent saved-evidence reconstruction; no production analysis imports."""
from pathlib import Path
import hashlib
import itertools
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'runs/model_selection_v2/selection_exploration_v1_2026-10-30'
OUT = ROOT / 'runs/model_selection_v2/selection_exploration_review_2026-10-30'
ORDER = ('logistic', 'decision_tree', 'random_forest', 'gradient_boosting')
FAMILIES = ('repayment', 'billing', 'payment')
SCENARIOS = ('clean',) + tuple(f'{f}__{m}' for f in FAMILIES for m in ('mask_only', 'corruption_only', 'joint'))
LOSSES = ('brier', 'log_loss')
MARGINAL = (1, 2, 4, 5, 7, 8)
JOINT = (3, 6, 9)
AXES = ('clean_brier', 'clean_logloss', 'marginal_max_brier', 'marginal_max_logloss', 'joint_max_brier', 'joint_max_logloss')
MAX_ERROR = 0.0


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected):
    global MAX_ERROR
    error = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    MAX_ERROR = max(MAX_ERROR, error)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)


def coordinates(loss):
    return np.concatenate((loss[:, 0, :], loss[:, MARGINAL, :].max(axis=1), loss[:, JOINT, :].max(axis=1)), axis=1)


def decisions(loss, safe):
    vector = coordinates(loss)
    result = {}

    def choose(name, ids, scores, minimum=False):
        target = min(scores[i] for i in ids) if minimum else max(scores[i] for i in ids)
        ties = [i for i in ids if scores[i] == target] if minimum else [i for i in ids if abs(scores[i]-target) <= 1e-12]
        result[name] = {'selected_id': ORDER[ties[0]], 'tied_ids': [ORDER[i] for i in ties], 'retained_ids': [ORDER[i] for i in ids]}

    for agg in ('arithmetic', 'geometric', 'quadratic'):
        choose('safe_'+agg, range(4), [safe[n][agg] for n in ORDER])
    for j, suffix in enumerate(('brier', 'logloss')):
        choose('clean_'+suffix, range(4), loss[:, 0, j], True)
        choose('worst_'+suffix, range(4), loss[:, :, j].max(axis=1), True)
    for name, width in (('screened_safe_clean', 2), ('screened_safe_marginal', 4), ('screened_safe_outcome_v0', 6)):
        x = vector[:, :width]
        dominates = (x[:, None, :] <= x[None, :, :]).all(axis=2) & (x[:, None, :] < x[None, :, :]).any(axis=2)
        retained = np.flatnonzero(~dominates.any(axis=0))
        choose(name, retained, [safe[n]['arithmetic'] for n in ORDER])
    return result


def interval(values, saved):
    close(np.percentile(values, (2.5, 50, 97.5), method='linear'), [saved['lower'], saved['median'], saved['upper']])
    assert saved['valid_draws'] == 2000 and saved['undefined_draws'] == 0


def main():
    receipt = read(RUN / 'VERIFIED.json')
    hashes = {p.relative_to(RUN).as_posix(): sha(p) for p in RUN.rglob('*') if p.is_file() and p.name != 'VERIFIED.json'}
    assert receipt['status'] == 'verified' and receipt['artifact_hashes'] == hashes
    manifest = read(RUN / 'run_manifest.json')
    for name, digest in manifest['source_hashes'].items():
        assert sha(ROOT / name) == digest
    preservation = read(RUN / 'preservation.json')
    for name, digest in preservation['before_hashes'].items():
        assert sha(ROOT / name) == digest
    meta = pd.read_csv(RUN / 'selection_identity.csv')
    assert len(meta) == 4801 and meta.ID.is_unique
    y = meta.default.to_numpy()
    safe = read(RUN / 'saved_safe_scores.json')
    groups = tuple(dict.fromkeys(meta.duplicate_group_id))
    mapping = {g: i for i, g in enumerate(groups)}
    positions = np.array([mapping[g] for g in meta.duplicate_group_id])
    with np.load(RUN / 'bootstrap_multiplicities.npz', allow_pickle=False) as file:
        counts = file['group_multiplicities']
        np.testing.assert_array_equal(file['row_group_positions'], positions)
        np.testing.assert_array_equal(file['group_order'], groups)
        np.testing.assert_array_equal(file['ID'], meta.ID)
    samples = np.random.default_rng(42).integers(0, len(groups), size=(2000, len(groups)))
    np.testing.assert_array_equal(counts, np.array([np.bincount(row, minlength=len(groups)) for row in samples]))
    sizes = np.bincount(positions, minlength=len(groups))
    summaries = {}
    for qname in ('q0.10', 'q0.20'):
        folder = RUN / qname
        saved_mean = read(folder / 'scenario_matrix.json')
        saved_rep = read(folder / 'per_realization_losses.json')
        saved_endpoints = read(folder / 'logloss_endpoint_counts.json')
        point = np.empty((4, 10, 2))
        record = np.empty((len(meta), 4, 10, 2))
        for c, name in enumerate(ORDER):
            with np.load(folder / f'predictions/{name}.npz', allow_pickle=False) as predictions:
                np.testing.assert_array_equal(predictions['ID'], meta.ID)
                for s, scenario in enumerate(SCENARIOS):
                    p = predictions[scenario]
                    assert p.shape == (1 if s == 0 else 30, len(meta))
                    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
                    clipped = np.clip(p, 2.**-52, 1-2.**-52)
                    values = ((p-y)**2, -y*np.log(clipped)-(1-y)*np.log1p(-clipped))
                    for j, metric in enumerate(LOSSES):
                        per_rep = values[j].mean(axis=1)
                        point[c, s, j] = per_rep.mean()
                        record[:, c, s, j] = values[j].mean(axis=0)
                        close(per_rep, saved_rep[name][scenario][metric]['values'])
                        close(point[c, s, j], saved_mean[name][scenario][metric])
                        if s:
                            close(per_rep.std(ddof=1), saved_rep[name][scenario][metric]['mc_sd'])
                    for r, row in enumerate(p):
                        endpoint = saved_endpoints[name][scenario]['per_repetition'][r]
                        assert endpoint['probability_zero'] == int((row == 0).sum())
                        assert endpoint['probability_one'] == int((row == 1).sum())
                        assert endpoint['clipped_below_epsilon'] == int((row < 2.**-52).sum())
                        assert endpoint['clipped_above_one_minus_epsilon'] == int((row > 1-2.**-52).sum())
        chosen = decisions(point, safe)
        saved = read(folder / 'selectors.json')
        close(coordinates(point), [[saved['risk_vectors'][n][axis] for axis in AXES] for n in ORDER])
        for method, choice in chosen.items():
            for field, value in choice.items():
                assert saved['decisions'][method][field] == value, (qname, method, field)
        grouped = np.zeros((len(groups), 4, 10, 2))
        np.add.at(grouped, positions, record)
        draws = ((counts @ grouped.reshape(len(groups), -1))/(counts @ sizes)[:, None]).reshape(2000, 4, 10, 2)
        # Also reconstruct sampled draws directly from record multiplicities.
        for d in (0, 1, 999, 1999):
            close(draws[d], np.average(record, axis=0, weights=counts[d, positions]))
        uncertainty = read(folder / 'conditional_bootstrap.json')
        for c, name in enumerate(ORDER):
            for s, scenario in enumerate(SCENARIOS):
                for j, metric in enumerate(LOSSES):
                    interval(draws[:, c, s, j], uncertainty['candidate_loss_intervals'][name][scenario][metric])
        contrast = np.stack([draws[:, :, 3+3*f, :]-draws[:, :, 1+3*f, :]-draws[:, :, 2+3*f, :]+draws[:, :, 0, :] for f in range(3)], axis=2)
        for c, name in enumerate(ORDER):
            for f, family in enumerate(FAMILIES):
                for j, metric in enumerate(LOSSES):
                    interval(contrast[:, c, f, j], uncertainty['candidate_interaction_intervals'][name][family][metric])
        for a, b in itertools.combinations(range(4), 2):
            pair = uncertainty['paired_fixed_candidate_intervals'][ORDER[b]+'_minus_'+ORDER[a]]
            for s, scenario in enumerate(SCENARIOS):
                for j, metric in enumerate(LOSSES):
                    interval(draws[:, b, s, j]-draws[:, a, s, j], pair['loss_intervals'][scenario][metric])
            for f, family in enumerate(FAMILIES):
                for j, metric in enumerate(LOSSES):
                    interval(contrast[:, b, f, j]-contrast[:, a, f, j], pair['interaction_difference_intervals'][family][metric])
        frequencies = {method: {field: dict.fromkeys(ORDER, 0) for field in ('selected_counts', 'retained_counts', 'tied_choice_membership_counts')} for method in chosen}
        for draw in draws:
            for method, choice in decisions(draw, safe).items():
                frequencies[method]['selected_counts'][choice['selected_id']] += 1
                for field, ids in (('retained_counts', choice['retained_ids']), ('tied_choice_membership_counts', choice['tied_ids'])):
                    for name in ids:
                        frequencies[method][field][name] += 1
        for method, fields in frequencies.items():
            for field, value in fields.items():
                assert value == uncertainty['selector_frequencies'][method][field], (qname, method, field)
        assert len(list((folder / 'registry').glob('*.json'))) == 270
        for path in (folder / 'registry').glob('*.json'):
            reg = read(path)
            assert reg['ordered_row_ids'] == meta.ID.tolist()
            assert len(reg['selected_row_ids']) == int(float(qname[1:])*len(meta))
            assert set(reg['changed_row_ids']) <= set(reg['selected_row_ids'])
            assert set(reg['clean_reused_row_ids']) | set(reg['changed_row_ids']) == set(meta.ID)
            assert not set(reg['clean_reused_row_ids']) & set(reg['changed_row_ids'])
        summaries[qname] = {'choices': {k: v['selected_id'] for k, v in chosen.items()}, 'risk_vectors': saved['risk_vectors'], 'bootstrap_selected_counts': {k: v['selected_counts'] for k, v in frequencies.items()}}
    audits = read(RUN / 'prediction_reuse_audits.json')
    assert len(audits) == 72 and max(a['maximum_absolute_error'] for a in audits) <= 1e-15
    output = {'status': 'verified', 'scope': 'Independent saved-prediction reconstruction; no model prediction, fitting or holdout access',
              'source_run_receipt_sha256': sha(RUN / 'VERIFIED.json'), 'review_source_sha256': sha(Path(__file__)),
              'protected_files_checked': len(preservation['before_hashes']), 'groups': len(groups), 'bootstrap_draws': 2000,
              'maximum_reconstruction_absolute_error': MAX_ERROR, 'reconstruction_tolerance': 2e-14,
              'tolerance_scope': 'Numerical reproduction only; selector/Pareto rules remain exact',
              'checks': ['artifact/source/preservation hashes', 'all predictions/IDs/shapes/endpoints', 'all scenario/repetition losses and MC SD',
                         'all point risk vectors and ten choices/ties/retained sets', 'bootstrap multiplicities and grouping',
                         'all candidate/pair loss and factorial-contrast intervals', 'all bootstrap selected/retained/tie-membership counts',
                         'all 540 registry identity/exposure partitions', '72 saved full-frame reuse audit records'],
              'results': summaries}
    OUT.mkdir(exist_ok=False)
    (OUT / 'review.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (OUT / 'VERIFIED.json').write_text(json.dumps({'status': 'verified', 'artifact_hashes': {'review.json': sha(OUT / 'review.json')}}, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'verified', 'maximum_error': MAX_ERROR, 'choices': {q: s['choices'] for q, s in summaries.items()}}))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
