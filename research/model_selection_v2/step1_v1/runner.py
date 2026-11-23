"""Refit/reproduce first; execute exactly 50 development-only repeats if gates pass."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from research.model_selection_v2 import pilot as p, safe2026 as s
from research.model_selection_v2.judge_core.tables import PredictionTable
from research.model_selection_v2.judge_core.metrics import losses
from research.model_selection_v2.judge_core.bootstrap import group_bootstrap_plan
from research.model_selection_v2.selection_exploration_v1.transforms import (
    paired_transforms, predict_changed_rows, reference_bundle)
from research.model_selection_v2.selection_exploration_v1.protocol import FAMILIES, CONDITIONS
from .core import (MILD, FULL, ARMS, SEXES, skill, score, ranking, choice,
                   group_means, fairness, percentile)

ROOT = p.ROOT
PILOT = ROOT/'runs/model_selection_v2/pilot_candidates_2026-10-26'
SAFE = ROOT/'runs/model_selection_v2/safe2026_swapping_2026-10-26'
STEP0 = ROOT/'runs/model_selection_v2/step0_2026-11-14'
SPEC = 'Step 1 specification accuracy-anchored SAFE versus published SAFE.docx'


def write(path, value):
    p.write_json(path, value)


def npz(path, **values):
    with Path(path).open('xb') as f:
        np.savez_compressed(f, **values)


def table(meta, probability, role):
    return PredictionTable(tuple(meta.ID), meta.default.to_numpy(), meta.SEX.to_numpy(),
                           tuple(meta.duplicate_group_id), probability, role)


def csv(path, frame):
    p.write_csv(path, frame)


def inputs(run):
    """Only development predictors/outcomes; historical IDs read for exclusion."""
    paths = [PILOT/'development_assignments.csv', PILOT/'protocol.json',
             p.DATA_PATH, p.DATA_PATH.parent/'train_metadata.csv',
             p.DATA_PATH.parent/'split_metadata.csv', p.DATA_PATH.parent/'preparation_manifest.json',
             SAFE/'integrated_scores.json', SAFE/'component_areas.csv', STEP0/'brier/orderings.csv',
             ROOT/'models/baseline_logistic_manifest.json']
    prep = json.loads((p.DATA_PATH.parent/'preparation_manifest.json').read_text())
    for path, key in [(p.DATA_PATH, 'processed_sha256'),
                      (p.DATA_PATH.parent/'split_metadata.csv', 'split_sha256')]:
        p.require(p.sha256(path) == prep[key], f'Data identity changed: {path}')
    membership = pd.read_csv(p.DATA_PATH.parent/'split_metadata.csv', usecols=['ID', 'split'])
    trainrows = set(np.flatnonzero(membership.split.eq('train')))
    order = json.loads((PILOT/'protocol.json').read_text())['feature_order']
    dev = pd.read_csv(p.DATA_PATH, usecols=['ID']+order,
                     skiprows=lambda row: row > 0 and row-1 not in trainrows)
    meta = pd.read_csv(p.DATA_PATH.parent/'train_metadata.csv')
    X = dev.set_index('ID').loc[meta.ID, order].reset_index(drop=True)
    assignment = pd.read_csv(PILOT/'development_assignments.csv')
    positions = p.validate_assignments(X, meta.default, meta, assignment)
    historical_ids = set(membership.loc[membership.split.eq('test'), 'ID'])
    p.require(len(X) == 24004 and len(historical_ids) == 5996 and
              not set(meta.ID)&historical_ids, 'Development/historical ID exclusion failed')
    p.require(set(X.columns) == set(p.FEATURES) and 'SEX' not in X.columns, 'Invalid predictors')
    for c in p.CANDIDATES:
        paths.append(PILOT/f'predictions/{c}.csv')
    write(run/'input_hashes.json', p.hashes(paths))
    write(run/'data_gate.json', {'passed': True, 'development_records': len(X),
          'historical_ids_checked_only': len(historical_ids), 'historical_predictors_or_outcomes_loaded': False,
          'feature_order': order, 'roles_original': {r: len(v) for r, v in positions.items()}})
    return X, meta, positions, historical_ids


def split(meta, repeat, historical_ids):
    groups = np.asarray(list(dict.fromkeys(meta.duplicate_group_id)))
    perm = np.random.default_rng(20261114+repeat).permutation(groups)
    sizes = meta.duplicate_group_id.value_counts().to_dict()
    assigned, counts = {}, {'judge': 0, 'selection': 0, 'fitting': 0}
    for g in perm:
        role = 'judge' if counts['judge'] < .20*len(meta) else (
            'selection' if counts['selection'] < .16*len(meta) else 'fitting')
        assigned[g] = role
        counts[role] += sizes[g]
    out = meta.copy()
    out['role'] = out.duplicate_group_id.map(assigned)
    p.require(not set(out.ID)&historical_ids and len(out) == 24004 and out.ID.is_unique,
              'Repeat historical-ID or partition gate failed')
    p.require(out.groupby('duplicate_group_id').role.nunique().max() == 1, 'Group crosses roles')
    positions, strata = {}, []
    for role in ('fitting', 'selection', 'judge'):
        positions[role] = np.flatnonzero(out.role.eq(role))
        for sex in (2, 1):
            sub = out[out.role.eq(role)&out.SEX.eq(sex)]
            p.require(set(sub.default) == {0, 1}, f'Invalid {role} sex {sex} outcome support')
            strata.append({'role': role, 'SEX': sex, 'records': len(sub), 'defaults': int(sub.default.sum())})
    p.require(sum(map(len, positions.values())) == len(out), 'Incomplete partition')
    return out, positions, {'seed': 20261114+repeat, 'sizes': counts, 'sex_counts': strata,
            'group_order_before_permutation': 'first occurrence in original development order',
            'threshold_rule': 'add whole groups until judge >=20% and selection >=16% of 24004',
            'passed': True, 'historical_ID_overlap': 0, 'crossing_groups': 0}


def fit_models(folder, Xfit, metafi):
    models, seconds = {}, {}
    (folder/'models').mkdir()
    for c in p.CANDIDATES:
        started = time.perf_counter()
        m = p.build_candidate(c)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            m.fit(Xfit, metafi.default)
        p.require(not any(issubclass(w.category, ConvergenceWarning) for w in caught), f'{c} failed convergence')
        seconds[c] = {'seconds': time.perf_counter()-started,
                      'warnings': [{'category': w.category.__name__, 'message': str(w.message)} for w in caught]}
        p.positive_probability(m, Xfit)
        joblib.dump(m, folder/f'models/{c}.joblib')
        models[c] = m
    write(folder/'fit_audit.json', seconds)
    return models


def selection(folder, Xfit, Xsel, meta, models, *, mild=True):
    full = {c: p.positive_probability(models[c], Xsel) for c in p.CANDIDATES}
    references = s.fitting_references(Xfit)
    write(folder/'masking_references.json', references)
    swap = {c: [] for c in p.CANDIDATES}
    mildp = {c: [] for c in p.CANDIDATES}
    exposure = []
    for strength in FULL:
        changed, audit = s.tail_swap(Xsel, meta.ID, float(strength))
        exposure.append(audit)
        for c in p.CANDIDATES:
            swap[c].append(p.positive_probability(models[c], changed))
    if mild:
        for strength in MILD:
            changed, audit = s.tail_swap(Xsel, meta.ID, float(strength))
            for c in p.CANDIDATES:
                mildp[c].append(p.positive_probability(models[c], changed))
    write(folder/'swap_exposure.json', exposure)
    rows, evidence = [], {}
    for c in p.CANDIDATES:
        a = s.accuracy_curve(meta.default, full[c])
        p.require(a['status'] == 'defined', 'Undefined accuracy curve')
        removal = s.greedy_preservation(Xsel.columns, full[c],
                     lambda removed: p.positive_probability(models[c], s.mask_original(Xsel, removed, references)))
        p.require(removal['status'] == 'defined' and len(removal['alternatives']) == 253,
                  f'Undefined/incomplete greedy path: {c}')
        q = np.asarray(swap[c])
        r = np.array([s.rank_graduation(full[c], v).value for v in q], dtype=float)
        p.require(np.isfinite(r).all(), 'Undefined rank preservation')
        integrated = s.integrate(a['values'], removal['values'], r)
        p.require(integrated['status'] == 'defined', 'Undefined SAFE score')
        av, ev, rv = np.mean(a['values']), np.mean(removal['values']), r.mean()
        row = {'candidate': c, 'A_bar': float(av), 'E_bar': float(ev), 'R_safe_bar': float(rv),
               'SAFE_score': integrated['arithmetic'], 'clean_brier': float(np.mean((full[c]-meta.default)**2)),
               'S_full_bar': float(skill(meta.default, q).mean())}
        if mild:
            qm = np.asarray(mildp[c])
            rm = np.r_[1., [s.rank_graduation(full[c], v).value for v in qm[1:]]]
            p.require(np.isfinite(rm).all(), 'Undefined mild rank preservation')
            sm = skill(meta.default, qm)
            row.update(S_bar=float(sm.mean()), R_mild_bar=float(rm.mean()),
                       Score2=score(av, ev, sm.mean()), order_mild_score=score(av, ev, rm.mean()),
                       accuracy_full_score=score(av, ev, row['S_full_bar']))
        arrays = {'ID': meta.ID.to_numpy(), 'SEX': meta.SEX.to_numpy(), 'y': meta.default.to_numpy(),
                  'groups': meta.duplicate_group_id.to_numpy(), 'clean': full[c],
                  'swap_full': q, 'mask_path': np.asarray(removal['predictions']),
                  'A': a['values'], 'E': removal['values'], 'R': r}
        if mild:
            arrays.update(swap_mild=qm, S_mild=sm, R_mild=rm, S_full=skill(meta.default, q))
        npz(folder/f'selection_{c}.npz', **arrays)
        write(folder/f'curves_{c}.json', {'accuracy': a, 'integrated': integrated,
              'removal': {k: v for k, v in removal.items() if k != 'predictions'}})
        evidence[c] = arrays
        rows.append(row)
    csv(folder/'selection_values.csv', pd.DataFrame(rows))
    return pd.DataFrame(rows).set_index('candidate'), evidence


def reproduce(run, X, meta, positions):
    folder = run/'reproduction'
    folder.mkdir(exist_ok=False)
    fi, se = positions['fitting'], positions['selection']
    fit, sel, ms = X.iloc[fi].reset_index(drop=True), X.iloc[se].reset_index(drop=True), meta.iloc[se].reset_index(drop=True)
    started = time.perf_counter()
    models = fit_models(folder, fit, meta.iloc[fi].reset_index(drop=True))
    # Prediction gate is checked immediately, before any expensive SAFE computation.
    errors = {}
    for c in p.CANDIDATES:
        saved = pd.read_csv(PILOT/f'predictions/{c}.csv', float_precision='round_trip')
        np.testing.assert_array_equal(ms.ID, saved.ID)
        prob = p.positive_probability(models[c], sel)
        errors[c] = float(np.max(np.abs(prob-saved.probability_default.to_numpy())))
        np.testing.assert_allclose(prob, saved.probability_default, rtol=0, atol=1e-12)
    write(folder/'clean_prediction_gate.json', {'passed': True, 'max_abs_errors': errors})
    rows, evidence = selection(folder, fit, sel, ms, models, mild=False)
    components = pd.read_csv(SAFE/'component_areas.csv', float_precision='round_trip').set_index('candidate_id')
    integrated = json.loads((SAFE/'integrated_scores.json').read_text())
    orderings = pd.read_csv(STEP0/'brier/orderings.csv', float_precision='round_trip')
    for c in p.CANDIDATES:
        for new, old in [('A_bar', 'mean_A'), ('E_bar', 'mean_E'), ('R_safe_bar', 'mean_R')]:
            np.testing.assert_allclose(rows.loc[c, new], components.loc[c, old], rtol=0, atol=1e-12)
        got = json.loads((folder/f'curves_{c}.json').read_text())['integrated']
        for key in ('arithmetic', 'geometric', 'quadratic'):
            np.testing.assert_allclose(got[key], integrated[c][key], rtol=0, atol=1e-12)
        expected = orderings[(orderings.model == c)&(orderings.sex == 'all')&orderings.disturbance.str.startswith('swap_')].sort_values('disturbance')
        p.require(len(expected) == 23, 'Step 0 reference nodes missing')
        brier = np.mean((evidence[c]['swap_full']-ms.default.to_numpy())**2, axis=1)
        np.testing.assert_allclose(brier, expected.O4, rtol=0, atol=1e-12)
    write(run/'reproduction_gate.json', {'passed': True, 'tolerance': 1e-12,
          'clean_prediction_max_errors': errors, 'component_means_checked': 12,
          'integrated_scores_checked': 12, 'Step0_full_grid_briers_checked': 92,
          'seconds': time.perf_counter()-started})
    print('Original-split reproduction gates passed.', flush=True)


def select_arms(folder, rows, evidence, meta, repeat):
    safe = ranking(rows.SAFE_score.to_dict())[0]
    ranked = ranking(rows.Score2.to_dict())
    arms = {'safe': safe, 'plain_loss': ranking((-rows.clean_brier).to_dict())[0],
            'order_mild': ranking(rows.order_mild_score.to_dict())[0],
            'accuracy_full': ranking(rows.accuracy_full_score.to_dict())[0],
            'selector2_before_check': ranked[0]}
    plan = group_bootstrap_plan(table(meta, evidence[safe]['clean'], 'selection'), n_draws=2000, seed=42+repeat)
    npz(folder/'bootstrap_plan.npz', ID=meta.ID.to_numpy(), group_order=plan.group_order,
        group_multiplicities=plan.group_multiplicities, row_group_positions=plan.row_group_positions)
    # One shared stream for all candidates and both sexes. Clean, stressed and deterioration.
    point = np.empty((4, 2, 3))
    draws = np.empty((2000, 4, 2, 3))
    for i, c in enumerate(p.CANDIDATES):
        clean = (evidence[c]['clean']-meta.default.to_numpy())**2
        stress = ((evidence[c]['swap_mild'][1:]-meta.default.to_numpy())**2).mean(axis=0)
        vals = np.column_stack((clean, stress, stress-clean))
        for j, sex in enumerate((2, 1)):
            mask = meta.SEX.to_numpy() == sex
            point[i, j] = vals[mask].mean(axis=0)
            draws[:, i, j] = group_means(vals, plan.group_multiplicities,
                                        plan.row_group_positions, mask)
    ref = p.CANDIDATES.index(safe)
    checks, blocked, visited = {}, {}, []
    for c in ranked:
        i = p.CANDIDATES.index(c)
        checks[c] = fairness(point[i, :, 1]-point[ref, :, 1], draws[:, i, :, 1]-draws[:, ref, :, 1])
        blocked[c] = checks[c]['blocked'] if c != safe else False
    final = choice(ranked, safe, blocked)
    for c in ranked:
        visited.append(c)
        if c == final:
            break
    arms['selector2'] = final
    write(folder/'fairness_check.json', {'seed': 42+repeat, 'draws': 2000, 'ranking': ranked,
          'safe': safe, 'top_before_check': ranked[0], 'final': final,
          'changed_top_choice': final != ranked[0], 'visited_in_choice_rule': visited,
          'blocked_on_path': [c for c in visited if blocked[c]], 'candidate_checks': checks,
          'coordinates': ['female', 'male'], 'conditional_fixed_models_predictions': True})
    within = []
    for c in dict.fromkeys((final, safe)):
        i = p.CANDIDATES.index(c)
        for j, sex in enumerate(('female', 'male')):
            for k, metric in enumerate(('clean_brier', 'mild_stressed_brier', 'deterioration')):
                within.append({'candidate': c, 'sex': sex, 'metric': metric,
                    'estimate': point[i, j, k], **percentile(draws[:, i, j, k])})
    # Paired chosen-minus-SAFE intervals accompany the absolute within-repeat values.
    i = p.CANDIDATES.index(final)
    for j, sex in enumerate(('female', 'male')):
        for k, metric in enumerate(('clean_brier', 'mild_stressed_brier', 'deterioration')):
            within.append({'candidate': 'selector2_minus_safe', 'sex': sex, 'metric': metric,
                'estimate': point[i, j, k]-point[ref, j, k],
                **percentile(draws[:, i, j, k]-draws[:, ref, j, k])})
    csv(folder/'within_repeat_intervals.csv', pd.DataFrame(within))
    npz(folder/'selection_bootstrap_values.npz', point=point, draws=draws)
    write(folder/'choices_before_judge.json', arms)
    return arms


def judge(folder, Xfit, Xjudge, meta, models):
    refs = reference_bundle(Xfit)
    write(folder/'judge_references.json', refs)
    clean = {c: p.positive_probability(models[c], Xjudge) for c in p.CANDIDATES}
    masks = {'all': np.ones(len(meta), dtype=bool), 'female': meta.SEX.eq(2).to_numpy(), 'male': meta.SEX.eq(1).to_numpy()}
    scores, preds = [], {c: {} for c in p.CANDIDATES}
    # These arrays retain individual realization losses, separate from sampling uncertainty.
    realization = np.empty((4, 3, 9, 30, 2))
    registry = []
    for f, family in enumerate(FAMILIES):
        for rep in range(1, 31):
            frames, reg, stream = paired_transforms(Xjudge, Xfit, refs, family=family, repetition=rep, q=.10)
            registry.append({'family': family, 'realization': rep, 'registry': reg, 'seeds': stream})
            for mode, condition in enumerate(CONDITIONS):
                scenario = f'{family}__{CONDITIONS[condition]}'
                for i, c in enumerate(p.CANDIDATES):
                    probability, audit = predict_changed_rows(models[c], Xjudge, frames[condition], clean[c],
                                                              p.positive_probability, audit_full=(rep == 1))
                    preds[c].setdefault(scenario, []).append(probability)
                    for j, (sex, mask) in enumerate(masks.items()):
                        val = losses(table(meta.loc[mask], probability[mask], 'judge'))
                        realization[i, j, f*3+mode, rep-1] = [val['brier'], val['log_loss']]
    for i, c in enumerate(p.CANDIDATES):
        npz(folder/f'judge_{c}.npz', ID=meta.ID.to_numpy(), SEX=meta.SEX.to_numpy(), y=meta.default.to_numpy(),
            groups=meta.duplicate_group_id.to_numpy(), clean=clean[c], **{k: np.asarray(v) for k, v in preds[c].items()})
        for j, (sex, mask) in enumerate(masks.items()):
            base = losses(table(meta.loc[mask], clean[c][mask], 'judge'))
            scenario_means = realization[i, j].mean(axis=1)
            primary = float(scenario_means[:, 0].mean())
            scores.append({'candidate': c, 'sex': sex, 'records': int(mask.sum()),
                'clean_brier': base['brier'], 'clean_log_loss': base['log_loss'],
                'primary': primary, 'stressed_log_loss': float(scenario_means[:, 1].mean()),
                'worst_scenario_brier': float(scenario_means[:, 0].max()),
                'deterioration': primary-base['brier']})
    csv(folder/'judge_scores.csv', pd.DataFrame(scores))
    npz(folder/'judge_realization_losses.npz', losses=realization)
    write(folder/'judge_registry.json', registry)
    return scores


def repeat_run(run, repeat, X, meta, historical):
    started = time.perf_counter()
    folder = run/f'repeats/{repeat:02d}'
    folder.mkdir(parents=True, exist_ok=False)
    assignment, positions, audit = split(meta, repeat, historical)
    csv(folder/'assignment.csv', assignment)
    write(folder/'split_gate.json', audit)
    role_X = {r: X.iloc[v].reset_index(drop=True) for r, v in positions.items()}
    role_m = {r: meta.iloc[v].reset_index(drop=True) for r, v in positions.items()}
    models = fit_models(folder, role_X['fitting'], role_m['fitting'])
    fit_end = time.perf_counter()
    rows, evidence = selection(folder, role_X['fitting'], role_X['selection'], role_m['selection'], models)
    selection_end = time.perf_counter()
    arms = select_arms(folder, rows, evidence, role_m['selection'], repeat)
    check_end = time.perf_counter()
    judge(folder, role_X['fitting'], role_X['judge'], role_m['judge'], models)
    timing = {'repeat': repeat, 'seconds': time.perf_counter()-started,
              'fit_seconds': fit_end-started, 'SAFE_selection_seconds': selection_end-fit_end,
              'fairness_seconds': check_end-selection_end, 'judge_seconds': time.perf_counter()-check_end}
    write(folder/'timing.json', timing)
    write(folder/'COMPLETE.json', {'repeat': repeat, 'timing': timing,
          'prediction_hashes': {q.name: p.sha256(q) for q in sorted(folder.glob('*.npz'))},
          'status': 'repeat completed; whole-run verification pending'})
    print(f'Repeat {repeat} completed in {timing["seconds"]:.2f} seconds.', flush=True)
    return timing


def source_freeze(run):
    paths = [ROOT/'research/model_selection_v2/pilot.py', ROOT/'research/model_selection_v2/safe2026.py',
             ROOT/'research/model_selection_v2/safe2026_specification.md',
             *list((ROOT/'research/model_selection_v2/selection_exploration_v1').glob('*.py')),
             *list((ROOT/'research/model_selection_v2/judge_core').glob('*.py')),
             ROOT/'research/model_selection_v2/step0_v1/core.py',
             ROOT/'src/robustness_protocol.py', ROOT/'src/baseline.py', ROOT/'src/performance_evaluation.py',
             ROOT/'article_safe/data.py', *list(Path(__file__).parent.glob('*.py')),
             Path(__file__).parent/'README.md']
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT) and '.venv' not in path.parts:
                paths.append(path)
    hashes = p.hashes(paths)
    write(run/'source_hashes.json', hashes)
    for relative in hashes:
        dst = run/'source_snapshots'/relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, dst)
    write(run/'environment.json', {'python': sys.version, 'platform': platform.platform(),
          'packages': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'pandas', 'scikit-learn', 'joblib', 'threadpoolctl')},
          'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
          'git_HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()})


def preservation(run):
    before = json.loads((run/'preservation_before.json').read_text())['files']
    changed = [n for n, h in before.items() if not (ROOT/n).exists() or p.sha256(ROOT/n) != h]
    result = {'passed': not changed, 'checked_files': len(before), 'changed_or_missing': changed}
    write(run/'preservation_after.json', result)
    p.require(not changed, 'Pre-existing project file preservation failed')
    return result


def tests(run, label):
    command = [sys.executable, '-m', 'unittest', 'research.model_selection_v2.step1_v1.test_step1', '-v']
    result = subprocess.run(command, capture_output=True, text=True)
    with (run/f'tests_{label}.txt').open('x', encoding='utf-8') as f:
        f.write(result.stdout+result.stderr)
    p.require(result.returncode == 0, f'Focused tests failed ({label})')


def execute(run):
    p.require((run/'preservation_before.json').exists(), 'Missing pre-existing-file inventory')
    freeze = json.loads((run/'protocol_sha256.json').read_text())
    p.require(p.sha256(run/SPEC) == freeze['sha256'], 'Specification hash mismatch')
    p.require(not (run/'RUN_STARTED.json').exists(), 'Run destination was already used')
    write(run/'RUN_STARTED.json', {'date_utc': datetime.now(timezone.utc).isoformat(),
          'protocol': freeze['sha256'], 'status': 'incomplete', 'J': 50, 'dataset': 'Taiwan development only'})
    source_freeze(run)
    tests(run, 'before')
    X, meta, positions, historical = inputs(run)
    reproduce(run, X, meta, positions)
    times = [repeat_run(run, r, X, meta, historical) for r in range(2)]
    projected = 50*np.mean([v['seconds'] for v in times])
    timing = {'repeats': times, 'projected_50_repeat_seconds': float(projected),
              'limit_seconds': 12*3600, 'continue': bool(projected <= 12*3600)}
    write(run/'runtime_gate.json', timing)
    print('Runtime gate: '+json.dumps(timing), flush=True)
    if projected > 12*3600:
        write(run/'STOPPED.json', {'reason': 'projected runtime exceeds 12 hours', 'gate': timing})
        preservation(run)
        return
    for r in range(2, 50):
        repeat_run(run, r, X, meta, historical)
    from .summarize import summarize
    from .verify import verify
    summarize(run)
    tests(run, 'after')
    verify(run)
    preserved = preservation(run)
    write(run/'VERIFIED.json', {'status': 'verified', 'review': 'same-session recheck, not independent review',
          'protocol_sha256': freeze['sha256'], 'repeats': 50, 'preservation': preserved,
          'artifact_hashes': {q.relative_to(run).as_posix(): p.sha256(q) for q in run.rglob('*') if q.is_file()}})
    print('Step 1 completed and same-session rechecked.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    try:
        execute(run)
    except BaseException as error:
        write(run/'STOPPED.json', {'reason': str(error), 'type': type(error).__name__, 'traceback': traceback.format_exc()})
        if not (run/'preservation_after.json').exists():
            preservation(run)
        raise


if __name__ == '__main__':
    main()
