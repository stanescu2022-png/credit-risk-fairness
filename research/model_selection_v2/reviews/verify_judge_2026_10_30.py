"""Root reconstruction from saved predictions, independent of judge calculations."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'runs/model_selection_v2/judge_v1_2026-10-30'
OUT = ROOT / 'runs/model_selection_v2/judge_v1_review_2026-10-30'
C = ('logistic', 'decision_tree', 'random_forest', 'gradient_boosting')
F = ('repayment', 'billing', 'payment')
S = ('clean',) + tuple(f'{f}__{m}' for f in F for m in ('mask_only', 'corruption_only', 'joint'))
L = ('brier', 'log_loss')
A = ('clean_brier', 'clean_logloss', 'absolute_tpr_gap', 'absolute_fpr_gap', 'worst_joint_brier', 'worst_joint_logloss')
SIGNED = ('signed_tpr_gap', 'signed_fpr_gap')
MAX_ERROR = 0.


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(x, y):
    global MAX_ERROR
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    np.testing.assert_allclose(x, y, rtol=0, atol=3e-14, equal_nan=True)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.any():
        MAX_ERROR = max(MAX_ERROR, float(np.max(np.abs(x-y)[finite])))


def interval(values, saved):
    values = np.asarray(values)
    valid = np.isfinite(values)
    assert saved['valid_draws'] == int(valid.sum()) and saved['undefined_draws'] == int((~valid).sum())
    if valid.any():
        close(np.percentile(values[valid], [2.5, 50, 97.5], method='linear'), [saved['lower'], saved['median'], saved['upper']])
    else:
        assert saved['lower'] is saved['median'] is saved['upper'] is None


def relations(vectors):
    complete = np.isfinite(vectors).all(axis=-1)
    less = (vectors[:, None, :] < vectors[None, :, :]).any(axis=-1)
    more = (vectors[:, None, :] > vectors[None, :, :]).any(axis=-1)
    status = np.full((4, 4), 'equal', dtype='<U20')
    status[less & ~more] = 'left_dominates'
    status[more & ~less] = 'right_dominates'
    status[less & more] = 'incomparable'
    status[~(complete[:, None] & complete[None, :])] = 'unresolved'
    frontier = complete & ~(status == 'left_dominates').any(axis=0)
    return status, frontier, bool(complete.all())


def main():
    hashes = {p.relative_to(RUN).as_posix(): sha(p) for p in RUN.rglob('*') if p.is_file() and p != RUN / 'VERIFIED.json'}
    assert read(RUN / 'VERIFIED.json')['artifact_hashes'] == hashes
    frozen = read(RUN / 'freeze/FREEZE.json')
    choices = frozen['reference_method_to_candidate']
    assert choices == {k: v['selected_id'] for k, v in frozen['reference_choices'].items()}
    for name, digest in frozen['source_hashes'].items():
        assert sha(ROOT / name) == digest
    meta = pd.read_csv(RUN / 'evaluation_identity.csv')
    assert len(meta) == 5996 and meta.ID.is_unique
    y, sex = meta.default.to_numpy(), meta.SEX.to_numpy()
    groups = tuple(dict.fromkeys(meta.duplicate_group_id))
    mapping = {g: i for i, g in enumerate(groups)}
    pos = np.array([mapping[g] for g in meta.duplicate_group_id])
    with np.load(RUN / 'bootstrap_multiplicities.npz', allow_pickle=False) as file:
        counts = file['group_multiplicities']
        np.testing.assert_array_equal(file['ID'], meta.ID)
        np.testing.assert_array_equal(file['group_order'], groups)
        np.testing.assert_array_equal(file['row_group_positions'], pos)
    draws = np.random.default_rng(42).integers(0, len(groups), size=(2000, len(groups)))
    np.testing.assert_array_equal(counts, np.array([np.bincount(d, minlength=len(groups)) for d in draws]))
    sizes = np.bincount(pos, minlength=len(groups))
    weights = np.vstack((np.ones((1, len(groups))), counts))
    denominators = {}
    masks = {}
    for group, outcome in itertools.product((1, 2), (1, 0)):
        mask = masks[group, outcome] = (sex == group) & (y == outcome)
        denominators[group, outcome] = weights @ np.bincount(pos, weights=mask, minlength=len(groups))
    report = {}
    first_clean = {}
    for q in ('q0.10', 'q0.20'):
        folder = RUN / q
        point, summary = read(folder / 'point.json'), read(folder / 'bootstrap_summary.json')
        assert point['comparisons']['frozen_reference_method_to_candidate'] == choices
        record = np.empty((len(meta), 4, 10, 2))
        signed = np.empty((2001, 4, 2))
        for c, candidate in enumerate(C):
            with np.load(folder / f'predictions/{candidate}.npz', allow_pickle=False) as file:
                np.testing.assert_array_equal(file['ID'], meta.ID)
                clean = file['clean'][0]
                if candidate in first_clean:
                    np.testing.assert_array_equal(clean, first_clean[candidate])
                first_clean[candidate] = clean
                for s, scenario in enumerate(S):
                    p = file[scenario]
                    assert p.shape == (1 if s == 0 else 30, len(meta))
                    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                    clip = np.clip(p, 2.**-52, 1-2.**-52)
                    losses = ((p-y)**2, -y*np.log(clip)-(1-y)*np.log1p(-clip))
                    for k, loss in enumerate(L):
                        record[:, c, s, k] = losses[k].mean(axis=0)
                        per_rep = losses[k].mean(axis=1)
                        diag = point['scenario_diagnostics'][candidate][scenario]['loss_repetitions'][loss]
                        close(per_rep, diag['values'])
                        if s:
                            close(per_rep.std(ddof=1), diag['mc_sd'])
                        close(per_rep.mean(), point['scenario_matrix'][candidate][scenario][loss])
            for k, outcome in enumerate((1, 0)):
                rates = {}
                for group in (1, 2):
                    numerator = weights @ np.bincount(pos, weights=masks[group, outcome] & (clean > .5), minlength=len(groups))
                    rates[group] = np.divide(numerator, denominators[group, outcome], out=np.full(2001, np.nan), where=denominators[group, outcome] > 0)
                signed[:, c, k] = rates[2]-rates[1]
        grouped = np.zeros((len(groups), 4, 10, 2))
        np.add.at(grouped, pos, record)
        losses = ((weights @ grouped.reshape(len(groups), -1))/(weights @ sizes)[:, None]).reshape(2001, 4, 10, 2)
        vectors = np.concatenate((losses[:, :, 0, :], np.abs(signed), losses[:, :, (3, 6, 9), :].max(axis=2)), axis=2)
        with np.load(folder / 'bootstrap_draw_evidence.npz', allow_pickle=False) as file:
            ev = {k: file[k] for k in file.files}
        for col, (candidate, scenario, loss) in enumerate(zip(ev['loss_key_candidate'], ev['loss_key_scenario'], ev['loss_key_name'])):
            close(ev['loss_matrix'][:, col], losses[1:, C.index(candidate), S.index(scenario), L.index(loss)])
        for c, candidate in enumerate(C):
            for a, axis in enumerate(A):
                close(vectors[0, c, a], point['coordinates'][candidate][axis])
                close(vectors[1:, c, a], ev[f'{candidate}___{axis}'])
                np.testing.assert_array_equal(np.isfinite(vectors[1:, c, a]), ev[f'{candidate}___{axis}___valid'])
            for k, axis in enumerate(SIGNED):
                close(signed[0, c, k], point['signed_gaps'][candidate][axis])
                close(signed[1:, c, k], ev[f'{candidate}___{axis}'])
            for k, axis in enumerate((*A, *SIGNED)):
                value = vectors[1:, c, k] if k < 6 else signed[1:, c, k-6]
                interval(value, summary['candidate_coordinate_and_signed_gap_intervals'][candidate][axis])
            for s, scenario in enumerate(S):
                for k, loss in enumerate(L):
                    interval(losses[1:, c, s, k], summary['candidate_scenario_loss_intervals'][candidate][scenario][loss])
        point_status, point_frontier, _ = relations(vectors[0])
        assert [C[i] for i in np.flatnonzero(point_frontier)] == point['comparisons']['frontier']['frontier_ids']
        for a, b in itertools.product(range(4), repeat=2):
            assert point_status[a, b] == point['comparisons']['pairwise_matrix'][C[a]][C[b]]['status']
        stats = [relations(v) for v in vectors[1:]]
        states = np.stack([v[0] for v in stats])
        fronts = np.stack([v[1] for v in stats])
        complete = np.array([v[2] for v in stats])
        np.testing.assert_array_equal(fronts, ev['frontier_membership'])
        np.testing.assert_array_equal(complete, ev['complete_frontier'])
        contrast = np.stack([losses[1:, :, 3+3*f, :]-losses[1:, :, 1+3*f, :]-losses[1:, :, 2+3*f, :]+losses[1:, :, 0, :] for f in range(3)], axis=2)
        for c, candidate in enumerate(C):
            for f, family in enumerate(F):
                for k, loss in enumerate(L):
                    interval(contrast[:, c, f, k], summary['candidate_interaction_intervals'][candidate][family][loss])
        pairs = [(a, b, summary['fixed_candidate_pairs'][C[b]+'_minus_'+C[a]]) for a, b in itertools.combinations(range(4), 2)]
        principal_a, principal_b = C.index(choices['safe_arithmetic']), C.index(choices['screened_safe_outcome_v0'])
        pairs.append((principal_a, principal_b, summary['principal_contrast']))
        for a, b, pair in pairs:
            assert dict(Counter(states[:, b, a])) == pair['relation_stability']['counts']
            for k, axis in enumerate((*A, *SIGNED)):
                value = vectors[1:, b, k]-vectors[1:, a, k] if k < 6 else signed[1:, b, k-6]-signed[1:, a, k-6]
                interval(value, pair['coordinate_and_signed_gap_intervals'][axis])
            for s, scenario in enumerate(S):
                for k, loss in enumerate(L):
                    interval(losses[1:, b, s, k]-losses[1:, a, s, k], pair['loss_intervals'][scenario][loss])
            for f, family in enumerate(F):
                for k, loss in enumerate(L):
                    interval(contrast[:, b, f, k]-contrast[:, a, f, k], pair['interaction_difference_intervals'][family][loss])
        report[q] = {'point_coordinates': point['coordinates'], 'principal_relation': str(point_status[principal_b, principal_a]),
                     'principal_bootstrap_relation_counts': dict(Counter(states[:, principal_b, principal_a])),
                     'point_frontier': [C[i] for i in np.flatnonzero(point_frontier)]}
    audits = read(RUN / 'prediction_reuse_audits.json')
    assert len(audits) == 72 and all(a['maximum_absolute_error'] <= 1e-15 for a in audits)
    result = {'status': 'verified', 'date_utc': datetime.now(timezone.utc).isoformat(), 'source_receipt_sha256': sha(RUN / 'VERIFIED.json'),
              'review_source_sha256': sha(Path(__file__)), 'maximum_absolute_reconstruction_difference': MAX_ERROR,
              'reconstruction_tolerance': 3e-14, 'tolerance_scope': 'reproduction only; comparisons use unrounded values',
              'all_2000_loss_and_signed_absolute_gap_draws_rebuilt_from_predictions': True,
              'all_candidate_pair_and_principal_loss_coordinate_interaction_intervals_checked': True,
              'all_full_vector_relations_and_frontiers_rebuilt': True, 'fitting_or_model_prediction': False, 'results': report}
    OUT.mkdir(exist_ok=False)
    with (OUT / 'review.json').open('x', encoding='utf-8') as file:
        json.dump(result, file, indent=2, allow_nan=False)
    with (OUT / 'VERIFIED.json').open('x', encoding='utf-8') as file:
        json.dump({'status': 'verified', 'artifact_hashes': {'review.json': sha(OUT / 'review.json')}}, file, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
