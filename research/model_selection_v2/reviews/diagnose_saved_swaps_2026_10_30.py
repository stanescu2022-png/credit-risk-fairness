"""Descriptive losses on saved SAFE swaps; never fit, predict or select."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'runs/model_selection_v2/safe2026_swapping_2026-10-26'
EXPLORATION = ROOT / 'runs/model_selection_v2/selection_exploration_v1_2026-10-30'
DEST = ROOT / 'runs/model_selection_v2/safe_saved_swap_loss_diagnostic_2026-10-30'
SPEC = ROOT / 'research/model_selection_v2/saved_swap_diagnostic_specification.md'
ORDER = ('logistic', 'decision_tree', 'random_forest', 'gradient_boosting')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    DEST.mkdir(exist_ok=False)
    with (DEST / 'specification.md').open('xb') as f:
        f.write(SPEC.read_bytes())
    source_files = {p.relative_to(SOURCE).as_posix(): sha(p) for p in SOURCE.rglob('*') if p.is_file() and p.name != 'VERIFIED.json'}
    assert read(SOURCE / 'VERIFIED.json')['artifact_hashes'] == source_files
    write(DEST / 'protocol.json', {'specified_at_utc': datetime.now(timezone.utc).isoformat(), 'specification_sha256': sha(SPEC),
          'analysis_source_sha256': sha(Path(__file__)), 'saved_source_receipt_sha256': sha(SOURCE / 'VERIFIED.json'),
          'source_artifact_hashes': source_files, 'status': 'post-exploration descriptive diagnostic', 'new_selector': False,
          'holdout_access': False, 'model_predictions_or_fits': 0})
    metadata = pd.read_csv(SOURCE / 'selection_identity.csv')
    clean = read(EXPLORATION / 'q0.10/scenario_matrix.json')
    rows = []
    for name in ORDER:
        curve = read(SOURCE / f'curves/{name}.json')['robustness']
        with np.load(SOURCE / f'predictions/{name}.npz', allow_pickle=False) as stored:
            np.testing.assert_array_equal(stored['ID'], metadata.ID)
            probabilities = stored['rgr_probabilities']
        assert probabilities.shape == (23, 4801)
        assert len(curve['raw_x']) == len(curve['x']) == len(curve['values']) == 23
        assert np.isfinite(probabilities).all() and ((0 <= probabilities) & (probabilities <= 1)).all()
        y = metadata.default.to_numpy()
        clipped = np.clip(probabilities, 2.**-52, 1-2.**-52)
        brier = ((probabilities-y)**2).mean(axis=1)
        logarithmic = (-y*np.log(clipped)-(1-y)*np.log1p(-clipped)).mean(axis=1)
        np.testing.assert_allclose(brier, [brier_score_loss(y, p) for p in probabilities], atol=1e-14, rtol=0)
        np.testing.assert_allclose(logarithmic, [log_loss(y, p, labels=[0, 1]) for p in probabilities], atol=1e-14, rtol=0)
        np.testing.assert_allclose([brier[0], logarithmic[0]], [clean[name]['clean']['brier'], clean[name]['clean']['log_loss']], atol=1e-14, rtol=0)
        for k in range(23):
            rows.append({'candidate': name, 'node': k, 'raw_severity': curve['raw_x'][k], 'normalized_position': curve['x'][k],
                         'rgr_preservation': curve['values'][k], 'brier': float(brier[k]), 'log_loss': float(logarithmic[k]),
                         'brier_delta': float(brier[k]-brier[0]), 'log_loss_delta': float(logarithmic[k]-logarithmic[0])})
    frame = pd.DataFrame(rows)
    summaries = frame.groupby('candidate', sort=False)[['rgr_preservation', 'brier', 'log_loss', 'brier_delta', 'log_loss_delta']].mean().to_dict('index')
    pairs = {}
    for earlier, later in itertools.combinations(ORDER, 2):
        a = frame[frame.candidate.eq(earlier)].reset_index(drop=True)
        b = frame[frame.candidate.eq(later)].reset_index(drop=True)
        pair = {}
        for loss in ('brier', 'log_loss'):
            rank_sign = np.sign(b.rgr_preservation.to_numpy()-a.rgr_preservation.to_numpy())
            loss_sign = np.sign(a[loss].to_numpy()-b[loss].to_numpy())
            labels = ['tie_in_at_least_one_measure' if r == 0 or s == 0 else 'agreement' if r == s else 'opposition' for r, s in zip(rank_sign, loss_sign)]
            pair[loss] = {'by_node': labels, 'counts': {label: labels.count(label) for label in ('agreement', 'opposition', 'tie_in_at_least_one_measure')}}
        pairs[later+'_minus_'+earlier] = pair
    with (DEST / 'all_node_profiles.csv').open('x', encoding='utf-8', newline='') as f:
        frame.to_csv(f, index=False)
    write(DEST / 'results.json', {'uniform_grid_means_descriptive_only': summaries, 'all_fixed_pair_order_comparisons': pairs,
                                'sampling_or_monte_carlo_intervals': None, 'new_selection_decision': None})
    for name, digest in source_files.items():
        assert sha(SOURCE / name) == digest
    write(DEST / 'verification.json', {'saved_predictions_and_IDs_checked': True, 'all_92_node_losses_checked_with_sklearn': True,
                                      'all_zero_severity_losses_match_verified_exploration': True, 'source_artifacts_preserved': True})
    write(DEST / 'VERIFIED.json', {'status': 'verified', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
          'artifact_hashes': {p.name: sha(p) for p in DEST.iterdir() if p.is_file()}})
    print(json.dumps({'grid_means': summaries, 'pair_counts': {pair: {metric: value['counts'] for metric, value in data.items()} for pair, data in pairs.items()}}))


if __name__ == '__main__':
    main()
