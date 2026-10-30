"""Preserve original diagnostic; correct only known zero-severity identity ties."""
from pathlib import Path
import hashlib
import itertools
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'runs/model_selection_v2/safe_saved_swap_loss_diagnostic_2026-10-30'
OUT = ROOT / 'runs/model_selection_v2/safe_saved_swap_comparison_correction_2026-10-30'
SPEC = ROOT / 'research/model_selection_v2/saved_swap_diagnostic_comparison_correction.md'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    receipt = json.loads((SOURCE / 'VERIFIED.json').read_text())
    assert receipt['artifact_hashes'] == {p.name: sha(p) for p in SOURCE.iterdir() if p.is_file() and p.name != 'VERIFIED.json'}
    frame = pd.read_csv(SOURCE / 'all_node_profiles.csv', float_precision='round_trip')
    order = tuple(dict.fromkeys(frame.candidate))
    corrected = {}
    for a, b in itertools.combinations(order, 2):
        left = frame[frame.candidate.eq(a)].reset_index(drop=True)
        right = frame[frame.candidate.eq(b)].reset_index(drop=True)
        np.testing.assert_array_equal(left.raw_severity, right.raw_severity)
        pair = {}
        for loss in ('brier', 'log_loss'):
            rank = np.sign(right.rgr_preservation.to_numpy()-left.rgr_preservation.to_numpy())
            quality = np.sign(left[loss].to_numpy()-right[loss].to_numpy())
            labels = ['tie_in_at_least_one_measure' if severity == 0 or r == 0 or s == 0 else 'agreement' if r == s else 'opposition'
                      for severity, r, s in zip(left.raw_severity, rank, quality)]
            assert labels[0] == 'tie_in_at_least_one_measure' and len(labels) == 23
            pair[loss] = {'by_node': labels, 'counts': {label: labels.count(label) for label in ('agreement', 'opposition', 'tie_in_at_least_one_measure')}}
        corrected[b+'_minus_'+a] = pair
    result = {'status': 'verified', 'date_utc': datetime.now(timezone.utc).isoformat(), 'source_receipt_sha256': sha(SOURCE / 'VERIFIED.json'),
              'analysis_source_sha256': sha(Path(__file__)), 'correction_specification_sha256': sha(SPEC),
              'zero_severity_raw_rgr': frame[frame.node.eq(0)].set_index('candidate').rgr_preservation.to_dict(),
              'scope': 'Only descriptive pair-order labels corrected for exact self-comparison ties; raw scores/losses/means unchanged',
              'corrected_pair_order_comparisons': corrected}
    OUT.mkdir(exist_ok=False)
    with (OUT / 'specification.md').open('xb') as f:
        f.write(SPEC.read_bytes())
    with (OUT / 'results.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    with (OUT / 'VERIFIED.json').open('x', encoding='utf-8') as f:
        json.dump({'status': 'verified', 'artifact_hashes': {name: sha(OUT / name) for name in ('specification.md', 'results.json')}}, f, indent=2)
    print(json.dumps({p: {m: v['counts'] for m, v in metrics.items()} for p, metrics in corrected.items()}))


if __name__ == '__main__':
    main()
