# Integrated SAFE 2026 assessment

**26 October 2026.** Separate assessment code for the four frozen pilot
candidates. The original [pilot guide](README.md), implementation/tests and
entire verified candidate run remain unchanged.

Read [the resolved specification and fidelity table](safe2026_specification.md)
before running. It separates published equations, user-approved Taiwan choices,
and unresolved reproduction details. `safe2026.py` contains numerical kernels;
`safe2026_run.py` handles the bounded assessment and traceability;
`safe2026_tests/` contains synthetic tests. `safe2026_sources/` is version-pinned
inspection evidence, not an installed dependency or replacement implementation.

The accepted v1 uses additive full-sample RGA area contributions, whole-original-
predictor greedy fixed-model masking at fitting-only means/modes, and original-
column percentile swaps. It does not fit any primary or auxiliary model. The
arithmetic swapping score is primary; geometric/quadratic scores are sensitivity
variants. Gaussian assessment is **deferred by user decision**. TOPSIS is
unsupported pending its exact specification. No extended method or final judge.

All assessments use the saved 4,801 selection records. The 19,203 fitting records
provide masking reference values and support diagnostics. The CSV reader skips
holdout rows and requests no target column from the full processed data; outcomes
come only from saved development assignments/training metadata. No holdout
predictors enter assessment functions. Metadata never enters model predictors.

## Commands and side effects

Use the existing project environment; do not install the inspected package or
change locks:

```powershell
.venv/Scripts/python.exe -m unittest discover -s research/model_selection_v2/safe2026_tests -v
.venv/Scripts/python.exe -m research.model_selection_v2.safe2026_run --run-id safe2026_swapping_2026-10-26
```

Training never occurs. A new run performs frozen-model predictions, the complete
greedy search (253 alternatives per candidate), numerical assessment, focused
tests, preservation checks and output writes in its **fresh** run folder only.
The completed name above cannot be reused: choose a new run ID for deliberate
re-execution. A preservation inventory includes existing tracked and untracked,
nonignored files except the editable startup documents. The initial implementation
run additionally used an earlier pre-task inventory to protect existing local work.

Read-only verification of saved completion hashes and independent reconstruction
from saved curves/prediction evidence, without fitting or regeneration:

```powershell
.venv/Scripts/python.exe -m research.model_selection_v2.safe2026_run --verify-run runs/model_selection_v2/safe2026_swapping_2026-10-26
```

Candidate run completion and its recorded input/implementation hashes are checked
before any trusted local joblib loading. Run completion requires a hash-valid
`VERIFIED.json`; an interrupted run/manifest alone is insufficient. Existing
directories and individual artifacts cannot be overwritten.

## Run contents

| Path | Purpose |
| --- | --- |
| `protocol.json`, `specification.md` | Frozen choices, fidelity table, source references, grids, weights and tie rules |
| `source_inspection.json` | Hash-pinned later-package provenance and synthetic counterexamples |
| `masking_references.json` | Fitting-only reference values and original-to-encoded feature maps |
| `curves/` | Raw scalar RGA, full area terms, partial contributions, curves, coordinates, areas and sampled means |
| `removal/` | Complete nested paths, all evaluated alternatives, score ties and prediction hashes |
| `predictions/` | Compact per-ID perturbation and chosen-removal-path probabilities; no holdout predictions |
| `perturbation_diagnostics.json` | Shared frame hashes, tail pairings/exposures, changed cells/rows and categorical support novelty |
| `integrated_scores.json`, `scores_and_rankings.csv`, `rankings.json` | Three cellwise mean scores, explicit ranks/ties/availability |
| `primary_selection.json` | Arithmetic swapping selector, selected ID and tie status |
| `component_areas.csv`, `component_curves.png`, `report.md` | Component summaries, compact plots and readable assessment |
| `tests.json`, `independent_verification.json` | Focused test output and independent numerical checks |
| `preservation.json`, `run_manifest.json`, `VERIFIED.json` | Preservation, versions/Git state and input/source/configuration/output identities |

Only chosen-path and perturbation predictions are retained. Large arrays for
unchosen alternatives are discarded after their scores and hashes are recorded;
they can be deterministically regenerated from the frozen model, fitting-only
references and recorded subset. No once-only importance ordering substitutes
for the greedy search.

Curves need not be monotone; observed increases are reported. Undefined reference
ratios are explicit, never changed to zero or used as evidence of poor performance.
Increase lists use zero-based interval indices: index k means node k to k+1.
Uniform node means entering aggregation are distinct from component quadrature
areas. The Cartesian grid is not a sampling distribution. Results concern SAFE
selection on this fixed candidate pool, not independent predictive superiority,
human explanation quality, legal compliance, real-world robustness or a comparison
against the future methodology. The independent judge remains to be specified.
