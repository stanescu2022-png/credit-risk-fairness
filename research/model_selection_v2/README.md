# Fixed candidate pilot v1

Accepted and implemented **26 October 2026**. This separate pilot prepares four
fresh, frozen candidates for a later model-selection comparison. It implements
no SAFE score, feature ablation, extended methodology or final judge.

## Separation and data roles

Original research remains in `src/`, `models/`, `outputs/` and `data/`; the
separate 2024 adaptation remains in `article_safe/`. This code reads those paths
and writes only a fresh folder under `runs/model_selection_v2/`.

The exact saved assignment source is
`article_safe/runs/article_v1_2026-10-10/development_assignments.csv`, SHA-256
`bc8993d37ce07979a34bd6757aa061a3947515b56669178d0ed8be78d5418588`.
No new partition is generated. The 19,203 fitting records alone train all four
new pipelines. The 4,801 selection records provide exploratory diagnostics.
Together these reproduce the saved 24,004 training records. The historically
inspected 5,996 holdout receives identity/alignment/group checks only; it is not
untouched confirmation evidence and receives no predictions or performance table.

Fixed cleaning is reused from `src.baseline.clean_data`: rename the target,
EDUCATION 0/5/6 → 4, MARRIAGE 0 → 3, and reject invalid IDs/missing values. These
operations are deterministic and learn no statistics or categories. Cleaning
the original workbook in memory must reproduce the saved processed data exactly.
No historical preparation, reconstruction or evaluation runner is called.

Saved LR manifest input order is preserved. ID, SEX, target and original
predictor-group IDs remain metadata. Every split/group check uses all original
22 predictors; the saved grouping labels are independently checked by equality.

## Fixed, untuned candidates

All use random_state=42 and independent fitting-only StandardScaler for the
existing numeric fields and OneHotEncoder(handle_unknown="ignore") for the
existing categorical fields. No imputation, feature selection, new pooling,
reweighting, calibration, tuning or combined fitting/selection refit.

| Candidate | Declared settings beyond defaults |
| --- | --- |
| Logistic Regression | L2, C=1, lbfgs, max_iter=2000, class_weight=None |
| Decision Tree | gini, depth=5, min_split=2, min_leaf=20, class_weight=None |
| Random Forest | 300 trees, gini, unlimited depth, min_split=2, min_leaf=5, sqrt features, bootstrap=True, class_weight=None, n_jobs=1 |
| Gradient Boosting | log_loss, 100 trees, rate=.1, depth=3, min_split=2, min_leaf=1, subsample=1 |

Installed sklearn 1.9 expresses LR L2 through `l1_ratio=0.0` and finite C;
the deprecated `penalty` argument is omitted. The complete resolved estimator
parameters, preprocessing parameters and package versions are saved per run.
Fitting time is elapsed wall time for the complete preprocessing-plus-estimator
`pipeline.fit` call. Warnings, including LR ConvergenceWarning, and n_iter_ are
recorded explicitly, without changing the declared configuration.

The target is 0=nondefault, 1=default. Each candidate must support both classes;
probabilities use the column located by `classes_ == 1`. The decision rule is
strict probability >.50, ties class 0. Undefined metric denominators are saved
as JSON null / CSV empty, with exact rules in protocol.json. AP is sklearn
average precision. Diagnostic metrics do not select an overall winner.

`diagnostics/<candidate>.json` preserves a mapping for all 22 original predictors
to transformed column indices/names and one-hot levels. It also reports unseen
selection levels and record counts separately for every original categorical
predictor, and checks both original and transformed inputs for nonfinite values.
This mapping supports later work; feature removal is not implemented here.

## Execution and outputs

From the repository root, using the existing environment:

```powershell
.venv/Scripts/python.exe -m unittest discover -s research/model_selection_v2/tests -v
.venv/Scripts/python.exe -m research.model_selection_v2.pilot --run-id pilot_candidates_2026-10-26
```

Training requires a **fresh run ID**. If that folder exists, choose a new name;
the implementation refuses to overwrite any destination or artifact. Focused
tests run again inside each run and their result is saved. Synthetic fixtures
exercise failure risks without repeatedly training the credit candidates.

| Run-relative path | Contents |
| --- | --- |
| `models/` | Four complete fitted pipelines |
| `predictions/` | Selection-only predictions keyed by original ID, candidate, outcome, SEX and original group |
| `diagnostics/selection_metrics.{csv,json}` | All requested diagnostic metrics, confusion counts and full fitting times |
| `diagnostics/<candidate>.json` | Feature mapping, unseen levels, finite-input checks, classes, warnings, n_iter_ |
| `diagnostics/data_audit.json` | Cleaning audit, partition identities/counts/alignment and group isolation |
| `diagnostics/preservation.json` | Before hashes and after equality for protected tracked files; 410 historical manifest references checked before/after |
| `diagnostics/focused_tests.json` | Focused test command, result and captured output |
| `development_assignments.csv` | Exact reused assignments, without repartitioning |
| `protocol.json`, `run_manifest.json` | Configuration, input/source/artifact hashes, timestamps, versions and starting Git state |
| `verification_report.md` | Human-readable verification and exploratory diagnostic table |

`RUN_STARTED.json` permanently records inception as incomplete. A run is complete
**only** when `VERIFIED.json` exists and its artifact hashes match. A manifest or
individual model file alone is insufficient. Catchable failures/interruptions
add `FAILURE.json`; a process kill still leaves no valid completion marker.
Read-only completion check (no training or historical regeneration):

```powershell
.venv/Scripts/python.exe -m research.model_selection_v2.pilot --verify-run runs/model_selection_v2/pilot_candidates_2026-10-26
```

Only trusted local joblib pipelines should be loaded. Pipeline reload, row/ID
alignment, class-1 probabilities, strict decisions and metrics from saved CSVs
are checked before completion. Existing historical scientific files, locks,
snapshots and receipts are preserved and hash checked.

## Handoff limits

Subsequent conclusions concern selection quality **conditional on this fixed
candidate pool**, not universal algorithm-family superiority. The four primary
candidates remain frozen. Later SAFE feature-removal experiments may separately
fit auxiliary models, but must never modify or replace these candidates. The
exact RGE protocol belongs to a later task.

Next: discuss and specify the SAFE 2026 implementation. The developing method,
independent judge (including provisional Brier/LogLoss options), resampling,
refit policy and confirmation dataset remain unresolved.
