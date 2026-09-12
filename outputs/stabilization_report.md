# Baseline stabilization report — 2026-09-05

## Scope and duplicate findings

The existing data understanding → EDA → logistic regression → gradient boosting
sequence is preserved. No later-stage fairness, SAFE, SHAP, proxy analysis,
robustness, mitigation, stability/PSI, intersectional or governance analysis was
implemented. No threshold selection, CV, calibration or uncertainty estimation
was added. Model hyperparameters and preprocessing are unchanged.

Duplicate counts depend on what is matched:

| Match definition | Excess rows beyond first | Records involved | Duplicate groups | Old crossing groups | Old affected train / test records |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw or cleaned full row, excluding ID but including SEX and target | 35 | 70 | 35 | 7 | 7 / 7 |
| Cleaned predictors including SEX, excluding ID and target | 56 | 108 | 52 | 15 | 16 / 15 |
| Actual 22 model predictors, excluding ID, SEX and target | 93 | 177 | 84 | 31 | 33 / 32 |

Thus the reported 35 / 70 and seven affected test rows were correct for complete
non-ID records, but understated predictor overlap. Exact predictor matches can
have different SEX values and different outcomes. There are 37 predictor groups
with conflicting outcomes; equality does not establish erroneous duplication.
The cleaned and raw full-row duplicate counts are the same.

## Decision and split changes

**Keep all 30,000 observations.** Match on the actual cleaned model predictors,
without ID, SEX or target. Start from the existing target-stratified 80/20 split,
seed 42. Assign every member of an exact predictor group to its lowest-ID member's
original partition. The anchor is a deterministic record key, never a predictor.
The destination rule does not inspect model results or choose according to group
outcomes. It follows the original assignment, which was target-stratified.

This is a conservative repair of the existing split: 34 assignments change,
15 train → test and 19 test → train. Retain the original order of records that
stay in each partition; append incoming records in source order. All 31 crossing
groups are resolved. **Zero exact predictor groups now cross train/test.**
No records, conflicting labels, or duplicate observations are deleted.

| Partition | Old records | New records | Old defaults | New defaults | Old default rate | New default rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 24,000 | 24,004 | 5,309 | 5,316 | 22.1208% | 22.1463% |
| Test | 6,000 | 5,996 | 1,327 | 1,320 | 22.1167% | 22.0147% |

Final partition sizes and stratification are approximate. This avoids a wholesale
resplit while enforcing the required group separation. It is not a claim that
duplicate groups are erroneous or that all possible forms of data leakage have
been investigated. Dependence within retained duplicate groups remains relevant
to later uncertainty estimation.

## Baseline comparison

All scores below are proportions on the relevant test partition. PR-AUC retains
the existing table name but is computed as sklearn average precision.

| Metric | Logistic before | Logistic after | Boosting before | Boosting after |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 0.816833 | 0.818045 | 0.817000 | 0.817712 |
| Balanced Accuracy | 0.650119 | 0.651717 | 0.648877 | 0.650415 |
| Precision | 0.661932 | 0.661952 | 0.665224 | 0.661912 |
| Recall | 0.351168 | 0.354545 | 0.347400 | 0.351515 |
| F1 | 0.458887 | 0.461766 | 0.456436 | 0.459179 |
| ROC-AUC | 0.756744 | 0.756949 | 0.779407 | 0.778132 |
| PR-AUC | 0.526198 | 0.527009 | 0.553329 | 0.552136 |

Logistic now leads slightly on accuracy, balanced accuracy, precision, recall,
and F1 at the unchanged default 0.5 decision boundary. Boosting still leads on
ranking metrics: ROC-AUC (0.7781 vs 0.7569) and average precision (0.5521 vs 0.5270).
The new accuracy difference is only 0.0334 percentage points. Logistic default
recall is 35.45%; boosting recall is 35.15%, so both still miss about 65% of defaults.
No statistical significance or general superiority is claimed.

All metrics changed numerically, although logistic precision barely changed.
The largest absolute metric change is under 0.005. The rerun uses a different
training sample and a slightly different test sample, including seven fewer test
defaults. Removing exact overlap changes the learned models and the evaluation
composition. These changes need not lower every metric. The comparison alone
does not estimate the causal amount of optimism from duplicate leakage.

As a control, the shared model definitions were also trained on the **original**
split. Both reproduced every original predicted class, with maximum probability
difference 1.11e-16 against the preserved CSVs. Thus model definition, environment,
or accidental use of ID/SEX does not explain the new scores; the repaired partition
and retraining do. Original fitted models were never available to compare directly.

## Reproducibility and verification

- ID is recovered directly from the original workbook and preserved in cleaned
  data, train/test metadata, and both prediction CSVs. Both models explicitly
  exclude ID and SEX from their 22 predictors.
- Shared preparation and model definitions live in `src/baseline.py`; the current
  scripts and notebooks use them. Both models consume the same saved partition.
- Both complete fitted preprocessing + classifier pipelines are saved with joblib.
  Reloaded class predictions and probabilities are checked for exact equality.
  Manifests record versions, classifier parameters, feature names, and data/split/model hashes.
- Four focused regression tests passed: all-record retention and actual predictor
  disjointness, reproducible metadata, grouping across SEX/outcome differences,
  and saved pipeline/prediction/metric alignment.
- Both standalone scripts ran successfully, including boosting launched from
  `notebooks/`, verifying the working-directory fix.
- All four notebooks executed with the project interpreter; generated data/split
  and baseline metric/prediction CSVs were byte-identical to script outputs.
- `pip check` passed. A no-index installation dry run accepted the pinned current
  environment. A fresh environment download/install was not performed.
- Requirements pin the current analysis and notebook dependency closure, including
  extras and `xlrd`; later-stage research packages are not needed. Python 3.11.9
  on Windows is the validated platform.

## Deferred evaluation recommendations

These are documented proposals, not implemented methodology changes: use
duplicate-group-aware CV within training data; select any future threshold using
validation predictions and a stated recall/cost objective; assess calibration
before choosing a calibration method; and consider group-aware bootstrap
intervals or repeated group-aware splits. Keep final test data out of selection
and tuning. Because this holdout has already been inspected, a future confirmatory
study must explicitly address test-set reuse. These refinements should be agreed
before extending the existing evaluation stage.

## Files created or modified

Modified existing files:

- `script_base_model.py`, `script_gradient_boost.py` — file-relative paths, shared
  split/model definitions, ID output, pipeline persistence.
- `notebooks/01_data_understanding.ipynb` — useful duplicate checks, ID retention,
  shared preparation; rerun outputs.
- `notebooks/02_exploratory_analysis.ipynb` — robust root lookup, omit ID from EDA;
  rerun outputs.
- `notebooks/03_baseline_model.ipynb`, `notebooks/04_gradient_boosting.ipynb` —
  aligned baseline flow, traceability, persistence; rerun outputs.
- `data/processed/taiwan_credit_clean.csv` — ID restored; original cleaning preserved.
- `outputs/tables/baseline_logistic_metrics.csv`,
  `outputs/tables/baseline_gradient_boosting_metrics.csv` — rerun metrics.
- `outputs/tables/baseline_logistic_predictions.csv`,
  `outputs/tables/baseline_gradient_boosting_predictions.csv` — rerun predictions plus ID.
- `.vscode/settings.json` — default project interpreter path.
- `credit_risk_fairness_setup_steps.txt` — current setup guidance above historical notes.

Created files:

- `src/baseline.py`, `script_prepare_data.py`, `tests/test_stabilization.py`.
- `README.md`, `data/README.md`, `requirements.txt`.
- `data/processed/split_metadata.csv`, `train_metadata.csv`, `test_metadata.csv`,
  `preparation_manifest.json` (all under `data/processed/`).
- `models/baseline_logistic.joblib`, `models/baseline_gradient_boosting.joblib`,
  and `models/baseline_logistic_manifest.json`, `models/baseline_gradient_boosting_manifest.json`.
- `outputs/tables/duplicate_audit.json`, `duplicate_records.csv`, `split_changes.csv`,
  `baseline_stabilization_comparison.csv`, `stabilization_verification.json`
  (all under `outputs/tables/`).
- `outputs/previous_baseline/README.md`, `test_metadata.csv`, and byte-preserved
  copies of the four original baseline metric/prediction CSVs.
- `outputs/stabilization_report.md` (this report).

The raw workbook, research-plan notes, and pre-stabilization issue list were preserved.
