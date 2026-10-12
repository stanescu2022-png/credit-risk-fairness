# Credit risk fairness research

**Start here: [Project wiki](PROJECT_CONTEXT.md).** It centralizes current results,
architecture, decisions and research leads for new chats. Use it for current status;
the instructions below also retain earlier stage descriptions.

This project follows the existing research plan: compare logistic regression and
gradient boosting on credit default prediction, then evaluate trustworthiness in
later stages. The completed Performance Evaluation stage covers training-only
group-aware CV, OOF threshold candidates and calibration assessment, plus fixed-model
holdout metrics and group-bootstrap confidence intervals.
The completed Fairness Evaluation stage adds conventional SEX-group metrics,
actual SAFE RGA/parity, group-bootstrap uncertainty, descriptive threshold
sensitivity and training-only predictor associations. Explainability is complete:
model-specific SHAP and actual SAFE RGE characterize both frozen models.
Mitigation, robustness/RGR, PSI/stability, formal intersectionality and governance remain deferred.

## Project status

| Stage | Status |
| --- | --- |
| Data understanding / EDA | Complete |
| Baseline models | Complete |
| Stabilization | Complete |
| Performance | Complete |
| Fairness | Complete |
| Explainability | Complete |
| Robustness | Complete |
| Stability / monitoring | Not started |

Data: I-Cheng Yeh (2009), *Default of Credit Card Clients*, UCI Machine Learning
Repository, [dataset and source documentation](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients),
[DOI: 10.24432/C55S3H](https://doi.org/10.24432/C55S3H), CC BY 4.0.
See [data documentation](data/README.md) for fields and cleaning decisions.

## Structure

- `data/raw/`: original `.xls` source.
- `data/processed/`: cleaned records (including ID), train/test metadata, preparation manifest.
- `notebooks/01`–`04`: existing understanding, EDA, logistic, and boosting workflow.
- `src/baseline.py`: shared cleaning, split, preprocessing/model definitions, persistence.
- `script_prepare_data.py`, `script_base_model.py`, `script_gradient_boost.py`: reproducible entry points.
- `models/`: complete fitted pipelines and manifests with versions and input hashes.
- `outputs/tables/`: current predictions, metrics, duplicate audit, and split changes.
- `outputs/previous_baseline/`: preserved original metrics/predictions and reconstructed test IDs.
- `outputs/stabilization_report.md`: decisions and before/after results.
- `src/performance_evaluation.py`, `src/performance_report.py`: reusable evaluation and reporting.
- `script_performance_evaluation.py`, `notebooks/05_performance_evaluation.ipynb`: Performance Evaluation entry point and results notebook.
- `outputs/performance_evaluation_report.md`: findings, limitations and pending decisions.
- `outputs/tables/performance_evaluation/`, `outputs/figures/performance_evaluation/`: stage tables, manifest and PNG/SVG figures.
- `tests/`: focused split, traceability, and persistence checks.
- `src/fairness_evaluation.py`, `src/safe_adapter.py`, `src/fairness_report.py`: integrated conventional and actual SAFE fairness logic.
- `script_fairness_evaluation.py`, `notebooks/06_fairness_evaluation.ipynb`: fairness entry point and verified results notebook.
- `outputs/fairness_evaluation_report.md`, `outputs/tables/fairness_evaluation/`, `outputs/figures/fairness_evaluation/`: fairness findings, tables, provenance and figures.
- `src/explainability_evaluation.py`, `src/explainability_report.py`, `script_explainability_evaluation.py`: frozen-model SHAP and authentic SAFE RGE.
- `outputs/explainability_evaluation_report.md`, `outputs/tables/explainability_evaluation/`, `outputs/figures/explainability_evaluation/`: explanation results, provenance and figures.
- `doc ideeas.txt`: existing research plan; `potential_issues.txt`: pre-stabilization review notes.

## Download and continue on another PC

On another Windows PC, install Git and Python 3.11 (validated version: 3.11.9),
then open PowerShell in the folder where you want the project:

```powershell
git clone https://github.com/stanescu2022-png/credit-risk-fairness.git
cd credit-risk-fairness
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-explainability.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m jupyterlab
```

For a private repository, sign in to the GitHub account that has access when
Git prompts you. The explainability requirements include the fairness and base requirements.
Calling the environment's Python directly avoids PowerShell activation-policy issues.
In VS Code, select `.venv\Scripts\python.exe` as the Python interpreter and
notebook kernel.

The repository includes the source dataset, processed data, frozen models and
saved evaluation outputs. Continue with those artifacts; the preparation and
training commands below are only needed if you deliberately reconstruct the
baseline. The `.venv` folder is local to each PC and is recreated, not uploaded.
Git preserves file bytes because the evaluation manifests check artifact hashes.

To move later changes between PCs, first save and close edited notebooks, then
run on the PC where you made the changes:

```powershell
git status
git add .
git commit -m "Describe your changes"
git push
```

On the other PC, run `git pull --ff-only` before starting work, with a clean
working tree. Commit and push your work before switching PCs.

## Setup and run

**Continuing from the stabilized project:** the repaired split is now a locked
holdout protocol. Use `python script_performance_evaluation.py` to regenerate only
Performance Evaluation outputs; it does not resplit data or overwrite saved models.
Open notebook 05 to inspect the saved results. The script runs from any working
directory and uses a non-interactive plotting backend. All required libraries are
already pinned in `requirements.txt`; no additional dependencies were added.

Validated with Python 3.11.9 on Windows. In PowerShell, from the project root:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python script_prepare_data.py
python script_base_model.py
python script_gradient_boost.py
python script_performance_evaluation.py
python -m unittest discover -s tests -v
```

The preparation/baseline commands above document reconstruction on a fresh copy;
they are not required for continuing from the current saved artifacts. Do not
change the partition, predictors, preprocessing or model definitions as part of
Performance Evaluation.

Place the UCI workbook at `data/raw/default of credit card clients.xls` before
preparation. `xlrd` is pinned for this legacy Excel format. Requirements pin the
installed dependency closure of current analysis and notebook tools, without
requiring packages reserved for future research stages.

Preparation replaces generated cleaned data and partition metadata; model runs
replace their current metrics, predictions, and fitted pipelines. The historical
outputs in `outputs/previous_baseline/` remain fixed. Scripts resolve paths from
their file locations, so they can also run from `notebooks/` or another directory.
For headless execution, set `$env:MPLBACKEND = 'Agg'` before running model scripts;
plots remain available interactively in the notebooks.

Alternatively run `jupyter lab` and select the project `.venv` kernel. Notebooks
01 through 04 document baseline reconstruction; notebook 05 reads the completed
performance artifacts. The current split and saved pipelines are locked for this
stage. Input hash mismatches cause evaluation to stop instead of silently preparing
a new split or fitting replacement baseline models.

## Baseline protocol

Keep ID and SEX out of the 22 predictors. Retain ID, SEX, and outcome in partition
metadata; prediction CSVs contain ID, actual, predicted, probability_default, SEX.
Clean education codes 0/5/6 to 4 and marriage code 0 to 3, as before.
Scale numerical features and one-hot encode categorical features, fitting the
entire preprocessing pipeline only on training data. Logistic regression uses
`max_iter=2000, random_state=42`; gradient boosting uses its existing defaults and
`random_state=42`. The library's default decision boundary remains 0.5.
Current-facing tables use **Average Precision (AP)** for sklearn
`average_precision_score`. Earlier archived outputs and the historical stabilization
report retain the former PR-AUC label; AP is not trapezoidal PR-AUC.

Start with the original target-stratified 80/20 split (seed 42). Group exact
cleaned predictor matches, excluding ID, SEX, and outcome; all members follow the
lowest-ID member's original partition. Keep every record and conflicting label.
This conservative repair changes only 34 assignments and eliminates exact
predictor overlap. The final split is approximately 80/20 and approximately
target-stratified, rather than enforcing exact proportions. Assignment uses no
model scores; it does not optimize test performance. This is a one-time repair
for the existing baseline, not a new validation study.

Load a saved pipeline with the same pinned environment:

```python
import joblib
import pandas as pd
from src.baseline import load_split

df = pd.read_csv("data/processed/taiwan_credit_clean.csv")
_, X_test, _, _, _, metadata_test = load_split(df)
model = joblib.load("models/baseline_logistic.joblib")
probabilities = model.predict_proba(X_test)[:, 1]
```

Manifests record data/split/model SHA-256 values and fitted classifier parameters.
Load only trusted joblib artifacts. Reproduction across other library versions
or platforms is not guaranteed to be bit-for-bit identical.

## Performance Evaluation protocol

The existing 24,004/5,996 partition is fixed. The holdout may report fixed-model
evaluation, but must not choose hyperparameters, features, preprocessing,
calibration methods or thresholds. Its earlier baseline-stage inspection remains
disclosed; locking it now does not make it historically unseen.

- Training-only `StratifiedGroupKFold`: 5 folds, shuffle=True, seed 42. Exact 22-predictor
  groups never cross folds. Both models use the same folds; preprocessing is fitted
  separately within each fold. Fold metrics include mean and sample SD (ddof=1).
- OOF predictions retain ID and SEX as metadata. Thresholds from 0.00 to 1.00 in
  steps of 0.01, plus empirical recall-target candidates (40/50/60/70%), use OOF data
  only. No new threshold is selected; holdout classification stays at 0.50.
- Calibration is assessed with reliability curves, 10 equal-width bins, count-weighted
  absolute-gap ECE, Brier score and log loss. No calibrator is fitted.
- Holdout 95% CIs use 2,000 whole-predictor-group bootstrap resamples, seed 42,
  and percentile intervals. Models stay fixed. No significance test is performed.

`outputs/tables/performance_evaluation/run_manifest.json` records locked input hashes,
settings, versions and output hashes. Subsequent evaluation runs verify that the
baseline inputs still match. `verification.json` in the same directory records
the completed two-run CSV reproducibility check, notebook execution and 10 passing
tests. Read probability/threshold CSVs with
`pd.read_csv(path, float_precision="round_trip")` when reproducing exact candidate
boundary decisions; displayed threshold values are rounded for readability.

These Performance Evaluation decisions remain open: agree a business objective and
threshold-selection/validation protocol, or retain 0.50; decide whether any
training-only calibration experiment is needed. Both models remain in the project.

## Fairness Evaluation

Continue with the frozen models and repaired split:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-fairness.txt
.venv\Scripts\python.exe script_fairness_evaluation.py
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The separate requirements extension includes safeaipackage==0.8.3 and its imported
CatBoost/XGBoost dependencies. The original requirements lock and Performance
Evaluation artifacts stay intact. SAFE was already installed; no baseline-library
upgrade or model refit was required. The inspected upstream code and examples are
documented in the [fairness report](outputs/fairness_evaluation_report.md).

Primary analysis uses SEX=1 male / SEX=2 female at 0.50. Signed gaps are female
minus male. A positive prediction is a default-risk flag, not an approval. The SAFE
adapter makes SEX available for grouping and passes only the original 22 features
to each fitted pipeline. The package's unsigned RGA parity return and signed group
RGA difference are reported separately, with no privileged/unprivileged mapping.

The fairness-stage authorization adds an exploratory **holdout sensitivity** grid
0.10–0.70, step 0.05. It does not authorize threshold selection: 0.50 remains the
reference, and model development/calibration/threshold optimization remain excluded
from the holdout. Conventional and SAFE uncertainties use 2,000 whole-predictor-group
bootstrap replicates, seed 42. Proxy-related descriptions use training data only;
association is not a causal-discrimination claim. Notebook 06 displays saved outputs
and verifies hashes without fitting a model.

Completion verification: all 14 tests passed, the fairness CSVs reproduced exactly
on rerun, and the executed notebook displays all three figures. The fairness
`verification.json` records these checks; the run manifest confirms that fitted
models, split metadata and Performance Evaluation artifacts were unchanged.

The [fairness report](outputs/fairness_evaluation_report.md) compares frameworks and
explains limits. No global fair/unfair verdict, mitigation, intersectional experiment,
explainability or robustness work was added in that stage. Remaining decisions concern the relevant
decision harms, fairness concepts, practical gap tolerances, and whether a separately
designed proxy/intersectional follow-up is warranted. Explainability is now complete below.

## Explainability Evaluation

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-explainability.txt
.venv\Scripts\python.exe script_explainability_evaluation.py --verify
```

Uses every one of the same 5,996 holdout records, the frozen pipelines and the
unchanged 0.50 threshold. LinearExplainer and interventional TreeExplainer explain
default log-odds using one seeded, 100-record training-only background. Signed
one-hot contributions are summed into the original 22 predictors; transformed
values and mappings are retained. This aggregation is not a separate original-feature
coalition calculation for boosting. SHAP additivity and unchanged predictions are checked.

Actual `safeaipackage.check_explainability.compute_rge_values` calculates individual
and grouped RGE using full-training means/modes. Existing categorical definitions
are communicated through pandas dtypes, with exact prediction equivalence checked.
No substitute RGE formula, retraining, calibration or threshold selection is used.

The [explainability report](outputs/explainability_evaluation_report.md) covers global
rankings, dependence plots, fixed-rule local examples, SEX-group patterns and
SHAP/RGE comparisons. PAY_0 and LIMIT_BAL lead both models. SHAP/RGE rank correlations
are 0.918 (logistic) and 0.927 (boosting); logistic has the more concentrated top-three
SHAP share (54.2% versus 40.9%). Group explanation differences are descriptive, not
evidence of discrimination. The report documents background, correlated-feature and
encoded-feature limitations.

`--verify` repeats the complete evaluation, checks byte-identical CSVs, and runs all
project tests. The stage manifest and `verification.json` record provenance and checks;
protected data, model and prior performance/fairness hashes must remain unchanged.
The separate requirements extension preserves both earlier dependency locks.

Next research stage: **Robustness evaluation**, including a separately specified
SAFE RGR protocol. Robustness and stability/PSI have not been implemented.

## Robustness Evaluation — 8 October 2026

Robustness is complete under the separately prespecified synthetic input-corruption
protocol. Earlier stage descriptions above are retained as historical context;
references there to robustness as future work have now been superseded.

```powershell
.venv\Scripts\python.exe script_robustness_evaluation.py --verify
```

Uses the same 5,996 holdout records, frozen LR/GBM pipelines, 22 predictors and
strict probability >0.50 reference decisions. Authentic SAFE 0.8.3 evaluates
individual and eight grouped tail-swap scenarios at p=.01/.025/.05/.10, plus the
prespecified row-order diagnostic. Supplementary semantic scenarios use training-only
IQRs, category frequencies, support maps and bounds at q=.10/.20, with 30 coupled
repetitions. The two primary joint scenarios receive 2,000 paired complete
predictor-group bootstrap resamples, seed 42. Repetition variability is reported
separately from sampling intervals. No new dependencies or earlier lock changes.

The [robustness report](outputs/robustness_evaluation_report.md) explains the
findings, protocol rationale, package behavior, descriptive SEX differences and
limitations. Code is in `src/robustness_{evaluation,protocol,safe,uncertainty,report,cache}.py`;
focused tests are in `tests/test_robustness_evaluation.py`. Saved protocol, tables,
prediction cache, source inspection, provenance and verification are under
`outputs/tables/robustness_evaluation/`; PNG/SVG figures have their own stage folder.
`--verify` executes the entire finalized evaluation twice, compares all numerical
CSVs and prediction-cache bytes, then runs every project test and pip check.

These experiments characterize synthetic sensitivity; they do not establish
real-world error rates, causal changes, adversarial security, future-population
reliability, a universally best model or framework novelty. Stability/PSI,
monitoring, mitigation and governance remain deferred.
