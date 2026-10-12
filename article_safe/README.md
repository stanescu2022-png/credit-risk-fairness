# Article SAFE implementation

This separate package implements the declared interpretation of Giudici, Centurelli and Turchetta's original 2024 SAFE/KAIRI article, using LR and GBM on the existing project data. The [approved specification](../docs/plans/article_safe_specification_2026-10-10.md) distinguishes article procedures, explicit interpretations and supporting diagnostics.

Current status: **checkpoint 4 of 4 complete, with the article LR limitation explicitly retained**. Read the [results report](runs/article_v1_2026-10-10/report/results.md). GBM selection, final fits and assessments have run. The article's unpenalized LR is explicitly unavailable under the current categorical design. No category pooling, substitute regularization or broader framework experiment was performed. The existing frozen regularized LR is an estimator reference only.

The article-style GBM selection retained PAY_0, LIMIT_BAL and PAY_AMT1. Adding PAY_2 failed the declared first-stop rule. On the saved holdout, full GBM AUC was 0.7781 and selected GBM AUC was 0.7512. This is a parsimony/accuracy trade-off, not evidence of overall framework superiority. The [LR discussion note](../docs/plans/article_safe_lr_limitation_2026-10-10.md) records the statistical obstacle and its implications for a fair later comparison.

## Package

- `protocol.py`, `data.py`: fixed specification/provenance, saved-data readers and whole original-predictor-group partitions.
- `inference.py`, `selection.py`: paired DeLong, likelihood ratios, multiplicity, Gini/KS and stopping decisions.
- `models.py`, `paths.py`: fresh GLM/GBM fits, reference coding and training-only selection paths; invalid fits are reported.
- `explainability.py`, `fairness.py`, `uncertainty.py`, `evaluation.py`: final assessments and conditional paired group-bootstrap intervals.
- `run.py`: preparation, selection and evaluation command interface.
- `report.py`: report, all four planned model profiles, fidelity/status tables and PNG/SVG figures, using verified saved outputs.
- `verification.py`: compare a fresh complete rerun with the saved run, independently check results, run both test suites and check preservation.
- `tests/`: numerical, structural and estimator fixtures, including the LR backward path on identifiable synthetic data.

## Read the saved results

The primary run is `runs/article_v1_2026-10-10/`. Its `selection/`, `evaluation/` and `report/` directories have source/configuration/output manifests. The fresh numerical reproduction is `runs/article_v1_reproduction_2026-10-10/`. The [final verification receipt](milestones/checkpoint_04/verification.json) and [completion note](milestones/checkpoint_04/README.md) record the successfully completed verification.

The [checkpoint-1 report](milestones/checkpoint_01/README.md), preserved source snapshot, and [checkpoint-3 report](milestones/checkpoint_03/README.md) remain historical milestone records. Preparation manifests describe the preparation stage only; a later successful stage does not rewrite that snapshot.

## Execute a new article run

Use the project environment from the repository root. All run IDs must be fresh. Existing runs, completed scientific stages and reports are never overwritten.

```powershell
.venv/Scripts/python.exe -m article_safe.run --stage prepare --run-id a_new_run_id
.venv/Scripts/python.exe -m article_safe.run --stage select --run-id a_new_run_id
.venv/Scripts/python.exe -m article_safe.run --stage evaluate --run-id a_new_run_id
.venv/Scripts/python.exe -m article_safe.report --run-id a_new_run_id
```

Reporting is a separate command so adding report code does not change the frozen entry-point code used by the historical scientific run. It verifies preparation, selection and evaluation identities before reading their outputs.

```powershell
.venv/Scripts/python.exe -m unittest discover -s article_safe/tests -v
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

The final verification command checks two completed independent runs and creates a new final receipt. Its completed milestone destination is protected against overwriting:

```powershell
.venv/Scripts/python.exe -m article_safe.verification --reference article_v1_2026-10-10 --replicate article_v1_reproduction_2026-10-10
```

Keep the existing 24,004/5,996 outer partition, 22 original grouping predictors, metadata exclusions and strict probability >0.50 decisions. Inner fitting/validation has 19,203/4,801 records. The holdout was historically inspected; results are exploratory. No prior baseline reconstruction or old stage `--verify` command is needed.

The GitHub reference is [GolnooshBabaei/safeaipackage](https://github.com/GolnooshBabaei/safeaipackage). Installed 0.8.3 functions implement the later rank-graduation metrics already used by the original project. This package retains their audited provenance as a reference; it does not substitute RGA/RGE/RGR for the original article's DeLong/likelihood selection and Gini/KS procedures. No overall SAFE score is invented.