# Credit risk fairness research

Start with [current context and decisions](PROJECT_CONTEXT.md). This repository
preserves an exploratory credit-default study of logistic regression (LR) and
gradient boosting (GBM), a separate adaptation of the 2024 SAFE article, and
sources for a future model-selection comparison with integrated SAFE 2026.

Original performance, fairness, explainability and robustness are complete.
The 2024 adaptation is complete with its unpenalized LR unavailable; both
independent article runs are retained. The future comparison is accepted as a
direction; four fixed pilot candidates and their selection-set SAFE swapping
assessment were verified on 26 October. Our developing method and independent
judge remain to be specified. PSI/stability, monitoring,
mitigation and governance remain deferred.

## Read saved evidence

| Need | Entry point |
| --- | --- |
| Direction, decisions, open questions | [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) |
| Plain-language results page | [Interactive offline HTML](output/briefing/research_progress_2026-10-26.html): research story, data roles, results and SAFE curves |
| Agent safeguards | [AGENTS.md](AGENTS.md) |
| Original findings | [Performance](outputs/performance_evaluation_report.md), [fairness](outputs/fairness_evaluation_report.md), [explainability](outputs/explainability_evaluation_report.md), [robustness](outputs/robustness_evaluation_report.md) |
| 2024 article branch and LR limitation | [Package guide](article_safe/README.md), [saved results](article_safe/runs/article_v1_2026-10-10/report/results.md) |
| New fixed-candidate pilot | [Implementation and commands](research/model_selection_v2/README.md), [exploratory selection diagnostics and verification](runs/model_selection_v2/pilot_candidates_2026-10-26/verification_report.md) |
| Integrated SAFE 2026 application | [Guide](research/model_selection_v2/safe2026_README.md), [resolved specification/fidelity](research/model_selection_v2/safe2026_specification.md), [selection-set scores](runs/model_selection_v2/safe2026_swapping_2026-10-26/report.md) |
| Offline interactive results | [Explorer](explorer/credit-risk-explorer.html), [guide](explorer/README.md) |
| Literature, earlier plans, presentation sources | [Historical index](docs/archive/README.md) |

The explorer is a saved snapshot. Open its HTML in Edge or Chrome without a
server or environment. Its dated research-status text describes that snapshot.

## Structure

- `data/`: source workbook, cleaned records, fixed split, preparation manifest;
  [data definitions and citation](data/README.md).
- `src/`, root `script_*.py`, `notebooks/`, `tests/`: original implementation,
  entry points, notebooks and tests. Notebooks 01-04 include reconstruction;
  05-06 display completed evaluation evidence.
- `models/`, `outputs/`: frozen pipelines, predictions, reports, tables, figures,
  caches and manifests; `outputs/previous_baseline/` preserves the earlier split.
- `article_safe/`: separate 2024 adaptation, tests, completed runs and milestones.
- `research/model_selection_v2/`: separate four-candidate pilot implementation/tests;
  `runs/model_selection_v2/`: new versioned models, selection predictions,
  diagnostics, protocols and verification records. Historical outputs stay in place.
- `docs/`: literature sources, historical plans/reports, presentation authoring
  sources and compact archive navigation.
- `output/`: presentation/PDF/briefing deliverables and historical diagrams.
- `explorer/`: offline viewer, editable sources and QA records.
- `tmp/`: ignored new scratch output; explicitly referenced historical literature
  renders remain tracked. It is not a research-output directory.

## Environment on another Windows PC

Validated research environment: Python 3.11.9. With Git and Python 3.11 installed,
run from PowerShell:

```powershell
git clone https://github.com/stanescu2022-png/credit-risk-fairness.git
cd credit-risk-fairness
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -r article_safe/requirements.txt
```

The article requirements extend `requirements-explainability.txt`, which extends
the fairness and base locks. For the original study alone, install
`requirements-explainability.txt`. Keep existing locks unchanged; do not replace
them with a fresh `pip freeze`. Data, models and results are included: setup needs
no reconstruction. The root `credit_risk_fairness_setup_steps.txt` is obsolete
hash-bound evidence, not setup guidance. Direct interpreter calls avoid activation-policy issues.
Select `.venv/Scripts/python.exe` as the editor interpreter/notebook kernel;
launch notebooks with `.venv/Scripts/python.exe -m jupyterlab`.

## Checks and command side effects

Routine tests, when a change warrants them:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m unittest discover -s article_safe/tests -v
```

Tests use fixtures and temporary artifacts rather than regenerating completed
research stages. For documentation-only work, check links, diff and saved hashes.
The original 410 manifest references can be checked without evaluation:

```powershell
.venv/Scripts/python.exe -c "from article_safe.data import check_evaluation_artifacts; print(check_evaluation_artifacts())"
```

| Command family | Effect; use only deliberately |
| --- | --- |
| `script_prepare_data.py` | Recreates cleaned data and partition metadata |
| `script_base_model.py`, `script_gradient_boost.py` | Refits and replaces saved baseline pipelines, metrics and predictions |
| `script_*_evaluation.py` | Regenerates stage evaluation artifacts; performance includes temporary training-CV fits |
| Evaluation `--verify` where supported | Repeats evaluation and writes outputs/checks; not a read-only hash check |
| `article_safe.run`, `article_safe.report` | Create stages/reports under a fresh run ID; see [article guide](article_safe/README.md) |
| `article_safe.verification` | Checks completed runs, runs tests and writes a receipt; existing final receipt protected |
| `explorer/build_explorer.py` | Reads saved outputs and rewrites viewer HTML and source manifest |

## Historical protocol and Git

UCI Taiwan data: 30,000 records; repaired partition: 24,004 training / 5,996
holdout. Keep all 22 original predictors together for grouping; ID, SEX and
target are metadata. Decisions are strictly `p(default) > .50`, ties class 0.
AP is sklearn average precision. The inspected holdout supports exploratory
findings. See [context](PROJECT_CONTEXT.md) for conventions and
[manifests](outputs/tables/performance_evaluation/run_manifest.json) for exact
provenance. Load only trusted joblib models in the pinned environment.

Before switching PCs, review Git status and diff, then deliberately commit/push
intended files. On the receiving PC, use `git pull --ff-only` with a clean tree.
Keep environments and credentials ignored. `.gitattributes` preserves exact bytes
because manifests verify SHA-256 hashes.

Annotated tag `pre-cleanup-2026-10-26` at `1a1c136` preserves the tracked state
before cleanup. The [archive index](docs/archive/README.md) records removals,
relocations and intentionally frozen evidence.
