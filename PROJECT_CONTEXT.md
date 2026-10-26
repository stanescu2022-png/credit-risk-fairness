# Credit-risk current context

Updated **26 October 2026**. Read this at startup, then only task-relevant evidence.
Latest user instructions override older plans. [README](README.md) covers setup
and command effects; [AGENTS](AGENTS.md) covers safeguards;
[historical index](docs/archive/README.md) routes older material.

## Current direction and next decision

**Accepted direction, 26 October:** compare published integrated **SAFE 2026**
with our developing method as model-selection procedures. Give both the same
candidate pool, initially **LR, Decision Tree, Random Forest and Gradient
Boosting**, with comparable development resources. Assess their chosen models
using an independently specified evaluation. Improvement is a hypothesis.

Evidence: the user's unchanged 21-slide [16 October decision deck](output/presentations/credit-risk-research-presentation-16102026_v2.pptx),
slides 7-8, 13 and 16-18. Copied on 26 October from
`C:/Users/Andrei/Downloads/credit-risk-research-presentation-16102026_v2.pptx`.
SHA-256: `28138bbb933e471448438f1a35c250e97b28cf8562a788fb00e586cabcf9bd3d`.
The title date is 16 October; the supplied file's filesystem timestamp is
22 October. The deck is evidence of direction, not an executable specification.

**Next unresolved decision:** specify the independent judge: performance-first,
minimum requirements followed by performance, or a multidimensional/Pareto
comparison; justify practical margins or gates before formal evaluation.
**Brier is proposed as primary**, not accepted as final. LogLoss may be secondary
or co-primary; an independent explainability criterion remains open. Also specify
our exact extension, the published SAFE variant/aggregation, candidate settings,
tuning budget, resampling and a confirmation dataset. Taiwan has already informed
development and cannot be called an untouched confirmation set. Romanian credit
data were requested but are not available here.

This direction supersedes the older blanket hold on pursuing the comparison.
It does not authorize implementing every older SAFE/KAIRI indicator or test.
No new study, refit, category pooling or LR repair was executed during cleanup.

## Completed work: three distinct branches

| Branch | Executed state |
| --- | --- |
| Original LR/GBM study | Data/EDA, frozen pipelines, performance, fairness, explainability and robustness complete by 8 October |
| Separate 2024 article adaptation | Four work packages complete and independently reproduced on 10 October; GBM works; unpenalized article LR unavailable |
| Future integrated SAFE 2026 selection comparison | Accepted direction; no method, score, new candidates or experiment implemented |

The original study uses native SAFE metrics plus conventional metrics, SHAP,
semantic stresses and uncertainty. That collection is not a new overall
selection rule. The 2024 article uses training-only feature selection and separate
profiles; adding an overall scalar would be our adaptation. Integrated SAFE 2026
is the intended published score comparator. The [older architecture diagram](output/diagrams/project_architecture_current.png)
describes the completed branches, not a finalized design for the new study.
PSI/stability, monitoring, mitigation, formal intersectionality and governance
remain deferred.

## Results that remain defensible

Same 5,996-record inspected holdout. AP = sklearn average precision; lower Brier
is better; decisions use strict probability >.50.

| Candidate | Inputs | AUC | AP | Brier | Default recall | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing regularized LR | 22 | .7569 | .5270 | .1382 | .3545 | .0511 |
| Full GBM, same predictions in both branches | 22 | .7781 | .5521 | .1350 | .3515 | .0507 |
| Article-selected GBM | 3 | .7512 | .5079 | .1402 | .3447 | .0475 |

- Full GBM ranks better and has lower probability losses; both full models detect
  about 35% of defaults at .50. No overall model recommendation follows.
- Article GBM retained **PAY_0, LIMIT_BAL, PAY_AMT1**. Adding PAY_2 triggered the
  first stop; later improvement did not reverse it. Selected-minus-full AUC:
  **-.02695**, conditional paired 95% interval **[-.03485, -.01895]**. Andrei
  considers this potentially marginal for 19 fewer inputs; significance alone
  does not establish material weakness or framework superiority.
- Fairness has no overall winner. Female-minus-male recall gaps: LR .0232,
  GBM .0335. Primary gap intervals include zero; that does not prove fairness.
- PAY_0 and LIMIT_BAL dominate SHAP. Top-three shares: LR 54.2%, full GBM 40.9%.
  Concentration is not a fairness or human-interpretability verdict.
- Semantic joint corruption q=.10: decision flips LR .79%, GBM 1.30%; native SAFE
  joint tail swapping p=.05: 11.44%, 11.12%. Exposure/severity differ, so these
  scenarios are not equivalent. Results describe synthetic sensitivity.

Evidence: [performance](outputs/performance_evaluation_report.md),
[fairness](outputs/fairness_evaluation_report.md),
[explainability](outputs/explainability_evaluation_report.md),
[robustness](outputs/robustness_evaluation_report.md),
[article results](article_safe/runs/article_v1_2026-10-10/report/results.md),
[offline explorer](explorer/credit-risk-explorer.html).

## Historical protocol to preserve

- UCI Taiwan: 30,000 rows; **24,004 training / 5,996 holdout**, with 1,320 holdout
  defaults. All records retained. [Data definitions](data/README.md).
- Keep exact original **22-predictor groups** intact for folds/bootstrap, even
  with reduced candidates. Matching inputs do not establish the same customer.
- Predictors: LIMIT_BAL, AGE, EDUCATION, MARRIAGE, PAY_0/PAY_2-6, BILL_AMT1-6,
  PAY_AMT1-6. ID/SEX/target are metadata. SEX 1 male / 2 female. Use saved order.
- EDUCATION 0/5/6 -> 4; MARRIAGE 0 -> 3. Numeric StandardScaler; categorical
  one-hot encoding. Original LR: regularized C=1, lbfgs, max_iter=2000; GBM:
  100 trees, learning rate .1, depth 3; seed 42. Exact settings in manifests.
- Strict `p(default) > .50`, ties class 0. Flags are not observed credit denials.
  No replacement threshold or calibrator selected.
- Training CV: five-fold StratifiedGroupKFold, shuffle/seed 42. Article inner
  partition: 19,203 fitting / 4,801 validation; freeze selection, refit on all
  training, assess holdout. Inner validation is our leakage-control adaptation.
- Uncertainty: 2,000 shared whole-group bootstrap draws, seed 42; conditional on
  fixed models/subsets, without refitting/selection uncertainty. Repetition
  spread in corruption experiments is separate Monte Carlo variability.
- Signed gaps: female minus male; stresses: perturbed minus baseline; paired
  family comparisons: GBM minus LR. Label selected-minus-full separately.
  SHAP explains log-odds, summed from encoded columns to original predictors.
- The holdout was historically inspected: findings remain exploratory. No
  holdout tuning, causal explanations, discrimination, real-world harm/error
  rates, temporal reliability, universal winner or established novelty claims.

## Article LR limitation

The unpenalized binomial GLM design has 78 columns, rank 74 on fitting data and
76 on full training. PAY_4=8, PAY_5=8 and PAY_6=8 identify the same single training
row; some rare indicators also have one outcome class. The strict full-rank guard
stopped fitting before optimization. This is not a prediction failure or proof
that all aliased GLMs are impossible. Dropping redundancy alone does not ensure
finite estimates. [Diagnosis](article_safe/runs/article_v1_2026-10-10/diagnostics/lr_design_identifiability.json),
[discussion](docs/plans/article_safe_lr_limitation_2026-10-10.md).

Andrei was open to common category pooling/model revision; no cutoff or new fit
was agreed. A six-plus-month category was an example. Any later change needs
consistent upstream treatment and a new version preserving v1. PAY_0 is September
2005, PAY_6 April; -2/0 remain distinct because the cited source does not define
them. No deterministic monthly update rule is established.

## Compact decision record

| Date | Status and rationale | Evidence |
| --- | --- | --- |
| 8-10 Oct 2026 | Executed: original assessments and separate article adaptation complete; retain both with limitations | Reports above; [final article receipt](article_safe/milestones/checkpoint_04/verification.json) |
| 10 Oct 2026 | Proposal: more models, evaluators and aggregation; averaging provisional. Different rankings alone cannot establish better selection | [Prior context](docs/context/project_context_history_2026-10-12.md); pre-cleanup Git tag preserves the 12 October wiki |
| 26 Oct 2026 | Accepted: proceed toward SAFE 2026 model-choice comparison on common candidates, judged independently; exact design open | User cleanup instruction; unchanged 16 October deck linked above |
| 26 Oct 2026 | Executed: simplified navigation, archived useful authoring sources, removed disposable files; no research changes | [Cleanup record](docs/archive/README.md); tag `pre-cleanup-2026-10-26` at `1a1c136` |

## Detail only when needed

| Need | Read |
| --- | --- |
| Exact article interpretation | [Package guide](article_safe/README.md), [specification](docs/plans/article_safe_specification_2026-10-10.md), [fidelity table](article_safe/runs/article_v1_2026-10-10/report/article_fidelity.csv) |
| Article completion | [Final receipt](article_safe/milestones/checkpoint_04/verification.json): historical 49 article + 30 original tests; 48 reproduced artifacts, 45 byte-identical. Old checkpoint pauses are superseded |
| Original provenance | `models/*_manifest.json`, `outputs/tables/*/run_manifest.json`, `data/processed/preparation_manifest.json` |
| Literature | [Full-text review and six-paper index](docs/reports/safe_full_text_review_2026-10-09.md), local PDFs/extracts in `docs/context/sources/` |
| Old proposals and provenance recovery | [Historical index](docs/archive/README.md), [context history](docs/context/project_context_history_2026-10-12.md), `docs/context/chat_history.json` |

SAFE package 0.8.3 supplies native RGA/parity, replacement RGE and tail-swap RGR in
the original study; it does not implement the article's DeLong/LR-selection/Gini/KS
procedures. Later literature already covers probability-sensitive measures and
aggregation; adding Brier, a score or uncertainty alone is not demonstrated novelty.
The Wasserstein printed-formula critique does not establish what author code used.

Use `.venv`; preserve all scientific artifacts and locks. The 26 October cleanup
was explicitly authorized to commit/push its changes and preservation tag; this
is not continuing authorization for later publication or external contact.
Maintain this compact context by replacement, not by appending a conversation diary.
