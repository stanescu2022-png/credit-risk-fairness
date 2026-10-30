# Independent Pareto judge: proposal

Date: **30 October 2026**. Status: **proposal, not an implemented protocol**.

## Decision boundary

Accepted in the current user request: move toward implementing the independent
judge, using a **Pareto comparison for now**. The primary evaluation remains
undecided. The request asks for a proposal taking account of the supplied deck
and the implemented partition. It does not settle the metric vector, stress
settings, practical margins or future refitting policy.

This document proposes those details for review. No judge code, new credit-data
predictions, model fits, resampling runs or evaluation results were produced.
The deck is research evidence, not execution instructions. Its earlier
suggestion of primary Brier does not override the current open decision.

## Research question and separation

Given the same four frozen candidates, do the models chosen by integrated
SAFE 2026 and our developing method dominate one another, or involve trade-offs,
on independently specified outcome and disparity measures?

The unit compared is the **chosen fitted model**, conditional on this candidate
pool and split. The judge does not rerun either selector on evaluation records.
SAFE's recorded choice is logistic. Our method and its choice remain unspecified.
If both methods choose the same saved pipeline, record **same choice**; identical
evaluation predictions provide no evidence of an advantage for either method.

Independence means a separately frozen evaluation protocol and separated data
roles. It does not require every metric to differ from a selector's inputs.
Conversely, different metric names do not create independence if judge results
are used to revise the method. Any revision informed by evaluation results
becomes further exploratory development and needs later independent assessment.

## Preserve the implemented split

| Role | Records | Proposed use |
| --- | ---: | --- |
| Fitting | 19,203 | Existing model/preprocessing fits; fitting-only stress reference scales, support and frequencies |
| Selection | 4,801 | Both procedures choose from the same LR, DT, RF and GBM pipelines |
| Historical holdout | 5,996 | Paired exploratory judge assessment after rules and both choices are locked |

The first two roles reproduce the existing 24,004 training records. Retain the
saved assignments and whole-original-22-predictor groups. No new partition,
threshold tuning, calibration or refitting on all 24,004 records is proposed
for judge v1. A later refit would define a different object of evaluation and
requires its own versioned protocol.

The 5,996 holdout has informed previous research. Even if the new candidates
have not been assessed there, it is **historically inspected exploratory
evidence**, not an untouched confirmation set. Do not recycle selection-set
metrics as independent judge results or relabel a new split of Taiwan as fresh
confirmation. A future confirmation dataset and repeated full-procedure design
remain open.

## Proposed measurement panel

Retain every measure in its natural units. Set `primary_evaluation: null`.
There is no overall score, weighting, performance-first ordering, eligibility
gate or default winner. Choosing which axes enter Pareto comparison is itself
a substantive choice: the following six-axis vector is proposed, not accepted.

| Dimension | Proposed Pareto coordinates (all lower is better) | Supporting diagnostics |
| --- | --- | --- |
| Clean probability quality | Brier; LogLoss | ROC-AUC, sklearn AP, reliability diagram and bin counts |
| Sex disparities at the existing decision rule | Absolute female-minus-male TPR gap; absolute female-minus-male FPR gap | Signed gaps, both groups' TPR/FPR, counts and overall recall/FPR |
| Probability quality under the declared stress | Mean stressed Brier; mean stressed LogLoss | Signed stressed-minus-clean losses, probability movement and decision flips |

This is six coordinates across three dimensions. Both probability losses remain
peers; neither is primary or a tie-breaker. Including both makes dominance more
restrictive and can leave all four candidates nondominated. That is an admissible
result, not grounds for removing an inconvenient coordinate after results.

Binary Brier is `mean((p_default - y)^2)`, on the 0–1 scale. LogLoss uses natural
logs, float64 probabilities and the pinned environment's explicitly documented
clipping convention, with counts of exact-zero/one predictions retained.
Brier and LogLoss assess overall probability quality; neither alone establishes
calibration quality. Proposed reliability plots use the same ten fixed
equal-width probability bins for all candidates and report empty/small bins.
No recalibration is fitted by the judge.

Decisions remain strict `p(default) > .50`, with equality class 0. SEX remains
metadata (1 male, 2 female). Signed gaps are female minus male, reported in
percentage points; Pareto coordinates use their absolute values. TPR and FPR
gaps stay separate, avoiding an arbitrary fairness average. Smaller gaps do not
establish legal fairness or discrimination, and group rates must remain visible
to expose equal but poor performance. Undefined denominators produce missing
values with reasons, never zero gaps.

Capture@q and Precision@q follow slide 10 as supporting operational diagnostics:
propose q = 5%, 10%, 20%, explicitly illustrative capacities. Use
`k=floor(q*n)` and fractional allocation across an exact boundary score tie to
fill k slots without using outcomes to break ties. Capture is captured defaults
divided by all defaults; precision is captured defaults divided by k. This is
expected capture under random tie resolution, not a realized review list. If
capacity-based selection becomes the decision setting, specify its fairness
assessment separately; the .50-rule gaps describe a different decision policy.

Explainability stays outside dominance and hard gates. Provide model structure,
input counts and any explicitly scoped descriptive explanation evidence, but
do not convert SHAP concentration, sparsity or SAFE RGE into an independent
human-explanation-quality verdict. All current candidates use the same 22
original predictors. An independent explainability criterion remains open.

## Stress proposal

Reuse the **rule definitions**, not the numerical references or old predictions,
from the existing semantic corruption experiment. Derive all references anew
from the 19,203 fitting records only. The old experiment used the larger training
set, so copying its scales would violate this pilot's data roles.

Proposed reference scenario: joint corruption, q=.10, 30 shared realizations.
Proposed sensitivity scenario: q=.20, kept separate from the reference frontier.
These are synthetic design settings, not measured error frequencies. Reference
scenario means the stress input used for this proposal's frontier, not a settled
primary evaluation. Gaussian remains deferred.

The existing rules select floor(q*n) records, share a selection mask across
features, apply signed monetary offsets based on fitting IQRs, preserve payment
nonnegativity and valid age/credit-limit bounds, and move ordinary repayment
delays by supported category steps while preserving special states. Education
and marriage replacements use fitting frequencies conditional on a different
category. Preserve original-variable transformations and valid one-hot groups.
The exact bounds, rounding, no-op rules, seed derivation and support checks must
be copied into a new versioned protocol before execution, with their fitting-only
reference values. No historical runner is invoked.

All four candidates receive identical corrupted records per realization. Keep
outcomes fixed: this assesses sensitivity to input measurement corruption,
not the causal effect of changing age, debt or behavior. Save actual exposure,
clipping and no-op counts. For each realization, calculate loss first, then
average losses; never evaluate an average prediction as if it were the stress
estimand. Keep the signed deterioration, including improvements, without clipping.

Use **absolute stressed loss** in the frontier alongside clean loss. Deterioration
alone can favor an already poor model. Show deterioration separately to distinguish
baseline quality from stress sensitivity. Outcome-based losses complement the
SAFE ranking-preservation objective without making a real-world robustness claim.

## Pareto rules and reporting

For the six loss-oriented coordinates `z`, A dominates B exactly when
`z_A[j] <= z_B[j]` for every j and `<` for at least one j. Compare unrounded
estimates. Numerical audit tolerances are not practical-equivalence margins.

Report the selected pair as one of:

- observed dominance by either choice;
- observed trade-off (each is better on at least one axis);
- equal measured coordinates, or the stronger identity statement **same choice**;
- unresolved because a required measure is unavailable.

Assess all four frozen candidates in the same run, after selector choices are
locked. Show the full pool's nondominated set and mark both choices. A model can
be nondominated against the other selected model yet dominated by a third
candidate. Neither result chooses a replacement model on the evaluation set.
If a candidate lacks a required coordinate, its complete Pareto status is
unavailable; do not silently exclude it and declare a complete-pool frontier.

Show a metric table, selected-pair signed differences, a dominance matrix and
two-axis plots such as clean Brier versus absolute TPR gap and clean versus
stressed Brier. Label plots as projections: visual dominance on two axes need
not imply dominance on all six. Report the Brier and LogLoss families separately
as predeclared diagnostic views, without promoting whichever view favors a method.

Do not add a weighted distance to an ideal point, hypervolume ranking or
minimum-gap gate in v1. Those introduce further choices of scales, reference
points or practical priorities. A nondominated set is a set of trade-offs, not
an overall ranking or proof that every member is equally good.

## Uncertainty and permitted conclusions

Propose 2,000 paired bootstrap draws, seed 42, sampling whole saved original
predictor groups with replacement. Retain every row in a sampled group and its
multiplicity. Share draws across all candidates, outcomes and stress realizations.
Do not stratify in a way that splits groups crossing SEX or outcome categories.

Recompute subgroup rates, absolute gaps, losses and selected-pair differences
inside each draw. For stress, recompute each saved realization's loss within the
draw, then average. Use **our choice minus SAFE choice** for the newly defined
selector comparison and label it explicitly; preserve GBM-minus-LR for historical
family comparisons. Save 95% percentile intervals, valid replicate counts and
reasons for undefined metrics. Incomplete draws cannot certify full dominance.

Report how often each model is nondominated and each pair dominates across valid
resamples as **bootstrap stability summaries**. They are not probabilities that
a method is truly superior, formal dominance tests or simultaneous confidence
statements. Coordinate-wise 95% intervals do not jointly certify dominance.
Near-zero absolute disparities are non-smooth; percentile intervals are exploratory
and require caution at that boundary. Preserve signed-gap intervals as well.

Separate sampling uncertainty from the SD across the 30 corruption realizations.
Both remain conditional on frozen candidates, selector choices and saved stresses.
They do not cover training variability, selector instability or uncertainty from
developing the method. The repeated full selection/training experiment envisioned
on slide 16 belongs to a later protocol, not a reinterpretation of this bootstrap.

Without justified practical margins, report observed dominance/trade-offs and
their uncertainty. Do not claim practical equivalence from an interval including
zero or claim meaningful improvement from a tiny difference. Margins can be
specified later on substantive grounds, with explicit acknowledgment of prior
exploratory exposure and a separate confirmation plan.

## Implementation sequence proposed

1. Build a prediction-table judge with explicit metric directions, missingness,
   Pareto comparison and reporting. Develop on synthetic fixtures. Keep primary
   evaluation null, and serialize the proposed axis list rather than hiding it
   in defaults.
2. Implement the separately versioned stress and paired group-bootstrap modules.
   Test ties, strict .50 behavior, known loss values, same-choice identity,
   dominance/trade-offs, undefined group rates, group multiplicities, class-1
   probability lookup, alignment and paired stress inputs. Reconstruct report
   values independently from saved prediction tables.
3. Freeze the judge protocol and our developing selection rule. Record both
   chosen candidate IDs, candidate hashes and selector receipts before accessing
   new evaluation results. SAFE's existing verified choice supplies its receipt.
4. Run the exploratory judge once under a fresh
   `runs/model_selection_v2/judge_<version>_<date>/` directory. Preserve existing
   runs. Save protocol, clean/stressed predictions, references/seeds, metric
   tables, paired differences, dominance matrix, bootstrap summaries, plots,
   source/input/output hashes and a verified completion receipt.

The first implementation can proceed before our selector is ready. The final
selected-model comparison cannot: our method still needs a specified choice.
Selection-set runs, if useful for development, must be explicitly labelled
development diagnostics and cannot substitute for the independent comparison.

## Evidence inspected

- [User's unchanged 21-slide deck](../../output/presentations/credit-risk-research-presentation-16102026_v2.pptx):
  slides 7–16 cover architecture, losses, calibration, capacity, fairness, stress,
  open explainability, Pareto trade-offs and evaluation separation; slides 17–18
  contain open questions. The supplied Downloads file was read directly and its
  SHA-256 matched the retained copy:
  `28138bbb933e471448438f1a35c250e97b28cf8562a788fb00e586cabcf9bd3d`.
- [Pilot implementation](pilot.py), [guide](README.md),
  [saved audit](../../runs/model_selection_v2/pilot_candidates_2026-10-26/diagnostics/data_audit.json)
  and [protocol](../../runs/model_selection_v2/pilot_candidates_2026-10-26/protocol.json):
  exact saved roles, models and group preservation.
- [SAFE specification](safe2026_specification.md) and
  [verified selection report](../../runs/model_selection_v2/safe2026_swapping_2026-10-26/report.md):
  fixed-model comparison and saved logistic choice.
- [Historical semantic-stress rationale](../../outputs/tables/robustness_evaluation/protocol_rationale.md):
  source of proposed rule reuse; its settings are not automatically accepted
  for the new judge.
- [Scikit-learn probability calibration documentation](https://scikit-learn.org/stable/modules/calibration.html),
  consulted 30 October 2026: proper probability scores assess more than calibration
  alone. This supports the separate reliability diagnostic.
- [Cawley & Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html),
  consulted 30 October 2026: model-selection overfitting can bias performance
  evaluation. This supports retaining a boundary between method development and
  judge evaluation, not a claim that this historically inspected holdout is pristine.
