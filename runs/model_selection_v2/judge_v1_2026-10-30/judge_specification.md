# Pareto judge v1: predeclared exploratory assessment

Specified **30 October 2026**, before selection-exploration results or new
historical-holdout predictions. This is an agent-specified implementation decision
under the user's research leeway. **Primary evaluation is null**: no scalar
primary metric, weight vector, hard gate or total-order winner is introduced.

This specification fixes the first assessment rather than repeatedly changing
the judge according to which selector it favors. It supersedes the earlier
judge proposal's broad menu for this particular v1 assessment. A new version
requires an explicit methodological rationale and disclosure of data already
seen. Later versions cannot retrospectively replace v1 findings.

## Question and claim boundary

For the unchanged candidate pool, what clean probability-quality, sex-disparity
and joint-stress trade-offs follow from choices made by the prespecified SAFE,
simple loss-based and screened-SAFE selectors?

The 5,996 Taiwan records are excluded from the current candidate fitting and
selection but have been historically inspected. This assessment is therefore
**exploratory out-of-sample evidence, not pristine confirmation or sufficient
independent evidence for a new superiority claim**. Both the original record
history and the prior development diagnostics remain disclosed.

The comparator is the verified Taiwan operationalization of integrated SAFE
2026, with its accepted masking/additive-RGA choices. It is not a claim to exactly
reproduce unpublished author settings or all later SAFE variants.

## Preconditions for execution

1. Verify existing candidate and SAFE run receipts, bound sources and split IDs.
2. Complete the selection-only exploration under its saved protocol without
   historical-holdout access. Preserve its results, including null findings.
3. Create a freeze receipt identifying the method version, this judge version,
   exploration specification, candidate hashes, all selector choices and ties,
   scenario code, software versions and source hashes before new evaluation
   predictions. The reference choice uses q=.10, even if q=.20 favors a method.
4. Root reviews tests and code for data-role separation, scenario coupling,
   original-group resampling and independent evidence reconstruction.
5. A new assessment run must have a fresh destination and cannot overwrite
   completed research. The pipeline must expose no fitting/tuning operation.

## Data and models

- Evaluate the four frozen pilot pipelines fitted on 19,203 records, with saved
  preprocessing, 22 original predictors and original feature order.
- Use the exact 5,996 saved evaluation IDs/order/outcomes/SEX/group metadata.
  Verify complete separation from fitting/selection IDs and original predictor
  groups. No repartition or refit on 24,004 development records.
- Fitting records supply every numeric mean/IQR, category mode/frequency and
  support/bound reference. No statistic is estimated from evaluation predictors
  to tune a transformation or a model.
- Decisions remain strict `p(default)>.50`, ties class 0. SEX=1 male, 2 female
  is metadata; signed gaps are female minus male. Flags are not observed denials.

## Fixed scenario and measurement design

Use the same declared paired mask/corruption/joint operations and the three
repayment/billing/payment families in
[exploration v1](exploration_v1_specification.md). Retain clean and all nine
family/condition combinations. Use q=.10 reference and q=.20 sensitivity, each
with 30 realizations. The new scope string is `judge_v1:<family>` in the same
SHA-256 seed construction, master 20261007, repetitions 1..30. This gives fresh
scenario streams without retuning fault definitions. All candidates share each
perturbed input frame, and all modes share the declared row masks.

The judge's six Pareto coordinates, all lower-is-better, are fixed as:

| Coordinate | Definition |
| --- | --- |
| clean_brier | Mean `(p-y)^2` on clean records |
| clean_logloss | Mean natural-log binary loss with float64 eps clipping |
| absolute_tpr_gap | Absolute female-minus-male clean TPR gap at strict .50 |
| absolute_fpr_gap | Absolute female-minus-male clean FPR gap at strict .50 |
| worst_joint_brier | Largest mean Brier over the three q=.10 joint scenarios |
| worst_joint_logloss | Largest mean LogLoss over the three q=.10 joint scenarios |

Calculate per-realization losses first, average within each scenario, then take
the maximum across the three scenarios. Never maximize per record or calculate
loss on averaged probabilities. Report which scenario attains each maximum.
Brier uses original probabilities; LogLoss clips only for its own computation
to `[eps,1-eps]`, `eps=2^-52`. Save clipping/exact-zero/one counts.

This panel follows the original deck's probability quality, error disparity and
outcome-based stress concerns. Both proper losses remain peers. The method's
selection vector does not contain fairness, so these judge axes also expose
possible fairness trade-offs. A small gap alone is not a fairness guarantee:
publish signed gaps, subgroup rates and denominator/confusion counts alongside it.

Supporting diagnostics (not extra undeclared dominance axes): clean AUC/sklearn
AP, overall recall/FPR, ten fixed equal-width reliability bins and counts, all
scenario losses, signed loss deterioration and factorial interaction contrasts,
probability movement and flips. Keep q=.20 as a separate sensitivity table and
frontier; it cannot replace the reference result. Explainability remains an
explicit unassessed independent dimension. Do not claim complete trustworthiness.

## Comparison rule

For each declared method, map its frozen selection receipt to its chosen saved
pipeline. Identical choices give identical predictions and are recorded as the
same choice, not independent wins by several methods.

Evaluate all four candidates once on shared data. A strictly dominates B if it
is no worse on every unrounded coordinate and better on at least one. Record
dominance, equality, trade-off or unresolved coordinates. Report the full pool
frontier and pairwise dominance matrix, with the selected models marked. Missing
required coordinates leave status unresolved; no record/coordinate/candidate
dropping or zero substitution is permitted to manufacture a frontier.

The declared principal contrast for presentation is
`screened_safe_outcome_v0 choice minus safe_arithmetic choice`, with all fixed
candidate-pair contrasts and baseline/ablation choices also reported. Principal
contrast identifies the thesis comparison; it does not select a primary outcome.
Retain GBM-minus-LR for that named family comparison. Signs for differences of
absolute gaps and losses are interpreted using lower-is-better directions.

There is no post-evaluation replacement selection. Finding a better-performing
third candidate is diagnostic evidence about selection quality, not permission
to retune the selector against this evaluation set.

## Conditional uncertainty

Use 2,000 whole-original-predictor-group draws with replacement, seed 42, shared
across candidates and every saved scenario realization. Preserve groups crossing
sex/outcome strata and use record-weighted metrics with group multiplicities.
Recompute rates, absolute gaps, scenario losses and scenario maxima in each draw.
Selected identities and models stay fixed. Scenario realizations stay fixed;
report Monte Carlo repetition SD separately from these sampling intervals.

Save 95% percentile intervals with valid counts for each coordinate, signed gaps,
paired differences and factorial contrasts. Save bootstrap Pareto relations as
exploratory stability frequencies, not probabilities of true superiority or
simultaneous confidence certification. Marginal 95% intervals do not jointly
certify dominance. Absolute gaps and maxima are non-smooth, so boundary/tie
behavior must remain visible; no formal equivalence or familywise significance
claim is made from these exploratory intervals.

This uncertainty is conditional on the fixed fitted models, development choices
and declared stresses. It omits training/selection variability and does not undo
historical use of Taiwan. Full-procedure repeated evaluation is a later design.

## Interpretation and independent-validation handoff

Allowed results include observed dominance, a trade-off, identical selection,
uncertain differences, and failure to improve over a simpler baseline. No practical
equivalence is declared without a justified margin. No legal discrimination,
causal mechanism, real fault frequency, future-population safety, universal
winner or novelty claim follows from this assessment.

Before making a stronger claim, freeze the claim and its evaluation design and
test on evidence not used to develop it. A separately specified additional
dataset must be chosen on documented relevance/access criteria rather than
favorable method results. A repeated synthetic study can test a scoped mechanism
and training/selection variation, but does not substitute for population evidence
or establish realistic corruption frequencies. Preserve negative findings in
both stages. This handoff is required even if the exploratory result looks strong.

## Deliverables and verification

Save the freeze receipt, exact scenario/fitting references and seeds, aligned
clean/stressed predictions, input hashes/exposure diagnostics, per-realization
metrics, scenario/coordinate tables, selector-to-model mapping, pairwise/frontier
results, bootstrap draw evidence/summaries, plots and an interpretation report.
Reconstruct losses, signed gaps, maxima, contrasts and dominance independently
from saved predictions; check original artifact preservation and receipt hashes.
Use plots only as labelled projections, not substitutes for full-vector dominance.

No empirical execution is established by this specification. Completion requires
the actual artifacts and their verification; publication readiness remains a
separate judgment about contribution, evidence and prior art.
