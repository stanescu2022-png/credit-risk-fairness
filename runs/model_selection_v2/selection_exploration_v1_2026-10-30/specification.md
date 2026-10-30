# Selection-only exploration v1

Specified **30 October 2026**, before generating this experiment's new scenario
predictions. Status: **exploratory protocol chosen under the user's research
leeway**, not the final independent judge or a preregistered confirmation study.
Primary evaluation remains undecided. Preserve this version once executed.

## Purpose and hypotheses

Test whether explicitly measured joint input disturbances and probability losses
provide decision information beyond the saved integrated SAFE score and simple
probability-loss selection. The current clean selection metrics and SAFE choice
were already known when designing this exploration. No historical holdout result
may determine its scenarios, settings, method or reporting.

H1 (mechanism, already synthetic): integrated-score affine invariance can coexist
with different probability losses. Do not rerun or expand the accepted empirical
candidate pool with transformed predictors for this exploration.

H2 (open): paired joint disturbances have outcome-loss consequences not adequately
described by their separate marginal effects for these fitted candidates.

H3 (open): adding outcome-risk screening to SAFE changes its choice in a useful
way beyond clean-loss screening and marginal-only screening. Same-choice and
inconclusive results count as results; no change to the method is required.

These data can develop hypotheses and assess feasibility. They cannot establish
out-of-sample superiority of a selector developed using them.

## Inputs and protected roles

- Reuse the verified four-candidate pilot and saved SAFE arithmetic scores.
  Verify run receipts and bound source/input hashes before model loading.
- Fitting: 19,203 records, used only for replacement values, scales and support.
- Selection: 4,801 records, exact saved IDs, labels, SEX, group IDs and order.
- No new holdout predictors, outcomes, predictions or metrics. Use the existing
  train-only contract loader, not the historical full-data evaluation runners.
- No refitting, calibration, tuning, threshold changes or category pooling.
- Metadata is excluded from the 22 predictors. Bootstrap uses the saved whole
  original-predictor groups, including groups crossing SEX/outcome categories.

## Fixed disturbance design

Three semantic feature families:

| Family | Original predictors |
| --- | --- |
| repayment | PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6 |
| billing | BILL_AMT1 through BILL_AMT6 |
| payment | PAY_AMT1 through PAY_AMT6 |

These are feature families, not empirically verified production feeds. Missingness
means a synthetic input-availability fault followed by a specified replacement;
it is not permanent feature elimination or a validated imputation policy.

For each family, use paired conditions:

- **00 clean:** saved original predictions.
- **10 mask only:** replace the family's six predictors on selected records with
  fitting-only numeric means or categorical modes (smallest category for a mode
  tie), consistently with SAFE's current masking convention.
- **01 corruption only:** apply the existing semantic-corruption rules to the
  complementary 16 predictors on the same selected records.
- **11 joint:** apply both operations on those records. The feature sets are
  disjoint, so the transformations must commute.

For each family/realization select `floor(q*n)` records uniformly without
replacement. Use q=.10 as the fixed reference design and q=.20 as a reported
sensitivity design; neither declares a primary outcome metric or a real-world
error frequency. Use 30 realizations. Preserve the historical semantic rule
definitions read-only, but recompute every reference on the fitting subset.
Monetary shifts are q times fitting IQR, age shifts are 1/2 at .10/.20, repayment
steps follow the fitting-supported map, and category replacements use fitting
frequencies conditional on a different value. Thus the .20 sensitivity changes
both exposure and perturbation magnitude, a disclosed design choice.

Use existing SHA-256 seed construction, master 20261007, with a new scope string
`exploration_v1:<family>`, repetitions 1..30 and the existing component labels
`mask`, `sign:<feature>`, `category:<feature>`. Scope and repetition are shared
across candidates/conditions; severity-independent seeds nest the .10 selected
records inside .20 and reuse signs/uniforms. Mask-only uses the identical
selection permutation as corruption-only. Save all actual child seeds.

Calculate the joint interaction contrast for each loss:
`I=L11-L10-L01+L00`. Keep its sign. This is a paired factorial description under
the declared input-error model, not a causal default-risk interaction. Exposure
is matched at the record level; joint scenarios alter more cells by design.
Record selected/changed rows and cells, no-ops, clipping and support diagnostics
for every condition. Do not interpret a larger raw joint loss by itself as an
interaction beyond the marginal effects.

Outcomes stay fixed because the operations represent measurement corruption.
All candidates see identical transformed inputs. It is permissible to predict
only changed rows and copy saved clean probabilities for other rows if full-frame
prediction equivalence is verified on fixed audit cases for every candidate:
repetition 1 for every family, severity and condition, absolute tolerance 1e-15.
Save maximum deviations and the aligned row identities. This numerical audit
tolerance does not enter Pareto dominance or practical-effect interpretation.

## Metrics and fixed exploratory selectors

For each candidate/condition/realization calculate binary Brier and natural-log
LogLoss with an explicit float64 endpoint convention. Average **losses** over
realizations, never compute loss of an averaged probability. Retain per-realization
values and Monte Carlo SD, plus clipping and exact-zero/one probability counts
by candidate/scenario. Clean AP/AUC and fixed-threshold signed sex gaps can
be reported as descriptive selection diagnostics, never independent results.

For q=.10 define a six-coordinate loss vector:

1. clean Brier;
2. clean LogLoss;
3. maximum mean Brier across all six marginal scenarios (3 families x 2 modes);
4. maximum mean LogLoss across those marginal scenarios;
5. maximum mean Brier across the three joint scenarios;
6. maximum mean LogLoss across those joint scenarios.

Take the maximum **after** averaging each scenario's repeated losses. The worst
scenario may differ by candidate and by loss. Preserve the complete scenario
matrix to interpret these maxima. All coordinates have direction lower-is-better.
This declares a conservative finite-scenario preference, not a probability model
over faults or a universal risk criterion.

Working selector `screened_safe_outcome_v0`: remove strictly Pareto-dominated
candidates on all six unrounded coordinates, then choose the largest saved SAFE
arithmetic score among retained candidates. Final SAFE ties follow its existing
1e-12 score tolerance and fixed candidate order. Practical dominance itself uses
no tolerance or margins. Any required undefined/nonfinite quantity makes the
selector unresolved; do not silently drop candidates or axes.

Compute all the following without choosing the most favorable after results:

| ID | Definition |
| --- | --- |
| safe_arithmetic | Unchanged saved arithmetic SAFE choice |
| safe_geometric / safe_quadratic | Saved sensitivity choices |
| clean_brier / clean_logloss | Minimize the respective clean selection loss |
| worst_brier / worst_logloss | Minimize the largest mean loss across clean and all nine reference scenarios |
| screened_safe_clean | Pareto screening on coordinates 1-2, then SAFE arithmetic |
| screened_safe_marginal | Pareto screening on coordinates 1-4, then SAFE arithmetic |
| screened_safe_outcome_v0 | Full six-coordinate screening, then SAFE arithmetic |

For exact minimum-loss ties use fixed pilot order and record all ties. Report
the q=.20 scenario matrix and the same selector definitions evaluated at that
level as a labelled sensitivity, never silently substitute its choice for the
q=.10 reference choice. Same candidate choices mean the methods coincide for
this experiment; they are not independent replications of model performance.

## Conditional uncertainty and limits

Use 2,000 shared whole-original-group bootstrap draws, seed 42, on the selection
records. Recompute marginal/joint mean losses, maxima, Pareto sets and the
screening/minimum-loss choices within each draw. Keep the saved SAFE scores fixed:
the resulting selection frequencies describe sensitivity of the added outcome
layer conditional on the existing SAFE assessment, **not** a bootstrap of the
complete SAFE pipeline. Full SAFE recomputation is outside this pilot.

Report paired loss and interaction-contrast percentile intervals, valid counts,
candidate retention/selection frequencies and scenario identities attaining maxima.
The paired comparisons are all six fixed candidate pairs, later pilot-order
candidate minus earlier; this preserves GBM-minus-LR. Also report each candidate's
interaction interval. Do not construct intervals for a different winning model
chosen afresh in each draw and call them fixed-pair uncertainty.
With additive losses it is valid to precompute each record's loss averaged over
realizations, then apply group multiplicities; this must be tested against direct
per-realization recomputation. Preserve Monte Carlo SD separately.

No significance-based claim of a new winner, practical equivalence, formal
simultaneous dominance certification, deployment recommendation or new method
superiority follows. Fairness remains descriptive and explainability unresolved.
Further full-procedure repeated fitting/selection and later independent evidence
are distinct stages in the research agenda.

## Outputs and verification

Implement under a new versioned module path, leaving completed sources unchanged.
Run into a fresh `runs/model_selection_v2/selection_exploration_v1_2026-10-30/`
directory with overwrite refusal. Save this specification and its hash before
prediction; fitting references, seeds, exposure counts, predictions, per-realization
losses, aggregated scenario matrix, interactions, selector decisions, bootstrap
summaries, source/input/artifact identities and a verified receipt.

Before empirical execution, test paired masks, nested exposure, transformation
commutation, valid categories, fitting-only references, frozen model behavior,
unchanged-row prediction reuse, exact source alignment, direct loss aggregation,
max-after-mean ordering, synthetic selector/ablation cases and whole-group
bootstrap reconstruction. Root reviews the implementation and interprets results.

Decision after the run: assess whether the proposed mechanism matters in this
pool, whether simple baselines explain any apparent benefit, and whether a
well-defined independent validation is warranted. Preserve failures and null
findings. No historical-holdout tuning or outcome-driven judge revision.
