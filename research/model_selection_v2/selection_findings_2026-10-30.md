# Selection exploration v1: findings and decision

Executed and independently reviewed **30 October 2026**. Evidence:
[run report](../../runs/model_selection_v2/selection_exploration_v1_2026-10-30/report.md),
[complete reference results](../../runs/model_selection_v2/selection_exploration_v1_2026-10-30/q0.10/selectors.json),
[independent reconstruction](../../runs/model_selection_v2/selection_exploration_review_2026-10-30/review.json).
These are development results on 4,801 selection records, not independent
validation or a superiority finding.

## Decisions under the unchanged protocol

| Rule | q=.10 reference | q=.20 sensitivity |
| --- | --- | --- |
| SAFE arithmetic / geometric / quadratic | Logistic | Logistic |
| Clean Brier / clean LogLoss | GBM | GBM |
| Worst-scenario Brier | GBM | RF |
| Worst-scenario LogLoss | RF | RF |
| Clean-only screen then SAFE | GBM | GBM |
| Marginal-only screen then SAFE | RF | RF |
| Full outcome screen then SAFE | RF | RF |

The clean screen retains only GBM. The marginal and full screens both retain RF
and GBM; RF has the higher saved SAFE score. The reference method therefore
chooses RF and unchanged SAFE chooses logistic. These identities will be frozen
for the already specified judge; sensitivity results do not replace them.

| Candidate | Clean Brier | Clean LogLoss | Worst marginal Brier | Worst marginal LogLoss | Worst joint Brier | Worst joint LogLoss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic | .136260 | .435218 | .140815 | .448000 | .140863 | .448386 |
| DT | .140238 | .448615 | .144210 | .459374 | .144172 | .459224 |
| RF | .135345 | .428699 | .138608 | .437072 | .138748 | .437573 |
| GBM | .134388 | .427711 | .138197 | .437641 | .138505 | .438842 |

Displayed values are rounded; all decisions use unrounded values. RF and GBM
each dominate logistic on this *development outcome vector*. GBM has lower clean
losses and lower worst joint Brier than RF, whereas RF has lower worst joint
LogLoss. This is a loss trade-off, not a universal winner.

## What this supports, and what it does not

The pilot supplies an actual selection consequence for the measurement-scope
question: the original SAFE choice is dominated on the specified outcome-risk
development vector. This does not establish that SAFE is invalid on its own
intended dimensions, or that its ranking alone caused a real-world problem.

The saved arithmetic SAFE score can also be decomposed without new predictions:
it is `(mean A + mean E + mean R)/3`. Relative to RF, logistic's score margin
of about .028901 consists of -.002274 from A, +.007938 from E and +.023237
from R, using the original saved
[component means](../../runs/model_selection_v2/safe2026_swapping_2026-10-26/component_areas.csv).
Its largest positive arithmetic advantage is therefore swapping preservation.
This is score accounting, not a causal attribution of the new disagreement to
affine invariance. The new assessment changes both what is measured (outcome loss)
and the disturbance design; it does not isolate the effect of either change.

The crucial negative result is preserved: **joint axes add no observed point-
choice change beyond the marginal screen, and the full method matches the
simpler worst-LogLoss rule**. This pool therefore does not establish a distinct
selection benefit from the full combination. It is not yet a persuasive new-
method superiority result or a demonstrated novelty claim.

Joint information does affect conditional resampling. At q=.10, RF is selected
in 1,234/2,000 draws by the marginal screen and 1,542/2,000 by the full screen;
the full screen selects GBM in 451 draws and logistic in 7. Worst LogLoss selects
RF in 1,549 draws and GBM in 451. These frequencies condition on frozen fitted
models, fixed SAFE scores and saved disturbances. They are not complete selector
uncertainty or probabilities that a model is truly best. At q=.20 the full screen
selects RF in 1,984 draws; that stronger design remains sensitivity only.

The paired RF-minus-logistic clean Brier interval is approximately
[-.002714, .000854], while clean LogLoss is [-.011864, -.001121]. For repayment
joint stress the corresponding intervals are [-.003781, -.000447] and
[-.015958, -.005764]. These are marginal conditional 95% group-bootstrap
intervals on development data, not simultaneous dominance certification or
independent evidence of the selected method's superiority. All six fixed pairs,
all scenarios and all interaction intervals remain available in the run.

Probability quality does not settle disparity: the descriptive selection-set
absolute FPR gap is .006254 for logistic and .010935 for RF, while the absolute
TPR gap is .069236 for logistic and .048451 for RF. This is another trade-off;
fairness is independently included in the predeclared judge and was not silently
added to the selector after observing it.

## Verification and next action

A subsequent, explicitly post-exploration
[same-swap diagnostic](../../runs/model_selection_v2/safe_saved_swap_loss_diagnostic_2026-10-30/results.json)
holds the original saved disturbances and predictions fixed. Across all 23
original swapping nodes, logistic's descriptive mean preservation is .574856
versus RF's .505145, while its mean Brier/LogLoss are .208083/.626233 versus
RF's .199509/.600946. Thus a preservation/outcome-loss disagreement exists even
without changing the disturbances. These uniform-grid averages are not measured
population risks or a new selection rule; no sampling interval was calculated.
All four models and all nodes remain saved.

The [separate comparison correction](../../runs/model_selection_v2/safe_saved_swap_comparison_correction_2026-10-30/results.json)
preserves the first diagnostic and corrects only zero-severity pair labels:
self-comparison scores are mathematically tied at one, despite stored roundoff.
RF-versus-logistic preservation/loss preferences oppose at 17 nodes, agree at 5,
and tie at zero severity, for each loss. Raw scores, losses and grid means are
unchanged. This supports a scoped measurement distinction, not causal attribution
to affine invariance or novel added value from the combined selector.

All 26 focused synthetic tests passed independently under root review. The run
passed 72 full-frame versus changed-row prediction checks, source/model/input
preservation and its artifact receipt. Root separately reconstructed every
scenario/repetition loss, six-axis vector, all ten point choices/ties/frontiers,
all candidate/pair loss and interaction intervals, and bootstrap choice/retention/
tie counts. All 2,000 group draws were reproduced; four draws were additionally
checked directly by record weighting. Maximum numerical reconstruction difference
was 6.39e-15, below the 2e-14 reproduction tolerance, which never enters dominance.
The review checked 623 pre-existing files at completion. No new holdout assessment,
fit, recalibration or tuning was performed.

Proceed with the already specified judge v1, freezing the method and choices
before evaluation. Do not change the method, reference severity, judge axes or
baseline panel to improve the narrative. A stronger contribution remains subject
to the [independent-validation handoff](validation_handoff_2026-10-30.md).
