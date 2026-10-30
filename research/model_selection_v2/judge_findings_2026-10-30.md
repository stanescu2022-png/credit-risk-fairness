# Judge v1: results and publication assessment

Completed **30 October 2026**. The predeclared exploratory comparison is
implemented, frozen, executed and independently reconstructed. It demonstrates
a probability-loss/disparity trade-off, **not Pareto superiority or a distinct
benefit of the full proposed selector over simpler alternatives**.

Evidence: [freeze receipt](../../runs/model_selection_v2/judge_v1_freeze_2026-10-30/FREEZE.json),
[assessment report](../../runs/model_selection_v2/judge_v1_2026-10-30/report.md),
[independent root reconstruction](../../runs/model_selection_v2/judge_v1_review_2026-10-30/review.json),
[development findings](selection_findings_2026-10-30.md).

## What was fixed before evaluation

The judge specification predates the new selection-exploration results. The
immutable freeze was created at 19:39:25 UTC on 30 October, before evaluation
data were read by this assessment. It binds code/source snapshots, settings,
candidate identities, all reference choices/ties, diagnostic sensitivity choices
and software versions. Implementation was not fully blind: the implementation
agent read a resume checkpoint containing development choices. No scientific
judge definitions were changed in response to those choices or evaluation results.

The four candidates retain preprocessing and model fits from 19,203 fitting
records. Selection used the separate 4,801 records. Evaluation used the exact
saved 5,996 records in 5,977 original-predictor groups. The historical evaluation
records were previously inspected, so these are exploratory out-of-sample
results, not pristine independent confirmation. No refitting, tuning,
recalibration, threshold change or post-evaluation reselection occurred.

The full outcome screen selected **RF**; original arithmetic SAFE selected
**logistic**. Marginal-only screening and simple worst-LogLoss selection also
selected RF. Clean Brier/LogLoss, clean-only screening and reference worst-Brier
selected GBM. Geometric/quadratic SAFE also selected logistic. Identical choices
are one candidate result, not separate empirical successes.

## Reference q=.10 findings

All six coordinates are lower-is-better peers. No primary metric, weights or
practical-dominance margin is introduced. Sex gaps below are absolute differences
in **percentage points**; signed gaps and subgroup counts remain in the artifacts.

| Candidate | Clean Brier | Clean LogLoss | TPR gap (pp) | FPR gap (pp) | Worst joint Brier | Worst joint LogLoss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic / SAFE | .138177 | .441159 | 2.2269 | .6632 | .142529 | .453942 |
| Decision tree | .140856 | .449886 | 2.2358 | .6024 | .144390 | .459532 |
| RF / full screen | .136250 | .434648 | 2.5518 | .7715 | .139541 | .443148 |
| GBM / clean loss | .135453 | .431304 | 3.0419 | .8587 | .139346 | .442063 |

RF has lower values on all four probability-loss coordinates than logistic,
but larger observed sex gaps on both disparity coordinates. Their relation is
**incomparable**. All four candidates are on the full six-coordinate frontier.
Frontier membership does not mean equal practical quality: even a small opposing
coordinate difference prevents exact dominance when no practical margin exists.

GBM has lower point estimates than RF on all four reference probability-loss
coordinates, while RF has smaller observed disparity gaps. Thus the full method
does not establish better reference probability quality than clean-loss selection.
All joint maxima in this assessment are attained by the repayment joint scenario;
the complete scenario matrix is retained rather than reporting maxima alone.

## Conditional uncertainty

The following are marginal 95% percentile intervals for **RF minus logistic**,
using 2,000 shared whole-original-group bootstrap draws. Models, choices and
saved disturbances remain fixed. These are not simultaneous dominance certificates
and do not include fitting/selection variability.

| Coordinate | Point difference | Conditional 95% interval |
| --- | ---: | --- |
| Clean Brier | -.001927 | [-.003579, -.000287] |
| Clean LogLoss | -.006511 | [-.011607, -.001675] |
| Absolute TPR gap (pp) | +.3249 | [-2.2451, +2.6773] |
| Absolute FPR gap (pp) | +.1083 | [-.6276, +.7774] |
| Worst joint Brier | -.002988 | [-.004547, -.001470] |
| Worst joint LogLoss | -.010794 | [-.015704, -.006124] |

The probability-loss intervals favor RF in this exploratory assessment. The gap
differences remain uncertain; larger observed gaps are not established population
deterioration. RF and logistic are incomparable in 1,618 bootstrap draws and RF
dominates in 382. These are conditional relation-stability counts, not an 80.9%
or 19.1% probability statement about true superiority.

For GBM-minus-RF, all six reference coordinate intervals include zero. This does
not establish equivalence or justify choosing an outcome after seeing results.
At q=.20, RF has lower point estimates for the two stressed losses than GBM, but
those pair intervals also include zero. q=.20 remains sensitivity only; the
reference outcome and fixed q=.10 identities are unchanged. The full frontier
and RF-versus-logistic trade-off remain at both severities.

## What this means for the article

The strongest supported direction is a **decision-scope audit**: preservation
scores and outcome losses answer different questions, and a high integrated
preservation-oriented score need not select the lowest-loss candidate. The
synthetic affine audit, development outcome domination and original same-swap
loss diagnostic provide complementary evidence with their scopes disclosed.
The independently specified judge exposes the resulting trade-offs.

The proposed screen is implemented and empirically nondominated on its declared
development vector, but that property follows from its definition. The pilot
does not show unique added value from joint screening or combining the screen
with SAFE: the marginal ablation and simpler worst-LogLoss baseline select the
same model. Probability-sensitive SAFE, integrated-score uncertainty and Pareto
risk-control methods already exist; see the [prior-art review](literature_update_2026-10-30.md).
Novelty remains unestablished. An article claiming that our new selector is
generally superior is not supported by this package.

Preserve v0 and this judge unchanged. A stronger claim needs the separately
specified [independent-validation stage](validation_handoff_2026-10-30.md), with
dataset/scenario relevance documented before results, appropriate full-procedure
uncertainty and simple baselines retained. The primary evaluation remains open;
choose it prospectively from the intended decision problem, not because a
particular metric or stronger disturbance makes this pilot look favorable.
Independent explainability assessment and realistic fault-frequency evidence
also remain unresolved. Publication readiness is not claimed.

## Verification and preserved outputs

Sol high implemented the judge; root reviewed it and independently reran all
23 synthetic tests. The empirical run passed exact role/group accounting,
1,285 pre-existing-file preservation checks, frozen model-state/source checks,
72 full-frame prediction-reuse audits (maximum deviation zero), and its saved
evidence reconstruction. All four projection figures were visually reviewed.

Root separately rebuilt every loss and signed/absolute-gap value for all 2,000
draws from saved predictions, reproduced the bootstrap draws, and checked all
candidate/pair/principal intervals, full-vector relations and frontiers at both
severities. Maximum reconstruction difference was 2.78e-16. Numerical reproduction
tolerances are not Pareto margins. No completed source or result was overwritten,
and no commit, push, publication or external contact was performed.
