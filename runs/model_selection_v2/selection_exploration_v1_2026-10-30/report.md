# Selection-only exploration v1 — 30 October 2026

Exploratory results on the saved selection role, conditional on four frozen pilot candidates and the declared synthetic measurement-error scenarios. The independent primary evaluation remains undecided. This is hypothesis development, not independent validation or evidence of selector superiority.

| Selector | q=.10 reference choice | q=.20 sensitivity choice |
| --- | --- | --- |
| safe_arithmetic | logistic | logistic |
| safe_geometric | logistic | logistic |
| safe_quadratic | logistic | logistic |
| clean_brier | gradient_boosting | gradient_boosting |
| worst_brier | gradient_boosting | random_forest |
| clean_logloss | gradient_boosting | gradient_boosting |
| worst_logloss | random_forest | random_forest |
| screened_safe_clean | gradient_boosting | gradient_boosting |
| screened_safe_marginal | random_forest | random_forest |
| screened_safe_outcome_v0 | random_forest | random_forest |

All selector definitions, baselines and ablations are reported. Same choices mean methods coincide in this pool; they are not independent replications of candidate performance. Scenario maxima are taken after repetition-loss means, with every exact maximizer retained. The q=.20 design increases exposure and magnitude and remains a labelled sensitivity; it never replaces the q=.10 reference choice.

Interaction contrasts are signed L11-L10-L01+L00 under matched record exposure. They describe interaction on the chosen loss scale, not causal or nonlinear model/feature interactions. Brier's quadratic scale can produce a nonzero contrast even when the joint probability response is exactly additive. A larger joint loss alone does not establish an interaction beyond marginal loss effects.

Pareto screening uses exact unrounded losses, with no dominance tolerance or practical margins. The final SAFE tie convention alone uses 1e-12. Adding coordinates expands the nondominated set only when old vectors are distinct; the chosen retained SAFE score then cannot decrease, apart from its declared score-tie policy. Exact old-vector ties can instead split on added axes, remove former frontier members and decrease the chosen SAFE score. Both cases are retained in synthetic tests. Selecting the highest-SAFE frontier candidate does not guarantee that a replacement dominates the original SAFE choice; the synthetic counterexample in the tests preserves this limitation.

The bootstrap shares whole-original-group multiplicities across candidates, scenarios and saved repetitions, retaining SEX/outcome crossings. It recomputes means, maxima, Pareto sets and the added outcome-layer choices while holding saved SAFE scores fixed. Frequencies therefore do not describe complete SAFE/selector uncertainty. Percentile intervals concern fixed candidates and all six fixed candidate pairs, later pilot-order ID minus earlier ID, plus candidate interaction contrasts. No dynamic-winning-model interval or joint confidence certification is claimed.

Monte Carlo repetition SD is saved separately from group sampling intervals. LogLoss uses float64 epsilon clipping only; exact zero/one and clipping counts are retained by candidate/scenario/repetition. Brier uses original probabilities. Clean signed sex gaps remain descriptive and explainability remains unresolved.

The experiment uses fitting-only references and the train-only contract, with no holdout predictors/outcomes/predictions, fitting, calibration or tuning. Per-condition registries bind every repetition to ordered IDs, selected/changed/reused row IDs, common input hashes, actual child seeds and support/exposure diagnostics. Fixed repetition-1 cases compare changed-row prediction reuse with full-frame predictions for every family/severity/condition/candidate at absolute 1e-15; this numerical reproduction tolerance never enters Pareto dominance.

Preserve negative, same-choice and inconclusive findings. Do not optimize on the historical holdout or revise a judge to favor a selector. Independent validation and future full-procedure fitting/selection resampling are distinct later stages. No real-world error frequency, deployment recommendation, practical equivalence or new-method superiority is established.
