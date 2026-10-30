# Predeclared Pareto judge v1 — 30 October 2026

Historical Taiwan out-of-sample exploration on the saved 5,996 records. These records were excluded from the current candidate fitting/selection but were historically inspected; this is not pristine confirmation or sufficient independent evidence of a new superiority claim.

The displayed q=.10 contrast is random_forest (full-screen choice) minus logistic (arithmetic-SAFE choice): **incomparable**. Primary evaluation remains null; no scalar primary metric or total-order winner is introduced.

| Method | Frozen q=.10 candidate used in both judge designs | q=.20 development choice (diagnostic only) |
| --- | --- | --- |
| safe_arithmetic | logistic | logistic |
| safe_geometric | logistic | logistic |
| safe_quadratic | logistic | logistic |
| clean_brier | gradient_boosting | gradient_boosting |
| clean_logloss | gradient_boosting | gradient_boosting |
| worst_brier | gradient_boosting | random_forest |
| worst_logloss | random_forest | random_forest |
| screened_safe_clean | gradient_boosting | gradient_boosting |
| screened_safe_marginal | random_forest | random_forest |
| screened_safe_outcome_v0 | random_forest | random_forest |

Identical method choices share one pipeline and predictions and are recorded as same_choice, not independent wins. Different candidates may have equal measured vectors. Every candidate was evaluated once on each common input. The q=.20 assessment changes stress severity only; it keeps q=.10 frozen identities and cannot replace the reference result.

## q0.10: reference

| Candidate | clean_brier | clean_logloss | absolute_tpr_gap | absolute_fpr_gap | worst_joint_brier | worst_joint_logloss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic | 0.13817667 | 0.44115879 | 0.02226871 | 0.00663214 | 0.14252876 | 0.45394200 |
| decision_tree | 0.14085587 | 0.44988563 | 0.02235812 | 0.00602354 | 0.14438979 | 0.45953216 |
| random_forest | 0.13625001 | 0.43464816 | 0.02551819 | 0.00771497 | 0.13954093 | 0.44314762 |
| gradient_boosting | 0.13545282 | 0.43130398 | 0.03041947 | 0.00858748 | 0.13934559 | 0.44206340 |

Reported frontier (full_pool): logistic, decision_tree, random_forest, gradient_boosting. 

## q0.20: separate stress sensitivity

| Candidate | clean_brier | clean_logloss | absolute_tpr_gap | absolute_fpr_gap | worst_joint_brier | worst_joint_logloss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic | 0.13817667 | 0.44115879 | 0.02226871 | 0.00663214 | 0.14666245 | 0.46618868 |
| decision_tree | 0.14085587 | 0.44988563 | 0.02235812 | 0.00602354 | 0.14839372 | 0.47114450 |
| random_forest | 0.13625001 | 0.43464816 | 0.02551819 | 0.00771497 | 0.14272625 | 0.45147761 |
| gradient_boosting | 0.13545282 | 0.43130398 | 0.03041947 | 0.00858748 | 0.14313403 | 0.45281090 |

Reported frontier (full_pool): logistic, decision_tree, random_forest, gradient_boosting. 

All six unrounded coordinates are lower-is-better peers. Dominance uses no numerical or practical margin. Display rounding and the 1e-15 prediction-reproduction tolerance never enter dominance. Full candidate pair matrices and all six fixed pair contrasts, later pilot order minus earlier (including GBM-minus-LR), accompany the principal contrast. A small sex gap is not a fairness guarantee; signed female-minus-male gaps, subgroup rates, denominators/confusion counts, clean AUC/sklearn AP and reliability bins remain visible.

Worst-joint coordinates maximize the three scenario means after averaging repeated losses. Every exact maximizing scenario is retained. LogLoss alone clips float64 probabilities to [2^-52,1-2^-52]; exact-zero/one and clipping counts are saved. Supporting diagnostics include all nine scenario losses, signed deterioration, mean absolute probability movement, strict p>.5 classification flips, and paired factorial loss contrasts L11-L10-L01+L00. These contrasts describe interaction on the chosen loss scale: Brier can show a contrast under additive probability responses. They do not establish causal synergy or nonlinear model/feature interactions. Explainability is unassessed; this is not complete trustworthiness.

Shared 2,000 whole-original-group bootstrap draws (seed 42) retain unequal group sizes and crossing SEX/outcome groups. Rates, signed/absolute gaps, scenario loss means, maxima and full-vector relations are recomputed; chosen IDs, fitted models and scenario realizations stay fixed. Percentile intervals include valid/undefined counts. Dominance/frontier frequencies describe exploratory conditional stability, not true-superiority probabilities or joint confidence certification. Monte Carlo repetition SD remains separate. Fitting and selection variability are omitted; boundary/tie behavior is retained.

Plots are labelled two-coordinate projections and cannot establish full-vector dominance. Source/input/choice identities are frozen before evaluation access. The guarded loader checks exact saved membership/order and original-group separation, without repartition, refitting, calibration, tuning or threshold change. Saved predictions support independent reconstruction of losses, gaps, maxima, contrasts and relations; preservation and hash receipts establish artifact completion, not publication readiness.

Preserve negative, same-choice, trade-off, uncertain and failure-to-improve findings. There is no post-evaluation replacement selection. No practical equivalence, legal discrimination, causal mechanism, real fault frequency, future-population safety, universal winner or novelty claim follows. A stronger claim requires a separately frozen design and evidence not used to develop it. Future dataset relevance/access criteria and full-procedure resampling are separate decisions.
