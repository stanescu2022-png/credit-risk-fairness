# Checkpoint 3: results and the LR decision to review

10 October 2026. Implementation is stopped at checkpoint 3 of 4, as requested. The selection and final-assessment stages ran. The GBM assessments are available; both planned article LR roles are explicitly unavailable under the approved full-rank, unpenalized specification.

The article-style GBM rule selected **PAY_0, LIMIT_BAL and PAY_AMT1**. It stopped when adding the fourth predictor, PAY_2, produced a small decrease in validation AUC. Later prefixes improve again, but the specified first-stop rule keeps the three-predictor choice. The complete [selection path](../../runs/article_v1_2026-10-10/selection/gbm_prefix_path.csv) is saved so this is inspectable.

Both GBMs were then fitted on all 24,004 training records and evaluated on the same saved 5,996 holdout records:

| Model | Predictors | AUC | Brier loss | Default recall at .50 |
| --- | ---: | ---: | ---: | ---: |
| Full GBM | 22 | 0.7781 | 0.1350 | 0.3515 |
| Selected GBM | 3 | 0.7512 | 0.1402 | 0.3447 |
| Saved regularized LR, reference only | 22 | 0.7569 | 0.1382 | 0.3545 |

The reduction saves 19 predictors, but it costs predictive performance in this sample. Selected-minus-full AUC is **-0.0270**, with a conditional paired group-bootstrap 95% interval **[-0.0348, -0.0190]**. The nominal DeLong test also detects a difference (Holm-adjusted p approximately 8.24e-11). These are exploratory results on a historically inspected holdout, not proof of population equivalence or universal superiority.

The three selected predictors account for **40.9%** of the full model's mean absolute SHAP importance on the holdout. Reduced-model explanations were computed separately. Both GBM outcome-on-SHAP regressions are estimable. Per-predictor Gini/KS tables, signed group effects and conditional intervals are saved; they do not assign an overall fairness verdict. See the [full-model group table](../../runs/article_v1_2026-10-10/evaluation/gbm_full_contribution_fairness.csv) and [selected-model group table](../../runs/article_v1_2026-10-10/evaluation/gbm_selected_contribution_fairness.csv).

The LR obstacle is concrete. Its fitting design has 78 columns including the intercept but rank 74; the complete training design has rank 76. Some rare category indicators coincide exactly. In particular, PAY_4=8, PAY_5=8 and PAY_6=8 describe the same single training record. The [diagnosis](../../runs/article_v1_2026-10-10/diagnostics/lr_design_identifiability.json) and [rare-level support table](../../runs/article_v1_2026-10-10/diagnostics/rare_repayment_indicator_support.csv) preserve the evidence. Some levels also occur only with one outcome class, so removing aliases alone would not guarantee a finite unpenalized fit.

The saved regularized LR remains a valid reference. We did not substitute it into the article likelihood tests. The LR reduction loop passed a separate identifiable synthetic-data test; the credit-data issue is reported without changing the encoding, pooling categories or adding a penalty.

Validation passed: **49 new tests, all 30 existing tests, and all 410 prior artifact checks**. Full GBM reproduced the frozen baseline probabilities exactly. Selection hashes were unchanged; all final fits preceded holdout evaluation. Saved metrics, stopping decisions and the first ten paired bootstrap draws were independently recomputed. The [verification receipt](verification.json) records these checks.

The review decision is whether to retain this v1 limitation or authorize a clearly specified training-only treatment of the rare repayment categories before completing the LR comparison. Checkpoint 4 still covers the final narrative/figures and full-run reproduction checks. Work is paused here for your checkup.