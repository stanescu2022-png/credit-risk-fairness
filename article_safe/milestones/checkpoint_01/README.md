# Checkpoint 1: foundation ready for review

10 October 2026. Stopped at the user-requested first milestone, approximately 25% of the implementation scope. The four work packages are foundation, model fitting/selection, final SAFE assessments, and reporting/complete-run verification. This is a scope estimate, not a measured percentage of runtime or scientific completion.

The implementation now has its own `article_safe/` package. Its protocol records the original article, the agreed statistical conventions and the distinction between article procedures and our explicit interpretations. The saved regularized LR remains an estimator reference; the unpenalized article LR will be implemented next.

The preparation stage ran successfully on the actual project data:

| Role | Records | Original predictor groups |
| --- | ---: | ---: |
| Fitting F | 19,203 | 19,147 |
| Selection V | 4,801 | 4,783 |
| Saved holdout H | 5,996 | Checked structurally, without performance evaluation |

F and V together preserve all 24,004 training records. No original predictor group crosses F/V or train/H. The selection interface exposes only saved training rows; ID, SEX and target remain metadata. The [data audit](../../runs/checkpoint_01_2026-10-10/data_audit.json) and [assignment table](../../runs/checkpoint_01_2026-10-10/development_assignments.csv) record the actual partition.

The statistical foundation is implemented and tested. Paired DeLong covariance agrees with an independent direct placement calculation, including ties. Likelihood-ratio tests use the whole block's rank difference. Gini distinguishes concentration from signed effects, and KS compares individual contribution distributions. Missing or degenerate tests stay visible. The selection tests enforce the .05 boundary, reject significant deteriorations and prevent a later improvement from undoing the first forward stop.

Validation passed: **36 new tests and all 30 existing project tests**. All 410 saved evaluation-manifest hash checks matched before and after preparation. An inventory of 369 pre-existing project files was captured before implementation; the final verification receipt documents preservation, with only the authorized project-context update allowed to differ.

The [SAFE package audit](../../runs/checkpoint_01_2026-10-10/package_reference.json) identifies version 0.8.3, inspected source hashes and available Python functions. It supports the existing native-metric baseline. The new article implementation supplies the older paper's additional procedures; it has not yet generated any package metric or credit prediction.

**There are no new credit-model results yet.** Full/reduced credit models, the actual selection paths, SHAP/parameter assessment, final fairness tables and the final results report belong to the remaining work.

After approval, the next work package will add the unpenalized GLM and fixed-setting GBM adapters, whole-predictor preprocessing, numerical-fit diagnostics, and both selection paths on F/V. Those choices will then be frozen before final fitting and holdout evaluation. The original methodology's profiles will determine the next discussion about experiments.

Review whether this separate foundation and its recorded conventions match the intended article copy. Implementation work is paused until Andrei gives the OK, as requested. No commit, publication or external contact occurred.