# Original SAFE/KAIRI: implementation specification v1

Date: 10 October 2026. Status: recommended methodology prepared for Andrei's review, not a record of user-approved numerical choices or executed results. Sol should implement this document after the user asks it to proceed. This specification replaces the earlier fault-aware comparison proposal as the immediate implementation task.

## 1. Purpose and source

Implement Giudici, Centurelli and Turchetta, *Artificial Intelligence risk measurement*, ESWA 235 (2024), 121220: [DOI](https://doi.org/10.1016/j.eswa.2023.121220). Main anchors are sections 2.1-2.4, Figure 1, section 3 and credit application 4.1. Local source: `docs/context/sources/3. 1-s2.0-S0957417423017220-main.pdf`; SHA-256 `df63164a9c8beab73863dcd642a0a5724bb25853aa49048f0b6f21ccc2361f49`. The text extraction and page images 3-5 were checked against each other.

The research question remains the LR/GBM trade-off. Our Taiwan data, SEX groups and saved partition differ from the paper's SME data, country/industry groups and 70/30 partition. This is a documented methodological adaptation, not a reproduction of its published numbers. GBM is explicitly discussed in section 3.2; random forest is the example used in section 4.1. LR and GBM are the primary families for v1. Do not add random forest to this first implementation.

Use the article's four-dimensional profile. There is no original overall SAFE ranking or scalar to reproduce. Feature-selection outcomes and accuracy comparisons are actual decisions; fairness and explanation outputs remain assessments. Do not implement the proposed worst-scenario score, synthetic input faults, later integrated SAFE score, RGA/RGE/RGR substitutes, PSI or governance in this stage.

## 2. Settled conventions for this blueprint

| Topic | Article anchor | v1 decision | Fidelity label |
| --- | --- | --- | --- |
| Accuracy | 2.1, Fig. 1 | AUC with paired, two-sided DeLong tests | Article procedure |
| LR reduction | 2.2, 3.1 | Backward removal of whole predictors by nested binomial likelihood-ratio tests | Article procedure; removal order explicitly supplied here |
| GBM reduction | 2.2, 4.1 | Fixed SHAP ordering; forward inclusion; first nonsignificant/nonpositive AUC improvement stops | Credit-example interpretation |
| Importance | 2.3, 4.1 | Mean absolute, original-predictor contributions in log-odds | Explicit aggregation/output convention |
| LR estimation | Eq. 2, 3.1 | Fresh unpenalized binomial GLM, separate from saved regularized LR | Necessary estimator adaptation |
| LR parameter inference | Fig. 1, 2.3 | Binomial-GLM Wald z tests, with cluster-covariance sensitivity | Standard binary-model interpretation of the generic T-test label |
| GBM explanation test | 2.3, 4.1 | Outcome-on-SHAP OLS on evaluation records; coefficient t tests | Explicit interpretation, not uniquely determined by paper |
| Gini | 2.4, Table 2 | Standard concentration Gini on nonnegative effect magnitudes; retain signed effects separately | Explicit repair of direction/domain ambiguity |
| GBM KS | 2.4, 4.1 | Two-sample KS on individual signed SHAP values, female versus male | Case-study-based interpretation |
| LR KS on coefficients | Fig. 1, 3.1 | Mark not identifiable from the article; no invented sampling distribution | Unresolved in source, deliberately not fabricated |
| LR distribution comparison | Companion to preceding row | KS on individual linear contributions, separately labeled adaptation | Supporting interpretation, not coefficient KS |
| Overall decision | Article's separate assessments | Four profiles, named pairwise comparisons, no weighted total or automatic all-round winner | Faithful boundary |

These are implementation conventions, not new statistical contributions. Use a machine-readable status for each result: `article_procedure`, `explicit_interpretation`, `supporting_diagnostic`, `not_identifiable_from_article`, or `not_estimable`. A missing statistic is never silently replaced by zero or a favorable verdict.

## 3. Data and sequence

Read the existing cleaned dataset, preparation manifest and saved split. Keep the original 22 predictor definitions and original feature order. ID, SEX and target remain metadata; SEX may index diagnostic subgroups but never enter a predictive candidate. Preserve all rows and the 24,004/5,996 partition.

Use the existing five-fold StratifiedGroupKFold convention (shuffle, seed 42, whole exact-22-predictor groups). In the training partition, take the first generated fold as selection set V and the other four as fitting set F. This is approximately 80/20 within training. Save assignments and actual counts; do not choose the fold by outcomes or performance. Group IDs remain those defined by all 22 original predictors, even when the model uses a subset. Save the ordering of input rows used to generate folds, using the existing `training_folds` interface where feasible.

The sequence is fixed:

1. On F: fit preprocessing, full models, SHAP ordering and all candidate models. LR likelihood-based removal also uses F only.
2. On V: compare consecutive GBM candidates with paired DeLong; freeze both selected predictor lists and their stopping reasons.
3. On the complete 24,004 training records: fit fresh full and selected-subset models, with the same declared settings. This refit occurs before final evaluation. Refit preprocessing at this point, but do not reorder/reselect predictors.
4. On the saved 5,996 holdout H: evaluate those fixed models. H cannot choose a subset, threshold, hyperparameter, formula convention or replacement model. No refit after this evaluation may inherit the old result.

Final model IDs: `lr_mle_full`, `lr_mle_selected`, `gbm_full`, `gbm_selected`. If a selected subset is full, retain both roles without fitting duplicate models. Additional reference row: `lr_frozen_regularized`, loaded read-only from its existing artifact. The reproduced full GBM should match the frozen GBM probabilities within the saved tolerance when the same implementation/configuration is used; investigate any mismatch before making comparisons. The MLE LR is expected to differ from the frozen regularized LR.

The extra V set is a leakage-control adaptation; the paper does not fully specify this separation. H was historically inspected, and this remains exploratory. The prior 60/20/20 proposal belongs to the deferred fault-selection study, not this v1.

## 4. Models and preprocessing

GBM uses the current saved classifier settings: 100 trees, learning rate .1, max depth 3, seed 42 and all other values from its manifest. Clone its numerical scaling and full one-hot encoding within the allowed fitting partition. Do not change complexity settings for reduced subsets.

Article LR uses statsmodels GLM with Binomial family and logit link, an intercept, no regularization, no class weights and no tuning. Scale numeric predictors and use reference-category encoding for categorical predictors (lowest observed category, deterministic ascending order, drop first). Derive category sets from the fitting partition in that phase. Use one common fitted transform for all nested comparisons within a phase. Unknown-category handling must be explicit, counted and reported; use reference-category encoding for LR and the existing all-zero unknown encoding for GBM. Do not pool categories, remove observations or recode repayment states to improve a result.

The installed environment already contains statsmodels 0.15.0 and patsy 1.0.3. Use `article_safe/requirements.txt` to extend `../requirements-explainability.txt` with those exact versions; retain all prior locks. Use project `.venv` for implementation and execution.

Start GLM IRLS with maxiter 200 and tolerance 1e-8. Require convergence, finite log likelihood, finite predictions and identifiable design columns. A numerical retry may use a higher iteration limit or an equivalent likelihood optimizer with the same data/model. Save warnings and gradient/design/Hessian diagnostics. A separated or unidentified fit is not repaired silently with ridge, pseudoinverse coefficient inference, data deletion or feature changes. Preserve other valid outputs, mark the affected branch `not_estimable` and describe the specific obstacle.

For constant or linearly dependent design columns, report them and the design rank. An unidentified term has no coefficient-based inference. Zero-df likelihood comparisons cannot be treated as ordinary chi-square tests. Every selected predictor is counted once, regardless of its number of dummy columns.

## 5. Sustainability as parsimony

### LR backward path

Start from all 22 predictors fitted on F. At each step, for each remaining original predictor j, fit the nested model that removes its entire encoded block. Compute D_j = 2*(logLik_current - logLik_without_j) and df_j = rank_current - rank_without_j. Use chi-square survival probability for valid, positive df. Both likelihoods must be unpenalized and use exactly the same rows and response.

Remove the predictor with the largest p-value if p >= .05. Break exact ties by the saved original feature order. Refit and repeat. Stop when every valid removal has p < .05; an intercept-only endpoint is allowed. Record all removal candidates, not only the chosen one. If a necessary comparison is unestimable, do not declare that all retained predictors are necessary; report an incomplete path.

This supplies an order that the article leaves open. Raw .05 stepwise tests reproduce its style; they are a selection heuristic with repeated data use, not a familywise error guarantee or proof that the chosen subset is the true model.

### GBM forward path

Fit a full GBM on F. Use exact TreeExplainer computation with interventional masking, raw log-odds output, and one 100-record background sampled uniformly without replacement from F after sorting by ID, seed 42. Explain all F records. Sum signed dummy-level contributions into original predictors, then calculate mean absolute contribution. Rank decreasingly, breaking exact ties by original feature order. Freeze this order.

Fit every prefix k=1,...,22 on F with identical GBM settings and store V probabilities/AUC. Computing the complete path is diagnostic; the decision must still follow the first-stop rule, never best-AUC-over-all-k. Begin with k=1. At k>=2, retain the addition only if AUC_k - AUC_(k-1) > 0 and the paired two-sided DeLong p-value is < .05. At the first nonpositive gain or p >= .05, select k-1 permanently. If all additions pass, select 22. The explicit positive-gain condition avoids accepting a significant deterioration. Later recovery along the diagnostic path does not change selection.

An invalid statistical comparison stops the path with an error status; it is not a nonsignificant result. The article's credit example supplies the first-nonsignificant-addition reading more clearly than the generic prose. Report the whole path and the stopping decision together. Do not describe nonsignificance as established equivalence or retained robustness.
## 6. Accuracy and comparisons

Primary outcome is AUC; predicted default probability is always the class-1 output. Use paired DeLong because the competing models predict the same records. Implement midrank/tie handling and verify its covariance against the defining positive/negative placement-vector calculation. The variance must include paired covariance; never add independent AUC variances.

Handle identical predictions or identical placements with zero difference and zero variance as an identical-result case (p=1). If difference is nonzero while variance is effectively zero, return a flagged degenerate comparison, not an invented p=0. Fewer than two observations of either outcome means the covariance is unestimable. Materially negative variances are errors. Use 64*machine_epsilon*max(var_A+var_B, smallest_positive_normal_float) as the covariance-roundoff tolerance and 64*machine_epsilon as the AUC-difference zero tolerance; save both. Only roundoff may be clipped.

On H, predeclare four comparisons: full GBM minus full article LR; selected GBM minus selected article LR; selected LR minus full LR; selected GBM minus full GBM. Keep this sign convention in columns and plots. Provide raw DeLong p-values and Holm-adjusted values across these four final comparisons. Keep all four planned hypotheses in the multiplicity family; an unestimable comparison remains missing in the report and may use an internal p=1 placeholder solely for conservative Holm bookkeeping. Selection-path raw p-values belong to a separate algorithm, not this final family. Comparisons involving the saved regularized LR are descriptive estimator-reference results, not extra primary tests.

DeLong supplies the article's nominal independent-record inference. Also provide 2,000 paired whole-original-predictor-group bootstrap draws, seed 42, for conditional 95% percentile intervals on these AUC differences. Reuse identical draws for all models. Each draw samples the original number of groups with replacement and includes every member at the sampled group multiplicity; do not stratify by SEX or outcome. Save any single-class/unavailable draws and their valid counts. If fewer than 1,900 of 2,000 draws are valid for a quantity, leave its interval unavailable; otherwise use its valid draws and disclose the count. The bootstrap does not redo fitting or selection and is not a guarantee for the full learning procedure. Do not merge its interval with the DeLong p-value into one test. If their readings differ, report both.

Supporting descriptive outcomes: sklearn AP, Brier loss, log loss, confusion counts, recall, precision, F1 and FPR at strict probability > .50, ties class 0. These make results comparable with existing work and do not replace the article's AUC selection rule. Report group versions with female-minus-male signed gaps; no threshold optimization.

## 7. Explainability

For final models, use one shared 100-record background sampled from the full training partition, sorted by ID, seed 42. Apply each model's fitted transform to those same records. Compute log-odds SHAP for every H record. For linear GLM the independent-background contribution is exactly beta_j*(z_ij - mean_background(z_j)); save the intercept-adjusted expected value and verify additivity. For GBM use interventional TreeExplainer with raw output and no approximation.

Save encoded contributions and their original-predictor aggregation. The latter preserves additivity but is not a newly computed grouped-coalition Shapley game. State that independent masking can break feature dependence and one-hot constraints.

Report mean absolute and mean signed contributions separately. For a selected set S, define retained importance using the FINAL FULL model's H contributions: 100*sum_(j in S) mean_abs(phi_full_j) / sum_all mean_abs(phi_full_j). Also save its F-based selection-stage counterpart. Do not calculate a selected model's importance divided by its own total, which would be tautologically 100%. If total importance is zero, report undefined. Recompute reduced-model explanations independently for its own profile.

For LR: report coefficients, odds ratios, model-based Wald z statistics and two-sided p-values; additionally report cluster-covariance Wald results using original predictor-group IDs. Label the normal-reference choice as the binary-GLM interpretation of the paper's generic T-test. Do not call it an exact Student t test. Provide raw and Holm-adjusted slope p-values within each LR model and covariance version, excluding the intercept. Categorical effects retain named reference levels. Reduced-model p-values are exploratory after selection, not valid selection-adjusted evidence. No regression coefficient is a causal effect.

For GBM: specify the otherwise missing Shapley regression as OLS `observed_default = intercept + sum_j theta_j * phi_ij + error` on H, for each fixed final model's active predictors. This is an auxiliary association analysis, never a new predictive candidate or calibration step. Report conventional coefficient t results and an original-group cluster-covariance version with finite-sample correction and G-1 reference degrees of freedom. The binary outcome motivates the robust sensitivity. Supply raw and Holm-adjusted p-values within each model's tested coefficient family, excluding the intercept.

The null is theta_j=0 conditional on the other contribution columns in this linear projection. It does not establish explanation correctness, causality or uniquely important variables. Detect rank deficiency and unidentifiable coefficients. Never regress the model's raw margin on its own additive SHAP contributions: that mostly restates the SHAP identity. Do not substitute a one-sample signed-mean SHAP t-test, whose cancellation can hide important effects. If there are no active predictors, report the regression's empty slope family explicitly.

This OLS choice is a defensible executable interpretation; it cannot be claimed to be the authors' uniquely intended regression.

## 8. Fairness: explicit definition and boundary

SEX=1 male, SEX=2 female. The article's country/industry comparisons become these two metadata groups. Interpret outputs as differences in effects/explanations, not a legal or causal fairness verdict.

For a black-box predictor j, let a_mj = mean_(i in group m) abs(phi_ij). Define equal-group shares q_mj = a_mj / sum_m a_mj; do not weight by group sample size. Report standard concentration G_j = sum_m sum_l abs(q_mj-q_lj)/(2*M*sum_m q_mj). For two groups the maximum is .5, so optionally show normalized concentration G_j/(1-1/M) in a separately named column. Lower means more equal magnitudes. Do not label either quantity an overall fairness percentage.

Preserve the article's raw ingredients: group mean SIGNED contributions v_mj, their sum, and the algebraic ratios v_mj/sum_m v_mj when the denominator is not near zero. Flag cancellation, negative shares and negative effects. Those diagnostic ratios are not substituted silently into the nonnegative magnitude Gini. Use abs(denominator) <= 1e-12 + 1e-10*sum_m abs(v_mj) as the signed-denominator cancellation rule, saved in configuration. If all magnitudes are zero, Gini is undefined; say the predictor has no measured contribution, not that fairness has been proved. Equal magnitudes with opposite signs must remain visible in the signed table.

Compute two-sample KS D between individual signed contributions phi_j in females and males. The operational null is equality of those two contribution distributions. Use `scipy.stats.ks_2samp(..., alternative='two-sided', method='asymp')` with finite inputs. This resolves the paper's inconsistent references to uniformity using its empirical group-comparison description. Do not test two group means against Uniform(0,1). Raw nominal p-values assume independent continuous observations; tied/clustered SHAP data do not meet that ideal. Label them as nominal, record group counts/tie fractions, and show Holm adjustment within each model's feature family. A large p-value is not proof of fairness; p-value counts cannot rank models as fairer.

For LR's article-specific coefficient view, fit diagnostic subgroup GLMs on the training records, separately by SEX, using the same final pooled-training design/scale/reference categories and each fixed full/selected feature set. These are diagnostic fits only. Report each estimable non-intercept term's signed group coefficients and the same Gini on their absolute magnitudes. Keep dummy terms individually named rather than inventing an aggregate coefficient for a categorical predictor. Opposite signs and unavailable terms must be explicit. Missing subgroup support or separation makes that term/group fit unestimable; never assign a zero coefficient by assumption.

There is only one estimated coefficient per term and group. A coefficient-level KS sampling test is not recoverable from the article's specification. Record `not_identifiable_from_article` for that named component. Do not treat unrelated coefficients as exchangeable observations or use bootstrap coefficient draws as a fictitious larger sample. To supply a comparable distribution diagnostic for LR, separately report its individual linear SHAP contribution Gini/KS on H using the preceding black-box definitions, clearly marked supporting interpretation.

For all four final predictive profiles, save 2,000 whole-original-group bootstrap intervals for female-minus-male differences in mean signed and mean absolute contributions, using the common H resampling draws. These are conditional effect-size diagnostics. They are not KS null calibration, group-label randomization, or selection-adjusted inference. Do not split a predictor group when it crosses SEX or outcome.
## 9. Reporting

Produce a short narrative and one main table with the four model profiles: number of predictors, AUC, AP, Brier, default recall/FPR, retained full-model importance and links to fairness/parameter detail. Fairness has multiple quantities: do not average them into an undeclared model score. Add full-versus-selected path plots and a small group-contribution comparison plot; keep exhaustive tables in CSV.

Explain: which predictors survived; where each path stopped; what accuracy changed; what group/explanation evidence changed; what is unestimable; which conventions depart from literal wording. Report all prespecified results even if reduction selects a single predictor, selects all predictors, or makes holdout performance worse.

A comparison of the new MLE LR with saved regularized LR is an estimator comparison. A comparison of new and old assessments on the same frozen model is a methodology comparison. Keep these distinct. Do not claim a new overall winner, novelty, superior robustness or equivalence from this run. Feature count is the paper's parsimony proxy, not measured environmental impact.

## 10. Implementation organization and outputs

Create `article_safe/` as a self-contained package inside this repository. Suggested modules: `data.py`, `models.py`, `inference.py`, `selection.py`, `explainability.py`, `fairness.py`, `report.py`, `run.py`, and focused `tests/`. Use qualified imports. Shared existing modules may be imported for pure readers/helpers, but no prior stage runner or reconstruction command may be called.

Save each run in a fresh `article_safe/runs/<run_id>/`. Write configuration and source/input hashes before fitting. Provide separate select and evaluate stages so model choices can be frozen and inspected before H outcomes are calculated. Evaluate must verify the selection/configuration hashes. Preserve prior run directories; a rerun uses a new ID or an explicit verification destination, not a silent overwrite.

Required machine-readable artifacts:

- Protocol/configuration, fidelity mapping, environment versions, source and input hashes, IDs/folds and predictor-group integrity checks.
- LR candidate-removal history; GBM importance order, complete prefix path, validation probabilities; selected subsets and stopping reasons.
- Final models with preprocessing, all H prediction rows keyed by ID, final metrics and paired comparisons.
- SHAP background IDs, encoded/original contributions, mappings, additivity checks, retained-importance tables.
- Coefficient/auxiliary-regression results with estimator, null, covariance, reference distribution and multiplicity family.
- Fairness effect tables, signed-mean audit, Gini/KS conventions, group support and conditional intervals.
- Report, figures, runtime/convergence records, output hashes, verification receipt and component status table.

Main commands should follow `python -m article_safe.run --stage select|evaluate --run-id <id>` from the project root; the README must use `.venv/Scripts/python.exe`. Record model parameters in every artifact manifest. No external messaging, commit or publication is included.

## 11. Validation and completion

Before credit execution, validate on small independent synthetic fixtures: paired DeLong versus direct placements (ties and covariance included); identical and degenerate scores; known nested-GLM likelihood calculations; removal of an entire categorical block; positive/negative forward gains, the .05 boundary, and a later recovery that cannot undo the first stop; Gini equal/unequal/zero and opposite-sign examples; correct KS object; SHAP additivity and original-group aggregation; impossible OLS SHAP identity regression excluded; sparse/rank-deficient fit statuses.

Integration checks must establish that changing H labels cannot alter selection, that no original predictor group crosses F/V/H, that metadata never becomes a predictive input, and that fresh reduced models are fitted rather than merely zeroing removed inputs. Final refits must occur before H evaluation. Reproduce the full GBM against the saved reference and check model reload predictions.

Run the isolated tests and normal project suite: `.venv/Scripts/python.exe -m unittest discover -s article_safe/tests -v` and `.venv/Scripts/python.exe -m unittest discover -s tests -v`. Verify the saved evaluation-manifest hashes before/after and protect the pre-existing uncommitted project files. Do not rerun earlier `--verify` stages. One complete new-stage run plus numerical rerun verification is appropriate; cache/re-evaluate fixed predictions for report verification without unnecessary repeated old work. Record which parts actually reproduced.

Completion means all specified executable components are implemented, checked and run; every nonidentifiable or unestimable component is visible; and the report can be traced to saved artifacts. It does not mean every ambiguity in the authors' original analysis has been recovered or every model fit is guaranteed estimable. If a new data-dependent obstacle requires a methodological change, report that concrete choice instead of silently changing this specification.

## 12. Supporting references and inspected environment

[DeLong, DeLong and Clarke-Pearson (1988)](https://scholars.duke.edu/publication/712661) supplies correlated AUC covariance. [R's GLM analysis-of-deviance documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/anova.glm.html) describes nested-model deviance comparisons and the binomial chi-square reference. [statsmodels GLM fitting](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.fit.html) distinguishes ordinary and regularized fitting and exposes inference settings. [statsmodels robust covariance](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html) documents cluster corrections and reference degrees of freedom. [SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) defines background, output and masking options. [SciPy KS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html) states the two-sample null and independent continuous-sample assumptions. These support statistical/software conventions; they do not establish the original authors' undocumented choices.

Verified installed versions: numpy 2.4.6; scipy 1.17.1; pandas 3.0.5; scikit-learn 1.9.0; shap 0.51.0; statsmodels 0.15.0; patsy 1.0.3. Current web documentation can show newer versions; implementation uses the installed/pinned interfaces. All 410 prior evaluation-manifest hash checks matched during planning. No credit execution occurred.