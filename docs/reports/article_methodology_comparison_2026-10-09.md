# Implemented project versus Giudici, Centurelli & Turchetta (2024)

Date: 9 October 2026. Scope: the current saved implementation versus **Artificial Intelligence risk measurement**, Expert Systems With Applications 235, 121220, DOI 10.1016/j.eswa.2023.121220. This compares the article-level methodology, not only safeaipackage. No research experiments or model changes were performed for this review.

## Main conclusion

Our project is an adaptation of the SAFE/KAIRI perspective with supplementary evaluation, rather than a complete replication or a demonstrated improvement over the article. We are more explicit about probability calibration, operating-point group errors, controlled input corruption and conditional sampling uncertainty. The article goes further in model parsimony/selection and formal statistical comparisons, which we have not implemented.

The current results do not establish that our methodology is superior to the article. The next research question should concern a precisely defined assessment task and a material benefit measured against the relevant article components.

## Sources and evidence boundary

- Article: `docs/context/sources/3. 1-s2.0-S0957417423017220-main.pdf`. Sections 2.1-2.4 (pp. 2-3), Fig. 1 (p. 4), credit-scoring case in §4.1 (pp. 4-5), Table 2 (p. 5), other applications in §§4.2-4.4 (pp. 5-8), conclusions (pp. 8-9).
- Implementation: `outputs/performance_evaluation_report.md`, `outputs/fairness_evaluation_report.md`, `outputs/explainability_evaluation_report.md`, `outputs/robustness_evaluation_report.md`, their saved tables/manifests, and `outputs/stabilization_report.md`.
- Recovered article text supports searches; the methodological figure, Table 2 and relevant surrounding pages were visually inspected in the PDF.
- The broader proposal from the colleague is a separate document. Its proposed methods are not counted as implemented here.
- A method absent from this article is an addition relative to this article, not automatically a novel contribution to the wider literature.

## 1. What the article actually proposes

SAFE means Sustainability, Accuracy, Fairness and Explainability. The article presents an integrated set of risk indicators, with associated statistical tests. Its main methodological figure is not a table of RGA/RGE/RGR outputs.

| Article dimension | Article procedure | Statistical assessment |
|---|---|---|
| Accuracy | RMSE for continuous outcomes, AUROC for classification | Diebold-Mariano for continuous prediction comparisons, DeLong for AUC comparisons |
| Sustainability | Reduce model complexity through variable/model selection while controlling predictive deterioration | Likelihood-based F/chi-square tests for the models considered; Shapley-ordered accuracy comparisons with DM/DeLong for black-box models |
| Explainability | Parameters for white-box models, Shapley contributions for black-box models; contribution retained by a selected parsimonious model | t-tests for coefficients or a regression using Shapley contributions, as described in the paper |
| Fairness | Concentration across groups of coefficients or mean Shapley contributions, summarized through a Gini-based measure | KS comparisons as described in the theory and case studies |

The article also discusses thresholds for turning default probabilities into decisions. Its AML and IT applications use precision and recall; AML explicitly reports F1. Other applications compare group AUC values. Therefore a comparator consisting only of AUC or only of the package's ranking parity would omit parts of this article.

The article discusses a future integrated trustworthiness measure, but does not specify a single operational weighted total score with a fully defined decision rule. Its four indicators remain linked assessments, not a delivered universal 0-100 score.

## 2. Data and empirical setting

| Aspect | Article credit-scoring application | Current project |
|---|---|---|
| Population | More than 100,000 SMEs scored by a fintech company | 30,000 UCI Taiwan credit-card records |
| Predictors / horizon | Balance-sheet information from 2020, status at end of 2021 | 22 retained predictors, next-month default target |
| Models | Random forest, with a logistic comparator and reduced models | Frozen sklearn LR and GradientBoostingClassifier |
| Split | 70% training / 30% testing | Repaired 24,004 / 5,996 split, no exact predictor group crossing |
| Fairness groups | Country, including France/Germany/Italy/Spain, and industry | Source SEX metadata, male/female |
| Wider applications | Credit scoring, AML, IT failure monitoring, image anomaly detection | One credit benchmark |

The article has broader application coverage; our experiment is more concentrated on one public benchmark. Its reported AUC of approximately 0.97 versus 0.89 is not comparable with our 0.778 versus 0.757 as evidence of framework superiority: models, targets, inputs and populations differ. A fair methodology comparison needs common cases and a common assessment objective.

The paper's credit example alternates between descriptions of eight and ten variables. Treat that as a reporting detail to resolve if exact replication is attempted. We do not have its SME data or execution code in this repository; the paper states data availability on request.

## 3. Accuracy and probability quality

**Shared or overlapping:** AUROC comparison of simpler and more complex models. Precision, recall and F1 are already used elsewhere in the article. Binary classification and operating thresholds are part of the paper's motivation.

**Implemented here, additional to the article's reported assessment:** sklearn average precision, Brier score, log loss, 10-bin ECE, reliability curves and calibration summaries; saved confusion matrices and detailed threshold diagnostics; training-only group-aware five-fold CV and OOF predictions; fixed-model group-bootstrap intervals.

**Missing relative to the article:** DeLong comparison of AUCs; formal parsimony comparisons. We also have no DM execution, although the paper proposes it primarily for continuous-target prediction comparisons. Our conditional bootstrap intervals are not the article's named tests.

**Consequence:** we assess probability quality explicitly, while the paper's binary accuracy indicator emphasizes ranking. That is a plausible complementary assessment dimension. Merely adding known probability scores does not prove novelty or materially better decisions. Our larger AUC/AP estimates for GBM are descriptive comparisons; there is no performance-stage paired hypothesis test.

Evidence: performance report, §§training CV, probability calibration, locked-holdout evaluation; `calibration_summary.csv`, `calibration_bins.csv`, `holdout_metrics.csv`, `holdout_confidence_intervals.csv`.

## 4. Sustainability and robustness

**Article:** sustainability is principally operationalized through selecting a more parsimonious model with acceptable predictive performance. For black-box models, it orders variables by Shapley importance and compares successively built models using accuracy tests. The credit case selects four important variables; its explainability discussion reports a high retained contribution. The cheese example demonstrates a simpler image model avoiding a false alert caused by an upside-down photo.

**Current project:** retains both full, fixed 22-predictor pipelines. It tests prediction sensitivity using authentic package tail swaps and supplementary semantic corruptions. It measures RGR, probability movement, Spearman, threshold flips and changes in AUC/AP/Brier, with shared perturbations and conditional paired intervals for two primary joint scenarios. It documents selection/exposure/no-ops, group results, ties and a row-order diagnostic. There are 167 scenarios and 1,501 saved realizations.

**Missing:** no Shapley-guided feature-selection path, reduced-model fitting, DeLong stopping rule, likelihood-based model-reduction test, or direct test that a parsimonious model is more robust. The existing top-three SHAP shares do not constitute feature selection.

**Consequence:** this is a substantially different empirical procedure, not completion of the paper's sustainability module. We add direct sensitivity measurements, but the article already treats robustness and parsimony as central and provides an illustrative failure/correction case. We cannot claim it ignores robustness.

Evidence: robustness report, frozen protocol, authentic SAFE, primary findings, exposure, paired uncertainty, semantic assumptions, and the saved `protocol.json`.

## 5. Explainability

**Shared:** local Shapley decomposition, global importance and contribution concentration. The paper already proposes percentage Shapley contribution retained by selected predictors. The paper supports explainability for both model classes, including parameter interpretation for regression.

**Current implementation:** LinearExplainer for LR and interventional TreeExplainer for GBM; shared seeded 100-row training background; all holdout records explained on log-odds; 85 transformed columns aggregated back to 22 predictors. We provide global plots, dependence patterns, deterministic TP/TN/FP/FN illustrations, group summaries, authentic package RGE and descriptive SHAP/RGE rank agreement.

**Important differences:** our top-three 54.2% / 40.9% measure is mean-absolute SHAP concentration for full models. The article ties its retained-contribution indicator to a selected parsimonious model. Our display is related but not the same completed procedure. The article describes aggregating Shapley values without all the implementation detail needed to establish an exact equivalence to our mean-absolute log-odds convention. For boosting, our encoded-column aggregation is not a separately computed original-feature coalition game.

**Missing:** coefficient significance tests, Shapley-regression t-tests, and testing whether feature contributions retained by a reduced model support the article's claims. No explanation uncertainty intervals or quality tests such as AOPC were run.

**Consequence:** global/local SHAP and concentration are overlapping methodology, not a new contribution relative to this article. Local-case selection, provenance and RGE comparison add explicit operational detail and complementary diagnostics.

Evidence: explainability report, methods, global/local explanations, SHAP versus SAFE RGE, and model summary.

## 6. Fairness

**Article:** compare coefficients or Shapley contributions across groups, calculate a Gini-based concentration indicator, and apply KS tests. Its credit Table 2 compares country-conditioned feature explanations; other cases inspect gender, hosting type or age. It also uses group AUC and, in IT monitoring, contrasts observed error distributions with explanation distributions.

**Current project:** group confusion matrices and rates; signed female-minus-male selection, recall, FPR and other gaps; explicitly labeled ratios; demographic parity, equal opportunity, equalized-odds components; authentic package RGA/ranking parity; whole-group-bootstrap intervals; descriptive input associations; group SHAP shares and ranks.

**Missing:** the Table-2 Gini/KS procedure, group coefficient fits and their Gini assessment, formal KS tests on raw/group explanations or error distributions. Our group SHAP comparisons are descriptive. Our bootstrap and SAFE package ranking parity do not reproduce Table 2.

**Consequence:** we give a more explicit operating-threshold/error-rate view and uncertainty description, while the paper gives a formal explanation-distribution testing procedure we do not have. Neither approach alone establishes causal discrimination or legal compliance. The article already considers fairness at input and output levels; cross-dimensional or group-based assessment is not wholly new.

Evidence: fairness report, conventional metrics, signed/unsigned definitions, uncertainty, SAFE parity and descriptive associations; explainability report group summaries.

## 7. Inference, reproducibility and decision integration

| Topic | Article | Current implementation |
|---|---|---|
| Statistical inference | Named tests are central to comparisons, selection and group explanation assessment | Conditional group-bootstrap intervals; paired intervals limited to two primary robustness scenarios; no corresponding DeLong/KS/t-test battery |
| Probability calibration | No explicit ECE/reliability/Brier/log-loss assessment specified in this article | Explicit descriptive calibration and probability scores, no fitted calibrator |
| Sampling / dependence | Case-specific train/test designs, including temporal testing in IT | Exact-predictor groups in split, CV and bootstrap; no temporal validation |
| Reproducibility | Published explanation and figures; data on request | Saved source/protocol/version identities, raw tables, cached predictions, repeat-run verification and tests |
| Human use | Documentation and manual review discussed in applications | Offline Explorer, presentation and technical Q&A |
| Overall score | Future integration discussed, no fully specified operational scalar here | No implemented aggregate score or validated pass/fail audit rule |
| Business criteria | Expert-set acceptance criterion in the AML case and discussion of false-alert workload | No agreed FN/FP costs, utility function, fairness tolerances or materiality criterion |
| Uncertainty decomposition / conformal | No such implemented module specified in this article | Also not implemented here; remains in the colleague's proposal |

Our engineering evidence supports traceability and repeatability, not demonstrated scientific superiority. The existing holdout was historically inspected before the lock. This affects confirmation of any hypothesis subsequently designed from these results.

## 8. Interpretation issues to resolve before treating the article as an executable baseline

These points do not establish that the framework fails. They mean a faithful baseline must state conventions rather than silently repair or reinterpret the paper.

1. **Gini direction.** Section 2.4 associates similar group shares with a Gini value near one, whereas the credit Table 2 has small values and is interpreted as fair. This is inconsistent with an unqualified standard concentration-Gini interpretation. We must resolve whether the intended quantity is Gini, one-minus-Gini or another convention.
2. **KS target and conclusion.** The theoretical discussion refers to departure from a uniform distribution, while cases refer to group distribution comparisons, and it includes wording that associates a small p-value with fairness. The precise test object/null needs to be stated. A nonsignificant difference does not prove equality or fairness.
3. **Signed Shapley aggregation.** Mean signed contributions can be negative or cancel. Dividing by a sum of group means may yield unstable/undefined quantities or fail to give nonnegative concentration shares. The paper does not fully specify all these cases.
4. **Explainability t-tests.** A reference to regressing on Shapley values does not by itself provide the complete regression specification, sampling assumptions and inference procedure required for reproduction.
5. **Selection stopping rule.** The general stepwise wording and the application description require an explicit rule for the first nonsignificant improvement, the retained model and the validation data used. Nonsignificance alone is not a demonstrated noninferiority guarantee, and sequential testing needs consideration.

These are legitimate protocol questions. They should be resolved transparently before comparing an extension to the article, ideally through the cited methods and available author implementation, rather than used as easy targets for a superiority claim.

## 9. What we can defend now

**Supported:** our study applies the SAFE/KAIRI perspective to a public credit benchmark and extends the documented evaluation with explicit probability-calibration diagnostics, decision-rate group assessment, direct input-corruption sensitivity and conditional uncertainty. It incorporates later package rank-graduation indicators as complementary evidence.

**Not supported:** full replication of the article; a fifth-axis framework already validated; a universally superior model; a methodology proven better than article-level SAFE; or a novel statistic merely because it is extra to the package.

The strongest potential next comparison is a specified probability/decision-quality failure that selected article indicators do not reveal adequately. A strictly ranking-preserving probability distortion remains a candidate, but we must compare against the full relevant article assessment, including Shapley evidence and decision metrics, not only native RGA/RGR. The distortion experiment has not been executed.

Before a SCORE is built, define the task, operational reference, comparator procedure, material effect, controls and a reporting rule. A score with arbitrary weights or thresholds would add another unvalidated design choice. For the 16 October meeting, a transparent comparator specification and one prespecified material counterexample with controls would be more defensible progress than claiming broad superiority from the current dashboard.
## Full-text literature follow-up, 9 October 2026
This document remains a comparison with the original 2024 article, not the whole subsequent SAFE literature. See [safe_full_text_review_2026-10-09.md](safe_full_text_review_2026-10-09.md) for full readings of five newer/related papers. Binary Brier-style SAFE criteria, rank-graduation inference, curve-based integrated scoring and scenario-dependent model recommendations already exist. The earlier probability-distortion candidate is retained as a control/teaching aid rather than recommended as the principal new contribution. The revised proposal is to audit assessment/score validity and then consider actual feature-reduction/data-source-fault interactions with explicit minimum decision requirements. This is proposed work, not an executed experiment or authorization to change frozen baselines.