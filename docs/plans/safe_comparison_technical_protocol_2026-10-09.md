# SAFE/KAIRI comparison: proposed technical protocol

Date: 9 October 2026. Read [the main plan](safe_comparison_plan_2026-10-09.md) first. This is an implementation specification for discussion, not executed research. All policy values labelled undecided must be fixed before new candidate results are inspected. Preserve the frozen baseline, 22-predictor source schema, original train/holdout assignment, strict >0.50 classification and existing dependency locks.

## A. Claim, task and scope

Primary task: select a parsimonious, predictively adequate binary default-probability model from a fixed candidate pool under a prespecified input-information failure. Optional audit task: determine whether the selected model meets separately declared requirements. These tasks have different evaluation endpoints.

Primary comparator: the original 2024 article's black-box forward variable selection, adapted to the project's model/data conventions. Primary comparison within GBM avoids inventing an article-level rule for ranking unrelated families. LR is a secondary arm. Using RF later is a fidelity sensitivity, requiring authorization for that separate new model arm. Neither frozen fitted model may be reused on an internal evaluation partition whose rows it saw during fitting.

The first loss target is Brier, not lending profit or approval utility. Outcome labels stay fixed because we simulate corrupted information about the same records. A result is recorded-outcome sensitivity, not a causal effect of borrower behavior. There are no supplied data on operational fault frequency, actual underwriting policy or FN/FP costs.

Do not claim that sustainability means carbon savings because feature count decreases, or that feature reliance/SHAP concentration establishes human explanation quality. The broader deferred framework remains a research ambition; version 1 implements only components needed for the chosen task.

## B. A common candidate pool

### Pilot pool

For each newly fitted full model, order the 22 original predictors by mean absolute grouped SHAP from fitting data only, on the same log-odds convention as the project. Use a deterministic 100-record fitting-only background. Aggregate one-hot contributions before taking absolute values. Never split category levels across subsets. Disclose that grouped encoded SHAP is not a separate original-feature coalition calculation for GBM.

Pilot subset sizes: 1,2,4,6,8,12,16,22. For each subset, construct fresh preprocessing using only its numeric/categorical predictors and fit the original model-family hyperparameters. Do not call the current build_model/save_pipeline functions in a way that overwrites baseline artifacts or requires absent columns. A new extension builder takes an explicit subset; source model conventions remain unchanged.

### Article-comparison pool

Expand to every subset size k=1,...,22 before reproducing the consecutive-variable stopping rule. Full-fitting-model SHAP order remains fixed for the path. Each candidate is fitted once on the fitting partition; all assessment methods share its exact predictions. This compares selection rules over a common path rather than different candidate-search algorithms. Path-learning changes are a later ablation, not mixed into the first comparison.

Original-predictor equality groups are always defined on all 22 cleaned inputs, using the saved duplicate_group_id. Reduced-feature collisions do not trigger method-specific regrouping, deletion or split changes. Keep all records, including conflicting outcomes. Persist feature order, rank values, subset definitions, fitted preprocessor categories, transformed widths and training IDs for every candidate.

The constant training-prevalence predictor and a fixed input-independent random-score vector are diagnostic controls. The latter must remain attached to record IDs and not be redrawn for each input fault; it is a benchmark, not a proposed credit model. Constant-original RGR is undefined under current package normalization; preserve that rather than assigning it perfect robustness.

## C. Data roles and leakage boundary

Recommended first internal split within the existing training partition:
- F: approximately 60% of 24,004, fitting and feature ranking.
- V: approximately 20%, model scoring and selection.
- A: approximately 20%, audit of fixed selected candidates.
- H: the unchanged 5,996 historical holdout, optional descriptive replication only after freeze.

Use the existing StratifiedGroupKFold convention through a new configurable development-split helper: reserve one of five folds for A, then one of four folds of the remaining records for V, leaving approximately 60% for F. Record the two seeds and chosen folds; stratification is approximate and groups never split. Do not promise exact counts in advance. Save a new internal assignment manifest without touching split_metadata.csv. Only F may estimate transformations, means/modes, prevalence, supports, feature order and fitted candidate parameters. V may supply outcome-dependent comparison metrics. A must not alter the selection or policy. H is not inspected during the pilot.

The pilot can examine V freely, but after such examination freeze the resulting protocol before using A. If A leads to redesign, A becomes development evidence; do not present a revised method's evaluation on A as fresh confirmation. Design revisions need a new evaluation source. The whole Taiwan dataset has prior project exposure, so even this separation is prospective procedural protection within an exploratory study, not recovered historical blindness.

Do not refit selected models on F+V after selection while treating audit statements about F-trained models as transferable. Version 1 keeps those candidates fixed through A/H. A later refit requires new evaluation. Repeated internal splits/nested group-aware CV may test procedure stability later; shared training samples mean fold/seed results are not independent new studies. Report split variability descriptively, and use full-procedure independent simulation replications to study selection variability.

General rationale: optimizing a finite-sample selection criterion can overfit it, so selection must be part of the evaluated pipeline. [Cawley and Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html).

## D. Fault generation

Define fault functions on the complete original raw predictor row first, then subset those corrupted rows for each candidate. This gives methods the same faulty information; a candidate that does not use PAY_0 is legitimately unaffected, rather than receiving a smaller independently sampled attack.

### Primary mechanism: latest-status unavailability with transparent replacement

On a common uniformly selected record mask, replace PAY_0 with the categorical mode estimated in F, breaking ties by ascending numeric category code. No outcome or SEX may influence the primary mask. Describe this as the specified mode-handling policy under unavailable/corrupted information, not a realistic observed outage mechanism. Do not use 0/-1/-2 as invented missing-value semantics or pass undocumented sentinels through OneHotEncoder.

V levels: q=.10,.20, plus clean q=0. A matched tests: same levels, new random draws. A transfer tests: q=.15,.30 and stale-status replacement. q=1 is an extreme diagnostic; it does not enter the primary score unless agreed independently before results. Counts are floor(q*N) selected records. Preserve selected and changed masks separately: mode/stale replacement can be a no-op. Report all-record loss as primary and selected/changed-record results as labelled supplements.

### Transfer mechanism: stale latest status

Replace PAY_0 with that row's PAY_2 on the same selected mask. The dataset documentation identifies these as September/August repayment status. Record replacement support and resulting preprocessing behavior. If an F-fitted PAY_0 encoder has not seen a substituted level, its configured unknown-category behavior is part of this defined mechanism and must be audited. Do not silently clip or map it to a favorable category. This tests a different error, not an equal-severity version of mode replacement.

### Scope controls

- All repayment statuses unavailable together: logical information-family fault, not proof of a shared production provider. A benefit dependent on older status information may vanish; disclose it.
- An agreed low-relevance feature failure: benign specificity control.
- Identity q=0, no-op and existing-mode fixtures.
- Optional alternative handling policy, such as fitting-only conditional imputation, applied to every candidate equally. If it changes the result, the claim is policy-dependent.

Use 30 common realizations per stochastic condition, with stream identities independent of model, selector, subset size and severity. Different selection versus audit streams. Nested masks across q are permitted, with independent declared streams across mechanisms. Record exposure, actual changes and child-seed identities. Keep sample uncertainty separate from finite Monte Carlo spread. Thirty draws are a proposed starting budget; do not claim high precision automatically.

## E. Comparator specifications

| ID | Decision rule on V | Fidelity and purpose |
| --- | --- | --- |
| K0 | Starting from first ranked feature, add the next only while its paired AUC improvement is positive and the two-sided DeLong p-value is below .05; at first failure retain previous subset | Case-study-guided adaptation of original article; if no subsequent comparison succeeds, retain k=1; record degenerate tests. Conventional DeLong treats records as independent, so run a group-aware paired-AUC sensitivity and disclose changed inference |
| C0 | Choose lowest clean Brier; deterministic tie handling | Probability-only comparator |
| S0 | Choose highest native integrated arithmetic SAFE score | Closest score comparator; native curve/removal/tail conventions must be frozen and checked |
| S1 | Choose highest integrated score after replacing its robustness sweep with the same operational development fault family available to us | Equal-information adaptation, not the exact published protocol |
| M0 | Choose lowest max_s L_s on matched V fault conditions; no extra policy constraints or uncertainty selection | Simple minimax baseline |
| P0 | Apply clean/skill/decision constraints to V point estimates, then minimize max_s L_s and apply parsimony margin | Minimal proposed extension |
| P1 | Apply simultaneous uncertainty-based V constraints, then the same ranking/tie rule | Optional uncertainty-aware selector; evaluates whether uncertainty adds value rather than just reducing acceptance |

S0 requires resolving grid endpoints, partial-contribution accuracy construction, reversed-RGE direction, greedy masking, numeric/category replacement and averaging conventions. A later author implementation can help; do not equate our one-feature mean/mode RGE values with a delivered AURGE curve. Freeze one arithmetic primary aggregator; geometric/RMS/TOPSIS are declared sensitivities, not a menu from which to select the losing SAFE version.

The original paper does not deliver a universal scalar or a complete cross-family selection rule. K0 is an operationalized black-box sustainability comparator, not the entire KAIRI framework reduced to a single p-value. Preserve fairness/explanation evidence in companion tables and explicitly limit the comparison claim to model selection under the specified task. Interpret neither nonsignificant AUC differences nor KS results as proof of practical adequacy/fairness.

A 'guarded SAFE' sensitivity applies the same clean/skill eligibility constraints before S1 ranking. If this matches P0, the main contribution is the safeguards, not a unique score. Do not exclude this stronger adaptation merely because it erases our advantage.

Minimum implementation for an informative pilot: K0, C0, M0, P0. A claim of superiority over integrated scoring additionally requires S0/S1. P1 and guarded-SAFE are necessary before claiming value specifically from uncertainty or constraints. If S0 cannot be implemented faithfully within the week, report the narrow original-article result and leave broader score superiority unresolved.

## F. Formal score and constraints

For candidate m, scenario s, realization r and evaluation record i:

ell_i,s,r(m) = [p_i,s,r(m) - y_i]^2.
L_s(m) = (1/N) * sum_i [(1/R_s) * sum_r ell_i,s,r(m)].
R(m) = max_{s in declared score set} L_s(m).
L_ref = (1/N) * sum_i [pi_F - y_i]^2.
S(m) = 100 * [1 - R(m)/L_ref], if L_ref>0; otherwise undefined.

The mode of loss aggregation matters: max across scenario means is different from mean of per-realization maxima, and max of means differs from mean of per-record worst losses. Version 1 uses max across expected-loss estimates for a fixed finite scenario set. These definitions cannot be switched after results.

Proposed constraint forms:
- Clean adequacy: L_0(m)-L_0(full reference) <= epsilon_clean.
- Useful performance: R(m)/L_ref <= 1-rho_min.
- Decision requirement if agreed: FNR_s(m)<=tau_FNR and/or FPR_s(m)<=tau_FPR for specified scenarios, including clean.
- Group requirement if agreed: |recall_female-recall_male|<=tau_gap, or a chosen error-rate alternative. Signed reporting remains female minus male.

Choose the full reference and its selection role before results. Within the primary GBM arm it is the F-fitted 22-predictor GBM. Across families either use family-specific references with clearly limited claims, or choose one global reference on V and audit its losses too; do not swap references per favorable result.

No pass/fail thresholds are determined by existing holdout values. For discussion, a clean Brier margin grid {.001,.0025,.005,.01} and corresponding material-gain grid can illustrate sensitivity. These are research sensitivity values, not recommended lending tolerances or adopted acceptance thresholds. The user/professor must choose the primary margin and meaningful gain before new outcome tables are inspected, or leave the analysis explicitly exploratory and report the whole declared curve.

Choose a parsimony tie margin eta separately from epsilon_clean: the former controls proximity to the best robust risk, the latter limits clean sacrifice. No penalty coefficient lambda that happens to yield our preferred model. Within eta of the best robust risk, prefer fewer original predictors; remaining ties use lower clean Brier then a fixed candidate ID. Encoded width, inference time and serialized size are supplementary; original feature count alone does not compare architectural complexity.

Predetermined desirable properties: no reward from duplicating an indicator; no across-candidate min-max normalization; no need to add dummy models to set the scale; negative skill retained; identity behavior correct; a breached mandatory condition cannot be compensated by a higher score; missing/undefined evidence cannot become a pass. The raw fixed-model score is independent of what other candidates happen to be listed. Adding a candidate may legitimately change which candidate is selected.

## G. Independent evaluation and inference

Freeze selected candidate IDs and all decision rules using V. A is used for audit/evaluation of those selections, not for new model selection. Always distinguish the ranking recommendation from its adequacy status.

For selection task, compute on common A:
G_c = max_s L_s(m_c) - max_s L_s(m_proposed).

This is a difference of maxima, not max_s[L_s(m_c)-L_s(m_proposed)]. Primary reporting: absolute G_c; secondary relative reduction 100*G_c/R(m_c) only when R(m_c)>0. A common L_ref makes the selected-model skill difference exactly 100*G_c/L_ref; the skill scale does not supply independent additional evidence. Independent means fresh observations/draws and a standard prespecified outcome loss, not judging methods by self-assigned indicator totals.

Analyze matched-mechanism and transfer-mechanism evaluation sets separately. Matched tests whether selection generalizes across records/draws; transfer tests whether it survives a new fault mechanism/severity. Do not collapse them into a post hoc scenario mixture. Keep q=1 outside the primary set unless prospectively declared.

### Uncertainty with maximum statistics

An ordinary percentile bootstrap directly applied to a max/min can fail near ties. Build simultaneous bounds for the smooth per-scenario mean losses first. Version 1 can use shared whole-original-predictor-group bootstrap draws with Bonferroni-adjusted two-sided intervals for the declared vector of selected-model/scenario/reference quantities. This is an approximate conditional procedure, not automatically a finite-sample guarantee; calibration must be checked in independent simulations before using guarantee language. A cluster-sandwich/normal-bound sensitivity can be added.

If simultaneous intervals for a candidate's scenario losses are [l_s,u_s], then [max_s l_s, max_s u_s] bounds its R. If L_ref has interval [l_ref,u_ref] with l_ref>0, derive skill bounds:
S_lower = 100*[1-R_upper/l_ref].
S_upper = 100*[1-R_lower/u_ref].

Given jointly covered comparator and proposed risk intervals, a conservative G interval is:
[R_c_lower-R_p_upper, R_c_upper-R_p_lower].

This avoids assuming a unique worst scenario. Use paired per-scenario loss-difference intervals as supporting evidence; the conservative max-risk interval can be wide. Do not replace it with a favorable pointwise interval or condition retrospectively on the empirical worst scenario. If the uncertainty prevents a materiality conclusion, report that.

For a signed group-gap interval crossing zero, an absolute-gap lower bound is zero and the upper bound is the larger absolute endpoint. A narrow nonzero disparity can be materially acceptable; a wide interval including zero is not proof of fairness.

A full acceptance decision is supported only when every relevant requirement's bound clears its tolerance. At least one proven breach means violated. Otherwise it is inconclusive; unassessed dimensions remain unassessed. V-based uncertainty selection (P1) needs simultaneous coverage across every candidate and constraint inspected on V. Audit on A only needs the prespecified selected-model comparison family, because the identities were fixed on V. If many candidate A results are inspected anyway, the family must include them; never hide adaptive checking.

Use shared complete predictor-group resamples across methods, conditions and metadata subsets. No splitting a group to impose outcome/SEX strata. Repetitions are not extra clients. Version 1 conditions on fitted candidates and saved fault draws. Monte Carlo SD/SE is reported separately; a nested row/fault procedure is a later option with a new explicitly unconditional target. Current tail quantile/greedy-mask conditioning limits remain disclosed for SAFE curves.

Acceptance-task comparison must report acceptance coverage and unresolved rates. Neither reject-all nor inconclusive-all is superiority. Compare error/coverage frontiers or assign a fallback loss only if the fallback action and its cost have been independently agreed. Do not manufacture ground-truth 'unsafe' labels by thresholding noisy A estimates. Known-truth simulations are the best route for false-acceptance calibration.

## H. What counts as success

A limited empirical superiority claim requires: positive G beyond agreed delta_material with a supporting interval; clean noninferiority under the agreed margin; acceptable declared decision/fairness requirements; no advantage manufactured by unequal candidates, extra tuning or withheld fault information; and persistence under new audit realizations. State whether transfer to the new fault mechanism also succeeded.

A stronger methodological claim requires incremental value beyond M0/guarded-SAFE, calibrated uncertainty, a meaningful complexity/performance trade-off, and independent confirmation. If only fault knowledge helps, call the result a justified fault-aware extension. If constraints mainly improve acceptance reliability, that is the separate claim, assessed at useful coverage. If the procedure chooses the same model as SAFE, better presentation alone is not selection superiority.

A numerical gain on a researcher-designed scenario does not establish bank profitability, discrimination, general security or universal SAFE inferiority. A theoretical property proof and a synthetic mechanism can explain why a method behaves differently, but cannot replace realistic empirical benefit.

## I. Planned implementation artifacts and checks

Proposed new module group (all under a new comparison namespace): protocol/partitions; subset-model factory; fault generator; comparator adapters; score/constraint logic; paired inference; report. New outputs live in a dedicated comparison directory, models in models/comparison/<run-id>, and protocol/hash manifests protect current artifacts. Use the existing .venv and pinned libraries; any genuinely new dependency gets a separate reviewed extension, not an upgrade of old locks.

Persist: policy and claim version; internal split IDs; candidate feature paths; fitting-only replacement/preprocessing rules; method-choice table; prediction cache by ID/candidate/scenario/realization; exposure/no-op table; metric vector; uncertainty bands; verdict explanations; resource timing; prior protected hashes; source identities and versions.

Tests to implement later: no F/V/A group overlap; no fitting/audit IDs used for the wrong role; original order/class index; strict .50 ties; same faults before subsetting; original predictor categories grouped; q=0/no-op identity; constant/undefined score behavior; conditional interval transformation and difference-of-max semantics; deterministic repeat; protected hashes. Do not build tests merely to duplicate formulas; prioritize leakage, scenario equality and numerical edge cases.

Pseudo-workflow:
1. Load locked training records and verify hashes.
2. Materialize internal F/V/A groups without touching original metadata.
3. Fit feature rankings/candidates on F; persist identities.
4. Generate V fault inputs once; compute shared predictions/metrics.
5. Each comparator emits candidate ID or unresolved; freeze the choices.
6. Generate independently seeded A matched/transfer faults; evaluate choices.
7. Build simultaneous bounds and report G, clean trade-offs, verdict/coverage.
8. Stop at go/no-go; no automatic escalation to a large indicator battery.

## J. Work sequence before 16 October

10 October: merge the user's ideas, decide task/score/gates, resolve comparator conventions and authorize the narrow implementation.
11-12 October: implement isolation, common candidates/faults and controls; pilot on F/V only; freeze the remaining audit protocol if premise is informative.
13-14 October: fixed-choice A evaluation, paired inference, equal-information baselines and ablations; allow a negative result.
15 October: synthesize one clear result, limitations and next decision; use H only if protocol is frozen and label it historical/descriptive.
16 October: present question, faithful comparators, score meaning, actual choice/outcome and honest go/no-go. This is a target sequence, not a runtime estimate or promise of positive evidence.

## K. Prior-art checks from this planning session

These checks limit novelty claims; they do not validate every implementation or reference in these papers.

- The [SAFE uncertainty preprint](https://arxiv.org/html/2609.29588v1) already covers joint covariance, bootstrap-based composite uncertainty and three-way decisions. It also warns about min-type ties. Accordingly, covariance/intervals/abstention are not our proposed novelty. Its code repository README mentions a multi-dataset extension, but those detailed files were not successfully retrieved or independently audited here.
- [Robust machine learning models: linear and nonlinear](https://doi.org/10.1007/s41060-024-00512-1), Giudici/Raffinetti/Riani (online 2024, journal 2025), already joins parsimony, robustness and explainability through Forward Search. Our contribution must be specific to selection under the declared information failure, not the idea of robust simplification itself.
- [Nightmare at test time](https://people.csail.mit.edu/gamir/pubs/fdrop_camera_postfix.pdf), Globerson/Roweis (2006), establishes minimax feature-deletion learning. We propose an assessment/selection experiment with fixed learning recipes, not invention of feature-deletion robustness.
- [Distributionally Robust Feature Selection](https://papers.neurips.cc/paper_files/paper/2025/hash/5b3f001200198bb4ecaac0a1ea89dd99-Abstract-Conference.html), Swaroop/Krishnamurti/Wilder (2025), already selects limited features for performance across populations. Abstract/proceedings evidence establishes neighboring prior art; its full algorithm has not been independently reproduced here.
- [RAISE](https://arxiv.org/html/2510.18559v1), Nguyen/Do (2025), already aggregates responsibility dimensions and emphasizes a multidimensional profile with predictive performance reported separately. Merely normalizing and averaging extra conventional metrics is insufficient originality.
- [Learn then Test](https://arxiv.org/html/2110.01052v5), Angelopoulos et al., establishes risk-threshold testing with multiplicity control. Its iid assumptions cannot be transferred without analysis to our grouped observational benchmark; version 1 uses explicitly approximate conditional uncertainty.
- [DomainBed](https://arxiv.org/abs/2007.01434), Gulrajani/Lopez-Paz (2020), supports the need for explicit selection protocols and strong simple baselines when comparing generalization methods. We are not claiming natural domain generalization from synthetic Taiwanese corruption.

The five supplied papers and original KAIRI source remain the main full-text foundation: see [the review](../reports/safe_full_text_review_2026-10-09.md) and [the original-article comparison](../reports/article_methodology_comparison_2026-10-09.md). Broad novelty remains unresolved. No code from external documents was executed.