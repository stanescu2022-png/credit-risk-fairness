# A simpler research direction

10 October 2026. Independent reassessment requested by Andrei; proposal only. No replacement research question, model change, pooling rule, threshold or experiment has been adopted or executed.

The project currently combines three jobs: reproduce the article, compare candidate models, and invent a superior assessment framework. They need not all succeed for the research to be useful. The unpenalized LR obstacle belongs to the first job. It need not block the second. The .027 AUC difference is a full-versus-reduced GBM result, not a comparison of two complete framework recommendations.

Recommended question: When does additional model complexity provide enough practical benefit to justify it in credit-risk prediction?

Separate two meanings of simplicity: how many inputs a model needs, and how complicated its decision rule is. A three-input GBM remains a tree ensemble. A full categorical LR is structurally transparent but has many coefficients; transparency does not establish ease of human use or fairness. Input count is a data-requirement proxy, not measured collection cost.

A compact eventual comparison crosses these two dimensions:

| Candidate | All 22 inputs | Article-selected 3 inputs |
| --- | --- | --- |
| Regularized LR | Existing | Proposed new fit |
| GBM | Existing | Existing |

The three inputs are PAY_0, LIMIT_BAL and PAY_AMT1. They were selected for GBM on development data; the new LR would test this published-procedure-derived information set, not claim to be the best three-input LR. Keeping preprocessing conventions and LR regularization settings consistent across its two variants isolates the input-set change within that family. The two-by-two comparison describes these specific configured pipelines, not optimized model families or a causal effect of complexity. All frameworks assessing a candidate use the same fitted artifact and records.

Begin even smaller: use the existing full LR and GBM out-of-fold predictions to compare practical trade-offs at common operating requirements. For example, at the same target default recall, compare false-default flags, then inspect group gaps. The saved descriptive training-only candidates already show FPR about 11.0% for LR and 10.2% for GBM at 50% recall. These candidate thresholds were located on those same OOF records, so this is exploratory evidence, not independent validation or permission to replace the historical .50 decisions. Any selected policy requires an agreed training-only validation procedure and separate evaluation. Keep the original .50 results as the historical reference. Report uncertainty on paired differences and achieved constraints, not only separate model intervals.

This yields a useful result even when the models are close: quantify what is gained by nonlinear modeling or extra inputs, and which practical preferences would justify that gain. Do not equate nonsignificance with equivalence. An acceptable-loss margin needs justification independent of the favorable observed gap; without one, report the range of tolerances under which the choice changes. Fairness is a specified additional constraint or trade-off, never an automatic property of LR. Existing synthetic robustness results can add context; a new missing-input scenario is not a prerequisite.

SAFE retains a useful role as the assessment baseline and article-based feature-selection reference. A secondary question is whether its indicators support the practical choices suggested by the outcome comparison. The original article has no complete overall cross-family ranking rule; do not invent one and attribute it to the authors. The later integrated paper already examines aggregation, weights and perturbation-dependent ranking, so those alone are not a new contribution.

Two alternative directions remain available. The cheapest is an assessment study of whether SAFE profiles translate into different model choices under declared operating requirements, using existing models. A more methodological but larger study compares significance-based feature stopping with selecting the smallest candidate within an independently justified acceptable-loss margin, over repeated development samples and additional data. Neither requires a predetermined winning method.

The recommendation revises the preceding note's proposed order: defer unpenalized-LR/pooling repair and new fault experiments while deciding whether the narrower model-comparison question is preferable. Regularized LR is a valid candidate for prediction-based comparison; it does not become a faithful substitute for the original article's ordinary likelihood tests. Preserve the article reproduction and its explicit limitations.

This is an established research area, not a novelty claim: Rudin (2019), https://arxiv.org/abs/1811.10154, distinguishes inherent interpretability from explanations of black boxes; Rudin et al. (2024), https://proceedings.mlr.press/v235/rudin24a.html, discuss multiple similarly accurate models and preferences. Our current candidates have not been demonstrated equivalent or characterized as a full Rashomon set. A publishable general method would require a focused literature review and broader validation; the extensively inspected Taiwan dataset can support an exploratory case study now.

Evidence reviewed: PROJECT_CONTEXT.md, current README/Git status, saved performance and robustness reports, article v1 report, checkpoint-4 verification receipt, performance manifest, and local integrated-SAFE extracted text. No research stages or tests were rerun for this planning review.
