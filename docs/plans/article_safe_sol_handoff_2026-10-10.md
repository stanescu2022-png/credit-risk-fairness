# Handoff to Sol 6.1: implement the original SAFE article

Prepared 10 October 2026. This is a prompt for the user to send after reviewing the methodological choices; its presence on disk does not authorize a run by itself.

Implement and run the separate article-based SAFE assessment described in `docs/plans/article_safe_specification_2026-10-10.md`. Read `AGENTS.md` and `PROJECT_CONTEXT.md` first. The short explanation is `docs/plans/article_safe_plan_2026-10-10.md`. The original source is `docs/context/sources/3. 1-s2.0-S0957417423017220-main.pdf`; verify the specified hash and use the technical specification's source mapping.

Create a separate `article_safe/` package and run directory inside the existing project. Use the existing cleaned data, 22 original predictors, fixed partition and project `.venv`. Preserve the current models, saved results and uncommitted work. Keep LR and GBM as the primary families. Fit a separate unpenalized article LR for likelihood-based procedures and retain the saved regularized LR only as a clearly labeled reference.

Implement the exact F/V selection, LR backward likelihood path, GBM forward SHAP/DeLong path, final refits and heldout assessment described in the specification. Supply the four SAFE profiles for full/selected LR and full/selected GBM, with the named paired comparisons and explicit interpretation labels. The original article has no overall SAFE score: do not invent one. Do not implement the deferred predictor-failure experiment or broader framework extensions.

Respect the specified Gini direction and magnitude convention, signed-effect audit, KS target and inference limits, and outcome-on-SHAP auxiliary regression. Do not create a coefficient-level KS test from two coefficient estimates; mark the unrecoverable article component as specified. Separation, rank deficiency and unestimable tests must be visible, not patched with an undeclared change of estimator or data.

First implement and verify the statistics on synthetic fixtures and test that H cannot influence selection. Then execute the new stage, save all selection/provenance/results artifacts, run the isolated and existing project tests, and verify prior artifact hashes. Use a short narrative report with supporting tables, not an opaque overall winner. Update project context with executed results and remaining limitations.

Routine coding choices are yours. The specification's statistical decisions are fixed for this run. If a concrete obstacle makes one of them impossible, preserve valid work and present the narrow methodological choice required; do not redesign the study silently. Finish by stating the actual selected subsets, principal model trade-offs, validation outcome and any missing/unestimable components.