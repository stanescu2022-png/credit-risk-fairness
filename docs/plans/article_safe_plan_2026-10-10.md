# Implementing the original SAFE article

Prepared 10 October 2026. This is the recommended specification for review before Sol 6.1 starts coding. No new model has been fitted and no article implementation has been run.

We will use the article to answer our original question: what do we gain, and what do we give up, when we use gradient boosting instead of logistic regression? The immediate result will be a working assessment of both model families under the article's four SAFE dimensions. We can decide which experiments are worth adding after we see those results.

I recommend keeping LR and GBM as the main comparison. The article discusses both model classes, although its credit example uses random forest. This makes our work an implementation of its methodology on our dataset, with an explicit model substitution. Adding random forest can be a later check; it is not necessary to turn this first step into another three-model project.

We will create a separate `article_safe/` folder in the same project. It will contain its own code, fitted models, tables and report. It will read the existing cleaned data and saved partition. Our current implementation remains the reference against which we can understand differences.

We will first fit full models and apply the article's reduction procedures. For GBM, we order predictors by their Shapley importance, add them one at a time, and stop when the next addition fails to improve AUC significantly. For LR, we start with all predictors and remove the least necessary predictor using likelihood-ratio tests. Categorical predictors are added or removed as complete groups.

There is one consequential change for LR: the article's likelihood tests require a separate, unpenalized logistic model. Our saved LR is regularized. We will keep that saved model as a reference and clearly label the new one as article LR. This prevents differences caused by the fitting method from being mistaken for differences caused by the SAFE assessment.

Predictor selection will take place inside the existing training partition. Once the subsets are fixed, we will fit the full and reduced models on all training records and evaluate them on the saved holdout. The output will therefore contain four main profiles: full LR, reduced LR, full GBM and reduced GBM. The holdout has been studied before, so this is an exploratory comparison.

Each profile will show predictive accuracy, how many predictors remain, what drives the predictions, and how the relevant effects differ between men and women. The article does not give an overall scoring rule, so the report will describe these trade-offs and its actual selection outcomes.

Some details need our interpretation. We will use mean absolute Shapley contributions to measure importance, report Gini as concentration with lower values meaning more equal magnitudes, and define the KS comparison explicitly. The article's explanation-regression test also needs a stated response variable. These choices are specified in the technical document. Where a named test cannot be recovered coherently from the article, the report will say so rather than manufacture a result.

Sol's job will be to implement this specification, test the statistics and data separation, run the assessment, and produce a short results report. The first discussion after that should be concrete: which predictors did each method retain, how did performance change, what happened to the group comparisons, and what evidence still prevents us from preferring one model?

Read the [technical specification](article_safe_specification_2026-10-10.md) only when you want the exact conventions. The [Sol handoff](article_safe_sol_handoff_2026-10-10.md) is ready to use once we accept those choices. The earlier worst-scenario scoring proposal remains a possible later experiment; it does not govern this implementation.