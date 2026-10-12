# Our proposed SAFE/KAIRI comparison

Rewritten 10 October 2026 for a simpler reading flow. This is a proposal for discussion; the experiments have not started.

We want to find out whether considering information failures helps us choose a better credit-risk model. The starting idea is that a variable can look unnecessary when all information is available, yet become useful when another variable fails. A smaller model may therefore perform well in normal conditions but lose more predictive accuracy when an important input becomes unreliable.

We would test this by creating several versions of the same model, each using a different number of predictors. We would first measure their performance with normal data. Then we would make the latest repayment-status information unreliable for a specified share of records and measure their performance again. This tells us whether keeping extra information actually helps. If it does not, we reconsider the idea before building a larger methodology.

Our proposed selection rule is straightforward: among models that meet our agreed requirements, choose the one with the lowest prediction error in its worst tested scenario. Those scenarios include normal data. We would use Brier loss, which measures how close predicted default probabilities are to the observed outcomes. Lower loss is better. When models perform about equally well, we would prefer the smaller one.

The score expresses that result relative to a simple reference that gives every borrower the default rate estimated from fitting data:

Score = 100 x (1 - worst-scenario Brier loss / reference Brier loss).

A score of 20 means the model's worst-scenario loss is 20% lower than the reference loss on the same evaluation records. Zero means it offers no improvement; a negative score means it performs worse. Because we measure prediction error, a model receives no benefit merely for keeping its predictions unchanged during a failure.

Before looking at the results, we need to agree how much normal-performance loss is acceptable, what improvement would be large enough to matter, and which decision-error or group-error limits are essential. These requirements remain separate from the score. We would report whether the evidence supports each requirement, shows a breach, or leaves the answer uncertain.

To test whether our methodology helps, we would compare the model returned by the original article's feature-selection procedure with the choices made by the later integrated SAFE score and our procedure, using a common candidate pool. The original article does not define an overall ranking across all four SAFE dimensions; its feature-selection outcome is the specific decision being compared. We would also include a simple rule that chooses the lowest worst-scenario Brier loss without the additional safeguards. Each method would receive the same resources; a matched comparison would give SAFE the same information about the failures.

Then we would freeze those choices and evaluate them on separate records with new realizations of the failures. The comparison is the worst-scenario loss of the model selected by SAFE minus that of the model selected by our method. A positive difference favors our selection. To count as success, the gain must exceed the minimum we agreed in advance, be supported by the uncertainty analysis, and retain acceptable normal performance and decision errors.

We would keep fitting, selection and evaluation separate within the existing training data. The saved models and original split would remain intact. This first experiment would be exploratory; a stronger claim would need confirmation beyond data we have already studied.

The first step is therefore a small experiment: does apparently redundant information provide useful backup, and does evaluating that backup change which model we choose? If all methods choose the same model, there is no selection advantage. If the simple worst-loss rule matches our full procedure, we should credit the benefit to evaluating failures and retain the simpler approach unless the safeguards demonstrate additional value.

The later SAFE papers already contain probability-sensitive metrics and composite scores. The value of our proposal must come from a demonstrated improvement in model selection for this defined problem. The score gives us a clear way to describe that outcome.

The [worked example](safe_scoring_worked_example_2026-10-09.md) illustrates the arithmetic. Implementation choices, precise comparator definitions and statistical details remain in the [technical protocol](safe_comparison_technical_protocol_2026-10-09.md). The [original main plan](safe_comparison_plan_2026-10-09_original.md) is preserved for reference.