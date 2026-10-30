# Original swapping predictions: descriptive outcome-loss diagnostic

Specified **30 October 2026**, after the verified selection exploration and
before new historical-holdout assessment. This is explicitly post-exploration
hypothesis diagnosis. It introduces no selector, tuning choice or judge change.

Question: on the exact saved original SAFE percentile-swap predictions, how does
its preservation ordering compare with absolute probability-loss ordering?
The new exploration changed both measurement and disturbance design, so it alone
cannot distinguish those explanations for the different selected model.

Read only the verified safe2026_swapping_2026-10-26 artifacts: selection identity,
four saved rgr_probabilities matrices and robustness curves. Check receipt and
ordered IDs. Use all 23 saved raw severity nodes, including zero, with no node
selection according to findings. No predictions, fitting, data repartition,
historical-holdout reads or modifications of existing evidence.

At every candidate/node compute Brier on original probabilities and natural-log
LogLoss clipped at float64 epsilon. Save absolute loss, loss minus that
candidate's zero-severity loss, saved RGR, raw severity and normalized position.
Report all four profiles and uniformly averaged node values for description
only. Such a grid average is not measured population risk or a new model-selection
rule. Compare every fixed candidate pair at each node, recording agreement,
opposition or ties between higher-RGR preference and lower-loss preference.
Clean RGR ties remain ties. Preserve negative and inconclusive findings.

Independently check probability losses with scikit-learn's per-node scoring and
check zero-severity losses against the already verified exploration. Save sources,
specification and input hashes, results and a fresh completion receipt in a
separate run. No Monte Carlo or sampling interval is calculated in this bounded
descriptive diagnostic. Interpretation cannot establish causal attribution,
novelty, superiority, real fault frequencies or validity across other stresses.
