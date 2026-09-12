"""Tables, static research figures, and a report for Performance Evaluation."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

from src.baseline import ROOT
from src.performance_evaluation import METRICS, MODELS, BOOTSTRAP_REPETITIONS

NAMES = {"logistic": "Logistic regression", "gradient_boosting": "Gradient boosting"}
COLORS = {"logistic": "#2463a0", "gradient_boosting": "#c15b22"}


def markdown_table(frame, digits=4):
    def format_value(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"
        return str(value)
    return "\n".join([
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
        *["| " + " | ".join(format_value(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)],
    ])


def make_plots(artifacts, directory):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.savefig(directory / f"{name}.png", dpi=170, bbox_inches="tight")
        fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    for partition, title in [("training_oof", "Training OOF"), ("locked_holdout", "Locked holdout — fixed models")]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
        for column, kind in enumerate(MODELS):
            table = artifacts["calibration_bins"].query("partition == @partition and model == @kind")
            nonempty = table[table["count"] > 0]
            ax = axes[0, column]
            ax.plot([0, 1], [0, 1], "--", color="#777777", label="Perfect calibration")
            ax.plot(nonempty.mean_probability, nonempty.observed_frequency, "o-", color=COLORS[kind], label=NAMES[kind])
            for row in nonempty.itertuples():
                ax.annotate(f"n={row.count}", (row.mean_probability, row.observed_frequency),
                            xytext=(3, 6), textcoords="offset points", fontsize=7)
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted default probability",
                   ylabel="Observed default frequency", title=NAMES[kind])
            ax.legend(loc="lower right", fontsize=8)
            axes[1, column].bar((table.lower + table.upper) / 2, table["count"], width=.09, color=COLORS[kind], alpha=.75)
            axes[1, column].set(xlim=(0, 1), xlabel="Probability bin", ylabel="Records")
        fig.suptitle(f"{title}: reliability in 10 equal-width bins", fontsize=14)
        save(fig, f"calibration_{partition}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, kind in zip(axes, MODELS):
        table = artifacts["threshold_tradeoffs"].query("model == @kind")
        for metric, color in [("Recall", "#2463a0"), ("Precision", "#c15b22"),
                              ("FPR", "#8a4796"), ("Predicted-positive Rate", "#33836c")]:
            ax.plot(table.threshold, table[metric], label=metric, color=color)
        ax.axvline(.50, color="#555555", linestyle="--", label="Existing 0.50")
        ax.set(xlabel="Candidate threshold (probability > threshold)", ylabel="Proportion",
               xlim=(0, 1), ylim=(0, 1), title=NAMES[kind])
        ax.legend(fontsize=8)
    fig.suptitle("Training OOF threshold trade-offs — no threshold selected", fontsize=14)
    save(fig, "oof_threshold_tradeoffs")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for kind in MODELS:
        table = artifacts["oof_predictions"].query("model == @kind")
        fpr, tpr, _ = roc_curve(table.actual, table.probability_default)
        precision, recall, _ = precision_recall_curve(table.actual, table.probability_default)
        scores = artifacts["oof_metrics"].set_index("model").loc[kind]
        axes[0].plot(fpr, tpr, color=COLORS[kind], label=f'{NAMES[kind]}: {scores["ROC-AUC"]:.4f}')
        axes[1].step(recall, precision, where="post", color=COLORS[kind], label=f'{NAMES[kind]} AP: {scores["Average Precision (AP)"]:.4f}')
    axes[0].plot([0, 1], [0, 1], "--", color="#777777")
    axes[1].axhline(table.actual.mean(), linestyle="--", color="#777777", label="Training default prevalence")
    axes[0].set(xlabel="False-positive rate", ylabel="Recall", title="ROC", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall", xlim=(0, 1), ylim=(0, 1))
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("Training OOF ranking performance", fontsize=14)
    save(fig, "oof_roc_precision_recall")


def write_report(artifacts):
    cv = artifacts["cv_fold_metrics"]
    summary = artifacts["cv_summary_metrics"]
    score_names = list(METRICS)
    cv_table = pd.DataFrame({"Metric": score_names})
    for kind in MODELS:
        indexed = summary[summary.model == kind].set_index("metric")
        cv_table[NAMES[kind] + " mean ± SD"] = [f'{indexed.loc[m, "mean"]:.4f} ± {indexed.loc[m, "std"]:.4f}' for m in score_names]
    folds = cv.pivot(index="fold", columns="model", values=["ROC-AUC", "Average Precision (AP)"])
    wins = {metric: int((folds[metric].gradient_boosting > folds[metric].logistic).sum())
            for metric in ("ROC-AUC", "Average Precision (AP)")}
    ranking_folds = pd.DataFrame({"Fold": folds.index})
    for metric in ("ROC-AUC", "Average Precision (AP)"):
        for kind in MODELS:
            ranking_folds[f"{NAMES[kind]} {metric}"] = folds[metric][kind].to_numpy()
    ci = artifacts["holdout_confidence_intervals"]
    ci_table = pd.DataFrame({"Metric": score_names})
    for kind in MODELS:
        indexed = ci[ci.model == kind].set_index("metric")
        ci_table[NAMES[kind] + " [95% CI]"] = [
            f'{indexed.loc[m, "estimate"]:.4f} [{indexed.loc[m, "ci_lower"]:.4f}, {indexed.loc[m, "ci_upper"]:.4f}]'
            for m in score_names]
    candidates = artifacts["recall_target_candidates"][["model", "recall_target", "threshold", "Recall", "Precision", "Specificity", "FPR", "Predicted-positive Rate"]].copy()
    candidates["model"] = candidates.model.map(NAMES)
    threshold_samples = artifacts["threshold_tradeoffs"]
    threshold_samples = threshold_samples[threshold_samples.threshold.round(2).isin([.5, .4, .3, .2])]
    threshold_samples = threshold_samples[["model", "threshold", "Recall", "Precision", "Specificity", "FPR", "Predicted-positive Rate", "F1"]].sort_values(["model", "threshold"], ascending=[True, False])
    holdout_counts = artifacts["holdout_metrics"][["model", "TN", "FP", "FN", "TP"]].copy()
    holdout_counts[["TN", "FP", "FN", "TP"]] = holdout_counts[["TN", "FP", "FN", "TP"]].astype(int)
    calibration = artifacts["calibration_summary"]
    cal_text = []
    for kind in MODELS:
        row = calibration.query('model == @kind and partition == "training_oof"').iloc[0]
        bins = artifacts["calibration_bins"].query('model == @kind and partition == "training_oof"')
        supported = bins[bins["count"] >= 100]
        worst = supported.loc[supported.absolute_gap.idxmax()]
        cal_text.append(
            f'{NAMES[kind]} OOF: mean probability {row.mean_probability:.4f} versus default frequency '
            f'{row.observed_frequency:.4f}; ECE {row.ECE_10_uniform_bins:.4f}. Among bins with at least 100 '
            f'records, the largest absolute gap is {worst.absolute_gap:.4f} in '
            f'[{worst.lower:.1f}, {worst.upper:.1f}), n={int(worst["count"])} '
            f'(predicted {worst.mean_probability:.4f}, observed {worst.observed_frequency:.4f}).')
    partitions = artifacts["cv_partition_summary"]
    report = f'''# Performance Evaluation

## Protocol and scope

The repaired partition is retained: **24,004 training records and 5,996 locked
holdout records**. Both original fitted preprocessing/model pipelines and their
prediction files remain unchanged, verified with SHA-256 hashes. All 22 existing
predictors and both model definitions are retained. ID and SEX are audit metadata.
The holdout was inspected during the earlier baseline stage; it is now locked
prospectively for development decisions and cannot become a historically unseen
test set retroactively.

Training-only `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
uses exact predictor groups, excluding ID, SEX and target. Each group appears in
one validation fold. Both algorithms share identical folds. Each fold fits a
fresh complete preprocessing/model pipeline on its four training folds, with
the existing hyperparameters. No model was tuned, calibrated, replaced, or saved
over the stabilized models. There is one OOF prediction per training ID per model.

## 1. Cross-validation consistency and stability

Gradient boosting has higher ROC-AUC in **{wins['ROC-AUC']}/5 folds** and higher AP in
**{wins['Average Precision (AP)']}/5 folds**. This is a descriptive within-fold
comparison, not a paired hypothesis test or significance claim.

{markdown_table(ranking_folds)}

All fold-level metrics are in `tables/performance_evaluation/cv_fold_metrics.csv`.
The following are unweighted fold means and **sample SD (ddof=1)**. SD describes
variation across five fixed folds; it is not a standard error or confidence interval.

{markdown_table(cv_table)}

Validation sizes range from {partitions.validation_records.min():,} to
{partitions.validation_records.max():,}, with default rates
{partitions.validation_default_rate.min():.4%}–{partitions.validation_default_rate.max():.4%}.
See `cv_partition_summary.csv` for group counts and zero group-overlap checks.
The SDs quantify metric stability on these folds only. Recall and precision can
vary even while the ranking advantage is consistent. Neither model is established
as superior across all performance dimensions: the fixed holdout favors different
models on different metrics, and no business objective or significance test was specified.

Pooled OOF metrics are also saved; nonlinear metrics of pooled predictions need
not equal the mean of fold metrics. OOF models train on roughly 80% of the training
partition, whereas the saved holdout models were fitted on all training records.

## 2. Training-only threshold trade-offs

The grid is 0.00 to 1.00 inclusive in steps of 0.01. Class 1 means probability
**strictly greater than** the threshold (ties are class 0), matching both baseline
estimators at 0.50. Precision and F1 are reported as zero when there are no positive
predictions. Every grid row reports positive rate, accuracy, balanced accuracy,
precision, recall, specificity, FPR, FNR and F1. These are pooled OOF diagnostics.

Selected grid rows, including the existing 0.50:

{markdown_table(threshold_samples)}

Lower thresholds increase recall, false positives and the fraction flagged for
default. Precision typically falls; its behavior is empirical rather than strictly
monotone. These tables quantify the workload and false-positive trade-off without
assuming that higher recall alone is preferable.

### Recall-target candidates — no threshold selected

For each target, sort scores of OOF positive cases and take the score of the
`ceil(target × number_of_positives)`th highest case. The candidate is the nearest
representable float below that score; this is the highest representable threshold
achieving at least the requested empirical recall. All tied scores are included,
so achieved recall can exceed the target. The table shows approximate thresholds;
use full-precision CSV values loaded with `float_precision="round_trip"` for
reproduction. Candidates are **not implemented**
and have no holdout performance estimates. Achieving an OOF target does not
guarantee that recall on future data.

{markdown_table(candidates)}

![OOF threshold trade-offs](figures/performance_evaluation/oof_threshold_tradeoffs.png)

## 3. Probability calibration assessment

{markdown_table(calibration)}

{' '.join(cal_text)}

At this ten-bin resolution, both OOF curves broadly track the diagonal, with
count-weighted absolute gaps around one percentage point. Neither has a substantial
overall mean offset. This is descriptive agreement, not a calibration guarantee.
Mean agreement alone does not imply calibration at each risk level. Reliability
curves show local deviations; sparse high-probability bins have noisy observed
frequencies. Use bin counts when interpreting them. There is no pre-specified ECE
pass/fail tolerance, so these findings do not justify declaring either model fully
calibrated or automatically choosing a correction.

Definitions: binary Brier score = mean `(probability - outcome)^2`; log loss uses
natural logs and sklearn's numerical clipping. ECE = sum over 10 equal-width bins
of `(bin_count / N) × abs(mean_probability - observed_frequency)`. Bins are
left-closed/right-open, with 1.0 included in the final bin. Empty bins remain in
the CSV with count zero and missing means; they contribute zero to ECE. ECE depends
on this binning choice. Brier and log loss evaluate overall probability quality,
including discrimination, rather than calibration alone.
[Source: sklearn calibration guide](https://scikit-learn.org/stable/modules/calibration.html).

![Training OOF calibration](figures/performance_evaluation/calibration_training_oof.png)

The separate locked-holdout calibration plot is fixed-model evaluation only.
No Platt scaling, isotonic regression, or other calibrator was fitted. If correction
is later authorized, compare candidate calibration methods using training-only
validation/nested OOF predictions and decide using a pre-specified criterion.

## 4. Locked holdout at the existing 0.50

Confusion matrices use rows = actual [0, 1], columns = predicted [0, 1]:
`[[TN, FP], [FN, TP]]`.

{markdown_table(holdout_counts)}

All fixed-threshold classification and probability/ranking scores, with 95% intervals:

{markdown_table(ci_table)}

Baseline point estimates remain unchanged; specificity, probability losses and
intervals extend their evaluation. The current-facing label is **Average Precision
(AP)** for `average_precision_score`, not trapezoidal PR-AUC. The archived earlier
artifacts retain their original labels.
[Source: sklearn AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

## 5. Confidence interval procedure

Use **{BOOTSTRAP_REPETITIONS:,} bootstrap repetitions**, NumPy `default_rng(42)`.
From the {int(ci.holdout_groups.iloc[0]):,} distinct holdout predictor groups, sample
that same number of groups with replacement. Every selected group contributes all
its records; repeated selection repeats the whole group. Implementation uses
integer row multiplicities as sample weights, equivalent to explicitly expanding
the sampled records. Sample record counts can vary because group sizes differ.
No bootstrap is stratified by label or SEX. Both models use the same resamples,
but no difference intervals, p-values, or paired model-comparison test are computed.

Intervals are the 2.5th and 97.5th percentiles (linear quantile interpolation) of
the bootstrap metric estimates. Samples missing an outcome class are skipped:
**{int(ci.single_class_samples_skipped.iloc[0])} skipped, {int(ci.valid_repetitions.iloc[0])} valid**.
All replicates are saved for inspection. These intervals condition on the fitted
models and observed holdout groups; they do not include model-development,
retraining, or future population-shift uncertainty. Overlap/non-overlap of marginal
intervals is not used as a significance test.

## 6. Limitations and decisions before fairness analysis

- One repaired split and one five-fold assignment do not characterize all possible
  sampling variability. Fold training sets overlap, so fold scores are correlated.
- Keeping groups together handles exact predictor dependence; it does not establish
  independence of all other customers or address unobserved relationships.
- Threshold candidates are described on the same OOF data used to locate them.
  They need training-only confirmation/nested validation if later selected; no
  claims about holdout performance at candidate thresholds are made.
- Calibration bin deviations and ECE are descriptive and bin-dependent. There is
  no fitted calibrator, calibration-method selection, or threshold optimization.
- The locked holdout's earlier inspection must remain disclosed in any final paper.
- Retain both the transparent logistic baseline and the nonlinear boosting model.
  Fairness, SAFE, SHAP, robustness, mitigation, PSI/stability, intersectional and
  governance work remain outside this stage.

Before fairness analysis, decide whether 0.50 remains the comparison threshold or
whether a training-only threshold-selection protocol should be agreed. Define a
business objective (costs, acceptable false-positive burden, or recall constraint),
whether models should share a threshold or an operating criterion, and how to
validate any choice. Also decide whether calibration assessment alone is sufficient
or a separately authorized training-only calibration experiment is warranted.
None of these choices has been made automatically.

## Reproduction and outputs

Run `python script_performance_evaluation.py` with the existing pinned environment.
It reads the stabilized artifacts, fits temporary CV models, and regenerates this
stage's tables and figures. It never invokes preparation, a full-data split, or
baseline model persistence. Subsequent runs verify the locked-input hashes in
`tables/performance_evaluation/run_manifest.json`. The notebook
`../notebooks/05_performance_evaluation.ipynb` documents and displays the saved
results without retraining or accessing alternate holdout thresholds.

Created implementation: `src/performance_evaluation.py`, `src/performance_report.py`,
`script_performance_evaluation.py`, `tests/test_performance_evaluation.py`, and
`notebooks/05_performance_evaluation.ipynb`. Updated: README, current baseline AP
labels in both scripts/model notebooks/current metric CSVs, and the corresponding
stabilization test. Historical reports/archives and fitted baseline artifacts remain fixed.

All new CSVs live under `outputs/tables/performance_evaluation/`:
`cv_fold_metrics`, `cv_summary_metrics`, `cv_fold_assignments`, `cv_partition_summary`,
`oof_predictions`, `oof_metrics`, `threshold_tradeoffs`, `recall_target_candidates`,
`calibration_bins`, `calibration_summary`, `holdout_predictions`, `holdout_metrics`,
`holdout_confidence_intervals`, `holdout_bootstrap_replicates`, and
`overall_performance_comparison`. `run_manifest.json` records inputs, settings,
versions, source hashes and CSV hashes. Four figures are saved as PNG and SVG under
`outputs/figures/performance_evaluation/`: OOF/holdout calibration, OOF threshold
trade-offs, and OOF ROC/precision-recall curves. This report is
`outputs/performance_evaluation_report.md`.
'''
    (ROOT / "outputs/performance_evaluation_report.md").write_text(report, encoding="utf-8")
