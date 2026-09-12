"""Static figures and research reporting for the integrated fairness stage."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.baseline import ROOT
from src.fairness_evaluation import TABLE_DIR, FIGURE_DIR, MODELS, REPETITIONS
from src.performance_report import markdown_table

NAMES = {"logistic": "Logistic regression", "gradient_boosting": "Gradient boosting"}
COLORS = {"male": "#2463a0", "female": "#bc5b29"}


def make_plots(a):
    def save(fig, name):
        fig.savefig(FIGURE_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
        fig.savefig(FIGURE_DIR / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
        plt.close(fig)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    groups, intervals = a["group_metrics_threshold_050"], a["fairness_confidence_intervals"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, metric, title in zip(axes, ["recall", "fpr", "selection_rate"], ["Recall / TPR", "False-positive rate", "Predicted-default rate"]):
        for offset, label in [(-.16, "male"), (.16, "female")]:
            values, lower, upper = [], [], []
            for kind in MODELS:
                row = intervals.query("model == @kind and scope == @label and metric == @metric").iloc[0]
                values.append(row.estimate)
                lower.append(row.estimate - row.ci_lower)
                upper.append(row.ci_upper - row.estimate)
            ax.bar(np.arange(2) + offset, values, width=.3, yerr=[lower, upper], capsize=4, color=COLORS[label], label=label)
        ax.set(xticks=np.arange(2), xticklabels=["Logistic", "Boosting"], ylabel="Proportion", title=title, ylim=(0, None))
        ax.legend()
    fig.suptitle("Locked holdout at 0.50: SEX metrics with marginal 95% group-bootstrap intervals")
    save(fig, "group_metrics_threshold_050")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, metric, title in zip(axes, ["recall_difference", "fpr_difference", "selection_rate_difference"], ["TPR gap", "FPR gap", "Predicted-default-rate gap"]):
        for kind, color in zip(MODELS, ["#2463a0", "#bc5b29"]):
            part = a["threshold_fairness_sensitivity"].query("model == @kind")
            ax.plot(part.threshold, part[metric] * 100, "o-", color=color, label=NAMES[kind])
        ax.axhline(0, color="#777777", linestyle=":")
        ax.axvline(.5, color="#777777", linestyle="--")
        ax.set(xlabel="Threshold", ylabel="Female minus male (percentage points)", title=title)
        ax.legend(fontsize=8)
    fig.suptitle("Exploratory holdout sensitivity — no threshold selected; dashed line = 0.50")
    save(fig, "fairness_gaps_vs_threshold")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for offset, label in [(-.16, "male"), (.16, "female")]:
        safe_table = a["safe_rga_results"]
        values = [safe_table.loc[(safe_table.model == kind) & (safe_table.group == label), "RGA"].iloc[0] for kind in MODELS]
        axes[0].bar(np.arange(2) + offset, values, width=.3, color=COLORS[label], label=label)
    axes[0].set(xticks=[0, 1], xticklabels=["Logistic", "Boosting"], ylim=(0, 1), ylabel="Package RGA", title="Subgroup ranking accuracy")
    axes[0].legend()
    parity = a["safe_rga_parity"].set_index("model").loc[list(MODELS)]
    axes[1].bar([0, 1], parity.package_parity, color=["#2463a0", "#bc5b29"])
    axes[1].set(xticks=[0, 1], xticklabels=["Logistic", "Boosting"], ylabel="Absolute RGA gap", title="SAFE package parity: max RGA − min RGA")
    for x, val in enumerate(parity.package_parity):
        axes[1].annotate(f"{val:.4f}", (x, val), xytext=(0, 4), textcoords="offset points", ha="center")
    axes[1].margins(y=.2)
    fig.suptitle("Actual safeaipackage 0.8.3 — fixed probabilities, no classification threshold")
    save(fig, "safe_rga_parity")


def write_report(a, provenance):
    groups = a["group_metrics_threshold_050"]
    gap_ci = a["fairness_confidence_intervals"]
    display_gaps = gap_ci[(gap_ci.scope == "female_vs_male") & gap_ci.metric.isin([
        "selection_rate_difference", "recall_difference", "fpr_difference", "precision_difference",
        "fnr_difference", "specificity_difference", "safe_rga_difference"])]
    display_gaps = display_gaps[["model", "metric", "estimate", "ci_lower", "ci_upper"]]
    sensitivity = a["threshold_fairness_sensitivity"]
    sensitivity_summary = []
    for kind in MODELS:
        for metric in ["recall_difference", "fpr_difference", "selection_rate_difference"]:
            values = sensitivity.query("model == @kind")[metric]
            sensitivity_summary.append({"model": kind, "gap_female_minus_male": metric,
                "minimum": values.min(), "maximum": values.max(), "sign_changes_on_grid": bool(values.min() < 0 < values.max())})
    comparisons = []
    table = a["conventional_vs_safe_comparison"].set_index("model")
    for metric in ["recall_difference", "fpr_difference", "selection_rate_difference", "SAFE_absolute_parity_gap"]:
        lr, gb = abs(table.loc["logistic", metric]), abs(table.loc["gradient_boosting", metric])
        bigger = "gradient boosting" if gb > lr else "logistic regression" if lr > gb else "neither model"
        comparisons.append(f"For {metric}, the absolute sample gap is {lr:.5f} for logistic and {gb:.5f} for boosting; {bigger} has the larger value.")
    associations = a["proxy_descriptive_associations"]
    numeric = associations.query('feature_type == "numeric"').sort_values("association", ascending=False).head(5)
    categorical = associations.query('feature_type == "categorical"').sort_values("association", ascending=False).head(5)
    numeric = numeric[["feature", "association", "signed_standardized_mean_difference_female_minus_male", "male_mean", "female_mean", "male_median", "female_median"]]
    categorical = categorical[["feature", "association"]]
    safe = a["safe_rga_results"]
    same = np.allclose(safe.RGA, safe["ROC-AUC"], rtol=0, atol=1e-12)
    report = f'''# Fairness Evaluation — fixed credit-risk baselines

## Scope and conventions

Both stabilized fitted pipelines, all 22 predictors, and the repaired 24,004/5,996
partition are unchanged. No model training, tuning, calibration or mitigation took
place. Primary threshold = **0.50**, with class 1 when probability > 0.50 and ties
assigned class 0, as in the existing baselines. SEX=1 means male, SEX=2 female;
SEX and ID never enter either pipeline's predictor matrix.

Here, a positive prediction means **flagged as likely to default**, an adverse-risk
flag rather than a favorable credit approval. 'Selection rate' means predicted-default
rate only. Which decisions/harms matter requires an explicit lending-policy context.
All signed gaps are **female minus male**. Ratios are labeled explicitly: female/male
for selection-rate ratio, min/max for demographic-parity ratio. Demographic-parity
difference is the conventional unsigned max-minus-min rate; the signed version is
selection_rate_difference. No legal threshold or 80% rule is used.

## 1. Protected-group composition

{markdown_table(a['group_summary'])}

These are observed sample default rates, not inherent characteristics of either
group. Base-rate differences can affect precision and selection rates even when
other error-rate measures are similar.

## 2. Group metrics at 0.50

Rows of each confusion matrix are actual [0,1], columns predicted [0,1]:
`[[TN, FP], [FN, TP]]`. All counts and rates are in `group_metrics_threshold_050.csv`.

{markdown_table(groups[['model','group','records','defaults','predicted_positives','TN','FP','FN','TP']])}

{markdown_table(groups[['model','group','selection_rate','accuracy','balanced_accuracy','precision','recall','specificity','fpr','fnr','npv','f1']])}

Undefined ratios or rates have missing values, rather than invented zero values.
At this sample size the reported primary point estimates are defined.

## 3. Demographic parity, equal opportunity and equalized odds

Demographic parity compares predicted-default rates. Equal opportunity is assessed
for the dataset's positive label (default detection) by comparing recall/TPR.
Equalized odds compares both TPR and FPR. The maximum of their two absolute gaps
is also reported, but both components remain visible.
[Conventional definitions](https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html).

{markdown_table(a['conventional_vs_safe_comparison'])}

{' '.join(comparisons)}

Closer to zero means closer observed parity for the particular difference metric,
and closer to one means closer parity for the specified ratios. Neither numerical
comparison establishes overall fairness or discrimination. No materiality tolerance
has been specified, and no between-model significance test has been conducted.

Important signed differences and marginal 95% intervals (proportions, not percentages):

{markdown_table(display_gaps)}

The CSV includes CIs for every group-specific rate, conventional gap/ratio, group
RGA and signed/absolute RGA gaps. Use these intervals to describe uncertainty;
small nonzero sample differences alone do not establish a population disparity.
Where a signed interval includes zero, it is compatible with either direction at
this precision. An interval excluding zero still does not establish causation or
practical significance, especially with multiple exploratory metrics.

For these fixed baselines, the selection-rate, TPR, FPR and signed RGA-gap intervals
all include zero. This audit therefore does not resolve their population directions
at the reported marginal interval precision. It is not evidence of equivalence or
proof that practically important disparities are absent.

## 4. Group-aware uncertainty

**{REPETITIONS:,} replicates, NumPy default_rng(42)**. Resample the number of distinct
holdout exact-predictor groups with replacement, including every record in each
sampled group at its group multiplicity. Groups are defined on the same 22 predictors,
excluding ID, SEX and target. A group containing both SEX codes remains intact:
resampling is not done independently by SEX. Both models use the same resamples.
Use integer row weights for confusion metrics and explicitly expanded sampled
records with the actual package `rga` for SAFE bootstrap metrics.

Intervals are 2.5th/97.5th percentiles, linear interpolation. Undefined estimates
are excluded metric by metric; valid/undefined replicate counts are saved.
These are marginal, unadjusted intervals, conditional on the fixed models and
observed groups. They omit retraining, model-selection and future-shift uncertainty.
No model-difference test, multiplicity correction or causal-discrimination test is
performed. Absolute-gap intervals are nonnegative by construction and must not be
interpreted as evidence of disparity merely because their lower endpoint exceeds
zero; use signed-gap intervals for direction and uncertainty.

![Group metrics and uncertainty](figures/fairness_evaluation/group_metrics_threshold_050.png)

## 5. Exploratory holdout threshold sensitivity

This stage explicitly authorizes **descriptive** sensitivity at 0.10–0.70 in 0.05
steps. This extends the earlier stage's fixed-threshold holdout reporting, not its
prohibition on holdout-based development. No threshold is selected, recommended
as optimal, or applied to model artifacts. The 0.50 reference remains fixed.

{markdown_table(pd.DataFrame(sensitivity_summary))}

These ranges describe changes in gaps across the common grid; sign changes indicate
that even the direction can depend on threshold. The CSV additionally reports overall
recall/precision/positive rate and both groups' TPR, FPR and selection rates at every
grid value. Magnitudes and model ordering can change without changes to the underlying
scores. These exploratory curves have no simultaneous confidence bands, and must
not be mined for a favorable threshold or used for a new operating choice.

On this grid, TPR-gap direction changes for both models. Logistic's FPR gap also
changes sign, while boosting's remains negative (lower female FPR). Logistic's
selection-rate gap remains negative, while boosting's crosses zero. The apparent
direction is therefore not stable across all thresholds and metrics.

![Threshold sensitivity](figures/fairness_evaluation/fairness_gaps_vs_threshold.png)

## 6. Actual SAFE RGA and parity

{markdown_table(safe[['model','group','records','RGA','ROC-AUC','Average Precision (AP)']])}

{markdown_table(a['safe_rga_parity'][['model','male_RGA','female_RGA','female_minus_male','package_parity']])}

Results use **safeaipackage {provenance['version']}**, without locally replacing its
RGA formula or parity implementation. The package describes RGA as normalized
agreement between outcome and predicted-score rankings, with higher values better;
its rank-graduation framework also addresses nonbinary outcomes. Here, numerical
agreement of returned RGA with ROC-AUC within 1e-12 is **{same}**. This is recorded
as a finding of this binary-outcome implementation, not an assumption that the
metrics are interchangeable for other tasks. AP remains a distinct metric.

SAFE parity is the package's **max(group RGA) − min(group RGA)**, so zero is exact
sample parity and the return value is unsigned. Its source returns a sentence,
which is preserved verbatim in `safe_rga_parity.csv`; only its numeric value is
parsed for tables. Signed female-minus-male group RGA is reported separately.

The adapter supplies SAFE a dataframe containing 22 predictors plus SEX. On every
prediction it selects only the saved pipeline's feature names, in their original
order, rejecting unexpected columns. It advertises sklearn classifier tags and
delegates predict/predict_proba to the frozen pipeline; fit is prohibited. Full
and subgroup predictions reproduce the original probabilities. SAFE enumerates
groups using training SEX value counts, resets input indices, and recomputes subgroup
probabilities; supplied yhat is validated but not used for subgroup scoring. Inputs
are aligned positionally before the call. No privileged/unprivileged label is assigned:
the source simply computes max-minus-min regardless of enumeration order.

Inspected GitHub commit: `{provenance['inspected_github_commit']}`. Installed core
matches exactly; fairness/utilities differ only in whitespace/newline. The installed
distribution does not record a Git commit, so the inspected commit is not presented
as installation provenance. Version and actual source SHA-256 hashes are in the run
manifest. The upstream example recodes its own gender column to 0/1; our audit retains
the dataset's 1/2 coding, which the function supports. No package patch was needed.
The distribution declares no dependencies although fairness/core utilities import
CatBoost and XGBoost; these and their missing pinned dependency closure are included
in `requirements-fairness.txt`, leaving the previous requirements lock unchanged.

[SAFE repository and framework](https://github.com/GolnooshBabaei/safeaipackage),
[inspected parity source](https://github.com/GolnooshBabaei/safeaipackage/blob/{provenance['inspected_github_commit']}/safeaipackage/check_fairness.py),
[inspected core](https://github.com/GolnooshBabaei/safeaipackage/blob/{provenance['inspected_github_commit']}/safeaipackage/core.py).

![SAFE results](figures/fairness_evaluation/safe_rga_parity.png)

## 7. Conventional versus SAFE interpretation

The table in section 3 keeps framework results side by side rather than forcing
agreement. SAFE evaluates disparity in **within-group ranking accuracy across the
scores**, independent of a classification threshold. It does not measure parity
of the fraction flagged, missed defaults or false alarms at a particular decision
boundary. Conventional TPR/FPR/selection-rate metrics capture those decision outcomes,
but do not describe the quality of the entire ranking.

Gradient boosting retains its previously established overall ranking advantage.
Whether it has a larger fairness gap depends on the chosen metric and threshold;
the absolute gaps listed above are the relevant descriptive evidence. A ranking
advantage is not proof of a fairness advantage. Disagreement in gap direction or
model ordering between frameworks is a substantive result, not automatically an
implementation error. A claim of a meaningful difference additionally needs an
agreed practical tolerance and appropriate comparative uncertainty analysis.

The substantive disagreement here is that boosting has a **smaller SAFE ranking
parity gap but larger TPR/FPR absolute gaps at 0.50**. Its selection-rate gap is
smaller. Logistic has lower female subgroup RGA but higher female TPR at 0.50;
for boosting, female subgroup RGA and TPR are both higher. These are point-estimate
comparisons with uncertainty, not proof that either model is overall fairer.

## 8. Unawareness and descriptive predictor associations

Excluding SEX prevents its direct use but **does not prove fairness**. Predictor
associations can remain. Descriptive analysis uses only the existing training data,
with no feature selection, removal, proxy classifier or retraining.

Largest numeric associations (ranked within numeric features only):

{markdown_table(numeric)}

Largest categorical associations (ranked within categorical features only):

{markdown_table(categorical)}

Numeric effect size = (female mean − male mean) / sqrt((male sample variance +
female sample variance)/2), with its absolute value used for ordering. Means,
medians and quartiles are provided for skewed financial distributions. Categorical
effect size is uncorrected Cramer's V = sqrt(chi-square / (N × min(r−1,c−1))),
without continuity correction; category counts/fractions are saved separately.
Numeric SMD and Cramer's V are different scales and are not pooled into one ranking.
These are descriptive effect sizes, with no p-value screening or arbitrary cutoff
used to label a predictor a proxy. Associations may reflect sample composition and
do not establish prohibited proxy use, a causal pathway, or unlawful discrimination.
Full causal/proxy investigation would require a separately agreed methodology.

## 9. Intersectionality readiness and next decisions

SEX × age bands, SEX × education and SEX × marriage could support a later study.
Before computing intersectional fairness, inspect each cell's total/default/non-default
counts and effective predictor-group counts, define handling of small cells and
multiple comparisons, and agree age bands in advance. No intersectional fairness
experiment is performed here.

Limitations include one already-inspected holdout, fixed-model conditional intervals,
exact-group dependence handling rather than complete independence guarantees,
multiple descriptive metrics, threshold-grid exploration, binary source SEX coding,
and historical Taiwanese sample representativeness. There is no defined materiality
tolerance, operational approval rule, utility/cost function or causal identification.

Before the next stage, agree which decision harms and fairness concepts matter,
what gap is practically material, and how to handle potentially competing criteria.
Retain 0.50 unless a separately agreed **training-only** protocol chooses otherwise;
these holdout sensitivity results cannot choose it. Decide whether a focused proxy
or intersectional follow-up is needed and approve its design first. Both models remain.
No mitigation, SHAP/RGE, RGR/robustness, PSI or governance monitoring has begun.

## Reproduction and files

Install the extension with `python -m pip install -r requirements-fairness.txt`, then
run `python script_fairness_evaluation.py`. Read probability CSVs with
`float_precision="round_trip"`. Notebook `06_fairness_evaluation.ipynb` verifies
and displays saved results without fitting models. Run `python -m unittest discover
-s tests -v` for focused checks.

Created: `src/safe_adapter.py`, `src/fairness_evaluation.py`, `src/fairness_report.py`,
`script_fairness_evaluation.py`, `requirements-fairness.txt`,
`tests/test_fairness_evaluation.py`, `notebooks/06_fairness_evaluation.ipynb`, this report,
and the dedicated fairness output directories. README is updated. Prior model,
split and Performance Evaluation artifacts are preserved and hashed.

Tables: group_summary, group_metrics_threshold_050, fairness_gaps_threshold_050,
fairness_confidence_intervals, threshold_fairness_sensitivity, safe_rga_results,
safe_rga_parity, conventional_vs_safe_comparison, proxy_descriptive_associations,
proxy_category_distributions, audit_predictions and bootstrap_replicates (CSV).
The run manifest records source/package hashes, settings and output hashes;
verification.json records the completion checks. Figures are available as PNG/SVG.
'''
    (ROOT / "outputs/fairness_evaluation_report.md").write_text(report, encoding="utf-8")
