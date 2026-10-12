"""Figures and calculated narrative for the frozen-model explainability stage."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.baseline import FEATURES, NUMERIC, CATEGORICAL
from src.explainability_evaluation import FIGURE_DIR, REPORT, MODELS, SEED

NAMES = {"logistic": "Logistic regression", "gradient_boosting": "Gradient boosting"}
plt.rcParams.update({"svg.hashsalt": "credit-risk-explainability", "font.size": 10})


def save(name):
    plt.savefig(FIGURE_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.savefig(FIGURE_DIR / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close()


def create_figures(artifacts, shap_data, X_test, meta_test):
    for kind in MODELS:
        values, base = shap_data[kind]
        explanation = shap.Explanation(values=values, base_values=base,
            data=X_test[FEATURES].to_numpy(), feature_names=FEATURES)
        np.random.seed(SEED)
        shap.plots.beeswarm(explanation, max_display=16, show=False)
        plt.xlabel("Contribution to default log-odds")
        plt.title(f"{NAMES[kind]}: holdout contributions (log-odds)")
        save(f"{kind}_shap_beeswarm")
        shap.plots.bar(explanation, max_display=16, show=False)
        plt.xlabel("Mean absolute contribution (default log-odds)")
        plt.title(f"{NAMES[kind]}: global importance")
        save(f"{kind}_shap_bar")
        ranked = artifacts[f"global_shap_importance_{kind}"].feature.tolist()
        # Three global leaders plus two leading numeric features: at most five panels.
        selected = list(dict.fromkeys(ranked[:3] + [f for f in ranked if f in NUMERIC][:2]))
        fig, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 4.5), squeeze=False)
        for ax, feature in zip(axes.ravel(), selected):
            j = FEATURES.index(feature)
            for sex, label, color in [(1, "Male", "#4277aa"), (2, "Female", "#d67b40")]:
                mask = meta_test.SEX.to_numpy() == sex
                ax.scatter(X_test[feature].to_numpy()[mask], values[mask, j], s=5, alpha=.25, label=label, color=color, rasterized=True)
            ax.axhline(0, color="grey", linewidth=.6)
            if feature in NUMERIC:
                ax.set_xscale("symlog", linthresh=1000 if feature != "AGE" else 10)
                minimum, maximum = X_test[feature].min(), X_test[feature].max()
                ticks = [v for v in [-1000000, -100000, -10000, -1000, 0, 1000, 10000, 100000, 1000000] if minimum <= v <= maximum]
                if feature == "AGE":
                    ticks = [v for v in [20, 30, 40, 50, 60, 70, 80] if minimum <= v <= maximum]
                labels = [f"{v / 1000:g}k" if abs(v) >= 1000 else f"{v:g}" for v in ticks]
                ax.set_xticks(ticks, labels)
                ax.set_xlim(minimum, maximum)
            else:
                ax.set_xticks(sorted(X_test[feature].unique()))
            ax.set_title(feature)
            ax.set_xlabel("Source value" + (" (symlog axis)" if feature in NUMERIC else " (category code)"))
            ax.set_ylabel("Contribution (log-odds)")
        axes[0, 0].legend(markerscale=2)
        fig.suptitle(f"{NAMES[kind]}: dependence on predictor values", y=1.02)
        fig.tight_layout()
        save(f"{kind}_shap_dependence")
        local = artifacts["local_shap_contributions"].query("model == @kind")
        identifiers = sorted(local.ID.unique())
        fig, axes = plt.subplots(len(identifiers), 1, figsize=(11, 3.9 * len(identifiers)), squeeze=False)
        for ax, identifier in zip(axes.ravel(), identifiers):
            case = local[local.ID == identifier].copy()
            # Show strongest positive and strongest negative features explicitly.
            positive = case[case.shap_value > 0].nlargest(4, "shap_value")
            negative = case[case.shap_value < 0].nsmallest(4, "shap_value")
            displayed = pd.concat([negative, positive]).sort_values("shap_value")
            remaining = case.loc[~case.index.isin(displayed.index), "shap_value"].sum()
            labels = [f"{r.feature} = {r.value:g}" for r in displayed.itertuples()]
            contributions = displayed.shap_value.tolist()
            labels.append("All remaining contributions")
            contributions.append(remaining)
            ax.barh(labels, contributions, color=["#c04b43" if v > 0 else "#397eb0" for v in contributions])
            row = case.iloc[0]
            outcome = {(1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN"}[(row.actual, row.predicted)]
            ax.axvline(0, color="black", linewidth=.6)
            ax.set_title(f"ID {identifier} | {outcome} | p(default)={row.probability_default:.3f} | base={row.base_value:.3f}")
            ax.set_xlabel("Contribution to default log-odds (red increases, blue decreases)")
        fig.suptitle(f"{NAMES[kind]}: fixed-rule local examples", y=1.001)
        fig.tight_layout()
        save(f"{kind}_local_explanations")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, kind in zip(axes, MODELS):
        part = artifacts["sex_shap_comparison"].query("model == @kind").copy()
        part["pooled"] = part.mean_abs_shap_male + part.mean_abs_shap_female
        part = part.nlargest(10, "pooled").sort_values("pooled")
        positions = np.arange(len(part))
        ax.barh(positions - .18, part.importance_share_male * 100, height=.36, label="Male")
        ax.barh(positions + .18, part.importance_share_female * 100, height=.36, label="Female")
        ax.set_yticks(positions, part.feature)
        ax.set_xlabel("Share of group mean absolute contributions (%)")
        ax.set_title(NAMES[kind])
        ax.legend()
    fig.tight_layout()
    save("sex_shap_importance")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, kind in zip(axes, MODELS):
        part = artifacts["shap_vs_rge"].query("model == @kind")
        ax.scatter(part.shap_rank, part.rge_rank)
        for row in part.itertuples():
            if row.shap_rank <= 5 or row.rge_rank <= 5 or abs(row.shap_rank - row.rge_rank) >= 7:
                ax.annotate(row.feature, (row.shap_rank, row.rge_rank), fontsize=8, xytext=(3, 4), textcoords="offset points")
        ax.plot([1, 22], [1, 22], linestyle="--", color="grey")
        ax.set(xlabel="SHAP rank (1 = highest)", ylabel="SAFE RGE rank (1 = highest)", title=NAMES[kind], xlim=(0, 24), ylim=(0, 24))
    fig.tight_layout()
    save("shap_vs_rge_ranks")


def markdown(frame, digits=5):
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"
        return str(value)
    return "\n".join(["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
        ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)])


def write_report(a):
    summary = a["model_summary"].set_index("model")
    comparison = a["shap_vs_rge"]
    global_table = pd.concat([a[f"global_shap_importance_{k}"].head(5) for k in MODELS])
    top_groups = a["sex_shap_comparison"].groupby("model", sort=False).head(5)
    local = a["local_shap_contributions"].groupby(["model", "ID"], sort=False).first().reset_index()
    report = ["# Explainability Evaluation — frozen credit-risk models\n",
"## Methods and preserved protocol\n",
"The complete repaired holdout contains **5,996 records** (2,404 male, 3,592 female); training contains **24,004**. "
"All original observations and the existing split are retained. The frozen logistic and gradient-boosting pipelines, preprocessing, "
"22 predictors and reference threshold **0.50** are unchanged. ID and SEX are metadata only. Class 1 is a default-risk flag; "
"the saved sklearn decision rule assigns ties at 0.50 to class 0. No fitting, tuning, calibration, threshold selection or mitigation occurs.\n",
"### SHAP formulation\n",
"SHAP 0.51.0 uses **LinearExplainer** with an Independent masker for logistic regression and **TreeExplainer** with "
"`feature_perturbation='interventional', model_output='raw'` for boosting. Both explain default **log-odds**, not probability points. "
"One shared reference contains 100 training observations, sampled uniformly without replacement by NumPy `default_rng(42)` "
"from training rows sorted by ID. No outcome or SEX stratification is used. IDs are saved in `background_metadata.csv`. "
"The masker retains exactly these 100 rows without further subsampling. All 5,996 holdout rows are explained; none are background rows.\n",
f"The frozen preprocessor creates {int(a['shap_validation'].transformed_features.iloc[0])} numerical/one-hot columns. SHAP is calculated on those columns, then signed contributions "
"for all levels of each categorical predictor are summed before computing absolute importance. The result has 22 original "
"predictor columns; transformed contributions and the full mapping are also saved. This preserves additivity, but for boosting "
"it is **an aggregation of encoded-feature SHAP, not a separately computed original-feature coalition game**. Independent "
"masking can break one-hot constraints and correlations. Linear additivity makes the categorical sum straightforward for logistic regression. "
"Dependence plots use original values and show the top three predictors plus the two leading numeric predictors (at most five panels). "
"Numeric axes use symlog scaling, so a straight numeric log-odds effect may look curved on that display. Category codes are not treated "
"as a continuous scale; no meaning is invented for undocumented repayment codes.\n",
"The reconstruction identity is `base_value + sum(contributions) = decision_function(X)`; applying the logistic function "
"recovers default probability. Positive contributions increase risk relative to the shared reference; negative ones decrease it. "
"Averaging signed contributions is different from averaging their magnitudes.\n",
markdown(a["shap_validation"].assign(max_margin_reconstruction_error=a["shap_validation"].max_margin_reconstruction_error.map(lambda v: f"{v:.3e}"))),
"\nImplementation references: [SHAP LinearExplainer](https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html) "
"and [TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).\n",
"## Global explanation results\n", markdown(global_table[["model", "feature", "shap_rank", "mean_abs_shap", "mean_signed_shap", "importance_share"]]),
"\nImportance shares divide each feature's mean absolute contribution by the model's total; they are descriptive concentration measures, not explained variance.\n"]
    for kind in MODELS:
        part = a[f"global_shap_importance_{kind}"]
        top = part.iloc[0]
        report.append(f"**{NAMES[kind]}:** {top.feature} leads with mean absolute SHAP {top.mean_abs_shap:.5f} log-odds "
            f"({100 * top.importance_share:.2f}% of total); top-three share is {100 * summary.loc[kind, 'top3_shap_share']:.2f}% "
            f"and top-five share {100 * summary.loc[kind, 'top5_shap_share']:.2f}%.\n")
        report.extend([f"![{kind} beeswarm](figures/explainability_evaluation/{kind}_shap_beeswarm.png)\n",
                       f"![{kind} bar](figures/explainability_evaluation/{kind}_shap_bar.png)\n",
                       f"![{kind} dependence](figures/explainability_evaluation/{kind}_shap_dependence.png)\n"])
    report.extend(["## Local explanations\n",
        "For each model and each TP/TN/FP/FN cell, select the **lowest original ID**, then explain the union with both models. "
        "This fixed rule uses outcome cells for illustration, without choosing explanation shapes or confidence levels. It is not "
        "a representative sample or evidence of model superiority. Selection reasons and eligible counts are saved separately. "
        "Each figure displays up to four strongest positive and four strongest negative contributions plus the signed remainder; "
        "all 22 contributions and source values are saved. Both models explain the same union of IDs.\n",
        markdown(a["local_case_selection"]), "\n", markdown(local[["model", "ID", "SEX", "actual", "predicted", "probability_default", "base_value"]])])
    for kind in MODELS:
        report.append(f"\n![{kind} local explanations](figures/explainability_evaluation/{kind}_local_explanations.png)\n")
    report.extend(["## SEX-group explanation patterns\n",
        "Male = SEX 1; female = SEX 2. The same background and explainer are used for both groups; group comparisons are post hoc "
        "stratifications of the same holdout explanations. Ranked below are the five largest absolute **importance-share gaps** per model. "
        "All 22 features, raw magnitude gaps, signed means and group rankings are in the machine-readable table. A share difference of "
        "0.01 is one percentage point. No materiality threshold or significance claim is imposed.\n",
        markdown(top_groups[["model", "feature", "mean_abs_shap_male", "mean_abs_shap_female", "importance_share_female_minus_male", "mean_signed_shap_female_minus_male", "shap_rank_male", "shap_rank_female"]]), "\n",
        markdown(a["model_summary"][["model", "sex_rank_spearman", "sex_importance_share_total_variation"]]),
        "\nTotal variation is half the sum of absolute gaps between normalized importance shares. It summarizes magnitude allocation, "
        "not fairness or harm. Raw log-odds magnitudes also reflect model score scale; normalization supports a complementary comparison.\n",
        "![Group importance](figures/explainability_evaluation/sex_shap_importance.png)\n"])
    for kind in MODELS:
        row = top_groups.query("model == @kind").iloc[0]
        report.append(f"For {NAMES[kind]}, the largest share gap is **{row.feature}**, female minus male "
            f"{100 * row.importance_share_female_minus_male:+.3f} percentage points. Its mean signed contribution differs by "
            f"{row.mean_signed_shap_female_minus_male:+.5f} log-odds.\n")
    report.append("These differences describe model behavior combined with group feature distributions. They do not identify a causal "
        "SEX effect or establish discrimination. Read them alongside the completed fairness audit, which reports different model orderings "
        "for ranking parity and threshold-specific error gaps. Practical materiality remains undecided.\n")
    report.extend(["## Actual SAFE Rank Graduation Explainability\n",
        "The unmodified installed **safeaipackage 0.8.3** function "
        "`safeaipackage.check_explainability.compute_rge_values(xtrain, xtest, yhat, model, variables, group=False)` "
        "is called for all 22 original predictors. Here `yhat` is the frozen model's full-holdout default probability, **not the outcome**. "
        "For each feature the package replaces its holdout values by a training reference, recomputes probabilities through the unchanged "
        "pipeline, and returns **1 − core.rga(original_probabilities, replaced_probabilities)**. RGA compares the ordering after replacement "
        "against the original scores, with a score-magnitude-weighted graduation construction and package tie handling. It is not Spearman "
        "correlation, SHAP or a drop in outcome ROC-AUC. Larger RGE means greater change in the model's score ordering under this replacement.\n",
        "The package uses training means for numeric columns and training modes for pandas categorical columns (first sorted mode on ties). "
        "The eight existing categorical predictors are explicitly represented as pandas Categoricals using the fitted encoder's categories. "
        "This communicates the original preprocessing semantics to SAFE without changing any values or fitted transformers. Full training "
        "and holdout probabilities are checked for exact equality before and after dtype conversion. No SEX column is passed. Training "
        "reference values are saved. Passing raw integer category codes instead would make SAFE average them and could invoke unknown "
        "one-hot categories; that inappropriate alternative is not used.\n",
        "With `group=True`, each member of a logical group is replaced in turn before one joint RGE is returned. The groups partition "
        "the 22 predictors: six repayment statuses, six bill amounts, six payment amounts, and LIMIT_BAL/AGE/EDUCATION/MARRIAGE. "
        "Grouped values need not equal sums of individual values. Computation is deterministic; no permutation seed or refitting is involved.\n",
        markdown(a["safe_rge_values"].groupby("model", sort=False).head(7)), "\n", markdown(a["safe_grouped_rge"]),
        "\nThe installed explainability module, core and utility sources were inspected and hashed; package provenance is in the run manifest. "
        "The earlier SAFE adapter documents an inspected upstream commit for core/fairness utilities. This stage makes no claim that "
        "the new explainability module was verified byte-for-byte against that GitHub commit; the remote file could not be retrieved. "
        "The installed module is the authentic package implementation, not a local formula replacement. "
        "[SAFE repository and function documentation](https://github.com/GolnooshBabaei/safeaipackage).\n",
        "## SHAP versus SAFE RGE\n",
        markdown(a["model_summary"][["model", "shap_rge_spearman"]]),
        "\nSpearman correlations use all 22 feature ranks with average ranks for ties; no p-values or inferential claims are made. "
        "SHAP averages absolute additive log-odds contributions relative to a sampled background. SAFE RGE examines score-order disruption "
        "when a feature is fixed to one full-training mean/mode. They use different reference operations and output concepts; RGE is "
        "unsigned and cannot supply local contribution direction. Neither ranking is a causal effect.\n"])
    for kind in MODELS:
        part = comparison.query("model == @kind").copy()
        shap_top = part.nsmallest(5, "shap_rank").feature.tolist()
        rge_top = part.nsmallest(5, "rge_rank").feature.tolist()
        shared = [f for f in shap_top if f in rge_top]
        part["absolute_rank_gap"] = part.rge_rank_minus_shap_rank.abs()
        report.append(f"**{NAMES[kind]}:** top-five overlap {len(shared)}/5 ({', '.join(shared)}). "
            f"RGE top five: {', '.join(rge_top)}. Largest rank disagreements:\n")
        report.append(markdown(part.nlargest(3, "absolute_rank_gap")[["feature", "shap_rank", "rge_rank", "mean_abs_shap", "RGE"]]))
    report.extend(["\n![Rank comparison](figures/explainability_evaluation/shap_vs_rge_ranks.png)\n",
        "Replacement effects can be small despite large additive contributions if ordering is preserved or correlated variables retain "
        "similar information. Conversely, moving everyone to one particular training reference can change ordering more than mean "
        "absolute background-centered attribution suggests. Encoded-feature aggregation is another source of disagreement for boosting.\n",
        "## Model comparison and effect direction\n",
        markdown(a["model_summary"][["model", "top1_shap_share", "top3_shap_share", "top5_shap_share", "total_mean_abs_shap"]])])
    concentration = max(MODELS, key=lambda k: summary.loc[k, "top3_shap_share"])
    larger_group = max(MODELS, key=lambda k: summary.loc[k, "sex_importance_share_total_variation"])
    report.append(f"\nOn the top-three normalized SHAP-share measure, **{NAMES[concentration]} is more concentrated**. "
        f"On total variation of SEX-group importance shares, **{NAMES[larger_group]} has the larger group difference**. "
        "These are descriptive choices of summary, not an overall explainability or fairness winner.\n")
    leaders = [a[f"global_shap_importance_{k}"].head(5).feature.tolist() for k in MODELS]
    shared = [f for f in leaders[0] if f in leaders[1]]
    report.append(f"The models share {len(shared)}/5 leading SHAP predictors: {', '.join(shared)}. "
        "The dependence plots show higher credit limits contributing lower default log-odds in both models. "
        "For boosting, PAY_AMT1 generally contributes less risk at higher payments, with steps and context-dependent spread. "
        "These are fitted-model patterns relative to the selected background, not lending recommendations.\n")
    # Concrete category effects at fixed PAY_0 values expose different response structures.
    category = a["categorical_shap_effects"].query("SEX == 0 and feature == 'PAY_0' and value in [0, 1, 2]")
    report.extend(["For PAY_0, selected observed code-level contributions are:\n",
        markdown(category[["model", "value", "records", "mean_signed_shap", "min_shap", "max_shap"]]),
        "\nThe logistic model is additive and linear in its scaled numeric inputs, but its one-hot categorical effects are already "
        "category-specific, not linear in numeric category codes. At a fixed category its summed categorical SHAP effect is constant. "
        "Boosting can have differing contributions at the same source value because its tree context involves other features; the "
        "reported ranges and dependence panels expose that context dependence. Numeric tree effects can be stepwise and non-linear, "
        "where logistic numeric effects are straight lines in untransformed source units. These plots are descriptive, not interaction "
        "tests or causal response curves.\n",
        "## Limitations and research implications\n",
        "The analysis characterizes these two fitted models on one historically inspected holdout. It does not establish population "
        "generalization, statistical significance or practical materiality of rank or group differences. No SHAP/RGE confidence intervals "
        "are calculated. Correlated repayment, billing and payment histories make attribution reference-dependent. The 100-row training "
        "background is reproducible but its sampling uncertainty has not been explored. Holdout group composition can explain mean "
        "contribution differences without a direct SEX input. Global averages can hide individual variation and signed cancellation.\n",
        "SAFE mean/mode replacement can create unrealistic combinations, and a small RGE can coexist with substantial probability-level "
        "changes that preserve ordering. Grouped RGE is not additive. The original probabilities enter RGA as continuous reference scores, "
        "so the fairness-stage observation that binary-outcome RGA matched AUC does not apply here. SHAP uses an additive log-odds scale "
        "and aggregated dummy contributions; neither method alone establishes human comprehensibility, fairness, a causal effect or a "
        "feature-removal recommendation. Local examples are illustrations selected by a fixed ID rule. Historical Taiwanese data and "
        "binary source SEX encoding limit wider interpretation.\n",
        "For the paper, this stage adds reproducible feature rankings, score-contribution direction, local error examples, group "
        "explanation patterns, and a documented comparison of additive attribution with SAFE ranking disruption. It complements "
        "performance and fairness evidence without collapsing them into one winner.\n",
        "**Exact next research stage: Robustness evaluation**, with a separately specified SAFE RGR protocol and defensible feature "
        "perturbations. No robustness experiment has been run here. Stability/PSI/monitoring follows later; mitigation, proxy follow-up, "
        "intersectionality and governance remain future work.\n",
        "## Reproduction, inputs and artifacts\n",
        "Run from the repository root (or invoke the script by absolute path):\n\n"
        "```powershell\n.venv\\Scripts\\python.exe -m pip install -r requirements-explainability.txt\n"
        ".venv\\Scripts\\python.exe script_explainability_evaluation.py --verify\n```\n",
        "Reused inputs: `data/processed/taiwan_credit_clean.csv`, persisted split metadata and preparation manifest; "
        "`models/baseline_logistic.joblib` and `models/baseline_gradient_boosting.joblib` plus manifests; "
        "both baseline prediction CSVs; performance/fairness run manifests; fairness `group_summary.csv`; shared utilities in "
        "`src/baseline.py`, `src/performance_evaluation.py`, `src/fairness_evaluation.py`, and SAFE provenance from `src/safe_adapter.py`. "
        "Raw data, processed data, models and prior performance/fairness artifacts are protected by hashes.\n",
        "New code: `src/explainability_evaluation.py`, `src/explainability_report.py`, "
        "`script_explainability_evaluation.py`, `tests/test_explainability_evaluation.py`, and `requirements-explainability.txt`. "
        "Tables under `outputs/tables/explainability_evaluation/` contain original and transformed SHAP values, column mappings, global "
        "and SEX-specific importance, group comparisons, local examples, categorical effects, RGE and grouped RGE, cross-method/model "
        "comparisons, reconstruction checks, and prediction audit. Figures are saved as PNG/SVG.\n",
        "`run_manifest.json` records settings, environment versions, inspected SAFE source hashes, protected artifact hashes, "
        "source hashes and output hashes. `--verify` executes two complete runs, requires byte-identical CSVs (including RGE), "
        "and runs the complete test suite. Its results and test log are stored in `verification.json` and `test_results.txt`. "
        "Read numerical CSVs using `float_precision='round_trip'` for exact probability comparisons.\n"])
    REPORT.write_text("\n".join(report), encoding="utf-8")
