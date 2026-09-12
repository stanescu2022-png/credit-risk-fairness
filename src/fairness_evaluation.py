"""Fixed-model SEX fairness audit, including actual SAFE metrics; no model fits."""
from datetime import datetime, timezone
from importlib.metadata import version
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score, average_precision_score
from safeaipackage import core as safe_core

from src.baseline import ROOT, DATA_PATH, NUMERIC, CATEGORICAL, sha256, load_split
from src.performance_evaluation import (predictor_groups, bootstrap_group_weights,
    validate_fitted_baselines, locked_input_hashes, class_metrics)
from src.safe_adapter import evaluate_safe, provenance

TABLE_DIR = ROOT / "outputs/tables/fairness_evaluation"
FIGURE_DIR = ROOT / "outputs/figures/fairness_evaluation"
MODELS = ("logistic", "gradient_boosting")
LABELS = {1: "male", 2: "female"}
REPETITIONS = 2000
SEED = 42
RATE_METRICS = ["selection_rate", "accuracy", "balanced_accuracy", "precision", "recall",
                "specificity", "fpr", "fnr", "npv", "f1"]


def metrics(y, predicted, weights=None):
    y, predicted = np.asarray(y), np.asarray(predicted)
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    tn = float(w[(y == 0) & (predicted == 0)].sum())
    fp = float(w[(y == 0) & (predicted == 1)].sum())
    fn = float(w[(y == 1) & (predicted == 0)].sum())
    tp = float(w[(y == 1) & (predicted == 1)].sum())
    def divide(a, b):
        return a / b if b else np.nan
    recall, specificity = divide(tp, tp + fn), divide(tn, tn + fp)
    return {"records": tn + fp + fn + tp, "defaults": tp + fn, "non_defaults": tn + fp,
        "predicted_positives": tp + fp, "predicted_negatives": tn + fn,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "selection_rate": divide(tp + fp, w.sum()), "accuracy": divide(tp + tn, w.sum()),
        "balanced_accuracy": (recall + specificity) / 2, "precision": divide(tp, tp + fp),
        "recall": recall, "specificity": specificity, "fpr": 1 - specificity, "fnr": 1 - recall,
        "npv": divide(tn, tn + fn), "f1": divide(2 * tp, 2 * tp + fp + fn)}


def gaps(male, female):
    values = {f"{name}_difference": female[name] - male[name] for name in RATE_METRICS}
    values.update({f"{name}_absolute_gap": abs(female[name] - male[name]) for name in RATE_METRICS})
    low, high = min(male["selection_rate"], female["selection_rate"]), max(male["selection_rate"], female["selection_rate"])
    values.update({
        "selection_rate_ratio_female_to_male": female["selection_rate"] / male["selection_rate"] if male["selection_rate"] else np.nan,
        "demographic_parity_difference": abs(female["selection_rate"] - male["selection_rate"]),
        "demographic_parity_ratio_min_to_max": low / high if high else np.nan,
        "equal_opportunity_difference": female["recall"] - male["recall"],
        "equalized_odds_max_absolute_gap": max(abs(female["recall"] - male["recall"]), abs(female["fpr"] - male["fpr"]))})
    return values


def group_summary(meta):
    rows = []
    for sex, label in LABELS.items():
        part = meta[meta.SEX == sex]
        rows.append({"SEX": sex, "group": label, "records": len(part), "defaults": int(part.default.sum()),
                     "non_defaults": int((part.default == 0).sum()), "observed_default_rate": part.default.mean(),
                     "holdout_percentage": 100 * len(part) / len(meta)})
    return pd.DataFrame(rows)


def threshold_audit(predictions):
    group_rows, gap_rows, sensitivity = [], [], []
    for kind, part in predictions.groupby("model", sort=False):
        y, p, sex = part.actual.to_numpy(), part.probability_default.to_numpy(), part.SEX.to_numpy()
        for threshold in np.round(np.arange(.10, .701, .05), 2):
            predicted = p > threshold
            grouped = {code: metrics(y[sex == code], predicted[sex == code]) for code in LABELS}
            disparity = gaps(grouped[1], grouped[2])
            overall = metrics(y, predicted)
            sensitivity.append({"model": kind, "threshold": threshold, "analysis": "descriptive_holdout_sensitivity_only",
                "overall_recall": overall["recall"], "overall_precision": overall["precision"],
                "overall_selection_rate": overall["selection_rate"],
                **{f"{LABELS[code]}_{name}": grouped[code][name] for code in LABELS for name in ("recall", "fpr", "selection_rate")},
                **disparity})
            if threshold == .50:
                np.testing.assert_array_equal(predicted, part.predicted)
                group_rows.extend({"model": kind, "threshold": .50, "SEX": code, "group": LABELS[code], **grouped[code]} for code in LABELS)
                for name, value in disparity.items():
                    direction = "female minus male" if name.endswith("_difference") and name != "demographic_parity_difference" else "unsigned absolute gap"
                    if "ratio" in name:
                        direction = "female / male" if "female_to_male" in name else "minimum / maximum"
                    gap_rows.append({"model": kind, "threshold": .50, "metric": name, "value": value,
                                     "convention": direction, "parity_reference": 1 if "ratio" in name else 0})
    return pd.DataFrame(group_rows), pd.DataFrame(gap_rows), pd.DataFrame(sensitivity)


def fairness_bootstrap(predictions, repetitions=REPETITIONS, seed=SEED):
    parts = {kind: part.reset_index(drop=True) for kind, part in predictions.groupby("model", sort=False)}
    ref = next(iter(parts.values()))
    for part in parts.values():
        pd.testing.assert_frame_equal(ref[["ID", "SEX", "actual", "duplicate_group_id"]], part[["ID", "SEX", "actual", "duplicate_group_id"]])
    rng, rows = np.random.default_rng(seed), []
    for iteration in range(1, repetitions + 1):
        weights = bootstrap_group_weights(ref.duplicate_group_id, rng)
        for kind, part in parts.items():
            grouped, group_rga = {}, {}
            for sex, label in LABELS.items():
                mask = part.SEX.to_numpy() == sex
                y, p = part.actual.to_numpy()[mask], part.probability_default.to_numpy()[mask]
                grouped[sex] = metrics(y, part.predicted.to_numpy()[mask], weights[mask])
                expanded = np.repeat(np.arange(len(y)), weights[mask])
                if len(np.unique(y[expanded])) == 2:
                    group_rga[sex] = float(safe_core.rga(y[expanded].tolist(), p[expanded].tolist()))
                else:
                    group_rga[sex] = np.nan
                rows.append({"model": kind, "repetition": iteration, "scope": label,
                    "sample_records": int(weights.sum()), **{m: grouped[sex][m] for m in RATE_METRICS}, "safe_rga": group_rga[sex]})
            rows.append({"model": kind, "repetition": iteration, "scope": "female_vs_male",
                "sample_records": int(weights.sum()), **gaps(grouped[1], grouped[2]),
                "safe_rga_difference": group_rga[2] - group_rga[1],
                "safe_rga_parity_absolute_gap": abs(group_rga[2] - group_rga[1])})
        if iteration % 250 == 0:
            print(f"Whole-group fairness bootstrap, including actual SAFE RGA: {iteration}/{repetitions}", flush=True)
    return pd.DataFrame(rows)


def confidence_intervals(replicates, groups, disparities, rga, parity):
    points = []
    for row in groups.to_dict("records"):
        for metric in RATE_METRICS:
            points.append((row["model"], row["group"], metric, row[metric]))
    for row in disparities.to_dict("records"):
        points.append((row["model"], "female_vs_male", row["metric"], row["value"]))
    for row in rga.query('group != "overall"').to_dict("records"):
        points.append((row["model"], row["group"], "safe_rga", row["RGA"]))
    for row in parity.to_dict("records"):
        points.extend([(row["model"], "female_vs_male", "safe_rga_difference", row["female_minus_male"]),
                       (row["model"], "female_vs_male", "safe_rga_parity_absolute_gap", row["package_parity"])])
    results = []
    for kind, scope, metric, point in points:
        samples = replicates[(replicates.model == kind) & (replicates.scope == scope)][metric]
        valid = samples[np.isfinite(samples)]
        lower, upper = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
        results.append({"model": kind, "scope": scope, "metric": metric, "estimate": point,
            "ci_lower": lower, "ci_upper": upper, "valid_repetitions": len(valid),
            "undefined_repetitions": len(samples) - len(valid), "confidence_level": .95,
            "method": "whole-predictor-group percentile bootstrap; marginal, unadjusted"})
    return pd.DataFrame(results)


def proxy_descriptives(X_train, metadata):
    """Training descriptive effect sizes only; no causal/proxy prediction model."""
    rows, categories = [], []
    for feature in NUMERIC:
        male = X_train.loc[metadata.SEX == 1, feature]
        female = X_train.loc[metadata.SEX == 2, feature]
        scale = np.sqrt((male.var(ddof=1) + female.var(ddof=1)) / 2)
        signed = (female.mean() - male.mean()) / scale if scale else np.nan
        rows.append({"feature": feature, "feature_type": "numeric", "partition": "training",
            "association_measure": "absolute standardized mean difference", "association": abs(signed),
            "signed_standardized_mean_difference_female_minus_male": signed,
            "male_mean": male.mean(), "female_mean": female.mean(), "male_median": male.median(),
            "female_median": female.median(), "male_q25": male.quantile(.25), "female_q25": female.quantile(.25),
            "male_q75": male.quantile(.75), "female_q75": female.quantile(.75)})
    for feature in CATEGORICAL:
        table = pd.crosstab(metadata.SEX, X_train[feature]).reindex(index=[1, 2], fill_value=0)
        statistic, _, _, _ = chi2_contingency(table, correction=False)
        v = np.sqrt(statistic / table.to_numpy().sum() / min(table.shape[0] - 1, table.shape[1] - 1))
        rows.append({"feature": feature, "feature_type": "categorical", "partition": "training",
                     "association_measure": "Cramer's V (uncorrected)", "association": v})
        for sex, label in LABELS.items():
            for category in table.columns:
                categories.append({"feature": feature, "category": int(category), "SEX": sex, "group": label,
                                   "records": int(table.loc[sex, category]), "within_group_fraction": table.loc[sex, category] / table.loc[sex].sum()})
    return pd.DataFrame(rows), pd.DataFrame(categories)


def protected_hashes():
    result = locked_input_hashes()
    paths = [*ROOT.joinpath("outputs/tables/performance_evaluation").glob("*"),
             *ROOT.joinpath("outputs/figures/performance_evaluation").glob("*"),
             ROOT / "outputs/performance_evaluation_report.md", ROOT / "requirements.txt"]
    result.update({str(p.relative_to(ROOT)): sha256(p) for p in paths if p.is_file()})
    return result


def run_fairness_evaluation():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    before, safe_source = protected_hashes(), provenance()
    manifest_path = TABLE_DIR / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["protected_artifact_hashes"] != before:
        raise ValueError("A locked baseline or Performance Evaluation artifact changed.")
    previous = json.loads((ROOT / "outputs/tables/performance_evaluation/run_manifest.json").read_text())
    for category in ("locked_input_hashes", "output_hashes", "source_hashes"):
        for name, expected in previous[category].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"Prior Performance Evaluation input/output changed: {name}")
    validate_fitted_baselines()
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test, meta_train, meta_test = load_split(df)
    if len(X_test) != 5996 or set(meta_test.SEX) != {1, 2} or set(meta_train.SEX) != {1, 2}:
        raise ValueError("Locked holdout or protected-group coding changed.")
    np.testing.assert_array_equal(predictor_groups(X_test, meta_test), meta_test.duplicate_group_id)
    artifacts = {"group_summary": group_summary(meta_test)}
    print(artifacts["group_summary"].to_string(index=False), flush=True)
    prediction_rows, rga_rows, parity_rows = [], [], []
    for kind in MODELS:
        model = joblib.load(ROOT / f"models/baseline_{kind}.joblib")
        p, predicted = model.predict_proba(X_test)[:, 1], model.predict(X_test)
        saved = pd.read_csv(ROOT / f"outputs/tables/baseline_{kind}_predictions.csv", float_precision="round_trip")
        np.testing.assert_array_equal(saved.ID, meta_test.ID)
        np.testing.assert_array_equal(saved.SEX, meta_test.SEX)
        np.testing.assert_array_equal(saved.actual, y_test)
        np.testing.assert_array_equal(saved.predicted, predicted)
        np.testing.assert_array_equal(saved.probability_default, p)
        part = meta_test[["ID", "SEX", "default", "duplicate_group_id"]].rename(columns={"default": "actual"}).copy()
        part["group"], part["model"] = part.SEX.map(LABELS), kind
        part["probability_default"], part["predicted"] = p, predicted
        prediction_rows.append(part.reset_index(drop=True))
        overall, group_rga, gap, raw = evaluate_safe(model, X_train, X_test, meta_train, meta_test, y_test, p)
        for sex in (None, 1, 2):
            mask = np.ones(len(p), dtype=bool) if sex is None else meta_test.SEX.to_numpy() == sex
            rga_rows.append({"model": kind, "SEX": sex, "group": "overall" if sex is None else LABELS[sex],
                "records": int(mask.sum()), "RGA": overall if sex is None else group_rga[sex],
                "ROC-AUC": roc_auc_score(y_test.to_numpy()[mask], p[mask]),
                "Average Precision (AP)": average_precision_score(y_test.to_numpy()[mask], p[mask]),
                "implementation": "safeaipackage.core.rga", "package_version": version("safeaipackage")})
        parity_rows.append({"model": kind, "male_RGA": group_rga[1], "female_RGA": group_rga[2],
            "female_minus_male": group_rga[2] - group_rga[1], "package_parity": gap, "raw_package_return": raw,
            "implementation": "safeaipackage.check_fairness.compute_rga_parity", "package_version": version("safeaipackage")})
    predictions = pd.concat(prediction_rows, ignore_index=True)
    artifacts["audit_predictions"] = predictions
    artifacts["safe_rga_results"], artifacts["safe_rga_parity"] = pd.DataFrame(rga_rows), pd.DataFrame(parity_rows)
    artifacts["group_metrics_threshold_050"], artifacts["fairness_gaps_threshold_050"], artifacts["threshold_fairness_sensitivity"] = threshold_audit(predictions)
    reps = fairness_bootstrap(predictions)
    artifacts["bootstrap_replicates"] = reps
    artifacts["fairness_confidence_intervals"] = confidence_intervals(reps, artifacts["group_metrics_threshold_050"], artifacts["fairness_gaps_threshold_050"], artifacts["safe_rga_results"], artifacts["safe_rga_parity"])
    artifacts["proxy_descriptive_associations"], artifacts["proxy_category_distributions"] = proxy_descriptives(X_train, meta_train)
    comparison = artifacts["fairness_gaps_threshold_050"].pivot(index="model", columns="metric", values="value")
    comparison = comparison[["selection_rate_difference", "selection_rate_ratio_female_to_male", "recall_difference", "fpr_difference", "equalized_odds_max_absolute_gap"]].reset_index()
    artifacts["conventional_vs_safe_comparison"] = comparison.merge(artifacts["safe_rga_parity"][["model", "female_minus_male", "package_parity"]], on="model").rename(columns={"female_minus_male": "safe_RGA_female_minus_male", "package_parity": "SAFE_absolute_parity_gap"})
    for name, table in artifacts.items():
        table.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    from src.fairness_report import make_plots, write_report
    make_plots(artifacts)
    write_report(artifacts, safe_source)
    if protected_hashes() != before:
        raise RuntimeError("Protected artifacts changed during fairness evaluation.")
    manifest = {"completed_utc": datetime.now(timezone.utc).isoformat(), "protected_artifact_hashes": before,
        "protected_artifacts_unchanged": True, "safe_provenance": safe_source,
        "versions": {p: version(p) for p in ("safeaipackage", "pandas", "numpy", "scikit-learn", "scipy", "catboost", "xgboost")},
        "group_coding": LABELS, "signed_direction": "female minus male", "threshold": .50,
        "sensitivity_grid": "0.10 to 0.70 inclusive, step 0.05; descriptive holdout sensitivity only",
        "selected_new_threshold": None, "models_fitted": False, "calibration_fitted": False,
        "bootstrap": {"repetitions": REPETITIONS, "seed": SEED, "unit": "complete exact 22-predictor group",
                      "interval": "2.5% and 97.5% percentiles; marginal, no multiplicity adjustment",
                      "SAFE_subgroup_RGA": "actual package rga on expanded whole-group samples"},
        "proxy_associations": "training data only; descriptive SMD/Cramer's V, no causal inference",
        "source_hashes": {p: sha256(ROOT / p) for p in ["src/fairness_evaluation.py", "src/safe_adapter.py", "src/fairness_report.py", "script_fairness_evaluation.py", "requirements-fairness.txt"]},
        "output_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in TABLE_DIR.glob("*.csv")}}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Fairness Evaluation complete; saved models and earlier stage artifacts unchanged.", flush=True)
    return artifacts
