"""Training-only diagnostics and fixed-model evaluation on the locked holdout.

No full-data split, tuning, calibration fit, threshold selection, or model saving
is performed here. CV fits are temporary, with preprocessing fitted per fold.
"""
from datetime import datetime, timezone
from importlib.metadata import version
import json
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.baseline import ROOT, DATA_PATH, RAW_PATH, FEATURES, build_model, load_split, sha256

TABLE_DIR = ROOT / "outputs/tables/performance_evaluation"
FIGURE_DIR = ROOT / "outputs/figures/performance_evaluation"
MODELS = ("logistic", "gradient_boosting")
SEED = 42
N_FOLDS = 5
BOOTSTRAP_REPETITIONS = 2000
METRICS = ("ROC-AUC", "Average Precision (AP)", "Accuracy", "Balanced Accuracy",
           "Precision", "Recall", "Specificity", "F1", "Log Loss", "Brier Score")


def predictor_groups(X, metadata):
    """Exact column equality, excluding ID, SEX and target; minimum ID labels."""
    if set(X.columns) != set(FEATURES) or not X.index.equals(metadata.index):
        raise ValueError("Predictor schema or metadata alignment changed.")
    frame = X.copy()
    frame["ID"] = metadata.ID
    return frame.groupby(FEATURES, sort=False, dropna=False).ID.transform("min").to_numpy()


def class_metrics(y, predicted, weights=None):
    y, predicted = np.asarray(y), np.asarray(predicted)
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    tn = w[(y == 0) & (predicted == 0)].sum()
    fp = w[(y == 0) & (predicted == 1)].sum()
    fn = w[(y == 1) & (predicted == 0)].sum()
    tp = w[(y == 1) & (predicted == 1)].sum()
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {"TN": float(tn), "FP": float(fp), "FN": float(fn), "TP": float(tp),
            "Predicted-positive Rate": (tp + fp) / total,
            "Accuracy": (tp + tn) / total, "Balanced Accuracy": (recall + specificity) / 2,
            "Precision": precision, "Recall": recall, "Specificity": specificity,
            "FPR": 1 - specificity, "FNR": 1 - recall,
            "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def performance_metrics(y, probability, predicted=None, weights=None):
    y, probability = np.asarray(y), np.asarray(probability)
    if predicted is None:
        predicted = probability > 0.50  # sklearn binary baseline tie rule: 0 at p=0.5.
    result = class_metrics(y, predicted, weights)
    result.update({
        "ROC-AUC": roc_auc_score(y, probability, sample_weight=weights),
        "Average Precision (AP)": average_precision_score(y, probability, sample_weight=weights),
        "Log Loss": log_loss(y, probability, labels=[0, 1], sample_weight=weights),
        "Brier Score": np.average((probability - y) ** 2, weights=weights),
    })
    return result


def training_folds(X_train, y_train, metadata_train):
    """This interface deliberately has no holdout inputs."""
    groups = predictor_groups(X_train, metadata_train)
    np.testing.assert_array_equal(groups, metadata_train.duplicate_group_id)
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(X_train, y_train, groups))
    assignment = np.zeros(len(X_train), dtype=int)
    for fold, (fit_rows, validation_rows) in enumerate(folds, 1):
        assert not set(groups[fit_rows]) & set(groups[validation_rows])
        assert not assignment[validation_rows].any()
        assignment[validation_rows] = fold
    assert (assignment > 0).all()
    return folds, assignment, groups


def cross_validate_training(X_train, y_train, metadata_train):
    folds, assignment, groups = training_folds(X_train, y_train, metadata_train)
    fold_rows, predictions, partition_rows = [], [], []
    for fold, (fit_rows, validation_rows) in enumerate(folds, 1):
        partition_rows.append({"fold": fold, "fit_records": len(fit_rows),
            "validation_records": len(validation_rows), "fit_groups": len(np.unique(groups[fit_rows])),
            "validation_groups": len(np.unique(groups[validation_rows])),
            "fit_default_rate": y_train.iloc[fit_rows].mean(),
            "validation_default_rate": y_train.iloc[validation_rows].mean(), "crossing_groups": 0})
    for kind in MODELS:
        probabilities = np.full(len(X_train), np.nan)
        classes = np.full(len(X_train), -1, dtype=int)
        for fold, (fit_rows, validation_rows) in enumerate(folds, 1):
            print(f"CV {kind}, fold {fold}/{N_FOLDS}: fitting on training-partition rows only", flush=True)
            model = build_model(kind)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(X_train.iloc[fit_rows], y_train.iloc[fit_rows])
            p = model.predict_proba(X_train.iloc[validation_rows])[:, 1]
            predicted = model.predict(X_train.iloc[validation_rows])
            np.testing.assert_array_equal(predicted, p > 0.50)
            probabilities[validation_rows], classes[validation_rows] = p, predicted
            scores = performance_metrics(y_train.iloc[validation_rows], p, predicted)
            fold_rows.append({"model": kind, **partition_rows[fold - 1], **scores})
        assert np.isfinite(probabilities).all() and (classes >= 0).all()
        out = metadata_train[["ID", "SEX", "default", "duplicate_group_id"]].copy()
        out = out.rename(columns={"default": "actual"})
        out["model"], out["partition"], out["fold"] = kind, "training_oof", assignment
        out["probability_default"], out["predicted"] = probabilities, classes
        predictions.append(out.reset_index(drop=True))
    fold_metrics = pd.DataFrame(fold_rows)
    summary = fold_metrics.melt(id_vars=["model", "fold"], value_vars=list(METRICS),
                                var_name="metric", value_name="value")
    summary = summary.groupby(["model", "metric"], sort=False).value.agg(
        mean="mean", std="std", minimum="min", maximum="max", folds="count").reset_index()
    assignment_table = metadata_train[["ID", "SEX", "default", "duplicate_group_id"]].copy()
    assignment_table["fold"] = assignment
    return {"cv_fold_metrics": fold_metrics, "cv_summary_metrics": summary,
            "cv_fold_assignments": assignment_table, "cv_partition_summary": pd.DataFrame(partition_rows),
            "oof_predictions": pd.concat(predictions, ignore_index=True)}


def threshold_analysis(oof_predictions):
    if set(oof_predictions.partition) != {"training_oof"}:
        raise ValueError("Threshold analysis accepts training OOF predictions only.")
    grid_rows, target_rows = [], []
    for kind, part in oof_predictions.groupby("model", sort=False):
        y, p = part.actual.to_numpy(), part.probability_default.to_numpy()
        for threshold in np.linspace(0, 1, 101):
            grid_rows.append({"model": kind, "partition": "training_oof", "threshold": threshold,
                              **class_metrics(y, p > threshold)})
        positive_scores = np.sort(p[y == 1])[::-1]
        for target in (0.40, 0.50, 0.60, 0.70):
            rank = int(np.ceil(target * len(positive_scores))) - 1
            boundary = positive_scores[rank]
            # Largest representable threshold below the required positive score:
            # includes all tied scores and achieves at least the requested recall.
            threshold = np.nextafter(boundary, -np.inf)
            target_rows.append({"model": kind, "partition": "training_oof",
                "recall_target": target, "threshold": threshold, "included_score_boundary": boundary,
                "status": "candidate_only", **class_metrics(y, p > threshold)})
    return pd.DataFrame(grid_rows), pd.DataFrame(target_rows)


def reliability_bins(y, probability, n_bins=10):
    """Equal-width [left, right) bins; last includes 1. ECE is count-weighted."""
    y, probability = np.asarray(y), np.asarray(probability)
    edges = np.linspace(0, 1, n_bins + 1)
    indices = np.minimum(np.searchsorted(edges, probability, side="right") - 1, n_bins - 1)
    rows = []
    for index in range(n_bins):
        mask = indices == index
        count = int(mask.sum())
        mean = float(probability[mask].mean()) if count else np.nan
        observed = float(y[mask].mean()) if count else np.nan
        rows.append({"bin": index + 1, "lower": edges[index], "upper": edges[index + 1],
                     "count": count, "mean_probability": mean, "observed_frequency": observed,
                     "absolute_gap": abs(mean - observed),
                     "ece_contribution": count / len(y) * abs(mean - observed) if count else 0.0})
    return pd.DataFrame(rows)


def calibration_assessment(predictions):
    bins, summaries = [], []
    for (partition, kind), part in predictions.groupby(["partition", "model"], sort=False):
        table = reliability_bins(part.actual, part.probability_default)
        table.insert(0, "model", kind)
        table.insert(0, "partition", partition)
        bins.append(table)
        scores = performance_metrics(part.actual, part.probability_default, part.predicted)
        summaries.append({"model": kind, "partition": partition, "records": len(part),
            "mean_probability": part.probability_default.mean(), "observed_frequency": part.actual.mean(),
            "Brier Score": scores["Brier Score"], "Log Loss": scores["Log Loss"],
            "ECE_10_uniform_bins": table.ece_contribution.sum()})
    return pd.concat(bins, ignore_index=True), pd.DataFrame(summaries)


def evaluate_fixed_holdout(X_test, y_test, metadata_test):
    """Only original fitted pipelines and their existing 0.50 decisions."""
    predictions, scores = [], []
    for kind in MODELS:
        model = joblib.load(ROOT / f"models/baseline_{kind}.joblib")
        p, predicted = model.predict_proba(X_test)[:, 1], model.predict(X_test)
        np.testing.assert_array_equal(predicted, p > 0.50)
        saved = pd.read_csv(ROOT / f"outputs/tables/baseline_{kind}_predictions.csv")
        np.testing.assert_array_equal(saved.ID, metadata_test.ID)
        np.testing.assert_array_equal(saved.actual, y_test)
        np.testing.assert_array_equal(saved.SEX, metadata_test.SEX)
        np.testing.assert_array_equal(saved.predicted, predicted)
        np.testing.assert_allclose(saved.probability_default, p, rtol=0, atol=1e-15)
        out = metadata_test[["ID", "SEX", "default", "duplicate_group_id"]].copy()
        out = out.rename(columns={"default": "actual"})
        out["model"], out["partition"] = kind, "locked_holdout"
        out["probability_default"], out["predicted"] = p, predicted
        predictions.append(out.reset_index(drop=True))
        scores.append({"model": kind, "partition": "locked_holdout", "threshold": 0.50,
                       "records": len(y_test), **performance_metrics(y_test, p, predicted)})
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(scores)


def bootstrap_group_weights(group_codes, rng):
    """Integer row multiplicities equivalent to expanding sampled whole groups."""
    _, inverse = np.unique(group_codes, return_inverse=True)
    group_count = inverse.max() + 1
    draws = rng.integers(0, group_count, size=group_count)
    return np.bincount(draws, minlength=group_count)[inverse]


def group_bootstrap(holdout_predictions, repetitions=BOOTSTRAP_REPETITIONS, seed=SEED):
    if set(holdout_predictions.partition) != {"locked_holdout"}:
        raise ValueError("This bootstrap evaluates fixed holdout predictions only.")
    parts = {kind: part.reset_index(drop=True) for kind, part in holdout_predictions.groupby("model", sort=False)}
    reference = next(iter(parts.values()))
    for part in parts.values():
        pd.testing.assert_frame_equal(reference[["ID", "actual", "duplicate_group_id"]],
                                      part[["ID", "actual", "duplicate_group_id"]])
    rng, rows, skipped = np.random.default_rng(seed), [], 0
    for repetition in range(1, repetitions + 1):
        weights = bootstrap_group_weights(reference.duplicate_group_id.to_numpy(), rng)
        y = reference.actual.to_numpy()
        if weights[y == 0].sum() == 0 or weights[y == 1].sum() == 0:
            skipped += 1
            continue
        for kind, part in parts.items():
            values = performance_metrics(y, part.probability_default, part.predicted, weights)
            rows.append({"model": kind, "repetition": repetition, "sample_records": int(weights.sum()),
                         **{metric: values[metric] for metric in METRICS}})
        if repetition % 250 == 0:
            print(f"Holdout group bootstrap: {repetition}/{repetitions}", flush=True)
    replicates = pd.DataFrame(rows)
    if replicates.empty:
        raise ValueError("No valid bootstrap samples containing both outcome classes.")
    intervals = []
    for kind, part in parts.items():
        point = performance_metrics(part.actual, part.probability_default, part.predicted)
        sample = replicates[replicates.model == kind]
        for metric in METRICS:
            low, high = np.quantile(sample[metric], [0.025, 0.975], method="linear")
            intervals.append({"model": kind, "metric": metric, "estimate": point[metric],
                "ci_lower": low, "ci_upper": high, "confidence_level": 0.95,
                "method": "predictor-group percentile bootstrap", "seed": seed,
                "attempted_repetitions": repetitions, "valid_repetitions": len(sample),
                "single_class_samples_skipped": skipped,
                "holdout_groups": reference.duplicate_group_id.nunique()})
    return pd.DataFrame(intervals), replicates


def locked_input_hashes():
    paths = [RAW_PATH, *DATA_PATH.parent.glob("*"), *ROOT.joinpath("models").glob("baseline_*"),
             *ROOT.joinpath("outputs/tables").glob("baseline_*_predictions.csv"),
             *ROOT.joinpath("outputs/previous_baseline").glob("*")]
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths) if path.is_file()}


def validate_fitted_baselines():
    for kind in MODELS:
        path = ROOT / f"models/baseline_{kind}.joblib"
        manifest = json.loads(path.with_name(f"baseline_{kind}_manifest.json").read_text())
        for key, source in [("model_sha256", path), ("processed_sha256", DATA_PATH),
                            ("split_sha256", DATA_PATH.parent / "split_metadata.csv")]:
            if manifest[key] != sha256(source):
                raise ValueError(f"Locked {kind} baseline no longer matches its manifest: {key}")
        if manifest["classifier_parameters"] != build_model(kind).named_steps["classifier"].get_params():
            raise ValueError("Baseline hyperparameters changed; performance evaluation must not retune.")
        for package, expected in manifest["versions"].items():
            if version(package) != expected:
                raise ValueError(f"Use the pinned baseline environment: {package} must be {expected}")


def run_performance_evaluation():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    before = locked_input_hashes()
    manifest_path = TABLE_DIR / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["locked_input_hashes"] != before:
            raise ValueError("Locked baseline inputs changed since this evaluation stage began.")
    validate_fitted_baselines()
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test, meta_train, meta_test = load_split(df)
    groups_train, groups_test = predictor_groups(X_train, meta_train), predictor_groups(X_test, meta_test)
    np.testing.assert_array_equal(groups_test, meta_test.duplicate_group_id)
    assert not set(groups_train) & set(groups_test)
    assert not set(map(tuple, X_train[FEATURES].to_numpy())) & set(map(tuple, X_test[FEATURES].to_numpy()))
    artifacts = cross_validate_training(X_train, y_train, meta_train)
    # All threshold exploration finishes before holdout prediction/evaluation.
    artifacts["threshold_tradeoffs"], artifacts["recall_target_candidates"] = threshold_analysis(artifacts["oof_predictions"])
    oof_scores = []
    for kind, part in artifacts["oof_predictions"].groupby("model", sort=False):
        oof_scores.append({"model": kind, "partition": "training_oof", "threshold": 0.50,
            "records": len(part), **performance_metrics(part.actual, part.probability_default, part.predicted)})
    artifacts["oof_metrics"] = pd.DataFrame(oof_scores)
    artifacts["holdout_predictions"], artifacts["holdout_metrics"] = evaluate_fixed_holdout(X_test, y_test, meta_test)
    artifacts["overall_performance_comparison"] = pd.concat(
        [artifacts["oof_metrics"], artifacts["holdout_metrics"]], ignore_index=True)
    predictions = pd.concat([artifacts["oof_predictions"], artifacts["holdout_predictions"]], ignore_index=True)
    artifacts["calibration_bins"], artifacts["calibration_summary"] = calibration_assessment(predictions)
    artifacts["holdout_confidence_intervals"], artifacts["holdout_bootstrap_replicates"] = group_bootstrap(artifacts["holdout_predictions"])
    for name, table in artifacts.items():
        table.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    from src.performance_report import write_report, make_plots
    make_plots(artifacts, FIGURE_DIR)
    write_report(artifacts)
    if locked_input_hashes() != before:
        raise RuntimeError("A protected baseline artifact changed during evaluation.")
    manifest = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "versions": {p: version(p) for p in ("pandas", "numpy", "scipy", "scikit-learn", "joblib", "matplotlib")},
        "locked_input_hashes": before, "locked_inputs_unchanged": True,
        "training_records": len(X_train), "holdout_records": len(X_test), "features": X_train.columns.tolist(),
        "cv": {"splitter": "StratifiedGroupKFold", "n_splits": N_FOLDS, "shuffle": True,
               "random_state": SEED, "std_ddof": 1, "preprocessing_fit_within_each_fold": True},
        "thresholds": {"source": "training_oof_only", "grid": "0.00 to 1.00 inclusive, step 0.01",
                       "class_rule": "probability > threshold; ties are class 0", "selected_threshold": None,
                       "fixed_holdout_threshold": 0.50, "recall_targets": [0.4, 0.5, 0.6, 0.7]},
        "calibration": {"fit_calibrator": False, "bins": "10 equal-width bins over [0,1]",
                        "ece": "sum(count / N * abs(mean_probability - observed_frequency)); empty bins contribute zero"},
        "bootstrap": {"unit": "exact predictor group", "repetitions": BOOTSTRAP_REPETITIONS,
                      "random_seed": SEED, "interval": "2.5th and 97.5th percentiles, linear quantiles",
                      "models_refitted": False, "same_resamples_for_both_models": True,
                      "model_difference_test": False},
        "source_hashes": {name: sha256(ROOT / name) for name in ["src/baseline.py", "src/performance_evaluation.py", "src/performance_report.py", "script_performance_evaluation.py", "requirements.txt"]},
        "output_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(TABLE_DIR.glob("*.csv"))},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Performance evaluation complete. Locked split, pipelines and baseline predictions unchanged.", flush=True)
    return artifacts
