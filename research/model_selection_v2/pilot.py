"""Train fresh pilot candidates from saved assignments; never mutate prior research."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from article_safe.data import check_evaluation_artifacts
from src.baseline import (ROOT, RAW_PATH, DATA_PATH, NUMERIC, CATEGORICAL, FEATURES,
                          clean_data, load_split, sha256)
from src.performance_evaluation import predictor_groups

ARTICLE = ROOT / "article_safe/runs/article_v1_2026-10-10"
RUNS = ROOT / "runs/model_selection_v2"
CANDIDATES = ("logistic", "decision_tree", "random_forest", "gradient_boosting")


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    def encode(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, type):
            return f"{item.__module__}.{item.__qualname__}"
        raise TypeError(f"Unsupported JSON value: {item!r}")
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=encode)
        stream.write("\n")


def write_csv(path, value):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        value.to_csv(stream, index=False)


def hashes(paths):
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(set(paths))}


def protected_hashes():
    # Preserve every pre-existing tracked file except the explicitly editable startup docs.
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    return hashes([ROOT / name for name in names if name and name not in
                   {"PROJECT_CONTEXT.md", "README.md", "AGENTS.md"}])


def finite_inputs(X):
    require(not X.isna().any().any(), "Missing model inputs.")
    require(np.isfinite(X.to_numpy(dtype=float)).all(), "Nonfinite model inputs.")


def validate_assignments(X, y, metadata, assignment):
    require(list(X.columns) == list(FEATURE_ORDER()), "Saved predictor order changed.")
    require(set(X.columns) == set(FEATURES) and len(X.columns) == 22,
            "Metadata must be excluded; require exactly 22 original predictors.")
    require(X.index.equals(y.index) and X.index.equals(metadata.index), "Row indexes misaligned.")
    require(len(assignment) == len(metadata), "Assignment length mismatch.")
    require(assignment.ID.is_unique and metadata.ID.is_unique, "Duplicate IDs.")
    require(np.array_equal(assignment.training_position, np.arange(len(metadata))),
            "Assignment training positions changed.")
    for field in ("ID", "SEX", "default", "duplicate_group_id"):
        require(np.array_equal(assignment[field], metadata[field]), f"Assignment {field} misaligned.")
    require(np.array_equal(y, metadata.default), "Outcome alignment mismatch.")
    require(set(assignment.original_partition) == {"train"}, "Non-training assignment.")
    require(set(metadata.split) == {"train"}, "Development records must be training records.")
    require(set(assignment.development_role) == {"fitting", "selection"}, "Unexpected roles.")
    require(np.array_equal(assignment.development_role.eq("selection"), assignment.fold.eq(1)),
            "Saved selection fold identity changed.")
    groups = predictor_groups(X, metadata)
    require(np.array_equal(groups, metadata.duplicate_group_id), "Original groups changed.")
    require(assignment.groupby("duplicate_group_id").development_role.nunique().max() == 1,
            "Original predictor group crosses fitting and selection.")
    finite_inputs(X)
    positions = {}
    for role in ("fitting", "selection"):
        positions[role] = np.flatnonzero(assignment.development_role.eq(role))
        require(set(y.iloc[positions[role]]) == {0, 1}, f"{role} must support both classes.")
    return positions


def FEATURE_ORDER():
    return json.loads((ROOT / "models/baseline_logistic_manifest.json").read_text())["feature_columns"]


def load_data():
    preparation = json.loads((DATA_PATH.parent / "preparation_manifest.json").read_text())
    article_manifest = json.loads((ARTICLE / "run_manifest.json").read_text())
    # The saved article manifest binds assignments and every reused input identity.
    for relative, expected in article_manifest["input_hashes"].items():
        require(sha256(ROOT / relative) == expected, f"Article input identity changed: {relative}")
    for name in ("development_assignments.csv", "protocol.json"):
        require(sha256(ARTICLE / name) == article_manifest["output_hashes"][name],
                f"Saved article {name} identity changed.")
    data = pd.read_csv(DATA_PATH)
    raw = pd.read_excel(RAW_PATH, header=1, engine="xlrd")
    cleaned = clean_data(raw)
    pd.testing.assert_frame_equal(cleaned, data, check_dtype=False)
    require(len(data) == 30000 and data.ID.is_unique, "Original record accounting changed.")
    require(set(data.default) == {0, 1}, "Target must be binary: 0 nondefault, 1 default.")
    parts = load_split(data)  # Read-only identity/alignment checks; no resplitting.
    X, y, metadata = parts[0].loc[:, FEATURE_ORDER()], parts[2], parts[4]
    holdout = parts[5]
    require(len(X) == 24004 and len(holdout) == 5996, "Outer partition sizes changed.")
    for name, saved in (("train", metadata), ("test", holdout)):
        pd.testing.assert_frame_equal(pd.read_csv(DATA_PATH.parent / f"{name}_metadata.csv"),
                                      saved.reset_index(drop=True))
        require(np.array_equal(saved.partition_order, np.arange(len(saved))), "Outer row order changed.")
    require(not (set(metadata.ID) & set(holdout.ID)), "Outer ID overlap.")
    require(set(metadata.ID) | set(holdout.ID) == set(data.ID), "Missing original records.")
    all_metadata = pd.read_csv(DATA_PATH.parent / "split_metadata.csv")
    all_X = data.loc[:, FEATURE_ORDER()]
    finite_inputs(all_X)
    require(np.array_equal(predictor_groups(all_X, all_metadata), all_metadata.duplicate_group_id),
            "Full original 22-predictor grouping changed.")
    require(all_metadata.groupby("duplicate_group_id").split.nunique().max() == 1,
            "Original predictor group crosses outer split.")
    assignment = pd.read_csv(ARTICLE / "development_assignments.csv")
    positions = validate_assignments(X, y, metadata, assignment)
    require([len(positions[r]) for r in ("fitting", "selection")] == [19203, 4801],
            "Saved development sizes changed.")
    audit = {
        "cleaning": {"source": "src.baseline.clean_data", "learned_statistics_or_categories": False,
                     "operations": ["rename target; 0=nondefault, 1=default",
                                    "EDUCATION: 0/5/6 -> 4", "MARRIAGE: 0 -> 3",
                                    "reject missing data or invalid IDs"],
                     "raw_cleaned_equals_saved": True, "conflicts": []},
        "records": {"original": 30000, "training": 24004, "fitting": 19203,
                    "selection": 4801, "historical_holdout": 5996},
        "development_default_counts": {r: int(y.iloc[p].sum()) for r, p in positions.items()},
        "assignment_source": (ARTICLE / "development_assignments.csv").relative_to(ROOT).as_posix(),
        "assignment_sha256": sha256(ARTICLE / "development_assignments.csv"),
        "missing_or_nonfinite_model_inputs": 0, "crossing_original_groups": 0,
        "outer_identity_and_label_alignment_checked": True,
        "holdout_predictions_or_metrics_computed": False,
        "partition_generation_executed": False,
        "preparation_manifest": preparation,
    }
    return X, y, metadata, assignment, positions, audit


def build_candidate(candidate):
    estimators = {
        # sklearn 1.9: l1_ratio=0 explicitly selects L2; penalty is deprecated.
        "logistic": lambda: LogisticRegression(solver="lbfgs", C=1.0, l1_ratio=0.0,
                                               max_iter=2000, class_weight=None, random_state=42),
        "decision_tree": lambda: DecisionTreeClassifier(criterion="gini", max_depth=5,
                      min_samples_split=2, min_samples_leaf=20, class_weight=None, random_state=42),
        "random_forest": lambda: RandomForestClassifier(n_estimators=300, criterion="gini",
                      max_depth=None, min_samples_split=2, min_samples_leaf=5, max_features="sqrt",
                      bootstrap=True, class_weight=None, n_jobs=1, random_state=42),
        "gradient_boosting": lambda: GradientBoostingClassifier(loss="log_loss", n_estimators=100,
                      learning_rate=0.1, max_depth=3, min_samples_split=2, min_samples_leaf=1,
                      subsample=1.0, random_state=42),
    }
    preprocessing = ColumnTransformer([("num", StandardScaler(), NUMERIC),
              ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return Pipeline([("preprocessor", preprocessing), ("classifier", estimators[candidate]())])


def decisions(probability):
    return (np.asarray(probability) > 0.50).astype(int)


def positive_probability(model, X):
    classes = np.asarray(model.classes_)
    require(classes.shape == (2,) and set(classes.tolist()) == {0, 1},
            "Each candidate must support both classes 0 and 1.")
    require(np.array_equal(classes, model.named_steps["classifier"].classes_), "Pipeline class mismatch.")
    index = int(np.flatnonzero(classes == 1)[0])
    probability = model.predict_proba(X)[:, index]
    require(probability.shape == (len(X),) and np.isfinite(probability).all(), "Invalid probabilities.")
    require(((probability >= 0) & (probability <= 1)).all(), "Probability outside [0,1].")
    return probability


def metrics(y, probability):
    y = np.asarray(y)
    predicted = decisions(probability)
    tn, fp, fn, tp = [int(np.sum((y == a) & (predicted == b))) for a, b in
                      ((0, 0), (0, 1), (1, 0), (1, 1))]
    ratio = lambda a, b: float(a / b) if b else None
    recall, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    both = set(y) == {0, 1}
    return {"roc_auc": float(roc_auc_score(y, probability)) if both else None,
            "average_precision": float(average_precision_score(y, probability)) if (y == 1).any() else None,
            "brier": float(brier_score_loss(y, probability)),
            "log_loss": float(log_loss(y, probability, labels=[0, 1])),
            "accuracy": ratio(tp + tn, len(y)),
            "balanced_accuracy": (recall + specificity) / 2 if both else None,
            "precision": ratio(tp, tp + fp), "recall": recall,
            "f1": ratio(2 * tp, 2 * tp + fp + fn), "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def preprocessing_audit(model, fitting, selection):
    pre = model.named_steps["preprocessor"]
    scaler, encoder = pre.named_transformers_["num"], pre.named_transformers_["cat"]
    require(np.all(scaler.n_samples_seen_ == len(fitting)), "Scaler fitting sample size mismatch.")
    np.testing.assert_allclose(scaler.mean_, fitting[NUMERIC].mean().to_numpy(), rtol=1e-13)
    np.testing.assert_allclose(scaler.var_, fitting[NUMERIC].var(ddof=0).to_numpy(), rtol=1e-13)
    require(np.array_equal(pre.feature_names_in_, fitting.columns), "Input feature order mismatch.")
    names = pre.get_feature_names_out().tolist()
    mapping, unseen = {}, {}
    for i, name in enumerate(NUMERIC):
        mapping[name] = [{"index": i, "column": names[i]}]
    offset = len(NUMERIC)
    for name, categories in zip(CATEGORICAL, encoder.categories_):
        np.testing.assert_array_equal(categories, np.sort(fitting[name].unique()))
        mask = ~selection[name].isin(categories)
        unseen[name] = {"records": int(mask.sum()), "levels": sorted(selection.loc[mask, name].unique().tolist()),
                        "level_counts": {str(k): int(v) for k, v in selection.loc[mask, name].value_counts().items()}}
        mapping[name] = [{"index": offset + j, "column": names[offset + j], "category": level.item()}
                         for j, level in enumerate(categories)]
        offset += len(categories)
    require(offset == len(names) and set(mapping) == set(FEATURES), "Incomplete original feature mapping.")
    for X in (fitting, selection):
        transformed = pre.transform(X)
        values = transformed.data if sparse.issparse(transformed) else transformed
        require(np.isfinite(values).all(), "Nonfinite transformed inputs.")
    return {"fitting_only_preprocessing_verified": True, "fitting_records_seen": len(fitting),
            "transformed_columns": names, "original_predictor_to_transformed_columns": mapping,
            "unseen_selection_categories": unseen, "transformed_inputs_finite": True}


def prediction_table(candidate, probability, metadata):
    require(metadata.ID.is_unique and len(probability) == len(metadata), "Prediction ID alignment failure.")
    result = metadata[["ID", "SEX", "default", "duplicate_group_id"]].reset_index(drop=True).copy()
    result.insert(0, "candidate_id", candidate)
    result["probability_default"] = probability
    result["prediction"] = decisions(probability)
    return result


def verify_predictions(model, X, metadata, table, expected_metrics):
    require(table.candidate_id.nunique() == 1 and len(table) == len(metadata), "Prediction table shape.")
    for field in ("ID", "SEX", "default", "duplicate_group_id"):
        np.testing.assert_array_equal(table[field], metadata[field])
    probability = positive_probability(model, X)
    np.testing.assert_allclose(probability, table.probability_default, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(decisions(probability), table.prediction)
    for name, value in metrics(table.default, table.probability_default).items():
        expected = expected_metrics[name]
        if value is None:
            require(expected is None or pd.isna(expected), f"Undefined metric {name} concealed.")
        else:
            np.testing.assert_allclose(value, expected, rtol=0, atol=1e-14)


def create_run(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "RUN_STARTED.json", {"status": "incomplete", "started_at_utc": now(),
               "completion_rule": "Only VERIFIED.json with valid hashes establishes completion."})
    return path


def verify_run(path):
    path = Path(path)
    marker = json.loads((path / "VERIFIED.json").read_text())
    require(marker["status"] == "verified", "Run incomplete.")
    actual = {p.relative_to(path).as_posix(): sha256(p) for p in path.rglob("*")
              if p.is_file() and p.name != "VERIFIED.json"}
    require(actual == marker["artifact_hashes"], "Verified run artifacts changed or incomplete.")
    return marker


def run(run_id):
    require(Path(run_id).name == run_id and run_id not in {".", ".."} and
            "/" not in run_id and "\\" not in run_id, "Run ID must be one directory name.")
    destination = create_run(RUNS / run_id)
    try:
        return execute(destination)
    except BaseException as error:
        write_json(destination / "FAILURE.json", {"status": "incomplete", "at_utc": now(),
                   "error_type": type(error).__name__, "error": str(error),
                   "traceback": traceback.format_exc()})
        raise


def execute(destination):
    started = now()
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--short", "--untracked-files=all"], cwd=ROOT, text=True)
    before = protected_hashes()
    source_paths = list((ROOT / "research/model_selection_v2").rglob("*.py")) + [
        ROOT / "src/baseline.py", ROOT / "src/performance_evaluation.py", ROOT / "article_safe/data.py"]
    source_hashes_before = hashes(source_paths)
    historical_before = check_evaluation_artifacts()
    X, y, metadata, assignment, positions, data_audit = load_data()
    fit, selection = positions["fitting"], positions["selection"]
    Xfit, yfit, Xselection = X.iloc[fit], y.iloc[fit], X.iloc[selection]
    selection_metadata = metadata.iloc[selection]
    models = {candidate: build_candidate(candidate) for candidate in CANDIDATES}
    protocol = {
        "version": "model_selection_v2_fixed_pilot_v1", "run_id": destination.name,
        "candidate_status": "four fixed, untuned pilot configurations; no universal family superiority inference",
        "target": {"0": "nondefault", "1": "default"}, "probability_class": 1,
        "decision_rule": "p(default) > 0.50; ties class 0", "seed": 42,
        "feature_order": FEATURE_ORDER(), "numeric": NUMERIC, "categorical": CATEGORICAL,
        "preprocessing": "Independent StandardScaler and OneHotEncoder(handle_unknown=ignore), fitting only",
        "logistic_regularization": "L2 via sklearn 1.9 l1_ratio=0.0, finite C=1.0; deprecated penalty omitted",
        "resolved_estimator_parameters": {c: m.named_steps["classifier"].get_params() for c, m in models.items()},
        "resolved_preprocessing_parameters": {"StandardScaler": StandardScaler().get_params(),
                          "OneHotEncoder": OneHotEncoder(handle_unknown="ignore").get_params(),
                          "ColumnTransformer": {"remainder": "drop", "sparse_threshold": 0.3,
                                                "n_jobs": None, "verbose_feature_names_out": True}},
        "fitting_time": "wall seconds for complete pipeline.fit (preprocessing plus estimator); excludes audit, serialization, predictions",
        "undefined_metrics": "JSON null / CSV empty; precision, recall, F1 null for zero denominator; AUC and balanced accuracy null unless both outcomes; AP null with no positives; LogLoss sklearn clipping with explicit labels [0,1]",
        "data_roles": data_audit["records"], "holdout": "historically inspected, exploratory; identity checks only",
        "later_auxiliary_models": "Future SAFE feature-removal auxiliary fits require a separate RGE protocol and must never modify or replace these frozen primary candidates",
        "excluded": ["SAFE", "feature ablation", "extended methodology", "judge", "tuning", "calibration",
                     "holdout predictions or performance", "refit on fitting plus selection"],
    }
    for subdir in ("models", "predictions", "diagnostics"):
        (destination / subdir).mkdir()
    test_result = subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), "-m", "unittest", "discover",
                 "-s", "research/model_selection_v2/tests", "-v"], cwd=ROOT, capture_output=True, text=True)
    write_json(destination / "diagnostics/focused_tests.json", {"command": test_result.args,
               "exit_code": test_result.returncode, "stdout": test_result.stdout, "stderr": test_result.stderr})
    require(test_result.returncode == 0, "Focused tests failed; see diagnostics/focused_tests.json.")
    write_json(destination / "protocol.json", protocol)
    write_json(destination / "diagnostics/data_audit.json", data_audit)
    write_csv(destination / "development_assignments.csv", assignment)
    rows, diagnostics = [], {}
    for candidate, model in models.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tick = time.perf_counter()
            model.fit(Xfit, yfit)
            elapsed = time.perf_counter() - tick
        fit_warnings = [{"category": item.category.__name__, "message": str(item.message),
                         "filename": item.filename, "line": item.lineno} for item in caught]
        probability = positive_probability(model, Xselection)
        audit = preprocessing_audit(model, Xfit, Xselection)
        estimator = model.named_steps["classifier"]
        audit.update({"classes_": model.classes_.tolist(), "positive_probability_column": int(np.flatnonzero(model.classes_ == 1)[0]),
                      "fit_warnings": fit_warnings,
                      "convergence_warning": any(issubclass(item.category, ConvergenceWarning) for item in caught),
                      "n_iter_": np.asarray(estimator.n_iter_).tolist() if hasattr(estimator, "n_iter_") else None,
                      "fitting_time_seconds": elapsed})
        diagnostics[candidate] = audit
        row = {"candidate_id": candidate, **metrics(y.iloc[selection], probability), "fitting_time_seconds": elapsed}
        rows.append(row)
        model_path = destination / f"models/{candidate}.joblib"
        with model_path.open("xb") as stream:
            joblib.dump(model, stream)
        restored = joblib.load(model_path)
        np.testing.assert_array_equal(positive_probability(restored, Xselection), probability)
        table = prediction_table(candidate, probability, selection_metadata)
        prediction_path = destination / f"predictions/{candidate}.csv"
        write_csv(prediction_path, table)
        verify_predictions(restored, Xselection, selection_metadata,
                           pd.read_csv(prediction_path, float_precision="round_trip"), row)
        write_json(destination / f"diagnostics/{candidate}.json", audit)
        print(f"Fitted and reloaded {candidate}: {elapsed:.3f}s; warnings={len(fit_warnings)}", flush=True)
    write_csv(destination / "diagnostics/selection_metrics.csv", pd.DataFrame(rows))
    write_json(destination / "diagnostics/selection_metrics.json", rows)
    saved_rows = pd.read_csv(destination / "diagnostics/selection_metrics.csv", float_precision="round_trip")
    for row in saved_rows.to_dict("records"):
        candidate = row["candidate_id"]
        verify_predictions(joblib.load(destination / f"models/{candidate}.joblib"), Xselection,
                           selection_metadata, pd.read_csv(destination / f"predictions/{candidate}.csv",
                           float_precision="round_trip"), row)
    after = protected_hashes()
    require(before == after, "Protected historical file hashes changed.")
    historical_after = check_evaluation_artifacts()
    write_json(destination / "diagnostics/preservation.json", {"before": before, "after_matches_before": True,
               "tracked_files_checked": len(before), "historical_manifest_checks_before": historical_before,
               "historical_manifest_checks_after": historical_after})
    report = make_report(rows, diagnostics, len(before))
    with (destination / "verification_report.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report)
    input_paths = [RAW_PATH, DATA_PATH, *DATA_PATH.parent.glob("*metadata.csv"),
                   DATA_PATH.parent / "preparation_manifest.json", ROOT / "models/baseline_logistic_manifest.json",
                   ARTICLE / "run_manifest.json", ARTICLE / "protocol.json", ARTICLE / "development_assignments.csv",
                   ROOT / "data/README.md"]
    source_hashes = hashes(source_paths)
    require(source_hashes == source_hashes_before, "Implementation changed during run.")
    inputs = hashes(input_paths)
    artifacts = {p.relative_to(destination).as_posix(): sha256(p) for p in destination.rglob("*") if p.is_file()}
    write_json(destination / "run_manifest.json", {
        "run_id": destination.name, "started_at_utc": started, "completed_at_utc": now(),
        "completion_rule": "Require VERIFIED.json and matching artifact hashes; manifest alone is insufficient",
        "starting_git_commit": git_commit, "starting_working_tree_status": git_status,
        "python": platform.python_version(), "versions": {p: version(p) for p in
                          ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "xlrd")},
        "input_hashes": inputs, "implementation_hashes": source_hashes,
        "configuration_sha256": sha256(destination / "protocol.json"), "artifact_hashes": artifacts,
        "candidates_fitted": list(CANDIDATES), "verification": {
            "partition_identity_and_original_group_isolation": True, "metadata_excluded": True,
            "fitting_only_preprocessing": True, "classes_and_positive_probability_column": True,
            "finite_probabilities": True, "reload_probabilities_bit_identical": True,
            "saved_predictions_and_diagnostics_agree": True, "historical_research_preserved": True,
            "holdout_assessment": False, "SAFE_or_judge": False, "tuning": False}})
    write_json(destination / "VERIFIED.json", {"status": "verified", "verified_at_utc": now(),
               "artifact_hashes": {p.relative_to(destination).as_posix(): sha256(p)
                                   for p in destination.rglob("*") if p.is_file()}})
    verify_run(destination)
    return destination


def make_report(rows, diagnostics, protected_count):
    columns = ["candidate_id", "roc_auc", "average_precision", "brier", "log_loss", "accuracy",
               "balanced_accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp", "fitting_time_seconds"]
    table = "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"] * len(columns)) + " |\n"
    for row in rows:
        table += "| " + " | ".join(f"{row[c]:.6f}" if isinstance(row[c], float) else str(row[c]) for c in columns) + " |\n"
    notes = []
    for candidate, audit in diagnostics.items():
        unseen = {name: value for name, value in audit["unseen_selection_categories"].items() if value["records"]}
        notes.append(f"- {candidate}: classes={audit['classes_']}; class-1 column={audit['positive_probability_column']}; "
                     f"n_iter_={audit['n_iter_']}; warnings={audit['fit_warnings']}; unseen={unseen or 'none'}.")
    return ("# Fixed pilot candidates — 26 October 2026\n\n"
        "Exploratory selection diagnostics, conditional on four fixed, untuned configurations. "
        "No overall winner or universal algorithm-family superiority is established.\n\n" + table + "\n"
        "Exact resolved settings are in protocol.json: L2 LR C=1, lbfgs, max_iter=2000; "
        "DT depth=5, min_leaf=20; RF 300 trees, min_leaf=5, sqrt features, n_jobs=1; "
        "GBM 100 trees, rate=.1, depth=3. Seeds=42. Each has independent fitting-only "
        "numeric StandardScaler and categorical OneHotEncoder(handle_unknown=ignore).\n\n"
        "Reused article_v1_2026-10-10/development_assignments.csv directly: 19,203 fitting, "
        "4,801 selection; together the saved 24,004 training records. All 30,000 IDs, "
        "labels, row order and whole original 22-predictor groups verified against saved metadata. "
        "The historical 5,996 holdout is inspected exploratory evidence; only its identity, "
        "alignment and group separation were checked. No holdout predictions or metrics.\n\n"
        "Cleaning audit: fixed target rename, EDUCATION 0/5/6→4, MARRIAGE 0→3, ID/missing "
        "validation. No learned cleaning statistics or mappings; in-memory raw cleaning equals "
        "saved processed data. No source conflict. Inputs and transformed matrices are finite.\n\n"
        "Each diagnostics/<candidate>.json maps every original predictor to transformed column "
        "indices/names and one-hot category levels; it separately counts unseen selection "
        "levels for all eight categorical predictors. Numeric sample counts, means, variances "
        "and category sets match fitting records exclusively.\n\n" + "\n".join(notes) + "\n\n"
        "0=nondefault; 1=default. Both estimator classes asserted, class-1 column looked up "
        "explicitly. Decisions use p(default)>.50, ties=0. Full pipeline fitting wall time "
        "includes preprocessing and estimator fit. Undefined metrics remain null/empty with "
        "denominator rules in protocol.json. AP is sklearn average precision.\n\n"
        "Four saved/reloaded pipelines reproduce selection probabilities bit for bit; saved "
        "IDs/outcomes/SEX/groups and decisions match; metrics recomputed from saved predictions "
        "agree with JSON and CSV diagnostics. "
        f"All {protected_count} protected tracked files match pre-run hashes; the 410 historical "
        "manifest references passed before and after, without stage regeneration.\n\n"
        "RUN_STARTED.json records inception as incomplete. Only VERIFIED.json with matching "
        "hashes establishes completion. Existing destinations and files are never overwritten; "
        "interrupted runs remain incomplete. No SAFE, feature ablation, fairness assessment, "
        "tuning, new judge or refit on combined fitting/selection data. Primary candidates are "
        "frozen. Future auxiliary feature-removal models must be separately fitted under a "
        "later RGE protocol and must never replace or modify these candidates. "
        "Next: discuss and specify SAFE 2026 implementation. Judge options remain unresolved.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--verify-run", type=Path, help="Read-only artifact integrity check; no training")
    arguments = parser.parse_args()
    if arguments.verify_run:
        print(verify_run(arguments.verify_run)["status"])
    else:
        parser.error("--run-id is required") if not arguments.run_id else print(run(arguments.run_id))
