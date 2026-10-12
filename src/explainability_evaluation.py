"""SHAP and authentic SAFE RGE on the frozen, complete holdout; never fit."""
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.special import expit
from scipy.stats import spearmanr
from safeaipackage import check_explainability as safe_explain

from src.baseline import ROOT, DATA_PATH, FEATURES, NUMERIC, CATEGORICAL, load_split, sha256
from src.performance_evaluation import validate_fitted_baselines
from src.fairness_evaluation import MODELS, LABELS, protected_hashes as fairness_protected_hashes
from src.safe_adapter import provenance

TABLE_DIR = ROOT / "outputs/tables/explainability_evaluation"
FIGURE_DIR = ROOT / "outputs/figures/explainability_evaluation"
REPORT = ROOT / "outputs/explainability_evaluation_report.md"
SEED, BACKGROUND_SIZE = 42, 100
SAFE_EXPLAIN_HASH = "aae9e33c566200f31f82685538bd0ee0fcddf7c0437cc032c1a1edb0313cba23"
GROUPS = {
    "repayment_status": ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
    "bill_amounts": [f"BILL_AMT{i}" for i in range(1, 7)],
    "payment_amounts": [f"PAY_AMT{i}" for i in range(1, 7)],
    "background_and_credit_limit": ["LIMIT_BAL", "AGE", "EDUCATION", "MARRIAGE"],
}
SOURCES = ["src/explainability_evaluation.py", "src/explainability_report.py",
           "script_explainability_evaluation.py", "requirements-explainability.txt",
           "tests/test_explainability_evaluation.py"]


def protected_hashes():
    result = fairness_protected_hashes()
    paths = [*ROOT.joinpath("data").rglob("*"), *ROOT.joinpath("models").glob("*"),
             *ROOT.joinpath("outputs/tables/fairness_evaluation").glob("*"),
             *ROOT.joinpath("outputs/figures/fairness_evaluation").glob("*"),
             ROOT / "outputs/fairness_evaluation_report.md", ROOT / "requirements-fairness.txt"]
    result.update({str(p.relative_to(ROOT)): sha256(p) for p in paths if p.is_file()})
    return result


def validate_prior_stages():
    validate_fitted_baselines()
    for stage in ("performance", "fairness"):
        manifest = json.loads((ROOT / f"outputs/tables/{stage}_evaluation/run_manifest.json").read_text())
        for category in ("locked_input_hashes", "protected_artifact_hashes", "source_hashes", "output_hashes"):
            for name, expected in manifest.get(category, {}).items():
                if sha256(ROOT / name) != expected:
                    raise ValueError(f"Prior-stage artifact changed: {name}")


def finite(name, values):
    if not np.isfinite(np.asarray(values, dtype=float)).all():
        raise ValueError(f"Non-finite values in {name}")


def dense(values):
    return values.toarray() if hasattr(values, "toarray") else np.asarray(values)


def background_positions(metadata, size=BACKGROUND_SIZE):
    # Sort by stable ID before sampling; neither target nor SEX drives selection.
    order = np.argsort(metadata.ID.to_numpy(), kind="stable")
    return order[np.sort(np.random.default_rng(SEED).choice(len(order), size=size, replace=False))]


def transformed_mapping(preprocessor):
    names = preprocessor.get_feature_names_out().tolist()
    encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    if encoder.drop is not None:
        raise ValueError("Unexpected dropped one-hot levels")
    originals = NUMERIC + [f for f, cats in zip(CATEGORICAL, encoder.categories_) for _ in cats]
    if len(names) != len(originals):
        raise ValueError("Preprocessing schema changed")
    expected = [f"num__{f}" for f in NUMERIC] + [f"cat__{f}_{c}" for f, cats in zip(CATEGORICAL, encoder.categories_) for c in cats]
    if names != expected:
        raise ValueError("Unexpected transformed column order")
    return pd.DataFrame({"transformed_feature": names, "feature": originals})


def explain_shap(model, kind, X_test, background):
    pre, classifier = model.named_steps["preprocessor"], model.named_steps["classifier"]
    transformed, reference = dense(pre.transform(X_test)), dense(pre.transform(background))
    mapping = transformed_mapping(pre)
    masker = shap.maskers.Independent(reference, max_samples=len(reference))
    if kind == "logistic":
        explainer = shap.LinearExplainer(classifier, masker)
    else:
        explainer = shap.TreeExplainer(classifier, data=masker, model_output="raw", feature_perturbation="interventional")
    np.testing.assert_array_equal(explainer.masker.data, reference)
    explanation = explainer(transformed)
    raw = np.asarray(explanation.values)
    base = np.broadcast_to(np.asarray(explanation.base_values), (len(X_test),)).copy()
    if raw.shape != transformed.shape:
        raise ValueError("SHAP transformed shape mismatch")
    # Sum signed dummy contributions BEFORE taking an absolute value.
    values = np.column_stack([raw[:, mapping.feature.eq(feature)].sum(axis=1) for feature in FEATURES])
    margin = model.decision_function(X_test)
    finite("SHAP", values)
    finite("SHAP transformed", raw)
    finite("SHAP base", base)
    np.testing.assert_allclose(base + values.sum(axis=1), margin, rtol=0, atol=2e-6)
    np.testing.assert_allclose(expit(base + values.sum(axis=1)), model.predict_proba(X_test)[:, 1], rtol=0, atol=5e-7)
    return values, raw, base, mapping, float(np.max(np.abs(base + values.sum(axis=1) - margin)))


def safe_frames(X_train, X_test, pipeline):
    # Dtypes communicate EXISTING categorical semantics to SAFE; values do not change.
    train, test = X_train.reset_index(drop=True).copy(), X_test.reset_index(drop=True).copy()
    pre = pipeline.named_steps["preprocessor"]
    cats = pre.named_transformers_["cat"].named_steps["onehot"].categories_
    for feature, categories in zip(CATEGORICAL, cats):
        dtype = pd.CategoricalDtype(categories=categories, ordered=False)
        train[feature], test[feature] = train[feature].astype(dtype), test[feature].astype(dtype)
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError("Categorical SAFE inputs would drop/unrecognize a value")
    np.testing.assert_array_equal(pipeline.predict_proba(test), pipeline.predict_proba(X_test))
    np.testing.assert_array_equal(pipeline.predict_proba(train), pipeline.predict_proba(X_train))
    return train, test


def evaluate_rge(pipeline, X_train, X_test):
    train, test = safe_frames(X_train, X_test, pipeline)
    probabilities = pipeline.predict_proba(X_test)[:, 1].tolist()
    individual = safe_explain.compute_rge_values(train, test, probabilities, pipeline, FEATURES, group=False)
    individual = individual.rename_axis("feature").reset_index()
    individual["rge_rank"] = individual.RGE.rank(ascending=False, method="average")
    grouped = []
    for name, features in GROUPS.items():
        result = safe_explain.compute_rge_values(train, test, probabilities, pipeline, features, group=True)
        grouped.append({"group": name, "features": ",".join(features), "RGE": float(result.RGE.iloc[0])})
    grouped = pd.DataFrame(grouped)
    for result in (individual, grouped):
        finite("SAFE RGE", result.RGE)
        if not result.RGE.between(-1e-12, 1 + 1e-12).all():
            raise ValueError("SAFE RGE outside the expected range")
    replacements = pd.DataFrame([{"feature": f, "replacement": float(train[f].mode().iloc[0] if f in CATEGORICAL else train[f].mean()),
        "rule": "training mode (first sorted tie)" if f in CATEGORICAL else "training mean"} for f in FEATURES])
    return individual, grouped, replacements


def importance(values, kind, metadata):
    rows = []
    for sex in (0, 1, 2):
        mask = np.ones(len(metadata), dtype=bool) if sex == 0 else metadata.SEX.to_numpy() == sex
        part = values[mask]
        absolute = np.abs(part).mean(axis=0)
        frame = pd.DataFrame({"model": kind, "SEX": sex, "group": "overall" if sex == 0 else LABELS[sex],
            "records": int(mask.sum()), "feature": FEATURES, "mean_abs_shap": absolute,
            "mean_signed_shap": part.mean(axis=0), "positive_fraction": (part > 0).mean(axis=0),
            "importance_share": absolute / absolute.sum()})
        frame["shap_rank"] = frame.mean_abs_shap.rank(ascending=False, method="average")
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def group_comparison(importance_table):
    keys = ["model", "feature"]
    cols = keys + ["mean_abs_shap", "mean_signed_shap", "importance_share", "shap_rank"]
    result = importance_table.query("SEX == 1")[cols].merge(importance_table.query("SEX == 2")[cols], on=keys, suffixes=("_male", "_female"))
    for column in ("mean_abs_shap", "mean_signed_shap", "importance_share", "shap_rank"):
        result[column + "_female_minus_male"] = result[column + "_female"] - result[column + "_male"]
    result["absolute_share_gap"] = result.importance_share_female_minus_male.abs()
    return result.sort_values(["model", "absolute_share_gap", "feature"], ascending=[True, False, True])


def select_local_cases(predictions):
    rows = []
    for kind in MODELS:
        part = predictions[predictions.model == kind]
        for outcome, actual, predicted in [("TP", 1, 1), ("TN", 0, 0), ("FP", 0, 1), ("FN", 1, 0)]:
            candidates = part[(part.actual == actual) & (part.predicted == predicted)]
            if not candidates.empty:
                rows.append({"selection_model": kind, "selection_outcome": outcome, "ID": int(candidates.ID.min()),
                             "eligible_records": len(candidates)})
    return pd.DataFrame(rows)


def run_explainability_evaluation():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    validate_prior_stages()
    before = protected_hashes()
    manifest_path = TABLE_DIR / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["protected_artifact_hashes"] != before:
        raise ValueError("Protected inputs changed since Explainability was frozen")
    safe_source = provenance()
    source_hash = sha256(safe_explain.__file__)
    normalized = hashlib.sha256(Path(safe_explain.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if SAFE_EXPLAIN_HASH not in (source_hash, normalized):
        raise ValueError("SAFE explainability source changed")
    safe_source.update({"explainability_source_hash": source_hash,
        "explainability_function": "safeaipackage.check_explainability.compute_rge_values",
        "explainability_provenance": "Installed 0.8.3 source inspected and hashed locally; no upstream commit equality claimed for this module"})
    X_train, X_test, _, y_test, meta_train, meta_test = load_split(pd.read_csv(DATA_PATH))
    if (len(X_train), len(X_test)) != (24004, 5996):
        raise ValueError("Frozen partition size changed")
    positions = background_positions(meta_train)
    background = X_train.iloc[positions]
    background_meta = meta_train.iloc[positions].reset_index(drop=True)
    if set(background_meta.ID) & set(meta_test.ID):
        raise ValueError("Holdout used as background")
    group_summary = pd.read_csv(ROOT / "outputs/tables/fairness_evaluation/group_summary.csv")
    for row in group_summary.itertuples():
        if int((meta_test.SEX == row.SEX).sum()) != row.records:
            raise ValueError("SEX counts differ from fairness")
    artifacts = {"background_metadata": background_meta, "group_summary": group_summary}
    shap_data, predictions, all_importance, rge_rows, grouped_rows, checks = {}, [], [], [], [], []
    effects = []
    for kind in MODELS:
        print(f"Explaining {kind}: all {len(X_test)} holdout rows", flush=True)
        model = joblib.load(ROOT / f"models/baseline_{kind}.joblib")
        p, predicted = model.predict_proba(X_test)[:, 1], model.predict(X_test)
        saved = pd.read_csv(ROOT / f"outputs/tables/baseline_{kind}_predictions.csv", float_precision="round_trip")
        for observed, expected in [(saved.ID, meta_test.ID), (saved.SEX, meta_test.SEX), (saved.actual, y_test),
                                   (saved.predicted, predicted), (saved.probability_default, p)]:
            np.testing.assert_array_equal(observed, expected)
        values, raw, base, mapping, error = explain_shap(model, kind, X_test, background)
        shap_data[kind] = (values, base)
        audit = pd.DataFrame({"model": kind, "ID": meta_test.ID.to_numpy(), "SEX": meta_test.SEX.to_numpy(),
            "actual": y_test.to_numpy(), "predicted": predicted, "probability_default": p,
            "margin": model.decision_function(X_test), "base_value": base,
            "reconstructed_margin": base + values.sum(axis=1)})
        predictions.append(audit)
        artifacts[f"shap_values_{kind}"] = pd.concat([audit[["ID", "SEX", "base_value"]], pd.DataFrame(values, columns=FEATURES)], axis=1)
        artifacts[f"shap_transformed_values_{kind}"] = pd.concat([audit[["ID"]], pd.DataFrame(raw, columns=mapping.transformed_feature)], axis=1)
        artifacts[f"feature_mapping_{kind}"] = mapping
        imp = importance(values, kind, meta_test)
        all_importance.append(imp)
        artifacts[f"global_shap_importance_{kind}"] = imp.query("SEX == 0").sort_values(["shap_rank", "feature"])
        single, grouped, replacements = evaluate_rge(model, X_train, X_test)
        single.insert(0, "model", kind)
        grouped.insert(0, "model", kind)
        rge_rows.append(single)
        grouped_rows.append(grouped)
        artifacts["safe_replacement_values"] = replacements
        np.testing.assert_array_equal(model.predict_proba(X_test)[:, 1], p)
        checks.append({"model": kind, "holdout_rows": len(values), "original_features": values.shape[1],
                       "transformed_features": raw.shape[1], "max_margin_reconstruction_error": error,
                       "predictions_unchanged": True})
        # Categorical direction summary; no assumed ordering/meaning of undocumented codes.
        for f in CATEGORICAL:
            j = FEATURES.index(f)
            for sex in (0, 1, 2):
                mask = np.ones(len(X_test), dtype=bool) if sex == 0 else meta_test.SEX.to_numpy() == sex
                frame = pd.DataFrame({"value": X_test[f].to_numpy()[mask], "shap": values[mask, j]})
                for val, part in frame.groupby("value", sort=True):
                    effects.append({"model": kind, "feature": f, "SEX": sex, "value": val,
                        "records": len(part), "mean_signed_shap": part.shap.mean(),
                        "min_shap": part.shap.min(), "max_shap": part.shap.max()})
    artifacts["audit_predictions"] = pd.concat(predictions, ignore_index=True)
    artifacts["shap_importance_by_sex"] = pd.concat(all_importance, ignore_index=True).query("SEX != 0")
    full_importance = pd.concat(all_importance, ignore_index=True)
    artifacts["sex_shap_comparison"] = group_comparison(full_importance)
    artifacts["safe_rge_values"] = pd.concat(rge_rows, ignore_index=True)
    artifacts["safe_grouped_rge"] = pd.concat(grouped_rows, ignore_index=True)
    artifacts["categorical_shap_effects"] = pd.DataFrame(effects)
    artifacts["shap_validation"] = pd.DataFrame(checks)
    comparison = full_importance.query("SEX == 0").merge(artifacts["safe_rge_values"], on=["model", "feature"], validate="one_to_one")
    comparison["rge_rank_minus_shap_rank"] = comparison.rge_rank - comparison.shap_rank
    artifacts["shap_vs_rge"] = comparison
    artifacts["rge_model_comparison"] = artifacts["safe_rge_values"].pivot(index="feature", columns="model", values="RGE").reset_index()
    summaries = []
    for kind in MODELS:
        part = comparison[comparison.model == kind].sort_values("shap_rank")
        groups = artifacts["sex_shap_comparison"].query("model == @kind")
        summaries.append({"model": kind, "shap_rge_spearman": spearmanr(part.shap_rank, part.rge_rank).statistic,
            "top1_shap_share": part.importance_share.iloc[0], "top3_shap_share": part.importance_share.iloc[:3].sum(),
            "top5_shap_share": part.importance_share.iloc[:5].sum(),
            "sex_rank_spearman": spearmanr(groups.shap_rank_male, groups.shap_rank_female).statistic,
            "sex_importance_share_total_variation": groups.absolute_share_gap.sum() / 2,
            "total_mean_abs_shap": part.mean_abs_shap.sum()})
    artifacts["model_summary"] = pd.DataFrame(summaries)
    selected = select_local_cases(artifacts["audit_predictions"])
    artifacts["local_case_selection"] = selected
    local = []
    for kind in MODELS:
        values, base = shap_data[kind]
        for identifier in sorted(selected.ID.unique()):
            pos = int(np.flatnonzero(meta_test.ID.to_numpy() == identifier)[0])
            row = artifacts["audit_predictions"].query("model == @kind and ID == @identifier").iloc[0]
            for j, f in enumerate(FEATURES):
                local.append({"model": kind, "ID": int(identifier), "SEX": int(row.SEX), "actual": int(row.actual),
                    "predicted": int(row.predicted), "probability_default": row.probability_default,
                    "base_value": base[pos], "feature": f, "value": X_test.iloc[pos][f], "shap_value": values[pos, j]})
    artifacts["local_shap_contributions"] = pd.DataFrame(local)
    for name, frame in artifacts.items():
        finite(name, frame.select_dtypes(include="number").to_numpy())
        frame.to_csv(TABLE_DIR / f"{name}.csv", index=False, float_format="%.17g")
    from src.explainability_report import create_figures, write_report
    create_figures(artifacts, shap_data, X_test, meta_test)
    write_report(artifacts)
    if protected_hashes() != before:
        raise RuntimeError("Protected artifacts changed during explainability")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "stage": "Explainability",
        "protected_artifact_hashes": before, "protected_artifacts_unchanged": True,
        "seed": SEED, "background_size": BACKGROUND_SIZE,
        "background_method": "default_rng(42), uniform without replacement from training rows sorted by ID",
        "holdout_rows": len(X_test), "training_rows": len(X_train), "threshold": .50,
        "predictors": FEATURES, "shap_units": "default log-odds",
        "shap_formulation": "interventional on transformed features; signed one-hot sums by original feature",
        "safe": safe_source, "safe_groups": GROUPS,
        "versions": {p: version(p) for p in ("shap", "safeaipackage", "numpy", "pandas", "scipy", "scikit-learn", "numba", "llvmlite", "slicer", "tqdm", "matplotlib")},
        "source_hashes": {name: sha256(ROOT / name) for name in SOURCES},
        "output_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in [*sorted(TABLE_DIR.glob("*.csv")), *sorted(FIGURE_DIR.glob("*")), REPORT]}}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(artifacts["model_summary"].to_string(index=False), flush=True)
    return manifest


def verify_completion():
    first = run_explainability_evaluation()
    second = run_explainability_evaluation()
    csvs = {name: value for name, value in first["output_hashes"].items() if name.endswith(".csv")}
    if any(second["output_hashes"][name] != value for name, value in csvs.items()):
        raise AssertionError("Explainability CSVs differ on rerun")
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, capture_output=True, text=True)
    log = tests.stdout + tests.stderr
    print(log, flush=True)
    (TABLE_DIR / "test_results.txt").write_text(log, encoding="utf-8")
    if tests.returncode:
        raise RuntimeError("Project tests failed")
    dependency_check = subprocess.run([sys.executable, "-m", "pip", "check"], cwd=ROOT, capture_output=True, text=True)
    if dependency_check.returncode:
        raise RuntimeError(dependency_check.stdout + dependency_check.stderr)
    result = {"two_full_runs_csvs_byte_identical": True, "csv_count": len(csvs),
        "protected_artifacts_unchanged": protected_hashes() == first["protected_artifact_hashes"],
        "all_project_tests_passed": True, "tests_passed": int(re.search(r"Ran (\d+) tests", log).group(1)),
        "pip_check_passed": True, "test_log_sha256": sha256(TABLE_DIR / "test_results.txt"),
        "manifest_sha256": sha256(TABLE_DIR / "run_manifest.json"),
        "checks": ["complete holdout", "training-only background", "SEX counts", "finite outputs", "SHAP additivity", "actual SAFE calls", "unchanged predictions"]}
    (TABLE_DIR / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
