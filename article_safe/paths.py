"""Candidate model paths restricted to the saved training development partition."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from article_safe.artifacts import existing_run, verify_preparation, write_json, stage_manifest
from article_safe.data import load_development_data, check_evaluation_artifacts
from article_safe.explainability import background, explain, importance
from article_safe.inference import paired_delong, likelihood_ratio
from article_safe.models import fit_design, fit_lr, fit_gbm
from article_safe.selection import choose_backward_removal, choose_forward_prefix


def backward_path(X, y, protocol, destination):
    """Likelihood decisions use F only; no validation or holdout outcomes enter."""
    design = fit_design(X, protocol["features"], reference_encoding=True)
    design.mapping.to_csv(destination / "lr_encoded_mapping.csv", index=False)
    current = tuple(protocol["features"])
    cache = {}
    diagnostics, history = [], []

    def get_fit(features):
        key = tuple(features)
        if key not in cache:
            model = fit_lr(design, y, key, protocol)
            cache[key] = model
            diagnostics.append({"fit_id": len(diagnostics), "features": list(key), **model.diagnostics})
        return cache[key]

    initial = get_fit(current)
    if not initial.estimable:
        return {"status": "not_estimable", "selected_features": None,
                "reason": initial.diagnostics["reason"]}, initial, pd.DataFrame(history), diagnostics
    step = 0
    while current:
        current_model = get_fit(current)
        comparisons = []
        for feature in current:
            reduced_features = tuple(f for f in current if f != feature)
            reduced_model = get_fit(reduced_features)
            if reduced_model.estimable:
                tested = likelihood_ratio(current_model.diagnostics["log_likelihood"], reduced_model.diagnostics["log_likelihood"],
                                          current_model.diagnostics["design_rank"], reduced_model.diagnostics["design_rank"], len(y), len(y))
            else:
                tested = {"p_value": None, "statistic": None, "df": None, "status": "not_estimable", "reason": reduced_model.diagnostics["reason"]}
            comparisons.append({"feature": feature, **tested})
        choice = choose_backward_removal(comparisons, current, protocol["selection"]["alpha"])
        history.extend({"step": step, "current_predictors": len(current), "removed": choice.get("feature") == row["feature"], **row} for row in comparisons)
        print(f"LR backward step {step}: {len(current)} predictors; {choice}", flush=True)
        # Keep progress even if a numerical obstacle is later encountered.
        pd.DataFrame(history).to_csv(destination / "lr_removal_history.csv", index=False)
        write_json(destination / "lr_fit_diagnostics.json", diagnostics)
        if choice["status"] != "complete":
            return {"status": "not_estimable", "selected_features": None, "last_valid_features": list(current),
                    "reason": choice["reason"]}, current_model, pd.DataFrame(history), diagnostics
        if choice["action"] == "stop":
            break
        current = tuple(f for f in current if f != choice["feature"])
        retained_model = cache[current]
        cache.clear()
        cache[current] = retained_model
        step += 1
    result = {"status": "complete", "selected_features": list(current), "stopping_step": step,
              "reason": "intercept_only" if not current else "all_removal_p_below_alpha"}
    return result, get_fit(current), pd.DataFrame(history), diagnostics


def forward_path(X_fit, y_fit, metadata_fit, X_validation, y_validation, protocol, destination):
    features = protocol["features"]
    full = fit_gbm(X_fit, y_fit, features, protocol)
    reference, reference_meta = background(X_fit, metadata_fit, protocol["shap"]["background_size"], protocol["seed"])
    reference_meta.to_csv(destination / "selection_shap_background.csv", index=False)
    explained = explain(full, X_fit, reference, features)
    importance_table = importance(explained["original"], features)
    importance_table.to_csv(destination / "gbm_fitting_importance.csv", index=False)
    explained["mapping"].to_csv(destination / "gbm_selection_encoded_mapping.csv", index=False)
    np.savez_compressed(destination / "gbm_fitting_shap.npz", original=explained["original"], encoded=explained["encoded"], base_value=explained["base_value"])
    ordering = importance_table.feature.tolist()
    predictions, history, comparisons = {}, [], []
    previous = None
    for k in range(1, len(ordering) + 1):
        model = full if k == len(ordering) else fit_gbm(X_fit, y_fit, ordering[:k], protocol)
        probability = model.probability(X_validation)
        predictions[f"k_{k}"] = probability
        auc = float(roc_auc_score(y_validation, probability))
        row = {"k": k, "features": "|".join(ordering[:k]), "auc": auc,
               "gain": None, "p_value": None, "comparison_status": "initial_model", "fit_seconds": model.diagnostics["seconds"]}
        if previous is not None:
            tested = paired_delong(y_validation, probability, previous)
            row.update(gain=tested.difference, p_value=tested.p_value, comparison_status=tested.status)
            comparisons.append({"k": k, "gain": tested.difference, "p_value": tested.p_value, "status": tested.status})
        history.append(row)
        previous = probability
        pd.DataFrame(history).to_csv(destination / "gbm_prefix_path.csv", index=False)
        print(f"GBM prefix {k}/22: validation AUC={auc:.6f}", flush=True)
    choice = choose_forward_prefix(comparisons, protocol["selection"]["alpha"])
    selected = ordering[:choice["selected_k"]] if choice["selected_k"] is not None else None
    result = {**choice, "selected_features": selected, "feature_order": ordering,
              "max_shap_additivity_error": explained["max_additivity_error"]}
    return result, pd.DataFrame(history), predictions


def select(run_id):
    path = existing_run(run_id)
    protocol = verify_preparation(path)
    stage = path / "selection"
    if stage.exists():
        raise FileExistsError("Selection artifacts already exist; preserve them and use a fresh run.")
    check_evaluation_artifacts()
    development = load_development_data(protocol)
    X_fit, y_fit, metadata_fit = development.fitting()
    X_validation, y_validation, metadata_validation = development.selection()
    stage.mkdir()
    with threadpool_limits(limits=1):
        lr_choice, lr_model, lr_history, diagnostics = backward_path(X_fit, y_fit, protocol, stage)
        write_json(stage / "lr_choice.json", lr_choice)
        write_json(stage / "lr_fit_diagnostics.json", diagnostics)
        if len(lr_history):
            lr_history.to_csv(stage / "lr_removal_history.csv", index=False)
        if lr_model.estimable:
            joblib.dump(lr_model, stage / "lr_fitting_last_model.joblib")
        gbm_choice, _, probabilities = forward_path(X_fit, y_fit, metadata_fit, X_validation, y_validation, protocol, stage)
    predictions = metadata_validation[["ID", "SEX", "default", "duplicate_group_id"]].copy()
    for name, values in probabilities.items():
        predictions[name] = values
    predictions.to_csv(stage / "gbm_validation_predictions.csv", index=False)
    write_json(stage / "gbm_choice.json", gbm_choice)
    choices = {"lr_mle": lr_choice, "gbm": gbm_choice,
               "selection_data": "F likelihood / V paired AUC; no H outcomes",
               "final_model_refit_started": False}
    write_json(stage / "selected_subsets.json", choices)
    stage_manifest(stage, "selection", protocol, {"holdout_evaluated": False,
                   "prior_artifact_checks": check_evaluation_artifacts(),
                   "selection_status": {name: choices[name]["status"] for name in ("lr_mle", "gbm")}})
    print(json.dumps(choices, indent=2), flush=True)
    return stage