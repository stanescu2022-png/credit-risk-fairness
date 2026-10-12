"""Fixed methodological choices and source traceability for the article run."""
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RELATIVE = "docs/context/sources/3. 1-s2.0-S0957417423017220-main.pdf"
SOURCE_SHA256 = "df63164a9c8beab73863dcd642a0a5724bb25853aa49048f0b6f21ccc2361f49"
SPEC_RELATIVE = "docs/plans/article_safe_specification_2026-10-10.md"
PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "shap", "statsmodels", "patsy")

FIDELITY = [
    {"component": "accuracy", "anchor": "2.1 / Figure 1", "method": "paired two-sided DeLong AUC", "status": "article_procedure"},
    {"component": "lr_reduction", "anchor": "2.2 / 3.1", "method": "whole-predictor backward likelihood ratios; largest p first", "status": "article_procedure", "convention": "removal order supplied by specification"},
    {"component": "gbm_reduction", "anchor": "2.2 / 4.1", "method": "SHAP-ordered forward inclusion; first nonpositive or nonsignificant gain stops", "status": "explicit_interpretation"},
    {"component": "importance", "anchor": "2.3 / 4.1", "method": "mean absolute original-predictor SHAP in log-odds", "status": "explicit_interpretation"},
    {"component": "lr_parameter_inference", "anchor": "Figure 1 / 2.3", "method": "binomial GLM Wald z; cluster-covariance sensitivity", "status": "explicit_interpretation"},
    {"component": "gbm_explanation_test", "anchor": "2.3 / 4.1", "method": "observed outcome on SHAP OLS; coefficient t tests", "status": "explicit_interpretation"},
    {"component": "gini", "anchor": "2.4 / Table 2", "method": "concentration of nonnegative effect magnitudes; lower is more equal", "status": "explicit_interpretation"},
    {"component": "gbm_ks", "anchor": "2.4 / 4.1", "method": "female versus male individual signed-contribution distributions", "status": "explicit_interpretation"},
    {"component": "lr_coefficient_ks", "anchor": "Figure 1 / 3.1", "method": None, "status": "not_identifiable_from_article"},
    {"component": "lr_contribution_ks", "anchor": "supporting convention", "method": "KS on individual linear contributions", "status": "supporting_diagnostic"},
    {"component": "conditional_uncertainty", "anchor": "project sampling convention", "method": "paired whole-original-predictor-group bootstrap", "status": "supporting_diagnostic"},
    {"component": "overall_safe_score", "anchor": "not supplied by original article", "method": None, "status": "not_identifiable_from_article"},
]

_DEFAULTS = {
    "protocol_version": "article_safe_v1",
    "source_doi": "10.1016/j.eswa.2023.121220",
    "primary_models": ["lr_mle", "gbm"],
    "final_roles": ["lr_mle_full", "lr_mle_selected", "gbm_full", "gbm_selected"],
    "estimator_reference": "lr_frozen_regularized",
    "metadata_only": ["ID", "SEX", "default"],
    "seed": 42,
    "threshold": 0.50,
    "decision_rule": "probability_default > 0.50; ties class 0",
    "outer_partition": {"training": 24004, "holdout": 5996, "historically_inspected": True},
    "development": {"splitter": "StratifiedGroupKFold", "n_splits": 5, "shuffle": True, "random_state": 42, "selection_fold": 1, "grouping": "exact original 22 predictors; minimum-ID label"},
    "selection": {"alpha": 0.05, "lr": "backward whole-block likelihood ratio; remove largest p >= alpha", "gbm": "fixed SHAP order; start k=1; require positive gain and paired p < alpha; first failure stops"},
    "lr": {"estimator": "statsmodels GLM Binomial logit", "regularized": False, "maxiter": 200, "tol": 1e-8, "fit_intercept": True, "class_weights": None, "categorical_encoding": "lowest observed reference; drop first", "unknown_categories": "reference coding; count and report"},
    "gbm": {"parameters_source": "models/baseline_gradient_boosting_manifest.json", "same_settings_for_subsets": True},
    "shap": {"background_size": 100, "background_seed": 42, "background_sampling": "uniform without replacement from allowed training rows sorted by ID", "units": "default log-odds", "gbm_perturbation": "interventional", "gbm_output": "raw", "approximate": False, "aggregate": "sum signed encoded contributions before taking absolute values"},
    "inference": {"delong": "paired two-sided; nominal independent-record inference", "roundoff_multiplier": 64, "final_family_size": 4, "multiplicity": "Holm; missing tests retained in family", "bootstrap_repetitions": 2000, "minimum_valid_bootstrap": 1900, "bootstrap_seed": 42, "bootstrap_unit": "whole original predictor group; no SEX/outcome strata", "bootstrap_refits": False},
    "fairness": {"female": 2, "male": 1, "signed_gap": "female minus male", "gini": "standard concentration; nonnegative magnitudes; equal group weights", "ks_alternative": "two-sided", "ks_method": "asymp", "ks_null": "same individual signed-contribution distribution across SEX groups", "ks_inference": "nominal independent continuous samples; ties and clustering disclosed", "signed_cancellation_atol": 1e-12, "signed_cancellation_rtol": 1e-10},
    "final_comparisons": [
        ["gbm_full", "lr_mle_full"], ["gbm_selected", "lr_mle_selected"],
        ["lr_mle_selected", "lr_mle_full"], ["gbm_selected", "gbm_full"],
    ],
    "comparison_sign": "first model minus second model",
    "excluded_scope": ["overall SAFE score", "synthetic information failure", "later integrated score", "RF arm", "RGA/RGE/RGR substitute", "PSI", "governance"],
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_protocol(root=ROOT):
    """Return a fresh config with verified source and baseline identities."""
    root = Path(root)
    source = root / SOURCE_RELATIVE
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("The original article PDF identity changed.")
    manifests = {}
    for kind in ("logistic", "gradient_boosting"):
        path = root / f"models/baseline_{kind}_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if sha256(root / f"models/baseline_{kind}.joblib") != manifest["model_sha256"]:
            raise ValueError(f"Frozen model identity changed: {kind}")
        manifests[kind] = manifest
    features = manifests["gradient_boosting"]["feature_columns"]
    from src.baseline import FEATURES
    if len(features) != 22 or len(set(features)) != 22 or set(features) != set(FEATURES):
        raise ValueError("Unexpected original-predictor schema.")
    if manifests["logistic"]["feature_columns"] != features:
        raise ValueError("Frozen model input orders differ.")
    protocol = deepcopy(_DEFAULTS)
    protocol["features"] = list(features)
    protocol["gbm"]["classifier_parameters"] = manifests["gradient_boosting"]["classifier_parameters"]
    protocol["source_pdf"] = {"path": SOURCE_RELATIVE, "sha256": SOURCE_SHA256}
    protocol["specification"] = {"path": SPEC_RELATIVE, "sha256": sha256(root / SPEC_RELATIVE)}
    protocol["fidelity"] = deepcopy(FIDELITY)
    protocol["environment"] = {name: version(name) for name in PACKAGES}
    protocol["baseline_manifests"] = {kind: sha256(root / f"models/baseline_{kind}_manifest.json") for kind in manifests}
    return protocol