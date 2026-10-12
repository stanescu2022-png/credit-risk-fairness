"""Training-background explanations and the explicitly interpreted outcome regression."""
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from scipy.stats import norm, t
from statsmodels.stats.sandwich_covariance import cov_cluster
from threadpoolctl import threadpool_limits

from article_safe.inference import holm_adjust


def dense(values):
    return values.toarray() if hasattr(values, "toarray") else np.asarray(values)


def background(X, metadata, size=100, seed=42):
    if not X.index.equals(metadata.index) or set(metadata["split"]) != {"train"}:
        raise ValueError("SHAP background must be aligned saved training rows.")
    order = np.argsort(metadata.ID.to_numpy(), kind="stable")
    positions = order[np.random.default_rng(seed).choice(len(order), size=min(size, len(order)), replace=False)]
    return X.iloc[positions].copy(), metadata.iloc[positions].copy()


def explain(model, X, reference, feature_order):
    encoded, reference_encoded = dense(model.encoded(X)), dense(model.encoded(reference))
    if not model.estimable:
        raise ValueError("Cannot explain an unestimable estimator.")
    if model.family == "lr_mle":
        coefficients = np.asarray(model.estimator.params)[1:]
        means = reference_encoded.mean(axis=0)
        contributions = (encoded - means) * coefficients
        base = float(model.estimator.params[0] + means @ coefficients)
    else:
        masker = shap.maskers.Independent(reference_encoded, max_samples=len(reference_encoded))
        explainer = shap.TreeExplainer(model.estimator, data=masker, model_output="raw", feature_perturbation="interventional")
        np.testing.assert_array_equal(explainer.masker.data, reference_encoded)
        contributions = np.asarray(explainer.shap_values(encoded, check_additivity=True))
        base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    if contributions.shape != encoded.shape or not np.isfinite(contributions).all():
        raise ValueError("Unexpected or nonfinite SHAP result.")
    original = np.zeros((len(X), len(feature_order)))
    positions = {feature: i for i, feature in enumerate(feature_order)}
    for encoded_column, feature in enumerate(model.active_mapping.feature):
        original[:, positions[feature]] += contributions[:, encoded_column]
    margin = np.asarray(model.margin(X))
    error = float(np.max(np.abs(base + contributions.sum(axis=1) - margin)))
    np.testing.assert_allclose(base + original.sum(axis=1), margin, atol=1e-6, rtol=1e-7)
    return {"original": original, "encoded": contributions, "base_value": base,
            "mapping": model.active_mapping, "max_additivity_error": error, "units": "default log-odds"}


def importance(values, feature_order):
    values = np.asarray(values)
    mean_abs = np.mean(np.abs(values), axis=0)
    total = float(mean_abs.sum())
    result = pd.DataFrame({"feature": feature_order, "mean_abs_shap": mean_abs,
                           "mean_signed_shap": values.mean(axis=0),
                           "importance_share": mean_abs / total if total else np.nan,
                           "original_order": np.arange(len(feature_order))})
    return result.sort_values(["mean_abs_shap", "original_order"], ascending=[False, True], kind="stable").reset_index(drop=True)


def _coefficient_table(parameters, covariance, terms, reference, df=None):
    variances = np.diag(covariance)
    rows = []
    for index, (term, coefficient, variance) in enumerate(zip(terms, parameters, variances)):
        estimable = bool(np.isfinite(coefficient) and np.isfinite(variance) and variance > 0)
        se = float(np.sqrt(variance)) if estimable else np.nan
        statistic = float(coefficient / se) if estimable else np.nan
        p = float(2 * (norm.sf(abs(statistic)) if df is None else t.sf(abs(statistic), df))) if estimable else np.nan
        rows.append({"term": term, "coefficient": float(coefficient), "standard_error": se,
                     "statistic": statistic, "p_value": p, "reference_distribution": reference,
                     "reference_df": df, "status": "ok" if estimable else "not_estimable"})
    table = pd.DataFrame(rows)
    table["p_value_holm"] = np.nan
    if len(table) > 1:
        table.loc[table.index[1:], "p_value_holm"] = holm_adjust(table.p_value.iloc[1:].tolist())
    return table


def lr_parameters(model, groups):
    result = model.estimator
    terms = ["Intercept"] + model.active_mapping.term.tolist()
    tables = []
    for name, covariance in (("model_based", np.asarray(result.cov_params())),
                             ("original_group_cluster", cov_cluster(result, np.asarray(groups), use_correction=True))):
        table = _coefficient_table(np.asarray(result.params), covariance, terms, "normal z")
        table["covariance"] = name
        table["odds_ratio"] = np.exp(np.minimum(table.coefficient.to_numpy(), np.log(np.finfo(float).max)))
        table["post_selection"] = False  # The caller sets this for the selected role.
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def auxiliary_regression(y, original_values, active_features, feature_order, groups):
    """Observed binary outcome on contributions; never the additive model margin."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("The auxiliary response must be observed binary outcomes, not model margins.")
    indices = [list(feature_order).index(feature) for feature in active_features]
    matrix = np.c_[np.ones(len(y)), np.asarray(original_values)[:, indices]]
    if not np.isfinite(matrix).all() or len(groups) != len(y):
        raise ValueError("Invalid or misaligned auxiliary-regression data.")
    with threadpool_limits(limits=1):
        rank = int(np.linalg.matrix_rank(matrix))
        if rank != matrix.shape[1] or len(y) <= rank:
            return None, {"status": "not_estimable", "reason": "rank_deficient_auxiliary_design", "rank": rank}
        result = sm.OLS(y, matrix).fit()
        if np.max(np.abs(result.resid)) < 1e-12:
            return None, {"status": "not_estimable", "reason": "degenerate_exact_reconstruction"}
        groups_count = len(np.unique(groups))
        tables = []
        terms = ["Intercept"] + list(active_features)
        for name, covariance, df in (("model_based", result.cov_params(), result.df_resid),
                                     ("original_group_cluster", cov_cluster(result, np.asarray(groups), use_correction=True), groups_count - 1)):
            table = _coefficient_table(result.params, np.asarray(covariance), terms, "Student t", df)
            table["covariance"] = name
            table["response"] = "observed default"
            table["interpretation"] = "conditional linear association with outcome; not explanation correctness"
            tables.append(table)
    return pd.concat(tables, ignore_index=True), {"status": "ok", "rank": rank, "records": len(y), "predictive_candidate": False}