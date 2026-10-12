"""Article statistical primitives, with explicit degenerate-result handling.

DeLong is nominal independent-record inference. The covariance is paired and
uses positive/negative placements. Output differences always mean A minus B.
"""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.stats import chi2, ks_2samp, norm

EPS = np.finfo(float).eps
TINY = np.finfo(float).tiny
ROUND_MULTIPLIER = 64


@dataclass(frozen=True)
class AUCComparison:
    auc_a: float | None
    auc_b: float | None
    difference: float | None
    variance: float | None
    statistic: float | None
    p_value: float | None
    covariance: list | None
    positives: int
    negatives: int
    status: str
    variance_tolerance: float | None = None
    difference_tolerance: float = ROUND_MULTIPLIER * EPS

    def to_dict(self):
        return asdict(self)


def _vector(values, name, allow_empty=False):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or (not allow_empty and not len(array)) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite one-dimensional sample.")
    return array


def midranks(values):
    """One-based average ranks; exact score ties get identical ranks."""
    values = _vector(values, "scores", allow_empty=True)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        sorted_ranks[start:end] = (start + end + 1) / 2
        start = end
    result = np.empty_like(sorted_ranks)
    result[order] = sorted_ranks
    return result


def paired_delong(y, scores_a, scores_b):
    """Paired two-sided DeLong test; scores need not be probabilities."""
    y = _vector(y, "outcomes")
    a, b = _vector(scores_a, "scores A"), _vector(scores_b, "scores B")
    if len(y) != len(a) or len(y) != len(b) or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected aligned binary outcomes and score vectors.")
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    m, n = len(positive), len(negative)
    if not m or not n:
        return AUCComparison(None, None, None, None, None, None, None, m, n, "missing_outcome_class")
    ordered = np.vstack([a, b])[:, np.r_[positive, negative]]
    all_ranks = np.vstack([midranks(row) for row in ordered])
    positive_ranks = np.vstack([midranks(row[:m]) for row in ordered])
    negative_ranks = np.vstack([midranks(row[m:]) for row in ordered])
    aucs = all_ranks[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    difference = float(aucs[0] - aucs[1])
    base = [float(aucs[0]), float(aucs[1]), difference]
    if m < 2 or n < 2:
        return AUCComparison(*base, None, None, None, None, m, n, "insufficient_class_support")
    positive_placements = (all_ranks[:, :m] - positive_ranks) / n
    negative_placements = 1 - (all_ranks[:, m:] - negative_ranks) / m
    covariance = np.cov(positive_placements, ddof=1) / m + np.cov(negative_placements, ddof=1) / n
    variance = float(covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1])
    tolerance = ROUND_MULTIPLIER * EPS * max(float(covariance[0, 0] + covariance[1, 1]), TINY)
    if variance < -tolerance:
        raise ArithmeticError("Paired DeLong variance is materially negative.")
    matrix = covariance.tolist()
    if abs(variance) <= tolerance:
        if abs(difference) <= ROUND_MULTIPLIER * EPS:
            return AUCComparison(*base, 0.0, 0.0, 1.0, matrix, m, n, "identical_result", tolerance)
        return AUCComparison(*base, 0.0, None, None, matrix, m, n, "degenerate_variance", tolerance)
    statistic = difference / np.sqrt(variance)
    p = float(2 * norm.sf(abs(statistic)))
    return AUCComparison(*base, variance, float(statistic), p, matrix, m, n, "ok", tolerance)


def likelihood_ratio(current_ll, reduced_ll, current_rank, reduced_rank,
                     current_nobs, reduced_nobs, *, penalized=False):
    """Nested binomial LR statistic; same row identities must also be checked by caller."""
    result = {"statistic": None, "df": None, "p_value": None, "status": "not_estimable", "reason": None}
    if penalized:
        result["reason"] = "penalized_likelihood_not_supported"
        return result
    if current_nobs != reduced_nobs or current_nobs <= 0:
        result["reason"] = "different_or_empty_observation_sets"
        return result
    if any(not np.isfinite(value) for value in (current_ll, reduced_ll, current_rank, reduced_rank)):
        result["reason"] = "nonfinite_fit"
        return result
    if any(int(rank) != rank or rank < 1 for rank in (current_rank, reduced_rank)):
        raise ValueError("Design ranks must be positive integers including the intercept.")
    df = int(current_rank - reduced_rank)
    result["df"] = df
    if df <= 0:
        result["reason"] = "nonpositive_rank_difference"
        return result
    statistic = float(2 * (current_ll - reduced_ll))
    tolerance = ROUND_MULTIPLIER * EPS * max(abs(current_ll), abs(reduced_ll), 1.0)
    if statistic < -tolerance:
        result["reason"] = "reduced_likelihood_exceeds_current"
        return result
    statistic = max(statistic, 0.0)
    result.update(statistic=statistic, p_value=float(chi2.sf(statistic, df)), status="ok", reason=None)
    return result


def holm_adjust(p_values, planned_family_size=None):
    """Missing tests stay missing; internal p=1 padding retains the planned family."""
    values = np.asarray([np.nan if value is None else value for value in p_values], dtype=float)
    if values.ndim != 1 or np.isinf(values).any():
        raise ValueError("Expected a one-dimensional p-value family.")
    finite = np.isfinite(values)
    if ((values[finite] < 0) | (values[finite] > 1)).any():
        raise ValueError("P-values must be in [0,1].")
    size = len(values) if planned_family_size is None else planned_family_size
    if int(size) != size or size < len(values):
        raise ValueError("The planned family must include every supplied test.")
    size = int(size)
    if not size:
        return values.copy()
    padded = np.ones(size)
    padded[:len(values)] = np.where(finite, values, 1.0)
    order = np.argsort(padded, kind="stable")
    adjusted_ordered = np.minimum(1.0, np.maximum.accumulate(padded[order] * (size - np.arange(size))))
    adjusted = np.empty(size)
    adjusted[order] = adjusted_ordered
    result = adjusted[:len(values)]
    result[~finite] = np.nan
    return result


def concentration_gini(magnitudes):
    """Equal-group standard Gini; no measured magnitude means undefined."""
    a = _vector(magnitudes, "group magnitudes")
    if len(a) < 2 or (a < 0).any():
        raise ValueError("At least two nonnegative group magnitudes are required.")
    if not np.any(a):
        return {"gini": None, "normalized_gini": None, "shares": None, "status": "no_measured_contribution"}
    # Scaling before normalization avoids overflow/underflow in the concentration formula.
    scaled = a / np.max(a)
    shares = scaled / scaled.sum()
    gini = float(np.abs(shares[:, None] - shares[None, :]).sum() / (2 * len(a)))
    return {"gini": gini, "normalized_gini": gini / (1 - 1 / len(a)), "shares": shares.tolist(), "status": "ok"}


def signed_share_audit(group_means, atol=1e-12, rtol=1e-10):
    """Preserve the signed formula's ingredients without treating it as a valid Gini."""
    values = _vector(group_means, "signed group means")
    if len(values) < 2 or atol < 0 or rtol < 0:
        raise ValueError("Expected at least two groups and nonnegative tolerances.")
    denominator = float(values.sum())
    tolerance = float(atol + rtol * np.abs(values).sum())
    cancelled = abs(denominator) <= tolerance
    shares = None if cancelled else values / denominator
    return {"signed_means": values.tolist(), "denominator": denominator,
            "cancellation_tolerance": tolerance, "cancellation": bool(cancelled),
            "negative_effect": bool((values < 0).any()),
            "negative_share": None if cancelled else bool((shares < 0).any()),
            "shares": None if cancelled else shares.tolist()}


def contribution_ks(female, male):
    """KS compares per-record signed contributions, never two group means."""
    female = _vector(female, "female contributions", allow_empty=True)
    male = _vector(male, "male contributions", allow_empty=True)
    result = {"statistic": None, "p_value": None, "status": "not_estimable",
              "female_records": len(female), "male_records": len(male),
              "inference": "nominal independent continuous samples; ties/clustering not calibrated"}
    if not len(female) or not len(male):
        result["reason"] = "empty_group"
        return result
    joined = np.r_[female, male]
    result["female_tie_fraction"] = 1 - len(np.unique(female)) / len(female)
    result["male_tie_fraction"] = 1 - len(np.unique(male)) / len(male)
    result["pooled_tie_fraction"] = 1 - len(np.unique(joined)) / len(joined)
    tested = ks_2samp(female, male, alternative="two-sided", method="asymp")
    result["statistic"] = float(tested.statistic)
    if np.isfinite(tested.pvalue):
        result.update(p_value=float(tested.pvalue), status="ok")
    else:
        result["reason"] = "nominal_p_value_unavailable"
    return result


def group_row_weights(groups, rng):
    """Sample whole original groups; a group crossing SEX/outcome stays together."""
    groups = np.asarray(groups)
    if groups.ndim != 1 or not len(groups):
        raise ValueError("Expected a nonempty group vector.")
    _, inverse = np.unique(groups, return_inverse=True)
    count = int(inverse.max() + 1)
    sampled = rng.integers(0, count, size=count)
    return np.bincount(sampled, minlength=count)[inverse]