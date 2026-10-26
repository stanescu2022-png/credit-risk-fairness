"""SAFE 2026 equations with explicitly specified Taiwan operational choices."""
from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

from research.model_selection_v2.pilot import require, finite_inputs, NUMERIC, CATEGORICAL

TOL = 1e-12


@dataclass
class RankResult:
    value: float | None
    denominator: float
    terms: np.ndarray
    reason: str | None = None
    roundoff_terms_zeroed: int = 0


def vector(values):
    result = np.asarray(values, dtype=float)
    require(result.ndim == 1 and len(result) > 0, "Require nonempty one-dimensional values.")
    require(np.isfinite(result).all(), "Nonfinite score inputs; records cannot be dropped.")
    return result


def rank_graduation(reference, ordering):
    """Common Eq. (4)/(8)/(12) kernel, computed from full cumulative areas.

    Exact comparison-score ties are averaged, i.e. expected concordance over
    all orders of tied records. Constant references remain explicitly undefined.
    Centering preserves the ratio and avoids cancellation from constant offsets.
    """
    reference, ordering = vector(reference), vector(ordering)
    require(reference.shape == ordering.shape, "Rank input alignment mismatch.")
    require((reference >= 0).all(), "Reference values must be nonnegative.")
    if np.ptp(reference) == 0:
        return RankResult(None, 0., np.full(len(reference), np.nan), "constant reference: zero Lorenz-area denominator")
    centered = reference - reference.min()
    order = np.argsort(ordering, kind="stable")
    score, ref = ordering[order], centered[order]
    _, first, counts = np.unique(score, return_index=True, return_counts=True)
    # Summing sorted within-block values makes floating tie support independent
    # of incoming row order, without using outcome order to break score ties.
    support = np.concatenate([np.full(count, np.sort(ref[start:start + count]).sum() / count)
                              for start, count in zip(first, counts)])
    ascending = np.sort(centered)
    denominator = float(np.cumsum(ascending[::-1] - ascending).sum())
    require(denominator > 0 and np.isfinite(denominator), "Invalid nonconstant-reference denominator.")
    terms = np.cumsum(ascending[::-1] - support) / denominator
    tiny = (terms < 0) & (terms >= -TOL)
    count = int(tiny.sum())
    require((terms >= -TOL).all(), "Substantial negative concordance area term.")
    terms[tiny] = 0.
    if abs(terms[-1]) <= TOL:
        count += int(terms[-1] != 0)
        terms[-1] = 0.
    value = float(terms.sum())
    require(-TOL <= value <= 1 + TOL, "Rank graduation outside mathematical bounds.")
    return RankResult(value, denominator, terms, roundoff_terms_zeroed=count)


def accuracy_curve(outcome, probability, segments=22):
    result = rank_graduation(outcome, probability)
    require(1 <= segments <= len(result.terms), "Invalid segment count.")
    pieces = np.array_split(np.arange(len(result.terms) - 1, -1, -1), segments)
    counts = np.array([len(piece) for piece in pieces])
    x = np.r_[0., np.cumsum(counts) / len(result.terms)]
    if result.value is None:
        return {"scalar_rga": None, "status": "undefined", "reason": result.reason,
                "x": x.tolist(), "values": [None] * (segments + 1),
                "partial_contributions": [None] * segments, "raw_step_area": None}
    partial = np.array([result.terms[piece].sum() for piece in pieces])
    values = np.r_[result.value, result.value - np.cumsum(partial)]
    require(abs(values[-1]) <= TOL, "RGA contribution accounting failed.")
    require((values >= -TOL).all(), "Substantial negative cumulative RGA value.")
    roundoff_curve_values = int(((values < 0) & (values >= -TOL)).sum())
    values[(values < 0) & (values >= -TOL)] = 0.
    values[-1] = 0.  # Only numerical zero, after the accounting assertion.
    np.testing.assert_allclose(partial.sum(), result.value, rtol=0, atol=TOL)
    return {"scalar_rga": result.value, "status": "defined", "reason": None,
            "x": x.tolist(), "values": values.tolist(), "partial_contributions": partial.tolist(),
            "segment_rank_cell_counts": counts.tolist(),
            "full_sample_area_terms_low_to_high_rank": result.terms.tolist(),
            "full_sample_denominator": result.denominator,
            "roundoff_terms_zeroed": result.roundoff_terms_zeroed,
            "roundoff_curve_values_zeroed": roundoff_curve_values,
            "raw_step_area": float(np.dot(np.diff(x), values[:-1])),
            "uniform_node_mean": float(values.mean()),
            "perfect_reference_area_normalization": False}


def tail_swap(X, ids, strength):
    """Coupled, original-column swaps; ID only breaks equal-value pairing ties."""
    finite_inputs(X)
    ids = np.asarray(ids)
    require(len(ids) == len(X) and len(set(ids)) == len(ids), "Swap IDs must be unique and aligned.")
    require(0 <= strength <= .5 and np.isfinite(strength), "Raw severity must be in [0,.5].")
    m = int(np.floor(strength * len(X)))
    result = X.copy(deep=True)
    exposure = {}
    for name in X.columns:
        original = X[name].to_numpy()
        order = np.lexsort((ids, original))
        low, high = order[:m], order[::-1][:m]
        changed = original.copy()
        changed[low], changed[high] = original[high], original[low]
        result[name] = changed
        np.testing.assert_array_equal(np.sort(original), np.sort(changed))
        exposure[name] = {"paired_records": 2 * m, "changed_records": int(np.count_nonzero(changed != original))}
    return result, {"raw_strength": float(strength), "normalized_strength": float(2 * strength),
                    "tail_size_per_feature": m, "per_feature": exposure,
                    "records_with_any_change": int((result.to_numpy() != X.to_numpy()).any(axis=1).sum()),
                    "changed_cells": int((result.to_numpy() != X.to_numpy()).sum()),
                    "category_multisets_preserved": True, "row_order_preserved": result.index.equals(X.index)}


def fitting_references(fitting):
    finite_inputs(fitting)
    require(set(fitting.columns) == set(NUMERIC + CATEGORICAL), "Reference predictor schema mismatch.")
    result = {name: float(fitting[name].mean()) for name in NUMERIC}
    for name in CATEGORICAL:
        result[name] = float(fitting[name].mode().min())
    return result


def mask_original(X, removed, references):
    require(len(set(removed)) == len(removed) and set(removed) <= set(X.columns), "Invalid removal units.")
    require(set(references) == set(X.columns), "Mask reference schema mismatch.")
    result = X.copy(deep=True)
    for name in removed:
        result[name] = references[name]
    return result


def prediction_hash(probability):
    return hashlib.sha256(np.asarray(probability, dtype="<f8").tobytes()).hexdigest()


def greedy_preservation(feature_order, full_probability, predict_removed):
    """True nested least-damaging search; callback performs prediction only.

    Return chosen-path prediction evidence and every evaluated alternative.
    An undefined alternative prevents an honest maximum; do not skip it.
    """
    feature_order = list(feature_order)
    require(len(feature_order) == len(set(feature_order)) > 0, "Removal units must be unique.")
    full_probability = vector(full_probability)
    initial = rank_graduation(full_probability, full_probability)
    if initial.value is None:
        return {"status": "undefined", "reason": initial.reason, "path": [], "alternatives": [],
                "values": [None], "predictions": [full_probability]}
    removed, remaining, alternatives, path = [], feature_order.copy(), [], []
    values, predictions = [initial.value], [full_probability.copy()]
    for step in range(1, len(feature_order) + 1):
        evaluated = []
        for name in remaining:
            subset = tuple(removed + [name])
            probability = vector(predict_removed(subset))
            require(probability.shape == full_probability.shape, "Masked prediction alignment changed.")
            score = rank_graduation(full_probability, probability)
            item = {"step": step, "removed_before": removed.copy(), "alternative": name,
                    "removed_subset": list(subset), "reversed_rge": score.value,
                    "original_rge": 1 - score.value if score.value is not None else None,
                    "reason": score.reason, "prediction_sha256": prediction_hash(probability)}
            evaluated.append((item, probability))
        alternatives.extend(item for item, _ in evaluated)
        if any(item["reversed_rge"] is None for item, _ in evaluated):
            return {"status": "undefined", "reason": "undefined greedy alternative", "path": path,
                    "alternatives": alternatives, "values": values, "predictions": predictions}
        maximum = max(item["reversed_rge"] for item, _ in evaluated)
        tied = [(item, probability) for item, probability in evaluated
                if maximum - item["reversed_rge"] <= TOL]
        chosen, probability = tied[0]  # Remaining stays in saved feature order.
        chosen = dict(chosen, tied_alternatives=[item["alternative"] for item, _ in tied])
        path.append(chosen)
        values.append(chosen["reversed_rge"])
        predictions.append(probability.copy())
        removed.append(chosen["alternative"])
        remaining.remove(chosen["alternative"])
    return {"status": "defined", "reason": None, "path": path, "alternatives": alternatives,
            "values": values, "predictions": predictions,
            "x": np.linspace(0, 1, len(values)).tolist()}


def curve_summary(x, values):
    x = vector(x)
    require(len(x) == len(values) and np.all(np.diff(x) > 0), "Invalid curve coordinates.")
    if any(value is None for value in values):
        return {"trapezoidal_area": None, "uniform_node_mean": None, "increases": None}
    y = vector(values)
    return {"trapezoidal_area": float(np.trapezoid(y, x)), "uniform_node_mean": float(y.mean()),
            "increases": np.flatnonzero(np.diff(y) > TOL).tolist()}


def integrate(A, E, R):
    if any(value is None for v in (A, E, R) for value in v):
        return {"status": "undefined", "arithmetic": None, "geometric": None, "quadratic": None}
    A, E, R = vector(A), vector(E), vector(R)
    require(len(A) == len(E) == len(R), "Published construction requires equal vector lengths.")
    for v in (A, E, R):
        require(((v >= -TOL) & (v <= 1 + TOL)).all(), "Component outside mathematical range.")
        require((v >= 0).all(), "Negative raw curve values cannot enter geometric means.")
    a, e, r = A[:, None, None], E[None, :, None], R[None, None, :]
    result = {"status": "defined", "arithmetic": float(((a + e + r) / 3).mean()),
              "geometric": float(np.cbrt(a * e * r).mean()),
              "quadratic": float(np.sqrt((a*a + e*e + r*r) / 3).mean())}
    np.testing.assert_allclose(result["arithmetic"], (A.mean() + E.mean() + R.mean()) / 3, rtol=0, atol=TOL)
    return result


def rank_candidates(scores, candidate_order):
    """Unrankable remains distinct from low scoring; deterministic tie resolution."""
    available = [name for name in candidate_order if scores[name] is not None]
    remaining = available.copy()
    rows, rank = [], 1
    while remaining:
        maximum = max(scores[name] for name in remaining)
        tied = [name for name in remaining if maximum - scores[name] <= TOL]
        for name in tied:
            rows.append({"candidate_id": name, "score": scores[name], "rank": rank,
                         "tied_ids": tied.copy(), "status": "tied" if len(tied) > 1 else "ranked"})
        remaining = [name for name in remaining if name not in tied]
        rank += len(tied)
    rows.extend({"candidate_id": name, "score": None, "rank": None,
                 "tied_ids": [], "status": "unrankable"} for name in candidate_order if scores[name] is None)
    best = rows[0] if available else None
    return {"rows": rows, "selected_id": best["candidate_id"] if best else None,
            "tied_selected_ids": best["tied_ids"] if best else [],
            "status": ("tie_resolved_deterministically" if len(best["tied_ids"]) > 1 else "selected") if best else "unavailable"}
