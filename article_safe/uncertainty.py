"""Conditional paired uncertainty using shared whole-predictor-group draws."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits


class FixedAUC:
    """Weighted Mann-Whitney AUC for fixed scores; tie groups are cached once."""
    def __init__(self, y, scores):
        y, scores = np.asarray(y), np.asarray(scores)
        if not len(y) or y.shape != scores.shape or y.ndim != 1 or not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
            raise ValueError("Expected aligned finite scores and binary outcomes.")
        self.order = np.argsort(scores, kind="stable")
        ordered = scores[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(ordered)) + 1]
        self.y = y[self.order]

    def __call__(self, weights):
        weights = np.asarray(weights, dtype=float)
        if weights.shape != self.order.shape or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Weights must be aligned, finite and nonnegative.")
        weights = weights[self.order]
        positive = np.add.reduceat(weights * self.y, self.starts)
        negative = np.add.reduceat(weights * (1 - self.y), self.starts)
        p, n = positive.sum(), negative.sum()
        if not p or not n:
            return np.nan
        lower_negative = np.cumsum(negative) - negative
        return float(np.dot(positive, lower_negative + .5 * negative) / (p * n))


def conditional_intervals(y, probabilities, contributions, sex, groups, feature_order, protocol, destination):
    y, sex, groups = np.asarray(y), np.asarray(sex), np.asarray(groups)
    models = list(probabilities)
    auc = {model: FixedAUC(y, probabilities[model]) for model in models}
    for model in models:
        np.testing.assert_allclose(auc[model](np.ones(len(y))), roc_auc_score(y, probabilities[model]), atol=1e-12)
    _, inverse = np.unique(groups, return_inverse=True)
    group_count = int(inverse.max() + 1)
    repetitions = protocol["inference"]["bootstrap_repetitions"]
    minimum_valid = protocol["inference"]["minimum_valid_bootstrap"]
    rng = np.random.default_rng(protocol["inference"]["bootstrap_seed"])
    pairs = protocol["final_comparisons"]
    auc_differences = np.full((repetitions, len(pairs)), np.nan)
    explain_models = list(contributions)
    if explain_models:
        matrix = np.concatenate([contributions[name] for name in explain_models], axis=1)
        both = np.c_[matrix, np.abs(matrix)]
        gaps = np.full((repetitions, both.shape[1]), np.nan)
    else:
        gaps = np.empty((repetitions, 0))
    with threadpool_limits(limits=1):
        for start in range(0, repetitions, 32):
            count = min(32, repetitions - start)
            weights = np.empty((count, len(y)), dtype=float)
            for offset in range(count):
                sampled = rng.integers(0, group_count, size=group_count)
                row_weights = np.bincount(sampled, minlength=group_count)[inverse]
                weights[offset] = row_weights
                estimates = {model: calculator(row_weights) for model, calculator in auc.items()}
                for j, (first, second) in enumerate(pairs):
                    if first in estimates and second in estimates:
                        auc_differences[start + offset, j] = estimates[first] - estimates[second]
            if explain_models:
                female_w, male_w = weights * (sex == 2), weights * (sex == 1)
                female_n, male_n = female_w.sum(axis=1), male_w.sum(axis=1)
                valid = (female_n > 0) & (male_n > 0)
                chunk = np.full((count, both.shape[1]), np.nan)
                chunk[valid] = (female_w[valid] @ both) / female_n[valid, None] - (male_w[valid] @ both) / male_n[valid, None]
                gaps[start:start + count] = chunk
            if start % 320 == 0:
                print(f"Conditional group bootstrap: {start + count}/{repetitions}", flush=True)
    rows = []

    def interval(values):
        valid = values[np.isfinite(values)]
        return (float(np.quantile(valid, .025)), float(np.quantile(valid, .975)), len(valid)) if len(valid) >= minimum_valid else (np.nan, np.nan, len(valid))

    for j, (first, second) in enumerate(pairs):
        low, high, count = interval(auc_differences[:, j])
        rows.append({"comparison": first + " minus " + second, "ci_lower": low, "ci_upper": high, "valid_draws": count,
                     "attempted_draws": repetitions, "status": "ok" if count >= minimum_valid else "not_estimable"})
    pd.DataFrame(rows).to_csv(destination / "auc_difference_intervals.csv", index=False)
    gap_rows = []
    width = len(feature_order)
    if explain_models:
        for i, model in enumerate(explain_models):
            for j, feature in enumerate(feature_order):
                for label, column in (("mean_signed", i * width + j), ("mean_absolute", len(explain_models) * width + i * width + j)):
                    low, high, count = interval(gaps[:, column])
                    gap_rows.append({"model": model, "feature": feature, "quantity": label, "gap_sign": "female minus male",
                                     "ci_lower": low, "ci_upper": high, "valid_draws": count, "attempted_draws": repetitions,
                                     "status": "ok" if count >= minimum_valid else "not_estimable"})
    pd.DataFrame(gap_rows).to_csv(destination / "explanation_gap_intervals.csv", index=False)
    np.savez_compressed(destination / "bootstrap_replicates.npz", auc_differences=auc_differences, explanation_gaps=gaps,
                        auc_pair_names=np.array([a + " minus " + b for a,b in pairs]), explanation_models=np.array(explain_models),
                        features=np.array(feature_order))
    return {"repetitions": repetitions, "original_groups": group_count, "paired_draws": True,
            "refitting_or_reselection": False, "sampling_unit": "whole original predictor group", "seed": protocol["seed"]}