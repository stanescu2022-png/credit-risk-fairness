"""Shared whole-original-group resampling of fixed prediction tables only."""
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .pareto import evaluate_candidates, validate_candidates
from .tables import PredictionTable, immutable_array, require, validate_alignment


@dataclass(frozen=True)
class BootstrapPlan:
    """Explicit group draw counts; all rows of a group share its multiplicity.

    Draw G groups with replacement from G observed groups and retain all their
    records. Groups crossing SEX/outcome remain intact; no stratification or
    per-group row normalization is performed. Supplied group IDs must already
    represent whole-original-predictor groups, even for reduced candidates.
    """
    reference: PredictionTable
    group_multiplicities: np.ndarray
    seed: int | None
    group_order: tuple = field(init=False)
    row_group_positions: np.ndarray = field(init=False)

    def __post_init__(self):
        require(isinstance(self.reference, PredictionTable), "Bootstrap reference must be a validated prediction table.")
        groups = tuple(dict.fromkeys(self.reference.original_group))
        positions = {group: i for i, group in enumerate(groups)}
        counts = np.asarray(self.group_multiplicities)
        require(counts.ndim == 2 and counts.shape[0] > 0 and counts.shape[1] == len(groups),
                "Bootstrap group multiplicities require nonempty draws x original groups.")
        require(counts.dtype.kind in "iu" and (counts >= 0).all() and (counts <= len(groups)).all(),
                "Group multiplicities must be nonnegative integers no larger than G.")
        require((counts.sum(axis=1) == len(groups)).all(), "Each draw must sample exactly G original groups with replacement.")
        require(self.seed is None or (isinstance(self.seed, (int, np.integer)) and not isinstance(self.seed, bool) and self.seed >= 0),
                "Seed must be an explicit nonnegative integer or None for supplied counts.")
        object.__setattr__(self, "group_order", groups)
        object.__setattr__(self, "group_multiplicities", immutable_array(counts, np.int64))
        object.__setattr__(self, "row_group_positions", immutable_array(
            [positions[group] for group in self.reference.original_group], np.int64))

    @property
    def n_draws(self):
        return len(self.group_multiplicities)

    def row_multiplicities(self, draw):
        require(isinstance(draw, (int, np.integer)) and not isinstance(draw, bool) and 0 <= draw < self.n_draws,
                "Bootstrap draw index out of range.")
        return self.group_multiplicities[draw, self.row_group_positions]

    def evidence(self):
        return {"seed": None if self.seed is None else int(self.seed), "draw_count": self.n_draws,
                "group_order": list(self.group_order), "group_multiplicities": self.group_multiplicities.tolist(),
                "row_group_positions": self.row_group_positions.tolist(), "row_ids": list(self.reference.ids),
                "role_label": self.reference.role_label,
                "resampling_unit": "whole supplied original-predictor group, retaining all crossing SEX/outcome records",
                "estimator": "record-weighted metrics on resampled groups, not equal-weight group-average losses"}


def group_bootstrap_plan(reference, *, n_draws, seed):
    """No production defaults: draw count and seed must be explicitly supplied."""
    require(isinstance(n_draws, (int, np.integer)) and not isinstance(n_draws, bool) and n_draws > 0,
            "n_draws must be an explicit positive integer.")
    require(isinstance(seed, (int, np.integer)) and not isinstance(seed, bool) and seed >= 0,
            "seed must be an explicit nonnegative integer.")
    require(isinstance(reference, PredictionTable), "Bootstrap reference must be validated.")
    count = len(set(reference.original_group))
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, count, size=(n_draws, count))
    multiplicities = np.asarray([np.bincount(draw, minlength=count) for draw in sampled], dtype=np.int64)
    return BootstrapPlan(reference, multiplicities, int(seed))


def percentile_summary(values, *, percentile_bounds):
    """Marginal linear-interpolated percentile interval; explicit valid counts."""
    try:
        bounds = tuple(float(value) for value in percentile_bounds)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Supply two ordered numeric percentile bounds in [0,100].") from error
    require(len(bounds) == 2 and all(np.isfinite(x) for x in bounds)
            and 0 <= bounds[0] <= bounds[1] <= 100, "Supply two ordered percentile bounds in [0,100].")
    values = tuple(values)
    valid = []
    for value in values:
        if value is not None:
            require(np.isscalar(value) and np.isfinite(value), "Undefined bootstrap values must be None, not NaN/inf.")
            valid.append(float(value))
    if valid:
        lower, upper = np.percentile(valid, bounds, method="linear")
        median = float(np.median(valid))
    else:
        lower = upper = median = None
    return {"lower": float(lower) if lower is not None else None,
            "upper": float(upper) if upper is not None else None, "median": median,
            "valid_draws": len(valid), "undefined_draws": len(values)-len(valid), "total_draws": len(values),
            "percentile_bounds": list(bounds), "percentile_method": "numpy linear",
            "scope": "marginal conditional fixed-prediction interval; not joint confidence certification"}


def bootstrap_with_plan(candidates, config, plan, *, percentile_bounds):
    """Use the same group multiplicities for every candidate/scenario/repetition.

    Intervals condition on supplied fixed models, choices, scenarios and outputs;
    they omit fitting/selection uncertainty. Repetitions are averaged within
    each group draw. This does not resample Monte Carlo repetition indices.
    """
    reference = validate_candidates(candidates, config)
    require(isinstance(plan, BootstrapPlan), "Supply an explicit BootstrapPlan.")
    validate_alignment((reference, plan.reference), role_label=config.role_label)
    # Validate bounds before executing draws, including when all rates vanish.
    percentile_bounds = tuple(percentile_bounds)
    percentile_summary([], percentile_bounds=percentile_bounds)
    point = evaluate_candidates(candidates, config)
    candidate_names = [name for name in config.candidate_order if name in candidates]
    metric_names = sorted({metric for values in point["candidate_metrics"].values() for metric in values})
    draws = {name: {metric: [] for metric in metric_names} for name in candidate_names}
    differences = {metric: [] for metric in metric_names}
    statuses, pareto_statuses = [], []
    frontier_counts = {name: 0 for name in config.candidate_order}
    complete_frontier_draws = 0
    for index in range(plan.n_draws):
        evaluated = evaluate_candidates(candidates, config, multiplicities=plan.row_multiplicities(index))
        vectors = evaluated["candidate_metrics"]
        for name in candidate_names:
            for metric in metric_names:
                draws[name][metric].append(vectors[name].get(metric))
        for metric in metric_names:
            our = vectors[config.our_choice_id].get(metric)
            safe = vectors[config.safe_choice_id].get(metric)
            differences[metric].append(our-safe if our is not None and safe is not None else None)
        comparison = evaluated["choice_comparison"]
        statuses.append(comparison["status"])
        pareto_statuses.append(comparison["pareto_status"])
        frontier = evaluated["pool_frontier"]
        if not frontier["incomplete_pool"]:
            complete_frontier_draws += 1
            for name in frontier["frontier_ids"]:
                frontier_counts[name] += 1
    status_counts, pareto_counts = Counter(statuses), Counter(pareto_statuses)
    valid = plan.n_draws-pareto_counts["unresolved"]
    return {"point": point, "bootstrap_plan": plan.evidence(),
            "candidate_metric_intervals": {name: {metric: percentile_summary(values, percentile_bounds=percentile_bounds)
                                                   for metric, values in samples.items()} for name, samples in draws.items()},
            "difference_name": "our_choice_minus_SAFE_choice",
            "paired_difference_intervals": {metric: percentile_summary(values, percentile_bounds=percentile_bounds)
                                             for metric, values in differences.items()},
            "dominance_stability": {"label": "exploratory conditional resampling stability; not joint confidence certification",
                "pair_status_counts": dict(status_counts), "pareto_status_counts": dict(pareto_counts),
                "resolved_draws": valid, "unresolved_draws": pareto_counts["unresolved"], "total_draws": plan.n_draws,
                "our_choice_dominates_fraction_resolved": pareto_counts["left_dominates"]/valid if valid else None,
                "SAFE_choice_dominates_fraction_resolved": pareto_counts["right_dominates"]/valid if valid else None,
                "same_choice": config.our_choice_id == config.safe_choice_id},
            "frontier_stability": {"label": "exploratory membership frequency among complete-pool draws only",
                "complete_pool_draws": complete_frontier_draws, "incomplete_pool_draws": plan.n_draws-complete_frontier_draws,
                "candidate_membership_counts": frontier_counts,
                "candidate_membership_fraction_complete_draws": {name: count/complete_frontier_draws if complete_frontier_draws else None
                                                                for name, count in frontier_counts.items()}},
            "draw_evidence": {"candidate_metrics": draws, "our_choice_minus_SAFE_choice": differences,
                              "pair_statuses": statuses, "pareto_statuses": pareto_statuses},
            "limitations": ["conditional on fixed supplied predictions and explicit selector choices",
                            "no fitting, selection, scenario-generation or calibration uncertainty",
                            "scenario repetition spread is separate; repetitions are not bootstrapped here",
                            "marginal intervals and dominance frequencies are not joint confidence certification"]}


def bootstrap_judge(candidates, config, *, n_draws, seed, percentile_bounds):
    """Convenience wrapper with no seed/draw-count/interval defaults."""
    reference = validate_candidates(candidates, config)
    plan = group_bootstrap_plan(reference, n_draws=n_draws, seed=seed)
    return bootstrap_with_plan(candidates, config, plan, percentile_bounds=percentile_bounds)
