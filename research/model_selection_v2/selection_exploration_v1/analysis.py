"""Outcome-risk screens and conditional uncertainty on supplied predictions."""
import itertools

import numpy as np

from research.model_selection_v2.judge_core import PredictionTable, pareto_relation, percentile_summary, validate_alignment
from research.model_selection_v2.judge_core.metrics import LOGLOSS_EPSILON
from research.model_selection_v2.judge_core.tables import row_multiplicities
from .protocol import CONDITIONS, FAMILIES, RISK_AXES, SAFE_TIE_TOLERANCE, scenario_name

LOSSES = ("brier", "log_loss")
SCENARIOS = ("clean",) + tuple(scenario_name(family, condition) for family in FAMILIES for condition in CONDITIONS)
MARGINALS = tuple(scenario_name(family, condition) for family in FAMILIES for condition in ("10", "01"))
JOINTS = tuple(scenario_name(family, "11") for family in FAMILIES)
SELECTORS = ("safe_arithmetic", "safe_geometric", "safe_quadratic", "clean_brier", "clean_logloss",
             "worst_brier", "worst_logloss", "screened_safe_clean", "screened_safe_marginal", "screened_safe_outcome_v0")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pointwise_losses(outcome, probability):
    p, y = np.asarray(probability, dtype=float), np.asarray(outcome)
    require(y.ndim == 1 and np.isin(y, [0, 1]).all() and p.shape[-1:] == y.shape, "Loss outcome/prediction alignment mismatch.")
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Loss inputs require finite probabilities in [0,1].")
    clipped = np.clip(p, LOGLOSS_EPSILON, 1-LOGLOSS_EPSILON)
    return {"brier": (p-y)**2,
            "log_loss": -np.where(y == 1, np.log(clipped), np.log1p(-clipped))}


def endpoint_counts(probability):
    p = np.asarray(probability, dtype=float)
    return {"records": int(p.size), "probability_zero": int((p == 0).sum()), "probability_one": int((p == 1).sum()),
            "clipped_below_epsilon": int((p < LOGLOSS_EPSILON).sum()),
            "clipped_above_one_minus_epsilon": int((p > 1-LOGLOSS_EPSILON).sum())}


def loss_cache(predictions, reference, candidate_order):
    """Precompute per-record means of losses; preserve repetition loss/SD separately."""
    order = tuple(candidate_order)
    require(set(predictions) == set(order) and len(set(order)) == len(order), "Require exactly the declared candidate pool.")
    require(isinstance(reference, PredictionTable), "Loss cache requires a validated identity-bound reference table.")
    y = reference.outcome
    require(y.ndim == 1 and len(y) > 0 and np.isin(y, [0, 1]).all(), "Require nonempty binary outcomes.")
    cache = {"candidate_order": order, "records": len(y), "reference": reference,
             "record_means": {}, "repetitions": {}, "endpoint_counts": {}}
    repetition_counts = {}
    for candidate in order:
        require(set(predictions[candidate]) == set(SCENARIOS), "Require clean and all nine specified scenarios, with no substitutions.")
        cache["record_means"][candidate], cache["repetitions"][candidate], cache["endpoint_counts"][candidate] = {}, {}, {}
        for scenario in SCENARIOS:
            probability = np.asarray(predictions[candidate][scenario], dtype=float)
            require(probability.ndim == 2 and probability.shape[1] == len(y) and probability.shape[0] > 0,
                    "Predictions require repetition x record arrays.")
            require(scenario != "clean" or probability.shape[0] == 1, "Clean probabilities occur exactly once.")
            if candidate == order[0] and scenario == "clean":
                np.testing.assert_array_equal(probability[0], reference.probability)
            if scenario in repetition_counts:
                require(repetition_counts[scenario] == len(probability), "Repetition counts must agree across candidates.")
            repetition_counts[scenario] = len(probability)
            values = pointwise_losses(y, probability)
            cache["record_means"][candidate][scenario] = {name: vector.mean(axis=0) for name, vector in values.items()}
            repetition = {name: vector.mean(axis=1) for name, vector in values.items()}
            cache["repetitions"][candidate][scenario] = {name: {"values": vector.tolist(), "mean": float(vector.mean()),
                "mc_sd": float(np.std(vector, ddof=1)) if len(vector) > 1 else None,
                "mc_scope": "between-realization sample SD; separate from group sampling uncertainty"}
                for name, vector in repetition.items()}
            cache["endpoint_counts"][candidate][scenario] = {
                "per_repetition": [endpoint_counts(row) for row in probability], "aggregate": endpoint_counts(probability),
                "convention": f"LogLoss only: float64 clip to [eps,1-eps], eps={LOGLOSS_EPSILON}"}
    require(len({count for scenario, count in repetition_counts.items() if scenario != "clean"}) == 1,
            "Every disturbed scenario must use the same number of realizations.")
    return cache


def mean_losses(cache, multiplicities=None):
    weights = row_multiplicities(cache["reference"], multiplicities)
    return {candidate: {scenario: {loss: float(np.average(values, weights=weights)) for loss, values in metrics.items()}
                        for scenario, metrics in scenarios.items()} for candidate, scenarios in cache["record_means"].items()}


def finite(value):
    try:
        return value is not None and np.isscalar(value) and bool(np.isfinite(value)) and not isinstance(value, (str, bytes, complex))
    except (ValueError, TypeError):
        return False


def maximum(values):
    if not values or not all(finite(value) for value in values.values()):
        return {"value": None, "attaining_scenarios": [], "status": "unresolved"}
    highest = max(values.values())
    return {"value": highest, "attaining_scenarios": [name for name, value in values.items() if value == highest], "status": "defined"}


def risk_vectors(means, candidate_order):
    vectors, maxima = {}, {}
    for candidate in candidate_order:
        scenarios = means.get(candidate, {})
        value = lambda scenario, loss: scenarios.get(scenario, {}).get(loss)
        vectors[candidate] = {"clean_brier": value("clean", "brier"), "clean_logloss": value("clean", "log_loss")}
        maxima[candidate] = {}
        for kind, names in (("marginal", MARGINALS), ("joint", JOINTS), ("worst", SCENARIOS)):
            for loss in LOSSES:
                result = maximum({name: value(name, loss) for name in names})
                suffix = "logloss" if loss == "log_loss" else loss
                maxima[candidate][f"{kind}_max_{suffix}"] = result
                if kind != "worst":
                    vectors[candidate][f"{kind}_max_{suffix}"] = result["value"]
    return vectors, maxima


def unresolved(reason):
    return {"status": "unresolved", "selected_id": None, "tied_ids": [], "retained_ids": [], "reason": reason}


def choose_scores(scores, candidate_order, *, minimize, tolerance=0.):
    order = tuple(candidate_order)
    if not order or any(not finite(scores.get(name)) for name in order):
        return unresolved("required candidate score missing or nonfinite")
    target = min(scores[name] for name in order) if minimize else max(scores[name] for name in order)
    tied = [name for name in order if (scores[name] == target if tolerance == 0 else abs(scores[name]-target) <= tolerance)]
    return {"status": "selected", "selected_id": tied[0], "tied_ids": tied, "retained_ids": list(order), "reason": None}


def screen(vectors, candidate_order, axes, safe_arithmetic):
    order = tuple(candidate_order)
    directions = {axis: "min" for axis in axes}
    if any(any(not finite(vectors.get(name, {}).get(axis)) for axis in axes) for name in order):
        return unresolved("required Pareto coordinate missing or nonfinite; no candidate/axis dropped")
    retained = [name for name in order if not any(other != name and
                pareto_relation(vectors[other], vectors[name], metric_directions=directions)["status"] == "left_dominates"
                for other in order)]
    chosen = choose_scores(safe_arithmetic, retained, minimize=False, tolerance=SAFE_TIE_TOLERANCE)
    chosen["retained_ids"] = retained
    return chosen


def selector_decisions(means, safe_scores, candidate_order):
    order = tuple(candidate_order)
    require(bool(order) and len(set(order)) == len(order), "Explicit unique candidate order required.")
    vectors, maxima = risk_vectors(means, order)
    decisions = {}
    for aggregation in ("arithmetic", "geometric", "quadratic"):
        decisions["safe_"+aggregation] = choose_scores({name: safe_scores.get(name, {}).get(aggregation) for name in order},
                                                      order, minimize=False, tolerance=SAFE_TIE_TOLERANCE)
    for loss, suffix in (("brier", "brier"), ("log_loss", "logloss")):
        decisions["clean_"+suffix] = choose_scores({name: vectors[name]["clean_"+suffix] for name in order}, order, minimize=True)
        decisions["worst_"+suffix] = choose_scores({name: maxima[name]["worst_max_"+suffix]["value"] for name in order}, order, minimize=True)
    arithmetic = {name: safe_scores.get(name, {}).get("arithmetic") for name in order}
    for name, axes in (("screened_safe_clean", RISK_AXES[:2]), ("screened_safe_marginal", RISK_AXES[:4]),
                       ("screened_safe_outcome_v0", RISK_AXES)):
        decisions[name] = screen(vectors, order, axes, arithmetic)
    return {"risk_vectors": vectors, "maxima": maxima, "decisions": decisions,
            "primary_evaluation": None, "scope": "exploratory fixed-pool selector definitions; no independent superiority evidence"}


def interactions(means, candidate_order):
    return {candidate: {family: {loss: means[candidate][scenario_name(family, "11")][loss]
        -means[candidate][scenario_name(family, "10")][loss]-means[candidate][scenario_name(family, "01")][loss]
        +means[candidate]["clean"][loss] for loss in LOSSES} for family in FAMILIES} for candidate in candidate_order}


def repetition_interactions(cache):
    result = {}
    for candidate in cache["candidate_order"]:
        result[candidate] = {}
        for family in FAMILIES:
            result[candidate][family] = {}
            for loss in LOSSES:
                values = cache["repetitions"][candidate]
                contrast = np.asarray(values[scenario_name(family, "11")][loss]["values"])
                contrast = contrast-np.asarray(values[scenario_name(family, "10")][loss]["values"])
                contrast = contrast-np.asarray(values[scenario_name(family, "01")][loss]["values"])+values["clean"][loss]["mean"]
                result[candidate][family][loss] = {"values": contrast.tolist(), "mean": float(contrast.mean()),
                    "mc_sd": float(np.std(contrast, ddof=1)) if len(contrast) > 1 else None,
                    "contrast": "L11-L10-L01+L00; paired factorial description, not causal"}
    return result


def bootstrap_loss_matrix(cache, plan):
    """Group sums of record-averaged losses, identical to direct repetition averaging."""
    validate_alignment((cache["reference"], plan.reference), role_label=cache["reference"].role_label)
    keys = [(candidate, scenario, loss) for candidate in cache["candidate_order"] for scenario in SCENARIOS for loss in LOSSES]
    sums = np.column_stack([np.bincount(plan.row_group_positions,
                           weights=cache["record_means"][candidate][scenario][loss], minlength=len(plan.group_order))
                            for candidate, scenario, loss in keys])
    sizes = np.bincount(plan.row_group_positions, minlength=len(plan.group_order))
    counts = plan.group_multiplicities.astype(float)
    denominator = counts @ sizes
    matrix = (counts @ sums)/denominator[:, None]
    return keys, matrix


def conditional_bootstrap(cache, plan, safe_scores, *, percentile_bounds):
    """Re-screen outcome layer; SAFE scores remain fixed in every draw."""
    order = cache["candidate_order"]
    percentile_bounds = tuple(percentile_bounds)
    percentile_summary([], percentile_bounds=percentile_bounds)
    keys, matrix = bootstrap_loss_matrix(cache, plan)
    column = {key: matrix[:, i] for i, key in enumerate(keys)}
    draw_interactions = {(candidate, family, loss): column[candidate, scenario_name(family, "11"), loss]
        -column[candidate, scenario_name(family, "10"), loss]-column[candidate, scenario_name(family, "01"), loss]
        +column[candidate, "clean", loss] for candidate in order for family in FAMILIES for loss in LOSSES}
    summary = lambda values: percentile_summary(values, percentile_bounds=percentile_bounds)
    frequency = {selector: {"selected_counts": {name: 0 for name in order}, "retained_counts": {name: 0 for name in order},
                            "tied_choice_membership_counts": {name: 0 for name in order}, "multiple_choice_tie_draws": 0,
                            "valid_draws": 0, "unresolved_draws": 0} for selector in SELECTORS}
    attaining = {candidate: {} for candidate in order}
    for index in range(plan.n_draws):
        means = {candidate: {scenario: {loss: column[candidate, scenario, loss][index] for loss in LOSSES}
                             for scenario in SCENARIOS} for candidate in order}
        selected = selector_decisions(means, safe_scores, order)
        for selector, decision in selected["decisions"].items():
            count = frequency[selector]
            if decision["status"] == "selected":
                count["valid_draws"] += 1
                count["selected_counts"][decision["selected_id"]] += 1
                for name in decision["tied_ids"]:
                    count["tied_choice_membership_counts"][name] += 1
                count["multiple_choice_tie_draws"] += int(len(decision["tied_ids"]) > 1)
                for name in decision["retained_ids"]:
                    count["retained_counts"][name] += 1
            else:
                count["unresolved_draws"] += 1
        for candidate in order:
            for axis, maximum_info in selected["maxima"][candidate].items():
                axis_counts = attaining[candidate].setdefault(axis, {})
                for name in maximum_info["attaining_scenarios"]:
                    axis_counts[name] = axis_counts.get(name, 0)+1
    for count in frequency.values():
        valid = count["valid_draws"]
        count["selection_fraction_valid_draws"] = {name: n/valid if valid else None for name, n in count["selected_counts"].items()}
        count["retention_fraction_valid_draws"] = {name: n/valid if valid else None for name, n in count["retained_counts"].items()}
        count["total_draws"] = plan.n_draws
    paired = {}
    for earlier, later in itertools.combinations(order, 2):
        pair = f"{later}_minus_{earlier}"
        paired[pair] = {"direction": {"later": later, "earlier": earlier},
            "loss_intervals": {scenario: {loss: summary(column[later, scenario, loss]-column[earlier, scenario, loss])
                               for loss in LOSSES} for scenario in SCENARIOS},
            "interaction_difference_intervals": {family: {loss: summary(draw_interactions[later, family, loss]-draw_interactions[earlier, family, loss])
                                                 for loss in LOSSES} for family in FAMILIES}}
    return {"scope": "exploratory added-outcome-layer resampling conditional on fixed saved SAFE scores; not complete selector uncertainty",
            "joint_confidence_certification": False, "draws": plan.n_draws,
            "candidate_loss_intervals": {candidate: {scenario: {loss: summary(column[candidate, scenario, loss]) for loss in LOSSES}
                                         for scenario in SCENARIOS} for candidate in order},
            "candidate_interaction_intervals": {candidate: {family: {loss: summary(draw_interactions[candidate, family, loss]) for loss in LOSSES}
                                                for family in FAMILIES} for candidate in order},
            "paired_fixed_candidate_intervals": paired, "selector_frequencies": frequency,
            "retention_definition": "Pareto-retained for screened selectors; complete supplied pool for unscreened baselines",
            "choice_tie_definition": "all exact minimum-loss ties or declared 1e-12 SAFE-score ties counted; selected ID uses fixed pilot order",
            "maximizer_membership_counts": attaining, "maximizer_counts_note": "all exact maximizers counted; tied memberships can sum above draw count",
            "monte_carlo_repetition_spread": "reported separately; no repetition resampling",
            "safe_scores_held_fixed": True}
