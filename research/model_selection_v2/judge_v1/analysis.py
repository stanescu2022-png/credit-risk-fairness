"""Pure fixed-choice judge calculations; no data loading, fitting or selection."""
from collections import Counter
import itertools

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from research.model_selection_v2.judge_core import (
    JudgeConfig, PredictionTable, compare_choices, fairness_gaps, pareto_relation, percentile_summary, pool_frontier, validate_alignment,
)
from research.model_selection_v2.selection_exploration_v1.analysis import (
    JOINTS, LOSSES, SCENARIOS, bootstrap_loss_matrix, interactions, loss_cache, mean_losses, repetition_interactions,
)
from .io import require
from .protocol import CANDIDATES, COORDINATES, DIRECTIONS, FAMILIES, METHODS, ROLE, scenario_name

SIGNED = ("signed_tpr_gap", "signed_fpr_gap")


def table(metadata, probability):
    return PredictionTable(
        tuple(metadata["ID"]), np.asarray(metadata["default"]), np.asarray(metadata["SEX"]), tuple(metadata["duplicate_group_id"]),
        np.asarray(probability), ROLE)


def prepare_cache(predictions, metadata):
    require(set(predictions) == set(CANDIDATES), "Judge requires exactly four frozen candidate outputs.")
    clean_tables = {candidate: table(metadata, np.asarray(predictions[candidate]["clean"])[0]) for candidate in CANDIDATES}
    reference = validate_alignment(clean_tables.values(), role_label=ROLE)
    cache = loss_cache(predictions, reference, CANDIDATES)
    return cache, clean_tables


def derived_vectors(means, clean_tables, *, multiplicities=None):
    validate_alignment(clean_tables.values(), role_label=ROLE)
    vectors, gaps, maxima, fairness = {}, {}, {}, {}
    for candidate in CANDIDATES:
        measured = fairness_gaps(clean_tables[candidate], multiplicities=multiplicities)
        fairness[candidate] = measured
        gaps[candidate] = {"signed_tpr_gap": measured["tpr_gap_signed"], "signed_fpr_gap": measured["fpr_gap_signed"]}
        vectors[candidate] = {"clean_brier": means[candidate]["clean"]["brier"], "clean_logloss": means[candidate]["clean"]["log_loss"],
            "absolute_tpr_gap": measured["tpr_gap_absolute"], "absolute_fpr_gap": measured["fpr_gap_absolute"]}
        maxima[candidate] = {}
        for loss, coordinate in (("brier", "worst_joint_brier"), ("log_loss", "worst_joint_logloss")):
            values = {scenario: means[candidate][scenario][loss] for scenario in JOINTS}
            if any(value is None or not np.isfinite(value) for value in values.values()):
                highest, attaining = None, []
            else:
                highest = max(values.values())
                attaining = [scenario for scenario, value in values.items() if value == highest]
            vectors[candidate][coordinate] = highest
            maxima[candidate][coordinate] = {"value": highest, "attaining_scenarios": attaining}
    return vectors, gaps, maxima, fairness


def config_for(choices):
    require(set(choices) == set(METHODS) and all(candidate in CANDIDATES for candidate in choices.values()), "Incomplete frozen method mapping.")
    return JudgeConfig(DIRECTIONS, choices["screened_safe_outcome_v0"], choices["safe_arithmetic"], CANDIDATES, ROLE, None)


def differences(left, right):
    return {name: left[name]-right[name] if left.get(name) is not None and right.get(name) is not None else None for name in left}


def comparison_bundle(vectors, signed_gaps, choices):
    config = config_for(choices)
    principal = compare_choices(vectors, config)
    principal.update({"display_contrast": "screened_safe_outcome_v0 choice minus safe_arithmetic choice",
                      "primary_evaluation": None, "signed_gap_differences": differences(signed_gaps[config.our_choice_id], signed_gaps[config.safe_choice_id])})
    pairs = {}
    for earlier, later in itertools.combinations(CANDIDATES, 2):
        pairs[f"{later}_minus_{earlier}"] = {"left_candidate": later, "right_candidate": earlier,
            "relation": pareto_relation(vectors[later], vectors[earlier], metric_directions=DIRECTIONS),
            "coordinate_differences": differences(vectors[later], vectors[earlier]),
            "signed_gap_differences": differences(signed_gaps[later], signed_gaps[earlier])}
    return {"principal_contrast": principal, "fixed_candidate_pairs": pairs,
            "pairwise_matrix": {left: {right: {"same_candidate": left == right,
                **pareto_relation(vectors[left], vectors[right], metric_directions=DIRECTIONS)} for right in CANDIDATES} for left in CANDIDATES},
            "frontier": pool_frontier(vectors, config), "frozen_reference_method_to_candidate": dict(choices),
            "same_choice_method_groups": {candidate: [name for name in METHODS if choices[name] == candidate] for candidate in CANDIDATES},
            "same_choice_note": "methods choosing one candidate share predictions; different candidates can also have equal metric vectors"}


def reliability_bins(outcome, probability):
    y, p = np.asarray(outcome), np.asarray(probability, dtype=float)
    bins = np.minimum(np.floor(p*10).astype(int), 9)
    result = []
    for index in range(10):
        mask = bins == index
        count = int(mask.sum())
        result.append({"bin": index, "lower": index/10, "upper": (index+1)/10,
                       "interval": "left closed, right open; final bin includes 1", "count": count,
                       "positive_count": int(y[mask].sum()), "mean_probability": float(p[mask].mean()) if count else None,
                       "observed_rate": float(y[mask].mean()) if count else None})
    require(sum(row["count"] for row in result) == len(y), "Reliability bins lost records.")
    return result


def clean_diagnostics(clean_tables):
    result = {}
    for candidate, values in clean_tables.items():
        y, p = values.outcome, values.probability
        pred = p > .5
        tp, fp = int(((y == 1) & pred).sum()), int(((y == 0) & pred).sum())
        pos, neg = int((y == 1).sum()), int((y == 0).sum())
        fairness = fairness_gaps(values)
        for rates in fairness["rates"].values():
            rates["fn"] = rates["positive_denominator"]-rates["tp"]
            rates["tn"] = rates["negative_denominator"]-rates["fp"]
        result[candidate] = {"auc": float(roc_auc_score(y, p)) if pos and neg else None,
            "average_precision": float(average_precision_score(y, p)) if pos else None,
            "recall": tp/pos if pos else None, "fpr": fp/neg if neg else None,
            "confusion": {"tp": tp, "fp": fp, "fn": pos-tp, "tn": neg-fp},
            "fairness": fairness, "reliability_bins": reliability_bins(y, p), "decision_rule": "p > .5; ties class 0"}
    return result


def scenario_diagnostics(predictions, cache, means):
    result = {}
    for candidate in CANDIDATES:
        clean = np.asarray(predictions[candidate]["clean"])[0]
        result[candidate] = {}
        for scenario in SCENARIOS:
            p = np.asarray(predictions[candidate][scenario])
            movement = np.abs(p-clean).mean(axis=1)
            flips = ((p > .5) != (clean > .5)).mean(axis=1)
            result[candidate][scenario] = {"mean_losses": means[candidate][scenario],
                "loss_deterioration": {loss: means[candidate][scenario][loss]-means[candidate]["clean"][loss] for loss in LOSSES},
                "loss_repetitions": cache["repetitions"][candidate][scenario],
                "mean_absolute_probability_movement": {"values": movement.tolist(), "mean": float(movement.mean()),
                    "mc_sd": float(np.std(movement, ddof=1)) if len(movement) > 1 else None},
                "strict_classification_flip_rate": {"values": flips.tolist(), "mean": float(flips.mean()),
                    "mc_sd": float(np.std(flips, ddof=1)) if len(flips) > 1 else None}}
    return result


def point_assessment(predictions, metadata, choices):
    cache, clean_tables = prepare_cache(predictions, metadata)
    means = mean_losses(cache)
    vectors, signed, maxima, fairness = derived_vectors(means, clean_tables)
    point = {"primary_evaluation": None, "coordinates": vectors, "signed_gaps": signed, "maxima": maxima,
             "fairness": fairness, "scenario_matrix": means, "comparisons": comparison_bundle(vectors, signed, choices),
             "clean_diagnostics": clean_diagnostics(clean_tables), "scenario_diagnostics": scenario_diagnostics(predictions, cache, means),
             "endpoint_counts": cache["endpoint_counts"], "interactions": {"mean": interactions(means, CANDIDATES),
             "per_realization": repetition_interactions(cache), "interpretation": "interaction on loss scale; not causal/nonlinear model interaction"}}
    return point, cache, clean_tables


def _stability(statuses):
    counts = Counter(statuses)
    valid = len(statuses)-counts["unresolved"]
    return {"counts": dict(counts), "resolved_draws": valid, "unresolved_draws": counts["unresolved"], "total_draws": len(statuses),
            "fraction_resolved": {name: count/valid if valid else None for name, count in counts.items() if name != "unresolved"},
            "label": "exploratory fixed-choice relation stability; not true superiority probability or joint confidence certification"}


def bootstrap_assessment(cache, clean_tables, choices, plan, *, percentile_bounds):
    """Shared group counts, fixed method identities; no outcome-driven reselection."""
    config = config_for(choices)
    validate_alignment((*clean_tables.values(), plan.reference), role_label=ROLE)
    keys, loss_matrix = bootstrap_loss_matrix(cache, plan)
    columns = {key: loss_matrix[:, index] for index, key in enumerate(keys)}
    samples = {candidate: {name: [] for name in (*COORDINATES, *SIGNED)} for candidate in CANDIDATES}
    maximizers = {candidate: {coordinate: [] for coordinate in ("worst_joint_brier", "worst_joint_logloss")} for candidate in CANDIDATES}
    pair_status = {f"{later}_minus_{earlier}": [] for earlier, later in itertools.combinations(CANDIDATES, 2)}
    principal_status, principal_pareto, frontier_masks, complete_frontiers = [], [], [], []
    for draw in range(plan.n_draws):
        means = {candidate: {scenario: {loss: columns[candidate, scenario, loss][draw] for loss in LOSSES} for scenario in SCENARIOS}
                 for candidate in CANDIDATES}
        vectors, signed, maxima, _ = derived_vectors(means, clean_tables, multiplicities=plan.row_multiplicities(draw))
        for candidate in CANDIDATES:
            for name, value in {**vectors[candidate], **signed[candidate]}.items():
                samples[candidate][name].append(value)
            for coordinate in maximizers[candidate]:
                maximizers[candidate][coordinate].append([scenario in maxima[candidate][coordinate]["attaining_scenarios"] for scenario in JOINTS])
        for earlier, later in itertools.combinations(CANDIDATES, 2):
            pair_status[f"{later}_minus_{earlier}"].append(pareto_relation(vectors[later], vectors[earlier], metric_directions=DIRECTIONS)["status"])
        principal_comparison = compare_choices(vectors, config)
        principal_status.append(principal_comparison["status"])
        principal_pareto.append(principal_comparison["pareto_status"])
        frontier = pool_frontier(vectors, config)
        complete_frontiers.append(not frontier["incomplete_pool"])
        frontier_masks.append([candidate in frontier["frontier_ids"] for candidate in CANDIDATES])
    bounds = tuple(percentile_bounds)
    summarize = lambda values: percentile_summary(values, percentile_bounds=bounds)
    def paired(left, right):
        values = {name: [x-y if x is not None and y is not None else None for x, y in zip(samples[left][name], samples[right][name])]
                  for name in (*COORDINATES, *SIGNED)}
        return {"left_candidate": left, "right_candidate": right, "same_choice": left == right,
                "coordinate_and_signed_gap_intervals": {name: summarize(vector) for name, vector in values.items()},
                "loss_intervals": {scenario: {loss: summarize(columns[left, scenario, loss]-columns[right, scenario, loss]) for loss in LOSSES}
                                   for scenario in SCENARIOS}}
    contrasts = {(candidate, family, loss): columns[candidate, scenario_name(family, "11"), loss]-columns[candidate, scenario_name(family, "10"), loss]
        -columns[candidate, scenario_name(family, "01"), loss]+columns[candidate, "clean", loss]
        for candidate in CANDIDATES for family in FAMILIES for loss in LOSSES}
    pairs = {}
    for earlier, later in itertools.combinations(CANDIDATES, 2):
        name = f"{later}_minus_{earlier}"
        pairs[name] = {**paired(later, earlier), "relation_stability": _stability(pair_status[name]),
            "interaction_difference_intervals": {family: {loss: summarize(contrasts[later, family, loss]-contrasts[earlier, family, loss])
                                                for loss in LOSSES} for family in FAMILIES}}
    principal = paired(config.our_choice_id, config.safe_choice_id)
    principal.update({"display_contrast": "screened_safe_outcome_v0 choice minus safe_arithmetic choice",
                      "relation_stability": {**_stability(principal_pareto), "identity_status_counts": dict(Counter(principal_status))},
                      "interaction_difference_intervals": {family: {loss: summarize(contrasts[config.our_choice_id, family, loss]
                           -contrasts[config.safe_choice_id, family, loss]) for loss in LOSSES} for family in FAMILIES}})
    valid_frontier = np.asarray(complete_frontiers, dtype=bool)
    frontier_array = np.asarray(frontier_masks, dtype=bool)
    complete_count = int(valid_frontier.sum())
    evidence = {"loss_matrix": loss_matrix, "loss_key_candidate": np.asarray([key[0] for key in keys]),
                "loss_key_scenario": np.asarray([key[1] for key in keys]), "loss_key_name": np.asarray([key[2] for key in keys]),
                "frontier_membership": frontier_array, "complete_frontier": valid_frontier,
                "principal_pair_status": np.asarray(principal_status), "principal_pareto_status": np.asarray(principal_pareto)}
    for candidate in CANDIDATES:
        for name, vector in samples[candidate].items():
            evidence[f"{candidate}___{name}"] = np.asarray([value if value is not None else np.nan for value in vector], dtype=float)
            evidence[f"{candidate}___{name}___valid"] = np.asarray([value is not None for value in vector], dtype=bool)
        for name, masks in maximizers[candidate].items():
            evidence[f"{candidate}___{name}___maximizers"] = np.asarray(masks, dtype=bool)
    for name, statuses in pair_status.items():
        evidence[f"{name}___relation"] = np.asarray(statuses)
    summary = {"primary_evaluation": None, "scope": "fixed frozen choices/models/saved scenarios; historically inspected out-of-sample exploration",
        "dynamic_reselection": False, "joint_confidence_certification": False,
        "candidate_coordinate_and_signed_gap_intervals": {candidate: {name: summarize(vector) for name, vector in metrics.items()} for candidate, metrics in samples.items()},
        "candidate_scenario_loss_intervals": {candidate: {scenario: {loss: summarize(columns[candidate, scenario, loss]) for loss in LOSSES}
                                             for scenario in SCENARIOS} for candidate in CANDIDATES},
        "candidate_interaction_intervals": {candidate: {family: {loss: summarize(contrasts[candidate, family, loss]) for loss in LOSSES}
                                           for family in FAMILIES} for candidate in CANDIDATES},
        "fixed_candidate_pairs": pairs, "principal_contrast": principal,
        "frontier_stability": {"complete_pool_draws": complete_count, "unresolved_pool_draws": plan.n_draws-complete_count,
            "membership_counts_complete_pool_draws": {candidate: int(frontier_array[valid_frontier, index].sum()) for index, candidate in enumerate(CANDIDATES)},
            "membership_fraction_complete_pool_draws": {candidate: float(frontier_array[valid_frontier, index].mean()) if complete_count else None for index, candidate in enumerate(CANDIDATES)},
            "label": "exploratory full-vector frontier stability; incomplete draws are not certified frontier draws"},
        "maximizer_membership_counts": {candidate: {name: {scenario: int(np.asarray(masks)[:, index].sum()) for index, scenario in enumerate(JOINTS)}
                                          for name, masks in axes.items()} for candidate, axes in maximizers.items()},
        "maximizer_note": "all exact maxima retained; tied memberships may sum above draw count",
        "draw_storage": "NPZ undefined numeric coordinates use NaN plus explicit valid flags; summaries preserve null/valid counts",
        "conditioning": "omits fitting/selection/scenario generation uncertainty; Monte Carlo repetition SD is separate"}
    return summary, evidence
