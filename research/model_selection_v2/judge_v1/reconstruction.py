"""Independent checks from saved judge outputs only; no original data or models."""
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from .io import read_json, require
from .protocol import CANDIDATES, COORDINATES, FAMILIES, SEVERITIES, scenario_name

LOSSES = ("brier", "log_loss")
SCENARIOS = ("clean",) + tuple(scenario_name(family, condition) for family in FAMILIES for condition in ("10", "01", "11"))
JOINTS = tuple(scenario_name(family, "11") for family in FAMILIES)
SIGNED = ("signed_tpr_gap", "signed_fpr_gap")
ATOL = 1e-12  # Numerical reconstruction agreement only; never a dominance margin.


def close(actual, expected):
    if expected is None:
        require(actual is None, "An undefined reconstructed quantity was concealed.")
    else:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=ATOL)


def direct_gaps(y, sex, p, weights):
    rates = {}
    for code in (1, 2):
        positive = (sex == code) & (y == 1)
        negative = (sex == code) & (y == 0)
        pos, neg = weights[positive].sum(), weights[negative].sum()
        rates[code] = {"tpr": weights[positive & (p > .5)].sum()/pos if pos else None,
                       "fpr": weights[negative & (p > .5)].sum()/neg if neg else None}
    return {f"signed_{name}_gap": rates[2][name]-rates[1][name] if rates[2][name] is not None and rates[1][name] is not None else None
            for name in ("tpr", "fpr")}


def direct_loss(y, p, weights=None):
    return {"brier": float(brier_score_loss(y, p, sample_weight=weights)),
            "log_loss": float(log_loss(y, p, labels=[0, 1], sample_weight=weights))}


def independent_relation(left, right):
    if any(left[name] is None or right[name] is None for name in COORDINATES):
        return "unresolved"
    less = any(left[name] < right[name] for name in COORDINATES)
    more = any(left[name] > right[name] for name in COORDINATES)
    return "incomparable" if less and more else "left_dominates" if less else "right_dominates" if more else "equal"


def interval_check(saved, values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    require(saved["valid_draws"] == int(valid.sum()) and saved["undefined_draws"] == int((~valid).sum()), "Bootstrap valid counts mismatch.")
    if valid.any():
        lower, upper = np.percentile(values[valid], saved["percentile_bounds"], method="linear")
        close(saved["lower"], lower)
        close(saved["upper"], upper)
    else:
        require(saved["lower"] is None and saved["upper"] is None, "Undefined interval must stay null.")


def direct_clean_diagnostics(y, p):
    """Independent tied-block AUC/AP and direct fixed-bin/count definitions."""
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    _, inverse = np.unique(p, return_inverse=True)
    positive = np.bincount(inverse, weights=(y == 1)).astype(float)
    negative = np.bincount(inverse, weights=(y == 0)).astype(float)
    auc = float(np.dot(positive, np.cumsum(negative)-negative+.5*negative)/(pos*neg)) if pos and neg else None
    cum_positive = np.cumsum(positive[::-1])
    cum_total = np.cumsum((positive+negative)[::-1])
    ap = float(np.sum(positive[::-1]/pos*cum_positive/cum_total)) if pos else None
    return {"auc": auc, "average_precision": ap,
            "recall": float(((y == 1) & (p > .5)).sum()/pos) if pos else None,
            "fpr": float(((y == 0) & (p > .5)).sum()/neg) if neg else None}


def check_stability(saved, statuses):
    counts = dict(Counter(statuses))
    unresolved = counts.get("unresolved", 0)
    require(saved["counts"] == counts and saved["resolved_draws"] == len(statuses)-unresolved and
            saved["unresolved_draws"] == unresolved and saved["total_draws"] == len(statuses), "Relation stability counts mismatch.")
    valid = len(statuses)-unresolved
    expected = {name: count/valid if valid else None for name, count in counts.items() if name != "unresolved"}
    require(saved["fraction_resolved"] == expected, "Relation stability denominator mismatch.")


def reconstruct_saved_run(destination):
    """Check declared point quantities and full saved draw consistency.

    Point losses/gaps/maxima/contrasts/relations and supporting clean AUC/AP,
    overall rates and reliability bins are reconstructed. Direct original-
    record/per-repetition weighted calculations check first, middle and final
    bootstrap draws. Every draw's maxima, absolute gaps, pair statuses,
    frontier/principal stability and loss/contrast intervals are checked
    using saved unrounded draw evidence. No regeneration or fitting occurs.
    """
    destination = Path(destination)
    metadata = pd.read_csv(destination / "evaluation_identity.csv")
    y, sex = metadata.default.to_numpy(), metadata.SEX.to_numpy()
    frozen = read_json(destination / "freeze/FREEZE.json")
    choices = frozen["reference_method_to_candidate"]
    principal_left, principal_right = choices["screened_safe_outcome_v0"], choices["safe_arithmetic"]
    with np.load(destination / "bootstrap_multiplicities.npz", allow_pickle=False) as packed:
        counts, positions, group_order, ids = [packed[name].copy() for name in ("group_multiplicities", "row_group_positions", "group_order", "ID")]
    np.testing.assert_array_equal(ids, metadata.ID)
    expected_groups = tuple(dict.fromkeys(metadata.duplicate_group_id))
    np.testing.assert_array_equal(group_order, expected_groups)
    group_positions = {group: index for index, group in enumerate(expected_groups)}
    expected_positions = np.array([group_positions[group] for group in metadata.duplicate_group_id])
    np.testing.assert_array_equal(positions, expected_positions)
    require((counts.sum(axis=1) == len(group_order)).all(), "Group draw totals changed.")
    audit_draws = sorted({0, len(counts)//2, len(counts)-1})
    checked_scenarios = checked_point_coordinates = checked_direct_draw_cells = 0
    baseline_clean = {}
    for q in SEVERITIES:
        folder = destination / f"q{q:.2f}"
        point = read_json(folder / "point.json")
        summary = read_json(folder / "bootstrap_summary.json")
        require(point["comparisons"]["frozen_reference_method_to_candidate"] == choices, "Reference choice identities changed in sensitivity.")
        predictions = {}
        for candidate in CANDIDATES:
            with np.load(folder / f"predictions/{candidate}.npz", allow_pickle=False) as packed:
                np.testing.assert_array_equal(packed["ID"], ids)
                require(set(packed.files) == {"ID", *SCENARIOS}, "Prediction scenario schema changed.")
                predictions[candidate] = {name: packed[name].copy() for name in SCENARIOS}
            if candidate in baseline_clean:
                np.testing.assert_array_equal(predictions[candidate]["clean"], baseline_clean[candidate])
            else:
                baseline_clean[candidate] = predictions[candidate]["clean"].copy()
            means, per_repetition = {}, {}
            for scenario in SCENARIOS:
                p = predictions[candidate][scenario]
                per_repetition[scenario] = [direct_loss(y, row) for row in p]
                means[scenario] = {loss: float(np.mean([row[loss] for row in per_repetition[scenario]])) for loss in LOSSES}
                for loss in LOSSES:
                    close(point["scenario_matrix"][candidate][scenario][loss], means[scenario][loss])
                    close(point["scenario_diagnostics"][candidate][scenario]["loss_deterioration"][loss], means[scenario][loss]-means["clean"][loss])
                    saved_rep = point["scenario_diagnostics"][candidate][scenario]["loss_repetitions"][loss]
                    expected_rep = [row[loss] for row in per_repetition[scenario]]
                    close(saved_rep["values"], expected_rep)
                    close(saved_rep["mc_sd"], float(np.std(expected_rep, ddof=1)) if len(expected_rep) > 1 else None)
                clean = predictions[candidate]["clean"][0]
                close(point["scenario_diagnostics"][candidate][scenario]["mean_absolute_probability_movement"]["mean"], np.abs(p-clean).mean())
                close(point["scenario_diagnostics"][candidate][scenario]["strict_classification_flip_rate"]["mean"], ((p > .5) != (clean > .5)).mean())
                endpoint = point["endpoint_counts"][candidate][scenario]["aggregate"]
                require(endpoint["probability_zero"] == int((p == 0).sum()) and endpoint["probability_one"] == int((p == 1).sum()), "Endpoint counts mismatch.")
                require(endpoint["clipped_below_epsilon"] == int((p < 2.**-52).sum()) and
                        endpoint["clipped_above_one_minus_epsilon"] == int((p > 1-2.**-52).sum()), "Clipping counts mismatch.")
                checked_scenarios += 1
            signed = direct_gaps(y, sex, predictions[candidate]["clean"][0], np.ones(len(y)))
            for name, value in signed.items():
                close(point["signed_gaps"][candidate][name], value)
            clean_p = predictions[candidate]["clean"][0]
            clean_supporting = point["clean_diagnostics"][candidate]
            for name, expected_value in direct_clean_diagnostics(y, clean_p).items():
                close(clean_supporting[name], expected_value)
            bin_indices = np.array([min(int(float(value)*10), 9) for value in clean_p])
            for index, row in enumerate(clean_supporting["reliability_bins"]):
                mask = bin_indices == index
                require(row["count"] == int(mask.sum()) and row["positive_count"] == int(y[mask].sum()), "Reliability bin counts mismatch.")
                close(row["mean_probability"], float(clean_p[mask].mean()) if mask.any() else None)
                close(row["observed_rate"], float(y[mask].mean()) if mask.any() else None)
            expected = {"clean_brier": means["clean"]["brier"], "clean_logloss": means["clean"]["log_loss"],
                "absolute_tpr_gap": abs(signed["signed_tpr_gap"]) if signed["signed_tpr_gap"] is not None else None,
                "absolute_fpr_gap": abs(signed["signed_fpr_gap"]) if signed["signed_fpr_gap"] is not None else None,
                "worst_joint_brier": max(means[name]["brier"] for name in JOINTS),
                "worst_joint_logloss": max(means[name]["log_loss"] for name in JOINTS)}
            for name, value in expected.items():
                close(point["coordinates"][candidate][name], value)
                checked_point_coordinates += 1
            for family in FAMILIES:
                for loss in LOSSES:
                    contrast = means[scenario_name(family, "11")][loss]-means[scenario_name(family, "10")][loss]-means[scenario_name(family, "01")][loss]+means["clean"][loss]
                    close(point["interactions"]["mean"][candidate][family][loss], contrast)
        vectors = point["coordinates"]
        for left, right in itertools.product(CANDIDATES, repeat=2):
            require(point["comparisons"]["pairwise_matrix"][left][right]["status"] == independent_relation(vectors[left], vectors[right]), "Point Pareto relation mismatch.")
        complete = [candidate for candidate in CANDIDATES if all(v is not None for v in vectors[candidate].values())]
        frontier = [candidate for candidate in complete if not any(independent_relation(vectors[other], vectors[candidate]) == "left_dominates" for other in complete if other != candidate)]
        require(point["comparisons"]["frontier"]["frontier_ids"] == frontier, "Point frontier mismatch.")
        require(point["comparisons"]["frontier"]["incomplete_pool"] == (len(complete) != len(CANDIDATES)), "Point frontier completeness mismatch.")
        principal_point = point["comparisons"]["principal_contrast"]
        principal_relation = independent_relation(vectors[principal_left], vectors[principal_right])
        principal_identity = "same_choice" if principal_left == principal_right else {
            "left_dominates": "our_choice_dominates", "right_dominates": "SAFE_choice_dominates"}.get(principal_relation, principal_relation)
        require(principal_point["status"] == principal_identity and principal_point["pareto_status"] == principal_relation, "Point principal identity/relation mismatch.")
        for name in COORDINATES:
            expected_difference = vectors[principal_left][name]-vectors[principal_right][name] if vectors[principal_left][name] is not None and vectors[principal_right][name] is not None else None
            close(principal_point["differences"][name], expected_difference)
        with np.load(folder / "bootstrap_draw_evidence.npz", allow_pickle=False) as packed:
            evidence = {key: packed[key].copy() for key in packed.files}
        keys = list(zip(evidence["loss_key_candidate"], evidence["loss_key_scenario"], evidence["loss_key_name"]))
        columns = {tuple(key): evidence["loss_matrix"][:, index] for index, key in enumerate(keys)}
        for draw in audit_draws:
            weights = counts[draw, positions]
            for candidate in CANDIDATES:
                for scenario in SCENARIOS:
                    direct = [direct_loss(y, p, weights) for p in predictions[candidate][scenario]]
                    for loss in LOSSES:
                        close(columns[candidate, scenario, loss][draw], np.mean([value[loss] for value in direct]))
                        checked_direct_draw_cells += 1
                signed = direct_gaps(y, sex, predictions[candidate]["clean"][0], weights)
                for name, expected in signed.items():
                    valid = evidence[f"{candidate}___{name}___valid"][draw]
                    require(bool(valid) == (expected is not None), "Direct signed-gap validity mismatch.")
                    if valid:
                        close(evidence[f"{candidate}___{name}"][draw], expected)
        for candidate in CANDIDATES:
            for coordinate in (*COORDINATES, *SIGNED):
                values = evidence[f"{candidate}___{coordinate}"]
                valid = evidence[f"{candidate}___{coordinate}___valid"]
                np.testing.assert_array_equal(valid, np.isfinite(values))
                interval_check(summary["candidate_coordinate_and_signed_gap_intervals"][candidate][coordinate], values)
            for coordinate, signed in (("absolute_tpr_gap", "signed_tpr_gap"), ("absolute_fpr_gap", "signed_fpr_gap")):
                np.testing.assert_array_equal(evidence[f"{candidate}___{coordinate}"], np.abs(evidence[f"{candidate}___{signed}"]))
            for loss, coordinate in (("brier", "worst_joint_brier"), ("log_loss", "worst_joint_logloss")):
                joined = np.column_stack([columns[candidate, name, loss] for name in JOINTS])
                highest = joined.max(axis=1)
                np.testing.assert_array_equal(evidence[f"{candidate}___{coordinate}"], highest)
                np.testing.assert_array_equal(evidence[f"{candidate}___{coordinate}___maximizers"], joined == highest[:, None])
                expected_memberships = {scenario: int((joined == highest[:, None])[:, index].sum()) for index, scenario in enumerate(JOINTS)}
                require(summary["maximizer_membership_counts"][candidate][coordinate] == expected_memberships, "Maximizer membership totals mismatch.")
            for scenario in SCENARIOS:
                for loss in LOSSES:
                    interval_check(summary["candidate_scenario_loss_intervals"][candidate][scenario][loss], columns[candidate, scenario, loss])
        contrasts = {(candidate, family, loss): columns[candidate, scenario_name(family, "11"), loss]
            -columns[candidate, scenario_name(family, "10"), loss]-columns[candidate, scenario_name(family, "01"), loss]
            +columns[candidate, "clean", loss] for candidate in CANDIDATES for family in FAMILIES for loss in LOSSES}
        for candidate in CANDIDATES:
            for family in FAMILIES:
                for loss in LOSSES:
                    interval_check(summary["candidate_interaction_intervals"][candidate][family][loss], contrasts[candidate, family, loss])
        expected_pairs = {f"{later}_minus_{earlier}": [] for earlier, later in itertools.combinations(CANDIDATES, 2)}
        expected_principal, expected_principal_pareto, expected_frontiers, expected_complete = [], [], [], []
        for draw in range(len(counts)):
            draw_vectors = {candidate: {name: evidence[f"{candidate}___{name}"][draw] if evidence[f"{candidate}___{name}___valid"][draw] else None
                            for name in COORDINATES} for candidate in CANDIDATES}
            complete_candidates = [candidate for candidate in CANDIDATES if all(value is not None for value in draw_vectors[candidate].values())]
            expected_complete.append(len(complete_candidates) == len(CANDIDATES))
            draw_frontier = [candidate for candidate in complete_candidates if not any(
                independent_relation(draw_vectors[other], draw_vectors[candidate]) == "left_dominates"
                for other in complete_candidates if other != candidate)]
            expected_frontiers.append([candidate in draw_frontier for candidate in CANDIDATES])
            for earlier, later in itertools.combinations(CANDIDATES, 2):
                expected_pairs[f"{later}_minus_{earlier}"].append(independent_relation(draw_vectors[later], draw_vectors[earlier]))
            principal_relation = independent_relation(draw_vectors[principal_left], draw_vectors[principal_right])
            expected_principal_pareto.append(principal_relation)
            expected_principal.append("same_choice" if principal_left == principal_right else {
                "left_dominates": "our_choice_dominates", "right_dominates": "SAFE_choice_dominates"}.get(principal_relation, principal_relation))
        np.testing.assert_array_equal(evidence["frontier_membership"], expected_frontiers)
        np.testing.assert_array_equal(evidence["complete_frontier"], expected_complete)
        np.testing.assert_array_equal(evidence["principal_pair_status"], expected_principal)
        np.testing.assert_array_equal(evidence["principal_pareto_status"], expected_principal_pareto)
        frontier_summary = summary["frontier_stability"]
        complete_mask = np.asarray(expected_complete, dtype=bool)
        complete_count = int(complete_mask.sum())
        require(frontier_summary["complete_pool_draws"] == complete_count and frontier_summary["unresolved_pool_draws"] == len(counts)-complete_count,
                "Bootstrap frontier completeness totals mismatch.")
        complete_membership = np.asarray(expected_frontiers)[complete_mask]
        for index, candidate in enumerate(CANDIDATES):
            require(frontier_summary["membership_counts_complete_pool_draws"][candidate] == int(complete_membership[:, index].sum()), "Frontier membership totals mismatch.")
            close(frontier_summary["membership_fraction_complete_pool_draws"][candidate], float(complete_membership[:, index].mean()) if complete_count else None)
        for earlier, later in itertools.combinations(CANDIDATES, 2):
            pair_name = f"{later}_minus_{earlier}"
            pair = summary["fixed_candidate_pairs"][pair_name]
            for coordinate in (*COORDINATES, *SIGNED):
                interval_check(pair["coordinate_and_signed_gap_intervals"][coordinate], evidence[f"{later}___{coordinate}"]-evidence[f"{earlier}___{coordinate}"])
            np.testing.assert_array_equal(evidence[f"{pair_name}___relation"], expected_pairs[pair_name])
            check_stability(pair["relation_stability"], expected_pairs[pair_name])
            for scenario in SCENARIOS:
                for loss in LOSSES:
                    interval_check(pair["loss_intervals"][scenario][loss], columns[later, scenario, loss]-columns[earlier, scenario, loss])
            for family in FAMILIES:
                for loss in LOSSES:
                    interval_check(pair["interaction_difference_intervals"][family][loss], contrasts[later, family, loss]-contrasts[earlier, family, loss])
        principal_summary = summary["principal_contrast"]
        require(principal_summary["same_choice"] == (principal_left == principal_right), "Principal same-choice identity changed.")
        check_stability(principal_summary["relation_stability"], expected_principal_pareto)
        require(principal_summary["relation_stability"]["identity_status_counts"] == dict(Counter(expected_principal)), "Principal identity frequencies mismatch.")
        for coordinate in (*COORDINATES, *SIGNED):
            interval_check(principal_summary["coordinate_and_signed_gap_intervals"][coordinate],
                           evidence[f"{principal_left}___{coordinate}"]-evidence[f"{principal_right}___{coordinate}"])
        for scenario in SCENARIOS:
            for loss in LOSSES:
                interval_check(principal_summary["loss_intervals"][scenario][loss], columns[principal_left, scenario, loss]-columns[principal_right, scenario, loss])
        for family in FAMILIES:
            for loss in LOSSES:
                interval_check(principal_summary["interaction_difference_intervals"][family][loss], contrasts[principal_left, family, loss]-contrasts[principal_right, family, loss])
    return {"status": "passed", "saved_prediction_scenarios_checked": checked_scenarios,
            "point_coordinates_checked": checked_point_coordinates, "direct_weighted_bootstrap_loss_cells_checked": checked_direct_draw_cells,
            "direct_record_repetition_bootstrap_draws": audit_draws,
            "all_saved_draw_maxima_absolute_gaps_pair_relations_and_coordinate_intervals_checked": True,
            "all_saved_draw_frontiers_principal_stability_loss_and_interaction_intervals_checked": True,
            "reconstruction_atol": ATOL, "tolerance_scope": "numerical reconstruction only; no dominance margin",
            "original_data_or_models_read": False, "outputs_regenerated": False}
