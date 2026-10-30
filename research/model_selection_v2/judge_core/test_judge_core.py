"""Deterministic unit checks; no data, model, run or source snapshot access."""
import builtins
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from research.model_selection_v2.judge_core import (
    BootstrapPlan, CandidateTables, JudgeConfig, PredictionTable, bootstrap_judge,
    bootstrap_with_plan, candidate_metrics, compare_choices, evaluate_candidates,
    fairness_gaps, group_bootstrap_plan, losses, pareto_relation, percentile_summary,
    pool_frontier, scenario_losses, validate_alignment,
)
from research.model_selection_v2.judge_core.metrics import LOGLOSS_EPSILON


def table(p=(0., .5, 1., .75), *, y=(0, 1, 0, 1), sex=(1, 1, 2, 2),
          groups=("cross", "cross", "cross", "single"), ids=(10, 20, 30, 40), role="synthetic"):
    return PredictionTable(tuple(ids), np.asarray(y), np.asarray(sex), tuple(groups), np.asarray(p), role)


def config(directions=None, *, ours="A", safe="B", order=("A", "B"), role="synthetic", primary=None):
    return JudgeConfig(directions if directions is not None else {"clean.brier": "min"},
                       ours, safe, order, role, primary)


class TableContractTests(unittest.TestCase):
    def test_copy_validation_and_read_only_arrays(self):
        original = np.array([.1, .2, .3, .4])
        validated = table(original)
        original[0] = .9
        self.assertEqual(validated.probability[0], .1)
        with self.assertRaises(ValueError):
            validated.probability[0] = .9

    def test_column_mapping_requires_no_empirical_schema(self):
        columns = {"id": [1, 2], "default": [0, 1], "sex": [1, 2], "g": [9, 9], "p": [.2, .8]}
        mapped = PredictionTable.from_columns(columns, role_label="fixture", column_map={
            "ID": "id", "outcome": "default", "SEX": "sex", "original_group": "g", "probability": "p"})
        self.assertEqual(mapped.ids, (1, 2))
        self.assertEqual(mapped.role_label, "fixture")

    def test_invalid_probabilities_targets_sex_ids_and_groups(self):
        for kwargs in ({"p": [0, .5, 1, np.nan]}, {"p": [0, .5, 1, np.inf]}, {"p": [0, .5, 1, 1.01]},
                       {"y": [0, 2, 0, 1]}, {"sex": [1, 0, 2, 2]}, {"ids": [1, 1, 3, 4]},
                       {"groups": [1, np.nan, 1, 2]}, {"role": ""}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                table(**kwargs)

    def test_id_outcome_sex_group_and_role_mismatch_are_rejected(self):
        reference = table()
        for kwargs in ({"ids": [20, 10, 30, 40]}, {"y": [1, 1, 0, 1]}, {"sex": [2, 1, 2, 2]},
                       {"groups": ["cross", "changed", "cross", "single"]}, {"role": "different"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                validate_alignment((reference, table(**kwargs)), role_label="synthetic")

    def test_crossing_sex_and_outcome_groups_are_allowed(self):
        reference = table()
        self.assertEqual(reference.original_group[:3], ("cross",)*3)
        self.assertEqual(set(reference.sex[:3]), {1, 2})
        self.assertEqual(set(reference.outcome[:3]), {0, 1})
        validate_alignment((reference, table([.1, .3, .7, .9])), role_label="synthetic")


class MetricTests(unittest.TestCase):
    def test_hand_brier_and_logloss_with_endpoint_policy(self):
        result = losses(table())
        self.assertEqual(result["brier"], (0+.25+1+.0625)/4)
        expected = (-math.log1p(-LOGLOSS_EPSILON)-math.log(.5)-math.log(LOGLOSS_EPSILON)-math.log(.75))/4
        self.assertAlmostEqual(result["log_loss"], expected, places=14)
        self.assertEqual(LOGLOSS_EPSILON, 2.**-52)

    def test_probability_endpoints_stay_finite_and_brier_unclipped(self):
        good = table([0, 1, 0, 1])
        bad = table([1, 0, 1, 0])
        self.assertEqual(losses(good)["brier"], 0.)
        self.assertEqual(losses(bad)["brier"], 1.)
        self.assertAlmostEqual(losses(good)["log_loss"], -math.log1p(-LOGLOSS_EPSILON), places=14)
        self.assertAlmostEqual(losses(bad)["log_loss"], -math.log(LOGLOSS_EPSILON), places=14)

    def test_strict_threshold_tie_and_female_minus_male_gaps(self):
        gaps = fairness_gaps(table())
        self.assertEqual(gaps["rates"]["male"]["tp"], 0)  # p=.5 is class zero.
        self.assertEqual(gaps["rates"]["female"]["tp"], 1)
        self.assertEqual(gaps["tpr_gap_signed"], 1.)
        self.assertEqual(gaps["fpr_gap_signed"], 1.)
        reversed_sex = fairness_gaps(table(sex=[2, 2, 1, 1]))
        self.assertEqual(reversed_sex["tpr_gap_signed"], -1.)
        self.assertEqual(reversed_sex["tpr_gap_absolute"], 1.)

    def test_undefined_rates_preserve_denominators_and_null_gaps(self):
        gaps = fairness_gaps(table(y=[0, 0, 0, 0]))
        self.assertIsNone(gaps["tpr_gap_signed"])
        self.assertIsNone(gaps["tpr_gap_absolute"])
        self.assertEqual(gaps["rates"]["female"]["positive_denominator"], 0)
        self.assertIsNotNone(gaps["fpr_gap_signed"])

    def test_integer_multiplicities_match_explicit_record_repetition(self):
        weights = np.array([2, 2, 2, 0])
        result = losses(table(), multiplicities=weights)
        self.assertEqual(result["brier"], (0+.25+1)/3)
        gaps = fairness_gaps(table(), multiplicities=weights)
        self.assertEqual(gaps["rates"]["male"]["positive_denominator"], 2)
        self.assertIsNone(gaps["rates"]["female"]["tpr"])
        for bad in ([1., 1., 1., 1.], [1, -1, 1, 1], [0, 0, 0, 0], [1, 1]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                losses(table(), multiplicities=bad)

    def test_repetition_mean_of_losses_differs_from_loss_of_mean_probability(self):
        clean = table([.5]*4)
        repetitions = (table([0, 1, 0, 1]), table([1, 0, 1, 0]))
        result = scenario_losses(clean, {"stress": repetitions})["stress"]
        self.assertEqual(result["brier"], .5)
        self.assertEqual(losses(clean)["brier"], .25)
        self.assertGreater(result["log_loss"], losses(clean)["log_loss"])
        self.assertEqual(result["repetition_count"], 2)
        self.assertEqual(result["brier"], sum(v["brier"] for v in result["repetition_losses"])/2)

    def test_scenario_alignment_and_nonempty_repetitions(self):
        for scenarios in ({"stress": []}, {"clean": [table()]}, {"stress": [table(ids=[1, 2, 3, 4])]}):
            with self.subTest(scenarios=scenarios), self.assertRaises(ValueError):
                scenario_losses(table(), scenarios)


class ParetoTests(unittest.TestCase):
    def test_directional_dominance_incomparability_and_exact_equality(self):
        directions = {"loss": "min", "benefit": "max"}
        examples = [({"loss": 1., "benefit": 2.}, {"loss": 2., "benefit": 1.}, "left_dominates"),
                    ({"loss": 2., "benefit": 1.}, {"loss": 1., "benefit": 2.}, "right_dominates"),
                    ({"loss": 1., "benefit": 1.}, {"loss": 2., "benefit": 2.}, "incomparable"),
                    ({"loss": 1., "benefit": 2.}, {"loss": 1., "benefit": 2.}, "equal")]
        for left, right, expected in examples:
            self.assertEqual(pareto_relation(left, right, metric_directions=directions)["status"], expected)

    def test_no_rounding_or_tolerance_in_practical_dominance(self):
        right = np.nextafter(.5, 1.)
        result = pareto_relation({"m": .5}, {"m": right}, metric_directions={"m": "min"})
        self.assertEqual(result["status"], "left_dominates")
        self.assertLess(result["left_minus_right"]["m"], 0)

    def test_missing_coordinate_is_unresolved_and_nan_rejected(self):
        result = pareto_relation({"a": 1., "b": None}, {"a": 2., "b": 1.}, metric_directions={"a": "min", "b": "min"})
        self.assertEqual(result["status"], "unresolved")
        self.assertEqual(result["missing_coordinates"], {"b": ["left"]})
        self.assertIsNone(result["left_minus_right"]["b"])
        with self.assertRaises(ValueError):
            pareto_relation({"a": np.nan}, {"a": 1}, metric_directions={"a": "min"})

    def test_same_choice_identity_distinct_from_different_candidate_tie(self):
        vectors = {"A": {"x": 1}, "B": {"x": 1}}
        tied = compare_choices(vectors, config({"x": "min"}))
        self.assertEqual(tied["status"], "equal")
        self.assertFalse(tied["same_choice"])
        same = compare_choices(vectors, config({"x": "min"}, ours="A", safe="A"))
        self.assertEqual(same["status"], "same_choice")
        self.assertTrue(same["same_choice"])
        self.assertEqual(same["differences"], {"x": 0.})
        undefined = compare_choices({"A": {"x": None}}, config({"x": "min"}, ours="A", safe="A"))
        self.assertEqual(undefined["status"], "same_choice")
        self.assertEqual(undefined["pareto_status"], "unresolved")

    def test_difference_name_and_direction_are_our_choice_minus_safe(self):
        comparison = compare_choices({"A": {"m": .1}, "B": {"m": .3}}, config({"m": "min"}))
        self.assertEqual(comparison["difference_name"], "our_choice_minus_SAFE_choice")
        self.assertAlmostEqual(comparison["differences"]["m"], -.2)
        self.assertEqual(comparison["status"], "our_choice_dominates")

    def test_full_pool_frontier_and_incomplete_pool_warning(self):
        cfg = config({"x": "min", "y": "min"}, order=("A", "B", "C", "D"))
        vectors = {"A": {"x": 1, "y": 2}, "B": {"x": 2, "y": 1},
                   "C": {"x": 2, "y": 2}, "D": {"x": 1, "y": 2}}
        complete = pool_frontier(vectors, cfg)
        self.assertEqual(complete["frontier_ids"], ["A", "B", "D"])
        self.assertFalse(complete["incomplete_pool"])
        del vectors["D"]
        vectors["C"]["x"] = None
        incomplete = pool_frontier(vectors, cfg)
        self.assertTrue(incomplete["incomplete_pool"])
        self.assertEqual(incomplete["unavailable_candidate_ids"], ["D"])
        self.assertEqual(incomplete["unresolved_candidate_ids"], ["C"])
        self.assertEqual(incomplete["frontier_scope"], "complete_observed_subset")
        self.assertIsNotNone(incomplete["warning"])

    def test_explicit_config_and_selected_ids_are_required(self):
        for kwargs in ({"directions": {}}, {"primary": "brier"}, {"ours": "unknown"}, {"role": ""},
                       {"order": ["A", "A"]}, {"directions": {"m": "up"}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                config(**kwargs)
        with self.assertRaises(ValueError):
            compare_choices({"A": {"clean.brier": .1}}, config())

    def test_evaluation_uses_supplied_roles_and_explicit_coordinates(self):
        candidates = {"A": CandidateTables(table([0, 1, 0, 1])), "B": CandidateTables(table())}
        result = evaluate_candidates(candidates, config({"clean.log_loss": "min"}))
        self.assertIsNone(result["primary_evaluation"])
        self.assertEqual(result["choice_comparison"]["status"], "our_choice_dominates")
        self.assertEqual(result["pool_frontier"]["frontier_ids"], ["A"])
        with self.assertRaises(ValueError):
            evaluate_candidates(candidates, config(role="different"))
        missing = evaluate_candidates(candidates, config({"unavailable.coordinate": "min"}))
        self.assertEqual(missing["choice_comparison"]["status"], "unresolved")

    def test_scenario_repetition_counts_must_match_across_candidates(self):
        candidates = {"A": CandidateTables(table(), {"stress": [table()]}),
                      "B": CandidateTables(table(), {"stress": [table(), table()]})}
        with self.assertRaisesRegex(ValueError, "repetition counts"):
            evaluate_candidates(candidates, config())


class BootstrapTests(unittest.TestCase):
    def test_explicit_whole_group_counts_preserve_crossings_and_variable_row_count(self):
        reference = table()
        plan = BootstrapPlan(reference, np.array([[2, 0], [1, 1], [0, 2]]), None)
        self.assertEqual(plan.group_order, ("cross", "single"))
        np.testing.assert_array_equal(plan.row_multiplicities(0), [2, 2, 2, 0])
        np.testing.assert_array_equal(plan.row_multiplicities(1), [1, 1, 1, 1])
        np.testing.assert_array_equal(plan.row_multiplicities(2), [0, 0, 0, 2])
        self.assertEqual(plan.row_multiplicities(0).sum(), 6)
        self.assertEqual(plan.row_multiplicities(2).sum(), 2)
        self.assertEqual(losses(reference, multiplicities=plan.row_multiplicities(0))["brier"], (0+.25+1)/3)

    def test_seeded_generation_reproducible_without_defaults(self):
        A = group_bootstrap_plan(table(), n_draws=7, seed=9)
        B = group_bootstrap_plan(table(), n_draws=7, seed=9)
        np.testing.assert_array_equal(A.group_multiplicities, B.group_multiplicities)
        np.testing.assert_array_equal(A.group_multiplicities.sum(axis=1), np.full(7, 2))
        with self.assertRaises(TypeError):
            group_bootstrap_plan(table())
        with self.assertRaises(ValueError):
            group_bootstrap_plan(table(), n_draws=0, seed=42)
        with self.assertRaises(ValueError):
            BootstrapPlan(table(), np.array([[1, 0]]), None)

    def test_percentile_hand_values_valid_counts_and_all_missing(self):
        result = percentile_summary([0, 1, None, 2, 3], percentile_bounds=(25, 75))
        self.assertEqual(result["lower"], .75)
        self.assertEqual(result["upper"], 2.25)
        self.assertEqual(result["valid_draws"], 4)
        self.assertEqual(result["undefined_draws"], 1)
        missing = percentile_summary([None, None], percentile_bounds=(2.5, 97.5))
        self.assertIsNone(missing["lower"])
        self.assertEqual(missing["valid_draws"], 0)

    def test_percentile_bounds_validation_and_iterator_reuse(self):
        for bounds in ((90, 10), (-1, 90), (0, np.inf), ("bad", 90)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                percentile_summary([1], percentile_bounds=bounds)
        candidates = {"A": CandidateTables(table()), "B": CandidateTables(table())}
        result = bootstrap_judge(candidates, config(), n_draws=2, seed=7,
                                 percentile_bounds=iter((0, 100)))
        self.assertEqual(result["paired_difference_intervals"]["clean.brier"]["lower"], 0.)

    def test_shared_bootstrap_signed_intervals_and_unresolved_counts(self):
        A, B = table([0, 1, 0, 1]), table()
        candidates = {"A": CandidateTables(A), "B": CandidateTables(B)}
        plan = BootstrapPlan(A, np.array([[2, 0], [1, 1], [0, 2]]), None)
        result = bootstrap_with_plan(candidates, config({"clean.brier": "min", "clean.tpr_gap_absolute": "min"}),
                                     plan, percentile_bounds=(0, 100))
        self.assertEqual(result["candidate_metric_intervals"]["B"]["clean.tpr_gap_signed"]["valid_draws"], 1)
        signed = result["paired_difference_intervals"]["clean.tpr_gap_signed"]
        self.assertEqual(signed["lower"], -1.)
        self.assertEqual(signed["upper"], -1.)
        self.assertEqual(signed["valid_draws"], 1)
        stability = result["dominance_stability"]
        self.assertEqual(stability["resolved_draws"], 1)
        self.assertEqual(stability["unresolved_draws"], 2)
        self.assertEqual(stability["our_choice_dominates_fraction_resolved"], 1.)
        self.assertIn("not joint confidence", stability["label"])
        self.assertEqual(result["bootstrap_plan"]["group_multiplicities"], [[2, 0], [1, 1], [0, 2]])

    def test_same_choice_has_identical_bootstrap_differences(self):
        reference = table()
        candidates = {"A": CandidateTables(reference)}
        result = bootstrap_judge(candidates, config(ours="A", safe="A", order=("A",)),
                                 n_draws=5, seed=17, percentile_bounds=(0, 100))
        self.assertEqual(result["dominance_stability"]["pair_status_counts"], {"same_choice": 5})
        for values in result["draw_evidence"]["our_choice_minus_SAFE_choice"].values():
            self.assertTrue(all(v is None or v == 0 for v in values))

    def test_repetition_averaging_uses_the_same_group_weights(self):
        reference = table()
        candidates = {"A": CandidateTables(reference, {"stress": [table([0, 1, 0, 1]), table([1, 0, 1, 0])]}),
                      "B": CandidateTables(reference, {"stress": [reference, reference]})}
        plan = BootstrapPlan(reference, np.array([[2, 0], [0, 2]]), None)
        result = bootstrap_with_plan(candidates, config({"stress.brier": "min"}), plan, percentile_bounds=(0, 100))
        self.assertEqual(result["draw_evidence"]["candidate_metrics"]["A"]["stress.brier"], [.5, .5])
        self.assertEqual(result["draw_evidence"]["candidate_metrics"]["B"]["stress.brier"], [(0+.25+1)/3, .0625])

    def test_bootstrap_plan_rejects_metadata_or_role_switch(self):
        candidates = {"A": CandidateTables(table()), "B": CandidateTables(table())}
        plan = BootstrapPlan(table(groups=["wrong"]*4), np.array([[1]]), None)
        with self.assertRaises(ValueError):
            bootstrap_with_plan(candidates, config(), plan, percentile_bounds=(0, 100))

    def test_no_empirical_access_model_calls_or_files(self):
        candidates = {"A": CandidateTables(table()), "B": CandidateTables(table([.1, .9, .1, .9]))}
        with patch.object(builtins, "open", side_effect=AssertionError("Unexpected file access")), \
             patch.object(Path, "read_text", side_effect=AssertionError("Unexpected file read")), \
             patch.object(Path, "read_bytes", side_effect=AssertionError("Unexpected file read")):
            result = bootstrap_judge(candidates, config(), n_draws=4, seed=11, percentile_bounds=(0, 100))
        self.assertEqual(result["bootstrap_plan"]["draw_count"], 4)


if __name__ == "__main__":
    unittest.main()
