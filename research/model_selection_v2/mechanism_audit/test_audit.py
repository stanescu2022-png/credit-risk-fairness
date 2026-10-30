"""Focused synthetic mechanism tests; no empirical loader is called."""
from fractions import Fraction
import itertools
import math
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from research.model_selection_v2 import safe2026 as safe
from research.model_selection_v2.mechanism_audit import audit
from research.model_selection_v2.mechanism_audit.joint_information import joint_fixture


class ExactReferenceTests(unittest.TestCase):
    def test_hand_calculated_weighted_pairs(self):
        # Inversion loses pair weight .2; preserved weights 1+.8; total 2.
        self.assertEqual(audit.pairwise_reference([0, Fraction(1, 5), 1], [2, 1, 3]), Fraction(9, 10))
        self.assertEqual(audit.pairwise_reference([0, 1], [1, 1]), Fraction(1, 2))
        self.assertIsNone(audit.pairwise_reference([.5, .5], [0, 1]))

    def test_exhaustive_small_orderings_and_reference_ties(self):
        for reference in ([0, .25, .5, 1], [.125, .125, .75, .75]):
            for ordering in itertools.product([.125, .5, .875], repeat=4):
                expected = audit.pairwise_reference(reference, ordering)
                self.assertAlmostEqual(safe.rank_graduation(reference, ordering).value, float(expected), places=14)

    def test_binary_reference_is_pairwise_auc(self):
        y = [0, 1, 0, 1]
        for p in ([.1, .2, .2, .9], [.5] * 4, [.9, .2, .8, .1]):
            self.assertAlmostEqual(float(audit.pairwise_reference(y, p)), roc_auc_score(y, p))

    def test_exact_affine_pairwise_cancellation_beyond_float_boundary(self):
        reference = list(map(Fraction, [0, .125, .5, 1]))
        ordering = list(map(Fraction, [.75, .5, .5, .125]))
        expected = audit.pairwise_reference(reference, ordering)
        for a, b in ((Fraction(3, 4), Fraction(1, 8)),
                     (Fraction(1, 2**100), Fraction(1, 2))):
            actual = audit.pairwise_reference([a*p+b for p in reference], [a*p+b for p in ordering])
            self.assertEqual(actual, expected)


class OperationalMechanismTests(unittest.TestCase):
    def test_affine_curves_every_greedy_alternative_and_all_means(self):
        for fixture_name in ("strict", "ties"):
            original = audit.fixture(fixture_name)
            baseline = audit.evaluate(original)
            for name, (a, b) in audit.TRANSFORMS.items():
                with self.subTest(fixture=fixture_name, transform=name):
                    transformed = audit.map_outputs(original, lambda p: a*p+b)
                    audit.assert_invariant(baseline, audit.evaluate(transformed))

    def test_near_constant_remains_nonconstant_and_preserves_exact_ties(self):
        for name in ("strict", "ties"):
            original = audit.fixture(name)
            transformed = audit.map_outputs(original, lambda p: p * (2. ** -40) + .5)
            self.assertGreater(np.ptp(transformed["clean"]), 0)
            for i, j in itertools.combinations(range(8), 2):
                self.assertEqual(original["clean"][i] == original["clean"][j],
                                 transformed["clean"][i] == transformed["clean"][j])
                self.assertEqual(original["clean"][i] < original["clean"][j],
                                 transformed["clean"][i] < transformed["clean"][j])

    def test_nonlinear_monotone_negative_control(self):
        for name in ("strict", "ties"):
            original = audit.fixture(name)
            baseline = audit.evaluate(original)
            square = audit.evaluate(audit.map_outputs(original, np.square))
            np.testing.assert_array_equal(baseline["accuracy"], square["accuracy"])
            self.assertGreater(max(abs(a-b) for a, b in zip(baseline["robustness"], square["robustness"])), audit.TOL)
            self.assertNotAlmostEqual(baseline["scores"]["arithmetic"], square["scores"]["arithmetic"], places=10)

    def test_constant_reference_and_float_collapse_are_undefined(self):
        original = audit.fixture("strict")
        for scale in (0., 2. ** -60):
            row = audit.evaluate(audit.map_outputs(original, lambda p: scale*p+.5))
            self.assertEqual(row["status"], "undefined")
            self.assertIn("constant reference", row["reason"])
            self.assertIsNone(row["scores"]["arithmetic"])
            self.assertEqual(row["robustness"], [None]*4)
            self.assertEqual(row["scalar_rga"], .5)

    def test_constant_comparison_is_half_with_nonconstant_reference(self):
        for name in ("strict", "ties"):
            result = audit.evaluate(audit.fixture(name))
            self.assertEqual(result["preservation"][-1], .5)
            self.assertEqual(result["robustness"][-1], .5)
            self.assertEqual(result["preservation"][0], 1.)

    def test_tie_expectation_accuracy_reference_is_row_permutation_invariant(self):
        original = audit.fixture("ties")
        expected = audit.accuracy_reference(original["outcome"], original["clean"])
        for permutation in ([7, 6, 5, 4, 3, 2, 1, 0], [1, 4, 6, 0, 5, 7, 3, 2]):
            actual = audit.accuracy_reference(original["outcome"][permutation], original["clean"][permutation])
            self.assertEqual(actual, expected)
        self.assertGreater(expected["distinct_tied_outcome_orders_enumerated"], 1)

    def test_direct_probability_losses_cross_checked_against_sklearn(self):
        for name in ("strict", "ties"):
            original = audit.fixture(name)
            for a, b in audit.TRANSFORMS.values():
                probability = a*original["clean"]+b
                losses = audit.proper_losses(original["outcome"], probability)
                self.assertAlmostEqual(losses["brier"], brier_score_loss(original["outcome"], probability), places=14)
                self.assertAlmostEqual(losses["log_loss"], log_loss(original["outcome"], probability, labels=[0, 1]), places=14)

    def test_independent_cartesian_aggregate_on_hand_values(self):
        A, E, R = [Fraction(0), Fraction(1)], [Fraction(1, 2)]*2, [Fraction(1)]*2
        expected = audit.integrated_reference(A, E, R)
        self.assertAlmostEqual(expected["arithmetic"], 2/3)
        self.assertAlmostEqual(expected["geometric"], .5*(.5**(1/3)))
        self.assertAlmostEqual(expected["quadratic"], .5*(math.sqrt(1.25/3)+math.sqrt(2.25/3)))
        actual = safe.integrate(list(map(float, A)), list(map(float, E)), list(map(float, R)))
        for name in expected:
            self.assertAlmostEqual(expected[name], actual[name])

    def test_transform_validation_rejects_outside_probability_range(self):
        with self.assertRaisesRegex(ValueError, "valid probabilities"):
            audit.map_outputs(audit.fixture("strict"), lambda p: p+1)

    def test_full_audit_does_not_call_empirical_access_or_fitting(self):
        with patch("research.model_selection_v2.pilot.load_data", side_effect=AssertionError("Empirical loader")), \
             patch("research.model_selection_v2.pilot.positive_probability", side_effect=AssertionError("Empirical prediction")), \
             patch("research.model_selection_v2.pilot.build_candidate", side_effect=AssertionError("Empirical candidate")):
            result = audit.make_audit()
        self.assertFalse(result["conclusions"]["new_empirical_data_or_predictions_accessed"])


class JointInformationTests(unittest.TestCase):
    def test_identical_separate_responses_and_distinct_joint_losses(self):
        values = joint_fixture()
        for scenario in ("clean", "stress_only", "mask_only"):
            np.testing.assert_array_equal(values["candidate_A"][scenario], values["candidate_B"][scenario])
        A = audit.proper_losses(values["outcome"], values["candidate_A"]["joint"])
        B = audit.proper_losses(values["outcome"], values["candidate_B"]["joint"])
        self.assertEqual(A["brier"], .0625)
        self.assertEqual(B["brier"], .5625)
        self.assertAlmostEqual(A["log_loss"], -math.log(.75))
        self.assertAlmostEqual(B["log_loss"], -math.log(.25))


if __name__ == "__main__":
    unittest.main()
