"""Independent numerical fixtures for the article's inference primitives."""
import unittest
import numpy as np
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score

from article_safe.inference import (
    paired_delong, midranks, likelihood_ratio, holm_adjust,
    concentration_gini, signed_share_audit, contribution_ks, group_row_weights,
)


def direct_placement_reference(y, a, b):
    """Brute-force Mann-Whitney placements; independent of midrank implementation."""
    positive, negative = [], []
    for scores in (a, b):
        scores = np.asarray(scores)
        comparisons = np.array([
            [float(p > n) + 0.5 * float(p == n) for n in scores[np.asarray(y) == 0]]
            for p in scores[np.asarray(y) == 1]
        ])
        positive.append(comparisons.mean(axis=1))
        negative.append(comparisons.mean(axis=0))
    positive, negative = np.asarray(positive), np.asarray(negative)
    covariance = np.cov(positive, ddof=1) / positive.shape[1] + np.cov(negative, ddof=1) / negative.shape[1]
    return positive.mean(axis=1), covariance


class DeLongTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([1, 0, 1, 0, 0, 1, 0, 1])
        self.a = np.array([.7, .1, .5, .5, .2, .9, .7, .5])
        self.b = np.array([.6, .1, .4, .5, .3, .8, .6, .6])

    def test_ties_and_paired_covariance_match_direct_placements(self):
        aucs, covariance = direct_placement_reference(self.y, self.a, self.b)
        result = paired_delong(self.y, self.a, self.b)
        np.testing.assert_allclose([result.auc_a, result.auc_b], aucs, atol=1e-14)
        np.testing.assert_allclose(result.covariance, covariance, atol=1e-14)
        expected_variance = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
        self.assertAlmostEqual(result.variance, expected_variance)
        self.assertGreater(covariance[0, 1], 0)
        self.assertLess(result.variance, covariance[0, 0] + covariance[1, 1])
        self.assertAlmostEqual(result.auc_a, roc_auc_score(self.y, self.a))
        self.assertAlmostEqual(result.auc_b, roc_auc_score(self.y, self.b))

    def test_model_swap_reverses_difference_and_preserves_p(self):
        ab = paired_delong(self.y, self.a, self.b)
        ba = paired_delong(self.y, self.b, self.a)
        self.assertAlmostEqual(ab.difference, -ba.difference)
        self.assertAlmostEqual(ab.variance, ba.variance)
        self.assertAlmostEqual(ab.p_value, ba.p_value)

    def test_row_permutation_preserves_the_paired_result(self):
        order = np.array([6, 2, 4, 7, 1, 0, 5, 3])
        expected = paired_delong(self.y, self.a, self.b)
        actual = paired_delong(self.y[order], self.a[order], self.b[order])
        np.testing.assert_allclose(actual.covariance, expected.covariance, atol=1e-14)
        self.assertAlmostEqual(actual.p_value, expected.p_value)

    def test_identical_predictions_return_identical_result(self):
        result = paired_delong(self.y, self.a, self.a)
        self.assertEqual(result.status, "identical_result")
        self.assertEqual(result.difference, 0)
        self.assertEqual(result.p_value, 1)

    def test_different_scores_with_same_placements_are_identical(self):
        result = paired_delong(self.y, self.a, self.a * 3 + 1)
        self.assertEqual(result.status, "identical_result")
        self.assertEqual(result.p_value, 1)

    def test_nonzero_difference_with_zero_variance_has_no_p(self):
        result = paired_delong([0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0])
        self.assertEqual(result.difference, 1)
        self.assertEqual(result.status, "degenerate_variance")
        self.assertIsNone(result.p_value)

    def test_single_positive_retains_auc_but_not_covariance_test(self):
        result = paired_delong([0, 0, 1], [.1, .2, .8], [.2, .3, .7])
        self.assertEqual(result.auc_a, 1)
        self.assertEqual(result.status, "insufficient_class_support")
        self.assertIsNone(result.p_value)

    def test_missing_class_is_visible(self):
        result = paired_delong([0, 0], [.1, .2], [.3, .4])
        self.assertEqual(result.status, "missing_outcome_class")
        self.assertIsNone(result.auc_a)

    def test_invalid_inputs_are_rejected(self):
        for y, a, b in [([0, 2], [.1, .2], [.2, .3]), ([0, 1], [.1], [.2, .3]),
                         ([0, 1], [.1, np.nan], [.2, .3])]:
            with self.assertRaises(ValueError):
                paired_delong(y, a, b)

    def test_midrank_tie_values(self):
        np.testing.assert_equal(midranks([4, 1, 4, 2]), [3.5, 1, 3.5, 2])


class LikelihoodAndMultiplicityTests(unittest.TestCase):
    def test_known_two_binomial_group_likelihood_statistic(self):
        # Two groups of ten: rates .8 and .2; pooled intercept-only rate .5.
        full = 2 * (8 * np.log(.8) + 2 * np.log(.2))
        reduced = 20 * np.log(.5)
        result = likelihood_ratio(full, reduced, 2, 1, 20, 20)
        expected = 2 * (full - reduced)
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["statistic"], expected)
        self.assertAlmostEqual(result["p_value"], chi2.sf(expected, 1))

    def test_whole_categorical_block_uses_rank_difference(self):
        result = likelihood_ratio(-100, -103, 7, 4, 200, 200)
        self.assertEqual(result["df"], 3)
        self.assertAlmostEqual(result["p_value"], chi2.sf(6, 3))

    def test_penalty_rows_zero_df_and_bad_fit_are_not_ordinary_tests(self):
        cases = [
            likelihood_ratio(-10, -12, 3, 2, 30, 30, penalized=True),
            likelihood_ratio(-10, -12, 3, 2, 30, 29),
            likelihood_ratio(-10, -12, 3, 3, 30, 30),
            likelihood_ratio(-12, -10, 3, 2, 30, 30),
            likelihood_ratio(np.nan, -10, 3, 2, 30, 30),
        ]
        for result in cases:
            self.assertEqual(result["status"], "not_estimable")
            self.assertIsNone(result["p_value"])

    def test_holm_known_family(self):
        np.testing.assert_allclose(holm_adjust([.01, .04, .03]), [.03, .06, .06])

    def test_missing_hypothesis_remains_missing_but_counts_in_family(self):
        result = holm_adjust([.01, None, .03], planned_family_size=4)
        np.testing.assert_allclose(result[[0, 2]], [.04, .09])
        self.assertTrue(np.isnan(result[1]))
        with self.assertRaises(ValueError):
            holm_adjust([.01, .02], planned_family_size=1)


class FairnessAndSamplingTests(unittest.TestCase):
    def test_equal_and_concentrated_gini_with_two_groups(self):
        self.assertEqual(concentration_gini([2, 2])["gini"], 0)
        concentrated = concentration_gini([1, 0])
        self.assertEqual(concentrated["gini"], .5)
        self.assertEqual(concentrated["normalized_gini"], 1)
        self.assertEqual(concentration_gini([1, 3])["gini"], .25)

    def test_gini_zero_and_negative_domain_handling(self):
        self.assertIsNone(concentration_gini([0, 0])["gini"])
        with self.assertRaises(ValueError):
            concentration_gini([-1, 1])
        for scale in (1e-200, 1e200):
            self.assertAlmostEqual(concentration_gini(np.array([1, 3]) * scale)["gini"], .25)

    def test_opposite_signs_remain_visible_despite_equal_magnitudes(self):
        audit = signed_share_audit([-1, 1])
        self.assertTrue(audit["cancellation"])
        self.assertTrue(audit["negative_effect"])
        self.assertIsNone(audit["shares"])
        self.assertEqual(concentration_gini([1, 1])["gini"], 0)
        self.assertTrue(signed_share_audit([-1, 2])["negative_share"])

    def test_ks_uses_full_distributions_even_when_means_match(self):
        female = [-1] * 10 + [1] * 10
        male = [-2] * 10 + [2] * 10
        self.assertEqual(np.mean(female), np.mean(male))
        result = contribution_ks(female, male)
        self.assertEqual(result["statistic"], .5)
        self.assertEqual(result["female_records"], 20)
        self.assertGreater(result["pooled_tie_fraction"], 0)
        self.assertIn("nominal", result["inference"])

    def test_constant_identical_ks_and_empty_group(self):
        self.assertEqual(contribution_ks([0] * 10, [0] * 10)["statistic"], 0)
        self.assertEqual(contribution_ks([0] * 10, [0] * 10)["p_value"], 1)
        self.assertEqual(contribution_ks([], [0])["status"], "not_estimable")

    def test_whole_group_weights_match_expanded_draws(self):
        # Group 10 can cross SEX and outcome; sampling never splits its two members.
        groups = np.array([10, 10, 20, 30, 30, 30])
        weights = group_row_weights(groups, np.random.default_rng(42))
        draws = np.random.default_rng(42).integers(0, 3, size=3)
        expanded = np.concatenate([np.flatnonzero(groups == np.unique(groups)[index]) for index in draws])
        np.testing.assert_equal(weights, np.bincount(expanded, minlength=len(groups)))
        for group in np.unique(groups):
            self.assertEqual(len(np.unique(weights[groups == group])), 1)