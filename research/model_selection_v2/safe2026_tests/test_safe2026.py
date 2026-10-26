import itertools
import math
import unittest

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from research.model_selection_v2 import safe2026 as s
from research.model_selection_v2 import pilot as p


def independent_rank(reference, ordering):
    y, q = np.asarray(reference), np.asarray(ordering)
    if np.ptp(y) == 0:
        return None
    w = np.arange(1, len(y) + 1)
    a = np.sort(y)
    return (np.dot(y, rankdata(q)) - np.dot(w, a[::-1])) / np.dot(w, a - a[::-1])


def frame(n=8):
    result = pd.DataFrame({name: np.arange(n, dtype=float) + i for i, name in enumerate(p.FEATURE_ORDER())})
    for name in p.CATEGORICAL:
        result[name] = np.resize([1, 2], n)
    return result


class RankTests(unittest.TestCase):
    def test_binary_scalar_auc_with_ties_and_permutations(self):
        y = np.array([0, 1, 0, 1])
        for score in ([.1, .2, .2, .9], [.1]*4, [0, 1, 0, 1], [1, 0, 1, 0]):
            expected = np.mean([float(score[i] > score[j]) + .5 * float(score[i] == score[j])
                                for i in np.flatnonzero(y == 1) for j in np.flatnonzero(y == 0)])
            self.assertAlmostEqual(expected, roc_auc_score(y, score))
            for perm in itertools.permutations(range(4)):
                perm = np.array(perm)
                actual = s.rank_graduation(y[perm], np.array(score)[perm]).value
                self.assertAlmostEqual(actual, expected)

    def test_identity_reverse_tied_and_constant(self):
        reference = np.array([.1, .1, .4, .8, .9])
        for ordering, expected in ((reference, 1.), (-reference, 0.), (np.ones(5), .5)):
            result = s.rank_graduation(reference, ordering)
            self.assertAlmostEqual(result.value, expected)
            self.assertAlmostEqual(result.value, independent_rank(reference, ordering))
        for reference in (np.zeros(5), np.ones(5), np.full(5, .2)):
            result = s.rank_graduation(reference, np.arange(5))
            self.assertIsNone(result.value)
            self.assertIn("constant reference", result.reason)

    def test_numerical_kernel_against_independent_rank_formula(self):
        rng = np.random.default_rng(27)
        for _ in range(20):
            reference = rng.choice([.1, .2, .8, .9], 30)
            ordering = rng.choice([.1, .5, .9], 30)
            self.assertAlmostEqual(s.rank_graduation(reference, ordering).value,
                                   independent_rank(reference, ordering), places=13)

    def test_nonfinite_negative_and_misaligned_rejected(self):
        for reference, ordering in (([0, 1], [1]), ([0, np.nan], [0, 1]),
                                    ([0, 1], [np.inf, 1]), ([-1, 1], [0, 1])):
            with self.assertRaises(ValueError):
                s.rank_graduation(reference, ordering)

    def test_rga_contribution_accounting_and_area(self):
        y, q = np.array([0, 1, 0, 1, 0, 1, 0]), np.array([.2, .8, .2, .8, .2, .5, .5])
        curve = s.accuracy_curve(y, q, segments=3)
        self.assertAlmostEqual(sum(curve["partial_contributions"]), roc_auc_score(y, q))
        self.assertEqual(curve["values"][-1], 0.)
        self.assertAlmostEqual(curve["values"][0], roc_auc_score(y, q))
        self.assertEqual(curve["segment_rank_cell_counts"], [3, 2, 2])
        expected = sum((b-a)*height for a, b, height in zip(curve["x"], curve["x"][1:], curve["values"]))
        self.assertAlmostEqual(curve["raw_step_area"], expected)
        self.assertNotAlmostEqual(curve["raw_step_area"], curve["uniform_node_mean"])
        for perm in (np.arange(7)[::-1], np.array([4, 2, 0, 6, 5, 3, 1])):
            permuted = s.accuracy_curve(y[perm], q[perm], segments=3)
            np.testing.assert_allclose(curve["values"], permuted["values"], atol=1e-15)

    def test_rga_full_terms_are_independent_prefix_areas(self):
        y, q = np.array([0, 1, 0, 1, 1]), np.array([.1, .3, .2, .5, .4])
        a, d = np.sort(y), np.sort(y)[::-1]
        ordered = y[np.argsort(q)]
        numerator = [sum(d[:i]) - sum(ordered[:i]) for i in range(1, 6)]
        denominator = sum(sum(d[:i]) - sum(a[:i]) for i in range(1, 6))
        result = s.rank_graduation(y, q)
        np.testing.assert_allclose(result.terms, np.array(numerator) / denominator)

    def test_one_class_rga_unavailable(self):
        self.assertEqual(s.accuracy_curve([1, 1, 1], [.1, .2, .3], 2)["status"], "undefined")


class PerturbationAndRemovalTests(unittest.TestCase):
    def test_zero_identity_and_tail_pairing(self):
        X = pd.DataFrame({"numeric": [0, 1, 2, 3, 4], "categorical": [1, 1, 2, 2, 3]}, index=[8, 3, 4, 1, 9])
        ids = [105, 101, 104, 103, 102]
        zero, audit = s.tail_swap(X, ids, 0.)
        pd.testing.assert_frame_equal(zero, X)
        self.assertEqual(audit["changed_cells"], 0)
        changed, audit = s.tail_swap(X, ids, .4)
        np.testing.assert_array_equal(changed.numeric, [4, 3, 2, 1, 0])
        self.assertEqual(audit["tail_size_per_feature"], 2)
        self.assertEqual(sorted(X.categorical), sorted(changed.categorical))
        self.assertEqual(audit["per_feature"]["numeric"]["paired_records"], 4)

    def test_swapping_coupled_and_id_permutation_invariant(self):
        X = pd.DataFrame({"a": [1, 1, 1, 3, 3, 3], "b": [1, 2, 3, 4, 5, 6]})
        ids = np.array([60, 50, 40, 30, 20, 10])
        expected, _ = s.tail_swap(X, ids, .5)
        again, _ = s.tail_swap(X, ids, .5)
        pd.testing.assert_frame_equal(expected, again)
        perm = np.array([5, 1, 3, 0, 4, 2])
        actual, _ = s.tail_swap(X.iloc[perm], ids[perm], .5)
        pd.testing.assert_frame_equal(actual.sort_index(), expected)
        np.testing.assert_array_equal(expected.b, [6, 5, 4, 3, 2, 1])

    def test_invalid_swap_inputs(self):
        for ids, strength in (([1, 1], .1), ([1], .1), ([1, 2], -.01), ([1, 2], .51)):
            with self.assertRaises(ValueError):
                s.tail_swap(pd.DataFrame({"x": [1, 2]}), ids, strength)

    def test_fitting_only_references_and_whole_original_onehot_group(self):
        fit = frame()
        selection = frame(4)
        selection["LIMIT_BAL"] = 1e6
        selection["EDUCATION"] = 2
        references = s.fitting_references(fit)
        self.assertEqual(references["LIMIT_BAL"], 3.5)
        self.assertEqual(references["EDUCATION"], 1.)  # Tied fitting modes -> smallest.
        model = p.build_candidate("decision_tree").fit(fit, np.resize([0, 1], len(fit)))
        pre = model.named_steps["preprocessor"]
        masked = s.mask_original(selection, ["EDUCATION"], references)
        np.testing.assert_array_equal(masked.EDUCATION, np.ones(4))
        pd.testing.assert_frame_equal(masked.drop(columns="EDUCATION"), selection.drop(columns="EDUCATION"))
        # All encoded columns belonging to the removed original field have
        # their fitting-reference representation, a valid whole one-hot block.
        encoded = pre.transform(masked)
        if hasattr(encoded, "toarray"):
            encoded = encoded.toarray()
        names = pre.get_feature_names_out()
        block = encoded[:, [i for i, name in enumerate(names) if name.startswith("cat__EDUCATION_")]]
        np.testing.assert_array_equal(block, np.tile([1., 0.], (4, 1)))
        self.assertNotEqual(s.fitting_references(pd.concat([fit, selection]))["LIMIT_BAL"], references["LIMIT_BAL"])

    def test_greedy_choices_against_exhaustive_small_example(self):
        X = np.array(list(itertools.product([0., 1.], repeat=3)))
        weights = np.array([.1, .2, .4])
        full = .1 + X @ weights
        features = ["a", "b", "c"]
        calls = []
        def predict(removed):
            calls.append(removed)
            changed = X.copy()
            changed[:, [features.index(name) for name in removed]] = .5
            return .1 + changed @ weights
        result = s.greedy_preservation(features, full, predict)
        self.assertEqual(len(calls), 6)
        removed = []
        for chosen in result["path"]:
            remaining = [name for name in features if name not in removed]
            exhaustive = {name: independent_rank(full, predict(removed + [name])) for name in remaining}
            maximum = max(exhaustive.values())
            expected = next(name for name in remaining if maximum - exhaustive[name] <= s.TOL)
            self.assertEqual(chosen["alternative"], expected)
            removed.append(expected)
        self.assertAlmostEqual(result["values"][0], 1.)
        self.assertAlmostEqual(result["values"][-1], .5)
        self.assertAlmostEqual(1 - result["values"][0], 0.)
        self.assertAlmostEqual(1 - result["values"][-1], .5)

    def test_greedy_ties_and_constant_reference(self):
        full = np.array([.1, .2, .4, .8])
        result = s.greedy_preservation(["b", "a"], full, lambda removed: np.ones(4))
        self.assertEqual(result["path"][0]["alternative"], "b")
        self.assertEqual(result["path"][0]["tied_alternatives"], ["b", "a"])
        self.assertEqual(s.greedy_preservation(["a"], np.ones(4), lambda removed: full)["status"], "undefined")

    def test_nonmonotonic_empirical_curve_retained(self):
        summary = s.curve_summary([0, .5, 1], [1, .4, .6])
        self.assertEqual(summary["increases"], [1])
        self.assertEqual(summary["trapezoidal_area"], .6)


class AggregationTests(unittest.TestCase):
    def test_presentation_arithmetic_example(self):
        values = s.integrate([.8, 0.], [1., .5], [1., .4])
        self.assertAlmostEqual(values["arithmetic"], .6166666666666667)

    def test_cellwise_against_explicit_cartesian_product(self):
        A, E, R = [.2, .8, 0.], [1., .9, .5], [1., .5, .2]
        actual = s.integrate(A, E, R)
        cells = list(itertools.product(A, E, R))
        expected = {"arithmetic": sum(sum(cell)/3 for cell in cells)/len(cells),
                    "geometric": sum(math.prod(cell)**(1/3) for cell in cells)/len(cells),
                    "quadratic": sum(math.sqrt(sum(v*v for v in cell)/3) for cell in cells)/len(cells)}
        for name in expected:
            self.assertAlmostEqual(actual[name], expected[name])
        self.assertNotAlmostEqual(actual["geometric"], (np.mean(A)*np.mean(E)*np.mean(R))**(1/3))
        self.assertNotAlmostEqual(actual["quadratic"], np.sqrt((np.mean(A)**2+np.mean(E)**2+np.mean(R)**2)/3))

    def test_undefined_not_zero_and_no_nonfinite_substitution(self):
        self.assertEqual(s.integrate([1., None], [1., .5], [1., .5])["status"], "undefined")
        with self.assertRaises(ValueError):
            s.integrate([1., np.nan], [1., .5], [1., .5])

    def test_explicit_tied_selection_and_unrankability(self):
        ranking = s.rank_candidates({"b": .6, "a": .6, "c": None}, ["b", "a", "c"])
        self.assertEqual(ranking["selected_id"], "b")
        self.assertEqual(ranking["tied_selected_ids"], ["b", "a"])
        self.assertEqual(ranking["rows"][-1]["status"], "unrankable")


if __name__ == "__main__":
    unittest.main()
