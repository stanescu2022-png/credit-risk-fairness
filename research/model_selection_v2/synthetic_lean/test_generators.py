import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm

from research.model_selection_v2.safe2026 import tail_swap
from research.model_selection_v2.synthetic_lean import generators as g


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = g.calibrate('R')
        with patch.object(g, 'REFERENCE_SIZE', 4096):
            cls.f = g.calibrate('F')

    def test_seed_scopes(self):
        seed = g.child_seed('timing', 'R', 1, 'x1')
        self.assertEqual(seed, 13630673599215377906)
        expected = [seed, g.child_seed('production', 'R', 1, 'x1'),
                    g.child_seed('timing', 'F', 1, 'x1'), g.child_seed('timing', 'R', 2, 'x1'),
                    g.child_seed('timing', 'R', 1, 'x2')]
        self.assertEqual(len(set(expected)), len(expected))
        with self.assertRaises(ValueError):
            g.child_seed('timing:R', 'R', 1, 'x1')

    def test_identity_and_replay(self):
        for setting, calibration in (('R', self.r), ('F', self.f)):
            with self.subTest(setting=setting):
                first = g.generate_population(setting, 'fixture', 0, calibration, n=200)
                replay = g.generate_population(setting, 'fixture', 0, calibration, n=200)
                other = g.generate_population(setting, 'fixture', 1, calibration, n=200)
                self.assertEqual(first.dataset_id, f'synthetic_lean:fixture:{setting}:0')
                self.assertNotEqual(first.dataset_id, other.dataset_id)
                pd.testing.assert_frame_equal(first.X, replay.X)
                for name in ('ids', 'sex', 'p', 'y'):
                    values = getattr(first, name)
                    np.testing.assert_array_equal(values, getattr(replay, name))
                    self.assertFalse(values.flags.writeable)
                np.testing.assert_array_equal(first.ids, np.arange(1, 201))
                np.testing.assert_array_equal(first.X.index.to_numpy(), first.ids)
                self.assertEqual(first.X.index.name, 'ID')
                self.assertEqual(set(first.sex), {1, 2})
                self.assertEqual(set(first.y), {0, 1})
                self.assertTrue(np.all((first.p >= 0.) & (first.p <= 1.)))
                self.assertFalse(first.X.equals(other.X))
                schema = g.study_schema(setting)
                schema.validate(first.X)
                self.assertEqual(set(first.X.dtypes), {np.dtype('float64')})
                groups = g.study_families(setting).values()
                self.assertEqual([len(fields) for fields in groups], [5] * 3)
                self.assertCountEqual([field for fields in groups for field in fields], schema.order)

    def test_r_generator_and_exact_prevalence_target(self):
        population = g.generate_population('R', 'fixture', 2, self.r, n=800)
        frame = population.X
        tail = np.select([frame.x1 > norm.ppf(.95), frame.x1 < norm.ppf(.05)], [1, -1], default=0)
        half = np.where(np.abs(frame.x2) > norm.ppf(.75), 1, -1)
        np.testing.assert_array_equal(population.p, expit(
            self.r.intercept + (.8 * (frame.x3 + frame.x4) + .4 * half) + .8 * tail * half
        ))
        self.assertEqual(g.study_families('R')['A'][:2], ('x1', 'x2'))

        def integrate_risk(z):
            conditional = sum(
                mass / 2 * expit(self.r.intercept + .8 * np.sqrt(2) * z + (.4 + .8 * sign) * side)
                for sign, mass in ((-1, .05), (0, .9), (1, .05)) for side in (-1, 1)
            )
            return norm.pdf(z) * conditional

        self.assertAlmostEqual(quad(integrate_risk, -np.inf, np.inf)[0], .22, places=12)

    def test_r_reference_scores_against_unmoved_truth(self):
        with patch.object(g, 'REFERENCE_SIZE', 1600):
            result = g.r_reference_curve(self.r)
        self.assertEqual(result['records'], 1600)
        self.assertEqual(result['ideal_clean_preferred'], 'full')
        self.assertEqual(result['ideal_six_node_preferred'], 'omitted')
        self.assertEqual(result['ideal_crossover_tail_fraction'], .0125)
        self.assertAlmostEqual(result['ideal_six_node_mean_in_a_squared']['full'], .20)
        self.assertAlmostEqual(result['ideal_six_node_mean_in_a_squared']['omitted'], .10)
        self.assertEqual(result['clean_preferred'], 'full')
        self.assertEqual(result['nodes'][0]['full_truth_mse'], 0.)
        self.assertEqual([row['q'] for row in result['nodes']], [0., .01, .02, .03, .04, .05])
        for node in result['nodes']:
            self.assertEqual(node['tail_size_per_feature'], int(np.floor(1600 * node['q'])))
        ids = g._row_ids(1600)
        frame = g._r_features('reference', 0, ids)
        changed, _ = tail_swap(frame, ids, .05)
        self.assertEqual(result['tail_counts']['lower'], int(np.count_nonzero(frame.x1 < norm.ppf(.05))))
        self.assertEqual(result['tail_counts']['upper'], int(np.count_nonzero(frame.x1 > norm.ppf(.95))))

        def risk(inputs):
            signed = np.sign(inputs.x1) * (np.abs(inputs.x1) > norm.ppf(.95))
            half = np.where(np.abs(inputs.x2) > norm.ppf(.75), 1, -1)
            return expit(self.r.intercept + .8 * (inputs.x3 + inputs.x4) + .4 * half + .8 * signed * half)

        self.assertAlmostEqual(result['nodes'][-1]['full_truth_mse'], ((risk(changed) - risk(frame)) ** 2).mean())

    def test_f_calibration_and_common_vs_replacement_risk(self):
        reference = self.f.f_reference
        self.assertAlmostEqual(reference.common_risk_mean, .22, places=12)
        self.assertAlmostEqual(reference.common_risk_auc, .78, places=11)
        self.assertNotAlmostEqual(reference.common_risk_mean, reference.f_risk_mean, places=6)
        self.assertEqual(g.Calibration.from_dict(json.loads(json.dumps(self.f.as_dict()))), self.f)
        pop = g.generate_population('F', 'fixture', 3, self.f, n=1000)
        self.assertTrue(pop.X.age.between(21, 70).all())
        self.assertTrue(pop.X.dependants.between(0, 6).all())
        self.assertTrue(pop.X.open_credit_lines.between(0, 10).all())
        self.assertTrue(pop.X.utilisation.between(0, 1).all())
        self.assertTrue((pop.X.loc[pop.X.worst_delay <= 0, 'months_with_delay'] == 0).all())
        self.assertTrue(pop.X.loc[pop.X.worst_delay > 0, 'months_with_delay'].between(1, 6).all())
        self.assertEqual(set(pop.X.education), {1., 2., 3., 4.})
        self.assertEqual(set(pop.X.worst_delay), {-1., 0., 1., 2., 3., 4.})

        raw_linear, raw_nonlinear, excess = g._f_terms(pop.X, reference.median_income)
        common = .33 * ((raw_linear - reference.linear_means) / reference.linear_scales).sum(axis=1)
        common += .67 * ((raw_nonlinear - reference.nonlinear_means) / reference.nonlinear_scales).sum(axis=1)
        common_logit = self.f.intercept + self.f.scale * (common - reference.mixture_mean) / reference.mixture_scale
        jump = ((excess > 0) - reference.nonlinear_means[-1]) / reference.nonlinear_scales[-1]
        replacement = .5 * (excess - reference.female_debt_mean) / reference.female_debt_scale
        delta = .67 * (replacement - jump) * self.f.scale / reference.mixture_scale
        np.testing.assert_allclose(pop.p, expit(common_logit + np.where(pop.sex == 2, delta, 0.)), rtol=1e-13)

    def test_weighted_auc_averages_ties(self):
        scores = np.array([2., 1., 3., 1.])
        probability = np.array([.3, .1, .9, .1])
        comparison = (scores[:, None] > scores) + .5 * (scores[:, None] == scores)
        paired = probability[:, None] * (1 - probability)
        expected = (paired * comparison).sum() / paired.sum()
        order, starts = g._rank(scores)
        self.assertAlmostEqual(g._weighted_auc(probability[order], starts), expected)

    def test_wrong_calibration_or_size_fails(self):
        with self.assertRaises(ValueError):
            g.generate_population('F', 'fixture', 1, self.r)
        with self.assertRaises(ValueError):
            g.generate_population('R', 'fixture', 1, self.r, n=0)
        with self.assertRaises(ValueError):
            g.r_reference_curve(self.f)
        with self.assertRaises(ValueError):
            g.Calibration.from_dict(dict(setting='F', intercept=0, scale=1))


if __name__ == '__main__':
    unittest.main()
