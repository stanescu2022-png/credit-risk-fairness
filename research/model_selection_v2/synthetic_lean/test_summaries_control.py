"""Check independent-repeat reporting and the no-fit control with small fixtures."""

from collections import Counter
from itertools import product
import math
import unittest
from unittest.mock import Mock, patch

import numpy as np

from research.model_selection_v2.step1_v1.core import fairness

from . import control, summaries
from .generators import child_seed


LR, DT, RF, GBM = 'logistic', 'decision_tree', 'random_forest', 'gradient_boosting'


def _truth_table(values):
    return {model: {'all': loss, 'female': loss, 'male': loss}
            for model, loss in zip((LR, DT, RF, GBM), values)}


def _r_repeat(repeat, direct, final, losses):
    return {
        'setting': 'R', 'repeat': repeat,
        'choices': dict(safe=LR, plain_loss=LR, direct_small_swap_brier=direct,
                        selector2_before_check=final, selector2=final),
        'judge_truth': {condition: _truth_table(values)
                        for condition, values in zip(('clean', 'J1', 'J3'), losses)},
    }


def _attempt(candidate, rank, blocked, female, male, triggers=()):
    deltas = {'female': female, 'male': male}
    return dict(candidate=candidate, rank=rank, blocked=blocked, truth_delta=deltas,
                triggering_groups=list(triggers),
                harmful_groups=[group for group, delta in deltas.items() if delta > 0],
                material_harm_groups=[group for group, delta in deltas.items() if delta >= .0005])


def _f_repeat(repeat, first, final, direct, *, later=(), prevented=False):
    harmless = _attempt(LR, 0, False, 0., 0.)
    attempts = ([first] if first else []) + list(later)
    chosen = next((attempt for attempt in attempts if attempt['candidate'] == final), harmless)
    j1 = {
        model: {'all': (female + male) / 2, 'female': female, 'male': male}
        for model, female, male in zip((LR, DT, RF, GBM), (.11, .10, .08, .07), (.09, .08, .08, .07))
    }
    return {
        'setting': 'F', 'repeat': repeat,
        'choices': dict(safe=LR, plain_loss=LR, direct_small_swap_brier=direct['candidate'],
                        selector2_before_check=first['candidate'] if first else LR, selector2=final),
        'judge_truth': {
            'clean': _truth_table((.05, .04, .03, .02)),
            'J1': j1,
        },
        'gate': dict(started_at_safe=first is None, first_attempt=first, later_attempts=later,
                     final_truth_delta=chosen['truth_delta'], final_harmful=bool(chosen['harmful_groups']),
                     prevented_harm=prevented, direct_harmful_groups=direct['harmful_groups'],
                     direct_material_harm_groups=direct['material_harm_groups']),
    }


class SummaryTests(unittest.TestCase):
    def test_r_reversals_use_same_repeat_and_report_full_selector_separately(self):
        rows = [
            _r_repeat(0, DT, LR, ((.02, .03, .04, .05), (.02, .01, .04, .05), (.02, .03, .04, .05))),
            _r_repeat(1, DT, RF, ((.02, .03, .04, .05), (.02, .03, .04, .05), (.02, .01, .04, .05))),
            _r_repeat(2, LR, LR, ((.02, .03, .04, .05), (.02, .01, .04, .05), (.02, .01, .04, .05))),
            _r_repeat(3, DT, LR, ((.02, .02, .04, .05), (.02, .01, .04, .05), (.02, .01, .04, .05))),
        ]
        result = summaries.summarize(rows)
        self.assertEqual(result['same_repeat_reversals'], {'J1': 1, 'J3': 1})
        contrasts = result['contrasts']
        direct = contrasts['direct_minus_clean']
        self.assertEqual(direct['clean']['all']['disagreements'], 3)
        self.assertEqual(direct['clean']['all']['exact_zero_differences'], 2)
        self.assertEqual(direct['J1']['all']['negative_differences'], 2)
        self.assertEqual(direct['J3']['all']['positive_differences'], 1)
        full = contrasts['selector2_minus_clean']
        for condition in ('clean', 'J1', 'J3'):
            for group in summaries.GROUPS:
                self.assertAlmostEqual(full[condition][group]['estimate'], .005)
                self.assertEqual(full[condition][group]['disagreements'], 1)
                self.assertEqual(full[condition][group]['exact_zero_differences'], 3)
        self.assertEqual(result['choice_counts']['selector2'], {LR: 3, DT: 0, RF: 1, GBM: 0})

    def test_f_tracks_first_veto_to_final_choice_and_scores_unvisited_direct(self):
        harmless = _attempt(LR, 0, False, 0., 0.)
        first = _attempt(GBM, 1, True, .001, 0., ('female',))
        later = _attempt(RF, 2, False, -.001, .0005)
        unvisited_direct = _attempt(DT, 3, False, -.001, .002)
        prevented = _attempt(DT, 1, True, .00001, 0., ('male',))
        harmless_veto = _attempt(GBM, 1, True, -.001, 0., ('male',))
        harmful_allowed = _attempt(RF, 1, False, .001, 0.)
        rows = [
            _f_repeat(0, first, RF, unvisited_direct, later=(later,)),
            _f_repeat(1, None, LR, unvisited_direct),
            _f_repeat(2, prevented, LR, harmless, prevented=True),
            _f_repeat(3, harmless_veto, LR, harmless),
            _f_repeat(4, harmful_allowed, RF, harmful_allowed),
        ]
        result = summaries.summarize(rows)
        gate = result['gate']
        self.assertEqual(gate['started_at_safe'], summaries.wilson(1, 5))
        self.assertEqual(gate['first_proposal_rates'], {
            'harmful_blocked': summaries.wilson(2, 3),
            'harmful_allowed': summaries.wilson(1, 3),
            'harmless_blocked': summaries.wilson(1, 1),
        })
        self.assertEqual(gate['first_proposals']['by_group']['male']['triggered_blocks'], 2)
        self.assertEqual(gate['first_proposals']['by_group']['male']['harmful'], 0)
        self.assertEqual(gate['later_visited_attempts']['by_rank']['2']['material_harmful_allowed'], 1)
        self.assertEqual(gate['harmful_first_blocked_but_final_harmful'], 1)
        self.assertEqual(gate['prevented_harm'], summaries.wilson(1, 3))
        self.assertEqual(gate['final_harm'], summaries.wilson(2, 5))
        self.assertEqual(gate['final_material_harm'], summaries.wilson(2, 5))
        self.assertEqual(gate['final_harm_by_group']['male'], summaries.wilson(1, 5))
        self.assertEqual(gate['direct_choice'], {
            'repeats': 5, 'harmful': 3, 'material_harmful': 3, 'harmful_and_final_harmless': 1,
            'disagreements_with_final': 2,
        })
        for group, mean in (('all', .010), ('female', .012), ('male', .008)):
            paired = result['contrasts']['final_minus_pre']['J1'][group]
            self.assertAlmostEqual(paired['estimate'], mean)
            self.assertEqual(paired['disagreements'], 3)
            self.assertEqual(paired['exact_zero_differences'], 2)

    def test_no_opportunity_has_no_rate_and_identical_choices_have_zero_interval(self):
        safe = _attempt(LR, 0, False, 0., 0.)
        result = summaries.summarize(_f_repeat(repeat, None, LR, safe) for repeat in range(2))
        gate = result['gate']
        for rate in (*gate['first_proposal_rates'].values(), gate['prevented_harm']):
            self.assertEqual(rate, dict(numerator=0, denominator=0, estimate=None,
                                        lower=None, upper=None, reason='no opportunity'))
        for condition in ('clean', 'J1'):
            paired = result['contrasts']['final_minus_pre'][condition]['all']
            self.assertEqual((paired['estimate'], paired['lower'], paired['upper']), (0., 0., 0.))
            self.assertEqual(paired['disagreements'], 0)
            self.assertEqual(paired['exact_zero_differences'], 2)

    def test_independent_t_interval_and_wilson_endpoints(self):
        interval = summaries.independent_t_interval([.01, .03, -.01])
        half_width = 4.3026527297 * .02 / math.sqrt(3)
        self.assertAlmostEqual(interval['estimate'], .01)
        self.assertAlmostEqual(interval['lower'], .01 - half_width)
        self.assertAlmostEqual(interval['upper'], .01 + half_width)
        self.assertEqual((interval['repeats'], interval['df'], interval['rho']), (3, 2, 0.))
        constant = summaries.independent_t_interval([.01] * 50)
        self.assertEqual((constant['estimate'], constant['lower'], constant['upper']), (.01, .01, .01))
        self.assertAlmostEqual(summaries.wilson(0, 10)['upper'], .2775328)
        self.assertAlmostEqual(summaries.wilson(10, 10)['lower'], 1 - .2775328)


class ControlTests(unittest.TestCase):
    def test_multinomial_cells_match_enumerated_borrower_bootstrap(self):
        # Five singleton borrowers, including two in the same sex-by-outcome cell.
        borrower_cells = [0, 0, 1, 2, 3]
        observed = np.bincount(borrower_cells).reshape(2, 2)
        sizes = len(borrower_cells)
        draws = list(product(range(sizes), repeat=sizes))
        cells = np.array([np.bincount([borrower_cells[i] for i in draw], minlength=4)
                          for draw in draws])
        counts = Counter(map(tuple, cells))
        for occupancy, frequency in counts.items():
            enumerated = frequency / sizes ** sizes
            mass = math.factorial(sizes) * math.prod((p / sizes) ** k / math.factorial(k)
                                                    for p, k in zip(observed.ravel(), occupancy))
            self.assertAlmostEqual(enumerated, mass, places=15)

        rng = Mock()
        rng.multinomial.return_value = cells
        cell_draws = control._sample_bootstrap_cells(observed, rng, len(draws))
        n, probabilities = rng.multinomial.call_args.args
        self.assertEqual(n, sizes)
        np.testing.assert_array_equal(probabilities, np.array([.4, .2, .2, .2]))
        self.assertEqual(rng.multinomial.call_args.kwargs, {'size': len(draws)})

        labels = np.array(borrower_cells) % 2
        sex = np.array(borrower_cells) // 2
        row_losses = (.27 - labels)**2 - (.17 - labels)**2
        keep = cell_draws.sum(axis=-1).min(axis=1) > 0
        self.assertFalse(keep.all())
        draw_ids = np.asarray(draws)[keep]
        point = np.array([row_losses[sex == group].mean() for group in range(2)])
        direct = np.column_stack([(row_losses[draw_ids] * (sex[draw_ids] == group)).sum(axis=1)
                                  / (sex[draw_ids] == group).sum(axis=1) for group in range(2)])
        expected = fairness(point, direct)
        actual = control._gate_from_cells(observed, cell_draws[keep])
        np.testing.assert_allclose(actual['point_female_male'], expected['point_female_male'], atol=1e-15)
        np.testing.assert_allclose(actual['joint_intervals_female_male'],
                                   expected['joint_intervals_female_male'], atol=1e-15)
        self.assertEqual(actual['blocked'], expected['blocked'])
        with self.assertRaisesRegex(ValueError, 'Empty bootstrap sex subgroup'):
            control._gate_from_cells(observed, cell_draws)

    def test_veto_requires_positive_joint_lower_bound(self):
        sizes = np.array([2880, 1920])
        cases = (([500, 600], True), ([800, 600], False), ([800, 300], True), ([633, 422], False))
        for events, expected in cases:
            with self.subTest(events=events):
                events = np.array(events)
                draw_events = events + np.arange(-10, 11)[:, None] * np.array([1, -2])
                observed = np.stack((sizes - events, events), axis=-1)
                draws = np.stack((sizes - draw_events, draw_events), axis=-1)
                point = .044 - .20 * events / sizes
                differences = .044 - .20 * draw_events / sizes
                sd = differences.std(axis=0, ddof=1)
                critical = np.quantile(np.max(np.abs((differences - point) / sd), axis=1),
                                       .95, method='higher')
                bounds = np.column_stack((point - critical * sd, point + critical * sd))
                actual = control._gate_from_cells(observed, draws)
                np.testing.assert_allclose(actual['joint_intervals_female_male'], bounds, atol=1e-15)
                self.assertEqual(actual['blocked'], expected)

    def test_control_uses_fixed_dimensions_and_separate_purpose_and_bootstrap_seeds(self):
        labels = np.zeros(4800, dtype=int)
        labels[:500] = 1
        labels[2880:3480] = 1
        exact_cells = np.array([[2380, 500], [1320, 600]])
        outcome_rng = Mock()
        outcome_rng.binomial.return_value = labels
        bootstrap_rng = Mock()
        bootstrap_rng.multinomial.return_value = np.tile(exact_cells.ravel(), (2000, 1))
        with patch.object(control.np.random, 'default_rng',
                          side_effect=[outcome_rng, bootstrap_rng] * 2) as make_rng:
            production = control.control_trials([7])
            timing = control.control_trials([7], purpose='timing')
        seeds = [child_seed(purpose, 'control', 7, field)
                 for purpose in ('control', 'timing') for field in ('outcome', 'bootstrap')]
        self.assertEqual(len(set(seeds)), 4)
        self.assertEqual([call.args[0] for call in make_rng.call_args_list], seeds)
        outcome_rng.binomial.assert_called_with(1, .22, size=4800)
        self.assertEqual(outcome_rng.binomial.call_count, 2)
        for call in bootstrap_rng.multinomial.call_args_list:
            self.assertEqual((call.args[0], call.kwargs['size']), (4800, 2000))
            np.testing.assert_array_equal(call.args[1], exact_cells.ravel() / 4800)
        self.assertEqual(production[0]['observed_cells_female_male'], exact_cells.tolist())
        self.assertEqual(production[0]['purpose'], 'control')
        self.assertEqual(timing[0]['purpose'], 'timing')
        self.assertEqual(control.CONTROL_REPEATS, 1000)
        summary = control.summarize_control(production)
        self.assertEqual((summary['female_borrowers_per_trial'], summary['male_borrowers_per_trial']),
                         (2880, 1920))
        self.assertEqual(summary['false_vetoes'], summaries.wilson(0, 1))
        for forecast in summary['forecasts'].values():
            self.assertAlmostEqual((forecast - summary['true_default_probability'])**2,
                                   summary['truth_squared_error_per_forecast'])
        with self.assertRaises(ValueError):
            control.summarize_control([*production, dict(timing[0], repeat=8)])
        with self.assertRaises(ValueError):
            control.control_trials([True])


if __name__ == '__main__':
    unittest.main()
