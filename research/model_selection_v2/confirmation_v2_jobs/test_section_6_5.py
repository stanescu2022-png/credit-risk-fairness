"""Section 6.5 reporting fixtures with precomputed scores and sealed job folders."""
from itertools import permutations
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.stats import t

from research.model_selection_v2.confirmation_v2.arms import ARMS, STRICT_ARM
from research.model_selection_v2.confirmation_v2.test_v2 import reporting_fixture
from research.model_selection_v2.pilot import CANDIDATES

from .artifacts import manifest, read_json, write_json
from .names import MARGIN_ARM, primary_summary, summarize
from .section_6_5 import (
    DIRECT_ARM,
    DIRECT_CHOICE,
    PRIMARY_ENDPOINT,
    add_comparisons,
    add_direct_choices,
    choose_direct_small_swap,
    collect,
    group_audit_records,
    verified_repeat,
)


def seal(folder, job, repeat):
    write_json(folder/'COMPLETE.json', dict(
        job=job, repeat=repeat, complete=True, artifact_manifest_sha256=manifest(folder)
    ))


def fixture():
    table = reporting_fixture().rename(columns={'choice_'+STRICT_ARM: 'choice_'+MARGIN_ARM})
    leader = np.select(
        [table.repeat.lt(20), table.repeat.lt(35)],
        ['random_forest', 'decision_tree'], default='gradient_boosting',
    )
    table['S_bar'] = np.where(table.candidate.eq(leader), .2, .1)
    # These inputs would select LR if A or E were used.
    table['A_bar'] = table['E_bar'] = table.candidate.eq('logistic').astype(float)
    return table


class DirectSmallSwapTests(unittest.TestCase):
    def test_selection_order_and_tolerance_anchor_to_maximum(self):
        scores = dict(zip(reversed(CANDIDATES), (0., 0., .5, .5 - .5e-12)))
        self.assertEqual(choose_direct_small_swap(scores), 'logistic')
        scores['logistic'] = .5 - 1.5e-12
        self.assertEqual(choose_direct_small_swap(scores), 'decision_tree')
        scores['random_forest'] = .5 + .75e-12
        self.assertEqual(choose_direct_small_swap(scores), 'decision_tree')
        scores['random_forest'] = .5 + 1.25e-12
        self.assertEqual(choose_direct_small_swap(scores), 'random_forest')
        scores = dict(zip(CANDIDATES, (-1e-12, 0., -1., -1.)))
        self.assertEqual(choose_direct_small_swap(scores), 'logistic')
        scores = dict.fromkeys(reversed(CANDIDATES), -1.0)
        self.assertEqual(choose_direct_small_swap(scores), 'logistic')

    def test_requires_four_defined_scores_and_compares_a_recorded_choice(self):
        scores = dict.fromkeys(CANDIDATES, -1.0)
        scores['random_forest'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite score'):
            choose_direct_small_swap(scores)
        del scores['random_forest']
        with self.assertRaisesRegex(ValueError, 'finite score'):
            choose_direct_small_swap(scores)
        table = fixture().loc[lambda rows: rows.repeat.eq(0)]
        recorded = add_direct_choices(table)
        self.assertTrue(recorded[DIRECT_CHOICE].eq('random_forest').all())
        with self.assertRaisesRegex(ValueError, 'Recorded direct choice'):
            add_direct_choices(recorded.assign(**{DIRECT_CHOICE: 'logistic'}))
        with self.assertRaisesRegex(ValueError, 'one selection S_bar'):
            add_direct_choices(pd.concat([table, table]))

    def test_all_matches_export_exact_zero_interval(self):
        table = fixture()
        table['S_bar'] = np.where(table.candidate.eq(table.choice_selector2), .1, -.1)
        table = add_direct_choices(table)
        result = primary_summary(table)
        add_comparisons(result, table, 'A')
        row = result['primary_contrasts'].set_index('contrast').loc['selector2_minus_'+DIRECT_ARM]
        self.assertEqual((row.estimate, row.lower, row.upper, row.disagreements), (0., 0., 0., 0))
        self.assertEqual(row.exact_zero_differences, 50)

    def test_historical_receipt_must_bind_required_files(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            path = folder/'per_repeat_primary.csv'
            seal(folder, 'A', 0)
            fixture().loc[lambda rows: rows.repeat.eq(0)].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'does not bind'):
                verified_repeat(folder, 'A', path.name)
            with (folder/'artifact_manifest.json').open('a') as stream:
                stream.write(' ')
            with self.assertRaisesRegex(ValueError, 'artifact manifest changed'):
                verified_repeat(folder, 'A', path.name)

    def test_candidate_level_audit_required_when_direct_model_was_unselected(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            row = dict(candidate='logistic', condition='J1', group='all', metric='brier', value=.14)
            write_json(folder/'J1_group_audit.json', [row])
            write_json(folder/'group_audit_by_arm.json', [dict(row, repeat=0, arm='safe')])
            with self.assertRaisesRegex(ValueError, 'Incomplete direct candidate group audit'):
                group_audit_records(folder, 0, 'gradient_boosting')

    def test_job_collectors_export_paired_intervals_and_keep_prior_results(self):
        table = fixture()
        pairs = list(permutations(CANDIDATES, 2))
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            a_folders, b_folders = [], []
            for repeat, saved in table.groupby('repeat', sort=False):
                repeat = int(repeat)
                a, b = root/f'A_{repeat}', root/f'B_{repeat}'
                a.mkdir()
                b.mkdir()
                a_folders.append(a)
                b_folders.append(b)
                saved.to_csv(a/'per_repeat_primary.csv', index=False)
                saved.to_csv(b/'per_repeat_full.csv', index=False)
                seal(a, 'A', repeat)
                write_json(b/'every_pair_gate_records.json', [
                    dict(repeat=repeat, challenger=left, incumbent=right, blocked=False, margin_blocked=False,
                         joint_half_width_female_male=[.001, .001], point_female_male=[0., 0.])
                    for left, right in pairs
                ])
                candidates = [
                    dict(candidate=candidate, condition=condition, group='female', metric='brier', value=.1 + i*.01)
                    for i, candidate in enumerate(CANDIDATES)
                    for condition in ('J1', 'clean')
                ]
                write_json(b/'J1_group_audit.json', candidates)
                write_json(b/'group_audit_by_arm.json', [
                    dict(row, repeat=repeat, arm=arm)
                    for arm in ARMS+(MARGIN_ARM,)
                    for row in candidates
                    if row['candidate'] == saved['choice_'+arm].iloc[0]
                ])
                seal(b, 'B', repeat)

            a_report = collect('A', reversed(a_folders), root/'report_A')
            b_report = collect('B', b_folders, root/'report_B')
            old_a, old_b = primary_summary(table), summarize(table)
            self.assertEqual(old_a['headline_flag'], a_report['headline_flag'])
            self.assertEqual(old_b['headline_flags'], read_json(root/'report_B/headline_flags.json'))
            self.assertEqual(old_b['fairness_summary'], b_report['fairness_summary'])
            for old, new, key in ((old_a, a_report, 'primary_contrasts'), (old_b, b_report, 'contrasts')):
                pd.testing.assert_frame_equal(old[key], new[key].iloc[:len(old[key])], check_exact=True)
                pd.testing.assert_frame_equal(
                    old['range_measure_contrasts'], new['range_measure_contrasts'], check_exact=True
                )
                pd.testing.assert_frame_equal(
                    old['pair_match_matrix'], new['pair_match_matrix'].loc[list(ARMS), list(ARMS)], check_exact=True
                )
                self.assertEqual(len(new['pair_matches']), 45)
                self.assertEqual(new['pair_match_matrix'].loc[DIRECT_ARM, 'selector2'], 20)

            for output, key in (('report_A', 'primary_contrasts'), ('report_B', 'contrasts')):
                values = pd.read_csv(root/output/(key+'.csv'))
                j1 = values.loc[values.endpoint.eq(PRIMARY_ENDPOINT)].set_index('contrast')
                for contrast, differences, disagreements in (
                    ('selector2_minus_'+DIRECT_ARM, [0.]*20 + [-.01]*10 + [.01]*5 + [.03]*15, 30),
                    ('selector2_minus_plain_loss', [0.]*30 + [.02]*20, 20),
                    (DIRECT_ARM+'_minus_safe', [-.02]*20 + [-.01]*15 + [-.03]*15, 50),
                    (DIRECT_ARM+'_minus_plain_loss', [0.]*20 + [.01]*15 + [-.01]*15, 30),
                ):
                    saved = j1.loc[contrast]
                    estimate = np.mean(differences)
                    half = t.ppf(.975, 49)*np.sqrt((1/50 + .25)*np.var(differences, ddof=1))
                    self.assertEqual(saved.disagreements, disagreements)
                    self.assertEqual(saved.exact_zero_differences, 50-disagreements)
                    self.assertAlmostEqual(saved.estimate, estimate)
                    self.assertAlmostEqual(saved.lower, estimate-half)
                    self.assertAlmostEqual(saved.upper, estimate+half)

            repeats = pd.read_csv(root/'report_B/per_repeat_full.csv').drop_duplicates('repeat')
            gbm_repeat = repeats.loc[repeats.repeat.eq(49)].iloc[0]
            self.assertEqual(gbm_repeat[DIRECT_CHOICE], 'gradient_boosting')
            self.assertTrue(all(gbm_repeat['choice_'+arm] != 'gradient_boosting' for arm in ARMS+(MARGIN_ARM,)))
            audits = pd.read_csv(root/'report_B/group_audit_intervals.csv')
            direct = audits.loc[audits.arm.eq(DIRECT_ARM)].set_index('condition')
            self.assertEqual(set(direct.index), {'clean', 'J1'})
            self.assertAlmostEqual(direct.loc['J1', 'estimate'], (20*.12 + 15*.11 + 15*.13)/50)
            self.assertFalse(pd.read_csv(a_folders[0]/'per_repeat_primary.csv').columns.isin([DIRECT_CHOICE]).any())
            self.assertFalse(pd.read_csv(b_folders[0]/'per_repeat_full.csv').columns.isin([DIRECT_CHOICE]).any())


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DirectSmallSwapTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
