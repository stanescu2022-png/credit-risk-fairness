"""Focused checks for the fixed-model audit and authentic SAFE integration."""
import inspect
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from safeaipackage import core, check_fairness

from src.baseline import ROOT, DATA_PATH, FEATURES, load_split, sha256
from src.performance_evaluation import bootstrap_group_weights
from src.safe_adapter import SafePipelineAdapter, evaluate_safe, provenance
from src.fairness_evaluation import TABLE_DIR, RATE_METRICS, metrics, gaps, fairness_bootstrap


class FairnessMethodTests(unittest.TestCase):
    def test_confusion_metrics_and_gap_direction(self):
        male = metrics([0, 0, 1, 1], [0, 1, 0, 1])
        female = metrics([0, 0, 1, 1], [0, 0, 0, 1])
        self.assertEqual([male[k] for k in ['TN', 'FP', 'FN', 'TP']], [1, 1, 1, 1])
        result = gaps(male, female)
        for name in RATE_METRICS:
            self.assertAlmostEqual(result[name + '_difference'], female[name] - male[name])
        self.assertEqual(result['selection_rate_difference'], -.25)
        self.assertEqual(result['demographic_parity_difference'], .25)
        self.assertEqual(result['selection_rate_ratio_female_to_male'], .5)
        self.assertEqual(result['demographic_parity_ratio_min_to_max'], .5)
        self.assertTrue(np.isnan(metrics([0, 1], [0, 0])['precision']))

    def test_bootstrap_preserves_mixed_sex_groups(self):
        groups = np.repeat([1, 2, 3], 4)
        part = pd.DataFrame({'ID': np.arange(12), 'SEX': np.tile([1, 1, 2, 2], 3),
            'model': 'example', 'duplicate_group_id': groups,
            'actual': np.tile([0, 1, 0, 1], 3), 'predicted': np.repeat([0, 1, 1], 4),
            'probability_default': np.repeat([.2, .6, .8], 4)})
        weights = bootstrap_group_weights(groups, np.random.default_rng(42))
        for group in np.unique(groups):
            self.assertEqual(np.unique(weights[groups == group]).size, 1)
        first = fairness_bootstrap(part, repetitions=4)
        second = fairness_bootstrap(part, repetitions=4)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first.query('scope == "female_vs_male"').recall_difference.eq(0).all())


class FairnessArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, _, cls.y, cls.meta_train, cls.meta_test = load_split(pd.read_csv(DATA_PATH))

    def test_safe_adapter_and_actual_package_calls(self):
        source = provenance()
        self.assertEqual(source['version'], '0.8.3')
        self.assertIn('site-packages', inspect.getfile(core.rga))
        self.assertIn('site-packages', inspect.getfile(check_fairness.compute_rga_parity))
        for kind in ('logistic', 'gradient_boosting'):
            path = ROOT / f'models/baseline_{kind}.joblib'
            before = sha256(path)
            pipeline = joblib.load(path)
            audit = self.X_test.reset_index(drop=True).assign(SEX=self.meta_test.SEX.to_numpy())
            adapter = SafePipelineAdapter(pipeline)
            self.assertTrue(is_classifier(adapter))
            original = pipeline.predict_proba(self.X_test)
            with patch.object(pipeline, 'predict_proba', wraps=pipeline.predict_proba) as predict:
                np.testing.assert_array_equal(adapter.predict_proba(audit), original)
                supplied = predict.call_args.args[0]
                self.assertEqual(set(supplied.columns), set(FEATURES))
                self.assertFalse({'ID', 'SEX', 'default'} & set(supplied.columns))
            changed = audit.copy()
            changed.SEX = 3 - changed.SEX
            np.testing.assert_array_equal(adapter.predict_proba(changed), original)
            with self.assertRaises(RuntimeError):
                adapter.fit(audit, self.y)
            with self.assertRaises(ValueError):
                adapter.predict_proba(audit.assign(ID=1))
            with patch.object(check_fairness, 'compute_rga_parity', wraps=check_fairness.compute_rga_parity) as parity_call:
                with patch.object(core, 'rga', wraps=core.rga) as rga_call:
                    overall, group_rga, parity, raw = evaluate_safe(pipeline, self.X_train, self.X_test,
                        self.meta_train, self.meta_test, self.y, original[:, 1])
                    parity_call.assert_called_once()
                    self.assertEqual(rga_call.call_count, 3)
                    self.assertAlmostEqual(parity, abs(group_rga[2] - group_rga[1]), places=14)
            self.assertEqual(sha256(path), before)

    def test_completed_artifacts_alignment_and_consistency(self):
        if not (TABLE_DIR / 'run_manifest.json').exists():
            self.skipTest('Run script_fairness_evaluation.py first.')
        manifest = json.loads((TABLE_DIR / 'run_manifest.json').read_text())
        for category in ('protected_artifact_hashes', 'source_hashes', 'output_hashes'):
            for name, expected in manifest[category].items():
                self.assertEqual(sha256(ROOT / name), expected, name)
        read = lambda name: pd.read_csv(TABLE_DIR / f'{name}.csv', float_precision='round_trip')
        groups, disparity, predictions = read('group_metrics_threshold_050'), read('fairness_gaps_threshold_050'), read('audit_predictions')
        summary = read('group_summary')
        self.assertEqual(summary.records.sum(), 5996)
        self.assertEqual(summary.defaults.sum(), self.y.sum())
        self.assertEqual(set(zip(summary.SEX, summary.group)), {(1, 'male'), (2, 'female')})
        for kind, part in predictions.groupby('model', sort=False):
            self.assertTrue(part.ID.is_unique)
            np.testing.assert_array_equal(part.ID, self.meta_test.ID)
            np.testing.assert_array_equal(part.SEX, self.meta_test.SEX)
            np.testing.assert_array_equal(part.actual, self.y)
            saved = pd.read_csv(ROOT / f'outputs/tables/baseline_{kind}_predictions.csv', float_precision='round_trip')
            np.testing.assert_array_equal(part.probability_default, saved.probability_default)
            np.testing.assert_array_equal(part.predicted, saved.predicted)
            gm = groups[groups.model == kind].set_index('group')
            self.assertEqual(gm.records.sum(), 5996)
            for label in ('male', 'female'):
                subset = part[part.group == label]
                np.testing.assert_array_equal(gm.loc[label, ['TN', 'FP', 'FN', 'TP']], confusion_matrix(subset.actual, subset.predicted).ravel())
                for name, function in [('precision', precision_score), ('recall', recall_score), ('f1', f1_score)]:
                    self.assertAlmostEqual(gm.loc[label, name], function(subset.actual, subset.predicted), places=13)
            gd = disparity[disparity.model == kind].set_index('metric')
            for name in RATE_METRICS:
                self.assertAlmostEqual(gd.loc[name + '_difference', 'value'], gm.loc['female', name] - gm.loc['male', name], places=13)
            sweep = read('threshold_fairness_sensitivity').query('model == @kind')
            self.assertEqual(len(sweep), 13)
            fixed = sweep[sweep.threshold == .5].iloc[0]
            self.assertAlmostEqual(fixed.recall_difference, gd.loc['recall_difference', 'value'], places=13)
        ci, replicates = read('fairness_confidence_intervals'), read('bootstrap_replicates')
        for row in ci.itertuples(index=False):
            values = replicates.query('model == @row.model and scope == @row.scope')[row.metric].dropna()
            self.assertEqual(len(values), row.valid_repetitions)
            np.testing.assert_allclose([row.ci_lower, row.ci_upper], np.quantile(values, [.025, .975]), rtol=0, atol=1e-13)


if __name__ == '__main__':
    unittest.main()
