"""Focused checks for training-only diagnostics and group-aware uncertainty."""
import json
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                            recall_score, f1_score, brier_score_loss, confusion_matrix)

from src.baseline import DATA_PATH, FEATURES, load_split, sha256, ROOT
from src.performance_evaluation import (TABLE_DIR, METRICS, training_folds, class_metrics,
    threshold_analysis, reliability_bins, bootstrap_group_weights, performance_metrics,
    group_bootstrap, predictor_groups)


class PerformanceMethodTests(unittest.TestCase):
    def test_threshold_candidates_handle_ties_and_reject_holdout(self):
        part = pd.DataFrame({"model": "example", "partition": "training_oof",
                             "actual": [1, 1, 1, 1, 0, 0],
                             "probability_default": [.8, .5, .5, .2, .6, .1]})
        grid, candidates = threshold_analysis(part)
        fixed = grid[grid.threshold == .5].iloc[0]
        self.assertEqual(fixed.Recall, .25)  # exact .5 ties remain negative
        for candidate in candidates.itertuples(index=False):
            chosen = class_metrics(part.actual, part.probability_default > candidate.threshold)
            self.assertGreaterEqual(chosen["Recall"], candidate.recall_target)
            excluded_boundary = class_metrics(part.actual, part.probability_default > candidate.included_score_boundary)
            self.assertLess(excluded_boundary["Recall"], candidate.recall_target)
        with self.assertRaises(ValueError):
            threshold_analysis(part.assign(partition="locked_holdout"))

    def test_reliability_edges_empty_bins_and_ece(self):
        table = reliability_bins([0, 0, 1, 1], [0, .1, .5, 1.0])
        self.assertEqual(table["count"].sum(), 4)
        self.assertEqual(table.loc[0, "count"], 1)
        self.assertEqual(table.loc[1, "count"], 1)
        self.assertEqual(table.loc[5, "count"], 1)
        self.assertEqual(table.loc[9, "count"], 1)
        self.assertAlmostEqual(table.ece_contribution.sum(), (.1 + .5) / 4)
        self.assertTrue(table.loc[table["count"] == 0, "mean_probability"].isna().all())

    def test_group_weights_equal_expanded_whole_groups(self):
        groups = np.array([10, 10, 20, 30, 30, 30])
        y = np.array([0, 1, 0, 0, 1, 1])
        probability = np.array([.2, .2, .6, .7, .7, .7])
        weights = bootstrap_group_weights(groups, np.random.default_rng(42))
        for group in np.unique(groups):
            self.assertEqual(len(np.unique(weights[groups == group])), 1)
        draws = np.random.default_rng(42).integers(0, 3, size=3)
        expanded = np.concatenate([np.flatnonzero(groups == np.unique(groups)[index]) for index in draws])
        actual = performance_metrics(y, probability, weights=weights)
        expected = performance_metrics(y[expanded], probability[expanded])
        for metric in METRICS:
            self.assertAlmostEqual(actual[metric], expected[metric], places=13)
        self.assertEqual(int(weights.sum()), len(expanded))

    def test_bootstrap_is_seeded_and_retains_fixed_predictions(self):
        part = pd.DataFrame({"model": "example", "partition": "locked_holdout",
            "ID": np.arange(6), "duplicate_group_id": [1, 1, 2, 2, 3, 3],
            "actual": [0, 1, 0, 1, 0, 1], "predicted": [0, 0, 1, 1, 1, 1],
            "probability_default": [.2, .2, .6, .6, .7, .7]})
        intervals, replicates = group_bootstrap(part, repetitions=25, seed=42)
        intervals2, replicates2 = group_bootstrap(part, repetitions=25, seed=42)
        pd.testing.assert_frame_equal(intervals, intervals2)
        pd.testing.assert_frame_equal(replicates, replicates2)
        self.assertTrue((intervals.valid_repetitions == 25).all())


class PerformanceArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATA_PATH)
        cls.X, cls.X_test, cls.y, cls.y_test, cls.metadata, cls.test_metadata = load_split(cls.df)

    def test_cv_uses_training_only_and_isolates_actual_predictors(self):
        folds, assignment, groups = training_folds(self.X, self.y, self.metadata)
        self.assertEqual(len(folds), 5)
        self.assertFalse(set(self.metadata.ID) & set(self.test_metadata.ID))
        for fit, validation in folds:
            self.assertFalse(set(map(tuple, self.X.iloc[fit][FEATURES].to_numpy())) &
                             set(map(tuple, self.X.iloc[validation][FEATURES].to_numpy())))
        self.assertTrue(pd.DataFrame({"group": groups, "fold": assignment}).groupby("group").fold.nunique().eq(1).all())
        # SEX and outcomes are absent from the grouping interface; metadata SEX changes do not alter groups.
        altered = self.metadata.copy()
        altered["SEX"], altered["default"] = 3 - altered.SEX, 1 - altered.default
        np.testing.assert_array_equal(groups, predictor_groups(self.X, altered))

    def test_completed_outputs_and_locked_hashes(self):
        manifest_path = TABLE_DIR / "run_manifest.json"
        if not manifest_path.exists():
            self.skipTest("Run script_performance_evaluation.py to generate evaluation artifacts first.")
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest["locked_input_hashes"].items():
            self.assertEqual(sha256(ROOT / name), expected, name)
        for name, expected in manifest["output_hashes"].items():
            self.assertEqual(sha256(ROOT / name), expected, name)
        self.assertIsNone(manifest["thresholds"]["selected_threshold"])
        oof = pd.read_csv(TABLE_DIR / "oof_predictions.csv", float_precision="round_trip")
        for kind, part in oof.groupby("model"):
            self.assertEqual(len(part), len(self.X))
            self.assertTrue(part.ID.is_unique)
            np.testing.assert_array_equal(part.ID, self.metadata.ID)
            np.testing.assert_array_equal(part.SEX, self.metadata.SEX)
            np.testing.assert_array_equal(part.actual, self.y)
            self.assertFalse(set(part.ID) & set(self.test_metadata.ID))
            self.assertTrue(part.groupby("duplicate_group_id").fold.nunique().eq(1).all())
            self.assertTrue(part.probability_default.between(0, 1).all())
            np.testing.assert_array_equal(part.predicted, part.probability_default > .5)
        candidates = pd.read_csv(TABLE_DIR / "recall_target_candidates.csv", float_precision="round_trip")
        for row in candidates.itertuples(index=False):
            part = oof[oof.model == row.model]
            recalculated = class_metrics(part.actual, part.probability_default > row.threshold)
            self.assertAlmostEqual(recalculated["Recall"], row.Recall, places=13)
            self.assertGreaterEqual(recalculated["Recall"], row.recall_target)
        fold_metrics = pd.read_csv(TABLE_DIR / "cv_fold_metrics.csv")
        for row in fold_metrics.itertuples(index=False):
            part = oof[(oof.model == row.model) & (oof.fold == row.fold)]
            saved = fold_metrics[(fold_metrics.model == row.model) & (fold_metrics.fold == row.fold)].iloc[0]
            for name, metric in [("Accuracy", accuracy_score), ("Balanced Accuracy", balanced_accuracy_score),
                                  ("Precision", precision_score), ("Recall", recall_score), ("F1", f1_score)]:
                self.assertAlmostEqual(saved[name], metric(part.actual, part.predicted), places=12)
            self.assertAlmostEqual(saved["Brier Score"], brier_score_loss(part.actual, part.probability_default), places=12)
            tn, fp, fn, tp = confusion_matrix(part.actual, part.predicted).ravel()
            self.assertAlmostEqual(saved.Specificity, tn / (tn + fp), places=12)
        intervals = pd.read_csv(TABLE_DIR / "holdout_confidence_intervals.csv")
        replicates = pd.read_csv(TABLE_DIR / "holdout_bootstrap_replicates.csv")
        for row in intervals.itertuples(index=False):
            values = replicates[replicates.model == row.model][row.metric]
            np.testing.assert_allclose([row.ci_lower, row.ci_upper], np.quantile(values, [.025, .975]), rtol=0, atol=1e-14)
        summary = pd.read_csv(TABLE_DIR / "cv_summary_metrics.csv")
        for row in summary.itertuples(index=False):
            values = fold_metrics[fold_metrics.model == row.model][row.metric]
            self.assertAlmostEqual(row.mean, values.mean(), places=13)
            self.assertAlmostEqual(row.std, values.std(ddof=1), places=13)


if __name__ == "__main__":
    unittest.main()
