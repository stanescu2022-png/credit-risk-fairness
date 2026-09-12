"""Focused regression checks for duplicate isolation and saved-artifact alignment."""
import unittest
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                            recall_score, f1_score, roc_auc_score, average_precision_score)
from src.baseline import DATA_PATH, ROOT, FEATURES, load_split, split_metadata


class StabilizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATA_PATH)
        cls.meta = pd.read_csv(DATA_PATH.parent / 'split_metadata.csv')

    def test_all_records_preserved_and_predictors_disjoint(self):
        raw = pd.read_excel(ROOT / 'data/raw/default of credit card clients.xls', header=1)
        self.assertEqual(set(raw.ID), set(self.df.ID))
        self.assertEqual(len(self.df), len(raw))
        train, test, _, _, tr_meta, te_meta = load_split(self.df)
        self.assertFalse(set(tr_meta.ID) & set(te_meta.ID))
        # Check actual values independently of stored group IDs.
        self.assertFalse(set(map(tuple, train[FEATURES].to_numpy())) &
                         set(map(tuple, test[FEATURES].to_numpy())))
        self.assertFalse({'ID', 'SEX', 'default'} & set(train.columns))

    def test_split_reproduces_and_metadata_order_matches(self):
        pd.testing.assert_frame_equal(split_metadata(self.df), self.meta)
        for name in ('train', 'test'):
            expected = self.meta[self.meta.split == name].sort_values('partition_order').reset_index(drop=True)
            pd.testing.assert_frame_equal(expected, pd.read_csv(DATA_PATH.parent / f'{name}_metadata.csv'))

    def test_groups_ignore_sex_and_outcome(self):
        sample = self.df.iloc[:20].copy().reset_index(drop=True)
        sample.loc[1, FEATURES] = sample.loc[0, FEATURES]
        sample.loc[1, 'SEX'] = 3 - sample.loc[0, 'SEX']
        sample.loc[1, 'default'] = 1 - sample.loc[0, 'default']
        meta = split_metadata(sample)
        self.assertEqual(meta.loc[0, 'duplicate_group_id'], meta.loc[1, 'duplicate_group_id'])
        self.assertEqual(meta.loc[0, 'split'], meta.loc[1, 'split'])
        self.assertEqual(len(meta), len(sample))

    def test_saved_models_predictions_and_metrics_align(self):
        _, X_test, _, y_test, _, metadata = load_split(self.df)
        for kind in ('logistic', 'gradient_boosting'):
            with self.subTest(model=kind):
                model = joblib.load(ROOT / f'models/baseline_{kind}.joblib')
                predictions = pd.read_csv(ROOT / f'outputs/tables/baseline_{kind}_predictions.csv')
                np.testing.assert_array_equal(predictions.ID, metadata.ID)
                np.testing.assert_array_equal(predictions.SEX, metadata.SEX)
                np.testing.assert_array_equal(predictions.actual, y_test)
                np.testing.assert_array_equal(predictions.predicted, model.predict(X_test))
                np.testing.assert_allclose(predictions.probability_default, model.predict_proba(X_test)[:, 1], rtol=0, atol=1e-15)
                self.assertFalse({'ID', 'SEX', 'default'} & set(model.feature_names_in_))
                self.assertEqual(set(model.feature_names_in_), set(FEATURES))
                saved = pd.read_csv(ROOT / f'outputs/tables/baseline_{kind}_metrics.csv').set_index('Metric').Value
                for name, function in [('Accuracy', accuracy_score), ('Balanced Accuracy', balanced_accuracy_score),
                                       ('Precision', precision_score), ('Recall', recall_score), ('F1', f1_score),
                                       ('ROC-AUC', roc_auc_score), ('Average Precision (AP)', average_precision_score)]:
                    values = predictions.probability_default if name in ('ROC-AUC', 'Average Precision (AP)') else predictions.predicted
                    self.assertAlmostEqual(saved[name], function(y_test, values), places=13)


if __name__ == '__main__':
    unittest.main()
