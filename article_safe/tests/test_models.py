"""Real estimator/design tests on synthetic data; no project model is fitted."""
import tempfile
from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from article_safe.models import fit_design, fit_lr, fit_gbm
from article_safe.explainability import explain, auxiliary_regression
from article_safe.inference import likelihood_ratio
from article_safe.protocol import build_protocol
from src.baseline import FEATURES


class ModelAndExplanationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = build_protocol()
        rng = np.random.default_rng(101)
        n = 400
        cls.X = pd.DataFrame(np.zeros((n, 22)), columns=FEATURES)
        cls.X['AGE'] = rng.normal(size=n)
        cls.X['EDUCATION'] = rng.integers(1, 4, size=n)
        cls.X['LIMIT_BAL'] = rng.normal(size=n)
        probabilities = expit(.1 + .7 * cls.X.AGE + .5 * (cls.X.EDUCATION == 2))
        cls.y = pd.Series(rng.binomial(1, probabilities))
        with threadpool_limits(limits=1):
            cls.design = fit_design(cls.X, ['AGE', 'EDUCATION'], reference_encoding=True)
            cls.lr = fit_lr(cls.design, cls.y, ('AGE', 'EDUCATION'), cls.protocol)
            cls.reduced_lr = fit_lr(cls.design, cls.y, ('AGE',), cls.protocol)
            cls.gbm = fit_gbm(cls.X, cls.y, ['AGE', 'EDUCATION'], cls.protocol)

    def test_unpenalized_models_converge_and_nested_categorical_df(self):
        self.assertTrue(self.lr.estimable, self.lr.diagnostics)
        self.assertFalse(self.lr.diagnostics['regularized'])
        tested = likelihood_ratio(self.lr.diagnostics['log_likelihood'], self.reduced_lr.diagnostics['log_likelihood'],
                                  self.lr.diagnostics['design_rank'], self.reduced_lr.diagnostics['design_rank'], 400, 400)
        self.assertEqual(tested['df'], 2)
        self.assertGreaterEqual(tested['statistic'], 0)
        self.assertEqual(len(self.reduced_lr.columns), 1)

    def test_reference_encoding_and_unknown_category_are_explicit(self):
        self.assertEqual(self.design.mapping.feature.tolist().count('EDUCATION'), 2)
        self.assertEqual(set(self.design.mapping.reference.dropna()), {'1'})
        altered = self.X.iloc[:3].copy()
        altered['EDUCATION'] = 99
        self.assertEqual(self.lr.unknown_categories(altered)['EDUCATION'], 3)
        self.assertTrue(np.isfinite(self.lr.probability(altered)).all())

    def test_metadata_never_enters_prediction(self):
        augmented = self.X.assign(SEX=1, ID=np.arange(len(self.X)), default=self.y)
        np.testing.assert_array_equal(self.lr.probability(augmented), self.lr.probability(augmented.assign(SEX=2, ID=-1)))
        with self.assertRaises(ValueError):
            fit_design(augmented, ['AGE', 'SEX'], reference_encoding=True)

    def test_rank_deficiency_and_separation_are_not_silently_regularized(self):
        duplicate = self.X.assign(LIMIT_BAL=self.X.AGE)
        design = fit_design(duplicate, ['AGE', 'LIMIT_BAL'], reference_encoding=True)
        result = fit_lr(design, self.y, ('AGE', 'LIMIT_BAL'), self.protocol)
        self.assertEqual(result.diagnostics['reason'], 'rank_deficient_design')
        separated_X = self.X.iloc[:100].copy()
        separated_X['AGE'] = np.r_[np.full(50,-1.), np.full(50,1.)]
        design = fit_design(separated_X, ['AGE'], reference_encoding=True)
        result = fit_lr(design, np.r_[np.zeros(50),np.ones(50)], ('AGE',), self.protocol)
        self.assertFalse(result.estimable)
        self.assertFalse(result.diagnostics['regularized'])

    def test_reduced_gbm_is_a_new_smaller_fit(self):
        reduced = fit_gbm(self.X, self.y, ['AGE'], self.protocol)
        self.assertIsNot(reduced.estimator, self.gbm.estimator)
        self.assertEqual(reduced.estimator.n_features_in_, 1)
        self.assertGreater(self.gbm.estimator.n_features_in_, 1)

    def test_lr_and_gbm_shap_additivity_and_grouping(self):
        for model in (self.lr, self.gbm, self.reduced_lr):
            result = explain(model, self.X.iloc[:30], self.X.iloc[100:200], self.protocol['features'])
            np.testing.assert_allclose(result['base_value']+result['original'].sum(axis=1), model.margin(self.X.iloc[:30]), atol=1e-6)
            self.assertEqual(result['original'].shape, (30,22))
            absent = [j for j,f in enumerate(self.protocol['features']) if f not in model.features]
            np.testing.assert_array_equal(result['original'][:,absent], 0)

    def test_saved_model_reload_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.joblib'
            joblib.dump(self.lr,path)
            restored=joblib.load(path)
            np.testing.assert_array_equal(restored.probability(self.X),self.lr.probability(self.X))

    def test_outcome_regression_rejects_margins_and_reports_rank_deficiency(self):
        values=np.zeros((400,22))
        with self.assertRaises(ValueError):
            auxiliary_regression(np.linspace(-1,1,400),values,['AGE'],self.protocol['features'],np.arange(400))
        table,status=auxiliary_regression(self.y,values,['AGE'],self.protocol['features'],np.arange(400))
        self.assertIsNone(table)
        self.assertEqual(status['status'],'not_estimable')