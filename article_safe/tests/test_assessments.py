"""Assessment checks against direct weighted/sklearn calculations."""
import tempfile
from pathlib import Path
import unittest
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from article_safe.uncertainty import FixedAUC, conditional_intervals
from article_safe.fairness import contribution_table
from article_safe.explainability import auxiliary_regression
from article_safe.protocol import build_protocol


class AssessmentTests(unittest.TestCase):
    def test_fixed_weighted_auc_matches_expanded_records_with_ties(self):
        y=np.array([0,1,0,1,0,1])
        scores=np.array([.2,.2,.5,.5,.8,.9])
        weights=np.array([2,2,0,1,1,1])
        expanded=np.repeat(np.arange(len(y)),weights)
        self.assertAlmostEqual(FixedAUC(y,scores)(weights),roc_auc_score(y[expanded],scores[expanded]))
        self.assertTrue(np.isnan(FixedAUC(y,scores)(np.array([1,0,1,0,1,0]))))
        with self.assertRaises(ValueError):
            FixedAUC(y,scores)([-1,1,1,1,1,1])

    def test_fairness_signed_effects_and_missing_predictors_remain_visible(self):
        values=np.array([[1,0],[1,0],[-1,0],[-1,0]],dtype=float)
        sex=np.array([2,2,1,1])
        result=contribution_table(values,sex,['AGE','LIMIT_BAL'],['AGE'],'gbm').set_index('feature')
        self.assertEqual(result.loc['AGE','gini'],0)
        self.assertEqual(result.loc['AGE','signed_mean_gap'],2)
        self.assertTrue(result.loc['AGE','signed_cancellation'])
        self.assertEqual(result.loc['LIMIT_BAL','gini_status'],'no_measured_contribution')
        self.assertTrue(pd.isna(result.loc['LIMIT_BAL','ks_p_value']))

    def test_outcome_regression_has_explicit_target_and_covariance_variants(self):
        rng=np.random.default_rng(71)
        y=rng.integers(0,2,120)
        phi=rng.normal(size=(120,2))
        table,status=auxiliary_regression(y,phi,['AGE','LIMIT_BAL'],['AGE','LIMIT_BAL'],np.repeat(np.arange(60),2))
        self.assertEqual(status['status'],'ok')
        self.assertFalse(status['predictive_candidate'])
        self.assertEqual(set(table['covariance']),{'model_based','original_group_cluster'})
        self.assertTrue((table['response']=='observed default').all())
        cluster=table[table['covariance']=='original_group_cluster']
        self.assertTrue((cluster['reference_df']==59).all())

    def test_bootstrap_zero_model_difference_and_original_group_gap(self):
        protocol=deepcopy(build_protocol())
        protocol['inference']['bootstrap_repetitions']=40
        protocol['inference']['minimum_valid_bootstrap']=35
        protocol['final_comparisons']=[['a','b']]
        y=np.tile([0,1],20)
        sex=np.tile([1,2],20)
        groups=np.repeat(np.arange(20),2)
        scores=np.tile([.2,.8],20)
        values=np.c_[sex==2].astype(float)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            conditional_intervals(y,{'a':scores,'b':scores},{'a':values},sex,groups,['AGE'],protocol,path)
            auc=pd.read_csv(path/'auc_difference_intervals.csv')
            self.assertEqual(auc.iloc[0]['ci_lower'],0)
            self.assertEqual(auc.iloc[0]['ci_upper'],0)
            gap=pd.read_csv(path/'explanation_gap_intervals.csv')
            self.assertTrue((gap['ci_lower']==1).all())
            self.assertTrue((gap['ci_upper']==1).all())
            self.assertTrue((gap['valid_draws']==40).all())