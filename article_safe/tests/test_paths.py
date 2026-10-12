"""Exercise the LR path on an identifiable synthetic design despite credit-data aliases."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy.special import expit
from article_safe.paths import backward_path
from article_safe.protocol import build_protocol
from src.baseline import FEATURES


class BackwardPathIntegrationTest(unittest.TestCase):
    def test_identifiable_path_removes_a_whole_unnecessary_predictor(self):
        rng=np.random.default_rng(2026)
        n=400
        age=rng.normal(size=n)
        education=rng.integers(1,4,size=n)
        outcome=rng.binomial(1,expit(-.2+1.2*age+.8*(education==2)))
        X=pd.DataFrame(np.zeros((2*n,22)),columns=FEATURES)
        X['AGE']=np.repeat(age,2)
        X['EDUCATION']=np.repeat(education,2)
        # Within each pair, changing this predictor cannot improve the Bernoulli likelihood.
        X['LIMIT_BAL']=np.tile([-1.,1.],n)
        y=np.repeat(outcome,2)
        protocol=deepcopy(build_protocol())
        protocol['features']=['AGE','EDUCATION','LIMIT_BAL']
        with tempfile.TemporaryDirectory() as directory:
            choice,model,history,diagnostics=backward_path(X,y,protocol,Path(directory))
        self.assertEqual(choice['status'],'complete')
        self.assertNotIn('LIMIT_BAL',choice['selected_features'])
        self.assertIn('AGE',choice['selected_features'])
        self.assertTrue(model.estimable)
        self.assertTrue((history.loc[history.feature=='EDUCATION','df']==2).all())
        self.assertTrue(all(not row['regularized'] for row in diagnostics))