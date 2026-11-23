"""Hand fixtures for runtime amendment, naming and primary/deferred reports."""
import json
from pathlib import Path
import tempfile
import unittest
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from research.model_selection_v2.confirmation_v2.test_v2 import reporting_fixture
from research.model_selection_v2.confirmation_v2 import arms as historical
from .names import rename,primary_summary,summarize,MARGIN_ARM,MARGIN_LABEL
from .parallel import budget_decision
from .artifacts import (compare_models,compare_repeat_folders,exact,manifest,model_snapshot,
    save_reloaded_model,sha,verify_model_files,write_json)


class JobFixtures(unittest.TestCase):
    def test_budget_default_extension_stop_and_boundaries(self):
        self.assertEqual(budget_decision(1000)['limit_hours'],12)
        extended=budget_decision(3000)
        self.assertEqual(extended['limit_hours'],24);self.assertTrue(extended['limit_extended_before_continuing'])
        self.assertTrue(budget_decision(6000)['stop_for_new_version'])
        self.assertEqual(budget_decision(12*3600/17)['limit_hours'],12)
        self.assertEqual(budget_decision(24*3600/17)['limit_hours'],24)

    def test_margin_gate_name_changes_no_values(self):
        value={historical.STRICT_ARM:'random_forest','stricter_margin':.0005,'stricter_blocked':True,
            'stricter_triggering_sex':['female'],'joint_intervals_female_male':[[0,.001],[-.001,.0001]]}
        result=rename(value)
        self.assertEqual(result[MARGIN_ARM],'random_forest');self.assertEqual(result['harm_margin'],.0005)
        self.assertEqual(result['margin_blocked'],True)
        self.assertEqual(result['joint_intervals_female_male'],value['joint_intervals_female_male'])
        self.assertNotIn('stricter',json.dumps(result))
        self.assertEqual(MARGIN_LABEL,'margin gate (diagnostic)')

    def test_job_A_primary_requires_no_secondary_values(self):
        table=reporting_fixture().rename(columns={'choice_'+historical.STRICT_ARM:'choice_'+MARGIN_ARM})
        keep=[c for c in table if not c.startswith('judge_') or c in ('judge_J1_primary','judge_J1_clean_brier',
            'judge_J1_female_primary','judge_J1_male_primary')]
        result=primary_summary(table[keep])
        self.assertEqual(result['headline_flag'],'awaiting_job_B_secondary_endpoints')
        self.assertEqual(len(result['primary_contrasts']),9)
        self.assertFalse(result['primary_contrasts'].contrast.str.contains('stricter').any())

    def test_job_B_public_outputs_use_margin_names(self):
        table=reporting_fixture().rename(columns={'choice_'+historical.STRICT_ARM:'choice_'+MARGIN_ARM})
        result=summarize(table)
        self.assertFalse(result['contrasts'].contrast.str.contains('stricter').any())
        self.assertIn('margin_primary_cost',result['fairness_summary'])

    def test_all_alternative_predictions_capture_preserves_curve(self):
        from research.model_selection_v2.safe2026 import greedy_preservation
        from .selection import capture_greedy
        clean=np.array([.1,.3,.7,.9])
        def predict(removed):return clean if len(removed)==1 else np.full(4,.5)
        old=greedy_preservation(['a','b'],clean,predict)
        new=capture_greedy(['a','b'],clean,predict)
        np.testing.assert_array_equal(old['values'],new['values'])
        self.assertEqual(new['all_alternative_predictions'].shape,(3,4))
        self.assertEqual(old['alternatives'],new['alternatives'])


class ModelIdentityFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from research.model_selection_v2.confirmation_v1.schema import Schema,build_candidate
        from research.model_selection_v2 import pilot
        rng=np.random.default_rng(12345)
        X=pd.DataFrame(dict(x=rng.normal(size=180),z=rng.normal(size=180),
            stage=rng.integers(0,3,180),category=rng.integers(0,3,180)))
        y=(X.x+.3*X.z+.2*X.stage>0).astype(int)
        schema=Schema(tuple(X.columns),('x','z'),{'stage':(0,1,2)},('category',))
        cls.role_X={name:X.iloc[part].reset_index(drop=True) for name,part in
            dict(fitting=slice(0,90),selection=slice(90,135),judge=slice(135,None)).items()}
        cls.models={}
        with threadpool_limits(limits=2):
            for candidate in pilot.CANDIDATES:
                cls.models[candidate]=build_candidate(candidate,schema).fit(cls.role_X['fitting'],y.iloc[:90])

    def temporary_folder(self):
        root=Path(tempfile.gettempdir()).resolve()
        temporary=tempfile.TemporaryDirectory(prefix='model_identity_unit_',dir=root)
        self.assertEqual(Path(temporary.name).resolve().parent,root)
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name)

    def test_each_candidate_save_reload_file_identity_and_exact_predictions(self):
        folder=self.temporary_folder();(folder/'models').mkdir()
        hashes={}
        for candidate,model in self.models.items():
            with self.subTest(candidate=candidate):
                path=folder/'models'/(candidate+'.joblib')
                reloaded,hashes[candidate]=save_reloaded_model(model,path)
                self.assertIsNot(reloaded,model)
                self.assertEqual(hashes[candidate],sha(path))
                result=compare_models(model,reloaded,self.role_X,self.role_X)
                self.assertEqual(result['probability_values_checked'],360)
                compare_models(reloaded,joblib.load(path),self.role_X,self.role_X)
                # This is also the padding-free Job B before/after guard.
                before=model_snapshot(reloaded,self.role_X)
                for X in self.role_X.values():reloaded.predict_proba(X)
                exact(before,model_snapshot(reloaded,self.role_X))
        write_json(folder/'model_file_sha256.json',hashes)
        self.assertEqual(verify_model_files(folder,self.models),hashes)

    def test_file_identity_rejects_changed_saved_bytes(self):
        folder=self.temporary_folder();(folder/'models').mkdir()
        path=folder/'models'/'logistic.joblib'
        _,checksum=save_reloaded_model(self.models['logistic'],path)
        write_json(folder/'model_file_sha256.json',{'logistic':checksum})
        with path.open('ab') as stream:stream.write(b'changed file bytes')
        with self.assertRaises(AssertionError):verify_model_files(folder,('logistic',))

    def test_tree_state_change_rejected_even_when_predictions_match(self):
        folder=self.temporary_folder();path=folder/'tree.joblib'
        left,_=save_reloaded_model(self.models['decision_tree'],path)
        right=joblib.load(path)
        before=model_snapshot(right,self.role_X)
        right.named_steps['classifier'].tree_.impurity[0]+=1
        exact(before['predict_proba'],model_snapshot(right,self.role_X)['predict_proba'])
        with self.assertRaises(AssertionError):compare_models(left,right,self.role_X,self.role_X)
        with self.assertRaises(AssertionError):exact(before,model_snapshot(right,self.role_X))

    def test_probability_mutation_rejected(self):
        folder=self.temporary_folder();path=folder/'logistic.joblib'
        model,_=save_reloaded_model(self.models['logistic'],path)
        before=model_snapshot(model,self.role_X)
        model.named_steps['classifier'].intercept_[0]+=1
        with self.assertRaises(AssertionError):exact(before,model_snapshot(model,self.role_X))

    def test_repeat_comparison_ignores_padding_but_checks_all_models_and_rows(self):
        folder=self.temporary_folder();left,right=folder/'left',folder/'right'
        for output in (left,right):
            (output/'models').mkdir(parents=True)
            joblib.dump(dict(role_X=self.role_X,repeat=123),output/'inputs.joblib')
        hashes_left={};hashes_right={}
        for candidate,model in self.models.items():
            path=left/'models'/(candidate+'.joblib')
            reloaded,hashes_left[candidate]=save_reloaded_model(model,path)
            if candidate=='decision_tree':
                nodes=reloaded.named_steps['classifier'].tree_.__getstate__()['nodes']
                occupied=np.zeros(nodes.dtype.itemsize,dtype=bool)
                for field in nodes.dtype.names:
                    dtype,offset=nodes.dtype.fields[field][:2]
                    occupied[offset:offset+dtype.itemsize]=True
                padding=np.flatnonzero(~occupied)
                self.assertGreater(len(padding),0,'Fixture must exercise node padding')
                raw=nodes.view(np.uint8).reshape(len(nodes),nodes.dtype.itemsize)
                raw[:,padding]^=np.uint8(255)
            _,hashes_right[candidate]=save_reloaded_model(reloaded,right/'models'/(candidate+'.joblib'))
        self.assertNotEqual(hashes_left['decision_tree'],hashes_right['decision_tree'])
        for output,hashes in ((left,hashes_left),(right,hashes_right)):
            write_json(output/'model_file_sha256.json',hashes);manifest(output)
        result=compare_repeat_folders(left,right)
        self.assertTrue(result['exact']);self.assertTrue(result['input_bundles_compared_exactly'])
        self.assertEqual(len(result['fitted_model_comparisons']),4)
        for compared in result['fitted_model_comparisons'].values():
            self.assertEqual(compared['rows_by_role'],dict(fitting=90,selection=45,judge=45))
            self.assertEqual(compared['probability_values_checked'],360)
        # Altering a saved input is still a real mismatch, even with a fresh manifest.
        payload=joblib.load(right/'inputs.joblib');payload['role_X']['judge'].iloc[0,0]+=1
        joblib.dump(payload,right/'inputs.joblib')
        (right/'artifact_manifest.json').unlink();manifest(right)
        with self.assertRaises(AssertionError):compare_repeat_folders(left,right)


if __name__=='__main__':
    # runpy's guarded launcher leaves its own module in sys.modules['__main__'].
    # Name the actual test classes so this entry point cannot report zero tests.
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case)
        for case in (JobFixtures,ModelIdentityFixtures))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
