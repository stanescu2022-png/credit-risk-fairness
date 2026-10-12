"""Focused protocol, transformation, authentic SAFE, and artifact checks."""
import inspect
import json
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from safeaipackage import check_robustness

from src.baseline import ROOT,DATA_PATH,load_split,sha256
from src.performance_evaluation import bootstrap_group_weights,performance_metrics
from src.robustness_evaluation import (TABLE_DIR,MODELS,protected_hashes,baseline_reproduction,
    aggregate,movement,frame_hash)
from src.robustness_protocol import (semantic_rules,semantic_transform,build_protocol,child_seed,
    PRIMARY_SAFE,PRIMARY_SEMANTIC,STATUS)
from src.robustness_safe import (validate_fixtures,transform,authentic_call,RobustnessAdapter)
from src.robustness_uncertainty import (WeightedRanks,validate_weighted_ranks,focus_metrics,FOCUS)
from src.robustness_cache import restore_probabilities,compact_cache,FORMAT as CACHE_FORMAT


class RobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train,cls.test,_,cls.y,cls.meta_train,cls.metadata=load_split(pd.read_csv(DATA_PATH))
        cls.models={k:joblib.load(ROOT/f'models/baseline_{k}.joblib') for k in MODELS}
        cls.rules=semantic_rules(cls.train)

    def read(self,name):
        return pd.read_csv(TABLE_DIR/f'{name}.csv',float_precision='round_trip')

    def test_frozen_baseline_classes_schema_order_and_no_fit(self):
        for model in self.models.values():
            with patch.object(model,'fit',side_effect=AssertionError('No refitting')):
                adapter=RobustnessAdapter(model)
                np.testing.assert_array_equal(adapter.predict_proba(self.test)[:,1],model.predict_proba(self.test)[:,1])
                np.testing.assert_array_equal(model.predict(self.test),model.predict_proba(self.test)[:,1]>.5)
                with self.assertRaises(RuntimeError):
                    adapter.fit(self.test,self.y)
        baseline,_,_,evidence=baseline_reproduction(self.test,self.y,self.metadata,self.models)
        self.assertEqual(len(evidence),8)
        self.assertTrue((evidence.max_probability_error<=1e-15).all())
        np.testing.assert_array_equal(np.array([.49,.5,.51])>.5,[False,False,True])

    def test_actual_safe_fixtures_marginals_and_api_input_equality(self):
        self.assertIn('site-packages',inspect.getfile(check_robustness.compute_rgr_values))
        fixtures=validate_fixtures()
        self.assertEqual((fixtures.status=='undefined').sum(),3)
        X=self.test.iloc[:120].reset_index(drop=True)
        before=X.copy(deep=True)
        features=['PAY_0','BILL_AMT1','EDUCATION']
        transformed,selected=transform(X,features,.05)
        self.assertTrue(all(v.sum()==12 for v in selected.values()))
        for kind,model in self.models.items():
            with patch.object(model,'fit',side_effect=AssertionError('No fitting')):
                with patch.object(check_robustness,'compute_rgr_values',wraps=check_robustness.compute_rgr_values) as api:
                    value,p=authentic_call(model,X,model.predict_proba(X)[:,1],features,.05,True,transformed)
                    self.assertEqual(api.call_count,1)
                    self.assertEqual(api.call_args.kwargs['group'],True)
            np.testing.assert_array_equal(p,model.predict_proba(transformed)[:,1])
        pd.testing.assert_frame_equal(X,before,check_exact=True)
        for f in features:
            np.testing.assert_array_equal(np.sort(X[f]),np.sort(transformed[f]))

    def test_training_only_scales_category_frequencies_and_exact_destinations(self):
        for f,rule in self.rules.items():
            if rule['kind']=='money':
                self.assertEqual(rule['iqr'],self.train[f].quantile(.75)-self.train[f].quantile(.25))
            if rule['kind']=='category':
                counts=self.train[f].value_counts().sort_index()
                np.testing.assert_array_equal(rule['counts'],counts)
                np.testing.assert_allclose(rule['frequencies'],counts/counts.sum(),atol=0,rtol=0)
            if rule['kind']=='status':
                for level,step in [('primary',1),('stronger',2)]:
                    for source,directions in rule['maps'][level].items():
                        s=int(source)
                        for direction,destination in directions.items():
                            expected=s+int(direction)*step
                            self.assertIn(destination,rule['support'])
                            self.assertEqual(destination,expected if s>=1 and expected>=1 and expected in rule['support'] else s)
        self.assertNotIn(1,self.rules['PAY_5']['support'])
        self.assertEqual(self.rules['PAY_5']['maps']['primary']['2']['-1'],2)
        self.assertEqual(self.rules['AGE']['lower'],self.train.AGE.min())
        self.assertEqual(self.rules['LIMIT_BAL']['lower'],self.train.LIMIT_BAL.min())

    def test_semantic_selection_coupling_unselected_cells_bounds_and_special_states(self):
        X=self.test.reset_index(drop=True)
        outputs=[]
        for level,q in [('primary',.1),('stronger',.2)]:
            scenario={'scope':'all_features','features':list(X.columns),'level':level,'q':q,'repetition':3}
            out,info=semantic_transform(X,scenario,self.rules)
            selected=info['AGE']['selected']
            self.assertEqual(selected.sum(),int(np.floor(q*len(X))))
            self.assertTrue(all(np.array_equal(selected,v['selected']) for v in info.values()))
            pd.testing.assert_frame_equal(out.loc[~selected],X.loc[~selected],check_exact=True)
            for f,details in info.items():
                np.testing.assert_array_equal(out[f][details['outside_support']],X[f][details['outside_support']])
                if f in STATUS:
                    np.testing.assert_array_equal(out.loc[X[f]<=0,f],X.loc[X[f]<=0,f])
                    self.assertTrue(out[f].isin(self.rules[f]['support']).all())
                if f.startswith('PAY_AMT'):
                    self.assertTrue((out[f]>=0).all())
                np.testing.assert_array_equal(out[f],np.rint(out[f]))
            outputs.append((out,info))
        primary,stronger=outputs
        small=primary[1]['AGE']['selected']; large=stronger[1]['AGE']['selected']
        self.assertTrue((~small|large).all())
        f='BILL_AMT1'
        d1=primary[0][f].to_numpy()-X[f].to_numpy()
        d2=stronger[0][f].to_numpy()-X[f].to_numpy()
        np.testing.assert_array_equal(np.sign(d1[small]),np.sign(d2[small]))
        repeated,_=semantic_transform(X,{'scope':'all_features','features':list(X.columns),'level':'primary','q':.1,'repetition':3},self.rules)
        pd.testing.assert_frame_equal(repeated,primary[0],check_exact=True)
        # Metadata cannot influence this interface or random streams.
        self.assertEqual(child_seed('all_features',3,'mask'),child_seed('all_features',3,'mask'))
        self.assertNotEqual(child_seed('all_features',3,'sign:AGE'),child_seed('all_features',3,'sign:BILL_AMT1'))

    def test_semantic_edge_cases_zero_iqr_category_no_alternative_and_outside_support(self):
        X=pd.DataFrame({'AGE':[19,21,75,76]*10,'LIMIT_BAL':[5000,10000,20000,30000]*10,
                        'PAY_AMT1':[0,1,2,3]*10,'EDUCATION':[1]*40})
        rules={'AGE':self.rules['AGE'],'LIMIT_BAL':{**self.rules['LIMIT_BAL'],'iqr':0},
               'PAY_AMT1':{**self.rules['PAY_AMT1'],'iqr':100},
               'EDUCATION':{'kind':'category','categories':[1],'counts':[40],'frequencies':[1.]}}
        out,info=semantic_transform(X,{'scope':'edges','features':list(X.columns),'level':'stronger','q':.2,'repetition':1},rules)
        np.testing.assert_array_equal(out.AGE[X.AGE.isin([19,76])],X.AGE[X.AGE.isin([19,76])])
        np.testing.assert_array_equal(out.LIMIT_BAL,X.LIMIT_BAL)
        np.testing.assert_array_equal(out.EDUCATION,X.EDUCATION)
        self.assertTrue(info['LIMIT_BAL']['zero_scale'])
        self.assertFalse(info['EDUCATION']['eligible'].any())
        self.assertTrue((out.PAY_AMT1>=0).all())

    def test_weighted_ranking_whole_groups_and_repetition_aggregation(self):
        checks=validate_weighted_ranks()
        self.assertTrue((checks.max_error<2e-14).all())
        groups=np.array([1,1,2,3,3,4])
        y=np.array([0,1,0,1,0,1]); baseline=np.array([.1,.2,.3,.4,.5,.6])
        predictions=np.array([[.8,.7,.6,.5,.4,.3],[.05,.1,.2,.4,.8,.9]])
        w=bootstrap_group_weights(groups,np.random.default_rng(42))
        for group in np.unique(groups):
            self.assertEqual(len(np.unique(w[groups==group])),1)
        actual=focus_metrics(y,baseline,predictions,w)
        expected=[]
        for p in predictions:
            b=performance_metrics(y,baseline,weights=w)
            v=performance_metrics(y,p,weights=w)
            expected.append([np.average(np.abs(p-baseline),weights=w),np.average((p>.5)!=(baseline>.5),weights=w),
                v['ROC-AUC']-b['ROC-AUC'],v['Average Precision (AP)']-b['Average Precision (AP)'],v['Brier Score']-b['Brier Score']])
        np.testing.assert_allclose(actual,np.mean(expected,axis=0),rtol=0,atol=2e-14)
        averaged=average_precision_score(y,predictions.mean(axis=0),sample_weight=w)
        self.assertNotAlmostEqual(actual[3]+average_precision_score(y,baseline,sample_weight=w),averaged)
        table=pd.DataFrame({'scenario':['semantic__example__primary']*2,'model':['logistic']*2,'SEX':[0]*2,'repetition':[1,2],'value':[-.1,.2]})
        summary=aggregate(table,['scenario','model','SEX'],['value']).iloc[0]
        self.assertAlmostEqual(summary['mean'],.05)
        self.assertAlmostEqual(summary.mc_sd,np.std([-.1,.2],ddof=1))
        values,reason=movement(np.array([.5,.5]),np.array([.5,.6]))
        self.assertEqual(values['flip_rate'],.5)
        self.assertTrue(np.isnan(values['Spearman']))
        self.assertEqual(reason,'constant score vector')

    def test_saved_protocol_full_coverage_seeds_and_all_artifact_hashes(self):
        manifest=json.loads((TABLE_DIR/'run_manifest.json').read_text())
        protocol=json.loads((TABLE_DIR/'protocol.json').read_text())
        self.assertEqual(manifest['protected_artifact_hashes'],protected_hashes())
        for category in ('source_hashes','output_hashes'):
            for name,digest in manifest[category].items():
                self.assertEqual(sha256(ROOT/name),digest,name)
        self.assertEqual(sha256(TABLE_DIR/'protocol.json'),manifest['protocol_sha256'])
        self.assertEqual(len(protocol['scenarios']),167)
        self.assertEqual(protocol['predictor_order'],list(self.test.columns))
        self.assertEqual(protocol['semantic_rules'],self.rules)
        self.assertEqual(protocol['bootstrap']['repetitions'],2000)
        self.assertEqual(protocol['classification']['rule'],'probability_default > 0.50; equality is class 0')
        self.assertFalse(manifest['models_fitted'])
        registry=self.read('realization_registry')
        self.assertEqual(len(registry),1501)
        self.assertTrue(registry.models_share_exact_input.all())
        self.assertTrue(registry.canonical_ID_order_restored.all())
        self.assertEqual(registry.groupby('scenario').size().value_counts().to_dict(),{1:121,30:46})
        self.assertEqual(len(self.read('individual_rgr')),176)
        self.assertEqual(len(self.read('grouped_rgr')),66)
        for seed in protocol['child_seeds']:
            self.assertEqual(seed['mask'],child_seed(seed['scope'],seed['repetition'],'mask'))

    def test_saved_cache_alignment_exposure_permutation_and_reconstruction(self):
        registry=self.read('realization_registry')
        exposure=self.read('exposure_counts')
        with np.load(TABLE_DIR/'prediction_cache.npz') as cache:
            self.assertEqual(str(cache['cache_format']),CACHE_FORMAT)
            np.testing.assert_array_equal(cache['ID'],self.metadata.ID)
            np.testing.assert_array_equal(cache['SEX'],self.metadata.SEX)
            np.testing.assert_array_equal(cache['actual'],self.y)
            np.testing.assert_array_equal(cache['duplicate_group_id'],self.metadata.duplicate_group_id)
            for j,kind in enumerate(MODELS):
                np.testing.assert_array_equal(cache['baseline'][j],self.models[kind].predict_proba(self.test)[:,1])
            for name in [PRIMARY_SAFE,PRIMARY_SEMANTIC]:
                row=registry[(registry.scenario==name)&(registry.repetition==1)].iloc[0]
                X=self.test.reset_index(drop=True)
                if name==PRIMARY_SAFE:
                    out,_=transform(X,list(X.columns),.05)
                else:
                    out,_=semantic_transform(X,{'scope':'all_features','features':list(X.columns),'level':'primary','q':.1,'repetition':1},self.rules)
                self.assertEqual(row.input_sha256,frame_hash(out))
                changed=(X.to_numpy()!=out.to_numpy()).any(axis=1)
                np.testing.assert_array_equal(cache[row.cache_key+'_changed'],changed)
                for j,kind in enumerate(MODELS):
                    np.testing.assert_array_equal(restore_probabilities(cache,row.cache_key)[j],self.models[kind].predict_proba(out)[:,1])
            permuted=self.read('permutation_record_comparison')
            np.testing.assert_array_equal(permuted.ID,self.metadata.ID)
            for row in registry[registry.scenario.isin([PRIMARY_SAFE,PRIMARY_SEMANTIC])].itertuples():
                totals=exposure[(exposure.scenario==row.scenario)&(exposure.repetition==row.repetition)&(exposure.feature=='__union__')]
                self.assertEqual(totals[totals.SEX==0].changed.iloc[0],cache[row.cache_key+'_changed'].sum())
                self.assertEqual(totals[totals.SEX==0].changed.iloc[0],totals[totals.SEX.isin([1,2])].changed.sum())
        per_feature=exposure[exposure.feature!='__union__']
        np.testing.assert_array_equal(per_feature.selected,per_feature.changed+per_feature.no_op)
        self.assertTrue((per_feature.clipped<=per_feature.selected).all())

    def test_saved_nonlinear_metrics_deltas_paired_means_intervals_and_report_trace(self):
        registry=self.read('realization_registry')
        diagnostics=self.read('conventional_diagnostics')
        summary=self.read('diagnostic_summary')
        paired=self.read('paired_comparisons')
        with np.load(TABLE_DIR/'prediction_cache.npz') as cache:
            for rep in (1,2):
                row=registry[(registry.scenario==PRIMARY_SEMANTIC)&(registry.repetition==rep)].iloc[0]
                values=[]
                for j,kind in enumerate(MODELS):
                    subset=diagnostics[(diagnostics.scenario==PRIMARY_SEMANTIC)&(diagnostics.model==kind)&(diagnostics.repetition==rep)&(diagnostics.SEX==0)].iloc[0]
                    original=cache['baseline'][j]; perturbed=restore_probabilities(cache,row.cache_key)[j]
                    actual=performance_metrics(self.y,perturbed)
                    baseline=performance_metrics(self.y,original)
                    for metric in actual:
                        self.assertAlmostEqual(subset[metric],actual[metric],places=12)
                        self.assertAlmostEqual(subset['delta_'+metric],actual[metric]-baseline[metric],places=12)
                    values.append(subset)
                pair=paired[(paired.scenario==PRIMARY_SEMANTIC)&(paired.repetition==rep)&(paired.SEX==0)].iloc[0]
                self.assertAlmostEqual(pair['delta_Average Precision (AP)'],values[1]['delta_Average Precision (AP)']-values[0]['delta_Average Precision (AP)'],places=12)
        for kind in MODELS:
            part=diagnostics[(diagnostics.scenario==PRIMARY_SEMANTIC)&(diagnostics.model==kind)&(diagnostics.SEX==0)]
            self.assertEqual(len(part),30)
            for metric in FOCUS:
                saved=summary[(summary.scenario==PRIMARY_SEMANTIC)&(summary.model==kind)&(summary.SEX==0)&(summary.metric==metric)].iloc[0]
                self.assertAlmostEqual(saved['mean'],part[metric].mean(),places=12)
                self.assertAlmostEqual(saved.mc_sd,part[metric].std(ddof=1),places=12)
        intervals=self.read('sampling_intervals')
        reps=self.read('bootstrap_replicates')
        self.assertEqual(len(reps),36000)
        self.assertTrue(intervals.attempted_replicates.eq(2000).all())
        for row in intervals.itertuples():
            part=reps[(reps.scenario==row.scenario)&(reps.model==row.model)&(reps.SEX==row.SEX)][row.metric].dropna()
            np.testing.assert_allclose([row.ci_lower,row.ci_upper],np.quantile(part,[.025,.975]),atol=1e-14,rtol=0)
            self.assertEqual(row.valid_replicates,len(part))
        report=(ROOT/'outputs/robustness_evaluation_report.md').read_text(encoding='utf-8')
        self.assertIn(PRIMARY_SAFE,report)
        self.assertIn(PRIMARY_SEMANTIC,report)
        for metric in ('mean_absolute_movement','flip_rate'):
            for kind in MODELS:
                value=summary[(summary.scenario==PRIMARY_SAFE)&(summary.model==kind)&(summary.SEX==0)&(summary.metric==metric)].iloc[0]['mean']
                if metric=='mean_absolute_movement':
                    self.assertIn(f'{value:.6f}',report)
        self.assertIn('does not validate the entire extended trustworthiness framework',report)


    def test_saved_bootstrap_matches_paired_sklearn_repetition_metrics(self):
        registry=self.read('realization_registry')
        reps=self.read('bootstrap_replicates')
        w=bootstrap_group_weights(self.metadata.duplicate_group_id,np.random.default_rng(42))
        with np.load(TABLE_DIR/'prediction_cache.npz') as cache:
            for scenario in [PRIMARY_SAFE,PRIMARY_SEMANTIC]:
                realization_rows=registry[registry.scenario==scenario]
                scores=np.array([restore_probabilities(cache,row.cache_key) for row in realization_rows.itertuples()])
                for sex in (0,1,2):
                    weights=w if sex==0 else w*(self.metadata.SEX.to_numpy()==sex)
                    expected={}
                    for j,kind in enumerate(MODELS):
                        baseline=cache['baseline'][j]
                        original=performance_metrics(self.y,baseline,weights=weights)
                        values=[]
                        for p in scores[:,j,:]:
                            metrics=performance_metrics(self.y,p,weights=weights)
                            values.append([np.average(np.abs(p-baseline),weights=weights),
                                np.average((p>.5)!=(baseline>.5),weights=weights),
                                metrics['ROC-AUC']-original['ROC-AUC'],
                                metrics['Average Precision (AP)']-original['Average Precision (AP)'],
                                metrics['Brier Score']-original['Brier Score']])
                        expected[kind]=np.mean(values,axis=0)
                    expected['GBM_minus_LR']=expected[MODELS[1]]-expected[MODELS[0]]
                    for kind,values in expected.items():
                        saved=reps[(reps.scenario==scenario)&(reps.SEX==sex)&(reps.model==kind)&(reps.bootstrap==1)].iloc[0]
                        np.testing.assert_allclose(saved[FOCUS].to_numpy(dtype=float),values,atol=2e-14,rtol=0)
                        self.assertEqual(saved.sample_records,int(weights.sum()))

    def test_sparse_cache_roundtrip_retains_zero_movement_and_float_bits(self):
        baseline=np.array([[.1,.2,.3],[.2,.3,.4]])
        scores=baseline.copy(); scores[0,1]=.9; scores[1,2]=np.nextafter(.4,1.)
        dense={'ID':np.arange(3),'SEX':np.array([1,2,2]),'actual':np.array([0,1,0]),
               'duplicate_group_id':np.arange(3),'baseline':baseline,
               'example':scores,'example_changed':np.array([False,True,False])}
        compact=compact_cache(dense,[{'cache_key':'example'}])
        np.testing.assert_array_equal(restore_probabilities(compact,'example'),scores)
        np.testing.assert_array_equal(compact['example_changed'],[False,True,False])
        self.assertEqual(compact['example'].shape,(2,2))

if __name__=='__main__':
    unittest.main()
