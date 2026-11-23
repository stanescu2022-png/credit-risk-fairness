"""Hand-calculated fixtures only; no version-2 Taiwan/Brazil secondary results."""
import io
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.stats import t
from research.model_selection_v2.confirmation_v1.schema import Schema
from research.model_selection_v2.confirmation_v1 import judge as v1
from research.model_selection_v2.pilot import CANDIDATES
from . import judge,arms,audit,reporting


def fixture():
    schema=Schema(('AGE','ord','cat'),('AGE',),{'ord':(1,5,10,15,20)},('cat',),{'ord':(-1,)})
    X=pd.DataFrame(dict(AGE=np.arange(20,60,dtype=float),ord=[1,5,10,15,20]*8,cat=[0,1,2,0]*10))
    meta=pd.DataFrame(dict(ID=np.arange(40),SEX=[0,2,1,2,1]*8,default=[0,1,1,0]*10,duplicate_group_id=np.arange(40)))
    return X,meta,schema,{'age':('AGE',),'ordinal':('ord',),'category':('cat',)}


class ToyModel:
    classes_=np.array([0,1])
    named_steps={'classifier':SimpleNamespace(classes_=np.array([0,1]))}
    def predict_proba(self,X):
        p=np.clip(.05+X.AGE.to_numpy()/100+X['ord'].to_numpy()/100+X.cat.to_numpy()/30,.01,.99)
        return np.column_stack((1-p,p))


def reporting_fixture():
    rows=[]
    for repeat in range(50):
        choices=dict(safe='logistic',selector2='random_forest' if repeat<30 else 'logistic',
            selector2_before_check='random_forest' if repeat<35 else 'decision_tree',plain_loss='random_forest',
            order_mild='decision_tree' if repeat<30 else 'logistic',accuracy_full='gradient_boosting' if repeat<20 else 'logistic',
            ae_only='logistic',clean_accuracy='gradient_boosting' if repeat<25 else 'logistic',
            true_retention='random_forest' if repeat<40 else 'logistic',selector2_stricter_gate='logistic')
        for i,c in enumerate(CANDIDATES):
            row=dict(repeat=repeat,candidate=c,n_judge=20,n_selection=16,n_fitting=64,
                **{'choice_'+a:v for a,v in choices.items()})
            for endpoint in reporting.ENDPOINTS:
                row[endpoint]=.14-.01*i+repeat*.00001
            row.update(judge_J1_clean_brier=.13-.005*i,judge_J1_female_primary=.13-.009*i,
                judge_J1_male_primary=.15-.011*i,judge_J1_stressed_auc=.7+.01*i,judge_J1_stressed_k=.8+.01*i)
            rows.append(row)
    return pd.DataFrame(rows)


class V2Tests(unittest.TestCase):
    def test_nested_records_and_shared_category_streams(self):
        X,meta,schema,families=fixture()
        values={level:judge.paired_transforms(X,X,schema,families,dataset_id='fixture',family='age',repetition=3,level=level)
                for level in judge.LEVELS}
        a,b,c=[set(values[level][1]['selected_positions']) for level in judge.LEVELS]
        self.assertEqual([len(a),len(b),len(c)],[4,10,20])
        self.assertTrue(a<b<c)
        for level in ('J2','J3'):
            self.assertEqual(values['J1'][2],values[level][2])
            selected=values['J1'][1]['selected_positions']
            np.testing.assert_array_equal(values['J1'][0]['01'].iloc[selected].cat,values[level][0]['01'].iloc[selected].cat)

    def test_J1_transform_exact(self):
        X,meta,schema,families=fixture()
        old=v1.paired_transforms(X,X,schema,families,dataset_id='fixture',family='ordinal',repetition=1)
        new=judge.paired_transforms(X,X,schema,families,dataset_id='fixture',family='ordinal',repetition=1)
        for mode in old[0]:pd.testing.assert_frame_equal(old[0][mode],new[0][mode],check_exact=True)
        self.assertEqual(old[1:],new[1:])

    def test_J3_ordered_exact_destination_and_special(self):
        rule=dict(levels=[1,5,10,15,20],observed=[-1,1,10,20],special=[-1])
        values,absent=judge.ordered_shift(np.array([1.,5.,10.,20.,-1.]),np.array([1,1,-1,1,1]),rule,2)
        np.testing.assert_array_equal(values,[10,5,1,20,-1])
        self.assertEqual(absent,1)

    def test_numeric_ladder_magnitudes(self):
        rule=dict(iqr=20.,precision=0,minimum=0.,maximum=100.)
        for level,expected in [('J1',52),('J2',55),('J3',60)]:
            got,_=v1.numeric_shift(np.array([50.]),np.array([1]),rule,judge.LEVELS[level][1])
            self.assertEqual(got[0],expected)

    def test_shared_numeric_signs_at_all_levels(self):
        from src.robustness_protocol import child_seed
        X,meta,schema,families=fixture()
        signs=np.random.default_rng(child_seed('confirm_v1:fixture:ordinal',2,'sign:AGE')).choice([-1,1],len(X))
        for level,shift in [('J1',2),('J2',5),('J3',10)]:
            frames,reg,streams=judge.paired_transforms(X,X,schema,families,dataset_id='fixture',family='ordinal',repetition=2,level=level)
            chosen=reg['selected_positions']
            expected=np.clip(X.AGE.to_numpy()[chosen]+signs[chosen]*shift,20,59)
            np.testing.assert_array_equal(frames['01'].AGE.to_numpy()[chosen],expected)

    def test_disturbed_Brier_auc_K_hand_calculations(self):
        meta=pd.DataFrame(dict(ID=[1,2,3,4],SEX=[2,1,2,1],default=[0,0,1,1],duplicate_group_id=[1,2,3,4]))
        clean=np.array([.1,.2,.8,.9]);reverse=clean[::-1]
        predictions={'toy':{'f__mask_only':[clean,reverse]}}
        registry=[dict(family='f',repetition=1,audit=dict(selected_positions=[0])),
            dict(family='f',repetition=2,audit=dict(selected_positions=[2]))]
        result=judge.secondary_endpoints(meta,{'toy':clean},predictions,registry,rank_measures=True)[0]
        self.assertAlmostEqual(result['disturbed_brier'],(.01+.64)/2)
        self.assertEqual(result['stressed_auc'],.5)
        self.assertAlmostEqual(result['stressed_k'],.5)
        all_score=judge.legacy_scores(meta,{'toy':clean},predictions)[0]
        self.assertAlmostEqual(all_score['primary'],(.025+.725)/2)
        self.assertNotAlmostEqual(all_score['primary'],.25) # loss of mean probabilities

    def test_secondary_means_over_scenarios(self):
        meta=pd.DataFrame(dict(ID=[1,2,3,4],SEX=[2,1,2,1],default=[0,0,1,1],duplicate_group_id=[1,2,3,4]))
        clean=np.array([.1,.2,.8,.9]);reverse=clean[::-1]
        predictions={'toy':{'a__mask_only':[clean,clean],'b__joint':[clean,reverse]}}
        registry=[dict(family=f,repetition=r,audit=dict(selected_positions=[0,1,2,3])) for f in ('a','b') for r in (1,2)]
        row=judge.secondary_endpoints(meta,{'toy':clean},predictions,registry,rank_measures=True)[0]
        self.assertEqual(row['stressed_auc'],.75)
        self.assertAlmostEqual(row['stressed_k'],.75)

    def test_approval_cutoff_ties_and_ceiling(self):
        np.testing.assert_array_equal(audit.approvals([.2]*4,[40,10,30,20]),[False,True,True,True])
        self.assertEqual(audit.approvals([.1,.2,.3,.4,.5],[1,2,3,4,5]).sum(),4)

    def test_group_decision_rates_and_age_empty_support(self):
        meta=pd.DataFrame(dict(ID=[40,10,30,20],SEX=[2,2,1,1],default=[1,0,1,0]))
        values=audit.one_prediction_audit(meta,[20,35,60,60],[.2]*4)
        self.assertEqual(values['female']['approval_rate'],.5)
        self.assertEqual(values['male']['approval_rate'],1)
        self.assertEqual(values['female']['defaulters_approved'],0)
        self.assertEqual(values['male']['defaulters_approved'],1)
        self.assertEqual(values['female']['nondefaulters_refused'],0)
        self.assertAlmostEqual(values['female']['brier'],.34)
        self.assertAlmostEqual(values['female']['predicted_minus_observed'],-.3)
        self.assertIsNone(values['female__age50plus']['auc'])
        self.assertEqual(values['female__age50plus']['records'],0)

    def test_unknown_sex_in_global_approval_cutoff(self):
        meta=pd.DataFrame(dict(ID=[0,1,2,3],SEX=[0,2,1,1],default=[1,1,0,1]))
        got=audit.one_prediction_audit(meta,[20,20,40,60],[.1,.9,.2,.3])
        self.assertEqual(got['all']['approval_rate'],.75)
        self.assertEqual(got['female']['approval_rate'],0)
        self.assertEqual(got['male']['approval_rate'],1)
        self.assertEqual(got['age_under30']['records'],2)

    def test_audit_gap_changes(self):
        meta=pd.DataFrame(dict(ID=[40,10,30,20],SEX=[2,2,1,1],default=[1,0,1,0]))
        a,b=np.array([.2]*4),np.array([.9,.1,.8,.2])
        records=audit.candidate_audit(meta,[20,35,60,60],{'a':a,'b':b},{'a':{'x':[a]},'b':{'x':[b]}})
        selected=audit.audit_for_choices(records,{'safe':'a','selector2':'b'},0)
        gap=next(r for r in selected if r['arm']=='selector2_minus_safe_gap_change' and r['condition']=='clean'
                 and r['group']=='female_minus_male' and r['metric']=='brier')
        self.assertAlmostEqual(gap['value'],-.03)

    def test_all_twelve_pairs_precision_and_margin(self):
        point=np.zeros((4,2,3));point[:,:,1]=[[0,0],[.002,0],[-.001,-.001],[.0001,.0001]]
        draws=np.tile(point,(2000,1,1,1))
        variation=np.linspace(-1e-5,1e-5,2000)
        for i in range(4):draws[:,i,:,1]+=variation[:,None]*(i+1)
        checks=arms.pair_checks(point,draws,ranked=list(CANDIDATES),repeat=0)
        self.assertEqual(len(checks),12)
        lookup={(r['challenger'],r['incumbent']):r for r in checks}
        harm=lookup['decision_tree','logistic']
        self.assertTrue(harm['blocked']);self.assertEqual(harm['triggering_sex'],['female'])
        self.assertTrue(harm['joint_half_width_female_male'][0]>0)
        # A positive margin means this diagnostic is not a logical subset of
        # the frozen zero-harm rule: small resolved harm may pass its margin.
        small=lookup['gradient_boosting','logistic']
        self.assertTrue(small['blocked']);self.assertFalse(small['stricter_blocked'])
        records=[dict(r,repeat=j) for j in range(50) for r in checks]
        summary=arms.summarize_pair_checks(records)
        self.assertEqual(len(summary),12)
        self.assertTrue(summary.female_defined_half_widths.eq(50).all())

    def test_strict_undefined_upper_blocks(self):
        point=np.zeros((4,2,3));draws=np.zeros((2000,4,2,3))
        checks=arms.pair_checks(point,draws,ranked=list(CANDIDATES),repeat=0)
        self.assertTrue(all(not r['blocked'] and r['stricter_blocked'] for r in checks))

    def test_three_diagnostic_score_formulas(self):
        A=[.3,.6,.8,.2];E=[.9,.5,.4,.8];s0=[-.1,.1,.05,.2];loss=[.1,.01,.2,.05]
        evidence={c:dict(S_mild=np.r_[s0[i],np.full(5,s0[i]-loss[i])]) for i,c in enumerate(CANDIDATES)}
        rows=pd.DataFrame([dict(candidate=c,A_bar=A[i],E_bar=E[i],S_bar=evidence[c]['S_mild'].mean()) for i,c in enumerate(CANDIDATES)]).set_index('candidate')
        got=arms.diagnostic_scores(rows,evidence)
        for i,c in enumerate(CANDIDATES):
            self.assertAlmostEqual(got['ae_only'][c],(A[i]+E[i])/2)
            self.assertAlmostEqual(got['clean_accuracy'][c],(A[i]+E[i]+s0[i])/3)
            self.assertAlmostEqual(got['true_retention'][c],(A[i]+E[i]-loss[i])/3)

    def test_range_measure_contrasts_and_matches(self):
        table=reporting_fixture();result=reporting.summarize(table)
        paired=result['range_measure_contrasts']
        self.assertEqual(paired.k.tolist(),[30,50,50,20,40])
        np.testing.assert_allclose(paired.estimate,[-.006,-.005,-.011,-.012,-.001],rtol=0,atol=1e-15)
        self.assertEqual(len(result['pair_matches']),36)
        self.assertTrue(np.diag(result['pair_match_matrix']).tolist()==[50]*9)
        values=np.r_[np.full(30,-.01),np.zeros(20)]
        half=t.ppf(.975,49)*np.sqrt((1/50+.25)*values.var(ddof=1))
        self.assertAlmostEqual(paired.iloc[0].lower,values.mean()-half)
        text=io.StringIO();reporting.print_flags(result,text)
        self.assertIn('"k": 30',text.getvalue());self.assertIn('"arithmetic_minimum_k": 27',text.getvalue())

    def test_27_of_50_arithmetic_and_zero_convention(self):
        self.assertEqual(reporting.minimum_equal_win_disagreements(.25),27)
        self.assertGreaterEqual(reporting.corrected([-1]*26+[0]*24,.25)['upper'],0)
        self.assertLess(reporting.corrected([-1]*27+[0]*23,.25)['upper'],0)
        self.assertEqual(reporting.corrected([0]*50,.25)['lower'],0)
        self.assertEqual(reporting.corrected([0]*50,.25)['upper'],0)
        missing=reporting.corrected([0]*49+[None],.25)
        self.assertIsNone(missing['estimate']);self.assertEqual(missing['repeats'],50)

    def test_headline_precedence(self):
        ci=lambda lo,hi:dict(lower=lo,upper=hi)
        good=ci(.01,.02);zero=ci(-.01,.01);negative=ci(-.02,-.01)
        def flags(primary=negative,sex=None,auc=good,k=good,log=negative):
            return reporting.headline_flags(primary,{'female':zero,'male':zero} if sex is None else sex,auc,k,log,k=50,rho=.25)
        # Positional k metric and keyword k disagreement are distinct parameters
        # of the wrapper; keyword is passed to headline_flags itself.
        self.assertEqual(flags(sex={'female':good})['headline_flag'],'group_harm')
        self.assertEqual(flags(primary=good)['headline_flag'],'safe_lower_brier')
        self.assertEqual(flags(primary=zero)['headline_flag'],'inconclusive')
        self.assertEqual(flags(auc=negative)['headline_flag'],'lower_brier_worse_auc')
        self.assertEqual(flags(k=negative)['headline_flag'],'lower_brier_worse_k')
        self.assertEqual(flags(log=good)['headline_flag'],'brier_log_loss_tradeoff')
        self.assertEqual(flags()['headline_flag'],'supports_one_dataset')
        self.assertFalse(flags(log=good)['judged_better'])
        self.assertTrue(flags()['judged_better'])

    def test_no_unauthorised_Taiwan_secondary(self):
        X,meta,schema,families=fixture()
        with self.assertRaises(ValueError):
            judge.evaluate(X,X,meta,{'toy':ToyModel()},schema,families,dataset_id='taiwan',purpose='fixture')

    def test_undefined_secondary_not_hidden_by_same_choice(self):
        table=reporting_fixture()
        table.loc[table.candidate.eq('logistic'),'judge_J1_stressed_k']=np.nan
        got=reporting.contrast(table,'ae_only','safe','judge_J1_stressed_k',.25)
        self.assertIsNone(got['estimate'])
        self.assertEqual(got['defined_repeats'],0)

    def test_audit_corrected_summary_keeps_undefined_cells(self):
        records=[]
        for repeat in range(50):
            records.append(dict(repeat=repeat,arm='safe',condition='clean',group='female',metric='brier',value=repeat/1000))
            records.append(dict(repeat=repeat,arm='safe',condition='clean',group='female__age50plus',metric='auc',value=None))
        result=reporting.summarize_audits(records,.25).set_index('metric')
        self.assertAlmostEqual(result.loc['brier','estimate'],.0245)
        self.assertEqual(result.loc['auc','defined_repeats'],0)
        self.assertEqual(result.loc['auc','repeats'],50)

    def test_relative_Brier_sizes_do_not_rescale_rank_metrics(self):
        table=reporting_fixture()
        brier=reporting.contrast(table,'order_mild','safe','judge_J1_primary',.25)
        auc=reporting.contrast(table,'order_mild','safe','judge_J1_stressed_auc',.25)
        self.assertEqual(brier['units'],'Brier')
        self.assertIsNotNone(brier['difference_as_share_of_safe_clean_brier'])
        self.assertEqual(auc['units'],'AUC')
        self.assertIsNone(auc['difference_as_share_of_safe_clean_brier'])

    def test_socket_guard_blocks_connection_events(self):
        from .offline import deny_socket
        with self.assertRaises(PermissionError):deny_socket('socket.connect',())


if __name__=='__main__':unittest.main()
