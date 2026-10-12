"""Read-only checkpoint checks against the actual saved selection/evaluation data."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from article_safe.artifacts import verify_preparation, verify_stage
from article_safe.data import check_evaluation_artifacts
from article_safe.inference import paired_delong, group_row_weights
from article_safe.selection import choose_forward_prefix
from article_safe.protocol import ROOT
from src.baseline import DATA_PATH, load_split
from src.performance_evaluation import performance_metrics

run=ROOT/'article_safe/runs/article_v1_2026-10-10'
protocol=verify_preparation(run)
selection_manifest=verify_stage(run/'selection',protocol)
evaluation_manifest=verify_stage(run/'evaluation',protocol)
choices=json.loads((run/'selection/selected_subsets.json').read_text())
path=pd.read_csv(run/'selection/gbm_prefix_path.csv',float_precision='round_trip')
validation=pd.read_csv(run/'selection/gbm_validation_predictions.csv',float_precision='round_trip')
steps=[]
for k in range(2,23):
    result=paired_delong(validation.default,validation[f'k_{k}'],validation[f'k_{k-1}'])
    saved=path[path.k==k].iloc[0]
    np.testing.assert_allclose(result.difference,saved.gain,atol=1e-14)
    np.testing.assert_allclose(result.p_value,saved.p_value,atol=1e-14)
    steps.append({'k':k,'gain':result.difference,'p_value':result.p_value,'status':result.status})
assert choose_forward_prefix(steps)['selected_k']==choices['gbm']['selected_k']==3
assert choices['lr_mle']['status']=='not_estimable'
parts=load_split(pd.read_csv(DATA_PATH))
y,meta=parts[3].to_numpy(),parts[5]
predictions=pd.read_csv(run/'evaluation/holdout_predictions.csv',float_precision='round_trip')
metrics=pd.read_csv(run/'evaluation/holdout_metrics.csv',float_precision='round_trip').set_index('model')
probabilities={}
for model,frame in predictions.groupby('model',sort=False):
    np.testing.assert_array_equal(frame.ID,meta.ID)
    np.testing.assert_array_equal(frame.SEX,meta.SEX)
    np.testing.assert_array_equal(frame.actual,y)
    probability=frame.probability_default.to_numpy()
    np.testing.assert_array_equal(frame.predicted,probability>.5)
    actual=performance_metrics(y,probability)
    for quantity,value in actual.items():
        np.testing.assert_allclose(value,metrics.loc[model,quantity],atol=1e-12)
    probabilities[model]=probability
rep=np.load(run/'evaluation/bootstrap_replicates.npz')
explanation_models=rep['explanation_models'].tolist()
features=rep['features'].tolist()
phi={model:np.load(run/f'evaluation/{model}_shap.npz')['original'] for model in explanation_models}
sex=meta.SEX.to_numpy()
rng=np.random.default_rng(42)
for draw in range(10):
    weights=group_row_weights(meta.duplicate_group_id.to_numpy(),rng)
    expected=roc_auc_score(y,probabilities['gbm_selected'],sample_weight=weights)-roc_auc_score(y,probabilities['gbm_full'],sample_weight=weights)
    np.testing.assert_allclose(expected,rep['auc_differences'][draw,3],atol=1e-12)
    for i,model in enumerate(explanation_models):
        for j in range(len(features)):
            for offset,values in ((0,phi[model]),(len(explanation_models)*len(features),np.abs(phi[model]))):
                expected=np.average(values[sex==2,j],weights=weights[sex==2])-np.average(values[sex==1,j],weights=weights[sex==1])
                np.testing.assert_allclose(expected,rep['explanation_gaps'][draw,offset+i*len(features)+j],atol=1e-12)
intervals=pd.read_csv(run/'evaluation/auc_difference_intervals.csv')
np.testing.assert_allclose(intervals.iloc[3][['ci_lower','ci_upper']].to_numpy(dtype=float),np.quantile(rep['auc_differences'][:,3],[.025,.975]),atol=1e-12)
events=json.loads((run/'evaluation/execution_events.json').read_text())
start=next(i for i,item in enumerate(events) if item['event']=='holdout_evaluation_started')
assert all(i<start for i,item in enumerate(events) if item['event']=='final_refit_finished')
assert evaluation_manifest['selected_subsets_sha256']==hashlib.sha256((run/'selection/selected_subsets.json').read_bytes()).hexdigest()
old=check_evaluation_artifacts()
original=json.loads((ROOT/'article_safe/milestones/checkpoint_01/preexisting_project_inventory.json').read_text())
changed=[relative for relative,digest in original.items() if not (ROOT/relative).is_file() or hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=digest]
assert set(changed)<={'PROJECT_CONTEXT.md'},changed
receipt={'client_date':'2026-10-10','milestone':3,'status':'ready_for_user_review_with_explicit_LR_obstacle',
         'saved_selection_replayed':True,'full_gbm_reproduces_baseline':json.loads((run/'evaluation/full_gbm_reproduction.json').read_text())['passed'],
         'heldout_metrics_and_decisions_recomputed':True,'first_10_paired_bootstrap_draws_independently_checked':True,
         'final_fits_precede_holdout_evaluation':True,'selection_hash_unchanged':True,
         'prior_evaluation_hash_checks':old['checks'],'unexpected_preexisting_changes':[],
         'LR_status':choices['lr_mle']['status'],'GBM_predictors':choices['gbm']['selected_features'],
         'isolated_tests_passed':49,'existing_tests_passed':30,'complete_run_rerun_verification':'deferred to checkpoint 4'}
(ROOT/'article_safe/milestones/checkpoint_03/verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))