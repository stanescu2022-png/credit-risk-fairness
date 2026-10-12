"""Reproducible robustness evaluation of existing frozen pipelines only."""
from datetime import datetime, timezone
from importlib.metadata import distributions, version
from pathlib import Path
import hashlib
import json
import platform
import re
import subprocess
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from src.baseline import ROOT, DATA_PATH, load_split, sha256
from src.performance_evaluation import (performance_metrics, class_metrics, reliability_bins,
    predictor_groups, evaluate_fixed_holdout)
from src.explainability_evaluation import validate_prior_stages
from src.robustness_protocol import (build_protocol, semantic_transform, RATIONALE,
    PRIMARY_SAFE, PRIMARY_SEMANTIC)
from src.robustness_safe import (source_identity, inspection_text, validate_fixtures,
    transform as safe_transform, authentic_call, RobustnessAdapter)
from src.robustness_uncertainty import paired_bootstrap, validate_weighted_ranks, FOCUS
from src.robustness_cache import compact_cache, FORMAT as CACHE_FORMAT

TABLE_DIR = ROOT / 'outputs/tables/robustness_evaluation'
FIGURE_DIR = ROOT / 'outputs/figures/robustness_evaluation'
REPORT = ROOT / 'outputs/robustness_evaluation_report.md'
MODELS = ('logistic','gradient_boosting')
LABELS = {0:'all',1:'male',2:'female'}
SOURCES = ['src/robustness_evaluation.py','src/robustness_protocol.py','src/robustness_safe.py',
           'src/robustness_uncertainty.py','src/robustness_report.py','src/robustness_cache.py',
           'script_robustness_evaluation.py','tests/test_robustness_evaluation.py']
MOVEMENT = ['mean_signed_movement','mean_absolute_movement','median_absolute_movement',
            'p95_absolute_movement','Spearman','flip_rate']


def protected_hashes():
    """Protect all earlier artifacts/code/locks; exclude only this additive stage."""
    paths=[]
    for directory in ('data','models','outputs','notebooks','src','tests'):
        paths.extend((ROOT/directory).rglob('*'))
    paths.extend(ROOT.glob('requirements*.txt'))
    paths.extend(ROOT.glob('script_*.py'))
    paths.extend([ROOT/'doc ideeas.txt',ROOT/'potential_issues.txt',ROOT/'credit_risk_fairness_setup_steps.txt'])
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in sorted(set(paths))
            if p.is_file() and 'robustness' not in str(p.relative_to(ROOT))
            and '__pycache__' not in p.parts}


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def frame_hash(X):
    data=np.ascontiguousarray(X.to_numpy())
    return hashlib.sha256(json.dumps(list(X.columns)).encode()+str(data.dtype).encode()+data.tobytes()).hexdigest()


def validate_all_prior():
    validate_prior_stages()
    manifest=json.loads((ROOT/'outputs/tables/explainability_evaluation/run_manifest.json').read_text())
    for category in ('protected_artifact_hashes','source_hashes','output_hashes'):
        for name,expected in manifest.get(category,{}).items():
            if sha256(ROOT/name)!=expected:
                raise ValueError(f'Explainability artifact changed: {name}')


def baseline_reproduction(X, y, metadata, models):
    reproduced, scores=evaluate_fixed_holdout(X,y,metadata)
    baseline={}
    evidence=[]
    perf=pd.read_csv(ROOT/'outputs/tables/performance_evaluation/holdout_metrics.csv',float_precision='round_trip')
    for kind,model in models.items():
        np.testing.assert_array_equal(model.classes_,[0,1])
        np.testing.assert_array_equal(model.feature_names_in_,X.columns)
        if {'ID','SEX','default'} & set(X.columns):
            raise AssertionError('Metadata leaked into predictors')
        part=reproduced[reproduced.model==kind]
        baseline[kind]=part.probability_default.to_numpy().copy()
        # ID-keyed equality plus canonical order checks are both required.
        for relative in [f'outputs/tables/baseline_{kind}_predictions.csv',
                         'outputs/tables/performance_evaluation/holdout_predictions.csv',
                         'outputs/tables/fairness_evaluation/audit_predictions.csv',
                         'outputs/tables/explainability_evaluation/audit_predictions.csv']:
            saved=pd.read_csv(ROOT/relative,float_precision='round_trip')
            if 'model' in saved:
                saved=saved[saved.model==kind]
            if not saved.ID.is_unique:
                raise AssertionError('Saved prediction IDs are not unique')
            np.testing.assert_array_equal(saved.ID,metadata.ID)
            indexed=saved.set_index('ID').loc[metadata.ID]
            np.testing.assert_array_equal(indexed.actual,y)
            np.testing.assert_array_equal(indexed.SEX,metadata.SEX)
            np.testing.assert_array_equal(indexed.predicted,baseline[kind]>.5)
            error=float(np.max(np.abs(indexed.probability_default.to_numpy()-baseline[kind])))
            if error>1e-15:
                raise ValueError('Material baseline discrepancy; stop robustness evaluation')
            evidence.append({'model':kind,'artifact':relative,'rows':len(saved),
                             'max_probability_error':error,'probability_tolerance':1e-15,
                             'ID_order_class_SEX_outcome_match':True,'classes':'[0,1]',
                             'class_rule':'p > .50; ties 0'})
        expected=perf[perf.model==kind].iloc[0]
        actual=scores[scores.model==kind].iloc[0]
        for metric in performance_metrics(y,baseline[kind]):
            if metric in perf.columns:
                np.testing.assert_allclose(actual[metric],expected[metric],rtol=0,atol=1e-12)
        adapter=RobustnessAdapter(model)
        np.testing.assert_array_equal(adapter.predict_proba(X.reset_index(drop=True))[:,1],baseline[kind])
    np.testing.assert_array_equal(predictor_groups(X,metadata),metadata.duplicate_group_id)
    return baseline,reproduced,scores,pd.DataFrame(evidence)


def movement(original, perturbed):
    if len(original)==0:
        return {name:np.nan for name in MOVEMENT},'empty changed-record subset'
    delta=perturbed-original
    absolute=np.abs(delta)
    constant=np.ptp(original)==0 or np.ptp(perturbed)==0
    return {'mean_signed_movement':float(delta.mean()),'mean_absolute_movement':float(absolute.mean()),
            'median_absolute_movement':float(np.median(absolute)),
            'p95_absolute_movement':float(np.quantile(absolute,.95)),
            'Spearman':np.nan if constant else float(spearmanr(original,perturbed).statistic),
            'flip_rate':float(np.mean((original>.5)!=(perturbed>.5)))}, ('constant score vector' if constant else '')


def outcome_metrics(y,p):
    if len(np.unique(y))==2:
        values=performance_metrics(y,p)
    else:
        values=class_metrics(y,p>.5)
        values.update({'ROC-AUC':np.nan,'Average Precision (AP)':np.nan,
                       'Brier Score':float(np.mean((p-y)**2)),
                       'Log Loss':float(log_loss(y,p,labels=[0,1]))})
    values['ECE_10_uniform_bins']=float(reliability_bins(y,p).ece_contribution.sum())
    return values


def exposure_rows(scenario, repetition, X, perturbed, info, metadata):
    rows=[]
    changes=X.to_numpy()!=perturbed.to_numpy()
    union=changes.any(axis=1)
    selected_union=np.logical_or.reduce([v['selected'] for v in info.values()])
    for sex in LABELS:
        mask=np.ones(len(X),bool) if sex==0 else metadata.SEX.to_numpy()==sex
        for feature,details in info.items():
            changed=X[feature].to_numpy()!=perturbed[feature].to_numpy()
            selected=details['selected']
            rows.append({'scenario':scenario['scenario'],'layer':scenario['layer'],'repetition':repetition,
                'SEX':sex,'group':LABELS[sex],'feature':feature,'records':int(mask.sum()),
                'eligible':int((details['eligible']&mask).sum()),'selected':int((selected&mask).sum()),
                'eligible_selected':int((details['eligible']&selected&mask).sum()),
                'changed':int((changed&mask).sum()),'changed_cells':int((changed&mask).sum()),
                'clipped':int((details['clipped']&mask).sum()),'no_op':int((selected&~changed&mask).sum()),
                'outside_support':int((details['outside_support']&mask).sum()),
                'selected_outside_support':int((details['outside_support']&selected&mask).sum()),
                'special_state_selected':int((details['special_state']&selected&mask).sum()),
                'zero_scale':bool(details['zero_scale']),
                'nominal_participation':int((selected&mask).sum()) if scenario['layer']=='safe' else 0,
                'changed_rate':float(changed[mask].mean())})
        rows.append({'scenario':scenario['scenario'],'layer':scenario['layer'],'repetition':repetition,
            'SEX':sex,'group':LABELS[sex],'feature':'__union__','records':int(mask.sum()),
            'eligible':int(np.logical_or.reduce([v['eligible'] for v in info.values()])[mask].sum()),
            'selected':int(selected_union[mask].sum()),
            'eligible_selected':int((np.logical_or.reduce([v['eligible'] for v in info.values()])&selected_union&mask).sum()),
            'changed':int(union[mask].sum()),'changed_cells':int(changes[mask].sum()),
            'clipped':int(sum(v['clipped'][mask].sum() for v in info.values())),
            'no_op':int((selected_union&~union&mask).sum()),
            'outside_support':int(sum(v['outside_support'][mask].sum() for v in info.values())),
            'selected_outside_support':int(sum((v['outside_support']&v['selected']&mask).sum() for v in info.values())),
            'special_state_selected':int(sum((v['special_state']&v['selected']&mask).sum() for v in info.values())),
            'zero_scale':False,'nominal_participation':int(selected_union[mask].sum()) if scenario['layer']=='safe' else 0,
            'changed_rate':float(union[mask].mean())})
    return rows,union


def aggregate(table,keys,metrics):
    rows=[]
    for values,part in table.groupby(keys,sort=False):
        if not isinstance(values,tuple):
            values=(values,)
        base=dict(zip(keys,values))
        stochastic=str(base['scenario']).startswith('semantic')
        for metric in metrics:
            sample=part[metric].to_numpy(dtype=float)
            finite=sample[np.isfinite(sample)]
            rows.append({**base,'metric':metric,'mean':float(finite.mean()) if len(finite)==len(sample) else np.nan,
                         'mc_sd':float(np.std(finite,ddof=1)) if stochastic and len(finite)>1 else np.nan,
                         'minimum':float(finite.min()) if len(finite) else np.nan,
                         'maximum':float(finite.max()) if len(finite) else np.nan,
                         'repetitions':len(part),'valid_repetitions':len(finite),
                         'undefined_count':len(sample)-len(finite),
                         'mc_status':'between-repetition sample SD' if stochastic else 'not applicable: deterministic'})
    return pd.DataFrame(rows)


def run_robustness_evaluation():
    started=datetime.now(timezone.utc)
    TABLE_DIR.mkdir(parents=True,exist_ok=True)
    FIGURE_DIR.mkdir(parents=True,exist_ok=True)
    before=protected_hashes()
    validate_all_prior()
    source=source_identity()
    train,test,_,y,meta_train,metadata=load_split(pd.read_csv(DATA_PATH))
    models={k:joblib.load(ROOT/f'models/baseline_{k}.joblib') for k in MODELS}
    baseline,audit,baseline_scores,evidence=baseline_reproduction(test,y,metadata,models)
    if set(meta_train.ID)&set(metadata.ID):
        raise AssertionError('Split membership overlap')
    protocol=build_protocol(train,before,source)
    protocol['holdout_rows']=len(test)
    protocol['source_hashes']={name:sha256(ROOT/name) for name in SOURCES}
    config_path=TABLE_DIR/'protocol.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8'))!=protocol:
        raise ValueError('Prespecified protocol or source changed; review explicitly before replacing it')
    write_json(config_path,protocol)
    (TABLE_DIR/'protocol_rationale.md').write_text(RATIONALE,encoding='utf-8')
    (TABLE_DIR/'safe_source_inspection.txt').write_text(inspection_text(),encoding='utf-8')
    # Protocol is now on disk; all analytical settings precede robustness results.
    fixtures=validate_fixtures()
    rank_checks=validate_weighted_ranks()
    pd.testing.assert_frame_equal(train,load_split(pd.read_csv(DATA_PATH))[0])
    print('Protocol saved; frozen baseline, SAFE fixtures, and weighted ranking calculations verified.',flush=True)
    X=test.reset_index(drop=True).copy(deep=True)
    metadata=metadata.reset_index(drop=True).copy(deep=True)
    y=np.asarray(y)
    base_metrics={(k,sex):outcome_metrics(y[(metadata.SEX.to_numpy()==sex) if sex else np.ones(len(X),bool)],
                                          baseline[k][(metadata.SEX.to_numpy()==sex) if sex else np.ones(len(X),bool)])
                  for k in MODELS for sex in LABELS}
    rows,changed_rows,exposures,rgr_rows,registry,calibration,undefined=[],[],[],[],[],[],[]
    cache={'ID':metadata.ID.to_numpy(),'SEX':metadata.SEX.to_numpy(),'actual':y,
           'duplicate_group_id':metadata.duplicate_group_id.to_numpy(),
           'baseline':np.stack([baseline[k] for k in MODELS])}
    primary={PRIMARY_SAFE:{'probabilities':[],'changed':[]},PRIMARY_SEMANTIC:{'probabilities':[],'changed':[]}}
    baseline_descriptive=[]
    for kind in MODELS:
        for sex in LABELS:
            mask=np.ones(len(X),bool) if not sex else metadata.SEX.to_numpy()==sex
            p=baseline[kind][mask]
            baseline_descriptive.append({'model':kind,'SEX':sex,'group':LABELS[sex],'records':int(mask.sum()),
                'default_rate':float(y[mask].mean()),'mean_probability':float(p.mean()),
                'median_probability':float(np.median(p)), 'fraction_within_0.05_of_threshold':float((np.abs(p-.5)<=.05).mean()),
                **base_metrics[(kind,sex)]})
            bins=reliability_bins(y[mask],p)
            bins['scenario']='baseline'; bins['model']=kind; bins['SEX']=sex; bins['repetition']=0
            calibration.extend(bins.to_dict('records'))
    feature_descriptive=[]
    for sex in (1,2):
        part=X[metadata.SEX==sex]
        for f in X.columns:
            feature_descriptive.append({'SEX':sex,'group':LABELS[sex],'feature':f,'records':len(part),
                'mean':float(part[f].mean()),'q25':float(part[f].quantile(.25)),
                'median':float(part[f].median()),'q75':float(part[f].quantile(.75))})
    count=0
    with threadpool_limits(limits=1):
        for specification in protocol['scenarios']:
            for repetition in range(1,specification['repetitions']+1):
                scenario={**specification,'repetition':repetition}
                name=scenario['scenario']
                features=scenario['features']
                order=np.arange(len(X))
                if 'permutation_seed' in scenario:
                    order=np.random.default_rng(scenario['permutation_seed']).permutation(len(X))
                inverse=np.argsort(order)
                source_X=X.iloc[order].reset_index(drop=True).copy(deep=True)
                if scenario['layer']=='safe':
                    transformed,participants=safe_transform(source_X,features,scenario['p'])
                    info={f:{'selected':participants[f][inverse], 'eligible':np.ones(len(X),bool),
                        'clipped':np.zeros(len(X),bool),'outside_support':np.zeros(len(X),bool),
                        'special_state':np.zeros(len(X),bool),'zero_scale':False} for f in features}
                else:
                    transformed,info=semantic_transform(source_X,scenario,protocol['semantic_rules'])
                canonical=transformed.iloc[inverse].reset_index(drop=True)
                exp,changed=exposure_rows(scenario,repetition,X,canonical,info,metadata)
                exposures.extend(exp)
                pair=[]
                for kind,model in models.items():
                    if scenario['layer']=='safe':
                        value,p_ordered=authentic_call(model,source_X,baseline[kind][order],features,
                            scenario['p'],scenario['kind']=='group',transformed)
                        p=p_ordered[inverse]
                        rgr_rows.append({'scenario':name,'model':kind,'kind':scenario['kind'],'scope':scenario['scope'],
                            'p':scenario['p'],'RGR':value,'undefined_reason':'' if np.isfinite(value) else 'zero original-score normalization',
                            'ordering':'permuted diagnostic' if 'permutation_seed' in scenario else 'canonical primary order',
                            'features':','.join(features)})
                    else:
                        p=model.predict_proba(transformed.copy(deep=True))[:,1]
                    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
                        raise ValueError('Invalid frozen-model probability')
                    pair.append(p)
                    for sex in LABELS:
                        mask=np.ones(len(X),bool) if sex==0 else metadata.SEX.to_numpy()==sex
                        movements,reason=movement(baseline[kind][mask],p[mask])
                        metrics=outcome_metrics(y[mask],p[mask])
                        base=base_metrics[(kind,sex)]
                        row={'scenario':name,'layer':scenario['layer'],'repetition':repetition,
                            'model':kind,'SEX':sex,'group':LABELS[sex],'records':int(mask.sum()),
                            'changed_records':int((changed&mask).sum()),'input_exposure_rate':float(changed[mask].mean()),
                            **movements,**metrics,**{'baseline_'+m:v for m,v in base.items()},
                            **{'delta_'+m:metrics[m]-base[m] for m in base}}
                        rows.append(row)
                        changed_mask=mask&changed
                        supplement,supp_reason=movement(baseline[kind][changed_mask],p[changed_mask])
                        changed_rows.append({'scenario':name,'model':kind,'SEX':sex,'repetition':repetition,
                            'scope':'changed_input_records_only','records':int(changed_mask.sum()),**supplement,
                            'undefined_reason':supp_reason})
                        for metric,val in {**movements,**metrics}.items():
                            if not np.isfinite(val):
                                undefined.append({'scenario':name,'repetition':repetition,'model':kind,'SEX':sex,
                                    'scope':'all_records','metric':metric,'reason':reason or 'single-outcome subgroup','count':1})
                        for metric,val in supplement.items():
                            if not np.isfinite(val):
                                undefined.append({'scenario':name,'repetition':repetition,'model':kind,'SEX':sex,
                                    'scope':'changed_input_records_only','metric':metric,'reason':supp_reason,'count':1})
                        if name in primary:
                            bins=reliability_bins(y[mask],p[mask])
                            bins['scenario']=name; bins['model']=kind; bins['SEX']=sex; bins['repetition']=repetition
                            calibration.extend(bins.to_dict('records'))
                key=f'realization_{count:04d}'
                cache[key]=np.stack(pair)
                cache[key+'_changed']=changed
                registry.append({'cache_key':key,'scenario':name,'repetition':repetition,'layer':scenario['layer'],
                    'scope':scenario['scope'],'kind':scenario['kind'],'severity':scenario.get('p',scenario.get('q')),
                    'level':scenario.get('level','deterministic'),'input_sha256':frame_hash(canonical),
                    'models_share_exact_input':True,'canonical_ID_order_restored':True,
                    'selected_records':int(np.logical_or.reduce([v['selected'] for v in info.values()]).sum()),
                    'changed_records':int(changed.sum()),'changed_cells':int((X.to_numpy()!=canonical.to_numpy()).sum())})
                if name in primary:
                    primary[name]['probabilities'].append(np.stack(pair))
                    primary[name]['changed'].append(changed.copy())
                count+=1
            print(f'Completed {name} ({count} saved realizations)',flush=True)
    for data in primary.values():
        data['probabilities']=np.array(data['probabilities'])
        data['changed']=np.array(data['changed'])
    pd.testing.assert_frame_equal(X,test.reset_index(drop=True),check_exact=True)
    np.testing.assert_array_equal(metadata.ID,audit[audit.model==MODELS[0]].ID)
    np.testing.assert_array_equal(metadata.SEX,audit[audit.model==MODELS[0]].SEX)
    np.testing.assert_array_equal(metadata.default,y)
    print('Saving cached predictions and diagnostics before paired sampling uncertainty.',flush=True)
    np.savez_compressed(TABLE_DIR/'prediction_cache.npz',**compact_cache(cache,registry))
    diagnostics=pd.DataFrame(rows)
    registry=pd.DataFrame(registry)
    exposure=pd.DataFrame(exposures)
    metric_columns=MOVEMENT+list(base_metrics[(MODELS[0],0)])+['delta_'+m for m in base_metrics[(MODELS[0],0)]]
    paired=[]
    for keys,part in diagnostics.groupby(['scenario','repetition','SEX'],sort=False):
        lr,gbm=[part[part.model==kind].iloc[0] for kind in MODELS]
        np.testing.assert_array_equal([lr.records,lr.changed_records],[gbm.records,gbm.changed_records])
        paired.append({'scenario':keys[0],'repetition':keys[1],'SEX':keys[2],'model':'GBM_minus_LR',
                       **{m:gbm[m]-lr[m] for m in metric_columns}})
    comparisons=pd.DataFrame(paired)
    artifacts={'baseline_reproduction':evidence,'baseline_metrics':baseline_scores,'audit_predictions':audit,
        'safe_fixture_validation':fixtures,'weighted_ranking_validation':rank_checks,'realization_registry':registry,
        'authentic_rgr':pd.DataFrame(rgr_rows),'conventional_diagnostics':diagnostics,
        'changed_record_movement':pd.DataFrame(changed_rows),'paired_comparisons':comparisons,
        'diagnostic_summary':aggregate(diagnostics,['scenario','model','SEX'],metric_columns),
        'paired_summary':aggregate(comparisons,['scenario','model','SEX'],metric_columns),
        'exposure_counts':exposure,
        'exposure_summary':aggregate(exposure,['scenario','feature','SEX'],['eligible','selected','eligible_selected','changed','changed_cells','clipped','no_op','outside_support','selected_outside_support','special_state_selected','changed_rate']),
        'baseline_by_sex':pd.DataFrame(baseline_descriptive),'baseline_features_by_sex':pd.DataFrame(feature_descriptive),
        'calibration_bins':pd.DataFrame(calibration),
        'undefined_results':pd.DataFrame(undefined,columns=['scenario','repetition','model','SEX','scope','metric','reason','count'])}
    artifacts['individual_rgr']=artifacts['authentic_rgr'].query("kind == 'individual'").copy()
    artifacts['grouped_rgr']=artifacts['authentic_rgr'].query("kind == 'group'").copy()
    # Save basic diagnostics first so interruption never loses completed realization work.
    for name,frame in artifacts.items():
        frame.to_csv(TABLE_DIR/f'{name}.csv',index=False,lineterminator='\n')
    intervals,replicates=paired_bootstrap(metadata,baseline,primary)
    artifacts['sampling_intervals']=intervals
    artifacts['bootstrap_replicates']=replicates
    intervals.to_csv(TABLE_DIR/'sampling_intervals.csv',index=False,lineterminator='\n')
    replicates.to_csv(TABLE_DIR/'bootstrap_replicates.csv',index=False,lineterminator='\n')
    permuted='safe__group_all_features__p_0.05__permuted'
    permutation=[]
    for kind in MODELS:
        for sex in LABELS:
            canonical=diagnostics[(diagnostics.scenario==PRIMARY_SAFE)&(diagnostics.model==kind)&(diagnostics.SEX==sex)].iloc[0]
            alternate=diagnostics[(diagnostics.scenario==permuted)&(diagnostics.model==kind)&(diagnostics.SEX==sex)].iloc[0]
            for m in ['input_exposure_rate']+FOCUS:
                permutation.append({'model':kind,'SEX':sex,'metric':m,'canonical':canonical[m],
                                    'permuted':alternate[m],'permuted_minus_canonical':alternate[m]-canonical[m]})
        for m in ['RGR']:
            canonical=artifacts['authentic_rgr'].query('scenario == @PRIMARY_SAFE and model == @kind').iloc[0][m]
            alternate=artifacts['authentic_rgr'].query('scenario == @permuted and model == @kind').iloc[0][m]
            permutation.append({'model':kind,'SEX':0,'metric':m,'canonical':canonical,'permuted':alternate,'permuted_minus_canonical':alternate-canonical})
    # Record-level comparison after restoration, including changes in tail membership.
    keys={r.scenario:r.cache_key for r in registry.itertuples() if r.scenario in (PRIMARY_SAFE,permuted)}
    one,two=keys[PRIMARY_SAFE],keys[permuted]
    record=pd.DataFrame({'ID':metadata.ID,'SEX':metadata.SEX,'actual':y,
                         'canonical_changed':cache[one+'_changed'],'permuted_changed':cache[two+'_changed']})
    for j,kind in enumerate(MODELS):
        record[kind+'_canonical_probability']=cache[one][j]
        record[kind+'_permuted_probability']=cache[two][j]
    artifacts['permutation_record_comparison']=record
    artifacts['row_order_diagnostic']=pd.DataFrame(permutation)
    for name in ('permutation_record_comparison','row_order_diagnostic'):
        artifacts[name].to_csv(TABLE_DIR/f'{name}.csv',index=False,lineterminator='\n')
    from src.robustness_report import make_report
    make_report(artifacts,protocol,FIGURE_DIR,REPORT)
    after=protected_hashes()
    if before!=after:
        raise AssertionError('Protected artifacts changed during robustness evaluation')
    environment={d.metadata['Name']:d.version for d in distributions()}
    manifest={'stage':'robustness_evaluation','status':'complete','started_utc':started.isoformat(),
        'completed_utc':datetime.now(timezone.utc).isoformat(),'holdout_rows':len(X),'training_rows':len(train),
        'protected_artifact_hashes':before,'protected_artifacts_unchanged':True,
        'source_hashes':{name:sha256(ROOT/name) for name in SOURCES},'safe_identity':source,
        'protocol_sha256':sha256(config_path),'rationale_sha256':sha256(TABLE_DIR/'protocol_rationale.md'),
        'models_fitted':False,'calibration_fitted':False,'selected_threshold':None,
        'threshold':.5,'class_rule':'p > .50; ties class 0','predictors':list(X.columns),
        'scenarios':len(protocol['scenarios']),'realizations':len(registry),
        'semantic_repetitions':30,'bootstrap_repetitions':2000,'bootstrap_seed':42,
        'python':platform.python_version(),'platform':platform.platform(),'environment_versions':dict(sorted(environment.items())),
        'prediction_cache_format':CACHE_FORMAT,
        'initial_prespecified_protocol_sha256':sha256(TABLE_DIR/'initial_prespecified_protocol.json'),
        'implementation_revision_sha256':sha256(TABLE_DIR/'implementation_revision.json'),
        'undefined_diagnostics_count':int(artifacts['undefined_results']['count'].sum()),
        'undefined_bootstrap_count':int(intervals.undefined_replicates.sum()),
        'undefined_SAFE_fixture_count':int((fixtures.status=='undefined').sum()),
        'output_hashes':{str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in
                         [*sorted(TABLE_DIR.glob('*.csv')),TABLE_DIR/'prediction_cache.npz',
                          TABLE_DIR/'protocol_rationale.md',TABLE_DIR/'safe_source_inspection.txt',
                          *sorted(FIGURE_DIR.glob('*')),REPORT]}}
    write_json(TABLE_DIR/'run_manifest.json',manifest)
    print(f'Complete: {len(protocol["scenarios"])} scenarios, {len(registry)} realizations; protected hashes unchanged.',flush=True)
    return manifest


def verify_completion():
    first=run_robustness_evaluation()
    # Preserve compact first-run hash evidence, then run every scenario and bootstrap again.
    first_hashes={name:digest for name,digest in first['output_hashes'].items() if name.endswith('.csv')}
    first_cache=hashlib.sha256((TABLE_DIR/'prediction_cache.npz').read_bytes()).hexdigest()
    first_numerical_manifest={k:v for k,v in first.items() if k not in ('started_utc','completed_utc','output_hashes')}
    second=run_robustness_evaluation()
    second_numerical_manifest={k:v for k,v in second.items() if k not in ('started_utc','completed_utc','output_hashes')}
    if first_numerical_manifest!=second_numerical_manifest:
        raise AssertionError('Non-timestamp manifest fields changed on rerun')
    if any(second['output_hashes'].get(name)!=digest for name,digest in first_hashes.items()):
        raise AssertionError('Robustness CSVs differ on second full run')
    if sha256(TABLE_DIR/'prediction_cache.npz')!=first_cache:
        raise AssertionError('Prediction cache differs on second full run')
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True)
    log=tests.stdout+tests.stderr
    (TABLE_DIR/'test_results.txt').write_text(log,encoding='utf-8')
    print(log,flush=True)
    dependency=subprocess.run([sys.executable,'-m','pip','check'],cwd=ROOT,capture_output=True,text=True)
    (TABLE_DIR/'dependency_check.txt').write_text(dependency.stdout+dependency.stderr,encoding='utf-8')
    if tests.returncode or dependency.returncode:
        raise RuntimeError('Verification failed; see test_results.txt and dependency_check.txt')
    result={'two_full_runs_csvs_byte_identical':True,'csv_count':len(first_hashes),
        'prediction_cache_byte_identical':True,'non_timestamp_manifest_fields_identical':True,
        'timestamp_exception':'started_utc/completed_utc differ; environment and all other metadata compared exactly',
        'protected_artifacts_unchanged':protected_hashes()==first['protected_artifact_hashes'],
        'all_project_tests_passed':True,'tests_passed':int(re.search(r'Ran (\d+) tests',log).group(1)),
        'pip_check_passed':True,'test_log_sha256':sha256(TABLE_DIR/'test_results.txt'),
        'manifest_sha256':sha256(TABLE_DIR/'run_manifest.json'),
        'first_run_csv_hashes':first_hashes,
        'checks':['baseline reproduction','authentic SAFE source/fixtures/input equality','training-only rules',
                  'ID/SEX/outcome/split/order invariance','shared model inputs','nested mask/sign coupling',
                  'paired metric directions','sklearn weighted metric equivalence','all scenario counts',
                  'complete 2000 resamples and 30 repetitions','report/table traceability']}
    if not result['protected_artifacts_unchanged']:
        raise AssertionError('Protected artifact hashes changed after verification')
    write_json(TABLE_DIR/'verification.json',result)
    print('Two full runs and all project/dependency checks passed.',flush=True)
