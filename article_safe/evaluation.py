"""Final refits and SAFE assessments after the predictor choices are frozen."""
from datetime import datetime, timezone
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from article_safe.artifacts import existing_run, verify_preparation, verify_stage, write_json, stage_manifest
from article_safe.data import load_development_data, check_evaluation_artifacts
from article_safe.protocol import ROOT, sha256
from article_safe.models import fit_design, fit_lr, fit_gbm
from article_safe.explainability import background, explain, importance, lr_parameters, auxiliary_regression
from article_safe.fairness import contribution_table, subgroup_coefficients
from article_safe.inference import paired_delong, holm_adjust
from article_safe.uncertainty import conditional_intervals
from src.baseline import DATA_PATH, load_split
from src.performance_evaluation import performance_metrics


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def evaluate(run_id):
    path = existing_run(run_id)
    protocol = verify_preparation(path)
    selection_manifest = verify_stage(path / 'selection', protocol)
    choices = json.loads((path / 'selection/selected_subsets.json').read_text())
    stage = path / 'evaluation'
    if stage.exists():
        raise FileExistsError('Evaluation already exists; preserve it and use a fresh run.')
    check_evaluation_artifacts()
    development = load_development_data(protocol)
    X_train, y_train, meta_train = development.X, development.y, development.metadata
    stage.mkdir()
    models_dir = stage / 'models'
    models_dir.mkdir()
    models, fit_records, events = {}, [], []
    events.append({'event': 'selection_verified', 'time_utc': _timestamp(),
                   'selected_subsets_sha256': sha256(path / 'selection/selected_subsets.json')})
    with threadpool_limits(limits=1):
        for family in ('lr_mle', 'gbm'):
            selected = choices[family].get('selected_features')
            for role, features in (('full', protocol['features']), ('selected', selected)):
                model_id = family + '_' + role
                if features is None:
                    fit_records.append({'model': model_id, 'status': 'not_estimable', 'reason': 'training_selection_not_estimable', 'predictors': None})
                    continue
                print(f'Final training refit: {model_id}, {len(features)} predictors', flush=True)
                if family == 'lr_mle':
                    design = fit_design(X_train, features, reference_encoding=True)
                    model = fit_lr(design, y_train, features, protocol)
                else:
                    model = fit_gbm(X_train, y_train, features, protocol)
                fit_records.append({'model': model_id, 'predictors': len(features), 'features': list(features), **model.diagnostics})
                model.active_mapping.to_csv(stage / f'{model_id}_encoded_mapping.csv', index=False)
                if model.estimable:
                    models[model_id] = model
                    joblib.dump(model, models_dir / f'{model_id}.joblib')
                events.append({'event': 'final_refit_finished', 'model': model_id, 'time_utc': _timestamp(), 'status': model.diagnostics['status']})
    # All final fits occur before the following holdout-loading/evaluation boundary.
    write_json(stage / 'final_fit_complete.json', {'fit_records': fit_records, 'events': events})
    parts = load_split(pd.read_csv(DATA_PATH))
    X_test, y_test, meta_test = parts[1], parts[3], parts[5]
    X_test = X_test.loc[:, protocol['features']]
    events.append({'event': 'holdout_evaluation_started', 'time_utc': _timestamp()})
    probabilities, prediction_rows, metric_rows, group_rows, gap_rows = {}, [], [], [], []
    statuses = {row['model']: row for row in fit_records}
    reference_lr = joblib.load(ROOT / 'models/baseline_logistic.joblib')
    reference_gbm = joblib.load(ROOT / 'models/baseline_gradient_boosting.joblib')
    with threadpool_limits(limits=1):
        for model_id in protocol['final_roles'] + [protocol['estimator_reference']]:
            if model_id == protocol['estimator_reference']:
                probability = reference_lr.predict_proba(X_test)[:, 1]
                predictors, kind, status = 22, 'regularized LR reference', 'reference_only'
            elif model_id not in models:
                row = statuses[model_id]
                metric_rows.append({'model': model_id, 'status': 'not_estimable', 'reason': row['reason'], 'predictors': row['predictors']})
                continue
            else:
                model = models[model_id]
                probability = model.probability(X_test)
                predictors, kind, status = len(model.features), model.family, 'ok'
                restored = joblib.load(models_dir / f'{model_id}.joblib')
                np.testing.assert_array_equal(probability, restored.probability(X_test))
                if model_id == 'gbm_full':
                    frozen_probability = reference_gbm.predict_proba(X_test)[:, 1]
                    difference = float(np.max(np.abs(probability - frozen_probability)))
                    np.testing.assert_allclose(probability, frozen_probability, atol=1e-15, rtol=0)
                    write_json(stage / 'full_gbm_reproduction.json', {'max_probability_difference': difference, 'tolerance': 1e-15, 'passed': True})
                write_json(stage / f'{model_id}_unknown_categories.json', model.unknown_categories(X_test))
            probabilities[model_id] = np.asarray(probability)
            scores = performance_metrics(y_test, probability)
            metric_rows.append({'model': model_id, 'family': kind, 'status': status, 'predictors': predictors, **scores})
            prediction = meta_test[['ID','SEX','default','duplicate_group_id']].copy().rename(columns={'default':'actual'})
            prediction['model'], prediction['probability_default'], prediction['predicted'] = model_id, probability, probability > .50
            prediction_rows.append(prediction)
            groups = {}
            for sex_code in (1,2):
                mask = meta_test.SEX.to_numpy() == sex_code
                groups[sex_code] = performance_metrics(np.asarray(y_test)[mask], probability[mask])
                group_rows.append({'model':model_id,'SEX':sex_code,'records':int(mask.sum()),**groups[sex_code]})
            gap_rows.extend({'model':model_id,'metric':metric,'female_minus_male':groups[2][metric]-groups[1][metric]} for metric in groups[1])
    pd.concat(prediction_rows,ignore_index=True).to_csv(stage/'holdout_predictions.csv',index=False)
    pd.DataFrame(metric_rows).to_csv(stage/'holdout_metrics.csv',index=False)
    pd.DataFrame(group_rows).to_csv(stage/'group_metrics.csv',index=False)
    pd.DataFrame(gap_rows).to_csv(stage/'group_metric_gaps.csv',index=False)
    comparisons = []
    for first, second in protocol['final_comparisons']:
        if first in probabilities and second in probabilities:
            result = paired_delong(y_test, probabilities[first], probabilities[second]).to_dict()
        else:
            result = {'status':'model_not_estimable','difference':None,'p_value':None}
        comparisons.append({'first':first,'second':second,**result})
    adjusted = holm_adjust([row['p_value'] for row in comparisons],planned_family_size=4)
    for row,p in zip(comparisons,adjusted):
        row['p_value_holm'] = float(p) if np.isfinite(p) else None
    write_json(stage/'auc_comparisons.json',comparisons)
    pd.DataFrame([{k:v for k,v in row.items() if k!='covariance'} for row in comparisons]).to_csv(stage/'auc_comparisons.csv',index=False)
    reference, reference_meta = background(X_train,meta_train,protocol['shap']['background_size'],protocol['seed'])
    reference_meta.to_csv(stage/'final_shap_background.csv',index=False)
    contributions, importance_tables, retained, component_statuses = {}, {}, {}, []
    with threadpool_limits(limits=1):
        for model_id,model in models.items():
            print(f'Final SAFE explanation/fairness assessment: {model_id}',flush=True)
            result = explain(model,X_test,reference,protocol['features'])
            contributions[model_id] = result['original']
            np.savez_compressed(stage/f'{model_id}_shap.npz',original=result['original'],encoded=result['encoded'],
                                base_value=result['base_value'],ID=meta_test.ID.to_numpy())
            result['mapping'].to_csv(stage/f'{model_id}_shap_mapping.csv',index=False)
            write_json(stage/f'{model_id}_shap_checks.json',{'max_additivity_error':result['max_additivity_error'],'units':result['units'],'background_training_only':True})
            table = importance(result['original'],protocol['features'])
            importance_tables[model_id] = table
            table.to_csv(stage/f'{model_id}_importance.csv',index=False)
            contribution_table(result['original'],meta_test.SEX,protocol['features'],model.features,model.family).to_csv(stage/f'{model_id}_contribution_fairness.csv',index=False)
            if model.family=='lr_mle':
                parameters=lr_parameters(model,development.groups)
                parameters['post_selection']=model_id.endswith('_selected')
                parameters.to_csv(stage/f'{model_id}_parameters.csv',index=False)
                coefficients,diagnostics=subgroup_coefficients(model,y_train,meta_train,protocol)
                coefficients.to_csv(stage/f'{model_id}_subgroup_coefficients.csv',index=False)
                write_json(stage/f'{model_id}_subgroup_fit_diagnostics.json',diagnostics)
                component_statuses.append({'model':model_id,'component':'coefficient_KS','status':'not_identifiable_from_article'})
            else:
                regression,status=auxiliary_regression(y_test,result['original'],model.features,protocol['features'],meta_test.duplicate_group_id.to_numpy())
                if regression is not None:
                    regression.to_csv(stage/f'{model_id}_shap_regression.csv',index=False)
                write_json(stage/f'{model_id}_shap_regression_status.json',status)
                component_statuses.append({'model':model_id,'component':'SHAP outcome regression',**status})
    for family in ('lr_mle','gbm'):
        full_id,selected_id=family+'_full',family+'_selected'
        if full_id in importance_tables and selected_id in models:
            table=importance_tables[full_id].set_index('feature')
            total=float(table.mean_abs_shap.sum())
            retained[selected_id]=100*float(table.loc[list(models[selected_id].features),'mean_abs_shap'].sum())/total if total else None
            retained[full_id]=100.0 if total else None
    metrics=pd.DataFrame(metric_rows)
    metrics['retained_full_model_importance_percent']=metrics.model.map(retained)
    metrics.to_csv(stage/'holdout_metrics.csv',index=False)
    for role in protocol['final_roles']:
        if role not in models:
            component_statuses.append({'model':role,'component':'article model and dependent assessments','status':'not_estimable','reason':statuses[role]['reason']})
    write_json(stage/'component_statuses.json',component_statuses)
    bootstrap=conditional_intervals(y_test,probabilities,contributions,meta_test.SEX,meta_test.duplicate_group_id,
                                    protocol['features'],protocol,stage)
    write_json(stage/'conditional_uncertainty.json',bootstrap)
    events.append({'event':'holdout_assessments_finished','time_utc':_timestamp()})
    write_json(stage/'execution_events.json',events)
    stage_manifest(stage,'evaluation',protocol,{'selection_manifest_sha256':sha256(path/'selection/stage_manifest.json'),
                   'selected_subsets_sha256':sha256(path/'selection/selected_subsets.json'),
                   'prior_artifact_checks':check_evaluation_artifacts(),'estimable_roles':list(models),
                   'heldout_records':len(X_test),'conditional_uncertainty':bootstrap})
    print(metrics[['model','status','predictors','ROC-AUC','Brier Score','Recall']].to_string(index=False),flush=True)
    return stage