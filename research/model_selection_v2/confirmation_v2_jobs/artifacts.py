"""Exclusive writes, checksum-verified handoff and exact scientific comparisons."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def private_folder(path):
    path=Path(path);path.mkdir(exist_ok=False,parents=True)
    (path/'.gitignore').write_text('*\n',encoding='ascii')
    return path


def save_judge(folder,level,result):
    write_json(folder/(level+'_scores.json'),result['scores'])
    write_json(folder/(level+'_registry.json'),result['registry'])
    if 'secondary' in result:write_json(folder/(level+'_secondary.json'),result['secondary'])
    if 'group_audit' in result:write_json(folder/(level+'_group_audit.json'),result['group_audit'])
    for candidate,clean in result['clean'].items():
        with (folder/(level+'_'+candidate+'.npz')).open('xb') as f:
            np.savez_compressed(f,clean=clean,**{k:np.asarray(v) for k,v in result['predictions'][candidate].items()})


def load_judge(folder,level,candidates):
    result=dict(scores=read_json(folder/(level+'_scores.json')),registry=read_json(folder/(level+'_registry.json')),
        clean={},predictions={})
    for candidate in candidates:
        with np.load(folder/(level+'_'+candidate+'.npz'),allow_pickle=False) as data:
            result['clean'][candidate]=data['clean'].copy()
            result['predictions'][candidate]={k:[p.copy() for p in data[k]] for k in data.files if k!='clean'}
    for name in ('secondary','group_audit'):
        path=folder/(level+'_'+name+'.json')
        if path.exists():result[name]=read_json(path)
    return result


def manifest(folder):
    files={p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*'))
        if p.is_file() and p.name not in ('artifact_manifest.json','COMPLETE.json')}
    write_json(folder/'artifact_manifest.json',files)
    return sha(folder/'artifact_manifest.json')


def verify(folder):
    files=read_json(folder/'artifact_manifest.json')
    for name,expected in files.items():
        if sha(folder/name)!=expected:raise ValueError('Job handoff artifact changed: '+name)
    return files


def exact(a,b):
    if isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,pd.DataFrame):pd.testing.assert_frame_equal(a,b,check_exact=True)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError('Dictionary keys differ')
        for key in a:exact(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise AssertionError('Sequence length differs')
        for x,y in zip(a,b):exact(x,y)
    else:
        if a!=b:raise AssertionError('Scientific scalar differs')


def save_reloaded_model(model,path):
    """Pin the file bytes and return the object that Job A must actually score."""
    joblib.dump(model,path)
    return joblib.load(path),sha(path)


def verify_model_files(folder,candidates):
    """File identity is SHA-256, never a hash of unpickled estimator memory."""
    expected=read_json(folder/'model_file_sha256.json')
    actual={c:sha(folder/'models'/(c+'.joblib')) for c in candidates}
    exact(expected,actual)
    return actual


def model_snapshot(model,role_X):
    """Copy probabilities and logical fitted values, excluding tree padding."""
    classifier=model.named_steps['classifier']
    trees=[]
    if hasattr(classifier,'tree_'):
        trees=[classifier]
    elif hasattr(classifier,'estimators_'):
        estimators=classifier.estimators_
        trees=list(estimators.flat) if isinstance(estimators,np.ndarray) else list(estimators)
    tree_states=[]
    for estimator in trees:
        state={}
        for key,value in estimator.tree_.__getstate__().items():
            if isinstance(value,np.ndarray):
                # Copy each named field independently: dtype padding is not state.
                state[key]=({field:value[field].copy() for field in value.dtype.names}
                    if value.dtype.names else value.copy())
            else:state[key]=value
        tree_states.append(state)
    preprocessor=model.named_steps['preprocessor']
    scaler=preprocessor.named_transformers_['num']
    encoder=preprocessor.named_transformers_['cat']
    # These are the fitted transforms and classifier arrays of our four fixed
    # pipelines. Object arrays holding trees are compared above, field by field.
    fitted_arrays={key:value.copy() for key,value in vars(classifier).items()
        if key.endswith('_') and isinstance(value,np.ndarray) and not value.dtype.hasobject}
    return dict(model_type=type(classifier).__module__+'.'+type(classifier).__qualname__,
        classes=np.asarray(model.classes_).copy(),
        predict_proba={role:np.asarray(model.predict_proba(X)).copy() for role,X in role_X.items()},
        tree_states=tree_states,classifier_arrays=fitted_arrays,
        feature_names=preprocessor.get_feature_names_out().copy(),
        scaler={key:np.asarray(getattr(scaler,key)).copy()
            for key in ('mean_','var_','scale_','n_samples_seen_')},
        encoder_categories=[values.copy() for values in encoder.categories_])


def compare_models(a,b,role_X_a,role_X_b):
    """Require exact all-row predictions and field-wise tree/fitted state."""
    exact(role_X_a,role_X_b)
    snapshot_a=model_snapshot(a,role_X_a)
    exact(snapshot_a,model_snapshot(b,role_X_b))
    return dict(exact=True,rows_by_role={role:len(X) for role,X in role_X_a.items()},
        probability_values_checked=sum(value.size for value in snapshot_a['predict_proba'].values()),
        trees_checked=len(snapshot_a['tree_states']),tree_comparison='named fields and values; no padding bytes')


def compare_repeat_folders(a,b):
    """Compare values, not ZIP timestamps or execution/path/checksum metadata."""
    verify(a);verify(b)
    files_a={p.relative_to(a).as_posix() for p in a.rglob('*') if p.is_file()}
    files_b={p.relative_to(b).as_posix() for p in b.rglob('*') if p.is_file()}
    if files_a!=files_b:raise AssertionError('Output file sets differ')
    checked=0;models={}
    skip={'.gitignore','COMPLETE.json','artifact_manifest.json','execution.json','upstream.json','thread_info.json',
        'model_file_sha256.json'}
    inputs_a=inputs_b=None
    if (a/'inputs.joblib').exists():
        inputs_a,inputs_b=joblib.load(a/'inputs.joblib'),joblib.load(b/'inputs.joblib')
        exact(inputs_a,inputs_b)
        candidates=[path.stem for path in sorted((a/'models').glob('*.joblib'))]
        verify_model_files(a,candidates);verify_model_files(b,candidates)
    for name in sorted(files_a):
        if Path(name).name in skip:continue
        pa,pb=a/name,b/name
        if name.endswith('.npz'):
            with np.load(pa,allow_pickle=False) as x,np.load(pb,allow_pickle=False) as y:
                if set(x.files)!=set(y.files):raise AssertionError('Array keys differ')
                for key in x.files:exact(x[key],y[key]);checked+=x[key].size
        elif name.endswith('.json'):
            exact(read_json(pa),read_json(pb))
        elif name.endswith('.csv'):
            exact(pd.read_csv(pa,float_precision='round_trip'),pd.read_csv(pb,float_precision='round_trip'))
        elif name.endswith('.joblib'):
            if name=='inputs.joblib':
                exact(inputs_a,inputs_b)
            elif Path(name).parent==Path('models') and inputs_a is not None:
                models[name]=compare_models(joblib.load(pa),joblib.load(pb),inputs_a['role_X'],inputs_b['role_X'])
                checked+=models[name]['probability_values_checked']
            else:raise AssertionError('Unexpected unverified joblib artifact: '+name)
        else:
            if pa.read_bytes()!=pb.read_bytes():raise AssertionError('Artifact differs: '+name)
    return dict(exact=True,array_values_checked=checked,fitted_model_comparisons=models,
        input_bundles_compared_exactly=inputs_a is not None,
        excluded='Execution/resource records, file-container SHA-256 records (verified within each run), upstream hashes and ZIP timestamps; all scientific values and seeds compared')
