"""Persist Job A; derive Job B from its pinned models, values and seeds."""
from pathlib import Path
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_info,threadpool_limits
from research.model_selection_v2 import pilot as p
from research.model_selection_v2.confirmation_v1.schema import build_candidate
from research.model_selection_v2.confirmation_v1.selection import split,select_arms
from research.model_selection_v2.confirmation_v2 import judge,audit
from research.model_selection_v2.confirmation_v2 import arms as prior_arms
from research.model_selection_v2.confirmation_v2.reporting import repeat_table
from research.model_selection_v2.step1_v1.core import ranking,choice
from .selection import selection
from .names import MARGIN_ARM,MARGIN_LABEL,rename
from .artifacts import (private_folder,write_json,read_json,save_judge,load_judge,manifest,verify,exact,
    save_reloaded_model,verify_model_files,model_snapshot)


def check_threads():
    info=[dict(prefix=x['prefix'],internal_api=x['internal_api'],num_threads=x['num_threads']) for x in threadpool_info()]
    if not info or any(x['num_threads']>2 for x in info):raise RuntimeError('Numerical thread cap not enforced')
    return sorted(info,key=lambda x:(x['prefix'],x['internal_api']))


def source_gate(source):
    if source.get('purpose') not in ('fixture','timing') or source.get('kind')!='artificial':
        raise RuntimeError('This preparation CLI is restricted to artificial data; no Brazil execution')


def select_job_A(folder,rows,evidence,meta,repeat):
    chosen=select_arms(folder,rows,evidence,meta,repeat)
    scores=prior_arms.diagnostic_scores(rows,evidence)
    chosen.update({arm:ranking(values)[0] for arm,values in scores.items()})
    checks=read_json(folder/'fairness_check.json')
    ranked=checks['ranking'];safe=chosen['safe']
    blocked={c:(any(upper is None or not upper<.0005 for lo,upper in
        checks['candidate_checks'][c]['joint_intervals_female_male']) if c!=safe else False) for c in p.CANDIDATES}
    chosen[MARGIN_ARM]=choice(ranked,safe,blocked)
    write_json(folder/'choices.json',chosen)
    write_json(folder/'diagnostic_scores.json',scores)
    return chosen


def run_A(source_path,folder,repeat):
    started=time.perf_counter();source=joblib.load(source_path);source_gate(source)
    return fit_and_score_A(source, folder, repeat, started)


def fit_and_score_A(source, folder, repeat, started):
    folder=private_folder(folder)
    data=source['data'];schema=data.schema
    meta,positions,split_audit=split(data.metadata,repeat,set())
    prepared,medians=data.prepare(positions['fitting'])
    role_X={role:prepared.iloc[pos].reset_index(drop=True) for role,pos in positions.items()}
    role_m={role:meta.iloc[pos].reset_index(drop=True) for role,pos in positions.items()}
    payload=dict(role_X=role_X,role_m=role_m,schema=schema,families=source['families'],
        dataset_id=source['dataset_id'],repeat=repeat,purpose=source['purpose'],role_sizes=split_audit['sizes'])
    joblib.dump(payload,folder/'inputs.joblib')
    meta.to_csv(folder/'assignment.csv',index=False)
    write_json(folder/'split_audit.json',split_audit)
    write_json(folder/'medians.json',medians)
    seeds=dict(repeat=repeat,split=20261114+repeat,bootstrap=42+repeat,candidate_random_state=42,
        judge_master=20261007,judge_scope='confirm_v1:'+source['dataset_id']+':<family>',
        data_seed=source['data_seed'],arm_labels={MARGIN_ARM:MARGIN_LABEL})
    write_json(folder/'seeds.json',seeds)
    model_folder=folder/'models';model_folder.mkdir()
    models={};file_hashes={}
    for candidate in p.CANDIDATES:
        model=build_candidate(candidate,schema)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(role_X['fitting'],role_m['fitting'].default)
        if any(issubclass(w.category,ConvergenceWarning) for w in caught):raise RuntimeError('Candidate convergence failure')
        models[candidate],file_hashes[candidate]=save_reloaded_model(model,model_folder/(candidate+'.joblib'))
        p.positive_probability(models[candidate],role_X['fitting'])
    write_json(folder/'model_file_sha256.json',file_hashes)
    rows,evidence=selection(folder,role_X['fitting'],role_X['selection'],role_m['selection'],models,schema)
    chosen=select_job_A(folder,rows,evidence,role_m['selection'],repeat)
    judged=judge.evaluate(role_X['judge'],role_X['fitting'],role_m['judge'],models,schema,source['families'],
        dataset_id=source['dataset_id'],levels=('J1',),secondary=False,group_audit=False,purpose=source['purpose'])['J1']
    save_judge(folder,'J1',judged)
    repeat_table(rows,chosen,{'J1':judged},repeat,split_audit['sizes']).to_csv(folder/'per_repeat_primary.csv',index=False)
    write_json(folder/'thread_info.json',check_threads())
    write_json(folder/'execution.json',dict(job='A',seconds=time.perf_counter()-started))
    bound=manifest(folder)
    write_json(folder/'COMPLETE.json',dict(job='A',repeat=repeat,complete=True,artifact_manifest_sha256=bound))


def run_B(source_A,folder):
    started=time.perf_counter();source_A=Path(source_A)
    if not read_json(source_A/'COMPLETE.json')['complete']:raise RuntimeError('Incomplete Job A')
    verify(source_A)
    payload=joblib.load(source_A/'inputs.joblib')
    source_gate(dict(kind='artificial',purpose=payload['purpose']))
    return score_B(source_A, folder, payload, started)


def score_B(source_A, folder, payload, started):
    folder=private_folder(folder)
    role_X,role_m=payload['role_X'],payload['role_m'];schema=payload['schema']
    models={c:joblib.load(source_A/'models'/(c+'.joblib')) for c in p.CANDIDATES}
    file_hashes=verify_model_files(source_A,p.CANDIDATES)
    before={c:model_snapshot(m,role_X) for c,m in models.items()}
    saved=load_judge(source_A,'J1',p.CANDIDATES)
    reproduced=judge.evaluate(role_X['judge'],role_X['fitting'],role_m['judge'],models,schema,payload['families'],
        dataset_id=payload['dataset_id'],levels=('J1',),secondary=False,group_audit=False,purpose=payload['purpose'])['J1']
    exact(saved,reproduced)
    write_json(folder/'J1_reproduction_gate.json',dict(passed=True,equality='exact; zero tolerance',
        models_refitted=False,all_clean_and_scenario_predictions_scores_registry_and_seeds_checked=True))
    reproduced['secondary']=judge.secondary_endpoints(role_m['judge'],reproduced['clean'],reproduced['predictions'],
        reproduced['registry'],rank_measures=True)
    reproduced['group_audit']=audit.candidate_audit(role_m['judge'],role_X['judge'].AGE.to_numpy(),
        reproduced['clean'],reproduced['predictions'])
    save_judge(folder,'J1',reproduced)
    higher=judge.evaluate(role_X['judge'],role_X['fitting'],role_m['judge'],models,schema,payload['families'],
        dataset_id=payload['dataset_id'],levels=('J2','J3'),secondary=True,group_audit=False,purpose=payload['purpose'])
    for level,result in higher.items():save_judge(folder,level,result)
    rows=pd.read_csv(source_A/'selection_values.csv',float_precision='round_trip').set_index('candidate')
    chosen=read_json(source_A/'choices.json')
    with np.load(source_A/'selection_bootstrap_values.npz',allow_pickle=False) as data:
        checks=prior_arms.pair_checks(data['point'],data['draws'],ranked=ranking(rows.Score2.to_dict()),repeat=payload['repeat'])
    safe=chosen['safe'];pairs={(r['challenger'],r['incumbent']):r for r in checks}
    ranked=ranking(rows.Score2.to_dict())
    frozen={c:(pairs[c,safe]['blocked'] if c!=safe else False) for c in p.CANDIDATES}
    margin={c:(pairs[c,safe]['stricter_blocked'] if c!=safe else False) for c in p.CANDIDATES}
    if choice(ranked,safe,frozen)!=chosen['selector2'] or choice(ranked,safe,margin)!=chosen[MARGIN_ARM]:
        raise AssertionError('Job A choices do not reproduce from saved bootstrap draws')
    write_json(folder/'every_pair_gate_records.json',rename(checks))
    write_json(folder/'choices.json',chosen)
    write_json(folder/'seeds.json',read_json(source_A/'seeds.json'))
    write_json(folder/'model_file_sha256.json',file_hashes)
    write_json(folder/'group_audit_by_arm.json',audit.audit_for_choices(reproduced['group_audit'],chosen,payload['repeat']))
    full=repeat_table(rows,chosen,dict(J1=reproduced,**higher),payload['repeat'],payload['role_sizes'])
    primary=pd.read_csv(source_A/'per_repeat_primary.csv',float_precision='round_trip')
    # Compare native in-memory values after matching CSV round-trip formatting.
    full.to_csv(folder/'per_repeat_full.csv',index=False)
    restored=pd.read_csv(folder/'per_repeat_full.csv',float_precision='round_trip')
    exact(primary,restored.loc[:,primary.columns])
    for candidate,model in models.items():exact(before[candidate],model_snapshot(model,role_X))
    verify_model_files(source_A,p.CANDIDATES)
    write_json(folder/'thread_info.json',check_threads())
    write_json(folder/'upstream.json',dict(job_A_manifest_sha256=read_json(source_A/'COMPLETE.json')['artifact_manifest_sha256']))
    write_json(folder/'execution.json',dict(job='B',seconds=time.perf_counter()-started,models_refitted=False))
    bound=manifest(folder)
    write_json(folder/'COMPLETE.json',dict(job='B',repeat=payload['repeat'],complete=True,artifact_manifest_sha256=bound))


def worker(specification):
    import sys
    if not getattr(sys,'_credit_risk_jobs_offline',False):raise RuntimeError('Use the network-denying thread-capped launcher')
    spec=read_json(specification)
    with threadpool_limits(limits=2):
        check_threads()
        if spec['job']=='A':run_A(Path(spec['source']),Path(spec['output']),int(spec['repeat']))
        elif spec['job']=='B':run_B(Path(spec['source_A']),Path(spec['output']))
        else:raise ValueError('Unknown job')


if __name__=='__main__':
    import sys
    worker(sys.argv[1])
