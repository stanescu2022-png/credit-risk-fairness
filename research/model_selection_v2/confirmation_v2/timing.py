"""One discarded 50k-row timing repeat; no Brazil records or outcomes loaded."""
import json
from pathlib import Path
import sys
import tempfile
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from research.model_selection_v2 import pilot as p
from research.model_selection_v2.confirmation_v1.schema import Schema,build_candidate
from research.model_selection_v2.confirmation_v1.brazil import BrazilInputs,NUMERIC
from research.model_selection_v2.confirmation_v1.selection import split,selection
from .arms import select
from .audit import audit_for_choices
from .judge import evaluate
from .reporting import repeat_table

FAMILIES={
    'income_and_employment':tuple('PERSONAL_MONTHLY_INCOME OTHER_INCOMES PERSONAL_ASSETS_VALUE QUANT_CARS COMPANY MONTHS_IN_THE_JOB PROFESSION_CODE OCCUPATION_TYPE'.split()),
    'residence_and_contact':tuple('RESIDENCIAL_STATE RESIDENCE_TYPE MONTHS_IN_RESIDENCE FLAG_RESIDENCIAL_PHONE POSTAL_ADDRESS_TYPE FLAG_EMAIL FLAG_PROFESSIONAL_PHONE'.split()),
    'existing_credit':tuple('FLAG_VISA FLAG_MASTERCARD FLAG_DINERS FLAG_AMERICAN_EXPRESS FLAG_OTHER_CARDS QUANT_BANKING_ACCOUNTS'.split())}
DATA_SEED=2026111701
DISCARDED_REPEAT=100000


def standin(root):
    """Schema and label cardinalities come from the prior outcome-free receipt."""
    receipt=json.loads((root/'research/model_selection_v2/confirmation_preparation_2026-11-16/brazil_loader_gate.json').read_text())
    order=tuple(receipt['predictor_order'])
    numeric=tuple(n for n in order if n in NUMERIC)
    unordered=tuple(n for n in order if n not in numeric and n!='PAYMENT_DAY')
    schema=Schema(order,numeric,{'PAYMENT_DAY':(1,5,10,15,20,25)},unordered,{'PAYMENT_DAY':(-1,)})
    rng=np.random.default_rng(DATA_SEED);n=50000;X=pd.DataFrame(index=np.arange(n))
    for name in order:
        if name=='AGE':X[name]=rng.integers(18,90,n).astype(float)
        elif name in numeric:
            if name=='QUANT_CARS':values=rng.integers(0,2,n)
            elif name=='QUANT_BANKING_ACCOUNTS':values=rng.integers(0,3,n)
            elif name=='QUANT_DEPENDANTS':values=rng.integers(0,8,n)
            elif name=='MONTHS_IN_THE_JOB':values=np.where(rng.random(n)<.998,0,rng.integers(1,240,n))
            elif name=='MONTHS_IN_RESIDENCE':values=rng.integers(0,120,n)
            else:values=np.rint(rng.lognormal(6,1,n))
            X[name]=values.astype(float)
        elif name=='PAYMENT_DAY':X[name]=rng.choice(schema.ordered[name],n)
        else:X[name]=rng.integers(0,len(receipt['label_codes'][name]),n)
    X.loc[rng.random(n)<.076,'MONTHS_IN_RESIDENCE']=np.nan
    groups=np.arange(n)
    for i in range(7):
        X.iloc[2*i+1]=X.iloc[2*i];groups[2*i+1]=groups[2*i]
    sex=np.where(rng.random(n)<.6,2,1);sex[:65]=0
    meta=pd.DataFrame(dict(ID=np.arange(n),SEX=sex,default=(rng.random(n)<.26).astype(int),duplicate_group_id=groups))
    return BrazilInputs(X,meta,schema,{},{}),receipt


def run(destination):
    if not getattr(sys,'_credit_risk_v2_offline',False):raise RuntimeError('Use the socket-denying offline runner')
    root=p.ROOT;out=Path(destination);out.mkdir(exist_ok=False)
    started=time.perf_counter();stages={}
    def mark(name,start):
        stages[name]=time.perf_counter()-start
        print(json.dumps(dict(stage=name,seconds=stages[name],elapsed_seconds=time.perf_counter()-started)),flush=True)
    point=time.perf_counter();data,schema_receipt=standin(root);mark('standin_generation',point)
    point=time.perf_counter()
    meta,positions,split_audit=split(data.metadata,DISCARDED_REPEAT,set())
    X,medians=data.prepare(positions['fitting'])
    role_X={role:X.iloc[pos].reset_index(drop=True) for role,pos in positions.items()}
    role_m={role:meta.iloc[pos].reset_index(drop=True) for role,pos in positions.items()}
    mark('split_and_fitting_median_preparation',point)
    models={}
    point=time.perf_counter()
    for c in p.CANDIDATES:
        candidate_started=time.perf_counter();model=build_candidate(c,data.schema)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(role_X['fitting'],role_m['fitting'].default)
        if any(issubclass(w.category,ConvergenceWarning) for w in caught):
            raise RuntimeError('Timing candidate did not converge under frozen settings: '+c)
        p.positive_probability(model,role_X['fitting']);models[c]=model
        print(json.dumps(dict(stage='fit_'+c,seconds=time.perf_counter()-candidate_started)),flush=True)
    mark('four_candidate_fits',point)
    # All scored stand-in artifacts are discarded; only wall-clock timings
    # are retained. Verify the exact temporary cleanup target before entry.
    temporary=tempfile.TemporaryDirectory(prefix='discarded_timing_',dir=out)
    if Path(temporary.name).resolve().parent!=out.resolve():
        raise RuntimeError('Unexpected temporary cleanup target')
    with temporary as temporary_path:
        folder=Path(temporary_path)
        point=time.perf_counter()
        rows,evidence=selection(folder,role_X['fitting'],role_X['selection'],role_m['selection'],models,data.schema)
        mark('SAFE_and_selection_values',point)
        point=time.perf_counter();selected=select(rows,evidence,role_m['selection'],DISCARDED_REPEAT)
        mark('nine_arms_and_twelve_pair_fairness_checks',point)
        point=time.perf_counter()
        def progress(level,family):
            print(json.dumps(dict(stage='judge_'+level+'_'+family,elapsed_seconds=time.perf_counter()-started)),flush=True)
        judged=evaluate(role_X['judge'],role_X['fitting'],role_m['judge'],models,data.schema,FAMILIES,
            dataset_id='discarded_timing_standin',purpose='timing',retain_predictions=False,progress=progress)
        mark('three_judge_levels_secondary_endpoints_and_group_audits',point)
        point=time.perf_counter()
        repeat_table(rows,selected['choices'],judged,0,split_audit['sizes'])
        audit_for_choices(judged['J1']['group_audit'],selected['choices'],0)
        mark('per_repeat_reporting_preparation',point)
    total=time.perf_counter()-started
    result=dict(completed=True,report_policy='Time only; stand-in metrics, predictions, outcomes and fitted models discarded',
        seconds=total,stages_seconds=stages,projected_50_repeat_hours=50*total/3600,
        projected_under_12_hours=50*total<12*3600,records=50000,predictors=29,
        data_seed=DATA_SEED,split_seed=20261114+DISCARDED_REPEAT,bootstrap_seed=42+DISCARDED_REPEAT,
        judge_seed_scope='confirm_v1:discarded_timing_standin:<family>',candidate_random_state=42,
        candidate_seed_reason='Frozen configuration retained; data/split/bootstrap/judge streams are discarded timing streams',
        raw_Brazil_files_or_outcomes_loaded=False,Brazil_models_fitted=False,python_socket_operations_denied=True,
        limitation='One artificial-schema timing repeat; distribution-dependent time is not a guarantee for Brazil')
    (out/'timing.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':run(sys.argv[1])
