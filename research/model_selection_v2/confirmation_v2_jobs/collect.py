"""All-50-repeat primary and deferred reports, with margin-gate naming."""
from pathlib import Path
import pandas as pd
from research.model_selection_v2.confirmation_v2 import reporting as prior_reporting
from .artifacts import read_json,write_json,verify
from .names import primary_summary,summarize,rename,legacy_table,legacy_pair_records


def persist(folder,result):
    for name,value in result.items():
        if isinstance(value,pd.DataFrame):value.to_csv(folder/(name+'.csv'),index=True if name=='pair_match_matrix' else False)
        else:write_json(folder/(name+'.json'),value)


def job_A(repeat_folders,output):
    folders=[Path(p) for p in repeat_folders]
    for p in folders:verify(p)
    table=pd.concat([pd.read_csv(p/'per_repeat_primary.csv',float_precision='round_trip') for p in folders],ignore_index=True)
    result=primary_summary(table)
    output=Path(output);output.mkdir(exist_ok=False)
    table.to_csv(output/'per_repeat_primary.csv',index=False);persist(output,result)
    return result


def job_B(repeat_folders,output):
    folders=[Path(p) for p in repeat_folders]
    for p in folders:verify(p)
    table=pd.concat([pd.read_csv(p/'per_repeat_full.csv',float_precision='round_trip') for p in folders],ignore_index=True)
    result=summarize(table)
    pairs=[r for p in folders for r in read_json(p/'every_pair_gate_records.json')]
    # Private compatibility conversion only; all returned/persisted names use
    # the amendment's margin-gate wording.
    old_pairs=legacy_pair_records(pairs)
    result['operating_characteristics']=rename(prior_reporting.operating_characteristics(old_pairs,legacy_table(table)))
    audits=[r for p in folders for r in read_json(p/'group_audit_by_arm.json')]
    result['group_audit_intervals']=prior_reporting.summarize_audits(audits,result['rho'])
    output=Path(output);output.mkdir(exist_ok=False)
    table.to_csv(output/'per_repeat_full.csv',index=False);persist(output,result)
    return result
