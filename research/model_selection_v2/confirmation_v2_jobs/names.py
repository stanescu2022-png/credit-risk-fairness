"""Public margin-gate names; private adapters preserve historical v2 inputs."""
import pandas as pd
from research.model_selection_v2.confirmation_v2 import arms as historical
from research.model_selection_v2.confirmation_v2 import reporting as historical_reporting

MARGIN_ARM='selector2_margin_gate'
MARGIN_LABEL='margin gate (diagnostic)'
ARMS=historical.ARMS
KEY_NAMES={'stricter_margin':'harm_margin'}


def rename(value):
    if isinstance(value,str):return KEY_NAMES.get(value,value.replace('stricter','margin'))
    if isinstance(value,dict):return {rename(k):rename(v) for k,v in value.items()}
    if isinstance(value,list):return [rename(v) for v in value]
    if isinstance(value,tuple):return tuple(rename(v) for v in value)
    if isinstance(value,pd.DataFrame):
        result=value.copy()
        result.columns=[rename(v) for v in result.columns]
        result.index=[rename(v) for v in result.index]
        for column in result.select_dtypes(include='object'):
            result[column]=result[column].map(rename)
        return result
    return value


def legacy_table(table):
    return table.rename(columns={'choice_'+MARGIN_ARM:'choice_'+historical.STRICT_ARM})


def legacy_pair_records(records):
    reverse={v:k for k,v in KEY_NAMES.items()}
    return [{reverse.get(k,k.replace('margin','stricter')):v for k,v in row.items()} for row in records]


def summarize(table):return rename(historical_reporting.summarize(legacy_table(table)))


def primary_summary(table):
    historical_reporting.validate_repeat_table(legacy_table(table))
    old=legacy_table(table)
    roles=old.drop_duplicates('repeat')
    rho=float(roles.n_judge.mean()/(roles.n_fitting.mean()+roles.n_selection.mean()))
    contrasts=[historical_reporting.contrast(old,a,'safe','judge_J1_primary',rho)
        for a in ARMS+(historical.STRICT_ARM,) if a!='safe']
    ranges=[historical_reporting.contrast(old,a,b,'judge_J1_primary',rho) for a,b in historical.PAIR_CONTRASTS]
    ordinary={s:historical_reporting.contrast(old,'selector2','safe','judge_J1_'+s+'_primary',rho)
        for s in ('female','male')}
    adjusted={s:historical_reporting.contrast(old,'selector2','safe','judge_J1_'+s+'_primary',rho,adjusted=True)
        for s in ('female','male')}
    matrix,pairs=historical_reporting.match_counts(roles)
    return rename(dict(primary_contrasts=pd.DataFrame(contrasts),range_measure_contrasts=pd.DataFrame(ranges),
        sex_ordinary=ordinary,sex_adjusted=adjusted,pair_match_matrix=matrix,pair_matches=pairs,rho=rho,
        headline_flag='awaiting_job_B_secondary_endpoints'))
