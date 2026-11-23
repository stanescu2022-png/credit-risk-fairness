"""Corrected paired summaries and deterministic headline flags; no prose verdicts."""
from itertools import combinations
import json
import numpy as np
import pandas as pd
from scipy.stats import t
from research.model_selection_v2.step1_v1.core import corrected as legacy_corrected
from research.model_selection_v2.pilot import CANDIDATES
from .arms import ARMS,STRICT_ARM,PAIR_CONTRASTS,summarize_pair_checks

ENDPOINTS=('judge_J1_primary','judge_J1_disturbed_brier','judge_J2_primary','judge_J2_disturbed_brier',
    'judge_J3_primary','judge_J3_disturbed_brier','judge_J1_stressed_auc','judge_J1_stressed_k',
    'judge_J1_deterioration','judge_J1_stressed_log_loss')


def corrected(values,rho,adjusted=False):
    values=list(values)
    if len(values)!=50:
        raise ValueError('Every corrected v2 summary requires all fifty repeats')
    valid=[v is not None and np.isfinite(v) for v in values]
    if not all(valid):
        return dict(estimate=None,lower=None,upper=None,repeats=50,defined_repeats=sum(valid),
            reason='undefined repeat metric; no repeats dropped',rho=float(rho),adjusted=adjusted)
    array=np.asarray(values,dtype=float)
    if np.all(array==0):
        return dict(estimate=0.,lower=0.,upper=0.,sample_sd=0.,corrected_se=0.,repeats=50,
            defined_repeats=50,rho=float(rho),adjusted=adjusted,reason='v2 all-zero contrast convention')
    return dict(legacy_corrected(array,rho,adjusted=adjusted),repeats=50,defined_repeats=50)


def minimum_equal_win_disagreements(rho,n=50):
    critical=float(t.ppf(.975,n-1))
    for k in range(1,n+1):
        # Scale-free unit wins: k values -1, n-k values 0.
        mean=-k/n
        variance=k*(n-k)/(n*(n-1))
        upper=mean+critical*np.sqrt((1/n+rho)*variance)
        if upper<0:return k
    return None


def match_counts(choice_rows):
    if len(choice_rows)!=50 or set(choice_rows.repeat)!=set(range(50)):
        raise ValueError('Fifty unique repeat choices required')
    matrix=pd.DataFrame(index=ARMS,columns=ARMS,dtype=int)
    pairs=[]
    for a in ARMS:
        for b in ARMS:
            matrix.loc[a,b]=int(choice_rows['choice_'+a].eq(choice_rows['choice_'+b]).sum())
    for a,b in combinations(ARMS,2):
        pairs.append(dict(arm_a=a,arm_b=b,matches=int(matrix.loc[a,b]),disagreements=50-int(matrix.loc[a,b])))
    return matrix.astype(int),pd.DataFrame(pairs)


def below(ci):return ci.get('upper') is not None and ci['upper']<0
def above(ci):return ci.get('lower') is not None and ci['lower']>0


def headline_flags(primary,sex_adjusted,stressed_auc,stressed_k,log_loss,*,plain_loss=None,k,rho):
    predicates=dict(group_harm=any(above(ci) for ci in sex_adjusted.values()),primary_above=above(primary),
        primary_includes_zero=primary.get('lower') is not None and primary.get('upper') is not None and primary['lower']<=0<=primary['upper'],
        primary_below=below(primary),auc_favours_safe=below(stressed_auc),k_favours_safe=below(stressed_k),
        log_loss_favours_safe=above(log_loss))
    if predicates['group_harm']:flag='group_harm'
    elif predicates['primary_above']:flag='safe_lower_brier'
    elif predicates['primary_includes_zero']:flag='inconclusive'
    elif not predicates['primary_below']:flag='unavailable_primary'
    elif predicates['auc_favours_safe']:flag='lower_brier_worse_auc'
    elif predicates['k_favours_safe']:flag='lower_brier_worse_k'
    elif predicates['log_loss_favours_safe']:flag='brier_log_loss_tradeoff'
    elif any(ci.get('lower') is None or ci.get('upper') is None for ci in (stressed_auc,stressed_k,log_loss)):
        flag='unavailable_secondary'
    else:flag='supports_one_dataset'
    threshold=minimum_equal_win_disagreements(rho)
    return dict(headline_flag=flag,predicates=predicates,k=int(k),repeats=50,
        minimum_equal_win_disagreements=threshold,below_arithmetic_disagreement_requirement=k<threshold,
        registered_primary_unchanged=True,reference_brier=.001,
        plain_loss_upper_below_reference=(plain_loss is not None and plain_loss.get('upper') is not None and plain_loss['upper']<.001),
        judged_better=flag=='supports_one_dataset')


def contrast(table,a,b,endpoint,rho,*,adjusted=False):
    values=[];k=wins=losses=0;clean_a=[];clean_b=[];safe_clean=[];base_endpoint=[]
    for repeat in range(50):
        frame=table[table.repeat.eq(repeat)].set_index('candidate')
        ca,cb=frame['choice_'+a].iloc[0],frame['choice_'+b].iloc[0]
        k+=int(ca!=cb)
        va,vb=frame.loc[ca,endpoint],frame.loc[cb,endpoint]
        delta=(0. if ca==cb else float(va-vb)) if pd.notna(va) and pd.notna(vb) else None
        values.append(delta)
        if delta is not None:
            wins+=int(delta<0);losses+=int(delta>0)
        base_endpoint.append(float(vb) if pd.notna(vb) else None)
        for target,candidate in ((clean_a,ca),(clean_b,cb),(safe_clean,frame.choice_safe.iloc[0])):
            target.append(float(frame.loc[candidate,'judge_J1_clean_brier']))
    ci=corrected(values,rho,adjusted=adjusted)
    base=float(np.mean(clean_b));safe_base=float(np.mean(safe_clean))
    brier_units=endpoint.endswith(('primary','disturbed_brier','deterioration','clean_brier'))
    return dict(contrast=a+'_minus_'+b,endpoint=endpoint,k=k,disagreements=k,
        favourable_difference_sign='positive' if endpoint.endswith(('stressed_auc','stressed_k')) else 'negative',
        units='Brier' if brier_units else ('AUC' if endpoint.endswith('stressed_auc') else 'K' if endpoint.endswith('stressed_k') else 'log_loss'),
        negative_differences=wins,positive_differences=losses,exact_zero_differences=sum(v==0 for v in values if v is not None),
        arithmetic_minimum_k=minimum_equal_win_disagreements(rho),**ci,
        chosen_clean_brier=float(np.mean(clean_a)),baseline_clean_brier=base,safe_clean_brier=safe_base,
        difference_as_share_of_baseline_clean_brier=ci['estimate']/base if brier_units and ci['estimate'] is not None and base>0 else None,
        difference_as_share_of_safe_clean_brier=ci['estimate']/safe_base if brier_units and ci['estimate'] is not None and safe_base>0 else None)


def validate_repeat_table(table):
    if len(table)!=200 or set(table.repeat)!=set(range(50)) or table.duplicated(['repeat','candidate']).any():
        raise ValueError('Exactly four candidates in each of fifty repeats required')
    for repeat,frame in table.groupby('repeat'):
        if set(frame.candidate)!=set(CANDIDATES):raise ValueError('Candidate pool changed')
        for arm in ARMS+(STRICT_ARM,):
            if frame['choice_'+arm].nunique()!=1 or frame['choice_'+arm].iloc[0] not in CANDIDATES:
                raise ValueError('Invalid repeated arm choice')
        for role in ('n_judge','n_selection','n_fitting'):
            if frame[role].nunique()!=1 or frame[role].iloc[0]<=0:raise ValueError('Invalid role sizes')


def summarize(table):
    validate_repeat_table(table)
    roles=table.drop_duplicates('repeat')
    rho=float(roles.n_judge.mean()/(roles.n_fitting.mean()+roles.n_selection.mean()))
    comparisons=[contrast(table,arm,'safe',endpoint,rho) for endpoint in ENDPOINTS for arm in ARMS+(STRICT_ARM,) if arm!='safe']
    comparisons.append(contrast(table,STRICT_ARM,'selector2','judge_J1_primary',rho))
    paired=[contrast(table,a,b,'judge_J1_primary',rho) for a,b in PAIR_CONTRASTS]
    sex={sex:contrast(table,'selector2','safe','judge_J1_'+sex+'_primary',rho,adjusted=True) for sex in ('female','male')}
    sex_unadjusted={sex:contrast(table,'selector2','safe','judge_J1_'+sex+'_primary',rho,adjusted=False) for sex in ('female','male')}
    def find(arm,endpoint):return next(r for r in comparisons if r['contrast']==arm+'_minus_safe' and r['endpoint']==endpoint)
    primary=find('selector2','judge_J1_primary')
    flags=headline_flags(primary,sex,find('selector2','judge_J1_stressed_auc'),find('selector2','judge_J1_stressed_k'),
        find('selector2','judge_J1_stressed_log_loss'),plain_loss=contrast(table,'selector2','plain_loss','judge_J1_primary',rho),k=primary['k'],rho=rho)
    choices=roles[['repeat']+['choice_'+a for a in ARMS+(STRICT_ARM,)]]
    matrix,pairs=match_counts(choices)
    fairness_summary=dict(repeats=50,
        frozen_changes_top=int(roles.choice_selector2.ne(roles.choice_selector2_before_check).sum()),
        stricter_changes_top=int(roles['choice_'+STRICT_ARM].ne(roles.choice_selector2_before_check).sum()),
        stricter_changes_frozen_choice=int(roles['choice_'+STRICT_ARM].ne(roles.choice_selector2).sum()),
        stricter_primary_cost=contrast(table,STRICT_ARM,'selector2','judge_J1_primary',rho))
    return dict(contrasts=pd.DataFrame(comparisons),range_measure_contrasts=pd.DataFrame(paired),
        pair_match_matrix=matrix,pair_matches=pairs,sex_adjusted=sex,sex_unadjusted=sex_unadjusted,
        fairness_summary=fairness_summary,headline_flags=flags,rho=rho)


def summarize_audits(records,rho):
    table=pd.DataFrame(records)
    result=[]
    for key,frame in table.groupby(['arm','condition','group','metric'],sort=False):
        if len(frame)!=50 or set(frame.repeat)!=set(range(50)):
            raise ValueError('Every audit cell needs all fifty repeats')
        result.append(dict(zip(('arm','condition','group','metric'),key),**corrected(frame.value.tolist(),rho)))
    return pd.DataFrame(result)


def operating_characteristics(records,table):
    """Pair precision/block frequency beside that pair's overall judge gain."""
    validate_repeat_table(table)
    summary=summarize_pair_checks(records)
    values=[]
    for row in summary.to_dict('records'):
        differences=[]
        for repeat in range(50):
            frame=table[table.repeat.eq(repeat)].set_index('candidate')
            differences.append(float(frame.loc[row['challenger'],'judge_J1_primary']-frame.loc[row['incumbent'],'judge_J1_primary']))
        row['overall_judge_primary_difference']=float(np.mean(differences));values.append(row)
    return pd.DataFrame(values)


def repeat_table(selection_rows,choices,judge_results,repeat,role_sizes):
    rows=[]
    for candidate in CANDIDATES:
        row=dict(candidate=candidate,repeat=repeat,**selection_rows.loc[candidate].to_dict(),
            **{'choice_'+a:c for a,c in choices.items()},**{'n_'+r:n for r,n in role_sizes.items()})
        for level,result in judge_results.items():
            for score in result['scores']:
                if score['candidate']!=candidate:continue
                prefix='judge_'+level+('_'+score['sex'] if score['sex']!='all' else '')+'_'
                row.update({prefix+k:v for k,v in score.items() if k not in ('candidate','sex')})
            for secondary in result.get('secondary',[]):
                if secondary['candidate']==candidate:
                    row.update({'judge_'+level+'_'+k:v for k,v in secondary.items() if k!='candidate'})
        rows.append(row)
    return pd.DataFrame(rows)


def print_flags(summary,stream=None):
    import sys
    stream=sys.stdout if stream is None else stream
    print(json.dumps(summary['headline_flags']),file=stream)
    for row in summary['range_measure_contrasts'].to_dict('records'):
        print(json.dumps(dict(contrast=row['contrast'],k=row['k'],repeats=50,
            arithmetic_minimum_k=row['arithmetic_minimum_k'])),file=stream)
