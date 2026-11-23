"""Nine registered arms, plus a separately labelled stricter-gate diagnostic."""
from itertools import permutations
import numpy as np
import pandas as pd
from research.model_selection_v2.pilot import CANDIDATES
from research.model_selection_v2.step1_v1.core import ranking,choice,fairness,group_means
from research.model_selection_v2.confirmation_v1.selection import bootstrap_plan

ARMS=('safe','selector2','selector2_before_check','plain_loss','order_mild','accuracy_full',
      'ae_only','clean_accuracy','true_retention')
STRICT_ARM='selector2_stricter_gate'
PAIR_CONTRASTS=(('order_mild','safe'),('selector2_before_check','accuracy_full'),
    ('selector2_before_check','order_mild'),('accuracy_full','safe'),('true_retention','clean_accuracy'))


def diagnostic_scores(rows,evidence):
    result={name:{} for name in ('ae_only','clean_accuracy','true_retention')}
    for candidate in CANDIDATES:
        row=rows.loc[candidate]
        mild=np.asarray(evidence[candidate]['S_mild'],dtype=float)
        if mild.shape!=(6,) or not np.isfinite(mild).all():
            raise ValueError('Six finite mild-grid skill values required')
        np.testing.assert_allclose(mild.mean(),row.S_bar,rtol=0,atol=1e-12)
        s0=mild[0];retention=float(np.mean(mild[1:]-s0))
        np.testing.assert_allclose(retention,6/5*(row.S_bar-s0),rtol=0,atol=1e-12)
        result['ae_only'][candidate]=float((row.A_bar+row.E_bar)/2)
        result['clean_accuracy'][candidate]=float((row.A_bar+row.E_bar+s0)/3)
        result['true_retention'][candidate]=float((row.A_bar+row.E_bar+retention)/3)
    return result


def pair_checks(point,draws,*,ranked,repeat):
    """Joint over the two sexes within each pair, not over all twelve pairs."""
    point,draws=np.asarray(point),np.asarray(draws)
    if point.shape!=(4,2,3) or draws.shape!=(2000,4,2,3):
        raise ValueError('Expected frozen 2000-draw four-candidate bootstrap layout')
    output=[]
    for challenger,incumbent in permutations(CANDIDATES,2):
        i,j=CANDIDATES.index(challenger),CANDIDATES.index(incumbent)
        check=fairness(point[i,:,1]-point[j,:,1],draws[:,i,:,1]-draws[:,j,:,1])
        intervals=check['joint_intervals_female_male']
        triggers=[sex for sex,(lo,hi) in zip(('female','male'),intervals) if lo is not None and lo>0]
        strict_triggers=[sex for sex,(lo,hi) in zip(('female','male'),intervals) if hi is None or not hi<.0005]
        half=[None if lo is None or hi is None else float((hi-lo)/2) for lo,hi in intervals]
        output.append(dict(repeat=repeat,challenger=challenger,incumbent=incumbent,rank=ranked.index(challenger)+1,
            **check,gate_result='blocked' if check['blocked'] else 'not_blocked',triggering_sex=triggers,
            joint_half_width_female_male=half,stricter_blocked=bool(strict_triggers),stricter_triggering_sex=strict_triggers,
            stricter_margin=.0005,precision_interpretation='conditional interval half-width, not detectable-harm power'))
    return output


def select(rows,evidence,meta,repeat):
    safe=ranking(rows.SAFE_score.to_dict())[0]
    ranked=ranking(rows.Score2.to_dict())
    plan=bootstrap_plan(meta,repeat)
    point=np.empty((4,2,3));draws=np.empty((2000,4,2,3))
    y=meta.default.to_numpy()
    for i,c in enumerate(CANDIDATES):
        clean=(evidence[c]['clean']-y)**2
        stress=((evidence[c]['swap_mild'][1:]-y)**2).mean(axis=0)
        values=np.column_stack((clean,stress,stress-clean))
        for j,sex in enumerate((2,1)):
            mask=meta.SEX.to_numpy()==sex
            if not mask.any():raise ValueError('Empty sex subgroup')
            point[i,j]=values[mask].mean(axis=0)
            draws[:,i,j]=group_means(values,plan.group_multiplicities,plan.row_group_positions,mask)
    checks=pair_checks(point,draws,ranked=ranked,repeat=repeat)
    by_pair={(r['challenger'],r['incumbent']):r for r in checks}
    frozen_block={c:(by_pair[c,safe]['blocked'] if c!=safe else False) for c in CANDIDATES}
    strict_block={c:(by_pair[c,safe]['stricter_blocked'] if c!=safe else False) for c in CANDIDATES}
    selected=choice(ranked,safe,frozen_block)
    strict=choice(ranked,safe,strict_block)
    scores=diagnostic_scores(rows,evidence)
    arms=dict(safe=safe,selector2=selected,selector2_before_check=ranked[0],
        plain_loss=ranking((-rows.clean_brier).to_dict())[0],order_mild=ranking(rows.order_mild_score.to_dict())[0],
        accuracy_full=ranking(rows.accuracy_full_score.to_dict())[0],
        **{arm:ranking(values)[0] for arm,values in scores.items()})
    arms[STRICT_ARM]=strict
    visited=[]
    for c in ranked:
        visited.append(c)
        if c==selected:break
    assessed=[dict(r,visited_in_choice_rule=r['challenger'] in visited,
        safe_incumbent=True) for r in checks if r['incumbent']==safe]
    return dict(choices=arms,diagnostic_scores=scores,pair_checks=checks,challenger_log=assessed,
        stricter_binds=strict!=ranked[0],stricter_changes_frozen_choice=strict!=selected,
        bootstrap_point=point,bootstrap_draws=draws)


def summarize_pair_checks(records):
    """Block frequencies and precision by ordered pair over all 50 repeats."""
    output=[]
    for challenger,incumbent in permutations(CANDIDATES,2):
        selected=[r for r in records if r['challenger']==challenger and r['incumbent']==incumbent]
        if len(selected)!=50 or {r['repeat'] for r in selected}!=set(range(50)):
            raise ValueError('Exactly fifty records per ordered pair required')
        row=dict(challenger=challenger,incumbent=incumbent,repeats=50,
            blocks=sum(r['blocked'] for r in selected),stricter_blocks=sum(r['stricter_blocked'] for r in selected))
        row.update(block_frequency=row['blocks']/50,stricter_block_frequency=row['stricter_blocks']/50)
        for j,sex in enumerate(('female','male')):
            half=[r['joint_half_width_female_male'][j] for r in selected]
            valid=[v for v in half if v is not None]
            row[sex+'_defined_half_widths']=len(valid)
            row[sex+'_mean_half_width']=float(np.mean(valid)) if valid else None
            # Paired point is reported beside precision; an overall judge gain
            # is separately supplied by the reporting table, not inferred here.
            row[sex+'_mean_stressed_brier_difference']=float(np.mean([r['point_female_male'][j] for r in selected]))
        output.append(row)
    return pd.DataFrame(output)
