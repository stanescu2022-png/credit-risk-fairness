"""One population-wide approval cutoff and descriptive sex/age group audits."""
import numpy as np
from .judge import auc,mean_defined

METRICS=('brier','auc','predicted_minus_observed','approval_rate','defaulters_approved','nondefaulters_refused')
AGE_BANDS=('age_under30','age30_49','age50plus')


def approvals(probability,ids):
    probability=np.asarray(probability,dtype=float);ids=np.asarray(ids)
    if probability.ndim!=1 or len(probability)!=len(ids) or not len(ids) or len(set(ids))!=len(ids):
        raise ValueError('Approval cutoff needs aligned probabilities and unique record IDs')
    if not np.isfinite(probability).all() or np.any((probability<0)|(probability>1)):
        raise ValueError('Invalid approval probabilities')
    order=np.lexsort((ids,probability))
    approved=np.zeros(len(ids),dtype=bool)
    approved[order[:int(np.ceil(.75*len(ids)))]]=True
    return approved


def group_masks(meta,age):
    age=np.asarray(age,dtype=float)
    if age.shape!=(len(meta),) or not np.isfinite(age).all() or not meta.SEX.isin([0,1,2]).all():
        raise ValueError('Invalid recorded age or sex metadata')
    groups={'all':np.ones(len(meta),dtype=bool),'female':meta.SEX.eq(2).to_numpy(),'male':meta.SEX.eq(1).to_numpy()}
    bands=dict(zip(AGE_BANDS,(age<30,(age>=30)&(age<50),age>=50)))
    groups.update(bands)
    for sex in ('female','male'):
        for band,mask in bands.items():groups[sex+'__'+band]=groups[sex]&mask
    return groups


def one_prediction_audit(meta,age,probability):
    probability=np.asarray(probability,dtype=float)
    approved=approvals(probability,meta.ID.to_numpy())
    y=meta.default.to_numpy()
    groups=group_masks(meta,age)
    result={}
    for group,mask in groups.items():
        yg,pg,ag=y[mask],probability[mask],approved[mask]
        n,events=len(yg),int(yg.sum())
        result[group]=dict(records=n,defaults=events,nondefaults=n-events,
            brier=float(np.mean((pg-yg)**2)) if n else None,
            auc=auc(yg,pg),predicted_minus_observed=float(np.mean(pg-yg)) if n else None,
            approval_rate=float(ag.mean()) if n else None,
            defaulters_approved=float(ag[yg==1].mean()) if events else None,
            nondefaulters_refused=float((~ag[yg==0]).mean()) if n-events else None)
    return result


def candidate_audit(meta,age,clean,predictions):
    """Average metrics per realization then per scenario, never probabilities.

    Age and sex are the original recorded metadata, including when AGE is
    corrupted as a predictor. Unknown sex is retained in all and age-only groups.
    """
    rows=[]
    for candidate,probability in clean.items():
        base=one_prediction_audit(meta,age,probability)
        conditions={'clean':base}
        scenario_means=[]
        for values in predictions[candidate].values():
            draws=[one_prediction_audit(meta,age,p) for p in values]
            scenario_means.append({g:{m:mean_defined([a[g][m] for a in draws]) for m in METRICS} for g in base})
        conditions['J1']={g:{**{k:base[g][k] for k in ('records','defaults','nondefaults')},
            **{m:mean_defined([a[g][m] for a in scenario_means]) for m in METRICS}} for g in base}
        for condition,audits in conditions.items():
            for group,values in audits.items():
                for metric in METRICS:
                    rows.append(dict(candidate=candidate,condition=condition,group=group,metric=metric,value=values[metric],
                        records=values['records'],defaults=values['defaults'],nondefaults=values['nondefaults']))
            for suffix in ('',)+tuple('__'+b for b in AGE_BANDS):
                for metric in METRICS:
                    female,male=audits['female'+suffix][metric],audits['male'+suffix][metric]
                    rows.append(dict(candidate=candidate,condition=condition,group='female_minus_male'+suffix,
                        metric=metric,value=None if female is None or male is None else female-male))
    return rows


def audit_for_choices(candidate_rows,choices,repeat):
    rows=[]
    for arm,candidate in choices.items():
        rows.extend(dict(r,repeat=repeat,arm=arm) for r in candidate_rows if r['candidate']==candidate)
    lookup={(r['candidate'],r['condition'],r['group'],r['metric']):r['value'] for r in candidate_rows}
    for condition in ('clean','J1'):
        for suffix in ('',)+tuple('__'+b for b in AGE_BANDS):
            group='female_minus_male'+suffix
            for metric in METRICS:
                a=lookup[choices['selector2'],condition,group,metric]
                b=lookup[choices['safe'],condition,group,metric]
                rows.append(dict(repeat=repeat,arm='selector2_minus_safe_gap_change',condition=condition,group=group,
                    metric=metric,value=None if a is None or b is None else a-b))
    return rows
