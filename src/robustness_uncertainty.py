"""Cached, paired whole-predictor-group uncertainty; never call model prediction."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from src.performance_evaluation import bootstrap_group_weights

FOCUS = ['mean_absolute_movement','flip_rate','delta_ROC-AUC','delta_Average Precision (AP)','delta_Brier Score']


class WeightedRanks:
    """Tie-aware weighted AUC/AP; cached ordering, identical sklearn estimands."""
    def __init__(self, y, p):
        self.order = np.argsort(p, kind='stable')
        self.y = np.asarray(y)[self.order]
        scores = np.asarray(p)[self.order]
        self.starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1])+1]

    def evaluate(self, weights):
        sorted_w = np.asarray(weights,dtype=float)[:,self.order]
        total = np.add.reduceat(sorted_w,self.starts,axis=1)
        positive = np.add.reduceat(sorted_w*self.y,self.starts,axis=1)
        negative = total-positive
        total_pos, total_neg = positive.sum(axis=1), negative.sum(axis=1)
        previous_neg = np.cumsum(negative,axis=1)-negative
        numerator = (positive*(previous_neg+.5*negative)).sum(axis=1)
        auc = np.divide(numerator,total_pos*total_neg,out=np.full(len(weights),np.nan),where=(total_pos*total_neg)>0)
        reverse_pos = total_pos[:,None]-(np.cumsum(positive,axis=1)-positive)
        reverse_total = total.sum(axis=1)[:,None]-(np.cumsum(total,axis=1)-total)
        precision = np.divide(reverse_pos,reverse_total,out=np.zeros_like(reverse_pos),where=reverse_total>0)
        ap = np.divide((positive*precision).sum(axis=1),total_pos,out=np.full(len(weights),np.nan),where=total_pos>0)
        ap[total_neg==0] = np.nan  # explicit one-class undefined convention for this study
        return auc,ap


def validate_weighted_ranks():
    rng = np.random.default_rng(822)
    rows=[]
    for fixture in range(20):
        y = rng.integers(0,2,60)
        p = rng.integers(0,6,60)/5 if fixture%2 else rng.random(60)
        w = rng.integers(0,5,60)
        auc,ap = WeightedRanks(y,p).evaluate(w[None,:])
        expected = [roc_auc_score(y,p,sample_weight=w),average_precision_score(y,p,sample_weight=w)]
        np.testing.assert_allclose([auc[0],ap[0]],expected,atol=2e-14,rtol=0)
        expanded = np.repeat(np.arange(len(y)),w)
        np.testing.assert_allclose([auc[0],ap[0]],[roc_auc_score(y[expanded],p[expanded]),average_precision_score(y[expanded],p[expanded])],atol=2e-14,rtol=0)
        rows.append({'fixture':fixture,'max_error':float(np.max(np.abs(np.array([auc[0],ap[0]])-expected)))})
    return pd.DataFrame(rows)


def focus_metrics(y, baseline, perturbed, weights):
    """Averages metrics over saved realizations, never averages probabilities."""
    predictions=np.atleast_2d(perturbed)
    w=np.asarray(weights,dtype=float)
    if w.sum()==0:
        return np.full(5,np.nan)
    base_auc,base_ap=WeightedRanks(y,baseline).evaluate(w[None,:])
    values=[]
    for p in predictions:
        auc,ap=WeightedRanks(y,p).evaluate(w[None,:])
        values.append([np.average(np.abs(p-baseline),weights=w),
                       np.average((p>.5)!=(baseline>.5),weights=w),
                       auc[0]-base_auc[0],ap[0]-base_ap[0],
                       np.average((p-y)**2-(baseline-y)**2,weights=w)])
    return np.mean(values,axis=0)


def paired_bootstrap(metadata, baseline, primary, repetitions=2000, seed=42):
    """primary: scenario -> {'probabilities': R x model x N, 'changed': R x N}."""
    y=metadata.default.to_numpy()
    models=list(baseline)
    baseline_ranks={k:WeightedRanks(y,p) for k,p in baseline.items()}
    prepared={}
    for scenario,data in primary.items():
        prepared[scenario]={}
        for j,kind in enumerate(models):
            predictions=data['probabilities'][:,j,:]
            prepared[scenario][kind]={'ranks':[WeightedRanks(y,p) for p in predictions],
                'absolute':np.abs(predictions-baseline[kind]),
                'flips':((predictions>.5)!=(baseline[kind]>.5)).astype(float),
                'brier':(predictions-y)**2-(baseline[kind]-y)**2}
    rng=np.random.default_rng(seed)
    rows=[]
    with threadpool_limits(limits=1):
        for start in range(0,repetitions,32):
            size=min(32,repetitions-start)
            weights=np.array([bootstrap_group_weights(metadata.duplicate_group_id,rng) for _ in range(size)],dtype=float)
            for sex in (0,1,2):
                w=weights if sex==0 else weights*(metadata.SEX.to_numpy()==sex)
                totals=w.sum(axis=1)
                denom=np.where(totals>0,totals,np.nan)
                base={k:baseline_ranks[k].evaluate(w) for k in models}
                for scenario,data in primary.items():
                    results={}
                    exposure=(w@data['changed'].astype(float).T).mean(axis=1)/denom
                    for kind in models:
                        part=prepared[scenario][kind]
                        ranks=[rank.evaluate(w) for rank in part['ranks']]
                        auc=np.mean([v[0] for v in ranks],axis=0)-base[kind][0]
                        ap=np.mean([v[1] for v in ranks],axis=0)-base[kind][1]
                        result=np.column_stack([(w@part['absolute'].T).mean(axis=1)/denom,
                            (w@part['flips'].T).mean(axis=1)/denom,auc,ap,
                            (w@part['brier'].T).mean(axis=1)/denom])
                        results[kind]=result
                    results['GBM_minus_LR']=results[models[1]]-results[models[0]]
                    for kind,result in results.items():
                        for i in range(size):
                            rows.append({'scenario':scenario,'SEX':sex,'model':kind,'bootstrap':start+i+1,
                                'sample_records':int(totals[i]), 'sample_defaults':int(w[i,y==1].sum()),
                                'exposure_rate':float(exposure[i]),
                                **dict(zip(FOCUS,result[i]))})
            if start%256==0 or start+size==repetitions:
                print(f'Paired whole-group bootstrap: {start+size}/{repetitions}',flush=True)
    replicates=pd.DataFrame(rows)
    intervals=[]
    for scenario,data in primary.items():
        for sex in (0,1,2):
            weights=np.ones(len(y)) if sex==0 else (metadata.SEX.to_numpy()==sex).astype(float)
            points={kind:focus_metrics(y,baseline[kind],data['probabilities'][:,j,:],weights) for j,kind in enumerate(models)}
            points['GBM_minus_LR']=points[models[1]]-points[models[0]]
            for kind,point in points.items():
                subset=replicates[(replicates.scenario==scenario)&(replicates.SEX==sex)&(replicates.model==kind)]
                for j,metric in enumerate(FOCUS):
                    values=subset[metric].to_numpy()
                    valid=values[np.isfinite(values)]
                    lower,upper=np.quantile(valid,[.025,.975]) if len(valid) else (np.nan,np.nan)
                    intervals.append({'scenario':scenario,'SEX':sex,'model':kind,'metric':metric,'estimate':point[j],
                        'ci_lower':lower,'ci_upper':upper,'attempted_replicates':repetitions,'valid_replicates':len(valid),
                        'undefined_replicates':repetitions-len(valid),
                        'undefined_reason':'empty or single-outcome SEX subset' if len(valid)<repetitions else '',
                        'method':'95% whole-predictor-group percentile; seed 42; conditional on frozen models and saved realizations'})
    return pd.DataFrame(intervals),replicates
