"""Nested severity ladder; J1 preserves confirmation_v1 without modification."""
import numpy as np
from sklearn.metrics import roc_auc_score
from src.robustness_protocol import child_seed
from research.model_selection_v2.confirmation_v1 import judge as v1
from research.model_selection_v2.confirmation_v1.tables import InclusivePredictionTable
from research.model_selection_v2.judge_core.metrics import losses
from research.model_selection_v2.pilot import positive_probability
from research.model_selection_v2.safe2026 import rank_graduation

LEVELS = {'J1':(.10,.10,1), 'J2':(.25,.25,1), 'J3':(.50,.50,2)}
MODES = {'10':'mask_only','01':'corruption_only','11':'joint'}


def ordered_shift(values, signs, rule, steps):
    if steps == 1:
        return v1.ordered_shift(values,signs,rule)
    output = values.copy()
    positions = {v:i for i,v in enumerate(rule['levels'])}
    observed = set(rule['observed'])
    absent = 0
    for i,(value,sign) in enumerate(zip(values,signs)):
        if value in rule['special']:
            continue
        if value not in positions:
            raise ValueError('Undocumented ordered evaluation code')
        target = positions[value]+steps*int(sign)
        if 0 <= target < len(rule['levels']):
            level = rule['levels'][target]
            if level in observed:
                output[i] = level
            else:
                absent += 1
    return output,absent


def paired_transforms(frame,fitting,schema,families,*,dataset_id,family,repetition,level='J1',reference=None):
    if level not in LEVELS:
        raise ValueError('Unknown severity level')
    if level == 'J1':
        return v1.paired_transforms(frame,fitting,schema,families,dataset_id=dataset_id,
            family=family,repetition=repetition,q=.10,reference=reference)
    schema.validate(frame)
    schema.validate(fitting)
    if repetition not in range(1,31) or family not in families or len(families)!=3:
        raise ValueError('Three families and realization 1..30 required')
    flat = [n for fields in families.values() for n in fields]
    if any(not fields for fields in families.values()) or len(flat)!=len(set(flat)) or not set(flat)<=set(schema.order):
        raise ValueError('Invalid or overlapping original-predictor families')
    share,shift_scale,steps = LEVELS[level]
    ref = v1.references(fitting,schema) if reference is None else reference
    scope = f'confirm_v1:{dataset_id}:{family}'
    seed = child_seed(scope,repetition,'mask')
    selected = np.random.default_rng(seed).permutation(len(frame))[:int(np.floor(share*len(frame)))]
    corrupt,masked = frame.copy(deep=True),frame.copy(deep=True)
    audits,streams = {},dict(scope=scope,master_seed=20261007,repetition=repetition,mask=seed)
    for name in schema.order:
        if name in families[family]:
            masked[name] = masked[name].astype(float)
            masked.iloc[selected,masked.columns.get_loc(name)] = ref['mask'][name]
            continue
        rule = ref['rules'][name]
        values = frame[name].to_numpy(dtype=float)[selected]
        sign_seed = child_seed(scope,repetition,'sign:'+name)
        signs = np.random.default_rng(sign_seed).choice([-1,1],len(frame))[selected]
        streams[name] = dict(sign=sign_seed)
        clipped = absent = 0
        if rule['kind']=='numeric':
            changed,clipped = v1.numeric_shift(values,signs,rule,shift_scale)
        elif rule['kind']=='ordered':
            changed,absent = ordered_shift(values,signs,rule,steps)
        else:
            category_seed = child_seed(scope,repetition,'category:'+name)
            streams[name]['category'] = category_seed
            uniforms = np.random.default_rng(category_seed).random(len(frame))[selected]
            changed = v1.unordered_shift(values,uniforms,rule)
        corrupt[name] = corrupt[name].astype(float)
        corrupt.iloc[selected,corrupt.columns.get_loc(name)] = changed
        observed = set(fitting[name])
        audits[name] = dict(selected_records=len(selected),changed_cells=int(np.count_nonzero(values!=changed)),
            unchanged_by_rule=int(np.count_nonzero(values==changed)),clipped=clipped,
            ordered_destination_absent=absent,input_levels_absent_from_fitting=(
                int(sum(v not in observed for v in values)) if rule['kind']!='numeric' else 0))
    joint = corrupt.copy(deep=True)
    for name in families[family]:
        joint[name] = masked[name]
    frames = {'10':masked,'01':corrupt,'11':joint}
    summary = {}
    for mode,changed in frames.items():
        cells = changed.to_numpy()!=frame.to_numpy()
        summary[mode] = dict(records_selected=len(selected),records_changed=int(cells.any(axis=1).sum()),
            cells_changed=int(cells.sum()))
    return frames,dict(selected_positions=selected.tolist(),per_feature=audits,scenarios=summary),streams


def measure(meta,probability,mask):
    sub = meta.loc[mask]
    return losses(InclusivePredictionTable(tuple(sub.ID),sub.default.to_numpy(),sub.SEX.to_numpy(),
        tuple(sub.duplicate_group_id),np.asarray(probability)[mask],'judge'))


def auc(y,probability):
    return float(roc_auc_score(y,probability)) if len(set(y))==2 else None


def mean_defined(values):
    """Never drop an undefined realization/subgroup to make a mean available."""
    return float(np.mean(values)) if values and all(v is not None and np.isfinite(v) for v in values) else None


def legacy_scores(meta,clean,predictions):
    masks = {'all':np.ones(len(meta),dtype=bool),'female':meta.SEX.eq(2).to_numpy(),'male':meta.SEX.eq(1).to_numpy()}
    if any(not m.any() for m in masks.values()):
        raise ValueError('Empty audited sex group')
    rows = []
    for c in clean:
        for sex,mask in masks.items():
            scenario_brier,scenario_log = [],[]
            for values in predictions[c].values():
                results = [measure(meta,value,mask) for value in values]
                scenario_brier.append(float(np.mean([r['brier'] for r in results])))
                scenario_log.append(float(np.mean([r['log_loss'] for r in results])))
            base = measure(meta,clean[c],mask)
            primary = float(np.mean(scenario_brier))
            rows.append(dict(candidate=c,sex=sex,records=int(mask.sum()),clean_brier=base['brier'],
                clean_log_loss=base['log_loss'],primary=primary,stressed_log_loss=float(np.mean(scenario_log)),
                worst_scenario_brier=float(max(scenario_brier)),deterioration=primary-base['brier']))
    return rows


def secondary_endpoints(meta,clean,predictions,registry,*,rank_measures):
    selected = {(r['family'],r['repetition']):r['audit']['selected_positions'] for r in registry}
    y = meta.default.to_numpy()
    rows = []
    for c in clean:
        disturbed,aucs,ks = [],[],[]
        for scenario,values in predictions[c].items():
            family = scenario.rsplit('__',1)[0]
            bs,aa,kk = [],[],[]
            for rep,probability in enumerate(values,1):
                chosen = np.asarray(selected[family,rep],dtype=int)
                if len(chosen)==0:
                    raise ValueError('No selected judge rows')
                bs.append(float(np.mean((probability[chosen]-y[chosen])**2)))
                if rank_measures:
                    aa.append(auc(y,probability))
                    kk.append(rank_graduation(clean[c],probability).value)
            disturbed.append(mean_defined(bs))
            if rank_measures:
                aucs.append(mean_defined(aa));ks.append(mean_defined(kk))
        row = dict(candidate=c,disturbed_brier=mean_defined(disturbed))
        if rank_measures:
            row.update(stressed_auc=mean_defined(aucs),stressed_k=mean_defined(ks))
        rows.append(row)
    return rows


def evaluate(frame,fitting,meta,models,schema,families,*,dataset_id,levels=('J1','J2','J3'),
             secondary=True,group_audit=True,purpose,retain_predictions=True,progress=None):
    """No fitting. Taiwan may be used only for the J1 legacy equivalence gate."""
    levels = tuple(levels)
    if not levels or len(levels)!=len(set(levels)) or not set(levels)<=set(LEVELS):
        raise ValueError('Invalid level list')
    if purpose not in ('equivalence','fixture','timing','registered'):
        raise ValueError('Explicit evaluation purpose required')
    if purpose=='equivalence' and (levels!=('J1',) or secondary or group_audit):
        raise ValueError('Equivalence permits J1 legacy outputs only')
    if dataset_id.startswith('taiwan') and purpose!='equivalence':
        raise ValueError('Taiwan secondary computation is locked pending document freeze/tag')
    schema.validate(frame);schema.validate(fitting)
    if len(meta)!=len(frame) or not np.isin(meta.default,[0,1]).all():
        raise ValueError('Invalid judge metadata')
    ref = v1.references(fitting,schema)
    clean = {c:positive_probability(m,frame) for c,m in models.items()}
    output = {}
    for level in levels:
        predictions = {c:{} for c in models}
        registry = []
        for family in families:
            for rep in range(1,31):
                frames,audit,streams = paired_transforms(frame,fitting,schema,families,dataset_id=dataset_id,
                    family=family,repetition=rep,level=level,reference=ref)
                registry.append(dict(family=family,repetition=rep,audit=audit,streams=streams))
                for mode,changed in frames.items():
                    key = family+'__'+MODES[mode]
                    for c,m in models.items():
                        predictions[c].setdefault(key,[]).append(positive_probability(m,changed))
            if progress is not None:
                progress(level,family)
        result = dict(scores=legacy_scores(meta,clean,predictions),clean=clean,predictions=predictions,registry=registry)
        if secondary:
            result['secondary'] = secondary_endpoints(meta,clean,predictions,registry,rank_measures=(level=='J1'))
        if group_audit and level=='J1':
            from .audit import candidate_audit
            result['group_audit'] = candidate_audit(meta,frame.AGE.to_numpy(),clean,predictions)
        if not retain_predictions:
            result.pop('predictions')
        output[level] = result
    return output
