"""Template 7.3--7.6: fitting-only, type-specific paired judge faults."""
from decimal import Decimal
import numpy as np
from src.robustness_protocol import child_seed
from .schema import fitting_references


def references(fitting, schema):
    schema.validate(fitting)
    rules = {}
    for name in schema.order:
        col = fitting[name].to_numpy(dtype=float)
        if name in schema.numeric:
            precision = max(max(0, -Decimal(str(float(v))).normalize().as_tuple().exponent) for v in col)
            q1, q3 = np.quantile(col, [.25, .75], method='linear')
            rules[name] = dict(kind='numeric', precision=precision, iqr=float(q3-q1),
                               minimum=float(col.min()), maximum=float(col.max()))
        elif name in schema.ordered:
            allowed = set(schema.ordered[name]) | set(schema.special.get(name, ()))
            if not set(col) <= allowed:
                raise ValueError('Undocumented ordered code: '+name)
            rules[name] = dict(kind='ordered', levels=list(schema.ordered[name]),
                               special=list(schema.special.get(name, ())), observed=np.unique(col).tolist())
        else:
            levels, counts = np.unique(col, return_counts=True)
            rules[name] = dict(kind='unordered', levels=levels.tolist(), counts=counts.tolist())
    return dict(mask=fitting_references(fitting, schema), rules=rules)


def numeric_shift(values, signs, rule, q):
    """Round the shift and result at the maximum observed decimal precision."""
    if rule['iqr'] == 0:
        return values.copy(), 0
    shift = float(np.round(q*rule['iqr'], rule['precision']))
    if rule['precision'] == 0 and shift == 0:
        shift = 1.
    raw = np.round(values+signs*shift, rule['precision'])
    return np.clip(raw, rule['minimum'], rule['maximum']), int(np.count_nonzero(
        (raw < rule['minimum']) | (raw > rule['maximum'])))


def ordered_shift(values, signs, rule):
    output = values.copy()
    positions = {v:i for i,v in enumerate(rule['levels'])}
    observed = set(rule['observed'])
    absent = 0
    for i, (value, sign) in enumerate(zip(values, signs)):
        if value in rule['special']:
            continue
        if value not in positions:
            raise ValueError('Undocumented ordered evaluation code')
        target = positions[value]+int(sign)
        if 0 <= target < len(rule['levels']):
            level = rule['levels'][target]
            if level in observed:
                output[i] = level
            else:
                absent += 1
    return output, absent


def unordered_shift(values, uniforms, rule):
    output = values.copy()
    levels = np.asarray(rule['levels'])
    counts = np.asarray(rule['counts'], dtype=float)
    if len(levels) == 1:
        return output
    for value in np.unique(values):
        where = values == value
        eligible = levels != value
        probabilities = counts[eligible]/counts[eligible].sum()
        slots = np.searchsorted(np.cumsum(probabilities), uniforms[where], side='right')
        output[where] = levels[eligible][np.minimum(slots, eligible.sum()-1)]
    return output


def paired_transforms(frame, fitting, schema, families, *, dataset_id, family, repetition, q=.10, reference=None):
    schema.validate(frame)
    if q != .10 or repetition not in range(1, 31) or family not in families:
        raise ValueError('Judge requires q=.10, realization 1..30 and a documented family')
    if len(families) != 3 or any(not fields or not set(fields) <= set(schema.order) for fields in families.values()):
        raise ValueError('Exactly three nonempty original-predictor families required')
    flat = [n for fields in families.values() for n in fields]
    if len(flat) != len(set(flat)):
        raise ValueError('Families must be disjoint')
    ref = references(fitting, schema) if reference is None else reference
    scope = f'confirm_v1:{dataset_id}:{family}'
    seed = child_seed(scope, repetition, 'mask')
    selected = np.random.default_rng(seed).permutation(len(frame))[:int(np.floor(q*len(frame)))]
    corrupt, masked = frame.copy(deep=True), frame.copy(deep=True)
    audits, streams = {}, {'scope':scope, 'master_seed':20261007, 'repetition':repetition, 'mask':seed}
    for name in schema.order:
        if name in families[family]:
            masked[name] = masked[name].astype(float)
            masked.iloc[selected, masked.columns.get_loc(name)] = ref['mask'][name]
            continue
        rule = ref['rules'][name]
        values = frame[name].to_numpy(dtype=float)[selected]
        signs_seed = child_seed(scope, repetition, 'sign:'+name)
        signs = np.random.default_rng(signs_seed).choice([-1, 1], len(frame))[selected]
        streams[name] = dict(sign=signs_seed)
        clipped = absent = 0
        if rule['kind'] == 'numeric':
            changed, clipped = numeric_shift(values, signs, rule, q)
        elif rule['kind'] == 'ordered':
            changed, absent = ordered_shift(values, signs, rule)
        else:
            category_seed = child_seed(scope, repetition, 'category:'+name)
            streams[name]['category'] = category_seed
            uniforms = np.random.default_rng(category_seed).random(len(frame))[selected]
            changed = unordered_shift(values, uniforms, rule)
        corrupt[name] = corrupt[name].astype(float)
        corrupt.iloc[selected, corrupt.columns.get_loc(name)] = changed
        observed = set(fitting[name])
        audits[name] = dict(selected_records=len(selected), changed_cells=int(np.count_nonzero(values != changed)),
            unchanged_by_rule=int(np.count_nonzero(values == changed)), clipped=clipped,
            ordered_destination_absent=absent, input_levels_absent_from_fitting=(
                int(sum(v not in observed for v in values)) if rule['kind'] != 'numeric' else 0))
    joint = corrupt.copy(deep=True)
    for name in families[family]:
        joint[name] = masked[name]
    frames = {'10':masked, '01':corrupt, '11':joint}
    summary = {}
    for mode, changed in frames.items():
        cells = changed.to_numpy() != frame.to_numpy()
        summary[mode] = dict(records_selected=len(selected), records_changed=int(cells.any(axis=1).sum()),
                             cells_changed=int(cells.sum()))
    return frames, dict(selected_positions=selected.tolist(), per_feature=audits, scenarios=summary), streams


def evaluate(frame, fitting, meta, models, schema, families, *, dataset_id):
    """Nine scenario loss means, preserving unknown sex in overall results.

    Does no fitting and writes no files. Call only after the external run gates.
    Repetitions are averaged as losses, never as probabilities.
    """
    from research.model_selection_v2.pilot import positive_probability
    from research.model_selection_v2.judge_core.metrics import losses
    from .tables import InclusivePredictionTable
    ref = references(fitting, schema)
    y = meta.default.to_numpy()
    masks = {'all': np.ones(len(meta), dtype=bool), 'female':meta.SEX.eq(2).to_numpy(),
             'male':meta.SEX.eq(1).to_numpy()}
    if len(y) != len(frame) or not np.isin(y, [0,1]).all() or any(not m.any() for m in masks.values()):
        raise ValueError('Invalid judge outcome alignment or empty sex group')
    clean = {c:positive_probability(m, frame) for c,m in models.items()}
    predictions = {c:{} for c in models}
    registry = []
    for family in families:
        for rep in range(1,31):
            frames, audit, streams = paired_transforms(frame, fitting, schema, families,
                dataset_id=dataset_id, family=family, repetition=rep, reference=ref)
            registry.append(dict(family=family, repetition=rep, audit=audit, streams=streams))
            for mode, changed in frames.items():
                key = family+'__'+{'10':'mask_only','01':'corruption_only','11':'joint'}[mode]
                for c,m in models.items():
                    predictions[c].setdefault(key,[]).append(positive_probability(m,changed))
    rows = []
    def measure(probability, mask):
        sub = meta.loc[mask]
        return losses(InclusivePredictionTable(tuple(sub.ID),sub.default.to_numpy(),sub.SEX.to_numpy(),
            tuple(sub.duplicate_group_id),probability[mask],'judge'))
    for c in models:
        for sex,mask in masks.items():
            scenario_brier, scenario_log = [], []
            for values in predictions[c].values():
                realization_losses = [measure(value,mask) for value in values]
                scenario_brier.append(float(np.mean([v['brier'] for v in realization_losses])))
                scenario_log.append(float(np.mean([v['log_loss'] for v in realization_losses])))
            baseline = measure(clean[c],mask)
            base,base_log = baseline['brier'],baseline['log_loss']
            primary = float(np.mean(scenario_brier))
            rows.append(dict(candidate=c, sex=sex, records=int(mask.sum()), clean_brier=base, clean_log_loss=base_log,
                primary=primary, stressed_log_loss=float(np.mean(scenario_log)),
                worst_scenario_brier=float(max(scenario_brier)), deterioration=primary-base))
    return dict(scores=rows, clean=clean, predictions=predictions, registry=registry)
