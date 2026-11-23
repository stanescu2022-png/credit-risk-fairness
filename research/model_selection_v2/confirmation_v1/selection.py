"""Generalized Step 1 selection and split, derived from the frozen runner.

Only predictor-dependent expressions, record count and bootstrap metadata adapter
are changed. Original files are imported or read, never modified.
"""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from research.model_selection_v2 import pilot as p, safe2026 as s
from research.model_selection_v2.step1_v1.runner import write, npz, csv
from research.model_selection_v2.step1_v1.core import MILD, skill, score, ranking, choice, group_means, fairness, percentile
from .schema import fitting_references, full_grid, accuracy_curve


def bootstrap_plan(meta, repeat):
    # Same whole-group draw algorithm as judge_core.bootstrap; permits unknown
    # sex in the overall sampling frame without assigning it to either subgroup.
    groups = tuple(dict.fromkeys(meta.duplicate_group_id))
    lookup = {g:i for i,g in enumerate(groups)}
    sampled = np.random.default_rng(42+repeat).integers(0, len(groups), size=(2000, len(groups)))
    counts = np.asarray([np.bincount(d, minlength=len(groups)) for d in sampled], dtype=np.int64)
    return SimpleNamespace(group_order=groups, group_multiplicities=counts,
        row_group_positions=np.array([lookup[g] for g in meta.duplicate_group_id]))

def split(meta, repeat, historical_ids):
    groups = np.asarray(list(dict.fromkeys(meta.duplicate_group_id)))
    perm = np.random.default_rng(20261114+repeat).permutation(groups)
    sizes = meta.duplicate_group_id.value_counts().to_dict()
    assigned, counts = {}, {'judge': 0, 'selection': 0, 'fitting': 0}
    for g in perm:
        role = 'judge' if counts['judge'] < .20*len(meta) else (
            'selection' if counts['selection'] < .16*len(meta) else 'fitting')
        assigned[g] = role
        counts[role] += sizes[g]
    out = meta.copy()
    out['role'] = out.duplicate_group_id.map(assigned)
    p.require(not set(out.ID)&historical_ids and len(out) == len(meta) and out.ID.is_unique,
              'Repeat historical-ID or partition gate failed')
    p.require(out.groupby('duplicate_group_id').role.nunique().max() == 1, 'Group crosses roles')
    positions, strata = {}, []
    for role in ('fitting', 'selection', 'judge'):
        positions[role] = np.flatnonzero(out.role.eq(role))
        for sex in (2, 1):
            sub = out[out.role.eq(role)&out.SEX.eq(sex)]
            p.require(set(sub.default) == {0, 1}, f'Invalid {role} sex {sex} outcome support')
            strata.append({'role': role, 'SEX': sex, 'records': len(sub), 'defaults': int(sub.default.sum())})
    p.require(sum(map(len, positions.values())) == len(out), 'Incomplete partition')
    return out, positions, {'seed': 20261114+repeat, 'sizes': counts, 'sex_counts': strata,
            'group_order_before_permutation': 'first occurrence in original development order',
            'threshold_rule': 'add whole groups until judge >=20% and selection >=16% of the cohort',
            'passed': True, 'historical_ID_overlap': 0, 'crossing_groups': 0}


def selection(folder, Xfit, Xsel, meta, models, schema, *, mild=True):
    schema.validate(Xfit)
    schema.validate(Xsel)
    full = {c: p.positive_probability(models[c], Xsel) for c in p.CANDIDATES}
    references = fitting_references(Xfit, schema)
    write(folder/'masking_references.json', references)
    swap = {c: [] for c in p.CANDIDATES}
    mildp = {c: [] for c in p.CANDIDATES}
    exposure = []
    for strength in full_grid(schema):
        changed, audit = s.tail_swap(Xsel, meta.ID, float(strength))
        exposure.append(audit)
        for c in p.CANDIDATES:
            swap[c].append(p.positive_probability(models[c], changed))
    if mild:
        for strength in MILD:
            changed, audit = s.tail_swap(Xsel, meta.ID, float(strength))
            for c in p.CANDIDATES:
                mildp[c].append(p.positive_probability(models[c], changed))
    write(folder/'swap_exposure.json', exposure)
    rows, evidence = [], {}
    for c in p.CANDIDATES:
        a = accuracy_curve(meta.default, full[c], schema)
        p.require(a['status'] == 'defined', 'Undefined accuracy curve')
        removal = s.greedy_preservation(Xsel.columns, full[c],
                     lambda removed: p.positive_probability(models[c], s.mask_original(Xsel, removed, references)))
        p.require(removal['status'] == 'defined' and len(removal['alternatives']) == len(schema.order)*(len(schema.order)+1)//2,
                  f'Undefined/incomplete greedy path: {c}')
        q = np.asarray(swap[c])
        r = np.array([s.rank_graduation(full[c], v).value for v in q], dtype=float)
        p.require(np.isfinite(r).all(), 'Undefined rank preservation')
        integrated = s.integrate(a['values'], removal['values'], r)
        p.require(integrated['status'] == 'defined', 'Undefined SAFE score')
        av, ev, rv = np.mean(a['values']), np.mean(removal['values']), r.mean()
        row = {'candidate': c, 'A_bar': float(av), 'E_bar': float(ev), 'R_safe_bar': float(rv),
               'SAFE_score': integrated['arithmetic'], 'clean_brier': float(np.mean((full[c]-meta.default)**2)),
               'S_full_bar': float(skill(meta.default, q).mean())}
        if mild:
            qm = np.asarray(mildp[c])
            rm = np.r_[1., [s.rank_graduation(full[c], v).value for v in qm[1:]]]
            p.require(np.isfinite(rm).all(), 'Undefined mild rank preservation')
            sm = skill(meta.default, qm)
            row.update(S_bar=float(sm.mean()), R_mild_bar=float(rm.mean()),
                       Score2=score(av, ev, sm.mean()), order_mild_score=score(av, ev, rm.mean()),
                       accuracy_full_score=score(av, ev, row['S_full_bar']))
        arrays = {'ID': meta.ID.to_numpy(), 'SEX': meta.SEX.to_numpy(), 'y': meta.default.to_numpy(),
                  'groups': meta.duplicate_group_id.to_numpy(), 'clean': full[c],
                  'swap_full': q, 'mask_path': np.asarray(removal['predictions']),
                  'A': a['values'], 'E': removal['values'], 'R': r}
        if mild:
            arrays.update(swap_mild=qm, S_mild=sm, R_mild=rm, S_full=skill(meta.default, q))
        npz(folder/f'selection_{c}.npz', **arrays)
        write(folder/f'curves_{c}.json', {'accuracy': a, 'integrated': integrated,
              'removal': {k: v for k, v in removal.items() if k != 'predictions'}})
        evidence[c] = arrays
        rows.append(row)
    csv(folder/'selection_values.csv', pd.DataFrame(rows))
    return pd.DataFrame(rows).set_index('candidate'), evidence


def select_arms(folder, rows, evidence, meta, repeat):
    safe = ranking(rows.SAFE_score.to_dict())[0]
    ranked = ranking(rows.Score2.to_dict())
    arms = {'safe': safe, 'plain_loss': ranking((-rows.clean_brier).to_dict())[0],
            'order_mild': ranking(rows.order_mild_score.to_dict())[0],
            'accuracy_full': ranking(rows.accuracy_full_score.to_dict())[0],
            'selector2_before_check': ranked[0]}
    plan = bootstrap_plan(meta, repeat)
    npz(folder/'bootstrap_plan.npz', ID=meta.ID.to_numpy(), group_order=plan.group_order,
        group_multiplicities=plan.group_multiplicities, row_group_positions=plan.row_group_positions)
    # One shared stream for all candidates and both sexes. Clean, stressed and deterioration.
    point = np.empty((4, 2, 3))
    draws = np.empty((2000, 4, 2, 3))
    for i, c in enumerate(p.CANDIDATES):
        clean = (evidence[c]['clean']-meta.default.to_numpy())**2
        stress = ((evidence[c]['swap_mild'][1:]-meta.default.to_numpy())**2).mean(axis=0)
        vals = np.column_stack((clean, stress, stress-clean))
        for j, sex in enumerate((2, 1)):
            mask = meta.SEX.to_numpy() == sex
            point[i, j] = vals[mask].mean(axis=0)
            draws[:, i, j] = group_means(vals, plan.group_multiplicities,
                                        plan.row_group_positions, mask)
    ref = p.CANDIDATES.index(safe)
    checks, blocked, visited = {}, {}, []
    for c in ranked:
        i = p.CANDIDATES.index(c)
        checks[c] = fairness(point[i, :, 1]-point[ref, :, 1], draws[:, i, :, 1]-draws[:, ref, :, 1])
        blocked[c] = checks[c]['blocked'] if c != safe else False
    final = choice(ranked, safe, blocked)
    for c in ranked:
        visited.append(c)
        if c == final:
            break
    arms['selector2'] = final
    write(folder/'fairness_check.json', {'seed': 42+repeat, 'draws': 2000, 'ranking': ranked,
          'safe': safe, 'top_before_check': ranked[0], 'final': final,
          'changed_top_choice': final != ranked[0], 'visited_in_choice_rule': visited,
          'blocked_on_path': [c for c in visited if blocked[c]], 'candidate_checks': checks,
          'coordinates': ['female', 'male'], 'conditional_fixed_models_predictions': True})
    within = []
    for c in dict.fromkeys((final, safe)):
        i = p.CANDIDATES.index(c)
        for j, sex in enumerate(('female', 'male')):
            for k, metric in enumerate(('clean_brier', 'mild_stressed_brier', 'deterioration')):
                within.append({'candidate': c, 'sex': sex, 'metric': metric,
                    'estimate': point[i, j, k], **percentile(draws[:, i, j, k])})
    # Paired chosen-minus-SAFE intervals accompany the absolute within-repeat values.
    i = p.CANDIDATES.index(final)
    for j, sex in enumerate(('female', 'male')):
        for k, metric in enumerate(('clean_brier', 'mild_stressed_brier', 'deterioration')):
            within.append({'candidate': 'selector2_minus_safe', 'sex': sex, 'metric': metric,
                'estimate': point[i, j, k]-point[ref, j, k],
                **percentile(draws[:, i, j, k]-draws[:, ref, j, k])})
    csv(folder/'within_repeat_intervals.csv', pd.DataFrame(within))
    npz(folder/'selection_bootstrap_values.npz', point=point, draws=draws)
    write(folder/'choices_before_judge.json', arms)
    return arms
