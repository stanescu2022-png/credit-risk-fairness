"""Frozen selection calculation, additionally retaining every greedy alternative.

Derived from confirmation_v1.selection.selection. Only evidence capture changes;
no score, grid, tie, masking or candidate rule is changed.
"""
import numpy as np
import pandas as pd
from research.model_selection_v2 import pilot as p, safe2026 as s
from research.model_selection_v2.step1_v1.runner import write,npz,csv
from research.model_selection_v2.step1_v1.core import MILD,skill,score
from research.model_selection_v2.confirmation_v1.schema import fitting_references,full_grid,accuracy_curve


def capture_greedy(order,clean,predict_removed):
    captured=[]
    def predict(removed):
        probability=predict_removed(removed)
        captured.append(probability.copy())
        return probability
    result=s.greedy_preservation(order,clean,predict)
    result['all_alternative_predictions']=np.asarray(captured)
    return result


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
        removal = capture_greedy(Xsel.columns, full[c],
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
                  'A': a['values'], 'E': removal['values'], 'R': r,
                  'mask_alternatives': removal['all_alternative_predictions']}
        if mild:
            arrays.update(swap_mild=qm, S_mild=sm, R_mild=rm, S_full=skill(meta.default, q))
        npz(folder/f'selection_{c}.npz', **arrays)
        write(folder/f'curves_{c}.json', {'accuracy': a, 'integrated': integrated,
              'removal': {k: v for k, v in removal.items() if k not in ('predictions','all_alternative_predictions')}})
        evidence[c] = arrays
        rows.append(row)
    csv(folder/'selection_values.csv', pd.DataFrame(rows))
    return pd.DataFrame(rows).set_index('candidate'), evidence
