"""Same-session reconstruction from saved per-repeat evidence, without fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from research.model_selection_v2 import pilot as p
from research.model_selection_v2.safe2026 import rank_candidates
from .runner import write, CONDITIONS, FAMILIES, ROOT


def verify(run):
    run = Path(run)
    f = pd.read_csv(run/'per_repeat_table.csv', float_precision='round_trip')
    roles = f.drop_duplicates('repeat')
    assert len(roles)==50 and len(f)==200
    rho = roles.n_judge.mean()/(roles.n_fitting.mean()+roles.n_selection.mean())
    principal = pd.read_csv(run/'principal_table.csv', float_precision='round_trip')
    max_error, checked = 0., 0
    def near(a, b):
        nonlocal max_error, checked
        av, bv = np.asarray(a, float), np.asarray(b, float)
        np.testing.assert_allclose(av, bv, rtol=0, atol=1e-12)
        max_error = max(max_error, float(np.max(np.abs(av-bv))))
        checked += av.size
    # Reconstruct principal contrasts by looking up arm choices in the saved long table.
    for row in principal.to_dict('records'):
        other, baseline = row['contrast'].split('_minus_')
        differences = []
        different = 0
        for r in range(50):
            tab = f[f.repeat.eq(r)].set_index('candidate')
            a, b = tab[f'choice_{other}'].iloc[0], tab[f'choice_{baseline}'].iloc[0]
            different += a != b
            differences.append(float(tab.loc[a, 'judge_all_primary']-tab.loc[b, 'judge_all_primary']))
        differences = np.asarray(differences)
        estimate = differences.sum()/50
        near(estimate, row['estimate'])
        assert different == row['different_choices']
        variance = np.sum((differences-estimate)**2)/49
        if variance == 0:
            assert pd.isna(row['lower']) and pd.isna(row['upper'])
        else:
            half = t.ppf(.975,49)*np.sqrt((1/50+rho)*variance)
            near([estimate-half,estimate+half], [row['lower'],row['upper']])
    for r in range(50):
        folder = run/f'repeats/{r:02d}'
        assignment = pd.read_csv(folder/'assignment.csv')
        assert len(assignment)==24004 and assignment.ID.is_unique
        assert assignment.groupby('duplicate_group_id').role.nunique().max()==1
        tab = f[f.repeat.eq(r)].set_index('candidate')
        check = json.loads((folder/'fairness_check.json').read_text())
        choices = json.loads((folder/'choices_before_judge.json').read_text())
        ranked = [x['candidate_id'] for x in rank_candidates(tab.Score2.to_dict(), p.CANDIDATES)['rows']]
        assert ranked == check['ranking']
        expected = next(c for c in ranked if c==choices['safe'] or not check['candidate_checks'][c]['blocked'])
        assert expected == choices['selector2']
        with np.load(folder/'bootstrap_plan.npz') as bp, np.load(folder/'selection_bootstrap_values.npz') as boot:
            meta = assignment[assignment.role.eq('selection')].reset_index(drop=True)
            np.testing.assert_array_equal(bp['ID'],meta.ID)
            # Direct row-multiplicity recheck, including any cross-sex groups.
            for cidx,c in enumerate(p.CANDIDATES):
                with np.load(folder/f'selection_{c}.npz') as z:
                    y = z['y']
                    near(z['A'].mean(),tab.loc[c,'A_bar'])
                    near(z['E'].mean(),tab.loc[c,'E_bar'])
                    near(z['R'].mean(),tab.loc[c,'R_safe_bar'])
                    sm = 1-((z['swap_mild']-y)**2).mean(axis=1)/(y.mean()*(1-y.mean()))
                    near(sm.mean(),tab.loc[c,'S_bar'])
                    clean = (z['clean']-y)**2
                    stressed = ((z['swap_mild'][1:]-y)**2).mean(axis=0)
                    vals = np.column_stack((clean,stressed,stressed-clean))
                    for b in (0,1,997,1999):
                        w = bp['group_multiplicities'][b,bp['row_group_positions']]
                        for j,sex in enumerate((2,1)):
                            weights=w*(meta.SEX.to_numpy()==sex)
                            near(np.average(vals,axis=0,weights=weights),boot['draws'][b,cidx,j])
        for c in p.CANDIDATES:
            with np.load(folder/f'judge_{c}.npz') as z:
                expected_ids = assignment.loc[assignment.role.eq('judge'),'ID']
                np.testing.assert_array_equal(z['ID'],expected_ids)
                y, sex, clean = z['y'],z['SEX'],z['clean']
                for name,code in (('all',None),('female',2),('male',1)):
                    mask = np.ones(len(y),bool) if code is None else sex==code
                    brier,logs = [],[]
                    for family in FAMILIES:
                        for condition in CONDITIONS.values():
                            q=z[family+'__'+condition][:,mask]
                            yy=y[mask]
                            brier.append(((q-yy)**2).mean(axis=1).mean())
                            clipped=np.clip(q,np.finfo(float).eps,1-np.finfo(float).eps)
                            logs.append((-np.where(yy==1,np.log(clipped),np.log1p(-clipped))).mean(axis=1).mean())
                    near(np.mean(brier),tab.loc[c,f'judge_{name}_primary'])
                    near(np.mean(logs),tab.loc[c,f'judge_{name}_stressed_log_loss'])
                    near(np.mean((clean[mask]-y[mask])**2),tab.loc[c,f'judge_{name}_clean_brier'])
    for relative,h in json.loads((run/'source_hashes.json').read_text()).items():
        assert p.sha256(ROOT/relative)==h
    for relative,h in json.loads((run/'input_hashes.json').read_text()).items():
        assert p.sha256(ROOT/relative)==h
    write(run/'reconstruction_receipt.json', {'passed':True,'review':'same-session recheck, not independent review',
        'repeats':50,'principal_rows':len(principal),'numeric_values_checked':int(checked),'max_absolute_error':max_error,
        'source_and_input_hashes_unchanged':True,'reconstruction':'principal table from per-repeat long table; all judge losses from raw probabilities; selected bootstrap draws via direct row weights'})
