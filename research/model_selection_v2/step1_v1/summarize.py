"""Tables from all repeats, read only after completion of repeat 49."""
from collections import Counter
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.model_selection_v2 import pilot as p
from .core import ARMS, corrected, paired_difference, pattern
from .runner import csv, write


def summarize(run):
    run = Path(run)
    candidate_rows, checks, roles, times = [], [], [], []
    for repeat in range(50):
        folder = run/f'repeats/{repeat:02d}'
        p.require((folder/'COMPLETE.json').exists(), f'Repeat {repeat} incomplete')
        sel = pd.read_csv(folder/'selection_values.csv', float_precision='round_trip')
        judge = pd.read_csv(folder/'judge_scores.csv', float_precision='round_trip')
        arms = json.loads((folder/'choices_before_judge.json').read_text())
        split = json.loads((folder/'split_gate.json').read_text())
        timing = json.loads((folder/'timing.json').read_text())
        check = json.loads((folder/'fairness_check.json').read_text())
        checks.append(check)
        times.append(timing)
        roles.append({'repeat': repeat, **split['sizes'], 'seed': split['seed']})
        for row in sel.to_dict('records'):
            c = row['candidate']
            row.update(repeat=repeat, **{f'choice_{a}': arms[a] for a in ARMS},
                       **{f'n_{r}': split['sizes'][r] for r in split['sizes']})
            for j in judge[judge.candidate.eq(c)].to_dict('records'):
                for metric, value in j.items():
                    if metric not in ('candidate', 'sex'):
                        row[f'judge_{j["sex"]}_{metric}'] = value
            candidate_rows.append(row)
    frame = pd.DataFrame(candidate_rows)
    csv(run/'per_repeat_table.csv', frame)
    csv(run/'role_sizes.csv', pd.DataFrame(roles))
    csv(run/'timings.csv', pd.DataFrame(times))
    rho = float(np.mean([r['judge'] for r in roles])/np.mean([r['fitting']+r['selection'] for r in roles]))

    def contrast(other, reference, metric):
        out = []
        for repeat in range(50):
            f = frame[frame.repeat.eq(repeat)].set_index('candidate')
            a, b = f[f'choice_{other}'].iloc[0], f[f'choice_{reference}'].iloc[0]
            out.append(paired_difference(f[metric].to_dict(), a, b))
        return np.asarray(out)

    def diff_count(other, reference):
        f = frame.drop_duplicates('repeat')
        return int((f[f'choice_{other}'] != f[f'choice_{reference}']).sum())

    comparisons = [('selector2', 'safe'), ('plain_loss', 'safe'), ('order_mild', 'safe'),
                   ('accuracy_full', 'safe'), ('selector2', 'plain_loss'), ('selector2_before_check', 'safe')]
    principal, diffs = [], []
    for a, b in comparisons:
        d = contrast(a, b, 'judge_all_primary')
        principal.append({'contrast': f'{a}_minus_{b}', 'status': 'primary' if (a,b)==('selector2','safe') else 'descriptive',
                          **corrected(d, rho), 'different_choices': diff_count(a, b), 'repeats': 50})
        diffs.extend({'repeat': r, 'contrast': f'{a}_minus_{b}', 'difference': v} for r, v in enumerate(d))
    csv(run/'principal_table.csv', pd.DataFrame(principal))
    csv(run/'per_repeat_primary_differences.csv', pd.DataFrame(diffs))
    sex_rows, secondary = [], []
    for sex in ('female', 'male'):
        for metric in ('primary', 'deterioration'):
            d = contrast('selector2', 'safe', f'judge_{sex}_{metric}')
            sex_rows.append({'sex': sex, 'metric': metric, 'status': 'descriptive',
                **corrected(d, rho), **{f'bonferroni_{k}': v for k,v in corrected(d, rho, True).items()
                                       if k in ('lower', 'upper') and metric == 'primary'}})
    for metric in ('clean_brier', 'clean_log_loss', 'stressed_log_loss'):
        secondary.append({'contrast': 'selector2_minus_safe', 'metric': metric, 'status': 'descriptive',
                          **corrected(contrast('selector2', 'safe', f'judge_all_{metric}'), rho)})
    csv(run/'fairness_table.csv', pd.DataFrame(sex_rows))
    csv(run/'secondary_contrasts.csv', pd.DataFrame(secondary))
    unique = frame.drop_duplicates('repeat')
    frequencies = []
    for arm in ARMS:
        for c in p.CANDIDATES:
            frequencies.append({'arm': arm, 'candidate': c, 'count': int(unique[f'choice_{arm}'].eq(c).sum()), 'out_of': 50})
    csv(run/'selection_frequencies.csv', pd.DataFrame(frequencies))
    patterns = Counter()
    both_same, both_different = 0, 0
    pattern_records = []
    for row in unique.to_dict('records'):
        key = pattern(row['choice_safe'], row['choice_order_mild'], row['choice_accuracy_full'], row['choice_selector2_before_check'])
        patterns[key] += 1
        same = row['choice_order_mild'] == row['choice_accuracy_full'] if key[0] and key[1] else None
        if same is not None:
            both_same += int(same)
            both_different += int(not same)
        pattern_records.append({'repeat': row['repeat'], 'measure_alone_changes': key[0],
             'range_alone_changes': key[1], 'both_changes': key[2], 'single_changes_choose_same_model': same})
    csv(run/'precheck_patterns.csv', pd.DataFrame([{'measure_alone_changes': a, 'range_alone_changes': b,
        'both_changes': c, 'count': patterns[(a,b,c)]} for a,b,c in itertools.product((0,1), repeat=3)]))
    csv(run/'precheck_patterns_by_repeat.csv', pd.DataFrame(pattern_records))
    # Axis distributions and absolute losses give the raw scale beside contrasts.
    axes = []
    for c in p.CANDIDATES:
        f = frame[frame.candidate.eq(c)]
        for axis in ('A_bar', 'E_bar', 'R_safe_bar', 'S_bar', 'S_full_bar', 'R_mild_bar'):
            x = f[axis]
            axes.append({'candidate': c, 'axis': axis, 'mean': x.mean(), 'sd': x.std(ddof=1),
                         'min': x.min(), 'p25': x.quantile(.25), 'median': x.median(), 'p75': x.quantile(.75), 'max': x.max()})
    csv(run/'axis_distributions.csv', pd.DataFrame(axes))
    candidate_losses = frame.groupby('candidate', sort=False)[['clean_brier','judge_all_clean_brier','judge_all_primary',
                         'judge_female_clean_brier','judge_female_primary','judge_male_clean_brier','judge_male_primary']].mean().reset_index()
    csv(run/'candidate_absolute_losses.csv', candidate_losses)
    arm_values = []
    for a in ARMS:
        vals = []
        for r in range(50):
            f = frame[frame.repeat.eq(r)].set_index('candidate')
            chosen = f[f'choice_{a}'].iloc[0]
            vals.append(f.loc[chosen])
        out = {'arm': a}
        for metric in ('judge_all_clean_brier','judge_all_primary','judge_female_primary','judge_male_primary'):
            out[metric] = float(np.mean([v[metric] for v in vals]))
        arm_values.append(out)
    csv(run/'arm_absolute_losses.csv', pd.DataFrame(arm_values))
    d = contrast('selector2','safe','judge_all_primary')
    different = unique.choice_selector2.to_numpy() != unique.choice_safe.to_numpy()
    changed_checks = [c for c in checks if c['changed_top_choice']]
    summary = {'J': 50, 'rho': rho, 'different_selector_choices': int(different.sum()),
        'conditional_on_different_choices': {'count': int(different.sum()),
            'mean': float(d[different].mean()) if different.any() else None,
            'sd': float(d[different].std(ddof=1)) if different.sum()>1 else None},
        'selector2_equals_plain_loss': 50-diff_count('selector2','plain_loss'),
        'fairness_changed_top_count': len(changed_checks),
        'fairness_replacements': dict(Counter(c['top_before_check']+' -> '+c['final'] for c in changed_checks)),
        'repeats_any_challenger_blocked_in_audit': sum(any(v['blocked'] for k,v in c['candidate_checks'].items() if k!=c['safe']) for c in checks),
        'blocked_on_choice_path_counts': dict(Counter(k for c in checks for k in c['blocked_on_path'])),
        'candidate_blocked_audit_counts': {k: sum(c['candidate_checks'][k]['blocked'] for c in checks if k!=c['safe']) for k in p.CANDIDATES},
        'both_single_changes_depart_same_model': both_same, 'both_single_changes_depart_different_models': both_different,
        'total_repeat_seconds': float(sum(v['seconds'] for v in times)),
        'timing_first_two': times[:2], 'deviations': [],
        'status': 'Taiwan development example; no confirmation; approximate corrected intervals, coverage unverified'}
    write(run/'summary.json', summary)
    def fmt(v):
        return 'not reported' if v is None or pd.isna(v) else f'{v:.8f}'
    lines = ['# Taiwan Step 1: accuracy-anchored SAFE', '', '| Primary endpoint contrast | Mean Brier difference | 95% corrected interval | Different choices /50 |',
             '| --- | ---: | --- | ---: |']
    for row in principal:
        lines.append(f'| {row["contrast"]}{" (primary)" if row["status"]=="primary" else " (descriptive)"} | {fmt(row["estimate"])} | [{fmt(row["lower"])}, {fmt(row["upper"])}] | {row["different_choices"]} |')
    lines += ['', '| Sex (descriptive) | Endpoint difference | 95% interval | Bonferroni interval |', '| --- | ---: | --- | --- |']
    for row in sex_rows:
        if row['metric'] != 'primary':
            continue
        lines.append(f'| {row["sex"]} | {fmt(row["estimate"])} | [{fmt(row["lower"])}, {fmt(row["upper"])}] | [{fmt(row["bonferroni_lower"])}, {fmt(row["bonferroni_upper"])}] |')
    primary = principal[0]
    verdict = ('Development support for the directional claim on Taiwan.' if primary['upper'] is not None and primary['upper']<0 else
               'SAFE is judged better here; stop and report.' if primary['lower'] is not None and primary['lower']>0 else 'The primary comparison is unresolved on Taiwan.')
    lines += ['', verdict, '', 'This is a development worked example, not confirmation. Selector 2 was designed after Taiwan results were seen. '
        'Before this run, SAFE selected LR on the original split; RF/GBM had lower inspected probability losses; '
        'Step 0 showed RF retaining accuracy better than LR across swaps and preserving order best at mild severities. '
        'The original-split result was largely predictable; repeated whole-procedure behavior supplies the new information.', '',
        f'Selector 2 and plain loss choose the same model in {summary["selector2_equals_plain_loss"]}/50 repeats. '
        f'Selector 2 and SAFE differ in {summary["different_selector_choices"]}/50. Same-choice repeats remain exact zero differences.', '',
        f'The fairness check changes the top-ranked choice in {len(changed_checks)}/50 repeats. '+
        ('The check did not bind on these data.' if not changed_checks else f'Replacements: {summary["fairness_replacements"]}.'), '',
        'The four-score comparison below is entirely before fairness. Measure alone = accuracy/full; range alone = order/mild; '
        'both = accuracy/mild. The primary judge comparison uses the final post-check Selector 2 choice.', '',
        '| Measure alone changes | Range alone changes | Both changes | Repeats |', '| ---: | ---: | ---: | ---: |']
    for a,b,c in itertools.product((0,1), repeat=3):
        lines.append(f'| {a} | {b} | {c} | {patterns[(a,b,c)]} |')
    lines += ['', f'When both single changes depart from SAFE, they choose the same model in {both_same} repeats and different models in {both_different}.', '',
        '| Candidate | Mean selection clean Brier | Mean judge clean Brier | Mean judge stressed Brier |', '| --- | ---: | ---: | ---: |']
    for row in candidate_losses.to_dict('records'):
        lines.append(f'| {row["candidate"]} | {fmt(row["clean_brier"])} | {fmt(row["judge_all_clean_brier"])} | {fmt(row["judge_all_primary"])} |')
    lines += ['', 'Brier units are raw. No practical cutoff was set; no materiality claim follows. A changed choice alone is not evidence of improvement. '
        'Every diagnostic version has its judge contrast in the principal table. Axis distributions are saved separately because equal nominal weights do not imply equal empirical influence.', '',
        f'The corrected interval uses J=50, sample variance ddof=1, t49 and rho={rho:.10f} from mean realized role sizes. '
        'It is approximate and its achieved coverage is unverified. It describes repeat split/fit/select/judge variation on these development records, '
        'conditional on the fixed configurations, fault mechanisms and data source. It does not cover other datasets, times, candidate pools, '
        'or post-Taiwan method design. A zero-variance contrast has no reported interval. '
        'Sex results are descriptive; adjusted intervals use t49 at .9875. Zero inclusion is unresolved, not evidence of equal treatment.', '',
        'Fairness-check and within-repeat percentile intervals use 2,000 whole-original-predictor-group draws, seed 42+r, '
        'and hold fitted models and disturbed predictions fixed. They are saved per repeat and are never pooled into the corrected interval. '
        'Step 0 zero-variation coordinates (SD <=1e-12) are explicitly excluded from joint intervals and cannot establish a resolved block. '
        'Judge realization losses are averaged after computing each loss; their Monte Carlo spread is separate from sampling uncertainty.', '',
        f'Repeats 0 and 1 took {times[0]["seconds"]:.2f}s and {times[1]["seconds"]:.2f}s. '
        f'Total repeat execution: {summary["total_repeat_seconds"]/3600:.3f} hours. All individual stage times are in timings.csv. '
        'No comparative outcomes were inspected between repeats 2 and 49. No protocol deviations. '
        'Only Taiwan development records were modelled; historical IDs served exclusion checks only. No tuning, calibration, threshold change, '
        'old-file edit, commit, push, publication, external contact or fresh-dataset screen.', '',
        'Reconstruction handoff: per_repeat_table.csv contains every candidate, all six arm choices, all selection values, judge scores by sex, '
        'and realized role sizes. principal_table.csv, fairness_table.csv and per_repeat_primary_differences.csv give the derived tables. '
        'choices_before_judge.json freezes every repeat choice before judge computation. Split assignments, models, all probabilities, '
        'bootstrap plans/draws, greedy alternatives, scenario registries and loss arrays are under repeats/. '
        'The attachment, hashes, source snapshots, test logs, reconstruction receipt and preservation comparison accompany the run. '
        'Verification is a same-session recheck, not independent review. Existing startup documents remain untouched under the protocol; '
        'this dated report is the accepted/executed Step 1 record. Fresh-dataset confirmation belongs to Astra.']
    for row in sex_rows:
        if row['metric']=='primary' and row.get('bonferroni_lower') is not None and row['bonferroni_lower']>0:
            lines.insert(0, f'**Judge subgroup harm: adjusted interval is above zero for {row["sex"]}; Selector 2 is judged worse for this group.**\n')
    with (run/'report.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')
