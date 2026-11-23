"""Collect section 6.5 reports from sealed Job A and Job B outputs."""
import argparse
from pathlib import Path

import pandas as pd

from research.model_selection_v2.confirmation_v2 import reporting
from research.model_selection_v2.pilot import CANDIDATES
from research.model_selection_v2.step1_v1.core import ranking

from .artifacts import read_json, sha, verify
from .collect import persist
from .names import legacy_pair_records, legacy_table, primary_summary, rename, summarize


DIRECT_ARM = 'direct_small_swap_brier'
DIRECT_CHOICE = 'choice_' + DIRECT_ARM
PRIMARY_ENDPOINT = 'judge_J1_primary'


def choose_direct_small_swap(s_bar):
    """Choose from the candidate-to-S-bar map using the frozen Step 1 ranking."""
    return ranking(s_bar)[0]


def add_direct_choices(table):
    """Recover each choice from the selection column, before looking up judge scores."""
    scores = table.set_index(['repeat', 'candidate'])['S_bar']
    if not scores.index.is_unique:
        raise ValueError('Expected one selection S_bar per repeat and candidate')
    choices = {
        repeat: choose_direct_small_swap(group.droplevel('repeat').to_dict())
        for repeat, group in scores.groupby(level='repeat')
    }
    selected = table.repeat.map(choices)
    if DIRECT_CHOICE in table and not table[DIRECT_CHOICE].eq(selected).all():
        raise ValueError('Recorded direct choice differs from the selection S_bar winner')
    return table.assign(**{DIRECT_CHOICE: selected})


def add_comparisons(result, table, job):
    rho = result['rho']
    key = 'primary_contrasts' if job == 'A' else 'contrasts'
    endpoints = (PRIMARY_ENDPOINT,) if job == 'A' else reporting.ENDPOINTS
    added = [reporting.contrast(table, DIRECT_ARM, 'safe', endpoint, rho) for endpoint in endpoints]
    for arm, baseline in (
        ('selector2', DIRECT_ARM),
        ('selector2', 'plain_loss'),
        (DIRECT_ARM, 'plain_loss'),
    ):
        added.append(reporting.contrast(table, arm, baseline, PRIMARY_ENDPOINT, rho))
    result[key] = pd.concat([result[key], pd.DataFrame(added)], ignore_index=True)

    choices = table.drop_duplicates('repeat')
    matrix = result['pair_match_matrix']
    pairs = []
    for arm in matrix.index:
        matches = int(choices['choice_' + arm].eq(choices[DIRECT_CHOICE]).sum())
        matrix.loc[arm, DIRECT_ARM] = matrix.loc[DIRECT_ARM, arm] = matches
        pairs.append(dict(arm_a=arm, arm_b=DIRECT_ARM, matches=matches, disagreements=len(choices) - matches))
    matrix.loc[DIRECT_ARM, DIRECT_ARM] = len(choices)
    result['pair_match_matrix'] = matrix.astype(int)
    result['pair_matches'] = pd.concat([result['pair_matches'], pd.DataFrame(pairs)], ignore_index=True)


def verified_repeat(folder, job, table_name):
    """Check the completion receipt and verify every handoff hash before reading."""
    receipt = read_json(folder/'COMPLETE.json')
    if receipt['job'] != job or receipt['complete'] is not True:
        raise ValueError(f'Expected completed Job {job} receipt in {folder}')
    if sha(folder/'artifact_manifest.json') != receipt['artifact_manifest_sha256']:
        raise ValueError(f'Job {job} artifact manifest changed in {folder}')
    files = verify(folder)
    required = [table_name]
    if job == 'B':
        required.extend(('every_pair_gate_records.json', 'group_audit_by_arm.json', 'J1_group_audit.json'))
    if any(name not in files for name in required):
        raise ValueError(f'Job {job} receipt does not bind the required reporting inputs in {folder}')
    table = pd.read_csv(folder/table_name, float_precision='round_trip')
    if len(table) != len(CANDIDATES) or not table.repeat.eq(receipt['repeat']).all():
        raise ValueError(f'Job {job} table does not match its repeat receipt in {folder}')
    return table


def group_audit_records(folder, repeat, candidate):
    audits = read_json(folder/'group_audit_by_arm.json')
    selected = [
        dict(row, repeat=repeat, arm=DIRECT_ARM)
        for row in read_json(folder/'J1_group_audit.json')
        if row['candidate'] == candidate
    ]

    def cells(rows):
        return sorted((row['condition'], row['group'], row['metric']) for row in rows)

    expected = cells(row for row in audits if row['arm'] == 'safe')
    if not expected or cells(selected) != expected:
        raise ValueError(f'Incomplete direct candidate group audit in {folder}')
    recorded = [row for row in audits if row['arm'] == DIRECT_ARM]
    if recorded and recorded != selected:
        raise ValueError(f'Recorded direct audit differs from saved candidate audit in {folder}')
    return [row for row in audits if row['arm'] != DIRECT_ARM] + selected


def collect(job, repeat_folders, output):
    """Export the labelled section 6.5 Job A or Job B tables from saved repeats."""
    if job not in ('A', 'B'):
        raise ValueError('Expected Job A or Job B')
    table_name = 'per_repeat_primary.csv' if job == 'A' else 'per_repeat_full.csv'
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Report destination already exists: {output}')
    folders = [Path(folder) for folder in repeat_folders]
    if len(folders) != 50:
        raise ValueError('Exactly fifty completed repeat folders are required')
    tables = [verified_repeat(folder, job, table_name) for folder in folders]
    table = add_direct_choices(pd.concat(tables, ignore_index=True))
    result = primary_summary(table) if job == 'A' else summarize(table)
    add_comparisons(result, table, job)
    if job == 'B':
        pairs = [row for folder in folders for row in read_json(folder/'every_pair_gate_records.json')]
        result['operating_characteristics'] = rename(
            reporting.operating_characteristics(legacy_pair_records(pairs), legacy_table(table))
        )
        direct_choices = table.drop_duplicates('repeat').set_index('repeat')[DIRECT_CHOICE]
        audits = []
        for folder, saved in zip(folders, tables):
            repeat = int(saved.repeat.iloc[0])
            audits.extend(group_audit_records(folder, repeat, direct_choices.loc[repeat]))
        result['group_audit_intervals'] = reporting.summarize_audits(audits, result['rho'])
    output.mkdir(exist_ok=False)
    table.to_csv(output/table_name, index=False)
    persist(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', choices=('A', 'B'))
    parser.add_argument('repeat_root', type=Path, help='directory containing the completed job repeat folders')
    parser.add_argument('output', type=Path, help='fresh report output directory')
    args = parser.parse_args()
    folders = [args.repeat_root/f'{args.job}_{repeat}' for repeat in range(50)]
    collect(args.job, folders, args.output)


if __name__ == '__main__':
    main()
