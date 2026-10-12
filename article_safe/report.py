"""Build the article report from verified saved artifacts, without fitting models.

Usage: .venv/Scripts/python.exe -m article_safe.report --run-id <completed_run>
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from article_safe.artifacts import existing_run, verify_preparation, verify_stage, stage_manifest
from article_safe.protocol import ROOT

LABELS = {
    'lr_mle_full': 'Article LR, full',
    'lr_mle_selected': 'Article LR, selected',
    'gbm_full': 'GBM, full',
    'gbm_selected': 'GBM, selected',
    'lr_frozen_regularized': 'Existing regularized LR (reference)',
}
PROFILE_COLUMNS = {
    'model': 'Model', 'status': 'Status', 'predictors': 'Predictors',
    'ROC-AUC': 'AUC', 'Average Precision (AP)': 'AP', 'Brier Score': 'Brier',
    'Recall': 'Default recall', 'FPR': 'FPR',
    'retained_full_model_importance_percent': 'Retained importance (%)',
}


def number(value, digits=4):
    return 'Unavailable' if pd.isna(value) else f'{float(value):.{digits}f}'


def profile_table(metrics, roles):
    """Keep all planned roles and missing values; the reference stays labeled."""
    indexed = metrics.set_index('model')
    if not indexed.index.is_unique or any(role not in indexed.index for role in roles):
        raise ValueError('Saved model profiles are incomplete or duplicated.')
    return indexed.loc[roles].reset_index()[list(PROFILE_COLUMNS)].copy()


def markdown_profiles(frame):
    headers = list(PROFILE_COLUMNS.values())
    rows = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for _, row in frame.iterrows():
        values = [LABELS[row.model], row.status.replace('_', ' '), number(row.predictors, 0)]
        values += [number(row[column], 1 if column.endswith('percent') else 4)
                   for column in list(PROFILE_COLUMNS)[3:]]
        rows.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(rows)


def save_figure(fig, directory, name):
    fig.savefig(directory / f'{name}.png', dpi=170, bbox_inches='tight')
    fig.savefig(directory / f'{name}.svg', bbox_inches='tight', metadata={'Date': None})
    plt.close(fig)


def make_figures(run, destination, choices, metrics):
    """Plot saved selection, the final comparison, and shared group effects."""
    destination.mkdir()
    path = pd.read_csv(run / 'selection/gbm_prefix_path.csv')
    k, stop = choices['gbm']['selected_k'], choices['gbm']['stopped_at_k']
    selected = path.loc[path.k == k].iloc[0]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(path.k, path.auc, '-o', color='#405b78', markersize=3, label='Validation AUC; all prefixes diagnostic')
    ax.scatter([k], [selected.auc], s=75, color='#19775a', zorder=4, label=f'Selected: {k} predictors')
    if stop is not None:
        stopped = path.loc[path.k == stop].iloc[0]
        ax.scatter([stop], [stopped.auc], s=65, marker='x', color='#b14a3c', zorder=4, label=f'First failure: {stop} predictors')
    ax.set(xlabel='Predictors in fixed training-SHAP order', ylabel='Validation ROC-AUC',
           title='GBM selection stops at the first failed step')
    ax.set_xticks([1, 3, 4, 8, 12, 16, 22])
    ax.grid(axis='y', alpha=.2)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    save_figure(fig, destination, 'gbm_selection_path')
    intervals = pd.read_csv(run / 'evaluation/auc_difference_intervals.csv')
    interval = intervals.loc[intervals['comparison'] == 'gbm_selected minus gbm_full'].iloc[0]
    comparison = pd.read_csv(run / 'evaluation/auc_comparisons.csv')
    delta = comparison.loc[(comparison['first'] == 'gbm_selected') & (comparison['second'] == 'gbm_full'), 'difference'].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7), gridspec_kw={'width_ratios': [1, 1.4]})
    values = metrics.set_index('model').loc[['gbm_full', 'gbm_selected'], 'ROC-AUC']
    axes[0].bar(['Full (22)', 'Selected (3)'], values, color=['#405b78', '#19775a'], width=.55)
    axes[0].set(ylim=(0, 1), ylabel='Holdout ROC-AUC', title='Final GBM models')
    for i, value in enumerate(values):
        axes[0].text(i, value + .025, f'{value:.4f}', ha='center', fontsize=10)
    axes[1].errorbar([delta], [0], xerr=[[delta-interval.ci_lower], [interval.ci_upper-delta]],
                    fmt='o', color='#b14a3c', capsize=6)
    axes[1].axvline(0, color='#666', linewidth=1)
    axes[1].set(yticks=[], xlabel='Selected minus full AUC', title='Conditional paired 95% interval', ylim=(-1, 1))
    axes[1].text(.02, .08, '2,000 whole predictor-group resamples\nFixed fitted models; exploratory holdout',
                 transform=axes[1].transAxes, fontsize=8)
    fig.tight_layout()
    save_figure(fig, destination, 'gbm_full_vs_selected')
    shared = choices['gbm']['selected_features']
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, model in zip(axes, ('gbm_full', 'gbm_selected')):
        effects = pd.read_csv(run / f'evaluation/{model}_contribution_fairness.csv').set_index('feature').loc[shared]
        positions = np.arange(len(shared))
        ax.barh(positions-.17, effects.male_mean_absolute, .32, color='#405b78', label='Male')
        ax.barh(positions+.17, effects.female_mean_absolute, .32, color='#c78638', label='Female')
        ax.set(yticks=positions, yticklabels=shared, xlabel='Mean absolute SHAP (default log-odds)', title=LABELS[model])
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        ax.grid(axis='x', alpha=.15)
    fig.suptitle('Group contributions for the three retained predictors', fontsize=12)
    fig.tight_layout()
    save_figure(fig, destination, 'group_contributions')


def report(run_id):
    run = existing_run(run_id)
    protocol = verify_preparation(run)
    verify_stage(run / 'selection', protocol)
    verify_stage(run / 'evaluation', protocol)
    destination = run / 'report'
    if destination.exists():
        raise FileExistsError('This report already exists; completed reports are not overwritten.')
    metrics = pd.read_csv(run / 'evaluation/holdout_metrics.csv', float_precision='round_trip')
    choices = json.loads((run / 'selection/selected_subsets.json').read_text())
    roles = protocol['final_roles'] + [protocol['estimator_reference']]
    profiles = profile_table(metrics, roles)
    comparison = pd.read_csv(run / 'evaluation/auc_comparisons.csv', float_precision='round_trip')
    delta = comparison.loc[(comparison['first'] == 'gbm_selected') & (comparison['second'] == 'gbm_full')].iloc[0]
    interval = pd.read_csv(run / 'evaluation/auc_difference_intervals.csv').iloc[3]
    indexed = metrics.set_index('model')
    full, reduced = indexed.loc['gbm_full'], indexed.loc['gbm_selected']
    destination.mkdir()
    profiles.to_csv(destination / 'model_profiles.csv', index=False)
    pd.DataFrame(protocol['fidelity']).to_csv(destination / 'article_fidelity.csv', index=False)
    statuses = json.loads((run / 'evaluation/component_statuses.json').read_text())
    statuses += [{'model': 'all', 'component': 'overall SAFE score', 'status': 'not_supplied_by_article'},
                 {'model': 'lr_mle', 'component': 'coefficient-distribution KS', 'status': 'not_identifiable_from_article'}]
    pd.DataFrame(statuses).to_csv(destination / 'component_statuses.csv', index=False)
    make_figures(run, destination / 'figures', choices, metrics)
    predictors = ', '.join(choices['gbm']['selected_features'])
    narrative = f'''# Article methodology: results and remaining limitation

The implemented article-style GBM procedure selected **{predictors}**. Its first-stop rule retained three predictors, but the final reduced model performed worse than the full GBM on this sample. The unpenalized article LR could not be estimated under our categorical design; its results remain explicitly unavailable. No categories were pooled and no regularized estimator was substituted into its tests.

## Model profiles

All four planned article roles are shown, followed by the existing LR reference. AP is sklearn average precision. Default recall and FPR use probability strictly greater than 0.50. Retained importance is the selected predictors' share of the **full model's** mean absolute SHAP importance.

{markdown_profiles(profiles)}

## What the selection did

Selection used 19,203 fitting records and 4,801 validation records from the saved training partition. The GBM order came from full-model training SHAP. Adding the fourth predictor, PAY_2, slightly decreased validation AUC, so the procedure stopped. Later prefixes recovered predictive performance, but did not reverse the first stop. The complete path is retained as a diagnostic. LR backward selection was not entered because its full fitting design was rank deficient; no successful LR selection is claimed.

![Validation selection path](figures/gbm_selection_path.png)

Both GBMs were refitted on all 24,004 training records before evaluation on the same 5,996 holdout records. Selected-minus-full AUC was **{delta.difference:.4f}**, with a conditional paired group-bootstrap 95% interval **[{interval.ci_lower:.4f}, {interval.ci_upper:.4f}]**. The nominal paired DeLong p-value was {delta.p_value:.3g}, or {delta.p_value_holm:.3g} after Holm adjustment across the four planned comparisons, including the unavailable LR comparisons. The selected predictors retained **{reduced.retained_full_model_importance_percent:.1f}%** of full-model mean absolute SHAP importance. These results quantify the cost of this particular reduction, not an overall SAFE winner.

![Final full-versus-selected result](figures/gbm_full_vs_selected.png)

## Explanation and group evidence

The GBM SHAP decompositions passed additivity checks. Both auxiliary regressions of the observed outcome on SHAP contributions were estimable. They are descriptive explanation tests, not predictive candidates or causal validation. Fairness tables give each predictor's group means, magnitude Gini, signed-effect audit and contribution-distribution KS. No fairness verdict or aggregate score is assigned. Contributions were recomputed for each fitted model, so the shared predictors' effects can change after reduction.

![Group contribution comparison](figures/group_contributions.png)

Read the [full group table](../evaluation/gbm_full_contribution_fairness.csv), [selected group table](../evaluation/gbm_selected_contribution_fairness.csv), [signed/absolute explanation-gap intervals](../evaluation/explanation_gap_intervals.csv), [decision metric gaps](../evaluation/group_metric_gaps.csv) and [full](../evaluation/gbm_full_shap_regression.csv)/[selected auxiliary regression](../evaluation/gbm_selected_shap_regression.csv) results. Signed gaps are female minus male. Low magnitude Gini indicates more equal contribution magnitudes; it does not establish fair decisions. KS assumes independent continuous observations, while these contributions have ties and predictor-group dependence; its p-values are nominal.

## What remains unavailable, and why

The full article LR design has 78 columns, including its intercept, but rank 74 on fitting data and rank 76 on complete training data. In particular, PAY_4=8, PAY_5=8 and PAY_6=8 occur for the same single training record. Other rare levels have only one outcome class, creating further estimability concerns. This prevents unique ordinary coefficient inference under the approved full-rank contract. It is not evidence that LR predicts poorly or that the article framework is inferior. The existing regularized LR remains an estimator reference, with no article likelihood tests attached.

The [discussion note](../../../../docs/plans/article_safe_lr_limitation_2026-10-10.md) explains the categorical coding, regularization distinction and comparison implications. The [saved diagnosis](../diagnostics/lr_design_identifiability.json) and [rare-level counts](../diagnostics/rare_repayment_indicator_support.csv) provide the training evidence. A coefficient-level KS procedure and an overall SAFE ranking are also not sufficiently specified in the original article; neither is invented here.

## How to use these results

The implementation supplies a reproducible article-based comparison branch with a visible LR limitation. It does not reproduce every unspecified author choice: GBM substitutes for the credit example's random forest; categorical coding, SHAP regression, Gini/KS targets and removal order follow the declared interpretation in the [specification](../../../../docs/plans/article_safe_specification_2026-10-10.md). The [fidelity table](article_fidelity.csv) distinguishes article procedures from interpretations and supporting diagnostics.

The original holdout was historically inspected, so these findings are exploratory. The 2,000 whole original-predictor-group bootstrap draws condition on fixed fits/subsets and do not include refitting or selection uncertainty. Full GBM reproduced the existing frozen GBM probabilities exactly. A fair later framework comparison should use shared candidate models and preprocessing where possible; changes of encoding or estimator need separate attribution. Any superiority claim requires a defined criterion and supporting experiments, rather than treating missing LR outputs as a win.

Saved [selection](../selection/stage_manifest.json) and [evaluation](../evaluation/stage_manifest.json) manifests record source/input/output identities. This report was generated only from those verified outputs; it did not refit models, change category treatment or rerun earlier research stages.
'''
    (destination / 'results.md').write_text(narrative, encoding='utf-8')
    stage_manifest(destination, 'reporting', protocol, {
        'selection_manifest_sha256': __import__('hashlib').sha256((run/'selection/stage_manifest.json').read_bytes()).hexdigest(),
        'evaluation_manifest_sha256': __import__('hashlib').sha256((run/'evaluation/stage_manifest.json').read_bytes()).hexdigest(),
        'models_refitted': False, 'category_pooling': False,
    })
    print(f'Report saved: {destination / "results.md"}', flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    report(parser.parse_args().run_id)


if __name__ == '__main__':
    main()