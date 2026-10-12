"""Tables and compact figures for the prespecified robustness experiment."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.robustness_protocol import PRIMARY_SAFE, PRIMARY_SEMANTIC
from src.robustness_uncertainty import FOCUS

MODEL_NAMES={'logistic':'LR','gradient_boosting':'GBM','GBM_minus_LR':'GBM − LR'}
COLORS={'logistic':'#2563a6','gradient_boosting':'#c35a24'}


def markdown(frame, digits=5):
    def cell(value):
        if isinstance(value,(float,np.floating)):
            return 'undefined' if not np.isfinite(value) else f'{value:.{digits}f}'
        return str(value).replace('|','/')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
                       '| '+' | '.join(['---']*len(frame.columns))+' |']+
                      ['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def save_figure(fig, directory, name):
    fig.tight_layout()
    fig.savefig(directory/f'{name}.png',dpi=160)
    fig.savefig(directory/f'{name}.svg',metadata={'Date':None})
    plt.close(fig)


def make_figures(artifacts, protocol, directory):
    summary=artifacts['diagnostic_summary']
    registry=artifacts['realization_registry'].drop_duplicates('scenario').set_index('scenario')
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    metrics=['mean_absolute_movement','flip_rate','delta_Average Precision (AP)']
    for i,layer in enumerate(['safe','semantic']):
        names=[s['scenario'] for s in protocol['scenarios'] if s['layer']==layer and s['scope']=='all_features' and 'permutation_seed' not in s]
        for j,metric in enumerate(metrics):
            ax=axes[i,j]
            for kind in COLORS:
                subset=summary[(summary.scenario.isin(names))&(summary.model==kind)&(summary.SEX==0)&(summary.metric==metric)].copy()
                subset['severity']=subset.scenario.map(registry.severity)
                subset=subset.sort_values('severity')
                if layer=='semantic':
                    ax.errorbar(subset.severity,subset['mean'],yerr=subset.mc_sd,color=COLORS[kind],marker='o',capsize=3,label=MODEL_NAMES[kind])
                else:
                    ax.plot(subset.severity,subset['mean'],color=COLORS[kind],marker='o',label=MODEL_NAMES[kind])
            ax.set_title(metric.replace('delta_','Δ ').replace('_',' '),fontsize=10)
            ax.set_xlabel('SAFE tail fraction p' if layer=='safe' else 'Semantic q and IQR magnitude fraction')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle('All-feature joint sensitivity: distinct corruption layers\nSemantic bars show repetition SD; SAFE is deterministic',fontsize=12)
    save_figure(fig,directory,'joint_severity')
    fig,axes=plt.subplots(1,2,figsize=(13,8))
    rgr=artifacts['authentic_rgr']
    for j,(kind,color) in enumerate(COLORS.items()):
        part=rgr[(rgr.model==kind)&(rgr.p==.05)&(rgr.kind=='individual')].set_index('scope').loc[protocol['predictor_order']]
        axes[j].barh(part.index,part.RGR,color=color)
        axes[j].invert_yaxis(); axes[j].set_xlim(0,1.01)
        axes[j].set_title(f'{MODEL_NAMES[kind]} authentic individual RGR, p=.05')
        axes[j].set_xlabel('Higher = greater original-score ordering preservation')
        axes[j].grid(axis='x',alpha=.2)
    save_figure(fig,directory,'individual_rgr_primary_severity')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for kind,color in COLORS.items():
        part=rgr[(rgr.model==kind)&(rgr.p==.05)&(rgr.kind=='group')&(~rgr.scenario.str.endswith('permuted'))]
        axes[0].plot(part.scope,part.RGR,marker='o',color=color,label=MODEL_NAMES[kind])
        names=[f'semantic__{f}__primary' for f in protocol['predictor_order']]
        sem=summary[(summary.scenario.isin(names))&(summary.model==kind)&(summary.SEX==0)&(summary.metric=='mean_absolute_movement')].set_index('scenario').loc[names]
        axes[1].plot(protocol['predictor_order'],sem['mean'],marker='.',color=color,label=MODEL_NAMES[kind])
    axes[0].set_ylabel('Authentic RGR'); axes[0].set_title('Grouped SAFE p=.05 (not individual averages)')
    axes[1].set_ylabel('Mean absolute probability movement'); axes[1].set_title('Semantic individual q=.10 (30-repetition means)')
    for ax in axes:
        ax.tick_params(axis='x',rotation=70,labelsize=8); ax.grid(alpha=.2); ax.legend()
    save_figure(fig,directory,'feature_family_sensitivity')


def make_report(artifacts, protocol, directory, report):
    make_figures(artifacts,protocol,directory)
    summary=artifacts['diagnostic_summary']
    paired=artifacts['paired_summary']
    exposure=artifacts['exposure_summary']
    intervals=artifacts['sampling_intervals']
    rgr=artifacts['authentic_rgr']
    global_ci=intervals[intervals.SEX==0][['scenario','model','metric','estimate','ci_lower','ci_upper']]
    primary=summary[(summary.scenario.isin(protocol['primary_scenarios']))&(summary.SEX==0)&(summary.metric.isin(FOCUS+['ROC-AUC','Average Precision (AP)','Brier Score','Spearman']))]
    table=primary.pivot(index=['scenario','model'],columns='metric',values='mean').reset_index()
    group_rows=[]
    for scenario in protocol['primary_scenarios']:
        for sex in (1,2):
            exp=exposure[(exposure.scenario==scenario)&(exposure.SEX==sex)&(exposure.feature=='__union__')].set_index('metric')['mean']
            for kind in COLORS:
                values=summary[(summary.scenario==scenario)&(summary.SEX==sex)&(summary.model==kind)].set_index('metric')['mean']
                group_rows.append({'scenario':scenario,'model':kind,'SEX':sex,
                    'selected':exp['selected'],'changed':exp['changed'],'exposure':exp['changed_rate'],
                    'mean_abs_movement':values['mean_absolute_movement'],'flip_rate':values['flip_rate'],
                    'delta_AP':values['delta_Average Precision (AP)'],'delta_AUC':values['delta_ROC-AUC'],
                    'delta_Brier':values['delta_Brier Score']})
    group_table=pd.DataFrame(group_rows)
    mc=primary[primary.scenario==PRIMARY_SEMANTIC][['model','metric','mean','mc_sd','minimum','maximum','repetitions']]
    safe_primary=rgr[rgr.scenario==PRIMARY_SAFE][['scenario','model','RGR']]
    primary_exposure=exposure[(exposure.scenario.isin(protocol['primary_scenarios']))&(exposure.SEX==0)&(exposure.feature=='__union__')]
    notes=[]
    for scenario in protocol['primary_scenarios']:
        points=summary[(summary.scenario==scenario)&(summary.SEX==0)].pivot(index='model',columns='metric',values='mean')
        notes.append(f'`{scenario}`: LR/GBM mean absolute movement {points.loc["logistic","mean_absolute_movement"]:.6f}/{points.loc["gradient_boosting","mean_absolute_movement"]:.6f}; flips {100*points.loc["logistic","flip_rate"]:.3f}%/{100*points.loc["gradient_boosting","flip_rate"]:.3f}%. AP changes are {points.loc["logistic","delta_Average Precision (AP)"]:+.6f}/{points.loc["gradient_boosting","delta_Average Precision (AP)"]:+.6f}; signed improvements and adverse changes are retained.')
    rgr_diff=float(safe_primary.set_index('model').loc['gradient_boosting','RGR']-safe_primary.set_index('model').loc['logistic','RGR'])
    perm=artifacts['row_order_diagnostic']
    record=artifacts['permutation_record_comparison']
    order_exposure_difference=int((record.canonical_changed!=record.permuted_changed).sum())
    undefined=artifacts['undefined_results']
    category_support={f:protocol['semantic_rules'][f].get('support',protocol['semantic_rules'][f].get('categories')) for f in ['PAY_0','PAY_5','PAY_6','EDUCATION','MARRIAGE']}
    content=[
        '# Robustness evaluation — 8 October 2026',
        'This completed empirical experiment measures prediction and recorded-outcome sensitivity of the existing frozen LR and GBM under prespecified synthetic input corruption. The two primary scenarios were fixed before results. It does not validate the entire extended trustworthiness framework or establish novelty.',
        '## Frozen baseline and prespecified protocol',
        f'The repository baseline reproduced: {protocol["training_rows"]:,} training and {protocol["holdout_rows"]:,} holdout records, unchanged split and 22 predictors in the frozen input order. Default is class 1 and SAFE selects probability column 1 of classes [0,1]. Decisions are strictly p>0.50, including equality as class 0. IDs, SEX, outcomes and canonical order match all four earlier-stage prediction artifacts per model. Probability tolerance is 1e-15 absolute, metric tolerance 1e-12. No baseline conflict was found; no fitting, resplitting, feature changes, calibration or threshold selection occurred.',
        markdown(artifacts['baseline_metrics'][['model','ROC-AUC','Average Precision (AP)','Brier Score','Accuracy','Recall','TN','FP','FN','TP']]),
        'Trace: `baseline_reproduction.csv`, `baseline_metrics.csv`, `audit_predictions.csv`. The protocol, training-only scales/maps, all child seeds, bounds, rounding, metrics and uncertainty estimands were saved in `protocol.json` and `protocol_rationale.md` before robustness results. The latter records the rationale and every scope limitation.',
        'The article-level motivation is comparison across trustworthiness dimensions. The package-defined SAFE operation is opposite-tail exchange on supplied evaluation inputs. Our semantic layer is supplementary domain-informed research design. The layers are not severity-equivalent, and neither is a measured error process.',
        '## Authentic SAFE implementation and validation',
        f'Installed safeaipackage {protocol["safe_identity"]["version"]}; robustness source SHA-256 `{protocol["safe_identity"]["robustness_source_hash"]}`. Installed commit metadata is unavailable. Core and utilities retain earlier verified source identities; this robustness module is identified by its inspected installed-file hash, with no upstream commit equality claim. Unmodified source is recorded in `safe_source_inspection.txt`.',
        'SAFE converts inputs with pandas.DataFrame and resets indices. perturb returns a reset-index frame and writes swapped values; independent deep copies isolate caller data. Python stable value sorting orders tails, then exchanges symmetric extreme pairs (upper tail reversed). For N=5,996 the nominal participation per feature is 118, 298, 598 and 1,198 at p=.01,.025,.05,.10 respectively, using min(ceil(pN), N−ceil((1−p)N)) pairs. Actual changes may be fewer for ties. Joint union counts are measured, not assumed to equal twice p.',
        'The actual compute_rgr_values API is invoked for every individual and grouped SAFE scenario. The classifier adapter forwards exactly the frozen ordered predictors and fitted preprocessing. Its predictions equal the frozen baseline; fitting raises. Its captured internal transformed inputs must exactly equal the inputs used for conventional diagnostics. Both models share those inputs. Original API arrays, reset indices and restored canonical IDs are verified. Individual calls use one variable at a time to isolate each scenario; grouped calls jointly transform the declared ordered group.',
        'core.rga computes rga(original default probabilities, perturbed default probabilities). It ranks the second vector with min ranks, replaces original scores tied on perturbed scores by their within-tie mean, sorts by perturbed scores and normalizes a weighted concordance sum between descending and ascending original magnitudes. Higher RGR means greater preservation; it is not simply rank-vector correlation. Model-specific normalization prevents treating an RGR difference as a common probability-error scale.',
        markdown(artifacts['safe_fixture_validation']),
        'Constant original scores give undefined RGR because inc−dec=0; these are retained as fixture findings. Constant perturbed scores give 0.5 for nonconstant original scores. Fixtures cover identity, preserved/reversed ordering, original/perturbed ties, both constant cases, authentic API calls and noncanonical indices. Runtime API/direct-core agreement is checked at 1e-14. Weighted AUC/AP bootstrap calculations separately agree with sklearn and expanded samples within 2e-14 (`weighted_ranking_validation.csv`).',
        '## Primary joint findings',
        '\n\n'.join(notes),
        markdown(safe_primary),
        f'GBM minus LR primary SAFE RGR difference: {rgr_diff:+.6f}. This must be interpreted with the following probability movement, flips and performance, since high RGR does not guarantee accurate probabilities, fairness, safety or general reliability.',
        markdown(table),
        'Each headline row identifies its saved scenario. Trace: `authentic_rgr.csv`, `diagnostic_summary.csv`, `conventional_diagnostics.csv`. Metric changes are perturbed minus baseline; paired differences are GBM minus LR. Higher AUC/AP/classification scores mean better recorded-outcome performance; higher Brier/log loss/ECE mean greater error; higher absolute movement and flips mean greater instability. Signed movement has no inherent favorable direction. Repetition metrics are calculated before averaging; probabilities are never averaged for primary nonlinear metrics.',
        markdown(primary_exposure[['scenario','metric','mean','mc_sd']]),
        'Exposure depends on changed inputs, never on whether a model prediction moved. The union row counts changed clients and changed cells separately. In union rows, clipping/outside-support/special-state counts sum feature cells; feature rows contain record counts. `exposure_counts.csv` contains selected, eligible, eligible-selected, changed, clipped, no-op, outside-support and special-state counts by feature/SEX/repetition; `exposure_summary.csv` reports repetition means and spread.',
        '## Paired sampling uncertainty and Monte Carlo variation',
        markdown(global_ci),
        'Trace: `sampling_intervals.csv` and `bootstrap_replicates.csv` also contain male and female intervals and within-resample subgroup counts. All intervals use 2,000 complete original predictor-group draws, seed 42, and 95% percentile bounds; both models and scenarios share draws. Cross-SEX/outcome groups are kept intact, so subgroup composition varies. Matching predictor inputs do not prove the same person. These are conditional intervals for frozen models and the observed SAFE tail swap or 30 saved semantic realizations. They do not estimate a repeated new-sample SAFE procedure. Repetition-wise nonlinear metrics and paired differences are recomputed within each resample and then averaged. No RGR interval or significance-testing grid is reported.',
        markdown(mc),
        'The semantic table gives between-repetition sample SD and range across 30 coupled realizations, separately from the sampling intervals. Deterministic SAFE has no Monte Carlo SD. Stronger semantic stress increases both selected fraction and magnitude; it is combined stress. Improvements and nonmonotonic changes are retained; monotonic deterioration is not assumed.',
        '## Descriptive SEX-group findings',
        markdown(artifacts['baseline_by_sex'][['model','SEX','group','records','default_rate','mean_probability','fraction_within_0.05_of_threshold','ROC-AUC','Average Precision (AP)']]),
        markdown(group_table),
        'Trace: `baseline_by_sex.csv`, `baseline_features_by_sex.csv`, `diagnostic_summary.csv`, `exposure_summary.csv`, and subgroup sampling intervals. SEX=1 male / SEX=2 female is unchanged metadata. All summaries subset globally perturbed inputs; SAFE tails are never recalculated within SEX groups. The tables distinguish unequal exposure, model response, baseline feature/score distributions and baseline proximity within .05 of the classification threshold. Their descriptive associations do not establish discrimination, causality, or replace completed fairness results.',
        '## Secondary order diagnostic and sensitivity analyses',
        f'The prespecified all-feature SAFE p=.05 permutation uses seed 20261008, aligned scores/metadata and the same input order for both models, with canonical ID order restored for comparisons. {order_exposure_difference} records differ in union exposure membership after permutation. Canonical order remains primary.',
        markdown(perm),
        'Trace: `row_order_diagnostic.csv`, `permutation_record_comparison.csv`, `realization_registry.csv`. Stable tie selection can affect tail membership and model response. Neither ordering is selected for a favorable comparison.',
        '![Joint severity](figures/robustness_evaluation/joint_severity.png)',
        '![Individual RGR](figures/robustness_evaluation/individual_rgr_primary_severity.png)',
        '![Feature and family sensitivity](figures/robustness_evaluation/feature_family_sensitivity.png)',
        'All 22 individual features and all eight SAFE groups run at each of the four severities. Grouped RGR is not an average of feature RGR. The semantic experiment covers every individual feature and the common-mask joint scenario at both levels and all 30 repetitions. All sensitivities, including null, adverse and unexpected results, remain in the saved tables. SAFE numeric category ordering is arbitrary; swaps preserve marginals but can break feature relationships and complete-record plausibility.',
        '## Semantic assumptions, exposure coverage and undefined quantities',
        f'Observed training supports include {category_support}. Positive ordinary repayment states receive exact ±1/±2 destinations only when training-supported; missing destination/endpoints are no-ops. Special -2,-1,0 states remain unchanged, limiting coverage of repayment-status errors. No special-state experiment was added. All per-feature destination maps are in the saved protocol.',
        'Monetary scales and category frequencies use only locked training records. Monetary changes are signed fixed IQR offsets, with integer NT-dollar rounding inferred from observed training precision. Nonnegative payments are bounded at zero, credit limits at the labelled training-support minimum; bills can be negative and monetary maxima are not capped. Age uses labelled training-support bounds. Selected baseline values outside a support bound stay unchanged. No-op reasons include special states, unsupported exact destinations, outside-support baselines, zero scale, rounding and clipping. Complete-holdout selection stays fixed regardless of eligibility.',
        f'All-record/changed-record undefined diagnostic entries: {int(undefined["count"].sum())}; undefined bootstrap metric replicates: {int(intervals.undefined_replicates.sum())}. Undefined fixture values: {int((artifacts["safe_fixture_validation"].status=="undefined").sum())}. Individual reasons and counts are in `undefined_results.csv`, `safe_fixture_validation.csv`, `changed_record_movement.csv` and `sampling_intervals.csv`. Empty calibration bins intentionally have missing means and contribute zero ECE; `calibration_bins.csv` retains their zero counts. Repository precision/F1 conventions report zero with no positive denominator.',
        '## Reproduction, provenance and limits',
        'Run `.venv/Scripts/python.exe script_robustness_evaluation.py --verify`. Two full evaluations include every realization and all bootstrap draws; numerical CSVs and the prediction cache are compared byte-for-byte. The complete existing and new test suite and pip check run afterwards. `verification.json`, `test_results.txt`, `dependency_check.txt` record results. Run timestamps may differ; all other manifest metadata must agree exactly. `run_manifest.json` identifies protected artifacts, source/package versions, protocol and result hashes. Earlier data/models/splits/reports/outputs/code/dependency locks remain protected. README receives only an additive stage note.',
        f'Artifacts: {len(protocol["scenarios"])} scenario IDs and {len(artifacts["realization_registry"])} saved realizations; compact `prediction_cache.npz` stores the shared baseline plus raw probabilities at saved record positions, input-change masks and storage masks in canonical ID order, keyed by `realization_registry.csv`. `src.robustness_cache.restore_probabilities` restores the exact original floating-point probabilities; storage masks also retain any numerical movement at unchanged inputs. The initial prespecified protocol is preserved in `initial_prespecified_protocol.json`; `implementation_revision.json` documents this storage-only revision with every analytical setting unchanged. Protocol parameters and child seeds reproduce perturbed inputs without full dataset copies. Primary bootstrap uses these cached original/perturbed pairs and never calls model prediction.',
        'These synthetic experiments do not establish real-world error rates, causal default-risk changes, adversarial security, future-population reliability, a universally best model, or framework novelty. Recorded outcomes remain fixed; they are not counterfactual outcomes for behavior changes. High ordering preservation can coexist with probability movement and classification errors. The historically inspected Taiwanese holdout and binary source SEX metadata constrain wider conclusions. The stronger and SAFE/semantic stresses are not severity-equivalent. No later stability, PSI, monitoring, mitigation or governance stage was implemented.'
    ]
    report.write_text('\n\n'.join(content)+'\n',encoding='utf-8')
