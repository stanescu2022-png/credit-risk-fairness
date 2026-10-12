"""Package saved research evidence as a portable offline HTML explorer."""
from pathlib import Path
import csv, hashlib, json, math
ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sources = []
def table(key, stage, file):
    p = ROOT / 'outputs/tables' / (stage + '_evaluation') / (file + '.csv')
    sources.append({'key': key, 'path': p.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    rows = []
    with p.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                if v == '': row[k] = None
                else:
                    try:
                        n = float(v)
                        row[k] = n if math.isfinite(n) else None
                    except ValueError: pass
            rows.append(row)
    return rows
D = {}
for key, stage, file in [
('performance','performance','holdout_metrics'),('performanceCI','performance','holdout_confidence_intervals'),
('cv','performance','cv_summary_metrics'),('calibration','performance','calibration_bins'),('calibrationSummary','performance','calibration_summary'),
('fairness','fairness','group_metrics_threshold_050'),('fairnessCI','fairness','fairness_confidence_intervals'),
('fairnessComparison','fairness','conventional_vs_safe_comparison'),('rga','fairness','safe_rga_results'),
('shapLR','explainability','global_shap_importance_logistic'),('shapGBM','explainability','global_shap_importance_gradient_boosting'),
('shapSummary','explainability','model_summary'),('localCases','explainability','local_case_selection'),
('localSHAP','explainability','local_shap_contributions'),('groupedRGE','explainability','safe_grouped_rge'),
('sexSHAP','explainability','sex_shap_comparison'),('robustnessCI','robustness','sampling_intervals'),
('rgr','robustness','authentic_rgr')]:
    D[key] = table(key, stage, file)
focus = ['mean_absolute_movement','median_absolute_movement','p95_absolute_movement','flip_rate','input_exposure_rate','delta_ROC-AUC','delta_Average Precision (AP)','delta_Brier Score','delta_Log Loss','Spearman']
# Embed only display metrics, retaining saved means and Monte Carlo SD separately.
D['robustness'] = [r for r in table('robustness','robustness','diagnostic_summary') if r['metric'] in focus]
D['exposure'] = [r for r in table('exposure','robustness','exposure_summary') if r['feature'] == '__union__' and r['metric'] == 'changed_rate']
D['paired'] = [r for r in table('paired','robustness','paired_summary') if r['metric'] in focus]
D['changed'] = table('changed','robustness','changed_record_movement')
# Changed-record table is large: restrict to the primary joint scenarios.
D['changed'] = [r for r in D['changed'] if r['scenario'] in ['safe__group_all_features__p_0.05','semantic__all_features__primary']]
p = ROOT / 'outputs/tables/robustness_evaluation/protocol.json'
protocol = json.loads(p.read_text(encoding='utf-8'))
D['protocol'] = {k: protocol[k] for k in ['protocol_date','predictor_order','groups','scenarios','primary_scenarios','semantic_rules','master_seed','coupling','bootstrap','classification']}
sources.append({'key':'protocol','path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
D['verification'] = {}
for stage in ['performance','fairness','explainability','robustness']:
    p = ROOT / f'outputs/tables/{stage}_evaluation/verification.json'
    v = json.loads(p.read_text(encoding='utf-8'))
    D['verification'][stage] = {k: v[k] for k in ['tests_passed','unit_and_artifact_tests_passed','csv_count','two_full_runs_csvs_byte_identical','csv_results_reproduced_on_rerun','protected_artifacts_unchanged','pip_check_passed'] if k in v}
    sources.append({'key':stage+'Verification','path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
D['sources'] = sources
D['order'] = table('order','robustness','row_order_diagnostic')
D['snapshot'] = '8 October 2026'
payload = json.dumps(D,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/')
html = (HERE/'template.html').read_text(encoding='utf-8').replace('__DATA__',payload).replace('__APP__',(HERE/'app.js').read_text(encoding='utf-8'))
(HERE/'credit-risk-explorer.html').write_text(html,encoding='utf-8')
(HERE/'source_manifest.json').write_text(json.dumps(sources,indent=2)+'\n',encoding='utf-8')
print(f'Packaged {len(sources)} source artifacts into {len(html.encode("utf-8")):,} bytes.')
