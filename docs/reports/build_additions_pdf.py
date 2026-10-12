from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/pdf/credit_risk_additions_beyond_safe.pdf'
for name,file in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf'),('Arial-Italic','ariali.ttf'),('Georgia','georgia.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic',boldItalic='Arial-Bold')
INK=colors.HexColor('#162b46'); TEAL=colors.HexColor('#007f83'); BLUE=colors.HexColor('#235dba'); MUTED=colors.HexColor('#536376'); LINE=colors.HexColor('#d9e1ea'); PALE=colors.HexColor('#f0f5fb')
W,H=A4; WIDTH=W-96
styles={
'body':ParagraphStyle('Body',fontName='Arial',fontSize=10,leading=14,textColor=INK,spaceAfter=6),
'small':ParagraphStyle('Small',fontName='Arial',fontSize=8.6,leading=12,textColor=MUTED,spaceAfter=5),
'heading':ParagraphStyle('Heading',fontName='Georgia',fontSize=24,leading=30,textColor=INK,spaceAfter=14),
'sub':ParagraphStyle('Sub',fontName='Arial-Bold',fontSize=11,leading=14,textColor=INK,spaceBefore=7,spaceAfter=4),
'eyebrow':ParagraphStyle('Eyebrow',fontName='Arial-Bold',fontSize=8.8,leading=12,textColor=TEAL,spaceAfter=6),
'cell':ParagraphStyle('Cell',fontName='Arial',fontSize=9.1,leading=13,textColor=INK),
'cellhead':ParagraphStyle('Cellhead',fontName='Arial-Bold',fontSize=9.1,leading=13,textColor=INK),
'cover':ParagraphStyle('Cover',fontName='Georgia',fontSize=31,leading=38,textColor=INK,spaceAfter=17),
}
story=[]
def p(text,style='body'):return Paragraph(text,styles[style])
def add(text,style='body'):story.append(p(text,style))
def block(title,text):
    story.append(KeepTogether([p(title,'sub'),p(text)]))
def note(text):
    t=Table([[p(text)]],colWidths=[WIDTH]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PALE),('LINEBEFORE',(0,0),(0,0),3,TEAL),('LEFTPADDING',(0,0),(-1,-1),15),('RIGHTPADDING',(0,0),(-1,-1),13),('TOPPADDING',(0,0),(-1,-1),11),('BOTTOMPADDING',(0,0),(-1,-1),7)]));story.extend([Spacer(1,8),t,Spacer(1,9)])
def table(headers,rows,widths):
    data=[[p(escape(c),'cellhead') for c in headers]]+[[p(c,'cell') for c in row] for row in rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),PALE),('LINEBELOW',(0,0),(-1,0),1,LINE),('LINEBELOW',(0,1),(-1,-1),.5,LINE),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));story.extend([t,Spacer(1,12)])
def new(number,title,intro):
    if story:story.append(PageBreak())
    add('CREDIT RISK RESEARCH / '+number,'eyebrow');add(title,'heading');add(intro)
def source(text):
    story.append(Spacer(1,7));add('<b>Project evidence:</b> '+text,'small')
def footer(c,doc):
    c.saveState();c.setStrokeColor(LINE);c.setLineWidth(.6);c.line(48,46,W-48,46);c.setFont('Arial',8);c.setFillColor(MUTED);c.drawString(48,31,'Implemented additions | Credit risk fairness project | 8 October 2026');c.setFont('Arial-Bold',9);c.drawRightString(W-48,31,str(doc.page).zfill(2));
    if doc.page>1:
        c.setFont('Arial',8);c.drawString(48,H-33,'BEYOND THE PACKAGE METRICS');c.setFillColor(TEAL);c.drawRightString(W-48,H-33,'IMPLEMENTED SCOPE');
    c.restoreState()

# Page 1
add('PROJECT INVENTORY / 8 OCTOBER 2026','eyebrow')
add('Implemented additions<br/>beyond the SAFE metrics','cover')
add('A meeting reference for the comparison of logistic regression and gradient boosting on credit-default prediction.')
note('<b>What we have added:</b> an evaluation and reproducibility layer around authentic SAFE, covering validation, probability quality, decision fairness, SHAP explanations, semantic corruption and paired uncertainty.')
add('How to read this document','sub')
add('Here, <b>SAFE baseline</b> means the inspected rank-graduation implementation in <b>safeaipackage 0.8.3</b>: RGA, group RGA parity, RGE and RGR. The inventory identifies work beyond those package outputs and study-specific choices around their use.')
add('It does <b>not</b> claim these methods are absent from all SAFE/KAIRI publications, or that assembling them establishes a novel framework. Conventional metrics and Shapley analysis already overlap with the broader literature. [1-5]')
table(['SAFE component retained','Our implementation'],[
['Accuracy and fairness','Authentic RGA and group RGA parity, with source SEX codes retained.'],
['Explainability','Authentic individual and grouped RGE using training reference values.'],
['Robustness','Authentic individual and grouped RGR under the package tail swaps.']], [165,WIDTH-165])
add('The additions at a glance','sub')
add('<b>1.</b> Data repair and validation &nbsp; <b>2.</b> Conventional performance and calibration<br/><b>3.</b> Decision fairness and predictor associations &nbsp; <b>4.</b> SHAP analysis<br/><b>5.</b> Supplementary robustness and uncertainty &nbsp; <b>6.</b> Reproducibility and presentation tools')
add('Scope: completed work only. Frozen models, 22 predictors, a repaired 24,004 / 5,996 partition, and strict probability &gt; 0.50 reference decisions remain in place. Deferred work appears separately on page 8.','small')

# Page 2
new('01','Data validation and performance','These additions assess whether the comparison is traceable and reproducible, how well each model predicts, and how the fixed decision rule behaves. [1,2]')
block('Exact-predictor overlap audit and split repair','We audited several duplicate definitions, including exact matches on the actual 22 predictors, without ID, SEX or target. All 30,000 records remain. Each predictor group follows its lowest-ID member\'s original partition; 34 assignments changed and no exact predictor group crosses train/test. Conflicting outcomes remain. Matching inputs do not prove the same person.')
block('Training-only group-aware cross-validation','Five-fold StratifiedGroupKFold uses shuffle=True and seed 42. Both models share folds; preprocessing is fitted within each training fold. We save fold assignments, counts, all fold metrics, means and sample SDs, plus one out-of-fold prediction per training record per model. Temporary CV fits do not replace the frozen pipelines.')
block('A broader conventional metric panel','Saved confusion counts and curves accompany accuracy, balanced accuracy, precision, default recall, specificity, F1, false-positive/false-negative rates, predicted-default rate, ROC-AUC and average precision (AP). AP is sklearn average_precision_score, not trapezoidal PR-AUC. RGA numerically agrees with ROC-AUC in this binary task; it is not an independent extra signal here.')
block('Probability quality and calibration assessment','Brier score and log loss evaluate probability quality. Training OOF and locked-holdout reliability curves use 10 equal-width bins, saved bin counts, mean predictions, observed frequencies and count-weighted absolute-gap expected calibration error (ECE). We also compare mean probability with overall default frequency. No calibrator or pass/fail calibration tolerance was fitted or selected.')
block('Training-only threshold trade-offs and candidates','OOF scores support a 0.00-1.00 grid in 0.01 steps and empirical recall-target candidates of 40%, 50%, 60% and 70%. Full-precision boundaries and tie handling are retained. These are candidates, not selected operating thresholds; no holdout performance is claimed at those candidates.')
note('<b>Why it matters:</b> both models have about 81.8% holdout accuracy but detect only about 35% of defaults at 0.50. Boosting\'s clearer advantage is ranking and probability quality, rather than fixed-threshold accuracy. [2]')
source('outputs/stabilization_report.md; outputs/performance_evaluation_report.md; split_changes.csv; cv_fold_metrics.csv; oof_predictions.csv; recall_target_candidates.csv; calibration_summary.csv.')

# Page 3
new('02','Fairness beyond ranking parity','The conventional fairness audit complements SAFE group ranking accuracy by examining default-risk flags, missed defaults and false alarms at an explicit decision threshold. [3]')
table(['Added analysis','What was implemented'],[
['Group composition and outcomes','Counts, default frequencies and confusion matrices for SEX=1 male and SEX=2 female.'],
['Group performance panel','Flag rate, accuracy, balanced accuracy, precision, recall, specificity, FPR, FNR, negative predictive value (NPV) and F1.'],
['Demographic parity','Signed flag-rate gap, unsigned max-minus-min gap, female/male ratio and min/max ratio.'],
['Equal opportunity and equalized odds','Signed default-recall gap; separate TPR/FPR gaps; maximum of their absolute gaps.'],
['Signed comparisons and uncertainty','Female-minus-male differences alongside absolute gaps, ratios and conditional group-bootstrap intervals.']], [155,WIDTH-155])
block('Descriptive holdout threshold sensitivity','A separately authorized grid from 0.10 to 0.70 in 0.05 steps shows how overall and group rates, gaps and their directions change. It is exploratory sensitivity only: the threshold stays 0.50, and the grid is not mined for an optimal or favorable operating rule.')
block('Training-only predictor associations','Numeric inputs have signed and absolute standardized mean differences, plus group means, medians and quartiles. Categorical inputs have uncorrected Cramer\'s V and saved category counts/fractions. We do not pool these different effect-size scales into one ranking, train a proxy classifier, or use p-value screening to remove inputs.')
block('Conventional versus SAFE comparisons','The report places threshold-specific disparities beside group RGA/parity. Boosting has a smaller SAFE absolute ranking-parity gap, but larger absolute recall and FPR gaps at 0.50. The selected metric changes the model comparison; no global fairness winner is declared.')
note('<b>Interpretation:</b> a positive prediction means flagged as likely to default, not approved for credit. Excluding SEX does not prove fairness. Descriptive associations and group gaps do not establish causal discrimination, and an interval containing zero does not establish equivalence.')
source('outputs/fairness_evaluation_report.md; group_metrics_threshold_050.csv; fairness_gaps_threshold_050.csv; threshold_fairness_sensitivity.csv; proxy_descriptive_associations.csv; conventional_vs_safe_comparison.csv.')

# Page 4
new('03','SHAP explanation analysis','SHAP adds score-level and record-level explanations alongside SAFE RGE. This is complementary analysis, not a claim that Shapley methods are new to SAFE/KAIRI. [4,5]')
block('Model-specific explainers and training reference','LinearExplainer with an Independent masker explains logistic regression; interventional TreeExplainer explains boosting. Both use one seeded 100-record training-only background and explain default log-odds for every one of the 5,996 holdout records.')
block('Encoded-feature mapping and reconstruction','SHAP is computed on 85 transformed columns. Signed one-hot contributions are summed into 22 original predictors before absolute importance is calculated; transformed values and mappings are retained. Additivity is checked against the model margin, and probabilities/predictions stay unchanged. Boosting aggregation is not a separately computed original-feature coalition game.')
block('Global importance and effect displays','Saved bar charts, beeswarms and dependence plots show mean absolute contributions, signed means, importance shares and rankings. We calculate top-one, top-three and top-five concentration. Dependence plots retain raw values and categorical meanings; plotted associations are not causal effects.')
block('Prespecified individual examples','For each model, the lowest original ID in each TP/TN/FP/FN cell is selected. Their union is explained with both models; in this run the union contains four records. All 22 contributions and input values are saved, alongside baseline log-odds and reconstructed default probabilities. These are illustrative examples, not a representative sample.')
block('Descriptive SEX-group explanations','We compare group mean absolute and signed SHAP values, normalized importance shares, ranks and female-minus-male differences. Rank correlation and total variation summarize group patterns without a materiality cutoff or discrimination claim.')
block('SHAP versus authentic SAFE RGE','Individual importance ranks are compared using Spearman correlation and top-five overlap. The saved model summary shows SHAP/RGE rank correlations of 0.918 for logistic and 0.927 for boosting. The two methods quantify different objects: additive score contributions versus ordering change under reference replacement.')
note('<b>Boundary:</b> the requested paper Table-2-style Gini and Kolmogorov-Smirnov tests on group Shapley values have not been implemented. Current group SHAP comparisons are descriptive. No explanation confidence intervals were calculated.')
source('outputs/explainability_evaluation_report.md; global_shap_importance_*.csv; local_case_selection.csv; local_shap_contributions.csv; sex_shap_comparison.csv; shap_vs_rge.csv; shap_validation.csv.')
# Page 5
new('04','Supplementary robustness experiments','The study adds severity and scope choices around authentic SAFE: p=0.01/0.025/0.05/0.10 across 22 features and eight groups, with joint p=0.05 primary. The separate semantic layer below uses our prespecified input-error rules, not the package RGR operation. [6]')
table(['Input family','Primary semantic error','Stronger error'],[
['Credit limit, bills, payments','Random signed 0.10 x training IQR','Random signed 0.20 x training IQR'],
['Age','+1 or -1 year','+2 or -2 years'],
['Ordinary repayment delays','Exact supported +1 / -1 month','Exact supported +2 / -2 months'],
['Education and marriage','Different valid category, drawn from training frequencies','Same substitution rule; larger record fraction']], [128,185,WIDTH-313])
block('Selection and repeated realizations','Uniform masks select floor(q x N) records from the complete holdout without using outcome or SEX, with q=0.10/0.20. All 22 individual-feature scopes and one joint scope run at both levels, with 30 repetitions. Joint corruption uses a common mask across predictors.')
block('Coupling, semantics and bounds','Deterministic child seeds from master seed 20261007 nest primary masks within stronger masks and reuse signs. Signs vary across records and features. Special repayment codes -2, -1 and 0 stay unchanged; unsupported destinations are no-ops. Training-support age/credit-limit bounds and integer rounding affect selected eligible cells only. Payments stay nonnegative, bills may be negative, and monetary maxima are uncapped.')
block('Conventional diagnostics on both layers','The exact same inputs feed both models and conventional diagnostics alongside SAFE RGR: signed, mean/median/p95 absolute probability movement; Spearman score correlation; 0.50 classification flips; ROC-AUC/AP; Brier/log loss; classification rates, confusion counts and calibration summaries. All-record movement is accompanied by labelled changed-record supplements. Changes are perturbed minus baseline.')
block('Exposure and descriptive subgroup context','We save eligible, selected, changed, clipped and no-op counts by input and SEX, including union-of-changed-records exposure. SEX summaries subset the globally perturbed records. Baseline feature/score distributions and proximity to 0.50 help separate unequal exposure from unequal model response.')
note('<b>Scale of the implemented study:</b> 120 canonical SAFE scenarios, one secondary order diagnostic, and 46 semantic scenarios x 30 runs: <b>167 scenarios and 1,501 realizations</b>. Primary joint exposure is 65.1% for SAFE versus about 10% for semantic corruption; these are not equally severe stresses.')
source('outputs/robustness_evaluation_report.md; protocol.json; conventional_diagnostics.csv; changed_record_movement.csv; exposure_summary.csv.')

# Page 6
new('05','Uncertainty and diagnostic checks','Beyond SAFE point estimates, the project defines conditional sampling uncertainty, paired model comparisons and separate Monte Carlo variability. [2,3,6]')
block('Whole-predictor-group bootstrap','Across performance, fairness and primary robustness analyses, 2,000 complete exact-predictor-group resamples use seed 42 and 95% percentile intervals. Every sampled group retains all its records and its multiplicity. Both models share draws. Groups spanning SEX or outcomes remain intact; subgroup composition can vary. This is conservative dependence handling, not proof of independent customers.')
block('Performance and fairness uncertainty','Performance intervals cover the holdout metric panel. Fairness intervals cover group rates, gaps/ratios and actual SAFE group RGA, including signed and absolute ranking gaps. Those stages did not conduct a paired model-difference test. Fairness intervals are marginal and unadjusted; signed-gap direction and absolute-gap construction are distinguished.')
block('Paired robustness differences','The two prespecified primary joint scenarios receive model-specific and GBM-minus-LR intervals for mean absolute movement, flips, and changes in AUC, AP and Brier score. SAFE intervals use saved original/perturbed score pairs. Semantic intervals condition on 30 saved realizations and recompute each repetition\'s metrics before averaging. No averaging of probabilities or concatenation of runs as new clients occurs.')
block('Monte Carlo spread kept separate','Semantic summaries include between-repetition sample SD, minimum and maximum over the 30 runs. This quantifies random-corruption variability, not the sampling interval. Deterministic SAFE scenarios have no Monte Carlo SD. The stronger semantic level raises both record fraction and magnitude, so it is combined stress.')
block('Secondary row-order diagnostic','One prespecified all-feature SAFE p=0.05 permutation uses seed 20261008. Inputs, metadata and original scores move together; canonical ID order is restored for comparisons. The report compares exposure, RGR, movement and performance. Ties produce differences in union exposure membership for 1,225 records. Canonical order remains primary.')
block('Undefined cases and numerical validation','Undefined metrics retain reasons and valid replicate counts. SAFE fixtures cover identity, reversed ordering, ties, constants and index alignment. Optimized weighted AUC/AP calculations are validated against sklearn and explicitly expanded whole-group resamples. No RGR sampling intervals or exhaustive significance-test grid were added.')
note('<b>What the intervals mean:</b> they condition on frozen models, observed predictor groups and, for robustness, saved corruptions. They do not include retraining, future population shift, or re-running the data-dependent SAFE tail procedure on new samples.')
source('holdout_confidence_intervals.csv; fairness_confidence_intervals.csv; sampling_intervals.csv; paired_summary.csv; row_order_diagnostic.csv; safe_fixture_validation.csv; weighted_ranking_validation.csv.')

# Page 7
new('06','Reproducibility and presentation','These are implementation and communication additions. They make the empirical evidence inspectable; they are not new trustworthiness dimensions or metrics. [1-4,6,7]')
block('Traceable data and complete frozen pipelines','Original IDs remain in cleaned data, split metadata and predictions, while ID and SEX remain outside model inputs. Shared preparation, model and metric utilities replace duplicated logic; scripts resolve paths from their own locations. Fitted preprocessing and classifiers are saved together, with reload/prediction checks. Historical baseline artifacts are preserved.')
block('Locked inputs and baseline reproduction','Data, split, model, predictor order, class order and saved predictions are checked before later evaluation. Robustness reproduces baseline probabilities against all preceding stages and checks strict p &gt; 0.50 tie handling. A material mismatch stops evaluation rather than silently fitting or resplitting.')
block('Prespecified configuration and provenance','Robustness protocol and rationale were saved before results. Seeds, transformations, feature groups, scales, supports and uncertainty estimands remain machine-readable. SHA-256 manifests identify inputs, protected artifacts, source code, outputs and the inspected SAFE implementation. Dependency extensions preserve earlier locks; environment versions are recorded.')
block('Reproducible outputs and focused verification','Separate stage folders hold CSVs, reports, figures and verification records; notebooks display completed results. The latest saved stage verification records 30 passing project tests and a successful dependency check. Two finalized robustness runs reproduce all 24 numerical CSVs and the prediction cache byte-for-byte. PNG/report equality and normalized SVG equality are documented.')
block('Compact prediction cache and honest reporting','The robustness cache stores exact paired probabilities, record identities and change/storage masks in canonical order. Bootstrap reuses predictions rather than scoring repeatedly. A storage-only revision is documented with analytical choices unchanged. Reports retain null, adverse and unexpected findings, link headline numbers to scenario IDs/tables, and state limitations.')
block('Offline evidence explorer and persistent context','A single English HTML file embeds saved results from 28 identified source artifacts. It supports metric/model/sample selectors, saved SHAP cases, 166 canonical scenario views, the secondary order comparison, definitions, intervals and CSV exports. It needs no network or database and refreshes only through its build script. Project context, instructions and recovered documents preserve research decisions for future work.')
note('<b>Current checks:</b> 410 saved evaluation-manifest fingerprints and all 28 embedded-source fingerprints matched when the explorer was packaged. Direct-file browser checks passed without network requests or JavaScript errors. These checks supplement earlier stage verification; they do not constitute a fresh rerun of every research experiment.')
source('models/*_manifest.json; data/processed/preparation_manifest.json; stage run_manifest.json and verification.json files; explorer/README.md; explorer/source_manifest.json; PROJECT_CONTEXT.md; AGENTS.md.')

# Page 8
new('07','Scope boundaries and evidence','The additions above are implemented. The broader proposal remains a separate, deferred research plan, subject to review of the current methodology.')
table(['Not implemented','Current boundary'],[
['PSI and temporal monitoring','No PSI/stability stage, temporal drift study or live monitoring system.'],
['Formal paper Table-2 replication','No Gini/KS significance analysis on group Shapley distributions. Descriptive SHAP group comparisons exist.'],
['Model intervention','No selected replacement threshold, fitted calibrator, fairness mitigation or new feature-selection experiment.'],
['Expanded indicator battery','No MCS, energy/MMD testing, adversarial-security study, explanation fidelity/sanity-check battery, conformal prediction or aleatoric/epistemic decomposition.'],
['Broader validation and claims','No new Romanian credit dataset, formal intersectional study, validated regulatory mapping, deployment approval, composite trustworthiness score or demonstrated framework novelty.']], [157,WIDTH-157])
add('Evidence index','sub')
for title,path,description in [
('[1] Baseline stabilization','outputs/stabilization_report.md','Duplicate audit, split repair, pipeline persistence and historical control comparisons.'),
('[2] Performance evaluation','outputs/performance_evaluation_report.md','CV, OOF thresholds, conventional metrics, calibration and holdout intervals.'),
('[3] Fairness evaluation','outputs/fairness_evaluation_report.md','Conventional disparities, sensitivity, descriptive associations and uncertainty.'),
('[4] Explainability evaluation','outputs/explainability_evaluation_report.md','SHAP formulation, cases, SEX summaries, SAFE comparison and reconstruction checks.'),
('[5] Recovered project brief','docs/context/sources/Brief_Python_SAFE_Table2_PSI_EN (1).docx','Original SAFE/Table-2/PSI brief; some proposals remain unimplemented.'),
('[6] Robustness evaluation','outputs/robustness_evaluation_report.md','Protocol, diagnostic tables, conditional uncertainty and provenance.'),
('[7] Offline explorer','explorer/README.md','Interface scope, source registry, rebuilding and browser verification.')]:
    add('<b>'+title+'</b> &nbsp; '+description,'small');add(escape(path),'small')
add('Companion machine-readable evidence sits under outputs/tables/&lt;stage&gt;_evaluation/. This inventory summarizes saved executed work as of 8 October 2026; document preparation did not run new analyses or alter protected artifacts.','small')
add('<b>Suggested meeting wording:</b> "We implemented authentic SAFE and added conventional evaluation, SHAP, semantic input-error sensitivity and reproducibility safeguards. We are seeking feedback on this empirical foundation before extending the proposed framework."','small')

doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=53,bottomMargin=58,title='Implemented additions beyond the SAFE metrics',author='Credit risk fairness research project',subject='Inventory of completed complementary analyses and implementation safeguards',pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)