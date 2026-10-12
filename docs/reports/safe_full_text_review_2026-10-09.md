# Full-text SAFE literature review and revised research directions

Date: 9 October 2026 (Europe/Bucharest).
Status: completed document review and small mathematical definition checks; proposed research directions only. This is not authorization to implement an expanded framework, refit frozen models, change predictors/splits/thresholds, or run a new credit experiment.

## Outcome

The five full papers substantially sharpen the plan. Later SAFE already includes binary Brier-style assessment, direct perturbation analysis, feature-removal curves, statistical inference and composite scores. Our contribution cannot be the mere addition of these ingredients. The most useful research question is whether a SAFE assessment supports a defensible model-selection or acceptance decision under a specified operational fault.

Recommended sequence: first establish exactly what each comparator measures and passes elementary validity checks; then define minimum predictive/decision requirements and test an actual combination of feature reduction and a data-source fault. The parsimony/removal-of-backup-information hypothesis remains worth investigating, but it is conditional and untested. It should be situated within later SAFE research on conflicting perturbation-dependent model recommendations.

Three new full-text findings matter especially:
1. The integrated paper's input-independent random benchmark exceeds every trained model on all four aggregate scores in its California tail-swapping table. The authors explicitly exclude it from substantive model recommendations as a benchmark. This is not evidence that they recommend deploying random predictions. It demonstrates that a high scalar alone does not establish useful prediction or readiness.
2. Its proposed 3D aggregation combines separately computed curves, not measured predictions from every simultaneous feature-removal and corruption combination. A reduction/fault interaction can therefore be an informative evaluation target, if demonstrated fairly against an appropriate baseline.
3. The Wasserstein paper's printed uniform-coupling derivation is mathematically inconsistent with optimal transport. Correct marginal transport also has a different limitation: it can miss borrower-level changes that preserve the prediction distribution. These are separate issues, not reasons to dismiss optimal transport in general.

## Evidence and provenance

The user supplied five public full-text PDFs from Downloads. Unchanged copies, SHA-256 values, page counts and extracted text are retained under `docs/context/sources/literature_2026-10-09/`; see `reading_provenance.json`. Total: 108 PDF pages, including the Wasserstein publisher cover and review references. All substantive sections, appendices, conclusions and references were read. Fourteen selected equation/table/interpretation pages were rendered and visually inspected under `tmp/pdfs/literature_2026-10-09/`. Column-aware extracts were used for the two Elsevier papers because ordinary extraction interleaved columns. PDF page numbers below are one-based; the Wasserstein printed journal page is also provided when useful.

| Key | Bibliographic reference | Archived PDF | PDF pages |
| --- | --- | --- | --- |
| COR | Babaei, G., & Giudici, P. (2025). Correlation Metrics for Safe Artificial Intelligence. Risks 13(9), 178. DOI 10.3390/risks13090178 | risks-13-00178.pdf | 12 |
| RGB | Babaei, G., Giudici, P., & Raffinetti, E. (2025). A Rank Graduation Box for SAFE AI. Expert Systems with Applications 259, 125239. DOI 10.1016/j.eswa.2024.125239 | 1-s2.0-S0957417424021067-main.pdf | 12 |
| REV | Nastoska, A., Jancheska, B., Rizinski, M., & Trajanov, D. (2025). Evaluating Trustworthiness in AI: Risks, Metrics, and Applications Across Industries. Electronics 14(13), 2717. DOI 10.3390/electronics14132717 | electronics-14-02717-v2.pdf | 49 |
| WAS | Spelta, A., & Raffinetti, E. (2024). Evaluating SAFE AI principles using Wasserstein distance: a comparative study of Machine Learning models. Statistics 58(6), 1283-1303. DOI 10.1080/02331888.2024.2419420 | Evaluating SAFE AI principles using Wasserstein distance  a comparative study of Machine Learning models.pdf | 22 |
| INT | Giudici, P., & Kolesnikov, V. (2026). SAFE AI metrics: An integrated approach. Machine Learning with Applications 23, 100821. DOI 10.1016/j.mlwa.2025.100821 | 1-s2.0-S266682702500204X-main.pdf | 13 |

The 2026 journal year for INT differs from its December 2025 online publication date; RGB's 2025 journal year differs from its August 2024 online availability. All five supplied papers are open access; the earlier difficulty concerned automated retrieval, not evidence of subscription restrictions. Their earlier partial-access status is superseded by this review.

This review checked the current README, all four stage-report protocols, manifests, Git state, recovered context/chat provenance and earlier comparison/research memos. All 410 evaluation-manifest hash checks passed. Existing uncommitted work was preserved. The bundled Python PDF libraries were used only for document extraction/rendering; the project .venv performed the constructed-number checks. No credit dataset, fitted pipeline or prediction file was loaded by those checks.

## 1. COR: probability-sensitive SAFE is explicit, including classification

### What the full paper establishes

Section 2, PDF pp. 3-5, extends rank-graduation measures with predictive squared-error quantities. Equation (3), p. 4, explicitly addresses binary responses and predicted probabilities:

R2 = 1 - sum((y_i - p_i)^2) / sum((y_i - prevalence)^2).

For nondegenerate binary outcomes, this is 1 - Brier/[prevalence*(1-prevalence)], using the evaluation sample prevalence as the constant reference. This is an algebraic relationship, not a new empirical calculation on our data. Equations (4)-(6) also define prediction-change robustness, feature-removal comparison, and an absolute difference of group-specific predictive R2. The main application predicts salary growth; pp. 8-10 additionally benchmark salary doubling with logistic classification. Thus describing the paper as only applicable to regression would omit its explicit binary extension and example, despite regression-focused conclusions.

### Changes to our plan

- Drop any claim that SAFE as a literature ignores probability magnitudes or that adding Brier is our novelty.
- Retain reliability/ECE/log-loss diagnostics for interpretation. A squared-error score measures overall probability quality; it is not by itself a complete calibration/reliability assessment.
- For a probability-distortion demonstration, include this probability-sensitive baseline. A rank-only comparison would be deliberately incomplete.
- Group R2 gaps do not equal demographic parity, recall/FPR parity, or a general fairness guarantee. Each uses a distinct target and, in classification, a group-specific prevalence normalization.

### Implementation ambiguities and cautions

Equation (5) prints 1 minus normalized squared prediction change, yet the following p. 5 text treats zero as negligible feature contribution and higher values as more important. If removing an irrelevant feature leaves predictions unchanged, the printed formula gives one, which contradicts that importance interpretation. Tables 3 and 6 report high values for important features. This may be a sign/complement or reporting error; without implementation it does not identify which quantity generated those results. Equation (4)-(5) centering uses y-bar notation inherited from observed-response means; whether it is redefined for prediction vectors should also be resolved. Do not silently choose a corrected implementation and call it exact replication.

Predictive R2 can be negative, and its denominator can vanish. It is not automatically a [0,1] compliance component. A normalized squared change is not a frequency of operational failure. Table 1's mapping of score complements to likelihood should not be adopted as an estimated incident probability.

## 2. RGB: distinguish paper estimands from the installed package

### What the full paper establishes

Sections 2.1-2.2, PDF pp. 2-5, define the common concordance measure and its RGA/RGR/RGE/RGF uses. RGA equals AUROC for binary outcomes with the appropriate tie treatment. RGR compares clean predictions against perturbed predictions. RGE takes one minus the concordance between full- and reduced-model outputs. The text explicitly describes fitting models without predictors. RGF compares a full model including group predictors against a reduced model excluding a group predictor.

Section 2.3, pp. 5-6, already provides dependent U-statistic comparisons, jackknife variance estimation and asymptotic tests. The paper includes 1,000-repetition simulations, an employee salary/salary-doubling application, comparisons with conventional metrics and SHAP, and repeated-split variation. Statistical uncertainty and paired dependence are therefore not absent from SAFE until a new bootstrap extension.

The simulation robustness design perturbs training predictors, while the real-data application describes a different tail-replacement scheme. Definitions and empirical descriptions must be checked against any implementation before equating them to our fixed-model evaluation-input tail exchanges.

### Changes to our plan

- Keep separate: the original 2024 KAIRI article; this RGB paper; installed safeaipackage 0.8.3; COR; and INT. They do not define one interchangeable baseline.
- Our frozen-model mean/mode replacement RGE measures reliance under that replacement. It is not automatically the paper's refitted feature-exclusion estimand.
- Our SEX-group RGA parity is not the paper's protected-predictor deletion RGF. SEX remains metadata in our frozen 22-predictor study. Adding it to a separate research model would be a methodological change requiring later authorization.
- Preserve whole-predictor-group sampling in any proposed inference. Paper row jackknife or repeated-split variability is not automatically valid for our grouping convention or identical to conditional bootstrap uncertainty.
- Retain fairness equivalence/false-reassurance as a backup, but do not claim SAFE lacks inference. It concerns inference targets, power, materiality and interpretation.

## 3. INT: the closest score comparator, and a sharper research opportunity

### What the full paper establishes

Sections 2.1-2.4, PDF pp. 2-5, define accuracy-contribution removal curves, robustness curves for tail swapping and Gaussian noise, and reversed RGE feature-removal curves. AURGA, AURGR and AURGE summarize these curves. The score averages arithmetic, geometric or quadratic aggregations over a Cartesian product of three metric vectors; TOPSIS is a comparison. Table 10, p. 12, already explores unequal weights. Therefore adding curve areas, averaging the axes, or varying weights is existing work.

The applications use 2017 California/New York HMDA approval/denial outcomes and a synthetic XOR task. These are different targets from our next-month default. Tables 3-4, p. 7, show different architecture rankings for swapping and Gaussian noise. The authors explicitly identify selection across conflicting degradation scenarios as future research on pp. 9 and 12.

### New finding A: score versus predictive usefulness

Table 8, p. 9, gives the input-independent Random benchmark arithmetic score 0.76681, above LR 0.59695 and RF 0.63218; it also exceeds every trained model under geometric mean, RMS and TOPSIS. Table 9's random row remains highest for arithmetic/RMS, but LR exceeds it for geometric/TOPSIS. Tables 3-5 give Random perfect robustness and feature-removal preservation. The authors explicitly state on pp. 7-8 that Random is a benchmark, not an input-dependent predictive model, and do not recommend its deployment.

Supported inference: aggregate scores can reward invariance or dispensable inputs independently of useful prediction. A universal readiness interpretation needs explicit eligibility/minimum performance conditions, or an equivalent decision rule. This is not proof that the authors select a useless model, nor evidence that every trained model's ranking is wrong.

Proposed test: use input-independent controls to understand the mechanism, then examine plausible weak/strong predictive candidates under independently justified requirements. Demonstrating only that Random wins repeats a fact visible in this paper and would be a weak publication claim. A novel contribution would need meaningful model-selection consequences beyond that control.

### New finding B: constructed cross-products do not measure combined faults

For arithmetic aggregation, Equations (13) and (17) reduce exactly to:

V = [mean(RGA vector) + mean(reversed-RGE vector) + mean(RGR vector)] / 3.

This follows by summing the three terms over the Cartesian product. It does not measure interaction between noise and feature removal. Geometric/RMS combinations likewise synthesize separate values without generating predictions under every combined operation. The resulting histograms describe a constructed metric grid; they are not automatically sampling distributions, client-level distributions, or measured frequencies of joint operational events.

Research implication: feature reduction and source faults should actually be applied together when the hypothesis concerns removed redundancy. Compare assessment recommendations with observed loss on those joint scenarios. A cross-product score is a valid aggregation construction; it simply answers a different question from empirical resilience to simultaneous operations.

### New finding C: severity normalization is not threat equivalence

Section 2.2 rescales each raw perturbation range to [0,1]. Matching axis endpoints does not establish matching corruption severity, record exposure or operational likelihood. Tail swapping and Gaussian noise affect different records and variables differently. The paper's synthetic Gaussian mechanism is not evidence of our data's production error process. Our current SAFE p and semantic q must remain non-equivalent.

### Fairness and implementation limits

The operational aggregate uses three dimensions. Fairness is discussed through protected-variable explanation and future extensions, not an independent fourth score curve in Equations (13)-(17). Adding a standalone error-rate fairness gate might be useful for a specific task, but generic fairness additions already have extensive prior art. Reversed-RGE curve preservation should not be treated as validated human comprehension. AURGA is a constructed ranked-contribution curve rather than ordinary repeated AUROC after new sample selection/retraining. Full operational details of greedy removal, randomness, curve interpolation and TOPSIS normalization need resolution before exact reproduction.

The XOR experiment is a legitimate mechanism illustration, not a guarantee that the same difference holds on credit data. Our planned experiment needs independent evaluation and controls; manufacturing a task which directly rewards our chosen new metric is insufficient.

## 4. WAS: separate a printed mathematical error from an estimand limitation

### What the full paper establishes

Sections 2.2-3 develop a Wasserstein/optimal-transport alternative and simulations; Section 4 applies it to 3,453 Modefinance SMEs. The real application explicitly compares binary predicted labels rather than probabilities. Fairness shifts from maximum pairwise subgroup distribution distance in Equation (17) to predictor deletion in simulations and dropping companies in a NACE category in the SME application. These procedures have different estimands.

### Confirmed printed derivation issue

PDF pp. 7-8 (journal pp. 1288-1289), Equations (8)-(11), state that uniform marginals force every transport-plan entry to equal 1/T^2. This is false: uniform marginals constrain row/column sums but permit other couplings. A diagonal plan is feasible when the two supports coincide and achieves zero squared cost for identical values.

The printed Equation (11) is the average all-pairs half-squared difference, i.e. the cost under an independent coupling. It is not generally the minimum transport cost. For identical empirical distributions on {0,1}, it equals 0.25, whereas true optimal transport cost is zero. In general it equals half of [variance(a)+variance(b)+(mean(a)-mean(b))^2], with uniform empirical weighting. It therefore fails the identity property for nonconstant identical distributions and does not capture detailed shape beyond these moments.

The printed toy example, y1=[0,1] and y2=[0,2], also has a cost-matrix inconsistency and arithmetic error: (0+1+1+4)/(2*2^2) is 0.75, not its printed 0.375. The actual half-squared pairwise cost matrix is [[0,2],[0.5,0.5]], its independent-coupling cost is 0.75, its optimal half-squared transport cost is 0.25, and standard W2 squared is 0.5. These checks were independently computed with an assignment solver, not by copying the paper's formula.

An entropically regularized transport objective may have a nonzero self-cost; that is a different object. It does not make uniform marginals force an independent optimal coupling in the printed unregularized derivation. This critique concerns what is printed, not an inspected author code implementation. We have not reproduced their simulations or proved that the published result tables were generated by Equation (11).

### Correct transport still cannot replace matched prediction assessment

Correct empirical Wasserstein distance compares distributions and ignores record correspondence. Reordering the same set of predictions leaves the marginal distance zero even if clients receive different predictions. A two-record example with y=[0,1], p=[0.1,0.9] and reassigned p=[0.9,0.1] has zero prediction-distribution transport distance but Brier changes from 0.01 to 0.81. This is an elementary scope example, not a newly discovered statistical result or a realistic credit fault demonstration. It justifies reporting paired changes and outcome loss alongside any distribution metric.

Further full-text reporting questions: Figure 2 and Tables 10-11 report values in the hundreds of thousands despite Section 4's stated 0/1 inputs. Normalized squared transport on scalar [0,1] labels cannot have such values. An unnormalized computation or other undocumented input might explain this, but must be established before treating these as executable baselines. Footnote 1 compares predictions on companies remaining after category exclusion; the fitting/evaluation population and alignment need clarification. The application is not a clean direct counterpart to our current SEX-error-rate audit.

### Changes to our plan

Do not copy this paper's printed coupling shortcut. If this line becomes a comparator, distinguish three objects explicitly: the as-written formula; correctly solved distributional transport; and paired loss/change. A critique or corrected reproduction is a possible separate methodological note, but repairing a formula or showing permutation invariance is not sufficient evidence of a new general credit framework. This branch can be investigated without refitting our frozen models, but any application would still need its own authorized protocol.

## 5. REV: confirms scope limitations, not an empirical defeat of KAIRI

Section 5.1.4, pp. 24-25, explicitly discusses KAIRI's quantitative focus, accessibility/complexity, qualitative trust, and systemic biases in data/institutions. Section 5.2 reiterates broader governance/operational gaps. Sections 3.6, 7.2 and 8 discuss context-dependent trade-offs. The review also covers privacy, security, monitoring, human oversight and organizational accountability.

The review reports no newly created/analyzed data (p. 43); its case studies synthesize other sources. Thus it is background and a source of primary-reading leads, not an empirical proof of a SAFE failure. Its descriptions of regulations/standards and broad application claims require primary verification before being reused as current legal facts or causal evidence. We are not validating its entire bibliography by reading the review.

Practical consequence: use one clear research task and conclusion, with supporting metrics available underneath. Do not expand into privacy, organizational governance, LLMs, or intersectionality merely because the review lists them. The professor's concern about too many indicators is consistent with this accessibility discussion; our presentation tools address communication, but are not a validated scientific innovation.

## 6. Changes relative to the earlier plan

| Earlier candidate/claim | Updated position | Reason |
| --- | --- | --- |
| Add Brier/probability sensitivity as the main SAFE novelty | Drop as a standalone novelty claim; retain as necessary assessment/control | COR Eq. (3) explicitly handles binary probabilities |
| Add a single score, curve areas, weights or generic uncertainty | Drop as sufficient novelty | INT delivers curve/score/weight methods; RGB already includes inference; the separate 2026 uncertainty preprint remains prior context |
| Models rank differently under different perturbations | Supporting observation, not new contribution by itself | INT already demonstrates this and calls for selection across scenarios |
| Clean-performance-preserving simplification can remove useful backup information | Retain as conditional empirical hypothesis; test actual reduction/fault interactions | Original KAIRI parsimony plus INT's separate curves create a meaningful comparison question |
| Fairness false reassurance from nonsignificance | Backup; frame around a specific null, tolerance and power | Existing inference is substantial; equivalence foundations are established |
| Use Wasserstein as a stronger value-sensitive SAFE comparator | Audit before use; distinguish as-written and corrected methods | Printed coupling/formula issues and record-correspondence limitations |
| Comprehensive five-axis framework now | Remains deferred | Full texts make focus and explicit scope more valuable, not automatic expansion |

## 7. Revised proposed research program

This is a recommendation, not a selected or executed new protocol.

**Research question:** Can a SAFE assessment recommend or accept a credit-risk model that has attractive aggregate/simplification metrics but breaches an independently specified decision requirement under a justified data-source fault, and does an assessment with explicit constraints avoid that error?

**Stage A: comparator and score audit before new model work.** Establish a specification sheet for original KAIRI, relevant RGB/package conventions, COR probability criteria and INT aggregation. Preserve the original 2024 article as the professor-facing comparator, but compare any broad SAFE novelty claim with the later relevant methods too. Check identity, reversal, constants/ties, input-independent controls, score direction/range and paired versus distributional targets. Resolve paper ambiguities transparently rather than score an intentionally weak misimplementation. The small checks executed in this review are methodological reading aids, not completion of this broader audit.

**Stage B: declare independent criteria.** Define clean-performance adequacy and fault-loss tolerances before seeing new results. Use additional missed defaults/false alarms, Brier deterioration or an explicitly hypothetical FN/FP cost ratio, chosen for the actual question. Default flags are not approval decisions. Business costs and fault frequencies are not provided by the Taiwan benchmark. A promising model must satisfy the relevant minimum requirements; averaging other axes should not erase a breach. Such a constraint rule is established methodology, so novelty would require the specific design, analysis and demonstrated decision benefit.

**Stage C: one focused interaction experiment if authorized.** Develop an article-style reduction path on training/development data, preserve the existing frozen pipelines, and actually evaluate reduced and full candidates under one justified repayment-source availability/corruption family. Distinguish learning with fewer features, inference with unavailable features and mean/mode masking. Model-specific missing-input handling can determine the result; it must be specified rather than improvised. Use shared fault realizations, original predictor groups, and paired differences. Hold out fault severities/mechanisms or obtain independent data to test whether the extension generalizes beyond the criterion it was designed to satisfy.

**Stage D: compare decisions and allow failure.** Compare which model each faithfully specified method selects/accepts and whether that selection actually meets the independent criterion. Include benign controls and weak models; demonstrate added value on credible input-dependent candidates, not only random controls. If no meaningful reduction occurs, or full/reduced candidates fail equally, the parsimony premise is unsupported. If the later SAFE baselines already identify the failure, say so and refine or stop the candidate. Never force a material crossing by post hoc severity or tolerances.

For 16 October, a clear comparator map, the score-validity concern from the published tables, resolved mathematical issues, and an agreed experiment specification are defensible progress. The current review does not commit us to delivering a dramatic empirical advantage in a week.

## 8. Executed work and remaining uncertainties

Executed: archived and fully read five PDFs; extracted 108 pages; visually checked 14 pages; verified 410 saved evaluation hashes; checked uniform versus optimal coupling, the toy example, permutation blindness and arithmetic aggregation with constructed numbers in `check_literature_definitions_2026_10_09.py`. Results: `literature_definition_checks_2026-10-09.json`.

Not executed: author-code validation, full reproduction of any paper, new data-source fault experiments, reduced-model training, mitigation, threshold/calibration changes, expanded framework, or new literature-wide novelty search. Existing models/data/splits/predictions/dependency locks/results remain unchanged. No commit, push, publication or contact occurred.

Unresolved: exact author implementations and error corrections; relevant later work beyond these five papers; operational tolerances/fault mechanism; whether reduction actually sacrifices backup information; independent validation data; scope and publication novelty. The earlier uncertainty/conditional-fairness/error-burden papers are saved background, not part of the five-PDF full-text review.