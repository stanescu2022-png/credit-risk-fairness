# A focused SAFE/KAIRI comparison: plan for review on 10 October 2026

Prepared 9 October 2026. Status: proposed design for discussion with Andrei; no new credit experiment or methodological implementation authorized or executed. The existing frozen study remains the historical foundation. This plan can be combined with the user's ideas before any experiment starts.

## The recommendation in one paragraph

Build a fault-aware model-assessment and selection procedure that asks whether a credit-default model remains predictively useful under a declared information failure, while satisfying explicit clean-performance and decision-error requirements. Compare it with a faithful adaptation of the original KAIRI selection procedure, the later integrated SAFE score, and simple strong alternatives. All methods use the same candidate models and validation resources. Superiority means that the model our procedure selects has materially lower loss on separate evaluation data, with acceptable trade-offs. It does not mean that our dashboard awards itself a higher trustworthiness number.

## 1. What we are trying to demonstrate

The mechanism hypothesis is that removing predictors which appear redundant under clean conditions can remove backup information needed under a particular information fault. The method hypothesis is that explicitly evaluating candidates under that fault leads to better selection than clean-performance parsimony or a compensating aggregate of separate indicators.

The desired result is specific: 'Under this declared input-information failure, our assessment selected models with lower held-out probability loss and acceptable clean performance, compared with these specified SAFE/KAIRI procedures.' Broader claims require broader evidence. We cannot promise that the hypothesis will work on Taiwan, or that the first implementation will be novel enough for publication.

Two predictions make the idea falsifiable:
- At least one reduced model must retain approximately the clean predictive quality of its full-model reference but deteriorate more when the declared input information fails.
- Our selection must improve performance on observations and fault realizations not used to make that selection. If a simple fault-aware selector does equally well, the useful contribution is primarily fault-aware assessment, not our additional score machinery.

## 2. The score: useful prediction under the worst declared scenario

Use Brier loss as the first primary objective because the target is default probability and no bank costs have been supplied:

L_s(m) = average over records and saved fault realizations of (predicted_default_probability - observed_default)^2.

Compute loss separately for every realization, then average losses. Do not average probabilities before squaring. Let scenario 0 be clean data, and let the remaining scenarios be prespecified input-information faults. Define:

R(m) = max over declared scenarios s of L_s(m).

Lower R is better. It is a worst-scenario expected loss over the declared finite scenario set, not an incident probability, universal adversarial guarantee, or worst possible individual-client loss.

For a readable reference, fit a constant predictor using prevalence in the fitting partition only. Let L_ref be its evaluation Brier loss. Its predictions do not change when inputs are corrupted, so its loss is the same across these label-preserving scenarios. Display:

S(m) = 100 * [1 - R(m)/L_ref].

A score of 0 means no improvement over that reference; 100 would mean perfect probabilities in every tested scenario; negative scores mean worse than the reference. Do not clip negative scores, and do not present S as a percentage of legal compliance or all-round trustworthiness. Within one evaluation population it is a monotone presentation of R, not a new statistical invention.

A weak model cannot become useful merely by being insensitive to inputs: unchanged poor predictions still have poor Brier loss. This addresses the distinction between prediction stability and prediction quality.

## 3. Conditions that the score cannot average away

The reported result has two parts: an adequacy status and a score. A score alone does not authorize acceptance.

Recommended conditions, to agree before seeing new results:
1. Clean Brier loss must be within a declared noninferiority margin of a specified full-model reference.
2. Worst-scenario predictive skill must exceed an agreed minimum relative to the constant reference.
3. Fixed-threshold decision errors must respect any agreed recall/FPR requirements. Strict probability >0.50, ties class 0, remains the reference rule.
4. If fairness is part of the claimed assessment, declare one relevant group-error requirement and its tolerance. Do not require every incompatible fairness definition at once.

Do not invent acceptable margins or bank costs from our existing results. The technical protocol supplies sensitivity ranges and a place to record the chosen policy. If no policy is agreed, we can compare losses and tolerance trade-offs, but cannot issue a defensible global pass/fail verdict. Missing evidence is labelled 'not assessed'.

For each requirement, use a simultaneous uncertainty bound: supported if the bound establishes the requirement, violated if it establishes a breach, and inconclusive otherwise. The newer SAFE uncertainty literature already has uncertainty-aware decisions; our distinction must come from the assessment target and demonstrated selection consequences.

Among eligible candidates, choose the smallest estimated R. If several candidates are within an agreed practical risk-equivalence margin, prefer fewer original predictor groups. This makes parsimony a tie-breaker after adequate prediction, rather than an assumption that fewer variables imply robustness. Feature count is not carbon emissions or validated human comprehension.

## 4. How we will measure whether the methodology is better

Freeze each method's selected model before the audit/evaluation partition is used. On that common partition calculate:

G = R(model selected by comparator) - R(model selected by our method).

Positive G favors our method. This is the primary comparison quantity. Report its uncertainty and compare it with a prespecified minimum meaningful gain delta_material. Also report relative reduction in loss for communication, but keep the absolute difference primary.

Separately show clean loss, recall, FPR, subgroup results, candidate complexity and the selection decision. A reduction in the main loss accompanied by an unacceptable clean or decision-error deterioration is not success. Prediction flips alone are not success: some flips correct mistakes.

A method that refuses to select/accept a model cannot be awarded zero loss. Report its unresolved outcome and acceptance rate; compare reliability at matched acceptance coverage or under an agreed fallback. This prevents a very cautious method from 'winning' by abstaining everywhere.

## 5. Compare against methods that could actually challenge us

| Comparator | Purpose |
| --- | --- |
| Original KAIRI black-box selection adaptation | User's chosen primary article comparator: Shapley-ordered forward inclusion with paired AUC stopping logic; state adaptation details and ambiguous conventions |
| Best clean-validation Brier candidate | Tests whether probability assessment alone is sufficient |
| Integrated SAFE arithmetic score | Closest delivered score comparator; include its native perturbation design and an equal-information fault variant |
| Simple minimax Brier selector | Tests whether direct fault-aware loss alone accounts for the benefit |
| Our score with safeguards and uncertainty | Proposed full assessment procedure |

The same candidate pool, fitting data, hyperparameters, fault draws and resource budget apply. For the original article, the initial primary comparison should stay within one black-box family so we compare assessment rules rather than an arbitrary rule for choosing between architectures. GBM is an adaptation of its random-forest example; an RF arm is a later fidelity check if authorized.

We need both native-information and equal-information comparisons. The native version asks whether our operational fault assessment adds useful information to the paper's procedure. The equal-information version gives later SAFE access to the same development fault family and asks whether the decision rule still matters. We must not conceal that additional fault knowledge, rather than scoring sophistication, might explain the gain.

The integrated paper's input-independent benchmark is a useful diagnostic control, not our principal empirical opponent. The Wasserstein derivation issue is a separate possible critique; beating a misprinted formula would not establish superiority over SAFE.

## 6. The first fault family and model pool

Start with corrupted or unavailable latest repayment status (PAY_0). This is a defined input-record/measurement failure, not a claim that the borrowers' behavior changes or that these fields come from independent production systems. Use training-only categorical mode replacement as a transparent baseline handling rule; document no-ops when a selected value already equals that mode.

Recommended development levels are affected-record fractions q=.10 and .20. The audit can include new realizations at those levels, interpolation q=.15, extrapolation q=.30, and a stale-report variant replacing PAY_0 with the previous month's PAY_2 on selected records. The latter tests a different mechanism; it must not be silently assumed equivalent in severity to missingness. Complete field outage q=1 is an explicitly extreme diagnostic, not a primary setting chosen to manufacture a large benefit.

An additional all-repayment-status block fault tests whether any observed backup benefit disappears when its entire information family fails. If so, that supports a limited source-specific interpretation rather than universal robustness. Benign/irrelevant-feature controls are also required.

For a fast pilot, compare top-k original predictor subsets for k in {1,2,4,6,8,12,16,22}, generated using a full model fitted on fitting data only. Keep model-family settings fixed. For faithful sequential KAIRI evaluation, extend to every k=1,...,22; a coarse path changes the stopping rule and cannot be labelled exact reproduction. Keep encoded levels of a predictor together. The existing frozen 22-predictor artifacts are never overwritten.

## 7. A small, auditable first implementation

Inside the existing 24,004 training records, reserve whole original predictor groups into fitting, selection and audit partitions, approximately 60/20/20. Preserve the original 5,996 holdout as a separate historically inspected reference. This is a proposed internal development split, not a replacement of the saved train/holdout partition.

- Fitting data: fit fresh extension candidates, rank features, set preprocessing and fault replacement values.
- Selection data: calculate comparator indicators, select models, and estimate development trade-offs.
- Audit data: evaluate the fixed selected models with new fault realizations and held-out mechanisms/severities. Do not use it to fix thresholds, alter features, replace candidates or choose a more favorable scenario.

Do not refit models after the audit and retain the old verdict. If we later refit on more data, that model needs new evaluation. Neither the training data nor the existing holdout is historically untouched at project level; this first study is an exploratory, prospectively separated comparison. A publication-strength confirmation should use fresh simulation replications and preferably an independent credit dataset under a frozen protocol.

The first implementation should stop at the smallest informative experiment. Only if the premise and selection difference survive should we add repeated/nested evaluation, stronger fairness constraints, additional datasets or other framework modules.

## 8. A staged go/no-go plan for the week

| Stage | Concrete output | Stop or change direction when |
| --- | --- | --- |
| Agree design | Comparator specification, score definition, fault meaning, margins and claim scope | The comparison cannot be made fair or the operational target remains unclear |
| Verify mechanics | Identity/tie/constant controls; subset and fault audit; protected hashes | A metric is undefined/misimplemented or exposure is not comparable |
| Pilot mechanism | Clean-versus-fault table for full/reduced candidates | No clean-adequate reduction exists, or useful redundancy is not removed |
| Compare selection | Fixed choices from each method; held-out loss and decision errors | Methods select the same candidate or simple minimax explains all benefit |
| Confirm limited claim | Prespecified material gain with acceptable trade-offs and sensitivity checks | The gain disappears under new realizations/mechanisms or independent replication |

A negative result is a decision to stop or refine a candidate, not permission to search the holdout until a dramatic example appears.

## 9. The decisions to combine with your ideas tomorrow

The important discussion is not which twenty new metrics to add. It is:
- Is model selection under a repayment-information failure the research task we want?
- Should probability loss be the primary objective, or do we have credible decision costs that justify a different one?
- What clean-performance sacrifice and minimum improvement are meaningful?
- Which decision-error/fairness requirements are essential, and which are supporting descriptions?
- Is the score a model-ranking aid, an acceptance procedure, or both? Keep their evaluation separate.
- Do we want a focused empirical SAFE extension, or a separate methodological critique of the Wasserstein construction?

My proposed first direction is the focused empirical extension. The scoring rule organizes it clearly; its scientific value must be demonstrated through better decisions on separate data.

## Reading next

The implementation and inference details are in [the technical protocol](safe_comparison_technical_protocol_2026-10-09.md). The score and comparison quantities are illustrated in [the worked example](safe_scoring_worked_example_2026-10-09.md). The preceding [full-text literature review](../reports/safe_full_text_review_2026-10-09.md) preserves the detailed evidence from the five supplied papers.