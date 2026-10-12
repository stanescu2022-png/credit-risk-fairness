> Full-text follow-up, 9 October 2026: All five previously retrieval-limited papers have now been supplied, archived and read. Access limitations below are historical. The full-text corrections and revised priorities in [safe_full_text_review_2026-10-09.md](safe_full_text_review_2026-10-09.md) supersede preliminary interpretations where they differ. No new credit experiment has been executed.

# SAFE/KAIRI: literature search and research directions

Research date: 9 October 2026. Requested focused research window began at 09:48 UTC. Next professor meeting: 16 October 2026.

Status: literature synthesis and proposed experiments only. No model, split, prediction, evaluation or dependency was changed. This memo does not authorize implementing the proposals.

## Main judgment

Our strongest next step is a faithful article-level baseline followed by a narrowly defined test of an assessment or model-selection blind spot. Adding conventional indicators to a dashboard is not sufficient evidence of methodological superiority. The original 2024 KAIRI paper and the later SAFE rank-graduation papers must be distinguished. A contribution can improve the original article without being novel relative to the subsequent literature.

The most promising hypothesis from this search is that performance-preserving parsimony can discard redundancy that matters under a specific data-source failure. The second is that interpreting nonsignificant group differences as fairness can create false reassurance under limited statistical power. A ranking-preserving probability distortion is a useful demonstration/control, but later SAFE research already addresses point-value sensitivity; it is a weak standalone novelty claim.

No definitive published refutation of the original article was located. This is a bounded search, not proof that none exists. Direct published assessments, related SAFE extensions, general component-level critiques and our own hypotheses are separated below.

## 1. What the newer SAFE literature changes

### Probability-sensitive metrics already exist

Babaei and Giudici, **Correlation Metrics for Safe Artificial Intelligence**, Risks 13(9), 178 (2025), adds point-value measures based on squared errors and contrasts them with rank-based measures. For binary predictions, its predictive squared-error criterion is closely related to Brier skill. This materially weakens any claim that our contribution is simply to notice that good rankings can coexist with bad probabilities. Access: primary publisher indexed methodological excerpts; direct page fetch was rate-limited.

Source: https://www.mdpi.com/2227-9091/13/9/178

### A single integrated score already exists

Giudici and Kolesnikov, **SAFE AI metrics: An integrated approach**, Machine Learning with Applications 23, 100821 (2026), integrates security/robustness, accuracy and explainability through perturbation/removal curves and alternative aggregators. Its three axes do not directly reproduce the four KAIRI headings. Access: primary publisher abstract and author conference material; full publisher page blocked automated access.

Source: https://www.sciencedirect.com/science/article/pii/S266682702500204X

### Joint uncertainty for that score already exists as a preprint

Sokolov, Giudici and Kolesnikov, **Confidence regions for SAFE AI compliance scores** (2026 preprint), uses paired bootstrap covariance for components measured on shared observations. It shows that ignoring dependence can either narrow or widen score uncertainty. Full arXiv HTML and abstract were accessible. This is a preprint, not established peer-reviewed validation, but it is relevant prior art for novelty.

Source: https://arxiv.org/abs/2609.29588
Full text: https://arxiv.org/html/2609.29588v1
Author code: https://github.com/safe-composed-uncertainty/composed-uncertainty

### Alternative distribution-sensitive SAFE assessments already exist

Spelta and Raffinetti, **Evaluating SAFE AI principles using Wasserstein distance: a comparative study of machine learning models**, Statistics 58(6), 1283-1303 (2024), proposes a distribution-sensitive alternative and discusses limitations of rank-based assessment for subtle distributional changes. Access: primary publisher indexed excerpts; direct publisher access restricted. This is a specific extension/critique of the rank-graduation stream, not a wholesale refutation of the original KAIRI article.

Source: https://www.tandfonline.com/doi/full/10.1080/02331888.2024.2419420

### Conditional and error-burden fairness are already active SAFE research

Giudici and Babaei, **Explainable Fairness and Propensity Score Matching** (2025), studies aggregate versus conditional fairness and Simpson-type reversals in lending. Full primary publisher text was accessible. Atif and Giudici, **A Rank Graduation metric for Algorithmic fairness** (2026 preprint), introduces error-burden fairness and inferential testing, with HMDA lending applications. These make generic claims such as 'we add subgroup fairness' or 'we consider errors rather than explanations' too broad to establish novelty. HMDA denial targets differ from our observed next-month default target.

Sources:
- https://onlinelibrary.wiley.com/doi/full/10.1002/asmb.70047
- https://arxiv.org/abs/2609.39025
- https://arxiv.org/html/2609.39025v1

### Consequence for our publication strategy

A comparison against the explicitly chosen 2024 article remains legitimate. However, a paper claiming a new SAFE extension should also address the closest 2024-2026 developments. We should not choose an obsolete or deliberately incomplete comparator simply because it is easier to beat. Relevant later baselines can be limited to the contribution under test; we need not reproduce every SAFE paper.

## 2. Direct assessments and component-level critiques

| Evidence | What it supports | What it does not prove |
| --- | --- | --- |
| Electronics 2025 review, section on KAIRI | Published concerns about complexity, quantitative emphasis and incomplete treatment of systemic or organizational factors | A demonstrated failure on Taiwanese credit data |
| Wasserstein SAFE paper | Rank-based assessment may miss distributional aspects; alternatives exist | All SAFE metrics are invalid |
| Shapley critique and fairwashing papers | Attribution properties do not automatically establish fairness, causal relevance or explanation fidelity | Our exact TreeExplainer/LinearExplainer implementation has been successfully attacked |
| Equivalence-testing literature | Failure to reject a difference does not establish practical equality | A new statistical theory invented by us |
| Feature-deletion robustness literature | Clean predictive quality does not guarantee resilience to unavailable features | Parsimony necessarily makes every model less robust |

**Direct KAIRI review:** *Evaluating Trustworthiness in AI: Risks, Metrics, and Applications Across Industries*, Electronics 14(13), 2717 (2025). The review identifies accessibility/complexity and broader qualitative/systemic coverage limitations. Access: indexed primary publisher section; full page rate-limited. These are review authors' assessments, not empirical counterexamples.

https://www.mdpi.com/2079-9292/14/13/2717

**Shapley feature-importance limitations:** Kumar et al., *Problems with Shapley-value-based explanations as feature importance measures*, ICML 2020. Mathematical attribution properties alone do not guarantee that an explanation answers the practical question of interest.

https://proceedings.mlr.press/v119/kumar20e.html

**Explanation fairwashing:** Slack et al., *Fooling LIME and SHAP* (2020). A model can exploit the distinction between genuine inputs and artificial explanation probes. This concerns particular perturbation-based explainers and adversarial constructions; it must not be transferred indiscriminately to our current explainers.

https://arxiv.org/abs/1911.02508

**Fairness and prediction explanations are different targets:** Begley et al., *Explainability for fair machine learning* (2020), explains why conventional prediction explanations do not necessarily identify the source of unfairness.

https://arxiv.org/abs/2010.07389

**Dependent features:** Aas, Jullum and Loland, *Explaining individual predictions when features are dependent* (2021), develops dependence-aware Shapley approaches. Conditional and interventional explanations answer different questions; neither automatically proves causality. Our independent masking and marginally valid corruptions therefore remain limitations to acknowledge.

https://arxiv.org/abs/1903.10464

**Nonsignificance versus equivalence:** Lakens, *Equivalence Tests: A Practical Primer* (2017). Evidence of practical equality requires an explicit margin and suitable inference, not merely p > .05.

https://journals.sagepub.com/doi/10.1177/1948550617697177

**Fairness audit uncertainty:** Cherian and Candes, *Statistical Inference for Fairness Auditing*, JMLR 25(149), 2024. Simultaneous inference for rich subgroup audits already exists. A generic bootstrap fairness certificate is consequently not sufficient novelty.

https://jmlr.org/papers/v25/23-0739.html

**Hidden subgroup failures:** Kearns et al., *Preventing Fairness Gerrymandering* (2018), formalizes failures of broad group parity on richer subgroups. Multicalibration similarly addresses overlapping-group probability quality. Merely adding subgroup slices would repeat an established idea.

https://proceedings.mlr.press/v80/kearns18a.html
https://proceedings.mlr.press/v80/hebert-johnson18a.html

**Availability robustness:** Globerson and Roweis, *Nightmare at test time: robust learning by feature deletion* (2006), studies classifiers under missing features. It supplies the threat-model rationale for our proposed parsimony experiment, not a SAFE-specific critique.

https://doi.org/10.1145/1143844.1143889

**Realistic tabular threats:** TabularBench (NeurIPS 2024) evaluates clean and adversarial performance together and warns that apparent robustness can reflect trivial prediction behavior. Related 2025 coherence/consistency research stresses dependencies and valid tabular modifications. Our current semantic corruptions enforce selected marginal constraints, not complete joint plausibility.

https://papers.nips.cc/paper_files/paper/2024/hash/8f5defaa8dc728c3e4aaee59ac8ca942-Abstract-Datasets_and_Benchmarks_Track.html
https://serval-uni-lu.github.io/tabularbench/
https://doi.org/10.1016/j.asoc.2025.113998

## 3. Our own hypotheses, ordered by usefulness

### A. Parsimony can remove operational redundancy

**Question:** Does the article's clean-performance-based selection favor a reduced model that becomes materially worse when a particular source of information fails?

**Why this is an actual methodological question:** The article treats sustainability principally through model parsimony and predictive comparison. A smaller model can preserve discrimination on clean data while depending more heavily on the remaining inputs. This is our inference, not a claim that the article promises resilience to every data failure.

**Illustrative counterexample, not a project result:** Suppose two measurements normally satisfy x1 = x2. A full predictor averages them, while a reduced predictor uses only x1. They give identical clean predictions. If x1 is corrupted by delta, the full predictor moves by delta/2 and the reduced predictor by delta. Clean equivalence did not establish equal fault tolerance. Real data can behave differently; extra features can also add attack surface or noise.

**Taiwan hypothesis:** PAY_0 is highly influential in both saved models. A reduced model might rely more strongly on a single recent repayment signal than a model retaining informative historical signals. Test delayed/unavailable repayment information or a documented source failure. This is not yet observed, and our dataset supplies no empirical frequency for such operational faults.

**Proposed comparison:** Faithfully construct the article's Shapley-ranked selection path on training data, with stopping rules stated in advance. Compare full and reduced models of the same family, then compare the article selection policy with a policy requiring clean predictive adequacy AND fault performance adequacy. Keep the current frozen pipelines as historical controls. Any newly fitted models require separate authorization and versioned outputs.

**What success would mean:** The original selection policy favors a reduced model; under a prespecified plausible fault it breaches a meaningful decision-quality limit; the augmented policy selects a model meeting that limit, or correctly withholds assurance. Include clean performance, fault performance, feature acquisition burden, and paired uncertainty.

**What would falsify the proposed case:** Reduced models are equally or more resilient; full models also breach the limit; results depend on arbitrary fault choice; another original article criterion already rejects the reduced model; or improvement disappears under a reasonable severity/budget sensitivity analysis.

**Novelty risk:** Robust feature selection and feature-deletion resilience are established subjects. Our possible contribution would be a precise failure analysis and justified extension of the SAFE selection procedure, validated on credible data-source faults. A single anecdotal failure is not automatically publishable.

**One-week feasibility:** A narrowly scoped proof of concept is plausible. A publication-ready method and external validation are not assured within one week.

### B. Fairness testing can confuse insufficient evidence with reassurance

**Question:** When true disparity is practically important but a group is small, does the article-style interpretation of a nonsignificant test incorrectly support a fairness claim?

**Controlled study:** Generate or subsample auditing datasets with fixed known disparity and varied group size; preserve predictor groups for project resampling. Compare the stated original interpretation against a three-way decision: supported within a prespecified margin, violated, or insufficient evidence. Evaluate false reassurance and unwarranted alarms against known simulation truth.

**Necessary precision:** The augmented audit cannot manufacture information. On a low-power sample its correct result may be 'insufficient evidence', not a confident detection of harm. This can still improve the assessment decision if the original procedure labels the same case fair.

**Fairness target:** Define the policy, positive outcome, group contrast and acceptable disparity before looking at results. A next-month default flag is not a loan denial; do not silently transfer a lending-outcome criterion. Nonsignificant SHAP distribution differences are not general proof of absence of harmful decision disparities.

**Novelty risk:** Equivalence and statistical fairness certification already exist, including recent SAFE uncertainty and fairness work. This is a clear critique and feasible demonstration, but a weak standalone invention unless we solve a more specific unresolved problem.

**One-week feasibility:** High for a controlled demonstration and baseline replication; weaker publication upside by itself.

### C. Ranking-preserving probability faults

**Question:** Can a model retain excellent ranking/explanation summaries while its probability-based decisions deteriorate?

A strictly increasing transformation preserves ranking. An affine transformation p' = a + b p, b > 0, with valid probability bounds can preserve AUC and AP while changing calibration and decisions at a fixed threshold. For a fixed probability-output Shapley game, affine scaling multiplies feature contributions by b; normalized importance shares can therefore remain unchanged. Our saved log-odds SHAP game is different, so this does not justify claiming invariance of our existing explanation artifacts.

An extreme illustrative transform p' = .6 + .4p flags everyone at .50. This is a mathematical control, not evidence of a plausible production event or a new discovery. The whole original methodology includes more than AUC, so all applicable criteria must be reported; later SAFE squared-error metrics would notice the fault.

**Conclusion:** Useful Explorer demonstration and negative control. Poor main publication contribution if described as 'SAFE misses calibration and we add Brier'. A stronger research question would need a new, defensible assessment task beyond the already published point-value extension.

### D. Explanation fairness versus decision fairness

A model can produce similar feature-attribution distributions across groups yet impose different errors or decision burdens. Conversely, different attribution distributions need not imply unacceptable treatment. That separation can be demonstrated on controlled models, but it is established literature and a recent focus of SAFE authors themselves. Sophisticated fairwashing experiments also require careful alignment with the exact explanation algorithm and realistic input distribution.

**Conclusion:** Important interpretation safeguard; medium-to-low priority for a fresh one-week contribution.

### E. Composite scores can conceal a critical failure

A compensating average can reward strong components despite a failed critical requirement. However, integrated SAFE scores and uncertainty now exist, and noncompensatory decision rules are not themselves new. Any extension must define the operational consequence that cannot be traded away, rather than merely propose new weights.

**Conclusion:** The score should summarize the substantive experiment. It should not be the entire contribution.
## 4. A score that answers the professor's question

First define a decision task. The most useful comparison is not 'which dashboard contains more metrics?' but 'which assessment method selects or certifies a model that remains adequate under the specified fault?'

An illustrative independent outcome measure is cost-weighted classification loss:

H = (c_FN * FN + c_FP * FP) / N.

This uses actual observed errors, not prediction flips. Costs must be declared scenario assumptions and examined across a reasonable range; the Taiwan data cannot establish bank profit, rejection costs or production fault frequency. If cost information is unavailable, report extra false negatives and false positives separately instead of inventing a financial score.

For prespecified critical scenarios j with meaningful tolerances tau_j, an illustrative noncompensatory summary is W = max_j(H_j / tau_j). Lower is better. W <= 1 means all point estimates are within their tolerances. This formula is a proposal and a familiar decision-rule pattern, not a claimed novel statistic.

A confidence-aware conclusion should be three-way:
- Supported: all simultaneous upper bounds are within the stated tolerances.
- Violated: at least one relevant lower bound exceeds its tolerance.
- Insufficient evidence: otherwise.

Multiple scenarios require an explicitly valid simultaneous procedure; separate 95% intervals are not automatically a joint 95% certificate. Recent SAFE uncertainty work must be compared where relevant. Empirical boundaries cannot be chosen afterward to manufacture a crossing.

Compare methods using the SAME independent outcome target, data access, training/validation budget, faults and test observations. Record selection outcomes, false reassurance, detection at matched false-alarm budgets, and decision quality of the selected model. A proposed extension that merely adds the exact fault metric used to define success risks a tautological comparison; justify the threat independently and test generalization to held-out fault severities or mechanisms.

The score should make one operational conclusion easy to read, with component evidence available underneath. It should not average AUC, a fairness gap, ECE and RGR simply because all can be rescaled to [0,1]. Those quantities answer different questions and can conflict.

## 5. Candidate one-week program, not an execution commitment

1. Freeze the research question: article-style parsimony under repayment-source availability faults. Identify the exact credit-application modules required from the original article; reproducing four unrelated application domains is unnecessary.
2. Specify a transparent article-level adaptation: model families, selection path, inferential conventions and ambiguities, preprocessing, development-only selection and separate outputs. Preserve the existing LR/GBM study. Consider original LR/RF families where matching the article matters.
3. Establish whether clean-performance-preserving reduction actually occurs. A failed premise is useful information; do not force a reduction to create a favorable result.
4. Pilot a small independently justified family of source failures. Existing semantic perturbations are supporting infrastructure, not automatically realistic missing-source experiments. Define replacement/imputation behavior, joint constraints, changed-record exposure and label preservation.
5. Compare paired fault losses, uncertainty and materiality. Test benign controls and fault severity sensitivity. The current historically inspected holdout cannot be advertised as a newly untouched confirmatory sample. New development choices must remain off the final evaluation, and publication should seek independent confirmation.
6. Compare the closest later SAFE baseline relevant to the proposed claim, especially point-sensitive assessment or integrated robustness metrics. A failure of only the original article cannot be generalized to all SAFE research.
7. Present a concise professor update: baseline fidelity, hypothesis, one informative experiment, an honest result, and a go/no-go decision. Progress can be a falsified candidate and a justified alternative; a guaranteed dramatic win is not a sound research promise.

If the parsimony hypothesis does not work, the fairness false-reassurance simulation is the best backup for a clear methodological discussion. A calibration-only demo is the best simple teaching aid, but currently the weakest claim to original research.

## 6. Sources for deeper follow-up and pitfalls to avoid

- Babaei, Giudici and Raffinetti, *A Rank Graduation Box for SAFE AI*, ESWA 259, 125239 (2025): establishes the common rank-graduation assessment stream. https://www.sciencedirect.com/science/article/pii/S0957417424021067
- Demler, Pencina and D'Agostino, *Misuse of DeLong test to compare AUCs for nested models* (2012): specific issues arise when nested models are developed and validated on the same data. This does not establish that every held-out use of DeLong in the original paper is invalid. https://pmc.ncbi.nlm.nih.gov/articles/PMC3684152/
- Hand, *Measuring classifier performance: a coherent alternative to the area under the ROC curve* (2009): cost interpretation is already an established criticism of AUC. A new cost-based summary alone is not novel. https://spiral.imperial.ac.uk/server/api/core/bitstreams/4daffcda-4e19-4e5f-bc5d-dee73f618939/content
- Vickers and Elkin, *Decision Curve Analysis* (2006): decision utility beyond discrimination is established. Financial interpretations require appropriate credit-domain assumptions. https://journals.sagepub.com/doi/10.1177/0272989X06295361
- Kumar, Liang and Ma, *Verified Uncertainty Calibration* (2019): finite-sample calibration estimation and verification have existing theory. https://arxiv.org/abs/1909.10155
- Kleinberg, Mullainathan and Raghavan, *Inherent Trade-Offs in the Fair Determination of Risk Scores* (2016/2017): multiple fairness desiderata can be incompatible. Our score should not demand every fairness definition simultaneously. https://arxiv.org/abs/1609.05807
- Mase, Owen and Seiler, *Cohort Shapley value for algorithmic fairness* (2021): observational/cohort attribution is an existing alternative to artificial feature combinations. https://arxiv.org/abs/2105.07168

Search scope included the exact article title and DOI; KAIRI and SAFE critiques/reviews; authors' subsequent SAFE papers; score aggregation and uncertainty; SHAP limitations/fairwashing/dependence; fairness inference; parsimony and feature-deletion robustness; and constrained tabular stress evaluation. Primary research and author materials were preferred. Publisher access restrictions meant some sources were assessed through indexed abstracts/excerpts rather than full text; those cases are identified above. There was no systematic citation-complete review, no novelty guarantee, and no new empirical verification.

## Proposed message to the professor

We are testing whether a model that looks adequate under the original selection procedure can lose important resilience after redundancy is removed. Our proposed extension makes the relevant operational fault an explicit selection constraint and reports one clear, uncertainty-aware assessment. We will compare against the original article and the closest newer SAFE methods, and judge success by independently defined decision harm rather than by the number of indicators.