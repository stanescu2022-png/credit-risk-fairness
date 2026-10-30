# Publication-oriented research agenda

Updated **30 October 2026**. Persistent goal: active. Primary agent owns research
planning and interpretation; **gpt-6.1-sol, high reasoning effort**, owns delegated
implementation. This file records the research process and candidate direction,
not a claim of novelty or a frozen empirical protocol.

## Accepted purpose and research boundaries

The intended output is a defensible research contribution toward an article:
compare published integrated SAFE 2026 with our methodology as model-selection
procedures on common candidates, and evaluate the consequences of their choices.
The user authorized autonomous pursuit, planning, implementation and evaluation
on 30 October. Improvement is a hypothesis, not the required result.

The user's updated goal explicitly requires:

- preserve negative and inconclusive findings;
- do not optimize our methodology against the historical holdout;
- do not repeatedly revise the judge according to which selector it favors;
- separate hypothesis generation, exploratory experiments, protocol freezing and
  subsequent independent validation;
- support any superiority claim with evidence not used to develop that claim.

No commit, push, publication or external contact is authorized. Existing research,
sources, models, splits and receipts remain protected. Gaussian and article-LR
repair remain outside the current work. The primary evaluation remains open.

## Research logic

The organizing chain is **limitation -> mechanism -> methodological response ->
falsifiable selection consequence -> independent evaluation**. A broad metric
dashboard is useful infrastructure but cannot substitute for this argument.

Distinguish measurement validity from decision sufficiency. A ranking-preservation
metric can correctly measure ranking preservation while failing to distinguish
probability forecasts that have different outcome losses. Likewise, aggregation
of separate stress responses does not automatically identify a response under
joint interventions. These are scope distinctions, not blanket claims that SAFE
is invalid or that its authors recommend unsuitable models.

## Candidate limitations and present evidence

| Question | Current evidence | Response worth testing | What would weaken the research argument? |
| --- | --- | --- | --- |
| Can integrated scores distinguish probability quality within an affine output-equivalence class? | Algebra and synthetic checks support positive-affine invariance of our fixed operationalization, while proper losses change | Include outcome-based probability losses in the selection decision | Already known probability-sensitive SAFE and a simple clean-loss selector solve the practical issue equally well |
| Do separately measured curves identify loss when interventions act together? | A finite response-table counterexample establishes non-identification in general; no Taiwan interaction demonstrated | Evaluate joint scenarios explicitly, with marginal-only ablation | Joint scenarios add no selection-relevant information, or changes come only from greater exposure/severity |
| Can an aggregate select a candidate worse on all declared outcome-risk criteria? | Executed selection exploration: RF and GBM each dominate SAFE's logistic choice on the six development outcome coordinates; no independent superiority established | Outcome-risk Pareto screening before the existing SAFE preference | The same practical decision is obtained by a simpler baseline, or independent evaluation exposes offsetting costs |
| Does prediction invariance imply useful performance under disturbance? | They are different estimands; input-independent controls in the integrated paper already illustrate the distinction | Absolute stressed loss alongside deterioration and preservation | Only degenerate controls expose it; plausible fitted candidates show no useful consequence |
| Can feature reliance justify human explainability claims? | Current RGE measures response to masking; human comprehension remains unevaluated | Keep this interpretation separate | A new proxy merely renames reliance without validating explanation quality |

Priority: investigate the first three as a connected decision problem. Keep the
fourth as a diagnostic and the fifth as an explicit scope limit. Do not expand
into unrelated governance, monitoring or mitigation work.

## Executed mechanism evidence

The [synthetic audit](../../runs/model_selection_v2/mechanism_audit_2026-10-30/report.md)
and its [prespecified protocol](mechanism_audit/PROTOCOL.md) are complete. Sol ran
15 focused tests; the primary agent reviewed the oracle/design and independently
reran all 15 successfully. No empirical loader, new model fit, empirical prediction
or holdout assessment was used.

For nonconstant reference r and comparison s, write the rank kernel as a weighted
pairwise concordance: pair weight `abs(r_i-r_j)`, concordant contribution 1,
discordant 0, comparison tie 1/2. Under `T(p)=a*p+b`, `a>0`, reference weights
scale by a and comparison order/ties remain fixed. The common scale cancels.
For accuracy, the outcome reference remains unchanged. In our operationalization,
all accuracy contributions, masking alternatives, greedy choices and integrated
arithmetic/geometric/quadratic scores therefore remain invariant in exact
arithmetic when every clean, masked and stressed output is transformed alike.

The primary agent additionally visually checked the published integrated PDF
pages 2-5, equations (4), (8), (10), (12)-(17), on 30 October. The affine
cancellation agrees with the printed scalar formulas. Full-curve conclusions
retain the explicitly documented Taiwan construction choices; no claim is made
about unverified author implementations or TOPSIS. Do not edit the completed
audit to retroactively incorporate this separate review.

The audit preserves a negative control: squaring probabilities preserves accuracy
ranking but changes reference gap weights and robustness/integrated scores.
Thus **arbitrary nonlinear monotone invariance is false** for this kernel.
Near-constant nonzero affine maps can retain scores while probability forecasts
approach a constant. Exact constants and float64 collapse are explicitly undefined,
not assigned a favorable or unfavorable substitute score.

A population-level interpretation follows without new credit data. If q is the
true conditional default probability, expected Brier loss for forecast p equals
`q*(1-q)+(p-q)^2`. Starting at p=q, a nonidentity affine map increases expected
loss wherever it changes q, while the operational rank quantities above cannot
distinguish it. This is an identifiability limitation of the chosen measurement
for a probability-quality decision. It is not evidence that every affine change
worsens a misspecified model, nor a novel proper-scoring theorem.

The separate two-intervention response-table example agrees on all clean and
single-intervention outputs but differs on joint outputs. It establishes that
those separate observations alone cannot determine joint loss. It is not an
end-to-end SAFE experiment, realistic corruption model or Taiwan result.

The empirical four-condition contrast is interaction on a **loss scale**. It
does not by itself establish a nonlinear prediction response or causal synergy.
For Brier, let `a=p10-p00`, `b=p01-p00`,
`c=p11-p10-p01+p00` and `r=p00-y`. Direct expansion gives
`L11-L10-L01+L00 = 2ab + 2c(r+a+b) + c^2`.
Even additive probability responses (`c=0`) produce contrast `2ab`. For example,
`y=0` and probabilities `(p00,p10,p01,p11)=(1/8,1/4,3/8,1/2)` give
zero probability interaction but a positive Brier contrast of `1/16`.
This analytic observation was recorded before empirical execution; it changes
the interpretation, not the predeclared empirical endpoint.

## Prior-art constraints

The [targeted literature update](literature_update_2026-10-30.md) records inspected
sources and search limits. Key consequences:

- Probability-sensitive SAFE already exists: the 2025 Correlation Metrics binary
  predictive R2 is a normalized Brier score. A clean-Brier baseline represents
  that accuracy ordering on a common sample. Do not invent a fully integrated
  COR selector where its aggregation/removal conventions are unresolved.
- Integrated SAFE explicitly discusses differing recommendations across
  perturbation families and future multi-scenario selection.
- Pareto screening, constrained multi-risk model selection and worst-case loss
  optimization have prior art. Their use here is not an automatic novelty claim.
- A newly located 2026 preprint already develops covariance-aware uncertainty
  for integrated SAFE. Bootstrap uncertainty is supporting validity work.

A plausible article contribution is a **focused formal and empirical study of
when a SAFE selection decision is insufficient for declared outcome risks,
and whether a minimal correction improves choices beyond simple alternatives**.
Its publication value remains to be established by the evidence and broader
literature review. Absence of a matching search result does not establish novelty.

## Working methodological candidate, not yet frozen

Investigate **outcome-risk Pareto screening followed by SAFE selection**:

1. Evaluate all fixed candidates on the same development records under clean,
   separately perturbed and jointly perturbed conditions.
2. Build an explicitly declared probability-loss vector. Preserve Brier and
   LogLoss separately; use absolute outcome losses, not only self-preservation
   or deterioration from each candidate's baseline.
3. Remove empirically dominated candidates under that vector.
4. Among the retained candidates, apply the existing arithmetic SAFE preference
   and its fixed tie rule. Report the retained set and all choices.

This is a deliberately minimal candidate correction, not a finalized new named
framework. It retains SAFE's preference among outcome-admissible trade-offs and
has an elementary in-sample property: the chosen candidate is nondominated on
the declared vector. That property does not imply out-of-sample dominance,
statistical risk control, fairness or a practically meaningful improvement.

Two further decision properties were identified before empirical execution:

- If no candidates have exactly equal vectors on the smaller coordinate set,
  adding coordinates can only expand its nondominated set. With well-separated
  saved SAFE scores, the selected SAFE score then cannot decrease. A return
  toward the original SAFE choice can reflect newly visible trade-offs. This
  statement is conditional: old A=(1,1), B=(1,1) retain both; adding a coordinate
  A=0, B=1 removes B. New coordinates can split old equality classes and remove
  former frontier members, potentially reducing the selected SAFE score. The
  initial unconditional nesting statement was corrected during root review,
  before empirical execution; a synthetic tie counterexample is required.
- A frontier replacement need not dominate the original SAFE choice. For two
  losses, original O=(2,2), A=(1,1), B=(3,0), with SAFE preference O>B>A, gives
  frontier {A,B} and selects B. B improves the second coordinate but worsens the
  first relative to O. This preserves the current rule's limitation; no direct-
  dominator repair or new method variant is silently substituted.

The Sol implementation assignment includes synthetic checks for both properties.
They concern the decision rule, not empirical candidate behavior or novelty.

The bounded exploration v1 protocol fixes clean losses plus worst
marginal-scenario and worst joint-scenario mean losses,
separately for Brier and LogLoss. Worst-case aggregation makes a conservative
scenario preference explicit; it does not claim measured scenario probabilities.
Scenario-wise results must remain visible, since maxima can hide which fault
drives a choice. Its exploration results cannot by themselves justify a final
method or superiority claim merely because the vector changes SAFE's pick.

Required comparators/ablations before claiming added value:

- unchanged integrated SAFE arithmetic selector; geometric/quadratic sensitivity;
- clean Brier selection and clean LogLoss selection;
- simple worst-scenario Brier and worst-scenario LogLoss selection;
- the screening method with clean losses only;
- screening using marginal scenarios only;
- screening including the joint scenarios.

If our method chooses the same candidate as SAFE or a simpler baseline, retain
that result. If joint scenarios do not add useful information, do not relabel
another diagnostic as the successful contribution after seeing holdout results.
Any alternative generated from exploratory results gets a new hypothesis/version
and separate later validation evidence.

## Four research stages

### 1. Hypothesis generation (initial mechanism audit completed)

Complete the mechanism audit, targeted prior-art map, scenario semantics and
comparators. The saved selection metrics have already been inspected: GBM has
the smallest clean Brier/LogLoss in this fixed pool while SAFE selects LR.
That is development context, not new evidence of superiority. The root reviewed
those saved metrics during planning; record this exposure rather than claiming
the proposal was conceived blind to all development outcomes.

### 2. Bounded exploratory development (executed and verified)

The [exploration v1 specification](exploration_v1_specification.md) now fixes the
bounded scenario design, loss vector, baseline/ablation selectors and conditional
uncertainty before new predictions. It is an exploratory implementation decision
under the user's leeway, not a final independent-evaluation freeze.

Executed results are in [selection findings](selection_findings_2026-10-30.md):
RF is chosen by the full screen, marginal screen and simple worst-LogLoss;
SAFE chooses logistic and clean losses choose GBM. No distinct point-selection
benefit from adding joint axes is established. Conditional frequencies differ,
but do not supply independent superiority evidence. Root independently passed
26 synthetic tests and reconstructed the saved empirical results. The following
development constraints were applied and remain requirements for future versions.

Use fitting/selection data only. Write a new protocol before generating new
scenario results. Preserve all attempted variants and the reason for each choice.
No refitting is needed for the first pilot. Candidate fault families may include
masked repayment/billing/payment blocks and corruption of disjoint remaining
inputs, with paired clean/single/joint conditions. These are semantic feature
families, not verified independent production data feeds. Specify exposure,
overlap, ordering, support and fitting-only replacement rules; compare matched
exposure so a joint effect is not merely a larger corrupted sample.

Selection bootstrap can assess instability conditional on fitted candidates;
it cannot establish independent superiority. Do not use historical holdout results
to decide which variant, scenario, metric or margin to retain.

### 3. Protocol freeze (executed)

The [immutable judge v1 freeze](../../runs/model_selection_v2/judge_v1_freeze_2026-10-30/FREEZE.json)
binds all method choices and sources before new evaluation access. Root review
and 23 independently rerun judge tests preceded execution. The design predates
selection-exploration results; implementation was not fully blind because the
agent read the checkpoint containing development choices. No outcome-driven
judge or method change followed.

Save a dated, hash-bound specification of candidate method, all baselines,
candidate pool, scenario suite, exposure rules, preprocessing, primary-evaluation
status, metric directions, comparison endpoints, uncertainty, missingness, ties
and claim rules. Record selector choices before evaluation. Code correctness
fixes after freezing require explicit version/provenance handling and disclosure,
not silent methodological replacement.

### 4. Exploratory assessment completed; independent validation remains future

The [judge findings](judge_findings_2026-10-30.md) and independent reconstruction
are complete. RF versus SAFE's logistic is a six-coordinate trade-off: lower
probability losses, larger observed but uncertain sex-gap differences. All four
models remain on the reference frontier. GBM has lower reference probability-
loss point estimates than RF. No distinct added value over the simpler baseline
panel or new-method superiority is established. Preserve these conclusions.

The inspected Taiwan holdout can supply an exploratory paired assessment after
freezing, never pristine confirmation. A stronger claim requires later evidence
not used to develop it: a prespecified additional dataset and/or independently
generated synthetic trials under a frozen design with appropriate scope. New
random seeds alone do not make a hand-designed favorable mechanism representative
of real credit populations. Fresh full-procedure evaluation must repeat fitting
and selection, preserve whole-original-predictor grouping, and account for overlap
between repeated splits. Historical results and the current frozen-model
bootstrap cannot be relabelled as this design.

## Work and verification ledger

| Status | Item |
| --- | --- |
| Executed | Persistent goal created and updated safeguards read back from the goal tool |
| Executed | Sol high synthetic mechanism audit, 15 tests; root independent test rerun and equation review |
| Executed | Targeted literature update; established overlap excludes several superficial novelty claims |
| Executed | Sol high pure prediction-table judge: explicit coordinate configuration, aligned losses/gaps, Pareto relations and conditional paired group bootstrap; 30 tests independently rerun by root |
| Executed | Exploration v1: 26 synthetic tests independently passed; fresh selection-only run and independent saved-output reconstruction verified; all baselines/ablations/conditional uncertainty and negative findings preserved |
| Executed | Judge v1: predeclared six-coordinate design, immutable method/choice/source freeze, 23 synthetic tests independently passed, historical exploratory assessment and all-2,000-draw root reconstruction verified; figures reviewed |
| Executed | Same-swap descriptive loss diagnostic on original saved SAFE predictions; a separate zero-severity tie-label correction preserves the first output and raw scores |
| Open future stage | Independent validation dataset/execution, primary evaluation, meaningful margins, independent explainability and article novelty/superiority claim |

The scoped exploratory study is complete; publication readiness and independent
superiority are not established. The earlier [judge proposal](judge_proposal_2026-10-30.md)
is a measurement menu superseded, for the first exploratory assessment only, by
the [judge v1 specification](judge_v1_specification.md). These are explicit
agent-specified endpoints under the user's leeway, not choices silently inferred
from generic utilities or subsequent results.
