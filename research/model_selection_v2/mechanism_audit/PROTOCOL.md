# SAFE probability-output mechanism audit v1

Specified **30 October 2026**, before this audit's synthetic execution. Status:
synthetic diagnostic protocol; no alteration to accepted empirical candidates,
SAFE v1, or the independent judge. This folder and its new run are isolated.

## Question and algebraic prediction

Apply the same `T(p)=a*p+b`, `a>0`, to every clean, perturbed and masked output
of a frozen synthetic prediction provider. Require all resulting probabilities
in [0,1]. Do not transform binary outcomes. For nonconstant references, predict
invariance of scalar RGA, its additive accuracy curve, `K(p,p_perturbed)`,
reversed RGE `K(p,p_masked)`, the greedy masking path, and the arithmetic,
geometric and quadratic integrated scores of our current operationalization.
Brier and binary LogLoss can change. No arbitrary nonlinear monotone invariance
is asserted for robustness or explainability.

For the operational rank kernel, an independent expression is weighted pairwise
concordance: each unordered pair contributes `|r_i-r_j|` times 1 for concordant
reference/comparison order, 0 for discordance and 1/2 for a comparison tie; divide
by total absolute reference-pair difference. Reference ties have zero weight.
Positive affine transformation multiplies every nonzero reference weight by a
common positive factor and preserves all comparison order/tie indicators. The
factor cancels. For accuracy, the binary reference stays unchanged. Its entire
area-term vector is unchanged because the comparison's order/ties are unchanged.
Thus each greedy alternative, deterministic tie set/path and each Cartesian
aggregate is invariant in exact arithmetic. This is an algebraic property, not
a statistical claim, calibration theorem or proof that models are equivalent.

## Fixed synthetic design

- Two eight-record fixtures, binary outcomes `[0,1,0,0,1,1,0,1]`.
- Strict clean probabilities `[1,3,5,7,9,11,13,15]/16`; tied probabilities
  `[2,2,5,5,10,10,14,14]/16`. All are strictly inside (0,1).
- Three synthetic removal units `a,b,c`. The complete output lookup for their
  seven nonempty subsets, three perturbations and clean inputs is fixed in
  `audit.py`, before execution. This is a frozen synthetic output provider,
  not a fitted empirical estimator or a claim about realistic input stresses.
  No-removal is clean; all-removal and final perturbation are constant 1/2.
- Four nodes per A/E/R axis: three accuracy segments, all greedy removal
  steps including endpoints, and clean plus three perturbation outputs.
- Transforms: identity; compression `a=1/2,b=1/4`; shift/compression
  `a=3/4,b=1/8`; near-constant but nonzero `a=2^-40,b=1/2`.
  Dyadic arithmetic preserves distinct levels even in the near-constant case.
- Boundary: `a=0,b=1/2` produces mathematically constant references; positive
  `a=2^-60,b=1/2` collapses to constants in float64. Both must be explicitly
  undefined for robustness/explainability, not zero-valued poor models.
- Negative control: square every output. This is strictly increasing on the
  supplied (0,1) values, preserves accuracy, but is allowed to change weighted
  robustness/explainability and integrated values. A single changing fixture
  refutes arbitrary nonlinear monotone invariance for this kernel.

## Additional information-insufficiency counterexample

Also specified before execution: a separate four-record response table with
outcomes `[0,1,0,1]` and two intervention flags (stress and mask). Candidate A/B
agree record by record on clean `[.25,.75,.25,.75]`, stress-only
`[.375,.625,.375,.625]` and mask-only `[.5,.5,.5,.5]` outputs. Under the joint
intervention A outputs `[.25,.75,.25,.75]` and B `[.75,.25,.75,.25]`.
Any deterministic aggregation supplied only the identical separate vectors,
outcomes and metadata must tie, although joint Brier/LogLoss differ. These are
coherent finite lookup functions of record identity and intervention flags.
This proves only information insufficiency of those marginal tables; it is
not a full SAFE experiment, empirical interaction claim, novelty claim or
evidence that these responses occur for Taiwan or accepted fitted models.
The fixture is separately defined in `joint_information.py` and is excluded
from the SAFE candidate/selector interpretation.

Improvement remains a hypothesis. Preserve negative and inconclusive findings,
including any failed invariance or nonlinear counterexamples. These diagnostics
are hypothesis-generation evidence; they are not independent validation.
Do not optimize an extension on the historical holdout or revise the independent
judge in response to a favored selector. A later superiority claim requires
separately frozen evaluation and evidence not used to develop that claim.

## Independent checks and outputs

Use exact `fractions.Fraction` weighted-pair concordance, independent of the
production cumulative-area code. Independently enumerate equally likely distinct
binary outcome arrangements within tied score blocks for
accuracy-area terms, and explicitly loop over Cartesian triples for aggregates.
Check every greedy alternative and selected path against the exact oracle.
Check proper losses by direct binary formulas; tests also cross-check sklearn.
Use absolute tolerance 1e-12 for production/reference agreement; preserve full
floating values in JSON. No fitting, empirical data or saved predictions are
loaded. Importing `safe2026` also imports `pilot`, whose loaders are not called.

Focused command: `.venv/Scripts/python.exe -m unittest discover -s
research/model_selection_v2/mechanism_audit -p test_audit.py -v`.
Run command: `.venv/Scripts/python.exe -m
research.model_selection_v2.mechanism_audit.audit --run-dir
runs/model_selection_v2/mechanism_audit_2026-10-30`.

The run refuses any existing destination, writes JSON and Markdown plus source
hashes and an integrity receipt, and never overwrites outputs. A read-only
`--verify-run` option checks receipts without rerunning the audit. The report
separates operationalization proof from article-fidelity verification. Existing
paper/specification provenance is referenced; this task does not freshly verify
the printed article or later author implementation. No holdout metrics, new
selector recommendation, empirical superiority or publication novelty follows.
