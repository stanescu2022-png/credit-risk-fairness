# Predeclared exploratory judge v1

Prepared **30 October 2026** against the root-owned
[judge specification](../judge_v1_specification.md), fixed before selection-run
results. Implementation and synthetic checks only: no freeze or empirical judge
run has been executed by this implementation task. Root reviews before use.
The four fitted pilot pipelines remain frozen; there is no fitting, refitting,
tuning, calibration, threshold selection or post-evaluation replacement choice.

## Boundaries and commands

The explicit freeze step requires a verified selection exploration and preserves
all ten q=.10 choices/ties plus q=.20 **diagnostic** choices, candidate/run hashes,
both specifications, settings, source snapshots and software versions. It reads
development artifacts, sources and model bytes, **not evaluation data files**.
Recorded data-input hashes are bound without opening those inputs. Root must
review the code/tests and development findings first.

```powershell
.venv/Scripts/python.exe -m research.model_selection_v2.judge_v1.runner --freeze runs/model_selection_v2/judge_v1_freeze_2026-10-30
```

The prepared assessment verifies that immutable `FREEZE.json`, all current
source/model/choice/run identities and versions before any evaluation-data read.
Then it checks actual data-input hashes and loads the exact saved 5,996 IDs,
order, outcomes, SEX and whole-original-predictor groups. All 30,000 records'
group identities and fitting/selection/evaluation separation are reconstructed
from the original 22 predictors. No repartition or full-development refit.
All references come only from the saved 19,203 fitting records.

```powershell
.venv/Scripts/python.exe -m research.model_selection_v2.judge_v1.runner --run runs/model_selection_v2/judge_v1_2026-10-30 --freeze-receipt runs/model_selection_v2/judge_v1_freeze_2026-10-30/FREEZE.json
```

Fresh destinations and exclusive writes refuse overwrites. Freeze verification
failure prevents data access. Interrupted assessments retain an incomplete marker
and failure details; only a hash-valid `VERIFIED.json` establishes completion.
The historical Taiwan records were inspected previously: this is **out-of-sample
exploration, not pristine validation** or sufficient evidence of superiority.

## Computation and outputs

The six lower-is-better coordinates are clean Brier, clean LogLoss, absolute clean
TPR/FPR female-minus-male gaps, and worst mean joint-scenario Brier/LogLoss across
three families. Average repeated losses **before** taking scenario maxima; retain
every exact argmax. The primary evaluation remains `None`. No score, normalized
weights, practical margin or dominance tolerance is added. Signed gaps, rates,
denominators/confusion counts and missing-coordinate status remain visible.
Strict decisions use `p>.5`, ties class 0. Only LogLoss clips at float64
`eps=2^-52`; Brier is unclipped. Endpoint/clipping counts are saved.

`transforms.py` is a separately tested adapter to the frozen reference/masking/
exposure helpers and semantic rules. It uses new `judge_v1:<family>` SHA-256 child
streams, 30 repetitions and nested q=.10/.20 exposure. All candidates share each
input; mask/corruption/joint modes share selected records. Disjoint operations
must commute. Registries bind seeds, input hashes, ordered/selected/changed/reused
IDs and support/exposure diagnostics. Changed-row prediction reuse is audited
against full-frame prediction in 72 fixed first-repetition cases at absolute
1e-15; that tolerance never enters dominance.

`analysis.py` evaluates all four candidates, maps frozen methods to those outputs,
records same-choice separately from different candidates with equal measures,
and reports full-vector frontier/pair matrices. The displayed principal contrast
is the full-screen **q=.10 choice minus arithmetic-SAFE q=.10 choice**, at both
judge severities. q=.20 development choices remain diagnostic and cannot replace
these identities. All six fixed pairs use later pilot-order candidate minus
earlier, retaining GBM-minus-LR. Clean AUC/sklearn AP, overall recall/FPR, ten fixed
equal-width reliability bins, all scenario losses/deterioration, movement/strict
flips and loss-scale factorial contrasts are supporting diagnostics, not new axes.
The contrast `L11-L10-L01+L00` does not prove causal or nonlinear model/feature
interaction: Brier can produce one under additive probability responses.

The same 2,000 whole-original-group draws, seed 42, apply to all candidates,
scenarios, repetitions and q levels. Unequal group sizes and crossing SEX/outcome
groups stay intact. Loss means/maxima, rates, signed and absolute gaps are
recomputed. **Chosen identities, fitted models and scenario realizations stay
fixed**; no selector is re-run. Percentile intervals have valid/undefined counts;
Monte Carlo repetition SD is separate. Frontier/relation frequencies describe
exploratory conditional stability, not true-superiority probabilities or joint
confidence certification. Fitting/selection variability is omitted.

Outputs preserve predictions, evaluation metadata, fitting references, common
input registries, repetition/endpoint diagnostics, coordinate and scenario tables,
all method mappings, pair/frontier results, bootstrap multiplicities/draw evidence
and summaries, labelled two-dimensional projections, an interpretation report,
source/input preservation and a verified receipt. Undefined numeric draw cells
in NPZ use NaN **with explicit validity flags**; JSON summaries retain nulls and
valid counts. No values are silently replaced with zero.

`reconstruction.py` reads only saved judge outputs. It independently checks point
losses with sklearn, manual strict-threshold signed gaps, maxima, contrasts and
Pareto definitions; checks direct record/repetition bootstrap calculations on
first/middle/final draws; and checks all saved draw maxima, absolute gaps, pair
relations and coordinate interval summaries. It never regenerates predictions.
It also checks all saved frontier memberships/completeness, principal relation
frequencies, scenario-loss and interaction intervals, and supporting clean
AUC/AP, overall rates and reliability bins. Root separately reviews and
reconstructs actual results.

```powershell
.venv/Scripts/python.exe -m unittest discover -s research/model_selection_v2/judge_v1 -p test_judge_v1.py -v
.venv/Scripts/python.exe -m research.model_selection_v2.judge_v1.runner --verify-run runs/model_selection_v2/judge_v1_2026-10-30
.venv/Scripts/python.exe -m research.model_selection_v2.judge_v1.runner --reconstruct-run runs/model_selection_v2/judge_v1_2026-10-30
```

Preserve same-choice, trade-off, null and negative findings. There is no practical
equivalence, legal, causal, real-fault-frequency, future-safety, universal-winner
or novelty claim. Explainability remains unassessed. A stronger claim needs a
separately frozen design and evidence not used to develop it; historical-holdout
tuning or a judge revision favoring a selector is not authorized.
