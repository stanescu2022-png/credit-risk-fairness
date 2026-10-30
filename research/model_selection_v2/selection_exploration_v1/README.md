# Selection-only exploration v1 implementation

Prepared **30 October 2026** against the root-owned
[exploratory specification](../exploration_v1_specification.md). This folder
implements that version; it is not an independent judge, a new confirmation
study or authority to use the historical holdout. Empirical execution awaits
root review. No empirical run has been executed by this implementation task.

`protocol.py` fixes the declared families, .10 reference / .20 sensitivity,
30 realizations, seed scopes, all ten selectors and conditional bootstrap
settings. `transforms.py` calls existing semantic rule functions read-only,
recomputes means/modes/scales/support on fitting records only, shares record
selection/sign/category streams, and verifies disjoint-transform commutation.
`analysis.py` averages per-record losses over realizations before applying group
multiplicities and takes scenario maxima **after** repetition means. It retains
per-realization losses/Monte Carlo SD and exact-zero/one/LogLoss clipping counts.

The runner reuses `safe2026_run.load_contract`, verifying SAFE and pilot receipts
and bound source/input hashes before loading models. It copies the exact
specification bytes and hash before prediction, binds each repetition/condition
to common input hashes and ordered/selected/changed/reused row IDs, and predicts
only changed rows. Every candidate's repetition-1 output is checked against full
frame prediction for every family/q/condition, at absolute 1e-15. This numerical
audit tolerance does not enter Pareto dominance. Model state and source/input
preservation checks precede completion.

The bootstrap uses the same 2,000 whole-original-group draws / seed 42 for all
candidates, scenarios, severities and saved realizations. It re-evaluates the
added outcome layer while holding saved SAFE scores fixed. Intervals concern
fixed candidates and all six fixed pairs, later pilot-order ID minus earlier,
and signed candidate loss-scale interaction contrasts. This is not complete
SAFE/selector uncertainty or joint confidence certification. Repetition spread
stays separate from group sampling intervals. Missing required coordinates
make the affected selector unresolved, without dropping candidates or axes.

When all candidates have distinct vectors on the smaller coordinate set,
adding coordinates expands its nondominated set, and with well-separated
SAFE scores the selected retained SAFE score cannot decrease. This is **not
unconditional**: exact old-vector ties can split on added axes, removing a
former frontier member and decreasing the selected SAFE score. Both cases
are preserved in synthetic tests. Choosing the highest-SAFE nondominated candidate
does **not** guarantee a replacement that dominates the original SAFE choice;
a synthetic counterexample preserves this limitation. The prescribed factorial
contrast describes interaction on the chosen **loss scale**. It is not proof
of causal synergy or nonlinear model/feature interaction; Brier can have a
nonzero contrast even under additive probability responses.

Focused synthetic checks (no empirical loaders or fitting):

```powershell
.venv/Scripts/python.exe -m unittest discover -s research/model_selection_v2/selection_exploration_v1 -p test_exploration.py -v
```

After root review, the prepared execution command is:

```powershell
.venv/Scripts/python.exe -m research.model_selection_v2.selection_exploration_v1.runner --run-dir runs/model_selection_v2/selection_exploration_v1_2026-10-30
```

The destination must be fresh; existing directories/files are refused. Failed
runs retain an incomplete marker and failure details, without `VERIFIED.json`.
Outputs include the specification/settings, fitting references, seed/input/ID
registries, saved predictions, repetition/scenario losses, endpoint counts,
interactions, all selectors/ablations, bootstrap multiplicities/summaries,
reuse audits, preservation/source/input manifests and a hash-bound receipt.
`--verify-run <directory>` only verifies saved artifact integrity; it performs
no prediction, fitting, data access or output regeneration.

Preserve negative, same-choice and inconclusive results. These inspected
selection data support hypothesis development, not selector superiority.
Independent validation, judge freezing and future full-procedure resampling
remain separate. No holdout tuning, outcome-driven judge revision, recalibration,
empirical transformed-wrapper candidates, category pooling or fitting occurs.
