# Pure prediction-table judge foundation

Implemented **30 October 2026**. This is reusable computation infrastructure,
not a completed scientific protocol, accepted metric panel, primary evaluation
or empirical judge run. It contains no file/data/model loading, fitting,
calibration, prediction generation or role switching. No empirical run or
receipt is created. Improvement remains a hypothesis; this core does not
authorize historical-holdout development or changing a judge to favor a choice.

## Inputs and API

`PredictionTable.from_columns(columns, role_label=..., column_map=...)` accepts
an in-memory mapping or DataFrame. Canonical columns are `ID`, `outcome`, `SEX`,
`original_group`, `probability`. `column_map` optionally maps canonical keys to
source column names. IDs must be unique; outcomes binary 0/1; SEX uses project
codes 1 male / 2 female; probabilities must be finite in [0,1]. Supplied groups
must come from the **whole original 22-predictor representation**, even for
reduced models. This cannot be established from a probability table alone.

Arrays are copied and marked read-only. Exact row order, IDs, outcomes, SEX,
groups and the explicit role label must agree across every candidate and
scenario. No automatic reordering or dropped records. A group may contain
different outcomes and both sexes; those crossings are retained.

`CandidateTables(clean, scenarios={name: sequence_of_repetition_tables})`
holds fixed outputs. Scenario names are explicit; `clean` is reserved. Every
scenario needs at least one repetition. Counts for a scenario supplied by
multiple candidates must agree. Repetition positions must refer to the same
perturbed inputs across candidates; the caller must bind their identities.
Output alignment alone cannot verify common perturbations or correct roles.

`JudgeConfig(metric_directions, our_choice_id, safe_choice_id, candidate_order,
role_label, primary_evaluation=None)` requires all fields explicitly. Directions
map chosen metric names to `min` or `max`; no metric vector or primary metric is
selected here. `primary_evaluation` must explicitly be `None` while undecided.
Both selected IDs must be supplied and appear in the explicit candidate order.

- `losses(table)` computes Brier on unmodified probabilities. LogLoss clips
  float64 probabilities to `[eps,1-eps]`, `eps=np.finfo(float64).eps=2^-52`,
  and uses `log` / `log1p`. Endpoints remain finite; Brier is never clipped.
- `fairness_gaps(table)` returns TPR/FPR, counts and denominators by SEX, signed
  gaps **female minus male**, and their absolute values. Decisions use strict
  `p>.5`; ties are class 0. Zero-denominator rates/gaps are `None`.
- `scenario_losses(clean, scenarios)` returns each repetition's losses and
  their arithmetic means. It never computes loss of mean probability.
- `candidate_metrics(candidate)` returns available flat diagnostic names:
  `clean.brier`, `clean.log_loss`, `clean.tpr_gap_signed`,
  `clean.tpr_gap_absolute`, `clean.fpr_gap_signed`, `clean.fpr_gap_absolute`,
  and `<scenario>.brier` / `<scenario>.log_loss`. Signed gaps remain available
  for interpretation/intervals without being automatically selected as judge
  coordinates.
- `evaluate_candidates(candidate_tables, config)` returns available metrics,
  the explicit-choice comparison and pool frontier. `pareto_relation`,
  `compare_choices` and `pool_frontier` also accept independently supplied
  metric mappings, allowing future coordinates without changing this core.

Pareto dominance compares unrounded float64 values exactly, without tolerance,
normalization or practical margins. Better on at least one coordinate and worse
on none is dominance; equal, incomparable and unresolved are distinct. Missing
or `None` required coordinates make a comparison unresolved. NaN/inf are errors.
Same candidate identity is reported separately from different candidates with
equal measures. Raw paired differences are named `our_choice_minus_SAFE_choice`.

The frontier follows the explicit full candidate order. If a candidate is
absent or has missing coordinates, the output warns that the reported frontier
is conditional on the complete observed subset; unresolved candidates are not
silently classified as dominated or frontier members.

## Fixed-prediction uncertainty

`group_bootstrap_plan(reference, n_draws=..., seed=...)` draws G whole original
groups with replacement from G observed groups. **No draw-count/seed defaults**
are supplied. Groups are enumerated by first occurrence in the supplied row
order; the complete enumeration and draw counts are returned for reproduction.
A later protocol must explicitly choose values, including any
use of 2,000 draws / seed 42. `BootstrapPlan(reference, group_multiplicities,
seed=None)` accepts explicit counts for deterministic checks. Every draw's group
counts sum to G. `plan.row_multiplicities(draw)` gives identical multiplicities
to every record in a group, including crossing SEX/outcome groups. Group sizes
remain intact: metrics weight resampled records, not equally weighted group
averages. Resampled record totals can vary.

`bootstrap_with_plan(candidates, config, plan, percentile_bounds=...)` shares
that plan across every candidate, scenario and repetition.
`bootstrap_judge(..., n_draws=..., seed=..., percentile_bounds=...)` is the
equivalent convenience wrapper. Both require explicit percentile bounds.
`percentile_summary` uses NumPy's linear interpolation and reports valid,
undefined and total draw counts. Candidate diagnostic intervals include signed
sex gaps; paired intervals preserve the raw difference direction. Repetition
losses are averaged inside each group draw; repetition indices are not
resampled, and Monte Carlo repetition spread is not a sampling interval.

Results include explicit group counts, row-group mapping, draw-level metrics
and differences, exploratory dominance frequencies with resolved/undefined
denominators, and frontier frequencies among complete-pool draws only. These
condition on fixed supplied models, choices, outputs and scenarios. They omit
fitting/selection uncertainty. Marginal intervals and dominance frequencies
are **not joint confidence certification** or independent superiority evidence.

Focused synthetic checks:

```powershell
.venv/Scripts/python.exe -m unittest discover -s research/model_selection_v2/judge_core -p test_judge_core.py -v
```
