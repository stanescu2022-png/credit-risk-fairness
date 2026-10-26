# Integrated SAFE 2026: Taiwan operational specification v1

Resolved **26 October 2026, before credit-data scoring**. Primary source:
Giudici & Kolesnikov, *SAFE AI metrics: An integrated approach*,
[DOI 10.1016/j.mlwa.2025.100821](https://doi.org/10.1016/j.mlwa.2025.100821).
Local published PDF: `docs/context/sources/literature_2026-10-09/1-s2.0-S266682702500204X-main.pdf`,
SHA-256 `fbc611acf0cf21463b8a63e7febc9d989b4ef651a2b0e08c7b0d9f4aa7814bc4`.
Equations (4), (8), (10), (12)-(17) and the curve text were checked visually on
PDF pages 2-5. This is an application to Taiwan, not reproduction of HMDA tables.

## Accepted choices and boundaries

The user explicitly accepted: (a) additive full-sample area contributions for
RGA; (b) fixed-primary-model masking with fitting-set numeric means/categorical
modes, greedily preserving full-model predictions; and (c) percentile swapping
of all 22 original predictors, with Gaussian assessment **deferred** until its
categorical treatment is specified. No auxiliary models will be fitted.

All primary pipelines and their completed candidate run remain unchanged. SAFE
uses the saved 4,801 selection records and exact saved default probabilities.
Fitting records (19,203) supply masking references and support diagnostics only.
No holdout records are exposed to this assessment. No primary fitting, tuning,
calibration, threshold change, extended method, independent judge or fairness
aggregate is implemented.

## Fidelity and executable definitions

| Component | Paper support | Executable definition | Operational choice / limitation |
| --- | --- | --- | --- |
| Common rank kernel | Eqs. (1)-(4), (8), (10), (12), pp. 2-4 | For reference `y`, let `a=sort(y)`, `d=a[::-1]`, and `s` be reference values ordered by ascending comparison scores. Replace values within exact score ties by their block mean. `D=sum(cumsum(d-a))`; `h=cumsum(d-s)/D`; `K=sum(h)`. Equivalent to the rank-weighted final expression in Eq. (4). | Exact ties receive their expectation over tied orders, consistent with installed historical rank kernel and binary AUROC with half credit. No rounding of predictions. Nonnegative finite reference values required. Constant reference makes D=0 and is undefined; constant comparison gives K=.5 for nonconstant reference. No record dropping or NaN-to-zero. |
| Scalar RGA | Eq. (4), p. 2 | `K(selection_y, saved_p_default)` | Validate against independent binary pairwise AUROC with ties and sklearn. One-class outcome is undefined. This does not measure probability calibration. |
| RGA partial contributions | Eqs. (4)-(6), p. 3 | Partition the **full-sample** area terms `h` into 22 contiguous slices, starting at highest prediction rank (`h[::-1]`). `partial[k]=sum(h[slice_k])`, using the same full D throughout. Curve nodes: `[sum(h), sum(h)-partial[0], ..., 0]`. | User-approved resolution of underspecified “restricted slice” prose. Unlike the later package, no local rescore, renormalization of slices, perfect-reference area normalization or shrinking-sample AUROC. The partials add to scalar RGA algebraically. |
| RGA coordinates / area | Eq. (7), p. 3 | `array_split` gives approximately equal rank-cell segments (first remainder segments have one extra cell); x is actual cumulative removed rank-cell count/n, including 0 and 1. Right-continuous step uses node k on `[x[k],x[k+1])`; raw area is `sum(diff(x)*A[:-1])`. | Splitting within an exact score tie uses the tie-block expectation, never ID or outcome ordering. Contributions describe area cells, not an outcome-selected subset of people. Area and uniform node mean are saved separately. No normalized area substituted for raw curve values. |
| RGR | Eq. (8), pp. 3-4 | `K(saved_p_default, perturbed_p_default)` | Same scalar kernel; reference remains the original frozen candidate probability vector. No Spearman substitution, AUROC versus outcomes or accuracy rescaling. |
| Percentile swapping | Sec. 2.2, pp. 3-4 | At raw t in `[0,.5]`, sort each original predictor by `(value, original ID)`; m=floor(t*n), pair its lowest m rows with its highest m rows in reverse rank order and swap simultaneously from the original column. Apply to all 22 predictors. | Deterministic tail pairing and tie handling are Taiwan choices. IDs provide stable pairing only; never model inputs. Odd n leaves the middle row unswapped. Each column's multiset/category support is preserved. All candidates receive the exact same perturbed frame. One deterministic realization; no Monte Carlo spread claimed. |
| Robustness grid / area | Eq. (9), pp. 3-4 | 23 nodes: normalized p=k/22, raw t=.5*p, k=0..22. Record both coordinates. Area: trapezoidal integration of raw RGR against normalized p. | Grid size and piecewise-linear quadrature are declared discretization choices. The 0-.5 raw range follows the paper. Normalizing severity does not establish practical equivalence with another perturbation regime. |
| Categorical perturbation | Sec. 2.2 says each variable; categorical details absent | Swap original categorical codes and then apply the unchanged valid one-hot pipeline. Verify category-level support and all encoded blocks. Count categorical combinations absent from fitting support and from original selection support. | User-approved Taiwan convention. Observed categorical combinations are an empirical support diagnostic, not a definition of impossible combinations. Sorting nominal category codes is a mechanical severity convention, not a semantic adversarial attack. No independent encoded-indicator swapping. |
| Gaussian noise | Sec. 2.2, pp. 3-4 | **Not assessed in v1** | Paper states independent Gaussian noise with scale alpha*sigma_data on all features, but does not resolve categories/representation or scale sample. User chose deferral. No fabricated Gaussian scores or mixed-regime aggregate. |
| Original RGE | Eq. (10), p. 4 | `original_RGE(S)=1-K(p_full,p_masked(S))` | This is disagreement, not the preservation curve used in the aggregate. Keep both named outputs. |
| Reversed RGE / greedy search | Eq. (12), Fig. 5 and conclusion's greedy description, pp. 4-5, 11 | Start S=empty; evaluate every remaining original predictor j; choose j maximizing `K(p_full,p_masked(S union {j}))`; repeat until all 22 are masked. Reference is always p_full, not the previous step. | Paper leaves the exact greedy objective and refit/mask distinction unspecified. Fixed-model, least-damaging masking is the user-approved implementation, supported by later author-credited code. No claim of exact historical reproduction. A true nested greedy search; never a once-only importance order. |
| Masking references / units | Model “without predictor” in Sec. 2.3; grouped units required by project | For each original numeric field, use its fitting-only arithmetic mean. For categories use fitting-only mode, smallest category if mode ties. Set an entire original column before unchanged preprocessing; all its encoded columns move together. | Adaptation avoids invalid all-zero or fractional one-hot reference categories. No selection-based references, no model fitting. All-removal input is identical for all rows, so Eq. (12) yields .5 when p_full is nonconstant; no-removal yields 1. Endpoints computed, never overwritten to match plots. |
| Path ties / unavailable values | Not specified in paper | Absolute score tie tolerance 1e-12; choose earliest in saved feature order. Record every tied alternative. If any required greedy alternative is undefined, the full comparison is unresolved and that path/specification is unavailable. | Do not interpret undefined as poor performance. Tolerance is numerical, not a practical-equivalence margin. Preserve measured nonmonotonicity; no isotonic correction. |
| Explainability grid / area | Eqs. (11)-(12), p. 4 | 23 nodes k/22, including no/all removal. Area of reversed curve uses trapezoidal rule; also retain original-RGE curve/area. | A high preservation area is not proof of human explanation quality. |
| Integrated mean scores | Eqs. (13)-(17), p. 5 | Use raw A, reversed E and R vectors, each length 23. Construct every Cartesian cell `(A[i]+E[j]+R[k])/3`, `cbrt(A[i]*E[j]*R[k])`, `sqrt((A[i]^2+E[j]^2+R[k]^2)/3)`; uniformly average all 23^3 cells. | K+1 nodes, rather than ambiguous paper n=number-of-features, retains both endpoints. Weights are exactly 1/23 per node, independently per axis. Sample averages differ from quadrature areas. No accuracy rescaling, candidate min-max normalization or fairness addition. |
| Primary selector | User request, not authors' universal preference | Arithmetic score under percentile swapping; higher preferred. Geometric/quadratic swapping are sensitivity variants. | Scores concern this finite pool. For final-score ties (absolute tolerance 1e-12), retain tied IDs and choose first in fixed pilot candidate order for a required single ID. Required undefined components make the candidate unrankable, not zero. |
| TOPSIS | Sec. 2.4 and Tables 8-10 | **Not implemented** | Published account does not fully identify its raw-input matrix, normalizations, directions, weights and ideal points. An arbitrary standard procedure would not be exact reproduction. This does not block mean scores. |

Roundoff within 1e-12 is checked and only tiny area terms/endpoints within this
bound may be snapped to mathematical zero; the count and tolerance are recorded.
Substantial negative/out-of-range values are errors, never silently clipped.
RGR/RGE empirical paths need not be monotone despite the paper's descriptive
claims; any increases are retained and reported.

## Author-credited source inspection

Inspected [safe-ai-metrics 0.2.0 on PyPI](https://pypi.org/project/safe-ai-metrics/0.2.0/),
author credited as Vasily Kolesnikov; released **10 June 2026**, after the paper
became available online on **15 December 2025**. The wheel SHA-256 is
`454d724b448a941345b94016d90e048b7d7c15fda0f18feb48434832ac9e6303`.
The wheel and relevant modules are preserved in `safe2026_sources/`, inspected
without installation or imports into the research environment. This is distinct
from installed safeaipackage 0.8.3. Its release/author metadata does not establish
that this code generated the paper's results.

Relevant differences:

- `rga._binary_rga_curve_partial`: local slice scores/Gini weights, replaces
  undefined slices with zero, rescales partials to total RGA, defaults to
  perfect-reference area normalization. These are not imported here.
- `cramer` functions use a weighted absolute curve divergence whose scalar
  implementation differs from Eq. (4) and can depend on row order within ties.
  Synthetic counterexamples are retained in `source_inspection.json`.
- `rge._rge_curve_tabular_core`: frozen-model greedy masking maximizing its
  preservation score; defaults to zero masking, optional input-based means,
  replaces undefined values, supports RGA rescaling. This supports masking as an
  operational precedent, but its baselines/kernel/rescaling are not adopted.
- `rgr._rgr_curve_noise_core`: Gaussian perturbations and optional RGA rescaling;
  undefined values become zero. No percentile-swap implementation was identified
  in the inspected rgr module. These conventions do not resolve paper fidelity.

## Verification and interpretation

Synthetic tests precede scoring: binary AUROC/ties; identity/reverse/constant
cases; additive accounting; row-permutation invariance; zero-strength identity;
tail pairing/category support; deterministic coupling; whole-original-feature
masking; exhaustive small greedy choice; fitting-only references; endpoints;
cellwise aggregation, including the presentation example = .6166666666666667.
Final scores are independently reconstructed from saved curves and saved aligned
prediction evidence. Primary and historical artifact hashes are checked before
and after; no historical evaluation is rerun. Plots are visually checked.

The Cartesian grid is a deterministic construction, not a sampling distribution
or measured simultaneous real-world faults. No sampling intervals or Monte Carlo
spread are inferred from it. Results establish neither independent predictive
superiority, human explanation quality, legal compliance, real-world robustness,
nor a winner against the future methodology. The independent judge remains an
explicit later decision; Brier/LogLoss and any explainability criterion are still
provisional.
