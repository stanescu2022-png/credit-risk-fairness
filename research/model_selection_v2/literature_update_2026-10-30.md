# Targeted prior-art update

Date: **30 October 2026**. Purpose: constrain the emerging model-selection
contribution. This is a targeted review, not an exhaustive novelty search.
Existing archived sources remain unchanged. Web references below were consulted
through search/open; no author repository code was run or adopted.

## Sources and implications

1. **Giudici & Kolesnikov (2026), SAFE AI metrics: An integrated approach**,
   Machine Learning with Applications 23, 100821.
   [DOI](https://doi.org/10.1016/j.mlwa.2025.100821).
   Archived PDF SHA-256:
   `fbc611acf0cf21463b8a63e7febc9d989b4ef651a2b0e08c7b0d9f4aa7814bc4`.
   Rechecked printed equations (4), (8), (10), (12)-(17), PDF pages 2-5 visually;
   consulted retained column-aware text for discussion on page 9. Scalar affine
   cancellation follows from weighted-sum differences. The Cartesian aggregation
   combines separately obtained curves; it does not observe the full response
   surface under combined interventions. The paper already identifies conflicting
   perturbation-family recommendations and multi-scenario selection as future work.
   Our interpretation concerns decision scope, with Taiwan adaptations disclosed.

2. **Babaei & Giudici (2025), Correlation Metrics for Safe Artificial Intelligence**,
   Risks 13(9), 178. [DOI](https://doi.org/10.3390/risks13090178).
   Archived PDF SHA-256:
   `fbc50a6261d9a62dac64b4271263a508c0e8e91790d5319b53dd33c705cefb6f`.
   Printed page 4, equation (3), visually rechecked: binary predictive R2 is
   `1-Brier/[prevalence*(1-prevalence)]` for nondegenerate outcomes. Thus it ranks
   candidates like Brier on the same sample. Probability sensitivity is existing
   SAFE work. The publisher web endpoint was rate-limited; the archived original
   provides the inspected evidence. Do not treat its ambiguous removal/sign
   conventions as an already specified integrated selection rule.

3. **Sokolov, Giudici & Kolesnikov (2026), Confidence regions for SAFE AI compliance
   scores**, preprint. [Abstract](https://arxiv.org/abs/2609.29588),
   [versioned HTML](https://arxiv.org/html/2609.29588v1).
   Inspected abstract and methodological introduction, not independently
   reproduced. It develops covariance-aware uncertainty for integrated SAFE using
   paired bootstrap and discusses shared-sample component dependence. This is
   direct overlap with an uncertainty-only extension. The record displays an
   August 2026 date despite its 2609 identifier; retain the versioned URL and
   verify bibliographic metadata before manuscript submission. Peer-reviewed
   publication status was not established here.

4. **Laufer-Goldshtein, Fisch, Barzilay & Jaakkola (2023), Efficiently Controlling
   Multiple Risks with Pareto Testing**, ICLR 2023.
   [Paper](https://openreview.net/pdf?id=cyg2YXn_BqF).
   Initially inspected through its indexed primary abstract; direct OpenReview
   opening encountered browser verification. On resumed review, inspected the
   [arXiv v1 methods](https://arxiv.org/pdf/2210.07913v1), pages 4-6, including
   Algorithm 1 and Proposition 5.1. The preprint is dated 14 October 2022;
   ICLR publication is 2023. It splits i.i.d. calibration data into optimization
   and testing subsets, constructs/orders a Pareto set on the first, then applies
   fixed-sequence testing on the second with specified risk bounds and valid
   p-values. Remaining unconstrained objectives guide the final selection.
   Our interpretation: empirical screening plus conditional group-bootstrap
   diagnostics does not inherit that guarantee. No risk limits or valid testing
   procedure are specified in our v0. Using this as an executable comparator
   would require an explicit compatible calibration/testing design and treatment
   of our grouping/estimands; it cannot be added by renaming our frontier. Author
   code and proof have not been independently reproduced; conference-version
   fidelity remains to be checked before relying on a theorem.

5. **Laufer-Goldshtein, Fisch, Barzilay & Jaakkola, Risk-Controlling Model Selection
   via Guided Bayesian Optimization**. Preprint submitted 2023.
   [Abstract](https://arxiv.org/abs/2312.01692),
   [versioned HTML](https://arxiv.org/html/2312.01692v1).
   Inspected abstract/introduction and its explanation of Pareto Testing. It
   focuses the search on relevant Pareto configurations and statistically verifies
   risk constraints. This rules out claiming that Pareto screening plus risk
   assessment is new in general. Our current exploratory Pareto relations are
   not finite-sample risk-control certificates.

6. **Gneiting & Raftery (2007), Strictly Proper Scoring Rules, Prediction, and
   Estimation**, JASA 102(477), 359-378.
   [Author-hosted paper](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).
   Consulted indexed primary abstract for the proper-scoring rationale. The
   expected-loss argument in the research agenda uses elementary binary squared
   loss algebra; no new proper-scoring theorem is claimed.

## Search scope and exclusions

Queries covered integrated SAFE calibration/scenarios, probability-sensitive
SAFE, SAFE affine invariance, Pareto risk-controlling model selection and tabular
joint corruption. Search also surfaced tabular missingness/noise and corruption
benchmarking literature. Those hits have not yet received a full methods review,
so they do not establish a novelty gap for joint-corruption assessment. No absence
claim is supported by this search.

The defensible next question is whether a specific correction improves selection
under explicitly stated risks **beyond** unchanged SAFE, clean-loss selection and
simple robust-loss selection. Stronger empirical design and fair ablations are
necessary; renaming existing ingredients is not enough.
