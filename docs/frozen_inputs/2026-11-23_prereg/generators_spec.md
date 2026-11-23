# Lean synthetic generators

21 November 2026. Implementation specification for R and F. The generators and reference checks use artificial records. The study's fitted timing and production are separate stages. See the [synthetic protocol](protocol.md) and [registration](registration.md).

## Common population and seed rules

- Each repeat draws 30,000 independent borrowers. Fields are in the listed order and use float64, including integer-valued fields. IDs run from 1 through 30,000 in original draw order. Sex is coded 1 for male and 2 for female, with independent Bernoulli(.60) female draws. Draw labels from Bernoulli(p) using each original borrower’s generated probability. Only the 15 predictors enter the model. Retain the original probability by ID for truth scoring.
- Each independent field stream uses NumPy `default_rng` with an unsigned integer seed. Hash the UTF-8 string `20261117:<purpose>:<setting>:<repeat>:<field>` with SHA-256 and interpret its first eight bytes as little-endian. The stream names for the draws are the output column names, plus `sex` and `outcome`. The three F lognormal residuals use their corresponding output field streams. Reference draws use purpose `reference`, their own setting, and repeat 0. Timing and production use distinct purpose tokens. The pipeline passes `synthetic_lean:<purpose>:<setting>:<repeat>` to the judge for population-specific, paired faults.
- `calibrate` computes label-free constants once per setting. Save the constants with the preparation outputs and use the same values across timing and production. There is no per-repeat recalibration. A non-finite or degenerate reference, failed target solve, or incompatible calibration stops the generator. The reference checks have no model-fitting or selection step.

## R, a fragile signed-tail interaction

Draw independent standard normal `x1` through `x15`. Put `x1, x2, x5, x6, x7` in family A, `x3, x8, x9, x10, x11` in B, and `x4, x12, x13, x14, x15` in C. Draw sex independently of the predictors. This is the declared logit.

\[
\ell_i=\alpha+.8(x_{3i}+x_{4i})+.4h_i+.8T_i h_i .
\]

\(T\) equals +1 above the standard normal 95th percentile, −1 below the standard normal 5th percentile, and zero between the two thresholds including their endpoints. \(h\) equals +1 where \(|x_2|\) exceeds the 75th percentile of the standard normal and −1 otherwise. The population mean risk is .22. Solve for \(\alpha\) using 64-point Gauss–Hermite integration of the normal background, with \(T\) masses .05, .90, .05 and the two equally likely \(h\) states.

### Reference curve before fitting

Draw a separate, fixed 1,000,000-row R reference using the reference streams. Record signed-tail and outside/inside central-half counts and mean original risk. Let the full forecast be the generated logistic formula and let its rival omit `.8*T*h` while sharing the intercept, `.4*h` and both background coefficients. At each of the six registered nodes \(q=0,.01,.02,.03,.04,.05\), use the frozen `safe2026.tail_swap` on all 15 original columns, which swaps `floor(q*n)` opposite empirical ranks per tail with ties broken by ID. Evaluate both forecast formulas on the same swapped rows and compare them to each row’s original probability \(p_i\).

Report full and term-omitted mean squared truth error at every node, the full-minus-omitted difference, the ordinary six-node means, the preferred reference forecast for clean and six-node mean loss, and the count of rows where a swap changes or reverses the original interaction signal. Compare to the fixed-background, constant-magnitude \(a\) approximation. Its losses in units of \(a^2\) are \(8\min(q,.05)\) for the full forecast and .10 for the omitted forecast. The approximation predicts a crossover at .0125, a clean preference for the full forecast, and a six-node mean preference for omission (.20 vs .10). The actual logistic reference includes simultaneous swaps to the background and varying probability effects. Record differences from the approximation without using the curve to modify either generator.

## F, one group-specific trade-off

The fields and judge families follow the 16 November draft, with the following exact draws and category codes. Random draws are independent except for the stated links. Let \(W=1\) for women and zero for men. Let \(E\) be education, \(I\) monthly income and \(D\) existing debt.

| Family | Field | Type | Distribution |
| --- | --- | --- | --- |
| A | `age` | Numeric | Discrete uniform, 21 through 70. |
| A | `dependants` | Numeric | Poisson(1), capped at 6. |
| A | `education` | Ordered | Codes 1–4 with shares .15, .40, .30, .15. |
| A | `marital_status` | Unordered | Codes 1–3 with shares .45, .45, .10. |
| A | `residence_type` | Unordered | Codes 1–4 with shares .40, .30, .20, .10. |
| B | `monthly_income` | Numeric | \(I=3000\exp[.15(E-2)+.5u]\,.9^W\), where \(u\) is standard normal. |
| B | `credit_limit_requested` | Numeric | \(3I\exp(2v/3)\), with standard normal \(v\). |
| B | `existing_debt` | Numeric | \(D=18000(I/3000)^{.3}\exp(.477w)\), with standard normal \(w\). |
| B | `months_in_job` | Numeric | Gamma(shape 4, scale \([60+.6(\text{age}-45.5)]/4\)). |
| B | `open_credit_lines` | Numeric | Poisson(2), capped at 10. |
| C | `worst_delay` | Ordered | Delay \(d\) has codes −1 to 4 with shares .30, .45, .12, .07, .04, .02. |
| C | `months_with_delay` | Numeric | Zero when \(d\le0\), otherwise \(1+\mathrm{Binomial}(5,d/5)\). |
| C | `utilisation` | Numeric | Beta(1.8, 2.2), with support from zero to one. |
| C | `product_type` | Unordered | Codes 1–3 with shares .60, .30, .10. |
| C | `past_restructuring` | Unordered | Bernoulli(.05), code 1 for yes. |

The common L2 logit is \(\alpha+s\operatorname{std}(.33L+.67N)\). `std` subtracts the fixed reference mean and divides by the fixed population standard deviation (`ddof=0`). Use the same reference standardization individually for every term in \(L\) and \(N\).

- \(L\) sums ten standardized terms with weight +1 or −1 each. Higher risk terms are `existing_debt`, `open_credit_lines`, `worst_delay`, `months_with_delay`, `utilisation` and `past_restructuring`. Lower risk terms are `age`, `education`, `monthly_income` and `months_in_job`. The truth function uses the integer codes for the ordered terms and the binary restructuring indicator. Fields absent from this list have zero linear weight. Candidate preprocessing uses the existing confirmation schema and its frozen encodings.
- \(N\) sums the five individually standardized terms \(1\{\text{utilisation}>.9\}\), \(1\{d\ge2\;\text{and}\;\text{utilisation}>.6\}\), \((\text{age}-45.5)^2\), \(-\min(I,\mathrm{median}(I))\) and \(1\{D>6I\}\), all with equal positive weight. The income median is calculated on the pooled reference.
- Use a single independent 1,000,000-row F-layout reference containing both sexes to set all medians, means and scales. Apply the common L2 formula to both sexes on this reference while solving for \(\alpha\) and \(s\). Target mean risk .22 and risk-weighted AUC .78. For that AUC, order by risk and weight each row as a positive by \(p_i\) and as a negative by \(1-p_i\). Average ties with weight one-half. No Bernoulli outcome is needed.
- For every female row, replace the standardized debt jump in \(N\) by \(.5\operatorname{std}(D-6I)\), using the pooled F-layout reference for the mean and scale of \(D-6I\). Keep the original L2 intercept, risk scale and standardization of the final mixture. Report achieved F prevalence and risk-weighted AUC from the reference after applying the replacement. The comparator for males remains the common L2 function.

The one F setting is therefore fixed before fitting. The female replacement is linear in the observed debt and income fields within women. The rest of the F truth includes common nonlinear terms, and the candidate models omit sex. The generator does not guarantee that any chosen candidate harms either group.
