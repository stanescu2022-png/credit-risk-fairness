# Step 1 specification: accuracy-anchored SAFE versus published SAFE

2026-11-14 · @Andrei

## 1. Claim, scope and status

Step 1 tests whether SAFE with its robustness axis anchored in accuracy chooses a model that an independent judge scores better than the model published SAFE chooses.

**Scientific question: order versus accuracy.** Published SAFE measures robustness as preservation of the order of predictions under disturbance. Selector 2 measures it as accuracy retained under disturbance. Everything else in SAFE stays as published.

**Claim under test (direction only)**

- The model chosen by Selector 2 has lower stressed Brier under the judge than the model chosen by SAFE, with an interval that excludes zero.
- No minimum size of improvement is set. The report states the size and does not call it material.
- Beating plain loss selection is not required. It is reported as a third arm.

**Status of this run**

This is a development run on Taiwan. Selector 2 was designed after Taiwan results were seen, so Taiwan can illustrate the method and cannot confirm it. Confirmation needs fresh datasets under this same specification (section 10).

**Already known before this run**

- On the original split, SAFE chooses logistic regression. Random forest and gradient boosting have lower Brier and log loss on the selection set and on the historical evaluation set.
- Step 0 showed that random forest keeps its accuracy better than logistic under swaps at every severity level, and is the most order-preserving model at mild levels.
- Rough arithmetic on saved Step 0 values suggests that, on the original selection set, Selector 2 would choose random forest, and so would published SAFE if its order measure were averaged over mild levels only.

The result on the original split is therefore largely predictable. The new information is how the selectors behave across repeated splits, with the whole procedure rerun each time.

**Tone**

The article is a friendly extension of SAFE. Reports from this run describe differences plainly and make no claim that SAFE is wrong.

## 2. Fixed inputs and limits

The run reuses the existing candidates, SAFE implementation and judge disturbances unchanged, and it refits models only on Taiwan development records.

**Reused without modification** (paths under `research/model_selection_v2/`)

| What | Where | Use |
| --- | --- | --- |
| Four candidate configurations and preprocessing | `pilot.py`: `build_candidate`, `positive_probability`, `CANDIDATES` | Refit in every repeat with the same settings. No tuning. |
| SAFE kernel, accuracy curve, swaps, masking, integration, ranking | `safe2026.py`: `rank_graduation`, `accuracy_curve`, `tail_swap`, `fitting_references`, `mask_original`, `greedy_preservation`, `integrate`, `rank_candidates` | Published SAFE as operationalized in `safe2026_specification.md`. |
| Judge disturbances | `selection_exploration_v1/transforms.py`: `paired_transforms`, `predict_changed_rows` | Feed-fault scenarios at q = .10, 30 repetitions. |
| Loss metrics and bootstrap conventions | `judge_core/metrics.py`, `judge_core/bootstrap.py` | Brier, log loss, quantile convention. |

Import these modules; do not copy or edit them. Record their hashes.

**Data**

- Use only the 24,004 Taiwan development records (the pilot's fitting and selection roles together).
- The 5,996 historical evaluation records are not loaded into any model, prediction or score. Assert this by ID.
- Recorded sex is never a model input. `SEX` 1 is male, 2 is female.

**Not allowed**

- Hyperparameter tuning, calibration, threshold changes or new candidate models.
- Editing or overwriting any existing file. Use a new package (suggested `step1_v1/`) and a fresh run directory that refuses an existing destination.
- Changing any definition, grid, seed or summary in this document after results are seen. Any change is a new, separately labelled version.
- Commit, push, publication or external contact.

**Sign conventions**

- Brier and log loss are lower-is-better.
- Every contrast between selectors is the other arm's chosen model minus SAFE's chosen model. Negative favours the other arm.
- Sex contrasts are female minus male.

## 3. Selector 2: accuracy-anchored SAFE

Selector 2 is the SAFE score with one change: the robustness axis measures accuracy retained under mild disturbance, in place of order preserved over the full severity grid.

### 3.1 Mild disturbance grid

- Apply SAFE's own swap, `tail_swap`, to the selection predictors of the repeat.
- Use six nodes: tail fractions 0, 0.01, 0.02, 0.03, 0.04 and 0.05.
- Predict with each fitted candidate at each node.

The range comes from the perturbation design. Five percent is the swap used in Babaei and Giudici's application (Risks 13:178, p. 8). Fixed fractions make the grid the same for any dataset, whatever its number of predictors.

### 3.2 Accuracy retained at a node

Write q_k for a candidate's predictions at node k, y for the outcome and ybar for the default rate, all on the selection records of the repeat.

```latex
S_k \;=\; 1-\frac{\tfrac{1}{n}\sum_i (q_{k,i}-y_i)^2}{\bar{y}\,(1-\bar{y})}
```

This is the Brier-based R-squared of Babaei and Giudici Eq. (3), applied to disturbed predictions. It is 1 for perfect forecasts, 0 for a forecast no better than the default rate, and negative below that. Do not clip it.

### 3.3 Score

```latex
\text{Score}_2 \;=\; \tfrac{1}{3}\left(\bar{A}+\bar{E}+\bar{S}\right)
```

- A-bar and E-bar are the uniform node means of SAFE's accuracy and explainability curves, taken from the SAFE computation of the same repeat, unchanged.
- S-bar is the uniform mean of S over the six mild nodes.
- Compute the score directly. Do not pass S through `integrate`, which requires equal-length curves in [0, 1].
- There are no geometric or quadratic versions of Selector 2.
- Rank candidates by Score 2, higher is better. Ties within 1e-12 go to fixed pilot order, as in SAFE.

### 3.4 Fairness check before a switch

A switch away from SAFE's choice must not make stressed accuracy resolvedly worse for women or for men.

For a challenger c and SAFE's choice s, on the selection records of the repeat:

1. For each record, average the stressed squared error over mild nodes 0.01 to 0.05, for c and for s.
2. For each sex, d is the mean of c's value minus the mean of s's value. Positive d means the challenger is worse for that group.
3. Form joint 95% intervals for the two values of d with the simultaneous construction of Step 0 section 7.3: 2,000 bootstrap draws of whole predictor groups from the selection records, SD with ddof = 1, 0.95 quantile with `method='higher'`, and the same zero-variance convention.
4. The challenger is **blocked** if, for either sex, the whole joint interval lies above zero.

The fitted models and disturbed predictions are held fixed inside this check. Use bootstrap seed 42 plus the repeat index.

### 3.5 Choice rule

Go down the Score 2 ranking from the top. Choose the first candidate that either is SAFE's choice or is not blocked.

Record, for every repeat: the Score 2 ranking, the top candidate before the check, every blocked candidate with its two intervals, and the final choice.

### 3.6 What accompanies every choice

For the chosen model and for SAFE's choice, on selection records: clean Brier, mean stressed Brier over the mild nodes and their difference, for women and for men, with 95% percentile intervals from the same draws.

## 4. Comparison arms

Six arms choose from the same four fitted candidates in every repeat: the two selectors under test, one reference arm and three diagnostic variants.

| Arm | Robustness axis | Severity range | Role |
| --- | --- | --- | --- |
| SAFE 2026 | Order preserved, K(p, q) | SAFE's full 23-node grid | Selector 1, exactly as published and implemented |
| Selector 2 | Accuracy retained, S | Six mild nodes | Selector 2, with the fairness check of section 3.4 |
| Plain loss | None: lowest clean Brier on selection records | Not applicable | Reference arm. Ties within 1e-12 go to pilot order. |
| Variant: order, mild | Order preserved, K(p, q) | Six mild nodes | Diagnostic only |
| Variant: accuracy, full | Accuracy retained, S | SAFE's 23 nodes | Diagnostic only |
| Selector 2 before the check | Accuracy retained, S | Six mild nodes | Diagnostic only: the top-ranked candidate by Score 2, before the fairness check |

**Why the two variants are there**

Selector 2 differs from SAFE in two ways at once: what is measured (order or accuracy) and where it is measured (full grid or mild range). The variants change one of these at a time, so the report can say which change drives any difference in the chosen model. On the original Taiwan selection set, rough arithmetic suggests that restricting the range alone may already change SAFE's choice. The run must show this honestly, whichever way it comes out.

**Rules for the variants**

- Both use the score form of section 3.3, with their own robustness mean in place of S-bar, and A-bar and E-bar unchanged.
- "Order, mild" uses the mean of K(clean predictions, disturbed predictions) over the six mild nodes, with K from `rank_graduation`. At tail fraction 0 use K = 1, as SAFE does.
- "Accuracy, full" uses the mean of S over SAFE's 23 swap nodes. Values may be negative.
- No fairness check is applied to the variants or to plain loss.
- The variants are not selectors for the claim in section 1.

## 5. Independent judge

The judge scores every candidate on records that no selector saw, under disturbances of a different kind from the swaps the selectors use.

**Records**

The judge role of the repeat (section 6). These records are never used to fit a model, to compute any selector's score or to run the fairness check.

**Disturbances**

- The nine existing feed-fault scenarios: three predictor families (repayment status, billing amounts, payment amounts) by three modes (mask only, corruption only, joint).
- Severity q = .10 and 30 realizations per family, generated with `paired_transforms` unchanged, with the fitting records of the repeat supplying reference values.
- These are masking and corruption of a random tenth of records. The selectors use deterministic tail swaps. Neither selector optimizes the judge's disturbances directly.

**Scores, for all four candidates in every repeat**

| Quantity | Definition |
| --- | --- |
| Clean Brier, clean log loss | On undisturbed judge records |
| Stressed Brier per scenario | Loss within each realization, then the mean over the 30 realizations |
| **Primary endpoint** | Mean stressed Brier over the nine scenarios |
| Mean stressed log loss | Same construction with log loss |
| Worst-scenario stressed Brier | Largest of the nine scenario means; descriptive only |
| By sex | Clean Brier, primary endpoint and their difference (deterioration), for women and for men |

Use the existing metric functions and their log-loss convention. Never compute a loss from averaged probabilities.

**From scores to arms**

Each arm's result in a repeat is the judge score of the candidate that arm chose in that repeat. Scoring all four candidates lets every arm be evaluated from one table.

**What the judge does not do**

It does not choose a model, tune anything or feed back into any selector. Its scenarios and endpoint are fixed here and do not change after results.

## 6. Repeated whole-procedure design on Taiwan

The whole procedure (split, fit, select, judge) is rerun 50 times on the development records, so that the results reflect the selection procedure and not one fitted set of models.

**Number of repeats**

J = 50, fixed in advance. With overlapping splits, the uncertainty is dominated by the overlap between repeats, so more repeats would add little (section 7.2).

**Split in repeat r (r = 0 to 49)**

- Seed: 20261114 + r.
- Randomly permute the original-predictor groups of the 24,004 development records.
- Assign whole groups in that order: judge until it holds 20% of records, then selection until it holds 16%, the rest to fitting (about 64%).
- No stratification. A group never crosses roles.
- Require each role to contain both outcomes for both sexes. If a repeat fails this, record it and stop.
- Save the assignment, the role sizes and the default counts by sex for every repeat.

**Steps in each repeat**

1. Split the development records as above.
2. Fit the four candidates on the fitting records with the pilot settings and seeds.
3. Predict clean probabilities on the selection and judge records.
4. Compute published SAFE on the selection records: accuracy curve, greedy masking path with fitting-record references, 23-node swap curve, integrated scores and ranking.
5. Compute the mild-grid swap predictions and Selector 2: S at each node, Score 2, the fairness check and the final choice.
6. Compute the plain-loss choice and the two diagnostic variants.
7. Generate the judge disturbances and score all four candidates on the judge records.
8. Save every per-candidate value, every arm's choice, and the hashes of the predictions.

**Order of work**

1. Save and hash this specification in the run directory.
2. Pass the gates in section 9, including the reproduction gate on the original split.
3. Run repeats 0 and 1 and report the time per repeat. They count as the first two repeats.
4. Run the remaining repeats without inspecting results in between.

If the projected total time exceeds 12 hours, stop after repeat 1 and report. Do not reduce J, the realizations or the bootstrap draws without a new version of this document.

## 7. Estimands and intervals

The primary quantity is the average, over repeats, of the judge's primary endpoint for Selector 2's choice minus the same endpoint for SAFE's choice.

### 7.1 Primary estimand

- In repeat r, the difference is the judge's mean stressed Brier of the model Selector 2 chose minus that of the model SAFE chose.
- A repeat in which both chose the same model contributes exactly zero. Keep these repeats.
- The estimate is the mean of the 50 differences. Negative favours Selector 2.

### 7.2 Interval across repeats

Repeats share records, so an ordinary standard error would be too small. Use the corrected resampled t interval of Nadeau and Bengio (2003).

```latex
\hat{\theta}\;\pm\;t_{J-1,\,0.975}\;\sqrt{\left(\frac{1}{J}+\rho\right)s^{2}},\qquad \rho=\frac{n_{\text{judge}}}{n_{\text{fitting}}+n_{\text{selection}}}
```

- s-squared is the sample variance (ddof = 1) of the 50 per-repeat differences.
- rho uses the mean realized role sizes; it is about 0.25.
- If s-squared is zero, report the estimate with no interval and say why.
- This interval is approximate. Its coverage is not verified in this run; say so in the report.

### 7.3 What the interval covers

It describes variation from which development records are used to fit, to select and to judge, with the four configurations, the disturbance mechanisms and the data source held fixed.

It does not cover other datasets or time periods, other candidate pools, or the fact that Selector 2 was designed after Taiwan results were seen.

### 7.4 Other contrasts, same construction

- Each other arm minus SAFE, on the primary endpoint.
- Selector 2 minus plain loss, on the primary endpoint.
- Selector 2 minus SAFE on clean Brier, clean log loss and mean stressed log loss.
- Selector 2 minus SAFE on the primary endpoint for women and for men separately, and on deterioration for each sex.

Only the contrast in 7.1 is primary. The others are descriptive and are labelled so. For the two sex contrasts on the primary endpoint, also give Bonferroni-adjusted intervals (t quantile at 0.9875).

### 7.5 Descriptive summaries without intervals

- How often each arm chooses each candidate, out of 50.
- How often Selector 2 and SAFE choose different models, and the mean and SD of the difference among those repeats, labelled conditional.
- How often Selector 2's choice equals the plain-loss choice.
- How often the fairness check blocks a candidate, and which.
- The distribution over repeats of A-bar, E-bar, SAFE's robustness mean and S-bar for each candidate.

### 7.6 Within-repeat intervals

The bootstrap intervals of sections 3.4 and 3.6 are conditional on one repeat's fitted models. Save them per repeat. Do not average or pool them into the across-repeat interval.

## 8. Reading the results

The claim is about direction only: Selector 2's choice is judged better than SAFE's when the primary interval lies entirely below zero.

**Principal results table (place first in the report)**

| Contrast on the primary endpoint | Mean difference | 95% corrected interval | Repeats with different choices |
| --- | --- | --- | --- |
| Selector 2 minus SAFE (primary) | estimate | interval | count of 50 |
| Plain loss minus SAFE | estimate | interval | count of 50 |
| Order, mild minus SAFE | estimate | interval | count of 50 |
| Accuracy, full minus SAFE | estimate | interval | count of 50 |
| Selector 2 minus plain loss | estimate | interval | count of 50 |
| Selector 2 before the check minus SAFE | estimate | interval | count of 50 |

**Primary reading**

| Result for Selector 2 minus SAFE | Reading |
| --- | --- |
| Interval entirely below zero | Development support for the claim on Taiwan. Proceed to fresh datasets. |
| Interval includes zero | Unresolved on Taiwan. Report the estimate and how often the choices differed. |
| Interval entirely above zero | SAFE's choice is judged better here. Stop and report. |
| The two selectors choose the same model in nearly every repeat | The comparison has little to say. Report this as the finding. |

The report gives the size of the difference in raw Brier units beside each candidate's clean Brier. It does not call a difference large, small or material. No practical cutoff exists.

**Selector 2 and plain loss**

If the two choose the same model in most repeats, say so in the summary, in plain words. That outcome is acceptable under the agreed success criterion and must not be hidden.

**Order versus accuracy, and range**

For each repeat, compare the models chosen by four versions of the score before applying the fairness check: published SAFE (order, full range), order over the mild range, accuracy over the full range, and accuracy over the mild range (Selector 2 before the check). Record whether changing the measure alone, changing the range alone, or changing both leads to a different choice from published SAFE. Report how often each pattern occurs across the 50 repeats, counting all eight combinations of the three comparisons, including the one in which no version departs from SAFE. If both single changes lead away from SAFE, state whether they select the same model.

Report separately how often the fairness check changes Selector 2's top-ranked choice and which model is selected instead. Use Selector 2's **final choice after the fairness check** for the primary comparison against published SAFE.

A changed choice is not by itself a better choice. The judge's contrast against SAFE for each of the three versions is in the principal results table.

**Fairness**

- Report Selector 2 minus SAFE on the judge's primary endpoint for women and for men, with ordinary and adjusted intervals.
- A sex contrast whose adjusted interval lies above zero means the switch is judged worse for that group. Report it prominently, even if the overall result favours Selector 2.
- An interval that includes zero is unresolved. It is not evidence that the groups are treated equally.
- Report how often the fairness check changed Selector 2's choice. If it never did, say that the check did not bind on these data.

**What this run cannot show**

It cannot confirm the method, show that it generalizes, or show that SAFE's choice is wrong. Taiwan is the worked example.

## 9. Checks, deliverables and stop conditions

No result counts until every gate passes, and any failed gate stops the run.

**Gates before the repeats**

- [ ] This specification saved and hashed in the run directory.
- [ ] Hash inventory of all pre-existing project files taken.
- [ ] Hashes of every reused module recorded (section 2).
- [ ] **Reproduction gate.** With the pilot's original fitting and selection assignment, the new pipeline refits the four candidates and reproduces the saved clean selection predictions, the saved SAFE component means and the saved integrated scores (`safe2026_swapping_2026-10-26/integrated_scores.json`, `component_areas.csv`) to 1e-12.
- [ ] On that same original split, the stressed Brier underlying S at the 23 SAFE nodes equals the value saved by Step 0 (`step0_2026-11-14/brier/orderings.csv`, column O4) to 1e-12.
- [ ] No historical evaluation ID appears in any role of any repeat.
- [ ] In every repeat, the three roles partition the development records and no predictor group crosses roles.

**Focused tests on synthetic fixtures**

- S for a perfect forecast, a default-rate forecast and a worse-than-default-rate forecast.
- Score 2 and its ranking, including a tie.
- The choice rule when the top candidate is blocked, when SAFE's choice is on top, and when every challenger is blocked.
- The fairness check with a group that contains both sexes and with a zero-variance coordinate.
- The corrected interval against a hand calculation, and the zero-variance case.
- A repeat in which two arms choose the same model contributes exactly zero.

**Deliverables**

- The principal results table of section 8, then the fairness table.
- One per-repeat table: every candidate's selection-side values (A-bar, E-bar, SAFE robustness mean, SAFE score, S-bar, Score 2, clean Brier), every arm's choice, and every candidate's judge scores overall and by sex.
- Selection frequencies for every arm, and the summaries of section 7.5.
- The fairness-check record for every repeat.
- Split assignments, seeds, role sizes and default counts by sex.
- A short report that states the development status, what was already known (section 1), every deviation, and the time per repeat.
- Source snapshots, test log, verification receipt, and the before-and-after hash comparison.

**Stop and report, without workaround, if**

- Any gate fails.
- A reused function cannot be used unchanged for a step in this document.
- Any step would need a historical evaluation record, tuning, or an edit to an existing file.
- A candidate fails to fit or returns invalid probabilities in any repeat.
- An undefined value appears where a number is expected.

**After the run**

Root reruns the tests and reconstructs the principal table from the per-repeat table. Describe a same-session recheck as a recheck, not as independent review.

## 10. Fresh datasets

Confirmation comes from applying this specification, unchanged, to public credit datasets the project has never modelled. This section is for Astra's dataset screen; it is not part of Sol's Taiwan run.

**Inclusion criteria, fixed before screening**

| Criterion | Requirement |
| --- | --- |
| Access | Public, with a licence that permits academic use and publication of results |
| Outcome | Observed repayment default per borrower or loan. A lender's approval decision does not qualify. |
| Sex | Recorded for each record, and usable without being a model input |
| Size | At least 20,000 records, 2,000 defaults, and 500 defaults for each sex. These thresholds are proposed, not derived. |
| Predictors | Tabular, usable by the four configurations after documented cleaning that does not look at the outcome |
| Independence | Not previously modelled in this project |
| Preferred, not required | Dated records that allow a later period as the judge; a different lender or country from Taiwan |

**What the screen records for each dataset**

Source and licence, number of records, defaults overall and by sex, predictor list and types, time span, known quality issues, and prior use in the SAFE or credit-fairness literature. The screen fits no model and computes no outcome-based statistic beyond these counts.

**Before any model touches a selected dataset**

A short dataset addendum is written and hashed. It fixes the predictor list, the sex coding, the grouping of duplicate predictor rows, the split, and the judge's disturbance families for that dataset. Sections 3 to 8 apply unchanged.

**The confirmatory claim**

Per dataset, the primary contrast of section 7.1 with its interval. Across datasets, the count of datasets in which the interval lies below zero, includes zero, or lies above zero. Results from every screened-in dataset are reported, whichever way they come out.

## 11. Open items for Andrei

Five design choices were made while drafting. The run uses the defaults below unless Andrei changes them before it starts.

| Item | Default in this specification | Alternative |
| --- | --- | --- |
| Records used on Taiwan | The 24,004 development records only. The 5,996 historical evaluation records stay untouched. | Resplit all 30,000 records. More data per repeat, but it ends the separate status of the historical evaluation set for good. |
| Mild range for Selector 2 | Tail fractions 0.01 to 0.05 in steps of 0.01, plus the clean node | Another range, justified from the perturbation design before the run |
| Primary endpoint of the judge | Mean stressed Brier over the nine q = .10 feed-fault scenarios | Clean Brier, or the worst scenario |
| Fairness check | A challenger is blocked when its stressed Brier on selection records is resolvedly worse than SAFE's choice for women or for men | A different rule, for example one based on the gap between the sexes |
| Number of repeats | 50 | A larger number, at proportionally longer run time |

Two points to keep in view:

- The equal weighting of the three axes on their raw scales is inherited from SAFE. S sits on the Brier R-squared scale, which is narrower than SAFE's order measure, so the report shows each axis separately for every candidate.
- The diagnostic variants may show that narrowing the severity range matters as much as switching from order to accuracy. If so, the article's question becomes order versus accuracy and where on the severity range each is measured.
