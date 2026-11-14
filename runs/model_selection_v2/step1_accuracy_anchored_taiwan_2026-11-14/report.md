# Taiwan Step 1: accuracy-anchored SAFE

| Primary endpoint contrast | Mean Brier difference | 95% corrected interval | Different choices /50 |
| --- | ---: | --- | ---: |
| selector2_minus_safe (primary) | -0.00158751 | [-0.00252422, -0.00065080] | 50 |
| plain_loss_minus_safe (descriptive) | -0.00128517 | [-0.00207441, -0.00049592] | 49 |
| order_mild_minus_safe (descriptive) | -0.00158976 | [-0.00252464, -0.00065488] | 50 |
| accuracy_full_minus_safe (descriptive) | -0.00158976 | [-0.00252464, -0.00065488] | 50 |
| selector2_minus_plain_loss (descriptive) | -0.00030234 | [-0.00099466, 0.00038998] | 30 |
| selector2_before_check_minus_safe (descriptive) | -0.00158751 | [-0.00252422, -0.00065080] | 50 |

| Sex (descriptive) | Endpoint difference | 95% interval | Bonferroni interval |
| --- | ---: | --- | --- |
| female | -0.00181371 | [-0.00277484, -0.00085258] | [-0.00291966, -0.00070776] |
| male | -0.00123727 | [-0.00269564, 0.00022110] | [-0.00291538, 0.00044085] |

Development support for the directional claim on Taiwan.

This is a development worked example, not confirmation. Selector 2 was designed after Taiwan results were seen. Before this run, SAFE selected LR on the original split; RF/GBM had lower inspected probability losses; Step 0 showed RF retaining accuracy better than LR across swaps and preserving order best at mild severities. The original-split result was largely predictable; repeated whole-procedure behavior supplies the new information.

Selector 2 and plain loss choose the same model in 20/50 repeats. Selector 2 and SAFE differ in 50/50. Same-choice repeats remain exact zero differences.

The fairness check changes the top-ranked choice in 0/50 repeats. The check did not bind on these data.

The four-score comparison below is entirely before fairness. Measure alone = accuracy/full; range alone = order/mild; both = accuracy/mild. The primary judge comparison uses the final post-check Selector 2 choice.

| Measure alone changes | Range alone changes | Both changes | Repeats |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 |
| 0 | 0 | 1 | 0 |
| 0 | 1 | 0 | 0 |
| 0 | 1 | 1 | 0 |
| 1 | 0 | 0 | 0 |
| 1 | 0 | 1 | 0 |
| 1 | 1 | 0 | 0 |
| 1 | 1 | 1 | 50 |

When both single changes depart from SAFE, they choose the same model in 50 repeats and different models in 0.

| Candidate | Mean selection clean Brier | Mean judge clean Brier | Mean judge stressed Brier |
| --- | ---: | ---: | ---: |
| logistic | 0.13631951 | 0.13657794 | 0.13808096 |
| decision_tree | 0.13887975 | 0.13922193 | 0.14178814 |
| random_forest | 0.13476609 | 0.13486970 | 0.13649120 |
| gradient_boosting | 0.13457128 | 0.13453253 | 0.13674832 |

Brier units are raw. No practical cutoff was set; no materiality claim follows. A changed choice alone is not evidence of improvement. Every diagnostic version has its judge contrast in the principal table. Axis distributions are saved separately because equal nominal weights do not imply equal empirical influence.

The corrected interval uses J=50, sample variance ddof=1, t49 and rho=0.2500130188 from mean realized role sizes. It is approximate and its achieved coverage is unverified. It describes repeat split/fit/select/judge variation on these development records, conditional on the fixed configurations, fault mechanisms and data source. It does not cover other datasets, times, candidate pools, or post-Taiwan method design. A zero-variance contrast has no reported interval. Sex results are descriptive; adjusted intervals use t49 at .9875. Zero inclusion is unresolved, not evidence of equal treatment.

Fairness-check and within-repeat percentile intervals use 2,000 whole-original-predictor-group draws, seed 42+r, and hold fitted models and disturbed predictions fixed. They are saved per repeat and are never pooled into the corrected interval. Step 0 zero-variation coordinates (SD <=1e-12) are explicitly excluded from joint intervals and cannot establish a resolved block. Judge realization losses are averaged after computing each loss; their Monte Carlo spread is separate from sampling uncertainty.

Repeats 0 and 1 took 157.10s and 160.81s. Total repeat execution: 2.189 hours. All individual stage times are in timings.csv. No comparative outcomes were inspected between repeats 2 and 49. No protocol deviations. Only Taiwan development records were modelled; historical IDs served exclusion checks only. No tuning, calibration, threshold change, old-file edit, commit, push, publication, external contact or fresh-dataset screen.

Reconstruction handoff: per_repeat_table.csv contains every candidate, all six arm choices, all selection values, judge scores by sex, and realized role sizes. principal_table.csv, fairness_table.csv and per_repeat_primary_differences.csv give the derived tables. choices_before_judge.json freezes every repeat choice before judge computation. Split assignments, models, all probabilities, bootstrap plans/draws, greedy alternatives, scenario registries and loss arrays are under repeats/. The attachment, hashes, source snapshots, test logs, reconstruction receipt and preservation comparison accompany the run. Verification is a same-session recheck, not independent review. Existing startup documents remain untouched under the protocol; this dated report is the accepted/executed Step 1 record. Fresh-dataset confirmation belongs to Astra.
