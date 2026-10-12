# The score and the superiority comparison: a worked example

Date: 9 October 2026. All numbers below are constructed illustrations. They are not Taiwan results, SAFE paper outputs, expected effects, lending costs or adopted policy tolerances. No credit model was run to generate them.

## 1. Why stability is not enough

Suppose a training-prevalence predictor has evaluation Brier loss 0.16. That value is possible when both training/evaluation prevalence are .20: .20*(1-.20)=.16. It makes the same prediction for every record and ignores the inputs. Its predictions are perfectly unchanged by input corruption, but they have no individualized predictive information.

Now suppose three candidate models have these probability losses:

| Candidate | Clean Brier | Brier under declared fault | Worst loss R | Skill score 100*(1-R/.16) |
| --- | ---: | ---: | ---: | ---: |
| A: small, fragile model | .110 | .240 | .240 | -50.00 |
| B: slightly larger model with backup information | .115 | .130 | .130 | 18.75 |
| C: constant prevalence reference | .160 | .160 | .160 | 0.00 |

A is best on clean probability loss. C changes nothing when information fails. B has the best worst-scenario probability quality. The proposed score summarizes that comparison without adding accuracy, explanation reliance and robustness numbers of unrelated meanings.

The clean trade-off between B and A is .005 Brier. Whether that sacrifice is acceptable is a separate policy decision. If a previously agreed clean margin were .010, B would satisfy the point-estimate condition. If it were .002, B would fail it. We cannot select .010 after seeing this table merely to make B acceptable.

The score's 18.75 means 18.75% lower worst-scenario Brier loss than the reference. It does not mean 18.75% trustworthy, 18.75% fewer defaults, or 18.75% higher bank profit. Negative values are important information and must remain visible.

## 2. The score does not establish superiority of the methodology

The selection table belongs to validation data. Now freeze the model chosen by each method and evaluate its actual predictions on separate audit records and new fault draws.

Suppose a clean-only or KAIRI-style selector chose A while our selector chose B, and the audit losses happened to equal the illustrative values above. Then:

G = R(A)-R(B) = .240-.130 = .110 Brier units.
Relative loss reduction = .110/.240 = 45.83%.

This is an improvement in measured probability loss of the selected model. It is not a difference between our internal score and the numerically unrelated SAFE score. Never subtract two heterogeneous framework scores and label the difference 'superiority'.

If SAFE and our method both chose B, G would be zero: the scoring displays might differ, but there would be no selection advantage in this example. If the simple minimax baseline also chose B, the gain over A would support fault-aware assessment; it would not establish extra value from our safeguards or uncertainty machinery.

A valid comparison would also need an uncertainty interval, a meaningful-gain threshold agreed before results, acceptable clean/error-rate trade-offs, and evidence beyond one researcher-created table. The 45.83% is not a forecast for the project.

## 3. The maximum operation must be specified correctly

Consider another illustration with three conditions:

Comparator losses: [.12,.24,.18].
Proposed losses: [.125,.13,.15].

Difference of worst losses: max(comparator)-max(proposed)=.24-.15=.09.
Worst of paired condition differences: max(comparator-proposed)=max([-.005,.11,.03])=.11.

These are different questions. Our primary G uses the first. The second is useful supporting information about the largest condition-specific improvement, but cannot replace the primary estimand after results.

## 4. A conservative uncertainty example

Suppose simultaneous component intervals on the audit data were:

| Candidate | Clean loss interval | Fault loss interval | Derived interval for R |
| --- | --- | --- | --- |
| A | [.105,.115] | [.230,.250] | [.230,.250] |
| B | [.110,.120] | [.120,.140] | [.120,.140] |

The implied conservative interval for G is [.230-.140, .250-.120]=[.090,.130]. With an independently declared material-gain threshold .020, this illustration supports a material gain. The reference-loss uncertainty would also enter intervals for the skill score; treating its .16 as exact here is only for explanation.

These component intervals must cover the declared family jointly. Several unrelated 95% intervals are not automatically a joint 95% statement. Deriving the max interval from component bounds avoids assuming there is one uniquely worst scenario.

## 5. 'Inconclusive' is not a free winning outcome

Suppose our audit refuses to accept any candidate because every interval crosses a tolerance. That may correctly recognize uncertainty, but it is not automatically better model selection. We must report how often a method accepts, rejects or remains inconclusive, and compare reliability at useful, matched acceptance coverage. Assigning zero loss to an inconclusive decision would make a reject-all procedure win artificially.

## 6. Why a familiar score can still be useful research

The Brier loss, reference skill transformation, minimax objective, constraints and uncertainty principles are established ideas. The proposed research value would be their justified application to the SAFE/KAIRI selection task, a demonstrated mechanism, and a measurable improvement in actual assessment decisions. If a simple baseline explains all the advantage, that should shape the paper honestly.

Read [the main plan](safe_comparison_plan_2026-10-09.md) for the proposed research direction and [the technical protocol](safe_comparison_technical_protocol_2026-10-09.md) for implementation details.