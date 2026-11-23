# Synthetic study for the first SAFE article

Implementation specification, 21 November 2026. This incorporates the independent referee's defaults for the lean design. It supersedes the wider November 16 draft for this simulation. Production counts were fixed on 23 November and are recorded in the [public registration](registration.md).

## Questions

Taiwan is the worked example. Brazil is the sole fresh real-data comparison of final Selector 2 against published SAFE. In the simulation we know the original default probability of each borrower. Use that to examine two mechanisms in isolation.

| Setting | Comparison | Evidence to report |
| --- | --- | --- |
| R, a signed interaction in the outer 5% of each tail | Direct small-swap Brier minus clean Brier, using the models those rules actually choose | Differences in mean squared error against original risk on clean judge records and at J1 and J3. Include all repeats, exact zeros on common choices, disagreements, and same-repeat clean-to-stressed reversals. Show SAFE and final Selector 2 alongside. |
| F, a group-specific relationship between income, debt and risk | Final Selector 2 versus its first proposed choice, with the SAFE incumbent and the direct rule's choice identified | Which proposals harmed women or men on the selection faults, which the gate vetoed, and whether the final choice avoided the harm. Also report final minus first proposed error under the independent J1 faults overall and by sex. |
| Equal-error control | Force the same two forecasts through the actual joint two-sex check | False vetoes out of 1,000 independent trials, with a 95% Wilson interval. |

Lower error is better. Use an observed binary outcome for fitting, selection and the gate. Use original probabilities for truth scoring, even when the judge changes inputs. Keep fitted-model results separate from the calculations below. Use the existing four candidate configurations and frozen selector rules. In particular section 6.5 uses the equal-weight average across clean and 1–5% pooled tail swaps, with its frozen tie rule.

## Predictions fixed before fitting

R has an interaction which disappears in the centre of x1 and changes sign between its low and high tails. For two ideal forecasts differing by a constant probability signal of magnitude a only in the tails, the clean cost of ignoring the signal is V = 0.10a². On a mirrored swap of q per tail, learning it costs 8 min(q, .05)a² when the background is held fixed. The crossover is at q = .0125. Across the six selection nodes the learner averages 0.20a² versus 0.10a² for omission.

Under the independent judge the same idealisation predicts the reverse ordering. Among the nine scenarios, two mask the interaction, four shift it and three leave it intact on affected rows. Masking removes its sign. With independent normal inputs and signed numeric shifts, the ideal signal learner costs about 0.048V at J1 and 0.677V at J3 across the whole cohort. The ideal forecast omitting it costs V at each level. These are fixed-background probability calculations. Actual logistic probabilities, shifting the background predictors, fitted model errors and empirical ranks can change the result. The implementation includes a separate six-node truth check of the declared logistic generator.

A fitted switch to a worse J1 model counts against transfer at J1. A J3-only gain is evidence at that severity only. Agreement means zero realised choice benefit for that repeat. Report full Selector 2 minus clean Brier separately in R. Attribute a loss to the full method only when its own choices and paired comparison support that conclusion. The F generator specifies one moderate group difference. It may yield few harmful attempted switches. A small or empty denominator limits what the experiment can establish.

## Borrowers, selections and endpoints

Use 30,000 fresh borrowers in each independent repeat with a 64/16/20% fitting/selection/judge split, grouping identical original predictor rows. Keep sex out of model inputs and the true probabilities out of fitting, model selection and vetoes. R uses 15 independent normal predictors. F uses the 15 credit-like fields with the single female debt effect fixed at 0.5. Use the exact definitions, calibration targets and seeds in `generators_spec.md` and the generator implementation. Keep the same fitted models throughout each repeat.

At each judge level, average across all borrowers, 30 realisations within each of the nine family-by-fault scenarios and the nine scenarios equally. R uses J1 (10% of records, numeric shifts of 0.10 fitting IQR) and J3 (50%, 0.50 IQR). F uses J1. Use the same records and fault draws across candidate models at each level and the frozen pairing across severity levels. For each difference use a pointwise paired 95% t interval across independent populations, using the actual repeat count and zero overlap correction. For all-zero differences report [0,0].

For F a challenger is harmful if its mean error against the original risk exceeds the incumbent's for either sex on the same selection borrowers and five nonzero pooled swaps. Record the sign and the separate threshold of at least 0.0005. Apply the existing group veto based on observed outcomes to proposals in rank order until an acceptable model or SAFE is reached. Record both the triggering group and truth harm. Count harmful first proposals blocked, harmful first proposals allowed and harmless first proposals blocked, each with its own applicable denominator and Wilson interval. Count starts at SAFE over all repeats, and show later visited proposals separately. Report prevention only when a harmful first proposal was blocked and the model ultimately selected is harmless for both groups relative to SAFE. An empty denominator is an unavailable rate. The direct rule can receive credit or blame only for the candidate it actually chose.

In the control, fix the group counts at 2,880 women and 1,920 men. The 4,800 singleton borrowers each have true risk .22, incumbent forecast .17 and challenger forecast .27. Both forecasts have truth loss .0025 per person. Label noise still produces paired observed loss differences of .044 minus .20 times the observed default. Use 2,000 whole-borrower bootstrap draws per trial and the frozen joint gate. Since there are four sex-by-outcome cells and the forecasts are constant, sampling their counts from the empirical multinomial gives the same bootstrap distribution as resampling individual borrowers. Use a separate purpose-specific control seed for the bootstrap. The resulting rate describes this specific no-harm case.

## Timing and production

Use independent seeds for calibration, control, timing and production. Use master 20261117 for generated data and purpose-specific streams. Keep split seed 20261114+r, bootstrap 42+r and candidate seed 42. Use a purpose-, setting- and repeat-specific judge dataset ID. Keep the timing populations separate from production.

The blinded benchmark used one three-worker batch of three discarded repeats per setting, with two numerical threads per worker. Its original timing rule selected the ceiling of 50 per setting. Andrei subsequently fixed **150 production repeats for R and 300 for F** before viewing fitted production results. F uses the separate 1,000-trial equal-error control. The projected total including timing and the 20% reserve is about five hours. Andrei launches the study after the public registration and Brazil production.

See the [version 2 amendments](version2_amendments.md), the [direct rule addendum](section_6_5.md), and the [generator specification](generators_spec.md).
