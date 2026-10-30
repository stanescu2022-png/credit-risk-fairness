# Synthetic SAFE mechanism audit — 30 October 2026

Executed result: positive affine output invariance is supported by the algebra and all fixed synthetic checks for our Taiwan operationalization. Each of the three integrated means is unchanged; proper probability losses change. This is a mechanism diagnostic, not an empirical model comparison or a proposed addition to the four candidates.

The independent oracle uses exact rational, absolute-reference-gap-weighted pairwise concordance. Positive affine maps scale every gap equally and preserve comparison order/ties. Accuracy's binary reference stays unchanged. Every greedy alternative, path/tie set, curve node and Cartesian aggregate was checked independently. Tie-order enumeration also checked the full additive accuracy curve.

| Fixture | Output transform | Arithmetic | Geometric | Quadratic | Brier | LogLoss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| strict | identity | 0.6552579365 | 0.4715693250 | 0.7224929247 | 0.2382812500 | 0.6721431373 |
| strict | compression | 0.6552579365 | 0.4715693250 | 0.7224929247 | 0.2236328125 | 0.6379754498 |
| strict | shift_compression | 0.6552579365 | 0.4715693250 | 0.7224929247 | 0.2258300781 | 0.6407767369 |
| strict | near_constant_nonzero | 0.6552579365 | 0.4715693250 | 0.7224929247 | 0.2500000000 | 0.6931471806 |
| strict | square_nonlinear_control | 0.6604662698 | 0.4749859552 | 0.7279434227 | 0.2716217041 | 0.8469433386 |
| ties | identity | 0.6387195122 | 0.4489462148 | 0.7087962154 | 0.2548828125 | 0.7644175032 |
| ties | compression | 0.6387195122 | 0.4489462148 | 0.7087962154 | 0.2316894531 | 0.6586263319 |
| ties | shift_compression | 0.6387195122 | 0.4489462148 | 0.7087962154 | 0.2380981445 | 0.6833234430 |
| ties | near_constant_nonzero | 0.6387195122 | 0.4489462148 | 0.7087962154 | 0.2500000000 | 0.6931471806 |
| ties | square_nonlinear_control | 0.6444892473 | 0.4518794356 | 0.7160952261 | 0.2965126038 | 0.9972576366 |

Near-constant affine scale is 2^-40 with offset 1/2; all clean probability levels remain distinct where originally distinct. Exact constant scale zero is undefined for K with the probability as reference. A positive 2^-60 scale at the same offset collapses all levels in float64 and is also undefined. Neither result is silently assigned a zero score. Constant comparison outputs remain defined at K=.5 when the clean reference is nonconstant.

Squaring outputs preserves accuracy order/ties but changes probability-reference pair weights and the robustness curve in both fixtures. It refutes a general nonlinear monotone invariance claim. These fixtures also show affine changes to Brier and LogLoss; they do not assert that every affine map changes every loss.

## Operational proof versus article fidelity

The proof applies to the centered cumulative-area / weighted-pairwise rank kernel, average exact-score ties, additive accuracy contributions, fixed-model greedy masking and raw Cartesian means currently specified in safe2026_specification.md. Its documented equation correspondence is Eqs. (4), (8), (10), (12)–(17). This audit does not newly inspect the printed article, establish the authors' software used that kernel, resolve alternate slice normalizations or prove TOPSIS invariance. Publication wording about published SAFE requires checking the printed equations and reporting all operational adaptations. The later author-credited package has documented kernel/normalization differences and is outside this proof.

## Separate marginals and joint loss

A second, finite response-table counterexample gives candidates A/B identical clean, stress-only and mask-only predictions record by record. Therefore every deterministic function given only those identical vectors (and the same outcomes/metadata) must tie. Their joint stress+mask outputs are different: candidate A has joint Brier .0625 and LogLoss -log(.75), while candidate B has joint Brier .5625 and LogLoss -log(.25). The separate tables therefore do not identify joint loss. This is an information insufficiency counterexample, not an end-to-end SAFE experiment, evidence of such interactions on Taiwan, or a claim that no method could use joint inputs. The lookup responses can be represented by finite functions of record identity and two binary intervention flags; no fitted estimator or empirical data is invoked.

Improvement remains a hypothesis. Both the invariance result and the nonlinear counterexample are preserved. These diagnostics belong to hypothesis generation; they provide no independent superiority evidence. Any later extension must avoid optimization on the historical holdout, and the independent judge must not be revised in response to a favored selector. Exploratory experiments, protocol freezing and independent validation remain distinct; a superiority claim requires evidence not used to develop that claim.

## Later judge interface recommendation (inspection only)

Keep pure assessment functions separate from identity-checked data and model access. Pass aligned `(ID, outcome, SEX, original_22_predictor_group, probability)` tables to loss/fairness functions, and paired clean/stressed tables to stress summaries. Carry the saved whole-original-predictor group ID; never reconstruct groups from a reduced candidate input set. Define undefined denominators explicitly.

pilot.load_data validates partitions but reads the full processed/raw data and holdout identity metadata. safe2026_run.load_contract is a cleaner train-only precedent: verify candidate receipts/input hashes before joblib loading, filter source rows by membership before parsing predictors, restore saved feature/ID order, and validate exact saved probabilities. Neither loader was executed here. A later explicit assessment-only loader should expose only the authorized role, frozen models and aligned metadata, refuse fitting/tuning, and run only after selector choices and the judge protocol are fixed. Identity/hash verification must stay separate from a method that creates empirical predictions. No new holdout access, prediction or metric occurred in this audit.

protocol.json records source identities and focused-test results; audit.json contains every synthetic output and independent score. VERIFIED.json binds all run files. Existing destinations are refused. No existing scientific files were edited.
