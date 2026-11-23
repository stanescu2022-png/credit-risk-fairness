# Version 2 amendments: Step 1 specification and dataset addendum template

2026-11-17 · @Andrei · Revised 21 and 23 November 2026 before the public timestamp (see the change log, section 7.1)

## 1. What stays frozen, what is added, and why

Version 2 adds secondary endpoints, diagnostic arms, reporting rules and records. It changes nothing that decides the primary verdict. It answers the referee report of 16 November (https://claude.ai/artifact/HUJKtZGwDZ2YETARMb9K14) and is agreed before any Brazil model is fitted.

**Unchanged from version 1.**

- Selector 2: its mild grid, Score 2, the fairness check and the choice rule (Step 1 section 3).
- The four candidate configurations (Step 1 section 2).
- The judge's primary endpoint: mean stressed Brier over nine scenarios at q = .10 (Step 1 section 5).
- The 50-repeat design, the primary estimand, the corrected interval and the verdict rules (Step 1 sections 6 to 8; template section 9 as amended on 16 November).
- The dataset rules of the template (sections 2 to 7), except the permission rule amended in section 6 below.

**Added.**

- Three registered diagnostic arms, and paired contrasts that separate the range of disturbance from the robustness measure (section 2).
- Secondary judge endpoints, including a severity ladder and SAFE's own rank measure (section 3).
- The fairness check's operating characteristics, and a group audit (section 4).
- Reporting rules for low disagreement and mixed results, and the wording allowed with one confirmation dataset (section 5).
- A permission amendment, a record of deviations and several clarifications (section 6).
- A split execution plan, bounded parallelism and separate job budgets (section 6.4), accepted on 17 November before any Brazil fit.
- A secondary comparison arm, direct small-swap Brier (section 6.5), accepted on 18 November before any Brazil fit.
- A change log and a public timestamp before the run (section 7).

**Why.** The referee's three main objections: the method was not separated from simpler alternatives; the judge mostly measures clean Brier and cannot reward what SAFE's order axis is for; and one confirmation dataset needs reporting rules fixed in advance. Every addition is labelled secondary, diagnostic or audit, and is reported whatever it shows.

**Scope.** Applies to the Brazil run and the synthetic study. Taiwan is not rerun; its values are quoted as development results. For the synthetic study, the lean specification of 21 November (`research/model_selection_v2/synthetic_lean/protocol.md` and `generators_spec.md`) names the settings, arms and endpoints actually run. Where it differs from sections 2 to 4 for the synthetic study, the lean specification governs; this is recorded in section 7.1.

## 2. Registered diagnostic arms and claim attribution

Three score-only arms, first computed post hoc on Taiwan on 16 November, are now registered for Brazil and the synthetic study. They use the same fitted candidates and the same score form as Step 1 section 4, with no fairness check, like the existing diagnostic variants.

| Arm | Score | Question it answers |
| --- | --- | --- |
| A/E only | (A-bar + E-bar) / 2 | Does removing SAFE's robustness axis alone change the choice? |
| Clean accuracy | (A-bar + E-bar + S_0) / 3 | Is Selector 2 just clean accuracy? |
| True retention | (A-bar + E-bar + mean over the five stressed mild nodes of (S_k − S_0)) / 3 | Does deterioration under mild disturbance drive the choice? |

With the existing arms (published SAFE, Selector 2, Selector 2 before the check, plain loss, order over the mild range, accuracy over the full grid) there are nine arms in all.

**Contrasts, with the corrected interval of Step 1 section 7.2.** Each arm minus SAFE on the judge's primary endpoint, and for every pair of arms the number of repeats in which they choose the same model. Match counts are descriptive only: two arms can agree in most repeats while the whole difference between them comes from the few repeats where they disagree.

**Taiwan values, for reference only (development data, post hoc):** A/E only −0.00042 (−0.00126 to +0.00043); clean accuracy −0.00115 (−0.00203 to −0.00026); true retention −0.00159 (−0.00252 to −0.00065), matching Selector 2 in 49 of 50 repeats and order over the mild range in 50 of 50.

**Range against measure: the paired contrasts that answer the article's question.** Each contrast changes one thing and holds the other fixed. All use the judge's primary endpoint and the corrected interval, with the number of repeats in which the two arms choose differently.

| Contrast | Holds fixed | Changes |
| --- | --- | --- |
| Order over the mild range minus published SAFE | Measure: order | Range: full to mild |
| Brier skill over the mild range (Selector 2 before the check) minus Brier skill over the full grid (accuracy, full) | Measure: Brier skill | Range: full to mild |
| Brier skill over the mild range minus order over the mild range | Range: mild | Measure: order to Brier skill |
| Accuracy, full minus published SAFE | Range: full | Measure: order to Brier skill |
| True retention minus clean accuracy | Third axis is accuracy-based | Level to deterioration |

**Reading.** If both range contrasts lie below zero and the two mild-range arms mostly choose the same model, the article says the pattern supports a range explanation: measuring robustness over a mild range, rather than the full grid, improved the judged choice whatever the measure. The measure contrasts are reported beside them with the same weight. No mechanism is assigned from match counts alone.

## 3. Judge secondary endpoints

The primary endpoint stays as frozen. Four secondary endpoints are computed for every candidate in every repeat, each reported for every arm minus SAFE with the corrected interval.

**Why they are needed.** At the frozen severity, 90% of judge records are undisturbed, numeric fields move by only a tenth of their interquartile range, and the endpoint scores Brier, the same loss Selector 2 uses. A model that is accurate but fragile under heavy faults can therefore win on the primary endpoint. The analytical argument of 16 November (theory note, setting R, a conjecture not yet simulated) suggests this can happen, and that raising only the share of disturbed rows does not expose it: the size of each disturbance must grow too.

**1. Brier on disturbed rows only.** As the primary endpoint, but each realization's Brier is computed on the selected records alone.

**2. Severity ladder.** The nine scenarios are rerun at two stronger levels, with the same families, modes and seeds.

| Level | Share of judge records disturbed | Numeric shift | Ordered fields | Unordered fields and masking |
| --- | --- | --- | --- | --- |
| J1 (frozen) | 10% | ±0.10 × fitting IQR | one level | as frozen |
| J2 | 25% | ±0.25 × fitting IQR | one level | as frozen, on more records |
| J3 | 50% | ±0.50 × fitting IQR | two levels, only if the destination occurs in fitting | as frozen, on more records |

Records are nested: the first 10% of the permutation are inside the first 25%, which are inside the first 50%. Signs and category draws are shared across levels. Rounding, the integer minimum shift and clipping to the fitting range follow template section 7.4 at every level. For each level, report the primary endpoint and Brier on disturbed rows.

**3. Rank measures on judge records.** Stressed AUC, and SAFE's own order measure K (rank graduation of clean against stressed predictions, as in `rank_graduation`), averaged over realizations and scenarios as the primary endpoint is. These let SAFE's choice win on SAFE's own terms.

**4. Overall deterioration.** Primary endpoint minus clean Brier, for all records. Step 1 defined it by sex only.

**Reading rule.** The article says Selector 2's choice was judged better only if the primary interval lies below zero and neither stressed AUC nor K favours SAFE's choice with an interval that excludes zero. If either does, the sentence reads "lower Brier, worse ranking under faults", with both numbers. The ladder is reported as a pattern by level and never replaces the primary verdict.

**Code.** The generalized `paired_transforms` accepts only q = .10. A new, labelled version accepts the three levels; with level J1 it must reproduce the frozen outputs exactly.

## 4. Fairness: operating characteristics and a group audit

The frozen check stays the registered rule. The additions show how sensitive it is and what each selector's choice does to women and men at the level of decisions.

**1. Save every interval.** For every challenger the check assesses, blocked or not, save both joint intervals (women and men). Version 1 saved them only for blocked candidates.

**2. Operating characteristics.** In every repeat, apply the check to every ordered pair of candidates (12 pairs). This needs no refit. Report:

- how often each pair would be blocked;
- the precision of the check: the half-width of its joint interval, in Brier units, beside the overall gain. This describes how precisely harm is estimated; it is not the smallest harm the check reliably catches;
- in the synthetic study, how often the check blocks proposed switches that truly harm women or men, from setting F run at one prespecified strength, and the false-block rate in a known no-harm control (two fixed forecasts with equal true error for both sexes, 1,000 independent trials). Amended 21 November: the earlier plan (F at all four values of delta; false blocks in settings L0 to L3) is deferred to a later article (section 7.1).

**3. Group audit of each arm's choice, on judge records.** For women and for men, clean and under the primary judge scenarios, averaged over repeats with the corrected interval:

- Brier, AUC, and mean predicted minus observed default rate;
- at a fixed approval rate of 75%: one cutoff for all applicants together, never separate cutoffs by sex. Rank judge records by predicted risk, lowest first, breaking ties by record ID, and approve the first ceil(0.75 × n). Report the approval rate by sex, and the shares of defaulters approved and of non-defaulters refused by sex;
- the female-minus-male gap in each measure, and how that gap changes between SAFE's choice and Selector 2's.

**4. Margin gate (diagnostic).** Block a challenger unless the upper bound of its joint interval is below a harm margin of 0.0005 Brier for both sexes. Report how often it binds and what it costs on the judge's primary endpoint. The frozen check remains the registered one. This is a name change only; its behaviour is unchanged, and the name does not imply that the two gates are logically nested.

**5. Age, audit only.** Age bands under 30, 30 to 49, and 50 and over, alone and crossed with sex, in the group audit. Age is not used in any check.

**6. Limits stated in the article.** Labels exist for approved applicants only. Sex is withheld from the models, but its proxies are not audited. The disturbance swaps pool the sexes: a woman's record can receive a man's value.

**Wording.** The article never calls the method fair. It may say the switch was not resolvedly worse for either sex, and must say so with the intervals. An unresolved interval is not evidence of equal treatment.

## 5. Reporting rules

These rules fix, before the run, how every possible Brazil result is described.

**1. Low disagreement.** Template section 9 governs: no agreement count overrides the verdict. Step 1 section 8's row "the two selectors choose the same model in nearly every repeat" is replaced by this rule. Beside every verdict the report states:

- the number of repeats k in which the selectors chose differently, and in how many of those Selector 2 was judged better;
- this arithmetic: with 50 repeats and rho near 0.25, if every disagreement is a win of the same size, the corrected interval can exclude zero only when k is at least 27 (checked by Claude, 16 November). Below that, "supports" is out of reach by construction, and the report says so.

**2. Zero variance.** If the selectors agree in all 50 repeats, the interval is reported as 0 to 0 and the verdict is Unresolved, using the sentence fixed in template section 9. Step 1 section 7.2's "no interval" is replaced by this wording.

**3. Mixed results.** The first matching row governs the headline sentence.

| If the results show | The headline says |
| --- | --- |
| A sex contrast with its adjusted interval above zero | The harm to that group, in the headline sentence |
| Primary interval entirely above zero | "On Brazil, published SAFE selected models with lower stressed Brier than Selector 2", with the estimate, interval and disagreement count |
| Primary interval includes zero | "The Brazil comparison was inconclusive", with the estimate, interval and number of repeats in which the selectors chose differently (including 0 of 50). The Taiwan finding is not described as replicated, and the synthetic study does not carry the claim instead |
| Primary interval below zero, but stressed AUC favours SAFE (interval excludes zero) | "Lower stressed Brier, but poorer separation of defaulters from non-defaulters under faults" |
| Primary interval below zero, but K favours SAFE (interval excludes zero) | "Lower stressed Brier, but the faults changed more of the chosen model's original ordering" |
| Primary interval below zero, but log loss favours SAFE (interval excludes zero) | Both results, without a verdict word |
| Primary interval below zero, nothing above applies | The supports sentence below |

Lower AUC and lower K mean different things, and a lender may care about either: AUC is how well defaulters are ranked above non-defaulters; K is how much of the model's own ordering survives the faults. The headline names whichever applies.

**4. One confirmation dataset.** Template section 9's "confirmed if at least one dataset supports and none is against" is replaced for this study: the article says "supported on one independent dataset", never "confirmed". The planned plural confirmation became one dataset; this is reported as a deviation from the plan (section 6).

**5. The supports sentence.** "In a pre-specified test on one independent dataset (Brazil, PAKDD 2010), the model chosen by Selector 2 had lower judged Brier than the model chosen by published SAFE: difference [x], 95% corrected interval [y to z]. The two rules chose different models in [k] of 50 repeats. [The range and measure contrasts of section 2.] Against accuracy-only selection the difference was [result]. One dataset does not establish the result for other portfolios."

**6. Size.** Every difference is reported in Brier units beside each chosen model's clean Brier, and as a share of SAFE's choice's Brier. The direction-only verdict stays as registered. 0.001 Brier is used as a reference value, not as a threshold of practical importance. Against plain loss, the article may say "the Brazil interval rules out a loss larger than 0.001 relative to accuracy-only selection" only if the interval's upper bound is below +0.001. Calling 0.001 practically meaningful would need a reason tied to lending decisions, which is not claimed.

**7. Words the article does not use.** Confirmed, validated, generalizes, more robust, fairer, "SAFE's choice is wrong", "indistinguishable", "without an accuracy penalty", and "practically meaningful". Never present 48 of 50 repeats as 50 independent trials.

## 6. Permission amendment, recorded deviations and clarifications

### 6.1 Permission amendment (template section 2)

Version 1 says prior published use "is not permission" and that a dataset without documented permission is not modelled. Brazil has no located licence, so using it needs this amendment, made before any Brazil model is fitted.

**Amended rule.** A dataset without a located licence may be used only if all of the following hold, and each is recorded in its form:

1. The holder released it publicly for research or a competition.
2. It has been used in peer-reviewed academic publications.
3. No located terms prohibit research use or publication of aggregate results.
4. Before the run, the authors write to the holder (for Brazil: the organisers at NeuroTech and the Center for Informatics, Federal University of Pernambuco), and record the attempt and any answer.
5. The project does not redistribute the data, and the article's data statement discloses the situation.

**Consequence for the repository.** The Brazil files now in `data/raw/brazil_pakdd2010/` must not be pushed to the public repository. Add the folder to `.gitignore` and record the hashes only.

**Re-screen of dropped datasets under the amended rule.** No change: Home Credit has explicit academic permission but was dropped for run time; Bondora's terms restrict use and no official download is available; Figshare PPDaiData has a licence but failed on structure; PPDAI LC/LP was never obtained; Renrendai has no archived data.

### 6.2 Recorded deviations

| Date | What happened | Effect |
| --- | --- | --- |
| 16 November | During the original-file check, Claude cross-tabulated Brazil's outcome by submission type and by product, beyond the counts template section 3 allows. The tables were seen by Claude only and were not used. | No preparation rule was chosen afterwards from outcome information. Every rule in the Brazil form comes from the variable list. |
| 16 November | Three template rules (duplicate fields, built-outcome completeness, verdict reporting) were added with Brazil's file open, after the preamble claimed all rules were fixed before any dataset was chosen. | No outcome information drove them. The preamble is corrected to point to the change log. |
| 16 November | The plan for several fresh datasets became one (Brazil). | Reasons recorded before any result (section 7). Reported in the article as a deviation. |
| 16 to 17 November | Brazil's permission decision was recorded before this amendment. | Formalised here, before any model is fitted. |
| 16 November | The Brazil form listed 24 excluded fields; the explicit list has 25. | Corrected; the lists were always applied as written. |
| 17 November | The artificial-data manual run `manual_run_2026-11-17_03` failed at Job B's model-identity check: `joblib.hash` of fitted/reloaded estimators was unstable, consistent with unused padding in tree-node arrays. Before run 04, Job A was changed to score reloaded saved models; model-file identity became SHA-256; cross-run and before/after comparisons became exact all-row probabilities plus named tree fields and fitted arrays. | Execution-adapter fix only: no refit in Job B, tolerance relaxation, science-rule, configuration or seed change. Ten unit tests and run 04's two-repeat serial/parallel and J1 replay gates passed. Old receipts stay unchanged; fresh output was required. See section 6.4 and the saved runtime deviations notes. |

### 6.3 Clarifications

- **Node counts.** S-bar averages six nodes: the clean node and tail fractions .01 to .05. The fairness check and section 3.6 of Step 1 use the five stressed nodes, .01 to .05.
- **Label.** "Accuracy retained" for S is replaced by "Brier skill under mild disturbance". By the identity S-bar = S_0 − (5/6) × mean mild loss of skill, it measures a level, not retention. True retention is the registered diagnostic arm of section 2.
- **Second reviewer.** Astra classified Brazil's predictors blind. Andrei settled the differences on 17 November: assets and cars stay under income and employment.
- **Candidate settings.** Sol states in the form whether the four configurations were tuned on Taiwan outcomes. The four-model pool is a stated limit of the claim.
- **Timing check.** Section 6.4 governs execution checks and runtime reporting. The artificial-data benchmark reports time and sampled peak memory, not model-selection results or performance estimates.

### 6.4 Run time and execution (accepted 17 November before any Brazil fit)

**Split into two jobs.** Job A performs all 50 repeats of selection and the frozen J1 judge, saves every fitted candidate model, predictions, selection-side values and seeds, and produces the primary results. Job B computes J2/J3, rank measures, group audits and every-pair gate records from Job A's saved models and seeds, without refitting. It must reproduce Job A's J1 outputs exactly. Headline flags requiring secondary endpoints wait for Job B.

**Parallel repeats.** Repeats may run independently in parallel: default three worker processes, each limited to two numerical-library threads, including OMP, MKL and OpenBLAS. Repeat seeds and outputs do not depend on worker identity or completion order. Before the benchmark, the same two discarded artificial-data repeats run serially and in parallel must match exactly for both jobs, including Job A-to-B J1 replay.

**Separate wall-clock budgets.** Each job starts with a 12-hour limit. First run a parallel benchmark of three discarded artificial-data repeats and report wall time and sampled peak memory separately for Jobs A and B. With three workers, the conservative projection for 50 repeats is `ceil(50/3) = 17` times the three-repeat batch wall time. If a job projects above 12 hours but no more than 24 hours, record its new 24-hour limit before continuing. Above 24 hours, stop for a new version. A job never borrows the other job's allowance. These are machine/distribution-dependent projections, not guarantees for Brazil.

**Diagnostic name.** Use "margin gate (diagnostic)" in current code and reports, identifier `selector2_margin_gate`; the section 4 rule is unchanged. Historical hash-bound files retain their original terminology.

**Saved-model identity.** Job A saves and reloads each candidate before scoring. SHA-256 and artifact manifests establish the identity of saved files. Across serial/parallel runs, compare exact probabilities on all saved fitting, selection and judge rows, input values and logical fitted state, with tree-node arrays compared by named fields rather than raw bytes. The same padding-free values guard models before/after Job B. Serialised files need not have identical bytes across independent runs when only unused padding differs; no statistical tolerance is introduced. This execution-adapter correction is recorded in section 6.2.

**Resource safeguard.** Sample aggregate RSS of the controller and its owned process tree every 100 ms; shared pages can be counted more than once. Stop owned workers if available system memory stays below 128 MiB for five seconds. Do not terminate unrelated processes or silently reduce worker count. This safeguard does not change a scientific rule.

**Recorded preparation outcome, 17 November.** Andrei ran `manual_run_2026-11-17_04` himself. Its saved unit, serial/parallel and J1 replay gates passed. Job A projects to 3.175107 hours and Job B to 4.351529 hours for 50 repeats; both recorded limits remain 12 hours, with no extension or stop for a new version. See the [section 6.4 status report](../../../research/model_selection_v2/confirmation_runtime_preparation_2026-11-17/report.md) for timings, memory, evidence and limitations. The report was written from saved files only; it is not a Brazil result or a public timestamp. Final documents, code and their updated hashes still join the pre-run frozen/tagged set under section 7.2.

### 6.5 Direct small-swap Brier (accepted 18 November before any Brazil fit)

A secondary comparison arm, direct small-swap Brier (`direct_small_swap_brier`), selects the candidate with the highest S-bar alone, with no A-bar, E-bar or fairness check. Its rule, contrasts, reporting and collectors are specified in full in `docs/frozen_inputs/2026-11-18_v2_1/section_6_5.md`, which governs and is listed in the manifest of section 7.2. The primary contrast that governs the headline remains Selector 2 minus SAFE.

## 7. Change log and public timestamp

### 7.1 Change log

Newest first. Every entry predates any Brazil model fit.

| Date | Document | Change |
| --- | --- | --- |
| 23 November | Registration and synthetic protocol | After blinded timing, Andrei fixed 150 independent repeats for R and 300 for F before production. F uses the accepted single strength of 0.5. Brazil remains at 50 repeats and runs first. The public registration records these choices, the two-stage Brazil execution, and the final synthetic reporting scope. |
| 21 November | This amendment, sections 1, 3, 4, 6.5, 7.2 and 8 | Synthetic study reduced to a lean design before any synthetic fit: one fragile-interaction setting R, one group setting F at a single prespecified strength, and a no-fit no-harm control (`research/model_selection_v2/synthetic_lean/`). Deferred to a later article: settings L0 to L3, F at four strengths, the nested coverage check of the corrected interval, every-pair hypothetical gate checks in the synthetic study, and separate attribution to A and E. Reason: the runtime budget considered at the 21 November review and the scale of simulation in the SAFE literature. Section 6.5 added by reference; section 3 wording on setting R corrected from "shows" to "suggests" (it was a conjecture); manifest list and open items updated. No Brazil fit, no change to the Brazil primary endpoint, verdict rules or registered arms. |
| 18 November | Section 6.5 addendum (separate file) | Direct small-swap Brier added as a secondary comparison arm for Brazil, after the exploratory Taiwan headroom analysis. |
| 17 November | This amendment, sections 4, 6.2, 6.3, 6.4 and 8 | Incorporated the accepted Job A/B split, three-process/two-thread execution, per-job 12/24-hour rules and diagnostic rename; recorded the padding-safe model-identity fix and the saved run 04 preparation outcome. No Brazil fit or scientific rule change. |
| 17 November | Template and Step 1 | Version 2 (this document): sections 2 to 6, revised after Astra's review (paired range and measure contrasts replace match-count attribution; headlines for every primary result; precision wording for the fairness check; one approval cutoff for all applicants; 0.001 as a reference value) |
| 17 November | Brazil form | Judge families settled by Andrei after Astra's blind review |
| 16 November | Study plan | Home Credit dropped (about 70 predictors; run time); Bondora dropped (no download; restrictive terms); Brazil the only confirmation dataset; synthetic study in the plan |
| 16 November | Template section 4 | Duplicate-field exclusion rule |
| 16 November | Template section 3 | Completeness rule for built outcomes |
| 16 November | Template section 7.2 | Dictionary-thirds fallback labelled a synthetic stress test |
| 16 November | Template section 9 | 45-of-50 "no test" override removed; disagreement counts reported beside every verdict |
| 16 November | Template section 12 | Remaining defaults accepted |
| 16 November | Template | First draft |
| 14 November | Step 1 | Specification frozen; Taiwan development run |

### 7.2 Public timestamp

Before the first Brazil model is fitted:

1. Export the final versions of these documents to `docs/frozen_inputs/` in the repository: Step 1 specification, template, this amendment, the section 6.5 addendum (`2026-11-18_v2_1/section_6_5.md`), the completed Brazil form (`handoff_documents_v2_2026-11-17/brazil_addendum_form_v2_CANDIDATE_ORIGIN_COMPLETED.md`, not the DRAFT copy), and the synthetic study specification (`synthetic_lean/protocol.md` and `generators_spec.md`), together with the synthetic generator and pipeline code and the Brazil entry-point code.
2. Publish the SHA-256 hashes of the final documents and code in the registration manifest. Bind the locally retained timing evidence through its manifest hash.
3. Andrei commits them, with the Brazil data folder excluded, and pushes a tag named `prereg-v2-2026-11` to the public repository. The tag's date is the public timestamp. An OSF registration with an embargo is an alternative.
4. Andrei launches the registered code in local PowerShell. The launcher verifies the tagged inputs and records the tag, commit and manifest hash. Brazil runs before the synthetic study.

Step 1's ban on commits and pushes applies to the agents running the code. This step is done by Andrei.

## 8. Open items for Andrei

Each has a default that applies unless changed before the timestamp in section 7.2.

**Numbers to fix now**

| Item | Value | Status and use |
| --- | --- | --- |
| Reference value for differences | 0.001 Brier | Reported beside every difference. Not called practically meaningful (section 5, item 6). |
| Bound against plain loss | 0.001 Brier | The article may say the interval rules out a loss larger than 0.001 relative to accuracy-only selection only if the upper bound is below +0.001. |
| Approval rate for the group audit | 75%, one cutoff for all applicants | Declared study scenario (section 4). |
| Harm margin of the margin gate (diagnostic) | 0.0005 Brier | Diagnostic only; behaviour unchanged (sections 4 and 6.4). |
| Severity ladder | J2 = 25% of records at 0.25 × IQR; J3 = 50% at 0.50 × IQR | Declared study scenarios (section 3). |

Accepted by Andrei and Astra on 17 November, with the wording changes above.

**Referee proposals not adopted by default**

| Proposal | Default | Cost if adopted |
| --- | --- | --- |
| Split-half replication on Brazil (two halves of 25,000, 50 repeats each) | Not adopted | About twice the Brazil run time; gives two verdicts from one portfolio |
| A judge that holds out whole Brazilian states (about 20% of records) | Not adopted | One more design; the state list must be fixed now without outcomes |
| A rerun with category codes permuted, to check that swaps on unordered fields do not drive results | Not adopted; cells changed per node are recorded instead | One more full run |
| A wider candidate pool (several settings per family) | Not adopted; the four-model pool is a stated limit | Changes a frozen input |
| A coverage check of the corrected interval by simulation | Moved to the synthetic study on 17 November; deferred to a later article on 21 November (section 7.1). The article reports corrected intervals without claiming their coverage was checked. | A nested design (many independent portfolios, 50 splits each) exceeds the synthetic budget |

**Actions before the timestamp**

- [x] Andrei accepted the defaults above on 17 November.
- [x] Andrei: write to the PAKDD 2010 organisers about research use (section 6.1); record the date. 17th nov 2026
- [x] Andrei: exclude `data/raw/brazil_pakdd2010/` from the public repository.
- [x] Sol: implement the judge levels, endpoints, arms and audits; level J1 must reproduce the frozen outputs exactly; re-establish the preservation baseline. (Implemented; 49 checks, noted 18 November.)
- [x] Sol: state the origin of the four candidate configurations in the Brazil form. (Done in `brazil_addendum_form_v2_CANDIDATE_ORIGIN_COMPLETED.md`.)
- [x] Section 6.5: direct small-swap Brier arm and its Job A/B collector, tested on artificial tables (18 November).
- [x] The Brazil entry point and section 6.5 exports passed artificial-data checks, including exact Job B replay of J1.
- [x] Section 6.4 preparation: saved run `manual_run_2026-11-17_04` passes the ten unit tests, two-repeat exact serial/parallel comparison and three-repeat artificial-data benchmark; both job limits remain 12 hours. Status report written from saved files only.
- [x] Synthetic study: scope settled 21 November (R, F, no-harm control); specification and code drafted in `research/model_selection_v2/synthetic_lean/`.
- [x] F is fixed at 0.5. R tests whether selecting on small swaps transfers to independent judge faults and can expose a limit of the rule.
- [x] Synthetic tests, the truth-only R reference and F calibration, the blinded timing batch and the repeat-count decision are complete. The final [synthetic protocol](../2026-11-23_prereg/protocol.md) reports judge error against true risk. Observed binary outcomes are used for fitting, selection and the veto.
- [ ] Freeze, hash and tag (section 7.2); then run Brazil.
