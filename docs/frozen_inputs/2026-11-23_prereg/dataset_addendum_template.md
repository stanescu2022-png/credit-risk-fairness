# Dataset addendum template for the Step 1 confirmation runs

2026-11-16 · @Andrei

## 1. Purpose and status

This template fixes, before any fresh dataset is chosen, every decision that the frozen Step 1 specification leaves to the dataset. Its purpose is that no rule is written with a dataset's results in view.

**What it is**

- A set of general rules (sections 2 to 9) that apply to every fresh dataset.
- A form (section 11) that is filled in for each dataset by applying those rules, then hashed before any model is fitted.

**What it is not**

It does not change the Step 1 specification. Selector 2, the fairness check, the judge's primary endpoint, the arms and the diagnostic variants stay exactly as frozen on 16 November 2026.

**Why it is needed**

The Taiwan code for the judge and for parts of SAFE names Taiwan's 22 predictors directly. The judge's nine scenarios are built from three Taiwan predictor families. Neither carries over to another dataset without a rule.

**Order of work for each dataset**

1. Confirm permission and pin the source file (section 2).
2. Write the cohort and outcome definition, then count once (section 3).
3. Classify the predictors and apply the cleaning rules (section 4).
4. Fix sex coding and grouping (section 5).
5. Form the judge's families (section 7).
6. Fill in and hash the form (section 11).
7. Pass the gates (section 10).
8. Only then run the frozen Step 1 procedure.

**Status**

Draft for Andrei's review. Once accepted it is frozen, and any later change is a new, labelled version made before the affected dataset is touched.

## 2. Dataset identity and permission

A dataset enters the study only when its source, exact file and permission to publish results are documented.

- **Source.** The organizer, lender or repository that released the data, with a link to the original release or the closest authoritative copy.
- **Permission.** The terms that cover academic use and publication of aggregate results, quoted or linked. Prior published use by others is supporting evidence. It is not permission.
- **Exact file.** File name, size and SHA-256 hash of every file used, and where it was obtained. A mirror is acceptable only if its content can be checked against published totals from the original release.
- **Dictionary.** The field dictionary used, with its source. Fields without a documented meaning are not used as predictors.
- **Prior use.** Whether the SAFE papers or this project have already used this release. A release already modelled in this project is not fresh.

If permission cannot be documented, the dataset is recorded as unresolved and is not modelled.

## 3. Cohort and outcome

The outcome definition is written first and the defaults are counted once; the definition is never adjusted after the counts are seen.

**When the release documents its own outcome**

Use that definition as documented, and quote it. Confirm which code means default by reproducing a published total.

**When the outcome must be built from repayment records**

| Element | Rule |
| --- | --- |
| Default | At least 90 days past due on any instalment that falls due within the performance window |
| Performance window | 12 months from origination, or the loan's term if shorter |
| Cohort | Loans whose whole window, plus 90 days, ends before the data extract date |
| Non-default | A loan in the cohort without such arrears whose repayment record is complete for the whole window: a payment status for every instalment due in the window, or full repayment recorded within it. Loans repaid early without such arrears are non-defaults. A loan without such arrears whose record has a gap in the window is neither: it leaves the cohort, and the number left out is reported overall and by sex before the thresholds are checked |
| One fallback, declared in advance | 60 days past due within 6 months, used only if the main definition cannot be computed from the available fields |

The fallback is for missing fields. It is not used because the main definition gave too few defaults.

**Counting**

- Count records and defaults overall and by recorded sex, including unknown sex codes, once.
- Compare with the inclusion thresholds of Step 1 section 10: 20,000 records, 2,000 defaults, 500 defaults for each sex.
- A cohort that misses a threshold is ineligible. Do not change the definition to rescue it.

**Large cohorts**

If the cohort has more than 60,000 records, draw one random subsample of 60,000 before anything else. Use seed 20261116, sample whole groups (section 5), and do not stratify. The records left out are not used. The cap keeps run time within the 12-hour limit of Step 1.

**Recorded for every cohort**

Origination period, whether records carry dates, the share of applicants who were approved if known, and any documented sampling by the lender. Most public credit data cover approved customers only; state this as a limit.

## 4. Predictors and cleaning

Predictors are the fields a lender would hold when scoring the applicant, prepared by fixed rules that never look at the outcome.

**Which fields are predictors**

Include every documented field known at the time of the credit decision. Exclude:

- Record and customer identifiers.
- Recorded sex.
- Anything measured after origination, and anything derived from the outcome.
- Risk grades, scores or prices assigned by the lender's own model (for example a platform rating or the interest rate set from it).
- Free-text fields.
- Unordered categorical fields with more than 50 distinct levels in the cohort (for example city or postcode).
- Fields missing for more than half of the cohort, and fields with a single value.
- A field identical in every record to a field earlier in the dictionary (the earlier field is kept). Fields that carry the same information under different labels are also listed, with the later one excluded.

List every excluded field with its reason.

**Type of each predictor**

Assign one type to each predictor from the dictionary, before any modelling. Section 7 uses these types.

| Type | Meaning | Examples |
| --- | --- | --- |
| Numeric | A measured amount, count or age | Income, loan amount, age, number of dependants |
| Ordered | Categories with a documented order | Education level, months of delay |
| Unordered | Categories without an order, including yes/no flags | Marital status, residence type, a verification flag |

For an ordered predictor, write down the order of its levels and any special codes that stand outside the order (for example "not applicable").

**Coding**

- Unordered and ordered levels are coded as integers. Ordered levels follow their documented order; unordered levels follow the sorted order of their original labels.
- Coding is fixed once for the whole cohort from the labels alone.

**Missing values**

- Numeric: replace with the median of the fitting records of the repeat.
- Ordered and unordered: missing becomes its own level. For an ordered predictor it is a special code outside the order.
- No missing-value indicator columns are added.

**Candidate models**

The four configurations and their preprocessing stay as in the pilot: standard scaling for numeric predictors, one-hot encoding for ordered and unordered predictors, the same estimator settings and seeds.

**Nothing else**

No feature engineering, selection, transformation, pooling of levels or outlier handling. Any departure is a stop condition.

## 5. Sex and grouping

Recorded sex is used only to audit results, and related records always stay on the same side of every split.

**Sex**

- Map the dataset's codes to female, male and unknown, and record the mapping with its dictionary source.
- Confirm the mapping against a published count where one exists.
- Records with unknown sex stay in every overall result. They are left out of the female and male results and of the fairness check.
- If unknown sex exceeds 5% of the cohort, say so beside every result by sex.
- Sex is never a model input, and no predictor is dropped for being correlated with it.

**Grouping**

A group is the unit that is kept whole in every split and resampled whole in every bootstrap.

| What the data offer | Group |
| --- | --- |
| A borrower or customer identifier | All records of that identifier |
| No identifier | Records with identical values on every predictor, as on Taiwan |

If the dictionary shows that borrowers can appear more than once and no identifier exists, record this as a limit. Identical-predictor grouping does not remove that dependence.

**Counts to record**

Number of groups, the distribution of group sizes, and how many groups contain both sexes or both outcomes.

## 6. Split and repeats

Every dataset gets the same 50-repeat random design as Taiwan for its primary result, and a later-period judge as a secondary result where records carry dates.

**Primary design, every dataset**

Exactly as Step 1 section 6: 50 repeats, seed 20261114 plus the repeat index, whole groups assigned to judge (20% of records), selection (16%) and fitting (the rest), no stratification. The primary estimand and the corrected interval of Step 1 section 7 apply unchanged.

Using one design everywhere keeps the results comparable across datasets.

**Secondary design, only where records carry an origination date**

1. Order groups by the earliest origination date in the group.
2. The latest groups, holding 20% of records, form the later-period judge.
3. Split the remaining groups at random into fitting (80%) and selection (20%), with seed 20261114.
4. Fit, select and judge once, with every definition as in Step 1.
5. For uncertainty, resample whole judge groups 2,000 times (seed 42) with the fitted models and the choices held fixed. Report 95% percentile intervals.

**How to read the secondary design**

- Its interval covers sampling of later-period borrowers only. It does not cover refitting or reselection, so it is not comparable in width to the primary interval.
- It answers a different question: does the choice made on earlier loans hold up on later ones?
- If its direction differs from the primary result, report that beside the primary result, not in a later section.
- It never replaces the primary result in the claim across datasets (section 9).

**Recorded for every design**

Role sizes, default counts by sex in each role, the date range of each role where dates exist, and the seeds.

## 7. The judge's disturbances on any dataset

The judge keeps the Taiwan structure of nine scenarios, built from three predictor families by fixed rules that use the field dictionary and the fitting records, and never the outcome.

### 7.1 Structure kept from Taiwan

- Three families of predictors.
- For each family, three modes: mask the family, corrupt every predictor outside the family, or both. That gives nine scenarios.
- In each realization, a random 10% of judge records is disturbed (q = .10). The same records are used for the three modes of a family.
- 30 realizations per family.
- Reference values come from the fitting records of the repeat.
- The primary endpoint stays the mean stressed Brier over the nine scenarios.

### 7.2 Forming the three families

A family is a set of fields that would plausibly reach the lender together and fail together. These are not verified production feeds.

1. If the release's documentation already groups its fields (form sections, source tables), use those groups.
2. Otherwise assign each predictor to one category from this fixed list: applicant characteristics; income and employment; requested credit and product terms; existing credit and past repayment; residence and contact; verification and documents; other.
3. A category with a single predictor joins "other".
4. The three categories with the most predictors are the families. Ties go to the category whose first field comes earlier in the dictionary.
5. Fallback: if fewer than three categories hold two or more predictors, or one family holds more than half of all predictors, use three blocks of near-equal size in dictionary order. Where this fallback is used, the report calls the judge's scenarios a synthetic stress test, because neighbouring fields in a dictionary are weak evidence that those inputs would fail together.

Two reviewers classify the predictors independently from the dictionary alone. Disagreements are listed, and Andrei settles them before the form is hashed.

### 7.3 Masking a family

On the selected records, each predictor of the family is replaced by its reference value: the fitting mean for a numeric predictor, the most frequent fitting level for an ordered or unordered one (smallest code on a tie). This is SAFE's own masking convention.

### 7.4 Corrupting the predictors outside the family

On the selected records, each predictor outside the family changes by its type. The direction is drawn at random, independently for each record and predictor.

| Type | Rule |
| --- | --- |
| Numeric | Add or subtract q times the fitting interquartile range. Round to the precision seen in the fitting records. For a whole-number field whose rounded shift would be zero while the range is positive, shift by 1. Keep the result within the fitting minimum and maximum. A zero range means no change. |
| Ordered | Move one level up or down in the documented order, only if that level occurs in the fitting records. Otherwise leave the value unchanged. Special codes are never changed and never a destination. |
| Unordered | Replace with a different level drawn in proportion to fitting frequencies. A predictor with one level is unchanged. |

These generalize the Taiwan rules: amounts moved by q times the interquartile range, age by one year, repayment status by one step, categories by frequency. One difference is deliberate: every numeric predictor is kept within its fitting range, where Taiwan bounded only some.

### 7.5 Seeds

Use the existing SHA-256 child-seed construction with master seed 20261007, scope `confirm_v1:<dataset id>:<family>`, repetitions 1 to 30, and components `mask`, `sign:<predictor>` and `category:<predictor>`. The same seeds are used in every repeat, as on Taiwan.

### 7.6 Recorded for every scenario

Records selected and changed, cells changed, values left unchanged by a rule, values clipped to the fitting range, and levels absent from the fitting records.

### 7.7 Limits to state

These are synthetic input faults chosen for this study. They are not measured error rates of any lender, and they differ in kind from the tail swaps that the selectors use.

## 8. What stays frozen and what may adapt

Everything that defines the comparison is frozen; only the items below may differ from the Taiwan run, and only as this template states.

**Frozen from the Step 1 specification**

- Selector 2: the mild grid (tail fractions 0 to 0.05), accuracy retained S, Score 2, the fairness check and the choice rule (section 3).
- The six arms, including plain loss and the diagnostic variants (section 4).
- The judge's primary and secondary endpoints (section 5).
- 50 repeats, the split proportions and seeds (section 6).
- The primary estimand, the corrected interval and the within-repeat intervals (section 7).
- The reading of results, direction only, with no practical cutoff (section 8).
- The four candidate configurations.

**Permitted adaptations (complete list)**

| Item | How it adapts |
| --- | --- |
| Predictor list, types and coding | Section 4 of this template |
| Missing values | Section 4 |
| Sex coding and grouping unit | Section 5 |
| SAFE grid size | SAFE's node count follows the number of predictors p, as in `safe2026_specification.md` with 22 replaced by p. Sol confirms the exact rule there before coding. |
| Judge families and corruption rules | Section 7 |
| Seed scope strings for the judge | Section 7.5 |
| Subsample cap | Section 3 |
| Later-period judge | Section 6, secondary only |

**Code**

Several existing functions name Taiwan's predictors directly (for example `fitting_references`, `build_candidate`, `paired_transforms` and `semantic_rules`). For fresh datasets Sol writes generalized versions in a new package that take the predictor lists and types as arguments. The existing files are not edited.

**Equivalence gate**

Before any fresh dataset is run, the generalized SAFE, Selector 2 and split code, given Taiwan's predictor lists, must reproduce repeat 0 of the Taiwan Step 1 run to 1e-12: all selection-side values and every arm's choice. The generalized judge is checked on synthetic fixtures against hand calculations, because its numeric bounds differ from Taiwan's by design.

**Known inheritances**

- Tail swaps on unordered predictors depend on an arbitrary code order. Taiwan has the same property.
- The three axes keep equal weights on their raw scales, as in SAFE.

## 9. The claim across datasets

Each fresh dataset gives one verdict from its primary contrast, and the article reports every verdict.

**Verdict for one dataset**

From Selector 2 minus SAFE on the judge's primary endpoint, with the corrected interval of the 50-repeat design:

| Result | Verdict |
| --- | --- |
| Interval entirely below zero | Supports the claim |
| Interval includes zero | Unresolved |
| Interval entirely above zero | Against the claim |

**How often the selectors chose differently.** The mean difference and its corrected interval are always computed over all 50 repeats. No repeat is dropped, and no agreement count overrides the verdict. Beside every verdict, report in how many of the 50 repeats the two selectors chose different models, and in how many of those repeats Selector 2's choice was judged better, for example "different models in 3 of 50 repeats; Selector 2's choice judged better in 3 of 3". Together these tell the reader how much room there was for an improvement and how many repeats a verdict rests on.

- If they chose the same model in all 50 repeats, write: "The procedures selected the same model in every repeat, so the observed difference was zero." The interval then includes zero and the verdict is Unresolved.
- If they chose differently in only a few repeats, say so with the number, for example "The procedures selected different models in 3 of 50 repeats."

**Statement across datasets**

| Pattern over the fresh datasets | Statement the article may make |
| --- | --- |
| At least one supports, none against | The Taiwan finding is confirmed on fresh data, with the count of each verdict |
| Any against | Mixed evidence; report each dataset |
| All unresolved | Not confirmed; the Taiwan result remains a single-dataset finding |

**Rules**

- Every dataset that passes the gates and is run is reported, whatever its verdict.
- A dataset is dropped after selection only for failing a gate in section 10, never for its results.
- Taiwan is reported separately as the development example and is not counted among the fresh datasets.
- No pooled estimate across datasets is the basis of a claim. With two or three datasets a pooled interval would not be credible.
- Sizes are reported in raw Brier units beside each model's clean Brier. No practical cutoff exists, and none is applied.
- The diagnostic variants, plain loss and the fairness results are reported for every dataset in the same form as on Taiwan.

## 10. Gates and stop conditions

No model is fitted on a fresh dataset until every gate below is passed and recorded.

**Gates before any model is fitted**

- [ ] Permission documented and source files hashed (section 2).
- [ ] Outcome definition written and dated before any default was counted (section 3).
- [ ] Counts reproduced from the pinned file and compared with published totals; inclusion thresholds met.
- [ ] Predictor list, exclusions, types and coding recorded from the dictionary (section 4).
- [ ] Sex mapping and grouping unit recorded (section 5).
- [ ] Families classified by two reviewers, disagreements settled (section 7.2).
- [ ] The filled form hashed, with this template's hash and the Step 1 specification's hash beside it.
- [ ] Equivalence gate on Taiwan passed (section 8).
- [ ] No statistic involving the outcome has been computed beyond the counts of section 3.
- [ ] No existing project file edited; hash inventory taken.

**During the run**

The gates, tests, deliverables and stop conditions of Step 1 section 9 apply, including the timing check after the first two repeats and the 12-hour limit.

**Stop and report, without workaround, if**

- A rule in this template cannot be applied to the dataset as written.
- A field's type, order or special codes cannot be settled from the dictionary.
- A candidate fails to fit, or returns invalid probabilities.
- The projected run time exceeds 12 hours after the subsample cap.
- Any step would need a rule chosen after seeing results.

A stop is reported with the reason. The remedy is a new, labelled version of the form or this template, agreed before the run resumes.

**After the run**

Results go into the report in the Step 1 format, with this dataset's form attached. A same-session recheck is called a recheck. Root reconstructs the principal table independently.

## 11. The form for each dataset

One copy of this form is filled in per dataset by applying sections 2 to 7, then hashed. An entry that cannot be filled is a stop, not a blank.

| Field | Entry |
| --- | --- |
| Dataset id (short, used in seed scopes) |  |
| Source and link to the original release |  |
| Permission: terms quoted or linked |  |
| Files used: name, size, SHA-256, where obtained |  |
| Dictionary used and its source |  |
| Prior use in SAFE papers or this project |  |
| Outcome: documented or built; exact definition |  |
| Date the outcome definition was written |  |
| Cohort rule and extract date |  |
| Records and defaults: overall, women, men, unknown sex |  |
| Published totals reproduced (which, and result) |  |
| Subsample cap applied (yes or no; records kept) |  |
| Predictors included, each with type |  |
| Fields excluded, each with reason |  |
| Order and special codes of every ordered predictor |  |
| Missing values: share per predictor |  |
| Sex code mapping and its source |  |
| Grouping unit; number and sizes of groups |  |
| Repeat borrowers possible without an identifier (yes, no, unknown) |  |
| Dates available; later-period judge planned (yes or no) |  |
| Families: source of grouping, the three families and their predictors |  |
| Family classification: reviewers, disagreements and how settled |  |
| Number of predictors p and resulting SAFE grid size |  |
| Known limits (approved-only sample, lender sampling, other) |  |
| Hashes: this form, the template, the Step 1 specification |  |
| Filled by, reviewed by, date frozen |  |

## 12. Open items for Andrei

These defaults were chosen while drafting. Each is a judgment without a derivation, and each should be accepted or changed before the template is frozen.

| Item | Default in this template | Alternative |
| --- | --- | --- |
| Lender's own grades and prices as predictors | Excluded | Include them, as much of the peer-to-peer literature does; models then lean on the lender's own score |
| Unordered fields with many levels | Excluded above 50 levels | Pool rare levels inside the model pipeline; keeps information but changes the pipeline |
| Missing numeric values | Fitting median, no indicator column | Add indicator columns; changes the predictor set |
| Built outcome | 90 days past due within 12 months | Another threshold or window, fixed before counting |
| Subsample cap | 60,000 records | A higher cap with a longer run-time limit |
| Numeric corruption bounds | Fitting minimum and maximum for every numeric predictor | Dictionary bounds only, closer to the Taiwan rule |
| Family rule | Documentation groups, else fixed category list, else dictionary-order thirds | Dictionary-order thirds always; no discretion, less meaning |
| Later-period judge | Secondary | Primary where dates exist; closer to credit practice, but its interval does not cover the selection procedure |
| Statement across datasets | Confirmed if at least one dataset supports and none is against | A stricter rule, for example a majority of datasets |

Two matters sit outside this template and remain open: the wording of the article's question, and whether more than two datasets can be found.
