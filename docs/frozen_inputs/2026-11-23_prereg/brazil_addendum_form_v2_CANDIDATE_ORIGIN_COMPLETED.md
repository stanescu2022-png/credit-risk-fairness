# Brazil PAKDD 2010 dataset addendum, version 2

Completed 17 November 2026 and included in the [public registration](registration.md) on 23 November 2026.

The template's rules applied to Brazil give 29 predictors, no subsample, 49,993 groups and three judge families. Every entry below was derived from the pinned file and the variable list without using the outcome, except the counts that section 3 of the template requires.

## 1. Identity, permission, outcome and counts

| Field | Entry |
| --- | --- |
| Dataset id | brazil_pakdd2010 |
| Source and link | PAKDD 2010 Data Mining Competition, Brazilian private-label credit card applications. Original release: the organizers' competition server, sede.neurotech.com.br/PAKDD2010 (NeuroTech Ltd. and the Center for Informatics, Federal University of Pernambuco). File used: a copy in a third-party GitHub repository, github.com/rishabhpahuja/Credit-Score-, Dataset/PAKDD/. It is not the organizers' own release; that it matches the original is supported by the counts reproducing the derived mirror and published totals. |
| Permission | No licence located. Decision: rely on long-standing published academic use (Andrei, 16 November 2026). Andrei: "Considering the long-standing published academic use of the data, I make the decision of using it for our study." |
| Files used | PAKDD2010_Modeling_Data.txt, 8,394,950 bytes, SHA-256 ae00807f566e7c9a4ab804432665a7d50ba54de0d9530f064c8e20f2d32c25c6. PAKDD2010_VariablesList.XLS, 13,824 bytes, SHA-256 afb8860352b26da04044b11425fbf4e96c51e71252a2eee39af5cca71a720355. Both supplied by Andrei, 16 November 2026. |
| File format | Tab-separated, no header row, CRLF line ends, Latin-1. 50,000 rows, 54 fields in every row. Column names from the variable list; column 44, a second EDUCATION_LEVEL, is the mate's and is named MATE_EDUCATION_LEVEL. |
| Dictionary | PAKDD2010_VariablesList.XLS (54 entries: name, description, codes) |
| Candidate-setting origin and tuning | No outcome-based hyperparameter search or tuning was performed, according to the saved 26 October pilot protocol and implementation. Four manually specified, fixed pilot configurations were adopted: LR L2 C=1/lbfgs/max_iter=2000; DT depth=5/min_leaf=20; RF 300 trees/min_leaf=5/sqrt/n_jobs=1; GBM 100 trees/rate=.1/depth=3. Seeds 42. These are not all sklearn defaults. Sources: `runs/model_selection_v2/pilot_candidates_2026-10-26/protocol.json` (`candidate_status`: "four fixed, untuned pilot configurations"; tuning listed under prohibited operations); `research/model_selection_v2/README.md`, "Fixed, untuned candidates"; `research/model_selection_v2/pilot.py`, `build_candidate`. This records declared and implemented absence of tuning, not proof that manual design choices were made blind to every prior Taiwan result. |
| Prior use in SAFE papers or this project | Not modelled in this project. No PAKDD 2010 use found in the core SAFE-family papers (Sol, 16 November; bounded check, sources in `confirmation_preparation_2026-11-16`). |
| Outcome | Documented: TARGET_LABEL_BAD=1, "BAD=1, GOOD=0". Defined in the release as 60 days late within the first year. Organisers' archived Problem Characterization page: "The delinquency of the modeling data set is 26% according to 60 days delay in any payment of the bills contracted along the first year after the credit has been granted." Source: https://web.archive.org/web/20150925091439/http://sede.neurotech.com.br/PAKDD2010/login.do?method=redirecionar&vs_Pagina=problemcharacterization (screenshots supplied by Andrei; see `organizer_outcome_evidence.md`). |
| Outcome definition written | Documented by the release; no definition built |
| Cohort | All 50,000 labelled records. No extract date in the release. |
| Records and defaults | Overall 50,000 and 13,041 (26.1%). Women 30,805 and 7,790. Men 19,130 and 5,228. Unknown sex 65 and 23. |
| Thresholds (20,000 records; 2,000 defaults; 500 per sex) | Met |
| Published totals reproduced | The derived mirror's counts reproduce exactly from the original columns |
| Subsample cap | Not applied: 50,000 is below 60,000 |
| Known limits | Permission: no licence located; used under the amended rule of version 2 section 6.1 (organisers contacted by Andrei on 17 November 2026). No dates. Collected 2006 to 2009 from a private-label credit card operation. The modelling set holds the company's approved clients only; the competition's theme was exactly this sampling bias. No borrower identifier. |

## 2. Predictors kept (p = 29)

Nine numeric, one ordered and 19 unordered predictors, all recorded on the application. None is a lender's own grade, score or price.

| Predictor | Type | Distinct values | Missing |
| --- | --- | --- | --- |
| AGE | Numeric | 89 | 0 |
| QUANT_DEPENDANTS | Numeric | 17 | 0 |
| MONTHS_IN_RESIDENCE | Numeric | 88 plus NULL | 7.6% |
| PERSONAL_MONTHLY_INCOME | Numeric | 5,495 | 0 |
| OTHER_INCOMES | Numeric | 615 | 0 |
| PERSONAL_ASSETS_VALUE | Numeric | 153 | 0 |
| QUANT_CARS | Numeric | 2 | 0 |
| QUANT_BANKING_ACCOUNTS | Numeric | 3 | 0 |
| MONTHS_IN_THE_JOB | Numeric | 21 | 0 (99.8% are 0) |
| PAYMENT_DAY | Ordered: 1 < 5 < 10 < 15 < 20 < 25; no special codes | 6 | 0 |
| APPLICATION_SUBMISSION_TYPE | Unordered | 3 (0, Carga, Web) | 0 |
| POSTAL_ADDRESS_TYPE | Unordered | 2 | 0 |
| MARITAL_STATUS | Unordered | 8 (0 to 7) | 0 |
| STATE_OF_BIRTH | Unordered | 28 plus blank | 4.1% |
| NACIONALITY | Unordered | 3 | 0 |
| RESIDENCIAL_STATE | Unordered | 27 | 0 |
| FLAG_RESIDENCIAL_PHONE | Unordered | 2 | 0 |
| RESIDENCE_TYPE | Unordered | 6 (0 to 5) plus NULL | 2.7% |
| FLAG_EMAIL | Unordered | 2 | 0 |
| FLAG_VISA | Unordered | 2 | 0 |
| FLAG_MASTERCARD | Unordered | 2 | 0 |
| FLAG_DINERS | Unordered | 2 | 0 |
| FLAG_AMERICAN_EXPRESS | Unordered | 2 | 0 |
| FLAG_OTHER_CARDS | Unordered | 2 | 0 |
| COMPANY | Unordered | 2 | 0 |
| FLAG_PROFESSIONAL_PHONE | Unordered | 2 | 0 |
| PROFESSION_CODE | Unordered | 19 plus NULL | 15.5% |
| OCCUPATION_TYPE | Unordered | 6 (0 to 5) plus NULL | 14.6% |
| PRODUCT | Unordered | 3 (1, 2, 7) | 0 |

**Type choice to confirm.** PAYMENT_DAY is typed ordered because the days have a natural order but are a chosen option, not a measured amount. The second reviewer should confirm or argue for numeric.

## 3. Fields excluded (25)

| Field | Rule applied (template section 4) |
| --- | --- |
| ID_CLIENT | Identifier |
| SEX | Recorded sex |
| TARGET_LABEL_BAD=1 | Outcome |
| CLERK_TYPE, QUANT_ADDITIONAL_CARDS, EDUCATION_LEVEL, FLAG_MOBILE_PHONE, FLAG_HOME_ADDRESS_DOCUMENT, FLAG_RG, FLAG_CPF, FLAG_INCOME_PROOF, FLAG_ACSP_RECORD | Single value in every record |
| CITY_OF_BIRTH (9,910 values), RESIDENCIAL_CITY (3,529), RESIDENCIAL_BOROUGH (14,511) | Unordered with more than 50 levels |
| RESIDENCIAL_PHONE_AREA_CODE (102 values), RESIDENCIAL_ZIP_3 (794), PROFESSIONAL_ZIP_3 (794) | Unordered with more than 50 levels |
| PROFESSIONAL_STATE (68.6% missing), PROFESSIONAL_CITY (68.3%), PROFESSIONAL_BOROUGH (69.5%), PROFESSIONAL_PHONE_AREA_CODE (73.1%), MATE_PROFESSION_CODE (57.8%), MATE_EDUCATION_LEVEL (64.7%) | Missing for more than half the cohort |
| QUANT_SPECIAL_BANKING_ACCOUNTS | Identical in every record to QUANT_BANKING_ACCOUNTS (duplicate rule, added 16 November) |

No field was excluded as measured after the application, derived from the outcome, or a lender's own grade. None appears in the variable list.

## 4. Missing values, unusual codes, sex, grouping and design

**Missing values.** NULL and blank are both treated as missing. MONTHS_IN_RESIDENCE, the only numeric predictor with missing values, takes the fitting median of each repeat. In the five unordered predictors with missing values, missing becomes its own level. The variable list says "NULL = 0" for RESIDENCE_TYPE and OCCUPATION_TYPE, but both fields contain 0 *and* NULL as separate values. So NULL is kept as missing and 0 as a level of its own; nothing is merged.

**Codes outside the variable list, kept as their own levels.** APPLICATION_SUBMISSION_TYPE has a third code, 0 (19,461 records), beside Web and Carga. MARITAL_STATUS has a code 0 (202 records) beside the documented 1 to 7. STATE_OF_BIRTH has XX as well as blank.

**Implausible values, kept unchanged.** 21 ages below 18 (minimum 6) and 3 above 100. Personal monthly income up to R$959,000 against a median of R$500. The template allows no outlier handling.

**Duplicate field: decided.** QUANT_SPECIAL_BANKING_ACCOUNTS equals QUANT_BANKING_ACCOUNTS in all 50,000 records. It is excluded under the duplicate rule Andrei added to the template on 16 November. A check of every pair of fields found no other identical pair and no pair carrying the same information under different labels, apart from RESIDENCIAL_ZIP_3 and PROFESSIONAL_ZIP_3, which are excluded anyway. Dropping it leaves the groups unchanged.

| Field | Entry |
| --- | --- |
| Sex mapping | F = female, M = male (variable list: "M=Male (0), F=Female (1)"). N (48 records) and blank (17) = unknown. |
| Unknown sex | 65 records, 0.13%. Kept in every overall result; left out of the female and male results and of the fairness check. |
| Grouping unit | Records identical on all 29 predictors. 49,993 groups: 49,986 of one record and 7 of two. One group contains both a woman and a man. |
| Repeat borrowers possible without an identifier | Unknown |
| Dates | None. Later-period judge: no. |
| Primary design | Template section 6, unchanged: 50 repeats, seed 20261114 plus the repeat index, judge 20%, selection 16%, fitting the rest |

## 5. The judge's three families

The variable list does not group its fields, so the template's fixed category list applies. This is the first reviewer's classification, made from the variable list alone. A second reviewer classifies independently, and Andrei settles any disagreement before hashing.

| Category | Predictors | Count | Family? |
| --- | --- | --- | --- |
| Income and employment | PERSONAL_MONTHLY_INCOME, OTHER_INCOMES, PERSONAL_ASSETS_VALUE, QUANT_CARS, COMPANY, MONTHS_IN_THE_JOB, PROFESSION_CODE, OCCUPATION_TYPE | 8 | Family 1 |
| Residence and contact | RESIDENCIAL_STATE, RESIDENCE_TYPE, MONTHS_IN_RESIDENCE, FLAG_RESIDENCIAL_PHONE, POSTAL_ADDRESS_TYPE, FLAG_EMAIL, FLAG_PROFESSIONAL_PHONE | 7 | Family 2 |
| Existing credit and past repayment | FLAG_VISA, FLAG_MASTERCARD, FLAG_DINERS, FLAG_AMERICAN_EXPRESS, FLAG_OTHER_CARDS, QUANT_BANKING_ACCOUNTS | 6 | Family 3 |
| Applicant characteristics | AGE, QUANT_DEPENDANTS, MARITAL_STATUS, STATE_OF_BIRTH, NACIONALITY | 5 | No |
| Requested credit and product terms | PAYMENT_DAY, PRODUCT, APPLICATION_SUBMISSION_TYPE | 3 | No |
| Verification and documents | None: all such fields are constant | 0 | No |

The fallback does not apply: three categories hold two or more predictors, and the largest family holds 8 of 29, under half.

**Close calls (first reviewer).** FLAG_PROFESSIONAL_PHONE (contact, or employment?). QUANT_CARS and PERSONAL_ASSETS_VALUE (income and wealth, or applicant characteristics?). The banking-account count (existing credit, or other?). Moving FLAG_PROFESSIONAL_PHONE to employment would make them 9, 6 and 6. With the duplicate dropped, existing credit holds 6 and is still the third family, ahead of applicant characteristics at 5.

**Settled by Andrei, 17 November 2026.** The second reviewer (Astra, blind) placed PERSONAL_ASSETS_VALUE, QUANT_CARS and APPLICATION_SUBMISSION_TYPE under "other" and agreed that PAYMENT_DAY is ordered. Andrei keeps PERSONAL_ASSETS_VALUE and QUANT_CARS under income and employment (wealth is part of capacity to pay; the category list has no wealth group). APPLICATION_SUBMISSION_TYPE does not affect the families. Families: income and employment (8), residence and contact (7), existing credit (6).

## 6. Preparation and registration record

- [x] **Andrei:** link to where the original file was downloaded: https://github.com/rishabhpahuja/Credit-Score-/blob/master/Dataset/PAKDD/PAKDD2010_Modeling_Data.txt
- [x] **Andrei:** written permission decision, in his own words (section 1).
- [x] **Andrei:** decide the duplicate-field rule and add it to the template before freezing. Decided: exclude duplicates, because SAFE relies on feature perturbation and removal.
- [x] **Andrei:** accept or change the remaining defaults in template section 12. All accepted.
- [x] **Sol:** organisers' outcome definition quoted; no prior SAFE-family use found; p = 29 gives 30 nodes per SAFE curve and 27,000 cells per candidate per aggregate.
- [x] **Second reviewer (Astra, blind):** classification done; agrees PAYMENT_DAY is ordered. Differences settled by Andrei on 17 November (see section 5).
- [x] Groups containing both outcomes: 0 of 49,993.
- [x] Andrei: organisers contacted on 17 November 2026 (version 2 section 6.1).
- [x] **Sol:** fixed, manually specified untuned configurations; source and limits recorded in section 1 (17 November 2026).
- [x] This form is included in the public checksum manifest under version 2 section 7.2.
