# Fresh dataset search — handoff to Sol, 16 November 2026

Andrei shortlisted two leads for Sol to check. Both are candidates for a fresh test of the Taiwan Step 1 comparison. Neither has been selected for modelling. The article question remains open.

The [Step 1 protocol, section 10](../../runs/model_selection_v2/step1_accuracy_anchored_taiwan_2026-11-14/protocol_extracted.txt) requires a public release permitting academic use and publication, observed repayment default, recorded sex for audit, and scoring-time tabular predictors. The eligible cohort needs at least 20,000 records, 2,000 defaults overall and 500 defaults for each sex. A dated cohort for a later-period evaluation is preferred. This search examined eligibility and counts only.

## 1. Brazil — PAKDD 2010 private-label credit cards

The labelled file covers 50,000 approved customers. Its event is a payment 60 days late on a bill in the first year after credit was granted. The competition used data collected in 2006–2009. The two later cohorts each have 20,000 records, but their public files lack outcomes. A later-period evaluation would need usable dates within the labelled file. Application fields include income, residence and employment information and existing cards. These are a mix of numerical and categorical fields. Prior credit-fairness research explicitly identifies the 2010 release. [Organizers](https://pakdd.org/archive/pakdd2010/PAKDDCompetition.html) · [Challenge data description, section 2](https://manuals.plus/m/c5c2d7df1305e4c55a8e976a8671967290fafa13d60408f11f7e570042052418.pdf) · [Prior fairness study repository](https://github.com/kozodoi/Fair_Credit_Scoring/blob/main/data/README.md)

| Evidence | Records | Defaults | How checked |
|---|---:|---:|---|
| All labelled customers | 50,000 | 13,041 | Counted all entries in a mirrored label file. Its 13,041 ones and 36,959 zeros match the published challenge totals. |
| Customers with recorded sex | 49,935 | 13,018 | Reported in a secondary study. |
| Women | 30,805 | 7,790 | Implied by the study's rounded percentages and its total defaults. |
| Men | 19,130 | 5,228 | Implied by the same figures. |

The two group counts are the only integer pair summing to 13,018 that reproduces 25.29% of 30,805 and 27.33% of 19,130 after rounding. The study links the 2010 competition but calls it “1999” in the text and reverses the original target coding. This makes a check against the original sex and target columns especially important. The label file used for the overall count has Git blob ID `480d49eceb3ce9b0c080be9bc4d276240df1f83d`. [Mirrored labels](https://github.com/rishabhpahuja/Credit-Score-/blob/master/Dataset/PAKDD/Data_labels.csv) · [Secondary study, section 5.1.3](https://er.ucu.edu.ua/bitstreams/a4cd87d4-76fc-47bb-8633-011b8fecb7b7/download)

**Sol's checks.** Locate terms from the organizer or rights holder that cover this academic study and publication of aggregate results. The archived competition announcement invites submissions but does not settle subsequent reuse. Pin the original modelling file and dictionary, reproduce the sex-by-outcome counts including unknown codes, and check the availability of dates and stable customer identifiers. Keep the 2009 competition and the Czech 1999 data separate from this release. [Organizer archive](https://pakdd.org/archive/pakdd2010/PAKDDCompetition.html) · [Teaching mirror of the original-format file](https://github.com/nickeubank/MIDS_Data/tree/master/PAKDD%202010)

## 2. China — PPDAI loan characteristics and repayment records

A published inspection reports 328,553 distinct loan IDs originated from January 2015 to January 2017. The LC table records sex and applicant and loan information available at origination, including amount, term, interest rate, prior borrowing and verification indicators. The companion LP repayment table is the lead for constructing a forward outcome. The overdue-payment count already in LC describes earlier borrowing. Using it as the outcome would answer the wrong question. Dates make this candidate attractive for testing on later loans. [Platform competition page](https://ai.ppdai.com/mirror/showCompetition) · [Published field and count inspection](https://developer.cloud.tencent.com.cn/article/1475073)

**Sol's checks.** Confirm the exact LC/LP release and its academic publication terms. Check the repayment dictionary, extract date, scheduled dates, actual payments and treatment of outstanding balances. Specify a serious-arrears threshold and follow-up window supported by those fields before counting eligible loans. Restrict the cohort to loans with sufficient follow-up, then give the record count and defaults overall and by recorded sex. Verify whether borrowers can recur across loan IDs and whether a borrower identifier supports grouping. Counts meeting the thresholds, prior SAFE use on this exact release and permission for this study are unresolved.

## Handoff decision

Sol can report **eligible**, **ineligible** or **unresolved** for each lead, with the source terms, release identifier, cohort definition, counts and outstanding evidence. The next scientific decision is the article question. Before fitting on an eligible dataset, settle that question and freeze the protocol's dataset-specific addendum covering predictors, sex coding, grouping, split and judge disturbances.
