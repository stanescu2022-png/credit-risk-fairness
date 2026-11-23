# Version 2 addendum, section 6.5. Direct small-swap Brier

**18 November 2026.** Andrei accepted this secondary benchmark before the Brazil run. Include this addendum and its matching implementation in the section 7.2 manifest for the public freeze. The [version 2 amendments](version2_amendments.md) govern the rest of the protocol.

## Selection rule

For each repeat, select the candidate with the highest saved selection-set S̄, the arithmetic mean of the six un-clipped Brier skills at tail-swap fractions 0 (clean), .01, .02, .03, .04 and .05. Selection prevalence is common to all candidates in a repeat. The exact ordering by S̄ is therefore the reverse of the ordering by mean Brier at those six equally weighted nodes. Apply the frozen tolerance of `1e-12` in S̄ units against the largest value, and resolve ties in candidate order: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting. This rule uses S̄ alone. It has no Ā, Ē or fairness gate.

The public arm label is **direct small-swap Brier** (`direct_small_swap_brier`). The October candidates, the registered SAFE and Selector 2 rules, the J1 judge, the paired range/measure contrasts and headline rules retain their specified definitions. The [Taiwan headroom analysis](../../../research/model_selection_v2/taiwan_selection_headroom_2026-11-18/report.md) explains why the benchmark was added. Its 49/50 choice agreement with Selector 2 is development evidence.

## Reporting

Use the existing candidate-level J1 primary endpoint for four additional paired rows in Job A and Job B.

| Contrast | Question |
| --- | --- |
| Direct small-swap Brier minus SAFE | How does the simpler stress-aware choice compare with published SAFE? |
| Final Selector 2 minus direct small-swap Brier | Does full Selector 2 add judged Brier value beyond choosing on S̄ alone? |
| Final Selector 2 minus plain clean Brier | What is the registered comparison against accuracy-only selection? |
| Direct small-swap Brier minus plain clean Brier | How does the stress-aware choice compare with clean-loss selection? |

Take paired differences across all 50 repeats, with exact zero on common choices. Use the approximate 95% corrected resampled t interval and realized role-size ratio already specified in Step 1 and the version 2 amendments. Export the estimate, interval, disagreements, signs of paired differences and clean Brier reference quantities in the same fields as the other Brier contrasts. Apply the version 2 all-zero interval of [0, 0]. The section 5 wording about a loss larger than 0.001 against plain loss still depends on the upper bound of the **Selector 2 minus plain clean Brier** interval being below +0.001. Lower Brier favours the first arm in a contrast. An interval spanning zero leaves the comparison unresolved.

Include the new arm in the per-repeat choice and all-registered-arm match tables. Job B also includes direct small-swap Brier minus SAFE for the existing secondary judge endpoints and the new choice in the existing clean/J1 sex and age audit. All comparisons, including any disagreeing results, are reported beside the existing comparisons. The primary contrast that governs the headline remains Selector 2 minus SAFE.

## Job A and Job B collection

Use the labelled section 6.5 collector on the completed repeat folders produced by the existing two-job execution path. From the project root, run the appropriate command with a fresh report destination. Each input root contains the 50 folders named `A_0` through `A_49` or `B_0` through `B_49`.

```powershell
.venv/Scripts/python.exe -B -S -m research.model_selection_v2.confirmation_v2_jobs.offline research.model_selection_v2.confirmation_v2_jobs.section_6_5 A <Job-A-repeat-root> <fresh-Job-A-report-folder>
.venv/Scripts/python.exe -B -S -m research.model_selection_v2.confirmation_v2_jobs.offline research.model_selection_v2.confirmation_v2_jobs.section_6_5 B <Job-B-repeat-root> <fresh-Job-B-report-folder>
```

Job A exports the added rows in `primary_contrasts.csv`. Job B exports them in `contrasts.csv` and the group summaries in `group_audit_intervals.csv`. Both collectors verify the completion receipts and handoff hashes. They derive the choice from the saved S̄ column, then read the already computed judge values for that candidate. They validate any pre-recorded direct choice against S̄. For a prospective worker that records the choice before J1, call `choose_direct_small_swap(rows.S_bar.to_dict())` immediately after computing the selection rows and carry the choice with the other choices into the per-repeat table. Freeze the code and choice rule before any fitting. Existing section 6.4 artifacts without that choice are supported by verified readback of their selection values. Job B uses the saved candidate-level audit, including when the direct arm chose a candidate no other arm selected.

## Verification

The unit fixtures exercise both complete 50-repeat table exports, the tie boundary, the all-agreement interval, receipt validation, and the group audit when a candidate is unique to the direct arm. The saved section 6.4 run 04 readback check verified all eight artificial-data repeat receipts, exact agreement of the primary columns and recovered direct choices across Jobs A and B, and all 192 direct-arm audit cells for each saved Job B repeat.

```powershell
.venv/Scripts/python.exe -B -S -m research.model_selection_v2.confirmation_v2_jobs.offline research.model_selection_v2.confirmation_v2_jobs.test_section_6_5
```

Verification used table fixtures and saved outputs. It required no new fitting, prediction or judge evaluation. The production Brazil entry point, final export and public timestamp remain separate.
