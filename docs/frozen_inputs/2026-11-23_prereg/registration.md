# Brazil and synthetic study registration, 23 November 2026

Brazil runs first. Andrei launches both studies in his local PowerShell session. The public GitHub registration contains the final documents, implementation and [checksum manifest](../../../research/model_selection_v2/prereg_v2_manifest.json). The launcher verifies the tag `prereg-v2-2026-11`, checks the registered inputs and records the commit for the run. The manifest also binds the local timing and calibration evidence.

## Registered documents

- [Step 1 specification](step1_specification.md) and [dataset addendum template](dataset_addendum_template.md)
- [Version 2 amendments and change log](version2_amendments.md) and [section 6.5](section_6_5.md)
- [Completed Brazil dataset addendum](brazil_addendum_form_v2_CANDIDATE_ORIGIN_COMPLETED.md)
- [Synthetic protocol](protocol.md) and [generator specification](generators_spec.md)

The version 2 amendments govern the application of the earlier Step 1 specification and template to Brazil. This registration and the lean synthetic protocol establish the final simulation design. The four-strength F study, L0–L3 settings and nested interval-coverage experiment are deferred.

## Production design

Brazil is the sole fresh real-data confirmation dataset. The completed Brazil form, Step 1 specification, dataset template, version 2 amendments and section 6.5 govern the analysis. Use all 50,000 applicants, 29 predictors, four fixed candidates, 50 whole-group splits numbered 0–49, the settled 8/7/6 judge families and the 2,000-draw selection bootstrap. Job A fits, selects and evaluates J1. Job B reloads the saved models, reproduces J1 exactly, and supplies J2, J3, rank endpoints and subgroup audits. Collect section 6.5 for each job, including direct small-swap Brier. Use three worker processes with two numerical threads each. Each job retains its own 12-hour computation limit. Brazil files and individual-level outputs remain local.

The synthetic protocol and generator specification govern the simulation with this amendment to the production count. Use exactly 150 independent R populations numbered 0–149 and 300 independent F populations numbered 0–299, each of 30,000 borrowers. F's female debt coefficient is 0.5. Retain the separate 1,000-trial equal-error control, all seeds, candidate settings, judge levels, comparisons and uncertainty rules. The original blinded timing benchmark selected 50 under its original ceiling. On 22 November Andrei increased the planned counts to 150 each. On 23 November he fixed 150 R and 300 F before production, to obtain more opportunities to observe harmful proposals in F. Use actual eligible counts for conditional rates. Replay the report from the saved repeat files. The combined synthetic projection is 4 h 06 min for production, about 5 h including the completed timing run and reserve.
