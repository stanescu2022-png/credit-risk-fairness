# Credit Risk Evidence Explorer

Open credit-risk-explorer.html in Edge or Chrome. Copy that single file to another laptop for offline use; no server, account, installation or internet connection is required. The interface is in English. All charts, scripts and display data are embedded.

## Views

- Performance: locked holdout metrics and confidence intervals; training CV means and SD; holdout confusion counts and calibration.
- Fairness: signed female-minus-male gaps, confidence intervals and fixed-threshold SEX group results.
- Explainability: model-specific top-ten SHAP shares, saved individual examples, descriptive SEX importance differences and grouped authentic SAFE RGE.
- Robustness: 166 canonical scenarios across both corruption layers, input scopes, severity levels and SEX groups. RGR is shown overall only, as evaluated. Sampling intervals and paired differences are available only where prespecified. A secondary row-order comparison is available under the primary joint SAFE scenario.
- Study & sources: protocol, limitations, research status and the 28-artifact embedded source registry.

pp means percentage points. Percentage-point differences are not relative percentage changes. Downloads retain unrounded source estimates, with numeric rates expressed as proportions. The download button exports the active comparison table; supporting calibration and confusion-count displays are not part of that export. Local SHAP case exports include all displayed contributions.

The file displays saved evidence; it does not score new clients, fit models, change thresholds, execute corruptions or rerun research. Synthetic perturbation outcomes are not causal counterfactuals. No overall trustworthiness score is calculated. The expanded framework remains deferred.

## Rebuild from current saved outputs

From the repository root:

    .venv/Scripts/python.exe explorer/build_explorer.py

The builder uses Python's standard library, reads saved CSV/protocol/verification files, and writes only the explorer HTML and source_manifest.json. template.html and app.js are the editable interface source. Refresh the packaged file deliberately when research outputs change; it does not update itself.

## Verification

The final HTML was opened directly through file:// in headless Microsoft Edge. Checks covered navigation, representative selectors, all 166 canonical scenario views, saved headline values, a CSV download, 390-pixel layout and enlarged browser text. No JavaScript errors or HTTP/network requests occurred. The original 410 protected/input/source/output hash checks remained unchanged, and all embedded source fingerprints matched the files used.

QA scripts and screenshots are under qa/. They are development records and are not needed on the presentation laptop. No research tests or full evaluation stages were rerun for this display-only addition.