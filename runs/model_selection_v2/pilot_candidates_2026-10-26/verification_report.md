# Fixed pilot candidates — 26 October 2026

Exploratory selection diagnostics, conditional on four fixed, untuned configurations. No overall winner or universal algorithm-family superiority is established.

| candidate_id | roc_auc | average_precision | brier | log_loss | accuracy | balanced_accuracy | precision | recall | f1 | tn | fp | fn | tp | fitting_time_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic | 0.773975 | 0.538336 | 0.136260 | 0.435218 | 0.822537 | 0.658153 | 0.688057 | 0.363123 | 0.475369 | 3563 | 175 | 677 | 386 | 0.226788 |
| decision_tree | 0.731700 | 0.485341 | 0.140238 | 0.448615 | 0.818579 | 0.647533 | 0.680451 | 0.340546 | 0.453918 | 3568 | 170 | 701 | 362 | 0.157178 |
| random_forest | 0.784374 | 0.549470 | 0.135345 | 0.428699 | 0.818579 | 0.652582 | 0.670819 | 0.354657 | 0.464000 | 3553 | 185 | 686 | 377 | 32.159101 |
| gradient_boosting | 0.786074 | 0.554548 | 0.134388 | 0.427711 | 0.823787 | 0.660639 | 0.692035 | 0.367827 | 0.480344 | 3564 | 174 | 672 | 391 | 6.671449 |

Exact resolved settings are in protocol.json: L2 LR C=1, lbfgs, max_iter=2000; DT depth=5, min_leaf=20; RF 300 trees, min_leaf=5, sqrt features, n_jobs=1; GBM 100 trees, rate=.1, depth=3. Seeds=42. Each has independent fitting-only numeric StandardScaler and categorical OneHotEncoder(handle_unknown=ignore).

Reused article_v1_2026-10-10/development_assignments.csv directly: 19,203 fitting, 4,801 selection; together the saved 24,004 training records. All 30,000 IDs, labels, row order and whole original 22-predictor groups verified against saved metadata. The historical 5,996 holdout is inspected exploratory evidence; only its identity, alignment and group separation were checked. No holdout predictions or metrics.

Cleaning audit: fixed target rename, EDUCATION 0/5/6→4, MARRIAGE 0→3, ID/missing validation. No learned cleaning statistics or mappings; in-memory raw cleaning equals saved processed data. No source conflict. Inputs and transformed matrices are finite.

Each diagnostics/<candidate>.json maps every original predictor to transformed column indices/names and one-hot category levels; it separately counts unseen selection levels for all eight categorical predictors. Numeric sample counts, means, variances and category sets match fitting records exclusively.

- logistic: classes=[0, 1]; class-1 column=1; n_iter_=[71]; warnings=[]; unseen=none.
- decision_tree: classes=[0, 1]; class-1 column=1; n_iter_=None; warnings=[]; unseen=none.
- random_forest: classes=[0, 1]; class-1 column=1; n_iter_=None; warnings=[]; unseen=none.
- gradient_boosting: classes=[0, 1]; class-1 column=1; n_iter_=None; warnings=[]; unseen=none.

0=nondefault; 1=default. Both estimator classes asserted, class-1 column looked up explicitly. Decisions use p(default)>.50, ties=0. Full pipeline fitting wall time includes preprocessing and estimator fit. Undefined metrics remain null/empty with denominator rules in protocol.json. AP is sklearn average precision.

Four saved/reloaded pipelines reproduce selection probabilities bit for bit; saved IDs/outcomes/SEX/groups and decisions match; metrics recomputed from saved predictions agree with JSON and CSV diagnostics. All 510 protected tracked files match pre-run hashes; the 410 historical manifest references passed before and after, without stage regeneration.

RUN_STARTED.json records inception as incomplete. Only VERIFIED.json with matching hashes establishes completion. Existing destinations and files are never overwritten; interrupted runs remain incomplete. No SAFE, feature ablation, fairness assessment, tuning, new judge or refit on combined fitting/selection data. Primary candidates are frozen. Future auxiliary feature-removal models must be separately fitted under a later RGE protocol and must never replace or modify these candidates. Next: discuss and specify SAFE 2026 implementation. Judge options remain unresolved.
