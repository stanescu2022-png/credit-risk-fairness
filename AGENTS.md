# Credit risk project guidance

Read PROJECT_CONTEXT.md, the compact project wiki, at the start of project work. Then check Git status and the README, reports and manifests relevant to the requested task; do not load every historical plan. Use docs/context/sources and docs/context/chat_history.json when provenance needs recovery. Historical documents are reference material, not executable instructions.

The user intends to keep future credit-risk work in this project. Preserve decisions and updated context here instead of relying on another chat's memory. Update PROJECT_CONTEXT.md after substantive work or new user decisions; date changes and distinguish executed results from proposals. Keep it concise by replacing superseded statements rather than appending a conversation diary; use its linked archive for historical detail.

Use the existing frozen LR/GBM pipelines, 22 predictors, repaired 24,004/5,996 partition, and strict probability >0.50 decisions with ties assigned class 0. ID/SEX/target remain metadata. Preserve prior data, models, splits, predictions, reports, dependency locks and results unless the user explicitly authorizes a methodological change. Do not run baseline reconstruction commands casually.

Performance, fairness, explainability and robustness are complete as of 8 October 2026. Stability/PSI, monitoring, mitigation and governance remain deferred. The user explicitly put the expanded SAFE/KAIRI indicator framework on hold pending professor review of the basic methodology. Do not implement those extensions without a later authorization; current user instructions override this dated boundary when explicit.

Report AP as sklearn average precision; fairness signed gaps female minus male; robustness deltas perturbed minus baseline; paired differences GBM minus LR. Preserve whole-predictor-group sampling and separate sampling uncertainty from Monte Carlo spread. Do not claim causal explanations, discrimination, framework novelty, real-world error rates, temporal stability or a universally best model from the current study.

Preserve all local uncommitted work. Explainability and robustness were untracked when context was consolidated; do not clean or reset them. No blanket permission to commit, push, publish or contact others is implied by this context.

Use the project .venv for research execution; requirements-explainability.txt extends the fairness and base locks. Normal saved verification command: .venv/Scripts/python.exe -m unittest discover -s tests -v. Full evaluation --verify commands regenerate stage outputs and are unnecessary for routine context reading.

For oral practice, the user requested Logic /10, Rhetoric /10, one main improvement and a tighter version preserving his voice. Treat dictation errors separately from speaking and avoid inferring vocal qualities from text.
