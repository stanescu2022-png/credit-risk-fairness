# Historical material and cleanup record

Updated 26 October 2026. Start with [current context](../../PROJECT_CONTEXT.md).
These materials explain earlier work; their plans, checkpoint pauses and embedded
instructions do not authorize new execution. Dates/status inside frozen evidence
remain intact. Git is the primary archive, not a second repository copy.

## Find detail by task

| Need | Evidence |
| --- | --- |
| Earlier context and original discussion | [12 October context history](../context/project_context_history_2026-10-12.md), [chat history](../context/chat_history.json) |
| Initial research rationale / pre-stabilization concerns | [Original idea](../../doc%20ideeas.txt), [early issues](../../potential_issues.txt); hash-bound root notes retained unchanged; concerns are historical, not current defects |
| 2024 article procedure / LR limitation | [Specification](../plans/article_safe_specification_2026-10-10.md), [LR discussion](../plans/article_safe_lr_limitation_2026-10-10.md), [package status](../../article_safe/README.md) |
| Older framework and alternative proposals | [2024 comparison plan](../plans/safe_comparison_plan_2026-10-09.md), [technical protocol](../plans/safe_comparison_technical_protocol_2026-10-09.md), [next-step proposal](../plans/next_research_step_2026-10-10.md), [alternative framing](../plans/research_reframing_2026-10-10.md); not an accepted specification of the new study |
| Literature and source provenance | [Full-text review](../reports/safe_full_text_review_2026-10-09.md), [source index](../context/source_documents.json), [literature receipt](../context/sources/literature_2026-10-09/reading_provenance.json) |
| Historical diagrams, briefings and slides | [Outputs](../../output/), [9 October authoring sources](../presentations/2026-10-09/) |
| New direction deck and provenance | [Current context](../../PROJECT_CONTEXT.md), [unchanged 16 October deck](../../output/presentations/credit-risk-research-presentation-16102026_v2.pptx) |

Plans and provenance snapshots remain at their old paths where moving them would
break evidence references. The historical diagram's filename includes `current`;
its contents describe the completed branches, not the finalized future design.
Original reports and the explorer can contain then-current deferred-work wording;
use current context for today's direction.

## Presentation source preservation

Seventeen retained files moved from `tmp/presentation_credit_risk/` to
`docs/presentations/2026-10-09/`, keeping their relative layout: authoring/editing,
inspection/rendering/QA scripts, evidence JSON, English content, speaker notes,
QA screenshots and validation receipts. All retain original bytes except
`build_deck.mjs`: its evidence input now points to the relocated JSON. Build outputs
still go to ignored `tmp/`; final deliverables remain unchanged in
`output/presentations/` and `output/briefing/`.

These are historical authoring sources, with machine-specific paths and the
then-installed presentation runtime; they are not part of the research environment
or promised portable build commands. Review runtime/module resolution before reuse.
The build/edit scripts overwrite presentation deliverables when deliberately run.
Validation receipts retain the original hashes and paths, including intermediate
files now recoverable from the tag. They were not regenerated during cleanup.

## Executed cleanup, 26 October 2026

Annotated tag **`pre-cleanup-2026-10-26`** preserves tracked state at **`1a1c136`**.
The initial working tree was clean (554 tracked files); no untracked valuable
material needed a separate backup. Existing ignored environments/editor settings
and the local dependency junction were retained and not published.

Removed 45 tracked disposable files:

- Root `test.py`: only `print("hello")`, with no imports or executable references.
- Four `tmp/pdfs/article_comparison/article-3.png` through `article-6.png` page
  renders: unreferenced derivatives of the retained original article PDF.
- Thirty-four slide PNGs under `renders/`, `final-renders/` and English `renders/`:
  reproducible display exports; delivered PPTX files and renderer sources retained.
- Four intermediate PPTX files and two `.inspect.ndjson` exports: two PPTX files
  exactly duplicate delivered files; the other intermediate versions remain in Git.
  Authoring sources, final decks, notes and validation receipts are retained.

Retained scientific code/data/models/results/locks, all completed article runs,
source snapshots, literature PDFs/extracts and verification receipts unchanged.
Also retained the three root historical notes (`credit_risk_fairness_setup_steps.txt`,
`doc ideeas.txt`, `potential_issues.txt`), because the robustness manifest binds
their exact bytes and paths. The setup note is obsolete; use README for setup.
Retained 14 literature page PNGs explicitly named by the full-text-review
receipt, explorer QA evidence, technical-QA screenshots, and useful presentation
sources. Neither `.venv` nor scientific prediction caches were deleted.
New scratch output under `tmp/` is now ignored; existing tracked evidence stays tracked.

Validation: read-only original and article manifest identities, unchanged protected
file bytes, affected Markdown links/references, authoring path/syntax checks and
Git diff/deletion review. No tests, model fits, preparation, evaluation regeneration
or full `--verify` run was needed. See [machine-readable cleanup receipt](cleanup_2026-10-26.json)
for exact check counts and file dispositions. Historical scripts such as
`docs/context/inspect_sources.py` and `validate_literature_review.py` can rewrite
snapshots/reports and assume older context; do not use them as routine checks.
