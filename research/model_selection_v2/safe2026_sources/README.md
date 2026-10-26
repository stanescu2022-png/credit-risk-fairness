# Inspected source evidence, not an installed dependency

Downloaded 26 October 2026 from the version-pinned
[PyPI release](https://pypi.org/project/safe-ai-metrics/0.2.0/).
The distribution credits Vasily Kolesnikov and was released 10 June 2026.
Its identity does not establish that it generated the published paper's tables.

Wheel SHA-256:
`454d724b448a941345b94016d90e048b7d7c15fda0f18feb48434832ac9e6303`.
The wheel is unmodified; the relevant numerical modules and metadata are copied
verbatim for inspection. The MIT license is preserved alongside them and inside
the wheel. Nothing was installed, and research dependency locks are unchanged.

Do not use these modules as the pilot implementation. The separate
`safe2026.py` follows the published equations with the user-approved Taiwan
operational choices documented in `../safe2026_specification.md`. Source audit
counterexamples are saved in each SAFE run's `source_inspection.json`. Only a
small list of inspected pure numerical function definitions is evaluated in
isolation for these counterexamples; no package or heavyweight dependencies
are imported.
