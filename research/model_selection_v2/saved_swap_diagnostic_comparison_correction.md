# Saved-swap diagnostic: zero-severity comparison correction

Recorded **30 October 2026**, after the first diagnostic output. The original
diagnostic source and completed run remain unchanged. Its probability losses,
grid means and scikit-learn checks are valid. Its pair-order counts used raw
floating-point RGR comparisons at zero severity, where the mathematical score
is one for every nonconstant self-comparison. Stored values range from
.9999999999999956 to 1.0, so roundoff incorrectly creates a strict preference.

Correct only the descriptive pair-order labels: zero severity is a known
self-comparison tie for every candidate pair. Preserve raw saved RGR values,
all 23 nodes, all losses and all other exact comparisons. Do not introduce a
general numerical dominance tolerance or alter any selector/judge. This is a
structural correctness fix applied to all six pairs, not outcome-based node
selection. Record the corrected counts separately with input/source hashes.
