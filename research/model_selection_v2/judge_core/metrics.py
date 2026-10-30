"""Probability losses and descriptive threshold gaps on fixed supplied tables."""
from dataclasses import dataclass, field
from collections.abc import Mapping

import numpy as np

from .tables import (PredictionTable, nonempty_label, require, row_multiplicities,
                     validate_alignment)

LOGLOSS_EPSILON = np.finfo(np.float64).eps


def losses(table, *, multiplicities=None):
    """Mean Brier/unclipped p; LogLoss uses float64 eps endpoint clipping.

    Clip p to [eps,1-eps] for LogLoss only, with eps=2^-52. This policy is
    explicit, deterministic and shared across all candidates/scenarios.
    """
    require(isinstance(table, PredictionTable), "Expected a validated PredictionTable.")
    weights = row_multiplicities(table, multiplicities)
    p = table.probability
    q = np.clip(p, LOGLOSS_EPSILON, 1-LOGLOSS_EPSILON)
    y = table.outcome
    log_terms = -np.where(y == 1, np.log(q), np.log1p(-q))
    return {"brier": float(np.average((p-y)**2, weights=weights)),
            "log_loss": float(np.average(log_terms, weights=weights))}


def fairness_gaps(table, *, multiplicities=None):
    """Strict p>.5 decisions; signed gaps female minus male; null rates stay null."""
    weights = row_multiplicities(table, multiplicities)
    predicted = table.probability > .5
    rates = {}
    for name, sex_code in (("female", 2), ("male", 1)):
        sex_mask = table.sex == sex_code
        positives = sex_mask & (table.outcome == 1)
        negatives = sex_mask & (table.outcome == 0)
        tp = int(weights[positives & predicted].sum())
        fp = int(weights[negatives & predicted].sum())
        pos = int(weights[positives].sum())
        neg = int(weights[negatives].sum())
        rates[name] = {"tpr": tp/pos if pos else None, "fpr": fp/neg if neg else None,
                       "tp": tp, "fp": fp, "positive_denominator": pos, "negative_denominator": neg}
    result = {"rates": rates, "decision_rule": "p > .5; ties class 0", "signed_direction": "female minus male"}
    for metric in ("tpr", "fpr"):
        female, male = rates["female"][metric], rates["male"][metric]
        gap = female-male if female is not None and male is not None else None
        result[f"{metric}_gap_signed"] = gap
        result[f"{metric}_gap_absolute"] = abs(gap) if gap is not None else None
    return result


def scenario_losses(reference, scenarios, *, multiplicities=None):
    """Each scenario's arithmetic mean of repetition losses, never loss(mean p).

    Repetition order/identity and common perturbation inputs are a caller
    contract; aligned outputs cannot establish that the inputs were shared.
    Repetition spread remains separate from group-bootstrap uncertainty.
    """
    require(isinstance(scenarios, Mapping), "Scenarios must map names to nonempty repetition sequences.")
    result = {}
    for name, repetitions in scenarios.items():
        nonempty_label(name, "scenario name")
        require(name != "clean", "Scenario name 'clean' is reserved.")
        repetitions = tuple(repetitions)
        require(bool(repetitions), "Every supplied scenario needs at least one repetition.")
        validate_alignment((reference, *repetitions), role_label=reference.role_label)
        values = [losses(table, multiplicities=multiplicities) for table in repetitions]
        result[name] = {"brier": float(np.mean([v["brier"] for v in values])),
                        "log_loss": float(np.mean([v["log_loss"] for v in values])),
                        "repetition_count": len(values), "repetition_losses": values,
                        "aggregation": "arithmetic mean of per-repetition losses"}
    return result


@dataclass(frozen=True)
class CandidateTables:
    clean: PredictionTable
    scenarios: Mapping = field(default_factory=dict)

    def __post_init__(self):
        require(isinstance(self.clean, PredictionTable), "Candidate clean table must be validated.")
        require(isinstance(self.scenarios, Mapping), "Candidate scenarios must be a mapping.")
        # Copy sequence containers so later caller list edits cannot change a run.
        copied = {name: tuple(tables) for name, tables in self.scenarios.items()}
        for name, tables in copied.items():
            nonempty_label(name, "scenario name")
            require(name != "clean" and bool(tables), "Scenarios need a non-reserved name and nonempty repetitions.")
            validate_alignment((self.clean, *tables), role_label=self.clean.role_label)
        from types import MappingProxyType
        object.__setattr__(self, "scenarios", MappingProxyType(copied))


def candidate_metrics(candidate, *, multiplicities=None):
    """Return available diagnostics; caller explicitly chooses Pareto coordinates."""
    result = {f"clean.{name}": value for name, value in losses(candidate.clean, multiplicities=multiplicities).items()}
    fairness = fairness_gaps(candidate.clean, multiplicities=multiplicities)
    result.update({f"clean.{name}": fairness[name] for name in
                   ("tpr_gap_signed", "tpr_gap_absolute", "fpr_gap_signed", "fpr_gap_absolute")})
    for scenario, summary in scenario_losses(candidate.clean, candidate.scenarios, multiplicities=multiplicities).items():
        result.update({f"{scenario}.{metric}": summary[metric] for metric in ("brier", "log_loss")})
    return result
