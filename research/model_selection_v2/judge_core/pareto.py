"""Explicit, unrounded Pareto comparisons; no scalar judge or primary metric."""
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .metrics import CandidateTables, candidate_metrics
from .tables import nonempty_label, require, validate_alignment


def validate_directions(directions):
    require(isinstance(directions, Mapping) and bool(directions), "Supply an explicit nonempty metric-directions mapping.")
    result = dict(directions)
    for name, direction in result.items():
        nonempty_label(name, "metric name")
        require(direction in ("min", "max"), "Metric direction must be 'min' or 'max'.")
    return result


@dataclass(frozen=True)
class JudgeConfig:
    metric_directions: Mapping
    our_choice_id: str
    safe_choice_id: str
    candidate_order: tuple
    role_label: str
    primary_evaluation: None

    def __post_init__(self):
        require(self.primary_evaluation is None, "Primary evaluation remains unspecified; explicitly supply None.")
        nonempty_label(self.role_label, "role_label")
        order = tuple(self.candidate_order)
        require(bool(order) and len(set(order)) == len(order), "Candidate order must be explicit, nonempty and unique.")
        for candidate in order:
            nonempty_label(candidate, "candidate ID")
        require(self.our_choice_id in order and self.safe_choice_id in order, "Both explicit selected IDs must be in candidate_order.")
        object.__setattr__(self, "candidate_order", order)
        object.__setattr__(self, "metric_directions", MappingProxyType(validate_directions(self.metric_directions)))


def coordinate_values(vector, directions):
    require(isinstance(vector, Mapping), "Each metric vector must be a mapping.")
    values = {}
    for name in directions:
        value = vector.get(name)
        if value is not None:
            require(np.isscalar(value) and not isinstance(value, (str, bytes, bool)), "Metric values must be numeric scalars or None.")
            try:
                value = float(value)
            except (ValueError, TypeError) as error:
                raise ValueError("Metric values must be numeric scalars or None.") from error
            require(np.isfinite(value), "Missing metrics must be explicit None, not NaN/inf.")
        values[name] = value
    return values


def pareto_relation(left, right, *, metric_directions):
    """Exact stored-value comparisons; None/absent coordinates are unresolved.

    Equality uses exact float64 equality. No rounding, tolerance, rescaling,
    normalization or practical-equivalence margin is imposed.
    """
    directions = validate_directions(metric_directions)
    L, R = coordinate_values(left, directions), coordinate_values(right, directions)
    missing = {name: [side for side, values in (("left", L), ("right", R)) if values[name] is None]
               for name in directions if L[name] is None or R[name] is None}
    differences = {name: L[name]-R[name] if L[name] is not None and R[name] is not None else None for name in directions}
    if missing:
        status = "unresolved"
    else:
        left_better = [L[name] < R[name] if direction == "min" else L[name] > R[name]
                       for name, direction in directions.items()]
        right_better = [R[name] < L[name] if direction == "min" else R[name] > L[name]
                        for name, direction in directions.items()]
        if not any(left_better) and not any(right_better):
            status = "equal"
        elif any(left_better) and not any(right_better):
            status = "left_dominates"
        elif any(right_better) and not any(left_better):
            status = "right_dominates"
        else:
            status = "incomparable"
    return {"status": status, "left": L, "right": R, "left_minus_right": differences,
            "missing_coordinates": missing, "comparison": "exact unrounded coordinate values"}


def compare_choices(vectors, config):
    require(config.our_choice_id in vectors and config.safe_choice_id in vectors,
            "Both selected candidates must have explicit supplied vectors.")
    result = pareto_relation(vectors[config.our_choice_id], vectors[config.safe_choice_id],
                             metric_directions=config.metric_directions)
    translated = {"left_dominates": "our_choice_dominates", "right_dominates": "SAFE_choice_dominates"}
    same_choice = config.our_choice_id == config.safe_choice_id
    return {"status": "same_choice" if same_choice else translated.get(result["status"], result["status"]),
            "same_choice": same_choice, "pareto_status": result["status"],
            "our_choice_id": config.our_choice_id, "SAFE_choice_id": config.safe_choice_id,
            "difference_name": "our_choice_minus_SAFE_choice", "differences": result["left_minus_right"],
            "our_choice_vector": result["left"], "SAFE_choice_vector": result["right"],
            "missing_coordinates": result["missing_coordinates"]}


def pool_frontier(vectors, config):
    require(isinstance(vectors, Mapping), "Candidate vectors must be a mapping.")
    require(set(vectors) <= set(config.candidate_order), "Supplied candidate absent from explicit candidate_order.")
    unavailable = [name for name in config.candidate_order if name not in vectors]
    values = {name: coordinate_values(vectors[name], config.metric_directions)
              for name in config.candidate_order if name in vectors}
    unresolved = [name for name in values if any(v is None for v in values[name].values())]
    complete = [name for name in values if name not in unresolved]
    dominated = {}
    for name in complete:
        dominated[name] = [other for other in complete if other != name and
                           pareto_relation(values[other], values[name], metric_directions=config.metric_directions)["status"] == "left_dominates"]
    incomplete = bool(unavailable or unresolved)
    return {"frontier_ids": [name for name in complete if not dominated[name]],
            "dominated_by": dominated, "incomplete_pool": incomplete,
            "frontier_scope": "complete_observed_subset" if incomplete else "full_pool",
            "unavailable_candidate_ids": unavailable, "unresolved_candidate_ids": unresolved,
            "warning": ("Frontier is conditional on complete supplied vectors; unavailable or unresolved candidates may change it."
                        if incomplete else None)}


def validate_candidates(candidates, config):
    require(isinstance(candidates, Mapping) and bool(candidates), "Supply fixed candidate prediction tables.")
    require(set(candidates) <= set(config.candidate_order), "Candidate not in explicit candidate_order.")
    require(config.our_choice_id in candidates and config.safe_choice_id in candidates, "Selected candidate tables must be supplied.")
    require(all(isinstance(value, CandidateTables) for value in candidates.values()), "Expected CandidateTables objects.")
    reference = validate_alignment([value.clean for value in candidates.values()], role_label=config.role_label)
    repetition_counts = {}
    for value in candidates.values():
        for scenario, tables in value.scenarios.items():
            validate_alignment((reference, *tables), role_label=config.role_label)
            if scenario in repetition_counts:
                require(repetition_counts[scenario] == len(tables), "Scenario repetition counts differ across candidates.")
            repetition_counts[scenario] = len(tables)
    return reference


def evaluate_candidates(candidates, config, *, multiplicities=None):
    """Assess only supplied tables; directions choose coordinates, not this core."""
    validate_candidates(candidates, config)
    vectors = {name: candidate_metrics(candidates[name], multiplicities=multiplicities)
               for name in config.candidate_order if name in candidates}
    return {"role_label": config.role_label, "primary_evaluation": None,
            "metric_directions": dict(config.metric_directions), "candidate_metrics": vectors,
            "choice_comparison": compare_choices(vectors, config),
            "pool_frontier": pool_frontier(vectors, config),
            "scope": "Fixed supplied predictions; no selection, fitting, calibration or data access"}
