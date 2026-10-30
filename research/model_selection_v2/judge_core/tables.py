"""In-memory prediction contracts; this module performs no file or model access."""
from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty_label(value, name):
    require(isinstance(value, str) and bool(value.strip()), f"{name} must be an explicit nonempty string.")


def immutable_array(values, dtype):
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def identifiers(values, name, unique=False):
    result = tuple(values)
    for value in result:
        require(np.isscalar(value) and value is not None, f"{name} must contain nonmissing scalar identifiers.")
        require(not isinstance(value, (float, np.floating)) or np.isfinite(value), f"{name} contains missing/nonfinite identifiers.")
        require(not isinstance(value, str) or bool(value.strip()), f"{name} contains empty identifiers.")
        try:
            hash(value)
        except TypeError as error:
            raise ValueError(f"{name} identifiers must be hashable.") from error
    if unique:
        require(len(set(result)) == len(result), f"{name} must be unique.")
    return result


@dataclass(frozen=True)
class PredictionTable:
    """Copied, validated arrays in the supplied row order, for one explicit role.

    SEX uses the project's coding: 1 male, 2 female. Groups are the supplied
    whole-original-predictor groups; they may cross outcome and SEX strata.
    """
    ids: tuple
    outcome: np.ndarray
    sex: np.ndarray
    original_group: tuple
    probability: np.ndarray
    role_label: str

    def __post_init__(self):
        nonempty_label(self.role_label, "role_label")
        ids = identifiers(self.ids, "ID", unique=True)
        groups = identifiers(self.original_group, "original_group")
        y, sex, p = (np.asarray(values) for values in (self.outcome, self.sex, self.probability))
        require(len(ids) > 0, "Prediction tables must be nonempty.")
        require(all(v.ndim == 1 and len(v) == len(ids) for v in (y, sex, p)) and len(groups) == len(ids),
                "Prediction columns must be aligned one-dimensional vectors.")
        require(np.isin(y, [0, 1]).all(), "Outcome must contain only binary 0/1 values.")
        require(np.isin(sex, [1, 2]).all(), "SEX must contain only project codes 1/2.")
        try:
            p = np.asarray(p, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("Probability must be numeric float64 values.") from error
        require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Probabilities must be finite and in [0,1].")
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "original_group", groups)
        object.__setattr__(self, "outcome", immutable_array(y, np.int8))
        object.__setattr__(self, "sex", immutable_array(sex, np.int8))
        object.__setattr__(self, "probability", immutable_array(p, np.float64))

    @classmethod
    def from_columns(cls, columns, *, role_label, column_map=None):
        """Accept a DataFrame or mapping; source-column mapping is optional.

        Canonical keys are ID, outcome, SEX, original_group, probability. No
        row reordering, ID inference or role switching is performed.
        """
        names = {name: name for name in ("ID", "outcome", "SEX", "original_group", "probability")}
        if column_map is not None:
            require(isinstance(column_map, Mapping) and set(column_map) <= set(names), "Unknown column-map keys.")
            names.update(column_map)
        require(len(set(names.values())) == len(names), "Source column names must be distinct.")
        require(set(names.values()) <= set(columns), "Missing required prediction-table columns.")
        return cls(tuple(columns[names["ID"]]), np.asarray(columns[names["outcome"]]),
                   np.asarray(columns[names["SEX"]]), tuple(columns[names["original_group"]]),
                   np.asarray(columns[names["probability"]]), role_label)


def validate_alignment(tables, *, role_label):
    """Require exact IDs, metadata and role in supplied order; return reference."""
    nonempty_label(role_label, "role_label")
    tables = tuple(tables)
    require(bool(tables), "At least one prediction table is required.")
    reference = tables[0]
    require(isinstance(reference, PredictionTable), "Expected validated PredictionTable objects.")
    for table in tables:
        require(isinstance(table, PredictionTable), "Expected validated PredictionTable objects.")
        require(table.role_label == role_label, "Prediction data-role mismatch; roles cannot switch implicitly.")
        require(table.ids == reference.ids, "Prediction IDs/order mismatch.")
        require(np.array_equal(table.outcome, reference.outcome), "Outcome alignment mismatch.")
        require(np.array_equal(table.sex, reference.sex), "SEX alignment mismatch.")
        require(table.original_group == reference.original_group, "Original-group alignment mismatch.")
    return reference


def row_multiplicities(table, multiplicities=None):
    """Integer record counts, not normalization weights or per-group averages."""
    if multiplicities is None:
        return np.ones(len(table.ids), dtype=np.int64)
    weights = np.asarray(multiplicities)
    require(weights.ndim == 1 and len(weights) == len(table.ids), "Row multiplicity alignment mismatch.")
    require(weights.dtype.kind in "iu" and (weights >= 0).all(), "Row multiplicities must be nonnegative integers.")
    require(weights.sum() > 0, "Row multiplicities must retain at least one record.")
    return weights
