"""Explicit original-predictor types; no metadata may enter the models."""
from dataclasses import dataclass, field
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from research.model_selection_v2 import pilot, safe2026


@dataclass(frozen=True)
class Schema:
    order: tuple
    numeric: tuple
    ordered: dict
    unordered: tuple
    special: dict = field(default_factory=dict)

    def __post_init__(self):
        names = self.numeric + tuple(self.ordered) + self.unordered
        if len(names) != len(set(names)) or set(names) != set(self.order) or len(self.order) != len(set(self.order)):
            raise ValueError('Types must partition the original predictor list exactly')
        if not self.order or not set(self.special) <= set(self.ordered):
            raise ValueError('Empty schema or special codes outside ordered fields')
        for n, levels in self.ordered.items():
            if len(levels) < 2 or len(levels) != len(set(levels)) or set(levels) & set(self.special.get(n, ())):
                raise ValueError('Invalid ordered levels or special-code overlap')

    @property
    def categorical(self):
        return tuple(self.ordered) + self.unordered

    def validate(self, frame):
        if list(frame.columns) != list(self.order) or not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError('Predictor order or finite-input gate failed')


def build_candidate(candidate, schema):
    # Reuse the frozen estimator configurations; replace only field lists.
    model = pilot.build_candidate(candidate)
    model.set_params(preprocessor=ColumnTransformer([
        ('num', StandardScaler(), list(schema.numeric)),
        ('cat', OneHotEncoder(handle_unknown='ignore'), list(schema.categorical))]))
    return model


def fitting_references(fitting, schema):
    schema.validate(fitting)
    return {n: float(fitting[n].mean()) if n in schema.numeric else float(fitting[n].mode().min())
            for n in schema.order}


def full_grid(schema):
    return np.linspace(0, 1, len(schema.order)+1)*.5


def accuracy_curve(outcome, probability, schema):
    return safe2026.accuracy_curve(outcome, probability, segments=len(schema.order))
