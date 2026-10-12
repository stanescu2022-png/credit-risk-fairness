"""Authentic installed SAFE RGR, with a read-only, schema-preserving adapter."""
import hashlib
import inspect
from importlib.metadata import distribution, version
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from safeaipackage import check_robustness as safe, core
from safeaipackage.util import utils
from src.baseline import sha256
from src.safe_adapter import provenance as established_provenance

INSPECTED_ROBUSTNESS_HASH = '0bbb45f8de43a546732010e9f468c10d44c849a845f4cfb4397ae585a120c066'


class RobustnessAdapter(ClassifierMixin, BaseEstimator):
    """Only forward frozen predictors; capture the authentic API's actual input."""
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.classes_ = pipeline.classes_
        self.last_input = None
        self.last_probabilities = None
        self.calls = 0
        np.testing.assert_array_equal(self.classes_, [0, 1])

    def predict_proba(self, X):
        if list(X.columns) != list(self.pipeline.feature_names_in_):
            raise ValueError('Frozen feature order changed in SAFE')
        self.last_input = X.copy(deep=True)
        self.last_probabilities = self.pipeline.predict_proba(X.copy(deep=True))
        self.calls += 1
        return self.last_probabilities.copy()

    def fit(self, X, y=None):
        raise RuntimeError('Frozen models and adapters must never be fitted')


def source_identity():
    result = established_provenance()
    digest = sha256(safe.__file__)
    normalized = hashlib.sha256(Path(safe.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    if INSPECTED_ROBUSTNESS_HASH not in (digest, normalized):
        raise ValueError('SAFE robustness source differs from the locally inspected implementation')
    result.update({'robustness_source_hash': digest,
        'installed_direct_url_metadata': distribution('safeaipackage').read_text('direct_url.json'),
        'robustness_commit': None,
        'robustness_identity': 'Installed 0.8.3 module inspected locally and SHA-256 pinned; no commit equality claimed',
        'functions': ['check_robustness.perturb', 'check_robustness.compute_rgr_values', 'util.utils.find_yhat', 'core.rga'],
        'default_probability_column': 1, 'class_order': [0, 1]})
    return result


def inspection_text():
    functions = [safe.perturb, safe.compute_rgr_values, utils.convert_to_dataframe, utils.find_yhat, core.rga]
    return '\n\n'.join(inspect.getsource(f) for f in functions)


def tail_participants(X, feature, p):
    """Exposure bookkeeping only; perturbation itself always uses installed SAFE."""
    n = min(int(np.ceil(p * len(X))), len(X) - int(np.ceil((1-p) * len(X))))
    values = X[feature].to_numpy(); order = sorted(range(len(X)), key=values.__getitem__)
    selected = np.zeros(len(X), dtype=bool)
    if n:
        selected[order[:n] + order[-n:]] = True
    return selected


def transform(X, features, p):
    original = X.reset_index(drop=True).copy(deep=True)
    changed = original.copy(deep=True)
    selected = {}
    for feature in features:
        selected[feature] = tail_participants(original, feature, p)
        changed = safe.perturb(changed.copy(deep=True), feature, p)
        np.testing.assert_array_equal(np.sort(original[feature]), np.sort(changed[feature]))
    return changed, selected


def authentic_call(model, X, original_scores, features, p, group, expected):
    adapter = RobustnessAdapter(model)
    assert is_classifier(adapter)
    before = X.copy(deep=True)
    np.testing.assert_array_equal(model.classes_, [0, 1])
    adapter.calls = 0
    result = safe.compute_rgr_values(X.copy(deep=True), np.asarray(original_scores).tolist(),
                                    adapter, list(features), perturbation_percentage=p, group=group)
    pd.testing.assert_frame_equal(X, before, check_exact=True)
    pd.testing.assert_frame_equal(adapter.last_input, expected, check_exact=True)
    if adapter.calls != 1:
        raise AssertionError('Expected exactly one frozen prediction call per scenario')
    probabilities = adapter.last_probabilities[:, 1].copy()
    direct = core.rga(np.asarray(original_scores).tolist(), probabilities.tolist())
    value = float(result.RGR.iloc[0])
    np.testing.assert_allclose(value, direct, atol=1e-14, rtol=0, equal_nan=True)
    return value, probabilities


class FixtureClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, constant=False):
        self.constant = constant
    def predict_proba(self, X):
        p = np.full(len(X), .4) if self.constant else X.score.to_numpy()
        return np.column_stack([1-p, p])
    def fit(self, X, y=None):
        raise RuntimeError('Fixture is never fitted')


def validate_fixtures():
    fixtures = [
        ('identity', [.1,.2,.3,.4], [.1,.2,.3,.4], 1.),
        ('preserved_order', [.1,.2,.3,.4], [.2,.4,.6,.8], 1.),
        ('reversed_order', [.1,.2,.3,.4], [.8,.6,.4,.2], 0.),
        ('original_ties', [.1,.1,.3,.3], [.1,.2,.3,.4], 1.),
        ('perturbed_ties', [.1,.2,.3,.4], [.2,.2,.8,.8], .9),
        ('both_ties', [.1,.1,.3,.3], [.1,.1,.3,.3], 1.),
        ('constant_original', [.2]*4, [.1,.2,.3,.4], np.nan),
        ('constant_perturbed', [.1,.2,.3,.4], [.2]*4, .5)]
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for name, original, perturbed, expected in fixtures:
            actual = float(core.rga(original, perturbed))
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14, equal_nan=True)
            rows.append({'fixture': name, 'value': actual, 'expected': expected,
                         'status': 'undefined' if not np.isfinite(actual) else 'defined',
                         'reason': 'constant original scores: inc minus dec equals zero' if not np.isfinite(actual) else ''})
        for constant in (False, True):
            X = pd.DataFrame({'score': [.1,.2,.3,.4]}, index=[80,4,61,17])
            before = X.copy(deep=True)
            model = FixtureClassifier(constant)
            original = model.predict_proba(X)[:, 1]
            for p in (0., .25):
                expected_frame, _ = transform(X, ['score'], p)
                result = safe.compute_rgr_values(X.copy(deep=True), original.tolist(), model, ['score'], p, group=True)
                expected = core.rga(original.tolist(), model.predict_proba(expected_frame)[:, 1].tolist())
                np.testing.assert_allclose(result.RGR.iloc[0], expected, rtol=0, atol=1e-14, equal_nan=True)
                pd.testing.assert_frame_equal(X, before)
                rows.append({'fixture': f'API_index_reset_constant_{constant}_p_{p}',
                             'value': float(result.RGR.iloc[0]), 'expected': float(expected),
                             'status': 'undefined' if not np.isfinite(expected) else 'defined',
                             'reason': 'constant original scores: zero normalization' if not np.isfinite(expected) else ''})
        X = pd.DataFrame({'score': [.1,.2,.3,.4]})
        result = safe.compute_rgr_values(X, X.score.tolist(), FixtureClassifier(True), ['score'], .25, group=False)
        np.testing.assert_allclose(result.RGR.iloc[0], .5, rtol=0, atol=1e-14)
        rows.append({'fixture': 'API_constant_perturbed', 'value': .5, 'expected': .5, 'status': 'defined', 'reason': ''})
    return pd.DataFrame(rows)
