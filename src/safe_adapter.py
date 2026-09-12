"""Use actual SAFE 0.8.3 code with frozen sklearn pipelines and audit metadata."""
from importlib.metadata import version
from pathlib import Path
import re

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator, is_classifier
from safeaipackage import core, check_fairness
from safeaipackage.util import utils

from src.baseline import FEATURES, sha256

INSPECTED_COMMIT = "1e5f2a84d183c741fecaa682f2b9d496c142bd45"
EXPECTED_SOURCE_HASHES = {
    "core.py": "01dd57608a959cebaedfc25a6dbaaa0aaa1c15999ae141d42c350bd3bbb70db2",
    "check_fairness.py": "aad33a6e14f59749b25bdd6023ec09e05ca5c15d034c333c00da74a031464f22",
    "util/utils.py": "242312ba5ea2a9be4484611f206fe8907fd372aabc6c6c3b0fcca5ce44d5e17f",
}


class SafePipelineAdapter(ClassifierMixin, BaseEstimator):
    """Expose classifier tags and strip SEX before every prediction; never fit."""
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.forwarded_columns = []

    def _features(self, audit_frame):
        columns = list(self.pipeline.feature_names_in_)
        if set(columns) != set(FEATURES) or {"ID", "SEX", "default"} & set(columns):
            raise ValueError("Saved baseline predictor schema changed.")
        if set(audit_frame.columns) != set(columns) | {"SEX"}:
            raise ValueError("SAFE audit frame must contain exactly the 22 predictors plus SEX.")
        selected = audit_frame.loc[:, columns]
        self.forwarded_columns.append(tuple(selected.columns))
        return selected

    def predict_proba(self, X):
        return self.pipeline.predict_proba(self._features(X))

    def predict(self, X):
        return self.pipeline.predict(self._features(X))

    def fit(self, X, y=None):
        raise RuntimeError("SAFE adapter is read-only: baseline retraining is prohibited.")


def provenance():
    if version("safeaipackage") != "0.8.3":
        raise ValueError("Use the inspected safeaipackage==0.8.3 environment.")
    modules = {"core.py": core, "check_fairness.py": check_fairness, "util/utils.py": utils}
    actual = {name: sha256(module.__file__) for name, module in modules.items()}
    # Normalize newline-only differences for installs on other platforms.
    import hashlib
    normalized = {name: hashlib.sha256(Path(module.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                  for name, module in modules.items()}
    if actual != EXPECTED_SOURCE_HASHES and normalized != EXPECTED_SOURCE_HASHES:
        raise ValueError("Installed SAFE source differs from the inspected implementation.")
    return {"package": "safeaipackage", "version": version("safeaipackage"),
        "source_hashes": actual, "inspected_github_commit": INSPECTED_COMMIT,
        "github_comparison": "core identical; fairness/utils differ only in whitespace/newline",
        "installed_commit_metadata": None, "implementation": "unmodified installed distribution",
        "functions": ["safeaipackage.core.rga", "safeaipackage.check_fairness.compute_rga_parity"],
        "parity_return": "string containing max(group RGA) - min(group RGA)",
        "group_order": "training SEX value_counts order; no privileged/unprivileged assignment",
        "yhat_behavior": "validated as input; subgroup probabilities recomputed through adapter",
        "input_index_behavior": "package resets inputs; supply aligned positional order"}


def evaluate_safe(pipeline, X_train, X_test, meta_train, meta_test, y_test, probabilities):
    train, test = X_train.reset_index(drop=True).copy(), X_test.reset_index(drop=True).copy()
    train["SEX"], test["SEX"] = meta_train.SEX.to_numpy(), meta_test.SEX.to_numpy()
    y, p = np.asarray(y_test), np.asarray(probabilities)
    adapter = SafePipelineAdapter(pipeline)
    assert is_classifier(adapter)
    np.testing.assert_array_equal(adapter.predict_proba(test)[:, 1], p)
    np.testing.assert_array_equal(adapter.predict(test), pipeline.predict(X_test))
    package_return = check_fairness.compute_rga_parity(train, test, y.tolist(), p.tolist(), adapter, "SEX")
    match = re.fullmatch(r"The RGA-based imparity between the protected gorups is (.+)\.", package_return)
    if match is None:
        raise ValueError(f"Unexpected SAFE return format: {package_return!r}")
    returned_gap = float(match.group(1))
    groups = {sex: float(core.rga(y[test.SEX == sex].tolist(), p[test.SEX == sex].tolist())) for sex in (1, 2)}
    np.testing.assert_allclose(returned_gap, abs(groups[2] - groups[1]), rtol=0, atol=1e-14)
    return float(core.rga(y.tolist(), p.tolist())), groups, returned_gap, package_return
