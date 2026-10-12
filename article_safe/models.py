"""Fresh article estimators and whole-predictor designs, separate from saved models."""
from dataclasses import dataclass
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning, PerfectSeparationError
from threadpoolctl import threadpool_limits

from src.baseline import CATEGORICAL, NUMERIC, FEATURES


def validate_predictors(X, features):
    features = tuple(features)
    if len(features) != len(set(features)) or set(features) - set(FEATURES):
        raise ValueError("Only unique original predictors are allowed; metadata cannot enter a model.")
    if set(features) - set(X.columns) or not np.isfinite(X.loc[:, features].to_numpy(dtype=float)).all():
        raise ValueError("Missing or nonfinite predictor inputs.")
    return features


@dataclass
class FittedDesign:
    preprocessor: object
    features: tuple
    mapping: pd.DataFrame
    matrix: np.ndarray

    def columns_for(self, features):
        requested = set(features)
        if requested - set(self.features):
            raise ValueError("A reduced design cannot introduce a new predictor.")
        return np.flatnonzero(self.mapping.feature.isin(requested).to_numpy())

    def transform(self, X):
        validate_predictors(X, self.features)
        if not self.features:
            return np.empty((len(X), 0))
        result = self.preprocessor.transform(X.loc[:, self.features])
        return result if hasattr(result, "toarray") else np.asarray(result, dtype=float)


def fit_design(X, features, *, reference_encoding):
    features = validate_predictors(X, features)
    numeric = [f for f in NUMERIC if f in features]
    categorical = [f for f in CATEGORICAL if f in features]
    if not features:
        return FittedDesign(None, (), pd.DataFrame(columns=["encoded_index", "term", "feature", "reference"]), np.empty((len(X), 0)))
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("scaler", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", drop="first" if reference_encoding else None, sparse_output=not reference_encoding))]), categorical))
    preprocessor = ColumnTransformer(transformers)
    transformed = preprocessor.fit_transform(X.loc[:, features])
    matrix = transformed if hasattr(transformed, "toarray") else np.asarray(transformed, dtype=float)
    mapping = []
    for feature in numeric:
        mapping.append({"encoded_index": len(mapping), "term": feature, "feature": feature, "reference": None})
    if categorical:
        encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        for j, feature in enumerate(categorical):
            dropped = int(encoder.drop_idx_[j]) if reference_encoding else None
            for k, category in enumerate(encoder.categories_[j]):
                if k != dropped:
                    mapping.append({"encoded_index": len(mapping), "term": f"{feature}={category}", "feature": feature,
                                    "reference": str(encoder.categories_[j][dropped]) if dropped is not None else None})
    table = pd.DataFrame(mapping, columns=["encoded_index", "term", "feature", "reference"])
    if matrix.shape[1] != len(table):
        raise ValueError("Encoded-to-original predictor mapping is incomplete.")
    return FittedDesign(preprocessor, features, table, matrix)


@dataclass
class ArticleModel:
    family: str
    features: tuple
    design: FittedDesign
    columns: np.ndarray
    estimator: object
    diagnostics: dict

    @property
    def estimable(self):
        return self.diagnostics["status"] == "ok"

    @property
    def active_mapping(self):
        return self.design.mapping.iloc[self.columns].reset_index(drop=True)

    def encoded(self, X):
        return self.design.transform(X)[:, self.columns]

    def margin(self, X):
        if not self.estimable:
            raise ValueError(f"Model is not estimable: {self.diagnostics['reason']}")
        encoded = self.encoded(X)
        if self.family == "lr_mle":
            return np.c_[np.ones(len(X)), encoded] @ np.asarray(self.estimator.params)
        return self.estimator.decision_function(encoded)

    def probability(self, X):
        if self.family == "lr_mle":
            return expit(self.margin(X))
        if not self.estimable:
            raise ValueError("Model is not estimable.")
        return self.estimator.predict_proba(self.encoded(X))[:, list(self.estimator.classes_).index(1)]

    def unknown_categories(self, X):
        if self.design.preprocessor is None or "cat" not in self.design.preprocessor.named_transformers_:
            return {}
        encoder = self.design.preprocessor.named_transformers_["cat"].named_steps["onehot"]
        category_features = [f for f in CATEGORICAL if f in self.design.features]
        return {f: int((~X[f].isin(categories)).sum()) for f, categories in zip(category_features, encoder.categories_) if f in self.features}


def fit_lr(design, y, features, protocol):
    """Reuse one F-fitted design for nested whole-block likelihood comparisons."""
    features = tuple(features)
    columns = design.columns_for(features)
    matrix = np.c_[np.ones(len(y)), design.matrix[:, columns]]
    y = np.asarray(y, dtype=float)
    if len(design.matrix) != len(y) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Expected aligned observations with both binary outcomes.")
    rank = int(np.linalg.matrix_rank(matrix))
    diagnostic = {"status": "not_estimable", "reason": None, "records": len(y), "design_columns": matrix.shape[1],
                  "design_rank": rank, "regularized": False, "warnings": [], "attempts": []}
    model = ArticleModel("lr_mle", features, design, columns, None, diagnostic)
    if rank != matrix.shape[1]:
        diagnostic["reason"] = "rank_deficient_design"
        return model
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        for maxiter in (protocol["lr"]["maxiter"], 1000):
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = sm.GLM(y, matrix, family=sm.families.Binomial()).fit(
                        maxiter=maxiter, tol=protocol["lr"]["tol"], use_t=False)
                messages = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
                diagnostic["warnings"].extend(messages)
                diagnostic["attempts"].append({"maxiter": maxiter, "converged": bool(result.converged)})
                if any(issubclass(w.category, PerfectSeparationWarning) for w in caught):
                    diagnostic["reason"] = "perfect_separation_warning"
                    break
                if not result.converged:
                    diagnostic["reason"] = "nonconvergence"
                    continue
                parameters = np.asarray(result.params)
                covariance = np.asarray(result.cov_params())
                if not np.isfinite(parameters).all() or not np.isfinite(covariance).all() or not np.isfinite(result.llf):
                    diagnostic["reason"] = "nonfinite_fit"
                    break
                probabilities = np.asarray(result.fittedvalues)
                information = matrix.T @ ((probabilities * (1 - probabilities))[:, None] * matrix)
                information_rank = int(np.linalg.matrix_rank(information))
                diagnostic.update(log_likelihood=float(result.llf), information_rank=information_rank,
                                  information_condition=float(np.linalg.cond(information)),
                                  max_abs_score=float(np.max(np.abs(matrix.T @ (y - probabilities)))),
                                  max_abs_coefficient=float(np.max(np.abs(parameters))))
                if information_rank != rank or (np.diag(covariance) < 0).any():
                    diagnostic["reason"] = "singular_information_or_invalid_covariance"
                    break
                model.estimator = result
                diagnostic.update(status="ok", reason=None)
                break
            except (PerfectSeparationError, np.linalg.LinAlgError, ValueError) as error:
                diagnostic["reason"] = type(error).__name__ + ": " + str(error)
                break
    diagnostic["seconds"] = time.perf_counter() - started
    return model


def fit_gbm(X, y, features, protocol):
    if not features:
        raise ValueError("The forward GBM path begins with one predictor.")
    design = fit_design(X, features, reference_encoding=False)
    classifier = GradientBoostingClassifier(**protocol["gbm"]["classifier_parameters"])
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        classifier.fit(design.matrix, y)
    diagnostic = {"status": "ok", "reason": None, "records": len(y), "design_columns": design.matrix.shape[1],
                  "seconds": time.perf_counter() - started, "classifier_parameters": classifier.get_params()}
    return ArticleModel("gbm", tuple(features), design, np.arange(design.matrix.shape[1]), classifier, diagnostic)