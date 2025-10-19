"""
Model selection utilities for the forecasting pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score


@dataclass
class CandidateModel:
    name: str
    estimator: BaseEstimator
    param_distributions: Optional[Dict[str, Sequence[Any]]] = None
    n_iter: int = 15


@dataclass
class ModelArtifact:
    name: str
    estimator: BaseEstimator
    mean_cv_score: float
    mean_absolute_error: float
    cv_scores: List[float]
    best_params: Dict[str, Any]
    feature_importances: Dict[str, float]


class ModelSelector:
    """
    Evaluates multiple candidate estimators with time-series aware validation and
    retains the best-performing configuration.
    """

    def __init__(
        self,
        candidates: Iterable[CandidateModel],
        scoring: str = "neg_mean_absolute_error",
        n_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        self.candidates = list(candidates)
        self.scoring = scoring
        self.cv = TimeSeriesSplit(n_splits=n_splits)
        self.random_state = random_state
        self.best_artifact_: Optional[ModelArtifact] = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Sequence[str]) -> ModelArtifact:
        if not self.candidates:
            raise ValueError("No candidate models provided for selection.")

        evaluated: List[ModelArtifact] = []
        best_artifact: Optional[ModelArtifact] = None

        for candidate in self.candidates:
            artifact = self._evaluate_candidate(candidate, X, y, feature_names)
            evaluated.append(artifact)
            if best_artifact is None or artifact.mean_cv_score > best_artifact.mean_cv_score:
                best_artifact = artifact

        if best_artifact is None:
            raise RuntimeError("Model selection failed to identify a viable estimator.")

        self.best_artifact_ = best_artifact
        return best_artifact

    # Internal ------------------------------------------------------------------------

    def _evaluate_candidate(
        self,
        candidate: CandidateModel,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str],
    ) -> ModelArtifact:
        if candidate.param_distributions:
            search = RandomizedSearchCV(
                estimator=candidate.estimator,
                param_distributions=candidate.param_distributions,
                n_iter=candidate.n_iter,
                cv=self.cv,
                scoring=self.scoring,
                random_state=self.random_state,
                n_jobs=1,
            )
            search.fit(X, y)
            best_estimator = search.best_estimator_
            cv_score = float(search.best_score_)
            cv_scores = search.cv_results_["mean_test_score"].tolist()
            best_params = dict(search.best_params_)
        else:
            estimator = clone(candidate.estimator)
            cv_scores_array = cross_val_score(
                estimator,
                X,
                y,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=1,
            )
            cv_scores = cv_scores_array.tolist()
            cv_score = float(np.mean(cv_scores_array))
            best_estimator = estimator.fit(X, y)
            best_params = {}

        best_estimator = clone(best_estimator).fit(X, y)
        feature_importances = extract_feature_importances(best_estimator, feature_names)

        return ModelArtifact(
            name=candidate.name,
            estimator=best_estimator,
            mean_cv_score=cv_score,
            mean_absolute_error=-cv_score,
            cv_scores=cv_scores,
            best_params=best_params,
            feature_importances=feature_importances,
        )


def default_candidates(random_state: int = 42, mode: str = "robust") -> List[CandidateModel]:
    """
    Provides a curated list of diverse estimators suitable for the forecasting task.
    """
    hist = CandidateModel(
        name="hist_gradient_boosting",
        estimator=HistGradientBoostingRegressor(
            loss="squared_error",
            random_state=random_state,
            max_iter=200,
        ),
        param_distributions={
            "max_depth": [3, 5, 7, None],
            "max_leaf_nodes": [31, 63, 127],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "l2_regularization": [0.0, 0.1, 0.3, 0.5],
            "min_samples_leaf": [10, 20, 40],
        },
        n_iter=3,
    )

    forest = CandidateModel(
        name="random_forest",
        estimator=RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=5,
        ),
        param_distributions=(
            {
                "max_depth": [6, 8, 12, None],
                "max_features": ["sqrt", "log2", 0.6, 0.8],
                "min_samples_leaf": [2, 5, 10],
            }
            if mode != "fast"
            else None
        ),
        n_iter=3 if mode != "fast" else 0,
    )

    ridge = CandidateModel(
        name="ridge_regression",
        estimator=Ridge(solver="lsqr", max_iter=1000),
        param_distributions={
            "alpha": [0.1, 0.5, 1.0, 5.0, 10.0, 25.0],
        },
        n_iter=3,
    )

    if mode == "fast":
        hist.param_distributions = None
        hist.n_iter = 0
        return [hist, ridge]

    return [hist, forest, ridge]


def extract_feature_importances(estimator: BaseEstimator, feature_names: Sequence[str]) -> Dict[str, float]:
    """
    Attempt to compute a ranked mapping of feature importances for the fitted estimator.
    """
    if hasattr(estimator, "feature_importances_"):
        importances = getattr(estimator, "feature_importances_")
    elif hasattr(estimator, "coef_"):
        coef = getattr(estimator, "coef_")
        importances = np.abs(np.atleast_1d(coef))
    else:
        return {}

    importances = np.asarray(importances, dtype=float)
    if importances.ndim > 1:
        importances = np.mean(importances, axis=0)

    total = float(np.sum(importances))
    if total == 0.0:
        return {name: 0.0 for name in feature_names}

    normalised = importances / total
    return {
        name: float(value)
        for name, value in zip(feature_names, normalised)
    }
