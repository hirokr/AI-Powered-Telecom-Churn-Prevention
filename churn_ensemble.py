"""Reusable probability ensemble for telecom churn scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ChurnEnsemble:
    """Blend one sklearn pipeline with native CatBoost classifiers."""

    hist_model: object
    catboost_models: list[object]
    categorical_columns: list[str]
    feature_columns: list[str]
    weights: list[float]

    def __post_init__(self) -> None:
        expected_weight_count = 1 + len(self.catboost_models)
        if len(self.weights) != expected_weight_count:
            raise ValueError("Each ensemble member must have one weight.")
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("Ensemble weights must sum to one.")

    def _validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_columns) - set(features.columns)
        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")
        return features.loc[:, self.feature_columns]

    def _prepare_catboost_features(self, features: pd.DataFrame) -> pd.DataFrame:
        prepared = features.copy()
        prepared[self.categorical_columns] = (
            prepared[self.categorical_columns].fillna("__MISSING__").astype(str)
        )
        return prepared

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return class probabilities using the validation-selected blend."""
        validated = self._validate_features(features)
        catboost_features = self._prepare_catboost_features(validated)

        churn_probability = self.weights[0] * self.hist_model.predict_proba(
            validated
        )[:, 1]
        for weight, model in zip(
            self.weights[1:], self.catboost_models, strict=True
        ):
            churn_probability += weight * model.predict_proba(catboost_features)[:, 1]

        return np.column_stack((1.0 - churn_probability, churn_probability))

    def predict(self, features: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary churn predictions at the requested threshold."""
        return (self.predict_proba(features)[:, 1] >= threshold).astype(int)
