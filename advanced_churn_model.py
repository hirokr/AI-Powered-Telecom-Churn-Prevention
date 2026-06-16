"""Deployable advanced churn model with feature engineering and model blending."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd


@dataclass
class AdvancedChurnModel:
    """Blend engineered LightGBM probabilities with the categorical ensemble."""

    feature_engineer: object
    preprocessor: object
    lightgbm_model: object
    categorical_ensemble: object
    feature_columns: list[str]
    lightgbm_weight: float = 0.65

    def _validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_columns) - set(features.columns)
        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")
        return features.loc[:, self.feature_columns]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return class probabilities for raw merged customer features."""
        validated = self._validate_features(features)
        engineered = self.feature_engineer.transform(validated)
        transformed = self.preprocessor.transform(engineered)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            lightgbm_probability = self.lightgbm_model.predict_proba(transformed)[
                :, 1
            ]
        categorical_probability = self.categorical_ensemble.predict_proba(validated)[
            :, 1
        ]
        churn_probability = (
            self.lightgbm_weight * lightgbm_probability
            + (1.0 - self.lightgbm_weight) * categorical_probability
        )
        return np.column_stack((1.0 - churn_probability, churn_probability))

    def predict(self, features: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary churn predictions at the requested threshold."""
        return (self.predict_proba(features)[:, 1] >= threshold).astype(int)
