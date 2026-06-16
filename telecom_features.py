"""Deterministic business features for the Company A telecom dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TelecomFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add leakage-safe customer behavior, service, and value features."""

    def fit(self, features: pd.DataFrame, target: pd.Series | None = None):
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        engineered = features.copy()

        def ratio(name: str, numerator: str, denominator: str) -> None:
            denominator_values = engineered[denominator].replace(0, np.nan)
            engineered[name] = engineered[numerator] / denominator_values

        def difference(name: str, left: str, right: str) -> None:
            engineered[name] = engineered[left] - engineered[right]

        ratio("call_completion_rate", "complete_Mean", "attempt_Mean")
        ratio("call_failure_rate", "drop_blk_Mean", "attempt_Mean")
        ratio("voice_drop_rate", "drop_vce_Mean", "plcd_vce_Mean")
        ratio("voice_block_rate", "blck_vce_Mean", "plcd_vce_Mean")
        ratio("voice_unanswered_rate", "unan_vce_Mean", "plcd_vce_Mean")
        ratio("data_drop_rate", "drop_dat_Mean", "plcd_dat_Mean")
        ratio("data_block_rate", "blck_dat_Mean", "plcd_dat_Mean")
        ratio("overage_usage_share", "ovrmou_Mean", "mou_Mean")
        ratio("overage_revenue_share", "ovrrev_Mean", "rev_Mean")
        ratio("recurring_revenue_share", "totmrc_Mean", "rev_Mean")
        ratio("customer_care_calls_per_100_calls", "custcare_Mean", "attempt_Mean")
        engineered["customer_care_calls_per_100_calls"] *= 100
        ratio("customer_care_minutes_per_call", "cc_mou_Mean", "custcare_Mean")
        ratio("active_subscriber_share", "actvsubs", "uniqsubs")
        ratio("handsets_per_active_subscriber", "phones", "actvsubs")
        ratio("models_per_handset", "models", "phones")
        ratio("equipment_age_to_tenure", "eqpdays", "months")
        engineered["equipment_age_to_tenure"] /= 30

        ratio("current_to_3m_usage", "mou_Mean", "avg3mou")
        ratio("current_to_6m_usage", "mou_Mean", "avg6mou")
        ratio("current_to_lifetime_usage", "mou_Mean", "avgmou")
        ratio("current_to_3m_revenue", "rev_Mean", "avg3rev")
        ratio("current_to_6m_revenue", "rev_Mean", "avg6rev")
        ratio("current_to_lifetime_revenue", "rev_Mean", "avgrev")
        difference("usage_vs_3m", "mou_Mean", "avg3mou")
        difference("usage_vs_6m", "mou_Mean", "avg6mou")
        difference("revenue_vs_3m", "rev_Mean", "avg3rev")
        difference("revenue_vs_6m", "rev_Mean", "avg6rev")

        ratio("peak_voice_share", "peak_vce_Mean", "attempt_Mean")
        ratio("offpeak_voice_share", "opk_vce_Mean", "attempt_Mean")
        ratio("received_voice_share", "recv_vce_Mean", "attempt_Mean")
        ratio("lifetime_calls_per_month", "totcalls", "months")
        ratio("lifetime_minutes_per_month", "totmou", "months")
        ratio("lifetime_revenue_per_month", "totrev", "months")
        ratio("revenue_per_minute", "rev_Mean", "mou_Mean")
        ratio("minutes_per_call", "mou_Mean", "attempt_Mean")

        numeric_columns = engineered.select_dtypes(include=np.number).columns
        engineered[numeric_columns] = engineered[numeric_columns].replace(
            [np.inf, -np.inf], np.nan
        )
        return engineered
