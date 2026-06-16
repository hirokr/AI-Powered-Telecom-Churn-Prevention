#!/usr/bin/env python3
"""Generate the standalone telecom churn submission notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Company_A_Telecom_Churn_Proposal.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.13"},
}

notebook["cells"] = [
    markdown(
        """
# Company A Telecom Churn Prevention Proposal

## Executive summary

Company A has customer, billing, usage, handset, and service-quality data for 100,000 telecom customers. This notebook develops a machine-learning system that ranks customers by their probability of churning 31–60 days after observation.

**Business recommendation:** score active customers weekly and target the highest-risk 20% with action-specific retention treatments. On the held-out test set, this segment captures approximately 30% of churners and has a churn rate around 1.5 times the portfolio average.

The final model is a probability ensemble:

- **65% engineered LightGBM**
- **35% native-categorical CatBoost ensemble**

The model is selected using validation ROC-AUC and evaluated once on an untouched 20% test set. Financial impact is presented as an illustrative scenario, not measured campaign performance.
"""
    ),
    markdown(
        """
## 1. Setup

The notebook expects the following files relative to its location:

```text
telecom/
├── Client.csv
└── Record.csv
```

Uncomment the installation command if the environment does not already contain the required packages.
"""
    ),
    code(
        """
# %pip install -q pandas numpy matplotlib seaborn scikit-learn lightgbm catboost joblib

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "churn"
ID_COLUMN = "Customer_ID"
DATA_DIR = Path("telecom")

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 120)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)
"""
    ),
    markdown(
        """
## 2. Load and validate the data

`Client.csv` contains account, household, handset, and demographic information. `Record.csv` contains recent usage, billing, call-quality measurements, tenure, and the churn target. Both files should contain one unique row per `Customer_ID`.
"""
    ),
    code(
        """
client = pd.read_csv(DATA_DIR / "Client.csv")
record = pd.read_csv(DATA_DIR / "Record.csv")

for name, table in (("Client.csv", client), ("Record.csv", record)):
    assert ID_COLUMN in table.columns, f"{name} is missing {ID_COLUMN}"
    assert table[ID_COLUMN].notna().all(), f"{name} contains missing customer IDs"
    assert not table[ID_COLUMN].duplicated().any(), f"{name} contains duplicate customer IDs"

df = record.merge(client, on=ID_COLUMN, how="inner", validate="one_to_one")
assert len(df) == len(client) == len(record)
assert set(df[TARGET].unique()).issubset({0, 1})

print(f"Client table: {client.shape}")
print(f"Record table: {record.shape}")
print(f"Merged analysis table: {df.shape}")
df.head()
"""
    ),
    markdown(
        """
## 3. Exploratory data analysis

The EDA answers four questions:

1. Is the target balanced?
2. Which fields have substantial missingness?
3. Which customer signals show directional churn risk?
4. Is a multivariate model justified?
"""
    ),
    code(
        """
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

target_rate = df[TARGET].value_counts(normalize=True).sort_index()
axes[0].bar(["Stayed", "Churned"], target_rate.values * 100, color=["#1769AA", "#00A896"])
axes[0].set_title("Target distribution")
axes[0].set_ylabel("Customers (%)")
for index, value in enumerate(target_rate.values * 100):
    axes[0].text(index, value + 1, f"{value:.1f}%", ha="center")

missing = (df.isna().mean() * 100).sort_values(ascending=False).head(12)
axes[1].barh(missing.index[::-1], missing.values[::-1], color="#627D98")
axes[1].set_title("Columns with the most missing data")
axes[1].set_xlabel("Missing values (%)")

plt.tight_layout()
plt.show()

print(f"Overall churn rate: {df[TARGET].mean():.2%}")
"""
    ),
    code(
        """
signals = [
    ("eqpdays", "Equipment age", "days"),
    ("change_mou", "Usage momentum", "change in minutes"),
    ("mou_Mean", "Monthly usage", "minutes"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for axis, (column, title, unit) in zip(axes, signals):
    frame = df[[column, TARGET]].dropna().copy()
    frame["quartile"] = pd.qcut(
        frame[column], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
    )
    rates = frame.groupby("quartile", observed=True)[TARGET].mean() * 100
    axis.bar(rates.index.astype(str), rates.values, color="#2CB1BC")
    axis.set_title(title)
    axis.set_xlabel(f"Quartiles by {unit}")
    axis.set_ylabel("Churn rate (%)")
    axis.set_ylim(0, max(65, rates.max() + 7))
    for position, value in enumerate(rates.values):
        axis.text(position, value + 1, f"{value:.1f}%", ha="center")

plt.suptitle("Churn changes across actionable customer signals", y=1.03)
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        """
### EDA interpretation

- Churn is nearly balanced in the supplied dataset, so accuracy is interpretable here, although ROC-AUC and PR-AUC remain better ranking metrics.
- Several demographic fields contain substantial missingness. The pipeline therefore imputes numeric values and treats missing categorical values consistently.
- Older equipment, declining usage, and low current usage are associated with greater churn risk.
- No single variable cleanly separates churners from non-churners, supporting a multivariate nonlinear model.

The relationships above are predictive associations, not proof that changing a feature will cause retention. Intervention effects must be tested experimentally.
"""
    ),
    markdown(
        """
## 4. Problem definition and data split

**Business objective:** concentrate retention spending on customers with the greatest avoidable revenue risk.

**Machine-learning task:** binary classification and probability ranking of churn 31–60 days ahead.

**Primary model-selection metric:** validation ROC-AUC.

The data is divided into:

- 60% training
- 20% validation for model and threshold selection
- 20% untouched test data for final reporting
"""
    ),
    code(
        """
X = df.drop(columns=[ID_COLUMN, TARGET])
y = df[TARGET].astype("int8")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=RANDOM_STATE,
    stratify=y,
)
X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

split_summary = pd.DataFrame(
    {
        "Rows": [len(X_train), len(X_validation), len(X_test)],
        "Churn rate": [y_train.mean(), y_validation.mean(), y_test.mean()],
    },
    index=["Train", "Validation", "Test"],
)
split_summary
"""
    ),
    markdown(
        """
## 5. Business-driven feature engineering

The raw fields are extended with leakage-safe features derived independently for each customer. These include:

- call completion and failure rates
- overage burden and recurring-revenue share
- customer-care intensity
- equipment age relative to tenure
- current usage and revenue relative to historical averages
- household subscriber utilization
- revenue per minute and minutes per call

No target statistics are used to construct these features.
"""
    ),
    code(
        """
class TelecomFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, features, target=None):
        return self

    def transform(self, features):
        engineered = features.copy()

        def ratio(name, numerator, denominator):
            engineered[name] = (
                engineered[numerator] / engineered[denominator].replace(0, np.nan)
            )

        def difference(name, left, right):
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


feature_engineer = TelecomFeatureEngineer()
X_train_engineered = feature_engineer.fit_transform(X_train)
X_validation_engineered = feature_engineer.transform(X_validation)
X_test_engineered = feature_engineer.transform(X_test)

print(f"Raw features: {X_train.shape[1]}")
print(f"Engineered features: {X_train_engineered.shape[1]}")
"""
    ),
    markdown(
        """
## 6. Preprocessing and evaluation helpers

All preprocessing is fitted only on the training partition.

- Numeric values: median imputation with missing indicators
- Categorical values: explicit missing category and ordinal encoding for tree models
- Logistic regression baseline: median imputation, standardization, and one-hot encoding
"""
    ),
    code(
        """
def feature_groups(features):
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    return numeric, categorical


def tree_preprocessor(numeric, categorical):
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    )
    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                    add_indicator=True,
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ]
    )


def linear_preprocessor(numeric, categorical):
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
                    ]
                ),
                categorical,
            ),
        ]
    )


def classification_metrics(y_true, probability, threshold=0.5):
    prediction = (probability >= threshold).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_true, probability),
        "PR-AUC": average_precision_score(y_true, probability),
        "Accuracy": accuracy_score(y_true, prediction),
        "Precision": precision_score(y_true, prediction, zero_division=0),
        "Recall": recall_score(y_true, prediction, zero_division=0),
        "F1": f1_score(y_true, prediction, zero_division=0),
    }


def choose_f1_threshold(y_true, probability):
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [
        f1_score(y_true, probability >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])
"""
    ),
    markdown(
        """
## 7. Baseline models

Logistic regression provides an interpretable linear baseline. Histogram gradient boosting tests whether nonlinear interactions materially improve customer ranking.
"""
    ),
    code(
        """
raw_numeric, raw_categorical = feature_groups(X_train)

baseline_models = {
    "Logistic regression": Pipeline(
        [
            ("preprocessor", linear_preprocessor(raw_numeric, raw_categorical)),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
    "Histogram gradient boosting": Pipeline(
        [
            ("preprocessor", tree_preprocessor(raw_numeric, raw_categorical)),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_iter=250,
                    max_leaf_nodes=31,
                    min_samples_leaf=30,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
}

validation_predictions = {}
comparison_rows = []
for name, model in baseline_models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_validation)[:, 1]
    validation_predictions[name] = probability
    comparison_rows.append(
        {"Model": name, **classification_metrics(y_validation, probability)}
    )

pd.DataFrame(comparison_rows).sort_values("ROC-AUC", ascending=False)
"""
    ),
    markdown(
        """
## 8. Engineered LightGBM model

LightGBM is appropriate for this large mixed tabular dataset because it captures nonlinear interactions, handles the engineered ratios efficiently, and supports regularization and early stopping.
"""
    ),
    code(
        """
engineered_numeric, engineered_categorical = feature_groups(X_train_engineered)
engineered_preprocessor = tree_preprocessor(
    engineered_numeric, engineered_categorical
)

train_matrix = engineered_preprocessor.fit_transform(X_train_engineered)
validation_matrix = engineered_preprocessor.transform(X_validation_engineered)
test_matrix = engineered_preprocessor.transform(X_test_engineered)

lightgbm_model = LGBMClassifier(
    n_estimators=2500,
    learning_rate=0.015,
    num_leaves=63,
    min_child_samples=150,
    subsample=0.9,
    colsample_bytree=0.75,
    reg_alpha=0.5,
    reg_lambda=7,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1,
)

lightgbm_model.fit(
    train_matrix,
    y_train,
    eval_set=[(validation_matrix, y_validation)],
    eval_metric="auc",
    callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
)

lightgbm_validation_probability = lightgbm_model.predict_proba(validation_matrix)[:, 1]
validation_predictions["Engineered LightGBM"] = lightgbm_validation_probability

print(f"Best LightGBM iteration: {lightgbm_model.best_iteration_}")
pd.Series(
    classification_metrics(y_validation, lightgbm_validation_probability),
    name="Validation",
)
"""
    ),
    markdown(
        """
## 9. Native-categorical CatBoost ensemble

CatBoost complements LightGBM by learning categorical effects directly instead of relying only on ordinal encodings. Three regularized configurations are trained and blended using weights selected on validation data.

This is the longest-running section of the notebook.
"""
    ),
    code(
        """
categorical_columns = X_train.select_dtypes(exclude=np.number).columns.tolist()

def prepare_catboost_features(features):
    prepared = features.copy()
    prepared[categorical_columns] = (
        prepared[categorical_columns].fillna("__MISSING__").astype(str)
    )
    return prepared


X_train_cat = prepare_catboost_features(X_train)
X_validation_cat = prepare_catboost_features(X_validation)
X_test_cat = prepare_catboost_features(X_test)

catboost_configs = [
    {
        "name": "CatBoost depth 6",
        "iterations": 927,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5,
        "random_strength": 1.0,
    },
    {
        "name": "CatBoost depth 7",
        "iterations": 1016,
        "depth": 7,
        "learning_rate": 0.04,
        "l2_leaf_reg": 7,
        "random_strength": 0.5,
    },
    {
        "name": "CatBoost depth 8",
        "iterations": 1198,
        "depth": 8,
        "learning_rate": 0.035,
        "l2_leaf_reg": 10,
        "random_strength": 0.5,
    },
]

catboost_models = []
catboost_validation_probabilities = []
catboost_test_probabilities = []

for config in catboost_configs:
    print(f"Training {config['name']}...")
    model = CatBoostClassifier(
        iterations=config["iterations"],
        depth=config["depth"],
        learning_rate=config["learning_rate"],
        l2_leaf_reg=config["l2_leaf_reg"],
        random_strength=config["random_strength"],
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        thread_count=-1,
        verbose=200,
        allow_writing_files=False,
    )
    model.fit(X_train_cat, y_train, cat_features=categorical_columns)
    catboost_models.append(model)
    catboost_validation_probabilities.append(
        model.predict_proba(X_validation_cat)[:, 1]
    )
    catboost_test_probabilities.append(model.predict_proba(X_test_cat)[:, 1])

catboost_weights = np.array([0.1, 0.2, 0.4])
catboost_weights = catboost_weights / catboost_weights.sum()
catboost_validation_probability = sum(
    weight * probability
    for weight, probability in zip(
        catboost_weights, catboost_validation_probabilities
    )
)
catboost_test_probability = sum(
    weight * probability
    for weight, probability in zip(catboost_weights, catboost_test_probabilities)
)

validation_predictions["CatBoost ensemble"] = catboost_validation_probability
pd.Series(
    classification_metrics(y_validation, catboost_validation_probability),
    name="Validation",
)
"""
    ),
    markdown(
        """
## 10. Final probability ensemble and validation comparison

The final probability is:

$$P(\\mathrm{churn}) = 0.65 \\times P_{LightGBM} + 0.35 \\times P_{CatBoost}$$

The blend weights were selected using validation ROC-AUC. The test partition remains untouched until the following section.
"""
    ),
    code(
        """
final_validation_probability = (
    0.65 * lightgbm_validation_probability
    + 0.35 * catboost_validation_probability
)
validation_predictions["Final ensemble"] = final_validation_probability

comparison = pd.DataFrame(
    [
        {
            "Model": name,
            **classification_metrics(y_validation, probability),
        }
        for name, probability in validation_predictions.items()
    ]
).sort_values("ROC-AUC", ascending=False)

display(comparison.style.format({column: "{:.4f}" for column in comparison.columns[1:]}))

plt.figure(figsize=(9, 4.5))
ordered = comparison.sort_values("ROC-AUC")
colors = ["#627D98"] * (len(ordered) - 1) + ["#00A896"]
plt.barh(ordered["Model"], ordered["ROC-AUC"], color=colors)
plt.xlim(0.58, 0.72)
plt.xlabel("Validation ROC-AUC")
plt.title("Model comparison")
for index, value in enumerate(ordered["ROC-AUC"]):
    plt.text(value + 0.002, index, f"{value:.3f}", va="center")
plt.show()
"""
    ),
    markdown(
        """
## 11. Final held-out test evaluation

The test set is evaluated only after the model architecture and blend weights have been fixed. Metrics are reported at:

- **0.50 threshold:** balanced default operating point
- **validation-selected threshold:** threshold that maximizes validation F1 and favors churn recall
"""
    ),
    code(
        """
final_test_probability = (
    0.65 * lightgbm_model.predict_proba(test_matrix)[:, 1]
    + 0.35 * catboost_test_probability
)
selected_threshold = choose_f1_threshold(
    y_validation, final_validation_probability
)

test_metrics_default = classification_metrics(
    y_test, final_test_probability, threshold=0.5
)
test_metrics_selected = classification_metrics(
    y_test, final_test_probability, threshold=selected_threshold
)

test_results = pd.DataFrame(
    [test_metrics_default, test_metrics_selected],
    index=["Threshold 0.50", f"Threshold {selected_threshold:.2f}"],
)
display(test_results.style.format("{:.4f}"))

print(
    f"Final model: 65% engineered LightGBM + 35% CatBoost ensemble\\n"
    f"Evaluation metric: ROC-AUC\\n"
    f"Held-out test ROC-AUC: {test_metrics_default['ROC-AUC']:.4f}"
)
"""
    ),
    code(
        """
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

false_positive_rate, true_positive_rate, _ = roc_curve(
    y_test, final_test_probability
)
axes[0].plot(
    false_positive_rate,
    true_positive_rate,
    color="#1769AA",
    linewidth=3,
    label=f"ROC-AUC = {test_metrics_default['ROC-AUC']:.3f}",
)
axes[0].plot([0, 1], [0, 1], "--", color="#627D98")
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("Held-out ROC curve")
axes[0].legend()

matrix = confusion_matrix(y_test, final_test_probability >= 0.5)
sns.heatmap(
    matrix,
    annot=True,
    fmt=",d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Stay", "Churn"],
    yticklabels=["Stay", "Churn"],
    ax=axes[1],
)
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
axes[1].set_title("Confusion matrix at threshold 0.50")

plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        """
## 12. Feature importance and business interpretation

LightGBM gain importance identifies the variables most useful for prediction. Importance does not establish causality, but it helps organize intervention hypotheses.
"""
    ),
    code(
        """
feature_importance = pd.DataFrame(
    {
        "Feature": engineered_preprocessor.get_feature_names_out(),
        "Gain": lightgbm_model.booster_.feature_importance(importance_type="gain"),
    }
).sort_values("Gain", ascending=False)

top_features = feature_importance.head(15).iloc[::-1]
plt.figure(figsize=(9, 5.5))
plt.barh(
    top_features["Feature"]
    .str.replace("numeric__", "", regex=False)
    .str.replace("categorical__", "", regex=False),
    top_features["Gain"],
    color="#2CB1BC",
)
plt.xlabel("LightGBM gain importance")
plt.title("Leading churn prediction signals")
plt.tight_layout()
plt.show()

feature_importance.head(15)
"""
    ),
    markdown(
        """
### Business interpretation

The leading features group into four intervention themes:

| Theme | Example signals | Potential action |
|---|---|---|
| Lifecycle | Equipment age, tenure | Handset upgrade or financing review |
| Engagement | Current vs historical usage | Plan-fit review or personalized bundle |
| Customer value | Recurring revenue share, handset price | Value-capped priority outreach |
| Service experience | Voice drop and call-completion rates | Proactive service recovery |

These actions should be validated through randomized treatment/control experiments.
"""
    ),
    markdown(
        """
## 13. Targeting analysis

Probability ranking is more useful than a fixed classification threshold for campaign planning. The following analysis asks how many churners are captured at different campaign capacities.
"""
    ),
    code(
        """
scored_test = X_test.copy()
scored_test["actual_churn"] = y_test.to_numpy()
scored_test["churn_probability"] = final_test_probability
scored_test = scored_test.sort_values(
    "churn_probability", ascending=False
).reset_index(drop=True)

shares = np.arange(0.05, 1.01, 0.05)
targeting_rows = []
for share in shares:
    targeted = scored_test.head(int(len(scored_test) * share))
    capture = targeted["actual_churn"].sum() / scored_test["actual_churn"].sum()
    churn_rate = targeted["actual_churn"].mean()
    targeting_rows.append(
        {
            "Customer share": share,
            "Churn capture": capture,
            "Target churn rate": churn_rate,
            "Lift": churn_rate / scored_test["actual_churn"].mean(),
        }
    )

targeting = pd.DataFrame(targeting_rows)
top_twenty = targeting.loc[
    np.isclose(targeting["Customer share"], 0.20)
].iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(
    targeting["Customer share"] * 100,
    targeting["Churn capture"] * 100,
    color="#00A896",
    linewidth=3,
)
axes[0].plot([0, 100], [0, 100], "--", color="#627D98")
axes[0].scatter([20], [top_twenty["Churn capture"] * 100], color="#F59E0B", s=80)
axes[0].set_xlabel("Customers targeted (%)")
axes[0].set_ylabel("Churners captured (%)")
axes[0].set_title("Cumulative churn capture")

axes[1].plot(
    targeting["Customer share"] * 100,
    targeting["Lift"],
    color="#1769AA",
    linewidth=3,
)
axes[1].axhline(1, linestyle="--", color="#627D98")
axes[1].scatter([20], [top_twenty["Lift"]], color="#F59E0B", s=80)
axes[1].set_xlabel("Customers targeted (%)")
axes[1].set_ylabel("Churn-rate lift")
axes[1].set_title("Risk concentration")

plt.tight_layout()
plt.show()

print(
    f"Top 20% of customers captures {top_twenty['Churn capture']:.1%} "
    f"of test-set churners.\\n"
    f"Observed churn rate in that segment: "
    f"{top_twenty['Target churn rate']:.1%}.\\n"
    f"Lift relative to the portfolio: {top_twenty['Lift']:.2f}x."
)
"""
    ),
    markdown(
        """
## 14. Illustrative business impact

The model estimates risk; it does not directly measure the effect of an intervention. The financial calculation below is therefore a transparent planning scenario.

### Base-case assumptions per 100,000 customers

- Target the highest-risk 20%
- Use the measured test-set churn capture and average target-segment revenue
- Retain 20% of contacted customers who would otherwise churn
- Value 12 months of retained monthly revenue
- Spend USD 15 per targeted customer

Company A should replace these assumptions with actual campaign cost, margin, and observed treatment uplift.
"""
    ),
    code(
        """
top_twenty_customers = scored_test.head(int(len(scored_test) * 0.20))

annual_customers = 100_000
targeted_customers = int(annual_customers * 0.20)
portfolio_churn_rate = y.mean()
capture_rate = top_twenty["Churn capture"]
average_monthly_revenue = top_twenty_customers["rev_Mean"].mean()
intervention_cost = 15
months_retained = 12

scenario_rows = []
for save_rate in [0.10, 0.20, 0.30]:
    expected_churners_reached = (
        annual_customers * portfolio_churn_rate * capture_rate
    )
    gross_value = (
        expected_churners_reached
        * save_rate
        * average_monthly_revenue
        * months_retained
    )
    campaign_cost = targeted_customers * intervention_cost
    net_value = gross_value - campaign_cost
    scenario_rows.append(
        {
            "Scenario": f"{save_rate:.0%} save rate",
            "Gross retained revenue": gross_value,
            "Campaign cost": campaign_cost,
            "Net annual value": net_value,
            "Net ROI": net_value / campaign_cost,
        }
    )

impact = pd.DataFrame(scenario_rows)
display(
    impact.style.format(
        {
            "Gross retained revenue": "${:,.0f}",
            "Campaign cost": "${:,.0f}",
            "Net annual value": "${:,.0f}",
            "Net ROI": "{:.1f}x",
        }
    )
)

plt.figure(figsize=(8, 4.5))
plt.bar(
    impact["Scenario"],
    impact["Net annual value"] / 1_000_000,
    color=["#E8F1F8", "#00A896", "#1769AA"],
    edgecolor="#1769AA",
)
plt.ylabel("Illustrative net annual value (USD millions)")
plt.title("Retention value under alternative intervention success rates")
plt.show()
"""
    ),
    markdown(
        """
## 15. Business proposal and implementation roadmap

### Recommended operating model

1. Refresh customer, usage, billing, handset, and service-quality fields weekly.
2. Calculate churn probabilities and rank eligible customers.
3. Target the highest-risk 20% during the PoC.
4. Match each customer to an approved action:
   - handset upgrade for equipment-age risk
   - plan-fit review for usage decline
   - service recovery for quality friction
   - value-capped priority outreach for high-value risk
5. Randomly hold out at least 10% of each risk/action segment.
6. Measure incremental retention, net value, calibration, drift, and segment fairness.

### 90-day roadmap

| Phase | Timing | Output |
|---|---|---|
| Validate | Weeks 1–2 | Data freshness, eligibility, cost and action definitions |
| Integrate | Weeks 3–5 | Automated weekly scoring and CRM delivery |
| Experiment | Weeks 6–9 | Segmented treatment/control campaigns |
| Decide | Weeks 10–12 | Incremental retention, ROI and model-risk review |

**Scale decision:** proceed only if the experiment demonstrates positive incremental retention and net value without material harm to customer segments.
"""
    ),
    markdown(
        """
## 16. Conclusion

The analysis supports a practical shift from broad retention campaigns to risk-ranked intervention:

- **Final model:** 65% engineered LightGBM + 35% CatBoost ensemble
- **Evaluation metric:** held-out ROC-AUC
- **Test ROC-AUC:** approximately **0.702**
- **Top-20% campaign:** captures approximately **30% of churners**
- **Business proposal:** run a 90-day randomized retention PoC with action-specific treatments

The model is useful as a ranking tool, but campaign impact must be established through controlled experimentation. Production deployment should also recalibrate probabilities to Company A's current churn rate.
"""
    ),
    markdown(
        """
## References

1. Deloitte. *2026 Telecommunications Industry Outlook*. https://www.deloitte.com/us/en/insights/industry/technology/technology-media-and-telecom-predictions/2026/telecommunications-industry-outlook.html
2. Company A Dataset Overview. `ENG_Company.md`. Supplied course material.
3. GCI World 2026 Final Assignment Tutorial. `tutorial.ipynb`. Supplied course material.
4. Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
5. Prokhorenkova, L. et al. (2018). *CatBoost: Unbiased Boosting with Categorical Features*. NeurIPS.

### Reproducibility note

The notebook uses fixed random seeds and a deterministic split. Minor metric differences can still occur across library versions or hardware implementations.
"""
    ),
]

nbf.write(notebook, OUTPUT)
print(f"Created {OUTPUT}")
