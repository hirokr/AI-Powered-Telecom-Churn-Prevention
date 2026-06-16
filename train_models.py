#!/usr/bin/env python3
"""Train and evaluate churn classifiers for the Company A telecom dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "churn"
ID_COLUMN = "Customer_ID"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("telecom"),
        help="Directory containing Client.csv and Record.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for trained model and evaluation artifacts.",
    )
    return parser.parse_args()


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    client = pd.read_csv(data_dir / "Client.csv")
    record = pd.read_csv(data_dir / "Record.csv")

    for name, table in (("Client.csv", client), ("Record.csv", record)):
        if ID_COLUMN not in table:
            raise ValueError(f"{name} does not contain {ID_COLUMN}.")
        if table[ID_COLUMN].isna().any():
            raise ValueError(f"{name} contains missing customer IDs.")
        if table[ID_COLUMN].duplicated().any():
            raise ValueError(f"{name} contains duplicate customer IDs.")

    merged = record.merge(
        client,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(record) or len(merged) != len(client):
        raise ValueError("The customer tables do not have identical ID coverage.")
    if TARGET not in merged:
        raise ValueError(f"Merged data does not contain target column {TARGET}.")
    if merged[TARGET].isna().any():
        raise ValueError("The churn target contains missing values.")
    if not set(merged[TARGET].unique()).issubset({0, 1}):
        raise ValueError("The churn target must contain only 0 and 1.")

    X = merged.drop(columns=[ID_COLUMN, TARGET])
    y = merged[TARGET].astype("int8")
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
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
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def feature_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    if len(categorical) + len(numeric) != X.shape[1]:
        raise ValueError("Some features were not classified as numeric or categorical.")
    return numeric, categorical


def linear_preprocessor(
    numeric: list[str], categorical: list[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ]
    )


def tree_preprocessor(
    numeric: list[str], categorical: list[str]
) -> ColumnTransformer:
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


def build_models(numeric: list[str], categorical: list[str]) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", linear_preprocessor(numeric, categorical)),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", tree_preprocessor(numeric, categorical)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocessor", tree_preprocessor(numeric, categorical)),
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


def classification_metrics(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
    }


def choose_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [
        f1_score(y_true, probabilities >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def save_confusion_matrix(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(y_true, probabilities >= threshold)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stay", "Churn"],
        yticklabels=["Stay", "Churn"],
    )
    plt.title(f"Test confusion matrix (threshold={threshold:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_model_comparison(results: pd.DataFrame, output_path: Path) -> None:
    ordered = results.sort_values("roc_auc", ascending=True)
    plt.figure(figsize=(7, 4.5))
    bars = plt.barh(ordered["model"], ordered["roc_auc"], color="#2878B5")
    plt.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.xlim(0.45, max(0.75, ordered["roc_auc"].max() + 0.03))
    plt.xlabel("Validation ROC-AUC")
    plt.title("Churn model comparison")
    for bar, value in zip(bars, ordered["roc_auc"], strict=True):
        plt.text(value + 0.004, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(args.data_dir)
    numeric, categorical = feature_groups(X)
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_data(X, y)

    models = build_models(numeric, categorical)
    validation_rows: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}
    validation_probabilities: dict[str, np.ndarray] = {}

    for name, pipeline in models.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_validation)[:, 1]
        metrics = classification_metrics(y_validation, probabilities, threshold=0.5)
        validation_rows.append({"model": name, **metrics})
        fitted_models[name] = pipeline
        validation_probabilities[name] = probabilities
        print(f"  validation ROC-AUC: {metrics['roc_auc']:.4f}")

    validation_results = pd.DataFrame(validation_rows).sort_values(
        "roc_auc", ascending=False
    )
    validation_results.to_csv(args.output_dir / "validation_metrics.csv", index=False)
    save_model_comparison(
        validation_results, args.output_dir / "model_comparison.png"
    )

    best_name = str(validation_results.iloc[0]["model"])
    best_model = fitted_models[best_name]
    threshold = choose_f1_threshold(
        y_validation, validation_probabilities[best_name]
    )

    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    default_test_metrics = classification_metrics(
        y_test, test_probabilities, threshold=0.5
    )
    tuned_test_metrics = classification_metrics(
        y_test, test_probabilities, threshold=threshold
    )

    permutation = permutation_importance(
        best_model,
        X_validation,
        y_validation,
        scoring="roc_auc",
        n_repeats=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_samples=5000,
    )
    feature_importance = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": permutation.importances_mean,
                "importance_std": permutation.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance.to_csv(
        args.output_dir / "permutation_importance.csv", index=False
    )

    metadata = {
        "target": TARGET,
        "selected_model": best_name,
        "selection_metric": "validation ROC-AUC",
        "decision_threshold": threshold,
        "split_rows": {
            "train": len(X_train),
            "validation": len(X_validation),
            "test": len(X_test),
        },
        "feature_count": X.shape[1],
        "numeric_feature_count": len(numeric),
        "categorical_feature_count": len(categorical),
        "target_rate": float(y.mean()),
        "test_metrics_at_0_5": default_test_metrics,
        "test_metrics_at_selected_threshold": tuned_test_metrics,
    }
    with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    joblib.dump(best_model, args.output_dir / "best_churn_model.joblib")
    save_confusion_matrix(
        y_test,
        test_probabilities,
        threshold,
        args.output_dir / "test_confusion_matrix.png",
    )

    print("\nValidation comparison:")
    print(validation_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSelected model: {best_name}")
    print(f"Validation-selected F1 threshold: {threshold:.3f}")
    print("Held-out test metrics:")
    for metric, value in tuned_test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print("\nTop permutation features:")
    print(feature_importance.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
