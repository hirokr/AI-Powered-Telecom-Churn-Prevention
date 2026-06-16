#!/usr/bin/env python3
"""Train the validation-selected CatBoost and histogram churn ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from churn_ensemble import ChurnEnsemble
from train_models import (
    build_models,
    choose_f1_threshold,
    classification_metrics,
    feature_groups,
    load_data,
    save_confusion_matrix,
    split_data,
)

CATBOOST_CONFIGS = (
    {
        "name": "catboost_d6",
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5,
        "random_strength": 1.0,
    },
    {
        "name": "catboost_d7",
        "depth": 7,
        "learning_rate": 0.04,
        "l2_leaf_reg": 7,
        "random_strength": 0.5,
    },
    {
        "name": "catboost_d8",
        "depth": 8,
        "learning_rate": 0.035,
        "l2_leaf_reg": 10,
        "random_strength": 0.5,
    },
)
ENSEMBLE_WEIGHTS = (0.3, 0.1, 0.2, 0.4)
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("telecom"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/improved"))
    return parser.parse_args()


def prepare_catboost_features(
    features: pd.DataFrame, categorical_columns: list[str]
) -> pd.DataFrame:
    prepared = features.copy()
    prepared[categorical_columns] = (
        prepared[categorical_columns].fillna("__MISSING__").astype(str)
    )
    return prepared


def train_catboost_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    categorical_columns: list[str],
) -> list[CatBoostClassifier]:
    models = []
    for config in CATBOOST_CONFIGS:
        print(f"Training {config['name']}...")
        model = CatBoostClassifier(
            iterations=1500,
            loss_function="Logloss",
            eval_metric="AUC",
            depth=config["depth"],
            learning_rate=config["learning_rate"],
            l2_leaf_reg=config["l2_leaf_reg"],
            random_strength=config["random_strength"],
            random_seed=RANDOM_STATE,
            thread_count=-1,
            od_type="Iter",
            od_wait=100,
            verbose=100,
            allow_writing_files=False,
        )
        model.fit(
            X_train,
            y_train,
            cat_features=categorical_columns,
            eval_set=(X_validation, y_validation),
            use_best_model=True,
        )
        models.append(model)
    return models


def catboost_feature_importance(
    models: list[CatBoostClassifier], feature_columns: list[str]
) -> pd.DataFrame:
    catboost_weights = np.asarray(ENSEMBLE_WEIGHTS[1:])
    normalized_weights = catboost_weights / catboost_weights.sum()
    importance = sum(
        weight * model.get_feature_importance()
        for weight, model in zip(normalized_weights, models, strict=True)
    )
    return pd.DataFrame(
        {"feature": feature_columns, "importance": importance}
    ).sort_values("importance", ascending=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(args.data_dir)
    numeric_columns, categorical_columns = feature_groups(X)
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_data(X, y)

    hist_model = build_models(numeric_columns, categorical_columns)[
        "hist_gradient_boosting"
    ]
    print("Training hist_gradient_boosting...")
    hist_model.fit(X_train, y_train)

    X_train_cat = prepare_catboost_features(X_train, categorical_columns)
    X_validation_cat = prepare_catboost_features(X_validation, categorical_columns)
    catboost_models = train_catboost_models(
        X_train_cat,
        y_train,
        X_validation_cat,
        y_validation,
        categorical_columns,
    )

    ensemble = ChurnEnsemble(
        hist_model=hist_model,
        catboost_models=catboost_models,
        categorical_columns=categorical_columns,
        feature_columns=X.columns.tolist(),
        weights=list(ENSEMBLE_WEIGHTS),
    )
    validation_probabilities = ensemble.predict_proba(X_validation)[:, 1]
    threshold = choose_f1_threshold(y_validation, validation_probabilities)

    test_probabilities = ensemble.predict_proba(X_test)[:, 1]
    metadata = {
        "model": "histogram_catboost_probability_ensemble",
        "weights": {
            "hist_gradient_boosting": ENSEMBLE_WEIGHTS[0],
            **{
                config["name"]: weight
                for config, weight in zip(
                    CATBOOST_CONFIGS, ENSEMBLE_WEIGHTS[1:], strict=True
                )
            },
        },
        "selection_metric": "validation ROC-AUC",
        "validation_metrics_at_0_5": classification_metrics(
            y_validation, validation_probabilities, 0.5
        ),
        "decision_threshold": threshold,
        "test_metrics_at_0_5": classification_metrics(
            y_test, test_probabilities, 0.5
        ),
        "test_metrics_at_selected_threshold": classification_metrics(
            y_test, test_probabilities, threshold
        ),
        "split_rows": {
            "train": len(X_train),
            "validation": len(X_validation),
            "test": len(X_test),
        },
    }

    joblib.dump(ensemble, args.output_dir / "improved_churn_model.joblib")
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    catboost_feature_importance(
        catboost_models, X.columns.tolist()
    ).to_csv(args.output_dir / "feature_importance.csv", index=False)
    save_confusion_matrix(
        y_test,
        test_probabilities,
        threshold,
        args.output_dir / "test_confusion_matrix.png",
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
