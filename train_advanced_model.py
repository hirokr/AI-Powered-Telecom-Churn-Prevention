#!/usr/bin/env python3
"""Train the engineered LightGBM and categorical churn ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from advanced_churn_model import AdvancedChurnModel
from telecom_features import TelecomFeatureEngineer
from train_models import (
    choose_f1_threshold,
    classification_metrics,
    feature_groups,
    load_data,
    save_confusion_matrix,
    split_data,
    tree_preprocessor,
)

LIGHTGBM_WEIGHT = 0.65
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("telecom"))
    parser.add_argument(
        "--categorical-model",
        type=Path,
        default=Path("artifacts/improved/improved_churn_model.joblib"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/advanced"))
    return parser.parse_args()


def build_lightgbm() -> LGBMClassifier:
    return LGBMClassifier(
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


def save_feature_importance(
    model: LGBMClassifier, preprocessor: object, output_path: Path
) -> None:
    names = preprocessor.get_feature_names_out()
    importance = pd.DataFrame(
        {
            "feature": names,
            "gain": model.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False)
    importance.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    if not args.categorical_model.exists():
        raise FileNotFoundError(
            f"{args.categorical_model} is missing. Run train_improved_model.py first."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_features, target = load_data(args.data_dir)
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_data(raw_features, target)

    feature_engineer = TelecomFeatureEngineer()
    X_train_engineered = feature_engineer.fit_transform(X_train)
    X_validation_engineered = feature_engineer.transform(X_validation)
    numeric_columns, categorical_columns = feature_groups(X_train_engineered)
    preprocessor = tree_preprocessor(numeric_columns, categorical_columns)
    train_matrix = preprocessor.fit_transform(X_train_engineered)
    validation_matrix = preprocessor.transform(X_validation_engineered)

    lightgbm_model = build_lightgbm()
    lightgbm_model.fit(
        train_matrix,
        y_train,
        eval_set=[(validation_matrix, y_validation)],
        eval_metric="auc",
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )

    advanced_model = AdvancedChurnModel(
        feature_engineer=feature_engineer,
        preprocessor=preprocessor,
        lightgbm_model=lightgbm_model,
        categorical_ensemble=joblib.load(args.categorical_model),
        feature_columns=raw_features.columns.tolist(),
        lightgbm_weight=LIGHTGBM_WEIGHT,
    )

    validation_probability = advanced_model.predict_proba(X_validation)[:, 1]
    threshold = choose_f1_threshold(y_validation, validation_probability)
    test_probability = advanced_model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "engineered_lightgbm_and_categorical_ensemble",
        "weights": {
            "engineered_lightgbm": LIGHTGBM_WEIGHT,
            "categorical_ensemble": 1.0 - LIGHTGBM_WEIGHT,
        },
        "lightgbm_best_iteration": lightgbm_model.best_iteration_,
        "engineered_feature_count": X_train_engineered.shape[1],
        "selection_metric": "validation ROC-AUC",
        "validation_metrics_at_0_5": classification_metrics(
            y_validation, validation_probability, 0.5
        ),
        "decision_threshold": threshold,
        "test_metrics_at_0_5": classification_metrics(
            y_test, test_probability, 0.5
        ),
        "test_metrics_at_selected_threshold": classification_metrics(
            y_test, test_probability, threshold
        ),
        "split_rows": {
            "train": len(X_train),
            "validation": len(X_validation),
            "test": len(X_test),
        },
    }

    joblib.dump(advanced_model, args.output_dir / "advanced_churn_model.joblib")
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    save_feature_importance(
        lightgbm_model,
        preprocessor,
        args.output_dir / "feature_importance.csv",
    )
    save_confusion_matrix(
        y_test,
        test_probability,
        threshold,
        args.output_dir / "test_confusion_matrix.png",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
