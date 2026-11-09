"""
EDITH Random Forest Training
============================

Trains a supervised Random Forest classifier for anomaly detection.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_common import (
    load_data, stratified_split_df, compute_metrics, save_model, save_metrics,
    MODEL_DIR, REPORT_DIR, TARGET_COL, ID_COLS, RANDOM_SEED,
)
from feature_engineering import engineer_features

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score


def train_random_forest():
    print("=" * 60)
    print("Training Random Forest")
    print("=" * 60)

    train_df, test_df = load_data()

    # Feature engineering on training data (fit scaler)
    train_eng, scaler, feature_cols = engineer_features(train_df)

    # Apply same engineering to test data (use fitted scaler)
    test_eng, _, _ = engineer_features(test_df, fit_scaler=scaler)

    # Stratified train/validation split
    X_full = train_eng[feature_cols]
    y_full = train_eng[TARGET_COL]

    X_train, X_val, y_train, y_val, _ = stratified_split_df(
        X_full, y_full, val_size=0.25, random_state=RANDOM_SEED
    )

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    print(f"Train anomaly rate: {y_train.mean():.4f}")
    print(f"Val anomaly rate: {y_val.mean():.4f}")

    # Compute class weight for imbalanced classes
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    class_weight = {0: 1.0, 1: neg_count / pos_count if pos_count > 0 else 1.0}
    print(f"class_weight: {class_weight}")

    # Hyperparameter distributions for RandomizedSearchCV
    # Use balanced class weight for imbalanced classes
    best_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=2,
    )

    print("\nTraining Random Forest (fixed hyperparameters)...")
    best_model.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate on validation set
    val_prob = best_model.predict_proba(X_val)[:, 1]
    # Find optimal threshold on validation data
    best_threshold = 0.5
    best_val_f1 = -1
    for threshold in np.arange(0.1, 0.95, 0.05):
        y_val_pred = (val_prob >= threshold).astype(int)
        val_metrics = compute_metrics(y_val.values, y_val_pred, val_prob)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_threshold = threshold

    print(f"\nBest threshold: {best_threshold:.2f}")
    print(f"Best val F1: {best_val_f1:.4f}")

    # Validation metrics with best threshold
    y_val_pred = (val_prob >= best_threshold).astype(int)
    val_metrics = compute_metrics(y_val.values, y_val_pred, val_prob)
    print(f"\nValidation metrics: {json.dumps({k: v for k, v in val_metrics.items() if k != 'confusion_matrix'}, indent=2)}")

    # Test set evaluation (held out)
    test_prob = best_model.predict_proba(test_eng[feature_cols])[:, 1]
    y_test_pred = (test_prob >= best_threshold).astype(int)
    test_metrics = compute_metrics(test_df[TARGET_COL].values, y_test_pred, test_prob)
    print(f"\nTest metrics: {json.dumps({k: v for k, v in test_metrics.items() if k != 'confusion_matrix'}, indent=2)}")

    # Save model, scaler, threshold, feature cols
    model_path = os.path.join(MODEL_DIR, "random_forest.pkl")
    scaler_path = os.path.join(MODEL_DIR, "random_forest_scaler.pkl")
    threshold_path = os.path.join(MODEL_DIR, "random_forest_threshold.json")
    features_path = os.path.join(MODEL_DIR, "random_forest_features.json")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    with open(threshold_path, "w") as f:
        json.dump({"threshold": float(best_threshold)}, f, indent=2)
    with open(features_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save metrics
    all_metrics = {
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_metrics(all_metrics, os.path.join(REPORT_DIR, "random_forest_metrics.json"))

    print(f"\nRandom Forest training complete.")
    return best_model, scaler, best_threshold, feature_cols, val_metrics, test_metrics


if __name__ == "__main__":
    train_random_forest()