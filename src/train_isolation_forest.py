"""
EDITH Isolation Forest Training
===============================

Trains an Isolation Forest model for anomaly detection.
- Uses normal-only training data where appropriate.
- Tunes contamination and decision threshold on validation data.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_common import (
    load_data, stratified_split, stratified_split_df, compute_metrics, save_model, save_metrics,
    MODEL_DIR, REPORT_DIR, TARGET_COL, ID_COLS, RANDOM_SEED,
)
from feature_engineering import engineer_features, get_feature_schema

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid


def train_isolation_forest():
    print("=" * 60)
    print("Training Isolation Forest")
    print("=" * 60)

    train_df, test_df = load_data()

    # Feature engineering on training data (fit scaler)
    train_eng, scaler, feature_cols = engineer_features(train_df)

    # Apply same engineering to test data (use fitted scaler)
    test_eng, _, _ = engineer_features(test_df, fit_scaler=scaler)

    # Stratified train/validation split using consistent feature columns
    X_full = train_eng[feature_cols]
    y_full = train_eng[TARGET_COL]

    X_train, X_val, y_train, y_val, _ = stratified_split_df(
        X_full, y_full, val_size=0.25, random_state=RANDOM_SEED
    )

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    print(f"Train anomaly rate: {y_train.mean():.4f}")
    print(f"Val anomaly rate: {y_val.mean():.4f}")

    # --- Train on normal-only data (unsupervised approach) ---
    X_train_normal = X_train[y_train == 0]
    print(f"Normal-only training samples: {len(X_train_normal)}")

    # Hyperparameter grid
    param_grid = {
        "n_estimators": [100, 200],
        "max_samples": [256, 512, "auto"],
        "max_features": [0.8, 1.0],
        "contamination": [0.1, 0.15, 0.2],
    }

    best_params = None
    best_val_f1 = -1
    best_model = None

    print("\nHyperparameter search...")
    for params in ParameterGrid(param_grid):
        model = IsolationForest(
            n_estimators=params["n_estimators"],
            max_samples=params["max_samples"],
            max_features=params["max_features"],
            contamination=params["contamination"],
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(X_train_normal)

        # Decision function: higher = more normal, lower = more anomalous
        val_scores = model.decision_function(X_val)
        # Convert to anomaly score (higher = more anomalous)
        val_anomaly_scores = -val_scores

        # Find optimal threshold on validation data
        # Try percentile-based thresholds
        best_threshold = None
        best_threshold_f1 = -1

        for percentile in [5, 10, 15, 20, 25, 30]:
            threshold = np.percentile(val_anomaly_scores, percentile)
            y_val_pred = (val_anomaly_scores > threshold).astype(int)
            val_metrics = compute_metrics(y_val.values, y_val_pred)
            if val_metrics["f1"] > best_threshold_f1:
                best_threshold_f1 = val_metrics["f1"]
                best_threshold = threshold

        if best_threshold_f1 > best_val_f1:
            best_val_f1 = best_threshold_f1
            best_params = params
            best_model = model
            best_threshold_val = best_threshold

    print(f"\nBest params: {best_params}")
    print(f"Best val F1: {best_val_f1:.4f}")
    print(f"Best threshold: {best_threshold_val:.4f}")

    # Retrain on full training data (normal only)
    X_train_normal_full = X_full[y_full == 0]
    best_model.fit(X_train_normal_full)

    # Evaluate on validation set
    val_scores = best_model.decision_function(X_val)
    val_anomaly_scores = -val_scores
    y_val_pred = (val_anomaly_scores > best_threshold_val).astype(int)
    val_metrics = compute_metrics(y_val.values, y_val_pred, val_anomaly_scores)
    print(f"\nValidation metrics: {json.dumps({k: v for k, v in val_metrics.items() if k != 'confusion_matrix'}, indent=2)}")

    # Evaluate on test set (held out)
    test_scores = best_model.decision_function(test_eng[feature_cols])
    test_anomaly_scores = -test_scores
    y_test_pred = (test_anomaly_scores > best_threshold_val).astype(int)
    test_metrics = compute_metrics(test_df[TARGET_COL].values, y_test_pred, test_anomaly_scores)
    print(f"\nTest metrics: {json.dumps({k: v for k, v in test_metrics.items() if k != 'confusion_matrix'}, indent=2)}")

    # Save model, scaler, threshold, feature cols
    model_path = os.path.join(MODEL_DIR, "isolation_forest.pkl")
    scaler_path = os.path.join(MODEL_DIR, "isolation_forest_scaler.pkl")
    threshold_path = os.path.join(MODEL_DIR, "isolation_forest_threshold.json")
    features_path = os.path.join(MODEL_DIR, "isolation_forest_features.json")

    save_model(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved to: {scaler_path}")

    with open(threshold_path, "w") as f:
        json.dump({"threshold": float(best_threshold_val)}, f, indent=2)
    with open(features_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save metrics
    all_metrics = {
        "best_params": best_params,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_metrics(all_metrics, os.path.join(REPORT_DIR, "isolation_forest_metrics.json"))

    print(f"\nIsolation Forest training complete.")
    return best_model, scaler, best_threshold_val, feature_cols, val_metrics, test_metrics


if __name__ == "__main__":
    train_isolation_forest()