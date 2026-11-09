"""
EDITH XGBoost Training
======================

Trains a supervised XGBoost classifier for anomaly detection.
Uses labelled normal and anomalous training samples.
Optimizes hyperparameters using RandomizedSearchCV.
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

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score


def train_xgboost():
    print("=" * 60)
    print("Training XGBoost")
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

    # Compute scale_pos_weight for imbalanced classes
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    # Hyperparameter distributions for RandomizedSearchCV
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "min_child_weight": [1, 3, 5, 7],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.3, 0.5],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 1.5, 2.0],
        "scale_pos_weight": [scale_pos_weight],
    }

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )

    f1_scorer = make_scorer(f1_score, zero_division=0)

    print("\nRunning RandomizedSearchCV...")
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=30,
        scoring=f1_scorer,
        cv=3,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X_train, y_train)

    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV F1: {search.best_score_:.4f}")

    best_model = search.best_estimator_

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
    model_path = os.path.join(MODEL_DIR, "xgboost.pkl")
    scaler_path = os.path.join(MODEL_DIR, "xgboost_scaler.pkl")
    threshold_path = os.path.join(MODEL_DIR, "xgboost_threshold.json")
    features_path = os.path.join(MODEL_DIR, "xgboost_features.json")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    with open(threshold_path, "w") as f:
        json.dump({"threshold": float(best_threshold)}, f, indent=2)
    with open(features_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save metrics
    all_metrics = {
        "best_params": search.best_params_,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_metrics(all_metrics, os.path.join(REPORT_DIR, "xgboost_metrics.json"))

    print(f"\nXGBoost training complete.")
    return best_model, scaler, best_threshold, feature_cols, val_metrics, test_metrics


if __name__ == "__main__":
    train_xgboost()