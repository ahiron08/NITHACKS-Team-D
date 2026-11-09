"""
EDITH Hybrid Ensemble Training
==============================

Combines Isolation Forest, XGBoost, Random Forest, and Autoencoder
using weighted probability combination and stacking.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_common import (
    load_data, stratified_split_df, compute_metrics, save_metrics,
    MODEL_DIR, REPORT_DIR, TARGET_COL, ID_COLS, RANDOM_SEED,
)
from feature_engineering import engineer_features


def load_isolation_forest():
    model = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "isolation_forest_scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "isolation_forest_threshold.json")) as f:
        threshold = json.load(f)["threshold"]
    with open(os.path.join(MODEL_DIR, "isolation_forest_features.json")) as f:
        features = json.load(f)
    return model, scaler, threshold, features


def load_xgboost():
    model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "xgboost_scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "xgboost_threshold.json")) as f:
        threshold = json.load(f)["threshold"]
    with open(os.path.join(MODEL_DIR, "xgboost_features.json")) as f:
        features = json.load(f)
    return model, scaler, threshold, features


def load_random_forest():
    model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "random_forest_scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "random_forest_threshold.json")) as f:
        threshold = json.load(f)["threshold"]
    with open(os.path.join(MODEL_DIR, "random_forest_features.json")) as f:
        features = json.load(f)
    return model, scaler, threshold, features


def load_autoencoder():
    with open(os.path.join(MODEL_DIR, "autoencoder_features.json")) as f:
        features = json.load(f)
    with open(os.path.join(MODEL_DIR, "autoencoder_threshold.json")) as f:
        threshold = json.load(f)["threshold"]
    scaler = joblib.load(os.path.join(MODEL_DIR, "autoencoder_scaler.pkl"))
    checkpoint = torch.load(os.path.join(MODEL_DIR, "autoencoder.pth"))
    from train_autoencoder import Autoencoder
    model = Autoencoder(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, scaler, threshold, features


def get_iforest_anomaly_score(model, X):
    scores = model.decision_function(X)
    return -scores


def get_xgboost_prob(model, X):
    return model.predict_proba(X)[:, 1]


def get_rf_prob(model, X):
    return model.predict_proba(X)[:, 1]


def get_autoencoder_error(model, X):
    with torch.no_grad():
        X_tensor = torch.tensor(X.astype(np.float32))
        reconstructed = model(X_tensor)
        errors = torch.mean((reconstructed - X_tensor) ** 2, dim=1).numpy()
    return errors


def train_ensemble():
    print("=" * 60)
    print("Training Hybrid Ensemble")
    print("=" * 60)

    train_df, test_df = load_data()

    train_eng, scaler, feature_cols = engineer_features(train_df)
    test_eng, _, _ = engineer_features(test_df, fit_scaler=scaler)

    X_full = train_eng[feature_cols]
    y_full = train_eng[TARGET_COL]

    X_train, X_val, y_train, y_val, _ = stratified_split_df(
        X_full, y_full, val_size=0.25, random_state=RANDOM_SEED
    )

    X_test = test_eng[feature_cols]
    y_test = test_df[TARGET_COL].values

    print("\nLoading trained models...")
    iforest_model, iforest_scaler, iforest_threshold, iforest_features = load_isolation_forest()
    xgb_model, xgb_scaler, xgb_threshold, xgb_features = load_xgboost()
    rf_model, rf_scaler, rf_threshold, rf_features = load_random_forest()
    ae_model, ae_scaler, ae_threshold, ae_features = load_autoencoder()

    print("\nGenerating validation predictions...")
    iforest_val_scores = get_iforest_anomaly_score(iforest_model, X_val[iforest_features].values)
    iforest_val_prob = (iforest_val_scores - iforest_val_scores.min()) / (iforest_val_scores.max() - iforest_val_scores.min() + 1e-8)

    xgb_val_prob = get_xgboost_prob(xgb_model, X_val[xgb_features].values)
    rf_val_prob = get_rf_prob(rf_model, X_val[rf_features].values)
    ae_val_errors = get_autoencoder_error(ae_model, X_val[ae_features].values)
    ae_val_prob = (ae_val_errors - ae_val_errors.min()) / (ae_val_errors.max() - ae_val_errors.min() + 1e-8)

    val_stack = np.column_stack([iforest_val_prob, xgb_val_prob, rf_val_prob, ae_val_prob])

    from sklearn.linear_model import LogisticRegression
    meta_learner = LogisticRegression(random_state=RANDOM_SEED, class_weight='balanced')
    meta_learner.fit(val_stack, y_val)

    val_ensemble_prob = meta_learner.predict_proba(val_stack)[:, 1]

    best_threshold = 0.5
    best_val_f1 = -1
    for threshold in np.arange(0.1, 0.95, 0.05):
        y_val_pred = (val_ensemble_prob >= threshold).astype(int)
        metrics = compute_metrics(y_val.values, y_val_pred, val_ensemble_prob)
        if metrics["f1"] > best_val_f1:
            best_val_f1 = metrics["f1"]
            best_threshold = threshold

    print(f"\nBest threshold: {best_threshold:.2f}")
    print(f"Best val F1: {best_val_f1:.4f}")

    y_val_pred = (val_ensemble_prob >= best_threshold).astype(int)
    val_metrics = compute_metrics(y_val.values, y_val_pred, val_ensemble_prob)
    print(f"\nValidation metrics: {json.dumps({k: v for k, v in val_metrics.items() if k != 'confusion_matrix'}, indent=2)}")

    print("\nGenerating test predictions...")
    iforest_test_scores = get_iforest_anomaly_score(iforest_model, X_test[iforest_features].values)
    iforest_test_prob = (iforest_test_scores - iforest_test_scores.min()) / (iforest_test_scores.max() - iforest_test_scores.min() + 1e-8)

    xgb_test_prob = get_xgboost_prob(xgb_model, X_test[xgb_features].values)
    rf_test_prob = get_rf_prob(rf_model, X_test[rf_features].values)
    ae_test_errors = get_autoencoder_error(ae_model, X_test[ae_features].values)
    ae_test_prob = (ae_test_errors - ae_test_errors.min()) / (ae_test_errors.max() - ae_test_errors.min() + 1e-8)

    test_stack = np.column_stack([iforest_test_prob, xgb_test_prob, rf_test_prob, ae_test_prob])
    test_ensemble_prob = meta_learner.predict_proba(test_stack)[:, 1]
    y_test_pred = (test_ensemble_prob >= best_threshold).astype(int)

    test_metrics = compute_metrics(y_test, y_test_pred, test_ensemble_prob)
    print(f"\nTest metrics: {json.dumps({k: v for k, v in test_metrics.items() if k != 'confusion_matrix'}, indent=2)}")

    ensemble = {
        "meta_learner": meta_learner,
        "threshold": best_threshold,
        "feature_cols": feature_cols,
        "component_features": {
            "isolation_forest": iforest_features,
            "xgboost": xgb_features,
            "random_forest": rf_features,
            "autoencoder": ae_features,
        },
        "component_scalers": {
            "isolation_forest": iforest_scaler,
            "xgboost": xgb_scaler,
            "random_forest": rf_scaler,
            "autoencoder": ae_scaler,
        },
        "component_models": {
            "isolation_forest": iforest_model,
            "xgboost": xgb_model,
            "random_forest": rf_model,
            "autoencoder": ae_model,
        },
    }

    ensemble_path = os.path.join(MODEL_DIR, "final_ensemble.pkl")
    joblib.dump(ensemble, ensemble_path)
    print(f"\nEnsemble saved to: {ensemble_path}")

    all_metrics = {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "threshold": float(best_threshold),
        "meta_learner_coef": meta_learner.coef_.tolist(),
        "meta_learner_intercept": meta_learner.intercept_.tolist(),
    }
    save_metrics(all_metrics, os.path.join(REPORT_DIR, "ensemble_metrics.json"))

    print(f"\nEnsemble training complete.")
    return ensemble, val_metrics, test_metrics


if __name__ == "__main__":
    train_ensemble()