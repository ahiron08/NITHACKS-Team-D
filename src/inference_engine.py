import numpy as np
import pandas as pd
import torch

from feature_engineering import engineer_features

BASE_DIR = __import__("os").path.dirname(__import__("os").path.abspath(__file__)).replace("\\src", "")
sys_path = BASE_DIR + "\\src"
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)
from model_loader import (
    load_isolation_forest, load_xgboost, load_random_forest,
    load_autoencoder, load_ensemble, get_all_scalers,
)

RISK_THRESHOLDS = {"LOW": 0.0, "MEDIUM": 0.3, "HIGH": 0.6, "CRITICAL": 0.8}


def preprocess_telemetry(telemetry_dict, scaler):
    df = pd.DataFrame([telemetry_dict])
    result_df, fitted_scaler, feature_cols = engineer_features(
        df.copy(), fit_scaler=scaler
    )
    return result_df[feature_cols].iloc[0].values.astype(np.float32)


def run_isolation_forest(model, features):
    decision = model.decision_function(features.reshape(1, -1))
    anomaly_score = -decision[0]
    prediction = 1 if anomaly_score > 0.5 else 0
    return float(anomaly_score), int(prediction)


def run_xgboost(model, features):
    probability = model.predict_proba(features.reshape(1, -1))[:, 1][0]
    anomaly_score = float(probability)
    prediction = 1 if probability > 0.5 else 0
    return float(anomaly_score), int(prediction)


def run_random_forest(model, features):
    probability = model.predict_proba(features.reshape(1, -1))[:, 1][0]
    anomaly_score = float(probability)
    prediction = 1 if probability > 0.5 else 0
    return float(anomaly_score), int(prediction)


def run_autoencoder(model, features):
    with torch.no_grad():
        X_tensor = torch.tensor(features.reshape(1, -1), dtype=torch.float32)
        reconstructed = model(X_tensor)
        error = torch.mean((reconstructed - X_tensor) ** 2, dim=1).numpy()[0]
    reconstruction_error = float(error)
    threshold = 2.54875936508179
    prediction = 1 if reconstruction_error > threshold else 0
    return reconstruction_error, float(threshold), int(prediction)


def run_ensemble(ensemble, scaler, features):
    # Isolation Forest component
    if_model = ensemble["component_models"]["isolation_forest"]
    if_scores = if_model.decision_function(features.reshape(1, -1))
    if_anomaly = -if_scores[0]
    if_min = if_anomaly
    if_max = if_anomaly
    if_prob = (if_anomaly - if_min) / (if_max - if_min + 1e-8)

    # XGBoost component
    xgb_prob = ensemble["component_models"]["xgboost"].predict_proba(
        features.reshape(1, -1)
    )[:, 1][0]

    # Random Forest component
    rf_prob = ensemble["component_models"]["random_forest"].predict_proba(
        features.reshape(1, -1)
    )[:, 1][0]

    # Autoencoder component
    ae_model = ensemble["component_models"]["autoencoder"]
    with torch.no_grad():
        X_tensor = torch.tensor(features.reshape(1, -1), dtype=torch.float32)
        reconstructed = ae_model(X_tensor)
        ae_error = torch.mean((reconstructed - X_tensor) ** 2, dim=1).numpy()[0]
    ae_min = ae_error
    ae_max = ae_error
    ae_prob = (ae_error - ae_min) / (ae_max - ae_min + 1e-8)

    # Stack features for meta-learner
    stack = np.array([[if_prob, xgb_prob, rf_prob, ae_prob]])

    # Meta-learner prediction
    ensemble_prob = ensemble["meta_learner"].predict_proba(stack)[:, 1][0]
    ensemble_score = float(ensemble_prob)
    threshold = ensemble["threshold"]
    prediction = 1 if ensemble_prob >= threshold else 0
    return float(ensemble_score), int(prediction)