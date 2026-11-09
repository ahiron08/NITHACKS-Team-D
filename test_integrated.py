""""
EDITH Integrated Anomaly Detection Script
Loads all five pretrained EDITH V2 models for inference on telemetry data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import pandas as pd
import numpy as np
import joblib
import torch

from model_loader import (
    load_isolation_forest, load_xgboost, load_random_forest,
    load_autoencoder, load_ensemble, get_all_scalers,
)
from inference_engine import (
    preprocess_telemetry, run_isolation_forest, run_xgboost,
    run_random_forest, run_autoencoder, run_ensemble,
)
from risk_classifier import classify_risk

your_telemetry = {
    "requests_per_min": 150.0,        # Unusually high request rate
    "denied_requests": 50,            # Many denials
    "sensitive_access_denied": 20,    # Many sensitive denials
    "request_rate_deviation": 15.0,   # Large deviation
    "microphone_access_count": 10,    # Excessive mic access
    "location_access_count": 10,      # Excessive location access
    "payment_access_count": 5,        # Unusual payment access
    "camera_access_count": 8,         # Unusual camera access
    "file_access_count": 200,         # Excessive file access
    "file_modification_count": 100,   # Excessive modifications
    "background_time_ratio": 0.8,     # Mostly background activity
    "background_requests": 100,       # Many background requests
    "background_mic_access": 5,       # Background mic activity
    "background_location_access": 8,  # Background location access
    "background_camera_access": 6,    # Background camera access
    "permission_count": 30,           # Many permissions
    "sensitive_permission_count": 20, # Many sensitive permissions
    "permission_changes": 15,         # Frequent permission changes
    "network_connections": 200,       # Many network connections
    "data_transferred_mb": 1000.0,    # Excessive data transfer
    "transaction_count": 50,          # Many transactions
    "failed_transaction_count": 30,   # Many failed transactions
    "unconfirmed_transaction_count": 20, # Many unconfirmed transactions
    "avg_transaction_amount": 200.0,  # High average amount
    "total_transaction_amount": 8000.0 # High total amount
}



# LOAD MODEL ARTIFACTS
print("[1] TELEMETRY INPUT")
print(f"  Successfully loaded {len(your_telemetry)} raw telemetry features.")

scaler = get_all_scalers()

iforest_model, iforest_scaler, iforest_threshold, iforest_features = load_isolation_forest()
print(f"  Isolation Forest loaded: {type(iforest_model).__name__}")

xgb_model, xgb_scaler, xgb_threshold, xgb_features = load_xgboost()
print(f"  XGBoost loaded: {type(xgb_model).__name__}")

rf_model, rf_scaler, rf_threshold, rf_features = load_random_forest()
print(f"  Random Forest loaded: {type(rf_model).__name__}")

ae_model, ae_scaler, ae_threshold, ae_features = load_autoencoder()
print(f"  Autoencoder loaded: AutoNN (input_dim=47)")

ensemble = load_ensemble()
print(f"  Ensemble loaded: Stacking with LogisticRegression meta-learner")

# FEATURE ENGINEERING
print()
print("[2] FEATURE ENGINEERING")
df_telemetry = pd.DataFrame([your_telemetry])
print(f"  Raw features: {len(your_telemetry)}")
features = preprocess_telemetry(your_telemetry, scaler)
print(f"  Engineered features: {len(features) - len(your_telemetry)}")
print(f"  Total features: {len(features)}")

# MODEL INFERENCE
print()
print("[3] MODEL INFERENCE")

print("Isolation Forest")
if_score, if_pred = run_isolation_forest(iforest_model, features)
print(f"    Anomaly Score: {if_score:.4f}")
print(f"    Prediction: {'Anomaly' if if_pred == 1 else 'Normal'}")

print("XGBoost")
xgb_score, xgb_pred = run_xgboost(xgb_model, features)
print(f"    Anomaly Probability: {xgb_score:.4f}")
print(f"    Prediction: {'Anomaly' if xgb_pred == 1 else 'Normal'}")

print("Random Forest")
rf_score, rf_pred = run_random_forest(rf_model, features)
print(f"    Anomaly Probability: {rf_score:.4f}")
print(f"    Prediction: {'Anomaly' if rf_pred == 1 else 'Normal'}")

print("Autoencoder")
ae_error, ae_thresh, ae_pred = run_autoencoder(ae_model, features)
print(f"    Reconstruction Error: {ae_error:.4f}")
print(f"    Threshold: {ae_thresh:.4f}")
print(f"    Prediction: {'Anomaly' if ae_pred == 1 else 'Normal'}")

print("Final Ensemble")
ens_score, ens_pred = run_ensemble(ensemble, scaler, features)
print(f"    Ensemble Score: {ens_score:.4f}")
print(f"    Prediction: {'Anomaly' if ens_pred == 1 else 'Normal'}")

# RISK CLASSIFICATION
print()
print("[4] FINAL RISK ASSESSMENT")
risk_result = classify_risk(ens_score)
anomaly_detected = ens_pred == 1
print(f"    Risk Level: {risk_result['category']}")
print(f"    Anomaly Detected: {anomaly_detected}")

# OUTPUT FORMAT
print()
print("=" * 60)
print("                  EDITH ANOMALY DETECTOR")
print("=" * 60)

print("\n[1] TELEMETRY INPUT")
print(f"  Successfully loaded {len(your_telemetry)} raw telemetry features.")

print("\n[2] FEATURE ENGINEERING")
print(f"  Raw features: {len(your_telemetry)}")
print(f"  Engineered features: {len(features) - len(your_telemetry)}")
print(f"  Total features: {len(features)}")

print("\n[3] MODEL INFERENCE")

print("Isolation Forest")
print(f"    Anomaly Score: {if_score:.4f}")
print(f"    Prediction: {'Anomaly' if if_pred == 1 else 'Normal'}")

print("XGBoost")
print(f"    Anomaly Probability: {xgb_score:.4f}")
print(f"    Prediction: {'Anomaly' if xgb_pred == 1 else 'Normal'}")

print("Random Forest")
print(f"    Anomaly Probability: {rf_score:.4f}")
print(f"    Prediction: {'Anomaly' if rf_pred == 1 else 'Normal'}")

print("Autoencoder")
print(f"    Reconstruction Error: {ae_error:.4f}")
print(f"    Threshold: {ae_thresh:.4f}")
print(f"    Prediction: {'Anomaly' if ae_pred == 1 else 'Normal'}")

print("Final Ensemble")
print(f"    Ensemble Score: {ens_score:.4f}")
print(f"    Prediction: {'Anomaly' if ens_pred == 1 else 'Normal'}")

print("\n[4] FINAL RISK ASSESSMENT")
print(f"    Risk Level: {risk_result['category']}")
print(f"    Anomaly Detected: {anomaly_detected}")

print()
print("=" * 60)
print("  SCRIPT COMPLETE")
print("=" * 60)