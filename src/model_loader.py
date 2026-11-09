"""
EDITH Model Loader

Loads all five pretrained EDITH V2 model artifacts and their associated
preprocessing components (scalers, thresholds, feature columns).
"""

import os
import sys
import json
import joblib
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\src", "")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_isolation_forest():
    """Load the pretrained Isolation Forest model and artifacts."""
    model = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "isolation_forest_scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "isolation_forest_threshold.json")) as f:
        threshold = json.load(f)["threshold"]
    with open(os.path.join(MODEL_DIR, "isolation_forest_features.json")) as f:
        features = json.load(f)
    return model, scaler, threshold, features


def load_xgboost():
    """Load the pretrained XGBoost model and artifacts."""
    model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "xgboost_scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "xgboost_threshold.json")) as f:
        threshold = json.load(f)["threshold"]
    with open(os.path.join(MODEL_DIR, "xgboost_features.json")) as f:
        features = json.load(f)
    return model, scaler, threshold, features


def load_random_forest():
    """Load the pretrained Random Forest model and artifacts."""
    model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "random_forest_scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "random_forest_threshold.json")) as f:
        threshold = json.load(f)["threshold"]
    with open(os.path.join(MODEL_DIR, "random_forest_features.json")) as f:
        features = json.load(f)
    return model, scaler, threshold, features


def load_autoencoder():
    """Load the pretrained Autoencoder model and artifacts."""
    with open(os.path.join(MODEL_DIR, "autoencoder_features.json")) as f:
        features = json.load(f)
    with open(os.path.join(MODEL_DIR, "autoencoder_threshold.json")) as f:
        threshold = json.load(f)["threshold"]
    scaler = joblib.load(os.path.join(MODEL_DIR, "autoencoder_scaler.pkl"))
    checkpoint = torch.load(os.path.join(MODEL_DIR, "autoencoder.pth"), weights_only=False)
    from train_autoencoder import Autoencoder
    model = Autoencoder(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, scaler, threshold, features


def load_ensemble():
    """Load the pretrained Final Ensemble model and artifacts."""
    ensemble = joblib.load(os.path.join(MODEL_DIR, "final_ensemble.pkl"))
    return ensemble


def get_all_scalers():
    """Get all fitted scalers. All EDITH V2 scalers are identical RobustScaler."""
    scaler_path = os.path.join(MODEL_DIR, "isolation_forest_scaler.pkl")
    return joblib.load(scaler_path)