"""
EDITH Inference Engine
======================

Reusable anomaly detection inference module for the EDITH system.

Provides anomaly scoring and risk categorisation for application telemetry.
Supports Isolation Forest, XGBoost, Random Forest, Autoencoder, and Ensemble models.

Risk categories:
- LOW: Normal behaviour
- MEDIUM: Unusual behaviour
- HIGH: Suspicious behaviour
- CRITICAL: Multiple serious behavioural indicators
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

# Ensure src module is importable for Autoencoder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from train_autoencoder import Autoencoder

MODEL_DIR = os.path.join(BASE_DIR, "models")

RISK_THRESHOLDS = {
    "LOW": 0.0,
    "MEDIUM": 0.3,
    "HIGH": 0.6,
    "CRITICAL": 0.8,
}

RISK_DESCRIPTIONS = {
    "LOW": "Normal behaviour",
    "MEDIUM": "Unusual behaviour",
    "HIGH": "Suspicious behaviour",
    "CRITICAL": "Multiple serious behavioural indicators",
}