"""Manual test script for EDITH anomaly detector"""
import sys
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, 'src')

from feature_engineering import engineer_features, RAW_NUMERIC_FEATURES
from edith_detector import EdithDetector

# ==========================================
# 1. Prepare your telemetry data - example values
# ==========================================
your_telemetry = {
    "requests_per_min": 25.0,        # Normal request rate
    "denied_requests": 2,            # Slight denials
    "sensitive_access_denied": 1,    # One sensitive denial
    "request_rate_deviation": 5.0,   # Moderate deviation
    "microphone_access_count": 0,    # No mic access
    "location_access_count": 1,      # One location access
    "payment_access_count": 0,       # No payment access
    "camera_access_count": 0,        # No camera access
    "file_access_count": 50,         # Normal file access
    "file_modification_count": 5,    # Moderate modifications
    "background_time_ratio": 0.2,    # 20% background time
    "background_requests": 10,       # 10 background requests
    "background_mic_access": 0,      # No background mic
    "background_location_access": 1, # One background location
    "background_camera_access": 0,   # No background camera
    "permission_count": 10,          # 10 permissions
    "sensitive_permission_count": 2, # 2 sensitive permissions
    "permission_changes": 1,         # One permission change
    "network_connections": 15,       # 15 network connections
    "data_transferred_mb": 100.0,    # 100 MB data
    "transaction_count": 3,          # 3 transactions
    "failed_transaction_count": 0,   # No failed transactions
    "unconfirmed_transaction_count": 0, # No unconfirmed
    "avg_transaction_amount": 50.0,  # $50 avg
    "total_transaction_amount": 150.0, # $150 total
}

# ==========================================
# 2. Engineer features (same as training)
# ==========================================
df = pd.DataFrame([your_telemetry])
df_eng, scaler, feature_cols = engineer_features(df, fit_scaler=None)
print(f"Feature engineering: {len(feature_cols)} features engineered\n")

# ==========================================
# 3. Test the detector with all model types
# ==========================================
model_types = ['xgboost', 'isolation_forest', 'random_forest', 'autoencoder', 'ensemble']

print("=" * 60)
print("EDITH ANOMALY DETECTOR - MANUAL TEST")
print("=" * 60 + "\n")

for model_type in model_types:
    print(f"--- {model_type.upper()} MODEL ---")
    try:
        detector = EdithDetector(model_type=model_type)
        detector.load()
        result = detector.predict(your_telemetry)
        
        print(f"  is_anomaly:   {result['is_anomaly']}")
        print(f"  anomaly_score: {result['anomaly_score']:.4f}")
        print(f"  risk_category: {result['risk_category']}")
        print(f"  risk_description: {result['risk_description']}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()