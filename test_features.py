"""Manual test - EDITH feature engineering and risk categorization"""
import sys
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, 'src')

from feature_engineering import engineer_features, RAW_NUMERIC_FEATURES
from edith_detector import RISK_THRESHOLDS, RISK_DESCRIPTIONS

# ==========================================
# 1. Prepare your telemetry data
# ==========================================
print("=" * 60)
print("EDITH MANUAL TEST - Feature Engineering")
print("=" * 60 + "\n")

your_telemetry = {
    "requests_per_min": 25.0,
    "denied_requests": 2,
    "sensitive_access_denied": 1,
    "request_rate_deviation": 5.0,
    "microphone_access_count": 0,
    "location_access_count": 1,
    "payment_access_count": 0,
    "camera_access_count": 0,
    "file_access_count": 50,
    "file_modification_count": 5,
    "background_time_ratio": 0.2,
    "background_requests": 10,
    "background_mic_access": 0,
    "background_location_access": 1,
    "background_camera_access": 0,
    "permission_count": 10,
    "sensitive_permission_count": 2,
    "permission_changes": 1,
    "network_connections": 15,
    "data_transferred_mb": 100.0,
    "transaction_count": 3,
    "failed_transaction_count": 0,
    "unconfirmed_transaction_count": 0,
    "avg_transaction_amount": 50.0,
    "total_transaction_amount": 150.0,
}

# ==========================================
# 2. Engineer features
# ==========================================
df = pd.DataFrame([your_telemetry])
df_eng, scaler, feature_cols = engineer_features(df, fit_scaler=None)

print(f"Original features: {len(your_telemetry)}")
print(f"Engineered features: {len(feature_cols)}")
print(f"\nFirst 10 engineered columns:")
for i, col in enumerate(feature_cols[:10]):
    val = df_eng[col].iloc[0]
    print(f"  {col}: {val:.4f}")

print(f"\nLast 5 engineered columns:")
for col in feature_cols[-5:]:
    val = df_eng[col].iloc[0]
    print(f"  {col}: {val:.4f}")

# ==========================================
# 3. Risk categorization demo
# ==========================================
print(f"\n" + "=" * 60)
print("RISK CATEGORIZATION")
print("=" * 60 + "\n")

# Use the last anomaly score as an example - in practice this comes from a model
# For demonstration, let's use a sample score
sample_score = 0.45  # This would come from detector.predict()

# Determine risk category
risk_category = "LOW"
for cat, threshold in sorted(RISK_THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
    if sample_score >= threshold:
        risk_category = cat
        break

risk_desc = RISK_DESCRIPTIONS.get(risk_category, "Unknown")

print(f"Anomaly Score: {sample_score:.4f}")
print(f"Risk Category: {risk_category}")
print(f"Risk Description: {risk_desc}")

print(f"\nThresholds:")
for cat, thr in RISK_THRESHOLDS.items():
    status = "✓" if sample_score >= thr else " "
    print(f"  {cat}: >= {thr:.1f} {status}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)