
"""
EDITH Interactive Feature Engineering Tester
=============================================

Uses manually supplied telemetry.
Displays all 47 unscaled features.
Accepts a manual anomaly score for risk classification.

NOTE:
This script does not execute the ML models.
"""

import sys
from pathlib import Path

import pandas as pd

# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SRC_DIR))

from feature_engineering import (
    engineer_features,
    RAW_NUMERIC_FEATURES,
    ENGINEERED_FEATURES
)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("EDITH ANOMALY DETECTOR")
print("=" * 60)


# ============================================================
# TELEMETRY INPUT
# EDIT THESE VALUES DIRECTLY
# ============================================================

your_telemetry = {
    "requests_per_min": 25.0,
    "denied_requests": 2,
    "sensitive_access_denied": 1,
    "request_rate_deviation": 3.0,

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
    "total_transaction_amount": 150.0
}


# ============================================================
# INPUT VALIDATION
# ============================================================

print("\nTELEMETRY INPUT")
print("-" * 60)

missing = [
    feature
    for feature in RAW_NUMERIC_FEATURES
    if feature not in your_telemetry
]

if missing:
    print("Missing features:", missing)
    sys.exit(1)

df = pd.DataFrame([your_telemetry])

# Enforce training feature order
df = df[RAW_NUMERIC_FEATURES]

print(f"Raw features: {len(RAW_NUMERIC_FEATURES)}")
print(f"Non-zero raw features: {(df.iloc[0] != 0).sum()}")

print("\nInput values:")

for feature in RAW_NUMERIC_FEATURES:
    print(f"{feature}: {df.iloc[0][feature]}")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nFEATURE ENGINEERING RESULTS")
print("-" * 60)

try:

    result_df, scaler, feature_cols = engineer_features(
        df.copy(),
        fit_scaler=False
    )

    print(f"Raw features: {len(RAW_NUMERIC_FEATURES)}")
    print(f"Engineered features: {len(feature_cols)}")

    if len(feature_cols) != 47:
        raise ValueError(
            f"Expected 47 features, received {len(feature_cols)}"
        )

    print("\nFeature values:")

    for feature in feature_cols:
        value = result_df.iloc[0][feature]
        print(f"{feature}: {value:.4f}")

    # Validation
    values = result_df[feature_cols].iloc[0]

    nonzero_count = int((values != 0).sum())
    zero_count = int((values == 0).sum())

    print("\nFEATURE VALIDATION")
    print("-" * 60)
    print(f"Total features: {len(feature_cols)}")
    print(f"Non-zero features: {nonzero_count}")
    print(f"Zero features: {zero_count}")

    # Check raw features survived
    for feature in RAW_NUMERIC_FEATURES:

        original = df.iloc[0][feature]
        processed = result_df.iloc[0][feature]

        if original != processed:
            raise ValueError(
                f"Raw feature changed unexpectedly: {feature}"
            )

    if nonzero_count == 0:
        raise ValueError(
            "All features are zero despite non-zero input."
        )

except Exception as error:

    print("\nFEATURE ENGINEERING FAILED")
    print(f"Error: {error}")
    sys.exit(1)


# ============================================================
# LOAD RISK THRESHOLDS
# ============================================================

try:

    from edith_detector import (
        RISK_THRESHOLDS,
        RISK_DESCRIPTIONS
    )

    print("\nLoaded EDITH risk thresholds.")

except ImportError:

    print("\nUsing fallback risk thresholds.")

    RISK_THRESHOLDS = {
        "LOW": 0.0,
        "MEDIUM": 0.3,
        "HIGH": 0.6,
        "CRITICAL": 0.8
    }

    RISK_DESCRIPTIONS = {
        "LOW": "Normal application behaviour.",
        "MEDIUM": "Unusual application behaviour.",
        "HIGH": "Potentially suspicious application behaviour.",
        "CRITICAL": "Highly suspicious application behaviour."
    }


# ============================================================
# MANUAL ANOMALY SCORE
# ============================================================

print("\nANOMALY SCORE INPUT")
print("-" * 60)

while True:

    try:

        score_input = input(
            "Enter anomaly score (0-1, default 0.45): "
        ).strip()

        anomaly_score = (
            0.45 if score_input == ""
            else float(score_input)
        )

        if not 0 <= anomaly_score <= 1:
            print("Invalid input: Score must be between 0 and 1.")
            continue

        break

    except ValueError:

        print("Invalid input. Enter a number between 0 and 1.")


# ============================================================
# RISK CLASSIFICATION
# ============================================================

if anomaly_score >= RISK_THRESHOLDS["CRITICAL"]:
    risk = "CRITICAL"

elif anomaly_score >= RISK_THRESHOLDS["HIGH"]:
    risk = "HIGH"

elif anomaly_score >= RISK_THRESHOLDS["MEDIUM"]:
    risk = "MEDIUM"

else:
    risk = "LOW"


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\nRISK ASSESSMENT")
print("=" * 60)

print(f"Anomaly Score : {anomaly_score:.4f}")
print(f"Risk Level    : {risk}")

print(
    "Description   :",
    RISK_DESCRIPTIONS.get(risk, "No description available.")
)

print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)

print(
    "\nNOTE: The anomaly score was entered manually. "
    "No ML model prediction was performed."
)
