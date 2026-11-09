
"""
EDITH Feature Engineering Module
=================================

Converts raw application telemetry into 47 behavioural features.

Supports:
- Raw telemetry processing
- Engineered behavioural features
- Training-time scaler fitting
- Inference-time scaler transformation
- Unscaled feature inspection

IMPORTANT:
fit_scaler=None  -> Fit a new scaler (training only)
fit_scaler=False -> Do not scale (inspection only)
fit_scaler=obj   -> Use an existing fitted scaler (inference)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

EPSILON = 1e-8


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

RAW_NUMERIC_FEATURES = [
    "requests_per_min",
    "denied_requests",
    "sensitive_access_denied",
    "request_rate_deviation",
    "microphone_access_count",
    "location_access_count",
    "payment_access_count",
    "camera_access_count",
    "file_access_count",
    "file_modification_count",
    "background_time_ratio",
    "background_requests",
    "background_mic_access",
    "background_location_access",
    "background_camera_access",
    "permission_count",
    "sensitive_permission_count",
    "permission_changes",
    "network_connections",
    "data_transferred_mb",
    "transaction_count",
    "failed_transaction_count",
    "unconfirmed_transaction_count",
    "avg_transaction_amount",
    "total_transaction_amount",
]

ENGINEERED_FEATURES = [
    "bg_sensitive_access_ratio",
    "bg_request_ratio",
    "bg_mic_access",
    "bg_camera_access",
    "bg_location_access",
    "permission_to_request_ratio",
    "permission_change_frequency",
    "sensitive_permission_concentration",
    "n_sensitive_resources_accessed",
    "request_burst_ratio",
    "request_rate_per_active_session",
    "background_request_intensity",
    "request_rate_deviation_magnitude",
    "file_mod_to_access_ratio",
    "file_activity_intensity",
    "unconfirmed_transaction_ratio",
    "failed_transaction_ratio",
    "avg_transaction_amount_clipped",
    "transaction_activity_intensity",
    "combined_camera_mic_activity",
    "audio_video_access_mismatch",
    "sensitive_sensor_in_background",
]

ALL_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_divide(numerator, denominator, fill_value=0.0):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    result = np.zeros_like(numerator, dtype=float)

    valid = np.abs(denominator) >= EPSILON

    np.divide(
        numerator,
        denominator,
        out=result,
        where=valid
    )

    if fill_value is not None:
        result = np.where(valid, result, fill_value)

    return result


def clip_series(series, lower_pct=1, upper_pct=99):
    """
    Percentile clipping.

    A single observation cannot provide meaningful population
    percentiles, so it is returned unchanged.
    """
    if len(series) < 2:
        return series.copy()

    lower = series.quantile(lower_pct / 100.0)
    upper = series.quantile(upper_pct / 100.0)

    return series.clip(lower=lower, upper=upper)


# ============================================================
# BACKGROUND ACTIVITY FEATURES
# ============================================================

def add_background_activity_features(df):

    df = df.copy()

    df["bg_sensitive_access_ratio"] = safe_divide(
        df["sensitive_access_denied"],
        df["denied_requests"],
        fill_value=0.0
    )

    df["bg_sensitive_access_ratio"] = (
        df["bg_sensitive_access_ratio"].clip(0, 1)
    )

    df["bg_request_ratio"] = safe_divide(
        df["background_requests"],
        df["requests_per_min"],
        fill_value=0.0
    )

    df["bg_request_ratio"] = (
        df["bg_request_ratio"].clip(0, 1)
    )

    df["bg_mic_access"] = df["background_mic_access"].astype(float)

    df["bg_camera_access"] = df["background_camera_access"].astype(float)

    df["bg_location_access"] = df["background_location_access"].astype(float)

    return df


# ============================================================
# PERMISSION BEHAVIOUR FEATURES
# ============================================================

def add_permission_behaviour_features(df):

    df = df.copy()

    df["permission_to_request_ratio"] = safe_divide(
        df["permission_count"],
        df["requests_per_min"],
        fill_value=0.0
    )

    df["permission_to_request_ratio"] = clip_series(
        df["permission_to_request_ratio"]
    )

    df["permission_change_frequency"] = safe_divide(
        df["permission_changes"],
        df["background_time_ratio"],
        fill_value=0.0
    )

    df["permission_change_frequency"] = clip_series(
        df["permission_change_frequency"]
    )

    df["sensitive_permission_concentration"] = safe_divide(
        df["sensitive_permission_count"],
        df["permission_count"],
        fill_value=0.0
    )

    df["sensitive_permission_concentration"] = (
        df["sensitive_permission_concentration"].clip(0, 1)
    )

    df["n_sensitive_resources_accessed"] = (
        df["sensitive_permission_count"].astype(float)
    )

    return df


# ============================================================
# NETWORK BEHAVIOUR FEATURES
# ============================================================

def add_network_behaviour_features(df):

    df = df.copy()

    df["request_burst_ratio"] = clip_series(
        df["request_rate_deviation"]
    )

    df["request_rate_per_active_session"] = safe_divide(
        df["requests_per_min"],
        1.0 - df["background_time_ratio"],
        fill_value=0.0
    )

    df["request_rate_per_active_session"] = clip_series(
        df["request_rate_per_active_session"]
    )

    df["background_request_intensity"] = safe_divide(
        df["background_requests"],
        df["background_time_ratio"],
        fill_value=0.0
    )

    df["background_request_intensity"] = clip_series(
        df["background_request_intensity"]
    )

    df["request_rate_deviation_magnitude"] = (
        df["request_rate_deviation"].abs()
    )

    return df


# ============================================================
# FILE BEHAVIOUR FEATURES
# ============================================================

def add_file_behaviour_features(df):

    df = df.copy()

    df["file_mod_to_access_ratio"] = safe_divide(
        df["file_modification_count"],
        df["file_access_count"],
        fill_value=0.0
    )

    df["file_mod_to_access_ratio"] = (
        df["file_mod_to_access_ratio"].clip(0, 1)
    )

    df["file_activity_intensity"] = safe_divide(
        (
            df["file_access_count"]
            + df["file_modification_count"]
        ),
        df["background_time_ratio"],
        fill_value=0.0
    )

    df["file_activity_intensity"] = clip_series(
        df["file_activity_intensity"]
    )

    return df


# ============================================================
# FINANCIAL BEHAVIOUR FEATURES
# ============================================================

def add_financial_behaviour_features(df):

    df = df.copy()

    df["unconfirmed_transaction_ratio"] = safe_divide(
        df["unconfirmed_transaction_count"],
        df["transaction_count"],
        fill_value=0.0
    )

    df["unconfirmed_transaction_ratio"] = (
        df["unconfirmed_transaction_ratio"].clip(0, 1)
    )

    df["failed_transaction_ratio"] = safe_divide(
        df["failed_transaction_count"],
        df["transaction_count"],
        fill_value=0.0
    )

    df["failed_transaction_ratio"] = (
        df["failed_transaction_ratio"].clip(0, 1)
    )

    df["avg_transaction_amount_clipped"] = clip_series(
        df["avg_transaction_amount"]
    )

    df["transaction_activity_intensity"] = safe_divide(
        df["transaction_count"],
        df["background_time_ratio"],
        fill_value=0.0
    )

    df["transaction_activity_intensity"] = clip_series(
        df["transaction_activity_intensity"]
    )

    return df


# ============================================================
# SENSOR BEHAVIOUR FEATURES
# ============================================================

def add_sensor_behaviour_features(df):

    df = df.copy()

    df["combined_camera_mic_activity"] = (
        df["camera_access_count"]
        + df["microphone_access_count"]
    ).astype(float)

    df["combined_camera_mic_activity"] = clip_series(
        df["combined_camera_mic_activity"]
    )

    df["audio_video_access_mismatch"] = np.abs(
        df["microphone_access_count"].astype(float)
        - df["camera_access_count"].astype(float)
    )

    df["sensitive_sensor_in_background"] = (
        df["background_mic_access"]
        + df["background_camera_access"]
        + df["background_location_access"]
    ).astype(float)

    return df


# ============================================================
# MAIN FEATURE ENGINEERING PIPELINE
# ============================================================

def engineer_features(df, fit_scaler=None):

    df = df.copy()

    # Validate input
    missing = [
        col for col in RAW_NUMERIC_FEATURES
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing raw telemetry features: {missing}"
        )

    # Ensure numeric input
    for col in RAW_NUMERIC_FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise"
        )

    if not np.isfinite(
        df[RAW_NUMERIC_FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError("Telemetry contains NaN or infinite values.")

    # Generate engineered features
    df = add_background_activity_features(df)
    df = add_permission_behaviour_features(df)
    df = add_network_behaviour_features(df)
    df = add_file_behaviour_features(df)
    df = add_financial_behaviour_features(df)
    df = add_sensor_behaviour_features(df)

    # Preserve exact feature order
    feature_df = df[ALL_FEATURES].copy()

    # --------------------------------------------------------
    # SCALING OPTIONS
    # --------------------------------------------------------

    if fit_scaler is False:

        # Inspection only: preserve original feature values.
        scaler = None

        feature_values = feature_df.to_numpy(dtype=float)

    elif fit_scaler is None:

        # Training only: fit on the complete training dataset.
        if len(feature_df) < 2:
            raise ValueError(
                "Cannot fit RobustScaler on a single sample. "
                "Provide the complete training dataset or use "
                "fit_scaler=False for inspection."
            )

        scaler = RobustScaler()

        feature_values = scaler.fit_transform(feature_df)

    else:

        # Inference: use the scaler fitted during training.
        scaler = fit_scaler

        if not hasattr(scaler, "center_"):
            raise ValueError(
                "Provided scaler does not appear to be fitted."
            )

        feature_values = scaler.transform(feature_df)

    # Construct transformed DataFrame
    transformed_df = pd.DataFrame(
        feature_values,
        columns=ALL_FEATURES,
        index=df.index
    )

    # Return original telemetry + transformed feature columns
    for col in ALL_FEATURES:
        df[col] = transformed_df[col]

    return df, scaler, ALL_FEATURES


# ============================================================
# FEATURE SCHEMA
# ============================================================

def get_feature_schema():

    return {
        "raw_numeric_features": RAW_NUMERIC_FEATURES,
        "engineered_features": ENGINEERED_FEATURES,
        "total_features": len(ALL_FEATURES),
        "feature_descriptions": {
            "bg_sensitive_access_ratio":
                "Sensitive access denials divided by denied requests",

            "bg_request_ratio":
                "Background requests divided by total requests",

            "bg_mic_access":
                "Background microphone access count",

            "bg_camera_access":
                "Background camera access count",

            "bg_location_access":
                "Background location access count",

            "permission_to_request_ratio":
                "Permissions divided by requests per minute",

            "permission_change_frequency":
                "Permission changes normalized by background time",

            "sensitive_permission_concentration":
                "Fraction of permissions that are sensitive",

            "n_sensitive_resources_accessed":
                "Number of sensitive permissions",

            "request_burst_ratio":
                "Request rate deviation",

            "request_rate_per_active_session":
                "Requests normalized by active session time",

            "background_request_intensity":
                "Background requests normalized by background time",

            "request_rate_deviation_magnitude":
                "Absolute request rate deviation",

            "file_mod_to_access_ratio":
                "File modifications divided by file accesses",

            "file_activity_intensity":
                "File accesses and modifications normalized by background time",

            "unconfirmed_transaction_ratio":
                "Fraction of unconfirmed transactions",

            "failed_transaction_ratio":
                "Fraction of failed transactions",

            "avg_transaction_amount_clipped":
                "Average transaction amount",

            "transaction_activity_intensity":
                "Transactions normalized by background time",

            "combined_camera_mic_activity":
                "Combined camera and microphone access count",

            "audio_video_access_mismatch":
                "Absolute difference between microphone and camera access",

            "sensitive_sensor_in_background":
                "Combined background microphone, camera and location access"
        }
    }
