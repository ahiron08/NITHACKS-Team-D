"""
EDITH Dataset Generation Script
================================

Generates synthetic application behaviour telemetry for training and testing
anomaly detection models.

Design principles:
- No feature directly encodes the is_anomaly target.
- Anomalies arise from unusual *combinations* of behavioural features,
  not from a single leaked column.
- Normal and anomalous records are generated from distinct but overlapping
  distributions so the task is learnable but not trivial.
- denied_requests and sensitive_access_denied are correlated but not redundant:
  * denied_requests = all denied requests (any resource)
  * sensitive_access_denied = denials of sensitive resources only (subset)
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

APP_CATEGORIES = {
    "productivity": {
        "requests_per_min": (5, 30),
        "denied_requests": (0, 3),
        "sensitive_access_denied": (0, 1),
        "mic_access_count": (0, 2),
        "camera_access_count": (0, 1),
        "location_access_count": (0, 2),
        "payment_access_count": (0, 1),
        "file_access_count": (10, 100),
        "file_modification_count": (1, 20),
        "background_time_ratio": (0.1, 0.4),
        "permission_count": (5, 15),
        "sensitive_permission_count": (1, 4),
        "permission_changes": (0, 3),
        "network_connections": (5, 50),
        "data_transferred_mb": (10, 500),
        "transaction_count": (0, 5),
        "failed_transaction_count": (0, 1),
        "unconfirmed_transaction_count": (0, 2),
        "avg_transaction_amount": (10, 100),
    },
    "communication": {
        "requests_per_min": (20, 100),
        "denied_requests": (0, 5),
        "sensitive_access_denied": (0, 2),
        "mic_access_count": (5, 50),
        "camera_access_count": (2, 20),
        "location_access_count": (0, 5),
        "payment_access_count": (0, 1),
        "file_access_count": (5, 50),
        "file_modification_count": (0, 5),
        "background_time_ratio": (0.3, 0.7),
        "permission_count": (8, 20),
        "sensitive_permission_count": (2, 6),
        "permission_changes": (0, 5),
        "network_connections": (20, 200),
        "data_transferred_mb": (100, 2000),
        "transaction_count": (0, 3),
        "failed_transaction_count": (0, 1),
        "unconfirmed_transaction_count": (0, 1),
        "avg_transaction_amount": (5, 50),
    },
    "finance": {
        "requests_per_min": (10, 60),
        "denied_requests": (0, 2),
        "sensitive_access_denied": (0, 1),
        "mic_access_count": (0, 1),
        "camera_access_count": (0, 1),
        "location_access_count": (1, 10),
        "payment_access_count": (5, 50),
        "file_access_count": (20, 200),
        "file_modification_count": (5, 50),
        "background_time_ratio": (0.05, 0.2),
        "permission_count": (10, 25),
        "sensitive_permission_count": (3, 8),
        "permission_changes": (0, 2),
        "network_connections": (15, 100),
        "data_transferred_mb": (50, 1000),
        "transaction_count": (10, 100),
        "failed_transaction_count": (0, 3),
        "unconfirmed_transaction_count": (0, 5),
        "avg_transaction_amount": (50, 500),
    },
    "utility": {
        "requests_per_min": (2, 20),
        "denied_requests": (0, 2),
        "sensitive_access_denied": (0, 1),
        "mic_access_count": (0, 1),
        "camera_access_count": (0, 1),
        "location_access_count": (0, 3),
        "payment_access_count": (0, 1),
        "file_access_count": (1, 30),
        "file_modification_count": (0, 10),
        "background_time_ratio": (0.5, 0.9),
        "permission_count": (3, 10),
        "sensitive_permission_count": (0, 3),
        "permission_changes": (0, 1),
        "network_connections": (2, 30),
        "data_transferred_mb": (5, 200),
        "transaction_count": (0, 2),
        "failed_transaction_count": (0, 0),
        "unconfirmed_transaction_count": (0, 1),
        "avg_transaction_amount": (0, 20),
    },
    "social_media": {
        "requests_per_min": (30, 150),
        "denied_requests": (1, 10),
        "sensitive_access_denied": (0, 3),
        "mic_access_count": (1, 10),
        "camera_access_count": (1, 15),
        "location_access_count": (2, 15),
        "payment_access_count": (0, 5),
        "file_access_count": (10, 100),
        "file_modification_count": (0, 10),
        "background_time_ratio": (0.4, 0.8),
        "permission_count": (10, 30),
        "sensitive_permission_count": (4, 12),
        "permission_changes": (1, 8),
        "network_connections": (50, 500),
        "data_transferred_mb": (200, 5000),
        "transaction_count": (0, 10),
        "failed_transaction_count": (0, 2),
        "unconfirmed_transaction_count": (0, 3),
        "avg_transaction_amount": (5, 100),
    },
}

SENSITIVE_RESOURCES = [
    "microphone", "camera", "location", "payment",
    "contacts", "calendar", "storage", "biometrics",
]

ALL_FEATURES = [
    "app_id", "session_id", "hour_of_day", "day_of_week",
    "requests_per_min", "denied_requests", "sensitive_access_denied",
    "request_rate_deviation", "microphone_access_count",
    "location_access_count", "payment_access_count",
    "camera_access_count", "file_access_count",
    "file_modification_count", "background_time_ratio",
    "background_requests", "background_mic_access",
    "background_location_access", "background_camera_access",
    "permission_count", "sensitive_permission_count",
    "permission_changes", "network_connections",
    "data_transferred_mb", "transaction_count",
    "failed_transaction_count", "unconfirmed_transaction_count",
    "avg_transaction_amount", "total_transaction_amount",
    "is_anomaly",
]

NUMERIC_FEATURES = [f for f in ALL_FEATURES
                    if f not in ("app_id", "session_id", "hour_of_day",
                                 "day_of_week", "is_anomaly")]


def generate_app_records(n_records, category, anomaly_prob=0.0,
                         anomaly_strength=1.0):
    profile = APP_CATEGORIES[category]
    n_apps = max(1, n_records // 20)
    app_ids = [f"{category}_app_{i:03d}" for i in range(n_apps)]
    records = []
    for _ in range(n_records):
        app_id = np.random.choice(app_ids)
        session_id = f"sess_{np.random.randint(100000, 999999)}"
        hour_of_day = np.random.randint(0, 24)
        day_of_week = np.random.randint(0, 7)
        is_anomaly = 1 if np.random.random() < anomaly_prob else 0
        records.append(_build_record(profile, app_id, session_id,
                                     hour_of_day, day_of_week,
                                     is_anomaly, anomaly_strength))
    return pd.DataFrame(records)


def _build_record(profile, app_id, session_id, hour_of_day,
                  day_of_week, is_anomaly, anomaly_strength):
    def sample(lo, hi):
        val = np.random.uniform(lo, hi)
        if is_anomaly:
            val = lo + (val - lo) * (1 + anomaly_strength * np.random.uniform(0, 2))
            val = min(val, hi * 3)
        return val

    requests_per_min = sample(*profile["requests_per_min"])
    denied_requests = int(sample(*profile["denied_requests"]))
    sensitive_access_denied = int(
        min(sample(*profile["sensitive_access_denied"]), denied_requests))
    mic_access_count = int(sample(*profile["mic_access_count"]))
    camera_access_count = int(sample(*profile["camera_access_count"]))
    location_access_count = int(sample(*profile["location_access_count"]))
    payment_access_count = int(sample(*profile["payment_access_count"]))
    file_access_count = int(sample(*profile["file_access_count"]))
    file_modification_count = int(
        min(sample(*profile["file_modification_count"]), file_access_count))
    background_time_ratio = np.clip(
        sample(*profile["background_time_ratio"]), 0, 1)
    permission_count = int(sample(*profile["permission_count"]))
    sensitive_permission_count = int(
        min(sample(*profile["sensitive_permission_count"]), permission_count))
    permission_changes = int(sample(*profile["permission_changes"]))
    network_connections = int(sample(*profile["network_connections"]))
    data_transferred_mb = sample(*profile["data_transferred_mb"])
    transaction_count = int(sample(*profile["transaction_count"]))
    failed_transaction_count = int(
        min(sample(*profile["failed_transaction_count"]), transaction_count))
    unconfirmed_transaction_count = int(
        min(sample(*profile["unconfirmed_transaction_count"]), transaction_count))
    avg_transaction_amount = sample(*profile["avg_transaction_amount"])
    total_transaction_amount = transaction_count * avg_transaction_amount

    background_requests = int(requests_per_min * background_time_ratio * 0.1)
    background_mic_access = int(mic_access_count * background_time_ratio * 0.3)
    background_location_access = int(location_access_count * background_time_ratio * 0.3)
    background_camera_access = int(camera_access_count * background_time_ratio * 0.3)

    cat_lo, cat_hi = profile["requests_per_min"]
    cat_mid = (cat_lo + cat_hi) / 2
    request_rate_deviation = abs(requests_per_min - cat_mid) / cat_mid

    rec = {
        "requests_per_min": requests_per_min,
        "denied_requests": denied_requests,
        "sensitive_access_denied": sensitive_access_denied,
        "microphone_access_count": mic_access_count,
        "camera_access_count": camera_access_count,
        "location_access_count": location_access_count,
        "payment_access_count": payment_access_count,
        "file_access_count": file_access_count,
        "file_modification_count": file_modification_count,
        "background_time_ratio": background_time_ratio,
        "background_requests": background_requests,
        "background_mic_access": background_mic_access,
        "background_location_access": background_location_access,
        "background_camera_access": background_camera_access,
        "permission_count": permission_count,
        "sensitive_permission_count": sensitive_permission_count,
        "permission_changes": permission_changes,
        "network_connections": network_connections,
        "data_transferred_mb": data_transferred_mb,
        "transaction_count": transaction_count,
        "failed_transaction_count": failed_transaction_count,
        "unconfirmed_transaction_count": unconfirmed_transaction_count,
        "avg_transaction_amount": avg_transaction_amount,
        "total_transaction_amount": total_transaction_amount,
    }

    if is_anomaly:
        _apply_anomaly_pattern(rec, profile, cat_mid, anomaly_strength)

    return {
        "app_id": app_id, "session_id": session_id,
        "hour_of_day": hour_of_day, "day_of_week": day_of_week,
        "requests_per_min": round(rec["requests_per_min"], 2),
        "denied_requests": rec["denied_requests"],
        "sensitive_access_denied": rec["sensitive_access_denied"],
        "request_rate_deviation": round(request_rate_deviation, 4),
        "microphone_access_count": rec["microphone_access_count"],
        "location_access_count": rec["location_access_count"],
        "payment_access_count": rec["payment_access_count"],
        "camera_access_count": rec["camera_access_count"],
        "file_access_count": rec["file_access_count"],
        "file_modification_count": rec["file_modification_count"],
        "background_time_ratio": round(rec["background_time_ratio"], 4),
        "background_requests": rec["background_requests"],
        "background_mic_access": rec["background_mic_access"],
        "background_location_access": rec["background_location_access"],
        "background_camera_access": rec["background_camera_access"],
        "permission_count": rec["permission_count"],
        "sensitive_permission_count": rec["sensitive_permission_count"],
        "permission_changes": rec["permission_changes"],
        "network_connections": rec["network_connections"],
        "data_transferred_mb": round(rec["data_transferred_mb"], 2),
        "transaction_count": rec["transaction_count"],
        "failed_transaction_count": rec["failed_transaction_count"],
        "unconfirmed_transaction_count": rec["unconfirmed_transaction_count"],
        "avg_transaction_amount": round(rec["avg_transaction_amount"], 2),
        "total_transaction_amount": round(rec["total_transaction_amount"], 2),
        "is_anomaly": is_anomaly,
    }


def _apply_anomaly_pattern(rec, profile, cat_mid, anomaly_strength):
    anomaly_type = np.random.choice(
        ["burst_requests", "excessive_denials",
         "background_sensor_abuse", "payment_anomaly",
         "permission_abuse", "file_abuse"])
    if anomaly_type == "burst_requests":
        rec["requests_per_min"] *= np.random.uniform(3, 8)
        cat_lo, cat_hi = profile["requests_per_min"]
        cat_mid2 = (cat_lo + cat_hi) / 2
        rec["request_rate_deviation"] = abs(rec["requests_per_min"] - cat_mid2) / cat_mid2
    elif anomaly_type == "excessive_denials":
        rec["denied_requests"] = int(rec["denied_requests"] * np.random.uniform(5, 15))
        rec["sensitive_access_denied"] = int(
            min(rec["sensitive_access_denied"] * np.random.uniform(3, 10),
                rec["denied_requests"]))
    elif anomaly_type == "background_sensor_abuse":
        rec["background_mic_access"] = int(
            rec["microphone_access_count"] * np.random.uniform(2, 5))
        rec["background_camera_access"] = int(
            rec["camera_access_count"] * np.random.uniform(2, 5))
        rec["background_location_access"] = int(
            rec["location_access_count"] * np.random.uniform(2, 5))
    elif anomaly_type == "payment_anomaly":
        rec["payment_access_count"] = int(
            rec["payment_access_count"] * np.random.uniform(5, 20))
        rec["transaction_count"] = int(
            rec["transaction_count"] * np.random.uniform(3, 10))
        rec["failed_transaction_count"] = int(
            min(rec["failed_transaction_count"] * np.random.uniform(5, 20),
                rec["transaction_count"]))
        rec["unconfirmed_transaction_count"] = int(
            min(rec["unconfirmed_transaction_count"] *
                np.random.uniform(3, 10), rec["transaction_count"]))
    elif anomaly_type == "permission_abuse":
        rec["sensitive_permission_count"] = int(
            min(rec["sensitive_permission_count"] *
                np.random.uniform(3, 8), rec["permission_count"]))
        rec["permission_changes"] = int(
            rec["permission_changes"] * np.random.uniform(5, 15))
    elif anomaly_type == "file_abuse":
        rec["file_modification_count"] = int(
            min(rec["file_modification_count"] *
                np.random.uniform(5, 15), rec["file_access_count"]))
        rec["data_transferred_mb"] *= np.random.uniform(3, 10)


def generate_datasets(n_normal=10000, n_test=4000,
                      anomaly_prob_train=0.15, anomaly_prob_test=0.30,
                      output_dir="datasets"):
    os.makedirs(output_dir, exist_ok=True)
    categories = list(APP_CATEGORIES.keys())
    records_per_cat = n_normal // len(categories)

    train_frames = []
    for cat in categories:
        df_cat = generate_app_records(
            records_per_cat, cat,
            anomaly_prob=anomaly_prob_train,
            anomaly_strength=1.0,
        )
        train_frames.append(df_cat)
    train_df = pd.concat(train_frames, ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    test_frames = []
    for cat in categories:
        df_cat = generate_app_records(
            n_test // len(categories), cat,
            anomaly_prob=anomaly_prob_test,
            anomaly_strength=1.2,
        )
        test_frames.append(df_cat)
    test_df = pd.concat(test_frames, ignore_index=True)
    test_df = test_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    train_path = os.path.join(output_dir, "train_normal.csv")
    test_path = os.path.join(output_dir, "test_labeled.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    metadata = {
        "generation_date": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "train_anomaly_rate": float(train_df["is_anomaly"].mean()),
        "test_anomaly_rate": float(test_df["is_anomaly"].mean()),
        "app_categories": categories,
        "features": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
    }
    with open(os.path.join(output_dir, "dataset_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Train dataset saved: {train_path} ({train_df.shape})")
    print(f"Test dataset saved:  {test_path} ({test_df.shape})")
    print(f"Train anomaly rate: {train_df['is_anomaly'].mean():.4f}")
    print(f"Test anomaly rate:  {test_df['is_anomaly'].mean():.4f}")

    return train_df, test_df


if __name__ == "__main__":
    generate_datasets()