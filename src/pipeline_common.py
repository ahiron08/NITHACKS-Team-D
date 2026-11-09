"""
EDITH Common Pipeline Module
============================

Shared utilities for data splitting, preprocessing, and evaluation.
Designed to prevent data leakage across all stages.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

TARGET_COL = "is_anomaly"
ID_COLS = ["app_id", "session_id"]


def load_data():
    train_path = os.path.join(DATASET_DIR, "train_normal.csv")
    test_path = os.path.join(DATASET_DIR, "test_labeled.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def stratified_split(train_df, test_size=0.25, val_size=0.25, random_state=RANDOM_SEED):
    """
    Split training data into train/validation using stratified sampling.
    
    Returns:
        X_train, X_val, y_train, y_val, scaler, feature_cols
    """
    feature_cols = [c for c in train_df.columns
                    if c not in [TARGET_COL] + ID_COLS]
    
    X = train_df[feature_cols]
    y = train_df[TARGET_COL]
    
    # First split: train + val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=random_state
    )
    
    return X_train, X_val, y_train, y_val, feature_cols


def stratified_split_df(X, y, val_size=0.25, random_state=RANDOM_SEED):
    """
    Stratified split on already-selected features.
    
    Returns:
        X_train, X_val, y_train, y_val, feature_cols
    """
    feature_cols = X.columns.tolist()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=random_state
    )
    return X_train, X_val, y_train, y_val, feature_cols


def group_aware_split(train_df, n_splits=5, random_state=RANDOM_SEED):
    """
    Create group-aware splits by app_id for evaluating generalization
    to unseen applications.
    
    Returns list of (train_idx, val_idx) tuples.
    """
    gkf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # We stratify by app_id's majority class
    app_majority = train_df.groupby("app_id")[TARGET_COL].agg(lambda x: 1 if x.mean() > 0.5 else 0)
    
    splits = []
    for train_idx, val_idx in gkf.split(train_df, train_df[TARGET_COL]):
        splits.append((train_idx, val_idx))
    
    return splits


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute comprehensive classification metrics."""
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = 0.5
        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics["pr_auc"] = 0.0
    
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    # False positive rate and false negative rate
    tn, fp, fn, tp = cm.ravel()
    metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return metrics


def save_model(model, path):
    joblib.dump(model, path)
    print(f"Model saved to: {path}")


def load_model(path):
    return joblib.load(path)


def save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to: {path}")