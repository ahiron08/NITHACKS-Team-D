"""
EDITH Autoencoder Training
==========================

Trains a fully connected autoencoder for anomaly detection using PyTorch.
- Trained initially on normal behavioural samples only.
- Uses reconstruction error as anomaly score.
- Threshold tuned on validation data.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_common import (
    load_data, stratified_split_df, compute_metrics, save_metrics,
    MODEL_DIR, REPORT_DIR, TARGET_COL, ID_COLS, RANDOM_SEED,
)
from feature_engineering import engineer_features

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dims=None):
        super(Autoencoder, self).__init__()
        if encoding_dims is None:
            encoding_dims = [max(32, input_dim // 2), max(16, input_dim // 4)]

        encoder_layers = []
        prev_dim = input_dim
        for dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.bottleneck_dim = prev_dim

        decoder_layers = []
        for dim in reversed(encoding_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder():
    print("=" * 60)
    print("Training Autoencoder")
    print("=" * 60)

    train_df, test_df = load_data()

    train_eng, scaler, feature_cols = engineer_features(train_df)
    test_eng, _, _ = engineer_features(test_df, fit_scaler=scaler)

    X_full = train_eng[feature_cols]
    y_full = train_eng[TARGET_COL]

    X_train, X_val, y_train, y_val, _ = stratified_split_df(
        X_full, y_full, val_size=0.25, random_state=RANDOM_SEED
    )

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    print(f"Train anomaly rate: {y_train.mean():.4f}")
    print(f"Val anomaly rate: {y_val.mean():.4f}")

    X_train_normal = X_train[y_train == 0].values.astype(np.float32)
    print(f"Normal-only training samples: {len(X_train_normal)}")

    input_dim = X_train_normal.shape[1]

    train_dataset = TensorDataset(
        torch.tensor(X_train_normal), torch.tensor(X_train_normal)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    X_val_all = X_val.values.astype(np.float32)
    y_val_all = y_val.values

    X_test_all = test_eng[feature_cols].values.astype(np.float32)
    y_test_all = test_df[TARGET_COL].values

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model = Autoencoder(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    n_epochs = 20
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.6f}")

    # Compute reconstruction errors on validation data
    model.eval()
    with torch.no_grad():
        val_tensor = torch.tensor(X_val_all)
        val_reconstructed = model(val_tensor)
        val_errors = torch.mean((val_reconstructed - val_tensor) ** 2, dim=1).numpy()

    # Find optimal threshold on validation data
    best_threshold = 0.0
    best_val_f1 = -1
    for threshold in np.percentile(val_errors, [50, 75, 85, 90, 95, 97, 99]):
        y_val_pred = (val_errors > threshold).astype(int)
        val_metrics = compute_metrics(y_val_all, y_val_pred, val_errors)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_threshold = threshold

    print(f"\nBest threshold: {best_threshold:.6f}")
    print(f"Best val F1: {best_val_f1:.4f}")

    # Validation metrics with best threshold
    y_val_pred = (val_errors > best_threshold).astype(int)
    val_metrics = compute_metrics(y_val_all, y_val_pred, val_errors)
    print(f"\nValidation metrics: {json.dumps({k: v for k, v in val_metrics.items() if k != 'confusion_matrix'}, indent=2)}")

    # Test set evaluation (held out)
    with torch.no_grad():
        test_tensor = torch.tensor(X_test_all)
        test_reconstructed = model(test_tensor)
        test_errors = torch.mean((test_reconstructed - test_tensor) ** 2, dim=1).numpy()

    y_test_pred = (test_errors > best_threshold).astype(int)
    test_metrics = compute_metrics(y_test_all, y_test_pred, test_errors)
    print(f"\nTest metrics: {json.dumps({k: v for k, v in test_metrics.items() if k != 'confusion_matrix'}, indent=2)}")

    # Save model, scaler, threshold, feature cols
    model_path = os.path.join(MODEL_DIR, "autoencoder.pth")
    scaler_path = os.path.join(MODEL_DIR, "autoencoder_scaler.pkl")
    threshold_path = os.path.join(MODEL_DIR, "autoencoder_threshold.json")
    features_path = os.path.join(MODEL_DIR, "autoencoder_features.json")

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
    }, model_path)
    joblib.dump(scaler, scaler_path)

    with open(threshold_path, "w") as f:
        json.dump({"threshold": float(best_threshold)}, f, indent=2)
    with open(features_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save metrics
    all_metrics = {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "threshold": float(best_threshold),
    }
    save_metrics(all_metrics, os.path.join(REPORT_DIR, "autoencoder_metrics.json"))

    print(f"\nAutoencoder training complete.")
    return model, scaler, best_threshold, feature_cols, val_metrics, test_metrics


if __name__ == "__main__":
    train_autoencoder()