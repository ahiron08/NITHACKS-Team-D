"""
EDITH Model Evaluation & Visualization
======================================

Compares all trained models and generates visualizations.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_common import (
    load_data, compute_metrics, MODEL_DIR, REPORT_DIR, TARGET_COL, ID_COLS,
    RANDOM_SEED,
)
from feature_engineering import engineer_features

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

FIGURE_DIR = os.path.join(REPORT_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default")


def load_all_metrics():
    metrics = {}
    for model_name in ["isolation_forest", "xgboost", "random_forest",
                       "autoencoder", "ensemble"]:
        path = os.path.join(REPORT_DIR, f"{model_name}_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                metrics[model_name] = json.load(f)
    return metrics


def create_comparison_table(metrics):
    rows = []
    for model_name, m in metrics.items():
        test_m = m.get("test_metrics", {})
        val_m = m.get("val_metrics", {})
        row = {
            "Model": model_name.replace("_", " ").title(),
            "Val_Accuracy": val_m.get("accuracy", 0),
            "Val_Precision": val_m.get("precision", 0),
            "Val_Recall": val_m.get("recall", 0),
            "Val_F1": val_m.get("f1", 0),
            "Val_ROC_AUC": val_m.get("roc_auc", 0),
            "Val_PR_AUC": val_m.get("pr_auc", 0),
            "Test_Accuracy": test_m.get("accuracy", 0),
            "Test_Precision": test_m.get("precision", 0),
            "Test_Recall": test_m.get("recall", 0),
            "Test_F1": test_m.get("f1", 0),
            "Test_ROC_AUC": test_m.get("roc_auc", 0),
            "Test_PR_AUC": test_m.get("pr_auc", 0),
            "Test_FPR": test_m.get("fpr", 0),
            "Test_FNR": test_m.get("fnr", 0),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric_comparison(metrics_df, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    models = metrics_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    ax = axes[0, 0]
    ax.bar(x - width/2, metrics_df["Val_Accuracy"], width, label="Validation", alpha=0.8)
    ax.bar(x + width/2, metrics_df["Test_Accuracy"], width, label="Test", alpha=0.8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)

    ax = axes[0, 1]
    ax.bar(x - width/2, metrics_df["Val_Precision"], width, label="Validation", alpha=0.8)
    ax.bar(x + width/2, metrics_df["Test_Precision"], width, label="Test", alpha=0.8)
    ax.set_ylabel("Precision")
    ax.set_title("Precision Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)

    ax = axes[1, 0]
    ax.bar(x - width/2, metrics_df["Val_Recall"], width, label="Validation", alpha=0.8)
    ax.bar(x + width/2, metrics_df["Test_Recall"], width, label="Test", alpha=0.8)
    ax.set_ylabel("Recall")
    ax.set_title("Recall Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)

    ax = axes[1, 1]
    ax.bar(x - width/2, metrics_df["Val_F1"], width, label="Validation", alpha=0.8)
    ax.bar(x + width/2, metrics_df["Test_F1"], width, label="Test", alpha=0.8)
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_roc_pr_curves(metrics_df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    models = metrics_df["Model"].tolist()
    x = np.arange(len(models))

    ax = axes[0]
    ax.bar(x, metrics_df["Test_ROC_AUC"], alpha=0.8, color="steelblue")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("ROC-AUC Comparison (Test Set)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0, 1)

    ax = axes[1]
    ax.bar(x, metrics_df["Test_PR_AUC"], alpha=0.8, color="darkorange")
    ax.set_ylabel("PR-AUC")
    ax.set_title("PR-AUC Comparison (Test Set)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrices(metrics, save_path):
    n_models = len(metrics)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for idx, (model_name, m) in enumerate(metrics.items()):
        test_m = m.get("test_metrics", {})
        cm = np.array(test_m.get("confusion_matrix", [[0, 0], [0, 0]]))
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Anomaly"],
                    yticklabels=["Normal", "Anomaly"])
        ax.set_title(f"{model_name.replace('_', ' ').title()}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_fpr_fnr(metrics_df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    models = metrics_df["Model"].tolist()
    x = np.arange(len(models))

    ax = axes[0]
    ax.bar(x, metrics_df["Test_FPR"], alpha=0.8, color="red")
    ax.set_ylabel("False Positive Rate")
    ax.set_title("FPR Comparison (Test Set)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0, max(metrics_df["Test_FPR"].max() * 1.2, 0.1))

    ax = axes[1]
    ax.bar(x, metrics_df["Test_FNR"], alpha=0.8, color="orange")
    ax.set_ylabel("False Negative Rate")
    ax.set_title("FNR Comparison (Test Set)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0, max(metrics_df["Test_FNR"].max() * 1.2, 0.1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_evaluation():
    print("=" * 60)
    print("Model Evaluation & Visualization")
    print("=" * 60)

    metrics = load_all_metrics()
    if not metrics:
        print("No metrics found. Train models first.")
        return

    comparison_df = create_comparison_table(metrics)
    print("\n" + "=" * 80)
    print("MODEL COMPARISON TABLE")
    print("=" * 80)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(comparison_df.round(4).to_string(index=False))

    comparison_path = os.path.join(REPORT_DIR, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison table saved to: {comparison_path}")

    print("\nGenerating visualizations...")
    plot_metric_comparison(comparison_df,
                           os.path.join(FIGURE_DIR, "metric_comparison.png"))
    plot_roc_pr_curves(comparison_df,
                       os.path.join(FIGURE_DIR, "roc_pr_auc.png"))
    plot_confusion_matrices(metrics,
                            os.path.join(FIGURE_DIR, "confusion_matrices.png"))
    plot_fpr_fnr(comparison_df,
                 os.path.join(FIGURE_DIR, "fpr_fnr.png"))

    report_lines = []
    report_lines.append("# EDITH Model Comparison Report\n")
    report_lines.append("\n## Performance Summary\n\n")
    report_lines.append(comparison_df.round(4).to_markdown(index=False))
    report_lines.append("\n\n## Analysis\n\n")

    best_idx = comparison_df["Test_F1"].idxmax()
    best_model = comparison_df.loc[best_idx, "Model"]
    best_f1 = comparison_df.loc[best_idx, "Test_F1"]
    report_lines.append(f"- **Best model by test F1**: {best_model} (F1={best_f1:.4f})\n")

    targets = {"accuracy": 0.90, "precision": 0.90, "recall": 0.90, "f1": 0.90}
    report_lines.append("\n### Target Achievement (Test Set)\n")
    report_lines.append("Targets: 90% accuracy, 90% precision, 90% recall, 90% F1\n\n")
    for _, row in comparison_df.iterrows():
        met = []
        for metric, target in targets.items():
            col = f"Test_{metric.title()}" if metric != "f1" else "Test_F1"
            val = row[col]
            status = "PASS" if val >= target else "FAIL"
            met.append(f"{metric.upper()}: {status} ({val:.3f})")
        report_lines.append(f"- {row['Model']}: {', '.join(met)}\n")

    report_path = os.path.join(REPORT_DIR, "model_comparison.md")
    with open(report_path, "w") as f:
        f.write("".join(report_lines))
    print(f"Report saved to: {report_path}")

    return comparison_df, metrics


if __name__ == "__main__":
    run_evaluation()