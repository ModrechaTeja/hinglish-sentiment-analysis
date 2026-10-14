"""
Shared evaluation helpers used by BOTH the classical baselines (Phase 4)
and the transformer models (Phase 5), so every model's outputs land in
exactly the same format and Phase 6/7 can treat them uniformly.

Outputs per model (``slug`` = filesystem-safe model name):
    outputs/predictions/<slug>_test.csv   per-example test predictions
    outputs/predictions/<slug>_val.csv    per-example validation predictions
    outputs/results/<slug>_classification_report.csv
    outputs/figures/<slug>_confusion_matrix[_normalized].png
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

for _d in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR, PREDICTIONS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

LABELS = [0, 1, 2]
LABEL_NAMES = ["Negative", "Neutral", "Positive"]


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_splits():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    for df in (train, val, test):
        df["text"] = df["text"].fillna("").astype(str)
        df["text_clean"] = df["text_clean"].fillna("").astype(str)
    return train, val, test


def compute_metrics(y_true, y_pred) -> dict:
    """Macro-averaged metrics + per-class F1 (macro-F1 is the headline metric)."""
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_negative": per_class[0],
        "f1_neutral": per_class[1],
        "f1_positive": per_class[2],
    }


def print_metrics(metrics: dict, split_name: str):
    print(f"\n{split_name} Results")
    print("-" * 40)
    for k in ("accuracy", "precision", "recall", "macro_f1"):
        print(f"{k:<10}: {metrics[k]:.4f}")


def save_predictions(df: pd.DataFrame, y_pred, model_name: str, split: str, probs=None):
    """Save one row per example: id, text, gold label, prediction (+ class probs)."""
    out = pd.DataFrame({
        "id": df["id"].values,
        "text": df["text"].values,
        "text_clean": df["text_clean"].values,
        "label": df["label"].values,
        "pred": np.asarray(y_pred),
    })
    if probs is not None:
        probs = np.asarray(probs)
        for i, name in enumerate(LABEL_NAMES):
            out[f"prob_{name.lower()}"] = probs[:, i]
    out["correct"] = out["label"] == out["pred"]
    path = PREDICTIONS_DIR / f"{slugify(model_name)}_{split}.csv"
    out.to_csv(path, index=False, encoding="utf-8")
    return path


def save_confusion_and_report(y_true, y_pred, model_name: str):
    slug = slugify(model_name)

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES).plot(cmap="Blues")
    plt.title(f"{model_name} - Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{slug}_confusion_matrix.png", dpi=200)
    plt.close()

    # Row-normalized: shows per-class recall directly (important with 50/35/15 imbalance)
    cm_norm = confusion_matrix(y_true, y_pred, labels=LABELS, normalize="true")
    ConfusionMatrixDisplay(cm_norm, display_labels=LABEL_NAMES).plot(
        values_format=".2f", cmap="Blues"
    )
    plt.title(f"{model_name} - Test Confusion Matrix (row-normalized)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{slug}_confusion_matrix_normalized.png", dpi=200)
    plt.close()

    report = classification_report(
        y_true, y_pred, labels=LABELS, target_names=LABEL_NAMES,
        zero_division=0, output_dict=True,
    )
    pd.DataFrame(report).transpose().to_csv(
        RESULTS_DIR / f"{slug}_classification_report.csv"
    )


def results_row(model_name: str, family: str, val_m: dict, test_m: dict, **extra) -> dict:
    row = {"model": model_name, "family": family}
    for prefix, m in (("validation", val_m), ("test", test_m)):
        for k, v in m.items():
            row[f"{prefix}_{k}"] = v
    row.update(extra)
    return row
