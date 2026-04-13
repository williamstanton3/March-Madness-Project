"""
seed_predictor_model.py

Handles training Seed prediction models and saving evaluation graphs.
Call `run_seed_models(df, output_folder)` from main.py.
"""

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    # Four Factors (offense)
    "eFGPct",
    "TOPct",
    "ORPct",
    "FTRate",

    # Shooting (offense)
    "Off2PtFG",
    "Off3PtFG",
    "OffFT",
    "FG2Pct",
    "FG3Pct",
    "FTPct",
    "FG3Rate",
    "BlockPct",

    # Shooting (defense)
    "Def2PtFG",
    "Def3PtFG",
    "DefFT",
    "OppFG2Pct",
    "OppFG3Pct",
    "OppFTPct",
    "OppFG3Rate",
    "OppBlockPct",

    # Playmaking
    "ARate",
    "StlRate",
    "OppARate",
    "OppStlRate",

    # Physical
    "AvgHeight",

    # Efficiency Stats
    "OE",
    "AdjOE",
    "DE",
    "AdjDE",
]

TARGET_COL = "Seed"


def _prepare_data(df: pd.DataFrame):
    """Filter to FEATURE_COLS + TARGET_COL, drop missing target rows."""
    df = df.copy()
    df = df.dropna(subset=[TARGET_COL])

    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[WARNING] These FEATURE_COLS were not found in the data and will be skipped: {missing}")

    X = df[available]
    y = df[TARGET_COL].astype(int)
    return X, y


def _build_metrics(y_test, y_pred) -> dict:
    return {
        "accuracy":               accuracy_score(y_test, y_pred),
        "precision":              precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":                 recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1":                     f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "confusion_matrix":       confusion_matrix(y_test, y_pred).tolist(),
        "classification_report":  classification_report(y_test, y_pred, zero_division=0),
    }


def _save_confusion_matrix(y_test, y_pred, model_name: str, output_folder: str):
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(y_test.unique())          # ← use actual seed values (1–16)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, xticks_rotation=45)
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    path = os.path.join(output_folder, f"cm_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()


def _save_metrics_bar(results: dict, output_folder: str):
    """Bar chart comparing accuracy / precision / recall / F1 across models."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    model_names = list(results.keys())

    data = {m: [results[name][m] for name in model_names] for m in metrics}
    x = range(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        bars = ax.bar([xi + offset for xi in x], data[metric], width, label=metric.capitalize())
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=7
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(model_names, rotation=15)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Seed Prediction")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_folder, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved comparison chart → {path}")


# ── Public entry point ────────────────────────────────────────────────────────

def run_seed_models(df: pd.DataFrame, output_folder: str, test_size: float = 0.2) -> dict:
    """
    Train four Seed-prediction models and save evaluation graphs.

    Parameters
    ----------
    df            : Clean March Madness DataFrame (must contain TARGET_COL + FEATURE_COLS).
    output_folder : Directory where PNG graphs are written (created if missing).
    test_size     : Fraction of data held out for evaluation (default 0.2).

    Returns
    -------
    dict  {model_name: {accuracy, precision, recall, f1,
                        confusion_matrix, classification_report}}
    """
    os.makedirs(output_folder, exist_ok=True)

    X, y = _prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    models = {
        "Baseline":          DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Random Forest":     RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = _build_metrics(y_test, y_pred)

        print(f"  Accuracy : {results[name]['accuracy']:.4f}")
        print(f"  F1 (wtd) : {results[name]['f1']:.4f}")

        _save_confusion_matrix(y_test, y_pred, name, output_folder)

    _save_metrics_bar(results, output_folder)

    return results