import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def build_roc_models(df: pd.DataFrame):
    """
    Builds logistic regression models for:
    1. Offensive efficiency only
    2. Defensive efficiency only

    Returns:
        dict with trained models, test data, predictions, and AUC scores
    """

    df = df.copy()

    # Keep only needed columns
    data = df[
        ["AdjOE",
         "AdjDE",
         "Final Four?"]
    ].dropna()

    X_off = data[["AdjOE"]]
    X_def = data[["AdjDE"]]
    y = data["Final Four?"]

    # Same split logic for fairness
    train_idx, test_idx = train_test_split(data.index, test_size=0.3, random_state=42)

    X_off_train, X_off_test = X_off.loc[train_idx], X_off.loc[test_idx]
    X_def_train, X_def_test = X_def.loc[train_idx], X_def.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    # Offensive model
    off_model = LogisticRegression()
    off_model.fit(X_off_train, y_train)
    off_probs = off_model.predict_proba(X_off_test)[:, 1]

    # Defensive model
    def_model = LogisticRegression()
    def_model.fit(X_def_train, y_train)
    def_probs = def_model.predict_proba(X_def_test)[:, 1]

    # ROC + AUC
    fpr_off, tpr_off, _ = roc_curve(y_test, off_probs)
    fpr_def, tpr_def, _ = roc_curve(y_test, def_probs)

    auc_off = roc_auc_score(y_test, off_probs)
    auc_def = roc_auc_score(y_test, def_probs)

    return {
        "y_test": y_test,

        "off_model": off_model,
        "def_model": def_model,

        "off_probs": off_probs,
        "def_probs": def_probs,

        "fpr_off": fpr_off,
        "tpr_off": tpr_off,

        "fpr_def": fpr_def,
        "tpr_def": tpr_def,

        "auc_off": auc_off,
        "auc_def": auc_def
    }


def plot_roc_comparison(results: dict, save_path = None):
    """
    Plots ROC curves comparing offensive vs defensive efficiency models.
    """

    fpr_off = results["fpr_off"]
    tpr_off = results["tpr_off"]
    auc_off = results["auc_off"]

    fpr_def = results["fpr_def"]
    tpr_def = results["tpr_def"]
    auc_def = results["auc_def"]

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr_off, tpr_off,
        label=f"Offense (AUC = {auc_off:.3f})"
    )

    plt.plot(
        fpr_def, tpr_def,
        label=f"Defense (AUC = {auc_def:.3f})"
    )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.title("ROC Comparison: Predicting Final Four Appearance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    