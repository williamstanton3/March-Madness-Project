import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE    

# Column settings
TARGET_COL = "Post-Season Tournament"

DROP_COLS = [
    "Mapped ESPN Team Name",
    "Current Coach",
    "Full Team Name",
    "Region",
    "Post-Season Tournament Sorting Index",
    "Tournament Winner?",
    "Tournament Championship?",
    "Final Four?",
    "Top 12 in AP Top 25 During Week 6?",
    "Seed"
]

ENCODE_COLS = [
    "Short Conference Name",
    "Mapped Conference Name"
]


# Model selection
def get_model(model_type):
    if model_type == "logistic":
        return LogisticRegression(C=1.0)
    elif model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_type == "svm":
        return SVC(C=1.0)
    else:
        raise ValueError(f"Unknown model: {model_type}")


# Prepare data from an already-cleaned DataFrame
def prepare_data(df: pd.DataFrame):
    """Takes the cleaned DataFrame and returns X, y ready for modeling."""
    df = df.copy()

    df = df.dropna(subset=[TARGET_COL])
    print(f"[ml_model] After dropping NA in target: {len(df)}")

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = pd.get_dummies(df, columns=[c for c in ENCODE_COLS if c in df.columns])

    X = df.drop(columns=[TARGET_COL]).select_dtypes(include="number")
    y = (df[TARGET_COL] == "March Madness").astype(int)

    print(f"[ml_model] Features: {X.shape[1]} | Target distribution: {y.value_counts().to_dict()}")
    return X, y


# Build metrics dict from predictions
def build_metrics(y_test, y_pred):
    return {
        "accuracy":               accuracy_score(y_test, y_pred),
        "precision":              precision_score(y_test, y_pred, average="binary", zero_division=0),
        "recall":                 recall_score(y_test, y_pred, average="binary", zero_division=0),
        "f1":                     f1_score(y_test, y_pred, average="binary", zero_division=0),
        "confusion_matrix":       confusion_matrix(y_test, y_pred).tolist(),
        "classification_report":  classification_report(y_test, y_pred, zero_division=0),
    }


# Compute baseline metrics
def compute_baseline(X_train, X_test, y_train, y_test):
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)
    return build_metrics(y_test, y_pred)


# Train models + apply SMOTE + return dictionary of metrics
def train_models_and_get_accuracies(df: pd.DataFrame, model_types=None, test_size=0.2, save_models=True, output_folder="output_models"):
    """
    Train ML models to predict March Madness qualification.

    Parameters
    ----------
    df            : Cleaned full DataFrame (all teams, not just MM teams).
    model_types   : List of model keys to train. Defaults to all four.
    test_size     : Fraction held out for evaluation.
    save_models   : Whether to persist trained models via joblib.
    output_folder : Directory for saved .joblib model files.

    Returns
    -------
    dict  {model_name: {accuracy, precision, recall, f1, confusion_matrix, classification_report}}
    """
    if model_types is None:
        model_types = ["logistic", "random_forest", "gradient_boosting", "svm"]

    if save_models and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    smote = SMOTE(sampling_strategy=0.50, k_neighbors=5, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    results = {}
    results["baseline"] = compute_baseline(X_train, X_test, y_train, y_test)

    for model_type in model_types:
        model = get_model(model_type)
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        results[model_type] = build_metrics(y_test, y_pred)

        print(f"  [{model_type}] Accuracy: {results[model_type]['accuracy']:.4f}  F1: {results[model_type]['f1']:.4f}")

        if save_models:
            model_path = os.path.join(output_folder, f"mm_qual_{model_type}_model.joblib")
            joblib.dump(model, model_path)
            print(f"  Saved {model_type} model → {model_path}")

    return results


# Plot comparison: Baseline vs GB (no SMOTE) vs GB (with SMOTE)
def plot_model_comparison(df: pd.DataFrame, test_size=0.2, save_path="Graphs/model_comparison_mm.png"):
    """
    Bar chart comparing Baseline, Gradient Boosting (no SMOTE), and
    Gradient Boosting (with SMOTE) across Accuracy / Precision / Recall / F1.
    """
    METRIC_KEYS   = ["accuracy", "precision", "recall", "f1"]
    METRIC_LABELS = ["Accuracy", "Precision", "Recall", "F1"]
    COLORS = {
        "baseline": "lightgray",
        "no_smote": "orange",
        "smote":    "royalblue",
    }

    def extract_scores(metrics):
        return [metrics[k] for k in METRIC_KEYS]

    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    baseline_scores = extract_scores(compute_baseline(X_train, X_test, y_train, y_test))

    gb_plain = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_plain.fit(X_train, y_train)
    plain_scores = extract_scores(build_metrics(y_test, gb_plain.predict(X_test)))

    smote = SMOTE(sampling_strategy=0.30, k_neighbors=5, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    gb_smote = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_smote.fit(X_res, y_res)
    smote_scores = extract_scores(build_metrics(y_test, gb_smote.predict(X_test)))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x     = np.arange(len(METRIC_LABELS))
    width = 0.25

    bars_b = ax.bar(x - width, baseline_scores, width, color=COLORS["baseline"], zorder=3, label="Baseline")
    bars_p = ax.bar(x,         plain_scores,    width, color=COLORS["no_smote"], zorder=3, label="Gradient Boosting (no SMOTE)")
    bars_s = ax.bar(x + width, smote_scores,    width, color=COLORS["smote"],    zorder=3, label="Gradient Boosting (SMOTE)")

    for bars in (bars_b, bars_p, bars_s):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.4f}", ha="center", va="bottom", fontsize=8, color="#333333"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, fontsize=11, color="#333333")
    ax.set_yticks(np.arange(0, 1.05, 0.1))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.05, 0.1)], fontsize=9, color="#333333")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11, color="#333333")
    ax.set_title(
        "Model Comparison: Baseline vs Gradient Boosting vs Gradient Boosting + SMOTE",
        fontsize=12, color="#222222", pad=14
    )
    ax.yaxis.grid(True, color="#cccccc", linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#333333", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

    ax.legend(
        handles=[
            mpatches.Patch(color=COLORS["baseline"], label="Baseline"),
            mpatches.Patch(color=COLORS["no_smote"], label="Gradient Boosting (no SMOTE)"),
            mpatches.Patch(color=COLORS["smote"],    label="Gradient Boosting (SMOTE)"),
        ],
        facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper right"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved model comparison → {save_path}")
    plt.close()


def plot_confusion_matrices(df: pd.DataFrame, test_size=0.2, save_path="Graphs/confusion_matrices_mm.png"):
    """
    Side-by-side confusion matrices for Baseline, GB (no SMOTE), and GB (with SMOTE).
    """
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_cm = confusion_matrix(y_test, baseline.predict(X_test))

    gb_plain = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_plain.fit(X_train, y_train)
    plain_cm = confusion_matrix(y_test, gb_plain.predict(X_test))

    smote = SMOTE(sampling_strategy=0.30, k_neighbors=5, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    gb_smote = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_smote.fit(X_res, y_res)
    smote_cm = confusion_matrix(y_test, gb_smote.predict(X_test))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("white")

    matrices = [
        ("Baseline",      baseline_cm),
        ("GB (no SMOTE)", plain_cm),
        ("GB (SMOTE)",    smote_cm),
    ]

    for ax, (title, cm) in zip(axes, matrices):
        ax.set_facecolor("white")
        ax.imshow(cm, cmap="Blues", aspect="auto")

        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold"
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Not MM", "March Madness"], fontsize=10)
        ax.set_yticklabels(["Not MM", "March Madness"], fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", color="#222222", pad=10)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved confusion matrices → {save_path}")
    plt.close()