"""
school_type_predictor_model.py

Handles merging data, training School Type models, and returning metrics including baseline.
"""

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier

# File and column settings
INPUT_FILE = "data/merged_data.csv"
TARGET_COL = "School Type"

DROP_COLS = [
    "Mapped ESPN Team Name",
    "Current Coach",
    "Full Team Name",
    "Region",
    "Post-Season Tournament"
]

ENCODE_COLS = [
    "Short Conference Name",
    "Mapped Conference Name"
]

# Model selection
def get_model(model_type):
    if model_type == "logistic":
        return LogisticRegression(max_iter=1000, C=1.0)
    elif model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_type == "svm":
        return SVC(C=1.0)
    else:
        raise ValueError(f"Unknown model: {model_type}")


# Load + preprocess merged data
def load_and_prepare_data(input_file=INPUT_FILE):
    df = pd.read_csv(input_file)
    df = df.dropna(subset=[TARGET_COL])
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = pd.get_dummies(df, columns=[c for c in ENCODE_COLS if c in df.columns])

    X = df.drop(columns=[TARGET_COL]).select_dtypes(include="number")
    y = df[TARGET_COL]
    return X, y


# Compute baseline metrics
def compute_baseline(X_train, X_test, y_train, y_test):
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)
    return build_metrics(y_test, y_pred)


# Build metrics dict from predictions
def build_metrics(y_test, y_pred):
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0)
    }


# Train models + return dictionary of metrics
def train_models_and_get_accuracies(model_types=None, input_file=INPUT_FILE, test_size=0.2, save_models=True):
    if model_types is None:
        model_types = ["logistic", "random_forest", "gradient_boosting", "svm"]

    # Ensure model folder exists
    model_dir = os.path.join(os.getcwd(), "output_models")
    if save_models and not os.path.exists(model_dir):
        os.makedirs(model_dir)

    X, y = load_and_prepare_data(input_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    results = {}
    results["baseline"] = compute_baseline(X_train, X_test, y_train, y_test)

    for model_type in model_types:
        model = get_model(model_type)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_type] = build_metrics(y_test, y_pred)

        if save_models:
            model_path = os.path.join(model_dir, f"{model_type}_model.joblib")
            joblib.dump(model, model_path)
            print(f"Saved {model_type} model to {model_path}")

    return results