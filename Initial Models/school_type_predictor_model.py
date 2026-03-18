"""
school_type_predictor_model.py

Reusable training pipeline for predicting School Type.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.dummy import DummyClassifier


INPUT_FILE = "merged_data.csv"
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


# -------------------------
# Model selection
# -------------------------
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


# -------------------------
# Data loading + preprocessing
# -------------------------
def load_and_prepare_data(input_file=INPUT_FILE):
    df = pd.read_csv(input_file)
    df = df.dropna(subset=[TARGET_COL])

    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Drop unwanted columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Encode categorical columns
    df = pd.get_dummies(df, columns=[c for c in ENCODE_COLS if c in df.columns])

    X = df.drop(columns=[TARGET_COL])
    X = X.select_dtypes(include="number")
    y = df[TARGET_COL]

    print(f"Features: {X.shape[1]} columns")
    print(f"Target classes: {sorted(y.unique())}")

    return X, y


# -------------------------
# Train/test split
# -------------------------
def split_data(X, y, test_size=0.2):
    return train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )


# -------------------------
# Train + evaluate
# -------------------------
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    # Baseline
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))

    print(f"\nBaseline accuracy: {baseline_acc:.4f}")

    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"\n=== Model Results ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, acc


# -------------------------
# Full pipeline (THIS is what main.py will call)
# -------------------------
def train_model_pipeline(
    model_type="logistic",
    input_file=INPUT_FILE,
    test_size=0.2,
    save_model=True
):
    X, y = load_and_prepare_data(input_file)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)

    model = get_model(model_type)
    model, acc = train_and_evaluate(model, X_train, X_test, y_train, y_test)

    if save_model:
        output_path = f"{model_type}_model.joblib"
        joblib.dump(model, output_path)
        print(f"Model saved to: {output_path}")

    return model, acc

