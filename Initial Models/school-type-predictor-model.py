"""
train_model.py - Train a classification model on cleaned CSV data.

Usage:
    python train_model.py --model <model_type> [--test_size <float>]

Model choices: logistic, random_forest, gradient_boosting, svm
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Train a classification model.")
    parser.add_argument("--model", required=True,
                        choices=["logistic", "random_forest", "gradient_boosting", "svm"],
                        help="Model type to train")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data for testing (default: 0.2)")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=[TARGET_COL])
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Drop unwanted columns (ignore if not present)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=[c for c in ENCODE_COLS if c in df.columns])

    X = df.drop(columns=[TARGET_COL])
    X = X.select_dtypes(include="number")
    y = df[TARGET_COL]
    print(f"Features: {X.shape[1]} columns")
    print(f"Target classes: {sorted(y.unique())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # Baseline
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
    print(f"\nBaseline accuracy (most frequent class): {baseline_acc:.4f}")

    # Train model
    model = get_model(args.model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n=== {args.model} Results ===")
    print(f"Test accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    output_path = f"{args.model}_model.joblib"
    joblib.dump(model, output_path)
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    main()
