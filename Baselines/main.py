import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

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


def load_and_prepare_data(df):
    df = df.dropna(subset=[TARGET_COL])
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = pd.get_dummies(df, columns=[c for c in ENCODE_COLS if c in df.columns])

    X = df.drop(columns=[TARGET_COL]).select_dtypes(include="number")
    y = df[TARGET_COL]
    return X, y


def build_metrics(y_test, y_pred):
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0)
    }


def baseline_school_type_most_frequent(X_train, X_test, y_train, y_test):
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return build_metrics(y_test, y_pred)


def baseline_school_type_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return build_metrics(y_test, y_pred)


def print_results(name, metrics):
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"\nClassification Report:\n{metrics['classification_report']}")


def main():
    # Load data
    clean_df = pd.read_csv("data/merged_data.csv")

    X, y = load_and_prepare_data(clean_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # School type predictor baseline
    baseline_metrics = baseline_school_type_most_frequent(X_train, X_test, y_train, y_test)
    print_results("Baseline (Most Frequent Class)", baseline_metrics)

    # Logistic regression
    logistic_metrics = baseline_school_type_logistic_regression(X_train, X_test, y_train, y_test)
    print_results("Baseline (Logistic Regression)", logistic_metrics)

    # Summary comparison
    print(f"\n{'='*40}")
    print("  Summary Comparison")
    print(f"{'='*40}")
    print(f"  {'Model':<30} {'Accuracy':>10} {'F1':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Baseline (Most Frequent Class)':<30} {baseline_metrics['accuracy']:>10.4f} {baseline_metrics['f1']:>10.4f}")
    print(f"  {'Baseline (Logistic Regression)':<30} {logistic_metrics['accuracy']:>10.4f} {logistic_metrics['f1']:>10.4f}")


main()