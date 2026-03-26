import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# import functions from other python files
from predict_seed import subset_data, baseline_predict_seed_most_frequent, baseline_predict_seed_logistic_regression, baseline_predict_seed_one_col

TARGET_COL = "Seed"

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
    clean_df = pd.read_csv("data/merged_data.csv")

    # predict seed baselines
    X_train_s, X_test_s, y_train_s, y_test_s = subset_data(clean_df)

    seed_baseline_metrics = baseline_predict_seed_most_frequent(X_train_s, X_test_s, y_train_s, y_test_s)
    print_results("Seed Prediction — Baseline (Most Frequent Class)", seed_baseline_metrics)

    seed_lr_metrics = baseline_predict_seed_logistic_regression(X_train_s, X_test_s, y_train_s, y_test_s)
    print_results("Seed Prediction — Baseline (Logistic Regression)", seed_lr_metrics)

    single_col = "Adjusted Offensive Efficiency Rank"
    seed_one_col_metrics = baseline_predict_seed_one_col(X_train_s, X_test_s, y_train_s, y_test_s, single_col) 
    print_results("Seed Prediction - Baseline (Single Column)", seed_one_col_metrics)

    print(f"\n{'='*40}")
    print("  Seed Prediction Summary Comparison")
    print(f"{'='*40}")
    print(f"  {'Model':<30} {'Accuracy':>10} {'F1':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Baseline (Most Frequent Class)':<30} {seed_baseline_metrics['accuracy']:>10.4f} {seed_baseline_metrics['f1']:>10.4f}")
    print(f"  {'Baseline (Logistic Regression)':<30} {seed_lr_metrics['accuracy']:>10.4f} {seed_lr_metrics['f1']:>10.4f}")
    print(f"  {'Baseline (Single Column)':<30} {seed_one_col_metrics['accuracy']:>10.4f} {seed_one_col_metrics['f1']:>10.4f}")


    # predict march madness apearance baselines
    


main()