import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# import functions from other python files 
from school_type import baseline_school_type_logistic_regression 
from school_type import baseline_school_type_most_frequent
from make_mm import (
    baseline_mm_most_appearances,
    baseline_mm_logistic_regression,
    baseline_mm_single_predictor,
)

TARGET_COL = "School Type"
MM_TARGET_COL = "Post-Season Tournament"

DROP_COLS = [
    "Mapped ESPN Team Name",
    "Current Coach",
    "Full Team Name",
    "Region",
    "Post-Season Tournament"
]

MM_DROP_COLS = [
    "Mapped ESPN Team Name",
    "Mapped Conference Name",
    "Short Conference Name",
    "Region",
    "Seed",
    "Post-Season Tournament Sorting Index",
    "Tournament Winner?",
    "Tournament Championship?",
    "Final Four?",
    "Top 12 in AP Top 25 During Week 6?",
    "Pre-Tournament.Tempo",
    "Pre-Tournament.AdjTempo",
    "School Type",
]

ENCODE_COLS = [
    "Short Conference Name",
    "Mapped Conference Name"
]

MM_NUMERIC_FEATURES = [
    "Season",
    "Adjusted Tempo", "Adjusted Tempo Rank",
    "Raw Tempo", "Raw Tempo Rank",
    "Adjusted Offensive Efficiency", "Adjusted Offensive Efficiency Rank",
    "Raw Offensive Efficiency", "Raw Offensive Efficiency Rank",
    "Adjusted Defensive Efficiency", "Adjusted Defensive Efficiency Rank",
    "Raw Defensive Efficiency", "Raw Defensive Efficiency Rank",
    "eFGPct", "RankeFGPct",
    "TOPct", "RankTOPct",
    "ORPct", "RankORPct",
    "FTRate", "RankFTRate",
    "OffFT", "RankOffFT",
    "Off2PtFG", "RankOff2PtFG",
    "Off3PtFG", "RankOff3PtFG",
    "DefFT", "RankDefFT",
    "Def2PtFG", "RankDef2PtFG",
    "Def3PtFG", "RankDef3PtFG",
    "FG2Pct", "RankFG2Pct",
    "FG3Pct", "RankFG3Pct",
    "FTPct", "RankFTPct",
    "BlockPct", "RankBlockPct",
    "OppFG2Pct", "RankOppFG2Pct",
    "OppFG3Pct", "RankOppFG3Pct",
    "OppFTPct", "RankOppFTPct",
    "OppBlockPct", "RankOppBlockPct",
    "FG3Rate", "RankFG3Rate",
    "OppFG3Rate", "RankOppFG3Rate",
    "ARate", "RankARate",
    "OppARate", "RankOppARate",
    "StlRate", "RankStlRate",
    "OppStlRate", "RankOppStlRate",
    "AvgHeight",
]


def load_and_prepare_data(df):
    df = df.dropna(subset=[TARGET_COL])
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = pd.get_dummies(df, columns=[c for c in ENCODE_COLS if c in df.columns])

    X = df.drop(columns=[TARGET_COL]).select_dtypes(include="number")
    y = df[TARGET_COL]
    return X, y


def load_and_prepare_mm_data(df):
    df = df.dropna(subset=[MM_TARGET_COL])
    y = (df[MM_TARGET_COL] == "March Madness").astype(int)
    available_features = [c for c in MM_NUMERIC_FEATURES if c in df.columns]
    X = df[available_features]
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
    # Load data
    clean_df = pd.read_csv("data/merged_data.csv")

    # ── School Type ──────────────────────────────────────────
    X, y = load_and_prepare_data(clean_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    baseline_metrics = baseline_school_type_most_frequent(X_train, X_test, y_train, y_test)
    print_results("Baseline (Most Frequent Class)", baseline_metrics)

    logistic_metrics = baseline_school_type_logistic_regression(X_train, X_test, y_train, y_test)
    print_results("Baseline (Logistic Regression)", logistic_metrics)

    print(f"\n{'='*40}")
    print("  School Type Summary Comparison")
    print(f"{'='*40}")
    print(f"  {'Model':<30} {'Accuracy':>10} {'F1':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Baseline (Most Frequent Class)':<30} {baseline_metrics['accuracy']:>10.4f} {baseline_metrics['f1']:>10.4f}")
    print(f"  {'Baseline (Logistic Regression)':<30} {logistic_metrics['accuracy']:>10.4f} {logistic_metrics['f1']:>10.4f}")

    # ── March Madness ─────────────────────────────────────────
    X_mm, y_mm = load_and_prepare_mm_data(clean_df)
    X_mm_train, X_mm_test, y_mm_train, y_mm_test = train_test_split(
        X_mm, y_mm, test_size=0.2, random_state=42, stratify=y_mm
    )

    mm_appearance_metrics = baseline_mm_most_appearances(X_mm_train, X_mm_test, y_mm_train, y_mm_test)
    print_results("MM Baseline (Most Frequent Class)", mm_appearance_metrics)

    mm_logistic_metrics = baseline_mm_logistic_regression(X_mm_train, X_mm_test, y_mm_train, y_mm_test)
    print_results("MM Baseline (Logistic Regression)", mm_logistic_metrics)

    mm_single_metrics = baseline_mm_single_predictor(X_mm_train, X_mm_test, y_mm_train, y_mm_test)
    print_results(f"MM Baseline (Single Predictor: {mm_single_metrics['feature_used']})", mm_single_metrics)

    print(f"\n{'='*40}")
    print("  March Madness Summary Comparison")
    print(f"{'='*40}")
    print(f"  {'Model':<50} {'Accuracy':>10} {'F1':>10}")
    print(f"  {'-'*70}")
    print(f"  {'MM Baseline (Most Frequent Class)':<50} {mm_appearance_metrics['accuracy']:>10.4f} {mm_appearance_metrics['f1']:>10.4f}")
    print(f"  {'MM Baseline (Logistic Regression)':<50} {mm_logistic_metrics['accuracy']:>10.4f} {mm_logistic_metrics['f1']:>10.4f}")
    single_label = f"MM Baseline (Single: {mm_single_metrics['feature_used']})"
    print(f"  {single_label:<50} {mm_single_metrics['accuracy']:>10.4f} {mm_single_metrics['f1']:>10.4f}")
main()