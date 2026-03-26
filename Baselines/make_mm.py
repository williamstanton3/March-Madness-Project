# march_madness.py
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


def build_metrics(y_test, y_pred):
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0)
    }


def baseline_mm_most_appearances(X_train, X_test, y_train, y_test):
    """
    Trivial baseline: predict True for any team that has previously appeared
    in March Madness (i.e. Seed is not null / > 0), else False.
    Falls back to predicting the most frequent class if the column is missing.
    """
    appearance_col = "Seed" if "Seed" in X_test.columns else None

    if appearance_col:
        y_pred = X_test[appearance_col].notna().astype(int)
        y_pred = y_pred.astype(y_test.dtype)
    else:
        most_frequent = y_train.value_counts().idxmax()
        y_pred = pd.Series([most_frequent] * len(y_test), index=y_test.index)

    return build_metrics(y_test, y_pred)


def baseline_mm_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Simple model: logistic regression over all numeric features.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return build_metrics(y_test, y_pred)


def baseline_mm_single_predictor(X_train, X_test, y_train, y_test, feature=None):
    """
    Single-predictor baseline: train logistic regression on one feature only.
    If `feature` is None, selects the feature most correlated with the target
    from the training set.
    Returns metrics dict with an added 'feature_used' key.
    """
    if feature is None:
        # Pick the numeric feature with the highest absolute correlation to y_train
        correlations = X_train.corrwith(y_train.astype(float)).abs()
        feature = correlations.idxmax()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0))
    ])
    model.fit(X_train[[feature]], y_train)
    y_pred = model.predict(X_test[[feature]])

    metrics = build_metrics(y_test, y_pred)
    metrics["feature_used"] = feature
    return metrics