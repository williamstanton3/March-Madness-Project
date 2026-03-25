# write baseline models for school type. run them in main
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

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
