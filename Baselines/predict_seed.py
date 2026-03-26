# write baseline models for predicting seed. called from main.py
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


def subset_data(all_data: pd.DataFrame):
    '''Only use regular season stats to predict which seed a team will have in march madness,
    and only use data from teams that made march madness.'''

    predict_seed_data = all_data[all_data["Post-Season Tournament"] == "March Madness"].copy()

    cols_to_use = [
        'Short Conference Name',
        'Mapped Conference Name',
        'Seed',
        'Adjusted Tempo', 'Adjusted Tempo Rank',
        'Raw Tempo', 'Raw Tempo Rank',
        'Adjusted Offensive Efficiency', 'Adjusted Offensive Efficiency Rank',
        'Raw Offensive Efficiency', 'Raw Offensive Efficiency Rank',
        'Adjusted Defensive Efficiency', 'Adjusted Defensive Efficiency Rank',
        'Raw Defensive Efficiency', 'Raw Defensive Efficiency Rank',
        'eFGPct', 'RankeFGPct',
        'TOPct', 'RankTOPct',
        'ORPct', 'RankORPct',
        'FTRate', 'RankFTRate',
        'OffFT', 'RankOffFT',
        'Off2PtFG', 'RankOff2PtFG',
        'Off3PtFG', 'RankOff3PtFG',
        'DefFT', 'RankDefFT',
        'Def2PtFG', 'RankDef2PtFG',
        'Def3PtFG', 'RankDef3PtFG',
        'FG2Pct', 'RankFG2Pct',
        'FG3Pct', 'RankFG3Pct',
        'FTPct', 'RankFTPct',
        'BlockPct', 'RankBlockPct',
        'OppFG2Pct', 'RankOppFG2Pct',
        'OppFG3Pct', 'RankOppFG3Pct',
        'OppFTPct', 'RankOppFTPct',
        'OppBlockPct', 'RankOppBlockPct',
        'FG3Rate', 'RankFG3Rate',
        'OppFG3Rate', 'RankOppFG3Rate',
        'ARate', 'RankARate',
        'OppARate', 'RankOppARate',
        'StlRate', 'RankStlRate',
        'OppStlRate', 'RankOppStlRate',
        'AvgHeight',
    ]

    predict_seed_data = predict_seed_data[cols_to_use]

    predict_seed_data = pd.get_dummies(
        predict_seed_data,
        columns=['Short Conference Name', 'Mapped Conference Name'],
        drop_first=False
    )

    y = predict_seed_data['Seed']
    X = predict_seed_data.drop(columns=['Seed'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def baseline_predict_seed_most_frequent(X_train, X_test, y_train, y_test):
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return build_metrics(y_test, y_pred)


def baseline_predict_seed_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return build_metrics(y_test, y_pred)

def baseline_predict_seed_one_col(X_train, X_test, y_train, y_test, predict_col):
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train[[predict_col]], y_train)
    y_pred = model.predict(X_test[[predict_col]])
    return build_metrics(y_test, y_pred)