import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_model(df: pd.DataFrame, target: str, features: list[str], 
                model_type: str, args: list = None, kwargs: dict = None) -> object:
    """
    Train a model on the provided dataframe.
    
    Args:
        df: Input dataframe
        target: Name of target column
        features: List of feature column names
        model_type: Type of model ('logistic_regression' or 'random_forest')
        args: Positional arguments for model
        kwargs: Keyword arguments for model
    
    Returns:
        Trained model object
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    
    # Extract features and target
    X = df[features].copy()
    y = df[target].copy()
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model based on type
    if model_type == 'logistic_regression':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(*args, max_iter=1000, **kwargs)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
    elif model_type == 'random_forest':
        model = RandomForestClassifier(*args, n_estimators=200, max_depth=10, random_state=42, **kwargs)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Print metrics
    print(f"\n{model_type.replace('_', ' ').title()}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob)}")
    
    return model

def save_model(model: object, filepath: str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model object
        filepath: Path where to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def main():
    # load in clean data 
    clean_data = pd.read_csv("../Data Report Project/data/mm_clean.csv")

    