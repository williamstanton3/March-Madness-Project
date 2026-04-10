import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def random_forest(df, target_col):
    # use all of the columns in the df to predict the target_col

    # define features (x) and target (y) data
    x = df.drop(columns=[target_col]) # remove target_col from feature data 
    y = df[target_col]

    # split into training/testing data 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # train random forest 
    rand_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rand_forest_model.fit(x_train, y_train)

    # get feature importances
    feat_importances = rand_forest_model.feature_importances_

    # put feature importance into df and sort in descending order 
    importances = pd.DataFrame({
        "Feature": x.columns,
        "Importance": feat_importances
    }).sort_values(by="Importance", ascending=False)

    return importances # a df of how important each feature is


def lasso_regression(df, target_col, alpha=0.01):
    # define features (x) and target (y) data
    x = df.drop(columns=[target_col])
    y = df[target_col]

    # scale features — lasso is sensitive to feature scale
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # split into training/testing data
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

    # train lasso model
    lasso_model = Lasso(alpha=alpha, random_state=42)
    lasso_model.fit(x_train, y_train)

    # get coefficients — lasso shrinks unimportant features to 0
    importances = pd.DataFrame({
        "Feature": x.columns,
        "Importance": abs(lasso_model.coef_)  # use absolute value for ranking
    }).sort_values(by="Importance", ascending=False)

    # how many features were zeroed out
    n_zeroed = (lasso_model.coef_ == 0).sum()
    print(f"Lasso zeroed out {n_zeroed}/{len(x.columns)} features")

    return importances


def graph_importance(importances_df, top_n=20, title="Feature Importances", save_path=None):
    # take top N features
    top_features = importances_df.head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.gca().invert_yaxis()  # most important at top
    plt.xlabel("Importance Score")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)


def main():
    # load in clean march madness data 
    clean_data = pd.read_csv("../Data/mm_clean.csv")

    # only use teams that made march madness
    clean_mm_data = clean_data[clean_data["Seed"] != -1]

    # GOAL: identify most important features in the dataset for predicting the "Seed" column
    # APPROACH: use 2 techniques - Random Forest model and Lasso Regression

    # drop columns that would cause data leakage or are identifiers
    cols_to_drop = ["Season", "Region", "Post-Season Tournament", "Post-Season Tournament Sorting Index",
                    "Tournament Winner?", "Tournament Championship?", "Final Four?",
                    "Mapped Conference Name", "Mapped ESPN Team Name", "Pre-Tournament.AdjTempo"]
    model_data = clean_mm_data.drop(columns=cols_to_drop)
    
    # get rid of the "rank" columns where we have data on the actual value
    rank_cols = [
        "Adjusted Tempo Rank", "Raw Tempo Rank",
        "Adjusted Offensive Efficiency Rank", "Raw Offensive Efficiency Rank",
        "Adjusted Defensive Efficiency Rank", "Raw Defensive Efficiency Rank",
        "RankeFGPct", "RankTOPct", "RankORPct", "RankFTRate",
        "RankOffFT", "RankOff2PtFG", "RankOff3PtFG",
        "RankDefFT", "RankDef2PtFG", "RankDef3PtFG",
        "RankFG2Pct", "RankFG3Pct", "RankFTPct", "RankBlockPct",
        "RankOppFG2Pct", "RankOppFG3Pct", "RankOppFTPct", "RankOppBlockPct",
        "RankFG3Rate", "RankOppFG3Rate", "RankARate", "RankOppARate",
        "RankStlRate", "RankOppStlRate"
    ]
    model_data = model_data.drop(columns=rank_cols)    

    print(model_data.columns)

    # one hot encode the categorical data (short conference name)
    model_data = pd.get_dummies(model_data, columns=["Short Conference Name"])

    # 1. Random Forest
    feat_importances_rf = random_forest(model_data, "Seed")
    print("\n--- Random Forest Feature Importances ---")
    print(feat_importances_rf)
    graph_importance(feat_importances_rf, top_n=20, title="Random Forest Feature Importances", save_path="rf_feature_importance.png")

    # 2. Lasso Regression
    feat_importances_lasso = lasso_regression(model_data, "Seed", alpha=0.01)
    print("\n--- Lasso Regression Feature Importances ---")
    print(feat_importances_lasso)
    graph_importance(feat_importances_lasso, top_n=20, title="Lasso Regression Feature Importances", save_path="lasso_feature_importance.png")


if __name__ == "__main__":
    main()