import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import time


def evaluate_model(model, x_train, x_test, y_train, y_test, model_name, runtime_s=None):
    """Print train/test evaluation metrics for a fitted model."""
    train_preds = model.predict(x_train)
    test_preds  = model.predict(x_test)

    runtime_str = f" | Runtime: {runtime_s:.2f}s" if runtime_s is not None else ""
    print(f"\n--- {model_name} Evaluation ---")
    print(f"  Features : {x_train.shape[1]}{runtime_str}")
    print(f"  Train  R²: {r2_score(y_train, train_preds):.4f} | MAE: {mean_absolute_error(y_train, train_preds):.4f} | RMSE: {np.sqrt(mean_squared_error(y_train, train_preds)):.4f}")
    print(f"  Test   R²: {r2_score(y_test,  test_preds ):.4f} | MAE: {mean_absolute_error(y_test,  test_preds ):.4f} | RMSE: {np.sqrt(mean_squared_error(y_test,  test_preds )):.4f}")
import json


def random_forest(df, target_col):
    x = df.drop(columns=[target_col])
    y = df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    t0 = time.perf_counter()
    rand_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rand_forest_model.fit(x_train, y_train)
    runtime = time.perf_counter() - t0

    evaluate_model(rand_forest_model, x_train, x_test, y_train, y_test,
                   "Random Forest (full features)", runtime_s=runtime)

    importances = pd.DataFrame({
        "Feature":    x.columns,
        "Importance": rand_forest_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return importances


def save_importances_to_json(rf_importances, lasso_importances, save_path="feature_importances.json"):
    output = {
        "random_forest": rf_importances.set_index("Feature")["Importance"].to_dict(),
        "lasso": lasso_importances.set_index("Feature")["Importance"].to_dict()
    }

    with open(save_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Saved feature importances to {save_path}")


def lasso_regression(df, target_col, alpha=0.01):
    x = df.drop(columns=[target_col])
    y = df[target_col]

    scaler   = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

    t0 = time.perf_counter()
    lasso_model = Lasso(alpha=alpha, random_state=42)
    lasso_model.fit(x_train, y_train)
    runtime = time.perf_counter() - t0

    n_zeroed = (lasso_model.coef_ == 0).sum()
    print(f"Lasso zeroed out {n_zeroed}/{len(x.columns)} features")

    evaluate_model(lasso_model, x_train, x_test, y_train, y_test,
                   f"Lasso (full features, α={alpha})", runtime_s=runtime)

    importances = pd.DataFrame({
        "Feature":    x.columns,
        "Importance": abs(lasso_model.coef_)
    }).sort_values(by="Importance", ascending=False)

    return importances


def rebuild_with_important_features(df, target_col, importances_rf, importances_lasso,
                                    top_n=30, lasso_alpha=0.01):
    """
    Select the union of the top-N features from each importance ranking,
    then retrain and evaluate both models on that reduced feature set.

    Parameters
    ----------
    df               : full model-ready dataframe (including target_col)
    target_col       : name of the column to predict
    importances_rf   : DataFrame returned by random_forest()
    importances_lasso: DataFrame returned by lasso_regression()
    top_n            : how many top features to keep from each model
    lasso_alpha      : regularisation strength for the rebuilt Lasso model
    """

    # --- select important features ---
    top_rf    = set(importances_rf.head(top_n)["Feature"])
    top_lasso = set(importances_lasso.head(top_n)["Feature"])
    selected  = sorted(top_rf | top_lasso)

    print(f"\n=== Rebuilding models with {len(selected)} important features "
          f"(top {top_n} from each model, union) ===")
    print("Features selected:", selected)

    reduced_df = df[selected + [target_col]]

    x = reduced_df.drop(columns=[target_col])
    y = reduced_df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # --- rebuilt Random Forest ---
    t0 = time.perf_counter()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_train)
    rf_runtime = time.perf_counter() - t0

    evaluate_model(rf_model, x_train, x_test, y_train, y_test,
                   f"Random Forest (top {top_n} features)", runtime_s=rf_runtime)

    # --- rebuilt Lasso ---
    scaler         = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled  = scaler.transform(x_test)

    t0 = time.perf_counter()
    lasso_model = Lasso(alpha=lasso_alpha, random_state=42)
    lasso_model.fit(x_train_scaled, y_train)
    lasso_runtime = time.perf_counter() - t0

    evaluate_model(lasso_model, x_train_scaled, x_test_scaled, y_train, y_test,
                   f"Lasso (top {top_n} features, α={lasso_alpha})", runtime_s=lasso_runtime)

    # --- feature importances for rebuilt models ---
    rf_importances_reduced = pd.DataFrame({
        "Feature":    x.columns,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    lasso_importances_reduced = pd.DataFrame({
        "Feature":    x.columns,
        "Importance": abs(lasso_model.coef_)
    }).sort_values(by="Importance", ascending=False)

    return rf_importances_reduced, lasso_importances_reduced


def graph_importance(importances_df, top_n=20, title="Feature Importances", save_path=None):
    top_features = importances_df.head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance Score")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)


def main():
    clean_data    = pd.read_csv("../Data/mm_clean.csv")
    clean_mm_data = clean_data[clean_data["Seed"] != -1]

    cols_to_drop = ["Season", "Region", "Post-Season Tournament", "Post-Season Tournament Sorting Index",
                    "Tournament Winner?", "Tournament Championship?", "Final Four?",
                    "Mapped Conference Name", "Mapped ESPN Team Name", "Pre-Tournament.AdjTempo"]
    model_data = clean_mm_data.drop(columns=cols_to_drop)

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

    model_data = pd.get_dummies(model_data, columns=["Short Conference Name"])

    # ── Phase 1: full-feature models ─────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 1 — Full feature models")
    print("="*60)

    feat_importances_rf = random_forest(model_data, "Seed")
    print("\n--- Random Forest Feature Importances ---")
    print(feat_importances_rf)
    graph_importance(feat_importances_rf, top_n=20, title="Random Forest Feature Importances",
                     save_path="rf_feature_importance.png")

    feat_importances_lasso = lasso_regression(model_data, "Seed", alpha=0.01)
    print("\n--- Lasso Regression Feature Importances ---")
    print(feat_importances_lasso)
    graph_importance(feat_importances_lasso, top_n=20, title="Lasso Regression Feature Importances",
                     save_path="lasso_feature_importance.png")

    # ── Phase 2: rebuild with only the important features ────────────────────
    print("\n" + "="*60)
    print("PHASE 2 — Rebuilt models (important features only)")
    print("="*60)

    rf_reduced, lasso_reduced = rebuild_with_important_features(
        df=model_data,
        target_col="Seed",
        importances_rf=feat_importances_rf,
        importances_lasso=feat_importances_lasso,
        top_n=30,
        lasso_alpha=0.01
    )

    print("\n--- Rebuilt Random Forest Feature Importances (reduced) ---")
    print(rf_reduced)
    graph_importance(rf_reduced, top_n=30,
                     title="Random Forest Feature Importances (Reduced)",
                     save_path="rf_reduced_feature_importance.png")

    print("\n--- Rebuilt Lasso Feature Importances (reduced) ---")
    print(lasso_reduced)
    graph_importance(lasso_reduced, top_n=30,
                     title="Lasso Feature Importances (Reduced)",
                     save_path="lasso_reduced_feature_importance.png")

    save_importances_to_json(feat_importances_rf, feat_importances_lasso)


if __name__ == "__main__":
    main()