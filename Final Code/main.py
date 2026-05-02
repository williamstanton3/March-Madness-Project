'''This file is where we run everyting for our project and obtain every output that we need'''

# import python libraries
import pandas as pd 

# import functions from other scripts
from clean import convert_strings, convert_seed, convert_yes_no, impute_avg_height, impute_pre_tournament, get_final_cols, verify_clean_data
from three_pt_rate import process_data, create_figure 
from off_def_efficiency import build_roc_models, plot_roc_comparison
from applicants import process_final_four_applications, plot_final_four_applications, plot_top10_applicant_jumps
from model import train_models_and_get_accuracies, plot_model_comparison, plot_confusion_matrices



def load_raw_mm_data() -> pd.DataFrame: 
    """Loads the raw dataset and returns a df with that raw data""" 
 
    df = pd.read_csv("Raw_Data/raw_mm.csv") 
    return df 

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Takes the raw data and cleans it. Returns a useable dataset"""

    """
        Cleaning Steps:
        1. 
    """

    FINAL_COLUMNS = get_final_cols()

    # select target columns
    important_cols = [col for col in FINAL_COLUMNS if col in df.columns]
    df = df[important_cols].copy()
    print(f"[clean_data] Selected {len(important_cols)} important columns.")

    # convert string object columns
    df = convert_strings(df)

    # seed: object → int (−1 for non-tournament teams)
    df = convert_seed(df)

    # yes/no columns → binary int
    df = convert_yes_no(df)

    # impute AvgHeight (5-year rolling school median)
    df = impute_avg_height(df)

    # impute pre-tournament columns (10-year rolling school median)
    df = impute_pre_tournament(df)

    # final check
    null_counts = df.isnull().sum()
    remaining_nulls = null_counts[null_counts > 0]

    if not remaining_nulls.empty:
        print("[clean_data] Remaining nulls after imputation:")
        print(remaining_nulls.to_string())
    else:
        print("[clean_data] No remaining nulls.")

    return df

def main():
    # LOAD AND CLEAN DATA
    raw_data = load_raw_mm_data()

    cleaned_data = clean_data(raw_data)

    print(f"NUMBER IN CLEANED_DATA: {len(cleaned_data)}")

    print("CLEAN DATA COLS:")
    print(cleaned_data.columns)

    if not verify_clean_data(cleaned_data):
        print("DATA IS NOT CLEAN — stopping.")
        return

    print("Data is clean — ready for modeling.")

    # COMPONENT 1: 3 Pt Rate (NBA vs NCAA)
    df_merged, nba_df = process_data(cleaned_data)
    create_figure(
        df_merged,
        nba_df,
        save_path="Graphs/nba_vs_ncaa_threes.png"
    )

    # COMPONENT 2: Offensive vs Defensive Efficiency to predict Final Four Appearance
    # only use teams who made march madness 
    mm_teams = cleaned_data[
        (cleaned_data["Seed"] != -1) & 
        (cleaned_data["Post-Season Tournament"] == "March Madness")
    ]

    # ensure data is right
    print(mm_teams["Season"].value_counts())
    print(mm_teams["Seed"].value_counts())
    print(mm_teams["Final Four?"].value_counts())


    print("STARTING OE vs DE")
    results = build_roc_models(mm_teams)
    plot_roc_comparison(results, save_path="Graphs/off_def_eff.png")

    # # COMPONENT 3: Do schools see an increase in applicants after a Final Four Appearance?
    # print("STARING APPLICATIONS")
    # applicants_data = process_final_four_applications(mm_teams)
    # plot_final_four_applications(applicants_data, save_path="Graphs/applicants.png")
    # plot_top10_applicant_jumps(applicants_data, save_path="Graphs/applicants_spread.png")

    # COMPONENT 4: Predict March Madness qualification using ML models (with SMOTE)
    train_models_and_get_accuracies(cleaned_data, output_folder="output_models")
    plot_model_comparison(cleaned_data, save_path="Graphs/model_comparison_mm.png")
    plot_confusion_matrices(cleaned_data, save_path="Graphs/confusion_matrices_mm.png")


if __name__ == "__main__":
    main()