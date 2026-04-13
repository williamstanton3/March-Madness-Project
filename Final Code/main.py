'''This file is where we run everyting for our project and obtain every output that we need'''

# import python libraries
import pandas as pd 

# import functions from other scripts
from clean import convert_strings, convert_seed, convert_yes_no, impute_avg_height, impute_pre_tournament, get_final_cols, verify_clean_data


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

    if not verify_clean_data(cleaned_data):
        print("DATA IS NOT CLEAN — stopping.")
        return

    print("Data is clean — ready for modeling.")

    # COMPONENT 1: 3 Pt Rate (NBA vs NCAA)
    

    # COMPONENT 2: Offensive vs Defensive Efficiency to predict Final Four Appearance


    # COMPONENT 3: Predict a team's Seed in March Madness (models)


    # COMPONENT 4: Do schools see an increase in applicants after a Final Four Appearance?


if __name__ == "__main__":
    main()