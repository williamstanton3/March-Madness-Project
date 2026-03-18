#In here we will list all of the initial questions we are asking and create the models that we created to answer these questions 
import pandas as pd
from school_type_predictor_model import train_model_pipeline


def merge_data(main_file="../Data Report Project/data/mm_clean.csv", key_file="ncaa_d1_mapping.csv", output_file="merged_data.csv"):
    """
    Merges main dataset with school type mapping (public/private)
    and outputs a cleaned dataset.
    """

    # Load data
    main_df = pd.read_csv(main_file)
    key_df = pd.read_csv(key_file)

    # Rename column for consistency
    key_df = key_df.rename(columns={"public_private": "School Type"})

    # Merge on team name
    df = pd.merge(main_df, key_df, on="Mapped ESPN Team Name", how="left")

    # Check for unmatched rows
    unmatched = df["School Type"].isna().sum()
    if unmatched > 0:
        print(f"Warning: {unmatched} rows could not be matched and will be dropped")

    # Drop unmatched rows
    df = df.dropna(subset=["School Type"])

    # Save result
    df.to_csv(output_file, index=False)
    print(f"Merged {len(df)} rows into {output_file}")

    return df



def main():
    # first add 'private' or 'public' to the data 
    initial_model_data = merge_data()

    print(initial_model_data[["Mapped ESPN Team Name", "School Type"]]) # eye check for validity

    # Question 1: Can we predict if a school is private or public?
    
    model, acc = train_model_pipeline(model_type="logistic")

    print(f"\nFinal accuracy: {acc:.4f}")

main()