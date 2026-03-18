# single file to create a clean dataset
# hello
import pandas as pd

# list of columns that we are going to use for our project
FINAL_COLUMNS = [
    "Season",
    "Short Conference Name",
    "Adjusted Tempo",
    "Adjusted Tempo Rank",
    "Raw Tempo",
    "Raw Tempo Rank",
    "Adjusted Offensive Efficiency",
    "Adjusted Offensive Efficiency Rank",
    "Raw Offensive Efficiency",
    "Raw Offensive Efficiency Rank",
    "Adjusted Defensive Efficiency",
    "Adjusted Defensive Efficiency Rank",
    "Raw Defensive Efficiency",
    "Raw Defensive Efficiency Rank",
    "eFGPct",
    "RankeFGPct",
    "TOPct",
    "RankTOPct",
    "ORPct",
    "RankORPct",
    "FTRate",
    "RankFTRate",
    "OffFT",
    "RankOffFT",
    "Off2PtFG",
    "RankOff2PtFG",
    "Off3PtFG",
    "RankOff3PtFG",
    "DefFT",
    "RankDefFT",
    "Def2PtFG",
    "RankDef2PtFG",
    "Def3PtFG",
    "RankDef3PtFG",
    "FG2Pct",
    "RankFG2Pct",
    "FG3Pct",
    "RankFG3Pct",
    "FTPct",
    "RankFTPct",
    "BlockPct",
    "RankBlockPct",
    "OppFG2Pct",
    "RankOppFG2Pct",
    "OppFG3Pct",
    "RankOppFG3Pct",
    "OppFTPct",
    "RankOppFTPct",
    "OppBlockPct",
    "RankOppBlockPct",
    "FG3Rate",
    "RankFG3Rate",
    "OppFG3Rate",
    "RankOppFG3Rate",
    "ARate",
    "RankARate",
    "OppARate",
    "RankOppARate",
    "StlRate",
    "RankStlRate",
    "OppStlRate",
    "RankOppStlRate",
    "AvgHeight",
    "Mapped Conference Name",
    "Mapped ESPN Team Name",
    "Seed",
    "Region",
    "Post-Season Tournament",
    "Post-Season Tournament Sorting Index",
    "Tournament Winner?",
    "Tournament Championship?",
    "Final Four?",
    "Top 12 in AP Top 25 During Week 6?",
    "Pre-Tournament.Tempo",
    "Pre-Tournament.AdjTempo",
    "Pre-Tournament.OE__",
    "Pre-Tournament.DE__",
]

# function to convert columns to strings
def convert_strings(df):

    string_cols = ["Short Conference Name","Mapped Conference Name","Mapped ESPN Team Name"]
    
    for col in string_cols:
        if col in df.columns:
            before_unique = df[col].nunique()
            df[col] = df[col].astype(str).str.strip()
            after_unique = df[col].nunique()
            print(f"[convert_strings] Column '{col}': stripped whitespace and converted to string. Unique values changed {before_unique} → {after_unique}")
    return df

# converts seed from object to int, using -1 magic value
def convert_seed(df):
    # tries to convert seed to an int, converts 0 to -1
    def parse(val):
        try:
            num = int(val)
            if num == 0:
                return -1  # treat 0 as not in tournament
            return num
        except:
            return -1  # if it can't become int, make it -1

    before_counts = df["Seed"].value_counts(dropna=False)
    df["Seed"] = df["Seed"].apply(parse)
    after_counts = df["Seed"].value_counts(dropna=False)

    print(f"[convert_seed] Converted 'Seed' to int with -1 for non-tournament teams and 0 seeds.")
    print(f"[convert_seed] Before counts (top 5):\n{before_counts.head()}")
    print(f"[convert_seed] After counts (top 5):\n{after_counts.head()}")
    return df


# converts "Yes", "No" columns to 1 and 0
def convert_yes_no(df):
    yes_no_cols = [
        "Tournament Winner?",
        "Tournament Championship?",
        "Final Four?",
        "Top 12 in AP Top 25 During Week 6?",
    ]

    for col in yes_no_cols:
        if col in df.columns:
            before_counts = df[col].value_counts(dropna=False)
            df[col] = (
                df[col].astype(str).str.strip().str.lower()
                .map({"yes": 1, "no": 0})
                .fillna(0)
                .astype(int)
            )
            after_counts = df[col].value_counts(dropna=False)
            print(f"[convert_yes_no] Column '{col}': converted Yes/No → 1/0")
            print(f"[convert_yes_no] Before counts:\n{before_counts}")
            print(f"[convert_yes_no] After counts:\n{after_counts}")
    return df

# relper function to impute a missing value by the median of that school's x number of previous years
def school_median(df, col, years):
    df = df.sort_values(["Mapped ESPN Team Name", "Season"]).copy()
    result = df[col].copy()

    # computes rolling median
    imputed = df.groupby("Mapped ESPN Team Name")[col].transform(
        lambda g: g.shift(1).rolling(window=years, min_periods=1).median()
    )

    # mask of missing values
    mask = result.isna()
    
    # only fill if imputed is not NaN
    fill_mask = mask & imputed.notna()
    result.loc[fill_mask] = imputed[fill_mask]

    changed_count = fill_mask.sum()
    print(f"[school_median] Column '{col}': actually imputed {changed_count} missing values using {years}-year rolling school median.")

    return result


# impute missing avg_height values based on the previous 5 years
def impute_avg_height(df: pd.DataFrame) -> pd.DataFrame:

    # use school_median function to impute
    df["AvgHeight"] = school_median(df, "AvgHeight", years=5)

    # if that doesn't work (rare) impute with global median
    remaining = df["AvgHeight"].isna().sum()

    if remaining:
        global_median = df["AvgHeight"].median()
        df["AvgHeight"] = df["AvgHeight"].fillna(global_median)
        print(f"[impute_avg_height] 'AvgHeight' — filled {remaining} remaining NaNs with global median ({global_median:.2f}).")
    else:
        print("[impute_avg_height] 'AvgHeight' — no remaining NaNs after school median imputation.")

    return df

# impute other missing values based on previous 10 years
def impute_pre_tournament(df: pd.DataFrame) -> pd.DataFrame:

    pre_tournament_cols = [
    "Pre-Tournament.Tempo",
    "Pre-Tournament.AdjTempo",
    "Pre-Tournament.OE__",
    "Pre-Tournament.DE__",
    ]
    
    for col in pre_tournament_cols:
        if col in df.columns:
            df[col] = school_median(df, col, years=10)

            # if that doesn't work impute with global median
            remaining = df[col].isna().sum()

            if remaining:
                global_median = df[col].median()
                df[col] = df[col].fillna(global_median)
                print(f"[impute_pre_tournament] '{col}' — filled {remaining} remaining NaNs with global median ({global_median:.4f}).")
            else:
                print(f"[impute_pre_tournament] '{col}' — no remaining NaNs after school median imputation.")
    return df

def main():

    df = pd.read_csv("data/mm.csv")
    print(f"[main] Loaded CSV with shape: {df.shape}")

    # select target columns
    important_cols = [col for col in FINAL_COLUMNS if col in df.columns]
    df = df[important_cols]
    print(f"[main] Selected {len(important_cols)} important columns.")

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

    # final check & save
    null_counts = df.isnull().sum()
    remaining_nulls = null_counts[null_counts > 0]

    if not remaining_nulls.empty:
        print("[main] Remaining nulls after imputation:")
        print(remaining_nulls.to_string())
    else:
        print("[main] No remaining nulls.")

    df.to_csv("data/mm_clean.csv", index=False)
    print("[main] Saved cleaned dataset to 'clean_data_set/mm_clean.csv'")

main()