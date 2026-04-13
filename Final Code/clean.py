# single file to create a clean dataset
import pandas as pd

# list of columns that we are going to use for our project
FINAL_COLUMNS = [
    # Identity
    "Season",
    "Mapped ESPN Team Name",
    "Mapped Conference Name",

    # Four Factors (offense)
    "eFGPct",
    "TOPct",
    "ORPct",
    "FTRate",

    # Shooting (offense)
    "Off2PtFG",
    "Off3PtFG",
    "OffFT",
    "FG2Pct",
    "FG3Pct",
    "FTPct",
    "FG3Rate",
    "BlockPct",

    # Shooting (defense)
    "Def2PtFG",
    "Def3PtFG",
    "DefFT",
    "OppFG2Pct",
    "OppFG3Pct",
    "OppFTPct",
    "OppFG3Rate",
    "OppBlockPct",

    # Playmaking
    "ARate",
    "StlRate",
    "OppARate",
    "OppStlRate",

    # Physical
    "AvgHeight",

    # Pre-tournament form
    "Pre-Tournament.AdjTempo",
    "Pre-Tournament.OE__",
    "Pre-Tournament.DE__",

    # Tournament context
    "Seed",
    "Region",
    "Post-Season Tournament",
    "Post-Season Tournament Sorting Index",
    "Top 12 in AP Top 25 During Week 6?",

    # Targets
    "Final Four?",
    "Tournament Championship?",
    "Tournament Winner?",
]

def get_final_cols() -> list: 
    ''' helper function for main.py'''
    return FINAL_COLUMNS

# function to convert columns to strings
def convert_strings(df):

    string_cols = ["Mapped Conference Name","Mapped ESPN Team Name"]
    
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

# helper function to impute a missing value by the median of that school's x number of previous years
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


def verify_clean_data(df: pd.DataFrame) -> bool:
    """Runs a series of checks to verify the cleaned dataset looks correct"""

    print("\n========== DATA VERIFICATION ==========")
    issues_found = 0

    # 1. Shape
    print(f"\n[shape] {df.shape[0]} rows x {df.shape[1]} columns")

    # 2. Null check
    null_counts = df.isnull().sum()
    remaining_nulls = null_counts[null_counts > 0]
    if not remaining_nulls.empty:
        print(f"[nulls] FAIL — nulls found:")
        print(remaining_nulls.to_string())
        issues_found += 1
    else:
        print("[nulls] PASS — no nulls")

    # 3. Seed values should only be -1 or 1-16
    invalid_seeds = df[~df["Seed"].isin(range(-1, 17))]
    if not invalid_seeds.empty:
        print(f"[seed] FAIL — {len(invalid_seeds)} rows with unexpected seed values")
        issues_found += 1
    else:
        print("[seed] PASS — all seeds in valid range (-1 or 1–16)")

    # 4. Yes/No columns should only be 0 or 1
    binary_cols = ["Tournament Winner?", "Tournament Championship?", "Final Four?", "Top 12 in AP Top 25 During Week 6?"]
    for col in binary_cols:
        if col in df.columns:
            bad = df[~df[col].isin([0, 1])]
            if not bad.empty:
                print(f"[binary] FAIL — '{col}' has {len(bad)} non-binary values")
                issues_found += 1
            else:
                print(f"[binary] PASS — '{col}' is clean")

    # 5. Tournament logic: winner must be champion, champion must be final four
    if all(c in df.columns for c in ["Tournament Winner?", "Tournament Championship?", "Final Four?"]):
        bad_winner = df[(df["Tournament Winner?"] == 1) & (df["Tournament Championship?"] == 0)]
        bad_champ = df[(df["Tournament Championship?"] == 1) & (df["Final Four?"] == 0)]
        if not bad_winner.empty:
            print(f"[tournament logic] FAIL — {len(bad_winner)} winners without championship flag")
            issues_found += 1
        elif not bad_champ.empty:
            print(f"[tournament logic] FAIL — {len(bad_champ)} champions without final four flag")
            issues_found += 1
        else:
            print("[tournament logic] PASS — winner/champion/final four hierarchy is consistent")

    # 6. String columns should have no leading/trailing whitespace
    for col in ["Mapped ESPN Team Name", "Mapped Conference Name"]:
        if col in df.columns:
            bad = df[df[col].str.strip() != df[col]]
            if not bad.empty:
                print(f"[strings] FAIL — '{col}' has {len(bad)} values with whitespace")
                issues_found += 1
            else:
                print(f"[strings] PASS — '{col}' is clean")

    # 8. Season should be a plausible year
    if "Season" in df.columns:
        bad_season = df[(df["Season"] < 1990) | (df["Season"] > 2030)]
        if not bad_season.empty:
            print(f"[season] FAIL — {len(bad_season)} rows with implausible season values")
            issues_found += 1
        else:
            print(f"[season] PASS — seasons range from {df['Season'].min()} to {df['Season'].max()}")

    # Summary
    is_clean = issues_found == 0
    print("\n========================================")
    print("ALL CHECKS PASSED" if is_clean else f"{issues_found} ISSUE(S) FOUND — review output above")
    print("========================================\n")

    return is_clean
