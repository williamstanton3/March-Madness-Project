import pandas as pd
import numpy as np

def get_data_info(col: pd.Series) -> pd.Series:
    """
    Takes a single pandas Series (column) and returns summary statistics.
    
    Returns a Series with:
    Name, Data Type, Percent Missing, Variable Type,
    Most Common Value (Mode), Mode Percentage,
    Minimum, Maximum, Median
    """
    
    total_count = len(col)
    missing_count = col.isna().sum()
    percent_missing = (missing_count / total_count) * 100
    
    # Determine variable type
    unique_values = col.dropna().unique()
    num_unique = len(unique_values)
    
    if num_unique == 2:
        var_type = "Binary"
    elif pd.api.types.is_numeric_dtype(col):
        var_type = "Ordinal/Numeric"
    else:
        var_type = "Nominal"
    
    # Mode and its percentage
    if num_unique > 0:
        mode_value = col.mode().iloc[0]
        mode_percent = (col.value_counts(normalize=True).iloc[0]) * 100
    else:
        mode_value = np.nan
        mode_percent = np.nan
    
    # Numeric statistics
    if pd.api.types.is_numeric_dtype(col):
        minimum = col.min()
        maximum = col.max()
        median = col.median()
    else:
        minimum = np.nan
        maximum = np.nan
        median = np.nan
    
    # Return as a Series (single-row structure)
    column_info = pd.Series({
        "Name": col.name,
        "Data Type": col.dtype,
        "Percent Missing": percent_missing,
        "Variable Type": var_type,
        "Mode": mode_value,
        "Mode Percentage": mode_percent,
        "Minimum": minimum,
        "Maximum": maximum,
        "Median": median
    })
    
    return column_info

    


def main():
    raw_data = pd.read_csv("../Data/mm.csv")

    selected_cols = [
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
        # below are the ones we are going to impute 
        "Pre-Tournament.Tempo",
        "Pre-Tournament.AdjTempo",
        "Pre-Tournament.OE",
        "Pre-Tournament.DE"
    ]

    rows = [] # list of the the columns in the data (cols flipped to rows)

    for col in selected_cols:
        rows.append(get_data_info(raw_data[col]))

    stats_summary_df = pd.concat(rows, axis=1).T.reset_index(drop=True)


    # save sats_summary_df to csv 
    stats_summary_df.to_csv("../Data/data_report_info.csv")


main()