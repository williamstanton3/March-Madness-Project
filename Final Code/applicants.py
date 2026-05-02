import pandas as pd
import glob
import os
import re
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

def process_final_four_applications(
    mm_data,
    app_folder="Raw_Data/application_data",
    school_file="Raw_Data/school_nums.csv",
    output_path="Output_Data/final_four_applicant_changes.csv"
):
    """
    Loads data, processes Final Four application changes,
    saves cleaned dataset, and prints summary statistics.
    """

    # Load March Madness data
    mm_clean = mm_data

    # Load application data
    applicant_data_post_2013 = get_applicant_data_post_2013(app_folder, school_file)
    applicant_data_pre_2014 = get_applicant_data_pre_2014(app_folder, school_file)

    total_application_data = pd.concat(
        [applicant_data_pre_2014, applicant_data_post_2013],
        ignore_index=True
    )

    # School name mapping
    update_school_names = {
        "University of Maryland-College Park": "Maryland",
        "Indiana University-Bloomington": "Indiana",
        "University of Kansas": "Kansas",
        "Michigan State University": "Michigan State",
        "Syracuse University": "Syracuse",
        "Marquette University": "Marquette",
        "The University of Texas at Austin": "Texas",
        "University of Connecticut": "UConn",
        "Georgia Institute of Technology-Main Campus": "Georgia Tech",
        "Duke University": "Duke",
        "Oklahoma State University-Main Campus": "Oklahoma State",
        "University of North Carolina at Chapel Hill": "North Carolina",
        "University of Illinois Urbana-Champaign": "Illinois",
        "University of Louisville": "Louisville",
        "University of Florida": "Florida",
        "University of California-Los Angeles": "UCLA",
        "George Mason University": "George Mason",
        "Louisiana State University and Agricultural & Mechanical College": "LSU",
        "Ohio State University-Main Campus": "Ohio State",
        "Georgetown University": "Georgetown",
        "University of Memphis": "Memphis",
        "Villanova University": "Villanova",
        "Butler University": "Butler",
        "West Virginia University": "West Virginia",
        "University of Kentucky": "Kentucky",
        "University of Michigan-Ann Arbor": "Michigan",
        "Wichita State University": "Wichita State",
        "University of Wisconsin-Madison": "Wisconsin",
        "University of Oklahoma-Norman Campus": "Oklahoma",
        "Gonzaga University": "Gonzaga",
        "University of Oregon": "Oregon",
        "University of South Carolina-Columbia": "South Carolina",
        "Loyola University Chicago": "Loyola Chicago",
        "University of Virginia-Main Campus": "Virginia",
        "Texas Tech University": "Texas Tech",
        "Auburn University": "Auburn",
        "Baylor University": "Baylor",
        "University of Houston": "Houston",
        "San Diego State University": "San Diego State",
        "University of Miami": "Miami",
        "Florida Atlantic University": "Florida Atlantic",
        "Purdue University-Main Campus": "Purdue",
        "The University of Alabama": "Alabama",
        "North Carolina State University at Raleigh": "NC State",
    }

    total_application_data["SCHOOL_NAME"] = total_application_data["SCHOOL_NAME"].replace(update_school_names)

    # Align datasets
    mm_clean = mm_clean.rename(columns={
        "Mapped ESPN Team Name": "SCHOOL_NAME",
        "Season": "YEAR"
    })

    total_application_data = total_application_data.rename(
        columns={"APPLCN": "Num_Applicants_That_Year"}
    )

    # Merge current year applicants
    final_four_data = mm_clean.merge(
        total_application_data[["SCHOOL_NAME", "YEAR", "Num_Applicants_That_Year"]],
        on=["SCHOOL_NAME", "YEAR"],
        how="left"
    )

    # Prepare next-year applicants
    next_year = total_application_data.copy()
    next_year["YEAR"] = next_year["YEAR"] - 1
    next_year = next_year.rename(columns={
        "Num_Applicants_That_Year": "Num_Applicants_Next_Year"
    })

    next_year = next_year[["SCHOOL_NAME", "YEAR", "Num_Applicants_Next_Year"]]

    final_four_data = final_four_data.merge(
        next_year,
        on=["SCHOOL_NAME", "YEAR"],
        how="left"
    )

    # Convert types
    final_four_data["Num_Applicants_That_Year"] = pd.to_numeric(
        final_four_data["Num_Applicants_That_Year"], errors="coerce"
    ).astype("Int64")

    final_four_data["Num_Applicants_Next_Year"] = pd.to_numeric(
        final_four_data["Num_Applicants_Next_Year"], errors="coerce"
    ).astype("Int64")

    # Percent change
    final_four_data["Pct_Change"] = (
        (final_four_data["Num_Applicants_Next_Year"]
         - final_four_data["Num_Applicants_That_Year"])
        / final_four_data["Num_Applicants_That_Year"]
    ) * 100

    # Filter Final Four teams only
    final_four_data = final_four_data[final_four_data["Final Four?"] == 1]

    # Remove 2024
    final_four_data = final_four_data[final_four_data["YEAR"] < 2024]

    # Keep only needed columns
    final_four_data = final_four_data[
        [
            "SCHOOL_NAME",
            "YEAR",
            "Num_Applicants_That_Year",
            "Num_Applicants_Next_Year",
            "Pct_Change",
        ]
    ]

    # Save
    final_four_data.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    # Stats
    print("Average Percent Change:", final_four_data["Pct_Change"].mean())
    print("Top 5 increases:\n", final_four_data.nlargest(5, "Pct_Change"))

    return final_four_data

def plot_final_four_applications(final_four_data, save_path=None):
    """
    Creates visualization for Final Four applicant changes using matplotlib.
    """

    # Get top 5 schools
    top_5 = final_four_data.nlargest(5, "Pct_Change").copy()

    # Get average applicant change 
    avg_change = final_four_data["Pct_Change"].mean() 

    # Sort so smallest is at bottom, largest at top (nice horizontal bar order)
    top_5 = top_5.sort_values("Pct_Change", ascending=True)

    plt.figure(figsize=(10, 6))

    # Horizontal bar chart
    plt.barh(
        top_5["SCHOOL_NAME"],
        top_5["Pct_Change"]
    )

    # add line to display average 
    plt.axvline(avg_change, color="royalblue", linestyle="--", linewidth=1.5, label=f"Avg: {avg_change:.1f}%")
    plt.legend()

    plt.xlabel("Percent Change in Applicants (%)")
    plt.ylabel("School")
    plt.title("Top 5 Final Four Schools with Largest Applicant Increase")

    # Add value labels on bars
    for i, (school, value) in enumerate(zip(top_5["SCHOOL_NAME"], top_5["Pct_Change"])):
        plt.text(value, i, f"{value:.1f}%", va="center")

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_top14_applicant_jumps(final_four_data, save_path=None):
    """
    Shows top 7 increases and top 7 decreases in applicant % change
    for Final Four teams.
    """

    # Get top 7 increases and top 7 decreases
    top_pos = final_four_data.nlargest(7, "Pct_Change")
    top_neg = final_four_data.nsmallest(7, "Pct_Change")

    # Combine and sort
    combined = pd.concat([top_neg, top_pos])
    combined = combined.sort_values("Pct_Change", ascending=True)

    # Get average applicant change
    avg_change = final_four_data["Pct_Change"].mean()

    labels = combined["SCHOOL_NAME"] + " (" + combined["YEAR"].astype(str) + ")"
    values = combined["Pct_Change"]

    # Colors
    colors = ["red" if v < 0 else "green" for v in values]

    plt.figure(figsize=(8, 8))

    plt.barh(labels, values, color=colors)

    plt.axvline(0, color="black", linewidth=1)

    # Add line to display average
    plt.axvline(avg_change, color="royalblue", linestyle="--", linewidth=1.5, label=f"Avg: {avg_change:.1f}%")
    plt.legend()

    # Force symmetric x-axis limits around zero
    max_abs = max(abs(values.min()), abs(values.max()))
    plt.xlim(-max_abs * 1.1, max_abs * 1.1)

    plt.xlabel("Percent Change in Applicants (%)")
    plt.ylabel("School (Year)")
    plt.title("(1)", fontsize=20)

    # Value labels
    for i, v in enumerate(values):
        if v < 0:
            plt.text(v - 1, i, f"{v:.1f}%", va="center", ha="right")
        else:
            plt.text(v + 1, i, f"{v:.1f}%", va="center", ha="left")

    plt.xlim(-max_abs * 1.1, max_abs * 1.2)  # more room on the right

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

def get_applicant_data_post_2013(folder_path="Raw_Data/application_data", school_file="Raw_Data/school_nums.csv"):
    files = glob.glob(os.path.join(folder_path, "adm*.csv"))
    data = []

    # read application files
    for file in files:
        # Extract year
        filename = os.path.basename(file)
        year_match = re.search(r'(\d{4})', filename)
        if not year_match:
            continue
        year = int(year_match.group(1))

        # Read CSV
        df = pd.read_csv(file)

        # Clean Column Names
        df.columns = df.columns.str.strip().str.upper()

        # Select columns
        subset = df[['UNITID', 'APPLCN']].copy()
        subset['YEAR'] = year

        data.append(subset)

    applicant_df = pd.concat(data, ignore_index=True)

    # 3️⃣ Read school names mapping
    schools_df = pd.read_csv(school_file)
    schools_df.columns = schools_df.columns.str.strip().str.upper()  # normalize
    schools_df.rename(columns={"UNITID": "UNITID", "INSTITUTION NAME": "SCHOOL_NAME"}, inplace=True)

    # 4️⃣ Merge UNITID to get school names
    merged_df = applicant_df.merge(schools_df[['UNITID', 'SCHOOL_NAME']], on='UNITID', how='left')

    # Optional: put school name first
    merged_df = merged_df[['YEAR', 'UNITID', 'SCHOOL_NAME', 'APPLCN']]

    return merged_df

def get_applicant_data_pre_2014(folder_path="Raw_Data/application_data", school_file="Raw_Data/school_nums.csv"):
    files = glob.glob(os.path.join(folder_path, "ic*.csv"))
    data = []

    # read application files
    for file in files:
        # Extract year
        filename = os.path.basename(file)
        year_match = re.search(r'(\d{4})', filename)
        if not year_match:
            continue
        year = int(year_match.group(1))

        # Read CSV
        df = pd.read_csv(file)

        # Clean Column Names
        df.columns = df.columns.str.strip().str.upper()

        # Select columns
        subset = df[['UNITID', 'APPLCN']].copy()
        subset['YEAR'] = year

        data.append(subset)

    applicant_df = pd.concat(data, ignore_index=True)

    # 3️ Read school names mapping
    schools_df = pd.read_csv(school_file)
    schools_df.columns = schools_df.columns.str.strip().str.upper()  # normalize
    schools_df.rename(columns={"UNITID": "UNITID", "INSTITUTION NAME": "SCHOOL_NAME"}, inplace=True)

    # 4️ Merge UNITID to get school names
    merged_df = applicant_df.merge(schools_df[['UNITID', 'SCHOOL_NAME']], on='UNITID', how='left')

    # Optional: put school name first
    merged_df = merged_df[['YEAR', 'UNITID', 'SCHOOL_NAME', 'APPLCN']]

    return merged_df