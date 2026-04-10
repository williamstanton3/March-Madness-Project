# QUESTION: Do schools whose basketball team makes a final four run in march madness see an increase in applicants over the next two years?

# the only columns I need to keep are UNITID (school identifier), APPLCN (applicants total), and year (from csv title)

import pandas as pd
import glob
import os
import re
import plotly.express as px


def main():
    # Load March Madness data
    mm_clean = pd.read_csv("data/mm_clean.csv")

    # Load application data
    applicant_data_post_2013 = get_applicant_data_post_2013()
    applicant_data_pre_2014 = get_applicant_data_pre_2014()

    # Combine datasets
    total_application_data = pd.concat([applicant_data_pre_2014, applicant_data_post_2013], ignore_index=True)

    # Map school names to match ESPN names
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

    total_application_data['SCHOOL_NAME'] = total_application_data['SCHOOL_NAME'].replace(update_school_names)

    # Rename columns in mm_clean to match application data
    mm_clean = mm_clean.rename(columns={'Mapped ESPN Team Name': 'SCHOOL_NAME', 'Season': 'YEAR'})

    # Rename applicant columns for clarity
    total_application_data = total_application_data.rename(columns={'APPLCN': 'Num_Applicants_That_Year'})

    # Merge Final Four schools with applicant data
    final_four_data = mm_clean.merge(total_application_data[['SCHOOL_NAME', 'YEAR', 'Num_Applicants_That_Year']],
                                     on=['SCHOOL_NAME', 'YEAR'],
                                     how='left')

    # Prepare next year's applicants
    next_year = total_application_data.copy()
    next_year['YEAR'] = next_year['YEAR'] - 1  # shift year to match next year
    next_year = next_year.rename(columns={'Num_Applicants_That_Year': 'Num_Applicants_Next_Year'})
    next_year = next_year[['SCHOOL_NAME', 'YEAR', 'Num_Applicants_Next_Year']]

    # Merge next year's applicants
    final_four_data = final_four_data.merge(next_year, on=['SCHOOL_NAME', 'YEAR'], how='left')

    # Convert applicant columns to integers
    final_four_data['Num_Applicants_That_Year'] = pd.to_numeric(final_four_data['Num_Applicants_That_Year'], errors='coerce').astype('Int64')
    final_four_data['Num_Applicants_Next_Year'] = pd.to_numeric(final_four_data['Num_Applicants_Next_Year'], errors='coerce').astype('Int64')

    # Calculate percent change
    final_four_data['Pct_Change'] = (
        final_four_data['Num_Applicants_Next_Year'] - final_four_data['Num_Applicants_That_Year']
    ) / final_four_data['Num_Applicants_That_Year'] * 100

    # only save teams that made the final four 
    final_four_data = final_four_data[final_four_data["Final Four?"] == 1]

    # Remove 2024 teams since we don't have next-year applicant data
    final_four_data = final_four_data[final_four_data['YEAR'] < 2024]   

    # only save columns we care about for this question (Team Name, Season, Applicants that year, Applicants next year, percent change)
    final_four_data = final_four_data[['SCHOOL_NAME', 'YEAR', 'Num_Applicants_That_Year', 'Num_Applicants_Next_Year', 'Pct_Change']]

    # Save results
    final_four_data.to_csv("data/final_four_applicant_changes.csv", index=False)
    print("Done! Final four data with applicant changes saved.")

    # do some math to evaluate the percent change 
    avg_percent_change = final_four_data["Pct_Change"].mean()
    print(f"Average Percent Change in Applicants Next Year: {avg_percent_change}")

    # print the top 5 schools that had the largest increase in applicants 
    top_5_change = final_four_data.nlargest(5, "Pct_Change") 
    print(f"Top 5 Schools with the largest increase in applicants: {top_5_change}")


    # Make a copy to avoid modifying the original
    top_5 = top_5_change.copy()

    # Create horizontal bar chart
    fig = px.bar(
        top_5,
        x='Pct_Change',
        y='SCHOOL_NAME',
        orientation='h',  # horizontal bars
        text='Pct_Change',  # show values on bars
        labels={'Pct_Change': 'Percent Change in Applicants (%)', 'SCHOOL_NAME': 'School'},
        title='Top 5 Final Four Schools with Largest Applicant Increase',
        color='Pct_Change',  # optional: color bars by value
        color_continuous_scale='Viridis'
    )

    # Reverse y-axis so the largest is on top
    fig.update_layout(yaxis={'categoryorder':'total ascending'})

    # Show interactive figure
    fig.show()




def get_applicant_data_post_2013(folder_path="data/application_data", school_file="data/school_nums.csv"):
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

def get_applicant_data_pre_2014(folder_path="data/application_data", school_file="data/school_nums.csv"):
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

    # 3️⃣ Read school names mapping
    schools_df = pd.read_csv(school_file)
    schools_df.columns = schools_df.columns.str.strip().str.upper()  # normalize
    schools_df.rename(columns={"UNITID": "UNITID", "INSTITUTION NAME": "SCHOOL_NAME"}, inplace=True)

    # 4️⃣ Merge UNITID to get school names
    merged_df = applicant_df.merge(schools_df[['UNITID', 'SCHOOL_NAME']], on='UNITID', how='left')

    # Optional: put school name first
    merged_df = merged_df[['YEAR', 'UNITID', 'SCHOOL_NAME', 'APPLCN']]

    return merged_df

if __name__ == "__main__":
    main()