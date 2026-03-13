# import libraries
import pandas as pd

# import and load data 

mm_data = pd.read_csv("../Data/DEV _ March Madness.csv")
print(mm_data.head())

# find out info about the columns
print(len(mm_data.columns)) # 165 columns 
print(mm_data.columns)

# print all the UMD records
print(mm_data[mm_data["Full Team Name"] == "Maryland Terrapins"]["Season"])

# show me every team that made the final four without being top 50 in Avg Possession Length (Defense) Rank
print(mm_data.loc[(mm_data["Avg Possession Length (Defense) Rank"] < 50) & (mm_data["Final Four?"] == "Yes"), ["Full Team Name", "Season"]])
# teams want an Avg Possession Length (Defense) Rank to be high. Higher rankings mean their average defensive possesions were longer, meaning the defense was better
# only two teams (2018 Villanova and 2023 Uconn), made the final four without a top 50 ranking in avg possesion length defense
