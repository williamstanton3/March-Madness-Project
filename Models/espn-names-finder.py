import pandas as pd

df = pd.read_csv('../Data Report Project/data/mm.csv')
unique_schools = df['Mapped ESPN Team Name'].unique()  # Replace 'school_name' with your actual column name
print('\n'.join(unique_schools))

