import pandas as pd

MAIN_FILE = "data/mm_clean.csv"
KEY_FILE = "data/ncaa_d1_mapping.csv"
OUTPUT_FILE = "data/merged_data.csv"

main_df = pd.read_csv(MAIN_FILE)
key_df = pd.read_csv(KEY_FILE)

key_df = key_df.rename(columns={"public_private": "School Type"})

df = pd.merge(main_df, key_df, on="Mapped ESPN Team Name", how="left")

unmatched = df["School Type"].isna().sum()
if unmatched > 0:
    print(f"Warning: {unmatched} rows could not be matched and will be dropped")

df = df.dropna(subset=["School Type"])
df.to_csv(OUTPUT_FILE, index=False)
print(f"Merged {len(df)} rows into {OUTPUT_FILE}")
