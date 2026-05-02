import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from nba_api.stats.endpoints import leaguedashteamstats


def process_data(df: pd.DataFrame, start_year=2002, end_year=2026, delay=0.6):
    """
    Only does:
    - NCAA data loading + merge
    - NBA FG3 rate scraping
    """

    df = df.copy()

    # ── NBA SCRAPER ──────────────────────────────────────────────
    def make_season_str(end_year):
        return f"{end_year-1}-{str(end_year)[-2:]}"

    def fetch_fg3_rate(season):
        try:
            res = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star="Regular Season",
                per_mode_detailed="PerGame",
            )
            df_s = res.get_data_frames()[0]

            fg3a = df_s["FG3A"].sum()
            fga = df_s["FGA"].sum()
            rate = (fg3a / fga * 100) if fga else 0

            year = int(season.split("-")[1]) + 2000

            return {"year": year, "fg3_rate": round(rate, 2)}
        except Exception:
            return None

    nba_rows = []
    for y in range(start_year, end_year + 1):
        season = make_season_str(y)
        row = fetch_fg3_rate(season)
        if row:
            nba_rows.append(row)
        time.sleep(delay)

    nba_df = pd.DataFrame(nba_rows)
    nba_df.to_csv("Raw_Data/nba_fg3_rates.csv", index=False)  # add this line


    # ── Merge NBA + NCAA ─────────────────────────────────────────
    df_merged = df.merge(nba_df, left_on="Season", right_on="year", how="left")

    return df_merged, nba_df


def create_figure(df, nba_df, save_path=None):
    if nba_df is None or nba_df.empty:
        print("nba_df is empty — no NBA data to plot.")
        return

    years = nba_df["year"].values
    rates = nba_df["fg3_rate"].values

    plt.figure(figsize=(8, 8))

    ncaa = df.groupby("Season")["FG3Rate"].mean()
    plt.plot(ncaa.index, ncaa.values,
                color="orange", linewidth=2, label="NCAA Avg FG3 Rate")

    # ── NBA line explicitly last and on top ──
    plt.plot(years, rates,
             color="royalblue", linewidth=2.5, zorder=5, label="NBA FG3 Rate")

    plt.xlabel("Year")
    plt.ylabel("3PT Attempt Rate (%)")
    plt.title("(2)", fontsize=20)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # add a dotted line to show when Curry won first MVP
    plt.axvline(2015, color="black", linestyle=":", linewidth=1.5, label="2015")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()