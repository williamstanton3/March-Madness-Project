"""
NBA FG3 Rate Scraper — stats.nba.com API
Fetches league-average 3-point attempt rate (FG3A / FGA) for each season
from 2001-02 through 2025-26 using the official NBA stats API.

Install dependencies:
    pip install nba_api pandas
"""

import time
import csv
import json
from nba_api.stats.endpoints import leaguedashteamstats
import matplotlib.pyplot as plt
import numpy as np



# Seasons to fetch: "2001-02" → "2025-26"
START_YEAR = 2002   # end-year of first season
END_YEAR   = 2026   # end-year of last season

# nba_api needs a small delay between requests or the API rate-limits you
REQUEST_DELAY = 0.6  # seconds


def make_season_str(end_year: int) -> str:
    """Convert end-year int to NBA season string, e.g. 2002 → '2001-02'."""
    start = end_year - 1
    return f"{start}-{str(end_year).zfill(4)[-2:]}"


def fetch_season_fg3_rate(season: str) -> dict | None:
    """
    Fetch league-wide FG3A and FGA for a single season by summing all teams,
    then compute FG3Rate = FG3A / FGA.

    Returns a dict or None on failure.
    """
    try:
        result = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
        )
        df = result.get_data_frames()[0]

        total_fg3a = df["FG3A"].sum()
        total_fga  = df["FGA"].sum()
        fg3_rate   = round(total_fg3a / total_fga * 100, 2) if total_fga else 0.0

        # Extract end year from season string (e.g., "2001-02" → 2002)
        end_year = int(season.split("-")[1]) + 2000

        return {
            "season":   season,
            "year":     end_year,
            "fg3a":     round(total_fg3a / len(df), 1),
            "fga":      round(total_fga  / len(df), 1),
            "fg3_rate": fg3_rate,
        }
    except Exception as e:
        print(f"  WARNING: Could not fetch {season} — {e}")
        return None


def scrape_fg3_rate(start_year: int = START_YEAR, end_year: int = END_YEAR) -> list[dict]:
    """
    Iterate over each season in [start_year, end_year] and collect FG3Rate data.
    Returns a list of dicts sorted by year.
    """
    seasons = [make_season_str(y) for y in range(start_year, end_year + 1)]
    results = []

    print(f"Fetching {len(seasons)} seasons from stats.nba.com...\n")

    for i, season in enumerate(seasons):
        print(f"  [{i+1}/{len(seasons)}] {season}", end=" ... ", flush=True)
        row = fetch_season_fg3_rate(season)
        if row:
            results.append(row)
            print(f"FG3Rate = {row['fg3_rate']}%")
        time.sleep(REQUEST_DELAY)

    return results


def print_table(data: list[dict]) -> None:
    print(f"\n{'Season':<10} {'Year':<6} {'FG3A/g':>7} {'FGA/g':>7} {'FG3Rate':>9}")
    print("-" * 45)
    for row in data:
        print(
            f"{row['season']:<10} {row['year']:<6} "
            f"{row['fg3a']:>7.1f} {row['fga']:>7.1f} {row['fg3_rate']:>8.1f}%"
        )


def save_csv(data: list[dict], filepath: str = "nba_fg3_rate.csv") -> None:
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["season", "year", "fg3a", "fga", "fg3_rate"])
        writer.writeheader()
        writer.writerows(data)
    print(f"\nCSV saved → {filepath}")


def save_json(data: list[dict], filepath: str = "nba_fg3_rate.json") -> None:
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"JSON saved → {filepath}")

def plot_fg3_rate(data: list[dict]) -> None:
    years = np.array([row["year"] for row in data])
    rates = np.array([row["fg3_rate"] for row in data])

    plt.figure()
    plt.plot(years, rates)

    # Trendline
    z = np.polyfit(years, rates, 1)
    p = np.poly1d(z)
    plt.plot(years, p(years), linestyle="--")

    plt.xlabel("Year")
    plt.ylabel("FG3 Rate (%)")
    plt.title("NBA 3-Point Attempt Rate Over Time")
    plt.grid()

    plt.show()

if __name__ == "__main__":
    data = scrape_fg3_rate()
    print_table(data)
    save_csv(data)
    save_json(data)
    plot_fg3_rate(data)