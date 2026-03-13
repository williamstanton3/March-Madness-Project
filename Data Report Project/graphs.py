import os
import tempfile

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


def get_image() -> Image.Image:
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "fig.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        img = Image.fromarray(cv2.imread(p))
    plt.clf()
    plt.close("all")
    return img


def save(img: Image.Image, filename: str) -> None:
    filepath = os.path.join("data_graphs", filename)
    img.save(filepath)
    print(f"Saved {filepath}")


def plot_offense_vs_defense(df: pd.DataFrame) -> Image.Image:
    # compute AdjOE and AdjDE from the cleaned dataset
    d = df[["Adjusted Offensive Efficiency", "Adjusted Defensive Efficiency", "Tournament Championship?"]].dropna()
    d = d.rename(columns={
        "Adjusted Offensive Efficiency": "AdjOE",
        "Adjusted Defensive Efficiency": "AdjDE"
    })

    # mark champions (1 = yes)
    champ = d["Tournament Championship?"] == 1

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(d.loc[~champ, "AdjOE"], d.loc[~champ, "AdjDE"], alpha=0.25, s=18, label="All teams")
    ax.scatter(d.loc[champ, "AdjOE"], d.loc[champ, "AdjDE"], s=80, zorder=5,
               edgecolors="black", linewidths=0.6, label="Champions")
    ax.axvline(d["AdjOE"].median(), color="grey", lw=1, ls="--", alpha=0.6)
    ax.axhline(d["AdjDE"].median(), color="grey", lw=1, ls="--", alpha=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("AdjOE")
    ax.set_ylabel("AdjDE (lower = better)")
    ax.set_title("Offense vs. Defense")
    ax.legend()
    plt.tight_layout()
    return get_image()


def plot_3pt_revolution(df: pd.DataFrame) -> Image.Image:
    d = df[["Season", "FG3Rate", "OppFG3Rate"]].dropna()
    by_season = d.groupby("Season")[["FG3Rate", "OppFG3Rate"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(by_season["Season"], by_season["FG3Rate"], marker="o", ms=4, label="FG3Rate")
    ax.plot(by_season["Season"], by_season["OppFG3Rate"], marker="s", ms=4, ls="--", label="OppFG3Rate")
    ax.set_xlabel("Season")
    ax.set_ylabel("Avg 3PT Rate (%)")
    ax.set_title("3PT Rate Over Time")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return get_image()


def plot_adjEM_by_seed(df: pd.DataFrame) -> Image.Image:
    # compute AdjEM on the fly
    df["AdjEM"] = df["Adjusted Offensive Efficiency"] - df["Adjusted Defensive Efficiency"]
    d = df[["Seed", "AdjEM"]].copy()
    d["Seed"] = pd.to_numeric(d["Seed"], errors="coerce")
    d = d.dropna()
    d["Seed"] = d["Seed"].astype(int)
    d = d[d["Seed"].between(1, 16)]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.violinplot(data=d, x="Seed", y="AdjEM", hue="Seed", inner="quartile",
                   linewidth=0.8, ax=ax, legend=False)
    ax.axhline(0, color="grey", lw=1, ls="--", alpha=0.5)
    ax.set_xlabel("Seed")
    ax.set_ylabel("AdjEM")
    ax.set_title("AdjEM by Seed")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return get_image()


def plot_correlation_heatmap(df: pd.DataFrame) -> Image.Image:
    cols = ["Adjusted Offensive Efficiency", "Adjusted Defensive Efficiency", 
            "FG3Rate", "FG3Pct", "FG2Pct", "AvgHeight", "Seed"]
    d = df[[c for c in cols if c in df.columns]].copy()
    # compute AdjEM for correlation
    d["AdjEM"] = d["Adjusted Offensive Efficiency"] - d["Adjusted Defensive Efficiency"]
    for col in d.columns:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(d.corr(), dtype=bool))
    sns.heatmap(d.corr(), mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, linewidths=0.4,
                annot_kws={"size": 8}, ax=ax, cbar_kws={"shrink": 0.75})
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    return get_image()


def plot_final_four_by_seed(df: pd.DataFrame) -> Image.Image:
    d = df[["Seed", "Final Four?"]].copy()
    d["Seed"] = pd.to_numeric(d["Seed"], errors="coerce")
    d = d.dropna()
    d["Seed"] = d["Seed"].astype(int)
    d = d[d["Seed"].between(1, 16)]
    
    # Use integer values directly
    d["FF"] = (d["Final Four?"] == 1).astype(int)

    rate = d.groupby("Seed")["FF"].mean().reset_index()
    rate["FF"] *= 100

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(rate["Seed"].astype(str), rate["FF"])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Final Four Rate (%)")
    ax.set_title("Final Four Rate by Seed")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return get_image()

def main():
    clean_data = pd.read_csv("data/mm_clean.csv")
    print(f"Loaded {len(clean_data):,} rows x {len(clean_data.columns)} columns\n")

    plots = [
        ("plot1_championship_quadrant.png", plot_offense_vs_defense),
        ("plot2_3pt_revolution.png",        plot_3pt_revolution),
        ("plot3_adjEM_by_seed_violin.png",  plot_adjEM_by_seed),
        ("plot4_correlation_heatmap.png",   plot_correlation_heatmap),
        ("plot5_final_four_by_seed.png",    plot_final_four_by_seed),
    ]

    for filename, fn in plots:
        print(f"Generating {filename} ...")
        save(fn(clean_data), filename)

    print("\nDone.")


main()