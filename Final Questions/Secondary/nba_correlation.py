import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

df = pd.read_csv("Initial Models/data/mm_clean.csv")

def get_round(row):
    if row.get("Tournament Winner?") == 1:
        return "Champion"
    elif row.get("Tournament Championship?") == 1:
        return "Runner Up"
    elif row.get("Final Four?") == 1:
        return "Final Four"
    else:
        return "Did Not Make Final Four"

df["Round"] = df.apply(get_round, axis=1)

round_order = ["Champion", "Runner Up", "Final Four", "Did Not Make Final Four"]
round_colors = {
    "Champion": "#534AB7",
    "Runner Up": "#1D9E75",
    "Final Four": "#D85A30",
    "Did Not Make Final Four": "#B4B2A9"
}

x = df["Adjusted Offensive Efficiency"].dropna()
y = df.loc[x.index, "Adjusted Defensive Efficiency"].dropna()
idx = x.index.intersection(y.index)
x, y = df.loc[idx, "Adjusted Offensive Efficiency"], df.loc[idx, "Adjusted Defensive Efficiency"]
m, b = np.polyfit(x, y, 1)
x_range = np.linspace(x.min(), x.max(), 100)

fig = go.Figure()

for round_label in round_order:
    subset = df[df["Round"] == round_label]
    fig.add_trace(go.Scatter(
        x=subset["Adjusted Offensive Efficiency"],
        y=subset["Adjusted Defensive Efficiency"],
        mode="markers",
        name=round_label,
        text=subset.apply(lambda r: f"{r.get('Mapped ESPN Team Name','?')} ({int(r['Season'])})<br>Seed: {r.get('Seed','?')}<br>Off: {r['Adjusted Offensive Efficiency']:.1f} | Def: {r['Adjusted Defensive Efficiency']:.1f}<br>{r['Round']}", axis=1),
        hoverinfo="text",
        marker=dict(color=round_colors[round_label], size=7, opacity=0.8, line=dict(width=0.5, color="white"))
    ))

fig.add_trace(go.Scatter(
    x=x_range, y=m * x_range + b,
    mode="lines", name="Trend",
    line=dict(color="#888780", width=1.5, dash="dot"),
    hoverinfo="skip", showlegend=False
))

fig.update_layout(
    title="Offense vs Defense: Final Four Appearances",
    xaxis_title="Adjusted Offensive Efficiency",
    yaxis_title="Adjusted Defensive Efficiency (lower = better)",
    yaxis_autorange="reversed",
    legend_title="Tournament Round",
    hoverlabel=dict(font_size=13),
    width=1100, height=650
)

fig.write_image("offense_vs_defense_final_four.png", width=1100, height=650, scale=2)