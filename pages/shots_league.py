import io
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
from _constants import DB_PATH, LEAGUE_ORDER, LEAGUE_COLORS, HIDE_UI_CSS, L_SCALE, W_SCALE
from _pitch import draw_pitch_lines, PITCH_W, PITCH_H

st.markdown(HIDE_UI_CSS, unsafe_allow_html=True)


@st.cache_data
def load_data():
    con          = duckdb.connect(str(DB_PATH), read_only=True)
    shots_league = con.execute("SELECT * FROM shots_league").df()
    shots        = con.execute("SELECT * FROM shots").df()
    match_scores = con.execute("SELECT match_id, league_id FROM match_scores").df()
    teams        = con.execute("SELECT team_id, league_id, league_name FROM teams").df()
    con.close()
    return shots_league, shots, match_scores, teams


shots_league, shots, match_scores, teams = load_data()

if "shot_outcome" not in st.session_state:
    st.session_state["shot_outcome"] = "All"

OUTCOME_MAP = {"Scored": "shot_scored", "Saved": "shot_saved",
               "Missed": "shot_missed", "Blocked": "shot_blocked"}

games_per_league = (
    match_scores
    .merge(teams[["league_id", "league_name"]].drop_duplicates("league_id"), on="league_id", how="left")
    .groupby("league_name")["match_id"].nunique()
    .reset_index(name="num_games")
)


def build_table_df(shots_f):
    raw = shots_f.groupby(["league_name", "end_type"]).size().reset_index(name="count")
    raw["end_type"] = raw["end_type"].map(
        {"shot_scored": "Scored", "shot_saved": "Saved",
         "shot_missed": "Missed", "shot_blocked": "Blocked"}
    ).fillna(raw["end_type"])
    pivot = raw.pivot(index="league_name", columns="end_type", values="count").fillna(0).reset_index()
    for col in ["Scored", "Saved", "Missed", "Blocked"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["Total"] = pivot[["Scored", "Saved", "Missed", "Blocked"]].sum(axis=1)
    pivot = pivot.merge(games_per_league, on="league_name", how="left")
    pivot["Shots"] = (pivot["Total"] / pivot["num_games"]).round(1)
    for col in ["Scored", "Saved", "Missed", "Blocked"]:
        pivot[col] = (pivot[col] / pivot["Total"] * 100).round(1)
    return (
        pivot[["league_name", "Shots", "Scored", "Saved", "Missed", "Blocked"]]
        .rename(columns={"league_name": "League"})
        .sort_values("League").reset_index(drop=True)
    )


def render_league_pitch(league, shots_df):
    colour  = LEAGUE_COLORS[league]
    lg      = shots_df[shots_df["league_name"] == league].dropna(subset=["end_x", "end_y"])
    fig, ax = plt.subplots(figsize=(4.5, 3.0), facecolor="white")
    ax.set_xlim(0, 120); ax.set_ylim(0, 80)
    draw_pitch_lines(ax)
    if not lg.empty:
        ax.scatter(lg["end_x"], lg["end_y"], s=6, color=colour, edgecolors="none",
                   alpha=0.15, zorder=6)
    ax.set_title(league, color=colour, fontsize=8, fontweight="bold", pad=3, loc="center")
    fig.tight_layout(pad=0.4)
    return fig


def make_table_figure(table_df):
    AX = dict(left=0.038796, bottom=0.018519, width=0.922408, height=0.922407)
    fig = plt.figure(figsize=(4.5, 3.0), facecolor="white")
    ax  = fig.add_axes([AX["left"], AX["bottom"], AX["width"], AX["height"]])
    ax.axis("off")
    ax.set_title("Per Game Metrics", fontsize=8, fontweight="bold", pad=3,
                 loc="center", color="#333333")

    col_labels = ["League", "Shots", "Scored%", "Saved%", "Missed%", "Blocked%"]
    col_widths = [0.26, 0.12, 0.155, 0.155, 0.155, 0.155]
    cell_data, cell_colors = [], []
    for _, row in table_df.iterrows():
        c = LEAGUE_COLORS.get(row["League"], "#ffffff")
        cell_data.append([row["League"],
                          f"{row['Shots']:.1f}",
                          f"{row['Scored']:.1f}%", f"{row['Saved']:.1f}%",
                          f"{row['Missed']:.1f}%", f"{row['Blocked']:.1f}%"])
        cell_colors.append([c + "22"] * 6)

    tbl = ax.table(cellText=cell_data, colLabels=col_labels, cellLoc="center",
                   loc="center", cellColours=cell_colors, colWidths=col_widths,
                   bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(fontweight="bold", fontsize=7.5)
        elif c == 0:
            cell.set_text_props(color=LEAGUE_COLORS.get(cell_data[r - 1][0], "#333333"),
                                fontweight="bold", fontsize=7)
    return fig


st.markdown("### Spatial Distribution of Shots and per-game Metrics")

with st.expander("ℹ️ Chart information"):
    st.markdown(
        "Each pitch shows the **spatial distribution of shot locations** for a given league. "
        "Each dot is a single shot, with low opacity so denser areas become naturally darker. "
        "The **bottom-right table** summarises per-game shot volume and outcome breakdown. "
        "Use the **filter at the bottom** to restrict all panels to a single shot outcome."
    )

selected = st.session_state["shot_outcome"]
shots_f  = shots if selected == "All" else shots[shots["end_type"] == OUTCOME_MAP[selected]]
table_df = build_table_df(shots_f)

row1  = st.columns(3, gap="small")
row2  = st.columns(3, gap="small")
slots = row1 + row2

for slot, league in zip(slots[:5], LEAGUE_ORDER):
    with slot:
        fig = render_league_pitch(league, shots_f)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

with slots[5]:
    fig_t = make_table_figure(table_df)
    st.pyplot(fig_t, use_container_width=True)
    plt.close(fig_t)

st.markdown("---")
st.selectbox("Filter by shot outcome", ["All", "Scored", "Saved", "Missed", "Blocked"],
             key="shot_outcome")
