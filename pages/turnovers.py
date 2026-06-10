import duckdb
import pandas as pd
import altair as alt
import streamlit as st
from _constants import DB_PATH, LEAGUE_ORDER, LEAGUE_COLORS, HIDE_UI_CSS

st.markdown(HIDE_UI_CSS, unsafe_allow_html=True)
st.markdown("### Turnovers Leading to the Other Half")

with st.expander("ℹ️ Chart information"):
    st.markdown(
        "This chart shows two metrics per league, both averaged per game: "
        "**Total Turnovers** (faded bar) and **Turnovers Leading to the Opposite Half** (solid bar). "
        "A **turnover** is a change of possession where play continues without the ball going out of bounds "
        "or the referee stopping the game. "
        "The **opposite half** metric counts only those turnovers where the ball then crossed the halfway "
        "line — a high value here indicates a more open, end-to-end style of play. "
        "Use the **turnover type filter** to isolate specific ways possession is lost. "
        "The **order leagues by** selector re-sorts the bars by either total volume or opposite-half impact."
    )

ALL_TYPES = [
    "failed_pass", "interception", "failed_dribble",
    "lost_duel", "miscontrol", "dispossessed", "clearance", "block",
]

if "selected_types" not in st.session_state:
    st.session_state["selected_types"] = ALL_TYPES
if "sort_criterion" not in st.session_state:
    st.session_state["sort_criterion"] = "Total Turnovers"


@st.cache_data
def load_data():
    con   = duckdb.connect(str(DB_PATH), read_only=True)
    seqs  = con.execute("SELECT * FROM sequences").df()
    teams = con.execute("SELECT team_id, league_name FROM teams").df()
    con.close()
    df = seqs.merge(teams, on="team_id", how="left").sort_values(["match_id", "possession"]).reset_index(drop=True)
    df["next_start_x"] = df.groupby("match_id")["start_x"].shift(-1)
    df["next_team_id"] = df.groupby("match_id")["team_id"].shift(-1)
    return df


sequences = load_data()

selected_types = st.session_state["selected_types"]
sort_criterion = st.session_state["sort_criterion"]

if not selected_types:
    st.warning("Select at least one turnover type.")
    st.stop()

active = sequences[sequences["end_type"].isin(selected_types)].copy()
active["opposite_half"] = (
    (active["next_team_id"] != active["team_id"]) &
    (
        ((active["end_x"] < 60) & (active["next_start_x"] >= 60)) |
        ((active["end_x"] >= 60) & (active["next_start_x"] < 60))
    )
)

agg = (
    active.groupby(["match_id", "league_name"])
    .agg(total=("sequence_id", "count"), opp_half=("opposite_half", "sum"))
    .reset_index()
    .groupby("league_name")
    .agg(total_per_game=("total", "mean"), opp_half_per_game=("opp_half", "mean"))
    .reset_index()
)

sort_col     = "total_per_game" if sort_criterion == "Total Turnovers" else "opp_half_per_game"
league_order = agg.sort_values(sort_col, ascending=False)["league_name"].tolist()

agg_long = pd.concat([
    agg[["league_name", "total_per_game"]].rename(columns={"total_per_game": "value"}).assign(metric="Total Turnovers"),
    agg[["league_name", "opp_half_per_game"]].rename(columns={"opp_half_per_game": "value"}).assign(metric="Led to Opposite Half"),
], ignore_index=True)

chart = (
    alt.Chart(agg_long)
    .mark_bar()
    .encode(
        y=alt.Y("league_name:N", sort=league_order, title=None, axis=alt.Axis(labelFontSize=11)),
        x=alt.X("value:Q", title="Turnovers per Game", axis=alt.Axis(labelFontSize=10)),
        color=alt.Color("league_name:N",
                        scale=alt.Scale(domain=LEAGUE_ORDER,
                                        range=[LEAGUE_COLORS[l] for l in LEAGUE_ORDER]),
                        legend=None),
        opacity=alt.Opacity("metric:N",
                            scale=alt.Scale(domain=["Total Turnovers", "Led to Opposite Half"],
                                            range=[0.35, 0.9]),
                            legend=alt.Legend(title="Metric", orient="bottom", offset=10)),
        yOffset=alt.YOffset("metric:N",
                            scale=alt.Scale(domain=["Total Turnovers", "Led to Opposite Half"])),
        tooltip=[
            alt.Tooltip("league_name:N", title="League"),
            alt.Tooltip("metric:N",      title="Metric"),
            alt.Tooltip("value:Q",       title="Per Game", format=".1f"),
        ],
    )
    .properties(width=900, height=380, title=f"Ordered by: {sort_criterion}")
    .configure_view(stroke=None)
    .configure_axis(grid=False)
)

st.altair_chart(chart, use_container_width=True)

st.markdown("---")
col_left, col_right = st.columns([2, 1])
with col_left:
    st.multiselect("Turnover types", options=ALL_TYPES,
                   format_func=lambda x: x.replace("_", " ").title(),
                   key="selected_types")
with col_right:
    st.selectbox("Order leagues by", ["Total Turnovers", "Led to Opposite Half"],
                 key="sort_criterion")
