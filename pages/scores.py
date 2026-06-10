import duckdb
import pandas as pd
import altair as alt
import streamlit as st
from _constants import DB_PATH, LEAGUE_ORDER, LEAGUE_COLORS, HIDE_UI_CSS

st.markdown(HIDE_UI_CSS, unsafe_allow_html=True)

CHART_W = 210
CHART_H = 195


@st.cache_data
def load_data():
    con          = duckdb.connect(str(DB_PATH), read_only=True)
    match_scores = con.execute("SELECT * FROM match_scores").df()
    matches      = con.execute("SELECT match_id, home_team_id, away_team_id, league_id FROM matches").df()
    teams        = con.execute("SELECT team_id, team_name, league_id, league_name FROM teams").df()
    con.close()
    return match_scores, matches, teams


match_scores, matches, teams = load_data()

name_map   = teams.set_index("team_id")["team_name"]
league_map = teams.drop_duplicates("league_id").set_index("league_id")["league_name"]

df = (
    match_scores
    .merge(matches, on="match_id", how="inner")
    .assign(
        home_team   = lambda d: d["home_team_id"].map(name_map),
        away_team   = lambda d: d["away_team_id"].map(name_map),
        league_name = lambda d: d["league_id"].map(league_map),
        goal_diff   = lambda d: (d["home_score"] - d["away_score"]).abs(),
    )
    .dropna(subset=["home_team", "away_team"])
)
df["result"] = df.apply(
    lambda r: "Home Win" if r.home_score > r.away_score
              else ("Draw" if r.home_score == r.away_score else "Away Win"),
    axis=1,
)

st.markdown("### Home vs Away Scorelines")

with st.expander("ℹ️ Chart information"):
    st.markdown(
        "Each scatter plot shows the **distribution of final scorelines** for a given league. "
        "The **x-axis** represents away team goals and the **y-axis** represents home team goals, "
        "so any point above the diagonal dashed line is a home win, on the line is a draw, "
        "and below the line is an away win. "
        "**Dot size** encodes how frequently that exact scoreline occurred. "
        "The **🏠 / ═ / ✈ percentages** above each chart show the overall split of home wins, draws "
        "and away wins for that league. "
        "The **min goal difference slider** removes low-margin results to isolate more decisive outcomes."
    )

row1_cols = st.columns(3, gap="large")
row2_cols = st.columns(3, gap="large")
grid      = row1_cols + row2_cols

for idx, league in enumerate(LEAGUE_ORDER):
    with grid[idx]:
        st.empty()

max_diff = int(df["goal_diff"].max())
with grid[5]:
    st.markdown("#### Filters")
    min_diff = st.slider("Min goal difference", 0, max_diff, 0, 1)

df_f = df[df["goal_diff"] >= min_diff]

agg = (
    df_f.groupby(["league_name", "home_score", "away_score", "result"])
    .size().reset_index(name="count")
)
summary = (
    df_f.groupby(["league_name", "result"]).size().reset_index(name="n")
    .merge(df_f.groupby("league_name").size().reset_index(name="total"), on="league_name")
    .assign(pct=lambda d: (d["n"] / d["total"] * 100).round(1))
)

max_goals        = int(max(agg["home_score"].max(), agg["away_score"].max())) if len(agg) else 8
global_max_count = int(agg["count"].max()) if len(agg) else 1
diag             = pd.DataFrame({"x": [0, max_goals], "y": [0, max_goals]})


def make_chart(league):
    color = LEAGUE_COLORS[league]
    data  = agg[agg["league_name"] == league]
    s     = summary[summary["league_name"] == league].set_index("result")["pct"]

    stats_text = (
        alt.Chart(pd.DataFrame([{"label": f"🏠 {s.get('Home Win', 0)}%    ═ {s.get('Draw', 0)}%    ✈ {s.get('Away Win', 0)}%"}]))
        .mark_text(align="center", baseline="middle", fontSize=11, color="#444444")
        .encode(text="label:N")
        .properties(width=CHART_W, height=16)
    )

    diag_line = (
        alt.Chart(diag)
        .mark_line(color="#cccccc", strokeDash=[4, 4], strokeWidth=1)
        .encode(x="x:Q", y="y:Q")
    )

    axis_kwargs = dict(tickMinStep=1, labelFontSize=7, titleFontSize=8)
    scatter = (
        alt.Chart(data)
        .mark_circle(opacity=0.85, color=color)
        .encode(
            x=alt.X("away_score:Q", title="Away Goals",
                    scale=alt.Scale(domain=[-0.5, max_goals + 0.5]),
                    axis=alt.Axis(**axis_kwargs)),
            y=alt.Y("home_score:Q", title="Home Goals",
                    scale=alt.Scale(domain=[-0.5, max_goals + 0.5]),
                    axis=alt.Axis(**axis_kwargs)),
            size=alt.Size("count:Q", title="# Matches",
                          scale=alt.Scale(domain=[0, global_max_count], range=[20, 400]),
                          legend=None),
            tooltip=[
                alt.Tooltip("home_score:Q", title="Home Goals"),
                alt.Tooltip("away_score:Q", title="Away Goals"),
                alt.Tooltip("result:N",     title="Result"),
                alt.Tooltip("count:Q",      title="# Matches"),
            ],
        )
    )

    return (
        alt.vconcat(
            stats_text,
            (diag_line + scatter).properties(
                title=alt.TitleParams(league, fontSize=16, fontWeight="bold", color=color),
                width=CHART_W, height=CHART_H,
            ),
            spacing=2,
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=True, gridColor="#eeeeee")
    )


for idx, league in enumerate(LEAGUE_ORDER):
    with grid[idx]:
        st.altair_chart(make_chart(league), use_container_width=False)
