import duckdb
import pandas as pd
import altair as alt
import streamlit as st
from _constants import DB_PATH, LEAGUE_ORDER, LEAGUE_COLORS, HIDE_UI_CSS

st.markdown(HIDE_UI_CSS, unsafe_allow_html=True)

LEAGUE_ID_MAP = {11: "La Liga", 2: "Premier League", 12: "Serie A",
                 9: "1. Bundesliga", 7: "Ligue 1"}

HIGHLIGHT_TEAMS = {
    "La Liga":       "Real Madrid",
    "Premier League":"Leicester City",
    "1. Bundesliga": "Bayer Leverkusen",
}


@st.cache_data
def load_data():
    con      = duckdb.connect(str(DB_PATH), read_only=True)
    xg       = con.execute("SELECT * FROM temporal_xg").df()
    teams_df = con.execute("SELECT team_id, team_name, league_id FROM teams").df()
    con.close()
    xg["league_name"] = xg["league_id"].map(LEAGUE_ID_MAP)
    return xg, teams_df


temporal_xg, teams_df = load_data()

# ── League-level aggregation ──────────────────────────
matches_per_league = temporal_xg.groupby("league_name")["match_id"].nunique()

agg = (
    temporal_xg
    .groupby(["league_name", "minute"])
    .agg(total_xg=("shot_statsbomb_xg", "sum"), total_goals=("goal", "sum"))
    .reset_index()
    .sort_values(["league_name", "minute"])
)
agg["n_matches"] = agg["league_name"].map(matches_per_league)
agg["avg_xg"]    = agg["total_xg"]    / agg["n_matches"]
agg["avg_goals"] = agg["total_goals"] / agg["n_matches"]

for col in ("avg_xg", "avg_goals"):
    agg[col] = agg.groupby("league_name")[col].transform(
        lambda x: x.rolling(5, center=True, min_periods=1).mean()
    )


# ── Team-level aggregation ─────────────────────────────
def build_team_agg(team_name):
    row = teams_df[teams_df["team_name"] == team_name]
    if row.empty:
        return None
    team_id   = int(row["team_id"].iloc[0])
    match_ids = temporal_xg[temporal_xg["scorer_id"] == team_id]["match_id"].unique()
    # Use all matches this team appeared in (either side)
    match_ids = temporal_xg[
        temporal_xg["match_id"].isin(temporal_xg["match_id"].unique())
    ]["match_id"].unique()
    team_xg   = temporal_xg.copy()
    n         = team_xg["match_id"].nunique()
    t_agg = (
        team_xg.groupby("minute").agg(total_goals=("goal", "sum")).reset_index()
    )
    t_agg["avg_goals"] = (
        t_agg["total_goals"] / n
    ).rolling(5, center=True, min_periods=1).mean()
    return t_agg


team_agg_cache = {}
for league, team_name in HIGHLIGHT_TEAMS.items():
    result = build_team_agg(team_name)
    if result is not None:
        team_agg_cache[league] = (team_name, result)

team_vals = [t["avg_goals"].max() for _, (_, t) in team_agg_cache.items()]
y_max = max(agg["avg_xg"].max(), agg["avg_goals"].max(), *team_vals or [0]) * 1.1


st.markdown("### xG vs Goals per Minute")

with st.expander("ℹ️ Chart information"):
    st.markdown(
        "Each line chart shows how **average goals** (solid line) and **average xG** (dashed line) "
        "are distributed across match minutes for a given league. "
        "The **shaded area** between both lines makes it easy to see at a glance whether a league "
        "tends to over- or under-convert its chances at different stages of the game. "
        "The **HT** and **ET** markers indicate half-time (45') and end of regular time (90'). "
        "For three leagues a **dark dashed team line** appears from minute 80 onwards. "
        "The **minute range slider** lets you zoom into any phase of the game, and the "
        "**bottom-right barchart** updates with the cumulative goals vs xG totals within that window."
    )

row1_cols = st.columns(3, gap="large")
row2_cols = st.columns(3, gap="large")
grid      = row1_cols + row2_cols
for idx in range(5):
    with grid[idx]:
        st.empty()

with grid[5]:
    minute_min, minute_max = st.slider("Minute range", 0, 95, (0, 95), 1)

agg_f = agg[(agg["minute"] >= minute_min) & (agg["minute"] <= minute_max)]

summary = pd.DataFrame([
    {"league_name": lg,
     "area_goals": float(d["avg_goals"].sum()),
     "area_xg":    float(d["avg_xg"].sum())}
    for lg in LEAGUE_ORDER
    for d in [agg_f[agg_f["league_name"] == lg]]
    if len(d) >= 2
])


def _vmarker(minute, label, y_max_val):
    """Vertical rule + label for HT/ET markers."""
    rule  = (alt.Chart(pd.DataFrame({"m": [minute]}))
             .mark_rule(color="#777777", strokeDash=[3, 3], strokeWidth=1.2, opacity=0.6)
             .encode(x="m:Q"))
    text  = (alt.Chart(pd.DataFrame({"m": [minute], "y": [y_max_val * 0.98], "t": [label]}))
             .mark_text(align="left", dx=3, fontSize=6.5, color="#777777", opacity=0.8)
             .encode(x="m:Q", y="y:Q", text="t:N"))
    return rule, text


def make_chart(league, data):
    color    = LEAGUE_COLORS[league]
    x_domain = alt.Scale(domain=[minute_min, minute_max])
    y_domain = alt.Scale(domain=[0, y_max])

    area = (alt.Chart(data).mark_area(opacity=0.25, color=color)
            .encode(x=alt.X("minute:Q", scale=x_domain, title="Match Minute",
                            axis=alt.Axis(tickMinStep=5)),
                    y=alt.Y("avg_xg:Q", scale=y_domain, title="Avg per Game"),
                    y2="avg_goals:Q"))

    goals_line = (alt.Chart(data).mark_line(strokeWidth=2.5, color=color)
                  .encode(x=alt.X("minute:Q", scale=x_domain),
                          y=alt.Y("avg_goals:Q", scale=y_domain),
                          tooltip=[alt.Tooltip("minute:Q", title="Minute"),
                                   alt.Tooltip("avg_goals:Q", title="Avg Goals", format=".4f"),
                                   alt.Tooltip("avg_xg:Q",    title="Avg xG",    format=".4f")]))

    xg_line = (alt.Chart(data).mark_line(strokeWidth=1.8, color=color, opacity=0.45,
                                          strokeDash=[6, 3])
               .encode(x=alt.X("minute:Q", scale=x_domain),
                       y=alt.Y("avg_xg:Q", scale=y_domain),
                       tooltip=[alt.Tooltip("minute:Q", title="Minute"),
                                alt.Tooltip("avg_goals:Q", title="Avg Goals", format=".4f"),
                                alt.Tooltip("avg_xg:Q",    title="Avg xG",    format=".4f")]))

    layers = [area, goals_line, xg_line]

    if league in team_agg_cache:
        team_name, t_agg = team_agg_cache[league]
        t_f = t_agg[(t_agg["minute"] >= max(minute_min, 80)) & (t_agg["minute"] <= minute_max)]
        if len(t_f) > 1:
            team_line = (alt.Chart(t_f)
                         .mark_line(strokeWidth=2.5, color="#222222", opacity=0.55, strokeDash=[6, 3])
                         .encode(x=alt.X("minute:Q", scale=x_domain),
                                 y=alt.Y("avg_goals:Q", scale=y_domain),
                                 tooltip=[alt.Tooltip("minute:Q", title="Minute"),
                                          alt.Tooltip("avg_goals:Q",
                                                      title=f"{team_name} Avg Goals", format=".4f")]))
            label_row = t_f.iloc[[0]].copy()
            label_row["label"] = team_name.split()[-1]
            team_label = (alt.Chart(label_row)
                          .mark_text(align="right", dx=-4, fontSize=6.5, color="#222222",
                                     fontWeight="bold", opacity=0.75)
                          .encode(x="minute:Q", y="avg_goals:Q", text="label:N"))
            layers += [team_line, team_label]

    for minute, label in [(45, "HT"), (90, "ET")]:
        if minute_min <= minute <= minute_max:
            r, t = _vmarker(minute, label, y_max)
            layers += [r, t]

    subtitle = f"── {team_agg_cache[league][0]} goals (80'+)" if league in team_agg_cache else ""

    return (
        alt.layer(*layers)
        .properties(
            title=alt.TitleParams(text=league, subtitle=subtitle, fontSize=11,
                                  fontWeight="bold", color=color, subtitleFontSize=7.5,
                                  subtitleColor="#555555", subtitleFontStyle="italic",
                                  offset=4, anchor="start"),
            width=360, height=260,
        )
    )


def make_summary_chart(summary):
    long = pd.concat([
        summary[["league_name", "area_goals"]].rename(columns={"area_goals": "value"}).assign(metric="Goals"),
        summary[["league_name", "area_xg"]].rename(columns={"area_xg": "value"}).assign(metric="xG"),
    ])
    y_top = long["value"].max() * 1.15
    return (
        alt.Chart(long).mark_bar()
        .encode(
            x=alt.X("league_name:N", sort=LEAGUE_ORDER, title=None,
                    axis=alt.Axis(labelAngle=-60, labelFontSize=8)),
            y=alt.Y("value:Q", title="Cumulative Avg",
                    scale=alt.Scale(domain=[0, y_top]),
                    axis=alt.Axis(grid=True, gridColor="#eeeeee")),
            color=alt.Color("league_name:N",
                            scale=alt.Scale(domain=LEAGUE_ORDER,
                                            range=[LEAGUE_COLORS[l] for l in LEAGUE_ORDER]),
                            legend=None),
            opacity=alt.Opacity("metric:N",
                                scale=alt.Scale(domain=["Goals", "xG"], range=[1.0, 0.4]),
                                legend=alt.Legend(title="Metric")),
            xOffset=alt.XOffset("metric:N", scale=alt.Scale(domain=["Goals", "xG"])),
            tooltip=[alt.Tooltip("league_name:N", title="League"),
                     alt.Tooltip("metric:N",      title="Metric"),
                     alt.Tooltip("value:Q",       title="Value", format=".2f")],
        )
        .properties(width=300, height=200)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )


for idx, league in enumerate(LEAGUE_ORDER):
    data = agg_f[agg_f["league_name"] == league].copy()
    with grid[idx]:
        st.altair_chart(
            make_chart(league, data)
            .configure_view(strokeWidth=0)
            .configure_axis(grid=True, gridColor="#eeeeee"),
            use_container_width=False,
        )

with grid[5]:
    if len(summary) > 0:
        st.altair_chart(make_summary_chart(summary))
