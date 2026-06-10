import duckdb
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from _constants import DB_PATH, LEAGUE_ORDER, LEAGUE_COLORS, HIDE_UI_CSS

st.markdown(HIDE_UI_CSS, unsafe_allow_html=True)


@st.cache_data
def load_data():
    con         = duckdb.connect(str(DB_PATH), read_only=True)
    temporal_xg = con.execute("SELECT match_id, minute, goal, scorer_id, home_score, away_score FROM temporal_xg").df()
    subs        = con.execute("SELECT game_id, team_id, sub_in_id FROM substitutions").df()
    teams       = con.execute("SELECT team_id, team_name, league_name FROM teams").df()
    matches     = con.execute("SELECT match_id, home_team_id, away_team_id FROM matches").df()
    players     = con.execute("SELECT player_id, team_id FROM players").df()
    passes      = con.execute("SELECT match_id, team_id, player_id, pass_successful, end_x FROM passes").df()
    con.close()
    return temporal_xg, subs, teams, matches, players, passes


@st.cache_data
def precompute(temporal_xg, subs, teams, matches, players, passes):
    # ── Per-team player sets ───────────────────────────
    sub_players = subs[["team_id", "sub_in_id"]].rename(columns={"sub_in_id": "player_id"})
    reg_players = players[["team_id", "player_id"]].copy()
    reg_players["player_id"] = reg_players["player_id"].astype("Int64")
    all_team_players = pd.concat([sub_players, reg_players]).drop_duplicates()

    # ── Sub lookup set: (match_id, player_id) ─────────
    sub_set = set(zip(
        *subs[["game_id", "sub_in_id"]].rename(columns={"game_id": "match_id"}).values.T
    ))

    # ── Match → team + is_home ─────────────────────────
    home = matches[["match_id", "home_team_id"]].rename(columns={"home_team_id": "team_id"}).assign(is_home=True)
    away = matches[["match_id", "away_team_id"]].rename(columns={"away_team_id": "team_id"}).assign(is_home=False)
    match_team = pd.concat([home, away])

    # ── Goals — filter to known scorers ───────────────
    goals = (
        temporal_xg[(temporal_xg["goal"] == 1) & temporal_xg["scorer_id"].notna()].copy()
    )
    goals["scorer_id"] = goals["scorer_id"].astype(int)
    goals = (
        goals.merge(match_team, on="match_id", how="inner")
             .merge(all_team_players.rename(columns={"player_id": "scorer_id"}),
                    on=["team_id", "scorer_id"], how="inner")
    )

    # ── Score before goal → situation ─────────────────
    goals["home_before"] = np.where(goals["is_home"], goals["home_score"] - 1, goals["home_score"])
    goals["away_before"] = np.where(goals["is_home"], goals["away_score"],     goals["away_score"] - 1)
    goals["team_before"] = np.where(goals["is_home"], goals["home_before"], goals["away_before"])
    goals["opp_before"]  = np.where(goals["is_home"], goals["away_before"], goals["home_before"])
    goals["situation_before"] = np.select(
        [goals["team_before"] < goals["opp_before"],
         goals["team_before"] == goals["opp_before"]],
        ["Losing", "Drawing"], default="Winning"
    )

    rc = goals[
        goals["situation_before"].isin(["Losing", "Drawing"]) &
        (goals["minute"] >= 45)
    ].copy().reset_index(drop=True)

    # ── Assist proxy: highest end_x successful pass per match+team ──
    assist_proxy = (
        passes[passes["pass_successful"]]
        .sort_values("end_x", ascending=False)
        .groupby(["match_id", "team_id"]).first()
        .reset_index()[["match_id", "team_id", "player_id"]]
        .rename(columns={"player_id": "assister_id"})
    )
    rc = rc.merge(assist_proxy, on=["match_id", "team_id"], how="left")

    rc["scorer_is_sub"]   = [(mid, int(sid)) in sub_set for mid, sid in zip(rc["match_id"], rc["scorer_id"])]
    rc["assister_is_sub"] = [
        (mid, int(aid)) in sub_set if pd.notna(aid) else False
        for mid, aid in zip(rc["match_id"], rc["assister_id"])
    ]

    return rc.merge(teams[["team_id", "team_name", "league_name"]], on="team_id", how="left")


def build_agg(rc, league, involvement):
    lg = rc[rc["league_name"] == league].copy()
    if involvement == "Scorer only":
        lg["sub_involved"] = lg["scorer_is_sub"]
    elif involvement == "Assister only":
        lg["sub_involved"] = lg["assister_is_sub"]
    else:
        lg["sub_involved"] = lg["scorer_is_sub"] | lg["assister_is_sub"]

    total    = lg.groupby("team_id").size().reset_index(name="total_goals")
    sub_g    = lg[lg["sub_involved"]].groupby("team_id").size().reset_index(name="sub_goals")
    teams_lg = rc[rc["league_name"] == league][["team_id", "team_name"]].drop_duplicates()

    agg = total.merge(sub_g, on="team_id", how="left").merge(teams_lg, on="team_id", how="left")
    agg["sub_goals"]     = agg["sub_goals"].fillna(0).astype(int)
    agg["non_sub_goals"] = agg["total_goals"] - agg["sub_goals"]
    agg["sub_pct"]       = (agg["sub_goals"] / agg["total_goals"] * 100).round(1)
    return agg


def make_chart(agg, league, sort_by):
    base_color = LEAGUE_COLORS[league]
    sort_col   = "sub_pct" if sort_by == "% Sub involvement" else "total_goals"
    agg        = agg.sort_values(sort_col, ascending=False).reset_index(drop=True)
    team_order = agg["team_name"].tolist()

    long = pd.melt(agg, id_vars=["team_name", "total_goals", "sub_pct"],
                   value_vars=["non_sub_goals", "sub_goals"], var_name="type", value_name="goals")
    long["label"] = long["type"].map({"non_sub_goals": "No substitute involved",
                                      "sub_goals":     "Substitute involved"})

    bars = (
        alt.Chart(long)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("team_name:N", sort=team_order, title=None,
                    axis=alt.Axis(labelAngle=-38, labelFontSize=9,
                                  labelLimit=200, labelOverlap=False)),
            y=alt.Y("goals:Q", title="Result-changing goals (2nd half)",
                    axis=alt.Axis(tickMinStep=1, labelFontSize=12.4, titleFontSize=13.8)),
            color=alt.Color("label:N",
                            scale=alt.Scale(domain=["No substitute involved", "Substitute involved"],
                                            range=[base_color, "#333333"]),
                            legend=alt.Legend(title="", orient="bottom", labelFontSize=12.4,
                                              direction="horizontal", titleOrient="left")),
            order=alt.Order("type:N", sort="ascending"),
            tooltip=[alt.Tooltip("team_name:N",   title="Team"),
                     alt.Tooltip("total_goals:Q", title="Total result-changing goals"),
                     alt.Tooltip("goals:Q",       title="Goals (segment)"),
                     alt.Tooltip("sub_pct:Q",     title="% sub involved", format=".1f")],
        )
    )

    pct_df = agg[agg["sub_goals"] > 0].copy()
    pct_df["pct_label"] = pct_df["sub_pct"].apply(lambda x: f"{x:.0f}%")
    text = (
        alt.Chart(pct_df)
        .mark_text(dy=-8, fontSize=11, color="#444444", fontWeight="bold")
        .encode(x=alt.X("team_name:N", sort=team_order),
                y=alt.Y("total_goals:Q"),
                text="pct_label:N")
    )

    return (
        (bars + text)
        .properties(width=938, height=469)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )


# ── App ────────────────────────────────────────────────
temporal_xg, subs, teams, matches, players, passes = load_data()
rc = precompute(temporal_xg, subs, teams, matches, players, passes)

if "selected_league" not in st.session_state:
    st.session_state.selected_league = LEAGUE_ORDER[0]

st.markdown("### Result-Changing Goals by Team — 2nd Half")

with st.expander("ℹ️ Chart information"):
    st.markdown(
        "This chart focuses on **result-changing goals scored in the second half** — "
        "goals scored from a Losing or Drawing situation from minute 45 onwards. "
        "Each bar is split into: the **league-coloured portion** (no substitute involved) "
        "and the **dark portion** (substitute involved). "
        "The **percentage label** shows what share had substitute involvement. "
        "**Substitute involvement** is the substitute being scorer or assister "
        "(assist = most advanced successful pass in that match, a proxy). "
        "Use the **Sub involvement filter** to narrow to scorer-only or assister-only."
    )

chart_col, sel_col = st.columns([6, 1], gap="medium")

with sel_col:
    st.markdown("**League**")
    new_league = st.selectbox("League", LEAGUE_ORDER,
                              index=LEAGUE_ORDER.index(st.session_state.selected_league),
                              label_visibility="collapsed")
    if new_league != st.session_state.selected_league:
        st.session_state.selected_league = new_league
        st.rerun()
    selected_league = st.session_state.selected_league

    st.markdown("**Sort by**")
    sort_by = st.selectbox("Sort by",
                           ["Total result-changing goals", "% Sub involvement"],
                           label_visibility="collapsed")
    st.markdown("**Sub involvement**")
    involvement = st.selectbox("Involvement",
                               ["Scorer or assister", "Scorer only", "Assister only"],
                               label_visibility="collapsed")

with chart_col:
    st.altair_chart(
        make_chart(build_agg(rc, selected_league, involvement), selected_league, sort_by),
        use_container_width=False,
    )
