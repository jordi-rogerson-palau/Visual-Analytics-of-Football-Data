import duckdb
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from scipy.stats import gaussian_kde
from _constants import DB_PATH, LEAGUE_ORDER, LEAGUE_COLORS, HIDE_UI_CSS

st.markdown(HIDE_UI_CSS, unsafe_allow_html=True)
st.markdown("### Duels & Fouls Distribution per Game")

with st.expander("ℹ️ Chart information"):
    st.markdown(
        "Both charts use a **ridge plot** (stacked kernel density estimates) to compare "
        "the per-game distribution of physical events across all five leagues. "
        "Each coloured area shows how frequently a given count occurs across all games in that league — "
        "a wide, flat shape means high variability between games, while a tall narrow peak means "
        "most games cluster around a consistent value. "
        "The **black vertical line** inside each distribution marks the median for that league. "
        "Both charts share the same density scale so the height of areas is directly comparable. "
        "Note that **duels and fouls are separate events**: a duel is a physical contest the referee "
        "lets play on, while a foul is called when the referee stops play. "
        "Hovering over the fouls chart also reveals the **card breakdown** for each league."
    )


@st.cache_data
def load_data():
    con   = duckdb.connect(str(DB_PATH), read_only=True)
    duels = con.execute("SELECT * FROM duels").df()
    teams = con.execute("SELECT team_id, league_name FROM teams").df()
    con.close()

    enriched = duels.merge(teams, on="team_id", how="left")

    duels_pg = (
        enriched[enriched["event_type"] == "Duel"]
        .groupby(["match_id", "league_name"]).size().reset_index(name="count")
    )

    fouls_only = enriched[enriched["event_type"] == "Foul Committed"]
    fouls_pg   = fouls_only.groupby(["match_id", "league_name"]).size().reset_index(name="count")

    card_stats = (
        fouls_only.groupby("league_name")
        .agg(total=("yellow_card", "count"), yellow=("yellow_card", "sum"), red=("red_card", "sum"))
        .reset_index()
    )
    card_stats["no_card_pct"] = ((card_stats["total"] - card_stats["yellow"] - card_stats["red"])
                                  / card_stats["total"] * 100).round(1)
    card_stats["yellow_pct"]  = (card_stats["yellow"] / card_stats["total"] * 100).round(1)
    card_stats["red_pct"]     = (card_stats["red"]    / card_stats["total"] * 100).round(1)

    fouls_pg = fouls_pg.merge(
        card_stats[["league_name", "no_card_pct", "yellow_pct", "red_pct"]],
        on="league_name", how="left"
    )
    return duels_pg, fouls_pg


duels_per_game, fouls_per_game = load_data()


def _peak(df):
    """Return the maximum KDE peak across all leagues in *df*."""
    peaks = []
    for values in df.groupby("league_name")["count"].apply(list):
        if len(values) > 1:
            x = np.linspace(min(values), max(values), 200)
            peaks.append(gaussian_kde(values)(x).max())
    return max(peaks) if peaks else 1.0


shared_peak   = max(_peak(duels_per_game), _peak(fouls_per_game))
DENSITY_SCALE = 0.4 / shared_peak
OFFSET_STEP   = 0.45


def _add_offsets(df):
    order  = sorted(df["league_name"].unique())
    offset = {lg: i * OFFSET_STEP for i, lg in enumerate(reversed(order))}
    return df.assign(offset=df["league_name"].map(offset)), order


def _add_stats(df):
    stats = df.groupby("league_name")["count"].agg(median="median", std="std").round(2).reset_index()
    return df.merge(stats, on="league_name", how="left")


def _median_lines(df, order):
    rows = []
    for league in order:
        sub    = df[df["league_name"] == league]
        vals   = sub["count"].values
        if len(vals) < 2:
            continue
        med    = float(np.median(vals))
        base   = float(sub["offset"].iloc[0])
        top    = float(gaussian_kde(vals)(np.array([med]))[0]) * DENSITY_SCALE + base
        rows.append({"league_name": league, "median": med, "offset": base, "top": top})
    return pd.DataFrame(rows)


def ridge_chart(df, title, show_legend=True, show_cards=False):
    df, order = _add_offsets(df)
    df        = _add_stats(df)

    groupby_cols = ["league_name", "offset", "median", "std"]
    tooltip      = [
        alt.Tooltip("league_name:N", title="League"),
        alt.Tooltip("median:Q",      title="Median",  format=".1f"),
        alt.Tooltip("std:Q",         title="Std Dev", format=".1f"),
    ]
    if show_cards:
        groupby_cols += ["no_card_pct", "yellow_pct", "red_pct"]
        tooltip += [
            alt.Tooltip("no_card_pct:Q", title="No Card %",   format=".1f"),
            alt.Tooltip("yellow_pct:Q",  title="🟡 Yellow %", format=".1f"),
            alt.Tooltip("red_pct:Q",     title="🔴 Red %",    format=".1f"),
        ]

    color_scale = alt.Scale(domain=LEAGUE_ORDER, range=[LEAGUE_COLORS[l] for l in LEAGUE_ORDER])

    areas = (
        alt.Chart(df)
        .transform_density(
            density="count",
            groupby=groupby_cols,
            as_=["count", "density"],
            extent=[df["count"].min() - 3, df["count"].max() + 3],
            steps=200,
        )
        .transform_calculate(
            density_scaled=f"datum.density * {DENSITY_SCALE}",
            shifted="datum.density_scaled + datum.offset",
        )
        .mark_area(fillOpacity=0.55, strokeWidth=1.5)
        .encode(
            x=alt.X("count:Q", title="Count per Game", axis=alt.Axis(grid=False)),
            y=alt.Y("shifted:Q", axis=None),
            y2=alt.Y2("offset:Q"),
            color=alt.Color("league_name:N", scale=color_scale,
                            legend=alt.Legend(title="League") if show_legend else None),
            stroke=alt.Stroke("league_name:N", scale=color_scale, legend=None),
            order=alt.Order("offset:Q", sort="ascending"),
            tooltip=tooltip,
        )
    )

    median_lines = (
        alt.Chart(_median_lines(df, order))
        .mark_rule(color="black", strokeWidth=1.2, opacity=0.7)
        .encode(
            x=alt.X("median:Q"),
            y=alt.Y("offset:Q"),
            y2=alt.Y2("top:Q"),
        )
    )

    return (
        alt.layer(areas, median_lines)
        .properties(width=400, height=300,
                    title=alt.TitleParams(title, fontSize=11, fontWeight="bold"))
        .configure_view(stroke=None)
        .configure_axis(grid=False)
    )


col1, col2 = st.columns(2)
with col1:
    st.altair_chart(ridge_chart(duels_per_game, "Duels per Game", show_legend=False))
with col2:
    st.altair_chart(ridge_chart(fouls_per_game, "Fouls Committed per Game", show_cards=True))
