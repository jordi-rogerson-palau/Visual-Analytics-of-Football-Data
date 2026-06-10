import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box
from PIL import Image as PILImage
import pandas as pd
import duckdb
import altair as alt
import streamlit as st
from _constants import (DB_PATH, LEAGUE_ORDER, LEAGUE_COLORS,
                         POSITION_GROUP_MAP, L_SCALE, W_SCALE)
from _pitch import draw_pitch_lines, PITCH_W, PITCH_H

FIG_W, FIG_H, DPI = 4.2, 2.8, 120
TOP_N = 5

LOLLIPOP_METRICS = [
    ("Goals",             "goals"),
    ("Assists",           "assists"),
    ("Key Passes",        "key_passes"),
    ("Successful Passes", "successful_passes"),
    ("xG",                "xg"),
]

st.markdown(
    """
    <style>
        #MainMenu { visibility: hidden; }
        header    { visibility: hidden; }
        footer    { visibility: hidden; }
        .block-container {
            padding-top: 0rem !important; margin-top: -1rem !important;
            padding-left: 1rem !important; padding-right: 1rem !important;
            max-width: 100% !important;
        }
        [data-testid="column"] { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
        [data-testid="stImage"] img { border-radius: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def ordinal(n):
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd','th','th','th','th','th','th'][n % 10]}"


@st.cache_data
def load_data():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    losses_raw = con.execute(
        "SELECT match_id, team_id, player_id, location_x, location_y FROM ball_losses"
    ).df()
    players = con.execute(
        "SELECT player_id, player_name, team_id, position, "
        "goals, assists, xg, key_passes, successful_passes FROM players"
    ).df()
    teams    = con.execute("SELECT team_id, team_name, league_name FROM teams").df()
    seq_dirs = con.execute(
        "SELECT match_id, team_id, AVG(start_x) AS avg_start_x FROM sequences GROUP BY match_id, team_id"
    ).df()
    con.close()

    losses_raw["player_id"] = losses_raw["player_id"].astype(float)
    players["player_id"]    = players["player_id"].astype(float)
    players["position_group"] = players["position"].map(POSITION_GROUP_MAP)

    # Usage % per player per team
    team_totals   = losses_raw.groupby("team_id").size().reset_index(name="team_total")
    player_losses = (
        losses_raw.groupby(["team_id", "player_id"]).size()
        .reset_index(name="losses").merge(team_totals, on="team_id")
    )
    player_losses["usage_pct"] = (player_losses["losses"] / player_losses["team_total"] * 100).round(2)
    player_losses["player_id"] = player_losses["player_id"].astype(float)
    player_losses = player_losses.merge(
        players[["player_id", "player_name", "position_group"]].drop_duplicates("player_id"),
        on="player_id", how="left"
    )
    top5_by_team = {
        tid: grp.sort_values("usage_pct", ascending=False).head(TOP_N).reset_index(drop=True)
        for tid, grp in player_losses.groupby("team_id")
    }

    # Direction-normalised losses per player
    losses_dir = losses_raw.merge(seq_dirs, on=["match_id", "team_id"], how="left")
    losses_dir["location_x"] = np.where(
        losses_dir["avg_start_x"] > 60, PITCH_W - losses_dir["location_x"], losses_dir["location_x"]
    )
    losses_dir = losses_dir.dropna(subset=["location_x", "location_y"])
    losses_by_player = {pid: grp.copy() for pid, grp in losses_dir.groupby("player_id")}

    return top5_by_team, players, teams, losses_by_player


@st.cache_data
def build_hex_grid():
    pitch_box = shapely_box(0, 0, PITCH_W, PITCH_H)
    APOTHEM   = 2.0
    r         = APOTHEM * 2.0 / np.sqrt(3.0)
    col_dx    = 1.5 * r
    row_dy    = 2.0 * APOTHEM

    valid_centres, hex_polys = [], []
    for col in range(int(np.ceil(PITCH_W / col_dx)) + 2):
        cx       = col * col_dx
        y_offset = APOTHEM if col % 2 else 0.0
        for row in range(int(np.floor(-y_offset / row_dy)) - 1,
                         int(np.ceil((PITCH_H - y_offset) / row_dy)) + 2):
            cy      = row * row_dy + APOTHEM + y_offset
            angles  = np.linspace(0, 2 * np.pi, 7)[:-1]
            poly    = ShapelyPolygon(zip(cx + r * np.cos(angles), cy + r * np.sin(angles)))
            clipped = poly.intersection(pitch_box)
            if clipped.is_empty or clipped.area < 1e-6:
                continue
            valid_centres.append((cx, cy))
            hex_polys.append(clipped)

    centres = np.array(valid_centres)
    cells   = []
    for clipped in hex_polys:
        coords  = np.array(clipped.exterior.coords)
        cx_c, cy_c = coords[:, 0].mean(), coords[:, 1].mean()
        idx = int(np.argmin((centres[:, 0] - cx_c)**2 + (centres[:, 1] - cy_c)**2))
        codes = [Path.MOVETO] + [Path.LINETO] * (len(coords) - 2) + [Path.CLOSEPOLY]
        cells.append((coords, codes, idx))

    CORNER_RADIUS = 6.0
    corner_set    = set()
    for cp in np.array([[120.0, 0.0], [120.0, 80.0]]):
        corner_set.update(np.where(np.hypot(centres[:, 0] - cp[0], centres[:, 1] - cp[1])
                                   <= CORNER_RADIUS)[0].tolist())
    return centres, cells, corner_set


def _assign_hex(coords, centres):
    cx, cy, out = centres[:, 0], centres[:, 1], []
    for i in range(0, len(coords), 5000):
        c = coords[i:i + 5000]
        out.extend(np.argmin((cx[None, :] - c[:, 0:1])**2 + (cy[None, :] - c[:, 1:2])**2, axis=1).tolist())
    return out


@st.cache_data
def render_base_pitch():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_facecolor("white")
    ax.set_xlim(0, 120); ax.set_ylim(80, 0)
    draw_pitch_lines(ax, lines_color="#bcbcbc", lw=1.5, alpha=0.9)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig); buf.seek(0)
    return buf.getvalue()


def render_hex_overlay(player_id, losses_by_player, centres, hex_cells, corner_set, color):
    df = losses_by_player.get(player_id, pd.DataFrame())
    if df.empty:
        return None
    df = df[df["location_x"] >= 16.5 * L_SCALE].copy()
    if df.empty:
        return None

    df["hex_idx"] = _assign_hex(df[["location_x", "location_y"]].to_numpy(), centres)
    df = df[~df["hex_idx"].isin(corner_set)]

    hc = df.groupby("hex_idx").size().reset_index(name="count")
    lo, hi = hc["count"].min(), hc["count"].max()
    hc["norm"] = 1.0 if hi == lo else (hc["count"] - lo) / (hi - lo)
    count_map = dict(zip(hc["hex_idx"], hc["norm"]))

    cmap  = mcolors.LinearSegmentedColormap.from_list("hm", ["#ffffff", color], N=256)
    GAMMA = 0.5

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_alpha(0); ax.set_facecolor((0, 0, 0, 0))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 120); ax.set_ylim(80, 0); ax.set_aspect("equal"); ax.axis("off")

    for ext, codes, hex_idx in hex_cells:
        if hex_idx in corner_set:
            continue
        val = count_map.get(hex_idx, 0.0)
        if val <= 0:
            continue
        val_s = val ** GAMMA
        ax.add_patch(PathPatch(Path(ext, codes),
                               facecolor=cmap(val_s), edgecolor="none",
                               alpha=0.15 + 0.80 * val_s, linewidth=0.0, zorder=1))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig); buf.seek(0)
    return buf.getvalue()


def composite_pitch(base_bytes, overlay_bytes):
    base = PILImage.open(io.BytesIO(base_bytes)).convert("RGBA")
    if overlay_bytes is None:
        out = io.BytesIO(); base.convert("RGB").save(out, format="png"); out.seek(0); return out.getvalue()
    overlay = PILImage.open(io.BytesIO(overlay_bytes)).convert("RGBA")
    if overlay.size != base.size:
        overlay = overlay.resize(base.size, PILImage.LANCZOS)
    base.paste(overlay, (0, 0), mask=overlay)
    out = io.BytesIO(); base.convert("RGB").save(out, format="png"); out.seek(0)
    return out.getvalue()


def make_metric_chart(player_row, team_players, color):
    rows = []
    for label, col in LOLLIPOP_METRICS:
        team_total = team_players[col].sum()
        player_val = float(player_row[col]) if pd.notna(player_row[col]) else 0.0
        pct        = (player_val / team_total * 100) if team_total > 0 else 0.0
        median_pct = float((team_players[col].fillna(0) / team_total * 100).median()) if team_total > 0 else 0.0
        rows.append({"metric": label, "pct": round(pct, 1), "median_pct": round(median_pct, 1),
                     "player_val": player_val, "team_total": team_total})

    df = pd.DataFrame(rows)
    metric_order = [r["metric"] for r in rows][::-1]

    bars = (alt.Chart(df).mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
            .encode(y=alt.Y("metric:N", sort=metric_order, title=None,
                            axis=alt.Axis(labelFontSize=10, labelLimit=160)),
                    x=alt.X("pct:Q", title="% of team total",
                            scale=alt.Scale(domain=[0, 40]),
                            axis=alt.Axis(labelFontSize=9, titleFontSize=10, tickMinStep=1)),
                    color=alt.value(color),
                    tooltip=[alt.Tooltip("metric:N",     title="Metric"),
                             alt.Tooltip("player_val:Q", title="Player value", format=".1f"),
                             alt.Tooltip("team_total:Q", title="Team total",   format=".1f"),
                             alt.Tooltip("pct:Q",        title="% of team",    format=".1f")]))

    median_tick = (alt.Chart(df).mark_tick(color="#888888", thickness=1.5, bandSize=18)
                   .encode(y=alt.Y("metric:N", sort=metric_order), x="median_pct:Q",
                           tooltip=[alt.Tooltip("median_pct:Q", title="Squad median %", format=".1f")]))

    text = (alt.Chart(df).mark_text(align="left", dx=4, fontSize=9.5, color="#333333")
            .encode(y=alt.Y("metric:N", sort=metric_order), x="pct:Q", text=alt.Text("pct:Q", format=".1f")))

    return (
        (bars + median_tick + text)
        .properties(width=220, height=280,
                    title=alt.TitleParams(text="% of Team Total per Metric", fontSize=11,
                                          fontWeight="bold", color="#333333",
                                          anchor="start", offset=5))
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )


def info_card(label, value, color):
    return f"""
    <div style="background:#f8f8f8; border-left:4px solid {color}; border-radius:6px;
                padding:10px 14px; margin-bottom:10px;">
        <div style="font-size:11px; color:#888888; font-weight:600;
                    text-transform:uppercase; letter-spacing:0.05em;">{label}</div>
        <div style="font-size:20px; font-weight:700; color:#222222; margin-top:3px;">{value}</div>
    </div>"""


# ── App ────────────────────────────────────────────────
centres, hex_cells, corner_hex_set                   = build_hex_grid()
top5_by_team, players_df, teams_df, losses_by_player = load_data()

if "ur_league" not in st.session_state:
    st.session_state.ur_league = "1. Bundesliga"
if "ur_team" not in st.session_state:
    st.session_state.ur_team = (teams_df[teams_df["league_name"] == "1. Bundesliga"]
                                  .sort_values("team_name")["team_name"].iloc[0])
if "ur2_player_label" not in st.session_state:
    st.session_state.ur2_player_label = None


def _on_league_change():
    new = st.session_state._ur2_league_widget
    st.session_state.ur_league = new
    st.session_state.ur_team   = (teams_df[teams_df["league_name"] == new]
                                   .sort_values("team_name")["team_name"].iloc[0])
    st.session_state.ur2_player_label = None

def _on_team_change():
    st.session_state.ur_team          = st.session_state._ur2_team_widget
    st.session_state.ur2_player_label = None

def _on_player_change():
    st.session_state.ur2_player_label = st.session_state._ur2_player_widget


selected_league = st.session_state.ur_league
selected_team   = st.session_state.ur_team
color           = LEAGUE_COLORS[selected_league]
team_row        = teams_df[teams_df["team_name"] == selected_team]
team_id         = int(team_row["team_id"].iloc[0]) if not team_row.empty else None

st.markdown("### Top Usage Players Study")

with st.expander("ℹ️ Chart information"):
    st.markdown(
        "**Usage Rate** measures the percentage of a team's possessions that end with a given player. "
        "The **pitch heatmap** shows the spatial distribution of ball losses for the selected player, "
        "direction-normalised so the team always attacks left to right. "
        "The **barchart** shows the player's share of the team's total for five metrics. "
        "The **grey tick mark** on each bar represents the squad median percentage."
    )

left_col, pitch_col, bar_col = st.columns([2, 4, 3], gap="small")
base_pitch_bytes = render_base_pitch()

with left_col:
    st.markdown("**Select team**")
    st.selectbox("League", LEAGUE_ORDER, index=LEAGUE_ORDER.index(selected_league),
                 key="_ur2_league_widget", on_change=_on_league_change,
                 label_visibility="collapsed")

    league_teams = (teams_df[teams_df["league_name"] == selected_league]
                    .sort_values("team_name")["team_name"].tolist())
    st.selectbox("Team", league_teams,
                 index=league_teams.index(selected_team) if selected_team in league_teams else 0,
                 key="_ur2_team_widget", on_change=_on_team_change, label_visibility="collapsed")

    top5 = top5_by_team.get(team_id, pd.DataFrame()) if team_id else pd.DataFrame()
    player_id, player_row = None, None

    if not top5.empty:
        player_labels = [f"{i+1}. {r['player_name']} ({r['usage_pct']:.1f}%)"
                         for i, r in top5.iterrows()]
        if st.session_state.ur2_player_label not in player_labels:
            st.session_state.ur2_player_label = player_labels[0]

        st.markdown("**Select player**")
        st.selectbox("Player", player_labels,
                     index=player_labels.index(st.session_state.ur2_player_label),
                     key="_ur2_player_widget", on_change=_on_player_change,
                     label_visibility="collapsed")

        sel_idx    = player_labels.index(st.session_state._ur2_player_widget)
        player_row = top5.iloc[sel_idx]
        player_id  = float(player_row["player_id"])
        pos_group  = player_row["position_group"] if pd.notna(player_row["position_group"]) else "N/A"

        st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)
        st.markdown(info_card("Position", pos_group, color), unsafe_allow_html=True)
    else:
        st.info("No usage data for this team.")

if player_id is not None:
    hex_key   = f"ur2_hex_{team_id}_{player_id}"
    pitch_key = f"ur2_pitch_final_{team_id}_{player_id}"
    if hex_key not in st.session_state:
        st.session_state[hex_key] = render_hex_overlay(
            player_id, losses_by_player, centres, hex_cells, corner_hex_set, color)
    if pitch_key not in st.session_state:
        st.session_state[pitch_key] = composite_pitch(base_pitch_bytes, st.session_state[hex_key])
    st.session_state["ur2_current_pitch_key"] = pitch_key

with pitch_col:
    key = st.session_state.get("ur2_current_pitch_key")
    st.image(st.session_state[key] if key and key in st.session_state else base_pitch_bytes,
             use_container_width=True)

if player_id is not None and player_row is not None:
    full_player = players_df[players_df["player_id"] == player_id]
    if not full_player.empty:
        bar_key = f"ur2_bar_{team_id}_{player_id}"
        if bar_key not in st.session_state:
            st.session_state[bar_key] = make_metric_chart(
                full_player.iloc[0], players_df[players_df["team_id"] == team_id], color)
        st.session_state["ur2_current_bar_key"] = bar_key

with bar_col:
    key = st.session_state.get("ur2_current_bar_key")
    if key and key in st.session_state:
        st.altair_chart(st.session_state[key], use_container_width=True)
