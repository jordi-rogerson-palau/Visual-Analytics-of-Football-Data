import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box
from PIL import Image
import pandas as pd
import duckdb
import altair as alt
import streamlit as st
from _constants import (DB_PATH, LEAGUE_ORDER, LEAGUE_COLORS,
                         POSITION_GROUP_MAP, L_SCALE, W_SCALE)
from _pitch import draw_pitch_lines, PITCH_W, PITCH_H

PITCH_FACECOLOR = "white"
FIG_W, FIG_H, DPI = 5.78, 3.85, 120

POSITION_ORDER = [
    "Goalkeeper", "Center Backs", "Left Backs", "Right Backs",
    "Defensive Midfielders", "Center Midfielders", "Attacking Midfielders",
    "Left Wingers", "Right Wingers", "Strikers",
]

st.markdown(
    """
    <style>
        #MainMenu { visibility: hidden; }
        header    { visibility: hidden; }
        footer    { visibility: hidden; }
        .block-container {
            padding-top:  0rem !important; margin-top:  -1rem !important;
            padding-left: 1rem !important; padding-right: 2rem !important;
        }
        [data-testid="stImage"] img { border-radius: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    con      = duckdb.connect(str(DB_PATH), read_only=True)
    usage    = con.execute("SELECT match_id, team_id, player_id, location_x, location_y FROM ball_losses").df()
    teams    = con.execute("SELECT team_id, team_name, league_name FROM teams").df()
    players  = con.execute("SELECT player_id, player_name, position FROM players").df()
    seq_dirs = con.execute(
        "SELECT match_id, team_id, AVG(start_x) AS avg_start_x FROM sequences GROUP BY match_id, team_id"
    ).df()
    con.close()
    usage_by_team = {tid: grp.copy() for tid, grp in usage.groupby("team_id")}
    return usage_by_team, teams, players, seq_dirs


@st.cache_data
def build_hex_grid():
    pitch_box = shapely_box(0, 0, PITCH_W, PITCH_H)
    r      = (APOTHEM := 2.0) * 2.0 / np.sqrt(3.0)
    col_dx = 1.5 * r
    row_dy = 2.0 * APOTHEM

    valid_centres, hex_polys = [], []
    for col in range(int(np.ceil(PITCH_W / col_dx)) + 2):
        cx       = col * col_dx
        y_offset = APOTHEM if col % 2 else 0.0
        row_start = int(np.floor(-y_offset / row_dy)) - 1
        row_end   = int(np.ceil((PITCH_H - y_offset) / row_dy)) + 1
        for row in range(row_start, row_end + 1):
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
        coords = np.array(clipped.exterior.coords)
        cx_c, cy_c = coords[:, 0].mean(), coords[:, 1].mean()
        idx    = int(np.argmin((centres[:, 0] - cx_c)**2 + (centres[:, 1] - cy_c)**2))
        ext    = coords.copy()
        codes  = [MplPath.MOVETO] + [MplPath.LINETO] * (len(ext) - 2) + [MplPath.CLOSEPOLY]
        cells.append((ext, codes, idx))

    CORNER_RADIUS = 6.0
    corner_pts    = np.array([[120.0, 0.0], [120.0, 80.0]])
    corner_idxs   = set()
    for cp in corner_pts:
        corner_idxs.update(np.where(np.hypot(centres[:, 0] - cp[0], centres[:, 1] - cp[1])
                                    <= CORNER_RADIUS)[0].tolist())
    return centres, cells, corner_idxs


def _assign_hex(coords, centres):
    cx, cy, out = centres[:, 0], centres[:, 1], []
    for i in range(0, len(coords), 5000):
        c = coords[i:i + 5000]
        out.extend(np.argmin((cx[None, :] - c[:, 0:1])**2 + (cy[None, :] - c[:, 1:2])**2, axis=1).tolist())
    return out


@st.cache_data
def precompute_team_data(_usage_by_team, _teams, _players, _seq_dirs):
    pinfo = (_players[["player_id", "player_name", "position"]].drop_duplicates("player_id").copy())
    pinfo["player_id"]      = pinfo["player_id"].astype(float)
    pinfo["position_group"] = pinfo["position"].map(POSITION_GROUP_MAP)

    team_data = {}
    for _, row in _teams.iterrows():
        tid, tname, lg = row["team_id"], row["team_name"], row["league_name"]
        df = _usage_by_team.get(tid, pd.DataFrame())
        if df.empty:
            continue

        dirs = _seq_dirs[_seq_dirs["team_id"] == tid]
        df   = df.merge(dirs[["match_id", "avg_start_x"]], on="match_id", how="left")
        df["location_x"] = np.where(df["avg_start_x"] > 60,
                                     PITCH_W - df["location_x"], df["location_x"])
        df = df.dropna(subset=["location_x", "location_y"])
        df["player_id"] = df["player_id"].astype(float)
        df = df.merge(pinfo, on="player_id", how="left")

        n_matches = df["match_id"].nunique()
        team_data[(lg, tname)] = {"df": df, "n_matches": n_matches, "team_id": tid}
    return team_data


@st.cache_data
def render_base_pitch(centres_shape, cells_len):
    """Static pitch PNG — cached by grid dimensions (stable after first build)."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_facecolor("white")
    ax.set_xlim(0, 120); ax.set_ylim(80, 0)
    draw_pitch_lines(ax, lines_color="#bcbcbc", lw=1.5, alpha=0.9)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_pitch_heatmap(df, n_matches, centres, hex_cells, corner_idxs, color):
    CUTOFF_X = 16.5 * L_SCALE
    df = df[df["location_x"] >= CUTOFF_X].copy()
    if df.empty:
        return render_base_pitch(centres.shape, len(hex_cells))

    idxs = _assign_hex(df[["location_x", "location_y"]].to_numpy(), centres)
    df["hex_idx"] = idxs
    df = df[~df["hex_idx"].isin(corner_idxs)]

    hc = df.groupby("hex_idx").size().reset_index(name="count")
    per_game = hc["count"] / n_matches if n_matches > 0 else hc["count"]
    hc["norm"] = (per_game - per_game.min()) / (per_game.max() - per_game.min() + 1e-9)
    count_map = dict(zip(hc["hex_idx"], hc["norm"]))

    cmap  = mcolors.LinearSegmentedColormap.from_list("hm", ["#ffffff", color], N=256)
    GAMMA = 0.5

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_facecolor("white")
    ax.set_xlim(0, 120); ax.set_ylim(80, 0)
    draw_pitch_lines(ax, lines_color="#bcbcbc", lw=1.5, alpha=0.9)

    for ext, codes, hex_idx in hex_cells:
        if hex_idx in corner_idxs:
            continue
        val = count_map.get(hex_idx, 0.0)
        if val <= 0:
            continue
        val_s = val ** GAMMA
        ax.add_patch(PathPatch(MplPath(ext, codes),
                               facecolor=cmap(val_s), edgecolor="none",
                               alpha=0.15 + 0.80 * val_s, linewidth=0.0, zorder=1))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def make_position_chart(df, color, team_name):
    df = df[df["position_group"].notna()].copy()
    if len(df) == 0:
        return alt.Chart(pd.DataFrame()).mark_text().encode(text=alt.value("No data"))

    total  = len(df)
    pct_df = (df["position_group"].value_counts()
              .reindex(POSITION_ORDER, fill_value=0)
              .reset_index(name="count").rename(columns={"index": "position_group"}))
    pct_df["pct"]   = (pct_df["count"] / total * 100).round(1)
    pct_df["label"] = pct_df["pct"].apply(lambda x: f"{x:.1f}%" if x > 0 else "")
    pct_df["position_group"] = pd.Categorical(pct_df["position_group"],
                                               categories=POSITION_ORDER[::-1], ordered=True)

    bars = (alt.Chart(pct_df).mark_bar(cornerRadiusTopRight=2, cornerRadiusBottomRight=2)
            .encode(y=alt.Y("position_group:O", sort=None, title=None,
                            axis=alt.Axis(labelFontSize=10, labelLimit=180)),
                    x=alt.X("pct:Q", title="% of ball losses",
                            axis=alt.Axis(labelFontSize=9, titleFontSize=10)),
                    color=alt.value(color),
                    tooltip=[alt.Tooltip("position_group:O", title="Position"),
                             alt.Tooltip("pct:Q",            title="%", format=".1f"),
                             alt.Tooltip("count:Q",          title="Count")]))

    text = (alt.Chart(pct_df[pct_df["pct"] > 0])
            .mark_text(align="left", dx=4, fontSize=9, color="#444444")
            .encode(y=alt.Y("position_group:O", sort=None),
                    x="pct:Q", text="label:N"))

    return (
        (bars + text)
        .properties(width=420, height=320,
                    title=alt.TitleParams(text="Ball Losses by Position", fontSize=11,
                                          fontWeight="bold", color="#333333",
                                          anchor="start", offset=5, limit=400))
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )


# ── App ────────────────────────────────────────────────
centres, hex_cells, corner_idxs     = build_hex_grid()
usage_by_team, teams, players, seqs = load_data()
team_data = precompute_team_data(usage_by_team, teams, players, seqs)

if "ur_league" not in st.session_state:
    st.session_state.ur_league = "1. Bundesliga"
if "ur_team" not in st.session_state:
    st.session_state.ur_team = (teams[teams["league_name"] == "1. Bundesliga"]
                                 .sort_values("team_name")["team_name"].iloc[0])

selected_league = st.session_state.ur_league
selected_team   = st.session_state.ur_team
color           = LEAGUE_COLORS[selected_league]

title_col, _ = st.columns([3, 5], gap="small")
with title_col:
    st.markdown('<h3 style="white-space: nowrap;">Ball Losses by Position & Location - Team Analysis</h3>',
                unsafe_allow_html=True)

with st.expander("ℹ️ Chart information"):
    st.markdown(
        "The **pitch heatmap** shows the spatial distribution of all ball losses for the selected team, "
        "direction-normalised so the team always attacks left to right. "
        "The **barchart** breaks down ball losses by position group. "
        "Use the **league and team selectors** to switch between teams."
    )

pitch_col, right_col = st.columns([4, 3], gap="large")
entry     = team_data.get((selected_league, selected_team))
pitch_key = f"ur_pitch_{selected_league}_{selected_team}"
bar_key   = f"ur_bar_{selected_league}_{selected_team}"

with pitch_col:
    ph = st.empty()
    last_img_key = st.session_state.get("ur_last_pitch_img")
    if pitch_key in st.session_state:
        ph.image(st.session_state[pitch_key], width=713)
    elif last_img_key and last_img_key in st.session_state:
        ph.image(st.session_state[last_img_key], width=713)

    if pitch_key not in st.session_state and entry is not None:
        img = render_pitch_heatmap(entry["df"], entry["n_matches"],
                                   centres, hex_cells, corner_idxs, color)
        st.session_state[pitch_key] = img
        ph.image(img, width=713)
    st.session_state["ur_last_pitch_img"] = pitch_key

with right_col:
    new_league = st.selectbox("League", LEAGUE_ORDER,
                               index=LEAGUE_ORDER.index(selected_league))
    if new_league != selected_league:
        st.session_state.ur_league = new_league
        st.session_state.ur_team   = (teams[teams["league_name"] == new_league]
                                       .sort_values("team_name")["team_name"].iloc[0])
        st.rerun()

    league_teams = (teams[teams["league_name"] == selected_league]
                    .sort_values("team_name")["team_name"].tolist())
    idx      = league_teams.index(selected_team) if selected_team in league_teams else 0
    new_team = st.selectbox("Team", league_teams, index=idx)
    if new_team != selected_team:
        st.session_state.ur_team = new_team
        st.rerun()

    st.markdown("---")
    if entry is not None:
        if bar_key not in st.session_state:
            st.session_state[bar_key] = make_position_chart(entry["df"], color, selected_team)
        st.altair_chart(st.session_state[bar_key], use_container_width=False)
    else:
        st.info("No data available for this team.")
