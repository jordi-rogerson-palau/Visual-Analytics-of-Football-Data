import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box
from PIL import Image
import pandas as pd
import duckdb
import streamlit as st
from _constants import DB_PATH, LEAGUE_ORDER, LEAGUE_COLOURS, HIDE_UI_CSS
from _pitch import draw_pitch_lines, PITCH_W, PITCH_H

# ── Note: recuperations.py uses LEAGUE_COLOURS (British spelling) ──
# If _constants exports LEAGUE_COLORS, alias it:
try:
    LEAGUE_COLOURS
except NameError:
    from _constants import LEAGUE_COLORS as LEAGUE_COLOURS

APOTHEM        = 2.0
FIG_W, FIG_H, DPI = 7, 4.67, 110

TURNOVER_TYPES = [
    "failed_pass", "interception", "failed_dribble",
    "lost_duel", "miscontrol", "dispossessed", "clearance", "block",
]

if "rec_x_range" not in st.session_state:
    st.session_state["rec_x_range"] = (0.0, 120.0)


@st.cache_data
def build_hex_grid():
    pitch_box = shapely_box(0, 0, PITCH_W, PITCH_H)
    r      = APOTHEM * 2.0 / np.sqrt(3.0)
    col_dx = 1.5 * r
    row_dy = 2.0 * APOTHEM

    valid_centres, hex_polys = [], []
    col_start = int(np.floor(0 / col_dx))
    col_end   = int(np.ceil(PITCH_W / col_dx)) + 1

    for col in range(col_start, col_end + 1):
        cx       = col * col_dx
        y_offset = APOTHEM if (col % 2 == 1) else 0.0
        row_start = int(np.floor(-y_offset / row_dy)) - 1
        row_end   = int(np.ceil((PITCH_H - y_offset) / row_dy)) + 1
        for row in range(row_start, row_end + 1):
            cy    = row * row_dy + APOTHEM + y_offset
            angles = np.linspace(0, 2 * np.pi, 7)[:-1]
            poly   = ShapelyPolygon(zip(cx + r * np.cos(angles), cy + r * np.sin(angles)))
            clipped = poly.intersection(pitch_box)
            if clipped.is_empty or clipped.area < 1e-6:
                continue
            valid_centres.append((cx, cy))
            hex_polys.append(clipped)

    centres = np.array(valid_centres)
    hex_cells, poly_paths = [], []
    for clipped in hex_polys:
        coords  = np.array(clipped.exterior.coords)
        cx_c    = coords[:, 0].mean()
        cy_c    = coords[:, 1].mean()
        hex_idx = int(np.argmin((centres[:, 0] - cx_c)**2 + (centres[:, 1] - cy_c)**2))
        # Mirror x for attacking-direction display
        mirrored = coords.copy(); mirrored[:, 0] = PITCH_W - mirrored[:, 0]
        codes = [Path.MOVETO] + [Path.LINETO] * (len(mirrored) - 2) + [Path.CLOSEPOLY]
        hex_cells.append((mirrored, hex_idx, cx_c))
        poly_paths.append((mirrored, codes, hex_idx))

    return centres, hex_cells, poly_paths


@st.cache_data
def load_raw(_valid_centres_bytes):
    centres = np.frombuffer(_valid_centres_bytes).reshape(-1, 2)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    sequences = con.execute("SELECT match_id, team_id, end_x, end_y, end_type FROM sequences").df()
    teams     = con.execute("SELECT team_id, league_name FROM teams").df()
    con.close()

    pt = (sequences[sequences["end_type"].isin(TURNOVER_TYPES)]
          .dropna(subset=["end_x", "end_y"])
          .merge(teams, on="team_id", how="left"))

    # Vectorised hex assignment (chunked to limit memory)
    coords, cx, cy, hex_idxs = pt[["end_x", "end_y"]].to_numpy(), centres[:, 0], centres[:, 1], []
    for i in range(0, len(coords), 5000):
        c    = coords[i:i + 5000]
        hex_idxs.extend(np.argmin((cx[None, :] - c[:, 0:1])**2 + (cy[None, :] - c[:, 1:2])**2, axis=1).tolist())
    pt["hex_idx"] = hex_idxs

    return {
        league: (grp, grp["match_id"].nunique())
        for league, grp in pt.groupby("league_name")
        if league in LEAGUE_ORDER
    }


def compute_counts(league_dfs, x_range):
    min_x = PITCH_W - x_range[1]
    max_x = PITCH_W - x_range[0]
    maps, global_max = {}, 0.0
    for league in LEAGUE_ORDER:
        if league not in league_dfs:
            maps[league] = {}
            continue
        lg, n = league_dfs[league]
        hc = (lg[(lg["end_x"] >= min_x) & (lg["end_x"] <= max_x)]
              .groupby("hex_idx").size().reset_index(name="count"))
        hc["per_game"] = hc["count"] / n if n > 0 else 0.0
        maps[league]   = dict(zip(hc["hex_idx"], hc["per_game"]))
        if not hc.empty:
            global_max = max(global_max, hc["per_game"].max())
    return maps, global_max


def _make_fig():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def _save_buf(fig, facecolor="white", transparent=False):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0,
                facecolor=facecolor, transparent=transparent)
    buf.seek(0)
    return buf


def render_base(league, poly_paths):
    colour  = LEAGUE_COLOURS[league]
    fig, ax = _make_fig()
    ax.set_facecolor("#f8f5f0")
    ax.set_xlim(0, 120); ax.set_ylim(0, 80)
    draw_pitch_lines(ax)
    for ext_coords, codes, _ in poly_paths:
        ax.add_patch(PathPatch(Path(ext_coords, codes),
                               facecolor="#eeeeee", edgecolor="none", alpha=0.3, linewidth=0.0, zorder=1))
    ax.set_title(league, color=colour, fontsize=10, fontweight="bold", pad=5, loc="center")
    buf = _save_buf(fig)
    img = Image.open(buf).copy()
    plt.close(fig)
    return img


def render_intensity(league, poly_paths, count_map, global_max, base_size):
    colour  = LEAGUE_COLOURS[league]
    fig, ax = _make_fig()
    fig.set_facecolor((0, 0, 0, 0)); ax.set_facecolor((0, 0, 0, 0))
    ax.set_xlim(0, 120); ax.set_ylim(0, 80); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(" ", fontsize=10, pad=5)

    for ext_coords, codes, hex_idx in poly_paths:
        val = count_map.get(hex_idx, 0.0)
        if val <= 0:
            continue
        alpha = 0.10 + 0.80 * (val / global_max) if global_max > 0 else 0.10
        ax.add_patch(PathPatch(Path(ext_coords, codes),
                               facecolor=colour, edgecolor="none", alpha=alpha, linewidth=0.0, zorder=1))

    buf = _save_buf(fig, facecolor=(0, 0, 0, 0), transparent=True)
    img = Image.open(buf).copy().convert("RGBA")
    plt.close(fig)
    if img.size != base_size:
        img = img.resize(base_size, Image.LANCZOS)
    return img


def composite(base_img, intensity_img):
    result = base_img.copy().convert("RGBA")
    result.paste(intensity_img, (0, 0), mask=intensity_img)
    buf = io.BytesIO()
    result.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


# ── CSS: remove oval clipping on images ───────────────
st.markdown(
    HIDE_UI_CSS.rstrip("</style>\n") +
    "\n        [data-testid='stImage'] img { border-radius: 0 !important; }\n    </style>",
    unsafe_allow_html=True,
)

st.markdown("### Recuperations Distribution by Pitch Location")

with st.expander("ℹ️ Chart information"):
    st.markdown(
        "Each pitch shows a **hexbin heatmap of ball recuperation locations** for a given league. "
        "Colour intensity is proportional to recuperations per game, all leagues sharing the same scale. "
        "The **x-range slider** filters by pitch location: x > 80 isolates high-press recuperations; "
        "x < 40 focuses on deep defensive recuperations. "
        "The **table** updates with the slider and shows average recuperations per game per league."
    )

with st.spinner("Loading…"):
    valid_centres, hex_cells, poly_paths = build_hex_grid()
    league_dfs                            = load_raw(valid_centres.tobytes())

row1  = st.columns(3)
row2  = st.columns(3)
cells = row1 + row2
placeholders = [cells[i].empty() for i in range(5)]

# Show last cached composites immediately on re-run
for i, league in enumerate(LEAGUE_ORDER):
    last = st.session_state.get(f"rec_last_composite_{league}")
    if last and last in st.session_state:
        placeholders[i].image(st.session_state[last], use_container_width=True)

with cells[5]:
    x_range = st.slider("Recuperation x range", 0.0, 120.0, step=1.0, key="rec_x_range")

x_range = st.session_state["rec_x_range"]
league_count_maps, global_max = compute_counts(league_dfs, x_range)
x_key = f"{x_range[0]:.0f}_{x_range[1]:.0f}"

for i, league in enumerate(LEAGUE_ORDER):
    ck = f"rec_composite_{league}_{x_key}"
    if ck not in st.session_state:
        base_key = f"rec_base_{league}"
        if base_key not in st.session_state:
            st.session_state[base_key] = render_base(league, poly_paths)
        base_img = st.session_state[base_key]
        st.session_state[ck] = composite(
            base_img,
            render_intensity(league, poly_paths, league_count_maps[league], global_max, base_img.size)
        )
    st.session_state[f"rec_last_composite_{league}"] = ck
    placeholders[i].image(st.session_state[ck], use_container_width=True)

with cells[5]:
    summary_df = (
        pd.DataFrame([{"League": lg,
                       "Recuperations / game": f"{sum(league_count_maps[lg].values()):.2f}"}
                      for lg in LEAGUE_ORDER])
        .sort_values("League").reset_index(drop=True)
    )

    def _colour_row(row):
        c = LEAGUE_COLOURS.get(row["League"], "#ffffff")
        return [f"background-color: {c}22; color: {c}; font-weight: bold",
                f"background-color: {c}22; color: #222222"]

    st.dataframe(summary_df.style.apply(_colour_row, axis=1),
                 hide_index=True, use_container_width=True)
