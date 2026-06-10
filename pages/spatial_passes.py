import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import Polygon as ShapelyPolygon
from PIL import Image
import pandas as pd
import duckdb
import streamlit as st
from _constants import DB_PATH, LEAGUE_ORDER, LEAGUE_COLOURS, HIDE_UI_CSS
from _pitch import draw_pitch_lines

# British-spelling alias (spatial_passes historically uses LEAGUE_COLOURS)
try:
    LEAGUE_COLOURS
except NameError:
    from _constants import LEAGUE_COLORS as LEAGUE_COLOURS

FIG_W, FIG_H, DPI = 7, 4.67, 110

for _k, _v in [("top_n", 5), ("min_start_x", 0.0)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


@st.cache_data
def build_hex_grid_from_db():
    con         = duckdb.connect(str(DB_PATH), read_only=True)
    centres_df  = con.execute("SELECT hex_idx, cx, cy FROM hex_centres ORDER BY hex_idx").df()
    polygons_df = con.execute(
        'SELECT hex_idx, vx, vy, "order" FROM hex_polygons ORDER BY hex_idx, "order"'
    ).df()
    con.close()

    centres = centres_df[["cx", "cy"]].to_numpy()
    hex_ids = centres_df["hex_idx"].tolist()

    clipped_polys = [
        ShapelyPolygon(zip(
            polygons_df[polygons_df["hex_idx"] == idx].sort_values("order")["vx"],
            polygons_df[polygons_df["hex_idx"] == idx].sort_values("order")["vy"],
        ))
        for idx in hex_ids
    ]
    hex_to_pos = {hid: pos for pos, hid in enumerate(hex_ids)}
    return centres, clipped_polys, hex_to_pos


@st.cache_data
def precompute_poly_paths(_clipped_polys):
    result = []
    for poly in _clipped_polys:
        ext   = np.array(poly.exterior.coords)
        codes = [Path.MOVETO] + [Path.LINETO] * (len(ext) - 2) + [Path.CLOSEPOLY]
        result.append((ext, codes))
    return result


@st.cache_data
def load_all_transitions(_centres, _hex_to_pos):
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df  = con.execute("SELECT league_name, src, dst, count FROM hex_transitions").df()
    con.close()

    cx_map = {hid: _centres[pos, 0] for hid, pos in _hex_to_pos.items()}
    cy_map = {hid: _centres[pos, 1] for hid, pos in _hex_to_pos.items()}

    result = {}
    for league, grp in df.groupby("league_name"):
        g = grp.drop(columns="league_name").copy()
        g[["src_cx", "src_cy"]] = g[["src", "dst"]].apply(
            lambda r: pd.Series([cx_map.get(r["src"]), cy_map.get(r["src"])]), axis=1
        )
        g[["dst_cx", "dst_cy"]] = g[["src", "dst"]].apply(
            lambda r: pd.Series([cx_map.get(r["dst"]), cy_map.get(r["dst"])]), axis=1
        )
        g = (g.dropna(subset=["src_cx", "src_cy", "dst_cx", "dst_cy"])
              .query("src != dst")
              .sort_values("count", ascending=False)
              .reset_index(drop=True))
        result[league] = g
    return result


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
    ax.set_facecolor("white")
    ax.set_xlim(0, 120); ax.set_ylim(0, 80)
    draw_pitch_lines(ax)
    for ext, codes in poly_paths:
        ax.add_patch(PathPatch(Path(ext, codes),
                               facecolor="#eeeeee", edgecolor="#aaaaaa",
                               alpha=0.6, linewidth=0.9, zorder=1))
    ax.set_title(league, color=colour, fontsize=10, fontweight="bold", pad=5, loc="center")
    img = Image.open(_save_buf(fig)).copy()
    plt.close(fig)
    return img


def render_arrows(league, top, poly_paths, hex_to_pos, base_size):
    colour  = LEAGUE_COLOURS[league]
    fig, ax = _make_fig()
    fig.set_facecolor((0, 0, 0, 0)); ax.set_facecolor((0, 0, 0, 0))
    ax.set_xlim(0, 120); ax.set_ylim(0, 80); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(" ", fontsize=10, pad=5)

    if not top.empty:
        max_count = top["count"].max()
        src_ids   = set(top["src"])
        dst_ids   = set(top["dst"])

        for hex_idx in src_ids | dst_ids:
            pos = hex_to_pos.get(hex_idx)
            if pos is None:
                continue
            ext, codes = poly_paths[pos]
            is_src = hex_idx in src_ids
            ax.add_patch(PathPatch(Path(ext, codes),
                                   facecolor=colour if is_src else "#ffffff",
                                   edgecolor=colour,
                                   alpha=0.35 if is_src else 0.20,
                                   linewidth=1.2, zorder=2))

        for _, row in top.iterrows():
            ratio = int(row["count"]) / max_count
            ax.annotate("",
                        xy=(row["dst_cx"], row["dst_cy"]),
                        xytext=(row["src_cx"], row["src_cy"]),
                        arrowprops=dict(arrowstyle="-|>", color=colour,
                                        lw=0.8 + 2.5 * ratio,
                                        mutation_scale=10 + 4 * ratio),
                        alpha=0.45 + 0.50 * ratio, zorder=5)

    buf = _save_buf(fig, facecolor=(0, 0, 0, 0), transparent=True)
    img = Image.open(buf).copy().convert("RGBA")
    plt.close(fig)
    if img.size != base_size:
        img = img.resize(base_size, Image.LANCZOS)
    return img


def composite(base_img, arrows_img):
    result = base_img.copy().convert("RGBA")
    result.paste(arrows_img, (0, 0), mask=arrows_img)
    buf = io.BytesIO()
    result.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


st.markdown(HIDE_UI_CSS, unsafe_allow_html=True)
st.markdown("### Most Frequent Pass Transitions")

with st.expander("ℹ️ Chart information"):
    st.markdown(
        "Each pitch shows the most frequent **pass transitions** between hexagonal zones for a given league. "
        "**Filled hexagons** are origin zones of the top N transitions; **outlined hexagons** are destinations. "
        "**Arrows** connect origin to destination — thicker and darker means higher transition frequency. "
        "Use **Top N** to control how many transitions are shown, and **Min starting x** to restrict "
        "transitions to a specific area of the pitch (e.g. 60 = opponent's half, ~101 = penalty box)."
    )

with st.spinner("Loading…"):
    centres, clipped_polys, hex_to_pos = build_hex_grid_from_db()
    poly_paths                          = precompute_poly_paths(clipped_polys)
    all_transitions                     = load_all_transitions(centres, hex_to_pos)

row1  = st.columns(3); row2 = st.columns(3)
cells = row1 + row2
placeholders = [cells[i].empty() for i in range(5)]

for i, league in enumerate(LEAGUE_ORDER):
    last = st.session_state.get(f"sp_last_composite_{league}")
    if last and last in st.session_state:
        placeholders[i].image(st.session_state[last], use_container_width=True)

with cells[5]:
    top_n       = st.slider("Top N transitions", 1, 30, key="top_n")
    min_start_x = st.slider("Min starting x", 0.0, 110.0, step=1.0, key="min_start_x")

for i, league in enumerate(LEAGUE_ORDER):
    ck = f"sp_composite_{league}_{top_n}_{int(min_start_x)}"
    if ck not in st.session_state:
        base_key = f"sp_base_{league}"
        if base_key not in st.session_state:
            st.session_state[base_key] = render_base(league, poly_paths)
        base_img = st.session_state[base_key]

        filtered = (
            all_transitions.get(league, pd.DataFrame())
            .pipe(lambda d: d[d["src_cx"] >= min_start_x] if not d.empty else d)
            .sort_values("count", ascending=False).iloc[:top_n]
        )
        st.session_state[ck] = composite(
            base_img, render_arrows(league, filtered, poly_paths, hex_to_pos, base_img.size)
        )

    st.session_state[f"sp_last_composite_{league}"] = ck
    placeholders[i].image(st.session_state[ck], use_container_width=True)
