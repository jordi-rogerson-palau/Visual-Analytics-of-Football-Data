import io
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
from pathlib import Path
from _constants import DB_PATH, LEAGUE_ORDER, LEAGUE_COLORS, HIDE_UI_CSS, L_SCALE, W_SCALE
from _pitch import draw_pitch_lines

FIELD_LENGTH = 120
FIELD_WIDTH  = 80
HALFWAY_X    = 60
BIN_LABELS   = ["0–10", "10–20", "20–30", "30–40", "40–50", "50–60"]
HALF1_ALPHA  = 0.88
HALF2_ALPHA  = 0.45


@st.cache_data
def load_data():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    bin_agg = con.execute(
        "SELECT league_name, half, bin_label, passes_per_game, avg_length FROM pass_bin_agg"
    ).df()
    metrics = con.execute(
        "SELECT league_name, total_passes_per_game, completion_rate, "
        "prog_passes_per_game, prog_completion FROM pass_metrics"
    ).df()
    con.close()

    global_max = float(bin_agg["passes_per_game"].max())

    # Passes-per-game-weighted average length per league × half
    wav = (
        bin_agg.assign(w_len=bin_agg["passes_per_game"] * bin_agg["avg_length"])
        .groupby(["league_name", "half"])
        .apply(lambda g: g["w_len"].sum() / g["passes_per_game"].sum()
               if g["passes_per_game"].sum() > 0 else 0)
        .reset_index(name="wavg_length")
    )
    avg_dict = {}
    for _, row in wav.iterrows():
        avg_dict.setdefault(row["league_name"], {})[row["half"]] = row["wavg_length"]

    return bin_agg, metrics, avg_dict, global_max


def _nice_tick_step(raw_step):
    """Return the nearest 'nice' tick step ≥ raw_step."""
    magnitude  = 10 ** np.floor(np.log10(raw_step)) if raw_step > 0 else 1
    nice_steps = [1, 2, 5, 10, 20, 25, 50, 100]
    return next((s * magnitude for s in nice_steps if s * magnitude >= raw_step),
                magnitude * nice_steps[-1])


def _tick_values(max_val, n_ticks=5):
    """Return a list of tick values from 0 up to slightly above max_val."""
    step   = _nice_tick_step(max_val / n_ticks)
    ticks  = []
    tv     = 0.0
    while tv <= max_val * 1.05:
        ticks.append(tv)
        tv = round(tv + step, 10)
    return ticks


def _draw_yaxis(ax, tick_vals, ph, max_bar_h):
    """Draw a manual left y-axis with ticks and grid lines."""
    TICK_X, TICK_LEN, LABEL_X = 0, 1.5, -2.5
    ax.plot([TICK_X, TICK_X], [0, max_bar_h], "-", color="#888888", lw=0.8, alpha=0.6, zorder=3)
    for tv in tick_vals:
        ty    = ph(tv)
        ax.plot([TICK_X - TICK_LEN, TICK_X], [ty, ty], "-", color="#888888", lw=0.8, alpha=0.7, zorder=3)
        ax.plot([0, FIELD_LENGTH],            [ty, ty], "-", color="#cccccc", lw=0.4, alpha=0.5, zorder=1)
        label = str(int(tv)) if tv == int(tv) else f"{tv:.1f}"
        ax.text(LABEL_X, ty, label, ha="right", va="center", fontsize=7.5, color="#555555", zorder=5)


def make_league_figure_bytes(league_name, bin_agg, avg_dict, global_max):
    color  = LEAGUE_COLORS[league_name]
    lg_agg = bin_agg[bin_agg["league_name"] == league_name]
    n_bins = len(BIN_LABELS)

    def get_counts(half_name):
        sub = lg_agg[lg_agg["half"] == half_name].set_index("bin_label")["passes_per_game"]
        return pd.Series([sub.get(lbl, 0.0) for lbl in BIN_LABELS], index=BIN_LABELS)

    c_own   = get_counts("Own Half")
    c_opp   = get_counts("Opposition Half")
    avgs    = avg_dict.get(league_name, {})
    avg_own = avgs.get("Own Half")
    avg_opp = avgs.get("Opposition Half")

    fig, ax = plt.subplots(figsize=(6.6, 4.4), facecolor="white")
    ax.set_facecolor("#f4f4ec")
    ax.set_xlim(-12, FIELD_LENGTH + 2)
    ax.set_ylim(-7, FIELD_WIDTH + 8)
    ax.set_aspect("equal")
    draw_pitch_lines(ax, lines_color="#b8b8b8", lw=2.0, alpha=0.80)

    MARGIN   = 2
    Z1_LEFT  = MARGIN
    Z1_RIGHT = HALFWAY_X - MARGIN
    Z2_LEFT  = HALFWAY_X + MARGIN
    Z1_W     = Z1_RIGHT - Z1_LEFT
    Z2_W     = (FIELD_LENGTH - MARGIN) - Z2_LEFT
    gap1     = Z1_W / n_bins
    gap2     = Z2_W / n_bins
    bar_w1   = gap1 * 0.76
    bar_w2   = gap2 * 0.76
    MAX_BAR_H = FIELD_WIDTH * 0.72

    def ph(count):
        return (count / global_max) * MAX_BAR_H if global_max > 0 else 0

    for i, lbl in enumerate(BIN_LABELS):
        for bx, w, count, alpha in [
            (Z1_LEFT + i * gap1 + (gap1 - bar_w1) / 2, bar_w1, c_own[lbl],  HALF1_ALPHA),
            (Z2_LEFT + i * gap2 + (gap2 - bar_w2) / 2, bar_w2, c_opp[lbl],  HALF2_ALPHA),
        ]:
            ax.add_patch(patches.Rectangle(
                (bx, 0), w, ph(count),
                facecolor=color, alpha=alpha, edgecolor="white", linewidth=0.4, zorder=4
            ))

    TEXT_PAD = 1.2
    for cx, cw, c_vals, avg in [
        (Z1_LEFT, Z1_W, c_own, avg_own),
        (Z2_LEFT, Z2_W, c_opp, avg_opp),
    ]:
        if avg is not None and c_vals.max() > 0:
            ax.text(cx + cw / 2, ph(c_vals.max()) + TEXT_PAD, f"avg {avg:.1f} m",
                    ha="center", va="bottom", fontsize=7.0, color=color, fontweight="bold", zorder=7)

    TOP_Y = FIELD_WIDTH + 1.0
    for cx, cw, label in [(Z1_LEFT, Z1_W, "Own Half"), (Z2_LEFT, Z2_W, "Opposition Half")]:
        ax.text(cx + cw / 2, TOP_Y, label, ha="center", va="bottom",
                fontsize=8.0, color="#222222", fontweight="bold", zorder=5)

    LABEL_Y = -1.8
    for i, lbl in enumerate(BIN_LABELS):
        for zone_left, gap in [(Z1_LEFT, gap1), (Z2_LEFT, gap2)]:
            ax.text(zone_left + i * gap + gap / 2, LABEL_Y, lbl,
                    ha="center", va="top", fontsize=7.5, color="#555555", rotation=38, zorder=5)

    tick_vals = _tick_values(global_max)
    _draw_yaxis(ax, tick_vals, ph, MAX_BAR_H)
    ax.text(-9, MAX_BAR_H / 2, "Successful passes per game",
            ha="center", va="center", fontsize=7.5, color="#555555", rotation=90, zorder=5)

    ax.axis("off")
    ax.set_title(league_name, fontsize=11.0, fontweight="bold", pad=4, color=color)
    fig.tight_layout(pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def make_metrics_chart_bytes(metrics, global_max):
    """Comparison bar chart (total vs progressive passes) matching pitch panel geometry."""
    FIELD_WIDTH = 80
    MAX_BAR_H   = FIELD_WIDTH * 0.72
    Y_BOT, Y_TOP = -7, FIELD_WIDTH + 8

    leagues  = LEAGUE_ORDER
    n        = len(leagues)
    totals   = {r["league_name"]: r["total_passes_per_game"] for _, r in metrics.iterrows()}
    tot_comp = {r["league_name"]: r["completion_rate"]       for _, r in metrics.iterrows()}
    progs    = {r["league_name"]: r["prog_passes_per_game"]  for _, r in metrics.iterrows()}
    prg_comp = {r["league_name"]: r["prog_completion"]       for _, r in metrics.iterrows()}

    all_vals  = [totals[lg] for lg in leagues] + [progs[lg] for lg in leagues]
    local_max = max(all_vals) if all_vals else 1.0

    def to_data(v):
        return (v / local_max) * MAX_BAR_H

    group_w  = FIELD_LENGTH / n
    bar_w    = group_w * 0.30
    gap      = group_w * 0.06
    x_starts = [i * group_w + group_w / 2 for i in range(n)]

    fig, ax = plt.subplots(figsize=(6.6, 4.4), facecolor="white")

    tick_vals = _tick_values(local_max)
    for tv in tick_vals:
        ax.plot([0, FIELD_LENGTH], [to_data(tv), to_data(tv)],
                "-", color="#cccccc", lw=0.4, alpha=0.5, zorder=1)

    for i, lg in enumerate(leagues):
        cx    = x_starts[i]
        color = LEAGUE_COLORS[lg]
        tot   = totals[lg]
        crate = tot_comp[lg] / 100.0
        h_comp   = to_data(tot * crate)
        h_incomp = to_data(tot * (1 - crate))
        bx_tot   = cx - gap / 2 - bar_w

        ax.add_patch(patches.Rectangle((bx_tot, 0), bar_w, h_comp + h_incomp,
                                       facecolor=color, alpha=0.28, edgecolor="none", zorder=3))
        ax.add_patch(patches.Rectangle((bx_tot, 0), bar_w, h_comp,
                                       facecolor=color, alpha=HALF1_ALPHA, edgecolor="none", zorder=4))

        prog  = progs[lg]
        prate = prg_comp[lg] / 100.0
        bx_prog = cx + gap / 2
        ax.add_patch(patches.Rectangle((bx_prog, 0), bar_w, to_data(prog),
                                       facecolor=color, alpha=0.28, edgecolor="none", zorder=3))
        ax.add_patch(patches.Rectangle((bx_prog, 0), bar_w, to_data(prog * prate),
                                       facecolor=color, alpha=HALF1_ALPHA, edgecolor="none", zorder=4))

        TEXT_PAD = 1.0
        ax.text(bx_tot  + bar_w / 2, to_data(tot)  + TEXT_PAD, f"{tot_comp[lg]:.1f}%",
                ha="center", va="bottom", fontsize=6.0, color=color, fontweight="bold", zorder=7)
        ax.text(bx_prog + bar_w / 2, to_data(prog) + TEXT_PAD, f"{prg_comp[lg]:.1f}%",
                ha="center", va="bottom", fontsize=6.0, color=color, fontweight="bold", zorder=7)
        ax.text(cx, Y_BOT + 1.0, lg, ha="center", va="top", fontsize=7.0,
                color=color, fontweight="bold", zorder=5)

    _draw_yaxis(ax, tick_vals, to_data, MAX_BAR_H)
    ax.text(-9, MAX_BAR_H / 2, "Avg passes per game",
            ha="center", va="center", fontsize=7.5, color="#555555", rotation=90, zorder=5)

    BOX_H    = 2.8
    legend_y = MAX_BAR_H + (Y_TOP - MAX_BAR_H) * 0.38
    item_w   = FIELD_LENGTH * 0.38
    legend_x0 = (FIELD_LENGTH - 2 * item_w) / 2
    for j, (lbl, alpha) in enumerate([("Total passes", HALF1_ALPHA), ("Progressive passes", HALF2_ALPHA)]):
        bx = legend_x0 + j * item_w
        ax.add_patch(patches.Rectangle((bx, legend_y - BOX_H / 2), 4.5, BOX_H,
                                       facecolor="#555555", alpha=alpha, edgecolor="none", zorder=6))
        ax.text(bx + 6.0, legend_y, lbl, ha="left", va="center", fontsize=6.5,
                color="#444444", zorder=7)

    ax.set_xlim(-12, FIELD_LENGTH + 2)
    ax.set_ylim(Y_BOT, Y_TOP)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Passes per Game · Completion Rate",
                 fontsize=11.0, fontweight="bold", pad=4, color="#333333")
    fig.tight_layout(pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def main():
    st.markdown(HIDE_UI_CSS, unsafe_allow_html=True)
    st.markdown("### Pass Length Distribution by Pitch Half")

    with st.expander("ℹ️ Chart information"):
        st.markdown(
            "Each pitch shows the **distribution of successful passes per game** binned by distance "
            "(0–10m … 50–60m), split between a team's **own half** (left, full opacity) "
            "and the **opposition half** (right, faded). "
            "Bar heights are scaled consistently across all leagues for direct comparison. "
            "The **avg Xm** label marks the passes-per-game-weighted mean pass length. "
            "The bottom-right panel shows total and progressive passes per game per league, "
            "with opacity reflecting completion rate."
        )

    if not Path(DB_PATH).exists():
        st.error(f"Database not found at `{DB_PATH}`.")
        st.stop()

    bin_agg, metrics, avg_dict, global_max = load_data()

    row1  = st.columns(3, gap="small")
    row2  = st.columns(3, gap="small")
    slots = row1 + row2

    for slot, league in zip(slots[:5], LEAGUE_ORDER):
        cache_key = f"bcp_fig_{league}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = make_league_figure_bytes(league, bin_agg, avg_dict, global_max)
        with slot:
            st.image(st.session_state[cache_key], use_container_width=True)

    with slots[5]:
        metrics_key = "bcp_metrics_chart"
        if metrics_key not in st.session_state:
            st.session_state[metrics_key] = make_metrics_chart_bytes(metrics, global_max)
        st.image(st.session_state[metrics_key], use_container_width=True)


main()
