"""
_pitch.py — shared matplotlib pitch drawing utilities.

All pages that render a pitch (barchart_pass, shots_league, spatial_passes,
recuperations, usage_rate, usage_metrics) import draw_pitch_lines from here
instead of duplicating the same 40-line function.

Usage:
    from _pitch import draw_pitch_lines
    fig, ax = plt.subplots(...)
    draw_pitch_lines(ax)
"""
import matplotlib.patches as patches
from _constants import L_SCALE, W_SCALE

# Canonical pitch dimensions (120×80 coordinate system)
PITCH_W = 120.0
PITCH_H =  80.0


def _sp(x, y):
    """Scale a StatsBomb (105×68) coordinate to the 120×80 system."""
    return x * L_SCALE, y * W_SCALE


# All line segments expressed as ((x1,y1),(x2,y2)) pairs.
# Outer boundary and halfway line use the 120×80 coords directly;
# penalty areas and goal areas use _sp() for StatsBomb-origin values.
_PITCH_LINES = [
    # Outer boundary
    ((0, 0),   (0, 80)),    ((120, 0),  (120, 80)),
    ((0, 80),  (120, 80)),  ((0, 0),    (120, 0)),
    # Halfway line
    ((60, 0),  (60, 80)),
    # Left penalty area
    (_sp(0, 13.85),    _sp(16.5, 13.85)),
    (_sp(0, 54.15),    _sp(16.5, 54.15)),
    (_sp(16.5, 13.85), _sp(16.5, 54.15)),
    # Left 6-yard box
    (_sp(0, 24.85),    _sp(5.5, 24.85)),
    (_sp(0, 43.15),    _sp(5.5, 43.15)),
    (_sp(5.5, 24.85),  _sp(5.5, 43.15)),
    # Right penalty area
    (_sp(88.5, 13.85), _sp(105, 13.85)),
    (_sp(88.5, 54.15), _sp(105, 54.15)),
    (_sp(88.5, 13.85), _sp(88.5, 54.15)),
    # Right 6-yard box
    (_sp(99.5, 24.85), _sp(105, 24.85)),
    (_sp(99.5, 43.15), _sp(105, 43.15)),
    (_sp(99.5, 24.85), _sp(99.5, 43.15)),
]

_ARC_SPECS = [
    # (centre_sb_x, centre_sb_y, radius_sb_units, theta1, theta2)
    (94.0, 34, 9,    128, 232),   # right penalty arc
    (11.0, 34, 9,    308, 52),    # left penalty arc
    (52.5, 34, 9.15, 0,   360),   # centre circle
]


def draw_pitch_lines(ax, lines_color="#444444", lw=1.2, alpha=0.8):
    """
    Draw all pitch markings onto *ax* using the 120×80 coordinate system.

    Parameters
    ----------
    ax          : matplotlib Axes (must have set_xlim/ylim applied separately)
    lines_color : colour for all lines and arc edges
    lw          : line width
    alpha       : line opacity
    """
    for (x1, y1), (x2, y2) in _PITCH_LINES:
        ax.plot([x1, x2], [y1, y2], "-", lw=lw, color=lines_color, alpha=alpha, zorder=3)

    for sb_cx, sb_cy, sb_r, t1, t2 in _ARC_SPECS:
        cx = sb_cx * L_SCALE
        cy = sb_cy * W_SCALE
        r  = sb_r  * L_SCALE
        ax.add_patch(patches.Wedge(
            (cx, cy), r, t1, t2,
            fill=False, edgecolor=lines_color, lw=lw, alpha=alpha, width=0.02, zorder=3
        ))

    ax.set_aspect("equal")
    ax.axis("off")
