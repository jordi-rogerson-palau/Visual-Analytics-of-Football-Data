"""
Shared constants used across all pages.
Import with:  from _constants import DB_PATH, LEAGUE_ORDER, LEAGUE_COLORS, HIDE_UI_CSS
"""
from pathlib import Path

# ── Database path ─────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent   # pages/
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH      = PROJECT_ROOT.parent / "statsbomb_2015_2016.duckdb"

# ── League encoding — shared by every visualisation ──
LEAGUE_ORDER = ["1. Bundesliga", "La Liga", "Ligue 1", "Premier League", "Serie A"]
LEAGUE_COLORS = {
    "1. Bundesliga": "#d62728",
    "La Liga":       "#1f77b4",
    "Ligue 1":       "#9467bd",
    "Premier League":"#ff7f0e",
    "Serie A":       "#2ca02c",
}

# ── Streamlit chrome — injected once per page ─────────
HIDE_UI_CSS = """
<style>
    #MainMenu        { visibility: hidden; }
    header           { visibility: hidden; }
    footer           { visibility: hidden; }
    .block-container {
        padding-top:  0rem  !important;
        margin-top:  -1rem  !important;
        padding-left: 1rem  !important;
    }
</style>
"""

# ── Position grouping — shared by subs_pos, usage_rate, usage_metrics ──
POSITION_GROUP_MAP = {
    "Goalkeeper":                "Goalkeeper",
    "Center Back":               "Center Backs",
    "Left Center Back":          "Center Backs",
    "Right Center Back":         "Center Backs",
    "Left Back":                 "Left Backs",
    "Left Wing Back":            "Left Backs",
    "Right Back":                "Right Backs",
    "Right Wing Back":           "Right Backs",
    "Left Defensive Midfield":   "Defensive Midfielders",
    "Center Defensive Midfield": "Defensive Midfielders",
    "Right Defensive Midfield":  "Defensive Midfielders",
    "Left Center Midfield":      "Center Midfielders",
    "Center Midfield":           "Center Midfielders",
    "Right Center Midfield":     "Center Midfielders",
    "Left Midfield":             "Left Wingers",
    "Right Midfield":            "Right Wingers",
    "Left Attacking Midfield":   "Attacking Midfielders",
    "Center Attacking Midfield": "Attacking Midfielders",
    "Right Attacking Midfield":  "Attacking Midfielders",
    "Left Wing":                 "Left Wingers",
    "Right Wing":                "Right Wingers",
    "Left Center Forward":       "Strikers",
    "Right Center Forward":      "Strikers",
    "Center Forward":            "Strikers",
}

# ── Pitch drawing dimensions ──────────────────────────
L_SCALE = 120 / 105   # StatsBomb → 120×80 coordinate system
W_SCALE = 80  / 68