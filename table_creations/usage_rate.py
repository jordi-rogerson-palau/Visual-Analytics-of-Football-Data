"""
usage_rate.py — build the `ball_losses` table in DuckDB.

One row per ball-loss event with:
  match_id, league_id, team_id, player_id, ball_lost_type, location_x, location_y.
"""
import json
import numpy as np
import pandas as pd
import duckdb
from _config import DATA_DIR, DUCKDB_PATH, TOP5_LEAGUES, TARGET_SEASON

# ── Collect valid match ids ───────────────────────────────────────────────
meta_rows = []
for season_file in (DATA_DIR / "matches").glob("**/*.json"):
    with open(season_file, encoding="utf-8") as f:
        for m in json.load(f):
            comp   = m.get("competition", {})
            season = m.get("season", {})
            if (comp.get("competition_name") in TOP5_LEAGUES and
                    season.get("season_name") == TARGET_SEASON):
                meta_rows.append({"match_id": m["match_id"],
                                   "league_id": comp["competition_id"]})

meta_df       = pd.DataFrame(meta_rows)
valid_ids     = sorted(meta_df["match_id"].unique())
league_id_map = dict(zip(meta_df["match_id"], meta_df["league_id"]))
print(f"Processing {len(valid_ids)} matches…")


# ── Ball-loss classifier ──────────────────────────────────────────────────
def classify_ball_losses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter events to those representing a loss of possession and add
    a `ball_lost_type` column.  Returns only the loss rows.
    """
    t = df.get("type.name", pd.Series("", index=df.index)).fillna("")

    out_col        = df.get("out",                       pd.Series(False, index=df.index)).fillna(False).astype(bool)
    pass_out       = df.get("pass.outcome.name",         pd.Series("", index=df.index)).fillna("")
    dribble_out    = df.get("dribble.outcome.name",      pd.Series("", index=df.index)).fillna("")
    duel_out       = df.get("duel.outcome.name",         pd.Series("", index=df.index)).fillna("")
    shot_out       = df.get("shot.outcome.name",         pd.Series("", index=df.index)).fillna("")
    foul_card      = df.get("bad_behaviour.card.name",   pd.Series("", index=df.index)).fillna("")
    foul_type      = df.get("foul_committed.type.name",  pd.Series("", index=df.index)).fillna("")
    interc_out     = df.get("interception.outcome.name", pd.Series("", index=df.index)).fillna("")
    gk_out         = df.get("goalkeeper.success_out",    pd.Series("", index=df.index)).fillna("")

    conditions = [
        out_col,
        (t == "Pass")         & pass_out.isin(["Incomplete", "Out", "Pass Offside", "Unknown"]),
        (t == "Dribble")      & (dribble_out != "") & (dribble_out != "Complete"),
        (t == "Duel")         & duel_out.isin(["Lost In Play", "Lost Out", "Success Out"]),
        t == "Dispossessed",
        t == "Miscontrol",
        (t == "Interception") & (interc_out != ""),
        t.str.contains("Clearance",  na=False),
        t.str.contains("Goalkeeper", na=False) | (gk_out != ""),
        (t == "Shot") & shot_out.isin(["Saved", "Saved Off Target", "Saved to Post"]),
        (t == "Shot") & (shot_out == "Blocked"),
        (t == "Shot") & shot_out.isin(["Off T", "Post", "Wayward"]),
        t == "Offside",
        (t == "Foul Committed") | (foul_card != "") | (foul_type != ""),
    ]
    labels = [
        "out_of_bounds", "failed_pass", "failed_dribble", "lost_duel",
        "dispossessed", "miscontrol", "interception", "clearance",
        "goalkeeper_action", "shot_saved", "shot_blocked", "shot_missed",
        "offside", "foul",
    ]

    loss_type = pd.Series(np.select(conditions, labels, default=None), index=df.index)
    result    = df[loss_type.notna()].copy()
    result["ball_lost_type"] = loss_type[loss_type.notna()]
    return result


# ── Main loop ─────────────────────────────────────────────────────────────
all_rows = []

for i, match_id in enumerate(valid_ids):
    try:
        with open(DATA_DIR / "events" / f"{match_id}.json", encoding="utf-8") as f:
            raw = json.load(f)

        events = pd.json_normalize(raw)
        events["match_id"]   = match_id
        events["league_id"]  = league_id_map[match_id]
        events["location_x"] = events["location"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) >= 2 else None)
        events["location_y"] = events["location"].apply(
            lambda x: x[1] if isinstance(x, list) and len(x) >= 2 else None)
        events["player_id"]  = events.get("player.id",  None)
        events["team_id"]    = events.get("team.id",    None)

        losses = classify_ball_losses(events)
        if not losses.empty:
            all_rows.append(losses[["match_id", "league_id", "team_id", "player_id",
                                     "ball_lost_type", "location_x", "location_y"]])

    except Exception as exc:
        print(f"  ERROR {match_id}: {exc}")

    if (i + 1) % 100 == 0 or (i + 1) == len(valid_ids):
        print(f"  {i + 1}/{len(valid_ids)}")

ball_losses_df = pd.concat(all_rows, ignore_index=True)

conn = duckdb.connect(str(DUCKDB_PATH))
conn.execute("CREATE OR REPLACE TABLE ball_losses AS SELECT * FROM ball_losses_df")
conn.close()

print(f"\n✅ ball_losses: {len(ball_losses_df):,} rows")
print(ball_losses_df["ball_lost_type"].value_counts().to_string())
