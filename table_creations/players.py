"""
players.py — build the `players` table in DuckDB.

One row per (player_id, team_id) with season totals for:
  goals, assists, key_passes, xg, successful_passes, position, player_name.
"""
import json
import numpy as np
import pandas as pd
import duckdb
from _config import DATA_DIR, DUCKDB_PATH, TOP5_LEAGUES, TARGET_SEASON

# ── Collect valid match ids ───────────────────────────────────────────────
valid_ids = []
for season_file in (DATA_DIR / "matches").glob("**/*.json"):
    with open(season_file, encoding="utf-8") as f:
        for m in json.load(f):
            if (m.get("competition", {}).get("competition_name") in TOP5_LEAGUES and
                    m.get("season", {}).get("season_name") == TARGET_SEASON):
                valid_ids.append(m["match_id"])

valid_ids = sorted(set(valid_ids))
print(f"Processing {len(valid_ids)} matches…")

# ── Per-match event parsing ───────────────────────────────────────────────
all_rows = []

for i, match_id in enumerate(valid_ids):
    try:
        with open(DATA_DIR / "events" / f"{match_id}.json", encoding="utf-8") as f:
            raw = json.load(f)

        events    = pd.json_normalize(raw)
        type_col  = events.get("type.name", pd.Series("", index=events.index)).fillna("")

        def row(e, is_goal=0, is_assist=0, is_key_pass=0, is_pass_success=0, xg=0.0):
            return {
                "player_id":       e.get("player.id"),
                "player_name":     e.get("player.name"),
                "team_id":         e.get("team.id"),
                "position":        e.get("position.name"),
                "is_goal":         is_goal,
                "is_assist":       is_assist,
                "is_key_pass":     is_key_pass,
                "xg":              xg,
                "is_pass_success": is_pass_success,
            }

        shots        = events[type_col == "Shot"]
        shot_outcome = shots.get("shot.outcome.name", pd.Series("", index=shots.index)).fillna("")

        # Goals and all-shot xG
        for _, e in shots.iterrows():
            is_goal = int(shot_outcome[e.name] == "Goal")
            all_rows.append(row(e, is_goal=is_goal,
                                xg=e.get("shot.statsbomb_xg") or 0.0))

        # Assists (key pass → goal) and key passes (key pass → non-goal)
        if "shot.key_pass_id" in shots.columns and "id" in events.columns:
            goal_kp_ids    = shots.loc[shot_outcome == "Goal",     "shot.key_pass_id"].dropna().unique()
            non_goal_kp_ids = shots.loc[shot_outcome != "Goal",    "shot.key_pass_id"].dropna().unique()
            for _, e in events[events["id"].isin(goal_kp_ids)].iterrows():
                all_rows.append(row(e, is_assist=1))
            for _, e in events[events["id"].isin(non_goal_kp_ids)].iterrows():
                all_rows.append(row(e, is_key_pass=1))

        # Successful passes
        passes       = events[type_col == "Pass"]
        pass_outcome = passes.get("pass.outcome.name", pd.Series("", index=passes.index)).fillna("")
        for _, e in passes[pass_outcome == ""].iterrows():
            all_rows.append(row(e, is_pass_success=1))

    except Exception as exc:
        print(f"  ERROR match {match_id}: {exc}")

    if (i + 1) % 100 == 0 or (i + 1) == len(valid_ids):
        print(f"  {i + 1}/{len(valid_ids)}")

# ── Aggregate ─────────────────────────────────────────────────────────────
df = pd.DataFrame(all_rows).dropna(subset=["player_id"])

first_name = (df.dropna(subset=["player_name"])
              .groupby("player_id")["player_name"].first())
first_pos  = (df.dropna(subset=["position"])
              .groupby("player_id")["position"].first())

stats = (
    df.groupby(["player_id", "team_id"])
    .agg(goals            =("is_goal",         "sum"),
         assists          =("is_assist",        "sum"),
         key_passes       =("is_key_pass",      "sum"),
         xg               =("xg",               "sum"),
         successful_passes=("is_pass_success",  "sum"))
    .reset_index()
    .join(first_name, on="player_id")
    .join(first_pos,  on="player_id")
    [["player_id", "player_name", "team_id", "position",
      "goals", "assists", "key_passes", "xg", "successful_passes"]]
    .sort_values(["player_id", "team_id"])
    .reset_index(drop=True)
)

# ── Write to DuckDB ───────────────────────────────────────────────────────
conn = duckdb.connect(str(DUCKDB_PATH))
conn.execute("CREATE OR REPLACE TABLE players AS SELECT * FROM stats")
conn.close()

print(f"\n✅ players: {len(stats)} rows  ({stats['player_id'].nunique()} unique players)")
print(f"   goals={stats['goals'].sum()}  assists={stats['assists'].sum()}  "
      f"xG={stats['xg'].sum():.1f}  key_passes={stats['key_passes'].sum()}")
