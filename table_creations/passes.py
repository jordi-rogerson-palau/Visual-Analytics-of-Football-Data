"""passes.py — build the `passes` table in DuckDB."""
import json
import pandas as pd
import duckdb
from _config import DATA_DIR, DUCKDB_PATH, TOP5_LEAGUES, TARGET_SEASON

# ── Collect match ids for target leagues/season ───────────────────────────
target_ids = set()
for season_file in (DATA_DIR / "matches").glob("**/*.json"):
    with open(season_file, encoding="utf-8") as f:
        for m in json.load(f):
            if (m.get("competition", {}).get("competition_name") in TOP5_LEAGUES and
                    m.get("season", {}).get("season_name") == TARGET_SEASON):
                target_ids.add(m["match_id"])

# ── Extract pass events ───────────────────────────────────────────────────
all_passes = []
for match_id in sorted(target_ids):
    events_path = DATA_DIR / "events" / f"{match_id}.json"
    if not events_path.exists():
        print(f"  ⚠ missing events: {match_id}")
        continue
    with open(events_path, encoding="utf-8") as f:
        events = json.load(f)

    rows = []
    for e in events:
        if e.get("type", {}).get("name") != "Pass":
            continue
        p            = e.get("pass", {})
        loc          = e.get("location", [None, None])
        end_loc      = p.get("end_location", [None, None])
        rows.append({
            "match_id":       match_id,
            "team_id":        e.get("team",   {}).get("id"),
            "player_id":      e.get("player", {}).get("id"),
            "pass_successful": p.get("outcome") is None,   # no outcome dict = successful
            "pass_length":    p.get("length"),
            "pass_height":    p.get("height", {}).get("name"),
            "start_x":        loc[0]     if isinstance(loc,     list) else None,
            "start_y":        loc[1]     if isinstance(loc,     list) else None,
            "end_x":          end_loc[0] if isinstance(end_loc, list) else None,
            "end_y":          end_loc[1] if isinstance(end_loc, list) else None,
        })
    if rows:
        all_passes.append(pd.DataFrame(rows))

passes_df = pd.concat(all_passes, ignore_index=True)

con = duckdb.connect(str(DUCKDB_PATH))
con.execute("CREATE OR REPLACE TABLE passes AS SELECT * FROM passes_df")
con.close()
print(f"✅ passes: {len(passes_df):,} rows")
