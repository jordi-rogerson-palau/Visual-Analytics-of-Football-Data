"""teams.py — build the `teams` table in DuckDB."""
import json
import pandas as pd
import duckdb
from _config import DATA_DIR, DUCKDB_PATH, TOP5_LEAGUES, TARGET_SEASON

rows = []
for comp_dir in (DATA_DIR / "matches").iterdir():
    for season_file in comp_dir.glob("*.json"):
        with open(season_file, encoding="utf-8") as f:
            matches = json.load(f)
        for m in matches:
            comp   = m.get("competition", {})
            season = m.get("season", {})
            if comp.get("competition_name") not in TOP5_LEAGUES or season.get("season_name") != TARGET_SEASON:
                continue
            league_name = comp["competition_name"]
            league_id   = comp["competition_id"]
            for side in ("home_team", "away_team"):
                rows.append({
                    "team_id":    m[side][f"{side}_id"],
                    "team_name":  m[side][f"{side}_name"],
                    "league_id":  league_id,
                    "league_name": league_name,
                })

teams_df = pd.DataFrame(rows).drop_duplicates()

con = duckdb.connect(str(DUCKDB_PATH))
con.execute("CREATE OR REPLACE TABLE teams AS SELECT * FROM teams_df")
con.close()
print(f"✅ teams: {len(teams_df)} rows")
