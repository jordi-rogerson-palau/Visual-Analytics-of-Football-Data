"""match_scores.py — build the `match_scores` table in DuckDB."""
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
            rows.append({
                "match_id":  m["match_id"],
                "league_id": comp["competition_id"],
                "home_score": m["home_score"],
                "away_score": m["away_score"],
            })

scores_df = pd.DataFrame(rows)

con = duckdb.connect(str(DUCKDB_PATH))
con.execute("CREATE OR REPLACE TABLE match_scores AS SELECT * FROM scores_df")
con.close()
print(f"✅ match_scores: {len(scores_df)} rows")
