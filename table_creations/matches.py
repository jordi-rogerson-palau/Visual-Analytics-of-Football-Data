"""matches.py — build the `matches` table in DuckDB."""
import json
import pandas as pd
import duckdb
from _config import DATA_DIR, DUCKDB_PATH

records = []
for season_file in (DATA_DIR / "matches").glob("**/*.json"):
    with open(season_file, encoding="utf-8") as f:
        for m in json.load(f):
            records.append({
                "match_id":     m["match_id"],
                "home_team_id": m["home_team"]["home_team_id"],
                "away_team_id": m["away_team"]["away_team_id"],
            })

matches_df = pd.DataFrame(records)

con = duckdb.connect(str(DUCKDB_PATH))
con.execute("CREATE OR REPLACE TABLE matches AS SELECT * FROM matches_df")
con.close()
print(f"✅ matches: {len(matches_df)} rows")
