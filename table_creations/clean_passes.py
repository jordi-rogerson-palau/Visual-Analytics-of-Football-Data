"""
clean_passes.py — post-process the raw `passes` table:
  1. Drop out-of-bounds coordinates.
  2. Create `league_passes` (passes enriched with league_name).
  3. Overwrite both tables in DuckDB.
"""
import duckdb
from _config import DUCKDB_PATH

con = duckdb.connect(str(DUCKDB_PATH))

passes_df = con.execute("""
    SELECT p.*
    FROM passes p
    WHERE start_x BETWEEN 0 AND 120
      AND start_y BETWEEN 0 AND 80
      AND end_x   BETWEEN 0 AND 120
      AND end_y   BETWEEN 0 AND 80
""").df()

league_passes_df = con.execute("""
    SELECT p.*, t.league_name
    FROM passes p
    JOIN teams  t USING (team_id)
    WHERE p.start_x BETWEEN 0 AND 120
      AND p.start_y BETWEEN 0 AND 80
      AND p.end_x   BETWEEN 0 AND 120
      AND p.end_y   BETWEEN 0 AND 80
""").df()

con.execute("CREATE OR REPLACE TABLE passes        AS SELECT * FROM passes_df")
con.execute("CREATE OR REPLACE TABLE league_passes AS SELECT * FROM league_passes_df")
con.close()

print(f"✅ passes:        {len(passes_df):,} rows")
print(f"✅ league_passes: {len(league_passes_df):,} rows")
