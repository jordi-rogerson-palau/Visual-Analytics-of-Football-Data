"""
shots.py — build shot tables from the `sequences` table already in DuckDB.

Outputs:
  shots         — sequence rows where end_type is a shot outcome
  shots_teams   — aggregated by (team_id, league_name, end_type)
  shots_league  — aggregated by (league_name, end_type)
"""
import duckdb
from _config import DUCKDB_PATH

SHOT_TYPES = ("shot_scored", "shot_blocked", "shot_saved", "shot_missed")

AGG_COLS = """
    COUNT(*)                          AS count,
    AVG(num_passes)                   AS avg_num_passes,
    AVG(num_carries)                  AS avg_num_carries,
    AVG(num_dribbles)                 AS avg_num_dribbles,
    AVG(num_duels)                    AS avg_num_duels,
    AVG(distance_progressed)          AS avg_distance_progressed,
    AVG(duration_seconds)             AS avg_duration_seconds,
    AVG(start_x)                      AS avg_start_x,
    AVG(start_y)                      AS avg_start_y,
    AVG(end_x)                        AS avg_end_x,
    AVG(end_y)                        AS avg_end_y
"""

shot_filter = "end_type IN ('shot_scored','shot_blocked','shot_saved','shot_missed')"

con = duckdb.connect(str(DUCKDB_PATH))

# ── shots: raw shot sequences enriched with league_name ──────────────────
con.execute(f"""
    CREATE OR REPLACE TABLE shots AS
    SELECT s.*, t.league_name
    FROM sequences s
    JOIN teams t USING (team_id)
    WHERE {shot_filter}
""")

# ── shots_teams: per team + end_type ─────────────────────────────────────
con.execute(f"""
    CREATE OR REPLACE TABLE shots_teams AS
    SELECT team_id, league_name, end_type, {AGG_COLS}
    FROM shots
    GROUP BY team_id, league_name, end_type
""")

# ── shots_league: per league + end_type ──────────────────────────────────
con.execute(f"""
    CREATE OR REPLACE TABLE shots_league AS
    SELECT league_name, end_type, {AGG_COLS}
    FROM shots
    GROUP BY league_name, end_type
""")

counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
          for t in ("shots", "shots_teams", "shots_league")}
con.close()

for table, n in counts.items():
    print(f"✅ {table}: {n:,} rows")
