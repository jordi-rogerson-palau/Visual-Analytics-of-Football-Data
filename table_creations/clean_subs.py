"""
clean_subs.py — enrich the `substitutions` table with home/away team names
and player names, then overwrite the table in DuckDB.

Requires substitutions, matches, teams, players to already exist.
"""
import duckdb
from _config import DUCKDB_PATH

con = duckdb.connect(str(DUCKDB_PATH))

enriched = con.execute("""
    SELECT
        s.*,
        m.home_team_id,
        m.away_team_id,
        ht.team_name  AS home_team_name,
        at.team_name  AS away_team_name,
        pi.player_name AS sub_in_name,
        po.player_name AS sub_out_name
    FROM substitutions s
    JOIN matches          m  ON m.match_id  = s.game_id
    JOIN teams            ht ON ht.team_id  = m.home_team_id
    JOIN teams            at ON at.team_id  = m.away_team_id
    LEFT JOIN (SELECT DISTINCT player_id, player_name FROM players) pi
           ON pi.player_id = s.sub_in_id
    LEFT JOIN (SELECT DISTINCT player_id, player_name FROM players) po
           ON po.player_id = s.sub_out_id
""").df()

con.execute("CREATE OR REPLACE TABLE substitutions AS SELECT * FROM enriched")

n = con.execute("SELECT COUNT(*) FROM substitutions").fetchone()[0]
con.close()
print(f"✅ substitutions (enriched): {n} rows")
