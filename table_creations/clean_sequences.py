"""
clean_sequences.py — post-process raw sequences and write derived tables.

Reads sequences_premier_league from new_seq.duckdb, applies cleaning,
and writes three tables to statsbomb_2015_2016.duckdb:
  - sequences          (cleaned, with average_speed)
  - sequences_teams    (team-level averages)
  - sequences_league   (league-level averages)
"""
import duckdb
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DUCKDB_PATH  = PROJECT_ROOT / "statsbomb_2015_2016.duckdb"
NEW_SEQ_PATH = PROJECT_ROOT / "new_seq.duckdb"

# ── Load raw sequences ────────────────────────────────────────────────────
src_con  = duckdb.connect(str(NEW_SEQ_PATH), read_only=True)
sequences = src_con.execute("SELECT * FROM sequences_premier_league").df()
src_con.close()

# ── Cleaning ──────────────────────────────────────────────────────────────
sequences = sequences[sequences["duration_seconds"] <= 1000].copy()
sequences["average_speed"] = (sequences["distance_progressed"] / sequences["duration_seconds"]).fillna(0)
sequences.loc[sequences["duration_seconds"] == 0, "average_speed"] = 0

# ── Team-level aggregation ────────────────────────────────────────────────
DROP_COLS = ["sequence_id", "match_id", "possession", "end_type"]
sequences_teams = (
    sequences.drop(columns=DROP_COLS)
    .groupby("team_id", dropna=False)
    .mean()
    .reset_index()
)

# Join team metadata (team_name, league_id, league_name)
dst_con = duckdb.connect(str(DUCKDB_PATH))
teams   = dst_con.execute("SELECT * FROM teams").df()
sequences_teams = sequences_teams.merge(teams, on="team_id")

# ── League-level aggregation ──────────────────────────────────────────────
sequences_league = (
    sequences_teams
    .drop(columns=["team_id", "team_name"])
    .groupby(["league_id", "league_name"], as_index=False)
    .mean(numeric_only=True)
)

# ── Write all three tables ────────────────────────────────────────────────
dst_con.execute("CREATE OR REPLACE TABLE sequences        AS SELECT * FROM sequences")
dst_con.execute("CREATE OR REPLACE TABLE sequences_teams  AS SELECT * FROM sequences_teams")
dst_con.execute("CREATE OR REPLACE TABLE sequences_league AS SELECT * FROM sequences_league")
dst_con.execute("CHECKPOINT")
dst_con.close()

print("✅ sequences, sequences_teams, sequences_league updated in statsbomb_2015_2016.duckdb")
