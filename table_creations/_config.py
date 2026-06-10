"""
_config.py — shared constants for all preprocessing scripts.

Convention (consistent with the visualisation layer):
    PROJECT_ROOT  → the folder that contains statsbomb_2015_2016.duckdb
    DATA_DIR      → open-data/data  (StatsBomb open-data repository)
    DUCKDB_PATH   → statsbomb_2015_2016.duckdb

Every preprocessing script should start with:
    from _config import DATA_DIR, DUCKDB_PATH, TOP5_LEAGUES, TARGET_SEASON
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "open-data" / "data"
DUCKDB_PATH  = PROJECT_ROOT / "statsbomb_2015_2016.duckdb"

TOP5_LEAGUES  = {"Premier League", "La Liga", "Serie A", "1. Bundesliga", "Ligue 1"}
TARGET_SEASON = "2015/2016"
