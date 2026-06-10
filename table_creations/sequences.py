"""
sequences.py — Kaggle preprocessing script.

Reads StatsBomb event JSONs, builds one row per possession sequence,
and writes the result to sequences_premier_league in DuckDB.

Runs on Kaggle; paths assume the standard Kaggle input/output layout.
"""
import os
import json
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path

EVENTS_DIR  = Path("/kaggle/input/datasets/saurabhshahane/statsbomb-football-data/data/events")
MATCHES_DIR = Path("/kaggle/input/datasets/saurabhshahane/statsbomb-football-data/data/matches")
DB_PATH     = Path("/kaggle/working/statsbomb_2015_2016.duckdb")

TARGET_COMPETITIONS = {"Premier League", "La Liga", "1. Bundesliga", "Ligue 1", "Serie A"}
TARGET_SEASON       = "2015/2016"
BATCH_SIZE          = 50


# ── Direction helpers ─────────────────────────────────────────────────────

def _get_attack_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame(match_id, possession, attack_right) using carry dx
    as the primary signal (aggregated per half), with per-possession fallback.
    """
    carry_mask = (
        (df["type.name"] == "Carry") &
        df["location"].apply(isinstance, args=(list,)) &
        df["carry.end_location"].apply(isinstance, args=(list,)) &
        (df["team.name"] == df["possession_team.name"])
    )
    carries = df[carry_mask].copy()
    carries["carry_dx"] = (
        carries["carry.end_location"].apply(lambda x: x[0]) -
        carries["location"].apply(lambda x: x[0])
    )

    # Primary: mean carry_dx per (match, period, possession_team)
    half_dir = (
        carries.groupby(["match_id", "period", "possession_team.name"])["carry_dx"]
        .mean().reset_index().rename(columns={"carry_dx": "mean_dx"})
    )
    half_dir["attack_right"] = half_dir["mean_dx"] > 0

    poss_teams = df[["match_id", "period", "possession", "possession_team.name"]].drop_duplicates()
    direction  = poss_teams.merge(
        half_dir[["match_id", "period", "possession_team.name", "attack_right"]],
        on=["match_id", "period", "possession_team.name"], how="left"
    )

    # Fallback: per-possession mean carry_dx
    missing = direction["attack_right"].isna()
    if missing.any():
        poss_dir = (carries.groupby(["match_id", "possession"])["carry_dx"]
                    .mean().reset_index().rename(columns={"carry_dx": "poss_dx"}))
        direction = direction.merge(poss_dir, on=["match_id", "possession"], how="left")
        direction.loc[missing, "attack_right"] = direction.loc[missing, "poss_dx"] > 0
        direction = direction.drop(columns=["poss_dx"])

    direction["attack_right"] = direction["attack_right"].fillna(True)
    return direction[["match_id", "possession", "attack_right"]]


def _normalize_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Flip x/y so every possession attacks right (x → 120)."""
    df = df.merge(_get_attack_direction(df), on=["match_id", "possession"], how="left")
    df["attack_right"] = df["attack_right"].fillna(True)
    flip = ~df["attack_right"]
    for x_col, y_col in [("start_x", "start_y"), ("event_end_x", "event_end_y")]:
        if x_col in df.columns:
            df.loc[flip, x_col] = 120 - df.loc[flip, x_col]
        if y_col in df.columns:
            df.loc[flip, y_col] = 80  - df.loc[flip, y_col]
    return df.drop(columns=["attack_right"])


# ── End-location extraction ───────────────────────────────────────────────

def _extract_end_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Add event_end_x / event_end_y using the most meaningful end coordinate per type."""
    def _list_idx(col, idx):
        if col not in df.columns:
            return pd.Series([None] * len(df), index=df.index)
        return df[col].apply(lambda x: x[idx] if isinstance(x, list) and len(x) > idx else None)

    t = df.get("type.name", pd.Series("", index=df.index))
    # Use type-specific end locations; shots and others fall back to their own location
    end_x = (pd.Series(None, index=df.index, dtype=object)
             .where(~(t == "Pass"),        _list_idx("pass.end_location",       0))
             .where(~(t == "Carry"),       _list_idx("carry.end_location",      0))
             .where(~(t == "Goal Keeper"), _list_idx("goalkeeper.end_location", 0))
             .combine_first(_list_idx("location", 0)))
    end_y = (pd.Series(None, index=df.index, dtype=object)
             .where(~(t == "Pass"),        _list_idx("pass.end_location",       1))
             .where(~(t == "Carry"),       _list_idx("carry.end_location",      1))
             .where(~(t == "Goal Keeper"), _list_idx("goalkeeper.end_location", 1))
             .combine_first(_list_idx("location", 1)))

    df["event_end_x"] = pd.to_numeric(end_x, errors="coerce")
    df["event_end_y"] = pd.to_numeric(end_y, errors="coerce")
    return df


# ── End-type classifier ───────────────────────────────────────────────────

_SHOT_PRIORITY = {"Goal": 0, "Saved": 1, "Saved to Post": 1, "Saved Off Target": 1,
                  "Blocked": 2, "Post": 3, "Off T": 4, "Wayward": 5}


def _classify_end(df: pd.DataFrame) -> pd.Series:
    """Return a Series of end-type labels for the given events."""
    t = df.get("type.name", pd.Series("", index=df.index)).fillna("")

    out_col     = df.get("out",                       pd.Series(False, index=df.index)).fillna(False).astype(bool)
    pass_out    = df.get("pass.outcome.name",         pd.Series("", index=df.index)).fillna("")
    dribble_out = df.get("dribble.outcome.name",      pd.Series("", index=df.index)).fillna("")
    duel_out    = df.get("duel.outcome.name",         pd.Series("", index=df.index)).fillna("")
    shot_out    = df.get("shot.outcome.name",         pd.Series("", index=df.index)).fillna("")
    foul_card   = df.get("bad_behaviour.card.name",   pd.Series("", index=df.index)).fillna("")
    foul_type   = df.get("foul_committed.type.name",  pd.Series("", index=df.index)).fillna("")
    interc_out  = df.get("interception.outcome.name", pd.Series("", index=df.index)).fillna("")
    gk_out      = df.get("goalkeeper.success_out",    pd.Series("", index=df.index)).fillna("")

    # Shot sub-classification (applied after the main np.select)
    shot_mask = t == "Shot"
    shot_type = pd.Series("shot_other", index=df.index)
    shot_type.loc[shot_mask & (shot_out == "Goal")]                                            = "shot_scored"
    shot_type.loc[shot_mask & shot_out.isin(["Saved", "Saved Off Target", "Saved to Post"])]  = "shot_saved"
    shot_type.loc[shot_mask & (shot_out == "Blocked")]                                         = "shot_blocked"
    shot_type.loc[shot_mask & shot_out.isin(["Off T", "Post", "Wayward"])]                     = "shot_missed"

    conditions = [
        shot_mask,
        out_col,
        (t == "Pass")         & (pass_out    != "") & (pass_out    != "Complete"),
        (t == "Dribble")      & (dribble_out != "") & (dribble_out != "Complete"),
        (t == "Duel")         & (duel_out    != "") & (duel_out    != "Won"),
        t == "Dispossessed",
        (t == "Interception") & (interc_out != ""),
        (foul_card != "") | (foul_type != ""),
        t.str.contains("Goalkeeper", na=False) | (gk_out != ""),
        t == "Miscontrol",
        t.str.contains("Clearance", na=False),
        t == "Offside",
        t == "Substitution",
        t == "Foul Won",
        t == "Block",
        t == "Ball Receipt*",
    ]
    choices = [
        None,              # shot rows handled by shot_type below
        "out_of_bounds", "failed_pass", "failed_dribble", "lost_duel",
        "dispossessed", "interception", "foul", "goalkeeper_action",
        "miscontrol", "clearance", "offside", "substitution",
        "foul_won", "block", "natural_transition",
    ]

    end_type = pd.Series(np.select(conditions, choices, default="other"), index=df.index)
    end_type.loc[shot_mask] = shot_type.loc[shot_mask]
    return end_type


def _build_last_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    Shot-aware last-event selection:
    - Possessions WITH a shot → pick highest-priority shot outcome
      (handles saved→rebound→goal sequences correctly).
    - Possessions WITHOUT a shot → true last event.
    """
    COLS = ["match_id", "possession", "possession_team_id",
            "event_end_x", "event_end_y", "end_type"]

    shot_df = events[events["type.name"] == "Shot"].copy()
    shot_df["end_type"]         = _classify_end(shot_df)
    shot_df["_outcome_priority"] = shot_df["shot.outcome.name"].map(_SHOT_PRIORITY).fillna(99)

    shot_last = (
        shot_df
        .sort_values(["match_id", "possession", "_outcome_priority"])
        .drop_duplicates(subset=["match_id", "possession"], keep="first")
        .drop(columns=["_outcome_priority"])
    )[COLS]

    # Possessions without any shot
    shot_keys     = shot_last[["match_id", "possession"]].assign(_has_shot=True)
    no_shot_mask  = events.merge(shot_keys, on=["match_id", "possession"], how="left")["_has_shot"].isna().values
    no_shot_last  = events[no_shot_mask].groupby(
        ["match_id", "possession", "possession_team_id"], sort=False).tail(1).copy()
    no_shot_last["end_type"] = _classify_end(no_shot_last)

    return pd.concat([shot_last, no_shot_last[COLS]], ignore_index=True)


# ── Collect target match ids ──────────────────────────────────────────────

target_ids = set()
comp_counts = {}
for comp_dir in MATCHES_DIR.iterdir():
    for season_file in comp_dir.glob("*.json"):
        with open(season_file, encoding="utf-8") as f:
            for m in json.load(f):
                comp_name   = m.get("competition", {}).get("competition_name", "")
                season_name = m.get("season",      {}).get("season_name",      "")
                if comp_name in TARGET_COMPETITIONS and season_name == TARGET_SEASON:
                    target_ids.add(m["match_id"])
                    comp_counts[comp_name] = comp_counts.get(comp_name, 0) + 1

print(f"Matches for {TARGET_SEASON}:")
for comp, n in sorted(comp_counts.items()):
    print(f"  {comp}: {n}")
print(f"  TOTAL: {len(target_ids)}")

all_files = sorted(
    f for f in EVENTS_DIR.iterdir()
    if f.suffix == ".json" and int(f.stem) in target_ids
)
print(f"Event files to process: {len(all_files)}")


# ── Batch processing ──────────────────────────────────────────────────────

con = duckdb.connect(str(DB_PATH))
con.execute("DROP TABLE IF EXISTS sequences_premier_league")
table_created = False

for batch_start in range(0, len(all_files), BATCH_SIZE):
    batch = all_files[batch_start: batch_start + BATCH_SIZE]
    print(f"  Batch {batch_start + 1}–{batch_start + len(batch)} / {len(all_files)} …")

    raw_batches = []
    for path in batch:
        with open(path, encoding="utf-8") as f:
            df = pd.json_normalize(json.load(f))
        df["match_id"] = int(path.stem)
        raw_batches.append(df)

    events = pd.concat(raw_batches, ignore_index=True)
    events = (events[events["possession"].notna() & events["possession_team.id"].notna()]
              .rename(columns={"possession_team.id": "possession_team_id"})
              .copy())

    events["start_x"] = pd.to_numeric(
        events["location"].apply(lambda x: x[0] if isinstance(x, list) else None), errors="coerce")
    events["start_y"] = pd.to_numeric(
        events["location"].apply(lambda x: x[1] if isinstance(x, list) else None), errors="coerce")
    events = _extract_end_locations(events)
    events = _normalize_direction(events)

    events["absolute_time"] = events["minute"] * 60 + events["second"]
    events = (events
              .sort_values(["match_id", "possession", "absolute_time"])
              .query("possession != 0")
              .dropna(subset=["start_x", "start_y"])
              [lambda d: d["start_x"].between(0, 120) & d["start_y"].between(0, 80)]
              [lambda d: d["event_end_x"].isna() |
                         (d["event_end_x"].between(0, 120) & d["event_end_y"].between(0, 80))])

    last_events = _build_last_events(events)
    type_col    = "type.name" if "type.name" in events.columns else "type"

    sequences = (
        events
        .assign(_type=events[type_col])
        .groupby(["match_id", "possession", "possession_team_id"], sort=False)
        .agg(num_passes   =("_type", lambda x: (x == "Pass").sum()),
             num_carries  =("_type", lambda x: (x == "Carry").sum()),
             num_dribbles =("_type", lambda x: (x == "Dribble").sum()),
             num_duels    =("_type", lambda x: (x == "Duel").sum()),
             start_x      =("start_x",        "first"),
             start_y      =("start_y",        "first"),
             start_time   =("absolute_time",  "first"),
             end_time     =("absolute_time",  "last"))
        .reset_index()
        .merge(last_events, on=["match_id", "possession", "possession_team_id"], how="left")
    )

    sequences["duration_seconds"]    = sequences["end_time"] - sequences["start_time"]
    sequences["distance_progressed"] = sequences["event_end_x"] - sequences["start_x"]
    sequences["sequence_id"] = (sequences["match_id"].astype(str) + "_" +
                                 sequences["possession_team_id"].astype(str) + "_" +
                                 sequences["possession"].astype(str))
    sequences = (sequences
                 .rename(columns={"possession_team_id": "team_id",
                                  "event_end_x": "end_x", "event_end_y": "end_y"})
                 [["sequence_id", "match_id", "team_id", "possession",
                   "num_passes", "num_carries", "num_dribbles", "num_duels",
                   "start_x", "start_y", "end_x", "end_y",
                   "duration_seconds", "distance_progressed", "end_type"]])

    if not table_created:
        con.execute("CREATE TABLE sequences_premier_league AS SELECT * FROM sequences")
        table_created = True
    else:
        con.execute("INSERT INTO sequences_premier_league SELECT * FROM sequences")

con.execute("CHECKPOINT")
con.close()
print(f"✅ Done — {DB_PATH}")
