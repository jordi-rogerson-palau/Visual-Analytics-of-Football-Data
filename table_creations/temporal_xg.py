"""
temporal_xg.py — build the `temporal_xg` table in DuckDB.

One row per shot event with:
  match_id, league_id, minute, second,
  shot_statsbomb_xg, goal, scorer_id, home_score, away_score.

Home/away assignment is resolved by matching the team's final goal tally
against the recorded home_score in match_scores.
"""
import json
import pandas as pd
import duckdb
from _config import DATA_DIR, DUCKDB_PATH, TOP5_LEAGUES, TARGET_SEASON

# ── Load match metadata and scores from DuckDB ───────────────────────────
conn = duckdb.connect(str(DUCKDB_PATH))
match_scores = conn.execute("SELECT * FROM match_scores").df()

meta_rows = []
for season_file in (DATA_DIR / "matches").glob("**/*.json"):
    with open(season_file, encoding="utf-8") as f:
        for m in json.load(f):
            comp   = m.get("competition", {})
            season = m.get("season", {})
            if (comp.get("competition_name") in TOP5_LEAGUES and
                    season.get("season_name") == TARGET_SEASON):
                meta_rows.append({"match_id": m["match_id"],
                                   "league_id": comp["competition_id"]})

meta_df = pd.DataFrame(meta_rows)
valid_ids = sorted(set(meta_df["match_id"]) & set(match_scores["match_id"]))
league_id_map   = dict(zip(meta_df["match_id"], meta_df["league_id"]))
home_score_map  = dict(zip(match_scores["match_id"], match_scores["home_score"]))
print(f"Processing {len(valid_ids)} matches…")


# ── Helper: rolling score tally ───────────────────────────────────────────
def _rolling_scores(shots_df):
    """
    Return shots_df with home_score and away_score columns (score BEFORE the shot).
    Home team is identified by matching final tally to match_scores.home_score.
    """
    shots_df = shots_df.sort_values(["minute", "second"]).copy()
    team_ids = shots_df["team_id"].dropna().unique()
    if len(team_ids) < 2:
        shots_df["home_score"] = 0
        shots_df["away_score"] = 0
        return shots_df

    tally = {tid: 0 for tid in team_ids}
    score_before = []
    for _, row in shots_df.iterrows():
        score_before.append(dict(tally))
        if row["goal"] == 1 and row["team_id"] in tally:
            tally[row["team_id"]] += 1

    # Identify home team: whichever team's final tally matches match_scores.home_score
    match_id  = shots_df["match_id"].iloc[0]
    expected  = home_score_map.get(match_id, -1)
    tid_a, tid_b = team_ids[0], team_ids[1]
    home_tid  = tid_a if tally[tid_a] == expected else tid_b
    away_tid  = tid_b if home_tid == tid_a else tid_a

    shots_df["home_score"] = [s.get(home_tid, 0) for s in score_before]
    shots_df["away_score"] = [s.get(away_tid, 0) for s in score_before]
    return shots_df


# ── Main loop ─────────────────────────────────────────────────────────────
all_rows = []

for i, match_id in enumerate(valid_ids):
    try:
        with open(DATA_DIR / "events" / f"{match_id}.json", encoding="utf-8") as f:
            events = json.load(f)

        events_df = pd.DataFrame(events)
        shots_df  = events_df[events_df["shot"].notna()].copy()
        if shots_df.empty:
            continue

        shots_df["shot_statsbomb_xg"] = shots_df["shot"].apply(
            lambda x: x.get("statsbomb_xg") if isinstance(x, dict) else None)
        shots_df["goal"]      = (shots_df["shot"].apply(
            lambda x: x.get("outcome", {}).get("name") if isinstance(x, dict) else None
        ) == "Goal").astype(int)
        shots_df["scorer_id"] = shots_df["player"].apply(
            lambda x: x.get("id") if isinstance(x, dict) else None)
        shots_df["team_id"]   = shots_df["team"].apply(
            lambda x: x.get("id") if isinstance(x, dict) else None)
        shots_df["second"]    = shots_df.get("second",
            shots_df["timestamp"].apply(
                lambda x: int(x.split(":")[1]) if isinstance(x, str) else 0))
        shots_df["match_id"]  = match_id
        shots_df["league_id"] = league_id_map[match_id]

        shots_df = _rolling_scores(shots_df)
        all_rows.append(shots_df[["match_id", "league_id", "minute", "second",
                                   "shot_statsbomb_xg", "goal", "scorer_id",
                                   "home_score", "away_score"]])

    except Exception as exc:
        print(f"  ERROR {match_id}: {exc}")

    if (i + 1) % 100 == 0 or (i + 1) == len(valid_ids):
        print(f"  {i + 1}/{len(valid_ids)}")

final_df = pd.concat(all_rows, ignore_index=True)
conn.execute("CREATE OR REPLACE TABLE temporal_xg AS SELECT * FROM final_df")
conn.close()
print(f"\n✅ temporal_xg: {len(final_df):,} rows")
