"""
subs.py — build the raw `substitutions` table in DuckDB.

Each row is one substitution event with:
  game_id, league_id, team_id, sub_in_id, sub_out_id,
  position_in, position_out, period, minute, second,
  timestamp, current_result.
"""
import json
import pandas as pd
import duckdb
from _config import DATA_DIR, DUCKDB_PATH, TOP5_LEAGUES, TARGET_SEASON

# ── Collect match metadata ────────────────────────────────────────────────
match_meta = []
for season_file in (DATA_DIR / "matches").glob("**/*.json"):
    with open(season_file, encoding="utf-8") as f:
        for m in json.load(f):
            comp   = m.get("competition", {})
            season = m.get("season", {})
            if (comp.get("competition_name") in TOP5_LEAGUES and
                    season.get("season_name") == TARGET_SEASON):
                match_meta.append({
                    "match_id":  m["match_id"],
                    "league_id": comp["competition_id"],
                })

meta_df         = pd.DataFrame(match_meta)
valid_match_ids = sorted(meta_df["match_id"].unique())
league_id_map   = dict(zip(meta_df["match_id"], meta_df["league_id"]))
print(f"Processing {len(valid_match_ids)} matches…")


# ── Helpers ───────────────────────────────────────────────────────────────
_PERIOD_OFFSET = {1: 0, 2: 45 * 60, 3: 90 * 60, 4: 105 * 60, 5: 120 * 60}


def _build_score_tracker(events_df):
    """Return (score_at_index, home_team_id) where score_at[idx] = (home, away) before that event."""
    kickoff = events_df[events_df["type"].apply(
        lambda x: x.get("name") == "Kick Off" if isinstance(x, dict) else False
    )]
    home_tid = (
        kickoff.iloc[0]["team"].get("id") if not kickoff.empty
        else events_df["team"].apply(lambda x: x.get("id") if isinstance(x, dict) else None).dropna().iloc[0]
    )

    home, away, score_at = 0, 0, {}
    for idx, row in events_df.iterrows():
        score_at[idx] = (home, away)
        shot = row.get("shot", {})
        if (isinstance(row.get("type"), dict) and row["type"].get("name") == "Shot" and
                isinstance(shot, dict) and shot.get("outcome", {}).get("name") == "Goal"):
            tid = row.get("team", {}).get("id") if isinstance(row.get("team"), dict) else None
            if tid == home_tid:
                home += 1
            else:
                away += 1
    return score_at, home_tid


def _position_at(positions: list, event_time_sec: float) -> str | None:
    """Return the active position name from a StatsBomb lineup positions list."""
    def to_sec(t):
        if t is None:
            return float("inf")
        parts = t.split(":")
        if len(parts) == 3:
            h, m, s = parts; return int(h) * 3600 + int(m) * 60 + float(s)
        m, s = parts; return int(m) * 60 + float(s)

    last = None
    for p in positions:
        start, end = to_sec(p.get("from")), to_sec(p.get("to"))
        if start <= event_time_sec < end:
            return p.get("position")
        if start <= event_time_sec:
            last = p.get("position")
    return last


# ── Main loop ─────────────────────────────────────────────────────────────
all_rows = []

for i, match_id in enumerate(valid_match_ids):
    try:
        with open(DATA_DIR / "events"  / f"{match_id}.json", encoding="utf-8") as f:
            events = json.load(f)
        with open(DATA_DIR / "lineups" / f"{match_id}.json", encoding="utf-8") as f:
            lineups = json.load(f)
    except FileNotFoundError as e:
        print(f"  ⚠ {e}"); continue

    events_df = (
        pd.DataFrame(events)
        .assign(**{c: 0 for c in ["period", "minute", "second"] if c not in pd.DataFrame(events).columns})
        .sort_values(["period", "minute", "second"])
        .reset_index(drop=True)
    )

    score_at, _ = _build_score_tracker(events_df)

    # player_id → {positions, team_id}
    player_info = {
        p["player_id"]: {"positions": p.get("positions", []), "team_id": team["team_id"]}
        for team in lineups
        for p in team.get("lineup", [])
    }

    for _, ev in events_df[events_df["type"].apply(
        lambda x: x.get("name") == "Substitution" if isinstance(x, dict) else False
    )].iterrows():
        minute, second, period = ev.get("minute", 0), ev.get("second", 0), ev.get("period", 1)
        event_sec = _PERIOD_OFFSET.get(period, 0) + minute * 60 + second
        team_id   = ev.get("team", {}).get("id")

        sub_out_id = ev.get("player", {}).get("id")
        sub_detail = ev.get("substitution", {})
        sub_in_id  = (sub_detail.get("replacement", {}).get("id")
                      if isinstance(sub_detail, dict) else None)

        out_pos = _position_at(player_info.get(sub_out_id, {}).get("positions", []), event_sec)
        in_poss = player_info.get(sub_in_id, {}).get("positions", [])
        in_pos  = in_poss[0].get("position") if in_poss else None

        home_g, away_g = score_at.get(ev.name, (0, 0))
        all_rows.append({
            "game_id":        match_id,
            "league_id":      league_id_map[match_id],
            "team_id":        team_id,
            "sub_in_id":      sub_in_id,
            "sub_out_id":     sub_out_id,
            "position_in":    in_pos,
            "position_out":   out_pos,
            "period":         period,
            "minute":         minute,
            "second":         second,
            "timestamp":      ev.get("timestamp", f"{minute:02d}:{second:02d}"),
            "current_result": f"{home_g}-{away_g}",
        })

    if (i + 1) % 50 == 0 or (i + 1) == len(valid_match_ids):
        print(f"  {i + 1}/{len(valid_match_ids)}")

subs_df = pd.DataFrame(all_rows)

conn = duckdb.connect(str(DUCKDB_PATH))
conn.execute("CREATE OR REPLACE TABLE substitutions AS SELECT * FROM subs_df")
conn.close()

print(f"\n✅ substitutions: {len(subs_df)} rows")
print(subs_df[["minute"]].describe().to_string())
