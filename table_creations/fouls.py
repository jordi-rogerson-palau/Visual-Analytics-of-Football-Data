"""fouls.py — build the `duels` table (Duel + Foul Committed events) in DuckDB."""
import json
import pandas as pd
import duckdb
from _config import DATA_DIR, DUCKDB_PATH, TOP5_LEAGUES, TARGET_SEASON

TARGET_TYPES = {"Duel", "Foul Committed"}

# ── Collect target match ids ──────────────────────────────────────────────
target_ids = set()
for season_file in (DATA_DIR / "matches").glob("**/*.json"):
    with open(season_file, encoding="utf-8") as f:
        for m in json.load(f):
            if (m.get("competition", {}).get("competition_name") in TOP5_LEAGUES and
                    m.get("season", {}).get("season_name") == TARGET_SEASON):
                target_ids.add(m["match_id"])

# ── Extract duel/foul events ──────────────────────────────────────────────
all_rows = []
for match_id in sorted(target_ids):
    events_path = DATA_DIR / "events" / f"{match_id}.json"
    if not events_path.exists():
        print(f"  ⚠ missing events: {match_id}")
        continue
    with open(events_path, encoding="utf-8") as f:
        events = json.load(f)

    for e in events:
        event_type = e.get("type", {}).get("name")
        if event_type not in TARGET_TYPES:
            continue
        foul          = e.get("foul_committed", {})
        bad_behaviour = e.get("bad_behaviour", {})
        # Cards live in foul_committed.card OR bad_behaviour.card
        card = (foul.get("card", {}).get("name") or
                bad_behaviour.get("card", {}).get("name") or "")
        all_rows.append({
            "match_id":     match_id,
            "event_id":     e.get("id"),
            "team_id":      e.get("team",   {}).get("id"),
            "player_id":    e.get("player", {}).get("id"),
            "event_type":   event_type,
            "duel_type":    e.get("duel", {}).get("type",    {}).get("name"),
            "duel_outcome": e.get("duel", {}).get("outcome", {}).get("name"),
            "foul":         event_type == "Foul Committed",
            "yellow_card":  card in ("Yellow Card", "Second Yellow"),
            "red_card":     card in ("Red Card",    "Second Yellow"),
        })

duels_df = pd.DataFrame(all_rows)

con = duckdb.connect(str(DUCKDB_PATH))
con.execute("CREATE OR REPLACE TABLE duels AS SELECT * FROM duels_df")
con.close()

print(f"✅ duels: {len(duels_df):,} rows")
print(duels_df[["foul", "yellow_card", "red_card"]].sum().to_string())

# ── Sanity checks ─────────────────────────────────────────────────────────
n = duels_df["match_id"].nunique()
print(f"\nMatches:           {n}")
print(f"Fouls/match:       {duels_df['foul'].sum() / n:.1f}   (expect ~25–30)")
print(f"Yellows/match:     {duels_df['yellow_card'].sum() / n:.1f}  (expect ~3–4)")
print(f"Reds/match:        {duels_df['red_card'].sum() / n:.2f} (expect ~0.1–0.3)")
