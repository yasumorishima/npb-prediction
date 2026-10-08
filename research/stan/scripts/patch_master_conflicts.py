"""Detect and patch foreign player master data conflicts.

Problem:
  foreign_players_master.csv uses katakana name as primary key.
  When two different foreign players share the same katakana name, only
  the first one gets an entry, causing the second to be missed in backtests.

Detection rules (either triggers flagging):
  (a) SAME_YEAR: two different teams appear for the same katakana name in the same year
      → definitely different people
  (b) GAP: a new team appears for a katakana name, but the name had no NPB appearances
      in the previous year → likely returned from overseas as a different person

Resolution:
  1. pykakasi: katakana → romaji (extract surname)
  2. pybaseball: FanGraphs surname search in years before NPB debut
  3. Single new candidate → add to master automatically
  4. Already in master (same player, just changed teams) → skip with note
  5. Multiple or no candidates → needs_review.csv for manual review

Outputs:
  data/foreign/foreign_players_master.csv  (updated with new entries, sorted)
  data/foreign/needs_review.csv            (unresolved conflicts)
"""

from __future__ import annotations

import csv
import difflib
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
from pybaseball import batting_stats, pitching_stats

ROOT = Path(__file__).resolve().parent.parent
MASTER_PATH = ROOT / "data" / "foreign" / "foreign_players_master.csv"
SABERMETRICS_PATH = ROOT / "data" / "raw" / "npb_sabermetrics_2015_2025.csv"
PITCHERS_PATH = ROOT / "data" / "raw" / "npb_pitchers_2015_2025.csv"
NEEDS_REVIEW_PATH = ROOT / "data" / "foreign" / "needs_review.csv"

MIN_PA = 30
MIN_IP = 10.0
MAX_LOOKBACK = 6
# Minimum PA to include a batter in FanGraphs candidate search.
# Filters out pitchers who occasionally appear as DH/PH in batting stats.
FG_MIN_PA = 50

# Pykakasi converts katakana via Japanese phonetic rules, which doesn't
# match Spanish/Latin surnames.  Map pykakasi output → actual surname.
ROMAJI_TO_SURNAME: dict[str, str] = {
    # Rodriguez family  (trailing "z" → "-su" in Japanese)
    "rodorigesu": "rodriguez",
    "gonzaresu": "gonzalez",
    "ramiresu": "ramirez",
    "peresu": "perez",
    "sanchesu": "sanchez",
    "suaresu": "suarez",
    "orutisu": "ortiz",
    "arubaresu": "alvarez",
    "arubarezu": "alvarez",
    # Martinez / Hernandez / Fernandez
    "marutinesu": "martinez",
    "maruteinesu": "martinez",
    "herunandesu": "hernandez",
    "erunandesu": "hernandez",
    "ferunandesu": "fernandez",
    "fuerunandesu": "fernandez",
    # Mejia / Jimenez  (Spanish "j" → "h" in Japanese)
    "mehia": "mejia",
    "himenesu": "jimenez",
    "jimenesu": "jimenez",
    # Garcia  (Japanese transcribes as ガルシア/アルシア)
    "garushia": "garcia",
    "arushia": "garcia",
    # Escobar  (エスコバール → long vowel at end)
    "esukoba": "escobar",
    "esukobaa": "escobar",
    "esukobaru": "escobar",
    # Other common patterns
    "toresu": "torres",
    "toreesu": "torres",
    "furooresu": "flores",
    "furoorezu": "flores",
}

try:
    import pykakasi
    _kks = pykakasi.kakasi()
    HAS_PYKAKASI = True
except ImportError:
    _kks = None
    HAS_PYKAKASI = False
    print("WARNING: pykakasi not installed. Romaji conversion disabled.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_katakana_name(name: str) -> bool:
    cleaned = name.replace("\u3000", "").replace(" ", "").replace("\u3000", "").strip()
    if not cleaned:
        return False
    n = sum(1 for c in cleaned if "\u30A0" <= c <= "\u30FF")
    return n / len(cleaned) > 0.5


def norm_katakana(name: str) -> str:
    return name.replace("\u3000", "").replace(" ", "").replace("\u3000", "").strip()


def katakana_to_romaji(text: str) -> str:
    if not HAS_PYKAKASI or _kks is None:
        return ""
    result = _kks.convert(text)
    return " ".join(item["hepburn"] for item in result).strip()


def normalize_ascii(name: str) -> str:
    """Remove accents, lowercase, strip suffixes."""
    nfkd = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in nfkd if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"\b(jr\.?|sr\.?|ii|iii|iv)\b", "", s)
    return " ".join(s.split()).strip()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_master() -> list[dict]:
    with open(MASTER_PATH, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_npb_data() -> tuple[
    dict[str, list[tuple[int, str]]],   # name → sorted [(year, team)]
    dict[str, str],                      # name → 'hitter' | 'pitcher'
    dict[tuple[str, int, str], dict],    # (name, year, team) → row stats
]:
    """Load all katakana player appearances from raw NPB stats.

    Returns:
        appearances: sorted list of (year, team) per katakana name
        player_types: dominant type per name
        row_stats: raw stat row for first-year NPB lookup
    """
    appearances: dict[str, set[tuple[int, str]]] = defaultdict(set)
    type_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"hitter": 0, "pitcher": 0})
    row_stats: dict[tuple[str, int, str], dict] = {}

    # Hitters
    with open(SABERMETRICS_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = norm_katakana(row["player"])
            if not is_katakana_name(name):
                continue
            try:
                pa = float(row["PA"])
                year = int(row["year"])
            except (ValueError, KeyError):
                continue
            if pa < MIN_PA:
                continue
            team = row["team"].strip()
            appearances[name].add((year, team))
            type_counts[name]["hitter"] += 1
            key = (name, year, team)
            if key not in row_stats:
                row_stats[key] = {
                    "npb_first_year_PA": row.get("PA", ""),
                    "npb_first_year_AVG": row.get("AVG", ""),
                    "npb_first_year_OPS": row.get("OPS", ""),
                    "npb_first_year_wOBA": row.get("wOBA", ""),
                    "npb_first_year_ERA": "",
                    "npb_first_year_IP": "",
                    "npb_first_year_WHIP": "",
                }

    # Pitchers
    with open(PITCHERS_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = norm_katakana(row["player"])
            if not is_katakana_name(name):
                continue
            try:
                ip = float(row["IP"])
                year = int(row["year"])
            except (ValueError, KeyError):
                continue
            if ip < MIN_IP:
                continue
            team = row["team"].strip()
            appearances[name].add((year, team))
            type_counts[name]["pitcher"] += 1
            key = (name, year, team)
            if key not in row_stats:
                row_stats[key] = {
                    "npb_first_year_PA": "",
                    "npb_first_year_AVG": "",
                    "npb_first_year_OPS": "",
                    "npb_first_year_wOBA": "",
                    "npb_first_year_ERA": row.get("ERA", ""),
                    "npb_first_year_IP": row.get("IP", ""),
                    "npb_first_year_WHIP": row.get("WHIP", ""),
                }

    sorted_apps = {k: sorted(v) for k, v in appearances.items()}
    player_types = {
        k: ("pitcher" if v["pitcher"] >= v["hitter"] else "hitter")
        for k, v in type_counts.items()
    }
    return sorted_apps, player_types, row_stats


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

def detect_conflicts(
    master: list[dict],
    appearances: dict[str, list[tuple[int, str]]],
    player_types: dict[str, str],
    row_stats: dict[tuple[str, int, str], dict],
) -> list[dict]:
    """Find (katakana_name, team) pairs in sabermetrics not covered by master."""
    # Index master: normalized npb_name → list of rows
    master_index: dict[str, list[dict]] = defaultdict(list)
    for row in master:
        master_index[norm_katakana(row["npb_name"])].append(row)

    conflicts = []

    for name, year_team_list in appearances.items():
        master_rows = master_index.get(name)
        if not master_rows:
            continue  # Unknown katakana name, might be Japanese with nickname

        all_years = {yr for yr, _ in year_team_list}
        master_teams = {r["first_team"].strip() for r in master_rows if r.get("first_team")}

        # First year per team in sabermetrics
        team_debut: dict[str, int] = {}
        for yr, team in year_team_list:
            if team not in team_debut or yr < team_debut[team]:
                team_debut[team] = yr

        for team, debut_year in team_debut.items():
            if team in master_teams:
                continue  # This team is already represented in master

            # Rule (a): same year, two teams → definitely different people
            same_year_teams = {tm for yr, tm in year_team_list if yr == debut_year}
            same_year_conflict = len(same_year_teams) > 1

            # Rule (b): no NPB appearance in year before debut → likely different person
            no_recent = (debut_year - 1) not in all_years

            if same_year_conflict or no_recent:
                stats = row_stats.get((name, debut_year, team), {})
                conflicts.append({
                    "npb_name": name,
                    "conflict_team": team,
                    "conflict_first_year": debut_year,
                    "player_type": player_types.get(name, master_rows[0].get("player_type", "")),
                    "master_english": master_rows[0]["english_name"],
                    "master_first_team": master_rows[0]["first_team"],
                    "master_first_year": master_rows[0]["npb_first_year"],
                    "master_origin_league": master_rows[0]["origin_league"],
                    "master_origin_country": master_rows[0]["origin_country"],
                    "same_year_conflict": same_year_conflict,
                    "no_recent": no_recent,
                    **stats,
                })

    return sorted(conflicts, key=lambda x: (x["npb_name"], x["conflict_first_year"]))


# ---------------------------------------------------------------------------
# FanGraphs lookup
# ---------------------------------------------------------------------------

def fetch_fg_data(years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch FanGraphs MLB batting + pitching stats for the given years."""
    batting_frames, pitching_frames = [], []

    for year in sorted(set(years)):
        if year < 2015 or year > 2025:
            continue
        print(f"  Fetching FanGraphs {year}...")
        try:
            df = batting_stats(year, year, qual=0)
            df["_year"] = year
            batting_frames.append(df)
        except Exception as e:
            print(f"    batting error: {e}")
        time.sleep(2)
        try:
            df = pitching_stats(year, year, qual=0)
            df["_year"] = year
            pitching_frames.append(df)
        except Exception as e:
            print(f"    pitching error: {e}")
        time.sleep(2)

    batting = pd.concat(batting_frames, ignore_index=True) if batting_frames else pd.DataFrame()
    pitching = pd.concat(pitching_frames, ignore_index=True) if pitching_frames else pd.DataFrame()
    return batting, pitching


def find_fg_candidates(
    surname_romaji: str,
    player_type: str,
    npb_year: int,
    batting_df: pd.DataFrame,
    pitching_df: pd.DataFrame,
) -> list[tuple[str, int, str]]:
    """Search FanGraphs for players whose surname matches surname_romaji.

    Returns list of (english_name, year, origin_league).
    All results are from FanGraphs MLB data → origin_league = 'MLB'.
    """
    if not surname_romaji:
        return []

    norm_srn = normalize_ascii(surname_romaji)
    df = pitching_df if player_type == "pitcher" else batting_df

    # For batters, drop rows where the player has very few PAs (typically pitchers
    # who batted as DH/PH — they can produce false surname matches).
    if player_type == "hitter" and "PA" in df.columns:
        df = df[df["PA"] >= FG_MIN_PA]

    if df.empty or "Name" not in df.columns:
        return []

    # Filter to years before NPB debut (within lookback window)
    df = df[(df["_year"] >= npb_year - MAX_LOOKBACK) & (df["_year"] < npb_year)].copy()

    if df.empty:
        return []

    # Build (fg_name, year, normalized_surname) list
    entries: list[tuple[str, int, str]] = []
    for _, row in df.iterrows():
        fg_name = str(row.get("Name", "")).strip()
        if not fg_name:
            continue
        parts = normalize_ascii(fg_name).split()
        if not parts:
            continue
        fg_surname = parts[-1]
        entries.append((fg_name, int(row["_year"]), fg_surname))

    # Exact match first
    exact = [(n, y, "MLB") for n, y, sn in entries if sn == norm_srn]

    # Fuzzy fallback (edit-distance ≤ 2 = cutoff ≈ 0.75 for short names)
    if not exact:
        all_surnames = list({sn for _, _, sn in entries})
        close = difflib.get_close_matches(norm_srn, all_surnames, n=3, cutoff=0.75)
        fuzzy = [(n, y, "MLB") for n, y, sn in entries if sn in close]
        exact = fuzzy

    # Deduplicate by English name (keep most recent year)
    seen: dict[str, tuple[str, int, str]] = {}
    for n, y, origin in exact:
        if n not in seen or y > seen[n][1]:
            seen[n] = (n, y, origin)

    return list(seen.values())


# ---------------------------------------------------------------------------
# Resolution & master update
# ---------------------------------------------------------------------------

MASTER_FIELDNAMES = [
    "npb_name", "english_name", "origin_league", "origin_country",
    "npb_first_year", "first_team", "player_type", "position", "mlb_id",
    "npb_first_year_PA", "npb_first_year_AVG", "npb_first_year_OPS",
    "npb_first_year_wOBA", "npb_first_year_ERA", "npb_first_year_IP",
    "npb_first_year_WHIP",
]


def resolve_and_update(
    conflicts: list[dict],
    master: list[dict],
    batting_df: pd.DataFrame,
    pitching_df: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    """Resolve conflicts via FanGraphs lookup.

    Returns (new_master_entries, needs_review_rows).
    """
    existing_english = {normalize_ascii(r["english_name"]) for r in master}

    new_entries: list[dict] = []
    needs_review: list[dict] = []

    for c in conflicts:
        name = c["npb_name"]
        team = c["conflict_team"]
        debut_year = c["conflict_first_year"]
        ptype = c["player_type"]

        romaji = katakana_to_romaji(name)
        # Surname = last space-separated token in romaji
        surname_raw = romaji.split()[-1] if romaji.split() else ""
        # Override with Spanish/Latin surname table when pykakasi romaji doesn't match
        surname = ROMAJI_TO_SURNAME.get(surname_raw, surname_raw)

        flag = "SAME_YEAR" if c["same_year_conflict"] else "GAP"
        print(f"\n[{flag}] {name} → {team} ({debut_year})  master={c['master_english']}")
        override_note = f" → {surname!r} (override)" if surname != surname_raw else ""
        print(f"  romaji: {romaji!r}  surname: {surname_raw!r}{override_note}")

        candidates = find_fg_candidates(surname, ptype, debut_year, batting_df, pitching_df)
        print(f"  FanGraphs candidates: {[n for n, _, _ in candidates]}")

        if len(candidates) == 1:
            eng_name, fg_year, origin = candidates[0]
            norm_eng = normalize_ascii(eng_name)

            if norm_eng in existing_english:
                # Same player who just changed teams in NPB — no new entry needed
                print(f"  → Same player already in master ({eng_name}), skipping")
                needs_review.append({
                    "npb_name": name,
                    "conflict_team": team,
                    "conflict_first_year": debut_year,
                    "player_type": ptype,
                    "romaji": romaji,
                    "fg_candidates": eng_name,
                    "reason": "same_player_team_change",
                    "master_english": c["master_english"],
                    "same_year_conflict": c.get("same_year_conflict", False),
                    "no_recent": c.get("no_recent", False),
                    "npb_first_year_PA": c.get("npb_first_year_PA", ""),
                    "npb_first_year_wOBA": c.get("npb_first_year_wOBA", ""),
                    "npb_first_year_ERA": c.get("npb_first_year_ERA", ""),
                    "npb_first_year_IP": c.get("npb_first_year_IP", ""),
                })
                continue

            entry = {f: "" for f in MASTER_FIELDNAMES}
            entry.update({
                "npb_name": name,
                "english_name": eng_name,
                "origin_league": origin,
                "npb_first_year": str(debut_year),
                "first_team": team,
                "player_type": ptype,
                "npb_first_year_PA": c.get("npb_first_year_PA", ""),
                "npb_first_year_AVG": c.get("npb_first_year_AVG", ""),
                "npb_first_year_OPS": c.get("npb_first_year_OPS", ""),
                "npb_first_year_wOBA": c.get("npb_first_year_wOBA", ""),
                "npb_first_year_ERA": c.get("npb_first_year_ERA", ""),
                "npb_first_year_IP": c.get("npb_first_year_IP", ""),
                "npb_first_year_WHIP": c.get("npb_first_year_WHIP", ""),
            })
            new_entries.append(entry)
            existing_english.add(norm_eng)
            print(f"  ✓ Auto-resolved: {eng_name} (MLB {fg_year})")

        else:
            reason = "no_match" if not candidates else f"{len(candidates)}_candidates"
            needs_review.append({
                "npb_name": name,
                "conflict_team": team,
                "conflict_first_year": debut_year,
                "player_type": ptype,
                "romaji": romaji,
                "fg_candidates": "; ".join(f"{n} ({y})" for n, y, _ in candidates[:5]),
                "reason": reason,
                "master_english": c["master_english"],
                "same_year_conflict": c["same_year_conflict"],
                "no_recent": c["no_recent"],
                "npb_first_year_PA": c.get("npb_first_year_PA", ""),
                "npb_first_year_wOBA": c.get("npb_first_year_wOBA", ""),
                "npb_first_year_ERA": c.get("npb_first_year_ERA", ""),
                "npb_first_year_IP": c.get("npb_first_year_IP", ""),
            })
            print(f"  ⚠ Needs review: {reason}")

    return new_entries, needs_review


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== patch_master_conflicts.py ===\n")

    master = load_master()
    print(f"Master entries: {len(master)}")

    appearances, player_types, row_stats = load_npb_data()
    print(f"Unique katakana names in sabermetrics: {len(appearances)}")

    conflicts = detect_conflicts(master, appearances, player_types, row_stats)
    print(f"\nConflicts detected: {len(conflicts)}")
    for c in conflicts:
        flag = "SAME_YEAR" if c["same_year_conflict"] else "GAP"
        print(f"  [{flag}] {c['npb_name']} → {c['conflict_team']} ({c['conflict_first_year']})"
              f"  [master: {c['master_english']}]")

    if not conflicts:
        print("\nNo conflicts found. Master is up to date.")
        NEEDS_REVIEW_PATH.write_text("", encoding="utf-8")
        return

    # Determine years to fetch from FanGraphs
    years_needed = set()
    for c in conflicts:
        y = c["conflict_first_year"]
        for offset in range(1, MAX_LOOKBACK + 1):
            y2 = y - offset
            if 2015 <= y2 <= 2025:
                years_needed.add(y2)
    print(f"\nFetching FanGraphs data for years: {sorted(years_needed)}")
    batting_df, pitching_df = fetch_fg_data(sorted(years_needed))

    new_entries, needs_review = resolve_and_update(conflicts, master, batting_df, pitching_df)

    # Write updated master (original + new, sorted by npb_name)
    if new_entries:
        all_entries = master + new_entries
        all_entries.sort(key=lambda r: r["npb_name"])
        with open(MASTER_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MASTER_FIELDNAMES)
            writer.writeheader()
            writer.writerows(all_entries)
        print(f"\n✓ Master updated: +{len(new_entries)} new entries")
    else:
        print("\nNo new entries added to master.")

    # Write needs_review
    if needs_review:
        nr_fields = list(needs_review[0].keys())
        with open(NEEDS_REVIEW_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=nr_fields)
            writer.writeheader()
            writer.writerows(needs_review)
        print(f"⚠ needs_review.csv: {len(needs_review)} cases require manual review")
    else:
        NEEDS_REVIEW_PATH.write_text("", encoding="utf-8")
        print("✓ All conflicts resolved automatically")

    print(f"\n=== Summary ===")
    print(f"  Conflicts detected : {len(conflicts)}")
    print(f"  Auto-resolved      : {len(new_entries)}")
    print(f"  Needs manual review: {len(needs_review)}")


if __name__ == "__main__":
    main()
