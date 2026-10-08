"""Fetch previous-league stats for foreign players using pybaseball (FanGraphs).

Run in GitHub Codespaces:
    pip install pybaseball pandas
    python scripts/fetch_prev_stats.py

Covers MLB-level stats only. Players with origin_league in
(MLB, AAA, MiLB, KBO) who had any MLB stint will be captured.
Pure Independent/Cuba/Amateur players require manual collection.
"""

from __future__ import annotations

import csv
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
from pybaseball import batting_stats, pitching_stats

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "data" / "foreign" / "foreign_players_master.csv"
OUTPUT = ROOT / "data" / "foreign" / "foreign_prev_stats.csv"

# Minimum thresholds for previous-league stats (lenient to capture partial seasons)
MIN_PA = 30
MIN_IP = 10.0

# Max years to look back from NPB entry
MAX_LOOKBACK = 6

# Name aliases: master CSV name → FanGraphs name
# For players whose name order or spelling differs between sources
NAME_ALIASES: dict[str, str] = {
    "Chen Wei-Yin": "Wei-Yin Chen",
    "Wily Mo Pena": "Wily Mo Peña",
}


def normalize_name(name: str) -> str:
    """Normalize player name for fuzzy matching."""
    # Remove accents
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Lowercase
    ascii_name = ascii_name.lower()
    # Remove Jr., Sr., II, III, IV
    ascii_name = re.sub(r"\b(jr\.?|sr\.?|ii|iii|iv)\b", "", ascii_name)
    # Remove extra whitespace
    ascii_name = " ".join(ascii_name.split())
    return ascii_name.strip()


def fetch_yearly_stats(
    years: set[int],
) -> tuple[dict[tuple[str, int], pd.Series], dict[tuple[str, int], pd.Series]]:
    """Fetch FanGraphs batting & pitching stats for given years.

    Returns:
        (batting_lookup, pitching_lookup) keyed by (normalized_name, year)
    """
    batting_lookup: dict[tuple[str, int], pd.Series] = {}
    pitching_lookup: dict[tuple[str, int], pd.Series] = {}

    for year in sorted(years):
        if year < 2010 or year > 2025:
            continue

        print(f"\n--- {year} ---")

        # Batting
        print(f"  Fetching batting stats...")
        try:
            df = batting_stats(year, year, qual=0)
            for _, row in df.iterrows():
                name = normalize_name(str(row["Name"]))
                pa = row.get("PA", 0)
                if pd.notna(pa) and int(pa) >= MIN_PA:
                    key = (name, year)
                    # Keep the row with more PA if duplicate
                    if key not in batting_lookup or int(row["PA"]) > int(
                        batting_lookup[key]["PA"]
                    ):
                        batting_lookup[key] = row
            print(f"  Batters indexed: {sum(1 for k in batting_lookup if k[1] == year)}")
        except Exception as e:
            print(f"  Batting error: {e}")
        time.sleep(3)

        # Pitching
        print(f"  Fetching pitching stats...")
        try:
            df = pitching_stats(year, year, qual=0)
            for _, row in df.iterrows():
                name = normalize_name(str(row["Name"]))
                ip = row.get("IP", 0)
                if pd.notna(ip) and float(ip) >= MIN_IP:
                    key = (name, year)
                    if key not in pitching_lookup or float(row["IP"]) > float(
                        pitching_lookup[key]["IP"]
                    ):
                        pitching_lookup[key] = row
            print(
                f"  Pitchers indexed: {sum(1 for k in pitching_lookup if k[1] == year)}"
            )
        except Exception as e:
            print(f"  Pitching error: {e}")
        time.sleep(3)

    return batting_lookup, pitching_lookup


def safe_float(val, fmt: str = ".3f") -> str:
    """Safely format a float value."""
    try:
        if pd.isna(val):
            return ""
        return f"{float(val):{fmt}}"
    except (ValueError, TypeError):
        return ""


def safe_int(val) -> str:
    """Safely format an int value."""
    try:
        if pd.isna(val):
            return ""
        return str(int(val))
    except (ValueError, TypeError):
        return ""


def safe_pct(val) -> str:
    """Format K% / BB% (FanGraphs stores as decimal like 0.225 or percentage 22.5)."""
    try:
        if pd.isna(val):
            return ""
        v = float(val)
        # FanGraphs sometimes returns as ratio (0.225) or pct (22.5)
        # If < 1, it's a ratio → convert to pct
        if v < 1:
            v = v * 100
        return f"{v:.1f}"
    except (ValueError, TypeError):
        return ""


def main() -> None:
    # Read master
    with open(MASTER, encoding="utf-8-sig") as f:
        master = list(csv.DictReader(f))

    # Filter to players with english_name
    eligible = [p for p in master if p.get("english_name", "").strip()]
    print(f"Total players: {len(master)}")
    print(f"Eligible (has english_name): {len(eligible)}")

    # Determine years needed: npb_first_year - 1 to -MAX_LOOKBACK (fallback)
    years_needed: set[int] = set()
    for p in eligible:
        try:
            y = int(p["npb_first_year"])
            for offset in range(1, MAX_LOOKBACK + 1):
                years_needed.add(y - offset)
        except ValueError:
            pass
    print(f"Years to fetch: {sorted(years_needed)}")

    # Fetch FanGraphs data
    batting_lookup, pitching_lookup = fetch_yearly_stats(years_needed)
    print(f"\nTotal batting entries: {len(batting_lookup)}")
    print(f"Total pitching entries: {len(pitching_lookup)}")

    # Match players
    results: list[dict] = []
    matched_names: set[str] = set()
    unmatched: list[str] = []

    for p in eligible:
        english_name = p["english_name"].strip()
        origin_league = p["origin_league"]
        player_type = p["player_type"]
        try:
            first_year = int(p["npb_first_year"])
        except ValueError:
            continue

        # Apply name alias if exists, then normalize for FanGraphs matching
        lookup_name = NAME_ALIASES.get(english_name, english_name)
        norm_name = normalize_name(lookup_name)

        # Try year-1 first, then year-2, ... up to MAX_LOOKBACK
        found = False
        for offset in range(1, MAX_LOOKBACK + 1):
            search_year = first_year - offset

            if player_type == "hitter":
                row = batting_lookup.get((norm_name, search_year))
                if row is not None:
                    results.append(
                        {
                            "english_name": english_name,
                            "npb_name": p["npb_name"],
                            "origin_league": origin_league,
                            "season": search_year,
                            "PA": safe_int(row.get("PA")),
                            "AVG": safe_float(row.get("AVG")),
                            "OBP": safe_float(row.get("OBP")),
                            "SLG": safe_float(row.get("SLG")),
                            "OPS": safe_float(row.get("OPS")),
                            "wOBA": safe_float(row.get("wOBA")),
                            "HR": safe_int(row.get("HR")),
                            "IP": "",
                            "ERA": "",
                            "FIP": "",
                            "K_pct": safe_pct(row.get("K%")),
                            "BB_pct": safe_pct(row.get("BB%")),
                            "WHIP": "",
                        }
                    )
                    matched_names.add(english_name)
                    found = True
                    break

            elif player_type == "pitcher":
                row = pitching_lookup.get((norm_name, search_year))
                if row is not None:
                    results.append(
                        {
                            "english_name": english_name,
                            "npb_name": p["npb_name"],
                            "origin_league": origin_league,
                            "season": search_year,
                            "PA": "",
                            "AVG": "",
                            "OBP": "",
                            "SLG": "",
                            "OPS": "",
                            "wOBA": "",
                            "HR": "",
                            "IP": safe_float(row.get("IP"), ".1f"),
                            "ERA": safe_float(row.get("ERA"), ".2f"),
                            "FIP": safe_float(row.get("FIP"), ".2f"),
                            "K_pct": safe_pct(row.get("K%")),
                            "BB_pct": safe_pct(row.get("BB%")),
                            "WHIP": safe_float(row.get("WHIP"), ".2f"),
                        }
                    )
                    matched_names.add(english_name)
                    found = True
                    break

        if not found:
            unmatched.append(
                f"{english_name} ({origin_league}, {player_type}, {first_year})"
            )

    # Write output
    fields = [
        "english_name",
        "npb_name",
        "origin_league",
        "season",
        "PA",
        "AVG",
        "OBP",
        "SLG",
        "OPS",
        "wOBA",
        "HR",
        "IP",
        "ERA",
        "FIP",
        "K_pct",
        "BB_pct",
        "WHIP",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'=' * 50}")
    print(f"Results written to: {OUTPUT}")
    print(f"Matched: {len(results)}/{len(eligible)}")
    print(f"Unmatched: {len(unmatched)}")

    # League breakdown of matched
    from collections import Counter

    league_counts = Counter(r["origin_league"] for r in results)
    print(f"\nMatched by league:")
    for league, count in league_counts.most_common():
        print(f"  {league}: {count}")

    # Unmatched list
    if unmatched:
        print(f"\nUnmatched players:")
        for u in unmatched:
            print(f"  {u}")


if __name__ == "__main__":
    main()
