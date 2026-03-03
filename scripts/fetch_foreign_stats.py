"""Fetch previous-league stats for 2026 NPB foreign newcomers using pybaseball.

Run in GitHub Actions (not locally — requires pybaseball + network access):
    pip install pybaseball pandas
    python scripts/fetch_foreign_stats.py

Based on npb-bayes-projection/scripts/fetch_prev_stats.py pattern.
Fetches FanGraphs MLB stats. AAA-only players use MANUAL_STATS fallback.
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
NAMES_CSV = ROOT / "data" / "foreign_2026_names.csv"
OUTPUT = ROOT / "data" / "foreign_2026.csv"

# Search parameters
MAX_LOOKBACK = 6  # years to look back from NPB entry
MIN_PA = 20       # low threshold to catch partial MLB seasons
MIN_IP = 5.0

# Default expected playing time for NPB first year
DEFAULT_PA = 350
DEFAULT_IP_STARTER = 120
DEFAULT_IP_RELIEVER = 55

# Name aliases: names CSV → FanGraphs name
# For players whose name spelling differs between sources
NAME_ALIASES: dict[str, str] = {
    "Jose Quijada": "José Quijada",
    "Miguel Sano": "Miguel Sanó",
    "Jesus Liranzo": "Jesús Liranzo",
}

# Manual stats for players NOT in FanGraphs MLB data
# (AAA-only, international leagues, etc.)
# Source: Deep Research ⑧ (2026-03-03)
MANUAL_STATS: dict[str, dict] = {
    # AAA pitchers
    "リランソ": {
        "prev_ERA": 3.39,
        "expected_ip": 50,
        "source": "Jesus Liranzo AAA Nashville 2025 (48G, ERA 3.39, K64)",
    },
    # AAA hitters — wOBA estimated from OPS (rough: wOBA ≈ OPS * 0.42)
    # McCusker/Castro/Seymour have some MLB PA so FanGraphs may find them.
    # These are fallbacks if FanGraphs doesn't match.
    "マッカスカー": {
        "prev_wOBA": 0.334,  # AAA OPS .795 → estimated wOBA
        "expected_pa": 350,
        "source": "Carson McCusker AAA St. Paul 2025 (106G, 22HR, OPS .795)",
    },
    "シーモア": {
        "prev_wOBA": 0.370,  # AAA OPS .880 → estimated wOBA
        "expected_pa": 400,
        "source": "Robert Seymour AAA Durham 2025 (105G, 30HR, OPS .880)",
    },
}

# Known starter/reliever classification for IP estimation
KNOWN_STARTERS: set[str] = {"ウィットリー", "コックス"}


def normalize_name(name: str) -> str:
    """Normalize player name for fuzzy matching."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_name = ascii_name.lower()
    ascii_name = re.sub(r"\b(jr\.?|sr\.?|ii|iii|iv)\b", "", ascii_name)
    ascii_name = " ".join(ascii_name.split())
    return ascii_name.strip()


def fetch_yearly_stats(
    years: list[int],
) -> tuple[dict[tuple[str, int], pd.Series], dict[tuple[str, int], pd.Series]]:
    """Fetch FanGraphs batting & pitching stats for given years."""
    batting_lookup: dict[tuple[str, int], pd.Series] = {}
    pitching_lookup: dict[tuple[str, int], pd.Series] = {}

    for year in years:
        if year < 2015 or year > 2025:
            continue

        print(f"\n--- {year} ---")

        print("  Fetching batting stats...")
        try:
            df = batting_stats(year, year, qual=0)
            for _, row in df.iterrows():
                name = normalize_name(str(row["Name"]))
                pa = row.get("PA", 0)
                if pd.notna(pa) and int(pa) >= MIN_PA:
                    key = (name, year)
                    if key not in batting_lookup or int(row["PA"]) > int(
                        batting_lookup[key]["PA"]
                    ):
                        batting_lookup[key] = row
            print(
                f"  Batters indexed: {sum(1 for k in batting_lookup if k[1] == year)}"
            )
        except Exception as e:
            print(f"  Batting error: {e}")
        time.sleep(3)

        print("  Fetching pitching stats...")
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


def main() -> None:
    # Read names CSV
    with open(NAMES_CSV, encoding="utf-8-sig") as f:
        players = list(csv.DictReader(f))

    print(f"Total players in names CSV: {len(players)}")

    # Determine years to fetch
    years_needed: set[int] = set()
    for p in players:
        try:
            y = int(p["npb_first_year"])
            for offset in range(1, MAX_LOOKBACK + 1):
                years_needed.add(y - offset)
        except ValueError:
            pass

    sorted_years = sorted(years_needed, reverse=True)
    print(f"Years to fetch: {sorted_years}")

    # Fetch FanGraphs data
    batting_lookup, pitching_lookup = fetch_yearly_stats(sorted_years)
    print(f"\nTotal batting entries: {len(batting_lookup)}")
    print(f"Total pitching entries: {len(pitching_lookup)}")

    # Match players
    results: list[dict] = []
    matched_fg: list[str] = []
    matched_manual: list[str] = []
    unmatched: list[str] = []

    for p in players:
        npb_name = p["npb_name"]
        english_name = p["english_name"].strip()
        player_type = p["player_type"]
        origin_league = p["origin_league"]
        try:
            first_year = int(p["npb_first_year"])
        except ValueError:
            continue

        # Apply name alias, then normalize for FanGraphs matching
        lookup_name = NAME_ALIASES.get(english_name, english_name)
        norm_name = normalize_name(lookup_name)

        found = False
        for offset in range(1, MAX_LOOKBACK + 1):
            search_year = first_year - offset

            if player_type == "hitter":
                row = batting_lookup.get((norm_name, search_year))
                if row is not None:
                    woba = safe_float(row.get("wOBA"))
                    pa = int(row.get("PA", 0))
                    source_str = (
                        f"{english_name} MLB {search_year} ({pa} PA, "
                        f"wOBA {woba})"
                    )
                    results.append({
                        "npb_name": npb_name,
                        "player_type": player_type,
                        "origin_league": origin_league,
                        "prev_wOBA": woba,
                        "prev_ERA": "",
                        "expected_pa": DEFAULT_PA,
                        "expected_ip": "",
                        "source": source_str,
                    })
                    matched_fg.append(f"{npb_name} ({english_name}, {search_year})")
                    found = True
                    break

            elif player_type == "pitcher":
                row = pitching_lookup.get((norm_name, search_year))
                if row is not None:
                    era = safe_float(row.get("ERA"), ".2f")
                    ip = float(row.get("IP", 0))
                    exp_ip = (
                        DEFAULT_IP_STARTER
                        if npb_name in KNOWN_STARTERS
                        else DEFAULT_IP_RELIEVER
                    )
                    source_str = (
                        f"{english_name} MLB {search_year} ({ip:.1f} IP, "
                        f"ERA {era})"
                    )
                    results.append({
                        "npb_name": npb_name,
                        "player_type": player_type,
                        "origin_league": origin_league,
                        "prev_wOBA": "",
                        "prev_ERA": era,
                        "expected_pa": "",
                        "expected_ip": exp_ip,
                        "source": source_str,
                    })
                    matched_fg.append(f"{npb_name} ({english_name}, {search_year})")
                    found = True
                    break

        if not found:
            # Try MANUAL_STATS fallback
            if npb_name in MANUAL_STATS:
                ms = MANUAL_STATS[npb_name]
                results.append({
                    "npb_name": npb_name,
                    "player_type": player_type,
                    "origin_league": origin_league,
                    "prev_wOBA": ms.get("prev_wOBA", ""),
                    "prev_ERA": ms.get("prev_ERA", ""),
                    "expected_pa": ms.get("expected_pa", ""),
                    "expected_ip": ms.get("expected_ip", ""),
                    "source": ms.get("source", "manual"),
                })
                matched_manual.append(f"{npb_name} (manual)")
            else:
                # No stats found — will use predict_no_prev_stats()
                exp = ""
                if player_type == "hitter":
                    exp_pa = DEFAULT_PA
                    exp_ip_val = ""
                else:
                    exp_pa = ""
                    exp_ip_val = (
                        DEFAULT_IP_STARTER
                        if npb_name in KNOWN_STARTERS
                        else DEFAULT_IP_RELIEVER
                    )
                results.append({
                    "npb_name": npb_name,
                    "player_type": player_type,
                    "origin_league": origin_league,
                    "prev_wOBA": "",
                    "prev_ERA": "",
                    "expected_pa": exp_pa,
                    "expected_ip": exp_ip_val,
                    "source": f"{english_name} — no FanGraphs stats found",
                })
                unmatched.append(
                    f"{npb_name} ({english_name}, {origin_league})"
                )

    # Write output
    fields = [
        "npb_name", "player_type", "origin_league",
        "prev_wOBA", "prev_ERA", "expected_pa", "expected_ip", "source",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'=' * 50}")
    print(f"Results written to: {OUTPUT}")
    print(f"FanGraphs matched: {len(matched_fg)}")
    print(f"Manual stats: {len(matched_manual)}")
    print(f"No stats: {len(unmatched)}")

    if matched_fg:
        print("\nFanGraphs matches:")
        for m in matched_fg:
            print(f"  {m}")

    if matched_manual:
        print("\nManual stats:")
        for m in matched_manual:
            print(f"  {m}")

    if unmatched:
        print("\nNo stats found (will use league average):")
        for u in unmatched:
            print(f"  {u}")


if __name__ == "__main__":
    main()
