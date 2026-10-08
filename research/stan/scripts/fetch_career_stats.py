"""Fetch career PA/IP for foreign players via MLB Stats API.

Uses the free MLB Stats API (no authentication required) to look up
career batting/pitching stats by player name.

Output: data/foreign/career_stats.csv
  columns: npb_name, english_name, origin_league, mlb_api_id, career_pa, career_ip, match_quality

Usage:
  python scripts/fetch_career_stats.py
"""

from __future__ import annotations

import csv
import sys
import time
import unicodedata
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
MASTER_CSV = ROOT / "data" / "foreign" / "foreign_players_master.csv"
OUTPUT_CSV = ROOT / "data" / "foreign" / "career_stats.csv"

MLB_SEARCH_URL = "https://statsapi.mlb.com/api/v1/people/search"
MLB_STATS_URL = "https://statsapi.mlb.com/api/v1/people/{pid}/stats"

# Only try API lookup for these leagues (others don't have MLB API entries)
SEARCHABLE_LEAGUES = {"MLB", "AAA", "MiLB"}

# Rate limiting: be nice to the API
REQUEST_DELAY = 0.3  # seconds between requests


def normalize_name(name: str) -> str:
    """Remove accents and normalize Unicode for matching."""
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).strip()


def search_player(name: str) -> list[dict]:
    """Search MLB Stats API for a player by name."""
    try:
        r = requests.get(
            MLB_SEARCH_URL,
            params={"names": name, "sportIds": "1,11,12,13,14"},
            timeout=15,
        )
        if r.status_code == 200:
            return r.json().get("people", [])
    except requests.RequestException:
        pass
    return []


def get_career_stats(player_id: int, group: str) -> dict:
    """Get career stats for a player from MLB Stats API."""
    try:
        r = requests.get(
            MLB_STATS_URL.format(pid=player_id),
            params={"stats": "career", "group": group},
            timeout=15,
        )
        if r.status_code == 200:
            stats = r.json().get("stats", [])
            if stats and stats[0].get("splits"):
                return stats[0]["splits"][0].get("stat", {})
    except requests.RequestException:
        pass
    return {}


def pick_best_match(
    candidates: list[dict],
    english_name: str,
    npb_first_year: int,
) -> tuple[dict | None, str]:
    """Pick the best matching player from API results.

    Returns (best_match, quality) where quality is 'exact', 'likely', or 'ambiguous'.
    """
    if not candidates:
        return None, "none"

    target = normalize_name(english_name).lower()
    target_parts = set(target.split())

    scored = []
    for c in candidates:
        full = normalize_name(c.get("fullName", "")).lower()
        full_parts = set(full.split())

        # Name similarity
        overlap = len(target_parts & full_parts)
        name_score = overlap / max(len(target_parts), 1)

        # Year proximity: prefer players active near NPB entry year
        debut_year = None
        try:
            debut_str = c.get("mlbDebutDate", "")
            if debut_str:
                debut_year = int(debut_str[:4])
        except (ValueError, TypeError):
            pass

        year_score = 0.0
        if debut_year and abs(npb_first_year - debut_year) < 15:
            year_score = 1.0 - abs(npb_first_year - debut_year) / 15.0

        total = name_score * 0.6 + year_score * 0.4
        scored.append((total, name_score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_total, best_name, best = scored[0]

    if best_name >= 0.99:
        quality = "exact"
    elif best_name >= 0.5 and best_total >= 0.5:
        quality = "likely"
    else:
        quality = "ambiguous"

    return best, quality


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    with open(MASTER_CSV, encoding="utf-8-sig") as f:
        master = list(csv.DictReader(f))

    print(f"Loading {len(master)} players from master CSV")

    results = []
    n_found = 0
    n_searched = 0

    for i, row in enumerate(master):
        npb_name = row["npb_name"]
        english_name = row["english_name"]
        origin_league = row["origin_league"]
        player_type = row["player_type"]
        npb_year = int(row["npb_first_year"])

        career_pa = None
        career_ip = None
        mlb_api_id = ""
        match_quality = "skip"

        if origin_league in SEARCHABLE_LEAGUES and english_name.strip():
            n_searched += 1

            # Search by full name
            candidates = search_player(english_name)

            # If no results, try without accents
            if not candidates:
                clean_name = normalize_name(english_name)
                if clean_name != english_name:
                    candidates = search_player(clean_name)

            # If still no results, try last name only
            if not candidates:
                parts = english_name.strip().split()
                if len(parts) >= 2:
                    candidates = search_player(parts[-1])

            best, match_quality = pick_best_match(candidates, english_name, npb_year)

            if best and match_quality in ("exact", "likely"):
                pid = best["id"]
                mlb_api_id = str(pid)

                # Get career hitting stats
                hitting = get_career_stats(pid, "hitting")
                if hitting:
                    career_pa = hitting.get("plateAppearances")

                # Get career pitching stats
                pitching = get_career_stats(pid, "pitching")
                if pitching:
                    career_ip_str = pitching.get("inningsPitched", "0")
                    try:
                        career_ip = float(career_ip_str)
                    except (ValueError, TypeError):
                        career_ip = None

                n_found += 1
                time.sleep(REQUEST_DELAY)

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(master)} (found: {n_found})")

        results.append({
            "npb_name": npb_name,
            "english_name": english_name,
            "origin_league": origin_league,
            "player_type": player_type,
            "mlb_api_id": mlb_api_id,
            "career_pa": career_pa if career_pa is not None else "",
            "career_ip": career_ip if career_ip is not None else "",
            "match_quality": match_quality,
        })

    # Save results
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "npb_name", "english_name", "origin_league", "player_type",
                "mlb_api_id", "career_pa", "career_ip", "match_quality",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone! Saved to {OUTPUT_CSV}")
    print(f"  Searched: {n_searched}/{len(master)}")
    print(f"  Found: {n_found}/{n_searched}")
    print(f"  Match quality: "
          f"exact={sum(1 for r in results if r['match_quality'] == 'exact')}, "
          f"likely={sum(1 for r in results if r['match_quality'] == 'likely')}, "
          f"ambiguous={sum(1 for r in results if r['match_quality'] == 'ambiguous')}, "
          f"skip={sum(1 for r in results if r['match_quality'] == 'skip')}")


if __name__ == "__main__":
    main()
