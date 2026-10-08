"""Identify foreign players in NPB datasets.

Extracts foreign (non-Japanese) players from NPB data using:
1. Katakana ratio detection (name > 50% katakana → likely foreign)
2. Profile CSV 出身地 (origin) field for country confirmation
3. Roster CSV for first-year-in-NPB identification

Output: data/foreign/foreign_players_candidates.csv
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "foreign"

JAPANESE_PREFECTURES = {
    "北海道", "青森", "岩手", "宮城", "秋田", "山形", "福島",
    "茨城", "栃木", "群馬", "埼玉", "千葉", "東京", "神奈川",
    "新潟", "富山", "石川", "福井", "山梨", "長野", "岐阜",
    "静岡", "愛知", "三重", "滋賀", "京都", "大阪", "兵庫",
    "奈良", "和歌山", "鳥取", "島根", "岡山", "広島", "山口",
    "徳島", "香川", "愛媛", "高知", "福岡", "佐賀", "長崎",
    "熊本", "大分", "宮崎", "鹿児島", "沖縄",
}

# Map Japanese country names to English + likely origin league
COUNTRY_TO_LEAGUE = {
    "アメリカ": ("USA", "MLB"),
    "ドミニカ共和国": ("Dominican Republic", "MLB"),
    "ベネズエラ": ("Venezuela", "MLB"),
    "キューバ": ("Cuba", "Cuba"),
    "プエルトリコ": ("Puerto Rico", "MLB"),
    "メキシコ": ("Mexico", "MLB"),
    "コロンビア": ("Colombia", "MLB"),
    "パナマ": ("Panama", "MLB"),
    "カナダ": ("Canada", "MLB"),
    "ブラジル": ("Brazil", "Independent"),
    "台湾": ("Taiwan", "CPBL"),
    "韓国": ("South Korea", "KBO"),
}


def is_katakana_name(name: str) -> bool:
    """Return True if name is predominantly katakana (likely foreign)."""
    cleaned = name.replace("\u3000", "").replace(" ", "").replace("　", "")
    if not cleaned:
        return False
    katakana_count = sum(1 for c in cleaned if "\u30A0" <= c <= "\u30FF")
    return katakana_count / len(cleaned) > 0.5


def normalize_name(name: str) -> str:
    """Normalize player name for matching (remove spaces)."""
    return name.replace("\u3000", "").replace(" ", "").replace("　", "").strip()


def load_profiles() -> dict[str, dict]:
    """Load player profiles from npb_players_profile_2024.csv.

    Returns dict keyed by normalized name.
    """
    profiles = {}
    path = RAW_DIR / "npb_players_profile_2024.csv"
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = normalize_name(row["選手名"])
            origin = row["出身地"].strip()
            years_str = row["年数"].replace("年", "").strip()
            try:
                years = int(years_str)
            except ValueError:
                years = 0
            profiles[name] = {
                "origin": origin,
                "years_in_npb": years,
                "team": row["team"].strip(),
                "position": row["守備"].strip(),
                "birthday": row["生年月日"].strip(),
            }
    return profiles


def load_rosters() -> dict[str, list[tuple[int, str]]]:
    """Load roster data, returning {normalized_name: [(year, team), ...]}.

    Sorted by year ascending to find first appearance.
    """
    rosters: dict[str, list[tuple[int, str]]] = defaultdict(list)
    path = RAW_DIR / "npb_rosters_2018_2025.csv"
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = normalize_name(row["player"])
            year = int(row["year"])
            team = row["team"].strip()
            rosters[name].append((year, team))
    for name in rosters:
        rosters[name].sort(key=lambda x: x[0])
    return rosters


def load_hitter_stats() -> dict[str, list[dict]]:
    """Load hitter stats by normalized name."""
    stats: dict[str, list[dict]] = defaultdict(list)
    path = RAW_DIR / "npb_hitters_2015_2025.csv"
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = normalize_name(row["player"])
            try:
                pa = int(row["PA"])
            except (ValueError, KeyError):
                pa = 0
            stats[name].append({
                "year": int(row["year"]),
                "team": row["team"].strip(),
                "AVG": row["AVG"],
                "OPS": row["OPS"],
                "PA": pa,
                "HR": row["HR"],
            })
    return stats


def load_pitcher_stats() -> dict[str, list[dict]]:
    """Load pitcher stats by normalized name."""
    stats: dict[str, list[dict]] = defaultdict(list)
    path = RAW_DIR / "npb_pitchers_2015_2025.csv"
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = normalize_name(row["player"])
            try:
                ip = float(row["IP"])
            except (ValueError, KeyError):
                ip = 0.0
            stats[name].append({
                "year": int(row["year"]),
                "team": row["team"].strip(),
                "ERA": row["ERA"],
                "WHIP": row["WHIP"],
                "IP": ip,
                "SO": row["SO"],
            })
    return stats


def load_sabermetrics() -> dict[tuple[str, int], dict]:
    """Load wOBA data keyed by (normalized_name, year)."""
    saber = {}
    path = RAW_DIR / "npb_sabermetrics_2015_2025.csv"
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = normalize_name(row["player"])
            year = int(row["year"])
            try:
                woba = float(row["wOBA"])
            except (ValueError, KeyError):
                woba = None
            saber[(name, year)] = {
                "wOBA": woba,
                "wRC+": row.get("wRC+", ""),
                "PA": row.get("PA", ""),
            }
    return saber


def identify_foreign_players() -> list[dict]:
    """Identify foreign players and compile their NPB first-year info.

    Strategy:
    1. Get all unique player names from hitter + pitcher stats (2015-2025)
    2. Filter by katakana detection
    3. Cross-reference with profile CSV for country
    4. Use roster CSV to find first NPB year (within data range)
    5. Attach first-year NPB stats
    """
    profiles = load_profiles()
    rosters = load_rosters()
    hitter_stats = load_hitter_stats()
    pitcher_stats = load_pitcher_stats()
    sabermetrics = load_sabermetrics()

    # Collect all player names from stats
    all_names = set(hitter_stats.keys()) | set(pitcher_stats.keys())

    foreign_players = []

    for name in sorted(all_names):
        if not is_katakana_name(name):
            continue

        # Get country from profile
        profile = profiles.get(name, {})
        origin = profile.get("origin", "")

        if origin in JAPANESE_PREFECTURES:
            continue  # Japanese player with katakana-heavy name (rare)

        country_info = COUNTRY_TO_LEAGUE.get(origin, (origin, "Unknown"))
        country_en, likely_league = country_info

        # Find first year in roster data
        roster_entries = rosters.get(name, [])
        first_roster_year = roster_entries[0][0] if roster_entries else None
        first_roster_team = roster_entries[0][1] if roster_entries else None

        # Find first year in stats data
        h_years = [s["year"] for s in hitter_stats.get(name, [])]
        p_years = [s["year"] for s in pitcher_stats.get(name, [])]
        all_years = h_years + p_years
        first_stats_year = min(all_years) if all_years else None

        # Use earliest available year as NPB first year (within our data)
        npb_first_year = None
        if first_roster_year and first_stats_year:
            npb_first_year = min(first_roster_year, first_stats_year)
        elif first_roster_year:
            npb_first_year = first_roster_year
        elif first_stats_year:
            npb_first_year = first_stats_year

        # Determine if hitter or pitcher (by which has more data)
        is_hitter = len(h_years) >= len(p_years) and len(h_years) > 0
        is_pitcher = len(p_years) > len(h_years) and len(p_years) > 0

        # Get first-year stats
        first_year_ops = ""
        first_year_avg = ""
        first_year_pa = ""
        first_year_era = ""
        first_year_ip = ""
        first_year_whip = ""
        first_year_woba = ""

        if npb_first_year and is_hitter:
            for s in hitter_stats.get(name, []):
                if s["year"] == npb_first_year:
                    first_year_ops = s["OPS"]
                    first_year_avg = s["AVG"]
                    first_year_pa = s["PA"]
                    break
            saber_key = (name, npb_first_year)
            if saber_key in sabermetrics and sabermetrics[saber_key]["wOBA"]:
                first_year_woba = f"{sabermetrics[saber_key]['wOBA']:.4f}"

        if npb_first_year and is_pitcher:
            for s in pitcher_stats.get(name, []):
                if s["year"] == npb_first_year:
                    first_year_era = s["ERA"]
                    first_year_ip = s["IP"]
                    first_year_whip = s["WHIP"]
                    break

        player_type = "hitter" if is_hitter else "pitcher" if is_pitcher else "unknown"

        foreign_players.append({
            "npb_name": name,
            "origin_country_ja": origin,
            "origin_country_en": country_en,
            "likely_origin_league": likely_league,
            "npb_first_year": npb_first_year or "",
            "first_team": first_roster_team or profile.get("team", ""),
            "player_type": player_type,
            "position": profile.get("position", ""),
            "english_name": "",  # To be filled manually
            "mlb_id": "",  # To be filled manually
            # First-year NPB stats
            "npb_first_year_PA": first_year_pa,
            "npb_first_year_AVG": first_year_avg,
            "npb_first_year_OPS": first_year_ops,
            "npb_first_year_wOBA": first_year_woba,
            "npb_first_year_ERA": first_year_era,
            "npb_first_year_IP": first_year_ip,
            "npb_first_year_WHIP": first_year_whip,
        })

    return foreign_players


def write_candidates_csv(players: list[dict]) -> Path:
    """Write foreign player candidates to CSV."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "foreign_players_candidates.csv"

    fieldnames = [
        "npb_name", "origin_country_ja", "origin_country_en",
        "likely_origin_league", "npb_first_year", "first_team",
        "player_type", "position", "english_name", "mlb_id",
        "npb_first_year_PA", "npb_first_year_AVG", "npb_first_year_OPS",
        "npb_first_year_wOBA", "npb_first_year_ERA", "npb_first_year_IP",
        "npb_first_year_WHIP",
    ]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(players)

    return out_path


def print_summary(players: list[dict]) -> None:
    """Print summary statistics."""
    print(f"\nTotal foreign players found: {len(players)}")

    by_country = defaultdict(int)
    by_league = defaultdict(int)
    by_type = defaultdict(int)

    for p in players:
        by_country[p["origin_country_en"]] += 1
        by_league[p["likely_origin_league"]] += 1
        by_type[p["player_type"]] += 1

    print("\nBy country:")
    for country, count in sorted(by_country.items(), key=lambda x: -x[1]):
        print(f"  {country}: {count}")

    print("\nBy likely origin league:")
    for league, count in sorted(by_league.items(), key=lambda x: -x[1]):
        print(f"  {league}: {count}")

    print("\nBy player type:")
    for ptype, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {ptype}: {count}")

    # Count players with sufficient first-year data
    hitters_with_pa = sum(
        1 for p in players
        if p["player_type"] == "hitter" and p["npb_first_year_PA"]
        and int(p["npb_first_year_PA"]) >= 100
    )
    pitchers_with_ip = sum(
        1 for p in players
        if p["player_type"] == "pitcher" and p["npb_first_year_IP"]
        and float(p["npb_first_year_IP"]) >= 30
    )
    print(f"\nHitters with PA >= 100 in first year: {hitters_with_pa}")
    print(f"Pitchers with IP >= 30 in first year: {pitchers_with_ip}")


def main() -> None:
    players = identify_foreign_players()
    out_path = write_candidates_csv(players)
    print(f"Wrote {len(players)} foreign player candidates to {out_path}")
    print_summary(players)


if __name__ == "__main__":
    main()
