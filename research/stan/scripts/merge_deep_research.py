"""Merge Deep Research results into foreign_players_candidates.csv"""
import csv
import re
import io
from pathlib import Path

DR_DIR = Path(r"C:\Users\fw_ya\Desktop\Deepリサーチ\結果")
CANDIDATES = Path(r"C:\Users\fw_ya\Desktop\Claude_code\npb-bayes-projection\data\foreign\foreign_players_candidates.csv")

# Normalize origin_league
LEAGUE_MAP = {
    "MLB": "MLB",
    "AAA": "AAA",
    "MiLB": "MiLB",
    "KBO": "KBO",
    "CPBL": "CPBL",
    "Independent": "Independent",
    "Mexican League": "Independent",
    "Cuba": "Cuba",
    "Amateur": "Amateur",
    "Amateur (該当なし)": "Amateur",
    "NPB Draft (該当なし)": None,  # domestic draft player, not foreign
    "United States": None,  # not a league
}

# Country name normalization
COUNTRY_MAP = {
    "USA": "USA",
    "United States": "USA",
    "U.S. Virgin Islands": "USA",
    "South Korea": "South Korea",
    "Dominican Republic": "Dominican Republic",
    "Venezuela": "Venezuela",
    "Cuba": "Cuba",
    "Mexico": "Mexico",
    "Canada": "Canada",
    "Australia": "Australia",
    "Netherlands": "Netherlands",
    "Italy": "Italy",
    "Taiwan": "Taiwan",
    "Puerto Rico": "Puerto Rico",
    "Brazil": "Brazil",
    "Panama": "Panama",
    "Curaçao": "Curaçao",
    "South Africa": "South Africa",
    "Lithuania": "Lithuania",
    "Czech Republic": "Czech Republic",
    "Belgium": "Belgium",
    "Pakistan": "Pakistan",
    "Colombia": "Colombia",
}

COUNTRY_JA = {
    "USA": "アメリカ",
    "Dominican Republic": "ドミニカ共和国",
    "Venezuela": "ベネズエラ",
    "Cuba": "キューバ",
    "Mexico": "メキシコ",
    "Canada": "カナダ",
    "Australia": "オーストラリア",
    "Netherlands": "オランダ",
    "Italy": "イタリア",
    "Taiwan": "台湾",
    "South Korea": "韓国",
    "Puerto Rico": "プエルトリコ",
    "Brazil": "ブラジル",
    "Panama": "パナマ",
    "Curaçao": "キュラソー",
    "South Africa": "南アフリカ",
    "Lithuania": "リトアニア",
    "Czech Republic": "チェコ",
    "Belgium": "ベルギー",
    "Pakistan": "パキスタン",
    "Colombia": "コロンビア",
}


def parse_csv_blocks(text):
    """Extract CSV data from markdown code blocks (```csv or plain ```)."""
    results = {}
    pattern = r"```(?:csv)?\n(.*?)```"
    blocks = re.findall(pattern, text, re.DOTALL)
    for block in blocks:
        reader = csv.DictReader(io.StringIO(block.strip()))
        for row in reader:
            name = row.get("npb_name", "").strip()
            if not name:
                continue
            en = row.get("english_name", "").strip()
            league = row.get("origin_league", "").strip()
            country = row.get("origin_country", "").strip()
            # Skip invalid english_name (e.g. "（該当選手を特定できず）")
            if en and league and "該当" not in en:
                results[name] = {
                    "english_name": en,
                    "origin_league": league,
                    "origin_country": country,
                }
    return results


def parse_table_rows(text):
    """Extract data from markdown table format (used in ⑤.md)."""
    results = {}
    # Match table rows: | # | name | team | english_name | origin_league | origin_country |
    pattern = r"\|\s*\d+\s*\|\s*([^|]+?)\s*\|\s*[^|]+?\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|"
    for m in re.finditer(pattern, text):
        name = m.group(1).strip()
        en = m.group(2).strip()
        league = m.group(3).strip()
        country = m.group(4).strip()
        if name and en and league and not name.startswith("#"):
            results[name] = {
                "english_name": en,
                "origin_league": league,
                "origin_country": country,
            }
    return results


def normalize_league(league):
    """Normalize origin_league to standard categories."""
    return LEAGUE_MAP.get(league, league)


def normalize_country(country):
    """Normalize country name."""
    return COUNTRY_MAP.get(country, country)


def main():
    # Parse all Deep Research files
    dr_data = {}
    for md_file in sorted(DR_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        csv_results = parse_csv_blocks(text)
        table_results = parse_table_rows(text)
        # CSV blocks take priority
        for name, data in csv_results.items():
            if name not in dr_data:
                dr_data[name] = data
        for name, data in table_results.items():
            if name not in dr_data:
                dr_data[name] = data

    print(f"Deep Research total entries: {len(dr_data)}")

    # Normalize
    for name, data in dr_data.items():
        data["origin_league"] = normalize_league(data["origin_league"])
        data["origin_country"] = normalize_country(data["origin_country"])

    # League distribution
    leagues = {}
    for v in dr_data.values():
        l = v["origin_league"]
        leagues[l] = leagues.get(l, 0) + 1
    print(f"League distribution: {leagues}")

    # Country distribution
    countries = {}
    for v in dr_data.values():
        c = v["origin_country"]
        countries[c] = countries.get(c, 0) + 1
    print(f"Country distribution (top 10): {dict(sorted(countries.items(), key=lambda x: -x[1])[:10])}")

    # Read current candidates
    with open(CANDIDATES, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    print(f"\nCandidates total: {len(rows)}")
    unknown_count = sum(1 for r in rows if r.get("likely_origin_league") == "Unknown")
    print(f"Currently Unknown: {unknown_count}")

    # Merge
    matched = 0
    updated = 0
    conflicts = []
    unmatched_dr = set(dr_data.keys())

    for row in rows:
        name = row["npb_name"]
        if name in dr_data:
            dr = dr_data[name]
            unmatched_dr.discard(name)
            matched += 1

            # Update english_name if empty
            if not row.get("english_name") and dr["english_name"]:
                row["english_name"] = dr["english_name"]
                updated += 1

            # Update origin_league — DR always wins (including conflicts)
            if dr["origin_league"]:
                row["likely_origin_league"] = dr["origin_league"]

            # Update origin_country if empty
            if not row.get("origin_country_en") and dr["origin_country"]:
                row["origin_country_en"] = dr["origin_country"]
                row["origin_country_ja"] = COUNTRY_JA.get(dr["origin_country"], "")

    print(f"\nMatched: {matched}")
    print(f"Updated english_name: {updated}")

    remaining_unknown = sum(1 for r in rows if r.get("likely_origin_league") == "Unknown")
    print(f"Remaining Unknown: {remaining_unknown}")

    if conflicts:
        print(f"\nConflicts ({len(conflicts)}):")
        for c in conflicts:
            print(c)

    if unmatched_dr:
        print(f"\nDR entries not in candidates ({len(unmatched_dr)}):")
        for name in sorted(unmatched_dr):
            print(f"  {name}")

    # Write updated CSV
    output = CANDIDATES
    with open(output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWritten to: {output}")


if __name__ == "__main__":
    main()
