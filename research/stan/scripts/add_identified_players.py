"""One-time patch: add manually-identified players from needs_review Step B.

Source: Deep Research report (⑨.md) — all 22 multiple-candidate cases resolved.
Cases skipped (same player, team change — already in master):
  - エスコバー 日本ハム 2017 → Edwin Escobar
  - スアレス   阪神    2020 → Robert Suárez
  - マルティネス 中日   2023 → Raidel Martínez
  - ロドリゲス  阪神   2022 → Aderlin Rodriguez (same as Orix 2020 entry)
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MASTER_PATH = ROOT / "data" / "foreign" / "foreign_players_master.csv"

MASTER_FIELDNAMES = [
    "npb_name", "english_name", "origin_league", "origin_country",
    "npb_first_year", "first_team", "player_type", "position", "mlb_id",
    "npb_first_year_PA", "npb_first_year_AVG", "npb_first_year_OPS",
    "npb_first_year_wOBA", "npb_first_year_ERA", "npb_first_year_IP",
    "npb_first_year_WHIP",
]

# fmt: off
NEW_ENTRIES = [
    # ── エスコバー ───────────────────────────────────────────────
    {"npb_name": "エスコバー",    "english_name": "Alcides Escobar",
     "origin_league": "MLB",     "origin_country": "Venezuela",
     "npb_first_year": "2020",   "first_team": "ヤクルト",
     "player_type": "hitter",
     "npb_first_year_PA": "402", "npb_first_year_wOBA": "0.294205670845494"},

    # ── ガルシア ─────────────────────────────────────────────────
    {"npb_name": "ガルシア",      "english_name": "Anthony Garcia",
     "origin_league": "MiLB",    "origin_country": "Dominican Republic",
     "npb_first_year": "2024",   "first_team": "西武",
     "player_type": "hitter",
     "npb_first_year_PA": "69",  "npb_first_year_wOBA": "0.24047309099677375"},

    # ── サンタナ ─────────────────────────────────────────────────
    {"npb_name": "サンタナ",      "english_name": "Domingo Santana",
     "origin_league": "MLB",     "origin_country": "Dominican Republic",
     "npb_first_year": "2021",   "first_team": "ヤクルト",
     "player_type": "hitter",
     "npb_first_year_PA": "418", "npb_first_year_wOBA": "0.39166705242033034"},

    # ── ジャクソン ───────────────────────────────────────────────
    {"npb_name": "ジャクソン",    "english_name": "Andre Jackson",
     "origin_league": "MLB",     "origin_country": "USA",
     "npb_first_year": "2024",   "first_team": "DeNA",
     "player_type": "pitcher",
     "npb_first_year_ERA": "",   "npb_first_year_IP": ""},

    # ── ジョンソン ───────────────────────────────────────────────
    {"npb_name": "ジョンソン",    "english_name": "Pierce Johnson",
     "origin_league": "MLB",     "origin_country": "USA",
     "npb_first_year": "2019",   "first_team": "阪神",
     "player_type": "pitcher",
     "npb_first_year_ERA": "1.38", "npb_first_year_IP": "58.2"},

    # ── スアレス ─────────────────────────────────────────────────
    {"npb_name": "スアレス",      "english_name": "Albert Suárez",
     "origin_league": "MLB",     "origin_country": "Venezuela",
     "npb_first_year": "2019",   "first_team": "ヤクルト",
     "player_type": "pitcher",
     "npb_first_year_ERA": "1.53", "npb_first_year_IP": "17.2"},

    # ── ヘルナンデス ─────────────────────────────────────────────
    {"npb_name": "ヘルナンデス",  "english_name": "Elier Hernández",
     "origin_league": "MLB",     "origin_country": "Dominican Republic",
     "npb_first_year": "2024",   "first_team": "巨人",
     "player_type": "hitter",
     "npb_first_year_PA": "240", "npb_first_year_wOBA": "0.3674386106785094"},

    {"npb_name": "ヘルナンデス",  "english_name": "Ramon Hernandez",
     "origin_league": "",        "origin_country": "",
     "npb_first_year": "2025",   "first_team": "阪神",
     "player_type": "hitter",
     "npb_first_year_PA": "101", "npb_first_year_wOBA": "0.2534626503593878"},

    # ── ペレス ───────────────────────────────────────────────────
    {"npb_name": "ペレス",        "english_name": "Luis Perez",
     "origin_league": "",        "origin_country": "",
     "npb_first_year": "2016",   "first_team": "ヤクルト",
     "player_type": "pitcher",
     "npb_first_year_ERA": "8.02", "npb_first_year_IP": "21.1"},

    {"npb_name": "ペレス",        "english_name": "Felix Perez",
     "origin_league": "Cuba",    "origin_country": "Cuba",
     "npb_first_year": "2016",   "first_team": "楽天",
     "player_type": "hitter",
     "npb_first_year_PA": "93",  "npb_first_year_wOBA": "0.32303310008304487"},

    # ── ペーニャ ─────────────────────────────────────────────────
    {"npb_name": "ペーニャ",      "english_name": "Wily Mo Peña",
     "origin_league": "MLB",     "origin_country": "Dominican Republic",
     "npb_first_year": "2017",   "first_team": "ロッテ",
     "player_type": "hitter",
     "npb_first_year_PA": "252", "npb_first_year_wOBA": "0.37462773539641414"},

    {"npb_name": "ペーニャ",      "english_name": "Ramiro Peña",
     "origin_league": "MLB",     "origin_country": "Mexico",
     "npb_first_year": "2017",   "first_team": "広島",
     "player_type": "hitter",
     "npb_first_year_PA": "39",  "npb_first_year_wOBA": "0.23195688601492806"},

    # ── マルテ ───────────────────────────────────────────────────
    {"npb_name": "マルテ",        "english_name": "Yunior Marte",
     "origin_league": "MLB",     "origin_country": "Dominican Republic",
     "npb_first_year": "2025",   "first_team": "中日",
     "player_type": "pitcher",
     "npb_first_year_ERA": "1.95", "npb_first_year_IP": "32.1"},

    # ── マルティネス ─────────────────────────────────────────────
    {"npb_name": "マルティネス",  "english_name": "Nick Martinez",
     "origin_league": "MLB",     "origin_country": "USA",
     "npb_first_year": "2018",   "first_team": "日本ハム",
     "player_type": "pitcher",
     "npb_first_year_ERA": "3.51", "npb_first_year_IP": "161.2"},

    # ── マーティン ───────────────────────────────────────────────
    {"npb_name": "マーティン",    "english_name": "Chris Martin",
     "origin_league": "MLB",     "origin_country": "USA",
     "npb_first_year": "2016",   "first_team": "日本ハム",
     "player_type": "pitcher",
     "npb_first_year_ERA": "1.07", "npb_first_year_IP": "50.2"},

    {"npb_name": "マーティン",    "english_name": "Leonys Martín",
     "origin_league": "MLB",     "origin_country": "Cuba",
     "npb_first_year": "2019",   "first_team": "ロッテ",
     "player_type": "hitter",
     "npb_first_year_PA": "228", "npb_first_year_wOBA": "0.3733436290256157"},

    # ── ロドリゲス ───────────────────────────────────────────────
    {"npb_name": "ロドリゲス",    "english_name": "Bryan Rodriguez",
     "origin_league": "MiLB",    "origin_country": "Dominican Republic",
     "npb_first_year": "2018",   "first_team": "日本ハム",
     "player_type": "pitcher",
     "npb_first_year_ERA": "5.26", "npb_first_year_IP": "37.2"},

    {"npb_name": "ロドリゲス",    "english_name": "Aderlin Rodriguez",
     "origin_league": "MiLB",    "origin_country": "Dominican Republic",
     "npb_first_year": "2020",   "first_team": "オリックス",
     "player_type": "hitter",
     "npb_first_year_PA": "211", "npb_first_year_wOBA": "0.29363512047632234"},

    {"npb_name": "ロドリゲス",    "english_name": "Elvin Rodriguez",
     "origin_league": "MLB",     "origin_country": "Dominican Republic",
     "npb_first_year": "2023",   "first_team": "ヤクルト",
     "player_type": "pitcher",
     "npb_first_year_ERA": "4.09", "npb_first_year_IP": "33.0"},

    # ── ＤＪ．ジョンソン ─────────────────────────────────────────
    {"npb_name": "ＤＪ．ジョンソン", "english_name": "DJ Johnson",
     "origin_league": "MLB",     "origin_country": "USA",
     "npb_first_year": "2020",   "first_team": "広島",
     "player_type": "pitcher",
     "npb_first_year_ERA": "4.61", "npb_first_year_IP": "13.2"},
]
# fmt: on


def main() -> None:
    # Read current master
    with open(MASTER_PATH, encoding="utf-8-sig") as f:
        master = list(csv.DictReader(f))
    print(f"Master before: {len(master)} entries")

    # Check for duplicates (same npb_name + first_team)
    existing_keys = {(r["npb_name"], r["first_team"]) for r in master}
    added = []
    skipped = []
    for entry in NEW_ENTRIES:
        key = (entry["npb_name"], entry["first_team"])
        if key in existing_keys:
            skipped.append(f"{entry['npb_name']} {entry['first_team']} ({entry['english_name']})")
        else:
            # Fill missing fields with empty string
            full = {f: "" for f in MASTER_FIELDNAMES}
            full.update(entry)
            master.append(full)
            existing_keys.add(key)
            added.append(f"{entry['npb_name']} {entry['first_team']} → {entry['english_name']}")

    master.sort(key=lambda r: r["npb_name"])

    with open(MASTER_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(master)

    print(f"\nAdded ({len(added)}):")
    for s in added:
        print(f"  + {s}")
    if skipped:
        print(f"\nSkipped (already in master) ({len(skipped)}):")
        for s in skipped:
            print(f"  = {s}")
    print(f"\nMaster after: {len(master)} entries")


if __name__ == "__main__":
    main()
