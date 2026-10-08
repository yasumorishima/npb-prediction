"""Analyze uncovered PA/IP by player category.

For each year (2018-2025), identifies players in actual data but NOT in
Marcel+Stan projections, and categorizes them:
  1. Foreign first-year: in foreign_players_master.csv, year == npb_first_year
  2. Rookie: first appearance in 1軍 data (no prior year in sabermetrics/pitchers)
  3. Prior-year below threshold: appeared in some prior year but PA < 50 / IP < 20
  4. Other — sub-categorized:
     a. low_pa_current: current year PA < MIN_PA (Marcel & _add_uncovered skip them)
     b. no_birthday: PA >= MIN_PA but dropped by add_age_from_peak (no birthday)
     c. gap_return: no data in prior 3 years (career gap / long 2軍 stint)
     d. truly_unknown: none of the above

Output: stdout table + data/projections/coverage_gap_analysis.json
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stan_jpn_model import MIN_IP, MIN_PA, build_dataset, compute_fip_column, ip_to_decimal, load_birthday_df, normalize_name

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
FOREIGN_DIR = ROOT / "data" / "foreign"
OUT_DIR = ROOT / "data" / "projections"

JPN_YEARS = list(range(2018, 2026))


def _has_birthday(player: str, bday_map: dict) -> bool:
    norm = normalize_name(player)
    bday = bday_map.get(norm)
    return bday is not None and not pd.isna(bday)


def _sub_categorize_other_h(player, pa, prior_3yr, saber_df, yr, bday_map) -> str:
    """Sub-categorize a hitter that fell into the old 'other' bucket."""
    if pa < MIN_PA:
        return "low_pa_current"
    if not _has_birthday(player, bday_map):
        return "no_birthday"
    if len(prior_3yr) == 0:
        return "gap_return"
    return "truly_unknown"


def _sub_categorize_other_p(player, ip, prior_3yr, pitchers_df, yr, bday_map) -> str:
    """Sub-categorize a pitcher that fell into the old 'other' bucket."""
    if ip < MIN_IP:
        return "low_ip_current"
    if not _has_birthday(player, bday_map):
        return "no_birthday"
    if len(prior_3yr) == 0:
        return "gap_return"
    return "truly_unknown"


def main():
    # Load data
    saber = pd.read_csv(RAW_DIR / "npb_sabermetrics_2015_2025.csv", encoding="utf-8-sig")
    pitchers_raw = pd.read_csv(RAW_DIR / "npb_pitchers_2015_2025.csv", encoding="utf-8-sig")
    saber = saber.dropna(subset=["wOBA"])
    pitchers_raw = compute_fip_column(pitchers_raw)
    bday_df = load_birthday_df()

    # Foreign players
    foreign_master = pd.read_csv(FOREIGN_DIR / "foreign_players_master.csv", encoding="utf-8")
    foreign_first = {}  # name -> first_year
    for _, row in foreign_master.iterrows():
        foreign_first[row["npb_name"]] = int(row["npb_first_year"])

    # All players' first appearance year in 1軍 data
    first_year_h = saber.groupby("player")["year"].min().to_dict()
    first_year_p = pitchers_raw.groupby("player")["year"].min().to_dict()

    # Birthday lookup for no_birthday sub-category
    bday_map = dict(zip(bday_df["player"].apply(normalize_name), bday_df["birthday"]))

    # Pitchers: add IP_dec
    pitchers_raw["IP_dec"] = pitchers_raw["IP"].apply(ip_to_decimal)

    # Run build_dataset for each year to get projected players
    print("=" * 80)
    print("Coverage Gap Analysis: Uncovered PA/IP by Player Category (2018-2025)")
    print("=" * 80)

    all_results = []

    for yr in JPN_YEARS:
        # Build Marcel projections for this year
        h_proj, p_proj, _ = build_dataset(saber, pitchers_raw, [yr], bday_df)

        proj_h_names = set(h_proj["player"].values) if len(h_proj) > 0 else set()
        proj_p_names = set(p_proj["player"].values) if len(p_proj) > 0 else set()

        # Actual players in this year
        actual_h = saber[(saber["year"] == yr) & (saber["PA"] >= 1)].copy()
        actual_p = pitchers_raw[(pitchers_raw["year"] == yr) & (pitchers_raw["IP_dec"] >= 1)].copy()

        # Missing hitters
        missing_h = actual_h[~actual_h["player"].isin(proj_h_names)].copy()
        # Missing pitchers
        missing_p = actual_p[~actual_p["player"].isin(proj_p_names)].copy()

        # Categorize hitters
        ALL_CATS_H = ["foreign_1st", "rookie", "below_threshold",
                      "low_pa_current", "no_birthday", "gap_return", "truly_unknown"]
        cats_h = {c: 0.0 for c in ALL_CATS_H}
        cats_h_n = {c: 0 for c in ALL_CATS_H}
        cats_h_examples = {c: [] for c in ALL_CATS_H}  # top examples per category
        total_pa_actual = float(actual_h["PA"].sum())
        total_pa_covered = float(actual_h[actual_h["player"].isin(proj_h_names)]["PA"].sum())
        total_pa_missing = float(missing_h["PA"].sum())

        for _, row in missing_h.iterrows():
            pa = float(row["PA"])
            player = row["player"]

            if player in foreign_first and foreign_first[player] == yr:
                cat = "foreign_1st"
            elif first_year_h.get(player, 9999) == yr and yr > 2015:
                cat = "rookie"
            else:
                prior = saber[(saber["player"] == player) & (saber["year"] < yr) & (saber["year"] >= yr - 3)]
                if first_year_h.get(player, 9999) == 2015 and yr == 2018:
                    if len(prior) > 0 and prior["PA"].max() < MIN_PA:
                        cat = "below_threshold"
                    else:
                        cat = _sub_categorize_other_h(player, pa, prior, saber, yr, bday_map)
                elif len(prior) > 0 and prior["PA"].max() < MIN_PA:
                    cat = "below_threshold"
                else:
                    cat = _sub_categorize_other_h(player, pa, prior, saber, yr, bday_map)

            cats_h[cat] += pa
            cats_h_n[cat] += 1
            cats_h_examples[cat].append((player, int(pa)))

        # Categorize pitchers
        ALL_CATS_P = ["foreign_1st", "rookie", "below_threshold",
                      "low_ip_current", "no_birthday", "gap_return", "truly_unknown"]
        cats_p = {c: 0.0 for c in ALL_CATS_P}
        cats_p_n = {c: 0 for c in ALL_CATS_P}
        cats_p_examples = {c: [] for c in ALL_CATS_P}
        total_ip_actual = float(actual_p["IP_dec"].sum())
        total_ip_covered = float(actual_p[actual_p["player"].isin(proj_p_names)]["IP_dec"].sum())
        total_ip_missing = float(missing_p["IP_dec"].sum())

        for _, row in missing_p.iterrows():
            ip = float(row["IP_dec"])
            player = row["player"]

            if player in foreign_first and foreign_first[player] == yr:
                cat = "foreign_1st"
            elif first_year_p.get(player, 9999) == yr and yr > 2015:
                cat = "rookie"
            else:
                prior = pitchers_raw[(pitchers_raw["player"] == player) & (pitchers_raw["year"] < yr) & (pitchers_raw["year"] >= yr - 3)]
                if first_year_p.get(player, 9999) == 2015 and yr == 2018:
                    if len(prior) > 0 and prior["IP_dec"].max() < MIN_IP:
                        cat = "below_threshold"
                    else:
                        cat = _sub_categorize_other_p(player, ip, prior, pitchers_raw, yr, bday_map)
                elif len(prior) > 0 and prior["IP_dec"].max() < MIN_IP:
                    cat = "below_threshold"
                else:
                    cat = _sub_categorize_other_p(player, ip, prior, pitchers_raw, yr, bday_map)

            cats_p[cat] += ip
            cats_p_n[cat] += 1
            cats_p_examples[cat].append((player, round(ip, 1)))

        pa_cov_pct = total_pa_covered / total_pa_actual * 100 if total_pa_actual > 0 else 0
        ip_cov_pct = total_ip_covered / total_ip_actual * 100 if total_ip_actual > 0 else 0

        print(f"\n{'='*80}")
        print(f"  {yr}  PA_cov={pa_cov_pct:.1f}%  IP_cov={ip_cov_pct:.1f}%")
        print(f"{'='*80}")
        print(f"  [Hitters] Total PA={total_pa_actual:.0f}  Covered={total_pa_covered:.0f}  Missing={total_pa_missing:.0f}")
        for cat in ALL_CATS_H:
            if cats_h[cat] == 0:
                continue
            pct = cats_h[cat] / total_pa_actual * 100 if total_pa_actual > 0 else 0
            print(f"    {cat:20s}: {cats_h[cat]:>7.0f} PA  ({pct:>5.1f}%)  n={cats_h_n[cat]}")
            # Show top-3 examples by PA
            top3 = sorted(cats_h_examples[cat], key=lambda x: -x[1])[:3]
            if top3:
                names = ", ".join(f"{n}({v})" for n, v in top3)
                print(f"      e.g. {names}")

        print(f"  [Pitchers] Total IP={total_ip_actual:.1f}  Covered={total_ip_covered:.1f}  Missing={total_ip_missing:.1f}")
        for cat in ALL_CATS_P:
            if cats_p[cat] == 0:
                continue
            pct = cats_p[cat] / total_ip_actual * 100 if total_ip_actual > 0 else 0
            print(f"    {cat:20s}: {cats_p[cat]:>7.1f} IP  ({pct:>5.1f}%)  n={cats_p_n[cat]}")
            top3 = sorted(cats_p_examples[cat], key=lambda x: -x[1])[:3]
            if top3:
                names = ", ".join(f"{n}({v})" for n, v in top3)
                print(f"      e.g. {names}")

        all_results.append({
            "year": yr,
            "pa_cov_pct": round(pa_cov_pct, 1),
            "ip_cov_pct": round(ip_cov_pct, 1),
            "total_pa": round(total_pa_actual),
            "total_ip": round(total_ip_actual, 1),
            "missing_pa": round(total_pa_missing),
            "missing_ip": round(total_ip_missing, 1),
            "hitter_categories": {
                cat: {
                    "pa": round(cats_h[cat]),
                    "pct_of_total": round(cats_h[cat] / total_pa_actual * 100, 1) if total_pa_actual > 0 else 0,
                    "n": cats_h_n[cat],
                    "top_examples": sorted(cats_h_examples[cat], key=lambda x: -x[1])[:5],
                }
                for cat in ALL_CATS_H if cats_h[cat] > 0
            },
            "pitcher_categories": {
                cat: {
                    "ip": round(cats_p[cat], 1),
                    "pct_of_total": round(cats_p[cat] / total_ip_actual * 100, 1) if total_ip_actual > 0 else 0,
                    "n": cats_p_n[cat],
                    "top_examples": sorted(cats_p_examples[cat], key=lambda x: -x[1])[:5],
                }
                for cat in ALL_CATS_P if cats_p[cat] > 0
            },
        })

    # Summary across all years
    print(f"\n{'='*80}")
    print("SUMMARY (2018-2025 Average)")
    print(f"{'='*80}")

    total_missing_pa = sum(r["missing_pa"] for r in all_results)
    total_missing_ip = sum(r["missing_ip"] for r in all_results)
    total_all_pa = sum(r["total_pa"] for r in all_results)
    total_all_ip = sum(r["total_ip"] for r in all_results)

    all_h_cats = sorted({cat for r in all_results for cat in r["hitter_categories"]})
    all_p_cats = sorted({cat for r in all_results for cat in r["pitcher_categories"]})

    sum_cats_h = {cat: sum(r["hitter_categories"].get(cat, {}).get("pa", 0) for r in all_results) for cat in all_h_cats}
    sum_cats_p = {cat: sum(r["pitcher_categories"].get(cat, {}).get("ip", 0) for r in all_results) for cat in all_p_cats}

    print(f"\n  [Hitters] Missing PA breakdown (8-year total):")
    for cat in all_h_cats:
        if sum_cats_h[cat] == 0:
            continue
        pct_of_total = sum_cats_h[cat] / total_all_pa * 100 if total_all_pa > 0 else 0
        pct_of_missing = sum_cats_h[cat] / total_missing_pa * 100 if total_missing_pa > 0 else 0
        print(f"    {cat:20s}: {sum_cats_h[cat]:>8.0f} PA  ({pct_of_total:>5.1f}% of total, {pct_of_missing:>5.1f}% of missing)")

    print(f"\n  [Pitchers] Missing IP breakdown (8-year total):")
    for cat in all_p_cats:
        if sum_cats_p[cat] == 0:
            continue
        pct_of_total = sum_cats_p[cat] / total_all_ip * 100 if total_all_ip > 0 else 0
        pct_of_missing = sum_cats_p[cat] / total_missing_ip * 100 if total_missing_ip > 0 else 0
        print(f"    {cat:20s}: {sum_cats_p[cat]:>8.1f} IP  ({pct_of_total:>5.1f}% of total, {pct_of_missing:>5.1f}% of missing)")

    # Save JSON
    out_path = OUT_DIR / "coverage_gap_analysis.json"
    output = {
        "years": all_results,
        "summary": {
            "hitter_missing_pa_breakdown": {
                cat: {"pa": sum_cats_h[cat], "pct_of_total": round(sum_cats_h[cat] / total_all_pa * 100, 1), "pct_of_missing": round(sum_cats_h[cat] / total_missing_pa * 100, 1)}
                for cat in all_h_cats if sum_cats_h[cat] > 0
            },
            "pitcher_missing_ip_breakdown": {
                cat: {"ip": round(sum_cats_p[cat], 1), "pct_of_total": round(sum_cats_p[cat] / total_all_ip * 100, 1), "pct_of_missing": round(sum_cats_p[cat] / total_missing_ip * 100, 1)}
                for cat in all_p_cats if sum_cats_p[cat] > 0
            },
        },
    }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
