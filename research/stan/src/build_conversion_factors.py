"""Build league-to-NPB conversion factors for foreign players.

Reads:
- data/foreign/foreign_players_master.csv (manually curated mapping)
- data/foreign/foreign_prev_stats.csv (previous league stats, fetched separately)
- data/raw/npb_sabermetrics_2015_2025.csv (NPB wOBA data)
- data/raw/npb_pitchers_2015_2025.csv (NPB pitcher data)

Output:
- data/foreign/conversion_factors.csv (league-level conversion factors with 95% CI)
- data/foreign/player_conversion_details.csv (per-player ratios for inspection)

Usage:
    python src/build_conversion_factors.py

Designed to run in GitHub Actions / Codespaces (not locally).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
FOREIGN_DIR = DATA_DIR / "foreign"

# Minimum thresholds for meaningful stats
MIN_PA_HITTER = 100
MIN_IP_PITCHER = 30.0


def load_master() -> list[dict]:
    """Load the manually curated foreign player master list."""
    path = FOREIGN_DIR / "foreign_players_master.csv"
    if not path.exists():
        print(f"WARNING: {path} not found. Run identify_foreign_players.py first,")
        print("then manually curate the output into foreign_players_master.csv.")
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_prev_stats() -> dict[str, dict]:
    """Load previous-league stats keyed by english_name.

    Expected columns: english_name, origin_league, season,
                      PA, AVG, OBP, SLG, OPS, wOBA, HR,
                      IP, ERA, FIP, K_pct, BB_pct, WHIP
    """
    path = FOREIGN_DIR / "foreign_prev_stats.csv"
    if not path.exists():
        print(f"WARNING: {path} not found.")
        print("Fetch previous-league stats using pybaseball or manual collection.")
        return {}
    stats: dict[str, dict] = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = row["english_name"].strip()
            stats[name] = row
    return stats


def load_npb_woba() -> dict[tuple[str, int], float]:
    """Load NPB wOBA data keyed by (normalized_name, year)."""
    woba_data = {}
    path = RAW_DIR / "npb_sabermetrics_2015_2025.csv"
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = row["player"].replace("\u3000", "").replace(" ", "").strip()
            year = int(row["year"])
            try:
                woba = float(row["wOBA"])
                pa = int(row["PA"])
            except (ValueError, KeyError):
                continue
            if pa >= MIN_PA_HITTER:
                woba_data[(name, year)] = woba
    return woba_data


def load_npb_pitching() -> dict[tuple[str, int], dict]:
    """Load NPB pitching stats keyed by (normalized_name, year)."""
    pitching = {}
    path = RAW_DIR / "npb_pitchers_2015_2025.csv"
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = row["player"].replace("\u3000", "").replace(" ", "").strip()
            year = int(row["year"])
            try:
                ip = float(row["IP"])
                era = float(row["ERA"])
            except (ValueError, KeyError):
                continue
            if ip >= MIN_IP_PITCHER:
                pitching[(name, year)] = {
                    "ERA": era,
                    "IP": ip,
                    "WHIP": float(row.get("WHIP", 0)),
                    "DIPS": row.get("DIPS", ""),
                }
    return pitching


def compute_conversion_factors(
    master: list[dict],
    prev_stats: dict[str, dict],
    npb_woba: dict[tuple[str, int], float],
    npb_pitching: dict[tuple[str, int], dict],
) -> tuple[list[dict], list[dict]]:
    """Compute per-player ratios and aggregate conversion factors by league.

    Returns:
        (player_details, league_factors)
    """
    player_details = []
    league_ratios: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for player in master:
        npb_name = player["npb_name"].replace("\u3000", "").replace(" ", "").strip()
        english_name = player.get("english_name", "").strip()
        origin_league = player.get("origin_league", "").strip()
        player_type = player.get("player_type", "").strip()

        if not english_name or not origin_league:
            continue

        try:
            first_year = int(player["npb_first_year"])
        except (ValueError, KeyError):
            continue

        prev = prev_stats.get(english_name, {})
        if not prev:
            continue

        detail = {
            "npb_name": npb_name,
            "english_name": english_name,
            "origin_league": origin_league,
            "npb_first_year": first_year,
            "player_type": player_type,
            "prev_K_pct": prev.get("K_pct", ""),
            "prev_BB_pct": prev.get("BB_pct", ""),
        }

        # Hitter: wOBA ratio
        if player_type == "hitter":
            try:
                prev_woba = float(prev.get("wOBA", 0))
            except (ValueError, TypeError):
                prev_woba = 0

            npb_first_woba = npb_woba.get((npb_name, first_year))

            if prev_woba > 0 and npb_first_woba is not None:
                ratio = npb_first_woba / prev_woba
                detail["prev_wOBA"] = f"{prev_woba:.4f}"
                detail["npb_first_wOBA"] = f"{npb_first_woba:.4f}"
                detail["wOBA_ratio"] = f"{ratio:.4f}"
                league_ratios[origin_league]["wOBA"].append(ratio)
            else:
                detail["prev_wOBA"] = prev.get("wOBA", "")
                detail["npb_first_wOBA"] = ""
                detail["wOBA_ratio"] = ""

        # Pitcher: ERA ratio
        if player_type == "pitcher":
            try:
                prev_era = float(prev.get("ERA", 0))
            except (ValueError, TypeError):
                prev_era = 0

            npb_first_pitch = npb_pitching.get((npb_name, first_year))

            if prev_era > 0 and npb_first_pitch is not None:
                npb_era = npb_first_pitch["ERA"]
                # For ERA, higher = worse, so ratio > 1 means worse in NPB
                ratio = npb_era / prev_era
                detail["prev_ERA"] = f"{prev_era:.2f}"
                detail["npb_first_ERA"] = f"{npb_era:.2f}"
                detail["ERA_ratio"] = f"{ratio:.4f}"
                league_ratios[origin_league]["ERA"].append(ratio)

                # FIP if available
                try:
                    prev_fip = float(prev.get("FIP", 0))
                except (ValueError, TypeError):
                    prev_fip = 0
                npb_dips = npb_first_pitch.get("DIPS", "")
                if prev_fip > 0 and npb_dips:
                    try:
                        npb_fip = float(npb_dips)
                        fip_ratio = npb_fip / prev_fip
                        detail["prev_FIP"] = f"{prev_fip:.2f}"
                        detail["npb_first_FIP"] = f"{npb_fip:.2f}"
                        detail["FIP_ratio"] = f"{fip_ratio:.4f}"
                        league_ratios[origin_league]["FIP"].append(fip_ratio)
                    except ValueError:
                        pass
                if "prev_FIP" not in detail:
                    detail["prev_FIP"] = prev.get("FIP", "")
            else:
                detail["prev_ERA"] = prev.get("ERA", "")
                detail["npb_first_ERA"] = ""
                detail["ERA_ratio"] = ""
                detail["prev_FIP"] = prev.get("FIP", "")

        player_details.append(detail)

    # Aggregate by league
    league_factors = []
    for league in sorted(league_ratios.keys()):
        factor = {"origin_league": league}
        for metric in ["wOBA", "ERA", "FIP"]:
            ratios = league_ratios[league].get(metric, [])
            if len(ratios) >= 3:
                arr = np.array(ratios)
                factor[f"{metric}_n"] = len(ratios)
                factor[f"{metric}_median"] = f"{np.median(arr):.4f}"
                factor[f"{metric}_mean"] = f"{np.mean(arr):.4f}"
                factor[f"{metric}_std"] = f"{np.std(arr, ddof=1):.4f}"

                # Bootstrap 95% CI
                rng = np.random.default_rng(42)
                boot_medians = []
                for _ in range(10000):
                    sample = rng.choice(arr, size=len(arr), replace=True)
                    boot_medians.append(np.median(sample))
                boot_medians = np.array(boot_medians)
                ci_low = np.percentile(boot_medians, 2.5)
                ci_high = np.percentile(boot_medians, 97.5)
                factor[f"{metric}_ci_low"] = f"{ci_low:.4f}"
                factor[f"{metric}_ci_high"] = f"{ci_high:.4f}"
            else:
                factor[f"{metric}_n"] = len(ratios)
                factor[f"{metric}_median"] = f"{np.median(ratios):.4f}" if ratios else ""
                factor[f"{metric}_mean"] = ""
                factor[f"{metric}_std"] = ""
                factor[f"{metric}_ci_low"] = ""
                factor[f"{metric}_ci_high"] = ""
                if ratios:
                    factor[f"{metric}_note"] = "too few samples for CI"

        league_factors.append(factor)

    return player_details, league_factors


def write_outputs(
    player_details: list[dict], league_factors: list[dict]
) -> None:
    """Write conversion factor outputs to CSV."""
    FOREIGN_DIR.mkdir(parents=True, exist_ok=True)

    # Player-level details
    if player_details:
        detail_path = FOREIGN_DIR / "player_conversion_details.csv"
        all_keys: list[str] = []
        for d in player_details:
            for k in d:
                if k not in all_keys:
                    all_keys.append(k)
        with open(detail_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(player_details)
        print(f"Wrote {len(player_details)} player details to {detail_path}")

    # League-level factors
    if league_factors:
        factor_path = FOREIGN_DIR / "conversion_factors.csv"
        all_keys = []
        for d in league_factors:
            for k in d:
                if k not in all_keys:
                    all_keys.append(k)
        with open(factor_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(league_factors)
        print(f"Wrote {len(league_factors)} league factors to {factor_path}")


def print_summary(league_factors: list[dict]) -> None:
    """Print conversion factor summary."""
    print("\n=== Conversion Factors Summary ===\n")
    for lf in league_factors:
        league = lf["origin_league"]
        print(f"--- {league} ---")
        for metric in ["wOBA", "ERA", "FIP"]:
            n = lf.get(f"{metric}_n", 0)
            if n and int(n) > 0:
                median = lf.get(f"{metric}_median", "N/A")
                ci_low = lf.get(f"{metric}_ci_low", "N/A")
                ci_high = lf.get(f"{metric}_ci_high", "N/A")
                print(f"  {metric}: median={median}, 95% CI=[{ci_low}, {ci_high}], n={n}")
        print()


def run_backtest(
    master: list[dict],
    prev_stats: dict[str, dict],
    npb_woba: dict[tuple[str, int], float],
    npb_pitching: dict[tuple[str, int], dict],
    league_factors: list[dict],
    test_years: range = range(2020, 2026),
) -> None:
    """Backtest: compare conversion-factor prediction vs wRAA=0 baseline.

    For each foreign player in test_years:
      - Baseline: predict league-average wOBA (= NPB league avg for that year)
      - Model: prev_league_wOBA * conversion_factor
      - Compare MAE
    """
    # Build factor lookup
    factor_lookup: dict[str, float] = {}
    for lf in league_factors:
        league = lf["origin_league"]
        median = lf.get("wOBA_median", "")
        if median:
            factor_lookup[league] = float(median)

    if not factor_lookup:
        print("\nBacktest skipped: no wOBA conversion factors available.")
        return

    # Compute NPB league-average wOBA per year
    yearly_woba: dict[int, list[float]] = defaultdict(list)
    for (_, year), woba in npb_woba.items():
        yearly_woba[year].append(woba)
    league_avg_woba = {y: np.mean(vals) for y, vals in yearly_woba.items()}

    baseline_errors = []
    model_errors = []
    test_count = 0

    for player in master:
        npb_name = player["npb_name"].replace("\u3000", "").replace(" ", "").strip()
        english_name = player.get("english_name", "").strip()
        origin_league = player.get("origin_league", "").strip()
        player_type = player.get("player_type", "").strip()

        if player_type != "hitter" or not english_name:
            continue

        try:
            first_year = int(player["npb_first_year"])
        except (ValueError, KeyError):
            continue

        if first_year not in test_years:
            continue

        prev = prev_stats.get(english_name, {})
        if not prev:
            continue

        try:
            prev_woba = float(prev.get("wOBA", 0))
        except (ValueError, TypeError):
            continue

        if prev_woba <= 0:
            continue

        npb_actual = npb_woba.get((npb_name, first_year))
        if npb_actual is None:
            continue

        avg_woba = league_avg_woba.get(first_year)
        if avg_woba is None:
            continue

        factor = factor_lookup.get(origin_league)
        if factor is None:
            continue

        # Baseline: league average
        baseline_pred = avg_woba
        # Model: prev_wOBA * conversion_factor
        model_pred = prev_woba * factor

        baseline_errors.append(abs(npb_actual - baseline_pred))
        model_errors.append(abs(npb_actual - model_pred))
        test_count += 1

    if test_count == 0:
        print("\nBacktest: no eligible players found in test years.")
        return

    baseline_mae = np.mean(baseline_errors)
    model_mae = np.mean(model_errors)
    improvement = (baseline_mae - model_mae) / baseline_mae * 100

    print(f"\n=== Backtest ({min(test_years)}-{max(test_years)}) ===")
    print(f"  Players tested: {test_count}")
    print(f"  Baseline MAE (league avg): {baseline_mae:.4f}")
    print(f"  Model MAE (conversion factor): {model_mae:.4f}")
    print(f"  Improvement: {improvement:+.1f}%")


def main() -> None:
    master = load_master()
    if not master:
        print("No master data. Exiting.")
        print("Next steps:")
        print("  1. Run: python src/identify_foreign_players.py")
        print("  2. Curate output into data/foreign/foreign_players_master.csv")
        print("  3. Collect previous-league stats into data/foreign/foreign_prev_stats.csv")
        print("  4. Re-run: python src/build_conversion_factors.py")
        return

    prev_stats = load_prev_stats()
    npb_woba = load_npb_woba()
    npb_pitching = load_npb_pitching()

    player_details, league_factors = compute_conversion_factors(
        master, prev_stats, npb_woba, npb_pitching
    )

    write_outputs(player_details, league_factors)
    print_summary(league_factors)

    # Backtest
    run_backtest(master, prev_stats, npb_woba, npb_pitching, league_factors)


if __name__ == "__main__":
    main()
