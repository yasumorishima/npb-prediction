"""Foreign player prediction model v5 — Bayesian Mixture of Experts.

Two-stage model:
  Stage 1: Gate function estimates P(bust) per player
  Stage 2: Mixture likelihood
    - Bust component: Normal(mu_bust, sigma_bust)
    - Performance component: Student-t(nu, mu_perf, sigma_perf) with v2 regression

New features over v2:
  - Career total PA/IP (from MLB Stats API)
  - Team-specific foreign player bust rate

Data sources (all relative to repo root):
  - data/foreign/foreign_players_master.csv
  - data/foreign/player_conversion_details.csv
  - data/foreign/foreign_prev_stats.csv
  - data/foreign/career_stats.csv          (new — from fetch_career_stats.py)
  - data/raw/npb_player_birthdays.csv
  - data/raw/npb_players_profile_2024.csv
  - data/raw/npb_sabermetrics_2015_2025.csv
  - data/raw/npb_pitchers_2015_2025.csv

Usage:
  python src/foreign_v5_model.py --draws 2000 --warmup 1000
  python src/foreign_v5_model.py --loo-cv
  python src/foreign_v5_model.py --fit-only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def _log_elapsed(label: str, start: float, budget_min: int = 360):
    elapsed_min = (time.time() - start) / 60
    print(f"  [{label}] elapsed: {elapsed_min:.1f} min / {budget_min} min budget")
    if elapsed_min > budget_min * 0.8:
        print(f"  WARNING: {label} used {elapsed_min:.0f}/{budget_min} min "
              f"({elapsed_min / budget_min * 100:.0f}%) -- timeout risk!")


ROOT = Path(__file__).resolve().parent.parent
DATA_FOREIGN = ROOT / "data" / "foreign"
DATA_RAW = ROOT / "data" / "raw"
DATA_MODEL = ROOT / "data" / "model"
MODELS_DIR = ROOT / "models"

PEAK_AGE = 29
MIN_PA_HITTER = 50
MIN_IP_PITCHER = 20
MIN_PREV_PA = 20
MIN_PREV_IP = 5

LEAGUE_GROUPS = {
    "MLB": "MLB",
    "AAA": "AAA",
    "MiLB": "AAA",
    "KBO": "Other",
    "CPBL": "Other",
    "Cuba": "Other",
    "Independent": "Other",
    "Amateur": "Other",
    "": "Other",
}
LEAGUE_INDEX = {"MLB": 1, "AAA": 2, "Other": 3}

CATCHER_KEYWORDS = {"捕手"}
MIDDLE_INF_KEYWORDS = {"遊撃手", "二塁手", "内野手"}

# Bust thresholds
BUST_WOBA = 0.250
BUST_ERA = 5.0


# ─────────────────────────────────────────────
# Data loading (same as v2 + career stats)
# ─────────────────────────────────────────────

def _norm_name(name: str) -> str:
    return name.replace("\u3000", " ").strip()


def load_master() -> list[dict]:
    with open(DATA_FOREIGN / "foreign_players_master.csv", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_conversion_details() -> dict[str, dict]:
    with open(DATA_FOREIGN / "player_conversion_details.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {_norm_name(r["npb_name"]): r for r in rows}


def load_prev_stats() -> dict[str, dict]:
    with open(DATA_FOREIGN / "foreign_prev_stats.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {_norm_name(r["npb_name"]): r for r in rows}


def load_birthdays() -> dict[str, str]:
    with open(DATA_RAW / "npb_player_birthdays.csv", encoding="utf-8-sig") as f:
        return {_norm_name(r["player"]): r["birthday"] for r in csv.DictReader(f)}


def load_profiles() -> dict[str, str]:
    path = DATA_RAW / "npb_players_profile_2024.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig") as f:
        return {_norm_name(r["選手名"]): r.get("守備", "") for r in csv.DictReader(f)}


def load_npb_hitter_years() -> set[tuple[str, int]]:
    result = set()
    with open(DATA_RAW / "npb_sabermetrics_2015_2025.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                pa = int(r["PA"])
            except (ValueError, KeyError):
                continue
            if pa >= MIN_PA_HITTER:
                result.add((_norm_name(r["player"]), int(r["year"])))
    return result


def load_npb_pitcher_years() -> set[tuple[str, int]]:
    result = set()
    with open(DATA_RAW / "npb_pitchers_2015_2025.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                ip = float(r["IP"])
            except (ValueError, KeyError):
                continue
            if ip >= MIN_IP_PITCHER:
                result.add((_norm_name(r["player"]), int(r["year"])))
    return result


def load_npb_pitcher_ip() -> dict[tuple[str, int], float]:
    result = {}
    with open(DATA_RAW / "npb_pitchers_2015_2025.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                ip = float(r["IP"])
                result[(_norm_name(r["player"]), int(r["year"]))] = ip
            except (ValueError, KeyError):
                continue
    return result


def load_league_averages() -> dict[int, dict]:
    woba_by_year: dict[int, list[float]] = {}
    with open(DATA_RAW / "npb_sabermetrics_2015_2025.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                pa = int(r["PA"])
                woba = float(r["wOBA"])
                year = int(r["year"])
            except (ValueError, KeyError):
                continue
            if pa >= 100:
                woba_by_year.setdefault(year, []).append(woba)

    era_by_year: dict[int, list[float]] = {}
    with open(DATA_RAW / "npb_pitchers_2015_2025.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                ip = float(r["IP"])
                era = float(r["ERA"])
                year = int(r["year"])
            except (ValueError, KeyError):
                continue
            if ip >= 30:
                era_by_year.setdefault(year, []).append(era)

    result = {}
    for y in set(woba_by_year.keys()) | set(era_by_year.keys()):
        result[y] = {
            "lg_woba": float(np.mean(woba_by_year.get(y, [0.310]))),
            "lg_era": float(np.mean(era_by_year.get(y, [3.50]))),
        }
    return result


def load_career_stats() -> dict[str, dict]:
    """Load career_stats.csv → {npb_name: {career_pa, career_ip, ...}}."""
    path = DATA_FOREIGN / "career_stats.csv"
    if not path.exists():
        print("  WARNING: career_stats.csv not found — using prev_pa/ip as fallback")
        return {}
    result = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            name = _norm_name(r["npb_name"])
            career_pa = _safe_float(r.get("career_pa"))
            career_ip = _safe_float(r.get("career_ip"))
            result[name] = {
                "career_pa": career_pa,
                "career_ip": career_ip,
                "match_quality": r.get("match_quality", ""),
            }
    n_pa = sum(1 for v in result.values() if v["career_pa"] is not None)
    n_ip = sum(1 for v in result.values() if v["career_ip"] is not None)
    print(f"  Career stats loaded: {len(result)} entries, {n_pa} with PA, {n_ip} with IP")
    return result


# ─────────────────────────────────────────────
# Team bust rate computation
# ─────────────────────────────────────────────

def compute_team_bust_rates(
    master: list[dict],
    exclude_name: str | None = None,
    exclude_year: int | None = None,
) -> dict[str, float]:
    """Compute bust rate per NPB team from master data.

    For LOO-CV, exclude the held-out player via exclude_name/exclude_year.
    Returns {team_name: bust_rate} where bust_rate is in [0, 1].
    """
    team_counts: dict[str, list[int]] = {}  # team → list of is_bust (0/1)

    for m in master:
        team = m.get("first_team", "").strip()
        if not team:
            continue

        name = _norm_name(m["npb_name"])
        year = int(m["npb_first_year"])

        # Exclude held-out player
        if exclude_name and exclude_year and name == exclude_name and year == exclude_year:
            continue

        ptype = m["player_type"]
        is_bust = 0

        if ptype == "hitter":
            woba = _safe_float(m.get("npb_first_year_wOBA"))
            pa = _safe_float(m.get("npb_first_year_PA"))
            if woba is not None and pa is not None and pa >= MIN_PA_HITTER:
                is_bust = 1 if woba < BUST_WOBA else 0
            else:
                continue  # can't determine bust status
        elif ptype == "pitcher":
            era = _safe_float(m.get("npb_first_year_ERA"))
            ip = _safe_float(m.get("npb_first_year_IP"))
            if era is not None and ip is not None and ip >= MIN_IP_PITCHER:
                is_bust = 1 if era > BUST_ERA else 0
            else:
                continue

        team_counts.setdefault(team, []).append(is_bust)

    # Compute rates with Laplace smoothing (add 1 success + 1 failure)
    result = {}
    for team, busts in team_counts.items():
        n = len(busts)
        n_bust = sum(busts)
        result[team] = (n_bust + 1) / (n + 2)  # Laplace smoothing

    return result


# ─────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────

def _safe_float(val: str | None) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _age_at_year(birthday_str: str, year: int) -> float:
    try:
        bday = datetime.strptime(birthday_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        try:
            bday = datetime.strptime(birthday_str, "%Y/%m/%d")
        except (ValueError, TypeError):
            return PEAK_AGE
    ref = datetime(year, 4, 1)
    return (ref - bday).days / 365.25


def _classify_position(pos_str: str) -> tuple[bool, bool]:
    if not pos_str:
        return False, False
    is_c = any(k in pos_str for k in CATCHER_KEYWORDS)
    is_mi = any(k in pos_str for k in MIDDLE_INF_KEYWORDS)
    return is_c, is_mi


def _league_group(origin_league: str) -> str:
    return LEAGUE_GROUPS.get(origin_league, "Other")


def build_dataset() -> tuple[list[dict], list[dict], list[dict]]:
    """Build datasets for hitters and pitchers with v5 features.

    Returns (hitters, pitchers, master_raw) where master_raw is the raw
    master list for team bust rate computation.
    """
    master = load_master()
    conv = load_conversion_details()
    prev_stats = load_prev_stats()
    birthdays = load_birthdays()
    profiles = load_profiles()
    hitter_years = load_npb_hitter_years()
    pitcher_years = load_npb_pitcher_years()
    pitcher_ip_map = load_npb_pitcher_ip()
    lg_avgs = load_league_averages()
    career = load_career_stats()

    # Compute team bust rates (full data, no exclusion)
    team_rates = compute_team_bust_rates(master)

    hitters = []
    pitchers = []

    for m in master:
        name = _norm_name(m["npb_name"])
        ptype = m["player_type"]
        first_year = int(m["npb_first_year"])
        origin = m.get("origin_league", "")
        lg_group = _league_group(origin)
        lg_idx = LEAGUE_INDEX[lg_group]
        pos_str = m.get("position", "") or profiles.get(name, "")
        team = m.get("first_team", "").strip()

        bday = birthdays.get(name)
        age = _age_at_year(bday, first_year) if bday else None

        c = conv.get(name, {})
        ps = prev_stats.get(name, {})
        cs = career.get(name, {})

        # Career stats (fallback to prev_pa/ip if not available)
        career_pa_raw = cs.get("career_pa")
        career_ip_raw = cs.get("career_ip")

        team_bust_rate = team_rates.get(team, 0.20)  # default to overall rate

        if ptype == "hitter":
            npb_woba = _safe_float(m.get("npb_first_year_wOBA"))
            npb_pa = _safe_float(m.get("npb_first_year_PA"))
            if npb_woba is None or npb_pa is None or npb_pa < MIN_PA_HITTER:
                continue

            prev_woba = _safe_float(c.get("prev_wOBA")) or _safe_float(ps.get("wOBA"))
            prev_K = _safe_float(c.get("prev_K_pct")) or _safe_float(ps.get("K_pct"))
            prev_BB = _safe_float(c.get("prev_BB_pct")) or _safe_float(ps.get("BB_pct"))
            prev_pa = _safe_float(ps.get("PA"))

            if prev_woba is None:
                continue

            # Career PA: use career_stats if available, else prev_pa
            c_pa = career_pa_raw if career_pa_raw is not None else (prev_pa or 200)

            is_c, is_mi = _classify_position(pos_str)
            lg = lg_avgs.get(first_year, {"lg_woba": 0.310})

            hitters.append({
                "npb_name": name,
                "year": first_year,
                "origin_league": origin,
                "league_group": lg_group,
                "league_idx": lg_idx,
                "first_team": team,
                "prev_woba": prev_woba,
                "prev_K_pct": prev_K,
                "prev_BB_pct": prev_BB,
                "prev_pa": prev_pa or 200,
                "career_pa": c_pa,
                "team_bust_rate": team_bust_rate,
                "npb_woba": npb_woba,
                "age": age or PEAK_AGE,
                "age_from_peak": (age or PEAK_AGE) - PEAK_AGE,
                "is_catcher": 1.0 if is_c else 0.0,
                "is_middle_inf": 1.0 if is_mi else 0.0,
                "is_second_year": 0.0,
                "lg_woba": lg["lg_woba"],
            })

            # 2nd year entry
            if (name, first_year + 1) in hitter_years:
                woba_2nd = _get_hitter_woba(name, first_year + 1)
                if woba_2nd is not None:
                    lg2 = lg_avgs.get(first_year + 1, {"lg_woba": 0.310})
                    hitters.append({
                        "npb_name": name,
                        "year": first_year + 1,
                        "origin_league": origin,
                        "league_group": lg_group,
                        "league_idx": lg_idx,
                        "first_team": team,
                        "prev_woba": prev_woba,
                        "prev_K_pct": prev_K,
                        "prev_BB_pct": prev_BB,
                        "prev_pa": prev_pa or 200,
                        "career_pa": c_pa,
                        "team_bust_rate": team_bust_rate,
                        "npb_woba": woba_2nd,
                        "age": (age or PEAK_AGE) + 1,
                        "age_from_peak": (age or PEAK_AGE) + 1 - PEAK_AGE,
                        "is_catcher": 1.0 if is_c else 0.0,
                        "is_middle_inf": 1.0 if is_mi else 0.0,
                        "is_second_year": 1.0,
                        "lg_woba": lg2["lg_woba"],
                    })

        elif ptype == "pitcher":
            npb_era = _safe_float(m.get("npb_first_year_ERA"))
            npb_ip = _safe_float(m.get("npb_first_year_IP"))
            if npb_era is None or npb_ip is None or npb_ip < MIN_IP_PITCHER:
                continue

            prev_era = _safe_float(c.get("prev_ERA")) or _safe_float(ps.get("ERA"))
            prev_fip = _safe_float(c.get("prev_FIP")) or _safe_float(ps.get("FIP"))
            prev_K = _safe_float(c.get("prev_K_pct")) or _safe_float(ps.get("K_pct"))
            prev_BB = _safe_float(c.get("prev_BB_pct")) or _safe_float(ps.get("BB_pct"))
            prev_ip = _safe_float(ps.get("IP"))

            if prev_era is None:
                continue

            c_ip = career_ip_raw if career_ip_raw is not None else (prev_ip or 50)

            is_rel = npb_ip < 50
            lg = lg_avgs.get(first_year, {"lg_era": 3.50})

            pitchers.append({
                "npb_name": name,
                "year": first_year,
                "origin_league": origin,
                "league_group": lg_group,
                "league_idx": lg_idx,
                "first_team": team,
                "prev_era": prev_era,
                "prev_fip": prev_fip,
                "prev_K_pct": prev_K,
                "prev_BB_pct": prev_BB,
                "prev_ip": prev_ip or 50,
                "career_ip": c_ip,
                "team_bust_rate": team_bust_rate,
                "npb_era": npb_era,
                "age": age or PEAK_AGE,
                "age_from_peak": (age or PEAK_AGE) - PEAK_AGE,
                "is_reliever": 1.0 if is_rel else 0.0,
                "is_second_year": 0.0,
                "lg_era": lg["lg_era"],
            })

            # 2nd year entry
            if (name, first_year + 1) in pitcher_years:
                era_2nd_ip = pitcher_ip_map.get((name, first_year + 1))
                era_2nd = _get_pitcher_era(name, first_year + 1)
                if era_2nd is not None and era_2nd_ip is not None and era_2nd_ip >= MIN_IP_PITCHER:
                    lg2 = lg_avgs.get(first_year + 1, {"lg_era": 3.50})
                    pitchers.append({
                        "npb_name": name,
                        "year": first_year + 1,
                        "origin_league": origin,
                        "league_group": lg_group,
                        "league_idx": lg_idx,
                        "first_team": team,
                        "prev_era": prev_era,
                        "prev_fip": prev_fip,
                        "prev_K_pct": prev_K,
                        "prev_BB_pct": prev_BB,
                        "prev_ip": prev_ip or 50,
                        "career_ip": c_ip,
                        "team_bust_rate": team_bust_rate,
                        "npb_era": era_2nd,
                        "age": (age or PEAK_AGE) + 1,
                        "age_from_peak": (age or PEAK_AGE) + 1 - PEAK_AGE,
                        "is_reliever": 1.0 if era_2nd_ip < 50 else 0.0,
                        "is_second_year": 1.0,
                        "lg_era": lg2["lg_era"],
                    })

    return hitters, pitchers, master


# Sabermetrics lookups for 2nd year
_saber_cache: dict[tuple[str, int], float] | None = None
_pitcher_era_cache: dict[tuple[str, int], float] | None = None


def _get_hitter_woba(name: str, year: int) -> float | None:
    global _saber_cache
    if _saber_cache is None:
        _saber_cache = {}
        with open(DATA_RAW / "npb_sabermetrics_2015_2025.csv", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    n = _norm_name(r["player"])
                    y = int(r["year"])
                    w = float(r["wOBA"])
                    pa = int(r["PA"])
                    if pa >= MIN_PA_HITTER:
                        _saber_cache[(n, y)] = w
                except (ValueError, KeyError):
                    continue
    return _saber_cache.get((name, year))


def _get_pitcher_era(name: str, year: int) -> float | None:
    global _pitcher_era_cache
    if _pitcher_era_cache is None:
        _pitcher_era_cache = {}
        with open(DATA_RAW / "npb_pitchers_2015_2025.csv", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    n = _norm_name(r["player"])
                    y = int(r["year"])
                    era = float(r["ERA"])
                    ip = float(r["IP"])
                    if ip >= MIN_IP_PITCHER:
                        _pitcher_era_cache[(n, y)] = era
                except (ValueError, KeyError):
                    continue
    return _pitcher_era_cache.get((name, year))


# ─────────────────────────────────────────────
# Standardization (v2 + career/team features)
# ─────────────────────────────────────────────

def standardize_hitters(data: list[dict]) -> tuple[dict, list[dict]]:
    vals = {
        "woba": [d["prev_woba"] for d in data],
        "K": [d["prev_K_pct"] for d in data if d["prev_K_pct"] is not None],
        "BB": [d["prev_BB_pct"] for d in data if d["prev_BB_pct"] is not None],
        "age": [d["age_from_peak"] for d in data],
        "log_pa": [np.log(d["prev_pa"]) for d in data],
        "log_career_pa": [np.log(max(d["career_pa"], 1)) for d in data],
        "team_bust": [d["team_bust_rate"] for d in data],
    }

    std = {}
    for k, v in vals.items():
        arr = np.array(v)
        std[f"{k}_mean"] = float(np.mean(arr))
        std[f"{k}_sd"] = float(np.std(arr)) if np.std(arr) > 0 else 1.0

    def z(val, key):
        if val is None:
            return 0.0
        return (val - std[f"{key}_mean"]) / std[f"{key}_sd"]

    for d in data:
        d["z_woba"] = z(d["prev_woba"], "woba")
        d["z_K"] = z(d["prev_K_pct"], "K")
        d["z_BB"] = z(d["prev_BB_pct"], "BB")
        d["z_woba_sq"] = d["z_woba"] ** 2
        d["z_K_BB"] = d["z_K"] * d["z_BB"]
        d["z_age"] = z(d["age_from_peak"], "age")
        d["z_log_pa"] = z(np.log(d["prev_pa"]), "log_pa")
        d["z_career_pa"] = z(np.log(max(d["career_pa"], 1)), "log_career_pa")
        d["z_team_bust"] = z(d["team_bust_rate"], "team_bust")

    return std, data


def standardize_pitchers(data: list[dict]) -> tuple[dict, list[dict]]:
    vals = {
        "era": [d["prev_era"] for d in data],
        "fip": [d["prev_fip"] for d in data if d["prev_fip"] is not None],
        "K": [d["prev_K_pct"] for d in data if d["prev_K_pct"] is not None],
        "BB": [d["prev_BB_pct"] for d in data if d["prev_BB_pct"] is not None],
        "age": [d["age_from_peak"] for d in data],
        "log_ip": [np.log(d["prev_ip"]) for d in data],
        "log_career_ip": [np.log(max(d["career_ip"], 1)) for d in data],
        "team_bust": [d["team_bust_rate"] for d in data],
    }

    std = {}
    for k, v in vals.items():
        arr = np.array(v)
        std[f"{k}_mean"] = float(np.mean(arr))
        std[f"{k}_sd"] = float(np.std(arr)) if np.std(arr) > 0 else 1.0

    def z(val, key):
        if val is None:
            return 0.0
        return (val - std[f"{key}_mean"]) / std[f"{key}_sd"]

    for d in data:
        d["z_era"] = z(d["prev_era"], "era")
        d["z_fip"] = z(d["prev_fip"], "fip")
        d["z_K"] = z(d["prev_K_pct"], "K")
        d["z_BB"] = z(d["prev_BB_pct"], "BB")
        d["z_era_sq"] = d["z_era"] ** 2
        d["z_K_BB"] = d["z_K"] * d["z_BB"]
        d["z_age"] = z(d["age_from_peak"], "age")
        d["z_log_ip"] = z(np.log(d["prev_ip"]), "log_ip")
        d["z_career_ip"] = z(np.log(max(d["career_ip"], 1)), "log_career_ip")
        d["z_team_bust"] = z(d["team_bust_rate"], "team_bust")

    return std, data


# ─────────────────────────────────────────────
# Stan data preparation
# ─────────────────────────────────────────────

def make_stan_data_hitters(data: list[dict]) -> dict:
    N = len(data)
    L = len(LEAGUE_INDEX)
    return {
        "N": N, "L": L,
        "league": [d["league_idx"] for d in data],
        "y": [d["npb_woba"] for d in data],
        "lg_avg": [d["lg_woba"] for d in data],
        "z_woba": [d["z_woba"] for d in data],
        "z_K": [d["z_K"] for d in data],
        "z_BB": [d["z_BB"] for d in data],
        "z_woba_sq": [d["z_woba_sq"] for d in data],
        "z_K_BB": [d["z_K_BB"] for d in data],
        "z_age": [d["z_age"] for d in data],
        "is_catcher": [d["is_catcher"] for d in data],
        "is_middle_inf": [d["is_middle_inf"] for d in data],
        "z_log_pa": [d["z_log_pa"] for d in data],
        "is_second_year": [d["is_second_year"] for d in data],
        # Gate features
        "z_career_pa": [d["z_career_pa"] for d in data],
        "z_team_bust": [d["z_team_bust"] for d in data],
    }


def make_stan_data_pitchers(data: list[dict]) -> dict:
    N = len(data)
    L = len(LEAGUE_INDEX)
    return {
        "N": N, "L": L,
        "league": [d["league_idx"] for d in data],
        "y": [d["npb_era"] for d in data],
        "lg_avg": [d["lg_era"] for d in data],
        "z_era": [d["z_era"] for d in data],
        "z_fip": [d["z_fip"] for d in data],
        "z_K": [d["z_K"] for d in data],
        "z_BB": [d["z_BB"] for d in data],
        "z_era_sq": [d["z_era_sq"] for d in data],
        "z_K_BB": [d["z_K_BB"] for d in data],
        "z_age": [d["z_age"] for d in data],
        "is_reliever": [d["is_reliever"] for d in data],
        "z_log_ip": [d["z_log_ip"] for d in data],
        "is_second_year": [d["is_second_year"] for d in data],
        # Gate features
        "z_career_ip": [d["z_career_ip"] for d in data],
        "z_team_bust": [d["z_team_bust"] for d in data],
    }


# ─────────────────────────────────────────────
# Stan fitting
# ─────────────────────────────────────────────

def fit_model(model_file: str, stan_data: dict,
              draws: int = 2000, warmup: int = 1000,
              chains: int = 4, seed: int = 42) -> "CmdStanMCMC":
    from cmdstanpy import CmdStanModel
    model = CmdStanModel(stan_file=str(MODELS_DIR / model_file))
    fit = model.sample(
        data=stan_data,
        chains=chains,
        iter_sampling=draws,
        iter_warmup=warmup,
        seed=seed,
        show_progress=True,
    )
    return fit


def extract_posteriors_hitters(fit) -> dict:
    params = {}
    for i, name in enumerate(["MLB", "AAA", "Other"], 1):
        samples = fit.stan_variable("beta_woba")[..., i - 1]
        params[f"beta_woba_{name}"] = (float(np.mean(samples)), float(np.std(samples)))

    for i, name in enumerate(["MLB", "AAA", "Other"], 1):
        samples = fit.stan_variable("alpha_gate")[..., i - 1]
        params[f"alpha_gate_{name}"] = (float(np.mean(samples)), float(np.std(samples)))

    for pname in ["beta_K", "beta_BB", "beta_woba_sq", "beta_K_BB",
                   "beta_age", "beta_catcher", "beta_middle_inf",
                   "beta_second_year", "sigma_base", "gamma_pa", "nu",
                   "beta_gate_career", "beta_gate_pa", "beta_gate_bb", "beta_gate_team",
                   "mu_bust", "sigma_bust"]:
        samples = fit.stan_variable(pname)
        params[pname] = (float(np.mean(samples)), float(np.std(samples)))

    return params


def extract_posteriors_pitchers(fit) -> dict:
    params = {}
    for i, name in enumerate(["MLB", "AAA", "Other"], 1):
        samples = fit.stan_variable("beta_era")[..., i - 1]
        params[f"beta_era_{name}"] = (float(np.mean(samples)), float(np.std(samples)))

    for i, name in enumerate(["MLB", "AAA", "Other"], 1):
        samples = fit.stan_variable("alpha_gate")[..., i - 1]
        params[f"alpha_gate_{name}"] = (float(np.mean(samples)), float(np.std(samples)))

    for pname in ["beta_fip", "beta_K", "beta_BB", "beta_era_sq", "beta_K_BB",
                   "beta_age", "beta_reliever", "beta_second_year",
                   "sigma_base", "gamma_ip", "nu",
                   "beta_gate_career", "beta_gate_ip", "beta_gate_k", "beta_gate_team",
                   "mu_bust", "sigma_bust"]:
        samples = fit.stan_variable(pname)
        params[pname] = (float(np.mean(samples)), float(np.std(samples)))

    return params


def check_diagnostics(fit):
    diag = fit.diagnose()
    print(diag)
    summary = fit.summary()
    rhat_col = "R_hat" if "R_hat" in summary.columns else "rhat"
    neff_col = "N_Eff" if "N_Eff" in summary.columns else "ess_bulk"
    max_rhat = summary[rhat_col].max() if rhat_col in summary.columns else float("nan")
    min_neff = summary[neff_col].min() if neff_col in summary.columns else float("nan")
    print(f"Max R-hat: {max_rhat:.4f}")
    print(f"Min N_Eff: {min_neff:.0f}")
    if max_rhat > 1.05:
        print("WARNING: R-hat > 1.05 detected — chains may not have converged")
    return max_rhat, min_neff


# ─────────────────────────────────────────────
# LOO Cross-Validation
# ─────────────────────────────────────────────

def loo_cv_hitters(data: list[dict], master: list[dict],
                   draws: int = 2000, warmup: int = 1000) -> dict:
    """Full leave-one-out CV for hitters using v5 mixture model.

    Team bust rate is recomputed per fold (excluding held-out player).
    """
    from cmdstanpy import CmdStanModel
    model = CmdStanModel(stan_file=str(MODELS_DIR / "hitter_foreign_v5.stan"))

    career = load_career_stats()

    N = len(data)
    predictions = []
    n_failed = 0
    print(f"LOO-CV: {N} hitters")

    for i in range(N):
        train_raw = data[:i] + data[i + 1:]
        test_raw = data[i]

        # Recompute team bust rates excluding held-out player
        team_rates = compute_team_bust_rates(
            master,
            exclude_name=test_raw["npb_name"],
            exclude_year=test_raw["year"],
        )

        # Update team bust rates in train data
        train = []
        for d in train_raw:
            d2 = dict(d)
            d2["team_bust_rate"] = team_rates.get(d2["first_team"], 0.20)
            train.append(d2)

        test = dict(test_raw)
        test["team_bust_rate"] = team_rates.get(test["first_team"], 0.20)

        std, train = standardize_hitters(train)
        test_z = _apply_std_hitter(test, std)

        stan_data = make_stan_data_hitters(train)

        try:
            fit = model.sample(
                data=stan_data, chains=2, iter_sampling=draws,
                iter_warmup=warmup, seed=42, show_progress=False,
            )
        except Exception as e:
            n_failed += 1
            print(f"  LOO {i + 1}/{N} FAILED: {e}")
            continue

        pred = _predict_hitter_from_fit(fit, test_z)
        predictions.append({
            "npb_name": test["npb_name"],
            "year": test["year"],
            "actual": test["npb_woba"],
            "pred_v5": pred["mean"],
            "pred_baseline": test["lg_woba"],
            "p_bust": pred["p_bust"],
            "hdi_80": pred["hdi_80"],
            "is_second_year": test["is_second_year"],
        })

        if (i + 1) % 10 == 0:
            print(f"  LOO {i + 1}/{N} done")

    if n_failed > 0:
        print(f"  WARNING: {n_failed}/{N} folds failed")
    if n_failed == N:
        raise RuntimeError(f"All {N} LOO-CV hitter folds failed — model is broken")

    return _summarize_cv(predictions, "wOBA")


def loo_cv_pitchers(data: list[dict], master: list[dict],
                    draws: int = 2000, warmup: int = 1000) -> dict:
    """Full leave-one-out CV for pitchers using v5 mixture model."""
    from cmdstanpy import CmdStanModel
    model = CmdStanModel(stan_file=str(MODELS_DIR / "pitcher_foreign_v5.stan"))

    career = load_career_stats()

    N = len(data)
    predictions = []
    n_failed = 0
    print(f"LOO-CV: {N} pitchers")

    for i in range(N):
        train_raw = data[:i] + data[i + 1:]
        test_raw = data[i]

        team_rates = compute_team_bust_rates(
            master,
            exclude_name=test_raw["npb_name"],
            exclude_year=test_raw["year"],
        )

        train = []
        for d in train_raw:
            d2 = dict(d)
            d2["team_bust_rate"] = team_rates.get(d2["first_team"], 0.20)
            train.append(d2)

        test = dict(test_raw)
        test["team_bust_rate"] = team_rates.get(test["first_team"], 0.20)

        std, train = standardize_pitchers(train)
        test_z = _apply_std_pitcher(test, std)

        stan_data = make_stan_data_pitchers(train)

        try:
            fit = model.sample(
                data=stan_data, chains=2, iter_sampling=draws,
                iter_warmup=warmup, seed=42, show_progress=False,
            )
        except Exception as e:
            n_failed += 1
            print(f"  LOO {i + 1}/{N} FAILED: {e}")
            continue

        pred = _predict_pitcher_from_fit(fit, test_z)
        predictions.append({
            "npb_name": test["npb_name"],
            "year": test["year"],
            "actual": test["npb_era"],
            "pred_v5": pred["mean"],
            "pred_baseline": test["lg_era"],
            "p_bust": pred["p_bust"],
            "hdi_80": pred["hdi_80"],
            "is_second_year": test["is_second_year"],
        })

        if (i + 1) % 10 == 0:
            print(f"  LOO {i + 1}/{N} done")

    if n_failed > 0:
        print(f"  WARNING: {n_failed}/{N} folds failed")
    if n_failed == N:
        raise RuntimeError(f"All {N} LOO-CV pitcher folds failed — model is broken")

    return _summarize_cv(predictions, "ERA")


# ─────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────

def _apply_std_hitter(row: dict, std: dict) -> dict:
    def z(val, key):
        if val is None:
            return 0.0
        return (val - std[f"{key}_mean"]) / std[f"{key}_sd"]

    row = dict(row)
    row["z_woba"] = z(row["prev_woba"], "woba")
    row["z_K"] = z(row["prev_K_pct"], "K")
    row["z_BB"] = z(row["prev_BB_pct"], "BB")
    row["z_woba_sq"] = row["z_woba"] ** 2
    row["z_K_BB"] = row["z_K"] * row["z_BB"]
    row["z_age"] = z(row["age_from_peak"], "age")
    row["z_log_pa"] = z(np.log(row["prev_pa"]), "log_pa")
    row["z_career_pa"] = z(np.log(max(row["career_pa"], 1)), "log_career_pa")
    row["z_team_bust"] = z(row["team_bust_rate"], "team_bust")
    return row


def _apply_std_pitcher(row: dict, std: dict) -> dict:
    def z(val, key):
        if val is None:
            return 0.0
        return (val - std[f"{key}_mean"]) / std[f"{key}_sd"]

    row = dict(row)
    row["z_era"] = z(row["prev_era"], "era")
    row["z_fip"] = z(row["prev_fip"], "fip")
    row["z_K"] = z(row["prev_K_pct"], "K")
    row["z_BB"] = z(row["prev_BB_pct"], "BB")
    row["z_era_sq"] = row["z_era"] ** 2
    row["z_K_BB"] = row["z_K"] * row["z_BB"]
    row["z_age"] = z(row["age_from_peak"], "age")
    row["z_log_ip"] = z(np.log(row["prev_ip"]), "log_ip")
    row["z_career_ip"] = z(np.log(max(row["career_ip"], 1)), "log_career_ip")
    row["z_team_bust"] = z(row["team_bust_rate"], "team_bust")
    return row


def _predict_hitter_from_fit(fit, row: dict, n_samples: int = 5000) -> dict:
    """Generate mixture posterior predictive samples for one hitter."""
    rng = np.random.default_rng(42)

    # Gate parameters
    alpha_gate_all = fit.stan_variable("alpha_gate")
    league_col = row["league_idx"] - 1
    alpha_gate_s = alpha_gate_all[:, league_col]

    gate_params = {}
    for p in ["beta_gate_career", "beta_gate_pa", "beta_gate_bb", "beta_gate_team"]:
        gate_params[p] = fit.stan_variable(p)

    # Bust parameters
    mu_bust_s = fit.stan_variable("mu_bust")
    sigma_bust_s = fit.stan_variable("sigma_bust")

    # Performance parameters
    beta_woba_all = fit.stan_variable("beta_woba")
    beta_woba_s = beta_woba_all[:, league_col]

    perf_params = {}
    for p in ["beta_K", "beta_BB", "beta_woba_sq", "beta_K_BB",
              "beta_age", "beta_catcher", "beta_middle_inf",
              "beta_second_year", "sigma_base", "gamma_pa", "nu"]:
        perf_params[p] = fit.stan_variable(p)

    n = min(n_samples, len(alpha_gate_s))
    idx = rng.choice(len(alpha_gate_s), size=n, replace=n > len(alpha_gate_s))

    # Gate logit
    gate = (alpha_gate_s[idx]
            + gate_params["beta_gate_career"][idx] * row["z_career_pa"]
            + gate_params["beta_gate_pa"][idx] * row["z_log_pa"]
            + gate_params["beta_gate_bb"][idx] * row["z_BB"]
            + gate_params["beta_gate_team"][idx] * row["z_team_bust"])

    pi_bust = 1.0 / (1.0 + np.exp(-gate))  # sigmoid

    # Performance mean
    mu_perf = (row["lg_woba"]
               + beta_woba_s[idx] * row["z_woba"]
               + perf_params["beta_K"][idx] * row["z_K"]
               + perf_params["beta_BB"][idx] * row["z_BB"]
               + perf_params["beta_woba_sq"][idx] * row["z_woba_sq"]
               + perf_params["beta_K_BB"][idx] * row["z_K_BB"]
               + perf_params["beta_age"][idx] * row["z_age"]
               + perf_params["beta_catcher"][idx] * row["is_catcher"]
               + perf_params["beta_middle_inf"][idx] * row["is_middle_inf"]
               + perf_params["beta_second_year"][idx] * row["is_second_year"])

    exponent = np.clip(perf_params["gamma_pa"][idx] * row["z_log_pa"], -5.0, 2.0)
    sigma_perf = perf_params["sigma_base"][idx] * np.exp(exponent)

    # Sample from mixture
    is_bust_draw = rng.random(n) < pi_bust
    nu_s = perf_params["nu"][idx]

    # Bust samples
    y_bust = mu_bust_s[idx] + sigma_bust_s[idx] * rng.standard_normal(n)

    # Performance samples (Student-t via normal/chi2 representation)
    chi2_samples = rng.chisquare(nu_s)
    t_samples = rng.standard_normal(n) / np.sqrt(chi2_samples / nu_s)
    y_perf = mu_perf + sigma_perf * t_samples

    y_pred = np.where(is_bust_draw, y_bust, y_perf)
    y_pred = np.clip(y_pred, 0, None)

    return {
        "mean": float(np.mean(y_pred)),
        "std": float(np.std(y_pred)),
        "p_bust": float(np.mean(pi_bust)),
        "hdi_80": (float(np.percentile(y_pred, 10)), float(np.percentile(y_pred, 90))),
        "hdi_95": (float(np.percentile(y_pred, 2.5)), float(np.percentile(y_pred, 97.5))),
    }


def _predict_pitcher_from_fit(fit, row: dict, n_samples: int = 5000) -> dict:
    """Generate mixture posterior predictive samples for one pitcher."""
    rng = np.random.default_rng(42)

    alpha_gate_all = fit.stan_variable("alpha_gate")
    league_col = row["league_idx"] - 1
    alpha_gate_s = alpha_gate_all[:, league_col]

    gate_params = {}
    for p in ["beta_gate_career", "beta_gate_ip", "beta_gate_k", "beta_gate_team"]:
        gate_params[p] = fit.stan_variable(p)

    mu_bust_s = fit.stan_variable("mu_bust")
    sigma_bust_s = fit.stan_variable("sigma_bust")

    beta_era_all = fit.stan_variable("beta_era")
    beta_era_s = beta_era_all[:, league_col]

    perf_params = {}
    for p in ["beta_fip", "beta_K", "beta_BB", "beta_era_sq", "beta_K_BB",
              "beta_age", "beta_reliever", "beta_second_year",
              "sigma_base", "gamma_ip", "nu"]:
        perf_params[p] = fit.stan_variable(p)

    n = min(n_samples, len(alpha_gate_s))
    idx = rng.choice(len(alpha_gate_s), size=n, replace=n > len(alpha_gate_s))

    gate = (alpha_gate_s[idx]
            + gate_params["beta_gate_career"][idx] * row["z_career_ip"]
            + gate_params["beta_gate_ip"][idx] * row["z_log_ip"]
            + gate_params["beta_gate_k"][idx] * row["z_K"]
            + gate_params["beta_gate_team"][idx] * row["z_team_bust"])

    pi_bust = 1.0 / (1.0 + np.exp(-gate))

    mu_perf = (row["lg_era"]
               + beta_era_s[idx] * row["z_era"]
               + perf_params["beta_fip"][idx] * row["z_fip"]
               + perf_params["beta_K"][idx] * row["z_K"]
               + perf_params["beta_BB"][idx] * row["z_BB"]
               + perf_params["beta_era_sq"][idx] * row["z_era_sq"]
               + perf_params["beta_K_BB"][idx] * row["z_K_BB"]
               + perf_params["beta_age"][idx] * row["z_age"]
               + perf_params["beta_reliever"][idx] * row["is_reliever"]
               + perf_params["beta_second_year"][idx] * row["is_second_year"])

    exponent = np.clip(perf_params["gamma_ip"][idx] * row["z_log_ip"], -5.0, 2.0)
    sigma_perf = perf_params["sigma_base"][idx] * np.exp(exponent)

    is_bust_draw = rng.random(n) < pi_bust
    nu_s = perf_params["nu"][idx]

    y_bust = mu_bust_s[idx] + sigma_bust_s[idx] * rng.standard_normal(n)
    chi2_samples = rng.chisquare(nu_s)
    t_samples = rng.standard_normal(n) / np.sqrt(chi2_samples / nu_s)
    y_perf = mu_perf + sigma_perf * t_samples

    y_pred = np.where(is_bust_draw, y_bust, y_perf)
    y_pred = np.clip(y_pred, 0, None)

    return {
        "mean": float(np.mean(y_pred)),
        "std": float(np.std(y_pred)),
        "p_bust": float(np.mean(pi_bust)),
        "hdi_80": (float(np.percentile(y_pred, 10)), float(np.percentile(y_pred, 90))),
        "hdi_95": (float(np.percentile(y_pred, 2.5)), float(np.percentile(y_pred, 97.5))),
    }


# ─────────────────────────────────────────────
# CV summary
# ─────────────────────────────────────────────

def _summarize_cv(predictions: list[dict], stat_name: str) -> dict:
    if not predictions:
        return {"error": "no predictions"}

    actuals = np.array([p["actual"] for p in predictions])
    preds_v5 = np.array([p["pred_v5"] for p in predictions])
    preds_bl = np.array([p["pred_baseline"] for p in predictions])

    mae_v5 = float(np.mean(np.abs(actuals - preds_v5)))
    mae_bl = float(np.mean(np.abs(actuals - preds_bl)))

    coverage = np.mean([
        p["hdi_80"][0] <= p["actual"] <= p["hdi_80"][1]
        for p in predictions
    ])

    improvement = (mae_bl - mae_v5) / mae_bl * 100

    # Bust detection accuracy
    if stat_name == "wOBA":
        actual_busts = actuals < BUST_WOBA
    else:
        actual_busts = actuals > BUST_ERA

    p_busts = np.array([p["p_bust"] for p in predictions])
    pred_busts = p_busts > 0.5

    n_actual_busts = int(actual_busts.sum())
    n_pred_busts = int(pred_busts.sum())
    n_correct_busts = int((actual_busts & pred_busts).sum())

    bust_precision = n_correct_busts / n_pred_busts if n_pred_busts > 0 else 0.0
    bust_recall = n_correct_busts / n_actual_busts if n_actual_busts > 0 else 0.0

    result = {
        "stat": stat_name,
        "n": len(predictions),
        "mae_v5": mae_v5,
        "mae_baseline": mae_bl,
        "improvement_pct": float(improvement),
        "coverage_80": float(coverage),
        "bust_detection": {
            "n_actual": n_actual_busts,
            "n_predicted": n_pred_busts,
            "n_correct": n_correct_busts,
            "precision": float(bust_precision),
            "recall": float(bust_recall),
        },
        "predictions": predictions,
    }

    print(f"\n{'=' * 60}")
    print(f"LOO-CV Results — {stat_name}")
    print(f"  N = {len(predictions)}")
    print(f"  MAE v5:       {mae_v5:.4f}")
    print(f"  MAE baseline: {mae_bl:.4f}")
    print(f"  Improvement:  {improvement:+.1f}%")
    print(f"  80% Coverage: {coverage:.1%}")
    print(f"\n  Bust detection (threshold p>0.5):")
    print(f"    Actual busts:    {n_actual_busts}")
    print(f"    Predicted busts: {n_pred_busts}")
    print(f"    Correct:         {n_correct_busts}")
    print(f"    Precision:       {bust_precision:.1%}")
    print(f"    Recall:          {bust_recall:.1%}")
    print(f"{'=' * 60}\n")

    return result


# ─────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────

def save_results(hitter_params: dict, pitcher_params: dict,
                 hitter_std: dict, pitcher_std: dict,
                 backtest: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    params_out = {
        "version": "v5",
        "generated": datetime.now().isoformat(),
        "league_index": LEAGUE_INDEX,
        "hitter": {
            "params": {k: {"mean": v[0], "sd": v[1]} for k, v in hitter_params.items()},
            "standardization": hitter_std,
        },
        "pitcher": {
            "params": {k: {"mean": v[0], "sd": v[1]} for k, v in pitcher_params.items()},
            "standardization": pitcher_std,
        },
    }
    with open(output_dir / "foreign_v5_posterior.json", "w") as f:
        json.dump(params_out, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_dir / 'foreign_v5_posterior.json'}")

    with open(output_dir / "foreign_v5_backtest.json", "w") as f:
        bt = {}
        for k, v in backtest.items():
            if isinstance(v, dict) and "predictions" in v:
                bt[k] = {kk: vv for kk, vv in v.items() if kk != "predictions"}
            else:
                bt[k] = v
        json.dump(bt, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_dir / 'foreign_v5_backtest.json'}")

    for ptype in ["hitter", "pitcher"]:
        key = f"loo_{ptype}"
        if key not in backtest or "predictions" not in backtest[key]:
            continue
        preds = backtest[key]["predictions"]
        csv_path = output_dir / f"foreign_v5_{ptype}_loo_predictions.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            if preds:
                writer = csv.DictWriter(f, fieldnames=preds[0].keys())
                writer.writeheader()
                for p in preds:
                    row = dict(p)
                    if "hdi_80" in row:
                        row["hdi_80"] = f"{row['hdi_80'][0]:.4f},{row['hdi_80'][1]:.4f}"
                    writer.writerow(row)
        print(f"Saved: {csv_path}")


# ─────────────────────────────────────────────
# v1 comparison (baseline)
# ─────────────────────────────────────────────

def v1_baseline_predictions(hitters: list[dict], pitchers: list[dict]) -> dict:
    h_mae = float(np.mean([abs(d["npb_woba"] - d["lg_woba"]) for d in hitters])) if hitters else 0
    p_mae = float(np.mean([abs(d["npb_era"] - d["lg_era"]) for d in pitchers])) if pitchers else 0
    return {
        "hitter_mae_v1_baseline": h_mae,
        "pitcher_mae_v1_baseline": p_mae,
        "hitter_n": len(hitters),
        "pitcher_n": len(pitchers),
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    t0 = time.time()

    parser = argparse.ArgumentParser(description="Foreign player v5 model (Mixture of Experts)")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--loo-cv", action="store_true", help="Run LOO cross-validation")
    parser.add_argument("--fit-only", action="store_true", help="Fit full model only (no CV)")
    args = parser.parse_args()

    print("=" * 60)
    print("Foreign Player Prediction Model v5 (Mixture of Experts)")
    print("=" * 60)

    print("\n[1/5] Building dataset...")
    hitters, pitchers, master = build_dataset()
    print(f"  Hitters: {len(hitters)} rows "
          f"(1st year: {sum(1 for d in hitters if d['is_second_year'] == 0)}, "
          f"2nd year: {sum(1 for d in hitters if d['is_second_year'] == 1)})")
    print(f"  Pitchers: {len(pitchers)} rows "
          f"(1st year: {sum(1 for d in pitchers if d['is_second_year'] == 0)}, "
          f"2nd year: {sum(1 for d in pitchers if d['is_second_year'] == 1)})")

    from collections import Counter
    for ptype, data in [("Hitters", hitters), ("Pitchers", pitchers)]:
        lg_dist = Counter(d["league_group"] for d in data)
        print(f"  {ptype} by league: {dict(lg_dist)}")

    print("\n[2/5] Computing v1 baseline...")
    v1 = v1_baseline_predictions(hitters, pitchers)
    print(f"  v1 hitter MAE (lg_avg baseline): {v1['hitter_mae_v1_baseline']:.4f}")
    print(f"  v1 pitcher MAE (lg_avg baseline): {v1['pitcher_mae_v1_baseline']:.4f}")

    _log_elapsed("dataset build + v1 baseline", t0)

    backtest = {"v1_baseline": v1}

    if args.loo_cv:
        print("\n[3/5] Running LOO-CV for hitters...")
        hitters_copy = [dict(d) for d in hitters]
        loo_h = loo_cv_hitters(hitters_copy, master, draws=args.draws, warmup=args.warmup)
        backtest["loo_hitter"] = loo_h
        _log_elapsed("LOO-CV hitters", t0)

        print("\n[4/5] Running LOO-CV for pitchers...")
        pitchers_copy = [dict(d) for d in pitchers]
        loo_p = loo_cv_pitchers(pitchers_copy, master, draws=args.draws, warmup=args.warmup)
        backtest["loo_pitcher"] = loo_p
        _log_elapsed("LOO-CV pitchers", t0)

    # Full model fit (always run)
    print("\n[5/5] Fitting full model on all data...")

    hitters_full = [dict(d) for d in hitters]
    std_h, hitters_full = standardize_hitters(hitters_full)
    stan_h = make_stan_data_hitters(hitters_full)
    print(f"  Hitter Stan data: N={stan_h['N']}, L={stan_h['L']}")
    fit_h = fit_model("hitter_foreign_v5.stan", stan_h,
                      draws=args.draws, warmup=args.warmup)
    _log_elapsed("full model hitter Stan sampling", t0)
    print("\n  Hitter diagnostics:")
    check_diagnostics(fit_h)
    hitter_params = extract_posteriors_hitters(fit_h)

    pitchers_full = [dict(d) for d in pitchers]
    std_p, pitchers_full = standardize_pitchers(pitchers_full)
    stan_p = make_stan_data_pitchers(pitchers_full)
    print(f"\n  Pitcher Stan data: N={stan_p['N']}, L={stan_p['L']}")
    fit_p = fit_model("pitcher_foreign_v5.stan", stan_p,
                      draws=args.draws, warmup=args.warmup)
    _log_elapsed("full model pitcher Stan sampling", t0)
    print("\n  Pitcher diagnostics:")
    check_diagnostics(fit_p)
    pitcher_params = extract_posteriors_pitchers(fit_p)

    # Print posterior summary
    print("\n" + "=" * 60)
    print("HITTER POSTERIOR PARAMETERS")
    print("=" * 60)
    for k, (mean, sd) in sorted(hitter_params.items()):
        sig = "***" if abs(mean) > 2 * sd else "**" if abs(mean) > 1.5 * sd else "*" if abs(mean) > sd else ""
        print(f"  {k:25s} = {mean:+.4f} ± {sd:.4f}  {sig}")

    print("\n" + "=" * 60)
    print("PITCHER POSTERIOR PARAMETERS")
    print("=" * 60)
    for k, (mean, sd) in sorted(pitcher_params.items()):
        sig = "***" if abs(mean) > 2 * sd else "**" if abs(mean) > 1.5 * sd else "*" if abs(mean) > sd else ""
        print(f"  {k:25s} = {mean:+.4f} ± {sd:.4f}  {sig}")

    print("\n[Save] Writing results...")
    save_results(hitter_params, pitcher_params, std_h, std_p,
                 backtest, DATA_MODEL)

    _log_elapsed("total", t0)
    print("\nDone!")


if __name__ == "__main__":
    main()
