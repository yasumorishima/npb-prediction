"""Foreign player prediction model v3b — league-specific affine transformation.

Key changes from v2:
  - Hitter: raw prev_woba with league-specific alpha + beta (not z-scored)
  - Pitcher: raw prev_K% and prev_BB% as main predictors (not ERA)
    K%/BB% have higher cross-league reproducibility than ERA
  - Hierarchical priors on league-specific parameters for partial pooling
  - lg_avg removed from Stan data (absorbed into alpha intercepts)
  - z_woba_sq removed (will be addressed by Student-t in step A)

Data sources (all relative to repo root):
  - data/foreign/foreign_players_master.csv
  - data/foreign/player_conversion_details.csv
  - data/foreign/foreign_prev_stats.csv
  - data/raw/npb_player_birthdays.csv
  - data/raw/npb_players_profile_2024.csv
  - data/raw/npb_sabermetrics_2015_2025.csv
  - data/raw/npb_pitchers_2015_2025.csv

Usage:
  python src/foreign_v3b_model.py --draws 2000 --warmup 1000
  python src/foreign_v3b_model.py --loo-cv
  python src/foreign_v3b_model.py --expanding-cv
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
MIN_PA_HITTER = 50    # NPB first-year minimum PA for inclusion
MIN_IP_PITCHER = 20   # NPB first-year minimum IP for inclusion
MIN_PREV_PA = 20      # Previous-league minimum PA
MIN_PREV_IP = 5       # Previous-league minimum IP

# League mapping: group small categories
LEAGUE_GROUPS = {
    "MLB": "MLB",
    "AAA": "AAA",
    "MiLB": "AAA",        # minor league -> AAA group
    "KBO": "Other",
    "CPBL": "Other",
    "Cuba": "Other",
    "Independent": "Other",
    "Amateur": "Other",
    "": "Other",
}
LEAGUE_INDEX = {"MLB": 1, "AAA": 2, "Other": 3}

# Position classification from profile field
CATCHER_KEYWORDS = {"捕手"}
MIDDLE_INF_KEYWORDS = {"遊撃手", "二塁手", "内野手"}


# ─────────────────────────────────────────────
# Data loading (unchanged from v2)
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
    all_years = set(woba_by_year.keys()) | set(era_by_year.keys())
    for y in all_years:
        result[y] = {
            "lg_woba": float(np.mean(woba_by_year.get(y, [0.310]))),
            "lg_era": float(np.mean(era_by_year.get(y, [3.50]))),
        }
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


def build_dataset() -> tuple[list[dict], list[dict]]:
    """Build complete datasets for hitters and pitchers."""
    master = load_master()
    conv = load_conversion_details()
    prev_stats = load_prev_stats()
    birthdays = load_birthdays()
    profiles = load_profiles()
    hitter_years = load_npb_hitter_years()
    pitcher_years = load_npb_pitcher_years()
    pitcher_ip_map = load_npb_pitcher_ip()
    lg_avgs = load_league_averages()

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

        bday = birthdays.get(name)
        age = _age_at_year(bday, first_year) if bday else None

        c = conv.get(name, {})
        ps = prev_stats.get(name, {})

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

            is_c, is_mi = _classify_position(pos_str)
            lg = lg_avgs.get(first_year, {"lg_woba": 0.310})

            hitters.append({
                "npb_name": name,
                "year": first_year,
                "origin_league": origin,
                "league_group": lg_group,
                "league_idx": lg_idx,
                "prev_woba": prev_woba,
                "prev_K_pct": prev_K,
                "prev_BB_pct": prev_BB,
                "prev_pa": prev_pa or 200,
                "npb_woba": npb_woba,
                "age": age or PEAK_AGE,
                "age_from_peak": (age or PEAK_AGE) - PEAK_AGE,
                "is_catcher": 1.0 if is_c else 0.0,
                "is_middle_inf": 1.0 if is_mi else 0.0,
                "is_second_year": 0.0,
                "lg_woba": lg["lg_woba"],
            })

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
                        "prev_woba": prev_woba,
                        "prev_K_pct": prev_K,
                        "prev_BB_pct": prev_BB,
                        "prev_pa": prev_pa or 200,
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

            # v3b requires K% and BB% as main predictors
            if prev_K is None or prev_BB is None:
                continue

            is_rel = npb_ip < 50
            lg = lg_avgs.get(first_year, {"lg_era": 3.50})

            pitchers.append({
                "npb_name": name,
                "year": first_year,
                "origin_league": origin,
                "league_group": lg_group,
                "league_idx": lg_idx,
                "prev_era": prev_era,
                "prev_fip": prev_fip,
                "prev_K_pct": prev_K,
                "prev_BB_pct": prev_BB,
                "prev_ip": prev_ip or 50,
                "npb_era": npb_era,
                "age": age or PEAK_AGE,
                "age_from_peak": (age or PEAK_AGE) - PEAK_AGE,
                "is_reliever": 1.0 if is_rel else 0.0,
                "is_second_year": 0.0,
                "lg_era": lg["lg_era"],
            })

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
                        "prev_era": prev_era,
                        "prev_fip": prev_fip,
                        "prev_K_pct": prev_K,
                        "prev_BB_pct": prev_BB,
                        "prev_ip": prev_ip or 50,
                        "npb_era": era_2nd,
                        "age": (age or PEAK_AGE) + 1,
                        "age_from_peak": (age or PEAK_AGE) + 1 - PEAK_AGE,
                        "is_reliever": 1.0 if era_2nd_ip < 50 else 0.0,
                        "is_second_year": 1.0,
                        "lg_era": lg2["lg_era"],
                    })

    return hitters, pitchers


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
# Standardization
# ─────────────────────────────────────────────

def standardize_hitters(data: list[dict]) -> tuple[dict, list[dict]]:
    """Compute standardization params and add z-scored features.

    v3b: prev_woba is NOT z-scored (passed raw to Stan).
    K%, BB%, age, log_pa are still z-scored for auxiliary features.
    """
    vals = {
        "K": [d["prev_K_pct"] for d in data if d["prev_K_pct"] is not None],
        "BB": [d["prev_BB_pct"] for d in data if d["prev_BB_pct"] is not None],
        "age": [d["age_from_peak"] for d in data],
        "log_pa": [np.log(d["prev_pa"]) for d in data],
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
        d["z_K"] = z(d["prev_K_pct"], "K")
        d["z_BB"] = z(d["prev_BB_pct"], "BB")
        d["z_K_BB"] = d["z_K"] * d["z_BB"]
        d["z_age"] = z(d["age_from_peak"], "age")
        d["z_log_pa"] = z(np.log(d["prev_pa"]), "log_pa")

    return std, data


def standardize_pitchers(data: list[dict]) -> tuple[dict, list[dict]]:
    """Compute standardization params for pitcher auxiliary features.

    v3b: prev_K% and prev_BB% are NOT z-scored for main predictors (raw to Stan).
    FIP, age, log_ip, and K*BB interaction use z-scores.
    """
    vals = {
        "fip": [d["prev_fip"] for d in data if d["prev_fip"] is not None],
        "K": [d["prev_K_pct"] for d in data],
        "BB": [d["prev_BB_pct"] for d in data],
        "age": [d["age_from_peak"] for d in data],
        "log_ip": [np.log(d["prev_ip"]) for d in data],
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
        d["z_fip"] = z(d["prev_fip"], "fip")
        # z-scored K/BB for interaction term only
        d["z_K"] = z(d["prev_K_pct"], "K")
        d["z_BB"] = z(d["prev_BB_pct"], "BB")
        d["z_K_BB"] = d["z_K"] * d["z_BB"]
        d["z_age"] = z(d["age_from_peak"], "age")
        d["z_log_ip"] = z(np.log(d["prev_ip"]), "log_ip")

    return std, data


# ─────────────────────────────────────────────
# Stan data preparation
# ─────────────────────────────────────────────

def make_stan_data_hitters(data: list[dict]) -> dict:
    N = len(data)
    L = len(LEAGUE_INDEX)
    return {
        "N": N,
        "L": L,
        "league": [d["league_idx"] for d in data],
        "y": [d["npb_woba"] for d in data],
        "prev_woba": [d["prev_woba"] for d in data],  # raw value
        "z_K": [d["z_K"] for d in data],
        "z_BB": [d["z_BB"] for d in data],
        "z_K_BB": [d["z_K_BB"] for d in data],
        "z_age": [d["z_age"] for d in data],
        "is_catcher": [d["is_catcher"] for d in data],
        "is_middle_inf": [d["is_middle_inf"] for d in data],
        "z_log_pa": [d["z_log_pa"] for d in data],
        "is_second_year": [d["is_second_year"] for d in data],
    }


def make_stan_data_pitchers(data: list[dict]) -> dict:
    N = len(data)
    L = len(LEAGUE_INDEX)
    return {
        "N": N,
        "L": L,
        "league": [d["league_idx"] for d in data],
        "y": [d["npb_era"] for d in data],
        "prev_K": [d["prev_K_pct"] for d in data],    # raw value
        "prev_BB": [d["prev_BB_pct"] for d in data],   # raw value
        "z_fip": [d["z_fip"] for d in data],
        "z_K_BB": [d["z_K_BB"] for d in data],
        "z_age": [d["z_age"] for d in data],
        "is_reliever": [d["is_reliever"] for d in data],
        "z_log_ip": [d["z_log_ip"] for d in data],
        "is_second_year": [d["is_second_year"] for d in data],
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
        for var in ["alpha", "beta_woba"]:
            samples = fit.stan_variable(var)[..., i - 1]
            params[f"{var}_{name}"] = (float(np.mean(samples)), float(np.std(samples)))

    # Hierarchical hyperparameters
    for pname in ["alpha_mu", "alpha_sigma", "beta_woba_mu", "beta_woba_sigma"]:
        samples = fit.stan_variable(pname)
        params[pname] = (float(np.mean(samples)), float(np.std(samples)))

    for pname in ["beta_K", "beta_BB", "beta_K_BB",
                   "beta_age", "beta_catcher", "beta_middle_inf",
                   "beta_second_year", "sigma_base", "gamma_pa"]:
        samples = fit.stan_variable(pname)
        params[pname] = (float(np.mean(samples)), float(np.std(samples)))

    return params


def extract_posteriors_pitchers(fit) -> dict:
    params = {}

    for i, name in enumerate(["MLB", "AAA", "Other"], 1):
        for var in ["alpha_K", "beta_K", "alpha_BB", "beta_BB"]:
            samples = fit.stan_variable(var)[..., i - 1]
            params[f"{var}_{name}"] = (float(np.mean(samples)), float(np.std(samples)))

    for pname in ["alpha_K_mu", "alpha_K_sigma", "beta_K_mu", "beta_K_sigma",
                   "alpha_BB_mu", "alpha_BB_sigma", "beta_BB_mu", "beta_BB_sigma"]:
        samples = fit.stan_variable(pname)
        params[pname] = (float(np.mean(samples)), float(np.std(samples)))

    for pname in ["beta_fip", "beta_K_BB",
                   "beta_age", "beta_reliever", "beta_second_year",
                   "sigma_base", "gamma_ip"]:
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
        print("WARNING: R-hat > 1.05 detected -- chains may not have converged")
    return max_rhat, min_neff


# ─────────────────────────────────────────────
# LOO Cross-Validation
# ─────────────────────────────────────────────

def loo_cv_hitters(data: list[dict], draws: int = 2000, warmup: int = 1000) -> dict:
    from cmdstanpy import CmdStanModel
    model = CmdStanModel(stan_file=str(MODELS_DIR / "hitter_foreign_v3b.stan"))

    N = len(data)
    predictions = []
    n_failed = 0
    print(f"LOO-CV: {N} hitters")

    for i in range(N):
        train = data[:i] + data[i + 1:]
        test = data[i]

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
            "pred_v3b": pred["mean"],
            "pred_baseline": test["lg_woba"],
            "hdi_80": pred["hdi_80"],
            "is_second_year": test["is_second_year"],
        })

        if (i + 1) % 10 == 0:
            print(f"  LOO {i + 1}/{N} done")

    if n_failed > 0:
        print(f"  WARNING: {n_failed}/{N} folds failed")
    if n_failed == N:
        raise RuntimeError(f"All {N} LOO-CV hitter folds failed")

    return _summarize_cv(predictions, "wOBA")


def loo_cv_pitchers(data: list[dict], draws: int = 2000, warmup: int = 1000) -> dict:
    from cmdstanpy import CmdStanModel
    model = CmdStanModel(stan_file=str(MODELS_DIR / "pitcher_foreign_v3b.stan"))

    N = len(data)
    predictions = []
    n_failed = 0
    print(f"LOO-CV: {N} pitchers")

    for i in range(N):
        train = data[:i] + data[i + 1:]
        test = data[i]

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
            "pred_v3b": pred["mean"],
            "pred_baseline": test["lg_era"],
            "hdi_80": pred["hdi_80"],
            "is_second_year": test["is_second_year"],
        })

        if (i + 1) % 10 == 0:
            print(f"  LOO {i + 1}/{N} done")

    if n_failed > 0:
        print(f"  WARNING: {n_failed}/{N} folds failed")
    if n_failed == N:
        raise RuntimeError(f"All {N} LOO-CV pitcher folds failed")

    return _summarize_cv(predictions, "ERA")


# ─────────────────────────────────────────────
# Expanding Window CV
# ─────────────────────────────────────────────

def expanding_cv_hitters(data: list[dict], draws: int = 2000, warmup: int = 1000) -> dict:
    from cmdstanpy import CmdStanModel
    model = CmdStanModel(stan_file=str(MODELS_DIR / "hitter_foreign_v3b.stan"))

    years = sorted(set(d["year"] for d in data))
    min_train_year = min(years)
    results_by_year = {}

    for test_year in years:
        if test_year <= min_train_year + 1:
            continue

        train = [d for d in data if d["year"] < test_year]
        test = [d for d in data if d["year"] == test_year]

        if len(train) < 10 or len(test) == 0:
            continue

        std, train = standardize_hitters(train)
        test_z = [_apply_std_hitter(t, std) for t in test]

        stan_data = make_stan_data_hitters(train)

        try:
            fit = model.sample(
                data=stan_data, chains=2, iter_sampling=draws,
                iter_warmup=warmup, seed=42, show_progress=False,
            )
        except Exception as e:
            print(f"  Year {test_year} FAILED: {e}")
            continue

        preds = []
        for t, tz in zip(test, test_z):
            pred = _predict_hitter_from_fit(fit, tz)
            preds.append({
                "npb_name": t["npb_name"],
                "year": t["year"],
                "actual": t["npb_woba"],
                "pred_v3b": pred["mean"],
                "pred_baseline": t["lg_woba"],
                "hdi_80": pred["hdi_80"],
            })

        mae_v3b = np.mean([abs(p["actual"] - p["pred_v3b"]) for p in preds])
        mae_bl = np.mean([abs(p["actual"] - p["pred_baseline"]) for p in preds])
        cov = np.mean([p["hdi_80"][0] <= p["actual"] <= p["hdi_80"][1] for p in preds])

        results_by_year[test_year] = {
            "n": len(preds),
            "mae_v3b": float(mae_v3b),
            "mae_baseline": float(mae_bl),
            "improvement": float((mae_bl - mae_v3b) / mae_bl * 100),
            "coverage_80": float(cov),
        }
        print(f"  Year {test_year}: n={len(preds)} MAE_v3b={mae_v3b:.4f} MAE_bl={mae_bl:.4f} "
              f"({(mae_bl-mae_v3b)/mae_bl*100:+.1f}%) cov={cov:.1%}")

    return results_by_year


def expanding_cv_pitchers(data: list[dict], draws: int = 2000, warmup: int = 1000) -> dict:
    from cmdstanpy import CmdStanModel
    model = CmdStanModel(stan_file=str(MODELS_DIR / "pitcher_foreign_v3b.stan"))

    years = sorted(set(d["year"] for d in data))
    min_train_year = min(years)
    results_by_year = {}

    for test_year in years:
        if test_year <= min_train_year + 1:
            continue

        train = [d for d in data if d["year"] < test_year]
        test = [d for d in data if d["year"] == test_year]

        if len(train) < 10 or len(test) == 0:
            continue

        std, train = standardize_pitchers(train)
        test_z = [_apply_std_pitcher(t, std) for t in test]

        stan_data = make_stan_data_pitchers(train)

        try:
            fit = model.sample(
                data=stan_data, chains=2, iter_sampling=draws,
                iter_warmup=warmup, seed=42, show_progress=False,
            )
        except Exception as e:
            print(f"  Year {test_year} FAILED: {e}")
            continue

        preds = []
        for t, tz in zip(test, test_z):
            pred = _predict_pitcher_from_fit(fit, tz)
            preds.append({
                "npb_name": t["npb_name"],
                "year": t["year"],
                "actual": t["npb_era"],
                "pred_v3b": pred["mean"],
                "pred_baseline": t["lg_era"],
                "hdi_80": pred["hdi_80"],
            })

        mae_v3b = np.mean([abs(p["actual"] - p["pred_v3b"]) for p in preds])
        mae_bl = np.mean([abs(p["actual"] - p["pred_baseline"]) for p in preds])
        cov = np.mean([p["hdi_80"][0] <= p["actual"] <= p["hdi_80"][1] for p in preds])

        results_by_year[test_year] = {
            "n": len(preds),
            "mae_v3b": float(mae_v3b),
            "mae_baseline": float(mae_bl),
            "improvement": float((mae_bl - mae_v3b) / mae_bl * 100),
            "coverage_80": float(cov),
        }
        print(f"  Year {test_year}: n={len(preds)} MAE_v3b={mae_v3b:.4f} MAE_bl={mae_bl:.4f} "
              f"({(mae_bl-mae_v3b)/mae_bl*100:+.1f}%) cov={cov:.1%}")

    return results_by_year


# ─────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────

def _apply_std_hitter(row: dict, std: dict) -> dict:
    """Apply standardization to a single hitter row (v3b)."""
    def z(val, key):
        if val is None:
            return 0.0
        return (val - std[f"{key}_mean"]) / std[f"{key}_sd"]

    row = dict(row)
    row["z_K"] = z(row["prev_K_pct"], "K")
    row["z_BB"] = z(row["prev_BB_pct"], "BB")
    row["z_K_BB"] = row["z_K"] * row["z_BB"]
    row["z_age"] = z(row["age_from_peak"], "age")
    row["z_log_pa"] = z(np.log(row["prev_pa"]), "log_pa")
    return row


def _apply_std_pitcher(row: dict, std: dict) -> dict:
    """Apply standardization to a single pitcher row (v3b)."""
    def z(val, key):
        if val is None:
            return 0.0
        return (val - std[f"{key}_mean"]) / std[f"{key}_sd"]

    row = dict(row)
    row["z_fip"] = z(row["prev_fip"], "fip")
    row["z_K"] = z(row["prev_K_pct"], "K")
    row["z_BB"] = z(row["prev_BB_pct"], "BB")
    row["z_K_BB"] = row["z_K"] * row["z_BB"]
    row["z_age"] = z(row["age_from_peak"], "age")
    row["z_log_ip"] = z(np.log(row["prev_ip"]), "log_ip")
    return row


def _predict_hitter_from_fit(fit, row: dict, n_samples: int = 5000) -> dict:
    """Generate posterior predictive samples for one hitter (v3b)."""
    rng = np.random.default_rng(42)

    alpha_all = fit.stan_variable("alpha")       # (draws*chains, L)
    beta_woba_all = fit.stan_variable("beta_woba")
    league_col = row["league_idx"] - 1
    alpha_s = alpha_all[:, league_col]
    beta_woba_s = beta_woba_all[:, league_col]

    params = {}
    for p in ["beta_K", "beta_BB", "beta_K_BB",
              "beta_age", "beta_catcher", "beta_middle_inf",
              "beta_second_year", "sigma_base", "gamma_pa"]:
        params[p] = fit.stan_variable(p)

    n = min(n_samples, len(alpha_s))
    idx = rng.choice(len(alpha_s), size=n, replace=n > len(alpha_s))

    mu = (alpha_s[idx]
          + beta_woba_s[idx] * row["prev_woba"]
          + params["beta_K"][idx] * row["z_K"]
          + params["beta_BB"][idx] * row["z_BB"]
          + params["beta_K_BB"][idx] * row["z_K_BB"]
          + params["beta_age"][idx] * row["z_age"]
          + params["beta_catcher"][idx] * row["is_catcher"]
          + params["beta_middle_inf"][idx] * row["is_middle_inf"]
          + params["beta_second_year"][idx] * row["is_second_year"])

    exponent = np.clip(params["gamma_pa"][idx] * row["z_log_pa"], -5.0, 2.0)
    sigma = params["sigma_base"][idx] * np.exp(exponent)
    y_pred = mu + rng.normal(0, sigma)
    y_pred = np.clip(y_pred, 0, None)  # wOBA >= 0

    return {
        "mean": float(np.mean(y_pred)),
        "std": float(np.std(y_pred)),
        "hdi_80": (float(np.percentile(y_pred, 10)), float(np.percentile(y_pred, 90))),
        "hdi_95": (float(np.percentile(y_pred, 2.5)), float(np.percentile(y_pred, 97.5))),
    }


def _predict_pitcher_from_fit(fit, row: dict, n_samples: int = 5000) -> dict:
    """Generate posterior predictive samples for one pitcher (v3b)."""
    rng = np.random.default_rng(42)

    alpha_K_all = fit.stan_variable("alpha_K")
    beta_K_all = fit.stan_variable("beta_K")
    alpha_BB_all = fit.stan_variable("alpha_BB")
    beta_BB_all = fit.stan_variable("beta_BB")
    league_col = row["league_idx"] - 1

    alpha_K_s = alpha_K_all[:, league_col]
    beta_K_s = beta_K_all[:, league_col]
    alpha_BB_s = alpha_BB_all[:, league_col]
    beta_BB_s = beta_BB_all[:, league_col]

    params = {}
    for p in ["beta_fip", "beta_K_BB",
              "beta_age", "beta_reliever", "beta_second_year",
              "sigma_base", "gamma_ip"]:
        params[p] = fit.stan_variable(p)

    n = min(n_samples, len(alpha_K_s))
    idx = rng.choice(len(alpha_K_s), size=n, replace=n > len(alpha_K_s))

    mu = (alpha_K_s[idx] + beta_K_s[idx] * row["prev_K_pct"]
          + alpha_BB_s[idx] + beta_BB_s[idx] * row["prev_BB_pct"]
          + params["beta_fip"][idx] * row["z_fip"]
          + params["beta_K_BB"][idx] * row["z_K_BB"]
          + params["beta_age"][idx] * row["z_age"]
          + params["beta_reliever"][idx] * row["is_reliever"]
          + params["beta_second_year"][idx] * row["is_second_year"])

    exponent = np.clip(params["gamma_ip"][idx] * row["z_log_ip"], -5.0, 2.0)
    sigma = params["sigma_base"][idx] * np.exp(exponent)
    y_pred = mu + rng.normal(0, sigma)
    y_pred = np.clip(y_pred, 0, None)  # ERA >= 0

    return {
        "mean": float(np.mean(y_pred)),
        "std": float(np.std(y_pred)),
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
    preds_v3b = np.array([p["pred_v3b"] for p in predictions])
    preds_bl = np.array([p["pred_baseline"] for p in predictions])

    mae_v3b = float(np.mean(np.abs(actuals - preds_v3b)))
    mae_bl = float(np.mean(np.abs(actuals - preds_bl)))

    coverage = np.mean([
        p["hdi_80"][0] <= p["actual"] <= p["hdi_80"][1]
        for p in predictions
    ])

    improvement = (mae_bl - mae_v3b) / mae_bl * 100

    result = {
        "stat": stat_name,
        "n": len(predictions),
        "mae_v3b": mae_v3b,
        "mae_baseline": mae_bl,
        "improvement_pct": float(improvement),
        "coverage_80": float(coverage),
        "predictions": predictions,
    }

    print(f"\n{'=' * 60}")
    print(f"LOO-CV Results -- {stat_name}")
    print(f"  N = {len(predictions)}")
    print(f"  MAE v3b:      {mae_v3b:.4f}")
    print(f"  MAE baseline: {mae_bl:.4f}")
    print(f"  Improvement:  {improvement:+.1f}%")
    print(f"  80% Coverage: {coverage:.1%}")
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
        "version": "v3b",
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
    with open(output_dir / "foreign_v3b_posterior.json", "w") as f:
        json.dump(params_out, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_dir / 'foreign_v3b_posterior.json'}")

    with open(output_dir / "foreign_v3b_backtest.json", "w") as f:
        bt = {}
        for k, v in backtest.items():
            if isinstance(v, dict) and "predictions" in v:
                bt[k] = {kk: vv for kk, vv in v.items() if kk != "predictions"}
            else:
                bt[k] = v
        json.dump(bt, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_dir / 'foreign_v3b_backtest.json'}")

    for ptype in ["hitter", "pitcher"]:
        key = f"loo_{ptype}"
        if key not in backtest or "predictions" not in backtest[key]:
            continue
        preds = backtest[key]["predictions"]
        csv_path = output_dir / f"foreign_v3b_{ptype}_loo_predictions.csv"
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
    """v1 baseline: predict league average (lg_avg)."""
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

    parser = argparse.ArgumentParser(description="Foreign player v3b model")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--loo-cv", action="store_true", help="Run LOO cross-validation")
    parser.add_argument("--expanding-cv", action="store_true", help="Run expanding-window CV")
    parser.add_argument("--fit-only", action="store_true", help="Fit full model only (no CV)")
    args = parser.parse_args()

    print("=" * 60)
    print("Foreign Player Prediction Model v3b")
    print("  Hitter: league-specific affine on raw wOBA")
    print("  Pitcher: league-specific affine on raw K% + BB%")
    print("=" * 60)

    print("\n[1/5] Building dataset...")
    hitters, pitchers = build_dataset()
    print(f"  Hitters: {len(hitters)} rows "
          f"(1st year: {sum(1 for d in hitters if d['is_second_year'] == 0)}, "
          f"2nd year: {sum(1 for d in hitters if d['is_second_year'] == 1)})")
    print(f"  Pitchers: {len(pitchers)} rows "
          f"(1st year: {sum(1 for d in pitchers if d['is_second_year'] == 0)}, "
          f"2nd year: {sum(1 for d in pitchers if d['is_second_year'] == 1)})")

    for ptype, data in [("Hitters", hitters), ("Pitchers", pitchers)]:
        from collections import Counter
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
        loo_h = loo_cv_hitters(hitters_copy, draws=args.draws, warmup=args.warmup)
        backtest["loo_hitter"] = loo_h
        _log_elapsed("LOO-CV hitters", t0)

        print("\n[4/5] Running LOO-CV for pitchers...")
        pitchers_copy = [dict(d) for d in pitchers]
        loo_p = loo_cv_pitchers(pitchers_copy, draws=args.draws, warmup=args.warmup)
        backtest["loo_pitcher"] = loo_p
        _log_elapsed("LOO-CV pitchers", t0)

    if args.expanding_cv:
        print("\n[3/5] Running expanding-window CV for hitters...")
        hitters_copy = [dict(d) for d in hitters]
        std_h, hitters_std = standardize_hitters(hitters_copy)
        exp_h = expanding_cv_hitters(hitters_std, draws=args.draws, warmup=args.warmup)
        backtest["expanding_hitter"] = exp_h
        _log_elapsed("expanding-CV hitters", t0)

        print("\n[4/5] Running expanding-window CV for pitchers...")
        pitchers_copy = [dict(d) for d in pitchers]
        std_p, pitchers_std = standardize_pitchers(pitchers_copy)
        exp_p = expanding_cv_pitchers(pitchers_std, draws=args.draws, warmup=args.warmup)
        backtest["expanding_pitcher"] = exp_p
        _log_elapsed("expanding-CV pitchers", t0)

    # Full model fit (always run)
    print("\n[5/5] Fitting full model on all data...")

    hitters_full = [dict(d) for d in hitters]
    std_h, hitters_full = standardize_hitters(hitters_full)
    stan_h = make_stan_data_hitters(hitters_full)
    print(f"  Hitter Stan data: N={stan_h['N']}, L={stan_h['L']}")
    fit_h = fit_model("hitter_foreign_v3b.stan", stan_h,
                      draws=args.draws, warmup=args.warmup)
    _log_elapsed("full model hitter Stan sampling", t0)
    print("\n  Hitter diagnostics:")
    check_diagnostics(fit_h)
    hitter_params = extract_posteriors_hitters(fit_h)

    pitchers_full = [dict(d) for d in pitchers]
    std_p, pitchers_full = standardize_pitchers(pitchers_full)
    stan_p = make_stan_data_pitchers(pitchers_full)
    print(f"\n  Pitcher Stan data: N={stan_p['N']}, L={stan_p['L']}")
    fit_p = fit_model("pitcher_foreign_v3b.stan", stan_p,
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
        print(f"  {k:25s} = {mean:+.4f} +/- {sd:.4f}  {sig}")

    print("\n" + "=" * 60)
    print("PITCHER POSTERIOR PARAMETERS")
    print("=" * 60)
    for k, (mean, sd) in sorted(pitcher_params.items()):
        sig = "***" if abs(mean) > 2 * sd else "**" if abs(mean) > 1.5 * sd else "*" if abs(mean) > sd else ""
        print(f"  {k:25s} = {mean:+.4f} +/- {sd:.4f}  {sig}")

    # Save everything
    print("\n[Save] Writing results...")
    save_results(hitter_params, pitcher_params, std_h, std_p,
                 backtest, DATA_MODEL)

    _log_elapsed("total", t0)
    print("\nDone!")


if __name__ == "__main__":
    main()
