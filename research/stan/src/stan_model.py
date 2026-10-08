"""Stan-based wOBA/ERA projection for NPB foreign players.

Uses cmdstanpy with K%/BB% skill features for improved prediction.
Compares v0 (single outcome stat) vs v1 (+ skill indicators) models.

Model (hitters):
    y = lg_avg + beta_woba * z_woba [+ beta_K * z_K + beta_BB * z_BB] + noise

Model (pitchers):
    y = lg_avg + beta_era * z_era [+ beta_fip * z_fip + beta_K * z_K + beta_BB * z_BB] + noise

Usage:
    python src/stan_model.py [--draws 2000] [--warmup 1000]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import arviz as az
import numpy as np
from cmdstanpy import CmdStanModel

def _log_elapsed(label: str, start: float, budget_min: int = 360):
    elapsed_min = (time.time() - start) / 60
    print(f"  [{label}] elapsed: {elapsed_min:.1f} min / {budget_min} min budget")
    if elapsed_min > budget_min * 0.8:
        print(f"  WARNING: {label} used {elapsed_min:.0f}/{budget_min} min "
              f"({elapsed_min / budget_min * 100:.0f}%) -- timeout risk!")


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FOREIGN_DIR = DATA_DIR / "foreign"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = DATA_DIR / "model"
STAN_DIR = Path(__file__).resolve().parent.parent / "models"

SPLIT_YEAR = 2020  # train: 2015-2019, test: 2020-2025


# ─── Data Loading ───────────────────────────────────────────────


def load_hitter_pairs() -> list[dict]:
    """Load hitters with prev_wOBA and NPB first-year wOBA.

    Also loads K%/BB% when available (for v1 model).
    """
    path = FOREIGN_DIR / "player_conversion_details.csv"
    pairs = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not (row.get("prev_wOBA") and row.get("npb_first_wOBA")
                    and row.get("wOBA_ratio")):
                continue
            try:
                d = {
                    "name": row["english_name"],
                    "npb_name": row["npb_name"],
                    "origin_league": row["origin_league"],
                    "year": int(row["npb_first_year"]),
                    "prev_wOBA": float(row["prev_wOBA"]),
                    "npb_wOBA": float(row["npb_first_wOBA"]),
                }
            except (ValueError, KeyError):
                continue
            # K%/BB% — optional, may be empty
            try:
                d["prev_K_pct"] = float(row["prev_K_pct"])
            except (ValueError, KeyError, TypeError):
                d["prev_K_pct"] = None
            try:
                d["prev_BB_pct"] = float(row["prev_BB_pct"])
            except (ValueError, KeyError, TypeError):
                d["prev_BB_pct"] = None
            pairs.append(d)
    return pairs


def load_pitcher_pairs() -> list[dict]:
    """Load pitchers with prev_ERA and NPB first-year ERA.

    Also loads FIP/K%/BB% when available (for v1 model).
    """
    path = FOREIGN_DIR / "player_conversion_details.csv"
    pairs = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not (row.get("prev_ERA") and row.get("npb_first_ERA")
                    and row.get("ERA_ratio")):
                continue
            try:
                d = {
                    "name": row["english_name"],
                    "npb_name": row["npb_name"],
                    "origin_league": row["origin_league"],
                    "year": int(row["npb_first_year"]),
                    "prev_ERA": float(row["prev_ERA"]),
                    "npb_ERA": float(row["npb_first_ERA"]),
                }
            except (ValueError, KeyError):
                continue
            # FIP/K%/BB% — optional
            for col, key in [("prev_FIP", "prev_FIP"),
                             ("prev_K_pct", "prev_K_pct"),
                             ("prev_BB_pct", "prev_BB_pct")]:
                try:
                    d[key] = float(row[col])
                except (ValueError, KeyError, TypeError):
                    d[key] = None
            pairs.append(d)
    return pairs


def load_npb_league_avg(
    stat: str, min_threshold: int | float = 100,
) -> dict[int, float]:
    """Load NPB league-average stat per year."""
    if stat == "wOBA":
        path = RAW_DIR / "npb_sabermetrics_2015_2025.csv"
        val_col, filter_col, filter_val = "wOBA", "PA", min_threshold
    else:
        path = RAW_DIR / "npb_pitchers_2015_2025.csv"
        val_col, filter_col, filter_val = "ERA", "IP", min_threshold

    yearly: dict[int, list[float]] = defaultdict(list)
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                year = int(row["year"])
                val = float(row[val_col])
                filt = float(row[filter_col])
                if filt >= filter_val:
                    yearly[year].append(val)
            except (ValueError, KeyError):
                continue
    return {y: float(np.mean(vals)) for y, vals in yearly.items()}


# ─── Standardization ───────────────────────────────────────────


def compute_standardization(
    values: list[float],
) -> tuple[float, float]:
    """Compute mean and std from training set for z-score standardization."""
    arr = np.array(values)
    return float(np.mean(arr)), max(float(np.std(arr, ddof=1)), 1e-8)


def standardize(
    values: list[float | None],
    mean: float,
    sd: float,
) -> np.ndarray:
    """Standardize values, replacing None with 0 (= training mean)."""
    return np.array([
        (v - mean) / sd if v is not None else 0.0
        for v in values
    ])


# ─── Model Fitting ──────────────────────────────────────────────


def fit_hitter_model(
    train_data: list[dict],
    league_avg_woba: dict[int, float],
    n_features: int = 3,
    draws: int = 2000,
    warmup: int = 1000,
) -> tuple[CmdStanModel, object, dict]:
    """Fit Stan hitter model. n_features=1 for v0, n_features=3 for v1."""
    prev_woba = [d["prev_wOBA"] for d in train_data]
    npb_woba = np.array([d["npb_wOBA"] for d in train_data])
    lg_avg = np.array([
        league_avg_woba.get(d["year"], 0.310) for d in train_data
    ])
    n = len(train_data)

    # Standardization params (from training set)
    woba_mean, woba_sd = compute_standardization(prev_woba)
    z_woba = standardize(prev_woba, woba_mean, woba_sd)

    # K%/BB% standardization
    k_vals = [d.get("prev_K_pct") for d in train_data]
    bb_vals = [d.get("prev_BB_pct") for d in train_data]

    k_available = [v for v in k_vals if v is not None]
    bb_available = [v for v in bb_vals if v is not None]

    k_mean, k_sd = compute_standardization(k_available) if k_available else (0.0, 1.0)
    bb_mean, bb_sd = compute_standardization(bb_available) if bb_available else (0.0, 1.0)

    z_K = standardize(k_vals, k_mean, k_sd)
    z_BB = standardize(bb_vals, bb_mean, bb_sd)

    stan_data = {
        "N": n,
        "y": npb_woba.tolist(),
        "lg_avg": lg_avg.tolist(),
        "z_woba": z_woba.tolist(),
        "z_K": z_K.tolist(),
        "z_BB": z_BB.tolist(),
        "N_features": n_features,
    }

    label = "v0" if n_features == 1 else "v1"
    print(f"\nFitting hitter Stan model {label} (n={n}, draws={draws}, warmup={warmup})...")

    model = CmdStanModel(stan_file=str(STAN_DIR / "hitter.stan"))
    fit = model.sample(
        data=stan_data,
        iter_sampling=draws,
        iter_warmup=warmup,
        chains=4,
        seed=42,
        show_console=False,
    )

    std_params = {
        "woba_mean": woba_mean, "woba_sd": woba_sd,
        "k_mean": k_mean, "k_sd": k_sd,
        "bb_mean": bb_mean, "bb_sd": bb_sd,
    }

    return model, fit, std_params


def fit_pitcher_model(
    train_data: list[dict],
    league_avg_era: dict[int, float],
    n_features: int = 4,
    draws: int = 2000,
    warmup: int = 1000,
) -> tuple[CmdStanModel, object, dict]:
    """Fit Stan pitcher model. n_features=1 for v0, n_features=4 for v1."""
    prev_era = [d["prev_ERA"] for d in train_data]
    npb_era = np.array([d["npb_ERA"] for d in train_data])
    lg_avg = np.array([
        league_avg_era.get(d["year"], 3.50) for d in train_data
    ])
    n = len(train_data)

    # Standardization params
    era_mean, era_sd = compute_standardization(prev_era)
    z_era = standardize(prev_era, era_mean, era_sd)

    # FIP/K%/BB%
    fip_vals = [d.get("prev_FIP") for d in train_data]
    k_vals = [d.get("prev_K_pct") for d in train_data]
    bb_vals = [d.get("prev_BB_pct") for d in train_data]

    fip_available = [v for v in fip_vals if v is not None]
    k_available = [v for v in k_vals if v is not None]
    bb_available = [v for v in bb_vals if v is not None]

    fip_mean, fip_sd = compute_standardization(fip_available) if fip_available else (0.0, 1.0)
    k_mean, k_sd = compute_standardization(k_available) if k_available else (0.0, 1.0)
    bb_mean, bb_sd = compute_standardization(bb_available) if bb_available else (0.0, 1.0)

    z_fip = standardize(fip_vals, fip_mean, fip_sd)
    z_K = standardize(k_vals, k_mean, k_sd)
    z_BB = standardize(bb_vals, bb_mean, bb_sd)

    stan_data = {
        "N": n,
        "y": npb_era.tolist(),
        "lg_avg": lg_avg.tolist(),
        "z_era": z_era.tolist(),
        "z_fip": z_fip.tolist(),
        "z_K": z_K.tolist(),
        "z_BB": z_BB.tolist(),
        "N_features": n_features,
    }

    label = "v0" if n_features == 1 else "v1"
    print(f"\nFitting pitcher Stan model {label} (n={n}, draws={draws}, warmup={warmup})...")

    model = CmdStanModel(stan_file=str(STAN_DIR / "pitcher.stan"))
    fit = model.sample(
        data=stan_data,
        iter_sampling=draws,
        iter_warmup=warmup,
        chains=4,
        seed=42,
        show_console=False,
    )

    std_params = {
        "era_mean": era_mean, "era_sd": era_sd,
        "fip_mean": fip_mean, "fip_sd": fip_sd,
        "k_mean": k_mean, "k_sd": k_sd,
        "bb_mean": bb_mean, "bb_sd": bb_sd,
    }

    return model, fit, std_params


# ─── Prediction ─────────────────────────────────────────────────


def predict_new_player_hitter(
    fit, std_params: dict,
    prev_woba: float,
    prev_K_pct: float | None,
    prev_BB_pct: float | None,
    league_avg: float,
    n_features: int = 3,
    n_samples: int = 4000,
) -> dict[str, float]:
    """Posterior predictive for a new hitter."""
    draws = fit.draws_pd()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(draws), size=n_samples, replace=True)

    beta_woba = draws["beta_woba"].values[idx]
    sigma = draws["sigma"].values[idx]

    z_w = (prev_woba - std_params["woba_mean"]) / std_params["woba_sd"]

    mu = league_avg + beta_woba * z_w
    if n_features >= 3:
        beta_K = draws["beta_K"].values[idx]
        beta_BB = draws["beta_BB"].values[idx]
        z_k = ((prev_K_pct - std_params["k_mean"]) / std_params["k_sd"]
               if prev_K_pct is not None else 0.0)
        z_bb = ((prev_BB_pct - std_params["bb_mean"]) / std_params["bb_sd"]
                if prev_BB_pct is not None else 0.0)
        mu = mu + beta_K * z_k + beta_BB * z_bb

    predicted = rng.normal(mu, sigma)

    return {
        "mean": float(np.mean(predicted)),
        "median": float(np.median(predicted)),
        "std": float(np.std(predicted)),
        "hdi_80_low": float(np.percentile(predicted, 10)),
        "hdi_80_high": float(np.percentile(predicted, 90)),
    }


def predict_new_player_pitcher(
    fit, std_params: dict,
    prev_era: float,
    prev_fip: float | None,
    prev_K_pct: float | None,
    prev_BB_pct: float | None,
    league_avg: float,
    n_features: int = 4,
    n_samples: int = 4000,
) -> dict[str, float]:
    """Posterior predictive for a new pitcher."""
    draws = fit.draws_pd()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(draws), size=n_samples, replace=True)

    beta_era = draws["beta_era"].values[idx]
    sigma = draws["sigma"].values[idx]

    z_e = (prev_era - std_params["era_mean"]) / std_params["era_sd"]

    mu = league_avg + beta_era * z_e
    if n_features >= 4:
        beta_fip = draws["beta_fip"].values[idx]
        beta_K = draws["beta_K"].values[idx]
        beta_BB = draws["beta_BB"].values[idx]
        z_f = ((prev_fip - std_params["fip_mean"]) / std_params["fip_sd"]
               if prev_fip is not None else 0.0)
        z_k = ((prev_K_pct - std_params["k_mean"]) / std_params["k_sd"]
               if prev_K_pct is not None else 0.0)
        z_bb = ((prev_BB_pct - std_params["bb_mean"]) / std_params["bb_sd"]
                if prev_BB_pct is not None else 0.0)
        mu = mu + beta_fip * z_f + beta_K * z_k + beta_BB * z_bb

    predicted = rng.normal(mu, sigma)

    return {
        "mean": float(np.mean(predicted)),
        "median": float(np.median(predicted)),
        "std": float(np.std(predicted)),
        "hdi_80_low": float(np.percentile(predicted, 10)),
        "hdi_80_high": float(np.percentile(predicted, 90)),
    }


# ─── Backtest ───────────────────────────────────────────────────


def backtest_hitters(
    fit, std_params: dict,
    test_data: list[dict],
    league_avg_woba: dict[int, float],
    n_features: int = 3,
) -> dict:
    """Backtest hitter model: baseline (league avg) vs Stan."""
    baseline_errors = []
    stan_errors = []
    predictions = []

    for d in test_data:
        actual = d["npb_wOBA"]
        lg_avg = league_avg_woba.get(d["year"], 0.310)

        baseline_pred = lg_avg
        stan_pred_dict = predict_new_player_hitter(
            fit, std_params,
            prev_woba=d["prev_wOBA"],
            prev_K_pct=d.get("prev_K_pct"),
            prev_BB_pct=d.get("prev_BB_pct"),
            league_avg=lg_avg,
            n_features=n_features,
        )
        stan_pred = stan_pred_dict["mean"]

        baseline_errors.append(abs(actual - baseline_pred))
        stan_errors.append(abs(actual - stan_pred))

        in_80 = stan_pred_dict["hdi_80_low"] <= actual <= stan_pred_dict["hdi_80_high"]
        predictions.append({
            "name": d["name"],
            "npb_name": d["npb_name"],
            "year": d["year"],
            "origin_league": d["origin_league"],
            "prev_wOBA": f"{d['prev_wOBA']:.4f}",
            "actual_wOBA": f"{actual:.4f}",
            "league_avg": f"{lg_avg:.4f}",
            "baseline_pred": f"{baseline_pred:.4f}",
            "stan_pred": f"{stan_pred:.4f}",
            "hdi_80": f"[{stan_pred_dict['hdi_80_low']:.4f}, {stan_pred_dict['hdi_80_high']:.4f}]",
            "in_80": in_80,
        })

    n_test = len(test_data)
    coverage_80 = sum(1 for p in predictions if p["in_80"]) / n_test if n_test else 0

    return {
        "baseline_mae": float(np.mean(baseline_errors)),
        "stan_mae": float(np.mean(stan_errors)),
        "n_test": n_test,
        "coverage_80": coverage_80,
        "predictions": predictions,
    }


def backtest_pitchers(
    fit, std_params: dict,
    test_data: list[dict],
    league_avg_era: dict[int, float],
    n_features: int = 4,
) -> dict:
    """Backtest pitcher model: baseline (league avg) vs Stan."""
    baseline_errors = []
    stan_errors = []
    predictions = []

    for d in test_data:
        actual = d["npb_ERA"]
        lg_avg = league_avg_era.get(d["year"], 3.50)

        baseline_pred = lg_avg
        stan_pred_dict = predict_new_player_pitcher(
            fit, std_params,
            prev_era=d["prev_ERA"],
            prev_fip=d.get("prev_FIP"),
            prev_K_pct=d.get("prev_K_pct"),
            prev_BB_pct=d.get("prev_BB_pct"),
            league_avg=lg_avg,
            n_features=n_features,
        )
        stan_pred = stan_pred_dict["mean"]

        baseline_errors.append(abs(actual - baseline_pred))
        stan_errors.append(abs(actual - stan_pred))

        in_80 = stan_pred_dict["hdi_80_low"] <= actual <= stan_pred_dict["hdi_80_high"]
        predictions.append({
            "name": d["name"],
            "npb_name": d["npb_name"],
            "year": d["year"],
            "origin_league": d["origin_league"],
            "prev_ERA": f"{d['prev_ERA']:.2f}",
            "actual_ERA": f"{actual:.2f}",
            "league_avg": f"{lg_avg:.2f}",
            "baseline_pred": f"{baseline_pred:.2f}",
            "stan_pred": f"{stan_pred:.2f}",
            "hdi_80": f"[{stan_pred_dict['hdi_80_low']:.2f}, {stan_pred_dict['hdi_80_high']:.2f}]",
            "in_80": in_80,
        })

    n_test = len(test_data)
    coverage_80 = sum(1 for p in predictions if p["in_80"]) / n_test if n_test else 0

    return {
        "baseline_mae": float(np.mean(baseline_errors)),
        "stan_mae": float(np.mean(stan_errors)),
        "n_test": n_test,
        "coverage_80": coverage_80,
        "predictions": predictions,
    }


# ─── Diagnostics ────────────────────────────────────────────────


def check_diagnostics(fit) -> dict:
    """Check MCMC diagnostics: R-hat, divergences, ESS."""
    summary = fit.summary()
    rhat_max = float(summary["R_hat"].max())
    n_eff_min = float(summary["N_Eff"].min()) if "N_Eff" in summary.columns else 0
    diags = fit.diagnose()

    return {
        "rhat_max": rhat_max,
        "n_eff_min": n_eff_min,
        "diagnostics": diags,
    }


# ─── Output ─────────────────────────────────────────────────────


def write_stan_outputs(results: dict) -> None:
    """Write Stan model comparison results."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Main comparison JSON
    comparison = {}
    for player_type in ["hitter", "pitcher"]:
        type_results = results.get(player_type)
        if not type_results:
            continue
        comparison[player_type] = {}
        for variant, bt in type_results.items():
            if bt is None:
                continue
            comparison[player_type][f"{variant}_mae"] = bt["stan_mae"]
            comparison[player_type][f"{variant}_coverage_80"] = bt["coverage_80"]
        # Always include baseline from any variant
        for variant, bt in type_results.items():
            if bt is not None:
                comparison[player_type]["baseline_mae"] = bt["baseline_mae"]
                comparison[player_type]["n_test"] = bt["n_test"]
                break

    with open(MODEL_DIR / "stan_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison written to {MODEL_DIR / 'stan_comparison.json'}")

    # Per-variant prediction CSVs
    for player_type in ["hitter", "pitcher"]:
        type_results = results.get(player_type, {})
        for variant, bt in type_results.items():
            if bt is None or not bt.get("predictions"):
                continue
            preds = bt["predictions"]
            keys = [k for k in preds[0].keys() if k != "in_80"]
            path = MODEL_DIR / f"stan_{player_type}_{variant}_predictions.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(preds)


# ─── Main ───────────────────────────────────────────────────────


def main() -> None:
    t0 = time.time()

    parser = argparse.ArgumentParser(description="Stan-based NPB foreign player projection")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=1000)
    args = parser.parse_args()

    # Load data
    hitter_pairs = load_hitter_pairs()
    pitcher_pairs = load_pitcher_pairs()
    league_avg_woba = load_npb_league_avg("wOBA", min_threshold=100)
    league_avg_era = load_npb_league_avg("ERA", min_threshold=30)

    print(f"Total hitter pairs: {len(hitter_pairs)}")
    print(f"Total pitcher pairs: {len(pitcher_pairs)}")

    # Split train/test
    h_train = [d for d in hitter_pairs if d["year"] < SPLIT_YEAR]
    h_test = [d for d in hitter_pairs if d["year"] >= SPLIT_YEAR]
    p_train = [d for d in pitcher_pairs if d["year"] < SPLIT_YEAR]
    p_test = [d for d in pitcher_pairs if d["year"] >= SPLIT_YEAR]

    print(f"\nHitters — train: {len(h_train)}, test: {len(h_test)}")
    print(f"Pitchers — train: {len(p_train)}, test: {len(p_test)}")

    _log_elapsed("data loading", t0)

    results: dict = {"hitter": {}, "pitcher": {}}

    # === Hitter Models ===
    if h_train and h_test:
        for label, n_feat in [("v0", 1), ("v1", 3)]:
            print(f"\n{'=' * 50}")
            print(f"Hitter model {label} (n_features={n_feat})")
            print(f"{'=' * 50}")

            _, fit, std_params = fit_hitter_model(
                h_train, league_avg_woba,
                n_features=n_feat, draws=args.draws, warmup=args.warmup,
            )
            _log_elapsed(f"hitter Stan sampling ({label})", t0)

            # Diagnostics
            diag = check_diagnostics(fit)
            print(f"  R-hat max: {diag['rhat_max']:.4f}")
            print(f"  Min ESS: {diag['n_eff_min']:.0f}")

            # Print parameter estimates
            summary = fit.summary()
            for param in ["beta_woba", "beta_K", "beta_BB", "sigma"]:
                if param in summary.index:
                    row = summary.loc[param]
                    print(f"  {param}: {row['Mean']:.4f} [{row['5%']:.4f}, {row['95%']:.4f}]")

            # Backtest
            bt = backtest_hitters(fit, std_params, h_test, league_avg_woba, n_feat)
            results["hitter"][label] = bt

            print(f"\n  Backtest ({label}):")
            print(f"    Baseline MAE: {bt['baseline_mae']:.4f}")
            print(f"    Stan MAE:     {bt['stan_mae']:.4f}")
            print(f"    80% coverage: {bt['coverage_80']:.1%}")
            imp = (bt["baseline_mae"] - bt["stan_mae"]) / bt["baseline_mae"] * 100
            print(f"    vs baseline:  {imp:+.1f}%")

    # === Pitcher Models ===
    if p_train and p_test:
        for label, n_feat in [("v0", 1), ("v1", 4)]:
            print(f"\n{'=' * 50}")
            print(f"Pitcher model {label} (n_features={n_feat})")
            print(f"{'=' * 50}")

            _, fit, std_params = fit_pitcher_model(
                p_train, league_avg_era,
                n_features=n_feat, draws=args.draws, warmup=args.warmup,
            )
            _log_elapsed(f"pitcher Stan sampling ({label})", t0)

            # Diagnostics
            diag = check_diagnostics(fit)
            print(f"  R-hat max: {diag['rhat_max']:.4f}")
            print(f"  Min ESS: {diag['n_eff_min']:.0f}")

            # Print parameter estimates
            summary = fit.summary()
            for param in ["beta_era", "beta_fip", "beta_K", "beta_BB", "sigma"]:
                if param in summary.index:
                    row = summary.loc[param]
                    print(f"  {param}: {row['Mean']:.4f} [{row['5%']:.4f}, {row['95%']:.4f}]")

            # Backtest
            bt = backtest_pitchers(fit, std_params, p_test, league_avg_era, n_feat)
            results["pitcher"][label] = bt

            print(f"\n  Backtest ({label}):")
            print(f"    Baseline MAE: {bt['baseline_mae']:.2f}")
            print(f"    Stan MAE:     {bt['stan_mae']:.2f}")
            print(f"    80% coverage: {bt['coverage_80']:.1%}")
            imp = (bt["baseline_mae"] - bt["stan_mae"]) / bt["baseline_mae"] * 100
            print(f"    vs baseline:  {imp:+.1f}%")

    # Write outputs
    _log_elapsed("all models complete", t0)
    write_stan_outputs(results)

    # Final summary
    print(f"\n{'=' * 60}")
    print("FINAL COMPARISON")
    print(f"{'=' * 60}")
    for ptype in ["hitter", "pitcher"]:
        if ptype not in results or not results[ptype]:
            continue
        stat = "wOBA" if ptype == "hitter" else "ERA"
        print(f"\n{ptype.title()} ({stat}):")
        for variant in ["v0", "v1"]:
            bt = results[ptype].get(variant)
            if bt:
                imp = (bt["baseline_mae"] - bt["stan_mae"]) / bt["baseline_mae"] * 100
                marker = " <-- WINNER" if bt["stan_mae"] < bt["baseline_mae"] else ""
                print(f"  {variant} MAE: {bt['stan_mae']:.4f} (baseline: {bt['baseline_mae']:.4f}, {imp:+.1f}%){marker}")


if __name__ == "__main__":
    main()
