"""Bayesian wOBA/ERA projection for NPB foreign players using PyMC.

Step 3: Shrinkage model — blends conversion-factor prediction with league average.

Model (hitters):
    cf_mu ~ Normal(1.2, 0.3)            # population conversion factor
    cf_sigma ~ HalfNormal(0.3)           # player-level variation
    cf_i ~ Normal(cf_mu, cf_sigma)       # per-player conversion factor
    w ~ Beta(2, 2)                       # shrinkage weight (learned)
    mu_npb_i = w * (prev_wOBA_i * cf_i) + (1-w) * league_avg_i
    sigma_obs ~ HalfNormal(0.1)          # residual noise
    wOBA_obs_i ~ Normal(mu_npb_i, sigma_obs)

Key insight: w controls how much to trust prev_stats vs league average.
If w → 0, model reduces to baseline (league avg). If w → 1, pure conversion factor.

Usage:
    python src/bayesian_model.py [--draws 2000] [--tune 1000]
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
import pymc as pm

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

SPLIT_YEAR = 2020  # train: 2015-2019, test: 2020-2025


# ─── Data Loading ───────────────────────────────────────────────


def load_hitter_pairs() -> list[dict]:
    """Load hitters with both prev_wOBA and NPB first-year wOBA."""
    path = FOREIGN_DIR / "player_conversion_details.csv"
    pairs = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("prev_wOBA") and row.get("npb_first_wOBA") and row.get("wOBA_ratio"):
                try:
                    pairs.append({
                        "name": row["english_name"],
                        "npb_name": row["npb_name"],
                        "origin_league": row["origin_league"],
                        "year": int(row["npb_first_year"]),
                        "prev_wOBA": float(row["prev_wOBA"]),
                        "npb_wOBA": float(row["npb_first_wOBA"]),
                    })
                except (ValueError, KeyError):
                    continue
    return pairs


def load_pitcher_pairs() -> list[dict]:
    """Load pitchers with both prev_ERA and NPB first-year ERA."""
    path = FOREIGN_DIR / "player_conversion_details.csv"
    pairs = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("prev_ERA") and row.get("npb_first_ERA") and row.get("ERA_ratio"):
                try:
                    pairs.append({
                        "name": row["english_name"],
                        "npb_name": row["npb_name"],
                        "origin_league": row["origin_league"],
                        "year": int(row["npb_first_year"]),
                        "prev_ERA": float(row["prev_ERA"]),
                        "npb_ERA": float(row["npb_first_ERA"]),
                    })
                except (ValueError, KeyError):
                    continue
    return pairs


def load_npb_league_avg(stat: str, min_threshold: int | float = 100) -> dict[int, float]:
    """Load NPB league-average stat per year.

    stat: 'wOBA' (from sabermetrics) or 'ERA' (from pitchers)
    """
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


# ─── Model Fitting ──────────────────────────────────────────────


def fit_hitter_model(
    train_data: list[dict],
    league_avg_woba: dict[int, float],
    draws: int = 2000,
    tune: int = 1000,
) -> tuple[pm.Model, az.InferenceData]:
    """Fit shrinkage model for hitters: blend(prev_wOBA * cf, league_avg)."""
    prev_woba = np.array([d["prev_wOBA"] for d in train_data])
    npb_woba = np.array([d["npb_wOBA"] for d in train_data])
    league_avg = np.array([league_avg_woba.get(d["year"], 0.310) for d in train_data])
    n = len(train_data)

    print(f"\nFitting hitter shrinkage model (n={n}, draws={draws}, tune={tune})...")

    with pm.Model() as model:
        # Conversion factor hierarchy
        cf_mu = pm.Normal("cf_mu", mu=1.2, sigma=0.3)
        cf_sigma = pm.HalfNormal("cf_sigma", sigma=0.3)
        cf_i = pm.Normal("cf_i", mu=cf_mu, sigma=cf_sigma, shape=n)

        # Shrinkage weight: 0 = league avg, 1 = pure conversion
        w = pm.Beta("w", alpha=2, beta=2)

        # Blended prediction
        cf_pred = prev_woba * cf_i
        mu_npb = w * cf_pred + (1 - w) * league_avg

        # Residual noise
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.1)

        # Likelihood
        pm.Normal("wOBA_obs", mu=mu_npb, sigma=sigma_obs, observed=npb_woba)

        trace = pm.sample(
            draws=draws, tune=tune, cores=2,
            random_seed=42, return_inferencedata=True,
        )

    return model, trace


def fit_pitcher_model(
    train_data: list[dict],
    league_avg_era: dict[int, float],
    draws: int = 2000,
    tune: int = 1000,
) -> tuple[pm.Model, az.InferenceData]:
    """Fit shrinkage model for pitchers: blend(prev_ERA * cf, league_avg)."""
    prev_era = np.array([d["prev_ERA"] for d in train_data])
    npb_era = np.array([d["npb_ERA"] for d in train_data])
    league_avg = np.array([league_avg_era.get(d["year"], 3.50) for d in train_data])
    n = len(train_data)

    print(f"\nFitting pitcher shrinkage model (n={n}, draws={draws}, tune={tune})...")

    with pm.Model() as model:
        cf_mu = pm.Normal("cf_mu", mu=0.6, sigma=0.3)
        cf_sigma = pm.HalfNormal("cf_sigma", sigma=0.3)
        cf_i = pm.Normal("cf_i", mu=cf_mu, sigma=cf_sigma, shape=n)

        w = pm.Beta("w", alpha=2, beta=2)

        cf_pred = prev_era * cf_i
        mu_npb = w * cf_pred + (1 - w) * league_avg

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)

        pm.Normal("ERA_obs", mu=mu_npb, sigma=sigma_obs, observed=npb_era)

        trace = pm.sample(
            draws=draws, tune=tune, cores=2,
            random_seed=42, return_inferencedata=True,
        )

    return model, trace


# ─── Prediction ─────────────────────────────────────────────────


def predict_new_player(
    trace: az.InferenceData,
    prev_stat: float,
    league_avg_value: float,
    n_samples: int = 4000,
) -> dict[str, float]:
    """Posterior predictive for a new player using shrinkage model."""
    cf_mu_s = trace.posterior["cf_mu"].values.flatten()
    cf_sigma_s = trace.posterior["cf_sigma"].values.flatten()
    w_s = trace.posterior["w"].values.flatten()
    sigma_obs_s = trace.posterior["sigma_obs"].values.flatten()

    rng = np.random.default_rng(42)
    idx = rng.choice(len(cf_mu_s), size=n_samples, replace=True)

    cf_new = rng.normal(cf_mu_s[idx], cf_sigma_s[idx])
    cf_pred = prev_stat * cf_new
    blended = w_s[idx] * cf_pred + (1 - w_s[idx]) * league_avg_value
    noise = rng.normal(0, sigma_obs_s[idx])
    predicted = blended + noise

    return {
        "mean": float(np.mean(predicted)),
        "median": float(np.median(predicted)),
        "std": float(np.std(predicted)),
        "hdi_80_low": float(np.percentile(predicted, 10)),
        "hdi_80_high": float(np.percentile(predicted, 90)),
        "hdi_95_low": float(np.percentile(predicted, 2.5)),
        "hdi_95_high": float(np.percentile(predicted, 97.5)),
    }


# ─── Backtest ───────────────────────────────────────────────────


def backtest(
    trace: az.InferenceData,
    test_data: list[dict],
    stat_key: str,
    prev_key: str,
    league_avg: dict[int, float],
    default_avg: float,
) -> dict:
    """Backtest: baseline vs raw CF vs Bayesian shrinkage."""
    cf_mu_median = float(np.median(trace.posterior["cf_mu"].values))
    w_median = float(np.median(trace.posterior["w"].values))

    baseline_errors = []
    raw_cf_errors = []
    bayes_errors = []
    predictions = []

    for d in test_data:
        actual = d[stat_key]
        prev = d[prev_key]
        year = d["year"]
        lg_avg = league_avg.get(year, default_avg)

        # Baseline: league average
        baseline_pred = lg_avg

        # Raw conversion factor (no shrinkage)
        raw_pred = prev * cf_mu_median

        # Bayesian shrinkage
        bayes = predict_new_player(trace, prev, lg_avg)
        bayes_pred = bayes["mean"]

        baseline_errors.append(abs(actual - baseline_pred))
        raw_cf_errors.append(abs(actual - raw_pred))
        bayes_errors.append(abs(actual - bayes_pred))

        predictions.append({
            "name": d["name"],
            "npb_name": d["npb_name"],
            "year": year,
            "origin_league": d["origin_league"],
            f"prev_{stat_key}": f"{prev:.4f}",
            f"actual_{stat_key}": f"{actual:.4f}",
            "league_avg": f"{lg_avg:.4f}",
            "baseline_pred": f"{baseline_pred:.4f}",
            "raw_cf_pred": f"{raw_pred:.4f}",
            "bayes_pred": f"{bayes_pred:.4f}",
            "bayes_hdi_80": f"[{bayes['hdi_80_low']:.4f}, {bayes['hdi_80_high']:.4f}]",
            "bayes_hdi_95": f"[{bayes['hdi_95_low']:.4f}, {bayes['hdi_95_high']:.4f}]",
        })

    results = {
        "baseline_mae": float(np.mean(baseline_errors)),
        "raw_cf_mae": float(np.mean(raw_cf_errors)),
        "bayes_mae": float(np.mean(bayes_errors)),
        "w_median": w_median,
        "cf_mu_median": cf_mu_median,
        "n_test": len(test_data),
        "predictions": predictions,
    }

    # Coverage
    in_80 = sum(
        1 for d, p in zip(test_data, predictions)
        if float(p["bayes_hdi_80"].strip("[]").split(",")[0]) <= d[stat_key]
        <= float(p["bayes_hdi_80"].strip("[]").split(",")[1])
    )
    in_95 = sum(
        1 for d, p in zip(test_data, predictions)
        if float(p["bayes_hdi_95"].strip("[]").split(",")[0]) <= d[stat_key]
        <= float(p["bayes_hdi_95"].strip("[]").split(",")[1])
    )
    results["coverage_80"] = in_80 / len(test_data)
    results["coverage_95"] = in_95 / len(test_data)

    return results


# ─── Output ─────────────────────────────────────────────────────


def write_outputs(
    hitter_trace: az.InferenceData | None,
    pitcher_trace: az.InferenceData | None,
    hitter_bt: dict | None,
    pitcher_bt: dict | None,
) -> None:
    """Write model outputs."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    var_names = ["cf_mu", "cf_sigma", "w", "sigma_obs"]

    for label, trace, bt in [
        ("hitter", hitter_trace, hitter_bt),
        ("pitcher", pitcher_trace, pitcher_bt),
    ]:
        if trace is not None:
            summary = az.summary(trace, var_names=var_names)
            summary.to_csv(MODEL_DIR / f"{label}_trace_summary.csv")
            print(f"\n{label.title()} trace summary:\n{summary}")

        if bt is not None:
            summary = {k: v for k, v in bt.items() if k != "predictions"}
            with open(MODEL_DIR / f"{label}_backtest_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            preds = bt["predictions"]
            if preds:
                keys = list(preds[0].keys())
                with open(MODEL_DIR / f"{label}_backtest_predictions.csv", "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(preds)


# ─── Main ───────────────────────────────────────────────────────


def main() -> None:
    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=1000)
    args = parser.parse_args()

    # Load data
    hitter_pairs = load_hitter_pairs()
    pitcher_pairs = load_pitcher_pairs()
    league_avg_woba = load_npb_league_avg("wOBA", min_threshold=100)
    league_avg_era = load_npb_league_avg("ERA", min_threshold=30)

    print(f"Total hitter pairs: {len(hitter_pairs)}")
    print(f"Total pitcher pairs: {len(pitcher_pairs)}")

    # Split
    h_train = [d for d in hitter_pairs if d["year"] < SPLIT_YEAR]
    h_test = [d for d in hitter_pairs if d["year"] >= SPLIT_YEAR]
    p_train = [d for d in pitcher_pairs if d["year"] < SPLIT_YEAR]
    p_test = [d for d in pitcher_pairs if d["year"] >= SPLIT_YEAR]

    print(f"\nHitters — train: {len(h_train)}, test: {len(h_test)}")
    print(f"Pitchers — train: {len(p_train)}, test: {len(p_test)}")

    _log_elapsed("data loading", t0)

    # === Hitter Model ===
    hitter_trace = None
    hitter_bt = None
    if h_train:
        _, hitter_trace = fit_hitter_model(
            h_train, league_avg_woba, draws=args.draws, tune=args.tune,
        )
        _log_elapsed("hitter PyMC sampling", t0)
        if h_test:
            print("\n=== Hitter Backtest (Shrinkage) ===")
            hitter_bt = backtest(
                hitter_trace, h_test,
                stat_key="npb_wOBA", prev_key="prev_wOBA",
                league_avg=league_avg_woba, default_avg=0.310,
            )
            print(f"  w (shrinkage): {hitter_bt['w_median']:.3f}")
            print(f"  Baseline MAE (league avg): {hitter_bt['baseline_mae']:.4f}")
            print(f"  Raw CF MAE:                {hitter_bt['raw_cf_mae']:.4f}")
            print(f"  Bayesian MAE:              {hitter_bt['bayes_mae']:.4f}")
            print(f"  80% coverage: {hitter_bt['coverage_80']:.1%}")
            print(f"  95% coverage: {hitter_bt['coverage_95']:.1%}")
            imp = (hitter_bt["baseline_mae"] - hitter_bt["bayes_mae"]) / hitter_bt["baseline_mae"] * 100
            print(f"  Improvement vs baseline: {imp:+.1f}%")

        _log_elapsed("hitter backtest", t0)

    # === Pitcher Model ===
    pitcher_trace = None
    pitcher_bt = None
    if p_train:
        _, pitcher_trace = fit_pitcher_model(
            p_train, league_avg_era, draws=args.draws, tune=args.tune,
        )
        _log_elapsed("pitcher PyMC sampling", t0)
        if p_test:
            print("\n=== Pitcher Backtest (Shrinkage) ===")
            pitcher_bt = backtest(
                pitcher_trace, p_test,
                stat_key="npb_ERA", prev_key="prev_ERA",
                league_avg=league_avg_era, default_avg=3.50,
            )
            print(f"  w (shrinkage): {pitcher_bt['w_median']:.3f}")
            print(f"  Baseline MAE (global avg): {pitcher_bt['baseline_mae']:.2f}")
            print(f"  Raw CF MAE:                {pitcher_bt['raw_cf_mae']:.2f}")
            print(f"  Bayesian MAE:              {pitcher_bt['bayes_mae']:.2f}")
            print(f"  80% coverage: {pitcher_bt['coverage_80']:.1%}")
            print(f"  95% coverage: {pitcher_bt['coverage_95']:.1%}")
            imp = (pitcher_bt["baseline_mae"] - pitcher_bt["bayes_mae"]) / pitcher_bt["baseline_mae"] * 100
            print(f"  Improvement vs baseline: {imp:+.1f}%")

        _log_elapsed("pitcher backtest", t0)

    # Write
    write_outputs(hitter_trace, pitcher_trace, hitter_bt, pitcher_bt)
    _log_elapsed("total", t0)
    print(f"\nOutputs written to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
