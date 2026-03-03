"""Bayesian foreign player prediction module.

Uses posterior parameters from npb-bayes-projection's Shrinkage model
to predict NPB performance for foreign players based on previous league stats.
PyMC not required — numpy sampling with hardcoded posterior parameters.
"""

import csv
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Posterior parameters (from npb-bayes-projection trace summaries)
# Model: NPB_stat = w * (prev_stat * cf_i) + (1 - w) * league_avg + noise
#   cf_i ~ Normal(cf_mu, cf_sigma)   (individual conversion factor)
#   noise ~ Normal(0, sigma_obs)
# ---------------------------------------------------------------------------
HITTER_PARAMS = {
    "cf_mu": (1.635, 0.216),     # conversion factor mean (mean, sd)
    "cf_sigma": (0.252, 0.147),  # conversion factor population sd
    "w": (0.136, 0.066),         # shrinkage weight toward prev stats
    "sigma_obs": (0.052, 0.008), # observation noise (wOBA scale)
}

PITCHER_PARAMS = {
    "cf_mu": (0.587, 0.196),     # conversion factor mean
    "cf_sigma": (0.283, 0.162),  # conversion factor population sd
    "w": (0.136, 0.082),         # shrinkage weight
    "sigma_obs": (1.11, 0.135),  # observation noise (ERA scale)
}

_N_SAMPLES = 5000
_RNG = np.random.default_rng(42)


def _summarize(samples: np.ndarray) -> dict:
    """Summarize posterior predictive samples."""
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "hdi_80": (float(np.percentile(samples, 10)),
                   float(np.percentile(samples, 90))),
        "hdi_95": (float(np.percentile(samples, 2.5)),
                   float(np.percentile(samples, 97.5))),
    }


def predict_foreign_hitter(prev_woba: float,
                           league_avg_woba: float = 0.310,
                           n_samples: int = _N_SAMPLES) -> dict:
    """Predict NPB wOBA for a foreign hitter with previous league stats."""
    p = HITTER_PARAMS
    w = np.clip(_RNG.normal(p["w"][0], p["w"][1], n_samples), 0, 1)
    cf_mu = _RNG.normal(p["cf_mu"][0], p["cf_mu"][1], n_samples)
    cf_sigma = np.abs(_RNG.normal(p["cf_sigma"][0], p["cf_sigma"][1], n_samples))
    cf_i = _RNG.normal(cf_mu, cf_sigma)
    sigma_obs = np.abs(_RNG.normal(p["sigma_obs"][0], p["sigma_obs"][1], n_samples))

    npb_woba = w * (prev_woba * cf_i) + (1 - w) * league_avg_woba
    npb_woba += _RNG.normal(0, sigma_obs)
    return _summarize(npb_woba)


def predict_foreign_pitcher(prev_era: float,
                            league_avg_era: float = 3.50,
                            n_samples: int = _N_SAMPLES) -> dict:
    """Predict NPB ERA for a foreign pitcher with previous league stats."""
    p = PITCHER_PARAMS
    w = np.clip(_RNG.normal(p["w"][0], p["w"][1], n_samples), 0, 1)
    cf_mu = _RNG.normal(p["cf_mu"][0], p["cf_mu"][1], n_samples)
    cf_sigma = np.abs(_RNG.normal(p["cf_sigma"][0], p["cf_sigma"][1], n_samples))
    cf_i = _RNG.normal(cf_mu, cf_sigma)
    sigma_obs = np.abs(_RNG.normal(p["sigma_obs"][0], p["sigma_obs"][1], n_samples))

    npb_era = w * (prev_era * cf_i) + (1 - w) * league_avg_era
    npb_era += _RNG.normal(0, sigma_obs)
    npb_era = np.clip(npb_era, 0, None)
    return _summarize(npb_era)


def predict_no_prev_stats(player_type: str,
                          league_avg_woba: float = 0.310,
                          league_avg_era: float = 3.50,
                          n_samples: int = _N_SAMPLES) -> dict:
    """Predict for a foreign player with no previous league stats.

    w=0 (no individual data) → league average + sigma_obs uncertainty.
    """
    if player_type == "hitter":
        center = league_avg_woba
        sigma = HITTER_PARAMS["sigma_obs"]
    else:
        center = league_avg_era
        sigma = PITCHER_PARAMS["sigma_obs"]

    sigma_s = np.abs(_RNG.normal(sigma[0], sigma[1], n_samples))
    samples = center + _RNG.normal(0, sigma_s)
    if player_type == "pitcher":
        samples = np.clip(samples, 0, None)
    return _summarize(samples)


def woba_to_wraa(pred_woba: float, lg_woba: float,
                 woba_scale: float = 1.15, pa: float = 400) -> float:
    """Convert predicted wOBA to wRAA estimate."""
    return (pred_woba - lg_woba) / woba_scale * pa


def era_to_ra_above_avg(pred_era: float, lg_era: float,
                        ip: float = 100) -> float:
    """Convert predicted ERA to runs allowed above average."""
    return (pred_era - lg_era) * ip / 9.0


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------
def load_foreign_2026() -> list[dict]:
    """Load data/foreign_2026.csv with previous league stats."""
    csv_path = Path(__file__).parent / "data" / "foreign_2026.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ("prev_wOBA", "prev_ERA", "expected_pa", "expected_ip"):
                if row.get(key):
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        row[key] = None
                else:
                    row[key] = None
            rows.append(row)
    return rows


def get_foreign_predictions(lg_woba: float = 0.310,
                            lg_era: float = 3.50,
                            woba_scale: float = 1.15) -> dict[str, dict]:
    """Compute Bayes predictions for all players in foreign_2026.csv.

    Returns: {npb_name: {pred, wraa_est|ra_above_avg, unc_wins, type, has_prev,
                         stat_label, stat_value, stat_range}}
    """
    data = load_foreign_2026()
    results: dict[str, dict] = {}

    for row in data:
        name = row["npb_name"]
        ptype = row["player_type"]

        if ptype == "hitter":
            if row["prev_wOBA"] is not None:
                pred = predict_foreign_hitter(row["prev_wOBA"], lg_woba)
                pa = row["expected_pa"] or 400
                wraa = woba_to_wraa(pred["mean"], lg_woba, woba_scale, pa)
                wraa_hi = woba_to_wraa(pred["hdi_80"][1], lg_woba, woba_scale, pa)
                wraa_lo = woba_to_wraa(pred["hdi_80"][0], lg_woba, woba_scale, pa)
                unc_wins = (wraa_hi - wraa_lo) / 10.0 / 2
                has_prev = True
            else:
                pred = predict_no_prev_stats("hitter", lg_woba, lg_era)
                wraa = 0.0
                unc_wins = 1.5
                has_prev = False

            results[name] = {
                "pred": pred, "wraa_est": wraa, "unc_wins": unc_wins,
                "type": ptype, "has_prev": has_prev,
                "stat_label": "wOBA",
                "stat_value": pred["mean"],
                "stat_range": pred["hdi_80"],
            }
        else:  # pitcher
            if row["prev_ERA"] is not None:
                pred = predict_foreign_pitcher(row["prev_ERA"], lg_era)
                ip = row["expected_ip"] or 100
                ra_above = era_to_ra_above_avg(pred["mean"], lg_era, ip)
                ra_hi = era_to_ra_above_avg(pred["hdi_80"][1], lg_era, ip)
                ra_lo = era_to_ra_above_avg(pred["hdi_80"][0], lg_era, ip)
                unc_wins = abs(ra_hi - ra_lo) / 10.0 / 2
                has_prev = True
            else:
                pred = predict_no_prev_stats("pitcher", lg_woba, lg_era)
                ra_above = 0.0
                unc_wins = 1.5
                has_prev = False

            results[name] = {
                "pred": pred, "ra_above_avg": ra_above, "unc_wins": unc_wins,
                "type": ptype, "has_prev": has_prev,
                "stat_label": "ERA",
                "stat_value": pred["mean"],
                "stat_range": pred["hdi_80"],
            }

    return results
