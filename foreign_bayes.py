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

# Historical NPB foreign player first-year averages
# Source: npb-bayes-projection foreign_players_master.csv (367 players, 2015-2025)
# Hitter: PA≥50, Pitcher: IP≥20
HISTORICAL_FOREIGN_DEFAULTS = {
    "hitter": {"mean_woba": 0.318, "std_woba": 0.053, "n": 117},
    "pitcher": {"mean_era": 3.412, "std_era": 1.436, "n": 151},
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

    Uses historical foreign player first-year averages as center
    (better than league average since NPB teams recruit above-average foreigners).
    """
    hist = _get_historical()
    if player_type == "hitter":
        center = hist["hitter"]["mean_woba"]
        sigma = HITTER_PARAMS["sigma_obs"]
    else:
        center = hist["pitcher"]["mean_era"]
        sigma = PITCHER_PARAMS["sigma_obs"]

    sigma_s = np.abs(_RNG.normal(sigma[0], sigma[1], n_samples))
    samples = center + _RNG.normal(0, sigma_s)
    if player_type == "pitcher":
        samples = np.clip(samples, 0, None)
    return _summarize(samples)


def _sample_hitter_woba(prev_woba: float, lg_woba: float,
                        n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n wOBA values from posterior for a foreign hitter."""
    p = HITTER_PARAMS
    w = np.clip(rng.normal(p["w"][0], p["w"][1], n), 0, 1)
    cf_mu = rng.normal(p["cf_mu"][0], p["cf_mu"][1], n)
    cf_sigma = np.abs(rng.normal(p["cf_sigma"][0], p["cf_sigma"][1], n))
    cf_i = rng.normal(cf_mu, cf_sigma)
    sigma_obs = np.abs(rng.normal(p["sigma_obs"][0], p["sigma_obs"][1], n))
    npb_woba = w * (prev_woba * cf_i) + (1 - w) * lg_woba
    npb_woba += rng.normal(0, sigma_obs)
    return npb_woba


def _sample_pitcher_era(prev_era: float, lg_era: float,
                        n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n ERA values from posterior for a foreign pitcher."""
    p = PITCHER_PARAMS
    w = np.clip(rng.normal(p["w"][0], p["w"][1], n), 0, 1)
    cf_mu = rng.normal(p["cf_mu"][0], p["cf_mu"][1], n)
    cf_sigma = np.abs(rng.normal(p["cf_sigma"][0], p["cf_sigma"][1], n))
    cf_i = rng.normal(cf_mu, cf_sigma)
    sigma_obs = np.abs(rng.normal(p["sigma_obs"][0], p["sigma_obs"][1], n))
    npb_era = w * (prev_era * cf_i) + (1 - w) * lg_era
    npb_era += rng.normal(0, sigma_obs)
    return np.clip(npb_era, 0, None)


def _sample_no_prev(player_type: str, lg_woba: float, lg_era: float,
                    n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample for a foreign player with no previous league stats.

    Uses historical foreign player first-year averages as center.
    """
    hist = _get_historical()
    if player_type == "hitter":
        center = hist["hitter"]["mean_woba"]
        sigma = HITTER_PARAMS["sigma_obs"]
    else:
        center = hist["pitcher"]["mean_era"]
        sigma = PITCHER_PARAMS["sigma_obs"]
    sigma_s = np.abs(rng.normal(sigma[0], sigma[1], n))
    samples = center + rng.normal(0, sigma_s)
    if player_type != "hitter":
        samples = np.clip(samples, 0, None)
    return samples


def simulate_team_wins_mc(
    pred_rs: float,
    pred_ra: float,
    missing_players: list,
    rs_scale: float,
    ra_scale: float,
    lg_woba: float,
    lg_era: float,
    woba_scale: float = 1.15,
    k: float = 1.72,
    n_samples: int = 5000,
) -> dict:
    """Monte Carlo simulation: posterior samples → team wins distribution.

    Perturbs the center estimate (pred_rs, pred_ra) by sampling from each
    missing player's posterior, then applies the Pythagorean formula to get
    the full wins distribution.  Diversification effect is captured naturally.
    """
    rng = np.random.default_rng(42)

    delta_rs = np.zeros(n_samples)
    delta_ra = np.zeros(n_samples)
    delta_wins = np.zeros(n_samples)

    for m in missing_players:
        b = m.get("bayes")
        kind = m.get("kind", "rookie")

        if kind == "rookie" or b is None:
            # Rookie: perturb directly in wins space (σ = 1.5 / 1.28 ≈ 1.17)
            delta_wins += rng.normal(0, 1.17, n_samples)
            continue

        if b.get("has_prev"):
            if b["type"] == "hitter":
                prev_stat = b.get("prev_stat")
                expected_pt = b.get("expected_pt", 400)
                woba_samples = _sample_hitter_woba(prev_stat, lg_woba,
                                                   n_samples, rng)
                wraa_samples = (woba_samples - lg_woba) / woba_scale * expected_pt
                wraa_mean = b.get("wraa_est", 0)
                delta_rs += (wraa_samples - wraa_mean) * rs_scale
            else:  # pitcher
                prev_stat = b.get("prev_stat")
                expected_pt = b.get("expected_pt", 100)
                era_samples = _sample_pitcher_era(prev_stat, lg_era,
                                                  n_samples, rng)
                ra_samples = (era_samples - lg_era) * expected_pt / 9.0
                ra_mean = b.get("ra_above_avg", 0)
                delta_ra += (ra_samples - ra_mean) * ra_scale
        else:
            # Foreign without prev stats
            if b["type"] == "hitter":
                woba_samples = _sample_no_prev("hitter", lg_woba, lg_era,
                                               n_samples, rng)
                expected_pt = b.get("expected_pt", 400)
                wraa_samples = (woba_samples - lg_woba) / woba_scale * expected_pt
                delta_rs += wraa_samples * rs_scale  # mean ≈ 0
            elif b["type"] == "pitcher":
                era_samples = _sample_no_prev("pitcher", lg_woba, lg_era,
                                              n_samples, rng)
                expected_pt = b.get("expected_pt", 100)
                ra_samples = (era_samples - lg_era) * expected_pt / 9.0
                delta_ra += ra_samples * ra_scale  # mean ≈ 0
            else:
                # unknown type → treat as rookie-like uncertainty
                delta_wins += rng.normal(0, 1.17, n_samples)

    sim_rs = pred_rs + delta_rs
    sim_ra = pred_ra + delta_ra
    sim_wpct = sim_rs**k / (sim_rs**k + sim_ra**k)
    sim_wins = np.clip(sim_wpct * 143 + delta_wins, 0, 143)

    return {
        "pred_W_low": float(np.percentile(sim_wins, 10)),
        "pred_W_high": float(np.percentile(sim_wins, 90)),
        "std_W": float(np.std(sim_wins)),
    }


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


def load_foreign_historical() -> dict:
    """Load historical foreign player first-year stats from CSV.

    Returns dict with 'hitter' and 'pitcher' sub-dicts containing
    mean, std, and n for wOBA (hitter) / ERA (pitcher).
    Falls back to HISTORICAL_FOREIGN_DEFAULTS if CSV not found.
    """
    csv_path = Path(__file__).parent / "data" / "foreign_historical.csv"
    if not csv_path.exists():
        return HISTORICAL_FOREIGN_DEFAULTS

    hitter_wobas = []
    pitcher_eras = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ptype = row.get("player_type", "").strip()
            if ptype == "hitter":
                pa_str = row.get("npb_first_year_PA", "")
                woba_str = row.get("npb_first_year_wOBA", "")
                if pa_str and woba_str:
                    try:
                        pa = float(pa_str)
                        woba = float(woba_str)
                        if pa >= 50:
                            hitter_wobas.append(woba)
                    except ValueError:
                        pass
            elif ptype == "pitcher":
                ip_str = row.get("npb_first_year_IP", "")
                era_str = row.get("npb_first_year_ERA", "")
                if ip_str and era_str:
                    try:
                        ip = float(ip_str)
                        era = float(era_str)
                        if ip >= 20:
                            pitcher_eras.append(era)
                    except ValueError:
                        pass

    result = {}
    if hitter_wobas:
        arr = np.array(hitter_wobas)
        result["hitter"] = {
            "mean_woba": float(np.mean(arr)),
            "std_woba": float(np.std(arr)),
            "n": len(arr),
        }
    else:
        result["hitter"] = HISTORICAL_FOREIGN_DEFAULTS["hitter"]

    if pitcher_eras:
        arr = np.array(pitcher_eras)
        result["pitcher"] = {
            "mean_era": float(np.mean(arr)),
            "std_era": float(np.std(arr)),
            "n": len(arr),
        }
    else:
        result["pitcher"] = HISTORICAL_FOREIGN_DEFAULTS["pitcher"]

    return result


_historical_cache: dict | None = None


def _get_historical() -> dict:
    """Return cached historical foreign player stats."""
    global _historical_cache
    if _historical_cache is None:
        _historical_cache = load_foreign_historical()
    return _historical_cache


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
                prev_stat = row["prev_wOBA"]
                expected_pt = pa
            else:
                pred = predict_no_prev_stats("hitter", lg_woba, lg_era)
                expected_pt = 400
                wraa = woba_to_wraa(pred["mean"], lg_woba, woba_scale,
                                    expected_pt)
                unc_wins = 1.5
                has_prev = False
                prev_stat = None

            results[name] = {
                "pred": pred, "wraa_est": wraa, "unc_wins": unc_wins,
                "type": ptype, "has_prev": has_prev,
                "has_historical": not has_prev,
                "prev_stat": prev_stat, "expected_pt": expected_pt,
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
                prev_stat = row["prev_ERA"]
                expected_pt = ip
            else:
                pred = predict_no_prev_stats("pitcher", lg_woba, lg_era)
                expected_pt = 100
                ra_above = era_to_ra_above_avg(pred["mean"], lg_era,
                                               expected_pt)
                unc_wins = 1.5
                has_prev = False
                prev_stat = None

            results[name] = {
                "pred": pred, "ra_above_avg": ra_above, "unc_wins": unc_wins,
                "type": ptype, "has_prev": has_prev,
                "has_historical": not has_prev,
                "prev_stat": prev_stat, "expected_pt": expected_pt,
                "stat_label": "ERA",
                "stat_value": pred["mean"],
                "stat_range": pred["hdi_80"],
            }

    return results
