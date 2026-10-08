"""Generate 2026 Stan-adjusted (Ridge) projections for NPB hitters and pitchers.

Trains Ridge regression on 2018-2025 full data, then applies delta corrections
to Marcel 2026 projections from npb-prediction.

Output:
  data/projections/stan_hitters_2026.csv
  data/projections/stan_pitchers_2026.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stan_jpn_model import (
    build_dataset,
    compute_fip_column,
    load_birthday_df,
    standardize_features,
)
from statistical_validation import (
    ALPHA_JPN_H,
    ALPHA_JPN_P,
    JPN_YEARS,
    ridge_fit_predict,
)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "projections"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Marcel CSV paths (from npb-prediction repo, committed to this repo too)
MARCEL_HITTERS_URL = "https://raw.githubusercontent.com/yasumorishima/npb-prediction/main/data/projections/marcel_hitters_2026.csv"
MARCEL_PITCHERS_URL = "https://raw.githubusercontent.com/yasumorishima/npb-prediction/main/data/projections/marcel_pitchers_2026.csv"

# Feature lists (same as statistical_validation.py LOO-CV)
FEAT_H = ["K_pct", "BB_pct", "BABIP", "age_from_peak", "pa_stability", "prev_woba_dev_sq"]
FEAT_P = ["K_pct", "BB_pct", "age_from_peak", "ip_stability", "prev_babip_p"]


def _norm(name: str) -> str:
    return name.replace("\u3000", " ").strip()


def main():
    print("=== Generate 2026 Stan Projections ===\n")

    # ── Load raw data ──────────────────────────────────────────────────────────
    print("Loading raw data...")
    saber = pd.read_csv(RAW_DIR / "npb_sabermetrics_2015_2025.csv", encoding="utf-8-sig")
    pitchers = pd.read_csv(RAW_DIR / "npb_pitchers_2015_2025.csv", encoding="utf-8-sig")
    saber = saber.dropna(subset=["wOBA"])
    pitchers = compute_fip_column(pitchers)
    bday_df = load_birthday_df()
    print(f"  Sabermetrics: {len(saber):,} rows")
    print(f"  Pitchers:     {len(pitchers):,} rows")

    # ── Build full training dataset (2018-2025) ────────────────────────────────
    print("\nBuilding 2018-2025 training data...")
    train_h, train_p, _ = build_dataset(saber, pitchers, JPN_YEARS, bday_df)
    print(f"  Hitters:  {len(train_h):,}")
    print(f"  Pitchers: {len(train_p):,}")

    if len(train_h) == 0 or len(train_p) == 0:
        print("ERROR: insufficient training data")
        return

    # ── Train Ridge on full data ───────────────────────────────────────────────
    # Hitters: y = actual_woba - marcel_woba
    y_h = (train_h["actual_woba"] - train_h["marcel_woba"]).values
    train_z_h = (train_h[FEAT_H] - train_h[FEAT_H].mean()) / train_h[FEAT_H].std().replace(0, 1)
    h_means = train_h[FEAT_H].mean()
    h_stds = train_h[FEAT_H].std().replace(0, 1)
    beta_h, _ = ridge_fit_predict(train_z_h.values, y_h, train_z_h.values[:1], ALPHA_JPN_H)
    # We only need beta, re-extract it
    n_feat_h = len(FEAT_H)
    X_h = train_z_h.values
    beta_h = np.linalg.solve(
        X_h.T @ X_h + ALPHA_JPN_H * np.eye(n_feat_h),
        X_h.T @ y_h,
    )
    print(f"\n  Hitter Ridge beta: {dict(zip(FEAT_H, beta_h.round(5)))}")

    # Pitchers: y = actual_era - marcel_era
    y_p = (train_p["actual_era"] - train_p["marcel_era"]).values
    p_means = train_p[FEAT_P].mean()
    p_stds = train_p[FEAT_P].std().replace(0, 1)
    X_p = ((train_p[FEAT_P] - p_means) / p_stds).values
    n_feat_p = len(FEAT_P)
    beta_p = np.linalg.solve(
        X_p.T @ X_p + ALPHA_JPN_P * np.eye(n_feat_p),
        X_p.T @ y_p,
    )
    print(f"  Pitcher Ridge beta: {dict(zip(FEAT_P, beta_p.round(5)))}")

    # ── Load Marcel 2026 CSVs ──────────────────────────────────────────────────
    print("\nLoading Marcel 2026 projections...")
    marcel_h = pd.read_csv(MARCEL_HITTERS_URL, encoding="utf-8-sig")
    marcel_p = pd.read_csv(MARCEL_PITCHERS_URL, encoding="utf-8-sig")
    marcel_h["player"] = marcel_h["player"].apply(_norm)
    marcel_p["player"] = marcel_p["player"].apply(_norm)
    print(f"  Marcel hitters:  {len(marcel_h)}")
    print(f"  Marcel pitchers: {len(marcel_p)}")

    # ── Compute 2025 features for each Marcel player ──────────────────────────
    # Hitter features from 2025 sabermetrics
    from stan_jpn_model import MIN_PA, MIN_IP, league_avg_woba, ip_to_decimal

    target_year = 2026
    prev_year = 2025

    # --- Hitter features ---
    prev_saber = saber[saber["year"] == prev_year].copy()
    prev_saber["player"] = prev_saber["player"].apply(_norm)
    prev_saber = prev_saber[prev_saber["PA"] >= MIN_PA]
    prev_saber["K_pct"] = prev_saber["SO"] / prev_saber["PA"]
    prev_saber["BB_pct"] = prev_saber["BB"] / prev_saber["PA"]
    denom = (prev_saber["AB"] - prev_saber["SO"] - prev_saber["HR"] + prev_saber["SF"]).clip(lower=1)
    prev_saber["BABIP"] = (prev_saber["H"] - prev_saber["HR"]) / denom
    lg_woba = league_avg_woba(saber, target_year)
    prev_saber["prev_woba_dev_sq"] = (prev_saber["wOBA"] - lg_woba) ** 2

    # Age from peak
    bday_map = dict(zip(bday_df["player"], bday_df["birthday"]))
    ages_h = []
    for _, row in prev_saber.iterrows():
        bday = bday_map.get(row["player"])
        if bday is not None and not pd.isna(bday):
            season_start = pd.Timestamp(year=target_year, month=4, day=1)
            age = (season_start - bday).days / 365.25
            ages_h.append(age - 29)  # PEAK_AGE = 29
        else:
            ages_h.append(np.nan)
    prev_saber["age_from_peak"] = ages_h

    # PA stability from Marcel hitters (approximate: use data_years as proxy)
    # For actual pa_stability, we'd need multi-year PA history
    # Use Marcel training data to compute per-player pa_stability
    from stan_jpn_model import compute_marcel_woba
    marcel_woba_2026 = compute_marcel_woba(saber, target_year)
    marcel_woba_2026["player"] = marcel_woba_2026["player"].apply(_norm)
    pa_stab_map = dict(zip(marcel_woba_2026["player"], marcel_woba_2026["pa_stability"]))

    # Merge features onto Marcel hitters
    feat_cols_h = ["player", "K_pct", "BB_pct", "BABIP", "prev_woba_dev_sq", "age_from_peak"]
    marcel_h_feat = marcel_h.merge(
        prev_saber[feat_cols_h].drop_duplicates("player"),
        on="player", how="left",
    )
    marcel_h_feat["pa_stability"] = marcel_h_feat["player"].map(pa_stab_map)

    # Fill missing features with training means (→ z-score ≈ 0 → delta ≈ 0)
    for col in FEAT_H:
        marcel_h_feat[col] = marcel_h_feat[col].fillna(h_means[col])

    # Compute Stan delta for hitters
    X_h_2026 = ((marcel_h_feat[FEAT_H] - h_means) / h_stds).values
    delta_woba = X_h_2026 @ beta_h

    # Compute stan_wOBA: Marcel wOBA + delta
    # First compute Marcel wOBA for each player (from regression: wOBA ~ OBP + SLG)
    # Use the same regression as streamlit_app.py's _enrich_projections
    df_fit = saber[saber["PA"] >= 100].dropna(subset=["wOBA", "OBP", "SLG"])
    X_reg = np.column_stack([df_fit["OBP"].values, df_fit["SLG"].values, np.ones(len(df_fit))])
    coeffs, _, _, _ = np.linalg.lstsq(X_reg, df_fit["wOBA"].values, rcond=None)
    a_obp, b_slg, intercept_w = coeffs
    marcel_h_feat["marcel_wOBA"] = (a_obp * marcel_h_feat["OBP"] + b_slg * marcel_h_feat["SLG"] + intercept_w).round(5)
    marcel_h_feat["stan_delta_wOBA"] = delta_woba.round(5)
    marcel_h_feat["stan_wOBA"] = (marcel_h_feat["marcel_wOBA"] + delta_woba).round(5)

    # Output hitters CSV (Marcel columns + stan columns)
    out_cols_h = list(marcel_h.columns) + ["marcel_wOBA", "stan_wOBA", "stan_delta_wOBA"]
    stan_h_out = marcel_h_feat[out_cols_h]
    stan_h_out.to_csv(OUT_DIR / "stan_hitters_2026.csv", index=False, encoding="utf-8-sig")
    print(f"\n  Saved: {OUT_DIR / 'stan_hitters_2026.csv'} ({len(stan_h_out)} rows)")
    print(f"  stan_delta_wOBA: mean={delta_woba.mean():+.5f}, std={delta_woba.std():.5f}")
    print(f"  Non-zero deltas: {(np.abs(delta_woba) > 0.0001).sum()} / {len(delta_woba)}")

    # --- Pitcher features ---
    from stan_jpn_model import compute_marcel_era, league_avg_era

    prev_pit = pitchers[pitchers["year"] == prev_year].copy()
    prev_pit["player"] = prev_pit["player"].apply(_norm)
    prev_pit["IP_dec"] = prev_pit["IP"].apply(ip_to_decimal)
    prev_pit = prev_pit[prev_pit["IP_dec"] >= MIN_IP]
    prev_pit = prev_pit[prev_pit["BF"] > 0]
    prev_pit["K_pct"] = prev_pit["SO"] / prev_pit["BF"]
    prev_pit["BB_pct"] = prev_pit["BB"] / prev_pit["BF"]
    babip_denom = (prev_pit["BF"] - prev_pit["SO"] - prev_pit["HRA"]).clip(lower=1)
    prev_pit["prev_babip_p"] = (prev_pit["HA"] - prev_pit["HRA"]) / babip_denom

    # Age
    ages_p = []
    for _, row in prev_pit.iterrows():
        bday = bday_map.get(row["player"])
        if bday is not None and not pd.isna(bday):
            season_start = pd.Timestamp(year=target_year, month=4, day=1)
            age = (season_start - bday).days / 365.25
            ages_p.append(age - 29)
        else:
            ages_p.append(np.nan)
    prev_pit["age_from_peak"] = ages_p

    # IP stability
    marcel_era_2026 = compute_marcel_era(pitchers, target_year)
    marcel_era_2026["player"] = marcel_era_2026["player"].apply(_norm)
    ip_stab_map = dict(zip(marcel_era_2026["player"], marcel_era_2026["ip_stability"]))

    feat_cols_p = ["player", "K_pct", "BB_pct", "prev_babip_p", "age_from_peak"]
    marcel_p_feat = marcel_p.merge(
        prev_pit[feat_cols_p].drop_duplicates("player"),
        on="player", how="left",
    )
    marcel_p_feat["ip_stability"] = marcel_p_feat["player"].map(ip_stab_map)

    for col in FEAT_P:
        marcel_p_feat[col] = marcel_p_feat[col].fillna(p_means[col])

    # Compute Stan delta for pitchers
    X_p_2026 = ((marcel_p_feat[FEAT_P] - p_means) / p_stds).values
    delta_era = X_p_2026 @ beta_p

    marcel_p_feat["stan_delta_ERA"] = delta_era.round(4)
    marcel_p_feat["stan_ERA"] = (marcel_p_feat["ERA"] + delta_era).round(4)

    out_cols_p = list(marcel_p.columns) + ["stan_ERA", "stan_delta_ERA"]
    stan_p_out = marcel_p_feat[out_cols_p]
    stan_p_out.to_csv(OUT_DIR / "stan_pitchers_2026.csv", index=False, encoding="utf-8-sig")
    print(f"  Saved: {OUT_DIR / 'stan_pitchers_2026.csv'} ({len(stan_p_out)} rows)")
    print(f"  stan_delta_ERA: mean={delta_era.mean():+.4f}, std={delta_era.std():.4f}")
    print(f"  Non-zero deltas: {(np.abs(delta_era) > 0.0001).sum()} / {len(delta_era)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
