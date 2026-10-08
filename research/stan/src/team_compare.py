"""
Marcel vs Stan full comparison (2022-2025).

Compares Marcel and Stan predictions at two levels:
  1. Player-level: wOBA MAE (hitters) and ERA MAE (pitchers)
  2. Team-level:   Pythagorean win MAE via RS/RA aggregation

Uses actual player PA/IP weights and Stan predictions for ALL players
(Japanese + foreign first-year).

Scaling: Marcel-anchored (both models calibrated using Marcel's league avg).

Output:
  data/projections/team_compare_results.json
  data/projections/team_compare_results.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "model"
RAW_DIR   = DATA_DIR / "raw"
OUT_DIR   = DATA_DIR / "projections"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
NPB_PYTH_EXP = 1.83
NPB_HIST_RS  = 535.0

# K_WOBA: RS ≈ K_WOBA × Σ(wOBA × PA)
# Calibrated: NPB_HIST_RS / (avg_wOBA × NPB_TARGET_PA) ≈ 535 / (0.310 × 5300) = 0.3256
K_WOBA   = 0.3256

COMPARE_YEARS = list(range(2022, 2026))  # overlap with Stan backtest period

# ── Data source (npb-prediction GitHub raw) ───────────────────────────────────
NPBP_BASE = (
    "https://raw.githubusercontent.com/yasumorishima/npb-prediction/main/data/projections"
)


def ip_to_decimal(ip: float) -> float:
    whole  = int(ip)
    thirds = round((ip - whole) * 10)
    return whole + thirds / 3.0


def load_player_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Stan predictions for all players:
    - Japanese: from stan_jpn_model output (data/model/jpn_*.csv)
    - Foreign first-year: from stan_model output (data/model/stan_*_v1_predictions.csv)
    Returns (hitters, pitchers) each with columns:
      year, player, team, marcel_woba/era, stan_woba/era
    """
    # Japanese predictions
    jpn_h = pd.read_csv(MODEL_DIR / "jpn_hitter_predictions.csv",  encoding="utf-8-sig")
    jpn_p = pd.read_csv(MODEL_DIR / "jpn_pitcher_predictions.csv", encoding="utf-8-sig")

    # Foreign first-year predictions (add team via raw NPB stats)
    fgn_h = pd.read_csv(MODEL_DIR / "stan_hitter_v1_predictions.csv")
    fgn_p = pd.read_csv(MODEL_DIR / "stan_pitcher_v1_predictions.csv")

    # Load raw data to get team/PA/IP for foreign players
    raw_h = pd.read_csv(RAW_DIR / "npb_hitters_2015_2025.csv",  encoding="utf-8-sig")
    raw_p = pd.read_csv(RAW_DIR / "npb_pitchers_2015_2025.csv", encoding="utf-8-sig")

    # For foreign hitters: baseline_pred → marcel_woba, stan_pred → stan_woba
    # Note: foreign model predicts wOBA (same metric as jpn model)
    fgn_h = fgn_h.rename(columns={"baseline_pred": "marcel_woba", "stan_pred": "stan_woba",
                                    "npb_name": "player", "actual_wOBA": "actual_woba"})
    fgn_h = fgn_h.merge(
        raw_h[["player", "year", "team", "PA"]],
        on=["player", "year"], how="inner"
    )
    fgn_h = fgn_h[["year", "player", "team", "marcel_woba", "stan_woba", "actual_woba", "PA"]]
    fgn_h["actual_PA"] = fgn_h["PA"]
    fgn_h = fgn_h.drop(columns=["PA"])

    # For foreign pitchers: baseline_pred → marcel_era, stan_pred → stan_era
    fgn_p = fgn_p.rename(columns={"baseline_pred": "marcel_era", "stan_pred": "stan_era",
                                    "npb_name": "player", "actual_ERA": "actual_era"})
    raw_p_dec = raw_p.copy()
    raw_p_dec["actual_IP"] = raw_p_dec["IP"].apply(ip_to_decimal)
    fgn_p = fgn_p.merge(
        raw_p_dec[["player", "year", "team", "actual_IP"]],
        on=["player", "year"], how="inner"
    )
    fgn_p = fgn_p[["year", "player", "team", "marcel_era", "stan_era", "actual_era", "actual_IP"]]

    # Filter to comparison years
    jpn_h = jpn_h[jpn_h["year"].isin(COMPARE_YEARS)]
    jpn_p = jpn_p[jpn_p["year"].isin(COMPARE_YEARS)]
    fgn_h = fgn_h[fgn_h["year"].isin(COMPARE_YEARS)]
    fgn_p = fgn_p[fgn_p["year"].isin(COMPARE_YEARS)]

    # Rename for merge
    jpn_h = jpn_h.rename(columns={"actual_PA": "actual_PA"})

    # Remove foreign first-year players from Japanese set (avoid double-counting)
    fgn_h_keys = set(zip(fgn_h["player"], fgn_h["year"]))
    fgn_p_keys = set(zip(fgn_p["player"], fgn_p["year"]))
    jpn_h = jpn_h[~jpn_h.apply(lambda r: (r["player"], r["year"]) in fgn_h_keys, axis=1)]
    jpn_p = jpn_p[~jpn_p.apply(lambda r: (r["player"], r["year"]) in fgn_p_keys, axis=1)]

    # Combine Japanese + Foreign
    all_h = pd.concat([
        jpn_h[["year", "player", "team", "marcel_woba", "stan_woba", "actual_woba", "actual_PA"]],
        fgn_h[["year", "player", "team", "marcel_woba", "stan_woba", "actual_woba", "actual_PA"]],
    ], ignore_index=True)

    all_p = pd.concat([
        jpn_p[["year", "player", "team", "marcel_era", "stan_era", "actual_era", "actual_IP"]],
        fgn_p[["year", "player", "team", "marcel_era", "stan_era", "actual_era", "actual_IP"]],
    ], ignore_index=True)

    print(f"  All hitters: {len(all_h)} player-years  "
          f"({len(fgn_h)} foreign first-year)")
    print(f"  All pitchers: {len(all_p)} player-years  "
          f"({len(fgn_p)} foreign first-year)")

    return all_h, all_p


def compute_team_rs_ra(all_h: pd.DataFrame, all_p: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player predictions to team-year RS/RA for both Marcel and Stan."""
    # Hitters → RS
    h = all_h.copy()
    h["rs_marcel"] = K_WOBA * h["marcel_woba"] * h["actual_PA"]
    h["rs_stan"]   = K_WOBA * h["stan_woba"]   * h["actual_PA"]
    rs = h.groupby(["year", "team"])[["rs_marcel", "rs_stan"]].sum().reset_index()

    # Pitchers → RA
    p = all_p.copy()
    p["ra_marcel"] = p["marcel_era"] * p["actual_IP"] / 9.0
    p["ra_stan"]   = p["stan_era"]   * p["actual_IP"] / 9.0
    ra = p.groupby(["year", "team"])[["ra_marcel", "ra_stan"]].sum().reset_index()

    # Merge
    team = rs.merge(ra, on=["year", "team"], how="inner")
    return team


def pythagorean_wins(rs, ra, g):
    rs = np.clip(rs, 1.0, None)
    ra = np.clip(ra, 1.0, None)
    wpct = rs ** NPB_PYTH_EXP / (rs ** NPB_PYTH_EXP + ra ** NPB_PYTH_EXP)
    return wpct * g


def run_comparison() -> None:
    print("Loading player predictions...")
    all_h, all_p = load_player_predictions()

    # ── Player-level comparison ─────────────────────────────────────────────
    print("\n── Player-level MAE ───────────────────────────────────────────────")
    h_valid = all_h.dropna(subset=["actual_woba"])
    p_valid = all_p.dropna(subset=["actual_era"])

    h_mae_m = float((h_valid["marcel_woba"] - h_valid["actual_woba"]).abs().mean())
    h_mae_s = float((h_valid["stan_woba"]   - h_valid["actual_woba"]).abs().mean())
    p_mae_m = float((p_valid["marcel_era"]  - p_valid["actual_era"]).abs().mean())
    p_mae_s = float((p_valid["stan_era"]    - p_valid["actual_era"]).abs().mean())

    print(f"  {'':20s}  {'Marcel':>8s}  {'Stan':>8s}  {'Δ':>10s}")
    print(f"  {'Hitter wOBA MAE':20s}  {h_mae_m:8.4f}  {h_mae_s:8.4f}  {h_mae_s - h_mae_m:+10.4f}"
          f"  (n={len(h_valid)})")
    print(f"  {'Pitcher ERA MAE':20s}  {p_mae_m:8.4f}  {p_mae_s:8.4f}  {p_mae_s - p_mae_m:+10.4f}"
          f"  (n={len(p_valid)})")

    # Year-by-year player-level
    print(f"\n  {'Year':>6}  {'H_MAE_M':>8}  {'H_MAE_S':>8}  {'H_Δ':>8}"
          f"  {'P_MAE_M':>8}  {'P_MAE_S':>8}  {'P_Δ':>8}")
    player_yearly = {}
    for yr in COMPARE_YEARS:
        hv = h_valid[h_valid["year"] == yr]
        pv = p_valid[p_valid["year"] == yr]
        hm = float((hv["marcel_woba"] - hv["actual_woba"]).abs().mean()) if len(hv) else np.nan
        hs = float((hv["stan_woba"]   - hv["actual_woba"]).abs().mean()) if len(hv) else np.nan
        pm = float((pv["marcel_era"]  - pv["actual_era"]).abs().mean()) if len(pv) else np.nan
        ps = float((pv["stan_era"]    - pv["actual_era"]).abs().mean()) if len(pv) else np.nan
        print(f"  {yr:>6}  {hm:8.4f}  {hs:8.4f}  {hs - hm:+8.4f}"
              f"  {pm:8.4f}  {ps:8.4f}  {ps - pm:+8.4f}")
        player_yearly[yr] = {
            "hitter_mae_marcel": round(hm, 4), "hitter_mae_stan": round(hs, 4),
            "pitcher_mae_marcel": round(pm, 4), "pitcher_mae_stan": round(ps, 4),
            "n_hitters": len(hv), "n_pitchers": len(pv),
        }

    print("\nAggregating to team-year RS/RA...")
    team = compute_team_rs_ra(all_h, all_p)

    print("Loading actual results...")
    actual = pd.read_csv(
        "https://raw.githubusercontent.com/yasumorishima/npb-prediction/main"
        "/data/projections/pythagorean_2015_2025.csv",
        encoding="utf-8-sig",
    )
    actual = actual[actual["year"].isin(COMPARE_YEARS)][["year", "team", "G", "W", "RS", "RA"]]

    merged = team.merge(actual, on=["year", "team"], how="inner")
    print(f"  Matched: {len(merged)} team-years")

    # Marcel-anchored scaling: calibrate both models using Marcel's league avg.
    # Pythagorean depends only on RS/RA ratio, so the absolute level doesn't
    # matter — but the K_WOBA constant and ERA-to-RA conversion create a
    # systematic RS >> RA bias (~1.35 ratio vs actual 1.0).
    # Independent scaling (old method) normalises each model's avg RS and RA
    # separately to 535, which removes Stan's systematic improvement (K%/BB%
    # trend correction).  Marcel-anchored scaling applies the SAME calibration
    # factors (derived from Marcel) to Stan, preserving Stan's systematic
    # improvement while correcting the shared calibration bias.
    for yr in COMPARE_YEARS:
        mask = merged["year"] == yr
        if not mask.any():
            continue
        avg_rs_m = merged.loc[mask, "rs_marcel"].mean()
        avg_ra_m = merged.loc[mask, "ra_marcel"].mean()
        if avg_rs_m > 0:
            f_rs = NPB_HIST_RS / avg_rs_m
            merged.loc[mask, "rs_marcel"] *= f_rs
            merged.loc[mask, "rs_stan"]   *= f_rs
        if avg_ra_m > 0:
            f_ra = NPB_HIST_RS / avg_ra_m
            merged.loc[mask, "ra_marcel"] *= f_ra
            merged.loc[mask, "ra_stan"]   *= f_ra

    # Pythagorean wins
    merged["W_marcel"] = pythagorean_wins(
        merged["rs_marcel"].values, merged["ra_marcel"].values, merged["G"].values)
    merged["W_stan"] = pythagorean_wins(
        merged["rs_stan"].values, merged["ra_stan"].values, merged["G"].values)

    merged["err_marcel"] = merged["W_marcel"] - merged["W"]
    merged["err_stan"]   = merged["W_stan"]   - merged["W"]

    mae_marcel = float(merged["err_marcel"].abs().mean())
    mae_stan   = float(merged["err_stan"].abs().mean())
    bias_m = float(merged["err_marcel"].mean())
    bias_s = float(merged["err_stan"].mean())

    print(f"\n── Team Win MAE: Marcel vs Stan (2022-2025, n={len(merged)}) ──────────")
    print(f"{'':20s}  {'Marcel':>8s}  {'Stan':>8s}  {'Δ (Stan-Marcel)':>15s}")
    print(f"{'MAE (wins)':20s}  {mae_marcel:8.3f}  {mae_stan:8.3f}  {mae_stan - mae_marcel:+15.3f}")
    print(f"{'Bias (wins)':20s}  {bias_m:+8.3f}  {bias_s:+8.3f}  {bias_s - bias_m:+15.3f}")

    # Year-by-year breakdown
    print(f"\n{'Year':>6}  {'MAE Marcel':>10}  {'MAE Stan':>8}  {'Δ MAE':>8}  N")
    for yr, grp in merged.groupby("year"):
        m = grp["err_marcel"].abs().mean()
        s = grp["err_stan"].abs().mean()
        print(f"  {yr}  {m:10.3f}  {s:8.3f}  {s - m:+8.3f}  {len(grp)}")

    # Save
    out_rows = []
    for _, row in merged.iterrows():
        out_rows.append({
            "year":      int(row["year"]),
            "team":      row["team"],
            "actual_W":  float(row["W"]),
            "W_marcel":  round(float(row["W_marcel"]), 1),
            "W_stan":    round(float(row["W_stan"]), 1),
            "err_marcel": round(float(row["err_marcel"]), 1),
            "err_stan":   round(float(row["err_stan"]), 1),
        })

    summary = {
        "scaling_method": "marcel_anchored",
        "player_level": {
            "hitter_woba_mae_marcel": round(h_mae_m, 4),
            "hitter_woba_mae_stan":   round(h_mae_s, 4),
            "hitter_delta":           round(h_mae_s - h_mae_m, 4),
            "n_hitters":              len(h_valid),
            "pitcher_era_mae_marcel": round(p_mae_m, 4),
            "pitcher_era_mae_stan":   round(p_mae_s, 4),
            "pitcher_delta":          round(p_mae_s - p_mae_m, 4),
            "n_pitchers":             len(p_valid),
            "yearly": player_yearly,
        },
        "team_level": {
            "mae_marcel":  round(mae_marcel, 3),
            "mae_stan":    round(mae_stan, 3),
            "delta_mae":   round(mae_stan - mae_marcel, 3),
            "bias_marcel": round(bias_m, 3),
            "bias_stan":   round(bias_s, 3),
            "n":           len(merged),
        },
        "years":       COMPARE_YEARS,
        "detail":      out_rows,
    }

    json_path = OUT_DIR / "team_compare_results.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {json_path}")

    (
        merged[["year", "team", "W", "W_marcel", "W_stan", "err_marcel", "err_stan"]]
        .rename(columns={"W": "actual_W"})
        .sort_values(["year", "team"])
        .to_csv(OUT_DIR / "team_compare_results.csv", index=False, encoding="utf-8-sig")
    )
    print(f"Saved -> {OUT_DIR / 'team_compare_results.csv'}")


if __name__ == "__main__":
    run_comparison()
