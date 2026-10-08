"""
Step 6: Team standings simulation with posterior uncertainty propagation.

Algorithm:
  1. Load Marcel 2026 hitter/pitcher projections from npb-prediction (GitHub raw)
  2. For each player: add Gaussian noise (Marcel MAE as sigma) -> N_SIM draws of OPS/ERA
  3. Aggregate to team RS (via K_HIT calibration) and RA (via ERA x IP/9)
  4. Apply Pythagorean expectation -> simulated win totals
  5. Rank teams within each league -> P(pennant), P(CS), 80% CI for wins

Output:
  data/projections/team_sim_2026.json
  data/projections/team_sim_2026.csv
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

def _log_elapsed(label: str, start: float, budget_min: int = 360):
    elapsed_min = (time.time() - start) / 60
    print(f"  [{label}] elapsed: {elapsed_min:.1f} min / {budget_min} min budget")
    if elapsed_min > budget_min * 0.8:
        print(f"  WARNING: {label} used {elapsed_min:.0f}/{budget_min} min "
              f"({elapsed_min / budget_min * 100:.0f}%) -- timeout risk!")


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "projections"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Simulation settings ────────────────────────────────────────────────────────
N_SIM = 10_000
NPB_GAMES = 143
NPB_PYTH_EXP = 1.83          # Pythagorean exponent for NPB
NPB_TARGET_PA = 5_300        # normalize each team to this many PA (≈143g × 37PA/g)
NPB_TARGET_IP = 1_287        # 143 games x 9 innings = total IP per team season
NPB_HIST_RS   = 535.0        # historical NPB avg RS per team (2015-2024)

# ── Uncertainty parameters ─────────────────────────────────────────────────────
# sigma from Marcel backtest (npb-prediction 2025 backtest):
SIGMA_OPS = 0.063            # Marcel OPS MAE -> per-player sigma
SIGMA_ERA = 0.78             # Marcel ERA MAE -> per-player sigma

# ── Team-level RS/RA uncertainty (measured from Marcel team backtest 2018-2025) ─
SIGMA_RS_HIST = 64.2         # std(pred_RS - actual_RS), 96 team-years
SIGMA_RA_HIST = 62.5         # std(pred_RA - actual_RA), 96 team-years

# ── RS calibration ─────────────────────────────────────────────────────────────
# After normalizing PA to NPB_TARGET_PA, avg OPS×PA = 0.6734 × 5300 = 3569
# K_HIT = NPB_HIST_RS / avg_OPS_PA = 535 / 3569 = 0.1499
K_HIT = 0.1499

# ── League structure ───────────────────────────────────────────────────────────
LEAGUES: dict[str, list[str]] = {
    "CL": ["\u962a\u795e", "\u5e83\u5cf6", "DeNA", "\u5de8\u4eba", "\u4e2d\u65e5", "\u30e4\u30af\u30eb\u30c8"],
    "PL": ["\u30bd\u30d5\u30c8\u30d0\u30f3\u30af", "\u65e5\u672c\u30cf\u30e0", "\u697d\u5929",
           "\u30aa\u30ea\u30c3\u30af\u30b9", "\u30ed\u30c3\u30c6", "\u897f\u6b66"],
}
# 阪神, 広島, DeNA, 巨人, 中日, ヤクルト
# ソフトバンク, 日本ハム, 楽天, オリックス, ロッテ, 西武

CS_SPOTS = 3  # top-3 qualify for Climax Series

# ── Roster turnover uncertainty ────────────────────────────────────────────────
TURNOVER_K = 1.0  # sigma scaling coefficient for roster turnover

# ── Data source: npb-prediction (public repo) ──────────────────────────────────
NPBP_BASE = (
    "https://raw.githubusercontent.com/yasumorishima/npb-prediction/main/data/projections"
)


def load_marcel() -> tuple[pd.DataFrame, pd.DataFrame]:
    hitters = pd.read_csv(f"{NPBP_BASE}/marcel_hitters_2026.csv", encoding="utf-8-sig")
    pitchers = pd.read_csv(f"{NPBP_BASE}/marcel_pitchers_2026.csv", encoding="utf-8-sig")
    return hitters, pitchers


def load_park_factors() -> dict[str, float]:
    """Load PF_5yr (latest year) per team from npb-prediction.

    Returns dict: team_ja -> PF_5yr (e.g. {"中日": 0.844, "日本ハム": 1.147, ...})
    """
    pf_df = pd.read_csv(f"{NPBP_BASE}/npb_park_factors.csv", encoding="utf-8-sig")
    latest_year = pf_df["year"].max()
    latest = pf_df[pf_df["year"] == latest_year][["team", "PF_5yr"]].copy()
    return dict(zip(latest["team"], latest["PF_5yr"]))


def load_historical() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Marcel team-level historical projections and actual results."""
    hist = pd.read_csv(f"{NPBP_BASE}/marcel_team_historical.csv", encoding="utf-8-sig")
    actual = pd.read_csv(
        "https://raw.githubusercontent.com/yasumorishima/npb-prediction/main"
        "/data/projections/pythagorean_2015_2025.csv",
        encoding="utf-8-sig",
    )
    return hist, actual


def compute_turnover(
    hitters_2026: pd.DataFrame,
    pitchers_2026: pd.DataFrame,
) -> dict[str, float]:
    """Compute roster turnover rate per team (2025 → 2026).

    Compares 2025 actual rosters (from npb-prediction raw data) with
    2026 projected rosters.  Turnover = departed_PA / total_2025_PA
    for hitters (analogous for pitchers, averaged).

    Returns dict: team -> turnover_pct (0.0 – 1.0).
    """
    saber_2025 = pd.read_csv(
        "https://raw.githubusercontent.com/yasumorishima/npb-prediction/main"
        "/data/raw/npb_hitters_2015_2025.csv",
        encoding="utf-8-sig",
    )
    saber_2025 = saber_2025[saber_2025["year"] == 2025]

    pitchers_2025 = pd.read_csv(
        "https://raw.githubusercontent.com/yasumorishima/npb-prediction/main"
        "/data/raw/npb_pitchers_2015_2025.csv",
        encoding="utf-8-sig",
    )
    pitchers_2025 = pitchers_2025[pitchers_2025["year"] == 2025]

    proj_h_names = set(hitters_2026["player"].values)
    proj_p_names = set(pitchers_2026["player"].values)

    turnover: dict[str, float] = {}
    all_teams = set(hitters_2026["team"].unique()) | set(saber_2025["team"].unique())

    for team in all_teams:
        # Hitter turnover
        team_h_2025 = saber_2025[saber_2025["team"] == team]
        total_pa = float(team_h_2025["PA"].sum())
        if total_pa > 0:
            departed_pa = float(
                team_h_2025[~team_h_2025["player"].isin(proj_h_names)]["PA"].sum()
            )
            h_turnover = departed_pa / total_pa
        else:
            h_turnover = 0.0

        # Pitcher turnover
        team_p_2025 = pitchers_2025[pitchers_2025["team"] == team]
        total_ip = float(team_p_2025["IP"].sum()) if len(team_p_2025) > 0 else 0.0
        if total_ip > 0:
            departed_ip = float(
                team_p_2025[~team_p_2025["player"].isin(proj_p_names)]["IP"].sum()
            )
            p_turnover = departed_ip / total_ip
        else:
            p_turnover = 0.0

        turnover[team] = (h_turnover + p_turnover) / 2.0

    return turnover


def normalize_hitter_pa(hitters: pd.DataFrame) -> pd.DataFrame:
    """Scale each team's total projected PA to NPB_TARGET_PA (≈ 143g × 37PA/g)."""
    hitters = hitters.copy()
    hitters["PA"] = hitters["PA"].astype(float)
    for team, grp in hitters.groupby("team"):
        total_pa = grp["PA"].sum()
        if total_pa > 0:
            hitters.loc[hitters["team"] == team, "PA"] *= NPB_TARGET_PA / total_pa
    return hitters


def normalize_pitcher_ip(pitchers: pd.DataFrame) -> pd.DataFrame:
    """Scale each team's total projected IP to NPB_TARGET_IP (= 143 * 9 innings)."""
    pitchers = pitchers.copy()
    for team, grp in pitchers.groupby("team"):
        total_ip = grp["IP"].sum()
        if total_ip > 0:
            pitchers.loc[pitchers["team"] == team, "IP"] *= NPB_TARGET_IP / total_ip
    return pitchers


def simulate(
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
    n_sim: int = N_SIM,
    seed: int = 42,
    turnover: dict[str, float] | None = None,
    park_factors: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Run Monte Carlo simulation. Returns dict: team -> (n_sim,) win totals.

    Args:
        turnover: Optional dict of team -> turnover_pct (0–1).  When provided,
            each team's RS/RA noise sigma is scaled by (1 + TURNOVER_K * pct)
            to widen the confidence interval for teams with high roster churn.
        park_factors: Optional dict of team -> PF_5yr.  When provided, raw
            RS/RA computed from Marcel stats are divided by (PF + 1) / 2 to
            remove the park effect embedded in player stats, yielding
            talent-neutral run estimates before re-calibration.
    """
    rng = np.random.default_rng(seed)

    # Hitters: sample OPS draws (n_players x n_sim)
    ops = hitters["OPS"].values[:, None]   # (P, 1)
    pa  = hitters["PA"].values[:, None]    # (P, 1)
    ops_sim = np.clip(ops + rng.normal(0, SIGMA_OPS, size=(len(hitters), n_sim)), 0.250, 1.200)

    # Pitchers: sample ERA draws (n_pitchers x n_sim)
    era = pitchers["ERA"].values[:, None]  # (Q, 1)
    ip  = pitchers["IP"].values[:, None]   # (Q, 1)
    era_sim = np.clip(era + rng.normal(0, SIGMA_ERA, size=(len(pitchers), n_sim)), 0.50, 12.0)

    # Aggregate per team
    h_teams = hitters["team"].values
    p_teams = pitchers["team"].values
    all_teams = sorted(set(hitters["team"].unique()) | set(pitchers["team"].unique()))

    # Compute raw RS / RA per team per simulation
    rs_raw: dict[str, np.ndarray] = {}
    ra_raw: dict[str, np.ndarray] = {}
    for team in all_teams:
        h_mask = h_teams == team
        p_mask = p_teams == team
        if not h_mask.any() or not p_mask.any():
            continue
        rs_raw[team] = K_HIT * (ops_sim[h_mask] * pa[h_mask]).sum(axis=0)
        ra_raw[team] = (era_sim[p_mask] * ip[p_mask]).sum(axis=0) / 9.0

    # Park factor correction: remove park effect embedded in Marcel stats.
    # Player stats are accumulated partly at their home park (PF effect).
    # Dividing by (PF + 1) / 2 neutralizes this bias.
    # E.g., Vantelin (PF=0.844): Dragons hitters' OPS is deflated by the
    # pitcher-friendly park → RS_raw too low. Dividing by 0.922 inflates to
    # neutral, correctly revealing the hitters' true run-scoring ability.
    if park_factors:
        for team in list(rs_raw.keys()):
            pf = park_factors.get(team)
            if pf is None or pf <= 0:
                continue
            pf_factor = (pf + 1.0) / 2.0
            rs_raw[team] = rs_raw[team] / pf_factor
            ra_raw[team] = ra_raw[team] / pf_factor

    # Post-hoc calibration: scale league-avg RS and RA to NPB_HIST_RS
    # This corrects Marcel's systematic optimism (OPS high / ERA low)
    valid_teams = list(rs_raw.keys())
    rs_matrix = np.stack([rs_raw[t] for t in valid_teams])  # (T, n_sim)
    ra_matrix = np.stack([ra_raw[t] for t in valid_teams])  # (T, n_sim)
    scale_rs = NPB_HIST_RS / rs_matrix.mean(axis=0)  # (n_sim,)
    scale_ra = NPB_HIST_RS / ra_matrix.mean(axis=0)  # (n_sim,)

    wins_sim: dict[str, np.ndarray] = {}
    for i, team in enumerate(valid_teams):
        rs = rs_matrix[i] * scale_rs
        ra = ra_matrix[i] * scale_ra

        # Turnover uncertainty: add extra noise proportional to turnover rate
        if turnover is not None:
            t_pct = turnover.get(team, 0.0)
            extra_sigma = TURNOVER_K * t_pct
            if extra_sigma > 0:
                rs = rs + rng.normal(0, extra_sigma * NPB_HIST_RS, n_sim)
                ra = ra + rng.normal(0, extra_sigma * NPB_HIST_RS, n_sim)
                rs = np.clip(rs, 1.0, None)
                ra = np.clip(ra, 1.0, None)

        rs_exp = np.power(np.clip(rs, 1.0, None), NPB_PYTH_EXP)
        ra_exp = np.power(np.clip(ra, 1.0, None), NPB_PYTH_EXP)
        wpct   = rs_exp / (rs_exp + ra_exp)
        wins_sim[team] = wpct * NPB_GAMES

    return wins_sim


def compute_probabilities(wins_sim: dict[str, np.ndarray]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for lg_name, lg_teams in LEAGUES.items():
        lg_sim = {t: wins_sim[t] for t in lg_teams if t in wins_sim}
        if not lg_sim:
            continue

        teams = list(lg_sim.keys())
        win_matrix = np.stack([lg_sim[t] for t in teams], axis=1)   # (n_sim, T)
        # Double argsort trick: ranks[i, j] = rank of team j in simulation i (1=best)
        ranks = (-win_matrix).argsort(axis=1).argsort(axis=1) + 1

        for i, team in enumerate(teams):
            w = win_matrix[:, i]
            r = ranks[:, i]
            results[team] = {
                "league":      lg_name,
                "p_pennant":   float((r == 1).mean()),
                "p_cs":        float((r <= CS_SPOTS).mean()),
                "median_wins": float(np.median(w)),
                "mean_wins":   float(w.mean()),
                "wins_80ci":   [float(np.percentile(w, 10)), float(np.percentile(w, 90))],
            }
    return results


def _run_one_backtest(
    merged: pd.DataFrame,
    rng: np.random.Generator,
    n_sim: int,
    pf_map: dict[tuple[int, str], float] | None,
) -> pd.DataFrame:
    """Inner loop: run simulation for each team-year, optionally with PF correction."""
    rows = []
    for _, row in merged.iterrows():
        pred_rs = float(row["pred_RS"])
        pred_ra = float(row["pred_RA"])

        # Park factor correction: remove park effect embedded in Marcel team stats.
        if pf_map is not None:
            pf = pf_map.get((int(row["year"]), str(row["team"])))
            if pf and pf > 0:
                pf_factor = (pf + 1.0) / 2.0
                pred_rs = pred_rs / pf_factor
                pred_ra = pred_ra / pf_factor

        g = int(row["G"])
        actual_w = float(row["W"])

        rs_sim = np.clip(rng.normal(pred_rs, SIGMA_RS_HIST, n_sim), 1.0, None)
        ra_sim = np.clip(rng.normal(pred_ra, SIGMA_RA_HIST, n_sim), 1.0, None)
        rs_exp = np.power(rs_sim, NPB_PYTH_EXP)
        ra_exp = np.power(ra_sim, NPB_PYTH_EXP)
        wins_sim = rs_exp / (rs_exp + ra_exp) * g

        median_w = float(np.median(wins_sim))
        ci_lo    = float(np.percentile(wins_sim, 10))
        ci_hi    = float(np.percentile(wins_sim, 90))
        pf_used = None
        if pf_map is not None:
            pf_val = pf_map.get((int(row["year"]), str(row["team"])))
            if pf_val and pf_val > 0:
                pf_used = round(pf_val, 3)

        rows.append({
            "year":        int(row["year"]),
            "league":      str(row.get("league", "")),
            "team":        row["team"],
            "pf_5yr":      pf_used,
            "pred_RS":     round(pred_rs, 1),
            "pred_RA":     round(pred_ra, 1),
            "actual_W":    actual_w,
            "actual_RS":   float(row["RS"]),
            "actual_RA":   float(row["RA"]),
            "median_wins": round(median_w, 1),
            "ci_lo":       round(ci_lo, 1),
            "ci_hi":       round(ci_hi, 1),
            "covered":     bool(ci_lo <= actual_w <= ci_hi),
            "error":       round(median_w - actual_w, 1),
        })
    return pd.DataFrame(rows)


def run_backtest(n_sim: int = N_SIM, seed: int = 42) -> None:
    """Validate team-level simulation against 2018-2025 Marcel team projections.

    Runs twice (without / with park factor correction) and prints a comparison.
    """
    t0 = time.time()
    print("Loading historical Marcel team projections...")
    hist, actual = load_historical()

    # Merge on year + team
    merged = hist.merge(
        actual[["year", "team", "league", "G", "W", "RS", "RA"]],
        on=["year", "team"],
        how="inner",
        suffixes=("_h", ""),
    )
    if "league_h" in merged.columns:
        merged = merged.rename(columns={"league": "league", "league_h": "league_hist"})
        merged["league"] = merged["league"].fillna(merged["league_hist"])
    print(f"  Matched: {len(merged)} team-years ({merged['year'].nunique()} seasons)")

    # Load park factors (all years)
    print("Loading park factors...")
    try:
        pf_df = pd.read_csv(f"{NPBP_BASE}/npb_park_factors.csv", encoding="utf-8-sig")
        pf_map: dict[tuple[int, str], float] = {
            (int(r["year"]), str(r["team"])): float(r["PF_5yr"])
            for _, r in pf_df.iterrows()
        }
        print(f"  Loaded PF data: {len(pf_map)} team-years")
    except Exception as e:
        print(f"  WARNING: Could not load park factors ({e})")
        pf_map = {}

    rng_base = np.random.default_rng(seed)
    rng_pf   = np.random.default_rng(seed)  # same seed → comparable results

    df_base = _run_one_backtest(merged, rng_base, n_sim, pf_map=None)
    df_pf   = _run_one_backtest(merged, rng_pf,   n_sim, pf_map=pf_map if pf_map else None)
    _log_elapsed("backtest simulation", t0)

    # ── Print comparison ──────────────────────────────────────────────────────
    def _stats(df: pd.DataFrame) -> tuple[float, float, float]:
        mae      = float(df["error"].abs().mean())
        bias     = float(df["error"].mean())
        coverage = float(df["covered"].mean())
        return mae, bias, coverage

    mae_b, bias_b, cov_b = _stats(df_base)
    mae_p, bias_p, cov_p = _stats(df_pf)

    print(f"\n{'':30s}  {'No PF':>8}  {'With PF':>8}  {'Δ':>8}")
    print(f"  {'MAE (wins)':28s}  {mae_b:8.2f}  {mae_p:8.2f}  {mae_p - mae_b:+8.2f}")
    print(f"  {'Bias (wins)':28s}  {bias_b:+8.2f}  {bias_p:+8.2f}  {bias_p - bias_b:+8.2f}")
    print(f"  {'80% CI coverage':28s}  {cov_b:8.1%}  {cov_p:8.1%}  {cov_p - cov_b:+8.1%}")

    print(f"\n── Year-by-year (With PF) ─────────────────────────────────────────")
    print(f"{'Year':>6}  {'MAE(base)':>9}  {'MAE(PF)':>9}  {'Δ':>7}  {'Cover(PF)':>10}")
    for yr, grp_p in df_pf.groupby("year"):
        grp_b = df_base[df_base["year"] == yr]
        print(
            f"  {yr}"
            f"  {grp_b['error'].abs().mean():9.2f}"
            f"  {grp_p['error'].abs().mean():9.2f}"
            f"  {grp_p['error'].abs().mean() - grp_b['error'].abs().mean():+7.2f}"
            f"  {grp_p['covered'].mean():10.1%}"
        )

    # ── Save results ─────────────────────────────────────────────────────────
    # backtest_results.csv: PF補正後の予測（canonical）
    df_pf.sort_values(["year", "league", "team"]).to_csv(
        OUT_DIR / "backtest_results.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\nSaved -> {OUT_DIR / 'backtest_results.csv'}")

    # backtest_comparison.csv: No PF vs With PF を全カラム横並び比較
    # RS/RA誤差 = pred - actual（正=過大推定、負=過小推定）
    df_base_idx = df_base.reset_index(drop=True)
    df_pf_idx   = df_pf.reset_index(drop=True)

    df_cmp = df_pf_idx[["year", "league", "team", "pf_5yr",
                         "actual_W", "actual_RS", "actual_RA"]].copy()

    # No PF 列
    df_cmp["pred_RS_base"]     = df_base_idx["pred_RS"]
    df_cmp["pred_RA_base"]     = df_base_idx["pred_RA"]
    df_cmp["rs_error_base"]    = (df_base_idx["pred_RS"] - df_pf_idx["actual_RS"]).round(1)
    df_cmp["ra_error_base"]    = (df_base_idx["pred_RA"] - df_pf_idx["actual_RA"]).round(1)
    df_cmp["median_wins_base"] = df_base_idx["median_wins"]
    df_cmp["ci_lo_base"]       = df_base_idx["ci_lo"]
    df_cmp["ci_hi_base"]       = df_base_idx["ci_hi"]
    df_cmp["error_base"]       = df_base_idx["error"]
    df_cmp["covered_base"]     = df_base_idx["covered"]

    # With PF 列
    df_cmp["pred_RS_pf"]       = df_pf_idx["pred_RS"]
    df_cmp["pred_RA_pf"]       = df_pf_idx["pred_RA"]
    df_cmp["rs_error_pf"]      = (df_pf_idx["pred_RS"] - df_pf_idx["actual_RS"]).round(1)
    df_cmp["ra_error_pf"]      = (df_pf_idx["pred_RA"] - df_pf_idx["actual_RA"]).round(1)
    df_cmp["median_wins_pf"]   = df_pf_idx["median_wins"]
    df_cmp["ci_lo_pf"]         = df_pf_idx["ci_lo"]
    df_cmp["ci_hi_pf"]         = df_pf_idx["ci_hi"]
    df_cmp["error_pf"]         = df_pf_idx["error"]
    df_cmp["covered_pf"]       = df_pf_idx["covered"]

    # 差分
    df_cmp["wins_delta"]       = (df_pf_idx["median_wins"] - df_base_idx["median_wins"]).round(1)
    df_cmp["rs_delta"]         = (df_pf_idx["pred_RS"] - df_base_idx["pred_RS"]).round(1)
    df_cmp["ra_delta"]         = (df_pf_idx["pred_RA"] - df_base_idx["pred_RA"]).round(1)

    df_cmp.sort_values(["year", "league", "team"]).to_csv(
        OUT_DIR / "backtest_comparison.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Saved -> {OUT_DIR / 'backtest_comparison.csv'}")

    rows_pf = df_pf.to_dict("records")
    summary = {
        "mae_no_pf":        round(mae_b, 3),
        "mae_with_pf":      round(mae_p, 3),
        "bias_no_pf":       round(bias_b, 3),
        "bias_with_pf":     round(bias_p, 3),
        "coverage_no_pf":   round(cov_b, 4),
        "coverage_with_pf": round(cov_p, 4),
        "n":                len(df_pf),
        "years":            sorted(df_pf["year"].unique().tolist()),
        "sigma_rs":         SIGMA_RS_HIST,
        "sigma_ra":         SIGMA_RA_HIST,
        "detail":           rows_pf,
    }
    json_path = OUT_DIR / "backtest_results.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {json_path}")
    _log_elapsed("backtest total", t0)


def main(n_sim: int = N_SIM) -> None:
    t0 = time.time()
    print("Loading Marcel 2026 projections from npb-prediction...")
    hitters, pitchers = load_marcel()
    print(f"  Hitters : {len(hitters):3d} players / {hitters['team'].nunique()} teams")
    print(f"  Pitchers: {len(pitchers):3d} players / {pitchers['team'].nunique()} teams")

    # Compute roster turnover uncertainty (2025 → 2026)
    print("Computing roster turnover (2025 → 2026)...")
    turnover = compute_turnover(hitters, pitchers)
    for team in sorted(turnover, key=turnover.get, reverse=True):
        print(f"  {team:14s}  turnover={turnover[team]:.1%}")

    hitters  = normalize_hitter_pa(hitters)
    pitchers = normalize_pitcher_ip(pitchers)

    print("Loading park factors (2025 PF_5yr)...")
    try:
        park_factors = load_park_factors()
        for team, pf in sorted(park_factors.items(), key=lambda x: -x[1]):
            print(f"  {team:12s}  PF_5yr={pf:.3f}")
    except Exception as e:
        print(f"  WARNING: Could not load park factors ({e}). Running without PF correction.")
        park_factors = None

    _log_elapsed("data loading + normalization", t0)

    print(f"Running {n_sim:,} simulations...")
    wins_sim = simulate(hitters, pitchers, n_sim, turnover=turnover, park_factors=park_factors)
    _log_elapsed("Monte Carlo simulation", t0)

    results = compute_probabilities(wins_sim)

    # Print results table
    for lg in ["CL", "PL"]:
        ranked = sorted(
            [(t, v) for t, v in results.items() if v["league"] == lg],
            key=lambda x: -x[1]["median_wins"],
        )
        print(f"\n-- {lg} -----------------------------------------------")
        print(f"{'Team':14s}  {'P(Pennant)':>10s}  {'P(CS)':>7s}  {'Median W':>8s}  80% CI")
        for t, v in ranked:
            lo, hi = v["wins_80ci"]
            print(
                f"{t:14s}  {v['p_pennant']:9.1%}  {v['p_cs']:6.1%}"
                f"  {v['median_wins']:7.1f}  [{lo:.1f}, {hi:.1f}]"
            )

    # Save JSON
    json_path = OUT_DIR / "team_sim_2026.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {json_path}")

    # Save CSV
    rows = []
    for team, v in results.items():
        lo, hi = v["wins_80ci"]
        rows.append({
            "team":         team,
            "league":       v["league"],
            "p_pennant":    round(v["p_pennant"], 4),
            "p_cs":         round(v["p_cs"], 4),
            "median_wins":  round(v["median_wins"], 1),
            "wins_80ci_lo": round(lo, 1),
            "wins_80ci_hi": round(hi, 1),
            "pf_5yr":       round(park_factors.get(team, float("nan")), 3) if park_factors else float("nan"),
        })
    (
        pd.DataFrame(rows)
        .sort_values(["league", "median_wins"], ascending=[True, False])
        .to_csv(OUT_DIR / "team_sim_2026.csv", index=False, encoding="utf-8-sig")
    )
    print(f"Saved -> {OUT_DIR / 'team_sim_2026.csv'}")
    _log_elapsed("total", t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sim", type=int, default=N_SIM)
    parser.add_argument("--backtest", action="store_true",
                        help="Run historical backtest (2018-2025) instead of 2026 forecast")
    args = parser.parse_args()
    if args.backtest:
        run_backtest(args.n_sim)
    else:
        main(args.n_sim)
