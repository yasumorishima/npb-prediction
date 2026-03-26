"""
チームモンテカルロシミュレーション（Phase 3）

選手レベルの予測分布からチーム勝率分布を導出する。
ベイズ予測の不確実性がチームレベルに伝播する。

Algorithm:
  1. ベイズ予測（CI付き）+ Marcel + ML の選手データをロード
  2. 各選手: 予測分布からN_SIMドロー（Normal(bayes_pred, sigma)）
  3. チームRS集計（OPS×PA加重 × K_HIT calibration）
  4. チームRA集計（ERA×IP/9）
  5. パークファクター補正
  6. ピタゴラス勝率 → 勝利数分布
  7. リーグ内順位 → P(優勝), P(CS), P(最下位)

Output:
  data/projections/team_sim_2026.json
  data/projections/team_sim_2026.csv

Data sources:
- プロ野球データFreak (https://baseball-data.com)
- 日本野球機構 NPB (https://npb.jp)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import TARGET_YEAR, PROJECTIONS_DIR

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = PROJECTIONS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Simulation settings ──────────────────────────────────────────────────────
N_SIM = 10_000
NPB_GAMES = 143
NPB_PYTH_EXP = 1.83
NPB_TARGET_PA = 5_300       # 143g × 37PA/g
NPB_TARGET_IP = 1_287       # 143 × 9
NPB_HIST_RS = 535.0         # historical NPB avg RS per team

# Sigma from research repo LOO-CV
SIGMA_OPS = 0.063           # per-player OPS sigma
SIGMA_ERA = 0.78            # per-player ERA sigma

# RS calibration: K_HIT = NPB_HIST_RS / avg(OPS×PA)
K_HIT = 0.1499

# ── League structure ─────────────────────────────────────────────────────────
LEAGUES: dict[str, list[str]] = {
    "CL": ["阪神", "広島", "DeNA", "巨人", "中日", "ヤクルト"],
    "PL": ["ソフトバンク", "日本ハム", "楽天", "オリックス", "ロッテ", "西武"],
}
CS_SPOTS = 3


def _log_elapsed(label: str, start: float, budget_min: int = 180):
    elapsed_min = (time.time() - start) / 60
    print(f"  [{label}] elapsed: {elapsed_min:.1f} min / {budget_min} min budget")
    if elapsed_min > budget_min * 0.8:
        print(f"  WARNING: {label} used {elapsed_min:.0f}/{budget_min} min "
              f"({elapsed_min / budget_min * 100:.0f}%) -- timeout risk!")


def load_bayes_hitters() -> pd.DataFrame:
    path = OUT_DIR / f"bayes_hitters_{TARGET_YEAR}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def load_bayes_pitchers() -> pd.DataFrame:
    path = OUT_DIR / f"bayes_pitchers_{TARGET_YEAR}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def load_foreign_hitters() -> pd.DataFrame:
    path = OUT_DIR / f"foreign_hitters_{TARGET_YEAR}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def load_foreign_pitchers() -> pd.DataFrame:
    path = OUT_DIR / f"foreign_pitchers_{TARGET_YEAR}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def load_park_factors() -> dict[str, float]:
    path = OUT_DIR / "npb_park_factors.csv"
    if not path.exists():
        return {}
    pf_df = pd.read_csv(path, encoding="utf-8-sig")
    latest_year = pf_df["year"].max()
    latest = pf_df[pf_df["year"] == latest_year][["team", "PF_5yr"]].copy()
    return dict(zip(latest["team"], latest["PF_5yr"]))


def normalize_pa(hitters: pd.DataFrame) -> pd.DataFrame:
    """Scale each team's total projected PA to NPB_TARGET_PA."""
    hitters = hitters.copy()
    hitters["PA"] = hitters["PA"].astype(float)
    for team in hitters["team"].unique():
        mask = hitters["team"] == team
        total_pa = hitters.loc[mask, "PA"].sum()
        if total_pa > 0:
            hitters.loc[mask, "PA"] *= NPB_TARGET_PA / total_pa
    return hitters


def normalize_ip(pitchers: pd.DataFrame) -> pd.DataFrame:
    """Scale each team's total projected IP to NPB_TARGET_IP."""
    pitchers = pitchers.copy()
    pitchers["IP"] = pitchers["IP"].astype(float)
    for team in pitchers["team"].unique():
        mask = pitchers["team"] == team
        total_ip = pitchers.loc[mask, "IP"].sum()
        if total_ip > 0:
            pitchers.loc[mask, "IP"] *= NPB_TARGET_IP / total_ip
    return pitchers


def simulate(
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
    foreign_h: pd.DataFrame,
    foreign_p: pd.DataFrame,
    n_sim: int = N_SIM,
    seed: int = 42,
    park_factors: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Run Monte Carlo simulation.

    ベイズ予測がある選手はbayes_OPS/bayes_ERAを中心にサンプリング。
    外国人選手はforeign CSVのbayes_OPS/bayes_ERAを使用（PA/IPは推定値）。
    """
    rng = np.random.default_rng(seed)

    # 日本人打者: bayes_OPS があればそれを使う、なければmarcel_OPS
    ops_vals = hitters["bayes_OPS"].fillna(hitters["marcel_OPS"]).values
    pa_vals = hitters["PA"].values
    h_teams = hitters["team"].values

    # 日本人打者サンプリング
    ops_sim = np.clip(
        ops_vals[:, None] + rng.normal(0, SIGMA_OPS, size=(len(hitters), n_sim)),
        0.250, 1.200,
    )

    # 日本人投手: bayes_ERA があればそれを使う、なければmarcel_ERA
    era_vals = pitchers["bayes_ERA"].fillna(pitchers["marcel_ERA"]).values
    ip_vals = pitchers["IP"].values
    p_teams = pitchers["team"].values

    era_sim = np.clip(
        era_vals[:, None] + rng.normal(0, SIGMA_ERA, size=(len(pitchers), n_sim)),
        0.50, 12.0,
    )

    # 全チーム収集
    all_teams = sorted(
        set(h_teams) | set(p_teams)
        | (set(foreign_h["team"]) if len(foreign_h) > 0 else set())
        | (set(foreign_p["team"]) if len(foreign_p) > 0 else set())
    )

    # 外国人打者のRS貢献（チーム別に集計）
    foreign_h_rs: dict[str, np.ndarray] = {}
    if len(foreign_h) > 0:
        for _, row in foreign_h.iterrows():
            team = row["team"]
            ops = row["bayes_OPS"]
            # 外国人のPA推定: 1軍定着なら300PA、それ以外100PA
            est_pa = 300 if ops >= 0.680 else 100
            player_ops_sim = np.clip(
                rng.normal(ops, SIGMA_OPS * 1.5, n_sim),  # 外国人は不確実性1.5倍
                0.200, 1.200,
            )
            contrib = K_HIT * player_ops_sim * est_pa
            if team not in foreign_h_rs:
                foreign_h_rs[team] = np.zeros(n_sim)
            foreign_h_rs[team] += contrib

    # 外国人投手のRA貢献
    foreign_p_ra: dict[str, np.ndarray] = {}
    if len(foreign_p) > 0:
        for _, row in foreign_p.iterrows():
            team = row["team"]
            era = row["bayes_ERA"]
            est_ip = 80 if era <= 4.0 else 40
            player_era_sim = np.clip(
                rng.normal(era, SIGMA_ERA * 1.5, n_sim),
                0.50, 12.0,
            )
            contrib = player_era_sim * est_ip / 9.0
            if team not in foreign_p_ra:
                foreign_p_ra[team] = np.zeros(n_sim)
            foreign_p_ra[team] += contrib

    # チーム別RS/RA集計
    rs_raw: dict[str, np.ndarray] = {}
    ra_raw: dict[str, np.ndarray] = {}

    for team in all_teams:
        # 日本人打者
        h_mask = h_teams == team
        if h_mask.any():
            rs = K_HIT * (ops_sim[h_mask] * pa_vals[h_mask, None]).sum(axis=0)
        else:
            rs = np.full(n_sim, NPB_HIST_RS)

        # 外国人打者のRS追加
        if team in foreign_h_rs:
            rs = rs + foreign_h_rs[team]

        rs_raw[team] = rs

        # 日本人投手
        p_mask = p_teams == team
        if p_mask.any():
            ra = (era_sim[p_mask] * ip_vals[p_mask, None]).sum(axis=0) / 9.0
        else:
            ra = np.full(n_sim, NPB_HIST_RS)

        # 外国人投手のRA追加
        if team in foreign_p_ra:
            ra = ra + foreign_p_ra[team]

        ra_raw[team] = ra

    # パークファクター補正
    if park_factors:
        for team in list(rs_raw.keys()):
            pf = park_factors.get(team)
            if pf is None or pf <= 0:
                continue
            pf_factor = (pf + 1.0) / 2.0
            rs_raw[team] = rs_raw[team] / pf_factor
            ra_raw[team] = ra_raw[team] / pf_factor

    # Post-hoc calibration: scale league-avg RS/RA to NPB_HIST_RS
    valid_teams = [t for t in all_teams if t in rs_raw and t in ra_raw]
    rs_matrix = np.stack([rs_raw[t] for t in valid_teams])
    ra_matrix = np.stack([ra_raw[t] for t in valid_teams])
    scale_rs = NPB_HIST_RS / rs_matrix.mean(axis=0)
    scale_ra = NPB_HIST_RS / ra_matrix.mean(axis=0)

    wins_sim: dict[str, np.ndarray] = {}
    for i, team in enumerate(valid_teams):
        rs = rs_matrix[i] * scale_rs
        ra = ra_matrix[i] * scale_ra

        rs_exp = np.power(np.clip(rs, 1.0, None), NPB_PYTH_EXP)
        ra_exp = np.power(np.clip(ra, 1.0, None), NPB_PYTH_EXP)
        wpct = rs_exp / (rs_exp + ra_exp)
        wins_sim[team] = wpct * NPB_GAMES

    return wins_sim


def compute_probabilities(wins_sim: dict[str, np.ndarray]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for lg_name, lg_teams in LEAGUES.items():
        lg_sim = {t: wins_sim[t] for t in lg_teams if t in wins_sim}
        if not lg_sim:
            continue

        teams = list(lg_sim.keys())
        win_matrix = np.stack([lg_sim[t] for t in teams], axis=1)
        ranks = (-win_matrix).argsort(axis=1).argsort(axis=1) + 1

        for i, team in enumerate(teams):
            w = win_matrix[:, i]
            r = ranks[:, i]
            results[team] = {
                "league": lg_name,
                "p_pennant": round(float((r == 1).mean()), 4),
                "p_cs": round(float((r <= CS_SPOTS).mean()), 4),
                "p_last": round(float((r == len(teams)).mean()), 4),
                "median_wins": round(float(np.median(w)), 1),
                "mean_wins": round(float(w.mean()), 1),
                "wins_80ci": [round(float(np.percentile(w, 10)), 1),
                              round(float(np.percentile(w, 90)), 1)],
                "wins_95ci": [round(float(np.percentile(w, 2.5)), 1),
                              round(float(np.percentile(w, 97.5)), 1)],
            }
    return results


def main(n_sim: int = N_SIM) -> None:
    t0 = time.time()
    print("=" * 60)
    print(f"チームモンテカルロシミュレーション (target: {TARGET_YEAR})")
    print("=" * 60)

    # データロード
    hitters = load_bayes_hitters()
    pitchers = load_bayes_pitchers()
    foreign_h = load_foreign_hitters()
    foreign_p = load_foreign_pitchers()

    print(f"日本人打者: {len(hitters)} players")
    print(f"日本人投手: {len(pitchers)} players")
    print(f"外国人打者: {len(foreign_h)} players")
    print(f"外国人投手: {len(foreign_p)} players")

    # PA/IP正規化
    hitters = normalize_pa(hitters)
    pitchers = normalize_ip(pitchers)

    # パークファクター
    park_factors = load_park_factors()
    if park_factors:
        print(f"\nパークファクター ({len(park_factors)} teams):")
        for team, pf in sorted(park_factors.items(), key=lambda x: -x[1]):
            print(f"  {team:12s}  PF={pf:.3f}")
    else:
        print("\nWARNING: パークファクターなし")

    _log_elapsed("data_load", t0)

    # シミュレーション実行
    print(f"\n{n_sim:,} simulations...")
    wins_sim = simulate(hitters, pitchers, foreign_h, foreign_p,
                        n_sim=n_sim, park_factors=park_factors)
    _log_elapsed("monte_carlo_sim", t0)

    results = compute_probabilities(wins_sim)

    # 結果表示
    for lg in ["CL", "PL"]:
        ranked = sorted(
            [(t, v) for t, v in results.items() if v["league"] == lg],
            key=lambda x: -x[1]["median_wins"],
        )
        print(f"\n── {lg} {'─' * 55}")
        print(f"{'Team':14s}  {'P(優勝)':>8s}  {'P(CS)':>7s}  {'P(最下位)':>9s}  {'Median W':>8s}  80% CI")
        for t, v in ranked:
            lo, hi = v["wins_80ci"]
            print(
                f"{t:14s}  {v['p_pennant']:7.1%}  {v['p_cs']:6.1%}"
                f"  {v['p_last']:8.1%}"
                f"  {v['median_wins']:7.1f}  [{lo:.1f}, {hi:.1f}]"
            )

    # JSON保存
    json_path = OUT_DIR / f"team_sim_{TARGET_YEAR}.json"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved: {json_path}")

    # CSV保存
    rows = []
    for team, v in results.items():
        lo80, hi80 = v["wins_80ci"]
        lo95, hi95 = v["wins_95ci"]
        rows.append({
            "team": team,
            "league": v["league"],
            "p_pennant": v["p_pennant"],
            "p_cs": v["p_cs"],
            "p_last": v["p_last"],
            "median_wins": v["median_wins"],
            "mean_wins": v["mean_wins"],
            "wins_80ci_lo": lo80,
            "wins_80ci_hi": hi80,
            "wins_95ci_lo": lo95,
            "wins_95ci_hi": hi95,
            "pf_5yr": round(park_factors.get(team, float("nan")), 3)
            if park_factors else float("nan"),
        })
    (
        pd.DataFrame(rows)
        .sort_values(["league", "median_wins"], ascending=[True, False])
        .to_csv(OUT_DIR / f"team_sim_{TARGET_YEAR}.csv",
                index=False, encoding="utf-8-sig")
    )
    print(f"Saved: {OUT_DIR / f'team_sim_{TARGET_YEAR}.csv'}")
    _log_elapsed("team_simulation_total", t0)


if __name__ == "__main__":
    main()
