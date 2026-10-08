"""
単年PF vs PF_5yr vs 補正なし — 3方式バックテスト完全比較スクリプト

対象: 2018-2025年 全チーム・全年度（96チーム-年度）
指標: 勝利数MAE / RS MAE / RA MAE / 80%CIカバー率
注目: 日本ハム（エスコンフィールド開場2023年前後）

出力:
  data/projections/pf_method_comparison.csv — 全96行、3方式の比較
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# team_sim の共通定数・関数を流用
import team_sim as ts

OUT_DIR = ROOT / "data" / "projections"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SIM = 10_000


# ── 1. データロード ────────────────────────────────────────────────────────────
print("Loading historical Marcel team projections...")
hist, actual = ts.load_historical()

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

# ── 2. PFデータ（単年・5年移動平均）をロード ──────────────────────────────────
print("Loading park factors...")
pf_df = pd.read_csv(f"{ts.NPBP_BASE}/npb_park_factors.csv", encoding="utf-8-sig")
print(f"  Loaded {len(pf_df)} rows  (teams: {pf_df['team'].nunique()}, years: {pf_df['year'].nunique()})")
print(f"  Columns: {list(pf_df.columns)}")

# 日本ハムのPFを表示（確認用）
ham = pf_df[pf_df["team"] == "日本ハム"].sort_values("year")
print("\n  ── 日本ハム PF推移 ──")
for _, r in ham.iterrows():
    print(f"    {int(r['year'])}  PF(単年)={r['PF']:.3f}  PF_5yr={r['PF_5yr']:.3f}  stadium={r.get('stadium', '?')}")

pf_map_1yr: dict[tuple[int, str], float] = {
    (int(r["year"]), str(r["team"])): float(r["PF"])
    for _, r in pf_df.iterrows()
}
pf_map_5yr: dict[tuple[int, str], float] = {
    (int(r["year"]), str(r["team"])): float(r["PF_5yr"])
    for _, r in pf_df.iterrows()
}

# ── 3. バックテスト実行（3条件） ───────────────────────────────────────────────
print(f"\nRunning 3 backtests (N={N_SIM:,})...")
rng_base = np.random.default_rng(SEED)
rng_1yr  = np.random.default_rng(SEED)
rng_5yr  = np.random.default_rng(SEED)

df_base = ts._run_one_backtest(merged, rng_base, N_SIM, pf_map=None)
df_1yr  = ts._run_one_backtest(merged, rng_1yr,  N_SIM, pf_map=pf_map_1yr)
df_5yr  = ts._run_one_backtest(merged, rng_5yr,  N_SIM, pf_map=pf_map_5yr)

# RS/RA誤差を追加（pred_RS - actual_RS）
for df in [df_base, df_1yr, df_5yr]:
    df["rs_error"] = (df["pred_RS"] - df["actual_RS"]).round(1)
    df["ra_error"] = (df["pred_RA"] - df["actual_RA"]).round(1)


# ── 4. 全体サマリー ──────────────────────────────────────────────────────────
def stats(df: pd.DataFrame) -> dict:
    return {
        "wins_mae":  df["error"].abs().mean(),
        "wins_bias": df["error"].mean(),
        "wins_cov":  df["covered"].mean(),
        "rs_mae":    df["rs_error"].abs().mean(),
        "ra_mae":    df["ra_error"].abs().mean(),
    }

s_base = stats(df_base)
s_1yr  = stats(df_1yr)
s_5yr  = stats(df_5yr)

print("\n" + "=" * 75)
print("【全体サマリー】  2018-2025  96チーム-年度")
print("=" * 75)
print(f"{'指標':20s}  {'No PF':>9}  {'単年PF':>9}  {'PF_5yr':>9}  {'Δ(1yr)':>8}  {'Δ(5yr)':>8}")
print("-" * 75)
for key, label in [("wins_mae", "勝利数MAE"), ("wins_bias", "勝利数Bias"),
                   ("wins_cov", "80%CI Coverage"),
                   ("rs_mae", "RS MAE"), ("ra_mae", "RA MAE")]:
    fmt = ".1%" if key == "wins_cov" else ".2f"
    print(f"{label:20s}  {s_base[key]:{fmt}}  {s_1yr[key]:{fmt}}  {s_5yr[key]:{fmt}}  "
          f"{s_1yr[key] - s_base[key]:+8.2f}  {s_5yr[key] - s_base[key]:+8.2f}")


# ── 5. 年度別内訳 ─────────────────────────────────────────────────────────────
print("\n" + "=" * 95)
print("【年度別 勝利数MAE / RS MAE / RA MAE】")
print("=" * 95)
print(f"{'Year':>5}  "
      f"{'W_MAE(base)':>11}  {'W_MAE(1yr)':>10}  {'W_MAE(5yr)':>10}  "
      f"{'RS_MAE(base)':>12}  {'RS_MAE(1yr)':>11}  {'RS_MAE(5yr)':>11}  "
      f"{'RA_MAE(base)':>12}  {'RA_MAE(1yr)':>11}  {'RA_MAE(5yr)':>11}")
print("-" * 115)
for yr in sorted(df_base["year"].unique()):
    b = df_base[df_base["year"] == yr]
    o = df_1yr[df_1yr["year"] == yr]
    f = df_5yr[df_5yr["year"] == yr]
    print(f"{yr:>5}  "
          f"{b['error'].abs().mean():11.2f}  {o['error'].abs().mean():10.2f}  {f['error'].abs().mean():10.2f}  "
          f"{b['rs_error'].abs().mean():12.1f}  {o['rs_error'].abs().mean():11.1f}  {f['rs_error'].abs().mean():11.1f}  "
          f"{b['ra_error'].abs().mean():12.1f}  {o['ra_error'].abs().mean():11.1f}  {f['ra_error'].abs().mean():11.1f}")


# ── 6. 日本ハム 全年度詳細（エスコン開場前後） ────────────────────────────────
print("\n" + "=" * 100)
print("【日本ハム 詳細】  エスコンフィールド開場: 2023年")
print("=" * 100)
print(f"{'Year':>5}  {'PF(単年)':>8}  {'PF_5yr':>7}  "
      f"{'W_base':>6}  {'W_1yr':>6}  {'W_5yr':>6}  {'Actual_W':>8}  "
      f"{'Err_base':>8}  {'Err_1yr':>7}  {'Err_5yr':>7}  "
      f"{'RS_base':>7}  {'RS_1yr':>7}  {'RS_5yr':>7}  "
      f"{'RA_base':>7}  {'RA_1yr':>7}  {'RA_5yr':>7}  {'Actual_RS':>9}  {'Actual_RA':>9}")
print("-" * 130)

ham_base = df_base[df_base["team"] == "日本ハム"].sort_values("year").reset_index(drop=True)
ham_1yr  = df_1yr[df_1yr["team"] == "日本ハム"].sort_values("year").reset_index(drop=True)
ham_5yr  = df_5yr[df_5yr["team"] == "日本ハム"].sort_values("year").reset_index(drop=True)

for i in range(len(ham_base)):
    yr = int(ham_base.loc[i, "year"])
    pf1 = pf_map_1yr.get((yr, "日本ハム"), float("nan"))
    pf5 = pf_map_5yr.get((yr, "日本ハム"), float("nan"))
    marker = " ← エスコン開場" if yr == 2023 else (" ← 移転前最終" if yr == 2022 else "")
    print(f"{yr:>5}  {pf1:8.3f}  {pf5:7.3f}  "
          f"{ham_base.loc[i, 'median_wins']:6.1f}  {ham_1yr.loc[i, 'median_wins']:6.1f}  {ham_5yr.loc[i, 'median_wins']:6.1f}  "
          f"{ham_base.loc[i, 'actual_W']:8.0f}  "
          f"{ham_base.loc[i, 'error']:+8.1f}  {ham_1yr.loc[i, 'error']:+7.1f}  {ham_5yr.loc[i, 'error']:+7.1f}  "
          f"{ham_base.loc[i, 'pred_RS']:7.1f}  {ham_1yr.loc[i, 'pred_RS']:7.1f}  {ham_5yr.loc[i, 'pred_RS']:7.1f}  "
          f"{ham_base.loc[i, 'pred_RA']:7.1f}  {ham_1yr.loc[i, 'pred_RA']:7.1f}  {ham_5yr.loc[i, 'pred_RA']:7.1f}  "
          f"{ham_base.loc[i, 'actual_RS']:9.0f}  {ham_base.loc[i, 'actual_RA']:9.0f}{marker}")


# ── 7. チーム別全年度 3方式比較 ──────────────────────────────────────────────
print("\n" + "=" * 85)
print("【チーム別 集計（8年平均）  PF_5yr低い順】")
print("=" * 85)
team_rows = []
for team in df_base["team"].unique():
    b = df_base[df_base["team"] == team]
    o = df_1yr[df_1yr["team"] == team]
    f = df_5yr[df_5yr["team"] == team]
    pf5_avg = df_5yr[df_5yr["team"] == team]["pf_5yr"].mean()
    team_rows.append({
        "team": team,
        "pf5_avg": pf5_avg,
        "w_mae_base": b["error"].abs().mean(),
        "w_mae_1yr":  o["error"].abs().mean(),
        "w_mae_5yr":  f["error"].abs().mean(),
        "rs_mae_base": b["rs_error"].abs().mean(),
        "rs_mae_1yr":  o["rs_error"].abs().mean(),
        "rs_mae_5yr":  f["rs_error"].abs().mean(),
        "ra_mae_base": b["ra_error"].abs().mean(),
        "ra_mae_1yr":  o["ra_error"].abs().mean(),
        "ra_mae_5yr":  f["ra_error"].abs().mean(),
    })

team_df = pd.DataFrame(team_rows).sort_values("pf5_avg")

print(f"{'チーム':12s}  {'PF5_avg':>7}  "
      f"{'W_MAE(base)':>11}  {'W_MAE(1yr)':>10}  {'W_MAE(5yr)':>10}  "
      f"{'RS_MAE(base)':>12}  {'RS_MAE(1yr)':>11}  {'RS_MAE(5yr)':>11}")
print("-" * 95)
for _, r in team_df.iterrows():
    print(f"{r['team']:12s}  {r['pf5_avg']:7.3f}  "
          f"{r['w_mae_base']:11.2f}  {r['w_mae_1yr']:10.2f}  {r['w_mae_5yr']:10.2f}  "
          f"{r['rs_mae_base']:12.1f}  {r['rs_mae_1yr']:11.1f}  {r['rs_mae_5yr']:11.1f}")


# ── 8. 全データをCSV保存 ──────────────────────────────────────────────────────
df_out = df_base[["year", "league", "team", "actual_W", "actual_RS", "actual_RA"]].copy()
# PF値を追加
df_out["pf_1yr"] = df_out.apply(lambda r: pf_map_1yr.get((int(r["year"]), r["team"]), None), axis=1)
df_out["pf_5yr"] = df_out.apply(lambda r: pf_map_5yr.get((int(r["year"]), r["team"]), None), axis=1)

for suffix, df_src in [("base", df_base), ("1yr", df_1yr), ("5yr", df_5yr)]:
    df_src2 = df_src.reset_index(drop=True)
    df_out[f"pred_RS_{suffix}"]    = df_src2["pred_RS"]
    df_out[f"pred_RA_{suffix}"]    = df_src2["pred_RA"]
    df_out[f"median_wins_{suffix}"] = df_src2["median_wins"]
    df_out[f"error_wins_{suffix}"] = df_src2["error"]
    df_out[f"rs_error_{suffix}"]   = df_src2["rs_error"]
    df_out[f"ra_error_{suffix}"]   = df_src2["ra_error"]
    df_out[f"covered_{suffix}"]    = df_src2["covered"]

out_path = OUT_DIR / "pf_method_comparison.csv"
df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\nSaved -> {out_path}  ({len(df_out)} rows, {len(df_out.columns)} cols)")
print("\nDone.")
