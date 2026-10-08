"""
PF補正バックテスト詳細分析スクリプト。
backtest_comparison.csv を読み込み、ブログ執筆用の分析結果を出力する。

出力:
  1. チーム別 RS/RA delta 平均（PF高い順）
  2. 年度別 MAE・CI coverage 比較
  3. coverage 変化があったチーム-年度（covered_base ≠ covered_pf）
  4. PF極端値チームの全年度詳細（バンテリン / エスコン / 神宮 / ZOZOマリン）
  5. data/projections/pf_analysis_summary.csv として保存
"""

from pathlib import Path
import pandas as pd

ROOT   = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "data" / "projections" / "backtest_comparison.csv"
OUT_CSV = ROOT / "data" / "projections" / "pf_analysis_summary.csv"

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(IN_CSV, encoding="utf-8-sig")
print(f"Loaded: {len(df)} rows, {len(df.columns)} cols  ({df['year'].min()}-{df['year'].max()})")

# ── 1. チーム別 RS/RA delta 平均（PF高い順） ──────────────────────────────────
print("\n" + "=" * 70)
print("1. チーム別 PF補正の影響（2018-2025 平均）")
print("=" * 70)
team_stats = (
    df.groupby("team")
    .agg(
        pf_avg   =("pf_5yr",       "mean"),
        rs_delta =("rs_delta",      "mean"),
        ra_delta =("ra_delta",      "mean"),
        wins_delta=("wins_delta",   "mean"),
        mae_base =("error_base",   lambda x: x.abs().mean()),
        mae_pf   =("error_pf",     lambda x: x.abs().mean()),
        cov_base =("covered_base", "mean"),
        cov_pf   =("covered_pf",   "mean"),
        n        =("year",         "count"),
    )
    .reset_index()
    .sort_values("pf_avg")
)
print(f"{'チーム':12s}  {'PF_avg':>7}  {'RS_Δ':>7}  {'RA_Δ':>7}  "
      f"{'W_Δ':>6}  {'MAE_base':>8}  {'MAE_pf':>7}  {'Cov_base':>8}  {'Cov_pf':>7}")
print("-" * 80)
for _, r in team_stats.iterrows():
    print(f"{r['team']:12s}  {r['pf_avg']:7.3f}  {r['rs_delta']:+7.1f}  {r['ra_delta']:+7.1f}  "
          f"{r['wins_delta']:+6.2f}  {r['mae_base']:8.2f}  {r['mae_pf']:7.2f}  "
          f"{r['cov_base']:8.1%}  {r['cov_pf']:7.1%}")

# ── 2. 年度別 MAE・CI coverage 比較 ───────────────────────────────────────────
print("\n" + "=" * 70)
print("2. 年度別 MAE・CI coverage 比較")
print("=" * 70)
yr_stats = (
    df.groupby("year")
    .agg(
        mae_base  =("error_base",   lambda x: x.abs().mean()),
        mae_pf    =("error_pf",     lambda x: x.abs().mean()),
        cov_base  =("covered_base", "mean"),
        cov_pf    =("covered_pf",   "mean"),
        bias_base =("error_base",   "mean"),
        bias_pf   =("error_pf",     "mean"),
    )
    .reset_index()
)
print(f"{'Year':>5}  {'MAE_base':>8}  {'MAE_pf':>7}  {'MAE_Δ':>7}  "
      f"{'Cov_base':>8}  {'Cov_pf':>7}  {'Bias_base':>9}  {'Bias_pf':>8}")
print("-" * 75)
for _, r in yr_stats.iterrows():
    print(f"{int(r['year']):>5}  {r['mae_base']:8.2f}  {r['mae_pf']:7.2f}  "
          f"{r['mae_pf'] - r['mae_base']:+7.2f}  "
          f"{r['cov_base']:8.1%}  {r['cov_pf']:7.1%}  "
          f"{r['bias_base']:+9.2f}  {r['bias_pf']:+8.2f}")

# ── 3. coverage 変化があったチーム-年度 ────────────────────────────────────────
print("\n" + "=" * 70)
print("3. PF補正でカバレッジが変化したチーム-年度")
print("=" * 70)
changed = df[df["covered_base"] != df["covered_pf"]].copy()
if len(changed) == 0:
    print("  (なし)")
else:
    for _, r in changed.iterrows():
        direction = "❌→✅ 改善" if r["covered_pf"] else "✅→❌ 悪化"
        print(f"  {int(r['year'])} {r['team']:10s}  PF={r['pf_5yr']:.3f}  "
              f"actual={r['actual_W']:.0f}勝  "
              f"base={r['median_wins_base']:.1f}[{r['ci_lo_base']:.0f},{r['ci_hi_base']:.0f}]  "
              f"pf={r['median_wins_pf']:.1f}[{r['ci_lo_pf']:.0f},{r['ci_hi_pf']:.0f}]  "
              f"{direction}")

# ── 4. 極端PF球場の全年度詳細 ─────────────────────────────────────────────────
FOCUS_TEAMS = ["中日", "楽天", "阪神", "オリックス", "西武", "ソフトバンク",
               "広島", "巨人", "ロッテ", "DeNA", "日本ハム", "ヤクルト"]
print("\n" + "=" * 70)
print("4. 全チーム全年度 詳細（PF低い順）")
print("=" * 70)
df_detail = df.sort_values(["pf_5yr", "year"])
print(f"{'year':>4}  {'team':10s}  {'PF':>6}  "
      f"{'RS_base':>7}  {'RS_pf':>6}  {'RS_Δ':>6}  "
      f"{'RA_base':>7}  {'RA_pf':>6}  {'RA_Δ':>6}  "
      f"{'W_base':>6}  {'W_pf':>5}  {'W_Δ':>5}  "
      f"{'actual_W':>8}  {'err_base':>8}  {'err_pf':>6}")
print("-" * 115)
for _, r in df_detail.iterrows():
    print(f"{int(r['year']):>4}  {r['team']:10s}  {r['pf_5yr']:6.3f}  "
          f"{r['pred_RS_base']:7.1f}  {r['pred_RS_pf']:6.1f}  {r['rs_delta']:+6.1f}  "
          f"{r['pred_RA_base']:7.1f}  {r['pred_RA_pf']:6.1f}  {r['ra_delta']:+6.1f}  "
          f"{r['median_wins_base']:6.1f}  {r['median_wins_pf']:5.1f}  {r['wins_delta']:+5.1f}  "
          f"{r['actual_W']:8.0f}  {r['error_base']:+8.1f}  {r['error_pf']:+6.1f}")

# ── 5. RS/RA予測精度の変化（MAE） ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. RS・RA予測精度（MAE） No PF vs With PF")
print("=" * 70)
rs_mae_base = df["rs_error_base"].abs().mean()
rs_mae_pf   = df["rs_error_pf"].abs().mean()
ra_mae_base = df["ra_error_base"].abs().mean()
ra_mae_pf   = df["ra_error_pf"].abs().mean()
print(f"  RS MAE: {rs_mae_base:.1f}  →  {rs_mae_pf:.1f}  (Δ {rs_mae_pf - rs_mae_base:+.1f})")
print(f"  RA MAE: {ra_mae_base:.1f}  →  {ra_mae_pf:.1f}  (Δ {ra_mae_pf - ra_mae_base:+.1f})")

# 全体 RS bias
rs_bias_base = df["rs_error_base"].mean()
rs_bias_pf   = df["rs_error_pf"].mean()
ra_bias_base = df["ra_error_base"].mean()
ra_bias_pf   = df["ra_error_pf"].mean()
print(f"  RS Bias: {rs_bias_base:+.1f}  →  {rs_bias_pf:+.1f}  (Δ {rs_bias_pf - rs_bias_base:+.1f})")
print(f"  RA Bias: {ra_bias_base:+.1f}  →  {ra_bias_pf:+.1f}  (Δ {ra_bias_pf - ra_bias_base:+.1f})")

# ── 6. Summary CSV保存 ────────────────────────────────────────────────────────
team_stats.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\nSaved -> {OUT_CSV}")
print("\nDone.")
