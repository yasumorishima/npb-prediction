"""
NPBセイバーメトリクス算出（wOBA / wRC+）

wOBA = weighted On-Base Average
  各打撃イベントの得点価値で重み付けした出塁率。
  NPB用の係数はリーグ・年度ごとに算出する。

wRC+ = weighted Runs Created Plus
  リーグ平均を100とした打撃貢献度。
  パークファクター補正は省略（NPB公開データなし）。

参考:
- Tom Tango, The Book (2007)
- FanGraphs wOBA: https://library.fangraphs.com/offense/woba/
- wRC+: https://library.fangraphs.com/offense/wrc/
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "projections"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- wOBA得点価値係数（MLB標準値をベースにNPBリーグ環境でスケーリング） ---
# MLB標準: BB=0.69, HBP=0.72, 1B=0.88, 2B=1.27, 3B=1.62, HR=2.10
# NPBは得点環境が年度で変動するため、リーグ平均wOBAが.320になるようスケーリングする

# 固定の相対得点価値（イベント間の比率はリーグに依存しにくい）
RAW_WEIGHTS = {
    "BB": 0.69,
    "HBP": 0.72,
    "1B": 0.88,
    "2B": 1.27,
    "3B": 1.62,
    "HR": 2.10,
}


def calc_singles(df: pd.DataFrame) -> pd.Series:
    """安打から二塁打・三塁打・本塁打を引いて単打を算出"""
    return df["H"] - df["2B"] - df["3B"] - df["HR"]


def calc_woba_weights(df_year: pd.DataFrame) -> dict:
    """
    年度別のwOBA係数をリーグ得点環境に合わせてスケーリング。
    リーグ全体のwOBA（生の重み付け）を算出し、OBPスケールに合わせる。
    """
    df = df_year.copy()
    df["1B"] = calc_singles(df)

    # リーグ合計
    totals = {}
    for event in ["BB", "HBP", "1B", "2B", "3B", "HR"]:
        totals[event] = df[event].sum()

    denom = df["AB"].sum() + df["BB"].sum() + df["SF"].sum() + df["HBP"].sum()
    if denom == 0:
        return RAW_WEIGHTS.copy()

    # 生のwOBA（リーグ合計）
    raw_woba_num = sum(RAW_WEIGHTS[ev] * totals[ev] for ev in RAW_WEIGHTS)
    raw_woba = raw_woba_num / denom

    # リーグOBPを算出
    league_obp = (df["H"].sum() + df["BB"].sum() + df["HBP"].sum()) / (
        df["AB"].sum() + df["BB"].sum() + df["HBP"].sum() + df["SF"].sum()
    )

    # wOBAスケール = リーグOBP / 生wOBA（wOBAをOBPスケールに揃える）
    if raw_woba > 0:
        woba_scale = league_obp / raw_woba
    else:
        woba_scale = 1.0

    # スケーリングされた係数
    scaled = {ev: w * woba_scale for ev, w in RAW_WEIGHTS.items()}
    return scaled


def calc_woba(df: pd.DataFrame, weights: dict) -> pd.Series:
    """個人wOBAを算出"""
    singles = calc_singles(df)
    numerator = (
        weights["BB"] * df["BB"]
        + weights["HBP"] * df["HBP"]
        + weights["1B"] * singles
        + weights["2B"] * df["2B"]
        + weights["3B"] * df["3B"]
        + weights["HR"] * df["HR"]
    )
    denominator = df["AB"] + df["BB"] + df["SF"] + df["HBP"]
    return numerator / denominator.replace(0, np.nan)


def calc_wraa(woba: pd.Series, league_woba: float, pa: pd.Series, woba_scale: float) -> pd.Series:
    """wRAA (weighted Runs Above Average) = ((wOBA - lgwOBA) / wOBAscale) * PA"""
    return ((woba - league_woba) / woba_scale) * pa


def calc_wrc_plus(woba: pd.Series, league_woba: float, woba_scale: float,
                  league_r_pa: float) -> pd.Series:
    """
    wRC+ = ((wRAA/PA + league_R/PA) / league_R/PA) * 100
    パークファクター補正なし（NPBデータ未公開のため）
    """
    wraa_per_pa = (woba - league_woba) / woba_scale
    wrc_plus = ((wraa_per_pa + league_r_pa) / league_r_pa) * 100
    return wrc_plus


def process_year(df_year: pd.DataFrame, year: int) -> pd.DataFrame:
    """1年度分のwOBA/wRC+を算出"""
    df = df_year.copy()

    # wOBA係数
    weights = calc_woba_weights(df)

    # 個人wOBA
    df["wOBA"] = calc_woba(df, weights)

    # リーグ平均wOBA
    league_woba = df["wOBA"].mean()  # PA加重の方が正確だが、簡易版
    # PA加重リーグ平均wOBA
    total_pa = df["PA"].sum()
    if total_pa > 0:
        league_woba = (df["wOBA"] * df["PA"]).sum() / total_pa

    # wOBAスケール（OBPとwOBAの比率）
    league_obp = (df["H"].sum() + df["BB"].sum() + df["HBP"].sum()) / (
        df["AB"].sum() + df["BB"].sum() + df["HBP"].sum() + df["SF"].sum()
    )
    raw_weights_num = sum(
        RAW_WEIGHTS[ev] * df[ev if ev != "1B" else "H"].sum()  # dummy, recalc below
        for ev in RAW_WEIGHTS
    )
    # 正確に再計算
    df["1B"] = calc_singles(df)
    raw_num = sum(RAW_WEIGHTS[ev] * df[ev].sum() for ev in RAW_WEIGHTS)
    denom = df["AB"].sum() + df["BB"].sum() + df["SF"].sum() + df["HBP"].sum()
    raw_woba = raw_num / denom if denom > 0 else 0
    woba_scale = league_obp / raw_woba if raw_woba > 0 else 1.0

    # リーグ得点/PA
    league_r_pa = df["R"].sum() / total_pa if total_pa > 0 else 0.1

    # wRC+
    df["wRC+"] = calc_wrc_plus(df["wOBA"], league_woba, woba_scale, league_r_pa)

    # wRAA
    df["wRAA"] = calc_wraa(df["wOBA"], league_woba, df["PA"], woba_scale)

    # 1B列は中間計算用なので削除
    df = df.drop(columns=["1B"])

    # メタ情報
    print(f"  [{year}] lgwOBA={league_woba:.3f}, lgOBP={league_obp:.3f}, "
          f"wOBAscale={woba_scale:.3f}, lgR/PA={league_r_pa:.4f}")

    return df


def main():
    print("=" * 60)
    print("NPBセイバーメトリクス算出 (wOBA / wRC+)")
    print("=" * 60)

    # 詳細打撃成績を読み込み
    df = pd.read_csv(RAW_DIR / "npb_batting_detailed_2015_2025.csv")
    print(f"Input: {len(df)} rows, years={sorted(df['year'].unique())}")

    # 年度別にwOBA/wRC+を算出
    results = []
    for year in sorted(df["year"].unique()):
        df_year = df[df["year"] == year].copy()
        df_year = process_year(df_year, year)
        results.append(df_year)

    combined = pd.concat(results, ignore_index=True)

    # 保存
    out_path = OUT_DIR / "npb_sabermetrics_2015_2025.csv"
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_path} ({len(combined)} rows)")

    # 2024年の規定打席以上（443PA）のwOBA/wRC+ Top10
    print("\n--- 2024 wRC+ Top 10 (PA >= 443) ---")
    q = combined[(combined["year"] == 2024) & (combined["PA"] >= 443)]
    top = q.nlargest(10, "wRC+")
    print(top[["player", "team", "PA", "AVG", "OBP", "SLG", "wOBA", "wRC+", "wRAA"]].to_string(
        index=False, float_format="%.3f"))

    # バリデーション: wOBA vs SLG の相関（NPB公式データにOPS列がないためSLGで代用）
    q_all = combined[combined["PA"] >= 200].copy()
    corr_slg = q_all["wOBA"].corr(q_all["SLG"])
    corr_obp = q_all["wOBA"].corr(q_all["OBP"])
    print(f"\nValidation (PA>=200):")
    print(f"  wOBA vs SLG correlation: {corr_slg:.4f}")
    print(f"  wOBA vs OBP correlation: {corr_obp:.4f}")
    print("  (Expected: 0.90+ → high correlation confirms calculation is correct)")


if __name__ == "__main__":
    main()
