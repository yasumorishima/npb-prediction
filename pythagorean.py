"""
ピタゴラス勝率によるNPBチーム勝率予測

ピタゴラス勝率: 得点^k / (得点^k + 失点^k)
- MLB最適指数: k = 1.83
- NPB最適指数: k = 1.72（先行研究より）

データソース: プロ野球データFreak (baseball-data.com)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "projections"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://baseball-data.com"

# NPB最適指数
NPB_EXPONENT = 1.72
MLB_EXPONENT = 1.83

YEARS = list(range(2015, 2026))

# 順位表カラムの正規化マッピング
STANDINGS_COLS = {
    "順位": "rank",
    "チーム": "team",
    "試合": "G",
    "勝利": "W",
    "敗北": "L",
    "引分": "D",
    "勝率": "WPCT",
    "得点": "RS",
    "失点": "RA",
    "平均得点": "RS_avg",
    "平均失点": "RA_avg",
    "得失点差": "RD",
    "打率": "AVG",
    "本塁打": "HR",
    "盗塁": "SB",
    "防御率": "ERA",
}


def build_standings_url(year: int) -> str:
    current_year = 2026
    yy = str(year)[2:]
    if year == current_year:
        return f"{BASE_URL}/team/standings.html"
    else:
        return f"{BASE_URL}/{yy}/team/standings.html"


def fetch_standings(year: int) -> pd.DataFrame | None:
    """指定年度の順位表（セ+パ）を取得"""
    url = build_standings_url(year)
    print(f"  Fetching: {url}")

    try:
        tables = pd.read_html(url, encoding="utf-8")
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    if not tables:
        print(f"  No tables found")
        return None

    all_teams = []
    for i, tbl in enumerate(tables):
        # MultiIndexフラット化
        if isinstance(tbl.columns, pd.MultiIndex):
            tbl.columns = [col[0] for col in tbl.columns]
        tbl.columns = [str(c).replace(" ", "").replace("　", "") for c in tbl.columns]

        # 英語カラム名に変換
        tbl = tbl.rename(columns=STANDINGS_COLS)

        if "team" in tbl.columns and "RS" in tbl.columns and "RA" in tbl.columns:
            tbl["league"] = "CL" if i == 0 else "PL"
            tbl["year"] = year
            all_teams.append(tbl)

    if not all_teams:
        print(f"  No standings tables found")
        return None

    df = pd.concat(all_teams, ignore_index=True)
    # 数値変換
    for col in ["G", "W", "L", "D", "RS", "RA"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  -> {len(df)} teams")
    return df


def fetch_all_standings() -> pd.DataFrame:
    """全年度の順位表を結合"""
    all_dfs = []
    for year in YEARS:
        print(f"[{year}] standings")
        df = fetch_standings(year)
        if df is not None and len(df) > 0:
            all_dfs.append(df)
        time.sleep(1)

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def pythagorean_wpct(rs: float, ra: float, k: float = NPB_EXPONENT) -> float:
    """ピタゴラス勝率を計算"""
    if rs == 0 and ra == 0:
        return 0.5
    return rs**k / (rs**k + ra**k)


def evaluate_pythagorean(df: pd.DataFrame) -> pd.DataFrame:
    """全チーム・全年度でピタゴラス勝率を計算し、実勝率と比較"""
    results = []
    for _, row in df.iterrows():
        if pd.isna(row["RS"]) or pd.isna(row["RA"]):
            continue

        actual_wpct = row["W"] / (row["W"] + row["L"]) if (row["W"] + row["L"]) > 0 else 0
        pyth_npb = pythagorean_wpct(row["RS"], row["RA"], NPB_EXPONENT)
        pyth_mlb = pythagorean_wpct(row["RS"], row["RA"], MLB_EXPONENT)

        # 勝数に換算（引分を除いた試合数ベース）
        decisions = row["W"] + row["L"]
        pyth_wins_npb = pyth_npb * decisions
        pyth_wins_mlb = pyth_mlb * decisions

        results.append({
            "year": row["year"],
            "league": row["league"],
            "team": row["team"],
            "G": row["G"],
            "W": row["W"],
            "L": row["L"],
            "D": row.get("D", 0),
            "actual_WPCT": round(actual_wpct, 3),
            "RS": row["RS"],
            "RA": row["RA"],
            "pyth_WPCT_npb": round(pyth_npb, 3),
            "pyth_WPCT_mlb": round(pyth_mlb, 3),
            "pyth_W_npb": round(pyth_wins_npb, 1),
            "pyth_W_mlb": round(pyth_wins_mlb, 1),
            "diff_W_npb": round(pyth_wins_npb - row["W"], 1),
            "diff_W_mlb": round(pyth_wins_mlb - row["W"], 1),
        })

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("ピタゴラス勝率 NPBチーム勝率予測")
    print("=" * 60)

    # データ取得
    csv_path = RAW_DIR / "npb_standings_2015_2025.csv"
    if csv_path.exists():
        print(f"\nLoading cached: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("\nFetching standings data...")
        df = fetch_all_standings()
        if len(df) > 0:
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"\nSaved: {csv_path} ({len(df)} rows)")

    if len(df) == 0:
        print("No data available")
        return

    # ピタゴラス勝率計算
    print(f"\n--- ピタゴラス勝率 vs 実勝率 ---")
    result = evaluate_pythagorean(df)

    # 全体の精度評価
    mae_npb = (result["pyth_W_npb"] - result["W"]).abs().mean()
    mae_mlb = (result["pyth_W_mlb"] - result["W"]).abs().mean()
    rmse_npb = np.sqrt(((result["pyth_W_npb"] - result["W"]) ** 2).mean())
    rmse_mlb = np.sqrt(((result["pyth_W_mlb"] - result["W"]) ** 2).mean())

    print(f"\n全{len(result)}チーム-年の精度比較:")
    print(f"  NPB指数(k={NPB_EXPONENT}): MAE={mae_npb:.2f}勝  RMSE={rmse_npb:.2f}勝")
    print(f"  MLB指数(k={MLB_EXPONENT}): MAE={mae_mlb:.2f}勝  RMSE={rmse_mlb:.2f}勝")

    # 年度別の精度
    print(f"\n年度別MAE (勝数):")
    print(f"  {'Year':>6} {'NPB(k=1.72)':>12} {'MLB(k=1.83)':>12} {'teams':>6}")
    for year in sorted(result["year"].unique()):
        yr_data = result[result["year"] == year]
        mae_n = (yr_data["pyth_W_npb"] - yr_data["W"]).abs().mean()
        mae_m = (yr_data["pyth_W_mlb"] - yr_data["W"]).abs().mean()
        print(f"  {year:>6} {mae_n:>12.2f} {mae_m:>12.2f} {len(yr_data):>6}")

    # 2024年の詳細
    print(f"\n--- 2024年 詳細 ---")
    yr_2024 = result[result["year"] == 2024].sort_values("W", ascending=False)
    if len(yr_2024) > 0:
        print(yr_2024[["league", "team", "W", "L", "RS", "RA",
                        "actual_WPCT", "pyth_WPCT_npb", "pyth_W_npb", "diff_W_npb"]].to_string(index=False))

    # 2025年詳細
    print(f"\n--- 2025年 詳細 ---")
    yr_2025 = result[result["year"] == 2025].sort_values("W", ascending=False)
    if len(yr_2025) > 0:
        print(yr_2025[["league", "team", "W", "L", "RS", "RA",
                        "actual_WPCT", "pyth_WPCT_npb", "pyth_W_npb", "diff_W_npb"]].to_string(index=False))

    # CSV出力
    out_path = OUT_DIR / "pythagorean_2015_2025.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
