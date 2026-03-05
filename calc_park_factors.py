"""
NPB パークファクター計算スクリプト
fetch_npb_games.py で取得した試合別スコアから計算する

パークファクターの計算式（Baseball Reference 標準）:
  PF = ((HomeRS + HomeRA) / HomeG) / ((AwayRS + AwayRA) / AwayG)

  PF > 1.00: 打者有利（点が入りやすい球場）
  PF = 1.00: 中立
  PF < 1.00: 投手有利（点が入りにくい球場）

出力:
  data/projections/npb_park_factors.csv
    columns: year, team, stadium, home_G, home_RS, home_RA,
             away_G, away_RS, away_RA, PF, PF_5yr
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROJ_DIR = DATA_DIR / "projections"
PROJ_DIR.mkdir(parents=True, exist_ok=True)

TEAMS = [
    "阪神", "DeNA", "巨人", "中日", "広島", "ヤクルト",
    "ソフトバンク", "日本ハム", "オリックス", "楽天", "西武", "ロッテ",
]

MIN_GAMES = 30   # 最低試合数（これ未満の年度は NaN）


def load_games(years: list[int]) -> pd.DataFrame:
    """複数年の試合データを結合して返す"""
    frames = []
    for year in years:
        path = RAW_DIR / f"npb_games_{year}.csv"
        if not path.exists():
            print(f"  [警告] {path.name} が見つかりません（{year}年はスキップ）")
            continue
        df = pd.read_csv(path)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "試合データCSVが1件もありません。先に fetch_npb_games.py を実行してください。"
        )

    return pd.concat(frames, ignore_index=True)


def calc_team_pf(games: pd.DataFrame, team: str, year: int) -> dict | None:
    """1チーム・1年のパークファクターを計算"""
    yr = games[games["year"] == year]

    # ホームゲーム（このチームがホーム）
    home = yr[yr["home_team"] == team]
    # アウェイゲーム（このチームがビジター）
    away = yr[yr["away_team"] == team]

    home_G = len(home)
    away_G = len(away)

    if home_G < MIN_GAMES or away_G < MIN_GAMES:
        print(f"  [{team} {year}] 試合数不足 (home={home_G}, away={away_G})")
        return None

    home_RS = home["home_score"].sum()   # ホームでの得点
    home_RA = home["away_score"].sum()   # ホームでの失点
    away_RS = away["away_score"].sum()   # アウェイでの得点
    away_RA = away["home_score"].sum()   # アウェイでの失点

    # 得失点が0なら計算不可（データ不備）
    if away_G == 0 or (away_RS + away_RA) == 0:
        return None

    pf = ((home_RS + home_RA) / home_G) / ((away_RS + away_RA) / away_G)

    stadium = home["stadium"].mode()[0] if "stadium" in home.columns and len(home) > 0 else ""

    return {
        "year": year, "team": team, "stadium": stadium,
        "home_G": home_G, "home_RS": home_RS, "home_RA": home_RA,
        "away_G": away_G, "away_RS": away_RS, "away_RA": away_RA,
        "PF": round(pf, 3),
    }


def calc_multiyear_pf(
    games: pd.DataFrame, team: str, year: int, window: int = 5
) -> float | None:
    """複数年平均パークファクター（FanGraphs 方式: 直近 N 年の得失点を合算して計算）"""
    years = list(range(max(year - window + 1, games["year"].min()), year + 1))

    total_home_RS = total_home_RA = total_home_G = 0
    total_away_RS = total_away_RA = total_away_G = 0

    for y in years:
        res = calc_team_pf(games, team, y)
        if res is None:
            continue
        total_home_RS += res["home_RS"]
        total_home_RA += res["home_RA"]
        total_home_G += res["home_G"]
        total_away_RS += res["away_RS"]
        total_away_RA += res["away_RA"]
        total_away_G += res["away_G"]

    if total_away_G == 0 or (total_away_RS + total_away_RA) == 0:
        return None

    return round(
        ((total_home_RS + total_home_RA) / total_home_G)
        / ((total_away_RS + total_away_RA) / total_away_G),
        3,
    )


def main():
    # 全年度の試合データを読み込み
    available_years = sorted(
        int(p.stem.replace("npb_games_", ""))
        for p in RAW_DIR.glob("npb_games_*.csv")
    )
    if not available_years:
        print("試合データが見つかりません。fetch_npb_games.py を先に実行してください。")
        return

    print(f"対象年度: {available_years}")
    games = load_games(available_years)
    print(f"総試合数: {len(games)}")

    records = []
    for year in available_years:
        for team in TEAMS:
            row = calc_team_pf(games, team, year)
            if row:
                row["PF_5yr"] = calc_multiyear_pf(games, team, year, window=5)
                records.append(row)

    if not records:
        print("パークファクターを計算できませんでした（試合データ不足の可能性）")
        return

    df = pd.DataFrame(records)

    # サマリー表示
    print("\n=== 最新年度 パークファクター ===")
    latest = df[df["year"] == df["year"].max()].sort_values("PF", ascending=False)
    print(latest[["team", "stadium", "PF", "PF_5yr", "home_G"]].to_string(index=False))

    out = PROJ_DIR / "npb_park_factors.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
