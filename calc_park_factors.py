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
             away_G, away_RS, away_RA, PF, PF_5yr, renovation_year
"""

from pathlib import Path

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

# 大規模改修年リスト（この年以降のデータのみ PF_5yr 計算に使う）
# 複数回改修がある場合はリストで管理し、対象年以前の最新改修年が使われる
# 参考: 各改修の詳細は npb-park-factors.md 参照
RENOVATION_BREAKS: dict[str, list[int]] = {
    "ソフトバンク": [2015],        # HRテラス設置（左右中間▲6m、フェンス高▲1.64m）
    "ロッテ":       [2019],        # HRラグーン設置（左右中間▲4m、フェンス高▲1.2m）
    "日本ハム":     [2023],        # エスコンフィールド移転（右中間▲6m、フェンス高▲2.95m）
    "楽天":         [2016, 2026],  # 2016: 天然芝化+左中間フェンス低下 / 2026: フェンス前方移設
    "中日":         [2026],        # HRウイング設置（左右中間▲6m、フェンス高▲1.2m）
}


def get_renovation_break(team: str, year: int) -> int | None:
    """チーム・年度に対応する最新の改修年（データ使用開始年）を返す。なければ None。"""
    breaks = RENOVATION_BREAKS.get(team, [])
    valid = [b for b in breaks if b <= year]
    return max(valid) if valid else None


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

    home = yr[yr["home_team"] == team]
    away = yr[yr["away_team"] == team]

    home_G = len(home)
    away_G = len(away)

    if home_G < MIN_GAMES or away_G < MIN_GAMES:
        return None

    home_RS = home["home_score"].sum()
    home_RA = home["away_score"].sum()
    away_RS = away["away_score"].sum()
    away_RA = away["home_score"].sum()

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
    """複数年平均パークファクター（改修年以降のデータのみ使用）

    改修年が window 内にある場合は改修年以前を除外して計算する。
    例: ソフトバンク 2017年・5年平均 → 2015-2017の3年分（2013-2014は除外）
    """
    reno = get_renovation_break(team, year)
    data_start = reno if reno else games["year"].min()
    years = list(range(max(year - window + 1, data_start), year + 1))

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

    if total_home_G == 0 or total_away_G == 0 or (total_away_RS + total_away_RA) == 0:
        return None

    return round(
        ((total_home_RS + total_home_RA) / total_home_G)
        / ((total_away_RS + total_away_RA) / total_away_G),
        3,
    )


def main():
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

    latest_year = max(available_years)
    records = []

    for year in available_years:
        for team in TEAMS:
            row = calc_team_pf(games, team, year)
            if not row:
                continue

            pf_5yr = calc_multiyear_pf(games, team, year, window=5)
            row["PF_5yr"] = pf_5yr

            reno = get_renovation_break(team, year)
            row["renovation_year"] = reno if reno else ""

            records.append(row)

    if not records:
        print("パークファクターを計算できませんでした（試合データ不足の可能性）")
        return

    df = pd.DataFrame(records)

    # 球場別・年度推移サマリー（改修年に * マーク）
    print("\n=== 球場別 年度推移 ===")
    for team in TEAMS:
        t = df[df["team"] == team].sort_values("year")
        if t.empty:
            continue
        reno_years = set(RENOVATION_BREAKS.get(team, []))
        stadium_name = t.iloc[-1]["stadium"]
        print(f"\n{team}（{stadium_name}）")
        print(f"  {'year':>4}  {'PF':>6}  {'PF_5yr':>7}  {'note'}")
        for _, r in t.iterrows():
            mark = " ← 改修" if r["year"] in reno_years else ""
            pf5 = f"{r['PF_5yr']:.3f}" if pd.notna(r["PF_5yr"]) else "  N/A"
            print(f"  {int(r['year']):>4}  {r['PF']:.3f}  {pf5:>7}{mark}")

    out = PROJ_DIR / "npb_park_factors.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
