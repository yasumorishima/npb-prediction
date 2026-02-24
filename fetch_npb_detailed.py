"""
NPB公式サイトから詳細打撃成績を取得
データソース: npb.jp/bis/{year}/stats/

baseball-data.comにはない二塁打(2B)・三塁打(3B)・犠飛(SF)を取得し、
wOBA/wRC+算出に使う。

取得カラム: G,PA,AB,R,H,2B,3B,HR,TB,RBI,SB,CS,SH,SF,BB,IBB,HBP,SO,GDP,AVG,SLG,OBP
"""

import pandas as pd
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# NPB公式のチームコード（idb1_{code}.html）
# オリックスは2015-2017が"bs"、2018以降が"b"
TEAM_CODES = {
    "広島": "c", "中日": "d", "DeNA": "db", "巨人": "g",
    "ヤクルト": "s", "阪神": "t",
    "オリックス": "b", "楽天": "e", "日本ハム": "f",
    "ソフトバンク": "h", "西武": "l", "ロッテ": "m",
}

# オリックスの年度別コード（2015-2017は"bs"、2018以降は"b"）
ORIX_CODE_BY_YEAR = {y: "bs" if y <= 2017 else "b" for y in range(2015, 2026)}

YEARS = list(range(2015, 2026))

# 2025年はカラム数が異なる（23列: flag列なし）
COL_NAMES_2025 = [
    "player", "G", "PA", "AB", "R", "H", "2B", "3B", "HR",
    "TB", "RBI", "SB", "CS", "SH", "SF", "BB", "IBB", "HBP", "SO",
    "GDP", "AVG", "SLG", "OBP",
]

# npb.jpテーブルのカラムマッピング（row index 1がヘッダー）
COL_NAMES = [
    "flag", "player", "G", "PA", "AB", "R", "H", "2B", "3B", "HR",
    "TB", "RBI", "SB", "CS", "SH", "SF", "BB", "IBB", "HBP", "SO",
    "GDP", "AVG", "SLG", "OBP",
]


def fetch_team_batting(year: int, team_name: str, team_code: str) -> pd.DataFrame | None:
    """1チーム・1年度の打撃成績を取得"""
    # オリックスの年度別コード対応
    if team_name == "オリックス":
        team_code = ORIX_CODE_BY_YEAR.get(year, team_code)
    url = f"https://npb.jp/bis/{year}/stats/idb1_{team_code}.html"

    try:
        tables = pd.read_html(url, encoding="utf-8")
    except Exception as e:
        print(f"  ERROR {team_name} {year}: {e}")
        return None

    if not tables:
        return None

    df = tables[0]

    # ヘッダー行（row 0=注釈, row 1=カラム名）を除去してデータ行のみ
    if len(df) < 3:
        return None

    df = df.iloc[2:].reset_index(drop=True)

    # カラム名を設定（2025年はflag列なしの23カラム）
    if len(df.columns) == len(COL_NAMES):
        df.columns = COL_NAMES
        df = df.drop(columns=["flag"])
    elif len(df.columns) == len(COL_NAMES_2025):
        df.columns = COL_NAMES_2025
    else:
        print(f"  WARNING {team_name} {year}: expected {len(COL_NAMES)} or {len(COL_NAMES_2025)} cols, got {len(df.columns)}")
        return None

    # 選手名クリーニング（全角スペース→半角）
    df["player"] = df["player"].astype(str).str.replace("　", " ").str.strip()

    # 数値カラムに変換
    num_cols = [c for c in df.columns if c not in ("player",)]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # NaN行・投手等のほぼ空行を除去
    df = df.dropna(subset=["player", "PA"])
    df = df[df["PA"] > 0]

    df["team"] = team_name
    df["year"] = year

    return df


def fetch_all_batting() -> pd.DataFrame:
    """全12チーム×全年度の打撃成績を取得"""
    all_dfs = []

    for year in YEARS:
        print(f"\n[{year}]")
        for team_name, team_code in TEAM_CODES.items():
            df = fetch_team_batting(year, team_name, team_code)
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                print(f"  {team_name}: {len(df)} players")
            time.sleep(0.5)  # 礼儀正しく

    if not all_dfs:
        print("No data fetched")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def main():
    print("=" * 60)
    print("NPB詳細打撃成績取得 (npb.jp)")
    print("=" * 60)

    df = fetch_all_batting()
    if len(df) > 0:
        out_path = DATA_DIR / "npb_batting_detailed_2015_2025.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved: {out_path}")
        print(f"  Rows: {len(df)}")
        print(f"  Years: {sorted(df['year'].unique())}")
        print(f"  Columns: {list(df.columns)}")
        print(f"\n  Sample (2024, top 3 by PA):")
        sample = df[df["year"] == 2024].nlargest(3, "PA")
        print(sample[["player", "team", "PA", "H", "2B", "3B", "HR", "BB", "SF"]].to_string(index=False))


if __name__ == "__main__":
    main()
