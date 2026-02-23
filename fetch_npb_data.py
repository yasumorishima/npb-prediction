"""
NPB成績データ取得スクリプト
データソース: プロ野球データFreak (baseball-data.com)
pandas.read_html() でテーブルを直接取得

取得データ:
- 打者成績 (2015-2025): 打率,試合,打席,打数,安打,HR,打点,盗塁,四球,死球,三振,犠打,併殺,出塁率,長打率,OPS,RC27,XR27
- 投手成績 (2015-2025): 防御率,試合,勝,敗,S,H,勝率,打者,投球回,被安打,被HR,四球,死球,奪三振,失点,自責,WHIP,DIPS
"""

import pandas as pd
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 年度コード: 2025→"25", 2024→"24", ...
# URL: baseball-data.com/{yy}/stats/hitter-all/tpa-1.html (全打者)
#      baseball-data.com/{yy}/stats/pitcher-all/era-1.html (全投手)
# 2025以降(当年)はyy部分なし: baseball-data.com/stats/hitter-all/tpa-1.html

BASE_URL = "https://baseball-data.com"

# チームコード（参考: 個別チーム取得用）
TEAM_CODES = {
    "阪神": "t", "DeNA": "yb", "巨人": "g",
    "中日": "d", "広島": "c", "ヤクルト": "s",
    "ソフトバンク": "h", "日本ハム": "f", "オリックス": "bs",
    "楽天": "e", "西武": "l", "ロッテ": "m",
}

YEARS = list(range(2015, 2026))  # 2015-2025 (Marcel法に必要な過去データ)


def build_url(year: int, stat_type: str) -> str:
    """年度別URL生成"""
    current_year = 2026
    yy = str(year)[2:]  # "15", "16", ...

    if stat_type == "hitter":
        path = f"stats/hitter-all/tpa-1.html"
    else:
        path = f"stats/pitcher-all/era-1.html"

    if year == current_year:
        return f"{BASE_URL}/{path}"
    else:
        return f"{BASE_URL}/{yy}/{path}"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """MultiIndexカラムをフラット化し、空白を除去"""
    if isinstance(df.columns, pd.MultiIndex):
        # MultiIndexの場合、最初のレベルを使う
        df.columns = [col[0] for col in df.columns]
    # カラム名から全角・半角スペースを除去
    df.columns = [str(c).replace(" ", "").replace("　", "") for c in df.columns]
    return df


# カラム名の正規化マッピング（打者）
HITTER_COLS = {
    "順位": "rank", "選手名": "player", "チーム": "team",
    "打率": "AVG", "試合": "G", "打席数": "PA", "打数": "AB",
    "安打": "H", "本塁打": "HR", "打点": "RBI", "盗塁": "SB",
    "四球": "BB", "死球": "HBP", "三振": "SO", "犠打": "SH",
    "併殺打": "GDP", "出塁率": "OBP", "長打率": "SLG",
    "OPS": "OPS", "RC27": "RC27", "XR27": "XR27",
}

# カラム名の正規化マッピング（投手）
PITCHER_COLS = {
    "順位": "rank", "選手名": "player", "チーム": "team",
    "防御率": "ERA", "試合": "G", "勝利": "W", "敗北": "L",
    "セーブ": "SV", "セlブ": "SV", "ホールド": "HLD", "ホlルド": "HLD", "勝率": "WPCT",
    "打者": "BF", "投球回": "IP", "被安打": "HA", "被本塁打": "HRA",
    "与四球": "BB", "与死球": "HBP", "奪三振": "SO",
    "失点": "R", "自責点": "ER", "WHIP": "WHIP", "DIPS": "DIPS",
}


def fetch_stats(year: int, stat_type: str) -> pd.DataFrame | None:
    """指定年度の成績テーブルを取得"""
    url = build_url(year, stat_type)
    print(f"  Fetching: {url}")

    try:
        tables = pd.read_html(url, encoding="utf-8")
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    if not tables:
        print(f"  No tables found")
        return None

    # 最大行数のテーブルを選択
    df = max(tables, key=len)

    # カラム整形
    df = clean_columns(df)

    # 英語カラム名に変換
    col_map = HITTER_COLS if stat_type == "hitter" else PITCHER_COLS
    df = df.rename(columns=col_map)

    # rank列削除
    if "rank" in df.columns:
        df = df.drop(columns=["rank"])

    # year列追加
    df["year"] = year

    # 空行・NaN行を削除
    if "player" in df.columns:
        df = df.dropna(subset=["player"])
        df = df[df["player"].astype(str).str.match(r"^(?!\d+$).+")]

    print(f"  -> {len(df)} rows")
    return df


def fetch_all(stat_type: str) -> pd.DataFrame:
    """全年度のデータを結合"""
    all_dfs = []
    for year in YEARS:
        print(f"[{year}] {stat_type}")
        df = fetch_stats(year, stat_type)
        if df is not None and len(df) > 0:
            all_dfs.append(df)
        time.sleep(1)  # 礼儀正しく

    if not all_dfs:
        print(f"No data fetched for {stat_type}")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def main():
    print("=" * 60)
    print("NPB成績データ取得 (baseball-data.com)")
    print("=" * 60)

    # 打者成績
    print("\n--- 打者成績 ---")
    df_hitters = fetch_all("hitter")
    if len(df_hitters) > 0:
        out_path = DATA_DIR / "npb_hitters_2015_2025.csv"
        df_hitters.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved: {out_path} ({len(df_hitters)} rows)")
        print(f"Columns: {list(df_hitters.columns)}")
        print(f"Years: {sorted(df_hitters['year'].unique())}")

    # 投手成績
    print("\n--- 投手成績 ---")
    df_pitchers = fetch_all("pitcher")
    if len(df_pitchers) > 0:
        out_path = DATA_DIR / "npb_pitchers_2015_2025.csv"
        df_pitchers.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved: {out_path} ({len(df_pitchers)} rows)")
        print(f"Columns: {list(df_pitchers.columns)}")
        print(f"Years: {sorted(df_pitchers['year'].unique())}")

    # サマリー
    print("\n" + "=" * 60)
    print("Summary:")
    if len(df_hitters) > 0:
        print(f"  Hitters: {len(df_hitters)} rows, {df_hitters['year'].nunique()} years")
    if len(df_pitchers) > 0:
        print(f"  Pitchers: {len(df_pitchers)} rows, {df_pitchers['year'].nunique()} years")


if __name__ == "__main__":
    main()
