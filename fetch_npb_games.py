"""
NPB 試合別スコアデータ取得スクリプト
パークファクター計算のためのホーム/アウェイ別得失点データを取得する

試行するデータソース（優先順）:
1. baseball-data.com  /{yy}/score/{team_code}/
2. npb.jp             /games/{year}/schedule_{month:02d}.html

出力: data/raw/npb_games_{year}.csv
  columns: year, date, home_team, away_team, home_score, away_score

使い方:
    python fetch_npb_games.py              # config.py の YEARS 全年度
    NPB_DATA_END_YEAR=2024 python fetch_npb_games.py
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import DATA_END_YEAR, YEARS

DATA_DIR = Path(__file__).parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}

# baseball-data.com チームコード（スコアページ用）
BD_TEAM_CODES = {
    "阪神": "t", "DeNA": "yb", "巨人": "g",
    "中日": "d", "広島": "c", "ヤクルト": "s",
    "ソフトバンク": "h", "日本ハム": "f", "オリックス": "bs",
    "楽天": "e", "西武": "l", "ロッテ": "m",
}

# npb.jp チームコード（スケジュールページ用）
NPB_TEAM_NAMES = list(BD_TEAM_CODES.keys())

# 球団ホーム球場マスタ（年度範囲で管理）
STADIUM_MAP = {
    "阪神": "甲子園",
    "DeNA": "横浜スタジアム",
    "巨人": "東京ドーム",
    "中日": "バンテリンドーム",
    "広島": "マツダスタジアム",
    "ヤクルト": "神宮球場",
    "ソフトバンク": "PayPayドーム",
    "日本ハム": "エスコンフィールド",   # 2023〜
    "オリックス": "京セラドーム",
    "楽天": "楽天モバイルパーク",
    "西武": "ベルーナドーム",
    "ロッテ": "ZOZOマリン",
}

# 日ハムは2022まで札幌ドーム
STADIUM_OVERRIDE = {
    ("日本ハム", 2015): "札幌ドーム",
    ("日本ハム", 2016): "札幌ドーム",
    ("日本ハム", 2017): "札幌ドーム",
    ("日本ハム", 2018): "札幌ドーム",
    ("日本ハム", 2019): "札幌ドーム",
    ("日本ハム", 2020): "札幌ドーム",
    ("日本ハム", 2021): "札幌ドーム",
    ("日本ハム", 2022): "札幌ドーム",
}


def get_home_stadium(team: str, year: int) -> str:
    return STADIUM_OVERRIDE.get((team, year), STADIUM_MAP.get(team, "不明"))


# ==========================================================================
# Source 1: baseball-data.com
# URL: https://baseball-data.com/{yy}/score/{team_code}/
# ==========================================================================

def fetch_games_baseball_data(year: int) -> pd.DataFrame:
    """baseball-data.com のスコアページから試合結果を取得"""
    yy = str(year)[2:]
    records = []

    for team_name, team_code in BD_TEAM_CODES.items():
        url = f"https://baseball-data.com/{yy}/score/{team_code}/"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"  [{team_name}] HTTP {resp.status_code}: {url}")
                continue

            tables = pd.read_html(resp.text, flavor="lxml")
            if not tables:
                print(f"  [{team_name}] テーブルなし: {url}")
                continue

            # 最もカラム数が多いテーブルを対象にする
            df = max(tables, key=lambda x: len(x.columns))
            cols_lower = [str(c).lower() for c in df.columns]
            print(f"  [{team_name}] カラム: {list(df.columns)[:8]} ({len(df)}行)")

            # スコア列を検出（数字パターン: "3-1", "○3-1" etc）
            score_col = None
            for col in df.columns:
                sample = df[col].dropna().astype(str)
                if sample.str.match(r"[○●△☓]?\d+-\d+").sum() > len(sample) * 0.3:
                    score_col = col
                    break

            if score_col is None:
                print(f"  [{team_name}] スコア列が見つからない")
                continue

            for _, row in df.iterrows():
                score_str = str(row[score_col])
                m = re.search(r"(\d+)-(\d+)", score_str)
                if not m:
                    continue

                # 勝敗記号から得点・失点を判定
                score_a, score_b = int(m.group(1)), int(m.group(2))
                result_mark = score_str[0] if score_str[0] in "○●△☓" else ""

                # 対戦相手を検出
                opponent = None
                for col in df.columns:
                    val = str(row[col])
                    for t in BD_TEAM_CODES:
                        if t in val and t != team_name:
                            opponent = t
                            break
                    if opponent:
                        break

                # ホーム/アウェイ判定（"H"列や球場名列から）
                is_home = None
                for col in df.columns:
                    val = str(row[col])
                    if val in ("主", "H", "ホーム"):
                        is_home = True
                        break
                    if val in ("客", "A", "アウェイ", "V", "ビジター"):
                        is_home = False
                        break

                # 日付を取得
                date = None
                for col in df.columns:
                    val = str(row[col])
                    if re.match(r"\d{1,2}/\d{1,2}", val) or re.match(r"\d{4}/\d{1,2}/\d{1,2}", val):
                        date = val
                        break

                if opponent and is_home is not None and date:
                    if is_home:
                        records.append({
                            "year": year, "date": date,
                            "home_team": team_name, "away_team": opponent,
                            "home_score": score_a, "away_score": score_b,
                            "stadium": get_home_stadium(team_name, year),
                        })

            time.sleep(1.5)

        except Exception as e:
            print(f"  [{team_name}] エラー: {e}")

    return pd.DataFrame(records)


# ==========================================================================
# Source 2: npb.jp スケジュールページ
# URL: https://npb.jp/games/{year}/schedule_{month:02d}.html
# ==========================================================================

NPB_TEAM_RE = "|".join(NPB_TEAM_NAMES)

def fetch_games_npb_jp(year: int) -> pd.DataFrame:
    """npb.jp のスケジュールページから試合結果を取得"""
    records = []

    for month in range(3, 12):   # 3月〜11月
        url = f"https://npb.jp/games/{year}/schedule_{month:02d}.html"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 404:
                continue
            if resp.status_code != 200:
                print(f"  [npb.jp month={month}] HTTP {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.text, "lxml")

            # 試合結果テーブルを探す
            tables = soup.find_all("table")
            for table in tables:
                rows = table.find_all("tr")
                for row in rows:
                    cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                    if len(cells) < 4:
                        continue

                    # スコアパターン: "3 - 1" or "3-1" を検出
                    score_m = None
                    score_idx = None
                    for i, cell in enumerate(cells):
                        m = re.search(r"(\d+)\s*[-−]\s*(\d+)", cell)
                        if m:
                            score_m = m
                            score_idx = i
                            break

                    if score_m is None:
                        continue

                    home_score = int(score_m.group(1))
                    away_score = int(score_m.group(2))

                    # チーム名を前後のセルから推定
                    home_team = away_team = None
                    for cell in cells:
                        for team in NPB_TEAM_NAMES:
                            if team in cell:
                                if home_team is None:
                                    home_team = team
                                elif away_team is None and team != home_team:
                                    away_team = team

                    # 日付
                    date = None
                    for cell in cells:
                        m = re.match(r"(\d{1,2})月(\d{1,2})日", cell)
                        if m:
                            date = f"{year}/{int(m.group(1)):02d}/{int(m.group(2)):02d}"
                            break

                    if home_team and away_team and date:
                        records.append({
                            "year": year, "date": date,
                            "home_team": home_team, "away_team": away_team,
                            "home_score": home_score, "away_score": away_score,
                            "stadium": get_home_stadium(home_team, year),
                        })

            time.sleep(1.0)

        except Exception as e:
            print(f"  [npb.jp month={month}] エラー: {e}")

    return pd.DataFrame(records)


# ==========================================================================
# メイン: 両ソースを試行して結果を保存
# ==========================================================================

def fetch_year(year: int) -> bool:
    """1年分の試合データを取得。成功したら True を返す"""
    out_path = DATA_DIR / f"npb_games_{year}.csv"

    print(f"\n=== {year}年 試合データ取得 ===")

    # Source 1: baseball-data.com
    print("[Source 1] baseball-data.com ...")
    df = fetch_games_baseball_data(year)
    if len(df) >= 100:   # 最低100試合取れれば成功とみなす
        # 重複除去（同一試合を双方チームで取得する可能性）
        df = df.drop_duplicates(subset=["date", "home_team", "away_team"]).reset_index(drop=True)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  -> {len(df)} 試合取得 → {out_path.name}")
        return True

    print(f"  -> {len(df)} 試合のみ（不足）。Source 2 を試行...")

    # Source 2: npb.jp
    print("[Source 2] npb.jp ...")
    df2 = fetch_games_npb_jp(year)
    if len(df2) >= 100:
        df2 = df2.drop_duplicates(subset=["date", "home_team", "away_team"]).reset_index(drop=True)
        df2.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  -> {len(df2)} 試合取得 → {out_path.name}")
        return True

    # 両ソース失敗
    combined = pd.concat([df, df2], ignore_index=True).drop_duplicates(
        subset=["date", "home_team", "away_team"]
    )
    print(f"  -> 両ソースとも不足 ({len(combined)} 試合)。URL構造を確認してください。")
    if len(combined) > 0:
        combined.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  -> 部分データを保存: {out_path.name}")
    return False


def main():
    success_years = []
    fail_years = []

    for year in YEARS:
        ok = fetch_year(year)
        (success_years if ok else fail_years).append(year)

    print(f"\n=== 完了 ===")
    print(f"成功: {success_years}")
    print(f"失敗/不足: {fail_years}")
    if fail_years:
        print("※ 失敗した年度はURL構造の変更またはデータなしの可能性があります")
        print("  baseball-data.com のスコアページURLを手動確認してください")


if __name__ == "__main__":
    main()
