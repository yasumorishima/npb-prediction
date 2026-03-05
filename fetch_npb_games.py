"""
NPB 試合別スコアデータ取得スクリプト
パークファクター計算のためのホーム/アウェイ別得失点データを取得する

試行するデータソース（優先順）:
1. baseball-data.com  /{yy}/score/{team_code}/
2. npb.jp             /games/{year}/schedule_{month:02d}_detail.html

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
# Source 2: npb.jp スケジュール詳細ページ
# URL: https://npb.jp/games/{year}/schedule_{month:02d}_detail.html
# 列構造: 日付 | チームA | スコア | チームB | 球場・時刻 | 投手成績
# ホーム/アウェイ判定: 球場名を STADIUM_MAP と照合
# ==========================================================================

def fetch_games_npb_jp(year: int) -> pd.DataFrame:
    """npb.jp のスケジュール詳細ページから試合結果を取得"""
    records = []

    for month in range(3, 12):   # 3月〜11月
        url = f"https://npb.jp/games/{year}/schedule_{month:02d}_detail.html"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 404:
                continue
            if resp.status_code != 200:
                print(f"  [npb.jp month={month}] HTTP {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.text, "lxml")

            for table in soup.find_all("table"):
                for row in table.find_all("tr"):
                    cells = row.find_all(["td", "th"])
                    if len(cells) < 5:
                        continue
                    texts = [c.get_text(strip=True) for c in cells]

                    # 日付: texts[0] に "月/日" パターン（例: "4/2（火）"）
                    date_m = re.match(r"(\d{1,2})/(\d{1,2})", texts[0])
                    if not date_m:
                        continue
                    date = f"{year}/{int(date_m.group(1)):02d}/{int(date_m.group(2)):02d}"

                    # チーム名: texts[1] と texts[3]
                    team1 = next((t for t in NPB_TEAM_NAMES if t in texts[1]), None)
                    team2 = next((t for t in NPB_TEAM_NAMES if t in texts[3]), None)
                    if not team1 or not team2:
                        continue

                    # スコア: texts[2]（例: "4-3"）
                    score_m = re.search(r"(\d+)[-−](\d+)", texts[2])
                    if not score_m:
                        continue
                    score_a, score_b = int(score_m.group(1)), int(score_m.group(2))

                    # 球場名からホームチームを判定（STADIUM_OVERRIDE優先）
                    stadium_text = texts[4] if len(texts) > 4 else ""
                    home_team = team1  # デフォルト: texts[1] をホームとみなす
                    for team in [team1, team2]:
                        expected = STADIUM_OVERRIDE.get((team, year), STADIUM_MAP.get(team, ""))
                        if expected and expected in stadium_text:
                            home_team = team
                            break

                    if home_team == team1:
                        home_score, away_score, away_team = score_a, score_b, team2
                    else:
                        home_score, away_score, away_team = score_b, score_a, team1

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
