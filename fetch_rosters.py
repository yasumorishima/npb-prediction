"""
年別NPB支配下登録選手一覧を取得するスクリプト
データソース: baseball-data.com/{yy}/player/{team-code}/

出力: data/raw/npb_rosters_YYYY_YYYY.csv
  columns: year, team, player

Marcel予測フィルタ用: target_yearの選手名鑑に存在しない選手は予測から除外する
"""

import io
import os
import time
import pandas as pd
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TEAM_CODES = {
    "阪神": "t",
    "DeNA": "yb",
    "巨人": "g",
    "中日": "d",
    "広島": "c",
    "ヤクルト": "s",
    "ソフトバンク": "h",
    "日本ハム": "f",
    "オリックス": "bs",
    "楽天": "e",
    "西武": "l",
    "ロッテ": "m",
}

START_YEAR = 2018
END_YEAR = int(os.environ.get("NPB_DATA_END_YEAR", 2025))


def fetch_team_roster(year: int, team: str, code: str) -> list[str]:
    """指定年・チームの選手名一覧を返す"""
    yy = str(year)[2:]
    url = f"https://baseball-data.com/{yy}/player/{code}/"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            raw = res.read()
        tables = pd.read_html(io.BytesIO(raw), flavor="lxml", encoding="utf-8")
        if not tables:
            print(f"  ⚠ テーブルなし: {url}")
            return []
        # 最初のテーブルに選手名カラムがある
        df = tables[0]
        # 「選手名」カラムを探す
        name_col = None
        for col in df.columns:
            if "選手" in str(col) or "名前" in str(col):
                name_col = col
                break
        if name_col is None:
            print(f"  ⚠ 選手名カラム不明: {df.columns.tolist()}")
            return []
        names = df[name_col].dropna().astype(str).tolist()
        # 空白・ヘッダ行などを除去
        names = [n.strip() for n in names if n.strip() and n != "選手名"]
        return names
    except Exception as e:
        print(f"  ❌ {url} → {e}")
        return []


def main():
    rows = []
    total = (END_YEAR - START_YEAR + 1) * len(TEAM_CODES)
    done = 0

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n=== {year}年 ===")
        for team, code in TEAM_CODES.items():
            names = fetch_team_roster(year, team, code)
            for name in names:
                rows.append({"year": year, "team": team, "player": name})
            print(f"  {team}: {len(names)}人")
            done += 1
            if done < total:
                time.sleep(1.0)  # サーバー負荷軽減

    df = pd.DataFrame(rows)
    out_path = DATA_DIR / f"npb_rosters_{START_YEAR}_{END_YEAR}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 保存完了: {out_path} ({len(df)}行)")


if __name__ == "__main__":
    main()
