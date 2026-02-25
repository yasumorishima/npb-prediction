"""
NPB成績予測 FastAPI

Marcel法・LightGBM/XGBoost・ピタゴラス勝率・wOBA/wRC+の予測結果をAPIで提供。

Data sources:
- プロ野球データFreak (https://baseball-data.com)
- 日本野球機構 NPB (https://npb.jp)
"""

from enum import Enum
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Path as PathParam, Query

app = FastAPI(
    title="NPB 成績予測 API",
    description=(
        "NPB（日本プロ野球）の選手成績予測・チーム勝率予測を提供するAPIです。\n\n"
        "## 予測手法\n"
        "- **Marcel法**: 過去3年の加重平均 + 平均回帰 + 年齢調整（最も精度が高い）\n"
        "- **機械学習**: LightGBM / XGBoost によるアンサンブル予測\n"
        "- **ピタゴラス勝率**: 得失点からチーム勝率を推定（NPB最適指数 k=1.72）\n"
        "- **セイバーメトリクス**: wOBA / wRC+ / wRAA（NPBリーグ環境にスケーリング）\n\n"
        "## データソース\n"
        "- [プロ野球データFreak](https://baseball-data.com)\n"
        "- [日本野球機構 NPB](https://npb.jp)\n"
    ),
    version="0.2.0",
)


class TeamName(str, Enum):
    """NPB 12球団"""
    DeNA = "DeNA"
    巨人 = "巨人"
    阪神 = "阪神"
    広島 = "広島"
    中日 = "中日"
    ヤクルト = "ヤクルト"
    ソフトバンク = "ソフトバンク"
    日本ハム = "日本ハム"
    楽天 = "楽天"
    ロッテ = "ロッテ"
    オリックス = "オリックス"
    西武 = "西武"

PROJ_DIR = Path(__file__).parent / "data" / "projections"


def _norm(name: str) -> str:
    """全角スペース→半角、前後空白除去"""
    return name.replace("\u3000", " ").strip()


def _load_csv(filename: str) -> pd.DataFrame:
    path = PROJ_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "player" in df.columns:
        df["player"] = df["player"].apply(_norm)
    if "team" in df.columns:
        df["team"] = df["team"].apply(_norm)
    return df


# --- 起動時にCSVをメモリに読み込み ---
marcel_hitters = _load_csv("marcel_hitters_2026.csv")
marcel_pitchers = _load_csv("marcel_pitchers_2026.csv")
ml_hitters = _load_csv("ml_hitters_2026.csv")
ml_pitchers = _load_csv("ml_pitchers_2026.csv")
sabermetrics = _load_csv("npb_sabermetrics_2015_2025.csv")
pythagorean = _load_csv("pythagorean_2015_2025.csv")


def _search_player(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """部分一致で選手を検索"""
    q = _norm(name)
    mask = df["player"].str.contains(q, na=False)
    return df[mask]


# ============================================================
# エンドポイント
# ============================================================


@app.get("/")
def root():
    return {
        "name": "NPB Prediction API",
        "version": "0.1.0",
        "endpoints": [
            "/predict/hitter/{name}",
            "/predict/pitcher/{name}",
            "/predict/team/{name}",
            "/sabermetrics/{name}",
            "/rankings/hitters",
            "/rankings/pitchers",
            "/pythagorean",
        ],
    }


@app.get(
    "/predict/hitter/{name}",
    summary="打者の2026年成績予測",
    description="選手名（部分一致可）を指定して、Marcel法とMLによる2026年シーズン予測を取得します。",
)
def predict_hitter(
    name: str = PathParam(description="選手名（部分一致OK）", examples=["牧", "近藤", "岡本"]),
):
    """打者の2026年成績予測（Marcel法 + ML）"""
    marcel = _search_player(marcel_hitters, name)
    ml = _search_player(ml_hitters, name)

    if marcel.empty and ml.empty:
        raise HTTPException(404, f"選手が見つかりません: {name}")

    results = []
    for _, row in marcel.iterrows():
        entry = {
            "選手名": row["player"],
            "チーム": row["team"],
            "Marcel予測": {
                "OPS": round(row["OPS"], 3),
                "打率": round(row["AVG"], 3),
                "出塁率": round(row["OBP"], 3),
                "長打率": round(row["SLG"], 3),
                "本塁打": round(row["HR"], 1),
                "打点": round(row["RBI"], 1),
            },
        }
        ml_match = ml[ml["player"] == row["player"]]
        if not ml_match.empty:
            entry["ML予測"] = {"OPS": round(ml_match.iloc[0]["pred_OPS"], 3)}
        results.append(entry)

    return {"検索": name, "件数": len(results), "予測": results}


@app.get(
    "/predict/pitcher/{name}",
    summary="投手の2026年成績予測",
    description="選手名（部分一致可）を指定して、Marcel法とMLによる2026年シーズン予測を取得します。",
)
def predict_pitcher(
    name: str = PathParam(description="選手名（部分一致OK）", examples=["今永", "山本", "佐々木"]),
):
    """投手の2026年成績予測（Marcel法 + ML）"""
    marcel = _search_player(marcel_pitchers, name)
    ml = _search_player(ml_pitchers, name)

    if marcel.empty and ml.empty:
        raise HTTPException(404, f"選手が見つかりません: {name}")

    results = []
    for _, row in marcel.iterrows():
        entry = {
            "選手名": row["player"],
            "チーム": row["team"],
            "Marcel予測": {
                "防御率": round(row["ERA"], 2),
                "WHIP": round(row["WHIP"], 2),
                "勝利": round(row["W"], 1),
                "敗北": round(row["L"], 1),
                "奪三振": round(row["SO"], 1),
                "投球回": round(row["IP"], 1),
            },
        }
        ml_match = ml[ml["player"] == row["player"]]
        if not ml_match.empty:
            entry["ML予測"] = {"防御率": round(ml_match.iloc[0]["pred_ERA"], 2)}
        results.append(entry)

    return {"検索": name, "件数": len(results), "予測": results}


@app.get(
    "/predict/team/{name}",
    summary="チームのピタゴラス勝率",
    description="得点・失点から理論上の勝率を推定します（NPB最適指数 k=1.72）。実際の勝数との差も表示。",
)
def predict_team(
    name: TeamName = PathParam(description="チーム名"),
    year: int = Query(default=2025, ge=2015, le=2025, description="対象年度（2015〜2025）"),
):
    """チームのピタゴラス勝率予測"""
    if pythagorean.empty:
        raise HTTPException(503, "ピタゴラス勝率データが読み込まれていません")

    q = _norm(name.value)
    mask = pythagorean["team"].str.contains(q, na=False) & (pythagorean["year"] == year)
    matched = pythagorean[mask]

    if matched.empty:
        raise HTTPException(404, f"チームが見つかりません: {name.value} ({year})")

    results = []
    for _, row in matched.iterrows():
        results.append({
            "チーム": row["team"],
            "年度": int(row["year"]),
            "リーグ": row["league"],
            "実際の勝数": int(row["W"]),
            "実際の敗数": int(row["L"]),
            "実際の勝率": round(row["actual_WPCT"], 3),
            "ピタゴラス勝率": round(row["pyth_WPCT_npb"], 3),
            "ピタゴラス期待勝数": round(row["pyth_W_npb"], 1),
            "差（実際-期待）": round(row["diff_W_npb"], 1),
            "得点": int(row["RS"]),
            "失点": int(row["RA"]),
        })

    return {"検索": name.value, "年度": year, "件数": len(results), "チーム": results}


@app.get(
    "/sabermetrics/{name}",
    summary="セイバーメトリクス（wOBA / wRC+ / wRAA）",
    description="NPBリーグ環境にスケーリングしたwOBA・wRC+・wRAAを取得します。wRC+はリーグ平均=100。",
)
def get_sabermetrics(
    name: str = PathParam(description="選手名（部分一致OK）", examples=["近藤", "牧", "オースティン"]),
    year: int | None = Query(default=None, ge=2015, le=2025, description="対象年度（省略で全年度）"),
):
    """選手のwOBA/wRC+/wRAA"""
    if sabermetrics.empty:
        raise HTTPException(503, "セイバーメトリクスデータが読み込まれていません")

    matched = _search_player(sabermetrics, name)
    if year is not None:
        matched = matched[matched["year"] == year]

    if matched.empty:
        raise HTTPException(404, f"選手が見つかりません: {name}")

    results = []
    for _, row in matched.iterrows():
        results.append({
            "選手名": row["player"],
            "チーム": row["team"],
            "年度": int(row["year"]),
            "打席数": int(row["PA"]),
            "wOBA": round(row["wOBA"], 3),
            "wRC+": round(row["wRC+"], 1),
            "wRAA": round(row["wRAA"], 1),
            "打率": round(row["AVG"], 3),
            "出塁率": round(row["OBP"], 3),
            "長打率": round(row["SLG"], 3),
        })

    return {"検索": name, "件数": len(results), "成績": results}


@app.get(
    "/rankings/hitters",
    summary="打者ランキング（Marcel法 2026予測）",
    description="Marcel法による2026年打者予測のランキング。ソート項目（OPS/打率/本塁打/打点）と表示人数を指定可能。",
)
def rankings_hitters(
    top: int = Query(default=20, ge=1, le=100, description="表示人数（1〜100）", examples=[10, 20, 50]),
    sort_by: str = Query(default="OPS", enum=["OPS", "AVG", "HR", "RBI"], description="ソート項目"),
):
    """打者ランキング（Marcel法 2026予測）"""
    if marcel_hitters.empty:
        raise HTTPException(503, "Marcel打者データが読み込まれていません")

    df = marcel_hitters.sort_values(sort_by, ascending=False).head(top)
    results = []
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        results.append({
            "順位": rank,
            "選手名": row["player"],
            "チーム": row["team"],
            "OPS": round(row["OPS"], 3),
            "打率": round(row["AVG"], 3),
            "本塁打": round(row["HR"], 1),
            "打点": round(row["RBI"], 1),
            "打席数": round(row["PA"], 0),
        })

    return {"ソート": sort_by, "件数": len(results), "ランキング": results}


@app.get(
    "/rankings/pitchers",
    summary="投手ランキング（Marcel法 2026予測）",
    description="Marcel法による2026年投手予測のランキング。規定投球回（50IP以上）の投手が対象。ソート項目（防御率/WHIP/奪三振/勝利）を指定可能。",
)
def rankings_pitchers(
    top: int = Query(default=20, ge=1, le=100, description="表示人数（1〜100）", examples=[10, 20, 50]),
    sort_by: str = Query(default="ERA", enum=["ERA", "WHIP", "SO", "W"], description="ソート項目（ERA/WHIPは昇順）"),
):
    """投手ランキング（Marcel法 2026予測）"""
    if marcel_pitchers.empty:
        raise HTTPException(503, "Marcel投手データが読み込まれていません")

    ascending = sort_by in ("ERA", "WHIP")
    # 規定投球回以上（50IP+）でフィルタ
    df = marcel_pitchers[marcel_pitchers["IP"] >= 50]
    df = df.sort_values(sort_by, ascending=ascending).head(top)

    results = []
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        results.append({
            "順位": rank,
            "選手名": row["player"],
            "チーム": row["team"],
            "防御率": round(row["ERA"], 2),
            "WHIP": round(row["WHIP"], 2),
            "奪三振": round(row["SO"], 1),
            "勝利": round(row["W"], 1),
            "投球回": round(row["IP"], 1),
        })

    return {"ソート": sort_by, "件数": len(results), "ランキング": results}


@app.get(
    "/pythagorean",
    summary="全チームのピタゴラス勝率（指定年）",
    description="得失点から推定した理論上の勝率で全12球団を順位付け。NPB最適指数 k=1.72 を使用。実際の勝数との差（＝運の要素）も表示。",
)
def pythagorean_all(
    year: int = Query(default=2025, ge=2015, le=2025, description="対象年度（2015〜2025）", examples=[2025, 2024, 2023]),
):
    """全チームのピタゴラス勝率（指定年）"""
    if pythagorean.empty:
        raise HTTPException(503, "ピタゴラス勝率データが読み込まれていません")

    df = pythagorean[pythagorean["year"] == year].sort_values("pyth_WPCT_npb", ascending=False)

    if df.empty:
        raise HTTPException(404, f"{year}年のデータがありません")

    results = []
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        results.append({
            "順位": rank,
            "チーム": row["team"],
            "リーグ": row["league"],
            "実際の勝数": int(row["W"]),
            "実際の敗数": int(row["L"]),
            "ピタゴラス勝率": round(row["pyth_WPCT_npb"], 3),
            "ピタゴラス期待勝数": round(row["pyth_W_npb"], 1),
            "差（実際-期待）": round(row["diff_W_npb"], 1),
        })

    return {"年度": year, "件数": len(results), "順位表": results}
