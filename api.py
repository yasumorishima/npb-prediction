"""
NPB成績予測 FastAPI

Marcel法・LightGBM/XGBoost・ピタゴラス勝率・wOBA/wRC+の予測結果をAPIで提供。

Data sources:
- プロ野球データFreak (https://baseball-data.com)
- 日本野球機構 NPB (https://npb.jp)
"""

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(
    title="NPB Prediction API",
    description="NPB選手成績予測・チーム勝率予測API（Marcel法 / ML / ピタゴラス勝率 / wOBA・wRC+）",
    version="0.1.0",
)

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


@app.get("/predict/hitter/{name}")
def predict_hitter(name: str):
    """打者の2026年成績予測（Marcel法 + ML）"""
    marcel = _search_player(marcel_hitters, name)
    ml = _search_player(ml_hitters, name)

    if marcel.empty and ml.empty:
        raise HTTPException(404, f"選手が見つかりません: {name}")

    results = []
    for _, row in marcel.iterrows():
        entry = {
            "player": row["player"],
            "team": row["team"],
            "marcel": {
                "OPS": round(row["OPS"], 3),
                "AVG": round(row["AVG"], 3),
                "OBP": round(row["OBP"], 3),
                "SLG": round(row["SLG"], 3),
                "HR": round(row["HR"], 1),
                "RBI": round(row["RBI"], 1),
            },
        }
        # ML予測をマージ
        ml_match = ml[ml["player"] == row["player"]]
        if not ml_match.empty:
            entry["ml"] = {"pred_OPS": round(ml_match.iloc[0]["pred_OPS"], 3)}
        results.append(entry)

    return {"query": name, "count": len(results), "predictions": results}


@app.get("/predict/pitcher/{name}")
def predict_pitcher(name: str):
    """投手の2026年成績予測（Marcel法 + ML）"""
    marcel = _search_player(marcel_pitchers, name)
    ml = _search_player(ml_pitchers, name)

    if marcel.empty and ml.empty:
        raise HTTPException(404, f"選手が見つかりません: {name}")

    results = []
    for _, row in marcel.iterrows():
        entry = {
            "player": row["player"],
            "team": row["team"],
            "marcel": {
                "ERA": round(row["ERA"], 2),
                "WHIP": round(row["WHIP"], 2),
                "W": round(row["W"], 1),
                "L": round(row["L"], 1),
                "SO": round(row["SO"], 1),
                "IP": round(row["IP"], 1),
            },
        }
        ml_match = ml[ml["player"] == row["player"]]
        if not ml_match.empty:
            entry["ml"] = {"pred_ERA": round(ml_match.iloc[0]["pred_ERA"], 2)}
        results.append(entry)

    return {"query": name, "count": len(results), "predictions": results}


@app.get("/predict/team/{name}")
def predict_team(name: str, year: int = Query(default=2025, ge=2015, le=2025)):
    """チームのピタゴラス勝率予測"""
    if pythagorean.empty:
        raise HTTPException(503, "ピタゴラス勝率データが読み込まれていません")

    q = _norm(name)
    mask = pythagorean["team"].str.contains(q, na=False) & (pythagorean["year"] == year)
    matched = pythagorean[mask]

    if matched.empty:
        raise HTTPException(404, f"チームが見つかりません: {name} ({year})")

    results = []
    for _, row in matched.iterrows():
        results.append({
            "team": row["team"],
            "year": int(row["year"]),
            "league": row["league"],
            "actual_W": int(row["W"]),
            "actual_L": int(row["L"]),
            "actual_WPCT": round(row["actual_WPCT"], 3),
            "pyth_WPCT": round(row["pyth_WPCT_npb"], 3),
            "pyth_W": round(row["pyth_W_npb"], 1),
            "diff_W": round(row["diff_W_npb"], 1),
            "RS": int(row["RS"]),
            "RA": int(row["RA"]),
        })

    return {"query": name, "year": year, "count": len(results), "teams": results}


@app.get("/sabermetrics/{name}")
def get_sabermetrics(
    name: str,
    year: int | None = Query(default=None, ge=2015, le=2025),
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
            "player": row["player"],
            "team": row["team"],
            "year": int(row["year"]),
            "PA": int(row["PA"]),
            "wOBA": round(row["wOBA"], 3),
            "wRC_plus": round(row["wRC+"], 1),
            "wRAA": round(row["wRAA"], 1),
            "AVG": round(row["AVG"], 3),
            "OBP": round(row["OBP"], 3),
            "SLG": round(row["SLG"], 3),
        })

    return {"query": name, "count": len(results), "stats": results}


@app.get("/rankings/hitters")
def rankings_hitters(
    top: int = Query(default=20, ge=1, le=100),
    sort_by: str = Query(default="OPS", enum=["OPS", "AVG", "HR", "RBI"]),
):
    """打者ランキング（Marcel法 2026予測）"""
    if marcel_hitters.empty:
        raise HTTPException(503, "Marcel打者データが読み込まれていません")

    df = marcel_hitters.sort_values(sort_by, ascending=False).head(top)
    results = []
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        results.append({
            "rank": rank,
            "player": row["player"],
            "team": row["team"],
            "OPS": round(row["OPS"], 3),
            "AVG": round(row["AVG"], 3),
            "HR": round(row["HR"], 1),
            "RBI": round(row["RBI"], 1),
            "PA": round(row["PA"], 0),
        })

    return {"sort_by": sort_by, "count": len(results), "rankings": results}


@app.get("/rankings/pitchers")
def rankings_pitchers(
    top: int = Query(default=20, ge=1, le=100),
    sort_by: str = Query(default="ERA", enum=["ERA", "WHIP", "SO", "W"]),
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
            "rank": rank,
            "player": row["player"],
            "team": row["team"],
            "ERA": round(row["ERA"], 2),
            "WHIP": round(row["WHIP"], 2),
            "SO": round(row["SO"], 1),
            "W": round(row["W"], 1),
            "IP": round(row["IP"], 1),
        })

    return {"sort_by": sort_by, "count": len(results), "rankings": results}


@app.get("/pythagorean")
def pythagorean_all(year: int = Query(default=2025, ge=2015, le=2025)):
    """全チームのピタゴラス勝率（指定年）"""
    if pythagorean.empty:
        raise HTTPException(503, "ピタゴラス勝率データが読み込まれていません")

    df = pythagorean[pythagorean["year"] == year].sort_values("pyth_WPCT_npb", ascending=False)

    if df.empty:
        raise HTTPException(404, f"{year}年のデータがありません")

    results = []
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        results.append({
            "rank": rank,
            "team": row["team"],
            "league": row["league"],
            "actual_W": int(row["W"]),
            "actual_L": int(row["L"]),
            "pyth_WPCT": round(row["pyth_WPCT_npb"], 3),
            "pyth_W": round(row["pyth_W_npb"], 1),
            "diff_W": round(row["diff_W_npb"], 1),
        })

    return {"year": year, "count": len(results), "standings": results}
