"""
BigQuery データロードスクリプト（NPB版）

NPB選手成績・予測データを BigQuery にロードする。
生データはWRITE_TRUNCATE（フルリプレース）。
model_metrics_history のみ WRITE_APPEND（時系列蓄積）。

Usage:
    python load_to_bq.py                       # 全テーブルロード
    python load_to_bq.py --table raw_hitters   # 単テーブル
    python load_to_bq.py --metrics             # メトリクスを履歴に追記
    python load_to_bq.py --projections         # projections のみ
    python load_to_bq.py --all --metrics       # 全テーブル + メトリクス
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from config import DATA_END_YEAR, DATA_START_YEAR

PROJECT_ID = "data-platform-490901"
DATASET_ID = "npb"
FULL_DATASET = f"{PROJECT_ID}.{DATASET_ID}"

RAW_DIR = Path(__file__).parent / "data" / "raw"
PROJ_DIR = Path(__file__).parent / "data" / "projections"

# --- 生データ CSV → BQ テーブル名マッピング ---
# ファイル名に年範囲が入るものは動的に解決する
_YEAR_RANGE = f"{DATA_START_YEAR}_{DATA_END_YEAR}"

RAW_TABLE_MAP = {
    f"npb_hitters_{_YEAR_RANGE}.csv": "raw_hitters",
    f"npb_pitchers_{_YEAR_RANGE}.csv": "raw_pitchers",
    f"npb_batting_detailed_{_YEAR_RANGE}.csv": "raw_batting_detailed",
    f"npb_standings_{_YEAR_RANGE}.csv": "raw_standings",
    "npb_rosters_2018_{end}.csv": "raw_rosters",
    "npb_player_birthdays.csv": "raw_player_birthdays",
    "npb_players_profile_2024.csv": "raw_player_profiles",
}

# 年度ごとの試合データ（動的生成）
GAME_TABLE_MAP = {
    f"npb_games_{year}.csv": f"raw_games_{year}"
    for year in range(DATA_START_YEAR + 1, DATA_END_YEAR + 1)
}

# --- projections CSV → BQ テーブル名マッピング ---
PROJ_TABLE_MAP = {
    f"marcel_hitters_{DATA_END_YEAR + 1}.csv": "marcel_hitters",
    f"marcel_pitchers_{DATA_END_YEAR + 1}.csv": "marcel_pitchers",
    f"ml_hitters_{DATA_END_YEAR + 1}.csv": "ml_hitters",
    f"ml_pitchers_{DATA_END_YEAR + 1}.csv": "ml_pitchers",
    f"npb_sabermetrics_{_YEAR_RANGE}.csv": "sabermetrics",
    f"pythagorean_{_YEAR_RANGE}.csv": "pythagorean",
    "npb_park_factors.csv": "park_factors",
    "marcel_team_historical.csv": "marcel_team_historical",
}


def _resolve_raw_table_map() -> dict:
    """年範囲付きファイル名を実際に存在するファイルにマッチさせる"""
    resolved = {}
    for pattern, table_name in RAW_TABLE_MAP.items():
        if "{end}" in pattern:
            # rosters: npb_rosters_2018_YYYY.csv — 実ファイルを探す
            matches = sorted(RAW_DIR.glob("npb_rosters_2018_*.csv"))
            if matches:
                resolved[matches[-1].name] = table_name
        else:
            resolved[pattern] = table_name
    return resolved


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """BQ非互換カラム名を修正する。

    NPB固有: 日本語カラム名（差, 首位差, 残試合, 貯金, 連勝連敗）→ 英語に変換
    共通: % → _pct, + → _plus, / → _per_, 先頭数字 → _prefix
    """
    # NPB日本語カラム名マッピング
    ja_rename = {
        "No.": "jersey_number",
        "差": "games_back",
        "首位差": "games_behind_leader",
        "残試合": "remaining_games",
        "貯金": "wins_above_losses",
        "連勝連敗": "streak",
        "選手名": "player_name",
        "守備": "position",
        "生年月日": "birth_date",
        "年齢": "age",
        "年数": "years_pro",
        "身長": "height",
        "体重": "weight",
        "血液型": "blood_type",
        "投打": "throws_bats",
        "出身地": "hometown",
        "年俸(推定)": "salary_est",
    }

    rename = {}
    for col in df.columns:
        # 日本語マッピングを優先チェック
        if col in ja_rename:
            rename[col] = ja_rename[col]
            continue

        new = col
        new = new.replace("%", "_pct")
        new = new.replace("/", "_per_")
        new = new.replace("+", "_plus")
        new = new.replace("-", "_")
        new = new.replace(".", "_")
        new = new.replace("(", "")
        new = new.replace(")", "")
        new = new.replace(" ", "_")
        # 先頭が数字の場合
        if new and new[0].isdigit():
            new = f"_{new}"
        if new != col:
            rename[col] = new

    if rename:
        df = df.rename(columns=rename)
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """数値であるべきカラムがobject(string)の場合にfloatに変換する。

    CSVのBOM等でautodetectがSTRINGと判定する問題の対策。
    ERA, WHIP, DIPS, AVG, OBP, SLG, OPS, WPCT 等が該当。
    """
    numeric_candidates = [
        "ERA", "WHIP", "DIPS", "AVG", "OBP", "SLG", "OPS", "WPCT",
        "RC27", "XR27", "PF", "PF_5yr",
        "wOBA", "wRC_plus", "wRAA",
        "actual_WPCT", "pyth_WPCT_npb", "pyth_WPCT_mlb",
    ]
    for col in df.columns:
        if col in numeric_candidates and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _get_client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT_ID)


def load_csv_to_bq(
    csv_path: Path,
    table_name: str,
    write_disposition: str = "WRITE_TRUNCATE",
) -> int:
    """CSVファイルをBigQueryテーブルにロードする。"""
    if not csv_path.exists():
        print(f"  SKIP: {csv_path.name} not found")
        return 0

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"  SKIP: {csv_path.name} is empty")
        return 0

    df = _sanitize_columns(df)
    df = _coerce_numeric(df)

    client = _get_client()
    table_id = f"{FULL_DATASET}.{table_name}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        autodetect=True,
    )

    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()

    table = client.get_table(table_id)
    print(f"  OK: {table_name} <- {csv_path.name} ({table.num_rows} rows)")
    return table.num_rows


def load_all_raw() -> dict:
    """全生データCSVをBigQueryにロード（WRITE_TRUNCATE）"""
    print("=== Loading raw NPB data to BigQuery ===")
    results = {}

    resolved_map = _resolve_raw_table_map()
    for csv_name, table_name in resolved_map.items():
        csv_path = RAW_DIR / csv_name
        rows = load_csv_to_bq(csv_path, table_name, "WRITE_TRUNCATE")
        results[table_name] = rows

    # 年度別試合データ
    print("\n--- Game data by year ---")
    for csv_name, table_name in GAME_TABLE_MAP.items():
        csv_path = RAW_DIR / csv_name
        rows = load_csv_to_bq(csv_path, table_name, "WRITE_TRUNCATE")
        results[table_name] = rows

    return results


def load_projections() -> dict:
    """projections/*.csv をBigQueryに上書き（WRITE_TRUNCATE）"""
    print("=== Loading projections to BigQuery ===")
    results = {}
    for csv_name, table_name in PROJ_TABLE_MAP.items():
        csv_path = PROJ_DIR / csv_name
        rows = load_csv_to_bq(csv_path, table_name, "WRITE_TRUNCATE")
        results[table_name] = rows
    return results


def append_metrics() -> None:
    """メトリクスJSONをmodel_metrics_historyテーブルに追記。

    ml_projection.py が出力する wandb summary やMAE等を蓄積する。
    """
    # wandb のsummary JSONか、独自メトリクスファイルを探す
    metrics_candidates = [
        PROJ_DIR / "model_metrics.json",
        Path(__file__).parent / "data" / "metrics" / "model_metrics.json",
    ]

    metrics_path = None
    for p in metrics_candidates:
        if p.exists():
            metrics_path = p
            break

    if metrics_path is None:
        print("  SKIP: model_metrics.json not found")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    row = {
        "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "data_end_year": DATA_END_YEAR,
    }
    row.update(metrics)

    df = pd.DataFrame([row])

    client = _get_client()
    table_id = f"{FULL_DATASET}.model_metrics_history"

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        autodetect=True,
    )

    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()

    table = client.get_table(table_id)
    print(f"  OK: model_metrics_history appended (total {table.num_rows} rows)")


def print_summary(results: dict) -> None:
    """ロード結果のサマリー"""
    loaded = {k: v for k, v in results.items() if v > 0}
    skipped = {k: v for k, v in results.items() if v == 0}
    total_rows = sum(loaded.values())

    print(f"\n=== Summary ===")
    print(f"  Loaded: {len(loaded)} tables, {total_rows:,} total rows")
    if skipped:
        print(f"  Skipped: {len(skipped)} tables ({', '.join(skipped.keys())})")


def main():
    parser = argparse.ArgumentParser(description="Load NPB data to BigQuery")
    parser.add_argument("--table", type=str, help="Load a single table by BQ name")
    parser.add_argument("--all", action="store_true", help="Load all raw + projections")
    parser.add_argument("--raw", action="store_true", help="Load raw data only")
    parser.add_argument("--projections", action="store_true", help="Load projections only")
    parser.add_argument("--metrics", action="store_true", help="Append model metrics")
    args = parser.parse_args()

    if not any([args.table, args.all, args.raw, args.projections, args.metrics]):
        args.all = True

    results = {}

    if args.table:
        all_maps = {
            **_resolve_raw_table_map(),
            **GAME_TABLE_MAP,
            **PROJ_TABLE_MAP,
        }
        csv_name = None
        for k, v in all_maps.items():
            if v == args.table:
                csv_name = k
                break
        if csv_name is None:
            print(f"ERROR: Unknown table '{args.table}'")
            all_tables = sorted(
                set(list(_resolve_raw_table_map().values())
                    + list(GAME_TABLE_MAP.values())
                    + list(PROJ_TABLE_MAP.values()))
            )
            print(f"Available: {', '.join(all_tables)}")
            sys.exit(1)
        # raw / games / projections のどれか判定
        if csv_name in _resolve_raw_table_map():
            csv_path = RAW_DIR / csv_name
        elif csv_name in GAME_TABLE_MAP:
            csv_path = RAW_DIR / csv_name
        else:
            csv_path = PROJ_DIR / csv_name
        rows = load_csv_to_bq(csv_path, args.table, "WRITE_TRUNCATE")
        results[args.table] = rows

    if args.all or args.raw:
        results.update(load_all_raw())

    if args.all or args.projections:
        results.update(load_projections())

    if args.all or args.metrics:
        append_metrics()

    if results:
        print_summary(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
