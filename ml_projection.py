"""
XGBoost/LightGBMによるNPB選手成績予測

Marcel法を上回ることが目標。
特徴量: 過去3年の成績（加重なし）+ 打席数トレンド + 年数ギャップ + wOBA/wRC+
ターゲット: 翌年のOPS（打者）/ ERA（投手）

Data sources:
- プロ野球データFreak (https://baseball-data.com)
- 日本野球機構 NPB (https://npb.jp) — wOBA/wRC+算出用の詳細打撃成績
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from marcel_projection import load_birthdays, calc_age

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("WARNING: lightgbm not installed. Using XGBoost only.")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed. Using LightGBM only.")

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "projections"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _norm_name(name: str) -> str:
    """選手名を正規化（全角スペース→半角スペース）"""
    return str(name).replace("\u3000", " ").strip()


def load_sabermetrics() -> pd.DataFrame:
    """wOBA/wRC+データをロード（npb.jpベース）"""
    path = OUT_DIR / "npb_sabermetrics_2015_2025.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["player_norm"] = df["player"].apply(_norm_name)
    return df


def load_hitters() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "npb_hitters_2015_2025.csv")
    for col in ["RC27", "XR27"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_pitchers() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "npb_pitchers_2015_2025.csv")
    for col in ["ERA", "WHIP", "DIPS", "IP"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # 投球回を数値化
    df["IP_num"] = df["IP"].apply(_parse_ip)
    return df


def _parse_ip(ip_val) -> float:
    try:
        s = str(ip_val)
        if "." in s:
            parts = s.split(".")
            return int(parts[0]) + int(parts[1]) / 3.0
        return float(s)
    except (ValueError, IndexError):
        return 0.0


# ==============================
# 特徴量エンジニアリング（打者）
# ==============================
def build_hitter_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    各選手・各年について、過去3年分の成績から特徴量を生成し、
    翌年のOPSをターゲットとするデータセットを作成。
    wOBA/wRC+も特徴量に追加（npb.jpのセイバーメトリクスデータと結合）。
    """
    birthdays = load_birthdays()
    saber = load_sabermetrics()
    has_saber = len(saber) > 0
    years = sorted(df["year"].unique())
    rows = []

    for target_year in years:
        # 過去3年
        y1, y2, y3 = target_year - 1, target_year - 2, target_year - 3

        # ターゲット年のデータ
        target_df = df[df["year"] == target_year]

        for _, tgt_row in target_df.iterrows():
            player = tgt_row["player"]
            if tgt_row["PA"] < 100:  # 最低打席数
                continue

            feat = {"player": player, "team": tgt_row["team"],
                    "target_year": target_year, "target_OPS": tgt_row["OPS"],
                    "target_AVG": tgt_row["AVG"], "target_SLG": tgt_row["SLG"],
                    "target_OBP": tgt_row["OBP"]}

            # 過去3年のデータを取得
            has_any = False
            rate_cols = ["AVG", "OBP", "SLG", "OPS", "RC27", "XR27"]
            count_cols = ["HR", "RBI", "SB", "BB", "SO", "H"]

            for offset, yr in enumerate([y1, y2, y3], start=1):
                pdata = df[(df["player"] == player) & (df["year"] == yr)]
                if len(pdata) == 0:
                    # 不在年は0/NaN
                    feat[f"PA_{offset}"] = 0
                    feat[f"G_{offset}"] = 0
                    for col in rate_cols:
                        feat[f"{col}_{offset}"] = np.nan
                    for col in count_cols:
                        feat[f"{col}_rate_{offset}"] = np.nan
                    feat[f"present_{offset}"] = 0
                else:
                    has_any = True
                    row = pdata.iloc[0]
                    pa = row["PA"]
                    feat[f"PA_{offset}"] = pa
                    feat[f"G_{offset}"] = row["G"]
                    for col in rate_cols:
                        feat[f"{col}_{offset}"] = row[col]
                    for col in count_cols:
                        feat[f"{col}_rate_{offset}"] = row[col] / pa if pa > 0 else 0
                    feat[f"present_{offset}"] = 1

            if not has_any:
                continue

            # wOBA/wRC+特徴量（npb.jpセイバーメトリクスデータから）
            if has_saber:
                player_norm = _norm_name(player)
                for offset, yr in enumerate([y1, y2, y3], start=1):
                    sdata = saber[(saber["player_norm"] == player_norm) & (saber["year"] == yr)]
                    if len(sdata) > 0:
                        feat[f"wOBA_{offset}"] = sdata.iloc[0]["wOBA"]
                        feat[f"wRC+_{offset}"] = sdata.iloc[0]["wRC+"]
                    else:
                        feat[f"wOBA_{offset}"] = np.nan
                        feat[f"wRC+_{offset}"] = np.nan
                # wOBAトレンド
                if not np.isnan(feat.get("wOBA_1", np.nan)) and not np.isnan(feat.get("wOBA_2", np.nan)):
                    feat["wOBA_trend"] = feat["wOBA_1"] - feat["wOBA_2"]
                else:
                    feat["wOBA_trend"] = 0

            # 追加特徴量
            # 直近年のPA
            feat["PA_total_3yr"] = feat["PA_1"] + feat["PA_2"] + feat["PA_3"]
            # PA トレンド（増加/減少）
            if feat["PA_1"] > 0 and feat["PA_2"] > 0:
                feat["PA_trend"] = feat["PA_1"] - feat["PA_2"]
            else:
                feat["PA_trend"] = 0
            # 過去年の在籍数
            feat["years_present"] = feat["present_1"] + feat["present_2"] + feat["present_3"]

            # OPSトレンド
            if not np.isnan(feat.get("OPS_1", np.nan)) and not np.isnan(feat.get("OPS_2", np.nan)):
                feat["OPS_trend"] = feat["OPS_1"] - feat["OPS_2"]
            else:
                feat["OPS_trend"] = 0

            # 年齢
            birthday = birthdays.get(player)
            feat["age"] = calc_age(birthday, target_year) if birthday is not None else np.nan

            rows.append(feat)

    return pd.DataFrame(rows)


# ==============================
# 特徴量エンジニアリング（投手）
# ==============================
def build_pitcher_features(df: pd.DataFrame) -> pd.DataFrame:
    birthdays = load_birthdays()
    years = sorted(df["year"].unique())
    rows = []

    for target_year in years:
        y1, y2, y3 = target_year - 1, target_year - 2, target_year - 3
        target_df = df[df["year"] == target_year]

        for _, tgt_row in target_df.iterrows():
            player = tgt_row["player"]
            if tgt_row["IP_num"] < 30:
                continue

            feat = {"player": player, "team": tgt_row["team"],
                    "target_year": target_year, "target_ERA": tgt_row["ERA"],
                    "target_WHIP": tgt_row["WHIP"]}

            has_any = False
            rate_cols = ["ERA", "WHIP"]

            for offset, yr in enumerate([y1, y2, y3], start=1):
                pdata = df[(df["player"] == player) & (df["year"] == yr)]
                if len(pdata) == 0:
                    feat[f"IP_{offset}"] = 0
                    feat[f"G_{offset}"] = 0
                    for col in rate_cols:
                        feat[f"{col}_{offset}"] = np.nan
                    feat[f"SO_rate_{offset}"] = np.nan
                    feat[f"BB_rate_{offset}"] = np.nan
                    feat[f"HRA_rate_{offset}"] = np.nan
                    feat[f"present_{offset}"] = 0
                else:
                    has_any = True
                    row = pdata.iloc[0]
                    ip = row["IP_num"]
                    feat[f"IP_{offset}"] = ip
                    feat[f"G_{offset}"] = row["G"]
                    for col in rate_cols:
                        feat[f"{col}_{offset}"] = row[col]
                    feat[f"SO_rate_{offset}"] = row["SO"] / ip if ip > 0 else 0
                    feat[f"BB_rate_{offset}"] = row["BB"] / ip if ip > 0 else 0
                    feat[f"HRA_rate_{offset}"] = row["HRA"] / ip if ip > 0 else 0
                    feat[f"present_{offset}"] = 1

            if not has_any:
                continue

            feat["IP_total_3yr"] = feat["IP_1"] + feat["IP_2"] + feat["IP_3"]
            if feat["IP_1"] > 0 and feat["IP_2"] > 0:
                feat["IP_trend"] = feat["IP_1"] - feat["IP_2"]
            else:
                feat["IP_trend"] = 0
            feat["years_present"] = feat["present_1"] + feat["present_2"] + feat["present_3"]

            if not np.isnan(feat.get("ERA_1", np.nan)) and not np.isnan(feat.get("ERA_2", np.nan)):
                feat["ERA_trend"] = feat["ERA_1"] - feat["ERA_2"]
            else:
                feat["ERA_trend"] = 0

            # 年齢
            birthday = birthdays.get(player)
            feat["age"] = calc_age(birthday, target_year) if birthday is not None else np.nan

            rows.append(feat)

    return pd.DataFrame(rows)


# ==============================
# モデル学習・評価
# ==============================
def get_feature_cols(feat_df: pd.DataFrame, exclude_prefixes=("player", "team", "target_")) -> list:
    """特徴量カラムを取得"""
    return [c for c in feat_df.columns
            if not any(c.startswith(p) for p in exclude_prefixes)]


def train_and_evaluate(feat_df: pd.DataFrame, target_col: str, label: str):
    """CVで学習・評価し、結果を返す"""
    feature_cols = get_feature_cols(feat_df)
    X = feat_df[feature_cols].copy()
    y = feat_df[target_col].copy()

    # NaNを-1で埋める（不在年のマーカー）
    X = X.fillna(-1)

    # holdout: 2024年を予測（2023以前で学習）
    mask_test = feat_df["target_year"] == 2024
    mask_train = feat_df["target_year"] < 2024

    X_train, y_train = X[mask_train], y[mask_train]
    X_test, y_test = X[mask_test], y[mask_test]

    print(f"\n{'=' * 60}")
    print(f"{label} 予測")
    print(f"{'=' * 60}")
    print(f"Train: {len(X_train)} samples (2018-2023 target years)")
    print(f"Test:  {len(X_test)} samples (2024 target year)")
    print(f"Features: {len(feature_cols)}")

    results = {}

    # LightGBM
    if HAS_LGB and len(X_train) > 0:
        lgb_params = {
            "objective": "regression",
            "metric": "mae",
            "verbosity": -1,
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 5,
            "num_leaves": 31,
            "min_child_samples": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        }
        model_lgb = lgb.LGBMRegressor(**lgb_params)
        model_lgb.fit(X_train, y_train)
        pred_lgb = model_lgb.predict(X_test)
        mae_lgb = mean_absolute_error(y_test, pred_lgb)
        rmse_lgb = np.sqrt(mean_squared_error(y_test, pred_lgb))
        print(f"\nLightGBM:  MAE={mae_lgb:.4f}  RMSE={rmse_lgb:.4f}")
        results["lgb"] = {"model": model_lgb, "pred": pred_lgb, "mae": mae_lgb, "rmse": rmse_lgb}

        # 特徴量重要度 top10
        imp = pd.Series(model_lgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print(f"Top 10 features:")
        for fname, fval in imp.head(10).items():
            print(f"  {fname}: {fval}")

    # XGBoost
    if HAS_XGB and len(X_train) > 0:
        xgb_params = {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "verbosity": 0,
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_child_weight": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        }
        model_xgb = xgb.XGBRegressor(**xgb_params)
        model_xgb.fit(X_train, y_train)
        pred_xgb = model_xgb.predict(X_test)
        mae_xgb = mean_absolute_error(y_test, pred_xgb)
        rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))
        print(f"\nXGBoost:   MAE={mae_xgb:.4f}  RMSE={rmse_xgb:.4f}")
        results["xgb"] = {"model": model_xgb, "pred": pred_xgb, "mae": mae_xgb, "rmse": rmse_xgb}

    # アンサンブル
    if "lgb" in results and "xgb" in results:
        pred_ens = (results["lgb"]["pred"] + results["xgb"]["pred"]) / 2
        mae_ens = mean_absolute_error(y_test, pred_ens)
        rmse_ens = np.sqrt(mean_squared_error(y_test, pred_ens))
        print(f"\nEnsemble:  MAE={mae_ens:.4f}  RMSE={rmse_ens:.4f}")
        results["ensemble"] = {"pred": pred_ens, "mae": mae_ens, "rmse": rmse_ens}

    return results, X_test, y_test, feat_df[mask_test]


def compare_with_marcel(ml_results: dict, feat_test: pd.DataFrame,
                        y_test: pd.Series, target_col: str,
                        marcel_path: str, marcel_col: str,
                        min_threshold: dict):
    """Marcel法との精度比較"""
    marcel_df = pd.read_csv(marcel_path)

    # テストデータとMarcel予測をマージ
    test_players = feat_test[["player", f"target_{target_col.replace('target_', '')}"]].copy()
    test_players.columns = ["player", f"actual"]
    merged = test_players.merge(marcel_df[["player", marcel_col]], on="player", how="inner")

    if len(merged) == 0:
        print(f"\nMarcel比較: マージできる選手がいません")
        return

    marcel_mae = mean_absolute_error(merged["actual"], merged[marcel_col])
    print(f"\n--- Marcel法との比較（共通 {len(merged)} 選手）---")
    print(f"Marcel:    MAE={marcel_mae:.4f}")

    # ML側も同じ選手に絞って比較
    common_idx = feat_test["player"].isin(merged["player"])
    for name, res in ml_results.items():
        if "pred" in res:
            pred_common = res["pred"][common_idx.values]
            y_common = y_test[common_idx.values]
            if len(pred_common) > 0:
                mae = mean_absolute_error(y_common, pred_common)
                print(f"{name:10s} MAE={mae:.4f}  {'<< BETTER' if mae < marcel_mae else '>> WORSE'}")


def main():
    print("=" * 60)
    print("XGBoost/LightGBM NPB成績予測")
    print("=" * 60)

    # 打者
    df_h = load_hitters()
    feat_h = build_hitter_features(df_h)
    print(f"\n打者特徴量: {len(feat_h)} samples, {len(get_feature_cols(feat_h))} features")

    h_results, X_test_h, y_test_h, feat_test_h = train_and_evaluate(
        feat_h, "target_OPS", "打者 OPS")

    # Marcel法の2024予測を再生成して比較
    from marcel_projection import marcel_hitter, load_hitters as marcel_load_h
    df_h_marcel = marcel_load_h()
    proj_h_2024 = marcel_hitter(df_h_marcel, 2024)
    if len(proj_h_2024) > 0:
        test_players = feat_test_h[["player", "target_OPS"]].copy()
        merged = test_players.merge(proj_h_2024[["player", "OPS"]], on="player", how="inner")
        if len(merged) > 0:
            marcel_mae = mean_absolute_error(merged["target_OPS"], merged["OPS"])
            print(f"\n--- Marcel法との直接比較（共通 {len(merged)} 選手、PA>=100）---")
            print(f"Marcel:    MAE={marcel_mae:.4f}")
            common_players = set(merged["player"])
            common_mask = feat_test_h["player"].isin(common_players)
            for name, res in h_results.items():
                if "pred" in res:
                    pred = res["pred"][common_mask.values]
                    actual = y_test_h[common_mask.values]
                    if len(pred) > 0:
                        mae = mean_absolute_error(actual, pred)
                        print(f"{name:10s} MAE={mae:.4f}  {'<< BETTER' if mae < marcel_mae else '>> WORSE'}")

    # 投手
    df_p = load_pitchers()
    feat_p = build_pitcher_features(df_p)
    print(f"\n投手特徴量: {len(feat_p)} samples, {len(get_feature_cols(feat_p))} features")

    p_results, X_test_p, y_test_p, feat_test_p = train_and_evaluate(
        feat_p, "target_ERA", "投手 ERA")

    # Marcel比較（投手）
    from marcel_projection import marcel_pitcher, load_pitchers as marcel_load_p
    df_p_marcel = marcel_load_p()
    proj_p_2024 = marcel_pitcher(df_p_marcel, 2024)
    if len(proj_p_2024) > 0:
        test_players_p = feat_test_p[["player", "target_ERA"]].copy()
        merged_p = test_players_p.merge(proj_p_2024[["player", "ERA"]], on="player", how="inner")
        if len(merged_p) > 0:
            marcel_mae_p = mean_absolute_error(merged_p["target_ERA"], merged_p["ERA"])
            print(f"\n--- Marcel法との直接比較（共通 {len(merged_p)} 選手、IP>=30）---")
            print(f"Marcel:    MAE={marcel_mae_p:.4f}")
            common_players_p = set(merged_p["player"])
            common_mask_p = feat_test_p["player"].isin(common_players_p)
            for name, res in p_results.items():
                if "pred" in res:
                    pred = res["pred"][common_mask_p.values]
                    actual = y_test_p[common_mask_p.values]
                    if len(pred) > 0:
                        mae = mean_absolute_error(actual, pred)
                        print(f"{name:10s} MAE={mae:.4f}  {'✅ BETTER' if mae < marcel_mae_p else '❌ WORSE'}")

    # 2025年予測を出力
    print(f"\n{'=' * 60}")
    print("2025年予測出力")
    print(f"{'=' * 60}")

    # 打者2025予測
    feat_h_2025 = build_hitter_features_for_prediction(df_h, 2025)
    if len(feat_h_2025) > 0 and h_results:
        # ensembleにはmodelがないのでlgb/xgbから選ぶ
        model_candidates = {k: v for k, v in h_results.items() if "model" in v}
        best_model_name = min(model_candidates, key=lambda k: model_candidates[k].get("mae", 999))
        best = model_candidates[best_model_name]
        if "model" in best:
            feature_cols = get_feature_cols(feat_h_2025)
            X_2025 = feat_h_2025[feature_cols].fillna(-1)
            pred = best["model"].predict(X_2025)
            feat_h_2025["pred_OPS"] = np.round(pred, 3)
            out = feat_h_2025[["player", "team", "PA_1", "pred_OPS"]].sort_values("pred_OPS", ascending=False)
            out = out[out["PA_1"] >= 200].head(20)
            print(f"\n打者 Top 20 by predicted OPS (model: {best_model_name}):")
            print(out.to_string(index=False))

            out_path = OUT_DIR / "ml_hitters_2025.csv"
            feat_h_2025[["player", "team", "pred_OPS"]].to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"Saved: {out_path}")

    # 投手2025予測
    feat_p_2025 = build_pitcher_features_for_prediction(df_p, 2025)
    if len(feat_p_2025) > 0 and p_results:
        model_candidates_p = {k: v for k, v in p_results.items() if "model" in v}
        best_model_name = min(model_candidates_p, key=lambda k: model_candidates_p[k].get("mae", 999))
        best = model_candidates_p[best_model_name]
        if "model" in best:
            feature_cols = get_feature_cols(feat_p_2025)
            X_2025 = feat_p_2025[feature_cols].fillna(-1)
            pred = best["model"].predict(X_2025)
            feat_p_2025["pred_ERA"] = np.round(pred, 2)
            out = feat_p_2025[["player", "team", "IP_1", "pred_ERA"]].sort_values("pred_ERA")
            out = out[out["IP_1"] >= 50].head(20)
            print(f"\n投手 Top 20 by predicted ERA (model: {best_model_name}):")
            print(out.to_string(index=False))

            out_path = OUT_DIR / "ml_pitchers_2025.csv"
            feat_p_2025[["player", "team", "pred_ERA"]].to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"Saved: {out_path}")


def build_hitter_features_for_prediction(df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """予測用: ターゲット年のデータなしで特徴量を構築"""
    birthdays = load_birthdays()
    saber = load_sabermetrics()
    has_saber = len(saber) > 0
    y1, y2, y3 = target_year - 1, target_year - 2, target_year - 3
    rate_cols = ["AVG", "OBP", "SLG", "OPS", "RC27", "XR27"]
    count_cols = ["HR", "RBI", "SB", "BB", "SO", "H"]

    # 過去3年にいた全選手
    past = df[df["year"].isin([y1, y2, y3])]
    players = past["player"].unique()
    rows = []

    for player in players:
        feat = {"player": player, "target_year": target_year}
        has_any = False

        for offset, yr in enumerate([y1, y2, y3], start=1):
            pdata = df[(df["player"] == player) & (df["year"] == yr)]
            if len(pdata) == 0:
                feat[f"PA_{offset}"] = 0
                feat[f"G_{offset}"] = 0
                for col in rate_cols:
                    feat[f"{col}_{offset}"] = np.nan
                for col in count_cols:
                    feat[f"{col}_rate_{offset}"] = np.nan
                feat[f"present_{offset}"] = 0
            else:
                has_any = True
                row = pdata.iloc[0]
                pa = row["PA"]
                feat[f"PA_{offset}"] = pa
                feat[f"G_{offset}"] = row["G"]
                feat["team"] = row["team"]
                for col in rate_cols:
                    feat[f"{col}_{offset}"] = row[col]
                for col in count_cols:
                    feat[f"{col}_rate_{offset}"] = row[col] / pa if pa > 0 else 0
                feat[f"present_{offset}"] = 1

        if not has_any:
            continue

        # wOBA/wRC+特徴量
        if has_saber:
            player_norm = _norm_name(player)
            for offset, yr in enumerate([y1, y2, y3], start=1):
                sdata = saber[(saber["player_norm"] == player_norm) & (saber["year"] == yr)]
                if len(sdata) > 0:
                    feat[f"wOBA_{offset}"] = sdata.iloc[0]["wOBA"]
                    feat[f"wRC+_{offset}"] = sdata.iloc[0]["wRC+"]
                else:
                    feat[f"wOBA_{offset}"] = np.nan
                    feat[f"wRC+_{offset}"] = np.nan
            if not np.isnan(feat.get("wOBA_1", np.nan)) and not np.isnan(feat.get("wOBA_2", np.nan)):
                feat["wOBA_trend"] = feat["wOBA_1"] - feat["wOBA_2"]
            else:
                feat["wOBA_trend"] = 0

        feat["PA_total_3yr"] = feat["PA_1"] + feat["PA_2"] + feat["PA_3"]
        feat["PA_trend"] = (feat["PA_1"] - feat["PA_2"]) if feat["PA_1"] > 0 and feat["PA_2"] > 0 else 0
        feat["years_present"] = feat["present_1"] + feat["present_2"] + feat["present_3"]
        feat["OPS_trend"] = (feat["OPS_1"] - feat["OPS_2"]) if not np.isnan(feat.get("OPS_1", np.nan)) and not np.isnan(feat.get("OPS_2", np.nan)) else 0

        birthday = birthdays.get(player)
        feat["age"] = calc_age(birthday, target_year) if birthday is not None else np.nan

        if "team" not in feat:
            feat["team"] = "?"
        rows.append(feat)

    return pd.DataFrame(rows)


def build_pitcher_features_for_prediction(df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    birthdays = load_birthdays()
    y1, y2, y3 = target_year - 1, target_year - 2, target_year - 3
    rate_cols = ["ERA", "WHIP"]

    past = df[df["year"].isin([y1, y2, y3])]
    players = past["player"].unique()
    rows = []

    for player in players:
        feat = {"player": player, "target_year": target_year}
        has_any = False

        for offset, yr in enumerate([y1, y2, y3], start=1):
            pdata = df[(df["player"] == player) & (df["year"] == yr)]
            if len(pdata) == 0:
                feat[f"IP_{offset}"] = 0
                feat[f"G_{offset}"] = 0
                for col in rate_cols:
                    feat[f"{col}_{offset}"] = np.nan
                feat[f"SO_rate_{offset}"] = np.nan
                feat[f"BB_rate_{offset}"] = np.nan
                feat[f"HRA_rate_{offset}"] = np.nan
                feat[f"present_{offset}"] = 0
            else:
                has_any = True
                row = pdata.iloc[0]
                ip = row["IP_num"]
                feat[f"IP_{offset}"] = ip
                feat[f"G_{offset}"] = row["G"]
                feat["team"] = row["team"]
                for col in rate_cols:
                    feat[f"{col}_{offset}"] = row[col]
                feat[f"SO_rate_{offset}"] = row["SO"] / ip if ip > 0 else 0
                feat[f"BB_rate_{offset}"] = row["BB"] / ip if ip > 0 else 0
                feat[f"HRA_rate_{offset}"] = row["HRA"] / ip if ip > 0 else 0
                feat[f"present_{offset}"] = 1

        if not has_any:
            continue

        feat["IP_total_3yr"] = feat["IP_1"] + feat["IP_2"] + feat["IP_3"]
        feat["IP_trend"] = (feat["IP_1"] - feat["IP_2"]) if feat["IP_1"] > 0 and feat["IP_2"] > 0 else 0
        feat["years_present"] = feat["present_1"] + feat["present_2"] + feat["present_3"]
        feat["ERA_trend"] = (feat["ERA_1"] - feat["ERA_2"]) if not np.isnan(feat.get("ERA_1", np.nan)) and not np.isnan(feat.get("ERA_2", np.nan)) else 0

        birthday = birthdays.get(player)
        feat["age"] = calc_age(birthday, target_year) if birthday is not None else np.nan

        if "team" not in feat:
            feat["team"] = "?"
        rows.append(feat)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
