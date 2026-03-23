"""
ベイズ予測モジュール（Phase 1: 日本人選手 + Phase 2: 外国人選手）

研究リポ(npb-bayes-projection)で検証済みのStan事後分布を用いて、
Marcel予測にRidge補正を適用し、信頼区間（CI）付き予測を生成する。

日本人選手:
  Layer 1: Marcel（基盤 — 変更なし）
  Layer 2: Stan補正（事前計算済み事後分布）
  Layer 3: ML（XGBoost/LightGBM — ml_projection.py）
  Layer 4: BMA（ベイズモデル平均）

外国人選手（NPB初年度）:
  前リーグ成績 × 換算係数 + Stan v2補正
  前リーグ成績なし → リーグ別平均値で代替

ランタイム設計:
  - Stan学習はGitHub Actionsのみ（cmdstanpy不要）
  - posteriors.json（beta/sigma/standardization）をロード
  - NumPyサンプリングでCI算出（5,000 draws）
  - RPi5 4GB RAM対応

Data sources:
- プロ野球データFreak (https://baseball-data.com)
- 日本野球機構 NPB (https://npb.jp)
"""

import json
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path

from config import (
    BAYES_DIR, DATA_END_YEAR, PROJECTIONS_DIR, TARGET_YEAR,
)
from marcel_projection import load_birthdays, calc_age

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = PROJECTIONS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 5000
PEAK_AGE = 29
MIN_PA_HITTER = 30
MIN_IP_PITCHER = 10


# ── Posterior Store ──────────────────────────────────────────────────────────

class PosteriorStore:
    """posteriors.json をロードし、モデルパラメータを提供する。"""

    def __init__(self, path: Path | None = None):
        if path is None:
            path = BAYES_DIR / "posteriors.json"
        with open(path, encoding="utf-8") as f:
            self._data = json.load(f)

    @property
    def version(self) -> str:
        return self._data["version"]

    def jpn_hitter(self) -> dict:
        return self._data["jpn_hitter"]

    def jpn_pitcher(self) -> dict:
        return self._data["jpn_pitcher"]

    def model_weights(self, category: str) -> dict:
        return self._data["model_weights"].get(category, {})

    def conversion_factors(self) -> dict:
        return self._data.get("conversion_factors", {})

    def foreign_hitter(self) -> dict:
        return self._data.get("foreign_hitter", {})

    def foreign_pitcher(self) -> dict:
        return self._data.get("foreign_pitcher", {})


# ── Data loaders ─────────────────────────────────────────────────────────────

def _norm_name(name: str) -> str:
    return str(name).replace("\u3000", " ").strip()


def _parse_ip(ip_val) -> float:
    try:
        s = str(ip_val)
        if "." in s:
            parts = s.split(".")
            return int(parts[0]) + int(parts[1]) / 3.0
        return float(s)
    except (ValueError, IndexError):
        return 0.0


def load_sabermetrics() -> pd.DataFrame:
    path = OUT_DIR / f"npb_sabermetrics_2015_{DATA_END_YEAR}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["player_norm"] = df["player"].apply(_norm_name)
    return df


def load_raw_pitchers() -> pd.DataFrame:
    path = RAW_DIR / f"npb_pitchers_2015_{DATA_END_YEAR}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ["ERA", "WHIP", "IP", "ER", "HA", "SO", "BB", "HBP", "HRA", "BF"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["IP_num"] = df["IP"].apply(_parse_ip)
    return df


def load_marcel_hitters() -> pd.DataFrame:
    path = OUT_DIR / f"marcel_hitters_{TARGET_YEAR}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def load_marcel_pitchers() -> pd.DataFrame:
    path = OUT_DIR / f"marcel_pitchers_{TARGET_YEAR}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def load_ml_hitters() -> pd.DataFrame:
    path = OUT_DIR / f"ml_hitters_{TARGET_YEAR}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def load_ml_pitchers() -> pd.DataFrame:
    path = OUT_DIR / f"ml_pitchers_{TARGET_YEAR}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_hitter_features(saber_df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """前年のK%/BB%/BABIP/age_from_peakを抽出。"""
    if len(saber_df) == 0:
        return pd.DataFrame()

    birthdays = load_birthdays()
    prev_year = target_year - 1
    prev = saber_df[saber_df["year"] == prev_year].copy()
    prev = prev[prev["PA"] >= MIN_PA_HITTER]

    # 選手名正規化（全角スペース→半角）でMarcelとのマッチ率を上げる
    prev["player_join"] = prev["player"].apply(_norm_name)

    # K%, BB%
    prev["K_pct"] = prev["SO"] / prev["PA"]
    prev["BB_pct"] = prev["BB"] / prev["PA"]

    # BABIP = (H - HR) / (AB - SO - HR + SF)
    denom = (prev["AB"] - prev["SO"] - prev["HR"] + prev["SF"]).clip(lower=1)
    prev["BABIP"] = (prev["H"] - prev["HR"]) / denom

    # age_from_peak
    ages = []
    for _, row in prev.iterrows():
        bday = birthdays.get(row["player"])
        if bday is None:
            bday = birthdays.get(row["player_join"])
        if bday is not None:
            age = calc_age(bday, target_year)
            ages.append(age - PEAK_AGE if not np.isnan(age) else np.nan)
        else:
            ages.append(np.nan)
    prev["age_from_peak"] = ages

    # NaNを列平均で埋める
    for col in ["K_pct", "BB_pct", "BABIP", "age_from_peak"]:
        col_mean = prev[col].mean()
        prev[col] = prev[col].fillna(col_mean if not pd.isna(col_mean) else 0.0)

    return prev[["player", "player_join", "K_pct", "BB_pct", "BABIP", "age_from_peak"]].copy()


def extract_pitcher_features(pitchers_df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """前年のK%/BB%/K/9/BB/9/age_from_peakを抽出。"""
    if len(pitchers_df) == 0:
        return pd.DataFrame()

    birthdays = load_birthdays()
    prev_year = target_year - 1
    prev = pitchers_df[pitchers_df["year"] == prev_year].copy()
    prev = prev[prev["IP_num"] >= MIN_IP_PITCHER]
    prev = prev[prev["BF"] > 0]

    # 選手名正規化
    prev["player_join"] = prev["player"].apply(_norm_name)

    prev["K_pct"] = prev["SO"] / prev["BF"]
    prev["BB_pct"] = prev["BB"] / prev["BF"]
    prev["K_per_9"] = prev["SO"] * 9.0 / prev["IP_num"].clip(lower=0.1)
    prev["BB_per_9"] = prev["BB"] * 9.0 / prev["IP_num"].clip(lower=0.1)

    ages = []
    for _, row in prev.iterrows():
        bday = birthdays.get(row["player"])
        if bday is None:
            bday = birthdays.get(row["player_join"])
        if bday is not None:
            age = calc_age(bday, target_year)
            ages.append(age - PEAK_AGE if not np.isnan(age) else np.nan)
        else:
            ages.append(np.nan)
    prev["age_from_peak"] = ages

    for col in ["K_pct", "BB_pct", "K_per_9", "BB_per_9", "age_from_peak"]:
        col_mean = prev[col].mean()
        prev[col] = prev[col].fillna(col_mean if not pd.isna(col_mean) else 0.0)

    return prev[["player", "player_join", "K_pct", "BB_pct", "K_per_9", "BB_per_9", "age_from_peak"]].copy()


# ── Stan correction ──────────────────────────────────────────────────────────

def apply_stan_correction(
    marcel_value: float,
    features: dict[str, float],
    model_params: dict,
) -> tuple[float, float, float, float, float]:
    """
    Marcel予測値にStan Ridge補正を適用し、点推定+CIを返す。

    Returns:
        (stan_pred, ci80_lo, ci80_hi, ci95_lo, ci95_hi)
    """
    beta = model_params["beta"]
    std_info = model_params["standardization"]
    sigma = model_params["sigma_residual"]

    # z-score standardization
    delta = 0.0
    for feat_name in model_params["features"]:
        raw = features.get(feat_name, 0.0)
        mean = std_info["means"][feat_name]
        std = std_info["stds"][feat_name]
        z = (raw - mean) / std if std > 0 else 0.0
        delta += beta[feat_name] * z

    stan_pred = marcel_value + delta

    # NumPy sampling for CI
    rng = np.random.default_rng(42)
    samples = rng.normal(stan_pred, sigma, size=N_SAMPLES)

    ci80_lo, ci80_hi = np.percentile(samples, [10, 90])
    ci95_lo, ci95_hi = np.percentile(samples, [2.5, 97.5])

    return stan_pred, ci80_lo, ci80_hi, ci95_lo, ci95_hi


# ── BMA ensemble ─────────────────────────────────────────────────────────────

def bma_predict(
    marcel_val: float | None,
    stan_val: float | None,
    ml_val: float | None,
    weights: dict,
) -> float | None:
    """ベイズモデル平均。使用可能なモデルの重みを再正規化。"""
    components = {}
    if marcel_val is not None and "marcel" in weights:
        components["marcel"] = (marcel_val, weights["marcel"])
    if stan_val is not None and "stan" in weights:
        components["stan"] = (stan_val, weights["stan"])
    if ml_val is not None and "ml" in weights:
        components["ml"] = (ml_val, weights["ml"])

    if not components:
        return None

    total_w = sum(w for _, w in components.values())
    if total_w == 0:
        return None

    return sum(val * w / total_w for val, w in components.values())


# ── OPS ↔ wOBA 変換 ─────────────────────────────────────────────────────────

def woba_to_ops_approx(woba: float) -> float:
    """wOBA→OPS近似変換（NPBリーグ平均付近で線形近似）。
    NPB 2015-2025平均: wOBA≈0.310 → OPS≈0.690, wOBA≈0.400 → OPS≈0.900
    slope ≈ 2.33
    """
    return 0.690 + (woba - 0.310) * 2.33


# ── Main predictions ─────────────────────────────────────────────────────────

def predict_hitters(store: PosteriorStore) -> pd.DataFrame:
    """日本人打者のベイズ予測を生成。"""
    marcel_df = load_marcel_hitters()
    if len(marcel_df) == 0:
        print("WARNING: Marcel hitter projections not found")
        return pd.DataFrame()

    saber_df = load_sabermetrics()
    features_df = extract_hitter_features(saber_df, TARGET_YEAR)
    ml_df = load_ml_hitters()
    model_params = store.jpn_hitter()

    # Marcel OPS → wOBA近似（逆変換）: wOBA ≈ 0.310 + (OPS - 0.690) / 2.33
    def ops_to_woba_approx(ops):
        return 0.310 + (ops - 0.690) / 2.33

    results = []
    for _, row in marcel_df.iterrows():
        player = row["player"]
        team = row["team"]
        marcel_ops = row["OPS"]
        marcel_pa = row.get("PA", 0)

        # Marcel OPS → wOBA
        marcel_woba = ops_to_woba_approx(marcel_ops)

        # 特徴量取得（全角/半角スペース両方で検索）
        player_norm = _norm_name(player)
        feat_row = features_df[features_df["player"] == player]
        if len(feat_row) == 0:
            feat_row = features_df[features_df["player_join"] == player_norm]
        if len(feat_row) == 0:
            # 特徴量なし → Stan補正なし、Marcel値をそのまま使用
            results.append({
                "player": player,
                "team": team,
                "PA": marcel_pa,
                "marcel_OPS": round(marcel_ops, 3),
                "stan_wOBA": None,
                "stan_OPS": None,
                "bayes_OPS": round(marcel_ops, 3),
                "bayes_OPS_lo80": None,
                "bayes_OPS_hi80": None,
                "bayes_OPS_lo95": None,
                "bayes_OPS_hi95": None,
                "stan_delta": None,
                "method": "marcel_only",
            })
            continue

        feat = feat_row.iloc[0]
        features = {
            "K_pct": feat["K_pct"],
            "BB_pct": feat["BB_pct"],
            "BABIP": feat["BABIP"],
            "age_from_peak": feat["age_from_peak"],
        }

        # Stan correction (wOBA空間で)
        stan_woba, ci80_lo, ci80_hi, ci95_lo, ci95_hi = apply_stan_correction(
            marcel_woba, features, model_params
        )
        stan_ops = woba_to_ops_approx(stan_woba)
        ops_ci80_lo = woba_to_ops_approx(ci80_lo)
        ops_ci80_hi = woba_to_ops_approx(ci80_hi)
        ops_ci95_lo = woba_to_ops_approx(ci95_lo)
        ops_ci95_hi = woba_to_ops_approx(ci95_hi)

        # ML予測取得（名前正規化してマッチ）
        ml_ops = None
        if len(ml_df) > 0:
            ml_row = ml_df[ml_df["player"] == player]
            if len(ml_row) == 0 and "player" in ml_df.columns:
                ml_df_norm = ml_df["player"].apply(_norm_name)
                ml_row = ml_df[ml_df_norm == player_norm]
            if len(ml_row) > 0:
                ml_ops = float(ml_row.iloc[0]["pred_OPS"])

        # BMA重み選択（PA >= 200: regular, それ以外: bench）
        category = "jpn_regular" if marcel_pa >= 200 else "jpn_bench"
        weights = store.model_weights(category)
        bayes_ops = bma_predict(marcel_ops, stan_ops, ml_ops, weights)
        if bayes_ops is None:
            bayes_ops = stan_ops

        results.append({
            "player": player,
            "team": team,
            "PA": marcel_pa,
            "marcel_OPS": round(marcel_ops, 3),
            "stan_wOBA": round(stan_woba, 4),
            "stan_OPS": round(stan_ops, 3),
            "bayes_OPS": round(bayes_ops, 3),
            "bayes_OPS_lo80": round(ops_ci80_lo, 3),
            "bayes_OPS_hi80": round(ops_ci80_hi, 3),
            "bayes_OPS_lo95": round(ops_ci95_lo, 3),
            "bayes_OPS_hi95": round(ops_ci95_hi, 3),
            "stan_delta": round(stan_woba - marcel_woba, 5),
            "method": "bma_jpn" if ml_ops is not None else "stan_marcel",
        })

    return pd.DataFrame(results)


def predict_pitchers(store: PosteriorStore) -> pd.DataFrame:
    """日本人投手のベイズ予測を生成。"""
    marcel_df = load_marcel_pitchers()
    if len(marcel_df) == 0:
        print("WARNING: Marcel pitcher projections not found")
        return pd.DataFrame()

    pitchers_df = load_raw_pitchers()
    features_df = extract_pitcher_features(pitchers_df, TARGET_YEAR)
    ml_df = load_ml_pitchers()
    model_params = store.jpn_pitcher()

    results = []
    for _, row in marcel_df.iterrows():
        player = row["player"]
        team = row["team"]
        marcel_era = row["ERA"]
        marcel_ip = row.get("IP", 0)

        player_norm = _norm_name(player)
        feat_row = features_df[features_df["player"] == player]
        if len(feat_row) == 0:
            feat_row = features_df[features_df["player_join"] == player_norm]
        if len(feat_row) == 0:
            results.append({
                "player": player,
                "team": team,
                "IP": marcel_ip,
                "marcel_ERA": round(marcel_era, 2),
                "stan_ERA": None,
                "bayes_ERA": round(marcel_era, 2),
                "bayes_ERA_lo80": None,
                "bayes_ERA_hi80": None,
                "bayes_ERA_lo95": None,
                "bayes_ERA_hi95": None,
                "stan_delta": None,
                "method": "marcel_only",
            })
            continue

        feat = feat_row.iloc[0]
        features = {
            "K_pct": feat["K_pct"],
            "BB_pct": feat["BB_pct"],
            "K_per_9": feat["K_per_9"],
            "BB_per_9": feat["BB_per_9"],
            "age_from_peak": feat["age_from_peak"],
        }

        # Stan correction (ERA空間で)
        stan_era, ci80_lo, ci80_hi, ci95_lo, ci95_hi = apply_stan_correction(
            marcel_era, features, model_params
        )

        # ERA下限クリップ（負のERAは物理的にありえない）
        stan_era = max(0.0, stan_era)
        ci80_lo = max(0.0, ci80_lo)
        ci95_lo = max(0.0, ci95_lo)

        # ML予測取得（名前正規化してマッチ）
        ml_era = None
        if len(ml_df) > 0:
            ml_row = ml_df[ml_df["player"] == player]
            if len(ml_row) == 0 and "player" in ml_df.columns:
                ml_df_norm = ml_df["player"].apply(_norm_name)
                ml_row = ml_df[ml_df_norm == player_norm]
            if len(ml_row) > 0:
                ml_era = float(ml_row.iloc[0]["pred_ERA"])

        # BMA重み選択（IP >= 50: regular, それ以外: bench）
        ip_num = float(marcel_ip) if not pd.isna(marcel_ip) else 0
        category = "jpn_regular" if ip_num >= 50 else "jpn_bench"
        weights = store.model_weights(category)
        bayes_era = bma_predict(marcel_era, stan_era, ml_era, weights)
        if bayes_era is None:
            bayes_era = stan_era

        results.append({
            "player": player,
            "team": team,
            "IP": marcel_ip,
            "marcel_ERA": round(marcel_era, 2),
            "stan_ERA": round(stan_era, 2),
            "bayes_ERA": round(bayes_era, 2),
            "bayes_ERA_lo80": round(ci80_lo, 2),
            "bayes_ERA_hi80": round(ci80_hi, 2),
            "bayes_ERA_lo95": round(ci95_lo, 2),
            "bayes_ERA_hi95": round(ci95_hi, 2),
            "stan_delta": round(stan_era - marcel_era, 4),
            "method": "bma_jpn" if ml_era is not None else "stan_marcel",
        })

    return pd.DataFrame(results)


# ── Foreign player predictions ───────────────────────────────────────────────

FOREIGN_DIR = DATA_DIR / "foreign"


def load_foreign_master() -> pd.DataFrame:
    path = FOREIGN_DIR / "foreign_players_master.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8")


def load_foreign_prev_stats() -> pd.DataFrame:
    path = FOREIGN_DIR / "foreign_prev_stats.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8")


def predict_foreign_hitters(store: PosteriorStore) -> pd.DataFrame:
    """外国人打者のベイズ予測を生成。"""
    master = load_foreign_master()
    prev_stats = load_foreign_prev_stats()
    model = store.foreign_hitter()
    cf = store.conversion_factors()

    if len(master) == 0 or not model:
        return pd.DataFrame()

    # 2026年の新外国人打者（NPB初年度 or Marcel予測なし）
    foreign_hitters = master[
        (master["npb_first_year"] == TARGET_YEAR) & (master["player_type"] == "hitter")
    ]

    params = model["params"]
    std = model["standardization"]
    lg_avg_woba = model.get("league_avg_woba", {})

    rng = np.random.default_rng(42)
    results = []

    for _, row in foreign_hitters.iterrows():
        npb_name = row["npb_name"]
        team = row["first_team"]
        league = row["origin_league"] if pd.notna(row["origin_league"]) else "Other"
        position = row.get("position", "")

        # 前リーグ成績を探す
        prev = prev_stats[prev_stats["npb_name"] == npb_name]
        has_prev = len(prev) > 0 and prev.iloc[0]["wOBA"] > 0 if len(prev) > 0 else False

        if has_prev:
            p = prev.iloc[0]
            prev_woba = float(p["wOBA"])
            prev_K = float(p["K_pct"]) if pd.notna(p.get("K_pct")) else std["K_mean"]
            prev_BB = float(p["BB_pct"]) if pd.notna(p.get("BB_pct")) else std["BB_mean"]
            prev_pa = float(p["PA"]) if pd.notna(p.get("PA")) else 100
        else:
            # リーグ平均で代替
            prev_woba = lg_avg_woba.get(league, 0.300)
            prev_K = std["K_mean"]
            prev_BB = std["BB_mean"]
            prev_pa = 100

        # NPBに換算: wOBA × conversion factor
        woba_cf = cf.get("MLB_to_NPB_wOBA", 1.235)
        if league not in ("MLB", "AAA"):
            woba_cf = 1.0  # 非MLB/AAAは換算不要（既にNPBスケール or 不明）
        npb_woba_base = prev_woba * woba_cf if league in ("MLB", "AAA") else prev_woba

        # z-score
        z_woba = (prev_woba - std["woba_mean"]) / std["woba_sd"]
        z_K = (prev_K - std["K_mean"]) / std["K_sd"]
        z_BB = (prev_BB - std["BB_mean"]) / std["BB_sd"]
        z_age = 0.0  # 年齢不明の場合は平均（z=0）
        z_log_pa = (np.log(prev_pa) - std["log_pa_mean"]) / std["log_pa_sd"]

        # リーグ別beta
        league_key = league if league in ("MLB", "AAA") else "Other"
        beta_woba = params[f"beta_woba_{league_key}"]["mean"]

        # Stan v2 prediction
        mu = (lg_avg_woba.get(league_key, 0.300)
              + beta_woba * z_woba
              + params["beta_K"]["mean"] * z_K
              + params["beta_BB"]["mean"] * z_BB
              + params["beta_woba_sq"]["mean"] * z_woba ** 2
              + params["beta_K_BB"]["mean"] * z_K * z_BB
              + params["beta_age"]["mean"] * z_age)

        # ポジション補正
        if position and "捕手" in str(position):
            mu += params["beta_catcher"]["mean"]
        elif position and ("内野" in str(position) or "遊撃" in str(position) or "二塁" in str(position)):
            mu += params["beta_middle_inf"]["mean"]

        # 異分散性: sigma = sigma_base * exp(gamma * z_log_pa)
        sigma = params["sigma_base"]["mean"] * np.exp(
            params["gamma_pa"]["mean"] * z_log_pa
        )

        # Sampling
        samples = rng.normal(mu, sigma, size=N_SAMPLES)
        ci80_lo, ci80_hi = np.percentile(samples, [10, 90])
        ci95_lo, ci95_hi = np.percentile(samples, [2.5, 97.5])

        # wOBA → OPS近似
        pred_ops = woba_to_ops_approx(mu)
        ops_ci80 = (woba_to_ops_approx(ci80_lo), woba_to_ops_approx(ci80_hi))
        ops_ci95 = (woba_to_ops_approx(ci95_lo), woba_to_ops_approx(ci95_hi))

        results.append({
            "player": npb_name,
            "team": team,
            "origin_league": league,
            "prev_wOBA": round(prev_woba, 4) if has_prev else None,
            "bayes_wOBA": round(mu, 4),
            "bayes_OPS": round(pred_ops, 3),
            "bayes_OPS_lo80": round(ops_ci80[0], 3),
            "bayes_OPS_hi80": round(ops_ci80[1], 3),
            "bayes_OPS_lo95": round(ops_ci95[0], 3),
            "bayes_OPS_hi95": round(ops_ci95[1], 3),
            "method": "stan_v2" if has_prev else "league_avg",
        })

    return pd.DataFrame(results)


def predict_foreign_pitchers(store: PosteriorStore) -> pd.DataFrame:
    """外国人投手のベイズ予測を生成。"""
    master = load_foreign_master()
    prev_stats = load_foreign_prev_stats()
    model = store.foreign_pitcher()
    cf = store.conversion_factors()

    if len(master) == 0 or not model:
        return pd.DataFrame()

    foreign_pitchers = master[
        (master["npb_first_year"] == TARGET_YEAR) & (master["player_type"] == "pitcher")
    ]

    params = model["params"]
    std = model["standardization"]
    lg_avg_era = model.get("league_avg_era", {})

    rng = np.random.default_rng(43)
    results = []

    for _, row in foreign_pitchers.iterrows():
        npb_name = row["npb_name"]
        team = row["first_team"]
        league = row["origin_league"] if pd.notna(row["origin_league"]) else "Other"

        # 前リーグ成績を探す
        prev = prev_stats[prev_stats["npb_name"] == npb_name]
        has_prev = len(prev) > 0 and pd.notna(prev.iloc[0].get("ERA")) if len(prev) > 0 else False

        if has_prev:
            p = prev.iloc[0]
            prev_era = float(p["ERA"])
            prev_fip = float(p["FIP"]) if pd.notna(p.get("FIP")) else std["fip_mean"]
            prev_K = float(p["K_pct"]) if pd.notna(p.get("K_pct")) else std["K_mean"]
            prev_BB = float(p["BB_pct"]) if pd.notna(p.get("BB_pct")) else std["BB_mean"]
            prev_ip = float(p["IP"]) if pd.notna(p.get("IP")) else 50
        else:
            prev_era = lg_avg_era.get(league, 4.50)
            prev_fip = std["fip_mean"]
            prev_K = std["K_mean"]
            prev_BB = std["BB_mean"]
            prev_ip = 50

        # z-score
        z_era = (prev_era - std["era_mean"]) / std["era_sd"]
        z_fip = (prev_fip - std["fip_mean"]) / std["fip_sd"]
        z_K = (prev_K - std["K_mean"]) / std["K_sd"]
        z_BB = (prev_BB - std["BB_mean"]) / std["BB_sd"]
        z_age = 0.0
        z_log_ip = (np.log(prev_ip) - std["log_ip_mean"]) / std["log_ip_sd"]

        # リーグ別beta
        league_key = league if league in ("MLB", "AAA") else "Other"
        beta_era = params[f"beta_era_{league_key}"]["mean"]

        # Stan v2 prediction
        mu = (lg_avg_era.get(league_key, 4.50)
              + beta_era * z_era
              + params["beta_fip"]["mean"] * z_fip
              + params["beta_K"]["mean"] * z_K
              + params["beta_BB"]["mean"] * z_BB
              + params["beta_era_sq"]["mean"] * z_era ** 2
              + params["beta_K_BB"]["mean"] * z_K * z_BB
              + params["beta_age"]["mean"] * z_age)

        # ERA下限クリップ
        mu = max(0.5, mu)

        # 異分散性
        sigma = params["sigma_base"]["mean"] * np.exp(
            params["gamma_ip"]["mean"] * z_log_ip
        )

        # Sampling
        samples = rng.normal(mu, sigma, size=N_SAMPLES)
        samples = np.clip(samples, 0.0, None)  # ERAは非負
        ci80_lo, ci80_hi = np.percentile(samples, [10, 90])
        ci95_lo, ci95_hi = np.percentile(samples, [2.5, 97.5])

        results.append({
            "player": npb_name,
            "team": team,
            "origin_league": league,
            "prev_ERA": round(prev_era, 2) if has_prev else None,
            "bayes_ERA": round(mu, 2),
            "bayes_ERA_lo80": round(ci80_lo, 2),
            "bayes_ERA_hi80": round(ci80_hi, 2),
            "bayes_ERA_lo95": round(ci95_lo, 2),
            "bayes_ERA_hi95": round(ci95_hi, 2),
            "method": "stan_v2" if has_prev else "league_avg",
        })

    return pd.DataFrame(results)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"ベイズ予測 (target: {TARGET_YEAR})")
    print("=" * 60)

    store = PosteriorStore()
    print(f"Posteriors version: {store.version}")

    # 打者
    print(f"\n--- 打者ベイズ予測 ---")
    hitters = predict_hitters(store)
    if len(hitters) > 0:
        n_stan = (hitters["method"] != "marcel_only").sum()
        n_bma = (hitters["method"] == "bma_jpn").sum()
        print(f"Total: {len(hitters)} players (Stan補正: {n_stan}, BMA: {n_bma})")

        top = hitters[hitters["PA"] >= 200].sort_values("bayes_OPS", ascending=False).head(20)
        print(f"\nTop 20 by bayes_OPS (PA >= 200):")
        cols = ["player", "team", "PA", "marcel_OPS", "stan_OPS", "bayes_OPS",
                "bayes_OPS_lo80", "bayes_OPS_hi80", "stan_delta", "method"]
        print(top[cols].to_string(index=False))

        # 保存
        out_path = OUT_DIR / f"bayes_hitters_{TARGET_YEAR}.csv"
        hitters.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved: {out_path}")

    # 投手
    print(f"\n--- 投手ベイズ予測 ---")
    pitchers = predict_pitchers(store)
    if len(pitchers) > 0:
        n_stan = (pitchers["method"] != "marcel_only").sum()
        n_bma = (pitchers["method"] == "bma_jpn").sum()
        print(f"Total: {len(pitchers)} players (Stan補正: {n_stan}, BMA: {n_bma})")

        top = pitchers[pitchers["IP"] >= 50].sort_values("bayes_ERA").head(20)
        print(f"\nTop 20 by bayes_ERA (IP >= 50):")
        cols = ["player", "team", "IP", "marcel_ERA", "stan_ERA", "bayes_ERA",
                "bayes_ERA_lo80", "bayes_ERA_hi80", "stan_delta", "method"]
        print(top[cols].to_string(index=False))

        out_path = OUT_DIR / f"bayes_pitchers_{TARGET_YEAR}.csv"
        pitchers.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved: {out_path}")

    # 外国人打者
    print(f"\n--- 外国人打者ベイズ予測 ---")
    foreign_h = predict_foreign_hitters(store)
    if len(foreign_h) > 0:
        n_v2 = (foreign_h["method"] == "stan_v2").sum()
        n_avg = (foreign_h["method"] == "league_avg").sum()
        print(f"Total: {len(foreign_h)} players (Stan v2: {n_v2}, league_avg: {n_avg})")
        print(foreign_h[["player", "team", "origin_league", "prev_wOBA",
                         "bayes_wOBA", "bayes_OPS", "method"]].to_string(index=False))

        out_path = OUT_DIR / f"foreign_hitters_{TARGET_YEAR}.csv"
        foreign_h.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved: {out_path}")

    # 外国人投手
    print(f"\n--- 外国人投手ベイズ予測 ---")
    foreign_p = predict_foreign_pitchers(store)
    if len(foreign_p) > 0:
        n_v2 = (foreign_p["method"] == "stan_v2").sum()
        n_avg = (foreign_p["method"] == "league_avg").sum()
        print(f"Total: {len(foreign_p)} players (Stan v2: {n_v2}, league_avg: {n_avg})")
        print(foreign_p[["player", "team", "origin_league", "prev_ERA",
                         "bayes_ERA", "method"]].to_string(index=False))

        out_path = OUT_DIR / f"foreign_pitchers_{TARGET_YEAR}.csv"
        foreign_p.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved: {out_path}")

    # サマリー
    print(f"\n{'=' * 60}")
    print("ベイズ予測サマリー")
    print(f"{'=' * 60}")
    if len(hitters) > 0:
        stan_h = hitters[hitters["stan_delta"].notna()]
        if len(stan_h) > 0:
            print(f"日本人打者 Stan delta: mean={stan_h['stan_delta'].mean():.5f}, "
                  f"std={stan_h['stan_delta'].std():.5f}, "
                  f"range=[{stan_h['stan_delta'].min():.5f}, {stan_h['stan_delta'].max():.5f}]")
    if len(pitchers) > 0:
        stan_p = pitchers[pitchers["stan_delta"].notna()]
        if len(stan_p) > 0:
            print(f"日本人投手 Stan delta: mean={stan_p['stan_delta'].mean():.4f}, "
                  f"std={stan_p['stan_delta'].std():.4f}, "
                  f"range=[{stan_p['stan_delta'].min():.4f}, {stan_p['stan_delta'].max():.4f}]")
    if len(foreign_h) > 0:
        print(f"外国人打者: {len(foreign_h)} players, mean bayes_OPS={foreign_h['bayes_OPS'].mean():.3f}")
    if len(foreign_p) > 0:
        print(f"外国人投手: {len(foreign_p)} players, mean bayes_ERA={foreign_p['bayes_ERA'].mean():.2f}")


if __name__ == "__main__":
    main()
