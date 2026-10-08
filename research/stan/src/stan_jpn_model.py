"""
Step 7a / 11b: Japanese player Stan model with K%/BB%/age features.

Marcel prediction serves as the prior mean. The Stan model tests whether
K%/BB%/age (skill-level + aging) add predictive power beyond Marcel.

Model:
  Hitter:  actual_wOBA = Marcel_wOBA + delta_K * z_K + delta_BB * z_BB
                        + delta_BABIP * z_babip + delta_age * z_age + noise
  Pitcher: actual_ERA  = Marcel_ERA  + delta_K * z_K + delta_BB * z_BB
                        + delta_age * z_age + noise
  Pitcher: actual_FIP  = Marcel_FIP  + delta_K * z_K + delta_BB * z_BB
                        + delta_age * z_age + noise

Training: 2018-2021 (using 2015-2020 history for Marcel)
Backtest: 2022-2025 (using 2019-2024 history for Marcel)

Output:
  data/model/jpn_hitter_predictions.csv  (year, player, team, marcel_woba, stan_woba, actual_woba, PA)
  data/model/jpn_pitcher_predictions.csv (year, player, team, marcel_era, stan_era, actual_era, IP)
  data/model/jpn_comparison.json         (MAE summary: Marcel vs Stan)
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _log_elapsed(label: str, start: float, budget_min: int = 360):
    elapsed_min = (time.time() - start) / 60
    print(f"  [{label}] elapsed: {elapsed_min:.1f} min / {budget_min} min budget")
    if elapsed_min > budget_min * 0.8:
        print(f"  WARNING: {label} used {elapsed_min:.0f}/{budget_min} min "
              f"({elapsed_min / budget_min * 100:.0f}%) -- timeout risk!")


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "model"
RAW_DIR   = DATA_DIR / "raw"

# ── Age curve ───────────────────────────────────────────────────────────────
PEAK_AGE = 29    # assumed peak age for NPB hitters/pitchers

# ── Cutoffs ─────────────────────────────────────────────────────────────────
MIN_PA  = 30     # minimum PA (lowered from 50 for broader coverage, Step 14B)
MIN_IP  = 10     # minimum IP (lowered from 20 for broader coverage, Step 14B)

# ── Marcel parameters ────────────────────────────────────────────────────────
MARCEL_WEIGHTS   = {1: 5, 2: 4, 3: 3}   # years back: weight
REGRESS_PA_HIT   = 200                   # regression-to-mean PA (hitters)
REGRESS_IP_PITCH = 300                   # regression-to-mean IP (pitchers)

# ── Years ────────────────────────────────────────────────────────────────────
TRAIN_YEARS    = list(range(2018, 2022))  # 2018-2021
BACKTEST_YEARS = list(range(2022, 2026))  # 2022-2025


# ── IP format helper ─────────────────────────────────────────────────────────
def ip_to_decimal(ip: float) -> float:
    """Convert NPB IP (X.Y where Y = thirds) to decimal innings."""
    whole  = int(ip)
    thirds = round((ip - whole) * 10)
    return whole + thirds / 3.0


# ── Age helpers ──────────────────────────────────────────────────────────────
def normalize_name(name: str) -> str:
    """Normalize full-width spaces to half-width for name matching."""
    return name.replace("\u3000", " ").strip()


def load_birthday_df() -> pd.DataFrame:
    """Load birthday CSV with normalized player names."""
    bday_df = pd.read_csv(RAW_DIR / "npb_player_birthdays.csv", encoding="utf-8-sig")
    bday_df["player"] = bday_df["player"].apply(normalize_name)
    bday_df["birthday"] = pd.to_datetime(bday_df["birthday"])
    return bday_df


def add_age_from_peak(df: pd.DataFrame, bday_df: pd.DataFrame) -> pd.DataFrame:
    """Add age_from_peak column (age - PEAK_AGE) using birthday data.

    Age is calculated as of April 1 of the target year (NPB season start).
    Players without birthday data are dropped.
    """
    df = df.copy()
    df["_norm_player"] = df["player"].apply(normalize_name)
    bday_map = dict(zip(bday_df["player"], bday_df["birthday"]))

    ages = []
    for _, row in df.iterrows():
        bday = bday_map.get(row["_norm_player"])
        if bday is not None and not pd.isna(bday):
            season_start = pd.Timestamp(year=int(row["year"]), month=4, day=1)
            age = (season_start - bday).days / 365.25
            ages.append(age - PEAK_AGE)
        else:
            ages.append(np.nan)

    df["age_from_peak"] = ages
    n_before = len(df)
    df = df.dropna(subset=["age_from_peak"])
    n_after = len(df)
    if n_before > n_after:
        print(f"    Age match: {n_after}/{n_before} ({100*n_after/n_before:.1f}%)")
    df = df.drop(columns=["_norm_player"])
    return df


# ── Marcel helpers ────────────────────────────────────────────────────────────
def league_avg_woba(saber_df: pd.DataFrame, year: int) -> float:
    sub = saber_df[(saber_df["year"] == year - 1) & (saber_df["PA"] >= MIN_PA)]
    if len(sub) == 0:
        return 0.310
    return float(np.average(sub["wOBA"], weights=sub["PA"]))


def league_avg_era(pitchers_df: pd.DataFrame, year: int) -> float:
    sub = pitchers_df[pitchers_df["year"] == year - 1].copy()
    sub["IP_dec"] = sub["IP"].apply(ip_to_decimal)
    sub["ERA"]    = pd.to_numeric(sub["ERA"], errors="coerce")
    sub = sub[(sub["IP_dec"] >= MIN_IP) & sub["ERA"].notna()]
    if len(sub) == 0:
        return 3.80
    total_ip = sub["IP_dec"].sum()
    return float((sub["ERA"] * sub["IP_dec"]).sum() / total_ip) if total_ip > 0 else 3.80


def compute_fip_column(pitchers_df: pd.DataFrame) -> pd.DataFrame:
    """Add FIP column to pitchers DataFrame.

    raw_FIP = (13*HRA + 3*(BB+HBP) - 2*SO) / IP_dec
    cFIP = league_avg_ERA - league_avg_raw_FIP (per year, IP >= MIN_IP)
    FIP = raw_FIP + cFIP
    """
    df = pitchers_df.copy()
    df["_IP_dec"] = df["IP"].apply(ip_to_decimal)
    df["_ERA_num"] = pd.to_numeric(df["ERA"], errors="coerce")
    df["_raw_FIP"] = (
        13 * df["HRA"] + 3 * (df["BB"] + df["HBP"]) - 2 * df["SO"]
    ) / df["_IP_dec"].clip(lower=0.1)

    cfip_map = {}
    for yr, grp in df[df["_IP_dec"] >= MIN_IP].groupby("year"):
        grp = grp[grp["_ERA_num"].notna()]
        if len(grp) == 0:
            continue
        total_ip = grp["_IP_dec"].sum()
        lg_era = float((grp["_ERA_num"] * grp["_IP_dec"]).sum() / total_ip)
        lg_raw_fip = float((grp["_raw_FIP"] * grp["_IP_dec"]).sum() / total_ip)
        cfip_map[int(yr)] = lg_era - lg_raw_fip

    df["FIP"] = df["_raw_FIP"] + df["year"].map(cfip_map)
    df = df.drop(columns=["_IP_dec", "_ERA_num", "_raw_FIP"])
    return df


def league_avg_fip(pitchers_df: pd.DataFrame, year: int) -> float:
    """IP-weighted league average FIP from the prior year."""
    sub = pitchers_df[pitchers_df["year"] == year - 1].copy()
    sub["IP_dec"] = sub["IP"].apply(ip_to_decimal)
    sub = sub[(sub["IP_dec"] >= MIN_IP) & sub["FIP"].notna()]
    if len(sub) == 0:
        return 3.80
    total_ip = sub["IP_dec"].sum()
    return float((sub["FIP"] * sub["IP_dec"]).sum() / total_ip) if total_ip > 0 else 3.80


def compute_marcel_fip(pitchers_df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Marcel FIP projection for all pitchers with sufficient history.

    Also computes ip_stability (same as compute_marcel_era).
    """
    lg_avg = league_avg_fip(pitchers_df, target_year)
    rows = []
    for player, grp in pitchers_df.groupby("player"):
        w_total = fip_sum = 0.0
        ip_vals: list[float] = []
        for lag, w in MARCEL_WEIGHTS.items():
            yr = target_year - lag
            row = grp[grp["year"] == yr]
            if len(row) == 0:
                continue
            ip = ip_to_decimal(float(row.iloc[0]["IP"]))
            fip = row.iloc[0]["FIP"]
            if ip < MIN_IP or pd.isna(fip):
                continue
            fip = float(fip)
            w_total += w * ip
            fip_sum += w * ip * fip
            ip_vals.append(ip)
        if w_total == 0:
            continue
        recent = grp[grp["year"] == target_year - 1]
        if len(recent) == 0:
            continue
        team = recent.iloc[0]["team"]
        fip_raw = fip_sum / w_total
        fip_proj = (fip_raw * w_total + lg_avg * REGRESS_IP_PITCH) / (w_total + REGRESS_IP_PITCH)
        ip_stability = min(ip_vals) / max(ip_vals) if len(ip_vals) >= 2 else 0.5
        rows.append({"player": player, "team": team, "year": target_year,
                     "marcel_fip": round(fip_proj, 4), "lg_avg_fip": round(lg_avg, 4),
                     "ip_stability": round(ip_stability, 4)})
    return pd.DataFrame(rows)


def add_actual_fip(pitchers_df: pd.DataFrame, target_year: int,
                   df: pd.DataFrame) -> pd.DataFrame:
    """Merge actual FIP from the target year."""
    actual = pitchers_df[pitchers_df["year"] == target_year][["player", "IP", "FIP"]].copy()
    actual["actual_IP"] = actual["IP"].apply(ip_to_decimal)
    actual["actual_fip"] = actual["FIP"]
    actual = actual[(actual["actual_IP"] >= MIN_IP) & actual["actual_fip"].notna()]
    return df.merge(actual[["player", "actual_IP", "actual_fip"]], on="player", how="inner")


def compute_rookie_avg_woba(saber_df: pd.DataFrame, target_year: int) -> float:
    """PA-weighted average wOBA for rookies (first 1軍 year, 2016+).

    Uses rookies who debuted between 2016 and target_year-1.
    Falls back to league average if no rookie data available.
    """
    first_yr = saber_df.groupby("player")["year"].min()
    rookie_players = first_yr[(first_yr >= 2016) & (first_yr < target_year)].index
    rookies = saber_df[saber_df["player"].isin(rookie_players)].copy()
    rookies = rookies.merge(
        first_yr.rename("first_year"), left_on="player", right_index=True
    )
    rookies = rookies[rookies["year"] == rookies["first_year"]]
    rookies = rookies[(rookies["PA"] >= MIN_PA) & rookies["wOBA"].notna()]
    if len(rookies) == 0:
        return league_avg_woba(saber_df, target_year)
    return float(np.average(rookies["wOBA"], weights=rookies["PA"]))


def compute_rookie_avg_era(pitchers_df: pd.DataFrame, target_year: int) -> float:
    """IP-weighted average ERA for rookies (first 1軍 year, 2016+).

    Falls back to league average if no rookie data available.
    """
    pdf = pitchers_df.copy()
    pdf["_IP_dec"] = pdf["IP"].apply(ip_to_decimal)
    pdf["_ERA_num"] = pd.to_numeric(pdf["ERA"], errors="coerce")
    first_yr = pdf.groupby("player")["year"].min()
    rookie_players = first_yr[(first_yr >= 2016) & (first_yr < target_year)].index
    rookies = pdf[pdf["player"].isin(rookie_players)].copy()
    rookies = rookies.merge(
        first_yr.rename("first_year"), left_on="player", right_index=True
    )
    rookies = rookies[rookies["year"] == rookies["first_year"]]
    rookies = rookies[(rookies["_IP_dec"] >= MIN_IP) & rookies["_ERA_num"].notna()]
    if len(rookies) == 0:
        return league_avg_era(pitchers_df, target_year)
    return float(np.average(rookies["_ERA_num"], weights=rookies["_IP_dec"]))


def _add_uncovered_hitters(saber_df: pd.DataFrame, yr: int,
                           marcel_df: pd.DataFrame) -> pd.DataFrame:
    """Add uncovered hitters with rookie average as Marcel projection (Step 14A)."""
    actual = saber_df[
        (saber_df["year"] == yr) & (saber_df["PA"] >= MIN_PA) & saber_df["wOBA"].notna()
    ]
    actual = actual.sort_values("PA", ascending=False).drop_duplicates("player", keep="first")
    projected = set(marcel_df["player"]) if len(marcel_df) > 0 else set()
    missing = actual[~actual["player"].isin(projected)]
    if len(missing) == 0:
        return marcel_df
    rookie_woba = compute_rookie_avg_woba(saber_df, yr)
    lg = league_avg_woba(saber_df, yr)
    new_rows = pd.DataFrame({
        "player": missing["player"].values,
        "team": missing["team"].values,
        "year": yr,
        "marcel_woba": round(rookie_woba, 5),
        "lg_avg_woba": round(lg, 5),
        "pa_stability": 0.5,
    })
    if len(marcel_df) == 0:
        return new_rows
    return pd.concat([marcel_df, new_rows], ignore_index=True)


def _add_uncovered_pitchers(pitchers_df: pd.DataFrame, yr: int,
                            marcel_df: pd.DataFrame, metric: str = "era") -> pd.DataFrame:
    """Add uncovered pitchers with rookie average as Marcel projection (Step 14A)."""
    pdf = pitchers_df.copy()
    pdf["_IP_dec"] = pdf["IP"].apply(ip_to_decimal)
    pdf["_ERA_num"] = pd.to_numeric(pdf["ERA"], errors="coerce")
    actual = pdf[(pdf["year"] == yr) & (pdf["_IP_dec"] >= MIN_IP) & pdf["_ERA_num"].notna()]
    actual = actual.sort_values("_IP_dec", ascending=False).drop_duplicates("player", keep="first")
    projected = set(marcel_df["player"]) if len(marcel_df) > 0 else set()
    missing = actual[~actual["player"].isin(projected)]
    if len(missing) == 0:
        return marcel_df
    if metric == "era":
        rookie_val = compute_rookie_avg_era(pitchers_df, yr)
        lg = league_avg_era(pitchers_df, yr)
        new_rows = pd.DataFrame({
            "player": missing["player"].values,
            "team": missing["team"].values,
            "year": yr,
            "marcel_era": round(rookie_val, 4),
            "lg_avg_era": round(lg, 4),
            "ip_stability": 0.5,
        })
    else:
        rookie_val = league_avg_fip(pitchers_df, yr)
        lg = league_avg_fip(pitchers_df, yr)
        new_rows = pd.DataFrame({
            "player": missing["player"].values,
            "team": missing["team"].values,
            "year": yr,
            "marcel_fip": round(rookie_val, 4),
            "lg_avg_fip": round(lg, 4),
            "ip_stability": 0.5,
        })
    if len(marcel_df) == 0:
        return new_rows
    return pd.concat([marcel_df, new_rows], ignore_index=True)


def compute_marcel_woba(saber_df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Marcel wOBA projection for all players with sufficient history.

    Also computes pa_stability = min(PA_3yr) / max(PA_3yr) as an injury-risk
    proxy.  Players with only 1 year of data get stability = 0.5 (conservative).
    """
    lg_avg = league_avg_woba(saber_df, target_year)
    rows = []
    for player, grp in saber_df.groupby("player"):
        w_total = woba_sum = 0.0
        pa_vals: list[float] = []
        for lag, w in MARCEL_WEIGHTS.items():
            yr = target_year - lag
            row = grp[grp["year"] == yr]
            if len(row) == 0:
                continue
            pa   = float(row.iloc[0]["PA"])
            woba = float(row.iloc[0]["wOBA"])
            if pa >= MIN_PA and not np.isnan(woba):
                w_total  += w * pa
                woba_sum += w * pa * woba
                pa_vals.append(pa)
        if w_total == 0:
            continue
        # Most-recent year for team assignment
        recent = grp[grp["year"] == target_year - 1]
        if len(recent) == 0:
            continue
        team = recent.iloc[0]["team"]
        woba_raw  = woba_sum / w_total
        woba_proj = (woba_raw * w_total + lg_avg * REGRESS_PA_HIT) / (w_total + REGRESS_PA_HIT)
        pa_stability = min(pa_vals) / max(pa_vals) if len(pa_vals) >= 2 else 0.5
        rows.append({"player": player, "team": team, "year": target_year,
                     "marcel_woba": round(woba_proj, 5), "lg_avg_woba": round(lg_avg, 5),
                     "pa_stability": round(pa_stability, 4)})
    return pd.DataFrame(rows)


def compute_marcel_era(pitchers_df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Marcel ERA projection for all pitchers with sufficient history.

    Also computes ip_stability = min(IP_3yr) / max(IP_3yr) as an injury-risk
    proxy.  Pitchers with only 1 year of data get stability = 0.5 (conservative).
    """
    lg_avg = league_avg_era(pitchers_df, target_year)
    rows = []
    for player, grp in pitchers_df.groupby("player"):
        w_total = era_sum = 0.0
        ip_vals: list[float] = []
        for lag, w in MARCEL_WEIGHTS.items():
            yr = target_year - lag
            row = grp[grp["year"] == yr]
            if len(row) == 0:
                continue
            ip  = ip_to_decimal(float(row.iloc[0]["IP"]))
            era = pd.to_numeric(row.iloc[0]["ERA"], errors="coerce")
            if ip >= MIN_IP and not np.isnan(era):
                era = float(era)
                w_total += w * ip
                era_sum += w * ip * era
                ip_vals.append(ip)
        if w_total == 0:
            continue
        recent = grp[grp["year"] == target_year - 1]
        if len(recent) == 0:
            continue
        team = recent.iloc[0]["team"]
        era_raw  = era_sum / w_total
        era_proj = (era_raw * w_total + lg_avg * REGRESS_IP_PITCH) / (w_total + REGRESS_IP_PITCH)
        ip_stability = min(ip_vals) / max(ip_vals) if len(ip_vals) >= 2 else 0.5
        rows.append({"player": player, "team": team, "year": target_year,
                     "marcel_era": round(era_proj, 4), "lg_avg_era": round(lg_avg, 4),
                     "ip_stability": round(ip_stability, 4)})
    return pd.DataFrame(rows)


def add_kpct_bbpct_hitter(saber_df: pd.DataFrame, target_year: int,
                           marcel_df: pd.DataFrame) -> pd.DataFrame:
    """Add K%/BB%/BABIP/prev_woba_dev_sq features from the year before target.

    BABIP = (H - HR) / (AB - SO - HR + SF) captures luck in year t-1.
    prev_woba_dev_sq = (wOBA_t-1 - lg_avg)^2 captures extreme-performance
    regression tendency — players far from league average in either direction
    tend to regress.
    """
    cols = ["player", "PA", "SO", "BB", "AB", "H", "HR", "SF", "wOBA"]
    prev = saber_df[saber_df["year"] == target_year - 1][cols].copy()
    prev = prev[prev["PA"] >= MIN_PA]
    prev["K_pct"]  = prev["SO"] / prev["PA"]
    prev["BB_pct"] = prev["BB"] / prev["PA"]
    denom = (prev["AB"] - prev["SO"] - prev["HR"] + prev["SF"]).clip(lower=1)
    prev["BABIP"]  = (prev["H"] - prev["HR"]) / denom
    lg_avg = league_avg_woba(saber_df, target_year)
    prev["prev_woba_dev_sq"] = (prev["wOBA"] - lg_avg) ** 2
    merged = marcel_df.merge(
        prev[["player", "K_pct", "BB_pct", "BABIP", "prev_woba_dev_sq"]],
        on="player", how="left",
    )
    # Fill missing features for players without t-1 data (rookies, injured)
    for col in ["K_pct", "BB_pct", "BABIP", "prev_woba_dev_sq"]:
        col_mean = merged[col].mean()
        merged[col] = merged[col].fillna(col_mean if not pd.isna(col_mean) else 0.0)
    return merged


def add_kpct_bbpct_pitcher(pitchers_df: pd.DataFrame, target_year: int,
                            marcel_df: pd.DataFrame) -> pd.DataFrame:
    """Add K%/BB%/K_per_9/BB_per_9/prev_babip_p features from prior year.

    prev_babip_p = (HA - HRA) / (BF - SO - HRA) captures pitcher BABIP luck
    in year t-1.  Pitchers with low BABIP in t-1 tend to see ERA regress upward.
    """
    prev = pitchers_df[pitchers_df["year"] == target_year - 1][
        ["player", "IP", "SO", "BB", "BF", "HA", "HRA"]
    ].copy()
    prev["IP_dec"] = prev["IP"].apply(ip_to_decimal)
    prev = prev[prev["IP_dec"] >= MIN_IP]
    prev = prev[prev["BF"] > 0]
    prev["K_pct"]    = prev["SO"] / prev["BF"]
    prev["BB_pct"]   = prev["BB"] / prev["BF"]
    prev["K_per_9"]  = prev["SO"] * 9.0 / prev["IP_dec"]
    prev["BB_per_9"] = prev["BB"] * 9.0 / prev["IP_dec"]
    babip_denom = (prev["BF"] - prev["SO"] - prev["HRA"]).clip(lower=1)
    prev["prev_babip_p"] = (prev["HA"] - prev["HRA"]) / babip_denom
    merged = marcel_df.merge(
        prev[["player", "K_pct", "BB_pct", "K_per_9", "BB_per_9", "prev_babip_p"]],
        on="player", how="left",
    )
    # Fill missing features for players without t-1 data (rookies, injured)
    for col in ["K_pct", "BB_pct", "K_per_9", "BB_per_9", "prev_babip_p"]:
        col_mean = merged[col].mean()
        merged[col] = merged[col].fillna(col_mean if not pd.isna(col_mean) else 0.0)
    return merged


def add_actual_woba(saber_df: pd.DataFrame, target_year: int,
                    df: pd.DataFrame) -> pd.DataFrame:
    actual = saber_df[saber_df["year"] == target_year][["player", "PA", "wOBA"]].copy()
    actual = actual.rename(columns={"PA": "actual_PA", "wOBA": "actual_woba"})
    actual = actual[actual["actual_PA"] >= MIN_PA]
    return df.merge(actual, on="player", how="inner")


def add_actual_era(pitchers_df: pd.DataFrame, target_year: int,
                   df: pd.DataFrame) -> pd.DataFrame:
    actual = pitchers_df[pitchers_df["year"] == target_year][["player", "IP", "ERA"]].copy()
    actual["actual_IP"]  = actual["IP"].apply(ip_to_decimal)
    actual["actual_era"] = pd.to_numeric(actual["ERA"], errors="coerce")
    actual = actual[(actual["actual_IP"] >= MIN_IP) & actual["actual_era"].notna()]
    return df.merge(actual[["player", "actual_IP", "actual_era"]], on="player", how="inner")


def build_dataset(saber_df, pitchers_df, years, bday_df=None, include_rookies=True):
    """Build combined DataFrame for hitters and pitchers across given years.

    If bday_df is provided, adds age_from_peak column (players without birthday
    data are dropped).

    If include_rookies is True (Step 14A), adds first-year players with no Marcel
    projection using rookie average wOBA/ERA as their Marcel value.  Features are
    filled with column means via left join (z-score ≈ 0 → Ridge correction ≈ 0).

    Returns (hitters, pitchers_era, pitchers_fip). pitchers_fip is empty
    DataFrame if pitchers_df has no FIP column.
    """
    hit_rows, pit_rows, pit_fip_rows = [], [], []
    has_fip = "FIP" in pitchers_df.columns
    for yr in years:
        m_h = compute_marcel_woba(saber_df, yr)
        if include_rookies:
            m_h = _add_uncovered_hitters(saber_df, yr, m_h)
        if len(m_h) == 0:
            continue
        m_h = add_kpct_bbpct_hitter(saber_df, yr, m_h)
        m_h = add_actual_woba(saber_df, yr, m_h)
        hit_rows.append(m_h)

        m_p = compute_marcel_era(pitchers_df, yr)
        if include_rookies:
            m_p = _add_uncovered_pitchers(pitchers_df, yr, m_p, metric="era")
        if len(m_p) == 0:
            continue
        m_p = add_kpct_bbpct_pitcher(pitchers_df, yr, m_p)
        m_p = add_actual_era(pitchers_df, yr, m_p)
        pit_rows.append(m_p)

        if has_fip:
            m_pf = compute_marcel_fip(pitchers_df, yr)
            if include_rookies:
                m_pf = _add_uncovered_pitchers(pitchers_df, yr, m_pf, metric="fip")
            if len(m_pf) > 0:
                m_pf = add_kpct_bbpct_pitcher(pitchers_df, yr, m_pf)
                m_pf = add_actual_fip(pitchers_df, yr, m_pf)
                if len(m_pf) > 0:
                    pit_fip_rows.append(m_pf)

    hitters  = pd.concat(hit_rows,  ignore_index=True) if hit_rows  else pd.DataFrame()
    pitchers = pd.concat(pit_rows,  ignore_index=True) if pit_rows  else pd.DataFrame()
    pitchers_fip = pd.concat(pit_fip_rows, ignore_index=True) if pit_fip_rows else pd.DataFrame()

    if bday_df is not None:
        if len(hitters) > 0:
            hitters = add_age_from_peak(hitters, bday_df)
        if len(pitchers) > 0:
            pitchers = add_age_from_peak(pitchers, bday_df)
        if len(pitchers_fip) > 0:
            pitchers_fip = add_age_from_peak(pitchers_fip, bday_df)

    return hitters, pitchers, pitchers_fip


def standardize_features(train_df, test_df, feat_cols):
    """Z-score features using training-set statistics. Returns scaled arrays."""
    means = train_df[feat_cols].mean()
    stds  = train_df[feat_cols].std().replace(0, 1)
    train_z = (train_df[feat_cols] - means) / stds
    test_z  = (test_df[feat_cols]  - means) / stds
    return train_z.values, test_z.values, means.to_dict(), stds.to_dict()


def run_stan_model(model_path: Path, stan_data: dict, draws=1000, warmup=500):
    """Run Stan model via cmdstanpy. Returns posterior means."""
    from cmdstanpy import CmdStanModel
    model = CmdStanModel(stan_file=str(model_path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.sample(
            data=stan_data,
            chains=2,
            iter_sampling=draws,
            iter_warmup=warmup,
            show_progress=False,
            show_console=False,
        )
    return fit


def main(draws=1000, warmup=500):
    t0 = time.time()
    print("Loading raw data...")
    saber    = pd.read_csv(RAW_DIR / "npb_sabermetrics_2015_2025.csv",    encoding="utf-8-sig")
    pitchers = pd.read_csv(RAW_DIR / "npb_pitchers_2015_2025.csv",        encoding="utf-8-sig")
    saber    = saber.dropna(subset=["wOBA"])
    bday_df  = load_birthday_df()

    print(f"  Sabermetrics: {len(saber):,} player-years")
    print(f"  Pitchers:     {len(pitchers):,} player-years")
    print(f"  Birthdays:    {len(bday_df):,} players")

    # ── Build datasets ────────────────────────────────────────────────────────
    print("\nBuilding training data (2018-2021)...")
    train_h, train_p, _ = build_dataset(saber, pitchers, TRAIN_YEARS, bday_df)
    print(f"  Hitters train:  {len(train_h):3d}")
    print(f"  Pitchers train: {len(train_p):3d}")

    print("Building backtest data (2022-2025)...")
    test_h, test_p, _ = build_dataset(saber, pitchers, BACKTEST_YEARS, bday_df)
    print(f"  Hitters test:   {len(test_h):3d}")
    print(f"  Pitchers test:  {len(test_p):3d}")

    if len(train_h) == 0 or len(test_h) == 0:
        print("ERROR: insufficient data")
        return

    _log_elapsed("data loading + build_dataset", t0)

    # ── Standardize ───────────────────────────────────────────────────────────
    feat_cols_h = ["K_pct", "BB_pct", "BABIP", "age_from_peak"]
    feat_cols_p = ["K_pct", "BB_pct", "K_per_9", "BB_per_9", "age_from_peak"]

    train_z_h, test_z_h, h_means, h_stds = standardize_features(train_h, test_h, feat_cols_h)
    train_z_p, test_z_p, p_means, p_stds = standardize_features(train_p, test_p, feat_cols_p)

    # ── Stan: Hitters ─────────────────────────────────────────────────────────
    print("\nRunning Stan model — hitters...")
    stan_data_h = {
        "N":                  len(train_h),
        "marcel_woba":        train_h["marcel_woba"].tolist(),
        "z_K":                train_z_h[:, 0].tolist(),
        "z_BB":               train_z_h[:, 1].tolist(),
        "z_babip":            train_z_h[:, 2].tolist(),
        "z_age":              train_z_h[:, 3].tolist(),
        "actual_woba":        train_h["actual_woba"].tolist(),
        "N_pred":             len(test_h),
        "marcel_woba_pred":   test_h["marcel_woba"].tolist(),
        "z_K_pred":           test_z_h[:, 0].tolist(),
        "z_BB_pred":          test_z_h[:, 1].tolist(),
        "z_babip_pred":       test_z_h[:, 2].tolist(),
        "z_age_pred":         test_z_h[:, 3].tolist(),
    }
    fit_h = run_stan_model(ROOT / "models" / "hitter_jpn.stan", stan_data_h, draws, warmup)
    _log_elapsed("hitter Stan compile + sampling", t0)

    # posterior means of test predictions
    stan_pred_h = fit_h.stan_variable("stan_pred").mean(axis=0)
    delta_K_h     = float(fit_h.stan_variable("delta_K").mean())
    delta_BB_h    = float(fit_h.stan_variable("delta_BB").mean())
    delta_BABIP_h = float(fit_h.stan_variable("delta_BABIP").mean())
    delta_age_h   = float(fit_h.stan_variable("delta_age").mean())
    print(f"  delta_K={delta_K_h:+.4f}  delta_BB={delta_BB_h:+.4f}  "
          f"delta_BABIP={delta_BABIP_h:+.4f}  delta_age={delta_age_h:+.4f}")

    test_h = test_h.copy()
    test_h["stan_woba"] = stan_pred_h

    mae_marcel_h = float((test_h["actual_woba"] - test_h["marcel_woba"]).abs().mean())
    mae_stan_h   = float((test_h["actual_woba"] - test_h["stan_woba"]).abs().mean())
    print(f"  Hitter wOBA MAE — Marcel: {mae_marcel_h:.4f}  Stan: {mae_stan_h:.4f}"
          f"  Δ: {mae_stan_h - mae_marcel_h:+.4f}")

    # ── Stan: Pitchers ────────────────────────────────────────────────────────
    print("\nRunning Stan model — pitchers...")
    stan_data_p = {
        "N":               len(train_p),
        "marcel_era":      train_p["marcel_era"].tolist(),
        "z_K":             train_z_p[:, 0].tolist(),
        "z_BB":            train_z_p[:, 1].tolist(),
        "z_K9":            train_z_p[:, 2].tolist(),
        "z_BB9":           train_z_p[:, 3].tolist(),
        "z_age":           train_z_p[:, 4].tolist(),
        "actual_era":      train_p["actual_era"].tolist(),
        "N_pred":          len(test_p),
        "marcel_era_pred": test_p["marcel_era"].tolist(),
        "z_K_pred":        test_z_p[:, 0].tolist(),
        "z_BB_pred":       test_z_p[:, 1].tolist(),
        "z_K9_pred":       test_z_p[:, 2].tolist(),
        "z_BB9_pred":      test_z_p[:, 3].tolist(),
        "z_age_pred":      test_z_p[:, 4].tolist(),
    }
    fit_p = run_stan_model(ROOT / "models" / "pitcher_jpn.stan", stan_data_p, draws, warmup)
    _log_elapsed("pitcher Stan compile + sampling", t0)

    stan_pred_p = fit_p.stan_variable("stan_pred").mean(axis=0)
    delta_K_p    = float(fit_p.stan_variable("delta_K").mean())
    delta_BB_p   = float(fit_p.stan_variable("delta_BB").mean())
    delta_K9_p   = float(fit_p.stan_variable("delta_K9").mean())
    delta_BB9_p  = float(fit_p.stan_variable("delta_BB9").mean())
    delta_age_p  = float(fit_p.stan_variable("delta_age").mean())
    print(f"  delta_K={delta_K_p:+.4f}  delta_BB={delta_BB_p:+.4f}  "
          f"delta_K9={delta_K9_p:+.4f}  delta_BB9={delta_BB9_p:+.4f}  "
          f"delta_age={delta_age_p:+.4f}")

    test_p = test_p.copy()
    test_p["stan_era"] = stan_pred_p

    mae_marcel_p = float((test_p["actual_era"] - test_p["marcel_era"]).abs().mean())
    mae_stan_p   = float((test_p["actual_era"] - test_p["stan_era"]).abs().mean())
    print(f"  Pitcher ERA  MAE — Marcel: {mae_marcel_p:.4f}  Stan: {mae_stan_p:.4f}"
          f"  Δ: {mae_stan_p - mae_marcel_p:+.4f}")

    # ── Save predictions ──────────────────────────────────────────────────────
    cols_h = ["year", "player", "team", "marcel_woba", "stan_woba", "actual_woba", "actual_PA"]
    cols_p = ["year", "player", "team", "marcel_era", "stan_era", "actual_era", "actual_IP"]

    test_h[cols_h].to_csv(MODEL_DIR / "jpn_hitter_predictions.csv",
                          index=False, encoding="utf-8-sig")
    test_p[cols_p].to_csv(MODEL_DIR / "jpn_pitcher_predictions.csv",
                          index=False, encoding="utf-8-sig")

    comparison = {
        "hitter": {
            "mae_marcel": round(mae_marcel_h, 4),
            "mae_stan":   round(mae_stan_h, 4),
            "delta_mae":  round(mae_stan_h - mae_marcel_h, 4),
            "delta_K":     round(delta_K_h, 4),
            "delta_BB":    round(delta_BB_h, 4),
            "delta_BABIP": round(delta_BABIP_h, 4),
            "delta_age":   round(delta_age_h, 4),
            "n_test":      len(test_h),
            "feature_means": h_means,
            "feature_stds":  h_stds,
        },
        "pitcher": {
            "mae_marcel": round(mae_marcel_p, 4),
            "mae_stan":   round(mae_stan_p, 4),
            "delta_mae":  round(mae_stan_p - mae_marcel_p, 4),
            "delta_K":    round(delta_K_p, 4),
            "delta_BB":   round(delta_BB_p, 4),
            "delta_K9":   round(delta_K9_p, 4),
            "delta_BB9":  round(delta_BB9_p, 4),
            "delta_age":  round(delta_age_p, 4),
            "n_test":     len(test_p),
            "feature_means": p_means,
            "feature_stds":  p_stds,
        },
    }
    (MODEL_DIR / "jpn_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log_elapsed("total", t0)
    print(f"\nSaved -> {MODEL_DIR / 'jpn_hitter_predictions.csv'}")
    print(f"Saved -> {MODEL_DIR / 'jpn_pitcher_predictions.csv'}")
    print(f"Saved -> {MODEL_DIR / 'jpn_comparison.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws",  type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=500)
    args = parser.parse_args()
    main(args.draws, args.warmup)
