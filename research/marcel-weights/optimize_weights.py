"""
Marcel法の包括的パラメータ最適化

以下を全て同時に探索する:
1. 年別重み w0/w1/w2（直近3年）
2. 回帰強度 REGRESSION_PA（打者）/ REGRESSION_IP（投手）
3. 打者: OPS / OBP / SLG / AVG で評価
4. 投手: ERA / WHIP で評価
5. 2020年（コロナ短縮）あり/なし両方
6. ブートストラップで統計的有意性を確認

データ: yasumorishima/npb-prediction（NPB 2015-2025）
"""

import os
import warnings
from datetime import date
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---- データURL ----
BASE = "https://raw.githubusercontent.com/yasumorishima/npb-prediction/main/data/raw"
HITTER_URL  = f"{BASE}/npb_hitters_2015_2025.csv"
PITCHER_URL = f"{BASE}/npb_pitchers_2015_2025.csv"
BIRTHDAY_URL = f"{BASE}/npb_player_birthdays.csv"

# ---- 探索パラメータ ----
W0_VALS  = range(3, 9)   # 最新年: 3〜8
W1_VALS  = range(1, 6)   # 1年前:  1〜5
W2_VALS  = range(1, 5)   # 2年前:  1〜4
# 計 6×5×4 = 120通り

REG_PA_VALS = [600, 800, 1000, 1200, 1500, 2000]   # 打者 回帰強度
REG_IP_VALS = [200, 300, 400, 600, 800]             # 投手 回帰強度（IP換算）
# 打者合計: 120×6 = 720通り / 投手: 120×5 = 600通り

EVAL_YEARS_FULL = list(range(2019, 2026))           # 2019-2025（7年）
EVAL_YEARS_NO20 = [y for y in EVAL_YEARS_FULL if y != 2020]  # 2020除外

MIN_PA_PROJ  = 100   # 予測PAの下限（フィルタ）
MIN_PA_EVAL  = 200   # 実績PAの下限（評価対象）
MIN_IP_PROJ  = 30    # 予測IPの下限
MIN_IP_EVAL  = 60    # 実績IPの下限（評価対象）

PEAK_AGE   = 29
AGE_FACTOR = 0.003

N_BOOTSTRAP = 300   # ブートストラップ回数


# ================================================================
# データ読み込み
# ================================================================

def load_hitters():
    df = pd.read_csv(HITTER_URL)
    for col in ["AVG", "OBP", "SLG", "OPS", "PA"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_pitchers():
    df = pd.read_csv(PITCHER_URL)
    for col in ["ERA", "WHIP", "IP", "ER", "HA", "BB"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["IP_num"] = df["IP"].apply(_parse_ip)
    return df


def load_birthdays():
    try:
        b = pd.read_csv(BIRTHDAY_URL)
        b["birthday"] = pd.to_datetime(b["birthday"], errors="coerce")
        return dict(zip(b["player"], b["birthday"]))
    except Exception:
        return {}


def _parse_ip(v) -> float:
    try:
        s = str(v)
        if "." in s:
            w, f = s.split(".")
            return int(w) + int(f) / 3.0
        return float(s)
    except Exception:
        return 0.0


# ================================================================
# ヘルパー
# ================================================================

def calc_age(birthday, target_year: int) -> float:
    if pd.isna(birthday):
        return np.nan
    op = date(target_year, 4, 1)
    age = op.year - birthday.year
    if (op.month, op.day) < (birthday.month, birthday.day):
        age -= 1
    return float(age)


def age_adj(age: float) -> float:
    return 0.0 if np.isnan(age) else (PEAK_AGE - age) * AGE_FACTOR


# ================================================================
# Marcel 計算 ― 打者
# ================================================================

BATTER_RATE_COLS = ["AVG", "OBP", "SLG", "OPS"]


def _league_avg_hitter(df, years):
    lg = {}
    for y in years:
        s = df[df["year"] == y]
        total_pa = s["PA"].sum()
        if total_pa > 0:
            lg[y] = {c: (s[c] * s["PA"]).sum() / total_pa for c in BATTER_RATE_COLS}
    return lg


def compute_marcel_hitter(df, target_year, weights, reg_pa, birthday_dict):
    yrs = [target_year - 1, target_year - 2, target_year - 3]
    lg = _league_avg_hitter(df, yrs)
    if not lg:
        return pd.DataFrame()
    lg_ref = lg.get(yrs[0], lg[max(lg)])

    past = df[df["year"].isin(yrs)]
    rows = []
    for player, pdata in past.groupby("player"):
        total_w = w_pa = 0.0
        w_rates = {c: 0.0 for c in BATTER_RATE_COLS}

        for i, y in enumerate(yrs):
            w = weights[i]
            s = pdata[pdata["year"] == y]
            if len(s) == 0:
                continue
            row = s.iloc[0]
            pa = row["PA"]
            if pd.isna(pa) or pa == 0:
                continue
            total_w += w
            w_pa += pa * w
            for c in BATTER_RATE_COLS:
                w_rates[c] += row[c] * pa * w

        if total_w == 0 or w_pa == 0:
            continue

        avg_pa = w_pa / total_w
        raw = {c: w_rates[c] / w_pa for c in BATTER_RATE_COLS}

        # 平均回帰
        proj = {
            f"proj_{c}": (raw[c] * w_pa + lg_ref[c] * reg_pa) / (w_pa + reg_pa)
            for c in BATTER_RATE_COLS
        }

        # 年齢調整
        bd = birthday_dict.get(player)
        age = calc_age(bd, target_year) if bd is not None else np.nan
        adj = age_adj(age)
        for c in BATTER_RATE_COLS:
            proj[f"proj_{c}"] += adj

        proj["player"] = player
        proj["proj_PA"] = avg_pa
        rows.append(proj)

    return pd.DataFrame(rows)


# ================================================================
# Marcel 計算 ― 投手
# ================================================================

def _league_avg_pitcher(df, years):
    lg = {}
    for y in years:
        s = df[df["year"] == y]
        total_ip = s["IP_num"].sum()
        if total_ip > 0:
            lg[y] = {
                "ERA":  s["ER"].sum() * 9 / total_ip,
                "WHIP": (s["HA"] + s["BB"]).sum() / total_ip,
            }
    return lg


def compute_marcel_pitcher(df, target_year, weights, reg_ip, birthday_dict):
    yrs = [target_year - 1, target_year - 2, target_year - 3]
    lg = _league_avg_pitcher(df, yrs)
    if not lg:
        return pd.DataFrame()
    lg_ref = lg.get(yrs[0], lg[max(lg)])

    past = df[df["year"].isin(yrs)]
    rows = []
    for player, pdata in past.groupby("player"):
        total_w = w_ip = w_er = w_ha_bb = 0.0

        for i, y in enumerate(yrs):
            w = weights[i]
            s = pdata[pdata["year"] == y]
            if len(s) == 0:
                continue
            row = s.iloc[0]
            ip = row["IP_num"]
            if pd.isna(ip) or ip == 0:
                continue
            total_w += w
            w_ip    += ip * w
            w_er    += row["ER"] * w
            w_ha_bb += (row["HA"] + row["BB"]) * w

        if total_w == 0 or w_ip == 0:
            continue

        avg_ip   = w_ip / total_w
        raw_era  = w_er * 9 / w_ip
        raw_whip = w_ha_bb / w_ip

        proj_era  = (raw_era  * w_ip + lg_ref["ERA"]  * reg_ip) / (w_ip + reg_ip)
        proj_whip = (raw_whip * w_ip + lg_ref["WHIP"] * reg_ip) / (w_ip + reg_ip)

        bd = birthday_dict.get(player)
        age = calc_age(bd, target_year) if bd is not None else np.nan
        adj = age_adj(age)
        # 投手: ERA/WHIP は低いほど良い → ピーク前は下げる
        proj_era  -= adj * (proj_era  / 0.3)
        proj_whip -= adj * (proj_whip / 0.3)

        rows.append({
            "player":   player,
            "proj_ERA": proj_era,
            "proj_WHIP": proj_whip,
            "proj_IP":  avg_ip,
        })

    return pd.DataFrame(rows)


# ================================================================
# MAE計算
# ================================================================

def mae_hitter(proj_df, actual_df, col):
    if len(proj_df) == 0:
        return None
    p = proj_df[proj_df["proj_PA"] >= MIN_PA_PROJ][["player", f"proj_{col}"]].copy()
    a = actual_df[actual_df["PA"] >= MIN_PA_EVAL][["player", col]].rename(
        columns={col: f"actual_{col}"}
    )
    m = p.merge(a, on="player")
    if len(m) < 10:
        return None
    return float((m[f"proj_{col}"] - m[f"actual_{col}"]).abs().mean())


def mae_pitcher(proj_df, actual_df, col):
    if len(proj_df) == 0:
        return None
    p = proj_df[proj_df["proj_IP"] >= MIN_IP_PROJ][["player", f"proj_{col}"]].copy()
    a = actual_df[actual_df["IP_num"] >= MIN_IP_EVAL][["player", col]].rename(
        columns={col: f"actual_{col}"}
    )
    m = p.merge(a, on="player")
    if len(m) < 5:
        return None
    return float((m[f"proj_{col}"] - m[f"actual_{col}"]).abs().mean())


# ================================================================
# グリッドサーチ ― 打者
# ================================================================

def grid_search_hitters(df, birthday_dict, eval_years, label):
    combos = list(product(W0_VALS, W1_VALS, W2_VALS, REG_PA_VALS))
    print(f"\n[打者 {label}] 組み合わせ: {len(combos)}, 評価年: {eval_years}")

    results = []
    for i, (w0, w1, w2, reg_pa) in enumerate(combos):
        weights = [w0, w1, w2]
        year_maes = {c: [] for c in BATTER_RATE_COLS}

        for target_year in eval_years:
            if target_year not in df["year"].values:
                continue
            proj = compute_marcel_hitter(df, target_year, weights, reg_pa, birthday_dict)
            actual = df[df["year"] == target_year]
            for c in BATTER_RATE_COLS:
                m = mae_hitter(proj, actual, c)
                if m is not None:
                    year_maes[c].append(m)

        if not year_maes["OPS"]:
            continue

        row = {
            "weights_str": f"{w0}/{w1}/{w2}",
            "w0": w0, "w1": w1, "w2": w2,
            "reg_pa": reg_pa,
        }
        for c in BATTER_RATE_COLS:
            row[f"mae_{c}"] = round(np.mean(year_maes[c]), 5) if year_maes[c] else np.nan
        results.append(row)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(combos)} 完了")

    print(f"  {len(combos)}/{len(combos)} 完了")
    return pd.DataFrame(results)


# ================================================================
# グリッドサーチ ― 投手
# ================================================================

def grid_search_pitchers(df, birthday_dict, eval_years, label):
    combos = list(product(W0_VALS, W1_VALS, W2_VALS, REG_IP_VALS))
    print(f"\n[投手 {label}] 組み合わせ: {len(combos)}, 評価年: {eval_years}")

    results = []
    for i, (w0, w1, w2, reg_ip) in enumerate(combos):
        weights = [w0, w1, w2]
        era_maes, whip_maes = [], []

        for target_year in eval_years:
            if target_year not in df["year"].values:
                continue
            proj = compute_marcel_pitcher(df, target_year, weights, reg_ip, birthday_dict)
            actual = df[df["year"] == target_year]
            m_era = mae_pitcher(proj, actual, "ERA")
            m_whip = mae_pitcher(proj, actual, "WHIP")
            if m_era is not None:
                era_maes.append(m_era)
            if m_whip is not None:
                whip_maes.append(m_whip)

        if not era_maes:
            continue

        results.append({
            "weights_str": f"{w0}/{w1}/{w2}",
            "w0": w0, "w1": w1, "w2": w2,
            "reg_ip": reg_ip,
            "mae_ERA":  round(np.mean(era_maes), 5) if era_maes else np.nan,
            "mae_WHIP": round(np.mean(whip_maes), 5) if whip_maes else np.nan,
        })

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(combos)} 完了")

    print(f"  {len(combos)}/{len(combos)} 完了")
    return pd.DataFrame(results)


# ================================================================
# ブートストラップ（有意性検定）
# ================================================================

def bootstrap_comparison(df_h, df_p, birthday_dict, best_h, default_h, best_p, default_p):
    """
    best vs default (5/4/3, reg=1200/600) の差をブートストラップで検定
    """
    print("\n[ブートストラップ] 300回リサンプリング中...")
    rng = np.random.default_rng(42)
    players = df_h["player"].unique()

    diffs_ops, diffs_era = [], []

    for _ in range(N_BOOTSTRAP):
        sampled = rng.choice(players, size=len(players), replace=True)
        df_bs = pd.concat([df_h[df_h["player"] == p] for p in sampled], ignore_index=True)
        df_bs_p = pd.concat([df_p[df_p["player"] == p] for p in sampled], ignore_index=True)

        mae_best_ops, mae_def_ops = [], []
        mae_best_era, mae_def_era = [], []

        for target_year in EVAL_YEARS_FULL:
            if target_year not in df_bs["year"].values:
                continue

            # 打者
            actual_h = df_h[df_h["year"] == target_year]  # 実績は元データ
            proj_best = compute_marcel_hitter(
                df_bs, target_year,
                [best_h["w0"], best_h["w1"], best_h["w2"]], best_h["reg_pa"],
                birthday_dict
            )
            proj_def = compute_marcel_hitter(
                df_bs, target_year,
                [5, 4, 3], default_h["reg_pa"],
                birthday_dict
            )
            m1 = mae_hitter(proj_best, actual_h, "OPS")
            m2 = mae_hitter(proj_def, actual_h, "OPS")
            if m1 and m2:
                mae_best_ops.append(m1)
                mae_def_ops.append(m2)

            # 投手
            actual_p = df_p[df_p["year"] == target_year]
            proj_best_p = compute_marcel_pitcher(
                df_bs_p, target_year,
                [best_p["w0"], best_p["w1"], best_p["w2"]], best_p["reg_ip"],
                birthday_dict
            )
            proj_def_p = compute_marcel_pitcher(
                df_bs_p, target_year,
                [5, 4, 3], default_p["reg_ip"],
                birthday_dict
            )
            m3 = mae_pitcher(proj_best_p, actual_p, "ERA")
            m4 = mae_pitcher(proj_def_p, actual_p, "ERA")
            if m3 and m4:
                mae_best_era.append(m3)
                mae_def_era.append(m4)

        if mae_best_ops:
            diffs_ops.append(np.mean(mae_def_ops) - np.mean(mae_best_ops))
        if mae_best_era:
            diffs_era.append(np.mean(mae_def_era) - np.mean(mae_best_era))

    return np.array(diffs_ops), np.array(diffs_era)


# ================================================================
# メイン
# ================================================================

def main():
    print("=" * 60)
    print("Marcel法 包括的パラメータ最適化 (NPB 2015-2025)")
    print("=" * 60)

    df_h = load_hitters()
    df_p = load_pitchers()
    bdays = load_birthdays()
    print(f"打者データ: {len(df_h)} 行 / 投手データ: {len(df_p)} 行 / 誕生日: {len(bdays)} 選手")

    os.makedirs("results", exist_ok=True)

    # ---- 打者グリッドサーチ（2020あり/なし） ----
    df_h_full = grid_search_hitters(df_h, bdays, EVAL_YEARS_FULL,  "2020込み")
    df_h_no20 = grid_search_hitters(df_h, bdays, EVAL_YEARS_NO20, "2020除外")

    df_h_full_sorted = df_h_full.sort_values("mae_OPS").reset_index(drop=True)
    df_h_no20_sorted = df_h_no20.sort_values("mae_OPS").reset_index(drop=True)

    df_h_full_sorted.to_csv("results/hitter_grid_full.csv", index=False, encoding="utf-8-sig")
    df_h_no20_sorted.to_csv("results/hitter_grid_no2020.csv", index=False, encoding="utf-8-sig")

    # ---- 投手グリッドサーチ（2020あり/なし） ----
    df_p_full = grid_search_pitchers(df_p, bdays, EVAL_YEARS_FULL,  "2020込み")
    df_p_no20 = grid_search_pitchers(df_p, bdays, EVAL_YEARS_NO20, "2020除外")

    df_p_full_sorted = df_p_full.sort_values("mae_ERA").reset_index(drop=True)
    df_p_no20_sorted = df_p_no20.sort_values("mae_ERA").reset_index(drop=True)

    df_p_full_sorted.to_csv("results/pitcher_grid_full.csv", index=False, encoding="utf-8-sig")
    df_p_no20_sorted.to_csv("results/pitcher_grid_no2020.csv", index=False, encoding="utf-8-sig")

    # ---- サマリー表示 ----
    print("\n" + "=" * 60)
    print("結果サマリー")
    print("=" * 60)

    def show_summary(df_sorted, label, primary_col, default_weights="5/4/3"):
        best = df_sorted.iloc[0]
        default_rows = df_sorted[df_sorted["weights_str"] == default_weights]
        reg_col = "reg_pa" if "reg_pa" in df_sorted.columns else "reg_ip"
        default_reg = 1200 if reg_col == "reg_pa" else 600

        print(f"\n【{label}】 primary: {primary_col}")
        print(f"  最適: {best['weights_str']}  reg={int(best[reg_col])}  MAE={best[primary_col]:.5f}")

        if len(default_rows) > 0:
            # Marcel原典と同じ重み+reg
            def_row = default_rows[default_rows[reg_col] == default_reg]
            if len(def_row) == 0:
                def_row = default_rows.iloc[[0]]
            d = def_row.iloc[0]
            rank = df_sorted[df_sorted["weights_str"] == default_weights].index[0] + 1
            improve = (d[primary_col] - best[primary_col]) / d[primary_col] * 100
            print(f"  従来値(5/4/3, reg={default_reg}): MAE={d[primary_col]:.5f}  順位={rank}/{len(df_sorted)}")
            print(f"  改善率: {improve:.2f}%")

        cols = ["weights_str", reg_col, primary_col]
        print(df_sorted[cols].head(10).to_string(index=False))

    show_summary(df_h_full_sorted, "打者(2020込み)",  "mae_OPS")
    show_summary(df_h_no20_sorted, "打者(2020除外)", "mae_OPS")
    show_summary(df_p_full_sorted, "投手(2020込み)",  "mae_ERA")
    show_summary(df_p_no20_sorted, "投手(2020除外)", "mae_ERA")

    # ---- OBP / SLG / AVG でも best は変わるか ----
    print("\n【打者 指標別 最適重み（2020込み）】")
    for c in BATTER_RATE_COLS:
        best = df_h_full_sorted.sort_values(f"mae_{c}").iloc[0]
        reg_col = "reg_pa"
        print(f"  {c}: {best['weights_str']}  reg={int(best[reg_col])}  MAE={best[f'mae_{c}']:.5f}")

    # ---- ブートストラップ ----
    best_h_row  = df_h_full_sorted.iloc[0]
    best_p_row  = df_p_full_sorted.iloc[0]

    # default: 5/4/3 + 従来値 reg
    def_h_rows = df_h_full_sorted[df_h_full_sorted["weights_str"] == "5/4/3"]
    def_p_rows = df_p_full_sorted[df_p_full_sorted["weights_str"] == "5/4/3"]
    def_h_row  = def_h_rows[def_h_rows["reg_pa"] == 1200].iloc[0] if len(def_h_rows) > 0 else def_h_rows.iloc[0]
    def_p_row  = def_p_rows[def_p_rows["reg_ip"] == 600].iloc[0]  if len(def_p_rows) > 0 else def_p_rows.iloc[0]

    diffs_ops, diffs_era = bootstrap_comparison(
        df_h, df_p, bdays,
        best_h_row, def_h_row,
        best_p_row, def_p_row,
    )

    print("\n【ブートストラップ結果】（best - default: 正=bestが優れている）")
    if len(diffs_ops) > 0:
        ci_lo, ci_hi = np.percentile(diffs_ops, [2.5, 97.5])
        p_val = (diffs_ops <= 0).mean()
        print(f"  打者OPS: mean={diffs_ops.mean():.5f}  95%CI=[{ci_lo:.5f}, {ci_hi:.5f}]  p(best<=default)={p_val:.3f}")
    if len(diffs_era) > 0:
        ci_lo, ci_hi = np.percentile(diffs_era, [2.5, 97.5])
        p_val = (diffs_era <= 0).mean()
        print(f"  投手ERA: mean={diffs_era.mean():.5f}  95%CI=[{ci_lo:.5f}, {ci_hi:.5f}]  p(best<=default)={p_val:.3f}")

    # bootstrap結果を保存（長さが異なる場合はNaNでパディング）
    n = max(len(diffs_ops), len(diffs_era), 1)
    arr_ops = np.full(n, np.nan)
    arr_era = np.full(n, np.nan)
    if len(diffs_ops):
        arr_ops[:len(diffs_ops)] = diffs_ops
    if len(diffs_era):
        arr_era[:len(diffs_era)] = diffs_era
    pd.DataFrame({
        "diff_ops": arr_ops,
        "diff_era": arr_era,
    }).to_csv("results/bootstrap.csv", index=False, encoding="utf-8-sig")

    print("\n保存先: results/hitter_grid_full.csv / hitter_grid_no2020.csv")
    print("       results/pitcher_grid_full.csv / pitcher_grid_no2020.csv")
    print("       results/bootstrap.csv")


if __name__ == "__main__":
    main()
