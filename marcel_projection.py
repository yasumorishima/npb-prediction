"""
Marcel法によるNPB選手成績予測

Marcel法（Tom Tangoが考案）:
1. 過去3年の成績を 5/4/3 で加重平均
2. リーグ平均への回帰（打者: 1200PA、投手: 600IP相当）
3. 年齢調整（ピーク=29歳、1年あたり±0.003程度）

参考: https://www.tangotiger.net/marcel/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "projections"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Marcel法の重み（直近3年: 5/4/3）
WEIGHTS = {0: 5, 1: 4, 2: 3}  # 0=直近年, 1=1年前, 2=2年前

# 平均回帰の強さ（PA数）: この値が大きいほどリーグ平均に引っ張られる
REGRESSION_PA = 1200  # 打者
REGRESSION_IP = 600   # 投手（投球回ベース）

# 年齢調整: ピーク年齢と1年あたりの変化率
PEAK_AGE = 29
AGE_FACTOR = 0.003  # OPS等の割合指標に対する1歳あたりの変化


def load_birthdays() -> dict:
    """生年月日データを読み込み、選手名→生年月日の辞書を返す"""
    path = RAW_DIR / "npb_player_birthdays.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df["birthday"] = pd.to_datetime(df["birthday"], errors="coerce")
    df = df.dropna(subset=["birthday"])
    return dict(zip(df["player"], df["birthday"]))


def calc_age(birthday, target_year: int) -> float:
    """target_year の開幕時点（4/1）での年齢を計算"""
    if pd.isna(birthday):
        return np.nan
    opening_day = date(target_year, 4, 1)
    age = opening_day.year - birthday.year
    if (opening_day.month, opening_day.day) < (birthday.month, birthday.day):
        age -= 1
    return age


def age_adjustment(age: float, peak: int = PEAK_AGE, factor: float = AGE_FACTOR) -> float:
    """年齢調整係数を返す（ピーク年齢からの距離に応じて）"""
    if np.isnan(age):
        return 0.0
    return (peak - age) * factor


def load_hitters() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "npb_hitters_2015_2025.csv")
    # 数値型に変換（RC27, XR27がobjectの場合がある）
    for col in ["RC27", "XR27"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_pitchers() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "npb_pitchers_2015_2025.csv")
    # 数値型に変換
    for col in ["ERA", "WHIP", "DIPS", "IP", "BB", "HBP", "HRA", "BF"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def calc_league_avg(df: pd.DataFrame, year: int, rate_cols: list, weight_col: str) -> dict:
    """指定年のリーグ平均レート指標を計算"""
    season = df[df["year"] == year]
    avgs = {}
    total_weight = season[weight_col].sum()
    if total_weight == 0:
        return {col: 0.0 for col in rate_cols}
    for col in rate_cols:
        avgs[col] = (season[col] * season[weight_col]).sum() / total_weight
    return avgs


def marcel_hitter(df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """
    Marcel法で打者成績を予測

    予測対象の指標（レート系）: AVG, OBP, SLG, OPS
    予測対象の指標（カウント系）: HR, RBI, SB, BB, SO（PA比率で予測→PA掛け算）
    """
    # 生年月日データの読み込み
    birthdays = load_birthdays()

    # 過去3年のデータ
    years = [target_year - 1, target_year - 2, target_year - 3]
    available_years = sorted(df["year"].unique())

    # 各年のリーグ平均
    rate_cols = ["AVG", "OBP", "SLG", "OPS"]
    count_cols = ["HR", "RBI", "SB", "BB", "SO", "H"]
    lg_avgs = {}
    for y in years:
        if y in available_years:
            lg_avgs[y] = calc_league_avg(df, y, rate_cols, "PA")
            # カウント系もPA比率で平均を出す
            season = df[df["year"] == y]
            total_pa = season["PA"].sum()
            if total_pa > 0:
                for col in count_cols:
                    lg_avgs[y][f"{col}_rate"] = season[col].sum() / total_pa

    # 対象選手: 過去3年のいずれかにデータがある選手
    past_data = df[df["year"].isin(years)]
    players = past_data["player"].unique()

    results = []
    for player in players:
        player_data = past_data[past_data["player"] == player]

        # 加重平均の計算
        total_weight = 0
        weighted_rates = {col: 0.0 for col in rate_cols}
        weighted_count_rates = {col: 0.0 for col in count_cols}
        weighted_pa = 0
        data_years = 0

        for i, y in enumerate(years):
            w = WEIGHTS[i]
            season = player_data[player_data["year"] == y]
            if len(season) == 0:
                continue
            row = season.iloc[0]
            pa = row["PA"]
            if pa == 0:
                continue

            data_years += 1
            total_weight += w
            weighted_pa += pa * w

            for col in rate_cols:
                weighted_rates[col] += row[col] * pa * w
            for col in count_cols:
                weighted_count_rates[col] += (row[col] / pa) * pa * w

        if total_weight == 0 or weighted_pa == 0:
            continue

        # 加重平均PA
        avg_pa = weighted_pa / total_weight

        # PA予測: 直近年PAの加重平均を使用
        proj_pa = avg_pa

        # 加重平均レート
        for col in rate_cols:
            weighted_rates[col] /= weighted_pa
        for col in count_cols:
            weighted_count_rates[col] /= weighted_pa

        # リーグ平均（直近年を使用）
        lg_year = years[0]
        if lg_year not in lg_avgs:
            lg_year = max(lg_avgs.keys()) if lg_avgs else None
        if lg_year is None:
            continue
        lg = lg_avgs[lg_year]

        # 平均回帰
        # proj = (weighted_rate * weighted_pa + lg_avg * REGRESSION_PA) / (weighted_pa + REGRESSION_PA)
        proj = {}
        for col in rate_cols:
            proj[col] = (
                weighted_rates[col] * weighted_pa + lg.get(col, 0) * REGRESSION_PA
            ) / (weighted_pa + REGRESSION_PA)

        for col in count_cols:
            lg_rate = lg.get(f"{col}_rate", 0)
            proj_rate = (
                weighted_count_rates[col] * weighted_pa + lg_rate * REGRESSION_PA
            ) / (weighted_pa + REGRESSION_PA)
            proj[col] = proj_rate * proj_pa  # レート→カウント変換

        # 年齢調整
        birthday = birthdays.get(player)
        age = calc_age(birthday, target_year) if birthday is not None else np.nan
        adj = age_adjustment(age)
        for col in rate_cols:
            proj[col] += adj  # ピーク前: +, ピーク後: -

        proj["player"] = player
        proj["team"] = player_data.iloc[-1]["team"]  # 最新チーム
        proj["PA"] = round(proj_pa)
        proj["target_year"] = target_year
        proj["age"] = age
        proj["data_years"] = data_years

        results.append(proj)

    result_df = pd.DataFrame(results)

    # 小数点整形
    if len(result_df) > 0:
        for col in rate_cols:
            result_df[col] = result_df[col].round(3)
        for col in count_cols:
            result_df[col] = result_df[col].round(1)

    return result_df


def marcel_pitcher(df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """
    Marcel法で投手成績を予測

    予測対象（レート系）: ERA, WHIP, DIPS
    予測対象（カウント系）: W, L, SV, SO（IP比率で予測）
    """
    birthdays = load_birthdays()
    years = [target_year - 1, target_year - 2, target_year - 3]
    available_years = sorted(df["year"].unique())

    rate_cols = ["ERA", "WHIP"]
    count_cols = ["W", "L", "SV", "SO", "BB", "HBP", "HRA", "BF"]

    # IP列を数値化（"123.1" → 123.333...）
    df = df.copy()
    df["IP_num"] = df["IP"].apply(lambda x: _parse_ip(x))

    lg_avgs = {}
    for y in years:
        if y in available_years:
            season = df[df["year"] == y]
            total_ip = season["IP_num"].sum()
            if total_ip > 0:
                lg_avgs[y] = {}
                lg_avgs[y]["ERA"] = season["ER"].sum() * 9 / total_ip
                lg_avgs[y]["WHIP"] = (season["HA"].sum() + season["BB"].sum()) / total_ip
                for col in count_cols:
                    lg_avgs[y][f"{col}_rate"] = season[col].sum() / total_ip

    past_data = df[df["year"].isin(years)]
    players = past_data["player"].unique()

    results = []
    for player in players:
        player_data = past_data[past_data["player"] == player]

        total_weight = 0
        weighted_ip = 0
        weighted_er = 0
        weighted_ha_bb = 0
        weighted_count_rates = {col: 0.0 for col in count_cols}
        data_years = 0

        for i, y in enumerate(years):
            w = WEIGHTS[i]
            season = player_data[player_data["year"] == y]
            if len(season) == 0:
                continue
            row = season.iloc[0]
            ip = row["IP_num"]
            if ip == 0:
                continue

            data_years += 1
            total_weight += w
            weighted_ip += ip * w
            weighted_er += row["ER"] * w
            weighted_ha_bb += (row["HA"] + row["BB"]) * w
            for col in count_cols:
                weighted_count_rates[col] += (row[col] / ip) * ip * w

        if total_weight == 0 or weighted_ip == 0:
            continue

        avg_ip = weighted_ip / total_weight
        w_era = weighted_er * 9 / weighted_ip
        w_whip = weighted_ha_bb / weighted_ip

        for col in count_cols:
            weighted_count_rates[col] /= weighted_ip

        lg_year = years[0]
        if lg_year not in lg_avgs:
            lg_year = max(lg_avgs.keys()) if lg_avgs else None
        if lg_year is None:
            continue
        lg = lg_avgs[lg_year]

        # 平均回帰
        proj = {}
        proj["ERA"] = (
            w_era * weighted_ip + lg.get("ERA", 0) * REGRESSION_IP
        ) / (weighted_ip + REGRESSION_IP)
        proj["WHIP"] = (
            w_whip * weighted_ip + lg.get("WHIP", 0) * REGRESSION_IP
        ) / (weighted_ip + REGRESSION_IP)

        for col in count_cols:
            lg_rate = lg.get(f"{col}_rate", 0)
            proj_rate = (
                weighted_count_rates[col] * weighted_ip + lg_rate * REGRESSION_IP
            ) / (weighted_ip + REGRESSION_IP)
            proj[col] = proj_rate * avg_ip

        # 年齢調整（投手: ERA/WHIPは低い方が良いので符号を逆に）
        birthday = birthdays.get(player)
        age = calc_age(birthday, target_year) if birthday is not None else np.nan
        adj = age_adjustment(age)
        proj["ERA"] -= adj * (proj["ERA"] / 0.300)  # ERA scale adjustment
        proj["WHIP"] -= adj * (proj["WHIP"] / 0.300)  # WHIP scale adjustment

        proj["player"] = player
        proj["team"] = player_data.iloc[-1]["team"]
        proj["IP"] = round(avg_ip, 1)
        proj["target_year"] = target_year
        proj["age"] = age
        proj["data_years"] = data_years

        results.append(proj)

    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        result_df["ERA"] = result_df["ERA"].round(2)
        result_df["WHIP"] = result_df["WHIP"].round(2)
        for col in count_cols:
            result_df[col] = result_df[col].round(1)

    return result_df


def _parse_ip(ip_val) -> float:
    """投球回を数値に変換: "123.1" → 123.333..."""
    try:
        s = str(ip_val)
        if "." in s:
            parts = s.split(".")
            whole = int(parts[0])
            frac = int(parts[1])  # 0, 1, 2 (= 0/3, 1/3, 2/3)
            return whole + frac / 3.0
        return float(s)
    except (ValueError, IndexError):
        return 0.0


def evaluate_marcel(df_actual: pd.DataFrame, df_proj: pd.DataFrame,
                    cols: list, merge_on: str = "player") -> pd.DataFrame:
    """予測 vs 実績の比較"""
    merged = df_actual.merge(df_proj, on=merge_on, suffixes=("_actual", "_proj"))
    eval_results = {}
    for col in cols:
        actual_col = f"{col}_actual"
        proj_col = f"{col}_proj"
        if actual_col in merged.columns and proj_col in merged.columns:
            diff = merged[proj_col] - merged[actual_col]
            eval_results[col] = {
                "MAE": diff.abs().mean(),
                "RMSE": np.sqrt((diff ** 2).mean()),
                "n": len(diff),
            }
    return pd.DataFrame(eval_results).T


def main():
    print("=" * 60)
    print("Marcel法 NPB成績予測")
    print("=" * 60)

    df_h = load_hitters()
    df_p = load_pitchers()

    # 2026年の成績を予測（2023-2025のデータから）
    target = 2026
    print(f"\n--- 打者予測 (target: {target}) ---")
    proj_h = marcel_hitter(df_h, target)
    if len(proj_h) > 0:
        # PA >= 200の選手に絞って表示
        top = proj_h[proj_h["PA"] >= 200].sort_values("OPS", ascending=False).head(20)
        print(f"Total projections: {len(proj_h)} players")
        print(f"\nTop 20 by OPS (PA >= 200):")
        print(top[["player", "team", "PA", "AVG", "OBP", "SLG", "OPS", "HR", "RBI"]].to_string(index=False))

        out_path = OUT_DIR / f"marcel_hitters_{target}.csv"
        proj_h.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved: {out_path}")

    print(f"\n--- 投手予測 (target: {target}) ---")
    proj_p = marcel_pitcher(df_p, target)
    if len(proj_p) > 0:
        top = proj_p[proj_p["IP"] >= 50].sort_values("ERA").head(20)
        print(f"Total projections: {len(proj_p)} players")
        print(f"\nTop 20 by ERA (IP >= 50):")
        print(top[["player", "team", "IP", "ERA", "WHIP", "W", "SO"]].to_string(index=False))

        out_path = OUT_DIR / f"marcel_pitchers_{target}.csv"
        proj_p.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved: {out_path}")

    # 予測精度評価: 2024年・2025年を予測して実績と比較
    for eval_target in [2024, 2025]:
        print(f"\n{'=' * 60}")
        print(f"予測精度評価: {eval_target}年を予測 vs 実績")
        print(f"{'=' * 60}")

        proj_h_eval = marcel_hitter(df_h, eval_target)
        actual_h = df_h[df_h["year"] == eval_target]

        qualified_pa = 400
        proj_h_q = proj_h_eval[proj_h_eval["PA"] >= qualified_pa]
        actual_h_q = actual_h[actual_h["PA"] >= qualified_pa]

        if len(proj_h_q) > 0 and len(actual_h_q) > 0:
            eval_h = evaluate_marcel(actual_h_q, proj_h_q, ["AVG", "OBP", "SLG", "OPS"])
            print(f"\n打者（PA >= {qualified_pa}）:")
            print(eval_h.to_string())

        proj_p_eval = marcel_pitcher(df_p, eval_target)
        actual_p = df_p[df_p["year"] == eval_target]
        qualified_ip = 100
        proj_p_q = proj_p_eval[proj_p_eval["IP"] >= qualified_ip]
        actual_p_q = actual_p[actual_p["IP"] >= qualified_ip]

        if len(proj_p_q) > 0 and len(actual_p_q) > 0:
            eval_p = evaluate_marcel(actual_p_q, proj_p_q, ["ERA", "WHIP"])
            print(f"\n投手（IP >= {qualified_ip}）:")
            print(eval_p.to_string())


if __name__ == "__main__":
    main()
