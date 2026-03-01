"""
過去年（2018-2025）のMarcel→ピタゴラス予測勝利数を生成する。
2026年予測と同じロジックを使い、各年の事前予測として計算する。
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from marcel_projection import marcel_hitter, marcel_pitcher

CENTRAL_TEAMS = {"巨人", "阪神", "広島", "DeNA", "ヤクルト", "中日"}
PACIFIC_TEAMS = {"ソフトバンク", "オリックス", "西武", "ロッテ", "日本ハム", "楽天"}
TEAMS = list(CENTRAL_TEAMS | PACIFIC_TEAMS)

WOBA_SCALE = 1.15
PYTH_K = 1.72

# 2020年は120試合、それ以外は143試合
SEASON_GAMES = {2020: 120}


def pyth_wpct(rs, ra, k=PYTH_K):
    if ra == 0:
        return 1.0
    return rs**k / (rs**k + ra**k)


def build_team_projections(target_year, df_h, df_p, df_pyth, df_saber):
    mh = marcel_hitter(df_h, target_year)
    mp = marcel_pitcher(df_p, target_year)

    if mh.empty or mp.empty:
        return pd.DataFrame()

    # チーム割り当てを target_year の実データで上書き（歴史的バックテスト用）
    # Marcel は前年チームを使うため、シーズン前移籍が反映されない問題を修正
    # 同一選手の重複（シーズン途中移籍）は最後のチームを採用
    actual_h = (df_h[df_h["year"] == target_year]
                .drop_duplicates(subset="player", keep="last")
                .set_index("player")["team"])
    actual_p = (df_p[df_p["year"] == target_year]
                .drop_duplicates(subset="player", keep="last")
                .set_index("player")["team"])
    mh = mh.copy()
    mp = mp.copy()
    mh["team"] = mh["player"].map(actual_h).fillna(mh["team"])
    mp["team"] = mp["player"].map(actual_p).fillna(mp["team"])

    # リーグ平均RS/RA: target_year より前の3年分
    ref_years = list(range(target_year - 3, target_year))
    recent_p = df_pyth[df_pyth["year"].isin(ref_years)]
    if recent_p.empty:
        return pd.DataFrame()
    lg_avg_rs = recent_p.groupby("year")["RS"].mean().mean()
    lg_avg_ra = recent_p.groupby("year")["RA"].mean().mean()

    # リーグ基準ERA（Marcel全投手のIP加重平均）
    lg_era = (mp["ERA"] * mp["IP"]).sum() / mp["IP"].sum() if mp["IP"].sum() > 0 else 3.5

    # wRAA計算
    mh = mh.copy()
    if "wOBA" in mh.columns and "wRAA" in mh.columns:
        mh["wRAA_est"] = mh["wRAA"]
    else:
        ref_saber = df_saber[df_saber["year"].isin(ref_years) & (df_saber["PA"] >= 100)].dropna(subset=["wOBA", "OBP", "SLG"])
        X = np.column_stack([ref_saber["OBP"].values, ref_saber["SLG"].values, np.ones(len(ref_saber))])
        coeffs, _, _, _ = np.linalg.lstsq(X, ref_saber["wOBA"].values, rcond=None)
        a_obp, b_slg, intercept_w = coeffs
        lg_woba = df_saber[df_saber["year"].isin(ref_years) & (df_saber["PA"] >= 50)]["wOBA"].mean()
        mh["wOBA_est"] = a_obp * mh["OBP"] + b_slg * mh["SLG"] + intercept_w
        mh["wRAA_est"] = (mh["wOBA_est"] - lg_woba) / WOBA_SCALE * mh["PA"]

    mp = mp.copy()
    mp["era_above_avg"] = mp["ERA"] - lg_era

    rows = []
    for team in TEAMS:
        h = mh[(mh["team"] == team) & (mh["PA"] >= 100)]
        p = mp[(mp["team"] == team) & (mp["IP"] >= 30)]
        rs_raw = lg_avg_rs + (h["wRAA_est"].sum() if not h.empty else 0)
        ra_raw = lg_avg_ra + ((p["era_above_avg"] * p["IP"] / 9.0).sum() if not p.empty else 0)
        league = "CL" if team in CENTRAL_TEAMS else "PL"
        rows.append({"year": target_year, "league": league, "team": team,
                     "rs_raw": rs_raw, "ra_raw": ra_raw})

    df = pd.DataFrame(rows)

    # スケーリング（2026と同ロジック）
    rs_scale = lg_avg_rs / df["rs_raw"].mean()
    ra_scale = lg_avg_ra / df["ra_raw"].mean()
    df["pred_RS"] = df["rs_raw"] * rs_scale
    df["pred_RA"] = df["ra_raw"] * ra_scale
    df["pred_WPCT"] = df.apply(lambda r: pyth_wpct(r["pred_RS"], r["pred_RA"]), axis=1)

    G = SEASON_GAMES.get(target_year, 143)
    df["pred_W"] = (df["pred_WPCT"] * G).round(1)
    df["G"] = G

    return df[["year", "league", "team", "G", "pred_RS", "pred_RA", "pred_WPCT", "pred_W"]]


def main():
    df_h = pd.read_csv("data/raw/npb_hitters_2015_2025.csv")
    df_p = pd.read_csv("data/raw/npb_pitchers_2015_2025.csv")
    df_pyth = pd.read_csv("data/projections/pythagorean_2015_2025.csv")
    df_saber = pd.read_csv("data/projections/npb_sabermetrics_2015_2025.csv")

    all_rows = []
    # 2018から（Marcel は3年分必要なので2015データから計算可能）
    for yr in range(2018, 2026):
        print(f"  {yr}年予測を計算中...", end=" ", flush=True)
        df_yr = build_team_projections(yr, df_h, df_p, df_pyth, df_saber)
        if not df_yr.empty:
            all_rows.append(df_yr)
            print(f"{len(df_yr)}チーム完了")
        else:
            print("スキップ")

    result = pd.concat(all_rows, ignore_index=True)
    out_path = "data/projections/marcel_team_historical.csv"
    result.to_csv(out_path, index=False)
    print(f"\n保存完了: {out_path}  ({len(result)}行)")
    print(result[result["team"].isin(["ヤクルト", "オリックス"])].to_string())


if __name__ == "__main__":
    main()
