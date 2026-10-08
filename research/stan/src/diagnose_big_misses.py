"""Step 12 diagnostic: Big-miss team-years breakdown.

For team-years with |err| > 10W, show:
  1. Year-over-year player analysis (PA/IP drops, wOBA/ERA changes)
  2. Key departures / arrivals
  3. Quantified RS/RA impact per factor
  4. Cross-team pattern summary

Output:
  data/projections/big_miss_diagnosis.json
  stdout: formatted tables
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stan_jpn_model import MIN_IP, MIN_PA, build_dataset, compute_fip_column, ip_to_decimal, load_birthday_df, standardize_features
from statistical_validation import (
    ALPHA_JPN_H, ALPHA_JPN_P, JPN_YEARS, K_WOBA, NPB_HIST_RS, NPB_PYTH_EXP,
    _compute_league_averages, _load_raw_data, _reassign_teams,
    ridge_fit_predict,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FOREIGN_DIR = DATA_DIR / "foreign"
OUT_DIR = DATA_DIR / "projections"

ERR_THRESHOLD = 10.0  # wins


def load_foreign_names():
    """Load set of foreign player names from master CSV."""
    master = pd.read_csv(FOREIGN_DIR / "foreign_players_master.csv", encoding="utf-8")
    return set(master["npb_name"].values)


def run_loocv_with_detail():
    """Run 8-year LOO-CV and return player-level h_df, p_df with reassigned teams."""
    saber = pd.read_csv(RAW_DIR / "npb_sabermetrics_2015_2025.csv", encoding="utf-8-sig")
    pitchers = pd.read_csv(RAW_DIR / "npb_pitchers_2015_2025.csv", encoding="utf-8-sig")
    saber = saber.dropna(subset=["wOBA"])
    pitchers = compute_fip_column(pitchers)
    bday_df = load_birthday_df()

    feat_h = ["K_pct", "BB_pct", "BABIP", "age_from_peak", "pa_stability", "prev_woba_dev_sq"]
    feat_p = ["K_pct", "BB_pct", "age_from_peak", "ip_stability", "prev_babip_p"]

    all_h, all_p = [], []

    for hold_yr in JPN_YEARS:
        train_years = [y for y in JPN_YEARS if y != hold_yr]
        train_h, train_p, _ = build_dataset(saber, pitchers, train_years, bday_df)
        test_h, test_p, _ = build_dataset(saber, pitchers, [hold_yr], bday_df)

        if len(test_h) == 0 or len(test_p) == 0:
            continue

        # Hitters
        train_z_h, test_z_h, _, _ = standardize_features(train_h, test_h, feat_h)
        y_h = (train_h["actual_woba"] - train_h["marcel_woba"]).values
        delta_h, _ = ridge_fit_predict(train_z_h, y_h, test_z_h, ALPHA_JPN_H)
        stan_woba = test_h["marcel_woba"].values + delta_h

        for i, (_, row) in enumerate(test_h.iterrows()):
            all_h.append({
                "year": hold_yr, "player": row["player"], "team": row["team"],
                "actual": row["actual_woba"], "marcel": row["marcel_woba"],
                "stan": stan_woba[i], "actual_PA": row["actual_PA"],
            })

        # Pitchers
        train_z_p, test_z_p, _, _ = standardize_features(train_p, test_p, feat_p)
        y_p = (train_p["actual_era"] - train_p["marcel_era"]).values
        delta_p, _ = ridge_fit_predict(train_z_p, y_p, test_z_p, ALPHA_JPN_P)
        stan_era = test_p["marcel_era"].values + delta_p

        for i, (_, row) in enumerate(test_p.iterrows()):
            all_p.append({
                "year": hold_yr, "player": row["player"], "team": row["team"],
                "actual": row["actual_era"], "marcel": row["marcel_era"],
                "stan": stan_era[i], "actual_IP": row["actual_IP"],
            })

    h_df = pd.DataFrame(all_h)
    p_df = pd.DataFrame(all_p)

    # Reassign teams
    saber_raw, pitchers_raw = _load_raw_data()
    h_df, p_df = _reassign_teams(h_df, p_df, saber_raw, pitchers_raw)

    return h_df, p_df, saber_raw, pitchers_raw


def _yoy_hitter_analysis(yr, tm, saber):
    """Year-over-year hitter analysis: PA drops and wOBA changes."""
    prev_yr = yr - 1
    # Current year hitters on this team (all PA, not just MIN_PA)
    cur = saber[(saber["year"] == yr) & (saber["team"] == tm) & (saber["PA"] >= MIN_PA)].copy()
    # Previous year: all teams (to catch trades/FA)
    prev_all = saber[(saber["year"] == prev_yr) & (saber["PA"] >= MIN_PA)].copy()
    # Previous year same team
    prev_tm = prev_all[prev_all["team"] == tm]

    # --- Players on team in both years (returners) ---
    cur_players = set(cur["player"])
    prev_tm_players = set(prev_tm["player"])

    # Merge current with their previous stats (any team)
    returners = cur.merge(
        prev_all[["player", "PA", "wOBA", "team"]].rename(
            columns={"PA": "prev_PA", "wOBA": "prev_wOBA", "team": "prev_team"}),
        on="player", how="inner"
    )
    # If player was on multiple teams in prev year, keep highest PA
    returners = returners.sort_values("prev_PA", ascending=False).drop_duplicates("player")
    returners["delta_PA"] = returners["PA"] - returners["prev_PA"]
    returners["delta_wOBA"] = returners["wOBA"] - returners["prev_wOBA"]
    returners["rs_impact_pa"] = K_WOBA * returners["prev_wOBA"] * returners["delta_PA"]
    returners["rs_impact_perf"] = K_WOBA * returners["delta_wOBA"] * returners["PA"]

    # --- Departures: on team in t-1 but not in t (any team) ---
    departed = prev_tm[~prev_tm["player"].isin(cur_players)].copy()
    # Check if they went to another team in year t
    cur_all = saber[(saber["year"] == yr) & (saber["PA"] >= MIN_PA)]
    departed = departed.merge(
        cur_all[["player", "PA", "team", "wOBA"]].rename(
            columns={"PA": "new_PA", "team": "new_team", "wOBA": "new_wOBA"}),
        on="player", how="left"
    )
    departed["lost_rs"] = K_WOBA * departed["wOBA"].fillna(0) * departed["PA"]

    # --- Arrivals: on team in t but not in t-1 (same team) ---
    arrivals = cur[~cur["player"].isin(prev_tm_players)].copy()
    arrivals = arrivals.merge(
        prev_all[["player", "PA", "wOBA", "team"]].rename(
            columns={"PA": "prev_PA", "wOBA": "prev_wOBA", "team": "prev_team"}),
        on="player", how="left"
    )
    arrivals["gained_rs"] = K_WOBA * arrivals["wOBA"].fillna(0) * arrivals["PA"]

    return returners, departed, arrivals


def _yoy_pitcher_analysis(yr, tm, pitchers_raw):
    """Year-over-year pitcher analysis: IP drops and ERA changes."""
    prev_yr = yr - 1
    cur = pitchers_raw[(pitchers_raw["year"] == yr) & (pitchers_raw["team"] == tm)
                       & (pitchers_raw["IP_dec"] >= MIN_IP)].copy()
    prev_all = pitchers_raw[(pitchers_raw["year"] == prev_yr)
                            & (pitchers_raw["IP_dec"] >= MIN_IP)].copy()
    prev_tm = prev_all[prev_all["team"] == tm]

    cur_players = set(cur["player"])
    prev_tm_players = set(prev_tm["player"])

    # Returners
    returners = cur.merge(
        prev_all[["player", "IP_dec", "ERA_num", "team"]].rename(
            columns={"IP_dec": "prev_IP", "ERA_num": "prev_ERA", "team": "prev_team"}),
        on="player", how="inner"
    )
    returners = returners.sort_values("prev_IP", ascending=False).drop_duplicates("player")
    returners["delta_IP"] = returners["IP_dec"] - returners["prev_IP"]
    returners["delta_ERA"] = returners["ERA_num"] - returners["prev_ERA"]
    returners["ra_impact_ip"] = returners["prev_ERA"] * returners["delta_IP"] / 9.0
    returners["ra_impact_perf"] = returners["delta_ERA"] * returners["IP_dec"] / 9.0

    # Departures
    departed = prev_tm[~prev_tm["player"].isin(cur_players)].copy()
    cur_all = pitchers_raw[(pitchers_raw["year"] == yr) & (pitchers_raw["IP_dec"] >= MIN_IP)]
    departed = departed.merge(
        cur_all[["player", "IP_dec", "team", "ERA_num"]].rename(
            columns={"IP_dec": "new_IP", "team": "new_team", "ERA_num": "new_ERA"}),
        on="player", how="left"
    )
    departed["lost_ra"] = departed["ERA_num"].fillna(0) * departed["IP_dec"] / 9.0

    # Arrivals
    arrivals = cur[~cur["player"].isin(prev_tm_players)].copy()
    arrivals = arrivals.merge(
        prev_all[["player", "IP_dec", "ERA_num", "team"]].rename(
            columns={"IP_dec": "prev_IP", "ERA_num": "prev_ERA", "team": "prev_team"}),
        on="player", how="left"
    )
    arrivals["gained_ra"] = arrivals["ERA_num"].fillna(0) * arrivals["IP_dec"] / 9.0

    return returners, departed, arrivals


def main():
    print("=" * 70)
    print("Step 12 Diagnostic: Big-Miss Team-Years (YoY Analysis)")
    print("=" * 70)

    foreign_names = load_foreign_names()
    h_df, p_df, saber, pitchers_raw = run_loocv_with_detail()

    # Load actual wins
    actual = pd.read_csv(
        "https://raw.githubusercontent.com/yasumorishima/npb-prediction/main"
        "/data/projections/pythagorean_2015_2025.csv",
        encoding="utf-8-sig",
    )

    # Load team detail for big-miss identification
    team_detail = pd.read_csv(OUT_DIR / "team_detail_2018_2025.csv")

    big = team_detail[
        (team_detail["err_M"].abs() > ERR_THRESHOLD)
        | (team_detail["err_S"].abs() > ERR_THRESHOLD)
    ].sort_values("err_M", key=abs, ascending=False)

    print(f"\nBig-miss team-years (|err| > {ERR_THRESHOLD}W): {len(big)}")

    # Collect pattern summary
    all_factors = []

    for _, row in big.iterrows():
        yr = int(row["year"])
        tm = row["team"]
        err_m = row["err_M"]
        direction = "overestimate (model too high)" if err_m > 0 else "underestimate (model too low)"

        print(f"\n{'='*70}")
        print(f"  {yr} {tm}  actual={int(row['actual_W'])}W  "
              f"Marcel={row['W_marcel']:.1f}  Stan={row['W_stan']:.1f}  "
              f"err_M={err_m:+.1f}  err_S={row['err_S']:+.1f}")
        print(f"  Direction: {direction}")
        print(f"  Coverage: PA={row['PA_cov']:.1f}%  IP={row['IP_cov']:.1f}%")
        print(f"{'='*70}")

        # === HITTER YoY ===
        ret_h, dep_h, arr_h = _yoy_hitter_analysis(yr, tm, saber)

        # Top PA droppers (negative delta = played less)
        pa_drops = ret_h[ret_h["delta_PA"] < -100].sort_values("delta_PA")
        if len(pa_drops) > 0:
            print(f"\n  [Hitter PA drops (>100 PA decrease)]")
            for _, r in pa_drops.head(5).iterrows():
                fg = " [FGN]" if r["player"] in foreign_names else ""
                print(f"    {r['player']:10s}{fg:6s}  "
                      f"PA: {int(r['prev_PA']):>3d}→{int(r['PA']):>3d} (Δ{int(r['delta_PA']):+d})  "
                      f"wOBA: {r['prev_wOBA']:.3f}→{r['wOBA']:.3f}  "
                      f"RS impact: {r['rs_impact_pa']:+.1f}")

        # Top wOBA changers
        perf_h = ret_h[ret_h["delta_wOBA"].abs() > 0.040].sort_values("rs_impact_perf")
        if len(perf_h) > 0:
            print(f"\n  [Hitter wOBA surprises (|Δ|>.040)]")
            for _, r in perf_h.iterrows():
                fg = " [FGN]" if r["player"] in foreign_names else ""
                print(f"    {r['player']:10s}{fg:6s}  "
                      f"PA={int(r['PA']):>3d}  "
                      f"wOBA: {r['prev_wOBA']:.3f}→{r['wOBA']:.3f} (Δ{r['delta_wOBA']:+.3f})  "
                      f"RS impact: {r['rs_impact_perf']:+.1f}")

        # Departures (top 5 by lost RS)
        dep_h_sorted = dep_h.sort_values("lost_rs", ascending=False)
        if len(dep_h_sorted) > 0:
            print(f"\n  [Hitter departures (left team)]")
            for _, r in dep_h_sorted.head(5).iterrows():
                fg = " [FGN]" if r["player"] in foreign_names else ""
                dest = f"→{r['new_team']}" if pd.notna(r.get("new_team")) else "→retired/2軍"
                print(f"    {r['player']:10s}{fg:6s}  "
                      f"prev PA={int(r['PA']):>3d}  wOBA={r['wOBA']:.3f}  "
                      f"lost RS={r['lost_rs']:.1f}  {dest}")

        # Arrivals (top 5 by gained RS)
        arr_h_sorted = arr_h.sort_values("gained_rs", ascending=False)
        if len(arr_h_sorted) > 0:
            print(f"\n  [Hitter arrivals (joined team)]")
            for _, r in arr_h_sorted.head(5).iterrows():
                fg = " [FGN]" if r["player"] in foreign_names else ""
                src = f"←{r['prev_team']}" if pd.notna(r.get("prev_team")) else "←rookie/2軍"
                print(f"    {r['player']:10s}{fg:6s}  "
                      f"PA={int(r['PA']):>3d}  wOBA={r['wOBA']:.3f}  "
                      f"gained RS={r['gained_rs']:.1f}  {src}")

        # === PITCHER YoY ===
        ret_p, dep_p, arr_p = _yoy_pitcher_analysis(yr, tm, pitchers_raw)

        # Top IP droppers
        ip_drops = ret_p[ret_p["delta_IP"] < -30].sort_values("delta_IP")
        if len(ip_drops) > 0:
            print(f"\n  [Pitcher IP drops (>30 IP decrease)]")
            for _, r in ip_drops.head(5).iterrows():
                fg = " [FGN]" if r["player"] in foreign_names else ""
                era_prev = r["prev_ERA"] if pd.notna(r["prev_ERA"]) else 0
                era_cur = r["ERA_num"] if pd.notna(r["ERA_num"]) else 0
                print(f"    {r['player']:10s}{fg:6s}  "
                      f"IP: {r['prev_IP']:>5.1f}→{r['IP_dec']:>5.1f} (Δ{r['delta_IP']:+.1f})  "
                      f"ERA: {era_prev:.2f}→{era_cur:.2f}  "
                      f"RA impact: {r['ra_impact_ip']:+.1f}")

        # Top ERA changers
        perf_p = ret_p[ret_p["delta_ERA"].abs() > 1.0].sort_values("ra_impact_perf", ascending=False)
        if len(perf_p) > 0:
            print(f"\n  [Pitcher ERA surprises (|Δ|>1.0)]")
            for _, r in perf_p.iterrows():
                fg = " [FGN]" if r["player"] in foreign_names else ""
                era_prev = r["prev_ERA"] if pd.notna(r["prev_ERA"]) else 0
                era_cur = r["ERA_num"] if pd.notna(r["ERA_num"]) else 0
                print(f"    {r['player']:10s}{fg:6s}  "
                      f"IP={r['IP_dec']:>5.1f}  "
                      f"ERA: {era_prev:.2f}→{era_cur:.2f} (Δ{r['delta_ERA']:+.2f})  "
                      f"RA impact: {r['ra_impact_perf']:+.1f}")

        # Pitcher departures
        dep_p_sorted = dep_p.sort_values("lost_ra", ascending=False)
        if len(dep_p_sorted) > 0:
            print(f"\n  [Pitcher departures]")
            for _, r in dep_p_sorted.head(5).iterrows():
                fg = " [FGN]" if r["player"] in foreign_names else ""
                era = r["ERA_num"] if pd.notna(r["ERA_num"]) else 0
                dest = f"→{r['new_team']}" if pd.notna(r.get("new_team")) else "→retired/2軍"
                print(f"    {r['player']:10s}{fg:6s}  "
                      f"prev IP={r['IP_dec']:>5.1f}  ERA={era:.2f}  "
                      f"lost RA={r['lost_ra']:.1f}  {dest}")

        # Pitcher arrivals
        arr_p_sorted = arr_p.sort_values("gained_ra", ascending=True)  # lower RA = better
        if len(arr_p_sorted) > 0:
            print(f"\n  [Pitcher arrivals]")
            for _, r in arr_p_sorted.head(5).iterrows():
                fg = " [FGN]" if r["player"] in foreign_names else ""
                era = r["ERA_num"] if pd.notna(r["ERA_num"]) else 0
                src = f"←{r['prev_team']}" if pd.notna(r.get("prev_team")) else "←rookie/2軍"
                print(f"    {r['player']:10s}{fg:6s}  "
                      f"IP={r['IP_dec']:>5.1f}  ERA={era:.2f}  "
                      f"gained RA={r['gained_ra']:.1f}  {src}")

        # === Factor quantification ===
        total_lost_rs = float(dep_h["lost_rs"].sum()) if len(dep_h) > 0 else 0
        total_gained_rs = float(arr_h["gained_rs"].sum()) if len(arr_h) > 0 else 0
        total_pa_impact = float(ret_h["rs_impact_pa"].sum()) if len(ret_h) > 0 else 0
        total_perf_impact_h = float(ret_h["rs_impact_perf"].sum()) if len(ret_h) > 0 else 0
        total_lost_ra = float(dep_p["lost_ra"].sum()) if len(dep_p) > 0 else 0
        total_gained_ra = float(arr_p["gained_ra"].sum()) if len(arr_p) > 0 else 0
        total_ip_impact = float(ret_p["ra_impact_ip"].sum()) if len(ret_p) > 0 else 0
        total_perf_impact_p = float(ret_p["ra_impact_perf"].sum()) if len(ret_p) > 0 else 0

        print(f"\n  [Factor Summary (RS/RA impact)]")
        print(f"    Roster turnover RS: lost={total_lost_rs:.1f}  gained={total_gained_rs:.1f}  "
              f"net={total_gained_rs - total_lost_rs:+.1f}")
        print(f"    PA change impact:   {total_pa_impact:+.1f}")
        print(f"    wOBA change impact: {total_perf_impact_h:+.1f}")
        print(f"    Roster turnover RA: lost={total_lost_ra:.1f}  gained={total_gained_ra:.1f}  "
              f"net={total_gained_ra - total_lost_ra:+.1f}")
        print(f"    IP change impact:   {total_ip_impact:+.1f}")
        print(f"    ERA change impact:  {total_perf_impact_p:+.1f}")

        all_factors.append({
            "year": yr, "team": tm,
            "err_M": round(float(err_m), 1),
            "pa_cov": round(float(row["PA_cov"]), 1),
            "ip_cov": round(float(row["IP_cov"]), 1),
            "n_departures_h": len(dep_h), "n_arrivals_h": len(arr_h),
            "n_departures_p": len(dep_p), "n_arrivals_p": len(arr_p),
            "n_pa_drops": len(pa_drops) if len(pa_drops) > 0 else 0,
            "n_ip_drops": len(ip_drops) if len(ip_drops) > 0 else 0,
            "n_woba_surprises": len(perf_h) if len(perf_h) > 0 else 0,
            "n_era_surprises": len(perf_p) if len(perf_p) > 0 else 0,
            "roster_net_rs": round(total_gained_rs - total_lost_rs, 1),
            "perf_impact_rs": round(total_perf_impact_h, 1),
            "roster_net_ra": round(total_gained_ra - total_lost_ra, 1),
            "perf_impact_ra": round(total_perf_impact_p, 1),
        })

    # === Cross-team pattern summary ===
    print("\n" + "=" * 70)
    print("Cross-Team Pattern Summary")
    print("=" * 70)

    factors_df = pd.DataFrame(all_factors)
    if len(factors_df) > 0:
        over = factors_df[factors_df["err_M"] > 0]  # model too high
        under = factors_df[factors_df["err_M"] < 0]  # model too low

        for label, sub in [("Overestimates (model > actual)", over),
                           ("Underestimates (model < actual)", under)]:
            if len(sub) == 0:
                continue
            print(f"\n  {label}: n={len(sub)}")
            print(f"    Avg PA_cov: {sub['pa_cov'].mean():.1f}%")
            print(f"    Avg IP_cov: {sub['ip_cov'].mean():.1f}%")
            print(f"    Avg departures: {sub['n_departures_h'].mean():.1f} hitters, "
                  f"{sub['n_departures_p'].mean():.1f} pitchers")
            print(f"    Avg arrivals:   {sub['n_arrivals_h'].mean():.1f} hitters, "
                  f"{sub['n_arrivals_p'].mean():.1f} pitchers")
            print(f"    Avg PA drops(>100):  {sub['n_pa_drops'].mean():.1f}")
            print(f"    Avg IP drops(>30):   {sub['n_ip_drops'].mean():.1f}")
            print(f"    Avg wOBA surprises:  {sub['n_woba_surprises'].mean():.1f}")
            print(f"    Avg ERA surprises:   {sub['n_era_surprises'].mean():.1f}")
            print(f"    Avg roster net RS:   {sub['roster_net_rs'].mean():+.1f}")
            print(f"    Avg perf impact RS:  {sub['perf_impact_rs'].mean():+.1f}")
            print(f"    Avg roster net RA:   {sub['roster_net_ra'].mean():+.1f}")
            print(f"    Avg perf impact RA:  {sub['perf_impact_ra'].mean():+.1f}")

        # Correlation: what factors best predict the error direction?
        print(f"\n  Correlation with err_M:")
        for col in ["pa_cov", "ip_cov", "n_departures_h", "n_arrivals_h",
                     "n_pa_drops", "n_woba_surprises", "n_era_surprises",
                     "roster_net_rs", "perf_impact_rs", "roster_net_ra", "perf_impact_ra"]:
            corr = factors_df["err_M"].corr(factors_df[col])
            print(f"    {col:>20s}: r={corr:+.3f}")

    # Save JSON
    out = {"threshold": ERR_THRESHOLD, "factors": all_factors}
    out_path = OUT_DIR / "big_miss_diagnosis.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
