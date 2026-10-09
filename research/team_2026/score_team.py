"""Score a season's team predictions as pre-registered in
data/backtest/2026_team_prereg.md.

T1  MAE of median_wins against final wins (12 teams)       -- primary
T2  Spearman(median_wins, wins): all 12 and each league, with both ranges
T3  Brier of p_pennant (actual champion = 1)
T4  Brier of p_last (actual last place = 1)
T5  coverage of the 80% / 95% win intervals, as counts k of 12
Floors on the same 12 teams: F1 constant (previous season's 12-team mean
wins), F2 previous season's wins, F3 Pythagorean wins of the previous season
(k = 1.72, times W + L), and 1/6 within the league for T3 / T4.

Usage: python score_team.py <year> <team_sim csv> <final standings csv> <standings 2015-2025 csv> <out json>
The final standings csv needs team, G, W, L, T (or D), and rank and league if
the year is not in the 2015-2025 file.
"""
import json
import sys

import numpy as np
import pandas as pd

K = 1.72
LEAGUE = {"阪神": "CL", "DeNA": "CL", "巨人": "CL", "中日": "CL", "広島": "CL", "ヤクルト": "CL",
          "ソフトバンク": "PL", "日本ハム": "PL", "オリックス": "PL", "楽天": "PL", "西武": "PL", "ロッテ": "PL"}


def spearman(a, b):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()   # midranks
    return float(np.corrcoef(ra, rb)[0, 1])


def main(year, sim_csv, final_csv, hist_csv, out_json):
    year = int(year)
    sim = pd.read_csv(sim_csv, encoding="utf-8-sig")
    fin = pd.read_csv(final_csv, encoding="utf-8-sig").rename(columns={"D": "T"})
    hist = pd.read_csv(hist_csv, encoding="utf-8-sig")
    prev = hist[hist["year"] == year - 1]
    for name, df in (("predictions", sim), ("final", fin), ("previous season", prev)):
        if len(df) != 12 or set(df["team"]) != set(LEAGUE):
            raise SystemExit(f"{name}: not the 12 teams")
    d = sim.merge(fin[["team", "G", "W", "L", "T"]], on="team").merge(
        prev[["team", "W", "L", "RS", "RA"]].rename(columns={"W": "W_prev", "L": "L_prev"}), on="team")
    d["league"] = d["team"].map(LEAGUE)
    if any(LEAGUE[t] != lg for t, lg in zip(sim["team"], sim["league"])):
        raise SystemExit("league labels in the predictions differ")

    # final rank within the league from the standings: by winning percentage (W / (W + L))
    d["wpct"] = d["W"] / (d["W"] + d["L"])
    if "rank" in fin.columns:
        d = d.merge(fin[["team", "rank"]], on="team")
    else:
        d["rank"] = d.groupby("league")["wpct"].rank(ascending=False, method="min")
    champ = d["rank"] == 1
    last = d.groupby("league")["rank"].transform("max") == d["rank"]
    if champ.sum() != 2 or last.sum() != 2:
        raise SystemExit("could not tell one champion and one last place per league")

    d["F1"] = prev["W"].mean()
    d["F2"] = d["W_prev"]
    d["F3"] = d["RS"] ** K / (d["RS"] ** K + d["RA"] ** K) * (d["W_prev"] + d["L_prev"])
    mae = lambda col: float((d[col] - d["W"]).abs().mean())
    brier = lambda p, hit: float(((d[p] - hit.astype(float)) ** 2).mean())
    uniform = float(((1 / 6 - champ.astype(float)) ** 2).mean())

    res = {"year": year, "games": {r.team: int(r.G) for r in d.itertuples()},
           "T1_mae_median_wins": mae("median_wins"), "T1_mae_mean_wins": mae("mean_wins"),
           "F1_constant": mae("F1"), "F2_previous": mae("F2"), "F3_pythagorean": mae("F3")}
    res["T2_spearman"] = {"all": spearman(d["median_wins"], d["W"])}
    for lg in ("CL", "PL"):
        g = d[d["league"] == lg]
        res["T2_spearman"][lg] = spearman(g["median_wins"], g["W"])
        res["T2_spearman"][f"{lg}_range_predicted"] = [float(g["median_wins"].min()), float(g["median_wins"].max())]
        res["T2_spearman"][f"{lg}_range_actual"] = [int(g["W"].min()), int(g["W"].max())]
    res["T3_brier_pennant"] = brier("p_pennant", champ)
    res["T4_brier_last"] = brier("p_last", last)
    res["T3_T4_uniform_floor"] = uniform
    for lvl in ("80", "95"):
        inside = (d["W"] >= d[f"wins_{lvl}ci_lo"]) & (d["W"] <= d[f"wins_{lvl}ci_hi"])
        res[f"T5_k_inside_{lvl}"] = int(inside.sum())
    res["champions"] = sorted(d.loc[champ, "team"])
    res["last_places"] = sorted(d.loc[last, "team"])
    res["per_team"] = d[["team", "league", "G", "W", "median_wins", "wins_80ci_lo", "wins_80ci_hi",
                         "wins_95ci_lo", "wins_95ci_hi", "F1", "F2", "F3", "p_pennant", "p_last"]].round(3).to_dict("records")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_team"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
