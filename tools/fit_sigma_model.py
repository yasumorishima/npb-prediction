"""予測区間の σ を出場機会の関数として fit し、held-out 年で確かめる。

  sigma_i = sigma_base * exp(gamma * z_log_pt)

式は `bayes_projection.predict_foreign_*` が元から使っている形で、結果は
`data/bayes/posteriors.json` の `jpn_hitter.sigma_model` / `jpn_pitcher.sigma_model`
に入る。

なぜ Marcel 残差を代理に使うか: posteriors.json の LOO-CV では Stan 補正が MAE を
0.9% しか動かさない（marcel 0.05023 -> stan 0.04980）ので、区間が担うべき残差の
大きさは Marcel 残差でほぼ決まる。Marcel は target_year-1..-3 しか読まない
（marcel_projection.py）ので各年の予測は因果。

🔴 出場機会の扱い（一度間違えた所）:
  - **実績 IP は NPB 表記**（"10.2" = 10 と 2/3 回）なので換算が要る。
  - **予測 IP は Marcel が出す平均の小数**（marcel_projection.py の round(avg_ip, 1)）で
    NPB 表記ではない。実データでは小数第 1 位が 3 以上の行が 61% ある。
    ここに換算を当てると 25.9 -> 28.0 のように壊れる。**換算しない。**
  - σ に食わせるのは **予測**（3 月に分かる値）。足切りに使うのは **実績**。

使い方:
    python tools/fit_sigma_model.py
    python tools/fit_sigma_model.py --current-hitters <csv> --current-pitchers <csv>

`--current-*` は当年（まだ repo に無い年）の実績 CSV。省略するとその年の
held-out 採点は飛ばす。当年の実績は fetch_npb_data.py と同じく
baseball-data.com の当年ページから取る。

出典: baseball-data.com
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NPB_DATA_END_YEAR", "2025")

import marcel_projection as M  # noqa: E402

Z80 = 1.2815515655446004
Z95 = 1.959963984540054
TRAIN_YEARS = list(range(2018, 2025))   # 2025 は held-out として残す
HITTER_PA_MIN = 100
PITCHER_IP_MIN = 30
WOBA_TO_OPS_SLOPE = 2.33                # woba_to_ops_approx の傾き


def norm_name(s):
    return str(s).replace(" ", "").replace("　", "").strip()


def npb_ip(value):
    """NPB 表記の投球回を実数へ。"10.2" = 10 と 2/3 回。実績にだけ当てる。

    使えない値（欠測・空欄・"-" 等）は NaN を返す。呼び出し側は後段の
    dropna() で落とす。ここで例外を投げると、その年に 1 行でも空欄があった
    だけで再現スクリプト全体が止まる。
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(v):
        return float("nan")
    whole = np.floor(v)
    return whole + round((v - whole) * 10) / 3.0


def flat_sigma_from_posteriors():
    """区間が従来使っていた平の σ を posteriors.json から読む（ベタ書きしない）。"""
    post = json.loads((ROOT / "data" / "bayes" / "posteriors.json").read_text(encoding="utf-8"))
    return {
        "hitter": post["jpn_hitter"]["sigma_residual"] * WOBA_TO_OPS_SLOPE,   # OPS 尺度へ
        "pitcher": post["jpn_pitcher"]["sigma_residual"],                     # ERA 尺度
    }


def residuals(proj, actual, stat, actual_is_npb_ip):
    """予測と実績を選手名で突き合わせ、残差・予測出場機会・実績出場機会を返す。"""
    pt_col = "IP" if actual_is_npb_ip else "PA"
    p, a = proj.copy(), actual.copy()
    p["_k"] = p["player"].map(norm_name)
    a["_k"] = a["player"].map(norm_name)
    p = p[~p["_k"].duplicated(keep=False)]
    a = a[~a["_k"].duplicated(keep=False)]
    m = p.merge(a, on="_k", suffixes=("_p", "_a"))
    # 実績: 投手だけ NPB 表記の換算が要る
    m["_actual_pt"] = m[pt_col + "_a"].map(npb_ip) if actual_is_npb_ip \
        else pd.to_numeric(m[pt_col + "_a"], errors="coerce")
    # 予測: Marcel が書く平均の小数。換算してはいけない
    m["_proj_pt"] = pd.to_numeric(m[pt_col + "_p"], errors="coerce")
    m["_err"] = (pd.to_numeric(m[stat + "_a"], errors="coerce")
                 - pd.to_numeric(m[stat + "_p"], errors="coerce"))
    return m.dropna(subset=["_err", "_proj_pt", "_actual_pt"])[["_k", "_proj_pt", "_actual_pt", "_err"]]


def gather(df, marcel_fn, stat, actual_is_npb_ip, years, pt_min):
    out = []
    for year in years:
        d = residuals(marcel_fn(df, year), df[df["year"] == year], stat, actual_is_npb_ip)
        out.append(d[d["_actual_pt"] >= pt_min].assign(year=year))
    return pd.concat(out, ignore_index=True)


def fit(train, floor):
    """sigma_base と gamma を最尤で。z は log(予測出場機会) の標準化。

    🔴 予測出場機会は **deploy 時と同じく floor で頭打ちにしてから** fit する。
    足切りは実績で行うので、実績は閾値を超えたのに予測が floor 未満、という
    訓練行が存在する（打者）。clamp せずに fit すると、
    その行は運用では起こりえない z で係数に効いてしまう。
    「当てる変換で fit する」を守る。
    """
    log_pt = np.log(np.maximum(train["_proj_pt"].values, floor))
    mean, sd = float(log_pt.mean()), float(log_pt.std(ddof=0))
    z = (log_pt - mean) / sd
    err = train["_err"].values

    def nll(params):
        sigma = np.exp(params[0]) * np.exp(params[1] * z)
        return float(np.sum(np.log(sigma) + err ** 2 / (2 * sigma ** 2)))

    res = minimize(nll, [np.log(err.std()), -0.1], method="Nelder-Mead",
                   options=dict(xatol=1e-9, fatol=1e-9, maxiter=5000))
    return float(np.exp(res.x[0])), float(res.x[1]), mean, sd


def coverage(err, sigma):
    err = np.asarray(err, float)
    sigma = np.asarray(sigma, float)
    return (float(np.mean(np.abs(err) <= Z80 * sigma)),
            float(np.mean(np.abs(err) <= Z95 * sigma)))


def sigma_at(base, gamma, mean, sd, pt, floor):
    pt = np.maximum(np.asarray(pt, float), floor)
    return base * np.exp(gamma * (np.log(pt) - mean) / sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current-hitters", help="当年の打者実績 CSV（baseball-data.com）")
    ap.add_argument("--current-pitchers", help="当年の投手実績 CSV")
    ap.add_argument("--current-year", type=int, default=2026)
    args = ap.parse_args()

    flat = flat_sigma_from_posteriors()
    hitters, pitchers = M.load_hitters(), M.load_pitchers()

    specs = [
        ("HITTER  OPS", hitters, M.marcel_hitter, "OPS", False, HITTER_PA_MIN,
         flat["hitter"], args.current_hitters),
        ("PITCHER ERA", pitchers, M.marcel_pitcher, "ERA", True, PITCHER_IP_MIN,
         flat["pitcher"], args.current_pitchers),
    ]

    for label, df, fn, stat, npb_ip_flag, pt_min, flat_sigma, current_csv in specs:
        train = gather(df, fn, stat, npb_ip_flag, TRAIN_YEARS, pt_min)
        below = int((train["_proj_pt"] < pt_min).sum())
        base, gamma, mean, sd = fit(train, pt_min)
        floor = pt_min                      # fit の台の下端。これより下は外挿しない
        print("")
        print("%s  train n=%d (%d-%d)  うち予測が floor 未満で clamp された行 %d"
              % (label, len(train), TRAIN_YEARS[0], TRAIN_YEARS[-1], below))
        print("  sigma_base %.6f  gamma %+.6f  log_mean %.6f  log_sd %.6f  pt_floor %d"
              % (base, gamma, mean, sd, floor))
        print("  flat sigma actually in posteriors.json: %.5f" % flat_sigma)
        for pt in (pt_min, pt_min * 2, pt_min * 4):
            print("      pt=%4d -> sigma %.5f" % (pt, sigma_at(base, gamma, mean, sd, pt, floor)))

        held = gather(df, fn, stat, npb_ip_flag, [2025], pt_min)
        c_flat = coverage(held["_err"], np.full(len(held), flat_sigma))
        c_fit = coverage(held["_err"], sigma_at(base, gamma, mean, sd, held["_proj_pt"], floor))
        print("  HELD-OUT 2025 (n=%d): flat 80%% %.3f / 95%% %.3f   fitted 80%% %.3f / 95%% %.3f"
              % (len(held), c_flat[0], c_flat[1], c_fit[0], c_fit[1]))

        if not current_csv:
            print("  HELD-OUT %d: skipped (--current-* not given)" % args.current_year)
            continue
        actual = pd.read_csv(current_csv, encoding="utf-8-sig")
        cur = residuals(fn(df, args.current_year), actual, stat, npb_ip_flag)
        cur = cur[cur["_actual_pt"] >= pt_min]
        c_flat = coverage(cur["_err"], np.full(len(cur), flat_sigma))
        c_fit = coverage(cur["_err"], sigma_at(base, gamma, mean, sd, cur["_proj_pt"], floor))
        print("  HELD-OUT %d (n=%d): flat 80%% %.3f / 95%% %.3f   fitted 80%% %.3f / 95%% %.3f"
              % (args.current_year, len(cur), c_flat[0], c_flat[1], c_fit[0], c_fit[1]))

    print("")
    print("打者の sigma_base は OPS 尺度。posteriors.json は wOBA 尺度で持つので")
    print("%.2f で割って入れる（区間は wOBA 空間で作られ woba_to_ops_approx で変換される）。"
          % WOBA_TO_OPS_SLOPE)
    print("出典: baseball-data.com")


if __name__ == "__main__":
    main()
