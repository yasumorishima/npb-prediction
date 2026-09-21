"""Fit sigma as a function of playing time, in the repo's OWN parameterisation
       sigma_i = sigma_base * exp(gamma * z_log_pt)
   (the form predict_foreign_* already uses at bayes_projection.py L616 / L717),
   so the result drops into apply_stan_correction.

DECLARED BEFORE RUNNING:
  fit on 2018-2024 only.  2025 and 2026 are never touched by the fit.
  P8 = P6 but sigma is driven by PROJECTED playing time.
  P6  on BOTH held-out years the fitted sigma gives 80%/95% coverage closer to
      nominal than the flat sigma does (0.1429 OPS / 1.6378 ERA).
  P7  gamma < 0 (sigma must SHRINK as playing time grows).  If gamma >= 0 the
      whole playing-time story is wrong and I say so.
Marcel residual is the proxy (Bayes layer moves MAE by -0.9%).
Source: baseball-data.com
"""
import sys, os
sys.path.insert(0, ".")
os.environ.setdefault("NPB_DATA_END_YEAR", "2025")
import numpy as np, pandas as pd
from scipy.optimize import minimize
import marcel_projection as M

Z80, Z95 = 1.2815515655446004, 1.959963984540054
SIG_OPS, SIG_ERA = 0.1429, 1.6378

def norm(s): return str(s).replace(" ", "").replace("　", "").strip()

def resid(proj, actual, tcol, thr, col, conv=None):
    P = proj.copy(); A = actual.copy()
    P["_k"] = P["player"].map(norm); A["_k"] = A["player"].map(norm)
    P = P[~P["_k"].duplicated(keep=False)]; A = A[~A["_k"].duplicated(keep=False)]
    m = P.merge(A, on="_k", suffixes=("_p", "_a"))
    tc = tcol + "_a" if (tcol + "_a") in m.columns else tcol
    tv = m[tc].map(conv) if conv else pd.to_numeric(m[tc], errors="coerce")
    pj = tcol + "_p" if (tcol + "_p") in m.columns else tcol
    pv = m[pj].map(conv) if conv else pd.to_numeric(m[pj], errors="coerce")
    m = m.assign(_pt=tv, _ptp=pv)
    m = m[m["_pt"] >= thr]
    pc = col + "_p" if (col + "_p") in m.columns else col
    ac = col + "_a" if (col + "_a") in m.columns else col
    m["_e"] = pd.to_numeric(m[ac], errors="coerce") - pd.to_numeric(m[pc], errors="coerce")
    return m.dropna(subset=["_e", "_pt", "_ptp"])[["_k", "_pt", "_ptp", "_e"]]

def gather(load, mk, tcol, thr, col, years, conv=None):
    out = []
    for y in years:
        d = resid(mk(load, y), load[load["year"] == y], tcol, thr, col, conv=conv)
        d = d.assign(year=y)
        out.append(d)
    return pd.concat(out, ignore_index=True)

def fit(d, mu_log, sd_log):
    z = (np.log(d["_ptp"].values) - mu_log) / sd_log
    e = d["_e"].values
    def nll(p):
        s = np.exp(p[0]) * np.exp(p[1] * z)
        return np.sum(np.log(s) + e ** 2 / (2 * s ** 2))
    r = minimize(nll, [np.log(e.std()), -0.1], method="Nelder-Mead",
                 options=dict(xatol=1e-8, fatol=1e-8, maxiter=4000))
    return float(np.exp(r.x[0])), float(r.x[1])

def cover(e, s):
    return float(np.mean(np.abs(e) <= Z80 * s)), float(np.mean(np.abs(e) <= Z95 * s))

print("=" * 88)
print("Fit sigma = sigma_base * exp(gamma * z_log_pt)   PROJECTED playing time (known in March)")
print("source: baseball-data.com")
print("=" * 88)

H = M.load_hitters(); P = M.load_pitchers()
TRAIN = list(range(2018, 2025))

specs = [("HITTER OPS", H, M.marcel_hitter, "PA", 100, "OPS", None, SIG_OPS,
          "bayes_hitters_2026.csv", "npb_2026_hitters.csv", "bayes_OPS", "OPS"),
         ("PITCHER ERA", P, M.marcel_pitcher, "IP", 30, "ERA", M._parse_ip, SIG_ERA,
          "bayes_pitchers_2026.csv", "npb_2026_pitchers.csv", "bayes_ERA", "ERA")]

for (tag, load, mk, tcol, thr, col, conv, flat, pf26, af26, mucol, acol) in specs:
    tr = gather(load, mk, tcol, thr, col, TRAIN, conv=conv)
    mu_log = float(np.log(tr["_ptp"]).mean()); sd_log = float(np.log(tr["_ptp"]).std(ddof=0))
    s0, g = fit(tr, mu_log, sd_log)
    print("")
    print("%s   train n=%d (2018-2024)" % (tag, len(tr)))
    print("  fitted: sigma_base %.6f   gamma %+.6f   (standardisation of log %s: mean %.6f sd %.6f)"
          % (s0, g, tcol, mu_log, sd_log))
    print("  P7 gamma < 0 : %s" % ("HOLDS" if g < 0 else "FAILS"))
    lo, hi = np.exp(np.log(np.array([thr, tr['_pt'].max()])))
    for pt in [thr, 200 if tcol == "PA" else 60, 400 if tcol == "PA" else 120, int(tr["_ptp"].max())]:
        z = (np.log(pt) - mu_log) / sd_log
        print("      %s=%4d -> sigma %.4f   (flat model uses %.4f)" % (tcol, pt, s0 * np.exp(g * z), flat))
    # ---- held-out 2025 (same source, causal Marcel) ----
    te = gather(load, mk, tcol, thr, col, [2025], conv=conv)
    zt = (np.log(te["_ptp"].values) - mu_log) / sd_log
    sfit = s0 * np.exp(g * zt)
    c_flat = cover(te["_e"].values, np.full(len(te), flat))
    c_fit = cover(te["_e"].values, sfit)
    print("  HELD-OUT 2025 (n=%d):  flat  80%% %.3f / 95%% %.3f    fitted  80%% %.3f / 95%% %.3f"
          % (len(te), c_flat[0], c_flat[1], c_fit[0], c_fit[1]))
    # ---- held-out 2026 (the frozen projections vs the real season) ----
    pr = pd.read_csv(pf26, encoding="utf-8-sig"); ac = pd.read_csv(af26, encoding="utf-8-sig")
    pr["_k"] = pr["player"].map(norm); ac["_k"] = ac["player"].map(norm)
    pr = pr[~pr["_k"].duplicated(keep=False)]; ac = ac[~ac["_k"].duplicated(keep=False)]
    m = pr.merge(ac, on="_k", suffixes=("_p", "_a"))
    tc = tcol + "_a" if (tcol + "_a") in m.columns else tcol
    m["_pt"] = m[tc].map(conv) if conv else pd.to_numeric(m[tc], errors="coerce")
    tp = tcol + "_p" if (tcol + "_p") in m.columns else tcol
    m["_ptp"] = m[tp].map(conv) if conv else pd.to_numeric(m[tp], errors="coerce")
    m = m[m["_pt"] >= thr]
    acn = acol + "_a" if (acol + "_a") in m.columns else acol
    m["_e"] = pd.to_numeric(m[acn], errors="coerce") - pd.to_numeric(m[mucol], errors="coerce")
    m = m.dropna(subset=["_e", "_pt", "_ptp"])
    z6 = (np.log(m["_ptp"].values) - mu_log) / sd_log
    s6 = s0 * np.exp(g * z6)
    c_flat6 = cover(m["_e"].values, np.full(len(m), flat))
    c_fit6 = cover(m["_e"].values, s6)
    print("  HELD-OUT 2026 (n=%d):  flat  80%% %.3f / 95%% %.3f    fitted  80%% %.3f / 95%% %.3f"
          % (len(m), c_flat6[0], c_flat6[1], c_fit6[0], c_fit6[1]))
    d_flat = abs(c_flat[0] - .80) + abs(c_flat[1] - .95) + abs(c_flat6[0] - .80) + abs(c_flat6[1] - .95)
    d_fit = abs(c_fit[0] - .80) + abs(c_fit[1] - .95) + abs(c_fit6[0] - .80) + abs(c_fit6[1] - .95)
    print("  P6 total distance from nominal over both held-out years: flat %.3f -> fitted %.3f  : %s"
          % (d_flat, d_fit, "HOLDS" if d_fit < d_flat else "FAILS"))
