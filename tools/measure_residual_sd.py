"""How wide SHOULD the interval be?  Measured on 8 causal seasons, not on 2026 alone.

Marcel for year Y uses only years Y-1, Y-2, Y-3 (marcel_projection.py L103/L228) -> causal.
The Bayes layer moves MAE by -0.9% (posteriors loocv: marcel 0.05023 -> stan 0.04980),
so the Marcel residual sd is used here as a proxy for the residual sd the interval should carry.

DECLARED BEFORE RUNNING:
  P4  residual sd on the PA>=100 / IP>=30 population is BELOW the sigma the model applies
      (OPS 0.1429 / ERA 1.6378) in all 8 years 2018-2025.
  P5  within each year residual sd falls as playing time rises (same shape as 2026).
If either fails, the 2026 reading is suspect and I say so.
NOTE 2020 was a 120-game season -> reported but flagged.
Source: baseball-data.com
"""
import sys, os
sys.path.insert(0, ".")
os.environ.setdefault("NPB_DATA_END_YEAR", "2025")
import numpy as np, pandas as pd
import marcel_projection as M

SIG_OPS, SIG_ERA = 0.1429, 1.6378          # recovered from the 2026 intervals
YEARS = list(range(2018, 2026))

def norm(s): return str(s).replace(" ", "").replace("　", "").strip()

def resid(proj, actual, key, thr_col, thr, pred_col, act_col, conv=None):
    P = proj.copy(); A = actual.copy()
    P["_k"] = P["player"].map(norm); A["_k"] = A["player"].map(norm)
    P = P[~P["_k"].duplicated(keep=False)]; A = A[~A["_k"].duplicated(keep=False)]
    m = P.merge(A, on="_k", suffixes=("_p", "_a"))
    tcol = thr_col + "_a" if (thr_col + "_a") in m.columns else thr_col
    tv = m[tcol].map(conv) if conv else pd.to_numeric(m[tcol], errors="coerce")
    m = m.assign(_pt=tv)
    m = m[m["_pt"] >= thr]
    pc = pred_col + "_p" if (pred_col + "_p") in m.columns else pred_col
    ac = act_col + "_a" if (act_col + "_a") in m.columns else act_col
    m["_e"] = pd.to_numeric(m[ac], errors="coerce") - pd.to_numeric(m[pc], errors="coerce")
    return m.dropna(subset=["_e"])

print("=" * 86)
print("Residual sd of a CAUSAL Marcel projection, 2018-2025   / source: baseball-data.com")
print("=" * 86)

H = M.load_hitters(); P = M.load_pitchers()
rows = []
for what, load, mk, tcol, thr, pcol, acol, conv, sig in [
        ("HITTER OPS", H, M.marcel_hitter, "PA", 100, "OPS", "OPS", None, SIG_OPS),
        ("PITCHER ERA", P, M.marcel_pitcher, "IP", 30, "ERA", "ERA", M._parse_ip, SIG_ERA)]:
    print("")
    print("%s   model applies a flat sigma of %.4f" % (what, sig))
    print("  %-6s %5s %9s %9s %9s   %s" % ("year", "n", "sd", "sd/sigma", "MAE", "sd by playing-time tertile (low/mid/high)"))
    for y in YEARS:
        proj = mk(load, y)
        act = load[load["year"] == y]
        d = resid(proj, act, "player", tcol, thr, pcol, acol, conv=conv)
        if len(d) < 30:
            print("  %-6d %5d  (too few)" % (y, len(d))); continue
        sd = float(d["_e"].std(ddof=1)); ma = float(d["_e"].abs().mean())
        q1, q2 = d["_pt"].quantile([1/3, 2/3])
        bins = [float(d[d["_pt"] <= q1]["_e"].std(ddof=1)),
                float(d[(d["_pt"] > q1) & (d["_pt"] <= q2)]["_e"].std(ddof=1)),
                float(d[d["_pt"] > q2]["_e"].std(ddof=1))]
        flag = " *120G" if y == 2020 else ""
        print("  %-6d %5d %9.4f %9.2f %9.4f   %.3f / %.3f / %.3f%s"
              % (y, len(d), sd, sd / sig, ma, bins[0], bins[1], bins[2], flag))
        rows.append(dict(what=what, year=y, n=len(d), sd=sd, ratio=sd / sig,
                         lo=bins[0], hi=bins[2], sig=sig))

df = pd.DataFrame(rows)
print("")
print("-" * 86)
print("VERDICT on the declared predictions")
print("-" * 86)
for what in df["what"].unique():
    s = df[df["what"] == what]
    p4 = bool((s["ratio"] < 1.0).all())
    p5 = bool((s["lo"] > s["hi"]).all())
    print("  %-12s P4 (sd < applied sigma in every year): %s   -- ratio range %.2f - %.2f, median %.2f"
          % (what, "HOLDS" if p4 else "FAILS", s["ratio"].min(), s["ratio"].max(), s["ratio"].median()))
    print("  %-12s P5 (sd falls with playing time every year): %s   -- %d of %d years"
          % ("", "HOLDS" if p5 else "FAILS", int((s["lo"] > s["hi"]).sum()), len(s)))
    ex = s[s["year"] != 2020]
    print("  %-12s excluding the 120-game 2020: ratio median %.2f (n=%d years); "
          "sigma should be about %.4f not %.4f"
          % ("", ex["ratio"].median(), len(ex), ex["sd"].median(), s["sig"].iloc[0]))
print("")
print("  2026 for comparison: hitter sd(z)=0.624 -> ratio 0.62 ; pitcher sd(z)=0.640 -> ratio 0.64")
df.to_csv("years_resid.csv", index=False)
print("  wrote years_resid.csv")
