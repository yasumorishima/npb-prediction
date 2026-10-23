"""Width of the frozen team intervals (2026, 2021), 2021 coverage k80 / k95,
binomial reference widths, and draws per year. Numbers used in the addendum of
data/backtest/2026_team_result.md. Run from the repository root:
    python research/team_2026/audit/interval_width.py
Source: baseball-data.com (standings)."""
import numpy as np, pandas as pd
from scipy.stats import binom
n=143
print("binomial n=143")
for p in [0.45,0.5,0.55]:
    lo10,hi90=binom.ppf(0.10,n,p),binom.ppf(0.90,n,p)
    lo25,hi975=binom.ppf(0.025,n,p),binom.ppf(0.975,n,p)
    sd=np.sqrt(n*p*(1-p))
    print(f"p={p}: sd={sd:.2f} 80%[{lo10:.0f},{hi90:.0f}] w={hi90-lo10:.0f}  95%[{lo25:.0f},{hi975:.0f}] w={hi975-lo25:.0f}  normal80w={2*1.2816*sd:.1f} normal95w={2*1.96*sd:.1f}")
base = ""  # run from the repository root
for yr in [2026,2021]:
    d=pd.read_csv(base+f"data/projections/team_sim_{yr}.csv",encoding="utf-8-sig")
    w80=d.wins_80ci_hi-d.wins_80ci_lo; w95=d.wins_95ci_hi-d.wins_95ci_lo
    print(yr,"n",len(d),"80 mean",round(w80.mean(),2),"range",round(w80.min(),1),round(w80.max(),1),"95 mean",round(w95.mean(),2),"range",round(w95.min(),1),round(w95.max(),1))
st=pd.read_csv(base+"data/raw/npb_standings_2015_2025.csv",encoding="utf-8-sig")
s21=st[st.year==2021]
d21=pd.read_csv(base+"data/projections/team_sim_2021.csv",encoding="utf-8-sig")
m=d21.merge(s21[["team","W","L","D","G"]],on="team",how="left")
print(m[["team","W","D","wins_80ci_lo","wins_80ci_hi","wins_95ci_lo","wins_95ci_hi","median_wins"]])
in80=((m.W>=m.wins_80ci_lo)&(m.W<=m.wins_80ci_hi)); in95=((m.W>=m.wins_95ci_lo)&(m.W<=m.wins_95ci_hi))
print("2021 k80",in80.sum(),"k95",in95.sum(),"unmatched",m.W.isna().sum())
print("draws by year:"); print(st.groupby("year").agg(D_sum=("D","sum"),D_max=("D","max"),G_mean=("G","mean"),W_mean=("W","mean"),W_sd=("W","std")))
