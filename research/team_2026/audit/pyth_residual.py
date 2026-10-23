"""Pythagorean residual sd of 2015-2025 (k = 1.72), used in the addendum of
data/backtest/2026_team_result.md (sd 3.9 wins). Run from the repository root:
    python research/team_2026/audit/pyth_residual.py
Source: baseball-data.com (standings)."""
import numpy as np, pandas as pd
base = ""  # run from the repository root
st=pd.read_csv(base+"data/raw/npb_standings_2015_2025.csv",encoding="utf-8-sig")
k=1.72
st["pyth_w"]=st.G*st.RS**k/(st.RS**k+st.RA**k)
st["pyth_w_dec"]=(st.W+st.L)*st.RS**k/(st.RS**k+st.RA**k)
for name,col in [("W-pyth(G)","pyth_w"),("W-pyth(W+L)","pyth_w_dec")]:
    r=st.W-st[col]
    r2=r[st.year!=2020]
    print(name,"sd all",round(r.std(),2),"sd excl2020",round(r2.std(),2),"rmse excl2020",round(np.sqrt((r2**2).mean()),2))
# resid 2021 only
print("2021 resid mean", round((st[st.year==2021].W-st[st.year==2021].pyth_w).mean(),2))
# implied sd of 2026 sim from 80% width 11.67 (normal): width/(2*1.2816)
print("implied sim sd 2026", round(11.67/(2*1.2816),2), "95-based", round(17.82/(2*1.96),2))
print("implied sim sd 2021", round(7.73/(2*1.2816),2))
