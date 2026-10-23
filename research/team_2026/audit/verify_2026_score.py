"""Independent (standard-library only) recomputation of the 2026 team score
T1-T5 and F1-F3, written separately from score_team.py as a cross-check.
Run from the repository root:
    python research/team_2026/audit/verify_2026_score.py
Source: baseball-data.com (standings)."""
import csv
R = ""  # run from the repository root
def rd(p):
    with open(p,encoding="utf-8-sig") as f: return list(csv.DictReader(f))
fc={r["team"]:r for r in rd(R+"data/projections/team_sim_2026.csv")}
st={r["team"]:r for r in rd(R+"research/team_2026/standings_2026.csv")}
h=[r for r in rd(R+"data/raw/npb_standings_2015_2025.csv") if r["year"]=="2025"]
h={r["team"]:r for r in h}
T=list(fc); assert set(T)==set(st)==set(h) and len(T)==12
W={t:int(st[t]["W"]) for t in T}
mae=lambda p: sum(abs(p[t]-W[t]) for t in T)/12
print("T1",mae({t:float(fc[t]["median_wins"]) for t in T}),"mean",mae({t:float(fc[t]["mean_wins"]) for t in T}))
avg=sum(int(h[t]["W"]) for t in T)/12
print("F1",mae({t:avg for t in T}),avg)
print("F2",mae({t:int(h[t]["W"]) for t in T}))
k=1.72
def py(t): rs,ra=float(h[t]["RS"]),float(h[t]["RA"]); return rs**k/(rs**k+ra**k)
print("F3 WL",mae({t:py(t)*(int(h[t]["W"])+int(h[t]["L"])) for t in T}))
print("F3 WL rounded1",mae({t:round(py(t)*(int(h[t]["W"])+int(h[t]["L"])),1) for t in T}))
print("F3 G",mae({t:py(t)*int(h[t]["G"]) for t in T}))
def rank(xs):
    s=sorted(range(len(xs)),key=lambda i:xs[i]); r=[0]*len(xs); i=0
    while i<len(s):
        j=i
        while j+1<len(s) and xs[s[j+1]]==xs[s[i]]: j+=1
        for q in range(i,j+1): r[s[q]]=(i+j)/2+1
        i=j+1
    return r
def sp(ts):
    a=rank([float(fc[t]["median_wins"]) for t in ts]); b=rank([W[t] for t in ts])
    ma=sum(a)/len(a); mb=sum(b)/len(b)
    c=sum((x-ma)*(y-mb) for x,y in zip(a,b)); return c/(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))**.5
print("T2 all",sp(T))
for L in ["CL","PL"]:
    ts=[t for t in T if fc[t]["league"]==L]
    pm=[float(fc[t]["median_wins"]) for t in ts]; aw=[W[t] for t in ts]
    print(L,sp(ts),min(pm),max(pm),min(aw),max(aw), max(pm)-min(pm), max(aw)-min(aw))
    # champion/last
    lg=[st[t] for t in ts]
    print(" champ",max(ts,key=lambda t:int(st[t]["W"])-int(st[t]["L"])), " last", sorted(ts,key=lambda t: int(st[t]["W"])/(int(st[t]["W"])+int(st[t]["L"])))[:2], [ (t,st[t]["rank"]) for t in ts])
champ={"阪神","ソフトバンク"}; last={"中日","楽天"}
print("T3",sum((float(fc[t]["p_pennant"])-(t in champ))**2 for t in T)/12)
print("T4",sum((float(fc[t]["p_last"])-(t in last))**2 for t in T)/12, "uniform",sum(((1/6)-(t in last))**2 for t in T)/12)
for lv in ["80","95"]:
    ins=[t for t in T if float(fc[t][f"wins_{lv}ci_lo"])<=W[t]<=float(fc[t][f"wins_{lv}ci_hi"])]
    out=[(t, "hi" if W[t]>float(fc[t][f"wins_{lv}ci_hi"]) else "lo") for t in T if t not in ins]
    print(lv,len(ins),out)
