"""Re-run team_simulation.simulate() on the committed inputs and compare with the
frozen data/projections/team_sim_<year>.csv. Does NOT call team_simulation.main()
(main() would overwrite the frozen CSV). Run from the repository root:
    python research/team_2026/audit/repro_team_sim.py 2025   # checks team_sim_2026.csv
    python research/team_2026/audit/repro_team_sim.py 2020   # checks team_sim_2021.csv
Expected: "max abs diff vs frozen CSV 0.0" for both (seed 42, N_SIM 10,000)."""
import os, sys
os.environ["NPB_DATA_END_YEAR"] = sys.argv[1]
sys.path.insert(0, ".")
import pandas as pd, numpy as np
import team_simulation as T
h = T.normalize_pa(T.load_bayes_hitters()); p = T.normalize_ip(T.load_bayes_pitchers())
fh, fp = T.load_foreign_hitters(), T.load_foreign_pitchers()
pf = T.load_park_factors()
print("TARGET", T.TARGET_YEAR, "rows", len(h), len(p), len(fh), len(fp), "pf", len(pf))
res = T.compute_probabilities(T.simulate(h, p, fh, fp, park_factors=pf))
fr = pd.read_csv(f"data/projections/team_sim_{T.TARGET_YEAR}.csv", encoding="utf-8-sig").set_index("team")
mx = 0.0
for t, v in res.items():
    got = [v["p_pennant"], v["p_cs"], v["p_last"], v["median_wins"], v["mean_wins"], *v["wins_80ci"], *v["wins_95ci"]]
    exp = fr.loc[t, ["p_pennant","p_cs","p_last","median_wins","mean_wins","wins_80ci_lo","wins_80ci_hi","wins_95ci_lo","wins_95ci_hi"]].astype(float).tolist()
    mx = max(mx, max(abs(a-b) for a, b in zip(got, exp)))
print("teams", len(res), "max abs diff vs frozen CSV", mx)
