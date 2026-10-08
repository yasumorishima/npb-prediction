# npb-stan-research

NPB (Nippon Professional Baseball) player performance projection using Bayesian (Stan) methods.

Successor to [npb-prediction](https://github.com/yasumorishima/npb-prediction)'s Marcel method. Adds **Bayesian corrections** to improve accuracy for both Japanese players and foreign first-year players.

## Results

### Player-Level Accuracy (LOO-CV, 2018-2025)

| Model | n | Marcel MAE | Stan MAE | p-value | Bootstrap P |
|---|---|---|---|---|---|
| **Japanese hitters (wOBA)** | 2,208 | 0.05023 | **0.04980** | **p=0.060** | 97.1% |
| **Japanese pitchers (ERA)** | 2,164 | 1.23008 | **1.22241** | **p=0.057** | 97.1% |
| Foreign first-year hitters (wOBA) | 78 | 0.0387 | **0.0385** | p=0.914 | — |
| Foreign first-year pitchers (ERA) | 91 | 0.8457 | 0.8733 | p=0.289 | — |

Stan improves both hitter wOBA and pitcher ERA at near-significant levels (p~0.06, bootstrap 97%).

### Team-Level Accuracy (LOO-CV, 2018-2025, n=96)

| Model | Win MAE | vs Marcel |
|---|---|---|
| Marcel (baseline) | 6.725 wins | — |
| Stan (+K%/BB%/BABIP) | 6.923 wins | +0.198 |

Stan's player-level improvement doesn't fully translate to team wins due to PA-weighting: Marcel is already accurate for high-PA regulars, while Stan's advantage concentrates on low-PA players who contribute less to team totals.

### Key Findings

- **Japanese pitchers**: K% and K/9 provide signal beyond Marcel ERA (ERA+5feat: **p=0.012**). ERA is influenced by luck (BABIP); K% is more stable year-to-year.
- **Japanese hitters**: K%/BB% don't add beyond Marcel wOBA (wOBA already incorporates these). BABIP (luck component) does add signal (`delta_BABIP = -0.006`): high BABIP in t-1 → Marcel overestimates year t. **p=0.0004** with full features.
- **Foreign first-year players**: K%/BB% from previous league add signal, but sample size is too small for statistical significance.
- **PA quartile analysis**: Stan wins most in Q1 (low PA, 30-64): 55% win rate, MAE +0.003. High-PA regulars (Q4): ~tied.

### Team Backtest (Marcel baseline + Park Factor correction, 2018-2025)

Park factor (PF_5yr) correction is applied before Pythagorean win% calculation to remove stadium bias embedded in Marcel projections.

| Metric | No PF | Single-year PF | PF_5yr | Δ (PF_5yr) |
|---|---|---|---|---|
| Win MAE | 6.41 | 6.41 | **6.41** | ±0.00 |
| Bias | +2.69 | +2.70 | +2.70 | +0.01 |
| RS MAE | 101.1 | **74.8** | **74.8** | **-26.3** |
| RA MAE | 97.5 | **73.0** | **73.0** | **-24.5** |
| 80% CI coverage | 86.5% | 86.5% | **87.5%** | **+1.0%** |

Win MAE is structurally unchanged: dividing RS and RA by the same park factor preserves the Pythagorean ratio. RS/RA absolute accuracy improves by ~25 runs. CI coverage improves with PF_5yr (smoothed) but not single-year PF (too noisy). Effect expected to grow as Vantelin Dome and Rakuten Mobile Park undergo renovations in 2026.

---

## Approach

### Japanese Players (all NPB players)

```
Hitter:  actual_wOBA = Marcel_wOBA + δ_K·z_K + δ_BB·z_BB + δ_BABIP·z_BABIP + noise
Pitcher: actual_ERA  = Marcel_ERA  + δ_K·z_K + δ_BB·z_BB + noise
```

- Marcel 3-year weighted average (weights 5:4:3) as prior mean
- Features: K%, BB%, BABIP, age_from_peak, pa/ip_stability, prev_woba_dev_sq/prev_babip_p
- Leave-one-out CV across 2018-2025 (8 years, n=2,208 hitters / 2,164 pitchers)

### Foreign First-Year Players

#### v1 (current baseline)

```
Hitter:  npb_wOBA = lg_avg + β_woba·z_woba + β_K·z_K + β_BB·z_BB + noise
Pitcher: npb_ERA  = lg_avg + β_era·z_era + β_fip·z_fip + β_K·z_K + β_BB·z_BB + noise
```

- League conversion factors (MLB→NPB, KBO→NPB, etc.) as prior
- K%/BB%/FIP from previous league as features
- Training: 2015-2019 | Backtest: 2020-2025
- **Problem**: Uniform coefficients (w≈0.14) → all players regress to league average regardless of skill level

#### v2 (in progress)

v1 applies the same weight to all foreign players. A strong MLB hitter with low K% and high BB% gets the same prediction as a marginal AAA player. v2 fixes this with 7 improvements:

| Improvement | Feature | v1 | v2 |
|---|---|---|---|
| 1. League-specific trust | `beta_woba[league]` | single β for all | **separate β for MLB / AAA / Other** |
| 2. Non-linearity | `z_woba²` | linear only | **quadratic term** (extreme values matter more) |
| 3. Interaction | `z_K × z_BB` | none | **low K% + high BB% combo evaluated** |
| 4. Age correction | `age_from_peak` | none | **30+ discounted** |
| 5. Position | `is_catcher`, `is_middle_inf` | none | **catcher / middle infield adjustment** |
| 6. NPB adaptation | `is_second_year` | first year only | **2nd year boost estimated** |
| 7. Sample size | `gamma_pa × log(PA)` | uniform σ | **fewer PA → wider uncertainty** |

v2 preliminary results (full model fit, LOO-CV in progress):
- **K% × BB% interaction is the strongest signal** (`beta_K_BB = +0.011 ± 0.005`)
- Non-linear effect confirmed (`beta_woba_sq = +0.006 ± 0.003`)
- 2nd year players improve by ~+.010 wOBA (`beta_second_year = +0.010 ± 0.008`)

#### v3 (planned)

Add MLB Statcast features for MLB-origin players (252 of 393):
- Exit velocity, barrel rate, xwOBA, Whiff%, Stuff+, sprint speed
- Same features that achieved +12.1% improvement over Marcel in [baseball-mlops](https://github.com/yasumorishima/baseball-mlops)

### Team Projection

```
team_RS = K_WOBA × Σ_player(Stan_wOBA × actual_PA)
team_RA = Σ_player(Stan_ERA × actual_IP / 9)
W = Pythagorean(RS, RA, exp=1.83)
```

Monte Carlo simulation (N=10,000) propagates player uncertainty to team win distribution.

---

## League-to-NPB Conversion Factors

| League | wOBA ratio [95%CI] (n) | ERA ratio [95%CI] (n) |
|---|---|---|
| **MLB** | 1.235 [1.14, 1.32] (56) | 0.579 [0.51, 0.67] (74) |
| **AAA** | 1.271 [1.01, 1.47] (9) | 0.462 [0.25, 0.69] (6) |

*Interpretation*: MLB hitters need ~24% higher wOBA to match NPB; MLB pitchers ERA inflates ~42% in NPB.

---

## Blog Articles

**Park Factor series:**
- [球場補正を加えたらNPB予測は改善したか — ベイズ順位予測への追加検証（Zenn）](https://zenn.dev/shogaku/articles/npb-bayes-pf-validation)
- [Did Adding Stadium Correction Improve My NPB Baseball Predictions? (DEV.to)](https://dev.to/yasumorishima/npb-bayes-pf-validation)
- [NPBベイズ順位予測にパークファクター補正を追加した（Zenn）](https://zenn.dev/shogaku/articles/npb-bayes-park-factors)

**Bayesian model series:**
- [Beyond Marcel: Adding Bayesian Regression to NPB Baseball Predictions (DEV.to)](https://dev.to/yasumorishima/beyond-marcel-adding-bayesian-regression-to-npb-baseball-predictions-a-15-step-journey-1b4f)
- [Marcel法の限界を超えたい — NPBベイズ回帰15ステップの記録（Zenn）](https://zenn.dev/shogaku/articles/npb-bayes-projection-story)
- [Did Bayesian Projection (Stan/Ridge) Predict the 2021 NPB Last-to-First Upsets? (DEV.to)](https://dev.to/yasumorishima/did-bayesian-projection-stanridge-predict-the-2021-npb-last-to-first-upsets-4595)
- [ベイズ予測（Stan/Ridge）で2021年ヤクルト・オリックスの優勝は見えたか（Zenn）](https://zenn.dev/shogaku/articles/npb-bayes-lastplace-to-champion)

## Data Sources

- **NPB stats**: [baseball-data.com](https://baseball-data.com/) + [npb.jp](https://npb.jp/)
- **MLB/MiLB stats**: [FanGraphs](https://www.fangraphs.com/) via [pybaseball](https://github.com/jldbc/pybaseball)

## Project Structure

```
npb-bayes-projection/
├── data/
│   ├── raw/              # NPB CSVs (hitters, pitchers, sabermetrics 2015-2025)
│   ├── foreign/          # Foreign player data & conversion factors
│   ├── model/            # Stan model outputs (predictions, comparison JSON)
│   └── projections/      # Team simulation outputs
├── models/
│   ├── hitter_jpn.stan          # Japanese hitter Stan model (Marcel + K%/BB%/BABIP)
│   ├── pitcher_jpn.stan         # Japanese pitcher Stan model (Marcel + K%/BB%)
│   ├── hitter.stan              # Foreign first-year hitter Stan model (v1)
│   ├── pitcher.stan             # Foreign first-year pitcher Stan model (v1)
│   ├── hitter_foreign_v2.stan   # Foreign hitter v2 (7 improvements)
│   └── pitcher_foreign_v2.stan  # Foreign pitcher v2 (7 improvements)
├── src/
│   ├── stan_jpn_model.py        # Japanese player model runner
│   ├── stan_model.py            # Foreign player model runner (v1)
│   ├── foreign_v2_model.py      # Foreign player v2 (data prep + Stan fit + LOO-CV)
│   ├── statistical_validation.py # LOO-CV + paired t-test + bootstrap (Steps 10-15)
│   ├── diagnose_big_misses.py   # YoY analysis of |err|>10W team-years
│   ├── analyze_coverage_gap.py  # PA/IP coverage gap diagnosis
│   ├── team_compare.py          # Marcel vs Stan team-level MAE comparison
│   ├── team_sim.py              # Monte Carlo team standings simulation (+ PF backtest)
│   ├── compare_pf_methods.py    # 3-way park factor comparison (No PF / single-year / 5yr avg)
│   ├── analyze_pf_comparison.py # Park factor backtest analysis for blog
│   ├── build_conversion_factors.py
│   └── identify_foreign_players.py
└── .github/workflows/
    └── build_factors.yml    # All pipeline steps via workflow_dispatch
```

## Running via GitHub Actions

```bash
gh workflow run build_factors.yml -f step=run_jpn_model           # Japanese player Stan model
gh workflow run build_factors.yml -f step=run_stan_model          # Foreign player Stan model
gh workflow run build_factors.yml -f step=statistical_validation  # LOO-CV + significance tests (Steps 10-15)
gh workflow run build_factors.yml -f step=diagnose_big_misses     # Team-year big miss analysis
gh workflow run build_factors.yml -f step=team_compare            # Marcel vs Stan comparison
gh workflow run build_factors.yml -f step=team_sim                # Team standings simulation (2026)
gh workflow run build_factors.yml -f step=team_backtest           # Backtest 2018-2025
gh workflow run build_factors.yml -f step=compare_pf_methods      # No PF vs single-year vs PF_5yr
gh workflow run build_factors.yml -f step=analyze_pf              # Park factor analysis report
gh workflow run build_factors.yml -f step=foreign_v2              # Foreign v2 — fit full model
gh workflow run build_factors.yml -f step=foreign_v2_loo          # Foreign v2 — LOO-CV (~2h)
gh workflow run build_factors.yml -f step=foreign_v2_expanding    # Foreign v2 — expanding-window CV
```

**CI タイムアウト保護**: ワークフローに `timeout-minutes: 360` を設定。全 9 スクリプト（Stan/PyMC/LOO-CV）に `PYTHONUNBUFFERED=1` + ステップ別経過時間ログを追加し、長時間 MCMC サンプリングのハング検知を容易にしている。

## License

MIT
