// Foreign hitter NPB first-year wOBA prediction — v5 (Mixture of Experts)
//
// Two-stage model:
//   Stage 1 (Gate): P(bust) via logistic regression on career/team features
//   Stage 2: Mixture likelihood
//     - Bust component:  Normal(mu_bust, sigma_bust)
//     - Performance component: Student-t(nu, mu_perf, sigma_perf) with v2 regression
//
// Key innovations over v2/v4a:
//   - Explicitly models bimodal outcome distribution
//   - New features: career total PA, team-specific bust rate
//   - Bust probability output per player

data {
  int<lower=0> N;
  int<lower=1> L;                     // number of origin leagues (3)
  array[N] int<lower=1, upper=L> league;
  vector[N] y;                        // NPB wOBA (actual)
  vector[N] lg_avg;                   // NPB league average wOBA for that year

  // Performance component features (same as v2)
  vector[N] z_woba;
  vector[N] z_K;
  vector[N] z_BB;
  vector[N] z_woba_sq;
  vector[N] z_K_BB;
  vector[N] z_age;
  vector[N] is_catcher;
  vector[N] is_middle_inf;
  vector[N] z_log_pa;
  vector[N] is_second_year;

  // Gate features (new for v5)
  vector[N] z_career_pa;              // z-scored log(career total PA)
  vector[N] z_team_bust;              // z-scored team bust rate
}

parameters {
  // --- Gate (bust probability) ---
  vector[L] alpha_gate;               // league-specific bust intercept
  real beta_gate_career;              // career PA effect (more PA → less bust)
  real beta_gate_pa;                  // prev-season PA effect
  real beta_gate_bb;                  // BB% effect (plate discipline)
  real beta_gate_team;                // team bust rate effect

  // --- Bust distribution ---
  real<upper=0.280> mu_bust;          // bust wOBA mean (constrained to prevent label switching)
  real<lower=0> sigma_bust;           // bust wOBA spread

  // --- Performance distribution (v2 + Student-t) ---
  vector[L] beta_woba;
  real beta_K;
  real beta_BB;
  real beta_woba_sq;
  real beta_K_BB;
  real beta_age;
  real beta_catcher;
  real beta_middle_inf;
  real beta_second_year;
  real<lower=0> sigma_base;
  real gamma_pa;
  real<lower=2> nu;                   // Student-t degrees of freedom
}

model {
  // --- Gate priors ---
  // base bust rate ~18% for hitters → logit(0.18) ≈ -1.5
  alpha_gate ~ normal(-1.5, 1.0);
  beta_gate_career ~ normal(-0.5, 0.3);   // more career PA → less bust
  beta_gate_pa ~ normal(-0.3, 0.3);       // more prev PA → less bust
  beta_gate_bb ~ normal(-0.2, 0.3);       // higher BB% → less bust
  beta_gate_team ~ normal(0.5, 0.5);      // higher team bust rate → more bust

  // --- Bust distribution priors ---
  mu_bust ~ normal(0.200, 0.03);
  sigma_bust ~ exponential(20);            // E[sigma_bust] = 0.05

  // --- Performance distribution priors (same as v2) ---
  beta_woba ~ normal(0, 0.10);
  beta_K ~ normal(0, 0.05);
  beta_BB ~ normal(0, 0.05);
  beta_woba_sq ~ normal(0, 0.03);
  beta_K_BB ~ normal(0, 0.03);
  beta_age ~ normal(-0.003, 0.02);
  beta_catcher ~ normal(0, 0.03);
  beta_middle_inf ~ normal(0, 0.03);
  beta_second_year ~ normal(0.005, 0.02);
  sigma_base ~ exponential(15);
  gamma_pa ~ normal(0, 0.3);
  nu ~ gamma(2, 0.1);                     // Student-t df

  // --- Mixture likelihood ---
  for (n in 1:N) {
    // Gate: log-odds of bust
    real gate_n = alpha_gate[league[n]]
      + beta_gate_career * z_career_pa[n]
      + beta_gate_pa * z_log_pa[n]
      + beta_gate_bb * z_BB[n]
      + beta_gate_team * z_team_bust[n];

    // Performance component mean (v2 regression)
    real mu_perf = lg_avg[n]
      + beta_woba[league[n]] * z_woba[n]
      + beta_K * z_K[n]
      + beta_BB * z_BB[n]
      + beta_woba_sq * z_woba_sq[n]
      + beta_K_BB * z_K_BB[n]
      + beta_age * z_age[n]
      + beta_catcher * is_catcher[n]
      + beta_middle_inf * is_middle_inf[n]
      + beta_second_year * is_second_year[n];

    real sigma_perf = sigma_base * exp(fmax(fmin(gamma_pa * z_log_pa[n], 2.0), -5.0));

    // Mixture: log_sum_exp for numerical stability
    target += log_sum_exp(
      log_inv_logit(gate_n) + normal_lpdf(y[n] | mu_bust, sigma_bust),
      log1m_inv_logit(gate_n) + student_t_lpdf(y[n] | nu, mu_perf, sigma_perf)
    );
  }
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  vector[N] p_bust;          // posterior bust probability per player

  for (n in 1:N) {
    real gate_n = alpha_gate[league[n]]
      + beta_gate_career * z_career_pa[n]
      + beta_gate_pa * z_log_pa[n]
      + beta_gate_bb * z_BB[n]
      + beta_gate_team * z_team_bust[n];

    real mu_perf = lg_avg[n]
      + beta_woba[league[n]] * z_woba[n]
      + beta_K * z_K[n]
      + beta_BB * z_BB[n]
      + beta_woba_sq * z_woba_sq[n]
      + beta_K_BB * z_K_BB[n]
      + beta_age * z_age[n]
      + beta_catcher * is_catcher[n]
      + beta_middle_inf * is_middle_inf[n]
      + beta_second_year * is_second_year[n];

    real sigma_perf = sigma_base * exp(fmax(fmin(gamma_pa * z_log_pa[n], 2.0), -5.0));

    // Prior bust probability from gate
    real pi_n = inv_logit(gate_n);

    // Posterior bust probability (Bayes update with observed y)
    real lp_bust = log(pi_n) + normal_lpdf(y[n] | mu_bust, sigma_bust);
    real lp_perf = log1m(pi_n) + student_t_lpdf(y[n] | nu, mu_perf, sigma_perf);
    p_bust[n] = exp(lp_bust - log_sum_exp(lp_bust, lp_perf));

    // Mixture log-likelihood
    log_lik[n] = log_sum_exp(lp_bust, lp_perf);

    // Posterior predictive: sample from mixture
    if (bernoulli_rng(pi_n)) {
      y_rep[n] = normal_rng(mu_bust, sigma_bust);
    } else {
      y_rep[n] = student_t_rng(nu, mu_perf, sigma_perf);
    }
  }
}
