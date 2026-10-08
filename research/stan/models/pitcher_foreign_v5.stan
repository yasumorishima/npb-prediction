// Foreign pitcher NPB first-year ERA prediction — v5 (Mixture of Experts)
//
// Two-stage model:
//   Stage 1 (Gate): P(bust) via logistic regression on career/team features
//   Stage 2: Mixture likelihood
//     - Bust component:  Normal(mu_bust, sigma_bust)
//     - Performance component: Student-t(nu, mu_perf, sigma_perf) with v2 regression
//
// For pitchers, bust = ERA > 5.0 (22% of foreign pitchers)

data {
  int<lower=0> N;
  int<lower=1> L;
  array[N] int<lower=1, upper=L> league;
  vector[N] y;                        // NPB ERA (actual)
  vector[N] lg_avg;                   // NPB league average ERA for that year

  // Performance component features (same as v2)
  vector[N] z_era;
  vector[N] z_fip;
  vector[N] z_K;
  vector[N] z_BB;
  vector[N] z_era_sq;
  vector[N] z_K_BB;
  vector[N] z_age;
  vector[N] is_reliever;
  vector[N] z_log_ip;
  vector[N] is_second_year;

  // Gate features (new for v5)
  vector[N] z_career_ip;              // z-scored log(career total IP)
  vector[N] z_team_bust;              // z-scored team bust rate
}

parameters {
  // --- Gate (bust probability) ---
  vector[L] alpha_gate;
  real beta_gate_career;              // career IP effect
  real beta_gate_ip;                  // prev-season IP effect
  real beta_gate_k;                   // K% effect (better K → less bust)
  real beta_gate_team;                // team bust rate effect

  // --- Bust distribution ---
  real<lower=4.0> mu_bust;            // bust ERA mean (constrained to prevent label switching)
  real<lower=0> sigma_bust;

  // --- Performance distribution (v2 + Student-t) ---
  vector[L] beta_era;
  real beta_fip;
  real beta_K;
  real beta_BB;
  real beta_era_sq;
  real beta_K_BB;
  real beta_age;
  real beta_reliever;
  real beta_second_year;
  real<lower=0> sigma_base;
  real gamma_ip;
  real<lower=2> nu;
}

model {
  // --- Gate priors ---
  // base bust rate ~22% for pitchers → logit(0.22) ≈ -1.27
  alpha_gate ~ normal(-1.3, 1.0);
  beta_gate_career ~ normal(-0.5, 0.3);
  beta_gate_ip ~ normal(-0.3, 0.3);
  beta_gate_k ~ normal(-0.3, 0.3);        // higher K% → less bust
  beta_gate_team ~ normal(0.5, 0.5);

  // --- Bust distribution priors ---
  mu_bust ~ normal(5.8, 1.0);
  sigma_bust ~ exponential(0.5);           // E[sigma_bust] = 2.0

  // --- Performance distribution priors (same as v2) ---
  beta_era ~ normal(0, 0.5);
  beta_fip ~ normal(0, 0.5);
  beta_K ~ normal(0, 0.3);
  beta_BB ~ normal(0, 0.3);
  beta_era_sq ~ normal(0, 0.15);
  beta_K_BB ~ normal(0, 0.15);
  beta_age ~ normal(0.05, 0.15);
  beta_reliever ~ normal(-0.3, 0.3);
  beta_second_year ~ normal(-0.1, 0.2);
  sigma_base ~ exponential(1);
  gamma_ip ~ normal(0, 0.3);
  nu ~ gamma(2, 0.1);

  // --- Mixture likelihood ---
  for (n in 1:N) {
    real gate_n = alpha_gate[league[n]]
      + beta_gate_career * z_career_ip[n]
      + beta_gate_ip * z_log_ip[n]
      + beta_gate_k * z_K[n]
      + beta_gate_team * z_team_bust[n];

    real mu_perf = lg_avg[n]
      + beta_era[league[n]] * z_era[n]
      + beta_fip * z_fip[n]
      + beta_K * z_K[n]
      + beta_BB * z_BB[n]
      + beta_era_sq * z_era_sq[n]
      + beta_K_BB * z_K_BB[n]
      + beta_age * z_age[n]
      + beta_reliever * is_reliever[n]
      + beta_second_year * is_second_year[n];

    real sigma_perf = sigma_base * exp(fmax(fmin(gamma_ip * z_log_ip[n], 2.0), -5.0));

    target += log_sum_exp(
      log_inv_logit(gate_n) + normal_lpdf(y[n] | mu_bust, sigma_bust),
      log1m_inv_logit(gate_n) + student_t_lpdf(y[n] | nu, mu_perf, sigma_perf)
    );
  }
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  vector[N] p_bust;

  for (n in 1:N) {
    real gate_n = alpha_gate[league[n]]
      + beta_gate_career * z_career_ip[n]
      + beta_gate_ip * z_log_ip[n]
      + beta_gate_k * z_K[n]
      + beta_gate_team * z_team_bust[n];

    real mu_perf = lg_avg[n]
      + beta_era[league[n]] * z_era[n]
      + beta_fip * z_fip[n]
      + beta_K * z_K[n]
      + beta_BB * z_BB[n]
      + beta_era_sq * z_era_sq[n]
      + beta_K_BB * z_K_BB[n]
      + beta_age * z_age[n]
      + beta_reliever * is_reliever[n]
      + beta_second_year * is_second_year[n];

    real sigma_perf = sigma_base * exp(fmax(fmin(gamma_ip * z_log_ip[n], 2.0), -5.0));

    real pi_n = inv_logit(gate_n);

    real lp_bust = log(pi_n) + normal_lpdf(y[n] | mu_bust, sigma_bust);
    real lp_perf = log1m(pi_n) + student_t_lpdf(y[n] | nu, mu_perf, sigma_perf);
    p_bust[n] = exp(lp_bust - log_sum_exp(lp_bust, lp_perf));

    log_lik[n] = log_sum_exp(lp_bust, lp_perf);

    if (bernoulli_rng(pi_n)) {
      y_rep[n] = normal_rng(mu_bust, sigma_bust);
    } else {
      y_rep[n] = student_t_rng(nu, mu_perf, sigma_perf);
    }
  }
}
