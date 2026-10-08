// Foreign pitcher NPB first-year ERA prediction — v3b
//
// Main predictors: raw K% and BB% with league-specific affine transformation
// (K%/BB% have higher cross-league reproducibility than ERA)
//
// Non-centered parameterization (NCP) to avoid divergent transitions.

data {
  int<lower=0> N;
  int<lower=1> L;
  array[N] int<lower=1, upper=L> league;
  vector[N] y;                        // NPB ERA (target)
  vector[N] prev_K;                   // raw previous-league K%
  vector[N] prev_BB;                  // raw previous-league BB%
  vector[N] z_fip;                    // z-scored previous-league FIP (auxiliary)
  vector[N] z_K_BB;                   // z_K * z_BB interaction (z-score version)
  vector[N] z_age;
  vector[N] is_reliever;
  vector[N] z_log_ip;
  vector[N] is_second_year;
}

parameters {
  // K% hierarchical (NCP)
  real alpha_K_mu;
  real<lower=0> alpha_K_sigma;
  vector[L] alpha_K_raw;

  real beta_K_mu;
  real<lower=0> beta_K_sigma;
  vector[L] beta_K_raw;

  // BB% hierarchical (NCP)
  real alpha_BB_mu;
  real<lower=0> alpha_BB_sigma;
  vector[L] alpha_BB_raw;

  real beta_BB_mu;
  real<lower=0> beta_BB_sigma;
  vector[L] beta_BB_raw;

  // Auxiliary features (league-independent)
  real beta_fip;
  real beta_K_BB;
  real beta_age;
  real beta_reliever;
  real beta_second_year;
  real<lower=0> sigma_base;
  real gamma_ip;
}

transformed parameters {
  vector[L] alpha_K = alpha_K_mu + alpha_K_sigma * alpha_K_raw;
  vector[L] beta_K = beta_K_mu + beta_K_sigma * beta_K_raw;
  vector[L] alpha_BB = alpha_BB_mu + alpha_BB_sigma * alpha_BB_raw;
  vector[L] beta_BB = beta_BB_mu + beta_BB_sigma * beta_BB_raw;
}

model {
  // K% hierarchical priors
  alpha_K_mu ~ normal(3.50, 0.5);
  alpha_K_sigma ~ exponential(5);
  alpha_K_raw ~ std_normal();

  beta_K_mu ~ normal(-0.05, 0.03);
  beta_K_sigma ~ exponential(10);
  beta_K_raw ~ std_normal();

  // BB% hierarchical priors
  alpha_BB_mu ~ normal(0, 0.5);
  alpha_BB_sigma ~ exponential(5);
  alpha_BB_raw ~ std_normal();

  beta_BB_mu ~ normal(0.08, 0.05);
  beta_BB_sigma ~ exponential(10);
  beta_BB_raw ~ std_normal();

  // Auxiliary priors
  beta_fip ~ normal(0, 0.3);
  beta_K_BB ~ normal(0, 0.15);
  beta_age ~ normal(0.05, 0.15);
  beta_reliever ~ normal(-0.3, 0.3);
  beta_second_year ~ normal(-0.1, 0.2);
  sigma_base ~ exponential(1);
  gamma_ip ~ normal(0, 0.3);

  // Likelihood
  for (n in 1:N) {
    real mu_n = alpha_K[league[n]] + beta_K[league[n]] * prev_K[n]
              + alpha_BB[league[n]] + beta_BB[league[n]] * prev_BB[n]
              + beta_fip * z_fip[n]
              + beta_K_BB * z_K_BB[n]
              + beta_age * z_age[n]
              + beta_reliever * is_reliever[n]
              + beta_second_year * is_second_year[n];

    real sigma_n = sigma_base * exp(fmax(fmin(gamma_ip * z_log_ip[n], 2.0), -5.0));
    y[n] ~ normal(mu_n, sigma_n);
  }
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  for (n in 1:N) {
    real mu_n = alpha_K[league[n]] + beta_K[league[n]] * prev_K[n]
              + alpha_BB[league[n]] + beta_BB[league[n]] * prev_BB[n]
              + beta_fip * z_fip[n]
              + beta_K_BB * z_K_BB[n]
              + beta_age * z_age[n]
              + beta_reliever * is_reliever[n]
              + beta_second_year * is_second_year[n];
    real sigma_n = sigma_base * exp(fmax(fmin(gamma_ip * z_log_ip[n], 2.0), -5.0));
    y_rep[n] = normal_rng(mu_n, sigma_n);
    log_lik[n] = normal_lpdf(y[n] | mu_n, sigma_n);
  }
}
