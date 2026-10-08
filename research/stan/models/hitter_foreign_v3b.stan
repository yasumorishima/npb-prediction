// Foreign hitter NPB first-year wOBA prediction — v3b
//
// Key change from v2: league-specific affine transformation on raw prev_woba
//   v2: mu = lg_avg + beta[league] * z_woba + ...
//   v3b: mu = alpha[league] + beta[league] * prev_woba_raw + ...
//
// Non-centered parameterization (NCP) to avoid divergent transitions
// in the hierarchical structure.

data {
  int<lower=0> N;
  int<lower=1> L;                     // number of origin leagues (3: MLB/AAA/Other)
  array[N] int<lower=1, upper=L> league;
  vector[N] y;                        // NPB wOBA (actual)
  vector[N] prev_woba;                // raw previous-league wOBA (NOT z-scored)
  vector[N] z_K;                      // z-scored previous-league K%
  vector[N] z_BB;                     // z-scored previous-league BB%
  vector[N] z_K_BB;                   // z_K * z_BB (interaction)
  vector[N] z_age;                    // z-scored age_from_peak
  vector[N] is_catcher;               // 1 if catcher
  vector[N] is_middle_inf;            // 1 if SS or 2B
  vector[N] z_log_pa;                 // z-scored log(previous-league PA)
  vector[N] is_second_year;           // 1 if NPB 2nd year
}

parameters {
  // Hierarchical hyperparameters
  real alpha_mu;
  real<lower=0> alpha_sigma;
  vector[L] alpha_raw;                // NCP: alpha = alpha_mu + alpha_sigma * alpha_raw

  real beta_woba_mu;
  real<lower=0> beta_woba_sigma;
  vector[L] beta_woba_raw;            // NCP: beta_woba = beta_woba_mu + beta_woba_sigma * beta_woba_raw

  // Auxiliary features (league-independent)
  real beta_K;
  real beta_BB;
  real beta_K_BB;
  real beta_age;
  real beta_catcher;
  real beta_middle_inf;
  real beta_second_year;
  real<lower=0> sigma_base;
  real gamma_pa;
}

transformed parameters {
  vector[L] alpha = alpha_mu + alpha_sigma * alpha_raw;
  vector[L] beta_woba = beta_woba_mu + beta_woba_sigma * beta_woba_raw;
}

model {
  // Hierarchical priors
  alpha_mu ~ normal(0.310, 0.05);
  alpha_sigma ~ exponential(20);
  alpha_raw ~ std_normal();

  beta_woba_mu ~ normal(0.3, 0.2);
  beta_woba_sigma ~ exponential(5);
  beta_woba_raw ~ std_normal();

  // Auxiliary priors
  beta_K ~ normal(0, 0.05);
  beta_BB ~ normal(0, 0.05);
  beta_K_BB ~ normal(0, 0.03);
  beta_age ~ normal(-0.003, 0.02);
  beta_catcher ~ normal(0, 0.03);
  beta_middle_inf ~ normal(0, 0.03);
  beta_second_year ~ normal(0.005, 0.02);
  sigma_base ~ exponential(15);
  gamma_pa ~ normal(0, 0.3);

  // Likelihood
  for (n in 1:N) {
    real mu_n = alpha[league[n]]
      + beta_woba[league[n]] * prev_woba[n]
      + beta_K * z_K[n]
      + beta_BB * z_BB[n]
      + beta_K_BB * z_K_BB[n]
      + beta_age * z_age[n]
      + beta_catcher * is_catcher[n]
      + beta_middle_inf * is_middle_inf[n]
      + beta_second_year * is_second_year[n];

    real sigma_n = sigma_base * exp(fmax(fmin(gamma_pa * z_log_pa[n], 2.0), -5.0));
    y[n] ~ normal(mu_n, sigma_n);
  }
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  for (n in 1:N) {
    real mu_n = alpha[league[n]]
      + beta_woba[league[n]] * prev_woba[n]
      + beta_K * z_K[n]
      + beta_BB * z_BB[n]
      + beta_K_BB * z_K_BB[n]
      + beta_age * z_age[n]
      + beta_catcher * is_catcher[n]
      + beta_middle_inf * is_middle_inf[n]
      + beta_second_year * is_second_year[n];
    real sigma_n = sigma_base * exp(fmax(fmin(gamma_pa * z_log_pa[n], 2.0), -5.0));
    y_rep[n] = normal_rng(mu_n, sigma_n);
    log_lik[n] = normal_lpdf(y[n] | mu_n, sigma_n);
  }
}
