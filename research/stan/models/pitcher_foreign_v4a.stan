// Foreign pitcher NPB first-year ERA prediction — v4a (Student-t)
//
// Based on v2, replacing Normal likelihood with Student-t for
// robustness to outliers.

data {
  int<lower=0> N;
  int<lower=1> L;
  array[N] int<lower=1, upper=L> league;
  vector[N] y;                        // NPB ERA (actual)
  vector[N] lg_avg;                   // NPB league average ERA for that year
  vector[N] z_era;                    // z-scored previous-league ERA
  vector[N] z_fip;                    // z-scored previous-league FIP
  vector[N] z_K;                      // z-scored previous-league K%
  vector[N] z_BB;                     // z-scored previous-league BB%
  vector[N] z_era_sq;                // z_era^2 (non-linearity)
  vector[N] z_K_BB;                  // z_K * z_BB (interaction)
  vector[N] z_age;                    // z-scored age_from_peak
  vector[N] is_reliever;             // 1 if reliever (IP < 50 or role)
  vector[N] z_log_ip;                // z-scored log(previous-league IP)
  vector[N] is_second_year;          // 1 if NPB 2nd year
}

parameters {
  vector[L] beta_era;                // league-specific ERA coefficient
  real beta_fip;                      // FIP effect
  real beta_K;                        // K% effect (negative = better)
  real beta_BB;                       // BB% effect (positive = worse)
  real beta_era_sq;                  // non-linear ERA effect
  real beta_K_BB;                    // K% x BB% interaction
  real beta_age;                     // age effect
  real beta_reliever;                // reliever adjustment
  real beta_second_year;             // 2nd year adaptation
  real<lower=0> sigma_base;
  real gamma_ip;                     // IP scaling for heteroscedasticity
  real<lower=2> nu;                  // Student-t degrees of freedom
}

model {
  // --- Priors ---
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
  nu ~ gamma(2, 0.1);                // prior: mode ~10, allows heavy tails

  // --- Likelihood ---
  for (n in 1:N) {
    real mu_n = lg_avg[n]
      + beta_era[league[n]] * z_era[n]
      + beta_fip * z_fip[n]
      + beta_K * z_K[n]
      + beta_BB * z_BB[n]
      + beta_era_sq * z_era_sq[n]
      + beta_K_BB * z_K_BB[n]
      + beta_age * z_age[n]
      + beta_reliever * is_reliever[n]
      + beta_second_year * is_second_year[n];

    real sigma_n = sigma_base * exp(fmax(fmin(gamma_ip * z_log_ip[n], 2.0), -5.0));

    y[n] ~ student_t(nu, mu_n, sigma_n);
  }
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  for (n in 1:N) {
    real mu_n = lg_avg[n]
      + beta_era[league[n]] * z_era[n]
      + beta_fip * z_fip[n]
      + beta_K * z_K[n]
      + beta_BB * z_BB[n]
      + beta_era_sq * z_era_sq[n]
      + beta_K_BB * z_K_BB[n]
      + beta_age * z_age[n]
      + beta_reliever * is_reliever[n]
      + beta_second_year * is_second_year[n];
    real sigma_n = sigma_base * exp(fmax(fmin(gamma_ip * z_log_ip[n], 2.0), -5.0));
    y_rep[n] = student_t_rng(nu, mu_n, sigma_n);
    log_lik[n] = student_t_lpdf(y[n] | nu, mu_n, sigma_n);
  }
}
