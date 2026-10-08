// Foreign hitter NPB first-year wOBA prediction — v4a (Student-t)
//
// Based on v2, replacing Normal likelihood with Student-t for
// robustness to outliers (e.g. extreme busts/booms).
// Only change: normal → student_t with estimated nu.

data {
  int<lower=0> N;
  int<lower=1> L;                     // number of origin leagues
  array[N] int<lower=1, upper=L> league;
  vector[N] y;                        // NPB wOBA (actual)
  vector[N] lg_avg;                   // NPB league average wOBA for that year
  vector[N] z_woba;                   // z-scored previous-league wOBA
  vector[N] z_K;                      // z-scored previous-league K%
  vector[N] z_BB;                     // z-scored previous-league BB%
  vector[N] z_woba_sq;               // z_woba^2 (non-linearity)
  vector[N] z_K_BB;                  // z_K * z_BB (interaction)
  vector[N] z_age;                    // z-scored age_from_peak
  vector[N] is_catcher;              // 1 if catcher
  vector[N] is_middle_inf;           // 1 if SS or 2B
  vector[N] z_log_pa;                // z-scored log(previous-league PA)
  vector[N] is_second_year;          // 1 if NPB 2nd year
}

parameters {
  vector[L] beta_woba;               // league-specific wOBA coefficient
  real beta_K;                        // K% effect on NPB wOBA
  real beta_BB;                       // BB% effect on NPB wOBA
  real beta_woba_sq;                 // non-linear wOBA effect
  real beta_K_BB;                    // K% x BB% interaction
  real beta_age;                     // age penalty
  real beta_catcher;                 // catcher position effect
  real beta_middle_inf;              // middle infield position effect
  real beta_second_year;             // 2nd year adaptation boost
  real<lower=0> sigma_base;          // base noise
  real gamma_pa;                     // PA scaling for heteroscedasticity
  real<lower=2> nu;                  // Student-t degrees of freedom
}

model {
  // --- Priors ---
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
  nu ~ gamma(2, 0.1);                // prior: mode ~10, allows heavy tails

  // --- Likelihood ---
  for (n in 1:N) {
    real mu_n = lg_avg[n]
      + beta_woba[league[n]] * z_woba[n]
      + beta_K * z_K[n]
      + beta_BB * z_BB[n]
      + beta_woba_sq * z_woba_sq[n]
      + beta_K_BB * z_K_BB[n]
      + beta_age * z_age[n]
      + beta_catcher * is_catcher[n]
      + beta_middle_inf * is_middle_inf[n]
      + beta_second_year * is_second_year[n];

    real sigma_n = sigma_base * exp(fmax(fmin(gamma_pa * z_log_pa[n], 2.0), -5.0));

    y[n] ~ student_t(nu, mu_n, sigma_n);
  }
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  for (n in 1:N) {
    real mu_n = lg_avg[n]
      + beta_woba[league[n]] * z_woba[n]
      + beta_K * z_K[n]
      + beta_BB * z_BB[n]
      + beta_woba_sq * z_woba_sq[n]
      + beta_K_BB * z_K_BB[n]
      + beta_age * z_age[n]
      + beta_catcher * is_catcher[n]
      + beta_middle_inf * is_middle_inf[n]
      + beta_second_year * is_second_year[n];
    real sigma_n = sigma_base * exp(fmax(fmin(gamma_pa * z_log_pa[n], 2.0), -5.0));
    y_rep[n] = student_t_rng(nu, mu_n, sigma_n);
    log_lik[n] = student_t_lpdf(y[n] | nu, mu_n, sigma_n);
  }
}
