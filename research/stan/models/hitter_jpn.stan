// Japanese hitter year-ahead wOBA prediction
//
// Model: actual_wOBA = Marcel_wOBA + delta_K * z_K + delta_BB * z_BB
//                     + delta_BABIP * z_babip + delta_age * z_age + noise
//
// If deltas = 0, reduces to pure Marcel (baseline).
// K%, BB%, BABIP, age_from_peak are z-scored using training-set mean/sd.
//
// Expected signs:
//   delta_K     < 0  (high K%  → lower wOBA)
//   delta_BB    > 0  (high BB% → higher wOBA)
//   delta_BABIP < 0  (high BABIP in t-1 inflates Marcel; regression expected)
//   delta_age   < 0  (older hitters decline → lower wOBA)

data {
  int<lower=1> N;               // training observations
  vector[N] marcel_woba;        // Marcel-projected wOBA (prior mean)
  vector[N] z_K;                // z-scored K%    (SO / PA)
  vector[N] z_BB;               // z-scored BB%   (BB / PA)
  vector[N] z_babip;            // z-scored BABIP (H-HR)/(AB-SO-HR+SF)
  vector[N] z_age;              // z-scored age_from_peak (age - 29)
  vector[N] actual_woba;        // actual next-year wOBA (target)

  int<lower=0> N_pred;          // test observations
  vector[N_pred] marcel_woba_pred;
  vector[N_pred] z_K_pred;
  vector[N_pred] z_BB_pred;
  vector[N_pred] z_babip_pred;
  vector[N_pred] z_age_pred;
}

parameters {
  real delta_K;                 // K%    correction weight
  real delta_BB;                // BB%   correction weight
  real delta_BABIP;             // BABIP correction weight (expected negative)
  real delta_age;               // age   correction weight (expected negative)
  real<lower=0> sigma;          // residual std
}

model {
  // Regularized priors: corrections start near zero
  delta_K     ~ normal(0, 0.05);
  delta_BB    ~ normal(0, 0.05);
  delta_BABIP ~ normal(0, 0.05);
  delta_age   ~ normal(0, 0.05);
  sigma       ~ exponential(1);

  actual_woba ~ normal(
    marcel_woba + delta_K * z_K + delta_BB * z_BB
                + delta_BABIP * z_babip + delta_age * z_age,
    sigma
  );
}

generated quantities {
  vector[N_pred] stan_pred;
  vector[N] log_lik;

  for (i in 1:N_pred) {
    stan_pred[i] = marcel_woba_pred[i]
                   + delta_K     * z_K_pred[i]
                   + delta_BB    * z_BB_pred[i]
                   + delta_BABIP * z_babip_pred[i]
                   + delta_age   * z_age_pred[i];
  }
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(
      actual_woba[n] |
      marcel_woba[n] + delta_K * z_K[n] + delta_BB * z_BB[n]
                     + delta_BABIP * z_babip[n] + delta_age * z_age[n],
      sigma
    );
  }
}
