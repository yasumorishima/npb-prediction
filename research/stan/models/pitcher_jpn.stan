// Japanese pitcher year-ahead ERA prediction
//
// Model: actual_ERA = Marcel_ERA + delta_K * z_K + delta_BB * z_BB
//                   + delta_K9 * z_K9 + delta_BB9 * z_BB9
//                   + delta_age * z_age + noise
//
// If all deltas = 0, reduces to pure Marcel (baseline).
// K%, BB% are per-BF rates; K/9, BB/9 are per-9-innings rates.
// The two normalizations provide complementary information.
// Negative delta_K/delta_K9:  high strikeout rate -> lower ERA (better)
// Positive delta_BB/delta_BB9: high walk rate      -> higher ERA (worse)
// Positive delta_age: older pitchers               -> higher ERA (worse)

data {
  int<lower=1> N;               // training observations
  vector[N] marcel_era;         // Marcel-projected ERA (prior mean)
  vector[N] z_K;                // z-scored K%  (SO / BF)
  vector[N] z_BB;               // z-scored BB% (BB / BF)
  vector[N] z_K9;               // z-scored K/9 (SO * 9 / IP)
  vector[N] z_BB9;              // z-scored BB/9 (BB * 9 / IP)
  vector[N] z_age;              // z-scored age_from_peak (age - 29)
  vector[N] actual_era;         // actual next-year ERA (target)

  int<lower=0> N_pred;          // test observations
  vector[N_pred] marcel_era_pred;
  vector[N_pred] z_K_pred;
  vector[N_pred] z_BB_pred;
  vector[N_pred] z_K9_pred;
  vector[N_pred] z_BB9_pred;
  vector[N_pred] z_age_pred;
}

parameters {
  real delta_K;                 // K% correction weight
  real delta_BB;                // BB% correction weight
  real delta_K9;                // K/9 correction weight
  real delta_BB9;               // BB/9 correction weight
  real delta_age;               // age correction weight (expected positive)
  real<lower=0> sigma;          // residual std
}

model {
  // Regularized priors: corrections start near zero
  delta_K    ~ normal(0, 0.5);
  delta_BB   ~ normal(0, 0.5);
  delta_K9   ~ normal(0, 0.5);
  delta_BB9  ~ normal(0, 0.5);
  delta_age  ~ normal(0, 0.5);
  sigma      ~ exponential(1);    // ERA residual scale ~1.0

  actual_era ~ normal(
    marcel_era + delta_K * z_K + delta_BB * z_BB
               + delta_K9 * z_K9 + delta_BB9 * z_BB9
               + delta_age * z_age,
    sigma
  );
}

generated quantities {
  vector[N_pred] stan_pred;
  vector[N] log_lik;

  for (i in 1:N_pred) {
    stan_pred[i] = marcel_era_pred[i]
                   + delta_K    * z_K_pred[i]
                   + delta_BB   * z_BB_pred[i]
                   + delta_K9   * z_K9_pred[i]
                   + delta_BB9  * z_BB9_pred[i]
                   + delta_age  * z_age_pred[i];
  }
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(
      actual_era[n] |
      marcel_era[n] + delta_K * z_K[n] + delta_BB * z_BB[n]
                    + delta_K9 * z_K9[n] + delta_BB9 * z_BB9[n]
                    + delta_age * z_age[n],
      sigma
    );
  }
}
