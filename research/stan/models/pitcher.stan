// Pitcher ERA projection: league average + standardized skill features
//
// Model:
//   y = lg_avg + beta_era * z_era + beta_fip * z_fip
//          + beta_K * z_K + beta_BB * z_BB + noise
//
// Features are standardized (z-score) using training-set mean/sd.
// Regularized priors push betas toward zero (= league average prediction).
// ERA/FIP collinearity is handled by regularization.

data {
  int<lower=1> N;            // number of observations
  vector[N] y;               // NPB first-year ERA
  vector[N] lg_avg;          // NPB league-average ERA for that year
  vector[N] z_era;           // standardized previous-league ERA
  vector[N] z_fip;           // standardized previous-league FIP
  vector[N] z_K;             // standardized previous-league K%
  vector[N] z_BB;            // standardized previous-league BB%
  int<lower=0> N_features;   // 1 (v0: ERA only) or 4 (v1: ERA+FIP+K%+BB%)
}

parameters {
  real beta_era;
  real beta_fip;
  real beta_K;
  real beta_BB;
  real<lower=0> sigma;
}

model {
  // Regularized priors — strong shrinkage toward zero
  beta_era  ~ normal(0, 0.5);
  beta_fip  ~ normal(0, 0.5);
  beta_K    ~ normal(0, 0.5);
  beta_BB   ~ normal(0, 0.5);
  sigma     ~ exponential(1);  // ERA scale ~ 1.0

  // Likelihood
  vector[N] mu;
  if (N_features == 1) {
    mu = lg_avg + beta_era * z_era;
  } else {
    mu = lg_avg + beta_era * z_era + beta_fip * z_fip + beta_K * z_K + beta_BB * z_BB;
  }
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  {
    vector[N] mu;
    if (N_features == 1) {
      mu = lg_avg + beta_era * z_era;
    } else {
      mu = lg_avg + beta_era * z_era + beta_fip * z_fip + beta_K * z_K + beta_BB * z_BB;
    }
    for (n in 1:N) {
      log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
      y_rep[n] = normal_rng(mu[n], sigma);
    }
  }
}
