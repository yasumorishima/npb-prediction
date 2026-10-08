// Hitter wOBA projection: league average + standardized skill features
//
// Model:
//   y = lg_avg + beta_woba * z_woba + beta_K * z_K + beta_BB * z_BB + noise
//
// Features are standardized (z-score) using training-set mean/sd.
// Regularized priors push betas toward zero (= league average prediction).

data {
  int<lower=1> N;            // number of observations
  vector[N] y;               // NPB first-year wOBA
  vector[N] lg_avg;          // NPB league-average wOBA for that year
  vector[N] z_woba;          // standardized previous-league wOBA
  vector[N] z_K;             // standardized previous-league K%
  vector[N] z_BB;            // standardized previous-league BB%
  int<lower=0> N_features;   // 1 (v0: wOBA only) or 3 (v1: wOBA+K%+BB%)
}

parameters {
  real beta_woba;
  real beta_K;
  real beta_BB;
  real<lower=0> sigma;
}

model {
  // Regularized priors — strong shrinkage toward zero
  beta_woba ~ normal(0, 0.02);
  beta_K    ~ normal(0, 0.02);
  beta_BB   ~ normal(0, 0.02);
  sigma     ~ exponential(20);  // wOBA scale ~ 0.05

  // Likelihood
  vector[N] mu;
  if (N_features == 1) {
    mu = lg_avg + beta_woba * z_woba;
  } else {
    mu = lg_avg + beta_woba * z_woba + beta_K * z_K + beta_BB * z_BB;
  }
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  {
    vector[N] mu;
    if (N_features == 1) {
      mu = lg_avg + beta_woba * z_woba;
    } else {
      mu = lg_avg + beta_woba * z_woba + beta_K * z_K + beta_BB * z_BB;
    }
    for (n in 1:N) {
      log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
      y_rep[n] = normal_rng(mu[n], sigma);
    }
  }
}
