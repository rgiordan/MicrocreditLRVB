data {
  int<lower=0> NG;  // number of groups
  int<lower=0> K;  // dimensionality of parameter vector which is jointly distributed

  # For speed, do calculations with summary statistics.
  vector<lower=0>[NG] num_obs;
  vector[NG] yty;
  vector[K] ytx[NG];
  matrix[K, K] xtx[NG];

  // For each group, the number of zeros in the control and treatment, in that order.
  vector[2] num_zeros[NG];

  vector[K] beta_prior_mean;
  cov_matrix[K] beta_prior_sigma;

  real<lower=0> tau_prior_alpha;
  real<lower=0> tau_prior_beta;

  real<lower=0> scale_prior_alpha;
  real<lower=0> scale_prior_beta;

  real<lower=0> lkj_prior_eta;
}
parameters {
  vector[K] beta_group[NG];
  vector[K] beta;
  real<lower=0> sigma_y[NG];

  corr_matrix[K] R;        //  Covariance correlation
  vector<lower=0>[K] S;    //  Information scale
}
transformed parameters {
  cov_matrix[K] sigma_beta;
  cov_matrix[K] lambda_beta;

  sigma_beta = quad_form_diag(R, S);
  lambda_beta = inverse_spd(sigma_beta);
}
model {
  // data variance priors
  for (g in 1:NG) {
    # sigma_y is the variance, not the standard deviation.
    sigma_y[g] ~ inv_gamma(tau_prior_alpha, tau_prior_beta);
  }

  // parameter variance priors
  for (k in 1:K) {
    // S[k] is the square root of the diagonals of the covariance matrix.
    S[k] ~ gamma(scale_prior_alpha, scale_prior_beta); // E(S) = prior_alpha / prior_beta
  }
  // Just a note: ljk_cov uses lognormal scales, not inverse gamma scales.
  R ~ lkj_corr(lkj_prior_eta);

  // hyperparameter priors
  beta ~ multi_normal(beta_prior_mean, beta_prior_sigma);

  for (g in 1:NG) {
    beta_group[g] ~ multi_normal(beta, sigma_beta);
    target += -0.5 * (
      yty[g] +
      -2 * dot_product(ytx[g], beta_group[g]) +
      dot_product(beta_group[g], xtx[g] * beta_group[g])) / sigma_y[g];
    target += -0.5 * num_obs[g] * log(sigma_y[g]);
  }
}
