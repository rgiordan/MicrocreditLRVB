data {
  int<lower=0> NG;  // number of groups
  int<lower=0> N;  // total number of observations
  int<lower=0> K;  // dimensionality of parameter vector which is jointly distributed
  real y[N];       // outcome variable of interest
  vector[K] x[N];       // Covariates
  int y_group[N];     // factor variable to split them out into K sites
  matrix[K,K] mu_prior_sigma;
  matrix[K,K] mu1_prior_sigma;
  vector[K] mu_prior_mean;
  vector[K] mu1_prior_mean;
  real tau_prior_alpha;
  real tau_prior_beta;
  real scale_prior_alpha;
  real scale_prior_beta;
  real lkj_prior_eta;
  int <lower=0,upper=1> use_mu1_prior;
}
parameters {
  vector[K] mu;
  vector[K] mu1[NG];
  real<lower=0> sigma_y[NG];

  corr_matrix[K] R;        //  Covariance correlation
  vector<lower=0>[K] S;    //  Information scale
}
transformed parameters {
  cov_matrix[K] sigma_mu;
  cov_matrix[K] lambda_mu;
  cov_matrix[K] mu1_prior_lambda;
  cov_matrix[K] mu1_cov;
  vector[K] mu1_mean;

  sigma_mu <- quad_form_diag(R, S);
  lambda_mu <- inverse_spd(sigma_mu);
  if (use_mu1_prior == 1) {
    mu1_prior_lambda <- inverse_spd(mu1_prior_sigma);
    mu1_cov <- inverse_spd(lambda_mu + mu1_prior_lambda);
    mu1_mean <- mu1_cov * (lambda_mu * mu + mu1_prior_lambda * mu1_prior_mean);
  } else {
    mu1_prior_lambda <- mu1_prior_sigma; // It has to take some value.
    mu1_cov <- sigma_mu;
    mu1_mean <- mu;
  }
}
model {
  // data variance priors
  for (g in 1:NG) {
    sigma_y[g] ~ inv_gamma(tau_prior_alpha, tau_prior_beta);
  }

  // parameter variance priors
  for (k in 1:K) {
    S[k] ~ gamma(scale_prior_alpha, scale_prior_beta); // E(S) = prior_alpha / prior_beta
  }
  R ~ lkj_corr(lkj_prior_eta);

  // hyperparameter priors
  mu ~ multi_normal(mu_prior_mean, mu_prior_sigma);

  for (g in 1:NG) {
    mu1[g] ~ multi_normal(mu1_mean, mu1_cov);
  }

  {
    vector[N] meanvec_y;
    vector[N] sigmavec_y;
    for (n in 1:N) {
      meanvec_y[n] <- dot_product(mu1[y_group[n]], x[n]);
      sigmavec_y[n] <- sigma_y[y_group[n]];
    }
    y ~ normal(meanvec_y, sigmavec_y); // data level
  }
}
