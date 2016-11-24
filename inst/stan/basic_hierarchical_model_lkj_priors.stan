data {
  int<lower=0> NG;  // number of groups
  int<lower=0> N;  // total number of observations
  int<lower=0> K;  // dimensionality of parameter vector which is jointly distributed
  real y[N];       // outcome variable of interest
  vector[K] x[N];       // Covariates
  int y_group[N];     // factor variable to split them out into K sites
  matrix[K,K] mu_prior_sigma;
  vector[K] mu_prior_mean;

  // A contaminating prior for mu.  "c" is for "contamination".
  real <lower=0, upper=1> mu_epsilon;
  matrix[K,K] mu_prior_sigma_c;
  vector[K] mu_prior_mean_c;
  real <lower=0> mu_prior_df;
  int <lower=0, upper=1> mu_prior_use_t_contamination;
  
  real tau_prior_alpha;
  real tau_prior_beta;
  real scale_prior_alpha;
  real scale_prior_beta;
  real lkj_prior_eta;
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

  // Log priors for sensitivity analysis.  
  real mu_log_prior;
  real mu_log_prior_c;
  real mu_log_prior_eps;

  mu_log_prior = multi_normal_lpdf(mu | mu_prior_mean, mu_prior_sigma);
  
  # Either use a MVN prior or a product of independent student t priors.
  if (mu_prior_use_t_contamination == 0) {
    mu_log_prior_c = multi_normal_lpdf(mu | mu_prior_mean_c, mu_prior_sigma_c);
  } else {
    mu_log_prior_c = 0.0;
    for (k in 1:K) {
      mu_log_prior_c = mu_log_prior_c + student_t_lpdf(mu[k] | mu_prior_df, mu_prior_mean_c, sqrt(mu_prior_sigma_c[k, k]));
    }
  }
  mu_log_prior_eps = log_sum_exp(log(1 - mu_epsilon) + mu_log_prior, log(mu_epsilon) + mu_log_prior_c);

  sigma_mu = quad_form_diag(R, S);
  lambda_mu = inverse_spd(sigma_mu);
}
model {
  // data variance priors
  for (g in 1:NG) {
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
  if (mu_epsilon == 0) {
    target += mu_log_prior;
  } else if (mu_epsilon == 1) {
    target += mu_log_prior_c;
  } else {
    // A mixture of the two with mixing weight epsilon.
    target += mu_log_prior_eps;
  }
  // mu ~ multi_normal(mu_prior_mean, mu_prior_sigma);

  for (g in 1:NG) {
    mu1[g] ~ multi_normal(mu, sigma_mu);
  }

  {
    vector[N] meanvec_y;
    vector[N] sigmavec_y;
    for (n in 1:N) {
      meanvec_y[n] = dot_product(mu1[y_group[n]], x[n]);
      sigmavec_y[n] = sigma_y[y_group[n]];
    }
    y ~ normal(meanvec_y, sigmavec_y); // data level
  }
}
