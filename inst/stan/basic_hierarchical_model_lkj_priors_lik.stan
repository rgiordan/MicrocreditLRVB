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

  vector[N] weights;
}
transformed parameters {
}
model {

  # A model for the log likelihood alone with a data weights vector.
  for (n in 1:N) {
    real meanvec_y;
    real sigmavec_y;
    meanvec_y = dot_product(mu1[y_group[n]], x[n]);
    sigmavec_y = sigma_y[y_group[n]];
    target += weights[n] * normal_lpdf(y[n] | meanvec_y, sigmavec_y);
  }
}
