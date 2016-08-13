# ifndef MICROCREDIT_MODEL_H
# define MICROCREDIT_MODEL_H

# include <Eigen/Dense>
# include <vector>
# include <boost/math/tools/promotion.hpp>

// Special functions for the Hessian.
#include "stan_lrvb_headers.h"

# include "variational_parameters.h"
# include "exponential_families.h"
# include "microcredit_model_parameters.h"

# include <stan/math.hpp>
# include <stan/math/fwd/scal.hpp>
# include <stan/math/mix/mat/functor/hessian.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixXi;
using Eigen::VectorXi;

using Eigen::Matrix;
using Eigen::Dynamic;

template <typename T> using VectorXT = Eigen::Matrix<T, Dynamic, 1>;
template <typename T> using MatrixXT = Eigen::Matrix<T, Dynamic, Dynamic>;

using std::vector;
using typename boost::math::tools::promote_args;

using var = stan::math::var;
using fvar = stan::math::fvar<var>;


////////////////////////////////////////
// Likelihood functions

template <typename T>
T GetHierarchyLogLikelihood(VariationalParameters<T> const &vp) {
    WishartMoments<T> lambda_moments(vp.lambda);

    MultivariateNormalMoments<T> mu_moments(vp.mu);
    T log_lik = 0.0;
    for (int g = 0; g < vp.n_g; g++) {
        MultivariateNormalMoments<T> mu_g_moments(vp.mu_g[g]);
        log_lik += mu_moments.ExpectedLogLikelihood(mu_moments, lambda_moments);
    }

    return log_lik;
};


template <typename T>
T GetObservationLogLikelihood(
    MicroCreditData const &data, VariationalParameters<T> const &vp) {

  UnivariateNormalMoments<T> y_obs_mean;

  T log_lik = 0.0;
  for (int n = 0; n < data.n; n++) {
    UnivariateNormalMoments<T> y_obs;
    VectorXT<T> x_row = data.x.row(n).template cast<T>();
    y_obs.e = data.y(n);
    y_obs.e2 = pow(data.y(n), 2);

    int g = data.y_g(n) - 1; // The group that this observation belongs to.
    // TODO: cache the moments or pass in moments.
    GammaMoments<T> tau_moments(vp.tau[g]);
    MultivariateNormalMoments<T> mu_g_moments(vp.mu_g[g]);

    y_obs_mean.e = x_row.dot(mu_g_moments.e_vec);
    y_obs_mean.e2 = x_row.dot(mu_g_moments.e_outer.mat * x_row);

    log_lik += y_obs.ExpectedLogLikelihood(y_obs_mean, tau_moments);
  }
  return log_lik;
};


template <typename Tlik, typename Tprior>
typename promote_args<Tlik, Tprior>::type  GetPriorLogLikelihood(
    VariationalParameters<Tlik> const &vp, PriorParameters<Tprior> const &pp) {

  typedef typename promote_args<Tlik, Tprior>::type T;

  // It's not the "likelihood", but the prior is part of the term that
  // isn't the entropy.
  T log_lik = 0.0;

  // Mu:
  MultivariateNormalNatural<T> vp_mu(vp.mu);
  MultivariateNormalMoments<T> vp_mu_moments(vp_mu);
  MultivariateNormalNatural<T> pp_mu = pp.mu;
  log_lik += vp_mu_moments.ExpectedLogLikelihood(pp_mu.loc, pp_mu.info.mat);

  // Tau:
  GammaNatural<T> pp_tau(pp.tau);
  for (int g = 0; g < vp.n_g; g++) {
    GammaNatural<T> vp_tau(vp.tau[g]);
    GammaMoments<T> vp_tau_moments(vp_tau);
    log_lik += vp_tau_moments.ExpectedLogLikelihood(pp_tau.alpha, pp_tau.beta);
  }

  // Lambda.  Note that in the variable names Sigma = Lambda ^ (-1)
  MatrixXT<T> v_inv = vp.lambda.v.mat.inverse();
  T n_par = vp.lambda.n;
  T e_log_sigma_term = digamma(0.5 * (n_par - pp.k + 1));
  T e_s_term = exp(lgamma(0.5 * (n_par - pp.k)) - lgamma(0.5 * (n_par - pp.k + 1)));
  T e_log_det_lambda = GetELogDetWishart(vp.lambda.v.mat, n_par);
  T e_log_det_r = -1 * e_log_det_lambda;
  T diag_prior = 0.0;

  T e_log_s, e_s, e_log_sigma_diag;
  for (int k=0; k < pp.k; k++) {
    e_log_sigma_diag =log(0.5 * v_inv(k, k)) - e_log_sigma_term;
    e_s = sqrt(0.5 * v_inv(k, k)) * e_s_term;
    e_log_s = 0.5 * e_log_sigma_diag;
    e_log_det_r -= e_log_sigma_diag;
    diag_prior += (pp.lambda_alpha - 1) * e_log_s -
                   pp.lambda_beta * e_s;
  }
  T lkj_prior = (pp.lambda_eta - 1) * e_log_det_r;

  log_lik += lkj_prior + diag_prior;

  return log_lik;
};



template <typename T> T
GetEntropy(VariationalParameters<T> const &vp) {
    T entropy = 0;
    entropy =
        GetMultivariateNormalEntropy(vp.mu.info.mat) +
        GetWishartEntropy(vp.lambda.v.mat, vp.lambda.n);
    for (int g = 0; g < vp.n_g; g++) {
        entropy += GetMultivariateNormalEntropy(vp.mu_g[g].info.mat);
        entropy += GetGammaEntropy(vp.tau[g].alpha, vp.tau[g].beta);
    }
    return entropy;
}



///////////////////////////////////
// Functors

// The log likelihood for the whole model (including the expected log prior)
struct MicroCreditLogLikelihood {
  MicroCreditData data;
  VariationalParameters<double> base_vp;
  PriorParameters<double> pp;

  MicroCreditLogLikelihood(
      MicroCreditData const &data,
      VariationalParameters<double> const &base_vp,
      PriorParameters<double> const &pp):
    data(data), base_vp(base_vp), pp(pp) {};

  template <typename T> T operator()(VectorXT<T> const &theta) const {
    VariationalParameters<T> vp(base_vp);
    SetFromVector(theta, vp);

    return
      GetObservationLogLikelihood(data, vp) +
      GetHierarchyLogLikelihood(vp) +
      GetPriorLogLikelihood(vp, pp);
  }
};


// The log likelihood for the whole model (including the expected log prior)
struct MicroCreditLogPrior {
  VariationalParameters<double> base_vp;
  PriorParameters<double> base_pp;

  MicroCreditLogPrior(
      VariationalParameters<double> const &base_vp,
      PriorParameters<double> const &base_pp):
    base_vp(base_vp), base_pp(base_pp) {};

  template <typename T> T operator()(VectorXT<T> const &theta) const {
    VariationalParameters<T> vp(base_vp);
    PriorParameters<T> pp(base_pp);

    if (theta.size() != vp.offsets.encoded_size + pp.offsets.encoded_size) {
        throw std::runtime_error("Theta is the wrong size.");
    }

    VectorXT<T> theta_sub;
    theta_sub = theta.segment(0, vp.offsets.encoded_size);
    SetFromVector(theta_sub, vp);

    theta_sub = theta.segment(vp.offsets.encoded_size + 1, pp.offsets.encoded_size);
    SetFromVector(theta_sub, pp);

    return GetPriorLogLikelihood(vp, pp);
  }
};


// Likelihood + entropy for lambda only.
struct MicroCreditElbo {
  MicroCreditData data;
  VariationalParameters<double> base_vp;
  PriorParameters<double> pp;

  MicroCreditElbo(
      MicroCreditData const &data,
      VariationalParameters<double> const &base_vp,
      PriorParameters<double> const &pp):
    data(data), base_vp(base_vp), pp(pp) {};

  template <typename T> T operator()(VectorXT<T> const &theta) const {
    VariationalParameters<T> vp(base_vp);
    SetFromVector(theta, vp);
    T obs_log_lik = GetObservationLogLikelihood(data, vp);

    T hier_log_lik = GetHierarchyLogLikelihood(vp);
    T prior = GetPriorLogLikelihood(vp, pp);
    T entropy = GetEntropy(vp);
    return obs_log_lik + hier_log_lik + prior + entropy;
  }
};



struct Derivatives {
  double val;
  VectorXd grad;
  MatrixXd hess;

  Derivatives(double val, VectorXd grad, MatrixXd hess):
  val(val), grad(grad), hess(hess) {};
};


// Get derivatives of the ELBO.
Derivatives GetElboDerivatives(
    MicroCreditData const &data,
    VariationalParameters<double> &vp,
    PriorParameters<double> const &pp,
    bool const unconstrained,
    bool const calculate_gradient,
    bool const calculate_hessian);


// This is not necessary, I think -- declaring GetElboDerivatives should do it.
// extern template
// double MicroCreditElbo::operator()(VectorXT<double> const &theta) const;


// Get the covariance of the moment parameters from the natural parameters.
SparseMatrix<double> GetCovariance(
    const VariationalParameters<double> &vp,
    const Offsets moment_offsets);


# endif
