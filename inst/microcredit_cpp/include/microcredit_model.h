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
  // TODO: maybe make this part of the variational parameters.
  WishartMoments<T> lambda_moments(vp.lambda);
  MultivariateNormalMoments<T> mu_moments(vp.mu);
  T log_lik = 0.0;
  for (int g = 0; g < vp.n_g; g++) {
      MultivariateNormalMoments<T> mu_g_moments(vp.mu_g_vec[g]);
      log_lik += mu_moments.ExpectedLogLikelihood(mu_moments, lambda_moments);
  }

  return log_lik;
};


template <typename T>
T GetObservationLogLikelihood(
    MicroCreditData const &data, VariationalParameters<T> const &vp) {

  UnivariateNormal<T> y_obs_mean;

  T log_lik = 0.0;
  for (int n = 0; n < data.n; n++) {
    UnivariateNormal<T> y_obs(data.y(n));

    VectorXT<T> x_row = data.x.row(n).template cast<T>();
    int g = data.y_g(n) - 1; // The group that this observation belongs to.
    // TODO: cache the moments or pass in moments.
    GammaMoments<T> tau_moments(vp.tau_vec[g]);
    MultivariateNormalMoments<T> mu_g_moments(vp.mu_g_vec[g]);

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
  MultivariateNormalMoments<T> vp_mu(vp.mu);
  MultivariateNormalNatural<T> pp_mu = pp.mu;
  log_lik += vp_mu.ExpectedLogLikelihood(pp_mu.loc, pp_mu.info);

  // Tau:
  GammaNatural<T> pp_tau(pp.tau);
  for (int g = 0; g < vp.n_g; g++) {
    GammaMoments<T> tau_moments(vp.tau);
    log_lik += tau_moments.ExpectedLogLikelihood(pp_tau.alpha, pp_tau.beta);
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


///////////////////////////////////
// Functors

// The log likelihood for the whole model (including the expected log prior)
struct MicroCreditLogLikelihood {
  MicroCreditData data;
  VariationalParameters<double> base_vp;
  PriorParameters<double> pp;
  VariationalParameterEncoder vp_encoder;

  MicroCreditLogLikelihood(
      MicroCreditData const &data,
      VariationalParameters<double> const &base_vp,
      PriorParameters<double> const &pp,
      VariationalParameterEncoder const &vp_encoder):
    data(data), base_vp(base_vp), pp(pp), vp_encoder(vp_encoder) {};

  template <typename T> T operator()(VectorXT<T> const &theta) const {
    VariationalParameters<T> vp(base_vp);
    vp_encoder.set_parameters_from_vector(theta, vp);

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
  ModelParameterEncoder encoder;

  MicroCreditLogPrior(
      VariationalParameters<double> const &base_vp,
      PriorParameters<double> const &base_pp,
      ModelParameterEncoder const &encoder):
    base_vp(base_vp), base_pp(base_pp), encoder(encoder) {};

  template <typename T> T operator()(VectorXT<T> const &theta) const {
    VariationalParameters<T> vp(base_vp);
    PriorParameters<T> pp(base_pp);
    encoder.set_parameters_from_vector(theta, vp, pp);
    return GetPriorLogLikelihood(vp, pp);
  }
};


// Likelihood + entropy for lambda only.
struct MicroCreditWishartElbo {
  MicroCreditData data;
  VariationalParameters<double> base_vp;
  PriorParameters<double> pp;
  WishartParameterEncoder lambda_encoder;

  MicroCreditWishartElbo(
      MicroCreditData const &data,
      VariationalParameters<double> const &base_vp,
      PriorParameters<double> const &pp,
      WishartParameterEncoder const &lambda_encoder):
    data(data), base_vp(base_vp), pp(pp), lambda_encoder(lambda_encoder) {};

  template <typename T> T operator()(VectorXT<T> const &theta) const {
    VariationalParameters<T> vp(base_vp);
    lambda_encoder.set_parameters_from_vector(theta, vp);
    MatrixXT<T> lambda_v_par = vp.lambda.v.mat;
    T lambda_n_par = vp.lambda.n;
    return
      GetHierarchyLogLikelihood(vp) +
      GetPriorLogLikelihood(vp, pp) +
      GetWishartEntropy(lambda_v_par, lambda_n_par);
  }
};


// entropy for lambda only, for debugging.
struct MicroCreditWishartEntropy {
  MicroCreditData data;
  VariationalParameters<double> base_vp;
  PriorParameters<double> pp;
  WishartParameterEncoder lambda_encoder;

  MicroCreditWishartEntropy(
      MicroCreditData const &data,
      VariationalParameters<double> const &base_vp,
      PriorParameters<double> const &pp,
      WishartParameterEncoder const &lambda_encoder):
    data(data), base_vp(base_vp), pp(pp), lambda_encoder(lambda_encoder) {};

  template <typename T> T operator()(VectorXT<T> const &theta) const {
    VariationalParameters<T> vp(base_vp);
    lambda_encoder.set_parameters_from_vector(theta, vp);
    MatrixXT<T> lambda_v_par = vp.lambda.v.mat;
    T lambda_n_par = vp.lambda.n;
    return GetWishartEntropy(lambda_v_par, lambda_n_par);
  }
};


// Likelihood for lambda only.
struct MicroCreditWishartLogLikelihood {
  MicroCreditData data;
  VariationalParameters<double> base_vp;
  PriorParameters<double> pp;
  WishartParameterEncoder lambda_encoder;

  MicroCreditWishartLogLikelihood(
      MicroCreditData const &data,
      VariationalParameters<double> const &base_vp,
      PriorParameters<double> const &pp,
      WishartParameterEncoder const &lambda_encoder):
    data(data), base_vp(base_vp), pp(pp), lambda_encoder(lambda_encoder) {};

  template <typename T> T operator()(VectorXT<T> const &theta) const {
    VariationalParameters<T> vp(base_vp);
    lambda_encoder.set_parameters_from_vector(theta, vp);
    MatrixXT<T> lambda_v_par = vp.lambda.v.mat;
    T lambda_n_par = vp.lambda.n;
    return
      GetHierarchyLogLikelihood(vp) + GetPriorLogLikelihood(vp, pp);
  }
};


// Derivatives of the moment parameterization of the Wishart.  I think
// this is actually not necessary, but let's do it as a proof of concept.
struct WishartMomentParameterization {
  VariationalParameters<double> base_vp;
  WishartParameterEncoder encoder;

  WishartMomentParameterization(
      VariationalParameters<double> const &base_vp,
      WishartParameterEncoder const &encoder):
    base_vp(base_vp), encoder(encoder) {};

  template <typename T> VectorXT<T> operator()(VectorXT<T> const &theta) const {
    VariationalParameters<T> vp(base_vp);
    VariationalParameters<T> moment_vp(base_vp);
    encoder.set_parameters_from_vector(theta, vp);

    MatrixXT<T> v_par = vp.lambda.v.mat;
    T n_par = vp.lambda.n;

    // TODO: we're just putting the moment parameters in the natural
    // parameter slots.  Any reason not to make e_lambda a member of the class?
    MatrixXT<T> e_lambda = v_par * n_par;
    moment_vp.lambda.v.set(e_lambda);

    T e_log_det_lambda = GetELogDetWishart(v_par, n_par);
    moment_vp.lambda.n = e_log_det_lambda;

    // This needs to be a non-transforming encoder.
    WishartParameterEncoder non_transforming_encoder(
      base_vp, encoder.lambda_diag_min, encoder.lambda_n_min, false);
    return non_transforming_encoder.get_parameter_vector(moment_vp);
  }
};



# endif
