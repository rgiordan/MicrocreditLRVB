# ifndef MICROCREDIT_MODEL_H
# define MICROCREDIT_MODEL_H

# include <Eigen/Dense>
# include <vector>
# include <boost/math/tools/promotion.hpp>

// Special functions for the Hessian.
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>

# include "variational_parameters.h"
# include "exponential_families.h"
# include "microcredit_model_parameters.h"

# include <stan/math.hpp>
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
  MatrixXT<T> e_lambda = vp.lambda_v_par.get() * vp.lambda_n_par.get();
  T e_log_det_lambda = GetELogDetWishart(
    vp.lambda_v_par.get(), vp.lambda_n_par.get());

  T e_lambda_e_mu2_trace = (e_lambda * vp.e_mu2.get()).trace();
  MatrixXT<T> e_mu_t = vp.e_mu.get().transpose();
  MatrixXT<T> e_mu = vp.e_mu.get();

  T log_lik = 0.0;
  for (int g = 0; g < vp.n_g; g++) {
    MatrixXT<T> e_mu_g = vp.e_mu_g_vec[g].get();
    MatrixXT<T> mu_g_mu_outer = e_mu_g * e_mu_t;
    MatrixXT<T> mu_mu_g_outer = e_mu * (e_mu_g.transpose());
    log_lik +=
      -0.5 * ((e_lambda * vp.e_mu2_g_vec[g].get()).trace() -
              (e_lambda * (mu_g_mu_outer + mu_mu_g_outer)).trace() +
              e_lambda_e_mu2_trace) + 0.5 * e_log_det_lambda;
  }

  return log_lik;
};


template <typename T>
T GetObservationLogLikelihood(
    MicroCreditData const &data, VariationalParameters<T> const &vp) {

  T log_lik = 0.0;
  for (int n = 0; n < data.n; n++) {
    VectorXT<T> x_row = data.x.row(n).template cast<T>();
    int g = data.y_g(n) - 1; // The group that this observation belongs to.
    T row_mean = x_row.dot(vp.e_mu_g_vec[g].get());
    T row_mean2 = x_row.dot(vp.e_mu2_g_vec[g].get() * x_row);
    T e_tau = vp.e_tau_vec[g].get();
    T e_log_tau = vp.e_log_tau_vec[g].get();
    log_lik += -0.5 * e_tau * (
        pow(data.y(n), 2) - 2 * data.y(n) * row_mean + row_mean2) +
        0.5 * e_log_tau;
  }
  return log_lik;
};


// template <typename Tlik, typename Tprior>
// typename promote_args<Tlik, Tprior>::type  GetPriorLogLikelihood(
//     VariationalParameters<Tlik> const &vp, PriorParameters<Tprior> const &pp);

template <typename Tlik, typename Tprior>
typename promote_args<Tlik, Tprior>::type  GetPriorLogLikelihood(
    VariationalParameters<Tlik> const &vp, PriorParameters<Tprior> const &pp) {

  typedef typename promote_args<Tlik, Tprior>::type T;

  // It's not the "likelihood", but the prior is part of the term that
  // isn't the entropy.
  T log_lik = 0.0;

  // Mu:
  VectorXT<T> vp_e_mu = vp.e_mu.get().template cast<T>();
  VectorXT<T> pp_mu_mean = pp.mu_mean.get().template cast<T>();

  MatrixXT<T> mu_mu_outer =
    vp_e_mu * pp_mu_mean.transpose() + pp_mu_mean * vp_e_mu.transpose();
  MatrixXT<T> mu_info = pp.mu_info.get().template cast<T>();
  MatrixXT<T> vp_e_mu2 = vp.e_mu2.get().template cast<T>();
  MatrixXT<T> pp_mu_mu_outer = pp_mu_mean * pp_mu_mean.transpose();
  log_lik += -0.5 * (mu_info * (vp_e_mu2 - mu_mu_outer + pp_mu_mu_outer)).trace();

  // Tau:
  for (int g = 0; g < vp.n_g; g++) {
    T tau_alpha = pp.tau_alpha.get();
    T tau_beta = pp.tau_beta.get();
    T e_tau = vp.e_tau_vec[g].get();
    T e_log_tau = vp.e_log_tau_vec[g].get();
    log_lik += (tau_alpha - 1) * e_log_tau - tau_beta * e_tau;
  }

  // Lambda.  Note that in the variable names Sigma = Lambda ^ (-1)
  MatrixXT<T> v_inv = vp.lambda_v_par.get().inverse();
  T n_par = vp.lambda_n_par.get();
  T e_log_sigma_term = digamma(0.5 * (n_par - pp.k + 1));
  T e_s_term = exp(lgamma(0.5 * (n_par - pp.k)) - lgamma(0.5 * (n_par - pp.k + 1)));
  T e_log_det_lambda = GetELogDetWishart(vp.lambda_v_par.get(), n_par);
  T e_log_det_r = -1 * e_log_det_lambda;
  T diag_prior = 0.0;

  T e_log_s, e_s, e_log_sigma_diag;
  for (int k=0; k < pp.k; k++) {
    e_log_sigma_diag =log(0.5 * v_inv(k, k)) - e_log_sigma_term;
    e_s = sqrt(0.5 * v_inv(k, k)) * e_s_term;
    e_log_s = 0.5 * e_log_sigma_diag;
    e_log_det_r -= e_log_sigma_diag;
    diag_prior += (pp.lambda_alpha.get() - 1) * e_log_s -
                   pp.lambda_beta.get() * e_s;
  }
  T lkj_prior = (pp.lambda_eta.get() - 1) * e_log_det_r;

  log_lik += lkj_prior + diag_prior;

  return log_lik;
};


////////////////////////////////////////
// Encoders to and from vectors

// An encoder for the variational parameters
class VariationalParameterEncoder {
public:
  // These set the order of the parameters.
  int e_mu_offset;
  int e_mu2_offset;

  int lambda_v_par_offset;
  int lambda_n_par_offset;

  vector<int> e_tau_offset;
  vector<int> e_log_tau_offset;

  vector<int> e_mu_g_offset;
  vector<int> e_mu2_g_offset;

  int dim;    // The total length of the parameters.

  bool unconstrained_wishart;
  double lambda_diag_min;
  double lambda_n_min;


  template <typename T> VariationalParameterEncoder(
      VariationalParameters<T> vp, double lambda_diag_min, double lambda_n_min,
      bool unconstrained_wishart):
      lambda_diag_min(lambda_diag_min), lambda_n_min(lambda_n_min),
      unconstrained_wishart(unconstrained_wishart)  {

    // If this is not the case we can leave the domain of the digamma function.
    if (lambda_n_min <= vp.lambda_v_par.size) {
      throw std::runtime_error("n_min must be greater than the dimension");
    }

    dim = 0; // Accumulate the total number of parameters.

    // This order defines the order in the vector.
    e_mu_offset = dim; dim += vp.e_mu.size;
    e_mu2_offset = dim; dim += vp.e_mu2.size_ud;
    lambda_v_par_offset = dim; dim += vp.lambda_v_par.size_ud;
    lambda_n_par_offset = dim; dim += 1;

    e_tau_offset.resize(vp.n_g);
    e_log_tau_offset.resize(vp.n_g);
    e_mu_g_offset.resize(vp.n_g);
    e_mu2_g_offset.resize(vp.n_g);

    for (int g = 0; g < vp.n_g; g++) {
      e_tau_offset[g] = dim; dim += 1;
      e_log_tau_offset[g] = dim; dim += 1;
      e_mu_g_offset[g] = dim; dim += vp.e_mu_g_vec[g].size;
      e_mu2_g_offset[g] = dim; dim += vp.e_mu2_g_vec[g].size_ud;
    }
  }


  VariationalParameterEncoder() {
    VariationalParameterEncoder(VariationalParameters<double>(), 0.0, 2.0, true);
  }


  template <typename T> void set_parameters_from_vector(
        VectorXT<T> const &theta, VariationalParameters<T> &vp) const {

    // This seems to be necessary to resolve eigen results into parameters
    // that the set() template can recognize.
    // TODO: why is this?

    VectorXT<T> theta_sub;
    theta_sub = theta.segment(e_mu_offset, vp.e_mu.size);
    vp.e_mu.set(theta_sub);

    theta_sub = theta.segment(e_mu2_offset, vp.e_mu2.size_ud);
    vp.e_mu2.set_vec(theta_sub);

    theta_sub = theta.segment(lambda_v_par_offset, vp.lambda_v_par.size_ud);
    if (unconstrained_wishart) {
      T lambda_n_min_cast = lambda_n_min;
      T lambda_diag_min_cast = lambda_diag_min;

      theta_sub = theta.segment(lambda_v_par_offset, vp.lambda_v_par.size_ud);
      vp.lambda_v_par.set_unconstrained_vec(theta_sub, lambda_diag_min_cast);

      T n_par_free = theta(lambda_n_par_offset);
      vp.lambda_n_par.set(exp(n_par_free) + lambda_n_min_cast);
    } else {
      vp.lambda_v_par.set_vec(theta_sub);
      vp.lambda_n_par.set(theta(lambda_n_par_offset));
    }

    for (int g = 0; g < vp.n_g; g++) {
      vp.e_tau_vec[g].set(theta(e_tau_offset[g]));
      vp.e_log_tau_vec[g].set(theta(e_log_tau_offset[g]));

      theta_sub = theta.segment(e_mu_g_offset[g], vp.e_mu_g_vec[g].size);
      vp.e_mu_g_vec[g].set(theta_sub);

      theta_sub = theta.segment(e_mu2_g_offset[g], vp.e_mu2_g_vec[g].size_ud);
      vp.e_mu2_g_vec[g].set_vec(theta_sub);
    }
  };


  template <class T>
  VectorXT<T> get_parameter_vector(
      VariationalParameters<T> const &vp) const {

    VectorXT<T> theta(dim);
    theta.segment(e_mu_offset, vp.e_mu.size) = vp.e_mu.get();
    theta.segment(e_mu2_offset, vp.e_mu2.size_ud) = vp.e_mu2.get_vec();

    if (unconstrained_wishart) {
      T lambda_n_min_cast = lambda_n_min;
      T lambda_diag_min_cast = lambda_diag_min;
      VectorXT<T> lambda_v_par_free =
        vp.lambda_v_par.get_unconstrained_vec(lambda_diag_min_cast);
      theta.segment(lambda_v_par_offset, vp.lambda_v_par.size_ud) =
        lambda_v_par_free;
      theta(lambda_n_par_offset) = log(vp.lambda_n_par.get() - lambda_n_min_cast);
    } else {
      theta.segment(lambda_v_par_offset, vp.lambda_v_par.size_ud) =
        vp.lambda_v_par.get_vec();
      theta(lambda_n_par_offset) = vp.lambda_n_par.get();
    }

    for (int g = 0; g < vp.n_g; g++) {
      theta(e_tau_offset[g]) = vp.e_tau_vec[g].get();
      theta(e_log_tau_offset[g]) = vp.e_log_tau_vec[g].get();

      theta.segment(e_mu_g_offset[g], vp.e_mu_g_vec[g].size) =
        vp.e_mu_g_vec[g].get();
      theta.segment(e_mu2_g_offset[g], vp.e_mu2_g_vec[g].size_ud) =
        vp.e_mu2_g_vec[g].get_vec();
    }
    return theta;
  };
};


//  An encoder for the Wishart parameters
class WishartParameterEncoder {
public:
  // These set the order of the parameters.
  int lambda_v_par_offset;
  int lambda_n_par_offset;
  double lambda_diag_min;
  double lambda_n_min;

  int dim;    // The total length of the parameters.
  bool unconstrained;

  template <typename T> WishartParameterEncoder(
      VariationalParameters<T> vp, double lambda_diag_min, double lambda_n_min,
      bool unconstrained):
      lambda_diag_min(lambda_diag_min), lambda_n_min(lambda_n_min),
      unconstrained(unconstrained) {

    // If this is not the case we can leave the domain of the digamma function.
    if (unconstrained && (lambda_n_min <= vp.lambda_v_par.size)) {
      throw std::runtime_error("n_min must be greater than the dimension");
    }

    dim = 0; // Accumulate the total number of parameters.

    // This order defines the order in the vector.
    lambda_v_par_offset = dim; dim += vp.lambda_v_par.size_ud;
    lambda_n_par_offset = dim; dim += 1;
  }

  WishartParameterEncoder() {
    WishartParameterEncoder(VariationalParameters<double>(), 0., 2.0, false);
  }


  template <typename T> void set_parameters_from_vector(
        VectorXT<T> const &theta, VariationalParameters<T> &vp) const {

    // This seems to be necessary to resolve eigen results into parameters
    // that the set() template can recognize.
    VectorXT<T> theta_sub;
    if (unconstrained) {
      T lambda_n_min_cast = lambda_n_min;
      T lambda_diag_min_cast = lambda_diag_min;

      theta_sub = theta.segment(lambda_v_par_offset, vp.lambda_v_par.size_ud);
      vp.lambda_v_par.set_unconstrained_vec(theta_sub, lambda_diag_min_cast);

      T n_par_free = theta(lambda_n_par_offset);
      vp.lambda_n_par.set(exp(n_par_free) + lambda_n_min_cast);
    } else {
      theta_sub = theta.segment(lambda_v_par_offset, vp.lambda_v_par.size_ud);
      vp.lambda_v_par.set_vec(theta_sub);

      T n_par = theta(lambda_n_par_offset);
      vp.lambda_n_par.set(n_par);
    }
  };


  template <class T> VectorXT<T> get_parameter_vector(
      VariationalParameters<T> const &vp) const {

    VectorXT<T> theta(dim);
    if (unconstrained) {
      T lambda_n_min_cast = lambda_n_min;
      T lambda_diag_min_cast = lambda_diag_min;
      VectorXT<T> lambda_v_par_free =
        vp.lambda_v_par.get_unconstrained_vec(lambda_diag_min_cast);
      theta.segment(lambda_v_par_offset, vp.lambda_v_par.size_ud) =
        lambda_v_par_free;
      theta(lambda_n_par_offset) = log(vp.lambda_n_par.get() - lambda_n_min_cast);
    } else {
      theta.segment(lambda_v_par_offset, vp.lambda_v_par.size_ud) =
        vp.lambda_v_par.get_vec();
      theta(lambda_n_par_offset) = vp.lambda_n_par.get();
    }
    return theta;
  };
};


// An encoder for the prior parameters
class PriorParameterEncoder {
public:
  // These set the order of the parameters.
  int mu_mean_offset;
  int mu_info_offset;

  int lambda_eta_offset;
  int lambda_alpha_offset;
  int lambda_beta_offset;

  int tau_alpha_offset;
  int tau_beta_offset;

  int dim;    // The total length of the parameters.

  template <typename T> PriorParameterEncoder(PriorParameters<T> pp) {

    dim = 0; // Accumulate the total number of parameters.

    // This order defines the order in the vector.
    mu_mean_offset = dim; dim += pp.mu_mean.size;
    mu_info_offset = dim; dim += pp.mu_info.size_ud;
    lambda_eta_offset = dim; dim += 1;
    lambda_alpha_offset = dim; dim += 1;
    lambda_beta_offset = dim; dim += 1;
    tau_alpha_offset = dim; dim += 1;
    tau_beta_offset = dim; dim += 1;
  };


  PriorParameterEncoder() {
    PriorParameterEncoder(PriorParameters<double>());
  };


  template <typename T> void set_parameters_from_vector(
        VectorXT<T> const &theta, PriorParameters<T> &pp) const {

    // This seems to be necessary to resolve eigen results into parameters
    // that the set() template can recognize.
    VectorXT<T> theta_sub;
    theta_sub = theta.segment(mu_mean_offset, pp.mu_mean.size);
    pp.mu_mean.set(theta_sub);

    theta_sub = theta.segment(mu_info_offset, pp.mu_info.size_ud);
    pp.mu_info.set_vec(theta_sub);

    pp.lambda_eta.set(theta(lambda_eta_offset));
    pp.lambda_alpha.set(theta(lambda_alpha_offset));
    pp.lambda_beta.set(theta(lambda_beta_offset));

    pp.tau_alpha.set(theta(tau_alpha_offset));
    pp.tau_beta.set(theta(tau_beta_offset));
  };


  template <class T>
  VectorXT<T> get_parameter_vector(PriorParameters<T> const &pp) const {

    VectorXT<T> theta(dim);
    theta.segment(mu_mean_offset, pp.mu_mean.size) = pp.mu_mean.get();
    theta.segment(mu_info_offset, pp.mu_info.size_ud) = pp.mu_info.get_vec();

    theta(lambda_eta_offset) = pp.lambda_eta.get();
    theta(lambda_alpha_offset) = pp.lambda_alpha.get();
    theta(lambda_beta_offset) = pp.lambda_beta.get();

    theta(tau_alpha_offset) = pp.tau_alpha.get();
    theta(tau_beta_offset) = pp.tau_beta.get();

    return theta;
  };
};


// An encoder for the variational and prior parameters
class ModelParameterEncoder {
public:
  // These set the order of the parameters.

  PriorParameterEncoder prior_encoder;
  VariationalParameterEncoder variational_encoder;

  int dim;    // The total length of the parameters.
  int prior_offset;
  int variational_offset;

  ModelParameterEncoder(
      VariationalParameterEncoder vp_encoder,
      PriorParameterEncoder pp_encoder):
      prior_encoder(pp_encoder), variational_encoder(vp_encoder) {

    dim = 0; // Accumulate the total number of parameters.
    variational_offset = dim; dim += variational_encoder.dim;
    prior_offset = dim; dim += prior_encoder.dim;
  }


  ModelParameterEncoder() {
    ModelParameterEncoder(VariationalParameterEncoder(),
                          PriorParameterEncoder());
  }


  template <typename T> void set_parameters_from_vector(
        VectorXT<T> const &theta,
        VariationalParameters<T> &vp,
        PriorParameters<T> &pp) const {

    // This seems to be necessary to resolve eigen results into parameters
    // that the set() template can recognize.
    VectorXT<T> theta_sub;
    theta_sub = theta.segment(variational_offset, variational_encoder.dim);
    variational_encoder.set_parameters_from_vector(theta_sub, vp);
    theta_sub = theta.segment(prior_offset, prior_encoder.dim);
    prior_encoder.set_parameters_from_vector(theta_sub, pp);
  };


  template <class T>
  VectorXT<T> get_parameter_vector(
      VariationalParameters<T> vp, PriorParameters<T> &pp) const {

    VectorXT<T> theta(dim);
    theta.segment(variational_offset, variational_encoder.dim) =
      variational_encoder.get_parameter_vector(vp);
    theta.segment(prior_offset, prior_encoder.dim) =
      prior_encoder.get_parameter_vector(pp);

    return theta;
  };
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
    MatrixXT<T> lambda_v_par = vp.lambda_v_par.get();
    T lambda_n_par = vp.lambda_n_par.get();
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
    MatrixXT<T> lambda_v_par = vp.lambda_v_par.get();
    T lambda_n_par = vp.lambda_n_par.get();
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
    MatrixXT<T> lambda_v_par = vp.lambda_v_par.get();
    T lambda_n_par = vp.lambda_n_par.get();
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

    MatrixXT<T> v_par = vp.lambda_v_par.get();
    T n_par = vp.lambda_n_par.get();

    // TODO: we're just putting the moment parameters in the natural
    // parameter slots.  Any reason not to make e_lambda a member of the class?
    MatrixXT<T> e_lambda = v_par * n_par;
    moment_vp.lambda_v_par.set(e_lambda);

    T e_log_det_lambda = GetELogDetWishart(v_par, n_par);
    moment_vp.lambda_n_par.set(e_log_det_lambda);

    // This needs to be a non-transforming encoder.
    WishartParameterEncoder non_transforming_encoder(
      base_vp, encoder.lambda_diag_min, encoder.lambda_n_min, false);
    return non_transforming_encoder.get_parameter_vector(moment_vp);
  }
};



# endif
