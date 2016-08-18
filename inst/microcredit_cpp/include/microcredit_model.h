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


# include "microcredit_probability_model.h"

///////////////////////////////////
// Functors

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


// Likelihood + entropy for a single group.  Note that global priors and
// entropy are not included.
struct MicroCreditGroupElbo {
  MicroCreditData data;
  VariationalParameters<double> base_vp;
  PriorParameters<double> pp;
  int g;

  bool include_obs;
  bool include_hier;
  bool include_prior;
  bool include_entropy;

  MicroCreditGroupElbo(
      MicroCreditData const &data,
      VariationalParameters<double> const &base_vp,
      PriorParameters<double> const &pp,
      int g):
    data(data), base_vp(base_vp), pp(pp), g(g) {
        include_obs = include_hier = include_prior = include_entropy = true;
    };

  template <typename T> T operator()(VectorXT<T> const &theta) const {
    VariationalParameters<T> vp(base_vp);
    SetFromGroupVector(theta, vp, g);

    T obs_log_lik = 0;
    T hier_log_lik = 0;
    T prior = 0;
    T entropy = 0;
    if (include_obs) obs_log_lik = GetGroupObservationLogLikelihood(data, vp, g);
    if (include_hier) hier_log_lik = GetGroupHierarchyLogLikelihood(vp, g);
    if (include_prior) prior = GetGroupPriorLogLikelihood(vp, pp, g);
    if (include_entropy) entropy = GetGroupEntropy(vp, g);
    return obs_log_lik + hier_log_lik + prior + entropy;
  }
};


// Likelihood + entropy for a global parameters not included in MicroCreditGroupElbo.
struct MicroCreditGlobalElbo {
  MicroCreditData data;
  VariationalParameters<double> base_vp;
  PriorParameters<double> pp;

  bool include_prior;
  bool include_entropy;

  MicroCreditGlobalElbo(
      MicroCreditData const &data,
      VariationalParameters<double> const &base_vp,
      PriorParameters<double> const &pp):
    data(data), base_vp(base_vp), pp(pp) {
        include_prior = include_entropy = true;
    };

  template <typename T> T operator()(VectorXT<T> const &theta) const {
    VariationalParameters<T> vp(base_vp);
    SetFromGlobalVector(theta, vp);
    T prior = 0;
    T entropy = 0;
    if (include_prior) prior = GetGlobalPriorLogLikelihood(vp, pp);
    if (include_entropy) entropy = GetGlobalEntropy(vp);
    return prior + entropy;
  }
};



struct MicroCreditElbo {
  MicroCreditData data;
  VariationalParameters<double> base_vp;
  PriorParameters<double> pp;

  bool include_obs;
  bool include_hier;
  bool include_prior;
  bool include_entropy;

  // If global_only evaluate the whole ELBO but only as a function of
  // global parameters.
  bool global_only;

  MicroCreditElbo(
      MicroCreditData const &data,
      VariationalParameters<double> const &base_vp,
      PriorParameters<double> const &pp):
    data(data), base_vp(base_vp), pp(pp) {
        include_obs = include_hier = include_prior = include_entropy = true;
        global_only = false;
    };

  template <typename T> T operator()(VectorXT<T> const &theta) const {
    VariationalParameters<T> vp(base_vp);
    if (global_only) {
        SetFromGlobalVector(theta, vp);
    } else {
        SetFromVector(theta, vp);
    }
    T obs_log_lik = 0;
    T hier_log_lik = 0;
    T prior = 0;
    T entropy = 0;
    if (include_obs) obs_log_lik = GetObservationLogLikelihood(data, vp);
    if (include_hier) hier_log_lik = GetHierarchyLogLikelihood(vp);
    if (include_prior) prior = GetPriorLogLikelihood(vp, pp);
    if (include_entropy) entropy = GetEntropy(vp);
    return obs_log_lik + hier_log_lik + prior + entropy;
  }
};


// Likelihood + entropy for lambda only.
struct NaturalToMomentParameters {
    VariationalParameters<double> base_vp;
    NaturalToMomentParameters(VariationalParameters<double> vp): base_vp(vp) {};

    template <typename T> VectorXT<T> operator()(VectorXT<T> const &theta) const {
        VariationalParameters<T> vp(base_vp);
        SetFromVector(theta, vp);
        MomentParameters<T> mp(vp);
        VectorXT<T> moment_vec = GetParameterVector(mp);
        return moment_vec;
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


// Get derivatives of the ELBO.
Derivatives GetElboDerivatives(
    MicroCreditData const &data,
    VariationalParameters<double> &vp,
    PriorParameters<double> const &pp,
    bool include_obs,
    bool include_hier,
    bool include_prior,
    bool include_entropy,
    bool global_only,
    bool const unconstrained,
    bool const calculate_gradient,
    bool const calculate_hessian);


// Get derivatives of the log prior.
Derivatives GetLogPriorDerivatives(
    VariationalParameters<double> &vp,
    PriorParameters<double> const &pp,
    bool const unconstrained,
    bool const calculate_gradient,
    bool const calculate_hessian);


// Get derivatives of the ELBO.
Derivatives GetMomentJacobian(VariationalParameters<double> &vp);

// Get the covariance of the moment parameters from the natural parameters.
SparseMatrix<double> GetCovariance(
    const VariationalParameters<double> &vp,
    const Offsets moment_offsets);

// Sparse Hessian
int GlobalIndex(int index, int g, Offsets offsets);



//////////////////////////////////////
// Sparse hessians.
// Func should be a functor that evaluates some objective for group g
// when its member functor.g = g
template <typename Func>
std::vector<Triplet> GetSparseGroupHessian(
    Func functor, VariationalParameters<double> vp) {

    std::vector<Triplet> all_terms;
    for (int g = 0; g < vp.n_g; g++) {
        functor.g = g;
        VectorXd theta = GetGroupParameterVector(vp, g);

        double val;
        VectorXd grad = VectorXd::Zero(theta.size());
        MatrixXd hess = MatrixXd::Zero(theta.size(), theta.size());
        stan::math::hessian(functor, theta, val, grad, hess);

        // The size of the beta, mu, and tau parameters
        for (int i1=0; i1 < theta.size(); i1++) {
            int gi1 = GlobalIndex(i1, g, vp.offsets);
            for (int i2=0; i2 < theta.size(); i2++) {
                int gi2 = GlobalIndex(i2, g, vp.offsets);
                all_terms.push_back(Triplet(gi1, gi2, hess(i1, i2)));
            }
        }
    }
    return all_terms;
};


SparseMatrix<double> GetSparseELBOHessian(
    MicroCreditData data,
    VariationalParameters<double> vp,
    PriorParameters<double> pp);


# endif
