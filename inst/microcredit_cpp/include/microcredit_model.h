# ifndef MICROCREDIT_MODEL_H
# define MICROCREDIT_MODEL_H

# include <Eigen/Dense>
# include <vector>
# include <boost/math/tools/promotion.hpp>

// Special functions for the Hessian.
// #include "stan_lrvb_headers.h"

# include "variational_parameters.h"
# include "exponential_families.h"
# include "monte_carlo_parameters.h"
# include "microcredit_model_parameters.h"

// # include <stan/math.hpp>
# include <stan/math/mix/mat.hpp>
// # include <stan/math/mix/mat/functor/hessian.hpp>
# include <boost/math/complex/fabs.hpp> // Why not included in stan?

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

// The log Likelihood  evaluated at a draw of the parameters as a function
// of the prior parameters.
struct MicroCreditLogPriorDraw {

    // Encode the draw in moment parameters, where the "expectations"
    // are just the values of the draw.
    MomentParameters<double> draw;
    PriorParameters<double> base_pp;

    bool include_mu;
    bool include_lambda;
    bool include_tau;

    MicroCreditLogPriorDraw(
        MomentParameters<double> const &draw,
        PriorParameters<double> const &base_pp):
        draw(draw), base_pp(base_pp) {

        include_mu = include_lambda = include_tau = true;
    };

    template <typename T> T operator()(VectorXT<T> const &theta) const {
        PriorParameters<T> pp(base_pp);
        SetFromVector(theta, pp);

        T log_prior = 0.0;

        // Prior for global variables.
        log_prior += GetGlobalPriorLogLikelihoodDraw(
            draw.lambda.e.mat, draw.mu.e_vec, pp, include_mu, include_lambda);

        // Prior for group variables.
        if (include_tau) {
            for (int g = 0; g < draw.n_g; g++) {
                GammaMoments<T> mp_tau(draw.tau[g]);
                log_prior += GetGroupPriorLogLikelihood(mp_tau, pp.tau);
            }
        }

        return log_prior;
    }
};


// The expected log prior for a variatoinal distribution
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

        SetFromVector(theta, vp, pp);
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


struct VariationalLogDensity {
    // The observation is encoded in a MomentParameters object.
    MomentParameters<double> obs;
    VariationalParameters<double> base_vp;
    bool include_mu;
    bool include_lambda;
    VectorXi include_mu_groups;
    VectorXi include_tau_groups;

    VariationalLogDensity(
        VariationalParameters<double> const &vp,
        MomentParameters<double> const & _obs) {

        base_vp = VariationalParameters<double>(vp);
        obs = MomentParameters<double>(_obs);
        include_mu = true;
        include_lambda = true;
        VectorXd include_mu_groups(base_vp.n_g);
        VectorXd include_tau_groups(base_vp.n_g);
        for (int g = 0; g < base_vp.n_g; g++) {
          include_mu_groups(g) = g;
          include_tau_groups(g) = g;
        }
    };

    template <typename T> T operator()(VectorXT<T> const &theta) const {
        VariationalParameters<T> vp(base_vp);
        SetFromVector(theta, vp);

        T q_log_dens = 0.0;
        if (include_mu) {
          VectorXT<T> mu_obs = obs.mu.e_vec.template cast<T>();
          q_log_dens += vp.mu.log_lik(mu_obs);
        }

        if (include_lambda) {
          MatrixXT<T> lambda_obs = obs.lambda.e.mat.template cast<T>();
          q_log_dens += vp.lambda.log_lik(lambda_obs);
        }

        for (int g_ind = 0; g_ind < include_mu_groups.size(); g_ind++) {
          int g = include_mu_groups(g_ind);
          if (g < 0 || g >= vp.n_g) {
            throw std::runtime_error("mu_g q log density: g out of bounds.");
          }
          VectorXT<T> mu_g_obs = obs.mu_g[g].e_vec.template cast<T>();
          q_log_dens += vp.mu_g[g].log_lik(mu_g_obs);
        }

        for (int g_ind = 0; g_ind < include_tau_groups.size(); g_ind++) {
          int g = include_tau_groups(g_ind);
          if (g < 0 || g >= vp.n_g) {
            throw std::runtime_error("tau q log density: g out of bounds.");
          }
          T tau_obs = obs.tau[g].e;
          q_log_dens += vp.tau[g].log_lik(tau_obs);
        }

        return q_log_dens;
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


Derivatives GetLogPriorDerivativesFromDraw(
    MomentParameters<double> const &draw,
    PriorParameters<double> const &pp,
    bool const include_mu,
    bool const include_lambda,
    bool const include_tau,
    bool const calculate_gradient);


Derivatives GetLogVariationalDensityDerivatives(
    MomentParameters<double> const &obs,
    VariationalParameters<double> const &vp,
    bool const include_mu,
    bool const include_lambda,
    VectorXi const include_mu_groups,
    VectorXi const include_tau_groups,
    bool const calculate_gradient);


# endif
