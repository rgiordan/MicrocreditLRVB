# include <Eigen/Dense>
# include <vector>
# include <boost/math/tools/promotion.hpp>

// Special functions for the Hessian.
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>

// To check the Jacobian orientation
# include "transform_hessian.h"

# include "variational_parameters.h"
# include "exponential_families.h"
# include "microcredit_model_parameters.h"

# include "microcredit_model.h"

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


Derivatives GetElboDerivatives(
  MicroCreditData const &data,
  VariationalParameters<double> &vp,
  PriorParameters<double> const &pp,
  bool const unconstrained,
  bool const calculate_gradient,
  bool const calculate_hessian) {

    vp.unconstrained = unconstrained;

    MicroCreditElbo ELBO(data, vp, pp);

    double val;
    VectorXd grad = VectorXd::Zero(vp.offsets.encoded_size);
    MatrixXd hess = MatrixXd::Zero(vp.offsets.encoded_size, vp.offsets.encoded_size);
    VectorXd theta = GetParameterVector(vp);

    stan::math::set_zero_all_adjoints();

    if (calculate_hessian) {
        stan::math::hessian(ELBO, theta, val, grad, hess);
    } else if (calculate_gradient) {
        stan::math::gradient(ELBO, theta, val, grad);
    } else {
        val = ELBO(theta);
    }

    return Derivatives(val, grad, hess);
};


// ELBO derivatives with customizable options.
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
  bool const calculate_hessian) {

    vp.unconstrained = unconstrained;

    MicroCreditElbo ELBO(data, vp, pp);
    ELBO.include_obs = include_obs;
    ELBO.include_hier = include_hier;
    ELBO.include_prior = include_prior;
    ELBO.include_entropy = include_entropy;

    ELBO.global_only = global_only;

    double val;
    VectorXd grad = VectorXd::Zero(vp.offsets.encoded_size);
    MatrixXd hess = MatrixXd::Zero(vp.offsets.encoded_size, vp.offsets.encoded_size);

    VectorXd theta;
    if (global_only) {
        theta = GetGlobalParameterVector(vp);
    } else {
        theta = GetParameterVector(vp);
    }

    stan::math::set_zero_all_adjoints();

    if (calculate_hessian) {
        stan::math::hessian(ELBO, theta, val, grad, hess);
    } else if (calculate_gradient) {
        stan::math::gradient(ELBO, theta, val, grad);
    } else {
        val = ELBO(theta);
    }

    return Derivatives(val, grad, hess);
};


Derivatives GetMomentJacobian(VariationalParameters<double> &vp) {
    NaturalToMomentParameters GetMoments(vp);
    MomentParameters<double> mp(vp);

    VectorXd theta = GetParameterVector(vp);
    VectorXd moments = VectorXd::Zero(mp.offsets.encoded_size);
    MatrixXd jacobian = MatrixXd::Zero(vp.offsets.encoded_size, mp.offsets.encoded_size);

    stan::math::jacobian(GetMoments, theta, moments, jacobian);

    bool jac_correct = CheckJacobianCorrectOrientation();
    if (!jac_correct) {
        MatrixXd jacobian_t = jacobian.transpose();
        jacobian = jacobian_t;
    }

    // Abuse the Derivatives type.
    return Derivatives(0, moments, jacobian);
};


// Get derivatives of the log prior.
Derivatives GetLogPriorDerivatives(
    VariationalParameters<double> &vp,
    PriorParameters<double> const &pp,
    bool const unconstrained,
    bool const calculate_gradient,
    bool const calculate_hessian) {

    vp.unconstrained = unconstrained;
    MicroCreditLogPrior LogPrior(vp, pp);

    double val;
    VectorXd grad = VectorXd::Zero(vp.offsets.encoded_size);
    MatrixXd hess = MatrixXd::Zero(vp.offsets.encoded_size, vp.offsets.encoded_size);
    VectorXd theta = GetParameterVector(vp);

    stan::math::set_zero_all_adjoints();

    if (calculate_hessian) {
        stan::math::hessian(MicroCreditLogPrior, theta, val, grad, hess);
    } else if (calculate_gradient) {
        stan::math::gradient(MicroCreditLogPrior, theta, val, grad);
    } else {
        val = MicroCreditLogPrior(theta);
    }

    return Derivatives(val, grad, hess);
}


// Get the covariance of the moment parameters from the natural parameters.
SparseMatrix<double> GetCovariance(
    const VariationalParameters<double> &vp,
    const Offsets moment_offsets) {

  std::vector<Triplet> all_terms;
  std::vector<Triplet> terms;

  if (vp.offsets.encoded_size != moment_offsets.encoded_size) {
      std::ostringstream err_msg;
      err_msg << "Size mismatch.  Natural parameter encoded size: " <<
          vp.offsets.encoded_size << "  Moment parameter encoded size: " <<
          moment_offsets.encoded_size;
      throw std::runtime_error(err_msg.str());
  }

  terms = GetMomentCovariance(vp.mu, moment_offsets.mu);
  all_terms.insert(all_terms.end(), terms.begin(), terms.end());

  terms = GetMomentCovariance(vp.lambda, moment_offsets.lambda);
  all_terms.insert(all_terms.end(), terms.begin(), terms.end());

  for (int g = 0; g < vp.n_g; g++) {
      terms = GetMomentCovariance(vp.mu_g[g], moment_offsets.mu_g[g]);
      all_terms.insert(all_terms.end(), terms.begin(), terms.end());

      terms = GetMomentCovariance(vp.tau[g], moment_offsets.tau[g]);
      all_terms.insert(all_terms.end(), terms.begin(), terms.end());
  }

  // Construct a sparse matrix.
  SparseMatrix<double>
    theta_cov(moment_offsets.encoded_size, moment_offsets.encoded_size);
  theta_cov.setFromTriplets(all_terms.begin(), all_terms.end());
  theta_cov.makeCompressed();

  return theta_cov;
};


SparseMatrix<double> GetSparseELBOHessian(
    MicroCreditData data,
    VariationalParameters<double> vp,
    PriorParameters<double> pp) {

    // The group terms.
    MicroCreditGroupElbo GroupElbo(data, vp, pp, 0);
    std::vector<Triplet> all_terms = GetSparseGroupHessian(GroupElbo, vp);

    // The global terms.
    MicroCreditGlobalElbo GlobalElbo(data, vp, pp);
    VectorXd theta = GetGlobalParameterVector(vp);

    double val;
    VectorXd grad = VectorXd::Zero(theta.size());
    MatrixXd hess = MatrixXd::Zero(theta.size(), theta.size());
    stan::math::hessian(GlobalElbo, theta, val, grad, hess);

    for (int i1 = 0; i1 < hess.rows(); i1++) {
        for (int i2 = 0; i2 < hess.cols(); i2++) {
            all_terms.push_back(Triplet(i1, i2, hess(i1, i2)));
        }
    }

    // Construct a sparse matrix.
    SparseMatrix<double>
      elbo_hess(vp.offsets.encoded_size, vp.offsets.encoded_size);
    elbo_hess.setFromTriplets(all_terms.begin(), all_terms.end());
    elbo_hess.makeCompressed();

    return elbo_hess;
};




// Convert index from a "local" vector containing first the global variables
// and then a single local variable to indices into a "global" vector containing
// first the global variables and then all the local variables.
int GlobalIndex(int index, int g, Offsets offsets) {
    if (index < offsets.local_offset) {
        // It is a "global" index: here, mu or lambda.
        return index;
    } else {
        // It is a local index whose location in the global array is offset
        // relative to the group vector.
        return index + g * offsets.local_encoded_size;
    }
};
