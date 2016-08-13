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
  bool const calculate_hessian) {

    vp.unconstrained = unconstrained;
    // vp.lambda.diag_min = diag_min;
    // vp.mu.diag_min = diag_min;
    // for (int g=0; g < vp.n_g; g++) {
    //     vp.mu_g[g].diag_min = diag_min;
    // }

    MicroCreditElbo ELBO(data, vp, pp);

    double val;
    VectorXd grad = VectorXd::Zero(vp.offsets.encoded_size);
    MatrixXd hess = MatrixXd::Zero(vp.offsets.encoded_size, vp.offsets.encoded_size);
    VectorXd theta = GetParameterVector(vp);

    stan::math::set_zero_all_adjoints();
    if (calculate_hessian) {
        stan::math::hessian(ELBO, theta, val, grad, hess);
    } else {
        stan::math::gradient(ELBO, theta, val, grad);
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
}
