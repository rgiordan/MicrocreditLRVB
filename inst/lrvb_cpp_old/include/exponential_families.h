# ifndef EXPONENTIAL_FAMILIES_H
# define EXPONENTIAL_FAMILIES_H

#include <cmath>

#include <boost/math/special_functions/trigamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>

#include "variational_parameters.h"

#include <Eigen/Sparse>
typedef Eigen::Triplet<double> Triplet; // For populating sparse matrices

// #include <stan/math/fwd/scal/fun/lgamma.hpp>
// #include <stan/math/fwd/scal/fun/digamma.hpp>
// #include <stan/math/fwd/scal/fun/trigamma.hpp> // Missing!

using boost::math::lgamma;
using boost::math::digamma;
using boost::math::trigamma;

// # include <stan/math.hpp>
// # include <stan/math/mix/mat/functor/hessian.hpp>
//
// using var = stan::math::var;
// using fvar = stan::math::fvar<var>;

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Dynamic;

template <typename T> using VectorXT = Eigen::Matrix<T, Dynamic, 1>;
template <typename T> using MatrixXT = Eigen::Matrix<T, Dynamic, Dynamic>;


////////////////////////////////////////////
// Some helper functions

template <typename T> T multivariate_lgamma(T x, int p) {
  T result = log(M_PI) * p * (p - 1.0) / 4.0;
  for (int i = 1; i <= p; i++) {
    result += lgamma(x + 0.5 * (1 - (double)i));
  }
  return result;
}


template <typename T> T multivariate_digamma(T x, int p) {
  T result = 0.0;
  for (int i = 1; i <= p; i++) {
    result += digamma(x + 0.5 * (1 - (double)i));
  }
  return result;
}


// Note: this won't work with STAN hessians..
template <typename T> T multivariate_trigamma(T x, int p) {
 T result = 0.0;
  for (int i = 1; i <= p; i++) {
    result += trigamma(x + 0.5 * (1 - (double)i));
  }
  return result;
}

extern template double multivariate_lgamma(double x, int p);
// extern template var multivariate_lgamma(var x, int p);
// extern template fvar multivariate_lgamma(fvar x, int p);

extern template double multivariate_digamma(double x, int p);
// extern template var multivariate_digamma(var x, int p);
// extern template fvar multivariate_digamma(fvar x, int p);

extern template double multivariate_trigamma(double x, int p);
// extern template var multivariate_trigamma(var x, int p);
// extern template fvar multivariate_trigamma(fvar x, int p);


///////////////////////////
// Multivariate normals

template <typename T>
T GetELogDetWishart(MatrixXT<T> v_par, T n_par) {
  int k = v_par.rows();
  if (v_par.rows() != v_par.cols()) {
    throw std::runtime_error("V is not square");
  }
  T v_par_det = v_par.determinant();
  T e_log_det_lambda =
    log(v_par_det) + k * log(2) + multivariate_digamma(0.5 * n_par, k);
  return e_log_det_lambda;
};


template <typename T>
T GetWishartEntropy(MatrixXT<T> const &v_par, T const n_par) {
  int k = v_par.rows();
  if (v_par.rows() != v_par.cols()) {
    throw std::runtime_error("V is not square");
  }
  T det_v = v_par.determinant();
  T entropy = 0.5 * (k + 1) * log(det_v) +
              0.5 * k * (k + 1) * log(2) +
              multivariate_lgamma(0.5 * n_par, k) -
              0.5 * (n_par - k - 1) * multivariate_digamma(0.5 * n_par, k) +
              0.5 * n_par * k;
  return entropy;
}


extern template double GetELogDetWishart(MatrixXT<double> v_par, double n_par);
// extern template var GetELogDetWishart(MatrixXT<var> v_par, var n_par);
// extern template fvar GetELogDetWishart(MatrixXT<fvar> v_par, fvar n_par);

extern template double GetWishartEntropy(MatrixXT<double> const &v_par, double const n_par);
// extern template var GetWishartEntropy(MatrixXT<var> const &v_par, var const n_par);
// extern template fvar GetWishartEntropy(MatrixXT<fvar> const &v_par, fvar const n_par);


////////////////////////////////////////
// Gamma distribution


template <typename T> T get_e_log_gamma(T alpha, T beta) {
  return digamma(alpha) - log(beta);
}

extern template double get_e_log_gamma(double alpha, double beta);
// extern template var get_e_log_gamma(var alpha, var beta);
// extern template fvar get_e_log_gamma(fvar alpha, fvar beta);

// Return a matrix with Cov((g, log(g))) where
// g ~ Gamma(alpha, beta) (parameterization E[g] = alpha / beta)
MatrixXd get_gamma_covariance(double alpha, double beta);


////////////////////////////
// Categorical

// Args:
//   p: A size k vector of the z probabilities.
// Returns:
//   The covariance matrix.
MatrixXd GetCategoricalCovariance(VectorXd p);


std::vector<Triplet> GetCategoricalCovarianceTerms(VectorXd p, int offset);


/////////////////////////////////
// Dirichlet

template <typename T> VectorXT<T> GetELogDirichlet(VectorXT<T> alpha) {
  // Args:
  //   - alpha: The dirichlet parameters Q ~ Dirichlet(alpha).  Should
  //     be a k-dimensional vector.
  // Returns:
  //   - A k-dimensional matrix representing E(log Q).

  int k = alpha.size();
  T alpha_sum = 0.0;
  for (int index = 0; index < k; index++) {
    alpha_sum += alpha(index);
  }
  T digamma_alpha_sum = boost::math::digamma(alpha_sum);
  VectorXT<T> e_log_alpha(k);
  for (int index = 0; index < k; index++) {
    e_log_alpha(index) = boost::math::digamma(alpha(index)) - digamma_alpha_sum;
  }
  return e_log_alpha;
}


template <typename T> T GetDirichletEntropy(VectorXT<T> alpha) {
  // Args:
  //   - alpha: The dirichlet parameters Q ~ Dirichlet(alpha).  Should
  //     be a k-dimensional vector.
  // Returns:
  //   - The entropy of the Dirichlet distribution.

  int k = alpha.size();
  T alpha_sum = alpha.sum();
  T entropy = 0.0;
  entropy += -lgamma(alpha_sum) - (k - alpha_sum) * digamma(alpha_sum);

  for (int index = 0; index < k; index++) {
    entropy += lgamma(alpha(index)) - (alpha(index) - 1) * digamma(alpha(index));
  }

  return entropy;
}

MatrixXd GetLogDirichletCovariance(VectorXd alpha);


extern template VectorXT<double> GetELogDirichlet(VectorXT<double> alpha);
// extern template VectorXT<var> GetELogDirichlet(VectorXT<var> alpha);
// extern template VectorXT<fvar> GetELogDirichlet(VectorXT<fvar> alpha);

extern template double GetDirichletEntropy(VectorXT<double> alpha);
// extern template var GetDirichletEntropy(VectorXT<var> alpha);
// extern template fvar GetDirichletEntropy(VectorXT<fvar> alpha);


///////////////////////////////////
// Coordinates and covariances for sparse matrices

// Assumes that e_mu and e_mu2_offset are stored linearly starting
// at their respective offsets.
std::vector<Triplet> get_mvn_covariance_terms(
    VectorXd e_mu, MatrixXd e_mu2, int e_mu_offset, int e_mu2_offset);

std::vector<Triplet> get_wishart_covariance_terms(
    MatrixXd v_par, double n_par, int e_lambda_offset, int e_log_det_lambda_offset);


std::vector<Triplet> get_gamma_covariance_terms(
    double alpha, double beta, int e_tau_offset, int e_log_tau_offset);

std::vector<Triplet> get_dirichlet_covariance_terms(VectorXd alpha, int offset);



# endif
