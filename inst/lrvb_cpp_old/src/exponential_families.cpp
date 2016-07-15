#include "exponential_families.h"

#include <cmath>

#include <boost/math/special_functions/trigamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>

#include "variational_parameters.h"

#include <Eigen/Sparse>
typedef Eigen::Triplet<double> Triplet; // For populating sparse matrices

// # include <stan/math.hpp>
// # include <stan/math/mix/mat/functor/hessian.hpp>
// using var = stan::math::var;
// using fvar = stan::math::fvar<var>;

// #include <stan/math/fwd/scal/fun/fabs.hpp>
// #include <stan/math/fwd/scal/fun/abs.hpp>
// #include <stan/math/fwd/scal/fun/tan.hpp>
// #include <stan/math/fwd/scal/fun/log.hpp>
// #include <stan/math/fwd/scal/fun/floor.hpp>
// #include <stan/math/fwd/scal/fun/lgamma.hpp>
// #include <stan/math/fwd/scal/fun/digamma.hpp>
// #include <stan/math/fwd/scal/fun/trigamma.hpp> // Missing!

using boost::math::lgamma;
using boost::math::digamma;
using boost::math::trigamma;


////////////////////////////////////////////
// Some helper functions

template double multivariate_lgamma(double x, int p);
// template var multivariate_lgamma(var x, int p);
// template fvar multivariate_lgamma(fvar x, int p);

template double multivariate_digamma(double x, int p);
// template var multivariate_digamma(var x, int p);
// template fvar multivariate_digamma(fvar x, int p);

template double multivariate_trigamma(double x, int p);
// template var multivariate_trigamma(var x, int p);
// template fvar multivariate_trigamma(fvar x, int p);



///////////////////////////
// Multivariate normals

MatrixXd GetNormalCovariance(VectorXd const &e_mu, MatrixXd const &e_mu2) {
  return e_mu2 - (e_mu * e_mu.transpose());
}


// Get Cov(mu_i1 mu_i2, mu_c mu_d) from the moment parameters of a multivariate
// normal distribution.
//
// e_mu = E(mu)
// cov_mu = E(mu mu^T) - E(mu) E(mu^T)
double GetNormalFourthOrderCovariance(
    VectorXd const &e_mu, MatrixXd const &cov_mu,
		int i1, int i2, int j1, int j2) {

  return (cov_mu(i1, j1) * cov_mu(i2, j2) +
          cov_mu(i1, j2) * cov_mu(i2, j1) +
	        cov_mu(i1, j1) * e_mu(i2) * e_mu(j2) +
          cov_mu(i1, j2) * e_mu(i2) * e_mu(j1) +
	        cov_mu(i2, j1) * e_mu(i1) * e_mu(j2) +
          cov_mu(i2, j2) * e_mu(i1) * e_mu(j1));
};


// Get Cov(mu_i, mu_j1 mu_j2) from the moment parameters of a multivariate
// normal distribution.
//
// e_mu = E(mu)
// cov_mu = E(mu mu^T) - E(mu) E(mu^T)
double GetNormalThirdOrderCovariance(
    VectorXd const &e_mu, MatrixXd const &cov_mu, int i, int j1, int j2) {

  return e_mu(j1) * cov_mu(i, j2) + e_mu(j2) * cov_mu(i, j1);
};


///////////////////////////////////////////
// Wishart distributions

// Construct the covariance of the elements of a Wishart-distributed
// matrix.
//
// Args:
//   - v_par: The wishart matrix parameter.
//   - n_par: The n parameter of the Wishart distribution.
//
// Returns:
//   - Cov(w_i1_j1, w_i2_j2), where w_i1_j1 and w_i2_j2 are terms of the
//     Wishart matrix parameterized by  and n_par.
double GetWishartLinearCovariance(
    MatrixXd const &v_par, double n_par, int i1, int j1, int i2, int j2) {

  return (n_par * (v_par(i1, j2) * v_par(i2, j1) +
			             v_par(i1, i2) * v_par(j1, j2)));
}


// Construct the covariance between the elements of a Wishart-distributed
// matrix and the log determinant.  A little silly as a function, so
// consider this documentation instead.
//
// Args:
//   - v_par: A linearized representation of the upper triangular portion
//            of the wishart parameter.
//
// Returns:
//   - Cov(w_i1_i2, log(det(w)))
double GetWishartLinearLogDetCovariance(MatrixXd const &v_par, int i1, int i2) {
  return 2.0 * v_par(i1, i2);
}


// As above, but
// Cov(log(det(w), log(det(w))))
// ... where k is the dimension of the matrix.
double GetWishartLogDetVariance(double n_par, int k) {
  return multivariate_trigamma(n_par / 2, k);
}


template double GetELogDetWishart(MatrixXT<double> v_par, double n_par);
// template var GetELogDetWishart(MatrixXT<var> v_par, var n_par);
// template fvar GetELogDetWishart(MatrixXT<fvar> v_par, fvar n_par);

template double GetWishartEntropy(MatrixXT<double> const &v_par, double const n_par);
// template var GetWishartEntropy(MatrixXT<var> const &v_par, var const n_par);
// template fvar GetWishartEntropy(MatrixXT<fvar> const &v_par, fvar const n_par);


////////////////////////////////////////
// Gamma distribution

template double get_e_log_gamma(double alpha, double beta);
// template var get_e_log_gamma(var alpha, var beta);
// template fvar get_e_log_gamma(fvar alpha, fvar beta);


// Return a matrix with Cov((g, log(g))) where
// g ~ Gamma(alpha, beta) (parameterization E[g] = alpha / beta)
MatrixXd get_gamma_covariance(double alpha, double beta) {
	MatrixXd gamma_cov(2, 2);
  gamma_cov(0, 0) = alpha / pow(beta, 2);
  gamma_cov(0, 1) = 1 / beta;
  gamma_cov(1, 0) = gamma_cov(0, 1);
  gamma_cov(1, 1) = boost::math::trigamma(alpha);
  return gamma_cov;
}

////////////////////////////
// Categorical

// Args:
//   p: A size k vector of the z probabilities.
// Returns:
//   The covariance matrix.
MatrixXd GetCategoricalCovariance(VectorXd p) {
  MatrixXd p_outer = (-1) * p * p.transpose();
  MatrixXd p_diagonal = p.asDiagonal();
  p_outer = p_outer + p_diagonal;
  return p_outer;
}


std::vector<Triplet> GetCategoricalCovarianceTerms(VectorXd p, int offset) {
  MatrixXd p_cov = GetCategoricalCovariance(p);
  std::vector<Triplet> terms;
  for (int i=0; i < p_cov.rows(); i++) {
    for (int j=0; j < p_cov.cols(); j++) {
      terms.push_back(Triplet(offset + i, offset + j, p_cov(i, j)));
    }
  }
  return terms;
}


/////////////////////////////////
// Dirichlet

template VectorXT<double> GetELogDirichlet(VectorXT<double> alpha);
// template VectorXT<var> GetELogDirichlet(VectorXT<var> alpha);
// template VectorXT<fvar> GetELogDirichlet(VectorXT<fvar> alpha);


MatrixXd GetLogDirichletCovariance(VectorXd alpha) {
  // Args:
  //  - alpha: A vector of dirichlet parameters.
  //
  // Returns:
  //  - The covariance of the log of a dirichlet distribution
  //    with parameters alpha.

  int k = alpha.size();
  int k_index;
  MatrixXd cov_mat(k, k);

  // Precomute the total.
  double alpha_0 = 0.0;
  for (k_index = 0; k_index < k; k_index++) {
    alpha_0 += alpha(k_index);
  }
  double covariance_term = -1.0 * boost::math::trigamma(alpha_0);
  cov_mat.setConstant(covariance_term);

  // Only the diagonal entries deviate from covariance_term.
  for (k_index = 0; k_index < k; k_index++) {
    cov_mat(k_index, k_index) += boost::math::trigamma(alpha(k_index));
  }
  return cov_mat;
};


template double GetDirichletEntropy(VectorXT<double> alpha);
// template var GetDirichletEntropy(VectorXT<var> alpha);
// template fvar GetDirichletEntropy(VectorXT<fvar> alpha);


///////////////////////////////////
// Coordinates and covariances for sparse matrices

// Assumes that e_mu and e_mu2_offset are stored linearly starting
// at their respective offsets.
std::vector<Triplet> get_mvn_covariance_terms(
    VectorXd e_mu, MatrixXd e_mu2, int e_mu_offset, int e_mu2_offset) {

  std::vector<Triplet> terms;
  int k = e_mu.size();
  if (k != e_mu2.rows() || k !=e_mu2.cols()) {
    throw std::runtime_error("e_mu2 is not square");
  }

  MatrixXd cov_mu = GetNormalCovariance(e_mu, e_mu2);

  // Cov(mu, mu^T)
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < k; j++) {
      terms.push_back(Triplet(e_mu_offset + i, e_mu_offset + j, cov_mu(i, j)));
    }
  }

  // Cov(mu, mu mu^T)
  for (int j1 = 0; j1 < k; j1++) {
    for (int j2 = 0; j2 <= j1; j2++) {
      for (int i = 0; i < k; i++) {
        double this_cov = GetNormalThirdOrderCovariance(e_mu, cov_mu, i, j1, j2);
        terms.push_back(
          Triplet(e_mu_offset + i, e_mu2_offset + get_ud_index(j1, j2),
                  this_cov));
        terms.push_back(
          Triplet(e_mu2_offset + get_ud_index(j1, j2), e_mu_offset + i,
                  this_cov));
      }
    }
  }

  // Cov(mu mu^T, mu mu^T)
  for (int i1 = 0; i1 < k; i1++) { for (int i2 = 0; i2 <= i1; i2++) {
    for (int j1 = 0; j1 < k; j1++) { for (int j2 = 0; j2 <= j1; j2++) {
      double this_cov = GetNormalFourthOrderCovariance(e_mu, cov_mu, i1, i2, j1, j2);
      terms.push_back(Triplet(
        e_mu2_offset + get_ud_index(i1, i2),
        e_mu2_offset + get_ud_index(j1, j2),
        this_cov));
      }}
  }}

  return terms;
};


std::vector<Triplet> get_wishart_covariance_terms(
    MatrixXd v_par, double n_par, int e_lambda_offset, int e_log_det_lambda_offset) {

  std::vector<Triplet> terms;
  int k = v_par.rows();
  if (k != v_par.cols()) {
    throw std::runtime_error("V is not square");
  }

  for (int i1 = 0; i1 < k; i1++) { for (int j1 = 0; j1 <= i1; j1++) {
    int i_ind = e_lambda_offset + get_ud_index(i1, j1);
    double this_cov = GetWishartLinearLogDetCovariance(v_par, i1, j1);
    terms.push_back(Triplet(i_ind, e_log_det_lambda_offset, this_cov));
    terms.push_back(Triplet(e_log_det_lambda_offset, i_ind, this_cov));
	  for (int i2 = 0; i2 < k; i2++) { for (int j2 = 0; j2 <= i2; j2++) {
      int j_ind = e_lambda_offset + get_ud_index(i2, j2);
	    terms.push_back(Triplet(i_ind, j_ind,
        GetWishartLinearCovariance(v_par, n_par, i1, j1, i2, j2)));
	  }}
  }}
  terms.push_back(Triplet(e_log_det_lambda_offset, e_log_det_lambda_offset,
    GetWishartLogDetVariance(n_par, k)));

  return terms;
};


std::vector<Triplet> get_gamma_covariance_terms(
    double alpha, double beta, int e_tau_offset, int e_log_tau_offset) {

  std::vector<Triplet> terms;
  MatrixXd tau_cov = get_gamma_covariance(alpha, beta);
  terms.push_back(Triplet(e_tau_offset, e_tau_offset, tau_cov(0, 0)));
  terms.push_back(Triplet(e_log_tau_offset, e_tau_offset, tau_cov(0, 1)));
  terms.push_back(Triplet(e_tau_offset, e_log_tau_offset, tau_cov(1, 0)));
  terms.push_back(Triplet(e_log_tau_offset, e_log_tau_offset, tau_cov(1, 1)));

  return terms;
};


std::vector<Triplet> get_dirichlet_covariance_terms(VectorXd alpha, int offset) {
  std::vector<Triplet> terms;
  MatrixXd q_cov = GetLogDirichletCovariance(alpha);
  for (int i=0; i < q_cov.rows(); i++) {
    for (int j=0; j < q_cov.cols(); j++) {
      terms.push_back(Triplet(offset + i, offset + j, q_cov(i, j)));
    }
  }

  return terms;
}
