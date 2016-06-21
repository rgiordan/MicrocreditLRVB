# ifndef MICROCREDIT_MODEL_PARAMETERS_H
# define MICROCREDIT_MODEL_PARAMETERS_H

# include <Eigen/Dense>
# include <vector>
# include <iostream>

# include "variational_parameters.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixXi;
using Eigen::VectorXi;

using Eigen::Matrix;
using Eigen::Dynamic;

template <typename T> using VectorXT = Eigen::Matrix<T, Dynamic, 1>;
template <typename T> using MatrixXT = Eigen::Matrix<T, Dynamic, Dynamic>;

using std::vector;

//////////////////////////////////////
// VariationalParameters
//////////////////////////////////////

template <class T>
class VariationalParameters {
public:
  // Parameters:
  int n_g;    // The number of groups
  int k;      // The dimension of the means

  // The global mean parameter.
  MultivariateNormal<T> mu;

  // Unlike the other parameters, use the natural parameters for lambda.
  WishartNatural<T> lambda;

  // A vector of per-group, E(tau), the observation noise precision
  vector<Gamma<T>> tau_vec;

  // Vectors of the per-group means.
  vector<MultivariateNormal<T>> mu_g_vec;

  // Methods:
  VariationalParameters(int k, int n_g): k(k), n_g(n_g) {
    mu = MultivariateNormal<T>(k);
    lambda = WishartNatural<T>(k);

    // Per-observation parameters
    mu_g_vec.resize(n_g);
    tau_vec.resize(n_g);

    for (int g = 0; g < n_g; g++) {
      mu_g_vec[g] = MultivariateNormal<T>(k);
      tau_vec[g] = Gamma<T>();
    }
  };


  VariationalParameters() {
    VariationalParameters(1, 1);
  }


  /////////////////
  template <typename Tnew>
  operator VariationalParameters<Tnew>() const {
    VariationalParameters<Tnew> vp = VariationalParameters<Tnew>(k, n_g);

    vp.mu = mu;
    vp.lambda = lambda;

    for (int g = 0; g < n_g; g++) {
      vp.mu_g_vec[g] = mu_g_vec[g];
      vp.tau_vec[g] = tau_vec[g];
    }
    return vp;
  }
};


//////////////////////////////
// Priors

template <class T> class PriorParameters {
public:
  // Parameters:
  int k;      // The dimension of the means

  // mu ~ MNV(mu_mean, mu_info^-1)
  // VectorParameter<T> mu_mean;
  // PosDefMatrixParameter<T> mu_info;
  VectorXT<T> mu_mean;
  PosDefMatrixParameter<T> mu_info;

  // lambda ~ LKJ(eta), scale ~ Gamma(alpha, beta)
  T lambda_eta;
  T lambda_alpha;
  T lambda_beta;

  // tau ~ Gamma(alpha, beta)
  T tau_alpha;
  T tau_beta;

  // Optimization parameters for lambda
  T lambda_diag_min;
  T lambda_n_min;


  // Methods:
  PriorParameters(int k): k(k) {
    mu_mean = VectorXT<T>(k);
    mu_info = PosDefMatrixParameter<T>(k);

    lambda_eta = 1;
    lambda_alpha = 1;
    lambda_beta = 1;

    tau_alpha = 1;
    tau_beta = 1;

    lambda_diag_min = 0.0;
    lambda_n_min = k + 0.01;
  };


  PriorParameters() {
    PriorParameters(1);
  };


  template <typename Tnew> operator PriorParameters<Tnew>() const {
    PriorParameters<Tnew> pp = PriorParameters<Tnew>(k);

    pp.mu_mean = mu_mean.template cast<Tnew>();
    pp.mu_info = mu_info;

    pp.lambda_eta = lambda_eta;
    pp.lambda_alpha = lambda_alpha;
    pp.lambda_beta = lambda_beta;

    pp.tau_alpha = tau_alpha;
    pp.tau_beta = tau_beta;

    pp.lambda_diag_min = lambda_diag_min;
    pp.lambda_n_min = lambda_n_min;

    return pp;
  };
};


//////////////////////////////
// Model data

struct MicroCreditData {

  VectorXd y;           // The observations.
  VectorXi y_g;         // The group indices.
  MatrixXd x;           // The explanatory variables.

  int n;                // The number of observations.
  int k;                // The dimension of X
  int n_g;              // The number of groups

  // Constructor
  MicroCreditData(MatrixXd x, VectorXd y, VectorXi y_g): x(x), y(y), y_g(y_g) {
    if (x.rows() != y.size() || x.rows() != y_g.size()) {
      throw std::runtime_error("Wrong number of rows");
    }

    int min_g_index, max_g_index;
    int min_g = y_g.minCoeff(&min_g_index);
    int max_g = y_g.maxCoeff(&max_g_index);
    if (min_g < 1) {
      throw std::runtime_error("Error -- y_g must have integers between 1 and n_g");
    }

    n = y.size();
    n_g = max_g;
    k = x.cols();
  };

  MicroCreditData() {
    MatrixXd x(1, 1);
    x.setZero();
    VectorXd y(1);
    y.setZero();
    VectorXi y_g(1);
    y_g(0) = 1;
    MicroCreditData(x, y, y_g);
  };

  void print() {
    printf("--------------------------\n");
    std::cout << "X matrix:" << std::endl << x << std::endl << std::endl;
    std::cout << "Y matrix:" << std::endl << y << std::endl << std::endl;
    std::cout << "Y_g matrix:" << std::endl << y_g << std::endl << std::endl;
    printf("--------------------------\n");
  }

};

# endif
