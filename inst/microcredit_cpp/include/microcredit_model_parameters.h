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

  VectorParameter<T> e_mu;    // The vector of E(mu)
  PosDefMatrixParameter<T> e_mu2;   // E(mu mu^T)

  // Unlike the other parameters, use the natural parameters for lambda.
  PosDefMatrixParameter<T> lambda_v_par;   // lambda ~ Wishart(v_par, n_par)
  ScalarParameter<T> lambda_n_par;

  // A vector of per-group, E(tau), the observation noise precision
  vector<ScalarParameter<T>> e_tau_vec;
  vector<ScalarParameter<T>> e_log_tau_vec;  // E(log(tau))

  // Vectors of the per-group means.
  vector<VectorParameter<T>> e_mu_g_vec;  // E(mu_k)
  vector<PosDefMatrixParameter<T>> e_mu2_g_vec; // E(mu_k mu_k^T)

  // Methods:
  VariationalParameters(int k, int n_g): k(k), n_g(n_g) {
    e_mu = VectorParameter<T>(k, "e_mu");
    e_mu2 = PosDefMatrixParameter<T>(k, "e_mu2");

    lambda_v_par = PosDefMatrixParameter<T>(k, "lambda_v_par");
    lambda_n_par = ScalarParameter<T>("lambda_n_par");

    // Do the mu_g vectors
    e_mu_g_vec.resize(n_g);
    e_mu2_g_vec.resize(n_g);
    e_tau_vec.resize(n_g);
    e_log_tau_vec.resize(n_g);

    for (int g = 0; g < n_g; g++) {
      e_mu_g_vec[g] = VectorParameter<T>(k, "e_mu_g" + g);
      e_mu2_g_vec[g] = PosDefMatrixParameter<T>(k, "e_mu2_g" + g);
      e_tau_vec[g] = ScalarParameter<T>("e_tau_g" + g);
      e_log_tau_vec[g] = ScalarParameter<T>("e_log_tau_g" + g);
    }
  };


  VariationalParameters() {
    VariationalParameters(1, 1);
  }


  /////////////////
  template <typename Tnew>
  operator VariationalParameters<Tnew>() const {
    VariationalParameters<Tnew> vp = VariationalParameters<Tnew>(k, n_g);

    vp.e_mu = e_mu;
    vp.e_mu2 = e_mu2;

    vp.lambda_v_par = lambda_v_par;
    vp.lambda_n_par = lambda_n_par;

    for (int g = 0; g < n_g; g++) {
      vp.e_mu_g_vec[g] = e_mu_g_vec[g];
      vp.e_mu2_g_vec[g] = e_mu2_g_vec[g];

      vp.e_tau_vec[g] = e_tau_vec[g];
      vp.e_log_tau_vec[g] = e_log_tau_vec[g];
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
  VectorParameter<T> mu_mean;
  PosDefMatrixParameter<T> mu_info;

  // lambda ~ LKJ(eta), scale ~ Gamma(alpha, beta)
  ScalarParameter<T> lambda_eta;
  ScalarParameter<T> lambda_alpha;
  ScalarParameter<T> lambda_beta;

  // tau ~ Gamma(alpha, beta)
  ScalarParameter<T> tau_alpha;
  ScalarParameter<T> tau_beta;

  // Optimization parameters for lambda
  T lambda_diag_min;
  T lambda_n_min;


  // Methods:
  PriorParameters(int k): k(k) {
    mu_mean = VectorParameter<T>(k, "mu_mean");
    mu_info = PosDefMatrixParameter<T>(k, "mu_info");

    lambda_eta = ScalarParameter<T>("lambda_eta");
    lambda_alpha = ScalarParameter<T>("lambda_alpha");
    lambda_beta = ScalarParameter<T>("lambda_beta");

    tau_alpha = ScalarParameter<T>("tau_alpha");
    tau_beta = ScalarParameter<T>("tau_beta");

    lambda_diag_min = 0.0;
    lambda_n_min = k + 0.01;
  };


  PriorParameters() {
    PriorParameters(1);
  };


  template <typename Tnew> operator PriorParameters<Tnew>() const {
    PriorParameters<Tnew> pp = PriorParameters<Tnew>(k);

    pp.mu_mean = mu_mean;
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