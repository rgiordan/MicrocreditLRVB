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

/////////////////////////////
// Offset types

struct Offsets {
    int mu;
    int lambda;
    vector<int> tau;
    vector<int> mu_g;

    int encoded_size;

    Offsets() {
        mu = 0;
        lambda = 0;
        tau.resize(0);
        mu_g.resize(0);
        encoded_size = 0;
    }
};


struct PriorOffsets {
    int mu;
    int tau;

    int lambda_eta;
    int lambda_alpha;
    int lambda_beta;

    int encoded_size;

    PriorOffsets() {
        mu = 0;
        tau = 0;
        lambda_eta = 0;
        lambda_alpha = 0;
        lambda_beta =0;
    }
};


//////////////////////////////////////
// VariationalParameters
//////////////////////////////////////

template <class T>
class VariationalParameters {
private:
    void Initialize(int k, int n_g, bool unconstrained) {
        k = k;
        n_g = n_g;
        unconstrained = unconstrained;

        mu = MultivariateNormalNatural<T>(k);
        lambda = WishartNatural<T>(k);

        // Per-observation parameters
        mu_g_vec.resize(n_g);
        tau_vec.resize(n_g);

        for (int g = 0; g < n_g; g++) {
          mu_g_vec[g] = MultivariateNormalNatural<T>(k);
          tau_vec[g] = GammaNatural<T>();
        }
        offsets = GetOffsets(*this);
    }
public:
  // Parameters:
  int n_g;    // The number of groups
  int k;      // The dimension of the means

  bool unconstrained;
  Offsets offsets;

  // The global mean parameter.
  MultivariateNormalNatural<T> mu;

  // The precision matrix of the grou p means.
  WishartNatural<T> lambda;

  // A vector of per-group, E(tau), the observation noise precision
  vector<GammaNatural<T>> tau_vec;

  // Vectors of the per-group means.
  vector<MultivariateNormalNatural<T>> mu_g_vec;

  // Methods:
  VariationalParameters(int k, int n_g, bool unconstrained):
  k(k), n_g(n_g), unconstrained(unconstrained) {
    Initialize(k, n_g, unconstrained);
  };

  VariationalParameters() {
    Initialize(1, 1, false);
  }

  /////////////////
  template <typename Tnew>
  operator VariationalParameters<Tnew>() const {
    VariationalParameters<Tnew> vp =
        VariationalParameters<Tnew>(k, n_g, unconstrained);

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
private:
    void Initialize(int k) {
        k = k;
        mu = MultivariateNormalNatural<T>(k);

        lambda_eta = 1;
        lambda_alpha = 1;
        lambda_beta = 1;

        tau = GammaNatural<T>();

        lambda_diag_min = 0.0;
        lambda_n_min = k + 0.01;

        offsets = PriorOffsets(*this);
    }
public:
  // Parameters:
  int k;      // The dimension of the means
  PriorOffsets offsets;

  MultivariateNormalNatural<T> mu;
  GammaNatural<T> tau;

  // lambda ~ LKJ(eta), scale ~ Gamma(alpha, beta)
  T lambda_eta;
  T lambda_alpha;
  T lambda_beta;

  // Optimization parameters for lambda
  T lambda_diag_min;
  T lambda_n_min;

  // Methods:
  PriorParameters(int k): k(k) {
      Initialize(k);
  };


  PriorParameters() {
      Initialize(1);
  };


  template <typename Tnew> operator PriorParameters<Tnew>() const {
    PriorParameters<Tnew> pp = PriorParameters<Tnew>(k);

    pp.mu = mu;

    pp.lambda_eta = lambda_eta;
    pp.lambda_alpha = lambda_alpha;
    pp.lambda_beta = lambda_beta;

    pp.tau = tau;

    pp.lambda_diag_min = lambda_diag_min;
    pp.lambda_n_min = lambda_n_min;

    return pp;
  };
};


/////////////////////////////////////
// Offsets

template <typename T, template<typename> class VPType>
Offsets GetOffsets(VPType<T> vp) {
    Offsets offsets;
    int encoded_size = 0;

    offsets.mu = encoded_size;
    encoded_size += vp.mu.encoded_size;

    offsets.lambda = encoded_size;
    encoded_size += vp.lambda.encoded_size;

    offsets.tau.resize(vp.n_g);
    for (int g = 0; g < vp.n_g; g++) {
        offsets.tau[g] = encoded_size;
        encoded_size += vp.tau[g].encoded_size;
    }

    offsets.mu_g.resize(vp.n_g);
    for (int g = 0; g < vp.n_g; g++) {
        offsets.mu_g[g] = encoded_size;
        encoded_size += vp.mu_g[g].encoded_size;
    }

    offsets.encoded_size = encoded_size;

    return offsets;
};


template <typename T> PriorOffsets GetPriorOffsets(PriorParameters<T> pp) {
    PriorOffsets offsets;
    int encoded_size = 0;

    offsets.mu = encoded_size; encoded_size += pp.mu.encoded_size;
    offsets.tau = encoded_size; encoded_size += pp.tau.encoded_size;
    offsets.lambda_eta = encoded_size; encoded_size += 1;
    offsets.lambda_alpha = encoded_size; encoded_size += 1;
    offsets.lambda_beta = encoded_size; encoded_size += 1;

    offsets.encoded_size = encoded_size;

    return offsets;
};



///////////////////////////////
// Converting to and from vectors

template <typename T, template<typename> class VPType>
VectorXT<T> GetParameterVector(VPType<T> vp) {
  VectorXT<T> theta(vp.offsets.encoded_size);

  theta.segment(vp.offsets.mu, vp.mu.encoded_size) =
    vp.mu.encode_vector(vp.unconstrained);

  theta.segment(vp.offsets.lambda, vp.lambda.encoded_size) =
    vp.lambda.encode_vector(vp.unconstrained);

  for (int g = 0; g < vp.tau_vec.size(); g++) {
    theta.segment(vp.offsets.tau_vec[g], vp.tau_vec[g].encoded_size) =
        vp.tau_vec[g].encode_vector(vp.unconstrained);
  }

  for (int g = 0; g < vp.mu_g_vec.size(); g++) {
    theta.segment(vp.offsets.mu_g_vec[g], vp.mu_g_vec[g].encoded_size) =
        vp.mu_g_vec[g].encode_vector(vp.unconstrained);
  }

  return theta;
}


template <typename T, template<typename> class VPType>
void SetFromVector(VectorXT<T> const &theta, VPType<T> &vp) {
  if (theta.size() != vp.offsets.encoded_size) {
      throw std::runtime_error("Vector is wrong size.");
  }

  VectorXT<T> theta_sub;

  theta_sub = theta.segment(vp.offsets.mu, vp.mu.encoded_size);
  vp.mu.decode_vector(theta_sub, vp.unconstrained);

  theta_sub = theta.segment(vp.offsets.lambda, vp.lambda.encoded_size);
  vp.lambda.decode_vector(theta_sub, vp.unconstrained);

  for (int g = 0; g < vp.tau_vec.size(); g++) {
      theta_sub = theta.segment(vp.offsets.tau_vec[g], vp.tau_vec[g].encoded_size);
      vp.tau_vec[g].decode_vector(theta_sub, vp.unconstrained);
  }

  for (int g = 0; g < vp.mu_g_vec.size(); g++) {
      theta_sub = theta.segment(vp.offsets.mu_g_vec[g], vp.mu_g_vec[g].encoded_size);
      vp.mu_g_vec[g].decode_vector(theta_sub, vp.unconstrained);
  }
}


template <typename T> VectorXT<T> GetParameterVector(PriorParameters<T> pp) {
  VectorXT<T> theta(pp.offsets.encoded_size);

  theta.segment(pp.offsets.mu, pp.mu.encoded_size) = pp.mu.encode_vector(false);
  theta.segment(pp.offsets.tau, pp.tau.encoded_size) = pp.tau.encode_vector(false);

  theta(pp.offsets.lambda_eta) = pp.lambda_eta;
  theta(pp.offsets.lambda_alpha) = pp.lambda_eta;
  theta(pp.offsets.lambda_beta) = pp.lambda_eta;

  return theta;
}


template <typename T>
void SetFromVector(VectorXT<T> const &theta, PriorParameters<T> &pp) {
  if (theta.size() != pp.offsets.encoded_size) {
      throw std::runtime_error("Vector is wrong size.");
  }

  VectorXT<T> theta_sub;

  theta_sub = theta.segment(pp.offsets.mu, pp.mu.encoded_size);
  pp.mu.decode_vector(theta_sub, false);

  theta_sub = theta.segment(pp.offsets.tau, pp.tau.encoded_size);
  pp.tau.decode_vector(theta_sub, false);

  pp.lambda_eta = theta(pp.offsets.lambda_eta);
  pp.lambda_alpha = theta(pp.offsets.lambda_alpha);
  pp.lambda_beta = theta(pp.offsets.lambda_beta);
}

//////////////////////////////
// Model data

struct MicroCreditData {

  VectorXd y;           // The observations.
  VectorXi y_g;         // The one-indexed group indices.
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
        std::ostringstream error_msg;
        error_msg <<
          "Error -- y_g must have integers between 1 and n_groups. " <<
          "Got min(y_g) = "  << min_g << " and max(y_g)  = " << max_g;
      throw std::runtime_error(error_msg.str());
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
    y_g(0) = 0;
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
