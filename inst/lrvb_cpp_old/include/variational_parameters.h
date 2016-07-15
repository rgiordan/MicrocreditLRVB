# ifndef VARIATIONAL_PARAMETERS_H
# define VARIATIONAL_PARAMETERS_H

# include <Eigen/Dense>
# include <vector>

# include "exponential_families.h"

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Dynamic;

template <typename T> using VectorXT = Eigen::Matrix<T, Dynamic, 1>;
template <typename T> using MatrixXT = Eigen::Matrix<T, Dynamic, Dynamic>;


////////////////////////////////////
// Positive definite matrices

// The index in a vector of lower diagonal terms of a particular matrix value.
int get_ud_index(int i, int j);

template <class T> class PosDefMatrixParameter {
// private:

public:
  int size;
  int size_ud;
  MatrixXT<T> mat;

  PosDefMatrixParameter(int size_): size(size_) {
    size_ud = size * (size + 1) / 2;
    mat = MatrixXT<T>::Zero(size, size);
  }

  PosDefMatrixParameter() {
    PosDefMatrixParameter(1);
  }

  template <typename TNumeric>
  void set(MatrixXT<TNumeric> set_value) {
    for (int row=0; row < size; row++) {
      for (int col=0; col <= row; col++) {
        T this_value = T(set_value(row, col));
        mat(row, col) = this_value;
        mat(col, row) = this_value;
      }
    }
  }

  template <typename TNumeric>
  void set_vec(VectorXT<TNumeric> set_value) {
    for (int row=0; row < size; row++) {
      for (int col=0; col <= row; col++) {
        T this_value = T(set_value(get_ud_index(row, col)));
        mat(row, col) = this_value;
        mat(col, row) = this_value;
      }
    }
  }

  // TODO: don't use the get() method.
  MatrixXT<T> get() const {
    return mat;
  }

  VectorXT<T> get_vec() const {
    VectorXT<T> vec_value(size_ud);
    for (int row=0; row < size; row++) {
      for (int col=0; col <= row; col++) {
        vec_value(get_ud_index(row, col)) = mat(row, col);
      }
    }
    return vec_value;
  }

  template <typename Tnew>
  operator PosDefMatrixParameter<Tnew>() const {
    PosDefMatrixParameter<Tnew> pdmp = PosDefMatrixParameter<Tnew>(size);
    pdmp.set(mat);
    return pdmp;
  }

  // Set from an unconstrained matrix parameterization based on the Cholesky
  // decomposition.
  void set_unconstrained_vec(MatrixXT<T> set_value, T min_diag = 0.0) {
    if (min_diag < 0) {
      throw std::runtime_error("min_diag must be non-negative");
    }
    MatrixXT<T> chol_mat(size, size);
    for (int row=0; row < size; row++) {
      for (int col=0; col <= row; col++) {
        T this_value = T(set_value(get_ud_index(row, col)));
        if (col == row) {
          chol_mat(row, col) = exp(this_value);
        } else {
          chol_mat(row, col) = this_value;
          chol_mat(col, row) = 0.0;
        }
      }
    }
    mat = chol_mat * chol_mat.transpose();
    if (min_diag > 0) {
      for (int k = 0; k < size; k++) {
        mat(k, k) += min_diag;
      }
    }
  }

  VectorXT<T> get_unconstrained_vec(T min_diag = 0.0)  const {
    if (min_diag < 0) {
      throw std::runtime_error("min_diag must be non-negative");
    }
    VectorXT<T> free_vec(size_ud);
    MatrixXT<T> mat_minus_diag(mat);
    if (min_diag > 0) {
      for (int k = 0; k < size; k++) {
        mat_minus_diag(k, k) -= min_diag;
        if (mat_minus_diag(k, k) <= 0) {
          throw std::runtime_error("Posdef diagonal entry out of bounds");
        }
      }
    }
    MatrixXT<T> chol_mat = mat_minus_diag.llt().matrixL();
    for (int row=0; row < size; row++) {
      for (int col=0; col <= row; col++) {
        if (col == row) {
          free_vec(get_ud_index(row, col)) = log(chol_mat(row, col));
        } else {
          free_vec(get_ud_index(row, col)) = chol_mat(row, col);
        }
      }
    }
    return free_vec;
  }
};


/////////////////////////////////////
// Variational parameters.

template <class T> class Gamma {
public:

  // TODO: constrain these to always be consistent?
  T e;
  T e_log;

  Gamma() {
    e = 1;
    e_log = 0;
  };

  template <typename Tnew> operator Gamma<Tnew>() const {
    Gamma<Tnew> gamma_new;
    gamma_new.e = e;
    gamma_new.e_log = e_log;

    return gamma_new;
  };

};


template <class T> class WishartNatural {
public:
  int dim;

  PosDefMatrixParameter<T> v;
  T n;

  WishartNatural(int dim): dim(dim) {
    n = 0;
    v = PosDefMatrixParameter<T>(dim);
    v.mat = MatrixXT<T>::Zero(dim, dim);
  };

  WishartNatural() {
    WishartNatural(1);
  }

  template <typename Tnew> operator WishartNatural<Tnew>() const {
    WishartNatural<Tnew> wishart_new(dim);
    wishart_new.dim = dim;
    wishart_new.v = v;
    wishart_new.n = n;
    return wishart_new;
  };
};


template <class T> class WishartMoments {
public:
  int dim;

  PosDefMatrixParameter<T> e;
  T e_log_det;

  WishartMoments(int dim): dim(dim) {
    e_log_det = 0;
    e = PosDefMatrixParameter<T>(dim);
    e.mat = MatrixXT<T>::Zero(dim, dim);
  };

  WishartMoments() {
    WishartMoments(1);
  }

  WishartMoments(WishartNatural<T> wishart_nat) {
    dim = wishart_nat.dim;
    e = PosDefMatrixParameter<T>(dim);
    e.mat = wishart_nat.v.mat * wishart_nat.n;
    e_log_det = GetELogDetWishart(wishart_nat.v.mat, wishart_nat.n);
  }

  template <typename Tnew> operator WishartMoments<Tnew>() const {
    WishartMoments<Tnew> wishart_new(dim);
    wishart_new.e = e;
    wishart_new.e_log_det = e_log_det;
    return wishart_new;
  };
};


template <class T> class MultivariateNormal {
public:
  int dim;
  VectorXT<T> e_vec;
  PosDefMatrixParameter<T> e_outer;

  MultivariateNormal(int dim): dim(dim) {
  e_vec = VectorXT<T>::Zero(dim);
    e_outer = PosDefMatrixParameter<T>(dim);
    e_outer.mat = MatrixXT<T>::Zero(dim, dim);
  };

  MultivariateNormal() {
    MultivariateNormal(1);
  };

  // Set from a vector of another type.
  template <typename Tnew> MultivariateNormal(VectorXT<Tnew> mean) {
    dim = mean.size();
    e_vec = mean.template cast<T>();
    MatrixXT<T> e_vec_outer = e_vec * e_vec.transpose();
    e_outer = PosDefMatrixParameter<Tnew>(dim);
    e_outer.set(e_vec_outer);
  };

  // Convert to another type.
  template <typename Tnew> operator MultivariateNormal<Tnew>() const {
    MultivariateNormal<Tnew> mvn(dim);
    mvn.e_vec = e_vec.template cast <Tnew>();
    mvn.e_outer.mat = e_outer.mat.template cast<Tnew>();
    return mvn;
  };

  // If this MVN is distributed N(mean, info^-1), get the expected log likelihood.
  T ExpectedLogLikelihood(MultivariateNormal<T> mean, WishartMoments<T> info) const {
    MatrixXT<T> mean_outer_prods = mean.e_vec * e_vec.transpose() +
                                   e_vec * mean.e_vec.transpose();
    return
      -0.5 * (info.e.mat * (e_outer.mat - mean_outer_prods + mean.e_outer.mat)).trace() +
      0.5 * info.e_log_det;
  };

  T ExpectedLogLikelihood(VectorXT<T> mean, WishartMoments<T> info) const {
    MatrixXT<T> mean_outer_prods = mean * e_vec.transpose() +
                                   e_vec * mean.transpose();
    MatrixXT<T> mean_outer = mean * mean.transpose();
    return
      -0.5 * (info.e.mat * (e_outer.mat - mean_outer_prods + mean_outer)).trace() +
      0.5 * info.e_log_det;
  };

  T ExpectedLogLikelihood(VectorXT<T> mean, MatrixXT<T> info) const {
    MatrixXT<T> mean_outer_prods = mean * e_vec.transpose() +
                                   e_vec * mean.transpose();
    MatrixXT<T> mean_outer = mean * mean.transpose();
    return
      -0.5 * (info * (e_outer.mat - mean_outer_prods + mean_outer)).trace() +
      0.5 * log(info.determinant());
  };
};


template <class T> class UnivariateNormal {
public:
  T e;  // The expectation.
  T e2; // The expectation of the square.

  UnivariateNormal() {
    e = 0;
    e2 = 0;
  };

  // Set from a vector of another type.
  template <typename Tnew> UnivariateNormal(Tnew mean) {
    e = mean;
    e2 = mean * mean;
  };

  // Convert to another type.
  template <typename Tnew> operator UnivariateNormal<Tnew>() const {
    UnivariateNormal<Tnew> uvn;
    uvn.e = e;
    uvn.e2 = e2;
    return uvn;
  };

  // If this MVN is distributed N(mean, info^-1), get the expected log likelihood.
  T ExpectedLogLikelihood(UnivariateNormal<T> mean, Gamma<T> info) const {
    return -0.5 * info.e * (e2 - 2 * mean.e * e + mean.e2) + 0.5 * info.e_log;
  };

  T ExpectedLogLikelihood(T mean, Gamma<T> info) const {
    return -0.5 * info.e * (e2 - 2 * mean * e + mean * mean) + 0.5 * info.e_log;
  };

  T ExpectedLogLikelihood(T mean, T info) const {
    return -0.5 * info.e * (e2 - 2 * mean * e + mean * mean) + 0.5 * log(info);
  };
};


# endif
