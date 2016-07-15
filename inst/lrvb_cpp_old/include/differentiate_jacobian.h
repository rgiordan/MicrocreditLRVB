# ifndef DIFFERENTIATE_JACOBIAN_H
# define DIFFERENTIATE_JACOBIAN_H

# include <Eigen/Dense>
# include <vector>

# include "transform_hessian.h"

# include <stan/math.hpp>
# include <stan/math/mix/mat/functor/hessian.hpp>


using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;

using Eigen::Dynamic;
using Eigen::VectorXd;

template<typename T> using VectorXT = Eigen::Matrix<T, Dynamic, 1>;

// Given a vector to vector function y_to_x, return only x[i].
template <typename F>
struct y_to_x_index_functor {
  int i; // Which index to return.
  F y_to_x; // The function.

  y_to_x_index_functor(F y_to_x): y_to_x(y_to_x) {
    i = 0;
  };

  template <typename T> T operator()(VectorXT<T> const &y) const {
    return y_to_x(y)(i);
  }
};


// Get a vector of hessians where
// d2x_dy2_vec[k] = dx[k] / dy dy^T
template <typename F>
vector<MatrixXd> GetJacobianHessians(F y_to_x, VectorXd y) {

  int K = y.size();
  MatrixXd d2x_dy2_ind(K, K);
  vector<MatrixXd> d2x_dy2_vec(K);

  y_to_x_index_functor<F> y_to_x_index(y_to_x);

  double x_unused;
  VectorXd x_grad_unused(K);

  for (int k = 0; k < K; k++) {
    y_to_x_index.i = k;
    stan::math::hessian(y_to_x_index, y, x_unused, x_grad_unused, d2x_dy2_ind);
    d2x_dy2_vec[k] = d2x_dy2_ind;
  }

  return d2x_dy2_vec;
}

# endif
