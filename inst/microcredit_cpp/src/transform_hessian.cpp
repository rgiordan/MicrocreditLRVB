# include <Eigen/Dense>
# include <vector>
# include <iostream>

# include "transform_hessian.h"

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;


// Input variables:
//   dx_dy[i, j] = dx[i] / dy[j]
//   d2x_dy2[k][i, j] = d2 x[k] / (dy[i] dy[j])
//   df_dy[i] = df / dy[i]
//   d2f_dy2[i, j] = d2f / (dy[i] dy[j])
//
// Returns:
//   d2f / dx2
MatrixXd transform_hessian(
    MatrixXd dx_dy, vector<MatrixXd> d2x_dy2,
    VectorXd df_dy, MatrixXd d2f_dy2) {

  int K = df_dy.size();
  if (dx_dy.rows() != K || dx_dy.cols() != K) {
    throw std::runtime_error("dx_dy wrong size");
  }
  if (d2x_dy2.size() != K) {
    throw std::runtime_error("d2x_dy2 wrong size");
  }
  if (d2f_dy2.rows() != K || d2f_dy2.cols() != K) {
    throw std::runtime_error("d2x_dy2 wrong size");
  }

  Eigen::FullPivLU<MatrixXd> dy_dx_t_piv(dx_dy.transpose());
  MatrixXd dy_dx_t = dy_dx_t_piv.inverse();

  // The same as dx_dy' \ d2f_dy2 * inv(dx_dy).  This is the only term
	// if you are at an optimum since in that case df_dy = 0.
  MatrixXd d2f_dx2 = dy_dx_t_piv.solve((dy_dx_t_piv.solve(d2f_dy2)).transpose());

  // Get the term that is non-zero away from the optimum.
  VectorXd df_dx = dy_dx_t_piv.solve(df_dy);

  // This is a vector of deriatives of the Jacobian matrix with respect to
  // each component of x, i.e.
  // djac_dxj[j][a, b] = d J[a, b] / dx[j]
  std::vector<MatrixXd> djac_dxj(K);
  for (int k = 0; k < K; k++) {
    MatrixXd empty_matrix(K, K);
    empty_matrix.setZero();
    djac_dxj[k] = empty_matrix;
  }

  // Loop over x[a] in jac[a, b] = dx[a] / dy[b] to minimize the solve()s
  for (int a = 0; a < K; a++) {
    MatrixXd jac_term = dy_dx_t_piv.solve(d2x_dy2[a]);
    for (int j = 0; j < K; j++) {
      for (int b = 0; b < K; b++) {
        (djac_dxj[j])(a, b) = jac_term(j, b);
      }
    }
  }

  // Now loop over the x[j] in d2f / dx dx[j]
  for (int j = 0; j < K; j++) {
    VectorXd term = dy_dx_t_piv.solve(djac_dxj[j].transpose()) * df_dx;
    for (int i = 0; i < K; i++) {
      d2f_dx2(i, j) -= term[i];
    }
  }

  return d2f_dx2;
}
