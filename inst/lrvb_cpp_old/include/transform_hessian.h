# ifndef TRANSFORM_HESSIAN_H
# define TRANSFORM_HESSIAN_H

# include <Eigen/Dense>
# include <vector>

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
    VectorXd df_dy, MatrixXd d2f_dy2);

# endif
