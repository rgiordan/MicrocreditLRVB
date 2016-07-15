# include <Eigen/Dense>
# include <vector>

# include "transform_hessian.h"
# include "differentiate_jacobian.h"

# include <stan/math.hpp>
# include <stan/math/mix/mat/functor/hessian.hpp>

# include <stan/math/fwd/scal/fun/pow.hpp>
# include <stan/math/fwd/scal/fun/exp.hpp>
# include <stan/math/fwd/scal/fun/log.hpp>
# include <stan/math/fwd/scal/fun/abs.hpp>

# include "gtest/gtest.h"

# include "test_eigen.h"

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;

using Eigen::Dynamic;
using Eigen::VectorXd;

template<typename T> using VectorXT = Eigen::Matrix<T, Dynamic, 1>;
template<typename T> using MatrixXT = Eigen::Matrix<T, Dynamic, Dynamic>;

MatrixXd get_a_mat() {
  MatrixXd a(2, 2);
  a << 3, 1, 1, 2;
  return a;
}

const MatrixXd a_mat = get_a_mat();


struct y_to_x_functor {
  template <typename T> VectorXT<T> operator()(VectorXT<T> const &y) const {
    VectorXT<T> log_x = a_mat.template cast<T>() * y;
    VectorXT<T> x(log_x.size());
    for (int i = 0; i < x.size(); i++) {
      x(i) = exp(log_x(i));
    }
    return x;
  }
};
y_to_x_functor y_to_x;


struct x_to_y_functor {
  template <typename T> VectorXT<T> operator()(VectorXT<T> const &x) const {
    VectorXT<T> log_x(x.size());
    for (int i = 0; i < x.size(); i++) {
      log_x(i) = log(x(i));
    }
    VectorXT<T> y = a_mat.template cast<T>().ldlt().solve(log_x);
    return y;
  }
};
x_to_y_functor x_to_y;


struct f_of_y_functor {
  template <typename T> T operator()(VectorXT<T> const &y) const {
    return pow(y(0), 3) + pow(y(1), 2);
  }
};
f_of_y_functor f_of_y;


struct f_of_x_functor {
  template <typename T> T operator()(VectorXT<T> const &x) const {
    VectorXT<T> y = x_to_y(x);
    return f_of_y(y);
  }
};
f_of_x_functor f_of_x;


TEST(y_to_x_to_y, is_inverse) {
  VectorXd y(2);
  y << 2, 3;
  VectorXd x = y_to_x(y);
  ASSERT_EQ(x.size(), y.size());
  VectorXd y_trans = x_to_y(x);
  ASSERT_EQ(y_trans.size(), x.size());
  EXPECT_VECTOR_EQ(y_trans, y);
  EXPECT_DOUBLE_EQ(f_of_y(y), f_of_x(x));
};


TEST(hessian_transforms, correct) {
  VectorXd y(2);
  y << 0.2, 0.3;

  VectorXd x(2);
  MatrixXd dxt_dy(2, 2);
  MatrixXd dyt_dx(2, 2);

  // Currently, stan::math::jacobian returns the transpose of the Jacobian!
  stan::math::set_zero_all_adjoints();
  stan::math::jacobian(y_to_x, y, x, dxt_dy);
  EXPECT_VECTOR_EQ(x, y_to_x(y));

  stan::math::jacobian(x_to_y, x, y, dyt_dx);
  EXPECT_VECTOR_EQ(y, x_to_y(x));

  double f_y_val;
  VectorXd df_dy(2);
  stan::math::gradient(f_of_y, y, f_y_val, df_dy);

  double f_x_val;
  VectorXd df_dx(2);
  stan::math::gradient(f_of_x, x, f_x_val, df_dx);

  EXPECT_DOUBLE_EQ(f_x_val, f_y_val);

  // Check the tranformation two different ways:

  // The inverse of dxt_dy is dyt_dx.
  VectorXd df_dx_from_jac2 = dyt_dx * df_dy;
  EXPECT_VECTOR_EQ(df_dx, df_dx_from_jac2);

  VectorXd df_dx_from_jac = dxt_dy.colPivHouseholderQr().solve(df_dy);
  EXPECT_VECTOR_EQ(df_dx, df_dx_from_jac);

  // Test the transformed hessian.

  vector<MatrixXd> d2x_dy2_vec = GetJacobianHessians(y_to_x, y);

  MatrixXd d2f_dy2(2, 2);
  VectorXd x_grad_unused(2);
  stan::math::hessian(f_of_y, y, f_y_val, df_dy, d2f_dy2);

  MatrixXd d2f_dx2 =
    transform_hessian(dxt_dy.transpose(), d2x_dy2_vec, df_dy, d2f_dy2);

  printf(".\n");
  MatrixXd d2f_dx2_test(2, 2);
  printf(".\n");
  stan::math::hessian(f_of_x, x, f_x_val, x_grad_unused, d2f_dx2_test);

  EXPECT_MATRIX_EQ(d2f_dx2, d2f_dx2_test);
};


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
