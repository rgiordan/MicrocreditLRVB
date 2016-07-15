# include "monte_carlo_parameters.h"
# include "gtest/gtest.h"

# include <Eigen/Dense>

using Eigen::VectorXd;

struct MeanAndVar {
  double mean;
  double var;
};


MeanAndVar GetMeanAndVar(VectorXd vec) {
  MeanAndVar result; // (mean, variance)
  result.mean = vec.sum() / vec.size();
  double sum_sq = 0.0;
  for (int i = 0; i < vec.size(); i++) {
    sum_sq += pow(vec(i), 2);
  }
  // vec.unaryExpr([](double x) { return pow(x, 2); });  // Why u no work?
  result.var = sum_sq / vec.size() - pow(result.mean, 2);
  return result;
};


TEST(monte_carlo_parameters, is_correct) {
  int n_sim = 1000;
  MonteCarloNormalParameter<double> norm_param(n_sim);

  MeanAndVar mean_and_var;
  mean_and_var = GetMeanAndVar(norm_param.std_draws);
  EXPECT_TRUE(abs(mean_and_var.mean) < 3 / sqrt(n_sim)) << mean_and_var.mean;
  EXPECT_TRUE(abs(mean_and_var.var - 1.0) < 6 / sqrt(n_sim))  << mean_and_var.var;

  double target_mean = 3.0;
  double target_var = 7.5;
  VectorXd check_vec = norm_param.Evaluate(target_mean, target_var);

  mean_and_var = GetMeanAndVar(check_vec);
  EXPECT_TRUE(abs(mean_and_var.mean - target_mean) < 3 / sqrt(n_sim)) <<
    mean_and_var.mean;
  EXPECT_TRUE(abs(mean_and_var.var - target_var) < 6 / sqrt(n_sim))  <<
    mean_and_var.var;

};


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
