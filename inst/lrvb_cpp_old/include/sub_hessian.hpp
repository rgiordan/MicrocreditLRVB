#ifndef STAN_MATH_MIX_MAT_FUNCTOR_SUB_HESSIAN_HPP
#define STAN_MATH_MIX_MAT_FUNCTOR_SUB_HESSIAN_HPP

#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/rev/core.hpp>
#include <vector>

namespace stan {

  namespace math {

    template <typename F>
    void
    sub_hessian(const F& f,
            vector<int>& indices,
            const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
            double& fx,
            Eigen::Matrix<double, Eigen::Dynamic, 1>& grad,
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& H) {
      H.resize(x.size(), indices.size());
      grad.resize(indices.size());
      try {
        for (int indices_i = 0; indices_i < indices.size(); ++indices_i) {
          int i = indices(indices_i);
          start_nested();
          Eigen::Matrix<fvar<var>, Eigen::Dynamic, 1> x_fvar(x.size());
          for (int j = 0; j < x.size(); ++j)
            x_fvar(j) = fvar<var>(x(j), i == j);
          fvar<var> fx_fvar = f(x_fvar);
          grad(indices_i) = fx_fvar.d_.val();
          if (indices_i == 0) fx = fx_fvar.val_.val();
          stan::math::grad(fx_fvar.d_.vi_);
          for (int j = 0; j < x.size(); ++j)
            H(j, indices_i) = x_fvar(j).val_.adj();
          stan::math::recover_memory_nested();
        }
      } catch (const std::exception& e) {
        stan::math::recover_memory_nested();
        throw;
      }
    }

  }  // namespace math
}  // namespace stan
#endif
