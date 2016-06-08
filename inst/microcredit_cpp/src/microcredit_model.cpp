# include <Eigen/Dense>
# include <vector>
# include <boost/math/tools/promotion.hpp>

// Special functions for the Hessian.
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>

# include "variational_parameters.h"
# include "exponential_families.h"
# include "microcredit_model_parameters.h"

# include "microcredit_model.h"

# include <stan/math.hpp>
# include <stan/math/mix/mat/functor/hessian.hpp>


using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixXi;
using Eigen::VectorXi;

using Eigen::Matrix;
using Eigen::Dynamic;

template <typename T> using VectorXT = Eigen::Matrix<T, Dynamic, 1>;
template <typename T> using MatrixXT = Eigen::Matrix<T, Dynamic, Dynamic>;

using std::vector;
using typename boost::math::tools::promote_args;

using var = stan::math::var;
using fvar = stan::math::fvar<var>;

////////////////////////////////////////
// Likelihood functions


// extern template double GetHierarchyLogLikelihood(VariationalParameters<double> const &);
// extern template var GetHierarchyLogLikelihood(VariationalParameters<var> const &);
// extern template fvar GetHierarchyLogLikelihood(VariationalParameters<fvar> const &);
//

// template <> double GetObservationLogLikelihood(
//   MicroCreditData const &, VariationalParameters<double> const &);
// template <> var GetObservationLogLikelihood(
//   MicroCreditData const &, VariationalParameters<var> const &);
// template <> fvar GetObservationLogLikelihood(
//   MicroCreditData const &, VariationalParameters<fvar> const &);



// template <> var GetPriorLogLikelihood(
//     VariationalParameters<var> const &, PriorParameters<double> const &);
// template <> var GetPriorLogLikelihood(
//     VariationalParameters<var> const &, PriorParameters<var> const &);
// template <> fvar GetPriorLogLikelihood(
//     VariationalParameters<fvar> const &, PriorParameters<double> const &);
// template <> fvar GetPriorLogLikelihood(
//     VariationalParameters<fvar> const &, PriorParameters<fvar> const &);
