// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// GetEmptyVariationalParameters
Rcpp::List GetEmptyVariationalParameters(int k, int n_g);
RcppExport SEXP MicrocreditLRVB_GetEmptyVariationalParameters(SEXP kSEXP, SEXP n_gSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< int >::type n_g(n_gSEXP);
    __result = Rcpp::wrap(GetEmptyVariationalParameters(k, n_g));
    return __result;
END_RCPP
}
// GetParametersFromVector
Rcpp::List GetParametersFromVector(const Rcpp::List r_vp, const Eigen::Map<Eigen::VectorXd> r_theta, bool unconstrained);
RcppExport SEXP MicrocreditLRVB_GetParametersFromVector(SEXP r_vpSEXP, SEXP r_thetaSEXP, SEXP unconstrainedSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type r_theta(r_thetaSEXP);
    Rcpp::traits::input_parameter< bool >::type unconstrained(unconstrainedSEXP);
    __result = Rcpp::wrap(GetParametersFromVector(r_vp, r_theta, unconstrained));
    return __result;
END_RCPP
}
// GetVectorFromParameters
Eigen::VectorXd GetVectorFromParameters(const Rcpp::List r_vp, bool unconstrained);
RcppExport SEXP MicrocreditLRVB_GetVectorFromParameters(SEXP r_vpSEXP, SEXP unconstrainedSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    Rcpp::traits::input_parameter< bool >::type unconstrained(unconstrainedSEXP);
    __result = Rcpp::wrap(GetVectorFromParameters(r_vp, unconstrained));
    return __result;
END_RCPP
}
// GetParametersFromGlobalVector
Rcpp::List GetParametersFromGlobalVector(const Rcpp::List r_vp, const Eigen::Map<Eigen::VectorXd> r_theta, bool unconstrained);
RcppExport SEXP MicrocreditLRVB_GetParametersFromGlobalVector(SEXP r_vpSEXP, SEXP r_thetaSEXP, SEXP unconstrainedSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type r_theta(r_thetaSEXP);
    Rcpp::traits::input_parameter< bool >::type unconstrained(unconstrainedSEXP);
    __result = Rcpp::wrap(GetParametersFromGlobalVector(r_vp, r_theta, unconstrained));
    return __result;
END_RCPP
}
// GetGlobalVectorFromParameters
Eigen::VectorXd GetGlobalVectorFromParameters(const Rcpp::List r_vp, bool unconstrained);
RcppExport SEXP MicrocreditLRVB_GetGlobalVectorFromParameters(SEXP r_vpSEXP, SEXP unconstrainedSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    Rcpp::traits::input_parameter< bool >::type unconstrained(unconstrainedSEXP);
    __result = Rcpp::wrap(GetGlobalVectorFromParameters(r_vp, unconstrained));
    return __result;
END_RCPP
}
// GetMomentsFromVector
Rcpp::List GetMomentsFromVector(const Rcpp::List r_mp, const Eigen::Map<Eigen::VectorXd> r_theta);
RcppExport SEXP MicrocreditLRVB_GetMomentsFromVector(SEXP r_mpSEXP, SEXP r_thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_mp(r_mpSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type r_theta(r_thetaSEXP);
    __result = Rcpp::wrap(GetMomentsFromVector(r_mp, r_theta));
    return __result;
END_RCPP
}
// ToAndFromParameters
Rcpp::List ToAndFromParameters(const Rcpp::List r_vp);
RcppExport SEXP MicrocreditLRVB_ToAndFromParameters(SEXP r_vpSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    __result = Rcpp::wrap(ToAndFromParameters(r_vp));
    return __result;
END_RCPP
}
// GetElboDerivatives
Rcpp::List GetElboDerivatives(const Eigen::Map<Eigen::MatrixXd> r_x, const Eigen::Map<Eigen::VectorXd> r_y, const Eigen::Map<Eigen::VectorXi> r_y_g, const Rcpp::List r_vp, const Rcpp::List r_pp, const bool calculate_gradient, const bool calculate_hessian, const bool unconstrained);
RcppExport SEXP MicrocreditLRVB_GetElboDerivatives(SEXP r_xSEXP, SEXP r_ySEXP, SEXP r_y_gSEXP, SEXP r_vpSEXP, SEXP r_ppSEXP, SEXP calculate_gradientSEXP, SEXP calculate_hessianSEXP, SEXP unconstrainedSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd> >::type r_x(r_xSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type r_y(r_ySEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXi> >::type r_y_g(r_y_gSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_pp(r_ppSEXP);
    Rcpp::traits::input_parameter< const bool >::type calculate_gradient(calculate_gradientSEXP);
    Rcpp::traits::input_parameter< const bool >::type calculate_hessian(calculate_hessianSEXP);
    Rcpp::traits::input_parameter< const bool >::type unconstrained(unconstrainedSEXP);
    __result = Rcpp::wrap(GetElboDerivatives(r_x, r_y, r_y_g, r_vp, r_pp, calculate_gradient, calculate_hessian, unconstrained));
    return __result;
END_RCPP
}
// GetCustomElboDerivatives
Rcpp::List GetCustomElboDerivatives(const Eigen::Map<Eigen::MatrixXd> r_x, const Eigen::Map<Eigen::VectorXd> r_y, const Eigen::Map<Eigen::VectorXi> r_y_g, const Rcpp::List r_vp, const Rcpp::List r_pp, bool include_obs, bool include_hier, bool include_prior, bool include_entropy, bool use_group, int g, const bool calculate_gradient, const bool calculate_hessian, const bool unconstrained);
RcppExport SEXP MicrocreditLRVB_GetCustomElboDerivatives(SEXP r_xSEXP, SEXP r_ySEXP, SEXP r_y_gSEXP, SEXP r_vpSEXP, SEXP r_ppSEXP, SEXP include_obsSEXP, SEXP include_hierSEXP, SEXP include_priorSEXP, SEXP include_entropySEXP, SEXP use_groupSEXP, SEXP gSEXP, SEXP calculate_gradientSEXP, SEXP calculate_hessianSEXP, SEXP unconstrainedSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd> >::type r_x(r_xSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type r_y(r_ySEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXi> >::type r_y_g(r_y_gSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_pp(r_ppSEXP);
    Rcpp::traits::input_parameter< bool >::type include_obs(include_obsSEXP);
    Rcpp::traits::input_parameter< bool >::type include_hier(include_hierSEXP);
    Rcpp::traits::input_parameter< bool >::type include_prior(include_priorSEXP);
    Rcpp::traits::input_parameter< bool >::type include_entropy(include_entropySEXP);
    Rcpp::traits::input_parameter< bool >::type use_group(use_groupSEXP);
    Rcpp::traits::input_parameter< int >::type g(gSEXP);
    Rcpp::traits::input_parameter< const bool >::type calculate_gradient(calculate_gradientSEXP);
    Rcpp::traits::input_parameter< const bool >::type calculate_hessian(calculate_hessianSEXP);
    Rcpp::traits::input_parameter< const bool >::type unconstrained(unconstrainedSEXP);
    __result = Rcpp::wrap(GetCustomElboDerivatives(r_x, r_y, r_y_g, r_vp, r_pp, include_obs, include_hier, include_prior, include_entropy, use_group, g, calculate_gradient, calculate_hessian, unconstrained));
    return __result;
END_RCPP
}
// GetMoments
Rcpp::List GetMoments(const Rcpp::List r_vp);
RcppExport SEXP MicrocreditLRVB_GetMoments(SEXP r_vpSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    __result = Rcpp::wrap(GetMoments(r_vp));
    return __result;
END_RCPP
}
// GetMomentJacobian
Rcpp::List GetMomentJacobian(const Rcpp::List r_vp);
RcppExport SEXP MicrocreditLRVB_GetMomentJacobian(SEXP r_vpSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    __result = Rcpp::wrap(GetMomentJacobian(r_vp));
    return __result;
END_RCPP
}
// GetCovariance
Eigen::SparseMatrix<double> GetCovariance(const Rcpp::List r_vp);
RcppExport SEXP MicrocreditLRVB_GetCovariance(SEXP r_vpSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    __result = Rcpp::wrap(GetCovariance(r_vp));
    return __result;
END_RCPP
}
// GetVariationalCovariance
Eigen::SparseMatrix<double> GetVariationalCovariance(const Rcpp::List r_vp);
RcppExport SEXP MicrocreditLRVB_GetVariationalCovariance(SEXP r_vpSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    __result = Rcpp::wrap(GetVariationalCovariance(r_vp));
    return __result;
END_RCPP
}
