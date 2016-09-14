// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// GetEmptyVariationalParameters
Rcpp::List GetEmptyVariationalParameters(int k, int n_g);
RcppExport SEXP MicrocreditLRVB_GetEmptyVariationalParameters(SEXP kSEXP, SEXP n_gSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< int >::type n_g(n_gSEXP);
    rcpp_result_gen = Rcpp::wrap(GetEmptyVariationalParameters(k, n_g));
    return rcpp_result_gen;
END_RCPP
}
// GetEmptyPriors
Rcpp::List GetEmptyPriors(int k);
RcppExport SEXP MicrocreditLRVB_GetEmptyPriors(SEXP kSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    rcpp_result_gen = Rcpp::wrap(GetEmptyPriors(k));
    return rcpp_result_gen;
END_RCPP
}
// GetParametersFromVector
Rcpp::List GetParametersFromVector(const Rcpp::List r_vp, const Eigen::Map<Eigen::VectorXd> r_theta, bool unconstrained);
RcppExport SEXP MicrocreditLRVB_GetParametersFromVector(SEXP r_vpSEXP, SEXP r_thetaSEXP, SEXP unconstrainedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type r_theta(r_thetaSEXP);
    Rcpp::traits::input_parameter< bool >::type unconstrained(unconstrainedSEXP);
    rcpp_result_gen = Rcpp::wrap(GetParametersFromVector(r_vp, r_theta, unconstrained));
    return rcpp_result_gen;
END_RCPP
}
// GetVectorFromParameters
Eigen::VectorXd GetVectorFromParameters(const Rcpp::List r_vp, bool unconstrained);
RcppExport SEXP MicrocreditLRVB_GetVectorFromParameters(SEXP r_vpSEXP, SEXP unconstrainedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    Rcpp::traits::input_parameter< bool >::type unconstrained(unconstrainedSEXP);
    rcpp_result_gen = Rcpp::wrap(GetVectorFromParameters(r_vp, unconstrained));
    return rcpp_result_gen;
END_RCPP
}
// GetParametersFromGlobalVector
Rcpp::List GetParametersFromGlobalVector(const Rcpp::List r_vp, const Eigen::Map<Eigen::VectorXd> r_theta, bool unconstrained);
RcppExport SEXP MicrocreditLRVB_GetParametersFromGlobalVector(SEXP r_vpSEXP, SEXP r_thetaSEXP, SEXP unconstrainedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type r_theta(r_thetaSEXP);
    Rcpp::traits::input_parameter< bool >::type unconstrained(unconstrainedSEXP);
    rcpp_result_gen = Rcpp::wrap(GetParametersFromGlobalVector(r_vp, r_theta, unconstrained));
    return rcpp_result_gen;
END_RCPP
}
// GetGlobalVectorFromParameters
Eigen::VectorXd GetGlobalVectorFromParameters(const Rcpp::List r_vp, bool unconstrained);
RcppExport SEXP MicrocreditLRVB_GetGlobalVectorFromParameters(SEXP r_vpSEXP, SEXP unconstrainedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    Rcpp::traits::input_parameter< bool >::type unconstrained(unconstrainedSEXP);
    rcpp_result_gen = Rcpp::wrap(GetGlobalVectorFromParameters(r_vp, unconstrained));
    return rcpp_result_gen;
END_RCPP
}
// GetMomentsFromVector
Rcpp::List GetMomentsFromVector(const Rcpp::List r_mp, const Eigen::Map<Eigen::VectorXd> r_theta);
RcppExport SEXP MicrocreditLRVB_GetMomentsFromVector(SEXP r_mpSEXP, SEXP r_thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_mp(r_mpSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type r_theta(r_thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(GetMomentsFromVector(r_mp, r_theta));
    return rcpp_result_gen;
END_RCPP
}
// GetVectorFromMoments
Eigen::VectorXd GetVectorFromMoments(const Rcpp::List r_mp);
RcppExport SEXP MicrocreditLRVB_GetVectorFromMoments(SEXP r_mpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_mp(r_mpSEXP);
    rcpp_result_gen = Rcpp::wrap(GetVectorFromMoments(r_mp));
    return rcpp_result_gen;
END_RCPP
}
// GetPriorsFromVector
Rcpp::List GetPriorsFromVector(const Rcpp::List r_pp, const Eigen::Map<Eigen::VectorXd> r_theta);
RcppExport SEXP MicrocreditLRVB_GetPriorsFromVector(SEXP r_ppSEXP, SEXP r_thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_pp(r_ppSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type r_theta(r_thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(GetPriorsFromVector(r_pp, r_theta));
    return rcpp_result_gen;
END_RCPP
}
// GetVectorFromPriors
Eigen::VectorXd GetVectorFromPriors(const Rcpp::List r_pp);
RcppExport SEXP MicrocreditLRVB_GetVectorFromPriors(SEXP r_ppSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_pp(r_ppSEXP);
    rcpp_result_gen = Rcpp::wrap(GetVectorFromPriors(r_pp));
    return rcpp_result_gen;
END_RCPP
}
// GetPriorsAndParametersFromVector
Rcpp::List GetPriorsAndParametersFromVector(const Rcpp::List r_vp, const Rcpp::List r_pp, const Eigen::Map<Eigen::VectorXd> r_theta);
RcppExport SEXP MicrocreditLRVB_GetPriorsAndParametersFromVector(SEXP r_vpSEXP, SEXP r_ppSEXP, SEXP r_thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_pp(r_ppSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type r_theta(r_thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(GetPriorsAndParametersFromVector(r_vp, r_pp, r_theta));
    return rcpp_result_gen;
END_RCPP
}
// GetElboDerivatives
Rcpp::List GetElboDerivatives(const Eigen::Map<Eigen::MatrixXd> r_x, const Eigen::Map<Eigen::VectorXd> r_y, const Eigen::Map<Eigen::VectorXi> r_y_g, const Rcpp::List r_vp, const Rcpp::List r_pp, const bool calculate_gradient, const bool calculate_hessian, const bool unconstrained);
RcppExport SEXP MicrocreditLRVB_GetElboDerivatives(SEXP r_xSEXP, SEXP r_ySEXP, SEXP r_y_gSEXP, SEXP r_vpSEXP, SEXP r_ppSEXP, SEXP calculate_gradientSEXP, SEXP calculate_hessianSEXP, SEXP unconstrainedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd> >::type r_x(r_xSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type r_y(r_ySEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXi> >::type r_y_g(r_y_gSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_pp(r_ppSEXP);
    Rcpp::traits::input_parameter< const bool >::type calculate_gradient(calculate_gradientSEXP);
    Rcpp::traits::input_parameter< const bool >::type calculate_hessian(calculate_hessianSEXP);
    Rcpp::traits::input_parameter< const bool >::type unconstrained(unconstrainedSEXP);
    rcpp_result_gen = Rcpp::wrap(GetElboDerivatives(r_x, r_y, r_y_g, r_vp, r_pp, calculate_gradient, calculate_hessian, unconstrained));
    return rcpp_result_gen;
END_RCPP
}
// GetCustomElboDerivatives
Rcpp::List GetCustomElboDerivatives(const Eigen::Map<Eigen::MatrixXd> r_x, const Eigen::Map<Eigen::VectorXd> r_y, const Eigen::Map<Eigen::VectorXi> r_y_g, const Rcpp::List r_vp, const Rcpp::List r_pp, bool include_obs, bool include_hier, bool include_prior, bool include_entropy, bool global_only, const bool calculate_gradient, const bool calculate_hessian, const bool unconstrained);
RcppExport SEXP MicrocreditLRVB_GetCustomElboDerivatives(SEXP r_xSEXP, SEXP r_ySEXP, SEXP r_y_gSEXP, SEXP r_vpSEXP, SEXP r_ppSEXP, SEXP include_obsSEXP, SEXP include_hierSEXP, SEXP include_priorSEXP, SEXP include_entropySEXP, SEXP global_onlySEXP, SEXP calculate_gradientSEXP, SEXP calculate_hessianSEXP, SEXP unconstrainedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd> >::type r_x(r_xSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type r_y(r_ySEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXi> >::type r_y_g(r_y_gSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_pp(r_ppSEXP);
    Rcpp::traits::input_parameter< bool >::type include_obs(include_obsSEXP);
    Rcpp::traits::input_parameter< bool >::type include_hier(include_hierSEXP);
    Rcpp::traits::input_parameter< bool >::type include_prior(include_priorSEXP);
    Rcpp::traits::input_parameter< bool >::type include_entropy(include_entropySEXP);
    Rcpp::traits::input_parameter< bool >::type global_only(global_onlySEXP);
    Rcpp::traits::input_parameter< const bool >::type calculate_gradient(calculate_gradientSEXP);
    Rcpp::traits::input_parameter< const bool >::type calculate_hessian(calculate_hessianSEXP);
    Rcpp::traits::input_parameter< const bool >::type unconstrained(unconstrainedSEXP);
    rcpp_result_gen = Rcpp::wrap(GetCustomElboDerivatives(r_x, r_y, r_y_g, r_vp, r_pp, include_obs, include_hier, include_prior, include_entropy, global_only, calculate_gradient, calculate_hessian, unconstrained));
    return rcpp_result_gen;
END_RCPP
}
// GetMoments
Rcpp::List GetMoments(const Rcpp::List r_vp);
RcppExport SEXP MicrocreditLRVB_GetMoments(SEXP r_vpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    rcpp_result_gen = Rcpp::wrap(GetMoments(r_vp));
    return rcpp_result_gen;
END_RCPP
}
// GetMomentJacobian
Rcpp::List GetMomentJacobian(const Rcpp::List r_vp, bool unconstrained);
RcppExport SEXP MicrocreditLRVB_GetMomentJacobian(SEXP r_vpSEXP, SEXP unconstrainedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    Rcpp::traits::input_parameter< bool >::type unconstrained(unconstrainedSEXP);
    rcpp_result_gen = Rcpp::wrap(GetMomentJacobian(r_vp, unconstrained));
    return rcpp_result_gen;
END_RCPP
}
// GetSparseELBOHessian
Eigen::SparseMatrix<double> GetSparseELBOHessian(const Eigen::Map<Eigen::MatrixXd> r_x, const Eigen::Map<Eigen::VectorXd> r_y, const Eigen::Map<Eigen::VectorXi> r_y_g, const Rcpp::List r_vp, const Rcpp::List r_pp, bool unconstrained);
RcppExport SEXP MicrocreditLRVB_GetSparseELBOHessian(SEXP r_xSEXP, SEXP r_ySEXP, SEXP r_y_gSEXP, SEXP r_vpSEXP, SEXP r_ppSEXP, SEXP unconstrainedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd> >::type r_x(r_xSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type r_y(r_ySEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXi> >::type r_y_g(r_y_gSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_pp(r_ppSEXP);
    Rcpp::traits::input_parameter< bool >::type unconstrained(unconstrainedSEXP);
    rcpp_result_gen = Rcpp::wrap(GetSparseELBOHessian(r_x, r_y, r_y_g, r_vp, r_pp, unconstrained));
    return rcpp_result_gen;
END_RCPP
}
// GetLogPriorDerivatives
Rcpp::List GetLogPriorDerivatives(const Rcpp::List r_vp, const Rcpp::List r_pp, const bool calculate_gradient, const bool calculate_hessian, const bool unconstrained);
RcppExport SEXP MicrocreditLRVB_GetLogPriorDerivatives(SEXP r_vpSEXP, SEXP r_ppSEXP, SEXP calculate_gradientSEXP, SEXP calculate_hessianSEXP, SEXP unconstrainedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_pp(r_ppSEXP);
    Rcpp::traits::input_parameter< const bool >::type calculate_gradient(calculate_gradientSEXP);
    Rcpp::traits::input_parameter< const bool >::type calculate_hessian(calculate_hessianSEXP);
    Rcpp::traits::input_parameter< const bool >::type unconstrained(unconstrainedSEXP);
    rcpp_result_gen = Rcpp::wrap(GetLogPriorDerivatives(r_vp, r_pp, calculate_gradient, calculate_hessian, unconstrained));
    return rcpp_result_gen;
END_RCPP
}
// GetLogVariationalDensityDerivatives
Rcpp::List GetLogVariationalDensityDerivatives(const Rcpp::List r_obs_mp, const Rcpp::List r_vp, const Rcpp::List r_pp, bool const include_mu, bool const include_lambda, const Eigen::Map<Eigen::VectorXi> r_include_mu_groups, const Eigen::Map<Eigen::VectorXi> r_include_tau_groups, bool const calculate_gradient);
RcppExport SEXP MicrocreditLRVB_GetLogVariationalDensityDerivatives(SEXP r_obs_mpSEXP, SEXP r_vpSEXP, SEXP r_ppSEXP, SEXP include_muSEXP, SEXP include_lambdaSEXP, SEXP r_include_mu_groupsSEXP, SEXP r_include_tau_groupsSEXP, SEXP calculate_gradientSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_obs_mp(r_obs_mpSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_pp(r_ppSEXP);
    Rcpp::traits::input_parameter< bool const >::type include_mu(include_muSEXP);
    Rcpp::traits::input_parameter< bool const >::type include_lambda(include_lambdaSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXi> >::type r_include_mu_groups(r_include_mu_groupsSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXi> >::type r_include_tau_groups(r_include_tau_groupsSEXP);
    Rcpp::traits::input_parameter< bool const >::type calculate_gradient(calculate_gradientSEXP);
    rcpp_result_gen = Rcpp::wrap(GetLogVariationalDensityDerivatives(r_obs_mp, r_vp, r_pp, include_mu, include_lambda, r_include_mu_groups, r_include_tau_groups, calculate_gradient));
    return rcpp_result_gen;
END_RCPP
}
// GetMCMCLogPriorDerivatives
Rcpp::List GetMCMCLogPriorDerivatives(const Rcpp::List draw_list, const Rcpp::List r_pp);
RcppExport SEXP MicrocreditLRVB_GetMCMCLogPriorDerivatives(SEXP draw_listSEXP, SEXP r_ppSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List >::type draw_list(draw_listSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_pp(r_ppSEXP);
    rcpp_result_gen = Rcpp::wrap(GetMCMCLogPriorDerivatives(draw_list, r_pp));
    return rcpp_result_gen;
END_RCPP
}
// GetObsLogPriorDerivatives
Rcpp::List GetObsLogPriorDerivatives(const Rcpp::List r_obs_mp, const Rcpp::List r_pp, bool include_mu, bool include_lambda, bool include_tau);
RcppExport SEXP MicrocreditLRVB_GetObsLogPriorDerivatives(SEXP r_obs_mpSEXP, SEXP r_ppSEXP, SEXP include_muSEXP, SEXP include_lambdaSEXP, SEXP include_tauSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_obs_mp(r_obs_mpSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_pp(r_ppSEXP);
    Rcpp::traits::input_parameter< bool >::type include_mu(include_muSEXP);
    Rcpp::traits::input_parameter< bool >::type include_lambda(include_lambdaSEXP);
    Rcpp::traits::input_parameter< bool >::type include_tau(include_tauSEXP);
    rcpp_result_gen = Rcpp::wrap(GetObsLogPriorDerivatives(r_obs_mp, r_pp, include_mu, include_lambda, include_tau));
    return rcpp_result_gen;
END_RCPP
}
// GetCovariance
Eigen::SparseMatrix<double> GetCovariance(const Rcpp::List r_vp);
RcppExport SEXP MicrocreditLRVB_GetCovariance(SEXP r_vpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List >::type r_vp(r_vpSEXP);
    rcpp_result_gen = Rcpp::wrap(GetCovariance(r_vp));
    return rcpp_result_gen;
END_RCPP
}
// EvaluateLKJPriorVB
double EvaluateLKJPriorVB(const Eigen::Map<Eigen::MatrixXd> r_v, double n, double alpha, double beta, double eta);
RcppExport SEXP MicrocreditLRVB_EvaluateLKJPriorVB(SEXP r_vSEXP, SEXP nSEXP, SEXP alphaSEXP, SEXP betaSEXP, SEXP etaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd> >::type r_v(r_vSEXP);
    Rcpp::traits::input_parameter< double >::type n(nSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type eta(etaSEXP);
    rcpp_result_gen = Rcpp::wrap(EvaluateLKJPriorVB(r_v, n, alpha, beta, eta));
    return rcpp_result_gen;
END_RCPP
}
// EvaluateLKJPriorDraw
double EvaluateLKJPriorDraw(const Eigen::Map<Eigen::MatrixXd> r_sigma, double log_det_sigma, double alpha, double beta, double eta);
RcppExport SEXP MicrocreditLRVB_EvaluateLKJPriorDraw(SEXP r_sigmaSEXP, SEXP log_det_sigmaSEXP, SEXP alphaSEXP, SEXP betaSEXP, SEXP etaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd> >::type r_sigma(r_sigmaSEXP);
    Rcpp::traits::input_parameter< double >::type log_det_sigma(log_det_sigmaSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type eta(etaSEXP);
    rcpp_result_gen = Rcpp::wrap(EvaluateLKJPriorDraw(r_sigma, log_det_sigma, alpha, beta, eta));
    return rcpp_result_gen;
END_RCPP
}
