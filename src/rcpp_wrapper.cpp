# include <Rcpp.h>
# include <RcppEigen.h>

# include <vector>
# include <ctime>

# include <stan/math.hpp>
# include <stan/math/mix/mat/functor/hessian.hpp>

// # include "variational_parameters.h"
// # include "exponential_families.h"
# include "microcredit_model.h"

# include "differentiate_jacobian.h"
# include "transform_hessian.h"

typedef Eigen::Triplet<double> Triplet; // For populating sparse matrices

// Export get_ud_index for the construction of covariance matrices.
// TODO: use namespaces to avoid renaming this.
// [[Rcpp::export]]
int get_upper_diagonal_index(int i, int j) {
  return get_ud_index(i, j);
}


// [[Rcpp::export]]
double r_mulitvariate_digamma(double x, int p) {
  return multivariate_digamma(x, p);
}


// For testing
// [[Rcpp::export]]
double GetWishartEntropy(const Eigen::Map<Eigen::MatrixXd> v_par_r, const double n_par_r) {
  Eigen::MatrixXd v_par = v_par_r;
  return GetWishartEntropy(v_par, n_par_r);
}


// For testing
// [[Rcpp::export]]
double r_GetELogDetWishart(const Eigen::Map<Eigen::MatrixXd> v_par_r, const double n_par_r) {
  Eigen::MatrixXd v_par = v_par_r;
  return GetELogDetWishart(v_par, n_par_r);
}


// Make an R list out of encoder offsets.
void convert_to_list(
    VariationalParameterEncoder const &vp_encoder, Rcpp::List &r_list, int offset=0) {

  r_list["dim"] = vp_encoder.dim;

  r_list["e_mu"] = vp_encoder.e_mu_offset + offset;
  r_list["e_mu2"] = vp_encoder.e_mu2_offset + offset;

  r_list["lambda_v_par"] = vp_encoder.lambda_v_par_offset + offset;
  r_list["lambda_n_par"] = vp_encoder.lambda_n_par_offset + offset;

  // TODO: size checking
  // I'm not sure why, but this initializiation seems to make a list
  // of length one longer than the argument.
  int n_g = vp_encoder.e_mu_g_offset.size();
  Rcpp::List e_mu_g_offset_list(n_g);
  Rcpp::List e_mu2_g_offset_list(n_g);
  Rcpp::List e_tau_offset_list(n_g);
  Rcpp::List e_log_tau_offset_list(n_g);
  for (int g = 0; g < n_g; g++) {
    e_tau_offset_list[g] = vp_encoder.e_tau_offset[g] + offset;
    e_log_tau_offset_list[g] = vp_encoder.e_log_tau_offset[g] + offset;
    e_mu_g_offset_list[g] = vp_encoder.e_mu_g_offset[g] + offset;
    e_mu2_g_offset_list[g] = vp_encoder.e_mu2_g_offset[g] + offset;
  }

  r_list["e_tau_vec"] = e_tau_offset_list;
  r_list["e_log_tau_vec"] = e_log_tau_offset_list;
  r_list["e_mu_g_vec"] = e_mu_g_offset_list;
  r_list["e_mu2_g_vec"] = e_mu2_g_offset_list;
}


Rcpp::List convert_to_list(VariationalParameterEncoder const &vp_encoder, int offset=0) {
  Rcpp::List r_list;
  convert_to_list(vp_encoder, r_list, offset);
  return r_list;
}


// Make an R list out of variational encoder offsets.
void convert_to_list(
    PriorParameterEncoder const &pp_encoder, Rcpp::List &r_list, int const offset=0) {

  r_list["dim"] = pp_encoder.dim;

  r_list["mu_mean_offset"] = pp_encoder.mu_mean_offset + offset;
  r_list["mu_info_offset"] = pp_encoder.mu_info_offset + offset;
  r_list["lambda_eta_offset"] = pp_encoder.lambda_eta_offset + offset;
  r_list["lambda_alpha_offset"] = pp_encoder.lambda_alpha_offset + offset;
  r_list["lambda_beta_offset"] = pp_encoder.lambda_beta_offset + offset;
  r_list["lambda_eta_offset"] = pp_encoder.lambda_eta_offset + offset;
  r_list["tau_alpha_offset"] = pp_encoder.tau_alpha_offset + offset;
  r_list["tau_beta_offset"] = pp_encoder.tau_beta_offset + offset;
}


Rcpp::List convert_to_list(
    PriorParameterEncoder const &pp_encoder, int const offset=0) {

  Rcpp::List r_list;
  convert_to_list(pp_encoder, r_list, offset);
  return r_list;
}


// Make an R list out of model encoder offsets.
Rcpp::List convert_to_list(ModelParameterEncoder const &encoder) {
  Rcpp::List r_list;
  convert_to_list(encoder.variational_encoder, r_list, encoder.variational_offset);
  convert_to_list(encoder.prior_encoder, r_list, encoder.prior_offset);
  return r_list;
}


// Make an R list out of variational parameters.
template <typename T> Rcpp::List convert_to_list(VariationalParameters<T> vp) {
  Rcpp::List r_list;

  r_list["k"] = vp.k;
  r_list["n_g"] = vp.n_g;

  r_list["e_mu"] = vp.mu.e_vec;
  r_list["e_mu2"] = vp.mu.e_outer.mat;

  r_list["lambda_v_par"] = vp.lambda.v.mat;
  r_list["lambda_n_par"] = vp.lambda.n;

  // TODO: size checking
  // I'm not sure why, but this initializiation seems to make a list
  // of length one longer than the argument.
  Rcpp::List e_mu_g_vec_list(vp.n_g);
  Rcpp::List e_mu2_g_vec_list(vp.n_g);
  Rcpp::List e_tau_vec_list(vp.n_g);
  Rcpp::List e_log_tau_vec_list(vp.n_g);
  for (int g = 0; g < vp.n_g; g++) {
    e_tau_vec_list[g] = vp.tau_vec[g].e;
    e_log_tau_vec_list[g] = vp.tau_vec[g].e_log;
    e_mu_g_vec_list[g] = vp.mu_g_vec[g].e_vec;
    e_mu2_g_vec_list[g] = vp.mu_g_vec[g].e_outer.mat;
  }

  r_list["e_tau_vec"] = e_tau_vec_list;
  r_list["e_log_tau_vec"] = e_log_tau_vec_list;
  r_list["e_mu_g_vec"] = e_mu_g_vec_list;
  r_list["e_mu2_g_vec"] = e_mu2_g_vec_list;

  return r_list;
}


// Update vp in place with the values from r_list.
template <typename T>
void convert_from_list(Rcpp::List const &r_list, VariationalParameters<T> &vp) {

  vp.mu.e_vec = Rcpp::as<Eigen::VectorXd>(r_list["e_mu"]);
  vp.mu.e_outer.mat = Rcpp::as<Eigen::MatrixXd>(r_list["e_mu2"]);

  vp.lambda.v.mat = Rcpp::as<Eigen::MatrixXd>(r_list["lambda_v_par"]);
  vp.lambda.n = Rcpp::as<double>(r_list["lambda_n_par"]);

  Rcpp::List e_mu_g_vec_list = r_list["e_mu_g_vec"];
  Rcpp::List e_mu2_g_vec_list = r_list["e_mu2_g_vec"];
  Rcpp::List e_tau_vec_list = r_list["e_tau_vec"];
  Rcpp::List e_log_tau_vec_list = r_list["e_log_tau_vec"];
  int n_g = e_mu_g_vec_list.size();
  if (e_mu2_g_vec_list.size() != n_g) {
    std::ostringstream error_msg;
    error_msg <<
        "e_mu2_g_vec_list does not have n_g elements.  " <<
        "List size is " << e_mu2_g_vec_list.size() << ".  " <<
        "n_g = " << n_g << "\n";
    throw std::runtime_error(error_msg.str());
  }
  if (vp.n_g != n_g) {
      std::ostringstream error_msg;
      error_msg <<
          "vp.n_g does not match n_g (which is from e_mu_g_vec_list.size()).  " <<
          "vp.n_g = " << vp.n_g << ".  " <<
          "n_g = " << n_g << "\n";
    throw std::runtime_error(error_msg.str());
  }

  for (int g = 0; g < n_g; g++) {
    vp.tau_vec[g].e = Rcpp::as<double>(e_tau_vec_list[g]);
    vp.tau_vec[g].e_log = Rcpp::as<double>(e_log_tau_vec_list[g]);
    vp.mu_g_vec[g].e_vec = Rcpp::as<Eigen::VectorXd>(e_mu_g_vec_list[g]);
    vp.mu_g_vec[g].e_outer.mat = Rcpp::as<Eigen::MatrixXd>(e_mu2_g_vec_list[g]);
  }
};


// Update pp in place with the values from r_list.
template <typename T>
void convert_from_list(Rcpp::List r_list, PriorParameters<T> &pp) {

  pp.mu_mean = Rcpp::as<Eigen::VectorXd>(r_list["mu_mean"]);
  pp.mu_info.mat = Rcpp::as<Eigen::MatrixXd>(r_list["mu_info"]);

  pp.lambda_eta = Rcpp::as<double>(r_list["lambda_eta"]);
  pp.lambda_alpha = Rcpp::as<double>(r_list["lambda_alpha"]);
  pp.lambda_beta = Rcpp::as<double>(r_list["lambda_beta"]);

  pp.tau_alpha = Rcpp::as<double>(r_list["tau_alpha"]);
  pp.tau_beta = Rcpp::as<double>(r_list["tau_beta"]);

  pp.lambda_diag_min = Rcpp::as<double>(r_list["lambda_diag_min"]);
  pp.lambda_n_min = Rcpp::as<double>(r_list["lambda_n_min"]);
};


// r_y_g should be one-indexed group indicators.
// [[Rcpp::export]]
Rcpp::List ModelGradient(
    const Eigen::Map<Eigen::MatrixXd> r_x,
    const Eigen::Map<Eigen::VectorXd> r_y,
    const Eigen::Map<Eigen::VectorXi> r_y_g,
    const Rcpp::List r_vp, const Rcpp::List r_pp,
    const bool calculate_hessian, const bool unconstrained) {

  Eigen::MatrixXd x = r_x;
  Eigen::VectorXd y = r_y;
  Eigen::VectorXi y_g = r_y_g;
  MicroCreditData data(x, y, y_g);
  VariationalParameters<double> base_vp(data.k, data.n_g);
  PriorParameters<double> pp(data.k);
  convert_from_list(r_vp, base_vp);
  convert_from_list(r_pp, pp);
  VariationalParameterEncoder vp_encoder(
    base_vp, pp.lambda_diag_min, pp.lambda_n_min, unconstrained);

  Eigen::VectorXd theta = vp_encoder.get_parameter_vector(base_vp);
  MicroCreditLogLikelihood LogLik(data, base_vp, pp, vp_encoder);

  double log_lik;
  Eigen::VectorXd dll_dtheta(vp_encoder.dim);
  Eigen::MatrixXd dll2_dtheta2(vp_encoder.dim, vp_encoder.dim);

  stan::math::set_zero_all_adjoints();
  if (calculate_hessian) {
    stan::math::hessian(LogLik, theta, log_lik, dll_dtheta, dll2_dtheta2);
  } else {
    stan::math::gradient(LogLik, theta, log_lik, dll_dtheta);
  }

  VariationalParameters<double> grad_vp(base_vp.k, base_vp.n_g);
  vp_encoder.set_parameters_from_vector(dll_dtheta, grad_vp);
  Rcpp::List ret = convert_to_list(grad_vp);
  ret["log_lik"] = log_lik;
  ret["theta"] = theta;
  ret["obs_grad"] = dll_dtheta;
  if (calculate_hessian) {
    ret["obs_hess"] = dll2_dtheta2;
  }

  return ret;
}


// [[Rcpp::export]]
Eigen::VectorXd EncodeLambda(const Rcpp::List r_vp, int k, int n_g,
                      double lambda_diag_min, double n_min) {
  // n_g is not used.
  VariationalParameters<double> base_vp(k, n_g);
  convert_from_list(r_vp, base_vp);
  WishartParameterEncoder lambda_encoder(base_vp, lambda_diag_min, n_min, true);
  Eigen::VectorXd theta = lambda_encoder.get_parameter_vector(base_vp);
  return theta;
}


// [[Rcpp::export]]
Rcpp::List DecodeLambda(const Eigen::VectorXd theta, int k, int n_g,
                        double lambda_diag_min, double n_min) {
  // n_g is not used.
  VariationalParameters<double> base_vp(k, n_g);
  WishartParameterEncoder lambda_encoder(base_vp, lambda_diag_min, n_min, true);
  lambda_encoder.set_parameters_from_vector(theta, base_vp);
  Rcpp::List return_list = convert_to_list(base_vp);
  return return_list;
}


// Convert a parameter vector to a list using an encoder.
// [[Rcpp::export]]
Rcpp::List DecodeParameters(
    const Eigen::VectorXd theta, const Rcpp::List r_vp, const Rcpp::List r_pp,
    const bool unconstrained_wishart) {

  int k = r_vp["k"];
  int n_g = r_vp["n_g"];
  VariationalParameters<double> vp(k, n_g);
  PriorParameters<double> pp(k);
  convert_from_list(r_pp, pp);
  VariationalParameterEncoder
    vp_encoder(vp, pp.lambda_diag_min, pp.lambda_n_min, unconstrained_wishart);
  vp_encoder.set_parameters_from_vector(theta, vp);
  Rcpp::List return_list = convert_to_list(vp);
  return return_list;
}


// [[Rcpp::export]]
Rcpp::List LambdaGradient(
    const Eigen::Map<Eigen::MatrixXd> x, const Eigen::Map<Eigen::VectorXd> y,
    const Eigen::Map<Eigen::VectorXi> y_g,
    const Rcpp::List r_vp, const Rcpp::List r_pp, const bool unconstrained) {

  MicroCreditData data(x, y, y_g);
  VariationalParameters<double> base_vp(data.k, data.n_g);
  PriorParameters<double> pp(data.k);

  convert_from_list(r_vp, base_vp);
  convert_from_list(r_pp, pp);

  WishartParameterEncoder lambda_encoder(base_vp, pp.lambda_diag_min,
                                         pp.lambda_n_min, unconstrained);

  Rcpp::List ret;
  double elbo;
  Eigen::VectorXd de_dtheta(lambda_encoder.dim);
  Eigen::MatrixXd de2_dtheta2(lambda_encoder.dim, lambda_encoder.dim);
  Eigen::VectorXd theta = lambda_encoder.get_parameter_vector(base_vp);
  MicroCreditWishartElbo LambdaElbo(data, base_vp, pp, lambda_encoder);
  stan::math::set_zero_all_adjoints();
  stan::math::hessian(LambdaElbo, theta, elbo, de_dtheta, de2_dtheta2);

  ret["lambda_hess"] = de2_dtheta2;
  ret["lambda_grad"] = de_dtheta;
  ret["elbo"] = elbo;

  return ret;
}


// [[Rcpp::export]]
Rcpp::List LambdaEntropyDerivs(
    const Eigen::Map<Eigen::MatrixXd> x, const Eigen::Map<Eigen::VectorXd> y,
    const Eigen::Map<Eigen::VectorXi> y_g,
    const Rcpp::List r_vp, const Rcpp::List r_pp, const bool unconstrained) {

  MicroCreditData data(x, y, y_g);
  VariationalParameters<double> base_vp(data.k, data.n_g);
  PriorParameters<double> pp(data.k);

  convert_from_list(r_vp, base_vp);
  convert_from_list(r_pp, pp);

  WishartParameterEncoder lambda_encoder(base_vp, pp.lambda_diag_min,
                                         pp.lambda_n_min, unconstrained);

  Rcpp::List ret;
  double entropy;
  Eigen::VectorXd de_dtheta(lambda_encoder.dim);
  Eigen::MatrixXd de2_dtheta2(lambda_encoder.dim, lambda_encoder.dim);
  Eigen::VectorXd theta = lambda_encoder.get_parameter_vector(base_vp);
  MicroCreditWishartEntropy LambdaEntropy(data, base_vp, pp, lambda_encoder);
  stan::math::set_zero_all_adjoints();
  stan::math::hessian(LambdaEntropy, theta, entropy, de_dtheta, de2_dtheta2);

  ret["lambda_hess"] = de2_dtheta2;
  ret["lambda_grad"] = de_dtheta;
  ret["entropy"] = entropy;

  return ret;
}


// [[Rcpp::export]]
Rcpp::List LambdaLikelihoodMomentDerivs(
    const Eigen::Map<Eigen::MatrixXd> x, const Eigen::Map<Eigen::VectorXd> y,
    const Eigen::Map<Eigen::VectorXi> y_g,
    const Rcpp::List r_vp, const Rcpp::List r_pp, const bool unconstrained) {

  MicroCreditData data(x, y, y_g);
  VariationalParameters<double> base_vp(data.k, data.n_g);
  PriorParameters<double> pp(data.k);

  convert_from_list(r_vp, base_vp);
  convert_from_list(r_pp, pp);

  WishartParameterEncoder
    lambda_encoder(base_vp, pp.lambda_diag_min, pp.lambda_n_min, unconstrained);
  MicroCreditWishartLogLikelihood LambdaLogLik(data, base_vp, pp, lambda_encoder);
  WishartMomentParameterization LambdaMoments(base_vp, lambda_encoder);

  Rcpp::List ret;

  double loglik;

  Eigen::VectorXd dl_dtheta(lambda_encoder.dim);
  Eigen::MatrixXd d2l_dtheta2(lambda_encoder.dim, lambda_encoder.dim);

  Eigen::VectorXd moments(lambda_encoder.dim);
  Eigen::MatrixXd dmoment_dtheta_t(lambda_encoder.dim, lambda_encoder.dim);

  Eigen::VectorXd theta = lambda_encoder.get_parameter_vector(base_vp);

  stan::math::jacobian(LambdaMoments, theta, moments, dmoment_dtheta_t);
  Eigen::MatrixXd dmoment_dtheta = dmoment_dtheta_t.transpose();
  stan::math::hessian(LambdaLogLik, theta, loglik, dl_dtheta, d2l_dtheta2);
  vector<Eigen::MatrixXd> d2moment_dtheta2_vec =
    GetJacobianHessians(LambdaMoments, theta);
  Eigen::MatrixXd d2l_dm2 = transform_hessian(dmoment_dtheta, d2moment_dtheta2_vec,
    dl_dtheta, d2l_dtheta2);
  Eigen::VectorXd dl_dm = dmoment_dtheta.transpose().colPivHouseholderQr().solve(dl_dtheta);

  ret["loglik"] = loglik;
  ret["theta"] = theta;
  ret["moments"] = moments;

  ret["dl_dm"] = dl_dm;
  ret["d2l_dm2"] = d2l_dm2;
  ret["dl_dtheta"] = dl_dtheta;
  ret["d2l_dtheta2"] = d2l_dtheta2;
  ret["dmoment_dtheta"] = dmoment_dtheta;

  return ret;
}


// Note: as of May 3 2016, due to a bug in Stan this returns the transpose
// of the Jacobian, not the Jacobian.
// [[Rcpp::export]]
Rcpp::List WishartMomentParameterizationJacobian(
    const Rcpp::List r_vp, const Rcpp::List r_pp, const bool unconstrained) {

  int k = r_vp["k"];
  int n_g = r_vp["n_g"];
  VariationalParameters<double> base_vp(k, n_g);
  PriorParameters<double> pp(k);

  convert_from_list(r_vp, base_vp);
  convert_from_list(r_pp, pp);

  WishartParameterEncoder lambda_encoder(base_vp, pp.lambda_diag_min,
                                         pp.lambda_n_min, unconstrained);
  WishartMomentParameterization LambdaMoments(base_vp, lambda_encoder);

  Eigen::VectorXd moments(lambda_encoder.dim);
  Eigen::MatrixXd dmoment_dparams(lambda_encoder.dim, lambda_encoder.dim);
  Eigen::VectorXd params = lambda_encoder.get_parameter_vector(base_vp);

  stan::math::set_zero_all_adjoints();
  stan::math::jacobian(LambdaMoments, params, moments, dmoment_dparams);

  Rcpp::List ret;
  // TODO: not necessarily the natural_params anymore
  ret["natural_params"] = params;
  ret["moment_params"] = moments;
  ret["dmoment_dparams"] = dmoment_dparams;

  return ret;
}


struct TestJacobianObjective {
  Eigen::MatrixXd A;
  TestJacobianObjective(const Eigen::MatrixXd A): A(A) {};
  template <typename T> VectorXT<T> operator()(VectorXT<T> const &theta) const {
    MatrixXT<T> A_T = A.template cast<T>();
    VectorXT<T> prod = A_T * theta;
    return prod;
  }
};


// A function to check whether Stan is returning the jacobian or its transpose.
// [[Rcpp::export]]
Rcpp::List TestJacobian() {

  Eigen::MatrixXd A(2, 3);
  A << 1, 2, 3, 4, 5, 6;

  TestJacobianObjective obj = TestJacobianObjective(A);

  Eigen::VectorXd theta(3);
  theta << 1, 2, 3;

  Eigen::VectorXd val(2);
  Eigen::MatrixXd jac(3, 3); // Make it too large on purpose to avoid segfaults.
  stan::math::set_zero_all_adjoints();
  stan::math::jacobian(obj, theta, val, jac);

  Rcpp::List ret;
  // TODO: not necessarily the natural_params anymore
  ret["theta"] = theta;
  ret["val"] = val;
  ret["jac"] = jac;
  ret["A"] = A;

  return ret;
}


// [[Rcpp::export]]
Rcpp::List PriorSensitivity(const Rcpp::List r_vp, const Rcpp::List r_pp) {
  // TODO: Add an unconstrained flag like elsewhere

  int k = r_vp["k"];
  int n_g = r_vp["n_g"];
  VariationalParameters<double> vp(k, n_g);
  PriorParameters<double> pp(k);

  convert_from_list(r_vp, vp);
  convert_from_list(r_pp, pp);

  VariationalParameterEncoder
    vp_encoder(vp, pp.lambda_diag_min, pp.lambda_n_min, true);
  PriorParameterEncoder pp_encoder(pp);
  ModelParameterEncoder encoder(vp_encoder, pp_encoder);

  double prior_val;
  Eigen::VectorXd prior_grad(encoder.dim);
  Eigen::MatrixXd prior_hess(encoder.dim, encoder.dim);
  Eigen::VectorXd theta = encoder.get_parameter_vector(vp, pp);
  MicroCreditLogPrior LogPrior(vp, pp, encoder);

  stan::math::set_zero_all_adjoints();
  stan::math::hessian(LogPrior, theta, prior_val, prior_grad, prior_hess);

  Rcpp::List ret;
  ret["prior_val"] = prior_val;
  ret["prior_grad"] = prior_grad;
  ret["prior_hess"] = prior_hess;

  return ret;
}


// This is needed to look up values in the LRVB covariance matrix.
// [[Rcpp::export]]
Rcpp::List GetParameterEncoder(const Rcpp::List r_vp, const Rcpp::List r_pp) {
  int k = r_vp["k"];
  int n_g = r_vp["n_g"];
  VariationalParameters<double> base_vp(k, n_g);
  PriorParameters<double> pp(k);
  convert_from_list(r_pp, pp);
  VariationalParameterEncoder
    vp_encoder(base_vp, pp.lambda_diag_min, pp.lambda_n_min, true);
  return convert_to_list(vp_encoder);
}


// [[Rcpp::export]]
Rcpp::List GetPriorParameterEncoder(const Rcpp::List r_pp) {
  int k = r_pp["k"];
  PriorParameters<double> pp(k);
  convert_from_list(r_pp, pp);
  PriorParameterEncoder pp_encoder(pp);
  return convert_to_list(pp_encoder);
}


// This is needed to look up values in the prior sensitivity matrix.
// [[Rcpp::export]]
Rcpp::List GetModelParameterEncoder(const Rcpp::List r_vp, const Rcpp::List r_pp) {
  int k = r_vp["k"];
  int n_g = r_vp["n_g"];
  VariationalParameters<double> base_vp(k, n_g);
  PriorParameters<double> pp(k);
  convert_from_list(r_pp, pp);
  VariationalParameterEncoder vp_encoder(base_vp, pp.lambda_diag_min, pp.lambda_n_min, true);
  PriorParameterEncoder pp_encoder(pp);
  ModelParameterEncoder model_encoder(vp_encoder, pp_encoder);
  Rcpp::List r_list = convert_to_list(model_encoder);
  r_list["dim"] = model_encoder.dim;
  r_list["variational_dim"] = vp_encoder.dim;
  r_list["prior_dim"] = pp_encoder.dim;
  r_list["variational_offset"] = model_encoder.variational_offset;
  r_list["prior_offset"] = model_encoder.prior_offset;
  return r_list;
}


// Note that, unlike above, vp_params must have the parameters necessary
// for the covariance matrix, which is different than those for the likelihood.
// This is because the likelihood uses the Wishart moment parameters but
// the covariance needs the natural parameters.

// [[Rcpp::export]]
Eigen::SparseMatrix<double> GetVariationalCovariance(
    const Rcpp::List vp_params, const Rcpp::List r_pp) {

  // Get an encoder.
  int k = vp_params["k"];
  int n_g = vp_params["n_g"];
  VariationalParameters<double> base_vp(k, n_g);
  PriorParameters<double> pp(k);
  convert_from_list(r_pp, pp);

  VariationalParameterEncoder
    vp_encoder(base_vp, pp.lambda_diag_min, pp.lambda_n_min, true);

  std::vector<Triplet> terms;
  std::vector<Triplet> all_terms;

  const Eigen::Map<Eigen::VectorXd> e_mu = vp_params["e_mu"];
  const Eigen::Map<Eigen::MatrixXd> e_mu2 = vp_params["e_mu2"];

  terms = get_mvn_covariance_terms(
    e_mu, e_mu2, vp_encoder.e_mu_offset, vp_encoder.e_mu2_offset);
  all_terms.insert(all_terms.end(), terms.begin(), terms.end());

  // Lambda natural parameters
  const Eigen::Map<Eigen::MatrixXd> v_par = vp_params["lambda_v_par"];
  const double n_par = vp_params["lambda_n_par"];
  terms = get_wishart_covariance_terms(
    v_par, n_par, vp_encoder.lambda_v_par_offset, vp_encoder.lambda_n_par_offset);
  all_terms.insert(all_terms.end(), terms.begin(), terms.end());

  Rcpp::List tau_alpha_vec_list = vp_params["tau_alpha_vec"];
  Rcpp::List tau_beta_vec_list = vp_params["tau_beta_vec"];
  Rcpp::List e_mu_g_vec_list = vp_params["e_mu_g_vec"];
  Rcpp::List e_mu2_g_vec_list = vp_params["e_mu2_g_vec"];
  if (e_mu2_g_vec_list.size() != n_g) {
    throw std::runtime_error("e_mu2_g_vec_list does not have n_g elements");
  }
  if (e_mu_g_vec_list.size() != n_g) {
    throw std::runtime_error("e_mu_g_vec_list does not have n_g elements");
  }
  for (int g = 0; g < n_g; g++) {
    // Tau:
    const double tau_alpha = tau_alpha_vec_list[g];
    const double tau_beta = tau_beta_vec_list[g];
    terms = get_gamma_covariance_terms(
      tau_alpha, tau_beta,
      vp_encoder.e_tau_offset[g], vp_encoder.e_log_tau_offset[g]);
    all_terms.insert(all_terms.end(), terms.begin(), terms.end());

    // mu_g:
    const Eigen::Map<Eigen::VectorXd> e_mu_g = e_mu_g_vec_list[g];
    const Eigen::Map<Eigen::MatrixXd> e_mu2_g = e_mu2_g_vec_list[g];
    terms = get_mvn_covariance_terms(
      e_mu_g, e_mu2_g, vp_encoder.e_mu_g_offset[g], vp_encoder.e_mu2_g_offset[g]);
    all_terms.insert(all_terms.end(), terms.begin(), terms.end());
  }

  // Construct a sparse matrix.
  Eigen::SparseMatrix<double> theta_cov(vp_encoder.dim, vp_encoder.dim);
  theta_cov.setFromTriplets(all_terms.begin(), all_terms.end());
  theta_cov.makeCompressed();

  return theta_cov;
}
