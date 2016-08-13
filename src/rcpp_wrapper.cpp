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


Rcpp::List ConvertParametersToList(VariationalParameters<double> const &vp) {
    Rcpp::List r_list;
    r_list["n_g"] = vp.n_g;
    r_list["k"] = vp.k;

    // Assume this is all the same.
    r_list["diag_min"] = vp.mu.diag_min;

    r_list["mu_loc"] = vp.mu.loc;
    r_list["mu_info"] = vp.mu.info.mat;

    r_list["lambda_v"] = vp.lambda.v.mat;
    r_list["lambvda_n"] = vp.mu.n;

    r_list["tau_alpha"] = vp.tau.alpha;
    r_list["tau_beta"] = vp.tau.beta;
    r_list["tau_alpha_min"] = vp.tau.alpha_min;
    r_list["tau_beta_min"] = vp.tau.beta_min;

    Rcpp::List mu_g(vp.n_groups);
    Rcpp::List tau(vp.n_groups);
    for (int g = 0; g < vp.n_groups; g++) {
        Rcpp::List this_mu_g;
        this_mu_g["loc"] = vp.mu_g[g].loc;
        this_mu_g["info"] = vp.mu_g[g].info.mat;

        Rcpp::List this_tau;
        this_tau["alpha"] = vp.tau[g].alpha;
        this_tau["beta"] = vp.tau[g].beta;
    }

    r_list["mu_g"] = mu_g;
    r_list["tau"] = tau;

    return r_list;
};


VariationalParameters<double> ConvertParametersFromList(Rcpp::List r_list) {

    int k_reg = r_list["k_reg"];
    int n_groups = r_list["n_groups"];
    VariationalParameters<double> vp(k_reg, n_groups);

    vp.beta.loc = Rcpp::as<VectorXd>(r_list["beta_loc"]);
    vp.beta.info.mat = Rcpp::as<MatrixXd>(r_list["beta_info"]);
    vp.beta.diag_min = Rcpp::as<double>(r_list["beta_diag_min"]);

    vp.mu.loc = Rcpp::as<double>(r_list["mu_loc"]);
    vp.mu.info = Rcpp::as<double>(r_list["mu_info"]);
    vp.mu.info_min = Rcpp::as<double>(r_list["mu_info_min"]);

    vp.tau.alpha = Rcpp::as<double>(r_list["tau_alpha"]);
    vp.tau.beta = Rcpp::as<double>(r_list["tau_beta"]);
    vp.tau.alpha_min = Rcpp::as<double>(r_list["tau_alpha_min"]);
    vp.tau.beta_min = Rcpp::as<double>(r_list["tau_beta_min"]);

    Rcpp::List u_list = r_list["u_vec"];
        if (vp.n_groups != u_list.size()) {
        throw std::runtime_error("u size does not match");
    }

    double u_info_min = Rcpp::as<double>(r_list["u_info_min"]);
    for (int g = 0; g < vp.n_groups; g++) {
        Rcpp::List this_u = u_list[g];
        vp.u[g].loc = Rcpp::as<double>(this_u["u_loc"]);
        vp.u[g].info = Rcpp::as<double>(this_u["u_info"]);
        vp.u[g].info_min = u_info_min;
    }

    return vp;
}


// Update pp in place with the values from r_list.
PriorParameters<double> ConvertPriorsFromlist(Rcpp::List r_list) {

    int k_reg = r_list["k_reg"];
    PriorParameters<double> pp(k_reg);
    pp.mu_mean = Rcpp::as<Eigen::VectorXd>(r_list["mu_mean"]);
    pp.mu_info.mat = Rcpp::as<Eigen::MatrixXd>(r_list["mu_info"]);

    pp.lambda_eta = Rcpp::as<double>(r_list["lambda_eta"]);
    pp.lambda_alpha = Rcpp::as<double>(r_list["lambda_alpha"]);
    pp.lambda_beta = Rcpp::as<double>(r_list["lambda_beta"]);

    pp.tau_alpha = Rcpp::as<double>(r_list["tau_alpha"]);
    pp.tau_beta = Rcpp::as<double>(r_list["tau_beta"]);

    pp.lambda_diag_min = Rcpp::as<double>(r_list["lambda_diag_min"]);
    pp.lambda_n_min = Rcpp::as<double>(r_list["lambda_n_min"]);

    return pp;
};


Rcpp::List ConvertDerivativesToList(Derivatives derivs) {
    Rcpp::List r_list;
    r_list["val"]  = derivs.val;
    r_list["grad"] = derivs.grad;
    r_list["hess"] = derivs.hess;
    return r_list;
}


// [[Rcpp::export]]
Rcpp::List GetEmptyVariationalParameters(int k, int n_g) {
    VariationalParameters<double> vp(k, n_g);
    return ConvertParametersToList(vp);
}


// r_y_g should be one-indexed group indicators.
// [[Rcpp::export]]
Rcpp::List GetElboDerivatives(
    const Eigen::Map<Eigen::MatrixXd> r_x,
    const Eigen::Map<Eigen::VectorXd> r_y,
    const Eigen::Map<Eigen::VectorXi> r_y_g,
    const Rcpp::List r_vp, const Rcpp::List r_pp,
    const bool calculate_hessian, const bool unconstrained) {

  Eigen::MatrixXd x = r_x;
  Eigen::VectorXd y = r_y;
  Eigen::VectorXi y_g = r_y_g;
  MicroCreditData data(x, y, y_g);
  VariationalParameters<double> vp = ConvertParametersFromList(r_vp);
  PriorParameters<double> pp = ConvertPriorsFromlist(r_pp);

  Derivatives derivs =
    GetElboDerivatives(data, vp, pp, unconstrained, calculate_hessian);

  Rcpp::List ret = ConvertDerivativesToList(derivs);
  return ret;
}


// [[Rcpp::export]]
// Rcpp::List PriorSensitivity(const Rcpp::List r_vp, const Rcpp::List r_pp) {
//   // TODO: Add an unconstrained flag like elsewhere
//
//   int k = r_vp["k"];
//   int n_g = r_vp["n_g"];
//   VariationalParameters<double> vp(k, n_g);
//   PriorParameters<double> pp(k);
//
//   convert_from_list(r_vp, vp);
//   convert_from_list(r_pp, pp);
//
//   VariationalParameterEncoder
//     vp_encoder(vp, pp.lambda_diag_min, pp.lambda_n_min, true);
//   PriorParameterEncoder pp_encoder(pp);
//   ModelParameterEncoder encoder(vp_encoder, pp_encoder);
//
//   double prior_val;
//   Eigen::VectorXd prior_grad(encoder.dim);
//   Eigen::MatrixXd prior_hess(encoder.dim, encoder.dim);
//   Eigen::VectorXd theta = encoder.get_parameter_vector(vp, pp);
//   MicroCreditLogPrior LogPrior(vp, pp, encoder);
//
//   stan::math::set_zero_all_adjoints();
//   stan::math::hessian(LogPrior, theta, prior_val, prior_grad, prior_hess);
//
//   Rcpp::List ret;
//   ret["prior_val"] = prior_val;
//   ret["prior_grad"] = prior_grad;
//   ret["prior_hess"] = prior_hess;
//
//   return ret;
// }



// [[Rcpp::export]]
Eigen::SparseMatrix<double>
GetVariationalCovariance(const Rcpp::List r_vp) {
    VariationalParameters<double> vp = ConvertParametersFromList(r_vp);
    return(GetCovariance(vp, vp.offsets));
}
