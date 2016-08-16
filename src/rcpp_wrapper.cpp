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
    r_list["k_reg"] = vp.k;

    // Assume this is all the same.
    r_list["diag_min"] = vp.mu.diag_min;

    r_list["mu_loc"] = vp.mu.loc;
    r_list["mu_info"] = vp.mu.info.mat;

    r_list["lambda_v"] = vp.lambda.v.mat;
    r_list["lambda_n"] = vp.lambda.n;

    if (vp.n_g > 0) {
        r_list["tau_alpha_min"] = vp.tau[0].alpha_min;
        r_list["tau_beta_min"] = vp.tau[0].beta_min;
    }

    Rcpp::List mu_g(vp.n_g);
    Rcpp::List tau(vp.n_g);
    for (int g = 0; g < vp.n_g; g++) {
        Rcpp::List this_mu_g;
        this_mu_g["loc"] = vp.mu_g[g].loc;
        this_mu_g["info"] = vp.mu_g[g].info.mat;
        mu_g[g] = this_mu_g;

        Rcpp::List this_tau;
        this_tau["alpha"] = vp.tau[g].alpha;
        this_tau["beta"] = vp.tau[g].beta;
        tau[g] = this_tau;
    }

    r_list["mu_g"] = mu_g;
    r_list["tau"] = tau;

    return r_list;
};


VariationalParameters<double>
ConvertParametersFromList(Rcpp::List r_list) {

    int k_reg = r_list["k_reg"];
    int n_g = r_list["n_g"];
    VariationalParameters<double> vp(k_reg, n_g, true);

    double diag_min = Rcpp::as<double>(r_list["diag_min"]);
    vp.mu.loc = Rcpp::as<Eigen::VectorXd>(r_list["mu_loc"]);
    vp.mu.info.mat = Rcpp::as<Eigen::MatrixXd>(r_list["mu_info"]);
    vp.mu.diag_min = diag_min;

    vp.lambda.v.mat = Rcpp::as<Eigen::MatrixXd>(r_list["lambda_v"]);
    vp.lambda.n = Rcpp::as<double>(r_list["lambda_n"]);
    vp.lambda.diag_min = diag_min;

    double tau_alpha_min = 0;
    double tau_beta_min = 0;
    if (vp.n_g > 0) {
        tau_alpha_min = Rcpp::as<double>(r_list["tau_alpha_min"]);
        tau_beta_min = Rcpp::as<double>(r_list["tau_beta_min"]);
    }

    Rcpp::List mu_g_list = r_list["mu_g"];
    if (vp.n_g != mu_g_list.size()) {
        throw std::runtime_error("mu_g size does not match");
    }

    for (int g = 0; g < vp.n_g; g++) {
        Rcpp::List this_mu_g = mu_g_list[g];
        vp.mu_g[g].loc = Rcpp::as<Eigen::VectorXd>(this_mu_g["loc"]);
        vp.mu_g[g].info.mat = Rcpp::as<Eigen::MatrixXd>(this_mu_g["info"]);
        vp.mu_g[g].diag_min = diag_min;
    }

    Rcpp::List tau_list = r_list["tau"];
    if (vp.n_g != tau_list.size()) {
        throw std::runtime_error("tau size does not match");
    }

    for (int g = 0; g < vp.n_g; g++) {
        Rcpp::List this_tau = tau_list[g];
        vp.tau[g].alpha = Rcpp::as<double>(this_tau["alpha"]);
        vp.tau[g].beta = Rcpp::as<double>(this_tau["beta"]);
        vp.tau[g].alpha_min = tau_alpha_min;
        vp.tau[g].beta_min = tau_beta_min;
    }

    return vp;
}


Rcpp::List ConvertMomentsToList(MomentParameters<double> const &mp) {
    Rcpp::List r_list;
    r_list["n_g"] = mp.n_g;
    r_list["k_reg"] = mp.k;

    r_list["mu_e_vec"] = mp.mu.e_vec;
    r_list["mu_e_outer"] = mp.mu.e_outer.mat;

    r_list["lambda_e"] = mp.lambda.e.mat;
    r_list["lambda_e_log_det"] = mp.lambda.e_log_det;

    Rcpp::List mu_g(mp.n_g);
    Rcpp::List tau(mp.n_g);
    for (int g = 0; g < mp.n_g; g++) {
        Rcpp::List this_mu_g;
        this_mu_g["e_vec"] = mp.mu_g[g].e_vec;
        this_mu_g["e_outer"] = mp.mu_g[g].e_outer.mat;
        mu_g[g] = this_mu_g;

        Rcpp::List this_tau;
        this_tau["e"] = mp.tau[g].e;
        this_tau["e_log"] = mp.tau[g].e_log;
        tau[g] = this_tau;
    }

    r_list["mu_g"] = mu_g;
    r_list["tau"] = tau;

    return r_list;
};


MomentParameters<double>
ConvertMomentsFromList(Rcpp::List r_list) {

    int k_reg = r_list["k_reg"];
    int n_g = r_list["n_g"];
    MomentParameters<double> mp(k_reg, n_g);

    mp.mu.e_vec = Rcpp::as<Eigen::VectorXd>(r_list["mu_e_vec"]);
    mp.mu.e_outer.mat = Rcpp::as<Eigen::MatrixXd>(r_list["mu_e_outer"]);

    mp.lambda.e.mat = Rcpp::as<Eigen::MatrixXd>(r_list["lambda_e"]);
    mp.lambda.e_log_det = Rcpp::as<double>(r_list["lambda_e_log_det"]);

    Rcpp::List mu_g_list = r_list["mu_g"];
    if (mp.n_g != mu_g_list.size()) {
        throw std::runtime_error("mu_g size does not match");
    }

    for (int g = 0; g < mp.n_g; g++) {
        Rcpp::List this_mu_g = mu_g_list[g];
        mp.mu_g[g].e_vec = Rcpp::as<Eigen::VectorXd>(this_mu_g["e_vec"]);
        mp.mu_g[g].e_outer.mat = Rcpp::as<Eigen::MatrixXd>(this_mu_g["e_outer"]);
    }

    Rcpp::List tau_list = r_list["tau"];
    if (mp.n_g != tau_list.size()) {
        throw std::runtime_error("tau size does not match");
    }

    for (int g = 0; g < mp.n_g; g++) {
        Rcpp::List this_tau = tau_list[g];
        mp.tau[g].e = Rcpp::as<double>(this_tau["e"]);
        mp.tau[g].e_log = Rcpp::as<double>(this_tau["e_log"]);
    }

    return mp;
}


// Update pp in place with the values from r_list.
PriorParameters<double> ConvertPriorsFromlist(Rcpp::List r_list) {

    int k_reg = r_list["k_reg"];
    PriorParameters<double> pp(k_reg);
    pp.mu.loc = Rcpp::as<Eigen::VectorXd>(r_list["mu_loc"]);
    pp.mu.info.mat = Rcpp::as<Eigen::MatrixXd>(r_list["mu_info"]);

    pp.lambda_eta = Rcpp::as<double>(r_list["lambda_eta"]);
    pp.lambda_alpha = Rcpp::as<double>(r_list["lambda_alpha"]);
    pp.lambda_beta = Rcpp::as<double>(r_list["lambda_beta"]);

    pp.tau.alpha = Rcpp::as<double>(r_list["tau_alpha"]);
    pp.tau.beta = Rcpp::as<double>(r_list["tau_beta"]);

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
    VariationalParameters<double> vp(k, n_g, true);
    return ConvertParametersToList(vp);
}


// [[Rcpp::export]]
Rcpp::List GetParametersFromVector(
    const Rcpp::List r_vp,
    const Eigen::Map<Eigen::VectorXd> r_theta,
    bool unconstrained) {

  VectorXd theta = r_theta;
  VariationalParameters<double> vp = ConvertParametersFromList(r_vp);
  if (theta.size() != vp.offsets.encoded_size) {
    throw std::runtime_error("Theta is the wrong size");
  }
  vp.unconstrained = unconstrained;
  SetFromVector(theta, vp);
  Rcpp::List vp_list = ConvertParametersToList(vp);
  return vp_list;
}


// [[Rcpp::export]]
Rcpp::List GetMomentsFromVector(
    const Rcpp::List r_mp,
    const Eigen::Map<Eigen::VectorXd> r_theta) {

  VectorXd theta = r_theta;
  MomentParameters<double> mp = ConvertMomentsFromList(r_mp);
  if (theta.size() != mp.offsets.encoded_size) {
    throw std::runtime_error("Theta is the wrong size");
  }
  SetFromVector(theta, mp);
  Rcpp::List mp_list = ConvertMomentsToList(mp);
  return mp_list;
}


// [[Rcpp::export]]
Eigen::VectorXd GetVectorFromParameters(
    const Rcpp::List r_vp,
    bool unconstrained) {

  VariationalParameters<double> vp = ConvertParametersFromList(r_vp);
  vp.unconstrained = unconstrained;
  VectorXd theta = GetParameterVector(vp);
  return theta;
}


// For testing.
// [[Rcpp::export]]
Rcpp::List ToAndFromParameters(const Rcpp::List r_vp) {
    VariationalParameters<double> vp = ConvertParametersFromList(r_vp);
    return ConvertParametersToList(vp);
}


// r_y_g should be one-indexed group indicators.
// [[Rcpp::export]]
Rcpp::List GetElboDerivatives(
    const Eigen::Map<Eigen::MatrixXd> r_x,
    const Eigen::Map<Eigen::VectorXd> r_y,
    const Eigen::Map<Eigen::VectorXi> r_y_g,
    const Rcpp::List r_vp, const Rcpp::List r_pp,
    const bool calculate_gradient,
    const bool calculate_hessian,
    const bool unconstrained) {

    Eigen::MatrixXd x = r_x;
    Eigen::VectorXd y = r_y;
    Eigen::VectorXi y_g = r_y_g;
    MicroCreditData data(x, y, y_g);
    VariationalParameters<double> vp = ConvertParametersFromList(r_vp);
    PriorParameters<double> pp = ConvertPriorsFromlist(r_pp);

    Derivatives derivs =
        GetElboDerivatives(data, vp, pp, unconstrained,
            calculate_gradient, calculate_hessian);
    Rcpp::List ret = ConvertDerivativesToList(derivs);
    return ret;
}


// r_y_g should be one-indexed group indicators.
// [[Rcpp::export]]
Rcpp::List GetCustomElboDerivatives(
    const Eigen::Map<Eigen::MatrixXd> r_x,
    const Eigen::Map<Eigen::VectorXd> r_y,
    const Eigen::Map<Eigen::VectorXi> r_y_g,
    const Rcpp::List r_vp, const Rcpp::List r_pp,
    bool include_obs,
    bool include_hier,
    bool include_prior,
    bool include_entropy,
    const bool calculate_gradient,
    const bool calculate_hessian,
    const bool unconstrained) {

    Eigen::MatrixXd x = r_x;
    Eigen::VectorXd y = r_y;
    Eigen::VectorXi y_g = r_y_g;
    MicroCreditData data(x, y, y_g);
    VariationalParameters<double> vp = ConvertParametersFromList(r_vp);
    PriorParameters<double> pp = ConvertPriorsFromlist(r_pp);

    Derivatives derivs =
        GetElboDerivatives(data, vp, pp,
            include_obs, include_hier, include_prior, include_entropy,
            unconstrained, calculate_gradient, calculate_hessian);
    Rcpp::List ret = ConvertDerivativesToList(derivs);
    return ret;
}


// [[Rcpp::export]]
Rcpp::List GetMoments(const Rcpp::List r_vp) {
    VariationalParameters<double> vp = ConvertParametersFromList(r_vp);
    MomentParameters<double> mp(vp);
    Rcpp::List r_mp = ConvertMomentsToList(mp);
    return r_mp;
};


// [[Rcpp::export]]
Rcpp::List GetMomentJacobian(const Rcpp::List r_vp) {
    VariationalParameters<double> vp = ConvertParametersFromList(r_vp);

    Derivatives derivs = GetMomentJacobian(vp);
    Rcpp::List ret = ConvertDerivativesToList(derivs);
    return ret;
}


// [[Rcpp::export]]
Eigen::SparseMatrix<double> GetCovariance(const Rcpp::List r_vp) {
    VariationalParameters<double> vp = ConvertParametersFromList(r_vp);
    MomentParameters<double> mp(vp);
    return GetCovariance(vp, mp.offsets);
}




// // [[Rcpp::export]]
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
