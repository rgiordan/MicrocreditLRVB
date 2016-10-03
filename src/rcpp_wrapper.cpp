# include <Rcpp.h>
# include <RcppEigen.h>

# include <vector>
# include <ctime>

# include <stan/math.hpp>
// # include <stan/math/mix/mat/functor/hessian.hpp>

// # include "variational_parameters.h"
// # include "exponential_families.h"
# include "microcredit_model.h"

// # include "differentiate_jacobian.h"
# include "transform_hessian.h"
# include "boost/math/complex/fabs.hpp" // Why not included in stan?

typedef Eigen::Triplet<double> Triplet; // For populating sparse matrices


Rcpp::List ConvertParametersToList(VariationalParameters<double> const &vp) {
    Rcpp::List r_list;
    r_list["n_g"] = vp.n_g;
    r_list["k_reg"] = vp.k;
    r_list["encoded_size"] = vp.offsets.encoded_size;

    r_list["mu_loc"] = vp.mu.loc;
    r_list["mu_info"] = vp.mu.info.mat;
    // Assume this is all the same for mu and mu_g
    r_list["mu_diag_min"] = vp.mu.diag_min;
    r_list["mu_draws"] = vp.mu_draws.std_draws;

    r_list["lambda_v"] = vp.lambda.v.mat;
    r_list["lambda_n"] = vp.lambda.n;
    r_list["lambda_n_min"] = vp.lambda.n_min;
    r_list["lambda_diag_min"] = vp.mu.diag_min;

    if (vp.n_g > 0) {
        r_list["tau_alpha_min"] = vp.tau[0].alpha_min;
        r_list["tau_beta_min"] = vp.tau[0].beta_min;
        r_list["tau_alpha_max"] = vp.tau[0].alpha_max;
        r_list["tau_beta_max"] = vp.tau[0].beta_max;
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

    vp.mu.loc = Rcpp::as<Eigen::VectorXd>(r_list["mu_loc"]);
    vp.mu.info.mat = Rcpp::as<Eigen::MatrixXd>(r_list["mu_info"]);
    vp.mu.diag_min = Rcpp::as<double>(r_list["mu_diag_min"]);;

    vp.lambda.v.mat = Rcpp::as<Eigen::MatrixXd>(r_list["lambda_v"]);
    vp.lambda.n = Rcpp::as<double>(r_list["lambda_n"]);
    vp.lambda.n_min = Rcpp::as<double>(r_list["lambda_n_min"]);
    vp.mu.diag_min = Rcpp::as<double>(r_list["lambda_diag_min"]);;

    // These should be standard normal draws.
    VectorXd mu_draws = Rcpp::as<Eigen::VectorXd>(r_list["mu_draws"]);
    vp.mu_draws.SetDraws(mu_draws);

    double tau_alpha_min = 0;
    double tau_beta_min = 0;
    double tau_alpha_max = 1e9;
    double tau_beta_max = 1e9;
    if (vp.n_g > 0) {
        tau_alpha_min = Rcpp::as<double>(r_list["tau_alpha_min"]);
        tau_beta_min = Rcpp::as<double>(r_list["tau_beta_min"]);
        tau_alpha_max = Rcpp::as<double>(r_list["tau_alpha_max"]);
        tau_beta_max = Rcpp::as<double>(r_list["tau_beta_max"]);
    }

    Rcpp::List mu_g_list = r_list["mu_g"];
    if (vp.n_g != mu_g_list.size()) {
        throw std::runtime_error("mu_g size does not match");
    }

    for (int g = 0; g < vp.n_g; g++) {
        Rcpp::List this_mu_g = mu_g_list[g];
        vp.mu_g[g].loc = Rcpp::as<Eigen::VectorXd>(this_mu_g["loc"]);
        vp.mu_g[g].info.mat = Rcpp::as<Eigen::MatrixXd>(this_mu_g["info"]);
        vp.mu_g[g].diag_min = vp.mu.diag_min;
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
    r_list["encoded_size"] = mp.offsets.encoded_size;

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
    mp.unconstrained = false;

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
PriorParameters<double> ConvertPriorsFromList(Rcpp::List r_list) {
    int k_reg = r_list["k_reg"];
    PriorParameters<double> pp(k_reg);

    pp.monte_carlo_prior = Rcpp::as<bool>(r_list["monte_carlo_prior"]);

    pp.mu.loc = Rcpp::as<Eigen::VectorXd>(r_list["mu_loc"]);
    pp.mu.info.mat = Rcpp::as<Eigen::MatrixXd>(r_list["mu_info"]);
    pp.mu_t_loc = Rcpp::as<double>(r_list["mu_t_loc"]);
    pp.mu_t_scale = Rcpp::as<double>(r_list["mu_t_scale"]);
    pp.mu_t_df = Rcpp::as<double>(r_list["mu_t_df"]);
    pp.epsilon = Rcpp::as<double>(r_list["epsilon"]);

    pp.lambda_eta = Rcpp::as<double>(r_list["lambda_eta"]);
    pp.lambda_alpha = Rcpp::as<double>(r_list["lambda_alpha"]);
    pp.lambda_beta = Rcpp::as<double>(r_list["lambda_beta"]);

    pp.tau.alpha = Rcpp::as<double>(r_list["tau_alpha"]);
    pp.tau.beta = Rcpp::as<double>(r_list["tau_beta"]);

    return pp;
};


Rcpp::List ConvertPriorsToList(PriorParameters<double> pp) {
    Rcpp::List r_list;

    r_list["encoded_size"] = pp.offsets.encoded_size;
    r_list["k_reg"] = pp.k;
    r_list["monte_carlo_prior"] = pp.monte_carlo_prior;

    r_list["mu_loc"] = pp.mu.loc;
    r_list["mu_info"] = pp.mu.info.mat;
    r_list["mu_t_loc"] = pp.mu_t_loc;
    r_list["mu_t_scale"] = pp.mu_t_scale;
    r_list["mu_t_df"] = pp.mu_t_df;
    r_list["epsilon"] = pp.epsilon;

    r_list["lambda_eta"] = pp.lambda_eta;
    r_list["lambda_alpha"] = pp.lambda_alpha;
    r_list["lambda_beta"] = pp.lambda_beta;

    r_list["tau_alpha"] = pp.tau.alpha;
    r_list["tau_beta"] = pp.tau.beta;

    return r_list;
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
Rcpp::List GetEmptyPriors(int k) {
    PriorParameters<double> pp(k);
    return ConvertPriorsToList(pp);
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
Eigen::VectorXd GetVectorFromParameters(
        const Rcpp::List r_vp,
        bool unconstrained) {

    VariationalParameters<double> vp = ConvertParametersFromList(r_vp);
    vp.unconstrained = unconstrained;
    VectorXd theta = GetParameterVector(vp);
    return theta;
}


// [[Rcpp::export]]
Rcpp::List GetParametersFromGlobalVector(
        const Rcpp::List r_vp,
        const Eigen::Map<Eigen::VectorXd> r_theta,
        bool unconstrained) {

    VectorXd theta = r_theta;
    VariationalParameters<double> vp = ConvertParametersFromList(r_vp);
    vp.unconstrained = unconstrained;
    SetFromGlobalVector(theta, vp);
    Rcpp::List vp_list = ConvertParametersToList(vp);
    return vp_list;
}


// [[Rcpp::export]]
Eigen::VectorXd GetGlobalVectorFromParameters(
        const Rcpp::List r_vp,
        bool unconstrained) {

    VariationalParameters<double> vp = ConvertParametersFromList(r_vp);
    vp.unconstrained = unconstrained;
    VectorXd theta = GetGlobalParameterVector(vp);
    return theta;
}


// [[Rcpp::export]]
Rcpp::List GetMomentsFromVector(
        const Rcpp::List r_mp,
        const Eigen::Map<Eigen::VectorXd> r_theta) {

    VectorXd theta = r_theta;
    MomentParameters<double> mp = ConvertMomentsFromList(r_mp);
    mp.unconstrained = false;
    if (theta.size() != mp.offsets.encoded_size) {
        throw std::runtime_error("Theta is the wrong size");
    }
    SetFromVector(theta, mp);
    Rcpp::List mp_list = ConvertMomentsToList(mp);
    return mp_list;
}


// [[Rcpp::export]]
Eigen::VectorXd GetVectorFromMoments(const Rcpp::List r_mp) {
    MomentParameters<double> mp = ConvertMomentsFromList(r_mp);
    mp.unconstrained = false;
    VectorXd theta = GetParameterVector(mp);
    return theta;
}


// [[Rcpp::export]]
Rcpp::List GetPriorsFromVector(
    const Rcpp::List r_pp, const Eigen::Map<Eigen::VectorXd> r_theta) {

    VectorXd theta = r_theta;
    PriorParameters<double> pp = ConvertPriorsFromList(r_pp);
    if (theta.size() != pp.offsets.encoded_size) {
        throw std::runtime_error("Theta is the wrong size");
    }
    SetFromVector(theta, pp);
    Rcpp::List pp_list = ConvertPriorsToList(pp);
    return pp_list;
}


// [[Rcpp::export]]
Eigen::VectorXd GetVectorFromPriors(const Rcpp::List r_pp) {
    PriorParameters<double> pp = ConvertPriorsFromList(r_pp);
    VectorXd theta = GetParameterVector(pp);
    return theta;
}


// [[Rcpp::export]]
Rcpp::List GetPriorsAndParametersFromVector(
    const Rcpp::List r_vp, const Rcpp::List r_pp,
    const Eigen::Map<Eigen::VectorXd> r_theta) {

    VectorXd theta = r_theta;
    VariationalParameters<double> vp = ConvertParametersFromList(r_vp);
    vp.unconstrained = false;
    PriorParameters<double> pp = ConvertPriorsFromList(r_pp);

    if (theta.size() != pp.offsets.encoded_size + vp.offsets.encoded_size) {
        throw std::runtime_error("Theta is the wrong size");
    }
    SetFromVector(theta, vp, pp);
    Rcpp::List vp_list = ConvertParametersToList(vp);
    Rcpp::List pp_list = ConvertPriorsToList(pp);

    Rcpp::List ret_list;
    ret_list["vp"] = vp_list;
    ret_list["pp"] = pp_list;
    return ret_list;
}



/////////////////////////////
// Derivative functions:


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
    PriorParameters<double> pp = ConvertPriorsFromList(r_pp);

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
    bool global_only,
    const bool calculate_gradient,
    const bool calculate_hessian,
    const bool unconstrained) {

    Eigen::MatrixXd x = r_x;
    Eigen::VectorXd y = r_y;
    Eigen::VectorXi y_g = r_y_g;
    MicroCreditData data(x, y, y_g);
    VariationalParameters<double> vp = ConvertParametersFromList(r_vp);
    PriorParameters<double> pp = ConvertPriorsFromList(r_pp);

    Derivatives derivs =
        GetElboDerivatives(data, vp, pp,
                include_obs, include_hier, include_prior, include_entropy,
                global_only, unconstrained, calculate_gradient, calculate_hessian);
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
Rcpp::List GetMomentJacobian(const Rcpp::List r_vp, bool unconstrained) {
    VariationalParameters<double> vp = ConvertParametersFromList(r_vp);
    vp.unconstrained = unconstrained;

    Derivatives derivs = GetMomentJacobian(vp);
    Rcpp::List ret = ConvertDerivativesToList(derivs);
    return ret;
}


// [[Rcpp::export]]
Eigen::SparseMatrix<double>
GetSparseELBOHessian(const Eigen::Map<Eigen::MatrixXd> r_x,
        const Eigen::Map<Eigen::VectorXd> r_y,
        const Eigen::Map<Eigen::VectorXi> r_y_g,
        const Rcpp::List r_vp, const Rcpp::List r_pp,
        bool unconstrained) {

        Eigen::MatrixXd x = r_x;
        Eigen::VectorXd y = r_y;
        Eigen::VectorXi y_g = r_y_g;
        MicroCreditData data(x, y, y_g);
        VariationalParameters<double> vp = ConvertParametersFromList(r_vp);
        vp.unconstrained = unconstrained;
        PriorParameters<double> pp = ConvertPriorsFromList(r_pp);
        return GetSparseELBOHessian(data, vp, pp);
}


// [[Rcpp::export]]
Rcpp::List GetLogPriorDerivatives(
    const Rcpp::List r_vp, const Rcpp::List r_pp,
    const bool calculate_gradient,
    const bool calculate_hessian,
    const bool unconstrained) {

    VariationalParameters<double> vp = ConvertParametersFromList(r_vp);
    PriorParameters<double> pp = ConvertPriorsFromList(r_pp);

    Derivatives derivs = GetLogPriorDerivatives(vp, pp, unconstrained,
                                            calculate_gradient,
                                            calculate_hessian);
    Rcpp::List ret = ConvertDerivativesToList(derivs);
    return ret;
}


// [[Rcpp::export]]
Rcpp::List GetLogVariationalDensityDerivatives(
    const Rcpp::List r_obs_mp, const Rcpp::List r_vp,
    const Rcpp::List r_pp, // TOOD: remove this, it's not necessary.
    bool const include_mu,
    bool const include_lambda,
    const Eigen::Map<Eigen::VectorXi> r_include_mu_groups,
    const Eigen::Map<Eigen::VectorXi> r_include_tau_groups,
    bool const unconstrained,
    bool const calculate_gradient) {

    MomentParameters<double> mp_obs = ConvertMomentsFromList(r_obs_mp);
    VariationalParameters<double> vp = ConvertParametersFromList(r_vp);
    vp.unconstrained = unconstrained;
    // PriorParameters<double> pp = ConvertPriorsFromList(r_pp);
    Eigen::VectorXi include_mu_groups = r_include_mu_groups;
    Eigen::VectorXi include_tau_groups = r_include_tau_groups;

    Derivatives derivatives = GetLogVariationalDensityDerivatives(
        mp_obs, vp, include_mu, include_lambda,
        include_mu_groups, include_tau_groups, calculate_gradient);
    Rcpp::List ret = ConvertDerivativesToList(derivatives);
    return ret;
}


// [[Rcpp::export]]
Rcpp::List GetMCMCLogPriorDerivatives(
    const Rcpp::List draw_list, const Rcpp::List r_pp) {

    int n_draws = draw_list.size();
    PriorParameters<double> pp = ConvertPriorsFromList(r_pp);
    Rcpp::Rcout << "Got " << n_draws << " draws.\n";
    Rcpp::List log_prior_gradients(n_draws);
    for (int draw = 0; draw < n_draws; draw++) {
        Rcpp::List this_draw_list = draw_list[draw];
        MomentParameters<double> mp_draw = ConvertMomentsFromList(this_draw_list);
        Derivatives derivs = GetLogPriorDerivativesFromDraw(mp_draw, pp, true, true, true, true);
        log_prior_gradients[draw] = derivs.grad;
    }
    return log_prior_gradients;
}


// [[Rcpp::export]]
Rcpp::List GetObsLogPriorDerivatives(const Rcpp::List r_obs_mp, const Rcpp::List r_pp,
    bool include_mu, bool include_lambda, bool include_tau) {

    MomentParameters<double> mp_obs = ConvertMomentsFromList(r_obs_mp);
    PriorParameters<double> pp = ConvertPriorsFromList(r_pp);
    Derivatives derivs =
        GetLogPriorDerivativesFromDraw(mp_obs, pp, include_mu, include_lambda, include_tau, true);
    Rcpp::List ret = ConvertDerivativesToList(derivs);
    return ret;
}



//////////////////////
// Covariance

// [[Rcpp::export]]
Eigen::SparseMatrix<double> GetCovariance(const Rcpp::List r_vp) {
        VariationalParameters<double> vp = ConvertParametersFromList(r_vp);
        MomentParameters<double> mp(vp);
        return GetCovariance(vp, mp.offsets);
}



////////////////////////////////////
// For debugging

// [[Rcpp::export]]
double EvaluateLKJPriorVB(
    const Eigen::Map<Eigen::MatrixXd> r_v,
    double n,
    double alpha, double beta, double eta) {

    MatrixXd v = r_v;
    WishartNatural<double> lambda(n, v);

    return EvaluateLKJPrior(lambda, alpha, beta, eta);
}


// [[Rcpp::export]]
double EvaluateLKJPriorDraw(
    const Eigen::Map<Eigen::MatrixXd> r_sigma,
    double log_det_sigma,
    double alpha, double beta, double eta) {

    MatrixXd sigma = r_sigma;
    return EvaluateLKJPrior(sigma, log_det_sigma, alpha, beta, eta);
}


// [[Rcpp::export]]
double student_t_log(double obs, double prior_df,
                     double prior_loc, double prior_scale) {
    return stan::math::student_t_log(obs, prior_df, prior_loc, prior_scale);
}
