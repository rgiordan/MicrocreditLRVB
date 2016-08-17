
// This is included in microcredit_model.h.

////////////////////////////////////////
// Likelihood functions


template <typename T>
T GetGroupHierarchyLogLikelihood(VariationalParameters<T> const &vp, int g) {
    // TODO: cache the moment calculations.
    WishartMoments<T> lambda_moments(vp.lambda);
    MultivariateNormalMoments<T> mu_moments(vp.mu);
    MultivariateNormalMoments<T> mu_g_moments(vp.mu_g[g]);
    T log_lik = mu_g_moments.ExpectedLogLikelihood(mu_moments, lambda_moments);
    return log_lik;
};


template <typename T>
T GetHierarchyLogLikelihood(VariationalParameters<T> const &vp) {
    T log_lik = 0.0;
    for (int g = 0; g < vp.n_g; g++) {
        log_lik += GetGroupHierarchyLogLikelihood(vp, g);
    }
    return log_lik;
};


template <typename T>
T GetGroupObservationLogLikelihood(
    MicroCreditData const &data, VariationalParameters<T> const &vp, int g) {

    UnivariateNormalMoments<T> y_obs_mean;

    T log_lik = 0.0;
    GammaMoments<T> tau_moments(vp.tau[g]);
    MultivariateNormalMoments<T> mu_g_moments(vp.mu_g[g]);

    for (int n = 0; n < data.n; n++) {
        int row_g = data.y_g(n) - 1; // The group that this observation belongs to.
        if (row_g == g) {
            UnivariateNormalMoments<T> y_obs;
            VectorXT<T> x_row = data.x.row(n).template cast<T>();
            y_obs.e = data.y(n);
            y_obs.e2 = pow(data.y(n), 2);
            y_obs_mean.e = x_row.dot(mu_g_moments.e_vec);
            y_obs_mean.e2 = x_row.dot(mu_g_moments.e_outer.mat * x_row);
            log_lik += y_obs.ExpectedLogLikelihood(y_obs_mean, tau_moments);
        }
    }
    return log_lik;
};


template <typename T>
T GetObservationLogLikelihood(
    MicroCreditData const &data, VariationalParameters<T> const &vp) {
    T log_lik = 0.0;
    for (int g = 0; g < vp.n_g; g++) {
        // TODO: This is a little redundant.
        log_lik += GetGroupObservationLogLikelihood(data, vp, g);
    }
    return log_lik;
};


template <typename Tlik, typename Tprior>
typename promote_args<Tlik, Tprior>::type  GetGroupPriorLogLikelihood(
    VariationalParameters<Tlik> const &vp, PriorParameters<Tprior> const &pp, int g) {

    typedef typename promote_args<Tlik, Tprior>::type T;

    // Tau is the only group parameter with a prior.
    GammaNatural<T> pp_tau(pp.tau);
    GammaNatural<T> vp_tau(vp.tau[g]);
    GammaMoments<T> vp_tau_moments(vp_tau);
    T log_prior = vp_tau_moments.ExpectedLogLikelihood(pp_tau.alpha, pp_tau.beta);
    std::cout << "tau " << g << ": " << log_prior << "\n";

    return log_prior;
};


// TODO: put this in libLinearResponseVariationalBayes.cpp
template <typename T> T EvaluateLKJPrior(
    WishartNatural<T> lambda, T lambda_alpha, T lambda_beta, T lambda_eta) {

    MatrixXT<T> v_inv = lambda.v.mat.inverse();
    WishartMoments<T> lambda_mom(lambda);

    T e_log_sigma_term = digamma(0.5 * (lambda.n - lambda.dim + 1));
    T e_s_term = exp(lgamma(0.5 * (lambda.n - lambda.dim)) -
                 lgamma(0.5 * (lambda.n - lambda.dim + 1)));

    T e_log_s, e_s, e_log_sigma_diag;
    T e_log_det_r = -1 * lambda_mom.e_log_det;
    T diag_prior = 0.0;
    for (int k=0; k < lambda.dim; k++) {
        e_log_sigma_diag =log(0.5 * v_inv(k, k)) - e_log_sigma_term;
        e_s = sqrt(0.5 * v_inv(k, k)) * e_s_term;
        e_log_s = 0.5 * e_log_sigma_diag;
        e_log_det_r -= e_log_sigma_diag;
        diag_prior += (lambda_alpha - 1) * e_log_s -
                       lambda_beta * e_s;
    }

    T lkj_prior = (lambda_eta - 1) * e_log_det_r;

    return lkj_prior + diag_prior;
};


template <typename Tlik, typename Tprior>
typename promote_args<Tlik, Tprior>::type  GetGlobalPriorLogLikelihood(
    VariationalParameters<Tlik> const &vp, PriorParameters<Tprior> const &pp) {

    typedef typename promote_args<Tlik, Tprior>::type T;

    T log_prior = 0.0;

    // Mu:
    MultivariateNormalNatural<T> vp_mu(vp.mu);
    MultivariateNormalMoments<T> vp_mu_moments(vp_mu);
    MultivariateNormalNatural<T> pp_mu = pp.mu;
    log_prior += vp_mu_moments.ExpectedLogLikelihood(pp_mu.loc, pp_mu.info.mat);
    std::cout << "vp.mu: " << log_prior << "\n";

    WishartNatural<T> vp_lambda(vp.lambda);
    T lambda_alpha = pp.lambda_alpha;
    T lambda_beta = pp.lambda_beta;
    T lambda_eta = pp.lambda_eta;
    log_prior += EvaluateLKJPrior(vp_lambda, lambda_alpha, lambda_beta, lambda_eta);
    std::cout << "vp.lambda: " << log_prior << "\n";

    return log_prior;
};


template <typename Tlik, typename Tprior>
typename promote_args<Tlik, Tprior>::type  GetPriorLogLikelihood(
    VariationalParameters<Tlik> const &vp, PriorParameters<Tprior> const &pp) {

    typedef typename promote_args<Tlik, Tprior>::type T;
    T log_prior = 0.0;
    log_prior += GetGlobalPriorLogLikelihood(vp, pp);
    std::cout << "global: " << log_prior << "\n";

    for (int g = 0; g < vp.n_g; g++) {
        log_prior += GetGroupPriorLogLikelihood(vp, pp, g);
        std::cout << "group " << g << ": " << log_prior << "\n";
    }
    return log_prior;
};


template <typename T> T
GetGroupEntropy(VariationalParameters<T> const &vp, int g) {
    T entropy = 0;
    entropy += GetMultivariateNormalEntropy(vp.mu_g[g].info.mat);
    entropy += GetGammaEntropy(vp.tau[g].alpha, vp.tau[g].beta);
    return entropy;
}


template <typename T> T
GetGlobalEntropy(VariationalParameters<T> const &vp) {
    T entropy = 0;
    entropy += GetMultivariateNormalEntropy(vp.mu.info.mat);
    entropy += GetWishartEntropy(vp.lambda.v.mat, vp.lambda.n);
    return entropy;
}


template <typename T> T
GetEntropy(VariationalParameters<T> const &vp) {
    T entropy = 0;
    entropy += GetGlobalEntropy(vp);
    for (int g = 0; g < vp.n_g; g++) {
        entropy += GetGroupEntropy(vp, g);
    }
    return entropy;
}
