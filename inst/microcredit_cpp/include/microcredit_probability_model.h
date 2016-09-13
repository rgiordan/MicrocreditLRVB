
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

/////////////////////////////////
// Priors.  For comparison with MCMC, we must be able to evaluate both
// the variational expected log prior and the prior at a particular MCMC draw.

// This can also be evaulated for a draw of tau..
template <typename T>
T GetGroupPriorLogLikelihood(GammaMoments<T> &mp_tau, GammaNatural<T> const &pp_tau) {
    // Tau is the only group parameter with a prior.
    T log_prior = mp_tau.ExpectedLogLikelihood(pp_tau.alpha, pp_tau.beta);
    return log_prior;
};


template <typename Tlik, typename Tprior>
typename promote_args<Tlik, Tprior>::type GetGroupPriorLogLikelihood(
    VariationalParameters<Tlik> const &vp, PriorParameters<Tprior> const &pp, int g) {

    typedef typename promote_args<Tlik, Tprior>::type T;

    // Tau is the only group parameter with a prior.
    GammaNatural<T> pp_tau(pp.tau);
    GammaNatural<T> vp_tau(vp.tau[g]);
    GammaMoments<T> mp_tau(vp_tau);
    return GetGroupPriorLogLikelihood(mp_tau, pp_tau);
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
        e_log_sigma_diag = log(0.5 * v_inv(k, k)) - e_log_sigma_term;
        e_s = sqrt(0.5 * v_inv(k, k)) * e_s_term;
        e_log_s = 0.5 * e_log_sigma_diag;
        e_log_det_r -= e_log_sigma_diag;
        diag_prior += (lambda_alpha - 1) * e_log_s - lambda_beta * e_s;
    }

    T lkj_prior = (lambda_eta - 1) * e_log_det_r;

    return lkj_prior + diag_prior;
};


// Evaluate the LKJ prior for a draw.
// TODO: put this in libLinearResponseVariationalBayes.cpp
template <typename T> T EvaluateLKJPrior(
    MatrixXT<T> sigma, T log_det_sigma, T alpha, T beta, T eta) {

    // Note: stan::math::lkj_cov_log uses lognormal scales, not inverse
    // gamma scales.

    if (sigma.rows() != sigma.cols()) {
            throw std::runtime_error("sigma must be square.");
    }

    int k_reg = sigma.rows();
    T scale_prior = 0;
    MatrixXT<T> scale_mat = MatrixXT<T>::Zero(k_reg, k_reg);
    for (int k = 0; k < k_reg; k++) {
        if (sigma(k, k) < 0) {
            throw std::runtime_error("sigma must have non-negative diagonals");
        }
        T s = sqrt(sigma(k, k));
        scale_mat(k, k) = 1 / s;

        scale_prior += stan::math::gamma_log(s, alpha, beta);
    }

    MatrixXT<T> sigma_corr = scale_mat * sigma * scale_mat;
    T corr_prior = stan::math::lkj_corr_log(sigma_corr, eta);

    return scale_prior + corr_prior;
};


template <typename T> T EvaluateStudentTPrior(
    MonteCarloNormalParameter mu_draws, T mu_mean, T mu_var,
    T prior_loc, T prior_scale, T prior_df) {

    VectorXT<T> draws = mu_draws.Evaluate(mu_mean, mu_var);
    T e_log_prior = 0;
    for (int ind = 0; ind < draws.size(); ind++) {
        T log_t = stan::math::student_t_log(draws(ind), prior_df, prior_loc, prior_scale);
        e_log_prior += log_t;
    }
    return e_log_prior;
};


template <typename Tlik, typename Tprior>
typename promote_args<Tlik, Tprior>::type  GetGlobalPriorLogLikelihood(
    VariationalParameters<Tlik> const &vp, PriorParameters<Tprior> const &pp) {

    typedef typename promote_args<Tlik, Tprior>::type T;

    T log_prior = 0.0;

    // Mu:
    if (pp.mu_student_t_prior) {
        // Each component has an independent student t prior.
        T pp_mu_t_loc = pp.mu_t_loc;
        T pp_mu_t_scale = pp.mu_t_scale;
        T pp_mu_t_df = pp.mu_t_df;
        for (int k = 0; k < vp.k; k++) {
            T mu_loc = vp.mu.loc(k);
            T mu_var = 1 / vp.mu.info.mat(k, k);
            log_prior += EvaluateStudentTPrior(
                vp.mu_draws, mu_loc, mu_var, pp_mu_t_loc, pp_mu_t_scale, pp_mu_t_df);
        }
    } else {
        MultivariateNormalNatural<T> vp_mu(vp.mu);
        MultivariateNormalMoments<T> vp_mu_moments(vp_mu);
        MultivariateNormalNatural<T> pp_mu = pp.mu;
        log_prior += vp_mu_moments.ExpectedLogLikelihood(pp_mu.loc, pp_mu.info.mat);
    }

    WishartNatural<T> vp_lambda(vp.lambda);
    T lambda_alpha = pp.lambda_alpha;
    T lambda_beta = pp.lambda_beta;
    T lambda_eta = pp.lambda_eta;
    log_prior += EvaluateLKJPrior(vp_lambda, lambda_alpha, lambda_beta, lambda_eta);

    return log_prior;
};


// The global prior for a draw.
template <typename Tlik, typename Tprior>
typename promote_args<Tlik, Tprior>::type  GetGlobalPriorLogLikelihoodDraw(
    MatrixXT<Tlik> const &lambda, VectorXT<Tlik> mu, PriorParameters<Tprior> const &pp) {

    typedef typename promote_args<Tlik, Tprior>::type T;

    T log_prior = 0.0;

    // Mu:
    VectorXT<T> mu_e = mu.template cast<T>();
    MatrixXT<T> mu_outer = mu_e * mu_e.transpose();
    MultivariateNormalMoments<T> mp_mu(mu_e, mu_outer);

    MultivariateNormalNatural<T> pp_mu = pp.mu;
    log_prior += mp_mu.ExpectedLogLikelihood(pp_mu.loc, pp_mu.info.mat);

    // Lambda
    MatrixXT<T> lambda_t = lambda.template cast<T>();
    MatrixXT<T> sigma = lambda_t.inverse();
    T log_det_sigma = log(sigma.determinant());
    T lambda_alpha = pp.lambda_alpha;
    T lambda_beta = pp.lambda_beta;
    T lambda_eta = pp.lambda_eta;
    log_prior += EvaluateLKJPrior(sigma, log_det_sigma,
                                  lambda_alpha, lambda_beta, lambda_eta);
    return log_prior;
};


template <typename Tlik, typename Tprior>
typename promote_args<Tlik, Tprior>::type  GetPriorLogLikelihood(
    VariationalParameters<Tlik> const &vp, PriorParameters<Tprior> const &pp) {

    typedef typename promote_args<Tlik, Tprior>::type T;
    T log_prior = 0.0;
    log_prior += GetGlobalPriorLogLikelihood(vp, pp);

    for (int g = 0; g < vp.n_g; g++) {
            log_prior += GetGroupPriorLogLikelihood(vp, pp, g);
    }
    return log_prior;
};


//////////////////////////////////
// Entropy

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
