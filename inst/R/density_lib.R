

GetMuLogPrior <- function(mu, pp) {
  # You can't use the VB priors because they are
  # (1) a function of the natural parameters whose variance would have to be zero and
  # (2) not normalized.
  dmvnorm(mu, mean=pp$mu_loc, sigma=solve(pp$mu_info), log=TRUE)
}


DrawFromQMu <- function(n_draws, vp_opt, rescale=1) {
  mu_info <- vp_opt$mu_info
  mu_info <- mu_info / (rescale ^ 2)
  return(rmvnorm(n_draws, vp_opt$mu_loc, solve(mu_info)))
}


GetMuLogDensity <- function(mu, vp_opt, draw, pp, unconstrained, calculate_gradient, global_only=TRUE) {
  draw$mu_e_vec <- mu
  draw$mu_e_outer <- mu %*% t(mu)
  q_derivs <- GetLogVariationalDensityDerivatives(
    draw, vp_opt, include_mu=TRUE, include_lambda=FALSE,
    integer(), integer(), unconstrained=unconstrained, global_only=global_only,
    calculate_gradient=calculate_gradient)
  return(q_derivs)
}


GetMuLogMarginalDensity <- function(mu, mu_comp, vp_opt, draw, unconstrained) {
  draw$mu_e_vec[mu_comp] <- mu
  draw$mu_e_outer[mu_comp, mu_comp] <- mu %*% t(mu)
  
  # The component is 0-indexed in C++
  if (mu_comp <= 0) {
    stop("GetMuLogMarginalDensity takes a one-indexed component.")
  }
  q_derivs <- GetVariationalLogMarginalMuDensityDerivatives(
    draw, vp_opt, component=mu_comp - 1, unconstrained=unconstrained)
  return(q_derivs)
}


GetTauLogMarginalDensity <- function(tau, group, vp_opt, mp_draw, unconstrained, calculate_gradient) {
  mp_draw$tau[[group]]$e <- tau
  mp_draw$tau[[group]]$e_log <- log(tau)
  tau_q_derivs <- GetLogVariationalDensityDerivatives(
    mp_draw, vp_opt, include_mu=FALSE, include_lambda=FALSE,
    r_include_tau_groups=as.integer(group - 1), r_include_mu_groups=integer(),
    unconstrained=TRUE, global_only=FALSE, calculate_gradient=calculate_gradient)
  return(tau_q_derivs)
}


GetMuLogStudentTPrior <- function(mu, pp) {
  log_t_prior <- 0
  for (k in 1:length(mu)) {
    log_t_prior <- log_t_prior + student_t_log(mu[k], pp$mu_t_df, pp$mu_t_loc, pp$mu_t_scale)
  }
  return(log_t_prior)
}


GetTauLogPrior <- function(u, pp) {
  return(dgamma(u, pp$tau_alpha, pp$tau_beta))
}




######################################
# Influence function helpers

# Get a function that converts a a draw from mu_k and a standard mvn into a draw from (mu_c | mu_k)
# k is the conditioned component, c is the "complement", i.e. the rest
GetConditionalMVNFunction <- function(k_ind, mvn_mean, mvn_info) {
  mvn_sigma <- solve(mvn_info)
  c_ind <- setdiff(1:length(mvn_mean), k_ind)
  
  # The scale in front of the mu_k for the mean of (mu_c | mu_k)
  # mu_cc_sigma <- mu_sigma[c_ind, c_ind, drop=FALSE]
  mu_kk_sigma <- mvn_sigma[k_ind, k_ind, drop=FALSE]
  mu_ck_sigma <- mvn_sigma[c_ind, k_ind, drop=FALSE]
  sig_cc_corr <- mu_ck_sigma %*% solve(mu_kk_sigma)
  
  # What to multiply by to get Cov(mu_c | mu_k)
  mu_c_cov <- solve(mvn_info[c_ind, c_ind])
  mu_c_scale <- t(chol(mu_c_cov))
  
  # Given u and a draws mu_c_std ~ Standard normal, convert mu_c_std to a draw from MVN( . | mu_k).
  # If there are multiple mu_c_std, each draw should be in its own column.
  GetConditionalDraw <- function(mu_k, mu_c_std) {
    mu_c_mean <- mvn_mean[c_ind] + sig_cc_corr %*% (mu_k - mvn_mean[k_ind, drop=FALSE])
    mu_c_scale %*% mu_c_std + matrix(rep(mu_c_mean, ncol(mu_c_std)), ncol=ncol(mu_c_std))
  }
}


if (FALSE) {
  # Test GetConditionalDraw
  n_draws <- 100000
  mvn_mean <- runif(4)
  mvn_cov <- diag(4) + 0.2
  k_ind <- c(1, 2)
  CondFn <- GetConditionalMVNFunction(k_ind, mvn_mean, solve(mvn_cov))
  mu_k_draws <- rmvnorm(n_draws, mean=mvn_mean[k_ind], sigma=mvn_cov[k_ind, k_ind])
  mu_std_draws <- rmvnorm(n_draws, mean=rep(0, length(mvn_mean) - length(k_ind)))
  mu_c_draws <- matrix(NaN, n_draws, ncol(mu_std_draws))
  for (k in 1:nrow(mu_k_draws)) {
    mu_c_draws[k,] <- CondFn(as.matrix(mu_k_draws[k, ]), as.matrix(mu_std_draws[k, ]))
  }
  
  mu_cond_draws <- matrix(NaN, n_draws, length(mvn_mean))
  mu_cond_draws[, k_ind] <- mu_k_draws
  mu_cond_draws[, setdiff(1:length(mvn_mean), k_ind)] <- mu_c_draws
  
  colMeans(mu_cond_draws)
  mvn_mean
  
  cov(mu_cond_draws)
  mvn_cov
}


# Mu draws:
GetMuImportanceFunctions <- function(mu_comp, vp_opt, pp, lrvb_terms) {
  mp_opt <- GetMoments(vp_opt)
  
  u_mean <- mp_opt$mu_e_vec[mu_comp]
  # Increase the variance for sampling.  How much is enough?
  u_cov <- (1.5 ^ 2) * solve(vp_opt$mu_info)[mu_comp, mu_comp]
  GetULogDensity <- function(u) {
    dnorm(u, mean=u_mean, sd=sqrt(u_cov), log=TRUE)
  }
  
  DrawU <- function(n_samples) {
    rnorm(n_samples, mean=u_mean, sd=sqrt(u_cov))
  }
  
  GetLogPrior <- function(u_vec) {
    sapply(u_vec, function(u) student_t_log(u, pp$mu_t_df, pp$mu_t_loc, pp$mu_t_scale))
  }

  DrawFromPrior <- function(n_samples) {
    rt(n_samples, pp$mu_t_df) * pp$mu_t_scale + pp$mu_t_loc
  }
  
  mu_cov <- solve(vp_opt$mu_info)
  GetLogVariationalDensity <- function(u) {
    return(dnorm(u, mean=vp_opt$mu_loc[mu_comp], sd=sqrt(mu_cov[mu_comp, mu_comp]), log=TRUE))
  }
  
  mp_draw <- mp_opt
  GetFullLogQGradTerm <- function(mu) {
    GetMuLogDensity(mu=mu, vp_opt=vp_opt, draw=mp_draw, pp=pp,
                    unconstrained=TRUE, calculate_gradient=TRUE, global_only=FALSE)$grad
    
  }
  
  lrvb_pre_factor <- -1 * lrvb_terms$jac %*% solve(lrvb_terms$elbo_hess)
  DrawConditionalMu <- GetConditionalMVNFunction(mu_comp, vp_opt$mu_loc, vp_opt$mu_info)
  GetLogQGradResults <- function(u_draws, num_mc_draws, normalize=TRUE) {
    c_ind <- setdiff(1:vp_opt$k, mu_comp)
    std_draws <- rmvnorm(num_mc_draws, mean=rep(0, vp_opt$k_reg - 1))
    
    # Draws from the rest of mu (mu "complement") given u_draws.
    DrawParametersGivenU <- function(u) {
      mu_draw <- matrix(NaN, vp_opt$k_reg, num_mc_draws)
      mu_draw[c_ind, ] <- DrawConditionalMu(u, t(std_draws))
      mu_draw[mu_comp, ] <- u
      return(mu_draw)
    }
    draws_list <- lapply(u_draws, DrawParametersGivenU)
    
    # The dimensions of param_u_draws are (component, mc draw, u draw)
    param_u_draws <- abind(draws_list, along=3)
    
    # The dimensions of lrvb_term_draws work out to be c(moment index, conditional draw, u draw)
    lrvb_term_draws <- apply(param_u_draws, MARGIN=c(2, 3), FUN=GetFullLogQGradTerm)
    lrvb_term_e <- apply(lrvb_term_draws, MARGIN=c(1, 3), FUN=mean)
    
    if (normalize) {
      imp_ratio <- exp(GetLogVariationalDensity(u_draws) - GetULogDensity(u_draws))
      lrvb_term_e_means <- colSums(imp_ratio * t(lrvb_term_e)) / sum(imp_ratio)
      lrvb_term_e <- lrvb_term_e - lrvb_term_e_means
    }
    
    lrvb_terms <- lrvb_pre_factor %*% lrvb_term_e
    
    return(list(lrvb_terms=lrvb_terms, lrvb_term_e=lrvb_term_e, lrvb_term_draws=lrvb_term_draws, param_u_draws=param_u_draws))
  }
  
  GetLogQGradTerms <- function(u_draws, num_mc_draws=20, normalize=TRUE) {
    as.matrix(GetLogQGradResults(u_draws, num_mc_draws, normalize=normalize)$lrvb_terms)
  }
  
  return(list(GetULogDensity=GetULogDensity,
              DrawU=DrawU,
              GetLogPrior=GetLogPrior,
              GetLogVariationalDensity=GetLogVariationalDensity,
              GetLogQGradTerms=GetLogQGradTerms,
              GetLogQGradResults=GetLogQGradResults,
              DrawFromPrior=DrawFromPrior))
}







# Tau draws:
GetTauImportanceFunctions <- function(group, vp_opt, pp, lrvb_terms) {
  mp_opt <- GetMoments(vp_opt)
  
  # Increase the variance for sampling.  How much is enough?
  u_shape <- 0.5 * vp_opt$tau[[group]]$alpha
  u_rate <- 0.5 * vp_opt$tau[[group]]$beta
  
  GetULogDensity <- function(u) {
    dgamma(u, shape=u_shape, rate=u_rate, log=TRUE)
  }
  
  DrawU <- function(n_samples) {
    rgamma(n_samples, shape=u_shape, rate=u_rate)
  }
  
  GetLogPrior <- function(u) {
    sapply(u, function(u) GetTauLogPrior(u, pp))
  }
  
  # This is the univariate density, analytically marginalized in C++
  GetLogVariationalDensitySingleObs <- function(u) {
    mp_draw <- mp_opt
    tau_q_derivs <- GetTauLogMarginalDensity(u, group, vp_opt, mp_draw, TRUE, FALSE)
    return(tau_q_derivs$val) 
  }
  
  GetLogVariationalDensity <- function(u) {
    sapply(u, GetLogVariationalDensitySingleObs)
  }
  
  lrvb_pre_factor <- -1 * lrvb_terms$jac %*% solve(lrvb_terms$elbo_hess)
  GetLogQGradTerm <- function(u) {
    mp_draw <- mp_opt
    tau_q_derivs <- sapply(u, function(u) GetTauLogMarginalDensity(u, group, vp_opt, mp_draw, TRUE, TRUE)$grad )
    as.matrix(lrvb_pre_factor %*% tau_q_derivs)
  }
  
  return(list(GetULogDensity=GetULogDensity,
              DrawU=DrawU,
              GetLogPrior=GetLogPrior,
              GetLogVariationalDensity=GetLogVariationalDensity,
              GetLogQGradTerm=GetLogQGradTerm))
}
