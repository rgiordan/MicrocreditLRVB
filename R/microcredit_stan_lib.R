library(Matrix) # Needed for Matrix::diag :(

# Get a mask for the global parameters only.
GlobalMask <- function(vp) {
  mask <- GetVectorFromParameters(vp_base, FALSE)
  mask <- rep(0, length(mask))
  vp_mask <- GetParametersFromVector(vp_base, mask, FALSE)
  vp_mask$mu_loc[] <- 1
  vp_mask$mu_info[] <- 1
  vp_mask$lambda_v[] <- 1
  vp_mask$lambda_n <- 1
  mask <- GetVectorFromParameters(vp_mask, FALSE) == 1
  return(mask)
}


# Generate sample data
SimulateData <- function(true_params, n_g, n_per_group, binary_x=TRUE) {
  y_vec <- list()
  y_g_vec <- list()
  x_vec <- list()
  true_mu_g <- list()
  for (g in 1:n_g) {
    true_mu_g[[g]] <-
      as.numeric(rmvnorm(1, mean=true_params$true_mu,
                         sigma=true_params$true_sigma))
    if (binary_x) {
      x_vec[[g]] <- cbind(rep(1.0, n_per_group), runif(n_per_group) > 0.5)
    } else {
      x_vec[[g]] <- cbind(rep(1.0, n_per_group), runif(n_per_group))
    }
    y_vec[[g]] <-
      rnorm(n_per_group, x_vec[[g]] %*% true_mu_g[[g]],
            1 / sqrt(true_params$true_tau))
    # C++ uses zero indexing
    y_g_vec[[g]] <- rep(g, n_per_group)
  }

  y <- do.call(c, y_vec)
  y_g <- do.call(c, y_g_vec)
  x <- do.call(rbind, x_vec)

  return(list(y=y, y_g=y_g, x=x, true_mu_g=true_mu_g))
}


GetVectorBounds <- function(vp_base, loc_bound=100, info_bound=10, tau_bound=7) {
  # Get sensible extreme bounds, for example for L-BFGS-B
  #info_lower <- diag(vp_base$k_reg) * vp_base$diag_min * (1 + min_bound)
  #info_upper <- diag(vp_base$k_reg) * info_bound
  
  info_lower <- matrix(-1 * info_bound, vp_base$k_reg, vp_base$k_reg)

  vp_nat_lower <- vp_base
  vp_nat_lower$mu_loc[] <- -1 * loc_bound
  vp_nat_lower$mu_info[,] <- info_lower
  vp_nat_lower$lambda_v <- info_lower
  vp_nat_lower$lambda_n <- -1 * info_bound 
  for (g in 1:vp_nat_lower$n_g) {
    vp_nat_lower$mu_g[[g]][["loc"]][] <- -1 * loc_bound
    vp_nat_lower$mu_g[[g]][["info"]] <- info_lower
    vp_nat_lower$tau[[g]][["alpha"]] <- -1 * tau_bound
    vp_nat_lower$tau[[g]][["beta"]] <- -1 * info_bound
  }
  
  theta_lower <- GetVectorFromParameters(vp_nat_lower, FALSE)
  theta_upper <- -1 * theta_lower
  
  return(list(theta_lower=theta_lower, theta_upper=theta_upper))  
}


GetOptimFunctions <- function(x, y, y_g, vp_base, pp,
                              mask=rep(TRUE, vp_base$encoded_size),
                              DerivFun=GetElboDerivatives) {
  GetOptimFunctionsBase(x, y, y_g, vp_base, pp, mask,
                        DerivFun, GetVectorFromParameters, GetParametersFromVector)
}


GetGlobalOptimFunctions <- function(x, y, y_g, vp_base, pp,
                                    mask=rep(TRUE, length(GetGlobalVectorFromParameters(vp_base))),
                                    DerivFun=GetElboDerivatives) {
  GetOptimFunctionsBase(x, y, y_g, vp_base, pp, mask,
                        DerivFun, GetGlobalVectorFromParameters, GetParametersFromGlobalVector)
}


GetOptimFunctionsBase <- function(x, y, y_g, vp_base, pp, mask,
                                  DerivFun, VectorFromParameters, ParametersFromVector) {

  theta_base <- VectorFromParameters(vp_base, TRUE)
  
  GetLocalVP <- function(theta) {
    full_theta <- theta_base
    full_theta[mask] <- theta
    return(ParametersFromVector(vp_base, full_theta, TRUE))
  }
  
  OptimVal <- function(theta) {
    ret <- DerivFun(x, y, y_g, GetLocalVP(theta), pp,
                calculate_gradient=FALSE, calculate_hessian=FALSE,
                unconstrained=TRUE)
    cat(ret$val, "\n")
    ret$val
  }
  
  OptimGrad <- function(theta) {
    ret <- DerivFun(x, y, y_g, GetLocalVP(theta), pp,
                calculate_gradient=TRUE, calculate_hessian=FALSE,
                unconstrained=TRUE)
    ret$grad[mask]
  }
  
  OptimHess <- function(theta) {
    ret <- DerivFun(x, y, y_g, GetLocalVP(theta), pp,
                calculate_gradient=TRUE, calculate_hessian=TRUE,
                unconstrained=TRUE)
    ret$hess[mask, mask]
  }
  return(list(OptimVal=OptimVal, OptimGrad=OptimGrad, OptimHess=OptimHess))
}


GetTrustRegionELBO <- function(x, y, y_g, vp_base, pp,
                               verbose=FALSE, mask=rep(TRUE, vp_base$encoded_size)) {
  
  theta_base <- GetVectorFromParameters(vp_base, TRUE)

  TrustFun <- function(theta) {
    full_theta <- theta_base
    full_theta[mask] <- theta
    this_vp <- GetParametersFromVector(vp_base, full_theta, TRUE)
    ret <- GetElboDerivatives(x, y, y_g, this_vp, pp,
                              calculate_gradient=TRUE, calculate_hessian=FALSE,
                              unconstrained=TRUE)
    sparse_hess <- GetSparseELBOHessian(x, y, y_g, this_vp, pp, TRUE)
    hess <- as.matrix(sparse_hess[mask, mask])

    if (verbose) {
      cat("Value: ", ret$val, "\n")
    }    
    list(value=ret$val, gradient=ret$grad[mask], hessian=hess)
  }
  
  return(list(TrustFun=TrustFun, mask=mask, theta_base=theta_base, theta_init=theta_base[mask]))
}


InitializeVariationalParameters <-
  function(x, y, y_g, mu_diag_min=0.01, lambda_diag_min=1e-5, tau_min=0.5, lambda_n_min=1e-6) {
  # Initial parameters from data
  vp <- GetEmptyVariationalParameters(ncol(x), max(y_g))
  vp$mu_diag_min <- mu_diag_min
  vp$lambda_diag_min <- lambda_diag_min
  vp$tau_alpha_min <- vp$tau_beta_min <- tau_min
  vp$lambda_n_min <- lambda_n_min
    
  vp$mu_loc <- summary(lm(y ~ x - 1))$coefficients[,"Estimate"]
  mu_cov <- vp$mu_loc %*% t(vp$mu_loc) + 10 * diag(vp$k)
  vp$mu_info <- solve(mu_cov) + diag(vp$k_reg) * mu_diag_min
  mu_g_mat <- matrix(NaN, vp$n_g, vp$k)
  for (g in 1:vp$n_g) {
    stopifnot(sum(y_g == g) >= 1)
    g_reg <- summary(lm(y ~ x - 1, subset=y_g == g))
    mu_g_mean <- g_reg$coefficients[,"Estimate"]
    mu_g_cov <- mu_g_mean %*% t(mu_g_mean) + 10 * diag(vp$k)
    vp$mu_g[[g]] <- list(loc=mu_g_mean, info=solve(mu_g_cov) + diag(vp$k_reg) * mu_diag_min)
    vp$tau[[g]] <- list(alpha=1 + tau_min, beta=g_reg$sigma ^ 2 + tau_min)
    mu_g_mat[g, ] <- mu_g_mean
  }
  vp$lambda_n <- vp$k + 1
  vp$lambda_v <- solve(cov(mu_g_mat)) / vp$lambda_n + diag(vp$k_reg) * lambda_diag_min

  return(vp)
}


InitializeZeroVariationalParameters <-
  function(x, y, y_g, mu_diag_min=1e-6, lambda_diag_min=1e-6, tau_min=1e-6, lambda_n_min=1e-6) {

  # Initial parameters from data
  vp <- GetEmptyVariationalParameters(ncol(x), max(y_g))
  vp$tau_alpha_min <- vp$tau_beta_min <- tau_min
  min_info <- diag(vp$k_reg)
  
  vp$mu_loc <- rep(0, vp$k_reg)
  vp$mu_info <- diag(vp$k_reg) + min_info
  vp$mu_diag_min <- mu_diag_min
  for (g in 1:vp$n_g) {
    stopifnot(sum(y_g == g) >= 1)
    vp$mu_g[[g]] <- list(loc=rep(0, vp$k_reg), info=diag(vp$k_reg) + min_info)
    vp$tau[[g]] <- list(alpha=1  + tau_min, beta=1 + tau_min)
  }
  vp$lambda_n <- vp$k + 1
  vp$lambda_v <- diag(vp$k_reg) + min_info
  vp$lambda_n_min <- 0.5
  
  return(vp)
}



#################################################
# Optimization:

FitVariationalModel <- function(x, y, y_g, vp_reg, pp) {
  # first BFGS
  mask <- rep(TRUE, vp_reg$encoded_size)
  bfgs_opt_fns <- GetOptimFunctions(x, y, y_g, vp_reg, pp, DerivFun=GetElboDerivatives, mask=mask)
  theta_init <- GetVectorFromParameters(vp_reg, TRUE)
  bounds <- GetVectorBounds(vp_reg, loc_bound=60, info_bound=10, tau_bound=100)
  
  bfgs_time <- Sys.time()
  bfgs_result <- optim(theta_init[mask],
                       bfgs_opt_fns$OptimVal, bfgs_opt_fns$OptimGrad,
                       method="L-BFGS-B", lower=bounds$theta_lower[mask], upper=bounds$theta_upper[mask],
                       control=list(fnscale=-1, maxit=1000, trace=0, factr=1e7))
  stopifnot(bfgs_result$convergence == 0)
  print(bfgs_result$message)
  bfgs_time <- Sys.time() - bfgs_time
  vp_bfgs <- GetParametersFromVector(vp_reg, bfgs_result$par, TRUE)
  
  # then trust region
  tr_time <- Sys.time()
  trust_fns <- GetTrustRegionELBO(x, y, y_g, vp_bfgs, pp, verbose=TRUE)
  trust_result <- trust(trust_fns$TrustFun, trust_fns$theta_init,
                        rinit=1, rmax=100, minimize=FALSE, blather=TRUE, iterlim=50)
  tr_time <- Sys.time() - tr_time
  trust_result$converged
  trust_result$value
  
  bfgs_time + tr_time
  
  vp_opt <- GetParametersFromVector(vp_reg, trust_result$argument, TRUE)
  
  return(list(bfgs_time=bfgs_time, tr_time=tr_time,
              bfgs_result=bfgs_result, trust_result=trust_result,
              vp_opt=vp_opt))
}


GetLRVB <- function(x, y, y_g, vp_opt, pp) {
  unconstrained <- TRUE # This seems to have a better condition number. 
  moment_derivs <- GetMomentJacobian(vp_opt, unconstrained)
  jac <- Matrix(moment_derivs$hess)
  elbo_hess <- GetSparseELBOHessian(x, y, y_g, vp_opt, pp, unconstrained)
  lrvb_cov <- -1 * jac %*% Matrix::solve(elbo_hess, Matrix::t(jac))
  stopifnot(min(diag(lrvb_cov)) > 0)
  
  return(list(lrvb_cov=lrvb_cov, jac=jac, elbo_hess=elbo_hess))
  
}

# Sensitivity
GetSensitivity <- function(vp_opt, pp, jac, elbo_hess) {
  # Get some indices
  comb_indices <- GetPriorsAndParametersFromVector(
    vp_opt, pp, as.numeric(1:(vp_opt$encoded_size + pp$encoded_size)))
  comb_prior_ind <- GetVectorFromPriors(comb_indices$pp)
  comb_vp_ind <- GetVectorFromParameters(comb_indices$vp, FALSE)
  
  log_prior_derivs <- GetLogPriorDerivatives(vp_opt, pp, TRUE, TRUE, TRUE)
  log_prior_param_prior <- Matrix(log_prior_derivs$hess[comb_vp_ind, comb_prior_ind])
  
  prior_sens <- -1 * jac %*% Matrix::solve(elbo_hess, log_prior_param_prior)
  return(prior_sens)
}


###################################
# Pack MCMC draws into a list of moment parameters to interact with C++.
# mp_reg provides only a template for the moment parameters.

PackMCMCSamplesIntoMoments <- function(mcmc_sample, mp_reg, n_draws=dim(mcmc_sample$mu)[1]) {
  mp_draws <- list()
  zero_mp <- GetMomentsFromVector(mp_reg, rep(NaN, mp_reg$encoded_size))
  
  draw <- 1
  for (draw in 1:n_draws) {
    if (draw %% 100 == 0) {
      cat("Writing draw ", draw, " of ", n_draws, "(", 100 * draw / n_draws, "%)\n")
    }
    mp_draw <- zero_mp
    
    mu <- mcmc_sample$mu[draw, ]
    lambda <- mcmc_sample$lambda[draw, , ]
    tau <- 1 / mcmc_sample$sigma_y[draw, ]^2
    mu_g <- mcmc_sample$mu1[draw, , ]
    
    mp_draw$mu_e_vec <- mu
    mp_draw$mu_e_outer <- mu %*% t(mu)
    
    mp_draw$lambda_e <- lambda
    mp_draw$lambda_e_log_det <- log(det(lambda))
    
    for (g in 1:(vp_reg$n_g)) {
      mp_draw$mu_g[[g]]$e_vec <- mu_g[g, ]
      mp_draw$mu_g[[g]]$e_outer <- mu_g[g, ] %*% t(mu_g[g, ])
      mp_draw$tau[[g]]$e <- tau[g]
      mp_draw$tau[[g]]$e_log <- log(tau[g])
    }
    
    GetVectorFromMoments(mp_draw)
    
    mp_draws[[draw]] <- mp_draw
  }

  return(mp_draws)
}




#################################################
# Formatting:

# Keep the formatting standard.
ResultRow <- function(par, component, group, method, metric, val) {
  return(data.frame(par=par, component=component, group=group, method=method, metric=metric, val=val))
}

SummarizeMCMCColumn <- function(draws, par, component=-1, group=-1, method="mcmc") {
  rbind(ResultRow(par, component, group, method=method, metric="mean", val=mean(draws)),
        ResultRow(par, component, group, method=method, metric="sd", val=sd(draws)))
}

# The Accessor function should take a vb list and return the appropriate component.
SummarizeVBVariable <- function(vp_mom, mfvb_sd, lrvb_sd, Accessor, par, component=-1, group=-1) {
  rbind(ResultRow(par, component, group, method="mfvb", metric="mean", val=Accessor(vp_mom)),
        ResultRow(par, component, group, method="mfvb", metric="sd", val=Accessor(mfvb_sd)),
        ResultRow(par, component, group, method="lrvb", metric="sd", val=Accessor(lrvb_sd)))
}


SummarizeNaturalParameters <- function(vp_nat) {
  vp_mom <- GetMoments(vp_nat)
  mfvb_cov <- GetCovariance(vp_nat)
  mfvb_sd <- GetMomentsFromVector(vp_mom, sqrt(diag(mfvb_cov)))
  lrvb_sd <- GetMomentsFromVector(vp_mom, rep(0, vp_nat$encoded_size))
  return(SummarizeMomentParameters(vp_mom, mfvb_sd, lrvb_sd))  
}


SummarizeRawMomentParameters <- function(vp_mom, metric, method) {
  k_reg <- vp_mom$k_reg
  n_g <- vp_mom$n_g
  
  results_list <- list()
  for (k in 1:k_reg) {
    Accessor <- function(vp) { vp[["mu_e_vec"]][k] }
    results_list[[length(results_list) + 1]] <-
      ResultRow(par="mu", component=k, group=-1,
                method=method, metric=metric, val=Accessor(vp_mom))
  }
  
  for (k1 in 1:k_reg) {
    for (k2 in 1:k_reg) {
      Accessor <- function(vp) { vp[["lambda_e"]][k1, k2] }
      component <- paste(k1, k2, sep="_")
      results_list[[length(results_list) + 1]] <-
        ResultRow(par="lambda", component=component, group=-1,
                  method=method, metric=metric, val=Accessor(vp_mom))
      }
  }
  
  for (g in 1:n_g) {
    for (k in 1:k_reg) {
      Accessor <- function(vp) { vp[["mu_g"]][[g]][["e_vec"]][k] }
      results_list[[length(results_list) + 1]] <-
        ResultRow(par="mu_g", component=k, group=g,
                  method=method, metric=metric, val=Accessor(vp_mom))
    }
  }
  
  for (g in 1:n_g) {
    Accessor <- function(vp) { vp[["tau"]][[g]][["e"]] }
    results_list[[length(results_list) + 1]] <-
      ResultRow(par="tau", component=-1, group=g,
                method=method, metric=metric, val=Accessor(vp_mom))
  }
  
  return(do.call(rbind, results_list))
}


SummarizeMomentParameters <- function(vp_mom, mfvb_sd, lrvb_sd) {
  rbind(SummarizeRawMomentParameters(vp_mom, metric="mean", method="mfvb"),
        SummarizeRawMomentParameters(mfvb_sd, metric="sd", method="mfvb"),
        SummarizeRawMomentParameters(lrvb_sd, metric="sd", method="lrvb"))
}


SummarizeMCMCResults <- function(mcmc_sample) {
  results_list <- list()
  
  k_reg <- ncol(mcmc_sample$mu)
  n_g <- ncol(mcmc_sample$mu1)
  
  for (k in 1:k_reg) {
    results_list[[length(results_list) + 1]] <-
        SummarizeMCMCColumn(mcmc_sample$mu[, k], par="mu", component=k)
  }
  
  for (k1 in 1:k_reg) {
    for (k2 in 1:k_reg) {
      component <- paste(k1, k2, sep="_")
      results_list[[length(results_list) + 1]] <-
        SummarizeMCMCColumn(mcmc_sample$lambda_mu[, k1, k2], par="lambda", component=component)
    }
  }

  for (g in 1:n_g) {
    tau_draws <- 1 / mcmc_sample$sigma_y[, g] ^ 2
    results_list[[length(results_list) + 1]] <-
      SummarizeMCMCColumn(tau_draws, par="tau", group=g)
    for (k in 1:k_reg) {
      results_list[[length(results_list) + 1]] <-
        SummarizeMCMCColumn(mcmc_sample$mu1[, g, k], par="mu_g", component=k, group=g)
    }
  }
  
  return(do.call(rbind, results_list))
}




#############
# Debugging

GetLambdaMask <- function(vp_base) {
  mask <- GetGlobalVectorFromParameters(vp_base, FALSE)
  mask <- rep(0, length(mask))
  vp_mask <- GetParametersFromGlobalVector(vp_base, mask, FALSE)
  # vp_mask$mu_loc[] <- 1
  # vp_mask$mu_info[] <- 1
  vp_mask$lambda_v[] <- 1
  vp_mask$lambda_n <- 1
  mask <- GetGlobalVectorFromParameters(vp_mask, FALSE) == 1
  return(mask)
}


GetLocalMask <- function(vp_base) {
  mask <- GetVectorFromParameters(vp_base, FALSE)
  mask <- rep(1, length(mask))
  vp_mask <- GetParametersFromVector(vp_base, mask, FALSE)
  vp_mask$mu_loc[] <- 0
  vp_mask$mu_info[] <- 0
  vp_mask$lambda_v[] <- 0
  vp_mask$lambda_n <- 0
  mask <- GetVectorFromParameters(vp_mask, FALSE) == 1
  return(mask)
}



GetTrustObj <- function(optim_fns) {
  TrustObj <- function(theta) {
    list(value=optim_fns$OptimVal(theta),
         gradient=optim_fns$OptimGrad(theta),
         hessian=optim_fns$OptimHess(theta))
  }
  return(TrustObj)
}


EvaluateOnGrid <- function(FUN, theta, dir, grid_min, grid_max, len) {
  result <- list()
  grid_points <- seq(grid_min, grid_max, length.out=len)
  for (i in 1:len) {
    epsilon <- grid_points[i]
    print(epsilon)
    val <- FUN(theta + dir * epsilon)
    result[[length(result) + 1]] <- data.frame(epsilon=epsilon, val=val)
  }
  return(do.call(rbind, result))
}


EntropyFun <- function(x, y, y_g, vp, pp) {
  GetCustomElboDerivatives(x, y, y_g, vp, pp,
                           include_obs=FALSE,
                           include_hier=FALSE,
                           include_prior=FALSE,
                           include_entropy=TRUE,
                           global_only=FALSE,
                           calculate_gradient=FALSE,
                           calculate_hessian=FALSE,
                           unconstrained=TRUE)$val
}


HierFun <- function(x, y, y_g, vp, pp) {
  GetCustomElboDerivatives(x, y, y_g, vp, pp,
                           include_obs=FALSE,
                           include_hier=TRUE,
                           include_prior=FALSE,
                           include_entropy=FALSE,
                           global_only=FALSE,
                           calculate_gradient=FALSE,
                           calculate_hessian=FALSE,
                           unconstrained=TRUE)$val
}

ObsFun <- function(x, y, y_g, vp, pp) {
  GetCustomElboDerivatives(x, y, y_g, vp, pp,
                           include_obs=TRUE,
                           include_hier=FALSE,
                           include_prior=FALSE,
                           include_entropy=FALSE,
                           global_only=FALSE,
                           calculate_gradient=FALSE,
                           calculate_hessian=FALSE,
                           unconstrained=TRUE)$val
}

PriorFun <- function(x, y, y_g, vp, pp) {
  GetCustomElboDerivatives(x, y, y_g, vp, pp,
                           include_obs=FALSE,
                           include_hier=FALSE,
                           include_prior=TRUE,
                           include_entropy=FALSE,
                           global_only=FALSE,
                           calculate_gradient=FALSE,
                           calculate_hessian=FALSE,
                           unconstrained=TRUE)$val
}

EntropyGrad <- function(x, y, y_g, vp, pp) {
  GetCustomElboDerivatives(x, y, y_g, vp, pp,
                           include_obs=FALSE,
                           include_hier=FALSE,
                           include_prior=FALSE,
                           include_entropy=TRUE,
                           global_only=FALSE,
                           calculate_gradient=TRUE,
                           calculate_hessian=FALSE,
                           unconstrained=TRUE)$grad
}


HierGrad <- function(x, y, y_g, vp, pp) {
  GetCustomElboDerivatives(x, y, y_g, vp, pp,
                           include_obs=FALSE,
                           include_hier=TRUE,
                           include_prior=FALSE,
                           include_entropy=FALSE,
                           global_only=FALSE,
                           calculate_gradient=TRUE,
                           calculate_hessian=FALSE,
                           unconstrained=TRUE)$grad
}

ObsGrad <- function(x, y, y_g, vp, pp) {
  GetCustomElboDerivatives(x, y, y_g, vp, pp,
                           include_obs=TRUE,
                           include_hier=FALSE,
                           include_prior=FALSE,
                           include_entropy=FALSE,
                           global_only=FALSE,
                           calculate_gradient=TRUE,
                           calculate_hessian=FALSE,
                           unconstrained=TRUE)$grad
}

PriorGrad <- function(x, y, y_g, vp, pp) {
  GetCustomElboDerivatives(x, y, y_g, vp, pp,
                           include_obs=FALSE,
                           include_hier=FALSE,
                           include_prior=TRUE,
                           include_entropy=FALSE,
                           global_only=FALSE,
                           calculate_gradient=TRUE,
                           calculate_hessian=FALSE,
                           unconstrained=TRUE)$grad
}
