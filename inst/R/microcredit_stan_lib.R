library(Matrix) # Needed for Matrix::diag :(


# Generate sample data
SimulateData <- function(true_params, n_g, n_per_group) {
  y_vec <- list()
  y_g_vec <- list()
  x_vec <- list()
  true_mu_g <- list()
  for (g in 1:n_g) {
    true_mu_g[[g]] <-
      as.numeric(rmvnorm(1, mean=true_params$true_mu,
                         sigma=true_params$true_sigma))
    x_vec[[g]] <- cbind(rep(1.0, n_per_group), runif(n_per_group) > 0.5)
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


GetVectorBounds <- function(vp_base, loc_bound=100, info_bound=10) {
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
    vp_nat_lower$tau[[g]][["alpha"]] <- -1 * info_bound
    vp_nat_lower$tau[[g]][["beta"]] <- -1 * info_bound
  }
  
  theta_lower <- GetVectorFromParameters(vp_nat_lower, FALSE)
  theta_upper <- -1 * theta_lower
  
  return(list(theta_lower=theta_lower, theta_upper=theta_upper))  
}


SummarizeVpNat <- function(vp_nat) {
  print("Mu:")
  print(vp_nat$mu_loc)
  print(vp_nat$mu_info)
  print("Lambda:")
  print(vp_nat$lambda_v)
  print(vp_nat$lambda_n)
  print("Mu_g[1]:")
  print(vp_nat$mu_g[[1]])
  print("Tau[1]:")
  print(vp_nat$tau[[1]])
}


GetOptimFunctions <- function(x, y, y_g, vp_base, pp,
                              DerivFun=GetElboDerivatives,
                              mask=rep(TRUE, length(GetVectorFromParameters(vp_base, TRUE)))) {
  theta_base <- GetVectorFromParameters(vp_base, TRUE)

  GetLocalVP <- function(theta) {
    full_theta <- theta_base
    full_theta[mask] <- theta
    return(GetParametersFromVector(vp_base, full_theta, TRUE))
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


InitializeVariationalParameters <- function(x, y, y_g, diag_min=1, tau_min=1) {
  # Initial parameters from data
  vp <- GetEmptyVariationalParameters(ncol(x), max(y_g))
  vp$diag_min <- diag_min
  vp$tau_alpha_min <- vp$tau_beta_min <- tau_min
  min_info <- diag(vp$k_reg) * diag_min
  
  vp$mu_loc <- summary(lm(y ~ x - 1))$coefficients[,"Estimate"]
  mu_cov <- vp$mu_loc %*% t(vp$mu_loc) + 10 * diag(vp$k)
  vp$mu_info <- solve(mu_cov) + min_info
  mu_g_mat <- matrix(NaN, vp$n_g, vp$k)
  for (g in 1:vp$n_g) {
    stopifnot(sum(y_g == g) > 1)
    g_reg <- summary(lm(y ~ x - 1, subset=y_g == g))
    mu_g_mean <- g_reg$coefficients[,"Estimate"]
    mu_g_cov <- mu_g_mean %*% t(mu_g_mean) + 10 * diag(vp$k)
    vp$mu_g[[g]] <- list(loc=mu_g_mean, info=solve(mu_g_cov) + min_info)

    e_tau <- 1 / g_reg$sigma ^ 2
    vp$tau[[g]] <- list(alpha=0.01 * e_tau + tau_min, beta=0.01 + tau_min)
    mu_g_mat[g, ] <- mu_g_mean
  }
  vp$lambda_n <- vp$k + 1
  vp$lambda_v <- solve(cov(mu_g_mat)) / vp$lambda_n + min_info

  return(vp)
}


InitializeZeroVariationalParameters <- function(x, y, y_g, diag_min=1, tau_min=1) {
  # Initial parameters from data
  vp <- GetEmptyVariationalParameters(ncol(x), max(y_g))
  vp$diag_min <- diag_min
  vp$tau_alpha_min <- vp$tau_beta_min <- tau_min
  min_info <- diag(vp$k_reg) * diag_min
  
  vp$mu_loc <- rep(0, vp$k_reg)
  vp$mu_info <- diag(vp$k_reg) + min_info
  for (g in 1:vp$n_g) {
    stopifnot(sum(y_g == g) > 1)
    vp$mu_g[[g]] <- list(loc=rep(0, vp$k_reg), info=diag(vp$k_reg) + min_info)
    vp$tau[[g]] <- list(alpha=1  + tau_min, beta=1 + tau_min)
  }
  vp$lambda_n <- vp$k + 1
  vp$lambda_v <- diag(vp$k_reg) + min_info
  
  return(vp)
}


#################################################
# Fitting:



MVNMeansFromGradient <- function(e_mu_grad, e_mu2_grad) {
  result <- list()
  k <- length(e_mu_grad)
  stopifnot(dim(e_mu2_grad) == c(k, k))
  # off diagonals * 0.5 because of double counting
  scale_matrix <- diag(0.5, k) + matrix(0.5, k, k)
  result$precision_mu <- -2 * scale_matrix * e_mu2_grad
  stopifnot(min(eigen(result$precision_mu)$values) > 0)
  result$e_mu <- solve(result$precision_mu, e_mu_grad)
  result$cov_mu <- solve(result$precision_mu)
  result$e_mu2 <- result$cov_mu + result$e_mu %*% t(result$e_mu)
  return(result)
}

GammaMeansFromGradient <- function(e_tau_grad, e_log_tau_grad) {
  result <- list()
  result$tau_alpha <- e_log_tau_grad + 1
  result$tau_beta <- -1 * e_tau_grad

  stopifnot(result$tau_alpha >= 0)
  stopifnot(result$tau_beta >= 0)

  result$e_tau <- result$tau_alpha / result$tau_beta
  result$e_log_tau <- digamma(result$tau_alpha) - log(result$tau_beta)
  return(result)
}


OptimLambdaDerivs <- function(lambda_theta, x, y, y_g, vp, pp, calculate_grad, calculate_hess) {
  vp_decode <- DecodeLambda(lambda_theta, vp$k, vp$n_g, pp$lambda_diag_min, pp$lambda_n_min)
  vp$lambda_v_par <- vp_decode$lambda_v_par
  vp$lambda_n_par <- vp_decode$lambda_n_par
  # cat("n: ", vp_decode$lambda_n_par, ", ", lambda_theta[4], "\n")
  return(LambdaGradient(x, y, y_g, vp, pp, TRUE))
}


OptimizeLambdaSubproblem <- function(x, y, y_g, vp, pp, factr=1e7, itnmax=100) {
  k <- vp$k
  n_g <- vp$n_g

  TrustRegionLambdaFunction <- function(lambda_theta) {
    lambda_derivs <- OptimLambdaDerivs(lambda_theta, x, y, y_g, vp, pp, TRUE, TRUE)
    return(list(value=lambda_derivs$elbo,
                gradient=lambda_derivs$lambda_grad,
                hessian=lambda_derivs$lambda_hess))
  }

  theta_init <- EncodeLambda(vp, k, n_g, pp$lambda_diag_min, pp$lambda_n_min)
  lambda_opt <- trust::trust(TrustRegionLambdaFunction, theta_init, rinit=10, rmax=1e10, minimize=FALSE)
  lambda_update <- DecodeLambda(lambda_opt$argument, k, n_g, pp$lambda_diag_min, pp$lambda_n_min)
  lambda_update$lambda_opt <- lambda_opt
  return(lambda_update)
}


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
  k_reg <- vp_nat$k_reg
  n_g <- vp_nat$n_g
  
  results_list <- list()
  for (k in 1:k_reg) {
    Accessor <- function(vp) { vp[["mu_loc"]][k] }
    results_list[[length(results_list) + 1]] <-
      ResultRow(par="mu", component=k, group=-1, method="mfvb", metric="mean", val=Accessor(vp_nat))
  }

  for (g in 1:n_g) {
    for (k in 1:k_reg) {
      Accessor <- function(vp) { vp[["mu_g"]][[g]][["loc"]][k] }
      results_list[[length(results_list) + 1]] <-
        ResultRow(par="mu_g", component=k, group=g, method="mfvb", metric="mean", val=Accessor(vp_nat))
    }
  }
  
  do.call(rbind, results_list)
}


SummarizeVbResults <- function(vp_mom, mfvb_sd, lrvb_sd) {
  
  k_reg <- vp_mom$k_reg
  n_groups <- vp_mom$n_groups
  
  results_list <- list()
  
  # VB results
  results_list[[length(results_list) + 1]] <-
    SummarizeVBVariable(vp_mom, mfvb_sd, lrvb_sd, function(vp) { vp[["mu_loc"]] }, par="mu")
  results_list[[length(results_list) + 1]] <-
    SummarizeVBVariable(vp_mom, mfvb_sd, lrvb_sd, function(vp) { vp[["tau_e"]] }, par="tau")
  for (k in 1:k_reg) {
    results_list[[length(results_list) + 1]] <-
      SummarizeVBVariable(vp_mom, mfvb_sd, lrvb_sd, function(vp) { vp[["beta_e_vec"]][k] },
                          par="beta", component=k)
  }
  for (g in 1:n_groups) {
    results_list[[length(results_list) + 1]] <-
      SummarizeVBVariable(vp_mom, mfvb_sd, lrvb_sd, function(vp) { vp[["u_vec"]][[g]][["u_e"]] },
                          par="u", group=g)
  }
  
  return(do.call(rbind, results_list))
}









######################### Old

FitModel <- function(x, y, y_g, vp, pp, num_iters=100, fit_lambda=TRUE, rel_tol=1e-8, verbose=FALSE) {
  n_g <- vp$n_g
  # pb <- txtProgressBar(0, num_iters, style=3)
  steps_list <- list()
  model_grads <- ModelGradient(x, y, y_g, vp, pp, FALSE, TRUE)
  last_theta <- model_grads$theta
  theta_rel_diff <- Inf
  iter <- 1
  model_grads <- ModelGradient(x, y, y_g, vp, pp, FALSE, TRUE)
  last_theta <- model_grads$theta
  while (iter <= num_iters && theta_rel_diff > rel_tol) {
    # setTxtProgressBar(pb, iter)

    # At this point model_grads was executed at the end of the loop (or before
    # the start of the loop) to check for convergence
    for (g in 1:n_g) {
      mu_g_update <- MVNMeansFromGradient(model_grads$e_mu_g[[g]],
                                          model_grads$e_mu2_g_vec[[g]])
      vp$e_mu_g[[g]] <- mu_g_update$e_mu
      vp$e_mu2_g_vec[[g]] <- mu_g_update$e_mu2
    }

    model_grads <- ModelGradient(x, y, y_g, vp, pp, FALSE, TRUE)
    mu_update <- MVNMeansFromGradient(model_grads$e_mu, model_grads$e_mu2)
    vp$e_mu <- mu_update$e_mu
    vp$e_mu2 <- mu_update$e_mu2

    model_grads <- ModelGradient(x, y, y_g, vp, pp, FALSE, TRUE)
    for (g in 1:n_g) {
      tau_update <- GammaMeansFromGradient(
        model_grads$e_tau[[g]], model_grads$e_log_tau[[g]])
      vp$e_tau[[g]] <- tau_update$e_tau
      vp$e_log_tau[[g]] <- tau_update$e_log_tau

      # Alpha and beta are used for the covariance and are easiset to get from
      # the variational updates
      vp$tau_alpha_vec[[g]]  <- tau_update$tau_alpha
      vp$tau_beta_vec[[g]]  <- tau_update$tau_beta
    }

    # Update lambda last to make extra sure the gradients are zero for the coordinate
    # transformation.
    if (fit_lambda) {
      lambda_update <- OptimizeLambdaSubproblem(x, y, y_g, vp, pp, itnmax=10, factr=factr)
      vp$lambda_v_par <- lambda_update$lambda_v_par
      vp$lambda_n_par <- lambda_update$lambda_n_par
    }

    iter <- iter + 1
    model_grads <- ModelGradient(x, y, y_g, vp, pp, FALSE, TRUE)
    theta_rel_diff <- max(abs(last_theta - model_grads$theta) / abs(model_grads$theta + rel_tol))
    last_theta <- model_grads$theta
    steps_list[[iter]] <- list(vp=vp, theta=model_grads$theta,
                               theta_rel_diff=theta_rel_diff,
                               log_lik=model_grads$log_lik,
                               iter=iter)
    if (verbose) {
      cat("Iter: ", iter, ": theta relative diff: ", theta_rel_diff, "\n")
    }
  }
  # close(pb)

  result_list <- list()
  result_list$vp <- vp
  result_list$lambda_update <- lambda_update
  result_list$steps_list <- steps_list
  return(result_list)
}








############################################
# Processing results

GetVBResultRow <- function(param_name, component, value, metric, method="mfvb", group=-1) {
  data.frame(param=param_name, component=component,
             value=value, metric=metric, method=method, group=group)
}


ConvertParameterListToDataframe <- function(vp, metric) {
  result <- data.frame()
  for (k in 1:vp$k) {
    result <- rbind(result, GetVBResultRow("mu", k, value=vp$e_mu[k], metric=metric))
  }

  for (g in 1:vp$n_g) { for (k in 1:vp$k) {
    result <- rbind(result, GetVBResultRow("mu_g", k, value=vp$e_mu_g[[g]][k],
                                           metric=metric, group=g))
  }}

  for (g in 1:vp$n_g) {
    result <- rbind(result, GetVBResultRow("log_tau", 1, value=vp$e_log_tau[[g]],
                                           metric=metric, group=g))
  }

  for (k1 in 1:vp$k) { for (k2 in 1:k1) {
    component <- k1 + 10 * k2
    result <- rbind(result, GetVBResultRow("lambda_v_par", component=component,
                                           value=vp$lambda_v_par[k1, k2], metric=metric))
  }}
  return(result)
}


GetResultDataframe <- function(mcmc_sample, vp, lrvb_cov, mfvb_cov, encoder) {
  GetMCMCResultRow <- function(param_name, component, draws, metric, FUN, method="mcmc", group=-1) {
    data.frame(param=param_name, component=component,
               value=FUN(draws), metric=metric, method=method, group=group)
  }
  result <- data.frame()

  param <- "mu"
  for (k in 1:vp$k) {
    result <-
      rbind(result, GetMCMCResultRow(param, k, FUN=mean, metric="mean", mcmc_sample$mu[,k]))
    result <-
      rbind(result, GetMCMCResultRow(param, k, FUN=sd, metric="sd", mcmc_sample$mu[,k]))
    result <- rbind(result, GetVBResultRow(param, k, value=vp$e_mu[k], metric="mean"))
    index <- encoder$e_mu + k
    result <-
      rbind(result, GetVBResultRow(param, k, value=sqrt(diag(mfvb_cov)[index]), metric="sd"))
    result <-
      rbind(result, GetVBResultRow(param, k, value=sqrt(diag(lrvb_cov)[index]), metric="sd", method="lrvb"))
  }

  param <- "mu_g"
  for (g in 1:vp$n_g) { for (k in 1:vp$k) {
    draws <- mcmc_sample$mu1[, g, k]
    result <- rbind(result, GetMCMCResultRow(param, k, FUN=mean, metric="mean", draws, group=g))
    result <- rbind(result, GetMCMCResultRow(param, k, FUN=sd, metric="sd", draws, group=g))
    result <- rbind(result, GetVBResultRow(param, k, value=vp$e_mu_g[[g]][k],
                                           metric="mean", group=g))
    index <- encoder$e_mu_g[[g]] + k
    mfvb_sd <- sqrt(diag(mfvb_cov)[index])
    lrvb_sd <- sqrt(diag(lrvb_cov)[index])
    result <- rbind(result, GetVBResultRow(param, k, value=mfvb_sd, metric="sd", group=g))
    result <- rbind(result, GetVBResultRow(param, k, value=lrvb_sd, metric="sd", method="lrvb", group=g))
  }}

  param <- "log_tau"
  k <- 1
  for (g in 1:vp$n_g) {
    draws <- -2 * log(mcmc_sample$sigma_y[, g])
    result <- rbind(result, GetMCMCResultRow(param, k, FUN=mean, metric="mean", draws, group=g))
    result <- rbind(result, GetMCMCResultRow(param, k, FUN=sd, metric="sd", draws, group=g))
    result <- rbind(result, GetVBResultRow(param, k, value=vp$e_log_tau[[g]],
                                           metric="mean", group=g))
    index <- encoder$e_log_tau[[g]] + 1
    mfvb_sd <- sqrt(diag(mfvb_cov)[index])
    lrvb_sd <- sqrt(diag(lrvb_cov)[index])
    result <- rbind(result, GetVBResultRow(param, k, value=mfvb_sd, metric="sd", group=g))
    result <- rbind(result, GetVBResultRow(param, k, value=lrvb_sd, metric="sd", method="lrvb", group=g))
  }

  param <- "lambda"
  vp$e_lambda <- vp$lambda_n_par * vp$lambda_v_par
  for (k1 in 1:vp$k) { for (k2 in 1:k1) {
    component <- k1 + 10 * k2
    draws <- mcmc_sample$lambda_mu[, k1, k2]
    result <- rbind(result, GetMCMCResultRow(param, component=component,
                                             FUN=mean, metric="mean", draws=draws))
    result <- rbind(result, GetMCMCResultRow(param, component=component,
                                             FUN=sd, metric="sd", draws=draws))
    result <- rbind(result, GetVBResultRow(param, component=component,
                                           value=vp$e_lambda[k1, k2], metric="mean"))
    index <- encoder$lambda_v_par + k2 + (k1 - 1) * vp$k
    mfvb_sd <- sqrt(diag(mfvb_cov)[index])
    lrvb_sd <- sqrt(diag(lrvb_cov)[index])
    result <- rbind(result, GetVBResultRow(param, component, value=mfvb_sd, metric="sd"))
    result <- rbind(result, GetVBResultRow(param, component, value=lrvb_sd, metric="sd", method="lrvb"))
  }}

  return(result)
}




############################################
# Setup and simulated data:

SampleParams <- function(k, n_g, sigma_scale = 1) {
  true_params <- list()
  true_params$k <- k
  true_params$n_g <- n_g
  true_params$mu <- as.double(1:k)
  true_params$sigma <- sigma_scale * (diag(k) + matrix(0.1, k, k))
  true_params$lambda <- solve(true_params$sigma)
  true_params$tau <- list()
  true_params$mu_g <- list()
  for (g in 1:n_g) {
    true_params$tau[[g]] <- 1.5 + g / n_g
    true_params$mu_g[[g]] <- mvrnorm(n=1, mu=true_params$mu, Sigma=true_params$sigma)
  }

  return(true_params)
}


SetVPFromTrueParams <- function(true_params) {
  k <- length(true_params$mu)
  second_moment_start <- function(x) {
    x %*% t(x) + diag(length(x))
  }
  vp <- list()
  vp$e_mu = true_params$mu
  vp$e_mu2 = second_moment_start(true_params$mu)
  vp$e_tau <- list()
  vp$e_log_tau <- list()
  vp$e_mu_g <- list()
  vp$e_mu2_g_vec <- list()
  for (g in 1:n_g) {
    vp$e_mu_g[[g]] = true_params$mu_g[[g]]
    vp$e_mu2_g_vec[[g]] = second_moment_start(true_params$mu_g[[g]])
    vp$e_tau[[g]] <- true_params$tau[[g]]
    vp$e_log_tau[[g]] <- log(true_params$tau[[g]]) - k
  }

  vp$lambda_v_par <- true_params$lambda / k
  vp$lambda_n_par <- k + 1

  vp$k <- k
  vp$n_g <- n_g

  return(vp)
}

SetPriorsFromVP <- function(vp) {
  pp <- list()
  k <- vp$k
  pp[["k_reg"]] <- k
  pp[["mu_loc"]] <- rep(0, k)
  pp[["mu_info"]] <- diag(k)
  pp[["lambda_eta"]] <- 5.0
  pp[["lambda_alpha"]] <- 3.0
  pp[["lambda_beta"]] <- 3.0
  pp[["tau_alpha"]] <- 3.0
  pp[["tau_beta"]] <- 3.0

  return(pp)
}
