
# library(RcppEigen)
# library(Rcpp)
library(Matrix) # Needed for Matrix::diag :(
# library(MASS)
# library(optimx)
# library(trust)


# For now...
# working_dir <- file.path(Sys.getenv("GIT_REPO_LOC"), "microcredit_vb/stan")
# setwd(working_dir)

# # # Test the Jacobian is transposed.
# # # https://github.com/stan-dev/math/issues/230
# This needs to go in a testing folder.
# jac_test <- TestJacobian()
# if (any(dim(jac_test$jac) != dim(jac_test$A))) {
#   print("Stan has transposed the Jacobian, as expected.")
# } else {
#   stop("This code expects the Stan Jacobian bug (#230) _not_ to be fixed, but it has been fixed.")
# }
#


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

## Now using natural parameters instead.
# WishartMeansFromGradient <- function(e_lambda_grad, e_log_det_lambda_grad) {
#   k <- nrow(e_lambda_grad)
#   stopifnot(ncol(e_lambda_grad) == k)
#   result <- list()
#   result$n_par <- 2 * e_log_det_lambda_grad + 1 + k
#   # off diagonals * 0.5 because of double counting
#   scale_matrix <- diag(0.5, k) + matrix(0.5, k, k)
#   result$w_inv_par <- -2 * scale_matrix * e_lambda_grad
#   stopifnot(min(eigen(result$w_inv_par)$values) > 0)
#   result$v_par <- solve(result$w_inv_par)
#   result$e_lambda <- result$n_par * result$v_par
#   result$e_log_det_lambda <-
#     r_mulitvariate_digamma(0.5 * result$n_par, k) + k * log(2) -
#     det(result$w_inv_par, log=TRUE) # NOTE: I think the log argument doesn't work
#   return(result)
# }

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
      mu_g_update <- MVNMeansFromGradient(model_grads$e_mu_g_vec[[g]],
                                          model_grads$e_mu2_g_vec[[g]])
      vp$e_mu_g_vec[[g]] <- mu_g_update$e_mu
      vp$e_mu2_g_vec[[g]] <- mu_g_update$e_mu2
    }

    model_grads <- ModelGradient(x, y, y_g, vp, pp, FALSE, TRUE)
    mu_update <- MVNMeansFromGradient(model_grads$e_mu, model_grads$e_mu2)
    vp$e_mu <- mu_update$e_mu
    vp$e_mu2 <- mu_update$e_mu2

    model_grads <- ModelGradient(x, y, y_g, vp, pp, FALSE, TRUE)
    for (g in 1:n_g) {
      tau_update <- GammaMeansFromGradient(
        model_grads$e_tau_vec[[g]], model_grads$e_log_tau_vec[[g]])
      vp$e_tau_vec[[g]] <- tau_update$e_tau
      vp$e_log_tau_vec[[g]] <- tau_update$e_log_tau

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
    result <- rbind(result, GetVBResultRow("mu_g", k, value=vp$e_mu_g_vec[[g]][k],
                                           metric=metric, group=g))
  }}

  for (g in 1:vp$n_g) {
    result <- rbind(result, GetVBResultRow("log_tau", 1, value=vp$e_log_tau_vec[[g]],
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
    result <- rbind(result, GetVBResultRow(param, k, value=vp$e_mu_g_vec[[g]][k],
                                           metric="mean", group=g))
    index <- encoder$e_mu_g_vec[[g]] + k
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
    result <- rbind(result, GetVBResultRow(param, k, value=vp$e_log_tau_vec[[g]],
                                           metric="mean", group=g))
    index <- encoder$e_log_tau_vec[[g]] + 1
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
  true_params$tau_vec <- list()
  true_params$mu_g_vec <- list()
  for (g in 1:n_g) {
    true_params$tau_vec[[g]] <- 1.5 + g / n_g
    true_params$mu_g_vec[[g]] <- mvrnorm(n=1, mu=true_params$mu, Sigma=true_params$sigma)
  }

  return(true_params)
}


SimulateData <- function(n_per_g, true_params) {
  k <- true_params$k
  n_g <- true_params$n_g
  n <- n_g * n_per_g
  data <- list()
  data$n <- n
  x <- matrix(runif(n * k), n, k)
  y_g <- rep(1:n_g, each=n_per_g)
  y <- rep(0, n)
  for (row in 1:n) {
    row_g <- y_g[row]
    y[row] <- rnorm(1, t(x[row, ]) %*% true_params$mu_g_vec[[row_g]],
                    sd = 1 / sqrt(true_params$tau_vec[[row_g]]))
  }

  data$x <- x
  data$y_g <- y_g
  data$y <- y

  return(data)
}


SetVPFromTrueParams <- function(true_params) {
  k <- length(true_params$mu)
  second_moment_start <- function(x) {
    x %*% t(x) + diag(length(x))
  }
  vp <- list()
  vp$e_mu = true_params$mu
  vp$e_mu2 = second_moment_start(true_params$mu)
  vp$e_tau_vec <- list()
  vp$e_log_tau_vec <- list()
  vp$e_mu_g_vec <- list()
  vp$e_mu2_g_vec <- list()
  for (g in 1:n_g) {
    vp$e_mu_g_vec[[g]] = true_params$mu_g_vec[[g]]
    vp$e_mu2_g_vec[[g]] = second_moment_start(true_params$mu_g_vec[[g]])
    vp$e_tau_vec[[g]] <- true_params$tau_vec[[g]]
    vp$e_log_tau_vec[[g]] <- log(true_params$tau_vec[[g]]) - k
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
  pp[["k"]] <- k
  pp[["mu_mean"]] <- rep(0, k)
  pp[["mu_info"]] <- diag(k)
  pp[["lambda_eta"]] <- 5.0
  pp[["lambda_alpha"]] <- 3.0
  pp[["lambda_beta"]] <- 3.0
  pp[["tau_alpha"]] <- 3.0
  pp[["tau_beta"]] <- 3.0

  # Optimization parameters
  pp[["lambda_diag_min"]] <- 0.0001
  pp[["lambda_n_min"]] <- k + 0.5

  return(pp)
}