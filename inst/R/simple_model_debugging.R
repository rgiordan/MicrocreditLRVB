library(ggplot2)
library(dplyr)
library(reshape2)
library(Matrix)
library(mvtnorm)

library(LRVBUtils)

library(MicrocreditLRVB)
library_location <- file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/")
source(file.path(library_location, "inst/R/microcredit_stan_lib.R"))

library(ggplot2)
library(dplyr)
library(reshape2)
library(Matrix)
library(mvtnorm)

library(MicrocreditLRVB)

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/simulated_data")

analysis_name <- "simulated_data_easy"

set.seed(42)

##########################
# Prior parameters

# The dimension of the regressors.
k <- 2

pp <- list()
pp[["k_reg"]] <- k
pp[["mu_loc"]] <- rep(0, k)
pp[["mu_info"]] <- matrix(c(0.02, 0., 0, 0.02), k, k)
pp[["lambda_eta"]] <- 15.01
pp[["lambda_alpha"]] <- 20.01
pp[["lambda_beta"]] <- 20.01
pp[["tau_alpha"]] <- 2.01
pp[["tau_beta"]] <- 2.01

#############################
# Simualate some data
true_params <- list()

# Set parameters similar to the microcredit data.  Note that the true mean is
# an unlikely value relative to the prior.  This will result in a non-robust
# posterior.
true_params$true_mu <- c(0, 1)
true_params$true_sigma <- matrix(c(12, 0, 0, 12), 2, 2)
true_params$true_lambda <- solve(true_params$true_sigma)
true_params$true_tau <- 1 / (0.001^2)

# Number of groups
n_g <- 100

# Number of data points per group
n_per_group <- 3

sim_data <- SimulateData(true_params, n_g, n_per_group, binary_x=FALSE)
x <- sim_data$x
y_g <- sim_data$y_g
y <- sim_data$y
true_params$true_mu_g <- sim_data$true_mu_g

# Sanity checks
mu_g_mat <- do.call(rbind, true_params$true_mu_g)
cov(mu_g_mat)
solve(true_params$true_lambda)

#############################
# Fit

diag_min <- 1e-5
vp_base <- InitializeZeroVariationalParameters(x, y, y_g, diag_min=diag_min, tau_min=0)
vp_reg <- InitializeVariationalParameters(x, y, y_g, diag_min=diag_min, tau_min=0)

mu_g_mat <- matrix(NaN, vp_base$n_g, vp_base$k_reg)
for (g in 1:vp_base$n_g) {
  mu_g_mat[g, ] <- vp_reg$mu_g[[g]][["loc"]]
  vp_reg$mu_g[[g]][["info"]] <- 1e6 * diag(vp_base$k_reg)
}
solve(cov(mu_g_mat))

vp_reg$mu_loc <- colMeans(mu_g_mat)
vp_reg$mu_info <- 1e6 * diag(vp_base$k_reg)
vp_reg$lambda_n <- 1000
vp_reg$lambda_v <- solve(cov(mu_g_mat)) / vp_reg$lambda_n

mask <- GetGlobalVectorFromParameters(vp_base, FALSE)
mask <- rep(0, length(mask))
vp_mask <- GetParametersFromGlobalVector(vp_base, mask, FALSE)
# vp_mask$mu_loc[] <- 1
# vp_mask$mu_info[] <- 1
vp_mask$lambda_v[] <- 1
vp_mask$lambda_n <- 1
mask <- GetGlobalVectorFromParameters(vp_mask, FALSE) == 1

theta_init <- GetGlobalVectorFromParameters(vp_reg, TRUE)
chol_v <- chol(vp_reg$lambda_v - diag_min * diag(vp_reg$k_reg))
chol_v_scale <- diag(chol_v)
log(chol_v_scale)
solve(diag(chol_v_scale), chol_v)
theta_init


vp_indices <- GetParametersFromGlobalVector(vp_base, as.numeric(1:length(theta_init)), FALSE)
vp_indices$lambda_v
vp_indices$lambda_n

vm_reg <- GetMoments(vp_reg)
vm_reg$lambda_e

DerivFun <- function(x, y, y_g, base_vp, pp,
                     calculate_gradient, calculate_hessian,
                     unconstrained) {
  GetCustomElboDerivatives(x, y, y_g, base_vp, pp,
                           include_obs=FALSE, include_hier=TRUE,
                           include_prior=FALSE, include_entropy=FALSE,
                           use_group=TRUE, g=-1,
                           calculate_gradient=calculate_gradient,
                           calculate_hessian=calculate_hessian,
                           unconstrained=unconstrained)
}

opt_fns <- GetGlobalOptimFunctions(x, y, y_g, vp_reg, pp, DerivFun=DerivFun, mask=mask)
opt_fns$OptimVal(theta_init[mask])
vm_reg$lambda_e
vm_reg$lambda_e_log_det
vm_reg$mu_g[[n_g]]

library(trust)
TrustObj <- function(theta) {
  list(value=opt_fns$OptimVal(theta),
       gradient=opt_fns$OptimGrad(theta),
       hessian=opt_fns$OptimHess(theta))
}

trust_result <- trust(TrustObj, theta_init[mask], rinit=1, rmax=100, minimize=FALSE)
trust_result$converged

tr_theta <- theta_init
tr_theta[mask] <- trust_result$argument

vp_ev$lambda_v
vp_ev$lambda_n


vp_tr <- GetParametersFromGlobalVector(vp_reg, tr_theta, TRUE)
vm_tr <- GetMoments(vp_tr)

vp_tr$lambda_v
vp_tr$lambda_n
vp_tr$lambda_v * vp_tr$lambda_n

true_params$true_lambda
vm_reg$lambda_e
vm_tr$lambda_e

vm_reg$lambda_e_log_det
vm_tr$lambda_e_log_det




############
# hess <- opt_fns$OptimHess(trust_result$argument)
hess <- opt_fns$OptimHess(theta_init[mask])
grad <- opt_fns$OptimGrad(theta_init[mask])

hess_eig <- eigen(hess)
hess_eig$values
ev <- rep(0, length(theta_init))
ev[mask] <- hess_eig$vectors[,4]
vp_zeros <- GetParametersFromVector(vp_base, rep(0, length(GetVectorFromParameters(vp_base, FALSE))), FALSE)
vp_ev <- GetParametersFromGlobalVector(vp_zeros, ev, FALSE)
vp_ev$lambda_v
vp_ev$lambda_n

############

############






if (FALSE) {
  bounds <- GetVectorBounds(vp_base, loc_bound=30, info_bound=20)
  bfgs_time <- Sys.time()
  optim_result0 <- optim(theta_init[mask], opt_fns$OptimVal, opt_fns$OptimGrad, method="L-BFGS-B",
                         lower=bounds$theta_lower[mask], upper=bounds$theta_upper[mask],
                         control=list(fnscale=-1, maxit=2000, trace=1, factr=1))
  stopifnot(optim_result0$convergence == 0)
  print(optim_result0$message)
  bfgs_time <- Sys.time() - bfgs_time
}

if (FALSE) {
  optim_result <- NewtonsMethod(opt_fns$OptimVal, opt_fns$OptimGrad, opt_fns$OptimHess,
                                theta_init=theta_init[mask], fn_scale=-1, tol=1e-8,
                                verbose=TRUE)
  
}

