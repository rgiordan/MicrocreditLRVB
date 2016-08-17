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

library(trust)

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
pp[["lambda_eta"]] <- 10
pp[["lambda_alpha"]] <- 3
pp[["lambda_beta"]] <- 3
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
# Initialize

diag_min <- 1e-5
vp_base <- InitializeZeroVariationalParameters(x, y, y_g, diag_min=diag_min, tau_min=0)
vp_indices <- GetParametersFromVector(vp_base, as.numeric(1:vp_base$encoded_size), FALSE)
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

vm_reg <- GetMoments(vp_reg)

vp_opt <- vp_reg


###################################
# Check Hessians

elbo_derivs <- GetElboDerivatives(x, y, y_g, vp_reg, pp, TRUE, TRUE, TRUE)
hess_sparse <- GetSparseELBOHessian(x, y, y_g, vp_reg, pp, TRUE) 
max(abs(hess_sparse - elbo_derivs$hess))
# image(hess_sparse)
# image(Matrix(elbo_derivs$hess))
hess_diff <- abs(hess_sparse- Matrix(elbo_derivs$hess))
hess_diff[abs(hess_diff) < 1e-9] <- 0
# image(Matrix(hess_diff))
hess_diff[1:15, 1:15]
which(hess_diff == max(hess_diff), arr.ind = TRUE)
hess_diff[454, 451]


###################################
# Optimize global parameters

global_theta_init <- GetGlobalVectorFromParameters(vp_opt, TRUE)
mask <- GetGlobalVectorFromParameters(vp_base, FALSE)
mask <- rep(TRUE, length(mask))

GlobalDerivFun <- function(x, y, y_g, vp, pp,
                           calculate_gradient, calculate_hessian, unconstrained) {
  GetCustomElboDerivatives(x, y, y_g, vp, pp,
                           include_obs=FALSE, include_hier=TRUE,
                           include_prior=TRUE, include_entropy=TRUE,
                           use_group=TRUE, g=-1,
                           calculate_gradient=calculate_gradient,
                           calculate_hessian=calculate_hessian,
                           unconstrained=unconstrained)
}

global_opt_fns <- GetGlobalOptimFunctions(x, y, y_g, vp_opt, pp, DerivFun=GlobalDerivFun, mask=mask)
GlobalTrustObj <- GetTrustObj(global_opt_fns)

trust_result <- trust(GlobalTrustObj, global_theta_init[mask], rinit=1, rmax=100, minimize=FALSE, blather=TRUE)
trust_result$converged

tr_theta <- global_theta_init
tr_theta[mask] <- trust_result$argument

vp_opt <- GetParametersFromGlobalVector(vp_opt, tr_theta, TRUE)



#####################################
# Optimize local parameters

theta_init <- GetVectorFromParameters(vp_opt, TRUE)
local_mask <- GetLocalMask(vp_base)

LocalDerivFun <- function(x, y, y_g, vp, pp,
                          calculate_gradient, calculate_hessian, unconstrained) {
  GetCustomElboDerivatives(x, y, y_g, vp, pp,
                           include_obs=TRUE, include_hier=TRUE,
                           include_prior=TRUE, include_entropy=TRUE,
                           global_only=FALSE,
                           calculate_gradient=calculate_gradient,
                           calculate_hessian=calculate_hessian,
                           unconstrained=unconstrained)
}

local_opt_fns <- GetOptimFunctions(x, y, y_g, vp_opt, pp, DerivFun=LocalDerivFun, mask=local_mask)
LocalTrustObj <- GetTrustObj(local_opt_fns)

trust_result <- trust(LocalTrustObj, theta_init[local_mask],
                      rinit=1, rmax=100, minimize=FALSE, blather=TRUE)
trust_result$converged

tr_theta <- global_theta_init
tr_theta[mask] <- trust_result$argument

vp_tr <- GetParametersFromGlobalVector(vp_opt, tr_theta, TRUE)
vm_tr <- GetMoments(vp_tr)


################################################
# Look at certain components.

theta_init <- GetVectorFromParameters(vp_opt, TRUE)
mask <- rep(TRUE, vp_base$encoded_size)

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

EntropyFun(x, y, y_g, vp_reg, pp)
HierFun(x, y, y_g, vp_reg, pp)
ObsFun(x, y, y_g, vp_reg, pp)
PriorFun(x, y, y_g, vp_reg, pp)



############
# hess <- opt_fns$OptimHess(trust_result$argument)
hess <- opt_fns$OptimHess(trust_result$argument)
grad <- opt_fns$OptimGrad(trust_result$argument)

hess_eig <- eigen(hess)
stopifnot(sign(max(hess_eig$values)) < 1e-8)
hess_eig$values
ev <- rep(0, length(theta_init))
ev[mask] <- hess_eig$vectors[,1]
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

