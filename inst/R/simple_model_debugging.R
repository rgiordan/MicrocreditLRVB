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
n_per_group <- 20

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

vp_base <- InitializeZeroVariationalParameters(
  x, y, y_g, mu_diag_min=0.01, lambda_diag_min=1e-5, tau_min=1, lambda_n_min=0.5)
vp_indices <- GetParametersFromVector(vp_base, as.numeric(1:vp_base$encoded_size), FALSE)
vp_reg <- InitializeVariationalParameters(
  x, y, y_g, mu_diag_min=vp_base$mu_diag_min, lambda_diag_min=vp_base$lambda_diag_min,
  tau_min=vp_base$tau_alpha_min, lambda_n_min=vp_base$lambda_n_min)


#############
# BFGS

DerivFun <- function(x, y, y_g, vp, pp,
                          calculate_gradient, calculate_hessian, unconstrained) {
  GetCustomElboDerivatives(x, y, y_g, vp, pp,
                           include_obs=TRUE, include_hier=TRUE,
                           include_prior=TRUE, include_entropy=TRUE,
                           global_only=FALSE,
                           calculate_gradient=calculate_gradient,
                           calculate_hessian=calculate_hessian,
                           unconstrained=unconstrained)
}
mask <- rep(TRUE, vp_reg$encoded_size)
bfgs_opt_fns <- GetOptimFunctions(x, y, y_g, vp_reg, pp, DerivFun=DerivFun, mask=mask)
theta_init <- GetVectorFromParameters(vp_reg, TRUE)
bfgs_opt_fns$OptimVal(theta_init)
length(bfgs_opt_fns$OptimGrad(theta_init)) == length(theta_init)
bounds <- GetVectorBounds(vp_base, loc_bound=30, info_bound=10, tau_bound=100)

bfgs_time <- Sys.time()
bfgs_result <- optim(theta_init[mask],
                       bfgs_opt_fns$OptimVal, bfgs_opt_fns$OptimGrad,
                       method="L-BFGS-B", lower=bounds$theta_lower[mask], upper=bounds$theta_upper[mask],
                       control=list(fnscale=-1, maxit=1000, trace=0, factr=1))
stopifnot(bfgs_result$convergence == 0)
print(bfgs_result$message)
bfgs_time <- Sys.time() - bfgs_time

vp_bfgs <- GetParametersFromVector(vp_reg, bfgs_result$par, TRUE)


#############
# Trust region

trust_fns <- GetTrustRegionELBO(x, y, y_g, vp_bfgs, pp, verbose=TRUE)
trust_result <- trust(trust_fns$TrustFun, trust_fns$theta_init,
                      rinit=1, rmax=100, minimize=FALSE, blather=TRUE,
                      iterlim=50)
trust_result$converged
trust_result$value



#################################
# LRVB

vp_opt <- GetParametersFromVector(vp_reg, trust_result$argument, TRUE)
vp_mom <- GetMoments(vp_opt)

moment_derivs <- GetMomentJacobian(vp_opt)
jac <- Matrix(moment_derivs$hess)

elbo_hess <- GetSparseELBOHessian(x, y, y_g, vp_opt, pp, TRUE)
lrvb_cov <- -1 * jac %*% Matrix::solve(elbo_hess, Matrix::t(jac))
min(diag(lrvb_cov))

mfvb_cov <- GetCovariance(vp_opt)
min(diag(mfvb_cov))
plot(sqrt(diag(lrvb_cov)), sqrt(diag(mfvb_cov))); abline(0, 1)

mfvb_sd <- GetMomentsFromVector(vp_mom, sqrt(diag(mfvb_cov)))
lrvb_sd <- GetMomentsFromVector(vp_mom, sqrt(diag(lrvb_cov)))








################################################
# Look at certain components.

theta_init <- GetVectorFromParameters(vp_opt, TRUE)
mask <- rep(TRUE, vp_base$encoded_size)

pp$tau_alpha <- 10
pp$tau_beta <- 10

EntropyFun(x, y, y_g, vp_reg, pp)
HierFun(x, y, y_g, vp_reg, pp)
ObsFun(x, y, y_g, vp_reg, pp)
PriorFun(x, y, y_g, vp_reg, pp)

tau  <- vp_reg$tau[[1]]
tau$alpha / tau$beta








################################

vp_bad <- GetParametersFromVector(vp_reg, bfgs_result$par, TRUE)
mp_bad <- GetMoments(vp_bad)
mp_bad$tau[[1]]
bfgs_opt_fns$OptimGrad(bfgs_result$par)
grad <- bfgs_opt_fns$OptimGrad(bfgs_result$par)

theta <- bfgs_result$par
FUN <- function(theta) {
  local_vp <- GetParametersFromVector(vp_bad, theta, TRUE)
  #ObsFun(x, y, y_g, local_vp, pp)
  #HierFun(x, y, y_g, local_vp, pp)
  PriorFun(x, y, y_g, local_vp, pp)
  #EntropyFun(x, y, y_g, local_vp, pp)
  #bfgs_opt_fns$OptimVal(theta)
}

# print(check_ind <- vp_indices$lambda_n)
# print(check_ind <- unique(as.numeric(vp_indices$lambda_v)))
# print(check_ind <- unique(as.numeric(vp_indices$mu_info)))
print(check_ind <- vp_indices$tau[[1]]$alpha)

trimmed_grad <- rep(0, vp_bad$encoded_size)
grad[check_ind]
theta[check_ind]
vp_bad$tau[[1]]
trimmed_grad[check_ind] <- grad[check_ind]
res <- EvaluateOnGrid(FUN, theta, trimmed_grad, -1, 1, 50)
qplot(epsilon, val, data=res)


which(grad %in% head(sort(grad), 10))
which.max(grad)
max(grad)


ObsGrad(x, y, y_g, vp_bad, pp)[check_ind]
HierGrad(x, y, y_g, vp_bad, pp)[check_ind]
PriorGrad(x, y, y_g, vp_bad, pp)[check_ind]
EntropyGrad(x, y, y_g, vp_bad, pp)[check_ind]

ObsFun(x, y, y_g, vp_bad, pp)
HierFun(x, y, y_g, vp_bad, pp)
PriorFun(x, y, y_g, vp_bad, pp)
EntropyFun(x, y, y_g, vp_bad, pp)


bfgs_opt_fns$OptimVal(theta)
theta_step <- theta + 1 * grad
bfgs_opt_fns$OptimVal(theta_step)
foo <- GetParametersFromVector(vp_base, theta_step, TRUE)
foo$lambda_n

vp_bad_summary <- SummarizeNaturalParameters(vp_bad)
vp_bad_summary %>% filter(metric == "mean")


vp_copy <- vp_bad
for (g in 1:vp_copy$n_g) {
  vp_copy$tau[[g]]$alpha <- 10000
}
PrintTauPrior(vp_copy, pp)







############
# hess <- opt_fns$OptimHess(trust_result$argument)
hess <- global_opt_fns$OptimHess(trust_result$argument)
grad <- global_opt_fns$OptimGrad(trust_result$argument)

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






###################################
# Check the sparse Hessian

if (FALSE) {
  elbo_derivs <- GetElboDerivatives(x, y, y_g, vp_reg, pp, TRUE, TRUE, TRUE)
  hess_sparse <- GetSparseELBOHessian(x, y, y_g, vp_reg, pp, TRUE) 
  max(abs(hess_sparse - elbo_derivs$hess))
  # image(hess_sparse)
  # image(Matrix(elbo_derivs$hess))
  hess_diff <- abs(hess_sparse- Matrix(elbo_derivs$hess))
  hess_diff[abs(hess_diff) < 1e-8] <- 0
}


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
                           global_only=TRUE,
                           calculate_gradient=calculate_gradient,
                           calculate_hessian=calculate_hessian,
                           unconstrained=unconstrained)
}

global_opt_fns <- GetGlobalOptimFunctions(x, y, y_g, vp_opt, pp, DerivFun=GlobalDerivFun, mask=mask)
GlobalTrustObj <- GetTrustObj(global_opt_fns)

trust_result <- trust(GlobalTrustObj, global_theta_init[mask],
                      rinit=1, rmax=100, minimize=FALSE, blather=TRUE)
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

local_trust_result <- trust(LocalTrustObj, theta_init[local_mask],
                            rinit=1, rmax=100, minimize=FALSE, blather=TRUE)
local_trust_result$converged





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
                           global_only=TRUE,
                           calculate_gradient=calculate_gradient,
                           calculate_hessian=calculate_hessian,
                           unconstrained=unconstrained)
}

global_opt_fns <- GetGlobalOptimFunctions(x, y, y_g, vp_opt, pp, DerivFun=GlobalDerivFun, mask=mask)
GlobalTrustObj <- GetTrustObj(global_opt_fns)

trust_result <- trust(GlobalTrustObj, global_theta_init[mask],
                      rinit=1, rmax=100, minimize=FALSE, blather=TRUE)
trust_result$converged

tr_theta <- global_theta_init
tr_theta[mask] <- trust_result$argument

vp_opt <- GetParametersFromGlobalVector(vp_opt, tr_theta, TRUE)

