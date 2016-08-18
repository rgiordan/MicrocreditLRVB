library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)
library(mvtnorm)
library(trust)
library(LRVBUtils)

library(MicrocreditLRVB)
library_location <- file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/")
source(file.path(library_location, "inst/R/microcredit_stan_lib.R"))

# Load previously computed Stan results
analysis_name <- "simulated_data_easy"
project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/simulated_data")

stan_draws_file <-
  file.path(project_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))
print(paste("Loading draws from ", stan_draws_file))

stan_results <- environment()
load(stan_draws_file, envir=stan_results)

x <- stan_results$stan_dat$x
y <- stan_results$stan_dat$y
y_g <- stan_results$stan_dat$y_group


####################
# Fix the prior

pp$k_reg <- pp$k
pp$mu_loc <- pp$mu_mean
pp_empty <- GetEmptyPriors(pp$k)
pp$encoded_size <- pp_empty$encoded_size
pp_indices <- GetPriorsFromVector(pp, as.numeric(1:pp$encoded_size))


#############################
# Initialize

vp_base <- InitializeZeroVariationalParameters(
  x, y, y_g, mu_diag_min=0.01, lambda_diag_min=1e-5, tau_min=1, lambda_n_min=0.5)
vp_indices <- GetParametersFromVector(vp_base, as.numeric(1:vp_base$encoded_size), FALSE)
mp_base <- GetMoments(vp_base)
mp_indices <- GetMomentsFromVector(mp_base, as.numeric(1:vp_base$encoded_size))
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
bounds <- GetVectorBounds(vp_base, loc_bound=30, info_bound=10, tau_bound=100)

bfgs_time <- Sys.time()
bfgs_result <- optim(theta_init[mask],
                     bfgs_opt_fns$OptimVal, bfgs_opt_fns$OptimGrad,
                     method="L-BFGS-B", lower=bounds$theta_lower[mask], upper=bounds$theta_upper[mask],
                     control=list(fnscale=-1, maxit=1000, trace=0, factr=1e9))
stopifnot(bfgs_result$convergence == 0)
print(bfgs_result$message)
bfgs_time <- Sys.time() - bfgs_time

vp_bfgs <- GetParametersFromVector(vp_reg, bfgs_result$par, TRUE)


#############
# Trust region

tr_time <- Sys.time()
trust_fns <- GetTrustRegionELBO(x, y, y_g, vp_bfgs, pp, verbose=TRUE)
trust_result <- trust(trust_fns$TrustFun, trust_fns$theta_init,
                      rinit=1, rmax=100, minimize=FALSE, blather=TRUE, iterlim=50)
tr_time <- Sys.time() - tr_time
trust_result$converged
trust_result$value

bfgs_time + tr_time


#################################
# LRVB

unconstrained <- TRUE # This seems to have a better condition number. 
vp_opt <- GetParametersFromVector(vp_reg, trust_result$argument, TRUE)
vp_mom <- GetMoments(vp_opt)
mfvb_cov <- GetCovariance(vp_opt)

moment_derivs <- GetMomentJacobian(vp_opt, unconstrained)
jac <- Matrix(moment_derivs$hess)

elbo_hess <- GetSparseELBOHessian(x, y, y_g, vp_opt, pp, unconstrained)

lrvb_cov <- -1 * jac %*% Matrix::solve(elbo_hess, Matrix::t(jac))
stopifnot(min(diag(lrvb_cov)) > 0)

min(diag(mfvb_cov))

mfvb_sd <- GetMomentsFromVector(vp_mom, sqrt(diag(mfvb_cov)))
lrvb_sd <- GetMomentsFromVector(vp_mom, sqrt(diag(lrvb_cov)))

#####################
# Sensitivity

# Get some indices
comb_indices <- GetPriorsAndParametersFromVector(
  vp_base, pp, as.numeric(1:(vp_base$encoded_size + pp$encoded_size)))
comb_prior_ind <- GetVectorFromPriors(comb_indices$pp)
comb_vp_ind <- GetVectorFromParameters(comb_indices$vp, FALSE)


log_prior_derivs <- GetLogPriorDerivatives(vp_opt, pp, TRUE, TRUE, TRUE)
log_prior_param_prior <- Matrix(log_prior_derivs$hess[comb_vp_ind, comb_prior_ind])

prior_sens <- -1 * jac %*% Matrix::solve(elbo_hess, log_prior_param_prior)
prior_sens_norm <- prior_sens / sqrt(diag(lrvb_cov))


#################
# View sensitivity

prior_sens_norm[mp_indices$mu_e_vec, unique(as.numeric(pp_indices$mu_info))]




###########################
# Sumamrize results

mcmc_sample <- extract(stan_results$stan_sim)
mcmc_sample_perturbed <- extract(stan_results$stan_sim_perturb)

results <- rbind(SummarizeMomentParameters(vp_mom, mfvb_sd, lrvb_sd),
                 SummarizeMCMCResults(mcmc_sample))

results_pert <- SummarizeMCMCResults(mcmc_sample_perturbed)
results_pert$method <- "mcmc_perturbed"


stop("Graphs follow -- not executing.")

mean_results <-
  filter(results, metric == "mean") %>%
  dcast(par + component + group ~ method, value.var="val")

ggplot(filter(mean_results, par != "mu_g")) +
  geom_point(aes(x=mcmc, y=mfvb, color=par), size=3) +
  geom_abline(aes(slope=1, intercept=0))

ggplot(filter(mean_results, par == "mu_g")) +
  geom_point(aes(x=mcmc, y=mfvb, color=par), size=3) +
  geom_abline(aes(slope=1, intercept=0))

ggplot(filter(mean_results, par == "tau")) +
  geom_point(aes(x=mcmc, y=mfvb, color=par), size=3) +
  geom_abline(aes(slope=1, intercept=0))


sd_results <-
  filter(results, metric == "sd") %>%
  dcast(par + component + group ~ method, value.var="val")

ggplot(filter(sd_results, par != "mu_g")) +
  geom_point(aes(x=mcmc, y=mfvb, shape=par, color="mfvb"), size=3) +
  geom_point(aes(x=mcmc, y=lrvb, shape=par, color="lrvb"), size=3) +
  geom_abline(aes(slope=1, intercept=0))

ggplot(filter(sd_results, par == "mu_g")) +
  geom_point(aes(x=mcmc, y=mfvb, shape=par, color="mfvb"), size=3) +
  geom_point(aes(x=mcmc, y=lrvb, shape=par, color="lrvb"), size=3) +
  geom_abline(aes(slope=1, intercept=0))



