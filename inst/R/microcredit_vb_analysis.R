library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)

library(MicrocreditLRVB)

# Load previously computed Stan results
#analysis_name <- "simulated_data_robust"
analysis_name <- "simulated_data_nonrobust"
#analysis_name <- "simulated_data_lambda_beta"

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/simulated_data")

stan_draws_file <-
  file.path(project_directory, paste(analysis_name, "_data_and_mcmc_draws.Rdata", sep=""))
print(paste("Loading draws from ", stan_draws_file))

LoadIntoEnvironment <- function(filename) {
  local_env <- environment()
  load(filename, envir=local_env)
  return(local_env)
}
mcmc_environment <- LoadIntoEnvironment(stan_draws_file)
stan_results <- mcmc_environment$results$epsilon_0.000000
stan_results_perturb <- mcmc_environment$results$epsilon_1.000000

x <- stan_results$dat$x
y <- stan_results$dat$y
y_g <- stan_results$dat$y_group

pp <- mcmc_environment$pp
pp_perturb <- mcmc_environment$pp_perturb

#############################
# Initialize

# "reg" for "regression"
vp_reg <- InitializeVariationalParameters(
  x, y, y_g, mu_diag_min=0.01, lambda_diag_min=1e-5, tau_min=1, lambda_n_min=0.5)
mp_reg <- GetMoments(vp_reg)


#########################################
# Fit and LRVB

vb_fit <- FitVariationalModel(x, y, y_g, vp_reg, pp)
vp_opt <- vb_fit$vp_opt
print(vb_fit$bfgs_time + vb_fit$tr_time)
lrvb_terms <- GetLRVB(x, y, y_g, vp_opt, pp)
prior_sens <- GetSensitivity(vp_opt, pp, lrvb_terms$jac, lrvb_terms$elbo_hess)

vb_fit_perturb <- FitVariationalModel(x, y, y_g, vp_opt, pp_perturb)



##########################################
# Get MCMC sensitivity measures

mcmc_sample <- extract(stan_results$sim)
mcmc_sample_perturbed <- extract(stan_results_perturb$sim)
mp_draws <- PackMCMCSamplesIntoMoments(mcmc_sample, mp_reg) # A little slow

draws_mat <- do.call(rbind, lapply(mp_draws, GetVectorFromMoments))
log_prior_grad_list <- GetMCMCLogPriorDerivatives(mp_draws, pp)
log_prior_grad_mat <- do.call(rbind, log_prior_grad_list)

# Save some of these objects in the mcmc_environment for use later.
mcmc_environment$mp_draws <- mp_draws
mcmc_environment$draws_mat <- draws_mat
mcmc_environment$log_prior_grad_mat <- log_prior_grad_mat
mcmc_environment$mcmc_sample_perturbed <- mcmc_sample_perturbed
mcmc_environment$mcmc_sample <- mcmc_sample


###########################################
# Save fits

fit_file <- file.path(project_directory, paste(analysis_name, "_mcmc_and_vb.Rdata", sep=""))
print(paste("Saving fits to ", fit_file))
save(mcmc_environment, vb_fit, vb_fit_perturb, lrvb_terms, prior_sens, file=fit_file)
