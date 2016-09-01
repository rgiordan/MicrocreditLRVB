library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)

library(MicrocreditLRVB)

# Load previously computed Stan results
analysis_name <- "simulated_data_robust"
#analysis_name <- "simulated_data_nonrobust"

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/simulated_data")

stan_draws_file <-
  file.path(project_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))
print(paste("Loading draws from ", stan_draws_file))

stan_results <- environment()
load(stan_draws_file, envir=stan_results)
stan_results <- as.list(stan_results)

x <- stan_results$stan_dat$x
y <- stan_results$stan_dat$y
y_g <- stan_results$stan_dat$y_group


#############################
# Initialize

# "reg" for "regression"
vp_reg <- InitializeVariationalParameters(
  x, y, y_g, mu_diag_min=0.01, lambda_diag_min=1e-5, tau_min=1, lambda_n_min=0.5)
mp_reg <- GetMoments(vp_reg)


#########################################
# Fit and LRVB

vb_fit <- FitVariationalModel(x, y, y_g, vp_reg, pp)
print(vb_fit$bfgs_time + vb_fit$tr_time)

vp_opt <- vb_fit$vp_opt
lrvb_terms <- GetLRVB(x, y, y_g, vp_opt, pp)
prior_sens <- GetSensitivity(vp_opt, pp, lrvb_terms$jac, lrvb_terms$elbo_hess)


##########################################
# Get MCMC sensitivity measures

mcmc_sample <- extract(stan_results$stan_sim)
mcmc_sample_perturbed <- extract(stan_results$stan_sim_perturb)
mp_draws <- PackMCMCSamplesIntoMoments(mcmc_sample, mp_reg) # A little slow

draws_mat <- do.call(rbind, lapply(mp_draws, GetVectorFromMoments))
log_prior_grad_list <- GetMCMCLogPriorDerivatives(mp_draws, pp)
log_prior_grad_mat <- do.call(rbind, log_prior_grad_list)

stan_results$draws_mat <- draws_mat
stan_results$log_prior_grad_mat <- log_prior_grad_mat
stan_results$mcmc_sample_perturbed <- mcmc_sample_perturbed
stan_results$mcmc_sample <- mcmc_sample


###########################################
# Save fits

fit_file <- file.path(project_directory, paste(analysis_name, "_mcmc_and_vb.Rdata", sep=""))
print(paste("Saving fits to ", fit_file))

save(stan_results, vb_fit, lrvb_terms, prior_sens, file=fit_file)
