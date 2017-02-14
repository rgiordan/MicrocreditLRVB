library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)

library(MicrocreditLRVB)

# Load previously computed Stan results
# analysis_name <- "simulated_data_robust"
# analysis_name <- "simulated_data_nonrobust"
# analysis_name <- "simulated_data_lambda_beta"
# analysis_name <- "real_data_informative_priors"
analysis_name <- "real_data_t_prior"

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
stan_results <- mcmc_environment$results$original
stan_results_perturb <- mcmc_environment$results$perturbed

x <- stan_results$dat$x
y <- stan_results$dat$y
y_g <- stan_results$dat$y_group

pp <- mcmc_environment$pp
pp_perturb <- mcmc_environment$pp_perturb

stan_results$mcmc_time

#############################
# Initialize

# "reg" for "regression"
vp_reg <- InitializeVariationalParameters(
  x, y, y_g, mu_diag_min=0.01, lambda_diag_min=1e-5, tau_min=1, lambda_n_min=0.5)
mp_reg <- GetMoments(vp_reg)


#########################################
# Fit and LRVB

vb_fit <- FitVariationalModel(x, y, y_g, vp_reg, pp,
                              loc_bound=60, info_bound=100, tau_bound=100, bfgs_factr=1e8)
vp_opt <- vb_fit$vp_opt
print(vb_fit$bfgs_time + vb_fit$tr_time)
print(stan_results$mcmc_time)
lrvb_time <- Sys.time()
lrvb_terms <- GetLRVB(x, y, y_g, vp_opt, pp)
lrvb_time <- Sys.time() - lrvb_time
prior_sens <- GetSensitivity(vp_opt, pp, lrvb_terms$jac, lrvb_terms$elbo_hess)
mp_reg <- GetMoments(vp_reg)

vb_fit_perturb <- FitVariationalModel(x, y, y_g, vp_opt, pp_perturb,
                                      loc_bound=60, info_bound=100, tau_bound=100, bfgs_factr=1e8)
lrvb_terms_perturb <- GetLRVB(x, y, y_g, vb_fit_perturb$vp_opt, pp_perturb)


###################
# Debugging the perturbation

if (FALSE) {
  pp_perturb$mu_t_scale <- 0.1
  vb_fit_perturb <- FitVariationalModel(x, y, y_g, vp_opt, pp_perturb,
                                        loc_bound=60, info_bound=100, tau_bound=100, bfgs_factr=1e8)
  
  mean_comp <-
    rbind(SummarizeRawMomentParameters(GetMoments(vb_fit$vp_opt),  metric="mean", method = "orig"),
          SummarizeRawMomentParameters(GetMoments(vb_fit_perturb$vp_opt),  metric="mean", method = "perturb")) %>%
    dcast(par + component + group + metric ~ method, value.var="val") %>%
    mutate(diff=orig - perturb)
  mean_comp
  
}


###################
# Check a more accurate convergence

if (FALSE) {
  library(trust)
  trust_fns <- GetTrustRegionELBO(x, y, y_g, vp_opt, pp, verbose = TRUE)
  trust_result <- trust(trust_fns$TrustFun, trust_fns$theta_init, 
                        rinit = 10, rmax = 100, minimize = FALSE, blather = TRUE, 
                        iterlim = 50, mterm=0, fterm=1e-10)
  new_vp_opt <- GetParametersFromVector(vp_reg, trust_result$argument, TRUE)  
  max(abs(trust_result$argument - vb_fit$trust_result$argument))
  
  lrvb_terms <- GetLRVB(x, y, y_g, vp_opt, pp)
  new_lrvb_terms <- GetLRVB(x, y, y_g, new_vp_opt, pp)
  
  plot(log(diag(lrvb_terms$lrvb_cov)), log(diag(new_lrvb_terms$lrvb_cov))); abline(0, 1)
}


##########################################
# Get MCMC sensitivity measures

mcmc_sample <- extract(stan_results$sim)
mcmc_sample_perturbed <- extract(stan_results_perturb$sim)
mp_draws <- PackMCMCSamplesIntoMoments(mcmc_sample, mp_reg)

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
save(mcmc_environment, vb_fit, vb_fit_perturb, lrvb_terms, lrvb_terms_perturb, prior_sens, lrvb_time, file=fit_file)

