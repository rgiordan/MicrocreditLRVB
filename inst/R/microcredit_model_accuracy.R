library(ggplot2)
library(dplyr)
library(reshape2)
library(Matrix)
library(gridExtra)

library(MicrocreditLRVB)

# Load previously computed Stan results
# analysis_name <- "simulated_data_robust"
# analysis_name <- "simulated_data_nonrobust"
# analysis_name <- "simulated_data_lambda_beta"
# analysis_name <- "real_data_informative_priors"
analysis_name <- "real_data_t_prior"


project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/simulated_data")

#################################################
# Load MCMC and VB results.

fit_file <- file.path(project_directory, paste(analysis_name, "_mcmc_and_vb.Rdata", sep=""))
print(paste("Loading fits from ", fit_file))

LoadIntoEnvironment <- function(filename) {
  my_env <- environment()
  load(filename, envir=my_env)
  return(my_env)
}

fit_env <- LoadIntoEnvironment(fit_file)

stan_results <- fit_env$mcmc_environment$results$original
stan_results_perturb <- fit_env$mcmc_environment$results$perturbed

x <- stan_results$dat$x
y <- stan_results$dat$y
y_g <- stan_results$dat$y_group

pp <- fit_env$mcmc_environment$pp
pp_perturb <- fit_env$mcmc_environment$pp_perturb


###########################################
# Extract results

vp_opt <- fit_env$vb_fit$vp_opt
lrvb_terms <- fit_env$lrvb_terms  
lrvb_cov <- lrvb_terms$lrvb_cov

mp_opt <- GetMoments(vp_opt)
mfvb_cov <- GetCovariance(vp_opt)

# Convenient indices
vp_indices <- GetParametersFromVector(vp_opt, as.numeric(1:vp_opt$encoded_size), FALSE)
mp_indices <- GetMomentsFromVector(mp_opt, as.numeric(1:mp_opt$encoded_size))
pp_indices <- GetPriorsFromVector(pp, as.numeric(1:pp$encoded_size))


############################################
# Inspect the results

GetResultsDF <- function(vp_opt, mcmc_sample, lrvb_terms) {
  mp_opt <- GetMoments(vp_opt)
  mfvb_cov <- GetCovariance(vp_opt)
  lrvb_cov <- lrvb_terms$lrvb_cov
  
  # Pack the standard deviations into readable forms.
  mfvb_sd <- GetMomentsFromVector(mp_opt, sqrt(diag(mfvb_cov)))
  lrvb_sd <- GetMomentsFromVector(mp_opt, sqrt(diag(lrvb_cov)))
  results_lrvb <- SummarizeMomentParameters(mp_opt, mfvb_sd, lrvb_sd)
  results_mcmc <- SummarizeMCMCResults(mcmc_sample)
  results <- rbind(results_lrvb, results_mcmc)
  
  return(results)  
}

results <- GetResultsDF(fit_env$vb_fit$vp_opt, fit_env$mcmc_environment$mcmc_sample, lrvb_terms)
results$data <- "original"
results_pert <- GetResultsDF(
  fit_env$vb_fit_perturb$vp_opt, fit_env$mcmc_environment$mcmc_sample_perturbed, fit_env$lrvb_terms_perturb)
results_pert$data <- "perturbed"

results_both <-
  rbind(results, results_pert) %>%
  dcast(par + component + group + metric + method ~ data, value.var="val") %>%
  filter(metric == "mean")


stop()

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
  geom_abline(aes(slope=1, intercept=0)) +
  scale_x_log10() + scale_y_log10()

ggplot(filter(mean_results, par == "log_tau")) +
  geom_point(aes(x=mcmc, y=mfvb, color=par), size=3) +
  geom_abline(aes(slope=1, intercept=0))

ggplot(filter(mean_results, par == "lambda")) +
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

# I find it very strange that LRVB gets this wrong when the means appear so accurate.
ggplot(filter(sd_results, par == "tau")) +
  geom_point(aes(x=mcmc, y=mfvb, shape=par, color="mfvb"), size=3) +
  geom_point(aes(x=mcmc, y=lrvb, shape=par, color="lrvb"), size=3) +
  geom_abline(aes(slope=1, intercept=0))

ggplot(filter(sd_results, par == "log_tau")) +
  geom_point(aes(x=mcmc, y=mfvb, shape=par, color="mfvb"), size=3) +
  geom_point(aes(x=mcmc, y=lrvb, shape=par, color="lrvb"), size=3) +
  geom_abline(aes(slope=1, intercept=0))

ggplot(filter(sd_results, par == "tau")) +
  geom_point(aes(x=mcmc, y=mfvb, shape=par, color="mfvb"), size=3) +
  geom_abline(aes(slope=1, intercept=0)) +
  scale_x_log10() + scale_y_log10()


#####################################
# Check some other parameters

for (group in 1:7) {
  print(mp_opt$tau[[group]]$e_log)
  print(mean(-2 * log(mcmc_sample$sigma_y[, group])))
}



for (group in 1:7) {
  alpha <- vp_opt$tau[[group]]$alpha 
  beta <- vp_opt$tau[[group]]$beta 
  print(beta / (alpha - 1))
  print(mean(mcmc_sample$sigma_y[, group] ^ 2))
}


filter(sd_results, par == "tau", group == 2)
