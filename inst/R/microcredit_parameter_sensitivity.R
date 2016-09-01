library(ggplot2)
library(dplyr)
library(reshape2)
library(Matrix)

library(MicrocreditLRVB)

# Load previously computed Stan results
analysis_name <- "simulated_data_robust"
#analysis_name <- "simulated_data_nonrobust"

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/simulated_data")

fit_file <- file.path(project_directory, paste(analysis_name, "_mcmc_and_vb.Rdata", sep=""))
print(paste("Loading fits from ", fit_file))

fit_env <- environment()
load(fit_file, envir=fit_env)
fit_env <- as.list(fit_env)

###########################################
# Extract results

vp_opt <- fit_env$vb_fit$vp_opt

lrvb_cov <- fit_env$lrvb_terms$lrvb_cov
prior_sens <- fit_env$prior_sens

mp_opt <- GetMoments(vp_opt)
mfvb_cov <- GetCovariance(vp_opt)

pp <- fit_env$stan_results$pp

# Convenient indices
vp_indices <- GetParametersFromVector(vp_opt, as.numeric(1:vp_opt$encoded_size), FALSE)
mp_indices <- GetMomentsFromVector(mp_opt, as.numeric(1:mp_opt$encoded_size))
pp_indices <- GetPriorsFromVector(pp, as.numeric(1:pp$encoded_size))


#######################################
# VB sensitivity measures

# Calculate vb perturbed estimates
mp_opt_vec <- GetVectorFromMoments(mp_opt)
mu_info_offdiag_sens <- prior_sens[, pp_indices$mu_info[1, 2]]
mp_opt_vec_pert <- mp_opt_vec + stan_results$perturb_epsilon * mu_info_offdiag_sens
mp_opt_pert <- GetMomentsFromVector(mp_opt, mp_opt_vec_pert)


###########################
# Summarize results

# Pack the standard deviations into readable forms.
mfvb_sd <- GetMomentsFromVector(mp_opt, sqrt(diag(mfvb_cov)))
lrvb_sd <- GetMomentsFromVector(mp_opt, sqrt(diag(lrvb_cov)))

results_vb <- SummarizeMomentParameters(mp_opt, mfvb_sd, lrvb_sd)
results_vb_pert <- SummarizeMomentParameters(mp_opt_pert, mfvb_sd, lrvb_sd)
results_vb_pert$method <- "mfvb_perturbed"

results_mcmc <- SummarizeMCMCResults(fit_env$stan_results$mcmc_sample)
results_mcmc_pert <- SummarizeMCMCResults(fit_env$stan_results$mcmc_sample_perturbed)
results_mcmc_pert$method <- "mcmc_perturbed"

results <- rbind(results_vb, results_mcmc)
result_pert <- rbind(results_vb, results_vb_pert, results_mcmc, results_mcmc_pert)

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


mean_pert_results <-
  filter(result_pert, metric == "mean") %>%
  dcast(par + component + group ~ method, value.var="val") %>%
  mutate(mfvb_diff = mfvb_perturbed - mfvb, mcmc_diff = mcmc_perturbed - mcmc)

ggplot(filter(mean_pert_results, par != "mu_g")) +
  geom_point(aes(x=mcmc_diff, y=mfvb_diff, color=par), size=3) +
  geom_abline(aes(slope=1, intercept=0))

ggplot(filter(mean_pert_results, par == "mu_g")) +
  geom_point(aes(x=mcmc_diff, y=mfvb_diff, color=par), size=3) +
  geom_abline(aes(slope=1, intercept=0))


#######################
# Prior sensitivity results with respect to a particular prior parameter.

draws_mat <- fit_env$stan_results$draws_mat
log_prior_grad_mat <- fit_env$stan_results$log_prior_grad_mat

# NOTE: the sign is wrong for the mu_info and tau derivatives 
# but not the lambda_eta derivatives.
# This points to a possible bug in my priors?

prior_sensitivity_ind <- pp_indices$mu_info[1, 2]; ind_name <- "mu_info_offdiag_sens"
#prior_sensitivity_ind <- pp_indices$mu_info[1, 1]; ind_name <- "mu_info_diag_sens"
#prior_sensitivity_ind <- pp_indices$lambda_eta; ind_name <- "lambda_eta"
#prior_sensitivity_ind <- pp_indices$lambda_alpha; ind_name <- "lambda_alpha"
#prior_sensitivity_ind <- pp_indices$lambda_beta; ind_name <- "lambda_beta"
#prior_sensitivity_ind <- pp_indices$tau_alpha; ind_name <- "tau_alpha"
#prior_sensitivity_ind <- pp_indices$tau_alpha; ind_name <- "tau_beta"
#prior_sensitivity_ind <- pp_indices$mu_loc[1]; ind_name <- "mu_loc_1"

mcmc_subset_size <- 1000
mcmc_prior_sens_subset <- cov(draws_mat[1:mcmc_subset_size, ],
                              log_prior_grad_mat[1:mcmc_subset_size, ])

mcmc_prior_sens <- cov(draws_mat, log_prior_grad_mat)

vb_prior_sens_mom <- GetMomentsFromVector(mp_opt, prior_sens[, prior_sensitivity_ind])
mcmc_prior_sens_mom <- GetMomentsFromVector(mp_opt, mcmc_prior_sens[, prior_sensitivity_ind])
mcmc_prior_sens_subset <- GetMomentsFromVector(mp_opt, mcmc_prior_sens_subset[, prior_sensitivity_ind])
prior_sens_results <-
  rbind(SummarizeRawMomentParameters(vb_prior_sens_mom, metric=ind_name, method="lrvb") %>%
          mutate(draws=""),
        SummarizeRawMomentParameters(mcmc_prior_sens_mom, metric=ind_name, method="mcmc") %>%
          mutate(draws=nrow(draws_mat)),
        SummarizeRawMomentParameters(mcmc_prior_sens_subset, metric=ind_name, "mcmc") %>%
          mutate(draws=mcmc_subset_size)
  )


prior_sens_results_graph <-
  inner_join(
  filter(prior_sens_results, method == "lrvb") %>%
    dcast(par + component + group + metric ~ method, value.var="val"),
  filter(prior_sens_results, method == "mcmc") %>%
    dcast(par + component + group + metric + draws ~ method, value.var="val"),
  by=c("par", "component", "group", "metric"))


ggplot(prior_sens_results_graph) +
  geom_point(aes(x=lrvb, y=mcmc, color=par), size=3) +
  geom_abline(aes(slope=1, intercept=0)) +
  facet_grid(~ draws) +
  ggtitle(ind_name)



