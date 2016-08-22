library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)

library(MicrocreditLRVB)

# Load previously computed Stan results
#analysis_name <- "simulated_data_easy"
analysis_name <- "simulated_data_nonrobust"

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


#############################
# Initialize

# "reg" for "regression"
vp_reg <- InitializeVariationalParameters(
  x, y, y_g, mu_diag_min=0.01, lambda_diag_min=1e-5, tau_min=1, lambda_n_min=0.5)
mp_reg <- GetMoments(vp_reg)

# Convenient indices
vp_indices <- GetParametersFromVector(vp_reg, as.numeric(1:vp_reg$encoded_size), FALSE)
mp_indices <- GetMomentsFromVector(mp_reg, as.numeric(1:vp_reg$encoded_size))
pp_indices <- GetPriorsFromVector(pp, as.numeric(1:pp$encoded_size))


#########################################
# Fit and LRVB

vb_fit <- FitVariationalModel(x, y, y_g, vp_reg, pp)
print(vb_fit$bfgs_time + vb_fit$tr_time)

vp_opt <- vb_fit$vp_opt
vp_mom <- GetMoments(vp_opt)
mfvb_cov <- GetCovariance(vp_opt)
lrvb_terms <- GetLRVB(x, y, y_g, vp_opt, pp)
lrvb_cov <- lrvb_terms$lrvb_cov
prior_sens <- GetSensitivity(vp_opt, pp, lrvb_terms$jac, lrvb_terms$elbo_hess)

# Calculate vb perturbed estimates
vp_mom_vec <- GetVectorFromMoments(vp_mom)
mu_info_offdiag_sens <- prior_sens[, pp_indices$mu_info[1, 2]]
vp_mom_vec_pert <- vp_mom_vec + stan_results$perturb_epsilon * mu_info_offdiag_sens
vp_mom_pert <- GetMomentsFromVector(vp_mom, vp_mom_vec_pert)


##########################################
# Get MCMC sensitivity measures

mcmc_sample <- extract(stan_results$stan_sim)
mcmc_sample_perturbed <- extract(stan_results$stan_sim_perturb)

draws_mat <- do.call(rbind, lapply(mp_draws, GetVectorFromMoments))
log_prior_grad_list <- GetMCMCLogPriorDerivatives(mp_draws, pp)
log_prior_grad_mat <- do.call(rbind, log_prior_grad_list)

###########################
# Summarize results

# Pack the standard deviations into readable forms.
mfvb_sd <- GetMomentsFromVector(vp_mom, sqrt(diag(mfvb_cov)))
lrvb_sd <- GetMomentsFromVector(vp_mom, sqrt(diag(lrvb_cov)))

results_vb <- SummarizeMomentParameters(vp_mom, mfvb_sd, lrvb_sd)
results_vb_pert <- SummarizeMomentParameters(vp_mom_pert, mfvb_sd, lrvb_sd)
results_vb_pert$method <- "mfvb_perturbed"

results_mcmc <- SummarizeMCMCResults(mcmc_sample)
results_mcmc_pert <- SummarizeMCMCResults(mcmc_sample_perturbed)
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


# Prior sensitivity results with respect to a particular prior parameter.
prior_sensitivity_ind <- pp_indices$mu_info[1, 2]; ind_name <- "mu_info_offdiag_sens"
#prior_sensitivity_ind <- pp_indices$mu_info[1, 1]; ind_name <- "mu_info_diag_sens"
#prior_sensitivity_ind <- pp_indices$lambda_eta; ind_name <- "lambda_eta"

mcmc_subset_size <- 1000
mcmc_prior_sens_subset <- cov(draws_mat[1:mcmc_subset_size, ], log_prior_grad_mat[1:mcmc_subset_size, ])

mcmc_prior_sens <- cov(draws_mat, log_prior_grad_mat)

vb_prior_sens_mom <- GetMomentsFromVector(mp_reg, prior_sens[, prior_sensitivity_ind])
mcmc_prior_sens_mom <- GetMomentsFromVector(mp_reg, mcmc_prior_sens[, prior_sensitivity_ind])
mcmc_prior_sens_subset <- GetMomentsFromVector(mp_reg, mcmc_prior_sens_subset[, prior_sensitivity_ind])
prior_sens_results <-
  rbind(SummarizeRawMomentParameters(vb_prior_sens_mom, metric=ind_name, method="lrvb") %>%
          mutate(draws=""),
        SummarizeRawMomentParameters(mcmc_prior_sens_mom, metric=ind_name, method="mcmc") %>%
          mutate(draws=nrow(draws_mat)),
        SummarizeRawMomentParameters(mcmc_prior_sens_subset, metric=ind_name, "mcmc") %>%
          mutate(draws=subset_size)
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
  facet_grid(~ draws)



