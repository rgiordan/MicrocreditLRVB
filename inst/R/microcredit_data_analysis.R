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
mp_opt <- GetMoments(vp_opt)
mfvb_cov <- GetCovariance(vp_opt)
lrvb_terms <- GetLRVB(x, y, y_g, vp_opt, pp)
lrvb_cov <- lrvb_terms$lrvb_cov
prior_sens <- GetSensitivity(vp_opt, pp, lrvb_terms$jac, lrvb_terms$elbo_hess)

# Calculate vb perturbed estimates
mp_opt_vec <- GetVectorFromMoments(mp_opt)
mu_info_offdiag_sens <- prior_sens[, pp_indices$mu_info[1, 2]]
mp_opt_vec_pert <- mp_opt_vec + stan_results$perturb_epsilon * mu_info_offdiag_sens
mp_opt_pert <- GetMomentsFromVector(mp_opt, mp_opt_vec_pert)


##########################################
# Get functional sensitivity measures

mcmc_sample <- extract(stan_results$stan_sim)
draw <- PackMCMCSamplesIntoMoments(mcmc_sample, mp_reg, n_draws=1)[[1]]

#include_tau_groups <- include_mu_groups <- as.integer(c())
include_tau_groups <- include_mu_groups <- as.integer(1:(vp_opt$n_g) - 1)
q_derivs <- GetLogVariationalDensityDerivatives(
    draw, vp_opt, pp, include_mu=TRUE, include_lambda=TRUE,
    include_mu_groups, include_tau_groups, calculate_gradient=TRUE)

q_derivs$grad

###############################
# Get the mu prior influence function

library(mvtnorm)

GetMuLogPrior <- function(mu) {
  # You can't use the VB priors because they are
  # (1) a function of the natural parameters whose variance would have to be zero and
  # (2) not normalized.
  dmvnorm(mu, mean=pp$mu_loc, sigma=solve(pp$mu_info), log=TRUE)
}


GetMuLogDensity <- function(mu, calculate_gradient) {
  draw_local <- draw  
  draw_local$mu_e_vec <- mu
  include_tau_groups <- include_mu_groups <- as.integer(c())
  q_derivs <- GetLogVariationalDensityDerivatives(
    draw_local, vp_opt, pp, include_mu=TRUE, include_lambda=FALSE,
    include_mu_groups, include_tau_groups, calculate_gradient=calculate_gradient)
  return(q_derivs)
}


lrvb_pre_factor <- -1 * lrvb_terms$jac %*% solve(lrvb_terms$elbo_hess)

mu <- mp_opt$mu_e_vec + c(0.1, 0.2)
GetInfluenceFunctionVector <- function(mu) {
  mu_prior_val <- GetMuLogPrior(mu)
  mu_q_res <- GetMuLogDensity(mu, TRUE)
  exp(mu_q_res$val - mu_prior_val) * lrvb_pre_factor %*% mu_q_res$grad
}

system.time(GetInfluenceFunctionComponent(mu))

component <- mp_indices$mu_e_vec[1]; component_name <- "E_q[mu[1]]"
component <- mp_indices$mu_e_vec[2]; component_name <- "E_q[mu[2]]"
component <- mp_indices$lambda_e[1, 1]; component_name <- "E_q[lambda[1, 1]]"
component <- mp_indices$lambda_e[2, 2]; component_name <- "E_q[lambda[2, 2]]"
component <- mp_indices$lambda_e[1, 2]; component_name <- "E_q[lambda[1, 2]]"
GetInfluenceFunctionComponent <-
  function(mu) GetInfluenceFunctionVector(mu)[component]

width <- 2
mu_influence <- EvaluateOn2dGrid(GetInfluenceFunctionComponent,
                                 mp_opt$mu_e_vec, -width, width, -width, width, len=30)
ggplot(mu_influence) +
  geom_tile(aes(x=theta1, y=theta2, fill=val)) +
  geom_point(aes(x=mp_opt$mu_e_vec[1], y=mp_opt$mu_e_vec[2], color="posterior mean"), size=2) +
  scale_fill_gradient2() +
  xlab("mu[1]") + ylab("mu[2]") +
  ggtitle(paste("Influence of mu prior on ", component_name,
                "\nCentered on the posterior", sep=""))


width <- 15
mu_influence <- EvaluateOn2dGrid(GetInfluenceFunctionComponent,
                                 pp$mu_loc, -width, width, -width, width, len=30)
ggplot(mu_influence) +
  geom_tile(aes(x=theta1, y=theta2, fill=val)) +
  geom_point(aes(x=pp$mu_loc[1], y=pp$mu_loc[2], color="prior mean"), size=2) +
  xlab("mu[1]") + ylab("mu[2]") +
  scale_fill_gradient2()  +
  ggtitle(paste("Influence of mu prior on ", component_name,
                "\nCentered on the prior", sep=""))



##########################################
# Get MCMC sensitivity measures

mcmc_sample <- extract(stan_results$stan_sim)
mcmc_sample_perturbed <- extract(stan_results$stan_sim_perturb)
mp_draws <- PackMCMCSamplesIntoMoments(mcmc_sample, mp_reg) # A little slow

draws_mat <- do.call(rbind, lapply(mp_draws, GetVectorFromMoments))
log_prior_grad_list <- GetMCMCLogPriorDerivatives(mp_draws, pp)
log_prior_grad_mat <- do.call(rbind, log_prior_grad_list)


###########################
# Summarize results

# Pack the standard deviations into readable forms.
mfvb_sd <- GetMomentsFromVector(mp_opt, sqrt(diag(mfvb_cov)))
lrvb_sd <- GetMomentsFromVector(mp_opt, sqrt(diag(lrvb_cov)))

results_vb <- SummarizeMomentParameters(mp_opt, mfvb_sd, lrvb_sd)
results_vb_pert <- SummarizeMomentParameters(mp_opt_pert, mfvb_sd, lrvb_sd)
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


#######################
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
  facet_grid(~ draws)



