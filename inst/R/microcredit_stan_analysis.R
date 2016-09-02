library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)

library(MicrocreditLRVB)

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/simulated_data")
# library_location <- file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/")
# source(file.path(library_location, "inst/R/microcredit_stan_lib.R"))

# Choose one.
#analysis_name <- "simulated_data_nonrobust"
#analysis_name <- "simulated_data_robust"
analysis_name <- "simulated_data_lambda_beta"

set.seed(42)

##########################
# Prior parameters

# The dimension of the regressors.
k <- 2

pp <- GetEmptyPriors(k)
pp[["mu_loc"]] <- rep(0, k)
mu_prior_sd <- 3
pp[["mu_info"]] <- matrix(c(mu_prior_sd ^ -2, 0., 0, mu_prior_sd ^ -2), k, k)
pp[["lambda_eta"]] <- 15.01
pp[["lambda_alpha"]] <- 20.01
pp[["lambda_beta"]] <- 20.01
pp[["tau_alpha"]] <- 2.01
pp[["tau_beta"]] <- 2.01


#############################
# Simualate some data
true_params <- list()

# Set parameters similar to the microcredit data.  Note that the true mean is
# an unlikely value relative to the prior.  This will result in a non-robust
# posterior.

if (analysis_name == "simulated_data_nonrobust") {
  true_params$true_mu <- c(4 * mu_prior_sd, -4 * mu_prior_sd)
} else {
  true_params$true_mu <- c(1 * mu_prior_sd, -1 * mu_prior_sd)
}
true_params$true_sigma <- matrix(c(12, 0, 0, 12), 2, 2)
true_params$true_lambda <- solve(true_params$true_sigma)
true_params$true_tau <- 1e-2

# Number of groups
n_g <- 30

# Number of data points per group
n_per_group <- 100

sim_data <- SimulateData(true_params, n_g, n_per_group)
x <- sim_data$x
y_g <- sim_data$y_g
y <- sim_data$y
true_params$true_mu_g <- sim_data$true_mu_g

# Sanity checks
mu_g_mat <- do.call(rbind, true_params$true_mu_g)
cov(mu_g_mat)
solve(true_params$true_lambda)


######################################
# STAN

# Load the STAN model
stan_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/stan")
stan_model_name <- "basic_hierarchical_model_lkj_priors"
model_file_rdata <-
  file.path(stan_directory, paste(stan_model_name, "Rdata", sep="."))
if (file.exists(model_file_rdata)) {
  print("Loading pre-compiled Stan model.")
  load(model_file_rdata)
} else {
  print("Compiling Stan model.")
  model_file <-
    file.path(stan_directory, paste(stan_model_name, "stan", sep="."))
  model <- stan_model(model_file)
  save(model, file=model_file_rdata)
}



# Perturb the prior.
pp_perturb <- pp
if (analysis_name == "simulated_data_lambda_beta") {
  perturb_epsilon <- 1.0
  pp_perturb$lambda_beta <- pp_perturb$lambda_beta + perturb_epsilon
} else {
  perturb_epsilon <- 0.05
  mu_prior_info_perturb <- pp$mu_info
  mu_prior_info_perturb[1,2] <- mu_prior_info_perturb[2,1] <-
    mu_prior_info_perturb[1,2] + perturb_epsilon
  pp_perturb$mu_info <- mu_prior_info_perturb
}



# Stan data.
SetStanDat <- function(prior_params) {
  list(NG = n_g,
       N = length(y),
       K = ncol(x),
       y_group = y_g,
       y = y,
       x = x,
       mu_prior_sigma = solve(prior_params$mu_info),
       mu_prior_mean = prior_params$mu_loc,
       use_mu1_prior = FALSE,
       mu1_prior_sigma = solve(prior_params$mu_info),
       mu1_prior_mean = prior_params$mu_loc,
       scale_prior_alpha = prior_params$lambda_alpha,
       scale_prior_beta = prior_params$lambda_beta,
       lkj_prior_eta = prior_params$lambda_eta,
       tau_prior_alpha = prior_params$tau_alpha,
       tau_prior_beta = prior_params$tau_beta)
}

stan_dat <- SetStanDat(pp)
stan_dat_perturbed <- SetStanDat(pp_perturb)


# Some knobs we can tweak.  Note that we need many iterations to accurately assess
# the prior sensitivity in the MCMC noise.
chains <- 1
iters <- 10000
seed <- 42

# Note: this takes a while.
stan_draws_file <-
  file.path(project_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))
mcmc_time <- Sys.time()
stan_sim <- sampling(model, data=stan_dat, seed=seed, chains=chains, iter=iters)
mcmc_time <- Sys.time() - mcmc_time
stan_sim_perturb <- sampling(model, data=stan_dat_perturbed, seed=seed, chains=chains, iter=iters)

stan_advi <- vb(model, data =stan_dat,  algorithm="meanfield", output_samples=iters)
stan_advi_perturb <- vb(model, data=stan_dat_perturbed,  algorithm="meanfield", output_samples=iters)
stan_advi_full <- vb(model, data=stan_dat,  algorithm="fullrank", output_samples=iters)
stan_advi_full_perturb <- vb(model, data=stan_dat_perturbed,  algorithm="meanfield", output_samples=iters)

save(stan_sim, stan_sim_perturb, mcmc_time, perturb_epsilon,
     stan_dat, stan_dat_perturbed, true_params, pp, pp_perturb,
     stan_advi, stan_advi_perturb, stan_advi_full, stan_advi_full_perturb,
     file=stan_draws_file)


