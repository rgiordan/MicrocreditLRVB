library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)
library(mvtnorm)

library(MicroCreditLRVB)

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/simulated_data")

analysis_name <- "simulated_data3"

set.seed(42)

##########################
# Prior parameters

pp <- list()
pp[["k"]] <- k
pp[["mu_mean"]] <- rep(0, k)
pp[["mu_info"]] <- matrix(c(0.03, 0., 0, 0.02), k, k)
pp[["lambda_eta"]] <- 15.01
pp[["lambda_alpha"]] <- 20.01
pp[["lambda_beta"]] <- 20.01
pp[["tau_alpha"]] <- 2.01
pp[["tau_beta"]] <- 2.01

# Optimization parameters stored in the prior:
pp[["lambda_diag_min"]] <- 1e-10
pp[["lambda_n_min"]] <- k + 0.5

#############################
# Simualate some data
true_params <- list()

# Set parameters similar to the microcredit data.  Note that the true mean is
# an unlikely value relative to the prior.  This will result in a non-robsut
# posterior.
true_params$true_mu <- c(15, 15)
true_params$true_sigma <- matrix(c(12, 0, 0, 12), 2, 2)
true_params$true_lambda <- solve(true_params$true_sigma)
true_params$true_tau <- 1e-4

# Number of groups
n_g <- 7

# Number of data points per group
n_per_group <- 100

sim_data <- SimulateData(true_params, n_g, n_per_group)
x <- sim_data$x
y_g <- sim_data$y_g
y <- sim_data$y



# The dimension of the explanatory variables.
k <- ncol(x)

# Sanity checks
mu_g_mat <- do.call(rbind, true_params$true_mu_g_vec)
solve(cov(mu_g_mat))
true_params$true_lambda

g <- 1
g_reg <- lm(y ~ x1 + x2 - 1,
            data.frame(y=y_vec[[g]], x1=x_vec[[g]][, 1], x2=x_vec[[g]][, 2]))
summary(g_reg)
true_params$true_mu_g_vec[[g]]
1 / var(g_reg$residuals)
true_params$true_tau


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

# Stan data.
stan_dat <- list(NG = n_g,
                 N = length(y),
                 K = ncol(x),
                 y_group = y_g,
                 y = y,
                 x = x,
                 mu_prior_sigma = solve(pp$mu_info),
                 mu_prior_mean = pp$mu_mean,
                 use_mu1_prior = FALSE,
                 mu1_prior_sigma = solve(pp$mu_info),
                 mu1_prior_mean = pp$mu_mean,
                 scale_prior_alpha = pp$lambda_alpha,
                 scale_prior_beta = pp$lambda_beta,
                 lkj_prior_eta = pp$lambda_eta,
                 tau_prior_alpha = pp$tau_alpha,
                 tau_prior_beta = pp$tau_beta)

perturb_epsilon <- 0.01
stan_dat_perturbed <- stan_dat
mu_prior_info_perturb <- pp$mu_info
mu_prior_info_perturb[1,2] <- mu_prior_info_perturb[2,1] <-
  mu_prior_info_perturb[1,2] + perturb_epsilon
stan_dat_perturbed$mu_prior_sigma <- solve(mu_prior_info_perturb)
stan_dat$mu_prior_sigma

# some knobs we can tweak
chains <- 1
iters <- 10000
control <- list(adapt_t0 = 10,       # default = 10
                stepsize = 1,        # default = 1
                max_treedepth = 6)   # default = 10
seed <- 42

# Note: this takes a while.
stan_draws_file <-
  file.path(project_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))
mcmc_time <- Sys.time()
stan_sim <- sampling(model, data = stan_dat, seed = seed,
                     chains = chains, iter = iters, control = control)
mcmc_time <- Sys.time() - mcmc_time
stan_sim_perturb <- sampling(model, data = stan_dat_perturbed, seed = seed,
                             chains = chains, iter = iters, control = control)
save(stan_sim, stan_sim_perturb, mcmc_time, perturb_epsilon,
     stan_dat, stan_dat_perturbed, true_params, pp,
     file=stan_draws_file)

