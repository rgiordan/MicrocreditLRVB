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
analysis_name <- "simulated_data_nonrobust"
#analysis_name <- "simulated_data_nonrobust_t_perturb"
#analysis_name <- "simulated_data_robust"
#analysis_name <- "simulated_data_lambda_beta"

stan_draws_file <- file.path(project_directory, sprintf("%s_data_and_mcmc_draws.Rdata", analysis_name))

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
  model_file <- file.path(stan_directory, paste(stan_model_name, "stan", sep="."))
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
stan_dat <-
  list(NG = n_g,
       N = length(y),
       K = ncol(x),
       y_group = y_g,
       y = y,
       x = x,
       mu_prior_sigma = solve(pp$mu_info),
       mu_prior_mean = pp$mu_loc,
       mu_prior_sigma_c = solve(pp_perturb$mu_info),
       mu_prior_mean_c = pp_perturb$mu_loc,
       mu_prior_df = 1,
       mu_prior_use_t_contamination = 0,
       mu_epsilon = 0,
       scale_prior_alpha = pp$lambda_alpha,
       scale_prior_beta = pp$lambda_beta,
       lkj_prior_eta = pp$lambda_eta,
       tau_prior_alpha = pp$tau_alpha,
       tau_prior_beta = pp$tau_beta)


# Some knobs we can tweak.  Note that we need many iterations to accurately assess
# the prior sensitivity in the MCMC noise.
chains <- 1
iters <- 10000
seed <- 42

SampleFromStanDat <- function(local_stan_dat, analyis_name) {
  mcmc_time <- Sys.time()
  stan_sim <- sampling(model, data=local_stan_dat, seed=seed, chains=chains, iter=iters)
  mcmc_time <- Sys.time() - mcmc_time
  return(list(mcmc_time=mcmc_time, sim=stan_sim, dat=local_stan_dat, analysis_name=analysis_name))
}

EpsilonName <- function(epsilon) {
  sprintf("epsilon_%f", epsilon)
}

results <- list()

# for (epsilon in seq(0, 0.001, length.out=20)) {
#   cat("\n\n", epsilon, "\n")
#   stan_dat$mu_epsilon <- epsilon
#   analysis <- EpsilonName(epsilon)
#   results[[analysis]] <- SampleFromStanDat(stan_dat)
#   print(results[[analysis]]$sim, "mu")
# }

epsilon <- 0
stan_dat$mu_epsilon <- epsilon
results[[EpsilonName(epsilon)]] <- SampleFromStanDat(stan_dat)

epsilon <- 1
stan_dat$mu_epsilon <- epsilon
results[[EpsilonName(epsilon)]] <- SampleFromStanDat(stan_dat)

# Make sure that 
print(results[[EpsilonName(0)]]$sim, "mu")
print(results[[EpsilonName(1)]]$sim, "mu")

print(sprintf("Saving to %s.", stan_draws_file))
save(results, pp, pp_perturb, file=stan_draws_file)



########################
# Investigate the results

foo <- list()
for (analysis in names(results)) {
  res <- results[[analysis]]
  mu <- get_posterior_mean(res[["sim"]], "mu")[1]
  epsilon <- res$dat$mu_epsilon
  foo[[length(foo) + 1]] <- data.frame(epsilon=epsilon, mu_1=mu, time=res$mcmc_time, analysis=analysis)
}
mu_eps_df <- do.call(rbind, foo)
ggplot(filter(mu_eps_df, epsilon < 1)) + geom_point(aes(x=epsilon, y=mu_1))


# Look at the weights
orig_res <- results[[filter(mu_eps_df, epsilon == 0)[["analysis"]] ]]
contam_res <- results[[filter(mu_eps_df, epsilon == 1)[["analysis"]] ]]

orig_draws <- extract(orig_res$sim)
weights <- exp(orig_draws$mu_log_prior_c - orig_draws$mu_log_prior)
weights <- length(weights) * weights / sum(weights)
mu1_draws <- orig_draws$mu[,1]

get_posterior_mean(orig_res$sim, "mu")[1]
mean(mu1_draws)

get_posterior_mean(contam_res$sim, "mu")[1]
mean(mu1_draws * weights)

weight_dist <- data.frame(num=(length(weights):1) / length(weights), w=sort(weights))


# The power law coefficient 
w_coeff <- coefficients(lm(log10(num) ~ log10(w), data=filter(weight_dist, w > quantile(weight_dist$w, 0.8))))
alpha <- -1 * w_coeff["log(w)"] + 1
ggplot(weight_dist) +
  geom_point(aes(x=log10(w), y=log10(num))) +
  geom_abline(intercept=w_coeff[1], slope=w_coeff[2]) +
  xlab("log10(Weight)") + ylab("log10(1 - Empirical CDF)")
