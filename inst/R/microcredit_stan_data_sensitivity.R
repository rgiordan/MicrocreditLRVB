library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)

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

x <- stan_results$dat$x
y <- stan_results$dat$y
y_g <- stan_results$dat$y_group

pp <- mcmc_environment$pp

# Load the STAN model
stan_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/stan")
stan_model_name <- "basic_hierarchical_model_lkj_priors_data_sensitivity"
model_file_rdata <-
  file.path(stan_directory, paste(stan_model_name, "Rdata", sep="."))
if (file.exists(model_file_rdata)) {
  print("Loading pre-compiled Stan model.")
  load(model_file_rdata)
} else {
  print("Compiling Stan model.")
  model_file <- file.path(stan_directory, "basic_hierarchical_model_lkj_priors_lik.stan")
  model <- stan_model(model_file)
  save(model, file=model_file_rdata)
}

mcmc_draws <- extract(stan_results$sim)
num_draws <- length(mcmc_draws$lp__)
weight_grad_mat <- matrix(NA, nrow=num_draws, ncol=stan_results$dat$N)

# Load the data object containing the priors and data.
stan_dat <- mcmc_environment$results$original$dat

# This is a "fit object" -- we won't use it for drawing, but instead use it for log gradients.
data_sensitivity_fit_obj <- sampling(model, stan_dat, iter=1, chains=1)

# These should be the prior param names
param_names <- data_sensitivity_fit_obj@.MISC$stan_fit_instance$unconstrained_param_names(FALSE, FALSE)

# These are the rows of the gradient that will correspond to the weights.
weight_param_rows <- grepl("^weight", param_names)

# Evaluate the gradient at equal weights on every point
id_weights <- rep(1.0, stan_dat$N)

prog_bar <- txtProgressBar(min=1, max=num_draws, style=3)
for (draw in 1:num_draws) {
  setTxtProgressBar(prog_bar, value=draw)
  mcmc_draw_dat <- list(mu=mcmc_draws$mu[draw, ],
                        mu1=mcmc_draws$mu1[draw,,],
                        sigma_y=mcmc_draws$sigma_y[draw,],
                        R=mcmc_draws$R[draw,,],
                        S=mcmc_draws$S[draw,],
                        weights=id_weights)
  pars_free <- unconstrain_pars(data_sensitivity_fit_obj, mcmc_draw_dat)
  weight_grad_mat[draw, ] <- grad_log_prob(data_sensitivity_fit_obj, pars_free, adjust_transform=FALSE)[weight_param_rows]
}
close(prog_bar)


mu_draws <- mcmc_draws$mu
mu_weight_sens <- cov(mu_draws, weight_grad_mat)

graph_df <- data.frame(mu=mu_weight_sens[1,], tau=mu_weight_sens[1,])

summary(graph_df$tau)
hist(graph_df$tau, 1000)


