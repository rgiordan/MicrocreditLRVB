library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/simulated_data")

analysis_name <- "real_data_t_prior"
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

#################
# Get the sensitivity using the MCMC draws from before and using stan's log model gradients.

mcmc_draws <- extract(stan_results$sim)
num_draws <- length(mcmc_draws$lp__)
weight_grad_mat <- matrix(NA, nrow=num_draws, ncol=stan_results$dat$N)

# Load the data object containing the priors and data.
stan_dat <- mcmc_environment$results$original$dat

# This is a "fit object" -- we won't use it for drawing, but instead use it for calculating log gradients.
# You can ignore the error.  It seems like there should be a way to get a fit object without calling
# sampling(), but I don't know it.
data_sensitivity_fit_obj <- sampling(model, stan_dat, iter=1, chains=1)

# These should be the prior param names.
param_names <- data_sensitivity_fit_obj@.MISC$stan_fit_instance$unconstrained_param_names(FALSE, FALSE)

# These are the rows of the gradient that will correspond to the weights.
weight_param_rows <- grepl("^weight", param_names)

# Evaluate the gradient at equal weights on every point
id_weights <- rep(1.0, stan_dat$N)

# Extract the gradients with respect to the weights for each draw.
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
  weight_grad_mat[draw, ] <-
    grad_log_prob(data_sensitivity_fit_obj, pars_free)[weight_param_rows]
}
close(prog_bar)


# The covariance is the sensitivity.
# mu_sens1 = dE[mu] / dw (where mu is the mu in the paper)
# mu_sens2 = dE[tau] / dw (where tau is the tau in the paper)
mu_draws <- mcmc_draws$mu
mu_weight_sens <- cov(mu_draws, weight_grad_mat)
mu_sd <- sqrt(diag(cov(mu_draws, mu_draws)))
graph_df <- data.frame(mu_sens1=mu_weight_sens[1,] / mu_sd[1], mu_sens2=mu_weight_sens[2,] / mu_sd[2], y=y, y_g=y_g, treat=x[, 2])


# View the effect on tau and mu as a function of y for each of the seven groups.
ggplot(graph_df) +
  geom_line(aes(x=y, y=mu_sens1, color="mu")) +
  geom_point(aes(x=y, y=mu_sens1, color="mu")) +
  geom_line(aes(x=y, y=mu_sens2, color="tau")) +
  geom_point(aes(x=y, y=mu_sens2, color="tau")) +
  facet_grid(treat ~ y_g)


ggplot(filter(graph_df, treat==1, y_g==1)) +
  #geom_line(aes(x=y, y=mu_sens1)) +
  geom_point(aes(x=y, y=mu_sens1), size=2) +
  xlab("Data value") + ylab("Sensitivity") +
  geom_hline(aes(yintercept=0)) +
  ggtitle(TeX("Sensitivity of $\\mu$ to individual data points"))