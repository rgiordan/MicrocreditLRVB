library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)

library(MicrocreditLRVB)

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/simulated_data")

raw_data_directory <- file.path(Sys.getenv("GIT_REPO_LOC"), "microcredit_vb/data")

# Choose one.
# analysis_name <- "real_data_informative_priors"
analysis_name <- "real_data_t_prior"

stan_draws_file <- file.path(project_directory, sprintf("%s_data_and_mcmc_draws.Rdata", analysis_name))

set.seed(42)


# Read in the data that was produced by the R script.
csv_data <- read.csv(file.path(raw_data_directory, "microcredit_data_processed.csv"))

# The dimension of the group means.  Here, the first dimension
# is the control effect and the second dimension is the treatment.
k <- 2

# The number of distinct groups.
n_g <- max(csv_data$site)

# Get the observations and the total number of observations.
y <- csv_data$profit
y_g <- as.integer(csv_data$site)

# The x array will indicate which rows should also get the
# treatment effect.  The model accomodates generic x, but for
# this application we only need indicators.
x <- cbind(rep(1, length(y)), as.numeric(csv_data$treatment))


pp <- GetEmptyPriors(k)
if (analysis_name == "real_data_informative_priors") {
  pp[["mu_loc"]] <- rep(0, k)
  pp[["mu_info"]] <- diag(c(0.3, 0.2))
  pp[["lambda_eta"]] <- 15.01
  pp[["lambda_alpha"]] <- 20.01
  pp[["lambda_beta"]] <- 20.01
  pp[["tau_alpha"]] <- 2.01
  pp[["tau_beta"]] <- 2.01
  
  # Perturb the prior.
  # Regrettably epsilon is being used two ways here -- both as the size of the
  # perturbation of the prior parameters, and of the mixture between the perturbed
  # and original prior.
  
  pp_perturb <- pp
  perturb_epsilon <- 0.001
  mu_prior_info_perturb <- pp$mu_info
  mu_prior_info_perturb[1,2] <- mu_prior_info_perturb[2,1] <-
    mu_prior_info_perturb[1,2] + perturb_epsilon
  pp_perturb$mu_info <- mu_prior_info_perturb
  stopifnot(min(eigen(pp_perturb$mu_info)$values) > 0)
  
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
         mu_prior_df = pp$mu_t_df,
         mu_prior_use_t_contamination = FALSE,
         mu_epsilon = pp$epsilon,
         scale_prior_alpha = pp$lambda_alpha,
         scale_prior_beta = pp$lambda_beta,
         lkj_prior_eta = pp$lambda_eta,
         tau_prior_alpha = pp$tau_alpha,
         tau_prior_beta = pp$tau_beta)

  stan_dat_perturbed <- stan_dat
  stan_dat_perturbed$mu_prior_sigma <- solve(pp_perturb$mu_info)

} else if (analysis_name == "real_data_t_prior") {
  pp[["mu_loc"]] <- rep(0, k)
  pp[["mu_info"]] <- diag(c(0.3, 0.2))
  pp[["lambda_eta"]] <- 15.01
  pp[["lambda_alpha"]] <- 20.01
  pp[["lambda_beta"]] <- 20.01
  pp[["tau_alpha"]] <- 2.01
  pp[["tau_beta"]] <- 2.01

  pp[["monte_carlo_prior"]] <- TRUE
  pp[["epsilon"]] <- 1
  pp[["mu_t_loc"]] <- 0
  pp[["mu_t_df"]] <- 1
  pp[["mu_t_scale"]] <- 10
  
  # Perturb the prior.
  pp_perturb <- pp
  perturb_epsilon <- 1
  pp_perturb[["lambda_beta"]] <- pp[["lambda_beta"]] + perturb_epsilon
  
  # Stan data.
  k <- ncol(x)
  stan_dat <-
    list(NG = n_g,
         N = length(y),
         K = ncol(x),
         y_group = y_g,
         y = y,
         x = x,
         mu_prior_sigma = solve(pp$mu_info),
         mu_prior_mean = pp$mu_loc,
         mu_prior_sigma_c = diag(rep(pp$mu_t_scale ^ 2, k)),
         mu_prior_mean_c = rep(pp$mu_t_loc, k),
         mu_prior_df = pp$mu_t_df,
         mu_prior_use_t_contamination = TRUE,
         mu_epsilon = pp$epsilon,
         scale_prior_alpha = pp$lambda_alpha,
         scale_prior_beta = pp$lambda_beta,
         lkj_prior_eta = pp$lambda_eta,
         tau_prior_alpha = pp$tau_alpha,
         tau_prior_beta = pp$tau_beta)
  
  stan_dat_perturbed <- stan_dat
  stan_dat_perturbed$scale_prior_beta <- pp_perturb$lambda_beta
} else {
  stop("Bad analysis name")
}


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



# Some knobs we can tweak.  Note that we need many iterations to accurately assess
# the prior sensitivity in the MCMC noise.
chains <- 1
iters <- 5000
seed <- 42

SampleFromStanDat <- function(local_stan_dat, analyis_name) {
  mcmc_time <- Sys.time()
  stan_sim <- sampling(model, data=local_stan_dat, seed=seed, chains=chains, iter=iters)
  mcmc_time <- Sys.time() - mcmc_time
  return(list(mcmc_time=mcmc_time, sim=stan_sim, dat=local_stan_dat, analysis_name=analysis_name))
}

results <- list()

results[["original"]] <- SampleFromStanDat(stan_dat)
results[["perturbed"]] <- SampleFromStanDat(stan_dat_perturbed)

# Make sure that something changed.
print(results[["original"]]$sim, "mu")
print(results[["perturbed"]]$sim, "mu")

print(sprintf("Saving to %s.", stan_draws_file))
save(results, pp, pp_perturb, perturb_epsilon, file=stan_draws_file)

