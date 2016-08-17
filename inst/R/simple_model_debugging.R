library(ggplot2)
library(dplyr)
library(reshape2)
library(Matrix)
library(mvtnorm)

library(LRVBUtils)

library(MicrocreditLRVB)
library_location <- file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/")
source(file.path(library_location, "inst/R/microcredit_stan_lib.R"))

library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)
library(mvtnorm)

library(MicrocreditLRVB)

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/simulated_data")

analysis_name <- "simulated_data_easy"

set.seed(42)

##########################
# Prior parameters

# The dimension of the regressors.
k <- 2

pp <- list()
pp[["k_reg"]] <- k
pp[["mu_loc"]] <- rep(0, k)
pp[["mu_info"]] <- matrix(c(0.02, 0., 0, 0.02), k, k)
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
true_params$true_mu <- c(0, 1)
true_params$true_sigma <- matrix(c(12, 0, 0, 12), 2, 2)
true_params$true_lambda <- solve(true_params$true_sigma)
true_params$true_tau <- 1 / (0.01^2)

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
