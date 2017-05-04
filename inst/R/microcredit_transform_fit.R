library(ggplot2)
library(dplyr)
library(reshape2)
library(Matrix)
library(rstan)

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/simulated_data")
load(file=file.path(project_directory, "boxcox_data.Rdata"))

head(data_df_transform)


# Load the STAN model
stan_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/stan")
stan_model_name <- "hierarchical_model_spike"
model_file_rdata <-
  file.path(stan_directory, paste(stan_model_name, "Rdata", sep="."))
recompile <- TRUE
if (file.exists(model_file_rdata) && !recompile) {
  print("Loading pre-compiled Stan model.")
  load(model_file_rdata)
} else {
  print("Compiling Stan model.")
  model_file <- file.path(stan_directory, paste(stan_model_name, "stan", sep="."))
  model <- stan_model(model_file)
  save(model, file=model_file_rdata)
}

y <- data_df_transform$y
x <- cbind(data_df_transform$x.1, data_df_transform$x.2)

GetGroupSummaryStats <- function(g) {
  rows <- (data_df_transform$y_g == g) & (!data_df_transform$zero_y)
  xtx <- t(x[rows, ]) %*% x[rows, ]
  ytx <- as.numeric(t(y[rows]) %*% x[rows, ])
  yty <- t(y[rows]) %*% y[rows]
  num_obs <- sum(rows)
  
  zero_rows <- (data_df_transform$y_g == g) & data_df_transform$zero_y
  num_arm_zero <- with(data_df_transform[zero_rows, ], c(sum(x.2 == 1), sum(x.2 == 0)))
  num_arm_obs <- with(data_df_transform[data_df_transform$y_g == g, ], c(sum(x.2 == 1), sum(x.2 == 0)))
  list(xtx=xtx, ytx=ytx, yty=yty, num_obs=num_obs, num_arm_zero=num_arm_zero, num_arm_obs=num_arm_obs)
}

num_groups <- max(data_df_transform$y_g)
summary_stats <- lapply(1:num_groups, GetGroupSummaryStats)

k <- 2
stan_dat <- list(
  xtx = lapply(1:num_groups, function(g) { summary_stats[[g]]$xtx }),
  ytx = lapply(1:num_groups, function(g) { summary_stats[[g]]$ytx }),
  yty = sapply(1:num_groups, function(g) { summary_stats[[g]]$yty }),
  num_obs = sapply(1:num_groups, function(g) { summary_stats[[g]]$num_obs }),
  num_arm_zero = lapply(1:num_groups, function(g) { summary_stats[[g]]$num_arm_zero }),
  num_arm_obs = lapply(1:num_groups, function(g) { summary_stats[[g]]$num_arm_obs }),
  NG=num_groups,
  K=k,
  num_zeros=matrix(0, num_groups, 2),
  beta_prior_mean=rep(0, k),
  beta_prior_sigma=1000 * diag(k),
  tau_prior_alpha=2.1,
  tau_prior_beta=2.1,
  scale_prior_alpha=2.1,
  scale_prior_beta=2.1,
  lkj_prior_eta=15  
)

chains <- 1
iters <- 1000
seed <- 42

mcmc_time <- Sys.time()
stan_sim <- sampling(model, data=stan_dat, seed=seed, chains=chains, iter=iters)
mcmc_time <- Sys.time() - mcmc_time

print(stan_sim)

# Sanity check
filter(data_df_transform, !zero_y) %>%
  group_by(y_g) %>%
  summarise(y_sd=sd(y)) %>%
  mutate(stan_sd=sqrt(apply(extract(stan_sim, pars="sigma_y")$sigma_y, 2, median)))

beta_draws <- extract(stan_sim, pars="beta_group")$beta_group
apply(beta_draws, mean, MARGIN=c(2, 3))
filter(data_df_transform, !zero_y) %>%
  group_by(y_g) %>%
  summarise(y_mean=mean(y))
  