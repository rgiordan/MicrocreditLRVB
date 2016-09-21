library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)
library(mvtnorm)
library(MicrocreditLRVB)

# Load previously computed Stan results
analysis_name <- "simulated_data_robust"
#analysis_name <- "simulated_data_nonrobust"

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/simulated_data")

r_directory <- file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/R")
source(file.path(r_directory, "density_lib.R"))

fit_file <- file.path(project_directory, paste(analysis_name, "_mcmc_and_vb.Rdata", sep=""))
print(paste("Loading fits from ", fit_file))

fit_env <- environment()
load(fit_file, envir=fit_env)
fit_env <- as.list(fit_env)

###########################################
# Extract results

pp <- fit_env$stan_results$pp

pp$mu_student_t_prior <- FALSE
pp$mu_t_loc <- 0
pp$mu_t_scale <- 1 / sqrt(pp$mu_info[1, 1])
pp$mu_t_df <- 1000

# Convenient indices
vp_indices <- GetParametersFromVector(vp_opt, as.numeric(1:vp_opt$encoded_size), FALSE)
mp_indices <- GetMomentsFromVector(mp_opt, as.numeric(1:mp_opt$encoded_size))
pp_indices <- GetPriorsFromVector(pp, as.numeric(1:pp$encoded_size))

n_mu_draws <- 100
fit_env$vb_fit$vp_opt$mu_draws <-
  qnorm(seq(1 / (n_mu_draws + 1), 1 - 1 / (n_mu_draws + 1), length.out = n_mu_draws))

#################
# Fit with a t prior

x <- stan_results$stan_dat$x
y <- stan_results$stan_dat$y
y_g <- stan_results$stan_dat$y_group

# Refit to make sure
pp$mu_t_df <- -1
pp$mu_student_t_prior <- TRUE
vb_fit <- FitVariationalModel(x, y, y_g, fit_env$vb_fit$vp_opt, pp)
vp_opt <- vb_fit$vp_opt
mp_opt <- GetMoments(vp_opt)
lrvb_terms <- GetLRVB(x, y, y_g, vp_opt, pp)
lrvb_cov <- lrvb_terms$lrvb_cov

fit_env$vb_fit$vp_opt$mu_loc
vp_opt$mu_loc

pp_perturb <- pp
pp_perturb$mu_t_df <- 2
pp_perturb$mu_student_t_prior <- TRUE

vb_fit_perturb <- FitVariationalModel(x, y, y_g, fit_env$vb_fit$vp_opt, pp_perturb, fit_bfgs=FALSE)
vp_opt_perturb <- vb_fit_perturb$vp_opt
mp_opt_perturb <- GetMoments(vp_opt_perturb)

diff_vec <- GetVectorFromMoments(mp_opt_perturb) - GetVectorFromMoments(mp_opt)
max(abs(diff_vec))

vp_opt$mu_loc
vp_opt_perturb$mu_loc


##############################################
# Calculate epsilon sensitivity


#########################################
# Monte Carlo integrate using q

# You could also do this more numerically stably with a Cholesky decomposition.
lrvb_pre_factor <- -1 * lrvb_terms$jac %*% solve(lrvb_terms$elbo_hess)

GetWeightedInfuenceFunctionVector <- function(mu) {
  mu_prior_val <- GetMuLogPrior(mu, pp)
  mu_q_res <- GetMuLogDensity(mu, vp_opt, pp, TRUE)
  mu_t_prior_val <- GetMuLogStudentTPrior(mu, pp_perturb)
  sens_pre_factor <- exp(mu_t_prior_val - mu_prior_val)
  # sens_lrvb_term <- -1 * lrvb_terms$jac %*% solve(lrvb_terms$elbo_hess, mu_q_res$grad)
  # sens <- sens_pre_factor * sens_lrvb_term
  sens <- sens_pre_factor * lrvb_pre_factor %*% mu_q_res$grad
  weight <- exp(mu_t_prior_val - mu_q_res$val)
  return(list(sens=sens, weight=weight, sens_pre_factor=sens_pre_factor))
}


weight_sum <- 0
sens_sum <- 0
q_draws <- DrawFromQMu(50000, vp_opt, rescale=1)
sens_list <- list()
for (ind in 1:nrow(q_draws)) {
  if (ind %% 100 == 0) {
    cat(".")
  }
  sens_list[[ind]] <- GetWeightedInfuenceFunctionVector(q_draws[ind, ])
}

weights <- unlist(lapply(sens_list, function(entry) { entry$weight} ))
max(weights) / median(weights)
hist(log10(weights), 1000)
sens_pre_factors <- unlist(lapply(sens_list, function(entry) { entry$sens_pre_factor} ))
sens_vec_list <- lapply(sens_list, function(entry) { as.numeric(entry$sens) } )
# sens_vec_mean <- Reduce(`+`, sens_vec_list) / sum(weights)
sens_vec_mean <- Reduce(`+`, sens_vec_list) / nrow(q_draws)
sum(weights) / nrow(q_draws)

diff_vec <- GetVectorFromMoments(mp_opt_perturb) - GetVectorFromMoments(mp_opt)

foo <-
  rbind(
    SummarizeRawMomentParameters(GetMomentsFromVector(mp_opt, sens_vec_mean), metric="mean", method="sens"),
    SummarizeRawMomentParameters(GetMomentsFromVector(mp_opt, diff_vec),      metric="mean", method="diff"))
bar <- dcast(foo, par + component + group ~ method, value.var="val")

ggplot(bar) +
  geom_point(aes(x=sens, y=diff, color=par)) +
  geom_abline((aes(intercept=0, slope=1)))



component <- mp_indices$mu_e_vec[1]; component_name <- "E_q[mu[1]]"
sens_vec_mean[component]
diff_vec[component]

foo <- unlist(lapply(sens_list, function(entry) { as.numeric(entry$sens[component]) } ))


#########################################
# Grid integrate

component <- mp_indices$mu_e_vec[1]; component_name <- "E_q[mu[1]]"

lrvb_pre_factor <- -1 * lrvb_terms$jac %*% solve(lrvb_terms$elbo_hess)

GetWeightedInfuenceFunctionVectorGrid <- function(mu) {
  mu_prior_val <- GetMuLogPrior(mu, pp)
  mu_q_res <- GetMuLogDensity(mu, vp_opt, pp, TRUE)
  mu_t_prior_val <- GetMuLogStudentTPrior(mu, pp_perturb)

  sens_pre_factor <- exp(mu_q_res$val + mu_t_prior_val - mu_prior_val)
  # cat(mu_q_res$val, " ", mu_t_prior_val, " ", mu_prior_val, "\n")
  sens_lrvb_term <- lrvb_pre_factor %*% mu_q_res$grad
  sens_vec <- sens_lrvb_term * sens_pre_factor
  # return(mu_q_res$val + mu_t_prior_val - mu_prior_val)
  # return(sens_lrvb_term[component])
  return(sens_vec[component])
  # return(exp(mu_q_res$val))
}

grid_range <- 3 * sqrt(diag(lrvb_cov)[vp_indices$mu_loc])

grid_n <- 50
mu_influence <- EvaluateOn2dGrid(FUN=GetWeightedInfuenceFunctionVectorGrid,
                                 mp_opt$mu_e_vec, -grid_range[1], grid_range[1], -grid_range[2], grid_range[2],
                                 len=grid_n)
sum(mu_influence$val) * (2 * grid_range[1] / grid_n) * (2 * grid_range[2] / grid_n)
diff_vec[component]
sens_vec_mean[component]




# val_clip <- 10
# mu_influence$val <- ifelse(abs(mu_influence$val) > val_clip, val_clip * sign(mu_influence$val), mu_influence$val)
ggplot(mu_influence) +
  geom_tile(aes(x=theta1, y=theta2, fill=exp(val))) +
  geom_point(aes(x=mp_opt$mu_e_vec[1], y=mp_opt$mu_e_vec[2], color="posterior mean"), size=2) +
  xlab("mu[1]") + ylab("mu[2]") +
  scale_fill_gradient2()






diff_vec <- GetVectorFromMoments(mp_opt_perturb) - GetVectorFromMoments(mp_opt)
plot(sens_vec_mean, diff_vec); abline(0, 1)


sens_vec_mean[component]
diff_vec[component]



##########################################
# Get functional sensitivity measures

draw <- PackMCMCSamplesIntoMoments(fit_env$stan_results$mcmc_sample, mp_opt, n_draws=1)[[1]]

#include_tau_groups <- include_mu_groups <- as.integer(c())
include_tau_groups <- include_mu_groups <- as.integer(1:(vp_opt$n_g) - 1)
q_derivs <- GetLogVariationalDensityDerivatives(
  draw, vp_opt, pp, include_mu=TRUE, include_lambda=TRUE,
  include_mu_groups, include_tau_groups, calculate_gradient=TRUE)

q_derivs$grad


###############################
# Plot the mu prior influence function

# You could also do this more numerically stably with a Cholesky decomposition.
lrvb_pre_factor <- -1 * lrvb_terms$jac %*% solve(lrvb_terms$elbo_hess)

mu <- mp_opt$mu_e_vec + c(0.1, 0.2)
GetInfluenceFunctionVector <- function(mu) {
  mu_prior_val <- GetMuLogPrior(mu, pp)
  mu_q_res <- GetMuLogDensity(mu, vp_opt, pp, TRUE)
  exp(mu_q_res$val - mu_prior_val) * lrvb_pre_factor %*% mu_q_res$grad
}

# component <- mp_indices$mu_e_vec[1]; component_name <- "E_q[mu[1]]"
# component <- mp_indices$mu_e_vec[2]; component_name <- "E_q[mu[2]]"
component <- mp_indices$lambda_e[1, 1]; component_name <- "E_q[lambda[1, 1]]"
# component <- mp_indices$lambda_e[2, 2]; component_name <- "E_q[lambda[2, 2]]"
# component <- mp_indices$lambda_e[1, 2]; component_name <- "E_q[lambda[1, 2]]"
# component <- mp_indices$tau[[1]]$e_log; component_name <- "E_q[log(tau[1])]"

GetInfluenceFunctionComponent <- function(mu) GetInfluenceFunctionVector(mu)[component]

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

width <- 2
q_mu <- EvaluateOn2dGrid(function(mu) { exp(GetMuLogDensity(mu, vp_opt, pp, FALSE)$val) },
                         mp_opt$mu_e_vec, -width, width, -width, width, len=30)
ggplot(q_mu) +
  geom_tile(aes(x=theta1, y=theta2, fill=val)) +
  geom_point(aes(x=mp_opt$mu_e_vec[1], y=mp_opt$mu_e_vec[2], color="posterior mean"), size=2) +
  scale_fill_gradient2() +
  xlab("mu[1]") + ylab("mu[2]") +
  ggtitle(paste("Influence of mu prior on ", component_name,
                "\nCentered on the posterior", sep=""))


width <- 5
mu_influence <- EvaluateOn2dGrid(GetInfluenceFunctionComponent,
                                 pp$mu_loc, -width, width, -width, width, len=40)
ggplot(mu_influence) +
  geom_tile(aes(x=theta1, y=theta2, fill=val)) +
  geom_point(aes(x=pp$mu_loc[1], y=pp$mu_loc[2], color="prior mean"), size=2) +
  xlab("mu[1]") + ylab("mu[2]") +
  scale_fill_gradient2()  +
  ggtitle(paste("Influence of mu prior on ", component_name,
                "\nCentered on the prior", sep=""))




################
# Plot


mean_results <- dcast(results, par + component + group ~ method, value.var="val")

ggplot(mean_results) +
  geom_point(aes(x=mfvb_norm, y=mfvb_t, color=par), size=3) +
  geom_abline(aes(slope=1, intercept=0))



