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

fit_file <- file.path(project_directory, paste(analysis_name, "_mcmc_and_vb.Rdata", sep=""))
print(paste("Loading fits from ", fit_file))

fit_env <- environment()
load(fit_file, envir=fit_env)
fit_env <- as.list(fit_env)

###########################################
# Extract results

vp_opt <- fit_env$vb_fit$vp_opt

lrvb_cov <- fit_env$lrvb_terms$lrvb_cov
prior_sens <- fit_env$prior_sens

mp_opt <- GetMoments(vp_opt)
mfvb_cov <- GetCovariance(vp_opt)

pp <- fit_env$stan_results$pp

# Convenient indices
vp_indices <- GetParametersFromVector(vp_opt, as.numeric(1:vp_opt$encoded_size), FALSE)
mp_indices <- GetMomentsFromVector(mp_opt, as.numeric(1:mp_opt$encoded_size))
pp_indices <- GetPriorsFromVector(pp, as.numeric(1:pp$encoded_size))


##########################################
# Get functional sensitivity measures

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


# You could also do this more numerically stably with a Cholesky decomposition.
lrvb_pre_factor <- -1 * lrvb_terms$jac %*% solve(lrvb_terms$elbo_hess)

mu <- mp_opt$mu_e_vec + c(0.1, 0.2)
GetInfluenceFunctionVector <- function(mu) {
  mu_prior_val <- GetMuLogPrior(mu)
  mu_q_res <- GetMuLogDensity(mu, TRUE)
  exp(mu_q_res$val - mu_prior_val) * lrvb_pre_factor %*% mu_q_res$grad
}

component <- mp_indices$mu_e_vec[1]; component_name <- "E_q[mu[1]]"
component <- mp_indices$mu_e_vec[2]; component_name <- "E_q[mu[2]]"
component <- mp_indices$lambda_e[1, 1]; component_name <- "E_q[lambda[1, 1]]"
component <- mp_indices$lambda_e[2, 2]; component_name <- "E_q[lambda[2, 2]]"
component <- mp_indices$lambda_e[1, 2]; component_name <- "E_q[lambda[1, 2]]"
GetInfluenceFunctionComponent <-
  function(mu) GetInfluenceFunctionVector(mu)[component]

width <- 2
mu_influence <- EvaluateOn2dGrid(GetInfluenceFunctionComponent,
                                 mp_opt$mu_e_vec, -width, width, -width, width, len=50)
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


