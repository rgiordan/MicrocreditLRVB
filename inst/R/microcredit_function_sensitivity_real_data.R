library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)
library(mvtnorm)
library(MicrocreditLRVB)
library(LRVBUtils)
library(gridExtra)

# Load previously computed Stan results
analysis_name <- "real_data_t_prior"

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/simulated_data")

r_directory <- file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/R")
source(file.path(r_directory, "density_lib.R"))

fit_file <- file.path(project_directory, paste(analysis_name, "_mcmc_and_vb.Rdata", sep=""))
print(paste("Loading fits from ", fit_file))

fit_env <- LoadIntoEnvironment(fit_file)
stan_results <- fit_env$mcmc_environment$results$original
stan_results_perturb <- fit_env$mcmc_environment$results$perturbed

# If true, save the results to a file readable by knitr.
save_results <- TRUE
results_file <- file.path(project_directory,
                          paste(analysis_name, "function_sensitivity.Rdata", sep="_"))

###########################################
# Extract results

pp <- fit_env$mcmc_environment$pp

vp_opt <- fit_env$vb_fit$vp_opt
lrvb_cov <- fit_env$lrvb_terms$lrvb_cov
prior_sens <- fit_env$prior_sens
lrvb_terms <- fit_env$lrvb_terms

mp_opt <- GetMoments(vp_opt)
mfvb_cov <- GetCovariance(vp_opt)

# Convenient indices
vp_indices <- GetParametersFromVector(vp_opt, as.numeric(1:vp_opt$encoded_size), FALSE)
mp_indices <- GetMomentsFromVector(mp_opt, as.numeric(1:mp_opt$encoded_size))
pp_indices <- GetPriorsFromVector(pp, as.numeric(1:pp$encoded_size))


##############################################
# Calculate epsilon sensitivity

# Monte Carlo integrate using importance sampling


# Mu draws:
GetMuImportanceFunctions <- function(mu_comp, vp_opt, pp, lrvb_terms) {
  mp_opt <- GetMoments(vp_opt)
  
  u_mean <- mp_opt$mu_e_vec[mu_comp]
  # Increase the variance for sampling.  How much is enough?
  u_cov <- (1.5 ^ 2) * solve(vp_opt$mu_info)[mu_comp, mu_comp]
  GetULogDensity <- function(u) {
    dnorm(u, mean=u_mean, sd=sqrt(u_cov), log=TRUE)
  }
  
  DrawU <- function(n_samples) {
    rnorm(n_samples, mean=u_mean, sd=sqrt(u_cov))
  }
  
  GetLogPrior <- function(u) {
    GetMuLogStudentTPrior(u, pp)
  }
  
  GetLogPriorVec <- function(u_vec) {
    sapply(u_vec, function(u) { GetLogPrior(u) } )
  }
  
  # This is the univariate density, analytically marginalized in C++
  GetLogVariationalDensity <- function(u) {
    mp_draw <- mp_opt
    mu_q_derivs <- GetMuLogMarginalDensity(u, mu_comp, vp_opt, mp_draw, TRUE)
    return(mu_q_derivs$val) 
  }
  
  lrvb_pre_factor <- -1 * lrvb_terms$jac %*% solve(lrvb_terms$elbo_hess)
  GetLogQGradTerm <- function(u) {
    mp_draw <- mp_opt
    mu_q_derivs <- GetMuLogMarginalDensity(u, mu_comp, vp_opt, mp_draw, TRUE)
    as.numeric(lrvb_pre_factor %*% mu_q_derivs$grad)
  }
  
  return(list(GetULogDensity=GetULogDensity,
              DrawU=DrawU,
              GetLogPrior=GetLogPrior,
              GetLogPriorVec=GetLogPriorVec,
              GetLogVariationalDensity=GetLogVariationalDensity,
              GetLogQGradTerm=GetLogQGradTerm))
}


# Tau draws:
GetTauImportanceFunctions <- function(group, vp_opt, pp, lrvb_terms) {
  mp_opt <- GetMoments(vp_opt)
  
  # Increase the variance for sampling.  How much is enough?
  u_shape <- 0.5 * vp_opt$tau[[group]]$alpha
  u_rate <- 0.5 * vp_opt$tau[[group]]$beta
  
  GetULogDensity <- function(u) {
    dgamma(u, shape=u_shape, rate=u_rate, log=TRUE)
  }
  
  DrawU <- function(n_samples) {
    rgamma(n_samples, shape=u_shape, rate=u_rate)
  }
  
  GetLogPrior <- function(u) {
    GetTauLogPrior(u, pp)
  }
  
  GetLogPriorVec <- function(u_vec) {
    sapply(u_vec, function(u) { GetLogPrior(u) } )
  }
  
  # This is the univariate density, analytically marginalized in C++
  GetLogVariationalDensity <- function(u) {
    mp_draw <- mp_opt
    tau_q_derivs <- GetTauLogMarginalDensity(u, group, vp_opt, mp_draw, TRUE, FALSE)
    return(tau_q_derivs$val) 
  }
  
  lrvb_pre_factor <- -1 * lrvb_terms$jac %*% solve(lrvb_terms$elbo_hess)
  GetLogQGradTerm <- function(u) {
    mp_draw <- mp_opt
    tau_q_derivs <- GetTauLogMarginalDensity(u, group, vp_opt, mp_draw, TRUE, TRUE)
    as.numeric(lrvb_pre_factor %*% tau_q_derivs$grad)
  }
  
  return(list(GetULogDensity=GetULogDensity,
              DrawU=DrawU,
              GetLogPrior=GetLogPrior,
              GetLogPriorVec=GetLogPriorVec,
              GetLogVariationalDensity=GetLogVariationalDensity,
              GetLogQGradTerm=GetLogQGradTerm))
}


GetWorstCaseResultsDataFrame <- function(vb_fns, param_draws, param_name, draws_mat, mp_opt) {
  vb_influence_results <- GetVariationalInfluenceResults(
    num_draws=n_samples,
    DrawImportanceSamples=vb_fns$DrawU,
    GetImportanceLogProb=vb_fns$GetULogDensity,
    GetLogQGradTerm=vb_fns$GetLogQGradTerm,
    GetLogQ=vb_fns$GetLogVariationalDensity,
    GetLogPrior=vb_fns$GetLogPrior)
  
  mcmc_funs <- GetMCMCInfluenceFunctions(param_draws, vb_fns$GetLogPriorVec)
  mcmc_worst_case <- sapply(1:ncol(draws_mat), function(ind) { mcmc_funs$GetMCMCWorstCase(draws_mat[, ind]) })
  
  return(rbind(
    SummarizeRawMomentParameters(
      GetMomentsFromVector(mp_opt, vb_influence_results$worst_case), metric=param_name, method="lrvb"),
    SummarizeRawMomentParameters(
      GetMomentsFromVector(mp_opt, mcmc_worst_case), metric=param_name, method="mcmc")
  ))
}


# Monte Carlo samples
n_samples <- 5000

# Define functions necessary to compute influence function stuff

results_df_list <- list()

# For Mu
for (mu_comp in 1:vp_opt$k_reg) {
  cat("Mu component ", mu_comp, "\n")
  vb_fns <- GetMuImportanceFunctions(mu_comp, vp_opt, pp, lrvb_terms)
  param_draws <- fit_env$mcmc_environment$mcmc_sample$mu[, mu_comp]
  param_name <- paste("mu", mu_comp, "wc", sep="_")
  results_df_list[[length(results_df_list) + 1]] <-
    GetWorstCaseResultsDataFrame(vb_fns, param_draws, param_name, fit_env$mcmc_environment$draws_mat, mp_opt)
} 

# For tau
for (group in 1:vp_opt$n_g) {
  cat("Tau group ", group, "\n")
  vb_fns <- GetTauImportanceFunctions(group, vp_opt, pp, lrvb_terms)
  param_draws <- 1 / fit_env$mcmc_environment$mcmc_sample$sigma_y[, group] ^ 2
  param_name <- paste("tau", group, "wc", sep="_")
  results_df_list[[length(results_df_list) + 1]] <-
    GetWorstCaseResultsDataFrame(vb_fns, param_draws, param_name, fit_env$mcmc_environment$draws_mat, mp_opt)
} 

res <- do.call(rbind, results_df_list)

ggplot(dcast(res, par + component + group + metric ~ method, value.var="val")) +
  geom_point(aes(color=par, x=mcmc, y=lrvb), size=3) +
  geom_abline(aes(slope=1, intercept=0)) + facet_grid(~ metric, scales="free")

ggplot(dcast(res, par + component + group + metric ~ method, value.var="val") %>% filter(metric == "tau_7_wc")) +
  geom_point(aes(color=par, x=mcmc, y=lrvb), size=3) +
  geom_abline(aes(slope=1, intercept=0))


stop()

group <- 1

vb_fns <- GetTauImportanceFunctions(group, vp_opt, pp, lrvb_terms)
param_draws <- 1 / fit_env$mcmc_environment$mcmc_sample$sigma_y[, group] ^ 2

vb_influence_results <- GetVariationalInfluenceResults(
  num_draws=n_samples,
  DrawImportanceSamples=vb_fns$DrawU,
  GetImportanceLogProb=vb_fns$GetULogDensity,
  GetLogQGradTerm=vb_fns$GetLogQGradTerm,
  GetLogQ=vb_fns$GetLogVariationalDensity,
  GetLogPrior=vb_fns$GetLogPrior)

mcmc_funs <- GetMCMCInfluenceFunctions(param_draws, vb_fns$GetLogPriorVec)

mcmc_influence_df <- data.frame(
  u=param_draws,
  dens=mcmc_funs$dens_at_draws$y)

vb_influence_df <- data.frame(
  u=vb_influence_results$u_draws,
  inf=vb_influence_results$influence_fun[, 1],
  imp_lp=vb_influence_results$importance_lp_ratio,
  lq=vb_influence_results$log_q,
  lp=vb_influence_results$log_prior)

ggplot() +
  geom_point(data=vb_influence_df, aes(x=u, y=exp(lq), color="vb")) +
  geom_point(data=mcmc_influence_df, aes(x=u, y=dens, color="mcmc"))



ggplot() +
  geom_point(data=vb_influence_df, aes(x=u, y=exp(lq), color="vb")) 

grid.arrange(
  ggplot() +
    geom_point(data=vb_influence_df, aes(x=u, y=inf, color="vb")) +
    geom_point(data=mcmc_influence_df, aes(x=u, y=inf, color="mcmc"))
  ,
  ggplot() +
    geom_point(data=vb_influence_df, aes(x=u, y=exp(lq), color="vb")) +
    geom_point(data=mcmc_influence_df, aes(x=u, y=dens, color="mcmc"))
  , ncol=2
)






##############################
# Save selected results for use in the paper

if (save_results) {
  component_df <- sample_n(component_df, 5000)
  save(sens_results, pp, pp_perturb, component_df, epsilon_df, file=results_file)
}
