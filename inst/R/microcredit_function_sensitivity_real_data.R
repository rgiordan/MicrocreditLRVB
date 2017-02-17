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

# Get a function that converts a a draw from mu_k and a standard mvn into a draw from (mu_c | mu_k)
# k is the conditioned component, c is the "complement", i.e. the rest
GetConditionalMVNFunction <- function(k_ind, mvn_mean, mvn_info) {
  mvn_sigma <- solve(mvn_info)
  c_ind <- setdiff(1:length(mvn_mean), k_ind)
  
  # The scale in front of the mu_k for the mean of (mu_c | mu_k)
  # mu_cc_sigma <- mu_sigma[c_ind, c_ind, drop=FALSE]
  mu_kk_sigma <- mvn_sigma[k_ind, k_ind, drop=FALSE]
  mu_ck_sigma <- mvn_sigma[c_ind, k_ind, drop=FALSE]
  sig_cc_corr <- mu_ck_sigma %*% solve(mu_kk_sigma)
  
  # What to multiply by to get Cov(mu_c | mu_k)
  mu_c_scale <- t(chol(solve(mvn_info[c_ind, c_ind])))
  
  # Given u and a draws mu_c_std ~ Standard normal, convert mu_c_std to a draw from MVN( . | mu_k)
  GetConditionalDraw <- function(mu_k, mu_c_std) {
    mu_c_scale %*% mu_c_std + mvn_mean[c_ind] + sig_cc_corr %*% (mu_k - mvn_mean[k_ind])
  }
}

if (FALSE) {
  # Test GetConditionalDraw
  n_draws <- 100000
  mvn_mean <- runif(4)
  mvn_cov <- diag(4) + 0.2
  k_ind <- c(1, 2)
  CondFn <- GetConditionalMVNFunction(k_ind, mvn_mean, solve(mvn_cov))
  mu_k_draws <- rmvnorm(n_draws, mean=mvn_mean[k_ind], sigma=mvn_cov[k_ind, k_ind])
  mu_std_draws <- rmvnorm(n_draws, mean=rep(0, length(mvn_mean) - length(k_ind)))
  mu_c_draws <- matrix(NaN, n_draws, ncol(mu_std_draws))
  for (k in 1:nrow(mu_k_draws)) {
    mu_c_draws[k,] <- CondFn(as.matrix(mu_k_draws[k, ]), as.matrix(mu_std_draws[k, ]))
  }
  
  mu_cond_draws <- matrix(NaN, n_draws, length(mvn_mean))
  mu_cond_draws[, k_ind] <- mu_k_draws
  mu_cond_draws[, setdiff(1:length(mvn_mean), k_ind)] <- mu_c_draws
  
  colMeans(mu_cond_draws)
  mvn_mean
  
  cov(mu_cond_draws)
  mvn_cov
}



# Mu draws:
GetMuImportanceFunctions <- function(mu_comp, vp_opt, pp, lrvb_terms, num_mu_c_draws=10) {
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
  
  # This is wrong.  The average grad term is not the grad of the marginal.
  # lrvb_pre_factor <- -1 * lrvb_terms$jac %*% solve(lrvb_terms$elbo_hess)
  # GetLogQGradTerm <- function(u) {
  #   mp_draw <- mp_opt
  #   mu_q_derivs <- GetLogVariationalDensityDerivatives(u, mu_comp, vp_opt, mp_draw, TRUE)
  #   as.numeric(lrvb_pre_factor %*% mu_q_derivs$grad)
  # }
  
  GetConditionalDraw <- GetConditionalMVNFunction(mu_comp, vp_opt$mu_loc, vp_opt$mu_info)
    
  mu_c_std_draws <- rmvnorm(num_mu_c_draws, mean=rep(0, length(c_ind)))
  lrvb_pre_factor <- -1 * lrvb_terms$jac %*% solve(lrvb_terms$elbo_hess)
  GetLogQGradTerm <- function(u) {
    # Estimate the average log term conditional on mu[mu_comp] = u
    mp_draw <- mp_opt
    
    # Draw mu2
    this_mu <- rep(NaN, vp_opt$k_reg)
    c_ind <- setdiff(1:vp_opt$k, mu_comp)
    avg_grad_term <- rep(0, vp_opt$encoded_size)
    for (row in 1:nrow(mu_c_std_draws)) {
      this_mu[mu_comp] <- u
      this_mu[c_ind] <- GetConditionalDraw(u, mu_c_std_draws[row, ])
      mu_q_derivs <- GetMuLogDensity(mu=this_mu, vp_opt=vp_opt, draw=mp_draw, pp=pp,
                                     unconstrained=TRUE, calculate_gradient=TRUE, global_only=FALSE)
      avg_grad_term <- avg_grad_term + as.numeric(lrvb_pre_factor %*% mu_q_derivs$grad)
    }
    avg_grad_term <- avg_grad_term / nrow(mu_c_std_draws)
    return(avg_grad_term)
  }
  
  return(list(GetULogDensity=GetULogDensity,
              DrawU=DrawU,
              GetLogPrior=GetLogPrior,
              GetLogPriorVec=GetLogPriorVec,
              GetLogVariationalDensity=GetLogVariationalDensity,
              GetLogQGradTerm=GetLogQGradTerm,
              GetConditionalDraw=GetConditionalDraw))
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
  
  worst_case_df <- rbind(
    SummarizeRawMomentParameters(
      GetMomentsFromVector(mp_opt, vb_influence_results$worst_case), metric=param_name, method="lrvb"),
    SummarizeRawMomentParameters(
      GetMomentsFromVector(mp_opt, mcmc_worst_case), metric=param_name, method="mcmc")
    
  return(worst_case_df)
}


# Monte Carlo samples
n_samples <- 5000

# Define functions necessary to compute influence function stuff

results_df_list <- list()

# For Mu
for (mu_comp in 1:vp_opt$k_reg) {
  cat("Mu component ", mu_comp, "\n")
  vb_fns <- GetMuImportanceFunctions(mu_comp, vp_opt, pp, lrvb_terms, num_mu_c_draws=100)
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
res_graph <- dcast(res, par + component + group + metric ~ method, value.var="val")

# Look at one example in detail.

mu_comp <- 1
vb_fns <- GetMuImportanceFunctions(mu_comp, vp_opt, pp, lrvb_terms)
vb_influence_results <- GetVariationalInfluenceResults(
  num_draws=n_samples,
  DrawImportanceSamples=vb_fns$DrawU,
  GetImportanceLogProb=vb_fns$GetULogDensity,
  GetLogQGradTerm=vb_fns$GetLogQGradTerm,
  GetLogQ=vb_fns$GetLogVariationalDensity,
  GetLogPrior=vb_fns$GetLogPrior)

param_draws <- fit_env$mcmc_environment$mcmc_sample$mu[, mu_comp]
mcmc_funs <- GetMCMCInfluenceFunctions(param_draws, vb_fns$GetLogPriorVec)
mcmc_inf <- mcmc_funs$GetMCMCInfluence(param_draws)

mcmc_influence_df <- data.frame(
  u=param_draws,
  inf=mcmc_inf,
  dens=mcmc_funs$dens_at_draws$y)

ind <- mp_indices$mu_e_vec[mu_comp]
log_q_grad_terms <- sapply(vb_influence_results$u_draws, function(u) { vb_fns$GetLogQGradTerm(u)[ind] })

vb_influence_df <- data.frame(
  u=vb_influence_results$u_draws,
  u_centered=vb_influence_results$u_draws - vp_opt$mu_loc[mu_comp],
  inf=vb_influence_results$influence_fun[, ind],
  imp_lp=vb_influence_results$importance_lp_ratio,
  lq=vb_influence_results$log_q,
  lp=vb_influence_results$log_prior,
  lq_grad=log_q_grad_terms)


if (save_results) {
  save(n_samples, res_graph, mu_comp, mcmc_influence_df, vb_influence_df, file=results_file)
}




######################
# Graphs

stop("graphs follow -- not executing.")

ggplot() +
  geom_point(data=vb_influence_df, aes(x=u_centered, y=log_q_grad_terms)) +
  geom_abline(aes(slope=1, intercept=0))

ggplot() +
  geom_point(data=vb_influence_df, aes(x=u, y=exp(lq))) +
  geom_point(data=mcmc_influence_df, aes(x=u, y=dens))

grid.arrange(
  ggplot() +
    geom_point(data=vb_influence_df, aes(x=u, y=exp(lq), color="vb")) +
    geom_point(data=mcmc_influence_df, aes(x=u, y=dens, color="mcmc"))
,
  ggplot() +
    geom_point(data=vb_influence_df, aes(x=u, y=log_q_grad_terms, color="vb_g")) +
    geom_point(data=mcmc_influence_df, aes(x=u, y=u - mean(mcmc_influence_df$u), color="mcmc_g"))
, ncol=1  
)
  
# ggplot(res_graph) +
#   geom_point(aes(color=par, x=mcmc, y=lrvb), size=3) +
#   geom_abline(aes(slope=1, intercept=0)) + facet_grid(~ metric, scales="free")

ggplot(filter(res_graph, grepl("mu", metric))) +
  geom_point(aes(color=par, x=mcmc, y=lrvb), size=3) +
  geom_abline(aes(slope=1, intercept=0)) + facet_grid(~ metric, scales="free")

ggplot(filter(res_graph, grepl("tau", metric))) +
  geom_point(aes(color=par, x=mcmc, y=lrvb), size=3) +
  geom_abline(aes(slope=1, intercept=0)) + facet_grid(~ metric, scales="free")

ggplot(filter(res_graph, grepl("tau", metric), grepl("1", metric))) +
  geom_point(aes(color=par, x=mcmc, y=lrvb), size=3) +
  geom_abline(aes(slope=1, intercept=0))

ggplot(filter(res_graph, grepl("tau", metric), par == "log_tau")) +
  geom_point(aes(color=par, x=mcmc, y=lrvb), size=3) +
  geom_abline(aes(slope=1, intercept=0)) + facet_grid(~ metric, scales="free")

ggplot(filter(res_graph, grepl("tau", metric), par == "mu_g")) +
  geom_point(aes(color=par, x=mcmc, y=lrvb), size=3) +
  geom_abline(aes(slope=1, intercept=0)) + facet_grid(~ metric, scales="free")


ggplot(res_graph %>% filter(par == "tau")) +
  geom_point(aes(color=par, x=mcmc, y=lrvb), size=3) +
  geom_abline(aes(slope=1, intercept=0)) + facet_grid(~ metric, scales="free")

ggplot(res_graph %>% filter(metric == "tau_7_wc")) +
  geom_point(aes(color=par, x=mcmc, y=lrvb), size=3) +
  geom_abline(aes(slope=1, intercept=0))


#####################
# Debugging and analysis

group <- 7
draws_mat <- fit_env$mcmc_environment$draws_mat
vb_fns <- GetTauImportanceFunctions(group, vp_opt, pp, lrvb_terms)
param_draws <- 1 / fit_env$mcmc_environment$mcmc_sample$sigma_y[, group] ^ 2

GetLogPrior <- vb_fns$GetLogPriorVec

vb_influence_results <- GetVariationalInfluenceResults(
  num_draws=n_samples,
  DrawImportanceSamples=vb_fns$DrawU,
  GetImportanceLogProb=vb_fns$GetULogDensity,
  GetLogQGradTerm=vb_fns$GetLogQGradTerm,
  GetLogQ=vb_fns$GetLogVariationalDensity,
  GetLogPrior=vb_fns$GetLogPrior)


ind <- mp_indices$tau[[7]]$e
g_draws <- draws_mat[, ind]

GetEYGivenX <- function(x_draws, y_draws) {
  e_y_given_x <- loess.smooth(x_draws, y_draws)
  return(approx(e_y_given_x$x, e_y_given_x$y, xout = x_draws)$y)
}

mcmc_dens <- density(param_draws)
dens_at_draws <- approx(mcmc_dens$x, mcmc_dens$y, xout = param_draws)
mcmc_influence_ratio <- exp(log(dens_at_draws$y) - GetLogPrior(param_draws))
mcmc_importance_ratio <- exp(GetLogPrior(param_draws) - log(dens_at_draws$y))
conditional_mean_diff <- GetEYGivenX(x_draws = param_draws, y_draws = g_draws) - mean(g_draws)
plot(param_draws, conditional_mean_diff * mcmc_influence_ratio)
return(conditional_mean_diff * mcmc_influence_ratio)


mcmc_funs <- GetMCMCInfluenceFunctions(param_draws, vb_fns$GetLogPriorVec)
conditional_mean_diff <- GetEYGivenX(x_draws = param_draws, y_draws = g_draws) - mean(g_draws)
g_draws - mean(g_draws)
return(conditional_mean_diff * mcmc_influence_ratio)

mcmc_inf <- mcmc_funs$GetMCMCInfluence(g_draws)

mcmc_influence_df <- data.frame(
  u=param_draws,
  inf=mcmc_inf,
  dens=mcmc_funs$dens_at_draws$y)

log_q_grad_terms <- sapply(vb_influence_results$u_draws, function(u) { vb_fns$GetLogQGradTerm(u)[ind] })

vb_influence_df <- data.frame(
  u=vb_influence_results$u_draws,
  inf=vb_influence_results$influence_fun[, ind],
  imp_lp=vb_influence_results$importance_lp_ratio,
  lq=vb_influence_results$log_q,
  lp=vb_influence_results$log_prior,
  lq_grad=log_q_grad_terms)


grid.arrange(
ggplot() +
  geom_point(data=vb_influence_df, aes(x=u, y=inf, color="vb"))
,
ggplot() +
  geom_point(data=mcmc_influence_df, aes(x=u, y=inf, color="mcmc"))
, ncol=2
)



ggplot() +
  geom_point(data=vb_influence_df, aes(x=u, y=exp(lq), color="vb")) +
  geom_point(data=mcmc_influence_df, aes(x=u, y=dens, color="mcmc"))


ggplot() +
  geom_point(data=vb_influence_df, aes(x=u, y=exp(lp), color="vb")) +
  geom_point(data=mcmc_influence_df, aes(x=u, y=exp(vb_fns$GetLogPriorVec(u)), color="mcmc"))


ggplot() +
  geom_point(data=vb_influence_df, aes(x=u, y=exp(lq - lp), color="vb")) +
  geom_point(data=mcmc_influence_df, aes(x=u, y=exp(log(dens) - vb_fns$GetLogPriorVec(u)), color="mcmc"))

# The two influence functions are way off because the log_q_grad_terms look nothing like the draws.
ggplot() +
  geom_point(data=vb_influence_df, aes(x=u, y=log_q_grad_terms, color="vb")) +
  geom_point(data=mcmc_influence_df, aes(x=u, y=g_draws, color="mcmc"))
  


# Recall that this is wrt the unconstrained variable.
LogQGrad <- function(u) {
  mp_draw <- mp_opt
  tau_q_derivs <- GetTauLogMarginalDensity(u, group, vp_opt, mp_draw, TRUE, TRUE)
  as.numeric(lrvb_terms$jac %*% tau_q_derivs$grad)
}

log_q_grad_only_terms <- sapply(vb_influence_results$u_draws, function(u) { LogQGrad(u)[mp_indices$tau[[group]]$e] })
plot(vb_influence_results$u_draws - mean(vb_influence_results$u_draws),
     log_q_grad_only_terms / vp_opt$tau[[group]]$beta); abline(0,1)

u <- median(vb_influence_results$u_draws)
lrvb_pre_factor <- -1 * lrvb_terms$jac %*% solve(lrvb_terms$elbo_hess)
mp_draw <- mp_opt
tau_q_derivs <- GetTauLogMarginalDensity(u, group, vp_opt, mp_draw, TRUE, TRUE)
tau_q_derivs$grad[vp_indices$tau[[group]]$alpha]
tau_q_derivs$grad[vp_indices$tau[[group]]$beta]

# Ultimately, the log q grad term doesn't look like the draws because of the Hessian.  I guess it's because
# both alpha and beta affect tau.  This deserves more thought.
beta <-vp_opt$tau[[group]]$beta
SummarizeRawMomentParameters(
  GetMomentsFromVector(mp_opt, lrvb_pre_factor[, vp_indices$tau[[group]]$beta]), metric="", method="beta")

SummarizeRawMomentParameters(
  GetMomentsFromVector(mp_opt, lrvb_pre_factor[, vp_indices$tau[[group]]$alpha]), metric="", method="alpha") 

u
SummarizeRawMomentParameters(
  GetMomentsFromVector(mp_opt, as.numeric(lrvb_pre_factor %*% tau_q_derivs$grad)), metric="", method="beta")

SummarizeRawMomentParameters(
  GetMomentsFromVector(mp_opt, as.numeric(lrvb_pre_factor %*% tau_q_derivs$grad)), metric="", method="alpha")


# ggplot() +
#   geom_point(data=vb_influence_df, aes(x=u, y=exp(lq), color="vb")) 
# 
# grid.arrange(
#   ggplot() +
#     geom_point(data=vb_influence_df, aes(x=u, y=inf, color="vb")) +
#     geom_point(data=mcmc_influence_df, aes(x=u, y=inf, color="mcmc"))
#   ,
#   ggplot() +
#     geom_point(data=vb_influence_df, aes(x=u, y=exp(lq), color="vb")) +
#     geom_point(data=mcmc_influence_df, aes(x=u, y=dens, color="mcmc"))
#   , ncol=2
# )
# 





##############################
# Save selected results for use in the paper

if (save_results) {
  component_df <- sample_n(component_df, 5000)
  save(sens_results, pp, pp_perturb, component_df, epsilon_df, file=results_file)
}
