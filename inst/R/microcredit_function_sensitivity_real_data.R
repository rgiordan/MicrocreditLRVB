library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)
library(mvtnorm)
library(MicrocreditLRVB)
library(LRVBUtils)
library(gridExtra)
library(abind)


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
# Calculate function sensitivity


GetVBWorstCaseResultsDataFrame <- function(vb_fns, param_name, mp_opt, n_samples) {
  vb_influence_results <- GetVariationalInfluenceResults(
    num_draws=n_samples,
    DrawImportanceSamples=vb_fns$DrawU,
    GetImportanceLogProb=vb_fns$GetULogDensity,
    GetLogQGradTerm=vb_fns$GetLogQGradTerm,
    GetLogQ=vb_fns$GetLogVariationalDensity,
    GetLogPrior=vb_fns$GetLogPrior)

  SummarizeRawMomentParameters(
    GetMomentsFromVector(mp_opt, vb_influence_results$worst_case), metric=param_name, method="lrvb")
}


GetMCMCWorstCaseResultsDataFrame <- function(param_draws, param_name, draws_mat, mp_opt) {
  mcmc_funs <- GetMCMCInfluenceFunctions(param_draws, vb_fns$GetLogPrior)
  mcmc_worst_case <- sapply(1:ncol(draws_mat), function(ind) { mcmc_funs$GetMCMCWorstCase(draws_mat[, ind]) })
  
  SummarizeRawMomentParameters(
    GetMomentsFromVector(mp_opt, mcmc_worst_case), metric=param_name, method="mcmc")
}


# Monte Carlo samples
n_samples <- 500
num_bootstraps <- 10

results_df_list <- list()
draws_mat <- fit_env$mcmc_environment$draws_mat

# For Mu
for (mu_comp in 1:vp_opt$k_reg) {
  cat("Mu component ", mu_comp, "\n")
  vb_fns <- GetMuImportanceFunctions(mu_comp, vp_opt, pp, lrvb_terms)
  param_draws <- fit_env$mcmc_environment$mcmc_sample$mu[, mu_comp]
  param_name <- paste("mu", mu_comp, "wc", sep="_")

  for (draw in 1:num_bootstraps) {
    cat(".")
    results_df_list[[length(results_df_list) + 1]] <-
      GetVBWorstCaseResultsDataFrame(vb_fns, param_name, mp_opt, n_samples) %>%
      mutate(draw=draw)

    mcmc_rows <- sample.int(nrow(draws_mat), replace=TRUE)
    results_df_list[[length(results_df_list) + 1]] <-
      GetMCMCWorstCaseResultsDataFrame(param_draws[mcmc_rows], param_name, draws_mat[mcmc_rows, ], mp_opt) %>%
      mutate(draw=draw)
  }
  cat("\n")
}

# For tau
for (group in 1:vp_opt$n_g) {
  cat("Tau group ", group, "\n")
  vb_fns <- GetTauImportanceFunctions(group, vp_opt, pp, lrvb_terms)
  param_draws <- 1 / fit_env$mcmc_environment$mcmc_sample$sigma_y[, group] ^ 2
  param_name <- paste("tau", group, "wc", sep="_")
  
  for (draw in 1:num_bootstraps) {
    cat(".")
    results_df_list[[length(results_df_list) + 1]] <-
      GetVBWorstCaseResultsDataFrame(vb_fns, param_name, mp_opt, n_samples) %>%
      mutate(draw=draw)
    
    mcmc_rows <- sample.int(nrow(draws_mat), replace=TRUE)
    results_df_list[[length(results_df_list) + 1]] <-
      GetMCMCWorstCaseResultsDataFrame(param_draws[mcmc_rows], param_name, draws_mat[mcmc_rows, ], mp_opt) %>%
      mutate(draw=draw)
  }
  cat("\n")
} 


worst_case_df <- do.call(rbind, results_df_list) %>%
  group_by(par, component, group, metric, method) %>%
  summarize(val_mean=mean(val), val_sd=sd(val)) %>%
  melt(id.vars=c("par", "component", "group", "metric", "method")) %>%
  mutate(summary=variable)

worst_case_cast <-
  dcast(worst_case_df, par + component + group + metric ~ method + summary, value.var="value")
  

# Add uncertainty ellipses
theta <- seq(0, 2 * pi, length.out=20)
ellipse_dfs <- list()
for (row in 1:nrow(worst_case_cast)) {
  sd_x <- worst_case_cast[row, "mcmc_val_sd"]
  sd_y <- worst_case_cast[row, "lrvb_val_sd"]
  loc_x <- worst_case_cast[row, "mcmc_val_mean"]
  loc_y <- worst_case_cast[row, "lrvb_val_mean"]
  ellipse_dfs[[length(ellipse_dfs) + 1]] <-
    data.frame(x=loc_x + 2 * sd_x * cos(theta), y=loc_y + 2 * sd_y * sin(theta), row=row)
}
worst_case_cast_sds <- do.call(rbind, ellipse_dfs) %>%
  inner_join(mutate(worst_case_cast, row=1:nrow(worst_case_cast)), by="row")


ggplot(filter(worst_case_cast_sds, grepl("mu", metric))) +
  geom_polygon(aes(x=x, y=y, group=row), alpha=0.1, color=NA) +
  geom_point(aes(x=mcmc_val_mean, y=lrvb_val_mean), size=2) +
  geom_abline(aes(slope=1, intercept=0)) +
  expand_limits(x=0, y=0) +
  xlab("MCMC") + ylab("VB") 

ggplot(filter(worst_case_cast_sds, !grepl("mu", metric))) +
  geom_polygon(aes(x=x, y=y, group=row), alpha=0.1, color=NA) +
  geom_point(aes(x=mcmc_val_mean, y=lrvb_val_mean), size=2) +
  geom_abline(aes(slope=1, intercept=0)) +
  expand_limits(x=0, y=0) +
  xlab("MCMC") + ylab("VB") 


# res_graph <- dcast(res, par + component + group + metric ~ method, value.var="val")


# Look at one example in detail.
mu_comp <- 1
ind <- mp_indices$mu_e_vec[mu_comp]
# ind <- mp_indices$mu_g[[5]]$e_vec[1]
influence_symbol <- "mu" # This will be used to make the graphs in the paper
vb_fns <- GetMuImportanceFunctions(mu_comp, vp_opt, pp, lrvb_terms)
vb_influence_results <- GetVariationalInfluenceResults(
  num_draws=n_samples,
  DrawImportanceSamples=vb_fns$DrawU,
  GetImportanceLogProb=vb_fns$GetULogDensity,
  GetLogQGradTerm=vb_fns$GetLogQGradTerm,
  GetLogQ=vb_fns$GetLogVariationalDensity,
  GetLogPrior=vb_fns$GetLogPrior)


param_draws <- fit_env$mcmc_environment$mcmc_sample$mu[, mu_comp]
mcmc_funs <- GetMCMCInfluenceFunctions(param_draws, vb_fns$GetLogPrior)
mcmc_inf <- mcmc_funs$GetMCMCInfluence(param_draws)

mcmc_influence_df <- data.frame(
  u=param_draws,
  inf=mcmc_inf,
  dens=mcmc_funs$dens_at_draws)

ind <- mp_indices$mu_e_vec[mu_comp]
log_q_grad_terms <- vb_fns$GetLogQGradTerm(vb_influence_results$u_draws)[ind, ]

vb_influence_df <- data.frame(
  u=vb_influence_results$u_draws,
  u_centered=vb_influence_results$u_draws - vp_opt$mu_loc[mu_comp],
  inf=vb_influence_results$influence_fun[, ind],
  imp_lp=vb_influence_results$importance_lp_ratio,
  lq=vb_influence_results$log_q,
  lp=vb_influence_results$log_prior,
  worst_case_u=vb_influence_results$worst_case_u[, ind],
  lq_grad=log_q_grad_terms)


if (save_results) {
  save(n_samples, worst_case_cast_sds, mu_comp, mcmc_influence_df, vb_influence_df, influence_symbol, file=results_file)
}


stop("graphs follow -- not executing.")


######################
# Graphs

ggplot(filter(res_graph, grepl("mu", metric))) +
  geom_point(aes(x=mcmc, y=lrvb, color=par, shape=factor(group)), size=3) +
  geom_abline(aes(slope=1, intercept=0)) +
  facet_grid(~ metric)

ggplot(filter(res_graph, !grepl("mu", metric))) +
  geom_point(aes(x=mcmc, y=lrvb, color=par)) +
  geom_abline(aes(slope=1, intercept=0)) +
  facet_grid(~ metric)



GetNormalizingConstant <- function(x, dens) {
  x_order <- order(x)
  sum(diff(x[x_order]) * dens[-1])
}


prior_draws <- vb_fns$DrawFromPrior(5000)
prior_draws <- prior_draws[ prior_draws < quantile(prior_draws, 0.97) & prior_draws > quantile(prior_draws, 0.03)]


worst_case_u_const <- with(vb_influence_df, GetNormalizingConstant(u_draws, worst_case_u))
ggplot() +
  geom_line(data=vb_influence_df, aes(x=u, y=worst_case_u / worst_case_u_const, color="u")) +
  geom_line(aes(x=prior_draws, y=exp(vb_fns$GetLogPrior(prior_draws)), color="prior"))
  


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
    geom_point(data=vb_influence_df, aes(x=u, y=lq_grad, color="vb_g")) +
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

mcmc_funs <- GetMCMCInfluenceFunctions(param_draws, vb_fns$GetLogPrior)
conditional_mean_diff <- mcmc_funs$GetConditionalMeanDiff(g_draws)

mcmc_inf <- mcmc_funs$GetMCMCInfluence(g_draws)

mcmc_influence_df <- data.frame(
  u=param_draws,
  inf=mcmc_inf,
  dens=mcmc_funs$dens_at_draws$y)


log_q_grad_terms <- vb_fns$GetLogQGradTerm(vb_influence_results$u_draws)[ind, ]

vb_influence_df <- data.frame(
  u=vb_influence_results$u_draws,
  inf=vb_influence_results$influence_fun[, ind],
  imp_lp=vb_influence_results$importance_lp_ratio,
  lq=vb_influence_results$log_q,
  lp=vb_influence_results$log_prior,
  lq_grad=log_q_grad_terms)


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

SummarizeRawMomentParameters(
  GetMomentsFromVector(mp_opt, as.numeric(lrvb_pre_factor %*% tau_q_derivs$grad)), metric="", method="beta")

SummarizeRawMomentParameters(
  GetMomentsFromVector(mp_opt, as.numeric(lrvb_pre_factor %*% tau_q_derivs$grad)), metric="", method="alpha")




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

