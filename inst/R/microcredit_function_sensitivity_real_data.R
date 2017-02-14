library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)
library(mvtnorm)
library(MicrocreditLRVB)
library(LRVBUtils)


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
# Monte Carlo integrate using a importance sampling

# Monte Carlo samples
n_samples <- 5000

# Define functions necessary to compute influence function stuff

# Proposals based on q
mu_comp <- 2 # The component of mu to perturb.

u_mean <- mp_opt$mu_e_vec[mu_comp]
# Increase the covariance for sampling.  How much is enough?
u_cov <- (1.5 ^ 2) * solve(vp_opt$mu_info)[mu_comp, mu_comp]
GetULogDensity <- function(mu) {
  dnorm(mu, mean=u_mean, sd=sqrt(u_cov), log=TRUE)
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
mp_opt <- GetMoments(vp_opt)
GetLogVariationalDensity <- function(u) {
  mp_draw <- mp_opt
  mu_q_derivs <- GetMuLogMarginalDensity(u, mu_comp, vp_opt, mp_draw, TRUE)
  return(mu_q_derivs$val) 
}

lrvb_pre_factor <- -1 * lrvb_terms$jac %*% solve(lrvb_terms$elbo_hess)
GetLogQGradTerm <- function(u) {
  mp_draw <- mp_opt
  mu_q_derivs <- GetMuLogMarginalDensity(u, mu_comp, vp_opt, mp_draw, TRUE)
  # as.numeric(-1 * lrvb_terms$jac %*% solve(lrvb_terms$elbo_hess, mu_q_derivs$grad))
  as.numeric(lrvb_pre_factor %*% mu_q_derivs$grad)
}

mu_influence_results <- GetVariationalInfluenceResults(
  num_draws=5000,
  DrawImportanceSamples=DrawU,
  GetImportanceLogProb=GetULogDensity,
  GetLogQGradTerm=GetLogQGradTerm,
  GetLogQ=GetLogVariationalDensity,
  GetLogPrior=GetLogPrior)


mu_influence_results$log_q


param_draws <- fit_env$mcmc_environment$mcmc_sample$mu[, mu_comp]
g_draws <- param_draws

mcmc_funs <- GetMCMCInfluenceFunctions(param_draws, GetLogPriorVec)
mcmc_inf <- mcmc_funs$GetMCMCInfluence(g_draws) 

mcmc_influence_df <- data.frame(
  mu=param_draws,
  inf=mcmc_inf,
  dens=mcmc_funs$dens_at_draws$y)

vb_influence_df <- data.frame(
  u=mu_influence_results$u_draws,
  inf=mu_influence_results$influence_fun[, mp_indices$mu_e_vec[mu_comp]],
  imp_lp=mu_influence_results$importance_lp_ratio,
  lq=mu_influence_results$log_q,
  lp=mu_influence_results$log_prior)


ggplot() +
  geom_point(data=vb_influence_df, aes(x=u, y=inf, color="vb")) +
  geom_point(data=mcmc_influence_df, aes(x=mu, y=inf, color="mcmc"))

ggplot() +
  geom_point(data=vb_influence_df, aes(x=u, y=exp(lq), color="vb")) +
  geom_point(data=mcmc_influence_df, aes(x=mu, y=dens, color="mcmc"))


mcmc_funs$GetMCMCWorstCase(param_draws)
mu_influence_results$worst_case[1]

stop()



pp_indices




#######################################
# Look at one component in detail

# component <- mp_indices$mu_e_vec[1]; component_name <- "E_q[mu[1]]"
# component <- mp_indices$mu_e_vec[2]; component_name <- "E_q[mu[2]]"
# component <- mp_indices$lambda_e[1, 1]; component_name <- "E_q[lambda[1, 1]]"
# component <- mp_indices$lambda_e[2, 2]; component_name <- "E_q[lambda[2, 2]]"
# component <- mp_indices$lambda_e[1, 2]; component_name <- "E_q[lambda[1, 2]]"
# component <- mp_indices$tau[[1]]$e_log; component_name <- "E_q[log(tau[1])]"
# component <- mp_indices$mu_g[[7]]$e_vec[1]; component_name <- "E_q[mu_g[7]][1]"
component <- mp_indices$mu_g[[1]]$e_vec[1]; component_name <- "E_q[mu_g[1]][1]"

influence_vec_list <- lapply(influence_list, function(entry) { as.numeric(entry$influence_function) } )
comp_sens_vec <- unlist(lapply(sensitivity_list, function(entry) { as.numeric(entry$sensitivity_draw[component]) } ))
comp_influence_vec <- unlist(lapply(influence_vec_list, function(entry) { as.numeric(entry[component]) } ))

component_df <- cbind(data.frame(sens_vec=comp_sens_vec, influence=comp_influence_vec), data.frame(u_draws))
component_df$component_name <- component_name

p1 <- ggplot(component_df) + geom_point(aes(x=X1, y=X2, color=sens_vec)) +
  scale_color_gradient2() + ggtitle("Sensitivity")
p2 <- ggplot(component_df) + geom_point(aes(x=X1, y=X2, color=influence)) +
  scale_color_gradient2() + ggtitle("Influence")
# p3 <- ggplot(component_df) + geom_point(aes(x=X1, y=X2, color=log10(prior_ratios))) +
#   scale_color_gradient2() + ggtitle("prior ratio")

# multiplot(p1, p2, p3)

# Estimate error
u_dist <- u_draws - rep(colMeans(u_draws), each=nrow(u_draws))
u_dist <- sqrt(rowSums(u_dist^2))
u_extreme <- which(u_dist > quantile(u_dist, 0.95))
max(abs(comp_influence_vec[u_extreme]))
ggplot(component_df[u_extreme, ]) + geom_point(aes(x=X1, y=X2, color=abs(influence))) + scale_color_gradient2()



############################################
# Worst-case

# component <- mp_indices$mu_e_vec[1]; component_name <- "E_q[mu[1]]"
# component <- mp_indices$mu_e_vec[2]; component_name <- "E_q[mu[2]]"
# component <- mp_indices$lambda_e[1, 1]; component_name <- "E_q[lambda[1, 1]]"
# component <- mp_indices$lambda_e[2, 2]; component_name <- "E_q[lambda[2, 2]]"
# component <- mp_indices$lambda_e[1, 2]; component_name <- "E_q[lambda[1, 2]]"
# component <- mp_indices$tau[[1]]$e_log; component_name <- "E_q[log(tau[1])]"
# component <- mp_indices$mu_g[[7]]$e_vec[1]; component_name <- "E_q[mu_g[7]][1]"
component <- mp_indices$mu_g[[1]]$e_vec[1]; component_name <- "E_q[mu_g[7]][1]"

influence_vec_list <- lapply(sens_list, function(entry) { as.numeric(entry$influence_fun) } )
comp_sens_vec <- unlist(lapply(sens_vec_list, function(entry) { as.numeric(entry[component]) } ))
comp_influence_vec <- unlist(lapply(influence_vec_list, function(entry) { as.numeric(entry[component]) } ))
weights <- unlist(lapply(sens_list, function(entry) { entry$weight} ))

component_df <- cbind(data.frame(sens_vec=comp_sens_vec, influence=comp_influence_vec), diagnostic_df)
component_df$component_name <- component_name

component_df$influence2plus <- ifelse(component_df$influence > 0, component_df$influence^2, 0)
component_df$influence2minus <- ifelse(component_df$influence < 0, component_df$influence^2, 0)

# Worst-case
ggplot(component_df) + geom_point(aes(x=X1, y=X2, color=influence2plus)) + scale_color_gradient2() + ggtitle("Influence squared")
ggplot(component_df) + geom_point(aes(x=X1, y=X2, color=influence2minus)) + scale_color_gradient2() + ggtitle("Influence")

sum(component_df$influence2plus * weights)
sum(component_df$influence2minus * weights)


#################################
# Fit with the t prior

moments_list <- list()
epsilon_vec <- seq(0, 1e-3, length.out=20)
pp_perturb_epsilon <- pp_perturb
for (epsilon in epsilon_vec) {
  cat("-------------   ", epsilon , "\n")
  pp_perturb_epsilon$epsilon <- epsilon
  vb_fit_perturb_eps <- FitVariationalModel(x, y, y_g, vp_opt, pp_perturb_epsilon, fit_bfgs=FALSE)
  vp_opt_perturb_eps <- vb_fit_perturb_eps$vp_opt
  mp_opt_perturb_eps <- GetMoments(vp_opt_perturb_eps)
  moments_list[[length(moments_list) + 1]] <- mp_opt_perturb_eps
}
# plot(epsilon_vec, unlist(lapply(moments_list, function(x) { x$mu_e_vec[1] } )))
# plot(epsilon_vec, unlist(lapply(moments_list, function(x) { x$mu_g[[1]]$e_vec[1] } )))
epsilon_df <- data.frame(epsilon=epsilon_vec,
                         val=unlist(lapply(moments_list, function(x) { x$mu_g[[1]]$e_vec[1] } )),
                         parameter_name="mu_g[1][1]")
ggplot(epsilon_df) +
  geom_point(aes(x=epsilon, y=val)) +
  ggtitle(unique(epsilon_df$parameter_name))


##############################
# Save selected results for use in the paper

if (save_results) {
  component_df <- sample_n(component_df, 5000)
  save(sens_results, pp, pp_perturb, component_df, epsilon_df, file=results_file)
}
