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
stan_results <- fit_env$mcmc_environment$results$epsilon_0.000000
stan_results_perturb <- fit_env$mcmc_environment$results$epsilon_1.000000

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

# You could also do this more numerically stably with a Cholesky decomposition.
lrvb_pre_factor <- -1 * lrvb_terms$jac %*% solve(lrvb_terms$elbo_hess)

# Proposals based on q
mu_comp <- 1
u_mean <- mp_opt$mu_e_vec[mu_comp]
# Increase the covariance for sampling.  How much is enough?
u_cov <- (1.5 ^ 2) * solve(vp_opt$mu_info)[mu_comp, mu_comp]
GetULogDensity <- function(mu) {
  dnorm(mu, mean=u_mean, sigma=u_cov, log=TRUE)
}

DrawU <- function(n_samples) {
  rnorm(n_samples, mean=u_mean, sigma=u_cov)
}

# GetLogPrior <- function(u) {
#   GetMuLogPrior(u, pp)
# }
# 
# GetLogContaminatingPrior <- function(u) {
#   GetMuLogStudentTPrior(u, pp_perturb)
# }

GetLogContaminatingPrior <- function(u) {
  GetMuLogPrior(u, pp)
}

GetLogPrior <- function(u) {
  GetMuLogStudentTPrior(u, pp_perturb)
}

# TODO: this needs to be the univariate density.
GetLogVariationalDensity <- function(u) {
  mu_q_derivs <- GetMuLogDensity(u, vp_opt, mp_opt, pp, TRUE, FALSE)
  return(mu_q_derivs$val) 
}

GetLogQGradTerm <- function(mu) {
  mu_q_derivs <- GetMuLogDensity(u, vp_opt, mp_draw, pp, TRUE, TRUE)
  as.numeric(-1 * moment_jac %*% solve(vb_fit$hessian, grad_log_q))
}

GetVariationalInfluenceResults(
  num_draws=5000,
  DrawImportanceSamples=DrawU, metric="", GetImportanceLogProb=GetULogDensity, GetLogQGradTerm=)





GetInfluenceFunctionSample <- GetInfluenceFunctionSampleFunction(
  GetLogVariationalDensity, GetLogPrior, GetULogDensity, lrvb_pre_factor)

Rprof("/tmp/rprof")
influence_list <- list()
pb <- txtProgressBar(min=1, max=nrow(u_draws), style=3)
for (ind in 1:nrow(u_draws)) {
  setTxtProgressBar(pb, ind)
  influence_list[[ind]] <- GetInfluenceFunctionSample(u_draws[ind, ])
}
close(pb)
summaryRprof("/tmp/rprof")


GetSensitivitySample <- GetSensitivitySampleFunction(GetLogContaminatingPrior)
sensitivity_list <- lapply(influence_list, GetSensitivitySample)

sensitivities <- UnpackSensitivityList(sensitivity_list)
diff_vec <- GetVectorFromMoments(mp_opt_perturb) - GetVectorFromMoments(mp_opt)

sens_results <-
  rbind(
    SummarizeRawMomentParameters(GetMomentsFromVector(mp_opt, sensitivities$sens_vec_mean),    metric="mean", method="sens"),
    SummarizeRawMomentParameters(GetMomentsFromVector(mp_opt, sensitivities$sens_vec_sd),      metric="sd", method="sens"),
    SummarizeRawMomentParameters(GetMomentsFromVector(mp_opt, sensitivities$mv_sens_vec_mean), metric="mean", method="mv_sens"),
    SummarizeRawMomentParameters(GetMomentsFromVector(mp_opt, sensitivities$mv_sens_vec_sd),   metric="sd", method="mv_sens"),
    SummarizeRawMomentParameters(GetMomentsFromVector(mp_opt, diff_vec),                       metric="mean", method="diff")) %>%
  dcast(par + component + group ~ method + metric, value.var="val")


# The "mean value" local sensitivity.  We expect this to be better.
p1 <- ggplot(sens_results) +
  geom_errorbar(aes(x=diff_mean / pp_perturb$epsilon, y=mv_sens_mean,
                    ymax=mv_sens_mean + 2 * mv_sens_sd,
                    ymin=mv_sens_mean - 2 * mv_sens_sd), color="gray") +
  geom_point(aes(x=diff_mean / pp_perturb$epsilon, y=mv_sens_mean, color=par)) +
  geom_abline((aes(intercept=0, slope=1))) +
  ggtitle("Mean value sens")

# The raw local sensitivity
p2 <- ggplot(sens_results) +
  geom_errorbar(aes(x=diff_mean / pp_perturb$epsilon, y=sens_mean,
                    ymax=sens_mean + 2 * sens_sd,
                    ymin=sens_mean - 2 * sens_sd), color="gray") +
  geom_point(aes(x=diff_mean / pp_perturb$epsilon, y=sens_mean, color=par)) +
  geom_abline((aes(intercept=0, slope=1))) +
  ggtitle("raw sens")

multiplot(p1, p2)



################################################
# Try evaluating directly rather than via the influence function and importance sampling.

global_mask <- GlobalMask(vp_opt)
mp_draw <- mp_opt
GetLogVariationalDensity <- function(u) {
  mu_q_derivs <- GetMuLogDensity(u, vp_opt, mp_draw, pp, TRUE, TRUE)
  log_q_grad[global_mask] <- mu_q_derivs$grad
  list(val=mu_q_derivs$val, grad=log_q_grad) 
}

# GetLogPrior <- function(u) {
#   GetMuLogPrior(u, pp)
# }
# 
# GetLogContaminatingPrior <- function(u) {
#   GetMuLogStudentTPrior(u, pp_perturb)
# }

GetLogPrior <- function(u) {
  GetMuLogStudentTPrior(u, pp_perturb)
}

GetLogContaminatingPrior <- function(u) {
  GetMuLogPrior(u, pp)
}


# You could also do this more numerically stably with a Cholesky decomposition.
lrvb_pre_factor <- -1 * lrvb_terms$jac %*% solve(lrvb_terms$elbo_hess)

n_draws <- 50e3
density_ratios <- rep(NaN, n_draws)
influence_mat <- matrix(NaN, nrow(lrvb_pre_factor), n_draws)

mu_draws <- DrawFromQMu(n_draws, vp_opt)

pb <- txtProgressBar(min=1, max=n_draws, style=3)
for (n in 1:n_draws) {
  setTxtProgressBar(pb, n)
  mu <- mu_draws[n, ]
  log_q_derivs <- GetLogVariationalDensity(mu)
  log_prior_val <- GetLogPrior(mu)
  log_prior_c_val <- GetLogContaminatingPrior(mu)
  lrvb_term <- lrvb_pre_factor %*% log_q_derivs$grad
  density_ratios[n] <- exp(log_prior_c_val - log_prior_val) 
  influence_mat[, n] <- density_ratios[n] * as.numeric(lrvb_term)
}

sensitivities <- apply(influence_mat, MARGIN=1, mean)
sensitivities_sd <- apply(influence_mat, MARGIN=1, sd)
hist(log10(density_ratios), 100)


vis_df <- data.frame(mu_draws) %>%
  mutate(density_ratio=density_ratios)

ggplot(vis_df) +
  geom_point(aes(x=X1, y=X2, color=density_ratio), size=2) +
  scale_color_gradient2()

sens_results <-
  rbind(
    SummarizeRawMomentParameters(GetMomentsFromVector(mp_opt, sensitivities),    metric="mean", method="sens"),
    SummarizeRawMomentParameters(GetMomentsFromVector(mp_opt, sensitivities_sd),      metric="sd", method="sens"),
    SummarizeRawMomentParameters(GetMomentsFromVector(mp_opt, diff_vec),                       metric="mean", method="diff")) %>%
  dcast(par + component + group ~ method + metric, value.var="val")


# # The "mean value" local sensitivity.  We expect this to be better.
# p1 <- ggplot(sens_results) +
#   geom_errorbar(aes(x=diff_mean / pp_perturb$epsilon, y=mv_sens_mean,
#                     ymax=mv_sens_mean + 2 * mv_sens_sd,
#                     ymin=mv_sens_mean - 2 * mv_sens_sd), color="gray") +
#   geom_point(aes(x=diff_mean / pp_perturb$epsilon, y=mv_sens_mean, color=par)) +
#   geom_abline((aes(intercept=0, slope=1))) +
#   ggtitle("Mean value sens")

# The raw local sensitivity
ggplot(sens_results) +
  geom_errorbar(aes(x=diff_mean / pp_perturb$epsilon, y=sens_mean,
                    ymax=sens_mean + 2 * sens_sd,
                    ymin=sens_mean - 2 * sens_sd), color="gray") +
  geom_point(aes(x=diff_mean / pp_perturb$epsilon, y=sens_mean, color=par)) +
  geom_abline((aes(intercept=0, slope=1))) +
  ggtitle("raw sens")





#################################################
# Diagnostics and debugging

# This doesn't seem to matter ever.
# lrvb_term_diff_list <- lapply(sens_list, function(entry) { as.numeric(entry$lrvb_term) - as.numeric(entry$lrvb_term_pre) } )
# Reduce(max, lrvb_term_diff_list)

prior_ratios <- unlist(lapply(sens_list, function(entry) { entry$prior_ratio} ))
weights <- unlist(lapply(sens_list, function(entry) { entry$weight} ))

sens_pre_factors <- unlist(lapply(sens_list, function(entry) { entry$sens_pre_factor} ))
mv_term <- unlist(lapply(sens_list, function(entry) { entry$mv_term} ))
max(sens_pre_factors) / median(sens_pre_factors)
summary(sens_pre_factors)

diagnostic_df <- data.frame(sens_pre_factor=sens_pre_factors, weight=weights, prior_ratios=prior_ratios, mv_term=mv_term)
diagnostic_df <- cbind(diagnostic_df, data.frame(u_draws))
mean(sens_pre_factors)
sd(sens_pre_factors)

# ggplot(diagnostic_df) + geom_point(aes(x=X1, y=X2, color=sens_pre_factor)) + scale_color_gradient2()
# ggplot(diagnostic_df) + geom_point(aes(x=X1, y=X2, color=log10(prior_ratios))) + scale_color_gradient2()
# ggplot(diagnostic_df) + geom_point(aes(x=X1, y=X2, color=mv_term)) + scale_color_gradient2()

# hist((sens_pre_factors), 100)
# hist(log10(weights))



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
