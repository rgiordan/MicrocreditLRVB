library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)
library(mvtnorm)
library(MicrocreditLRVB)

# Load previously computed Stan results
#analysis_name <- "simulated_data_robust"
analysis_name <- "simulated_data_nonrobust"
#analysis_name <- "simulated_data_lambda_beta"

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

# Convenient indices
vp_indices <- GetParametersFromVector(vp_opt, as.numeric(1:vp_opt$encoded_size), FALSE)
mp_indices <- GetMomentsFromVector(mp_opt, as.numeric(1:mp_opt$encoded_size))
pp_indices <- GetPriorsFromVector(pp, as.numeric(1:pp$encoded_size))

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


diff_df <- SummarizeRawMomentParameters(GetMomentsFromVector(mp_opt, diff_vec), metric="mean", method="diff")
ggplot(diff_df) + geom_histogram(aes(x=diff_df$val)) + facet_grid( ~ par)


##############################################
# Calculate epsilon sensitivity
# Monte Carlo integrate using a importance sampling

# Monte Carlo samples
n_samples <- 10000

# You could also do this more numerically stably with a Cholesky decomposition.
lrvb_pre_factor <- -1 * lrvb_terms$jac %*% solve(lrvb_terms$elbo_hess)


if (FALSE) {
  # Uniform proposals
  grid_range <- 3 * sqrt(diag(lrvb_cov)[vp_indices$mu_loc])
  grid_center <- mp_opt$mu_e_vec
  
  GetULogDensity <- function(mu) {
    log(prod(1 / (2 * grid_range)))
  }
  
  u_draws <- matrix(NA, n_samples, vp_opt$k_reg)
  for (k in 1:vp_opt$k_reg) {
    # Rescale and center uniform draws
    u_draws[, k] <- grid_range[k] * (2 * runif(n_samples) - 1) + grid_center[k]
  }
} else {
  # Proposals based on q
  grid_range <- 2 * sqrt(diag(lrvb_cov)[vp_indices$mu_loc])
  u_cov <- diag(grid_range ^ 2)
  grid_center <- mp_opt$mu_e_vec
  
  GetULogDensity <- function(mu) {
    dmvnorm(mu, mean=grid_center, sigma=u_cov, log=TRUE)  
  }
  
  u_draws <- rmvnorm(n_samples, mean=grid_center, sigma=u_cov)
}


GetUniformWeightedInfuenceFunctionVector <- function(mu) {
  mu_prior_val <- GetMuLogPrior(mu, pp)
  mu_q_res <- GetMuLogDensity(mu, vp_opt, pp, TRUE)
  mu_t_prior_val <- GetMuLogStudentTPrior(mu, pp_perturb)
  u_log_density <- GetULogDensity(mu)
  weight <- exp(mu_t_prior_val- u_log_density)
  sens_pre_factor <- exp(mu_q_res$val - mu_prior_val)
  prior_ratio <- exp(mu_t_prior_val - mu_prior_val)
  lrvb_term <- lrvb_pre_factor %*% mu_q_res$grad
  influence_fun <- sens_pre_factor * lrvb_term 
  sens <- exp(mu_t_prior_val- u_log_density + mu_q_res$val - mu_prior_val) * lrvb_term
  
  # The "mean value theorem" sensitivity
  mv_term <- (mu_t_prior_val - mu_prior_val) * prior_ratio / (prior_ratio - 1)
  mv_sens <- lrvb_term * mv_term * exp(mu_q_res$val - u_log_density)
  mv_int <- (mu_t_prior_val - mu_prior_val) *
            exp(mu_t_prior_val + mu_prior_val) / (exp(mu_t_prior_val) - exp(mu_prior_val))
  
  return(list(sens=sens, weight=weight, sens_pre_factor=sens_pre_factor,
              influence_fun=influence_fun, prior_ratio=prior_ratio,
              mv_sens=mv_sens, mv_term=mv_term, mv_int=mv_int))
}

sens_list <- list()
pb <- txtProgressBar(min=1, max=nrow(u_draws), style=3)
for (ind in 1:nrow(u_draws)) {
  setTxtProgressBar(pb, ind)
  sens_list[[ind]] <- GetUniformWeightedInfuenceFunctionVector(u_draws[ind, ])
}
close(pb)


prior_ratios <- unlist(lapply(sens_list, function(entry) { entry$prior_ratio} ))
weights <- unlist(lapply(sens_list, function(entry) { entry$weight} ))
max(weights) / median(weights)
# hist(log10(weights), 1000)
summary(weights)
mean(weights)
sum(weights) / n_samples

sens_pre_factors <- unlist(lapply(sens_list, function(entry) { entry$sens_pre_factor} ))
mv_term <- unlist(lapply(sens_list, function(entry) { entry$mv_term} ))
max(sens_pre_factors) / median(sens_pre_factors)
summary(sens_pre_factors)


# hist(log10(sens_pre_factors), 1000)
sens_vec_list <- lapply(sens_list, function(entry) { as.numeric(entry$sens) } )
mv_sens_vec_list <- lapply(sens_list, function(entry) { as.numeric(entry$mv_sens) } )

influence_vec_list <- lapply(sens_list, function(entry) { as.numeric(entry$influence_fun) } )
sens_vec_list_squared <- lapply(sens_list, function(entry) { as.numeric(entry$sens) ^ 2 } )
length(sens_vec_list)
# sens_vec_mean <- Reduce(`+`, sens_vec_list) / sum(weights)
sens_vec_mean <- Reduce(`+`, sens_vec_list) / n_samples # TODO: why this and not the sum of the weights?
sens_vec_mean_square <- Reduce(`+`, sens_vec_list_squared) / n_samples
sens_vec_sd <- sqrt(sens_vec_mean_square - sens_vec_mean^2) / sqrt(n_samples)

mv_sens_vec_mean <- Reduce(`+`, mv_sens_vec_list) / n_samples # TODO: why this and not the sum of the weights?
mv_sens_vec_list_squared <- lapply(sens_list, function(entry) { as.numeric(entry$mv_sens) ^ 2 } )
mv_sens_vec_mean_square <- Reduce(`+`, mv_sens_vec_list_squared) / n_samples
mv_sens_vec_sd <- sqrt(mv_sens_vec_mean_square - mv_sens_vec_mean^2) / sqrt(n_samples)


diagnostic_df <- data.frame(sens_pre_factor=sens_pre_factors, weight=weights, prior_ratios=prior_ratios, mv_term=mv_term)
diagnostic_df <- cbind(diagnostic_df, data.frame(u_draws))
mean(sens_pre_factors)
sd(sens_pre_factors)
# ggplot(diagnostic_df) + geom_point(aes(x=X1, y=X2, color=sens_pre_factor))
# ggplot(diagnostic_df) + geom_point(aes(x=X1, y=X2, color=log10(prior_ratios))) + scale_color_gradient2()
ggplot(diagnostic_df) + geom_point(aes(x=X1, y=X2, color=mv_term)) + scale_color_gradient2()

# hist((sens_pre_factors), 100)
# hist(log10(weights))


diff_vec <- GetVectorFromMoments(mp_opt_perturb) - GetVectorFromMoments(mp_opt)

sens_results <-
  rbind(
    SummarizeRawMomentParameters(GetMomentsFromVector(mp_opt, sens_vec_mean), metric="mean", method="sens"),
    SummarizeRawMomentParameters(GetMomentsFromVector(mp_opt, sens_vec_sd), metric="sd", method="sens"),
    SummarizeRawMomentParameters(GetMomentsFromVector(mp_opt, mv_sens_vec_mean), metric="mean", method="mv_sens"),
    SummarizeRawMomentParameters(GetMomentsFromVector(mp_opt, mv_sens_vec_sd), metric="sd", method="mv_sens"),
    SummarizeRawMomentParameters(GetMomentsFromVector(mp_opt, diff_vec),      metric="mean", method="diff")) %>%
  dcast(par + component + group ~ method + metric, value.var="val")

ggplot(sens_results) +
  geom_errorbar(aes(x=diff_mean, y=mv_sens_mean,
                    ymax=mv_sens_mean + 2 * mv_sens_sd,
                    ymin=mv_sens_mean - 2 * mv_sens_sd), color="gray") +
  geom_point(aes(x=diff_mean, y=mv_sens_mean, color=par)) +
  geom_abline((aes(intercept=0, slope=1)))

ggplot(sens_results) +
  geom_errorbar(aes(x=diff_mean, y=sens_mean,
                    ymax=sens_mean + 2 * sens_sd,
                    ymin=sens_mean - 2 * sens_sd), color="gray") +
  geom_point(aes(x=diff_mean, y=sens_mean, color=par)) +
  geom_abline((aes(intercept=0, slope=1)))



#######################################
# Look at one component in detail

# component <- mp_indices$mu_e_vec[1]; component_name <- "E_q[mu[1]]"
# component <- mp_indices$mu_e_vec[2]; component_name <- "E_q[mu[2]]"
# component <- mp_indices$lambda_e[1, 1]; component_name <- "E_q[lambda[1, 1]]"
# component <- mp_indices$lambda_e[2, 2]; component_name <- "E_q[lambda[2, 2]]"
# component <- mp_indices$lambda_e[1, 2]; component_name <- "E_q[lambda[1, 2]]"
# component <- mp_indices$tau[[1]]$e_log; component_name <- "E_q[log(tau[1])]"
component <- mp_indices$mu_g[[7]]$e_vec[1]; component_name <- "E_q[mu_g[7]][1]"

comp_sens_vec <- unlist(lapply(sens_vec_list, function(entry) { as.numeric(entry[component]) } ))
comp_influence_vec <- unlist(lapply(influence_vec_list, function(entry) { as.numeric(entry[component]) } ))

component_df <- cbind(data.frame(sens_vec=comp_sens_vec, influence=comp_influence_vec), diagnostic_df)

p1 <- ggplot(component_df) + geom_point(aes(x=X1, y=X2, color=sens_vec)) + scale_color_gradient2()
p2 <- ggplot(component_df) + geom_point(aes(x=X1, y=X2, color=influence)) + scale_color_gradient2()
p3 <- ggplot(component_df) + geom_point(aes(x=X1, y=X2, color=log10(prior_ratios))) + scale_color_gradient2()

mean(component_df$influence)

multiplot(p1, p2, p3)

# ggplot(component_df) + geom_histogram(aes(x=log10(sens_vec)), bins=1000)

mean(component_df$sens_vec)
sens_vec_sd[component]

# Estimate error
u_dist <- u_draws - rep(colMeans(u_draws), each=nrow(u_draws))
u_dist <- sqrt(rowSums(u_dist^2))
u_extreme <- which(u_dist > quantile(u_dist, 0.95))
max(abs(comp_influence_vec[u_extreme]))
# ggplot(component_df[u_extreme, ]) + geom_point(aes(x=X1, y=X2, color=abs(influence))) + scale_color_gradient2()



