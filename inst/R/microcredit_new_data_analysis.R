library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)
library(mvtnorm)

library(MicrocreditLRVB)
library_location <- file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/")
source(file.path(library_location, "inst/R/microcredit_stan_lib.R"))

# Load previously computed Stan results
analysis_name <- "simulated_data5"
project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/simulated_data")

stan_draws_file <-
  file.path(project_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))
print(paste("Loading draws from ", stan_draws_file))

stan_results <- environment()
load(stan_draws_file, envir=stan_results)

x <- stan_results$stan_dat$x
y <- stan_results$stan_dat$y
y_g <- stan_results$stan_dat$y_group

##################

MakeSymmetric <- function(mat) {
  return(0.5 * (mat + t(mat)))
}

################################
# Initialize the data

vp <- InitializeVariationalParameters(x, y, y_g)
pp <- SetPriorsFromVP(vp)

derivs <- GetElboDerivatives(x, y, y_g, vp, pp, TRUE, TRUE, TRUE)
theta_init <- GetVectorFromParameters(vp, TRUE)


GetVectorBounds <- function(vp_base, loc_bound=100, info_bound=1e9, min_bound=0.0000001) {
  # Get sensible extreme bounds, for example for L-BFGS-B
  info_lower <- diag(vp_nat_lower$k_reg) * vp_nat_lower$diag_min * (1 + min_bound)
  info_upper <- diag(vp_nat_lower$k_reg) * info_bound
  
  vp_nat_lower <- vp_base
  vp_nat_lower$mu_loc[] <- -1 * loc_bound
  vp_nat_lower$mu_info[,] <- info_lower
  vp_nat_lower$lambda_v <- info_lower
  vp_nat_lower$lambda_n <- min_bound 
  for (g in 1:vp_nat_lower$n_g) {
    vp_nat_lower$mu_g[[g]][["loc"]] <- -1 * loc_bound
    vp_nat_lower$mu_g[[g]][["info"]] <- info_lower
    vp_nat_lower$tau[[g]][["alpha"]] <- vp_nat_lower$tau_alpha_min + min_bound
    vp_nat_lower$tau[[g]][["beta"]] <- vp_nat_lower$tau_beta_min + min_bound
  }

  vp_nat_upper <- vp_base
  vp_nat_upper$mu_loc[] <- loc_bound
  vp_nat_upper$mu_info[,] <- info_upper
  vp_nat_upper$lambda_v <- info_upper
  vp_nat_upper$lambda_n <- min_bound 
  for (g in 1:vp_nat_lower$n_g) {
    vp_nat_upper$mu_g[[g]][["loc"]] <- loc_bound
    vp_nat_upper$mu_g[[g]][["info"]] <- info_upper
    vp_nat_upper$tau[[g]][["alpha"]] <- vp_nat_upper$tau_alpha_min + min_bound
    vp_nat_upper$tau[[g]][["beta"]] <- vp_nat_upper$tau_beta_min + min_bound
  }
  
  theta_lower <- GetVectorFromParameters(vp_nat_lower, TRUE)
  theta_upper <- GetVectorFromParameters(vp_nat_upper, TRUE)
  
  return(list(theta_lower=theta_lower, theta_upper=theta_upper))  
}


OptimVal <- function(theta) {
  this_vp <- GetParametersFromVector(vp, theta, TRUE)
  ret <- GetElboDerivatives(x, y, y_g, this_vp, pp,
                            calculate_gradient=FALSE, calculate_hessian=FALSE,
                            unconstrained=TRUE)
  cat(ret$val, "\n")
  ret$val
}

OptimGrad <- function(theta) {
  this_vp <- GetParametersFromVector(vp, theta, TRUE)
  ret <- GetElboDerivatives(x, y, y_g, this_vp, pp,
                            calculate_gradient=TRUE, calculate_hessian=FALSE,
                            unconstrained=TRUE)
  ret$grad
}

OptimHess <- function(theta) {
  this_vp <- GetParametersFromVector(vp, theta, TRUE)
  ret <- GetElboDerivatives(x, y, y_g, this_vp, pp,
                            calculate_gradient=TRUE, calculate_hessian=TRUE,
                            unconstrained=TRUE)
  ret$hess
}


bounds <- GetVectorBounds(vp)

optim_result <- optim(theta_init, OptimVal, OptimGrad, method="L-BFGS-B",
                      lower=bounds$theta_lower, upper=bounds$theta_upper,
                      control=list(fnscale=-1, factr=1))
vp_opt <- GetParametersFromVector(vp, optim_result$par, TRUE)


##########
# Fit it with VB


max_iters <- 500
vb_tol <- 1e-12
vb_time <- Sys.time()
vb_fit <- FitModel(x, y, y_g, vp, pp,
                   num_iters=max_iters, rel_tol=vb_tol, fit_lambda=TRUE, verbose=TRUE)
vb_time <- Sys.time() - vb_time

# Linear response covariance:
lrvb_time <- Sys.time()

ll_derivs <- ModelGradient(x, y, y_g, vb_fit$vp, pp, TRUE, TRUE)

mfvb_cov <- MakeSymmetric(GetVariationalCovariance(vb_fit$vp, pp))
lambda_lik_derivs <- LambdaLikelihoodMomentDerivs(x, y, y_g, vb_fit$vp, pp, TRUE)

dmoment_dpar_t <- Diagonal(encoder$dim)
dmoment_dpar_t[lambda_ind, lambda_ind] <- t(lambda_lik_derivs$dmoment_dtheta)

obs_hess <- Matrix(MakeSymmetric(ll_derivs$obs_hess))
h_mat <- solve(dmoment_dpar_t, t(solve(dmoment_dpar_t, obs_hess)))
h_mat[lambda_ind, lambda_ind] <- MakeSymmetric(lambda_lik_derivs$d2l_dm2)

lrvb_id_mat <- Diagonal(encoder$dim)

lrvb_inv_term_orig <- (lrvb_id_mat - mfvb_cov %*% h_mat)
lrvb_cov <- MakeSymmetric(solve(lrvb_inv_term_orig, mfvb_cov))

lrvb_time <- Sys.time() - lrvb_time


################################################
# VB sensitivity

moment_ind <- model_encoder$variational_offset + 1:model_encoder$variational_dim
prior_ind <- model_encoder$prior_offset + 1:(model_encoder$prior_dim)

k_ud <- vp$k * (vp$k + 1) / 2

mu_info_ind <- prior_encoder$mu_info_offset + 1:k_ud
mu_ind <- model_encoder$e_mu + 1:vp$k

# Change the derivatives to be with respect to the moments
prior_derivs <- PriorSensitivity(vb_fit$vp, pp)
prior_sub_hess <- prior_derivs$prior_hess[moment_ind, prior_ind]
prior_sens <- solve(lrvb_inv_term_orig, mfvb_cov %*% solve(dmoment_dpar_t, prior_sub_hess))

GetSensitivityDataframe <- function(offset, metric) {
  prior_sens_this <- prior_sens[, offset]
  prior_sens_this_list <- DecodeParameters(prior_sens_this, vb_fit$vp, pp, FALSE)
  prior_sens_this_df <- ConvertParameterListToDataframe(prior_sens_this_list, metric)
  prior_sens_this_df$param <- sub("lambda_v_par", "lambda", prior_sens_this_df$param)
  prior_sens_this_df$method <- "lrvb"
  return(prior_sens_this_df)
}

prior_sens_df <- data.frame()
for (k in 1:k_ud) {
  prior_sens_df <- rbind(prior_sens_df, GetSensitivityDataframe(prior_encoder$mu_info_offset + k,
                                                                paste("lambda", k, sep="_")))
}
for (k in 1:vp$k) {
  prior_sens_df <-
    rbind(prior_sens_df, GetSensitivityDataframe(prior_encoder$mu_mean_offset + k,
                                                 paste("mu", k, sep="_")))
}

prior_sens_df <-
  rbind(prior_sens_df,
        GetSensitivityDataframe(prior_encoder$lambda_eta_offset + 1, "lambda_eta"))

prior_sens_df <-
  rbind(prior_sens_df,
        GetSensitivityDataframe(prior_encoder$lambda_beta_offset + 1, "lambda_beta"))

prior_sens_df <-
  rbind(prior_sens_df,
        GetSensitivityDataframe(prior_encoder$lambda_alpha_offset + 1, "lambda_alpha"))


lrvb_sd_list <- DecodeParameters(sqrt(diag(lrvb_cov)), vb_fit$vp, pp, FALSE)
lrvb_sd_df <-
  ConvertParameterListToDataframe(lrvb_sd_list, "sd") %>%
  dplyr::select(-method, -metric) %>% rename(lrvb_sd=value)
lrvb_sd_df$param <- as.character(lrvb_sd_df$param)

prior_sens_df <- inner_join(prior_sens_df, lrvb_sd_df, by=c("param", "component", "group"))


###################
# Put the results in a tidy format and graph

mcmc_sample <- extract(stan_results$stan_sim)
mcmc_sample_perturb <- extract(stan_results$stan_sim_perturb)

result <- GetResultDataframe(mcmc_sample, vb_fit$vp, lrvb_cov, mfvb_cov, encoder)

# VB sensitivity for comparison:
prior_sens_lambda_offdiag_df <-
  GetSensitivityDataframe(prior_encoder$mu_info_offset + 1, "lambda_12_sens") %>%
  mutate(diff = value * perturb_epsilon)

result_perturb <-
  GetResultDataframe(mcmc_sample_perturb, vb_fit$vp, lrvb_cov, mfvb_cov, encoder) %>%
  filter(method=="mcmc") %>% mutate(method="mcmc_perturbed")

result_perturb_diff <-
  rbind(filter(result, method=="mcmc"), result_perturb) %>%
  filter(metric == "mean") %>% dplyr::select(-matches("metric")) %>%
  dcast(param + component + group ~ method) %>%
  mutate(diff = mcmc_perturbed - mcmc, value = diff / perturb_epsilon) %>%
  mutate(metric="lambda_12_sens", method="mcmc") %>%
  dplyr::select(-mcmc, -mcmc_perturbed) %>%
  rbind(prior_sens_lambda_offdiag_df) %>%
  dcast(param + component + group + metric ~ method, value.var="diff") %>%
  filter(!is.na(mcmc))


ggplot(filter(result, metric == "mean") %>%
  dcast(param + component + group ~ method)) +
  geom_point(aes(x=mcmc, y=mfvb, color=param), size=3) +
  geom_abline(aes(slope=1, intercept=0)) +
  expand_limits(x=0, y=0) + expand_limits(x=1, y=1) +
  xlab("MCMC") + ylab("VB") +
  ggtitle("Comparison of means")

ggplot(filter(result, metric == "sd") %>%
     dcast(param + component + group ~ method)) +
  geom_point(aes(x=mcmc, y=mfvb, color="mfvb"), size=3) +
  geom_point(aes(x=mcmc, y=lrvb, color="lrvb"), size=3) +
  geom_abline(aes(slope=1, intercept=0)) +
  expand_limits(x=0, y=0) + expand_limits(x=1, y=1) +
  xlab("MCMC") + ylab("VB") +
  ggtitle("Comparison of standard deviations")

# Note: make sure that the sensitivity is enough to be detected by
# the sampling error in MCMC.
ggplot(result_perturb_diff) +
  geom_point(aes(x=mcmc, y=lrvb, color=param), size=2) +
  geom_abline(aes(slope=1, intercept=0)) +
  xlab("MCMC") + ylab("VB") +
  ggtitle("Comparison of sensitivity")

