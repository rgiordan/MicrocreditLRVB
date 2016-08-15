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
analysis_name <- "simulated_data_easy"
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


################################
# Initialize the data

#vp_base <- InitializeVariationalParameters(x, y, y_g, diag_min=1, tau_min=0)
vp_base <- InitializeZeroVariationalParameters(x, y, y_g, diag_min=1, tau_min=0)
pp <- SetPriorsFromVP(vp_base)


DerivFun <- function(x, y, y_g, base_vp, pp,
                     calculate_gradient, calculate_hessian,
                     unconstrained) {
  GetCustomElboDerivatives(x, y, y_g, base_vp, pp,
                           calculate_gradient=calculate_gradient, calculate_hessian=calculate_hessian,
                           include_obs=TRUE, include_hier=TRUE, include_prior=TRUE, include_entropy=TRUE,
                           unconstrained=TRUE)
}

vp_index <- GetParametersFromVector(vp_base, as.numeric(1:length(theta_init)), FALSE)

bounds <- GetVectorBounds(vp_base)
theta_init <- GetVectorFromParameters(vp_base, TRUE)
opt_fns <- GetOptimFunctions(x, y, y_g, vp_base, pp, DerivFun=DerivFun)
opt_fns$OptimVal(theta_init)
opt_fns$OptimGrad(theta_init)

stopifnot(all(bounds$theta_lower < theta_init) && all(bounds$theta_upper > theta_init))

optim_result <- optim(theta_init, opt_fns$OptimVal, opt_fns$OptimGrad, method="L-BFGS-B",
                      lower=bounds$theta_lower, upper=bounds$theta_upper,
                      control=list(fnscale=-1, maxit=1000))
# optim_result <- optim(theta_init, opt_fns$OptimVal, opt_fns$OptimGrad, method="Nelder-Mead",
#                       control=list(fnscale=-1))

vp_opt <- GetParametersFromVector(vp_base, optim_result$par, TRUE)
nat_result <- SummarizeNaturalParameters(vp_opt)

##############
# Comare to GLM
library(lme4)

glm_data <- data.frame(y=y, y_g=y_g)
glm_data <- cbind(glm_data, data.frame(x))
glm_res <- lmer(y ~ (X1 + 0 | y_g) + (X2 + 0 | y_g) + 0, data=glm_data)
glm_summary <- summary(glm_res)


true_mu_g <- list()
for (g in 1:vp_base$n_g) {
  true_mu_g[[g]] <- stan_results$true_params$true_mu_g[[g]]
}
true_mu_g <- do.call(rbind, true_mu_g)

vb_mu_g <-
  filter(nat_result, par == "mu_g") %>%
  dcast(group ~ component, value.var="val")

if (FALSE) {
  plot(as.matrix(ranef(glm_res)$y_g)[,1], true_mu_g[,1]); abline(0, 1)
  plot(as.matrix(ranef(glm_res)$y_g)[,1], vb_mu_g[["1"]]); abline(0, 1)
}


opt_fns$OptimGrad(optim_result$par)

hess <- opt_fns$OptimHess(optim_result$par)
hess_eig <- eigen(hess)
min(hess_eig$values)
max(hess_eig$values)



##########
# Fit it with VB
# Test
vp_new <- GetParametersFromVector(vp_base, GetVectorFromParameters(vp_base, TRUE), TRUE)


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

