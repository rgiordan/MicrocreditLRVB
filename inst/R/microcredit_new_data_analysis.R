library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)
library(mvtnorm)
library(trust)
library(LRVBUtils)

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


#############################
# Initialize

vp_base <- InitializeZeroVariationalParameters(
  x, y, y_g, mu_diag_min=0.01, lambda_diag_min=1e-5, tau_min=1, lambda_n_min=0.5)
vp_indices <- GetParametersFromVector(vp_base, as.numeric(1:vp_base$encoded_size), FALSE)
vm_base <- GetMoments(vp_base)
vm_indices <- GetMomentsFromVector(vm_base, as.numeric(1:vp_base$encoded_size))
vp_reg <- InitializeVariationalParameters(
  x, y, y_g, mu_diag_min=vp_base$mu_diag_min, lambda_diag_min=vp_base$lambda_diag_min,
  tau_min=vp_base$tau_alpha_min, lambda_n_min=vp_base$lambda_n_min)


####################
# Fix the prior

pp$k_reg <- pp$k
pp$mu_loc <- pp$mu_mean

#############
# BFGS

DerivFun <- function(x, y, y_g, vp, pp,
                     calculate_gradient, calculate_hessian, unconstrained) {
  GetCustomElboDerivatives(x, y, y_g, vp, pp,
                           include_obs=TRUE, include_hier=TRUE,
                           include_prior=TRUE, include_entropy=TRUE,
                           global_only=FALSE,
                           calculate_gradient=calculate_gradient,
                           calculate_hessian=calculate_hessian,
                           unconstrained=unconstrained)
}
mask <- rep(TRUE, vp_reg$encoded_size)
bfgs_opt_fns <- GetOptimFunctions(x, y, y_g, vp_reg, pp, DerivFun=DerivFun, mask=mask)
theta_init <- GetVectorFromParameters(vp_reg, TRUE)
bounds <- GetVectorBounds(vp_base, loc_bound=30, info_bound=10, tau_bound=100)

bfgs_time <- Sys.time()
bfgs_result <- optim(theta_init[mask],
                     bfgs_opt_fns$OptimVal, bfgs_opt_fns$OptimGrad,
                     method="L-BFGS-B", lower=bounds$theta_lower[mask], upper=bounds$theta_upper[mask],
                     control=list(fnscale=-1, maxit=1000, trace=0, factr=1e9))
stopifnot(bfgs_result$convergence == 0)
print(bfgs_result$message)
bfgs_time <- Sys.time() - bfgs_time

vp_bfgs <- GetParametersFromVector(vp_reg, bfgs_result$par, TRUE)


#############
# Trust region

tr_time <- Sys.time()
trust_fns <- GetTrustRegionELBO(x, y, y_g, vp_bfgs, pp, verbose=TRUE)
trust_result <- trust(trust_fns$TrustFun, trust_fns$theta_init,
                      rinit=1, rmax=100, minimize=FALSE, blather=TRUE, iterlim=50)
tr_time <- Sys.time() - tr_time
trust_result$converged
trust_result$value

bfgs_time + tr_time


#################################
# LRVB

unconstrained <- TRUE # This seems to have a better condition number. 
vp_opt <- GetParametersFromVector(vp_reg, trust_result$argument, TRUE)
vp_mom <- GetMoments(vp_opt)

moment_derivs <- GetMomentJacobian(vp_opt, unconstrained)
jac <- Matrix(moment_derivs$hess)

elbo_hess <- GetSparseELBOHessian(x, y, y_g, vp_opt, pp, unconstrained)
lrvb_cov <- -1 * jac %*% Matrix::solve(elbo_hess, Matrix::t(jac))
min(diag(lrvb_cov))

mfvb_cov <- GetCovariance(vp_opt)
min(diag(mfvb_cov))
# plot(sqrt(diag(lrvb_cov)), sqrt(diag(mfvb_cov))); abline(0, 1)

mfvb_sd <- GetMomentsFromVector(vp_mom, sqrt(diag(mfvb_cov)))
lrvb_sd <- GetMomentsFromVector(vp_mom, sqrt(diag(lrvb_cov)))


###############################
# Debugging

lrvb_cov_diag1 <- diag(lrvb_cov)

elbo_hess_eig <- eigen(elbo_hess)
max(elbo_hess_eig$values)
max(elbo_hess_eig$values) / min(elbo_hess_eig$values)

###########################
# Sumamrize results

mcmc_sample <- extract(stan_results$stan_sim)

results <- rbind(SummarizeMomentParameters(vp_mom, mfvb_sd, lrvb_sd),
                 SummarizeMCMCResults(mcmc_sample))

if (FALSE) {
  mean_results <-
    filter(results, metric == "mean") %>%
    dcast(par + component + group ~ method, value.var="val")
  
  ggplot(filter(mean_results, par != "mu_g")) +
    geom_point(aes(x=mcmc, y=mfvb, color=par), size=3) +
    geom_abline(aes(slope=1, intercept=0))
  
  ggplot(filter(mean_results, par == "mu_g")) +
    geom_point(aes(x=mcmc, y=mfvb, color=par), size=3) +
    geom_abline(aes(slope=1, intercept=0))

  ggplot(filter(mean_results, par == "tau")) +
    geom_point(aes(x=mcmc, y=mfvb, color=par), size=3) +
    geom_abline(aes(slope=1, intercept=0))
  
  sd_results <-
    filter(results, metric == "sd") %>%
    dcast(par + component + group ~ method, value.var="val")
  
  ggplot(filter(sd_results, par != "mu_g")) +
    geom_point(aes(x=mcmc, y=mfvb, shape=par, color="mfvb"), size=3) +
    geom_point(aes(x=mcmc, y=lrvb, shape=par, color="lrvb"), size=3) +
    geom_abline(aes(slope=1, intercept=0))
  
  ggplot(filter(sd_results, par == "mu_g")) +
    geom_point(aes(x=mcmc, y=mfvb, shape=par, color="mfvb"), size=3) +
    geom_point(aes(x=mcmc, y=lrvb, shape=par, color="lrvb"), size=3) +
    geom_abline(aes(slope=1, intercept=0))
}




########################
# Sanity check

nat_result <- SummarizeNaturalParameters(vp_opt)

vb_mu_g <-
  filter(nat_result, par == "mu_g") %>%
  dcast(group ~ component, value.var="val")

vp_opt$lambda_v * vp_opt$lambda_n
true_params$true_lambda

vb_tau_g <- filter(nat_result, par == "tau")
true_params$true_tau



##############
# Comare to GLM



library(lme4)

glm_data <- data.frame(y=y, y_g=y_g)
glm_data <- cbind(glm_data, data.frame(x))
glm_res <- lmer(y ~ (X1 + 0 | y_g) + (X2 + 0 | y_g) + X1 + X2 + 0, data=glm_data)

glm_res
filter(nat_result, par == "mu")

vb_mu_g <-
  filter(nat_result, par == "mu_g") %>%
  dcast(group ~ component, value.var="val")

if (FALSE) {
  plot(as.matrix(true_mu_g[,1]), as.matrix(ranef(glm_res)$y_g)[,1]); abline(0, 1)
  plot(as.matrix(ranef(glm_res)$y_g)[,1], vb_mu_g[["1"]]); abline(0, 1)
  plot(as.matrix(ranef(glm_res)$y_g)[,2], vb_mu_g[["2"]]); abline(0, 1)
  plot(as.matrix(true_mu_g[,1]), vb_mu_g[["1"]]); abline(0, 1)
  plot(as.matrix(true_mu_g[,2]), vb_mu_g[["2"]]); abline(0, 1)
}




opt_fns$OptimGrad(optim_result$theta)

grad <- opt_fns$OptimGrad(optim_result$theta)
hess <- opt_fns$OptimHess(optim_result$theta)
hess_eig <- eigen(hess)
min(hess_eig$values)
max(hess_eig$values)

newton_step <- solve(hess, grad)
diff <- opt_fns$OptimVal(optim_result$theta - newton_step) - opt_fns$OptimVal(optim_result$theta)












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

