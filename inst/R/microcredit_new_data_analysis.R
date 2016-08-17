library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)
library(mvtnorm)

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
# Check conversion

vp_base <- InitializeVariationalParameters(x, y, y_g, diag_min=1, tau_min=0)
theta_init <- GetVectorFromParameters(vp_base, TRUE)
vp_check <- GetParametersFromVector(vp_base, theta_init, TRUE)

################################
# Initialize the data

diag_min <- 1e-6
vp_reg <- InitializeVariationalParameters(x, y, y_g, diag_min=diag_min, tau_min=0)
vp_base <- InitializeZeroVariationalParameters(x, y, y_g, diag_min=diag_min, tau_min=0)
theta_init <- GetVectorFromParameters(vp_reg, TRUE)
pp <- stan_results$pp
pp[["k_reg"]] <- pp$k # Fix typo
pp[["mu_loc"]] <- pp$mu_mean # Fix typo
vp_index <- GetParametersFromVector(vp_base, as.numeric(1:length(theta_init)), FALSE)


# Optimize everything
mask <- rep(TRUE, length(theta_init))

DerivFun <- function(x, y, y_g, base_vp, pp,
                     calculate_gradient, calculate_hessian,
                     unconstrained) {
  GetCustomElboDerivatives(x, y, y_g, base_vp, pp,
                           include_obs=TRUE, include_hier=TRUE,
                           include_prior=TRUE, include_entropy=TRUE,
                           use_group=FALSE, g=-2,
                           calculate_gradient=calculate_gradient,
                           calculate_hessian=calculate_hessian,
                           unconstrained=unconstrained)
}


bounds <- GetVectorBounds(vp_base, loc_bound=30, info_bound=20)
GetParametersFromVector(vp_base, theta_init, TRUE)
opt_fns <- GetOptimFunctions(x, y, y_g, vp_base, pp, DerivFun=DerivFun, mask=mask)
opt_fns$OptimVal(theta_init[mask])
opt_fns$OptimGrad(theta_init[mask])

stopifnot(all(bounds$theta_lower < theta_init) && all(bounds$theta_upper > theta_init))

optim_time <- Sys.time()
bfgs_time <- Sys.time()
optim_result0 <- optim(theta_init[mask], opt_fns$OptimVal, opt_fns$OptimGrad, method="L-BFGS-B",
                      lower=bounds$theta_lower[mask], upper=bounds$theta_upper[mask],
                      control=list(fnscale=-1, maxit=2000, trace=1, factr=1))
stopifnot(optim_result0$convergence == 0)
print(optim_result0$message)
bfgs_time <- Sys.time() - bfgs_time


if (FALSE) {
  # Debugging
  EvalFun <- opt_fns$OptimVal
  EvalGrad <- opt_fns$OptimGrad
  theta <- optim_result0$par[mask]
  
  vp_bfgs <- GetParametersFromVector(vp_base, theta, TRUE)
  theta[unique(as.numeric(vp_index$lambda_v))]
  mp_bfgs <- GetMoments(vp_bfgs)
  mfvb_cov <- GetCovariance(vp_bfgs)
  mfvb_sd <- GetMomentsFromVector(mp_bfgs, sqrt(diag(mfvb_cov)))
  SummarizeMomentParameters(mp_bfgs, mfvb_sd, mfvb_sd) %>% filter(method != "lrvb")
  
  hess <- opt_fns$OptimHess(theta)
  grad <- opt_fns$OptimGrad(theta)
  hess_eig <- eigen(hess)
  sum(hess_eig$values > 1e-8)
  length(hess_eig$values)
  
  ind <- which(hess_eig$values > 1e-8)
  hess_eig$values[ind]
  hess_p <- hess_eig$vectors[, ind]
  hess_p_outer <- t(hess_p) %*% hess_p 
  grad_p <- hess_p %*% solve(hess_p_outer, t(hess_p) %*% grad)
  grad_perp <- grad - grad_p
  grad_params <- GetParametersFromVector(vp_base, grad_p, FALSE)

  eig_p <- hess_eig$vectors[, which.max(hess_eig$values)]
  
  grid_vals <- list()
  eps_grid <- seq(-10, 28, length.out=50)
  for (i in 1:length(eps_grid)) {
    eps <- eps_grid[i]
    grid_vals[[i]] <- data.frame(eps=eps, f=EvalFun(theta + grad_p * eps))
  }
  grid_vals <- do.call(rbind, grid_vals)
  qplot(eps, f, data=grid_vals)

  grid_vals <- list()
  eps_grid1 <- seq(-10, 30, length.out=20)
  eps_grid2 <- seq(-1e-5, 1e-5, length.out=20)
  for (i in 1:length(eps_grid)) { cat("-------\n"); for (j in 1:length(eps_grid)) {
    eps1 <- eps_grid1[i]
    eps2 <- eps_grid2[j]
    new_theta <- theta + grad_p * eps1 + grad_perp * eps2
    grid_vals[[length(grid_vals) + 1]] <- data.frame(eps1=eps1, eps2=eps2, f=EvalFun(new_theta))
  }}
  grid_vals <- do.call(rbind, grid_vals)

  ggplot(grid_vals) + geom_tile(aes(x=eps1, y=eps2, fill=f))
  
  theta1 <- theta + grad_p * 8e5
  EvalFun(theta1) - EvalFun(theta)
  
  theta <- theta1
  
  grad_p[grad_p < 1e-8] <- 0
  
  step_direction <- grad_p
  ls_result <- LineSearch(EvalFun, EvalGrad, theta, step_direction,
                          step_scale=0.5, max_iters=5000,
                          step_max=100, initial_step=1,
                          fn_scale=fn_scale, verbose=FALSE)
  
}

optim_result <- NewtonsMethod(opt_fns$OptimVal, opt_fns$OptimGrad, opt_fns$OptimHess,
                              theta_init=optim_result0$par, fn_scale=-1, tol=1e-8,
                              verbose=TRUE)
any(abs(optim_result$theta - bounds$theta_lower[mask]) < 1e-8) ||
  any(abs(optim_result$theta - bounds$theta_upper[mask]) < 1e-8)

base_theta <- GetVectorFromParameters(vp_base, TRUE)
base_theta[mask] <- optim_result$theta
vp_opt <- GetParametersFromVector(vp_base, base_theta, TRUE)

vp_mom <- GetMoments(vp_opt)
moment_derivs <- GetMomentJacobian(vp_opt)
jac <- Matrix(moment_derivs$hess)

elbo_hess <- opt_fns$OptimHess(optim_result$theta)

optim_time <- Sys.time() - optim_time



lrvb_cov <- -1 * jac %*% Matrix::solve(elbo_hess, Matrix::t(jac))
min(diag(lrvb_cov))

mfvb_cov <- GetCovariance(vp_opt)
min(diag(mfvb_cov))
plot(sqrt(diag(lrvb_cov)), sqrt(diag(mfvb_cov))); abline(0, 1)

mfvb_sd <- GetMomentsFromVector(vp_mom, sqrt(diag(mfvb_cov)))
lrvb_sd <- GetMomentsFromVector(vp_mom, sqrt(diag(lrvb_cov)))



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

