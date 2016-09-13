library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)

library(MicrocreditLRVB)

# Load previously computed Stan results
#analysis_name <- "simulated_data_robust"
#analysis_name <- "simulated_data_nonrobust"
analysis_name <- "simulated_data_lambda_beta"

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/simulated_data")

stan_draws_file <-
  file.path(project_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))
print(paste("Loading draws from ", stan_draws_file))

stan_results <- environment()
load(stan_draws_file, envir=stan_results)
stan_results <- as.list(stan_results)

x <- stan_results$stan_dat$x
y <- stan_results$stan_dat$y
y_g <- stan_results$stan_dat$y_group


#############################
# Initialize

# "reg" for "regression"
vp_reg <- InitializeVariationalParameters(
  x, y, y_g, mu_diag_min=0.01, lambda_diag_min=1e-5, tau_min=1, lambda_n_min=0.5)
mp_reg <- GetMoments(vp_reg)


#########################################
# Fit and LRVB

vb_fit <- FitVariationalModel(x, y, y_g, vp_reg, pp)
vp_opt <- vb_fit$vp_opt
print(vb_fit$bfgs_time + vb_fit$tr_time)
lrvb_terms <- GetLRVB(x, y, y_g, vp_opt, pp)
mp_opt <- GetMoments(vp_opt)

# Convenient indices
vp_indices <- GetParametersFromVector(vp_opt, as.numeric(1:vp_opt$encoded_size), FALSE)
mp_indices <- GetMomentsFromVector(mp_opt, as.numeric(1:mp_opt$encoded_size))
pp_indices <- GetPriorsFromVector(pp, as.numeric(1:pp$encoded_size))

lrvb_cov <- lrvb_terms$lrvb_cov
mfvb_cov <- GetCovariance(vp_opt)

#################
# Pick a parameter to perturb and re-fit

# pp_perturb <- pp
# perturb_epsilon <- 0.01
# pp_perturb$mu_loc[1] <- pp$mu_loc[1] + perturb_epsilon;
# pp_perturb_index <- pp_indices$mu_loc[1]

# pp_perturb <- pp
# perturb_epsilon <- 0.01
# pp_perturb$lambda_eta <- pp_perturb$lambda_eta + perturb_epsilon;
# pp_perturb_index <- pp_indices$lambda_eta

# pp_perturb <- pp
# perturb_epsilon <- 0.3
# pp_perturb$mu_info[1, 1] <- pp_perturb$mu_info[1, 1] + perturb_epsilon;
# pp_perturb_index <- pp_indices$mu_info[1, 1]

pp_perturb <- pp
perturb_epsilon <- 2
pp_perturb$lambda_alpha <- pp_perturb$lambda_alpha + perturb_epsilon;
pp_perturb_index <- pp_indices$lambda_alpha

vb_fit_perturb <- FitVariationalModel(x, y, y_g, vp_opt, pp_perturb)
vp_opt_perturb <- vb_fit_perturb$vp_opt
mp_opt_perturb <- GetMoments(vp_opt_perturb)
lrvb_terms_perturb <- GetLRVB(x, y, y_g, vp_opt_perturb, pp_perturb)


##################
# Get sensitivity by hand.

# prior_sens <- GetSensitivity(vp_opt, pp, lrvb_terms$jac, lrvb_terms$elbo_hess)
comb_indices <- GetPriorsAndParametersFromVector(
  vp_opt, pp, as.numeric(1:(vp_opt$encoded_size + pp$encoded_size)))
comb_prior_ind <- GetVectorFromPriors(comb_indices$pp)
comb_vp_ind <- GetVectorFromParameters(comb_indices$vp, FALSE)

log_prior_derivs <- GetLogPriorDerivatives(vp_opt, pp, TRUE, TRUE, TRUE)
log_prior_param_prior <- Matrix(log_prior_derivs$hess[comb_vp_ind, comb_prior_ind])
cbind(log_prior_param_prior[1:12, pp_indices$lambda_alpha],
      log_prior_param_prior[1:12, pp_indices$lambda_beta],
      log_prior_param_prior[1:12, pp_indices$lambda_eta])
log_prior_param_prior[1:12, ]

prior_sens <- -1 * lrvb_terms$jac %*% Matrix::solve(lrvb_terms$elbo_hess, log_prior_param_prior)

jac_ev <- eigen(lrvb_terms$jac)$values
max(abs(jac_ev)) / min(abs(jac_ev))

hess_evec <- eigen(lrvb_terms$elbo_hess)$vector
hess_ev <- eigen(lrvb_terms$elbo_hess)$values
max(abs(hess_ev)) / min(abs(hess_ev))


#######################################
# VB sensitivity measures

# Calculate vb perturbed estimates
mp_opt_vec <- GetVectorFromMoments(mp_opt)
vb_sens_vec <- prior_sens[, pp_perturb_index]
mp_opt_vec_pert <- mp_opt_vec + perturb_epsilon * vb_sens_vec
mp_opt_lrvb_pert <- GetMomentsFromVector(mp_opt, mp_opt_vec_pert)


###########################
# Summarize results

# Pack the standard deviations into readable forms.
mfvb_sd <- GetMomentsFromVector(mp_opt, sqrt(diag(mfvb_cov)))
lrvb_sd <- GetMomentsFromVector(mp_opt, sqrt(diag(lrvb_cov)))

results_vb <- SummarizeRawMomentParameters(mp_opt, metric="mean", method="mfvb")
results_lrvb <- SummarizeRawMomentParameters(mp_opt_lrvb_pert, metric="mean", method="lrvb_pred")
results_vb_pert <- SummarizeRawMomentParameters(mp_opt_perturb, metric="mean", method="mfvb_perturbed")

results <- rbind(results_vb, results_vb_pert, results_lrvb)

#######################
# Graphs

# stop("Graphs follow -- not executing.")

mean_results <-
  dcast(results, par + component + group ~ method, value.var="val")

ggplot(mean_results) +
  geom_point(aes(x=mfvb_perturbed - mfvb, y=lrvb_pred - mfvb, color=par), size=3) +
  geom_abline(aes(slope=1, intercept=0))
