library(ggplot2)
library(dplyr)
library(reshape2)
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


###########
# Set missing prior parameters

pp$mu_t_loc <- 0
pp$mu_t_scale <- 1
pp$mu_t_df <- 20
pp$mu_student_t_prior <- FALSE
pp$encoded_size <- length(GetVectorFromPriors(pp))

#############################
# Initialize

# "reg" for "regression"
vp_reg <- InitializeVariationalParameters(
  x, y, y_g, mu_diag_min=0.01, lambda_diag_min=1e-5, tau_min=1, lambda_n_min=0.5)
mp_reg <- GetMoments(vp_reg)

n_mu_draws <- 10
vp_reg$mu_draws <- qnorm(seq(1 / (n_mu_draws + 1), 1 - 1 / (n_mu_draws + 1),
                             length.out = n_mu_draws))

#################
# Fit with a t prior

pp_perturb <- pp
pp_perturb$mu_student_t_prior <- TRUE

vb_fit_perturb <- FitVariationalModel(x, y, y_g, vp_reg, pp_perturb)
vp_opt_perturb <- vb_fit_perturb$vp_opt
mp_opt_perturb <- GetMoments(vp_opt_perturb)
lrvb_terms_perturb <- GetLRVB(x, y, y_g, vp_opt_perturb, pp_perturb)


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

###########################
# Summarize results

results <-
  rbind(SummarizeRawMomentParameters(mp_opt, metric="mean", method="mfvb_norm"),
        SummarizeRawMomentParameters(mp_opt_perturb, metric="mean", method="mfvb_t"))


###############################
# Epsilon sensitivity by monte carlo

n_sim <- 10000
t_draws <- rt(n_sim, pp_perturb$mu_t_df) * pp_perturb$mu_t_scale + pp_perturb$mu_t_loc

mp_draw <- mp_opt
terms_list <- list()
debug_list <- list()
for (sim in 1:n_sim){
  cat(".")
  mp_draw$mu_e_vec[1] <- mp_draw$mu_e_vec[2] <- t_draws[sim]
  mp_draw$mu_e_outer <- mp_draw$mu_e_vec %*% t(mp_draw$mu_e_vec)
  log_q_derivs <- GetLogVariationalDensityDerivatives(mp_draw, vp_opt, pp,
                                                      include_mu=TRUE, include_lambda=FALSE,
                                                      r_include_mu_groups=integer(),
                                                      r_include_tau_groups=integer(),
                                                      calculate_gradient=TRUE)

  # TODO: this is wrong because my prior values do not have the normalizing constants.
  log_prior_derivs <-
    GetObsLogPriorDerivatives(mp_draw, pp,
                              include_mu=TRUE, include_lambda=FALSE, include_tau=FALSE)
  
  terms_list[[sim]] <- exp(log_q_derivs$val - log_prior_derivs$val) * log_q_derivs$grad
  debug_list[[sim]] <- data.frame(log_q=log_q_derivs$val, log_prior=log_prior_derivs$val)
}

debug_df <- do.call(rbind, debug_list)



#######################
# Graphs

# stop("Graphs follow -- not executing.")

mean_results <-
  dcast(results, par + component + group ~ method, value.var="val")

ggplot(mean_results) +
  geom_point(aes(x=mfvb_norm, y=mfvb_t, color=par), size=3) +
  geom_abline(aes(slope=1, intercept=0))
