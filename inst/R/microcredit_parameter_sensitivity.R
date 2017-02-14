library(ggplot2)
library(dplyr)
library(reshape2)
library(Matrix)
library(gridExtra)

library(MicrocreditLRVB)

# Load previously computed Stan results
# analysis_name <- "simulated_data_robust"
# analysis_name <- "simulated_data_nonrobust"
# analysis_name <- "simulated_data_lambda_beta"
# analysis_name <- "real_data_informative_priors"
analysis_name <- "real_data_t_prior"


project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/simulated_data")

# If true, save the results to a file readable by knitr.
save_results <- TRUE
results_file <- file.path(project_directory,
                          paste(analysis_name, "parameteric_sensitivity.Rdata", sep="_"))


#################################################
# Load MCMC and VB results.

fit_file <- file.path(project_directory, paste(analysis_name, "_mcmc_and_vb.Rdata", sep=""))
print(paste("Loading fits from ", fit_file))

LoadIntoEnvironment <- function(filename) {
  my_env <- environment()
  load(filename, envir=my_env)
  return(my_env)
}

fit_env <- LoadIntoEnvironment(fit_file)

stan_results <- fit_env$mcmc_environment$results$original
stan_results_perturb <- fit_env$mcmc_environment$results$perturbed

x <- stan_results$dat$x
y <- stan_results$dat$y
y_g <- stan_results$dat$y_group

pp <- fit_env$mcmc_environment$pp
pp_perturb <- fit_env$mcmc_environment$pp_perturb


###########################################
# Extract results

vp_opt <- fit_env$vb_fit$vp_opt

lrvb_cov <- fit_env$lrvb_terms$lrvb_cov

mp_opt <- GetMoments(vp_opt)
mfvb_cov <- GetCovariance(vp_opt)

# Convenient indices
vp_indices <- GetParametersFromVector(vp_opt, as.numeric(1:vp_opt$encoded_size), FALSE)
mp_indices <- GetMomentsFromVector(mp_opt, as.numeric(1:mp_opt$encoded_size))
pp_indices <- GetPriorsFromVector(pp, as.numeric(1:pp$encoded_size))


#######################################
# VB sensitivity measures

# Calculate vb perturbed estimates
mp_opt_vec <- GetVectorFromMoments(mp_opt)
if (analysis_name == "simulated_data_lambda_beta") {
  vb_sens_vec <- prior_sens[, pp_indices$lambda_beta]
} else if (analysis_name %in% c("real_data_informative_priors",
                                "simulated_data_robust",
                                "simulated_data_nonrobust")) {
  vb_sens_vec <- prior_sens[, pp_indices$mu_info[1, 2]]
} else if (analysis_name == "real_data_t_prior") {
  vb_sens_vec <- prior_sens[, pp_indices$lambda_beta]
} else {
  stop("Bad analysis name")
}

mp_opt_vec_pert <- mp_opt_vec + fit_env$mcmc_environment$perturb_epsilon * vb_sens_vec
mp_opt_lrvb_pert <- GetMomentsFromVector(mp_opt, mp_opt_vec_pert)


# Make a dataframe out of the prior sensitivity
prior_sens_norm <- fit_env$prior_sens / sqrt(diag(lrvb_cov))
vb_sensitivity_results <-
  SummarizePriorSensitivityMatrix(prior_sens_norm, pp_indices, mp_opt, method="lrvb")


##########################################
# MCMC sensitivity measures

GetMCMCNormalizedCovarianceSensitivity <- function(draws_mat, log_prior_grad_mat, keep_rows=nrow(draws_mat)) {
  keep_rows <- min(c(nrow(log_prior_grad_mat), keep_rows))
  draws_mat_small <- t(draws_mat[1:keep_rows, ])
  draws_mat_small <- draws_mat_small - rowMeans(draws_mat_small)
  
  mcmc_sd_scale_small <- sqrt(diag(cov(t(draws_mat_small)))) 
  log_prior_grad_mat_small <- log_prior_grad_mat[1:keep_rows, ]

  draws_mat_small_norm <- draws_mat_small / mcmc_sd_scale_small
  prior_sens_mcmc_norm_small <- draws_mat_small_norm  %*% log_prior_grad_mat_small / keep_rows
  prior_sens_mcmc_norm_squares <- (draws_mat_small_norm ^ 2)  %*% (log_prior_grad_mat_small ^ 2) / keep_rows
  prior_sens_mcmc_norm_sd <- sqrt(prior_sens_mcmc_norm_squares - prior_sens_mcmc_norm_small ^ 2) / sqrt(keep_rows)
  
  return(list(prior_sens_norm=prior_sens_mcmc_norm_small, prior_sens_norm_sd=prior_sens_mcmc_norm_sd))
}

mcmc_prior_sens_list <- GetMCMCNormalizedCovarianceSensitivity(
  fit_env$mcmc_environment$draws_mat, fit_env$mcmc_environment$log_prior_grad_mat)
mcmc_sensitivity_results <- SummarizePriorSensitivityMatrix(
  mcmc_prior_sens_list$prior_sens_norm, pp_indices, mp_opt, method="mcmc")

mcmc_subset_size <- 50
mcmc_prior_sens_subset_list <- GetMCMCNormalizedCovarianceSensitivity(
  fit_env$mcmc_environment$draws_mat, fit_env$mcmc_environment$log_prior_grad_mat, keep_rows=mcmc_subset_size)
mcmc_subset_sensitivity_results <- SummarizePriorSensitivityMatrix(
  mcmc_prior_sens_subset_list$prior_sens_norm, pp_indices, mp_opt, method="mcmc_subset")


###########################
# Summarize results

# Pack the standard deviations into readable forms.
mfvb_sd <- GetMomentsFromVector(mp_opt, sqrt(diag(mfvb_cov)))
lrvb_sd <- GetMomentsFromVector(mp_opt, sqrt(diag(lrvb_cov)))

results_lrvb <- SummarizeMomentParameters(mp_opt, mfvb_sd, lrvb_sd)
results_lrvb_pert <- SummarizeMomentParameters(mp_opt_lrvb_pert, mfvb_sd, lrvb_sd)
results_lrvb_pert$method <- "lrvb_perturbed"

# Get the VB result from manually perturbing and refitting
vp_opt_pert <- fit_env$vb_fit_perturb$vp_opt
mp_opt_pert <- GetMoments(vp_opt_pert)
lrvb_terms_pert <- GetLRVB(x, y, y_g, vp_opt_pert, pp_perturb)
prior_sens_pert <- GetSensitivity(vp_opt_pert, pp_perturb, lrvb_terms_pert$jac, lrvb_terms_pert$elbo_hess)

results_vb <- SummarizeMomentParameters(mp_opt, mfvb_sd, lrvb_sd)
results_vb_pert <- SummarizeMomentParameters(mp_opt_pert, mfvb_sd, lrvb_sd)
results_vb_pert$method <- "mfvb_perturbed"

stan_results$sim
results_mcmc <- SummarizeMCMCResults(fit_env$mcmc_environment$mcmc_sample)
results_mcmc_pert <- SummarizeMCMCResults(fit_env$mcmc_environment$mcmc_sample_perturbed)
results_mcmc_pert$method <- "mcmc_perturbed"

results <- rbind(results_vb, results_mcmc)
results_pert <- rbind(results_vb, results_lrvb_pert, results_vb_pert, results_mcmc, results_mcmc_pert)

#######################
# Graphs

stop("Graphs follow -- not executing.")

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

ggplot(filter(mean_results, par == "lambda")) +
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

###################
# Perturbations

mean_pert_results <-
  filter(results_pert, metric == "mean") %>%
  dcast(par + component + group ~ method, value.var="val") %>%
  mutate(mfvb_diff = mfvb_perturbed - mfvb,
         lrvb_diff = lrvb_perturbed - mfvb,
         mcmc_diff = mcmc_perturbed - mcmc)

# MCMC vs MFVB after manually perturbing and refitting.
ggplot(filter(mean_pert_results, par != "mu_g")) +
  geom_point(aes(x=mcmc_diff, y=mfvb_diff, color=par), size=3) +
  geom_abline(aes(slope=1, intercept=0))

ggplot(filter(mean_pert_results, par == "mu_g")) +
  geom_point(aes(x=mcmc_diff, y=mfvb_diff, color=par), size=3) +
  geom_abline(aes(slope=1, intercept=0))

# MCMC vs predicted change.
ggplot(filter(mean_pert_results, par != "mu_g")) +
  geom_point(aes(x=mcmc_diff, y=lrvb_diff, color=par), size=3) +
  geom_abline(aes(slope=1, intercept=0))

ggplot(filter(mean_pert_results, par == "mu_g")) +
  geom_point(aes(x=mcmc_diff, y=lrvb_diff, color=par), size=3) +
  geom_abline(aes(slope=1, intercept=0))


# LRVB predicted vs actual change.  This should be spot on.
ggplot(filter(mean_pert_results, par != "mu_g")) +
  geom_point(aes(x=mfvb_diff, y=lrvb_diff, color=par), size=3) +
  geom_abline(aes(slope=1, intercept=0))

ggplot(filter(mean_pert_results, par == "mu_g")) +
  geom_point(aes(x=mfvb_diff, y=lrvb_diff, color=par), size=3) +
  geom_abline(aes(slope=1, intercept=0))


#############

prior_sens_results <-
  rbind(mcmc_sensitivity_results, mcmc_subset_sensitivity_results, vb_sensitivity_results)

prior_sens_results_graph <-
  dcast(prior_sens_results, par + component + group + metric ~ method, value.var="val")

grid.arrange(
  ggplot(prior_sens_results_graph) +
    geom_point(aes(x=lrvb, y=mcmc, color=par), size=3) +
    geom_abline(aes(slope=1, intercept=0)) +
    ggtitle(paste("Local sensitivity to prior parameters"))
,
ggplot(prior_sens_results_graph) +
  geom_point(aes(x=lrvb, y=mcmc_subset, color=par), size=3) +
  geom_abline(aes(slope=1, intercept=0)) +
  ggtitle(paste("Local sensitivity to prior parameters\nMCMC data subset ", mcmc_subset_size))
, ncol=2
)



#############

# Full bar chart of the vb changes
sens_graph_df <- filter(vb_sensitivity_results)
# sens_graph_df <- filter(vb_sensitivity_results, par == "mu",
#                         metric %in% c("mu_info_11", "mu_info_12", "mu_info_22", "lambda_eta"))
sens_breaks <- unique(sens_graph_df$metric)

WrapInExpressions <- function(string_vec) {
  ExprWrap <- function(x) { paste("expression(", x, ")", collapse="", sep="") }
  expr_string_vec <- sapply(string_vec, ExprWrap)
  paste("c(", paste(expr_string_vec, collapse=", "), ")", sep="")
}


ggplot(sens_graph_df) +
  geom_bar(aes(x=paste(par, component), y=val, fill=metric),
           position="dodge", stat="identity") +
  scale_fill_discrete(name="Prior parameter",
                      breaks=sens_breaks,
                      labels=eval(parse(text=WrapInExpressions(unique(sens_graph_df$expression))))) +
  scale_x_discrete(breaks=c("e_mu_1_-1", "e_mu_2_-1"),
                   labels=c(expression(mu), expression(tau))) +
  xlab("Prior top-level mean component") + ylab("Sensitivity") +
  ggtitle("Derivative of the posterior mean")


ggplot(sens_graph_df) +
  geom_bar(aes(x=paste(par, component), y=val, fill=metric),
           position="dodge", stat="identity") +
  xlab("Prior top-level mean component") + ylab("Sensitivity") +
  ggtitle("Derivative of the posterior mean")


#######################################
# Export selected results for use in the paper

if (save_results) {
  num_obs <- nrow(x)
  save(results_pert, pp, vb_sensitivity_results, prior_sens_results_graph, num_obs, file=results_file)
}

