library(ggplot2)
library(dplyr)
library(reshape2)
library(Matrix)

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
prior_sens <- fit_env$prior_sens

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
} else {
  vb_sens_vec <- prior_sens[, pp_indices$mu_info[1, 2]]
}

mp_opt_vec_pert <- mp_opt_vec + fit_env$mcmc_environment$perturb_epsilon * vb_sens_vec
mp_opt_lrvb_pert <- GetMomentsFromVector(mp_opt, mp_opt_vec_pert)


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
# results_mcmc <- SummarizeMCMCResults(fit_env$stan_results$mcmc_sample)
# results_mcmc_pert <- SummarizeMCMCResults(fit_env$stan_results$mcmc_sample_perturbed)
results_mcmc <- SummarizeMCMCResults(fit_env$mcmc_environment$mcmc_sample)
results_mcmc_pert <- SummarizeMCMCResults(fit_env$mcmc_environment$mcmc_sample_perturbed)
results_mcmc_pert$method <- "mcmc_perturbed"

results <- rbind(results_vb, results_mcmc)
results_pert <- rbind(results_vb, results_lrvb_pert, results_vb_pert, results_mcmc, results_mcmc_pert)

# A result summarizing the VB prior sensitivity
vb_sensitivity_results <- data.frame()
AppendVBSensitivityResults <- function(ind, prior_param, display_expression) {
  this_mp <- GetMomentsFromVector(mp_opt, prior_sens[, ind])
  vb_sensitivity_results <<- rbind(
    vb_sensitivity_results,
    SummarizeRawMomentParameters(this_mp, metric=prior_param, method="lrvb") %>%
      mutate(expression=display_expression))
}
AppendVBSensitivityResults(pp_indices$mu_info[1, 2], "mu_info_offdiag", "Lambda[12]^mu")
AppendVBSensitivityResults(pp_indices$mu_info[1, 1], "mu_info_11", "Lambda[11]^mu")
AppendVBSensitivityResults(pp_indices$mu_info[1, 1], "mu_info_22", "Lambda[22]^mu")
AppendVBSensitivityResults(pp_indices$lambda_eta, "lambda_eta", "eta")
AppendVBSensitivityResults(pp_indices$lambda_alpha, "lambda_alpha", "Lambda^alpha")
AppendVBSensitivityResults(pp_indices$lambda_beta, "lambda_beta", "Lambda^beta")
AppendVBSensitivityResults(pp_indices$tau_alpha, "tau_alpha", "tau[alpha]")
AppendVBSensitivityResults(pp_indices$tau_beta, "tau_beta", "tau[beta]")


##########################################
# Get MCMC sensitivity measures
mcmc_sample <- fit_env$stan_results$mcmc_sample
draws_mat <- fit_env$stan_results$draws_mat
mp_draws <- fit_env$stan_results$mp_draws
log_prior_grad_mat <- fit_env$stan_results$log_prior_grad_mat

# log_prior_grad_list <- GetMCMCLogPriorDerivatives(mp_draws, pp)
# log_prior_grad_mat <- do.call(rbind, log_prior_grad_list)


#######################
# Prior sensitivity results with respect to a particular prior parameter.

# TODO: use these after you've fixed the LKJ prior
draws_mat <- fit_env$mcmc_environment$draws_mat
log_prior_grad_mat <- fit_env$mcmc_environment$log_prior_grad_mat

prior_sensitivity_ind <- pp_indices$mu_info[1, 2]; ind_name <- "mu_info_offdiag_sens"
# prior_sensitivity_ind <- pp_indices$mu_info[1, 1]; ind_name <- "mu_info_diag_sens"
# prior_sensitivity_ind <- pp_indices$lambda_eta; ind_name <- "lambda_eta"
# prior_sensitivity_ind <- pp_indices$lambda_alpha; ind_name <- "lambda_alpha"
# prior_sensitivity_ind <- pp_indices$lambda_beta; ind_name <- "lambda_beta"
# prior_sensitivity_ind <- pp_indices$tau_alpha; ind_name <- "tau_alpha"
# prior_sensitivity_ind <- pp_indices$tau_alpha; ind_name <- "tau_beta"
# prior_sensitivity_ind <- pp_indices$mu_loc[1]; ind_name <- "mu_loc_1"

mcmc_subset_size <- 1000
mcmc_prior_sens_subset <- cov(draws_mat[1:mcmc_subset_size, ],
                              log_prior_grad_mat[1:mcmc_subset_size, ])

mcmc_prior_sens <- cov(draws_mat, log_prior_grad_mat)

vb_prior_sens_mom <- GetMomentsFromVector(mp_opt, prior_sens[, prior_sensitivity_ind])
mcmc_prior_sens_mom <- GetMomentsFromVector(mp_opt, mcmc_prior_sens[, prior_sensitivity_ind])
mcmc_prior_sens_subset <- GetMomentsFromVector(mp_opt, mcmc_prior_sens_subset[, prior_sensitivity_ind])
prior_sens_results <-
  rbind(SummarizeRawMomentParameters(vb_prior_sens_mom, metric=ind_name, method="lrvb") %>%
          mutate(draws=""),
        SummarizeRawMomentParameters(mcmc_prior_sens_mom, metric=ind_name, method="mcmc") %>%
          mutate(draws=nrow(draws_mat)),
        SummarizeRawMomentParameters(mcmc_prior_sens_subset, metric=ind_name, method="mcmc_subset") %>%
          mutate(draws=mcmc_subset_size)
  )


prior_sens_results_graph <-
  inner_join(
  filter(prior_sens_results, method == "lrvb") %>%
    dcast(par + component + group + metric ~ method, value.var="val"),
  filter(prior_sens_results, method == "mcmc") %>%
    dcast(par + component + group + metric + draws ~ method, value.var="val"),
  by=c("par", "component", "group", "metric"))

prior_sens_results_graph$ind_name <- ind_name

ggplot(prior_sens_results_graph) +
  geom_point(aes(x=lrvb, y=mcmc, color=par), size=3) +
  geom_abline(aes(slope=1, intercept=0)) +
  facet_grid(~ draws) +
  ggtitle(paste("Local sensitivity to", unique(prior_sens_results_graph$ind_name),
                "as measured by VB and MCMC, grouped by # of MCMC draws"))


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

