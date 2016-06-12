library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)
library(mvtnorm)

library(MicroCreditLRVB)

# Load previously computed Stan results
analysis_name <- "simulated_data"
project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/simulated_data")

stan_draws_file <-
  file.path(project_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))
print(paste("Loading draws from ", stan_draws_file))
load(stan_draws_file)

x <- stan_dat$x
y <- stan_dat$y
y_g <- stan_dat$y_group

##################

jacobian_check <- TestJacobian()
transpose_jacobian <- any(dim(jacobian_check$A) != dim(jacobian_check$jac))

MakeSymmetric <- function(mat) {
  return(0.5 * (mat + t(mat)))
}


# In the paper, mu[1] is mu and mu[2] is tau and tau is 1 / sigma^2.
PaperParam <- function(param, component) {
  if (param == "mu") {
    if (component == 1) {
      return("mu")
    } else {
      return("tau")
    }
  }
  if (param == "mu_g") {
    return(paste(PaperParam("mu", component), "g", sep="_"))
  }
  if (param == "log_tau") {
    return("minus_log_sigma_sq")
  }
  return(param)
}


################################
# Initialize the data

vp <- InitializeVariationalParameters(x, y, y_g)

# Prior parameters
pp <- list()
pp[["k"]] <- vp$k
pp[["mu_mean"]] <- rep(0, vp$k)
pp[["mu_info"]] <- matrix(c(0.03, 0., 0, 0.02), vp$k, vp$k)
pp[["lambda_eta"]] <- 15.01
pp[["lambda_alpha"]] <- 20.01
pp[["lambda_beta"]] <- 20.01
pp[["tau_alpha"]] <- 2.01
pp[["tau_beta"]] <- 2.01

# Optimization parameters stored in the prior:
pp[["lambda_diag_min"]] <- lambda_diag_min
pp[["lambda_n_min"]] <- vp$k + 0.5


##########
# Fit it with VB

max_iters <- 500
vb_tol <- 1e-6
vb_time <- Sys.time()
vb_fit <- FitModel(x, y, y_g, vp, pp,
                   num_iters=max_iters, rel_tol=vb_tol, fit_lambda=TRUE, verbose=TRUE)
vb_time <- Sys.time() - vb_time

# LRVB stuff:
lrvb_time <- Sys.time()

mfvb_cov <- MakeSymmetric(GetVariationalCovariance(vb_fit$vp, pp))

lambda_lik_derivs <-
  LambdaLikelihoodMomentDerivs(x, y, y_g, vb_fit$vp, pp, TRUE)

dmoment_dpar_t <- Diagonal(encoder$dim)
dmoment_dpar_t[lambda_parameters_indices, lambda_parameters_indices] <-
  t(lambda_lik_derivs$dmoment_dtheta)

obs_hess <- Matrix(MakeSymmetric(ll_derivs$obs_hess))
h_mat <- solve(dmoment_dpar_t, t(solve(dmoment_dpar_t, obs_hess)))
h_mat[lambda_parameters_indices, lambda_parameters_indices] <-
  MakeSymmetric(lambda_lik_derivs$d2l_dm2)

lrvb_id_mat <- Diagonal(encoder$dim)

lrvb_inv_term_orig <- (lrvb_id_mat - mfvb_cov %*% h_mat)
lrvb_cov <- MakeSymmetric(solve(lrvb_inv_term_orig, mfvb_cov))

lrvb_time <- Sys.time() - lrvb_time


################################################
# VB sensitivity

k_ud <- vp$k * (vp$k + 1) / 2
model_encoder <- GetModelParameterEncoder(vp, pp)

prior_encoder <- GetPriorParameterEncoder(pp)

# Get some useful indices
moment_ind <- model_encoder$variational_offset + 1:model_encoder$variational_dim
prior_ind <- model_encoder$prior_offset + 1:(model_encoder$prior_dim)

mu_info_ind <- prior_encoder$mu_info_offset + 1:k_ud
mu_ind <- encoder$e_mu + 1:vp$k

prior_derivs <- PriorSensitivity(vb_fit$vp, pp)
prior_sub_hess <- prior_derivs$prior_hess[moment_ind, prior_ind]

# Change the derivatives to be with respect to the moments
# It should be transposed because the moments
# must be in the rows, and the inverse of the Jacobian is the Jacobian of the
# inverse, meaning the moments would be in the columns.
prior_sens_bad <- lrvb_cov %*% solve(t(dmoment_dpar), prior_sub_hess)
prior_sens <- solve(lrvb_inv_term_orig, mfvb_cov %*% solve(t(dmoment_dpar), prior_sub_hess))
max(abs(prior_sens_bad - prior_sens))
#image(Matrix(prior_sens_bad - prior_sens))
# Note that t(dmoment_dpar) could be canceled explicitly avoiding an inverse.

GetSensitivityDataframe <- function(offset, metric) {
  prior_sens_this <- prior_sens[, offset]
  prior_sens_this_list <- DecodeParameters(prior_sens_this, vb_fit$vp, pp, FALSE)
  prior_sens_this_df <- ConvertParameterListToDataframe(prior_sens_this_list, metric)
  prior_sens_this_df$param <- sub("lambda_v_par", "lambda", prior_sens_this_df$param)
  prior_sens_this_df$method <- "lrvb"
  return(prior_sens_this_df)
}

# This will be compared to MCMC:
prior_sens_lambda_11_df <-
  GetSensitivityDataframe(prior_encoder$mu_info_offset + 1, "lambda_11_sens")

# These will be graphed in the paper:
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

# lambda_eta seems off based on finite differences...
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

prior_sens_df$paper_param <- ""
for (i in 1:nrow(prior_sens_df)) {
  prior_sens_df[i, "paper_param"] <- PaperParam(prior_sens_df[i, "param"], prior_sens_df[i, "component"])
}


mcmc_sample <- extract(stan_sim)
mcmc_sample_perturb <- extract(stan_sim_perturb)


###################
# Put the results in a tidy format and graph

result <- GetResultDataframe(mcmc_sample, vb_fit$vp, lrvb_cov, mfvb_cov, encoder)
result_perturb <- GetResultDataframe(mcmc_sample_perturb, vb_fit$vp, lrvb_cov, mfvb_cov, encoder)

result_perturb <- filter(result_perturb, method=="mcmc")
result_perturb$method <- "mcmc_perturbed"
result_perturb_diff <-
  rbind(filter(result, method=="mcmc"), result_perturb) %>%
  filter(metric == "mean") %>% dplyr::select(-matches("metric")) %>%
  dcast(param + component + group ~ method) %>%
  mutate(diff = mcmc_perturbed - mcmc) %>%
  mutate(value = diff / perturb_epsilon, metric="lambda_11_sens", method="mcmc") %>%
  # filter(diff > 0.05) %>% # Get rid of differences too small to detect with MCMC error
  dplyr::select(-mcmc, -mcmc_perturbed, -diff) %>%
  rbind(prior_sens_lambda_11_df) %>%
  dcast(param + component + group + metric ~ method) %>%
  filter(!is.na(mcmc))

result$paper_param <- ""
for (i in 1:nrow(result)) {
  result[i, "paper_param"] <- PaperParam(result[i, "param"], result[i, "component"])
}

result_perturb_diff$paper_param <- ""
for (i in 1:nrow(result_perturb_diff)) {
  result_perturb_diff[i, "paper_param"] <-
    PaperParam(result_perturb_diff[i, "param"], result_perturb_diff[i, "component"])
}



if (FALSE) {
  ggplot(filter(result, metric == "sd", param == "lambda") %>%
           dcast(param + component + group ~ method)) +
    geom_point(aes(x=mcmc, y=mfvb, color="mfvb"), size=3) +
    geom_point(aes(x=mcmc, y=lrvb, color="lrvb"), size=3) +
    geom_abline(aes(slope=1, intercept=0)) +
    expand_limits(x=0, y=0) + expand_limits(x=1, y=1)

  ggplot(result_perturb_diff) +
    geom_point(aes(x=mcmc, y=lrvb, color=param), size=2) +
    geom_abline(aes(slope=1, intercept=0))


  ggplot(filter(result, metric == "mean") %>%
           dcast(param + component + group ~ method)) +
    geom_point(aes(x=mcmc, y=mfvb, color=param), size=3) +
    geom_abline(aes(slope=1, intercept=0)) +
    expand_limits(x=0, y=0) + expand_limits(x=1, y=1)

  ggplot(filter(result, metric == "mean", param == "lambda") %>%
           dcast(param + component + group ~ method)) +
    geom_point(aes(x=mcmc, y=mfvb, color=param), size=3) +
    geom_abline(aes(slope=1, intercept=0)) +
    expand_limits(x=0, y=0) + expand_limits(x=1, y=1)
}




#############################
# Look at the paper's tau parameter in more detail.

tau_results <- filter(result, param %in% c("mu", "mu_g"), component == 2) %>%
  dcast(component + param + group ~ metric + method) %>%
  dplyr::select(-sd_mfvb)

tau_results$sig_mcmc <- abs(tau_results$mean_mcmc / tau_results$sd_mcmc) > 2
tau_results$sig_lrvb <- abs(tau_results$mean_mfvb / tau_results$sd_lrvb) > 2

if (F) {
  ggplot(tau_results) +
    geom_point(aes(x=mean_mcmc, y=mean_mfvb)) + geom_abline(slope=1, intercept=0)

  ggplot(tau_results) +
    geom_point(aes(x=sd_mcmc, y=sd_lrvb)) + geom_abline(slope=1, intercept=0)
}

tau_sens <-
  filter(prior_sens_df, param == "mu", component == 2) %>%
  dcast(component + param + group + lrvb_sd ~ metric)

tau_sens$lambda_1 / tau_sens$lrvb_sd

tau_results <-
  filter(result, param == "mu", component == 2) %>%
  dcast(component + param + group ~ metric + method) %>%
  dplyr::select(-sd_mfvb)


######################################
# Graphs for the paper


if (F) {
  knitr_data_file_name <- paste(analysis_name, "_results.Rdata", sep="")
  save(result, result_perturb_diff, prior_sens_df,
       mcmc_time, vb_time, lrvb_time, pp, file=file.path(data_path, knitr_data_file_name))

  # Posteriors
  ggplot(filter(result, metric == "mean") %>%
           dcast(paper_param + component + group ~ method)) +
    geom_point(aes(x=mcmc, y=mfvb, color=paper_param), size=3) +
    geom_abline(aes(slope=1, intercept=0)) +
    expand_limits(x=0, y=0) + expand_limits(x=1, y=1) +
    scale_color_discrete(name="Parameter",
                         breaks=c("mu", "mu_g", "tau", "tau_g", "minus_log_sigma_sq", "lambda"),
                         labels=c(expression(mu),
                                  expression(mu[k]),
                                  expression(tau),
                                  expression(tau[k]),
                                  expression(-log(sigma^2)),
                                  expression(C^-1)))

  ggplot(filter(result, metric == "sd") %>%
           dcast(param + component + group ~ method)) +
    geom_point(aes(x=mcmc, y=mfvb, color="MFVB"), size=3) +
    geom_point(aes(x=mcmc, y=lrvb, color="LRVB"), size=3) +
    geom_abline(aes(slope=1, intercept=0)) +
    scale_color_discrete(name="Method") +
    expand_limits(x=0, y=0) + expand_limits(x=1, y=1) +
    xlab("MCMC (ground truth posterior standard deviation)") +
    ylab("VB and LRVB standard deviations")



  # Sensitivity
  prior_sens_df$sens_over_sd <- prior_sens_df$value / prior_sens_df$lrvb_sd
  filter(prior_sens_df, param == "mu")
  # lambda_eta is a little suspect based on finite differences.  Let's be on the safe side
  # and exclude it from the paper.

  ggplot(filter(prior_sens_df, paper_param %in% c("tau"),
                metric %in% c("lambda_1", "lambda_2", "lambda_3"))) +
    geom_bar(aes(x=paper_param, y=value / lrvb_sd, fill=metric),
             position="dodge", stat="identity") +
    scale_fill_discrete(name="Prior parameter",
                        breaks=c("lambda_1",
                                 "lambda_2",
                                 "lambda_3"),
                        labels=c(expression(Lambda[11]),
                                 expression(Lambda[12]),
                                 expression(Lambda[22]))) +
    scale_x_discrete(breaks=c("tau"),
                     labels=c(expression(mu), expression(tau))) +
    xlab("Prior top-level mean component") + ylab("Sensitivity / sd") +
    ggtitle("Normalized sensitivity")

  ggplot(filter(prior_sens_df, paper_param %in% c("tau"),
                metric %in% c("mu_1", "mu_2"))) +
    geom_bar(aes(x=paper_param, y=value / lrvb_sd, fill=metric),
             position="dodge", stat="identity") +
    scale_fill_discrete(name="Prior parameter",
                        breaks=c("mu_1", "mu_2"),
                        labels=c(expression(mu[0]), expression(tau[0]))) +
    scale_x_discrete(breaks=c("tau"),
                     labels=c(expression(mu), expression(tau))) +
    xlab("Prior top-level mean component") + ylab("Sensitivity / sd") +
    ggtitle("Normalized sensitivity")



  ggplot(result_perturb_diff) +
    geom_point(aes(x=mcmc, y=lrvb, color=paper_param), size=2) +
    geom_abline(aes(slope=1, intercept=0)) +
    scale_color_discrete(name="Parameter",
                         breaks=c("mu", "mu_g", "tau", "tau_g", "minus_log_sigma_sq", "lambda"),
                         labels=c(expression(mu),
                                  expression(mu[k]),
                                  expression(tau),
                                  expression(tau[k]),
                                  expression(-log(sigma^2)),
                                  expression(C^-1))) +
    ylab("LRVB predicted change") + xlab("MCMC actual change")

}
