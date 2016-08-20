library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)

library(MicrocreditLRVB)

# Load previously computed Stan results
#analysis_name <- "simulated_data_easy"
analysis_name <- "simulated_data_nonrobust"

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

# "reg" for "regression"
vp_reg <- InitializeVariationalParameters(
  x, y, y_g, mu_diag_min=0.01, lambda_diag_min=1e-5, tau_min=1, lambda_n_min=0.5)
mp_reg <- GetMoments(vp_reg)

# Convenient indices
vp_indices <- GetParametersFromVector(vp_reg, as.numeric(1:vp_reg$encoded_size), FALSE)
mp_indices <- GetMomentsFromVector(mp_reg, as.numeric(1:vp_reg$encoded_size))
pp_indices <- GetPriorsFromVector(pp, as.numeric(1:pp$encoded_size))


#########################################
# Fit and LRVB

vb_fit <- FitVariationalModel(x, y, y_g, vp_reg, pp)
print(vb_fit$bfgs_time + vb_fit$tr_time)

vp_opt <- vb_fit$vp_opt
vp_mom <- GetMoments(vp_opt)
mfvb_cov <- GetCovariance(vp_opt)
lrvb_terms <- GetLRVB(x, y, y_g, vp_opt, pp)
lrvb_cov <- lrvb_terms$lrvb_cov
prior_sens <- GetSensitivity(vp_opt, pp, lrvb_terms$jac, lrvb_terms$elbo_hess)

# Calculate vb perturbed estimates
vp_mom_vec <- GetVectorFromMoments(vp_mom)
mu_info_offdiag_sens <- prior_sens[, pp_indices$mu_info[1, 2]]
vp_mom_vec_pert <- vp_mom_vec + stan_results$perturb_epsilon * mu_info_offdiag_sens
vp_mom_pert <- GetMomentsFromVector(vp_mom, vp_mom_vec_pert)


##########################################
# Get MCMC sensitivity measures

mcmc_sample <- extract(stan_results$stan_sim)
mcmc_sample_perturbed <- extract(stan_results$stan_sim_perturb)

draw <- 1

mu <- mcmc_sample$mu[draw, ]
lambda <- mcmc_sample$lambda[draw, , ]
tau <- 1 / mcmc_sample$sigma_y[draw, ]^2
mu_g <- mcmc_sample$mu1[draw, , ]

# Evidently Rcpp ignores all but the last index for NumericVectors.
library(Rcpp)
cppFunction(
"
void FooFun8(Rcpp::NumericVector foo, int i, int j, int k) {
  Rcpp::Rcout << foo[0, 1, 0] << \"\\n\";
  Rcpp::Rcout << foo[0, 1, 1] << \"\\n\";
  Rcpp::Rcout << foo[0, 1, 2] << \"\\n\";
  Rcpp::Rcout << foo[0, 1, 3] << \"\\n\";
  Rcpp::Rcout << foo[0, 0, 0] << \"\\n\";
  Rcpp::Rcout << foo[0, 0, 1] << \"\\n\";
  Rcpp::Rcout << foo[1, 1, 0] << \"\\n\";
  Rcpp::Rcout << foo[1, 1, 1] << \"\\n\";
  Rcpp::Rcout << foo[10000, 23, 1001010011, 1, 1, 1] << \"\\n\";
  Rcpp::Rcout << foo.size() << \"\\n\";
  Rcpp::NumericVector foo_dims = foo.attr(\"dim\");
  Rcpp::Rcout << foo_dims << \"\\n\";
}"
)

foo <- array(as.numeric(1:24), dim=c(2, 3, 4))
attr(x, "dim")
FooFun8(foo)
foo[1, 2, 3]


cppFunction(
  "
void IndexIntoFoo(Rcpp::NumericVector foo, int i, int j, int k) {
  Rcpp::NumericVector foo_dims = foo.attr(\"dim\");
  if (foo_dims.size() != 3) {
    throw std::runtime_error(\"no way jose\");
  }
  int ind = i + foo_dims[0] * j + foo_dims[0] * foo_dims[1] * k;
  Rcpp::Rcout << foo[ind] << \"\\n\";
}"
)


foo <- array(as.numeric(1:24), dim=c(2, 3, 4))
attr(x, "dim")
IndexIntoFoo(foo, 0, 1, 3)
foo[1, 2, 4]


# This is over-designing.  Put the draws into an array of lists of moment parameters.


###########################
# Summarize results

# Pack the standard deviations into readable forms.
mfvb_sd <- GetMomentsFromVector(vp_mom, sqrt(diag(mfvb_cov)))
lrvb_sd <- GetMomentsFromVector(vp_mom, sqrt(diag(lrvb_cov)))

results_vb <- SummarizeMomentParameters(vp_mom, mfvb_sd, lrvb_sd)
results_vb_pert <- SummarizeMomentParameters(vp_mom_pert, mfvb_sd, lrvb_sd)
results_vb_pert$method <- "mfvb_perturbed"

results_mcmc <- SummarizeMCMCResults(mcmc_sample)
results_mcmc_pert <- SummarizeMCMCResults(mcmc_sample_perturbed)
results_mcmc_pert$method <- "mcmc_perturbed"

results <- rbind(results_vb, results_mcmc)
result_pert <- rbind(results_vb, results_vb_pert, results_mcmc, results_mcmc_pert)

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


mean_pert_results <-
  filter(result_pert, metric == "mean") %>%
  dcast(par + component + group ~ method, value.var="val") %>%
  mutate(mfvb_diff = mfvb_perturbed - mfvb, mcmc_diff = mcmc_perturbed - mcmc)

ggplot(filter(mean_pert_results, par != "mu_g")) +
  geom_point(aes(x=mcmc_diff, y=mfvb_diff, color=par), size=3) +
  geom_abline(aes(slope=1, intercept=0))

ggplot(filter(mean_pert_results, par == "mu_g")) +
  geom_point(aes(x=mcmc_diff, y=mfvb_diff, color=par), size=3) +
  geom_abline(aes(slope=1, intercept=0))


