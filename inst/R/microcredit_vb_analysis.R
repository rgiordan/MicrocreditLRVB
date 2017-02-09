library(ggplot2)
library(dplyr)
library(reshape2)
library(rstan)
library(Matrix)

library(MicrocreditLRVB)

# Load previously computed Stan results
#analysis_name <- "simulated_data_robust"
#analysis_name <- "simulated_data_nonrobust"
#analysis_name <- "simulated_data_lambda_beta"
analysis_name <- "real_data_informative_priors"

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/simulated_data")

stan_draws_file <-
  file.path(project_directory, paste(analysis_name, "_data_and_mcmc_draws.Rdata", sep=""))
print(paste("Loading draws from ", stan_draws_file))

LoadIntoEnvironment <- function(filename) {
  local_env <- environment()
  load(filename, envir=local_env)
  return(local_env)
}
mcmc_environment <- LoadIntoEnvironment(stan_draws_file)
stan_results <- mcmc_environment$results$epsilon_0.000000
stan_results_perturb <- mcmc_environment$results$epsilon_1.000000

x <- stan_results$dat$x
y <- stan_results$dat$y
y_g <- stan_results$dat$y_group

pp <- mcmc_environment$pp
pp_perturb <- mcmc_environment$pp_perturb

stan_results$mcmc_time

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
print(mcmc_environment$results$epsilon_0.000000$mcmc_time)
lrvb_time <- Sys.time()
lrvb_terms <- GetLRVB(x, y, y_g, vp_opt, pp)
lrvb_time <- Sys.time() - lrvb_time
prior_sens <- GetSensitivity(vp_opt, pp, lrvb_terms$jac, lrvb_terms$elbo_hess)

vb_fit_perturb <- FitVariationalModel(x, y, y_g, vp_opt, pp_perturb)


# "reg" for "regression"
vp_reg <- InitializeVariationalParameters(
  x, y, y_g, mu_diag_min=0.01, lambda_diag_min=1e-5, tau_min=1, lambda_n_min=0.5)
mp_reg <- GetMoments(vp_reg)


#########################################
# Fit transformed model

# TODO: put this in MCMC too if it works here
TransformFunction <- function(y) {
  sign(y) * (abs(y) ^ (1 / 2))
}

d <- data.frame(x=x, y=y, y_g=y_g)

trim_level <- 0.01
group_by(d, y_g, x.2) %>%
  summarize(count=n(), maxy=max(y), miny=min(y), medy=median(y),
            qlow=quantile(y, trim_level), qhigh=quantile(y, 1 - trim_level))

d <-
  group_by(d, y_g, x.2) %>%
  mutate(qhigh=quantile(y, 1 - trim_level),
         qlow=quantile(y, trim_level),
         trim=((y > qhigh) | (y < qlow)),
         y_trans=TransformFunction(y))

d_filter <- filter(d, !trim)
x_filter <- cbind(d_filter$x.1, d_filter$x.2) 
vp_reg <- InitializeVariationalParameters(
  x_filter, d_filter$y_trans, d_filter$y_g,
  mu_diag_min=0.01, lambda_diag_min=1e-5, tau_min=1, lambda_n_min=0.5)
mp_reg <- GetMoments(vp_reg)


# Debug fit
mask <- rep(TRUE, vp_reg$encoded_size)
bfgs_opt_fns <- GetOptimFunctions(x_filter, d_filter$y_trans, d_filter$y_g, vp_reg, pp, DerivFun=GetElboDerivatives, mask=mask)
theta_init <- GetVectorFromParameters(vp_reg, TRUE)
bounds <- GetVectorBounds(vp_reg, loc_bound=30, info_bound=10, tau_bound=100)

bfgs_result <- optim(theta_init[mask],
                     bfgs_opt_fns$OptimVal, bfgs_opt_fns$OptimGrad,
                     method="L-BFGS-B", lower=bounds$theta_lower[mask], upper=bounds$theta_upper[mask],
                     control=list(fnscale=-1, maxit=1000, trace=0, factr=1e10))
stopifnot(bfgs_result$convergence == 0)

vp_bfgs <- GetParametersFromVector(vp_reg, bfgs_result$par, TRUE)
trust_fns <- GetTrustRegionELBO(x_filter, d_filter$y_trans, d_filter$y_g, vp_bfgs, pp, verbose=TRUE)
trust_time <- Sys.time()
trust_result <- trust(trust_fns$TrustFun, trust_fns$theta_init,
                      rinit=10, rmax=100, minimize=FALSE, blather=TRUE, iterlim=50)
trust_time <- Sys.time() - trust_time


# Look at the optimization plots

plot(trust_result$r); points(trust_result$stepnorm, col="red")
plot(trust_result$r - trust_result$stepnorm)
plot(trust_result$rho, ylim=c(-1,1))

tdf <- with(trust_result, data.frame(r=r, rho=rho, stepnorm=stepnorm, valpath=valpath))
tdf$rho <- ifelse(abs(tdf$rho) > 1e2, NaN, tdf$rho)
tdf$step <- 1:nrow(tdf)
ggplot(melt(select(tdf, step, rho, r), id.vars="step"), aes(x=step, y=value)) +
  geom_line() + geom_point() + facet_grid(variable ~ ., scales="free")


GetMomentDataframeFromParamVector <- function(vp_vec) {
  vp_mom <- GetMoments(GetParametersFromVector(vp_reg, vp_vec, TRUE))
  SummarizeRawMomentParameters(vp_mom, metric = "mean", method = "optim")
}

trust_moments_list <- list()
for (step in 1:nrow(trust_result$argtry)) {
  trust_moments_list[[step]] <- GetMomentDataframeFromParamVector(trust_result$argtry[step,])
  trust_moments_list[[step]]$step <- step
}

trust_moments <- do.call(rbind, trust_moments_list)

bad_steps <- which(!is.finite(tdf$rho))
trust_moments <- filter(trust_moments, !(step %in% bad_steps))

trust_moments <- filter(trust_moments, step > 0)

trust_moments <-
  group_by(trust_moments, par, component, group) %>%
  mutate(val_norm=(val - min(val)) / (max(val) - min(val)))


ggplot(filter(trust_moments), aes(x=step, y=val_norm)) +
  geom_line() + geom_point() +
  facet_grid(par + component ~ group)

ggplot(filter(trust_moments), aes(x=step, y=val)) +
  geom_line() + geom_point() +
  facet_grid(par + component ~ group, scale="free")


ggplot(filter(trust_moments, par == "mu" || par == "mu_g"), aes(x=step, y=val_norm)) +
  geom_line() + geom_point() +
  facet_grid(par + component + group ~ .)


GetMomentDataframeFromParamVector(trust_result$argtry[50,])

step_m <-
  filter(GetMomentDataframeFromParamVector(trust_result$argtry[50,]), par == "lambda") %>%
  select(component, val) %>%
  arrange(component)
lambda <- matrix(step_m$val, 2, 2)
det(lambda)


ggplot(
  filter(trust_moments, par == "mu_g") %>%
    mutate(component_name=paste("x", component, sep="")) %>%
    dcast(par + group + step ~ component_name, value.var="val")
) +
  geom_point(aes(x=x1, y=x2, color=ordered(step)))


tr_ret <- trust_fns$TrustFun(trust_result$argtry[step,]) 


##########################################
# Get MCMC sensitivity measures

mcmc_sample <- extract(stan_results$sim)
mcmc_sample_perturbed <- extract(stan_results_perturb$sim)
mp_draws <- PackMCMCSamplesIntoMoments(mcmc_sample, mp_reg) # A little slow

draws_mat <- do.call(rbind, lapply(mp_draws, GetVectorFromMoments))
log_prior_grad_list <- GetMCMCLogPriorDerivatives(mp_draws, pp)
log_prior_grad_mat <- do.call(rbind, log_prior_grad_list)

# Save some of these objects in the mcmc_environment for use later.
mcmc_environment$mp_draws <- mp_draws
mcmc_environment$draws_mat <- draws_mat
mcmc_environment$log_prior_grad_mat <- log_prior_grad_mat
mcmc_environment$mcmc_sample_perturbed <- mcmc_sample_perturbed
mcmc_environment$mcmc_sample <- mcmc_sample


###########################################
# Save fits

fit_file <- file.path(project_directory, paste(analysis_name, "_mcmc_and_vb.Rdata", sep=""))
print(paste("Saving fits to ", fit_file))
save(mcmc_environment, vb_fit, vb_fit_perturb, lrvb_terms, prior_sens, lrvb_time, file=fit_file)
