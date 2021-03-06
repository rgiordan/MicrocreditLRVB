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

data_df <- data.frame(x=x, y=y, y_g=y_g)

# Group 3 has no negative profit?
group <- 1
ggplot(filter(data_df, y_g==group, abs(y) > 1e-8)) +
  geom_histogram(aes(x=log(abs(y)), y=..density..), bins=100) +
  facet_grid(y > 0 ~ .) + ggtitle(group)

group <- 2
ggplot(filter(data_df, y_g==group, abs(y) > 1e-8, abs(y) < 500)) +
  geom_histogram(aes(x=abs(y), y=..density..), bins=100) +
  facet_grid(y > 0 ~ .) + ggtitle(group)

#####################
# A good reference for the Box-Cox transform:
# https://www.ime.usp.br/~abe/lista/pdfm9cJKUmFZp.pdf
library(MASS)

data_df_transform <-
  data_df %>%
  mutate(zero_y=abs(y) < 1e-8)

# Non-zero values of y_trans will be sent in the loop below.
data_df_transform$y_trans <- 0.0
data_df_transform$lambda <- NaN

for (group in 1:max(y_g)) { for (y_sign in c(-1, 1)) {
  rows <- with(data_df_transform, (y_g == group) & (!zero_y) & (y * y_sign > 0))
  bc_y <- y_sign * data_df_transform[rows, ]$y
  if (length(bc_y) > 0) {
    # The MASS boxcox function is pretty primitive.  Better to do it yourself with optim.
    bc <- boxcox(bc_y ~ 1, plotit=FALSE, lambda=seq(-1, 1, 0.001))
    lambda <- bc$x[which.max(bc$y)]
    if (abs(lambda) < 0.001) {
      lambda <- 0
    }
    if (lambda == 0) {
      y_trans <- log(bc_y)
    } else {
      y_trans <- ((bc_y ^ lambda) - 1) / lambda
    }
    print(qqnorm(y_trans, main=lambda))
    readline(prompt="Press [enter] to continue")
    data_df_transform[rows, "y_trans"] <- y_sign * y_trans
    data_df_transform[rows, "lambda"] <- lambda
  }
}}


ggplot(filter(data_df_transform, !zero_y)) +
  geom_histogram(aes(x=y_trans, y=..density..), bins=100) +
  facet_grid(y_g ~ .)

mutate(data_df_transform, y_pos=y > 0) %>%
  filter(!zero_y) %>%
  group_by(y_g, y_pos) %>%
  summarize(lambda=unique(lambda))

save(data-)




#####################

pp <- mcmc_environment$pp
pp_perturb <- mcmc_environment$pp_perturblibrary(ggplot2)
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


d <- data.frame(x=x, y=y, y_g=y_g)

trim_level <- 0.01
group_by(d, y_g, x.2) %>%
  summarize(count=n(), maxy=max(y), miny=min(y), medy=median(y),
            qlow=quantile(y, trim_level), qhigh=quantile(y, 1 - trim_level))

d_trim <-
  group_by(d, y_g, x.2) %>%
  mutate(qhigh=quantile(y, 1 - trim_level),
         qlow=quantile(y, trim_level),
         trim=((y > qhigh) | (y < qlow)))  

ggplot(filter(d_trim, !trim, y_g == 1)) +
  geom_histogram(aes(x=y), bins=100) +
  facet_grid(x.2 ~ .)


TrasformFunction <- function(y) {
  sign(y) * (abs(y) ^ (1 / 2))
}

# View(d_trim)
foo <- filter(d_trim, y_g == 1, x.2 == 0, !trim)
foo$y_trans <- with(foo, TrasformFunction(y))
qqnorm(foo$y_trans)

foo <- filter(d_trim, y_g == 1, x.2 == 0, abs(y) > 1e-3, !trim)
foo$y_trans <- with(foo, TrasformFunction(y))
qqnorm(foo$y_trans)









#######################################
# Debug original data

pp[["mu_t_loc"]] <- 0
pp[["mu_t_df"]] <- 1
pp[["mu_t_scale"]] <- 100

pp$monte_carlo_prior <- TRUE
pp$epsilon <- 0

mask <- rep(TRUE, vp_reg$encoded_size)
bfgs_opt_fns <- GetOptimFunctions(x, y, y_g, vp_reg, pp, DerivFun=GetElboDerivatives, mask=mask)
theta_init <- GetVectorFromParameters(vp_reg, TRUE)
bounds <- GetVectorBounds(vp_reg, loc_bound=100, info_bound=100, tau_bound=100)

bfgs_result <- optim(theta_init[mask],
                     bfgs_opt_fns$OptimVal, bfgs_opt_fns$OptimGrad,
                     method="L-BFGS-B", lower=bounds$theta_lower[mask], upper=bounds$theta_upper[mask],
                     control=list(fnscale=-1, maxit=1000, trace=0, factr=1e10))
stopifnot(bfgs_result$convergence == 0)

vp_bfgs <- GetParametersFromVector(vp_reg, bfgs_result$par, TRUE)

trust_fns <- GetTrustRegionELBO(x, y, y_g, vp_bfgs, pp, verbose=TRUE)

tr_ret <- trust_fns$TrustFun(bfgs_result$par) 
ev <- eigen(tr_ret$hessian)$values
max(ev)
min(ev)

trust_time <- Sys.time()
trust_result <- trust(trust_fns$TrustFun, trust_fns$theta_init,
                      rinit=1, rmax=100, minimize=FALSE, blather=TRUE, iterlim=50)
trust_time <- Sys.time() - trust_time


vp_opt <- GetParametersFromVector(vp_reg, trust_result$argument, TRUE)
mp_opt <- GetMoments(vp_opt)

cbind(SummarizeRawMomentParameters(mp_opt, metric = "mean", method = "optim"),
      SummarizeRawMomentParameters(mp_opt_t, metric = "mean", method = "optim"))


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


ggplot(d) +
  geom_histogram(aes(x=y_trans), bins=50) +
  facet_grid(y_g ~ x.2, scale="free")


# Debug fit
mask <- rep(TRUE, vp_reg$encoded_size)
bfgs_opt_fns <- GetOptimFunctions(x_filter, d_filter$y_trans, d_filter$y_g, vp_reg, pp, DerivFun=GetElboDerivatives, mask=mask)
theta_init <- GetVectorFromParameters(vp_reg, TRUE)
bounds <- GetVectorBounds(vp_reg, loc_bound=100, info_bound=100, tau_bound=100)

bfgs_result <- optim(theta_init[mask],
                     bfgs_opt_fns$OptimVal, bfgs_opt_fns$OptimGrad,
                     method="L-BFGS-B", lower=bounds$theta_lower[mask], upper=bounds$theta_upper[mask],
                     control=list(fnscale=-1, maxit=1000, trace=0, factr=1e10))
stopifnot(bfgs_result$convergence == 0)

vp_bfgs <- GetParametersFromVector(vp_reg, bfgs_result$par, TRUE)


trust_fns <- GetTrustRegionELBO(x_filter, d_filter$y_trans, d_filter$y_g, vp_bfgs, pp, verbose=TRUE)

tr_ret <- trust_fns$TrustFun(bfgs_result$par) 
ev <- eigen(tr_ret$hessian)$values
max(ev)
min(ev)

trust_time <- Sys.time()
trust_result <- trust(trust_fns$TrustFun, trust_fns$theta_init,
                      rinit=1, rmax=100, minimize=FALSE, blather=TRUE, iterlim=50)
trust_time <- Sys.time() - trust_time


# Look at the optimization plots

plot(trust_result$r); points(trust_result$stepnorm, col="red")
plot(trust_result$r - trust_result$stepnorm)
plot(trust_result$rho, ylim=c(-1,1))

tdf <- with(trust_result, data.frame(r=r, rho=rho, stepnorm=stepnorm, valpath=valpath))
tdf$rho <- ifelse(abs(tdf$rho) > 5, NaN, tdf$rho)
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

step <- 50
tr_ret <- trust_fns$TrustFun(trust_result$argtry[step,]) 
ev <- eigen(tr_ret$hessian)$values
max(ev)
min(ev)




