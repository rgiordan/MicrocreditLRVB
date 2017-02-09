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
