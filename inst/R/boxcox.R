library(ggplot2)
library(dplyr)
library(reshape2)
library(Matrix)
library(gridExtra)

project_directory <-
  file.path(Sys.getenv("GIT_REPO_LOC"), "MicrocreditLRVB/inst/simulated_data")
raw_data_directory <- file.path(Sys.getenv("GIT_REPO_LOC"), "microcredit_vb/data")

# Read in the data that was produced by the R script.
csv_data <- read.csv(file.path(raw_data_directory, "microcredit_data_processed.csv"))

# The dimension of the group means.  Here, the first dimension
# is the control effect and the second dimension is the treatment.
k <- 2

# The number of distinct groups.
n_g <- max(csv_data$site)

# Get the observations and the total number of observations.
y <- csv_data$profit
y_g <- as.integer(csv_data$site)

# The x array will indicate which rows should also get the
# treatment effect.  The model accomodates generic x, but for
# this application we only need indicators.
x <- cbind(rep(1, length(y)), as.numeric(csv_data$treatment))

data_df <- data.frame(x=x, y=y, y_g=y_g)

data_df_transform <-
  data_df %>%
  mutate(zero_y=abs(y) < 1e-8)

#######################
# Look at the raw distributions with outlier trimming.

trim_level <- 0.3
y_quantiles <-
  filter(data_df_transform, !zero_y) %>%
  group_by(y_g) %>%
  summarize(qlower=quantile(y, trim_level), qupper=quantile(y, 1 - trim_level))

data_df_trim <-
  filter(data_df_transform, !zero_y) %>%
  inner_join(y_quantiles, by="y_g") %>%
  filter(y < qupper & y > qlower) %>%
  group_by(y_g) %>%
  arrange(y) %>%
  mutate(q=(1:length(y)) / length(y)) %>%
  mutate(norm=qnorm(q))

# Qqplots
ggplot(filter(data_df_trim)) +
  geom_point(aes(x=norm, y=y)) +
  facet_grid(y_g ~ ., scales="free")

ggplot(filter(data_df_trim)) +
  geom_histogram(aes(x=y, y=..ndensity..), bins=20) +
  facet_grid(~ y_g, scales="free") + 
  geom_vline(aes(xintercept=0))





###################
# Look at cumulative distributions to check for power law behavior.
# If it is a power law, the cumulative distribution will be a straight line
# with slope given by the power law coefficient plus one.

quantile_list <- list()
for (group in 1:max(data_df_transform$y_g)) { for (y_sign in c(-1, 1)) { for (arm in c(0, 1)) {
  rows <- with(data_df_transform, (y_g == group) & (!zero_y) & (y * y_sign > 0) & (x.2 == arm))
  if (sum(rows) > 0) {
    quantile_df <- data.frame(y=sort(y_sign * data_df_transform[rows, ]$y),
                              quantile=(sum(rows):1) / sum(rows),
                              group=group, y_sign=y_sign, arm=arm)
    quantile_list[[length(quantile_list) + 1]] <- quantile_df
  }
}}}
quantile_df <- do.call(rbind, quantile_list)

grid.arrange(
  ggplot(quantile_df) +
    geom_point(aes(x=y_sign * log10(y), y=log10(quantile), color=factor(arm))) +
    facet_grid(group ~ y_sign) +
    ggtitle("Overlaid arms")
,
  ggplot(quantile_df) +
    geom_point(aes(x=log10(y), y=log10(quantile), color=factor(y_sign))) +
    facet_grid(group ~ arm) +
    ggtitle("Overlaid signs")
,
  ggplot(quantile_df) +
    geom_point(aes(x=log10(y), y=log10(quantile), color=paste(arm, y_sign))) +
    facet_grid(group ~ .) +
    ggtitle("Everything overlaid")
, ncol=3
)



##########################################
# Execute box-cox transforms and save.

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
    #print(qqnorm(y_trans, main=lambda))
    #readline(prompt="Press [enter] to continue")
    data_df_transform[rows, "y_trans"] <- y_sign * y_trans
    data_df_transform[rows, "lambda"] <- lambda
  }
}}


ggplot(filter(data_df_transform, !zero_y)) +
  geom_histogram(aes(x=y_trans, y=..density..), bins=100) +
  facet_grid(y_g ~ .)

# Look at the lambdas chosen.  They are similar mostly except for group 2.
mutate(data_df_transform, y_pos=y > 0) %>%
  filter(!zero_y) %>%
  group_by(y_g, y_pos) %>%
  summarize(lambda=unique(lambda))

save(data_df_transform, file=file.path(project_directory, "boxcox_data.Rdata"))


##########################################
# Look at the treatments transformed by the inverse CDF of the control.  This is not basically
# different than quantile regression and has all the same attendant problems.

# switch the role control and treatment to make sure you're not building in biases.
switcheroo <- TRUE

result_list <- list()
for (group in 1:max(data_df_transform$y_g)) {
  group_df <- filter(data_df_transform, !zero_y, y_g == group) 
  
  if (switcheroo) {
    y_c <- filter(group_df, x.2 == 1)$y
    y_t <- filter(group_df, x.2 == 0)$y
  } else {
    y_c <- filter(group_df, x.2 == 0)$y
    y_t <- filter(group_df, x.2 == 1)$y
  }
  cdf <- data.frame(q=1:length(y_c) / (length(y_c) + 1), y=sort(y_c))
  y_t_q <- approx(x=cdf$y, y=cdf$q, xout=y_t, rule=2)$y

  cdf_t <- data.frame(y=y_t, q=y_t_q, group=group)
  # cdf_t <- filter(cdf_t, is.finite(y_t_q))

  # Look at a uniform binning.
  num_bins <- 10
  
  # Look at a normal transform.
  cdf_t$y_normed <- qnorm(p=cdf_t$q)
  cdf_t$std_normal <- rnorm(nrow(cdf_t)) # For easy graphing

  result_list[[length(result_list) + 1]] <- cdf_t
}

cdf_t <- do.call(rbind, result_list)

# y_normed_range <- with(cdf_t, seq(min(y_normed) - 0.01, max(y_normed) + 0.01, length.out=length(y_normed)))

# ggplot(cdf_t) +
#   geom_density(aes(x=y_normed, color="treatment"), lwd=2) +
#   geom_density(aes(x=std_normal, color="null"), lwd=1) +
#   facet_grid(~ group)

# Why does each direction shrink the first and last quantiles?
num_bins <- 10
ggplot(cdf_t) +
  geom_histogram(aes(x=q, y=..density..), bins=num_bins, fill="gray", color="black") +
  geom_hline(aes(yintercept=1)) +
  facet_grid(~ group)


#######################
# Look at an alternative parameterization of the box-cox transform.

n_sim <- 100
std_normal <- qnorm(seq(1 / n_sim, 1 - 1 / n_sim, length.out=n_sim))

GetBoxCoxQuantiles <- function(lambda, norm_mu, norm_sd) {
  bc_draws <- (lambda * (std_normal * norm_sd + norm_mu) + 1) ^ (1 / lambda)
  bc_draws <- bc_draws[is.finite(bc_draws)]
  return(quantile(bc_draws, c(0.05, 0.5, 0.95)))
}

norm_mus <- seq(-1, 1, length.out=20)
lambdas <- seq(0.1, 0.4, length.out=20)

norm_sd <- 1
results_list <- list()
for (norm_mu in norm_mus) { for (lambda in lambdas) {
  bcq <- GetBoxCoxQuantiles(lambda, norm_mu, norm_sd)
  results_list[[length(results_list) + 1]] <-
    data.frame(norm_sd=norm_sd, norm_mu=norm_mu, lambda=lambda, bcq1=bcq[1], bcq2=bcq[2], bcq3=bcq[3])
}}
bc_param_df <- do.call(rbind, results_list)
head(bc_param_df)

grid.arrange(
  ggplot(bc_param_df) +
    geom_tile(aes(x=norm_mu, y=lambda, fill=bcq1))
,
  ggplot(bc_param_df) +
    geom_tile(aes(x=norm_mu, y=lambda, fill=bcq2))
,
  ggplot(bc_param_df) +
    geom_tile(aes(x=norm_mu, y=lambda, fill=bcq3))
, ncol=3
)


grid.arrange(
  ggplot(bc_param_df) +
    geom_point(aes(x=bcq1, y=bcq2, color=norm_mu), size=2)
  ,
  ggplot(bc_param_df) +
    geom_point(aes(x=bcq1, y=bcq3, color=norm_mu), size=2)
  ,
  ggplot(bc_param_df) +
    geom_point(aes(x=bcq2, y=bcq3, color=norm_mu), size=2)
  , ncol=3
)

grid.arrange(
  ggplot(bc_param_df) +
    geom_point(aes(x=bcq1, y=bcq2, color=lambda), size=2)
  ,
  ggplot(bc_param_df) +
    geom_point(aes(x=bcq1, y=bcq3, color=lambda), size=2)
  ,
  ggplot(bc_param_df) +
    geom_point(aes(x=bcq2, y=bcq3, color=lambda), size=2)
  , ncol=3
)

