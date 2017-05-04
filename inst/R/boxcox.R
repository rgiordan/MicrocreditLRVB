library(ggplot2)
library(dplyr)
library(reshape2)
library(Matrix)

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
    #print(qqnorm(y_trans, main=lambda))
    #readline(prompt="Press [enter] to continue")
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

save(data_df_transform, file=file.path(project_directory, "boxcox_data.Rdata"))
