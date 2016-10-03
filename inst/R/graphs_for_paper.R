library(ggplot2)

script_path <- file.path(Sys.getenv("GIT_REPO_LOC"), "VariationalRobustBayesPaper/writing/R_graphs")
source(file.path(script_path, "Initialize.R"), echo=TRUE)
source(file.path(script_path, "MicrocreditLoadData.R"), echo=TRUE)

sens_results <- mc_fun_env$sens_results
pp <- mc_fun_env$pp
pp_perturb <- mc_fun_env$pp_perturb

# The raw local sensitivity
ggplot(sens_results) +
  geom_errorbar(aes(x=diff_mean / pp_perturb$epsilon, y=sens_mean,
                    ymax=sens_mean + 2 * sens_sd,
                    ymin=sens_mean - 2 * sens_sd), color="gray") +
  geom_point(aes(x=diff_mean / pp_perturb$epsilon, y=sens_mean, color=par)) +
  geom_abline((aes(intercept=0, slope=1))) +
  ggtitle("raw sens")

# The "mean value" local sensitivity.  We expect this to be better.
ggplot(sens_results) +
  geom_errorbar(aes(x=diff_mean / pp_perturb$epsilon, y=mv_sens_mean,
                    ymax=mv_sens_mean + 2 * mv_sens_sd,
                    ymin=mv_sens_mean - 2 * mv_sens_sd), color="gray") +
  geom_point(aes(x=diff_mean / pp_perturb$epsilon, y=mv_sens_mean, color=par)) +
  geom_abline((aes(intercept=0, slope=1))) +
  ggtitle("Mean value sens")


mu_post_mean <- filter(mc_par_env$results_pert, par == "mu", method == "mfvb", metric == "mean")$val

ggplot(sample_n(mc_fun_env$component_df, 5e4)) +
  geom_point(aes(x=X1, y=X2, color=influence)) +
  geom_point(aes(x=mu_post_mean[1], y=mu_post_mean[2]), size=2) +
  scale_color_gradient2() +
  ggtitle(paste("Influence function of the mu prior on",
                unique(mc_fun_env$component_df$component_name)))
