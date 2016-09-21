GetMuLogPrior <- function(mu, pp) {
  # You can't use the VB priors because they are
  # (1) a function of the natural parameters whose variance would have to be zero and
  # (2) not normalized.
  dmvnorm(mu, mean=pp$mu_loc, sigma=solve(pp$mu_info), log=TRUE)
}


DrawFromQMu <- function(n_draws, vp_opt, rescale=1) {
  mu_info <- vp_opt$mu_info
  if (double_var) {
    mu_info <- mu_info / (rescale ^ 2)
  }
  return(rmvnorm(n_draws, vp_opt$mu_loc, solve(mu_info)))
}


GetMuLogDensity <- function(mu, vp_opt, pp, calculate_gradient) {
  draw_local <- GetMoments(vp_opt)
  draw_local$mu_e_vec <- mu
  include_tau_groups <- include_mu_groups <- as.integer(c())
  q_derivs <- GetLogVariationalDensityDerivatives(
    draw_local, vp_opt, pp, include_mu=TRUE, include_lambda=FALSE,
    include_mu_groups, include_tau_groups, calculate_gradient=calculate_gradient)
  return(q_derivs)
}


GetMuLogStudentTPrior <- function(mu, pp_perturb) {
  log_t_prior <- 0
  for (k in 1:length(mu)) {
    log_t_prior <- log_t_prior + student_t_log(mu[k], pp_perturb$mu_t_df, pp_perturb$mu_t_loc, pp_perturb$mu_t_scale)
  }
  return(log_t_prior)
}

