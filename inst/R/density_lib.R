GetMuLogPrior <- function(mu) {
  # You can't use the VB priors because they are
  # (1) a function of the natural parameters whose variance would have to be zero and
  # (2) not normalized.
  dmvnorm(mu, mean=pp$mu_loc, sigma=solve(pp$mu_info), log=TRUE)
}


DrawFromQMu <- function(n_draws, vp_opt, double_var=FALSE) {
  mu_info <- vp_opt$mu_info
  if (double_var) {
    mu_info <- 0.5 * mu_info
  }
  return(rmvnorm(n_draws, vp_opt$mu_loc, solve(mu_info)))
}


GetMuLogDensity <- function(mu, calculate_gradient) {
  draw_local <- draw  
  draw_local$mu_e_vec <- mu
  include_tau_groups <- include_mu_groups <- as.integer(c())
  q_derivs <- GetLogVariationalDensityDerivatives(
    draw_local, vp_opt, pp, include_mu=TRUE, include_lambda=FALSE,
    include_mu_groups, include_tau_groups, calculate_gradient=calculate_gradient)
  return(q_derivs)
}


GetMuLogStudentTPrior <- function(mu) {
  log_t_prior <- 0
  for (k in 1:length(mu)) {
    log_t_prior <- log_t_prior + student_t_log(mu[k], pp_perturb$mu_t_df, pp_perturb$mu_t_loc, pp_perturb$mu_t_scale)
  }
  return(log_t_prior)
}

