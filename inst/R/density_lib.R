GetMuLogPrior <- function(mu, pp) {
  # You can't use the VB priors because they are
  # (1) a function of the natural parameters whose variance would have to be zero and
  # (2) not normalized.
  dmvnorm(mu, mean=pp$mu_loc, sigma=solve(pp$mu_info), log=TRUE)
}


DrawFromQMu <- function(n_draws, vp_opt, rescale=1) {
  mu_info <- vp_opt$mu_info
  mu_info <- mu_info / (rescale ^ 2)
  return(rmvnorm(n_draws, vp_opt$mu_loc, solve(mu_info)))
}


GetMuLogDensity <- function(mu, vp_opt, draw, pp, unconstrained, calculate_gradient) {
  draw$mu_e_vec <- mu
  draw$mu_e_outer <- mu %*% t(mu)
  q_derivs <- GetLogVariationalDensityDerivatives(
    draw, vp_opt, include_mu=TRUE, include_lambda=FALSE,
    integer(), integer(), unconstrained=unconstrained, global_only=TRUE,
    calculate_gradient=calculate_gradient)
  return(q_derivs)
}


GetMuLogMarginalDensity <- function(mu, mu_comp, vp_opt, draw, unconstrained) {
  draw$mu_e_vec[mu_comp] <- mu
  draw$mu_e_outer[mu_comp, mu_comp] <- mu %*% t(mu)
  
  # The component is 0-indexed in C++
  if (mu_comp <= 0) {
    stop("GetMuLogMarginalDensity takes a one-indexed component.")
  }
  q_derivs <- GetVariationalLogMarginalMuDensityDerivatives(
    draw, vp_opt, component=mu_comp - 1, unconstrained=unconstrained)
  return(q_derivs)
}

GetMuLogStudentTPrior <- function(mu, pp_perturb) {
  log_t_prior <- 0
  for (k in 1:length(mu)) {
    log_t_prior <- log_t_prior + student_t_log(mu[k], pp_perturb$mu_t_df, pp_perturb$mu_t_loc, pp_perturb$mu_t_scale)
  }
  return(log_t_prior)
}



# A dataframe summarizing the VB prior sensitivity
SummarizePriorSensitivityMatrix <- function(prior_sens, pp_indices, mp_opt, method) {
  AppendVBSensitivityResults <- function(ind, prior_param, method) {
    this_mp <- GetMomentsFromVector(mp_opt, prior_sens[, ind])
    return(SummarizeRawMomentParameters(this_mp, metric=prior_param, method=method))
  }
  
  k_reg <- pp_indices$k_reg
  
  results_list <- list()
  for (k in 1:k_reg) {
    results_list[[length(results_list) + 1]] <- AppendVBSensitivityResults(
      pp_indices$mu_loc[k], prior_param=paste("mu_loc", k, sep="_"), method=method)
  }
  for (k1 in 1:k_reg) { for (k2 in 1:k1) {
    results_list[[length(results_list) + 1]] <- AppendVBSensitivityResults(
      pp_indices$mu_info[k1, k2], prior_param=paste("mu_info", k1, k2, sep="_"), method=method)
  }}
  results_list[[length(results_list) + 1]] <- AppendVBSensitivityResults(
    pp_indices$mu_t_loc, prior_param="mu_t_loc", method=method)
  results_list[[length(results_list) + 1]] <- AppendVBSensitivityResults(
    pp_indices$mu_t_scale, prior_param="mu_t_scale", method=method)
  results_list[[length(results_list) + 1]] <- AppendVBSensitivityResults(
    pp_indices$mu_t_df, prior_param="mu_t_df", method=method)
  results_list[[length(results_list) + 1]] <- AppendVBSensitivityResults(
    pp_indices$lambda_eta, prior_param="lambda_eta", method=method)
  results_list[[length(results_list) + 1]] <- AppendVBSensitivityResults(
    pp_indices$lambda_alpha, prior_param="lambda_alpha", method=method)
  results_list[[length(results_list) + 1]] <- AppendVBSensitivityResults(
    pp_indices$lambda_beta, prior_param="lambda_beta", method=method)
  results_list[[length(results_list) + 1]] <- AppendVBSensitivityResults(
    pp_indices$tau_alpha, prior_param="tau_alpha", method=method)
  results_list[[length(results_list) + 1]] <- AppendVBSensitivityResults(
    pp_indices$tau_beta, prior_param="tau_beta", method=method)
  return(do.call(rbind, results_list))
}



######################################
# Influence function helpers

# # Get everything necessary for the calculation of influence functions from a draw, u.
# GetInfluenceFunctionSampleFunction <- function(
#   GetLogVariationalDensity,
#   GetLogPrior,
#   GetULogDensity,
#   lrvb_pre_factor) {
#   
#   function(u) {
#     log_q_derivs <- GetLogVariationalDensity(u)
#     log_prior_val <- GetLogPrior(u)
#     log_u_density <- GetULogDensity(u)
#     
#     # It doesn't seem to matter if I just use the inverse.
#     # lrvb_term <- -1 * lrvb_terms$jac %*% solve(lrvb_terms$elbo_hess, mu_q_derivs$grad)
#     lrvb_term <- lrvb_pre_factor %*% log_q_derivs$grad
#     influence_function <- exp(log_q_derivs$val - log_prior_val) * lrvb_term
#     
#     return(list(u=u,
#                 lrvb_term=lrvb_term,
#                 log_q_val=log_q_derivs$val,
#                 log_prior_val=log_prior_val,
#                 log_u_density=log_u_density,
#                 influence_function=influence_function))
#   }
# }
# 
# 
# # Use the output of a draw from GetInfluenceFunctionSampleFunction to get sensitivity
# # to a particular contaminating function.
# GetSensitivitySampleFunction <- function(GetLogContaminatingPrior) {
#   function(u_draw) {
#     log_contaminating_prior_val <- GetLogContaminatingPrior(u_draw$u)
#     
#     # The vector of sensitivities.
#     log_weight <- log_contaminating_prior_val - u_draw$log_u_density
#     log_influence_factor <- u_draw$log_q_val - u_draw$log_prior_val
#     sensitivity_draw <- exp(log_weight + log_influence_factor) * u_draw$lrvb_term
#     
#     # The "mean value theorem" sensitivity
#     prior_val <- exp(u_draw$log_prior_val)
#     contaminating_prior_val <- exp(log_contaminating_prior_val)
#     prior_ratio <- exp(u_draw$log_prior_val - log_contaminating_prior_val)
#     
#     mv_sensitivity_draw <-
#       exp(log_influence_factor + u_draw$log_prior_val + log_contaminating_prior_val - u_draw$log_u_density) *
#       (log_contaminating_prior_val - u_draw$log_prior_val) *
#       u_draw$lrvb_term / (contaminating_prior_val - prior_val)
#     
#     return(list(sensitivity_draw=sensitivity_draw,
#                 mv_sensitivity_draw=mv_sensitivity_draw,
#                 log_weight=log_weight,
#                 log_influence_factor=log_influence_factor))
#   }
# }
# 
# 
# # Unpack the output of GetSensitivitySampleFunction
# UnpackSensitivityList <- function(sensitivity_list) {
#   sens_vec_list <- lapply(sensitivity_list, function(entry) { as.numeric(entry$sensitivity_draw) } )
#   sens_vec_list_squared <- lapply(sensitivity_list, function(entry) { as.numeric(entry$sensitivity_draw) ^ 2 } )
#   sens_vec_mean <- Reduce(`+`, sens_vec_list) / n_samples
#   sens_vec_mean_square <- Reduce(`+`, sens_vec_list_squared) / n_samples
#   sens_vec_sd <- sqrt(sens_vec_mean_square - sens_vec_mean^2) / sqrt(n_samples)
#   
#   mv_sens_vec_list <- lapply(sensitivity_list, function(entry) { as.numeric(entry$mv_sensitivity_draw) } )
#   mv_sens_vec_list_squared <- lapply(sensitivity_list, function(entry) { as.numeric(entry$mv_sensitivity_draw) ^ 2 } )
#   mv_sens_vec_mean <- Reduce(`+`, mv_sens_vec_list) / n_samples
#   mv_sens_vec_mean_square <- Reduce(`+`, mv_sens_vec_list_squared) / n_samples
#   mv_sens_vec_sd <- sqrt(mv_sens_vec_mean_square - mv_sens_vec_mean^2) / sqrt(n_samples)
#   
#   return(list(sens_vec_mean=sens_vec_mean, sens_vec_sd=sens_vec_sd,
#               mv_sens_vec_mean=mv_sens_vec_mean, mv_sens_vec_sd=mv_sens_vec_sd))
# }
# 
# 
# 
# # Multiple plot function
# #
# # ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# # - cols:   Number of columns in layout
# # - layout: A matrix specifying the layout. If present, 'cols' is ignored.
# #
# # If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# # then plot 1 will go in the upper left, 2 will go in the upper right, and
# # 3 will go all the way across the bottom.
# #
# multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
#   library(grid)
#   
#   # Make a list from the ... arguments and plotlist
#   plots <- c(list(...), plotlist)
#   
#   numPlots = length(plots)
#   
#   # If layout is NULL, then use 'cols' to determine layout
#   if (is.null(layout)) {
#     # Make the panel
#     # ncol: Number of columns of plots
#     # nrow: Number of rows needed, calculated from # of cols
#     layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
#                      ncol = cols, nrow = ceiling(numPlots/cols))
#   }
#   
#   if (numPlots==1) {
#     print(plots[[1]])
#     
#   } else {
#     # Set up the page
#     grid.newpage()
#     pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
#     
#     # Make each plot, in the correct location
#     for (i in 1:numPlots) {
#       # Get the i,j matrix positions of the regions that contain this subplot
#       matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
# 
#       print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
#                                       layout.pos.col = matchidx$col))
#     }
#   }
# }
