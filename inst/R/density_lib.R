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
  q_derivs <- GetLogVariationalDensityDerivatives(
    draw, vp_opt, pp, include_mu=TRUE, include_lambda=FALSE,
    integer(), integer(), unconstrained=unconstrained,
    calculate_gradient=calculate_gradient)
  return(q_derivs)
}


GetMuLogStudentTPrior <- function(mu, pp_perturb) {
  log_t_prior <- 0
  for (k in 1:length(mu)) {
    log_t_prior <- log_t_prior + student_t_log(mu[k], pp_perturb$mu_t_df, pp_perturb$mu_t_loc, pp_perturb$mu_t_scale)
  }
  return(log_t_prior)
}



# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }

  if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}
