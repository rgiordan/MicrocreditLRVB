# This is just screwing around.

###############################
# Optimization visualization

# Each row is a step
tr_steps <- trust_result$argpath

step_dist <- dist(tr_steps)
step_mds <- cmdscale(step_dist, eig=TRUE, k=2)
mds_tr <- data.frame(step_mds$points)
mds_tr$val <- trust_result$valpath


# Randomly choose points in the convex hull of the optimization path.
n_draws <- 1e3
theta_draws <- matrix(NaN, n_draws, ncol(tr_steps))
val_draws <- rep(NaN, n_draws)
for (ind in 1:n_draws) {
  tr_steps_t <- t(tr_steps)
  w <- rgamma(nrow(tr_steps), 1, 1)
  w <- w / sum(w)
  theta_draw <- tr_steps_t %*% w
  val <- bfgs_opt_fns$OptimVal(theta_draw)
  theta_draws[ind, ] <- theta_draw
  val_draws[ind] <- val
}

draw_dist <- dist(theta_draws)
draw_mds <- cmdscale(draw_dist, eig=TRUE, k=2)
mds <- data.frame(draw_mds$points)
mds$val <- val_draws

GridRawData <- function(raw_x, n_grid) {
  breaks <- seq(min(raw_x) - 1e-6, max(raw_x) + 1e-6, length.out=n_grid)
  breaks <- quantile(raw_x, c(0, (1:n_grid) / n_grid))
  intervals_x <- findInterval(raw_x, breaks)
  intervals_x[intervals_x == n_grid + 1] <- n_grid # Handle the max gracefully
  grid_x <- breaks[intervals_x]
  return(grid_x)  
}

n_grid <- 6
mds$X1_grid <- GridRawData(mds$X1, n_grid)
mds$X2_grid <- GridRawData(mds$X2, n_grid)

ggplot() +
  geom_raster(aes(x=X1_grid, y=X2_grid, fill=val), data=mds) +
  geom_line(aes(x=X1, y=X2), color="red", lwd=2, data=mds_tr)



# Try this? 
# https://www.r-bloggers.com/barycentric-interpolation-fast-interpolation-on-arbitrary-grids/
n <- 100
foo <- cbind(rnorm(n), rnorm(n))

#2D barycentric interpolation at points Xi for a function with values
# f measured at locations X
#For N-D interpolation simply replace tsearch with tsearchn and modify the
# sparse matrix definition to have non-zero values in the right spots.
interp.barycentric <- function(X, f, Xi)
{
  require(geometry)
  require(Matrix)
  dn <- delaunayn(X)
  tri <- tsearch(X[,1], X[,2], dn, Xi[,1], Xi[,2], bary=T)
  #For each line in Xi, defines which points in X contribute to the interpolation
  active <- dn[tri$idx,]
  #Define the interpolation as a sparse matrix operation. Faster than using apply,
  # probably slower than a C implementation
  M <- sparseMatrix(i=rep(1:nrow(Xi),each=3),
                    j=as.numeric(t(active)),
                    x=as.numeric(t(tri$p)),
                    dims=c(nrow(Xi), length(f)))
  as.numeric(M %*% f)
}


