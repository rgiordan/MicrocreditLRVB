
# Load your data and then...

library(lme4)

glm_data <- data.frame(y=y, y_g=y_g)
glm_data <- cbind(glm_data, data.frame(x))
glm_res <- lmer(y ~ (X1 + 0 | y_g) + (X2 + 0 | y_g) + X1 + X2 + 0, data=glm_data)

glm_res
filter(nat_result, par == "mu")

vb_mu_g <-
  filter(nat_result, par == "mu_g") %>%
  dcast(group ~ component, value.var="val")

if (FALSE) {
  plot(as.matrix(true_mu_g[,1]), as.matrix(ranef(glm_res)$y_g)[,1]); abline(0, 1)
  plot(as.matrix(ranef(glm_res)$y_g)[,1], vb_mu_g[["1"]]); abline(0, 1)
  plot(as.matrix(ranef(glm_res)$y_g)[,2], vb_mu_g[["2"]]); abline(0, 1)
  plot(as.matrix(true_mu_g[,1]), vb_mu_g[["1"]]); abline(0, 1)
  plot(as.matrix(true_mu_g[,2]), vb_mu_g[["2"]]); abline(0, 1)
}

opt_fns$OptimGrad(optim_result$theta)

grad <- opt_fns$OptimGrad(optim_result$theta)
hess <- opt_fns$OptimHess(optim_result$theta)
hess_eig <- eigen(hess)
min(hess_eig$values)
max(hess_eig$values)

newton_step <- solve(hess, grad)
diff <- opt_fns$OptimVal(optim_result$theta - newton_step) - opt_fns$OptimVal(optim_result$theta)


