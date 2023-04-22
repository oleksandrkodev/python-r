library("cfcausal")
set.seed(1)
n <- 1000
d <- 5

X <- matrix(rnorm(n * d), nrow = n)

beta <- rep(1, 5)

Y1 <- X %*% beta + rnorm(n)

Y0 <- rnorm(n)
ps <- pnorm(X[, 1])
T <- as.numeric(ps < runif(n))

Y <- ifelse(T == 1, Y1, Y0)

ntest <- 5
Xtest <- matrix(rnorm(ntest * d), nrow = ntest)
X
Xtest
class(Xtest)

"x"
class(X)
X[1,2]
"y"
class(Y)
Y[1]
"t"
class(T)

CIfun <- conformalIte(X, Y, T, alpha = 0.1,
                      algo = "nest", exact = FALSE, type = "CQR",
                      quantiles = c(0.05, 0.95),
                      outfun = "quantRF", useCV = FALSE)
CIfun(Xtest)
