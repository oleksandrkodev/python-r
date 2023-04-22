import numpy as np
import pandas as pd
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects.packages as rpackages
import scipy.stats
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector

# Load the numpy2ri module
numpy2ri.activate()


# import R's "base" package
base = importr("base")

# import R's "utils" package
utils = importr("utils")

# R package names
packnames = ("ggplot2", "hexbin")

# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

# import R's cfcausal package
cfcausal = importr("cfcausal")


# np.random.seed(2000)


def f(x):
    return 2 / (1 + np.exp(-12 * (x - 0.5)))


def propensity_score(x):
    beta_cdf = scipy.stats.beta.cdf(x, 2, 4)
    return 0.25 * (1 + beta_cdf)


n = 1000
d = 10

# Generate covariates X
X = np.random.uniform(0, 1, (n, d))

# Calculate expected Y(1) given X
E_Y1_given_X = f(X[:, 0]) * f(X[:, 1])

# Generate errors
errors = np.random.normal(0, 1, n)

# Calculate Y(1)
Y1 = E_Y1_given_X + errors

# Calculate propensity scores
prop_scores = propensity_score(X[:, 0])

# Generate treatment assignment T
T = np.random.binomial(1, prop_scores)

# Calculate observed outcomes Y
Y = T * Y1

# Create the synthetic dataset
synthetic_dataset = np.column_stack((X, T, Y))
column_names = [f"X{i}" for i in range(1, d + 1)] + ["T", "Y"]
df = pd.DataFrame(synthetic_dataset, columns=column_names)
# df.to_csv("asset`s/1.csv", index=False)
# print(df.head())


quantiles = np.array([0.05, 0.95])

nr, nc = X.shape
X_r = robjects.r.matrix(X, nrow=nr, ncol=nc)
print(type(Y))

# nr,nc = Y.shape
Y_r = robjects.r.matrix(Y, nrow=nr, ncol=nc)

# nr,nc = T.shape
T_r = robjects.r.matrix(T, nrow=nr, ncol=nc)

# nr,nc = T.shape
quantiles_r = robjects.r.matrix(quantiles, nrow=nr, ncol=nc)

Y_r = robjects.IntVector(Y)
T_r = robjects.IntVector(T)

robjects.r.assign("X", X_r)
robjects.r.assign("Y", Y_r)
robjects.r.assign("T", T_r)
robjects.r.assign("quantiles", quantiles_r)


x = cfcausal.conformalIte(
    robjects.r["Y"],
    robjects.r["Y"],
    robjects.r["T"],
    alpha=0.1,
    algo="nest",
    exact=False,
    type="CQR",
    quantiles=robjects.r["quantiles"],
    outfun="quantRF",
    useCV=False,
)

# print(x)

p = x(robjects.r["X"])
print(type(p))
