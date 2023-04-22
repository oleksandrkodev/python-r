import numpy as np
import pandas as pd
import scipy.stats

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
print(df.head())
