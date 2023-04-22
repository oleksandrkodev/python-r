# Python - R (cfcausal)

I'd like your assistance in writing some code such that I can run an r library (https://lihualei71.github.io/cfcausal/index.html) called cfcausal on a pandas dataframe.

I have some data that I've generated in a pandas dataframe. Below is the code needed to create the dataframe.

https://pastebin.com/niNXRWee

```py
import numpy as np
import scipy.stats
import pandas as pd

#np.random.seed(2000)

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
column_names = [f'X{i}' for i in range(1, d+1)] + ['T', 'Y']
df = pd.DataFrame(synthetic_dataset, columns=column_names)
df.head()
```

I'd like to run some commands from the cfcausal library in R. the documentation is here
https://lihualei71.github.io/cfcausal/index.html

In particular, I would like to run the following methods from that page on our dataframe. And I'd like to store the following outputs from running these r methods on our data

```R
# Inexact nested method
CIfun <- conformalIte(X, Y, T, alpha = 0.1,
    algo = "nest", exact = FALSE, type = "CQR",
    quantiles =  c(0.05, 0.95),
    outfun = "quantRF", useCV = FALSE)

CIfun(Xtest)

# Exact nested method
CIfun <- conformalIte(X, Y, T, alpha = 0.1,
    algo = "nest", exact = TRUE, type = "CQR",
    quantiles = c(0.05, 0.95),
    outfun = "quantRF", useCV = FALSE)
CIfun(Xtest)

# Native method
CIfun <- conformalIte(X, Y, T, alpha = 0.1,
    algo = "naive", type = "CQR",
    quantiles = c(0.05, 0.95),
    outfun = "quantRF", useCV = FALSE)
CIfun(Xtest)
```

inexact_nested_bounds <- CIfun(Xtest)
The conditional_coverage and nested_width values should be returned in a python object. Does that make sense?

# Check if the true values of ξ (xi) fall within the prediction intervals
covered <- Ytest >= inexact_nested_bounds$lower & Ytest <= inexact_nested_bounds$upper

# Calculate the conditional coverage
conditional_coverage = mean(covered)
inexact_nested_width <- mean(inexact_nested_bounds$upper - inexact_nested_bounds$lower)

The conditional_coverage and nested_width values should be returned in a python object. Does that make sense?


Does the task make sense? So we'd like to run the inexact nested, exact nested, and naive methods as outlined on the page+screenshot I sent above. and then for each we'd like to output the conditional_coverage and nested_width values. I described how to do this for the inexact method above and the way to do it for the exact and naive methods is the exact same.
