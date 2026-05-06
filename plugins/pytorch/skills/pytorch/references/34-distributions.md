# PyTorch Probability Distributions - Comprehensive Reference

This chapter covers the `torch.distributions` module, which provides composable probability distributions and composable tensor operations for probabilistic modeling. It includes all distribution classes, transforms, constraints, and KL divergence utilities.

---

## 1. Distribution Base Class

All distributions inherit from `torch.distributions.Distribution` and share a common interface.

### Properties and Methods

```python
class torch.distributions.Distribution(
    batch_shape=torch.Size(),    # shape of batch dimensions
    event_shape=torch.Size(),    # shape of event dimensions
    validate_args=None,          # whether to validate parameters
)
```

### Core Methods

```python
import torch
from torch.distributions import Distribution

# Key properties
dist.batch_shape    # Shape of batch dimensions
dist.event_shape    # Shape of event dimensions

# Sampling
sample = dist.sample(sample_shape=torch.Size())      # Non-differentiable sample
rsample = dist.rsample(sample_shape=torch.Size())     # Differentiable (reparameterized) sample

# Probability / Log-probability
log_prob = dist.log_prob(value)    # Log probability of value under distribution
prob = dist.log_prob(value).exp()  # Probability (computed via log_prob)

# Entropy
entropy = dist.entropy()          # Shannon entropy

# Moments
mean = dist.mean                  # Mean of the distribution
variance = dist.variance          # Variance of the distribution
stddev = dist.stddev              # Standard deviation

# Cumulative distribution function
cdf = dist.cdf(value)            # CDF at value (not available for all distributions)

# Other
support = dist.support            # The support of the distribution
```

### Shape Semantics

```python
import torch
from torch.distributions import Normal

# batch_shape vs event_shape
dist = Normal(torch.zeros(3, 5), torch.ones(3, 5))
print(dist.batch_shape)  # torch.Size([3, 5])
print(dist.event_shape)  # torch.Size([])

# sample_shape is prepended to the result
samples = dist.sample((10,))  # torch.Size([10, 3, 5])
print(samples.shape)

# Multivariate example
from torch.distributions import MultivariateNormal
dist = MultivariateNormal(torch.zeros(3), torch.eye(3))
print(dist.batch_shape)  # torch.Size([])
print(dist.event_shape)  # torch.Size([3])
```

---

## 2. Continuous Distributions

### Normal (Gaussian)

```python
torch.distributions.Normal(
    loc,        # (Tensor) mean of the distribution
    scale,      # (Tensor) standard deviation (must be positive)
    validate_args=None,
)
```

```python
from torch.distributions import Normal

# Scalar normal
dist = Normal(torch.tensor(0.0), torch.tensor(1.0))

# Batched normal
dist = Normal(torch.zeros(10), torch.ones(10))

# Sample
samples = dist.sample((100,))  # [100, 10]

# Reparameterized sample (differentiable)
rsamples = dist.rsample((100,))

# Log probability
lp = dist.log_prob(torch.tensor(0.0))  # tensor(-0.9189...)

# Entropy
ent = dist.entropy()  # tensor(1.4189...)

# Properties
print(dist.mean)      # tensor(0.)
print(dist.variance)  # tensor(1.)
print(dist.stddev)    # tensor(1.)

# CDF
cdf = dist.cdf(torch.tensor(0.0))  # tensor(0.5)
```

### LogNormal

```python
torch.distributions.LogNormal(
    loc,        # (Tensor) mean of the underlying normal distribution
    scale,      # (Tensor) std of the underlying normal distribution
    validate_args=None,
)
```

```python
from torch.distributions import LogNormal

dist = LogNormal(torch.tensor(0.0), torch.tensor(1.0))
sample = dist.sample()  # Always positive
print(dist.mean)        # tensor(exp(0.5))
print(dist.variance)    # tensor((exp(1) - 1) * exp(1))
```

### Uniform

```python
torch.distributions.Uniform(
    low,        # (Tensor) lower bound
    high,       # (Tensor) upper bound
    validate_args=None,
)
```

```python
from torch.distributions import Uniform

dist = Uniform(torch.tensor(0.0), torch.tensor(1.0))
samples = dist.sample((1000,))  # 1000 samples in [0, 1)

# Batched uniform over different ranges
dist = Uniform(torch.tensor([0.0, -1.0]), torch.tensor([1.0, 1.0]))
print(dist.mean)  # tensor([0.5, 0.0])
```

### Beta

```python
torch.distributions.Beta(
    concentration1,  # (Tensor) alpha parameter (must be positive)
    concentration0,  # (Tensor) beta parameter (must be positive)
    validate_args=None,
)
```

```python
from torch.distributions import Beta

# Beta(2, 5) - skewed toward 0
dist = Beta(torch.tensor(2.0), torch.tensor(5.0))
sample = dist.sample()  # in (0, 1)

# Beta(1, 1) - same as Uniform(0, 1)
dist = Beta(torch.tensor(1.0), torch.tensor(1.0))

# Symmetric Beta
dist = Beta(torch.tensor(2.0), torch.tensor(2.0))  # Bell-shaped on (0, 1)

print(dist.mean)      # concentration1 / (concentration1 + concentration0)
print(dist.variance)  # computed from concentration parameters
```

### Gamma

```python
torch.distributions.Gamma(
    concentration,  # (Tensor) shape parameter alpha (must be positive)
    rate,           # (Tensor) rate parameter beta = 1/scale (must be positive)
    validate_args=None,
)
```

```python
from torch.distributions import Gamma

# Gamma(1, 1) - Exponential(1)
dist = Gamma(torch.tensor(1.0), torch.tensor(1.0))

# Gamma with shape=2, rate=3
dist = Gamma(torch.tensor(2.0), torch.tensor(3.0))
sample = dist.sample()  # Always positive

print(dist.mean)      # concentration / rate
print(dist.variance)  # concentration / rate^2
```

### Exponential

```python
torch.distributions.Exponential(
    rate,       # (Tensor) rate parameter (must be positive)
    validate_args=None,
)
```

```python
from torch.distributions import Exponential

dist = Exponential(torch.tensor(1.0))
samples = dist.sample((100,))  # All positive
print(dist.mean)      # 1 / rate = 1.0
print(dist.variance)  # 1 / rate^2 = 1.0
```

### Laplace

```python
torch.distributions.Laplace(
    loc,        # (Tensor) location parameter
    scale,      # (Tensor) scale parameter (must be positive)
    validate_args=None,
)
```

```python
from torch.distributions import Laplace

dist = Laplace(torch.tensor(0.0), torch.tensor(1.0))
samples = dist.sample((100,))
print(dist.mean)      # tensor(0.)
print(dist.variance)  # tensor(2.)
```

### Cauchy

```python
torch.distributions.Cauchy(
    loc,        # (Tensor) mode/median
    scale,      # (Tensor) half-width at half-maximum (must be positive)
    validate_args=None,
)
```

```python
from torch.distributions import Cauchy

dist = Cauchy(torch.tensor(0.0), torch.tensor(1.0))
samples = dist.sample((100,))  # Heavy-tailed distribution
# Note: mean and variance are undefined for Cauchy
```

### Dirichlet

```python
torch.distributions.Dirichlet(
    concentration,  # (Tensor) concentration parameters (must be positive)
    validate_args=None,
)
```

```python
from torch.distributions import Dirichlet

# Symmetric Dirichlet
dist = Dirichlet(torch.ones(3))  # 3 categories
sample = dist.sample()  # Sums to 1.0

# Sparse Dirichlet (concentration < 1)
dist = Dirichlet(torch.tensor([0.1, 0.1, 0.1]))

# Dense Dirichlet (concentration > 1)
dist = Dirichlet(torch.tensor([10.0, 10.0, 10.0]))

print(dist.mean)      # concentration / sum(concentration)
print(dist.variance)  # Computed from concentration parameters
```

### MultivariateNormal

```python
torch.distributions.MultivariateNormal(
    loc,                    # (Tensor) mean vector
    covariance_matrix=None, # (Tensor) positive-definite covariance matrix
    precision_matrix=None,  # (Tensor) positive-definite precision matrix
    scale_tril=None,        # (Tensor) lower-triangular factor of covariance
    validate_args=None,
)
```

```python
from torch.distributions import MultivariateNormal

# Using covariance matrix
mean = torch.zeros(3)
cov = torch.eye(3)
dist = MultivariateNormal(mean, covariance_matrix=cov)

# Using scale_tril (more efficient, e.g., from Cholesky)
L = torch.linalg.cholesky(cov)
dist = MultivariateNormal(mean, scale_tril=L)

# Using precision matrix
precision = torch.inverse(cov)
dist = MultivariateNormal(mean, precision_matrix=precision)

# Sample
sample = dist.sample()  # torch.Size([3])
samples = dist.sample((100,))  # torch.Size([100, 3])

# Log probability
lp = dist.log_prob(torch.zeros(3))

# Batched: N distributions of dimension D
mean = torch.zeros(5, 3)  # 5 distributions, each in R^3
cov = torch.eye(3).unsqueeze(0).expand(5, -1, -1)
dist = MultivariateNormal(mean, covariance_matrix=cov)
```

### Chi2

```python
torch.distributions.Chi2(
    df,         # (Tensor) degrees of freedom (must be positive)
    validate_args=None,
)
```

```python
from torch.distributions import Chi2

dist = Chi2(torch.tensor(3.0))
sample = dist.sample()  # Always positive
print(dist.mean)  # tensor(3.)
```

### StudentT

```python
torch.distributions.StudentT(
    df,         # (Tensor) degrees of freedom (must be positive)
    loc=0.0,    # (Tensor) location parameter
    scale=1.0,  # (Tensor) scale parameter (must be positive)
    validate_args=None,
)
```

```python
from torch.distributions import StudentT

# Standard t-distribution with 5 degrees of freedom
dist = StudentT(torch.tensor(5.0))
sample = dist.sample()

# Non-standard location and scale
dist = StudentT(torch.tensor(10.0), loc=torch.tensor(2.0), scale=torch.tensor(0.5))
```

### FisherSnedecor (F-distribution)

```python
torch.distributions.FisherSnedecor(
    df1,        # (Tensor) degrees of freedom parameter 1
    df2,        # (Tensor) degrees of freedom parameter 2
    validate_args=None,
)
```

```python
from torch.distributions import FisherSnedecor

dist = FisherSnedecor(torch.tensor(5.0), torch.tensor(10.0))
sample = dist.sample()
```

### Wishart

```python
torch.distributions.Wishart(
    df,                     # (Tensor) degrees of freedom
    covariance_matrix=None,
    precision_matrix=None,
    scale_tril=None,
    validate_args=None,
)
```

```python
from torch.distributions import Wishart

df = torch.tensor(5.0)
cov = torch.eye(3)
dist = Wishart(df, covariance_matrix=cov)
sample = dist.sample()  # 3x3 positive definite matrix
```

###LKJCholesky

```python
torch.distributions.LKJCholesky(
    dim,        # (int) dimension of correlation matrix
    concentration,  # (Tensor) concentration parameter (>= 0)
    validate_args=None,
)
```

```python
from torch.distributions import LKJCholesky

# Concentration=1: uniform over correlation matrices
dist = LKJCholesky(dim=3, concentration=torch.tensor(1.0))
sample = dist.sample()  # Lower triangular Cholesky factor of correlation matrix
```

### HalfNormal

```python
torch.distributions.HalfNormal(
    scale,      # (Tensor) scale parameter (must be positive)
    validate_args=None,
)
```

```python
from torch.distributions import HalfNormal

dist = HalfNormal(torch.tensor(1.0))
sample = dist.sample()  # Always non-negative
```

### Pareto

```python
torch.distributions.Pareto(
    scale,      # (Tensor) scale parameter (must be positive)
    alpha,      # (Tensor) shape parameter (must be positive)
    validate_args=None,
)
```

```python
from torch.distributions import Pareto

dist = Pareto(torch.tensor(1.0), torch.tensor(1.0))
sample = dist.sample()  # Always >= scale
```

### Weibull

```python
torch.distributions.Weibull(
    scale,      # (Tensor) scale parameter (lambda, must be positive)
    concentration,  # (Tensor) shape parameter (k, must be positive)
    validate_args=None,
)
```

```python
from torch.distributions import Weibull

dist = Weibull(torch.tensor(1.0), torch.tensor(1.5))
sample = dist.sample()  # Always positive
```

---

## 3. Discrete Distributions

### Bernoulli

```python
torch.distributions.Bernoulli(
    probs=None,      # (Tensor) probability of 1
    logits=None,     # (Tensor) log-odds of 1
    validate_args=None,
)
```

```python
from torch.distributions import Bernoulli

# Using probabilities
dist = Bernoulli(torch.tensor(0.7))
sample = dist.sample()  # 0 or 1

# Using logits
dist = Bernoulli(logits=torch.tensor(0.847))  # log(0.7/0.3)

# Batched
dist = Bernoulli(torch.tensor([0.1, 0.5, 0.9]))
samples = dist.sample((100,))  # [100, 3] binary values

# Log probability
lp = dist.log_prob(torch.tensor(1.0))  # log(probs) when value=1

# Entropy
ent = dist.entropy()

# Properties
print(dist.mean)      # probs
print(dist.variance)  # probs * (1 - probs)
```

### Binomial

```python
torch.distributions.Binomial(
    total_count,     # (int or Tensor) number of trials
    probs=None,
    logits=None,
    validate_args=None,
)
```

```python
from torch.distributions import Binomial

# 10 trials, p=0.5
dist = Binomial(total_count=10, probs=torch.tensor(0.5))
sample = dist.sample()  # Integer in [0, 10]

# Multiple probabilities
dist = Binomial(total_count=20, probs=torch.tensor([0.1, 0.5, 0.9]))
samples = dist.sample((100,))  # [100, 3]

print(dist.mean)      # total_count * probs
print(dist.variance)  # total_count * probs * (1 - probs)
```

### Categorical

```python
torch.distributions.Categorical(
    probs=None,      # (Tensor) event probabilities (must sum to 1)
    logits=None,     # (Tensor) event log-odds
    validate_args=None,
)
```

```python
from torch.distributions import Categorical

# Fair die (6 sides)
dist = Categorical(probs=torch.ones(6) / 6)
sample = dist.sample()  # Integer in [0, 5]

# Loaded coin
dist = Categorical(probs=torch.tensor([0.7, 0.3]))

# Using logits
dist = Categorical(logits=torch.tensor([1.0, 2.0, 0.5]))

# Batched
dist = Categorical(probs=torch.tensor([
    [0.1, 0.5, 0.4],
    [0.3, 0.3, 0.4],
]))

# Log probability
lp = dist.log_prob(torch.tensor(0))  # log probability of class 0

print(dist.mean)
print(dist.variance)
```

### Geometric

```python
torch.distributions.Geometric(
    probs=None,
    logits=None,
    validate_args=None,
)
```

```python
from torch.distributions import Geometric

# Number of failures before first success
dist = Geometric(probs=torch.tensor(0.3))
sample = dist.sample()  # Integer >= 0

print(dist.mean)      # (1 - probs) / probs
print(dist.variance)  # (1 - probs) / probs^2
```

### Multinomial

```python
torch.distributions.Multinomial(
    total_count,     # (int or Tensor) number of trials
    probs=None,
    logits=None,
    validate_args=None,
)
```

```python
from torch.distributions import Multinomial

# Roll a die 10 times, count each face
dist = Multinomial(total_count=10, probs=torch.ones(6) / 6)
sample = dist.sample()  # [6] tensor, sum = 10

# With different probabilities
dist = Multinomial(total_count=100, probs=torch.tensor([0.5, 0.3, 0.2]))
sample = dist.sample()  # [3] tensor, sum = 100
```

### Poisson

```python
torch.distributions.Poisson(
    rate,       # (Tensor) rate parameter (must be positive)
    validate_args=None,
)
```

```python
from torch.distributions import Poisson

dist = Poisson(torch.tensor(5.0))
sample = dist.sample()  # Non-negative integer

# Batched
dist = Poisson(torch.tensor([1.0, 3.0, 10.0]))
samples = dist.sample((100,))  # [100, 3]

print(dist.mean)      # rate
print(dist.variance)  # rate
```

### NegativeBinomial

```python
torch.distributions.NegativeBinomial(
    total_count,     # (Tensor) number of successful trials
    probs=None,
    logits=None,
    validate_args=None,
)
```

```python
from torch.distributions import NegativeBinomial

dist = NegativeBinomial(total_count=5, probs=torch.tensor(0.5))
sample = dist.sample()  # Number of failures before 5 successes
```

---

## 4. Transforms

Transforms convert between different probability distributions using bijective mappings.

### Transform Base Class

```python
class torch.distributions.Transform(
    cache_size=0,
)
```

```python
# Core methods
y = transform(x)           # Forward transform
x = transform.inv(y)       # Inverse transform
lp = transform.log_abs_det_jacobian(x, y)  # Log |det J|
```

### ExpTransform

```python
from torch.distributions import transforms as T

transform = T.ExpTransform()
y = transform(torch.tensor(1.0))  # tensor(e)
x = transform.inv(y)              # tensor(1.0)
```

### SigmoidTransform

```python
transform = T.SigmoidTransform()
y = transform(torch.tensor(0.0))  # tensor(0.5)
```

### AffineTransform

```python
T.AffineTransform(
    loc,        # (Tensor) shift parameter
    scale,      # (Tensor) scale parameter
    event_dim=0,
)
```

```python
transform = T.AffineTransform(loc=1.0, scale=2.0)
y = transform(torch.tensor(3.0))  # tensor(7.0)
x = transform.inv(y)              # tensor(3.0)
```

### SoftplusTransform

```python
transform = T.SoftplusTransform()
y = transform(torch.tensor(0.0))  # tensor(log(2))
```

### TanhTransform

```python
transform = T.TanhTransform()
y = transform(torch.tensor(0.0))  # tensor(0.0)
```

### PowerTransform

```python
transform = T.PowerTransform(exponent=torch.tensor(2.0))
y = transform(torch.tensor(3.0))  # tensor(9.0)
```

### StickBreakingTransform

```python
# Used for Dirichlet: maps from R^{K-1} to simplex S^{K-1}
transform = T.StickBreakingTransform()
y = transform(torch.randn(4))  # 5-dim simplex vector
```

### LowerCholeskyTransform

```python
# Maps from flat tensor to lower-triangular matrix
transform = T.LowerCholeskyTransform()
L = transform(torch.randn(6))  # 3x3 lower triangular
```

### IndependentTransform

```python
# Wraps a transform to treat some dimensions as event dimensions
transform = T.IndependentTransform(
    T.AffineTransform(torch.zeros(3), torch.ones(3)),
    reinterpreted_batch_ndims=1,
)
```

### ComposeTransform

```python
# Chain multiple transforms
composed = T.ComposeTransform([
    T.AffineTransform(loc=0.0, scale=2.0),
    T.ExpTransform(),
])
y = composed(torch.tensor(1.0))  # exp(2.0)
```

---

## 5. TransformedDistribution

Combines a base distribution with one or more transforms.

```python
from torch.distributions import TransformedDistribution, Normal
from torch.distributions.transforms import ExpTransform, AffineTransform

# LogNormal via transform
base = Normal(torch.tensor(0.0), torch.tensor(1.0))
log_normal = TransformedDistribution(base, [ExpTransform()])

# Custom transformed distribution
# Y = exp(Affine(X)) where X ~ Normal(0, 1)
base = Normal(torch.tensor(0.0), torch.tensor(1.0))
transformed = TransformedDistribution(base, [
    AffineTransform(loc=1.0, scale=2.0),
    ExpTransform(),
])

sample = transformed.sample()
lp = transformed.log_prob(sample)
```

### Creating Custom Distributions via Transforms

```python
from torch.distributions import TransformedDistribution, Uniform
from torch.distributions.transforms import SigmoidTransform, AffineTransform

# Logistic distribution: Sigmoid(Uniform(0, 1)) scaled
logistic = TransformedDistribution(
    Uniform(torch.zeros(10), torch.ones(10)),
    [
        AffineTransform(loc=0.0, scale=1.0),
        SigmoidTransform().inv,  # logit = inverse sigmoid
    ]
)
```

---

## 6. Constraints

Constraints define the valid support for distribution parameters.

```python
from torch.distributions import constraints

# Common constraints
constraints.real                   # (-inf, inf)
constraints.positive               # (0, inf)
constraints.unit_interval          # [0, 1]
constraints.simplex                # sum-to-1 vectors
constraints.greater_than(lower)    # (lower, inf)
constraints.less_than(upper)       # (-inf, upper)
constraints.interval(lower, upper) # [lower, upper]
constraints.integer_interval(lower, upper)  # integers in [lower, upper]
constraints.nonnegative_integer    # {0, 1, 2, ...}
constraints.positive_integer       # {1, 2, 3, ...}
constraints.boolean                # {0, 1}
constraints.real_vector            # R^n
constraints.lower_cholesky         # Lower triangular with positive diagonal
constraints.positive_definite      # Positive definite matrices
constraints.positive_semidefinite  # PSD matrices
constraints.cat(groups, lengths)   # Concatenated constraints
constraints.stack(groups, dim)     # Stacked constraints
constraints.independent(base, dims) # Relaxed constraint over event dims
```

### Using Constraints

```python
from torch.distributions import Normal, constraints

# The constraint on each parameter
dist = Normal(torch.tensor(0.0), torch.tensor(1.0))
print(dist.arg_constraints)
# {'loc': Real(), 'scale': GreaterThan(lower_bound=0.0)}

# Check if value is in support
print(dist.support.check(torch.tensor(0.0)))  # tensor(True)
```

### Custom Constraints

```python
from torch.distributions import constraints
from torch.distributions.constraints import Constraint

class EvenPositiveInteger(Constraint):
    """Constraint for even positive integers."""

    def check(self, value):
        return (value >= 0) & (value % 2 == 0)

    def __repr__(self):
        return "EvenPositiveInteger()"

even_constraint = EvenPositiveInteger()
print(even_constraint.check(torch.tensor(4)))   # tensor(True)
print(even_constraint.check(torch.tensor(3)))   # tensor(False)
```

---

## 7. KL Divergence

### kl_divergence Function

```python
from torch.distributions import kl_divergence, Normal

p = Normal(torch.tensor(0.0), torch.tensor(1.0))
q = Normal(torch.tensor(1.0), torch.tensor(2.0))

kl = kl_divergence(p, q)  # KL(p || q)
print(kl)  # tensor(0.4431...)
```

### Registering Custom KL Divergence

```python
from torch.distributions import kl, Distribution

def custom_kl_divergence(p, q):
    """Compute KL(p || q) for custom distributions."""
    # KL = E_p[log p(x)] - E_p[log q(x)]
    # = -H(p) - E_p[log q(x)]
    return -p.entropy() - torch.mean(q.log_prob(p.sample((1000,)))

# Register for a pair of distribution types
kl.register_kl(CustomDistA, CustomDistB)(custom_kl_divergence)
```

### KL Divergence for Common Distribution Pairs

```python
from torch.distributions import kl_divergence, Normal, Categorical, Beta

# Normal vs Normal (closed form)
p = Normal(torch.zeros(10), torch.ones(10))
q = Normal(torch.ones(10), 2 * torch.ones(10))
kl = kl_divergence(p, q)  # torch.Size([10])

# Categorical vs Categorical
p = Categorical(probs=torch.tensor([0.25, 0.25, 0.25, 0.25]))
q = Categorical(probs=torch.tensor([0.1, 0.4, 0.3, 0.2]))
kl = kl_divergence(p, q)

# Beta vs Beta
p = Beta(torch.tensor(2.0), torch.tensor(5.0))
q = Beta(torch.tensor(1.0), torch.tensor(1.0))
kl = kl_divergence(p, q)
```

---

## 8. Independent

Reinterprets batch dimensions as event dimensions.

```python
torch.distributions.Independent(
    base_distribution,       # Base distribution
    reinterpreted_batch_ndims,  # Number of batch dims to treat as event dims
    validate_args=None,
)
```

```python
from torch.distributions import Independent, Normal

# Univariate Normal with batch shape [10]
base = Normal(torch.zeros(10), torch.ones(10))
print(base.batch_shape)  # torch.Size([10])
print(base.event_shape)  # torch.Size([])

# Treat last dim as event: multivariate-like
multi = Independent(base, 1)
print(multi.batch_shape)  # torch.Size([])
print(multi.event_shape)  # torch.Size([10])

# Log prob now sums over the event dimensions
x = torch.zeros(10)
print(base.log_prob(x).shape)   # torch.Size([10])
print(multi.log_prob(x).shape)  # torch.Size([]) - sum of 10 log probs
```

---

## 9. MixtureSameFamily

Mixture distribution with components from the same family.

```python
torch.distributions.MixtureSameFamily(
    mixture,        # Categorical distribution over components
    component,      # Distribution of components (batch shape = [K])
    validate_args=None,
)
```

```python
from torch.distributions import MixtureSameFamily, Categorical, Normal

# 3-component Gaussian mixture
mixture_weights = Categorical(probs=torch.tensor([0.3, 0.5, 0.2]))
components = Normal(
    loc=torch.tensor([-2.0, 0.0, 3.0]),   # means
    scale=torch.tensor([0.5, 1.0, 0.5]),   # stds
)

gmm = MixtureSameFamily(mixture_weights, components)

sample = gmm.sample((1000,))
lp = gmm.log_prob(torch.linspace(-5, 5, 100))
print(gmm.mean)
```

---

## 10. LowRankMultivariateNormal

Multivariate normal with low-rank plus diagonal covariance.

```python
torch.distributions.LowRankMultivariateNormal(
    loc,            # (Tensor) mean
    cov_factor,     # (Tensor) low-rank factor (n x k)
    cov_diag,       # (Tensor) diagonal of covariance
    validate_args=None,
)
```

```python
from torch.distributions import LowRankMultivariateNormal

loc = torch.zeros(10)
cov_factor = torch.randn(10, 2)  # Rank-2 factor
cov_diag = torch.ones(10)

dist = LowRankMultivariateNormal(loc, cov_factor, cov_diag)
sample = dist.sample()
```

---

## 11. Von Mises

```python
torch.distributions.VonMises(
    loc,        # (Tensor) mode of the distribution (in radians)
    concentration,  # (Tensor) concentration parameter (must be positive)
    validate_args=None,
)
```

```python
from torch.distributions import VonMises

# Directional distribution on a circle
dist = VonMises(torch.tensor(0.0), torch.tensor(5.0))
sample = dist.sample()  # Angle in [-pi, pi]
```

---

## 12. OneHotCategorical

```python
torch.distributions.OneHotCategorical(
    probs=None,
    logits=None,
    validate_args=None,
)
```

```python
from torch.distributions import OneHotCategorical

dist = OneHotCategorical(probs=torch.tensor([0.1, 0.3, 0.6]))
sample = dist.sample()  # One-hot vector like [0, 0, 1]
```

### OneHotCategoricalStraightThrough

```python
from torch.distributions import OneHotCategoricalStraightThrough

# Straight-through estimator for differentiable sampling
dist = OneHotCategoricalStraightThrough(probs=torch.tensor([0.1, 0.3, 0.6]))
sample = dist.rsample()  # One-hot with gradient (straight-through)
```

---

## 13. RelaxedOneHotCategorical (Concrete)

```python
torch.distributions.RelaxedOneHotCategorical(
    temperature,    # (Tensor) temperature parameter (lower = more discrete)
    probs=None,
    logits=None,
    validate_args=None,
)
```

```python
from torch.distributions import RelaxedOneHotCategorical

# Gumbel-Softmax / Concrete distribution
temperature = torch.tensor(0.5)
dist = RelaxedOneHotCategorical(temperature, probs=torch.tensor([0.1, 0.3, 0.6]))

# Differentiable sample (sums to 1 but not exactly one-hot)
sample = dist.rsample()  # e.g., [0.05, 0.15, 0.80]
```

---

## 14. Complete Example: Variational Autoencoder with Distributions

```python
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def loss(self, x, x_recon, mu, logvar):
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')

        # KL divergence using distributions API
        p = Normal(torch.zeros_like(mu), torch.ones_like(std))  # Prior N(0, 1)
        q = Normal(mu, (0.5 * logvar).exp())                     # Posterior
        kl = kl_divergence(q, p).sum()

        return recon_loss + kl

# Usage
model = VAE(input_dim=784, hidden_dim=400, latent_dim=20)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for batch_x, _ in train_loader:
        batch_x = batch_x.view(-1, 784)
        x_recon, mu, logvar = model(batch_x)
        loss = model.loss(batch_x, x_recon, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
