# PyTorch Probability Distributions - Comprehensive Reference

This chapter covers the `torch.distributions` module, which provides composable probability distributions and composable tensor operations for probabilistic modeling. It includes all distribution classes, transforms, constraints, and KL divergence utilities.

---

## 1. Distribution Base Class

### torch.distributions.Distribution

```python
torch.distributions.Distribution(
    batch_shape=torch.Size(),
    event_shape=torch.Size(),
    validate_args=False,
)
```

All probability distributions inherit from this base class.

### Core Properties

| Property | Type | Description |
|----------|------|-------------|
| `batch_shape` | torch.Size | Shape of independent distributions |
| `event_shape` | torch.Size | Shape of a single sample (event) |
| `support` | Constraint | Support of the distribution |

### Core Methods

#### sample(sample_shape=torch.Size())

Generates a sample from the distribution.

```python
dist = torch.distributions.Normal(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0]))
sample = dist.sample()            # tensor([...]) shape: batch_shape
samples = dist.sample((100,))     # shape: (100,) + batch_shape
samples = dist.sample((10, 20))   # shape: (10, 20) + batch_shape
```

#### rsample(sample_shape=torch.Size())

Generates a reparameterized sample (differentiable w.r.t. parameters).

```python
# Only available for distributions with reparameterized gradients
dist = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0], requires_grad=True))
sample = dist.rsample()          # Differentiable
sample.backward()
```

#### log_prob(value)

Returns the log probability density/mass function evaluated at `value`.

```python
dist = torch.distributions.Normal(0, 1)
lp = dist.log_prob(torch.tensor(0.0))   # tensor(-0.9189...) = -0.5*log(2*pi)
```

#### entropy()

Returns the entropy of the distribution.

```python
dist = torch.distributions.Normal(0, 1)
h = dist.entropy()  # tensor(1.4189...) = 0.5 + 0.5*log(2*pi)
```

#### enumerate_support(expand=True)

Returns tensor containing all values supported by a discrete distribution.

```python
dist = torch.distributions.Bernoulli(0.3)
support = dist.enumerate_support()  # tensor([0., 1.])
```

### Statistical Properties

```python
dist = torch.distributions.Normal(0, 1)

dist.mean          # tensor(0.) - Mean of the distribution
dist.variance      # tensor(1.) - Variance
dist.stddev        # tensor(1.) - Standard deviation
dist.mode          # tensor(0.) - Mode (not all distributions support)
```

### Shape Convention

```
sample_shape:  independent draws from the distribution
batch_shape:   independent (but not identical) distributions
event_shape:   dependent dimensions within a single distribution

Total sample shape = sample_shape + batch_shape + event_shape
```

```python
# 3 independent Normal distributions, each producing 2D events
dist = torch.distributions.MultivariateNormal(
    loc=torch.randn(3, 2),                      # batch_shape=(3,), event_shape=(2,)
    covariance_matrix=torch.eye(2).unsqueeze(0).expand(3, 2, 2),
)
print(dist.batch_shape)   # torch.Size([3])
print(dist.event_shape)   # torch.Size([2])
print(dist.sample().shape)          # torch.Size([3, 2])
print(dist.sample((5,)).shape)      # torch.Size([5, 3, 2])
```

---

## 2. Continuous Distributions

### Normal(loc, scale)

Univariate normal (Gaussian) distribution.

**Formula:** `f(x) = (1 / (scale * sqrt(2*pi))) * exp(-0.5 * ((x - loc) / scale)^2)`

```python
from torch.distributions import Normal

dist = Normal(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0]))
dist.sample()          # tensor([...])
dist.log_prob(0.0)     # Log probability at x=0
dist.entropy()         # tensor([1.4189, 2.1121])
dist.mean              # tensor([0., 1.])
dist.variance          # tensor([1., 4.])
dist.stddev            # tensor([1., 2.])

# Reparameterized sampling (differentiable)
mu = torch.tensor([0.0], requires_grad=True)
sigma = torch.tensor([1.0], requires_grad=True)
dist = Normal(mu, sigma)
z = dist.rsample()          # Differentiable w.r.t. mu and sigma
z.sum().backward()
```

### LogNormal(loc, scale)

Log-normal distribution. If `X ~ Normal(loc, scale)`, then `exp(X) ~ LogNormal(loc, scale)`.

```python
from torch.distributions import LogNormal

dist = LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
dist.sample()       # Always positive
dist.mean           # tensor([1.6487]) = exp(0 + 0.5)
dist.variance       # tensor([4.6708])
```

### Uniform(low, high)

Continuous uniform distribution.

```python
from torch.distributions import Uniform

dist = Uniform(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 10.0]))
dist.sample()       # tensor([...]) in [low, high)
dist.log_prob(0.5)  # tensor([-0.0000, -2.3026])
dist.entropy()      # tensor([0.0000, 2.3026]) = log(high - low)
```

### Beta(concentration1, concentration0)

Beta distribution parameterized by alpha (concentration1) and beta (concentration0).

```python
from torch.distributions import Beta

dist = Beta(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0]))
dist.sample()       # tensor([...]) in (0, 1)
dist.mean           # tensor([0.5000, 0.4000]) = alpha / (alpha + beta)
dist.variance       # alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1))
```

### Gamma(concentration, rate)

Gamma distribution.

```python
from torch.distributions import Gamma

dist = Gamma(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]))
dist.sample()       # Always positive
dist.mean           # concentration / rate
dist.entropy()
```

### Exponential(rate)

Exponential distribution.

```python
from torch.distributions import Exponential

dist = Exponential(torch.tensor([1.0, 0.5]))
dist.sample()       # Always positive
dist.mean           # 1 / rate
```

### Laplace(loc, scale)

Laplace distribution.

```python
from torch.distributions import Laplace

dist = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
dist.sample()
dist.log_prob(0.0)  # -log(2*scale)
```

### Cauchy(loc, scale)

Cauchy (Lorentz) distribution. Heavy tails, undefined mean and variance.

```python
from torch.distributions import Cauchy

dist = Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
dist.sample()
dist.log_prob(0.0)  # -log(pi * scale)
```

### Chi2(df)

Chi-squared distribution with `df` degrees of freedom. Special case of Gamma(df/2, 1/2).

```python
from torch.distributions import Chi2

dist = Chi2(torch.tensor([1.0, 2.0, 5.0]))
dist.sample()
dist.mean           # df
dist.variance       # 2 * df
```

### Dirichlet(concentration)

Dirichlet distribution over the simplex.

```python
from torch.distributions import Dirichlet

dist = Dirichlet(torch.tensor([1.0, 1.0, 1.0]))
dist.sample()       # tensor([...]) sums to 1.0
dist.mean           # concentration / sum(concentration)
dist.entropy()

# Sparse Dirichlet (concentration < 1)
sparse = Dirichlet(torch.tensor([0.1, 0.1, 0.1]))

# Symmetric Dirichlet
sym = Dirichlet(torch.tensor([2.0, 2.0, 2.0]))
```

### FisherSnedecor(df1, df2)

F-distribution (Fisher-Snedecor).

```python
from torch.distributions import FisherSnedecor

dist = FisherSnedecor(
    df1=torch.tensor([1.0]),   # Numerator degrees of freedom
    df2=torch.tensor([2.0]),   # Denominator degrees of freedom
)
dist.sample()
dist.mean           # df2 / (df2 - 2) for df2 > 2
```

### Gumbel(loc, scale)

Gumbel distribution (Type I extreme value).

```python
from torch.distributions import Gumbel

dist = Gumbel(torch.tensor([0.0]), torch.tensor([1.0]))
dist.sample()
dist.mean           # loc + scale * euler_gamma
```

### HalfCauchy(scale)

Half-Cauchy distribution (Cauchy restricted to positive reals).

```python
from torch.distributions import HalfCauchy

dist = HalfCauchy(torch.tensor([1.0]))
dist.sample()       # Always positive
```

### HalfNormal(scale)

Half-normal distribution (folded normal).

```python
from torch.distributions import HalfNormal

dist = HalfNormal(torch.tensor([1.0]))
dist.sample()       # Always non-negative
dist.mean           # scale * sqrt(2/pi)
```

### Independent(base_distribution, reinterpreted_batch_ndims)

Wraps a distribution to reinterpret some batch dimensions as event dimensions.

```python
from torch.distributions import Independent, Normal

# 10 independent Normal distributions, each producing a 3D event
base = Normal(torch.zeros(10, 3), torch.ones(10, 3))
dist = Independent(base, reinterpreted_batch_ndims=1)

print(base.batch_shape)   # torch.Size([10, 3])
print(base.event_shape)   # torch.Size([])
print(dist.batch_shape)   # torch.Size([10])
print(dist.event_shape)   # torch.Size([3])

# log_prob now sums over the event dimensions
dist.log_prob(torch.zeros(10, 3))  # shape: (10,)
```

### KolmogorovSmirnov(distribution)

Kolmogorov-Smirnov test distribution.

```python
from torch.distributions import KolmogorovSmirnov

# Used for KS test p-value computation
dist = KolmogorovSmirnov(distribution)
```

### LKJCholesky(dim, concentration, validate_args)

LKJ distribution over Cholesky factors of correlation matrices.

```python
from torch.distributions import LKJCholesky

dist = LKJCholesky(dim=3, concentration=torch.tensor([1.0]))
sample = dist.sample()  # Lower triangular Cholesky factor of correlation matrix
```

### Logistic(loc, scale)

Logistic distribution.

```python
from torch.distributions import Logistic

dist = Logistic(torch.tensor([0.0]), torch.tensor([1.0]))
dist.sample()
dist.mean           # loc
dist.variance       # scale^2 * pi^2 / 3
```

### LowRankMultivariateNormal(loc, cov_factor, cov_diag)

Multivariate normal with low-rank plus diagonal covariance.

```python
from torch.distributions import LowRankMultivariateNormal

loc = torch.zeros(5)
cov_factor = torch.randn(5, 2)
cov_diag = torch.ones(5)

dist = LowRankMultivariateNormal(loc, cov_factor, cov_diag)
dist.sample()       # shape: (5,)
dist.mean           # loc
dist.entropy()
```

### MixtureSameFamily(mixture_distribution, component_distribution)

Mixture distribution where all components are from the same family.

```python
from torch.distributions import MixtureSameFamily, Categorical, Normal

# 1D Gaussian mixture model with 3 components
mix = Categorical(torch.tensor([0.3, 0.5, 0.2]))
comp = Normal(torch.tensor([-1.0, 0.0, 2.0]), torch.tensor([0.5, 1.0, 0.3]))
gmm = MixtureSameFamily(mix, comp)

gmm.sample()        # Sample from mixture
gmm.log_prob(0.0)   # Log probability at x=0
gmm.mean            # Weighted mean of components

# 2D Gaussian mixture
mix = Categorical(torch.ones(5) / 5)
comp = Normal(torch.randn(5, 3), torch.ones(5, 3))
gmm = MixtureSameFamily(mix, comp)
```

### MultivariateNormal(loc, covariance_matrix, precision_matrix, scale_tril)

Multivariate normal (Gaussian) distribution.

```python
from torch.distributions import MultivariateNormal

# From covariance matrix
dist = MultivariateNormal(
    loc=torch.zeros(3),
    covariance_matrix=torch.eye(3),
)

# From scale_tril (Cholesky factor of covariance)
L = torch.tensor([[1.0, 0.0, 0.0],
                   [0.5, 1.0, 0.0],
                   [0.3, 0.2, 1.0]])
dist = MultivariateNormal(torch.zeros(3), scale_tril=L)

# From precision matrix
dist = MultivariateNormal(torch.zeros(3), precision_matrix=torch.eye(3))

# Batch of multivariate normals
dist = MultivariateNormal(
    loc=torch.randn(10, 3),    # 10 distributions in 3D
    covariance_matrix=torch.eye(3).unsqueeze(0).expand(10, 3, 3),
)

dist.sample()        # shape: (10, 3)
dist.log_prob(torch.zeros(10, 3))  # shape: (10,)
dist.entropy()       # shape: (10,)
dist.mean            # loc
dist.covariance_matrix
dist.precision_matrix
```

### NegativeBinomial(total_count, probs, logits)

Negative binomial distribution (counts until `total_count` successes).

```python
from torch.distributions import NegativeBinomial

dist = NegativeBinomial(
    total_count=torch.tensor([5.0]),
    probs=torch.tensor([0.5]),
)
dist.sample()
dist.mean           # total_count * probs / (1 - probs)
```

### Pareto(scale, alpha)

Pareto (Type I) distribution.

```python
from torch.distributions import Pareto

dist = Pareto(torch.tensor([1.0]), torch.tensor([1.0]))
dist.sample()       # Always >= scale
dist.mean           # alpha * scale / (alpha - 1) for alpha > 1
```

### StudentT(df, loc, scale)

Student's t-distribution.

```python
from torch.distributions import StudentT

dist = StudentT(df=torch.tensor([1.0, 5.0, 30.0]))
dist.sample()
dist.mean           # loc for df > 1
dist.variance       # df / (df - 2) * scale^2 for df > 2

# With location and scale
dist = StudentT(df=3, loc=0, scale=1)
```

### TransformedDistribution(base_distribution, transforms)

Distribution obtained by applying a sequence of transforms to a base distribution.

```python
from torch.distributions import TransformedDistribution, Normal, ExpTransform

# LogNormal via transform
base = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
transforms = [ExpTransform()]
log_normal = TransformedDistribution(base, transforms)

# This is equivalent to LogNormal(0, 1)
log_normal.sample()
log_normal.log_prob(torch.tensor([1.0]))
```

### VonMises(loc, concentration)

Von Mises distribution (circular normal).

```python
from torch.distributions import VonMises

dist = VonMises(torch.tensor([0.0]), torch.tensor([1.0]))
dist.sample()       # Angle in [-pi, pi]
dist.mean           # loc
```

### Weibull(scale, concentration)

Weibull distribution.

```python
from torch.distributions import Weibull

dist = Weibull(torch.tensor([1.0]), torch.tensor([2.0]))
dist.sample()       # Always positive
dist.mean           # scale * gamma(1 + 1/concentration)
```

### Wishart(df, covariance_matrix, precision_matrix, scale_tril)

Wishart distribution over positive definite matrices.

```python
from torch.distributions import Wishart

dist = Wishart(
    df=torch.tensor([5.0]),
    covariance_matrix=torch.eye(3).unsqueeze(0),
)
dist.sample()       # 3x3 positive definite matrix
```

---

## 3. Discrete Distributions

### Bernoulli(probs, logits)

Bernoulli distribution (binary: 0 or 1).

```python
from torch.distributions import Bernoulli

dist = Bernoulli(torch.tensor([0.3, 0.7]))
dist.sample()           # tensor([0. or 1., 0. or 1.])
dist.probs              # tensor([0.3, 0.7])
dist.logits             # log(p / (1-p))
dist.log_prob(torch.tensor([1.0, 0.0]))
dist.entropy()          # -p*log(p) - (1-p)*log(1-p)
dist.enumerate_support()  # tensor([0., 1.])
```

### Binomial(total_count, probs, logits)

Binomial distribution (number of successes in n independent trials).

```python
from torch.distributions import Binomial

dist = Binomial(
    total_count=torch.tensor([100]),
    probs=torch.tensor([0.5]),
)
dist.sample()           # Integer count <= total_count
dist.mean               # total_count * probs
dist.variance           # total_count * probs * (1 - probs)
```

### Categorical(probs, logits)

Categorical distribution over `{0, 1, ..., K-1}`.

```python
from torch.distributions import Categorical

dist = Categorical(torch.tensor([0.1, 0.2, 0.7]))
dist.sample()           # tensor(0, 1, or 2)
dist.probs              # Normalized probabilities
dist.logits             # Unnormalized log probabilities
dist.log_prob(torch.tensor(2))
dist.entropy()
dist.enumerate_support()  # tensor([0, 1, 2])
```

### Geometric(probs, logits)

Geometric distribution (number of trials until first success).

```python
from torch.distributions import Geometric

dist = Geometric(torch.tensor([0.5]))
dist.sample()           # Integer >= 0
dist.mean               # (1 - probs) / probs
dist.entropy()
```

### Multinomial(total_count, probs, logits)

Multinomial distribution (generalization of Binomial to K categories).

```python
from torch.distributions import Multinomial

dist = Multinomial(
    total_count=torch.tensor([100]),
    probs=torch.tensor([0.2, 0.3, 0.5]),
)
dist.sample()           # tensor([...]) summing to total_count
dist.mean               # total_count * probs
```

### Poisson(rate)

Poisson distribution.

```python
from torch.distributions import Poisson

dist = Poisson(torch.tensor([1.0, 5.0, 10.0]))
dist.sample()           # Non-negative integer
dist.mean               # rate
dist.variance           # rate
```

### OneHotCategorical(probs, logits)

One-hot representation of a Categorical distribution.

```python
from torch.distributions import OneHotCategorical

dist = OneHotCategorical(torch.tensor([0.1, 0.2, 0.7]))
dist.sample()           # tensor([0., 0., 1.]) (one-hot)
dist.log_prob(torch.tensor([0., 0., 1.]))
dist.entropy()
```

### OneHotCategoricalStraightThrough(probs, logits)

One-hot categorical with straight-through gradient estimator.

```python
from torch.distributions import OneHotCategoricalStraightThrough

dist = OneHotCategoricalStraightThrough(torch.tensor([0.1, 0.2, 0.7]))
sample = dist.rsample()  # Differentiable, forward pass is one-hot
```

---

## 4. Transforms

### Transform Base Class

```python
class torch.distributions.transforms.Transform(
    cache_size=0,
)
```

**Key methods:**
- `forward(x)`: Apply the transform.
- `inverse(y)`: Compute the inverse.
- `log_abs_det_jacobian(x, y)`: Log absolute value of determinant of Jacobian.
- `inv`: Returns the inverse transform.

### ExpTransform()

Exponential transform: `y = exp(x)`.

```python
from torch.distributions.transforms import ExpTransform

t = ExpTransform()
t.forward(torch.tensor(0.0))    # tensor(1.0)
t.inverse(torch.tensor(1.0))    # tensor(0.0)
t.log_abs_det_jacobian(torch.tensor(0.0), torch.tensor(1.0))  # tensor(0.0) = log(1)
```

### PowerTransform(exponent)

Power transform: `y = x^exponent`.

```python
from torch.distributions.transforms import PowerTransform

t = PowerTransform(exponent=2.0)
t.forward(torch.tensor(3.0))    # tensor(9.0)
```

### SigmoidTransform()

Sigmoid transform: `y = 1 / (1 + exp(-x))`.

```python
from torch.distributions.transforms import SigmoidTransform

t = SigmoidTransform()
t.forward(torch.tensor(0.0))    # tensor(0.5)
t.inverse(torch.tensor(0.5))    # tensor(0.0)
```

### AffineTransform(loc, scale)

Affine transform: `y = loc + scale * x`.

```python
from torch.distributions.transforms import AffineTransform

t = AffineTransform(loc=1.0, scale=2.0)
t.forward(torch.tensor(3.0))    # tensor(7.0)
t.inverse(torch.tensor(7.0))    # tensor(3.0)
t.log_abs_det_jacobian(torch.tensor(3.0), torch.tensor(7.0))  # log(|2|) = 0.693
```

### SoftmaxTransform()

Softmax transform (maps R^K -> simplex^K). Not bijective (sum is always 1).

```python
from torch.distributions.transforms import SoftmaxTransform

t = SoftmaxTransform()
t.forward(torch.tensor([1.0, 2.0, 3.0]))  # tensor([0.0900, 0.2447, 0.6652])
```

### StickBreakingTransform()

Stick-breaking transform for constructing simplex vectors.

```python
from torch.distributions.transforms import StickBreakingTransform

t = StickBreakingTransform()
# Maps unconstrained reals to the simplex
```

### LowerCholeskyTransform()

Maps unconstrained matrices to lower-triangular matrices with positive diagonal.

```python
from torch.distributions.transforms import LowerCholeskyTransform

t = LowerCholeskyTransform()
# Used for covariance matrix parameterization
```

### CorrCholeskyTransform()

Maps unconstrained matrices to Cholesky factors of correlation matrices.

```python
from torch.distributions.transforms import CorrCholeskyTransform

t = CorrCholeskyTransform()
```

### IndependentTransform(base_transform, reinterpreted_batch_ndims)

Wraps a transform to reinterpret batch dims as event dims.

```python
from torch.distributions.transforms import IndependentTransform

t = IndependentTransform(ExpTransform(), reinterpreted_batch_ndims=1)
```

### ComposeTransform(parts)

Composes multiple transforms.

```python
from torch.distributions.transforms import ComposeTransform, AffineTransform, ExpTransform

t = ComposeTransform([AffineTransform(0, 2), ExpTransform()])
t.forward(torch.tensor(0.0))  # exp(2*0) = 1.0
```

---

## 5. Constraints

### torch.distributions.constraints

Constraint objects define the support of distributions.

```python
from torch.distributions import constraints

# Standard constraints
constraints.real              # (-inf, inf)
constraints.positive          # (0, inf)
constraints.unit_interval     # (0, 1)
constraints.simplex           # x_i >= 0, sum(x) = 1
constraints.categorical       # {0, 1, ..., K-1}
constraints.boolean           # {0, 1}
constraints.lower_cholesky    # Lower triangular with positive diagonal
constraints.positive_definite # Symmetric positive definite matrices
constraints.corr_cholesky     # Cholesky of correlation matrix
constraints.real_vector       # R^n (like real but for vectors)
constraints.nonnegative_integer
constraints.positive_integer
constraints.integer_interval(lower, upper)

# Check if a value is in the support
constraint = constraints.unit_interval
constraint.check(torch.tensor([0.5, 1.5, -0.1]))
# tensor([ True, False, False])
```

### Custom Constraint

```python
class Interval(constraints.Constraint):
    """Custom interval constraint."""

    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def check(self, value):
        return (value >= self.lower) & (value <= self.upper)
```

---

## 6. KL Divergence

### kl_divergence(p, q)

Computes the KL divergence `KL(p || q)` between two distributions.

```python
from torch.distributions import kl_divergence, Normal

p = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
q = Normal(torch.tensor([1.0]), torch.tensor([2.0]))

kl = kl_divergence(p, q)
# KL(p || q) = log(sigma_q/sigma_p) + (sigma_p^2 + (mu_p - mu_q)^2) / (2*sigma_q^2) - 0.5
```

### register_kl(type_p, type_q)

Registers a custom KL divergence function for a pair of distribution types.

```python
from torch.distributions import kl, Distribution

@kl.register_kl(MyDistribution, Normal)
def _kl_my_dist_normal(p, q):
    """Custom KL divergence between MyDistribution and Normal."""
    # Compute KL(p || q) analytically
    kl_div = torch.log(q.scale / p.stddev) + \
             (p.variance + (p.mean - q.loc)**2) / (2 * q.scale**2) - 0.5
    return kl_div
```

### Common KL Divergence Formulas

| p | q | KL(p || q) |
|---|---|-------------|
| Normal(mu1, s1) | Normal(mu2, s2) | log(s2/s1) + (s1^2 + (mu1-mu2)^2)/(2*s2^2) - 0.5 |
| Bernoulli(p) | Bernoulli(q) | p*log(p/q) + (1-p)*log((1-p)/(1-q)) |
| Categorical(p) | Categorical(q) | sum(p * log(p/q)) |
| Beta(a1,b1) | Beta(a2,b2) | log(B(a2,b2)/B(a1,b1)) + (a1-a2)*psi(a1) + (b1-b2)*psi(b1) + (a2-a1+b2-b1)*psi(a1+b1) |

---

## 7. Distribution Validation

### validate_args

```python
# Enable validation (raises errors for invalid parameters)
dist = Normal(torch.tensor([0.0]), torch.tensor([-1.0]))  # Error: scale must be positive

# Disable validation (silent, faster)
dist = Normal(torch.tensor([0.0]), torch.tensor([-1.0]), validate_args=False)

# Global validation setting
torch.distributions.Distribution.set_default_validate_args(False)
```

### Checking Support

```python
dist = Normal(0, 1)
value = torch.tensor([0.0, 5.0])
is_valid = dist.support.check(value)
# tensor([True, True])  (Normal has real support)

dist = Bernoulli(0.5)
is_valid = dist.support.check(torch.tensor([0.0, 0.5, 1.0]))
# tensor([True, False, True])
```

---

## 8. Example: Variational Inference

### Variational Auto-Encoder (VAE)

```python
import torch
import torch.nn as nn
import torch.distributions as dist

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def loss_function(self, x, x_recon, mu, logvar):
        # Reconstruction loss (Bernoulli or Gaussian)
        BCE = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')

        # KL divergence using analytical formula
        # KL(q(z|x) || p(z)) where q = Normal(mu, std), p = Normal(0, 1)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Using distributions API:
        # q = dist.Normal(mu, torch.exp(0.5 * logvar))
        # p = dist.Normal(torch.zeros_like(mu), torch.ones_like(mu))
        # KLD = dist.kl_divergence(q, p).sum()

        return BCE + KLD
```

### Using distributions for ELBO computation

```python
def compute_elbo(model, x):
    """Compute Evidence Lower Bound using distributions API."""
    mu, logvar = model.encode(x)
    std = torch.exp(0.5 * logvar)

    # Variational posterior: q(z|x)
    q_z = dist.Normal(mu, std)
    # Prior: p(z)
    p_z = dist.Normal(torch.zeros_like(mu), torch.ones_like(std))

    # KL divergence
    kl = dist.kl_divergence(q_z, p_z).sum(dim=-1)

    # Sample z from q(z|x)
    z = q_z.rsample()

    # Reconstruction: p(x|z)
    x_recon = model.decode(z)
    p_x_z = dist.Bernoulli(probs=x_recon)
    reconstruction_log_prob = p_x_z.log_prob(x).sum(dim=-1)

    # ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
    elbo = reconstruction_log_prob - kl
    return -elbo.mean()  # Negative ELBO for minimization
```

---

## 9. Example: Policy Gradient Methods

### Reinforce with Categorical Policy

```python
import torch
import torch.distributions as dist
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state):
        return self.net(state)

# REINFORCE algorithm
def reinforce_update(policy_net, optimizer, episodes):
    policy_loss = []

    for states, actions, rewards in episodes:
        # Convert to tensors
        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)

        # Compute returns
        returns = compute_returns(rewards, gamma=0.99)

        # Compute log probabilities using Categorical distribution
        probs = policy_net(states)
        distribution = dist.Categorical(probs=probs)
        log_probs = distribution.log_prob(actions)

        # Policy gradient loss
        loss = -(log_probs * returns).sum()
        policy_loss.append(loss)

    # Update
    total_loss = torch.stack(policy_loss).sum()
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

### Continuous Policy with Normal Distribution

```python
class ContinuousPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        h = self.fc(state)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(-20, 2)
        std = torch.exp(log_std)
        return mu, std

    def get_distribution(self, state):
        mu, std = self.forward(state)
        return dist.Normal(mu, std)

    def act(self, state):
        distribution = self.get_distribution(state)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum(dim=-1)
        return action, log_prob
```

---

## 10. Distribution Summary Table

### Continuous Distributions

| Distribution | Parameters | Support | rsample |
|-------------|-----------|---------|---------|
| Normal | loc, scale | real | Yes |
| LogNormal | loc, scale | positive | Yes |
| Uniform | low, high | [low, high) | Yes |
| Beta | concentration1, concentration0 | (0, 1) | Yes |
| Gamma | concentration, rate | positive | Yes |
| Exponential | rate | positive | Yes |
| Laplace | loc, scale | real | Yes |
| Cauchy | loc, scale | real | Yes |
| Chi2 | df | positive | Yes |
| Dirichlet | concentration | simplex | Yes |
| FisherSnedecor | df1, df2 | positive | Yes |
| Gumbel | loc, scale | real | Yes |
| HalfCauchy | scale | positive | Yes |
| HalfNormal | scale | positive | Yes |
| Logistic | loc, scale | real | Yes |
| MultivariateNormal | loc, cov/prec/scale | real^n | Yes |
| LowRankMultivariateNormal | loc, cov_factor, cov_diag | real^n | Yes |
| MixtureSameFamily | mixture, component | component support | No |
| NegativeBinomial | total_count, probs | nonneg int | No |
| Pareto | scale, alpha | [scale, inf) | Yes |
| StudentT | df, loc, scale | real | Yes |
| TransformedDistribution | base, transforms | transform-dependent | conditional |
| VonMises | loc, concentration | [-pi, pi] | No |
| Weibull | scale, concentration | positive | Yes |
| Wishart | df, cov | pos. def. | No |

### Discrete Distributions

| Distribution | Parameters | Support |
|-------------|-----------|---------|
| Bernoulli | probs/logits | {0, 1} |
| Binomial | total_count, probs | {0, 1, ..., n} |
| Categorical | probs/logits | {0, 1, ..., K-1} |
| Geometric | probs | {0, 1, 2, ...} |
| Multinomial | total_count, probs | {count_0, ..., count_K} |
| Poisson | rate | {0, 1, 2, ...} |
| OneHotCategorical | probs/logits | {e_0, ..., e_{K-1}} |
| OneHotCategoricalStraightThrough | probs/logits | {e_0, ..., e_{K-1}} |
