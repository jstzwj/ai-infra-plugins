# Random Numbers in JAX

This document provides an exhaustive reference for JAX's random number generation system. JAX uses an explicit, stateless PRNG (Pseudo-Random Number Generator) design that differs fundamentally from NumPy's global state approach. This reference covers the design philosophy, PRNG key management, all distribution functions, PRNG algorithms, and best practices.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [PRNG Key Management](#2-prng-key-management)
3. [Continuous Distributions](#3-continuous-distributions)
4. [Discrete Distributions](#4-discrete-distributions)
5. [Multivariate Distributions](#5-multivariate-distributions)
6. [Sampling Utilities](#6-sampling-utilities)
7. [PRNG Algorithms](#7-prng-algorithms)
8. [Key Types and key_impl](#8-key-types-and-key_impl)
9. [Best Practices](#9-best-practices)

---

## 1. Design Philosophy

### Why Explicit State?

NumPy uses a global random state (`np.random.seed(42)`) that is implicitly mutated on each call. This design creates problems for:

1. **Reproducibility**: Side effects make it hard to reproduce specific sequences.
2. **Parallelism**: A global mutable state cannot be safely shared across threads or devices.
3. **Composability**: JAX transformations like `jit`, `vmap`, and `pmap` require deterministic behavior.

JAX solves this by making the PRNG state explicit. Instead of a global seed, you pass a PRNG **key** to every random function. Functions never mutate keys; instead, you "split" keys to generate new independent keys.

```python
import jax
import jax.numpy as jnp

# JAX: explicit state (keys are never mutated)
key = jax.random.key(42)
key, subkey = jax.random.split(key)
x = jax.random.normal(subkey, shape=(3,))
print(x)  # deterministic given key
```

### Key Principles

1. **Stateless**: Random functions never mutate the key. The same key always produces the same output.
2. **Explicit**: You must pass a key to every random function call.
3. **Splittable**: Keys can be split into independent subkeys for use in different parts of the program.
4. **JIT-compatible**: Random operations can be safely used inside `jax.jit`.

---

## 2. PRNG Key Management

### Creating Keys

```python
import jax
import jax.numpy as jnp

# Create a key from an integer seed
key = jax.random.key(42)
print(f"Key: {key}")
print(f"Key type: {type(key)}")
print(f"Key dtype: {key.dtype}")

# Create multiple keys from different seeds
key1 = jax.random.key(0)
key2 = jax.random.key(1)
key3 = jax.random.key(42)

# Same seed produces the same key
key_a = jax.random.key(42)
key_b = jax.random.key(42)
print(f"Same seed: {jax.random.key_data(key_a) == jax.random.key_data(key_b)}")
```

### Splitting Keys

`jax.random.split` is the primary way to generate independent subkeys from a parent key.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(42)

# Split into 2 keys (default)
key1, key2 = jax.random.split(key)
print(f"key1: {key1}")
print(f"key2: {key2}")

# Split into N keys
keys = jax.random.split(key, num=5)
print(f"5 keys shape: {keys.shape}")  # (5,)
for i, k in enumerate(keys):
    print(f"  key[{i}]: {k}")
```

### The Split Pattern

The standard pattern for using keys in JAX is to split the key before each use, saving the remainder for future operations:

```python
import jax
import jax.numpy as jnp

key = jax.random.key(42)

# Split before each use
key, subkey1 = jax.random.split(key)
x = jax.random.normal(subkey1, (3,))

key, subkey2 = jax.random.split(key)
y = jax.random.uniform(subkey2, (3,))

key, subkey3 = jax.random.split(key)
mask = jax.random.bernoulli(subkey3, 0.5, (3,))

print(f"x: {x}")
print(f"y: {y}")
print(f"mask: {mask}")
```

### fold_in: Deterministic Key Derivation

`jax.random.fold_in` combines a key with integer data to produce a new key. This is useful for creating deterministic, data-dependent keys.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(42)

# Create a key that depends on an index
# Useful in loops: each iteration gets a deterministic key
for i in range(3):
    iteration_key = jax.random.fold_in(key, i)
    value = jax.random.normal(iteration_key)
    print(f"iteration {i}: {value:.4f}")

# Always produces the same values for the same (key, i) pair
for i in range(3):
    iteration_key = jax.random.fold_in(key, i)
    value = jax.random.normal(iteration_key)
    print(f"repeat {i}: {value:.4f}")
```

### Keys Inside JIT

```python
import jax
import jax.numpy as jnp

@jax.jit
def random_function(key):
    key1, key2 = jax.random.split(key)
    x = jax.random.normal(key1, (3,))
    y = jax.random.uniform(key2, (3,))
    return x + y

key = jax.random.key(42)
result = random_function(key)
print(f"Result: {result}")

# Same key always produces the same result
result2 = random_function(key)
print(f"Same result: {jnp.allclose(result, result2)}")
```

### Key Data (Advanced)

```python
import jax
import jax.numpy as jnp

key = jax.random.key(42)

# Extract raw key data (uint32 array)
data = jax.random.key_data(key)
print(f"Key data: {data}, dtype: {data.dtype}")

# Create a key from raw data
key_from_data = jax.random.wrap_key_data(data)
print(f"Reconstructed key: {key_from_data}")

# Same key produces same random numbers
x1 = jax.random.normal(key)
x2 = jax.random.normal(key_from_data)
print(f"Same output: {jnp.allclose(x1, x2)}")
```

---

## 3. Continuous Distributions

### uniform

Uniform distribution over `[minval, maxval)`.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# Basic: uniform in [0, 1)
x = jax.random.uniform(key, shape=(5,))
print(f"Uniform [0,1): {x}")

# Custom range: uniform in [a, b)
y = jax.random.uniform(key, shape=(5,), minval=-2.0, maxval=3.0)
print(f"Uniform [-2,3): {y}")

# Multi-dimensional
z = jax.random.uniform(key, shape=(3, 4), minval=0.0, maxval=10.0)
print(f"Uniform (3,4): shape={z.shape}")

# Integer uniform (with minval/maxval as integers)
w = jax.random.uniform(key, shape=(5,), minval=0, maxval=10, dtype=jnp.int32)
print(f"Uniform int: {w}")
```

### normal

Standard normal (Gaussian) distribution with mean 0 and standard deviation 1.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# Standard normal
x = jax.random.normal(key, shape=(5,))
print(f"Normal: {x}")

# Custom shape
x2 = jax.random.normal(key, shape=(3, 100))
print(f"Mean: {jnp.mean(x2):.4f}, Std: {jnp.std(x2):.4f}")

# Scale to custom mean and std
mean, std = 5.0, 2.0
x_scaled = mean + std * jax.random.normal(key, shape=(10000,))
print(f"Scaled mean: {jnp.mean(x_scaled):.2f}, Scaled std: {jnp.std(x_scaled):.2f}")
```

### exponential

Exponential distribution with rate parameter 1.0. PDF: `f(x) = exp(-x)` for `x >= 0`.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# Standard exponential (rate=1)
x = jax.random.exponential(key, shape=(5,))
print(f"Exponential: {x}")

# Exponential with custom rate (lambda)
rate = 2.0
x_scaled = jax.random.exponential(key, shape=(10000,)) / rate
print(f"Mean (should be 1/rate={1/rate}): {jnp.mean(x_scaled):.4f}")
print(f"Std (should be 1/rate={1/rate}): {jnp.std(x_scaled):.4f}")
```

### gamma

Gamma distribution with shape parameter `a`. PDF: `f(x) = x^(a-1) * exp(-x) / Gamma(a)`.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# Gamma with shape parameter a=2.0
x = jax.random.gamma(key, a=2.0, shape=(5,))
print(f"Gamma(a=2): {x}")

# Different shape parameters
for a in [0.5, 1.0, 2.0, 5.0, 10.0]:
    samples = jax.random.gamma(key, a=a, shape=(10000,))
    print(f"Gamma(a={a}): mean={jnp.mean(samples):.3f} (expected {a}), "
          f"std={jnp.std(samples):.3f} (expected {jnp.sqrt(a):.3f})")
```

### beta

Beta distribution with parameters `a` and `b`. Supported on `[0, 1]`.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# Beta distribution
x = jax.random.beta(key, a=2.0, b=5.0, shape=(5,))
print(f"Beta(2,5): {x}")

# Verify mean: a/(a+b)
a, b = 2.0, 5.0
samples = jax.random.beta(key, a=a, b=b, shape=(10000,))
expected_mean = a / (a + b)
print(f"Beta({a},{b}): mean={jnp.mean(samples):.4f} (expected {expected_mean:.4f})")
```

### lognormal

Log-normal distribution. If `X ~ LogNormal`, then `log(X) ~ Normal`.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# Log-normal distribution
x = jax.random.lognormal(key, shape=(5,))
print(f"Lognormal: {x}")

# All values are positive
samples = jax.random.lognormal(key, shape=(10000,))
print(f"Min value: {jnp.min(samples):.4f} (should be positive)")
print(f"Mean: {jnp.mean(samples):.4f}")
print(f"Median: {jnp.median(samples):.4f} (should be ~1.0 for standard lognormal)")
```

### laplace

Laplace distribution with location 0 and scale 1. PDF: `f(x) = exp(-|x|) / 2`.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

x = jax.random.laplace(key, shape=(5,))
print(f"Laplace: {x}")

samples = jax.random.laplace(key, shape=(10000,))
print(f"Mean: {jnp.mean(samples):.4f} (should be ~0)")
print(f"Std: {jnp.std(samples):.4f} (should be ~sqrt(2)={jnp.sqrt(2):.4f})")
```

### logistic

Logistic distribution with location 0 and scale 1.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

x = jax.random.logistic(key, shape=(5,))
print(f"Logistic: {x}")

samples = jax.random.logistic(key, shape=(10000,))
print(f"Mean: {jnp.mean(samples):.4f} (should be ~0)")
print(f"Std: {jnp.std(samples):.4f} (should be ~pi/sqrt(3)={jnp.pi/jnp.sqrt(3):.4f})")
```

### cauchy

Cauchy (Lorentz) distribution. Note: mean and variance are undefined.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

x = jax.random.cauchy(key, shape=(5,))
print(f"Cauchy: {x}")

# Cauchy has heavy tails -- median is well-defined but mean is not
samples = jax.random.cauchy(key, shape=(10000,))
print(f"Median: {jnp.median(samples):.4f} (should be ~0)")
```

### pareto

Pareto distribution with shape parameter `b` (tail index).

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# Pareto with minimum value 1 and shape parameter b
b = 2.0
x = jax.random.pareto(key, b=b, shape=(5,))
print(f"Pareto(b={b}): {x}")

# All values >= 1
samples = jax.random.pareto(key, b=b, shape=(10000,))
print(f"Min value: {jnp.min(samples):.4f} (should be >= 1)")
print(f"Mean: {jnp.mean(samples):.4f} (expected b/(b-1)={b/(b-1):.4f} for b>1)")
```

### t

Student's t-distribution with `df` degrees of freedom.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# t-distribution with 5 degrees of freedom
df = 5.0
x = jax.random.t(key, df=df, shape=(5,))
print(f"t(df={df}): {x}")

# Compare different degrees of freedom
for df in [1.0, 5.0, 30.0, 100.0]:
    samples = jax.random.t(key, df=df, shape=(10000,))
    print(f"t(df={df}): mean={jnp.mean(samples):.4f}, std={jnp.std(samples):.4f}")

# df=1 is Cauchy, df=inf is Normal
```

### gumbel

Gumbel (maximum extreme value) distribution.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

x = jax.random.gumbel(key, shape=(5,))
print(f"Gumbel: {x}")

samples = jax.random.gumbel(key, shape=(10000,))
print(f"Mean: {jnp.mean(samples):.4f} (should be ~{jnp.euler_gamma:.4f})")
print(f"Std: {jnp.std(samples):.4f} (should be ~{jnp.pi/jnp.sqrt(6):.4f})")
```

### triangular

Triangular distribution over `[left, right]` with mode at `mid`.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# Triangular on [0, 1] with mode at 0.5
x = jax.random.triangular(key, left=0.0, right=1.0, mid=0.5, shape=(5,))
print(f"Triangular(0,1,0.5): {x}")

# Asymmetric triangular
y = jax.random.triangular(key, left=0.0, right=1.0, mid=0.2, shape=(10000,))
print(f"Triangular(0,1,0.2) mean: {jnp.mean(y):.4f}")
```

### truncated_normal

Truncated normal distribution, samples from a normal truncated to `[lower, upper]`.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# Normal truncated to [-2, 2]
x = jax.random.truncated_normal(key, lower=-2.0, upper=2.0, shape=(5,))
print(f"Truncated normal [-2, 2]: {x}")

# Verify bounds
samples = jax.random.truncated_normal(key, lower=-1.0, upper=1.0, shape=(10000,))
print(f"Min: {jnp.min(samples):.4f}, Max: {jnp.max(samples):.4f}")
print(f"All in [-1,1]: {jnp.all(samples >= -1.0) and jnp.all(samples <= 1.0)}")
```

### chisquare

Chi-squared distribution with `df` degrees of freedom. Equivalent to `gamma(df/2, 2)`.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

df = 3.0
x = jax.random.chisquare(key, df=df, shape=(5,))
print(f"Chi-squared(df={df}): {x}")

samples = jax.random.chisquare(key, df=df, shape=(10000,))
print(f"Mean: {jnp.mean(samples):.4f} (expected {df})")
print(f"Std: {jnp.std(samples):.4f} (expected {jnp.sqrt(2*df):.4f})")
```

### f

F-distribution with `dfnum` and `dfden` degrees of freedom.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

dfnum, dfden = 5.0, 10.0
x = jax.random.f(key, dfnum=dfnum, dfden=dfden, shape=(5,))
print(f"F({dfnum},{dfden}): {x}")

samples = jax.random.f(key, dfnum=dfnum, dfden=dfden, shape=(10000,))
expected_mean = dfden / (dfden - 2)  # valid for dfden > 2
print(f"Mean: {jnp.mean(samples):.4f} (expected ~{expected_mean:.4f})")
```

### weibull_min

Weibull minimum distribution with shape parameter `scale` (k) and scale 1.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

scale = 1.5  # shape parameter (k)
x = jax.random.weibull_min(key, scale=scale, shape=(5,))
print(f"Weibull_min(scale={scale}): {x}")

# All values positive
samples = jax.random.weibull_min(key, scale=scale, shape=(10000,))
print(f"Min: {jnp.min(samples):.4f} (positive)")
print(f"Mean: {jnp.mean(samples):.4f}")
```

### rademacher

Rademacher distribution: samples are +1 or -1 with equal probability.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

x = jax.random.rademacher(key, shape=(10,))
print(f"Rademacher: {x}")  # all values are +1 or -1
```

### maxwell

Maxwell-Boltzmann distribution. Used in physics for particle speeds.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

x = jax.random.maxwell(key, shape=(5,))
print(f"Maxwell: {x}")

# All values positive
samples = jax.random.maxwell(key, shape=(10000,))
print(f"All positive: {jnp.all(samples > 0)}")
print(f"Mean: {jnp.mean(samples):.4f}")
```

### double_sided_maxwell

Double-sided Maxwell distribution.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

x = jax.random.double_sided_maxwell(key, shape=(5,))
print(f"Double-sided Maxwell: {x}")

# Can be negative
samples = jax.random.double_sided_maxwell(key, shape=(10000,))
print(f"Mean: {jnp.mean(samples):.4f} (should be ~0)")
```

### generalized_normal

Generalized normal (exponential power) distribution with shape parameter `p`.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# p=2 is Gaussian, p=1 is Laplace
for p in [0.5, 1.0, 2.0, 5.0]:
    x = jax.random.generalized_normal(key, p=p, shape=(5,))
    print(f"Generalized normal (p={p}): {x}")
```

### rayleigh

Rayleigh distribution with scale parameter `sigma`.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

sigma = 1.0
x = jax.random.rayleigh(key, shape=(5,))
print(f"Rayleigh: {x}")

samples = jax.random.rayleigh(key, shape=(10000,))
print(f"All positive: {jnp.all(samples > 0)}")
print(f"Mean: {jnp.mean(samples):.4f}")
```

### wald

Wald (inverse Gaussian) distribution with mean `mean` and shape parameter `scale`.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

mean, scale = 1.0, 1.0
x = jax.random.wald(key, mean=mean, scale=scale, shape=(5,))
print(f"Wald: {x}")

samples = jax.random.wald(key, mean=mean, scale=scale, shape=(10000,))
print(f"Mean: {jnp.mean(samples):.4f} (expected ~{mean})")
```

### loggamma

Log-gamma distribution: log of a gamma random variable.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

a = 2.0
x = jax.random.loggamma(key, a=a, shape=(5,))
print(f"Loggamma(a={a}): {x}")

# Compare: loggamma should equal log(gamma)
key1, key2 = jax.random.split(key)
gamma_samples = jax.random.gamma(key1, a=a, shape=(5,))
loggamma_samples = jax.random.loggamma(key2, a=a, shape=(5,))
# Note: they use different keys, so values differ, but distribution should match
```

### dirichlet

Dirichlet distribution. Returns samples on the simplex (all positive, sum to 1).

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# Dirichlet with concentration parameters
alpha = jnp.array([2.0, 5.0, 3.0])
x = jax.random.dirichlet(key, alpha=alpha, shape=(3,))
print(f"Dirichlet samples:\n{x}")
print(f"Row sums: {jnp.sum(x, axis=-1)}")  # all ~1.0

# Batched
alpha = jnp.array([1.0, 1.0, 1.0, 1.0])
x = jax.random.dirichlet(key, alpha=alpha, shape=(10000,))
print(f"Mean per component: {jnp.mean(x, axis=0)}")
print(f"Expected: {alpha / jnp.sum(alpha)}")
```

### multivariate_normal

Multivariate normal (Gaussian) distribution with given mean and covariance.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# 2D multivariate normal
mean = jnp.array([1.0, 2.0])
cov = jnp.array([[1.0, 0.5],
                  [0.5, 2.0]])
x = jax.random.multivariate_normal(key, mean=mean, cov=cov, shape=(5,))
print(f"Multivariate normal:\n{x}")

# Diagonal covariance (use for efficiency)
mean = jnp.zeros(10)
cov = jnp.eye(10)
x = jax.random.multivariate_normal(key, mean=mean, cov=cov, shape=(1000,))
print(f"Sample mean: {jnp.mean(x, axis=0)}")
print(f"Sample cov diagonal: {jnp.var(x, axis=0)}")
```

---

## 4. Discrete Distributions

### bernoulli

Bernoulli distribution: returns True/False (or 1/0) with probability `p`.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# Fair coin flip
x = jax.random.bernoulli(key, p=0.5, shape=(10,))
print(f"Bernoulli(0.5): {x}")

# Biased coin
y = jax.random.bernoulli(key, p=0.9, shape=(10,))
print(f"Bernoulli(0.9): {y}")

# Verify probability
samples = jax.random.bernoulli(key, p=0.3, shape=(10000,))
print(f"P(True) = {jnp.mean(samples):.4f} (expected 0.3)")

# Per-element probabilities
probs = jnp.array([0.1, 0.5, 0.9])
z = jax.random.bernoulli(key, p=probs)
print(f"Per-element: {z}")
```

### randint

Uniform random integers in `[minval, maxval)`.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# Random integers in [0, 10)
x = jax.random.randint(key, shape=(5,), minval=0, maxval=10)
print(f"Randint [0,10): {x}")

# Negative range
y = jax.random.randint(key, shape=(5,), minval=-5, maxval=5)
print(f"Randint [-5,5): {y}")

# Specific dtype
z = jax.random.randint(key, shape=(5,), minval=0, maxval=100, dtype=jnp.int64)
print(f"Randint int64: {z}")
```

### binomial

Binomial distribution: number of successes in `n` independent Bernoulli trials.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# Binomial(n=10, p=0.5)
n, p = 10, 0.5
x = jax.random.binomial(key, n=n, p=p, shape=(5,))
print(f"Binomial(n={n}, p={p}): {x}")

# Verify mean = n*p
samples = jax.random.binomial(key, n=n, p=p, shape=(10000,))
print(f"Mean: {jnp.mean(samples):.4f} (expected {n*p})")
```

### geometric

Geometric distribution: number of trials until first success.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

p = 0.3
x = jax.random.geometric(key, p=p, shape=(10,))
print(f"Geometric(p={p}): {x}")  # values >= 1

samples = jax.random.geometric(key, p=p, shape=(10000,))
print(f"Mean: {jnp.mean(samples):.4f} (expected {1/p:.4f})")
```

### poisson

Poisson distribution with rate parameter `lam`.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

lam = 5.0
x = jax.random.poisson(key, lam=lam, shape=(10,))
print(f"Poisson(lam={lam}): {x}")

# Per-element lambda values
lams = jnp.array([1.0, 5.0, 10.0])
y = jax.random.poisson(key, lam=lams, shape=(5,))
print(f"Poisson (per-element):\n{y}")

# Verify mean
samples = jax.random.poisson(key, lam=lam, shape=(10000,))
print(f"Mean: {jnp.mean(samples):.4f} (expected {lam})")
print(f"Var: {jnp.var(samples):.4f} (expected {lam})")
```

### multinomial

Multinomial distribution: counts of each category in `n` trials.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# Roll a fair die 100 times
n = 100
probs = jnp.array([1/6] * 6)
counts = jax.random.multinomial(key, n=n, p=probs, shape=(1,))
print(f"Die roll counts: {counts}")
print(f"Sum: {jnp.sum(counts)}")  # should be n=100

# Biased multinomial
probs = jnp.array([0.1, 0.2, 0.3, 0.4])
counts = jax.random.multinomial(key, n=1000, p=probs, shape=(5,))
print(f"Biased multinomial:\n{counts}")
print(f"Means: {jnp.mean(counts, axis=0)}")
print(f"Expected: {1000 * probs}")
```

### categorical

Categorical distribution: sample an index from a probability distribution.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# Sample from categories with given probabilities
probs = jnp.array([0.1, 0.3, 0.4, 0.2])
indices = jax.random.categorical(key, logits=jnp.log(probs), shape=(10,))
print(f"Categorical indices: {indices}")

# Using logits directly (more numerically stable)
logits = jnp.array([1.0, 3.0, 4.0, 2.0])
indices = jax.random.categorical(key, logits=logits, shape=(10,))
print(f"Categorical (logits): {indices}")

# Verify distribution
indices = jax.random.categorical(key, logits=jnp.log(probs), shape=(10000,))
for i in range(4):
    count = jnp.sum(indices == i)
    print(f"Category {i}: {count/10000:.4f} (expected {probs[i]:.4f})")

# 2D logits (e.g., sequence of distributions)
logits_2d = jax.random.normal(key, (5, 10))  # 5 sequences, 10 categories
indices_2d = jax.random.categorical(key, logits=logits_2d, shape=(3, 5))
print(f"2D categorical shape: {indices_2d.shape}")  # (3, 5)
```

---

## 5. Multivariate Distributions

### multivariate_normal (Detailed)

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# Full covariance matrix
mean = jnp.array([0.0, 0.0])
cov = jnp.array([[2.0, 0.8],
                  [0.8, 1.0]])

samples = jax.random.multivariate_normal(key, mean=mean, cov=cov, shape=(1000,))
print(f"Shape: {samples.shape}")  # (1000, 2)
print(f"Sample mean: {jnp.mean(samples, axis=0)}")
print(f"Sample cov:\n{jnp.cov(samples.T)}")

# Batch of different means
means = jnp.array([[0.0, 0.0], [5.0, 5.0]])  # 2 different means
cov = jnp.eye(2)
samples = jax.random.multivariate_normal(key, mean=means, cov=cov)
print(f"Batched shape: {samples.shape}")
```

### dirichlet (Detailed)

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# Symmetric Dirichlet (uniform on simplex when alpha=1)
alpha = jnp.ones(5)
x = jax.random.dirichlet(key, alpha=alpha, shape=(3,))
print(f"Symmetric Dirichlet:\n{x}")
print(f"Sums: {jnp.sum(x, axis=-1)}")

# Sparse Dirichlet (alpha < 1)
alpha_sparse = jnp.array([0.1, 0.1, 0.1, 0.1])
x_sparse = jax.random.dirichlet(key, alpha=alpha_sparse, shape=(3,))
print(f"Sparse Dirichlet:\n{x_sparse}")

# Dense Dirichlet (alpha >> 1)
alpha_dense = jnp.array([10.0, 10.0, 10.0, 10.0])
x_dense = jax.random.dirichlet(key, alpha=alpha_dense, shape=(3,))
print(f"Dense Dirichlet:\n{x_dense}")
```

### orthogonal

Random orthogonal matrix (from the Haar measure on the orthogonal group).

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# 3x3 random orthogonal matrix
Q = jax.random.orthogonal(key, n=3)
print(f"Orthogonal matrix:\n{Q}")
print(f"Q^T Q:\n{Q @ Q.T}")  # should be identity
print(f"det(Q): {jnp.linalg.det(Q):.4f}")  # should be +/- 1
```

### ball

Uniform random samples from an n-dimensional unit ball.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# Random points in a 3D unit ball
points = jax.random.ball(key, d=3, shape=(5,))
print(f"Ball points:\n{points}")
print(f"Norms: {jnp.linalg.norm(points, axis=-1)}")
print(f"All inside unit ball: {jnp.all(jnp.linalg.norm(points, axis=-1) <= 1.0)}")
```

---

## 6. Sampling Utilities

### permutation

Random permutation of integers `[0, n)` or of an array's elements.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# Permutation of [0, n)
perm = jax.random.permutation(key, 10)
print(f"Permutation of 0-9: {perm}")

# Permutation of an array
arr = jnp.array([10, 20, 30, 40, 50])
perm_arr = jax.random.permutation(key, arr)
print(f"Permuted array: {perm_arr}")

# Independent: permute along an axis
matrix = jnp.arange(12).reshape(3, 4)
perm_rows = jax.random.permutation(key, matrix, axis=0)
print(f"Permuted rows:\n{perm_rows}")

perm_cols = jax.random.permutation(key, matrix, axis=1)
print(f"Permuted cols:\n{perm_cols}")
```

### choice

Random sampling with or without replacement.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# Choose 3 elements from [0, 10) without replacement
chosen = jax.random.choice(key, 10, shape=(3,), replace=False)
print(f"Choice (no replace): {chosen}")

# Choose with replacement
chosen_replace = jax.random.choice(key, 10, shape=(5,), replace=True)
print(f"Choice (replace): {chosen_replace}")

# Choose from an array
arr = jnp.array([10, 20, 30, 40, 50])
chosen_arr = jax.random.choice(key, arr, shape=(3,), replace=False)
print(f"Choice from array: {chosen_arr}")

# Weighted choice (using p parameter)
p = jnp.array([0.4, 0.1, 0.1, 0.1, 0.3])
chosen_weighted = jax.random.choice(key, 5, shape=(10,), replace=True, p=p)
print(f"Weighted choices: {chosen_weighted}")
```

### bits

Random bits (uniformly distributed 0s and 1s at the bit level).

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# Random bits as uint32
b = jax.random.bits(key, shape=(5,), dtype=jnp.uint32)
print(f"Random bits (uint32): {b}")

# Random bits as uint8
b8 = jax.random.bits(key, shape=(10,), dtype=jnp.uint8)
print(f"Random bits (uint8): {b8}")

# Large shape
b_large = jax.random.bits(key, shape=(3, 4), dtype=jnp.uint32)
print(f"Random bits (3,4):\n{b_large}")
```

---

## 7. PRNG Algorithms

JAX supports multiple PRNG algorithms. The algorithm determines the statistical quality and performance characteristics of the random numbers.

### Available Algorithms

```python
import jax
import jax.numpy as jnp

# Default algorithm (threefry2x32)
key = jax.random.key(42)
print(f"Default key impl: {jax.random.key_impl(key)}")

# Explicit algorithm selection
key_threefry = jax.random.key(42, impl="threefry2x32")
print(f"Threefry key: {jax.random.key_impl(key_threefry)}")

key_rbg = jax.random.key(42, impl="rbg")
print(f"RBG key: {jax.random.key_impl(key_rbg)}")

key_unsafe = jax.random.key(42, impl="unsafe_rbg")
print(f"Unsafe RBG key: {jax.random.key_impl(key_unsafe)}")
```

### threefry2x32 (Default)

The default PRNG algorithm in JAX. Based on the Threefish hash function.

- **Quality**: Cryptographically inspired, passes standard statistical tests.
- **Performance**: Good balance of quality and speed.
- **Reproducibility**: Fully deterministic across platforms and devices.

```python
import jax
import jax.numpy as jnp

# Threefry is the default
key = jax.random.key(42, impl="threefry2x32")

key1, key2 = jax.random.split(key)
x = jax.random.normal(key1, (3,))
print(f"Threefry normal: {x}")

# Always produces the same sequence given the same key
key_repeat = jax.random.key(42, impl="threefry2x32")
key1r, _ = jax.random.split(key_repeat)
x_repeat = jax.random.normal(key1r, (3,))
print(f"Reproducible: {jnp.allclose(x, x_repeat)}")
```

### rbg

Random Bit Generator -- a faster algorithm optimized for GPU/TPU.

- **Quality**: Good statistical quality but not as thoroughly analyzed as threefry.
- **Performance**: Faster on accelerators, especially for large arrays.
- **Use case**: Recommended for ML training workloads where speed matters.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(42, impl="rbg")

key1, key2 = jax.random.split(key)
x = jax.random.normal(key1, (1000,))
print(f"RBG normal mean: {jnp.mean(x):.4f}, std: {jnp.std(x):.4f}")
```

### unsafe_rbg

A faster but less safe variant of RBG.

- **Quality**: May have lower statistical quality.
- **Performance**: Fastest option.
- **Use case**: Only use when you understand the tradeoffs and need maximum speed.

```python
import jax
import jax.numpy as jnp

key = jax.random.key(42, impl="unsafe_rbg")

key1, key2 = jax.random.split(key)
x = jax.random.normal(key1, (1000,))
print(f"Unsafe RBG normal mean: {jnp.mean(x):.4f}, std: {jnp.std(x):.4f}")
```

### Setting the Default Algorithm

```python
import jax

# Set globally via config
jax.config.update("jax_default_prng_impl", "rbg")

# Now all keys use RBG by default
key = jax.random.key(42)
print(f"Global default: {jax.random.key_impl(key)}")

# Reset to threefry
jax.config.update("jax_default_prng_impl", "threefry2x32")
```

---

## 8. Key Types and key_impl

### key_impl

`jax.random.key_impl` returns the implementation name of a key.

```python
import jax
import jax

for impl in ["threefry2x32", "rbg", "unsafe_rbg"]:
    key = jax.random.key(42, impl=impl)
    print(f"Impl '{impl}': key_impl={jax.random.key_impl(key)}, dtype={key.dtype}")
```

### Key Shapes and dtypes

```python
import jax
import jax.numpy as jnp

key = jax.random.key(42)
print(f"Key shape: {key.shape}")
print(f"Key dtype: {key.dtype}")
print(f"Key ndim: {key.ndim}")

# Batched keys
keys = jax.random.split(key, 5)
print(f"Batched keys shape: {keys.shape}")

# Key data extraction
data = jax.random.key_data(key)
print(f"Key data shape: {data.shape}, dtype: {data.dtype}")

# Reconstruct key from data
reconstructed = jax.random.wrap_key_data(data)
print(f"Reconstructed: {jax.random.key_data(reconstructed)}")
```

### Batched Key Operations

```python
import jax
import jax.numpy as jnp

key = jax.random.key(42)

# Create batched keys
keys = jax.random.split(key, 10)
print(f"Keys shape: {keys.shape}")  # (10,)

# vmap over keys for independent random streams
results = jax.vmap(lambda k: jax.random.normal(k, (3,)))(keys)
print(f"Batched results shape: {results.shape}")  # (10, 3)

# Each row is independently generated
print(f"First two rows different: {not jnp.allclose(results[0], results[1])}")
```

### Key Serialization

```python
import jax
import jax.numpy as jnp
import numpy as np

key = jax.random.key(42)

# Serialize key data
data = jax.random.key_data(key)
np_data = np.array(data)  # convert to numpy for serialization

# Deserialize
restored_data = jnp.array(np_data)
restored_key = jax.random.wrap_key_data(restored_data)

# Verify
x1 = jax.random.normal(key)
x2 = jax.random.normal(restored_key)
print(f"Same after serialize/deserialize: {jnp.allclose(x1, x2)}")
```

---

## 9. Best Practices

### Rule 1: Never Reuse a Key for Different Purposes

```python
import jax
import jax.numpy as jnp

# BAD: reusing the same key
key = jax.random.key(42)
x = jax.random.normal(key, (3,))
y = jax.random.normal(key, (3,))  # Same as x! Not independent.

# GOOD: split before each use
key = jax.random.key(42)
key, subkey1 = jax.random.split(key)
key, subkey2 = jax.random.split(key)
x = jax.random.normal(subkey1, (3,))
y = jax.random.normal(subkey2, (3,))
```

### Rule 2: Split at the Beginning of Functions

```python
import jax
import jax.numpy as jnp

def model_init(key, input_dim, hidden_dim, output_dim):
    """Initialize a 2-layer MLP."""
    key1, key2 = jax.random.split(key)
    w1 = jax.random.normal(key1, (input_dim, hidden_dim)) * 0.01
    w2 = jax.random.normal(key2, (hidden_dim, output_dim)) * 0.01
    b1 = jnp.zeros(hidden_dim)
    b2 = jnp.zeros(output_dim)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

key = jax.random.key(42)
params = model_init(key, 784, 256, 10)
print(f"w1 shape: {params['w1'].shape}")
```

### Rule 3: Use fold_in for Deterministic Iteration Keys

```python
import jax
import jax.numpy as jnp

@jax.jit
def training_step(key, params, x, y, step):
    """Each step gets a deterministic key via fold_in."""
    step_key = jax.random.fold_in(key, step)

    # Use step_key for dropout, data augmentation, etc.
    dropout_key, noise_key = jax.random.split(step_key)
    dropout_mask = jax.random.bernoulli(dropout_key, 0.9, x.shape)
    noise = jax.random.normal(noise_key, x.shape) * 0.01

    x_noisy = x * dropout_mask + noise
    loss = jnp.mean((x_noisy @ params["w"] - y) ** 2)
    return loss

key = jax.random.key(42)
params = {"w": jax.random.normal(key, (4, 2)) * 0.01}
x = jax.random.normal(jax.random.split(key)[0], (10, 4))
y = jax.random.normal(jax.random.split(key)[1], (10, 2))

for step in range(5):
    loss = training_step(key, params, x, y, step)
    print(f"Step {step}: loss = {loss:.4f}")
```

### Rule 4: Use vmap for Batched Random Operations

```python
import jax
import jax.numpy as jnp

def generate_samples(key, n_samples):
    return jax.random.normal(key, (n_samples,))

# Generate independent samples for each batch element
keys = jax.random.split(jax.random.key(0), 4)
samples = jax.vmap(generate_samples, in_axes=(0, None))(keys, 100)
print(f"Samples shape: {samples.shape}")  # (4, 100)
```

### Rule 5: Dropout with Explicit Keys

```python
import jax
import jax.numpy as jnp

def dropout(x, key, rate=0.5):
    """Apply dropout with explicit key."""
    keep_mask = jax.random.bernoulli(key, p=1 - rate, shape=x.shape)
    return jnp.where(keep_mask, x / (1 - rate), 0.0)

key = jax.random.key(42)
x = jnp.ones((5, 10))

key, drop_key = jax.random.split(key)
x_dropped = dropout(x, drop_key, rate=0.5)
print(f"After dropout: {x_dropped}")
print(f"Fraction kept: {jnp.mean(x_dropped != 0.0):.4f}")
```

### Complete Training Loop Example

```python
import jax
import jax.numpy as jnp

def init_params(key, layer_sizes):
    """Initialize MLP parameters."""
    params = []
    for i, (fan_in, fan_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        key, wkey, bkey = jax.random.split(key, 3)
        w = jax.random.normal(wkey, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in)
        b = jnp.zeros(fan_out)
        params.append({"w": w, "b": b})
    return params

def forward(params, x, key=None, dropout_rate=0.0):
    """Forward pass with optional dropout."""
    for i, layer in enumerate(params[:-1]):
        x = x @ layer["w"] + layer["b"]
        x = jnp.maximum(0, x)  # ReLU
        if key is not None and dropout_rate > 0:
            key, subkey = jax.random.split(key)
            mask = jax.random.bernoulli(subkey, 1 - dropout_rate, x.shape)
            x = jnp.where(mask, x / (1 - dropout_rate), 0.0)
    x = x @ params[-1]["w"] + params[-1]["b"]
    return x

def loss_fn(params, x, y, key=None):
    preds = forward(params, x, key=key, dropout_rate=0.1)
    return jnp.mean((preds - y) ** 2)

# Setup
key = jax.random.key(42)
key, init_key, data_key = jax.random.split(key, 3)

layer_sizes = [4, 32, 16, 2]
params = init_params(init_key, layer_sizes)

x = jax.random.normal(data_key, (50, 4))
y = jax.random.normal(jax.random.split(data_key)[0], (50, 2))

# Training
lr = 0.01
for step in range(100):
    key, train_key = jax.random.split(key)
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y, key=train_key)
    params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    if step % 20 == 0:
        print(f"Step {step}: loss = {loss:.6f}")

print("Training complete")
```

### Distribution Quick Reference

| Function | Parameters | Output Range | Notes |
|----------|-----------|--------------|-------|
| `uniform` | `minval, maxval` | `[minval, maxval)` | Continuous or discrete |
| `normal` | - | `(-inf, inf)` | Mean 0, std 1 |
| `bernoulli` | `p` | `{False, True}` | Boolean output |
| `randint` | `minval, maxval` | `[minval, maxval)` | Integer output |
| `exponential` | - | `[0, inf)` | Rate 1 |
| `poisson` | `lam` | `{0, 1, 2, ...}` | Integer output |
| `gamma` | `a` | `[0, inf)` | Shape parameter |
| `beta` | `a, b` | `[0, 1]` | On simplex |
| `dirichlet` | `alpha` | Simplex | Vector output, sums to 1 |
| `gumbel` | - | `(-inf, inf)` | Max extreme value |
| `laplace` | - | `(-inf, inf)` | Double exponential |
| `logistic` | - | `(-inf, inf)` | Sigmoid inverse |
| `lognormal` | - | `(0, inf)` | Log is normal |
| `pareto` | `b` | `[1, inf)` | Heavy-tailed |
| `t` | `df` | `(-inf, inf)` | Student's t |
| `triangular` | `left, right, mid` | `[left, right]` | Triangle PDF |
| `multivariate_normal` | `mean, cov` | `R^n` | Gaussian vector |
| `categorical` | `logits` | `{0, ..., K-1}` | Integer index |
| `permutation` | `x` or `n` | Permutation | Shuffled indices or array |
| `choice` | `a, shape` | Elements of `a` | With/without replacement |
| `binomial` | `n, p` | `{0, ..., n}` | Integer output |
| `multinomial` | `n, p` | Count vector | Sums to `n` |
| `truncated_normal` | `lower, upper` | `[lower, upper]` | Clipped normal |
| `cauchy` | - | `(-inf, inf)` | Heavy-tailed, no mean |
| `chisquare` | `df` | `[0, inf)` | Sum of squared normals |
| `f` | `dfnum, dfden` | `[0, inf)` | F-test distribution |
| `rademacher` | - | `{-1, +1}` | Symmetric Bernoulli |
| `maxwell` | - | `(0, inf)` | Speed distribution |
| `double_sided_maxwell` | - | `(-inf, inf)` | Symmetric |
| `weibull_min` | `scale` | `(0, inf)` | Reliability analysis |
| `orthogonal` | `n` | `O(n)` | Random orthogonal matrix |
| `generalized_normal` | `p` | `(-inf, inf)` | p=2 is Gaussian |
| `ball` | `d` | `R^d` | Uniform in unit ball |
| `rayleigh` | - | `(0, inf)` | Signal processing |
| `wald` | `mean, scale` | `(0, inf)` | Inverse Gaussian |
| `geometric` | `p` | `{1, 2, ...}` | Trials until success |
| `loggamma` | `a` | `(-inf, inf)` | Log of gamma |
| `bits` | - | uint32/uint8 | Raw random bits |
