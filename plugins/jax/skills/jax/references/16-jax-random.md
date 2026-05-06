# 16 - jax.random Deep Dive

## Overview

This chapter provides an in-depth reference for JAX's random number system, covering the internal architecture, all 38+ distribution functions, and advanced usage patterns.

---

## 1. PRNG Architecture

### Key array structure

```python
import jax.random as random
import jax.numpy as jnp

# A PRNG key is a special array type
key = random.key(42)
# key is a PRNGKeyArray with shape () and dtype key<fry>

# Keys can be batched
keys = random.split(key, 5)  # shape: (5,), dtype: key<fry>

# Keys are immutable
key[0] = 5  # Error!
```

### Three-layer architecture

```
jax.random (high-level API)
    │
    ▼
jax._src.prng (PRNG implementations)
    │
    ▼
threefry2x32 / rbg / unsafe_rbg (algorithms)
```

### PRNG implementations

| Implementation | Algorithm | Key size | Quality | Speed |
|---|---|---|---|---|
| `threefry2x32` (default) | Threefish-based | 2×uint32 | High | Baseline |
| `rbg` | AES-based | 4×uint32 | High | ~2× faster |
| `unsafe_rbg` | AES-based | 4×uint32 | High | ~2.5× faster, non-portable |

```python
# Select implementation globally
jax.config.update("jax_default_prng_impl", "rbg")

# Or per-key
key = random.key(42, impl="rbg")
```

---

## 2. Complete Distribution Reference

### Uniform distributions

```python
# Uniform [minval, maxval)
random.uniform(key, shape=(100,), minval=0.0, maxval=1.0, dtype=jnp.float32)

# Uniform integers
random.randint(key, shape=(10,), minval=0, maxval=100, dtype=jnp.int32)
# Note: maxval is exclusive

# Bernoulli
random.bernoulli(key, p=0.7, shape=(100,))  # bool array

# Ball distribution (uniform in unit ball)
random.ball(key, d=3, shape=(100,))  # shape: (100, 3), vectors in unit ball

# Spherical distribution (uniform on unit sphere)
random.spherical(key, shape=(100, 3))  # shape: (100, 3), unit vectors
```

### Normal and related distributions

```python
# Standard normal N(0, 1)
random.normal(key, shape=(100, 50), dtype=jnp.float32)

# Log-normal
random.lognormal(key, shape=(100,))

# Truncated normal
random.truncated_normal(key, lower=-2.0, upper=2.0, shape=(100,))

# Multivariate normal
mean = jnp.zeros(3)
cov = jnp.eye(3)
random.multivariate_normal(key, mean, cov, shape=(100,))  # (100, 3)
```

### Exponential family

```python
# Exponential
random.exponential(key, shape=(100,))

# Gamma
random.gamma(key, a=2.0, shape=(100,))

# Beta
random.beta(key, a=2.0, b=5.0, shape=(100,))

# Dirichlet (simplex)
random.dirichlet(key, alpha=jnp.array([1.0, 2.0, 3.0]), shape=(100,))  # (100, 3)

# Chi-squared
random.chi2(key, df=5.0, shape=(100,))

# Student's t
random.t(key, df=10.0, shape=(100,))

# F-distribution (via chi2)
def f_dist(key, d1, d2, shape=()):
    k1, k2 = random.split(key)
    x1 = random.chi2(k1, d1, shape) / d1
    x2 = random.chi2(k2, d2, shape) / d2
    return x1 / x2
```

### Heavy-tailed distributions

```python
# Cauchy (no finite moments)
random.cauchy(key, shape=(100,))

# Laplace
random.laplace(key, shape=(100,))

# Logistic
random.logistic(key, shape=(100,))

# Pareto
random.pareto(key, b=2.0, shape=(100,))
```

### Extreme value distributions

```python
# Gumbel
random.gumbel(key, shape=(100,))

# Weibull (minimum)
random.weibull_min(key, scale=1.0, concentration=1.5, shape=(100,))
```

### Discrete distributions

```python
# Categorical (from logits or probs)
logits = jnp.log(jnp.array([0.1, 0.3, 0.4, 0.2]))
random.categorical(key, logits, shape=(1000,))  # indices in {0,1,2,3}

# Multinomial
random.multinomial(key, n=10, p=jnp.array([0.2, 0.3, 0.5]))

# Geometric
random.geometric(key, p=0.3, shape=(100,))

# Binomial
random.binomial(key, n=20, p=0.5, shape=(100,))

# Poisson
random.poisson(key, lam=5.0, shape=(100,))

# Rademacher (±1)
random.rademacher(key, shape=(100,), dtype=jnp.int32)
```

### Matrix distributions

```python
# Random orthogonal matrix (Haar measure)
Q = random.orthogonal(key, n=10)  # (10, 10) orthogonal matrix

# Wishart (via chi2 + normal)
def wishart(key, df, scale, shape=()):
    p = scale.shape[0]
    k1, k2 = random.split(key)
    A = random.normal(k1, shape=shape + (df, p)) @ jnp.linalg.cholesky(scale)
    return jnp.einsum('...ij,...kj->...ik', A, A)
```

---

## 3. Key Manipulation API

```python
# Create key
key = random.key(seed)
key = random.PRNGKey(seed)  # Legacy

# Split
keys = random.split(key, num=2)       # 2 keys
keys = random.split(key, num=100)     # 100 keys

# Fold in (combine key with data)
new_key = random.fold_in(key, data)

# Key array operations
key.shape   # Key shape
key.dtype   # Key dtype (e.g., key<fry>)

# Convert to raw uint32
raw = jax.random.key_data(key)  # uint32 array

# Create key from raw data
key = jax.random.key_data(raw)  # Inverse
```

---

## 4. Advanced Patterns

### Deterministic key sequences

```python
def key_sequence(base_key, n):
    """Generate n deterministic keys from a base key."""
    return random.split(base_key, n)

# Or using fold_in for indexed access
def key_at(base_key, index):
    """Get key at a specific index without generating all keys."""
    return random.fold_in(base_key, index)
```

### Stratified sampling

```python
def stratified_uniform(key, n_bins, n_per_bin):
    """Stratified uniform samples."""
    keys = random.split(key, n_bins)
    bins = jnp.arange(n_bins, dtype=jnp.float32) / n_bins
    samples = []
    for i, k in enumerate(keys):
        u = random.uniform(k, shape=(n_per_bin,),
                          minval=bins[i], maxval=bins[i] + 1.0 / n_bins)
        samples.append(u)
    return jnp.concatenate(samples)
```

### Rejection sampling

```python
def rejection_sample(key, target_log_prob, proposal_log_prob,
                     proposal_sample, n_samples, n_max=10000):
    """Generic rejection sampling."""
    def body_fn(state):
        key, samples, n_accepted = state
        k1, k2 = random.split(key)
        proposal = proposal_sample(k1)
        u = random.uniform(k2)
        log_accept = target_log_prob(proposal) - proposal_log_prob(proposal)
        accept = jnp.log(u) < log_accept
        if accept:
            samples = samples.at[n_accepted].set(proposal)
            n_accepted += 1
        return k2, samples, n_accepted

    def cond_fn(state):
        _, _, n_accepted = state
        return n_accepted < n_samples

    samples = jnp.zeros((n_samples,))
    _, samples, _ = jax.lax.while_loop(cond_fn, body_fn,
                                        (key, samples, 0))
    return samples
```

### Gumbel-max trick for differentiable sampling

```python
def gumbel_softmax_sample(key, logits, temperature=1.0):
    """Differentiable categorical sampling via Gumbel-softmax."""
    gumbels = random.gumbel(key, logits.shape)
    y = jnp.exp((logits + gumbels) / temperature)
    return y / y.sum(axis=-1, keepdims=True)
```

---

## 5. Random with Sharding

```python
# Create sharded keys for multi-device
from jax.sharding import Mesh, PartitionSpec as P

mesh = Mesh(jax.devices(), ('devices',))

@jax.jit
def sharded_random(key):
    keys = random.split(key, jax.device_count())
    samples = jax.pmap(lambda k: random.normal(k, (1000,)))(keys)
    return samples
```

---

## 6. Complete Function List

| Function | Signature |
|---|---|
| `key` | `(seed, *, impl=None)` |
| `split` | `(key, num=2)` |
| `fold_in` | `(key, data)` |
| `uniform` | `(key, shape, dtype, minval, maxval)` |
| `normal` | `(key, shape, dtype)` |
| `truncated_normal` | `(key, lower, upper, shape, dtype)` |
| `lognormal` | `(key, sigma, shape, dtype)` |
| `exponential` | `(key, shape, dtype)` |
| `gamma` | `(key, a, shape, dtype)` |
| `beta` | `(key, a, b, shape, dtype)` |
| `chi2` | `(key, df, shape, dtype)` |
| `t` | `(key, df, shape, dtype)` |
| `laplace` | `(key, shape, dtype)` |
| `cauchy` | `(key, shape, dtype)` |
| `logistic` | `(key, shape, dtype)` |
| `pareto` | `(key, b, shape, dtype)` |
| `gumbel` | `(key, shape, dtype)` |
| `weibull_min` | `(key, scale, concentration, shape, dtype)` |
| `maxwell` | `(key, shape, dtype)` |
| `double_sided_maxwell` | `(key, loc, scale, shape, dtype)` |
| `poisson` | `(key, lam, shape, dtype)` |
| `generalized_normal` | `(key, p, shape, dtype)` |
| `bernoulli` | `(key, p, shape)` |
| `binomial` | `(key, n, p, shape, dtype)` |
| `multinomial` | `(key, n, p, shape, dtype)` |
| `categorical` | `(key, logits, axis, shape)` |
| `geometric` | `(key, p, shape, dtype)` |
| `randint` | `(key, shape, minval, maxval, dtype)` |
| `choice` | `(key, a, shape, replace, p, axis)` |
| `permutation` | `(key, x, axis, independent)` |
| `ball` | `(key, d, shape, dtype)` |
| `spherical` | `(key, shape, dtype)` |
| `orthogonal` | `(key, n, shape, dtype)` |
| `dirichlet` | `(key, alpha, shape, dtype)` |
| `multivariate_normal` | `(key, mean, cov, shape, dtype, method)` |
| `rademacher` | `(key, shape, dtype)` |
