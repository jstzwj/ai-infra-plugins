# JAX Neural Network Module (jax.nn)

## Table of Contents

- [1. Overview](#1-overview)
- [2. Activation Functions](#2-activation-functions)
  - [2.1 relu](#21-relu)
  - [2.2 relu6](#22-relu6)
  - [2.3 sigmoid](#23-sigmoid)
  - [2.4 softplus](#24-softplus)
  - [2.5 soft_sign](#25-soft_sign)
  - [2.6 silu (swish)](#26-silu-swish)
  - [2.7 log_sigmoid](#27-log_sigmoid)
  - [2.8 elu](#28-elu)
  - [2.9 selu](#29-selu)
  - [2.10 gelu](#210-gelu)
  - [2.11 glu](#211-glu)
  - [2.12 leaky_relu](#212-leaky_relu)
  - [2.13 hard_tanh](#213-hard_tanh)
  - [2.14 hard_sigmoid](#214-hard_sigmoid)
  - [2.15 hard_silu / hard_swish](#215-hard_silu--hard_swish)
  - [2.16 log_softmax](#216-log_softmax)
  - [2.17 softmax](#217-softmax)
  - [2.18 standardize](#218-standardize)
  - [2.19 one_hot](#219-one_hot)
  - [2.20 tanh](#220-tanh)
  - [2.21 exponential](#221-exponential)
  - [2.22 mish](#222-mish)
  - [2.23 celu](#223-celu)
  - [2.24 approximate_gelu](#224-approximate_gelu)
- [3. Initializers (jax.nn.initializers)](#3-initializers-jaxnninitializers)
  - [3.1 Overview and Key Concepts](#31-overview-and-key-concepts)
  - [3.2 variance_scaling (Base Class)](#32-variance_scaling-base-class)
  - [3.3 uniform](#33-uniform)
  - [3.4 normal](#34-normal)
  - [3.5 he_uniform / he_normal](#35-he_uniform--he_normal)
  - [3.6 glorot_uniform / glorot_normal (Xavier)](#36-glorot_uniform--glorot_normal-xavier)
  - [3.7 lecun_uniform / lecun_normal](#37-lecun_uniform--lecun_normal)
  - [3.8 zeros / ones / constant](#38-zeros--ones--constant)
  - [3.9 orthogonal](#39-orthogonal)
  - [3.10 delta_orthogonal](#310-delta_orthogonal)
  - [3.11 truncated_normal](#311-truncated_normal)
  - [3.12 kaiming_uniform / kaiming_normal](#312-kaiming_uniform--kaiming_normal)
  - [3.13 random_uniform / random_normal](#313-random_uniform--random_normal)
- [4. Practical Patterns and Best Practices](#4-practical-patterns-and-best-practices)

---

## 1. Overview

The `jax.nn` module provides neural network specific primitives: activation functions and parameter initializers. These functions are pure, JIT-compatible, and differentiable, making them building blocks for constructing neural networks in JAX. Unlike frameworks such as PyTorch or TensorFlow, JAX does not ship with a high-level neural network library -- `jax.nn` provides the low-level primitives that libraries like Flax, Haiku, or Optax build upon.

```python
import jax
import jax.numpy as jnp
import jax.nn as jnn

# All jax.nn functions are compatible with jit, vmap, grad
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (3, 4))

# Activation functions are pure: (Array) -> Array
y = jnn.relu(x)

# They are differentiable
grad_fn = jax.grad(lambda x: jnp.sum(jnn.sigmoid(x)))
gradients = grad_fn(x)

# They are vectorizable
batched_relu = jax.vmap(jnn.relu)
```

---

## 2. Activation Functions

All activation functions in `jax.nn` accept a JAX array as input and return a JAX array of the same shape. Most accept an optional `dtype` argument or a `negative_slope` / `approximate` parameter as applicable.

### 2.1 relu

**Signature:** `jax.nn.relu(x, /)`

**Mathematical formula:**

```
relu(x) = max(0, x)
```

The Rectified Linear Unit is the most widely used activation function in deep learning. It is computationally efficient and helps mitigate the vanishing gradient problem, though it can suffer from "dying ReLU" where neurons permanently output zero.

```python
import jax
import jax.numpy as jnp
import jax.nn as jnn

x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])

# Basic usage
y = jnn.relu(x)
# Array([0., 0., 0., 1., 3.], dtype=float32)

# Works element-wise on arbitrary shapes
x_2d = jnp.array([[-2.0, 1.0], [0.5, -0.5]])
y_2d = jnn.relu(x_2d)
# Array([[0., 1.], [0.5, 0.]], dtype=float32)

# JIT-compatible
relu_jit = jax.jit(jnn.relu)
y_jit = relu_jit(x)

# Gradient computation
grad_relu = jax.grad(lambda x: jnp.sum(jnn.relu(x)))
g = grad_relu(x)
# Array([0., 0., 0., 1., 1.], dtype=float32) -- gradient is 0 for x<0, 1 for x>0

# Using relu in a simple linear layer
def linear_relu(x, w, b):
    return jnn.relu(x @ w + b)

key = jax.random.PRNGKey(42)
w = jax.random.normal(key, (4, 3))
b = jnp.zeros(3)
output = linear_relu(x_2d, w, b)
```

### 2.2 relu6

**Signature:** `jax.nn.relu6(x, /)`

**Mathematical formula:**

```
relu6(x) = min(max(0, x), 6)
```

A clipped ReLU that caps the output at 6. Originally designed for mobile neural networks, it provides bounded activations which can be beneficial for fixed-point quantization.

```python
x = jnp.array([-3.0, 0.0, 2.0, 6.0, 10.0])

y = jnn.relu6(x)
# Array([0., 0., 2., 6., 6.], dtype=float32)

# Useful in quantized models where bounded range is desired
def bounded_layer(x, w, b):
    return jnn.relu6(x @ w + b)
```

### 2.3 sigmoid

**Signature:** `jax.nn.sigmoid(x, /)`

**Mathematical formula:**

```
sigmoid(x) = 1 / (1 + exp(-x))
```

Maps any real number to the range (0, 1). Commonly used for binary classification output layers and gating mechanisms (LSTMs, GRUs). Can suffer from vanishing gradients for very large or very small inputs.

```python
x = jnp.array([-5.0, -1.0, 0.0, 1.0, 5.0])

y = jnn.sigmoid(x)
# Array([0.00669286, 0.26894143, 0.5, 0.7310586, 0.9933072], dtype=float32)

# Binary classification output
def binary_classifier(x, params):
    logits = x @ params['w'] + params['b']
    return jnn.sigmoid(logits)

# Gating mechanism
def gate(x, w_gate, w_candidate):
    g = jnn.sigmoid(x @ w_gate)
    h = jnp.tanh(x @ w_candidate)
    return g * h
```

### 2.4 softplus

**Signature:** `jax.nn.softplus(x, /)`

**Mathematical formula:**

```
softplus(x) = log(1 + exp(x))
```

A smooth approximation of ReLU. Unlike ReLU, softplus is differentiable everywhere. For large positive values it approximates the identity function; for large negative values it approaches zero.

```python
x = jnp.array([-5.0, -1.0, 0.0, 1.0, 5.0])

y = jnn.softplus(x)
# Array([0.00671534, 0.31326166, 0.6931472, 1.3132616, 5.006715], dtype=float32)

# Often used to ensure positive outputs (e.g., standard deviation parameter)
def gaussian_params(x, w_mu, w_sigma):
    mu = x @ w_mu
    sigma = jnn.softplus(x @ w_sigma)  # ensures sigma > 0
    return mu, sigma
```

### 2.5 soft_sign

**Signature:** `jax.nn.soft_sign(x, /)`

**Mathematical formula:**

```
soft_sign(x) = x / (1 + |x|)
```

A smooth activation function that maps inputs to the range (-1, 1). Compared to tanh, soft_sign has a flatter derivative and is less prone to saturation.

```python
x = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])

y = jnn.soft_sign(x)
# Array([-0.9090909, -0.5, 0., 0.5, 0.9090909], dtype=float32)

# Can be used as an alternative to tanh in hidden layers
def hidden_layer(x, w, b):
    return jnn.soft_sign(x @ w + b)
```

### 2.6 silu (swish)

**Signature:** `jax.nn.silu(x, /)`

**Mathematical formula:**

```
silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

Also known as Swish, this self-gated activation function was discovered through automated search. It is non-monotonic (has a small negative region for negative inputs), which can help with gradient flow. Widely used in modern architectures such as EfficientNet and many transformer models.

```python
x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])

y = jnn.silu(x)
# Array([-0.14227863, -0.26894143, 0., 0.7310586, 2.8577213], dtype=float32)

# Note: jnn.swish is an alias for jnn.silu
assert jnp.allclose(jnn.silu(x), jnn.swish(x))

# Common in transformer feed-forward blocks
def transformer_ffn(x, w1, b1, w2, b2):
    hidden = jnn.silu(x @ w1 + b1)
    return hidden @ w2 + b2
```

### 2.7 log_sigmoid

**Signature:** `jax.nn.log_sigmoid(x, /)`

**Mathematical formula:**

```
log_sigmoid(x) = log(sigmoid(x)) = -log(1 + exp(-x)) = -softplus(-x)
```

Computes the log of the sigmoid in a numerically stable way. Useful for computing log-probabilities in binary classification or in variational inference.

```python
x = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])

y = jnn.log_sigmoid(x)
# Array([-10.000045, -1.3132616, -0.6931472, -0.31326166, -0.0000454], dtype=float32)

# Numerically stable binary cross-entropy
def binary_log_loss(logits, labels):
    log_probs = jnn.log_sigmoid(logits)
    log_1_minus_probs = jnn.log_sigmoid(-logits)
    return -jnp.mean(labels * log_probs + (1 - labels) * log_1_minus_probs)
```

### 2.8 elu

**Signature:** `jax.nn.elu(x, alpha=1.0, /)`

**Mathematical formula:**

```
elu(x) = x                    if x > 0
         alpha * (exp(x) - 1) if x <= 0
```

Exponential Linear Unit. Unlike ReLU, ELU has a non-zero gradient for negative inputs, which helps avoid dead neurons. The output mean is closer to zero, which can speed up convergence.

```python
x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])

# Default alpha=1.0
y = jnn.elu(x)
# Array([-0.95021296, -0.63212055, 0., 1., 3.], dtype=float32)

# Custom alpha
y_alpha = jnn.elu(x, alpha=0.5)
# Array([-0.47510648, -0.31606028, 0., 1., 3.], dtype=float32)

# Using ELU in a convolutional block
def conv_block(x, w, b, alpha=1.0):
    out = jax.lax.conv_general_dilated(x, w, (1, 1), 'SAME') + b
    return jnn.elu(out, alpha=alpha)
```

### 2.9 selu

**Signature:** `jax.nn.selu(x, /)`

**Mathematical formula:**

```
selu(x) = scale * elu(x, alpha)
          where scale ~ 1.0507 and alpha ~ 1.6733
```

Scaled Exponential Linear Unit. When used with "Self-Normalizing Neural Networks" (Klambauer et al., 2017), SELU activations automatically drive activations toward zero mean and unit variance, enabling self-normalizing properties. Requires lecun_normal initialization.

```python
x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])

y = jnn.selu(x)
# Array([-1.6705687, -1.1113307, 0., 1.050701, 3.152103], dtype=float32)

# SELU network with proper initialization
from jax.nn.initializers import lecun_normal

key = jax.random.PRNGKey(0)
init_fn = lecun_normal()
w = init_fn(key, (128, 64))

def selu_layer(x, w, b):
    return jnn.selu(x @ w + b)
```

### 2.10 gelu

**Signature:** `jax.nn.gelu(x, approximate=True, /)`

**Mathematical formula (exact):**

```
gelu(x) = x * Phi(x)
           where Phi(x) is the standard Gaussian CDF
```

**Mathematical formula (approximate):**

```
gelu(x) ~ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

Gaussian Error Linear Unit. Used extensively in transformer models (BERT, GPT, etc.). The approximate version uses a tanh-based approximation that is faster to compute.

```python
x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])

# Approximate (default, faster)
y_approx = jnn.gelu(x, approximate=True)
# Array([-0.00303739, -0.15880796, 0., 0.841192, 1.9962963], dtype=float32)

# Exact (slower, more precise)
y_exact = jnn.gelu(x, approximate=False)
# Array([-0.00404923, -0.15865529, 0., 0.8413447, 2.9959507], dtype=float32)

# GPT-style transformer block
def transformer_mlp(x, w1, b1, w2, b2):
    h = jnn.gelu(x @ w1 + b1, approximate=False)
    return h @ w2 + b2

# BERT-style (uses approximate)
def bert_feedforward(x, w1, b1, w2, b2):
    h = jnn.gelu(x @ w1 + b1)
    return h @ w2 + b2
```

### 2.11 glu

**Signature:** `jax.nn.glu(x, axis=-1)`

**Mathematical formula:**

```
glu(x) = x[..., :n] * sigmoid(x[..., n:])
         where n = x.shape[axis] // 2
```

Gated Linear Unit. The input is split in half along the specified axis; the first half is multiplied by the sigmoid of the second half. Used in language models and in the Gated ConvNet architecture.

```python
# Input dimension must be even along the split axis
x = jnp.array([[1.0, 2.0, -1.0, 0.5]])  # shape (1, 4)

y = jnn.glu(x, axis=-1)
# Splits into [1.0, 2.0] and [-1.0, 0.5]
# sigmoid([-1.0, 0.5]) = [0.2689, 0.6225]
# Result: [1.0*0.2689, 2.0*0.6225] = [0.2689, 1.2449]

# GLU block in a language model
def glu_block(x, w, b):
    projected = x @ w + b  # shape: (batch, 2*d_model)
    return jnn.glu(projected, axis=-1)  # shape: (batch, d_model)
```

### 2.12 leaky_relu

**Signature:** `jax.nn.leaky_relu(x, negative_slope=0.01, /)`

**Mathematical formula:**

```
leaky_relu(x) = x                          if x >= 0
                negative_slope * x          if x < 0
```

Leaky ReLU allows a small, non-zero gradient for negative inputs, addressing the "dying ReLU" problem. The `negative_slope` parameter controls the slope for negative values.

```python
x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])

# Default negative_slope=0.01
y = jnn.leaky_relu(x)
# Array([-0.03, -0.01, 0., 1., 3.], dtype=float32)

# Custom negative_slope
y_steep = jnn.leaky_relu(x, negative_slope=0.2)
# Array([-0.6, -0.2, 0., 1., 3.], dtype=float32)

# Common in GANs (discriminator networks)
def discriminator_layer(x, w, b):
    return jnn.leaky_relu(x @ w + b, negative_slope=0.2)
```

### 2.13 hard_tanh

**Signature:** `jax.nn.hard_tanh(x, /)`

**Mathematical formula:**

```
hard_tanh(x) = -1   if x < -1
                x    if -1 <= x <= 1
                1    if x > 1
```

A piecewise linear approximation of tanh. Cheaper to compute than tanh since it avoids exponentials.

```python
x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])

y = jnn.hard_tanh(x)
# Array([-1., -1., 0., 1., 1.], dtype=float32)

# Useful in quantized networks
def quantized_activation(x, w, b):
    return jnn.hard_tanh(x @ w + b)
```

### 2.14 hard_sigmoid

**Signature:** `jax.nn.hard_sigmoid(x, /)`

**Mathematical formula:**

```
hard_sigmoid(x) = 0              if x <= -3
                   1              if x >= 3
                   x / 6 + 0.5   otherwise
```

A piecewise linear approximation of sigmoid. Used in mobile-oriented architectures for faster computation.

```python
x = jnp.array([-5.0, -3.0, 0.0, 3.0, 5.0])

y = jnn.hard_sigmoid(x)
# Array([0., 0., 0.5, 1., 1.], dtype=float32)

# Efficient gating for mobile models
def mobile_gate(x, w_gate, w_transform):
    g = jnn.hard_sigmoid(x @ w_gate)
    return g * (x @ w_transform)
```

### 2.15 hard_silu / hard_swish

**Signature:** `jax.nn.hard_silu(x, /)` / `jax.nn.hard_swish(x, /)`

**Mathematical formula:**

```
hard_silu(x) = x * hard_sigmoid(x) = x * clip(x/6 + 0.5, 0, 1)
```

`hard_swish` is an alias for `hard_silu`. This is the activation used in MobileNetV3, providing a computationally cheaper alternative to swish/silu.

```python
x = jnp.array([-5.0, -3.0, 0.0, 3.0, 5.0])

y = jnn.hard_silu(x)
# Array([0., 0., 0., 3., 5.], dtype=float32)

# hard_swish is an alias
assert jnp.allclose(jnn.hard_silu(x), jnn.hard_swish(x))

# MobileNetV3 block
def mobilenetv3_block(x, w_expand, w_project, w_gate):
    expanded = jnn.hard_silu(x @ w_expand)
    gated = expanded * jnn.hard_sigmoid(expanded @ w_gate)
    return gated @ w_project
```

### 2.16 log_softmax

**Signature:** `jax.nn.log_softmax(x, axis=-1, where=None, initial=None)`

**Mathematical formula:**

```
log_softmax(x)_i = x_i - log(sum_j(exp(x_j)))
```

Computes the log of the softmax function in a numerically stable way (subtracts the max before exponentiation). This is the preferred form for computing cross-entropy losses, as it avoids the numerical instability of computing log(softmax(x)) separately.

```python
logits = jnp.array([[2.0, 1.0, 0.1], [0.5, 2.5, 1.0]])

log_probs = jnn.log_softmax(logits, axis=-1)
# Each row sums to ~0 in exp-space, i.e., exp(log_probs).sum(axis=-1) == 1

# Cross-entropy loss using log_softmax (numerically stable)
def cross_entropy_loss(logits, labels):
    log_probs = jnn.log_softmax(logits, axis=-1)
    # labels can be integer class indices or one-hot vectors
    return -jnp.mean(jnp.take_along_axis(log_probs, labels[:, None], axis=1))

# With label smoothing
def smoothed_cross_entropy(logits, labels, smoothing=0.1, num_classes=10):
    log_probs = jnn.log_softmax(logits, axis=-1)
    one_hot = jnn.one_hot(labels, num_classes)
    smooth_labels = one_hot * (1 - smoothing) + smoothing / num_classes
    return -jnp.mean(jnp.sum(smooth_labels * log_probs, axis=-1))
```

### 2.17 softmax

**Signature:** `jax.nn.softmax(x, axis=-1, where=None, initial=None)`

**Mathematical formula:**

```
softmax(x)_i = exp(x_i) / sum_j(exp(x_j))
```

Converts a vector of real numbers into a probability distribution. The output values are in (0, 1) and sum to 1 along the specified axis. Commonly used for multi-class classification output layers and attention mechanisms.

```python
logits = jnp.array([[2.0, 1.0, 0.1], [0.5, 2.5, 1.0]])

probs = jnn.softmax(logits, axis=-1)
# Each row sums to 1.0
# Array([[0.6590012, 0.242433, 0.0985656],
#         [0.118499, 0.875601, 0.1966119]], dtype=float32)

# Attention weights
def attention(query, key, value):
    d_k = query.shape[-1]
    scores = (query @ key.T) / jnp.sqrt(d_k)
    weights = jnn.softmax(scores, axis=-1)
    return weights @ value

# Temperature-scaled softmax
def temperature_softmax(logits, temperature=0.5):
    return jnn.softmax(logits / temperature, axis=-1)
```

### 2.18 standardize

**Signature:** `jax.nn.standardize(x, axis=-1, mean=None, variance=None, epsilon=1e-05, where=None)`

**Mathematical formula:**

```
standardize(x) = (x - mean) / sqrt(variance + epsilon)
```

Normalizes the input along the specified axis to zero mean and unit variance. If `mean` and `variance` are not provided, they are computed from the input along the specified axis.

```python
x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# Normalize along last axis (each row)
y = jnn.standardize(x, axis=-1)
# Each row has mean ~0 and std ~1

# Layer normalization (manual implementation)
def layer_norm(x, gamma, beta, epsilon=1e-5):
    normalized = jnn.standardize(x, axis=-1, epsilon=epsilon)
    return gamma * normalized + beta

# Using with explicit statistics
mean = jnp.mean(x, axis=-1, keepdims=True)
var = jnp.var(x, axis=-1, keepdims=True)
y = jnn.standardize(x, axis=-1, mean=mean, variance=var)
```

### 2.19 one_hot

**Signature:** `jax.nn.one_hot(x, num_classes, *, dtype=jnp.float32)`

**Mathematical formula:**

```
one_hot(x, num_classes)_ij = 1  if x_i == j
                              0  otherwise
```

Converts integer class indices into one-hot encoded vectors. Essential for converting labels into a format compatible with softmax cross-entropy loss.

```python
# Integer class indices
labels = jnp.array([0, 2, 1, 3])

# Convert to one-hot
one_hot_labels = jnn.one_hot(labels, num_classes=4, dtype=jnp.float32)
# Array([[1., 0., 0., 0.],
#         [0., 0., 1., 0.],
#         [0., 1., 0., 0.],
#         [0., 0., 0., 1.]], dtype=float32)

# Multi-dimensional input (e.g., sequence labels)
seq_labels = jnp.array([[1, 0], [2, 1]])
one_hot_seq = jnn.one_hot(seq_labels, num_classes=3)
# Shape: (2, 2, 3)

# Cross-entropy with one_hot
def cross_entropy(logits, labels, num_classes):
    one_hot = jnn.one_hot(labels, num_classes)
    log_probs = jnn.log_softmax(logits)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))

# Integer dtype for indexing masks
one_hot_int = jnn.one_hot(labels, num_classes=4, dtype=jnp.int32)
```

### 2.20 tanh

**Signature:** `jax.nn.tanh(x, /)`

**Mathematical formula:**

```
tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```

Hyperbolic tangent. Maps inputs to the range (-1, 1). Commonly used in recurrent neural networks (LSTMs, GRUs) and as a general-purpose activation.

```python
x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])

y = jnn.tanh(x)
# Array([-0.9950547, -0.7615942, 0., 0.7615942, 0.9950547], dtype=float32)

# LSTM cell candidate
def lstm_candidate(h_prev, x, w, b):
    return jnp.tanh((jnp.concatenate([h_prev, x]) @ w) + b)
```

### 2.21 exponential

**Signature:** `jax.nn.exponential(x, /)`

**Mathematical formula:**

```
exponential(x) = exp(x)
```

Element-wise exponential. This is simply `jnp.exp(x)` exposed through the `jax.nn` namespace for convenience.

```python
x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

y = jnn.exponential(x)
# Array([0.13533528, 0.36787945, 1., 2.7182817, 7.389056], dtype=float32)

# Ensuring positive outputs for rate parameters
def poisson_rate(x, w, b):
    return jnn.exponential(x @ w + b)
```

### 2.22 mish

**Signature:** `jax.nn.mish(x, /)`

**Mathematical formula:**

```
mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
```

A self-regularizing non-monotonic activation function. Mish has shown strong empirical results, often outperforming ReLU and swish in deep networks.

```python
x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])

y = jnn.mish(x)
# Array([-0.14598134, -0.30340147, 0., 0.86509836, 2.9865353], dtype=float32)

# Mish in a residual block
def mish_block(x, w1, b1, w2, b2):
    h = jnn.mish(x @ w1 + b1)
    return x + h @ w2 + b2  # residual connection
```

### 2.23 celu

**Signature:** `jax.nn.celu(x, alpha=1.0, /)`

**Mathematical formula:**

```
celu(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1))
```

Continuously-differentiable Exponential Linear Unit. Unlike ELU, CELU is continuously differentiable at x=0, which can be beneficial for optimization.

```python
x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])

y = jnn.celu(x, alpha=1.0)
# Array([-0.95021296, -0.63212055, 0., 1., 3.], dtype=float32)

# Custom alpha
y_alpha = jnn.celu(x, alpha=0.5)
# Different behavior for negative values
```

### 2.24 approximate_gelu

**Signature:** `jax.nn.approximate_gelu(x, /)`

**Mathematical formula:**

```
approximate_gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

This is the tanh-approximation version of GELU. It is equivalent to `gelu(x, approximate=True)`.

```python
x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])

y = jnn.approximate_gelu(x)
# Same as jnn.gelu(x, approximate=True)

# Verify equivalence
assert jnp.allclose(jnn.approximate_gelu(x), jnn.gelu(x, approximate=True))
```

---

## 3. Initializers (jax.nn.initializers)

### 3.1 Overview and Key Concepts

Initializers in JAX are functions that return **initializer functions**. The pattern is:

1. Call an initializer constructor (e.g., `jax.nn.initializers.normal()`) to get an `init_fn`.
2. Call `init_fn(key, shape)` with a PRNG key and desired shape to get an array.

This two-level design allows initializers to be passed around as configuration objects.

```python
import jax
import jax.nn.initializers as init

# Step 1: Create an initializer function
init_fn = init.normal(stddev=0.01)

# Step 2: Call it with a key and shape
key = jax.random.PRNGKey(0)
params = init_fn(key, (128, 64))
# params is a (128, 64) array drawn from N(0, 0.01^2)

# Can also specify dtype
params_f16 = init_fn(key, (128, 64), dtype=jnp.float16)
```

Most initializers accept a `dtype` parameter in the call to `init_fn`. The `variance_scaling` base class also accepts an `axis` parameter that controls how fan-in/fan-out are computed.

### 3.2 variance_scaling (Base Class)

**Signature:**

```python
jax.nn.initializers.variance_scaling(
    scale, mode, distribution, *, in_axis=-2, out_axis=-1,
    dtype=jnp.float_, precision=None
)
```

This is the base class for most initializers. It controls the variance of the initialization based on the number of input/output units.

**Parameters:**
- `scale` (float): Scale factor multiplied to the variance.
- `mode` (str): One of `"fan_in"`, `"fan_out"`, `"fan_avg"`. Determines which fan value to use for variance computation.
- `distribution` (str): One of `"truncated_normal"`, `"normal"`, `"uniform"`. The sampling distribution.
- `in_axis` (int): Axis of the input dimension (default -2).
- `out_axis` (int): Axis of the output dimension (default -1).

**Mathematical formula:**

```
stddev = sqrt(scale / n)
where n = fan_in   (mode="fan_in")
      n = fan_out  (mode="fan_out")
      n = (fan_in + fan_out) / 2  (mode="fan_avg")
```

```python
import jax
import jax.nn.initializers as init
import jnp as jax.numpy

key = jax.random.PRNGKey(0)

# Custom variance scaling: fan_in mode, uniform distribution
custom_init = init.variance_scaling(
    scale=2.0,
    mode='fan_in',
    distribution='uniform'
)
w = custom_init(key, (256, 128))

# For convolutional layers, the shape is (kH, kW, C_in, C_out)
# in_axis=-2 and out_axis=-1 correctly identify C_in and C_out
conv_init = init.variance_scaling(
    scale=1.0,
    mode='fan_out',
    distribution='normal'
)
conv_w = conv_init(key, (3, 3, 64, 128))
```

### 3.3 uniform

**Signature:**

```python
jax.nn.initializers.uniform(scale=0.01, dtype=jnp.float_)
```

Initializes with a uniform distribution in the range `[-scale, scale)`.

```python
import jax
import jax.nn.initializers as init

key = jax.random.PRNGKey(42)

# Default scale=0.01
init_fn = init.uniform()
w = init_fn(key, (100, 50))
# Values uniformly distributed in [-0.01, 0.01)

# Custom scale
init_fn = init.uniform(scale=1.0)
w = init_fn(key, (100, 50))
# Values uniformly distributed in [-1.0, 1.0)

# For bias initialization (small uniform)
bias_init = init.uniform(scale=1e-4)
b = bias_init(key, (50,))
```

### 3.4 normal

**Signature:**

```python
jax.nn.initializers.normal(stddev=0.01, dtype=jnp.float_)
```

Initializes with a normal (Gaussian) distribution.

```python
key = jax.random.PRNGKey(42)

# Default stddev=0.01
init_fn = init.normal()
w = init_fn(key, (100, 50))
# Values from N(0, 0.01^2)

# Custom stddev
init_fn = init.normal(stddev=0.05)
w = init_fn(key, (100, 50))

# Common for weight initialization in simple models
def init_mlp_params(layer_sizes, key):
    params = []
    for i in range(len(layer_sizes) - 1):
        key, k1, k2 = jax.random.split(key, 3)
        w = init.normal(stddev=0.01)(k1, (layer_sizes[i], layer_sizes[i+1]))
        b = jnp.zeros((layer_sizes[i+1],))
        params.append({'w': w, 'b': b})
    return params
```

### 3.5 he_uniform / he_normal

**Signature:**

```python
jax.nn.initializers.he_uniform(dtype=jnp.float_)
jax.nn.initializers.he_normal(dtype=jnp.float_)
```

He initialization (also called Kaiming initialization) is designed for layers using ReLU-family activations. It sets the variance to `2 / fan_in`.

**Mathematical formula:**

```
he_normal:  W ~ N(0, sqrt(2 / fan_in))
he_uniform: W ~ U(-sqrt(6 / fan_in), sqrt(6 / fan_in))
```

**Recommended use:** Layers followed by ReLU, LeakyReLU, or other ReLU variants.

```python
key = jax.random.PRNGKey(0)

# He normal for ReLU networks
init_fn = init.he_normal()
w = init_fn(key, (512, 256))
# std ~= sqrt(2/512) ~= 0.0625

# He uniform
init_fn = init.he_uniform()
w = init_fn(key, (512, 256))

# Typical usage in a ReLU network
def init_relu_network(sizes, key):
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        w = init.he_normal()(k, (fan_in, fan_out))
        b = jnp.zeros((fan_out,))
        params.append((w, b))
    return params

network = init_relu_network([784, 512, 256, 10], key)
```

### 3.6 glorot_uniform / glorot_normal (Xavier)

**Signature:**

```python
jax.nn.initializers.glorot_uniform(dtype=jnp.float_)
jax.nn.initializers.glorot_normal(dtype=jnp.float_)
```

Glorot (Xavier) initialization is designed to maintain the variance of activations across layers. It uses `fan_avg` mode.

**Mathematical formula:**

```
glorot_normal:  W ~ N(0, sqrt(2 / (fan_in + fan_out)))
glorot_uniform: W ~ U(-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out)))
```

**Recommended use:** Layers followed by sigmoid, tanh, or softmax activations.

```python
key = jax.random.PRNGKey(0)

# Glorot normal for sigmoid/tanh networks
init_fn = init.glorot_normal()
w = init_fn(key, (512, 256))

# Glorot uniform
init_fn = init.glorot_uniform()
w = init_fn(key, (512, 256))

# LSTM weight initialization (uses tanh/sigmoid gates)
def init_lstm_params(input_dim, hidden_dim, key):
    k1, k2, k3 = jax.random.split(key, 3)

    # Input-to-hidden weights: use glorot
    w_ih = init.glorot_uniform()(k1, (input_dim, 4 * hidden_dim))

    # Hidden-to-hidden weights: use orthogonal for better gradient flow
    w_hh = init.orthogonal()(k2, (hidden_dim, 4 * hidden_dim))

    b = jnp.zeros((4 * hidden_dim,))
    return w_ih, w_hh, b
```

### 3.7 lecun_uniform / lecun_normal

**Signature:**

```python
jax.nn.initializers.lecun_uniform(dtype=jnp.float_)
jax.nn.initializers.lecun_normal(dtype=jnp.float_)
```

LeCun initialization uses `fan_in` mode with scale=1.0. Designed for SELU activations in self-normalizing networks.

**Mathematical formula:**

```
lecun_normal:  W ~ N(0, sqrt(1 / fan_in))
lecun_uniform: W ~ U(-sqrt(3 / fan_in), sqrt(3 / fan_in))
```

**Recommended use:** Layers followed by SELU activations (self-normalizing networks).

```python
key = jax.random.PRNGKey(0)

# LeCun normal for SELU networks
init_fn = init.lecun_normal()
w = init_fn(key, (512, 256))
# std ~= sqrt(1/512) ~= 0.0442

# SELU self-normalizing network
def init_selu_network(sizes, key):
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        w = init.lecun_normal()(k, (fan_in, fan_out))
        b = jnp.zeros((fan_out,))
        params.append((w, b))
    return params
```

### 3.8 zeros / ones / constant

**Signature:**

```python
jax.nn.initializers.zeros(dtype=jnp.float_)
jax.nn.initializers.ones(dtype=jnp.float_)
jax.nn.initializers.constant(value, dtype=jnp.float_)
```

Simple constant initializers. Typically used for bias terms, output layers, or parameters that should start at a specific value.

```python
key = jax.random.PRNGKey(0)

# Zeros (common for biases)
bias_init = init.zeros()
b = bias_init(key, (256,))  # Always returns zeros

# Ones
ones_init = init.ones()
mask = ones_init(key, (10,))  # Always returns ones

# Constant value
const_init = init.constant(0.5)
log_var = const_init(key, (256,))  # All elements are 0.5

# Typical parameter initialization pattern
def init_layer(fan_in, fan_out, key, use_he=True):
    k1, k2 = jax.random.split(key)
    if use_he:
        w = init.he_normal()(k1, (fan_in, fan_out))
    else:
        w = init.glorot_normal()(k1, (fan_in, fan_out))
    b = init.zeros()(k2, (fan_out,))
    return {'w': w, 'b': b}
```

### 3.9 orthogonal

**Signature:**

```python
jax.nn.initializers.orthogonal(dtype=jnp.float_, column=False)
```

Initializes with a random orthogonal matrix. Produces a (semi-)orthogonal matrix where the columns (or rows, if `column=True`) are mutually orthogonal. This helps preserve the magnitude of activations and gradients.

**Parameters:**
- `column` (bool): If True, generates an orthonormal basis for the columns. Default False.

**Recommended use:** Recurrent neural networks (RNNs, LSTMs, GRUs) hidden-to-hidden weights.

```python
key = jax.random.PRNGKey(0)

# Orthogonal matrix
init_fn = init.orthogonal()
w = init_fn(key, (512, 256))
# w.T @ w ~= I (256x256 identity)

# Column-mode
init_fn = init.orthogonal(column=True)
w = init_fn(key, (512, 256))

# RNN hidden weight initialization
def init_rnn(input_dim, hidden_dim, key):
    k1, k2 = jax.random.split(key)
    w_xh = init.glorot_uniform()(k1, (input_dim, hidden_dim))
    w_hh = init.orthogonal()(k2, (hidden_dim, hidden_dim))
    b = jnp.zeros((hidden_dim,))
    return w_xh, w_hh, b

# Verify orthogonality
w = init.orthogonal()(key, (256, 128))
print(jnp.linalg.norm(w.T @ w - jnp.eye(128)))  # Should be close to 0
```

### 3.10 delta_orthogonal

**Signature:**

```python
jax.nn.initializers.delta_orthogonal(dtype=jnp.float_, column=False)
```

Initializes with a delta-orthogonal matrix. This variant is designed for convolutional layers, ensuring the initialization is orthogonal when the kernel size is greater than 1x1. Only the central spatial location of each filter is initialized orthogonally; all others are set to zero.

**Recommended use:** Convolutional layers with kernel sizes > 1.

```python
key = jax.random.PRNGKey(0)

# For a 3x3 convolution with 64 input and 128 output channels
init_fn = init.delta_orthogonal()
conv_w = init_fn(key, (3, 3, 64, 128))
# Shape: (3, 3, 64, 128) - only center pixel initialized

# For 1x1 convolution, delta_orthogonal is equivalent to orthogonal
w_1x1 = init.delta_orthogonal()(key, (1, 1, 64, 128))
```

### 3.11 truncated_normal

**Signature:**

```python
jax.nn.initializers.truncated_normal(stddev=0.01, dtype=jnp.float_, lower=-2.0, upper=2.0)
```

Initializes with a truncated normal distribution, where values outside `[lower * stddev, upper * stddev]` are resampled. This avoids extreme values that can destabilize training.

```python
key = jax.random.PRNGKey(0)

# Default stddev=0.01
init_fn = init.truncated_normal(stddev=0.05)
w = init_fn(key, (512, 256))
# Values from N(0, 0.05^2) truncated at [-0.1, 0.1]

# Custom truncation bounds
init_fn = init.truncated_normal(stddev=1.0, lower=-3.0, upper=3.0)
w = init_fn(key, (512, 256))

# Common in transformer models
def init_transformer_params(d_model, d_ff, key):
    k1, k2, k3 = jax.random.split(key, 3)
    w_qkv = init.truncated_normal(stddev=0.02)(k1, (d_model, 3 * d_model))
    w_ff1 = init.truncated_normal(stddev=0.02)(k2, (d_model, d_ff))
    w_ff2 = init.truncated_normal(stddev=0.02)(k3, (d_ff, d_model))
    return w_qkv, w_ff1, w_ff2
```

### 3.12 kaiming_uniform / kaiming_normal

**Signature:**

```python
jax.nn.initializers.kaiming_uniform(dtype=jnp.float_)
jax.nn.initializers.kaiming_normal(dtype=jnp.float_)
```

These are aliases for `he_uniform` and `he_normal` respectively, following the naming convention from PyTorch.

```python
key = jax.random.PRNGKey(0)

# kaiming_normal is equivalent to he_normal
w1 = init.kaiming_normal()(key, (512, 256))
w2 = init.he_normal()(key, (512, 256))
# These produce the same distribution but different samples

# kaiming_uniform is equivalent to he_uniform
w3 = init.kaiming_uniform()(key, (512, 256))
```

### 3.13 random_uniform / random_normal

**Signature:**

```python
jax.nn.initializers.random_uniform(minval=0.0, maxval=1.0, dtype=jnp.float_)
jax.nn.initializers.random_normal(stddev=1.0, mean=0.0, dtype=jnp.float_)
```

Basic random initializers with configurable parameters. These do not use any fan-in/fan-out scaling.

```python
key = jax.random.PRNGKey(0)

# Uniform in [0, 1) (default)
init_fn = init.random_uniform()
w = init_fn(key, (100, 50))

# Uniform in custom range
init_fn = init.random_uniform(minval=-0.5, maxval=0.5)
w = init_fn(key, (100, 50))

# Normal with mean=0, stddev=1 (default)
init_fn = init.random_normal()
w = init_fn(key, (100, 50))

# Normal with custom parameters
init_fn = init.random_normal(mean=0.5, stddev=0.1)
w = init_fn(key, (100, 50))
```

---

## 4. Practical Patterns and Best Practices

### 4.1 Choosing the Right Activation Function

```python
import jax
import jax.numpy as jnp
import jax.nn as jnn

# Summary guide for activation selection:
#
# | Architecture / Layer      | Recommended Activation  |
# |---------------------------|------------------------|
# | General hidden layers      | relu, gelu, silu      |
# | Transformer FFN           | gelu (exact or approx) |
# | Transformer attention      | softmax               |
# | Binary classification     | sigmoid (output)       |
# | Multi-class classification| softmax (output)       |
# | Multi-label classification| sigmoid (output)       |
# | Regression (unbounded)     | linear (no activation)|
# | Regression (positive)      | softplus, exponential  |
# | RNN/GRU gates              | sigmoid               |
# | RNN/GRU candidate          | tanh                  |
# | LSTM gates                 | sigmoid               |
# | LSTM candidate             | tanh                  |
# | Mobile/efficient nets      | hard_silu, relu6      |
# | Self-normalizing nets      | selu                  |
# | GAN discriminators         | leaky_relu            |
```

### 4.2 Choosing the Right Initializer

```python
import jax.nn.initializers as init

# Summary guide for initializer selection:
#
# | Activation Function | Recommended Initializer     |
# |--------------------|-----------------------------|
# | relu               | he_normal, he_uniform       |
# | leaky_relu         | he_normal, he_uniform       |
# | selu               | lecun_normal, lecun_uniform |
# | sigmoid            | glorot_normal, glorot_uniform|
# | tanh               | glorot_normal, glorot_uniform|
# | gelu / silu        | he_normal, truncated_normal |
# | RNN hidden-hidden  | orthogonal                  |
# | Conv kernel > 1x1  | delta_orthogonal            |
# | Bias terms         | zeros                       |
# | BatchNorm params   | ones (scale), zeros (offset)|
```

### 4.3 Full Network Initialization Example

```python
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.nn.initializers as init
from typing import NamedTuple

class MLPParams(NamedTuple):
    weights: list
    biases: list

def init_mlp(
    layer_sizes: list,
    activation: str = 'relu',
    key: jax.Array = jax.random.PRNGKey(0),
):
    """Initialize MLP parameters with appropriate initializers."""
    # Select initializer based on activation
    if activation in ('relu', 'leaky_relu', 'silu', 'gelu'):
        kernel_init = init.he_normal()
    elif activation in ('selu',):
        kernel_init = init.lecun_normal()
    else:
        kernel_init = init.glorot_normal()

    weights, biases = [], []
    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        key, k1 = jax.random.split(key)
        w = kernel_init(k1, (fan_in, fan_out))
        b = jnp.zeros((fan_out,))
        weights.append(w)
        biases.append(b)

    return MLPParams(weights, biases)

def mlp_forward(params: MLPParams, x: jnp.ndarray, activation: str = 'relu'):
    """Forward pass through MLP."""
    act_fn = getattr(jnn, activation)

    for w, b in zip(params.weights[:-1], params.biases[:-1]):
        x = act_fn(x @ w + b)

    # Output layer without activation
    x = x @ params.weights[-1] + params.biases[-1]
    return x

# Initialize and run
key = jax.random.PRNGKey(42)
params = init_mlp([784, 256, 128, 10], activation='gelu', key=key)
x = jax.random.normal(key, (32, 784))
logits = mlp_forward(params, x, activation='gelu')
probs = jnn.softmax(logits, axis=-1)
```

### 4.4 Transformer Initialization Pattern

```python
def init_transformer_params(d_model, n_heads, d_ff, key):
    """Initialize a single transformer layer."""
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    head_dim = d_model // n_heads

    params = {}

    # Self-attention: Q, K, V projections
    for name in ['query', 'key', 'value']:
        params[f'attn_{name}_w'] = init.he_normal()(
            k1, (d_model, d_model)
        )
        params[f'attn_{name}_b'] = jnp.zeros((d_model,))
        k1, = jax.random.split(k1, 1)

    # Output projection
    params['attn_out_w'] = init.he_normal()(k2, (d_model, d_model))
    params['attn_out_b'] = jnp.zeros((d_model,))

    # Feed-forward network
    params['ffn_w1'] = init.he_normal()(k3, (d_model, d_ff))
    params['ffn_b1'] = jnp.zeros((d_ff,))
    params['ffn_w2'] = init.he_normal()(k4, (d_ff, d_model))
    params['ffn_b2'] = jnp.zeros((d_model,))

    # Layer norm
    params['ln1_gamma'] = jnp.ones((d_model,))
    params['ln1_beta'] = jnp.zeros((d_model,))
    params['ln2_gamma'] = jnp.ones((d_model,))
    params['ln2_beta'] = jnp.zeros((d_model,))

    return params
```

### 4.5 Activation Function Comparison

```python
import jax
import jax.numpy as jnp
import jax.nn as jnn

def compare_activations():
    """Compare all activation functions on a range of inputs."""
    x = jnp.linspace(-5, 5, 100)

    activations = {
        'relu': jnn.relu,
        'leaky_relu': jnn.leaky_relu,
        'sigmoid': jnn.sigmoid,
        'tanh': jnn.tanh,
        'elu': jnn.elu,
        'selu': jnn.selu,
        'gelu': lambda x: jnn.gelu(x, approximate=False),
        'silu': jnn.silu,
        'mish': jnn.mish,
        'softplus': jnn.softplus,
        'soft_sign': jnn.soft_sign,
        'hard_silu': jnn.hard_silu,
        'relu6': jnn.relu6,
    }

    results = {}
    for name, fn in activations.items():
        y = fn(x)
        grad_fn = jax.grad(lambda x: fn(x).sum())
        g = jax.vmap(grad_fn)(x)
        results[name] = {'output': y, 'gradient': g}

    return results

# Check for NaN/Inf issues at extreme inputs
extreme = jnp.array([-1000.0, -100.0, -10.0, 10.0, 100.0, 1000.0])
for name, fn in [('softmax', lambda x: jnn.softmax(x)),
                 ('log_softmax', lambda x: jnn.log_softmax(x)),
                 ('sigmoid', jnn.sigmoid),
                 ('gelu', jnn.gelu)]:
    y = fn(extreme)
    has_nan = jnp.any(jnp.isnan(y))
    has_inf = jnp.any(jnp.isinf(y))
    print(f"{name}: nan={has_nan}, inf={has_inf}")
```

### 4.6 Custom Activation Functions

```python
import jax
import jax.numpy as jnp
import jax.nn as jnn

# Swish with learnable beta parameter
def swish_beta(x, beta=1.0):
    """Swish activation with learnable beta: x * sigmoid(beta * x)"""
    return x * jnn.sigmoid(beta * x)

# Mish variant
def Mish_custom(x):
    """Custom mish: x * tanh(softplus + epsilon * x^2)"""
    return x * jnp.tanh(jnn.softplus(x) + 0.01 * x ** 2)

# bent_identity
def bent_identity(x):
    """Bent identity: (sqrt(x^2 + 1) - 1) / 2 + x"""
    return (jnp.sqrt(x ** 2 + 1) - 1) / 2 + x

# All custom activations are automatically differentiable
grad_swish = jax.grad(swish_beta)
g = grad_swish(1.5)
```

### 4.7 Numerical Stability Considerations

```python
import jax
import jax.numpy as jnp
import jax.nn as jnn

# ALWAYS use log_softmax instead of log(softmax) for cross-entropy
def stable_cross_entropy(logits, labels):
    """Numerically stable cross-entropy."""
    return -jnp.sum(jnn.one_hot(labels, logits.shape[-1]) * jnn.log_softmax(logits))

def unstable_cross_entropy(logits, labels):
    """Numerically UNSTABLE - do not use."""
    return -jnp.sum(jnn.one_hot(labels, logits.shape[-1]) * jnp.log(jnn.softmax(logits)))

# ALWAYS use log_sigmoid instead of log(sigmoid)
def stable_bce(logits, labels):
    """Numerically stable binary cross-entropy."""
    return -jnp.mean(labels * jnn.log_sigmoid(logits) + (1 - labels) * jnn.log_sigmoid(-logits))

# softplus is numerically stable for large inputs
# For very large positive x, softplus(x) ~= x
# For very large negative x, softplus(x) ~= 0
x_large = jnp.array([1000.0, -1000.0])
print(jnn.softplus(x_large))  # [1000.0, 0.0] - no overflow/underflow
```
