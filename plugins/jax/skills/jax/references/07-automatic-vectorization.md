# Automatic Vectorization (jax.vmap)

This document provides an exhaustive reference for `jax.vmap`, JAX's automatic vectorization transformation. It covers the full API surface, batching semantics, composition with other transformations, advanced patterns, performance considerations, and known limitations.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Manual vs Automatic Vectorization](#2-manual-vs-automatic-vectorization)
3. [jax.vmap API Reference](#3-jaxvmap-api-reference)
4. [in_axes Examples](#4-in_axes-examples)
5. [out_axes](#5-out_axes)
6. [axis_name and spmd_axis_name](#6-axis_name-and-spmd_axis_name)
7. [Batch Dimension Positioning](#7-batch-dimension-positioning)
8. [Composing vmap with jit](#8-composing-vmap-with-jit)
9. [Composing vmap with grad](#9-composing-vmap-with-grad)
10. [Nested vmap](#10-nested-vmap)
11. [vmap and Control Flow](#11-vmap-and-control-flow)
12. [vmap with Pytrees](#12-vmap-with-pytrees)
13. [vmap Performance Considerations](#13-vmap-performance-considerations)
14. [Limitations and When vmap May Not Work](#14-limitations-and-when-vmap-may-not-work)
15. [custom_vmap](#15-custom_vmap)
16. [Complete Examples](#16-complete-examples)

---

## 1. Overview

`jax.vmap` (vectorizing map) is a transformation that automatically adds a batch dimension to a function written for single examples. Instead of manually rewriting code to handle batched inputs, you wrap the function with `vmap` and JAX handles the batching transformation at the trace level.

**Key properties:**
- Produces a single compiled kernel (not a loop of separate calls)
- Composes freely with `jit`, `grad`, and other transformations
- Works on pure functions that operate on JAX arrays and pytrees
- Handles nested batching through composition

```python
import jax
import jax.numpy as jnp

# Function written for a single vector
def normalize(x):
    return x / jnp.linalg.norm(x)

# Automatically batch it to handle a batch of vectors
batched_normalize = jax.vmap(normalize)

# Works on a batch of 100 vectors
x_batch = jax.random.normal(jax.random.key(0), (100, 50))
result = batched_normalize(x_batch)  # shape (100, 50)
```

---

## 2. Manual vs Automatic Vectorization

### Manual Vectorization (Before vmap)

Manual vectorization requires restructuring the function to handle an extra batch dimension, which can be error-prone and verbose.

```python
import jax.numpy as jnp

def predict_single(params, x):
    """Single-example prediction: x has shape (features,)"""
    h = x
    for w, b in params:
        h = jnp.dot(h, w) + b
        h = jnp.maximum(h, 0)  # ReLU
    return h

# Manual vectorization: rewrite to handle batch dimension
def predict_batched(params, x_batch):
    """Batched prediction: x_batch has shape (batch, features)"""
    h = x_batch  # (batch, features)
    for w, b in params:
        h = jnp.dot(h, w) + b  # (batch, features) @ (features, hidden) -> (batch, hidden)
        h = jnp.maximum(h, 0)
    return h

# Problem: more complex logic is hard to vectorize manually
def complex_fn_single(x):
    """Hard to manually vectorize due to per-element logic"""
    norm = jnp.linalg.norm(x)
    if norm > 1.0:  # This won't work with a batch dimension!
        return x / norm
    else:
        return x * 2.0
```

### Automatic Vectorization (With vmap)

```python
import jax
import jax.numpy as jnp

def predict_single(params, x):
    """Same function, written for single examples"""
    h = x
    for w, b in params:
        h = jnp.dot(h, w) + b
        h = jnp.maximum(h, 0)
    return h

# One line to batch it
predict_batched = jax.vmap(predict_single, in_axes=(None, 0))

# Works identically to the manual version
params = [(jnp.ones((50, 30)), jnp.zeros(30)),
          (jnp.ones((30, 10)), jnp.zeros(10))]
x_batch = jnp.ones((100, 50))
result = predict_batched(params, x_batch)  # shape (100, 10)
```

---

## 3. jax.vmap API Reference

```python
jax.vmap(
    fun,            # Function to vectorize
    in_axes=0,      # Which axes to vectorize over for each input
    out_axes=0,     # Which axes to place the batch in for each output
    axis_name=None,  # Name for the vmapped axis (for axis-index operations)
    spmd_axis_name=None,  # SPMD sharding axis name
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fun` | Callable | required | Function to vectorize. Must be a pure function. |
| `in_axes` | int, None, sequence, or pytree | `0` | Axis to map over for each argument. `None` means do not map (broadcast). An integer specifies the axis position. Can be a pytree matching the input structure. |
| `out_axes` | int, None, sequence, or pytree | `0` | Where to place the mapped axis in each output. `None` means the output must be the same across the batch and will be squeezed. |
| `axis_name` | Hashable | `None` | Optional name for the mapped axis. Used with `jax.lax.axis_index` and `jax.lax.psum` inside the function. |
| `spmd_axis_name` | Hashable | `None` | For SPMD sharding: names the axis for mesh-related operations. |

### Returns

A vectorized version of `fun`. The returned function applies `fun` across the specified batch dimension(s) of its inputs.

---

## 4. in_axes Examples

### Integer in_axes

When `in_axes` is a single integer, it applies to all positional arguments.

```python
import jax
import jax.numpy as jnp

def add(x, y):
    return x + y

# Vectorize over axis 0 of both x and y
vmap_add = jax.vmap(add, in_axes=0)

x = jnp.array([[1, 2], [3, 4]])    # shape (2, 2)
y = jnp.array([[5, 6], [7, 8]])    # shape (2, 2)
result = vmap_add(x, y)             # shape (2, 2)
# result = [[6, 8], [10, 12]]
```

### List of in_axes (Per-Argument)

```python
import jax
import jax.numpy as jnp

def weighted_sum(weights, values):
    return jnp.sum(weights * values)

# Vectorize over axis 0 of values, but not weights (weights is shared)
vmap_ws = jax.vmap(weighted_sum, in_axes=(None, 0))

weights = jnp.array([0.1, 0.3, 0.6])        # shape (3,)
values = jnp.array([[1.0, 2.0, 3.0],        # shape (2, 3)
                     [4.0, 5.0, 6.0]])
result = vmap_ws(weights, values)            # shape (2,)
# result = [2.2, 5.2]
```

### None in_axes (Broadcast / No Mapping)

```python
import jax
import jax.numpy as jnp

def scale_and_bias(x, scale, bias):
    return x * scale + bias

# Vectorize only x; scale and bias are shared across the batch
vmap_fn = jax.vmap(scale_and_bias, in_axes=(0, None, None))

x_batch = jnp.arange(10.0).reshape(5, 2)  # (5, 2)
scale = jnp.array([2.0, 3.0])              # (2,)
bias = jnp.array([1.0, 1.0])               # (2,)
result = vmap_fn(x_batch, scale, bias)     # (5, 2)
```

### Nested Pytree in_axes

```python
import jax
import jax.numpy as jnp

def apply_params(params, x):
    """params is a pytree, x is an array"""
    return params["w"] @ x + params["b"]

# in_axes as a pytree matching the function's input structure
# params: vectorize over axis 0 of 'w' and 'b', x: vectorize over axis 0
vmap_fn = jax.vmap(
    apply_params,
    in_axes=(
        {"w": 0, "b": 0},  # Vectorize over axis 0 of each param
        0,                   # Vectorize over axis 0 of x
    )
)

params_batch = {
    "w": jnp.ones((5, 3, 4)),   # batch of 5 weight matrices
    "b": jnp.ones((5, 4)),       # batch of 5 bias vectors
}
x_batch = jnp.ones((5, 3))       # batch of 5 input vectors
result = vmap_fn(params_batch, x_batch)  # shape (5, 4)
```

### Mixed None and Integer in a Pytree

```python
import jax
import jax.numpy as jnp

def layer_forward(params, x):
    # Only the weight is batched; bias is shared
    return jnp.dot(params["w"], x) + params["b"]

vmap_fn = jax.vmap(
    layer_forward,
    in_axes=(
        {"w": 0, "b": None},  # w is batched, b is shared
        0,                      # x is batched
    )
)

params = {
    "w": jnp.ones((5, 4, 3)),  # batch of 5 weight matrices
    "b": jnp.zeros(3),          # single shared bias
}
x = jnp.ones((5, 4))
result = vmap_fn(params, x)    # shape (5, 3)
```

---

## 5. out_axes

### Default out_axes (0)

By default, the batch dimension is placed as axis 0 in the output.

```python
import jax
import jax.numpy as jnp

def get_row(x):
    return x  # x has shape (3,)

vmap_fn = jax.vmap(get_row, in_axes=0, out_axes=0)
x = jnp.arange(12.0).reshape(4, 3)  # (4, 3)
result = vmap_fn(x)                   # (4, 3), batch dim is axis 0
```

### Custom out_axes Position

```python
import jax
import jax.numpy as jnp

def get_row(x):
    return x  # x has shape (3,)

# Place batch dimension as last axis
vmap_fn = jax.vmap(get_row, in_axes=0, out_axes=-1)
x = jnp.arange(12.0).reshape(4, 3)  # (4, 3)
result = vmap_fn(x)                   # (3, 4), batch dim moved to axis 1
```

### out_axes=None (Require Identical Outputs)

When `out_axes=None`, the mapped dimension must produce identical values across the batch. If they differ, an error is raised. This is useful for asserting invariants.

```python
import jax
import jax.numpy as jnp

def sum_and_assert_same(x):
    return jnp.sum(x)  # scalar output

# With out_axes=None, all batch elements must produce the same result
vmap_fn = jax.vmap(sum_and_assert_same, in_axes=0, out_axes=None)
x = jnp.ones((5, 3)) * 3.0  # All rows sum to 9.0
result = vmap_fn(x)           # scalar: 9.0
```

### Pytree out_axes

```python
import jax
import jax.numpy as jnp

def multi_output(x):
    return {"mean": jnp.mean(x), "std": jnp.std(x)}

vmap_fn = jax.vmap(multi_output, in_axes=0, out_axes={"mean": 0, "std": -1})
x = jnp.arange(12.0).reshape(4, 3)
result = vmap_fn(x)
# result["mean"] has shape (4,), result["std"] has shape (3,) because out_axes=-1
# Wait -- actually for 1D outputs, axis -1 == axis 0, so both are (4,)
```

---

## 6. axis_name and spmd_axis_name

### axis_name for Axis-Index Operations

When `axis_name` is provided, the function can use `jax.lax.axis_index` to get the index of the current batch element, and `jax.lax.psum` for collective operations across the batch.

```python
import jax
import jax.numpy as jnp

def position_encoding(x):
    # Get the batch index for each element
    idx = jax.lax.axis_index("batch")
    # Use it to create a position-dependent encoding
    return x + idx.astype(jnp.float32)

vmap_fn = jax.vmap(position_encoding, axis_name="batch")
x = jnp.zeros((5, 3))
result = vmap_fn(x)
# result[i, :] = i for each row i
# [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]
```

### Using psum with axis_name

```python
import jax
import jax.numpy as jnp

def normalized_with_psum(x):
    # Sum all x's across the batch dimension
    total = jax.lax.psum(x, axis_name="batch")
    return x / total

vmap_fn = jax.vmap(normalized_with_psum, axis_name="batch")
x = jnp.array([1.0, 2.0, 3.0])[:, None] * jnp.ones((1, 4))
result = vmap_fn(x)
# Each element divided by sum across batch: total = 1+2+3=6 per column
# result[0] = 1/6, result[1] = 2/6, result[2] = 3/6 (repeated across columns)
```

### spmd_axis_name

The `spmd_axis_name` parameter is used for advanced SPMD (Single Program Multiple Data) sharding scenarios. It links the vmapped axis to a named mesh axis, enabling the compiler to reason about sharding.

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

# spmd_axis_name tells the XLA compiler how the vmapped axis
# corresponds to a device mesh axis, enabling efficient sharding
def compute(x):
    return x * 2.0

vmap_fn = jax.vmap(compute, spmd_axis_name="data")
# When used with a Mesh that has a "data" axis, the vmapped dimension
# can be automatically sharded across the corresponding devices
```

---

## 7. Batch Dimension Positioning

vmap adds a batch axis at the position specified by `in_axes`. Understanding how this interacts with existing dimensions is critical.

```python
import jax
import jax.numpy as jnp

# Function operating on a matrix (2D)
def mat_diag_sum(m):
    """Sum of diagonal elements of a 2x2 matrix"""
    return jnp.sum(jnp.diag(m))

# in_axes=0: batch over rows (each element is a row, not a matrix!)
# This would fail because each element is 1D, not 2D

# in_axes=0 on a 3D tensor: batch over matrices
vmap_fn = jax.vmap(mat_diag_sum, in_axes=0)
matrices = jnp.arange(24.0).reshape(4, 2, 2)  # 4 matrices of size 2x2
result = vmap_fn(matrices)  # shape (4,)

# in_axes=1: batch over columns
vmap_fn2 = jax.vmap(mat_diag_sum, in_axes=1)
data = jnp.arange(24.0).reshape(2, 4, 2)  # 4 matrices of size 2x2 (axis 1)
result2 = vmap_fn2(data)  # shape (4,)

# in_axes=2: batch over the last axis
vmap_fn3 = jax.vmap(mat_diag_sum, in_axes=2)
data3 = jnp.arange(24.0).reshape(2, 2, 4)
result3 = vmap_fn3(data3)  # shape (4,)
```

---

## 8. Composing vmap with jit

`vmap` and `jit` compose freely. The order of composition matters for performance semantics but both orderings produce correct results.

### jit(vmap(f)) -- JIT the vectorized function

```python
import jax
import jax.numpy as jnp

def elementwise_fn(x):
    return jnp.sin(x) ** 2 + jnp.cos(x) ** 2

# Vectorize first, then JIT compile the vectorized version
jitted_vmap = jax.jit(jax.vmap(elementwise_fn))

x = jnp.arange(1000.0).reshape(100, 10)
result = jitted_vmap(x)  # Single compiled kernel for the batched operation
```

### vmap(jit(f)) -- JIT the single-element function, then vectorize

```python
import jax
import jax.numpy as jnp

@jax.jit
def elementwise_fn(x):
    return jnp.sin(x) ** 2 + jnp.cos(x) ** 2

# The inner JIT compiles the function once; vmap then vectorizes it
vmapped_jit = jax.vmap(elementwise_fn)

x = jnp.arange(1000.0).reshape(100, 10)
result = vmapped_jit(x)
# Note: vmap(jit(f)) may be less efficient than jit(vmap(f))
# because it cannot fuse operations across the batch dimension
```

### Recommended Pattern: jit(vmap(f))

For best performance, apply `jit` on the outside:

```python
import jax
import jax.numpy as jnp

@jax.jit
def batched_process(x_batch):
    # vmap inside JIT
    def process_single(x):
        return jnp.exp(x) / jnp.sum(jnp.exp(x))
    return jax.vmap(process_single)(x_batch)

x = jax.random.normal(jax.random.key(0), (128, 100))
result = batched_process(x)
```

---

## 9. Composing vmap with grad

vmap and grad compose to produce per-example gradients, which is essential for differential privacy and stochastic gradient estimation.

### Per-Example Gradients

```python
import jax
import jax.numpy as jnp

def loss_fn(params, x, y):
    pred = jnp.dot(params, x)
    return (pred - y) ** 2

params = jnp.array([1.0, 2.0, 3.0])
x_batch = jnp.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
y_batch = jnp.array([1.0, 2.0, 3.0])

# Gradient for each example individually
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
grads = per_example_grads(params, x_batch, y_batch)
# grads has shape (3, 3) -- one gradient vector per example
```

### vmap(grad) vs grad(vmap)

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.sum(x ** 2)

x_batch = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# vmap(grad(f)): gradient of each example separately
# Result shape: (3, 2) -- one gradient per example
per_grads = jax.vmap(jax.grad(f))(x_batch)

# grad(vmap(f)): gradient of the sum over the batch
# Result shape: (3, 2) -- but this is the gradient of sum_i(sum_j(x_ij^2))
total_grad = jax.grad(lambda x: jnp.sum(jax.vmap(f)(x)))(x_batch)
# Both give the same result in this case, but per_example is more useful
```

### Per-Example Gradients for Neural Networks

```python
import jax
import jax.numpy as jnp

def predict(params, x):
    """Single-example prediction"""
    for w, b in params:
        x = jnp.dot(x, w) + b
        x = jax.nn.relu(x)
    return x[-1]  # scalar output

def loss_fn(params, x, y):
    pred = predict(params, x)
    return (pred - y) ** 2

# Compute per-example gradients efficiently
per_example_grad = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))

# Initialize
key = jax.random.key(0)
params = [
    (jax.random.normal(key, (784, 256)) * 0.01, jnp.zeros(256)),
    (jax.random.normal(key, (256, 10)) * 0.01, jnp.zeros(10)),
]
x_batch = jax.random.normal(key, (32, 784))
y_batch = jax.random.normal(key, (32,))

grads = per_example_grad(params, x_batch, y_batch)
# grads is a pytree matching params structure, but with an extra batch dimension
# grads[0][0].shape = (32, 784, 256) -- per-example weight gradients
```

### Using value_and_grad with vmap

```python
import jax
import jax.numpy as jnp

def loss_fn(params, x, y):
    pred = jnp.dot(params, x)
    return jnp.sum((pred - y) ** 2)

params = jnp.ones(5)
x_batch = jnp.arange(30.0).reshape(6, 5)
y_batch = jnp.arange(6.0)

# Get both per-example losses and per-example gradients
per_example = jax.vmap(jax.value_and_grad(loss_fn), in_axes=(None, 0, 0))
losses, grads = per_example(params, x_batch, y_batch)
# losses.shape = (6,), grads.shape = (6, 5)
```

---

## 10. Nested vmap

Multiple `vmap` calls can be nested to vectorize over multiple dimensions. Each `vmap` adds one batch dimension.

### Pairwise Operations

```python
import jax
import jax.numpy as jnp

def distance(x1, x2):
    """Euclidean distance between two vectors"""
    return jnp.sqrt(jnp.sum((x1 - x2) ** 2))

# Double vmap for all pairwise distances
# Outer vmap: iterate over x1
# Inner vmap: iterate over x2
pairwise_distances = jax.vmap(
    jax.vmap(distance, in_axes=(None, 0)),  # fix x1, vectorize x2
    in_axes=(0, None)                         # vectorize x1, fix x2
)

points = jnp.array([[0.0, 0.0],
                     [1.0, 0.0],
                     [0.0, 1.0],
                     [1.0, 1.0]])
dist_matrix = pairwise_distances(points, points)
# dist_matrix[i, j] = distance between point i and point j
# shape: (4, 4)
```

### Triple vmap for Batched Pairwise

```python
import jax
import jax.numpy as jnp

def kernel(x1, x2):
    """RBF kernel between two vectors"""
    return jnp.exp(-jnp.sum((x1 - x2) ** 2) / 2.0)

# Batch of point sets: compute pairwise kernel for each set in the batch
batch_kernel = jax.vmap(                           # over batch
    jax.vmap(                                      # over first point set
        jax.vmap(kernel, in_axes=(None, 0)),       # over second point set
        in_axes=(0, None)
    ),
    in_axes=(0, 0)
)

batch_points1 = jax.random.normal(jax.random.key(0), (8, 5, 3))  # 8 sets, 5 points, 3 dims
batch_points2 = jax.random.normal(jax.random.key(1), (8, 4, 3))  # 8 sets, 4 points, 3 dims
result = batch_kernel(batch_points1, batch_points2)
# result.shape = (8, 5, 4) -- pairwise kernel matrix for each batch element
```

### Nested vmap with Different in_axes

```python
import jax
import jax.numpy as jnp

def apply_matrix(mat, vec):
    return mat @ vec

# Batch over matrices (axis 0) and vectors (axis 0) independently
batched_apply = jax.vmap(
    jax.vmap(apply_matrix, in_axes=(None, 0)),  # vectorize over vectors
    in_axes=(0, None)                             # vectorize over matrices
)

matrices = jnp.arange(12.0).reshape(3, 2, 2)  # 3 matrices of 2x2
vectors = jnp.arange(10.0).reshape(5, 2)       # 5 vectors of size 2
result = batched_apply(matrices, vectors)
# result.shape = (3, 5, 2) -- 3 matrices x 5 vectors, each output is size 2
```

---

## 11. vmap and Control Flow

vmap works correctly with JAX's structured control flow primitives. However, Python-level control flow that depends on array values requires special handling.

### vmap with lax.cond

```python
import jax
import jax.numpy as jnp

def abs_or_square(x, flag):
    # flag determines which branch to take
    return jax.lax.cond(
        flag,
        lambda x: jnp.abs(x),
        lambda x: x ** 2,
        x
    )

# vmap over x (axis 0) but NOT flag -- all elements use the same flag
vmap_fn = jax.vmap(abs_or_square, in_axes=(0, None))
x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
result_true = vmap_fn(x, True)   # absolute values: [2, 1, 0, 1, 2]
result_false = vmap_fn(x, False) # squares: [4, 1, 0, 1, 4]
```

### vmap with lax.select (Per-Element Control Flow)

```python
import jax
import jax.numpy as jnp

def piecewise(x, conditions):
    """Apply different functions based on per-element conditions"""
    return jax.lax.select(
        conditions,
        jnp.exp(x),       # where condition is True
        jnp.log1p(x),     # where condition is False
    )

vmap_fn = jax.vmap(piecewise, in_axes=(0, 0))
x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
conditions = jnp.array([[True, False], [True, True]])
result = vmap_fn(x, conditions)
```

### vmap with lax.scan

```python
import jax
import jax.numpy as jnp

def cumulative_sum(x_sequence):
    """Compute cumulative sum of a sequence"""
    def step(carry, x):
        new_carry = carry + x
        return new_carry, new_carry
    final, accum = jax.lax.scan(step, 0.0, x_sequence)
    return accum

# vmap over a batch of sequences
vmap_cumsum = jax.vmap(cumulative_sum, in_axes=0)
sequences = jnp.arange(12.0).reshape(3, 4)  # 3 sequences of length 4
result = vmap_cumsum(sequences)
# result.shape = (3, 4)
```

### vmap with lax.while_loop

```python
import jax
import jax.numpy as jnp

def newton_sqrt(x):
    """Newton's method for square root"""
    def cond(state):
        guess, _ = state
        return jnp.abs(guess * guess - x) > 1e-6

    def body(state):
        guess, iters = state
        return (0.5 * (guess + x / guess), iters + 1)

    init = (x, 0)
    final_guess, iters = jax.lax.while_loop(cond, body, init)
    return final_guess

# Vectorize over a batch of inputs
vmap_sqrt = jax.vmap(newton_sqrt)
x_batch = jnp.array([2.0, 4.0, 9.0, 16.0, 25.0])
result = vmap_sqrt(x_batch)
# Approximately: [1.414, 2.0, 3.0, 4.0, 5.0]
```

---

## 12. vmap with Pytrees

vmap naturally handles pytree inputs and outputs. The `in_axes` and `out_axes` can be pytrees matching the function's input/output structure.

### Vectorizing Over a Parameter Pytree

```python
import jax
import jax.numpy as jnp

def mlp_forward(params, x):
    """params is a list of (weight, bias) tuples"""
    h = x
    for w, b in params:
        h = jnp.dot(h, w) + b
        h = jax.nn.relu(h)
    return h

# Define in_axes as a pytree matching params structure
# None means the parameter is shared across the batch
in_axes_params = [(None, None)] * 2  # all params shared

vmap_mlp = jax.vmap(mlp_forward, in_axes=(in_axes_params, 0))

params = [
    (jnp.ones((4, 3)), jnp.zeros(3)),
    (jnp.ones((3, 2)), jnp.zeros(2)),
]
x_batch = jnp.ones((10, 4))
result = vmap_mlp(params, x_batch)  # shape (10, 2)
```

### Batched Params and Batched Inputs

```python
import jax
import jax.numpy as jnp

def model_forward(params, x):
    return params["w"] @ x + params["b"]

# Batch over both params and inputs
vmap_fn = jax.vmap(
    model_forward,
    in_axes=(
        {"w": 0, "b": 0},  # batch of parameters
        0,                   # batch of inputs
    )
)

batch_params = {
    "w": jnp.arange(20.0).reshape(5, 3, 4),  # 5 different weight matrices
    "b": jnp.arange(20.0).reshape(5, 4),       # 5 different biases
}
batch_x = jnp.ones((5, 3))
result = vmap_fn(batch_params, batch_x)  # shape (5, 4)
```

---

## 13. vmap Performance Considerations

### Memory vs Compute Tradeoff

vmap generates a single compiled program that processes the entire batch. This is generally more efficient than a Python loop but uses more memory.

```python
import jax
import jax.numpy as jnp

# Bad: Python loop (no fusion, poor GPU utilization)
def slow_batch(fn, x_batch):
    return jnp.stack([fn(x) for x in x_batch])

# Good: vmap (single kernel, memory-efficient)
def fast_batch(fn, x_batch):
    return jax.vmap(fn)(x_batch)
```

### Chunking for Large Batches

When the batch is too large to fit in memory, manually chunk it:

```python
import jax
import jax.numpy as jnp

def process(x):
    return jnp.sum(x ** 2)

vmapped = jax.vmap(process)
jitted = jax.jit(vmapped)

def chunked_vmap(fn, x, chunk_size=1024):
    """Process x in chunks to avoid OOM"""
    results = []
    for i in range(0, x.shape[0], chunk_size):
        chunk = x[i:i + chunk_size]
        results.append(fn(chunk))
    return jnp.concatenate(results, axis=0)

x_large = jax.random.normal(jax.random.key(0), (10000, 100))
result = chunked_vmap(jitted, x_large, chunk_size=2048)
```

### vmap vs Manual Vectorization Performance

```python
import jax
import jax.numpy as jnp

@jax.jit
def manual_batch(x_batch):
    # Manual vectorization using broadcasting
    return jnp.sum(x_batch ** 2, axis=-1)

@jax.jit
def vmap_batch(x_batch):
    return jax.vmap(lambda x: jnp.sum(x ** 2))(x_batch)

# Both produce the same result with similar performance
# for simple operations. vmap may be slower for ops that
# already broadcast naturally, but is essential for complex logic.
x = jax.random.normal(jax.random.key(0), (1000, 100))
assert jnp.allclose(manual_batch(x), vmap_batch(x))
```

### Avoid Unnecessary vmap

For operations that already support batching via broadcasting, vmap adds overhead:

```python
import jax
import jax.numpy as jnp

# Unnecessary vmap for simple element-wise ops
def square(x):
    return x ** 2

x = jnp.arange(10.0)

# These are equivalent, but the first is faster
result1 = square(x)                  # Direct broadcasting -- fast
result2 = jax.vmap(square)(x)       # vmap overhead -- slightly slower
```

---

## 14. Limitations and When vmap May Not Work

### Dynamic Shapes

vmap requires that all batch elements produce outputs with the same shape. Functions whose output shape depends on the input values will fail.

```python
import jax
import jax.numpy as jnp

# This WILL NOT work with vmap
def dynamic_slice(x):
    """Output shape depends on input value -- cannot vmap"""
    n = x[0].astype(int)
    return x[:n]

# This raises an error because different batch elements may produce
# different output shapes
try:
    result = jax.vmap(dynamic_slice)(jnp.array([[3, 1, 2], [2, 4, 5]]))
except Exception as e:
    print(f"Error: {e}")
```

### Python-Level Control Flow on Array Values

```python
import jax
import jax.numpy as jnp

# This WILL NOT work with vmap
def python_conditional(x):
    if x > 0:  # Python if on traced value -- fails under vmap
        return x
    else:
        return -x

# Use jnp.where or lax.cond instead:
def correct_conditional(x):
    return jnp.where(x > 0, x, -x)

result = jax.vmap(correct_conditional)(jnp.array([-1.0, 2.0, -3.0]))
```

### Variable-Length Sequences

vmap cannot handle variable-length sequences without padding or masking:

```python
import jax
import jax.numpy as jnp

# For variable-length sequences, use masking
def masked_mean(x, mask):
    return jnp.sum(x * mask) / jnp.sum(mask)

x_padded = jnp.array([[1.0, 2.0, 3.0, 0.0],   # effective length 3
                       [4.0, 5.0, 0.0, 0.0]])   # effective length 2
masks = jnp.array([[1.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0]])

result = jax.vmap(masked_mean)(x_padded, masks)
# result = [2.0, 4.5]
```

### Side Effects

vmap requires pure functions. Side effects (printing, mutation, I/O) do not work as expected:

```python
import jax
import jax.numpy as jnp

# Side effects don't work properly with vmap
def bad_fn(x):
    # This print may execute only once (during tracing) or not at all
    print(f"Processing {x}")
    return x * 2

# Use jax.debug.print for debugging inside vmap
def good_fn(x):
    jax.debug.print("Processing {}", x)
    return x * 2
```

---

## 15. custom_vmap

`jax.custom_vmap` allows you to define custom batching behavior for a function, which is useful when the default vmap transformation does not produce the desired result or when you want to optimize the batched version.

### Basic custom_vmap

```python
import jax
import jax.numpy as jnp

@jax.custom_vmap
def custom_elementwise(x):
    # The original (unbatched) function
    return jnp.sin(x) + jnp.cos(x)

@custom_elementwise.def_vmap
def custom_elementwise_vmap(axis_size, in_batched, x):
    # Custom vectorized implementation
    # axis_size: size of the batch dimension
    # in_batched: tuple of bools indicating which args are batched
    # x: the (possibly batched) input
    if in_batched[0]:
        # Efficient custom batched implementation
        return jnp.sin(x) + jnp.cos(x)
    else:
        return jnp.sin(x) + jnp.cos(x)

result = custom_elementwise(jnp.arange(5.0))
batched_result = jax.vmap(custom_elementwise)(jnp.arange(6.0).reshape(2, 3))
```

### custom_vmap for Optimized Batched Operations

```python
import jax
import jax.numpy as jnp

@jax.custom_vmap
def my_softmax(x):
    """Numerically stable softmax for a single vector"""
    x_max = jnp.max(x)
    exp_x = jnp.exp(x - x_max)
    return exp_x / jnp.sum(exp_x)

@my_softmax.def_vmap
def my_softmax_vmap(axis_size, in_batched, x):
    """Custom batched softmax using jax.nn.softmax directly"""
    if in_batched[0]:
        # Use the optimized batched version
        return jax.nn.softmax(x, axis=-1)
    else:
        x_max = jnp.max(x)
        exp_x = jnp.exp(x - x_max)
        return exp_x / jnp.sum(exp_x)

x_batch = jax.random.normal(jax.random.key(0), (32, 100))
result = jax.vmap(my_softmax)(x_batch)  # Uses the custom def_vmap
```

### custom_vmap with Multiple Arguments

```python
import jax
import jax.numpy as jnp

@jax.custom_vmap
def weighted_normalize(x, weight):
    return x * weight / jnp.linalg.norm(x * weight)

@weighted_normalize.def_vmap
def weighted_normalize_vmap(axis_size, in_batched, x, weight):
    x_b, w_b = in_batched
    if x_b and w_b:
        # Both batched: normalize each independently
        xw = x * weight
        norms = jnp.linalg.norm(xw, axis=-1, keepdims=True)
        return xw / norms
    elif x_b:
        # Only x batched
        xw = x * weight
        return xw / jnp.linalg.norm(xw, axis=-1, keepdims=True)
    else:
        return x * weight / jnp.linalg.norm(x * weight)

x_batch = jax.random.normal(jax.random.key(0), (5, 3))
w_batch = jnp.ones((5, 3))
result = jax.vmap(weighted_normalize)(x_batch, w_batch)
```

---

## 16. Complete Examples

### Example 1: Batched Matrix Operations

```python
import jax
import jax.numpy as jnp

def solve_triangular_single(A, b):
    """Solve a single triangular system Ax = b using substitution"""
    n = A.shape[0]
    x = jnp.zeros(n)
    def step(i, x):
        row = A[i]
        x_i = (b[i] - jnp.dot(row, x)) / row[i]
        return x.at[i].set(x_i)
    return jax.lax.fori_loop(0, n, step, x)

# Batch over multiple systems
batched_solve = jax.vmap(solve_triangular_single, in_axes=(0, 0))

# Create batch of upper triangular systems
key = jax.random.key(42)
A_batch = jax.random.uniform(key, (10, 5, 5))
A_batch = jnp.triu(A_batch) + jnp.eye(5)  # Make sure diagonal is nonzero
b_batch = jax.random.uniform(key, (10, 5))

solutions = batched_solve(A_batch, b_batch)  # shape (10, 5)
```

### Example 2: Batched Attention Mechanism

```python
import jax
import jax.numpy as jnp

def single_head_attention(q, k, v):
    """Scaled dot-product attention for single query, single head"""
    d_k = q.shape[-1]
    scores = jnp.dot(q, k.T) / jnp.sqrt(d_k)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.dot(weights, v)

# Batch over queries
batched_attention = jax.vmap(single_head_attention, in_axes=(0, None, None))

q = jax.random.normal(jax.random.key(0), (8, 64))   # 8 queries, dim 64
k = jax.random.normal(jax.random.key(1), (16, 64))   # 16 keys
v = jax.random.normal(jax.random.key(2), (16, 64))   # 16 values

output = batched_attention(q, k, v)  # shape (8, 64)
```

### Example 3: Batched Gradient Computation

```python
import jax
import jax.numpy as jnp
import optax

def loss_fn(params, x, y):
    pred = params["w"] @ x + params["b"]
    return jnp.mean((pred - y) ** 2)

# Compute per-example gradients
per_example_grad = jax.vmap(
    jax.grad(loss_fn),
    in_axes=(None, 0, 0)  # params shared, x and y batched
)

# Compute clipped gradients (for differential privacy)
def privatized_grad_update(params, x_batch, y_batch, key, clip_norm=1.0):
    grads = per_example_grad(params, x_batch, y_batch)
    # Clip per-example gradients
    grad_norms = jax.tree.map(
        lambda g: jnp.sum(g ** 2, axis=tuple(range(1, g.ndim)), keepdims=True),
        grads
    )
    global_norm = jnp.sqrt(
        sum(jnp.sum(n) for n in jax.tree.leaves(grad_norms))
    )
    scale = jnp.minimum(1.0, clip_norm / (global_norm + 1e-8))
    clipped_grads = jax.tree.map(lambda g: g * scale[:, None], grads)
    # Average and add noise
    avg_grads = jax.tree.map(lambda g: jnp.mean(g, axis=0), clipped_grads)
    noise = jax.tree.map(
        lambda g: jax.random.normal(key, g.shape) * 0.01,
        avg_grads
    )
    return jax.tree.map(lambda g, n: g + n, avg_grads, noise)
```

### Example 4: Monte Carlo Integration

```python
import jax
import jax.numpy as jnp

def integrate_mc(fn, low, high, num_samples, key):
    """Monte Carlo integration of fn over [low, high]"""
    keys = jax.random.split(key, num_samples)
    samples = jax.random.uniform(keys, shape=(num_samples,), minval=low, maxval=high)
    values = jax.vmap(fn)(samples)
    return (high - low) * jnp.mean(values)

# Integrate sin(x) from 0 to pi
def sin_fn(x):
    return jnp.sin(x)

key = jax.random.key(0)
result = integrate_mc(sin_fn, 0.0, jnp.pi, 10000, key)
# Should be approximately 2.0
print(f"Integral of sin from 0 to pi: {result:.4f}")
```

### Example 5: Batched Optimization (Newton's Method)

```python
import jax
import jax.numpy as jnp

def find_root(f, x0, tol=1e-6, max_steps=50):
    """Find root of f using Newton's method"""
    def step(x):
        fx = f(x)
        dfx = jax.grad(f)(x)
        return x - fx / dfx

    def cond(state):
        x, i = state
        return (jnp.abs(f(x)) > tol) & (i < max_steps)

    def body(state):
        x, i = state
        return step(x), i + 1

    x_final, _ = jax.lax.while_loop(cond, body, (x0, 0))
    return x_final

# Find roots for multiple starting points
batched_root = jax.vmap(lambda x0: find_root(lambda x: x ** 3 - 2.0, x0))
x0s = jnp.array([0.5, 1.0, 1.5, 2.0, 3.0])
roots = batched_root(x0s)
# All should be approximately 2^(1/3) = 1.2599...
```

### Example 6: Image Processing Batch

```python
import jax
import jax.numpy as jnp

def apply_gaussian_blur(image, kernel_size=5, sigma=1.0):
    """Apply Gaussian blur to a single 2D image"""
    # Create 1D Gaussian kernel
    coords = jnp.arange(kernel_size) - kernel_size // 2
    kernel_1d = jnp.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    h, w = image.shape
    # Pad image
    pad = kernel_size // 2
    padded = jnp.pad(image, pad, mode="reflect")

    # Horizontal blur
    def blur_h(row):
        def slide(i):
            patch = row[i:i + kernel_size]
            return jnp.dot(patch, kernel_1d)
        return jax.lax.map(slide, jnp.arange(w))

    blurred = jax.vmap(blur_h)(padded)

    # Vertical blur
    blurred = jax.vmap(blur_h, in_axes=1, out_axes=1)(blurred)
    return blurred[:h, :w]

# Batch over multiple images
batched_blur = jax.vmap(apply_gaussian_blur)

images = jax.random.normal(jax.random.key(0), (8, 64, 64))
blurred_images = batched_blur(images)  # shape (8, 64, 64)
```

---

## Summary

| Feature | Support Level |
|---------|---------------|
| Basic array operations | Full |
| Pytree inputs/outputs | Full |
| Composition with jit | Full |
| Composition with grad | Full |
| Nested vmap | Full |
| lax.cond | Full |
| lax.while_loop | Full |
| lax.scan | Full |
| lax.fori_loop | Full |
| Python if/else on array values | Not supported (use lax.cond/jnp.where) |
| Dynamic output shapes | Not supported |
| Side effects | Not supported |
| Variable-length sequences | Requires masking |
| custom_vmap | Full |
