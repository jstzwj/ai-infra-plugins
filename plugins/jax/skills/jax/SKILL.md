---
name: jax
description: >
  Comprehensive reference documentation and skill for JAX - Google's library for
  high-performance numerical computing and machine learning research. Covers JAX core
  (transformations, tracing, jaxprs, pytrees), jax.numpy, jax.lax, jax.nn, jax.random,
  automatic differentiation (grad, custom_jvp, custom_vjp, checkpoint/remat),
  automatic vectorization (vmap), parallel computing (pjit, shard_map, Mesh, Sharding),
  jax.scipy, jax.image, Pallas GPU/TPU kernels, export, FFI, debugging, profiling,
  distributed computing, experimental features, and JAX Enhancement Proposals (JEPs).
version: 0.6.1
---

# JAX - High-Performance Numerical Computing and Machine Learning Research

## Overview

JAX is Google's library for high-performance numerical computing and machine learning research. It provides a familiar NumPy-style API with composable function transformations that enable just-in-time compilation to CPU, GPU, and TPU backends via XLA (Accelerated Linear Algebra).

JAX extends the NumPy API with four key transformations that compose freely:

1. **`jax.jit`** - Just-in-time compilation for accelerating JAX programs on accelerators
2. **`jax.grad`** - Automatic differentiation for computing gradients of arbitrary functions
3. **`jax.vmap`** - Automatic vectorization for batching without rewriting functions
4. **`jax.pjit`** - Parallel computation across multiple devices with data sharding

**Supported Hardware:** CPU, NVIDIA GPUs (CUDA 12+), AMD GPUs (ROCm), Google TPUs, Apple Metal (experimental), Intel GPUs (experimental)

**Supported Platforms:** Linux, macOS, Windows (CPU only)

**JAX Version:** 0.6.1 | **jaxlib Version:** 0.6.1 | **XLA Version:** OpenXLA

## Key Architecture Concepts

- **Transformations**: Composable function transforms (jit, grad, vmap, pjit) that work on pure functions
- **Tracing**: Mechanism by which JAX inspects Python functions to build intermediate representations
- **Jaxpr**: JAX's intermediate language (JAX expression) representing traced computations
- **Pytrees**: Tree-like structures of nested containers (dicts, lists, tuples) that JAX can operate on
- **XLA**: Accelerated Linear Algebra compiler that generates optimized hardware code
- **Sharding**: Describing how data is distributed across devices (GSPMD, shard_map)
- **Pallas**: Low-level kernel language for writing custom GPU/TPU kernels within JAX
- **Export**: Ahead-of-time lowering and export of JAX programs to stable artifacts

## Architecture Overview

```
+--------------------------------------------------------------+
|                    Python Frontend                            |
|  jax.numpy  |  jax.lax  |  jax.nn  |  jax.random  |  ...    |
+--------------------------------------------------------------+
|                 Function Transformations                       |
|  jit  |  grad  |  vmap  |  pjit  |  checkpoint  |  ...       |
+--------------------------------------------------------------+
|                    Tracing Engine                             |
|  jax.core.Trace  |  jax.core.Tracer  |  jax.core.Jaxpr      |
+--------------------------------------------------------------+
|                    XLA Compiler                               |
|  HLO -> MHLO -> Linalg -> Hardware-specific optimizations    |
+--------------------------------------------------------------+
|                    Runtime Backends                           |
|  CPU (LLVM)  |  NVIDIA GPU (PTX)  |  AMD GPU  |  TPU        |
+--------------------------------------------------------------+
```

## Quick Reference

### Installation

```bash
# CPU only
pip install jax

# NVIDIA GPU (CUDA 12)
pip install jax[cuda12]

# NVIDIA GPU (CUDA 12 + latest jaxlib nightly)
pip install -U jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html

# TPU (Google Cloud)
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Minimal Example

```python
import jax
import jax.numpy as jnp

# JIT-compiled function
@jax.jit
def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

# Automatic differentiation
@jax.grad
def loss_fn(params, x, y):
    pred = params["w"] @ x + params["b"]
    return jnp.mean((pred - y) ** 2)

# Automatic vectorization
batched_selu = jax.vmap(selu)

# Random numbers (explicit PRNG state)
key = jax.random.PRNGKey(42)
x = jax.random.normal(key, (3, 3))

# Compute gradients
params = {"w": jnp.ones((2, 3)), "b": jnp.zeros(2)}
grads = loss_fn(params, jnp.ones((3,)), jnp.ones((2,)))
```

### Basic Array Operations

```python
import jax.numpy as jnp

# Array creation
x = jnp.array([1.0, 2.0, 3.0])
z = jnp.zeros((3, 4))
o = jnp.ones((2, 3))
r = jnp.arange(0, 10, 2)       # [0, 2, 4, 6, 8]
l = jnp.linspace(0, 1, 5)      # [0.0, 0.25, 0.5, 0.75, 1.0]
e = jnp.eye(3)                  # Identity matrix

# Math operations
a = jnp.dot(x, x)               # Dot product
b = jnp.matmul(z, z.T)          # Matrix multiply
c = jnp.sum(x)                   # Reduction
d = jnp.mean(x, axis=0)          # Mean along axis

# Indexing (same as NumPy)
m = jnp.arange(12).reshape(3, 4)
row = m[1, :]                    # Second row
col = m[:, 2]                    # Third column
sub = m[0:2, 1:3]                # Sub-matrix

# IMPORTANT: JAX arrays are immutable
# m[0, 0] = 99  # ERROR! Use functional update instead:
m = m.at[0, 0].set(99)
```

### Composable Transformations

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.sum(x ** 2)

# Compose transformations freely
f_jit = jax.jit(f)                    # Compiled
f_grad = jax.grad(f)                   # Gradient
f_vmap = jax.vmap(f)                   # Vectorized
f_jit_grad = jax.jit(jax.grad(f))     # Compiled gradient
f_grad_vmap = jax.vmap(jax.grad(f))   # Per-sample gradients
f_jit_grad_vmap = jax.jit(jax.vmap(jax.grad(f)))  # All three
```

### Training Loop

```python
import jax
import jax.numpy as jnp
import optax  # Optimizer library

# Model as a pure function
def init_params(key, layer_sizes):
    params = []
    for i in range(len(layer_sizes) - 1):
        key, k1, k2 = jax.random.split(key, 3)
        w = jax.random.normal(k1, (layer_sizes[i], layer_sizes[i + 1])) * 0.01
        b = jax.random.normal(k2, (layer_sizes[i + 1],)) * 0.01
        params.append({"w": w, "b": b})
    return params

def predict(params, x):
    for layer in params[:-1]:
        x = jnp.dot(x, layer["w"]) + layer["b"]
        x = jax.nn.relu(x)
    x = jnp.dot(x, params[-1]["w"]) + params[-1]["b"]
    return x

def loss_fn(params, x, y):
    pred = predict(params, x)
    return jnp.mean(optax.l2_loss(pred, y).sum())

@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Initialize
key = jax.random.PRNGKey(0)
params = init_params(key, [784, 256, 10])
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

# Training loop
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        params, opt_state, loss = train_step(params, opt_state, batch_x, batch_y)
```

### Distributed Computing

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

# Define device mesh
devices = jax.devices()
mesh = Mesh(devices.reshape((len(devices), 1)), ("data", "model"))

# Shard data across devices
from jax.sharding import NamedSharding
sharding = NamedSharding(mesh, P("data", None))
x = jax.random.normal(jax.random.PRNGKey(0), (1024, 784))
x_sharded = jax.device_put(x, sharding)

# Use pjit for distributed computation
@jax.jit
def distributed_train_step(params, x_sharded):
    ...
```

### Pytrees

```python
import jax

# Pytrees are tree-like structures of containers
params = {
    "layer1": {"w": jnp.ones((3, 4)), "b": jnp.zeros(4)},
    "layer2": {"w": jnp.ones((4, 2)), "b": jnp.zeros(2)},
}

# Flatten and unflatten
flat_values, tree_def = jax.tree.flatten(params)
params_restored = jax.tree.unflatten(tree_def, flat_values)

# Apply a function to every leaf
scaled = jax.tree.map(lambda x: x * 2, params)

# Tree utilities
struct = jax.tree.map(lambda x: jnp.zeros_like(x), params)  # Same structure, zero values
```

## Reference Chapters

### Core Concepts

1. [Overview and Architecture](references/01-overview-and-architecture.md) - What is JAX, design philosophy, architecture layers, package structure, and ecosystem
2. [Installation and Setup](references/02-installation-and-setup.md) - CPU/GPU/TPU installation, Docker, nightly builds, verification, environment configuration
3. [Key Concepts](references/03-key-concepts.md) - Transformations, tracing, jaxprs, pytrees, functional programming model, pure functions
4. [JIT Compilation](references/04-jit-compilation.md) - jax.jit, compilation caching, tracing mechanics, static arguments, control flow restrictions
5. [Automatic Differentiation](references/05-automatic-differentiation.md) - jax.grad, forward-mode (jvp), reverse-mode (vjp), higher-order derivatives, stop_gradient
6. [Advanced Automatic Differentiation](references/06-advanced-autodiff.md) - Custom derivative rules, perturbations, tangents, cotangents, complex number differentiation
7. [Automatic Vectorization (vmap)](references/07-automatic-vectorization.md) - jax.vmap, batching semantics, spmd_axis_name, in_axes/out_axes, nested vmap
8. [Control Flow](references/08-control-flow.md) - jax.lax.cond, jax.lax.while_loop, jax.lax.fori_loop, jax.lax.scan, jax.lax.switch, Python control flow in traced code
9. [Pytrees](references/09-pytrees.md) - Tree flattening/unflattening, custom pytree nodes, tree_map, tree utilities, common patterns
10. [Random Numbers](references/10-random-numbers.md) - PRNG design, explicit state, jax.random module, Threefry/philox PRNG, key splitting

### Numerical Libraries

11. [jax.lax - Linear Algebra Extensions](references/11-jax-lax.md) - Low-level operators, convolution, reduce_window, scatter/gather, clamp, sorting, platform-specific ops
12. [jax.numpy - NumPy API](references/12-jax-numpy.md) - Complete NumPy-compatible API, differences from NumPy, indexing, broadcasting, ufuncs
13. [jax.nn - Neural Network Functions](references/13-jax-nn.md) - Activations (relu, gelu, silu, softmax, etc.), initializers, one_hot, standardize
14. [jax.scipy - SciPy API](references/14-jax-scipy.md) - Linear algebra, optimization, special functions, signal processing, stats functions
15. [jax.image - Image Processing](references/15-jax-image.md) - Resize, affine transforms, scaling functions, interpolation methods
16. [jax.random - Random Number Generation](references/16-jax-random.md) - PRNGKey, split, normal, uniform, categorical, permutation, bernoulli, beta, gamma, distributions

### Core Internals

17. [Core Types and Abstract Values](references/17-core-types.md) - ShapedArray, DShapedArray, abstract evaluation, shaped abstract values, bond dimensions
18. [Tracing and Jaxpr Internals](references/18-tracing-jaxpr.md) - Trace/Tracer objects, jaxpr language, eqns, vars, constants, evaluation, pretty-printing
19. [Custom Derivatives (custom_jvp, custom_vjp)](references/19-custom-derivatives.md) - Defining custom differentiation rules, custom_jvp, custom_vjp, nondiff_argnums, bypassing AD
20. [Gradient Checkpointing and Rematerialization](references/20-checkpoint-remat.md) - jax.checkpoint (remat), policy-based rematerialization, memory-compute tradeoffs, scan checkpointing

### Parallel and Distributed Computing

21. [Sharding and Distributed Computing](references/21-sharding-distributed.md) - NamedSharding, PositionalSharding, GSPMD, device_put, Mesh, multi-host, distributed arrays
22. [shard_map](references/22-shard-map.md) - Per-device computation, collective operations, explicit communication, SPMD vs manual sharding
23. [Pallas - GPU and TPU Kernels](references/23-pallas-overview.md) - Low-level kernel language, grid programming model, memory spaces, Pallas primitives
24. [Pallas GPU Programming](references/24-pallas-gpu.md) - GPU-specific Pallas features, WMMA, VMEM, SMEM, HBM, async copies, barriers, TMA
25. [Pallas TPU Programming](references/25-pallas-tpu.md) - TPU-specific Pallas features, VMEM, SEM, dot primitives, matmul fusion, systolic array

### Debugging and Performance

26. [Debugging and Error Handling](references/26-debugging-errors.md) - jax.debug.print, jax.debug.breakpoint, checkify, common errors, NaN debugging, shape errors
27. [Profiling and Performance](references/27-profiling-performance.md) - jax.profiler, TensorBoard integration, memory profiling, timeline analysis, performance best practices
28. [GPU Performance Tips](references/28-gpu-performance.md) - Kernel fusion, memory layout, async dispatch, data transfer optimization, kernel launch overhead

### Compilation and Export

29. [Ahead-of-Time Compilation (AOT)](references/29-aot-compilation.md) - jax.jit with static args, lowering, compilation, AOT lowering and compilation APIs
30. [Export and jax2tf](references/30-export-jax2tf.md) - jax.export, StableHLO export, jax2tf conversion, TensorFlow SavedModel, cross-platform deployment
31. [Foreign Function Interface (FFI)](references/31-ffi.md) - Calling C/C++ code from JAX, jax.extend.ffi, XLA custom calls, registration, lowering rules
32. [External Callbacks](references/32-external-callbacks.md) - jax.debug.print, jax.debug.breakpoint, jax.pure_callback, jax.io_callback, host callbacks

### Extension and Advanced Topics

33. [Building on JAX (Extensions)](references/33-building-on-jax.md) - Creating JAX extensions, custom interpreters, custom traverse rules, library development patterns
34. [Type Promotion](references/34-type-promotion.md) - Type promotion rules, NumPy compatibility, weak types, dtype behavior in transformations
35. [Data Types (dtypes)](references/35-dtypes.md) - float16, bfloat16, float32, float64, int8, int32, uint32, complex64, extended precision, custom dtypes
36. [Stateful Computations and Effects](references/36-state-effects.md) - Handling state in a functional framework, jax.extend, effect systems, side-effect management
37. [Checkify - Functional Error Checking](references/37-checkify.md) - jax.checkify, functional error handling, assert-enable transforms, error propagation in JIT
38. [JAX Enhancement Proposals (JEPs)](references/38-jeps.md) - Design documents for major features, JEP process, historical context, evolution of JAX
39. [Configuration and Environment Variables](references/39-configuration.md) - jax.config, JAX_PLATFORMS, JAX_TRACEBACK_FILTERING, XLA_FLAGS, all config options
40. [Common Patterns and Gotchas](references/40-patterns-gotchas.md) - Stateful patterns, in-place updates, array mutation, device placement, async dispatch, common pitfalls

### Extension API and Experimental Features

41. [jax.extend - Extension API](references/41-jax-extend.md) - Public extension API, custom JAXpr interpreters, interop with JAX internals, stable extension points
42. [Experimental Features](references/42-experimental.md) - jax.experimental modules, Array, sharding extensions, custom mesh utils, work in progress features
43. [Sparse Operations](references/43-sparse.md) - jax.experimental.sparse, BCOO format, sparse matrix operations, sparse autodiff, sparse Pallas
44. [Training Cookbook](references/44-training-cookbook.md) - End-to-end training recipes, multi-GPU training, mixed precision, checkpointing, data loading patterns
45. [Benchmarking](references/45-benchmarking.md) - Benchmarking JAX code, BlockUntilReady, timing best practices, throughput measurement, comparison techniques

## Key APIs Quick Reference

### Transformations

| Transform | Purpose | Example |
|-----------|---------|---------|
| `jax.jit` | Just-in-time compilation | `jax.jit(f)(x)` or `@jax.jit` |
| `jax.grad` | Reverse-mode differentiation | `jax.grad(f)(x)` returns df/dx |
| `jax.jacfwd` | Forward-mode Jacobian | `jax.jacfwd(f)(x)` |
| `jax.jacrev` | Reverse-mode Jacobian | `jax.jacrev(f)(x)` |
| `jax.hessian` | Hessian matrix | `jax.hessian(f)(x)` |
| `jax.vmap` | Automatic vectorization | `jax.vmap(f)(batch_x)` |
| `jax.pmap` | Parallel map (legacy) | `jax.pmap(f)(sharded_x)` |
| `jax.value_and_grad` | Value + gradient | `loss, grads = jax.value_and_grad(f)(x)` |
| `jax.checkpoint` | Gradient checkpointing | `jax.checkpoint(f)(x)` or `@jax.checkpoint` |
| `jax.custom_jvp` | Custom forward-mode rule | Decorator for custom AD |
| `jax.custom_vjp` | Custom reverse-mode rule | Decorator for custom AD |
| `jax.linearize` | Forward-mode AD | `y, f_jvp = jax.linearize(f, x)` |
| `jax.vjp` | Reverse-mode AD | `y, f_vjp = jax.vjp(f, x)` |

### Array Creation

```python
import jax.numpy as jnp

jnp.array([1, 2, 3])           # From Python list
jnp.zeros((3, 4))              # Zeros
jnp.ones((2, 3))               # Ones
jnp.eye(3)                     # Identity matrix
jnp.arange(0, 10, 2)           # Range: [0, 2, 4, 6, 8]
jnp.linspace(0, 1, 5)          # 5 evenly-spaced values
jnp.full((2, 3), 7.0)          # Filled with value
jnp.zeros_like(x)              # Same shape/dtype, zeros
jnp.ones_like(x)               # Same shape/dtype, ones
jnp.empty((2, 3))              # Uninitialized (do not rely on values)
```

### Common Neural Network Operations

```python
import jax.nn as nn
import jax.numpy as jnp

# Activations
nn.relu(x)                     # ReLU
nn.gelu(x)                     # GELU
nn.silu(x)                     # SiLU (Swish)
nn.softmax(x, axis=-1)         # Softmax
nn.log_softmax(x, axis=-1)     # Log-softmax (numerically stable)
nn.sigmoid(x)                  # Sigmoid
nn.tanh(x)                     # Tanh
nn.elu(x)                      # ELU
nn.leaky_relu(x)               # Leaky ReLU
nn.one_hot(x, 10)              # One-hot encoding

# Initializers
import jax.nn.initializers as init
key = jax.random.PRNGKey(0)
w = init.kaiming_normal()(key, (256, 128))
b = init.zeros(key, (128,))
```

### Device Management

```python
import jax

# Query devices
jax.devices()                   # All available devices
jax.devices("cpu")              # CPU devices
jax.devices("gpu")              # GPU devices
jax.devices("tpu")              # TPU devices
jax.local_devices()             # Local devices only
jax.device_count()              # Number of devices
jax.local_device_count()        # Number of local devices

# Data placement
x = jax.device_put(x, jax.devices()[0])   # Move to specific device
x = jax.device_put(x, sharding)            # Shard across devices
y = jax.device_get(x)                      # Copy to host (numpy)

# Async dispatch
x = jnp.ones(1000)
y = jnp.dot(x, x)
y.block_until_ready()           # Wait for computation to finish
```

## Important Differences from NumPy

1. **Immutable arrays**: JAX arrays are immutable; use `.at[].set()` for functional updates
2. **PRNG state**: JAX uses explicit PRNG state (no global random state); pass `key` everywhere
3. **64-bit disabled by default**: Use `jax.config.update("jax_enable_x64", True)` to enable float64
4. **Pure functions**: Transformations require pure functions (no side effects)
5. **Control flow**: Use `jax.lax.cond`, `jax.lax.while_loop`, `jax.lax.scan` inside JIT
6. **Async dispatch**: Operations are asynchronous; use `.block_until_ready()` for timing

## JAX Ecosystem

| Library | Description |
|---------|-------------|
| **Flax** | Neural network library with Linen module system |
| **Optax** | Gradient processing and optimization library |
| **Haiku** | Sonnet-style neural network library |
| **Diffrax** | Differential equation solvers in JAX |
| **Equinox** | Neural networks via elegant Pytree manipulations |
| **Chex** | Testing utilities and dataclass extensions |
| **DM-Haiku** | DeepMind's neural network library |
| **JAXopt** | Hardware-accelerated optimization library |
| **Orbax** | Checkpointing and export utilities |
| **T5X** | T5 model implementation in JAX/Flax |
| **MaxText** | Large-scale Transformer training in pure JAX |
| **Levanter** | Scalable LLM training with JAX |
| **Penzai** | JAX model visualization and manipulation |
| **Keras 3** | Multi-backend Keras with JAX support |
| **NumPyro** | Probabilistic programming with JAX |
| **JAX-Cosmo** | Cosmological computations |
| **Oryx** | Probabilistic programming and bijectors |

## Version and Compatibility

- **JAX**: 0.6.1
- **Minimum Python**: 3.10+
- **NumPy**: 1.24+
- **Supported CUDA**: 12.x
- **Supported ROCm**: 6.x
- **XLA**: OpenXLA (StableHLO)
