# 01 - Overview and Architecture

## What is JAX

JAX is a high-performance numerical computing library developed by Google for machine learning research and general-purpose scientific computing. It provides a NumPy-compatible API with composable function transformations that enable automatic differentiation, just-in-time compilation, and parallel execution across CPUs, GPUs, and TPUs.

JAX stands for **J**ust **A**fter e**X**ecution (though this is more a backronym than a strict definition). The library was created by the Google Brain team (now Google DeepMind) as a successor to the Autograd project, combining Autograd's automatic differentiation capabilities with XLA (Accelerated Linear Algebra) compilation for high performance on accelerators.

The central insight of JAX is that many useful numerical computing transformations -- differentiation, vectorization, parallelization, compilation -- can be expressed as function transformations on pure Python functions. By requiring functions to be pure (no side effects, no mutation), JAX can safely transform and compose these operations in arbitrary order.

### Key Capabilities

- **Automatic differentiation**: Forward-mode (jvp) and reverse-mode (vjp/grad) automatic differentiation of arbitrary Python/NumPy functions, including higher-order derivatives
- **Just-in-time compilation**: XLA-based JIT compilation that fuses operations, optimizes memory layout, and generates efficient code for CPUs, GPUs, and TPUs
- **Automatic vectorization**: The vmap transform automatically adds batch dimensions without rewriting functions
- **Parallelization**: pjit and shard_map enable data-parallel and model-parallel computation across multiple devices
- **NumPy compatibility**: The jax.numpy API mirrors NumPy, making it easy to port existing code

## Design Philosophy

### Composable Transformations

The foundational design principle of JAX is that transformations are composable. Each transformation takes a function and returns a new transformed function. Because the result is itself a function, you can apply another transformation to it:

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.sum(x ** 2)

# Compose freely: jit(grad(vmap(f)))
composed = jax.jit(jax.grad(jax.vmap(f)))

# Or equivalently with decorators:
@jax.jit
@jax.grad
def g(x):
    return jnp.sum(x ** 2)
```

This compositionality means you do not need separate APIs for "compiled gradients" or "vectorized compiled gradients" -- you simply compose the base transformations. This design is inspired by functional programming principles, where transformations are higher-order functions.

### Functional Programming Model

JAX adopts a strict functional programming model:

1. **Pure functions**: Functions passed to JAX transformations must be pure -- their output depends only on their inputs, and they must have no side effects (no mutation of global state, no I/O, no in-place array modification).

2. **Explicit state**: Unlike PyTorch or TensorFlow 1.x, JAX does not have global state or mutable objects. State (such as model parameters, optimizer state, random number generator keys) is explicitly threaded through function calls and returned as outputs.

3. **Immutable data**: JAX arrays are immutable. "In-place" updates are expressed as functional updates that return new arrays:
   ```python
   # NumPy (mutating):
   # x[0] = 1.0

   # JAX (functional):
   x = x.at[0].set(1.0)
   ```

4. **Explicit PRNG state**: Random number generation uses explicit state (PRNG keys) rather than a global random seed. This ensures reproducibility and makes randomness compatible with JAX transformations:
   ```python
   key = jax.random.PRNGKey(42)
   key, subkey = jax.random.split(key)
   x = jax.random.normal(subkey, (3, 3))
   ```

### Why Pure Functions?

The pure function requirement is essential because JAX transformations work by tracing the function. When you call `jax.jit(f)`, JAX traces `f` with abstract placeholder values (Tracer objects) to build a graph of the computation (a jaxpr). This tracing mechanism requires that:

- The function always executes the same operations given the same input shapes and dtypes (no Python-level data-dependent control flow without special handling)
- The function does not modify external state (since tracing may execute the function multiple times or in unusual contexts)
- The function captures its dependencies explicitly through arguments (not through closures over mutable state)

## Architecture Overview

JAX's architecture consists of several layered components that transform Python functions into optimized machine code:

```
+================================================================+
|                        User Code                                 |
|    Python functions using jax.numpy, jax.lax, jax.nn, etc.      |
+================================================================+
                               |
                               v
+================================================================+
|                   Transformations Layer                          |
|  jax.jit / jax.grad / jax.vmap / jax.pjit / jax.checkpoint     |
|  These intercept function calls and modify execution behavior    |
+================================================================+
                               |
                               v
+================================================================+
|                     Tracing Engine                               |
|  jax.core.Trace, jax.core.Tracer                                |
|  Records operations on abstract values to build jaxprs           |
+================================================================+
                               |
                               v
+================================================================+
|                  Jaxpr (JAX Expression)                          |
|  Intermediate representation: sequence of eqns (equations)       |
|  Each eqn has: primitive, invars, outvars, params               |
+================================================================+
                               |
                               v
+================================================================+
|                    Lowering to XLA                               |
|  jaxpr -> MHLO/StableHLO (Machine HLO)                          |
|  Platform-independent intermediate representation               |
+================================================================+
                               |
                               v
+================================================================+
|                    XLA Compiler                                  |
|  Optimizes MHLO: fusion, layout, memory scheduling              |
|  Generates hardware-specific code                                |
+================================================================+
                               |
                    +----------+----------+
                    |          |          |
                    v          v          v
+============+ +============+ +============+
| CPU (LLVM) | | GPU (PTX)  | |    TPU     |
| Host code  | | CUDA/ROCm  | | Systolic   |
| LLVM IR -> | | PTX/driver | | array ops  |
| native     | | execution   | | via libtpu |
+============+ +============+ +============+
```

### Pipeline Stages in Detail

1. **User Code**: The user writes Python functions using JAX APIs (jax.numpy, jax.lax, etc.). These functions operate on JAX arrays (DeviceArray / Array objects).

2. **Transformations Layer**: When a transformed function is called, the transformation intercepts the call. For example, `jax.jit(f)` wraps `f` so that the first call triggers tracing and compilation. `jax.grad(f)` wraps `f` to evaluate both the primal computation and the adjoint (backward pass).

3. **Tracing Engine**: JAX traces the function by replacing input arrays with abstract Tracer objects. Each operation on a Tracer records itself as an equation in a jaxpr (JAX program). The tracer uses abstract values (shapes and dtypes) rather than concrete data, so the traced program is shape/dtype-polymorphic where possible.

4. **Jaxpr**: The traced computation is captured as a jaxpr -- a data structure representing the computation as a sequence of primitive equations. Each equation has a primitive (like `add`, `mul`, `dot_general`), input variables, output variables, and parameters. Jaxprs can be inspected, transformed, and composed.

5. **Lowering**: The jaxpr is lowered to MHLO (Machine HLO, also known as StableHLO) -- a stable, platform-independent compiler IR. Lowering maps each JAX primitive to one or more HLO operations.

6. **XLA Compilation**: The XLA compiler takes MHLO and performs hardware-specific optimizations: operator fusion, memory layout optimization, tiling strategies, memory scheduling, and more. It then generates executable code for the target platform.

7. **Execution**: The compiled program runs on the target device. Data is transferred to/from the device as needed.

## Core Components

### jax.core - Core Tracing and IR

The `jax.core` module (and its implementation in `jax._src.core`) contains the fundamental abstractions:

- **`Trace`**: An abstract interpreter that processes operations. Different trace types handle different transformations (JitTrace for JIT, JVPTrace for forward-mode AD, etc.)
- **`Tracer`**: A stand-in for a real value during tracing. Tracks the abstract value and records operations into the current trace.
- **`Jaxpr`**: The intermediate representation. A `Jaxpr` contains a list of `JaxprEqn` equations, each with a primitive, input variables, output variables, and parameters.
- **`Var`/`Literal`**: Variables (bound by equations) and literal constants in jaxprs.
- **Abstract values**: `ShapedArray`, `DShapedArray` represent the shape and dtype of arrays without concrete data.

### jax._src - Implementation Details

The `jax._src` package contains the implementation of all JAX public APIs. The public `jax` namespace re-exports from `jax._src`. This separation allows the JAX team to change implementation details without breaking user code that imports from `jax`.

Key submodules:
- `jax._src.core` - Core tracing, jaxpr, primitives
- `jax._src.numpy` - NumPy API implementation
- `jax._src.lax` - Lax operator definitions
- `jax._src.ad_util` - Automatic differentiation utilities
- `jax._src.dispatch` - Compilation and dispatch logic
- `jax._src.sharding` - Sharding implementations
- `jax._src.pjit` - Parallel JIT implementation

### jaxlib - C++ Extension Library

`jaxlib` is the C++/Python extension package that ships alongside JAX. It contains:

- **XLA bindings**: Python wrappers around XLA's C++ client libraries
- **Runtime components**: Device memory management, execution engine
- **Custom call handlers**: Mechanism for calling C/C++ functions from XLA programs
- **Lapack/BLAS wrappers**: For CPU linear algebra operations
- **GPU kernel implementations**: CUDA/ROCm kernel wrappers
- **Version-specific builds**: jaxlib is built for specific CUDA/ROCm versions and platforms

The `jaxlib` package is typically installed as a prebuilt wheel matching your platform and accelerator.

### XLA (Accelerated Linear Algebra)

XLA is the compiler backend that JAX uses to generate optimized machine code. Originally developed for TensorFlow, XLA is now part of the OpenXLA project:

- **HLO (High-Level Optimizer)**: XLA's internal representation for computations
- **StableHLO**: A stable, portable version of HLO used for JAX export and interoperability
- **MHLO (Machine HLO)**: The dialect used internally by JAX for lowering
- **Compiler passes**: Fusion, layout assignment, memory scheduling, tiling, buffer donation
- **Backends**: CPU (via LLVM), GPU (NVIDIA via PTX, AMD via ROCm), TPU (via libtpu)

XLA performs critical optimizations that make JAX fast:
- **Kernel fusion**: Combining multiple operations into a single GPU/TPU kernel to avoid memory bandwidth bottlenecks
- **Memory layout optimization**: Arranging data in memory layouts optimal for the target hardware
- **Buffer reuse**: Reusing memory buffers when possible (e.g., in-place operations)
- **Computation scheduling**: Ordering operations to minimize memory usage and maximize parallelism

## API Layering

JAX provides multiple layers of API, from high-level convenience to low-level control:

### Layer 1: jax.numpy (High-Level)

The NumPy-compatible API. Most users should work primarily at this level:

```python
import jax.numpy as jnp

x = jnp.array([1.0, 2.0, 3.0])
y = jnp.dot(x, x)
z = jnp.sum(jnp.exp(x))
```

This layer provides familiar NumPy functions that work with JAX's immutable arrays and are compatible with all JAX transformations. The API is deliberately similar to NumPy, though there are some documented differences (immutability, PRNG, etc.).

### Layer 2: jax.lax (Mid-Level)

The "lax" (linear algebra extensions) layer provides lower-level primitives that may not have NumPy equivalents. These are often more explicit about their semantics:

```python
import jax.lax as lax

# Reduce window (pooling)
result = lax.reduce_window(x, init_val=0.0, computation=lax.add,
                           window_dimensions=(2,), window_strides=(1,),
                           padding='VALID')

# Conditional
result = lax.cond(pred, lambda x: x + 1, lambda x: x - 1, operand)

# Scan (fold + collect)
final_state, collected = lax.scan(f, init_state, xs)

# Dot general (general matrix multiply with dimension specifications)
result = lax.dot_general(x, y, (((1,), (0,)), ((), ())))
```

Many `jax.numpy` functions are implemented using `jax.lax` primitives. When you need more control over dimension handling, contraction specifications, or windowed operations, `jax.lax` provides the necessary primitives.

### Layer 3: XLA Primitives (Low-Level)

At the lowest level, JAX primitives map directly to XLA HLO operations. These are exposed through `jax.lax` and internal modules. Most users never need to work at this level, but it is accessible for kernel developers and those building JAX extensions:

```python
# Primitives are defined in jax._src.lax.lax
# Each primitive has associated lowering rules to XLA HLO
from jax._src.lax.lax import dot_general_p  # The actual primitive

# Custom lowering rules can be registered for new backends
```

### API Layer Comparison

| Aspect | jax.numpy | jax.lax | XLA/HLO |
|--------|-----------|---------|---------|
| Familiarity | High (NumPy-like) | Medium | Low |
| Flexibility | Standard operations | Extended ops | Full control |
| Verbosity | Low | Medium | High |
| Use case | Most code | Specialized ops | Backend dev |
| Documentation | NumPy docs + JAX diffs | JAX docs | XLA docs |

## Supported Hardware

### CPU

JAX runs on any CPU with LLVM support. This is the default backend and requires no special hardware or drivers.

- **Compiler**: XLA CPU backend via LLVM
- **Parallelism**: Multi-threaded via XLA's threading runtime
- **BLAS/LAPACK**: Uses jaxlib-bundled OpenBLAS or system BLAS

### NVIDIA GPUs (CUDA)

JAX supports NVIDIA GPUs through CUDA. This is the most commonly used accelerator backend.

- **Required**: NVIDIA GPU with compute capability 5.2+ (Maxwell and later), CUDA 12.x
- **Compiler**: XLA GPU backend generating PTX code
- **Libraries**: cuBLAS, cuDNN, cuFFT bundled in jaxlib
- **Installation**: `pip install jax[cuda12]`

```python
# Check GPU availability
import jax
print(jax.devices("gpu"))  # [cuda:0, cuda:1, ...]
print(jax.devices()[0].device_kind)  # e.g., "NVIDIA A100-SXM4-80GB"
```

### AMD GPUs (ROCm)

JAX supports AMD GPUs through the ROCm platform.

- **Required**: AMD GPU (MI250, MI300, etc.), ROCm 6.x
- **Compiler**: XLA GPU backend targeting AMD via ROCm
- **Installation**: `pip install jax[rocm]` (specific wheel from JAX releases)

### Google TPUs

JAX has first-class support for Google TPU hardware.

- **Required**: Google Cloud TPU v2, v3, v4, v5, or later
- **Compiler**: XLA TPU backend via libtpu
- **Installation**: `pip install jax[tpu]`
- **Access**: Google Cloud TPU VMs, Google Colab (TPU runtime), Kaggle (TPU)

### Experimental Platforms

- **Apple Metal (MPS)**: Experimental support for Apple Silicon GPUs via jax-metal
- **Intel GPUs**: Experimental support via Intel Extension for OpenXLA

## Package Structure and Module Organization

### Top-Level jax Package

```
jax/
  __init__.py          # Public API re-exports
  _src/                # Implementation (not public API)
    core/              # Core types, tracing, jaxpr
    numpy/             # NumPy API implementation
    lax/               # Lax operator definitions
    nn/                # Neural network functions
    scipy/             # SciPy API implementation
    image/             # Image processing
    random.py          # PRNG implementation
    ad_util.py         # AD utilities
    dispatch.py        # Compilation and dispatch
    sharding/          # Sharding implementations
    pjit.py            # Parallel JIT
    pmap.py            # Parallel map (legacy)
    export/            # Export functionality
    interpreters/      # Tracing interpreters
    lax_linalg.py      # Linear algebra primitives
    lax_control_flow.py # Control flow primitives
  api.py               # jit, grad, vmap public wrappers
  api_util.py          # API utilities
  config.py            # Configuration system
  core.py              # Re-exports from _src.core
  dlpack.py            # DLPack interop
  dtypes.py            # Dtype definitions
  errors.py            # Error types
  flatten_util.py      # Pytree utilities
  image.py             # Re-exports from _src.image
  lax.py               # Re-exports from _src.lax
  lazy.py              # Lazy loading utilities
  linear_util.py       # Linear function utilities
  nn.py                # Re-exports from _src.nn
  numpy.py             # Re-exports from _src.numpy
  ops.py               # Array operations
  profiler.py          # Profiling utilities
  random.py            # Re-exports from _src.random
  scipy.py             # Re-exports from _src.scipy
  stages.py            # Compilation stages
  tree.py              # Pytree public API
  tree_util.py         # Pytree utilities
  typing.py            # Type annotations
  extend/              # Extension API (stable)
    core.py            # Core extension points
    ffi.py             # Foreign function interface
    random.py          # PRNG extension points
  experimental/        # Experimental features
    sparse/            # Sparse arrays (BCOO)
    pallas/            # Low-level kernel language
    shard_map.py       # Per-device computation
    checkify.py        # Functional error checking
    export.py          # Export API
    array.py           # Array type
    mesh_utils.py      # Device mesh utilities
    custom_dtypes.py   # Custom dtype support
```

### Key Module Descriptions

| Module | Description |
|--------|-------------|
| `jax.numpy` | NumPy-compatible API, the primary user-facing module |
| `jax.lax` | Low-level operators: conv, dot, reduce, scan, control flow |
| `jax.nn` | Neural network specific: activations, initializers, one_hot |
| `jax.random` | PRNG: key creation, splitting, sampling distributions |
| `jax.scipy` | SciPy-compatible: special functions, linalg, optimize, signal |
| `jax.image` | Image processing: resize, scale, affine transforms |
| `jax.sharding` | Sharding: Mesh, PartitionSpec, NamedSharding, PositionalSharding |
| `jax.extend` | Stable extension API for building on JAX |
| `jax.experimental` | Work-in-progress features (sparse, Pallas, shard_map) |
| `jax.profiler` | Performance profiling: trace, server, step markers |
| `jax.core` | Core types: Trace, Tracer, Jaxpr, Var, Primitive |

## Relationship to Other Libraries

### NumPy

JAX's `jax.numpy` API is deliberately modeled after NumPy. Most NumPy code can be made to work with JAX by changing `import numpy as np` to `import jax.numpy as jnp`. Key differences:

| Feature | NumPy | JAX |
|---------|-------|-----|
| Array mutation | `x[0] = 5` | `x = x.at[0].set(5)` |
| Random state | Global (np.random.seed) | Explicit (jax.random.PRNGKey) |
| float64 | Default | Disabled by default |
| Execution | Eager | Async dispatch, JIT compilation |
| GPU support | Limited (via CuPy) | Native |
| Differentiation | None | Built-in (grad, vjp, jvp) |
| Shape info | Static only | Static + dynamic (symbolic) |

### SciPy

`jax.scipy` provides a subset of SciPy's API that works with JAX arrays and is differentiable and JIT-compilable. Not all SciPy modules are covered -- the focus is on functions commonly needed in machine learning and scientific computing:

- `jax.scipy.linalg` - LU, cholesky, svd, solve, etc.
- `jax.scipy.special` - erf, gammaln, logsumexp, etc.
- `jax.scipy.signal` - convolve, correlate, etc.
- `jax.scipy.optimize` - minimize (limited), line_search
- `jax.scipy.stats` - Statistical distributions (limited)

### Autograd

JAX is the spiritual successor to Autograd (Harvard's automatic differentiation library for NumPy). JAX's reverse-mode AD implementation draws heavily from Autograd's approach but extends it with:

- JIT compilation via XLA (Autograd is interpretive)
- Forward-mode AD (Autograd is reverse-mode only)
- GPU/TPU support (Autograd is CPU-only)
- vmap for automatic vectorization
- pjit/pmap for parallelization
- A more comprehensive set of supported NumPy operations

### TensorFlow

While JAX and TensorFlow are both developed at Google, they have fundamentally different designs:

| Aspect | TensorFlow | JAX |
|--------|-----------|-----|
| Programming model | Graph-based (TF2: eager + tf.function) | Functional transformations |
| State | Variable objects with mutable state | Explicit, immutable state |
| Compilation | tf.function / XLA | jax.jit / XLA |
| Differentiation | tf.GradientTape (imperative) | jax.grad (functional) |
| Deployment | TensorFlow Serving, TFLite, TF.js | jax.export, jax2tf (convert to TF) |

JAX's `jax2tf` module allows converting JAX functions to TensorFlow graphs, enabling deployment through TensorFlow's ecosystem.

### PyTorch

JAX and PyTorch represent different approaches to ML frameworks:

| Aspect | PyTorch | JAX |
|--------|---------|-----|
| Programming model | Object-oriented, stateful | Functional, stateless |
| State | Module.parameters() | Explicit pytree params |
| Differentiation | torch.autograd (imperative) | jax.grad (functional, composable) |
| Compilation | torch.compile / Dynamo | jax.jit / XLA |
| Hardware | CUDA, ROCm, XPU, MPS | CUDA, ROCm, TPU |
| Distribution | DDP, FSDP | pjit, shard_map, Mesh |

## JAX Ecosystem

JAX is designed as a low-level computation engine. Higher-level ML functionality is provided by ecosystem libraries:

### Neural Network Libraries

- **Flax** (by Google): The most widely used NN library for JAX. Provides a Linen module system with `nn.Module`, compact syntax, variable collections, and a robust ecosystem. Used in T5X, PaLM, and other major models.
  ```python
  import flax.linen as nn
  class MLP(nn.Module):
      features: int
      @nn.compact
      def __call__(self, x):
          x = nn.Dense(self.features)(x)
          x = nn.relu(x)
          return nn.Dense(10)(x)
  ```

- **Haiku** (by DeepMind): A module system inspired by Sonnet. Transforms stateful modules into pure functions compatible with JAX transforms.

- **Equinox**: A library that treats neural networks as pytrees, enabling elegant composition with JAX transforms without a separate module system.

- **Keras 3**: The latest version of Keras supports JAX as a backend, providing a high-level API familiar to Keras users.

### Optimization

- **Optax**: The standard optimization library for JAX. Provides gradient transformations (scale_by_adam, clip, etc.) that compose into optimizers.
  ```python
  import optax
  optimizer = optax.adam(learning_rate=1e-3)
  grads = jax.grad(loss_fn)(params, x, y)
  updates, state = optimizer.update(grads, state, params)
  params = optax.apply_updates(params, updates)
  ```

### Probabilistic Programming

- **NumPyro**: Probabilistic programming with JAX backend (port of Pyro). MCMC sampling, SVI, and probabilistic models.
- **Oryx**: DeepMind's probabilistic programming and bijector library.
- **Distrax**: DeepMind's probability distributions library.

### Scientific Computing

- **Diffrax**: Differential equation solvers (ODEs, SDEs, CDEs) in JAX.
- **JAX-Cosmo**: Cosmological computations and simulations.
- **JAX-Fluids**: Computational fluid dynamics.
- **QMCI**: Quantum Monte Carlo with JAX.

### Utilities

- **Chex**: Testing utilities (fake JAX, assert_max_traces), dataclass extensions, and pytree utilities.
- **Orbax**: Checkpointing (save/load model state), export, and array serialization.
- **Penzai**: Neural network visualization, manipulation, and analysis tools.

### Large-Scale Training

- **T5X**: Google's implementation of T5 and related models in JAX/Flax.
- **MaxText**: A simple, performant, and scalable Transformer training implementation in pure JAX.
- **Levanter**: Scalable LLM training with JAX, supporting models like GPT-2, LLaMA, etc.

## Version History

| Version | Date | Key Features |
|---------|------|-------------|
| 0.1.x | 2018-2019 | Initial release, basic JIT and grad |
| 0.2.x | 2020 | pmap, multi-device, initial TPU support |
| 0.3.x | 2021 | jax.Array unification, custom_vjp improvements |
| 0.4.x | 2022-2023 | pjit, shard_map, Pallas, export, StableHLO |
| 0.5.x | 2024 | Pallas matmul, improved sharding, export stabilization |
| 0.6.x | 2025-2026 | jax.extend API, Pallas TPU maturity, FFI, continued stabilization |

## Summary

JAX is a high-performance numerical computing library built on three pillars:

1. **Composable transformations**: jit, grad, vmap, pjit can be composed in arbitrary order, enabling powerful abstractions without code duplication.

2. **XLA compilation**: All computations are compiled through XLA, enabling aggressive optimization (fusion, layout, scheduling) and support for multiple hardware backends (CPU, GPU, TPU).

3. **Functional programming model**: Pure functions with explicit state and immutability provide the mathematical foundation that makes transformations correct and composable.

This combination makes JAX particularly well-suited for:
- Machine learning research requiring custom differentiation
- Large-scale training across GPU/TPU clusters
- Scientific computing requiring high-performance numerics
- Rapid prototyping with the ability to productionize (via export)
