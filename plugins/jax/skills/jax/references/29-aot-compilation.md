# Chapter 29: Ahead-of-Time (AOT) Compilation

## Overview

JAX's standard compilation flow is just-in-time (JIT): functions are traced, lowered, and compiled the first time they are called with specific input types. Ahead-of-time (AOT) compilation separates these stages, allowing you to compile functions before executing them. This is useful for:

1. **Reducing first-call latency** -- compile before the first inference request
2. **Type checking** -- verify input types match compiled types at call time
3. **Cost analysis** -- inspect computation cost before running
4. **Deployment** -- ship pre-compiled executables
5. **Persistent caching** -- store compiled artifacts across process restarts

This chapter covers the four compilation stages, the AOT API, persistent compilation caching, and the differences between AOT compilation and JAX export.

## The Four Compilation Stages

JAX transforms Python functions through four distinct stages:

```
Python Function
     |
     v
[1] Tracing          -- Python function -> Jaxpr (JAX intermediate representation)
     |
     v
[2] Lowering         -- Jaxpr -> StableHLO (MLIR-based compiler IR)
     |
     v
[3] Compilation      -- StableHLO -> XLA executable (device-specific binary)
     |
     v
[4] Execution        -- Run the executable on the target device
```

### Stage 1: Tracing

Tracing executes the Python function with abstract tracer objects instead of real arrays. JAX records every operation to build a Jaxpr (JAX program representation).

```python
import jax
import jax.numpy as jnp

def my_fn(x, y):
    z = x + y
    w = jnp.dot(z, z.T)
    return jnp.sum(w)

# Create abstract values for tracing
x_abstract = jax.ShapeDtypeStruct(shape=(3, 4), dtype=jnp.float32)
y_abstract = jax.ShapeDtypeStruct(shape=(3, 4), dtype=jnp.float32)

# Trace the function
jaxpr = jax.make_jaxpr(my_fn)(x_abstract, y_abstract)
print(jaxpr)
# { lambda ; a:f32[3,4] b:f32[3,4]. let
#     c:f32[3,4] = add a b
#     d:f32[3,3] = dot_general[...] c c
#     e:f32[] = reduce_sum[axes=(0, 1)] d
#   in (e,) }
```

### Stage 2: Lowering

Lowering converts the Jaxpr to StableHLO (Stable High-Level Operations), which is the compiler IR used by OpenXLA.

```python
import jax
import jax.numpy as jnp

@jax.jit
def my_fn(x, y):
    return jnp.sum(x + y)

# Lower to StableHLO
lowered = my_fn.lower(
    jnp.ones((3, 4)),
    jnp.ones((3, 4))
)

# View the StableHLO / HLO text
print(lowered.as_text())
```

### Stage 3: Compilation

Compilation converts the lowered StableHLO to a device-specific executable via XLA.

```python
# Compile the lowered program
compiled = lowered.compile()

# The compiled object contains the device-specific executable
print(type(compiled))
# <class 'jax._src.xla_bridge.CompiledFunction'>
```

### Stage 4: Execution

The compiled executable runs on the target device.

```python
# Execute the compiled function
result = compiled(jnp.ones((3, 4)), jnp.ones((3, 4)))
print(result)
# 24.0
```

## The AOT Compilation API

### Complete Example: Trace -> Lower -> Compile -> Execute

```python
import jax
import jax.numpy as jnp

def my_fn(x, y):
    z = x + y
    w = jnp.dot(z, z.T)
    return jnp.sum(w)

# Step 1: Create a JIT-wrapped function
jit_fn = jax.jit(my_fn)

# Step 2: Trace with specific input types
# .trace() takes concrete arrays and creates the traced representation
traced = jit_fn.trace(
    jnp.ones((3, 4), dtype=jnp.float32),
    jnp.ones((3, 4), dtype=jnp.float32),
)

# Step 3: Lower to StableHLO
lowered = traced.lower()

# Step 4: Compile to device executable
compiled = lowered.compile()

# Step 5: Execute
result = compiled(
    jnp.ones((3, 4), dtype=jnp.float32),
    jnp.ones((3, 4), dtype=jnp.float32),
)
print(result)  # 24.0

# Execute with different values (same shape/dtype)
result2 = compiled(
    jnp.array([[1.0, 2.0, 3.0, 4.0]] * 3),
    jnp.array([[0.1, 0.2, 0.3, 0.4]] * 3),
)
print(result2)
```

### Shortcut: Direct lower() and compile()

The `jax.jit` object also supports calling `lower()` and `compile()` directly, skipping explicit trace creation:

```python
import jax
import jax.numpy as jnp

@jax.jit
def my_fn(x, y):
    return jnp.sum(x * y)

# Direct lower (traces implicitly)
lowered = my_fn.lower(
    jnp.ones((3, 4)),
    jnp.ones((3, 4)),
)

# Direct compile
compiled = lowered.compile()

# Execute
result = compiled(jnp.ones((3, 4)), jnp.ones((3, 4)))
print(result)  # 12.0
```

## jax.jit(f).trace(*args) -- Creating Traced Objects

### Basic Trace Creation

```python
import jax
import jax.numpy as jnp

def my_fn(x, y):
    return jnp.dot(x, y)

# Create traced object with concrete arrays
jit_fn = jax.jit(my_fn)
traced = jit_fn.trace(
    jax.ShapeDtypeStruct((4, 3), jnp.float32),
    jax.ShapeDtypeStruct((3, 2), jnp.float32),
)

# The traced object captures the computation graph
print(type(traced))
# <class 'jax._src.dispatch._TracedFunction'>
```

### Using ShapeDtypeStruct for Tracing

You can trace with abstract shapes without creating actual arrays:

```python
import jax
import jax.numpy as jnp

def complex_fn(params, x):
    hidden = jnp.dot(x, params["w"]) + params["b"]
    return jax.nn.softmax(hidden)

jit_fn = jax.jit(complex_fn)

# Trace with abstract shapes -- no actual data needed
params_abstract = {
    "w": jax.ShapeDtypeStruct((784, 10), jnp.float32),
    "b": jax.ShapeDtypeStruct((10,), jnp.float32),
}
x_abstract = jax.ShapeDtypeStruct((1, 784), jnp.float32)

traced = jit_fn.trace(params_abstract, x_abstract)
```

### Inspecting the Traced Object

```python
import jax
import jax.numpy as jnp

@jax.jit
def my_fn(x):
    return x ** 2 + 2 * x + 1

traced = my_fn.trace(jnp.ones(5))

# The traced object has properties for inspection
# (Some may require lowering first)
```

## Traced.lower() -- Lowering to StableHLO

### Basic Lowering

```python
import jax
import jax.numpy as jnp

@jax.jit
def my_fn(x):
    return jnp.sum(x ** 2)

traced = my_fn.trace(jnp.ones(5))
lowered = traced.lower()

# View the lowered representation
hlo_text = lowered.as_text()
print(hlo_text)
```

### Lowering with Compiler Options

```python
import jax
import jax.numpy as jnp

@jax.jit
def my_fn(x):
    return jnp.sum(x ** 2)

# Lower with specific platform target
lowered = my_fn.lower(jnp.ones(5))

# The lowered object captures platform-specific information
# based on the current default backend
```

### Viewing StableHLO IR

```python
import jax
import jax.numpy as jnp

@jax.jit
def matrix_multiply(x, y):
    return jnp.dot(x, y)

lowered = matrix_multiply.lower(
    jnp.ones((64, 64)),
    jnp.ones((64, 64)),
)

# Get the text representation
text = lowered.as_text()
print(text[:1000])  # Print first 1000 chars

# The output shows:
# - HLO instructions (dot, add, constant, etc.)
# - Shapes of intermediate values
# - Compiler metadata and annotations
```

### Lowering Multiple Functions

```python
import jax
import jax.numpy as jnp

# Lower multiple functions for batch compilation
@jax.jit
def fn1(x):
    return jnp.sum(x)

@jax.jit
def fn2(x):
    return jnp.mean(x)

@jax.jit
def fn3(x):
    return jnp.max(x)

x_spec = jnp.ones(100)

lowered1 = fn1.lower(x_spec)
lowered2 = fn2.lower(x_spec)
lowered3 = fn3.lower(x_spec)

# Each lowered object can be compiled independently
compiled1 = lowered1.compile()
compiled2 = lowered2.compile()
compiled3 = lowered3.compile()

# Or reuse the same input for all
result1 = compiled1(jnp.arange(100.0))
result2 = compiled2(jnp.arange(100.0))
result3 = compiled3(jnp.arange(100.0))
print(f"Sum: {result1}, Mean: {result2}, Max: {result3}")
```

## Lowered.compile() -- Compiling to Executable

### Basic Compilation

```python
import jax
import jax.numpy as jnp

@jax.jit
def my_fn(x, y):
    return jnp.dot(x, y) + jnp.sum(x)

lowered = my_fn.lower(
    jnp.ones((4, 4)),
    jnp.ones((4, 4)),
)

# Compile the lowered program
compiled = lowered.compile()

# Execute with matching input shapes and dtypes
result = compiled(jnp.ones((4, 4)), jnp.ones((4, 4)))
print(result)
```

### Compile with Options

```python
import jax
import jax.numpy as jnp

@jax.jit
def my_fn(x):
    return jnp.sum(x ** 2)

lowered = my_fn.lower(jnp.ones(100))

# Compile with specific backend
compiled_gpu = lowered.compile()  # Uses default backend

# Compile for a specific device
compiled_device0 = lowered.compile()
```

### Parallel Compilation

For large programs, you can increase compilation parallelism:

```python
import os

# Set parallel compilation threads
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=8"

import jax
import jax.numpy as jnp

@jax.jit
def large_fn(x):
    for _ in range(50):
        x = jnp.dot(x, x.T)
        x = x / jnp.sum(x)
    return x

lowered = large_fn.lower(jnp.ones((256, 256)))
compiled = lowered.compile()  # Uses multiple threads for compilation
```

## The Compiled Object

### Calling a Compiled Function

```python
import jax
import jax.numpy as jnp

@jax.jit
def my_fn(x):
    return jnp.sum(x ** 2)

# Compile for specific input types
compiled = my_fn.lower(jnp.ones(5)).compile()

# Call with matching shapes/dtypes
result = compiled(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))
print(result)  # 55.0

# Call with different values (same shape/dtype)
result = compiled(jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]))
print(result)  # 5500.0
```

### Type Checking at Call Time

The compiled function checks that inputs match the expected shapes and dtypes. If they do not match, an error is raised:

```python
import jax
import jax.numpy as jnp

@jax.jit
def my_fn(x):
    return jnp.sum(x)

# Compile for shape (5,) and dtype float32
compiled = my_fn.lower(jnp.ones(5, dtype=jnp.float32)).compile()

# OK: same shape and dtype
result = compiled(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))

# ERROR: wrong shape
try:
    compiled(jnp.ones(10))  # Shape (10,) != (5,)
except TypeError as e:
    print(f"Shape mismatch: {e}")

# ERROR: wrong dtype
try:
    compiled(jnp.ones(5, dtype=jnp.int32))  # int32 != float32
except TypeError as e:
    print(f"Dtype mismatch: {e}")
```

### cost_analysis()

The compiled object provides cost analysis that estimates the computational cost of the program:

```python
import jax
import jax.numpy as jnp

@jax.jit
def my_fn(x):
    return jnp.sum(x ** 2)

# Lower and compile
lowered = my_fn.lower(jnp.ones((1000, 1000)))
compiled = lowered.compile()

# Get cost analysis
cost = compiled.cost_analysis()
print("Cost Analysis:")
for key, value in cost[0].items():
    print(f"  {key}: {value}")

# Typical output:
#   flops: 2000000
#   bytes accessed: 8000000
#   optimal_seconds: 0.0000125
#   flops_per_byte: 0.25
#   transcendental_flops: 0
```

### Detailed Cost Analysis Example

```python
import jax
import jax.numpy as jnp

@jax.jit
def matmul_fn(a, b):
    return jnp.dot(a, b)

@jax.jit
def elementwise_fn(x):
    return jnp.exp(x) + jnp.log(x + 1e-8)

# Compare cost of different operations
a = jnp.ones((1024, 1024))
b = jnp.ones((1024, 1024))

matmul_compiled = matmul_fn.lower(a, b).compile()
matmul_cost = matmul_compiled.cost_analysis()[0]

x = jnp.ones((1024, 1024))
elem_compiled = elementwise_fn.lower(x).compile()
elem_cost = elem_compiled.cost_analysis()[0]

print("Matmul cost:")
print(f"  FLOPs: {matmul_cost.get('flops', 0):.2e}")
print(f"  Bytes accessed: {matmul_cost.get('bytes accessed', 0):.2e}")

print("Elementwise cost:")
print(f"  FLOPs: {elem_cost.get('flops', 0):.2e}")
print(f"  Bytes accessed: {elem_cost.get('bytes accessed', 0):.2e}")
```

### as_text()

```python
import jax
import jax.numpy as jnp

@jax.jit
def simple_fn(x):
    return x * 2 + 1

# Lower to get text representation
lowered = simple_fn.lower(jnp.ones(5))
hlo_text = lowered.as_text()
print(hlo_text)

# This shows the HLO/StableHLO IR:
# HloModule simple_fn ...
#   ENTRY main {
#     parameter.1 = f32[5]{0} parameter(0)
#     constant.2 = f32[] constant(2)
#     broadcast.3 = f32[5]{0} broadcast(constant.2), dimensions={}
#     multiply.4 = f32[5]{0} multiply(parameter.1, broadcast.3)
#     ...
#   }
```

## Static Arguments with AOT

### Using static_argnums

Static arguments are evaluated at trace time (not compiled into the executable). They cause recompilation when their values change.

```python
import jax
import jax.numpy as jnp

def my_fn(x, multiplier):
    return x * multiplier

# Mark 'multiplier' as static (argnum 1)
jit_fn = jax.jit(my_fn, static_argnums=(1,))

# Compile for multiplier=3
lowered = jit_fn.lower(jnp.ones(5), 3)
compiled = lowered.compile()

# Execute with multiplier=3 (matching the compiled value)
result = compiled(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
print(result)  # [ 3.  6.  9. 12. 15.]

# ERROR: multiplier=5 doesn't match compiled multiplier=3
# This would work with regular jit (triggers recompilation)
# but AOT compiled functions cannot recompile
try:
    result = compiled(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), 5)
except Exception as e:
    print(f"Static arg mismatch: {e}")
```

### Multiple Static Arguments

```python
import jax
import jax.numpy as jnp

def configurable_fn(x, scale, offset, activation):
    x = x * scale + offset
    if activation == "relu":
        return jax.nn.relu(x)
    elif activation == "gelu":
        return jax.nn.gelu(x)
    else:
        return x

# All non-array arguments are static
jit_fn = jax.jit(configurable_fn, static_argnums=(1, 2, 3))

# Compile for specific configuration
lowered = jit_fn.lower(jnp.ones(5), 2.0, 1.0, "relu")
compiled = lowered.compile()

# Execute with matching static args
result = compiled(jnp.array([1.0, -2.0, 3.0, -4.0, 5.0]), 2.0, 1.0, "relu")
print(result)  # [3. 0. 7. 0. 11.]
```

### Static Arguments and Shape Dependence

```python
import jax
import jax.numpy as jnp

def reshape_fn(x, rows, cols):
    return x.reshape(rows, cols)

# rows and cols determine the output shape -- must be static
jit_fn = jax.jit(reshape_fn, static_argnums=(1, 2))

# Compile for 2x3 reshape
lowered = jit_fn.lower(jnp.arange(6.0), 2, 3)
compiled = lowered.compile()

result = compiled(jnp.arange(6.0), 2, 3)
print(result)
# [[0. 1. 2.]
#  [3. 4. 5.]]
```

## Debug Information

### Getting HLO Text at Each Stage

```python
import jax
import jax.numpy as jnp

@jax.jit
def my_fn(x, y):
    z = x + y
    w = jnp.dot(z, z.T)
    return jnp.sum(w)

x = jnp.ones((4, 4))
y = jnp.ones((4, 4))

# Stage 1: Jaxpr (after tracing)
jaxpr = jax.make_jaxpr(my_fn)(x, y)
print("=== Jaxpr ===")
print(jaxpr)

# Stage 2: Lowered HLO
lowered = my_fn.lower(x, y)
print("\n=== Lowered HLO ===")
print(lowered.as_text())

# Stage 3: Compiled HLO (with optimizations)
compiled = lowered.compile()
print("\n=== Cost Analysis ===")
print(compiled.cost_analysis())
```

### Inspecting Shapes Through the Pipeline

```python
import jax
import jax.numpy as jnp

@jax.jit
def multi_output_fn(x):
    return x + 1, x * 2, jnp.sum(x)

x = jnp.ones((3, 4))

# Trace
jaxpr = jax.make_jaxpr(multi_output_fn)(x)
print("Output shapes from jaxpr:")
for aval in jaxpr.out_avals:
    print(f"  {aval.shape}, {aval.dtype}")

# Lower
lowered = multi_output_fn.lower(x)
print("\nLowered HLO (first 300 chars):")
print(lowered.as_text()[:300])

# Compile and execute
compiled = lowered.compile()
a, b, c = compiled(x)
print(f"\nResults: a.shape={a.shape}, b.shape={b.shape}, c={c}")
```

### Using XLA Dump for Full Debug Information

```python
import os

# Dump all compilation artifacts
os.environ["XLA_FLAGS"] = (
    "--xla_dump_to=/tmp/xla_aot_dump "
    "--xla_dump_hlo_as_text "
    "--xla_dump_hlo_as_dot "
    "--xla_dump_hlo_pass_re=.*"
)

import jax
import jax.numpy as jnp

@jax.jit
def my_fn(x):
    return jnp.sum(x ** 2)

lowered = my_fn.lower(jnp.ones(100))
compiled = lowered.compile()

# Check /tmp/xla_aot_dump/ for:
# - before_optimizations.txt   (unoptimized HLO)
# - after_optimizations.txt    (optimized HLO)
# - *.dot                      (GraphViz visualization)
# - GPU assembly (PTX) or CPU assembly
```

## Limitations of AOT Compilation

### No Transformations on AOT Functions

AOT-compiled functions cannot be further transformed (grad, vmap, jit, etc.):

```python
import jax
import jax.numpy as jnp

@jax.jit
def my_fn(x):
    return jnp.sum(x ** 2)

# Compile AOT
compiled = my_fn.lower(jnp.ones(5)).compile()

# ERROR: Cannot apply grad to a compiled function
try:
    grad_fn = jax.grad(compiled)  # This will fail
except Exception as e:
    print(f"Cannot grad a compiled function: {e}")

# WORKAROUND: Apply transformations before AOT compilation
grad_fn = jax.grad(my_fn)  # Apply grad first
grad_compiled = jax.jit(grad_fn).lower(jnp.ones(5)).compile()  # Then AOT compile
result = grad_compiled(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))
print(result)  # [ 2.  4.  6.  8. 10.]
```

### No Dynamic Shapes

AOT-compiled functions are specialized to specific shapes:

```python
import jax
import jax.numpy as jnp

@jax.jit
def my_fn(x):
    return jnp.sum(x)

# Compile for shape (5,)
compiled = my_fn.lower(jnp.ones(5)).compile()

# This works
result = compiled(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))

# This does NOT work (shape mismatch)
try:
    result = compiled(jnp.ones(10))
except TypeError as e:
    print(f"Shape error: {e}")

# Unlike regular jit, AOT does NOT auto-recompile
```

### No Python Side Effects

```python
import jax
import jax.numpy as jnp

counter = 0

def fn_with_side_effect(x):
    global counter
    counter += 1  # This only happens during tracing, not execution
    return x + 1

compiled = jax.jit(fn_with_side_effect).lower(jnp.ones(3)).compile()
print(f"Counter after compile: {counter}")  # 1 (traced once)

result = compiled(jnp.array([1.0, 2.0, 3.0]))
print(f"Counter after execute: {counter}")  # Still 1 (no re-tracing)
```

### Cannot Use Python Callbacks

```python
import jax
import jax.numpy as jnp

def fn_with_callback(x):
    # debug.print works but is a no-op in AOT-compiled code
    jax.debug.print("value: {}", x)
    return x + 1

# AOT compilation may warn about debug operations
compiled = jax.jit(fn_with_callback).lower(jnp.ones(3)).compile()
result = compiled(jnp.array([1.0, 2.0, 3.0]))
# The debug.print may not produce output in AOT mode
```

## Export vs AOT Compilation

### Key Differences

AOT compilation and JAX Export serve different purposes:

| Feature | AOT Compilation | JAX Export |
|---------|----------------|------------|
| Output | In-memory executable | StableHLO artifact (serialized) |
| Platform | Specific to compile-time device | Cross-platform (via StableHLO) |
| Persistence | Process-lifetime (unless cached) | Can be saved to disk indefinitely |
| API | `jit.lower().compile()` | `jax.export.export()` |
| Use case | Reduce first-call latency | Deploy to TF/Serving/TFLite |
| Device dependency | Tied to specific GPU/TPU model | Portable across compatible devices |
| Transformations | Cannot apply after compilation | Cannot apply after export |
| Input types | Must match exactly | Must match exactly |

### When to Use Each

```
Use AOT Compilation when:
  - You want to reduce first-call latency in a Python process
  - You want to inspect compilation cost before execution
  - You need cost analysis or HLO inspection
  - You want persistent compilation caching

Use Export when:
  - You need to deploy to TensorFlow Serving
  - You want cross-platform portability
  - You need to serialize the computation graph
  - You are integrating with non-JAX systems
```

### Export Example (for comparison)

```python
import jax
import jax.numpy as jnp

@jax.jit
def my_fn(x):
    return jnp.sum(x ** 2)

# AOT compilation (in-memory)
aot_compiled = my_fn.lower(jnp.ones(5)).compile()
result_aot = aot_compiled(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))

# Export (serializable)
try:
    exported = jax.export.export(my_fn)(
        jnp.ones(5)
    )
    # The exported object contains StableHLO that can be serialized
    # and run on different platforms (TF, TFLite, etc.)
    print(f"Exported: {exported}")
except AttributeError:
    # jax.export may require specific JAX versions
    print("Export API may not be available in this JAX version")
```

## Persistent Compilation Cache

JAX can cache compiled executables to disk, avoiding recompilation across process restarts.

### Basic Persistent Caching

```python
import os

# Set compilation cache directory
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_compilation_cache"

import jax
import jax.numpy as jnp

@jax.jit
def expensive_fn(x):
    # Simulate expensive compilation
    result = x
    for _ in range(100):
        result = jnp.dot(result, result.T)
        result = result / jnp.max(result)
    return result

# First run: compiles and caches
x = jnp.ones((64, 64))
result = expensive_fn(x)
result.block_until_ready()

# Subsequent process restarts will load from cache instead of recompiling
```

### Cache Configuration

```python
import os

# Set cache directory
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"

# Set maximum cache size (in bytes)
os.environ["JAX_COMPILATION_CACHE_MAX_SIZE"] = "10737418240"  # 10 GB

# Enable strict cache checking
os.environ["JAX_COMPILATION_CACHE_STRICT"] = "true"

import jax
```

### Cache Invalidation

The compilation cache is invalidated when:

1. **JAX version changes** -- Different jaxlib versions produce different executables
2. **Input shapes/dtypes change** -- Each unique type signature gets its own cache entry
3. **XLA flags change** -- Different compiler flags produce different code
4. **Hardware changes** -- Different GPU architectures need different executables

```python
import os

# The cache key includes:
# - Function hash (based on jaxpr)
# - Input shapes and dtypes
# - XLA compiler version
# - Target device architecture
# - XLA flags

os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"

import jax
import jax.numpy as jnp

@jax.jit
def my_fn(x):
    return jnp.sum(x ** 2)

# This creates one cache entry for shape (5,)
result = my_fn(jnp.ones(5))

# This creates a separate cache entry for shape (10,)
result = my_fn(jnp.ones(10))

# Cache entries:
# /tmp/jax_cache/
#   <hash_for_shape_5>/
#     executable.so
#     metadata.json
#   <hash_for_shape_10>/
#     executable.so
#     metadata.json
```

### Using AOT with Persistent Cache

```python
import os

os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_aot_cache"

import jax
import jax.numpy as jnp

# AOT compilation also benefits from persistent caching
@jax.jit
def my_fn(x):
    return jnp.sum(x ** 2)

# Lower and compile (cached on disk)
lowered = my_fn.lower(jnp.ones(5))
compiled = lowered.compile()

# On next process startup, this will load from cache
# instead of recompiling
```

### Pre-Warming the Cache

For deployment scenarios, you can pre-warm the compilation cache:

```python
import os
import jax
import jax.numpy as jnp

os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_precompilation_cache"

def precompile_model():
    """Pre-compile all model functions and populate the cache."""
    from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

    # Define model shapes
    batch_size = 32
    seq_len = 128
    hidden_dim = 768

    @jax.jit
    def forward(params, x):
        return jnp.dot(x, params["w"]) + params["b"]

    @jax.jit
    def train_step(params, x, y):
        def loss_fn(p):
            pred = jnp.dot(x, p["w"]) + p["b"]
            return jnp.mean((pred - y) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        return loss, grads

    # Create dummy inputs
    params = {
        "w": jnp.ones((hidden_dim, hidden_dim)),
        "b": jnp.zeros(hidden_dim),
    }
    x = jnp.ones((batch_size, hidden_dim))
    y = jnp.ones((batch_size, hidden_dim))

    # Compile all functions (populates cache)
    print("Compiling forward...")
    _ = forward(params, x).block_until_ready()

    print("Compiling train_step...")
    _ = train_step(params, x, y)[0].block_until_ready()

    print("Pre-compilation complete. Cache populated.")

# Run pre-compilation
precompile_model()

# In production:
# 1. Set JAX_COMPILATION_CACHE_DIR to the same directory
# 2. The first call loads from cache instead of compiling
# 3. First-call latency is reduced to cache load time
```

## Compilation Cache Configuration

### Environment Variables for Cache Control

```bash
# Cache directory (required for persistent caching)
export JAX_COMPILATION_CACHE_DIR=/path/to/cache

# Maximum cache size in bytes (default: unlimited)
export JAX_COMPILATION_CACHE_MAX_SIZE=10737418240  # 10 GB

# Strict mode: validate cache entries before loading
export JAX_COMPILATION_CACHE_STRICT=true

# PGLE directory for profile-guided optimization
export JAX_PGLE_EMBEDDING_DIR=/path/to/pgle

# Compilation statistics logging
export JAX_COMPILATION_CACHE_LOG_LEVEL=INFO
```

### Programmatic Cache Control

```python
import jax

# Get the current cache configuration
print(f"Cache dir: {jax.config.jax_compilation_cache_dir}")

# Clear the compilation cache (if needed)
# This forces recompilation on next use
```

### Cache Best Practices

```
1. Use a dedicated cache directory per project
   - Different projects may have different XLA flags

2. Use consistent XLA flags across cache-using processes
   - Changing XLA flags invalidates cache entries

3. Monitor cache size
   - Large models with many input shapes can fill disk
   - Set JAX_COMPILATION_CACHE_MAX_SIZE appropriately

4. Pre-warm cache during deployment setup
   - Run all critical functions once before serving traffic

5. Version your cache directory
   - Include JAX version in the path for safety
   - e.g., /tmp/jax_cache_v0.6.1/

6. Do not share cache across different GPU architectures
   - A100 executables will not work on V100
```

## AOT Compilation Best Practices

### 1. Compile at Startup

```python
import jax
import jax.numpy as jnp
import time

class ModelServer:
    def __init__(self):
        """Initialize and compile the model at startup."""
        self.params = self._init_params()

        # Compile inference function at startup
        start = time.perf_counter()
        self._compiled_infer = self._compile_inference()
        compile_time = time.perf_counter() - start
        print(f"Compilation took {compile_time:.2f}s")

    def _init_params(self):
        key = jax.random.PRNGKey(0)
        return {
            "w": jax.random.normal(key, (784, 10)),
            "b": jnp.zeros(10),
        }

    def _inference(self, params, x):
        return jax.nn.softmax(jnp.dot(x, params["w"]) + params["b"])

    def _compile_inference(self):
        """AOT compile the inference function."""
        dummy_x = jnp.ones((1, 784))
        return jax.jit(self._inference).lower(self.params, dummy_x).compile()

    def predict(self, x):
        """Run inference with pre-compiled function."""
        return self._compiled_infer(self.params, x)

# Usage
server = ModelServer()  # Compiles at startup
result = server.predict(jnp.ones((1, 784)))  # No compilation overhead
```

### 2. Compile Multiple Input Shapes

```python
import jax
import jax.numpy as jnp

class FlexibleModel:
    def __init__(self):
        self.params = jnp.ones((100, 10))

        # Pre-compile for common batch sizes
        self.compiled_fns = {}
        for batch_size in [1, 8, 16, 32, 64]:
            self.compiled_fns[batch_size] = self._compile(batch_size)

    def _forward(self, params, x):
        return jnp.dot(x, params)

    def _compile(self, batch_size):
        dummy_x = jnp.ones((batch_size, 100))
        return jax.jit(self._forward).lower(self.params, dummy_x).compile()

    def predict(self, x):
        batch_size = x.shape[0]
        if batch_size in self.compiled_fns:
            return self.compiled_fns[batch_size](self.params, x)
        else:
            # Fallback: regular JIT for unexpected batch sizes
            return jax.jit(self._forward)(self.params, x)
```

### 3. Cost-Based Function Selection

```python
import jax
import jax.numpy as jnp

def compare_implementations(fn1, fn2, x):
    """Compare two implementations using AOT cost analysis."""
    # Compile both
    compiled1 = jax.jit(fn1).lower(x).compile()
    compiled2 = jax.jit(fn2).lower(x).compile()

    # Compare cost analysis
    cost1 = compiled1.cost_analysis()[0]
    cost2 = compiled2.cost_analysis()[0]

    flops1 = cost1.get("flops", float("inf"))
    flops2 = cost2.get("flops", float("inf"))

    print(f"Implementation 1 FLOPs: {flops1:.2e}")
    print(f"Implementation 2 FLOPs: {flops2:.2e}")

    # Return the cheaper implementation
    if flops1 <= flops2:
        return compiled1
    else:
        return compiled2

# Usage
def impl1(x):
    return jnp.exp(x) / jnp.sum(jnp.exp(x))

def impl2(x):
    return jax.nn.softmax(x)

x = jnp.ones((100, 100))
best = compare_implementations(impl1, impl2, x)
```

## Summary: AOT Compilation Quick Reference

| Operation | API | Description |
|-----------|-----|-------------|
| Trace | `jit_fn.trace(*args)` | Create traced representation |
| Lower | `traced.lower()` or `jit_fn.lower(*args)` | Lower to StableHLO |
| Compile | `lowered.compile()` | Compile to executable |
| Execute | `compiled(*args)` | Run the compiled function |
| View HLO | `lowered.as_text()` | View the HLO IR text |
| Cost analysis | `compiled.cost_analysis()` | Get estimated compute cost |
| Static args | `jax.jit(fn, static_argnums=(1,))` | Mark args as compile-time constants |
| Persistent cache | `JAX_COMPILATION_CACHE_DIR` env var | Cache compiled executables |
| Export | `jax.export.export(fn)(*args)` | Export to portable StableHLO |

### Stage-by-Stage Reference

```
Python Function
    |  jax.jit(fn)
    v
[jit_fn] -- callable wrapper that manages compilation
    |  jit_fn.lower(*args) or jit_fn.trace(*args).lower()
    v
[lowered] -- StableHLO representation
    |  lowered.as_text()    -- view IR
    |  lowered.compile()    -- compile to executable
    v
[compiled] -- device-specific executable
    |  compiled(*args)      -- execute
    |  compiled.cost_analysis() -- inspect cost
    v
result -- JAX array(s) on device
```
