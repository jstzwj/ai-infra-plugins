# 04 - JIT Compilation (jax.jit)

## Overview

`jax.jit` (Just-In-Time compilation) is JAX's primary optimization transformation. It traces a Python function, converts it to a jaxpr, and compiles it via XLA for efficient execution on CPU, GPU, or TPU.

---

## 1. Basic Usage

### Function decoration

```python
import jax
import jax.numpy as jnp

@jax.jit
def selu(x, alpha=1.67, lambda_=1.05):
    return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

# Equivalent:
selu_jit = jax.jit(selu)
```

### First call = compilation

```python
x = jnp.arange(1000000)

# First call: traces + compiles (includes overhead)
result = selu_jit(x)  # ~100ms (first time)

# Subsequent calls: uses cached compiled code
result = selu_jit(x)  # ~0.1ms (cached)
```

---

## 2. How JIT Works

### Execution pipeline

```
1. User calls f_jit(x)
2. JAX wraps x in Tracer objects
3. Tracer records all JAX operations → Jaxpr
4. Jaxpr is lowered to HLO/StableHLO
5. XLA compiles HLO → device executable
6. Executable is cached
7. Input data is passed to executable
8. Result is returned (asynchronously)
```

### Viewing the jaxpr

```python
jax.make_jaxpr(selu)(jnp.ones(3))
# { lambda ; a:f32[3]. let
#     b:bool[3] = gt a 0.0
#     c:f32[3] = exp a
#     d:f32[3] = mul 1.67 c
#     e:f32[3] = sub d 1.67
#     f:f32[3] = select_n b e a
#     g:f32[3] = mul 1.05 f
#   in (g,) }
```

---

## 3. When JIT Doesn't Work

### Value-dependent control flow

```python
# BAD: Branching on array values
@jax.jit
def f(x):
    if x > 0:          # x is a Tracer, not a concrete value!
        return x
    else:
        return 2 * x

f(10)  # ConcretizationTypeError!
```

**Why**: During tracing, `x` is an abstract `Traced<ShapedArray>`. The `>` comparison produces a traced boolean, not a Python `True`/`False`.

### Fixes

**Option 1: Use `jax.numpy.where` (no branching)**
```python
@jax.jit
def f(x):
    return jnp.where(x > 0, x, 2 * x)
```

**Option 2: Use `jax.lax.cond` (structured conditional)**
```python
@jax.jit
def f(x):
    return jax.lax.cond(x > 0, lambda x: x, lambda x: 2 * x, x)
```

**Option 3: Mark argument as static**
```python
@jax.jit
def f(x):
    if x > 0:
        return x
    else:
        return 2 * x

# Use static_argnums — but x must be a Python scalar, not array
f_static = jax.jit(f, static_argnums=0)
f_static(10)  # Works, but recompiles for each new value of x
```

### Side effects

```python
# BAD: Print only happens during tracing
@jax.jit
def f(x):
    print(x)  # Prints Traced<...> once during trace
    return x + 1

# GOOD: Debug print
@jax.jit
def f(x):
    jax.debug.print("x = {}", x)  # Prints actual runtime values
    return x + 1
```

---

## 4. Static Arguments

### `static_argnums`

Specify which arguments should be treated as compile-time constants:

```python
@jax.jit
def g(x, n):
    i = 0
    while i < n:    # n controls loop → must be static
        i += 1
    return x + i

g_static = jax.jit(g, static_argnums=1)
g_static(10, 20)  # Compiles with n=20
g_static(10, 30)  # Recompiles with n=30 (new value!)
```

### `static_argnames`

```python
g_static = jax.jit(g, static_argnames=['n'])
```

### Using with `functools.partial`

```python
from functools import partial

@partial(jax.jit, static_argnames=['n'])
def g(x, n):
    i = 0
    while i < n:
        i += 1
    return x + i
```

### Cost of static arguments

- Each unique value triggers recompilation
- Only use for arguments with limited distinct values
- Don't use for data-dependent values

---

## 5. Compilation Caching

### Cache keys

JIT caches are keyed on:
1. Function identity (hash of the Python function object)
2. Input shapes and dtypes
3. Static argument values

### Cache behavior

```python
f_jit = jax.jit(f)

f_jit(jnp.ones(3))    # Compile (float32[3])
f_jit(jnp.ones(3))    # Cache hit ✓
f_jit(jnp.ones(5))    # Recompile (float32[5]) — different shape
f_jit(jnp.ones(3, jnp.float64))  # Recompile (float64[3]) — different dtype
f_jit(2 * jnp.ones(3))  # Cache hit ✓ — same shape/dtype, different values
```

### Anti-patterns: Cache misses

```python
# BAD: Lambda/partial in a loop → new hash each iteration
def loop(x, n):
    for i in range(n):
        x = jax.jit(lambda x: x + 1)(x)  # New function hash each time!
    return x

# GOOD: Define once outside the loop
@jax.jit
def add_one(x):
    return x + 1

def loop(x, n):
    for i in range(n):
        x = add_one(x)  # Same function hash, cache hit
    return x
```

### Donating arguments

```python
# donate_argnums allows JAX to reuse input buffers for outputs
@jax.jit(donate_argnums=0)
def f(x):
    return x + 1

x = jnp.ones(1000000)
result = f(x)
# x's buffer is donated to result — no extra allocation
# WARNING: x is no longer usable after this call!
```

---

## 6. `device` Parameter

Compile for a specific device:

```python
# Compile for GPU 0
f_gpu = jax.jit(f, device=jax.devices('gpu')[0])

# Compile for specific TPU core
f_tpu = jax.jit(f, device=jax.devices('tpu')[2])
```

---

## 7. `backend` Parameter

Specify the compilation backend:

```python
f_gpu = jax.jit(f, backend='gpu')
f_cpu = jax.jit(f, backend='cpu')
f_tpu = jax.jit(f, backend='tpu')
```

---

## 8. `keep_unused` and `inline`

### `keep_unused`

```python
# By default, unused arguments are pruned from the jaxpr
@jax.jit(keep_unused=True)
def f(x, unused):
    return x + 1
# unused is kept in the compiled function signature
```

### `inline`

```python
# Force inlining of the function into the caller's jaxpr
@jax.jit(inline=True)
def small_fn(x):
    return x + 1
```

---

## 9. `abstract_eval` and Shape-Only Tracing

For shape/dtype-only tracing (no data needed):

```python
# Create abstract shapes
x_abstract = jax.ShapeDtypeStruct(shape=(3, 4), dtype=jnp.float32)
jaxpr = jax.make_jaxpr(f)(x_abstract)
```

---

## 10. JIT with Pytrees

JIT works seamlessly with pytree inputs/outputs:

```python
Params = dict  # e.g., {'layer0': {'w': ..., 'b': ...}, ...}

@jax.jit
def predict(params: Params, x):
    for layer_name in sorted(params):
        x = x @ params[layer_name]['w'] + params[layer_name]['b']
        x = jax.nn.relu(x)
    return x
```

Recompilation occurs when the **pytree structure** changes:

```python
params2 = {'layer0': {'w': ..., 'b': ..., 'bn_scale': ...}}  # Extra key
predict(params2, x)  # Recompiles — different pytree structure
```

---

## 11. Nested JIT

JAX automatically handles nested `jit`:

```python
@jax.jit
def outer(x):
    return inner(x) + 1

@jax.jit
def inner(x):
    return x * 2

outer(jnp.ones(3))  # inner is inlined into outer's jaxpr
```

### Avoiding recompilation in nested JIT

```python
# BAD: inner is redefined each call
@jax.jit
def outer(x, flag):
    if flag:
        @jax.jit
        def inner(y):  # New function object each time!
            return y + 1
    else:
        @jax.jit
        def inner(y):
            return y - 1
    return inner(x)
```

---

## 12. Compilation Time Optimization

### Minimize compilation

1. **Pad/batch to fixed shapes**: Avoid dynamic shapes that trigger recompilation
2. **Use `static_argnums` sparingly**: Each unique value → recompile
3. **Pre-compile with warm-up**: Call once before timing
4. **Use `jax.jit` at the outermost level**: JIT the whole training step, not individual ops

### Check compilation cache

```python
# See how many compilations happened
jax.clear_cache()  # Clear all cached compilations
```

### Compilation cost estimation

| Model complexity | Compilation time (approx) |
|---|---|
| Small MLP | 0.1–1s |
| ResNet-50 | 5–30s |
| Transformer (large) | 30s–5min |

---

## 13. Debugging JIT

### Common errors

| Error | Cause | Fix |
|---|---|---|
| `ConcretizationTypeError` | Branching on traced value | Use `where`/`cond` or `static_argnums` |
| `TracerIntegerConversionError` | Using traced int as Python int | Use `static_argnums` |
| `UnexpectedTracerError` | Leaked tracer from JIT | Don't return tracers |
| `CompilationFailureError` | XLA can't compile | Simplify or check shapes |

### Debugging techniques

```python
# 1. Disable JIT for debugging
with jax.disable_jit():
    result = f(x)  # Runs as regular Python

# 2. Use jax.debug.print inside JIT
@jax.jit
def f(x):
    jax.debug.print("x = {}", x)
    return x + 1

# 3. Check jaxpr before compiling
jax.make_jaxpr(f)(x)

# 4. Enable verbose logging
jax.config.update("jax_log_compiles", True)
```

---

## 14. Advanced: `jax.jit` API Reference

```python
jax.jit(
    fun: Callable,
    *,
    static_argnums: int | Sequence[int] = (),    # Compile-time constant args
    static_argnames: str | Sequence[str] = (),    # By name
    device: Device | None = None,                 # Target device
    backend: str | None = None,                   # Target backend
    donate_argnums: int | Sequence[int] = (),     # Buffers to donate
    inline: bool = False,                         # Force inlining
    keep_unused: bool = False,                    # Keep unused args
    abstracted_axes: ... = None,                  # Shape polymorphism
) -> Callable
```

### Return value

Returns a `CompiledFunction` that:
- Caches compiled executables
- Traces on first call or when cache misses
- Dispatches asynchronously to the device
