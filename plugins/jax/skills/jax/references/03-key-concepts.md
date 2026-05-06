# 03 - JAX Key Concepts

## Overview

JAX is built on a small set of foundational concepts that compose together to enable high-performance numerical computing. This chapter covers the core abstractions every JAX user must understand.

---

## 1. Transformations

JAX transformations are higher-order functions: they accept a Python function and return a new transformed function. The core transformations are:

| Transformation | Module | Purpose |
|---|---|---|
| `jax.jit` | `jax` | Just-in-time compilation via XLA |
| `jax.grad` | `jax` | Reverse-mode automatic differentiation |
| `jax.vmap` | `jax` | Automatic vectorization |
| `jax.jvp` | `jax` | Forward-mode automatic differentiation |
| `jax.vjp` | `jax` | Reverse-mode AD (vector-Jacobian product) |
| `pmap` | `jax` | Parallel map across devices |
| `jax.checkpoint` | `jax` | Gradient rematerialization |

### Composition

Transformations compose freely:

```python
# JIT + grad
@jax.jit
def train_step(params, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)

# vmap + grad (per-sample gradients)
per_sample_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))

# Nested grad (Hessian)
hessian = jax.jacfwd(jax.jacrev(f))
```

---

## 2. Tracing

When a transformation is applied, JAX traces the function: it replaces input arrays with abstract **Tracer** objects and records all JAX operations performed on them.

### What Tracers capture

```python
@jax.jit
def f(x):
    print(f"x = {x}")          # Prints Traced<ShapedArray(float32[5])>
    print(f"x.shape = {x.shape}")  # (5,) — static attribute
    print(f"x.dtype = {x.dtype}")  # float32 — static attribute
    return x + 1

f(jnp.arange(5.0))
```

### Static vs traced values

| Property | Access | Available during trace |
|---|---|---|
| `shape` | `x.shape` | Yes (static) |
| `dtype` | `x.dtype` | Yes (static) |
| `ndim` | `x.ndim` | Yes (static) |
| `size` | `x.size` | Yes (static) |
| Data values | `x[0]` etc. | No (traced/abstract) |

### Shape polymorphism

JAX supports tracing with abstract shapes using symbolic dimensions:

```python
# Trace with batch dimension as symbolic
f_jit = jax.jit(f, abstracted_axes={0: 'n'})
# or with export:
f_exported = jax.export.export(shape polymorphism='(b, _)')(f)
```

---

## 3. Jaxpr (JAX Expression)

A jaxpr is JAX's internal intermediate representation — a simple functional language recording the sequence of primitive operations.

### Viewing jaxprs

```python
jax.make_jaxpr(f)(x)
```

Output structure:
```
{ lambda ; a:f32[3]. let
    b:f32[3] = exp a
    c:f32[3] = log a
    d:f32[3] = div b c
  in (d,) }
```

### Jaxpr components

- **`Jaxpr`**: Contains `invars`, `outvars`, `eqns` (equations), `consts`
- **`Var`**: A named variable (`a`, `b`, etc.)
- **`Literal`**: A constant value
- **`JaxprEqn`**: One equation binding primitives to inputs/outputs
- **`Primitive`**: Named operation (`add`, `mul`, `conv_general_dilated`, etc.)

### Jaxpr properties

- **Side-effect free**: `print()`, file I/O, and mutation are not captured
- **First-order**: Higher-order functions are inlined during tracing
- **Typed**: Every variable carries an abstract value (shape + dtype)

---

## 4. Pytrees

A **pytree** is any nested structure of container-like Python objects. JAX treats registered containers (list, tuple, dict, namedtuple, OrderedDict) as tree nodes, and everything else as leaves.

### Quick examples

```python
# Leaf — treated as a single value
jax.tree.leaves(jnp.array([1, 2, 3]))  # [DeviceArray([1, 2, 3])]

# List — each element is a leaf
jax.tree.leaves([1, 2, 3])  # [1, 2, 3]

# Nested dict — flattened
jax.tree.leaves({'a': 1, 'b': {'c': 2}})  # [1, 2]

# None is NOT a leaf by default
jax.tree.leaves([None, 1])  # [1]
```

### Key pytree operations

```python
# Map over leaves
jax.tree.map(lambda x: x * 2, {'a': jnp.ones(3), 'b': jnp.zeros(2)})

# Flatten / unflatten
leaves, treedef = jax.tree.flatten(pytree)
pytree_restored = jax.tree.unflatten(treedef, leaves)

# Structure
treedef = jax.tree.structure(pytree)
num_leaves = treedef.num_leaves
num_children = treedef.num_children

# Reduce
total = jax.tree.reduce(lambda acc, x: acc + x.sum(), params, 0.0)

# Transpose
jax.tree.transpose(
    outer_treedef=jax.tree.structure([0, 0]),
    inner_treedef=jax.tree.structure({'a': 0}),
    pytree_to_transpose=[{'a': 1}, {'a': 2}]
)
# {'a': [1, 2]}
```

### Key paths

```python
keys_leaves = jax.tree_util.tree_flatten_with_path(pytree)
for path, val in keys_leaves[0]:
    print(f"tree{jax.tree_util.keystr(path)} = {val}")
# tree[0] = ...
# tree['weights'] = ...
```

---

## 5. Array Model

### `jax.Array`

The fundamental data structure in JAX. Key properties:

- **Immutable**: Operations return new arrays
- **Device-backed**: Lives on CPU, GPU, or TPU
- **Sharded**: Can span multiple devices (see Chapter 21)
- **Lazy**: Execution is asynchronous (use `.block_until_ready()`)

### Creating arrays

```python
import jax.numpy as jnp

# From Python/NumPy
x = jnp.array([1.0, 2.0, 3.0])
x = jnp.arange(10)
x = jnp.ones((3, 4))
x = jnp.zeros((2, 3))
x = jnp.eye(3)

# From NumPy (copies to device)
import numpy as np
x = jnp.array(np.random.randn(100))

# On specific device
x = jax.device_put(np.ones(10), jax.devices('gpu')[0])
```

### Device transfer

```python
# To device
x = jax.device_put(np_array)

# To host (NumPy)
np_array = np.asarray(x)
# or
np_array = x.to_py()

# Block until computation completes
x.block_until_ready()
```

---

## 6. Functional Programming Model

JAX requires **pure functions** — no side effects. This is fundamental because:

1. **Tracing**: Side effects are not captured in jaxprs
2. **Caching**: JIT caches by input shapes/types, not values
3. **Transformations**: `grad`, `vmap` assume mathematical purity

### Common violations

```python
# BAD: Mutation
state = {'count': 0}
def f(x):
    state['count'] += 1  # Not captured!
    return x

# BAD: Print
def f(x):
    print(x)  # Only runs during tracing
    return x

# BAD: Random state
rng = np.random.RandomState(0)
def f(x):
    return x + rng.randn()  # Stateful random
```

### Correct patterns

```python
# GOOD: Pass state explicitly
def f(state, x):
    new_state = {'count': state['count'] + 1}
    return new_state, x + 1

# GOOD: Debug printing
def f(x):
    jax.debug.print("x = {}", x)
    return x

# GOOD: JAX random
key = jax.random.key(42)
def f(key, x):
    noise = jax.random.normal(key, x.shape)
    return x + noise
```

---

## 7. API Layering

JAX provides three levels of API, from high to low level:

```
User Code
    │
    ▼
jax.numpy          ← NumPy-compatible, high-level
    │
    ▼
jax.lax            ← Lower-level, stricter, more powerful
    │
    ▼
XLA                ← Accelerated Linear Algebra compiler
    │
    ▼
Hardware (CPU / GPU / TPU)
```

### jax.numpy vs jax.lax

| Feature | `jax.numpy` | `jax.lax` |
|---|---|---|
| Type promotion | Automatic | Explicit required |
| Broadcasting | Automatic | Manual (`broadcast_in_dim`) |
| API style | NumPy-compatible | Functional, explicit |
| Flexibility | Standard operations | General operations |

```python
# jax.numpy — automatic promotion
jnp.add(1, 1.0)  # Works

# jax.lax — requires explicit types
lax.add(jnp.float32(1), 1.0)  # Works
lax.add(1, 1.0)  # TypeError!
```

---

## 8. Asynchronous Dispatch

JAX uses **asynchronous dispatch**: operations return immediately, before the device finishes computing. This hides latency but means:

```python
x = jnp.dot(a, b)  # Returns immediately (async)
print(x)            # Waits for result (blocks)
x_copy = x.copy()   # Waits for result (blocks)
x.block_until_ready()  # Explicit wait
```

### When to block

- **Benchmarking**: Always use `block_until_ready()`
- **Debugging**: Print or copy forces a block
- **Production**: Let async run; only block at checkpoints

---

## 9. Compilation Model

### XLA compilation

```
Python Function
    │  (tracing)
    ▼
Jaxpr
    │  (lowering)
    ▼
HLO / StableHLO
    │  (XLA compilation)
    ▼
Device Executable
    │  (execution)
    ▼
Result
```

### Caching

`jax.jit` caches compiled executables keyed by:
- Function identity (hash)
- Input shapes and dtypes
- Static argument values (if `static_argnums`)

```python
f_jit = jax.jit(f)

# First call: traces + compiles (slow)
f_jit(jnp.ones(3))       # shape=(3,), dtype=float32 → cache key A

# Second call: cache hit (fast)
f_jit(jnp.ones(3))       # Same key A → reuse

# Third call: recompiles (different shape)
f_jit(jnp.ones(5))       # shape=(5,) → cache key B → recompile

# Fourth call: cache hit
f_jit(jnp.ones(5))       # Same key B → reuse
```

---

## 10. Common Gotchas Summary

| Gotcha | Symptom | Fix |
|---|---|---|
| Impure function | Wrong results or silent errors | Use pure functions |
| Value-dependent control flow | ConcretizationTypeError | Use `jax.lax.cond`/`where` |
| In-place mutation | `TypeError` | Create new arrays |
| `None` not a leaf | Missing elements in pytree | Use `is_leaf=lambda x: x is None` |
| Array shape as tuple leaf | `jnp.ones` called on integers | Wrap shape in `np.array` |
| Async dispatch | Timing wrong | Use `block_until_ready()` |
| Non-hashable JIT args | Recompilation | Use `static_argnums` |
| `np.random` in JIT | Same values every call | Use `jax.random` |
