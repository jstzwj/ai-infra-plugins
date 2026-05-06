# 41 - jax.extend API

## Overview

`jax.extend` provides access to JAX internal APIs for library developers. These APIs are not subject to backward compatibility guarantees and may change between releases.

---

## 1. Module Structure

```
jax.extend/
├── core/            # Core types and primitives
├── ffi/             # Foreign Function Interface
├── linear_util/     # Linear transformation utilities
├── random/          # PRNG internals
├── mesh/            # Mesh internals
└── ...
```

---

## 2. Core Module

### `jax.extend.core`

Provides direct access to JAX's core types:

```python
from jax.extend import core

# Access core types
Primitive = core.Primitive
AbstractValue = core.AbstractValue
ShapedArray = core.ShapedArray
ConcreteArray = core.ConcreteArray
Var = core.Var
Literal = core.Literal
Jaxpr = core.Jaxpr
JaxprEqn = core.JaxprEqn
ClosedJaxpr = core.ClosedJaxpr
Trace = core.Trace
Tracer = core.Tracer
```

### Creating custom primitives

```python
from jax.extend.core import Primitive

my_prim = Primitive("my_prim")

# Define abstract evaluation
@my_prim.def_abstract_eval
def my_prim_abstract(x_aval, **params):
    return core.ShapedArray(x_aval.shape, x_aval.dtype)

# Define implementation
@my_prim.def_impl
def my_prim_impl(x, **params):
    return x * 2

# Bind
result = my_prim.bind(jnp.ones(3))
```

---

## 3. FFI Module

### `jax.extend.ffi`

```python
from jax.extend import ffi

# Create XLA custom call lowering
lowering = ffi.ffi_lowering("my_custom_call")

# Use in a primitive
@my_prim.def_lowering
def my_prim_lowering(ctx, x):
    return lowering(ctx, x, param1="value")
```

---

## 4. Random Module

### `jax.extend.random`

```python
from jax.extend import random

# Access PRNG internals
PRNGImpl = random.PRNGImpl
# Create custom PRNG implementations
```

---

## 5. Linear Utilities

### `jax.extend.linear_util`

```python
from jax.extend import linear_util

# Transformation utilities
# Used internally by jit, grad, vmap
```

---

## 6. Interpreter Registration

### Register transformation rules

```python
from jax.interpreters import ad, batching, mlir

# JVP rule
ad.primitive_jvps[my_prim] = my_jvp_rule

# VJP rule
ad.primitive_vjps[my_prim] = my_vjp_rule

# Batching rule
batching.primitive_batchers[my_prim] = my_batch_rule

# MLIR lowering
mlir.register_lowering(my_prim, my_lowering_rule)
```

---

## 7. Stability Guarantees

| Module | Stability | Notes |
|---|---|---|
| `jax.numpy` | Stable | NumPy-compatible API |
| `jax.lax` | Stable | Low-level operations |
| `jax.extend.core` | Unstable | May change without notice |
| `jax.extend.ffi` | Unstable | For library developers only |
| `jax._src` | Private | Do not use directly |

### Best practices

1. **Import from `jax.extend`**, not `jax._src`
2. **Pin JAX version** in your library dependencies
3. **Test against JAX nightlies** for early warning of breaking changes
4. **Use public APIs** when available
5. **File issues** on GitHub for missing public API needs

---

## 8. Common Use Cases

### Custom XLA operation

```python
from jax.extend.core import Primitive
from jax.interpreters import mlir, ad, batching

# 1. Define primitive
my_op_p = Primitive("my_op")
my_op_p.multiple_results = False

# 2. Abstract eval
@my_op_p.def_abstract_eval
def my_op_abstract(x):
    return core.ShapedArray(x.shape, x.dtype)

# 3. Implementation
@my_op_p.def_impl
def my_op_impl(x):
    return x * 2  # CPU fallback

# 4. MLIR lowering
def my_op_lowering(ctx, x):
    # Generate HLO
    ...
mlir.register_lowering(my_op_p, my_op_lowering)

# 5. JVP rule
def my_op_jvp(primals, tangents):
    x, = primals
    dx, = tangents
    return my_op(x), my_op(dx) * 2
ad.primitive_jvps[my_op_p] = my_op_jvp

# 6. Batching rule
def my_op_batch(args, dims):
    x, = args
    d, = dims
    return my_op(x), d
batching.primitive_batchers[my_op_p] = my_op_batch
```

---

## 9. Version Compatibility

```python
import jax

# Check version
jax.__version__       # e.g., "0.4.35"
jax.__version_info__  # e.g., (0, 4, 35)

# Check features
hasattr(jax.extend, 'ffi')  # True for recent versions
```
