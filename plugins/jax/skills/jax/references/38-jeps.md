# 38 - JAX Enhancement Proposals (JEPs)

## Overview

JAX Enhancement Proposals (JEPs) are design documents describing significant changes to JAX. This chapter summarizes the most important JEPs that shape the current JAX API.

---

## 1. Key JEPs

### JEP 263: Typed Keys

**Status**: Implemented (default since JAX 0.4.16)

Previously, PRNG keys were untyped uint32 arrays. Now they carry their PRNG implementation in the dtype:

```python
# Old style (legacy)
key = jax.random.PRNGKey(42)  # uint32[2] array

# New style (typed keys)
key = jax.random.key(42)      # key<fry> scalar
```

Benefits:
- Type safety: can't accidentally use a key as a regular array
- PRNG implementation is encoded in the type
- `jax.random.split` preserves the type

---

### JEP 2026: Shape Polymorphic Export

**Status**: Implemented

Allows exporting JAX functions with symbolic (polymorphic) shapes:

```python
import jax
import jax.export

def f(x, y):
    return x @ y

# Export with symbolic batch dimension
exported = jax.export.export(
    shape_polymorphism='(b, n), (n, m) -> (b, m)'
)(f)

# Can now call with any concrete batch size
```

Benefits:
- Compile once, run with multiple shapes
- StableHLO serialization
- Cross-language deployment

---

### JEP 9263: Sharding

**Status**: Implemented

Introduced the modern sharding API:

```python
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

mesh = Mesh(jax.devices(), ('x',))
sharding = NamedSharding(mesh, P('x', None))

with mesh:
    x = jax.device_put(data, sharding)
```

Key concepts:
- `Mesh`: Logical device grid with named axes
- `PartitionSpec`: How to shard across mesh axes
- `NamedSharding`: Mesh + PartitionSpec combination
- `jax.Array`: Multi-device array with sharding info

---

### JEP 14273: Explicit-Sharding Mode

**Status**: Implemented

Added explicit sharding annotations for fine-grained control:

```python
from jax.sharding import PartitionSpec as P

@jax.jit
def f(x):
    # Explicitly specify how intermediate should be sharded
    y = jax.lax.with_sharding_constraint(x, P('x', None))
    return y @ w
```

---

### JEP 17111: shard_map

**Status**: Implemented

Introduced `shard_map` for per-device programming with explicit collectives:

```python
from jax.shard_map import shard_map

@shard_map(mesh, in_specs=P('x', None), out_specs=P('x', None))
def f(x):
    # x is the local shard
    return jax.lax.psum(x, 'x')
```

Key features:
- Explicit per-device code
- Direct access to collective operations
- Composable with `jit` and `grad`
- Transposition support for automatic differentiation

---

## 2. Other Notable JEPs

### JEP 15806: Auto-sharding

The auto-sharding mode where the compiler determines optimal sharding:

```python
# Auto mode: compiler decides sharding
with mesh:
    result = jax.jit(f)(x)  # XLA propagates sharding
```

### JEP 18188: Manual-sharding Mode

Manual control over sharding within a function:

```python
@jax.jit
def f(x):
    # Must use with_sharding_constraint for all intermediates
    y = jax.lax.with_sharding_constraint(x, P('x', None))
    return y
```

---

## 3. JEP Categories

| Category | JEPs | Topic |
|---|---|---|
| Random | 263 | Typed PRNG keys |
| Export | 2026 | Shape-polymorphic export |
| Sharding | 9263, 14273, 17111, 15806, 18188 | Distributed computing |
| Autodiff | — | Custom derivative rules |
| Pallas | — | Kernel programming |

---

## 4. Tracking JEPs

JEPs are tracked in the JAX repository:
- GitHub: `jax-ml/jax` repository
- Documentation: `jax.readthedocs.io`
- Design docs: Often linked from PR descriptions

---

## 5. Migration Guides

### Migrating to typed keys (JEP 263)

```python
# Before
key = jax.random.PRNGKey(42)
subkeys = jax.random.split(key, 5)

# After (same behavior, better type safety)
key = jax.random.key(42)
subkeys = jax.random.split(key, 5)

# Migration is mostly transparent
# Just replace PRNGKey with key()
```

### Migrating to modern sharding (JEP 9263)

```python
# Before (pmap)
result = jax.pmap(f)(sharded_data)

# After (shard_map or auto-sharding)
mesh = Mesh(jax.devices(), ('x',))
with mesh:
    result = jax.jit(f)(data)
```
