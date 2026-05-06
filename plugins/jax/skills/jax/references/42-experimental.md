# 42 - Experimental Features

## Overview

JAX includes experimental features that are not yet part of the stable API. These may change or be removed in future versions.

---

## 1. jax.experimental

### Module list

| Module | Description | Status |
|---|---|---|
| `jax.experimental.mesh_utils` | Device mesh utilities | Moving to stable |
| `jax.experimental.pallas` | Kernel programming | Active development |
| `jax.experimental.shard_map` | Per-device programming | Now stable (`jax.shard_map`) |
| `jax.experimental.checkify` | Functional error checking | Active development |
| `jax.experimental.io_callback` | Side-effecting callbacks | Experimental |
| `jax.experimental.export` | Model export | Moving to stable |
| `jax.experimental.sparse` | Sparse operations | Active development |
| `jax.experimental.jax2tf` | JAX to TensorFlow | Being replaced by export |
| `jax.experimental.array_serialization` | Array serialization | Experimental |

---

## 2. Checkify

Functional error checking that works with JAX transformations:

```python
from jax.experimental import checkify

@checkify.checkify
def f(x):
    checkify.check(x > 0, "x must be positive, got {}", x)
    return jnp.log(x)

# Returns (error, result)
err, result = f(1.0)  # No error
err, result = f(-1.0)  # err is set

# Check the error
err.throw()  # Raises if error occurred
err.get()    # Returns error message or None
```

### Error categories

```python
# Enable specific checks
checked_f = checkify.checkify(
    f,
    errors=checkify.all_checks  # All possible checks
)

# Available categories
checkify.user_checks     # Only explicit check() calls
checkify.nan_checks      # NaN detection
checkify.float_checks    # Float infinity/NaN
checkify.div_checks      # Division by zero
checkify.index_checks    # Out-of-bounds indexing
checkify.all_checks      # All of the above
```

### With JIT

```python
@jax.jit
@checkify.checkify
def f(x):
    checkify.check(jnp.all(x >= 0), "negative values")
    return jnp.sqrt(x)

err, result = f(jnp.array([1.0, 2.0]))
err.throw()  # No error
```

---

## 3. Sparse Operations

```python
from jax.experimental import sparse

# Create sparse matrix
dense = jnp.array([[1., 0., 2.], [0., 0., 3.], [4., 0., 0.]])
sp_mat = sparse.BCOO.fromdense(dense)
# BCOO: Batched Coordinate format

# Sparse operations
sp_result = sp_mat @ jnp.ones(3)

# Properties
sp_mat.nse     # Number of stored elements
sp_mat.shape   # (3, 3)
sp_mat.dtype   # float32
```

### Supported operations

| Operation | Supported |
|---|---|
| Matrix multiply (sp @ dense) | Yes |
| Element-wise ops | Partial |
| Slicing | Limited |
| Reduction | Yes |
| Transpose | Yes |

---

## 4. Export (jax.export)

```python
import jax.export

# Export a function to StableHLO
def f(x, y):
    return jnp.dot(x, y)

# Export with fixed shapes
exported = jax.export.export(f)(jnp.ones(3), jnp.ones(3))

# Export with polymorphic shapes
exported = jax.export.export(
    shape_polymorphism='(b, n), (n, m) -> (b, m)'
)(f)

# Serialize
serialized = exported.serialize()

# Lower to MLIR
mlir_text = exported.mlir_module()
```

---

## 5. Array Serialization

```python
from jax.experimental import array_serialization

# Save/load sharded arrays
# Useful for checkpointing distributed training
```

---

## 6. jax2tf (Legacy)

```python
from jax.experimental import jax2tf

# Convert JAX function to TensorFlow
tf_fn = jax2tf.convert(f)

# Save as SavedModel
tf.saved_model.save(tf_fn, '/path/to/model')

# Note: Being replaced by jax.export
```

---

## 7. Migration from Experimental to Stable

| Feature | Experimental | Stable | Migration |
|---|---|---|---|
| shard_map | `jax.experimental.shard_map` | `jax.shard_map` | Change import |
| Pallas | `jax.experimental.pallas` | `jax.pallas` (future) | Pending |
| Checkify | `jax.experimental.checkify` | `jax.checkify` (future) | Pending |
| Export | `jax.experimental.export` | `jax.export` | Change import |
