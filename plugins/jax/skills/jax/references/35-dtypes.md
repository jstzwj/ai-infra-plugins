# 35 - Data Types (DTypes)

## Overview

JAX supports a comprehensive set of data types including NumPy-compatible types and JAX-specific types like bfloat16. This chapter covers all dtypes, their properties, and best practices.

---

## 1. Supported DTypes

### Integer types

| JAX dtype | NumPy equivalent | Range | Size |
|---|---|---|---|
| `jnp.bool_` | `np.bool_` | True/False | 1 byte |
| `jnp.int8` | `np.int8` | -128 to 127 | 1 byte |
| `jnp.int16` | `np.int16` | -32768 to 32767 | 2 bytes |
| `jnp.int32` | `np.int32` | -2^31 to 2^31-1 | 4 bytes |
| `jnp.int64` | `np.int64` | -2^63 to 2^63-1 | 8 bytes |
| `jnp.uint8` | `np.uint8` | 0 to 255 | 1 byte |
| `jnp.uint16` | `np.uint16` | 0 to 65535 | 2 bytes |
| `jnp.uint32` | `np.uint32` | 0 to 2^32-1 | 4 bytes |
| `jnp.uint64` | `np.uint64` | 0 to 2^64-1 | 8 bytes |

### Floating-point types

| JAX dtype | NumPy equivalent | Precision | Range | Size |
|---|---|---|---|---|
| `jnp.float16` | `np.float16` | 3 decimal digits | ±65504 | 2 bytes |
| `jnp.bfloat16` | — | 2 decimal digits | ±3.4e38 | 2 bytes |
| `jnp.float32` | `np.float32` | 7 decimal digits | ±3.4e38 | 4 bytes |
| `jnp.float64` | `np.float64` | 15 decimal digits | ±1.8e308 | 8 bytes |

### Complex types

| JAX dtype | NumPy equivalent | Components | Size |
|---|---|---|---|
| `jnp.complex64` | `np.complex64` | 2×float32 | 8 bytes |
| `jnp.complex128` | `np.complex128` | 2×float64 | 16 bytes |

---

## 2. bfloat16 Details

### Comparison with float16 and float32

| Property | float16 | bfloat16 | float32 |
|---|---|---|---|
| Sign bits | 1 | 1 | 1 |
| Exponent bits | 5 | 8 | 8 |
| Mantissa bits | 10 | 7 | 23 |
| Total bits | 16 | 16 | 16 |
| Exponent range | ±65504 | ±3.4e38 | ±3.4e38 |
| Decimal precision | 3.3 | 2.0 | 7.2 |

### Key differences

- **bfloat16** has the same exponent range as float32 (8 exponent bits)
- **bfloat16** has lower precision than float16 (7 vs 10 mantissa bits)
- **bfloat16** avoids overflow/underflow issues common with float16
- **bfloat16** is the preferred format for deep learning on TPUs and GPUs

### Usage

```python
# Create bfloat16 arrays
x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.bfloat16)

# Convert to/from bfloat16
x_bf16 = x.astype(jnp.bfloat16)
x_f32 = x_bf16.astype(jnp.float32)

# bfloat16 arithmetic
y = x_bf16 + x_bf16  # Result is bfloat16
```

### Limitations

```python
# bfloat16 does not support all NumPy operations
# Some operations may implicitly promote to float32

# No native Python bfloat16 scalar
# x = jnp.bfloat16(1.0)  # Works but returns DeviceArray
```

---

## 3. Type Information Functions

### `jnp.finfo` — Floating-point info

```python
info = jnp.finfo(jnp.float32)
info.bits        # 32
info.eps         # 1.1920929e-07  (machine epsilon)
info.max         # 3.4028235e+38
info.min         # -3.4028235e+38
info.tiny        # 1.1754944e-38  (smallest positive normal)
info.smallest_normal  # same as tiny
info.dtype       # dtype('float32')
```

### `jnp.iinfo` — Integer info

```python
info = jnp.iinfo(jnp.int32)
info.min    # -2147483648
info.max    # 2147483647
info.dtype  # dtype('int32')
info.bits   # 32
```

### `jnp.issubdtype`

```python
jnp.issubdtype(jnp.float32, jnp.floating)  # True
jnp.issubdtype(jnp.int32, jnp.integer)      # True
jnp.issubdtype(jnp.bool_, jnp.integer)      # False
```

---

## 4. Type Categories

```python
# Type category constants
jnp.bool_       # Boolean
jnp.integer     # All integer types
jnp.signedinteger  # Signed integers (int8, int16, int32, int64)
jnp.unsignedinteger  # Unsigned integers (uint8, uint16, uint32, uint64)
jnp.floating    # All float types (float16, bfloat16, float32, float64)
jnp.complexfloating  # Complex types (complex64, complex128)
jnp.inexact     # floating + complexfloating
jnp.number      # integer + inexact
jnp.generic     # All dtypes
```

---

## 5. Creating Arrays with Specific DTypes

```python
# Explicit dtype
x = jnp.array([1, 2, 3], dtype=jnp.float32)

# From Python types
x = jnp.float32(3.14)
x = jnp.int32(42)

# Type-preserving operations
jnp.zeros(3, dtype=jnp.bfloat16)
jnp.ones((2, 3), dtype=jnp.float16)
jnp.arange(10, dtype=jnp.int64)

# *_like functions preserve dtype
x = jnp.ones(3, dtype=jnp.float16)
y = jnp.zeros_like(x)   # Same dtype: float16
z = jnp.empty_like(x)   # Same dtype: float16
```

---

## 6. Mixed Precision Best Practices

### Automatic mixed precision

```python
# Compute in lower precision, accumulate in higher
@jax.jit
def linear(x, w, b):
    # Cast to bf16 for matmul
    x_bf = x.astype(jnp.bfloat16)
    w_bf = w.astype(jnp.bfloat16)
    # Matmul in bf16, accumulate in f32
    out = jnp.dot(x_bf, w_bf).astype(jnp.float32) + b
    return out
```

### Loss scaling for float16

```python
# Scale loss to prevent underflow in float16
LOSS_SCALE = 1024.0

@jax.jit
def train_step(params, x, y):
    def scaled_loss(params):
        loss = loss_fn(params, x, y)
        return loss * LOSS_SCALE

    scaled_grads = jax.grad(scaled_loss)(params)
    grads = jax.tree.map(lambda g: g / LOSS_SCALE, scaled_grads)
    return grads
```

---

## 7. Device-Specific Considerations

### GPU

- float16 (half): Supported on all NVIDIA GPUs
- bfloat16: Supported on Ampere+ (SM 80+)
- TensorFloat-32: Default matmul mode on Ampere+

```python
# Control TF32 behavior
jax.config.update("jax_default_matmul_precision", "highest")  # float32
jax.config.update("jax_default_matmul_precision", "high")     # TF32
jax.config.update("jax_default_matmul_precision", "bfloat16") # bf16
```

### TPU

- bfloat16: Native format, most efficient
- float32: Supported (emulated on some generations)
- float16: Not natively supported

---

## 8. Serialization Considerations

```python
# bfloat16 arrays need conversion for serialization
import numpy as np

x_bf16 = jnp.array([1.0, 2.0], dtype=jnp.bfloat16)

# Save as float32 (bfloat16 not directly serializable in many formats)
np.save('data.npy', np.asarray(x_bf16.astype(jnp.float32)))

# Load and convert back
loaded = jnp.array(np.load('data.npy')).astype(jnp.bfloat16)
```
