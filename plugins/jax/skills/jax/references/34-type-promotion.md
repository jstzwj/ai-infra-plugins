# 34 - Type Promotion

## Overview

JAX follows NumPy's type promotion rules with some extensions for bfloat16. Understanding type promotion is essential for writing numerically correct JAX code.

---

## 1. Promotion Rules

### Basic rules

1. If both operands are the same type, result is that type
2. If one operand has a larger category (int < float < complex), result is the larger category
3. Within a category, the larger type wins
4. Mixing signed and unsigned promotes to signed

### Category hierarchy

```
bool < int < uint < float < complex
  ↑       ↑       ↑        ↑
  |       |       |        |
 8-bit → 16-bit → 32-bit → 64-bit
```

---

## 2. Type Promotion Table

### Numeric types

|       | bool | i8 | i16 | i32 | i64 | u8 | u16 | u32 | u64 | f16 | bf16 | f32 | f64 | c64 | c128 |
|-------|------|-----|------|------|------|------|------|------|------|------|------|------|------|------|------|
| bool  | bool | i32 | i32  | i32  | i64  | u8  | u16  | u32  | u64  | f16  | bf16 | f32  | f64  | c64  | c128 |
| i8    | i32  | i8  | i16  | i32  | i64  | i16 | i16  | i32  | i64  | f16  | bf16 | f32  | f64  | c64  | c128 |
| i16   | i32  | i16 | i16  | i32  | i64  | i16 | i16  | i32  | i64  | f32  | f32  | f32  | f64  | c64  | c128 |
| i32   | i32  | i32 | i32  | i32  | i64  | i32 | i32  | i32  | i64  | f32  | f32  | f32  | f64  | c64  | c128 |
| i64   | i64  | i64 | i64  | i64  | i64  | i64 | i64  | i64  | i64  | f64  | f64  | f64  | f64  | c128 | c128 |
| u8    | u8   | i16 | i16  | i32  | i64  | u8  | u16  | u32  | u64  | f16  | bf16 | f32  | f64  | c64  | c128 |
| u16   | u16  | i16 | i16  | i32  | i64  | u16 | u16  | u32  | u64  | f32  | f32  | f32  | f64  | c64  | c128 |
| u32   | u32  | i32 | i32  | i32  | i64  | u32 | u32  | u32  | u64  | f32  | f32  | f32  | f64  | c64  | c128 |
| u64   | u64  | i64 | i64  | i64  | i64  | u64 | u64  | u64  | u64  | f64  | f64  | f64  | f64  | c128 | c128 |
| f16   | f16  | f16 | f32  | f32  | f64  | f16 | f32  | f32  | f64  | f16  | f32  | f32  | f64  | c32  | c64  |
| bf16  | bf16 | bf16| f32  | f32  | f64  | bf16| f32  | f32  | f64  | f32  | bf16 | f32  | f64  | c64  | c128 |
| f32   | f32  | f32 | f32  | f32  | f64  | f32 | f32  | f32  | f64  | f32  | f32  | f32  | f64  | c64  | c128 |
| f64   | f64  | f64 | f64  | f64  | f64  | f64 | f64  | f64  | f64  | f64  | f64  | f64  | f64  | c128 | c128 |
| c64   | c64  | c64 | c64  | c64  | c128 | c64 | c64  | c64  | c128 | c32  | c64  | c64  | c128 | c64  | c128 |
| c128  | c128 | c128| c128 | c128 | c128 | c128| c128 | c128 | c128 | c64  | c128 | c128 | c128 | c128 | c128 |

---

## 3. Weak Types

JAX uses "weak types" for Python scalars to avoid unnecessary upcasting:

```python
# Python int → weak int (promotes to match, not upcast to i64)
jnp.add(1, jnp.int32(5))     # Result: int32 (not int64)
jnp.add(1, jnp.int64(5))     # Result: int64

# Python float → weak float
jnp.add(1.0, jnp.float16(5)) # Result: float16 (not float64)
jnp.add(1.0, jnp.float32(5)) # Result: float32

# Disable x64 by default
jax.config.update("jax_enable_x64", False)  # Default
jnp.array(1.0).dtype  # float32 (not float64)
```

### Weak type rules

| Python type | Weak dtype |
|---|---|
| `bool` | weak bool |
| `int` | weak int |
| `float` | weak float |
| `complex` | weak complex |

Weak types promote to match the strongest array type involved.

---

## 4. x64 Mode

```python
# Enable 64-bit types
jax.config.update("jax_enable_x64", True)

# Now Python floats create float64
jnp.array(1.0).dtype  # float64

# Disable 64-bit (default)
jax.config.update("jax_enable_x64", False)
jnp.array(1.0).dtype  # float32
```

---

## 5. Explicit Type Control

### `astype`

```python
x = jnp.ones(3, dtype=jnp.float32)
x_f16 = x.astype(jnp.float16)
x_bf16 = x.astype(jnp.bfloat16)
```

### `jnp.asarray` with dtype

```python
x = jnp.asarray([1, 2, 3], dtype=jnp.float32)
```

### `lax.convert_element_type`

```python
from jax import lax
x = lax.convert_element_type(x, jnp.bfloat16)
```

---

## 6. bfloat16 Promotion

bfloat16 (bf16) is JAX-specific with the same range as float32 but lower precision:

```python
# bf16 + f32 → f32
jnp.add(jnp.ones(3, jnp.bfloat16), jnp.ones(3, jnp.float32)).dtype
# dtype('float32')

# bf16 + f16 → f32 (both are "small" floats)
jnp.add(jnp.ones(3, jnp.bfloat16), jnp.ones(3, jnp.float16)).dtype
# dtype('float32')
```

---

## 7. Promotion APIs

```python
# Query promotion result
jnp.result_type(jnp.float16, jnp.float32)  # dtype('float32')
jnp.promote_types(jnp.int32, jnp.float32)   # dtype('float64')

# Check if promotion is possible
jnp.can_cast(jnp.float16, jnp.float32)  # True
jnp.can_cast(jnp.float32, jnp.int32)     # False (float → int not allowed)
```

---

## 8. Common Pitfalls

### Accidental float64

```python
# BAD: Python float literal → float64 with x64 enabled
x = jnp.array(1.0)  # float64 if x64 enabled

# GOOD: Explicit dtype
x = jnp.array(1.0, dtype=jnp.float32)
```

### Integer overflow

```python
# int32 can overflow
x = jnp.array(2147483647, dtype=jnp.int32)
y = x + 1  # -2147483648 (overflow!)

# Use int64 for large values
x = jnp.array(2147483647, dtype=jnp.int64)
y = x + 1  # 2147483648 (correct)
```

### Mixed precision training

```python
# Compute in bf16, accumulate in f32
@jax.jit
def matmul_mixed(x, w):
    x_bf16 = x.astype(jnp.bfloat16)
    w_bf16 = w.astype(jnp.bfloat16)
    result_bf16 = x_bf16 @ w_bf16
    return result_bf16.astype(jnp.float32)  # Accumulate in f32
```
