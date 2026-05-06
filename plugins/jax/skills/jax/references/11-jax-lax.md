# jax.lax — Mid-Level Operations API Reference

`jax.lax` is the **mid-level API** in JAX, sitting between the high-level `jax.numpy` interface and the low-level XLA compiler operations. It provides a comprehensive set of functional, side-effect-free operations that map directly to XLA HLO (High-Level Optimizer) instructions. Every `jax.numpy` function is ultimately implemented in terms of `jax.lax` primitives.

Understanding `jax.lax` is essential for:
- Writing custom ops that need fine-grained control
- Debugging performance issues at the XLA level
- Building control flow primitives (scans, loops, conditionals)
- Implementing operations not available in `jax.numpy`

**Key principle:** All `jax.lax` operations are **pure functions** — they never modify inputs in place.

---

## Table of Contents

1. [Arithmetic Operations](#arithmetic-operations)
2. [Exponential / Logarithmic / Trigonometric Functions](#exponential--logarithmic--trigonometric-functions)
3. [Comparison Operations](#comparison-operations)
4. [General Shape and Layout Operations](#general-shape-and-layout-operations)
5. [Indexing and Slicing Operations](#indexing-and-slicing-operations)
6. [Gather and Scatter Operations](#gather-and-scatter-operations)
7. [Control Flow Operations](#control-flow-operations)
8. [Linear Algebra Operations](#linear-algebra-operations)
9. [Reduction Operations](#reduction-operations)
10. [Bitwise Operations](#bitwise-operations)
11. [Type Conversion Operations](#type-conversion-operations)
12. [Special Mathematical Functions](#special-mathematical-functions)
13. [FFT Operations](#fft-operations)
14. [Random Number Generation](#random-number-generation)
15. [Other Operations](#other-operations)

---

## Arithmetic Operations

### lax.add

```python
jax.lax.add(x, y)
```

Element-wise addition. Both operands must have the same dtype, or one must be a scalar that can be broadcast.

```python
import jax
import jax.numpy as jnp
from jax import lax

x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])
result = lax.add(x, y)
# [5.0, 7.0, 9.0]

# Scalar broadcast
result = lax.add(x, 10.0)
# [11.0, 12.0, 13.0]
```

### lax.sub

```python
jax.lax.sub(x, y)
```

Element-wise subtraction.

```python
x = jnp.array([5.0, 6.0, 7.0])
y = jnp.array([1.0, 2.0, 3.0])
result = lax.sub(x, y)
# [4.0, 4.0, 4.0]
```

### lax.mul

```python
jax.lax.mul(x, y)
```

Element-wise multiplication.

```python
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])
result = lax.mul(x, y)
# [4.0, 10.0, 18.0]
```

### lax.div

```python
jax.lax.div(x, y)
```

Element-wise division. For integer division, truncates toward zero (matching C semantics).

```python
x = jnp.array([10.0, 20.0, 30.0])
y = jnp.array([2.0, 3.0, 4.0])
result = lax.div(x, y)
# [5.0, 6.666..., 7.5]

# Integer division truncates toward zero
x = jnp.array([7, -7])
y = jnp.array([3, 3])
result = lax.div(x, y)
# [2, -2]
```

### lax.rem

```python
jax.lax.rem(x, y)
```

Element-wise remainder. The result has the same sign as the dividend `x`.

```python
x = jnp.array([7.0, -7.0, 7.0, -7.0])
y = jnp.array([3.0, 3.0, -3.0, -3.0])
result = lax.rem(x, y)
# [1.0, -1.0, 1.0, -1.0]
```

### lax.neg

```python
jax.lax.neg(x)
```

Element-wise negation.

```python
x = jnp.array([1.0, -2.0, 3.0])
result = lax.neg(x)
# [-1.0, 2.0, -3.0]
```

### lax.sign

```python
jax.lax.sign(x)
```

Element-wise sign function. Returns -1, 0, or 1 depending on the sign of each element. For complex numbers, returns `x / |x|`.

```python
x = jnp.array([-3.0, 0.0, 5.0])
result = lax.sign(x)
# [-1.0, 0.0, 1.0]
```

### lax.nextafter

```python
jax.lax.nextafter(x, y)
```

Returns the next representable floating-point value after `x` in the direction of `y`.

```python
x = jnp.float32(1.0)
y = jnp.float32(2.0)
result = lax.nextafter(x, y)
# 1.0000001 (the next float32 after 1.0)

x = jnp.float32(1.0)
y = jnp.float32(0.0)
result = lax.nextafter(x, y)
# 0.99999994 (the next float32 before 1.0)
```

### lax.abs

```python
jax.lax.abs(x)
```

Element-wise absolute value. For complex inputs, returns the magnitude.

```python
x = jnp.array([-3.0, 0.0, 5.0])
result = lax.abs(x)
# [3.0, 0.0, 5.0]

x = jnp.array([3.0 + 4.0j])
result = lax.abs(x)
# [5.0]
```

### lax.floor

```python
jax.lax.floor(x)
```

Element-wise floor. Rounds toward negative infinity.

```python
x = jnp.array([-2.7, -1.2, 0.0, 1.2, 2.7])
result = lax.floor(x)
# [-3.0, -2.0, 0.0, 1.0, 2.0]
```

### lax.ceil

```python
jax.lax.ceil(x)
```

Element-wise ceiling. Rounds toward positive infinity.

```python
x = jnp.array([-2.7, -1.2, 0.0, 1.2, 2.7])
result = lax.ceil(x)
# [-2.0, -1.0, 0.0, 2.0, 3.0]
```

### lax.round

```python
jax.lax.round(x, rounding_method=lax.RoundingMethod.AWAY_FROM_ZERO)
```

Element-wise rounding. The `rounding_method` parameter controls behavior at the midpoint:
- `RoundingMethod.AWAY_FROM_ZERO` (default): rounds 0.5 away from zero
- `RoundingMethod.TO_NEAREST_EVEN`: rounds to nearest even (banker's rounding)

```python
x = jnp.array([0.5, 1.5, 2.5, 3.5])
result = lax.round(x, rounding_method=lax.RoundingMethod.AWAY_FROM_ZERO)
# [1.0, 2.0, 3.0, 4.0]

result = lax.round(x, rounding_method=lax.RoundingMethod.TO_NEAREST_EVEN)
# [0.0, 2.0, 2.0, 4.0]
```

### lax.max

```python
jax.lax.max(x, y)
```

Element-wise maximum. For NaN inputs, returns NaN (unlike `jnp.maximum` which returns the non-NaN value).

```python
x = jnp.array([1.0, 4.0, 2.0])
y = jnp.array([3.0, 2.0, 5.0])
result = lax.max(x, y)
# [3.0, 4.0, 5.0]
```

### lax.min

```python
jax.lax.min(x, y)
```

Element-wise minimum.

```python
x = jnp.array([1.0, 4.0, 2.0])
y = jnp.array([3.0, 2.0, 5.0])
result = lax.min(x, y)
# [1.0, 2.0, 2.0]
```

### lax.clamp

```python
jax.lax.clamp(min, x, max)
```

Element-wise clamp. Clamps every element of `x` to be in the range `[min, max]`. All three arguments broadcast together.

```python
x = jnp.array([0.0, 5.0, 10.0, 15.0, 20.0])
result = lax.clamp(3.0, x, 12.0)
# [3.0, 5.0, 10.0, 12.0, 12.0]
```

---

## Exponential / Logarithmic / Trigonometric Functions

### Exponential and Logarithmic Functions

#### lax.exp

```python
jax.lax.exp(x)
```

Element-wise exponential: `e^x`.

```python
x = jnp.array([0.0, 1.0, 2.0])
result = lax.exp(x)
# [1.0, 2.7182817, 7.389056]
```

#### lax.exp2

```python
jax.lax.exp2(x)
```

Element-wise base-2 exponential: `2^x`.

```python
x = jnp.array([0.0, 1.0, 2.0, 3.0])
result = lax.exp2(x)
# [1.0, 2.0, 4.0, 8.0]
```

#### lax.expm1

```python
jax.lax.expm1(x)
```

Element-wise `exp(x) - 1`. More numerically stable than `exp(x) - 1` for small `x`.

```python
x = jnp.array([0.0, 1e-15, 1e-10])
result = lax.expm1(x)
# [0.0, 1e-15, 1e-10] -- accurate even for tiny values
```

#### lax.log

```python
jax.lax.log(x)
```

Element-wise natural logarithm.

```python
x = jnp.array([1.0, 2.718281828, 10.0])
result = lax.log(x)
# [0.0, 1.0, 2.302585]
```

#### lax.log2

```python
jax.lax.log2(x)
```

Element-wise base-2 logarithm.

```python
x = jnp.array([1.0, 2.0, 4.0, 8.0])
result = lax.log2(x)
# [0.0, 1.0, 2.0, 3.0]
```

#### lax.log10

```python
jax.lax.log10(x)
```

Element-wise base-10 logarithm.

```python
x = jnp.array([1.0, 10.0, 100.0])
result = lax.log10(x)
# [0.0, 1.0, 2.0]
```

#### lax.log1p

```python
jax.lax.log1p(x)
```

Element-wise `log(1 + x)`. More numerically stable than `log(1 + x)` for small `x`.

```python
x = jnp.array([0.0, 1e-15, 1e-10])
result = lax.log1p(x)
# [0.0, 1e-15, 1e-10]
```

### Trigonometric Functions

#### lax.sin / lax.cos / lax.tan

```python
jax.lax.sin(x)
jax.lax.cos(x)
jax.lax.tan(x)
```

Element-wise sine, cosine, and tangent. Input in radians.

```python
x = jnp.array([0.0, jnp.pi / 2, jnp.pi])
result_sin = lax.sin(x)   # [0.0, 1.0, 0.0]
result_cos = lax.cos(x)   # [1.0, 0.0, -1.0]
```

#### lax.asin / lax.acos / lax.atan

```python
jax.lax.asin(x)
jax.lax.acos(x)
jax.lax.atan(x)
```

Element-wise inverse trigonometric functions. Output in radians.

```python
x = jnp.array([0.0, 0.5, 1.0])
result = lax.asin(x)
# [0.0, 0.5236..., 1.5708...]
```

#### lax.atan2

```python
jax.lax.atan2(y, x)
```

Element-wise arc tangent of `y/x`, respecting the signs of both arguments to determine the correct quadrant.

```python
y = jnp.array([1.0, 1.0, -1.0, -1.0])
x = jnp.array([1.0, -1.0, 1.0, -1.0])
result = lax.atan2(y, x)
# [0.7854, 2.3562, -0.7854, -2.3562] (pi/4, 3pi/4, -pi/4, -3pi/4)
```

#### lax.sinh / lax.cosh / lax.tanh

```python
jax.lax.sinh(x)
jax.lax.cosh(x)
jax.lax.tanh(x)
```

Element-wise hyperbolic functions.

```python
x = jnp.array([0.0, 1.0, 2.0])
result = lax.tanh(x)
# [0.0, 0.7616..., 0.9640...]
```

#### lax.asinh / lax.acosh / lax.atanh

```python
jax.lax.asinh(x)
jax.lax.acosh(x)
jax.lax.atanh(x)
```

Element-wise inverse hyperbolic functions.

```python
x = jnp.array([0.0, 1.0, 2.0])
result = lax.asinh(x)
# [0.0, 0.8814..., 1.4436...]
```

### Power and Root Functions

#### lax.sqrt

```python
jax.lax.sqrt(x)
```

Element-wise square root.

```python
x = jnp.array([0.0, 1.0, 4.0, 9.0])
result = lax.sqrt(x)
# [0.0, 1.0, 2.0, 3.0]
```

#### lax.rsqrt

```python
jax.lax.rsqrt(x)
```

Element-wise reciprocal square root: `1 / sqrt(x)`. More numerically stable and efficient than computing the reciprocal separately.

```python
x = jnp.array([1.0, 4.0, 9.0, 16.0])
result = lax.rsqrt(x)
# [1.0, 0.5, 0.333..., 0.25]
```

#### lax.pow

```python
jax.lax.pow(x, y)
```

Element-wise power: `x^y`. Both arguments are arrays (or broadcastable scalars).

```python
x = jnp.array([2.0, 3.0, 4.0])
y = jnp.array([3.0, 2.0, 0.5])
result = lax.pow(x, y)
# [8.0, 9.0, 2.0]
```

#### lax.cbrt

```python
jax.lax.cbrt(x)
```

Element-wise cube root.

```python
x = jnp.array([1.0, 8.0, 27.0, -8.0])
result = lax.cbrt(x)
# [1.0, 2.0, 3.0, -2.0]
```

#### lax.reciprocal

```python
jax.lax.reciprocal(x)
```

Element-wise reciprocal: `1/x`.

```python
x = jnp.array([2.0, 4.0, 5.0])
result = lax.reciprocal(x)
# [0.5, 0.25, 0.2]
```

---

## Comparison Operations

### lax.eq / lax.ne / lax.gt / lax.ge / lax.lt / lax.le

```python
jax.lax.eq(x, y)    # x == y
jax.lax.ne(x, y)    # x != y
jax.lax.gt(x, y)    # x > y
jax.lax.ge(x, y)    # x >= y
jax.lax.lt(x, y)    # x < y
jax.lax.le(x, y)    # x <= y
```

Element-wise comparison operations. Return boolean arrays.

```python
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([2.0, 2.0, 1.0])

lax.eq(x, y)   # [False, True, False]
lax.ne(x, y)   # [True, False, True]
lax.gt(x, y)   # [False, False, True]
lax.ge(x, y)   # [False, True, True]
lax.lt(x, y)   # [True, False, False]
lax.le(x, y)   # [True, True, False]
```

### lax.sort

```python
jax.lax.sort(operand, dimension=-1, is_stable=True, num_keys=1)
```

Sorts `operand` along the given `dimension`. When `is_stable=True`, equal elements maintain their relative order. Supports multi-key sorting via `num_keys` (sorts by the first `num_keys` operands as keys).

```python
x = jnp.array([[3, 1, 2],
               [6, 4, 5]])
result = lax.sort(x, dimension=-1)
# [[1, 2, 3],
#  [4, 5, 6]]

# Sort along dimension 0
result = lax.sort(x, dimension=0)
# [[3, 1, 2],
#  [6, 4, 5]]
```

### lax.top_k

```python
jax.lax.top_k(operand, k)
```

Returns a tuple `(values, indices)` of the top `k` values and their indices along the last axis.

```python
x = jnp.array([5.0, 2.0, 8.0, 1.0, 9.0, 3.0])
values, indices = lax.top_k(x, 3)
# values: [9.0, 8.0, 5.0]
# indices: [4, 2, 0]
```

---

## General Shape and Layout Operations

### lax.broadcast

```python
jax.lax.broadcast(operand, sizes)
```

Broadcasts `operand` by adding leading dimensions of the given `sizes`. The original data is replicated across the new dimensions.

```python
x = jnp.array([1, 2, 3])          # shape (3,)
result = lax.broadcast(x, (2, 4))  # shape (2, 4, 3)
```

### lax.broadcast_in_dim

```python
jax.lax.broadcast_in_dim(operand, shape, broadcast_dimensions)
```

General broadcasting. The dimensions of `operand` are mapped to the output dimensions specified by `broadcast_dimensions`, and new dimensions are broadcast (size 1).

```python
x = jnp.array([[1, 2], [3, 4]])  # shape (2, 2)
# Broadcast to shape (3, 2, 2) by mapping dim 0->1, dim 1->2
result = lax.broadcast_in_dim(x, shape=(3, 2, 2), broadcast_dimensions=(1, 2))
# shape (3, 2, 2), with the (2, 2) data replicated 3 times along axis 0
```

### lax.broadcast_to_rank

```python
jax.lax.broadcast_to_rank(x, rank)
```

Adds leading size-1 dimensions to make `x` have the specified `rank`.

```python
x = jnp.array([1, 2, 3])  # shape (3,), rank 1
result = lax.broadcast_to_rank(x, 3)
# shape (1, 1, 3), rank 3
```

### lax.reshape

```python
jax.lax.reshape(operand, new_sizes, dimensions=None)
```

Reshapes `operand` to `new_sizes`. If `dimensions` is specified, the operand is first transposed according to those dimension indices, then reshaped.

```python
x = jnp.arange(6)                  # [0, 1, 2, 3, 4, 5], shape (6,)
result = lax.reshape(x, (2, 3))    # shape (2, 3)
# [[0, 1, 2],
#  [3, 4, 5]]

# Reshape with dimension reordering
result = lax.reshape(x, (3, 2), dimensions=(0,))
# [[0, 1],
#  [2, 3],
#  [4, 5]]
```

### lax.squeeze

```python
jax.lax.squeeze(array, dimensions)
```

Removes dimensions of size 1 at the specified positions.

```python
x = jnp.ones((1, 3, 1, 4))
result = lax.squeeze(x, dimensions=(0, 2))
# shape (3, 4)
```

### lax.expand_dims

```python
jax.lax.expand_dims(array, dimensions)
```

Adds size-1 dimensions at the specified positions.

```python
x = jnp.ones((3, 4))
result = lax.expand_dims(x, dimensions=(0, 2))
# shape (1, 3, 1, 4)
```

### lax.transpose

```python
jax.lax.transpose(operand, permutation)
```

Transposes the operand according to the given permutation of dimensions.

```python
x = jnp.ones((2, 3, 4))
result = lax.transpose(x, (2, 0, 1))
# shape (4, 2, 3)
```

### lax.rev / lax.flip

```python
jax.lax.rev(operand, dimensions)
jax.lax.flip(operand, dimensions)  # alias for rev
```

Reverses the operand along the specified dimensions.

```python
x = jnp.array([[1, 2, 3],
               [4, 5, 6]])
result = lax.rev(x, dimensions=(1,))
# [[3, 2, 1],
#  [6, 5, 4]]

result = lax.rev(x, dimensions=(0, 1))
# [[6, 5, 4],
#  [3, 2, 1]]
```

---

## Indexing and Slicing Operations

### lax.concatenate

```python
jax.lax.concatenate(operands, dimension)
```

Concatenates a sequence of arrays along the given `dimension`. All operands must have the same shape except along the concatenation dimension.

```python
x = jnp.array([[1, 2], [3, 4]])
y = jnp.array([[5, 6], [7, 8]])

# Concatenate along axis 0
result = lax.concatenate([x, y], dimension=0)
# [[1, 2],
#  [3, 4],
#  [5, 6],
#  [7, 8]]

# Concatenate along axis 1
result = lax.concatenate([x, y], dimension=1)
# [[1, 2, 5, 6],
#  [3, 4, 7, 8]]
```

### lax.split

```python
jax.lax.split(operand, sizes, dimension)
```

Splits `operand` along `dimension` into a list of subarrays. `sizes` is a list of integer sizes for each chunk (must sum to `operand.shape[dimension]`).

```python
x = jnp.arange(10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
parts = lax.split(x, sizes=[3, 3, 4], dimension=0)
# parts[0]: [0, 1, 2]
# parts[1]: [3, 4, 5]
# parts[2]: [6, 7, 8, 9]
```

### lax.slice

```python
jax.lax.slice(operand, start_indices, limit_indices, strides=None)
```

Extracts a subarray using start indices, limit indices, and optional strides. All indices are tuples with the same length as the operand rank.

```python
x = jnp.arange(24).reshape(4, 6)
# [[ 0,  1,  2,  3,  4,  5],
#  [ 6,  7,  8,  9, 10, 11],
#  [12, 13, 14, 15, 16, 17],
#  [18, 19, 20, 21, 22, 23]]

result = lax.slice(x, start_indices=(1, 2), limit_indices=(3, 5))
# [[ 8,  9, 10],
#  [14, 15, 16]]

# With strides
result = lax.slice(x, start_indices=(0, 0), limit_indices=(4, 6), strides=(2, 3))
# [[ 0,  3],
#  [12, 15]]
```

### lax.dynamic_slice

```python
jax.lax.dynamic_slice(operand, start_indices, slice_sizes)
```

Similar to `slice`, but `start_indices` are runtime values (JAX arrays) rather than static Python integers. The slice size is fixed and must be known at compile time.

```python
x = jnp.arange(10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
start = jnp.array([3])
result = lax.dynamic_slice(x, start, slice_sizes=(4,))
# [3, 4, 5, 6]

# start is clamped to valid range
start = jnp.array([8])
result = lax.dynamic_slice(x, start, slice_sizes=(4,))
# [6, 7, 8, 9] (start clamped to 7 so that start + size <= 10)
```

### lax.dynamic_update_slice

```python
jax.lax.dynamic_update_slice(operand, update, start_indices)
```

Returns a copy of `operand` with the `update` array spliced in at the `start_indices`. The shape of `update` determines the size of the updated region.

```python
operand = jnp.arange(10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
update = jnp.array([-1, -2, -3])
start = jnp.array([4])
result = lax.dynamic_update_slice(operand, update, start)
# [0, 1, 2, 3, -1, -2, -3, 7, 8, 9]
```

---

## Gather and Scatter Operations

### lax.gather

```python
jax.lax.gather(
    operand,
    start_indices,
    dimension_numbers,
    slice_sizes,
    *,
    unique_indices=False,
    indices_are_sorted=False,
    mode=None,
)
```

A general indexing operation. Extracts subarrays from `operand` at positions specified by `start_indices`. The `dimension_numbers` parameter (a `GatherDimensionNumbers` named tuple) specifies how the indices map to the operand dimensions.

```python
from jax.lax import GatherDimensionNumbers

# Simple indexing: gather single elements from a 1D array
operand = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
indices = jnp.array([[0, 2], [3, 4]])  # shape (2, 2)

dnums = GatherDimensionNumbers(
    offset_dims=(),            # no extra dims in output beyond index dims
    collapsed_slice_dims=(0,), # collapse the single operand dim
    start_index_map=(0,)       # index maps to operand dim 0
)

result = lax.gather(operand, indices, dimension_numbers=dnums, slice_sizes=(1,))
# [[10.0, 30.0],
#  [40.0, 50.0]]
```

### lax.scatter

```python
jax.lax.scatter(
    operand,
    scatter_indices,
    updates,
    dimension_numbers,
    *,
    unique_indices=False,
    indices_are_sorted=False,
    mode=None,
)
```

The base scatter operation. Writes `updates` into `operand` at positions specified by `scatter_indices`. When multiple updates target the same location, the behavior depends on the scatter variant used.

### lax.scatter_add

```python
jax.lax.scatter_add(
    operand,
    scatter_indices,
    updates,
    dimension_numbers,
    *,
    unique_indices=False,
    indices_are_sorted=False,
    mode=None,
)
```

Scatter with addition: when multiple updates target the same position, they are summed.

```python
from jax.lax import ScatterDimensionNumbers

operand = jnp.zeros((5,))
indices = jnp.array([[1], [2], [2], [4]])
updates = jnp.array([10.0, 20.0, 30.0, 40.0])

dnums = ScatterDimensionNumbers(
    update_window_dims=(),
    inserted_window_dims=(0,),
    scatter_dims_to_operand_dims=(0,)
)

result = lax.scatter_add(operand, indices, updates, dnums)
# [0.0, 10.0, 50.0, 0.0, 40.0]
# Note: index 2 receives both 20.0 and 30.0, summed to 50.0
```

### lax.scatter_mul

```python
jax.lax.scatter_mul(operand, scatter_indices, updates, dimension_numbers, ...)
```

Scatter with multiplication: when multiple updates target the same position, their product is taken.

### lax.scatter_min

```python
jax.lax.scatter_min(operand, scatter_indices, updates, dimension_numbers, ...)
```

Scatter taking the minimum of all updates at each position.

### lax.scatter_max

```python
jax.lax.scatter_max(operand, scatter_indices, updates, dimension_numbers, ...)
```

Scatter taking the maximum of all updates at each position.

```python
operand = jnp.full((5,), -jnp.inf)
indices = jnp.array([[1], [1], [3], [3], [3]])
updates = jnp.array([3.0, 7.0, 2.0, 9.0, 5.0])

dnums = ScatterDimensionNumbers(
    update_window_dims=(),
    inserted_window_dims=(0,),
    scatter_dims_to_operand_dims=(0,)
)

result = lax.scatter_max(operand, indices, updates, dnums)
# [-inf, 7.0, -inf, 9.0, -inf]
```

---

## Control Flow Operations

### lax.select

```python
jax.lax.select(pred, on_true, on_false)
```

Element-wise conditional selection. For each element, returns `on_true[i]` if `pred[i]` is True, otherwise `on_false[i]`. All three arrays broadcast together.

```python
pred = jnp.array([True, False, True])
on_true = jnp.array([1.0, 2.0, 3.0])
on_false = jnp.array([10.0, 20.0, 30.0])
result = lax.select(pred, on_true, on_false)
# [1.0, 20.0, 3.0]
```

### lax.cond

```python
jax.lax.cond(pred, true_fun, false_fun, *operands, linear=None)
```

Conditional execution of one of two functions. Only the selected branch is executed (important for performance and side effects).

```python
def square(x):
    return x ** 2

def cube(x):
    return x ** 3

result = lax.cond(True, square, cube, 4.0)   # 16.0
result = lax.cond(False, square, cube, 4.0)  # 64.0

# Multiple operands
def add_ab(a, b):
    return a + b

def mul_ab(a, b):
    return a * b

result = lax.cond(True, add_ab, mul_ab, 3.0, 5.0)  # 8.0
```

### lax.switch

```python
jax.lax.switch(index, branches, *operands)
```

Selects and executes one of several branches based on `index`. Like a switch/case statement.

```python
def fn0(x): return x + 1
def fn1(x): return x * 2
def fn2(x): return x ** 2

result = lax.switch(0, [fn0, fn1, fn2], 5.0)  # 6.0 (x + 1)
result = lax.switch(1, [fn0, fn1, fn2], 5.0)  # 10.0 (x * 2)
result = lax.switch(2, [fn0, fn1, fn2], 5.0)  # 25.0 (x ** 2)
```

### lax.while_loop

```python
jax.lax.while_loop(cond_fun, body_fun, init_val)
```

Runs a loop: repeatedly applies `body_fun` as long as `cond_fun` returns True. The loop is compiled to a single XLA While operation. All intermediate values must have the same shape/dtype.

```python
# Sum numbers from 1 to n
def cond_fun(state):
    i, total, n = state
    return i < n

def body_fun(state):
    i, total, n = state
    return (i + 1, total + i, n)

init_val = (jnp.int32(0), jnp.int32(0), jnp.int32(10))
_, result, _ = lax.while_loop(cond_fun, body_fun, init_val)
# result = 45 (sum of 0..9)
```

### lax.fori_loop

```python
jax.lax.fori_loop(lower, upper, body_fun, init_val)
```

A counted loop equivalent to `for i in range(lower, upper): init_val = body_fun(i, init_val)`.

```python
# Compute cumulative sum
def body_fn(i, carry):
    return carry + jnp.array([i])

result = lax.fori_loop(0, 5, body_fn, jnp.array([0]))
# [10] (0 + 0 + 1 + 2 + 3 + 4)
```

### lax.scan

```python
jax.lax.scan(f, init, xs, length=None, reverse=False, unroll=1)
```

Scans a function over leading axis of `xs` while carrying state. Returns `(final_carry, stacked_outputs)`. This is the fundamental building block for RNNs, rolling windows, and sequential computation.

```python
# Cumulative sum using scan
def scan_fn(carry, x):
    new_carry = carry + x
    return new_carry, new_carry

xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
final_carry, outputs = lax.scan(scan_fn, 0.0, xs)
# final_carry: 15.0
# outputs: [1.0, 3.0, 6.0, 10.0, 15.0]

# Reverse scan (process from end to start)
final_carry, outputs = lax.scan(scan_fn, 0.0, xs, reverse=True)
# final_carry: 15.0
# outputs: [15.0, 14.0, 12.0, 9.0, 5.0]

# Scan with multiple outputs
def rnn_step(carry, x):
    h_prev = carry
    h_new = jnp.tanh(jnp.dot(x, W_h) + jnp.dot(h_prev, W_hh) + b)
    return h_new, (h_new, x)

final_h, (all_h, all_x) = lax.scan(rnn_step, h0, inputs)
```

### lax.associative_scan

```python
jax.lax.associative_scan(fn, elems, reverse=False)
```

Performs a parallel prefix scan using an associative binary operation. More efficient than sequential `scan` for associative operations (like addition, multiplication, min, max) on hardware with parallelism.

```python
# Parallel cumulative sum
elems = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
result = lax.associative_scan(lax.add, elems)
# [1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0]
```

---

## Linear Algebra Operations

### lax.dot

```python
jax.lax.dot(lhs, rhs)
```

Matrix multiplication / dot product. For 1D inputs, computes vector dot product. For 2D inputs, computes matrix multiplication.

```python
# Vector dot product
a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
result = lax.dot(a, b)  # 32.0

# Matrix multiplication
A = jnp.ones((3, 4))
B = jnp.ones((4, 5))
result = lax.dot(A, B)  # shape (3, 5), all elements = 4.0
```

### lax.dot_general

```python
jax.lax.dot_general(lhs, rhs, dimension_numbers, precision=None,
                     preferred_element_type=None)
```

The most general dot product operation. `dimension_numbers` is a tuple of:
- `((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims))`

This single operation can express dot products, matmuls, batched matmuls, and arbitrary tensor contractions.

```python
# Batched matrix multiplication
# lhs: (batch, m, k), rhs: (batch, k, n)
lhs = jnp.ones((2, 3, 4))
rhs = jnp.ones((2, 4, 5))

dimension_numbers = (
    ((2,), (1,)),  # contracting: lhs dim 2 with rhs dim 1
    ((0,), (0,))   # batch: lhs dim 0 with rhs dim 0
)

result = lax.dot_general(lhs, rhs, dimension_numbers)
# shape (2, 3, 5)

# Einstein summation equivalent: "bik,bkj->bij"
```

### lax.conv_general_dilated

```python
jax.lax.conv_general_dilated(
    lhs, rhs, window_strides, padding,
    lhs_dilation=None, rhs_dilation=None,
    dimension_numbers=None, feature_group_count=1,
    batch_group_count=1, precision=None,
    preferred_element_type=None
)
```

The most general convolution operation. Supports strided, padded, dilated, grouped, and transposed convolutions.

```python
# Standard 2D convolution
# lhs: (N, H, W, C_in) in NHWC format
# rhs: (kH, kW, C_in, C_out) kernel
lhs = jnp.ones((1, 8, 8, 3))   # batch=1, 8x8 image, 3 channels
rhs = jnp.ones((3, 3, 3, 16))  # 3x3 kernel, 3->16 channels

from jax.lax import ConvDimensionNumbers
dn = ConvDimensionNumbers(
    lhs_spec=(0, 1, 2, 3),  # NHWC
    rhs_spec=(3, 0, 1, 2),  # OIHW (out, in, h, w)
    out_spec=(0, 1, 2, 3)   # NHWC
)

result = lax.conv_general_dilated(
    lhs, rhs,
    window_strides=(1, 1),
    padding='SAME',
    dimension_numbers=dn
)
# shape (1, 8, 8, 16)

# Dilated convolution
result = lax.conv_general_dilated(
    lhs, rhs,
    window_strides=(2, 2),
    padding=((0, 0), (0, 0)),
    lhs_dilation=(1, 1),
    rhs_dilation=(2, 2),  # dilation rate 2
    dimension_numbers=dn
)
```

### lax.conv

```python
jax.lax.conv(lhs, rhs, window_strides, padding, precision=None,
             preferred_element_type=None)
```

A simpler convolution interface. Defaults to NCW/NCHW data format.

```python
lhs = jnp.ones((1, 3, 8))   # batch=1, channels=3, width=8
rhs = jnp.ones((16, 3, 3))  # out_channels=16, in_channels=3, kernel_width=3

result = lax.conv(lhs, rhs, window_strides=(1,), padding='SAME')
# shape (1, 16, 8)
```

### lax.conv_transpose

```python
jax.lax.conv_transpose(lhs, rhs, strides, padding, dimension_numbers=None,
                       transpose_kernel=False, precision=None,
                       preferred_element_type=None)
```

Transposed convolution (fractionally-strided convolution), commonly used in decoders and generators.

```python
lhs = jnp.ones((1, 16, 4))   # batch=1, channels=16, width=4
rhs = jnp.ones((16, 3, 3))   # channels=16, kernel_in=3, kernel=3

result = lax.conv_transpose(lhs, rhs, strides=(2,), padding='SAME')
# shape (1, 3, 8) -- upsampled by factor of 2
```

### lax.batch_matmul

```python
jax.lax.batch_matmul(lhs, rhs, precision=None, preferred_element_type=None)
```

Batched matrix multiplication. Contract the last dimension of `lhs` with the second-to-last dimension of `rhs`, with all other dimensions batched.

```python
lhs = jnp.ones((4, 2, 3))  # batch=4, 2x3
rhs = jnp.ones((4, 3, 5))  # batch=4, 3x5

result = lax.batch_matmul(lhs, rhs)
# shape (4, 2, 5)
```

### lax.triangular_solve

```python
jax.lax.triangular_solve(a, b, left_side=True, lower=False,
                          transpose_a=False, conjugate_a=False,
                          unit_diagonal=False)
```

Solves a triangular system of linear equations.

```python
# Solve L @ x = b where L is lower triangular
L = jnp.array([[2.0, 0.0, 0.0],
               [1.0, 3.0, 0.0],
               [4.0, 2.0, 1.0]])
b = jnp.array([4.0, 7.0, 10.0])

x = lax.triangular_solve(L, b, left_side=True, lower=True)
# x such that L @ x = b
```

### lax.cholesky

```python
jax.lax.cholesky(x, upper=False)
```

Computes the Cholesky decomposition of a positive definite matrix.

```python
A = jnp.array([[4.0, 2.0],
               [2.0, 3.0]])
L = lax.cholesky(A, upper=False)
# L @ L.T == A
```

### lax.eig

```python
jax.lax.eig(x, compute_left_eigenvectors=True, compute_right_eigenvectors=True)
```

Computes the eigenvalue decomposition of a general matrix.

```python
A = jnp.array([[1.0, 2.0],
               [3.0, 4.0]])
eigenvalues, left_vectors, right_vectors = lax.eig(A)
```

### lax.eigh

```python
jax.lax.eigh(x, lower=True, sort_eigenvalues=True)
```

Computes the eigenvalue decomposition of a Hermitian (symmetric) matrix. Returns `(eigenvalues, eigenvectors)`.

```python
A = jnp.array([[2.0, 1.0],
               [1.0, 2.0]])
eigenvalues, eigenvectors = lax.eigh(A)
# eigenvalues: [1.0, 3.0]
```

### lax.svd

```python
jax.lax.svd(x, full_matrices=True, compute_uv=True)
```

Computes the singular value decomposition.

```python
A = jnp.array([[1.0, 2.0],
               [3.0, 4.0],
               [5.0, 6.0]])
U, S, Vt = lax.svd(A, full_matrices=False)
# A ~= U @ jnp.diag(S) @ Vt
```

### lax.qr

```python
jax.lax.qr(x, full_matrices=True)
```

Computes the QR decomposition.

```python
A = jnp.array([[1.0, 2.0],
               [3.0, 4.0],
               [5.0, 6.0]])
Q, R = lax.qr(A, full_matrices=False)
# A ~= Q @ R
```

### lax.lu

```python
jax.lax.lu(x)
```

Computes the LU decomposition with partial pivoting. Returns `(lu, pivots, permutation)`.

```python
A = jnp.array([[2.0, 1.0, 1.0],
               [4.0, 3.0, 3.0],
               [8.0, 7.0, 9.0]])
lu, pivots, perm = lax.lu(A)
```

---

## Reduction Operations

### lax.reduce

```python
jax.lax.reduce(operand, init_value, computation, dimensions)
```

The general reduction operation. Reduces `operand` along `dimensions` using the binary `computation` function, starting from `init_value`.

```python
x = jnp.array([[1, 2, 3],
               [4, 5, 6]])

# Sum along axis 0
result = lax.reduce(x, jnp.int32(0), lax.add, dimensions=(0,))
# [5, 7, 9]

# Sum along axis 1
result = lax.reduce(x, jnp.int32(0), lax.add, dimensions=(1,))
# [6, 15]

# Product along axis 1
result = lax.reduce(x, jnp.int32(1), lax.mul, dimensions=(1,))
# [6, 120]
```

### lax.reduce_sum / lax.reduce_prod / lax.reduce_max / lax.reduce_min

```python
jax.lax.reduce_sum(operand, axes)
jax.lax.reduce_prod(operand, axes)
jax.lax.reduce_max(operand, axes)
jax.lax.reduce_min(operand, axes)
```

Convenience wrappers around `reduce` for common operations.

```python
x = jnp.array([[1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0]])

lax.reduce_sum(x, axes=(0,))    # [5.0, 7.0, 9.0]
lax.reduce_sum(x, axes=(1,))    # [6.0, 15.0]
lax.reduce_sum(x, axes=(0, 1))  # 21.0

lax.reduce_max(x, axes=(1,))    # [3.0, 6.0]
lax.reduce_min(x, axes=(0,))    # [1.0, 2.0, 3.0]
lax.reduce_prod(x, axes=(1,))   # [6.0, 120.0]
```

### lax.reduce_or / lax.reduce_and

```python
jax.lax.reduce_or(operand, axes)
jax.lax.reduce_and(operand, axes)
```

Boolean reduction: logical OR / AND over the specified axes.

```python
x = jnp.array([[True, False, True],
               [False, False, True]])

lax.reduce_or(x, axes=(0,))    # [True, False, True]
lax.reduce_and(x, axes=(1,))   # [False, False]
```

### lax.reduce_window

```python
jax.lax.reduce_window(
    operand, init_value, computation,
    window_dimensions, window_strides, padding,
    base_dilation=None, window_dilation=None
)
```

Reduces over a sliding window. This is the building block for pooling operations.

```python
x = jnp.array([[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12]])

# 2x2 max pooling with stride 2
result = lax.reduce_window(
    x, -jnp.inf, lax.max,
    window_dimensions=(2, 2),
    window_strides=(2, 2),
    padding='VALID'
)
# [[ 6,  8],
#  [10, 12]]

# 2x2 average pooling
result = lax.reduce_window(
    x, 0.0, lax.add,
    window_dimensions=(2, 2),
    window_strides=(2, 2),
    padding='VALID'
) / 4.0
# [[ 3.5,  5.5],
#  [ 7.5,  9.5]]
```

### lax.cumsum / lax.cumprod / lax.cummax / lax.cummin

```python
jax.lax.cumsum(operand, axis=0, reverse=False)
jax.lax.cumprod(operand, axis=0, reverse=False)
jax.lax.cummax(operand, axis=0, reverse=False)
jax.lax.cummin(operand, axis=0, reverse=False)
```

Cumulative operations along an axis.

```python
x = jnp.array([1, 2, 3, 4, 5])

lax.cumsum(x)     # [1, 3, 6, 10, 15]
lax.cumprod(x)    # [1, 2, 6, 24, 120]
lax.cummax(x)     # [1, 2, 3, 4, 5]
lax.cummin(x)     # [1, 1, 1, 1, 1]

# Reverse
lax.cumsum(x, reverse=True)  # [15, 14, 12, 9, 5]
```

### lax.cumulative_sum / lax.cumulative_prod / lax.cumulative_max / lax.cumulative_min

```python
jax.lax.cumulative_sum(operand, axis=0, reverse=False)
jax.lax.cumulative_prod(operand, axis=0, reverse=False)
jax.lax.cumulative_max(operand, axis=0, reverse=False)
jax.lax.cumulative_min(operand, axis=0, reverse=False)
```

Aliases for the `cum*` operations above.

---

## Bitwise Operations

### lax.bitwise_and / lax.bitwise_or / lax.bitwise_xor

```python
jax.lax.bitwise_and(x, y)
jax.lax.bitwise_or(x, y)
jax.lax.bitwise_xor(x, y)
```

Element-wise bitwise operations on integer arrays.

```python
x = jnp.array([0b1100, 0b1010, 0b1111], dtype=jnp.int32)
y = jnp.array([0b1010, 0b1100, 0b0000], dtype=jnp.int32)

lax.bitwise_and(x, y)  # [0b1000, 0b1000, 0b0000] = [8, 8, 0]
lax.bitwise_or(x, y)   # [0b1110, 0b1110, 0b1111] = [14, 14, 15]
lax.bitwise_xor(x, y)  # [0b0110, 0b0110, 0b1111] = [6, 6, 15]
```

### lax.bitwise_not

```python
jax.lax.bitwise_not(x)
```

Element-wise bitwise complement.

```python
x = jnp.array([0, 1, 255], dtype=jnp.uint8)
result = lax.bitwise_not(x)
# [255, 254, 0]
```

### lax.population_count

```python
jax.lax.population_count(x)
```

Element-wise count of set bits (population count / popcount).

```python
x = jnp.array([0, 1, 3, 7, 15, 255], dtype=jnp.uint8)
result = lax.population_count(x)
# [0, 1, 2, 3, 4, 8]
```

### lax.clz

```python
jax.lax.clz(x)
```

Element-wise count leading zeros.

```python
x = jnp.array([1, 2, 4, 8, 16, 0], dtype=jnp.uint32)
result = lax.clz(x)
# [31, 30, 29, 28, 27, 32]
```

---

## Type Conversion Operations

### lax.convert_element_type

```python
jax.lax.convert_element_type(operand, new_dtype)
```

Converts the element type of `operand` to `new_dtype`.

```python
x = jnp.array([1.5, 2.7, 3.9])
result = lax.convert_element_type(x, jnp.int32)
# [1, 2, 3] (truncates toward zero)

x = jnp.array([1, 2, 3])
result = lax.convert_element_type(x, jnp.float32)
# [1.0, 2.0, 3.0]
```

### lax.bitcast_convert_type

```python
jax.lax.bitcast_convert_type(operand, new_dtype)
```

Bitcast converts the element type without changing the underlying bits. The old and new dtypes must have the same bit width.

```python
x = jnp.float32(1.0)
result = lax.bitcast_convert_type(x, jnp.int32)
# 1065353216 (IEEE 754 bit pattern for 1.0f)

x = jnp.int32(0x40490FDB)
result = lax.bitcast_convert_type(x, jnp.float32)
# 3.14159274... (IEEE 754 bit pattern for pi)
```

---

## Special Mathematical Functions

### lax.igamma / lax.igammac

```python
jax.lax.igamma(a, x)    # Regularized lower incomplete gamma function P(a, x)
jax.lax.igammac(a, x)   # Regularized upper incomplete gamma function Q(a, x) = 1 - P(a, x)
```

```python
a = jnp.array([1.0, 2.0, 3.0])
x = jnp.array([1.0, 1.0, 1.0])

lax.igamma(a, x)   # [0.6321, 0.2642, 0.0803]
lax.igammac(a, x)  # [0.3679, 0.7358, 0.9197]
```

### lax.betainc

```python
jax.lax.betainc(a, b, x)
```

Regularized incomplete beta function.

```python
a = jnp.array([1.0, 2.0])
b = jnp.array([2.0, 1.0])
x = jnp.array([0.5, 0.5])

result = lax.betainc(a, b, x)
# [0.75, 0.75]
```

### lax.digamma / lax.lgamma

```python
jax.lax.digamma(x)   # Psi(x) = d/dx ln(Gamma(x))
jax.lax.lgamma(x)    # ln(|Gamma(x)|)
```

```python
x = jnp.array([1.0, 2.0, 3.0, 4.0])

lax.digamma(x)   # [-0.5772, 0.4228, 0.9228, 1.2561]
lax.lgamma(x)    # [0.0, 0.0, 0.6931, 1.7918]
```

### lax.erf / lax.erfc

```python
jax.lax.erf(x)    # Error function
jax.lax.erfc(x)   # Complementary error function = 1 - erf(x)
```

```python
x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

lax.erf(x)   # [-0.9953, -0.8427, 0.0, 0.8427, 0.9953]
lax.erfc(x)  # [1.9953, 1.8427, 1.0, 0.1573, 0.0047]
```

### lax.bessel_i0e / lax.bessel_i1e

```python
jax.lax.bessel_i0e(x)   # Exponentially scaled modified Bessel function of order 0: exp(-|x|) * I0(x)
jax.lax.bessel_i1e(x)   # Exponentially scaled modified Bessel function of order 1: exp(-|x|) * I1(x)
```

```python
x = jnp.array([0.0, 1.0, 2.0, 5.0])

lax.bessel_i0e(x)
# [1.0, 0.4658, 0.3085, 0.1835]

lax.bessel_i1e(x)
# [0.0, 0.2079, 0.2153, 0.1631]
```

### lax.regularized_incomplete_gamma_p / lax.regularized_incomplete_gamma_q

Lower-level aliases for `igamma` and `igammac` respectively.

---

## FFT Operations

### lax.fft / lax.ifft

```python
jax.lax.fft(x, fft_type, fft_lengths)
```

General FFT operation. `fft_type` is one of `FFTType.FFT`, `FFTType.IFFT`, `FFTType.RFFT`, `FFTType.IRFFT`.

```python
from jax.lax import FFTType

x = jnp.array([1.0, 2.0, 3.0, 4.0])
result = lax.fft(x, FFTType.FFT, fft_lengths=(4,))
# [10+0j, -2+2j, -2+0j, -2-2j]

# Inverse FFT
result = lax.fft(result, FFTType.IFFT, fft_lengths=(4,))
# [1+0j, 2+0j, 3+0j, 4+0j]

# Real FFT (real input -> complex output)
x = jnp.array([1.0, 2.0, 3.0, 4.0])
result = lax.fft(x, FFTType.RFFT, fft_lengths=(4,))
# [10+0j, -2+2j, -2+0j]
```

---

## Random Number Generation

### lax.rng_bit_generator

```python
jax.lax.rng_bit_generator(key, shape, dtype=jnp.uint32)
```

Generates random bits using the specified PRNG key. Returns `(new_key, bits_array)`.

```python
key = jnp.array([0, 0], dtype=jnp.uint32)
new_key, bits = lax.rng_bit_generator(key, shape=(5,), dtype=jnp.uint32)
```

### lax.rng_uniform

```python
jax.lax.rng_uniform(a, b, shape)
```

Generates uniform random values in `[a, b)`. Note: This uses the global PRNG state, not JAX's explicit PRNG system.

```python
result = lax.rng_uniform(0.0, 1.0, shape=(3,))
# Array of 3 uniform random floats in [0, 1)
```

---

## Other Operations

### lax.pad

```python
jax.lax.pad(operand, padding_value, padding_config)
```

Adds padding to an array. `padding_config` is a sequence of `(low, high, interior)` tuples for each dimension, specifying the number of elements to pad on each side and between elements.

```python
x = jnp.array([1, 2, 3])

# Pad with zeros: 2 on left, 1 on right, 0 interior
result = lax.pad(x, 0, [(2, 1, 0)])
# [0, 0, 1, 2, 3, 0]

# Pad with interior spacing
result = lax.pad(x, 0, [(0, 0, 1)])
# [1, 0, 2, 0, 3]

# 2D padding
x = jnp.array([[1, 2], [3, 4]])
result = lax.pad(x, 0, [(1, 1, 0), (1, 1, 0)])
# [[0, 0, 0, 0],
#  [0, 1, 2, 0],
#  [0, 3, 4, 0],
#  [0, 0, 0, 0]]
```

### lax.real / lax.imag / lax.complex

```python
jax.lax.real(x)        # Extract real part of complex array
jax.lax.imag(x)        # Extract imaginary part of complex array
jax.lax.complex(x, y)  # Construct complex array from real and imaginary parts
```

```python
z = jnp.array([1.0 + 2.0j, 3.0 + 4.0j])

lax.real(z)  # [1.0, 3.0]
lax.imag(z)  # [2.0, 4.0]

lax.complex(jnp.array([1.0, 3.0]), jnp.array([2.0, 4.0]))
# [1.0+2.0j, 3.0+4.0j]
```

### lax.conj

```python
jax.lax.conj(x)
```

Element-wise complex conjugate.

```python
z = jnp.array([1.0 + 2.0j, 3.0 + 4.0j])
result = lax.conj(z)
# [1.0-2.0j, 3.0-4.0j]
```

### lax.abs

```python
jax.lax.abs(x)
```

Element-wise absolute value. For complex inputs, returns the magnitude `sqrt(real^2 + imag^2)`.

### lax.select_n

```python
jax.lax.select_n(which, *choices)
```

Selects from a list of arrays based on an integer index. Unlike `select` which uses boolean predicates, `select_n` uses an integer index.

```python
which = jnp.array([0, 1, 2, 0])
a = jnp.array([10, 10, 10, 10])
b = jnp.array([20, 20, 20, 20])
c = jnp.array([30, 30, 30, 30])

result = lax.select_n(which, a, b, c)
# [10, 20, 30, 10]
```

### lax.stop_gradient

```python
jax.lax.stop_gradient(x)
```

Prevents gradients from flowing through `x`. The identity function in the forward pass, but zero in the backward pass.

```python
def loss_fn(x):
    # Use x for computation but don't propagate gradients through it
    return jnp.sum(x * lax.stop_gradient(x))

grad_fn = jax.grad(loss_fn)
result = grad_fn(jnp.array([1.0, 2.0, 3.0]))
# [1.0, 2.0, 3.0] -- gradient of sum(x * sg(x)) w.r.t. x is sg(x) = x
```

### lax.custom_linear_solve

```python
jax.lax.custom_linear_solve(matvec, b, solve, transpose_solve=None, symmetric=False)
```

Provides a custom, differentiable linear solve. You specify the forward solve and optionally the transpose solve, and JAX automatically derives the backward pass.

```python
def matvec(x):
    return A @ x

def solve(matvec_fn, b):
    return jnp.linalg.solve(A, b)

x = lax.custom_linear_solve(matvec, b, solve)
```

### lax.custom_root

```python
jax.lax.custom_root(f, initial_guess, solve, tangent_solve=None, has_aux=False)
```

Provides a custom, differentiable root-finding operation.

### lax.custom_jvp

Decorator for defining custom JVP (Jacobian-vector product) rules. See `jax.custom_jvp` for the higher-level API.

### lax.custom_vjp

Decorator for defining custom VJP (vector-Jacobian product) rules. See `jax.custom_vjp` for the higher-level API.

---

## Quick Reference: Dimension Number Types

`jax.lax` uses several named tuple types for specifying dimension mappings:

### ConvDimensionNumbers

```python
from jax.lax import ConvDimensionNumbers

dn = ConvDimensionNumbers(
    lhs_spec=(0, 1, 2, 3),   # Input layout (e.g., NCHW)
    rhs_spec=(0, 1, 2, 3),   # Kernel layout (e.g., OIHW)
    out_spec=(0, 1, 2, 3)    # Output layout (e.g., NCHW)
)
```

### DotDimensionNumbers

```python
from jax.lax import DotDimensionNumbers

dn = DotDimensionNumbers(
    lhs_contracting_dims=(1,),
    rhs_contracting_dims=(0,),
    lhs_batch_dims=(),
    rhs_batch_dims=()
)
```

### GatherDimensionNumbers

```python
from jax.lax import GatherDimensionNumbers

dn = GatherDimensionNumbers(
    offset_dims=(),              # Output dims not from index mapping
    collapsed_slice_dims=(0,),   # Operand dims collapsed in output
    start_index_map=(0,)         # How index dims map to operand dims
)
```

### ScatterDimensionNumbers

```python
from jax.lax import ScatterDimensionNumbers

dn = ScatterDimensionNumbers(
    update_window_dims=(),           # Update dims appearing in output
    inserted_window_dims=(0,),       # Operand dims inserted in output
    scatter_dims_to_operand_dims=(0,)  # How scatter index maps to operand
)
```

---

## Quick Reference: RoundingMethod

```python
from jax.lax import RoundingMethod

RoundingMethod.AWAY_FROM_ZERO   # 0.5 rounds away from zero
RoundingMethod.TO_NEAREST_EVEN  # 0.5 rounds to nearest even integer
```

---

## Quick Reference: FFTType

```python
from jax.lax import FFTType

FFTType.FFT     # Complex-to-complex FFT
FFTType.IFFT    # Complex-to-complex inverse FFT
FFTType.RFFT    # Real-to-complex FFT
FFTType.IRFFT   # Complex-to-real inverse FFT
```
