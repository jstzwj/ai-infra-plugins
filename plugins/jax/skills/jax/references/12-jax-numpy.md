# jax.numpy (jnp) — Comprehensive API Reference

`jax.numpy` (conventionally imported as `jnp`) is a high-level NumPy-compatible API backed by JAX's XLA compiler. It provides near-complete coverage of NumPy's functionality with key differences:

- **Immutable arrays:** JAX arrays are immutable; in-place operations return new arrays.
- **Functional design:** All operations are pure functions with no side effects.
- **Device execution:** Operations run on accelerators (GPU/TPU) automatically.
- **JIT compilable:** All operations work inside `jax.jit`.
- **32-bit default:** JAX defaults to 32-bit dtypes (`float32`, `int32`) unlike NumPy's 64-bit defaults.

```python
import jax
import jax.numpy as jnp

# Enable 64-bit (optional, disabled by default)
jax.config.update("jax_enable_x64", True)
```

---

## Table of Contents

1. [Array Creation](#array-creation)
2. [Array Manipulation](#array-manipulation)
3. [Mathematical Functions](#mathematical-functions)
4. [Linear Algebra (jnp.linalg)](#linear-algebra-jnplinalg)
5. [Statistics](#statistics)
6. [FFT (jnp.fft)](#fft-jnpfft)
7. [Searching and Sorting](#searching-and-sorting)
8. [Logic and Comparison](#logic-and-comparison)
9. [Type Operations](#type-operations)
10. [Differences from NumPy](#differences-from-numpy)

---

## Array Creation

### jnp.array

```python
jnp.array(object, dtype=None, copy=True, order='K', ndmin=0)
```

Creates a JAX array from a Python object (list, tuple, NumPy array, scalar).

```python
# From list
a = jnp.array([1, 2, 3, 4])
# [1 2 3 4], dtype=int32

# With explicit dtype
a = jnp.array([1, 2, 3], dtype=jnp.float32)
# [1.0 2.0 3.0], dtype=float32

# From nested list (2D)
a = jnp.array([[1, 2], [3, 4]])
# [[1 2]
#  [3 4]]

# From NumPy array
import numpy as np
np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
jnp_arr = jnp.array(np_arr)  # Will be float32 by default unless x64 enabled
```

### jnp.zeros / jnp.ones / jnp.empty

```python
jnp.zeros(shape, dtype=None)
jnp.ones(shape, dtype=None)
jnp.empty(shape, dtype=None)
```

Create arrays filled with zeros, ones, or uninitialized values.

```python
jnp.zeros((3, 4))          # 3x4 array of 0.0
jnp.ones((2, 3), dtype=int) # 2x3 array of 1
jnp.empty((5,))             # Uninitialized (may contain arbitrary values)
```

**Note:** `jnp.empty` does not guarantee zero-initialization. Unlike NumPy where `empty` is faster than `zeros`, in JAX both are equally fast because XLA initializes all memory.

### jnp.full

```python
jnp.full(shape, fill_value, dtype=None)
```

Creates an array filled with a constant value.

```python
jnp.full((2, 3), 7.0)
# [[7.0 7.0 7.0]
#  [7.0 7.0 7.0]]

jnp.full((3,), fill_value=jnp.pi)
# [3.1415927 3.1415927 3.1415927]
```

### jnp.arange

```python
jnp.arange(start=None, stop=None, step=1, dtype=None)
```

Creates an array with evenly spaced values within a given range.

```python
jnp.arange(5)        # [0 1 2 3 4]
jnp.arange(1, 10, 2) # [1 3 5 7 9]
jnp.arange(0, 1, 0.1) # [0.0 0.1 0.2 ... 0.9]

# JAX difference: arange requires known static bounds inside jit.
# Use jnp.linspace for dynamic ranges.
```

### jnp.linspace / jnp.logspace / jnp.geomspace

```python
jnp.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
jnp.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0)
jnp.geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0)
```

```python
jnp.linspace(0, 1, 5)      # [0.0, 0.25, 0.5, 0.75, 1.0]
jnp.linspace(0, 1, 5, endpoint=False)  # [0.0, 0.2, 0.4, 0.6, 0.8]

jnp.logspace(0, 2, 5)      # [1.0, 3.162, 10.0, 31.623, 100.0]
jnp.logspace(0, 2, 5, base=2)  # [1.0, 1.682, 2.828, 4.757, 8.0]

jnp.geomspace(1, 1000, 5)  # [1.0, 5.623, 31.623, 177.828, 1000.0]
```

### jnp.eye / jnp.identity

```python
jnp.eye(N, M=None, k=0, dtype=None)
jnp.identity(n, dtype=None)
```

Creates an identity matrix or a matrix with ones on a diagonal.

```python
jnp.eye(3)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

jnp.eye(3, 4, k=1)
# [[0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]

jnp.identity(4)  # Equivalent to eye(4)
```

### jnp.diag / jnp.diagflat

```python
jnp.diag(v, k=0)
jnp.diagflat(v, k=0)
```

Extract a diagonal or construct a diagonal matrix.

```python
# Construct diagonal matrix
jnp.diag(jnp.array([1, 2, 3]))
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]

# Extract diagonal
a = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
jnp.diag(a)  # [1, 5, 9]

# Offset diagonal
jnp.diag(jnp.array([1, 2]), k=1)
# [[0 1 0]
#  [0 0 2]
#  [0 0 0]]

# diagflat: always constructs a diagonal matrix
jnp.diagflat(jnp.array([1, 2, 3]))
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]
```

### jnp.tri / jnp.tril / jnp.triu

```python
jnp.tri(N, M=None, k=0, dtype=None)
jnp.tril(m, k=0)
jnp.triu(m, k=0)
```

```python
jnp.tri(3)
# [[1. 0. 0.]
#  [1. 1. 0.]
#  [1. 1. 1.]]

a = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
jnp.tril(a)
# [[1 0 0]
#  [4 5 0]
#  [7 8 9]]

jnp.triu(a, k=1)
# [[0 2 3]
#  [0 0 6]
#  [0 0 0]]
```

### jnp.zeros_like / jnp.ones_like / jnp.empty_like / jnp.full_like

```python
jnp.zeros_like(a, dtype=None, shape=None)
jnp.ones_like(a, dtype=None, shape=None)
jnp.empty_like(a, dtype=None, shape=None)
jnp.full_like(a, fill_value, dtype=None, shape=None)
```

Create new arrays with the same shape and dtype as the input.

```python
x = jnp.array([[1, 2, 3], [4, 5, 6]])
jnp.zeros_like(x)
# [[0 0 0]
#  [0 0 0]]

jnp.ones_like(x, dtype=jnp.float32)
# [[1.0 1.0 1.0]
#  [1.0 1.0 1.0]]

jnp.full_like(x, fill_value=7)
# [[7 7 7]
#  [7 7 7]]

# Override shape
jnp.zeros_like(x, shape=(3,))
# [0 0 0]
```

### jnp.asarray / jnp.copy / jnp.from_numpy / jnp.from_dlpack

```python
jnp.asarray(a, dtype=None)
jnp.copy(a, order='K')
jnp.from_numpy(ndarray)
jnp.from_dlpack(x)
```

```python
# asarray: avoids copying if already a JAX array with correct dtype
a = jnp.array([1, 2, 3])
b = jnp.asarray(a)  # No copy, same object
c = jnp.asarray(a, dtype=jnp.float32)  # New array (dtype changed)

# copy: always creates a new copy
d = jnp.copy(a)

# from_numpy: converts NumPy array to JAX array
np_array = np.array([1.0, 2.0])
jax_array = jnp.from_numpy(np_array)

# from_dlpack: imports from any DLpack-compatible framework
# jnp.from_dlpack(torch_tensor)  # If PyTorch tensor supports __dlpack__
```

---

## Array Manipulation

### jnp.reshape

```python
jnp.reshape(a, newshape, order='C')
```

```python
a = jnp.arange(12)
jnp.reshape(a, (3, 4))
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Use -1 to infer one dimension
jnp.reshape(a, (3, -1))  # Same result

jnp.reshape(a, (2, 2, 3))
# [[[ 0  1  2]
#   [ 3  4  5]]
#  [[ 6  7  8]
#   [ 9 10 11]]]
```

### jnp.ravel / jnp.flatten

```python
jnp.ravel(a, order='C')
a.flatten(order='C')  # Identical to ravel for JAX arrays
```

```python
a = jnp.array([[1, 2, 3], [4, 5, 6]])
jnp.ravel(a)  # [1 2 3 4 5 6]
```

### jnp.squeeze / jnp.expand_dims

```python
jnp.squeeze(a, axis=None)
jnp.expand_dims(a, axis)
```

```python
a = jnp.ones((1, 3, 1, 4))
jnp.squeeze(a).shape           # (3, 4)
jnp.squeeze(a, axis=2).shape   # (1, 3, 4)
jnp.squeeze(a, axis=(0, 2)).shape  # (3, 4)

b = jnp.ones((3, 4))
jnp.expand_dims(b, axis=0).shape   # (1, 3, 4)
jnp.expand_dims(b, axis=2).shape   # (3, 4, 1)
jnp.expand_dims(b, axis=(0, 2)).shape  # (1, 3, 1, 4)
```

### jnp.moveaxis / jnp.swapaxes / jnp.transpose

```python
jnp.moveaxis(a, source, destination)
jnp.swapaxes(a, axis1, axis2)
jnp.transpose(a, axes=None)
```

```python
a = jnp.ones((2, 3, 4))

jnp.transpose(a).shape            # (4, 3, 2)
jnp.transpose(a, (2, 0, 1)).shape # (4, 2, 3)

jnp.swapaxes(a, 0, 2).shape      # (4, 3, 2)

jnp.moveaxis(a, 0, -1).shape     # (3, 4, 2)
jnp.moveaxis(a, [0, 1], [2, 0]).shape  # (4, 2, 3)
```

### jnp.concatenate / jnp.stack / jnp.vstack / jnp.hstack / jnp.dstack / jnp.column_stack / jnp.row_stack

```python
jnp.concatenate(arrays, axis=0, dtype=None)
jnp.stack(arrays, axis=0, dtype=None)
jnp.vstack(tup)     # Vertical stack (along axis 0)
jnp.hstack(tup)     # Horizontal stack (along axis 1 for 2D+)
jnp.dstack(tup)     # Depth stack (along axis 2)
jnp.column_stack(tup) # Stack 1D arrays as columns
jnp.row_stack(tup)    # Alias for vstack
```

```python
a = jnp.array([[1, 2], [3, 4]])
b = jnp.array([[5, 6], [7, 8]])

jnp.concatenate([a, b], axis=0)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

jnp.concatenate([a, b], axis=1)
# [[1 2 5 6]
#  [3 4 7 8]]

jnp.stack([a, b], axis=0)
# [[[1 2]
#   [3 4]]
#  [[5 6]
#   [7 8]]]
# shape (2, 2, 2)

jnp.vstack([jnp.array([1, 2, 3]), jnp.array([4, 5, 6])])
# [[1 2 3]
#  [4 5 6]]

jnp.hstack([jnp.array([1, 2]), jnp.array([3, 4])])
# [1 2 3 4]

jnp.column_stack([jnp.array([1, 2]), jnp.array([3, 4])])
# [[1 3]
#  [2 4]]
```

### jnp.split / jnp.vsplit / jnp.hsplit / jnp.dsplit / jnp.array_split

```python
jnp.split(ary, indices_or_sections, axis=0)
jnp.array_split(ary, indices_or_sections, axis=0)
jnp.vsplit(ary, indices_or_sections)  # Split along axis 0
jnp.hsplit(ary, indices_or_sections)  # Split along axis 1
jnp.dsplit(ary, indices_or_sections)  # Split along axis 2
```

```python
a = jnp.arange(10)
jnp.split(a, 2)
# [Array([0, 1, 2, 3, 4]), Array([5, 6, 7, 8, 9])]

jnp.split(a, [3, 5, 7])
# [Array([0, 1, 2]), Array([3, 4]), Array([5, 6]), Array([7, 8, 9])]

# array_split allows unequal splits
a = jnp.arange(7)
jnp.array_split(a, 3)
# [Array([0, 1, 2]), Array([3, 4]), Array([5, 6])]
```

### jnp.tile / jnp.repeat

```python
jnp.tile(A, reps)
jnp.repeat(a, repeats, axis=None, *, total_repeat_length=None)
```

```python
a = jnp.array([1, 2, 3])
jnp.tile(a, 3)          # [1 2 3 1 2 3 1 2 3]
jnp.tile(a, (2, 2))
# [[1 2 3 1 2 3]
#  [1 2 3 1 2 3]]

jnp.repeat(a, 2)        # [1 1 2 2 3 3]
jnp.repeat(a, [1, 2, 3]) # [1 2 2 3 3 3]

# 2D repeat
b = jnp.array([[1, 2], [3, 4]])
jnp.repeat(b, 2, axis=0)
# [[1 2]
#  [1 2]
#  [3 4]
#  [3 4]]
```

### jnp.pad

```python
jnp.pad(array, pad_width, mode='constant', **kwargs)
```

```python
a = jnp.array([1, 2, 3])

# Constant padding (default)
jnp.pad(a, 2)
# [0 0 1 2 3 0 0]

jnp.pad(a, (1, 2), constant_values=99)
# [99 1 2 3 99 99]

# Edge padding
jnp.pad(a, 2, mode='edge')
# [1 1 1 2 3 3 3]

# Reflect padding
jnp.pad(a, 2, mode='reflect')
# [3 2 1 2 3 2 1]

# Symmetric padding
jnp.pad(a, 2, mode='symmetric')
# [2 1 1 2 3 3 2]

# Wrap padding
jnp.pad(a, 2, mode='wrap')
# [2 3 1 2 3 1 2]

# 2D padding
b = jnp.ones((2, 2))
jnp.pad(b, ((1, 1), (2, 2)))
# [[0 0 0 0 0 0]
#  [0 0 1 1 0 0]
#  [0 0 1 1 0 0]
#  [0 0 0 0 0 0]]
```

### jnp.flip / jnp.fliplr / jnp.flipud

```python
jnp.flip(m, axis=None)
jnp.fliplr(m)   # Flip left-right (axis=1)
jnp.flipud(m)   # Flip up-down (axis=0)
```

```python
a = jnp.array([[1, 2, 3], [4, 5, 6]])

jnp.flip(a)
# [[6 5 4]
#  [3 2 1]]

jnp.fliplr(a)
# [[3 2 1]
#  [6 5 4]]

jnp.flipud(a)
# [[4 5 6]
#  [1 2 3]]
```

### jnp.roll / jnp.rot90

```python
jnp.roll(a, shift, axis=None)
jnp.rot90(m, k=1, axes=(0, 1))
```

```python
a = jnp.array([1, 2, 3, 4, 5])
jnp.roll(a, 2)       # [4 5 1 2 3]
jnp.roll(a, -2)      # [3 4 5 1 2]

b = jnp.array([[1, 2], [3, 4]])
jnp.roll(b, 1, axis=0)  # [[3 4], [1 2]]

jnp.rot90(b)
# [[2 4]
#  [1 3]]
```

### jnp.delete / jnp.insert / jnp.append

```python
jnp.delete(arr, obj, axis=None)
jnp.insert(arr, obj, values, axis=None)
jnp.append(arr, values, axis=None)
```

```python
a = jnp.array([1, 2, 3, 4, 5])
jnp.delete(a, [1, 3])    # [1 3 5]
jnp.insert(a, 2, 99)     # [1 2 99 3 4 5]
jnp.append(a, [6, 7])    # [1 2 3 4 5 6 7]
```

**Note:** These operations are not efficient in JAX because they require copying the entire array. Avoid using them in tight loops.

### jnp.unique / jnp.sort / jnp.argsort / jnp.lexsort / jnp.searchsorted

```python
jnp.unique(ar, return_index=False, return_inverse=False, return_counts=False,
           axis=None, size=None, fill_value=None, equal_nan=True)
jnp.sort(a, axis=-1, kind='quicksort', order=None, stable=False, descending=False)
jnp.argsort(a, axis=-1, kind='quicksort', order=None, stable=False, descending=False)
jnp.lexsort(keys, axis=-1)
jnp.searchsorted(a, v, side='left', sorter=None, method='scan')
```

```python
a = jnp.array([3, 1, 4, 1, 5, 9, 2, 6])

jnp.sort(a)         # [1 1 2 3 4 5 6 9]
jnp.argsort(a)      # [1 3 6 0 2 4 7 5]  (indices that would sort)
jnp.unique(a)       # [1 2 3 4 5 6 9]

vals, idx, inv, counts = jnp.unique(a, return_index=True,
                                      return_inverse=True,
                                      return_counts=True)
# vals: [1 2 3 4 5 6 9]
# idx:  [1 6 0 2 4 7 5]  (first occurrence indices)
# inv:  [2 0 3 0 4 6 1 5] (indices to reconstruct original from unique)
# counts: [2 1 1 1 1 1 1]

# Sorted array for searchsorted
sorted_a = jnp.sort(a)
jnp.searchsorted(sorted_a, jnp.array([3, 5, 7]))
# [3 4 6]  (insert positions)

# Lexsort: sort by multiple keys
names = jnp.array(['bob', 'amy', 'bob', 'amy'])
scores = jnp.array([90, 85, 80, 92])
idx = jnp.lexsort((scores, names))  # Sort by name, then by score
```

**JAX difference for `unique`:** The `size` and `fill_value` parameters are JAX-specific. Since JAX requires known output shapes at compile time, you must specify `size` (or the number of unique elements must be statically known) when using `unique` inside `jax.jit`.

### jnp.where / jnp.argmin / jnp.argmax / jnp.nonzero

```python
jnp.where(condition, x=None, y=None)
jnp.argmin(a, axis=None, keepdims=False)
jnp.argmax(a, axis=None, keepdims=False)
jnp.nonzero(a, *, size=None, fill_value=None)
```

```python
# Three-argument where: element-wise conditional
a = jnp.array([1, 2, 3, 4, 5])
jnp.where(a > 3, a, -a)
# [-1 -2 -3  4  5]

# One-argument where: indices where condition is True
jnp.where(a > 3)
# (Array([3, 4]),)

jnp.argmin(a)   # 0
jnp.argmax(a)   # 4

b = jnp.array([[1, 0, 3], [0, 5, 0]])
jnp.nonzero(b)
# (Array([0, 0, 1]), Array([0, 2, 1]))

jnp.argmax(b, axis=0)  # [0, 1, 0]
jnp.argmax(b, axis=1)  # [2, 1]
```

**JAX difference for `nonzero`:** Requires `size` parameter inside `jax.jit` for output shape stability.

---

## Mathematical Functions

### Basic Arithmetic

```python
jnp.add(x1, x2, /)
jnp.subtract(x1, x2, /)
jnp.multiply(x1, x2, /)
jnp.divide(x1, x2, /)
jnp.floor_divide(x1, x2, /)
jnp.power(x1, x2, /)
jnp.remainder(x1, x2, /)
jnp.mod(x1, x2, /)           # Alias for remainder
jnp.fmod(x1, x2, /)          # C-style remainder (sign of dividend)
jnp.divmod(x1, x2, /)        # Returns (floor_divide, remainder) tuple
jnp.negative(x, /)
jnp.positive(x, /)
jnp.reciprocal(x)
jnp.square(x)
jnp.cbrt(x)
jnp.sqrt(x)
jnp.hypot(x1, x2, /)
```

```python
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])

jnp.add(x, y)         # [5.0, 7.0, 9.0]
jnp.subtract(x, y)    # [-3.0, -3.0, -3.0]
jnp.multiply(x, y)    # [4.0, 10.0, 18.0]
jnp.divide(x, y)      # [0.25, 0.4, 0.5]
jnp.floor_divide(x, y) # [0.0, 0.0, 0.0]
jnp.power(x, 3)       # [1.0, 8.0, 27.0]
jnp.remainder(y, x)   # [0.0, 1.0, 0.0]
jnp.negative(x)       # [-1.0, -2.0, -3.0]
jnp.reciprocal(x)     # [1.0, 0.5, 0.333...]
jnp.square(x)         # [1.0, 4.0, 9.0]
jnp.sqrt(x)           # [1.0, 1.414..., 1.732...]
jnp.cbrt(jnp.array([1., 8., 27.]))  # [1.0, 2.0, 3.0]
jnp.hypot(3.0, 4.0)   # 5.0
```

### Absolute Value and Sign

```python
jnp.absolute(x, /)     # |x|
jnp.fabs(x, /)         # Float absolute value
jnp.sign(x, /)
jnp.rint(x, /)         # Round to nearest integer
jnp.conj(x, /)
jnp.conjugate(x, /)    # Alias for conj
```

```python
jnp.absolute(jnp.array([-3, 0, 4]))  # [3 0 4]
jnp.sign(jnp.array([-5, 0, 3]))      # [-1 0 1]
jnp.rint(jnp.array([-2.7, 1.5, 2.3])) # [-3.0, 2.0, 2.0]
```

### Exponential and Logarithmic

```python
jnp.exp(x, /)
jnp.exp2(x, /)
jnp.expm1(x, /)
jnp.log(x, /)
jnp.log2(x, /)
jnp.log10(x, /)
jnp.log1p(x, /)
jnp.logaddexp(x1, x2, /)
jnp.logaddexp2(x1, x2, /)
jnp.log2e = jnp.log2(jnp.e)
jnp.log10e = jnp.log10(jnp.e)
jnp.ln2 = jnp.log(2.0)
jnp.ln10 = jnp.log(10.0)
```

```python
x = jnp.array([0.0, 1.0, 2.0])

jnp.exp(x)      # [1.0, 2.718, 7.389]
jnp.exp2(x)     # [1.0, 2.0, 4.0]
jnp.expm1(x)    # [0.0, 1.718, 6.389]
jnp.log(x[1:])  # [0.0, 0.693]
jnp.log2(jnp.array([1.0, 2.0, 4.0, 8.0]))  # [0.0, 1.0, 2.0, 3.0]
jnp.log10(jnp.array([1.0, 10.0, 100.0]))    # [0.0, 1.0, 2.0]

# logaddexp: log(exp(a) + exp(b)), numerically stable
a = jnp.array([-1000.0, -500.0])
b = jnp.array([-999.0, -499.0])
jnp.logaddexp(a, b)  # [-999.307, -498.307]
# Contrast with naive: jnp.log(jnp.exp(a) + jnp.exp(b)) would underflow
```

### Trigonometric Functions

```python
jnp.sin(x, /)
jnp.cos(x, /)
jnp.tan(x, /)
jnp.arcsin(x, /)
jnp.arccos(x, /)
jnp.arctan(x, /)
jnp.arctan2(x1, x2, /)
jnp.sinh(x, /)
jnp.cosh(x, /)
jnp.tanh(x, /)
jnp.arcsinh(x, /)
jnp.arccosh(x, /)
jnp.arctanh(x, /)
jnp.degrees(x, /)
jnp.radians(x, /)
jnp.deg2rad(x, /)
jnp.rad2deg(x, /)
jnp.unwrap(p, discont=None, axis=-1)
```

```python
# Common patterns
angles = jnp.linspace(0, jnp.pi, 5)

jnp.sin(angles)
# [0.0, 0.707, 1.0, 0.707, 0.0]

jnp.cos(angles)
# [1.0, 0.707, 0.0, -0.707, -1.0]

# atan2 for correct quadrant handling
y = jnp.array([1.0, 1.0, -1.0])
x = jnp.array([1.0, -1.0, -1.0])
jnp.arctan2(y, x)
# [0.785, 2.356, -2.356]  (pi/4, 3pi/4, -3pi/4)

# Hyperbolic functions (common in neural networks)
x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
jnp.tanh(x)
# [-0.964, -0.762, 0.0, 0.762, 0.964]

# Degree/radian conversion
jnp.degrees(jnp.pi)     # 180.0
jnp.radians(180.0)      # 3.14159...
```

### Rounding Functions

```python
jnp.around(a, decimals=0, out=None)
jnp.round(a, decimals=0, out=None)     # Alias for around
jnp.rint(x, /)                          # Round to nearest int
jnp.fix(x, /)                           # Round toward zero
jnp.floor(x, /)                         # Round toward -inf
jnp.ceil(x, /)                          # Round toward +inf
jnp.trunc(x, /)                         # Truncate toward zero (same as fix)
```

```python
x = jnp.array([-2.7, -1.5, -0.2, 0.2, 1.5, 2.7])

jnp.round(x)    # [-3.0, -2.0, -0.0, 0.0, 2.0, 3.0] (banker's rounding)
jnp.floor(x)    # [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
jnp.ceil(x)     # [-2.0, -1.0, -0.0, 1.0, 2.0, 3.0]
jnp.trunc(x)    # [-2.0, -1.0, -0.0, 0.0, 1.0, 2.0]
jnp.fix(x)      # [-2.0, -1.0, -0.0, 0.0, 1.0, 2.0]

# Round to specific decimal places
jnp.round(jnp.array([1.234, 5.678]), decimals=2)
# [1.23, 5.68]
```

### Min/Max/Clamp

```python
jnp.maximum(x1, x2, /)
jnp.minimum(x1, x2, /)
jnp.fmax(x1, x2, /)     # NaN-propagating max
jnp.fmin(x1, x2, /)     # NaN-propagating min
jnp.clip(a, a_min=None, a_max=None, out=None)
jnp.heaviside(x1, x2, /)
```

```python
a = jnp.array([1, 3, 5, 7, 9])
b = jnp.array([2, 3, 4, 8, 6])

jnp.maximum(a, b)  # [2, 3, 5, 8, 9]
jnp.minimum(a, b)  # [1, 3, 4, 7, 6]

# fmax/fmin handle NaN differently from maximum/minimum
x = jnp.array([1.0, jnp.nan, 3.0])
y = jnp.array([2.0, 2.0, jnp.nan])
jnp.maximum(x, y)  # [2.0, nan, nan]
jnp.fmax(x, y)     # [2.0, 2.0, 3.0]

# Clip
x = jnp.array([-5, -1, 0, 2, 10])
jnp.clip(x, a_min=0, a_max=5)  # [0, 0, 0, 2, 5]

# Heaviside step function
x = jnp.array([-1.0, 0.0, 1.0])
jnp.heaviside(x, 0.5)  # [0.0, 0.5, 1.0]
```

### Floating-Point Tests

```python
jnp.isnan(x, /)
jnp.isinf(x, /)
jnp.isfinite(x, /)
jnp.isposinf(x, /)
jnp.isneginf(x, /)
jnp.isreal(x, /)        # Always True for JAX (no Python complex scalars in arrays)
jnp.iscomplex(x, /)
```

```python
x = jnp.array([1.0, jnp.nan, jnp.inf, -jnp.inf, 0.0])

jnp.isnan(x)      # [False, True, False, False, False]
jnp.isinf(x)      # [False, False, True, True, False]
jnp.isfinite(x)   # [True, False, False, False, True]
jnp.isposinf(x)   # [False, False, True, False, False]
jnp.isneginf(x)   # [False, False, False, True, False]
```

### Complex Number Functions

```python
jnp.real(val, /)
jnp.imag(val, /)
jnp.angle(z, deg=False)
jnp.conj(x, /)
jnp.conjugate(x, /)
jnp.complex(real, imag)
```

```python
z = jnp.array([1+2j, 3+4j, 5-6j])

jnp.real(z)     # [1.0, 3.0, 5.0]
jnp.imag(z)     # [2.0, 4.0, -6.0]
jnp.abs(z)      # [2.236, 5.0, 7.81]
jnp.angle(z)    # [1.107, 0.927, -0.876] (radians)
jnp.angle(z, deg=True)  # [63.43, 53.13, -50.19]
jnp.conj(z)     # [1-2j, 3-4j, 5+6j]
```

---

## Linear Algebra (jnp.linalg)

### jnp.dot / jnp.vdot / jnp.inner / jnp.outer / jnp.matmul / jnp.tensordot / jnp.einsum / jnp.kron

```python
jnp.dot(a, b, *, precision=None, preferred_element_type=None)
jnp.vdot(a, b, *, precision=None, preferred_element_type=None)
jnp.inner(a, b, *, precision=None)
jnp.outer(a, b, out=None)
jnp.matmul(a, b, *, precision=None, preferred_element_type=None)
jnp.tensordot(a, b, axes=2, *, precision=None, preferred_element_type=None)
jnp.einsum(subscripts, *operands, out=None, dtype=None, order='K',
           casting='safe', optimize='auto', precision=None,
           preferred_element_type=None, _use_xeinsum=False)
jnp.kron(a, b)
```

```python
# Vector dot product
a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
jnp.dot(a, b)    # 32.0
jnp.vdot(a, b)   # 32.0 (conjugates first arg for complex)

# Matrix multiplication
A = jnp.ones((3, 4))
B = jnp.ones((4, 5))
jnp.dot(A, B).shape    # (3, 5)
jnp.matmul(A, B).shape # (3, 5) (same for 2D)

# The @ operator is equivalent to matmul
result = A @ B

# Inner product
jnp.inner(a, b)   # 32.0

# Outer product
jnp.outer(a, b)
# [[ 4.  5.  6.]
#  [ 8. 10. 12.]
#  [12. 15. 18.]]

# Tensor dot
A = jnp.ones((2, 3, 4))
B = jnp.ones((4, 5, 6))
jnp.tensordot(A, B, axes=([2], [0])).shape  # (2, 3, 5, 6)

# Einstein summation
A = jnp.ones((3, 4))
B = jnp.ones((4, 5))
jnp.einsum('ij,jk->ik', A, B).shape  # (3, 5) -- matrix multiply

# Batched matmul with einsum
A = jnp.ones((2, 3, 4))
B = jnp.ones((2, 4, 5))
jnp.einsum('bij,bjk->bik', A, B).shape  # (2, 3, 5)

# Kronecker product
jnp.kron(jnp.array([1, 2]), jnp.array([3, 4]))
# [3, 4, 6, 8]
```

### jnp.linalg.det / jnp.linalg.slogdet

```python
jnp.linalg.det(a)
jnp.linalg.slogdet(a)  # Returns (sign, logabsdet)
```

```python
A = jnp.array([[1, 2], [3, 4]])
jnp.linalg.det(A)  # -2.0

sign, logabsdet = jnp.linalg.slogdet(A)
# sign: -1.0, logabsdet: 0.6931... (ln(2))
```

### jnp.linalg.inv / jnp.linalg.pinv

```python
jnp.linalg.inv(a)
jnp.linalg.pinv(a, rcond=None, hermitian=False)
```

```python
A = jnp.array([[1, 2], [3, 4]])
A_inv = jnp.linalg.inv(A)
# [[-2.0, 1.0],
#  [1.5, -0.5]]

# Verify: A @ A_inv should be identity
A @ A_inv  # approximately [[1, 0], [0, 1]]

# Pseudoinverse (for non-square or singular matrices)
B = jnp.array([[1, 2, 3], [4, 5, 6]])
B_pinv = jnp.linalg.pinv(B)  # shape (3, 2)
```

### jnp.linalg.solve / jnp.linalg.lstsq

```python
jnp.linalg.solve(a, b)
jnp.linalg.lstsq(a, b, rcond=None, *, numpy_resid=False)
```

```python
# Solve Ax = b
A = jnp.array([[3, 1], [1, 2]])
b = jnp.array([9, 8])
x = jnp.linalg.solve(A, b)
# [2.0, 3.0]  (3*2 + 1*3 = 9, 1*2 + 2*3 = 8)

# Least squares: minimize ||Ax - b||^2
A = jnp.array([[1, 1], [1, 2], [1, 3]])
b = jnp.array([1, 2, 2])
x, residuals, rank, sv = jnp.linalg.lstsq(A, b)
```

### jnp.linalg.norm

```python
jnp.linalg.norm(x, ord=None, axis=None, keepdims=False)
```

```python
v = jnp.array([3.0, 4.0])
jnp.linalg.norm(v)        # 5.0 (L2 norm)
jnp.linalg.norm(v, ord=1) # 7.0 (L1 norm)
jnp.linalg.norm(v, ord=jnp.inf)  # 4.0 (max norm)

# Frobenius norm of matrix
A = jnp.array([[1, 2], [3, 4]])
jnp.linalg.norm(A)  # 5.477... (sqrt(1+4+9+16))

# Nuclear norm
jnp.linalg.norm(A, ord='nuc')  # Sum of singular values
```

### jnp.linalg.eig / jnp.linalg.eigh / jnp.linalg.eigvals / jnp.linalg.eigvalsh

```python
jnp.linalg.eig(a)              # Eigenvalues and eigenvectors of general matrix
jnp.linalg.eigh(a, UPLO='L')   # Eigenvalues and eigenvectors of Hermitian matrix
jnp.linalg.eigvals(a)          # Eigenvalues only (general)
jnp.linalg.eigvalsh(a, UPLO='L')  # Eigenvalues only (Hermitian)
```

```python
# Symmetric matrix (use eigh for efficiency)
A = jnp.array([[2, 1], [1, 2]])
eigenvalues, eigenvectors = jnp.linalg.eigh(A)
# eigenvalues: [1.0, 3.0]
# eigenvectors: columns are the eigenvectors

# General matrix
B = jnp.array([[0, -1], [1, 0]])
eigenvalues, eigenvectors = jnp.linalg.eig(B)
# eigenvalues: [0+1j, 0-1j]
```

### jnp.linalg.svd

```python
jnp.linalg.svd(a, full_matrices=True, compute_uv=True, hermitian=False)
```

```python
A = jnp.array([[1, 2], [3, 4], [5, 6]])
U, S, Vt = jnp.linalg.svd(A, full_matrices=False)
# U: shape (3, 2), S: shape (2,), Vt: shape (2, 2)
# Reconstruction: U @ jnp.diag(S) @ Vt ~= A
```

### jnp.linalg.qr

```python
jnp.linalg.qr(a, mode='reduced')
```

```python
A = jnp.array([[1, 2], [3, 4], [5, 6]])
Q, R = jnp.linalg.qr(A)
# Q: shape (3, 2) orthogonal, R: shape (2, 2) upper triangular
```

### jnp.linalg.cholesky

```python
jnp.linalg.cholesky(a)
```

```python
# Positive definite matrix
A = jnp.array([[4, 2], [2, 3]])
L = jnp.linalg.cholesky(A)
# [[2.0, 0.0],
#  [1.0, 1.0]]
# Verify: L @ L.T == A
```

### jnp.linalg.lu

```python
jnp.linalg.lu(a)
```

Returns `(P, L, U)` where `P @ L @ U = a`.

```python
A = jnp.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]])
P, L, U = jnp.linalg.lu(A)
```

### jnp.linalg.matrix_rank / jnp.linalg.matrix_power / jnp.linalg.matrix_exp / jnp.linalg.cond / jnp.linalg.slogdet

```python
jnp.linalg.matrix_rank(a, tol=None, hermitian=False)
jnp.linalg.matrix_power(a, n)
jnp.linalg.matrix_exp(A)
jnp.linalg.cond(x, p=None)
jnp.linalg.multi_dot(arrays, *, precision=None)
jnp.linalg.tensorinv(a, ind=2)
jnp.linalg.tensorsolve(a, b, axes=None)
jnp.linalg.schur(a, output='real')
```

```python
# Matrix rank
A = jnp.array([[1, 2], [2, 4]])  # Rank-deficient
jnp.linalg.matrix_rank(A)  # 1

# Matrix power
A = jnp.array([[1, 0], [0, 1]])
jnp.linalg.matrix_power(A, 3)  # Identity

# Condition number
A = jnp.array([[1, 0], [0, 2]])
jnp.linalg.cond(A)  # 2.0

# Matrix exponential (e^A, not element-wise)
A = jnp.array([[0, 1], [-1, 0]])
jnp.linalg.matrix_exp(A)  # Rotation matrix ~= [[cos(1), sin(1)], [-sin(1), cos(1)]]
```

---

## Statistics

### Sum and Product

```python
jnp.sum(a, axis=None, dtype=None, keepdims=False, initial=None, where=None)
jnp.prod(a, axis=None, dtype=None, keepdims=False, initial=None, where=None)
jnp.nansum(a, axis=None, dtype=None, keepdims=False, initial=None, where=None)
jnp.nanprod(a, axis=None, dtype=None, keepdims=False, initial=None, where=None)
jnp.cumsum(a, axis=None, dtype=None, out=None)
jnp.cumprod(a, axis=None, dtype=None, out=None)
jnp.nancumsum(a, axis=None, dtype=None, out=None)
jnp.nancumprod(a, axis=None, dtype=None, out=None)
```

```python
a = jnp.array([[1, 2, 3], [4, 5, 6]])

jnp.sum(a)           # 21
jnp.sum(a, axis=0)   # [5, 7, 9]
jnp.sum(a, axis=1)   # [6, 15]
jnp.sum(a, axis=1, keepdims=True)  # [[6], [15]]

jnp.prod(a, axis=0)  # [4, 10, 18]
jnp.cumsum(a[0])     # [1, 3, 6]
jnp.cumprod(a[0])    # [1, 2, 6]

# NaN-safe versions
b = jnp.array([1.0, jnp.nan, 3.0, jnp.nan, 5.0])
jnp.nansum(b)        # 9.0 (ignores NaN)
jnp.sum(b)           # nan
jnp.nancumsum(b)     # [1.0, 1.0, 4.0, 4.0, 9.0]
```

### Mean, Variance, Standard Deviation

```python
jnp.mean(a, axis=None, dtype=None, keepdims=False, where=None)
jnp.var(a, axis=None, dtype=None, ddof=0, keepdims=False, where=None)
jnp.std(a, axis=None, dtype=None, ddof=0, keepdims=False, where=None)
jnp.nanmean(a, axis=None, dtype=None, keepdims=False, where=None)
jnp.nanvar(a, axis=None, dtype=None, ddof=0, keepdims=False, where=None)
jnp.nanstd(a, axis=None, dtype=None, ddof=0, keepdims=False, where=None)
```

```python
a = jnp.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])

jnp.mean(a)   # 5.0
jnp.var(a)    # 4.0
jnp.std(a)    # 2.0

# With ddof (delta degrees of freedom)
jnp.var(a, ddof=1)  # 4.571... (unbiased sample variance)

# NaN-safe versions
b = jnp.array([1.0, jnp.nan, 3.0, 4.0])
jnp.nanmean(b)  # 2.667...

# 2D with axis
m = jnp.array([[1, 2], [3, 4], [5, 6]], dtype=jnp.float32)
jnp.mean(m, axis=0)  # [3.0, 4.0]
jnp.mean(m, axis=1)  # [1.5, 3.5, 5.5]
```

### Median, Percentile, Quantile

```python
jnp.median(a, axis=None, keepdims=False, *, where=None)
jnp.average(a, axis=None, weights=None, returned=False, keepdims=False)
jnp.percentile(a, q, axis=None, out=None, overwrite_input=False,
               method='linear', keepdims=False, interpolation=None)
jnp.quantile(a, q, axis=None, out=None, overwrite_input=False,
             method='linear', keepdims=False, interpolation=None)
jnp.nanpercentile(a, q, axis=None, out=None, overwrite_input=False,
                  method='linear', keepdims=False, interpolation=None)
jnp.nanquantile(a, q, axis=None, out=None, overwrite_input=False,
                method='linear', keepdims=False, interpolation=None)
```

```python
a = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

jnp.median(a)                 # 3.0
jnp.percentile(a, 50)         # 3.0
jnp.percentile(a, [25, 50, 75])  # [2.0, 3.0, 4.0]
jnp.quantile(a, 0.5)          # 3.0

# Weighted average
weights = jnp.array([1.0, 1.0, 1.0, 1.0, 4.0])
jnp.average(a, weights=weights)  # 3.714... (biased toward 5.0)

# Interpolation methods for percentile/quantile
# 'linear' (default), 'lower', 'higher', 'midpoint', 'nearest'
```

### Min/Max and Range

```python
jnp.min(a, axis=None, keepdims=False, initial=None, where=None)
jnp.max(a, axis=None, keepdims=False, initial=None, where=None)
jnp.amin(a, ...)  # Alias for min
jnp.amax(a, ...)  # Alias for max
jnp.argmin(a, axis=None, keepdims=False)
jnp.argmax(a, axis=None, keepdims=False)
jnp.ptp(a, axis=None, keepdims=False)
jnp.nanmin(a, axis=None, keepdims=False, initial=None, where=None)
jnp.nanmax(a, axis=None, keepdims=False, initial=None, where=None)
```

```python
a = jnp.array([[1, 4, 2], [5, 3, 6]])

jnp.min(a)            # 1
jnp.max(a)            # 6
jnp.min(a, axis=0)    # [1, 3, 2]
jnp.min(a, axis=1)    # [1, 3]
jnp.ptp(a)            # 5 (max - min = 6 - 1)

jnp.argmin(a)         # 0 (flat index)
jnp.argmax(a)         # 5 (flat index)
jnp.argmax(a, axis=0) # [1, 0, 1]
jnp.argmax(a, axis=1) # [1, 2]
```

### Correlation and Covariance

```python
jnp.corrcoef(x, y=None, rowvar=True)
jnp.cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None)
```

```python
x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = jnp.array([2.0, 4.0, 5.0, 4.0, 5.0])

jnp.corrcoef(x, y)
# [[1.0, 0.7746...],
#  [0.7746..., 1.0]]

jnp.cov(x, y)
# [[2.5, 1.5],
#  [1.5, 1.3]]
```

### Histograms

```python
jnp.histogram(a, bins=10, range=None, weights=None, density=None)
jnp.histogram2d(x, y, bins=10, range=None, weights=None, density=None)
jnp.histogramdd(sample, bins=10, range=None, weights=None, density=None)
jnp.bincount(x, weights=None, minlength=0, *, length=None)
jnp.digitize(x, bins, right=False)
```

```python
# 1D histogram
data = jnp.array([1.2, 1.5, 2.3, 2.8, 3.1, 3.5, 4.0, 4.2, 4.8, 5.0])
counts, edges = jnp.histogram(data, bins=5)
# counts: [2, 1, 2, 2, 3] (depending on range)

# Custom bin edges
counts, edges = jnp.histogram(data, bins=jnp.array([0, 2, 4, 6]))
# counts: [2, 5, 3]

# bincount
indices = jnp.array([0, 1, 1, 2, 2, 2, 3])
jnp.bincount(indices)  # [1, 2, 3, 1]

# With weights
weights = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
jnp.bincount(indices, weights=weights)  # [1.0, 5.0, 15.0, 7.0]

# digitize
x = jnp.array([0.2, 6.4, 3.0, 1.6])
bins = jnp.array([0.0, 1.0, 2.5, 4.0, 10.0])
jnp.digitize(x, bins)  # [1, 4, 3, 2]
```

**JAX difference:** `jnp.bincount` requires the `length` parameter inside `jax.jit` to specify the output size.

---

## FFT (jnp.fft)

### Standard FFT

```python
jnp.fft.fft(a, n=None, axis=-1, norm=None)
jnp.fft.ifft(a, n=None, axis=-1, norm=None)
jnp.fft.fft2(a, s=None, axes=(-2, -1), norm=None)
jnp.fft.ifft2(a, s=None, axes=(-2, -1), norm=None)
jnp.fft.fftn(a, s=None, axes=None, norm=None)
jnp.fft.ifftn(a, s=None, axes=None, norm=None)
```

```python
# 1D FFT
x = jnp.array([1.0, 2.0, 3.0, 4.0])
jnp.fft.fft(x)
# [10+0j, -2+2j, -2+0j, -2-2j]

jnp.fft.ifft(jnp.fft.fft(x))
# [1+0j, 2+0j, 3+0j, 4+0j] (round-trip)

# 2D FFT
img = jnp.ones((4, 4))
jnp.fft.fft2(img)
# 16 at (0,0), 0 elsewhere for constant input

# N-dimensional FFT
data_3d = jnp.ones((4, 4, 4))
jnp.fft.fftn(data_3d)  # 64 at (0,0,0), 0 elsewhere
```

### Real FFT (more efficient for real inputs)

```python
jnp.fft.rfft(a, n=None, axis=-1, norm=None)
jnp.fft.irfft(a, n=None, axis=-1, norm=None)
jnp.fft.rfft2(a, s=None, axes=(-2, -1), norm=None)
jnp.fft.irfft2(a, s=None, axes=(-2, -1), norm=None)
jnp.fft.rfftn(a, s=None, axes=None, norm=None)
jnp.fft.irfftn(a, s=None, axes=None, norm=None)
```

```python
# rfft for real input: output is complex, length is n//2+1
x = jnp.array([1.0, 2.0, 3.0, 4.0])
y = jnp.fft.rfft(x)
# [10+0j, -2+2j, -2+0j] (length 3 = 4//2 + 1)

# irfft recovers real signal
jnp.fft.irfft(y, n=4)
# [1.0, 2.0, 3.0, 4.0]
```

### Hermitian FFT

```python
jnp.fft.hfft(a, n=None, axis=-1, norm=None)
jnp.fft.ihfft(a, n=None, axis=-1, norm=None)
```

### Frequency and Shift Utilities

```python
jnp.fft.fftfreq(n, d=1.0)
jnp.fft.rfftfreq(n, d=1.0)
jnp.fft.fftshift(x, axes=None)
jnp.fft.ifftshift(x, axes=None)
```

```python
# Frequency bins
jnp.fft.fftfreq(8)
# [0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125]

jnp.fft.rfftfreq(8)
# [0.0, 0.125, 0.25, 0.375, 0.5]

# Shift zero-frequency to center
x = jnp.array([0, 1, 2, 3, 4, 5, 6, 7])
jnp.fft.fftshift(x)
# [4, 5, 6, 7, 0, 1, 2, 3]

jnp.fft.ifftshift(jnp.fft.fftshift(x))
# [0, 1, 2, 3, 4, 5, 6, 7] (round-trip)
```

---

## Searching and Sorting

### Sorting Functions

```python
jnp.sort(a, axis=-1, kind='quicksort', order=None, stable=False, descending=False)
jnp.argsort(a, axis=-1, kind='quicksort', order=None, stable=False, descending=False)
jnp.lexsort(keys, axis=-1)
jnp.msort(a)  # Merge sort along first axis
jnp.sort_complex(a)
jnp.partition(a, kth, axis=-1, kind='introselect', order=None)
jnp.argpartition(a, kth, axis=-1, kind='introselect', order=None)
```

```python
a = jnp.array([3, 1, 4, 1, 5, 9, 2, 6])

jnp.sort(a)                    # [1 1 2 3 4 5 6 9]
jnp.sort(a, descending=True)   # [9 6 5 4 3 2 1 1]
jnp.argsort(a)                 # [1 3 6 0 2 4 7 5]

# Stable sort (preserves order of equal elements)
data = jnp.array([3, 1, 4, 1])
jnp.sort(data, stable=True)

# Partition: partial sort
jnp.partition(a, 3)
# [1, 1, 2, 3, 4, 5, 6, 9] -- element at index 3 is in its final sorted position
```

### Search Functions

```python
jnp.searchsorted(a, v, side='left', sorter=None, method='scan')
jnp.argmin(a, axis=None, keepdims=False)
jnp.argmax(a, axis=None, keepdims=False)
jnp.nonzero(a, *, size=None, fill_value=None)
jnp.where(condition, x=None, y=None)
jnp.extract(condition, arr)
jnp.count_nonzero(a, axis=None, keepdims=False)
```

```python
sorted_a = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
jnp.searchsorted(sorted_a, jnp.array([3, 5, 10]))
# [2, 4, 9]

jnp.searchsorted(sorted_a, jnp.array([3, 5, 10]), side='right')
# [3, 5, 9]

# nonzero
x = jnp.array([[1, 0, 3], [0, 5, 0]])
rows, cols = jnp.nonzero(x)
# rows: [0, 0, 1], cols: [0, 2, 1]

# extract
x = jnp.arange(10)
jnp.extract(x > 5, x)  # [6, 7, 8, 9]

# count_nonzero
jnp.count_nonzero(x > 5)  # 4
```

---

## Logic and Comparison

### Boolean Array Functions

```python
jnp.all(a, axis=None, keepdims=False, *, where=None)
jnp.any(a, axis=None, keepdims=False, *, where=None)
jnp.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
jnp.isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
jnp.array_equal(a1, a2, equal_nan=False)
jnp.array_equiv(a1, a2)
```

```python
a = jnp.array([True, True, False])
jnp.all(a)   # False
jnp.any(a)   # True

# allclose for approximate comparison
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([1.0, 2.00001, 3.0])
jnp.allclose(x, y)   # True
jnp.isclose(x, y)    # [True, True, True]

# Exact equality
jnp.array_equal(jnp.array([1, 2]), jnp.array([1, 2]))  # True
jnp.array_equal(jnp.array([1, 2]), jnp.array([1.0, 2.0]))  # False (different dtype)
jnp.array_equiv(jnp.array([1, 2]), jnp.array([1.0, 2.0]))  # True (ignores dtype)
```

### Element-wise Comparison

```python
jnp.equal(x1, x2, /)           # ==
jnp.not_equal(x1, x2, /)       # !=
jnp.less(x1, x2, /)            # <
jnp.less_equal(x1, x2, /)      # <=
jnp.greater(x1, x2, /)         # >
jnp.greater_equal(x1, x2, /)   # >=
```

```python
a = jnp.array([1, 2, 3, 4, 5])
b = jnp.array([3, 2, 1, 4, 6])

jnp.equal(a, b)          # [False, True, False, True, False]
jnp.not_equal(a, b)      # [True, False, True, False, True]
jnp.less(a, b)           # [True, False, False, False, True]
jnp.greater_equal(a, b)  # [False, True, True, True, False]
```

### Logical Operations

```python
jnp.logical_and(x1, x2, /)
jnp.logical_or(x1, x2, /)
jnp.logical_not(x, /)
jnp.logical_xor(x1, x2, /)
```

```python
a = jnp.array([True, True, False, False])
b = jnp.array([True, False, True, False])

jnp.logical_and(a, b)  # [True, False, False, False]
jnp.logical_or(a, b)   # [True, True, True, False]
jnp.logical_not(a)     # [False, False, True, True]
jnp.logical_xor(a, b)  # [False, True, True, False]

# Works with non-boolean arrays (truthy/falsy)
jnp.logical_and(jnp.array([0, 1, 2]), jnp.array([0, 0, 3]))
# [False, False, True]
```

### Bitwise Operations (on integer arrays)

```python
jnp.bitwise_and(x1, x2, /)
jnp.bitwise_or(x1, x2, /)
jnp.bitwise_xor(x1, x2, /)
jnp.bitwise_not(x, /)
jnp.invert(x, /)     # Alias for bitwise_not
jnp.left_shift(x1, x2, /)
jnp.right_shift(x1, x2, /)
```

```python
a = jnp.array([0b1100, 0b1010], dtype=jnp.int32)
b = jnp.array([0b1010, 0b1100], dtype=jnp.int32)

jnp.bitwise_and(a, b)  # [0b1000, 0b1000]
jnp.bitwise_or(a, b)   # [0b1110, 0b1110]
jnp.bitwise_xor(a, b)  # [0b0110, 0b0110]

jnp.left_shift(jnp.array([1, 2, 4], dtype=jnp.int32), 2)
# [4, 8, 16]

jnp.right_shift(jnp.array([4, 8, 16], dtype=jnp.int32), 2)
# [1, 2, 4]
```

---

## Type Operations

### jnp.astype

```python
a.astype(dtype)
```

```python
x = jnp.array([1.5, 2.7, 3.9])
x.astype(jnp.int32)      # [1, 2, 3] (truncates)
x.astype(jnp.float16)    # [1.5, 2.7, 3.9] (lower precision)
```

### jnp.can_cast / jnp.promote_types / jnp.result_type

```python
jnp.can_cast(from_, to, casting='safe')
jnp.promote_types(type1, type2)
jnp.result_type(*arrays_and_dtypes)
```

```python
jnp.can_cast(jnp.int32, jnp.float32)    # True (safe)
jnp.can_cast(jnp.float32, jnp.int32)    # False (lossy)
jnp.can_cast(jnp.float32, jnp.int32, casting='unsafe')  # True

jnp.promote_types(jnp.float32, jnp.float64)  # float64
jnp.promote_types(jnp.int32, jnp.float32)     # float32

jnp.result_type(jnp.array([1], dtype=jnp.int32), jnp.array([1.0], dtype=jnp.float64))
# float64
```

### jnp.finfo / jnp.iinfo

```python
jnp.finfo(dtype)
jnp.iinfo(dtype)
```

```python
# Float info
info = jnp.finfo(jnp.float32)
info.bits        # 32
info.eps         # 1.1920929e-07 (smallest representable step near 1.0)
info.max         # 3.4028235e+38
info.min         # -3.4028235e+38
info.tiny        # 1.1754944e-38 (smallest positive normal)

# Integer info
info = jnp.iinfo(jnp.int32)
info.min    # -2147483648
info.max    # 2147483647
```

### jnp.issubdtype

```python
jnp.issubdtype(arg1, arg2)
```

```python
jnp.issubdtype(jnp.float32, jnp.floating)   # True
jnp.issubdtype(jnp.int32, jnp.integer)       # True
jnp.issubdtype(jnp.float32, jnp.integer)     # False
jnp.issubdtype(jnp.complex64, jnp.complexfloating)  # True
```

---

## Differences from NumPy

Understanding the key differences between `jax.numpy` and standard `numpy` is critical for writing correct JAX code.

### 1. Immutability

JAX arrays are immutable. In-place operations are not supported.

```python
# NumPy
a = np.array([1, 2, 3])
a[0] = 10  # Works

# JAX
b = jnp.array([1, 2, 3])
b = b.at[0].set(10)  # Returns a new array; original unchanged
```

### 2. Default dtypes

JAX defaults to 32-bit; NumPy defaults to 64-bit.

```python
# NumPy
np.array([1, 2, 3]).dtype       # int64
np.array([1.0, 2.0]).dtype      # float64

# JAX
jnp.array([1, 2, 3]).dtype      # int32
jnp.array([1.0, 2.0]).dtype     # float32
```

### 3. Pure Functions and Random State

JAX requires explicit PRNG state. `jnp.random` does not exist; use `jax.random` instead.

```python
# NumPy
np.random.seed(42)
np.random.randn(3)  # Uses global state

# JAX
key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
jax.random.normal(subkey, (3,))
```

### 4. Out-of-Bounds Indexing

NumPy allows out-of-bounds indexing (raises errors or clips). JAX clips indices to valid range.

```python
# NumPy
a = np.array([1, 2, 3])
a[5]  # IndexError

# JAX
b = jnp.array([1, 2, 3])
b[5]  # 3 (clamped to last index)
b.at[-1].get()  # 3
b.at[-5].get()  # 1 (clamped to first index)
```

### 5. Dynamic Shapes

JAX requires array shapes to be known at compile time (for JIT). Operations that produce variable-sized outputs require special handling.

```python
# This will fail under jit because boolean indexing produces variable-size output
@jax.jit
def bad(x):
    return x[x > 0]  # Error: output shape depends on data

# Fix: use fixed-size output with masking
@jax.jit
def good(x):
    mask = x > 0
    return jnp.where(mask, x, 0.0)
```

### 6. The `.at` Indexing API

JAX uses the `.at` property for all indexed updates:

```python
x = jnp.zeros(5)

x.at[2].set(1.0)          # [0, 0, 1, 0, 0]
x.at[1:4].add(1.0)        # [0, 1, 1, 1, 0]
x.at[0].max(5.0)          # [5.0, 0, 0, 0, 0]
x.at[0].min(-1.0)         # [-1.0, 0, 0, 0, 0]
x.at[::2].mul(2.0)        # [0, 0, 0, 0, 0]
x.at[[0, 2, 4]].set(1.0)  # [1.0, 0, 1.0, 0, 1.0]

# Apply a function at specific indices
x.at[2].apply(lambda v: v + 10)
```

### 7. Supported Subset

Most NumPy functions are supported, but some are not:

**Not supported or partially supported:**
- `jnp.histogramdd` with non-uniform bins
- Some advanced indexing patterns (e.g., assigning to boolean masks)
- In-place operations (`*=`, `+=`, etc.)
- `jnp.ndarray.view` (use `jax.lax.bitcast_convert_type`)
- Some `numpy.lib.stride_tricks` functions

**JAX-specific additions:**
- `jnp.take_along_axis` / `jnp.argmax` with `keepdims` support
- `jnp.top_k` (via `jax.lax.top_k`)
- `jax.nn` activation functions (`relu`, `softmax`, `sigmoid`, etc.)
- Device placement control (`jax.device_put`)
- `jax.vmap` for automatic vectorization of any `jnp` function

### 8. Precision Control

JAX allows specifying XLA precision for matrix operations:

```python
# Control precision for matmul on TPUs
jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)

# Three precision levels:
# Precision.DEFAULT     - fastest, may use bfloat16 on some hardware
# Precision.HIGH        - float32 computation
# Precision.HIGHEST     - float32 with highest accuracy algorithm
```

### 9. Device Arrays

JAX arrays reside on specific devices (CPU, GPU, TPU). Transfers between devices are explicit.

```python
# Check device
x = jnp.array([1, 2, 3])
x.devices()  # {CpuDevice(id=0)}

# Transfer to specific device
x_gpu = jax.device_put(x, jax.devices('gpu')[0])

# Block until computation completes (for timing)
x.block_until_ready()
```

### 10. Control Flow

Use JAX-specific control flow instead of Python control flow inside `jax.jit`:

```python
# DON'T: Python if/for inside jit ( traced values become static)
@jax.jit
def bad(x):
    if x > 0:    # x is a tracer, this is evaluated at trace time
        return x
    return -x

# DO: Use jnp.where or lax.cond
@jax.jit
def good(x):
    return jnp.where(x > 0, x, -x)
    # Or: return jax.lax.cond(x > 0, lambda x: x, lambda x: -x, x)
```
