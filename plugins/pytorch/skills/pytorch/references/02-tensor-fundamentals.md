# PyTorch Reference - Chapter 2: Tensor Fundamentals

This chapter covers tensor creation, indexing, slicing, reshaping, properties, types, and all fundamental tensor manipulation operations in comprehensive detail.

---

## 2.1 Tensor Creation (Factory Functions)

### 2.1.1 torch.tensor

Creates a tensor from Python data (list, tuple, NumPy array, scalar, etc.).

```python
torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False)
```

**Parameters:**
- `data` (array_like): Initial data for the tensor. Can be a list, tuple, NumPy ndarray, scalar, or other array-like object.
- `dtype` (`torch.dtype`, optional): The desired data type of the tensor. If `None`, inferred from `data`.
- `device` (`torch.device`, optional): The desired device of the tensor. If `None`, uses the default device.
- `requires_grad` (bool, optional): If `True`, the tensor will track gradients. Default: `False`.
- `pin_memory` (bool, optional): If `True`, allocates the tensor in pinned memory (CPU only). Default: `False`.

**Examples:**
```python
import torch

# From a Python list
t = torch.tensor([[1, 2, 3], [4, 5, 6]])
# tensor([[1, 2, 3],
#         [4, 5, 6]])

# Specify dtype
t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
# tensor([1., 2., 3.], dtype=torch.float64)

# On GPU
t = torch.tensor([1, 2, 3], device='cuda:0')

# With gradient tracking
t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Scalar (0-dimensional)
t = torch.tensor(42)  # shape: torch.Size([])

# Boolean
t = torch.tensor([True, False, True])
```

**Notes:**
- `torch.tensor()` always copies the data. Use `torch.as_tensor()` to avoid copying when possible.
- Nested lists must have consistent shapes (no jagged arrays).
- For scalars, consider `torch.tensor(5)` vs `torch.tensor([5])` -- the first is 0-dimensional.

### 2.1.2 torch.as_tensor

Creates a tensor sharing data with the original when possible (avoids copying).

```python
torch.as_tensor(data, dtype=None, device=None)
```

**Examples:**
```python
import numpy as np

arr = np.array([1, 2, 3])
t = torch.as_tensor(arr)     # Shares memory with arr (no copy)
t[0] = 99                    # Also modifies arr

t = torch.as_tensor(arr, dtype=torch.float32)  # Copies (dtype differs)

# From Python list (always copies since lists have no shared memory protocol)
t = torch.as_tensor([1, 2, 3])
```

**Notes:**
- When `data` is a NumPy array and the dtype matches, no copy is made.
- When `data` is already a tensor with matching dtype and device, returns the same tensor.

### 2.1.3 torch.from_numpy

Creates a tensor from a NumPy array that shares the same memory.

```python
torch.from_numpy(ndarray)
```

**Examples:**
```python
import numpy as np

arr = np.array([1, 2, 3], dtype=np.float32)
t = torch.from_numpy(arr)    # Shares memory, dtype: torch.float32

arr = np.array([1, 2, 3], dtype=np.float64)
t = torch.from_numpy(arr)    # dtype: torch.float64

arr = np.array([1, 2, 3], dtype=np.complex64)
t = torch.from_numpy(arr)    # dtype: torch.complex64
```

**Notes:**
- Only supports arrays with dtypes that have a PyTorch equivalent (bool, uint8, int8/16/32/64, float32/64, complex64/128).
- Modifications to the tensor modify the NumPy array and vice versa.
- Use `t.numpy()` to convert back to NumPy (only for CPU tensors).

### 2.1.4 torch.zeros

Creates a tensor filled with zeros.

```python
torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)
```

**Parameters:**
- `*size` (int...): A sequence of integers defining the shape, or a single integer for 1D, or a `torch.Size`.
- `out` (Tensor, optional): Output tensor to write into.
- `dtype` (`torch.dtype`, optional): Element type. Default: `torch.float32` (or the default dtype).
- `layout` (`torch.layout`, optional): Memory layout. Default: `torch.strided`.
- `device` (`torch.device`, optional): Device. Default: default device.
- `requires_grad` (bool, optional): Track gradients. Default: `False`.
- `pin_memory` (bool, optional): Allocate in pinned memory. Default: `False`.

**Examples:**
```python
torch.zeros(3)                  # tensor([0., 0., 0.])
torch.zeros(2, 3)              # 2x3 matrix of zeros
torch.zeros(2, 3, 4)          # 2x3x4 tensor of zeros
torch.zeros((2, 3), dtype=torch.int32)
torch.zeros(2, 3, device='cuda')
torch.zeros(2, 3, requires_grad=True)
```

### 2.1.5 torch.ones

Creates a tensor filled with ones.

```python
torch.ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)
```

**Examples:**
```python
torch.ones(3)                   # tensor([1., 1., 1.])
torch.ones(2, 3)               # 2x3 matrix of ones
torch.ones(2, 3, dtype=torch.int64, device='cuda')
```

### 2.1.6 torch.empty

Creates an uninitialized tensor (contains garbage values -- must be filled before use).

```python
torch.empty(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False, memory_format=torch.contiguous_format)
```

**Parameters:**
- `memory_format` (`torch.memory_format`): The desired memory format. Options: `torch.contiguous_format`, `torch.channels_last`, `torch.preserve_format`.

**Examples:**
```python
torch.empty(2, 3)              # 2x3 uninitialized tensor
torch.empty(2, 3, memory_format=torch.channels_last)
```

**Notes:**
- The contents of an empty tensor are undefined. Use `torch.zeros()` if you need guaranteed initial values.
- Often used as an output buffer: `torch.empty(2, 3, out=existing_tensor)`.

### 2.1.7 torch.full

Creates a tensor filled with a given value.

```python
torch.full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)
```

**Examples:**
```python
torch.full((2, 3), 7.0)        # 2x3 filled with 7.0
torch.full((3,), 3.14, dtype=torch.float64)
torch.full((2, 3), float('inf'))  # Fill with infinity
torch.full((2, 3), True)       # Boolean tensor filled with True
```

### 2.1.8 torch.arange

Returns a 1D tensor with values from `start` to `end` with the given `step`.

```python
torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

**Parameters:**
- `start` (Number): The beginning value. Default: 0.
- `end` (Number): The ending value (exclusive).
- `step` (Number): The gap between each pair of values. Default: 1.

**Examples:**
```python
torch.arange(5)                # tensor([0, 1, 2, 3, 4])
torch.arange(1, 4)            # tensor([1, 2, 3])
torch.arange(0, 10, 2)        # tensor([0, 2, 4, 6, 8])
torch.arange(5, 0, -1)        # tensor([5, 4, 3, 2, 1])
torch.arange(1, 2.5, 0.5)     # tensor([1.0000, 1.5000, 2.0000])
torch.arange(3, dtype=torch.float32)  # tensor([0., 1., 2.])
```

**Notes:**
- `end` is exclusive (unlike `torch.linspace` which includes both endpoints).
- When using floating-point `step`, the result may be affected by floating-point rounding errors.
- If `dtype` is not specified, it is inferred from the other arguments. Mixing int and float arguments produces a float tensor.

### 2.1.9 torch.linspace

Returns a 1D tensor with `steps` evenly spaced values from `start` to `end` (inclusive).

```python
torch.linspace(start, end, steps=100, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

**Parameters:**
- `start` (Number): The starting value.
- `end` (Number): The ending value (inclusive).
- `steps` (int): Number of points. Default: 100.

**Examples:**
```python
torch.linspace(0, 1, 5)        # tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
torch.linspace(-10, 10, 5)    # tensor([-10., -5., 0., 5., 10.])
torch.linspace(0, 10, 1)      # tensor([0.])
torch.linspace(1, 2, 3)       # tensor([1.0000, 1.5000, 2.0000])
torch.linspace(0, 2 * 3.14159, 100)  # 100 points from 0 to 2*pi
```

### 2.1.10 torch.logspace

Returns a 1D tensor with `steps` logarithmically spaced values from `base^start` to `base^end`.

```python
torch.logspace(start, end, steps=100, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

**Examples:**
```python
torch.logspace(0, 3, 4)        # tensor([1., 10., 100., 1000.])
torch.logspace(1, 2, 5)       # tensor([10.0000, 17.7828, 31.6228, 56.2341, 100.])
torch.logspace(0, 1, 5, base=2)  # [1.0, 1.1892, 1.4142, 1.6818, 2.0]
torch.logspace(-1, 1, 5)      # [0.1, 0.316, 1.0, 3.162, 10.0]
```

### 2.1.11 torch.eye

Returns a 2D tensor with ones on the diagonal and zeros elsewhere (identity matrix).

```python
torch.eye(n, m=None, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

**Parameters:**
- `n` (int): Number of rows.
- `m` (int, optional): Number of columns. Default: `n`.

**Examples:**
```python
torch.eye(3)
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])

torch.eye(2, 3)
# tensor([[1., 0., 0.],
#         [0., 1., 0.]])

torch.eye(4, device='cuda')
```

### 2.1.12 torch.rand

Returns a tensor filled with random numbers from a uniform distribution on [0, 1).

```python
torch.rand(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)
```

**Examples:**
```python
torch.rand(3)                  # 1D: 3 random values in [0, 1)
torch.rand(2, 3)              # 2x3 random matrix
torch.rand(2, 3, 4, device='cuda')
torch.rand(2, 3, generator=torch.Generator().manual_seed(42))  # Reproducible
```

### 2.1.13 torch.randn

Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (standard normal).

```python
torch.randn(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)
```

**Examples:**
```python
torch.randn(3)                # 1D: 3 values from N(0, 1)
torch.randn(2, 3)            # 2x3 matrix from N(0, 1)
torch.randn(3, 4, device='cuda')
torch.randn(2, 3, dtype=torch.float64)
```

### 2.1.14 torch.randint

Returns a tensor filled with random integers generated uniformly between `low` (inclusive) and `high` (exclusive).

```python
torch.randint(low=0, high, size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

**Examples:**
```python
torch.randint(0, 10, (3,))    # e.g., tensor([3, 7, 1])
torch.randint(0, 5, (2, 3))  # e.g., tensor([[0, 4, 2], [3, 1, 4]])
torch.randint(3, (2, 2))     # low=0, high=3: values in {0, 1, 2}
torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)  # Random image
```

### 2.1.15 torch.randperm

Returns a random permutation of integers from 0 to `n-1`.

```python
torch.randperm(n, *, generator=None, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)
```

**Examples:**
```python
torch.randperm(5)             # e.g., tensor([2, 0, 4, 1, 3])
torch.randperm(10, device='cuda')
torch.randperm(100, generator=torch.Generator().manual_seed(0))
```

### 2.1.16 torch.randn_like

Returns a tensor with the same shape as `input`, filled with random numbers from N(0, 1).

```python
torch.randn_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
```

**Examples:**
```python
x = torch.empty(2, 3)
torch.randn_like(x)           # 2x3 tensor from N(0, 1)
torch.randn_like(x, dtype=torch.float64)  # Override dtype
```

### 2.1.17 torch.zeros_like

Returns a tensor with the same shape as `input`, filled with zeros.

```python
torch.zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
```

**Examples:**
```python
x = torch.randn(3, 4)
torch.zeros_like(x)            # 3x4 zeros, same dtype/device
torch.zeros_like(x, dtype=torch.int32)  # Override dtype
```

### 2.1.18 torch.ones_like

Returns a tensor with the same shape as `input`, filled with ones.

```python
torch.ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
```

### 2.1.19 torch.empty_like

Returns an uninitialized tensor with the same shape as `input`.

```python
torch.empty_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
```

### 2.1.20 torch.full_like

Returns a tensor with the same shape as `input`, filled with `fill_value`.

```python
torch.full_like(input, fill_value, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
```

**Examples:**
```python
x = torch.randn(2, 3)
torch.full_like(x, 3.14)      # 2x3 filled with 3.14, same dtype as x
torch.full_like(x, 0, dtype=torch.int32)  # Override dtype
```

### 2.1.21 torch.randint_like

Returns a tensor with the same shape as `input`, filled with random integers.

```python
torch.randint_like(input, low=0, high, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
```

### 2.1.22 torch.rand_like

Returns a tensor with the same shape as `input`, filled with uniform [0, 1) values.

```python
torch.rand_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
```

### 2.1.23 Other Creation Functions

```python
# From raw memory buffer
torch.frombuffer(buffer, *, dtype, count=-1, offset=0, requires_grad=False)

# Flexible creation (like as_tensor but handles more cases)
torch.asarray(data, *, dtype=None, device=None)

# Sparse tensor creation
torch.sparse_coo_tensor(indices, values, size=None, *, dtype=None, device=None, requires_grad=False)
torch.sparse_csr_tensor(crow_indices, col_indices, values, size=None, *, dtype=None, device=None)

# Complex number creation
torch.complex(real, imag)       # Create complex tensor from real and imaginary parts
torch.polar(abs, angle)         # Create complex tensor from magnitude and angle

# Block diagonal matrix
torch.block_diag(*tensors)

# Vandermonde matrix
torch.vander(x, N=None, increasing=False)

# Triangular
torch.tril(input, diagonal=0, *, out=None)   # Lower triangle
torch.triu(input, diagonal=0, *, out=None)   # Upper triangle

# Tensor from another tensor's metadata
torch.empty_strided(size, stride, *, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False)

# Scalar tensor
torch.scalar_tensor(s, *, dtype=None, layout=None, device=None, requires_grad=False)
```

---

## 2.2 Tensor Indexing

### 2.2.1 Basic Indexing

PyTorch supports Python-style indexing with integers, slices, and tensors.

```python
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Single element access
t[0, 1]                        # tensor(2) - element at row 0, col 1
t[-1, -1]                      # tensor(9) - last element

# Row/column selection
t[0]                           # tensor([1, 2, 3]) - first row
t[:, 0]                        # tensor([1, 4, 7]) - first column

# Slicing
t[0:2]                         # first two rows
t[:, 1:]                       # columns from index 1 onward
t[::2]                         # every other row (step=2)
t[::-1]                        # reversed rows (step=-1)

# Slicing with step
t[0:3:2]                       # rows 0 and 2 (step 2)
t[:, ::2]                      # every other column

# Mixed indexing
t[0, 1:3]                      # tensor([2, 3]) - first row, cols 1 and 2
t[1:, :2]                      # last two rows, first two columns
```

### 2.2.2 Advanced Indexing

Advanced indexing uses integer or boolean tensors to select elements.

```python
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Integer array indexing
indices = torch.tensor([0, 2])
t[indices]                     # rows 0 and 2 - shape (2, 3)
t[:, indices]                  # columns 0 and 2 - shape (3, 2)

# Fancy indexing (multiple index arrays)
rows = torch.tensor([0, 1, 2])
cols = torch.tensor([2, 1, 0])
t[rows, cols]                  # tensor([3, 5, 7]) - elements at (0,2), (1,1), (2,0)

# Broadcasting with index arrays
rows = torch.tensor([[0, 0], [1, 1]])
cols = torch.tensor([[0, 1], [0, 1]])
t[rows, cols]                  # tensor([[1, 2], [4, 5]])

# Assignment with advanced indexing
t[torch.tensor([0, 2]), torch.tensor([1, 1])] = 99
# t[0][1] and t[2][1] become 99
```

### 2.2.3 Boolean Masking

```python
t = torch.tensor([1, 2, 3, 4, 5, 6])

# Boolean mask
mask = t > 3                   # tensor([False, False, False, True, True, True])
t[mask]                        # tensor([4, 5, 6])

# Direct boolean indexing
t[t > 3]                       # tensor([4, 5, 6])
t[t % 2 == 0]                  # tensor([2, 4, 6]) - even elements

# Assignment with boolean mask
t[t > 3] = 0                   # tensor([1, 2, 3, 0, 0, 0])

# Boolean mask on multi-dimensional tensor
t = torch.randn(3, 4)
mask = t > 0
t[mask]                        # 1D tensor of all positive values

# torch.where for conditional selection
torch.where(t > 3, t, torch.zeros_like(t))  # Keep values > 3, else 0
```

### 2.2.4 torch.where

```python
# Three-argument form: conditional selection
torch.where(condition, x, y)

# One-argument form: find nonzero indices
torch.where(condition)
```

**Parameters (three-argument form):**
- `condition` (BoolTensor): Where `True`, yield `x`; where `False`, yield `y`.
- `x` (Tensor): Values selected where condition is `True`.
- `y` (Tensor): Values selected where condition is `False`.

**Examples:**
```python
x = torch.randn(3, 2)
y = torch.ones(3, 2)

# Conditional selection
torch.where(x > 0, x, y)      # Use x where positive, else 1

# Scalar arguments
torch.where(x > 0, x, 0)      # Use x where positive, else 0

# One-arg form: find indices of nonzero elements
t = torch.tensor([[1, 0], [0, 1]])
torch.where(t > 0)            # (tensor([0, 1]), tensor([0, 1]))

# Multi-dimensional
t = torch.tensor([[[1, 0], [2, 3]], [[0, 0], [4, 5]]])
torch.where(t > 0)
# (tensor([0, 0, 1, 1]), tensor([0, 1, 1, 1]), tensor([0, 1, 0, 1]))
```

### 2.2.5 torch.index_select

Selects elements from `input` at the 1D `index` tensor along `dim`.

```python
torch.index_select(input, dim, index, *, out=None)
```

**Parameters:**
- `input` (Tensor): The input tensor.
- `dim` (int): The dimension to index.
- `index` (1D LongTensor): The indices to select.

**Examples:**
```python
t = torch.randn(3, 4)
indices = torch.tensor([0, 2])

torch.index_select(t, 0, indices)   # Select rows 0 and 2, shape: (2, 4)
torch.index_select(t, 1, indices)   # Select columns 0 and 2, shape: (3, 2)

# Negative indices not supported - use torch.tensor([0, 2])
```

### 2.2.6 torch.masked_select

Returns a new 1D tensor with elements from `input` where `mask` is `True`.

```python
torch.masked_select(input, mask, *, out=None)
```

**Examples:**
```python
t = torch.randn(3, 4)
mask = t > 0
torch.masked_select(t, mask)   # 1D tensor of all positive values

# Equivalent to: t[mask]
# But masked_select returns a new tensor (not a view)
```

### 2.2.7 torch.gather

Gathers values along `dim` at the specified indices.

```python
torch.gather(input, dim, index, *, sparse_grad=False, out=None)
```

**Parameters:**
- `input` (Tensor): The source tensor.
- `dim` (int): The axis along which to index.
- `index` (LongTensor): The indices of elements to gather. Must have the same number of dimensions as `input`.
- `sparse_grad` (bool): If `True`, gradient w.r.t. `input` will be a sparse tensor.

**Formula:**
- For 3D tensor with `dim=0`: `out[i][j][k] = input[index[i][j][k]][j][k]`
- For 3D tensor with `dim=1`: `out[i][j][k] = input[i][index[i][j][k]][k]`
- For 3D tensor with `dim=2`: `out[i][j][k] = input[i][j][index[i][j][k]]`

**Examples:**
```python
t = torch.tensor([[1, 2], [3, 4]])

# Gather along dim=0 (rows)
index = torch.tensor([[0, 1], [1, 0]])
torch.gather(t, 0, index)     # tensor([[1, 4], [3, 2]])
# out[0][0] = input[index[0][0]][0] = input[0][0] = 1
# out[0][1] = input[index[0][1]][1] = input[1][1] = 4
# out[1][0] = input[index[1][0]][0] = input[1][0] = 3
# out[1][1] = input[index[1][1]][1] = input[0][1] = 2

# Gather along dim=1 (columns)
index = torch.tensor([[1, 0], [0, 1]])
torch.gather(t, 1, index)     # tensor([[2, 1], [3, 4]])
```

### 2.2.8 torch.scatter

The inverse of `gather`. Writes values from `src` into `self` at positions specified by `index`.

```python
torch.scatter(input, dim, index, src)
self.scatter_(dim, index, src)      # In-place version
```

**Parameters:**
- `dim` (int): The axis along which to scatter.
- `index` (LongTensor): The target indices.
- `src` (Tensor or float): The source values.

**Formula (for 3D, dim=0):**
`self[index[i][j][k]][j][k] = src[i][j][k]`

**Examples:**
```python
# Scatter from a tensor
t = torch.zeros(2, 4)
index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
src = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
t.scatter_(1, index, src)
# tensor([[1., 2., 3., 4.],
#         [5., 6., 7., 8.]])

# Scatter a scalar value
t = torch.zeros(3, 5)
index = torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]])
t.scatter_(0, index, -1.0)   # Put -1.0 at specified (row, col) positions

# Scatter with reduction (scatter_add)
t = torch.zeros(2, 4)
index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
src = torch.ones(2, 4)
t.scatter_add_(1, index, src)  # Accumulate: each position gets +1
```

### 2.2.9 torch.scatter_add

Adds values from `src` into `self` at positions specified by `index`.

```python
self.scatter_add_(dim, index, src)
```

**Examples:**
```python
t = torch.zeros(2, 4)
index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
src = torch.ones(2, 4)
t.scatter_add_(1, index, src)  # Adds 1.0 to each position

# Useful for bincount-like operations
t = torch.zeros(5)
index = torch.tensor([0, 1, 1, 3, 3, 3])
src = torch.ones(6)
t.scatter_add_(0, index, src)  # tensor([1., 2., 0., 3., 0.])
```

### 2.2.10 torch.index_add

Adds values from `source` to `self` at the indices specified by `index`.

```python
self.index_add_(dim, index, source, *, alpha=1)
```

**Parameters:**
- `dim` (int): The dimension along which to add.
- `index` (1D LongTensor): Indices of `self` to add to.
- `source` (Tensor): Values to add.
- `alpha` (Number): Scalar multiplier for `source`.

**Examples:**
```python
t = torch.ones(5, 3)
index = torch.tensor([0, 2, 4])
source = torch.randn(3, 3)
t.index_add_(0, index, source)  # Add source rows to rows 0, 2, 4 of t

# With alpha
t.index_add_(0, index, source, alpha=0.5)  # Add 0.5 * source
```

### 2.2.11 torch.index_copy

Copies values from `source` into `self` at the specified indices.

```python
self.index_copy_(dim, index, source)
```

**Examples:**
```python
t = torch.zeros(5, 3)
index = torch.tensor([0, 2, 4])
source = torch.ones(3, 3)
t.index_copy_(0, index, source)  # Copy source rows into rows 0, 2, 4
```

### 2.2.12 torch.index_fill

Fills elements of `self` at the given indices with a value.

```python
self.index_fill_(dim, index, value)
```

**Examples:**
```python
t = torch.ones(3, 4)
index = torch.tensor([0, 2])
t.index_fill_(0, index, -1.0)  # Fill rows 0 and 2 with -1.0
t.index_fill_(1, index, 0.0)   # Fill columns 0 and 2 with 0.0
```

### 2.2.13 torch.index_put

Puts values from `values` into `self` at the indices specified by the tuple `indices`.

```python
self.index_put_(indices, values, accumulate=False)
```

**Parameters:**
- `indices` (tuple of LongTensor): One 1D tensor per dimension.
- `values` (Tensor): Values to put.
- `accumulate` (bool): If `True`, adds values instead of replacing.

**Examples:**
```python
t = torch.zeros(3, 3)
idx = (torch.tensor([0, 1]), torch.tensor([1, 2]))
t.index_put_(idx, torch.tensor([5.0, 7.0]))
# tensor([[0., 5., 0.],
#         [0., 0., 7.],
#         [0., 0., 0.]])

# With accumulate
t.index_put_(idx, torch.tensor([1.0, 1.0]), accumulate=True)
# Adds 1.0 to positions (0,1) and (1,2)
```

### 2.2.14 torch.take / torch.take_along_dim

```python
# torch.take: Flattens tensor and selects at 1D indices
torch.take(input, index)

# torch.take_along_dim: Takes values along a given dimension
torch.take_along_dim(input, indices, dim=None)
```

**Examples:**
```python
t = torch.tensor([[1, 2, 3], [4, 5, 6]])
torch.take(t, torch.tensor([0, 2, 5]))  # tensor([1, 3, 6])
torch.take(t, torch.tensor([0, 3]))     # tensor([1, 4]) - row-major order

# take_along_dim: gather that matches the output shape
values, indices = torch.max(t, dim=1, keepdim=True)
# values shape: (2, 1), indices shape: (2, 1)
torch.take_along_dim(t, indices, dim=1)  # shape: (2, 1) - max per row

# Sort and take along dim
sorted_indices = torch.argsort(t, dim=1)
torch.take_along_dim(t, sorted_indices, dim=1)  # Sorted tensor
```

### 2.2.15 torch.nonzero

Returns indices of nonzero elements.

```python
torch.nonzero(input, *, as_tuple=False)
```

**Examples:**
```python
t = torch.tensor([[1, 0, 2], [0, 3, 0]])
torch.nonzero(t)
# tensor([[0, 0], [0, 2], [1, 1]])

torch.nonzero(t, as_tuple=True)
# (tensor([0, 0, 1]), tensor([0, 2, 1]))
```

### 2.2.16 torch.count_nonzero

```python
torch.count_nonzero(input, dim=None)
```

**Examples:**
```python
t = torch.tensor([[0, 1, 0], [1, 1, 0]])
torch.count_nonzero(t)          # tensor(3)
torch.count_nonzero(t, dim=0)   # tensor([1, 2, 0])
torch.count_nonzero(t, dim=1)   # tensor([1, 2])
```

---

## 2.3 Slicing

### 2.3.1 Slice Patterns

PyTorch supports all Python slice patterns: `start:stop:step`.

```python
t = torch.arange(10)  # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Basic slicing
t[2:5]                         # tensor([2, 3, 4])
t[:5]                          # tensor([0, 1, 2, 3, 4])
t[5:]                          # tensor([5, 6, 7, 8, 9])
t[:]                           # All elements (view, not copy)

# Step
t[::2]                         # tensor([0, 2, 4, 6, 8]) - every other element
t[1::2]                        # tensor([1, 3, 5, 7, 9]) - odd indices
t[::-1]                        # tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]) - reverse
t[::-2]                        # tensor([9, 7, 5, 3, 1]) - reverse, every other

# Negative indexing
t[-3:]                         # tensor([7, 8, 9]) - last 3 elements
t[-5:-2]                       # tensor([5, 6, 7]) - 5th from end to 2nd from end
t[-1]                          # tensor(9) - last element
```

### 2.3.2 Multi-dimensional Slicing

```python
t = torch.randn(3, 4, 5)

# Slicing each dimension independently
t[0]                           # shape: (4, 5) - first "page"
t[0, :, :]                     # Same as t[0]
t[:, 1, :]                     # shape: (3, 5) - second row of each page
t[:, :, 2]                     # shape: (3, 4) - third column of each page
t[0:2, 1:3, :]                # shape: (2, 2, 5)
t[0, 1:3, 2:4]                # shape: (2, 2)

# Ellipsis: expands to fill missing dimensions
t[..., 0]                      # Same as t[:, :, 0] - shape: (3, 4)
t[0, ...]                      # Same as t[0, :, :] - shape: (4, 5)
t[..., ::2, :]                 # Every other along second-to-last dim

# None: inserts a new dimension (unsqueeze via indexing)
t[:, None, :]                  # shape: (3, 1, 4, 5)
t[None]                        # shape: (1, 3, 4, 5)
t[None, ..., None]             # shape: (1, 3, 4, 5, 1)

# Mixed basic and advanced indexing
t[0, torch.tensor([1, 3])]    # shape: (2,) - row 0, columns 1 and 3
t[:, torch.tensor([0, 2]), :]  # shape: (3, 2, 5) - select specific rows
```

### 2.3.3 torch.narrow

Narrow a tensor along a dimension (returns a view).

```python
torch.narrow(input, dim, start, length)
```

**Examples:**
```python
t = torch.arange(12).reshape(3, 4)
torch.narrow(t, 0, 0, 2)      # First 2 rows (like t[0:2])
torch.narrow(t, 1, 1, 2)      # 2 columns starting at index 1 (like t[:, 1:3])
```

### 2.3.4 torch.select

Selects a single slice along a dimension at the given index (reduces dimensionality by 1).

```python
torch.select(input, dim, index)
```

**Examples:**
```python
t = torch.randn(3, 4)
torch.select(t, 0, 1)         # Same as t[1] - second row, shape: (4,)
torch.select(t, 1, 2)         # Same as t[:, 2] - third column, shape: (3,)
```

### 2.3.5 torch.index_select (see Section 2.2.5)

---

## 2.4 Reshaping

### 2.4.1 view

Returns a new tensor with the same data but a different shape. Does not copy data.

```python
tensor.view(*shape)
```

**Examples:**
```python
t = torch.arange(6)
t.view(2, 3)                  # shape: (2, 3)
t.view(3, 2)                  # shape: (3, 2)
t.view(-1, 2)                 # shape: (3, 2) - -1 infers dimension
t.view(2, -1)                 # shape: (2, 3)
t.view(1, 2, 3)               # shape: (1, 2, 3)

# Use -1 to infer one dimension
t = torch.arange(24)
t.view(2, -1, 4)              # shape: (2, 3, 4)
```

**Notes:**
- The tensor must be contiguous for `view()` to work. If not, call `.contiguous()` first or use `.reshape()` instead.
- The total number of elements must remain the same.
- Exactly one dimension can be -1 (inferred from remaining elements).
- `view()` returns a view (shares memory), not a copy.

### 2.4.2 reshape

Returns a tensor with the same data but a different shape. Works on both contiguous and non-contiguous tensors.

```python
torch.reshape(input, shape)
tensor.reshape(*shape)
```

**Examples:**
```python
t = torch.arange(6)
t.reshape(2, 3)               # shape: (2, 3) - may or may not share memory
t.reshape(-1)                 # Flatten to 1D

# Key difference from view:
t = torch.arange(6).reshape(2, 3)[:, 0]  # Non-contiguous slice
t.view(3, 1)                  # RuntimeError! (not contiguous)
t.reshape(3, 1)               # Works (copies data if needed)

# Using a tuple or individual args
torch.reshape(t, (3, 2))
t.reshape(3, 2)
```

### 2.4.3 permute

Returns a view of the original tensor with its dimensions rearranged.

```python
tensor.permute(*dims)
```

**Examples:**
```python
t = torch.randn(2, 3, 5)
t.permute(2, 0, 1)            # shape: (5, 2, 3) - rearranged dimensions
t.permute(0, 2, 1)            # shape: (2, 5, 3)

# Common pattern: transpose last two dims
t = torch.randn(2, 3, 4)
t.permute(0, 2, 1)            # Equivalent to t.transpose(1, 2)

# BHWC -> BCHW (channels last to channels first)
img = torch.randn(1, 224, 224, 3)
img.permute(0, 3, 1, 2)       # shape: (1, 3, 224, 224)

# BCHW -> BHWC (channels first to channels last)
feat = torch.randn(1, 64, 56, 56)
feat.permute(0, 2, 3, 1)      # shape: (1, 56, 56, 64)
```

### 2.4.4 transpose

Returns a view that transposes two specified dimensions.

```python
tensor.transpose(dim0, dim1)
torch.transpose(input, dim0, dim1)
```

**Examples:**
```python
t = torch.randn(2, 3)
t.transpose(0, 1)             # shape: (3, 2) - standard matrix transpose
t.T                            # Shorthand for .transpose(0, 1) on 2D tensors

# For higher-dimensional tensors
t = torch.randn(2, 3, 4)
t.transpose(1, 2)             # shape: (2, 4, 3) - swap dims 1 and 2

# Notes on .T behavior:
# - For 1D tensors: returns the same tensor (no change)
# - For 2D tensors: equivalent to .transpose(0, 1)
# - For >=3D tensors: reverses ALL dimensions (use .mT for just last two)
```

### 2.4.5 contiguous

Returns a contiguous in-memory tensor containing the same data.

```python
tensor.contiguous()
```

**Examples:**
```python
t = torch.randn(3, 4)
s = t[:, 0]                   # Non-contiguous view
s.is_contiguous()              # False
c = s.contiguous()             # Contiguous copy
c.is_contiguous()              # True

# contiguous() only copies if the tensor is not already contiguous
t.contiguous()                 # Returns same tensor (already contiguous)
```

### 2.4.6 flatten

Flattens a range of dimensions into one.

```python
torch.flatten(input, start_dim=0, end_dim=-1)
tensor.flatten(start_dim=0, end_dim=-1)
```

**Examples:**
```python
t = torch.randn(2, 3, 4, 5)
t.flatten()                    # shape: (120,) - all dims
t.flatten(0)                   # shape: (120,) - flatten from dim 0
t.flatten(1)                   # shape: (2, 60) - flatten dims 1..3
t.flatten(1, 2)                # shape: (2, 12, 5) - flatten dims 1..2
t.flatten(start_dim=2)         # shape: (2, 3, 20) - flatten dims 2..3

# Common usage in neural networks
x = torch.randn(32, 3, 28, 28)  # Batch of images
x.flatten(1)                     # shape: (32, 2352) - flatten per sample
```

### 2.4.7 squeeze / unsqueeze

```python
# squeeze: Removes dimensions of size 1
tensor.squeeze(dim=None)
torch.squeeze(input, dim=None)

# unsqueeze: Adds a dimension of size 1 at the specified position
tensor.unsqueeze(dim)
torch.unsqueeze(input, dim)
```

**Examples of squeeze:**
```python
t = torch.randn(1, 3, 1, 4)

t.squeeze()                    # shape: (3, 4) - removes ALL size-1 dims
t.squeeze(0)                   # shape: (3, 1, 4) - removes dim 0
t.squeeze(2)                   # shape: (1, 3, 4) - removes dim 2
t.squeeze(1)                   # shape: (1, 3, 1, 4) - no change (dim 1 is size 3)
t.squeeze(-1)                  # shape: (1, 3, 1) - removes last dim
```

**Examples of unsqueeze:**
```python
t = torch.randn(3, 4)

t.unsqueeze(0)                 # shape: (1, 3, 4) - add batch dimension
t.unsqueeze(1)                 # shape: (3, 1, 4) - add dim between
t.unsqueeze(-1)                # shape: (3, 4, 1) - add trailing dim
t.unsqueeze(-2)                # shape: (3, 1, 4)

# Equivalent using None indexing
t[None]                        # Same as t.unsqueeze(0)
t[:, None]                     # Same as t.unsqueeze(1)
t[:, :, None]                  # Same as t.unsqueeze(-1)
```

### 2.4.8 expand / expand_as

Returns a new view with singleton dimensions expanded to a larger size (without copying data).

```python
tensor.expand(*sizes)
tensor.expand_as(other)
```

**Examples:**
```python
t = torch.randn(1, 3)
t.expand(4, 3)                 # shape: (4, 3) - row repeated 4 times (view!)
t.expand(2, 4, 3)              # shape: (2, 4, 3)
t.expand(-1, 3)                # shape: (1, 3) - -1 means "keep unchanged"

t = torch.randn(3, 1)
other = torch.randn(3, 4)
t.expand_as(other)             # shape: (3, 4)
```

**Notes:**
- `expand()` does NOT allocate new memory. It returns a view with stride 0 for expanded dimensions.
- Only dimensions of size 1 can be expanded.
- Use `-1` to keep a dimension unchanged.

### 2.4.9 repeat

Repeats the tensor along each dimension. This DOES allocate new memory.

```python
tensor.repeat(*sizes)
```

**Examples:**
```python
t = torch.tensor([1, 2, 3])
t.repeat(2)                    # tensor([1, 2, 3, 1, 2, 3])
t.repeat(2, 3)                 # shape: (2, 9) - 2 copies of 3 copies
t.repeat(3, 1)                 # shape: (3, 3)

t = torch.randn(2, 3)
t.repeat(3, 1)                 # shape: (6, 3) - 3 copies along dim 0
t.repeat(1, 2)                 # shape: (2, 6) - 2 copies along dim 1
t.repeat(2, 2)                 # shape: (4, 6) - 2 copies along both dims
```

### 2.4.10 broadcast_to

Broadcasts a tensor to a given shape (similar to `expand` but validates shape).

```python
torch.broadcast_to(input, size)
```

**Examples:**
```python
t = torch.randn(1, 3)
torch.broadcast_to(t, (4, 3))  # shape: (4, 3)

# Must follow broadcasting rules
torch.broadcast_to(t, (2, 4, 3))  # shape: (2, 4, 3)
```

### 2.4.11 reshape_as

Reshapes this tensor to the same shape as `other`.

```python
tensor.reshape_as(other)
```

### 2.4.12 movedim / moveaxis

Moves dimensions from source positions to destination positions.

```python
torch.movedim(input, source, destination)
torch.moveaxis(input, source, destination)
```

**Examples:**
```python
t = torch.randn(2, 3, 4)
torch.movedim(t, 0, 2)        # shape: (3, 4, 2)
torch.movedim(t, [0, 1], [1, 0])  # shape: (3, 2, 4) - swap dims 0 and 1
torch.movedim(t, [0, 1, 2], [2, 0, 1])  # shape: (4, 2, 3)
```

### 2.4.13 swapdims / swapaxes

Swaps two dimensions.

```python
torch.swapdims(input, dim0, dim1)
torch.swapaxes(input, dim0, dim1)
```

**Examples:**
```python
t = torch.randn(2, 3, 4)
torch.swapdims(t, 0, 2)       # shape: (4, 3, 2)
torch.swapdims(t, 1, 2)       # shape: (2, 4, 3) - same as t.transpose(1, 2)
```

### 2.4.14 unflatten

Unflattens a dimension into multiple dimensions (inverse of flatten).

```python
tensor.unflatten(dim, sizes)
```

**Examples:**
```python
t = torch.randn(2, 12)
t.unflatten(1, (3, 4))         # shape: (2, 3, 4)
t.unflatten(1, (2, 2, 3))     # shape: (2, 2, 2, 3)
t.unflatten(1, (-1, 4))       # shape: (2, 3, 4) - -1 inferred
```

### 2.4.15 .t() and .mT

```python
# .t(): 2D transpose (alias for .transpose(0, 1))
tensor.t()

# .mT: Matrix transpose of last two dimensions
tensor.mT

# .mH: Matrix conjugate transpose of last two dimensions
tensor.mH

# .H: Conjugate transpose (for 2D: conjugate + transpose)
tensor.H
```

---

## 2.5 Tensor Properties

### 2.5.1 Shape and Size

```python
t = torch.randn(2, 3, 4)

t.shape                         # torch.Size([2, 3, 4])
t.size()                        # torch.Size([2, 3, 4])
t.size(0)                       # 2 (size of dimension 0)
t.size(1)                       # 3 (size of dimension 1)
t.size(-1)                      # 4 (size of last dimension)
t.dim()                         # 3 (number of dimensions, aka ndim)
t.ndim                          # 3 (alias for dim())
t.numel()                       # 24 (total number of elements)
t.element_size()                # 4 (bytes per element for float32)
```

### 2.5.2 Memory Properties

```python
t = torch.randn(2, 3, 4)

t.is_contiguous()               # True
t.stride()                      # (12, 4, 1) - element strides per dimension
t.storage_offset()              # 0 (offset into underlying storage)
t.data_ptr()                    # Memory address (integer pointer)

# Stride interpretation:
# To move along dim 0: skip 12 elements
# To move along dim 1: skip 4 elements
# To move along dim 2: skip 1 element

# Non-contiguous example
s = t.transpose(0, 1)
s.is_contiguous()               # False
s.stride()                      # (4, 12, 1)
s.contiguous().stride()         # (12, 4, 1)
```

### 2.5.3 Device and Type Properties

```python
t = torch.randn(2, 3, device='cuda:0')

t.device                        # device(type='cuda', index=0)
t.is_cuda                       # True
t.is_cpu                        # False
t.is_xpu                        # False
t.is_mps                        # False
t.dtype                         # torch.float32
t.layout                        # torch.strided
```

### 2.5.4 Gradient Properties

```python
t = torch.randn(2, 3, requires_grad=True)

t.requires_grad                 # True
t.grad                          # None (before backward)
t.grad_fn                       # None (leaf tensor)
t.is_leaf                       # True
t.retains_grad                  # False (default)

# Non-leaf tensor
y = t * 2 + 1
y.requires_grad                 # True
y.grad_fn                       # <AddBackward0 object>
y.is_leaf                       # False

# Control gradient retention
t.retain_grad()                 # Retain gradient for non-leaf tensor
```

### 2.5.5 Other Properties

```python
t = torch.randn(2, 3)

t.is_sparse                     # False
t.is_quantized                  # False
t.is_complex()                  # False
t.is_floating_point()           # True
t.is_inference()                # False
t.is_conj()                     # False
t.is_neg()                      # False
t.names                         # Named dimension names (or None)

# For complex tensors
c = torch.complex(torch.randn(3), torch.randn(3))
c.real                          # Real part
c.imag                          # Imaginary part
c.is_complex()                  # True

# Shorthand properties
t.T                             # Transpose (reverse all dims)
t.mT                            # Matrix transpose (transpose last 2 dims)
t.mH                            # Matrix conjugate transpose
t.H                             # Conjugate transpose (2D only)
t.real                          # Real part
t.imag                          # Imaginary part
t.A                             # Alias for .T (deprecated)
```

---

## 2.6 In-Place Operations

PyTorch uses the underscore suffix (`_`) convention for in-place operations.

### 2.6.1 In-Place Arithmetic

```python
t = torch.randn(3, 4)

t.add_(1)                      # t = t + 1
t.sub_(1)                      # t = t - 1
t.mul_(2)                      # t = t * 2
t.div_(2)                      # t = t / 2
t.fill_(0)                     # Fill with zeros
t.zero_()                      # Fill with zeros
t.ones_()                      # Fill with ones

# In-place with alpha
t.add_(other, alpha=0.5)       # t = t + 0.5 * other
```

### 2.6.2 In-Place Math

```python
t.abs_()                       # Absolute value
t.sqrt_()                      # Square root
t.exp_()                       # Exponential
t.log_()                       # Natural log
t.log2_()                      # Log base 2
t.log10_()                     # Log base 10
t.pow_(2)                      # Power
t.rsqrt_()                     # 1/sqrt
t.reciprocal_()                # 1/x
t.ceil_()                      # Ceiling
t.floor_()                     # Floor
t.round_()                     # Round
t.trunc_()                     # Truncate toward zero
t.frac_()                      # Fractional part
t.clamp_(min=0, max=1)         # Clamp values
t.sigmoid_()                   # Sigmoid
t.relu_()                      # ReLU
t.tanh_()                      # Tanh
t.sign_()                      # Sign
t.neg_()                       # Negate
```

### 2.6.3 In-Place Random

```python
t.normal_(mean=0, std=1)       # Fill with N(0, 1)
t.uniform_(low=0, high=1)     # Fill with U(0, 1)
t.random_(low=0, high=10)     # Fill with random integers [0, 10)
t.bernoulli_(p=0.5)            # Fill with Bernoulli(0.5)
t.cauchy_(median=0, sigma=1)  # Fill with Cauchy distribution
t.exponential_(lambd=1.0)      # Fill with Exp(1.0)
t.geometric_(p=0.5)            # Fill with Geometric(0.5)
t.log_normal_(mean=0, std=1)   # Fill with LogNormal(0, 1)
```

### 2.6.4 In-Place Scatter/Index

```python
t.scatter_(dim, index, src)
t.scatter_add_(dim, index, src)
t.index_add_(dim, index, source)
t.index_copy_(dim, index, source)
t.index_fill_(dim, index, value)
t.masked_fill_(mask, value)
t.masked_scatter_(mask, tensor)
```

### 2.6.5 In-Place Copy

```python
t.copy_(src, non_blocking=False)  # Copy src into t
```

**Important notes on in-place operations:**
- In-place operations can break autograd if applied to tensors needed for gradient computation. PyTorch will raise a `RuntimeError` in such cases.
- In-place operations on leaf tensors with `requires_grad=True` will raise an error unless wrapped in `torch.no_grad()`.
- In-place operations on views propagate changes to the base tensor.

---

## 2.7 clone, detach, to, Type Casting

### 2.7.1 clone

Returns a deep copy of the tensor that preserves the computation graph.

```python
tensor.clone(memory_format=torch.preserve_format)
```

**Examples:**
```python
x = torch.randn(3, 4, requires_grad=True)
y = x.clone()                  # Deep copy, requires_grad=True, grad_fn=CloneBackward
y.sum().backward()             # Gradients flow back to x
print(x.grad)                  # tensor([[1., 1., 1., 1.], ...])

# With specific memory format
y = x.clone(memory_format=torch.contiguous_format)
```

### 2.7.2 detach

Returns a new tensor that shares storage but is detached from the computation graph.

```python
tensor.detach()
```

**Examples:**
```python
x = torch.randn(3, 4, requires_grad=True)
y = x.detach()                 # Shares data, requires_grad=False, no grad_fn
y.sum().backward()             # RuntimeError! y is not part of the graph

# Common pattern: logging without affecting gradients
loss = model(input)
with torch.no_grad():
    logged_loss = loss.detach().item()  # Safe to use outside graph
```

### 2.7.3 detach_ (in-place)

Detaches the tensor from the graph in-place. All references to this tensor will now be detached.

```python
tensor.detach_()
```

### 2.7.4 to

Moves and/or casts the tensor. This is the primary method for device and type conversion.

```python
tensor.to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format)
tensor.to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format)
tensor.to(other, non_blocking=False, copy=False)
```

**Examples:**
```python
t = torch.randn(3, 4)

# Change dtype
t.to(torch.float64)            # Cast to float64
t.to(torch.int32)              # Cast to int32
t.to(torch.bfloat16)           # Cast to bfloat16

# Change device
t.to('cuda')                   # Move to default GPU
t.to('cuda:0')                 # Move to GPU 0
t.to(torch.device('cuda:0'))   # Same as above
t.to('cpu')                    # Move to CPU

# Change device and dtype simultaneously
t.to('cuda', torch.float64)    # Move to GPU and cast to float64

# Match another tensor's dtype and device
other = torch.randn(3, 4, dtype=torch.float64, device='cuda')
t.to(other)                    # Same dtype and device as other

# Non-blocking transfer (useful with pinned memory)
t.to('cuda', non_blocking=True)

# Force copy
t.to(torch.float32, copy=True)  # Always creates a new tensor

# From the TensorOptions API
t.to(memory_format=torch.channels_last)
```

### 2.7.5 Type Casting Methods (Shorthand)

```python
t = torch.randn(3, 4)

t.float()                      # Cast to torch.float32
t.double()                     # Cast to torch.float64
t.half()                       # Cast to torch.float16
t.bfloat16()                   # Cast to torch.bfloat16
t.int()                        # Cast to torch.int32
t.long()                       # Cast to torch.int64
t.short()                      # Cast to torch.int16
t.char()                       # Cast to torch.int8
t.byte()                       # Cast to torch.uint8
t.bool()                       # Cast to torch.bool

# For complex types
t.complex()                    # View float tensor as complex (pairs of floats)
```

### 2.7.6 type()

Returns the type name or casts to a specific type.

```python
tensor.type(dtype=None, non_blocking=False)
```

**Examples:**
```python
t = torch.randn(3, 4)
t.type()                       # 'torch.FloatTensor'
t.type('torch.DoubleTensor')   # Cast to double and return
t.type(torch.float64)          # Same
```

---

## 2.8 Cat, Stack, Split, Chunk

### 2.8.1 torch.cat

Concatenates tensors along an existing dimension.

```python
torch.cat(tensors, dim=0, *, out=None)
```

**Parameters:**
- `tensors` (sequence of Tensors): Must have the same shape except in the concatenating dimension.
- `dim` (int): The dimension over which to concatenate.

**Examples:**
```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)

torch.cat([a, b], dim=0)       # shape: (4, 3) - vertical stack
torch.cat([a, b], dim=1)       # shape: (2, 6) - horizontal stack

# Multiple tensors
c = torch.randn(2, 3)
torch.cat([a, b, c], dim=0)    # shape: (6, 3)

# Different shapes (except concat dim)
d = torch.randn(2, 5)
torch.cat([a, d], dim=0)       # shape: (4, ...) -- wait, this fails!
# a is (2, 3), d is (2, 5): dims 1 don't match for dim=0 concat
# Actually dim=0 concat only requires dim!=0 to match:
# a is (2, 3), d is (2, 5) -> sizes must match except in dim 0
# Since dim 1 differs (3 vs 5), this would fail

# Correct: concat along dim where other dims match
e = torch.randn(2, 3)
torch.cat([a, e], dim=0)       # shape: (4, 3) - works
```

### 2.8.2 torch.stack

Concatenates tensors along a **new** dimension.

```python
torch.stack(tensors, dim=0, *, out=None)
```

**Parameters:**
- `tensors` (sequence of Tensors): Must all have the same shape.
- `dim` (int): The new dimension to insert.

**Examples:**
```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)

torch.stack([a, b], dim=0)     # shape: (2, 2, 3) - stack along new dim 0
torch.stack([a, b], dim=1)     # shape: (2, 2, 3) - stack along new dim 1
torch.stack([a, b], dim=2)     # shape: (2, 3, 2) - stack along new dim 2
torch.stack([a, b], dim=-1)    # shape: (2, 3, 2) - stack along new last dim

# Three tensors
c = torch.randn(2, 3)
torch.stack([a, b, c])         # shape: (3, 2, 3) - default dim=0
```

### 2.8.3 torch.split

Splits a tensor into sub-tensors along a given dimension.

```python
torch.split(tensor, split_size_or_sections, dim=0)
```

**Parameters:**
- `split_size_or_sections` (int or list of ints): If int, the size of each chunk (last chunk may be smaller). If list, the sizes of each chunk (must sum to the dimension size).
- `dim` (int): The dimension along which to split.

**Examples:**
```python
t = torch.arange(10)

# Equal-sized chunks
torch.split(t, 3)              # (tensor([0,1,2]), tensor([3,4,5]), tensor([6,7,8,9]))
# Last chunk may be smaller

# Specified sizes
torch.split(t, [2, 3, 5])     # (tensor([0,1]), tensor([2,3,4]), tensor([5,6,7,8,9]))

# 2D splitting
t = torch.randn(4, 6)
torch.split(t, 2, dim=0)      # Two (2, 6) tensors
torch.split(t, 3, dim=1)      # Two (4, 3) tensors
torch.split(t, [1, 2, 3], dim=1)  # (4,1), (4,2), (4,3)
```

### 2.8.4 torch.chunk

Splits a tensor into a specific number of chunks.

```python
torch.chunk(input, chunks, dim=0)
```

**Parameters:**
- `chunks` (int): Number of chunks to return.
- `dim` (int): The dimension along which to split.

**Examples:**
```python
t = torch.arange(10)
torch.chunk(t, 3)              # (tensor([0,1,2,3]), tensor([4,5,6]), tensor([7,8,9]))

t = torch.randn(4, 6)
torch.chunk(t, 2, dim=0)      # Two (2, 6) tensors
torch.chunk(t, 3, dim=1)      # Three (4, 2) tensors

# If the tensor cannot be evenly divided, the last chunk is smaller
t = torch.arange(7)
torch.chunk(t, 3)              # (tensor([0,1,2]), tensor([3,4,5]), tensor([6]))
```

### 2.8.5 torch.unbind

Removes a dimension and returns a tuple of slices along that dimension.

```python
torch.unbind(input, dim=0)
```

**Examples:**
```python
t = torch.arange(12).reshape(3, 4)
torch.unbind(t, dim=0)         # (tensor([0,1,2,3]), tensor([4,5,6,7]), tensor([8,9,10,11]))
torch.unbind(t, dim=1)         # (tensor([0,4,8]), tensor([1,5,9]), tensor([2,6,10]), tensor([3,7,11]))

# Equivalent to list(t) or tuple(t) for dim=0
```

### 2.8.6 torch.tensor_split

Splits a tensor into sub-tensors (more flexible than split).

```python
torch.tensor_split(input, sections_or_indices, dim=0)
```

**Examples:**
```python
t = torch.arange(8)
torch.tensor_split(t, 3)                # Sizes: 3, 3, 2 (evenly divided)
torch.tensor_split(t, [3, 5])           # Sizes: 3, 2, 3 (split at indices)

t = torch.arange(14).reshape(7, 2)
torch.tensor_split(t, 3, dim=0)         # 3 sub-tensors along rows
torch.tensor_split(t, [2, 5], dim=0)    # Split at row 2 and row 5
```

### 2.8.7 Convenience Functions

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# column_stack: Stack 1D tensors as columns
torch.column_stack([a, b])     # shape: (3, 2) - [[1,4],[2,5],[3,6]]

# row_stack / vstack: Stack as rows
torch.row_stack([a, b])        # shape: (2, 3) - [[1,2,3],[4,5,6]]
torch.vstack([a, b])           # Same as row_stack

# hstack: Horizontal concatenation
torch.hstack([a, b])           # tensor([1,2,3,4,5,6]) - 1D: concatenate

# dstack: Depth-wise stacking
torch.dstack([a, b])           # shape: (3, 2) - stack along new last dim

# For 2D tensors:
a = torch.randn(2, 3)
b = torch.randn(2, 3)
torch.hstack([a, b])           # shape: (2, 6) - concat along dim=1
torch.vstack([a, b])           # shape: (4, 3) - concat along dim=0
torch.dstack([a, b])           # shape: (2, 3, 2) - stack along dim=2
```

### 2.8.8 torch.dsplit / torch.hsplit / torch.vsplit

```python
t = torch.arange(24).reshape(2, 3, 4)

torch.dsplit(t, 2)             # Split along depth (dim 2): 2 tensors of (2, 3, 2)
torch.hsplit(t, 3)             # Split along height (dim 1): 3 tensors of (2, 1, 4)
torch.vsplit(t, 2)             # Split along vertical (dim 0): 2 tensors of (1, 3, 4)
```

---

## 2.9 Other Useful Tensor Operations

### 2.9.1 torch.flip / torch.fliplr / torch.flipud

```python
t = torch.arange(8).reshape(2, 4)
torch.flip(t, [0])              # Flip along dim 0
torch.flip(t, [1])              # Flip along dim 1
torch.flip(t, [0, 1])          # Flip both dims
torch.fliplr(t)                 # Flip left-right (alias for flip dim=1)
torch.flipud(t)                 # Flip up-down (alias for flip dim=0)
```

### 2.9.2 torch.rot90

```python
t = torch.arange(8).reshape(2, 4)
torch.rot90(t, 1, [0, 1])      # Rotate 90 degrees
torch.rot90(t, 2, [0, 1])      # Rotate 180 degrees
torch.rot90(t, 3, [0, 1])      # Rotate 270 degrees
torch.rot90(t, -1, [0, 1])     # Rotate -90 degrees (same as 270)
```

### 2.9.3 torch.roll

```python
t = torch.arange(5)
torch.roll(t, 2)                # tensor([3, 4, 0, 1, 2])
torch.roll(t, -2)               # tensor([2, 3, 4, 0, 1])

t = torch.arange(6).reshape(2, 3)
torch.roll(t, 1)                # Roll all elements: tensor([[5, 0, 1], [2, 3, 4]])
torch.roll(t, 1, dims=1)        # Roll along columns: tensor([[2, 0, 1], [5, 3, 4]])
```

### 2.9.4 torch.diag / torch.diag_embed / torch.diagflat

```python
# torch.diag: Extract diagonal or create diagonal matrix
t = torch.randn(3, 3)
torch.diag(t)                   # Main diagonal: shape (3,)
torch.diag(t, diagonal=1)      # Super-diagonal: shape (2,)
torch.diag(t, diagonal=-1)     # Sub-diagonal: shape (2,)

# Create diagonal matrix from 1D tensor
torch.diag(torch.tensor([1, 2, 3]))
# tensor([[1, 0, 0],
#         [0, 2, 0],
#         [0, 0, 3]])

# torch.diag_embed: Create n-dim diagonal tensor
torch.diag_embed(torch.tensor([1, 2, 3]))  # 3x3 diagonal matrix
torch.diag_embed(torch.tensor([[1, 2], [3, 4]]))  # 2x2x2 diagonal tensor

# torch.diagflat: Create diagonal matrix from flattened input
torch.diagflat(torch.tensor([1, 2]))  # 2x2 diagonal matrix
torch.diagflat(torch.tensor([[1, 2], [3, 4]]))  # 4x4 diagonal (flattened)
```

### 2.9.5 torch.triu / torch.tril

```python
t = torch.randn(3, 3)
torch.triu(t)                   # Upper triangle (including diagonal)
torch.triu(t, diagonal=1)      # Upper triangle (excluding diagonal)
torch.triu(t, diagonal=-1)     # Upper triangle (including one sub-diagonal)

torch.tril(t)                   # Lower triangle (including diagonal)
torch.tril(t, diagonal=-1)     # Lower triangle (excluding diagonal)
torch.tril(t, diagonal=1)      # Lower triangle (including one super-diagonal)
```

### 2.9.6 torch.triu_indices / torch.tril_indices

```python
# Get indices for triangular regions
torch.triu_indices(3, 3, offset=0)     # Indices of upper triangle of 3x3
torch.tril_indices(3, 3, offset=-1)    # Indices of lower triangle, excluding diagonal
```

### 2.9.7 torch.broadcast_shapes / torch.broadcast_tensors

```python
# Compute broadcast shape without actually broadcasting
torch.broadcast_shapes((1, 3), (2, 1))  # torch.Size([2, 3])
torch.broadcast_shapes((3,), (2, 1))    # torch.Size([2, 3])

# Broadcast multiple tensors together
a = torch.randn(1, 3)
b = torch.randn(2, 1)
a_expanded, b_expanded = torch.broadcast_tensors(a, b)
# Both are shape (2, 3)
```

### 2.9.8 torch.squeeze / torch.unsqueeze (function forms)

```python
t = torch.randn(1, 3, 1)

torch.squeeze(t)                    # shape: (3,) - remove all size-1 dims
torch.squeeze(t, 0)                 # shape: (3, 1)
torch.squeeze(t, 2)                 # shape: (1, 3)

torch.unsqueeze(t, 0)               # shape: (1, 1, 3, 1)
torch.unsqueeze(t, 2)               # shape: (1, 3, 1, 1)
```

### 2.9.9 Utility Check Functions

```python
torch.is_tensor(obj)               # Check if obj is a PyTorch tensor
torch.is_complex(input)            # Check if tensor has complex dtype
torch.is_floating_point(input)     # Check if tensor has floating-point dtype
torch.is_nonzero(input)            # Check if single-element tensor is nonzero
torch.is_inference(input)          # Check if tensor was created in inference mode
torch.is_storage(obj)              # Check if obj is a Storage

# Size and shape checks
torch.numel(input)                 # Total number of elements
torch.dim(input)                   # Number of dimensions
```

---

## 2.10 Summary of Key Operations by Category

| Category | Operations |
|----------|-----------|
| **Creation** | `tensor`, `zeros`, `ones`, `empty`, `full`, `rand`, `randn`, `randint`, `arange`, `linspace`, `logspace`, `eye`, `randperm` |
| **Like-variants** | `zeros_like`, `ones_like`, `empty_like`, `full_like`, `randn_like`, `randint_like`, `rand_like` |
| **From data** | `as_tensor`, `from_numpy`, `asarray`, `frombuffer` |
| **Indexing** | `[]`, `where`, `index_select`, `masked_select`, `gather`, `scatter`, `scatter_add`, `index_add`, `index_copy`, `index_fill`, `index_put`, `take`, `take_along_dim`, `nonzero` |
| **Reshaping** | `view`, `reshape`, `permute`, `transpose`, `flatten`, `squeeze`, `unsqueeze`, `expand`, `repeat`, `unflatten`, `movedim` |
| **Combining/Splitting** | `cat`, `stack`, `split`, `chunk`, `unbind`, `tensor_split`, `hstack`, `vstack`, `dstack`, `column_stack` |
| **Properties** | `shape`, `size`, `dim`, `ndim`, `numel`, `dtype`, `device`, `layout`, `stride`, `is_contiguous`, `requires_grad`, `element_size` |
| **Type Cast** | `to`, `float`, `double`, `half`, `bfloat16`, `int`, `long`, `short`, `char`, `byte`, `bool`, `type` |
| **Copy** | `clone`, `detach`, `copy_`, `contiguous` |
| **In-place** | `add_`, `sub_`, `mul_`, `div_`, `fill_`, `zero_`, `relu_`, `normal_`, `uniform_`, `scatter_`, etc. |
| **Flip/Rotate** | `flip`, `fliplr`, `flipud`, `rot90`, `roll` |
| **Diagonal** | `diag`, `diag_embed`, `diagflat`, `triu`, `tril` |
