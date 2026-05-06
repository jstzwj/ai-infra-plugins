# Chapter 6: Language Standard Module (`triton.language.standard`)

Provides common higher-level tensor operations built on top of core primitives.

## Reduction Operations

### `tl.max(input, axis=None, return_indices=False, keep_dims=False) -> tensor`
Compute maximum along axis.

```python
@triton.jit
def kernel(ptr, out_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    row_data = tl.load(ptr + row * n_cols + offs, mask=mask, other=-float('inf'))
    row_max = tl.max(row_data, axis=0)          # scalar max
    tl.store(out_ptr + row, row_max)

# With indices
max_val, max_idx = tl.max(data, axis=0, return_indices=True)
```

### `tl.min(input, axis=None, keep_dims=False) -> tensor`
Compute minimum along axis. Same signature as `max`.

### `tl.sum(input, axis=None, keep_dims=False, dtype=None) -> tensor`
Sum along axis. For integer types, automatically promotes to int32 or int64.

```python
row_sum = tl.sum(row_data, axis=0)  # Sum all elements
```

### `tl.argmax(input, axis, keep_dims=False) -> tensor`
Return index of maximum value along axis.

### `tl.argmin(input, axis, keep_dims=False) -> tensor`
Return index of minimum value along axis.

### `tl.xor_sum(input, axis=None, keep_dims=False) -> tensor`
XOR reduction (integer types only).

### `tl.reduce_or(input, axis, keep_dims=False) -> tensor`
OR reduction (integer types only).

### `tl.reduce(input, axis, combine_fn, keep_dims=False) -> tensor`
General reduction with custom combine function.

```python
@triton.jit
def product_combine(a, b):
    return a * b

@triton.jit
def kernel(ptr, out, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    data = tl.load(ptr + offs, mask=mask, other=1)
    product = tl.reduce(data, axis=0, combine_fn=product_combine)
    tl.store(out, product)
```

## Cumulative Operations

### `tl.cumsum(input, axis=0, reverse=False, dtype=None) -> tensor`
Cumulative sum (prefix sum).

```python
@triton.jit
def kernel(ptr, out, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    data = tl.load(ptr + offs, mask=mask)
    result = tl.cumsum(data, axis=0)
    tl.store(out + offs, result, mask=mask)
```

### `tl.cumprod(input, axis=0, reverse=False) -> tensor`
Cumulative product.

## Scan Operations

### `tl.associative_scan(input, axis, combine_fn, reverse=False) -> tensor`
General associative scan (prefix scan) with custom combine function.

```python
@triton.jit
def max_combine(a, b):
    return tl.maximum(a, b)

# Running maximum
running_max = tl.associative_scan(data, axis=0, combine_fn=max_combine)
```

## Sorting

### `tl.sort(x, dim=None, descending=False) -> tensor`
Sort tensor using bitonic sort algorithm.

```python
@triton.jit
def kernel(ptr, out, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    data = tl.load(ptr + offs, mask=mask, other=float('inf'))
    sorted_data = tl.sort(data)
    tl.store(out + offs, sorted_data, mask=mask)
```

### `tl.topk(x, k, dim=None, descending=True) -> tuple`
Return top-k elements and their indices.

```python
values, indices = tl.topk(data, k=5)
```

### `tl.bitonic_merge(x, dim=None, descending=False) -> tensor`
Single step of bitonic merge sort. Used internally by `sort`.

## Activation Functions

### `tl.sigmoid(x) -> tensor`
Sigmoid activation: `1 / (1 + exp(-x))`.

```python
output = tl.sigmoid(x)  # x.sigmoid() also works
```

### `tl.softmax(x, dim=None, keep_dims=False, ieee_rounding=False) -> tensor`
Numerically stable softmax.

```python
@triton.jit
def softmax_kernel(ptr, out, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    data = tl.load(ptr + row * n_cols + offs, mask=mask, other=-float('inf'))
    result = tl.softmax(data)
    tl.store(out + row * n_cols + offs, result, mask=mask)
```

## Utility Functions

### `tl.cdiv(x, div) -> constexpr`
Ceiling division: `(x + div - 1) // div`.

```python
num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
```

### `tl.ravel(x, can_reorder=False) -> tensor`
Flatten tensor to 1D contiguous.

### `tl.zeros(shape, dtype) -> tensor`
Create zero-filled tensor.

```python
z = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
```

### `tl.zeros_like(input) -> tensor`
Create zero-filled tensor with same shape and dtype.

### `tl.flip(x, dim=None) -> tensor`
Reverse tensor along dimension.

```python
reversed_data = tl.flip(data)  # x.flip() also works
```

### `tl.interleave(a, b) -> tensor`
Interleave two tensors along last dimension.

### `tl.squeeze(x, dim) -> tensor`
Remove dimension of size 1.

### `tl.unsqueeze(x, dim) -> tensor`
Add dimension of size 1.

### `tl.swizzle2d(i, j, size_i, size_j, size_g) -> tensor`
Swizzle 2D indices for grouped access pattern.

```python
# Used in matrix multiplication for L2 cache optimization
pid_m = pid // (tl.cdiv(N, BN))
pid_n = pid % (tl.cdiv(N, BN))
pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, M // BM, N // BN, GROUP_SIZE)
```

## Member Functions

All standard operations are available as member functions on `tensor`:

| Free Function | Member Function |
|--------------|-----------------|
| `tl.max(x)` | `x.max()` |
| `tl.min(x)` | `x.min()` |
| `tl.sum(x)` | `x.sum()` |
| `tl.argmax(x, axis)` | `x.argmax(axis)` |
| `tl.argmin(x, axis)` | `x.argmin(axis)` |
| `tl.cumsum(x)` | `x.cumsum()` |
| `tl.cumprod(x)` | `x.cumprod()` |
| `tl.sort(x)` | `x.sort()` |
| `tl.softmax(x)` | `x.softmax()` |
| `tl.sigmoid(x)` | `x.sigmoid()` |
| `tl.flip(x)` | `x.flip()` |
| `tl.ravel(x)` | `x.ravel()` |
| `tl.squeeze(x, dim)` | `x.squeeze(dim)` |
| `tl.unsqueeze(x, dim)` | `x.unsqueeze(dim)` |
