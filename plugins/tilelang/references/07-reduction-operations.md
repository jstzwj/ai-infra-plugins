# TileLang Reduction Operations Reference

This reference covers all reduction operations available in TileLang, including cross-thread reductions, warp-level reductions, cumulative operations, and reducer finalization. Reductions are fundamental building blocks for patterns such as softmax, layer normalization, loss computation, and many other neural network operations.

---

## Table of Contents

1. [Overview](#overview)
2. [T.reduce -- Core Reduction](#treduce----core-reduction)
3. [T.reduce_max](#treduce_max)
4. [T.reduce_min](#treduce_min)
5. [T.reduce_sum](#treduce_sum)
6. [T.reduce_abssum](#treduce_abssum)
7. [T.reduce_absmax](#treduce_absmax)
8. [Bitwise Reductions](#bitwise-reductions)
9. [T.cumsum -- Cumulative Sum](#tcumsum----cumulative-sum)
10. [T.finalize_reducer](#tfinalize_reducer)
11. [Warp-level Reductions](#warp-level-reductions)
12. [Reduction Strategies](#reduction-strategies)
13. [Batch Reductions](#batch-reductions)
14. [NaN Propagation Behavior](#nan-propagation-behavior)
15. [Practical Examples](#practical-examples)

---

## Overview

Reduction operations in TileLang combine multiple elements along a specified dimension to produce a single value (or a smaller tensor). They are used extensively in:

- **Normalization**: Computing mean, variance, max, and sum for layer norms.
- **Softmax**: Finding max values for numerical stability, summing exponentials.
- **Loss functions**: Computing cross-entropy loss, sum of squared errors.
- **Pooling**: Global average pooling, max pooling.
- **Metrics**: Computing accuracy, perplexity.

TileLang reductions operate on buffer objects and support flexible dimension selection, batch processing, and NaN-aware behavior. All reductions are designed to map efficiently to GPU hardware, using warp shuffles, shared memory, and thread-group cooperation as appropriate.

### Reduction Type Overview

| Reduction | Operation | Identity Element | NaN Propagation |
|-----------|-----------|-----------------|----------------|
| `reduce_sum` | `out = sum(x)` | 0 | N/A (0 + NaN = NaN) |
| `reduce_max` | `out = max(x)` | `-inf` | Configurable |
| `reduce_min` | `out = min(x)` | `+inf` | Configurable |
| `reduce_abssum` | `out = sum(abs(x))` | 0 | N/A |
| `reduce_absmax` | `out = max(abs(x))` | 0 | Configurable |
| `reduce_bitand` | `out = AND(x)` | `~0` (all 1s) | N/A |
| `reduce_bitor` | `out = OR(x)` | 0 | N/A |
| `reduce_bitxor` | `out = XOR(x)` | 0 | N/A |

---

## T.reduce -- Core Reduction

### Signature

```python
T.reduce(
    buffer,                 # Input buffer to reduce
    out,                    # Output buffer for the result
    reduce_type,            # Type of reduction: "sum", "max", "min", etc.
    dim=-1,                 # Dimension along which to reduce
    clear=True,             # Whether to clear the output before reducing
    batch=1,                # Number of independent batch reductions
    nan_propagate=False,    # Whether to propagate NaN values
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `buffer` | Buffer | required | Input buffer to reduce. Can be a shared memory buffer, local (register) buffer, or global buffer. |
| `out` | Buffer | required | Output buffer for the reduction result. The shape should be the input shape with the `dim` dimension removed (or size 1 in that dimension). |
| `reduce_type` | str | required | The reduction operation. One of `"sum"`, `"max"`, `"min"`, `"abssum"`, `"absmax"`, `"bitand"`, `"bitor"`, `"bitxor"`. |
| `dim` | int | `-1` | The dimension along which to reduce. `-1` means the last dimension. Negative values index from the last dimension. |
| `clear` | bool | `True` | If `True`, the output buffer is initialized to the identity element before reduction. If `False`, the reduction result is combined with existing values in the output. |
| `batch` | int | `1` | The number of independent batch reductions. When `batch > 1`, the input is treated as `[batch, ...]` and each batch is reduced independently. |
| `nan_propagate` | bool | `False` | If `True`, NaN values in the input will propagate to the output. If `False`, NaN values are skipped (treated as not participating in the reduction). |

### Supported Reduce Types

| Reduce Type | String | Identity Element | Description |
|-------------|--------|-----------------|-------------|
| Sum | `"sum"` | 0 | Sum of all elements |
| Max | `"max"` | `-inf` | Maximum element |
| Min | `"min"` | `+inf` | Minimum element |
| Absolute Sum | `"abssum"` | 0 | Sum of absolute values |
| Absolute Max | `"absmax"` | 0 | Maximum absolute value |
| Bitwise AND | `"bitand"` | `~0` (all 1s) | Bitwise AND of all elements |
| Bitwise OR | `"bitor"` | 0 | Bitwise OR of all elements |
| Bitwise XOR | `"bitxor"` | 0 | Bitwise XOR of all elements |

### Dimension Specification

The `dim` parameter specifies the dimension along which the reduction is performed:

```python
# For a buffer of shape [M, N]:

# Reduce along last dimension (dim=-1 or dim=1): result shape [M]
T.reduce(buffer_2d, out_1d, "sum", dim=-1)

# Reduce along first dimension (dim=0): result shape [N]
T.reduce(buffer_2d, out_1d, "sum", dim=0)

# For a buffer of shape [B, M, N]:

# Reduce along last dimension: result shape [B, M]
T.reduce(buffer_3d, out_2d, "sum", dim=-1)

# Reduce along middle dimension: result shape [B, N]
T.reduce(buffer_3d, out_2d, "sum", dim=1)
```

### Clear Behavior

The `clear` parameter controls whether the output is initialized before reduction:

```python
# With clear=True (default): out = reduce(buffer)
T.reduce(buffer, out, "sum", clear=True)

# With clear=False: out = existing_out + reduce(buffer)
# This is useful for accumulating partial reductions:
T.reduce(batch_0, out, "sum", clear=True)   # out = sum(batch_0)
T.reduce(batch_1, out, "sum", clear=False)   # out = sum(batch_0) + sum(batch_1)
```

### Example: Generic Reduction

```python
import tilelang
import tilelang.language as T

# Sum reduction along last dimension
@tilelang.jit(out_idx=[1])
def sum_reduce(M, N, dtype="float32"):
    A = T.alloc_shared([M, N], dtype)
    out = T.alloc_shared([M], dtype)

    T.copy(A_global, A)
    T.reduce(A, out, "sum", dim=-1, clear=True)
    T.copy(out, out_global)

    return out_global
```

---

## T.reduce_max

### Signature

```python
T.reduce_max(
    buffer,                 # Input buffer
    out,                    # Output buffer
    dim=-1,                 # Reduction dimension
    clear=True,             # Clear output before reduction
    batch=1,                # Number of batch reductions
    nan_propagate=False,    # Propagate NaN values
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `buffer` | Buffer | required | Input buffer. |
| `out` | Buffer | required | Output buffer for max values. |
| `dim` | int | `-1` | Dimension to reduce along. |
| `clear` | bool | `True` | Initialize output to `-inf` before reducing. |
| `batch` | int | `1` | Number of independent batches. |
| `nan_propagate` | bool | `False` | If `True`, any NaN in the input causes the output to be NaN. |

### Behavior

Computes the maximum value along the specified dimension:

```
out[i] = max(buffer[i, :])    # When reducing along dim=-1
```

When `nan_propagate=True`, the result is NaN if any element in the reduction window is NaN. When `nan_propagate=False`, NaN values are ignored and only non-NaN values participate in the max operation.

### Example: Max Reduction for Softmax Stability

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[1])
def softmax_max(M, N, dtype="float16"):
    # Compute the maximum value in each row for numerical stability
    x = T.alloc_shared([M, N], dtype)
    x_max = T.alloc_shared([M], "float32")

    T.copy(x_global, x)
    T.reduce_max(x, x_max, dim=-1, clear=True)
    T.copy(x_max, out_global)

    return out_global
```

### Example: NaN-aware Max Reduction

```python
# Propagate NaN: if any element is NaN, the result is NaN
T.reduce_max(buffer, out, dim=-1, nan_propagate=True)

# Ignore NaN: NaN values are skipped, result is max of non-NaN values
T.reduce_max(buffer, out, dim=-1, nan_propagate=False)
```

---

## T.reduce_min

### Signature

```python
T.reduce_min(
    buffer,                 # Input buffer
    out,                    # Output buffer
    dim=-1,                 # Reduction dimension
    clear=True,             # Clear output before reduction
    batch=1,                # Number of batch reductions
    nan_propagate=False,    # Propagate NaN values
)
```

### Parameters

Identical to `T.reduce_max` except the identity element is `+inf` instead of `-inf`.

### Behavior

Computes the minimum value along the specified dimension:

```
out[i] = min(buffer[i, :])    # When reducing along dim=-1
```

### Example

```python
# Find minimum value in each column
T.reduce_min(buffer, out, dim=0, clear=True)

# Find minimum value in each row
T.reduce_min(buffer, out, dim=-1, clear=True)
```

---

## T.reduce_sum

### Signature

```python
T.reduce_sum(
    buffer,                 # Input buffer
    out,                    # Output buffer
    dim=-1,                 # Reduction dimension
    clear=True,             # Clear output before reduction
    batch=1,                # Number of batch reductions
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `buffer` | Buffer | required | Input buffer. |
| `out` | Buffer | required | Output buffer for sum values. |
| `dim` | int | `-1` | Dimension to reduce along. |
| `clear` | bool | `True` | Initialize output to 0 before reducing. |
| `batch` | int | `1` | Number of independent batches. |

### Behavior

Computes the sum of all elements along the specified dimension:

```
out[i] = sum(buffer[i, :])    # When reducing along dim=-1
```

The identity element for sum is 0. NaN values in the input will result in NaN output (0 + NaN = NaN).

### Example: Row-wise Sum

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[1])
def row_sum(M, N, dtype="float16"):
    A = T.alloc_shared([M, N], dtype)
    row_sums = T.alloc_shared([M], "float32")

    T.copy(A_global, A)
    T.reduce_sum(A, row_sums, dim=-1, clear=True)
    T.copy(row_sums, out_global)

    return out_global
```

### Example: Column-wise Sum

```python
# Sum along the first dimension (rows) to get column sums
T.reduce_sum(A, col_sums, dim=0, clear=True)
```

### Example: Accumulating Partial Sums

```python
# Clear on first batch, accumulate on subsequent batches
T.reduce_sum(batch_0, partial_sum, dim=-1, clear=True)
T.reduce_sum(batch_1, partial_sum, dim=-1, clear=False)
T.reduce_sum(batch_2, partial_sum, dim=-1, clear=False)
# partial_sum now contains sum of all three batches
```

---

## T.reduce_abssum

### Signature

```python
T.reduce_abssum(
    buffer,                 # Input buffer
    out,                    # Output buffer
    dim=-1,                 # Reduction dimension
    batch=1,                # Number of batch reductions
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `buffer` | Buffer | required | Input buffer. |
| `out` | Buffer | required | Output buffer. |
| `dim` | int | `-1` | Dimension to reduce along. |
| `batch` | int | `1` | Number of independent batches. |

### Behavior

Computes the sum of absolute values along the specified dimension:

```
out[i] = sum(|buffer[i, j]|)    # When reducing along dim=-1
```

This is also known as the L1 norm. The identity element is 0.

### Example: L1 Norm Computation

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[1])
def l1_norm(M, N, dtype="float16"):
    A = T.alloc_shared([M, N], dtype)
    l1 = T.alloc_shared([M], "float32")

    T.copy(A_global, A)
    T.reduce_abssum(A, l1, dim=-1, batch=1)
    T.copy(l1, out_global)

    return out_global
```

---

## T.reduce_absmax

### Signature

```python
T.reduce_absmax(
    buffer,                 # Input buffer
    out,                    # Output buffer
    dim=-1,                 # Reduction dimension
    clear=True,             # Clear output before reduction
    batch=1,                # Number of batch reductions
    nan_propagate=False,    # Propagate NaN values
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `buffer` | Buffer | required | Input buffer. |
| `out` | Buffer | required | Output buffer for max absolute values. |
| `dim` | int | `-1` | Dimension to reduce along. |
| `clear` | bool | `True` | Initialize output to 0 before reducing. |
| `batch` | int | `1` | Number of independent batches. |
| `nan_propagate` | bool | `False` | If `True`, NaN propagates to output. |

### Behavior

Computes the maximum absolute value along the specified dimension:

```
out[i] = max(|buffer[i, j]|)    # When reducing along dim=-1
```

This is also known as the infinity norm (L-infinity norm in a transposed sense). The identity element is 0.

### Example: Max Absolute Value per Row

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[1])
def abs_max_reduce(M, N, dtype="float16"):
    A = T.alloc_shared([M, N], dtype)
    absmax = T.alloc_shared([M], "float32")

    T.copy(A_global, A)
    T.reduce_absmax(A, absmax, dim=-1, clear=True)
    T.copy(absmax, out_global)

    return out_global
```

---

## Bitwise Reductions

### T.reduce_bitand

```python
T.reduce_bitand(
    buffer,                 # Input buffer (integer types)
    out,                    # Output buffer
    dim=-1,                 # Reduction dimension
    clear=True,             # Clear before reduction
    batch=1,                # Number of batch reductions
)
```

Computes the bitwise AND of all elements along the specified dimension:

```
out[i] = buffer[i, 0] & buffer[i, 1] & ... & buffer[i, N-1]
```

The identity element is `~0` (all bits set to 1). Useful for determining if all elements satisfy a bitwise condition.

### T.reduce_bitor

```python
T.reduce_bitor(
    buffer,                 # Input buffer (integer types)
    out,                    # Output buffer
    dim=-1,                 # Reduction dimension
    clear=True,             # Clear before reduction
    batch=1,                # Number of batch reductions
)
```

Computes the bitwise OR of all elements along the specified dimension:

```
out[i] = buffer[i, 0] | buffer[i, 1] | ... | buffer[i, N-1]
```

The identity element is 0. Useful for determining if any element satisfies a bitwise condition.

### T.reduce_bitxor

```python
T.reduce_bitxor(
    buffer,                 # Input buffer (integer types)
    out,                    # Output buffer
    dim=-1,                 # Reduction dimension
    clear=True,             # Clear before reduction
    batch=1,                # Number of batch reductions
)
```

Computes the bitwise XOR of all elements along the specified dimension:

```
out[i] = buffer[i, 0] ^ buffer[i, 1] ^ ... ^ buffer[i, N-1]
```

The identity element is 0. Useful for parity checks and certain hashing operations.

### Example: Bitwise Reductions

```python
import tilelang
import tilelang.language as T

# Bitwise AND: find common bits across all elements
T.reduce_bitand(flags_buffer, common_flags, dim=-1)

# Bitwise OR: find any set bit across all elements
T.reduce_bitor(flags_buffer, any_flags, dim=-1)

# Bitwise XOR: compute parity
T.reduce_bitxor(data_buffer, parity, dim=-1)
```

---

## T.cumsum -- Cumulative Sum

### Signature

```python
T.cumsum(
    src,                    # Source input buffer
    dst=None,               # Destination output buffer (None = in-place)
    dim=0,                  # Dimension for cumulative sum
    reverse=False,          # If True, compute reverse cumulative sum
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `src` | Buffer | required | Input buffer. |
| `dst` | Buffer or None | `None` | Output buffer. If `None`, the cumulative sum is computed in-place into `src`. |
| `dim` | int | `0` | Dimension along which to compute the cumulative sum. |
| `reverse` | bool | `False` | If `True`, compute the cumulative sum from the end to the beginning (reverse scan). |

### Behavior

Computes the cumulative (prefix) sum along the specified dimension:

```
# Forward cumsum (reverse=False):
dst[i] = sum(src[0:i+1])    # For each i along the specified dimension

# Reverse cumsum (reverse=True):
dst[i] = sum(src[i:end])    # For each i along the specified dimension
```

### Example: Forward Cumulative Sum

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[1])
def cumsum_example(N, dtype="float32"):
    src = T.alloc_shared([N], dtype)
    dst = T.alloc_shared([N], dtype)

    T.copy(src_global, src)
    T.cumsum(src, dst, dim=0, reverse=False)
    T.copy(dst, out_global)

    return out_global
```

### Example: Reverse Cumulative Sum

```python
# Reverse cumsum: dst[i] = sum(src[i], src[i+1], ..., src[N-1])
T.cumsum(src, dst, dim=0, reverse=True)
```

### Example: In-place Cumulative Sum

```python
# When dst=None, the result is stored back in src
T.cumsum(src, dim=0, reverse=False)
# src now contains the cumulative sum
```

### Example: Multi-dimensional Cumulative Sum

```python
# For a buffer of shape [M, N]:
# Cumsum along rows (dim=0): each column gets cumsummed independently
T.cumsum(src_2d, dst_2d, dim=0)

# Cumsum along columns (dim=1): each row gets cumsummed independently
T.cumsum(src_2d, dst_2d, dim=1)
```

### Implementation Details

The cumulative sum operation uses an efficient parallel scan (prefix sum) algorithm:

1. **Thread-level scan**: Each thread computes a local prefix sum over its assigned elements.
2. **Block-level scan**: A tree-based scan across thread groups combines local partial sums.
3. **Final add**: Each thread adds the group prefix to its local elements.

For a buffer of size N, the algorithm runs in O(log N) steps with O(N) total work, achieving near-optimal throughput on GPU hardware.

---

## T.finalize_reducer

### Signature

```python
T.finalize_reducer(
    reducer,                # The reducer object to finalize
    batch=1,                # Number of batch elements
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reducer` | Reducer | required | The reducer object created by a reduction operation. Contains intermediate state. |
| `batch` | int | `1` | Number of batch elements in the reducer. |

### Overview

`T.finalize_reducer` completes a multi-stage reduction by performing the final combination step. When reductions are performed in a hierarchical manner (e.g., per-warp, then per-block, then cross-block), the intermediate results are stored in a reducer object. `T.finalize_reducer` combines these intermediate results into the final output.

### When to Use

`T.finalize_reducer` is typically used in scenarios where:

1. The reduction is split across multiple kernel launches.
2. The reduction requires cross-thread-block synchronization (which uses atomic operations or global memory).
3. A partial reduction was performed earlier, and the final combination needs to happen after other operations.

### Example: Finalizing a Multi-Stage Reduction

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[1])
def multi_stage_reduce(M, N, dtype="float32"):
    # Stage 1: Per-row partial reduction within the thread block
    A_smem = T.alloc_shared([M, N], dtype)
    partial = T.alloc_shared([M], dtype)

    T.copy(A_global, A_smem)

    # Create a reducer for row-wise sum
    reducer = T.reduce_sum(A_smem, partial, dim=-1)

    # ... perform other operations ...

    # Stage 2: Finalize the reduction
    T.finalize_reducer(reducer, batch=M)

    T.copy(partial, out_global)
    return out_global
```

---

## Warp-level Reductions

Warp-level reductions operate within a single warp (32 threads) and are the fastest available reduction primitives. They use hardware warp shuffle instructions (`__shfl_down_sync`) to exchange data between threads without using shared memory.

### T.warp_reduce_sum

```python
result = T.warp_reduce_sum(value)
```

Computes the sum of `value` across all 32 threads in the warp. Each thread provides one value.

**Implementation**: Uses a butterfly reduction pattern with `__shfl_down_sync`:

```
Step 0: Thread i gets value from thread i+16, adds to local
Step 1: Thread i gets value from thread i+8, adds to local
Step 2: Thread i gets value from thread i+4, adds to local
Step 3: Thread i gets value from thread i+2, adds to local
Step 4: Thread i gets value from thread i+1, adds to local
Result: Thread 0 holds the sum of all 32 values
```

### T.warp_reduce_max

```python
result = T.warp_reduce_max(value)
```

Computes the maximum of `value` across all 32 threads in the warp.

### T.warp_reduce_min

```python
result = T.warp_reduce_min(value)
```

Computes the minimum of `value` across all 32 threads in the warp.

### T.warp_reduce_bitand / T.warp_reduce_bitor

```python
result = T.warp_reduce_bitand(value)  # Bitwise AND across warp
result = T.warp_reduce_bitor(value)   # Bitwise OR across warp
```

### Warp Reduction Properties

| Property | Value |
|----------|-------|
| Latency | ~5 cycles (for float32) |
| Threads involved | 32 (entire warp) |
| Shared memory required | None |
| Synchronization | Implicit (hardware-level) |
| Result location | Thread 0 (lane 0) of the warp |
| Supported types | float16, float32, int32, uint32 |

### Example: Warp-level Softmax

```python
import tilelang
import tilelang.language as T

# Each thread holds one element of a 32-element vector
# Compute softmax using warp reductions
def warp_softmax(value):
    # Step 1: Find max across the warp
    max_val = T.warp_reduce_max(value)

    # Step 2: Subtract max and exponentiate (per thread)
    exp_val = T.exp(value - max_val)

    # Step 3: Sum exponentials across the warp
    sum_exp = T.warp_reduce_sum(exp_val)

    # Step 4: Compute softmax (per thread)
    softmax_val = exp_val / sum_exp

    return softmax_val
```

### Example: Warp Reduction for Dot Product

```python
import tilelang
import tilelang.language as T

# Compute dot product of two 32-element vectors using warp-level operations
def warp_dot_product(a_val, b_val):
    # Each thread multiplies its elements
    product = a_val * b_val

    # Sum products across the warp
    dot = T.warp_reduce_sum(product)

    return dot  # Available in thread 0
```

---

## Reduction Strategies

TileLang supports multiple strategies for performing reductions, each with different performance characteristics. The strategy is automatically selected based on the reduction dimension size and hardware capabilities, but understanding the tradeoffs is important for optimization.

### Sequential Reduction

In sequential reduction, one thread (or a small group of threads) iterates through all elements to be reduced. This is simple but does not utilize the full parallelism of the GPU.

```
For reducing [N] elements:
- Thread 0 reads all N elements sequentially
- Performs N-1 reduction operations
- Latency: O(N)
- Throughput: O(1) operations per thread
```

**When used**: Very small reduction dimensions (N <= 4).

### Parallel Reduction (Tree-based)

In parallel reduction, all threads participate in a tree-based reduction pattern:

```
Round 1: N/2 threads each reduce 2 elements -> N/2 partial results
Round 2: N/4 threads each reduce 2 partials -> N/4 partial results
...
Final: 1 thread holds the final result

Latency: O(log N)
Total work: O(N)
Active threads per round: N/2, N/4, ..., 1
```

**When used**: Medium to large reduction dimensions (N > 4). This is the default strategy.

### Warp Shuffle Reduction

Uses hardware warp shuffle instructions for reductions within a warp. No shared memory is needed.

```
5 shuffle steps to reduce 32 elements
Latency: O(log WARP_SIZE) = O(5)
```

**When used**: When the reduction dimension exactly matches the warp size, or when reducing across threads in a warp.

### Cross-block Reduction

For reductions that span multiple thread blocks, TileLang uses atomic operations on global memory:

```
1. Each block computes its partial result
2. Each block atomically updates the global result
3. The final result is the combination of all partial results
```

**When used**: Grid-level reductions where the data exceeds a single thread block's capacity.

### Strategy Selection Guide

| Reduction Size | Recommended Strategy | Rationale |
|---------------|---------------------|-----------|
| 1-4 elements | Sequential | Overhead of parallelism exceeds benefit |
| 5-32 elements | Warp shuffle | Fits within a single warp |
| 33-1024 elements | Tree-based (shared memory) | Efficient parallel reduction |
| 1024+ elements | Multi-stage tree | Hierarchical reduction to avoid contention |
| Cross-block | Atomic global memory | Required for grid-level reduction |

---

## Batch Reductions

The `batch` parameter enables independent reductions across multiple groups within a single operation. This is more efficient than launching separate reductions for each group.

### How Batch Reduction Works

When `batch=B`, the input buffer is conceptually divided into `B` independent slices, and each slice is reduced independently:

```
Input shape: [B, N] (conceptually)
Batch reduction along dim=-1 with batch=B:
  out[0] = reduce(buffer[0, :])
  out[1] = reduce(buffer[1, :])
  ...
  out[B-1] = reduce(buffer[B-1, :])
```

### Example: Batch Reduction

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[1])
def batch_reduce(B, N, dtype="float32"):
    # B independent vectors, each of length N
    data = T.alloc_shared([B, N], dtype)
    results = T.alloc_shared([B], dtype)

    T.copy(data_global, data)

    # Reduce each of the B vectors independently
    T.reduce_sum(data, results, dim=-1, clear=True, batch=B)

    T.copy(results, out_global)
    return out_global

# Alternative: reduce along dim=1 with batch handled implicitly
T.reduce_sum(data, results, dim=1, clear=True)
```

### Batch Reduction with Multiple Dimensions

```python
# Input shape: [B, M, N]
# Reduce along last dimension with batch=B
# Output shape: [B, M]
data_3d = T.alloc_shared([B, M, N], dtype)
out_2d = T.alloc_shared([B, M], dtype)
T.reduce_sum(data_3d, out_2d, dim=-1, clear=True, batch=B)
```

### Performance Considerations for Batch Reductions

- Batch reductions are most efficient when the batch count is a multiple of the number of warps in the thread block.
- Each batch element should ideally be processed by a different warp to avoid inter-warp synchronization.
- The reduction dimension should be large enough (>32 elements) to fully utilize the warp's parallel reduction capacity.

---

## NaN Propagation Behavior

NaN (Not a Number) propagation during reductions is a subtle but important consideration for numerical correctness. TileLang provides explicit control over NaN propagation through the `nan_propagate` parameter.

### Default Behavior (nan_propagate=False)

By default, NaN values are **ignored** during reduction:

```python
buffer = [1.0, NaN, 3.0, 2.0]
T.reduce_max(buffer, out, dim=-1, nan_propagate=False)
# Result: 3.0 (NaN is skipped)

T.reduce_sum(buffer, out, dim=-1)
# Result: 6.0 (NaN is treated as 0, since 1 + 0 + 3 + 2 = 6)
```

### NaN Propagation Enabled (nan_propagate=True)

When `nan_propagate=True`, any NaN in the reduction window causes the result to be NaN:

```python
buffer = [1.0, NaN, 3.0, 2.0]
T.reduce_max(buffer, out, dim=-1, nan_propagate=True)
# Result: NaN (NaN present in input)

T.reduce_min(buffer, out, dim=-1, nan_propagate=True)
# Result: NaN (NaN present in input)
```

### NaN Behavior by Reduction Type

| Reduction Type | nan_propagate=False | nan_propagate=True |
|---------------|--------------------|--------------------|
| `reduce_sum` | NaN treated as 0 | NaN propagates (result is NaN) |
| `reduce_max` | NaN skipped, max of non-NaN | Any NaN -> result is NaN |
| `reduce_min` | NaN skipped, min of non-NaN | Any NaN -> result is NaN |
| `reduce_abssum` | NaN treated as 0 | NaN propagates |
| `reduce_absmax` | NaN skipped | Any NaN -> result is NaN |
| `reduce_bitand/bitor/xor` | N/A (integer types) | N/A |

### Hardware Implementation

On NVIDIA GPUs, NaN propagation is controlled at the instruction level:

- **max/min with NaN propagation**: Uses `max.NaN.ftz.f32` / `min.NaN.ftz.f32` PTX instructions.
- **max/min without NaN propagation**: Uses standard `max.ftz.f32` / `min.ftz.f32` instructions.

The performance difference between the two modes is negligible in most cases.

---

## Practical Examples

### Complete Softmax Using Reductions

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[1])
def softmax_kernel(M, N, block_M=64, dtype="float16"):
    # Allocate buffers
    x_smem = T.alloc_shared([block_M, N], dtype)
    x_max = T.alloc_shared([block_M], "float32")
    exp_sum = T.alloc_shared([block_M], "float32")
    result = T.alloc_shared([block_M, N], dtype)

    # Load input tile
    T.copy(x_global, x_smem)

    # Step 1: Find max of each row (numerical stability)
    T.reduce_max(x_smem, x_max, dim=-1, clear=True)

    # Step 2: Subtract max and exponentiate
    for i in range(block_M):
        for j in range(N):
            result[i, j] = T.exp(x_smem[i, j] - x_max[i])

    # Step 3: Sum exponentials
    T.reduce_sum(result, exp_sum, dim=-1, clear=True)

    # Step 4: Normalize
    for i in range(block_M):
        for j in range(N):
            result[i, j] = result[i, j] / exp_sum[i]

    T.copy(result, out_global)
    return out_global
```

### Layer Normalization with Reductions

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[1])
def layernorm_kernel(M, N, block_M=64, dtype="float16", eps=1e-5):
    x_smem = T.alloc_shared([block_M, N], dtype)
    mean = T.alloc_shared([block_M], "float32")
    var = T.alloc_shared([block_M], "float32")
    out_smem = T.alloc_shared([block_M, N], dtype)

    T.copy(x_global, x_smem)

    # Step 1: Compute mean = sum(x) / N
    T.reduce_sum(x_smem, mean, dim=-1, clear=True)
    for i in range(block_M):
        mean[i] = mean[i] / N

    # Step 2: Compute variance = sum((x - mean)^2) / N
    for i in range(block_M):
        for j in range(N):
            diff = x_smem[i, j] - mean[i]
            out_smem[i, j] = diff * diff

    T.reduce_sum(out_smem, var, dim=-1, clear=True)
    for i in range(block_M):
        var[i] = var[i] / N

    # Step 3: Normalize: (x - mean) / sqrt(var + eps)
    for i in range(block_M):
        inv_std = T.rsqrt(var[i] + eps)
        for j in range(N):
            out_smem[i, j] = (x_smem[i, j] - mean[i]) * inv_std

    # Step 4: Apply affine transform (gamma * x + beta)
    for i in range(block_M):
        for j in range(N):
            out_smem[i, j] = out_smem[i, j] * gamma[j] + beta[j]

    T.copy(out_smem, out_global)
    return out_global
```

### RMS Normalization

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[1])
def rmsnorm_kernel(M, N, block_M=64, dtype="float16", eps=1e-5):
    x_smem = T.alloc_shared([block_M, N], dtype)
    x_sq_smem = T.alloc_shared([block_M, N], "float32")
    mean_sq = T.alloc_shared([block_M], "float32")
    out_smem = T.alloc_shared([block_M, N], dtype)

    T.copy(x_global, x_smem)

    # Compute x^2
    for i in range(block_M):
        for j in range(N):
            x_sq_smem[i, j] = x_smem[i, j] * x_smem[i, j]

    # Compute mean of x^2
    T.reduce_sum(x_sq_smem, mean_sq, dim=-1, clear=True)
    for i in range(block_M):
        mean_sq[i] = mean_sq[i] / N

    # Normalize: x * rsqrt(mean(x^2) + eps)
    for i in range(block_M):
        inv_rms = T.rsqrt(mean_sq[i] + eps)
        for j in range(N):
            out_smem[i, j] = x_smem[i, j] * inv_rms * gamma[j]

    T.copy(out_smem, out_global)
    return out_global
```

### Cross-Entropy Loss with Reductions

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[0])
def cross_entropy_loss(N, V, block_N=64, dtype="float16"):
    # Logits: [N, V], Targets: [N]
    logits = T.alloc_shared([block_N, V], dtype)
    log_sum_exp = T.alloc_shared([block_N], "float32")
    target_vals = T.alloc_shared([block_N], "float32")
    loss = T.alloc_local([1], "float32")

    T.copy(logits_global, logits)

    # Step 1: Find max for numerical stability
    max_logits = T.alloc_shared([block_N], "float32")
    T.reduce_max(logits, max_logits, dim=-1, clear=True)

    # Step 2: Compute log-sum-exp
    shifted = T.alloc_shared([block_N, V], "float32")
    for i in range(block_N):
        for j in range(V):
            shifted[i, j] = T.exp(logits[i, j] - max_logits[i])

    T.reduce_sum(shifted, log_sum_exp, dim=-1, clear=True)
    for i in range(block_N):
        log_sum_exp[i] = max_logits[i] + T.log(log_sum_exp[i])

    # Step 3: Compute loss = -logits[target] + log_sum_exp
    total_loss = T.alloc_local([1], "float32")
    T.clear(total_loss)
    for i in range(block_N):
        total_loss[0] = total_loss[0] + (-logits[i, targets[i]] + log_sum_exp[i])

    total_loss[0] = total_loss[0] / block_N
    T.copy(total_loss, out_global)
    return out_global
```

### Online Softmax with Running Max and Sum

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[1])
def online_softmax(M, N, block_M=64, block_N=64, dtype="float16"):
    """
    Online softmax using the online algorithm:
    For each new block of columns:
      1. Compute local max
      2. Update global max
      3. Compute correction factor
      4. Update running sum
    """
    x_smem = T.alloc_shared([block_M, block_N], dtype)
    out = T.alloc_local([block_M], "float32")      # Running output
    m_running = T.alloc_local([block_M], "float32")  # Running max
    l_running = T.alloc_local([block_M], "float32")  # Running sum of exp

    T.clear(m_running)
    T.clear(l_running)
    T.clear(out)

    for n_start in range(0, N, block_N):
        # Load current block
        T.copy(x_global[:, n_start:n_start+block_N], x_smem)

        # Compute local max
        local_max = T.alloc_shared([block_M], "float32")
        T.reduce_max(x_smem, local_max, dim=-1, clear=True)

        # Update running max and compute correction
        new_max = T.max(m_running, local_max)
        correction = T.exp(m_running - new_max)
        l_running = l_running * correction
        out = out * correction

        # Compute exp(x - new_max) for current block
        for i in range(block_M):
            for j in range(block_N):
                x_smem[i, j] = T.exp(x_smem[i, j] - new_max[i])

        # Sum of exp for current block
        block_sum = T.alloc_shared([block_M], "float32")
        T.reduce_sum(x_smem, block_sum, dim=-1, clear=True)
        l_running = l_running + block_sum

        # Update output with weighted values
        for i in range(block_M):
            for j in range(block_N):
                out[i] = out[i] + x_smem[i, j]

        m_running = new_max

    # Final normalization
    for i in range(block_M):
        out[i] = out[i] / l_running[i]

    T.copy(out, out_global)
    return out_global
```

### Prefix Sum (Exclusive Scan) Using T.cumsum

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[1])
def exclusive_scan(N, dtype="float32"):
    src = T.alloc_shared([N], dtype)
    dst = T.alloc_shared([N], dtype)

    T.copy(src_global, src)

    # Exclusive scan = shift cumsum right by 1, insert 0 at front
    T.cumsum(src, dst, dim=0, reverse=False)

    # Shift right by 1 to make it exclusive
    for i in range(N - 1, 0, -1):
        dst[i] = dst[i - 1]
    dst[0] = 0.0

    T.copy(dst, out_global)
    return out_global
```

---

## Summary

| Operation | Scope | Shared Memory | Typical Use |
|-----------|-------|--------------|-------------|
| `T.reduce` | Block | Yes | Generic configurable reduction |
| `T.reduce_max` | Block | Yes | Finding maximum values |
| `T.reduce_min` | Block | Yes | Finding minimum values |
| `T.reduce_sum` | Block | Yes | Summing elements |
| `T.reduce_abssum` | Block | Yes | L1 norm computation |
| `T.reduce_absmax` | Block | Yes | Infinity norm computation |
| `T.reduce_bitand/bitor/xor` | Block | Yes | Bitwise operations on integers |
| `T.cumsum` | Block | Yes | Prefix sum, cumulative operations |
| `T.finalize_reducer` | Block | Yes | Multi-stage reduction completion |
| `T.warp_reduce_sum` | Warp | No | Fast per-warp sum |
| `T.warp_reduce_max` | Warp | No | Fast per-warp max |
| `T.warp_reduce_min` | Warp | No | Fast per-warp min |
| `T.warp_reduce_bitand/bitor` | Warp | No | Fast per-warp bitwise |

The reduction primitives in TileLang provide a comprehensive toolkit for all common reduction patterns in GPU computing. From simple element-wise sums to complex online softmax algorithms, these primitives are designed to map efficiently to GPU hardware while providing flexible configuration options for numerical precision, NaN handling, and batch processing.
