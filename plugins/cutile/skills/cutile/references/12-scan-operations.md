# Scan Operations

This chapter covers scan (prefix sum) operations in cuTile. Scan operations compute running totals along a dimension, producing outputs of the same shape as the input. These are fundamental for parallel algorithms, indexing, and cumulative computations.

## Overview

Scan operations, also known as prefix sums, compute cumulative aggregates across a sequence. Unlike reduction operations which produce a smaller output, scan operations maintain the same shape but compute running totals:

- **Cumulative sums**: Running totals (cumsum)
- **Cumulative products**: Running products (cumprod)
- **Custom scans**: User-defined associative operations

Scan semantics:
- **Inclusive scan**: `output[i] = combine(input[0], input[1], ..., input[i])`
- **Exclusive scan**: `output[i] = combine(input[0], input[1], ..., input[i-1])` (output[0] = identity)

All scan operations preserve the input shape and perform computation along a single axis.

## Cumulative Sum Operations

### `ct.cumsum(tile, axis)`

Computes the cumulative sum of elements along a specified axis.

**Signature:**
```python
ct.cumsum(tile: Tile, axis: int) -> Tile
```

**Parameters:**
- `tile`: Input tile
- `axis`: Axis along which to compute cumulative sum

**Returns:**
- Tile of same shape with cumulative sums

**Semantics:**
- Inclusive scan: `output[i] = sum(input[0:i+1])`
- For 1D: `[1, 2, 3, 4] → [1, 3, 6, 10]`

**Examples:**

**1D cumulative sum:**
```python
import cutile as ct

# Create 1D tile
vec = ct.from_list([1.0, 2.0, 3.0, 4.0, 5.0], dtype=ct.float32)

# Cumulative sum
cumsum_vec = ct.cumsum(vec, axis=0)
print(cumsum_vec.shape)  # (5,)

# Result: [1.0, 3.0, 6.0, 10.0, 15.0]
# 1.0 = 1.0
# 3.0 = 1.0 + 2.0
# 6.0 = 1.0 + 2.0 + 3.0
# 10.0 = 1.0 + 2.0 + 3.0 + 4.0
# 15.0 = 1.0 + 2.0 + 3.0 + 4.0 + 5.0
```

**2D cumulative sum along rows:**
```python
# Create matrix
matrix = ct.from_list([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
], dtype=ct.float32)

# Cumulative sum along columns (axis=1)
# Each row is independently cumsummed
row_cumsum = ct.cumsum(matrix, axis=1)
print(row_cumsum.shape)  # (3, 4)

# Result:
# Row 0: [1.0, 3.0, 6.0, 10.0]
# Row 1: [5.0, 11.0, 18.0, 26.0]
# Row 2: [9.0, 19.0, 30.0, 42.0]
```

**2D cumulative sum along columns:**
```python
# Cumulative sum along rows (axis=0)
# Each column is independently cumsummed
col_cumsum = ct.cumsum(matrix, axis=0)
print(col_cumsum.shape)  # (3, 4)

# Result:
# Col 0: [1.0, 6.0, 15.0]   (1, 1+5, 1+5+9)
# Col 1: [2.0, 8.0, 18.0]   (2, 2+6, 2+6+10)
# Col 2: [3.0, 10.0, 21.0]  (3, 3+7, 3+7+11)
# Col 3: [4.0, 12.0, 24.0]  (4, 4+8, 4+8+12)
```

**3D cumulative sum:**
```python
# Create 3D tile: (batch, seq_len, features)
tile = ct.randn((8, 16, 32), dtype=ct.float32)

# Cumulative sum along sequence dimension
seq_cumsum = ct.cumsum(tile, axis=1)
print(seq_cumsum.shape)  # (8, 16, 32)

# Each batch and feature dimension is independently cumsummed
# across the sequence length
```

**Use case - prefix sum for indexing:**
```python
def prefix_sum_indices(counts):
    """
    Convert counts to start indices using prefix sum.
    
    Args:
        counts: (num_groups,) count per group
    
    Returns:
        (num_groups,) start index for each group
    """
    return ct.cumsum(counts, axis=0) - counts

# Example
# Suppose we have 4 groups with counts [3, 5, 2, 4]
counts = ct.from_list([3, 5, 2, 4], dtype=ct.int32)

# Prefix sum (exclusive): [0, 3, 8, 10]
start_indices = prefix_sum_indices(counts)
print(start_indices.shape)  # (4,)

# This gives us:
# Group 0: indices 0, 1, 2
# Group 1: indices 3, 4, 5, 6, 7
# Group 2: indices 8, 9
# Group 3: indices 10, 11, 12, 13
```

**Use case - running total:**
```python
def running_total(values):
    """
    Compute running total of a sequence.
    
    Args:
        values: (seq_len,) sequence of values
    
    Returns:
        (seq_len,) running total at each position
    """
    return ct.cumsum(values, axis=0)

# Example
sales = ct.from_list([100.0, 50.0, 75.0, 125.0, 80.0], dtype=ct.float32)
total_sales = running_total(sales)
print(total_sales.shape)  # (5,)

# Result: [100.0, 150.0, 225.0, 350.0, 430.0]
```

### `ct.cumprod(tile, axis)`

Computes the cumulative product of elements along a specified axis.

**Signature:**
```python
ct.cumprod(tile: Tile, axis: int) -> Tile
```

**Parameters:**
- `tile`: Input tile
- `axis`: Axis along which to compute cumulative product

**Returns:**
- Tile of same shape with cumulative products

**Examples:**

**1D cumulative product:**
```python
# Create 1D tile
vec = ct.from_list([1.0, 2.0, 3.0, 4.0], dtype=ct.float32)

# Cumulative product
cumprod_vec = ct.cumprod(vec, axis=0)
print(cumprod_vec.shape)  # (4,)

# Result: [1.0, 2.0, 6.0, 24.0]
# 1.0 = 1.0
# 2.0 = 1.0 * 2.0
# 6.0 = 1.0 * 2.0 * 3.0
# 24.0 = 1.0 * 2.0 * 3.0 * 4.0
```

**2D cumulative product:**
```python
# Create matrix
matrix = ct.from_list([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
], dtype=ct.float32)

# Cumulative product along columns (axis=1)
row_cumprod = ct.cumprod(matrix, axis=1)
print(row_cumprod.shape)  # (2, 3)

# Result:
# Row 0: [1.0, 2.0, 6.0]
# Row 1: [4.0, 20.0, 120.0]
```

**Use case - compound interest:**
```python
def compound_growth(principal, rates):
    """
    Calculate compound growth over time.
    
    Args:
        principal: initial amount
        rates: (num_periods,) growth rate per period
    
    Returns:
        (num_periods,) value at each period
    """
    # Convert rates to multipliers: 1 + rate
    multipliers = 1.0 + rates
    
    # Cumulative product of multipliers
    growth_factors = ct.cumprod(multipliers, axis=0)
    
    # Apply to principal
    values = principal * growth_factors
    
    return values

# Example
principal = 1000.0
rates = ct.from_list([0.05, 0.03, 0.04, 0.02], dtype=ct.float32)

values = compound_growth(principal, rates)
print(values.shape)  # (4,)

# Year 0: 1000 * 1.05 = 1050
# Year 1: 1050 * 1.03 = 1081.5
# Year 2: 1081.5 * 1.04 = 1124.76
# Year 3: 1124.76 * 1.02 = 1147.26
```

**Use case - factorial sequence:**
```python
def factorial_sequence(n):
    """
    Generate sequence of factorials: [1!, 2!, ..., n!]
    
    Args:
        n: maximum factorial
    
    Returns:
        (n,) factorial values
    """
    # Create sequence [1, 2, 3, ..., n]
    seq = ct.arange(1, n + 1, dtype=ct.float32)
    
    # Cumulative product gives factorials
    factorials = ct.cumprod(seq, axis=0)
    
    return factorials

# Example
factorials = factorial_sequence(10)
print(factorials.shape)  # (10,)

# Result: [1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]
```

## Custom Scan Operations

### `ct.scan(tile, axis, init, fn)`

Performs a custom inclusive prefix scan using a user-defined binary function.

**Signature:**
```python
ct.scan(tile: Tile, axis: int, init: float, fn: callable) -> Tile
```

**Parameters:**
- `tile`: Input tile
- `axis`: Axis along which to scan
- `init`: Initial value for the scan (identity element)
- `fn`: Binary scan function `(a, b) -> result`

**Requirements:**
- `fn` must be associative: `(a op b) op c == a op (b op c)`
- `fn` must handle the input dtype
- `init` should be the identity element for the operation

**Examples:**

**Custom cumulative sum:**
```python
def custom_cumsum(tile, axis):
    """Implement cumsum using scan."""
    init = 0.0
    return ct.scan(tile, axis, init, lambda a, b: a + b)

# Test
vec = ct.from_list([1.0, 2.0, 3.0, 4.0], dtype=ct.float32)
result = custom_cumsum(vec, axis=0)
print(result.shape)  # (4,)
# Result: [1.0, 3.0, 6.0, 10.0]
```

**Custom cumulative max:**
```python
def cumulative_max(tile, axis):
    """Compute cumulative maximum along axis."""
    init = -float('inf')
    return ct.scan(tile, axis, init, lambda a, b: ct.maximum(a, b))

# Example
vec = ct.from_list([3.0, 1.0, 4.0, 1.0, 5.0, 9.0], dtype=ct.float32)
cummax = cumulative_max(vec, axis=0)
print(cummax.shape)  # (6,)

# Result: [3.0, 3.0, 4.0, 4.0, 5.0, 9.0]
```

**Custom cumulative min:**
```python
def cumulative_min(tile, axis):
    """Compute cumulative minimum along axis."""
    init = float('inf')
    return ct.scan(tile, axis, init, lambda a, b: ct.minimum(a, b))

# Example
vec = ct.from_list([3.0, 1.0, 4.0, 1.0, 5.0, 9.0], dtype=ct.float32)
cummin = cumulative_min(vec, axis=0)
print(cummin.shape)  # (6,)

# Result: [3.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

**Custom logical AND:**
```python
def cumulative_all(tile, axis):
    """
    Check if all elements up to current position are non-zero.
    
    Args:
        tile: input tile (treated as boolean)
        axis: scan axis
    
    Returns:
        tile of same shape with cumulative AND
    """
    init = True
    return ct.scan(tile, axis, init, lambda a, b: a and b)

# Example
mask = ct.from_list([1, 1, 0, 1, 1], dtype=ct.int32)
cumall = cumulative_all(mask, axis=0)
print(cumall.shape)  # (5,)

# Result: [True, True, False, False, False]
# Once we hit 0, all subsequent values are False
```

**Custom logical OR:**
```python
def cumulative_any(tile, axis):
    """
    Check if any element up to current position is non-zero.
    
    Args:
        tile: input tile (treated as boolean)
        axis: scan axis
    
    Returns:
        tile of same shape with cumulative OR
    """
    init = False
    return ct.scan(tile, axis, init, lambda a, b: a or b)

# Example
mask = ct.from_list([0, 0, 1, 0, 0], dtype=ct.int32)
cumany = cumulative_any(mask, axis=0)
print(cumany.shape)  # (5,)

# Result: [False, False, True, True, True]
# Once we hit 1, all subsequent values are True
```

## Scan Patterns and Applications

### Pattern 1: Exclusive Scan (Prefix Sum)

Exclusive scan computes the sum of all previous elements but not the current one. This is useful for computing output positions in scatter operations.

```python
def exclusive_scan(tile, axis):
    """
    Compute exclusive prefix sum.
    
    output[i] = sum(input[0:i])  # Note: not including input[i]
    output[0] = 0
    """
    # Compute inclusive scan
    inclusive = ct.cumsum(tile, axis=axis)
    
    # Shift right and set first element to 0
    # This is a simplified version - actual implementation
    # would need proper handling
    return inclusive - tile

# Example
vec = ct.from_list([1, 2, 3, 4, 5], dtype=ct.int32)
exclusive = exclusive_scan(vec, axis=0)
print(exclusive.shape)  # (5,)

# Result: [0, 1, 3, 6, 10]
```

### Pattern 2: Running Average

```python
def running_average(values, window_size):
    """
    Compute running average using cumulative sum.
    
    Args:
        values: (seq_len,) sequence of values
        window_size: size of averaging window
    
    Returns:
        (seq_len - window_size + 1,) running averages
    """
    # Compute cumulative sum
    cumsum = ct.cumsum(values, axis=0)
    
    # Compute sum over windows using difference of cumsum
    # sum[i:i+window] = cumsum[i+window] - cumsum[i]
    window_sums = cumsum[window_size:] - cumsum[:-window_size]
    
    # Divide by window size
    averages = window_sums / window_size
    
    return averages

# Example
values = ct.from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=ct.float32)
avg = running_average(values, window_size=3)
print(avg.shape)  # (6,)

# Result: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
# Window 0: (1+2+3)/3 = 2.0
# Window 1: (2+3+4)/3 = 3.0
# etc.
```

### Pattern 3: Prefix Sum for Parallel Algorithms

```python
def parallel_histogram_counts(values, num_bins):
    """
    Count values per bin using prefix sum.
    
    Args:
        values: (num_values,) values in range [0, num_bins)
        num_bins: number of histogram bins
    
    Returns:
        (num_bins,) count per bin
    """
    # Initialize histogram
    histogram = ct.zeros((num_bins,), dtype=ct.int32)
    
    # For each value, increment corresponding bin
    # (In practice, this would use scatter operations)
    for i in range(values.numel):
        bin_idx = values[i]
        histogram[bin_idx] += 1
    
    return histogram

# Example
values = ct.from_list([0, 2, 1, 2, 0, 1, 2, 0], dtype=ct.int32)
counts = parallel_histogram_counts(values, num_bins=3)
print(counts.shape)  # (3,)

# Result: [3, 2, 3]
# Bin 0: values 0, 0, 0 → 3 occurrences
# Bin 1: values 1, 1 → 2 occurrences
# Bin 2: values 2, 2, 2 → 3 occurrences
```

### Pattern 4: Inclusive vs Exclusive Scan

```python
def demonstrate_scan_types():
    """Show difference between inclusive and exclusive scans."""
    
    # Input sequence
    x = ct.from_list([1, 2, 3, 4, 5], dtype=ct.int32)
    
    # Inclusive scan: output[i] = sum of elements 0 through i
    inclusive = ct.cumsum(x, axis=0)
    # Result: [1, 3, 6, 10, 15]
    
    # Exclusive scan: output[i] = sum of elements 0 through i-1
    # output[0] = 0 (identity for addition)
    exclusive = ct.concatenate([
        ct.zeros((1,), dtype=ct.int32),
        inclusive[:-1]
    ], axis=0)
    # Result: [0, 1, 3, 6, 10]
    
    return inclusive, exclusive

inclusive, exclusive = demonstrate_scan_types()
print(f"Inclusive: {inclusive.shape}")
print(f"Exclusive: {exclusive.shape}")
```

## Complete Examples

### Example 1: Stream Compaction

```python
def stream_compact(values, mask):
    """
    Compact a stream by keeping only elements where mask is true.
    
    Args:
        values: (num_elements,) values to compact
        mask: (num_elements,) boolean mask (1 = keep, 0 = discard)
    
    Returns:
        (num_kept,) compacted values
    """
    # Compute prefix sum of mask to get output indices
    prefix_sum = ct.cumsum(mask, axis=0)
    
    # Total number of kept elements
    total_kept = prefix_sum[-1]
    
    # Initialize output
    output = ct.zeros((total_kept,), dtype=values.dtype)
    
    # Scatter values to output positions
    # (In practice, this would use parallel scatter)
    for i in range(values.numel):
        if mask[i] == 1:
            output_idx = prefix_sum[i] - 1
            output[output_idx] = values[i]
    
    return output

# Example
values = ct.from_list([10, 20, 30, 40, 50, 60], dtype=ct.int32)
mask = ct.from_list([1, 0, 1, 1, 0, 1], dtype=ct.int32)

compacted = stream_compact(values, mask)
print(compacted.shape)  # (4,)

# Result: [10, 30, 40, 60]
# Kept elements at indices 0, 2, 3, 5
```

### Example 2: Run-Length Encoding

```python
def run_length_encode(values):
    """
    Perform run-length encoding on a sequence.
    
    Args:
        values: (seq_len,) sequence of values
    
    Returns:
        runs: (num_runs,) run values
        lengths: (num_runs,) run lengths
    """
    seq_len = values.numel
    
    # Find run boundaries
    is_boundary = ct.zeros((seq_len,), dtype=ct.int32)
    is_boundary[0] = 1  # First element is always a boundary
    
    for i in range(1, seq_len):
        if values[i] != values[i-1]:
            is_boundary[i] = 1
    
    # Compute run indices using prefix sum
    run_indices = ct.cumsum(is_boundary, axis=0) - 1
    
    # Total number of runs
    num_runs = run_indices[-1] + 1
    
    # Extract run values (first occurrence of each run)
    runs = ct.zeros((num_runs,), dtype=values.dtype)
    for i in range(seq_len):
        run_idx = run_indices[i]
        if is_boundary[i] == 1:
            runs[run_idx] = values[i]
    
    # Compute run lengths
    lengths = ct.zeros((num_runs,), dtype=ct.int32)
    for i in range(seq_len):
        run_idx = run_indices[i]
        lengths[run_idx] += 1
    
    return runs, lengths

# Example
values = ct.from_list([1, 1, 1, 2, 2, 3, 3, 3, 3], dtype=ct.int32)
runs, lengths = run_length_encode(values)

print(f"Runs: {runs.shape}")     # (3,)
print(f"Lengths: {lengths.shape}")  # (3,)

# Runs: [1, 2, 3]
# Lengths: [3, 2, 4]
# Three runs: 1 appears 3 times, 2 appears 2 times, 3 appears 4 times
```

### Example 3: Difference of Prefix Sums

```python
def range_sum(prefix_sum, left, right):
    """
    Compute sum of elements in range [left, right] using prefix sum.
    
    Args:
        prefix_sum: (n,) prefix sum array
        left: left index (inclusive)
        right: right index (exclusive)
    
    Returns:
        sum of elements in range
    """
    if left == 0:
        return prefix_sum[right - 1]
    else:
        return prefix_sum[right - 1] - prefix_sum[left - 1]

# Example
# Original array: [1, 2, 3, 4, 5]
# Prefix sum: [1, 3, 6, 10, 15]
prefix_sum = ct.from_list([1, 3, 6, 10, 15], dtype=ct.int32)

# Sum of elements [1, 3) = elements[1] + elements[2] = 2 + 3 = 5
# Using prefix sum: prefix_sum[2] - prefix_sum[0] = 6 - 1 = 5
sum_range = range_sum(prefix_sum, 1, 3)
print(f"Sum of range [1, 3): {sum_range}")  # 5
```

### Example 4: Cumulative Moving Average

```python
def cumulative_moving_average(values):
    """
    Compute cumulative moving average.
    
    The CMA at position i is the average of all elements up to i.
    
    Args:
        values: (seq_len,) sequence of values
    
    Returns:
        (seq_len,) cumulative moving averages
    """
    # Compute cumulative sum
    cumsum = ct.cumsum(values, axis=0)
    
    # Create divisor: [1, 2, 3, ..., seq_len]
    divisors = ct.arange(1, values.numel + 1, dtype=ct.float32)
    
    # Compute averages
    cma = cumsum / divisors
    
    return cma

# Example
values = ct.from_list([10.0, 20.0, 30.0, 40.0], dtype=ct.float32)
cma = cumulative_moving_average(values)
print(cma.shape)  # (4,)

# Result: [10.0, 15.0, 20.0, 25.0]
# CMA[0] = 10/1 = 10.0
# CMA[1] = (10+20)/2 = 15.0
# CMA[2] = (10+20+30)/3 = 20.0
# CMA[3] = (10+20+30+40)/4 = 25.0
```

### Example 5: Scan for Parallel Prefix Computation

```python
def parallel_prefix_sum(data):
    """
    Demonstrate how scan enables parallel prefix computation.
    
    This is a simplified version - actual parallel implementation
    would use a tree-based approach.
    
    Args:
        data: (n,) input array
    
    Returns:
        (n,) prefix sums
    """
    return ct.cumsum(data, axis=0)

# Example
data = ct.from_list([1, 1, 1, 1, 1, 1, 1, 1], dtype=ct.int32)
prefix = parallel_prefix_sum(data)
print(prefix.shape)  # (8,)

# Result: [1, 2, 3, 4, 5, 6, 7, 8]
# This is much faster than sequential computation for large arrays
```

### Example 6: Sparse Matrix Vector Product

```python
def sparse_matvec_csr(values, col_indices, row_offsets, vector):
    """
    Sparse matrix-vector product using CSR format.
    
    Args:
        values: (nnz,) non-zero values
        col_indices: (nnz,) column indices
        row_offsets: (num_rows + 1,) row offset pointers (prefix sum of row nnz)
        vector: (num_cols,) dense vector
    
    Returns:
        (num_rows,) result vector
    """
    num_rows = row_offsets.numel - 1
    output = ct.zeros((num_rows,), dtype=ct.float32)
    
    # For each row
    for row in range(num_rows):
        start = row_offsets[row]
        end = row_offsets[row + 1]
        
        # Compute dot product of sparse row with dense vector
        row_sum = 0.0
        for idx in range(start, end):
            col = col_indices[idx]
            val = values[idx]
            row_sum += val * vector[col]
        
        output[row] = row_sum
    
    return output

# Example: 3x4 matrix with 5 non-zeros
# Matrix:
# [1, 0, 2, 0]
# [0, 3, 0, 4]
# [5, 0, 0, 0]

values = ct.from_list([1.0, 2.0, 3.0, 4.0, 5.0], dtype=ct.float32)
col_indices = ct.from_list([0, 2, 1, 3, 0], dtype=ct.int32)
row_offsets = ct.from_list([0, 2, 4, 5], dtype=ct.int32)  # Prefix sum of [2, 2, 1]
vector = ct.from_list([1.0, 2.0, 3.0, 4.0], dtype=ct.float32)

result = sparse_matvec_csr(values, col_indices, row_offsets, vector)
print(result.shape)  # (3,)

# Result: [7.0, 22.0, 5.0]
# Row 0: 1*1 + 2*3 = 7
# Row 1: 3*2 + 4*4 = 22
# Row 2: 5*1 = 5
```

### Example 7: Prefix Sum for Load Balancing

```python
def distribute_workload(num_items, num_workers):
    """
    Distribute items among workers using prefix sum.
    
    Args:
        num_items: total number of items
        num_workers: number of workers
    
    Returns:
        (num_workers + 1,) start and end indices for each worker
    """
    # Compute base chunk size
    base_size = num_items // num_workers
    remainder = num_items % num_workers
    
    # Compute chunk sizes (first 'remainder' workers get one extra item)
    chunk_sizes = ct.full((num_workers,), base_size, dtype=ct.int32)
    for i in range(remainder):
        chunk_sizes[i] += 1
    
    # Compute offsets using prefix sum
    offsets = ct.cumsum(chunk_sizes, axis=0)
    
    # Add leading 0 and trailing num_items
    distribution = ct.zeros((num_workers + 1,), dtype=ct.int32)
    for i in range(num_workers):
        distribution[i + 1] = offsets[i]
    
    return distribution

# Example
num_items = 17
num_workers = 4

distribution = distribute_workload(num_items, num_workers)
print(distribution.shape)  # (5,)

# Result: [0, 5, 9, 13, 17]
# Worker 0: items [0, 5)    → 5 items
# Worker 1: items [5, 9)    → 4 items
# Worker 2: items [9, 13)   → 4 items
# Worker 3: items [13, 17)  → 4 items
```

### Example 8: Inclusive Scan for Probability Distributions

```python
def probability_to_cumulative(probs):
    """
    Convert probability distribution to cumulative distribution.
    
    Args:
        probs: (num_bins,) probability distribution (sums to 1)
    
    Returns:
        (num_bins,) cumulative distribution
    """
    return ct.cumsum(probs, axis=0)

def sample_from_cumulative(cumulative, random_value):
    """
    Sample from a cumulative distribution using binary search.
    
    Args:
        cumulative: (num_bins,) cumulative distribution
        random_value: uniform random value in [0, 1]
    
    Returns:
        sampled bin index
    """
    # Binary search to find bin
    for i in range(cumulative.numel):
        if random_value < cumulative[i]:
            return i
    return cumulative.numel - 1

# Example
# Probability distribution: [0.1, 0.3, 0.4, 0.2]
probs = ct.from_list([0.1, 0.3, 0.4, 0.2], dtype=ct.float32)
cumulative = probability_to_cumulative(probs)
print(cumulative.shape)  # (4,)

# Cumulative: [0.1, 0.4, 0.8, 1.0]
# Sample random value 0.5 → falls in range [0.4, 0.8) → bin 2
sample = sample_from_cumulative(cumulative, 0.5)
print(f"Sampled bin: {sample}")  # 2
```

## Best Practices

1. **Numerical Stability**: For cumulative operations with large values, be aware of potential overflow. Consider using logarithms or scaling for very large sequences.

2. **Exclusive vs Inclusive**: Choose the right scan type for your application. Use exclusive scan for output position computation and inclusive scan for running totals.

3. **Memory Access Patterns**: Scan operations have efficient parallel implementations but may have memory bandwidth limitations for very large tiles.

4. **Associativity**: Custom scan functions must be associative for correct results. Verify that `(a op b) op c == a op (b op c)`.

5. **Identity Elements**: Choose appropriate identity elements for custom scans (0 for sum, 1 for product, -inf for max, etc.).

6. **Multi-dimensional Scans**: When scanning along specific axes, be clear about whether each row/column is scanned independently or the scan wraps around dimensions.

7. **Performance**: Scan operations are typically O(n log n) in parallel implementations. For small sequences, sequential computation may be faster.

8. **Combining Operations**: Many operations can be expressed as combinations of scans and reductions. Choose the most efficient representation for your use case.

9. **Data Types**: Be aware of precision issues when using float16 for cumulative operations, as errors can accumulate.

10. **Testing**: Always verify scan operations with small test cases where you can compute expected results manually.
