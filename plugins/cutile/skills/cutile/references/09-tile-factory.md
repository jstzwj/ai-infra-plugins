# Tile Factory Functions

This chapter provides a comprehensive guide to tile factory functions in cuTile. These functions create new tiles with specific contents or patterns, which are essential for initializing accumulators, creating index arrays, generating masks, and preparing data for computation.

## Overview

cuTile provides several factory functions for creating tiles:

- **`ct.zeros()`** — Create a tile filled with zeros
- **`ct.ones()`** — Create a tile filled with ones
- **`ct.full()`** — Create a tile filled with a specific value
- **`ct.arange()`** — Create a tile with sequential values

These functions are used to initialize tiles that reside in fast on-chip memory (registers and shared memory), as opposed to global arrays which reside in device memory.

## ct.zeros()

Creates a tile filled with zeros.

### Syntax

```python
ct.zeros(
    shape: tuple[int, ...],
    dtype: DType
) -> Tile
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple[int, ...]` | Dimensions of the tile (each must be a power of 2) |
| `dtype` | `DType` | Data type for the tile elements |

### Returns

A `Tile` of the specified shape and dtype, with all elements set to zero.

### Description

`ct.zeros()` is the most commonly used factory function. It creates a tile initialized to zero, which is essential for:

- **Accumulators** in reductions and matrix operations
- **Masks** for conditional operations
- **Temporary storage** that will be overwritten
- **Padding** in boundary handling

### Examples

**1D Zero Tile:**

```python
import cutile as ct

# Create a 1D tile of 32 zeros
tile = ct.zeros((32,), ct.float32)
# Result: [0.0, 0.0, 0.0, ..., 0.0]  (32 elements)
```

**2D Zero Tile:**

```python
# Create a 64×64 tile of zeros
tile = ct.zeros((64, 64), ct.float32)
# Result: 64×64 matrix, all elements = 0.0
```

**3D Zero Tile:**

```python
# Create a 32×32×16 tile of zeros
tile = ct.zeros((32, 32, 16), ct.float32)
```

**Different Data Types:**

```python
# Float32 zeros
f32_zeros = ct.zeros((128, 128), ct.float32)

# Float16 zeros
f16_zeros = ct.zeros((64, 64), ct.float16)

# Integer zeros
int_zeros = ct.zeros((256,), ct.int32)

# Boolean zeros (all False)
bool_zeros = ct.zeros((64,), ct.bool_)
```

**Common Use Case: Matrix Accumulator**

```python
def matmul_kernel(A, B, C):
    """
    Matrix multiplication: C = A @ B
    """
    TM, TN, TK = 128, 128, 32
    
    # Get block position
    i = ct.bid(0)
    j = ct.bid(1)
    
    # Initialize accumulator with zeros
    # This is the most common use of ct.zeros()
    c_tile = ct.zeros((TM, TN), ct.float32)
    
    # Loop over K dimension
    for k in range(K // TK):
        a_tile = ct.load(A, (i, k), (TM, TK))
        b_tile = ct.load(B, (k, j), (TK, TN))
        
        # Accumulate
        c_tile = c_tile + ct.dot(a_tile, b_tile)
    
    # Store result
    ct.store(C, (i, j), c_tile)
```

**Common Use Case: Reduction Accumulator**

```python
def sum_rows_kernel(A, sums):
    """
    Sum each row of matrix A
    """
    TM, TN = 64, 1024
    
    i = ct.bid(0)
    
    # Load row tile
    row_tile = ct.load(A, (i, 0), (TM, TN))
    
    # Initialize accumulator
    row_sums = ct.zeros((TM, 1), ct.float32)
    
    # Sum along columns
    for j in range(0, TN, 64):
        chunk = row_tile[:, j:j+64]
        row_sums = row_sums + ct.sum(chunk, axis=1, keepdims=True)
    
    # Store
    ct.store(sums, (i, 0), row_sums)
```

**Common Use Case: Mask Initialization**

```python
def masked_operation_kernel(A, B, mask):
    """
    Apply operation only where mask is True
    """
    TM, TN = 64, 64
    
    i = ct.bid(0)
    j = ct.bid(1)
    
    # Load data
    a_tile = ct.load(A, (i, j), (TM, TN))
    mask_tile = ct.load(mask, (i, j), (TM, TN))
    
    # Initialize result with zeros
    result = ct.zeros((TM, TN), ct.float32)
    
    # Only compute where mask is True
    # Masked elements stay zero
    result = ct.where(mask_tile, a_tile * 2.0, result)
    
    ct.store(B, (i, j), result)
```

**Common Use Case: Temporary Storage**

```python
def complex_kernel(A, B):
    """
    Multi-pass computation needing temporary storage
    """
    TM, TN = 64, 64
    
    i = ct.bid(0)
    j = ct.bid(1)
    
    # Load input
    a_tile = ct.load(A, (i, j), (TM, TN))
    
    # Temporary tile for intermediate results
    temp = ct.zeros((TM, TN), ct.float32)
    
    # First pass
    temp = first_pass(a_tile, temp)
    
    # Second pass (reuse temp)
    temp = second_pass(a_tile, temp)
    
    # Final result
    result = finalize(temp)
    
    ct.store(B, (i, j), result)
```

## ct.ones()

Creates a tile filled with ones.

### Syntax

```python
ct.ones(
    shape: tuple[int, ...],
    dtype: DType
) -> Tile
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple[int, ...]` | Dimensions of the tile (each must be a power of 2) |
| `dtype` | `DType` | Data type for the tile elements |

### Returns

A `Tile` of the specified shape and dtype, with all elements set to one.

### Description

`ct.ones()` creates a tile initialized to one. This is useful for:

- **Scaling operations** (multiplying by a constant)
- **Creating uniform weights** for weighted averages
- **Identity masks** for element-wise operations
- **Normalization** in certain algorithms

### Examples

**Basic Usage:**

```python
import cutile as ct

# Create a 1D tile of 32 ones
tile = ct.ones((32,), ct.float32)
# Result: [1.0, 1.0, 1.0, ..., 1.0]  (32 elements)
```

**2D Ones Tile:**

```python
# Create a 64×64 tile of ones
tile = ct.ones((64, 64), ct.float32)
# Result: 64×64 matrix, all elements = 1.0
```

**Different Data Types:**

```python
# Float32 ones
f32_ones = ct.ones((128, 128), ct.float32)

# Float16 ones
f16_ones = ct.ones((64, 64), ct.float16)

# Integer ones
int_ones = ct.ones((256,), ct.int32)
```

**Use Case: Scaling Operation**

```python
def scale_kernel(A, B, scale_factor):
    """
    B = A * scale_factor
    """
    TM, TN = 64, 64
    
    i = ct.bid(0)
    j = ct.bid(1)
    
    # Load input
    a_tile = ct.load(A, (i, j), (TM, TN))
    
    # Create tile of ones scaled by factor
    scale_tile = ct.ones((TM, TN), ct.float32) * scale_factor
    
    # Scale
    b_tile = a_tile * scale_tile
    
    ct.store(B, (i, j), b_tile)
```

**Use Case: Weighted Sum**

```python
def weighted_sum_kernel(A, B, C, alpha, beta):
    """
    C = alpha * A + beta * B
    """
    TM, TN = 64, 64
    
    i = ct.bid(0)
    j = ct.bid(1)
    
    # Load inputs
    a_tile = ct.load(A, (i, j), (TM, TN))
    b_tile = ct.load(B, (i, j), (TM, TN))
    
    # Create weight tiles
    alpha_tile = ct.ones((TM, TN), ct.float32) * alpha
    beta_tile = ct.ones((TM, TN), ct.float32) * beta
    
    # Weighted sum
    c_tile = alpha_tile * a_tile + beta_tile * b_tile
    
    ct.store(C, (i, j), c_tile)
```

**Use Case: Uniform Smoothing**

```python
def smooth_kernel(A, B):
    """
    Apply uniform smoothing filter
    Each pixel becomes average of its neighbors
    """
    TILE = 64
    
    i = ct.bid(0)
    j = ct.bid(1)
    
    # Load center tile
    center = ct.load(A, (i, j), (TILE, TILE))
    
    # Initialize accumulator
    smoothed = ct.zeros((TILE, TILE), ct.float32)
    
    # Sum 3×3 neighborhood
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            ni = i + di
            nj = j + dj
            neighbor = ct.load(A, (ni, nj), (TILE, TILE))
            smoothed = smoothed + neighbor
    
    # Divide by 9 (average)
    ones = ct.ones((TILE, TILE), ct.float32)
    smoothed = smoothed / (ones * 9.0)
    
    ct.store(B, (i, j), smoothed)
```

**Use Case: Identity Mask**

```python
def apply_mask_kernel(A, B, mask):
    """
    Apply mask: output = input where mask=True, else input
    (Identity operation when mask is all True)
    """
    TM, TN = 64, 64
    
    i = ct.bid(0)
    j = ct.bid(1)
    
    a_tile = ct.load(A, (i, j), (TM, TN))
    mask_tile = ct.load(mask, (i, j), (TM, TN))
    
    # Create identity (all ones)
    identity = ct.ones((TM, TN), ct.float32)
    
    # Apply mask
    result = a_tile * mask_tile * identity
    
    ct.store(B, (i, j), result)
```

## ct.full()

Creates a tile filled with a specified value.

### Syntax

```python
ct.full(
    shape: tuple[int, ...],
    value: int | float,
    dtype: DType
) -> Tile
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple[int, ...]` | Dimensions of the tile (each must be a power of 2) |
| `value` | `int \| float` | The fill value |
| `dtype` | `DType` | Data type for the tile elements |

### Returns

A `Tile` of the specified shape and dtype, with all elements set to `value`.

### Description

`ct.full()` is a generalization of `ct.zeros()` and `ct.ones()`. It creates a tile where every element has the same specified value. This is useful for:

- **Special constants** (π, e, etc.)
- **Algorithm-specific values**
- **Thresholds** in comparisons
- **Padding values** in certain operations

### Examples

**Basic Usage:**

```python
import cutile as ct

# Create a tile filled with 3.14
tile = ct.full((32, 32), 3.14, ct.float32)
# Result: 32×32 matrix, all elements = 3.14
```

**Integer Values:**

```python
# Create a tile filled with 42
tile = ct.full((64, 64), 42, ct.int32)
# Result: 64×64 matrix, all elements = 42
```

**Negative Values:**

```python
# Create a tile filled with -1.0
tile = ct.full((128, 128), -1.0, ct.float32)
```

**Mathematical Constants:**

```python
# Pi
pi_tile = ct.full((64, 64), 3.14159265359, ct.float32)

# Euler's number
e_tile = ct.full((64, 64), 2.71828182846, ct.float32)

# Golden ratio
phi_tile = ct.full((64, 64), 1.61803398875, ct.float32)
```

**Use Case: Threshold Operation**

```python
def threshold_kernel(A, B, threshold):
    """
    Binarize: output = 1 if input > threshold, else 0
    """
    TM, TN = 64, 64
    
    i = ct.bid(0)
    j = ct.bid(1)
    
    # Load input
    a_tile = ct.load(A, (i, j), (TM, TN))
    
    # Create threshold tile
    thresh_tile = ct.full((TM, TN), threshold, ct.float32)
    
    # Apply threshold
    b_tile = ct.where(a_tile > thresh_tile, 1.0, 0.0)
    
    ct.store(B, (i, j), b_tile)
```

**Use Case: Clipping**

```python
def clip_kernel(A, B, min_val, max_val):
    """
    Clip values to [min_val, max_val] range
    """
    TM, TN = 64, 64
    
    i = ct.bid(0)
    j = ct.bid(1)
    
    # Load input
    a_tile = ct.load(A, (i, j), (TM, TN))
    
    # Create min/max tiles
    min_tile = ct.full((TM, TN), min_val, ct.float32)
    max_tile = ct.full((TM, TN), max_val, ct.float32)
    
    # Clip
    clipped = ct.maximum(min_tile, ct.minimum(max_tile, a_tile))
    
    ct.store(B, (i, j), clipped)
```

**Use Case: Activation Function**

```python
def relu_kernel(A, B):
    """
    ReLU activation: max(0, x)
    """
    TM, TN = 64, 64
    
    i = ct.bid(0)
    j = ct.bid(1)
    
    # Load input
    a_tile = ct.load(A, (i, j), (TM, TN))
    
    # Create zero tile
    zero_tile = ct.full((TM, TN), 0.0, ct.float32)
    
    # ReLU
    b_tile = ct.maximum(zero_tile, a_tile)
    
    ct.store(B, (i, j), b_tile)
```

**Use Case: Leaky ReLU**

```python
def leaky_relu_kernel(A, B, alpha=0.01):
    """
    Leaky ReLU: max(alpha*x, x)
    """
    TM, TN = 64, 64
    
    i = ct.bid(0)
    j = ct.bid(1)
    
    # Load input
    a_tile = ct.load(A, (i, j), (TM, TN))
    
    # Create alpha tile
    alpha_tile = ct.full((TM, TN), alpha, ct.float32)
    
    # Leaky ReLU
    scaled = a_tile * alpha_tile
    b_tile = ct.maximum(scaled, a_tile)
    
    ct.store(B, (i, j), b_tile)
```

**Use Case: Adding Constant**

```python
def add_constant_kernel(A, B, constant):
    """
    B = A + constant
    """
    TM, TN = 64, 64
    
    i = ct.bid(0)
    j = ct.bid(1)
    
    # Load input
    a_tile = ct.load(A, (i, j), (TM, TN))
    
    # Create constant tile
    const_tile = ct.full((TM, TN), constant, ct.float32)
    
    # Add
    b_tile = a_tile + const_tile
    
    ct.store(B, (i, j), b_tile)
```

## ct.arange()

Creates a 1D tile with sequential values starting from 0.

### Syntax

```python
ct.arange(
    size: int
) -> Tile
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `size` | `int` | Number of elements (must be a power of 2) |

### Returns

A 1D `Tile` of length `size` with values `[0, 1, 2, ..., size-1]`.

### Description

`ct.arange()` generates a sequence of integers from 0 to `size-1`. This is essential for:

- **Index generation** for gather/scatter operations
- **Loop counters** in tile-based algorithms
- **Creating coordinate grids**
- **Address calculations**

### Examples

**Basic Usage:**

```python
import cutile as ct

# Create a sequence [0, 1, 2, ..., 15]
indices = ct.arange(16)
# Result: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
```

**Larger Sequences:**

```python
# Create [0, 1, 2, ..., 127]
indices = ct.arange(128)
```

**Use Case: Index Tiles for Gather**

```python
def indexed_load_kernel(data, indices, output):
    """
    Load data at specified indices
    """
    TILE_SIZE = 64
    
    i = ct.bid(0)
    
    # Load indices
    idx_tile = ct.load(indices, (i,), (TILE_SIZE,))
    
    # Gather data at those indices
    values = ct.gather(data, idx_tile)
    
    ct.store(output, (i,), values)
```

**Use Case: Generating Indices**

```python
def generate_indices_kernel(indices):
    """
    Generate sequential indices [0, 1, 2, ...]
    """
    TILE_SIZE = 128
    
    i = ct.bid(0)
    
    # Generate local indices [0, 1, 2, ..., TILE_SIZE-1]
    local_indices = ct.arange(TILE_SIZE)
    
    # Compute global indices
    offset = i * TILE_SIZE
    global_indices = local_indices + offset
    
    ct.store(indices, (i,), global_indices)
```

**Use Case: Coordinate Grid**

```python
def coordinate_grid_kernel(X, Y):
    """
    Generate 2D coordinate grid
    X contains x-coordinates, Y contains y-coordinates
    """
    TM, TN = 64, 64
    
    i = ct.bid(0)
    j = ct.bid(1)
    
    # Generate x coordinates: [j*TN, j*TN+1, ..., j*TN+TN-1]
    x_indices = ct.arange(TN)
    x_offset = j * TN
    x_coords = x_indices + x_offset
    
    # Generate y coordinates: [i*TM, i*TM+1, ..., i*TM+TM-1]
    y_indices = ct.arange(TM)
    y_offset = i * TM
    y_coords = y_indices + y_offset
    
    # Broadcast to create grid
    x_grid = ct.broadcast(x_coords, (TM, TN))
    y_grid = ct.broadcast(y_coords.reshape((TM, 1)), (TM, TN))
    
    ct.store(X, (i, j), x_grid)
    ct.store(Y, (i, j), y_grid)
```

**Use Case: Diagonal Access**

```python
def diagonal_kernel(A, diag):
    """
    Extract diagonal elements
    """
    TILE_SIZE = 64
    
    i = ct.bid(0)
    
    # Load diagonal tile
    tile = ct.load(A, (i, i), (TILE_SIZE, TILE_SIZE))
    
    # Extract diagonal using arange
    indices = ct.arange(TILE_SIZE)
    diagonal_elements = tile[indices, indices]
    
    ct.store(diag, (i,), diagonal_elements)
```

**Use Case: Strided Access**

```python
def strided_load_kernel(A, B, stride):
    """
    Load every stride-th element
    """
    TILE_SIZE = 64
    
    i = ct.bid(0)
    
    # Generate indices: [0, stride, 2*stride, ...]
    base_indices = ct.arange(TILE_SIZE)
    strided_indices = base_indices * stride
    
    # Add offset for this tile
    offset = i * TILE_SIZE * stride
    global_indices = strided_indices + offset
    
    # Gather elements
    values = ct.gather(A, global_indices)
    
    ct.store(B, (i,), values)
```

**Use Case: Transpose Indices**

```python
def transpose_index_kernel(src_indices, dst_indices):
    """
    Convert (i, j) indices to (j, i) for transpose
    """
    TM, TN = 64, 64
    
    i = ct.bid(0)
    j = ct.bid(1)
    
    # Generate local indices
    i_local = ct.arange(TM)
    j_local = ct.arange(TN)
    
    # Compute global indices
    i_global = i * TM + i_local
    j_global = j * TN + j_local
    
    # Create 2D index grid
    # For transpose: src[i, j] -> dst[j, i]
    src_i = ct.broadcast(i_global.reshape((TM, 1)), (TM, TN))
    src_j = ct.broadcast(j_global, (TM, TN))
    
    # Compute linear indices
    src_linear = src_i * TN + src_j
    dst_linear = src_j * TM + src_i
    
    # Store mapping
    ct.store(src_indices, (i, j), src_linear)
    ct.store(dst_indices, (i, j), dst_linear)
```

## Common Patterns

### Pattern 1: Accumulator Initialization

```python
# Initialize accumulator for reduction
acc = ct.zeros((TM, TN), ct.float32)

# Accumulate
for k in range(num_k_tiles):
    tile = ct.load(A, (i, k), (TM, TN))
    acc = acc + tile

# acc now contains the sum
```

### Pattern 2: Creating Constant Tiles

```python
# Method 1: Using ct.full()
const_tile = ct.full((64, 64), 3.14, ct.float32)

# Method 2: Using ct.ones() with multiplication
const_tile = ct.ones((64, 64), ct.float32) * 3.14

# Method 3: Using ct.zeros() with addition (less efficient)
const_tile = ct.zeros((64, 64), ct.float32) + 3.14

# Prefer ct.full() for clarity and efficiency
```

### Pattern 3: Index Generation

```python
# Generate indices for gather/scatter
size = 128
indices = ct.arange(size)

# Add offset for global indexing
offset = ct.bid(0) * size
global_indices = indices + offset

# Use in gather
values = ct.gather(array, global_indices)
```

### Pattern 4: Broadcast Pattern

```python
# Create row tile (broadcast 1D to 2D)
row_values = ct.arange(TN)
row_tile = ct.broadcast(row_values, (TM, TN))

# Create column tile (reshape then broadcast)
col_values = ct.arange(TM).reshape((TM, 1))
col_tile = ct.broadcast(col_values, (TM, TN))

# Use for coordinate operations
result = row_tile + col_tile  # Outer addition
```

### Pattern 5: Mask Creation

```python
# Create boolean mask
mask = ct.zeros((TM, TN), ct.bool_)

# Set specific region to True
for i in range(TM):
    for j in range(TN):
        if condition(i, j):
            mask[i, j] = True

# Use mask
result = ct.where(mask, true_values, false_values)
```

### Pattern 6: Multi-Dimensional Initialization

```python
# Initialize 3D tile
tile_3d = ct.zeros((TM, TN, TK), ct.float32)

# Initialize 4D tile
tile_4d = ct.zeros((TM, TN, TK, TL), ct.float32)
```

## Complete Examples

### Example 1: Matrix Multiplication with Factory Functions

```python
import cutile as ct

def matmul_kernel(A, B, C):
    """
    C = A @ B using factory functions
    """
    TM, TN, TK = 128, 128, 32
    
    i = ct.bid(0)
    j = ct.bid(1)
    
    # Initialize accumulator with zeros
    c_tile = ct.zeros((TM, TN), ct.float32)
    
    # Loop over K dimension
    num_k_tiles = (A.shape[1] + TK - 1) // TK
    for k in range(num_k_tiles):
        # Load tiles
        a_tile = ct.load(A, (i, k), (TM, TK), padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(B, (k, j), (TK, TN), padding_mode=ct.PaddingMode.ZERO)
        
        # Accumulate
        c_tile = c_tile + ct.dot(a_tile, b_tile)
    
    # Store result
    ct.store(C, (i, j), c_tile)
```

### Example 2: Softmax with Factory Functions

```python
def softmax_kernel(input, output):
    """
    Row-wise softmax: exp(x) / sum(exp(x))
    """
    TILE_M = 64
    TILE_N = 1024
    
    i = ct.bid(0)
    
    # Load input tile
    x_tile = ct.load(input, (i, 0), (TILE_M, TILE_N))
    
    # Find max for numerical stability
    x_max = ct.max(x_tile, axis=1, keepdims=True)
    
    # Subtract max and exp
    # Create tile of zeros for subtraction
    max_tile = ct.broadcast(x_max, (TILE_M, TILE_N))
    exp_x = ct.exp(x_tile - max_tile)
    
    # Sum and normalize
    exp_sum = ct.sum(exp_x, axis=1, keepdims=True)
    sum_tile = ct.broadcast(exp_sum, (TILE_M, TILE_N))
    
    # Divide
    softmax = exp_x / sum_tile
    
    # Store
    ct.store(output, (i, 0), softmax)
```

### Example 3: Convolution with Factory Functions

```python
def conv2d_kernel(input, kernel, output):
    """
    2D convolution
    """
    TILE_SIZE = 64
    KERNEL_SIZE = 3
    
    i = ct.bid(0)
    j = ct.bid(1)
    
    # Initialize output tile with zeros
    out_tile = ct.zeros((TILE_SIZE, TILE_SIZE), ct.float32)
    
    # Convolve with kernel
    for ki in range(KERNEL_SIZE):
        for kj in range(KERNEL_SIZE):
            # Load neighbor
            ni = i + ki - 1
            nj = j + kj - 1
            neighbor = ct.load(input, (ni, nj), (TILE_SIZE, TILE_SIZE),
                             padding_mode=ct.PaddingMode.ZERO)
            
            # Get kernel weight
            weight = kernel[ki, kj]
            
            # Create weight tile
            weight_tile = ct.full((TILE_SIZE, TILE_SIZE), weight, ct.float32)
            
            # Accumulate
            out_tile = out_tile + weight_tile * neighbor
    
    # Store
    ct.store(output, (i, j), out_tile)
```

### Example 4: Layer Normalization

```python
def layer_norm_kernel(input, output, gamma, beta, eps=1e-5):
    """
    Layer normalization
    """
    TILE_M = 64
    TILE_N = 1024
    
    i = ct.bid(0)
    
    # Load input
    x_tile = ct.load(input, (i, 0), (TILE_M, TILE_N))
    
    # Compute mean
    mean = ct.mean(x_tile, axis=1, keepdims=True)
    mean_tile = ct.broadcast(mean, (TILE_M, TILE_N))
    
    # Compute variance
    var = ct.mean((x_tile - mean_tile) ** 2, axis=1, keepdims=True)
    var_tile = ct.broadcast(var, (TILE_M, TILE_N))
    
    # Create epsilon tile
    eps_tile = ct.full((TILE_M, TILE_N), eps, ct.float32)
    
    # Normalize
    normalized = (x_tile - mean_tile) / ct.sqrt(var_tile + eps_tile)
    
    # Scale and shift
    gamma_tile = ct.load(gamma, (i, 0), (TILE_M, TILE_N))
    beta_tile = ct.load(beta, (i, 0), (TILE_M, TILE_N))
    
    # Apply affine transformation
    y_tile = gamma_tile * normalized + beta_tile
    
    # Store
    ct.store(output, (i, 0), y_tile)
```

### Example 5: Indexed Operations

```python
def embedding_lookup_kernel(input_indices, embedding_matrix, output):
    """
    Lookup embeddings at specified indices
    """
    TILE_SIZE = 64
    EMBEDDING_DIM = 128
    
    i = ct.bid(0)
    
    # Load indices
    idx_tile = ct.load(input_indices, (i,), (TILE_SIZE,))
    
    # Initialize output tile
    embed_tile = ct.zeros((TILE_SIZE, EMBEDDING_DIM), ct.float32)
    
    # Gather embeddings
    for j in range(TILE_SIZE):
        idx = idx_tile[j]
        embedding = ct.gather(embedding_matrix, idx)
        embed_tile[j, :] = embedding
    
    # Store
    ct.store(output, (i, 0), embed_tile)
```

### Example 6: Attention Mechanism

```python
def attention_kernel(Q, K, V, output):
    """
    Scaled dot-product attention
    """
    TM, TN, TK = 64, 64, 64
    
    i = ct.bid(0)
    j = ct.bid(1)
    
    # Load query tile
    q_tile = ct.load(Q, (i, 0), (TM, TK))
    
    # Initialize attention weights
    attn = ct.zeros((TM, TN), ct.float32)
    
    # Compute attention scores
    for k in range(num_k_tiles):
        k_tile = ct.load(K, (k, j), (TK, TN))
        scores = ct.dot(q_tile, k_tile)
        attn = attn + scores
    
    # Scale
    scale_tile = ct.full((TM, TN), 1.0 / (TK ** 0.5), ct.float32)
    attn = attn * scale_tile
    
    # Softmax
    attn_max = ct.max(attn, axis=1, keepdims=True)
    attn_max_tile = ct.broadcast(attn_max, (TM, TN))
    exp_attn = ct.exp(attn - attn_max_tile)
    attn_sum = ct.sum(exp_attn, axis=1, keepdims=True)
    attn_sum_tile = ct.broadcast(attn_sum, (TM, TN))
    attn = exp_attn / attn_sum_tile
    
    # Apply attention to values
    result = ct.zeros((TM, TN), ct.float32)
    for k in range(num_v_tiles):
        v_tile = ct.load(V, (k, j), (TK, TN))
        result = result + ct.dot(attn, v_tile)
    
    # Store
    ct.store(output, (i, j), result)
```

## Performance Considerations

### Zeros vs Ones vs Full

```python
# All create tiles with the same performance
zeros = ct.zeros((128, 128), ct.float32)
ones = ct.ones((128, 128), ct.float32)
full = ct.full((128, 128), 3.14, ct.float32)
```

The choice between them should be based on readability and the value you need, not performance.

### Broadcasting Cost

```python
# Efficient: Create once, broadcast
const = ct.full((64,), 3.14, ct.float32)
tile = ct.broadcast(const, (128, 128))

# Less efficient: Create full-size tile directly
tile = ct.full((128, 128), 3.14, ct.float32)

# For very large constants, broadcasting is more efficient
# For small tiles, the difference is negligible
```

### Arange Usage

```python
# Efficient: Use arange for sequential values
indices = ct.arange(128)

# Less efficient: Build manually
indices = ct.zeros((128,), ct.int32)
for i in range(128):
    indices[i] = i
```

### Reusing Factory Tiles

```python
# Good: Create once, use multiple times
zero_tile = ct.zeros((64, 64), ct.float32)
for i in range(10):
    result = zero_tile + ct.load(A, (i,), (64, 64))

# Avoid: Recreating in loop
for i in range(10):
    zero_tile = ct.zeros((64, 64), ct.float32)  # Unnecessary
    result = zero_tile + ct.load(A, (i,), (64, 64))
```

## Troubleshooting

### Issue: Non-Power-of-2 Shapes

```python
# Error: Shape must be power of 2
tile = ct.zeros((100, 100), ct.float32)  # Error

# Correct: Use power of 2
tile = ct.zeros((128, 128), ct.float32)  # OK
```

### Issue: Incorrect Data Type

```python
# May cause precision issues
tile = ct.zeros((64, 64), ct.float16)  # Limited precision

# Better: Use appropriate dtype
tile = ct.zeros((64, 64), ct.float32)  # Full precision
```

### Issue: Arange Size Mismatch

```python
# Error if size doesn't match expected usage
indices = ct.arange(64)  # 64 elements
tile = ct.load(A, (i,), (128,))  # Expects 128 indices

# Correct: Match sizes
indices = ct.arange(128)
tile = ct.load(A, (i,), (128,))
```

## Summary Table

| Function | Purpose | Common Use Case |
|----------|---------|-----------------|
| `ct.zeros(shape, dtype)` | Create zero-filled tile | Accumulators, temporary storage |
| `ct.ones(shape, dtype)` | Create one-filled tile | Scaling, weights, masks |
| `ct.full(shape, value, dtype)` | Create tile with specific value | Constants, thresholds |
| `ct.arange(size)` | Create sequential values | Indices, loop counters |

## Best Practices

1. **Use `ct.zeros()` for accumulators** in reductions and matrix operations
2. **Prefer `ct.full()` over `ct.ones() * value`** for clarity when creating constant tiles
3. **Reuse factory tiles** when possible instead of recreating them
4. **Use `ct.arange()` for index generation** rather than manual loops
5. **Always use power-of-2 shapes** for tile dimensions
6. **Choose appropriate dtype** based on precision requirements
7. **Consider broadcasting** for creating large constant tiles from small ones
8. **Initialize masks with `ct.zeros()`** then set True values as needed
