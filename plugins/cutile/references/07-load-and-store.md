# Load and Store Operations

This chapter provides a comprehensive guide to memory operations in cuTile, including loading data from global memory, storing results back, and handling irregular access patterns through gather/scatter operations. Understanding these operations is essential for writing efficient GPU kernels.

## Overview

cuTile provides several operations for moving data between global memory and the fast on-chip memory (shared memory and registers):

- **`ct.bid()`** - Get current block index
- **`ct.num_blocks()`** - Get total number of blocks
- **`ct.load()`** - Load a tile from global memory
- **`ct.store()`** - Store a tile to global memory
- **`ct.gather()`** - Load irregular elements
- **`ct.scatter()`** - Store to irregular locations

## Block Identification

### ct.bid(axis)

Gets the index of the current block along the specified grid dimension.

**Syntax:**
```python
ct.bid(axis: int) -> Tile
```

**Parameters:**
- `axis` (int) — The grid dimension index (0 for x, 1 for y, 2 for z)

**Returns:**
- A scalar tile containing the block index along the specified axis

**Description:**

`ct.bid()` returns the position of the current thread block within the grid along the specified dimension. This is essential for determining which portion of the global data each block should process.

**Examples:**

```python
import cutile as ct

# 1D grid: Get block ID along the x-dimension
pid = ct.bid(0)

# 2D grid: Get block coordinates
block_i = ct.bid(0)  # Row index
block_j = ct.bid(1)  # Column index

# 3D grid: Get block coordinates
block_x = ct.bid(0)
block_y = ct.bid(1)
block_z = ct.bid(2)
```

**Common Use Cases:**

```python
# Calculate which tile of a matrix this block should process
def matrix_multiply(X, Y, Z, TM, TN, TK):
    # Grid is 2D: (M // TM, N // TN)
    i = ct.bid(0)  # Which row block
    j = ct.bid(1)  # Which column block
    
    # Load the appropriate tiles
    x_tile = ct.load(X, (i, 0), (TM, TK))
    y_tile = ct.load(Y, (0, j), (TK, TN))
    
    # Compute and store
    z_tile = ct.dot(x_tile, y_tile)
    ct.store(Z, (i, j), z_tile)
```

### ct.num_blocks(axis)

Gets the total number of blocks along the specified grid dimension.

**Syntax:**
```python
ct.num_blocks(axis: int) -> Tile
```

**Parameters:**
- `axis` (int) — The grid dimension index

**Returns:**
- A scalar tile containing the number of blocks along the axis

**Examples:**

```python
# Get total number of blocks in 1D grid
total_blocks = ct.num_blocks(0)

# 2D grid dimensions
grid_rows = ct.num_blocks(0)
grid_cols = ct.num_blocks(1)

# Check if this is the last block
is_last = ct.bid(0) == ct.num_blocks(0) - 1
```

### ct.num_tiles(array, axis)

Gets the number of tiles required to partition an array along a given axis.

**Syntax:**
```python
ct.num_tiles(array: Array, axis: int) -> Tile
```

**Parameters:**
- `array` — The global array to query
- `axis` (int) — Which dimension to query

**Returns:**
- A scalar tile with the number of tiles along the axis

**Description:**

This function calculates how many tiles of the current size fit along the specified dimension of the array. This is equivalent to `ceil(array.shape[axis] / tile_shape[axis])`.

**Note:** This operation is typically accessed through `TiledView.num_tiles()` rather than directly.

## Loading Tiles

### ct.load()

Loads a tile of data from global memory into fast on-chip memory.

**Syntax:**
```python
ct.load(
    array: Array,
    index: tuple[int, ...],
    shape: tuple[int, ...],
    *,
    latency: int | None = None,
    allow_tma: bool | None = None,
    memory_order: MemoryOrder = MemoryOrder.WEAK,
    memory_scope: MemoryScope = MemoryScope.NONE,
    padding_mode: PaddingMode | None = None
) -> Tile
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array` | `Array` | — | The global array to load from |
| `index` | `tuple[int, ...]` | — | Tile space coordinates (which tile to load) |
| `shape` | `tuple[int, ...]` | — | Dimensions of the tile to load (must be powers of 2) |
| `latency` | `int \| None` | `None` | Hint for DRAM latency (1-10, higher = more DRAM traffic) |
| `allow_tma` | `bool \| None` | `None` | Whether to use Tensor Memory Accelerator (Hopper+) |
| `memory_order` | `MemoryOrder` | `WEAK` | Memory ordering constraints |
| `memory_scope` | `MemoryScope` | `NONE` | Memory scope for visibility |
| `padding_mode` | `PaddingMode \| None` | `None` | How to handle out-of-bounds accesses |

**Returns:**
- A `Tile` containing the loaded data

**Description:**

`ct.load()` fetches a contiguous tile of data from global memory. The tile is specified in *tile space* rather than element space. For example, to load a 128×128 tile from a 1024×1024 matrix in 32×32 tiles, you would specify the tile index rather than the element index.

**Tile Shape Constraints:**
All tile dimensions must be powers of 2 (e.g., 32, 64, 128, 256). This ensures efficient memory coalescing and alignment with GPU memory transactions.

**Padding Behavior:**
When a tile extends beyond the array bounds (common when array dimensions aren't divisible by tile size), `padding_mode` determines the value of out-of-bounds elements:

| Padding Mode | Behavior |
|--------------|----------|
| `UNDETERMINED` | Out-of-bounds values are undefined (fastest) |
| `ZERO` | Out-of-bounds elements are zero |
| `NEG_ZERO` | Out-of-bounds elements are negative zero |
| `NAN` | Out-of-bounds elements are NaN |
| `POS_INF` | Out-of-bounds elements are +∞ |
| `NEG_INF` | Out-of-bounds elements are -∞ |

### 1D Load Examples

**Basic Vector Load:**

```python
import cutile as ct

# Global array: 1D vector of 1024 elements
X = ct.array((1024,), ct.float32)

# Each block loads a 128-element tile
def vector_kernel(X, Y):
    tile_size = 128
    pid = ct.bid(0)  # Block index [0, 8)
    
    # Load tile at position pid
    x_tile = ct.load(X, (pid,), (tile_size,))
    
    # Process the tile
    y_tile = x_tile * 2.0
    
    # Store back
    ct.store(Y, (pid,), y_tile)
```

**Handling Padding:**

```python
# Array size: 1000 elements (not divisible by 128)
X = ct.array((1000,), ct.float32)

# Load with zero padding for last tile
x_tile = ct.load(
    X,
    (ct.bid(0),),
    (128,),
    padding_mode=ct.PaddingMode.ZERO  # Out-of-bounds → 0.0
)

# Now we can safely process all 8 tiles
# Tiles 0-6: 128 valid elements
# Tile 7: 72 valid elements + 56 zeros
```

### 2D Load Examples

**Basic Matrix Load:**

```python
# Global matrix: 1024×1024
A = ct.array((1024, 1024), ct.float32)

# Each block loads a 64×64 tile
def matrix_kernel(A, B):
    TM, TN = 64, 64
    
    # Get block position in 2D grid
    i = ct.bid(0)  # Row index [0, 16)
    j = ct.bid(1)  # Column index [0, 16)
    
    # Load tile at block position
    a_tile = ct.load(A, (i, j), (TM, TN))
    
    # Process
    b_tile = process_tile(a_tile)
    
    # Store back
    ct.store(B, (i, j), b_tile)
```

**Loading Different Strides:**

```python
# Load a tile with different dimensions
# Matrix: 2048×1024, tile: 128×64
M = ct.array((2048, 1024), ct.float32)

tile = ct.load(M, (i, j), (128, 64))
# i ranges from 0 to 15 (2048/128)
# j ranges from 0 to 15 (1024/64)
```

### 3D Load Examples

**Tensor Processing:**

```python
# 3D tensor: 256×256×128 (batch×height×width)
T = ct.array((256, 256, 128), ct.float32)

# Load 64×64×32 tiles
def tensor_kernel(T, U):
    TB, TH, TW = 64, 64, 32
    
    b = ct.bid(0)  # Batch index [0, 4)
    h = ct.bid(1)  # Height index [0, 4)
    w = ct.bid(2)  # Width index [0, 4)
    
    # Load 3D tile
    t_tile = ct.load(T, (b, h, w), (TB, TH, TW))
    
    # Process (e.g., convolution)
    u_tile = convolution_3d(t_tile)
    
    # Store result
    ct.store(U, (b, h, w), u_tile)
```

## Storing Tiles

### ct.store()

Writes a tile from on-chip memory back to global memory.

**Syntax:**
```python
ct.store(
    array: Array,
    index: tuple[int, ...],
    tile: Tile,
    *,
    latency: int | None = None,
    allow_tma: bool | None = None,
    memory_order: MemoryOrder = MemoryOrder.WEAK,
    memory_scope: MemoryScope = MemoryScope.NONE
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array` | `Array` | — | The global array to store to |
| `index` | `tuple[int, ...]` | — | Tile space coordinates (where to store) |
| `tile` | `Tile` | — | The tile data to store |
| `latency` | `int \| None` | `None` | Latency hint (1-10) |
| `allow_tma` | `bool \| None` | `None` | Whether to use TMA |
| `memory_order` | `MemoryOrder` | `WEAK` | Memory ordering |
| `memory_scope` | `MemoryScope` | `NONE` | Memory scope |

**Description:**

`ct.store()` writes the tile data back to global memory at the specified tile space location. The tile dimensions must match the shape used when the destination array was tiled (either implicitly or via `TiledView`).

### Store Examples

**Basic 1D Store:**

```python
def saxpy(X, Y, a, Z):
    # Z = a * X + Y
    tile_size = 128
    
    # Load input tiles
    x_tile = ct.load(X, (ct.bid(0),), (tile_size,))
    y_tile = ct.load(Y, (ct.bid(0),), (tile_size,))
    
    # Compute
    z_tile = a * x_tile + y_tile
    
    # Store result
    ct.store(Z, (ct.bid(0),), z_tile)
```

**2D Matrix Store:**

```python
def matrix_add(A, B, C):
    # C = A + B
    TM, TN = 64, 64
    
    i = ct.bid(0)
    j = ct.bid(1)
    
    # Load tiles
    a_tile = ct.load(A, (i, j), (TM, TN))
    b_tile = ct.load(B, (i, j), (TM, TN))
    
    # Add
    c_tile = a_tile + b_tile
    
    # Store result
    ct.store(C, (i, j), c_tile)
```

**Conditional Store with Mask:**

```python
def store_with_mask(A, B, mask_array):
    # Only store elements where mask is True
    TM, TN = 64, 64
    
    i = ct.bid(0)
    j = ct.bid(1)
    
    # Load data and mask
    a_tile = ct.load(A, (i, j), (TM, TN))
    mask_tile = ct.load(mask_array, (i, j), (TM, TN))
    
    # Apply mask: set masked-out elements to zero
    masked_a = a_tile * mask_tile.astype(ct.float32)
    
    # Store
    ct.store(B, (i, j), masked_a)
```

## Irregular Access Patterns

### ct.gather()

Loads elements from non-contiguous memory locations.

**Syntax:**
```python
ct.gather(
    array: Array,
    indices: Tile,
    *,
    mask: Tile | None = None
) -> Tile
```

**Parameters:**
- `array` — The global array to gather from
- `indices` — A tile of integer indices specifying which elements to load
- `mask` — Optional boolean tile; `False` elements are not loaded

**Returns:**
- A tile containing the gathered values

**Description:**

Gather is useful for irregular access patterns like sparse matrices, lookup tables, or indirect addressing. Unlike `ct.load()` which fetches contiguous tiles, `gather()` fetches individual elements scattered throughout memory.

**Examples:**

```python
# Lookup table example
def lookup_table(inputs, table, outputs):
    # inputs: array of indices [0, 256)
    # table: array of 256 values
    
    tile_size = 64
    
    # Load indices
    idx_tile = ct.load(inputs, (ct.bid(0),), (tile_size,))
    
    # Gather values from table using indices
    # idx_tile contains indices into table
    values = ct.gather(table, idx_tile)
    
    # Store results
    ct.store(outputs, (ct.bid(0),), values)
```

```python
# Sparse matrix access (CSR format)
def sparse_csr_matrix_vector(data, indices, indptr, x, y):
    # y = A @ x where A is in CSR format
    # Each block processes a set of rows
    
    rows_per_block = 32
    
    # Get which rows this block handles
    start_row = ct.bid(0) * rows_per_block
    end_row = start_row + rows_per_block
    
    # For each row in this block
    for r in range(rows_per_block):
        row_idx = start_row + r
        
        # Get range of nonzeros in this row
        row_start = ct.gather(indptr, row_idx)
        row_end = ct.gather(indptr, row_idx + 1)
        
        # Load column indices and values
        cols = ct.gather(indices, ct.arange(row_end - row_start) + row_start)
        vals = ct.gather(data, ct.arange(row_end - row_start) + row_start)
        
        # Gather x values at those columns
        x_vals = ct.gather(x, cols)
        
        # Compute dot product
        y[row_idx] = ct.sum(vals * x_vals)
```

**Gather with Mask:**

```python
def selective_gather(array, indices, mask, output):
    # Only load elements where mask is True
    idx_tile = ct.load(indices, (ct.bid(0),), (64,))
    mask_tile = ct.load(mask, (ct.bid(0),), (64,))
    
    # Gather with mask
    values = ct.gather(array, idx_tile, mask=mask_tile)
    
    ct.store(output, (ct.bid(0),), values)
```

### ct.scatter()

Stores elements to non-contiguous memory locations.

**Syntax:**
```python
ct.scatter(
    array: Array,
    indices: Tile,
    values: Tile,
    *,
    mask: Tile | None = None
) -> None
```

**Parameters:**
- `array` — The global array to scatter to
- `indices` — A tile of integer indices specifying where to store
- `values` — A tile of values to store
- `mask` — Optional boolean tile; `False` elements are not stored

**Description:**

Scatter is the inverse of gather—it writes values to scattered memory locations. This is useful for operations like histogram computation, scattering results back to original positions, or building sparse data structures.

**Examples:**

```python
# Histogram example
def histogram(values, hist, num_bins):
    # Compute histogram of values
    tile_size = 64
    
    # Load values
    vals = ct.load(values, (ct.bid(0),), (tile_size,))
    
    # Compute bin indices
    bin_idx = vals % num_bins
    
    # Initialize local histogram
    local_hist = ct.zeros((num_bins,), ct.int32)
    
    # Atomic add to local histogram
    for i in range(tile_size):
        local_hist[bin_idx[i]] += 1
    
    # Scatter to global histogram
    ct.scatter(hist, ct.arange(num_bins), local_hist)
```

```python
# In-place scatter
def add_at_positions(array, indices, values):
    # array[indices] += values
    
    tile_size = 64
    
    # Load indices and values
    idx = ct.load(indices, (ct.bid(0),), (tile_size,))
    vals = ct.load(values, (ct.bid(0),), (tile_size,))
    
    # Load current values at those positions
    current = ct.gather(array, idx)
    
    # Add
    updated = current + vals
    
    # Scatter back
    ct.scatter(array, idx, updated)
```

```python
# Scatter with mask
def masked_scatter(array, indices, values, mask):
    # Only store where mask is True
    idx_tile = ct.load(indices, (ct.bid(0),), (64,))
    val_tile = ct.load(values, (ct.bid(0),), (64,))
    mask_tile = ct.load(mask, (ct.bid(0),), (64,))
    
    # Scatter with mask
    ct.scatter(array, idx_tile, val_tile, mask=mask_tile)
```

## Memory Ordering and Synchronization

### Memory Order

The `memory_order` parameter controls how memory operations are ordered relative to other operations.

**MemoryOrder.WEAK** (default):
- No ordering guarantees beyond normal GPU memory consistency
- Fastest option
- Suitable for most kernels where blocks don't communicate

**MemoryOrder.RELAXED**:
- Slightly stronger ordering
- Useful when some synchronization is needed

**Stronger Ordering**:
- Not typically needed in cuTile kernels
- Can significantly impact performance

### Memory Scope

The `memory_scope` parameter controls the visibility of memory operations.

**MemoryScope.NONE** (default):
- Normal GPU memory scope
- Operations visible to all blocks

**MemoryScope.BLOCK**:
- Operations only visible within block (rarely used)

### Inter-Block Synchronization Example

```python
def producer_consumer(A, B, flag):
    # Block 0 produces, block 1 consumes
    
    if ct.bid(0) == 0:
        # Producer: load and process
        tile = ct.load(A, (0,), (128,))
        result = expensive_computation(tile)
        ct.store(B, (0,), result)
        
        # Write flag with release semantics
        flag[0] = 1
        ct.fence(memory_order=ct.MemoryOrder.RELEASE)
    
    else:
        # Consumer: wait for flag
        while flag[0] == 0:
            pass
        
        # Read flag with acquire semantics
        ct.fence(memory_order=ct.MemoryOrder.ACQUIRE)
        result = ct.load(B, (0,), (128,))
        
        # Process result
        output = process(result)
```

## Performance Optimization

### Latency Hints

The `latency` parameter provides hints to the compiler about DRAM traffic:

| Value | Description | Use Case |
|-------|-------------|----------|
| 1-3 | Low latency | Data likely in cache |
| 4-6 | Medium latency | Moderate DRAM access |
| 7-10 | High latency | Heavy DRAM traffic |

```python
# High-latency load (streaming large dataset)
tile = ct.load(
    large_array,
    (i, j),
    (128, 128),
    latency=9  # Heavy DRAM traffic expected
)

# Low-latency load (reusing cached data)
tile = ct.load(
    small_array,
    (i,),
    (32,),
    latency=2  # Likely cached
)
```

### Tensor Memory Accelerator (TMA)

TMA is available on Hopper (H100) and newer GPUs. It offloads memory transfer work to a dedicated hardware unit.

```python
# Enable TMA for large transfers
tile = ct.load(
    array,
    (i, j),
    (256, 256),
    allow_tma=True  # Use TMA if available
)
```

**When to use TMA:**
- Large tile sizes (>128×128)
- Hopper or newer GPUs
- Memory-bound kernels

**When to avoid TMA:**
- Small tiles (overhead not worth it)
- Older GPU architectures
- Compute-bound kernels

### Load/Store Performance Tips

1. **Tile Size Selection**
   ```python
   # Good: powers of 2, aligned
   tile = ct.load(A, (i, j), (64, 64))
   
   # Avoid: non-power-of-2 dimensions
   # tile = ct.load(A, (i, j), (100, 50))  # Bad
   ```

2. **Maximize Coalescing**
   ```python
   # Good: contiguous access pattern
   for j in range(num_tiles_col):
       tile = ct.load(A, (i, j), (64, 64))  # Row-major
   
   # Good: transposed access
   for i in range(num_tiles_row):
       tile = ct.load(A.T, (j, i), (64, 64))  # Column-major as row-major
   ```

3. **Overlap Computation and Memory**
   ```python
   # Load next tile while computing current
   tile_k = ct.load(A, (i, k), (TM, TK))
   for k in range(1, K_tiles):
       # Prefetch next
       tile_next = ct.load(A, (i, k), (TM, TK), latency=8)
       # Compute current
       accumulator = ct.dot(accumulator, tile_k)
       # Advance
       tile_k = tile_next
   ```

4. **Use Padding Mode Correctly**
   ```python
   # Fast: UNDETERMINED (when you know you won't access padding)
   tile = ct.load(A, (i, j), (128, 128), padding_mode=ct.PaddingMode.UNDETERMINED)
   
   # Safe: ZERO (when you might access padding and need defined behavior)
   tile = ct.load(A, (i, j), (128, 128), padding_mode=ct.PaddingMode.ZERO)
   ```

## Complete Examples

### Example 1: Matrix Multiplication with Load/Store

```python
import cutile as ct

def matmul_kernel(A, B, C, M, N, K):
    """
    Matrix multiplication: C = A @ B
    A: M × K, B: K × N, C: M × N
    """
    # Tile sizes
    TM, TN, TK = 128, 128, 32
    
    # Block position
    i = ct.bid(0)  # Row block of C [0, ceil(M/TM))
    j = ct.bid(1)  # Column block of C [0, ceil(N/TN))
    
    # Accumulator tile
    c_tile = ct.zeros((TM, TN), ct.float32)
    
    # Loop over K dimension
    num_k_tiles = (K + TK - 1) // TK
    for k in range(num_k_tiles):
        # Load tiles from A and B
        a_tile = ct.load(
            A, (i, k), (TM, TK),
            padding_mode=ct.PaddingMode.ZERO
        )
        b_tile = ct.load(
            B, (k, j), (TK, TN),
            padding_mode=ct.PaddingMode.ZERO
        )
        
        # Accumulate
        c_tile = c_tile + ct.dot(a_tile, b_tile)
    
    # Store result
    ct.store(C, (i, j), c_tile)
```

### Example 2: Convolution with Gather

```python
def conv2d_kernel(input, kernel, output):
    """
    2D convolution with gather for im2col-style access
    """
    TILE_SIZE = 64
    KERNEL_SIZE = 3
    
    i = ct.bid(0)
    j = ct.bid(1)
    
    # Load input tile
    in_tile = ct.load(input, (i, j), (TILE_SIZE, TILE_SIZE))
    
    # Generate offsets for 3×3 kernel
    offsets = ct.arange(9)
    di = offsets // 3 - 1  # [-1, 0, 1, -1, 0, 1, ...]
    dj = offsets % 3 - 1   # [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    
    # Gather neighbors
    neighbors = ct.zeros((TILE_SIZE, TILE_SIZE, 9), ct.float32)
    for k in range(9):
        # Gather with padding for boundary
        neighbor = ct.gather(
            input,
            compute_neighbor_index(i, j, di[k], dj[k]),
            mask=create_boundary_mask(TILE_SIZE, di[k], dj[k])
        )
        neighbors[:, :, k] = neighbor
    
    # Load kernel and convolve
    kernel_flat = kernel.reshape(9)
    output_tile = ct.sum(neighbors * kernel_flat, axis=2)
    
    # Store
    ct.store(output, (i, j), output_tile)
```

### Example 3: Softmax with Load/Store

```python
def softmax_kernel(input, output):
    """
    Row-wise softmax: exp(x) / sum(exp(x))
    """
    TILE_M = 64
    TILE_N = 1024
    
    i = ct.bid(0)
    
    # Load tile
    x_tile = ct.load(input, (i, 0), (TILE_M, TILE_N))
    
    # Find max (for numerical stability)
    x_max = ct.max(x_tile, axis=1, keepdims=True)
    
    # Exp and sum
    exp_x = ct.exp(x_tile - x_max)
    exp_sum = ct.sum(exp_x, axis=1, keepdims=True)
    
    # Normalize
    softmax = exp_x / exp_sum
    
    # Store
    ct.store(output, (i, 0), softmax)
```

## Summary Table

| Operation | Purpose | Input | Output |
|-----------|---------|-------|--------|
| `ct.bid(axis)` | Get block index | axis | Tile (scalar) |
| `ct.num_blocks(axis)` | Get block count | axis | Tile (scalar) |
| `ct.load(array, index, shape)` | Load contiguous tile | array, position, shape | Tile |
| `ct.store(array, index, tile)` | Store contiguous tile | array, position, tile | None |
| `ct.gather(array, indices)` | Load scattered elements | array, indices | Tile |
| `ct.scatter(array, indices, values)` | Store scattered elements | array, indices, values | None |

## Best Practices

1. **Always use power-of-2 tile dimensions** for optimal performance
2. **Use `padding_mode=ZERO` when array size isn't divisible by tile size**
3. **Prefer `TiledView` for complex multi-tile patterns**
4. **Use `latency` hints to guide the compiler**
5. **Enable TMA for large transfers on Hopper+ GPUs**
6. **Gather/scatter should be used sparingly—they're slower than load/store**
7. **Consider memory ordering only when blocks synchronize**
8. **Maximize memory coalescing by accessing contiguous memory**
