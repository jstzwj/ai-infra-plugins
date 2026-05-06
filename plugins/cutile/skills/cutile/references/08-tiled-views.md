# Tiled Views

This chapter provides a comprehensive guide to `TiledView`, a powerful abstraction for working with tile spaces in cuTile. Tiled views provide a structured way to iterate over global arrays in tiles, handling padding, bounds checking, and coordinate translation automatically.

## Overview

A `TiledView` represents the *tile space* of a global array—the grid of tiles that partition the array. Instead of manually computing which tile to load and how to handle boundaries, you create a tiled view that encapsulates this information.

**Key Concepts:**

- **Element Space**: The original array coordinates (e.g., a 1024×1024 matrix)
- **Tile Space**: The grid of tiles (e.g., 16×16 tiles of size 64×64)
- **TiledView**: An object that maps between element space and tile space

## Creating Tiled Views

### Basic Syntax

```python
tiled_view = array.tiled_view(
    tile_shape: tuple[int, ...],
    padding_mode: PaddingMode = PaddingMode.UNDETERMINED
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tile_shape` | `tuple[int, ...]` | — | Dimensions of each tile (must be powers of 2) |
| `padding_mode` | `PaddingMode` | `UNDETERMINED` | How to handle out-of-bounds accesses |

**Returns:**
- A `TiledView` object representing the tile space

### Creating 1D Tiled Views

```python
import cutile as ct

# Create a 1D array
X = ct.array((1024,), ct.float32)

# Create a tiled view with 128-element tiles
X_view = X.tiled_view((128,))

# The view now represents 8 tiles (1024 / 128 = 8)
print(X_view.num_tiles)  # (8,)
print(X_view.num_tiles(0))  # 8
```

### Creating 2D Tiled Views

```python
# Create a 2D array
A = ct.array((1024, 2048), ct.float32)

# Create a tiled view with 64×128 tiles
A_view = A.tiled_view((64, 128))

# The view represents 16×16 tiles
print(A_view.num_tiles)  # (16, 16)
print(A_view.num_tiles(0))  # 16 (rows)
print(A_view.num_tiles(1))  # 16 (columns)
```

### Creating 3D Tiled Views

```python
# Create a 3D tensor
T = ct.array((256, 256, 128), ct.float32)

# Create a tiled view with 64×64×32 tiles
T_view = T.tiled_view((64, 64, 32))

# The view represents 4×4×4 tiles
print(T_view.num_tiles)  # (4, 4, 4)
```

### Padding Mode Specification

```python
# Array with non-divisible dimensions
A = ct.array((1000, 1000), ct.float32)

# Create view with zero padding
A_view = A.tiled_view(
    (128, 128),
    padding_mode=ct.PaddingMode.ZERO
)

# num_tiles: (8, 8) - ceil(1000/128) = 8
# Last tile in each dimension will have padding
```

## TiledView Properties and Methods

### Properties

#### num_tiles

Returns the total number of tiles along each dimension as a tuple.

```python
view = array.tiled_view((64, 128))
print(view.num_tiles)  # (16, 8)
```

#### tile_shape

Returns the shape of each tile.

```python
view = array.tiled_view((64, 128))
print(view.tile_shape)  # (64, 128)
```

### Methods

#### num_tiles(axis)

Returns the number of tiles along a specific axis.

**Syntax:**
```python
view.num_tiles(axis: int) -> int
```

**Parameters:**
- `axis` (int) — The dimension to query

**Returns:**
- Number of tiles along the specified axis

**Examples:**

```python
view = array.tiled_view((64, 128, 32))

# Get tiles along each axis
n_x = view.num_tiles(0)  # e.g., 16
n_y = view.num_tiles(1)  # e.g., 8
n_z = view.num_tiles(2)  # e.g., 4
```

#### load(tile_index)

Loads a tile at the specified tile space coordinates.

**Syntax:**
```python
view.load(tile_index: tuple[int, ...]) -> Tile
```

**Parameters:**
- `tile_index` — Tuple of tile coordinates (e.g., `(i, j)` for 2D)

**Returns:**
- A `Tile` containing the loaded data

**Description:**

This method is equivalent to `ct.load(array, tile_index, tile_shape)` but uses the tile shape and padding mode from the view.

**Examples:**

```python
view = A.tiled_view((64, 64))

# Load tile at position (2, 3) in tile space
tile = view.load((2, 3))

# Same as:
# tile = ct.load(A, (2, 3), (64, 64))
```

#### store(tile_index, tile)

Stores a tile at the specified tile space coordinates.

**Syntax:**
```python
view.store(tile_index: tuple[int, ...], tile: Tile) -> None
```

**Parameters:**
- `tile_index` — Tuple of tile coordinates
- `tile` — The tile data to store

**Description:**

This method is equivalent to `ct.store(array, tile_index, tile)` but uses the tile shape from the view.

**Examples:**

```python
view = B.tiled_view((64, 64))

# Store tile at position (0, 0)
result_tile = compute_something()
view.store((0, 0), result_tile)

# Same as:
# ct.store(B, (0, 0), result_tile)
```

## Padding Modes

When array dimensions are not divisible by the tile shape, the last tile along each dimension will extend beyond the array bounds. The `padding_mode` parameter determines how these out-of-bounds elements are handled.

### Available Padding Modes

| Padding Mode | Description | Use Case |
|--------------|-------------|----------|
| `UNDETERMINED` | Out-of-bounds values are undefined | Fastest; use when you know padding won't be accessed |
| `ZERO` | Out-of-bounds elements are zero | Safe default; prevents undefined behavior |
| `NEG_ZERO` | Out-of-bounds elements are negative zero | Mathematical operations requiring signed zero |
| `NAN` | Out-of-bounds elements are NaN | Floating-point operations needing NaN propagation |
| `POS_INF` | Out-of-bounds elements are +∞ | Max-reduction operations |
| `NEG_INF` | Out-of-bounds elements are -∞ | Min-reduction operations |

### Padding Mode Examples

**UNDETERMINED (Fastest):**

```python
A = ct.array((1000,), ct.float32)
view = A.tiled_view(
    (128,),
    padding_mode=ct.PaddingMode.UNDETERMINED
)

# Tiles 0-6: 128 valid elements each
# Tile 7: 72 valid elements + 56 UNDETERMINED
#
# Only use this if you're sure you won't access the padding!
# For example, if you mask out the padding:

i = ct.bid(0)
tile = view.load((i,))
valid_count = 128 if i < 7 else 72

# Only process valid elements
for j in range(valid_count):
    result[j] = process(tile[j])
```

**ZERO (Safe Default):**

```python
A = ct.array((1000,), ct.float32)
view = A.tiled_view(
    (128,),
    padding_mode=ct.PaddingMode.ZERO
)

# Tiles 0-6: 128 valid elements each
# Tile 7: 72 valid elements + 56 zeros
#
# Safe to process entire tile without bounds checking

i = ct.bid(0)
tile = view.load((i,))

# Can process all 128 elements safely
result = tile * 2.0  # Padding elements stay zero
```

**NAN (Error Detection):**

```python
A = ct.array((1000,), ct.float32)
view = A.tiled_view(
    (128,),
    padding_mode=ct.PaddingMode.NAN
)

# Padding elements are NaN
# Useful for detecting accidental padding access

tile = view.load((7,))
result = ct.sum(tile)  # Will be NaN if padding is accessed

# Check for NaN
if ct.isnan(result):
    # Handle padding case
    pass
```

**POS_INF (Max Reduction):**

```python
A = ct.array((1000,), ct.float32)
view = A.tiled_view(
    (128,),
    padding_mode=ct.PaddingMode.POS_INF
)

# Padding elements are +∞
# Useful for finding maximum values

tile = view.load((i,))
max_val = ct.min(tile)  # Min ignores +∞ padding
```

**NEG_INF (Min Reduction):**

```python
A = ct.array((1000,), ct.float32)
view = A.tiled_view(
    (128,),
    padding_mode=ct.PaddingMode.NEG_INF
)

# Padding elements are -∞
# Useful for finding minimum values

tile = view.load((i,))
min_val = ct.max(tile)  # Max ignores -∞ padding
```

## Tile Space Concepts

### Relationship Between Element Space and Tile Space

**Element Space**: The original array coordinates
- Array shape: `(M, N)` = `(1024, 2048)`
- Element at position: `(i, j)` where `0 ≤ i < 1024`, `0 ≤ j < 2048`

**Tile Space**: The grid of tiles
- Tile shape: `(TM, TN)` = `(64, 128)`
- Number of tiles: `(ceil(1024/64), ceil(2048/128))` = `(16, 16)`
- Tile at position: `(ti, tj)` where `0 ≤ ti < 16`, `0 ≤ tj < 16`

**Mapping:**

```python
# Element → Tile
tile_i = element_i // TM
tile_j = element_j // TN

# Tile → Element (start of tile)
element_i_start = tile_i * TM
element_j_start = tile_j * TN

# Element → Position within tile
local_i = element_i % TM
local_j = element_j % TN
```

### Number of Tiles Calculation

The number of tiles along each dimension is computed as:

```python
num_tiles_dim = ceil(array_shape_dim / tile_shape_dim)
                = (array_shape_dim + tile_shape_dim - 1) // tile_shape_dim
```

**Examples:**

```python
# Exact division
array_shape = (1024, 2048)
tile_shape = (64, 128)
num_tiles = (1024//64, 2048//128) = (16, 16)

# Non-exact division
array_shape = (1000, 1000)
tile_shape = (128, 128)
num_tiles = ((1000+127)//128, (1000+127)//128) = (8, 8)
```

### Iterating Over Tile Space

The combination of `ct.bid()` and `view.num_tiles()` allows you to iterate over the tile space:

```python
view = A.tiled_view((64, 64))

# Each block handles one tile
i = ct.bid(0)  # [0, view.num_tiles(0))
j = ct.bid(1)  # [0, view.num_tiles(1))

# Load this block's tile
tile = view.load((i, j))
```

For multi-tile operations (e.g., matrix multiplication with tiling):

```python
A_view = A.tiled_view((TM, TK))
B_view = B.tiled_view((TK, TN))

i = ct.bid(0)
j = ct.bid(1)

# Loop over K dimension
for k in range(A_view.num_tiles(1)):  # or B_view.num_tiles(0)
    a_tile = A_view.load((i, k))
    b_tile = B_view.load((k, j))
    # Compute...
```

## Complete Examples

### Example 1: Basic Matrix Addition

```python
import cutile as ct

def matrix_add_kernel(A, B, C):
    """
    C = A + B using tiled views
    A, B, C: 1024×1024 matrices
    """
    # Create tiled views
    A_view = A.tiled_view((64, 64), padding_mode=ct.PaddingMode.ZERO)
    B_view = B.tiled_view((64, 64), padding_mode=ct.PaddingMode.ZERO)
    C_view = C.tiled_view((64, 64))
    
    # Get tile position
    i = ct.bid(0)  # [0, 16)
    j = ct.bid(1)  # [0, 16)
    
    # Load tiles
    a_tile = A_view.load((i, j))
    b_tile = B_view.load((i, j))
    
    # Add
    c_tile = a_tile + b_tile
    
    # Store
    C_view.store((i, j), c_tile)
```

### Example 2: Matrix Multiplication

```python
def matmul_kernel(A, B, C):
    """
    C = A @ B using tiled views
    A: M×K, B: K×N, C: M×N
    """
    TM, TN, TK = 128, 128, 32
    
    # Create tiled views
    A_view = A.tiled_view((TM, TK), padding_mode=ct.PaddingMode.ZERO)
    B_view = B.tiled_view((TK, TN), padding_mode=ct.PaddingMode.ZERO)
    C_view = C.tiled_view((TM, TN))
    
    # Get output tile position
    i = ct.bid(0)  # [0, C_view.num_tiles(0))
    j = ct.bid(1)  # [0, C_view.num_tiles(1))
    
    # Initialize accumulator
    c_tile = ct.zeros((TM, TN), ct.float32)
    
    # Loop over K dimension in tiles
    for k in range(A_view.num_tiles(1)):
        # Load tiles
        a_tile = A_view.load((i, k))
        b_tile = B_view.load((k, j))
        
        # Compute outer product and accumulate
        c_tile = c_tile + ct.dot(a_tile, b_tile)
    
    # Store result
    C_view.store((i, j), c_tile)
```

### Example 3: 2D Convolution

```python
def conv2d_kernel(input, output, kernel):
    """
    2D convolution using tiled views
    """
    TILE_SIZE = 64
    KERNEL_SIZE = 3
    
    # Create tiled view for input
    input_view = input.tiled_view(
        (TILE_SIZE, TILE_SIZE),
        padding_mode=ct.PaddingMode.ZERO
    )
    output_view = output.tiled_view(
        (TILE_SIZE, TILE_SIZE),
        padding_mode=ct.PaddingMode.ZERO
    )
    
    i = ct.bid(0)
    j = ct.bid(1)
    
    # Load center tile
    center = input_view.load((i, j))
    
    # Initialize output
    out_tile = ct.zeros((TILE_SIZE, TILE_SIZE), ct.float32)
    
    # Load neighboring tiles for convolution
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            ni = i + di
            nj = j + dj
            
            # Check bounds (padding mode handles out-of-bounds)
            neighbor = input_view.load((ni, nj))
            
            # Apply kernel weight and accumulate
            weight = kernel[di+1, dj+1]
            out_tile = out_tile + weight * neighbor
    
    # Store result
    output_view.store((i, j), out_tile)
```

### Example 4: Row-wise Operations

```python
def softmax_kernel(input, output):
    """
    Row-wise softmax: exp(x) / sum(exp(x))
    input: M×N matrix
    output: M×N matrix
    """
    TILE_M = 64
    TILE_N = 1024
    
    # Create tiled views
    input_view = input.tiled_view((TILE_M, TILE_N))
    output_view = output.tiled_view((TILE_M, TILE_N))
    
    i = ct.bid(0)  # Row tile [0, num_row_tiles)
    
    # Load row tile
    x_tile = input_view.load((i, 0))
    
    # Find max for numerical stability
    x_max = ct.max(x_tile, axis=1, keepdims=True)
    
    # Compute exp(x - max)
    exp_x = ct.exp(x_tile - x_max)
    
    # Sum and normalize
    exp_sum = ct.sum(exp_x, axis=1, keepdims=True)
    softmax = exp_x / exp_sum
    
    # Store
    output_view.store((i, 0), softmax)
```

### Example 5: Transpose

```python
def transpose_kernel(A, B):
    """
    B = A.T using tiled views
    A: M×N, B: N×M
    """
    TM, TN = 128, 128
    
    # Create tiled views
    A_view = A.tiled_view((TM, TN), padding_mode=ct.PaddingMode.ZERO)
    B_view = B.tiled_view((TN, TM))
    
    # We're transposing, so we read from A and write to transposed position in B
    i = ct.bid(0)  # Row in A
    j = ct.bid(1)  # Column in A
    
    # Load tile from A
    a_tile = A_view.load((i, j))
    
    # Transpose the tile
    b_tile = ct.transpose(a_tile)
    
    # Store to transposed position in B
    # Note: We store to (j, i) because we transposed
    B_view.store((j, i), b_tile)
```

### Example 6: Matrix-Matrix Multiplication with Accumulation

```python
def matmul_accum_kernel(A, B, C, alpha, beta):
    """
    C = alpha * (A @ B) + beta * C using tiled views
    """
    TM, TN, TK = 128, 128, 32
    
    # Create tiled views
    A_view = A.tiled_view((TM, TK), padding_mode=ct.PaddingMode.ZERO)
    B_view = B.tiled_view((TK, TN), padding_mode=ct.PaddingMode.ZERO)
    C_view = C.tiled_view((TM, TN), padding_mode=ct.PaddingMode.ZERO)
    
    i = ct.bid(0)
    j = ct.bid(1)
    
    # Load current C tile (for beta * C)
    c_tile = C_view.load((i, j))
    accumulator = beta * c_tile
    
    # Matrix multiplication
    for k in range(A_view.num_tiles(1)):
        a_tile = A_view.load((i, k))
        b_tile = B_view.load((k, j))
        accumulator = accumulator + alpha * ct.dot(a_tile, b_tile)
    
    # Store result
    C_view.store((i, j), accumulator)
```

### Example 7: Batched Operations

```python
def batched_matmul_kernel(A, B, C):
    """
    Batched matrix multiplication
    A: batch×M×K
    B: batch×K×N
    C: batch×M×N
    """
    TM, TN, TK = 64, 64, 32
    
    # Create tiled views
    # Note: We tile over the last two dimensions, keeping batch intact
    A_view = A.tiled_view((TM, TK), padding_mode=ct.PaddingMode.ZERO)
    B_view = B.tiled_view((TK, TN), padding_mode=ct.PaddingMode.ZERO)
    C_view = C.tiled_view((TM, TN))
    
    # Grid is 3D: batch × M_tiles × N_tiles
    batch = ct.bid(0)
    i = ct.bid(1)
    j = ct.bid(2)
    
    # Get the specific batch slice
    A_batch = A[batch]  # M×K
    B_batch = B[batch]  # K×N
    C_batch = C[batch]  # M×N
    
    # Create per-batch tiled views
    A_batch_view = A_batch.tiled_view((TM, TK), padding_mode=ct.PaddingMode.ZERO)
    B_batch_view = B_batch.tiled_view((TK, TN), padding_mode=ct.PaddingMode.ZERO)
    C_batch_view = C_batch.tiled_view((TM, TN))
    
    # Accumulator
    c_tile = ct.zeros((TM, TN), ct.float32)
    
    # Multiply
    for k in range(A_batch_view.num_tiles(1)):
        a_tile = A_batch_view.load((i, k))
        b_tile = B_batch_view.load((k, j))
        c_tile = c_tile + ct.dot(a_tile, b_tile)
    
    # Store
    C_batch_view.store((i, j), c_tile)
```

### Example 8: Reduction Using Tiled Views

```python
def sum_reduction_kernel(input, output):
    """
    Sum reduction along rows
    input: M×N
    output: M×1
    """
    TILE_M = 64
    TILE_N = 1024
    
    # Create tiled views
    input_view = input.tiled_view((TILE_M, TILE_N))
    
    i = ct.bid(0)  # Row tile
    
    # Load tile
    tile = input_view.load((i, 0))
    
    # Sum along columns
    row_sums = ct.sum(tile, axis=1, keepdims=True)
    
    # Store (need to handle inter-block reduction for final result)
    # For simplicity, storing partial sums
    ct.store(output, (i, 0), row_sums)
```

## TiledView vs Direct Load/Store

### When to Use TiledView

**Use TiledView when:**
- You need to iterate over multiple tiles
- You want to handle padding automatically
- You need to query the number of tiles
- Your code is cleaner with view abstraction

```python
# Clean, readable code with TiledView
view = A.tiled_view((64, 64))
for i in range(view.num_tiles(0)):
    for j in range(view.num_tiles(1)):
        tile = view.load((i, j))
        process(tile)
```

### When to Use Direct Load/Store

**Use direct ct.load/ct.store when:**
- You only load/store one tile
- You need explicit control over padding per-load
- You're computing tile indices dynamically

```python
# Direct load for simple case
tile = ct.load(A, (i, j), (64, 64), padding_mode=ct.PaddingMode.ZERO)
```

## Performance Considerations

### Tile Size Selection

**General Guidelines:**
- Larger tiles = better utilization, more register pressure
- Smaller tiles = less register pressure, more overhead
- Powers of 2 are required for efficient memory access

```python
# Good: Balanced tile size
view = A.tiled_view((128, 128))  # 16K elements per tile

# Too small: Excessive overhead
view = A.tiled_view((16, 16))  # 256 elements per tile

# Too large: Register spillage
view = A.tiled_view((256, 256))  # 65K elements per tile
```

### Padding Mode Performance

```python
# Fastest: UNDETERMINED (no cost for padding)
view = A.tiled_view((128, 128), padding_mode=ct.PaddingMode.UNDETERMINED)

# Fast: ZERO (minimal cost)
view = A.tiled_view((128, 128), padding_mode=ct.PaddingMode.ZERO)

# Slower: NAN, INF (requires special value handling)
view = A.tiled_view((128, 128), padding_mode=ct.PaddingMode.NAN)
```

### Reusing Tiled Views

```python
# Good: Create view once, reuse many times
A_view = A.tiled_view((128, 128))
for i in range(A_view.num_tiles(0)):
    tile = A_view.load((i, 0))
    process(tile)

# Avoid: Recreating view repeatedly
for i in range(num_tiles):
    view = A.tiled_view((128, 128))  # Unnecessary overhead
    tile = view.load((i, 0))
```

## Common Patterns

### Pattern 1: Iterate Over All Tiles

```python
view = A.tiled_view((64, 64))

# In a kernel, each block handles one tile
i = ct.bid(0)
j = ct.bid(1)

tile = view.load((i, j))
result = process(tile)
view.store((i, j), result)
```

### Pattern 2: Loop Over One Dimension

```python
A_view = A.tiled_view((TM, TK))
B_view = B.tiled_view((TK, TN))

i = ct.bid(0)
j = ct.bid(1)

for k in range(A_view.num_tiles(1)):
    a_tile = A_view.load((i, k))
    b_tile = B_view.load((k, j))
    # Accumulate...
```

### Pattern 3: Check Boundary Tiles

```python
view = A.tiled_view((128, 128))

i = ct.bid(0)
is_last_row = (i == view.num_tiles(0) - 1)
is_last_col = (j == view.num_tiles(1) - 1)

if is_last_row or is_last_col:
    # Handle boundary
    tile = view.load((i, j))  # Padding applies automatically
else:
    # Interior tile, no padding
    tile = view.load((i, j))
```

### Pattern 4: Multi-Level Tiling

```python
# Outer tile level
outer_view = A.tiled_view((256, 256))
# Inner tile level
inner_view = A.tiled_view((64, 64))

oi = ct.bid(0)  # Outer tile index
oj = ct.bid(1)

# Process inner tiles within outer tile
for ii in range(4):  # 256/64 = 4
    for ij in range(4):
        i = oi * 4 + ii
        j = oj * 4 + ij
        tile = inner_view.load((i, j))
        process(tile)
```

## Troubleshooting

### Issue: Out-of-Bounds Access

**Problem:** Accessing elements beyond array bounds.

**Solution:** Use appropriate padding mode.

```python
# Wrong: UNDETERMINED with potential padding access
view = A.tiled_view((128, 128), padding_mode=ct.PaddingMode.UNDETERMINED)
tile = view.load((i, j))  # May have undefined values

# Correct: ZERO padding for defined behavior
view = A.tiled_view((128, 128), padding_mode=ct.PaddingMode.ZERO)
tile = view.load((i, j))  # Padding elements are zero
```

### Issue: Mismatched Tile Shapes

**Problem:** Tile shape doesn't match between load and store.

**Solution:** Use the same tiled view or ensure shapes match.

```python
# Wrong: Different tile shapes
A_view = A.tiled_view((64, 128))
B_view = B.tiled_view((128, 64))  # Mismatch!
tile = A_view.load((0, 0))
B_view.store((0, 0), tile)  # Error: shape mismatch

# Correct: Same tile shapes
A_view = A.tiled_view((64, 64))
B_view = B.tiled_view((64, 64))
tile = A_view.load((0, 0))
B_view.store((0, 0), tile)  # OK
```

### Issue: Incorrect num_tiles Usage

**Problem:** Using num_tiles() instead of num_tiles property.

```python
# Wrong
num = view.num_tiles()  # Error: not callable

# Correct
num = view.num_tiles  # Returns tuple
num_i = view.num_tiles(0)  # Returns int for axis 0
```

## Summary Table

| Feature | TiledView | Direct Load/Store |
|---------|-----------|-------------------|
| **Syntax** | `view.load((i, j))` | `ct.load(array, (i, j), shape)` |
| **Tile Shape** | Specified once at creation | Specified per-load |
| **Padding** | Configured at creation | Configured per-load |
| **Num Tiles** | `view.num_tiles` | Manual computation |
| **Use Case** | Multi-tile operations | Single-tile operations |

## Best Practices

1. **Always specify padding_mode** when array dimensions might not be divisible by tile size
2. **Prefer ZERO padding** for safety unless you're certain padding won't be accessed
3. **Reuse tiled views** rather than recreating them
4. **Use num_tiles(axis)** in loops instead of hardcoding values
5. **Keep tile shapes as powers of 2** for optimal performance
6. **Match tile shapes** between corresponding input and output views
7. **Consider register pressure** when choosing tile sizes
8. **Use TiledView for complex iteration patterns**, direct load/store for simple cases
