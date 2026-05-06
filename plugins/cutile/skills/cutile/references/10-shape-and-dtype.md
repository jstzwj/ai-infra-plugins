# Shape and Data Type Manipulation

This chapter covers operations for manipulating tensor shapes and data types in cuTile. These operations form the foundation for preparing data layouts and type conversions needed in computation kernels.

## Overview

Shape and dtype manipulation operations in cuTile follow NumPy-like semantics but are specialized for tensor core operations where all dimensions must be powers of 2. These operations include:

- **Concatenation**: Combining tiles along specified axes
- **Broadcasting**: Expanding tiles to larger shapes following compatibility rules
- **Dimension manipulation**: Adding, removing, or reordering dimensions
- **Reshaping**: Changing the shape while preserving element count
- **Type conversion**: Converting between different data types

## Concatenation Operations

### `ct.cat(tiles, axis)`

Concatenates a sequence of tiles along a specified axis. All tiles must have identical shapes except along the concatenation dimension.

**Signature:**
```python
ct.cat(tiles: tuple[Tile, ...], axis: int) -> Tile
```

**Parameters:**
- `tiles`: Tuple of tiles to concatenate
- `axis`: Dimension along which to concatenate (0-indexed)

**Requirements:**
- All tiles must have the same number of dimensions
- All tiles must have identical shapes except along the `axis` dimension
- At least two tiles must be provided
- Result shape along `axis` equals sum of input shapes along `axis`

**Examples:**

**Horizontal concatenation (axis=1):**
```python
import cutile as ct
import torch

# Create two 32×32 tiles
a = ct.arange(32, dtype=ct.float32).reshape(1, 32).broadcast_to((32, 32))
b = ct.arange(32, 64, dtype=ct.float32).reshape(1, 32).broadcast_to((32, 32))

# Concatenate horizontally: (32, 32) cat (32, 32) → (32, 64)
result = ct.cat((a, b), axis=1)
print(result.shape)  # (32, 64)

# Verify values in each half
# result[:, 0:32] contains values from a
# result[:, 32:64] contains values from b
```

**Vertical concatenation (axis=0):**
```python
# Create two 32×32 tiles
a = ct.full((32, 32), 1.0, dtype=ct.float32)
b = ct.full((32, 32), 2.0, dtype=ct.float32)

# Concatenate vertically: (32, 32) cat (32, 32) → (64, 32)
result = ct.cat((a, b), axis=0)
print(result.shape)  # (64, 32)

# Verify values in each half
# result[0:32, :] contains 1.0
# result[32:64, :] contains 2.0
```

**Concatenating multiple tiles:**
```python
# Concatenate 4 tiles along axis 0
tiles = (
    ct.full((16, 32), 1.0, dtype=ct.float32),
    ct.full((16, 32), 2.0, dtype=ct.float32),
    ct.full((16, 32), 3.0, dtype=ct.float32),
    ct.full((16, 32), 4.0, dtype=ct.float32),
)

result = ct.cat(tiles, axis=0)
print(result.shape)  # (64, 32)
```

**3D concatenation:**
```python
# Concatenate along batch dimension
a = ct.randn((8, 32, 32), dtype=ct.float32)
b = ct.randn((8, 32, 32), dtype=ct.float32)

# Concatenate batch: (8, 32, 32) cat (8, 32, 32) → (16, 32, 32)
batch = ct.cat((a, b), axis=0)
print(batch.shape)  # (16, 32, 32)
```

**Common use case - matrix assembly:**
```python
def create_block_matrix():
    """Assemble a block matrix from four quadrants."""
    # Create four 16×16 blocks
    q11 = ct.full((16, 16), 1.0, dtype=ct.float32)  # Top-left
    q12 = ct.full((16, 16), 2.0, dtype=ct.float32)  # Top-right
    q21 = ct.full((16, 16), 3.0, dtype=ct.float32)  # Bottom-left
    q22 = ct.full((16, 16), 4.0, dtype=ct.float32)  # Bottom-right
    
    # Assemble rows
    row0 = ct.cat((q11, q12), axis=1)  # (16, 32)
    row1 = ct.cat((q21, q22), axis=1)  # (16, 32)
    
    # Assemble final matrix
    matrix = ct.cat((row0, row1), axis=0)  # (32, 32)
    return matrix
```

## Broadcasting Operations

### `ct.broadcast_to(tile, shape)`

Broadcasts a tile to a specified shape following NumPy broadcasting rules. Broadcasting is a memory-efficient operation that creates a view of the original tile with expanded dimensions.

**Signature:**
```python
ct.broadcast_to(tile: Tile, shape: tuple[int, ...]) -> Tile
```

**Parameters:**
- `tile`: Input tile to broadcast
- `shape`: Target shape tuple

**Broadcasting Rules:**
1. Dimensions are aligned from the trailing (rightmost) dimension
2. Two dimensions are compatible when:
   - They are equal, OR
   - One of them is 1
3. Missing dimensions are prepended with size 1

**Requirements:**
- All resulting dimensions must be powers of 2
- Broadcast must be unambiguous (no dimension > 1 trying to broadcast to different sizes)

**Examples:**

**Scalar to 2D:**
```python
# Create a scalar tile
scalar = ct.full((), 5.0, dtype=ct.float32)

# Broadcast to 32×32 matrix
matrix = ct.broadcast_to(scalar, (32, 32))
print(matrix.shape)  # (32, 32)
# All elements are 5.0
```

**1D to 2D (row broadcasting):**
```python
# Create a row vector (1, 32)
row = ct.arange(32, dtype=ct.float32).reshape((1, 32))

# Broadcast to (16, 32) - replicate row 16 times
matrix = ct.broadcast_to(row, (16, 32))
print(matrix.shape)  # (16, 32)
# Each row is identical: [0, 1, 2, ..., 31]
```

**1D to 2D (column broadcasting):**
```python
# Create a column vector (32, 1)
col = ct.arange(32, dtype=ct.float32).reshape((32, 1))

# Broadcast to (32, 16) - replicate column 16 times
matrix = ct.broadcast_to(col, (32, 16))
print(matrix.shape)  # (32, 16)
# Each column is identical: [0, 1, 2, ..., 31]^T
```

**Higher-dimensional broadcasting:**
```python
# Broadcast 2D to 4D
tile = ct.full((1, 32, 1, 64), 1.0, dtype=ct.float32)

# Broadcast to (8, 32, 16, 64)
result = ct.broadcast_to(tile, (8, 32, 16, 64))
print(result.shape)  # (8, 32, 16, 64)
```

**Broadcasting in arithmetic operations:**
```python
# Element-wise operations use implicit broadcasting
a = ct.full((32, 1), 2.0, dtype=ct.float32)
b = ct.full((1, 32), 3.0, dtype=ct.float32)

# Result is (32, 32)
# Each element [i,j] = a[i,0] * b[0,j] = 2.0 * 3.0 = 6.0
c = a * b
print(c.shape)  # (32, 32)
```

**Broadcasting compatibility check:**
```python
def can_broadcast(shape1, shape2):
    """Check if two shapes are broadcastable."""
    # Align from right
    ndim = max(len(shape1), len(shape2))
    s1 = (1,) * (ndim - len(shape1)) + shape1
    s2 = (1,) * (ndim - len(shape2)) + shape2
    
    # Check compatibility
    for d1, d2 in zip(s1, s2):
        if d1 != d2 and d1 != 1 and d2 != 1:
            return False
    return True

# Examples
print(can_broadcast((32, 1), (1, 32)))   # True
print(can_broadcast((16, 32), (1, 32)))  # True
print(can_broadcast((32, 16), (32, 1)))  # True
print(can_broadcast((32, 32), (16, 32))) # False - incompatible
```

**Use case - adding bias to batches:**
```python
def add_bias(hidden, bias):
    """
    Add bias vector to hidden states.
    
    Args:
        hidden: (batch_size, seq_len, hidden_dim)
        bias: (hidden_dim,)
    
    Returns:
        (batch_size, seq_len, hidden_dim)
    """
    # Reshape bias to (1, 1, hidden_dim)
    bias_reshaped = bias.reshape((1, 1, -1))
    
    # Broadcast and add
    return hidden + bias_reshaped

# Example
hidden = ct.randn((8, 64, 128), dtype=ct.float32)
bias = ct.randn((128,), dtype=ct.float32)
result = add_bias(hidden, bias)
print(result.shape)  # (8, 64, 128)
```

## Dimension Manipulation

### `ct.expand_dims(tile, axis)`

Inserts a new axis of size 1 at the specified position, increasing the dimensionality of the tile by 1.

**Signature:**
```python
ct.expand_dims(tile: Tile, axis: int) -> Tile
```

**Parameters:**
- `tile`: Input tile
- `axis`: Position where new axis is inserted (0-indexed, can be negative)

**Requirements:**
- `axis` must be in range `[-ndim-1, ndim]` where `ndim` is current dimensionality

**Examples:**

**Convert 1D to 2D (row vector):**
```python
# Create 1D tile
vec = ct.arange(32, dtype=ct.float32)
print(vec.shape)  # (32,)

# Convert to row vector (1, 32)
row = ct.expand_dims(vec, axis=0)
print(row.shape)  # (1, 32)
```

**Convert 1D to 2D (column vector):**
```python
# Convert to column vector (32, 1)
col = ct.expand_dims(vec, axis=1)
print(col.shape)  # (32, 1)
```

**Add batch dimension:**
```python
# Create 2D tile
matrix = ct.randn((32, 32), dtype=ct.float32)

# Add batch dimension at front
batched = ct.expand_dims(matrix, axis=0)
print(batched.shape)  # (1, 32, 32)

# Can broadcast to larger batch
batch = ct.broadcast_to(batched, (16, 32, 32))
print(batch.shape)  # (16, 32, 32)
```

**Using negative axis:**
```python
tile = ct.randn((32, 32), dtype=ct.float32)

# axis=-1 adds dimension at the end
result = ct.expand_dims(tile, axis=-1)
print(result.shape)  # (32, 32, 1)

# axis=-2 adds dimension before the last
result = ct.expand_dims(tile, axis=-2)
print(result.shape)  # (32, 1, 32)
```

**Multiple expand_dims operations:**
```python
# Build higher-dimensional tensors
scalar = ct.full((), 1.0, dtype=ct.float32)

# Build up to 4D
tensor = ct.expand_dims(scalar, axis=0)   # (1,)
tensor = ct.expand_dims(tensor, axis=0)   # (1, 1)
tensor = ct.expand_dims(tensor, axis=0)   # (1, 1, 1)
tensor = ct.expand_dims(tensor, axis=0)   # (1, 1, 1, 1)
print(tensor.shape)
```

**Use case - preparing for broadcasting:**
```python
def outer_product(a, b):
    """
    Compute outer product of two vectors.
    
    Args:
        a: (M,) vector
        b: (N,) vector
    
    Returns:
        (M, N) outer product
    """
    # Reshape to (M, 1) and (1, N)
    a_col = ct.expand_dims(a, axis=1)  # (M, 1)
    b_row = ct.expand_dims(b, axis=0)  # (1, N)
    
    # Broadcast multiplication
    return a_col * b_row  # (M, N)

# Example
a = ct.arange(32, dtype=ct.float32)
b = ct.arange(16, dtype=ct.float32)
result = outer_product(a, b)
print(result.shape)  # (32, 16)
```

## Reshaping Operations

### `ct.reshape(tile, shape)`

Changes the shape of a tile without changing the underlying data. The total number of elements must remain constant.

**Signature:**
```python
ct.reshape(tile: Tile, shape: tuple[int, ...]) -> Tile
```

**Parameters:**
- `tile`: Input tile to reshape
- `shape`: Target shape tuple (use -1 to infer dimension)

**Requirements:**
- Product of all dimensions in `shape` must equal product of input dimensions
- All dimensions must be powers of 2
- Can use at most one `-1` in shape (inferred from element count)

**Examples:**

**Flatten to 1D:**
```python
# Create 2D tile
matrix = ct.arange(1024, dtype=ct.float32).reshape((32, 32))

# Flatten to 1D
flat = ct.reshape(matrix, (1024,))
print(flat.shape)  # (1024,)

# Or use -1 to infer
flat = ct.reshape(matrix, (-1,))
print(flat.shape)  # (1024,)
```

**1D to 2D:**
```python
# Create 1D tile with 256 elements
vec = ct.arange(256, dtype=ct.float32)

# Reshape to 16×16
matrix = ct.reshape(vec, (16, 16))
print(matrix.shape)  # (16, 16)

# Reshape to 8×32
matrix = ct.reshape(vec, (8, 32))
print(matrix.shape)  # (8, 32)
```

**Infer dimension with -1:**
```python
# Create tile with 4096 elements
tile = ct.arange(4096, dtype=ct.float32)

# Infer height given width
matrix = ct.reshape(tile, (-1, 64))
print(matrix.shape)  # (64, 64) - 4096/64 = 64

# Infer width given height
matrix = ct.reshape(tile, (32, -1))
print(matrix.shape)  # (32, 128) - 4096/32 = 128
```

**3D reshaping:**
```python
# Create 2D tile
matrix = ct.arange(4096, dtype=ct.float32).reshape((64, 64))

# Add batch dimension: (64, 64) → (4, 16, 64)
batched = ct.reshape(matrix, (4, 16, 64))
print(batched.shape)  # (4, 16, 64)
```

**4D reshaping (batch processing):**
```python
# Flatten batch of images
images = ct.randn((32, 3, 32, 32), dtype=ct.float32)  # 32 images, 3 channels, 32×32

# Flatten each image: (32, 3, 32, 32) → (32, 3072)
flattened = ct.reshape(images, (32, -1))
print(flattened.shape)  # (32, 3072)
```

**Row-major vs column-major:**
```python
# cuTile uses row-major (C-style) ordering
tile = ct.arange(16, dtype=ct.float32).reshape((4, 4))

# Memory layout: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# Shape:      [[0,  1,  2,  3],
#              [4,  5,  6,  7],
#              [8,  9, 10, 11],
#              [12,13, 14, 15]]

# Reshape to (2, 8)
reshaped = ct.reshape(tile, (2, 8))
# Memory layout: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# Shape:      [[0, 1, 2, 3, 4, 5, 6, 7],
#              [8, 9,10,11,12,13,14,15]]
```

**Use case - matrix multiplication preparation:**
```python
def prepare_matmul(A, B):
    """
    Prepare matrices for multiplication.
    
    Args:
        A: (M, K) matrix
        B: (K, N) matrix
    
    Returns:
        (M, N) result
    """
    # Ensure inputs are 2D
    A_2d = ct.reshape(A, (-1, A.shape[-1]))
    B_2d = ct.reshape(B, (B.shape[-2], -1))
    
    # Compute matrix multiplication
    return ct.matmul(A_2d, B_2d)

# Example
A = ct.randn((32, 64), dtype=ct.float32)
B = ct.randn((64, 128), dtype=ct.float32)
result = prepare_matmul(A, B)
print(result.shape)  # (32, 128)
```

## Axis Permutation

### `ct.permute(tile, axes)`

Permutes the axes of a tile according to a specified ordering. This is a generalization of transpose that can handle any number of dimensions.

**Signature:**
```python
ct.permute(tile: Tile, axes: tuple[int, ...]) -> Tile
```

**Parameters:**
- `tile`: Input tile
- `axes`: Tuple specifying new axis order (permutation of [0, 1, ..., ndim-1])

**Requirements:**
- `axes` must be a permutation of all axis indices
- Length of `axes` must equal number of dimensions

**Examples:**

**2D transpose:**
```python
# Create 32×64 matrix
matrix = ct.arange(2048, dtype=ct.float32).reshape((32, 64))
print(matrix.shape)  # (32, 64)

# Transpose: (32, 64) → (64, 32)
transposed = ct.permute(matrix, (1, 0))
print(transposed.shape)  # (64, 32)
```

**3D permutation:**
```python
# Create 4D tile for batch processing: (batch, channels, height, width)
image = ct.randn((16, 3, 32, 32), dtype=ct.float32)
print(image.shape)  # (16, 3, 32, 32)

# Permute to (channels, batch, height, width)
permuted = ct.permute(image, (1, 0, 2, 3))
print(permuted.shape)  # (3, 16, 32, 32)

# Permute to (height, width, channels, batch)
permuted = ct.permute(image, (2, 3, 1, 0))
print(permuted.shape)  # (32, 32, 3, 16)
```

**Channel-first to channel-last:**
```python
def channels_first_to_last(x):
    """
    Convert from (batch, channels, height, width) to (batch, height, width, channels).
    """
    return ct.permute(x, (0, 2, 3, 1))

def channels_last_to_first(x):
    """
    Convert from (batch, height, width, channels) to (batch, channels, height, width).
    """
    return ct.permute(x, (0, 3, 1, 2))

# Example
images_cf = ct.randn((8, 3, 32, 32), dtype=ct.float32)  # Channels first
images_cl = channels_first_to_last(images_cf)
print(images_cl.shape)  # (8, 32, 32, 3)

# Convert back
images_back = channels_last_to_first(images_cl)
print(images_back.shape)  # (8, 3, 32, 32)
```

**Multi-head attention rearrangement:**
```python
def rearrange_attention(q, k, v, num_heads):
    """
    Rearrange query, key, value for multi-head attention.
    
    Args:
        q: (batch, seq_len, embed_dim)
        k: (batch, seq_len, embed_dim)
        v: (batch, seq_len, embed_dim)
        num_heads: number of attention heads
    
    Returns:
        q_heads: (batch, num_heads, seq_len, head_dim)
        k_heads: (batch, num_heads, seq_len, head_dim)
        v_heads: (batch, num_heads, seq_len, head_dim)
    """
    batch_size, seq_len, embed_dim = q.shape
    head_dim = embed_dim // num_heads
    
    # Reshape and permute
    # (batch, seq_len, num_heads, head_dim) → (batch, num_heads, seq_len, head_dim)
    q_heads = ct.permute(
        ct.reshape(q, (batch_size, seq_len, num_heads, head_dim)),
        (0, 2, 1, 3)
    )
    k_heads = ct.permute(
        ct.reshape(k, (batch_size, seq_len, num_heads, head_dim)),
        (0, 2, 1, 3)
    )
    v_heads = ct.permute(
        ct.reshape(v, (batch_size, seq_len, num_heads, head_dim)),
        (0, 2, 1, 3)
    )
    
    return q_heads, k_heads, v_heads

# Example
batch_size, seq_len, embed_dim = 8, 64, 128
num_heads = 8

q = ct.randn((batch_size, seq_len, embed_dim), dtype=ct.float32)
k = ct.randn((batch_size, seq_len, embed_dim), dtype=ct.float32)
v = ct.randn((batch_size, seq_len, embed_dim), dtype=ct.float32)

q_heads, k_heads, v_heads = rearrange_attention(q, k, v, num_heads)
print(q_heads.shape)  # (8, 8, 64, 16)
```

### `ct.transpose(tile, axis0, axis1)`

Transposes two axes of a tile. This is a convenience function for common 2D transpose operations.

**Signature:**
```python
ct.transpose(tile: Tile, axis0: int = -2, axis1: int = -1) -> Tile
```

**Parameters:**
- `tile`: Input tile (must have at least 2 dimensions)
- `axis0`: First axis to transpose (default: -2, second-to-last)
- `axis1`: Second axis to transpose (default: -1, last)

**Examples:**

**Default 2D transpose:**
```python
# Create 32×64 matrix
matrix = ct.arange(2048, dtype=ct.float32).reshape((32, 64))

# Transpose last two dimensions: (32, 64) → (64, 32)
result = ct.transpose(matrix)
print(result.shape)  # (64, 32)
```

**Specifying axes explicitly:**
```python
matrix = ct.randn((32, 64), dtype=ct.float32)

# Same as default
result1 = ct.transpose(matrix, 0, 1)
result2 = ct.transpose(matrix)
# result1 and result2 are identical
```

**3D transpose:**
```python
# Create 3D tile: (batch, rows, cols)
tensor = ct.randn((16, 32, 64), dtype=ct.float32)

# Transpose rows and cols: (16, 32, 64) → (16, 64, 32)
result = ct.transpose(tensor, axis0=1, axis1=2)
print(result.shape)  # (16, 64, 32)

# Transpose batch and rows: (16, 32, 64) → (32, 16, 64)
result = ct.transpose(tensor, axis0=0, axis1=1)
print(result.shape)  # (32, 16, 64)
```

**Using negative indices:**
```python
tensor = ct.randn((8, 16, 32, 64), dtype=ct.float32)

# Transpose last two dimensions: (8, 16, 32, 64) → (8, 16, 64, 32)
result = ct.transpose(tensor)  # Same as transpose(-2, -1)
print(result.shape)  # (8, 16, 64, 32)

# Transpose first and last: (8, 16, 32, 64) → (64, 16, 32, 8)
result = ct.transpose(tensor, axis0=0, axis1=-1)
print(result.shape)  # (64, 16, 32, 8)
```

**Matrix multiplication with transpose:**
```python
def matmul_transpose(A, B):
    """
    Compute A @ B^T efficiently.
    
    Args:
        A: (M, K)
        B: (N, K)
    
    Returns:
        (M, N) where result[i,j] = sum_k A[i,k] * B[j,k]
    """
    # Transpose B: (N, K) → (K, N)
    B_T = ct.transpose(B)
    
    # Multiply: (M, K) @ (K, N) → (M, N)
    return ct.matmul(A, B_T)

# Example
A = ct.randn((32, 64), dtype=ct.float32)
B = ct.randn((128, 64), dtype=ct.float32)
result = matmul_transpose(A, B)
print(result.shape)  # (32, 128)
```

## Data Type Conversion

### `ct.astype(tile, dtype)`

Converts a tile to a specified data type, performing appropriate type conversion and rounding.

**Signature:**
```python
ct.astype(tile: Tile, dtype: DType) -> Tile
```

**Parameters:**
- `tile`: Input tile
- `dtype`: Target data type (ct.float16, ct.float32, ct.int32, etc.)

**Conversion Rules:**
- **Floating-point to floating-point**: Standard rounding (round-to-nearest-even)
- **Floating-point to integer**: Truncate toward zero
- **Integer to floating-point**: Exact conversion
- **Integer to integer**: Exact conversion (sign extension for signed types)
- **Boolean conversions**: True → 1, False → 0

**Examples:**

**Float32 to Float16 (quantization):**
```python
# Create float32 tile
f32 = ct.randn((32, 32), dtype=ct.float32)

# Convert to float16 (reduces precision)
f16 = ct.astype(f32, ct.float16)
print(f16.dtype)  # float16
print(f16.shape)  # (32, 32)
```

**Float to Integer (quantization):**
```python
# Create float tile
f32 = ct.randn((32, 32), dtype=ct.float32) * 100

# Convert to int32 (truncates toward zero)
i32 = ct.astype(f32, ct.int32)
print(i32.dtype)  # int32
```

**Integer to Float (dequantization):**
```python
# Create int32 tile
i32 = ct.arange(1024, dtype=ct.int32).reshape((32, 32))

# Convert to float32
f32 = ct.astype(i32, ct.float32)
print(f32.dtype)  # float32
```

**Quantization simulation:**
```python
def quantize_to_int8(tile, scale, zero_point=0):
    """
    Quantize float32 tile to int8.
    
    Args:
        tile: float32 tile
        scale: quantization scale
        zero_point: zero point for int8 (default: 0)
    
    Returns:
        int8 tile
    """
    # Scale and round
    scaled = tile / scale
    rounded = ct.round(scaled)
    
    # Add zero point and convert to int8
    quantized = ct.astype(rounded + zero_point, ct.int8)
    
    return quantized

def dequantize_from_int8(tile, scale, zero_point=0):
    """
    Dequantize int8 tile to float32.
    """
    # Subtract zero point and convert to float32
    shifted = ct.astype(tile - zero_point, ct.float32)
    
    # Scale back
    return shifted * scale

# Example
f32 = ct.randn((32, 32), dtype=ct.float32)
scale = 0.1

# Quantize and dequantize
i8 = quantize_to_int8(f32, scale)
reconstructed = dequantize_from_int8(i8, scale)

# Compute quantization error
error = ct.abs(f32 - reconstructed)
print(f"Quantization error: {ct.sum(error) / error.numel}")
```

**Mixed precision computation:**
```python
def mixed_precision_matmul(A, B):
    """
    Compute matrix multiplication with mixed precision.
    
    - Convert inputs to float16 for computation
    - Accumulate in float32
    - Return float32 result
    """
    # Convert to float16
    A_f16 = ct.astype(A, ct.float16)
    B_f16 = ct.astype(B, ct.float16)
    
    # Compute in float16
    C_f16 = ct.matmul(A_f16, B_f16)
    
    # Convert back to float32
    return ct.astype(C_f16, ct.float32)

# Example
A = ct.randn((64, 128), dtype=ct.float32)
B = ct.randn((128, 256), dtype=ct.float32)
result = mixed_precision_matmul(A, B)
print(result.shape)  # (64, 128)
```

**Arithmetic promotion:**
```python
# Arithmetic operations follow type promotion rules
a = ct.full((32, 32), 1.0, dtype=ct.float16)
b = ct.full((32, 32), 2, dtype=ct.int32)

# Result is float16 (float16 + int32 → float16)
c = a + b
print(c.dtype)  # float16
```

### `ct.bitcast(tile, dtype)`

Reinterprets the bits of a tile as a different data type without any conversion. The total bitwidth must match exactly.

**Signature:**
```python
ct.bitcast(tile: Tile, dtype: DType) -> Tile
```

**Parameters:**
- `tile`: Input tile
- `dtype`: Target data type (must have same total bitwidth as input)

**Requirements:**
- Input and output types must have exactly the same bitwidth
- No data conversion or rounding is performed
- Shape may change if element size changes

**Examples:**

**Float32 ↔ Int32 bitcast:**
```python
# Create float32 tile
f32 = ct.full((32, 32), 1.0, dtype=ct.float32)

# Bitcast to int32 (same bitwidth: 32 bits)
i32 = ct.bitcast(f32, ct.int32)
print(i32.dtype)  # int32
print(i32.shape)  # (32, 32) - same shape

# The bit pattern for 1.0 in IEEE 754 is 0x3F800000
# So i32 contains the integer value 1065353216
```

**Float16 ↔ Int16 bitcast:**
```python
# Create float16 tile
f16 = ct.full((32, 32), 1.0, dtype=ct.float16)

# Bitcast to int16
i16 = ct.bitcast(f16, ct.int16)
print(i16.dtype)  # int16
print(i16.shape)  # (32, 32)
```

**Shape-preserving bitcast (same element size):**
```python
# Float32 (4 bytes) ↔ Int32 (4 bytes): shape unchanged
f32 = ct.randn((64, 64), dtype=ct.float32)
i32 = ct.bitcast(f32, ct.int32)
print(i32.shape)  # (64, 64)
```

**Shape-changing bitcast (different element size):**
```python
# Float32 (4 bytes) ↔ Int8 (1 byte): elements split
f32 = ct.arange(256, dtype=ct.float32)
i8 = ct.bitcast(f32, ct.int8)
print(f32.shape)  # (256,)  - 256 * 4 bytes = 1024 bytes
print(i8.shape)   # (1024,) - 1024 * 1 byte = 1024 bytes

# Int8 (1 byte) ↔ Float32 (4 bytes): elements grouped
i8 = ct.arange(1024, dtype=ct.int8)
f32 = ct.bitcast(i8, ct.float32)
print(f32.shape)  # (256,)
```

**Use case - fast type conversion:**
```python
def float_as_int(f):
    """Extract the bit representation of a float."""
    return ct.bitcast(f, ct.int32)

def int_as_float(i):
    """Convert int bits to float."""
    return ct.bitcast(i, ct.float32)

# Example
f = ct.full((32, 32), 3.14159, dtype=ct.float32)
i = float_as_int(f)
print(f.dtype)  # float32
print(i.dtype)  # int32

# Convert back
f_reconstructed = int_as_float(i)
# f_reconstructed == f
```

**Use case - packing/unpacking:**
```python
def pack_float32_to_int8(f32):
    """
    Pack float32 values into int8 array.
    Each float32 (4 bytes) becomes 4 int8 values.
    """
    return ct.bitcast(f32, ct.int8)

def unpack_int8_to_float32(i8):
    """
    Unpack int8 array back to float32.
    Every 4 int8 values become 1 float32.
    """
    return ct.bitcast(i8, ct.float32)

# Example
f32 = ct.randn((32, 32), dtype=ct.float32)
i8_packed = pack_float32_to_int8(f32)
print(f32.shape)     # (32, 32)  = 1024 elements * 4 bytes = 4096 bytes
print(i8_packed.shape)  # (1024, 4) = 4096 elements * 1 byte = 4096 bytes

# Unpack
f32_unpacked = unpack_int8_to_float32(i8_packed)
print(f32_unpacked.shape)  # (32, 32)
```

## Complete Examples

### Example 1: Batch Matrix Multiplication

```python
def batch_matmul(A, B):
    """
    Batch matrix multiplication.
    
    Args:
        A: (batch, M, K)
        B: (batch, K, N)
    
    Returns:
        (batch, M, N)
    """
    # Ensure 3D shapes
    A_3d = ct.reshape(A, (-1, A.shape[-2], A.shape[-1]))
    B_3d = ct.reshape(B, (-1, B.shape[-2], B.shape[-1]))
    
    # Compute batched matmul
    return ct.matmul(A_3d, B_3d)

# Example
batch_size = 16
A = ct.randn((batch_size, 32, 64), dtype=ct.float32)
B = ct.randn((batch_size, 64, 128), dtype=ct.float32)
result = batch_matmul(A, B)
print(result.shape)  # (16, 32, 128)
```

### Example 2: Image Layout Conversion

```python
def nchw_to_nhwc(images):
    """
    Convert images from NCHW to NHWC format.
    
    Args:
        images: (batch, channels, height, width)
    
    Returns:
        (batch, height, width, channels)
    """
    return ct.permute(images, (0, 2, 3, 1))

def nhwc_to_nchw(images):
    """
    Convert images from NHWC to NCHW format.
    
    Args:
        images: (batch, height, width, channels)
    
    Returns:
        (batch, channels, height, width)
    """
    return ct.permute(images, (0, 3, 1, 2))

# Example
images_nchw = ct.randn((8, 3, 32, 32), dtype=ct.float32)
images_nhwc = nchw_to_nhwc(images_nchw)
print(images_nhwc.shape)  # (8, 32, 32, 3)

# Convert back
images_back = nhwc_to_nchw(images_nhwc)
print(images_back.shape)  # (8, 3, 32, 32)
```

### Example 3: Quantization Pipeline

```python
def quantize_pipeline(tile, target_dtype=ct.float16):
    """
    Complete quantization pipeline.
    
    Args:
        tile: float32 input
        target_dtype: target dtype (float16 or int8)
    
    Returns:
        quantized tile and scale
    """
    # Compute range for scaling
    min_val = ct.min(tile)
    max_val = ct.max(tile)
    
    if target_dtype == ct.float16:
        # Direct conversion
        quantized = ct.astype(tile, ct.float16)
        scale = 1.0
    elif target_dtype == ct.int8:
        # Compute scale
        range_val = ct.maximum(ct.abs(min_val), ct.abs(max_val))
        scale = range_val / 127.0
        
        # Quantize
        quantized = ct.astype(ct.round(tile / scale), ct.int8)
    else:
        raise ValueError(f"Unsupported target dtype: {target_dtype}")
    
    return quantized, scale

def dequantize_pipeline(tile, scale, original_dtype=ct.float32):
    """Dequantize tile back to original dtype."""
    if tile.dtype == ct.float16:
        return ct.astype(tile, original_dtype)
    elif tile.dtype == ct.int8:
        dequantized = ct.astype(tile, ct.float32) * scale
        if original_dtype == ct.float32:
            return dequantized
        else:
            return ct.astype(dequantized, original_dtype)
    else:
        return tile

# Example
f32 = ct.randn((64, 64), dtype=ct.float32)

# Quantize to float16
f16, scale1 = quantize_pipeline(f32, ct.float16)
print(f"Float16 shape: {f16.shape}, dtype: {f16.dtype}")

# Quantize to int8
i8, scale2 = quantize_pipeline(f32, ct.int8)
print(f"Int8 shape: {i8.shape}, dtype: {i8.dtype}, scale: {scale2}")

# Dequantize
f32_recon_f16 = dequantize_pipeline(f16, scale1)
f32_recon_i8 = dequantize_pipeline(i8, scale2)

# Compute errors
error_f16 = ct.sum(ct.abs(f32 - f32_recon_f16)) / f32.numel
error_i8 = ct.sum(ct.abs(f32 - f32_recon_i8)) / f32.numel
print(f"Float16 error: {error_f16}")
print(f"Int8 error: {error_i8}")
```

### Example 4: Dynamic Shape Broadcasting

```python
def broadcast_and_apply(func, *inputs):
    """
    Apply function to inputs with automatic broadcasting.
    
    Args:
        func: function to apply
        *inputs: input tiles with potentially different shapes
    
    Returns:
        result with broadcasted shape
    """
    # Determine output shape (broadcast all inputs)
    shapes = [inp.shape for inp in inputs]
    max_ndim = max(len(s) for s in shapes)
    
    # Pad shapes with 1s on left
    padded_shapes = []
    for shape in shapes:
        pad = (1,) * (max_ndim - len(shape))
        padded_shapes.append(pad + shape)
    
    # Compute broadcast shape
    broadcast_shape = []
    for dims in zip(*padded_shapes):
        # Compatible if all same or one is 1
        non_ones = [d for d in dims if d != 1]
        if len(non_ones) == 0:
            broadcast_shape.append(1)
        else:
            # All non-1 dims must be equal
            broadcast_shape.append(non_ones[0])
    
    # Broadcast all inputs
    broadcasted_inputs = [
        ct.broadcast_to(inp, tuple(broadcast_shape))
        for inp in inputs
    ]
    
    # Apply function
    return func(*broadcasted_inputs)

# Example
a = ct.full((32, 1), 2.0, dtype=ct.float32)
b = ct.full((1, 64), 3.0, dtype=ct.float32)
c = ct.full((32, 64), 1.0, dtype=ct.float32)

# Apply with broadcasting
result = broadcast_and_apply(lambda x, y, z: x + y * z, a, b, c)
print(result.shape)  # (32, 64)
```

## Best Practices

1. **Power of 2 Dimensions**: Always ensure all dimensions are powers of 2 for optimal tensor core utilization.

2. **Memory Efficiency**: Use `broadcast_to` instead of explicit replication to save memory.

3. **Shape Inference**: Use `-1` in `reshape` to infer dimensions automatically when the total size is known.

4. **Type Conversion**: Be aware of precision loss when converting from higher to lower precision types.

5. **Bitcasting**: Use `bitcast` only when you need exact bit preservation; use `astype` for normal type conversions.

6. **Performance**: Minimize unnecessary reshape and transpose operations as they may introduce memory overhead.

7. **Batch Operations**: Use broadcasting and proper dimension ordering to enable efficient batch processing.
