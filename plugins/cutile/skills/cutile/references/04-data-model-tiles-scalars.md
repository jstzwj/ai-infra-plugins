# Data Model: Tiles and Scalars

## Overview

The cuTile data model is built around two fundamental concepts: **tiles** and **scalars**. Understanding these concepts is essential for writing effective cuTile programs. This chapter provides a comprehensive exploration of tiles, scalars, their properties, creation methods, and how they relate to the execution model.

## What is a Tile?

A **tile** is an immutable multidimensional collection of elements of a specific data type (`dtype`). Tiles are the primary units of data manipulation in cuTile programs and form the foundation for all tensor operations.

### Fundamental Characteristics

**Immutability**: Once created, a tile cannot be modified. All operations on tiles produce new tiles as results. This immutability enables compiler optimizations and ensures predictable behavior in parallel execution contexts.

```python
import cuda.tile as ct

@ct.function
def tile_operations():
    # Create a tile
    x = ct.full((16, 16), 5.0, dtype=ct.float32)
    
    # Operations create new tiles; x remains unchanged
    y = x + 10.0      # Creates new tile y
    z = y * 2.0       # Creates new tile z
    
    # x, y, and z are all distinct tiles
    return x, y, z
```

**Multidimensional Structure**: Tiles can have any number of dimensions (rank), from zero-dimensional scalars to higher-dimensional tensors. Each dimension represents a distinct axis along which elements are organized.

```python
# Various tile shapes
scalar = ct.full((), 1.0, dtype=ct.float32)              # 0D: scalar
vector = ct.arange(32, dtype=ct.int32)                   # 1D: vector
matrix = ct.zeros((16, 16), dtype=ct.float32)            # 2D: matrix
tensor = ct.ones((8, 16, 32), dtype=ct.float32)          # 3D: tensor
```

### Tile Shape

The **shape** of a tile is a tuple of integers that specifies its size along each dimension. Tile shapes have several important characteristics:

**Compile-Time Knowledge**: Tile shapes must be known at compile time. This requirement enables the compiler to generate specialized code for specific tile sizes, optimizing memory access patterns and computational efficiency.

```python
@ct.function
def shaped_tile_operations():
    # Valid: shape is known at compile time
    tile_32x32 = ct.zeros((32, 32), dtype=ct.float32)
    
    # Invalid: shape not known at compile time
    # This will raise a compilation error
    # n = 32
    # dynamic_tile = ct.zeros((n, n), dtype=ct.float32)
    
    return tile_32x32
```

**Power-of-2 Constraint**: Each dimension of a tile must be a power of 2. This constraint aligns with hardware architecture optimizations and enables efficient memory coalescing and vectorization.

```python
# Valid tile shapes (all dimensions are powers of 2)
valid_shapes = [
    (1,),           # 2^0
    (2,),           # 2^1
    (4, 8),         # 2^2, 2^3
    (16, 32, 64),   # 2^4, 2^5, 2^6
    (128, 256),     # 2^7, 2^8
]

# Invalid tile shapes (dimensions not powers of 2)
invalid_shapes = [
    (3,),           # 3 is not a power of 2
    (10, 20),       # Neither 10 nor 20 are powers of 2
    (5, 7, 9),      # All dimensions invalid
]
```

**Shape Attribute**: The `shape` attribute of a tile provides compile-time access to its dimensional structure:

```python
@ct.function
def access_shape():
    tile = ct.full((16, 32, 64), 1.0, dtype=ct.float32)
    
    # Access shape dimensions
    dim0 = tile.shape[0]  # 16
    dim1 = tile.shape[1]  # 32
    dim2 = tile.shape[2]  # 64
    rank = len(tile.shape)  # 3
    
    # Shape is a compile-time constant
    # Can be used in compile-time expressions
    total_elements = dim0 * dim1 * dim2  # 32768
    
    return tile, total_elements
```

### Tile Data Type

Every tile has a specific **dtype** (data type) that defines the type of elements it contains. The dtype is a compile-time constant and cannot be changed after tile creation.

```python
@ct.function
def dtype_operations():
    # Create tiles with different dtypes
    int_tile = ct.full((8, 8), 42, dtype=ct.int32)
    float_tile = ct.full((8, 8), 3.14, dtype=ct.float32)
    bool_tile = ct.full((8, 8), True, dtype=ct.bool_)
    
    # Access dtype attribute
    int_dtype = int_tile.dtype      # cuda.tile.int32
    float_dtype = float_tile.dtype  # cuda.tile.float32
    bool_dtype = bool_tile.dtype    # cuda.tile.bool_
    
    # Dtype determines operations and precision
    result = float_tile + 1.5  # Float operation
    # result = int_tile + 1.5   # Would promote to float
    
    return int_tile, float_tile, bool_tile
```

## What is a Scalar?

A **scalar** in cuTile is a special case of a tile with zero dimensions. Despite the name, scalars are still tiles and possess all tile properties, including dtype and shape.

### Scalar Characteristics

**Zero-Dimensional**: A scalar has an empty tuple `()` as its shape. This distinguishes it from single-element vectors or matrices.

```python
@ct.function
def scalar_characteristics():
    # Create scalars
    scalar_int = ct.int32(42)
    scalar_float = ct.float32(3.14)
    scalar_bool = ct.bool_(True)
    
    # All have empty shape
    assert scalar_int.shape == ()
    assert scalar_float.shape == ()
    assert scalar_bool.shape == ()
    
    # But different dtypes
    assert scalar_int.dtype == ct.int32
    assert scalar_float.dtype == ct.float32
    assert scalar_bool.dtype == ct.bool_
    
    return scalar_int, scalar_float, scalar_bool
```

**Single Element**: A scalar contains exactly one element of its dtype.

```python
@ct.function
def scalar_element():
    scalar = ct.float32(2.71828)
    
    # Can be used in computations
    result = scalar * 2.0  # Produces ct.float32(5.43656)
    
    # Can be compared
    is_positive = scalar > 0.0  # Produces ct.bool_(True)
    
    return scalar, result, is_positive
```

### Numeric Literals as Scalars

Numeric literals in cuTile code are automatically treated as constant scalars. This provides convenient syntax for scalar values.

```python
@ct.function
def literal_scalars():
    # Integer literal treated as scalar
    int_literal = 42  # Zero-dim tile with infinite precision
    
    # Float literal treated as scalar
    float_literal = 3.14  # Zero-dim tile in IEEE 754 double precision
    
    # Can be used in tile operations
    tile = ct.full((16, 16), 1.0, dtype=ct.float32)
    result = tile * 2.5  # Multiplies each element by scalar 2.5
    
    # Can be compared
    is_large = tile > 1.5  # Produces boolean tile
    
    return tile, result, is_large
```

### Scalar vs Python Types

It's important to distinguish between cuTile scalars and Python numeric types:

**Python Int/Float**: Dynamic precision, unlimited range for integers, 64-bit floats, mutable objects.

**cuTile Scalars**: Fixed dtype, limited range based on dtype, immutable tiles, compile-time type checking.

```python
# Python code (host side)
python_int = 42        # Python int: unlimited precision
python_float = 3.14    # Python float: 64-bit IEEE 754

# cuTile code (tile side)
@ct.function
def scalar_types():
    cutile_int = 42                    # Loosely typed scalar
    cutile_float = 3.14                # Loosely typed scalar
    strict_int = ct.int32(42)          # Strictly typed scalar
    strict_float = ct.float32(3.14)    # Strictly typed scalar
    
    # Each has specific dtype and shape
    return cutile_int, cutile_float, strict_int, strict_float
```

## Tile Creation

Tiles are created through several mechanisms: memory operations, factory functions, and computational operations.

### Memory Operations

**Loading from Global Memory**: The `ct.load()` operation creates tiles by reading data from global device memory.

```python
@ct.function
def load_tiles(global_array: ct.Array):
    # Load single tile from global memory
    tile = ct.load(global_array)
    
    # Load tile with specific indexing
    indexed_tile = ct.load(global_array[0:16, 0:16])
    
    return tile, indexed_tile
```

**Gathering from Memory**: The `ct.gather()` operation creates tiles by gathering scattered elements from memory.

```python
@ct.function
def gather_tiles(global_array: ct.Array, indices: ct.Tile):
    # Gather elements at specified indices
    gathered = ct.gather(global_array, indices)
    
    return gathered
```

### Factory Functions

**zeros**: Create a tile filled with zeros.

```python
@ct.function
def zeros_factory():
    # Zero tile of specified shape and dtype
    zero_tile = ct.zeros((16, 16), dtype=ct.float32)
    
    # Default dtype is float32
    default_zero = ct.zeros((8, 8))
    
    return zero_tile, default_zero
```

**ones**: Create a tile filled with ones.

```python
@ct.function
def ones_factory():
    # Ones tile of specified shape and dtype
    ones_tile = ct.ones((32, 32), dtype=ct.float32)
    
    # Integer ones
    int_ones = ct.ones((16, 16), dtype=ct.int32)
    
    return ones_tile, int_ones
```

**full**: Create a tile filled with a specific value.

```python
@ct.function
def full_factory():
    # Fill with float value
    pi_tile = ct.full((8, 8), 3.14159, dtype=ct.float32)
    
    # Fill with integer value
    value_tile = ct.full((16, 16), 42, dtype=ct.int32)
    
    # Fill with boolean value
    true_tile = ct.full((4, 4), True, dtype=ct.bool_)
    
    return pi_tile, value_tile, true_tile
```

**arange**: Create a tile with evenly spaced values.

```python
@ct.function
def arange_factory():
    # Sequential values 0, 1, 2, ..., 31
    seq_tile = ct.arange(32, dtype=ct.int32)
    
    # Start at 10, end before 20
    range_tile = ct.arange(10, 20, dtype=ct.int32)
    
    # Float range with step
    float_range = ct.arange(0.0, 1.0, 0.1, dtype=ct.float32)
    
    return seq_tile, range_tile, float_range
```

### Computational Operations

Operations on tiles create new tiles as results. This includes arithmetic operations, mathematical functions, and transformations.

```python
@ct.function
def computational_creation():
    # Create base tiles
    tile_a = ct.full((16, 16), 3.0, dtype=ct.float32)
    tile_b = ct.full((16, 16), 4.0, dtype=ct.float32)
    
    # Arithmetic operations create new tiles
    sum_tile = tile_a + tile_b          # All elements are 7.0
    diff_tile = tile_a - tile_b          # All elements are -1.0
    prod_tile = tile_a * tile_b          # All elements are 12.0
    
    # Mathematical functions
    sqrt_tile = ct.sqrt(tile_a)          # Square root
    exp_tile = ct.exp(tile_a)            # Exponential
    log_tile = ct.log(tile_a)            # Natural logarithm
    
    return sum_tile, diff_tile, prod_tile, sqrt_tile, exp_tile, log_tile
```

## Tile Storage

Tiles are stored back to global device memory through explicit operations.

### Storing to Global Memory

The `ct.store()` operation writes tile contents to global device memory.

```python
@ct.function
def store_tiles(global_array: ct.Array):
    # Load tile from memory
    input_tile = ct.load(global_array)
    
    # Process tile
    output_tile = input_tile * 2.0
    
    # Store result back to memory
    ct.store(global_array, output_tile)
```

**Store with Indexing**: Tiles can be stored to specific memory regions.

```python
@ct.function
def indexed_store(global_array: ct.Array):
    # Process and store to specific region
    tile = ct.full((16, 16), 1.0, dtype=ct.float32)
    ct.store(global_array[32:48, 32:48], tile)
```

### Scattering to Memory

The `ct.scatter()` operation writes tile elements to scattered memory locations.

```python
@ct.function
def scatter_tiles(global_array: ct.Array, indices: ct.Tile, values: ct.Tile):
    # Scatter values to indexed locations
    ct.scatter(global_array, indices, values)
```

### Physical Memory Representation

An important characteristic of tiles is that their contents don't necessarily have a physical memory representation. Tiles are logical collections of elements that may exist:

- In registers during computation
- In shared memory for block-level sharing
- In global memory for persistence
- As intermediate values that never materialize in memory

This abstraction allows the compiler to optimize memory usage and minimize data movement.

```python
@ct.function
def memory_abstraction(global_array: ct.Array):
    # Intermediate tiles may never exist in memory
    tile = ct.load(global_array)
    
    # Each operation creates a new tile
    # Compiler may optimize to keep everything in registers
    temp1 = tile * 2.0    # May be register-allocated
    temp2 = temp1 + 1.0   # May be register-allocated
    temp3 = temp2 ** 2.0  # May be register-allocated
    
    # Only final result stored to memory
    ct.store(global_array, temp3)
```

## Scalar Constants

Scalar constants in cuTile have specific typing and promotion rules that affect their behavior in computations.

### Loosely Typed Constants

Numeric literals are **loosely typed** by default, meaning they have flexible precision:

- Integer literals have infinite precision (mathematical integers)
- Float literals use IEEE 754 double precision (64-bit)

```python
@ct.function
def loose_constants():
    # Integer literal: infinite precision
    infinite_int = 12345678901234567890
    
    # Float literal: double precision
    double_float = 1.7976931348623157e+308
    
    # These acquire concrete types through operations
    int_tile = ct.full((8, 8), 1, dtype=ct.int32)
    
    # Infinite precision literal promotes to match tile
    result = int_tile + infinite_int  # Result is int32
    
    return result
```

### Strictly Typed Constants

Strictly typed constants are created using dtype constructors and have fixed precision:

```python
@ct.function
def strict_constants():
    # Strictly typed integer constants
    int8_val = ct.int8(127)
    int16_val = ct.int16(32767)
    int32_val = ct.int32(2147483647)
    int64_val = ct.int64(9223372036854775807)
    
    # Strictly typed float constants
    float16_val = ct.float16(3.14)
    float32_val = ct.float32(3.14159265359)
    float64_val = ct.float64(2.718281828459045)
    
    # Boolean constant
    bool_val = ct.bool_(True)
    
    # These maintain their exact types in operations
    result = int32_val + int16_val  # Promoted per type rules
    
    return result
```

### Type Promotion with Constants

When mixing loosely and strictly typed constants, specific promotion rules apply:

```python
@ct.function
def constant_promotion():
    # Strictly typed tile
    int32_tile = ct.full((8, 8), 10, dtype=ct.int32)
    
    # Loosely typed constant
    loose_int = 42
    
    # Strictly typed constant
    strict_int = ct.int64(100)
    
    # Loose constant promotes to match tile
    result1 = int32_tile + loose_int  # int32 result
    
    # Strict constant may promote
    result2 = int32_tile + strict_int  # May promote to int64
    
    return result1, result2
```

## Element Space vs Tile Space

cuTile distinguishes between two conceptual spaces for organizing data: element space and tile space.

### Element Space

**Element space** is the multidimensional space of individual elements stored in memory. Each point in element space corresponds to a single data element.

```python
# 1024x1024 array in element space
# Elements are indexed as array[i, j] where 0 <= i, j < 1024
element_space_size = (1024, 1024)
total_elements = 1024 * 1024  # 1,048,576 elements
```

### Tile Space

**Tile space** is the multidimensional space of tiles for a given tile shape. Each point in tile space corresponds to a tile rather than an individual element.

```python
# For 1024x1024 array with 16x16 tiles
# Tile space dimensions: (64, 64) tiles
tile_shape = (16, 16)
tile_space_dims = (1024 // 16, 1024 // 16)  # (64, 64)
total_tiles = 64 * 64  # 4,096 tiles
```

**Tile Indexing**: A tile index (i, j, ...) refers to the (i+1)-th tile along the first dimension, (j+1)-th tile along the second dimension, etc.

```python
@ct.function
def tile_indexing(global_array: ct.Array):
    # Load tile at position (2, 3) in tile space
    # This loads elements [32:48, 48:64] in element space
    tile = ct.load(global_array[32:48, 48:64])
    
    return tile
```

### Relationship Between Spaces

The relationship between element space and tile space is fundamental to understanding cuTile's execution model:

```python
def space_relationship():
    # Array dimensions in element space
    element_shape = (1024, 1024)
    
    # Tile dimensions
    tile_shape = (16, 16)
    
    # Resulting tile space
    tile_space = (
        element_shape[0] // tile_shape[0],
        element_shape[1] // tile_shape[1]
    )  # (64, 64)
    
    # Element (i, j) maps to tile (i // 16, j // 16)
    # Tile (ti, tj) contains elements [(ti*16):(ti*16 + 16), (tj*16):(tj*16 + 16)]
    
    return element_shape, tile_shape, tile_space
```

### Practical Example

```python
@ct.function
def tiled_processing(global_array: ct.Array):
    # Process 1024x1024 array with 16x16 tiles
    # Each block handles one tile
    
    # Load tile from element space region
    tile = ct.load(global_array)
    
    # Process tile elements
    processed = tile * 2.0 + 1.0
    
    # Store back to same element space region
    ct.store(global_array, processed)
```

## Tile Immutability

The immutability of tiles is a fundamental design choice with important implications.

### Immutability Benefits

**Compiler Optimization**: Immutable tiles enable aggressive compiler optimizations since data dependencies are explicit and side effects are eliminated.

```python
@ct.function
def immutable_optimizations():
    tile = ct.full((16, 16), 1.0, dtype=ct.float32)
    
    # Compiler can optimize sequence knowing tile is unchanged
    temp1 = tile + 1.0  # Could compute once and reuse
    temp2 = tile + 2.0  # Independent computation
    temp3 = tile + 3.0  # Independent computation
    
    # Could potentially vectorize or parallelize
    return temp1, temp2, temp3
```

**Predictable Behavior**: Immutability ensures operations don't have hidden side effects.

```python
@ct.function
def predictable_behavior(input_tile: ct.Tile):
    # Original tile guaranteed unchanged
    result1 = input_tile * 2.0
    
    # Can safely use input_tile again
    result2 = input_tile + 1.0
    
    # result1 and result2 are independent
    return result1, result2
```

**Memory Safety**: Immutability prevents issues with shared mutable state in parallel contexts.

### Creating Modified Tiles

Since tiles cannot be modified, operations create new tiles with desired modifications:

```python
@ct.function
def tile_modifications():
    original = ct.full((16, 16), 5.0, dtype=ct.float32)
    
    # Create modified versions
    doubled = original * 2.0
    incremented = original + 1.0
    squared = original ** 2.0
    
    # Original remains unchanged
    # Each operation creates a new tile
    
    return original, doubled, incremented, squared
```

## Tiles vs Blocks

Understanding the distinction between tiles and blocks is crucial for effective cuTile programming.

### Block: Unit of Execution

A **block** is the unit of execution in cuTile's execution model:

- Contains one or more threads executing in SIMT fashion
- Has shared memory accessible to all threads in the block
- Operates on tiles of data
- Identified by its position in a grid of blocks

```python
# Launching kernels with grid of blocks
grid = (8, 8)  # 8x8 grid of blocks
# Each block executes independently
```

### Tile: Unit of Data

A **tile** is the unit of data manipulated by blocks:

- Multidimensional collection of elements
- Processed by operations within a block
- Can be stored in various memory spaces
- May be transferred between memory spaces

```python
@ct.function
def block_tile_operations(global_array: ct.Array):
    # Block loads tile from global memory
    tile = ct.load(global_array)
    
    # Block processes tile with parallel operations
    processed = ct.sqrt(tile) + 1.0
    
    # Block stores tile back to global memory
    ct.store(global_array, processed)
```

### Relationship

- **One block operates on one or more tiles**: Each block processes tiles of data during its execution.
- **Multiple tiles per block**: A single block can work with multiple tiles of different shapes.
- **Tile shape independent of block configuration**: Tiles can have any valid shape regardless of block dimensions.

```python
@ct.function
def multi_tile_block(global_a: ct.Array, global_b: ct.Array):
    # Single block operates on multiple tiles
    tile_a = ct.load(global_a)      # Load tile A
    tile_b = ct.load(global_b)      # Load tile B
    
    # Process tiles independently
    result_a = tile_a * 2.0
    result_b = tile_b + 1.0
    
    # Combine tiles
    combined = result_a + result_b
    
    # Store result
    ct.store(global_a, combined)
```

### Shape Flexibility

Multiple tiles of different shapes can be used within a single block:

```python
@ct.function
def multiple_tile_shapes():
    # Create tiles of different shapes
    tile_16x16 = ct.zeros((16, 16), dtype=ct.float32)
    tile_32x32 = ct.ones((32, 32), dtype=ct.float32)
    tile_8x8x8 = ct.full((8, 8, 8), 2.0, dtype=ct.float32)
    
    # Can operate on tiles of different shapes
    # (with broadcasting where applicable)
    expanded_8x8 = tile_8x8x8[0, :, :]  # Extract 2D slice
    
    return tile_16x16, tile_32x32, expanded_8x8
```

## Advanced Tile Concepts

### Tile Views and Slicing

Tiles can be viewed and sliced to create new tile objects:

```python
@ct.function
def tile_views():
    # Create base tile
    base_tile = ct.arange(128, dtype=ct.int32)
    
    # Slice to create view (new tile)
    first_half = base_tile[0:64]      # Elements 0-63
    second_half = base_tile[64:128]   # Elements 64-127
    
    # 2D tile slicing
    matrix = ct.full((32, 32), 1.0, dtype=ct.float32)
    quadrant = matrix[0:16, 0:16]     # Upper-left quadrant
    
    return first_half, second_half, quadrant
```

### Tile Broadcasting

Tiles of different shapes can be combined through broadcasting rules:

```python
@ct.function
def tile_broadcasting():
    # Scalar (0D) broadcasts to any shape
    scalar = ct.float32(5.0)
    matrix = ct.full((16, 16), 1.0, dtype=ct.float32)
    result = matrix + scalar  # scalar broadcasts to 16x16
    
    # Vector (1D) can broadcast along matrix dimensions
    vector = ct.full((16,), 2.0, dtype=ct.float32)
    result2 = matrix + vector  # vector broadcasts to 16x16
    
    return result, result2
```

### Tile Reshaping

Tiles can be reshaped while preserving total element count:

```python
@ct.function
def tile_reshape():
    # 1D to 2D
    vector = ct.arange(64, dtype=ct.float32)
    matrix = ct.reshape(vector, (8, 8))  # 64 elements -> 8x8
    
    # 2D to 3D
    matrix = ct.full((16, 16), 1.0, dtype=ct.float32)
    tensor = ct.reshape(matrix, (4, 8, 8))  # 256 elements -> 4x8x8
    
    return matrix, tensor
```

## Performance Considerations

### Tile Size Selection

Choosing appropriate tile sizes is crucial for performance:

```python
# Good tile sizes (match hardware characteristics)
good_tiles = [
    (16, 16),    # Fits well in shared memory
    (32, 32),    # Good balance for compute/memory
    (64, 64),    # For larger datasets
]

# Consider:
# - Shared memory capacity (48KB per SM)
# - Register file capacity (64K 32-bit registers per SM)
# - Memory coalescing patterns
# - Cache line utilization
```

### Memory Access Patterns

Efficient memory access patterns optimize tile operations:

```python
@ct.function
def efficient_access(global_array: ct.Array):
    # Coalesced access pattern
    tile = ct.load(global_array)  # Assumes proper alignment
    
    # Process to maximize reuse
    temp1 = tile * 2.0
    temp2 = tile + 1.0
    result = temp1 + temp2
    
    # Single store back to memory
    ct.store(global_array, result)
```

### Compiler Optimizations

The cuTile compiler performs various optimizations on tile operations:

- **Loop unrolling**: For small, fixed-size tiles
- **Vectorization**: Using SIMD instructions
- **Register allocation**: Keeping frequently accessed tiles in registers
- **Shared memory tiling**: Reducing global memory accesses
- **Operation fusion**: Combining multiple operations into single kernel

## Best Practices

### Tile Creation

1. **Use factory functions for simple tiles**: `zeros()`, `ones()`, `full()` are optimized for common patterns.
2. **Specify dtypes explicitly**: Avoids implicit type conversions.
3. **Choose appropriate sizes**: Match tile sizes to hardware characteristics.

```python
@ct.function
def best_practice_creation():
    # Explicit dtype specification
    tile = ct.zeros((16, 16), dtype=ct.float32)
    
    # Appropriate size for hardware
    optimal_tile = ct.full((32, 32), 1.0, dtype=ct.float32)
    
    return tile, optimal_tile
```

### Tile Operations

1. **Minimize tile creation**: Reuse tiles where possible.
2. **Chain operations**: Allow compiler to fuse operations.
3. **Avoid unnecessary copies**: Work with views when appropriate.

```python
@ct.function
def best_practice_operations(tile: ct.Tile):
    # Chain operations for potential fusion
    result = ct.sqrt(tile * 2.0 + 1.0)
    
    # Single pass through data
    return result
```

### Memory Management

1. **Minimize global memory access**: Use tiles to cache data.
2. **Coalesce memory accesses**: Ensure sequential access patterns.
3. **Reuse loaded tiles**: Avoid reloading the same data.

```python
@ct.function
def best_practice_memory(global_array: ct.Array):
    # Single load, multiple uses
    tile = ct.load(global_array)
    
    # Reuse loaded tile
    temp1 = tile + 1.0
    temp2 = tile * 2.0
    temp3 = tile ** 2.0
    
    # Combine results
    result = temp1 + temp2 + temp3
    
    # Single store
    ct.store(global_array, result)
```

## Conclusion

Tiles and scalars form the foundation of cuTile's data model. Their immutable nature, compile-time shape requirements, and clear distinction from execution blocks enable efficient compilation and execution. Understanding these concepts thoroughly is essential for writing effective cuTile programs that maximize hardware performance and maintainability.

The key takeaways are:

- Tiles are immutable multidimensional collections with compile-time known shapes
- Scalars are zero-dimensional tiles with special handling for literals
- Element space and tile space provide complementary views of data organization
- The distinction between tiles (data) and blocks (execution) enables flexible programming
- Proper tile size selection and memory access patterns are crucial for performance

Mastering these concepts provides the foundation for advanced cuTile programming and optimization techniques.
