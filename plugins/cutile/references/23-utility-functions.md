# Chapter 23: Utility Functions

## Overview

cuTile provides a comprehensive set of utility functions that simplify common programming tasks, enable runtime debugging, and provide control over execution. These utilities range from printing and assertions to mathematical helpers and tuple operations. This chapter covers all available utility functions in detail.

## 23.1 Printing Functions

### 23.1.1 ct.printf() — C-Style Formatted Printing

The `ct.printf()` function provides C-style formatted output from device code, similar to CUDA's native `printf()`.

#### Syntax

```python
ct.printf(format_string, *args)
```

#### Parameters

- `format_string`: A C-style format string with format specifiers
- `*args`: Values to format and print

#### Format Specifiers

| Specifier | Type | Description |
|-----------|------|-------------|
| `%d` | int | Signed decimal integer |
| `%u` | int | Unsigned decimal integer |
| `%f` | float | Decimal floating point |
| `%e` | float | Scientific notation |
| `%g` | float | Shortest of %e or %f |
| `%x` | int | Unsigned hexadecimal |
| `%p` | pointer | Pointer address |
| `%s` | str | String (limited support) |

#### Basic Examples

```python
import cuda.tile as ct

@ct.tile
def print_values(x: ct.tile.float32, idx: ct.tile.int32):
    # Print a simple value
    ct.printf("Value: %f\n", x)
    
    # Print multiple values
    ct.printf("Index %d, Value %f\n", idx, x)
    
    # Print in different formats
    ct.printf("Float: %f, Scientific: %e\n", x, x)
    
    return x * 2.0

grid = (128, 128)
stream = ct.Stream()
x = ct.array([1.0, 2.0, 3.0, 4.0], dtype=ct.float32)
idx = ct.array([0, 1, 2, 3], dtype=ct.int32)

result = ct.launch(stream, grid, print_values, (x, idx))
stream.synchronize()
```

#### Printing Arrays

```python
@ct.tile
def print_array(arr: ct.tile.float32[1024]):
    # Print first element
    ct.printf("First element: %f\n", arr[0])
    
    # Print every 100th element
    for i in ct.static_range(0, 1024, 100):
        ct.printf("arr[%d] = %f\n", i, arr[i])
    
    return arr
```

#### Formatting Options

```python
@ct.tile
def formatted_print(x: ct.tile.float32, y: ct.tile.float32):
    # Width and precision
    ct.printf("Fixed width: %10.4f\n", x)
    
    # Left alignment
    ct.printf("Left align: %-10.2f\n", y)
    
    # Zero padding
    ct.printf("Zero pad: %08d\n", ct.int32(x))
    
    # Hexadecimal
    ct.printf("Hex: 0x%x\n", ct.int32(x))
    
    return x + y
```

#### Printing in Conditional Code

```python
@ct.tile
def conditional_print(x: ct.tile.float32, threshold: ct.tile.float32):
    # Only print values above threshold
    if x > threshold:
        ct.printf("Above threshold: %f\n", x)
    
    return x
```

#### Limitations

- `%s` format specifier has limited support for strings
- Format string parsing is less flexible than C printf
- Total output buffer is limited (CUDA printf limitation)
- Performance impact when printing frequently

### 23.1.2 ct.print() — Python-Style Printing

The `ct.print()` function provides Python-style printing with support for f-strings and multiple arguments.

#### Syntax

```python
ct.print(*args, sep=' ', end='\n', file=None)
```

#### Parameters

- `*args`: Values to print
- `sep`: Separator between values (default: space)
- `end`: String appended after output (default: newline)
- `file`: Output file (default: stdout)

#### Basic Examples

```python
import cuda.tile as ct

@ct.tile
def python_print(x: ct.tile.float32, y: ct.tile.float32):
    # Simple print
    ct.print(x)
    
    # Multiple values
    ct.print("x =", x, "y =", y)
    
    # Custom separator
    ct.print(x, y, sep=', ')
    
    # Custom end
    ct.print(x, end=' | ')
    
    return x + y
```

#### F-String Support

cuTile supports Python f-string syntax in device code (v1.2.0+):

```python
@ct.tile
def fstring_example(x: ct.tile.float32, idx: ct.tile.int32):
    # Basic f-string
    ct.print(f"Value: {x}")
    
    # Multiple expressions
    ct.print(f"Index {idx}, value {x}")
    
    # Expressions in f-strings
    ct.print(f"Double: {x * 2.0}")
    
    # Format specifications
    ct.print(f"Formatted: {x:.4f}")
    
    return x
```

#### Nested F-Strings (v1.3.0+)

```python
@ct.tile
def nested_fstring(x: ct.tile.float32, y: ct.tile.float32):
    # Nested expressions
    result = x + y
    ct.print(f"Result of {x} + {y} = {result}")
    
    # Complex nesting
    ct.print(f"Squared: {x * x}, Sum: {x + y}")
    
    return result
```

#### Printing Tuples (v1.3.0+)

```python
@ct.tile
def print_tuple(data: ct.tile.float32):
    # Create and print tuple
    values = (data, data * 2.0, data * 3.0)
    ct.print("Tuple:", values)
    
    # Nested tuples
    nested = ((data, data), (data * 2.0, data * 3.0))
    ct.print("Nested:", nested)
    
    return data
```

#### Comparison: printf vs print

| Feature | ct.printf() | ct.print() |
|---------|-------------|------------|
| Format style | C-style | Python-style |
| F-strings | No | Yes (v1.2.0+) |
| Multiple arguments | Manual | Automatic |
| Type conversion | Manual specifiers | Automatic |
| Tuple support | No | Yes (v1.3.0+) |
| Performance | Slightly faster | Slightly slower |

## 23.2 Assertion Functions

### 23.2.1 ct.assert_() — Runtime Assertion

The `ct.assert_()` function provides runtime assertion checking in device code.

#### Syntax

```python
ct.assert_(condition, message="")
```

#### Parameters

- `condition`: Boolean expression to check
- `message`: Optional error message (future feature)

#### Basic Usage

```python
import cuda.tile as ct

@ct.tile
def safe_divide(a: ct.tile.float32, b: ct.tile.float32):
    # Assert denominator is not zero
    ct.assert_(b != 0.0, "Division by zero")
    
    return a / b

# This will raise an error if any element of b is zero
result = ct.launch(stream, grid, safe_divide, (a, b))
```

#### Array Assertions

```python
@ct.tile
def array_assert(x: ct.tile.float32):
    # Assert all values are positive
    ct.assert_(x > 0.0)
    
    # Assert value in range
    ct.assert_((x >= 0.0) & (x <= 1.0))
    
    return x
```

#### Debug Build Usage

Assertions are particularly useful in debug builds:

```python
@ct.tile
def debug_kernel(x: ct.tile.float32):
    # Expensive checks only in debug mode
    if DEBUG:
        ct.assert_(ct.isfinite(x))
        ct.assert_(x >= 0.0)
    
    # Computation
    return ct.sqrt(x)
```

#### Invariant Checking

Use assertions to check invariants:

```python
@ct.tile
def process_data(data: ct.tile.float32):
    # Precondition invariant
    ct.assert_(ct.isfinite(data))
    
    # Processing
    result = ct.sqrt(data * data + 1.0)
    
    # Postcondition invariant
    ct.assert_(result >= 0.0)
    
    return result
```

#### Assertion Behavior

When an assertion fails:
1. Kernel execution is halted
2. Error is raised on host
3. Error message includes location and condition
4. Device state may be inconsistent

## 23.3 Mathematical Utilities

### 23.3.1 ct.cdiv() — Ceiling Division

The `ct.cdiv()` function computes ceiling division, essential for calculating grid dimensions.

#### Syntax

```python
ct.cdiv(x, y)  # Returns ceil(x / y)
```

#### Parameters

- `x`: Dividend (number of elements)
- `y`: Divisor (elements per thread/block)

#### Return Value

Returns the smallest integer greater than or equal to x / y.

#### Basic Usage

```python
import cuda.tile as ct

# Calculate grid dimensions
N = 1000
TILE_SIZE = 128
grid_x = ct.cdiv(N, TILE_SIZE)  # = 8

grid = (grid_x, TILE_SIZE)
```

#### Common Pattern

The most common use case is calculating grid dimensions:

```python
@ct.tile
def kernel(x: ct.tile.float32):
    return x * 2.0

# Standard launch pattern
N = 1024
TILE_SIZE = 128
grid = (ct.cdiv(N, TILE_SIZE), TILE_SIZE)
stream = ct.Stream()

x = ct.random.random(N, dtype=ct.float32)
result = ct.launch(stream, grid, kernel, (x,))
```

#### Multi-Dimensional Grids

```python
# 2D grid
rows, cols = 1024, 2048
tile_rows, tile_cols = 16, 16

grid = (
    ct.cdiv(rows, tile_rows),   # Grid X
    ct.cdiv(cols, tile_cols),   # Grid Y
    (tile_rows, tile_cols)      # Tile/block size
)
```

#### Works in Both Host and Device Code

```python
# Host code
grid_x = ct.cdiv(N, TILE_SIZE)

# Device code
@ct.tile
def device_cdiv():
    n_threads = 1024
    n_blocks = ct.cdiv(n_threads, 32)
    ct.print(f"Need {n_blocks} blocks")
    return n_blocks
```

#### Edge Cases

```python
# Exact division
ct.cdiv(100, 25)  # = 4

# Ceiling needed
ct.cdiv(101, 25)  # = 5

# Divisor larger than dividend
ct.cdiv(10, 100)  # = 1

# Power of 2
ct.cdiv(1024, 128)  # = 8
```

#### Performance Note

`ct.cdiv()` is optimized for common patterns and compiles to efficient CUDA code:

```cpp
// Compiled CUDA code (simplified)
__device__ int cdiv(int x, int y) {
    return (x + y - 1) / y;
}
```

For power-of-2 divisors, this becomes a bit shift:
```cpp
__device__ int cdiv_pow2(int x, int y) {
    return (x + y - 1) >> (log2(y));
}
```

## 23.4 Block and Grid Functions

### 23.4.1 ct.bid() — Block Index

Returns the block index for a given axis.

#### Syntax

```python
ct.bid(axis)
```

#### Parameters

- `axis`: Axis index (0 for X, 1 for Y, 2 for Z)

#### Return Value

Returns the current block's index along the specified axis.

#### Basic Usage

```python
import cuda.tile as ct

@ct.tile
def block_index_kernel(x: ct.tile.float32):
    # Get block index in X dimension
    block_x = ct.bid(0)
    
    # Use block index in computation
    offset = block_x * 128  # Assuming 128 threads per block
    
    return x + offset
```

#### Multi-Dimensional Block Indices

```python
@ct.tile
def multi_dim_kernel(x: ct.tile.float32):
    block_x = ct.bid(0)
    block_y = ct.bid(1)
    block_z = ct.bid(2)
    
    # Compute global position
    global_id = (block_z * grid_y + block_y) * grid_x + block_x
    
    ct.print(f"Block: ({block_x}, {block_y}, {block_z})")
    
    return x
```

#### Debugging with Block Indices

```python
@ct.tile
def debug_by_block(x: ct.tile.float32):
    block_id = ct.bid(0)
    
    # Only print from first block
    if block_id == 0:
        ct.print(f"First block processing: {x}")
    
    return x * 2.0
```

### 23.4.2 ct.num_blocks() — Number of Blocks

Returns the total number of blocks in the grid for a given axis.

#### Syntax

```python
ct.num_blocks(axis)
```

#### Parameters

- `axis`: Axis index (0 for X, 1 for Y, 2 for Z)

#### Return Value

Returns the total number of blocks along the specified axis.

#### Basic Usage

```python
@ct.tile
def count_blocks_kernel(x: ct.tile.float32):
    # Total blocks in X dimension
    total_blocks = ct.num_blocks(0)
    
    ct.print(f"Total blocks: {total_blocks}")
    
    return x
```

#### Conditional Execution Based on Block Count

```python
@ct.tile
def conditional_kernel(x: ct.tile.float32):
    block_id = ct.bid(0)
    total_blocks = ct.num_blocks(0)
    
    # Only last block does special processing
    if block_id == total_blocks - 1:
        # Handle remainder elements
        pass
    
    return x
```

### 23.4.3 ct.num_tiles() — Number of Tiles

Returns the number of tiles in a tiled view along a given axis.

#### Syntax

```python
ct.num_tiles(axis)
```

#### Parameters

- `axis`: Axis index in the tiled view

#### Return Value

Returns the number of tiles along the specified axis.

#### Usage with Tiled Views

```python
@ct.tile
def tiled_kernel(data: ct.tile.float32):
    # Create tiled view
    tiled = data.tiled_view(tile_shape=(16, 16))
    
    # Get tile dimensions
    num_tiles_x = ct.num_tiles(0)
    num_tiles_y = ct.num_tiles(1)
    
    ct.print(f"Tile grid: {num_tiles_x} x {num_tiles_y}")
    
    # Process tiles
    for tile_i in ct.static_range(num_tiles_x):
        for tile_j in ct.static_range(num_tiles_y):
            tile = tiled[tile_i, tile_j]
            # Process tile...
    
    return data
```

## 23.5 Tuple Operations

### 23.5.1 Tuple Basics

Tuples can be used in tile code but cannot be kernel parameters.

#### Creating Tuples

```python
import cuda.tile as ct

@ct.tile
def make_tuple(x: ct.tile.float32, y: ct.tile.float32):
    # Create tuple
    pair = (x, y)
    triple = (x, y, x + y)
    
    return x  # Can't return tuple from kernel (yet)
```

#### Tuple Limitations

```python
# NOT ALLOWED: Tuple as kernel parameter
@ct.tile
def kernel_with_tuple_param(data: ct.tile(ct.tile.float32, ct.tile.float32)):
    pass  # Error!

# ALLOWED: Tuple as local variable
@ct.tile
def kernel_with_local_tuple(x: ct.tile.float32):
    local_tuple = (x, x * 2.0)
    return x
```

### 23.5.2 Tuple Concatenation (v1.2.0+)

Tuples support concatenation using the `+` operator.

#### Basic Concatenation

```python
@ct.tile
def concat_tuples(x: ct.tile.float32, y: ct.tile.float32, z: ct.tile.float32):
    # Concatenate tuples
    pair1 = (x, y)
    pair2 = (y, z)
    quadruple = pair1 + pair2  # (x, y, y, z)
    
    return x
```

#### Practical Example

```python
@ct.tile
def build_result_tuple(x: ct.tile.float32, y: ct.tile.float32):
    # Build complex tuple
    base = (x, y)
    computed = (x * 2.0, y * 3.0)
    extended = (x + y, x - y)
    
    # Concatenate all
    result = base + computed + extended
    
    # result = (x, y, x*2, y*3, x+y, x-y)
    return x
```

### 23.5.3 Nested Tuple Unpacking

cuTile supports nested tuple unpacking syntax.

#### Basic Unpacking

```python
@ct.tile
def unpack_tuple(x: ct.tile.float32, y: ct.tile.float32):
    # Create nested tuple
    nested = ((x, y), (x * 2.0, y * 2.0))
    
    # Unpack nested structure
    (a, b), (c, d) = nested
    
    # a = x, b = y, c = x*2, d = y*2
    return a + b + c + d
```

#### Complex Unpacking

```python
@ct.tile
def complex_unpack(x: ct.tile.float32, y: ct.tile.float32, z: ct.tile.float32):
    # Create deeply nested structure
    deep = (((x, y), z), (x * 2.0, (y, z)))
    
    # Unpack
    ((a, b), c), (d, (e, f)) = deep
    
    # a=x, b=y, c=z, d=x*2, e=y, f=z
    return a + b + c + d + e + f
```

### 23.5.4 Square Bracket Unpacking

cuTile supports Python 3.10+ style square bracket unpacking.

#### Basic Syntax

```python
@ct.tile
def bracket_unpack(x: ct.tile.float32, y: ct.tile.float32):
    # Square bracket unpacking
    [a, b] = (x, y)
    
    # Same as: a, b = x, y
    return a + b
```

#### Nested Square Brackets

```python
@ct.tile
def nested_bracket_unpack(x: ct.tile.float32, y: ct.tile.float32):
    data = ((x, y), (x * 2.0, y * 2.0))
    
    # Mixed brackets and parens
    [a, b], (c, d) = data
    
    return a + b + c + d
```

### 23.5.5 Tuple Use Cases

#### Multiple Return Values (Future)

```python
# Future feature: Multiple returns
@ct.tile
def compute_multiple(x: ct.tile.float32):
    doubled = x * 2.0
    tripled = x * 3.0
    return (doubled, tripled)  # Will be supported
```

#### Data Grouping

```python
@ct.tile
def process_groups(x: ct.tile.float32):
    # Group related computations
    originals = (x, x * 2.0, x * 3.0)
    squares = (x * x, (x * 2.0) * (x * 2.0))
    
    # Use groups for different processing
    all_data = originals + squares
    
    return x
```

## 23.6 Type Conversion Utilities

### 23.6.1 ct.cast()

Explicit type conversion between tile types.

#### Syntax

```python
ct.cast(value, dtype)
```

#### Basic Usage

```python
@ct.tile
def type_conversion(x: ct.tile.float32):
    # Cast to different types
    as_int16 = ct.cast(x, ct.tile.int16)
    as_float16 = ct.cast(x, ct.tile.float16)
    as_float64 = ct.cast(x, ct.tile.float64)
    
    return as_float64
```

#### Safe Conversions

```python
@ct.tile
def safe_cast(x: ct.tile.float32):
    # Check before conversion
    if x >= 0.0 and x < 256.0:
        as_uint8 = ct.cast(x, ct.tile.uint8)
    else:
        as_uint8 = ct.tile.uint8(255)
    
    return ct.cast(as_uint8, ct.tile.float32)
```

## 23.7 Mathematical Functions

### 23.7.1 ct.isnan()

Check if values are NaN (Not a Number).

#### Syntax

```python
ct.isnan(x)
```

#### Basic Usage

```python
@ct.tile
def check_nan(x: ct.tile.float32):
    # Check for NaN
    if ct.isnan(x):
        ct.print("Found NaN!")
        return ct.tile.float32(0.0)
    
    return x
```

#### Array Validation

```python
@ct.tile
def validate_array(arr: ct.tile.float32[1024]):
    # Check all elements
    for i in ct.static_range(1024):
        if ct.isnan(arr[i]):
            ct.print(f"NaN at index {i}")
            arr[i] = 0.0
    
    return arr
```

### 23.7.2 ct.isfinite()

Check if values are finite (not NaN or infinite).

#### Syntax

```python
ct.isfinite(x)
```

#### Usage

```python
@ct.tile
def safe_computation(x: ct.tile.float32):
    # Validate input
    if not ct.isfinite(x):
        ct.print("Invalid input!")
        return ct.tile.float32(0.0)
    
    # Safe to compute
    result = ct.sqrt(x)
    return result
```

## 23.8 Static Evaluation Functions

### 23.8.1 ct.static_eval()

Evaluate expressions at compile time.

#### Syntax

```python
ct.static_eval(expression)
```

#### Basic Usage

```python
@ct.tile
def static_example():
    # Compile-time computation
    size = ct.static_eval(16 * 16)
    
    # size is constant at compile time
    ct.print(f"Size: {size}")
    
    return size
```

#### Compile-Time Constants

```python
@ct.tile
def use_constant():
    # Compute tile size at compile time
    TILE_SIZE = ct.static_eval(128)
    GRID_SIZE = ct.static_eval(1024 / TILE_SIZE)
    
    # These become constants in generated code
    return GRID_SIZE
```

### 23.8.2 ct.static_assert()

Assert conditions at compile time.

#### Syntax

```python
ct.static_assert(condition, message)
```

#### Basic Usage

```python
@ct.tile
def static_check():
    # Compile-time assertion
    ct.static_assert(sizeof(ct.tile.float32) == 4, "Float32 must be 4 bytes")
    
    return ct.tile.float32(0.0)
```

#### Type Checking

```python
@ct.tile
def check_types():
    # Verify type properties
    ct.static_assert(ct.tile.int64.min < 0, "int64 should be signed")
    ct.static_assert(ct.tile.uint64.min >= 0, "uint64 should be unsigned")
    
    return ct.tile.int32(0)
```

### 23.8.3 ct.static_iter()

Iterate at compile time (unrolls loops).

#### Syntax

```python
ct.static_iter(function, start, end)
```

#### Basic Usage

```python
@ct.tile
def unroll_loop(x: ct.tile.float32):
    # Unroll loop at compile time
    def process(i):
        ct.print(f"Iteration {i}")
    
    ct.static_iter(process, 0, 4)
    # Unrolls to:
    # process(0)
    # process(1)
    # process(2)
    # process(3)
    
    return x
```

#### Compile-Time Unrolling

```python
@ct.tile
def vector_add(a: ct.tile.float32[4], b: ct.tile.float32[4]):
    result = ct.array([0.0, 0.0, 0.0, 0.0], dtype=ct.float32)
    
    def add_component(i):
        result[i] = a[i] + b[i]
    
    # Unroll loop for performance
    ct.static_iter(add_component, 0, 4)
    
    return result
```

## 23.9 Advanced Utilities

### 23.9.1 ct.min() and ct.max()

Element-wise minimum and maximum.

#### Syntax

```python
ct.min(x, y)
ct.max(x, y)
```

#### Basic Usage

```python
@ct.tile
def clamp(x: ct.tile.float32, min_val: ct.tile.float32, max_val: ct.tile.float32):
    # Clamp value to range
    return ct.max(min_val, ct.min(x, max_val))
```

#### Array Operations

```python
@ct.tile
def reduce_min_max(arr: ct.tile.float32[1024]):
    # Find min and max
    current_min = arr[0]
    current_max = arr[0]
    
    for i in ct.static_range(1, 1024):
        current_min = ct.min(current_min, arr[i])
        current_max = ct.max(current_max, arr[i])
    
    return (current_min, current_max)
```

### 23.9.2 ct.clamp()

Clamp values to a range (convenience function).

#### Syntax

```python
ct.clamp(x, min_val, max_val)
```

#### Basic Usage

```python
@ct.tile
def clamp_values(x: ct.tile.float32):
    # Clamp to [0, 1]
    return ct.clamp(x, 0.0, 1.0)
```

#### Image Processing Example

```python
@ct.tile
def process_pixel(pixel: ct.tile.float32):
    # Ensure pixel values stay valid
    valid = ct.clamp(pixel, 0.0, 255.0)
    return valid
```

## 23.10 Performance Considerations

### 23.10.1 Printing Overhead

Printing functions have significant performance overhead:

```python
# BAD: Printing in tight loop
@ct.tile
def slow_kernel(x: ct.tile.float32):
    for i in ct.static_range(1000):
        ct.print(f"Iteration {i}")  # Very slow!
    return x

# GOOD: Conditional printing
@ct.tile
def fast_kernel(x: ct.tile.float32):
    for i in ct.static_range(1000):
        if i % 100 == 0:  # Print every 100 iterations
            ct.print(f"Iteration {i}")
    return x
```

### 23.10.2 Assertion Overhead

Assertions have minimal overhead when disabled:

```python
# Debug mode: Assertions enabled
DEBUG = True

@ct.tile
def debug_kernel(x: ct.tile.float32):
    if DEBUG:
        ct.assert_(x >= 0.0)
    
    return ct.sqrt(x)

# Release mode: Assertions disabled (faster)
DEBUG = False
```

### 23.10.3 Static vs Dynamic

Prefer static operations when possible:

```python
# GOOD: Compile-time computation
@ct.tile
def good_kernel():
    size = ct.static_eval(128 * 128)
    return size

# AVOIDABLE: Runtime computation
@ct.tile
def avoidable_kernel():
    size = 128 * 128  # Still computed at runtime
    return size
```

## 23.11 Debugging Tips

### 23.11.1 Strategic Printing

Place print statements strategically:

```python
@ct.tile
def debug_kernel(x: ct.tile.float32):
    # Print input
    ct.print(f"Input: {x}")
    
    # Computation
    y = x * 2.0
    
    # Print intermediate
    if ct.bid(0) == 0:  # Only first block
        ct.print(f"Intermediate: {y}")
    
    # More computation
    z = y + 1.0
    
    # Print output
    ct.print(f"Output: {z}")
    
    return z
```

### 23.11.2 Assertion Placement

Place assertions at function boundaries:

```python
@ct.tile
def robust_kernel(x: ct.tile.float32):
    # Input validation
    ct.assert_(ct.isfinite(x))
    ct.assert_(x >= 0.0)
    
    # Computation
    result = ct.sqrt(x)
    
    # Output validation
    ct.assert_(ct.isfinite(result))
    ct.assert_(result >= 0.0)
    
    return result
```

### 23.11.3 Block-Level Debugging

Use block indices to limit debug output:

```python
@ct.tile
def selective_debug(x: ct.tile.float32):
    block_id = ct.bid(0)
    
    # Only debug specific blocks
    if block_id == 0:
        ct.print(f"Block 0: {x}")
    elif block_id == ct.num_blocks(0) - 1:
        ct.print(f"Last block: {x}")
    
    return x * 2.0
```

## 23.12 Summary

cuTile's utility functions provide:

- **Debugging**: `ct.printf()`, `ct.print()`, `ct.assert_()`
- **Mathematical**: `ct.cdiv()`, `ct.min()`, `ct.max()`, `ct.clamp()`
- **Type operations**: `ct.cast()`, `ct.isnan()`, `ct.isfinite()`
- **Control flow**: `ct.bid()`, `ct.num_blocks()`, `ct.num_tiles()`
- **Compile-time**: `ct.static_eval()`, `ct.static_assert()`, `ct.static_iter()`
- **Data structures**: Tuples with concatenation and unpacking

These utilities enable efficient, correct, and debuggable tile code while maintaining compatibility with CUDA programming models.
