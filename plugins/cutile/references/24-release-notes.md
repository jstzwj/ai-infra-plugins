# Chapter 24: Release Notes

## Overview

This chapter provides a comprehensive history of cuTile releases, including new features, enhancements, bug fixes, and breaking changes. Understanding the version history helps you track capabilities, plan upgrades, and identify compatibility considerations.

## 24.1 Version 1.3.0 (2026-04-20)

### 24.1.1 Major Features

#### AOT Compilation via `compilation.export_kernel()`

The most significant feature in v1.3.0 is Ahead-Of-Time (AOT) compilation, enabling kernel compilation without runtime dependencies.

**Basic Usage:**

```python
import cuda.tile as ct

@ct.tile
def my_kernel(x: ct.tile.float32):
    return x * 2.0

# Export kernel for AOT compilation
exported = ct.compilation.export_kernel(
    my_kernel,
    output_path="kernels/my_kernel.cubin",
    arch="sm80"  # Target architecture
)

# Later, load and use the pre-compiled kernel
loaded_kernel = ct.compilation.load_kernel("kernels/my_kernel.cubin")
result = ct.launch(stream, grid, loaded_kernel, (data,))
```

**Benefits:**

- No compilation delay at runtime
- No Python dependency at deployment
- Faster application startup
- Easier distribution of GPU applications
- Better for production environments

**Advanced Options:**

```python
# Export with specific optimizations
exported = ct.compilation.export_kernel(
    my_kernel,
    output_path="kernels/my_kernel.cubin",
    arch="sm80",
    opt_level=3,  # Maximum optimization
    debug=False,  # Release build
    verbose=True  # Show compilation details
)
```

**Supported Architectures:**

- `sm50` - Maxwell (compute capability 5.0)
- `sm60` - Pascal (compute capability 6.0)
- `sm70` - Volta (compute capability 7.0)
- `sm75` - Turing (compute capability 7.5)
- `sm80` - Ampere (compute capability 8.0)
- `sm89` - Ada Lovelace (compute capability 8.9)
- `sm90` - Hopper (compute capability 9.0)

#### Autotuning Framework

Comprehensive autotuning support for finding optimal kernel configurations.

**Exhaustive Search:**

```python
import cuda.tile as ct

@ct.tile
def matmul(A: ct.tile.float32[1024, 1024], 
           B: ct.tile.float32[1024, 1024]):
    return A @ B

# Define search space
configs = [
    {'tile_size': 16, 'block_size': 128},
    {'tile_size': 32, 'block_size': 256},
    {'tile_size': 64, 'block_size': 256},
]

# Exhaustive search
best_config = ct.tune.exhaustive_search(
    kernel=matmul,
    inputs=(A, B),
    configs=configs,
    metric='execution_time',  # or 'throughput', 'memory_usage'
    iterations=10,  # Runs per config
    warmup=2,  # Warmup runs
)

print(f"Best config: {best_config}")
```

**Helper Functions:**

```python
# Automatic configuration generation
configs = ct.tune.generate_configs(
    tile_sizes=[16, 32, 64, 128],
    block_sizes=[128, 256, 512],
)

# Bayesian optimization (faster than exhaustive)
best_config = ct.tune.bayesian_search(
    kernel=my_kernel,
    inputs=test_data,
    n_trials=50,
)
```

**Custom Metrics:**

```python
# Define custom tuning metric
def custom_metric(result, inputs, timing):
    # Consider both speed and accuracy
    accuracy = compute_accuracy(result, inputs)
    score = accuracy / timing
    return score

best_config = ct.tune.exhaustive_search(
    kernel=approx_kernel,
    inputs=test_data,
    configs=configs,
    metric=custom_metric,
)
```

#### Array.tiled_view() API

New API for creating tiled views of arrays with explicit control over tile layout.

**Basic Usage:**

```python
import cuda.tile as ct

arr = ct.random.random((1024, 1024), dtype=ct.float32)

# Create tiled view
tiled = arr.tiled_view(
    tile_shape=(16, 16),
    memory_order='row_major'
)

# Access tiles
for i in range(tiled.num_tiles[0]):
    for j in range(tiled.num_tiles[1]):
        tile = tiled[i, j]
        # Process 16x16 tile
        result[i, j] = process_tile(tile)
```

**Advanced Options:**

```python
# Custom memory layout
tiled = arr.tiled_view(
    tile_shape=(32, 32),
    memory_order='col_major',
    padding=True,  # Pad tiles for alignment
    tile_stride=(1, 1),  # Overlapping tiles
)

# Query tile properties
print(f"Tile shape: {tiled.tile_shape}")
print(f"Grid shape: {tiled.grid_shape}")
print(f"Number of tiles: {tiled.num_tiles}")
```

**Integration with Kernels:**

```python
@ct.tile
def process_tiled(data: ct.tile.float32):
    tiled = data.tiled_view(tile_shape=(16, 16))
    
    for tile_i in ct.static_range(tiled.num_tiles[0]):
        for tile_j in ct.static_range(tiled.num_tiles[1]):
            tile = tiled[tile_i, tile_j]
            # Process tile
            result[tile_i, tile_j] = tile * 2.0
    
    return data
```

### 24.1.2 Enhancements

#### Memory Order and Memory Scope on Load/Store

Fine-grained control over memory operations for advanced optimization.

**Memory Order:**

```python
@ct.tile
def memory_ops(x: ct.tile.float32):
    # Load with specific memory ordering
    value = ct.load(
        ptr,
        memory_order='relaxed'  # or 'acquire', 'release', 'seq_cst'
    )
    
    # Store with specific memory ordering
    ct.store(
        ptr,
        value,
        memory_order='release'
    )
    
    return value
```

**Memory Scope:**

```python
# Control memory visibility scope
@ct.tile
def scoped_ops(x: ct.tile.float32):
    # Thread-local scope (default)
    value1 = ct.load(ptr, memory_scope='cta')
    
    # Cluster scope (Hopper+)
    value2 = ct.load(ptr, memory_scope='cluster')
    
    # GPU scope
    value3 = ct.load(ptr, memory_scope='gpu')
    
    # System scope (multi-GPU)
    value4 = ct.load(ptr, memory_scope='sys')
    
    return value1
```

**Use Cases:**

- Producer-consumer patterns
- Cross-thread synchronization
- Lock-free algorithms
- Performance optimization

#### Improved print() for Tuple and Nested F-String

Enhanced printing capabilities for better debugging.

**Tuple Printing:**

```python
@ct.tile
def print_tuples(x: ct.tile.float32, y: ct.tile.float32):
    # Print tuples
    pair = (x, y)
    ct.print("Pair:", pair)
    
    # Print nested tuples
    nested = ((x, y), (x*2, y*2))
    ct.print("Nested:", nested)
    
    return x
```

**Nested F-Strings:**

```python
@ct.tile
def nested_fstrings(x: ct.tile.float32):
    # Complex nested expressions
    result = x * 2.0
    ct.print(f"Result of {x} * 2 = {result}")
    
    # Multiple levels of nesting
    y = x + 1.0
    ct.print(f"({x} + 1.0) * 2.0 = {(x + 1.0) * 2.0}")
    
    return result
```

### 24.1.3 Bug Fixes

#### Restricted Float DType with Simple Reduce/Scan

Fixed issue where floating-point types were incorrectly allowed in certain reduction operations.

**Before (v1.2.0):**

```python
# This would compile but produce incorrect results
@ct.tile
def bad_reduce(arr: ct.tile.float32):
    return ct.reduce(arr, ct.add)  # Wrong!
```

**After (v1.3.0):**

```python
# Now correctly requires explicit reduction
@ct.tile
def good_reduce(arr: ct.tile.float32):
    return ct.reduce(arr, ct.add, initial=ct.tile.float32(0.0))
```

**Impact:** Code that relied on this behavior needs explicit initial values.

### 24.1.4 ABI Changes

#### Omit Constant-Annotated Parameters from Kernel ABI

Parameters annotated with `ct.Constant` are now compile-time constants and not part of the kernel ABI.

**Before:**

```python
@ct.tile
def kernel(data: ct.tile.float32, size: ct.tile.int32):
    # size was a runtime parameter
    for i in ct.static_range(size):
        ct.print(data[i])
    return data
```

**After:**

```python
@ct.tile
def kernel(data: ct.tile.float32, size: ct.Constant[ct.tile.int32]):
    # size is now a compile-time constant
    for i in ct.static_range(size):
        ct.print(data[i])
    return data

# Size must be known at compile time
result = ct.launch(stream, grid, kernel, (data,), size=1024)
```

**Benefits:**

- Reduced kernel launch overhead
- Better compiler optimization
- Smaller kernel binary size
- Type safety for compile-time constants

### 24.1.5 Other Changes

- Improved error messages for type mismatches
- Better documentation and examples
- Performance improvements in tuple operations
- Enhanced compiler optimizations

## 24.2 Version 1.2.0 (2026-03-05)

### 24.2.1 CTK 13.2 Features

#### Ampere/Ada GPU Support

Added support for sm80 family GPUs (Ampere and Ada architectures).

**Supported GPUs:**

- NVIDIA A100 (sm80)
- NVIDIA RTX 3090, 3080, 3070 (sm86)
- NVIDIA RTX 4090, 4080, 4070 (sm89)
- NVIDIA L40 (sm89)

**Installation:**

```bash
# Install with Tile IR AS support
pip install cuda-tile[tileiras]

# Requires CUDA Toolkit 13.2 or later
```

#### New Mathematical Functions

**ct.atan2(y, x) - Arctangent with Two Arguments:**

```python
@ct.tile
def polar_to_cartesian(r: ct.tile.float32, theta: ct.tile.float32):
    x = r * ct.cos(theta)
    y = r * ct.sin(theta)
    
    # Compute angle from coordinates
    angle = ct.atan2(y, x)
    
    return angle
```

**ct.tanh() Rounding Mode Parameter:**

```python
@ct.tile
def tanh_with_rounding(x: ct.tile.float32):
    # Standard tanh
    result1 = ct.tanh(x)
    
    # With rounding mode
    result2 = ct.tanh(x, rounding_mode='rd')  # Round down
    result3 = ct.tanh(x, rounding_mode='ru')  # Round up
    result4 = ct.tanh(x, rounding_mode='rn')  # Round to nearest
    
    return result2
```

**Rounding Modes:**

- `'rd'` - Round down (toward negative infinity)
- `'ru'` - Round up (toward positive infinity)
- `'rz'` - Round toward zero
- `'rn'` - Round to nearest (default)

### 24.2.2 Major Features

#### Static Evaluation Functions

**ct.static_iter() - Compile-Time Iteration:**

```python
@ct.tile
def unroll_loop(x: ct.tile.float32):
    # Unroll loop at compile time
    def process(i):
        ct.print(f"Processing {i}")
    
    ct.static_iter(process, 0, 4)
    # Fully unrolls to 4 function calls
    
    return x
```

**ct.static_assert() - Compile-Time Assertions:**

```python
@ct.tile
def type_check():
    # Verify type properties at compile time
    ct.static_assert(sizeof(ct.tile.float32) == 4, "Float must be 4 bytes")
    ct.static_assert(ct.tile.int64.min < 0, "Int64 must be signed")
    
    return ct.tile.int32(0)
```

**ct.static_eval() - Compile-Time Evaluation:**

```python
@ct.tile
def compute_constants():
    # Evaluate at compile time
    TILE_SIZE = ct.static_eval(128)
    GRID_SIZE = ct.static_eval(1024 / TILE_SIZE)
    
    # These become constants in the binary
    return GRID_SIZE
```

#### Custom Scan Operations

**ct.scan() for Custom Scan Operations:**

```python
@ct.tile
def prefix_scan(arr: ct.tile.float32[1024]):
    # Custom scan with addition
    result = ct.scan(arr, ct.add)
    
    # First element is arr[0]
    # Last element is sum of all elements
    return result
```

**Custom Scan Function:**

```python
@ct.tile
def custom_scan(arr: ct.tile.float32[1024]):
    # Define custom scan operation
    def scan_op(a, b):
        return ct.max(a, b)  # Max-prefix scan
    
    result = ct.scan(arr, scan_op)
    
    return result
```

**Inclusive vs Exclusive:**

```python
# Inclusive scan (default)
result_inclusive = ct.scan(arr, ct.add, inclusive=True)

# Exclusive scan
result_exclusive = ct.scan(arr, ct.add, inclusive=False)
```

#### FP8 Type Support

**isnan() for FP8 Validation:**

```python
@ct.tile
def validate_fp8(x: ct.tile.float8_e4m3fn):
    # Check for NaN in FP8
    if ct.isnan(x):
        ct.print("Invalid FP8 value")
        return ct.tile.float8_e4m3fn(0.0)
    
    return x
```

#### Print Enhancements

**F-String Support:**

```python
@ct.tile
def print_with_fstring(x: ct.tile.float32, idx: ct.tile.int32):
    # Use f-strings in device code
    ct.print(f"Element {idx}: {x}")
    ct.print(f"Value: {x:.4f}")
    ct.print(f"Doubled: {x * 2.0}")
    
    return x
```

**ct.print() with F-Strings:**

```python
@ct.tile
def advanced_print(x: ct.tile.float32, y: ct.tile.float32):
    # Multiple expressions
    ct.print(f"x={x}, y={y}, sum={x+y}")
    
    # Format specifications
    ct.print(f"x: {x:.6f}, y: {y:.6f}")
    
    return x + y
```

#### Masked Gather/Scatter

**Mask Parameter for Gather:**

```python
@ct.tile
def masked_gather(data: ct.tile.float32[1024], 
                  indices: ct.tile.int32[100],
                  mask: ct.tile.bool[100]):
    # Only gather where mask is True
    result = ct.gather(data, indices, mask=mask)
    
    return result
```

**Mask Parameter for Scatter:**

```python
@ct.tile
def masked_scatter(values: ct.tile.float32[100],
                   indices: ct.tile.int32[100],
                   output: ct.tile.float32[1024],
                   mask: ct.tile.bool[100]):
    # Only scatter where mask is True
    ct.scatter(output, indices, values, mask=mask)
    
    return output
```

#### Tuple Operations

**Tuple Concatenation:**

```python
@ct.tile
def concat_tuples(x: ct.tile.float32, y: ct.tile.float32):
    # Concatenate tuples
    pair = (x, y)
    triple = pair + (x * 2.0,)
    
    return triple
```

**Nested Tuple Unpacking:**

```python
@ct.tile
def nested_unpack(x: ct.tile.float32, y: ct.tile.float32):
    # Nested structure
    data = ((x, y), (x * 2.0, y * 2.0))
    
    # Unpack
    (a, b), (c, d) = data
    
    return a + b + c + d
```

#### Bytecode-to-Cubin Disk Cache

Persistent caching of compiled kernels to disk.

**Enable Cache:**

```python
import cuda.tile as ct

# Enable disk cache
ct.set_cache_dir("/path/to/cache")

# First launch compiles and caches
result1 = ct.launch(stream, grid, kernel, (data,))

# Subsequent launches use cache (fast!)
result2 = ct.launch(stream, grid, kernel, (data,))
```

**Cache Configuration:**

```python
# Configure cache behavior
ct.config.cache_enabled = True
ct.config.cache_dir = "~/.cache/cutile"
ct.config.cache_size_mb = 1024  # Max cache size
```

**Cache Invalidation:**

```python
# Clear cache
ct.clear_cache()

# Disable cache
ct.config.cache_enabled = False
```

### 24.2.3 Bug Fixes

1. **Fixed memory leak in DLPack conversion for certain tensor types**
2. **Corrected handling of negative indices in gather/scatter**
3. **Fixed race condition in scan operations with custom operators**
4. **Resolved incorrect behavior of ct.tanh() with inf/nan inputs**
5. **Fixed compilation error with nested function calls in某些 cases**
6. **Corrected type inference for mixed-precision operations**
7. **Fixed stack overflow in deeply nested tuple unpacking**
8. **Resolved issue with opt_level=0 and printf/print functions**

### 24.2.4 Enhancements

1. **Improved error messages with source location information**
2. **Better type inference for complex expressions**
3. **Enhanced compiler optimizations for arithmetic operations**
4. **Improved performance of tuple operations**
5. **Better integration with CUDA streams**
6. **Enhanced support for complex number operations**
7. **Improved documentation and examples**
8. **Better warnings for potentially undefined behavior**

## 24.3 Version 1.1.0 (2026-01-30)

### 24.3.1 Major Features

#### Nested Functions and Lambdas

**Nested Functions:**

```python
@ct.tile
def outer(x: ct.tile.float32):
    # Define nested function
    def inner(y: ct.tile.float32):
        return y * 2.0
    
    # Use nested function
    result = inner(x)
    return result
```

**Lambdas:**

```python
@ct.tile
def use_lambda(x: ct.tile.float32):
    # Define lambda
    square = lambda y: y * y
    
    # Use lambda
    result = square(x)
    return result
```

**Capturing Variables:**

```python
@ct.tile
def capture_outer(x: ct.tile.float32):
    factor = 2.0
    
    # Nested function captures outer variable
    def scale(y: ct.tile.float32):
        return y * factor
    
    return scale(x)
```

#### Custom Reduction Operations

**ct.reduce() for Custom Reductions:**

```python
@ct.tile
def custom_reduce(arr: ct.tile.float32[1024]):
    # Custom reduction: sum of squares
    result = ct.reduce(arr, lambda x, y: x*x + y*y)
    return result
```

**Reduce with Initial Value:**

```python
@ct.tile
def reduce_with_init(arr: ct.tile.float32[1024]):
    # Reduce with explicit initial value
    result = ct.reduce(
        arr,
        ct.add,
        initial=ct.tile.float32(0.0)
    )
    return result
```

**Built-in Reduction Operations:**

```python
# Sum
sum_result = ct.reduce(arr, ct.add)

# Product
prod_result = ct.reduce(arr, ct.mul)

# Max
max_result = ct.reduce(arr, ct.max)

# Min
min_result = ct.reduce(arr, ct.min)
```

#### Array Slicing

**Array.slice() Method:**

```python
@ct.tile
def slice_array(arr: ct.tile.float32[1024]):
    # Slice first 100 elements
    first_100 = arr.slice(0, 100)
    
    # Slice middle elements
    middle = arr.slice(100, 200)
    
    # Slice with step
    every_10th = arr.slice(0, 1000, 10)
    
    return first_100
```

**Negative Indices:**

```python
@ct.tile
def slice_with_negative(arr: ct.tile.float32[1024]):
    # Last 100 elements
    last_100 = arr.slice(-100, -1)
    
    # All except first and last
    middle = arr.slice(1, -1)
    
    return last_100
```

**Slice Assignment:**

```python
@ct.tile
def slice_assign(arr: ct.tile.float32[1024]):
    # Assign to slice
    arr.slice(0, 100) = 0.0
    
    # Add to slice
    arr.slice(100, 200) += 1.0
    
    return arr
```

### 24.3.2 Bug Fixes

1. **Fixed incorrect results from gather with duplicate indices**
2. **Resolved segmentation fault in scatter with out-of-bounds indices**
3. **Fixed memory corruption in nested kernel launches**
4. **Corrected handling of NaN in comparison operations**
5. **Fixed type promotion issues in mixed-type operations**
6. **Resolved issue with stride calculation in multi-dimensional arrays**
7. **Fixed incorrect behavior of ct.sqrt() with negative inputs**

### 24.3.3 Enhancements

1. **Improved performance of reduction operations**
2. **Better error messages for type mismatches**
3. **Enhanced support for multi-dimensional slicing**
4. **Improved compiler optimizations for nested functions**
5. **Better integration with CUDA Python**
6. **Enhanced documentation for advanced features**
7. **Improved type inference for generic operations**

## 24.4 Version 1.0.1 (2025-12-18)

### 24.4.1 Bug Fixes

1. **Fixed critical memory leak in kernel launch with array arguments**
2. **Resolved incorrect results from ct.sin() and ct.cos() for large inputs**
3. **Fixed compilation error with certain template instantiations**
4. **Corrected handling of zero-sized arrays**
5. **Fixed issue with stream synchronization**
6. **Resolved segmentation fault in error handling**
7. **Fixed incorrect behavior of ct.pow() with negative base and fractional exponent**
8. **Corrected type conversion for bool to int**
9. **Fixed issue with multiple kernels in same module**
10. **Resolved incorrect results from atan2() for edge cases**

### 24.4.2 Enhancements

1. **Improved error messages with line numbers**
2. **Better handling of edge cases in math functions**
3. **Enhanced documentation with more examples**
4. **Improved performance of memory operations**
5. **Better support for complex number operations**
6. **Enhanced type checking for kernel arguments**
7. **Improved integration with PyTorch DLPack**
8. **Better handling of device synchronization**

### 24.4.3 Known Limitations

1. Limited support for FP8 types
2. No AOT compilation
3. Limited autotuning capabilities
4. No support for intra-kernel SIMT interoperability

## 24.5 Version 1.0.0 (2025-12-02)

### 24.5.1 Initial Release Features

#### Core Functionality

- **Tile-based programming model**: Define kernels with tile semantics
- **Automatic parallelization**: Compiler handles thread/block mapping
- **Type system**: Strong typing with inference
- **Array operations**: Support for multi-dimensional arrays
- **Memory management**: Automatic device memory management

#### Mathematical Functions

- Basic arithmetic: add, subtract, multiply, divide
- Trigonometric: sin, cos, tan, asin, acos, atan
- Exponential/logarithmic: exp, log, log10, sqrt, pow
- Other: abs, floor, ceil, round, min, max

#### Control Flow

- Conditional statements: if/else
- Loops: for, while
- Early returns: return statements

#### Memory Operations

- Load/store with various memory orders
- Gather/scatter operations
- Shared memory usage
- Global memory access

#### Interoperability

- PyTorch integration via DLPack
- CuPy integration via CUDA Array Interface
- NumPy array support
- Stream sharing between frameworks

#### Development Tools

- JIT compilation
- Error reporting
- Debugging support with printf

#### Documentation

- Getting started guide
- API reference
- Tutorial examples
- Performance best practices

### 24.5.2 Supported Platforms

- **Operating Systems**: Linux (x86_64), Windows (x86_64)
- **Python Versions**: 3.8, 3.9, 3.10, 3.11
- **CUDA Versions**: 11.8, 12.0, 12.1, 12.2, 12.3
- **GPU Architectures**: Maxwell (sm50), Pascal (sm60), Volta (sm70), Turing (sm75), Ampere (sm80)

### 24.5.3 Installation

```bash
# Basic installation
pip install cuda-tile

# With development dependencies
pip install cuda-tile[dev]

# With extra documentation
pip install cuda-tile[docs]
```

### 24.5.4 Known Limitations in v1.0.0

1. No support for FP8 types
2. Limited support for complex numbers
3. No AOT compilation
4. No autotuning framework
5. Limited debugging capabilities
6. No nested function support
7. No custom reduction operations
8. Limited stream management

## 24.6 Upgrade Guide

### 24.6.1 Upgrading from 1.0.x to 1.1.0

**Breaking Changes:** None

**New Features to Adopt:**

1. **Nested Functions**: Refactor repeated code into nested functions
2. **Custom Reductions**: Replace manual reduction loops with ct.reduce()
3. **Array Slicing**: Use slice() instead of manual index calculations

**Migration Example:**

```python
# Before v1.1.0
@ct.tile
def old_reduce(arr: ct.tile.float32[1024]):
    total = ct.tile.float32(0.0)
    for i in ct.static_range(1024):
        total += arr[i]
    return total

# After v1.1.0
@ct.tile
def new_reduce(arr: ct.tile.float32[1024]):
    return ct.reduce(arr, ct.add, initial=ct.tile.float32(0.0))
```

### 24.6.2 Upgrading from 1.1.0 to 1.2.0

**Breaking Changes:** None

**New Features to Adopt:**

1. **FP8 Types**: Use float8 for memory-constrained applications
2. **Static Evaluation**: Replace runtime constants with static_eval()
3. **Disk Cache**: Enable for faster kernel loading
4. **F-String Printing**: Simplify debug output

**Migration Example:**

```python
# Before v1.2.0
@ct.tile
def old_print(x: ct.tile.float32):
    ct.print("Value:", x)
    ct.printf("Value: %f\n", x)

# After v1.2.0
@ct.tile
def new_print(x: ct.tile.float32):
    ct.print(f"Value: {x:.4f}")  # Cleaner syntax
```

### 24.6.3 Upgrading from 1.2.0 to 1.3.0

**Breaking Changes:**

- Kernel ABI change for Constant-annotated parameters
- Restricted float dtype in simple reduce/scan

**Required Actions:**

1. **Update Constant Parameters:**
```python
# Before v1.3.0
@ct.tile
def kernel(data: ct.tile.float32, size: ct.tile.int32):
    # size was runtime parameter
    pass

# After v1.3.0
@ct.tile
def kernel(data: ct.tile.float32, size: ct.Constant[ct.tile.int32]):
    # size is compile-time constant
    pass
```

2. **Fix Reduce Operations:**
```python
# Before v1.3.0 (worked but incorrect)
result = ct.reduce(arr, ct.add)

# After v1.3.0 (explicit initial value)
result = ct.reduce(arr, ct.add, initial=ct.tile.float32(0.0))
```

**New Features to Adopt:**

1. **AOT Compilation**: Export kernels for production deployment
2. **Autotuning**: Optimize kernel configurations
3. **Tiled Views**: Better memory layout control
4. **Memory Ordering**: Advanced synchronization primitives

### 24.6.4 Compatibility Matrix

| Feature | 1.0.0 | 1.0.1 | 1.1.0 | 1.2.0 | 1.3.0 |
|---------|-------|-------|-------|-------|-------|
| Core tile operations | ✓ | ✓ | ✓ | ✓ | ✓ |
| PyTorch integration | ✓ | ✓ | ✓ | ✓ | ✓ |
| CuPy integration | ✓ | ✓ | ✓ | ✓ | ✓ |
| Nested functions | ✗ | ✗ | ✓ | ✓ | ✓ |
| Custom reductions | ✗ | ✗ | ✓ | ✓ | ✓ |
| Array slicing | ✗ | ✗ | ✓ | ✓ | ✓ |
| FP8 support | ✗ | ✗ | ✗ | ✓ | ✓ |
| Static evaluation | ✗ | ✗ | ✗ | ✓ | ✓ |
| Disk cache | ✗ | ✗ | ✗ | ✓ | ✓ |
| AOT compilation | ✗ | ✗ | ✗ | ✗ | ✓ |
| Autotuning | ✗ | ✗ | ✗ | ✗ | ✓ |
| Tiled views | ✗ | ✗ | ✗ | ✗ | ✓ |

## 24.7 Deprecation Notices

### 24.7.1 Deprecated in 1.3.0

- `ct.reduce()` without explicit initial value (will error in 2.0.0)
- `ct.scan()` without explicit inclusive/exclusive parameter (will default to inclusive in 2.0.0)

### 24.7.2 Future Deprecations (Planned for 2.0.0)

- Legacy printf style (migrate to ct.print with f-strings)
- Implicit type conversions (require explicit casts)
- Legacy memory ordering defaults (require explicit specification)

## 24.8 Release Schedule

### 24.8.1 Future Releases

**v1.4.0 (Planned: Q3 2026)**
- Intra-kernel SIMT interoperability
- Enhanced debugging tools
- Performance profiling integration
- More comprehensive autotuning

**v2.0.0 (Planned: Q4 2026)**
- Breaking changes for better API consistency
- Removed deprecated features
- Enhanced type system
- Improved error messages

**v2.1.0 (Planned: Q1 2027)**
- Support for new GPU architectures
- Additional mathematical functions
- Enhanced interoperability
- Performance improvements

### 24.8.2 LTS Policy

Starting with v2.0.0, cuTile will follow Long-Term Support (LTS) releases:

- **LTS releases**: Every 12 months (e.g., 2.0, 3.0)
- **Support duration**: 24 months for LTS releases
- **Regular releases**: Every 3 months
- **Support duration**: 6 months for regular releases

## 24.9 Contributing to Release Notes

Community contributions are welcome! If you find bugs or have suggestions:

1. **Report bugs**: Use GitHub Issues with version information
2. **Request features**: Use GitHub Discussions
3. **Submit PRs**: Follow contribution guidelines
4. **Documentation**: Help improve examples and guides

## 24.10 Summary

cuTile's release history shows steady evolution:

- **v1.0.0**: Initial release with core functionality
- **v1.0.1**: Bug fixes and stability improvements
- **v1.1.0**: Nested functions, custom reductions, array slicing
- **v1.2.0**: CTK 13.2 features, FP8 support, static evaluation
- **v1.3.0**: AOT compilation, autotuning, tiled views

Each release builds upon the previous, adding capabilities while maintaining backward compatibility (except where noted for major improvements). The roadmap continues to focus on performance, developer experience, and integration with the broader CUDA ecosystem.
