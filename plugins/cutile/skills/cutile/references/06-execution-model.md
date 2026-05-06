# Execution Model

## Overview

The cuTile execution model defines how programs are executed on GPU hardware, providing a structured abstraction over the underlying SIMT (Single Instruction, Multiple Threads) architecture. This chapter comprehensively covers the execution model, from the abstract machine definition to practical kernel launching and advanced optimization techniques.

## Abstract Machine Model

cuTile programs execute on an abstract machine that models GPU hardware at a high level, hiding low-level details while exposing essential parallelism concepts.

### Thread Blocks and Grids

The fundamental execution units are **thread blocks** (often called "blocks") organized into a **grid**:

- **Grid**: A 1D, 2D, or 3D array of blocks
- **Block**: A collection of threads that execute together and can share data
- **Thread**: Individual execution unit (abstracted away in cuTile)

```python
# Launching a kernel with a grid of blocks
grid_shape = (8, 8, 1)  # 8x8x1 grid of blocks (64 blocks total)
# Each block executes the kernel independently
ct.launch(stream=0, grid=grid_shape, kernel=my_kernel, kernel_args=args)
```

### Block-Level Parallelism

cuTile exposes **block-level parallelism** only, abstracting away individual threads:

- Each block executes the kernel body independently
- Within a block, operations execute in SIMT fashion
- No explicit thread identification or synchronization
- Compiler handles thread-level parallelism transparently

```python
@ct.kernel
def simple_kernel(input_array: ct.Array, output_array: ct.Array):
    # Load tile (all threads in block cooperate)
    tile = ct.load(input_array)
    
    # Process tile (parallel operation across block)
    result = tile * 2.0 + 1.0
    
    # Store tile (all threads cooperate)
    ct.store(output_array, result)
```

### Execution Units

**Scalar Operations**: Execute serially on a single thread within the block.

```python
@ct.function
def scalar_operations():
    # These operations execute serially on one thread
    scalar_val = ct.float32(3.14)
    result = scalar_val * 2.0
    return result
```

**Array Operations**: Execute in parallel across all threads in the block.

```python
@ct.function
def array_operations():
    # These operations execute in parallel across block
    tile = ct.full((16, 16), 1.0, dtype=ct.float32)
    result = tile * 2.0  # Parallel across 256 threads
    return result
```

## Blocks vs Tiles: Units of Execution vs Data

Understanding the distinction between blocks (execution units) and tiles (data units) is fundamental to cuTile programming.

### Block: Unit of Execution

A block represents a unit of execution with specific characteristics:

- Contains multiple threads executing in lockstep (SIMT)
- Has access to shared memory for thread communication
- Identified by its position in the grid
- Operates on one or more tiles of data
- Cannot synchronize with other blocks during execution

```python
@ct.kernel
def block_execution_example(global_array: ct.Array):
    # Each block loads its tile independently
    tile = ct.load(global_array)
    
    # Block processes its tile with parallel operations
    processed = ct.sqrt(tile) + 1.0
    
    # Block stores its result independently
    ct.store(global_array, processed)
```

### Tile: Unit of Data

A tile represents a unit of data with specific characteristics:

- Multidimensional collection of elements
- Processed by operations within a block
- Can exist in various memory spaces (registers, shared memory, global memory)
- May be transferred between memory spaces
- Independent of block configuration

```python
@ct.function
def tile_data_example():
    # Create tiles of different shapes
    tile_a = ct.full((16, 16), 1.0, dtype=ct.float32)
    tile_b = ct.full((32, 32), 2.0, dtype=ct.float32)
    tile_c = ct.full((8, 8, 8), 3.0, dtype=ct.float32)
    
    # Multiple tiles can exist in one block
    return tile_a, tile_b, tile_c
```

### Relationship Between Blocks and Tiles

- **One block operates on one or more tiles**: Each block processes tiles during execution
- **Tile shape independent of block configuration**: Tiles can have any valid shape
- **Multiple tiles per block**: Blocks can work with multiple tiles of different shapes
- **Data parallelism**: Different blocks process different tiles in parallel

```python
@ct.kernel
def multi_tile_block(global_a: ct.Array, global_b: ct.Array, 
                    global_c: ct.Array):
    # Single block operates on three tiles
    tile_a = ct.load(global_a)  # Load tile A
    tile_b = ct.load(global_b)  # Load tile B
    tile_c = ct.load(global_c)  # Load tile C
    
    # Process all three tiles
    result = tile_a * tile_b + tile_c
    
    # Store combined result
    ct.store(global_a, result)
```

## Execution Spaces

cuTile programs operate in three distinct execution spaces, each with different capabilities and restrictions.

### Host Code

**Characteristics**:
- Executes on CPU
- Full Python language support
- Manages device memory and kernel launches
- No direct access to device data

```python
# Host code example
def host_function():
    # Allocate device memory
    device_array = ct.zeros((1024, 1024), dtype=ct.float32)
    
    # Launch kernel
    grid = (8, 8)
    ct.launch(stream=0, grid=grid, kernel=my_kernel, 
              kernel_args=[device_array])
    
    # Synchronize and copy results
    ct.synchronize(stream=0)
    result = device_array.copy_to_host()
    return result
```

### SIMT Code

**Characteristics**:
- Executes on GPU in SIMT fashion
- Low-level thread parallelism
- Explicit synchronization primitives
- Direct hardware access

**Note**: SIMT code is rarely used directly in cuTile; the compiler generates SIMT code from tile code.

### Tile Code

**Characteristics**:
- Executes on GPU at block level
- Abstracts away thread-level details
- No thread synchronization (block operates as unit)
- Rich set of operations on tiles

```python
@ct.function
def tile_code_example():
    # Tile code operates on tiles, not threads
    tile = ct.full((16, 16), 1.0, dtype=ct.float32)
    
    # Operations are parallel across block
    result = tile * 2.0 + 1.0
    
    # No thread synchronization needed
    return result
```

### Execution Space Annotations

Functions are annotated to explicitly specify their execution space.

#### @ct.function Decorator

The `@ct.function` decorator marks a function as tile code:

```python
@ct.function(host=False, tile=True)
def annotated_function(input_tile: ct.Tile):
    # This function executes in tile space
    result = input_tile * 2.0
    return result
```

**Default Parameters**:
- `host=False`: Function cannot execute on host
- `tile=True`: Function executes in tile space

#### Execution Space Inheritance

Unannotated functions called by tile functions automatically inherit tile execution space:

```python
# Unannotated function
def helper_function(value: ct.Tile):
    return value + 1.0

# Tile function
@ct.function
def main_function():
    tile = ct.full((8, 8), 1.0, dtype=ct.float32)
    
    # helper_function inherits tile execution space
    result = helper_function(tile)
    
    return result
```

## Kernel Definition and Launching

Kernels are entry points for tile code execution on the GPU.

### @ct.kernel Decorator

The `@ct.kernel` decorator defines a kernel function:

```python
@ct.kernel
def my_kernel(input_array: ct.Array, output_array: ct.Array):
    # Load tile from global memory
    tile = ct.load(input_array)
    
    # Process tile
    result = tile * 2.0
    
    # Store result to global memory
    ct.store(output_array, result)
```

**Kernel Characteristics**:
- **Tile-only**: Kernels execute in tile space only
- **Entry point**: Kernels cannot be called directly from other tile functions
- **Direct launch**: Kernels are launched via `ct.launch()`
- **Grid execution**: Each block in the grid executes the kernel independently

### Kernel Parameters and Hints

Kernels accept parameters that control execution behavior:

**num_ctas**: Number of Cooperative Thread Arrays (CTAs) per cluster (power of 2, range 1-16).

```python
@ct.kernel(num_ctas=8)
def num_cta_kernel(array: ct.Array):
    tile = ct.load(array)
    ct.store(array, tile * 2.0)
```

**occupancy**: Target occupancy level (1-32 threads per CTA).

```python
@ct.kernel(occupancy=32)
def high_occupancy_kernel(array: ct.Array):
    tile = ct.load(array)
    ct.store(array, tile * 2.0)
```

**opt_level**: Optimization level (0-3).

```python
@ct.kernel(opt_level=3)
def optimized_kernel(array: ct.Array):
    # Maximum optimization
    tile = ct.load(array)
    ct.store(array, ct.sqrt(tile))
```

### Architecture-Specific Hints

The `ByTarget` class allows architecture-specific kernel parameters:

```python
from cuda.tile import ByTarget

@ct.kernel(
    num_ctas=ByTarget(
        default=4,
        ampere=8,      # NVIDIA Ampere architecture
        hopper=16      # NVIDIA Hopper architecture
    )
)
def adaptive_kernel(array: ct.Array):
    tile = ct.load(array)
    ct.store(array, tile * 2.0)
```

### Kernel Launching

Kernels are launched using the `ct.launch()` function:

```python
def launch_kernel():
    # Allocate device memory
    input_array = ct.zeros((1024, 1024), dtype=ct.float32)
    output_array = ct.zeros((1024, 1024), dtype=ct.float32)
    
    # Launch kernel
    stream = 0
    grid = (8, 8)  # 8x8 grid of blocks
    ct.launch(
        stream=stream,
        grid=grid,
        kernel=my_kernel,
        kernel_args=[input_array, output_array]
    )
    
    # Wait for completion
    ct.synchronize(stream)
```

**Launch Parameters**:
- `stream`: CUDA stream for execution (0 is default stream)
- `grid`: 1D, 2D, or 3D grid specification
- `kernel`: Kernel function to launch
- `kernel_args`: List of kernel arguments

### Kernel Hint Replacement

The `kernel.replace_hints()` method creates a new kernel with updated hints:

```python
# Original kernel
@ct.kernel(num_ctas=4, occupancy=16)
def base_kernel(array: ct.Array):
    tile = ct.load(array)
    ct.store(array, tile * 2.0)

# Create new kernel with different hints
optimized_kernel = base_kernel.replace_hints(num_ctas=8, occupancy=32)

# Each variant has separate JIT cache
ct.launch(stream=0, grid=(4, 4), kernel=base_kernel, 
          kernel_args=[array])
ct.launch(stream=0, grid=(4, 4), kernel=optimized_kernel, 
          kernel_args=[array])
```

## Python Subset for Tile Code

Tile code uses a restricted subset of Python to ensure compilability and predictable execution.

### Supported Language Features

**Basic Types**: Integers, floats, booleans, tuples, lists.

```python
@ct.function
def basic_types():
    integer = 42
    floating = 3.14
    boolean = True
    tuple_val = (1, 2, 3)
    list_val = [1, 2, 3]
    return integer, floating, boolean, tuple_val, list_val
```

**Control Flow**: `if`, `for`, `while` statements with arbitrary nesting.

```python
@ct.function
def control_flow(tile: ct.Tile):
    # If statement
    if tile.shape[0] > 16:
        result = tile * 2.0
    else:
        result = tile + 1.0
    
    # For loop
    accumulator = ct.float32(0.0)
    for i in range(10):
        accumulator = accumulator + tile
    
    # While loop
    count = 0
    while count < 5:
        count = count + 1
    
    return result, accumulator
```

**Functions**: Function definition and calling.

```python
@ct.function
def function_calls():
    def helper(x: ct.Tile):
        return x * 2.0
    
    tile = ct.full((8, 8), 1.0, dtype=ct.float32)
    result = helper(tile)
    return result
```

### Unsupported Language Features

**Exceptions**: No exception handling or raising.

```python
@ct.function
def no_exceptions():
    tile = ct.full((8, 8), 1.0, dtype=ct.float32)
    
    # NOT SUPPORTED
    # try:
    #     result = tile / 0.0
    # except ZeroDivisionError:
    #     result = tile
    
    # Alternative: Check before operation
    zero_tile = ct.full((8, 8), 0.0, dtype=ct.float32)
    is_zero = zero_tile == 0.0
    safe_result = ct.where(is_zero, tile, tile / zero_tile)
    
    return safe_result
```

**Coroutines**: No async/await or generator functions.

```python
# NOT SUPPORTED in tile code
@ct.function
async def async_function():  # Error
    pass

@ct.function
def generator_function():  # Error
    yield 1
```

**Dynamic Attributes**: No adding attributes to objects at runtime.

```python
@ct.function
def no_dynamic_attributes():
    tile = ct.full((8, 8), 1.0, dtype=ct.float32)
    
    # NOT SUPPORTED
    # tile.new_attribute = 42  # Error
    
    return tile
```

### Control Flow Limitations

**For Loop Step**: Must be strictly positive.

```python
@ct.function
def for_loop_steps():
    # SUPPORTED: Positive step
    for i in range(0, 10, 2):  # Step = 2
        pass
    
    # NOT SUPPORTED: Negative step
    # for i in range(10, 0, -1):  # Step = -1
    #     pass
    
    # Alternative: Use range with positive step
    for i in range(0, 10):
        j = 9 - i  # Reverse index
```

**Negative-Step Ranges**: Ranges with negative steps are not supported.

```python
@ct.function
def no_negative_ranges():
    # NOT SUPPORTED
    # for i in range(10, 0, -1):
    #     pass
    
    # Alternative: Count forward
    for i in range(10):
        j = 9 - i  # Equivalent to range(10, 0, -1)
        pass
```

## Object Model

The cuTile object model defines how objects behave in tile code.

### Immutability

All tile code objects are immutable, ensuring predictable behavior:

```python
@ct.function
def immutable_objects():
    # Tiles are immutable
    tile = ct.full((8, 8), 1.0, dtype=ct.float32)
    modified = tile * 2.0  # Creates new tile, doesn't modify original
    
    # Tuples are immutable
    coords = (0, 1, 2)
    # coords[0] = 5  # Error: cannot modify tuple
    
    return tile, modified
```

### Global Arrays

Global arrays are views that read/write device memory:

```python
@ct.function
def global_array_view(array: ct.Array):
    # Load from global memory
    tile = ct.load(array)
    
    # Process tile
    result = tile * 2.0
    
    # Store to global memory
    ct.store(array, result)
```

**Array Characteristics**:
- **View**: Arrays are views into device memory, not the memory itself
- **Immutable views**: The array view object is immutable
- **Memory access**: Arrays provide read/write access to device memory
- **Lifetime**: Arrays must remain valid until kernel completes

### Caller Responsibilities

The caller must ensure several conditions for correct execution:

**No Aliasing**: Arrays passed to a kernel must not alias (overlap in memory).

```python
# INCORRECT: Aliased arrays
def bad_launch():
    array = ct.zeros((1024, 1024), dtype=ct.float32)
    # Aliasing input and output
    ct.launch(stream=0, grid=(4, 4), kernel=my_kernel, 
              kernel_args=[array, array])  # WRONG!

# CORRECT: Separate arrays
def good_launch():
    input_array = ct.zeros((1024, 1024), dtype=ct.float32)
    output_array = ct.zeros((1024, 1024), dtype=ct.float32)
    ct.launch(stream=0, grid=(4, 4), kernel=my_kernel, 
              kernel_args=[input_array, output_array])  # OK
```

**Array Validity**: Arrays must remain valid until kernel completion.

```python
def array_lifetime():
    # INCORRECT: Array freed before kernel completes
    def bad_launch():
        array = ct.zeros((1024, 1024), dtype=ct.float32)
        ct.launch(stream=0, grid=(4, 4), kernel=my_kernel, 
                  kernel_args=[array])
        # array goes out of scope here - KERNEL MAY FAIL
    
    # CORRECT: Array persists until kernel completes
    def good_launch():
        array = ct.zeros((1024, 1024), dtype=ct.float32)
        ct.launch(stream=0, grid=(4, 4), kernel=my_kernel, 
                  kernel_args=[array])
        ct.synchronize(stream=0)  # Wait for kernel
        # array safe to go out of scope now
```

## Constant Expressions and Objects

Constant expressions are values known at compile time, enabling powerful optimizations.

### Compile-Time Constants

Values that can be determined at compile time:

```python
@ct.function
def compile_time_constants():
    # Literal values
    int_literal = 42
    float_literal = 3.14
    
    # Shape dimensions
    tile = ct.full((16, 16), 1.0, dtype=ct.float32)
    dim0 = tile.shape[0]  # Known at compile time: 16
    
    # Compile-time computations
    total_elements = dim0 * dim0  # 256
    
    return int_literal, float_literal, total_elements
```

### Constant Embedding

Constant embedding replaces parameters with literal values, creating specialized kernels:

```python
@ct.kernel
def embedded_constant_kernel(array: ct.Array, multiplier: ct.Constant[int]):
    # multiplier is embedded as literal constant
    tile = ct.load(array)
    result = tile * multiplier  # Compiler knows exact value
    ct.store(array, result)

# Launching with different constants creates specialized kernels
ct.launch(stream=0, grid=(4, 4), kernel=embedded_constant_kernel, 
          kernel_args=[array, 2])  # Specializes for multiplier=2
ct.launch(stream=0, grid=(4, 4), kernel=embedded_constant_kernel, 
          kernel_args=[array, 3])  # Specializes for multiplier=3
```

**JIT Cache**: Each unique constant value creates a separate kernel in the JIT cache.

### ct.Constant[T] Type Hint

The `ct.Constant[T]` type hint marks parameters for constant embedding:

```python
@ct.kernel
def specialized_kernel(
    array: ct.Array,
    size: ct.Constant[int],      # Embedded integer constant
    threshold: ct.Constant[float] # Embedded float constant
):
    # size and threshold are compile-time constants
    tile = ct.load(array)
    
    # Can use in conditionals
    if size > 16:
        result = tile * threshold
    else:
        result = tile + threshold
    
    ct.store(array, result)
```

### ct.ConstantAnnotation Class

The `ct.ConstantAnnotation` class provides advanced constant embedding control:

```python
from cuda.tile import ConstantAnnotation

# Define constant annotation
size_annotation = ct.ConstantAnnotation(
    name="size",
    dtype=ct.int32,
    default=16,
    min=1,
    max=1024
)

@ct.kernel
def annotated_kernel(
    array: ct.Array,
    size: Annotated[int, size_annotation]
):
    tile = ct.load(array)
    # size is embedded constant with validation
    result = tile[:size, :size]  # Uses constant size
    ct.store(array, result)
```

## Synchronization and Coordination

### No Intra-Block Synchronization

cuTile does not provide intra-block synchronization primitives:

```python
@ct.function
def no_intra_block_sync():
    tile = ct.full((16, 16), 1.0, dtype=ct.float32)
    
    # NOT SUPPORTED: No __syncthreads() or similar
    # All operations in block execute without explicit synchronization
    
    # Block operates as unit; no thread-level coordination needed
    result = tile * 2.0
    return result
```

### Inter-Block Coordination

Inter-block coordination is limited and must be explicitly managed:

```python
def inter_block_launch():
    array = ct.zeros((1024, 1024), dtype=ct.float32)
    
    # Launch multiple kernels sequentially
    ct.launch(stream=0, grid=(4, 4), kernel=first_kernel, 
              kernel_args=[array])
    ct.synchronize(stream=0)  # Wait for first kernel
    
    ct.launch(stream=0, grid=(4, 4), kernel=second_kernel, 
              kernel_args=[array])
    ct.synchronize(stream=0)  # Wait for second kernel
```

## Performance Optimization

### Occupancy Optimization

Occupancy refers to the ratio of active warps to maximum warps per SM:

```python
@ct.kernel(occupancy=32)  # Maximum occupancy
def high_occupancy_kernel(array: ct.Array):
    # More registers/shared memory per thread
    tile = ct.load(array)
    
    # Complex computation that benefits from high occupancy
    temp1 = ct.sqrt(tile)
    temp2 = ct.exp(temp1)
    temp3 = ct.log(temp2 + 1.0)
    
    ct.store(array, temp3)
```

### CTA Cluster Optimization

Multiple CTAs can work together as a cluster:

```python
@ct.kernel(num_ctas=8)  # 8 CTAs per cluster
def clustered_kernel(array: ct.Array):
    # Load tile
    tile = ct.load(array)
    
    # Process with cooperation between CTAs
    result = tile * 2.0
    
    ct.store(array, result)
```

### Optimization Levels

Different optimization levels trade compilation time for performance:

```python
@ct.kernel(opt_level=0)  # No optimization (fastest compilation)
def debug_kernel(array: ct.Array):
    tile = ct.load(array)
    ct.store(array, tile * 2.0)

@ct.kernel(opt_level=2)  # Moderate optimization
def standard_kernel(array: ct.Array):
    tile = ct.load(array)
    ct.store(array, ct.sqrt(tile))

@ct.kernel(opt_level=3)  # Maximum optimization (slowest compilation)
def production_kernel(array: ct.Array):
    tile = ct.load(array)
    result = ct.sqrt(ct.exp(tile) + 1.0)
    ct.store(array, result)
```

## Advanced Execution Patterns

### Pipeline Execution

Chaining multiple kernels for pipelined processing:

```python
def pipeline_execution(data):
    # Allocate intermediate buffers
    temp1 = ct.zeros((1024, 1024), dtype=ct.float32)
    temp2 = ct.zeros((1024, 1024), dtype=ct.float32)
    
    # Stage 1
    ct.launch(stream=0, grid=(4, 4), kernel=stage1_kernel, 
              kernel_args=[data, temp1])
    
    # Stage 2 (waits for stage 1)
    ct.synchronize(stream=0)
    ct.launch(stream=0, grid=(4, 4), kernel=stage2_kernel, 
              kernel_args=[temp1, temp2])
    
    # Stage 3 (waits for stage 2)
    ct.synchronize(stream=0)
    ct.launch(stream=0, grid=(4, 4), kernel=stage3_kernel, 
              kernel_args=[temp2, data])
    
    ct.synchronize(stream=0)
```

### Stream Parallelism

Using multiple streams for concurrent kernel execution:

```python
def stream_parallelism(data1, data2):
    # Create streams
    stream1 = ct.create_stream()
    stream2 = ct.create_stream()
    
    # Launch concurrent kernels
    ct.launch(stream=stream1, grid=(4, 4), kernel=process_kernel, 
              kernel_args=[data1])
    ct.launch(stream=stream2, grid=(4, 4), kernel=process_kernel, 
              kernel_args=[data2])
    
    # Wait for both streams
    ct.synchronize(stream1)
    ct.synchronize(stream2)
```

## Best Practices

### Kernel Design

1. **Maximize parallelism**: Design kernels to utilize full grid
2. **Minimize global memory access**: Use tiles to cache data
3. **Avoid bank conflicts**: Organize memory access patterns
4. **Use appropriate block sizes**: Match hardware characteristics

```python
@ct.kernel
def optimized_kernel(input_array: ct.Array, output_array: ct.Array):
    # Load tile once (minimize global memory access)
    tile = ct.load(input_array)
    
    # Process tile (maximize computation per load)
    temp1 = tile * 2.0
    temp2 = ct.sqrt(temp1)
    temp3 = ct.exp(temp2)
    result = temp3 + 1.0
    
    # Store result once
    ct.store(output_array, result)
```

### Memory Management

1. **Reuse memory**: Avoid repeated allocations
2. **Pool arrays**: Pre-allocate and reuse arrays
3. **Minimize transfers**: Keep data on device when possible

```python
def memory_efficient_processing():
    # Reuse arrays across multiple kernel launches
    work_buffer = ct.zeros((1024, 1024), dtype=ct.float32)
    
    for i in range(10):
        # Reuse work_buffer
        ct.launch(stream=0, grid=(4, 4), kernel=process_kernel, 
                  kernel_args=[work_buffer])
        ct.synchronize(stream=0)
```

### Error Handling

1. **Check launch status**: Verify kernel launches succeed
2. **Validate parameters**: Ensure arrays and parameters are valid
3. **Handle synchronization**: Properly wait for kernel completion

```python
def safe_kernel_launch(array):
    try:
        # Validate input
        if array.shape[0] % 16 != 0 or array.shape[1] % 16 != 0:
            raise ValueError("Array dimensions must be multiples of 16")
        
        # Launch kernel
        ct.launch(stream=0, grid=(4, 4), kernel=safe_kernel, 
                  kernel_args=[array])
        
        # Wait for completion
        ct.synchronize(stream=0)
        
    except Exception as e:
        print(f"Kernel launch failed: {e}")
        raise
```

## Conclusion

The cuTile execution model provides a high-level abstraction over GPU hardware while enabling efficient parallel computation. Key takeaways include:

- Blocks are units of execution, tiles are units of data
- Execution spaces (host, SIMT, tile) have distinct characteristics
- Kernels are launched with grid specifications and execution hints
- Tile code uses a restricted Python subset with immutability guarantees
- Constant embedding enables kernel specialization and optimization
- Proper memory management and synchronization are essential

Mastering the execution model enables you to write efficient, correct cuTile programs that fully leverage GPU parallelism while maintaining high-level abstractions that improve productivity and code maintainability.
