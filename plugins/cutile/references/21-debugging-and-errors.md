# 21. Debugging and Errors

This chapter covers comprehensive debugging strategies for cuTile kernels, including exception types, error handling, profiling tools, and troubleshooting techniques. Effective debugging is essential for developing high-performance GPU kernels.

## 21.1 Exception Types

cuTile provides specific exception types for different categories of errors. Understanding these exceptions helps diagnose and fix issues quickly.

### 21.1.1 TileSyntaxError

**Raised when:** The kernel contains unsupported Python syntax or language features.

**Common Causes:**
- Using Python features not supported by cuTile (e.g., classes, generators)
- Invalid syntax in kernel definitions
- Incorrect use of decorators

**Example:**
```python
import cuda.tile as ct

# ERROR: Class definitions not supported in kernels
@ct.kernel
def bad_kernel(x: ct.Buffer[float]):
    class Myclass:  # Raises TileSyntaxError
        pass

# ERROR: Generators not supported
@ct.kernel
def bad_generator(x: ct.Buffer[float]):
    yield x[0]  # Raises TileSyntaxError

# ERROR: List comprehensions with runtime values
@ct.kernel
def bad_listcomp(x: ct.Buffer[float], n: int):
    result = [x[i] for i in range(n)]  # May raise TileSyntaxError
```

**Solution:**
Rewrite the kernel using supported cuTile operations:
```python
@ct.kernel
def good_kernel(x: ct.Buffer[float], result: ct.Buffer[float]):
    # Use explicit loops instead of list comprehensions
    n = 128
    for i in range(n):
        result[i] = x[i]
```

### 21.1.2 TileTypeError

**Raised when:** Unexpected type or dtype is encountered during compilation or execution.

**Common Causes:**
- Mismatched buffer data types
- Incorrect type annotations
- Type conversion issues
- Passing wrong dtype to operations

**Example:**
```python
# ERROR: Type mismatch in buffer
@ct.kernel
def type_mismatch(
    x: ct.Buffer[float],  # Float buffer
    y: ct.Buffer[int]     # Int buffer
):
    tid = ct.tid()
    val = x[tid]  # Load float
    y[tid] = val  # ERROR: Cannot store float to int buffer

# ERROR: Wrong dtype for operation
@ct.kernel
def wrong_dtype(x: ct.Buffer[int]):
    tid = ct.tid()
    result = x[tid] * 1.5  # ERROR: Float literal with int buffer

# ERROR: Invalid cast
@ct.kernel
def invalid_cast(x: ct.Buffer[float]):
    tid = ct.tid()
    val = x[tid]
    # ERROR: Cannot cast float to complex (unsupported)
    complex_val = ct.cast(val, complex)
```

**Solution:**
Ensure type consistency throughout the kernel:
```python
@ct.kernel
def type_correct(
    x: ct.Buffer[float],
    y: ct.Buffer[float]
):
    tid = ct.tid()
    val = x[tid]  # Load float
    y[tid] = val  # Store float to float buffer
```

**Debugging TileTypeError:**
Enable TileIR logging to see type information:
```bash
export CUDA_TILE_LOGS=CUTILEIR
python3 my_kernel.py
```

### 21.1.3 TileValueError

**Raised when:** Unexpected Python value is encountered.

**Common Causes:**
- Invalid constant values
- Out-of-range parameters
- Invalid configuration values
- Negative values where positive expected

**Example:**
```python
# ERROR: Negative buffer size
@ct.kernel
def negative_size(x: ct.Buffer[float]):
    # Invalid: shape cannot be negative
    tile = ct.load(x, index=(0,), shape=(-16,))

# ERROR: Invalid grid dimension
@ct.kernel
def invalid_grid(x: ct.Buffer[float]):
    pass

# ERROR: Zero or negative thread count
stream = cuda.cuda.Stream()
grid = (1, 1, 1)  # OK
bad_grid = (0, 1, 1)  # ERROR: Invalid grid dimension
# kernel[stream, bad_grid](args)  # Raises TileValueError
```

**Solution:**
Validate parameters before use:
```python
@ct.kernel
def validated_kernel(x: ct.Buffer[float], SIZE: ct.Constant[int]):
    # Add static assertion
    ct.static_assert(SIZE > 0, "SIZE must be positive")
    ct.static_assert(SIZE <= 1024, "SIZE must be <= 1024")
    
    tile = ct.load(x, index=(0,), shape=(SIZE,))
```

### 21.1.4 TileUnsupportedFeatureError

**Raised when:** Feature is not supported by the compiler or target GPU architecture.

**Common Causes:**
- Using GPU features not available on target architecture
- Requesting operations beyond hardware capabilities
- Using compiler features not supported by cuTile version

**Example:**
```python
# ERROR: TMA not available on pre-Hopper architectures
@ct.kernel
def tma_kernel(x: ct.Buffer[float], y: ct.Buffer[float]):
    # On A100 (sm_80), TMA is not available
    tx = ct.load(x, index=(0,), shape=(1024,), allow_tma=True)
    # Raises TileUnsupportedFeatureError on sm_80

# ERROR: Double precision not supported on some GPUs
@ct.kernel
def double_kernel(x: ct.Buffer[float64]):
    # Some GPUs don't support float64 efficiently
    pass
```

**Solution:**
Check architecture capabilities and use features conditionally:
```python
@ct.kernel
def adaptive_kernel(x: ct.Buffer[float], ARCH: ct.Constant[str]):
    if ARCH == "sm_90":
        # Use TMA on Hopper
        tx = ct.load(x, index=(0,), shape=(1024,), allow_tma=True)
    else:
        # Regular load on other architectures
        tx = ct.load(x, index=(0,), shape=(1024,), allow_tma=False)
```

### 21.1.5 TileCompilerExecutionError

**Raised when:** The tileiras compiler encounters an error during code generation.

**Common Causes:**
- Compiler internal errors
- Invalid intermediate representation
- Resource exhaustion during compilation
- Bugs in cuTile compiler

**Example:**
```python
@ct.kernel
def complex_kernel(x: ct.Buffer[float]):
    # Complex operations that may trigger compiler bug
    # Very large kernel or unusual code patterns
    for i in range(1000):
        for j in range(1000):
            for k in range(1000):
                # Deeply nested loops may cause issues
                pass
```

**Solution:**
1. Simplify the kernel
2. Break into smaller kernels
3. Report bug with minimal reproducible example
4. Try different optimization levels

```python
# Break complex kernel into smaller pieces
@ct.kernel
def simpler_kernel_part1(x: ct.Buffer[float], temp: ct.Buffer[float]):
    # First part of computation
    pass

@ct.kernel
def simpler_kernel_part2(temp: ct.Buffer[float], y: ct.Buffer[float]):
    # Second part of computation
    pass
```

### 21.1.6 TileCompilerTimeoutError

**Raised when:** Kernel compilation exceeds the specified timeout limit.

**Common Causes:**
- Extremely complex kernels
- Large search spaces during autotuning
- Compiler getting stuck in optimization passes
- Resource constraints

**Example:**
```python
import time

# ERROR: Compilation takes too long
@ct.kernel(opt_level=3)
def huge_kernel(x: ct.Buffer[float]):
    # Very complex computation
    for i in range(10000):
        for j in range(10000):
            # Millions of operations
            pass

# With timeout
try:
    with ct.compiler_timeout(10):
        compiled = huge_kernel.compile(args)
except TileCompilerTimeoutError:
    print("Compilation timed out")
```

**Solution:**
1. Increase timeout if needed:
```python
with ct.compiler_timeout(120):  # 2 minutes
    compiled = huge_kernel.compile(args)
```

2. Reduce optimization level:
```python
@ct.kernel(opt_level=2)  # Lower optimization level
def manageable_kernel(x: ct.Buffer[float]):
    # Same computation, faster compilation
    pass
```

3. Simplify kernel structure

### 21.1.7 TileRecursionError

**Raised when:** Recursion limit is reached during function inlining.

**Common Causes:**
- Excessive recursion in kernel functions
- Circular function dependencies
- Deep call chains

**Example:**
```python
# ERROR: Recursive function (not supported)
@ct.kernel
def factorial_kernel(n: int, result: ct.Buffer[int]):
    # Recursive functions not supported
    if n <= 1:
        return 1
    else:
        return n * factorial_kernel(n - 1, result)
```

**Solution:**
Convert recursion to iteration:
```python
@ct.kernel
def factorial_iterative(n: int, result: ct.Buffer[int]):
    # Iterative version
    fact = 1
    for i in range(2, n + 1):
        fact = fact * i
    result[0] = fact
```

## 21.2 Compiler Timeout Management

### 21.2.1 Setting Compiler Timeout

Use the `compiler_timeout()` context manager to limit compilation time:

```python
with ct.compiler_timeout(timeout_sec):
    # Kernel compilation must complete within timeout_sec
    compiled_kernel = kernel.compile(args)
```

**Parameters:**
- `timeout_sec` (int): Maximum compilation time in seconds

**Behavior:**
- Raises `TileCompilerTimeoutError` if exceeded
- Applies to all compilations within the context
- **Not thread-safe:** modifies global process state

### 21.2.2 Timeout Examples

**Basic Usage:**
```python
@ct.kernel
def my_kernel(x: ct.Buffer[float], y: ct.Buffer[float]):
    tid = ct.tid()
    y[tid] = x[tid] * 2.0

# Set timeout for compilation
try:
    with ct.compiler_timeout(30):
        compiled = my_kernel.compile((x_buf, y_buf))
except TileCompilerTimeoutError:
    print("Compilation timed out after 30 seconds")
```

**Autotuning with Timeout:**
```python
import cuda.tile as ct

search_space = [  # Large search space
    {'tile_size': ts, 'num_ctas': nc}
    for ts in [16, 32, 64, 128, 256]
    for nc in [1, 2, 4, 8, 16]
]

def tune_with_timeout(search_space, stream, grid_fn, kernel, args_fn):
    results = []
    for config in search_space:
        try:
            # Timeout per configuration
            with ct.compiler_timeout(20):
                result = ct.tune.exhaustive_search(
                    search_space=[config],
                    stream=stream,
                    grid_fn=grid_fn,
                    kernel=kernel,
                    args_fn=args_fn
                )
                results.append(result.best)
        except TileCompilerTimeoutError:
            print(f"Config {config} timed out, skipping")
            continue
    return results
```

**Global Timeout via Environment Variable:**
```bash
# Set default timeout for all compilations
export CUDA_TILE_COMPILER_TIMEOUT_SEC=60
python3 my_kernel.py
```

## 21.3 Environment Variables for Debugging

### 21.3.1 CUDA_TILE_ENABLE_CRASH_DUMP

**Purpose:** Generate diagnostic archive for bug reports.

**Usage:**
```bash
export CUDA_TILE_ENABLE_CRASH_DUMP=1
python3 my_kernel.py
```

**Behavior:**
- Creates archive with diagnostic information on crash
- Includes kernel source, compiler state, and error details
- Useful for reporting bugs to cuTile developers

**Output:**
- Archive file: `cutile_crash_dump_<timestamp>.tar.gz`
- Contains:
  - Kernel source code
  - Compiler error messages
  - System information
  - TileIR (if available)

### 21.3.2 CUDA_TILE_COMPILER_TIMEOUT_SEC

**Purpose:** Set default compiler timeout.

**Usage:**
```bash
export CUDA_TILE_COMPILER_TIMEOUT_SEC=120
python3 my_kernel.py
```

**Default:** 60 seconds

**Override:** Use `compiler_timeout()` context manager for specific kernels

### 21.3.3 CUDA_TILE_LOGS

**Purpose:** Enable logging of compiler intermediate representations.

**Usage:**
```bash
export CUDA_TILE_LOGS=CUTILEIR
python3 my_kernel.py
```

**Values:**
- `CUTILEIR`: Print TileIR during compilation
- `CUTILEIR,CUDA`: Print both TileIR and CUDA IR
- `ALL`: Print all available debug information

**Output Example:**
```
TileIR for kernel 'my_kernel':
function my_kernel(x: Buffer<float32>, y: Buffer<float32>) {
  %0 = program_id(0)
  %1 = tid()
  %2 = load(x, %0 * 128 + %1 * 4, 128)
  %3 = mul(%2, 2.0)
  store(y, %0 * 128 + %1 * 4, %3)
}
```

**Use Cases:**
1. Debugging `TileTypeError`: See actual types in IR
2. Understanding compiler optimizations
3. Verifying code generation
4. Identifying performance bottlenecks

### 21.3.4 CUDA_TILE_TEMP_DIR

**Purpose:** Specify directory for temporary compilation files.

**Usage:**
```bash
export CUDA_TILE_TEMP_DIR=/tmp/cutile_temp
python3 my_kernel.py
```

**Default:** System temporary directory (`/tmp` on Linux)

**Contains:**
- Intermediate compilation artifacts
- Temporary PTX files
- Debug information

### 21.3.5 CUDA_TILE_CACHE_DIR

**Purpose:** Specify disk cache for bytecode-to-cubin compilation.

**Usage:**
```bash
export CUDA_TILE_CACHE_DIR=~/.cache/cutile-python
python3 my_kernel.py
```

**Default:** `~/.cache/cutile-python`

**Disable Cache:**
```bash
export CUDA_TILE_CACHE_DIR=0
# or
export CUDA_TILE_CACHE_DIR=off
# or
export CUDA_TILE_CACHE_DIR=none
# or
export CUDA_TILE_CACHE_DIR=""
```

**Benefits:**
- Faster subsequent kernel launches
- Reduced compilation overhead
- Persistence across Python sessions

**Cache Size:**
```bash
# Limit cache size (default: 2GB)
export CUDA_TILE_CACHE_SIZE=4294967296  # 4GB
```

## 21.4 Debugging Strategies

### 21.4.1 Device-Side Debugging

**Using ct.print() and ct.printf():**

As of cuTile 1.0, print statements work at all optimization levels.

**Basic Print:**
```python
@ct.kernel
def debug_print_kernel(x: ct.Buffer[float], y: ct.Buffer[float]):
    tid = ct.tid()
    
    # Print values for debugging
    ct.print("Thread ID:", tid)
    ct.print("x[tid] =", x[tid])
    
    y[tid] = x[tid] * 2.0
```

**Formatted Printf:**
```python
@ct.kernel
def debug_printf_kernel(x: ct.Buffer[float], y: ct.Buffer[float]):
    bid = ct.program_id(0)
    tid = ct.tid()
    
    # Formatted output
    ct.printf("Block %d, Thread %d: x=%f, y=%f\n",
              bid, tid, x[tid], y[tid])
    
    y[tid] = x[tid] * 2.0
```

**Conditional Debugging:**
```python
@ct.kernel
def conditional_debug_kernel(x: ct.Buffer[float], DEBUG: ct.Constant[bool]):
    tid = ct.tid()
    
    # Only print if DEBUG is True
    if DEBUG:
        ct.printf("Thread %d: x=%f\n", tid, x[tid])
    
    y[tid] = x[tid] * 2.0
```

**Debugging Specific Threads:**
```python
@ct.kernel
def selective_debug_kernel(x: ct.Buffer[float]):
    bid = ct.program_id(0)
    tid = ct.tid()
    
    # Only debug first block and thread
    if bid == 0 and tid == 0:
        ct.printf("First thread: x[0]=%f\n", x[0])
    
    # Computation continues for all threads
    result = x[tid] * 2.0
```

**Best Practices:**
1. Use selective printing to avoid overwhelming output
2. Print before and after critical operations
3. Include thread/block IDs for context
4. Disable in production (use conditional compilation)

### 21.4.2 Reading TileIR Output

Enable TileIR logging to see compiler's intermediate representation:

```bash
export CUDA_TILE_LOGS=CUTILEIR
python3 my_kernel.py
```

**Understanding TileIR:**

TileIR (Tile Intermediate Representation) shows:
- Function signatures and types
- All operations and their order
- Memory access patterns
- Optimization transformations

**Example Analysis:**

```python
@ct.kernel
def example_kernel(x: ct.Buffer[float], y: ct.Buffer[float]):
    tid = ct.tid()
    tx = ct.load(x, index=(tid,), shape=(16,))
    ty = tx * 2.0
    ct.store(y, index=(tid,), tile=ty)
```

**Corresponding TileIR:**
```
function example_kernel(x: Buffer<float32>, y: Buffer<float32>) {
  %tid = tid()
  %tx = load(x, %tid * 64, 16)
  %ty = mul(%tx, 2.0)
  store(y, %tid * 64, %ty)
}
```

**Debugging with TileIR:**
1. Check type annotations match your intent
2. Verify memory access patterns are correct
3. Ensure operations are in expected order
4. Look for unexpected optimization transformations

### 21.4.3 Common Error Messages and Solutions

**Error: "TileTypeError: Cannot convert type X to type Y"**

**Cause:** Type mismatch in operations or assignments

**Solution:**
```python
# WRONG
@ct.kernel
def mismatch(x: ct.Buffer[float], y: ct.Buffer[int]):
    y[0] = x[0]  # Cannot convert float to int

# CORRECT
@ct.kernel
def correct(x: ct.Buffer[float], y: ct.Buffer[float]):
    y[0] = x[0]  # Types match
```

**Error: "TileValueError: Invalid grid dimension"**

**Cause:** Grid dimension is zero or negative

**Solution:**
```python
# WRONG
grid = (0, 1, 1)  # Invalid: zero dimension

# CORRECT
grid = (1, 1, 1)  # Valid: all dimensions >= 1
```

**Error: "TileSyntaxError: Unsupported syntax: list comprehension"**

**Cause:** Using unsupported Python features

**Solution:**
```python
# WRONG
@ct.kernel
def bad_syntax(x: ct.Buffer[float]):
    result = [x[i] * 2 for i in range(128)]  # List comprehension

# CORRECT
@ct.kernel
def good_syntax(x: ct.Buffer[float], result: ct.Buffer[float]):
    for i in range(128):
        result[i] = x[i] * 2
```

**Error: "TileCompilerTimeoutError: Compilation exceeded timeout"**

**Cause:** Kernel too complex or optimization too aggressive

**Solution:**
```python
# Reduce optimization level
@ct.kernel(opt_level=2)  # Was 3
def simpler_kernel(x: ct.Buffer[float]):
    # Same logic, faster compilation
    pass

# Or increase timeout
with ct.compiler_timeout(120):
    compiled = complex_kernel.compile(args)
```

**Error: "TileUnsupportedFeatureError: TMA not available on this architecture"**

**Cause:** Using Tensor Memory Accelerator on unsupported GPU

**Solution:**
```python
# WRONG on pre-Hopper
@ct.kernel
def tma_kernel(x: ct.Buffer[float]):
    tx = ct.load(x, index=(0,), shape=(1024,), allow_tma=True)

# CORRECT
@ct.kernel
def adaptive_kernel(x: ct.Buffer[float], ARCH: ct.Constant[str]):
    if ARCH == "sm_90":
        tx = ct.load(x, index=(0,), shape=(1024,), allow_tma=True)
    else:
        tx = ct.load(x, index=(0,), shape=(1024,), allow_tma=False)
```

## 21.5 Profiling with Nsight Compute

### 21.5.1 Basic Profiling

NVIDIA Nsight Compute provides detailed GPU kernel profiling:

```bash
# Basic profiling
ncu -o profile --set detailed python3 my_kernel.py

# View results
ncu-ui profile.ncu-rep
```

**Requirements:**
- NVIDIA Driver >= r580.126.09 (Linux)
- NVIDIA Driver >= r582.16 (Windows)
- Nsight Compute >= 2024.2

### 21.5.2 Profiling Options

**Profile Specific Kernel:**
```bash
ncu -o profile -k my_kernel python3 my_script.py
```

**Detailed Metrics:**
```bash
ncu -o profile --set full python3 my_script.py
```

**Section-Based Profiling:**
```bash
# Memory workload
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
    -o profile python3 my_script.py

# Compute workload
ncu --section SpeedOfLight --section ComputeWorkloadAnalysis \
    -o profile python3 my_script.py
```

**Override Kernel Duration:**
```bash
# Run kernel multiple times for accurate measurements
ncu --kernel-name-base function --kernel-name my_kernel \
    --launch-count 100 -o profile python3 my_script.py
```

### 21.5.3 Key Metrics

**Memory Metrics:**
- **DRAM Throughput:** Actual memory bandwidth utilization
- **L2 Cache Hit Rate:** Percentage of memory accesses served by L2
- **Memory Workload:** Total bytes transferred

**Compute Metrics:**
- **Achieved Occupancy:** Actual vs theoretical SM utilization
- **Warp Execution Efficiency:** Percentage of active warps
- **FLOPS:** Floating point operations per second

**Analysis Metrics:**
- **Speed of Light:** Percentage of theoretical peak performance
- **Instruction Mix:** Ratio of different instruction types
- **Branch Divergence:** Impact of conditional branches

### 21.5.4 Interpreting Results

**Low DRAM Throughput:**
- **Cause:** Poor memory access patterns, low arithmetic intensity
- **Solution:** Improve coalescing, increase tile size, use shared memory

**Low Occupancy:**
- **Cause:** High register usage, large shared memory, small block size
- **Solution:** Reduce register pressure, decrease shared memory, increase block size

**High Branch Divergence:**
- **Cause:** Conditional branches with different execution paths per warp
- **Solution:** Restructure conditionals, use warp-level primitives

## 21.6 Comprehensive Debugging Workflow

### 21.6.1 Step-by-Step Debugging Process

**1. Enable Verbose Logging:**
```bash
export CUDA_TILE_LOGS=CUTILEIR
export CUDA_TILE_ENABLE_CRASH_DUMP=1
```

**2. Add Debug Output:**
```python
@ct.kernel
def debug_kernel(x: ct.Buffer[float], y: ct.Buffer[float]):
    bid = ct.program_id(0)
    tid = ct.tid()
    
    # Debug output
    if bid == 0 and tid == 0:
        ct.printf("Kernel launch: x[0]=%f\n", x[0])
    
    # Computation
    result = x[tid] * 2.0
    
    # Intermediate check
    if bid == 0 and tid == 0:
        ct.printf("After computation: result=%f\n", result)
    
    y[tid] = result
```

**3. Validate with CPU Reference:**
```python
import numpy as np

# Create test data
x_np = np.random.randn(1024).astype(np.float32)
y_np = np.zeros_like(x_np)

# GPU computation
x_buf = ct.Buffer(x_np)
y_buf = ct.Buffer(y_np)
debug_kernel[stream, grid](x_buf, y_buf)
stream.synchronize()

# CPU reference
y_expected = x_np * 2.0

# Compare
y_result = y_buf.copy_to_host()
assert np.allclose(y_result, y_expected, rtol=1e-5), "Mismatch!"
print("Results match!")
```

**4. Profile with Nsight:**
```bash
ncu -o profile python3 my_script.py
ncu-ui profile.ncu-rep
```

**5. Analyze TileIR:**
```bash
# Check for unexpected transformations
export CUDA_TILE_LOGS=CUTILEIR
python3 my_script.py > tileir_output.txt
# Review tileir_output.txt
```

### 21.6.2 Common Debugging Scenarios

**Scenario 1: Incorrect Results**

**Symptoms:** Kernel runs without errors but produces wrong output

**Debugging Steps:**
1. Add print statements to verify inputs
2. Print intermediate values
3. Compare with CPU reference implementation
4. Check for race conditions (multiple threads writing same location)
5. Verify memory access patterns

**Example:**
```python
@ct.kernel
def buggy_reduction(x: ct.Buffer[float], result: ct.Buffer[float]):
    tid = ct.tid()
    
    # BUG: Race condition - all threads write to same location
    result[0] = result[0] + x[tid]

# CORRECT: Use atomic operation
@ct.kernel
def correct_reduction(x: ct.Buffer[float], result: ct.Buffer[float]):
    tid = ct.tid()
    
    # Use atomic add (if supported) or parallel reduction algorithm
    ct.atomic_add(result, index=(0,), value=x[tid])
```

**Scenario 2: Poor Performance**

**Symptoms:** Kernel runs but slower than expected

**Debugging Steps:**
1. Profile with Nsight Compute
2. Check memory access patterns (coalescing)
3. Verify occupancy is reasonable
4. Look for branch divergence
5. Compare against optimized reference

**Example:**
```python
# BAD: Strided memory access
@ct.kernel
def bad_access(x: ct.Buffer[float], y: ct.Buffer[float]):
    tid = ct.tid()
    # Threads 0-31 access addresses 0, 32, 64, ... (not coalesced)
    y[tid] = x[tid * 32]

# GOOD: Contiguous access
@ct.kernel
def good_access(x: ct.Buffer[float], y: ct.Buffer[float]):
    bid = ct.program_id(0)
    tid = ct.tid()
    # Threads 0-31 access addresses bid*32+0 to bid*32+31 (coalesced)
    y[bid * 32 + tid] = x[bid * 32 + tid]
```

**Scenario 3: Compilation Errors**

**Symptoms:** Kernel fails to compile with obscure error

**Debugging Steps:**
1. Enable TileIR logging
2. Simplify kernel to minimal example
3. Check type annotations
4. Verify all values are constants where required
5. Look for unsupported Python features

**Example:**
```python
# Complex kernel with error
@ct.kernel
def complex_kernel(x: ct.Buffer[float], y: ct.Buffer[float]):
    # Many operations, hard to debug
    for i in range(100):
        for j in range(100):
            # ... complex logic ...
            pass

# Simplified to isolate error
@ct.kernel
def simple_kernel(x: ct.Buffer[float], y: ct.Buffer[float]):
    # Minimal version that still shows error
    tid = ct.tid()
    y[tid] = x[tid]
```

## 21.7 Best Practices for Debugging

### 21.7.1 Development Workflow

**1. Start Simple:**
- Begin with minimal kernel
- Verify correctness before adding complexity
- Test with small inputs

**2. Incremental Development:**
- Add features one at a time
- Test after each change
- Use version control to track changes

**3. Use Assertions:**
```python
@ct.kernel
def safe_kernel(x: ct.Buffer[float], SIZE: ct.Constant[int]):
    # Validate at compile time
    ct.static_assert(SIZE > 0, "SIZE must be positive")
    ct.static_assert(SIZE <= 1024, "SIZE too large")
    
    # Validate at runtime (for debugging)
    if SIZE <= 0 or SIZE > 1024:
        ct.printf("ERROR: Invalid SIZE=%d\n", SIZE)
        return
    
    # Computation
    # ...
```

**4. Profile Early and Often:**
- Don't wait until optimization phase
- Profile after major features
- Compare against baselines

### 21.7.2 Testing Strategies

**Unit Testing:**
```python
def test_vector_add():
    # Test data
    size = 1024
    x = np.random.randn(size).astype(np.float32)
    y = np.random.randn(size).astype(np.float32)
    expected = x + y
    
    # GPU computation
    x_buf = ct.Buffer(x)
    y_buf = ct.Buffer(y)
    result_buf = ct.Buffer(np.zeros_like(x))
    
    stream = cuda.cuda.Stream()
    grid = (size + 255) // 256
    vector_add[stream, grid](x_buf, y_buf, result_buf)
    stream.synchronize()
    
    # Verify
    result = result_buf.copy_to_host()
    assert np.allclose(result, expected, rtol=1e-5)
    print("PASSED: vector_add")
```

**Regression Testing:**
```python
def test_regression():
    # Known good input/output pairs
    test_cases = [
        (np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])),
        (np.array([0.0, -1.0, 5.5]), np.array([0.0, -2.0, 11.0])),
    ]
    
    for input_data, expected_output in test_cases:
        result = run_kernel(input_data)
        assert np.allclose(result, expected_output)
```

**Property-Based Testing:**
```python
def test_properties():
    # Test mathematical properties
    for _ in range(100):
        # Random input
        x = np.random.randn(256).astype(np.float32)
        
        # Test property: f(f(x)) should equal x for inverse function
        y = run_kernel(x)
        z = inverse_kernel(y)
        
        assert np.allclose(x, z, rtol=1e-5)
```

### 21.7.3 Performance Debugging Checklist

- [ ] Are memory accesses coalesced?
- [ ] Is shared memory bank-free?
- [ ] Is occupancy reasonable (> 50%)?
- [ ] Are there race conditions?
- [ ] Is branch divergence minimal?
- [ ] Are appropriate latency hints set?
- [ ] Is tile size optimal?
- [ ] Is TMA enabled where applicable?
- [ ] Are compiler optimizations enabled?
- [ ] Is GPU clock fixed for consistent measurement?

## 21.8 Summary

This chapter covered comprehensive debugging strategies for cuTile kernels:

1. **Exception Types:** Understanding different error categories helps diagnose issues quickly
2. **Compiler Timeout:** Use `compiler_timeout()` to prevent runaway compilation
3. **Environment Variables:** Enable logging, crash dumps, and configure caching
4. **Debugging Tools:** Use print statements, TileIR, and Nsight Compute
5. **Common Errors:** Solutions for frequent issues
6. **Best Practices:** Systematic debugging workflow and testing strategies

Effective debugging combines:
- Understanding error messages and their causes
- Using appropriate tools for the problem
- Following systematic debugging workflows
- Implementing comprehensive testing
- Leveraging profiling and performance analysis

Key takeaways:
- Enable TileIR logging to understand compiler transformations
- Use print statements strategically to debug device code
- Profile with Nsight Compute to identify performance bottlenecks
- Follow incremental development with testing at each step
- Use static assertions to catch errors at compile time
- Compare against CPU reference implementations for correctness
