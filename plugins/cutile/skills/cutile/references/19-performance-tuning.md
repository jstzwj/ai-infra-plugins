# 19. Performance Tuning

This chapter covers comprehensive performance tuning techniques for cuTile kernels, including architecture-specific configuration, kernel parameters, memory access optimization, and automated tuning strategies.

## 19.1 Architecture-Specific Configuration

### 19.1.1 The ByTarget Class

The `cuda.tile.ByTarget` class enables architecture-specific configuration of kernel parameters. This is essential for optimizing kernels across different GPU architectures with varying capabilities.

```python
class cuda.tile.ByTarget(*, default=UNSPECIFIED, **value_by_target)
```

**Parameters:**
- `default` (Any): Fallback value for architectures not explicitly listed in `value_by_target`
- `**value_by_target`: Keyword arguments mapping architecture identifiers to values

**Architecture Identifiers:**
Architecture identifiers follow the format "sm_XY" where:
- X indicates the GPU architecture generation (e.g., 8 for Ampere, 9 for Hopper)
- Y indicates the specific implementation variant

Common architecture identifiers:
- `sm_80`: Ampere A100
- `sm_86`: Ampere RTX 3090, 3080
- `sm_89`: Ada Lovelace RTX 4090, 4080
- `sm_90`: Hopper H100

**Usage Pattern:**

The `ByTarget` class can be used with any kernel configuration parameter, including:
- `num_ctas`: Number of CTAs in the Collective Group Array (CGA)
- `occupancy`: Expected active CTAs per streaming multiprocessor
- `opt_level`: Optimization level
- Custom kernel parameters

### 19.1.2 Basic ByTarget Examples

**Uniform Configuration (No ByTarget):**
```python
import cuda.tile as ct

@ct.kernel(num_ctas=8)
def my_kernel(x: ct.Buffer[float], y: ct.Buffer[float]):
    # Same num_ctas=8 for all architectures
    pass
```

**Architecture-Specific Configuration:**
```python
@ct.kernel(num_ctas=ByTarget(sm_80=8, sm_90=4, default=2))
def tuned_kernel(x: ct.Buffer[float], y: ct.Buffer[float]):
    # 8 CTAs on Ampere, 4 CTAs on Hopper, 2 CTAs on others
    pass
```

**Multiple Parameters with ByTarget:**
```python
@ct.kernel(
    num_ctas=ByTarget(sm_80=8, sm_90=4, default=2),
    occupancy=ByTarget(sm_90=16, default=8),
    opt_level=ByTarget(sm_80=3, sm_90=3, default=2)
)
def fully_configured_kernel(x: ct.Buffer[float], y: ct.Buffer[float]):
    pass
```

### 19.1.3 Architecture-Specific Best Practices

**When to Use ByTarget:**

1. **Different Memory Bandwidth:** Newer architectures may benefit from different CTA counts
2. **Shared Memory Capacity:** Varies between architectures, affecting tile sizes
3. **Tensor Core Availability:** Different generations have varying tensor core capabilities
4. **Register File Size:** Affects occupancy targets

**Example: Memory-Bound vs Compute-Bound Kernels:**
```python
@ct.kernel(
    num_ctas=ByTarget(
        sm_80=8,   # A100: High bandwidth, can use more CTAs
        sm_86=4,   # RTX 3080: Lower bandwidth, fewer CTAs
        sm_90=16,  # H100: Even higher bandwidth
        default=2
    )
)
def memory_bound_kernel(x: ct.Buffer[float], y: ct.Buffer[float]):
    # Memory-bound kernels benefit from architecture tuning
    bid = ct.program_id(0)
    tx = ct.load(x, index=(bid,), shape=(1024,))
    ct.store(y, index=(bid,), tile=tx)
```

## 19.2 Kernel Configuration Parameters

### 19.2.1 num_ctas - Collective Group Array Size

The `num_ctas` parameter controls the number of Cooperative Thread Arrays (CTAs) in the Collective Group Array (CGA). This enables kernel-wide cooperation across multiple thread blocks.

**Specification:**
- **Type:** `int` or `ByTarget`
- **Range:** Power of 2, between 1 and 16 (inclusive)
- **Valid Values:** 1, 2, 4, 8, 16
- **Default:** 1

**Purpose:**
Multiple CTAs in a CGA can:
1. Share workloads across thread blocks
2. Aggregate memory bandwidth for large transfers
3. Perform collective operations across larger datasets
4. Improve utilization for kernels with low per-CTA work

**When to Use Multiple CTAs:**

| Use Case | Recommended num_ctas | Rationale |
|----------|---------------------|-----------|
| Small kernels | 1 | Avoid overhead |
| Large memory transfers | 4-8 | Aggregate bandwidth |
| Reduction operations | 8-16 | Maximize parallel reduction |
| Compute-bound | 1-2 | Prevent SM oversubscription |

**Example: Large Matrix Operations:**
```python
@ct.kernel(num_ctas=8)
def large_matmul(
    A: ct.Buffer[float],
    B: ct.Buffer[float],
    C: ct.Buffer[float],
    M: int,
    N: int,
    K: int
):
    """
    Matrix multiplication using 8 CTAs for large matrices.
    Each CTA handles a portion of the output matrix.
    """
    # Determine which CTA this is within the CGA
    cta_id = ct.cta_id()
    
    # Calculate global thread ID across all CTAs
    tid = cta_id * 128 + ct.tid()  # Assume 128 threads per CTA
    
    # Each CTA computes its portion of the output
    # ... computation logic ...
```

### 19.2.2 occupancy - Expected Active CTAs per SM

The `occupancy` parameter provides a hint to the compiler about the expected number of active CTAs per streaming multiprocessor.

**Specification:**
- **Type:** `int` or `ByTarget`
- **Range:** 1 to 32 (inclusive)
- **Default:** Architecture-dependent

**Purpose:**
The occupancy hint helps the compiler:
1. Optimize register allocation
2. Plan shared memory usage
3. Schedule instructions for better latency hiding
4. Choose appropriate unrolling strategies

**Occupancy vs Performance:**

Higher occupancy doesn't always mean better performance. Consider:

| Scenario | Optimal Occupancy | Reasoning |
|----------|-------------------|-----------|
| Memory-bound kernels | Moderate (4-8) | More CTAs don't help if waiting on memory |
| Compute-bound kernels | High (16-32) | Keep arithmetic units busy |
| High register usage | Low (2-4) | Avoid register spilling |
| Shared memory intensive | Low (2-4) | Don't exceed shared memory capacity |

**Example: Compute-Bound Kernel:**
```python
@ct.kernel(occupancy=16)
def compute_bound_kernel(x: ct.Buffer[float], y: ct.Buffer[float]):
    """
    High occupancy for compute-intensive operations.
    """
    tid = ct.tid()
    
    # Lots of computation per thread
    # Compiler knows to expect 16 active CTAs
    # and will optimize register usage accordingly
    result = 0.0
    for i in range(100):
        result += x[tid + i] * x[tid + i]
    
    y[tid] = result
```

**Example: Memory-Bound Kernel:**
```python
@ct.kernel(occupancy=4)
def memory_bound_kernel(x: ct.Buffer[float], y: ct.Buffer[float]):
    """
    Lower occupancy for memory-bound operations.
    """
    tid = ct.tid()
    
    # Simple memory copy, low computation
    # Don't need many active CTAs
    y[tid] = x[tid]
```

### 19.2.3 opt_level - Optimization Level

The `opt_level` parameter controls the compiler's optimization aggressiveness.

**Specification:**
- **Type:** `int` or `ByTarget`
- **Range:** 0 to 3 (inclusive)
- **Default:** 3

**Optimization Levels:**

| Level | Description | Compile Time | Performance | Debugging |
|-------|-------------|--------------|-------------|-----------|
| 0 | No optimization | Fast | Low | Excellent |
| 1 | Basic optimizations | Moderate | Moderate | Good |
| 2 | Standard optimizations | Slow | High | Fair |
| 3 | Aggressive optimizations | Slowest | Highest | Poor |

**Level-Specific Optimizations:**

**Level 0:**
- No loop unrolling
- No instruction reordering
- Minimal register allocation optimization
- Preserves original code structure
- Useful for debugging

**Level 1:**
- Basic loop unrolling
- Simple constant propagation
- Basic dead code elimination
- Minimal instruction scheduling

**Level 2:**
- Aggressive loop unrolling
- Advanced constant propagation
- Inter-procedural optimizations
- Instruction-level parallelism
- Memory access coalescing

**Level 3:**
- All level 2 optimizations
- Aggressive instruction reordering
- Advanced register allocation
- Kernel fusion opportunities
- Architecture-specific optimizations
- Tensor core utilization (when applicable)

**Example: Debug Mode:**
```python
@ct.kernel(opt_level=0)
def debug_kernel(x: ct.Buffer[float], y: ct.Buffer[float]):
    """
    Use opt_level=0 when debugging to preserve code structure.
    Note: As of cuTile 1.0, print/printf work at all opt levels.
    """
    tid = ct.tid()
    ct.printf("Thread %d: x=%f\n", tid, x[tid])
    y[tid] = x[tid] * 2.0
```

**Example: Production Mode:**
```python
@ct.kernel(opt_level=3)
def production_kernel(x: ct.Buffer[float], y: ct.Buffer[float]):
    """
    Use opt_level=3 for maximum performance in production.
    """
    tid = ct.tid()
    
    # Aggressive optimization will:
    # - Unroll loops
    # - Reorder instructions
    # - Fuse operations
    # - Optimize memory access
    
    y[tid] = x[tid] * 2.0 + 1.0
```

## 19.3 Load/Store Performance Hints

### 19.3.1 latency Hint

The `latency` parameter provides performance hints to the compiler about the expected DRAM latency for memory operations.

**Specification:**
- **Type:** `int` (1-10)
- **Range:** 1 (low DRAM traffic) to 10 (high DRAM traffic)
- **Default:** Architecture-dependent

**Latency Guidelines:**

| Latency Value | Use Case | Examples |
|--------------|----------|----------|
| 1-3 | Cached data, L2 hits | Repeated access to small datasets |
| 4-6 | Moderate DRAM traffic | Typical global memory access |
| 7-10 | High DRAM traffic | Large streaming loads, strided access |

**How It Works:**
The compiler uses the latency hint to:
1. Schedule instructions to hide memory latency
2. Insert prefetch instructions
3. Optimize warp scheduling
4. Choose between load instructions (LDG, LDS, etc.)

**Example: Low Latency (Cached Data):**
```python
@ct.kernel
def low_latency_kernel(x: ct.Buffer[float], y: ct.Buffer[float]):
    """
    Data is expected to be in cache.
    """
    tid = ct.tid()
    
    # Low latency hint - assume data is cached
    tx = ct.load(x, index=(tid,), shape=(16,), latency=2)
    
    # Multiple accesses to same data
    result = tx[0] + tx[1] + tx[2]
    
    ct.store(y, index=(tid,), tile=result, latency=2)
```

**Example: High Latency (Streaming Access):**
```python
@ct.kernel
def high_latency_kernel(x: ct.Buffer[float], y: ct.Buffer[float]):
    """
    Large streaming load from DRAM.
    """
    bid = ct.program_id(0)
    
    # High latency hint - expect DRAM access
    # Compiler will insert prefetch and better schedule instructions
    tx = ct.load(x, index=(bid * 1024,), shape=(1024,), latency=8)
    
    # Process data (compiler hides latency during computation)
    result = tx * 2.0
    
    ct.store(y, index=(bid * 1024,), tile=result, latency=8)
```

### 19.3.2 allow_tma Hint

The `allow_tma` parameter controls whether the compiler may use Tensor Memory Accelerator (TMA) for memory operations.

**Specification:**
- **Type:** `bool`
- **Range:** `True` or `False`
- **Default:** `True`
- **Availability:** Hopper (H100) and later architectures

**Tensor Memory Accelerator (TMA):**
TMA is a hardware feature introduced in Hopper that accelerates memory transfers by:
1. Offloading memory copy operations from the GPU
2. Performing asynchronous transfers
3. Reducing register usage for memory operations
4. Improving bandwidth utilization for large transfers

**When to Enable TMA (allow_tma=True):**
- Large memory transfers (> 1KB)
- Regular access patterns
- Hopper (sm_90) and later architectures
- Memory-bound kernels

**When to Disable TMA (allow_tma=False):**
- Small transfers (< 256 bytes)
- Irregular access patterns
- Pre-Hopper architectures
- Compute-bound kernels where TMA overhead isn't justified

**Example: Enable TMA for Large Transfers:**
```python
@ct.kernel
def large_transfer_kernel(
    x: ct.Buffer[float],
    y: ct.Buffer[float]
):
    """
    Large matrix copy - perfect for TMA on Hopper.
    """
    bid = ct.program_id(0)
    
    # TMA will handle this large transfer efficiently
    tx = ct.load(x, index=(bid * 4096,), shape=(4096,), allow_tma=True)
    ct.store(y, index=(bid * 4096,), tile=tx, allow_tma=True)
```

**Example: Disable TMA for Small Transfers:**
```python
@ct.kernel
def small_transfer_kernel(
    x: ct.Buffer[float],
    y: ct.Buffer[float]
):
    """
    Small element-wise operation - TMA overhead not worth it.
    """
    tid = ct.tid()
    
    # Direct load/store is better for small transfers
    val = ct.load(x, index=(tid,), shape=(1,), allow_tma=False)
    ct.store(y, index=(tid,), tile=val, allow_tma=False)
```

**Example: Architecture-Specific TMA:**
```python
@ct.kernel
def adaptive_kernel(
    x: ct.Buffer[float],
    y: ct.Buffer[float]
):
    """
    Use TMA on Hopper, regular loads elsewhere.
    """
    bid = ct.program_id(0)
    
    # On Hopper: use TMA, on others: regular loads
    # Compiler handles this automatically with allow_tma=True
    tx = ct.load(x, index=(bid * 2048,), shape=(2048,), allow_tma=True)
    ct.store(y, index=(bid * 2048,), tile=tx, allow_tma=True)
```

## 19.4 Autotuning

### 19.4.1 Exhaustive Search

The `cuda.tile.tune.exhaustive_search` function performs automated parameter tuning by evaluating all configurations in a search space.

```python
cuda.tile.tune.exhaustive_search(
    search_space,
    stream,
    grid_fn,
    kernel,
    args_fn,
    hints_fn=None,
    *,
    quiet=False
)
```

**Parameters:**
- `search_space` (Sequence[Mapping]): Sequence of configuration dictionaries to test
- `stream` (CUDA Stream): CUDA stream for kernel execution
- `grid_fn` (Callable): Function mapping config → grid dimensions
  ```python
  def grid_fn(config):
      return (config['num_blocks'], 1, 1)
  ```
- `kernel` (Callable): Kernel function to tune
- `args_fn` (Callable): Function mapping config → kernel arguments
  ```python
  def args_fn(config):
      return (input_buf, output_buf, config['tile_size'])
  ```
- `hints_fn` (Callable, optional): Function mapping config → compiler hints
  ```python
  def hints_fn(config):
      return {'opt_level': config['opt_level']}
  ```
- `quiet` (bool, optional): Suppress progress output (default: False)

**Returns:**
- `tune.TuningResult`: Object containing tuning results

### 19.4.2 TuningResult Class

The `tune.TuningResult` class encapsulates the results of an autotuning run.

```python
class tune.TuningResult:
    best: tune.Measurement  # Best configuration found
    all_results: List[tune.Measurement]  # All measurements
```

**Attributes:**
- `best`: The `Measurement` object with the best (lowest) execution time
- `all_results`: List of all `Measurement` objects in the order they were tested

**Methods:**
```python
# Get the best configuration
best_config = result.best.config

# Get the best time in microseconds
best_time = result.best.mean_us

# Get all configurations sorted by time
sorted_results = sorted(result.all_results, key=lambda m: m.mean_us)

# Print summary
print(f"Best config: {result.best.config}")
print(f"Best time: {result.best.mean_us:.2f} us")
```

### 19.4.3 Measurement Class

The `tune.Measurement` class represents a single kernel execution measurement.

```python
class tune.Measurement:
    config: Dict[str, Any]  # Configuration dictionary
    mean_us: float  # Mean execution time in microseconds
```

**Attributes:**
- `config`: The configuration dictionary used for this measurement
- `mean_us`: Average execution time across multiple runs in microseconds

**Usage:**
```python
# Access configuration
tile_size = measurement.config['tile_size']
num_ctas = measurement.config['num_ctas']

# Access timing
time_ms = measurement.mean_us / 1000.0  # Convert to milliseconds

# Calculate speedup vs baseline
speedup = baseline_time / measurement.mean_us
```

## 19.4.4 Complete Autotuning Example

Here's a complete example of autotuning a matrix multiplication kernel:

```python
import cuda.tile as ct
import numpy as np

# Define the kernel
@ct.kernel
def matmul_kernel(
    A: ct.Buffer[float],
    B: ct.Buffer[float],
    C: ct.Buffer[float],
    M: int,
    N: int,
    K: int,
    TILE_SIZE: int
):
    """
    Matrix multiplication: C = A @ B
    A: M x K, B: K x N, C: M x N
    """
    # Block and thread indices
    bid_x = ct.program_id(0)
    bid_y = ct.program_id(1)
    tid_x = ct.tid() % TILE_SIZE
    tid_y = ct.tid() // TILE_SIZE
    
    # Global row and column
    row = bid_y * TILE_SIZE + tid_y
    col = bid_x * TILE_SIZE + tid_x
    
    # Accumulator
    acc = 0.0
    
    # Loop over tiles
    for k in range(0, K, TILE_SIZE):
        # Load tile of A
        a_tile = ct.load(
            A,
            index=(row, k),
            shape=(TILE_SIZE, TILE_SIZE),
            latency=5
        )
        
        # Load tile of B
        b_tile = ct.load(
            B,
            index=(k, col),
            shape=(TILE_SIZE, TILE_SIZE),
            latency=5
        )
        
        # Compute partial result
        for i in range(TILE_SIZE):
            acc += a_tile[tid_y, i] * b_tile[i, tid_x]
    
    # Store result
    ct.store(C, index=(row, col), tile=acc)

# Define search space
search_space = [
    {'tile_size': 16, 'num_ctas': 1, 'opt_level': 2},
    {'tile_size': 16, 'num_ctas': 1, 'opt_level': 3},
    {'tile_size': 16, 'num_ctas': 2, 'opt_level': 2},
    {'tile_size': 16, 'num_ctas': 2, 'opt_level': 3},
    {'tile_size': 32, 'num_ctas': 1, 'opt_level': 2},
    {'tile_size': 32, 'num_ctas': 1, 'opt_level': 3},
    {'tile_size': 32, 'num_ctas': 2, 'opt_level': 2},
    {'tile_size': 32, 'num_ctas': 2, 'opt_level': 3},
    {'tile_size': 64, 'num_ctas': 1, 'opt_level': 2},
    {'tile_size': 64, 'num_ctas': 1, 'opt_level': 3},
]

# Define grid function
def grid_fn(config):
    grid_x = (N + config['tile_size'] - 1) // config['tile_size']
    grid_y = (M + config['tile_size'] - 1) // config['tile_size']
    return (grid_x, grid_y, 1)

# Define arguments function
def args_fn(config):
    return (
        A_buf, B_buf, C_buf,
        M, N, K,
        config['tile_size']
    )

# Define hints function
def hints_fn(config):
    return {
        'num_ctas': config['num_ctas'],
        'opt_level': config['opt_level']
    }

# Create test data
M, N, K = 1024, 1024, 1024
A_np = np.random.randn(M, K).astype(np.float32)
B_np = np.random.randn(K, N).astype(np.float32)
C_np = np.zeros((M, N), dtype=np.float32)

# Create buffers
stream = cuda.cuda.Stream()
A_buf = ct.Buffer(A_np, stream=stream)
B_buf = ct.Buffer(B_np, stream=stream)
C_buf = ct.Buffer(C_np, stream=stream)

# Run autotuning
result = ct.tune.exhaustive_search(
    search_space=search_space,
    stream=stream,
    grid_fn=grid_fn,
    kernel=matmul_kernel,
    args_fn=args_fn,
    hints_fn=hints_fn,
    quiet=False
)

# Print results
print(f"Best configuration: {result.best.config}")
print(f"Best time: {result.best.mean_us:.2f} us")
print(f"\nAll results:")
for i, measurement in enumerate(result.all_results):
    print(f"  {i}: {measurement.config} -> {measurement.mean_us:.2f} us")

# Use the best configuration
best_kernel = matmul_kernel.replace_hints(**hints_fn(result.best.config))
grid = grid_fn(result.best.config)
best_kernel[stream, grid](*args_fn(result.best.config))
stream.synchronize()
```

## 19.5 Advanced Performance Features

### 19.5.1 Dynamic Hint Replacement

The `kernel.replace_hints()` method creates a new kernel with updated compiler hints without recompiling the base kernel.

```python
new_kernel = kernel.replace_hints(**new_hints)
```

**Use Cases:**
1. Switch between configurations after autotuning
2. Adjust parameters based on runtime conditions
3. Create kernel variants without code duplication

**Example:**
```python
# Base kernel
@ct.kernel(opt_level=2)
def flexible_kernel(x, y):
    tid = ct.tid()
    y[tid] = x[tid] * 2.0

# Create optimized version
optimized = flexible_kernel.replace_hints(opt_level=3)

# Create debug version
debug = flexible_kernel.replace_hints(opt_level=0)

# Use appropriate version based on context
if debug_mode:
    kernel_to_use = debug
else:
    kernel_to_use = optimized
```

### 19.5.2 Compiler Timeout Control

The `compiler_timeout()` context manager sets a timeout limit for kernel compilation.

```python
with ct.compiler_timeout(timeout_sec):
    # Kernel compilation must complete within timeout_sec
    compiled_kernel = kernel.compile(args)
```

**Parameters:**
- `timeout_sec` (int): Maximum compilation time in seconds

**Behavior:**
- Raises `TileCompilerTimeoutError` if compilation exceeds timeout
- Not thread-safe (modifies global process state)
- Useful for preventing runaway compilation during autotuning

**Example:**
```python
# Use timeout during autotuning
for config in search_space:
    try:
        with ct.compiler_timeout(30):
            kernel_compiled = kernel.compile(args_fn(config))
    except TileCompilerTimeoutError:
        print(f"Config {config} timed out, skipping")
        continue
```

## 19.6 Performance Best Practices

### 19.6.1 GPU Clock Management

For consistent and reproducible performance measurements, fix GPU clocks:

```bash
# Enable persistence mode
sudo nvidia-smi -pm 1

# Lock graphics clock
sudo nvidia-smi -lgc 1590

# Lock memory clock
sudo nvidia-smi -lmc 7007

# Reset to automatic
sudo nvidia-smi -rgc
sudo nvidia-smi -rmc
```

**Why Fix Clocks:**
1. Eliminate thermal throttling during measurements
2. Ensure consistent performance across runs
3. Get reproducible autotuning results
4. Compare configurations fairly

### 19.6.2 Tile Size Selection Guidelines

**General Principles:**

| Aspect | Recommendation | Reasoning |
|--------|----------------|-----------|
| Multiple of warp size | 32, 64, 96, 128 | Aligns with warp execution |
| Power of 2 | 16, 32, 64, 128 | Easier index calculations |
| Memory alignment | 128 bytes (32 floats) | Cache line alignment |
| Shared memory | < 48KB per SM | Avoid bank conflicts and capacity issues |

**Memory Access Pattern Guidelines:**

1. **Coalesced Access:** Ensure contiguous threads access contiguous memory
   ```python
   # Good: Coalesced
   tid = ct.tid()
   val = x[tid]  # Threads 0-31 access addresses 0-31
   
   # Bad: Strided
   tid = ct.tid()
   val = x[tid * 32]  # Threads 0-31 access addresses 0, 32, 64, ...
   ```

2. **Aligned Access:** Align to 128-byte cache lines
   ```python
   # Ensure starting address is aligned
   aligned_bid = bid * 128  # 128 floats = 512 bytes (4 cache lines)
   ```

3. **Sequential Access:** Prefer sequential over random access
   ```python
   # Good: Sequential
   for i in range(TILE_SIZE):
       val = x[i]
   
   # Bad: Random
   indices = [5, 12, 3, 8, ...]  # Random access pattern
   for i, idx in enumerate(indices):
       val = x[idx]
   ```

### 19.6.3 Occupancy Tuning

**Calculating Occupancy:**

Theoretical occupancy depends on:
1. Threads per CTA (block size)
2. Registers per thread
3. Shared memory per CTA

```python
# Example: Estimate occupancy
@ct.kernel
def occupancy_example(x, y):
    # Assume:
    # - 256 threads per CTA
    # - 40 registers per thread (set by compiler based on code)
    # - 16KB shared memory per CTA
    
    # On A100 (sm_80):
    # - Max 1536 threads per SM
    # - 65536 registers per SM
    # - 164KB shared memory per SM
    
    # Registers limit: 65536 / (256 * 40) = 6 CTAs
    # Threads limit: 1536 / 256 = 6 CTAs
    # Shared memory limit: 164KB / 16KB = 10 CTAs
    
    # Theoretical occupancy: min(6, 6, 10) = 6 CTAs
    # Actual occupancy: min(6, 32) = 6 CTAs
    pass
```

**Tuning Strategies:**

1. **Reduce Register Usage:**
   - Use smaller data types (float16 vs float32)
   - Avoid complex expressions in loops
   - Manually reuse variables

2. **Reduce Shared Memory:**
   - Use smaller tile sizes
   - Use scalar loads instead of tile loads when possible
   - Stream computation to reduce tile storage

3. **Adjust CTA Size:**
   - Smaller CTAs → more CTAs per SM → higher occupancy
   - Larger CTAs → more work per CTA → less overhead
   - Balance based on kernel characteristics

### 19.6.4 Memory Access Optimization

**Using Hints Effectively:**

```python
@ct.kernel
def optimized_memory_kernel(x, y):
    bid = ct.program_id(0)
    
    # Large streaming load → high latency
    large_tile = ct.load(
        x,
        index=(bid * 4096,),
        shape=(4096,),
        latency=9,  # Expect DRAM access
        allow_tma=True  # Use TMA if available
    )
    
    # Computation (hides memory latency)
    result = large_tile * 2.0 + 1.0
    
    # Small, cached load → low latency
    small_val = ct.load(
        y,
        index=(bid,),
        shape=(1,),
        latency=2,  # Expect cache hit
        allow_tma=False  # TMA overhead not worth it
    )
    
    # Store with appropriate hints
    ct.store(
        y,
        index=(bid * 4096,),
        tile=result,
        latency=9,
        allow_tma=True
    )
```

**Memory Access Pattern Checklist:**

- [ ] Accesses are coalesced (contiguous threads → contiguous memory)
- [ ] Starting addresses are aligned to 128 bytes
- [ ] Stride is a power of 2
- [ ] No unaligned accesses
- [ ] Appropriate latency hints set
- [ ] TMA enabled for large transfers on Hopper
- [ ] Avoid shared memory bank conflicts
- [ ] Prefer regular access patterns

### 19.6.5 Autotuning Best Practices

**Search Space Design:**

1. **Start Coarse, Then Refine:**
   ```python
   # Phase 1: Coarse search
   coarse_space = [
       {'tile_size': ts, 'num_ctas': nc}
       for ts in [16, 32, 64, 128]
       for nc in [1, 2, 4, 8]
   ]
   
   # Phase 2: Fine search around best
   best = coarse_result.best.config
   fine_space = [
       {'tile_size': ts, 'num_ctas': best['num_ctas']}
       for ts in range(best['tile_size'] - 8, best['tile_size'] + 9, 4)
   ]
   ```

2. **Prune Obviously Bad Configs:**
   ```python
   # Skip configs that would exceed shared memory
   max_shared = 48 * 1024  # 48KB
   valid_configs = []
   for config in search_space:
       tile_bytes = config['tile_size'] ** 2 * 4  # float32
       if tile_bytes < max_shared:
           valid_configs.append(config)
   ```

3. **Use Domain Knowledge:**
   ```python
   # For matrix multiplication, tile sizes should typically:
   # - Be powers of 2
   # - Be multiples of 16 (warp size / 2)
   # - Not exceed register capacity
   
   good_tile_sizes = [16, 32, 64, 128]
   search_space = [
       {'tile_size': ts}
       for ts in good_tile_sizes
   ]
   ```

**Measurement Best Practices:**

1. **Warm-up Kernels:**
   ```python
   # Run kernel once before timing to warm up GPU
   kernel[stream, grid](*args)
   stream.synchronize()
   
   # Now measure
   start = time.time()
   kernel[stream, grid](*args)
   stream.synchronize()
   elapsed = time.time() - start
   ```

2. **Multiple Iterations:**
   ```python
   # Run multiple times and take average
   times = []
   for _ in range(10):
       start = time.time()
       kernel[stream, grid](*args)
       stream.synchronize()
       times.append(time.time() - start)
   
   mean_time = np.mean(times)
   std_time = np.std(times)
   ```

3. **Check Correctness:**
   ```python
   # Verify results match reference
   result_np = C_buf.copy_to_host()
   expected = A_np @ B_np
   assert np.allclose(result_np, expected, rtol=1e-5)
   ```

## 19.7 Performance Profiling

### 19.7.1 Nsight Compute Integration

Profile cuTile kernels with NVIDIA Nsight Compute:

```bash
# Basic profiling
ncu -o profile --set detailed python3 my_kernel.py

# Specific kernel
ncu -o profile -k my_kernel python3 my_script.py

# Detailed metrics
ncu -o profile --set full python3 my_script.py

# View results
ncu-ui profile.ncu-rep
```

**Key Metrics:**
- **DRAM Throughput:** Memory bandwidth utilization
- **Achieved Occupancy:** Actual vs theoretical occupancy
- **Warp Execution Efficiency:** Percent of warps active
- **Memory Workload:** Bytes read/written
- **Compute Throughput:** FLOPS achieved

### 19.7.2 Custom Timing

```python
import time

def benchmark_kernel(kernel, grid, args, stream, n_iters=100):
    """Benchmark a kernel with multiple iterations."""
    
    # Warm-up
    kernel[stream, grid](*args)
    stream.synchronize()
    
    # Timing
    start = time.perf_counter()
    for _ in range(n_iters):
        kernel[stream, grid](*args)
    stream.synchronize()
    end = time.perf_counter()
    
    # Statistics
    avg_time_ms = (end - start) / n_iters * 1000
    return avg_time_ms
```

## 19.8 Summary

This chapter covered comprehensive performance tuning techniques for cuTile kernels:

1. **Architecture-Specific Configuration:** Use `ByTarget` to optimize for different GPU architectures
2. **Kernel Parameters:** Tune `num_ctas`, `occupancy`, and `opt_level` for optimal performance
3. **Memory Hints:** Set `latency` and `allow_tma` to guide compiler optimizations
4. **Autotuning:** Use `tune.exhaustive_search` to automatically find optimal configurations
5. **Best Practices:** Follow guidelines for GPU clock management, tile size selection, and memory access patterns

Effective performance tuning requires understanding both your kernel's characteristics and the target GPU's capabilities. Use the profiling and autotuning tools to systematically explore the configuration space and find optimal settings for your specific workload.
