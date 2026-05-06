# Annotations and Optimization Hints

This document provides comprehensive reference for TileLang's annotation and optimization hint APIs. These annotations allow fine-grained control over memory access patterns, buffer layouts, cache behavior, and code generation decisions.

## Table of Contents

- [Overview](#overview)
- [T.use_swizzle](#tuse_swizzle)
- [T.annotate_layout](#tannotate_layout)
- [T.annotate_safe_value](#tannotate_safe_value)
- [T.annotate_l2_hit_ratio](#tannotate_l2_hit_ratio)
- [T.annotate_min_blocks_per_sm](#tannotate_min_blocks_per_sm)
- [T.annotate_restrict_buffers](#tannotate_restrict_buffers)
- [Performance Optimization Strategies](#performance-optimization-strategies)
- [Memory Access Pattern Guidelines](#memory-access-pattern-guidelines)
- [Register Allocation Hints](#register-allocation-hints)

---

## Overview

TileLang provides several annotation primitives that serve as optimization hints to the compiler. Unlike directives that change program semantics, annotations guide the compiler toward better code generation without altering the logical result. These annotations are particularly important for achieving peak GPU performance in production kernels.

All annotation functions are accessed through the `T` (TileLang language) namespace and are typically placed at the beginning of a kernel body or within a `T.Kernel` context.

---

## T.use_swizzle

```python
T.use_swizzle(panel_size: int, order: str = "row", enable: bool = True)
```

### Description

`T.use_swizzle` annotates a kernel to use a specific threadblock swizzle (rasterization) pattern for mapping thread blocks to the grid. Threadblock swizzling remaps the linear block ID to a 2D pattern that improves L2 cache locality and reduces shared memory bank conflicts when accessing global memory tiles.

### Parameters

| Parameter  | Type   | Default | Description                                                       |
|-----------|--------|---------|-------------------------------------------------------------------|
| `panel_size` | `int`  | required | The number of thread blocks per swizzle panel (group).            |
| `order`    | `str`  | `"row"` | The rasterization order. Either `"row"` or `"column"`.            |
| `enable`   | `bool` | `True`  | Whether to enable swizzling. When `False`, returns `None` (no-op). |

### Panel Size Selection

The `panel_size` parameter controls how many consecutive thread blocks form a swizzle panel. Selecting the right panel size is critical for performance:

- **Small panel sizes (2-8)**: Best for kernels with small tiles where many blocks can benefit from spatial locality. The thread blocks within a panel access contiguous regions of global memory.
- **Medium panel sizes (8-32)**: Common for GEMM and convolution kernels with moderate tile sizes. Provides a balance between L2 cache reuse and scheduling flexibility.
- **Large panel sizes (32+)**: Used for kernels with very large tiles or when the computation has strong temporal locality across adjacent blocks.

**Guidelines for panel size selection:**

1. Match the panel size to the number of blocks that share a common memory access footprint along one dimension.
2. For GEMM kernels with block sizes of 128x128, a panel size of 8 to 16 is typical.
3. For bandwidth-bound kernels (e.g., elementwise operations), larger panel sizes can improve L2 cache hit rates.
4. The panel size should typically be a power of 2 for efficient hardware address computation.

### Row vs Column Order

The `order` parameter controls the direction of the swizzle pattern:

- **`"row"` (default)**: Uses `rasterization2DRow` pattern. Thread blocks are grouped along rows first. This is the standard choice for kernels where the dominant memory access pattern is row-major (e.g., row-major GEMM where blocks iterate along the K dimension in rows).
- **`"column"`**: Uses `rasterization2DColumn` pattern. Thread blocks are grouped along columns first. Use this when the dominant access pattern is column-major or when the kernel's memory layout favors column-wise iteration.

**When to choose each:**

```python
# Row-major GEMM: A is MxK (row-major), B is KxN (row-major)
# Access A along rows -> use row order
T.use_swizzle(panel_size=16, order="row")

# Column-major GEMM or transposed access patterns
T.use_swizzle(panel_size=16, order="column")
```

### When Swizzling Helps

Swizzling is most beneficial in the following scenarios:

1. **Shared memory bank conflict reduction**: When adjacent thread blocks access shared memory in patterns that would cause bank conflicts without swizzling, the remapped block IDs distribute accesses across different banks.

2. **L2 cache locality**: Without swizzling, linear block IDs 0, 1, 2, ... may map to physically distant regions of the output matrix. Swizzling remaps these so that blocks accessing the same L2 cache lines are scheduled close together.

3. **DRAM burst efficiency**: Swizzling encourages coalesced global memory access across the thread blocks within a panel, improving DRAM burst utilization.

4. **Persistent threadblocks**: When using persistent threadblock scheduling (where blocks stay active and pull work from a queue), swizzling can improve the spatial locality of the work items assigned to each SM.

### Examples

#### Basic Usage in a GEMM Kernel

```python
@T.prim_func
def matmul_swizzled(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    with T.Kernel(T.ceildiv(M, 128), T.ceildiv(N, 128), threads=128) as (bx, by):
        # Apply swizzle for better L2 cache behavior
        T.use_swizzle(8, order="row")
        # ... kernel body ...
```

#### Disabling Swizzle Conditionally

```python
T.use_swizzle(16, enable=False)  # No-op, returns None
```

#### Column-Order Swizzle for Transposed Kernels

```python
@T.prim_func
def matmul_transposed(
    A: T.Tensor((K, M), "float16"),  # Note: A is transposed
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    with T.Kernel(T.ceildiv(M, 128), T.ceildiv(N, 128), threads=128) as (bx, by):
        T.use_swizzle(8, order="column")
        # ... kernel body ...
```

### Implementation Details

Internally, `T.use_swizzle` maps to a `threadblock_swizzle_pattern` attribute on the kernel. The attribute value is a tuple containing the rasterization function name and the panel size:

```python
# Internal representation (row order):
attr(None, "threadblock_swizzle_pattern", tvm_tuple("rasterization2DRow", panel_size))

# Internal representation (column order):
attr(None, "threadblock_swizzle_pattern", tvm_tuple("rasterization2DColumn", panel_size))
```

The swizzle pattern is applied during kernel launch when computing the physical block ID from the logical block ID. The generated host code remaps `(bx, by)` coordinates according to the selected pattern before passing them to the kernel.

---

## T.annotate_layout

```python
T.annotate_layout(layout_map: dict)
```

### Description

`T.annotate_layout` attaches custom layout annotations to buffers, specifying how data elements are arranged in memory. This is essential for achieving optimal memory access patterns, particularly for shared memory and register files (fragments) used in Tensor Core operations.

### Parameters

| Parameter    | Type   | Description                                                      |
|-------------|--------|------------------------------------------------------------------|
| `layout_map` | `dict` | A dictionary mapping buffers to their desired layouts. Keys are buffer objects; values are `Layout` objects or callable layout functions. |

### Layout Map Format

The `layout_map` dictionary accepts two types of values for each buffer:

1. **`Layout` object**: A pre-constructed `Layout` instance from `tilelang.layout.Layout`. This directly specifies the memory arrangement.

2. **Callable function**: A function that takes a coordinate tuple and returns the linear offset. The function is automatically wrapped in a `Layout` object using the buffer's shape.

For `Fragment` buffers (register-level Tensor Core accumulators), the layout must be a `Fragment` instance from `tilelang.layout.Fragment`.

### Integration with Layout Class

TileLang provides two layout classes:

- **`Layout`**: Represents a generic memory layout mapping from logical coordinates to linear offsets. Used for shared memory and global memory buffers.
- **`Fragment`**: A specialized layout for Tensor Core register fragments. Encodes the mapping between logical tensor elements and physical register locations within a warp's register file.

```python
from tilelang.layout import Layout, Fragment

# Using a pre-built Layout
layout = Layout((128, 128), lambda i, j: i * 128 + j)  # row-major
T.annotate_layout({shared_buf: layout})

# Using a callable
T.annotate_layout({shared_buf: lambda i, j: j * 128 + i})  # column-major

# Using a Fragment for Tensor Core accumulators
frag_layout = Fragment(...)  # WGMMA or MMA fragment layout
T.annotate_layout({accum_buf: frag_layout})
```

### Examples

#### Annotating Shared Memory Layout for Tensor Core

```python
@T.prim_func
def matmul_annotated(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    with T.Kernel(T.ceildiv(M, 128), T.ceildiv(N, 128), threads=128) as (bx, by):
        A_shared = T.alloc_shared((128, 32), "float16")
        B_shared = T.alloc_shared((32, 128), "float16")

        # Annotate with swizzled layouts to avoid bank conflicts
        T.annotate_layout({
            A_shared: swizzled_layout_128x32,
            B_shared: swizzled_layout_32x128,
        })

        # ... copy and compute ...
```

#### Annotating Fragment Layout for WGMMA Accumulators

```python
from tilelang.layout import Fragment

@T.prim_func
def wgmma_kernel(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float32"),
):
    with T.Kernel(...) as (bx, by):
        C_frag = T.alloc_fragment((128, 128), "float32")

        # Annotate the accumulator with WGMMA fragment layout
        T.annotate_layout({C_frag: wgmma_fragment_layout})

        # ... WGMMA operations use the annotated layout ...
```

### How Layout Annotations Affect Code Generation

When a buffer has an annotated layout:

1. **Copy operations**: `T.copy` uses the annotated layout to determine the optimal load/store pattern for the target architecture.
2. **Tensor Core operations**: `T.gemm` / WGMMA operations use the layout to correctly map between logical tensor coordinates and physical register/SMEM locations.
3. **Shared memory access**: The layout determines the addressing formula used when accessing shared memory, which directly affects bank conflict behavior.

Without layout annotations, TileLang's `LayoutInferences` pass automatically infers layouts based on the operations that consume each buffer. Annotations override the inferred layout when the user has domain-specific knowledge about the optimal arrangement.

---

## T.annotate_safe_value

```python
T.annotate_safe_value(safe_value_map: dict)
```

### Description

`T.annotate_safe_value` specifies safe (default) values for buffers that may be accessed out of bounds. When TileLang inserts safety checks for boundary conditions (e.g., when the tensor dimensions are not perfectly divisible by the tile size), this annotation tells the compiler what value to use for out-of-bound elements rather than inserting conditional guards.

### Parameters

| Parameter         | Type   | Description                                                       |
|------------------|--------|-------------------------------------------------------------------|
| `safe_value_map` | `dict` | A dictionary mapping buffers to their safe (padding) values. Keys are buffer objects; values are scalar constants. |

### Use Cases

1. **GEMM with non-divisible dimensions**: When `M` is not a multiple of the block size, the last tile along the M dimension reads beyond the boundary. Setting the safe value to 0 for the input matrix ensures those out-of-bound elements contribute nothing to the dot product.

2. **Convolution with padding**: Padding values for input feature maps can be specified as safe values.

3. **Reduction operations**: When reducing along a dimension that is not evenly divisible by the reduction tile size, safe values ensure correctness without branching.

### Examples

#### GEMM with Safe Padding

```python
@T.prim_func
def matmul_safe(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    with T.Kernel(T.ceildiv(M, BLOCK_M), T.ceildiv(N, BLOCK_N), threads=128) as (bx, by):
        # When M or N are not divisible by BLOCK_M/BLOCK_N,
        # out-of-bound reads will return 0 instead of undefined values
        T.annotate_safe_value({
            A: T.float16(0),
            B: T.float16(0),
        })
        # ... kernel body ...
```

#### Specifying Different Safe Values Per Buffer

```python
T.annotate_safe_value({
    input_buf: T.float16(0),      # Zero-fill for inputs
    mask_buf: T.bool(True),        # Default to "masked" for safety
    scale_buf: T.float32(1.0),     # Default scale of 1.0
})
```

### Interaction with LegalizeSafeMemoryAccess

The `LegalizeSafeMemoryAccess` transform pass uses safe value annotations to generate efficient boundary-handling code. When safe values are annotated:

1. The pass generates unconditional loads (no branching) for the common case.
2. For edge tiles, a predicated store writes the safe value to a temporary buffer location before the main copy operation.
3. This avoids warp divergence caused by per-element bounds checking.

Without safe value annotations, `LegalizeSafeMemoryAccess` inserts conditional guards on every element access in edge tiles, which can significantly impact performance due to warp divergence.

---

## T.annotate_l2_hit_ratio

```python
T.annotate_l2_hit_ratio(l2_hit_ratio_map: dict)
```

### Description

`T.annotate_l2_hit_ratio` sets L2 cache persistence hints for global memory buffers. On NVIDIA GPUs with compute capability 8.0+ (Ampere and later), this annotation leverages the L2 cache persistence feature to control how long data remains in the L2 cache.

### Parameters

| Parameter            | Type   | Description                                                       |
|---------------------|--------|-------------------------------------------------------------------|
| `l2_hit_ratio_map` | `dict` | A dictionary mapping global-scope buffers to their desired L2 hit ratios. Values must be floats between 0.0 and 1.0. |

### Constraints

- Only global-scope buffers can be annotated. Attempting to annotate shared or local buffers will raise an assertion error.
- The hit ratio is a float between 0.0 (no persistence) and 1.0 (maximum persistence).

### How L2 Persistence Works

On Ampere+ GPUs, the L2 cache can be partitioned into a persistent region and a streaming region:

- **Persistent region**: Data in this region is less likely to be evicted. Suitable for frequently accessed data (e.g., weight matrices in inference).
- **Streaming region**: Normal L2 cache behavior. Data is evicted using the standard replacement policy.

The hit ratio parameter maps to the NVIDIA `cudaAccessPolicyWindow` API:

- A ratio of 0.5 means approximately 50% of the L2 cache is reserved for persistent data.
- A ratio of 1.0 means the entire L2 cache is used for persistent data (aggressive).
- A ratio of 0.0 means no persistent reservation (default behavior).

### Examples

#### Persistent L2 for Weight Matrix in Inference

```python
@T.prim_func
def matmul_persistent_l2(
    A: T.Tensor((M, K), "float16"),  # Activations (streaming)
    B: T.Tensor((K, N), "float16"),  # Weights (persistent)
    C: T.Tensor((M, N), "float16"),
):
    with T.Kernel(T.ceildiv(M, 128), T.ceildiv(N, 128), threads=128) as (bx, by):
        # Keep weights in L2 cache across multiple inference batches
        T.annotate_l2_hit_ratio({
            B: 0.7,  # 70% of L2 reserved for weight data
        })
        # ... kernel body ...
```

#### Multiple Buffers with Different Ratios

```python
T.annotate_l2_hit_ratio({
    key_buffer: 0.8,     # Keys should stay cached (attention mechanism)
    value_buffer: 0.8,   # Values should stay cached
    query_buffer: 0.2,   # Queries are used once, lower priority
})
```

### Implementation Details

Internally, the annotation converts the float hit ratio to a `FloatImm("float32", float(hit_ratio))` and attaches it as a block attribute named `l2_hit_ratio_map`. The `LowerL2Persistent` transform pass processes these annotations and generates the appropriate `cudaAccessPolicyWindow` setup code in the host-side launcher.

### When to Use

- **Batched inference**: When the same weights are used across many input batches, persisting them in L2 cache can dramatically reduce memory bandwidth requirements.
- **Attention mechanisms**: Key and value matrices that are reused across multiple query positions benefit from L2 persistence.
- **Iterative algorithms**: Kernels that repeatedly access the same data in multiple passes (e.g., some optimization algorithms).

---

## T.annotate_min_blocks_per_sm

```python
T.annotate_min_blocks_per_sm(n: int)
```

### Description

`T.annotate_min_blocks_per_sm` sets the minimum number of thread blocks that should reside on each Streaming Multiprocessor (SM) simultaneously. This hint is passed as the second argument of CUDA's `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` attribute.

### Parameters

| Parameter | Type  | Description                                                       |
|----------|-------|-------------------------------------------------------------------|
| `n`      | `int` | Minimum number of thread blocks per SM. Must be a positive integer. |

### How It Works

The `minBlocksPerMultiprocessor` parameter provides the compiler with an occupancy target:

1. **Register allocation**: The compiler limits per-block register usage to ensure `n` blocks can fit on each SM. Since each SM has a fixed register file (typically 65536 registers), more blocks per SM means fewer registers per block.
2. **Shared memory allocation**: Similarly, shared memory usage per block is constrained so that `n` blocks can share the SM's shared memory.
3. **Occupancy**: Higher minimum blocks per SM increases occupancy, which helps hide memory latency through warp-level interleaving.

### Trade-offs

| Setting     | Register Usage | Occupancy | Latency Hiding | Register Spilling |
|------------|----------------|-----------|----------------|-------------------|
| `n = 1`    | Maximum        | Low       | Poor           | None              |
| `n = 2`    | Moderate       | Medium    | Good           | Minimal           |
| `n = 4`    | Limited        | High      | Excellent      | Possible          |
| `n = 8`    | Very limited   | Very high | Excellent      | Likely            |

### Examples

#### Compute-Bound Kernel with Moderate Occupancy

```python
@T.prim_func
def matmul_compute_bound(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float32"),
):
    with T.Kernel(T.ceildiv(M, 128), T.ceildiv(N, 128), threads=128) as (bx, by):
        T.annotate_min_blocks_per_sm(2)
        # ... kernel body ...
```

#### Memory-Bound Kernel Requiring High Occupancy

```python
@T.prim_func
def elementwise_memory_bound(
    X: T.Tensor((N,), "float32"),
    Y: T.Tensor((N,), "float32"),
):
    with T.Kernel(T.ceildiv(N, 1024), threads=512) as pid:
        T.annotate_min_blocks_per_sm(4)
        # ... kernel body ...
```

### Implementation Details

The annotation is stored as a `tl.min_blocks_per_sm` attribute on the kernel function. During code generation, this value is emitted in the `__launch_bounds__` annotation:

```c
// Generated CUDA code:
__global__ void __launch_bounds__(128, 2) my_kernel(...) { ... }
```

---

## T.annotate_restrict_buffers

```python
T.annotate_restrict_buffers(*buffers)
```

### Description

`T.annotate_restrict_buffers` marks the specified buffer parameters as potentially aliasing, which causes the code generator to omit the `__restrict__` qualifier for those parameters. By default, TileLang assumes all buffer parameters are non-aliasing (restrict-qualified). This annotation is used when buffers may overlap.

### Parameters

| Parameter  | Type     | Description                                                       |
|-----------|----------|-------------------------------------------------------------------|
| `*buffers` | `Buffer` | One or more buffer objects that may alias each other.              |

### When to Use

- **In-place operations**: When an output buffer is the same as an input buffer (e.g., in-place ReLU).
- **Overlapping slices**: When two buffer parameters are slices of the same base tensor.
- **Aliased views**: When different buffer parameters represent different views (e.g., real and imaginary parts) of the same memory.

### Examples

#### In-Place Addition

```python
@T.prim_func
def inplace_add(
    X: T.Tensor((N,), "float32"),
    Y: T.Tensor((N,), "float32"),
):
    # X and Y might point to the same memory
    T.annotate_restrict_buffers(X, Y)
    with T.Kernel(T.ceildiv(N, 256), threads=256) as pid:
        Y[pid] = X[pid] + Y[pid]
```

#### Multiple Aliased Buffers

```python
@T.prim_func
def fused_op(
    input: T.Tensor((N,), "float32"),
    output: T.Tensor((N,), "float32"),
    workspace: T.Tensor((N,), "float32"),
):
    # input and output might be the same buffer (in-place)
    T.annotate_restrict_buffers(input, output)
    with T.Kernel(T.ceildiv(N, 256), threads=256) as pid:
        # Safe even if input == output
        output[pid] = input[pid] * 2.0 + workspace[pid]
```

### Implementation Details

The annotation stores the buffer data variables in a `tl.non_restrict_params` block attribute. The `HoistNonRestrictParams` transform pass reads this attribute and marks the corresponding function parameters so that the CUDA code generator omits `__restrict__`:

```c
// Without annotation (default):
__global__ void kernel(float* __restrict__ X, float* __restrict__ Y) { ... }

// With T.annotate_restrict_buffers(X, Y):
__global__ void kernel(float* X, float* Y) { ... }
```

Note that removing `__restrict__` may prevent the compiler from performing certain optimizations (e.g., load-store reordering) that assume no aliasing.

---

## Performance Optimization Strategies

### Strategy 1: Maximize Memory Throughput

For memory-bound kernels:

1. Use `T.use_swizzle` to improve L2 cache locality across thread blocks.
2. Use `T.annotate_l2_hit_ratio` for data that is reused across multiple blocks or kernel launches.
3. Ensure shared memory layouts avoid bank conflicts using `T.annotate_layout`.
4. Use `T.annotate_min_blocks_per_sm` to increase occupancy for better latency hiding.

### Strategy 2: Maximize Compute Throughput

For compute-bound kernels:

1. Use Tensor Core operations (WGMMA, MMA) whenever possible.
2. Use `T.annotate_layout` with proper Fragment layouts for Tensor Core accumulators.
3. Set `T.annotate_min_blocks_per_sm` to balance register usage vs. occupancy.
4. Avoid excessive register spilling by not setting `min_blocks_per_sm` too high.

### Strategy 3: Optimize Pipeline Efficiency

For pipelined kernels:

1. Use `T.annotate_safe_value` to eliminate branching in edge tiles.
2. Annotate shared memory layouts for conflict-free access in the steady-state pipeline.
3. Consider persistent L2 cache for data that persists across pipeline stages.

### Strategy 4: Optimize for Specific Architectures

- **Ampere (SM80)**: Leverage async copy (`cp.async`) and L2 persistence.
- **Hopper (SM90)**: Use TMA (Tensor Memory Access) for global-to-shared copies and WGMMA for compute. Cluster-level swizzling is beneficial.
- **Blackwell (SM100)**: Use TCGEN05 operations and TMEM for advanced matrix operations.

---

## Memory Access Pattern Guidelines

### Global Memory Access

1. **Coalescing**: Ensure consecutive threads access consecutive memory addresses. TileLang's `T.copy` and vectorized operations handle this automatically when layouts are properly annotated.

2. **Vectorized loads**: Use wider load instructions (128-bit, 256-bit) when possible. TileLang's `T.ldg128`, `T.ldg256` intrinsics provide explicit control.

3. **Read-only cache**: Use `T.__ldg()` to route loads through the read-only data cache when data will not be written during the kernel's lifetime.

### Shared Memory Access

1. **Bank conflict avoidance**: NVIDIA GPUs have 32 shared memory banks. When multiple threads in a warp access the same bank simultaneously, bank conflicts occur and accesses are serialized. Use `T.annotate_layout` with swizzled layouts to avoid conflicts.

2. **Padding**: A simple technique to avoid bank conflicts is to pad shared memory allocations by one element per row. However, annotated swizzled layouts are more memory-efficient.

3. **Async copy**: For Ampere+, use `T.copy` with async semantics to overlap global-to-shared copies with computation.

### Register Access

1. **Fragment layout**: Register files used for Tensor Core operations have specific layout requirements. Always use `T.annotate_layout` with the correct `Fragment` layout for accumulator buffers.

2. **Register pressure**: Monitor register usage when setting `T.annotate_min_blocks_per_sm`. Excessive register spilling to local memory can negate the benefits of higher occupancy.

---

## Register Allocation Hints

TileLang provides several APIs for controlling register allocation, particularly in the context of warp-specialized kernels:

### T.inc_max_nreg / T.dec_max_nreg

```python
T.inc_max_nreg(reg_count: int)   # Increment max registers
T.dec_max_nreg(reg_count: int)   # Decrement max registers
```

These intrinsics control the per-warp-group register allocation on Hopper+ GPUs. They map to the PTX `setmaxnreg` instruction:

- `T.inc_max_nreg(n)`: Increases the register allocation for the current warp group by `n` registers.
- `T.dec_max_nreg(n)`: Decreases the register allocation for the current warp group by `n` registers.

### T.no_set_max_nreg

```python
T.no_set_max_nreg()
```

Disables automatic register allocation management for the current kernel. When called, the compiler will not emit `setmaxnreg` instructions, giving full control to the programmer.

### T.annotate_producer_reg_dealloc / T.annotate_consumer_reg_alloc

```python
T.annotate_producer_reg_dealloc(reg_count: int = 24)
T.annotate_consumer_reg_alloc(reg_count: int = 240)
```

These annotations hint at register allocation behavior in warp-specialized kernels:

- `T.annotate_producer_reg_dealloc`: Hints that the producer warp group will release `reg_count` registers, making them available for other warp groups.
- `T.annotate_consumer_reg_alloc`: Hints that the consumer warp group needs `reg_count` registers for its computation.

### AnnotateWarpGroupRegAlloc Transform Pass

The `AnnotateWarpGroupRegAlloc` transform pass automatically injects `set_max_nreg` calls into warp-specialized functions based on the register hints:

1. It collects `set_max_nreg` and `no_set_max_nreg` calls from the function body.
2. For warp-specialized producer-consumer patterns, it injects register deallocation in the producer branch and allocation in the consumer branch.
3. This ensures the producer uses fewer registers (freeing them for the consumer's Tensor Core operations).

### Example: Warp-Specialized GEMM with Register Hints

```python
@T.prim_func
def warp_specialized_gemm(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float32"),
):
    with T.Kernel(T.ceildiv(M, 128), T.ceildiv(N, 256), threads=256) as (bx, by):
        T.annotate_producer_reg_dealloc(40)
        T.annotate_consumer_reg_alloc(232)

        # Producer warp group: load data
        # Consumer warp group: compute with WGMMA
        # Register allocation is automatically managed by the compiler
        ...
```

### Register Allocation Best Practices

1. **Measure first**: Use the profiler to determine if register pressure is a bottleneck before applying register hints.

2. **Producer minimization**: In warp-specialized kernels, the producer (load) warp group should use as few registers as possible to maximize registers available for the consumer (compute) warp group.

3. **Consumer maximization**: The consumer warp group typically needs many registers for Tensor Core accumulator fragments. Ensure sufficient registers are allocated to avoid spilling.

4. **Verify with SASS**: Use `JITKernel.show_sass()` to verify that the generated code has the expected register allocation and that no unexpected spilling occurs.

5. **Use `T.annotate_min_blocks_per_sm` together**: When using warp specialization, the minimum blocks per SM is effectively 1 (one warp-specialized block per SM). Set `T.annotate_min_blocks_per_sm(1)` for consistency.

---

## Summary of Annotations by Kernel Type

| Kernel Type          | Recommended Annotations                                           |
|---------------------|------------------------------------------------------------------|
| GEMM (standard)     | `use_swizzle`, `annotate_min_blocks_per_sm`, `annotate_safe_value` |
| GEMM (warp-specialized) | `annotate_producer_reg_dealloc`, `annotate_consumer_reg_alloc`, `annotate_min_blocks_per_sm` |
| GEMM (non-divisible) | `annotate_safe_value`, `use_swizzle`                             |
| Elementwise          | `use_swizzle`, `annotate_l2_hit_ratio`, `annotate_min_blocks_per_sm` |
| Reduction            | `annotate_safe_value`, `annotate_min_blocks_per_sm`              |
| Convolution          | `annotate_layout`, `annotate_safe_value`, `use_swizzle`          |
| Attention            | `annotate_l2_hit_ratio`, `use_swizzle`, `annotate_safe_value`    |
| Inference (batched)  | `annotate_l2_hit_ratio` for weights                              |

---

## Environment Variable Overrides

Several annotation behaviors can be controlled through environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `TILELANG_TARGET` | Default compilation target | `"auto"` |
| `TILELANG_EXECUTION_BACKEND` | Default execution backend | `"auto"` |
| `TILELANG_VERBOSE` | Enable verbose compilation output | `False` |

These can be overridden on a per-compilation basis through the `tilelang.compile()` and `tilelang.jit()` parameters.
