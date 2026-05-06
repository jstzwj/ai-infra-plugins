# TileLang Control Flow Reference

This reference covers all control flow constructs available in TileLang, from basic serial and parallel loops to advanced software pipelining and persistent kernel patterns. Understanding these constructs is essential for writing efficient GPU kernels that maximize hardware utilization.

---

## Table of Contents

1. [Overview](#overview)
2. [T.Parallel -- Parallel Loops](#tparallel----parallel-loops)
3. [T.Pipelined -- Software Pipelining](#tpipelined----software-pipelining)
4. [T.serial -- Serial Loops](#tserial----serial-loops)
5. [T.unroll -- Unrolled Loops](#tunroll----unrolled-loops)
6. [T.Persistent -- Persistent Kernels](#tpersistent----persistent-kernels)
7. [T.Vectorized -- Vectorized Loops](#tvectorized----vectorized-loops)
8. [Conditionals](#conditionals)
9. [T.break and T.continue](#tbreak-and-tcontinue)
10. [While Loops](#while-loops)
11. [Loop Annotations](#loop-annotations)
12. [Boundary Handling](#boundary-handling)
13. [Practical Examples](#practical-examples)

---

## Overview

TileLang provides a rich set of control flow constructs that map directly to efficient GPU execution patterns. Unlike standard Python loops which execute sequentially, TileLang loops are compiled into GPU-parallel execution with explicit control over:

- **Parallelism**: How work is distributed across threads, warps, and blocks.
- **Pipelining**: How memory and compute operations overlap.
- **Unrolling**: How loop bodies are replicated for instruction-level parallelism.
- **Vectorization**: How memory accesses are coalesced into wide loads/stores.

### Control Flow Construct Summary

| Construct | Execution Model | Primary Use Case |
|-----------|----------------|-----------------|
| `T.Parallel` | Multi-thread parallel | Data-parallel work distribution |
| `T.Pipelined` | Software pipelined | Overlap compute and memory |
| `T.serial` | Sequential within a thread | Ordered operations, dependencies |
| `T.unroll` | Fully unrolled | Small fixed-size iterations |
| `T.Persistent` | Persistent threads | Long-running kernels, work queues |
| `T.Vectorized` | Vectorized memory | Coalesced memory access patterns |
| `if/else` | Per-thread conditional | Branching logic |
| `while` | Per-thread iteration | Dynamic iteration counts |
| `T.break` | Loop exit | Early termination |
| `T.continue` | Loop skip | Skip iterations |

---

## T.Parallel -- Parallel Loops

### Signature

```python
T.Parallel(
    *extents,                           # Extent(s) of the parallel loop
    coalesced_width=None,               # Width for memory coalescing
    loop_layout=None,                   # Layout specification for multi-dim loops
    prefer_async=None,                  # Prefer async execution
    annotations=None,                   # Additional annotations
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `*extents` | int(s) | required | One or more integers specifying the extent of each dimension. For a single extent, creates a 1D parallel loop. For multiple extents, creates a multi-dimensional parallel loop. |
| `coalesced_width` | int or None | `None` | Specifies the width of the coalesced access pattern. When set, threads are arranged so that consecutive threads access consecutive memory locations for better coalescing. |
| `loop_layout` | Layout or None | `None` | Explicit layout specification for how the parallel iteration space maps to threads. Overrides the default row-major mapping. |
| `prefer_async` | bool or None | `None` | If `True`, the compiler will attempt to generate async instructions where possible. |
| `annotations` | dict or None | `None` | Additional annotations to attach to the loop for compiler hints. |

### Basic Usage

A `T.Parallel` loop distributes its iterations across all threads in the thread block. Each thread executes a subset of the total iterations:

```python
import tilelang.language as T

# 1D parallel loop: 128 threads each handle (N / 128) elements
with T.Parallel(N) as i:
    buffer[i] = buffer[i] * 2.0
```

### Multi-dimensional Parallel Loops

When multiple extents are provided, `T.Parallel` creates a multi-dimensional iteration space that is flattened into the 1D thread hierarchy:

```python
# 2D parallel loop over a tile of [block_M, block_N]
with T.Parallel(block_M, block_N) as i, j:
    output[i, j] = input[i, j] + bias[j]

# 3D parallel loop
with T.Parallel(D, H, W) as d, h, w:
    volume[d, h, w] = volume[d, h, w] * scale
```

The total number of iterations (`block_M * block_N`) must not exceed the number of threads in the thread block. If it does, each thread handles multiple iterations in a grid-stride pattern.

### Coalesced Width Optimization

The `coalesced_width` parameter controls how thread IDs map to iteration indices for optimal memory access patterns:

```python
# Without coalesced_width: threads may access memory in a scattered pattern
with T.Parallel(M, N) as i, j:
    matrix[i, j] = matrix[i, j] + 1.0

# With coalesced_width: consecutive threads access consecutive j indices
# This ensures memory accesses are coalesced when j varies along the
# last (contiguous) dimension
with T.Parallel(M, N, coalesced_width=N) as i, j:
    matrix[i, j] = matrix[i, j] + 1.0
```

When `coalesced_width` is set to the extent of the innermost dimension, the compiler arranges threads so that consecutive thread IDs correspond to consecutive indices in the innermost loop. This produces coalesced memory accesses, which are critical for achieving peak memory bandwidth.

**How it works internally**:

```
Without coalescing (row-major thread assignment):
  Thread 0 -> (i=0, j=0), Thread 1 -> (i=0, j=1), ...
  Thread N -> (i=1, j=0), Thread N+1 -> (i=1, j=1), ...

With coalesced_width=N:
  Thread 0 -> (i=0, j=0), Thread 1 -> (i=0, j=1), ...
  Thread N -> (i=1, j=0), Thread N+1 -> (i=1, j=1), ...
  (Same mapping, but the compiler ensures the innermost index
   varies fastest across consecutive threads)
```

### Loop Layout Specification

The `loop_layout` parameter provides fine-grained control over how the multi-dimensional iteration space maps to the 1D thread index space:

```python
from tilelang.language import Layout

# Custom layout: column-major thread mapping
# Thread k maps to (i=k % M, j=k / M)
layout = Layout.row_major(block_M, block_N)

with T.Parallel(block_M, block_N, loop_layout=layout) as i, j:
    data[i, j] = data[i, j] * 2.0
```

### Parallel Loop as Context Manager

`T.Parallel` is typically used as a context manager with Python's `with` statement:

```python
with T.Parallel(N) as i:
    # This body runs in parallel across all threads
    # Each thread gets a different value of i
    result[i] = compute(input[i])
```

### Example: Parallel Element-wise Operation

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[1])
def parallel_relu(N, dtype="float16"):
    input_buf = T.alloc_shared([N], dtype)
    output_buf = T.alloc_shared([N], dtype)

    T.copy(input_global, input_buf)

    # Apply ReLU in parallel across all elements
    with T.Parallel(N) as i:
        output_buf[i] = T.max(input_buf[i], 0.0)

    T.copy(output_buf, output_global)
    return output_global
```

### Example: 2D Parallel Matrix Addition

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])
def parallel_matadd(M, N, block_M=64, block_N=64, dtype="float16"):
    A = T.alloc_shared([block_M, block_N], dtype)
    B = T.alloc_shared([block_M, block_N], dtype)
    C = T.alloc_shared([block_M, block_N], dtype)

    T.copy(A_global, A)
    T.copy(B_global, B)

    # Parallel 2D element-wise addition
    with T.Parallel(block_M, block_N, coalesced_width=block_N) as i, j:
        C[i, j] = A[i, j] + B[i, j]

    T.copy(C, C_global)
    return C_global
```

---

## T.Pipelined -- Software Pipelining

### Signature

```python
T.Pipelined(
    start,                      # Start index (inclusive)
    stop=None,                  # Stop index (exclusive)
    num_stages=0,               # Number of pipeline stages
    order=None,                 # Execution order
    stage=None,                 # Current stage specification
    sync=None,                  # Synchronization mode
    group=None,                 # Thread group for pipelining
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | int | required | Starting index of the pipeline loop. Can also be a range object. |
| `stop` | int or None | `None` | Ending index (exclusive). If `None`, `start` is treated as the stop value and start defaults to 0. |
| `num_stages` | int | `0` | Number of pipeline stages. When `0`, the compiler auto-selects based on the loop body. Typical values: 2-5. |
| `order` | Order or None | `None` | The execution order for the pipeline stages. Controls how stages are interleaved. |
| `stage` | Stage or None | `None` | Specifies which part of the loop body belongs to which pipeline stage. |
| `sync` | Sync or None | `None` | Synchronization mode between pipeline stages. |
| `group` | int or None | `None` | Thread group size for cooperative pipeline stages. |

### Overview

Software pipelining is a technique that overlaps the execution of different loop iterations to hide memory latency. In a traditional serial loop, each iteration must complete before the next one begins:

```
Serial execution:
  Iter 0: [Load A0,B0] -> [Compute C0] -> [Store C0]
  Iter 1: [Load A1,B1] -> [Compute C1] -> [Store C1]
  Iter 2: [Load A2,B2] -> [Compute C2] -> [Store C2]
```

With software pipelining, the load phase of iteration N+1 overlaps with the compute phase of iteration N:

```
Pipelined execution (3 stages):
  Stage 1:  [Load A0,B0]
  Stage 2:  [Load A1,B1]  [Compute C0]
  Stage 3:  [Load A2,B2]  [Compute C1]  [Store C0]
  Steady:   [Load Ai,Bi]  [Compute Ci-1] [Store Ci-2]
```

This overlapping can significantly improve throughput by keeping both the memory subsystem and compute units busy.

### Pipeline Stages Explanation

The `num_stages` parameter determines how many iterations of the loop are "in flight" simultaneously:

- **1 stage**: No pipelining. Equivalent to a serial loop.
- **2 stages**: Load of next iteration overlaps with compute of current iteration.
- **3 stages**: Load of next-next iteration overlaps with compute of next and store of current.
- **4+ stages**: Further overlap, but requires more shared memory for buffering.

### Stage Selection Guide

| Scenario | Recommended Stages | Rationale |
|----------|-------------------|-----------|
| Memory-bound kernel | 2-3 | Overlap memory with compute |
| Compute-bound kernel | 1-2 | Compute already saturates SM |
| Small tiles (16-32 KB) | 3-4 | Can afford the memory overhead |
| Large tiles (64+ KB) | 2 | Limited shared memory |
| Hopper with TMA | 3-4 | TMA async copy enables deep pipelining |

### Order, Stage, Sync, and Group Parameters

These parameters provide fine-grained control over the pipeline execution:

#### Order Parameter

Controls how iterations are mapped to pipeline stages:

```python
from tilelang.language import Layout

# Outer-Inner order: pipeline the outer loop of a nested structure
T.Pipelined(K // block_K, num_stages=3, order=Layout.kOuterInner)

# Inner-Outer order: pipeline the inner loop
T.Pipelined(K // block_K, num_stages=3, order=Layout.kInnerOuter)
```

#### Stage Parameter

Explicitly assigns loop body operations to pipeline stages:

```python
# Stage 0: Memory operations (load)
# Stage 1: Compute operations (GEMM)
# Stage 2: Store operations
with T.Pipelined(K // block_K, num_stages=3, stage=stage_def) as k:
    with stage_def.stage(0):
        T.copy(A_global[k], A_smem)
        T.copy(B_global[k], B_smem)
    with stage_def.stage(1):
        T.gemm(A_smem, B_smem, C_local)
    with stage_def.stage(2):
        T.copy(C_local, C_global)
```

#### Sync Parameter

Controls synchronization between pipeline stages:

```python
# Sync after each stage: strict ordering
T.Pipelined(K // block_K, num_stages=3, sync=SyncStage.kSyncAfterStage)

# No sync between stages: maximum overlap (requires care)
T.Pipelined(K // block_K, num_stages=2, sync=SyncStage.kNoSync)
```

#### Group Parameter

Specifies the thread group size for cooperative operations within pipeline stages:

```python
# Group of 128 threads (4 warps) cooperate on each stage
T.Pipelined(K // block_K, num_stages=3, group=128)
```

### Example: Basic Pipelined GEMM

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])
def pipelined_gemm(M, N, K, block_M=128, block_N=128, block_K=32, num_stages=3):
    # Allocate multi-stage shared memory buffers
    A_smem = T.alloc_shared([num_stages, block_M, block_K], "float16")
    B_smem = T.alloc_shared([num_stages, block_K, block_N], "float16")
    C_local = T.alloc_local([block_M, block_N], "float32")

    T.clear(C_local)

    # Pipelined K loop with 3 stages
    for k in T.Pipelined(K // block_K, num_stages=num_stages):
        stage_idx = k % num_stages

        # Load next tile (overlaps with compute of previous tile)
        T.copy(A_global[k * block_K:(k+1) * block_K], A_smem[stage_idx])
        T.copy(B_global[k * block_K:(k+1) * block_K], B_smem[stage_idx])

        # Synchronize shared memory
        T.sync_shared_memory()

        # Compute on current tile
        T.gemm(A_smem[stage_idx], B_smem[stage_idx], C_local)

    T.copy(C_local, C_global)
    return C_global
```

### Example: Multi-stage Pipeline with TMA on Hopper

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])
def tma_pipelined_gemm(M, N, K, block_M=128, block_N=128, block_K=64):
    num_stages = 3

    A_smem = T.alloc_shared([num_stages, block_M, block_K], "float16")
    B_smem = T.alloc_shared([num_stages, block_K, block_N], "float16")
    C_local = T.alloc_local([block_M, block_N], "float32")
    mbar = T.alloc_shared([num_stages], "uint64")

    T.clear(C_local)

    for k in T.Pipelined(K // block_K, num_stages=num_stages):
        s = k % num_stages

        # Wait for this stage's buffer to be available
        T.wait_memory_barrier(mbar[s])

        # Launch async TMA copy for next iteration
        next_s = (k + 1) % num_stages
        T.copy(A_global[(k+1) * block_K:(k+2) * block_K], A_smem[next_s], mbar=mbar[next_s])
        T.copy(B_global[(k+1) * block_K:(k+2) * block_K], B_smem[next_s], mbar=mbar[next_s])

        # Compute on current tile
        T.gemm(A_smem[s], B_smem[s], C_local)

    T.copy(C_local, C_global)
    return C_global
```

### How Pipelining Overlaps Compute and Memory

The key insight behind software pipelining is that GPUs have separate hardware units for memory operations and arithmetic operations. By interleaving these operations, both units can be active simultaneously:

```
Timeline without pipelining:
  Memory: [====Load 0====]                  [====Load 1====]                  [====Load 2====]
  Compute:                 [====GEMM 0====]                  [====GEMM 1====]
  Total time: ~3 * (load_time + compute_time)

Timeline with 2-stage pipelining:
  Memory: [====Load 0====][====Load 1====][====Load 2====]
  Compute:                  [====GEMM 0====][====GEMM 1====]
  Total time: ~load_time + 2 * max(load_time, compute_time)

Timeline with 3-stage pipelining:
  Memory: [Load 0][Load 1][Load 2][Load 3]
  Compute:         [GEMM 0][GEMM 1][GEMM 2]
  Total time: ~2 * load_time + max(load_time, compute_time)
```

### Optimal Stage Count Selection

The optimal number of stages depends on the ratio of memory latency to compute time:

```
optimal_stages = ceil(memory_latency / compute_time) + 1
```

However, practical considerations include:

1. **Shared memory budget**: Each additional stage requires a full buffer copy in shared memory.
2. **Register pressure**: More stages may require more registers for bookkeeping.
3. **Startup overhead**: The pipeline prologue and epilogue add overhead proportional to the stage count.
4. **Synchronization cost**: Each stage boundary requires a `__syncthreads()` or barrier arrival.

---

## T.serial -- Serial Loops

### Signature

```python
T.serial(
    start,                      # Start index or range
    stop=None,                  # Stop index (exclusive)
    step=None,                  # Step size
    annotations=None,           # Additional annotations
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | int or range | required | Starting index. If `stop` is `None`, this is treated as `stop` with `start=0`. |
| `stop` | int or None | `None` | Ending index (exclusive). |
| `step` | int or None | `None` | Step size. If `None`, defaults to 1. |
| `annotations` | dict or None | `None` | Compiler annotations for the loop. |

### Overview

`T.serial` creates a loop that executes sequentially within each thread. Unlike `T.Parallel`, every thread executes all iterations of the loop. This is useful when:

- Loop iterations have dependencies on each other.
- The loop body contains operations that must be ordered (e.g., sequential accumulation).
- The iteration count is determined at kernel compile time.

### Basic Usage

```python
# Simple serial loop from 0 to N
for i in T.serial(N):
    buffer[i] = buffer[i] * 2.0

# Serial loop with explicit start and stop
for i in T.serial(start=1, stop=N):
    buffer[i] = buffer[i] + buffer[i-1]

# Serial loop with step
for k in T.serial(start=0, stop=K, step=block_K):
    T.copy(A_global[k:k+block_K], A_smem)
    T.gemm(A_smem, B_smem, C_local)
```

### Serial vs Python for Loop

While a standard Python `for` loop can also be used in TileLang kernels, `T.serial` provides additional benefits:

1. **Compiler annotations**: Can attach optimization hints.
2. **Explicit intent**: Makes it clear the loop is meant to be serial (not accidentally parallelizable).
3. **Range validation**: The compiler can validate loop bounds at compile time.
4. **Integration with passes**: TileLang compiler passes recognize and optimize `T.serial` loops.

```python
# Equivalent to:
for i in T.serial(N):
    body(i)

# But T.serial provides compiler integration:
for i in T.serial(N, annotations={"unroll": True}):
    body(i)
```

### Example: Serial K-Dimension Accumulation

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])
def serial_gemm(M, N, K, block_M=64, block_N=64, block_K=32):
    A_smem = T.alloc_shared([block_M, block_K], "float16")
    B_smem = T.alloc_shared([block_K, block_N], "float16")
    C_local = T.alloc_local([block_M, block_N], "float32")

    T.clear(C_local)

    # Serial K loop: each iteration depends on accumulated C_local
    for k in T.serial(0, K, block_K):
        T.copy(A_global[k:k+block_K], A_smem)
        T.copy(B_global[k:k+block_K], B_smem)
        T.sync_shared_memory()
        T.gemm(A_smem, B_smem, C_local)

    T.copy(C_local, C_global)
    return C_global
```

---

## T.unroll -- Unrolled Loops

### Signature

```python
T.unroll(
    start,                      # Start index or range
    stop=None,                  # Stop index (exclusive)
    step=None,                  # Step size
    explicit=False,             # Whether to force unrolling
    unroll_factor=None,         # Partial unroll factor
    annotations=None,           # Additional annotations
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | int or range | required | Starting index. |
| `stop` | int or None | `None` | Ending index (exclusive). |
| `step` | int or None | `None` | Step size. Defaults to 1. |
| `explicit` | bool | `False` | If `True`, force the compiler to fully unroll the loop regardless of iteration count. |
| `unroll_factor` | int or None | `None` | When specified, unroll only `unroll_factor` iterations at a time (partial unrolling). The remaining iterations form a serial loop. |
| `annotations` | dict or None | `None` | Additional compiler annotations. |

### Overview

Loop unrolling replicates the loop body multiple times to reduce loop overhead (branching, counter increments) and enable instruction-level parallelism. TileLang's `T.unroll` provides explicit control over unrolling behavior.

### Full Unrolling

When `explicit=True` (or when the iteration count is small and the compiler decides to unroll), the loop body is replicated for each iteration:

```python
# Before unrolling:
for i in T.unroll(4, explicit=True):
    result[i] = input[i] * scale

# After unrolling (conceptually):
result[0] = input[0] * scale
result[1] = input[1] * scale
result[2] = input[2] * scale
result[3] = input[3] * scale
```

### Partial Unrolling

When `unroll_factor` is specified, only `unroll_factor` iterations are unrolled at a time:

```python
# Unroll 4 iterations at a time
for i in T.unroll(32, unroll_factor=4):
    result[i] = input[i] * scale

# Conceptually produces:
for i in T.serial(0, 32, 4):
    result[i+0] = input[i+0] * scale
    result[i+1] = input[i+1] * scale
    result[i+2] = input[i+2] * scale
    result[i+3] = input[i+3] * scale
```

### When to Use Unrolling

| Scenario | Recommendation | Rationale |
|----------|---------------|-----------|
| Small fixed-size loops (<=8) | Full unroll | Eliminates loop overhead |
| Medium loops (8-32) | Partial unroll (factor 4-8) | Balance overhead vs code size |
| Large loops (>32) | Do not unroll | Code size explosion, I-cache pressure |
| Memory-bound loops | Unroll factor 2-4 | Hide latency with independent loads |
| Compute-bound loops | Unroll factor 4-8 | Expose instruction-level parallelism |

### Example: Unrolled Reduction

```python
import tilelang.language as T

# Fully unrolled small reduction for maximum throughput
def fast_reduce_8(buffer):
    acc = 0.0
    for i in T.unroll(8, explicit=True):
        acc = acc + buffer[i]
    return acc
```

### Example: Partial Unrolling for GEMM Inner Loop

```python
import tilelang.language as T

# Partially unroll the inner K loop of a GEMM for better ILP
for k in T.serial(0, K, step=4):
    for kk in T.unroll(4, explicit=True):
        a_val = A_smem[i, k + kk]
        b_val = B_smem[k + kk, j]
        acc = acc + a_val * b_val
```

---

## T.Persistent -- Persistent Kernels

### Signature

```python
T.Persistent(
    domain,                     # Total work domain size
    wave_size,                  # Size of each work wave
    index,                      # Index variable for current work item
    group_size=8,               # Thread group size for cooperative fetching
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `domain` | int | required | Total number of work items to process across all waves. |
| `wave_size` | int | required | Number of work items processed per wave (per thread block). |
| `index` | str or Var | required | Variable name or variable to use as the loop index. |
| `group_size` | int | `8` | Number of threads that cooperatively fetch data for a single work item. |

### Overview

Persistent kernels are a pattern where thread blocks stay active for the lifetime of the kernel, continuously pulling work from a shared work queue. This is in contrast to the traditional model where each thread block processes one tile and exits.

Benefits of persistent kernels:

1. **Amortized launch overhead**: Fewer kernel launches for repeated operations.
2. **Dynamic load balancing**: Work is distributed on-demand to available thread blocks.
3. **Work stealing**: Fast thread blocks can take more work items.
4. **Reduced synchronization**: No need for cross-kernel synchronization.

### How Persistent Kernels Work

```
Traditional model:
  Kernel launch -> Each block processes one tile -> Block exits -> Kernel completes

Persistent model:
  Kernel launch -> Blocks loop:
    1. Atomically get next work item from global queue
    2. Process the work item
    3. Check if more work available
    4. If yes, go to step 1
    5. If no, block exits
  -> All blocks exit -> Kernel completes
```

### Example: Persistent GEMM

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])
def persistent_gemm(
    M, N, K, num_tiles,
    block_M=64, block_N=64, block_K=32,
    in_dtype="float16",
):
    A_smem = T.alloc_shared([block_M, block_K], in_dtype)
    B_smem = T.alloc_shared([block_K, block_N], in_dtype)
    C_local = T.alloc_local([block_M, block_N], "float32")

    T.clear(C_local)

    # Persistent loop: keep pulling tiles until all are done
    with T.Persistent(domain=num_tiles, wave_size=1, index="tile_idx", group_size=8):
        # Compute which M, N tile this iteration handles
        m_tile = tile_idx // (N // block_N)
        n_tile = tile_idx % (N // block_N)

        # Load and compute this tile
        for k in T.serial(0, K, block_K):
            T.copy(A_global[m_tile*block_M:(m_tile+1)*block_M, k:k+block_K], A_smem)
            T.copy(B_global[k:k+block_K, n_tile*block_N:(n_tile+1)*block_N], B_smem)
            T.sync_shared_memory()
            T.gemm(A_smem, B_smem, C_local)

        T.copy(C_local, C_global[m_tile*block_M:(m_tile+1)*block_M,
                                   n_tile*block_N:(n_tile+1)*block_N])

        # Clear accumulator for next tile
        T.clear(C_local)

    return C_global
```

### Group Size Parameter

The `group_size` parameter controls how many threads cooperatively participate in fetching data for a single work item. A larger group size:

- Reduces per-thread memory traffic.
- Increases memory coalescing for each fetch.
- May reduce overall parallelism if group_size > available threads per work item.

```python
# Each work item fetched by 8 threads cooperatively
with T.Persistent(domain=total_work, wave_size=1, index="idx", group_size=8):
    # ...

# Each work item fetched by 16 threads (more coalescing, less parallelism)
with T.Persistent(domain=total_work, wave_size=1, index="idx", group_size=16):
    # ...
```

---

## T.Vectorized -- Vectorized Loops

### Signature

```python
T.Vectorized(
    start,                      # Start index or range
    stop=None,                  # Stop index (exclusive)
    annotations=None,           # Additional annotations
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | int or range | required | Starting index. |
| `stop` | int or None | `None` | Ending index (exclusive). |
| `annotations` | dict or None | `None` | Additional compiler annotations. |

### Overview

`T.Vectorized` creates a loop where each iteration is compiled to a vector memory instruction. Instead of loading/storing one element at a time, each iteration loads/stores multiple contiguous elements using instructions like:

- `LDG.128` (load 128 bits = 4 float32 values at once)
- `STG.128` (store 128 bits = 4 float32 values at once)
- `LDG.64` (load 64 bits = 2 float32 values)
- Vector shared memory loads/stores

### When to Use Vectorized Loops

Use `T.Vectorized` when:

1. The loop body consists primarily of memory load/store operations.
2. Memory accesses are to contiguous addresses.
3. The data type width allows vectorization (4-byte or 2-byte types work best).

### Example: Vectorized Memory Copy

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[1])
def vectorized_copy(N, dtype="float32"):
    src = T.alloc_shared([N], dtype)
    dst = T.alloc_shared([N], dtype)

    T.copy(src_global, src)

    # Vectorized loop: each iteration processes 4 elements with a single 128-bit load
    for i in T.Vectorized(0, N, annotations={"vector_width": 4}):
        dst[i] = src[i]

    T.copy(dst, dst_global)
    return dst_global
```

### Example: Vectorized Element-wise Operation

```python
import tilelang.language as T

# Vectorized ReLU
for i in T.Vectorized(0, N):
    output[i] = T.max(input[i], 0.0)

# Vectorized scaling
for i in T.Vectorized(0, N):
    output[i] = input[i] * scale + bias
```

### Vector Width Selection

The vector width is automatically determined based on the data type and memory alignment:

| Data Type | Auto Vector Width | Memory Width |
|-----------|------------------|-------------|
| float32 | 4 | 128 bits |
| float16 | 8 | 128 bits |
| bfloat16 | 8 | 128 bits |
| int8 | 16 | 128 bits |
| float64 | 2 | 128 bits |

---

## Conditionals

### If/Else in TileLang

TileLang supports standard Python `if/else` statements within kernel code. These compile to GPU branching instructions (`BRA`, `SSY`, `PBK`):

```python
# Simple conditional
if x > 0:
    result = x
else:
    result = -x

# Nested conditional
if condition_a:
    if condition_b:
        result = value_a
    else:
        result = value_b
else:
    result = value_c
```

### Per-thread Conditionals

In GPU kernels, conditionals execute per-thread. Different threads can take different branches (divergent branching):

```python
# Each thread evaluates its own condition
with T.Parallel(N) as i:
    if i < threshold:
        output[i] = input[i] * 2.0  # Taken by threads with i < threshold
    else:
        output[i] = input[i] * 0.5  # Taken by threads with i >= threshold
```

### Warp Divergence

When threads within the same warp take different branches, both paths are executed serially with the inactive threads masked off. This is called **warp divergence** and can reduce performance:

```
If threads 0-15 take the "if" branch and threads 16-31 take the "else" branch:
  1. Execute "if" branch with threads 0-15 active, threads 16-31 masked
  2. Execute "else" branch with threads 16-31 active, threads 0-15 masked
  Total: 2x the branch execution time

Best case (all threads agree): no divergence penalty
Worst case (50/50 split): 2x branch execution time
```

### Performance Tips for Conditionals

1. **Minimize divergence within warps**: Structure conditions so threads in the same warp take the same branch when possible.
2. **Use T.max/T.min instead of conditionals**: For simple comparisons, use built-in functions that compile to branchless instructions:

```python
# Instead of:
if x > 0:
    result = x
else:
    result = 0

# Use:
result = T.max(x, 0.0)  # Branchless, faster
```

3. **Use conditional moves**: For small conditional assignments:

```python
# Instead of branching:
# result = condition ? a : b
result = T.select(condition, a, b)
```

---

## T.break and T.continue

### T.break

`T.break` exits the innermost loop immediately:

```python
for i in T.serial(N):
    result = compute(input[i])
    if result > threshold:
        T.break()  # Exit the loop early
    output[i] = result
```

### T.continue

`T.continue` skips the rest of the current loop iteration and proceeds to the next:

```python
for i in T.serial(N):
    if input[i] == 0:
        T.continue()  # Skip zero elements
    output[i] = 1.0 / input[i]
```

### Restrictions

- `T.break` and `T.continue` can only be used inside `T.serial`, `T.unroll`, `T.Pipelined`, and standard Python `for`/`while` loops.
- They **cannot** be used inside `T.Parallel` loops (parallel loops cannot have early termination).
- `T.break` is not supported in `T.Pipelined` loops (pipeline stages must all execute).

### Example: Early Termination Search

```python
import tilelang.language as T

# Find first element greater than threshold
for i in T.serial(N):
    if buffer[i] > threshold:
        result[0] = i
        T.break()
```

---

## While Loops

### Python While in TileLang

Standard Python `while` loops can be used in TileLang kernels. They compile to GPU conditional branch instructions:

```python
# Convergence loop
error = T.alloc_local([1], "float32")
error[0] = 1.0
iteration = 0

while error[0] > tolerance and iteration < max_iterations:
    # Perform one iteration
    result = compute_step(input, result)
    error[0] = compute_error(result, target)
    iteration += 1
```

### While Loop Considerations

1. **All threads must agree on the loop condition**: If threads in the same warp disagree on whether to continue, warp divergence occurs.
2. **Termination guarantee**: The compiler may not be able to verify that the loop terminates. Ensure there is always a convergence condition.
3. **Performance**: While loops with many iterations can be slow due to the per-iteration condition evaluation and potential divergence.

### Example: Iterative Algorithm

```python
import tilelang.language as T

# Simple iterative refinement
x = T.alloc_local([N], "float32")
for i in range(N):
    x[i] = initial_value

residual_norm = 1.0
max_iter = 100
tol = 1e-6

iteration = 0
while residual_norm > tol and iteration < max_iter:
    # Compute: x_new = A @ x + b
    for i in range(N):
        acc = 0.0
        for j in range(N):
            acc = acc + A[i, j] * x[j]
        x[i] = acc + b[i]

    # Compute residual (simplified)
    residual_norm = 0.0
    for i in range(N):
        diff = x[i] - x_prev[i]
        residual_norm = residual_norm + diff * diff

    residual_norm = T.sqrt(residual_norm)
    iteration += 1
```

---

## Loop Annotations

Loop annotations provide hints to the TileLang compiler for optimization. They can be applied to any loop construct:

### Available Annotations

| Annotation | Values | Description |
|-----------|--------|-------------|
| `"unroll"` | `True`, `False` | Hint to unroll the loop |
| `"vectorize"` | `True`, `False` | Hint to vectorize the loop body |
| `"parallel"` | `True`, `False` | Hint that iterations are independent |
| `"pragma_unroll"` | int | Explicit unroll factor |
| `"pragma_no_unroll"` | `True` | Prevent unrolling |
| `"pipeline"` | int | Number of pipeline stages |

### Example: Annotated Loops

```python
# Hint the compiler to unroll this loop
for i in T.serial(N, annotations={"unroll": True}):
    buffer[i] = buffer[i] * scale

# Prevent unrolling
for i in T.serial(N, annotations={"pragma_no_unroll": True}):
    result[i] = complex_computation(input[i])

# Specify pipeline stages
for k in T.serial(K // block_K, annotations={"pipeline": 3}):
    T.copy(A[k], A_smem)
    T.gemm(A_smem, B_smem, C_local)
```

---

## Boundary Handling

When the problem dimensions are not evenly divisible by the tile sizes, boundary handling ensures correct behavior at the edges of the data.

### Common Boundary Scenarios

```python
# Scenario 1: M is not divisible by block_M
# The last tile in the M dimension has fewer rows than block_M
M = 100, block_M = 32
# Tiles: [0:32], [32:64], [64:96], [96:100]  <- last tile is only 4 rows

# Scenario 2: K is not divisible by block_K
# The last K iteration has fewer elements
K = 100, block_K = 32
# Iterations: k=0, k=32, k=64, k=96  <- last iteration only has 4 elements
```

### Boundary Handling Strategies

#### Strategy 1: Predicated Execution (Recommended)

Use conditionals to guard against out-of-bounds access:

```python
import tilelang.language as T

with T.Parallel(block_M, block_N) as i, j:
    # Compute global coordinates
    gi = block_idx_x * block_M + i
    gj = block_idx_y * block_N + j

    # Only compute if within bounds
    if gi < M and gj < N:
        output[gi, gj] = input[gi, gj] * scale
```

#### Strategy 2: Padding

Pad the input to a multiple of the tile size:

```python
# Pad M to the next multiple of block_M
M_padded = ((M + block_M - 1) // block_M) * block_M

# Process the padded dimensions (out-of-bounds reads return 0)
for i in T.serial(M_padded):
    for j in T.serial(N):
        val = input[i, j] if i < M and j < N else 0.0
        output[i, j] = val * scale
```

#### Strategy 3: Masked Operations

Use masks to zero out out-of-bounds results:

```python
with T.Parallel(block_M, block_N) as i, j:
    gi = block_idx_x * block_M + i
    gj = block_idx_y * block_N + j

    # Compute regardless, but mask the result
    result = input[i, j] * scale
    mask = 1.0 if (gi < M and gj < N) else 0.0
    output[i, j] = result * mask
```

### Boundary Handling for GEMM

For GEMM operations, the K-dimension boundary is most critical:

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])
def boundary_safe_gemm(M, N, K, block_M=64, block_N=64, block_K=32):
    A_smem = T.alloc_shared([block_M, block_K], "float16")
    B_smem = T.alloc_shared([block_K, block_N], "float16")
    C_local = T.alloc_local([block_M, block_N], "float32")

    T.clear(C_local)

    for k_start in T.serial(0, K, block_K):
        k_end = min(k_start + block_K, K)
        actual_k = k_end - k_start

        # Clear shared memory to handle partial K tiles
        T.clear(A_smem)
        T.clear(B_smem)

        # Load only the valid portion
        T.copy(A_global[k_start:k_end], A_smem[:actual_k])
        T.copy(B_global[k_start:k_end], B_smem[:actual_k])
        T.sync_shared_memory()

        # GEMM with partial K tile (clear_accum to avoid accumulating garbage)
        if k_start == 0:
            T.gemm(A_smem, B_smem, C_local, clear_accum=True)
        else:
            T.gemm(A_smem, B_smem, C_local)

    T.copy(C_local, C_global)
    return C_global
```

---

## Practical Examples

### Complete Flash Attention with Pipelining

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])
def flash_attention_pipelined(
    seq_len, dim, block_M=64, block_N=64,
    in_dtype="float16", num_stages=2,
):
    Q_smem = T.alloc_shared([block_M, dim], in_dtype)
    K_smem = T.alloc_shared([block_N, dim], in_dtype)
    V_smem = T.alloc_shared([block_N, dim], in_dtype)
    S_local = T.alloc_local([block_M, block_N], "float32")
    O_local = T.alloc_local([block_M, dim], "float32")
    m_prev = T.alloc_local([block_M], "float32")
    l_prev = T.alloc_local([block_M], "float32")

    T.clear(O_local)
    T.clear(m_prev)
    T.clear(l_prev)

    # Load Q once
    T.copy(Q_global, Q_smem)
    T.sync_shared_memory()

    # Pipelined loop over KV blocks
    for n in T.Pipelined(seq_len // block_N, num_stages=num_stages):
        T.copy(K_global[n * block_N:(n+1) * block_N], K_smem)
        T.copy(V_global[n * block_N:(n+1) * block_N], V_smem)
        T.sync_shared_memory()

        # S = Q @ K^T
        T.gemm(Q_smem, K_smem, S_local, transpose_B=True, clear_accum=True)

        # Online softmax update
        m_new = T.alloc_local([block_M], "float32")
        T.reduce_max(S_local, m_new, dim=-1, clear=True)

        # Update running statistics
        correction = T.exp(m_prev - T.max(m_prev, m_new))
        m_prev = T.max(m_prev, m_new)

        # Update output and sum
        l_prev = l_prev * correction
        O_local = O_local * correction

        # Softmax(S) @ V
        for i in T.serial(block_M):
            for j in T.serial(block_N):
                S_local[i, j] = T.exp(S_local[i, j] - m_prev[i])

        exp_sum = T.alloc_local([block_M], "float32")
        T.reduce_sum(S_local, exp_sum, dim=-1, clear=True)
        l_prev = l_prev + exp_sum

        T.gemm(S_local, V_smem, O_local)

    # Normalize
    for i in T.serial(block_M):
        for j in T.serial(dim):
            O_local[i, j] = O_local[i, j] / l_prev[i]

    T.copy(O_local, O_global)
    return O_global
```

### Fused Bias Add and Activation with Vectorized Loop

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[1])
def fused_bias_gelu(M, N, block_M=64, dtype="float16"):
    input_smem = T.alloc_shared([block_M, N], dtype)
    bias_smem = T.alloc_shared([N], dtype)
    output_smem = T.alloc_shared([block_M, N], dtype)

    T.copy(input_global, input_smem)
    T.copy(bias_global, bias_smem)
    T.sync_shared_memory()

    # Parallel over rows, vectorized over columns
    with T.Parallel(block_M) as i:
        for j in T.Vectorized(N):
            x = input_smem[i, j] + bias_smem[j]
            # GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            output_smem[i, j] = T.gelu(x)

    T.copy(output_smem, output_global)
    return output_global
```

### Persistent Kernel for Batched Small GEMMs

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])
def persistent_batched_gemm(
    batch, M, N, K,
    block_M=16, block_N=16, block_K=16,
    in_dtype="float16",
):
    total_tiles = batch  # One tile per batch item
    A_smem = T.alloc_shared([block_M, block_K], in_dtype)
    B_smem = T.alloc_shared([block_K, block_N], in_dtype)
    C_local = T.alloc_local([block_M, block_N], "float32")

    with T.Persistent(domain=total_tiles, wave_size=1, index="tile_idx", group_size=4):
        T.clear(C_local)

        # Get batch index from tile index
        b = tile_idx

        for k in T.serial(0, K, block_K):
            T.copy(A_global[b, k:k+block_K], A_smem)
            T.copy(B_global[b, k:k+block_K], B_smem)
            T.sync_shared_memory()
            T.gemm(A_smem, B_smem, C_local)

        T.copy(C_local, C_global[b])

    return C_global
```

### Grid-Stride Loop Pattern

```python
import tilelang.language as T

# Process more elements than there are threads using grid-stride pattern
total_elements = 10000
stride = grid_size  # Total threads across all blocks

with T.Parallel(1) as _:
    idx = thread_id  # Global thread index
    while idx < total_elements:
        result[idx] = compute(input[idx])
        idx = idx + stride
```

### Nested Parallel and Serial Loops

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[1])
def nested_loop_example(M, N, K, block_M=64, block_N=64, block_K=16):
    # Outer serial loop over K tiles
    for k in T.serial(0, K, block_K):
        # Inner parallel loop over M, N tiles
        with T.Parallel(block_M, block_N, coalesced_width=block_N) as i, j:
            acc = 0.0
            # Innermost unrolled loop for small compute
            for kk in T.unroll(block_K, explicit=True):
                acc = acc + A[i, k + kk] * B[k + kk, j]
            C[i, j] = C[i, j] + acc
```

---

## Summary

| Construct | Execution | Use Case | Key Parameter |
|-----------|-----------|----------|--------------|
| `T.Parallel` | Multi-thread | Data-parallel work | `coalesced_width` |
| `T.Pipelined` | Overlapped | Memory-compute overlap | `num_stages` |
| `T.serial` | Sequential | Ordered operations | `step` |
| `T.unroll` | Unrolled | Small fixed loops | `unroll_factor` |
| `T.Persistent` | Work queue | Dynamic load balancing | `domain`, `wave_size` |
| `T.Vectorized` | Vectorized | Contiguous memory ops | Vector width (auto) |
| `if/else` | Per-thread | Conditional logic | N/A |
| `while` | Per-thread | Dynamic iteration | N/A |
| `T.break` | Per-thread | Early exit | N/A |
| `T.continue` | Per-thread | Skip iteration | N/A |

TileLang's control flow constructs are designed to provide both high-level abstractions (like automatic pipelining and persistent kernels) and low-level control (like explicit unrolling and vectorization). The key to writing efficient kernels is choosing the right construct for each part of the computation: parallel loops for independent work, serial loops for accumulation, pipelining for memory-compute overlap, and unrolling for small inner loops.
