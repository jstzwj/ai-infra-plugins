# TileLang Memory Management

## Table of Contents

- [1. Memory Hierarchy Overview](#1-memory-hierarchy-overview)
- [2. T.alloc_shared](#2-talloc_shared)
- [3. T.alloc_fragment](#3-talloc_fragment)
- [4. T.alloc_local](#4-talloc_local)
- [5. T.alloc_global](#5-talloc_global)
- [6. T.alloc_var](#6-talloc_var)
- [7. T.alloc_barrier](#7-talloc_barrier)
- [8. T.alloc_cluster_barrier](#8-talloc_cluster_barrier)
- [9. T.alloc_tmem](#9-talloc_tmem)
- [10. T.alloc_reducer](#10-talloc_reducer)
- [11. T.alloc_descriptor and Related](#11-talloc_descriptor-and-related)
- [12. T.empty](#12-tempty)
- [13. T.fill and T.clear](#13-tfill-and-tclear)
- [14. Memory Scope Strings](#14-memory-scope-strings)
- [15. Memory Layout Considerations](#15-memory-layout-considerations)
- [16. Shared Memory Bank Conflicts and Swizzling](#16-shared-memory-bank-conflicts-and-swizzling)
- [17. Memory Capacity Planning](#17-memory-capacity-planning)
- [18. Advanced Memory Operations](#18-advanced-memory-operations)

---

## 1. Memory Hierarchy Overview

TileLang exposes the full GPU memory hierarchy through typed buffer allocations. Understanding the memory hierarchy is essential for writing high-performance kernels.

### 1.1 Memory Hierarchy Diagram

```
+-------------------+
| Global Memory     |  <- High latency (~400-800 cycles), large capacity (16-80 GB)
| (HBM/GDDR)       |  <- Accessible by all threads across all blocks
+-------------------+
         |
         v
+-------------------+
| Shared Memory     |  <- Low latency (~20-30 cycles), small capacity (48-228 KB/SM)
| (SRAM on-chip)    |  <- Shared by all threads in a thread block
|                   |  <- User-managed via T.alloc_shared
+-------------------+
         |
         v
+-------------------+
| Tensor Memory     |  <- Blackwell (SM100+) only, 512 cols x 128 rows
| (TMEM)            |  <- Dedicated on-chip memory for TCGEN5 MMA
+-------------------+
         |
         v
+-------------------+
| Register File     |  <- Lowest latency (~1 cycle), per-thread private
| (Registers)       |  <- Managed via T.alloc_fragment / T.alloc_local
|                   |  <- Layout auto-inferred for Tensor Core compatibility
+-------------------+
```

### 1.2 Memory Type Comparison

| Property | Global | Shared | TMEM | Fragment/Local |
|----------|--------|--------|------|----------------|
| Scope | Grid | Block | Warp-group | Thread |
| Latency | High | Low | Low | Lowest |
| Capacity | GB | KB/SM | KB | Bytes/thread |
| Access | All threads | Block threads | TCGEN5 ops | Private |
| Allocation | `T.alloc_global` | `T.alloc_shared` | `T.alloc_tmem` | `T.alloc_fragment` / `T.alloc_local` |
| DMA Support | TMA | N/A | TCGEN5 CP | N/A |
| Persistence | Kernel lifetime | Block lifetime | Allocation scope | Scope lifetime |

### 1.3 Data Flow Pattern

The typical data flow in a TileLang kernel follows this pattern:

```
Global Memory --> Shared Memory --> Register File --> Shared Memory --> Global Memory
     |                |                   |                |                |
  T.copy()      T.alloc_shared()  T.alloc_fragment()  T.copy()      T.copy()
  T.tma_copy()                     T.gemm()
  T.async_copy()
```

---

## 2. T.alloc_shared

### 2.1 Function Signature

```python
def alloc_shared(shape: ShapeType, dtype: DType, scope="shared.dyn") -> Buffer:
```

### 2.2 Description

Allocates a shared memory buffer for inter-thread communication within a thread block. Shared memory is a low-latency, user-managed scratchpad that is shared by all threads in the same CUDA block.

### 2.3 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `tuple` or `list` | Required | Shape of the buffer, e.g., `(128, 32)` |
| `dtype` | `str` | Required | Data type, e.g., `"float16"`, `"float32"` |
| `scope` | `str` | `"shared.dyn"` | Memory scope string |

### 2.4 Return Value

Returns a `tvm.tir.Buffer` object allocated in shared memory scope.

### 2.5 Usage Examples

```python
# 1D shared memory buffer
A_shared = T.alloc_shared((128,), "float32")

# 2D shared memory buffer (most common for GEMM tiles)
A_shared = T.alloc_shared((block_M, block_K), "float16")

# 3D shared memory buffer (for multi-stage pipeline)
A_shared = T.alloc_shared((num_stages, block_M, block_K), "float16")

# Bool buffer (automatically uses "shared" scope instead of "shared.dyn")
flag_buf = T.alloc_shared((128,), "bool")
```

### 2.6 Scope Behavior

- **`"shared.dyn"` (default)**: Dynamic shared memory. TileLang's `merge_smem` pass combines all `shared.dyn` allocations into a single dynamic shared memory region, reducing memory waste. This is the recommended scope for most shared memory allocations.
- **`"shared"`: Static shared memory. The buffer is allocated at a fixed offset. Use this when you need the buffer address to be stable (e.g., for TMA descriptors).

When `dtype="bool"`, the scope is automatically changed from `"shared.dyn"` to `"shared"` because the merge shared memory pass does not support bool type.

### 2.7 Multi-Stage Shared Memory

For software pipelining, you typically allocate shared memory with an extra dimension for the pipeline stages:

```python
num_stages = 3
A_shared = T.alloc_shared((num_stages, block_M, block_K), "float16")
B_shared = T.alloc_shared((num_stages, block_K, block_N), "float16")

for k in T.Pipelined(T.ceildiv(K, BK), num_stages=num_stages):
    stage = k % num_stages
    T.copy(A[by * BM, k * BK], A_shared[stage])
    T.copy(B[k * BK, bx * BN], B_shared[stage])
    T.gemm(A_shared[stage], B_shared[stage], C_local)
```

### 2.8 Shared Memory for Reductions

```python
# Allocate shared memory for reduction scratch space
reduction_buf = T.alloc_shared((num_threads,), "float32")
```

### 2.9 Capacity Considerations

Typical shared memory capacity per SM:

| GPU | Shared Memory per SM |
|-----|---------------------|
| A100 | 164 KB (max configurable) |
| H100 | 228 KB (max configurable) |
| RTX 4090 | 100 KB (max configurable) |

To calculate shared memory usage:

```python
# For a (128, 32) float16 buffer:
bytes = 128 * 32 * 2  # 8,192 bytes = 8 KB
```

---

## 3. T.alloc_fragment

### 3.1 Function Signature

```python
def alloc_fragment(shape: ShapeType, dtype: DType, scope="local.fragment") -> Buffer:
```

### 3.2 Description

Allocates a register file buffer (fragment) for thread-private computation. Fragments are the primary storage for intermediate results, especially for Tensor Core operations. The key feature of `alloc_fragment` is that the compiler automatically infers the optimal data layout for downstream operations (e.g., GEMM).

### 3.3 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `tuple` or `list` | Required | Shape of the buffer |
| `dtype` | `str` | Required | Data type |
| `scope` | `str` | `"local.fragment"` | Always `"local.fragment"` |

### 3.4 Return Value

Returns a `tvm.tir.Buffer` object in the `local.fragment` scope.

### 3.5 Usage Examples

```python
# GEMM accumulator
C_local = T.alloc_fragment((block_M, block_N), "float32")

# Vector fragment for element-wise operations
vec_frag = T.alloc_fragment((128,), "float16")

# Fragment used as reduction scratch
max_frag = T.alloc_fragment((block_N,), "float32")
```

### 3.6 Layout Inference

The TileLang compiler performs layout inference on fragment buffers during the `LayoutInference` pass. When a fragment is used with `T.gemm()`, the compiler assigns a layout that is compatible with the target Tensor Core intrinsic:

- **MMA (SM80)**: The fragment is laid out to match the MMA register distribution across threads.
- **WGMMA (SM90)**: The fragment is laid out for WGMMA accumulator registers.
- **TCGEN05 (SM100)**: The fragment maps to Tensor Memory (TMEM) columns.

You can override the inferred layout with `T.annotate_layout`:

```python
C_local = T.alloc_fragment((block_M, block_N), "float32")
T.annotate_layout({C_local: my_custom_layout})
```

### 3.7 Fragment in GEMM

The most common use of fragments is as GEMM accumulators:

```python
C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
T.clear(C_local)

for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
    T.copy(A[by * BM, k * BK], A_shared)
    T.copy(B[k * BK, bx * BN], B_shared)
    T.gemm(A_shared, B_shared, C_local)  # C_local += A_shared @ B_shared

T.copy(C_local, C[by * BM, bx * BN])
```

### 3.8 Fragment with Reductions

Fragments are used with reduction operations:

```python
# Reduce max along a dimension
max_val = T.alloc_fragment((block_N,), "float32")
T.reduce_max(input_buf, max_val, dim=0)
```

---

## 4. T.alloc_local

### 4.1 Function Signature

```python
def alloc_local(shape: ShapeType, dtype: DType, scope="local") -> Buffer:
```

### 4.2 Description

Allocates thread-local memory. Unlike `alloc_fragment`, the layout is not automatically inferred for Tensor Core operations. Use `alloc_local` for general-purpose thread-private storage that does not participate in GEMM or other Tensor Core operations.

### 4.3 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `tuple` or `list` | Required | Shape of the buffer |
| `dtype` | `str` | Required | Data type |
| `scope` | `str` | `"local"` | Memory scope |

### 4.4 Usage Examples

```python
# Thread-local scratch buffer
scratch = T.alloc_local((16,), "float32")

# Local index buffer
indices = T.alloc_local((64,), "int32")
```

### 4.5 When to Use alloc_local vs alloc_fragment

| Use Case | Use | Reason |
|----------|-----|--------|
| GEMM accumulator | `alloc_fragment` | Layout inference for Tensor Core |
| Reduction scratch | `alloc_fragment` | Required by T.reduce_* |
| Element-wise temp | Either | No layout dependency |
| Index/compute scratch | `alloc_local` | No Tensor Core involvement |
| Loop-carried value | `alloc_var` | Scalar value |

---

## 5. T.alloc_global

### 5.1 Function Signature

```python
def alloc_global(shape: ShapeType, dtype: DType, scope="global") -> Buffer:
```

### 5.2 Description

Allocates a global memory buffer as a workspace. This allocation bypasses the PyTorch memory allocator and uses direct CUDA allocation (`cudaMalloc` or equivalent).

### 5.3 Important Notes

- Memory allocated with `T.alloc_global` does not go through the PyTorch allocator. It is allocated directly by the backend API.
- This API is primarily for testing purposes and specific use cases.
- For production code, prefer allocating workspace tensors on the Python/host side and passing them as kernel arguments.
- This API may not be available in all backends (e.g., CuteDSL).

### 5.4 Usage Examples

```python
# Allocate a global workspace
workspace = T.alloc_global((M, N), "float32")

# Use as intermediate buffer between two kernel launches
with T.Kernel(...) as bx:
    T.copy(A, workspace)
    # ... first kernel writes to workspace

with T.Kernel(...) as bx:
    T.copy(workspace, B)  # ... second kernel reads from workspace
```

---

## 6. T.alloc_var

### 6.1 Function Signature

```python
# Multiple overloads:
def alloc_var(dtype: DType, init: PrimExpr | int | float, scope: str = "local.var") -> Buffer:
def alloc_var(dtype: DType, scope: str = "local.var", *, init: PrimExpr | int | float | None = None) -> Buffer:
def alloc_var(dtype: DType, *args, scope: str = "local.var", init: PrimExpr | int | float | None = None) -> Buffer:
```

### 6.2 Description

Allocates a single-element variable buffer for scalar storage. This is the TileLang equivalent of declaring a local scalar variable. The returned buffer has shape `[1]` and is accessed via `[0]`.

### 6.3 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dtype` | `str` | Required | Data type of the variable |
| `init` | `int`, `float`, or `PrimExpr` | `None` | Optional initializer value |
| `scope` | `str` | `"local.var"` | Memory scope |

### 6.4 Usage Examples

```python
# Uninitialized variable
count = T.alloc_var("int32")
count[0] = 0  # Must initialize manually

# Variable with initializer
sum = T.alloc_var("float32", 0.0)    # Initialized to 0.0
max_val = T.alloc_var("float32", -1e30)  # Initialized to -inf for max tracking
idx = T.alloc_var("int32", 0)         # Initialized to 0

# All equivalent forms:
a = T.alloc_var('int32', 1)                    # var with init 1
a = T.alloc_var('int32', 'local.var')          # var with local.var scope, no init
a = T.alloc_var('int32', 1, 'local.var')       # var with init 1 and local.var scope
a = T.alloc_var('int32', 'local.var', init=1)  # var with init 1 and local.var scope
a = T.alloc_var('int32', init=1)               # var with init 1 and default scope

# Reading and writing
count[0] = count[0] + 1
current_val = sum[0]
```

### 6.5 Use Cases

```python
# Loop counter
i = T.alloc_var("int32", 0)

# Accumulator for scalar reduction
total = T.alloc_var("float32", 0.0)
for k in T.serial(K):
    total[0] += A[k]

# Condition flag
found = T.alloc_var("bool", 0)

# Pointer offset
offset = T.alloc_var("int32", bx * stride)
```

### 6.6 Overload Resolution

The overloads handle different calling conventions to support both positional and keyword arguments:

| Call Pattern | Interpretation |
|-------------|---------------|
| `alloc_var("int32")` | No init, default scope |
| `alloc_var("int32", 1)` | init=1, default scope |
| `alloc_var("int32", "local.var")` | No init, scope="local.var" |
| `alloc_var("int32", 1, "local.var")` | init=1, scope="local.var" |
| `alloc_var("int32", init=1)` | init=1, default scope |

---

## 7. T.alloc_barrier

### 7.1 Function Signature

```python
def alloc_barrier(arrive_count: int | list[int]) -> Buffer:
```

### 7.2 Description

Allocates a memory barrier (mbarrier) object in shared memory. Mbarriers are used for asynchronous synchronization, especially with TMA and cp.async operations. Each barrier tracks the arrival of a specified number of threads.

### 7.3 Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `arrive_count` | `int` or `list[int]` | Number of threads that must arrive at each barrier. If a list, allocates multiple barriers with individual arrive counts. |

### 7.4 Return Value

Returns a `tvm.tir.Buffer` of dtype `uint64` and shape `(num_barriers,)` in `shared.barrier` scope.

### 7.5 Usage Examples

```python
# Single barrier for all 128 threads
barrier = T.alloc_barrier(128)

# Multiple barriers (e.g., for multi-stage pipeline)
# 4 barriers, each expecting 128 thread arrivals
barriers = T.alloc_barrier([128, 128, 128, 128])

# Or equivalently:
barriers = T.alloc_barrier([128] * 4)

# Access individual barrier
barrier_0 = barrier[0]  # First barrier
barrier_k = barriers[k]  # k-th barrier
```

### 7.6 Barrier Operations

```python
# Wait on barrier with parity
T.mbarrier_wait_parity(barrier[0], parity=0)

# Arrive at barrier (signal completion)
T.barrier_arrive(barrier[0])

# Set expected transaction count (for TMA)
T.mbarrier_expect_tx(barrier[0], tx_bytes)

# Arrive and expect transaction count in one operation
T.mbarrier_arrive_expect_tx(barrier[0], tx_bytes)
```

### 7.7 Typical Pipeline Pattern

```python
with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
    A_shared = T.alloc_shared((2, BM, BK), "float16")
    B_shared = T.alloc_shared((2, BK, BN), "float16")
    C_local = T.alloc_fragment((BM, BN), "float32")

    # Producer-consumer barriers
    producer_bar = T.alloc_barrier([1] * 2)  # 1 producer warp
    consumer_bar = T.alloc_barrier([1] * 2)  # 1 consumer warp

    T.clear(C_local)

    for k in T.serial(T.ceildiv(K, BK)):
        # Producer waits for consumer to finish previous stage
        T.mbarrier_wait_parity(consumer_bar[k % 2], k % 2)

        # Producer loads data
        T.copy(A[by * BM, k * BK], A_shared[k % 2])
        T.copy(B[k * BK, bx * BN], B_shared[k % 2])

        # Producer signals data is ready
        T.barrier_arrive(producer_bar[k % 2])

        # Consumer waits for producer
        T.mbarrier_wait_parity(producer_bar[k % 2], k % 2)

        # Consumer computes
        T.gemm(A_shared[k % 2], B_shared[k % 2], C_local)

        # Consumer signals completion
        T.barrier_arrive(consumer_bar[k % 2])

    T.copy(C_local, C[by * BM, bx * BN])
```

---

## 8. T.alloc_cluster_barrier

### 8.1 Function Signature

```python
def alloc_cluster_barrier(arrive_count: int | list[int]) -> Buffer:
```

### 8.2 Description

Allocates a cluster-level barrier for synchronization across CTAs within a thread cluster. Only available on SM90+ (compute capability 9.0 and later).

### 8.3 Parameters

Same as `T.alloc_barrier`.

### 8.4 Usage Examples

```python
with T.Kernel(grid_x, grid_y, threads=128,
              cluster_dims=(2, 1, 1)) as (bx, by):
    # Cluster barrier for 2 CTAs
    cluster_bar = T.alloc_cluster_barrier(2)

    # Arrive at cluster barrier
    T.barrier_arrive(cluster_bar[0])

    # Arrive at peer CTA's barrier
    T.barrier_arrive(cluster_bar[0], cta_id=1)
```

### 8.5 Scope

Cluster barriers are allocated in `shared.cluster_barrier` scope, which is visible to all CTAs in the cluster.

---

## 9. T.alloc_tmem

### 9.1 Function Signature

```python
def alloc_tmem(shape: ShapeType, dtype: DType) -> Buffer:
```

### 9.2 Description

Allocates Tensor Memory (TMEM) for use with 5th-generation Tensor Core operations (TCGEN5.MMA) on Blackwell GPUs (SM100+). TMEM is a dedicated on-chip memory designed to reduce register pressure and enable asynchronous, single-threaded MMA operations.

### 9.3 TMEM Properties

- **Organization**: 512 columns x 128 rows (lanes), each cell is 32 bits.
- **Allocation unit**: Columns, allocated in powers of 2, minimum 32 columns.
- **Deallocation**: Automatic at the end of the allocation block, or manual via `T.deallocate_tmem`.
- **Access**: Only TCGEN5.MMA and specific TMEM load/store instructions can access TMEM.
- **Thread requirement**: Both allocation and deallocation must be performed by the same warp.

### 9.4 Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple` | Must be 2D, e.g., `(128, 64)`. The second dimension represents the number of TMEM columns. |
| `dtype` | `str` | Data type of each TMEM cell. |

### 9.5 Constraints

- `shape` must be 2D.
- The number of columns (second dimension) must be a power of 2 and at least 32.
- TMEM is only available on Blackwell (SM100+) and later architectures.

### 9.6 Usage Examples

```python
# Allocate TMEM for MMA accumulator
C_tmem = T.alloc_tmem((128, 128), "float32")

# Use as TCGEN05 MMA accumulator
T.tcgen05_gemm(A_smem, B_smem, C_tmem, mbar=mbar)

# Copy from TMEM to shared/global memory
T.copy(C_tmem, C_global[...])

# Manual deallocation (optional; auto-deallocated at scope exit)
T.deallocate_tmem(C_tmem)
```

### 9.7 TMEM Scale Factors

For block-scaled GEMM, scale factors are also stored in TMEM:

```python
# Scale factor TMEM buffers
SFA_tmem = T.alloc_tmem((...), "uint32")
SFB_tmem = T.alloc_tmem((...), "uint32")

# Block-scaled GEMM
T.tcgen05_gemm_blockscaled(
    A_smem, B_smem, C_tmem,
    SFA_tmem, SFB_tmem,
    mbar=mbar,
)
```

### 9.8 T.deallocate_tmem

```python
def deallocate_tmem(tmem: tir.Buffer) -> None:
```

Explicitly deallocates a TMEM buffer. By default, TileLang inserts automatic deallocation at the end of the allocation block. Calling `T.deallocate_tmem` suppresses the automatic deallocation and performs it at the call site instead.

```python
C_tmem = T.alloc_tmem((128, 128), "float32")
# ... use C_tmem ...
T.deallocate_tmem(C_tmem)  # Manual deallocation
```

---

## 10. T.alloc_reducer

### 10.1 Function Signature

```python
def alloc_reducer(shape: ShapeType, dtype: DType, op: ReducerOp = "sum", replication=None) -> Buffer:
```

### 10.2 Description

Allocates a reducer buffer for parallel reduction operations. The reducer maintains thread-private partial results that are combined during `T.finalize_reducer`.

### 10.3 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `tuple` | Required | Shape of the reduction output |
| `dtype` | `str` | Required | Data type |
| `op` | `str` | `"sum"` | Reduction operation: `"sum"`, `"max"`, or `"min"` |
| `replication` | `str` | `"none"` | Replication strategy: `"all"` or `"none"` |

### 10.4 Reduction Operations

| `op` value | Initial value (via T.fill) | Update pattern |
|------------|---------------------------|----------------|
| `"sum"` | 0 | `reducer[...] += ...` |
| `"max"` | `T.min_value(dtype)` | `reducer[...] = T.max(reducer[...], ...)` |
| `"min"` | `T.max_value(dtype)` | `reducer[...] = T.min(reducer[...], ...)` |

### 10.5 Usage Examples

```python
# Allocate a sum reducer
reducer = T.alloc_reducer((block_N,), "float32", op="sum")

# Must fill with proper initializer before reduction
T.fill(reducer, 0)  # 0 for sum

# Use in parallel loop
for i in T.Parallel(block_M):
    for j in T.Parallel(block_N):
        reducer[j] += A[i, j]

# Finalize: combine partial results from all threads
T.finalize_reducer(reducer)

# reducer now contains the final reduced values
```

```python
# Max reducer
max_reducer = T.alloc_reducer((block_N,), "float32", op="max")
T.fill(max_reducer, T.min_value("float32"))  # -inf for max

for i in T.Parallel(block_M):
    for j in T.Parallel(block_N):
        max_reducer[j] = T.max(max_reducer[j], A[i, j])

T.finalize_reducer(max_reducer)
```

### 10.6 Replication

- `"none"`: The compiler decides the replication strategy.
- `"all"`: Replicate the reducer across all threads for maximum parallelism.

### 10.7 finalize_reducer

```python
def finalize_reducer(reducer: tir.Buffer, batch: int = 1) -> tir.PrimExpr:
```

Finalizes a reducer buffer by combining thread-private partial results into the final output. Must be called after all updates to the reducer are complete.

Parameters:
- `reducer`: The reducer buffer to finalize.
- `batch`: Number of output elements per batched AllReduce call (default 1).

---

## 11. T.alloc_descriptor and Related

### 11.1 T.alloc_descriptor

```python
def alloc_descriptor(kind: DescKind = "wgmma", dtype: DType = "uint64") -> Buffer:
```

Allocates a descriptor buffer for hardware-specific memory access patterns. Descriptors encode memory layout information for Tensor Core operations.

Parameters:
- `kind`: Descriptor type: `"wgmma"`, `"tcgen05_smem"`, or `"tcgen05_instr"`.
- `dtype`: Data type of the descriptor (default: `"uint64"`).

### 11.2 T.alloc_wgmma_desc

```python
def alloc_wgmma_desc(dtype: DType = "uint64") -> Buffer:
```

Convenience function for allocating a WGMMA (Hopper SM90) descriptor. Equivalent to `alloc_descriptor("wgmma")`.

```python
desc_a = T.alloc_wgmma_desc()
desc_b = T.alloc_wgmma_desc()

# Initialize with buffer address and layout info
T.initialize_wgmma_descriptor(
    desc_a,
    T.address_of(A_shared[0, 0]),
    layout_type_=0,
    leading_byte_offset=0,
    stride_byte_offset=0,
)
```

### 11.3 T.alloc_tcgen05_smem_desc

```python
def alloc_tcgen05_smem_desc(dtype: DType = "uint64") -> Buffer:
```

Allocates a TCGEN05 (Blackwell SM100) shared memory descriptor. Equivalent to `alloc_descriptor("tcgen05_smem")`.

```python
desc = T.alloc_tcgen05_smem_desc()

T.initialize_tcgen05_descriptor(
    desc,
    start_address=T.address_of(smem[0, 0]),
    leading_byte_offset=0,
    stride_byte_offset=0,
    base_offset=0,
    leading_is_absolute=False,
    swizzle_mode=0,
)
```

### 11.4 T.alloc_tcgen05_instr_desc

```python
def alloc_tcgen05_instr_desc(dtype: DType = "uint32") -> Buffer:
```

Allocates a TCGEN05 instruction descriptor. Equivalent to `alloc_descriptor("tcgen05_instr")`.

### 11.5 Descriptor Initialization

#### WGMMA Descriptor

```python
T.initialize_wgmma_descriptor(
    descriptor,         # Buffer from alloc_wgmma_desc()
    start_address,      # Address of the shared memory buffer
    layout_type_=0,     # Layout type (0 = default)
    leading_byte_offset=0,  # Byte offset between rows
    stride_byte_offset=0,   # Byte stride
)
```

#### TCGEN05 Descriptor

```python
T.initialize_tcgen05_descriptor(
    descriptor,             # Buffer from alloc_tcgen05_smem_desc()
    start_address,          # Address of shared memory
    leading_byte_offset,    # Leading dimension byte offset
    stride_byte_offset,     # Stride dimension byte offset
    base_offset=0,          # Base offset
    leading_is_absolute=False,  # Whether leading is absolute
    swizzle_mode=0,         # Swizzle mode
)
```

### 11.6 Descriptor Offset Increment

```python
# Increase the offset of a descriptor by a specified amount
T.increase_descriptor_offset(descriptor, offset)
```

---

## 12. T.empty

### 12.1 Function Signature

```python
def empty(shape, dtype: DType = "float32") -> Tensor:
```

### 12.2 Description

Declares an output tensor in eager mode. Tensors created with `T.empty` are returned as the function's output when the kernel is executed.

### 12.3 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `tuple` or positional `int`s | Required | Shape of the output tensor |
| `dtype` | `str` | `"float32"` | Data type of the output |

### 12.4 Usage Examples

```python
# Single output
C = T.empty([M, N], "float16")
return C

# Multiple outputs
C = T.empty([M, N], "float16")
D = T.empty([M, N], "float16")
return C, D

# Shape as positional arguments
C = T.empty(M, N, dtype="float16")
```

### 12.5 Eager Mode vs Lazy Mode Output

| Mode | Output Declaration | Return |
|------|-------------------|--------|
| Eager | `C = T.empty(...)` | Return `C` directly |
| Lazy | `C: T.Tensor(...)` as parameter | Use `out_idx=[-1]` in `@tilelang.jit` |

---

## 13. T.fill and T.clear

### 13.1 T.fill

```python
def fill(buffer: BufferLikeType, value: tir.PrimExpr) -> tir.PrimExpr:
```

Fills a buffer or buffer region with a specified value.

```python
# Fill entire buffer with a value
T.fill(C_local, 0.0)
T.fill(mask, 1)

# Fill a sub-region
T.fill(C_local[0:64, :], 0.0)
```

### 13.2 T.clear

```python
def clear(buffer: BufferLikeType) -> tir.PrimExpr:
```

Clears a buffer by filling it with zeros. Equivalent to `T.fill(buffer, 0)`.

```python
# Clear the accumulator
T.clear(C_local)
```

### 13.3 Use with Reducers

```python
reducer = T.alloc_reducer((N,), "float32", op="sum")

# Must fill before reduction begins
T.fill(reducer, 0)  # 0 for sum operation

# Perform reductions
for i in T.Parallel(M):
    reducer[0] += A[i]

# Finalize
T.finalize_reducer(reducer)
```

---

## 14. Memory Scope Strings

TileLang uses scope strings to identify memory regions:

| Scope String | Memory Type | Description |
|-------------|-------------|-------------|
| `"global"` | Global memory | HBM/GDDR, accessible by all threads |
| `"shared"` | Static shared memory | Fixed-offset shared memory |
| `"shared.dyn"` | Dynamic shared memory | Merged dynamic shared memory |
| `"shared.barrier"` | Barrier memory | Mbarrier objects in shared memory |
| `"shared.cluster_barrier"` | Cluster barrier | Cluster-level barriers (SM90+) |
| `"shared.tmem"` | Tensor Memory | Blackwell TMEM |
| `"local"` | Local memory | Thread-private memory |
| `"local.fragment"` | Fragment | Register file with layout inference |
| `"local.var"` | Variable | Single-element scalar variable |
| `"local.descriptor.wgmma"` | WGMMA descriptor | Hopper WGMMA descriptor |
| `"local.descriptor.tcgen05_smem"` | TCGEN05 smem desc | Blackwell shared memory descriptor |
| `"local.descriptor.tcgen05_instr"` | TCGEN05 instr desc | Blackwell instruction descriptor |

---

## 15. Memory Layout Considerations

### 15.1 Automatic Layout Inference

TileLang automatically infers optimal layouts for `alloc_fragment` buffers based on their usage:

- **GEMM accumulator**: Layout matches the Tensor Core register distribution for the target architecture.
- **Copy destination**: Layout may be optimized for the copy source pattern.
- **Reduction buffer**: Layout is determined by the reduction dimension.

### 15.2 Manual Layout Annotation

Override automatic inference with `T.annotate_layout`:

```python
from tilelang.layout import Layout, Fragment

C_local = T.alloc_fragment((block_M, block_N), "float32")

# Define a custom layout
custom_layout = Fragment(
    (block_M, block_N),
    lambda i, j: ...  # Layout function
)

T.annotate_layout({C_local: custom_layout})
```

### 15.3 Layout Propagation

Layouts propagate through operations:

1. `T.copy(global, shared)` -- Shared memory layout may be inferred from global access pattern
2. `T.gemm(shared_a, shared_b, fragment_c)` -- Fragment C layout is inferred from the GEMM intrinsic
3. `T.copy(fragment, global)` -- Global store pattern is determined by fragment layout

---

## 16. Shared Memory Bank Conflicts and Swizzling

### 16.1 Bank Conflict Overview

Shared memory is organized into 32 banks (on NVIDIA GPUs). When multiple threads access the same bank simultaneously, bank conflicts occur, serializing the accesses and degrading performance.

### 16.2 Automatic Swizzling

TileLang can automatically apply swizzle patterns to shared memory to avoid bank conflicts. This is triggered by the `T.use_swizzle` annotation:

```python
T.use_swizzle(panel_size=10)
```

This applies a 2D swizzle pattern to the kernel grid for better L2 cache locality.

### 16.3 Manual Swizzle via Layout

For fine-grained control, use the `tilelang.layout.swizzle` module:

```python
from tilelang.layout import swizzle

# Create a swizzled shared memory layout
swizzled_layout = swizzle.create_swizzle_layout(...)
```

### 16.4 Bank Conflict Avoidance Strategies

| Strategy | Description |
|----------|-------------|
| Padding | Add extra columns to shared memory arrays to break bank alignment |
| Swizzling | Interleave data using XOR patterns to distribute across banks |
| Layout annotation | Use `T.annotate_layout` to specify conflict-free layouts |
| Compiler automatic | TileLang's layout inference may handle common cases |

---

## 17. Memory Capacity Planning

### 17.1 Shared Memory Budget

Calculate shared memory usage per block:

```python
# Example: FP16 GEMM with 3-stage pipeline
block_M, block_N, block_K = 128, 128, 32
num_stages = 3
bytes_per_elem = 2  # float16

# Per stage:
A_per_stage = block_M * block_K * bytes_per_elem  # 128 * 32 * 2 = 8,192 bytes
B_per_stage = block_K * block_N * bytes_per_elem  # 32 * 128 * 2 = 8,192 bytes

# Total with stages:
total_shared = (A_per_stage + B_per_stage) * num_stages
# = 16,384 * 3 = 49,152 bytes = 48 KB
```

### 17.2 Register Budget

Fragment allocations consume registers. Each thread has a limited register file (typically 255 registers of 32 bits each):

```python
# Example: FP32 accumulator for 128x128 GEMM
block_M, block_N = 128, 128
accum_dtype = "float32"
bytes_per_elem = 4

# Total elements in accumulator
total_elems = block_M * block_N  # 16,384

# Registers per thread (for 128 threads):
regs_per_thread = total_elems * bytes_per_elem // 4 // 128
# = 16,384 * 4 / 4 / 128 = 128 registers per thread
```

### 17.3 Occupancy Impact

Memory usage directly affects occupancy (number of concurrent blocks per SM):

```python
# Use T.annotate_min_blocks_per_sm to control the trade-off
T.annotate_min_blocks_per_sm(2)  # At least 2 blocks per SM
```

Higher occupancy requires:
- Less shared memory per block
- Fewer registers per thread
- Smaller block sizes

---

## 18. Advanced Memory Operations

### 18.1 T.address_of / T.access_ptr

```python
# Get address of a buffer element
addr = T.address_of(A_shared[i, j])

# Create an access pointer with read/write intent
ptr_r = T.access_ptr(A, "r")                # Read pointer to entire buffer
ptr_w = T.access_ptr(A[i, j], "rw", M, N)  # Read-write pointer with extents
ptr_r_offset = T.access_ptr(A, "r", offset=10)  # With offset
```

### 18.2 T.make_tensor / T.make_tensor_from_addr

```python
# Create a tensor from an existing pointer
new_buf = T.make_tensor(ptr_var, (128, 64), "float16")

# Create from address expression
new_buf = T.make_tensor_from_addr(addr_expr, (128, 64), "float16")
```

### 18.3 T.reshape / T.view

```python
# Reshape a buffer (same data, different shape)
reshaped = T.reshape(buf, (new_M, new_N))

# View with different dtype (bit reinterpretation)
viewed = T.view(buf, shape=(M, N * 2), dtype="int32")
```

### 18.4 Fence Operations

```python
# Fence for async proxy operations (TMA stores)
T.fence_proxy_async()

# TMA store arrive
T.tma_store_arrive()

# TMA store wait
T.tma_store_wait(count=0)
```

### 18.5 Register Control

```python
# Adjust register allocation at runtime
T.inc_max_nreg(232)   # Increase register pool
T.dec_max_nreg(232)   # Decrease register pool
```
