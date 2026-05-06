# TileLang Data Movement

## Table of Contents

- [1. Overview of Data Movement in TileLang](#1-overview-of-data-movement-in-tilelang)
- [2. T.copy](#2-tcopy)
- [3. T.async_copy](#3-tasync_copy)
- [4. T.tma_copy](#4-ttma_copy)
- [5. T.transpose](#5-ttranspose)
- [6. T.c2d_im2col](#6-tc2d_im2col)
- [7. Copy Optimization](#7-copy-optimization)
- [8. Eviction Policies](#8-eviction-policies)
- [9. TMA Descriptors and Tensor Maps](#9-tma-descriptors-and-tensor-maps)
- [10. Memory Access Patterns](#10-memory-access-patterns)
- [11. Copy Between Different Memory Scopes](#11-copy-between-different-memory-scopes)
- [12. Vectorized Loads and Stores](#12-vectorized-loads-and-stores)
- [13. Copy with Layout Hints](#13-copy-with-layout-hints)
- [14. Advanced Copy Patterns](#14-advanced-copy-patterns)

---

## 1. Overview of Data Movement in TileLang

Data movement is often the primary bottleneck in GPU kernels. TileLang provides a hierarchy of copy operations that automatically select the optimal hardware mechanism based on the source/destination memory scopes and target architecture.

### 1.1 Copy Operation Hierarchy

```
T.copy (highest level)
  |-- Auto-dispatches to:
  |     |-- TMA (Tensor Memory Access) -- SM90+ global -> shared
  |     |-- cp.async (async copy)       -- SM80+ global -> shared
  |     |-- LDSM/STSM (load/store matrix) -- shared <-> fragment for Tensor Core
  |     |-- SIMT (scalar loop)           -- fallback for all targets
  |
T.async_copy (mid level)
  |-- Explicitly requests cp.async path
  |
T.tma_copy (lowest level)
  |-- Explicit TMA with user-managed synchronization
```

### 1.2 Data Flow Patterns

```
+------------------+                    +------------------+
|  Global Memory   | <--- T.copy() ---> |  Shared Memory   |
|  (HBM)           | <--- T.async_copy  |  (SRAM)          |
|                  | <--- T.tma_copy()  |                  |
+------------------+                    +------------------+
        |                                       |
        | T.ldg32/64/128/256                    | T.copy() / T.gemm()
        | T.stg32/64/128/256                    | LDSM / STSM
        v                                       v
+------------------+                    +------------------+
|  Global Memory   |                    | Register File    |
|  (direct access) |                    | (Fragment)       |
+------------------+                    +------------------+
```

---

## 2. T.copy

### 2.1 Function Signature

```python
def copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    coalesced_width: int | None = None,
    disable_tma: bool = False,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    annotations: dict | None = None,
    loop_layout: Any | None = None,
) -> tir.PrimExpr | tir.Stmt:
```

### 2.2 Description

`T.copy` is the primary data movement operation in TileLang. It automatically selects the optimal copy mechanism based on:
- Source and destination memory scopes
- Data shapes and alignment
- Target GPU architecture
- Whether TMA is available

### 2.3 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `src` | `Buffer`, `BufferLoad`, `BufferRegion` | Required | Source memory region |
| `dst` | `Buffer`, `BufferLoad`, `BufferRegion` | Required | Destination memory region |
| `coalesced_width` | `int` or `None` | `None` | Width for coalesced memory access |
| `disable_tma` | `bool` | `False` | Whether to disable TMA acceleration |
| `eviction_policy` | `str` or `None` | `None` | Cache eviction policy |
| `annotations` | `dict` or `None` | `None` | Additional annotations |
| `loop_layout` | `Fragment` or `None` | `None` | Parallel loop layout hint |

### 2.4 Source/Destination Types

`T.copy` accepts multiple argument types:

| Type | Description | Example |
|------|-------------|---------|
| `tir.Buffer` | Entire buffer | `T.copy(A, A_shared)` |
| `BufferLoad` | Specific element | `T.copy(A[i, j], val)` |
| `BufferRegion` | Sub-region | `T.copy(A[i:i+M, j:j+N], A_shared)` |

#### Implicit Region Selection

When you write `T.copy(A[by * BM, k * BK], A_shared)`, TileLang infers the region from the destination shape:
- `A_shared` has shape `(BM, BK)`
- So `A[by * BM, k * BK]` selects the `(BM, BK)` region starting at `(by * BM, k * BK)`

### 2.5 Automatic Dispatch

The copy backend is automatically selected:

| Source Scope | Dest Scope | SM80 (Ampere) | SM90 (Hopper) | SM100 (Blackwell) |
|-------------|-----------|---------------|---------------|-------------------|
| `global` | `shared.dyn` | cp.async | TMA (or cp.async) | TMA |
| `global` | `local.fragment` | SIMT | SIMT | SIMT |
| `shared` | `local.fragment` | LDSM | LDSM | LDSM |
| `local.fragment` | `shared` | STSM | STSM | STSM |
| `shared` | `global` | SIMT | TMA store | TMA store |
| `local.fragment` | `global` | SIMT | SIMT | SIMT |
| `shared` | `shared` | SIMT | SIMT | SIMT |
| `local.fragment` | `local.fragment` | SIMT | SIMT | SIMT |

### 2.6 Usage Examples

#### Basic Global to Shared Copy

```python
# Load a tile of A from global to shared memory
T.copy(A[by * BM, k * BK], A_shared)
```

#### Basic Shared to Global Copy (Store)

```python
# Store result from registers to global memory
T.copy(C_local, C[by * BM, bx * BN])
```

#### Buffer-to-Buffer Copy

```python
# Copy entire buffers (shapes must match)
T.copy(src_shared, dst_shared)
```

#### Copy with Coalesced Width

```python
# Hint for coalesced memory access pattern
T.copy(A[by * BM, k * BK], A_shared, coalesced_width=16)
```

The `coalesced_width` parameter controls how many consecutive elements are accessed by each thread in the inner dimension, improving memory coalescing.

#### Copy with Eviction Policy

```python
# Specify cache eviction policy
T.copy(A[by * BM, k * BK], A_shared,
       eviction_policy="evict_last")  # Keep data in L2 cache
```

#### Copy with TMA Disabled

```python
# Force non-TMA copy path (useful for debugging)
T.copy(A[by * BM, k * BK], A_shared, disable_tma=True)
```

#### Scalar Copy

```python
# When both src and dst are scalar BufferLoad, lowers to a simple store
T.copy(A[i], B[j])  # Equivalent to B[j] = A[i]
```

#### Copy with Layout Hint

```python
# Attach a parallel loop layout hint for the SIMT copy
T.copy(A_global, A_shared, loop_layout=my_fragment_layout)
```

### 2.7 Region Extent Rules

- Normally, source and destination extents must match.
- If one side has extent 1 in a dimension, limited broadcasting-like behavior is applied (syntactic sugar, not general broadcasting).
- If extents differ and neither is 1, the copy uses internal rules to select one side as the base range, which may produce unexpected results.

### 2.8 Annotations Dict

The `annotations` parameter accepts a dictionary that can override individual parameters:

```python
T.copy(A, B, annotations={
    "coalesced_width": 16,
    "disable_tma": False,
    "eviction_policy": 2,  # 0=normal, 1=first, 2=last
    "parallel_loop_layout": my_layout,
})
```

Values in `annotations` take precedence over individual keyword arguments.

---

## 3. T.async_copy

### 3.1 Function Signature

```python
def async_copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    coalesced_width: int | None = None,
    annotations: dict | None = None,
    loop_layout: Any | None = None,
) -> tir.PrimExpr | tir.Stmt:
```

### 3.2 Description

Explicit asynchronous copy primitive that lowers through `cp.async`. This operation issues an asynchronous memory copy from global to shared memory without waiting for completion.

### 3.3 Key Differences from T.copy

| Feature | T.copy | T.async_copy |
|---------|--------|-------------|
| Synchronization | Implicit wait after copy | No wait (user manages sync) |
| Backend | Auto-dispatch | Always cp.async |
| Direction | Any | Primarily global -> shared |
| Pipeline compatible | Yes (when in T.Pipelined) | Manual scheduling |

### 3.4 Asynchronous Copy Pattern

```python
with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
    A_shared = T.alloc_shared((2, BM, BK), "float16")
    B_shared = T.alloc_shared((2, BK, BN), "float16")
    C_local = T.alloc_fragment((BM, BN), "float32")

    T.clear(C_local)

    for k in T.serial(T.ceildiv(K, BK)):
        stage = k % 2

        # Issue async copies (don't wait)
        T.async_copy(A[by * BM, k * BK], A_shared[stage])
        T.async_copy(B[k * BK, bx * BN], B_shared[stage])

        # Commit the async group
        T.cp_async_barrier_noinc(barrier)

        # Wait for previous async copies to complete
        # (synchronize before using the data)
        T.sync_threads()

        # Compute using data from the previous iteration
        prev_stage = (k - 1) % 2
        T.gemm(A_shared[prev_stage], B_shared[prev_stage], C_local)

    T.copy(C_local, C[by * BM, bx * BN])
```

### 3.5 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `src` | BufferLikeType | Required | Source region (typically global memory) |
| `dst` | BufferLikeType | Required | Destination region (typically shared memory) |
| `coalesced_width` | `int` or `None` | `None` | Coalesced access width |
| `annotations` | `dict` or `None` | `None` | Additional annotations |
| `loop_layout` | `Fragment` or `None` | `None` | Parallel loop layout hint |

### 3.6 Backend Emission

The backend enforces `cp.async` constraints and emits:
1. `ptx_cp_async(...)` -- The actual async copy instruction
2. `ptx_commit_group()` -- Commit the current async group

No wait is auto-inserted for `T.async_copy`. Synchronization is explicit via:
- `T.sync_threads()` -- Block-level barrier
- `T.cp_async_barrier_noinc(barrier)` -- Async barrier arrive
- `T.mbarrier_wait_parity(barrier, parity)` -- Wait on barrier

---

## 4. T.tma_copy

### 4.1 Function Signature

```python
def tma_copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    barrier=None,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    annotations: dict | None = None,
) -> tir.PrimExpr | tir.Stmt:
```

### 4.2 Description

TMA (Tensor Memory Access) copy with user-managed synchronization. This is the lowest-level copy interface that gives explicit control over TMA operations.

### 4.3 Key Differences from T.copy

| Feature | T.copy (with TMA) | T.tma_copy |
|---------|-------------------|------------|
| Synchronization | Full auto (arrive + load + wait) | User-managed |
| Load (global -> shared) | expect_tx + load + wait | expect_tx + load only |
| Store (shared -> global) | store + arrive + wait | store + arrive only |
| Barrier | Auto-created | Required (for loads) |
| Flexibility | Easy, automatic | Maximum control |

### 4.4 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `src` | BufferLikeType | Required | Source region |
| `dst` | BufferLikeType | Required | Destination region |
| `barrier` | BarrierType | `None` | Mbarrier for TMA load synchronization. **Required for loads**. |
| `eviction_policy` | `str` or `None` | `None` | Cache eviction policy |
| `annotations` | `dict` or `None` | `None` | Additional annotations |

### 4.5 TMA Load (Global -> Shared)

For loads, the barrier is required. The TMA load emits:
1. `mbarrier_expect_tx` -- Set expected transaction count
2. `tma_load` -- Issue the TMA load

The user must wait on the barrier explicitly:

```python
with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
    A_shared = T.alloc_shared((BM, BK), "float16")
    mbar = T.alloc_barrier(1)

    # Calculate expected bytes
    tx_bytes = BM * BK * 2  # float16 = 2 bytes

    # Set expected transaction count
    T.mbarrier_expect_tx(mbar[0], tx_bytes)

    # Issue TMA load (does not wait)
    T.tma_copy(A[by * BM, k * BK], A_shared, barrier=mbar[0])

    # Wait for TMA load to complete
    T.mbarrier_wait_parity(mbar[0], parity=0)

    # Data is now available in A_shared
    T.gemm(A_shared, B_shared, C_local)
```

### 4.6 TMA Store (Shared -> Global)

For stores, the barrier is not needed. The TMA store emits:
1. `tma_store` -- Issue the TMA store
2. `tma_store_arrive` -- Signal store completion

The user must wait for stores explicitly:

```python
# Issue TMA store (does not wait)
T.tma_copy(C_shared, C[by * BM, bx * BN])

# Wait for TMA stores to complete
T.tma_store_wait(count=0)  # Wait for all stores
```

### 4.7 TMA with Multi-Buffer Pipeline

```python
with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
    A_shared = T.alloc_shared((num_stages, BM, BK), "float16")
    B_shared = T.alloc_shared((num_stages, BK, BN), "float16")
    C_local = T.alloc_fragment((BM, BN), "float32")

    # One barrier per stage
    load_bar = T.alloc_barrier([1] * num_stages)

    T.clear(C_local)

    for k in T.serial(T.ceildiv(K, BK)):
        stage = k % num_stages
        parity = k % 2

        # Wait for previous use of this stage's buffer to complete
        T.mbarrier_wait_parity(load_bar[stage], parity ^ 1)

        # Issue TMA loads
        T.mbarrier_expect_tx(load_bar[stage], BM * BK * 2 + BK * BN * 2)
        T.tma_copy(A[by * BM, k * BK], A_shared[stage], barrier=load_bar[stage])
        T.tma_copy(B[k * BK, bx * BN], B_shared[stage], barrier=load_bar[stage])

        # Wait for loads to complete
        T.mbarrier_wait_parity(load_bar[stage], parity)

        # Compute
        T.gemm(A_shared[stage], B_shared[stage], C_local)

    T.copy(C_local, C[by * BM, bx * BN])
```

---

## 5. T.transpose

### 5.1 Function Signature

```python
def transpose(
    src: BufferLikeType,
    dst: BufferLikeType,
) -> tir.PrimExpr:
```

### 5.2 Description

Transposes a 2D buffer in shared memory: `dst[j, i] = src[i, j]`. Both source and destination should be shared memory buffers. If `src` has shape `(M, N)`, then `dst` should have shape `(N, M)`.

### 5.3 Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | BufferLikeType | Source buffer or region of shape `(..., M, N)` |
| `dst` | BufferLikeType | Destination buffer or region of shape `(..., N, M)` |

### 5.4 Constraints

- Both buffers must have at least 2 dimensions.
- The last two dimensions are transposed.
- Higher dimensions (batch) must match between source and destination.

### 5.5 Usage Examples

```python
# Simple 2D transpose
A_shared = T.alloc_shared((M, N), "float16")
A_transposed = T.alloc_shared((N, M), "float16")
T.transpose(A_shared, A_transposed)

# Transpose a sub-region
T.transpose(A_shared[i*M:(i+1)*M, :], A_transposed[:, i*M:(i+1)*M])
```

### 5.6 Use Case: GEMM with Transposed Input

```python
# When matrix B needs to be transposed for the GEMM
B_shared = T.alloc_shared((BK, BN), "float16")
B_transposed = T.alloc_shared((BN, BK), "float16")
T.copy(B[k * BK, bx * BN], B_shared)
T.transpose(B_shared, B_transposed)
T.gemm(A_shared, B_transposed, C_local, transpose_B=True)
```

---

## 6. T.c2d_im2col

### 6.1 Function Signature

```python
def c2d_im2col(
    img: BufferLikeType,
    col: BufferLikeType,
    nhw_step: tir.PrimExpr,
    c_step: tir.PrimExpr,
    kernel: int,
    stride: int,
    dilation: int,
    pad: int,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
) -> tir.PrimExpr:
```

### 6.2 Description

Performs Im2Col (Image to Column) transformation for 2D convolution. This operation rearranges image patches into columns for efficient GEMM-based convolution.

### 6.3 Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `img` | BufferLikeType | Input image buffer |
| `col` | BufferLikeType | Output column buffer |
| `nhw_step` | `PrimExpr` | Step size for batch and spatial dimensions |
| `c_step` | `PrimExpr` | Step size for channel dimension |
| `kernel` | `int` | Kernel size (assumes square kernel) |
| `stride` | `int` | Convolution stride |
| `dilation` | `int` | Dilation rate |
| `pad` | `int` | Padding size |
| `eviction_policy` | `str` or `None` | Cache eviction policy |

### 6.4 Usage Example

```python
@tilelang.jit(out_idx=[-1])
def conv2d(N, H, W, CI, CO, KH, KW, stride, pad, block_N, block_K):
    @T.prim_func
    def kernel(
        input: T.Tensor((N, H, W, CI), "float16"),
        weight: T.Tensor((CO, KH, KW, CI), "float16"),
        output: T.Tensor((N, OH, OW, CO), "float16"),
    ):
        OH = (H + 2 * pad - KH) // stride + 1
        OW = (W + 2 * pad - KW) // stride + 1

        with T.Kernel(...) as (bx, by):
            img_smem = T.alloc_shared((...), "float16")
            col_smem = T.alloc_shared((...), "float16")

            # Perform Im2Col
            T.c2d_im2col(
                img_smem, col_smem,
                nhw_step=1,
                c_step=CI,
                kernel=KH,
                stride=stride,
                dilation=1,
                pad=pad,
            )

            # GEMM: output = col @ weight
            T.gemm(col_smem, weight_frag, output_frag)

    return kernel
```

---

## 7. Copy Optimization

### 7.1 Coalesced Memory Access

GPU memory accesses are most efficient when threads in a warp access consecutive memory addresses. The `coalesced_width` parameter controls this:

```python
# Each thread accesses coalesced_width consecutive elements
T.copy(A[by * BM, k * BK], A_shared, coalesced_width=16)
```

**Effect of coalesced_width:**
- `None` (default): Compiler selects optimal width based on data type and shape.
- `4`, `8`, `16`: Explicit width for the innermost dimension of the copy loop.
- Larger values generally improve coalescing but may increase register pressure.

**Recommended values by dtype:**

| dtype | Recommended coalesced_width | Bytes per access |
|-------|----------------------------|------------------|
| `float32` | 4 | 16 bytes |
| `float16` | 8 | 16 bytes |
| `bfloat16` | 8 | 16 bytes |
| `int8` | 16 | 16 bytes |
| `float8_e4m3fn` | 16 | 16 bytes |

Target: 16 bytes (128 bits) per thread memory transaction is optimal for most architectures.

### 7.2 TMA Utilization

Tensor Memory Access (TMA) is the most efficient way to copy data from global to shared memory on SM90+ (Hopper). TMA benefits:

- **Single-thread issue**: Only one thread needs to initiate the copy.
- **Hardware-managed**: The GPU handles the copy without thread involvement.
- **Bulk transfer**: Efficiently moves large contiguous regions.
- **Supports swizzling**: Built-in support for shared memory bank conflict avoidance.

TMA is automatically enabled when:
1. Target is SM90+ (Hopper or later)
2. Source is global memory, destination is shared memory
3. Data is aligned and contiguous
4. `disable_tma=False` (default)

To disable TMA:
```python
T.copy(A[...], A_shared, disable_tma=True)
```

### 7.3 Async Copy Optimization

Asynchronous copy (cp.async) overlaps memory transfer with computation:

```python
# Within T.Pipelined, T.copy automatically uses async copy
for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
    T.copy(A[by * BM, k * BK], A_shared)  # May become async
    T.copy(B[k * BK, bx * BN], B_shared)  # May become async
    T.gemm(A_shared, B_shared, C_local)
```

The pipeline pass transforms these into:
1. Prologue: Load first tiles
2. Steady state: Async load next tiles + compute current tiles
3. Epilogue: Compute last tiles + store results

### 7.4 Copy for Different Architectures

| Architecture | Copy Mechanism | Notes |
|-------------|---------------|-------|
| SM70 (Volta) | SIMT | Manual coalescing |
| SM75 (Turing) | SIMT | Manual coalescing |
| SM80 (Ampere) | cp.async | Asynchronous global->shared |
| SM89 (Ada) | cp.async | Same as Ampere |
| SM90 (Hopper) | TMA / cp.async | TMA preferred for bulk transfers |
| SM100 (Blackwell) | TMA | Enhanced TMA with 2CTA support |
| AMD CDNA2 | Async DMA | AMD-specific async copy |
| AMD CDNA3 | Enhanced DMA | Improved async copy |
| Metal | Metal compute | Threadgroup memory operations |

---

## 8. Eviction Policies

### 8.1 Overview

Eviction policies control how the GPU L2 cache manages data loaded during copy operations:

| Policy | Value | Description | Use Case |
|--------|-------|-------------|----------|
| `"evict_normal"` | 0 | Default L2 cache behavior | General purpose |
| `"evict_first"` | 1 | Data is evicted from L2 first | Data used only once |
| `"evict_last"` | 2 | Data stays in L2 as long as possible | Data reused soon |

### 8.2 Usage

```python
# Keep data in L2 (will be reused)
T.copy(A[by * BM, k * BK], A_shared, eviction_policy="evict_last")

# Data will not be reused, evict quickly
T.copy(A[by * BM, k * BK], A_shared, eviction_policy="evict_first")

# Default behavior
T.copy(A[by * BM, k * BK], A_shared, eviction_policy="evict_normal")
```

### 8.3 L2 Cache Hit Ratio Annotation

For persistent L2 cache control:

```python
# Hint that 50% of the data from A should stay in L2
T.annotate_l2_hit_ratio({A_global: 0.5})
```

This annotation is only applicable to global memory buffers.

### 8.4 When to Use Each Policy

| Scenario | Recommended Policy |
|----------|-------------------|
| Loading tiles for immediate GEMM | `"evict_normal"` |
| Loading tiles that will be reused across iterations | `"evict_last"` |
| Streaming data that won't be reused | `"evict_first"` |
| TMA loads in a pipeline | `"evict_normal"` or `"evict_last"` |

---

## 9. TMA Descriptors and Tensor Maps

### 9.1 TMA Descriptor Creation

TMA descriptors (tensor maps) encode the memory layout of a tensor for hardware-accelerated TMA operations. TileLang automatically creates TMA descriptors during compilation.

The internal API for creating TMA descriptors:

```python
T.create_tma_descriptor(
    data_type,     # Element data type
    rank,          # Number of dimensions
    global_addr,   # Base address of global memory tensor
    *global_shape, # Shape of the tensor
    *global_stride,# Strides of the tensor
    *smem_box,     # Shared memory box dimensions
    *smem_stride,  # Shared memory strides
    interleave,    # Interleave mode
    swizzle,       # Swizzle mode
    l2_promotion,  # L2 cache promotion
    oob_fill,      # Out-of-bounds fill mode
)
```

Total arguments: 7 + 4 * rank.

### 9.2 TMA Load Operations

Internal API for TMA loads:

```python
# Single-CTA TMA load
T.tma_load(descriptor, mbarrier, smem_addr, coord_0, ..., coord_n, eviction_policy)

# 2-CTA TMA load (Blackwell)
T.tma_load_2sm(descriptor, mbarrier, smem_addr, coord_0, ..., coord_n, eviction_policy)
```

### 9.3 User-Managed TMA with Descriptors

For advanced use cases, you can manually manage TMA descriptors:

```python
# Allocate a barrier
mbar = T.alloc_barrier(1)

# Set expected transaction bytes
tx_bytes = BM * BK * dtype_bytes
T.mbarrier_expect_tx(mbar[0], tx_bytes)

# Issue TMA copy
T.tma_copy(A[by * BM, k * BK], A_shared, barrier=mbar[0])

# Wait for completion
T.mbarrier_wait_parity(mbar[0], parity)
```

---

## 10. Memory Access Patterns

### 10.1 Global Memory Access

For optimal global memory performance:

1. **Coalescing**: Ensure consecutive threads access consecutive addresses.
2. **Alignment**: Access 16-byte aligned addresses when possible.
3. **Vectorization**: Use wider load/store operations.

```python
# Good: Consecutive threads access consecutive elements
for i in T.Parallel(N):
    C[i] = A[i] + B[i]

# Bad: Strided access pattern
for i in T.Parallel(N):
    C[i * stride] = A[i * stride] + B[i * stride]
```

### 10.2 Shared Memory Access

For optimal shared memory performance:

1. **Avoid bank conflicts**: Ensure threads don't access the same bank simultaneously.
2. **Use swizzling**: Apply XOR-based swizzle patterns.
3. **Broadcast**: Use shared memory broadcast for read-only data.

### 10.3 Copy Access Patterns by Architecture

#### Ampere (SM80) - cp.async Pattern

```
Global Memory ---------> Shared Memory
     |                         |
     | cp.async.cg             |
     | (128 bytes per thread)  |
     v                         v
Thread 0: Load elements 0-3    -> Shared[0-3]
Thread 1: Load elements 4-7    -> Shared[4-7]
...
```

#### Hopper (SM90) - TMA Pattern

```
Global Memory ---------> Shared Memory
     |                         |
     | TMA bulk tensor copy    |
     | (single thread issues)  |
     v                         v
Thread 0: Issue TMA descriptor -> Hardware handles transfer
Thread 1-N: Free to compute    -> ...
```

### 10.4 Copy Direction Performance

| Direction | Mechanism | Throughput | Latency |
|-----------|-----------|------------|---------|
| Global -> Shared | TMA | ~1 TB/s (H100) | ~200-400 cycles |
| Global -> Shared | cp.async | ~800 GB/s (A100) | ~200 cycles |
| Global -> Shared | SIMT | ~400 GB/s | ~400 cycles |
| Shared -> Fragment | LDSM | ~19 TB/s | ~20 cycles |
| Fragment -> Shared | STSM | ~19 TB/s | ~20 cycles |
| Fragment -> Global | SIMT | ~800 GB/s | ~400 cycles |
| Shared -> Global | TMA store | ~1 TB/s | ~200 cycles |

---

## 11. Copy Between Different Memory Scopes

### 11.1 Global to Shared

```python
# Automatic (recommmended)
T.copy(A_global[...], A_shared)

# With TMA (SM90+)
T.tma_copy(A_global[...], A_shared, barrier=mbar)

# With async (SM80+)
T.async_copy(A_global[...], A_shared)
```

### 11.2 Shared to Fragment

```python
# Automatic dispatch to LDSM (load shared matrix)
T.copy(A_shared, A_frag)
```

When copying from shared memory to a fragment used by GEMM, TileLang dispatches to LDSM (Load Shared Matrix) instructions that are optimized for Tensor Core input.

### 11.3 Fragment to Shared

```python
# Automatic dispatch to STSM (store shared matrix)
T.copy(C_frag, C_shared)
```

### 11.4 Fragment to Global

```python
# Store results back to global memory
T.copy(C_frag, C_global[by * BM, bx * BN])
```

### 11.5 Shared to Shared

```python
# Intra-shared-memory copy
T.copy(A_shared, B_shared)
```

### 11.6 Global to Fragment (Direct)

```python
# Direct load from global to registers (bypasses shared memory)
T.copy(A_global[i], A_frag)
```

This path does not benefit from TMA or cp.async and uses SIMT loads.

### 11.7 Scope Compatibility Matrix

| src\dst | global | shared | fragment | local |
|---------|--------|--------|----------|-------|
| global | N/A | TMA/cp.async/SIMT | SIMT | SIMT |
| shared | TMA/SIMT | SIMT | LDSM | SIMT |
| fragment | SIMT | STSM | SIMT | SIMT |
| local | SIMT | SIMT | SIMT | SIMT |

---

## 12. Vectorized Loads and Stores

### 12.1 Explicit Load Functions

TileLang provides explicit PTX-level load functions for maximum control:

```python
# 32-bit load (4 bytes)
val = T.ldg32(A[i])
val = T.ldg32(A[i], pred=condition)  # Predicated

# 64-bit load (8 bytes)
val = T.ldg64(A[i:i+2])   # Load 2 x float16

# 128-bit load (16 bytes)
val = T.ldg128(A[i:i+4])  # Load 4 x float16

# 256-bit load (32 bytes)
val = T.ldg256(A[i:i+8])  # Load 8 x float16
```

Return types:

| Function | Return Type | Elements Loaded |
|----------|------------|-----------------|
| `T.ldg32` | `uint32` | 1 x 32-bit |
| `T.ldg64` | `uint32x2` | 2 x 32-bit or 4 x 16-bit |
| `T.ldg128` | `uint32x4` | 4 x 32-bit or 8 x 16-bit |
| `T.ldg256` | `uint32x8` | 8 x 32-bit or 16 x 16-bit |

### 12.2 Explicit Store Functions

```python
# 32-bit store
T.stg32(B[i], val)
T.stg32(B[i], val, pred=condition)

# 64-bit store
T.stg64(B[i:i+2], val)

# 128-bit store
T.stg128(B[i:i+4], val)

# 256-bit store
T.stg256(B[i:i+8], val)
```

### 12.3 Read-Only Cache Load

```python
# Load via read-only data cache (__ldg)
val = T.__ldg(A[i])
```

### 12.4 Predicated Loads/Stores

All explicit load/store functions support optional predicates:

```python
# Only load if condition is true
val = T.ldg32(A[i], pred=(i < N))

# Only store if condition is true
T.stg32(B[i], val, pred=(i < N))
```

When the predicate is false, the load/store is skipped.

---

## 13. Copy with Layout Hints

### 13.1 Parallel Loop Layout

The `loop_layout` parameter accepts a `Fragment` that defines the iteration pattern for the SIMT copy:

```python
from tilelang.layout import Fragment

# Define a custom iteration layout
copy_layout = Fragment((BM, BK), lambda i, j: ...)

# Apply to copy operation
T.copy(A[by * BM, k * BK], A_shared, loop_layout=copy_layout)
```

This is only valid for SIMT copy paths and is incompatible with TMA, LDSM, STSM, and TMEM copies.

### 13.2 Layout Inference for Copy

The compiler infers copy layouts based on:
1. The source buffer's layout (if annotated)
2. The destination buffer's layout (if annotated)
3. The downstream consumer's layout preference (e.g., GEMM)

### 13.3 Copy and GEMM Layout Coordination

For optimal Tensor Core performance, the copy into shared memory should produce a layout that matches the GEMM input layout:

```python
A_shared = T.alloc_shared((BM, BK), "float16")
B_shared = T.alloc_shared((BK, BN), "float16")
C_local = T.alloc_fragment((BM, BN), "float32")

# The compiler infers layouts for A_shared and B_shared
# based on how they're used in T.gemm
T.copy(A[by * BM, k * BK], A_shared)
T.copy(B[k * BK, bx * BN], B_shared)
T.gemm(A_shared, B_shared, C_local)
```

---

## 14. Advanced Copy Patterns

### 14.1 Double Buffering

```python
A_shared = T.alloc_shared((2, BM, BK), "float16")

# Load into buffer 0
T.copy(A[by * BM, 0], A_shared[0])

for k in T.serial(1, T.ceildiv(K, BK)):
    # Load into next buffer while computing current
    stage = k % 2
    T.copy(A[by * BM, k * BK], A_shared[stage])
    T.gemm(A_shared[1 - stage], B_shared, C_local)

# Compute last buffer
T.gemm(A_shared[(K // BK - 1) % 2], B_shared, C_local)
```

### 14.2 Multi-Stage Pipeline with T.copy

```python
for k in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
    # The compiler transforms this into:
    # - Prologue: Load first 2 tiles
    # - Body: Load tile k+2, compute tile k
    # - Epilogue: Compute remaining tiles
    T.copy(A[by * BM, k * BK], A_shared)
    T.copy(B[k * BK, bx * BN], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

### 14.3 Persistent Kernel Copy

```python
for tile_idx in T.Persistent(domain=[total_tiles], wave_size=num_sms, index=0):
    # Compute coordinates for this tile
    bx = tile_idx % grid_x
    by = tile_idx // grid_x

    # Load and compute
    T.copy(A[by * BM, 0], A_shared)
    T.copy(B[0, bx * BN], B_shared)
    T.gemm(A_shared, B_shared, C_local)
    T.copy(C_local, C[by * BM, bx * BN])
```

### 14.4 Copy with Bounds Checking

```python
# For tiles at the edge of the matrix, use safe value annotations
T.annotate_safe_value({C_local: 0})

# The compiler handles out-of-bounds reads by substituting safe values
T.copy(A[by * BM, k * BK], A_shared)  # Safe even if near matrix boundary
```

### 14.5 AMD-Specific Copy

On AMD GPUs, TileLang uses MFMA-compatible layouts and DMA engines:

```python
# AMD async copy
T.async_copy(A[by * BM, k * BK], A_shared)

# AMD LDS transpose reads (gfx950 only)
val = T.ds_read_tr16_b64(smem[i])  # 16-element transpose, 64-bit
val = T.ds_read_tr8_b64(smem[i])   # 8-element transpose, 64-bit
```

### 14.6 Blackwell TCGEN05 Copy

```python
# Copy from shared memory to TMEM using TCGEN05
T.tcgen05_cp_warpx4(smem_scale, tmem_dst, tmem_col_offset=0)

# Transpose scale factors in shared memory
T.tcgen05_sf_warp_transpose(smem_scale)

# Copy with 2-CTA mode
T.tcgen05_cp_warpx4(smem_scale, tmem_dst, use_2cta=True)
```

### 14.7 Copy with Cluster Operations

On SM90+, data can be shared across CTAs within a cluster:

```python
with T.Kernel(grid_x, grid_y, threads=128, cluster_dims=(2, 1, 1)) as (bx, by):
    A_shared = T.alloc_shared((BM, BK), "float16")

    # Copy data, then share with neighbor CTA
    T.copy(A[by * BM, k * BK], A_shared)

    # Synchronize within cluster
    T.cluster_sync()

    # Neighbor CTA can now access A_shared
```

### 14.8 Copy with Prefer Async Hint

```python
# Hint that this parallel loop should prefer cp.async for copies
for i, j in T.Parallel(BM, BN, prefer_async=True):
    ...
```

### 14.9 TMA Store Pattern

```python
# Store to global memory via TMA
T.copy(C_shared, C[by * BM, bx * BN])

# Or explicit TMA store
T.tma_copy(C_shared, C[by * BM, bx * BN])
T.tma_store_arrive()  # Signal store completion
T.tma_store_wait(0)    # Wait for all stores
```

### 14.10 Fence Operations

```python
# Fence for async proxy operations
T.fence_proxy_async()

# This ensures prior async operations (e.g., TMA stores) are visible
# to subsequent memory accesses
```
