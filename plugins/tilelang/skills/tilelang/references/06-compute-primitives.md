# TileLang Compute Primitives Reference

This reference covers all compute operations available in TileLang, from matrix multiplication primitives that map directly to GPU tensor cores, to element-wise math operations and buffer management utilities.

---

## Table of Contents

1. [Overview](#overview)
2. [T.gemm -- General Matrix Multiply](#tgemm----general-matrix-multiply)
3. [GemmWarpPolicy Options](#gemmmwarppolicy-options)
4. [T.wgmma_gemm -- Hopper WGMMA Async GEMM](#twgmma_gemm----hopper-wgmma-async-gemm)
5. [T.tcgen05_gemm -- Blackwell TCGEN05 GEMM](#ttcgen05_gemm----blackwell-tcgen05-gemm)
6. [T.tcgen05_gemm_blockscaled -- Block-Scaled GEMM for MX Formats](#ttcgen05_gemm_blockscaled----block-scaled-gemm-for-mx-formats)
7. [make_blockscaled_gemm_layout](#make_blockscaled_gemm_layout)
8. [T.gemm_sp -- Sparse GEMM with 2:4 Sparsity](#tgemm_sp----sparse-gemm-with-24-sparsity)
9. [T.gemm_sp_v2 -- Sparse GEMM Variant](#tgemm_sp_v2----sparse-gemm-variant)
10. [Element-wise Math Operations](#element-wise-math-operations)
11. [T.clear -- Buffer Clearing](#tclear----buffer-clearing)
12. [Accumulator Management](#accumulator-management)
13. [Practical Examples](#practical-examples)

---

## Overview

TileLang provides a set of high-level compute primitives that abstract the complexity of GPU tensor core programming. These primitives are designed to map efficiently to hardware-specific instructions across NVIDIA GPU architectures:

| Architecture | Compute Capability | Tensor Core Instruction | TileLang Primitive |
|-------------|-------------------|------------------------|-------------------|
| Ampere | SM 80 | MMA (m16n8k16) | `T.gemm` |
| Hopper | SM 90 | WGMMA (async) | `T.gemm`, `T.wgmma_gemm` |
| Blackwell | SM 100 | TCGEN05 | `T.gemm`, `T.tcgen05_gemm` |

All compute primitives operate on TileLang buffer objects that have been allocated via `T.alloc_shared`, `T.alloc_local`, or passed as kernel parameters. The primitives handle register allocation, warp-level scheduling, and memory access patterns internally.

---

## T.gemm -- General Matrix Multiply

### Signature

```python
T.gemm(
    A,                          # Input buffer: left operand (M x K)
    B,                          # Input buffer: right operand (K x N)
    C,                          # Output buffer: result (M x N)
    transpose_A=False,          # Whether A is transposed
    transpose_B=False,          # Whether B is transposed
    policy=GemmWarpPolicy.Square,  # Warp assignment policy
    clear_accum=False,          # Whether to clear accumulator before GEMM
    k_pack=1,                   # K-dimension packing factor
    mbar=None,                  # Memory barrier for async operations
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A` | Buffer | required | Left input matrix buffer. Must be a 2D buffer of shape `[M, K]` (or `[K, M]` if `transpose_A=True`). |
| `B` | Buffer | required | Right input matrix buffer. Must be a 2D buffer of shape `[K, N]` (or `[N, K]` if `transpose_B=True`). |
| `C` | Buffer | required | Output accumulator buffer of shape `[M, N]`. Typically a local (register) buffer. |
| `transpose_A` | bool | `False` | If `True`, interpret A as transposed (`K x M` layout). |
| `transpose_B` | bool | `False` | If `True`, interpret B as transposed (`N x K` layout). |
| `policy` | GemmWarpPolicy | `Square` | Warp-to-fragment assignment policy. Controls how warps within a thread block divide the output tile. |
| `clear_accum` | bool | `False` | If `True`, the accumulator C is zeroed before the GEMM is performed. Equivalent to `C = A @ B`. If `False`, the result is accumulated: `C += A @ B`. |
| `k_pack` | int | `1` | The number of K-dimension elements packed into a single tensor core operation. Values greater than 1 can improve throughput for narrow data types (e.g., `int8`, `float8_e4m3`). |
| `mbar` | Buffer or None | `None` | A memory barrier object for synchronizing async GEMM operations. Required for WGMMA and TCGEN05 on Hopper/Blackwell architectures. |

### Operation Semantics

The GEMM operation computes:

```
if clear_accum:
    C[m, n] = sum_k A[m, k] * B[k, n]    # C = A @ B
else:
    C[m, n] += sum_k A[m, k] * B[k, n]   # C += A @ B
```

Where the interpretation of `A` and `B` dimensions depends on the transpose flags:

```
A shape: [M, K] if not transpose_A, [K, M] if transpose_A
B shape: [K, N] if not transpose_B, [N, K] if transpose_B
C shape: [M, N] always
```

### Tensor Core Mapping

`T.gemm` automatically selects the optimal hardware instruction based on the target architecture:

#### Ampere (SM 80) -- MMA Instructions

On Ampere GPUs, `T.gemm` compiles to `nvmma.mma.async` instructions. The MMA instruction operates on fixed-size fragments:

| Input Type | MMA Shape (m x n x k) | Accumulator Type |
|-----------|----------------------|-----------------|
| float16 | 16 x 8 x 16 | float32 |
| int8 | 16 x 8 x 32 | int32 |
| bfloat16 | 16 x 8 x 16 | float32 |
| float32 | 16 x 8 x 8 | float32 |
| tf32 | 16 x 8 x 8 | float32 |

Each fragment is distributed across the 32 threads of a warp. A single warp-level MMA instruction computes a small tile of the output matrix.

#### Hopper (SM 90) -- WGMMA Instructions

On Hopper GPUs, `T.gemm` compiles to `wgmma.mma_async` instructions. WGMMA is a major evolution over Ampere MMA:

- **Async execution**: WGMMA operates asynchronously, allowing overlap with other operations.
- **Larger tile sizes**: WGMMA supports larger M dimensions (up to 256) compared to Ampere's 16.
- **Shared memory source**: A-matrix data is read directly from shared memory, reducing register pressure.
- **Multiple data types**: Supports float16, bfloat16, tf32, int8, float8_e4m3, float8_e5m2.

When `mbar` is provided, the WGMMA operation uses it for synchronization. The barrier must be arrived on after all input data is ready:

```python
mbar = T.alloc_shared([1], "uint64")
# ... load A and B into shared memory ...
T.gemm(A_shared, B_shared, C_local, mbar=mbar)
```

#### Blackwell (SM 100) -- TCGEN05 Instructions

On Blackwell GPUs, `T.gemm` may compile to TCGEN05 instructions. See the `T.tcgen05_gemm` section for details on explicit TCGEN05 usage.

### Supported Data Type Combinations

| Input A | Input B | Accumulator C | Tensor Core Availability |
|---------|---------|--------------|-------------------------|
| float16 | float16 | float16 | SM 70+ |
| float16 | float16 | float32 | SM 70+ |
| bfloat16 | bfloat16 | float32 | SM 80+ |
| tf32 | tf32 | float32 | SM 80+ |
| int8 | int8 | int32 | SM 75+ |
| int8 | int8 | float32 | SM 80+ |
| float8_e4m3 | float8_e5m2 | float32 | SM 90+ |
| float8_e5m2 | float8_e4m3 | float32 | SM 90+ |
| float8_e4m3 | float8_e4m3 | float32 | SM 90+ |
| float16 | float16 | float16 | SM 90+ |
| int4 | int4 | int32 | SM 90+ |

### Example: Basic GEMM with Tensor Cores

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])
def matmul(M, N, K, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype):
    A = T.alloc_shared([block_M, block_K], in_dtype)
    B = T.alloc_shared([block_K, block_N], in_dtype)
    C = T.alloc_local([block_M, block_N], accum_dtype)

    # Initialize accumulator to zero
    T.clear(C)

    # Pipeline the K-dimension for better overlap
    for k in range(0, K, block_K):
        # Load tiles from global memory to shared memory
        T.copy(A_global[k:k+block_K], A)
        T.copy(B_global[k:k+block_K], B)

        # Synchronize to ensure shared memory writes are visible
        T.sync_shared_memory()

        # Perform tensor core GEMM: C += A @ B
        T.gemm(A, B, C)

    # Copy result from local (register) to global memory
    T.copy(C, C_global)

    return C_global
```

### Example: Transposed GEMM

```python
# When A is stored in column-major (transposed) layout:
T.gemm(A, B, C, transpose_A=True)

# When B is stored in row-major (transposed) layout:
T.gemm(A, B, C, transpose_B=True)

# Both transposed:
T.gemm(A, B, C, transpose_A=True, transpose_B=True)
```

### Example: Using k_pack for Packed Operations

```python
# For int8 or float8 data types, k_pack > 1 enables packing multiple
# K-dimension elements into a single tensor core operation, improving
# throughput by better utilizing the tensor core's wide data paths.

# Pack 2 elements along K for improved int8 throughput:
T.gemm(A, B, C, k_pack=2)

# Pack 4 elements for maximum throughput with narrow types:
T.gemm(A, B, C, k_pack=4)
```

---

## GemmWarpPolicy Options

The `GemmWarpPolicy` enum controls how warps within a thread block are assigned to sub-tiles of the output matrix C. This has significant performance implications depending on the matrix shape and memory access patterns.

### GemmWarpPolicy.Square (Default)

Warps are arranged in a 2D grid that is as close to square as possible. For example, with 4 warps (128 threads):

```
Output tile C[block_M x block_N] is divided into 4 sub-tiles:

+--------+--------+
| Warp 0 | Warp 1 |
+--------+--------+
| Warp 2 | Warp 3 |
+--------+--------+
```

The square policy provides balanced shared memory access patterns for both A and B matrices, making it the best default choice for most workloads.

**When to use**: General-purpose GEMM where M and N dimensions are similar in size.

### GemmWarpPolicy.FullRow

All warps are assigned along the N (column) dimension. Each warp computes a full row of the output tile:

```
Output tile C[block_M x block_N]:

+---------------------------+
| Warp 0 (full row 0)      |
+---------------------------+
| Warp 1 (full row 1)      |
+---------------------------+
| Warp 2 (full row 2)      |
+---------------------------+
| Warp 3 (full row 3)      |
+---------------------------+
```

**When to use**: When `block_M >> block_N` (tall, skinny tiles). This policy allows each warp to stream B data more efficiently since each warp reads the entire B row.

```python
T.gemm(A, B, C, policy=GemmWarpPolicy.FullRow)
```

### GemmWarpPolicy.FullCol

All warps are assigned along the M (row) dimension. Each warp computes a full column of the output tile:

```
Output tile C[block_M x block_N]:

+----+----+----+----+
|    |    |    |    |
| W0 | W1 | W2 | W3 |
|    |    |    |    |
+----+----+----+----+
```

**When to use**: When `block_N >> block_M` (wide, short tiles). This policy allows each warp to stream A data more efficiently.

```python
T.gemm(A, B, C, policy=GemmWarpPolicy.FullCol)
```

### Policy Selection Guide

| Scenario | Recommended Policy | Rationale |
|----------|-------------------|-----------|
| `block_M ~ block_N` | `Square` | Balanced memory access |
| `block_M >> block_N` | `FullRow` | Better B-matrix reuse per warp |
| `block_N >> block_M` | `FullCol` | Better A-matrix reuse per warp |
| Small tiles (e.g., 16x16) | `Square` | Minimal overhead |
| Large tiles (e.g., 128x256) | `FullCol` | Better parallelism utilization |
| Batched small GEMMs | `Square` | Uniform workload distribution |

---

## T.wgmma_gemm -- Hopper WGMMA Async GEMM

### Signature

```python
T.wgmma_gemm(
    A,                      # Shared memory buffer for A matrix
    B,                      # Register or shared memory buffer for B matrix
    C,                      # Register accumulator buffer for C matrix
    transpose_A=False,      # Whether A is transposed
    transpose_B=False,      # Whether B is transposed
    mbar=None,              # Memory barrier for async completion
)
```

### Overview

`T.wgmma_gemm` provides explicit access to the Hopper (SM 90) WGMMA (Warp Group Matrix Multiply Accumulate) instruction. Unlike `T.gemm` which abstracts the hardware instruction selection, `T.wgmma_gemm` always generates WGMMA instructions.

WGMMA is a warp-group-level instruction (4 warps = 128 threads cooperating) that performs asynchronous matrix multiplication. Key characteristics:

- **Asynchronous**: The operation is fire-and-forget. Use `mbar` to synchronize.
- **Shared memory source for A**: The A matrix is read directly from shared memory, reducing register pressure.
- **Large tile support**: Supports M dimensions up to 256 in a single instruction.
- **Warp-group scope**: All 128 threads in a warp group cooperate on a single matrix tile.

### Supported Tile Sizes

| M | N | K | Notes |
|---|---|---|-------|
| 64 | N/4 | 16 | Minimum M for float16 |
| 128 | N/4 | 16 | High throughput |
| 256 | N/4 | 16 | Maximum M dimension |
| 64 | N/4 | 32 | For int8 / float8 types |

Note: N is divided by 4 because each warp in the warp group handles a quarter of the N dimension.

### Example: WGMMA with Memory Barrier

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])
def wgmma_matmul(M, N, K, block_M=128, block_N=128, block_K=32):
    # Allocate shared memory buffers
    A_smem = T.alloc_shared([block_M, block_K], "float16")
    B_smem = T.alloc_shared([block_K, block_N], "float16")

    # Allocate register accumulator
    C_local = T.alloc_local([block_M, block_N], "float32")

    # Allocate memory barrier for async operations
    mbar = T.alloc_shared([1], "uint64")
    T.clear(C_local)

    for k in range(0, K, block_K):
        # Initiate async copy from global to shared memory
        T.copy(A_global[k:k+block_K], A_smem, mbar=mbar)
        T.copy(B_global[k:k+block_K], B_smem, mbar=mbar)

        # Wait for async copies to complete
        T.wait_memory_barrier(mbar)

        # Issue async WGMMA
        T.wgmma_gemm(A_smem, B_smem, C_local, mbar=mbar)

    # Wait for all WGMMA operations to complete
    T.wait_memory_barrier(mbar)

    T.copy(C_local, C_global)
    return C_global
```

### WGMMA vs MMA Comparison

| Feature | MMA (Ampere) | WGMMA (Hopper) |
|---------|-------------|----------------|
| Scope | Warp (32 threads) | Warp group (128 threads) |
| Execution | Synchronous | Asynchronous |
| Max M dimension | 16 | 256 |
| A source | Registers | Shared memory or registers |
| Barrier | Not needed | Required (`mbar`) |
| Register pressure | Higher (A in registers) | Lower (A from shared memory) |

---

## T.tcgen05_gemm -- Blackwell TCGEN05 GEMM

### Signature

```python
T.tcgen05_gemm(
    A,                      # Input buffer A
    B,                      # Input buffer B
    C,                      # Output accumulator buffer C
    transpose_A=False,      # Whether A is transposed
    transpose_B=False,      # Whether B is transposed
    mbar=None,              # Memory barrier for synchronization
    use_2cta=False,         # Enable 2-CTA mode for larger tiles
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A` | Buffer | required | Shared memory buffer for the A matrix. |
| `B` | Buffer | required | Shared memory buffer for the B matrix. |
| `C` | Buffer | required | Register accumulator buffer for the result. |
| `transpose_A` | bool | `False` | Whether to transpose A. |
| `transpose_B` | bool | `False` | Whether to transpose B. |
| `mbar` | Buffer | `None` | Memory barrier object. Required for TCGEN05 synchronization. |
| `use_2cta` | bool | `False` | If `True`, enables 2-CTA mode where two thread blocks cooperate on a single GEMM tile, doubling the effective tile size. |

### Overview

TCGEN05 is the tensor core instruction introduced with NVIDIA Blackwell (SM 100) architecture. It provides significant improvements over Hopper WGMMA:

- **Higher throughput**: Up to 2x the matrix multiply throughput of Hopper.
- **Block-scaled support**: Native support for MX-format block-scaled operations.
- **2-CTA mode**: Two thread blocks can cooperate on a single GEMM operation, enabling larger effective tile sizes.
- **Tensor memory**: Direct integration with Blackwell's tensor memory (TMEM) subsystem.

### 2-CTA Mode

When `use_2cta=True`, two thread blocks cooperate on the same GEMM tile. This is useful for very large tile sizes where a single thread block does not have enough warps to efficiently cover the output:

```python
# Enable 2-CTA mode for large tiles
T.tcgen05_gemm(
    A, B, C,
    mbar=mbar,
    use_2cta=True
)
```

In 2-CTA mode:
- Thread block 0 handles the first half of the M dimension.
- Thread block 1 handles the second half.
- Both blocks must synchronize via the shared memory barrier.

### Example: Blackwell TCGEN05 GEMM

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])
def blackwell_matmul(M, N, K, block_M=128, block_N=256, block_K=64):
    A_smem = T.alloc_shared([block_M, block_K], "float8_e4m3")
    B_smem = T.alloc_shared([block_K, block_N], "float8_e5m2")
    C_local = T.alloc_local([block_M, block_N], "float32")

    mbar = T.alloc_shared([1], "uint64")
    T.clear(C_local)

    for k in range(0, K, block_K):
        T.copy(A_global[k:k+block_K], A_smem, mbar=mbar)
        T.copy(B_global[k:k+block_K], B_smem, mbar=mbar)
        T.wait_memory_barrier(mbar)

        T.tcgen05_gemm(A_smem, B_smem, C_local, mbar=mbar)

    T.wait_memory_barrier(mbar)
    T.copy(C_local, C_global)
    return C_global
```

---

## T.tcgen05_gemm_blockscaled -- Block-Scaled GEMM for MX Formats

### Signature

```python
T.tcgen05_gemm_blockscaled(
    A,                      # Input buffer A (MX-format values)
    B,                      # Input buffer B (MX-format values)
    C,                      # Output accumulator buffer
    SFA_tmem,               # Scale factors for A in tensor memory
    SFB_tmem,               # Scale factors for B in tensor memory
    transpose_A=False,      # Whether A is transposed
    transpose_B=False,      # Whether B is transposed
    mbar=None,              # Memory barrier
    use_2cta=False,         # 2-CTA mode
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A` | Buffer | required | MX-format value buffer for A matrix (e.g., `float8_e4m3` values). |
| `B` | Buffer | required | MX-format value buffer for B matrix. |
| `C` | Buffer | required | Output accumulator (typically `float32`). |
| `SFA_tmem` | Buffer | required | Block scale factors for A, stored in tensor memory. Shape: `[M / 32, K / block_size]`. |
| `SFB_tmem` | Buffer | required | Block scale factors for B, stored in tensor memory. Shape: `[K / block_size, N / 32]`. |
| `transpose_A` | bool | `False` | Whether to transpose A. |
| `transpose_B` | bool | `False` | Whether to transpose B. |
| `mbar` | Buffer | `None` | Memory barrier for async synchronization. |
| `use_2cta` | bool | `False` | Enable 2-CTA cooperative mode. |

### Overview

Block-scaled GEMM is a key feature of Blackwell tensor cores that enables efficient computation with **MX (Microscaling) formats**. MX formats represent a block of values (typically 32 elements) using a shared scale factor and per-element narrow-format values:

```
MX block layout (32 elements):
[value_0, value_1, ..., value_31] * shared_scale_factor
```

The supported MX formats include:

| MX Format | Value Type | Scale Type | Block Size |
|-----------|-----------|-----------|-----------|
| MXFP8 | float8_e4m3 or float8_e5m2 | float8_e8m0 | 32 |
| MXFP6 | float6_e2m3fn or float6_e3m2fn | float8_e8m0 | 32 |
| MXFP4 | float4_e2m1fn | float8_e8m0 | 32 |
| MXINT8 | int8 | float8_e8m0 | 32 |

The block-scaled GEMM operation computes:

```
C[m, n] = sum_k (A[m, k] * SFA[m, k_block]) * (B[k, n] * SFB[k_block, n])
```

Where `SFA` and `SFB` are the block scale factors that convert the narrow-format values back to full precision before accumulation.

### Example: Block-Scaled GEMM

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])
def blockscaled_matmul(M, N, K, block_M=128, block_N=256, block_K=128):
    # MX-format value buffers
    A_smem = T.alloc_shared([block_M, block_K], "float8_e4m3")
    B_smem = T.alloc_shared([block_K, block_N], "float8_e5m2")

    # Block scale factor buffers in tensor memory
    block_size = 32
    SFA_tmem = T.alloc_tmem([block_M // block_size, block_K // block_size], "float8_e8m0")
    SFB_tmem = T.alloc_tmem([block_K // block_size, block_N // block_size], "float8_e8m0")

    # Output accumulator
    C_local = T.alloc_local([block_M, block_N], "float32")

    mbar = T.alloc_shared([1], "uint64")
    T.clear(C_local)

    for k in range(0, K, block_K):
        # Load values and scales
        T.copy(A_vals_global[k:k+block_K], A_smem, mbar=mbar)
        T.copy(B_vals_global[k:k+block_K], B_smem, mbar=mbar)
        T.copy(A_scales_global, SFA_tmem)
        T.copy(B_scales_global, SFB_tmem)
        T.wait_memory_barrier(mbar)

        # Block-scaled GEMM
        T.tcgen05_gemm_blockscaled(
            A_smem, B_smem, C_local,
            SFA_tmem, SFB_tmem,
            mbar=mbar
        )

    T.wait_memory_barrier(mbar)
    T.copy(C_local, C_global)
    return C_global
```

---

## make_blockscaled_gemm_layout

### Signature

```python
make_blockscaled_gemm_layout(C, A, transpose_A=False)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `C` | Buffer | required | The output accumulator buffer. |
| `A` | Buffer | required | The input matrix buffer (MX-format values). |
| `transpose_A` | bool | `False` | Whether A is in transposed layout. |

### Overview

`make_blockscaled_gemm_layout` is a layout helper that computes the appropriate shared memory and tensor memory layouts for block-scaled GEMM operations. It determines:

1. The swizzle pattern for shared memory access to avoid bank conflicts.
2. The tensor memory layout for scale factors.
3. The mapping between value elements and their corresponding scale factors.

### Example

```python
from tilelang.language import make_blockscaled_gemm_layout

# Create the layout for a block-scaled GEMM
layout = make_blockscaled_gemm_layout(C_local, A_smem, transpose_A=False)

# The layout object contains:
# - layout.smem_layout_a: Shared memory layout for A values
# - layout.smem_layout_b: Shared memory layout for B values
# - layout.tmem_layout_sfa: Tensor memory layout for A scales
# - layout.tmem_layout_sfb: Tensor memory layout for B scales
# - layout.gemm_intrin_info: Tensor core instruction parameters
```

---

## T.gemm_sp -- Sparse GEMM with 2:4 Sparsity

### Signature

```python
T.gemm_sp(
    A,                      # Dense buffer for B matrix
    B,                      # Sparse buffer for A matrix (2:4 sparse)
    C,                      # Output accumulator buffer
    E,                      # Sparse metadata (index) buffer
    transpose_A=False,      # Whether A is transposed
    transpose_B=False,      # Whether B is transposed
    policy=GemmWarpPolicy.Square,
    clear_accum=False,
    mbar=None,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A` | Buffer | required | Dense input matrix (replaces the sparse dimension). |
| `B` | Buffer | required | Sparse input matrix with 2:4 sparsity pattern. |
| `C` | Buffer | required | Output accumulator buffer. |
| `E` | Buffer | required | Sparse metadata buffer encoding the 2:4 sparsity pattern (which 2 of 4 elements are non-zero). |
| `transpose_A` | bool | `False` | Whether A is transposed. |
| `transpose_B` | bool | `False` | Whether B is transposed. |
| `policy` | GemmWarpPolicy | `Square` | Warp assignment policy. |
| `clear_accum` | bool | `False` | Whether to clear the accumulator before the operation. |
| `mbar` | Buffer | `None` | Memory barrier for async operations. |

### Overview

2:4 structured sparsity is a hardware feature introduced with NVIDIA Ampere (SM 80) tensor cores. In a 2:4 sparse pattern, exactly 2 out of every 4 consecutive elements are zero. This allows the tensor core to skip computation on the zero elements, effectively doubling the throughput compared to dense GEMM for the same input dimensions.

The sparse metadata buffer `E` encodes which elements are non-zero. For each group of 4 elements, a 2-bit index indicates the position of the non-zero elements. The metadata is typically precomputed during the sparse matrix pruning step.

**Experimental**: This API is marked experimental and may change in future releases.

### Example

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])
def sparse_matmul(M, N, K, block_M=64, block_N=64, block_K=32):
    # Dense B matrix
    B_dense = T.alloc_shared([block_K, block_N], "float16")
    # Sparse A matrix (pruned to 2:4 pattern, stored compressed)
    A_sparse = T.alloc_shared([block_M, block_K // 2], "float16")
    # Sparse metadata
    E_meta = T.alloc_shared([block_M, block_K // 4], "uint8")

    C_local = T.alloc_local([block_M, block_N], "float32")
    T.clear(C_local)

    for k in range(0, K, block_K):
        T.copy(A_sparse_global[k:k+block_K], A_sparse)
        T.copy(B_dense_global[k:k+block_K], B_dense)
        T.copy(E_global, E_meta)
        T.sync_shared_memory()

        T.gemm_sp(
            B_dense, A_sparse, C_local, E_meta,
            clear_accum=(k == 0)
        )

    T.copy(C_local, C_global)
    return C_global
```

---

## T.gemm_sp_v2 -- Sparse GEMM Variant

### Signature

```python
T.gemm_sp_v2(
    A,                      # Dense buffer
    B,                      # Sparse buffer (2:4 sparse)
    C,                      # Output accumulator
    E,                      # Sparse metadata
    transpose_A=False,
    transpose_B=False,
    policy=GemmWarpPolicy.Square,
    clear_accum=False,
    mbar=None,
)
```

### Overview

`T.gemm_sp_v2` is an alternative sparse GEMM API that provides the same functionality as `T.gemm_sp` but with an updated internal implementation. The key differences include:

- **Improved register allocation**: Better handling of sparse metadata in registers.
- **Optimized index computation**: Faster sparse index resolution.
- **Support for Hopper WGMMA sparse**: Uses the Hopper sparse WGMMA instruction when available.

The API signature and parameters are identical to `T.gemm_sp`. Use `T.gemm_sp_v2` when targeting Hopper or later architectures for best performance.

---

## Element-wise Math Operations

TileLang provides a comprehensive set of element-wise math operations that can be applied within kernels. These operations map to hardware math instructions on the GPU and support all standard floating-point data types.

### T.exp -- Exponential

```python
T.exp(buffer)  # Element-wise exp: out[i] = exp(buffer[i])
```

Computes the base-e exponential of each element. Maps to hardware `ex2.approx.ftz.f32` instruction (using `exp2(x * log2(e))`).

```python
import tilelang.language as T

# Apply exponential to all elements
result = T.exp(input_buffer)

# Typical usage in softmax computation
x_max = T.reduce_max(x, dim=-1)
x_shifted = x - x_max  # Numerical stability
exp_values = T.exp(x_shifted)
```

### T.max -- Maximum

```python
T.max(a, b)  # Element-wise max: out[i] = max(a[i], b[i])
```

Computes the element-wise maximum of two buffers or a buffer and a scalar.

```python
# Buffer-buffer maximum
result = T.max(buffer_a, buffer_b)

# Buffer-scalar maximum (broadcast)
result = T.max(buffer, 0.0)  # ReLU activation
```

### T.min -- Minimum

```python
T.min(a, b)  # Element-wise min: out[i] = min(a[i], b[i])
```

Computes the element-wise minimum of two buffers or a buffer and a scalar.

```python
# Clamp values between 0 and 1
clamped = T.min(T.max(buffer, 0.0), 1.0)
```

### T.log -- Natural Logarithm

```python
T.log(buffer)  # Element-wise log: out[i] = log(buffer[i])
```

Computes the natural logarithm of each element. Maps to hardware `lg2.approx.ftz.f32` instruction (using `log2(x) / log2(e)`).

```python
# Log-softmax computation
log_sum_exp = T.log(T.reduce_sum(T.exp(x_shifted), dim=-1))
log_softmax = x_shifted - log_sum_exp
```

### T.sin -- Sine

```python
T.sin(buffer)  # Element-wise sin: out[i] = sin(buffer[i])
```

Computes the sine of each element (input in radians).

### T.cos -- Cosine

```python
T.cos(buffer)  # Element-wise cos: out[i] = cos(buffer[i])
```

Computes the cosine of each element (input in radians).

```python
# Rotary Position Embedding (RoPE)
cos_theta = T.cos(freqs)
sin_theta = T.sin(freqs)
q_rotated = q * cos_theta + rotate_half(q) * sin_theta
```

### T.tanh -- Hyperbolic Tangent

```python
T.tanh(buffer)  # Element-wise tanh: out[i] = tanh(buffer[i])
```

### T.rsqrt -- Reciprocal Square Root

```python
T.rsqrt(buffer)  # Element-wise rsqrt: out[i] = 1 / sqrt(buffer[i])
```

Highly optimized -- maps directly to the hardware `rsqrt.approx.ftz.f32` instruction.

```python
# RMS normalization
x_sq = x * x
mean_sq = T.reduce_sum(x_sq, dim=-1) / dim
rsqrt_mean = T.rsqrt(mean_sq)
normalized = x * rsqrt_mean
```

### T.sqrt -- Square Root

```python
T.sqrt(buffer)  # Element-wise sqrt: out[i] = sqrt(buffer[i])
```

### T.abs -- Absolute Value

```python
T.abs(buffer)  # Element-wise abs: out[i] = abs(buffer[i])
```

### T.neg -- Negation

```python
T.neg(buffer)  # Element-wise neg: out[i] = -buffer[i]
```

### T.clamp -- Value Clamping

```python
T.clamp(buffer, min_val, max_val)  # out[i] = max(min_val, min(buffer[i], max_val))
```

### T.floor / T.ceil -- Rounding

```python
T.floor(buffer)  # Round toward negative infinity
T.ceil(buffer)   # Round toward positive infinity
```

### T.pow -- Power

```python
T.pow(base, exponent)  # out[i] = base[i] ^ exponent[i]
```

### Element-wise Arithmetic Operators

TileLang also supports standard Python arithmetic operators on buffers:

```python
# Addition
C = A + B

# Subtraction
C = A - B

# Multiplication (element-wise, NOT matrix multiply)
C = A * B

# Division
C = A / B

# Scalar operations
C = A * 2.0
C = A + 1.0
C = 1.0 / A
```

### Complete Math Function Reference

| Function | Signature | Hardware Mapping | Supported Types |
|----------|-----------|-----------------|----------------|
| `T.exp` | `T.exp(x)` | `ex2.approx` | float16, float32, bfloat16 |
| `T.log` | `T.log(x)` | `lg2.approx` | float16, float32, bfloat16 |
| `T.sin` | `T.sin(x)` | `sin.approx` | float16, float32 |
| `T.cos` | `T.cos(x)` | `cos.approx` | float16, float32 |
| `T.tanh` | `T.tanh(x)` | Software | float16, float32 |
| `T.rsqrt` | `T.rsqrt(x)` | `rsqrt.approx` | float16, float32 |
| `T.sqrt` | `T.sqrt(x)` | `sqrt.approx` | float16, float32 |
| `T.abs` | `T.abs(x)` | `abs.ftz` | float16, float32, int32 |
| `T.neg` | `T.neg(x)` | `neg.ftz` | float16, float32 |
| `T.max` | `T.max(a, b)` | `max.ftz` | float16, float32 |
| `T.min` | `T.min(a, b)` | `min.ftz` | float16, float32 |
| `T.clamp` | `T.clamp(x, lo, hi)` | `max(min(...))` | float16, float32 |
| `T.floor` | `T.floor(x)` | `cvt.rni` | float16, float32 |
| `T.ceil` | `T.ceil(x)` | `cvt.rpi` | float16, float32 |
| `T.pow` | `T.pow(b, e)` | `ex2(lg2 * e)` | float16, float32 |
| `T.silu` | `T.silu(x)` | `x / (1 + exp(-x))` | float16, float32 |
| `T.sigmoid` | `T.sigmoid(x)` | `1 / (1 + exp(-x))` | float16, float32 |
| `T.relu` | `T.relu(x)` | `max(0, x)` | float16, float32 |
| `T.gelu` | `T.gelu(x)` | Software | float16, float32 |

---

## T.clear -- Buffer Clearing

### Signature

```python
T.clear(buffer)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `buffer` | Buffer | The buffer to clear. All elements will be set to zero. Supports shared, local (register), and global buffers. |

### Overview

`T.clear` zeroes all elements in a buffer. It is commonly used to initialize accumulators before GEMM operations. The implementation uses optimized vectorized store instructions (e.g., `st.shared.v4.f32` for shared memory, or `st.local.v4.f32` for registers).

### Example

```python
# Clear register accumulator before GEMM
C_local = T.alloc_local([128, 128], "float32")
T.clear(C_local)

# Clear shared memory buffer
smem = T.alloc_shared([64, 64], "float16")
T.clear(smem)

# Clear can also be used as an alternative to clear_accum=True in T.gemm:
T.clear(C_local)
for k in range(0, K, block_K):
    T.copy(A[k], A_smem)
    T.copy(B[k], B_smem)
    T.sync_shared_memory()
    T.gemm(A_smem, B_smem, C_local)  # First iteration accumulates into cleared buffer
```

### Performance Notes

- For large shared memory buffers, `T.clear` uses vectorized stores and is significantly faster than a loop-based clear.
- For register buffers, `T.clear` compiles to `mov.f32 R, 0` instructions.
- `T.clear` is automatically inserted by `T.gemm` when `clear_accum=True`, but explicit use gives more control over placement.

---

## Accumulator Management

Accumulator management is a critical aspect of high-performance GEMM in TileLang. The accumulator buffer C holds partial sums that are built up over multiple K-dimension iterations.

### Accumulator Data Types

The accumulator type should always be wider than or equal to the input type to prevent numerical overflow:

| Input Type | Recommended Accumulator | Rationale |
|-----------|------------------------|-----------|
| float16 | float32 | Prevent overflow in sum of 1024+ products |
| bfloat16 | float32 | Same as float16 |
| tf32 | float32 | Natural pairing |
| int8 | int32 | Prevent overflow (127 * 127 * K) |
| float8_e4m3 | float32 | Narrow mantissa requires wider accumulation |
| float4_e2m1fn | float32 | Very narrow type, accumulation essential |

### Accumulator Clearing Strategies

```python
# Strategy 1: Use clear_accum parameter (recommended for first K iteration)
T.gemm(A, B, C, clear_accum=True)  # First iteration

# Strategy 2: Explicit clear before the loop
T.clear(C)
for k in range(0, K, block_K):
    T.gemm(A[k], B[k], C)  # All iterations accumulate

# Strategy 3: Clear inside loop with conditional
T.clear(C)
for k in range(0, K, block_K):
    if k > 0:
        T.gemm(A[k], B[k], C)  # Accumulate
    else:
        T.gemm(A[k], B[k], C, clear_accum=True)  # First iter clears
```

### Mixed Precision Patterns

TileLang supports flexible mixed-precision GEMM where input and accumulation types differ:

```python
# float16 input with float32 accumulation (most common for deep learning)
A = T.alloc_shared([M, K], "float16")
B = T.alloc_shared([K, N], "float16")
C = T.alloc_local([M, N], "float32")

# int8 quantized input with int32 accumulation
A = T.alloc_shared([M, K], "int8")
B = T.alloc_shared([K, N], "int8")
C = T.alloc_local([M, N], "int32")

# float8 mixed: e4m3 for A, e5m2 for B
A = T.alloc_shared([M, K], "float8_e4m3")
B = T.alloc_shared([K, N], "float8_e5m2")
C = T.alloc_local([M, N], "float32")
```

---

## Practical Examples

### Complete Flash Attention with T.gemm

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])
def flash_attention(
    batch, heads, seq_len, dim,
    block_M=64, block_N=64,
    in_dtype="float16", out_dtype="float16",
):
    # Shared memory for Q, K, V tiles
    Q_smem = T.alloc_shared([block_M, dim], in_dtype)
    K_smem = T.alloc_shared([block_N, dim], in_dtype)
    V_smem = T.alloc_shared([block_N, dim], in_dtype)

    # Local (register) buffers
    S_local = T.alloc_local([block_M, block_N], "float32")  # Attention scores
    O_local = T.alloc_local([block_M, dim], "float32")      # Output accumulator
    m_local = T.alloc_local([block_M], "float32")            # Running max
    l_local = T.alloc_local([block_M], "float32")            # Running sum

    # Initialize running stats
    T.clear(m_local)
    T.clear(l_local)
    T.clear(O_local)

    # Load Q tile (once, reused across K/V iterations)
    T.copy(Q_global, Q_smem)
    T.sync_shared_memory()

    for n in range(0, seq_len, block_N):
        # Load K, V tiles
        T.copy(K_global[n:n+block_N], K_smem)
        T.copy(V_global[n:n+block_N], V_smem)
        T.sync_shared_memory()

        # Compute S = Q @ K^T
        T.gemm(Q_smem, K_smem, S_local, transpose_B=True, clear_accum=True)

        # Online softmax update
        S_max = T.reduce_max(S_local, dim=-1)
        S_new_max = T.max(m_local, S_max)
        correction = T.exp(m_local - S_new_max)
        T.clear(l_local)
        l_local = l_local * correction + T.reduce_sum(T.exp(S_local - S_new_max), dim=-1)

        # Update output accumulator
        O_local = O_local * correction

        # Compute O += softmax(S) @ V
        T.gemm(S_local, V_smem, O_local)  # Accumulates into O

        m_local = S_new_max

    # Final normalization
    O_local = O_local / l_local
    T.copy(O_local, O_global)
    return O_global
```

### Complete GEMM with Software Pipelining

```python
import tilelang
import tilelang.language as T
from tilelang.language import GemmWarpPolicy

@tilelang.jit(out_idx=[2])
def pipelined_matmul(
    M, N, K,
    block_M=128, block_N=128, block_K=32,
    in_dtype="float16", out_dtype="float16",
    num_stages=3,
):
    A_smem = T.alloc_shared([num_stages, block_M, block_K], in_dtype)
    B_smem = T.alloc_shared([num_stages, block_K, block_N], in_dtype)
    C_local = T.alloc_local([block_M, block_N], "float32")

    T.clear(C_local)

    # Use software pipelining to overlap memory and compute
    ko = T.Pipelined(
        start=0, stop=K // block_K, num_stages=num_stages,
        order=T.Layout.kOuterInner,
    )
    for k in ko:
        # Stage: async copy next tile while computing current tile
        T.copy(A_global[k * block_K:(k+1) * block_K], A_smem[k % num_stages])
        T.copy(B_global[k * block_K:(k+1) * block_K], B_smem[k % num_stages])
        T.sync_shared_memory()

        T.gemm(
            A_smem[k % num_stages],
            B_smem[k % num_stages],
            C_local,
            policy=GemmWarpPolicy.Square,
        )

    T.copy(C_local, C_global)
    return C_global
```

### Element-wise Operations with GEMM

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])
def fused_gemm_silu(
    M, N, K, block_M, block_N, block_K,
    in_dtype="float16",
):
    A_smem = T.alloc_shared([block_M, block_K], in_dtype)
    B_smem = T.alloc_shared([block_K, block_N], in_dtype)
    C_local = T.alloc_local([block_M, block_N], "float32")

    T.clear(C_local)

    for k in range(0, K, block_K):
        T.copy(A_global[k:k+block_K], A_smem)
        T.copy(B_global[k:k+block_K], B_smem)
        T.sync_shared_memory()
        T.gemm(A_smem, B_smem, C_local)

    # Apply SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
    neg_c = T.neg(C_local)
    exp_neg = T.exp(neg_c)
    one_plus_exp = exp_neg + 1.0
    sigmoid_c = 1.0 / one_plus_exp
    result = C_local * sigmoid_c

    T.copy(result, C_global)
    return C_global
```

### Batched GEMM with Reduction

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])
def batched_matmul(
    batch, M, N, K, block_M, block_N, block_K,
    in_dtype="float16",
):
    A_smem = T.alloc_shared([block_M, block_K], in_dtype)
    B_smem = T.alloc_shared([block_K, block_N], in_dtype)
    C_local = T.alloc_local([block_M, block_N], "float32")

    T.clear(C_local)

    # Iterate over batch dimension
    for b in range(batch):
        # Iterate over K dimension
        for k in range(0, K, block_K):
            T.copy(A_global[b, k:k+block_K], A_smem)
            T.copy(B_global[b, k:k+block_K], B_smem)
            T.sync_shared_memory()
            T.gemm(A_smem, B_smem, C_local)

        # Store this batch's result and clear for next
        T.copy(C_local, C_global[b])
        T.clear(C_local)

    return C_global
```

---

## Summary

| Primitive | Architecture | Key Feature |
|-----------|-------------|-------------|
| `T.gemm` | SM 70+ | Auto-selecting tensor core GEMM |
| `T.wgmma_gemm` | SM 90+ | Explicit async WGMMA |
| `T.tcgen05_gemm` | SM 100+ | Blackwell TCGEN05 |
| `T.tcgen05_gemm_blockscaled` | SM 100+ | MX-format block-scaled GEMM |
| `T.gemm_sp` | SM 80+ | 2:4 structured sparsity |
| `T.gemm_sp_v2` | SM 80+ | Improved sparse GEMM |
| `T.exp`, `T.log`, etc. | All | Element-wise math |
| `T.clear` | All | Buffer zeroing |

The compute primitives in TileLang are designed to provide both high-level convenience (via `T.gemm` with automatic hardware selection) and low-level control (via architecture-specific primitives like `T.wgmma_gemm` and `T.tcgen05_gemm`). This allows developers to write portable kernels that run efficiently across GPU generations while retaining the ability to exploit architecture-specific features when needed.
