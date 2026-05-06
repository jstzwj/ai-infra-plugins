# Tensor Core Operations

## Table of Contents

- [1. Tensor Core Overview](#1-tensor-core-overview)
- [2. Operation Classes](#2-operation-classes)
- [3. WMMA Operations (SM70+)](#3-wmma-operations-sm70)
- [4. MMA Operations (SM75+)](#4-mma-operations-sm75)
- [5. GMMA Operations (SM90)](#5-gmma-operations-sm90)
- [6. UMMA Operations (SM100)](#6-umma-operations-sm100)
- [7. Data Type Support per Operation](#7-data-type-support-per-operation)
- [8. Math Operators](#8-math-operators)
- [9. Code Examples per Architecture](#9-code-examples-per-architecture)

---

## 1. Tensor Core Overview

Tensor Cores are specialized hardware units in NVIDIA GPUs that perform matrix multiply-accumulate operations at significantly higher throughput than scalar (SIMT) CUDA cores. Since their introduction in the Volta architecture (SM70), Tensor Cores have evolved through several generations, each adding new data types, larger matrix sizes, and more flexible programming models.

### Tensor Core Evolution

| Architecture | SM Version | Tensor Core Gen | Key Instruction | Max Shape (MxNxK) |
|---|---|---|---|---|
| Volta | SM70 | 1st Gen | WMMA | 16x16x16 |
| Turing | SM75 | 2nd Gen | MMA | 16x8x16 (FP16) |
| Ampere | SM80/SM89 | 3rd Gen | MMA | 16x8x32 (FP16) |
| Hopper | SM90 | 4th Gen | GMMA/WGMMA | 64x8xK (async) |
| Blackwell | SM100+ | 5th Gen | UMMA | 64x8x256 (block-scaled) |

### Performance Characteristics

Tensor Cores provide dramatically higher throughput compared to SIMT CUDA cores:

| GPU | FP16 Tensor Core | FP32 CUDA Core | Ratio |
|---|---|---|---|
| V100 (SM70) | 125 TFLOPS | 15.7 TFLOPS | ~8x |
| A100 (SM80) | 312 TFLOPS | 19.5 TFLOPS | ~16x |
| H100 (SM90) | 989 TFLOPS | 67 TFLOPS | ~15x |
| B200 (SM100) | 2250 TFLOPS | 90 TFLOPS | ~25x |

### Programming Models

CUTLASS supports three programming models for Tensor Cores:

1. **WMMA API (SM70+)**: High-level warp-level API using `wmma::fragment` types. Simple to use but less flexible.
2. **MMA Inline Assembly (SM75+)**: Low-level inline assembly instructions with direct control over register allocation. More flexible, higher performance.
3. **CuTe Atoms (SM75+)**: CUTLASS's composable abstraction over hardware instructions. Recommended for new code.

---

## 2. Operation Classes

CUTLASS defines operation class tags that determine whether a GEMM operation uses Tensor Cores or SIMT (scalar) CUDA cores.

### OpClassSimt

SIMT (Single-Instruction Multiple-Thread) operations use standard CUDA cores:

```cpp
namespace cutlass::arch {
struct OpClassSimt {
    static constexpr int ThreadsPerWarp = 32;
    // Each thread computes one or more scalar multiply-accumulate operations
    // No hardware matrix instruction is used
};
}
```

Characteristics:
- Each thread performs scalar multiply-add operations
- Supports all data types
- Lower throughput than Tensor Cores
- More flexible data layout requirements
- Used as fallback when Tensor Cores don't support a given data type

### OpClassTensorOp

Tensor Core operations using MMA instructions:

```cpp
namespace cutlass::arch {
struct OpClassTensorOp {
    // Uses hardware Tensor Core instructions (MMA, GMMA, UMMA)
    // Warp-level cooperative matrix multiply
};
}
```

Characteristics:
- Uses hardware Tensor Core matrix instructions
- Much higher throughput for supported data types
- Specific data layout requirements (fragment layouts)
- Supports FP16, BF16, TF32, FP8, INT8, INT4, FP64

### OpClassWmmaTensorOp

Tensor Core operations using the WMMA API:

```cpp
namespace cutlass::arch {
struct OpClassWmmaTensorOp {
    // Uses WMMA (Warp Matrix Multiply-Accumulate) API
    // Available on SM70+
};
}
```

Characteristics:
- Uses `nvcuda::wmma` API
- Available on all Tensor Core-capable architectures
- Simpler programming model but less control over register layout
- Supported shapes: 16x16x16, 32x8x16, 8x32x16

### OpClassSparseTensorOp

Tensor Core operations with structured sparsity support:

```cpp
namespace cutlass::arch {
struct OpClassSparseTensorOp {
    // Uses Tensor Core instructions with 2:4 structured sparsity
    // Only the non-zero elements are stored and processed
};
}
```

Characteristics:
- Supports 2:4 structured sparsity (2 non-zero per 4-element group)
- Available on SM80+ (Ampere and later)
- Doubles effective throughput for sparse matrices
- Operand A is stored in compressed format with metadata

---

## 3. WMMA Operations (SM70+)

The WMMA (Warp Matrix Multiply-Accumulate) API was the first Tensor Core programming interface, introduced with Volta (SM70).

### WMMA Fragment Types

WMMA uses `fragment` types to store matrix data in registers:

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// Fragment types for matrix A, B, and accumulator
fragment<matrix_a, 16, 16, 16, half, row_major> frag_a;
fragment<matrix_b, 16, 16, 16, half, col_major> frag_b;
fragment<accumulator, 16, 16, 16, float> frag_c;
```

Fragment type parameters:
- `matrix_a` / `matrix_b` / `accumulator`: Role of the fragment
- `16, 16, 16`: M, N, K dimensions
- `half` / `float`: Data type
- `row_major` / `col_major`: Memory layout

### Supported WMMA Shapes

```cpp
// Volta (SM70): FP16 only
// 16x16x16
fragment<matrix_a, 16, 16, 16, half, row_major> a_16x16;
fragment<matrix_b, 16, 16, 16, half, col_major> b_16x16;
fragment<accumulator, 16, 16, 16, float> c_16x16;

// 32x8x16
fragment<matrix_a, 32, 8, 16, half, row_major> a_32x8;

// 8x32x16
fragment<matrix_a, 8, 32, 16, half, row_major> a_8x32;
```

### load_matrix_sync

Loads a matrix tile from memory into a fragment:

```cpp
// Load matrix A from shared memory
load_matrix_sync(frag_a, smem_ptr_a, K_stride);  // K is the leading dimension

// Load matrix B from shared memory
load_matrix_sync(frag_b, smem_ptr_b, N_stride);  // N is the leading dimension

// Load accumulator C from shared memory
load_matrix_sync(frag_c, smem_ptr_c, N_stride, mem_row_major);
```

The `load_matrix_sync` function:
- Synchronizes all threads in the warp
- Cooperatively loads the matrix tile from shared/global memory
- Distributes data across warp threads' registers

### store_matrix_sync

Stores a fragment to memory:

```cpp
// Store accumulator to shared memory
store_matrix_sync(smem_ptr_c, frag_c, N_stride, mem_row_major);
```

### fill_fragment

Fills a fragment with a constant value:

```cpp
// Clear accumulator
fill_fragment(frag_c, 0.0f);
```

### mma_sync

Performs the matrix multiply-accumulate:

```cpp
// C += A * B
mma_sync(frag_c, frag_a, frag_b, frag_c);
```

### Complete WMMA Example

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void wmma_gemm(half const* A, half const* B, float* C,
                           int M, int N, int K) {
    // WMMA tile size
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    // Fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_a;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> frag_b;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;

    // Initialize accumulator
    fill_fragment(frag_c, 0.0f);

    // Compute tile coordinates
    int warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warp_n = (blockIdx.y * blockDim.y + threadIdx.y) / 32;

    if (warp_m * WMMA_M < M && warp_n * WMMA_N < N) {
        // GEMM loop over K
        for (int k = 0; k < K; k += WMMA_K) {
            // Load A tile
            load_matrix_sync(frag_a, A + warp_m * WMMA_M * K + k, K);
            // Load B tile
            load_matrix_sync(frag_b, B + k * N + warp_n * WMMA_N, N);
            // MMA
            mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        // Store result
        store_matrix_sync(C + warp_m * WMMA_M * N + warp_n * WMMA_N,
                          frag_c, N, mem_row_major);
    }
}
```

### WMMA Data Types (SM70-SM75)

| Data Type | Input A | Input B | Accumulator | SM Version |
|---|---|---|---|---|
| FP16-FP16-FP16 | half | half | half | SM70 |
| FP16-FP16-FP32 | half | half | float | SM70 |
| Mixed INT8 | int8 | int8 | int32 | SM75 |
| Mixed INT4 | int4 | int4 | int32 | SM75 |
| Mixed b1 | bit | bit | int32 | SM75 |

### WMMA Limitations

- Fixed fragment layout (no control over register assignment)
- Only 16x16x16, 32x8x16, 8x32x16 shapes
- Higher register pressure than MMA inline assembly
- Less efficient than direct MMA instructions (SM75+)

---

## 4. MMA Operations (SM75+)

MMA (Matrix Multiply-Accumulate) inline assembly instructions provide finer-grained control over Tensor Core operations, replacing WMMA for SM75 and later architectures.

### MMA Instruction Overview

MMA instructions operate at the warp level with explicit register assignments:

```
mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32
    {d0, d1, d2, d3},
    {a0, a1},
    {b0, b1},
    {c0, c1, c2, c3};
```

- `m8n8k4`: Shape (8 rows, 8 columns, 4 inner)
- `row.col`: A is row-major, B is column-major
- `f32.f16.f16.f32`: D=FP32, A=FP16, B=FP16, C=FP32
- `{d0..d3}`: Output registers (4 FP32 values)
- `{a0, a1}`: A input registers (2 FP16 pairs)
- `{b0, b1}`: B input registers (2 FP16 pairs)
- `{c0..c3}`: Accumulator registers (4 FP32 values)

### Supported Shapes

#### SM75 (Turing) MMA Shapes

| Shape | Data Types | Instruction |
|---|---|---|
| 8x8x4 | FP16 x FP16 -> FP16 | `mma.sync.aligned.m8n8k4` |
| 8x8x4 | FP16 x FP16 -> FP32 | `mma.sync.aligned.m8n8k4` |
| 16x8x8 | FP16 x FP16 -> FP16 | `mma.sync.aligned.m16n8k8` |
| 16x8x8 | FP16 x FP16 -> FP32 | `mma.sync.aligned.m16n8k8` |
| 16x8x16 | FP16 x FP16 -> FP16 | `mma.sync.aligned.m16n8k16` |
| 16x8x16 | FP16 x FP16 -> FP32 | `mma.sync.aligned.m16n8k16` |
| 16x8x16 | INT8 x INT8 -> INT32 | `mma.sync.aligned.m16n8k16` |
| 16x8x32 | INT4 x INT4 -> INT32 | `mma.sync.aligned.m16n8k32` |
| 16x8x64 | INT1 x INT1 -> INT32 | `mma.sync.aligned.m16n8k64` |
| 8x8x16 | INT8 x INT8 -> INT32 | `mma.sync.aligned.m8n8k16` |

#### SM80 (Ampere) Additional Shapes

| Shape | Data Types | Instruction |
|---|---|---|
| 16x8x4 | TF32 x TF32 -> FP32 | `mma.sync.aligned.m16n8k4` |
| 16x8x8 | TF32 x TF32 -> FP32 | `mma.sync.aligned.m16n8k8` |
| 16x8x16 | BF16 x BF16 -> FP32 | `mma.sync.aligned.m16n8k16` |
| 16x8x32 | FP16 x FP16 -> FP16 | `mma.sync.aligned.m16n8k32` |
| 16x8x32 | FP16 x FP16 -> FP32 | `mma.sync.aligned.m16n8k32` |
| 16x8x32 | INT8 x INT8 -> INT32 | `mma.sync.aligned.m16n8k32` |
| 16x8x64 | INT4 x INT4 -> INT32 | `mma.sync.aligned.m16n8k64` |
| 16x8x128 | INT1 x INT1 -> INT32 | `mma.sync.aligned.m16n8k128` |
| 8x8x4 | FP64 x FP64 -> FP64 | `mma.sync.aligned.m8n8k4` |
| 16x8x16 | FP16 x FP16 -> FP32 (sparse) | `mma.sync.aligned.m16n8k16` |

### Fragment Layouts per Architecture

The fragment layout determines how data is stored in registers for each thread in the warp.

#### FP16 MMA Fragment Layout (SM75/SM80, 16x8x16)

For a 16x8x16 FP16 MMA with FP32 accumulation:

**A Fragment (16xK, row-major)**:
- 32 threads in a warp
- 4 threads per group along M
- Each thread holds: 2 registers x uint32_t (packed FP16 pairs)
- Total per thread: 8 FP16 values (4 values per register, 2 registers)

```
Thread layout for A:
  Thread 0: rows 0-3, columns 0-1
  Thread 1: rows 4-7, columns 0-1
  Thread 2: rows 8-11, columns 0-1
  Thread 3: rows 12-15, columns 0-1
  ...
```

**B Fragment (Kx8, column-major)**:
- 4 threads per group along N
- Each thread holds: 2 registers x uint32_t (packed FP16 pairs)
- Total per thread: 4 FP16 values

**C Fragment (16x8, accumulator)**:
- Each thread holds: 4 FP32 registers
- Arrangement: 2 rows x 2 columns per thread

#### TF32 MMA Fragment Layout (SM80, 16x8x8)

**A Fragment**:
- Each thread holds: 2 TF32 values per register (packed as uint32_t)
- 4 threads along M, multiple groups

**B Fragment**:
- Each thread holds: 2 TF32 values per register
- 4 threads along N

**C Fragment**:
- 4 FP32 accumulators per thread

#### BF16 MMA Fragment Layout (SM80, 16x8x16)

Similar to FP16 layout but using BF16 data type:
- Packed as uint32_t (2 BF16 per register)
- Same thread-value mapping as FP16

#### FP64 MMA Fragment Layout (SM80, 8x8x4)

**A Fragment**:
- Each thread holds: 1 FP64 value
- 8 threads along M
- 4 groups along K

**B Fragment**:
- Each thread holds: 1 FP64 value
- 8 threads along N

**C Fragment**:
- Each thread holds: 1 FP64 accumulator

#### INT8 MMA Fragment Layout (SM80, 16x8x32)

**A Fragment**:
- Packed as uint32_t (4 INT8 per register)
- Each thread holds: 4 registers = 16 INT8 values

**B Fragment**:
- Packed as uint32_t (4 INT8 per register)
- Each thread holds: 2 registers = 8 INT8 values

### Sparse MMA Fragment Layouts (SM80)

For 2:4 structured sparsity, the A fragment is compressed:
- Only 2 of every 4 elements are stored (non-zero)
- Metadata (2 bits per 4-element group) indicates which elements are non-zero
- Fragment size is halved compared to dense MMA

---

## 5. GMMA Operations (SM90)

SM90 (Hopper) introduces GMMA (Group Matrix Multiply-Accumulate), also known as WGMMA (Warp Group MMA). GMMA is a fundamentally different programming model:

### WGMMA Characteristics

- **Asynchronous**: The MMA operation is started asynchronously; results are read after a fence.
- **Warp group execution**: Operates across 4 warps (128 threads) instead of a single warp (32 threads).
- **Larger M dimension**: Supports M=16, 32, 64 in a single instruction.
- **Shared memory source**: B operand can come directly from shared memory without explicit register loading.
- **Descriptor-based**: Uses a matrix descriptor for the shared memory operand.

### WGMMA Instructions

```asm
wgmma.mma_async.aligned.m64n8k16
    {%0, %1, %2, %3},
    %4,
    {%5, %6, %7, %8, %9, %10, %11, %12},
    %13,
    %14, %15, %16, %17;
```

Where:
- M=64, N=8, K=16 (or K=32, K=64 for other data types)
- First operand group: accumulator registers
- Second operand: A matrix descriptor (shared or register)
- Third operand group: B matrix data (shared memory)
- Fourth operand: scale factor

### GMMA Shapes

| Shape (MxNxK) | A Type | B Type | C/D Type | Notes |
|---|---|---|---|---|
| 64x8x16 | FP16 | FP16 | FP32 | Register A |
| 64x8x32 | FP16 | FP16 | FP32 | Register A |
| 64x8x8 | TF32 | TF32 | FP32 | Register A |
| 64x8x16 | BF16 | BF16 | FP32 | Register A |
| 64x8x32 | FP8 E4M3 | FP8 E5M2 | FP32 | Register A |
| 64x8x32 | FP8 E5M2 | FP8 E4M3 | FP32 | Register A |
| 64x8x64 | FP8 E4M3 | FP8 E4M3 | FP32 | Register A |
| 64x8x64 | INT8 | INT8 | INT32 | Register A |
| 64x8x128 | INT4 | INT4 | INT32 | Register A |
| 8x8x4 | FP64 | FP64 | FP64 | Register A |

GMMA also supports M=16 and M=32:

| Shape | M Values | Notes |
|---|---|---|
| Mx8xK | 16, 32, 64 | M dimension varies |
| Nx8 | N=8 fixed | Multiple N tiles combined for larger N |

### GMMA A Operand

The A operand can come from:
1. **Registers**: Loaded explicitly by the warp group into register fragments
2. **Shared memory descriptor**: (Future architecture support)

For register-sourced A:
```cpp
// Each thread in the warp group holds A data in registers
// The layout is determined by the GMMA atom's AtomLayoutA_TV
// For M=64, K=16 FP16: 128 threads each hold 8 FP16 values
```

### GMMA B Operand

The B operand comes directly from shared memory via a matrix descriptor:

```cpp
// Create a shared memory matrix descriptor for B
// The descriptor encodes the base address, stride, and layout
auto smem_desc_b = make_smem_desc(smem_ptr_b, stride_k, stride_n, 0);

// The descriptor is used by the WGMMA instruction to fetch B data
// No explicit register loading needed for B
```

### Async Matrix Multiply

GMMA is asynchronous, requiring fence and wait operations:

```cpp
// Start async MMA
wgmma_mma_async(tiled_mma, rA, sB, rC);

// Issue fence to commit all pending WGMMA operations
wgmma_fence();

// Wait for all WGMMA operations to complete
wgmma_wait_group<0>();
// Template parameter: wait for N or fewer pending groups
// 0 = wait for all to complete
```

### GMMA in CuTe

```cpp
#include "cute/arch/mma_sm90.hpp"

// SM90 GMMA atom
auto gmma_atom = MMA_Atom<SM90_64x8x16_F16F16F32F32_TN_GMMA>{};

// Tile across warp group
auto tiled_mma = make_tiled_mma(
    gmma_atom,
    Layout<Shape<_1, _8, _1>>{}  // 8x tiling along N for 64x64 output
);

// The MMA is asynchronous
// Issue: gemm(tiled_mma, rA, sB, rC);
// Wait: wgmma_fence(); wgmma_wait_group<0>();
```

### WGMMA Register Layout

For M=64, N=8, K=16 FP16:

**A Register Layout (per warp group of 128 threads)**:
- 128 threads organized as 8 groups of 16 threads
- Each thread holds 4 registers (uint32_t) = 8 FP16 values
- Total A data: 128 threads * 8 FP16 = 1024 FP16 values = 64 * 16 (M * K)

**B Shared Memory Layout**:
- B is addressed by the WGMMA descriptor
- Must be in shared memory with proper alignment
- Layout: column-major by default

**C Accumulator Layout**:
- 128 threads, each holds 4 FP32 registers
- Thread t holds accumulators for row (t % 64) and column (t / 64 * 4 + lane_offset)
- Total: 64 * 8 = 512 FP32 accumulators

---

## 6. UMMA Operations (SM100)

SM100 (Blackwell) introduces UMMA (Unified MMA), the fifth generation of Tensor Core instructions.

### UMMA Characteristics

- **Unified interface**: Single API across all data types and operation modes
- **Block-scaled operations**: Native support for block-scaled floating point (NVFP4, MXFP4/6/8)
- **Green context support**: Can operate within green contexts for improved resource management
- **Higher throughput**: Up to 2250 TFLOPS on B200

### Block-Scaled Operations

UMMA introduces native support for block-scaled matrix multiplication where each block of elements has an associated scale factor:

```
A_scaled[i] = A_raw[i] * scale_A[block_idx]
B_scaled[j] = B_raw[j] * scale_B[block_idx]
C[i,j] += sum_k(A_scaled[i,k] * B_scaled[k,j])
```

Scale factors are stored separately and applied during the MMA operation:

```cpp
// Block-scaled UMMA
// A: NVFP4 (4-bit) with scale factors
// B: NVFP4 (4-bit) with scale factors
// C: FP32 accumulator
struct SM100_64x8x256_NVFP4NVFP4F32F32_TN_UMMA {
    using Shape_MNK = Shape<_64, _8, _256>;
    using ValTypeA = float_nvfp4_t;        // 4-bit values
    using ValTypeB = float_nvfp4_t;        // 4-bit values
    using ScaleTypeA = float;               // Scale factor type
    using ScaleTypeB = float;               // Scale factor type
    using ValTypeC = float;                 // Accumulator type
    // K=256 because 256 NVFP4 values = 128 bytes (same footprint as 64 FP16)
};
```

### Scale Factor Handling

Scale factors are passed as separate operands to the UMMA instruction:

```cpp
// UMMA with block scaling
// A values (NVFP4): 64x256 = 16384 values, each 4 bits = 8192 bytes
// A scale factors: 64x8 = 512 FP16 values (one per 32-element block along K)
// B values (NVFP4): 256x8 = 2048 values, each 4 bits = 1024 bytes
// B scale factors: 8x8 = 64 FP16 values (one per 32-element block along K)

// In CuTe:
auto tCrA_val = ...;    // Tensor of NVFP4 values
auto tCrA_scale = ...;  // Tensor of scale factors
auto tCrB_val = ...;    // Tensor of NVFP4 values
auto tCrB_scale = ...;  // Tensor of scale factors

// UMMA operation uses both values and scales
gemm(tiled_UMMA, tCrA_val, tCrA_scale, tCrB_val, tCrB_scale, rC);
```

### UMMA Shapes

| Shape (MxNxK) | A Type | B Type | C/D Type | Scale Type |
|---|---|---|---|---|
| 64x8x16 | FP16 | FP16 | FP32 | N/A |
| 64x8x16 | BF16 | BF16 | FP32 | N/A |
| 64x8x32 | FP8 E4M3 | FP8 E4M3 | FP32 | N/A |
| 64x8x256 | NVFP4 | NVFP4 | FP32 | FP16/FP32 |
| 64x8x64 | MXFP8 | MXFP8 | FP32 | FP8 |
| 64x8x128 | MXFP6 | MXFP6 | FP32 | FP8 |
| 64x8x256 | MXFP4 | MXFP4 | FP32 | FP8 |
| 64x8x64 | INT8 | INT8 | INT32 | N/A |
| 8x8x4 | FP64 | FP64 | FP64 | N/A |

---

## 7. Data Type Support per Operation

### FP16 x FP16 -> FP16/FP32

The most widely supported Tensor Core operation:

| Architecture | Instruction | Shape | Accumulator |
|---|---|---|---|
| SM70 | WMMA | 16x16x16, 32x8x16, 8x32x16 | FP16, FP32 |
| SM75 | MMA | 8x8x4, 16x8x8, 16x8x16 | FP16, FP32 |
| SM80 | MMA | All SM75 + 16x8x32 | FP16, FP32 |
| SM90 | GMMA | 64x8x16, 64x8x32, 32x8x*, 16x8x* | FP32 |
| SM100 | UMMA | 64x8x16 | FP32 |

```cpp
// CUTLASS 2.x example
using GemmF16F16F32 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,                           // Accumulator
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>
>;
```

### BF16 x BF16 -> FP32

Brain float 16 (BF16) has the same exponent range as FP32 but reduced mantissa:

| Architecture | Instruction | Shape |
|---|---|---|
| SM80 | MMA | 16x8x16 |
| SM90 | GMMA | 64x8x16 |
| SM100 | UMMA | 64x8x16 |

```cpp
using GemmBF16 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>
>;
```

### TF32 x TF32 -> FP32

TF32 (TensorFloat-32) uses FP32 format with reduced mantissa (10 bits instead of 23):

| Architecture | Instruction | Shape |
|---|---|---|
| SM80 | MMA | 16x8x4, 16x8x8 |
| SM90 | GMMA | 64x8x8 |

```cpp
using GemmTF32 = cutlass::gemm::device::Gemm<
    cutlass::tfloat32_t, cutlass::layout::RowMajor,
    cutlass::tfloat32_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>
>;
```

### FP64 x FP64 -> FP64

Double precision Tensor Core operations:

| Architecture | Instruction | Shape |
|---|---|---|
| SM80 | MMA | 8x8x4 |
| SM90 | GMMA | 8x8x4 |

```cpp
using GemmFP64 = cutlass::gemm::device::Gemm<
    double, cutlass::layout::RowMajor,
    double, cutlass::layout::ColumnMajor,
    double, cutlass::layout::RowMajor,
    double,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 16, 16>,
    cutlass::gemm::GemmShape<8, 8, 4>
>;
```

### INT8 x INT8 -> INT32

Integer 8-bit Tensor Core operations:

| Architecture | Instruction | Shape |
|---|---|---|
| SM75 | MMA | 16x8x16, 8x8x16 |
| SM80 | MMA | 16x8x32, 8x8x16 |
| SM90 | GMMA | 64x8x64 |

```cpp
using GemmINT8 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>
>;
```

### INT4 x INT4 -> INT32

4-bit integer Tensor Core operations:

| Architecture | Instruction | Shape |
|---|---|---|
| SM75 | MMA | 16x8x32 |
| SM80 | MMA | 16x8x64 |
| SM90 | GMMA | 64x8x128 |

### FP8 x FP8 -> FP32

FP8 (8-bit floating point) with two encodings:
- **E4M3**: 4-bit exponent, 3-bit mantissa (higher precision)
- **E5M2**: 5-bit exponent, 2-bit mantissa (larger range)

| Architecture | Instruction | Shape | A Type | B Type |
|---|---|---|---|---|
| SM90 | GMMA | 64x8x32 | FP8 E4M3 | FP8 E5M2 |
| SM90 | GMMA | 64x8x32 | FP8 E5M2 | FP8 E4M3 |
| SM90 | GMMA | 64x8x64 | FP8 E4M3 | FP8 E4M3 |
| SM100 | UMMA | 64x8x32 | FP8 E4M3 | FP8 E4M3 |

### Mixed Precision Variants

CUTLASS supports mixed precision where input types differ from accumulator/output types:

| Input A | Input B | Accumulator | Output | Architecture |
|---|---|---|---|---|
| FP16 | FP16 | FP32 | FP16 | SM70+ |
| FP16 | FP16 | FP32 | FP32 | SM70+ |
| BF16 | BF16 | FP32 | BF16 | SM80+ |
| BF16 | BF16 | FP32 | FP32 | SM80+ |
| TF32 | TF32 | FP32 | FP32 | SM80+ |
| FP8 | FP8 | FP32 | FP8 | SM90+ |
| FP8 | FP8 | FP32 | FP32 | SM90+ |
| FP8 | FP8 | FP32 | BF16 | SM90+ |
| INT8 | INT8 | INT32 | INT8 | SM75+ |
| INT8 | INT8 | INT32 | FP16 | SM75+ |
| NVFP4 | NVFP4 | FP32 | FP16 | SM100+ |
| MXFP4 | MXFP4 | FP32 | FP32 | SM100+ |

---

## 8. Math Operators

CUTLASS defines math operator tags that specify the arithmetic precision of the Tensor Core operation.

### OpMultiplyAdd

Standard multiply-add with full precision for the given data type:

```cpp
namespace cutlass::arch {
struct OpMultiplyAdd {
    // Standard multiplication: a * b
    // Full precision for the input type
};
}
```

This is the default operator for most data types. The multiplication is performed at the precision of the input type, and the accumulation is performed at the precision of the accumulator type.

### OpMultiplyAddFastF32

Reduced-precision FP32 multiplication using TF32 internally:

```cpp
namespace cutlass::arch {
struct OpMultiplyAddFastF32 {
    // FP32 inputs are truncated to TF32 (19-bit: 8 exp + 10 mantissa + 1 sign)
    // before multiplication
    // Accumulation is still FP32
    // Higher throughput than full FP32 multiply
};
}
```

This operator enables FP32 Tensor Core operations by internally converting inputs to TF32 format. It provides ~8x higher throughput than SIMT FP32 GEMM with minimal accuracy loss:

```cpp
// Fast FP32 GEMM using TF32 internally
using GemmFastF32 = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    cutlass::arch::OpMultiplyAddFastF32  // <-- Fast F32 operator
>;
```

### OpMultiplyAddFastBF16

Reduced-precision BF16 multiplication:

```cpp
namespace cutlass::arch {
struct OpMultiplyAddFastBF16 {
    // BF16 inputs may use reduced precision internally
    // Higher throughput on some architectures
};
}
```

### OpMultiplyAddComplex

Complex number multiply-add:

```cpp
namespace cutlass::arch {
template <class RealOp>
struct OpMultiplyAddComplex {
    using Operator = RealOp;
    // Complex multiply: (a_r + i*a_i) * (b_r + i*b_b) = (a_r*b_r - a_i*b_i) + i*(a_r*b_i + a_i*b_r)
};
}
```

### OpMultiplyAddGaussianComplex

Gaussian complex multiplication (fewer multiplications):

```cpp
namespace cutlass::arch {
struct OpMultiplyAddGaussianComplex {
    // Uses Karatsuba-like decomposition:
    // k1 = a_r * b_r
    // k2 = a_i * b_i
    // k3 = (a_r + a_i) * (b_r + b_i)
    // result_real = k1 - k2
    // result_imag = k3 - k1 - k2
    // Uses 3 multiplications instead of 4
};
}
```

### Operator Selection by Architecture

| Operator | Supported Architectures | Input Types |
|---|---|---|
| OpMultiplyAdd | All | All |
| OpMultiplyAddFastF32 | SM80+ | FP32 (internally TF32) |
| OpMultiplyAddFastBF16 | SM80+ | BF16 |
| OpMultiplyAddComplex | SM80+ | Complex<FP32>, Complex<FP16> |
| OpMultiplyAddGaussianComplex | SM80+ | Complex<FP32>, Complex<FP16> |

---

## 9. Code Examples per Architecture

### SM70: WMMA FP16 GEMM

```cpp
#include <mma.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

using namespace nvcuda::wmma;

// CUTLASS 2.x WMMA GEMM
using GemmSM70 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassWmmaTensorOp,  // WMMA operation class
    cutlass::arch::Sm70,                  // Volta
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 16, 16>  // WMMA instruction shape
>;

// Launch
GemmSM70 gemm_op;
gemm_op({M, N, K}, alpha, ptr_A, lda, ptr_B, ldb, beta, ptr_C, ldc, ptr_D, ldd);
```

### SM75: MMA FP16 GEMM

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

using GemmSM75 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>  // MMA instruction shape
>;

GemmSM75 gemm_op;
gemm_op({M, N, K}, alpha, ptr_A, lda, ptr_B, ldb, beta, ptr_C, ldc, ptr_D, ldd);
```

### SM80: MMA BF16 GEMM with FP32 Accumulation

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

using GemmBF16SM80 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,                               // Accumulator
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,  // BF16 MMA shape
    cutlass::epilogue::thread::LinearCombination<
        float, 4, float, float>
>;

GemmBF16SM80 gemm_op;
gemm_op({M, N, K}, alpha, ptr_A, lda, ptr_B, ldb, beta, ptr_C, ldc, ptr_D, ldd);
```

### SM80: TF32 Fast FP32 GEMM

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

using GemmTF32SM80 = cutlass::gemm::device::Gemm<
    cutlass::tfloat32_t, cutlass::layout::RowMajor,
    cutlass::tfloat32_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,  // TF32 MMA shape
    cutlass::epilogue::thread::LinearCombination<
        float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    cutlass::arch::OpMultiplyAddFastF32  // TF32 precision
>;
```

### SM90: GMMA FP16 GEMM with CollectiveBuilder

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

// SM90 FP16 GEMM using CollectiveBuilder
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,     // A: FP16, RowMajor, 8-byte alignment
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,  // B: FP16, ColMajor, 8-byte alignment
    float,                                              // C: FP32
    cutlass::gemm::GemmShape<128, 128, 64>,             // Tile shape
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto       // Auto-select: TMA + GMMA
>::CollectiveOp;

using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::layout::RowMajor, cutlass::layout::RowMajor,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>;

using Kernel = cutlass::gemm::kernel::GemmUniversal<
    CollectiveOp, EpilogueOp
>;

using GemmSM90 = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

// Launch
GemmSM90 gemm_op;
typename GemmSM90::Arguments args{
    {M, N, K},
    {ptr_A, stride_A},
    {ptr_B, stride_B},
    {ptr_C, stride_C},
    {ptr_D, stride_D},
    {alpha, beta}
};
gemm_op(args);
```

### SM90: FP8 GEMM

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

// FP8 E4M3 x E5M2 -> FP32
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::float_e4m3_t, cutlass::layout::RowMajor, 16,
    cutlass::float_e5m2_t, cutlass::layout::ColumnMajor, 16,
    float,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

### SM90: FP64 Tensor Core GEMM

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

using GemmFP64SM90 = cutlass::gemm::device::Gemm<
    double, cutlass::layout::RowMajor,
    double, cutlass::layout::ColumnMajor,
    double, cutlass::layout::RowMajor,
    double,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm90,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 16, 16>,
    cutlass::gemm::GemmShape<8, 8, 4>  // FP64 MMA instruction shape
>;
```

### SM100: UMMA NVFP4 Block-Scaled GEMM

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

// NVFP4 block-scaled GEMM on Blackwell
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass::float_nvfp4_t, cutlass::layout::RowMajor, 32,
    cutlass::float_nvfp4_t, cutlass::layout::ColumnMajor, 32,
    float,
    cutlass::gemm::GemmShape<128, 128, 256>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
// Scale factors are handled automatically by the CollectiveBuilder
```

### SM80: Sparse 2:4 GEMM

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"

// 2:4 Structured sparsity on SM80
using GemmSparse = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,  // A (sparse)
    cutlass::half_t, cutlass::layout::ColumnMajor, // B (dense)
    float, cutlass::layout::RowMajor,              // C/D
    float,                                          // Accumulator
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,           // Sparse MMA shape (2x K)
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    cutlass::arch::OpMultiplyAdd,
    cutlass::gemm::kernel::GemmType::SparseGemm     // <-- Sparse GEMM type
>;
```

---

## Summary

Tensor Core operations are the foundation of high-performance matrix operations on NVIDIA GPUs:

1. **Operation classes** (Simt, TensorOp, WmmaTensorOp, SparseTensorOp) determine the hardware path.
2. **WMMA** (SM70+) provides a simple but less flexible API for Tensor Core access.
3. **MMA** (SM75+) provides fine-grained inline assembly control with explicit register layouts.
4. **GMMA/WGMMA** (SM90) introduces asynchronous warp-group-level operations with larger matrix sizes.
5. **UMMA** (SM100) adds block-scaled operations with unified data type support.
6. **Data type support** expands with each generation: FP16 -> BF16/TF32/INT8 -> FP8 -> NVFP4/MXFP.
7. **Math operators** control arithmetic precision: full precision (OpMultiplyAdd) or reduced precision (OpMultiplyAddFastF32) for higher throughput.
