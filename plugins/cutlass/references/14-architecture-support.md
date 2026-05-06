# CUTLASS - Chapter 14: Architecture Support

This reference covers all NVIDIA GPU architectures supported by CUTLASS, from Maxwell (SM50) through future architectures (SM120). Each architecture section details the supported data types, instruction sets, key features, and CUTLASS-specific capabilities.

---

## 14.1 Overview

CUTLASS provides architecture-specific optimizations for each supported NVIDIA GPU generation. The library uses template parameters and compile-time dispatch to select the optimal code path for each architecture while maintaining a unified API. The primary architecture parameter is the **SM version** (Streaming Multiprocessor version), which determines available hardware features.

**Architecture timeline and key capabilities:**

| Architecture | SM Version | Tensor Cores | Key Data Types | CUTLASS Support |
|---|---|---|---|---|
| Maxwell | SM50 | No | FP64, FP32, INT32 | CUTLASS 2.x (SIMT) |
| Pascal | SM60/SM61 | No | FP16 (storage) | CUTLASS 2.x |
| Volta | SM70 | 1st Gen (WMMA) | FP16 | CUTLASS 2.x |
| Turing | SM75 | 2nd Gen (MMA) | FP16, INT8, INT4, INT1, BF16 | CUTLASS 2.x |
| Ampere | SM80 | 3rd Gen (MMA) | FP16, BF16, TF32, INT8, INT4 | CUTLASS 2.x, 3.x |
| Ada | SM89 | 3rd Gen+ | + FP8 (E5M2, E4M3) | CUTLASS 2.x, 3.x |
| Hopper | SM90 | 4th Gen (GMMA) | + FP8, via TMA | CUTLASS 3.x primary |
| Blackwell | SM100/101/103 | 5th Gen (UMMA) | + NVFP4, MXFP4/6/8 | CUTLASS 3.x primary |
| Future | SM120 | TBD | TBD | CUTLASS 3.x |

---

## 14.2 SM50 (Maxwell)

### 14.2.1 Architecture Overview

Maxwell (GTX 900 series, Tesla M-series) introduced significant improvements in power efficiency and was the first architecture supported by early versions of CUTLASS.

**Key specifications:**

- 128 KB L2 cache per SM
- 64 KB shared memory per SM
- No Tensor Cores (SIMT-only computation)
- 128 CUDA cores per SM
- Compute capability 5.0

### 14.2.2 Supported Operations

CUTLASS on SM50 uses SIMT (Single-Instruction Multiple-Thread) operations for GEMM:

```cpp
// SM50 GEMM uses SIMT operations (no Tensor Cores)
using GemmSM50 = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,                               // Accumulator
    cutlass::arch::OpClassSimt,          // SIMT operation class
    cutlass::arch::Sm50,                 // Architecture
    cutlass::gemm::GemmShape<128, 128, 8>,  // Threadblock shape
    cutlass::gemm::GemmShape<32, 32, 8>,    // Warp shape (smaller for SIMT)
    cutlass::gemm::GemmShape<1, 1, 1>,      // Instruction shape (scalar per thread)
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>
>;
```

### 14.2.3 Data Types

| Type | Support | Notes |
|---|---|---|
| FP64 | Yes (SM50 only; limited rate) | Full-rate double precision on some models |
| FP32 | Yes | Primary compute type |
| INT32 | Yes | Integer arithmetic |
| FP16 | No | Not natively supported (software emulation only) |

### 14.2.4 Performance Characteristics

- SIMT GEMM relies on carefully tuned shared memory access patterns.
- No hardware matrix multiply -- each thread performs scalar multiply-accumulate.
- Threadblock shapes are typically smaller than Tensor Core variants.
- Performance is limited by instruction throughput rather than memory bandwidth for small matrices.

---

## 14.3 SM60 (Pascal)

### 14.3.1 Architecture Overview

Pascal (GTX 10-series, Tesla P100) introduced native FP16 storage and DP4A integer dot product instructions.

**Key specifications:**

- 64 KB shared memory per SM
- 4 MB L2 cache
- FP16 storage with FP32 compute (FP16 data is converted to FP32 for arithmetic)
- DP4A instruction: 4-element integer dot product

### 14.3.2 FP16 Support

Pascal supports FP16 as a storage type but performs arithmetic in FP32:

```cpp
// SM60 FP16 GEMM (FP16 storage, FP32 compute)
using GemmSM60FP16 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,     // A: FP16 storage
    cutlass::half_t, cutlass::layout::ColumnMajor,   // B: FP16 storage
    float, cutlass::layout::RowMajor,                // C: FP32 output
    float,                                           // Accumulator: FP32
    cutlass::arch::OpClassSimt,                      // SIMT (no Tensor Cores)
    cutlass::arch::Sm60,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<1, 1, 1>
>;
```

### 14.3.3 DP4A Integer Dot Product

The DP4A instruction computes a 4-element dot product with accumulation:

```cpp
// DP4A: dst = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + c
// Each element is INT8 (or UINT8), accumulated in INT32

using GemmDP4A = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,       // A: INT8
    int8_t, cutlass::layout::ColumnMajor,     // B: INT8
    int32_t, cutlass::layout::RowMajor,       // C: INT32
    int32_t,                                  // Accumulator: INT32
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm60,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<1, 1, 4>          // Instruction processes 4 INT8 elements
>;
```

### 14.3.4 Data Types

| Type | Support | Notes |
|---|---|---|
| FP64 | Yes | Full-rate on P100 |
| FP32 | Yes | Primary compute type |
| FP16 | Storage only | Arithmetic in FP32 |
| INT32 | Yes | General integer |
| INT8 | DP4A | 4-element dot product |

---

## 14.4 SM70 (Volta)

### 14.4.1 Architecture Overview

Volta (V100) introduced the first generation of Tensor Cores, providing hardware-accelerated matrix multiply-accumulate through the WMMA (Warp Matrix Multiply-Accumulate) API.

**Key specifications:**

- 128 KB shared memory per SM
- 6 MB L2 cache
- First Tensor Cores: 4x4 matrix multiply-accumulate in hardware
- WMMA API: 16x16x16, 32x8x16, 8x32x16 fragment shapes
- Independent thread scheduling

### 14.4.2 Tensor Core Operations (WMMA)

Volta Tensor Cores support FP16 matrix multiply with FP32 accumulation via the WMMA API:

```cpp
// SM70 FP16 Tensor Core GEMM using WMMA
using GemmSM70 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,                                   // Accumulator
    cutlass::arch::OpClassTensorOp,          // Tensor Core operation
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<128, 128, 32>,  // Threadblock shape
    cutlass::gemm::GemmShape<64, 64, 32>,    // Warp shape
    cutlass::gemm::GemmShape<8, 32, 16>,     // WMMA instruction shape
    cutlass::epilogue::thread::LinearCombination<float, 8, float, float>
>;

// Available WMMA instruction shapes on SM70:
// 16x16x16 (most common)
// 32x8x16
// 8x32x16
// All operate on FP16 inputs with FP32 accumulation
```

### 14.4.3 WMMA Fragment Layout

WMMA fragments have specific layouts that threads must respect:

```cpp
// WMMA fragment for m16n16k16 with FP16:
// Each warp (32 threads) jointly holds a 16x16 FP16 matrix tile
// Fragment A (16x16): each thread holds 8 FP16 values
// Fragment B (16x16): each thread holds 8 FP16 values
// Fragment C (16x16): each thread holds 4 FP32 values (accumulators)

// CUTLASS wraps WMMA in the arch::Mma operation:
#include "cutlass/arch/wmma.h"
cutlass::arch::Wmma<
    cutlass::gemm::GemmShape<16, 16, 16>,
    cutlass::half_t,    // A type
    cutlass::layout::RowMajor,
    cutlass::half_t,    // B type
    cutlass::layout::ColumnMajor,
    float,              // C type (accumulator)
    cutlass::layout::RowMajor
>::mma(accum, frag_A, frag_B, accum);
```

### 14.4.4 Data Types

| Type | Support | Tensor Core | Notes |
|---|---|---|---|
| FP64 | Yes | No | SIMT only |
| FP32 | Yes | No | SIMT only |
| FP16 | Yes | Yes (WMMA) | FP16 input, FP32 accumulate |
| INT32 | Yes | No | SIMT only |

---

## 14.5 SM75 (Turing)

### 14.5.1 Architecture Overview

Turing (RTX 20-series, T4) introduced the second generation of Tensor Cores with the MMA (Matrix Multiply-Accumulate) instruction, supporting a wider range of data types including INT8, INT4, and binary.

**Key specifications:**

- 64 KB shared memory per SM
- First MMA instructions (replacing WMMA)
- Integer Tensor Core support: INT8, INT4, INT1
- BF16 support (software level)
- DP4A instruction (from Pascal, improved)

### 14.5.2 Tensor Core MMA Instructions

Turing introduces the `mma.sync` instruction, which provides finer-grained control than WMMA:

```cpp
// SM75 FP16 Tensor Core GEMM
using GemmSM75_FP16 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,     // MMA instruction shape
    cutlass::epilogue::thread::LinearCombination<float, 8, float, float>
>;

// SM75 INT8 Tensor Core GEMM
using GemmSM75_INT8 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor,
    int32_t,                                // Accumulator: INT32
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,     // INT8 MMA shape
    cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, float>
>;
```

### 14.5.3 Available MMA Shapes

**FP16 on SM75:**
- `16x8x8`: 16 rows, 8 columns, 8 elements of K per instruction

**INT8 on SM75:**
- `8x8x16`: 8 rows, 8 columns, 16 INT8 elements of K

**INT4 on SM75:**
- `8x8x32`: 8 rows, 8 columns, 32 INT4 elements of K

**Binary (INT1) on SM75:**
- `8x8x128`: 8 rows, 8 columns, 128 binary elements of K

```cpp
// INT4 GEMM
using GemmINT4 = cutlass::gemm::device::Gemm<
    cutlass::int4b_t, cutlass::layout::RowMajor,
    cutlass::int4b_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<8, 8, 32>
>;

// Binary (1-bit) GEMM
using GemmBinary = cutlass::gemm::device::Gemm<
    cutlass::uint1b_t, cutlass::layout::RowMajor,
    cutlass::uint1b_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 512>,
    cutlass::gemm::GemmShape<64, 64, 512>,
    cutlass::gemm::GemmShape<8, 8, 128>
>;
```

### 14.5.4 dp4a Instruction

The DP4A instruction computes a 4-element dot product, useful for INT8 inference:

```cpp
#include "cutlass/arch/mma_sm50.h"  // DP4A defined here

// DP4A: c += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
// Available on SM60+ (Pascal and later)
// Still used on SM75 for non-Tensor-Core INT8 operations
```

---

## 14.6 SM80 (Ampere)

### 14.6.1 Architecture Overview

Ampere (A100, RTX 30-series) is a major architecture for CUTLASS, introducing TF32 Tensor Cores, BF16 support, and the `cp.async` instruction for pipelined memory operations.

**Key specifications:**

- 164 KB shared memory per SM (configurable)
- 40 MB L2 cache (A100)
- TF32 Tensor Cores: near-FP32 accuracy with Tensor Core throughput
- BF16 Tensor Cores
- `cp.async`: asynchronous global-to-shared memory copy
- 3rd-generation Tensor Cores with `mma.sync` instruction

### 14.6.2 TF32 Tensor Cores

TF32 (TensorFloat-32) uses 8-bit exponent (from FP32) and 10-bit mantissa for Tensor Core operations, providing near-FP32 accuracy at FP16-like throughput:

```cpp
// SM80 TF32 Tensor Core GEMM
using GemmSM80_TF32 = cutlass::gemm::device::Gemm<
    cutlass::tfloat32_t, cutlass::layout::RowMajor,   // A: TF32
    cutlass::tfloat32_t, cutlass::layout::ColumnMajor, // B: TF32
    float, cutlass::layout::RowMajor,                  // C: FP32
    float,                                             // Accumulator: FP32
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,    // TF32 MMA shape
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>
>;
```

**TF32 MMA shapes:**
- `16x8x4`: 16 M, 8 N, 4 K (4 TF32 elements per instruction per thread pair)
- `16x8x8`: 16 M, 8 N, 8 K (double throughput variant)

### 14.6.3 BF16 Tensor Cores

Brain Float 16 (BF16) uses 8-bit exponent and 7-bit mantissa, providing a larger dynamic range than FP16:

```cpp
// SM80 BF16 Tensor Core GEMM
using GemmSM80_BF16 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,   // BF16 MMA shape (same as FP16)
    cutlass::epilogue::thread::LinearCombination<float, 8, float, float>
>;
```

**BF16 MMA shapes (same as FP16):**
- `16x8x16`: 16 M, 8 N, 16 K

### 14.6.4 FP16 Tensor Cores on SM80

SM80 improves FP16 Tensor Core performance over SM75:

```cpp
// FP16 MMA shapes on SM80:
// 16x8x16: standard FP16 MMA instruction
// Each thread in the warp participates in the MMA operation
// Warp-level: 32 threads cooperate on one MMA instruction
```

### 14.6.5 cp.async Support

SM80 introduces asynchronous global-to-shared memory copies:

```cpp
// cp.async enables pipelined data movement
// LOAD (GMEM -> SMEM) overlaps with COMPUTE on previous data

cutlass::arch::cp_async_ca(smem_ptr, gmem_ptr, bytes);
cutlass::arch::cp_async_cg(smem_ptr, gmem_ptr, bytes);
cutlass::arch::cp_async_fence();
cutlass::arch::cp_async_wait<0>();

// This enables multi-stage pipeline patterns:
// - 2-stage (double buffer)
// - 3-stage (triple buffer)
// - Up to 4+ stages (limited by shared memory)
```

### 14.6.6 Complete SM80 Data Type Support

| Type | SIMT | Tensor Core | MMA Shape | Accumulator |
|---|---|---|---|---|
| FP64 | Yes | Yes (SM80) | 8x8x4 | FP64 |
| FP32 | Yes | No | - | FP32 |
| TF32 | Storage | Yes | 16x8x4, 16x8x8 | FP32 |
| BF16 | Storage | Yes | 16x8x16 | FP32 |
| FP16 | Storage | Yes | 16x8x16 | FP32 |
| INT8 | Storage | Yes | 16x8x16, 8x8x16 | INT32 |
| INT4 | Storage | Yes | 8x8x32 | INT32 |
| Binary | Storage | Yes | 8x8x128 | INT32 |

### 14.6.7 CUTLASS 3.x on SM80

CUTLASS 3.x supports SM80 through the CollectiveBuilder:

```cpp
using CollectiveOp = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm80, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

// SM80 CUTLASS 3.x uses cp.async pipeline (not TMA)
// The CollectiveBuilder selects the appropriate pipeline automatically
```

---

## 14.7 SM89 (Ada Lovelace)

### 14.7.1 Architecture Overview

Ada (RTX 40-series, L40) extends the Ampere architecture with FP8 (E5M2 and E4M3) support for Tensor Core operations.

**Key specifications:**

- 128 KB or 164 KB shared memory per SM
- FP8 Tensor Core support (both E5M2 and E4M3 formats)
- Otherwise similar to SM80 for CUTLASS purposes

### 14.7.2 FP8 Support

FP8 provides two formats for different use cases:

```cpp
// FP8 E4M3: 4-bit exponent, 3-bit mantissa (for forward pass, higher precision)
// Range: approximately [-448, 448], max value = 448
// No inf/NaN representation

// FP8 E5M2: 5-bit exponent, 2-bit mantissa (for backward pass, wider range)
// Range: approximately [-57344, 57344], max value = 57344
// Supports inf/NaN

// SM89 FP8 Tensor Core GEMM
using GemmFP8 = cutlass::gemm::device::Gemm<
    cutlass::float_e4m3_t, cutlass::layout::RowMajor,    // A: FP8 E4M3
    cutlass::float_e5m2_t, cutlass::layout::ColumnMajor,  // B: FP8 E5M2
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,    // FP8 MMA shape
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>
>;

// Common FP8 usage: E4M3 for both A and B (maximum precision)
// Or E4M3 for A and E5M2 for B (mixed precision for backward pass)
```

**FP8 MMA shapes on SM89:**
- `16x8x32`: 16 M, 8 N, 32 K elements per instruction
- `16x8x16`: alternative shape

---

## 14.8 SM90 (Hopper)

### 14.8.1 Architecture Overview

Hopper (H100, H200) is the primary target for CUTLASS 3.x, introducing revolutionary features including TMA (Tensor Memory Accelerator), GMMA/WGMMA (Global/Warp Group Matrix Multiply-Accumulate), and thread block clusters.

**Key specifications:**

- 228 KB shared memory per SM (configurable)
- 50 MB L2 cache (H100)
- TMA: hardware-accelerated tensor memory access
- WGMMA: warp-group-level (128 threads) matrix multiply
- Thread block clusters: groups of CTAs that can share distributed shared memory
- Warp specialization: dedicated warp groups for load vs compute
- Dynamic shared memory with hardware management

### 14.8.2 TMA (Tensor Memory Accelerator)

TMA is a dedicated hardware unit for tensor data movement:

```cpp
// TMA features:
// 1. Hardware-accelerated address computation for multi-dimensional tensors
// 2. Automatic bounds checking (no predication needed)
// 3. Automatic swizzling for bank-conflict-free SMEM layout
// 4. Single-thread initiation (frees other threads for compute)
// 5. Multicast: one load can fill SMEM in multiple CTAs
// 6. Supports up to 5D tensors natively

// TMA descriptor creation
#include "cute/arch/copy_sm90_tma.hpp"

// auto tma_desc = make_tma_copy(
//     SM90_TMA_LOAD{},
//     tensor_g,              // Global tensor (CuTe tensor)
//     SmemLayout{}           // Shared memory layout
// );

// TMA load
// copy(tma_desc.with(smem_ptr, barrier), tensor_g(coords));
// Only 1 thread issues the TMA, all other threads are free

// TMA multicast (to all CTAs in cluster)
// auto tma_mcast = make_tma_copy(
//     SM90_TMA_LOAD_MULTICAST{},
//     tensor_g,
//     SmemLayout{},
//     cluster_shape
// );
```

### 14.8.3 GMMA / WGMMA

WGMMA (Warp Group Matrix Multiply-Accumulate) operates on a full warp group (128 threads = 4 warps) instead of a single warp (32 threads):

```cpp
// WGMMA characteristics:
// - Operates on 128 threads (4 warps) simultaneously
// - Larger effective MMA shape per operation
// - Can read directly from shared memory (no register staging needed for A)
// - Supports FP16, BF16, TF32, FP8, INT8 data types
// - Higher throughput than mma.sync

// WGMMA shapes (example for FP16):
// - 64x64x16 per warp group per instruction
// - Or 64xN where N varies by data type

// In CuTe, WGMMA is accessed through TiledMMA:
// auto tiled_mma = make_tiled_mma(
//     SM90_64x64x16_F16F16F32_SS{},  // GMMA atom
//     Layout<Shape<_2,_2>>{}          // Tiling for larger output
// );
```

**WGMMA operation variants:**

| Variant | A Source | B Source | Description |
|---|---|---|---|
| SS | Shared Memory | Shared Memory | Both operands from SMEM |
| RS | Register File | Shared Memory | A in registers, B in SMEM |
| SR | Shared Memory | Register File | A in SMEM, B in registers |

### 14.8.4 Thread Block Clusters

Clusters group multiple CTAs that can coordinate and share data:

```cpp
#include "cutlass/cluster_launch.hpp"

// Cluster configuration
dim3 cluster_dims(2, 2, 1);  // 2x2 = 4 CTAs per cluster

// Cluster features:
// 1. Distributed Shared Memory (DSMEM): CTAs can read each other's SMEM
// 2. Cluster-level barriers: synchronize across CTAs
// 3. TMA multicast: one TMA load broadcasts to all CTAs
// 4. Atomic operations across DSMEM

// Launch with clusters
cutlass::ClusterLaunchParams params{
    grid_dims, block_dims, cluster_dims, smem_size, stream
};
```

### 14.8.5 Warp Specialization

SM90 supports assigning different warp groups to different roles:

```cpp
// Warp specialization patterns:

// 1. Warp-specialized (producer-consumer):
//    Warp group 0 (warps 0-3): TMA loads (producer)
//    Warp group 1 (warps 4-7): WGMMA compute (consumer)
using Schedule = cutlass::gemm::collective::KernelTmaWarpSpecialized;

// 2. Cooperative:
//    Both warp groups compute on the same output tile
using Schedule = cutlass::gemm::collective::KernelTmaWarpSpecializedCooperative;

// 3. Ping-pong:
//    Warp groups alternate between producer and consumer roles
using Schedule = cutlass::gemm::collective::KernelTmaWarpSpecializedPingpong;
```

### 14.8.6 SM90 CUTLASS 3.x Example

```cpp
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;

using CollectiveOp = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 8,
    ElementB, LayoutB, 8,
    ElementC,
    TileShape,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::layout::RowMajor,
    cutlass::layout::RowMajor,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>;

using Kernel = cutlass::gemm::kernel::GemmUniversal<
    CollectiveOp, EpilogueOp,
    cutlass::gemm::StreamKPolicy
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

Gemm gemm_op;
auto args = Gemm::Arguments{
    {M, N, K},
    {ptr_A, stride_A},
    {ptr_B, stride_B},
    {ptr_C, stride_C},
    {ptr_D, stride_D},
    {alpha, beta}
};
gemm_op(args);
```

### 14.8.7 SM90 Data Type Support

| Type | Tensor Core | MMA Shape | Accumulator | Notes |
|---|---|---|---|---|
| FP64 | Yes (WGMMA) | 64x64x4 / 64x32x8 | FP64 | First FP64 Tensor Core |
| TF32 | Yes (WGMMA) | 64x64x8 | FP32 | Higher throughput than SM80 |
| BF16 | Yes (WGMMA) | 64x64x16 | FP32 | WGMMA from SMEM |
| FP16 | Yes (WGMMA) | 64x64x16 | FP32 | WGMMA from SMEM |
| FP8 E4M3 | Yes (WGMMA) | 64x128x32 | FP32/FP16 | Native FP8 support |
| FP8 E5M2 | Yes (WGMMA) | 64x128x32 | FP32/FP16 | Native FP8 support |
| INT8 | Yes (WGMMA) | 64x64x32 | INT32 | WGMMA from SMEM |

---

## 14.9 SM100/SM101/SM103 (Blackwell)

### 14.9.1 Architecture Overview

Blackwell (B100, B200) introduces the fifth generation of Tensor Cores with UMMA (Unified Matrix Multiply-Accumulate), block-scaled data types (NVFP4, MXFP4/6/8), and distributed GEMM support.

**Key specifications:**

- 228 KB+ shared memory per SM
- UMMA: unified MMA supporting block-scaled operations
- Block-scaled types: NVFP4, MXFP4, MXFP6, MXFP8
- Green contexts: lightweight context switching
- Distributed GEMM: NVLink-based tensor parallelism
- Enhanced TMA with larger tile support

### 14.9.2 UMMA (Unified Matrix Multiply-Accumulate)

UMMA extends WGMMA with support for block-scaled data types:

```cpp
// UMMA features:
// 1. Block-scaled matrix multiply: multiplies quantized blocks with associated scale factors
// 2. Supports NVFP4, MXFP4, MXFP6, MXFP8 data types
// 3. Automatic scale factor handling in hardware
// 4. Larger matrix sizes per operation

// In CUTLASS 3.x, UMMA is accessed through the CollectiveBuilder:
using CollectiveOp = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass::float_e4m3_t, cutlass::layout::RowMajor, 16,  // FP8 input
    cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
    float,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

### 14.9.3 Block-Scaled Types (NVFP4, MXFP4/6/8)

Blackwell introduces sub-byte block-scaled floating-point types:

```cpp
// NVFP4: 4-bit floating-point with block-level scaling
// - 2-bit exponent, 1-bit mantissa per element
// - Block size: 16 elements share one scale factor (E8M0 format)
// - Extremely compact: 2 bits per element + 8 bits per block of 16

// MXFP4: Microscaling FP4 (OCP standard)
// - Similar to NVFP4 but with standardized scaling format
// - Block size: 32 elements

// MXFP6: Microscaling FP6
// - 6-bit floating-point with block-level scaling
// - Two variants: E2M3 and E3M2

// MXFP8: Microscaling FP8
// - Standard FP8 (E4M3/E5M2) with block-level scaling
// - Block size: 32 elements

// CUTLASS block-scaled GEMM example:
using GemmNVFP4 = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<
        cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
            cutlass::nvfp4_t, cutlass::layout::RowMajor, 32,
            cutlass::nvfp4_t, cutlass::layout::ColumnMajor, 32,
            float,
            cutlass::gemm::GemmShape<128, 128, 256>,
            cutlass::gemm::collective::StageCountAutoCarveout<0>,
            cutlass::gemm::collective::KernelScheduleAuto
        >::CollectiveOp,
        cutlass::epilogue::collective::DefaultEpilogue<
            cutlass::layout::RowMajor,
            cutlass::layout::RowMajor,
            cutlass::epilogue::collective::EpilogueScheduleAuto
        >
    >
>;
```

### 14.9.4 Green Contexts

Green contexts provide lightweight GPU context management:

```cpp
// Green contexts:
// - Lightweight alternative to CUDA streams and contexts
// - Enable faster context switching between workloads
// - Allow fine-grained resource partitioning on the GPU
// - Useful for multi-tenant GPU scenarios

// CUTLASS can target green contexts for kernel launch:
// cutlass::GreenContextLaunch(green_ctx, kernel, args);
```

### 14.9.5 Distributed GEMM

Blackwell supports distributed GEMM across multiple GPUs connected via NVLink:

```cpp
// Distributed GEMM: matrix multiply across multiple GPUs
// - Tensor parallelism: split M or N dimension across GPUs
// - NVLink provides high-bandwidth interconnect
// - CUTLASS handles the communication pattern

// See: cutlass/gemm/collective/sm100_distributed_gemm.hpp
// Uses NVLink for all-reduce communication during GEMM
```

### 14.9.6 SM100 Variants

| SM Version | GPU | Key Differences |
|---|---|---|
| SM100 | B100 | Data center GPU, full feature set |
| SM101 | B200 | Enhanced version, same ISA |
| SM103 | Consumer variant | May have reduced shared memory or Tensor Core count |

---

## 14.10 SM120 (Future Architecture)

SM120 represents future NVIDIA architectures beyond Blackwell. CUTLASS provides preliminary support through forward-looking template specializations:

```cpp
// SM120 is a placeholder for future architectures
// CUTLASS prepares for SM120 through:
// - Template specialization points in CollectiveBuilder
// - Forward-compatible API design in CUTLASS 3.x
// - Extensible CuTe abstraction layer

// When SM120 hardware is available, CUTLASS will add:
// - SM120-specific MMA atoms
// - SM120-specific copy atoms
// - New pipeline patterns if hardware changes warrant them

// The CollectiveBuilder auto-selection will extend to SM120:
using CollectiveOp = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape,
    StageCount,
    ScheduleType
>::CollectiveOp;
```

---

## 14.11 Compute Capability Detection

### 14.11.1 Runtime Detection

```cpp
// Query the compute capability at runtime
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
int major = prop.major;
int minor = prop.minor;
int sm_version = major * 10 + minor;

// Switch on architecture
switch (sm_version) {
    case 50: /* Maxwell */ break;
    case 60: case 61: /* Pascal */ break;
    case 70: /* Volta */ break;
    case 75: /* Turing */ break;
    case 80: /* Ampere (A100) */ break;
    case 89: /* Ada (RTX 40) */ break;
    case 90: /* Hopper (H100) */ break;
    case 100: /* Blackwell (B100) */ break;
}
```

### 14.11.2 CUTLASS Architecture Tags

CUTLASS uses architecture tag types for compile-time dispatch:

```cpp
// Architecture tags (compile-time)
cutlass::arch::Sm50    // Maxwell
cutlass::arch::Sm60    // Pascal
cutlass::arch::Sm70    // Volta
cutlass::arch::Sm75    // Turing
cutlass::arch::Sm80    // Ampere
cutlass::arch::Sm89    // Ada
cutlass::arch::Sm90    // Hopper
cutlass::arch::Sm100   // Blackwell
cutlass::arch::Sm120   // Future

// Operation class tags
cutlass::arch::OpClassSimt       // SIMT (scalar per thread)
cutlass::arch::OpClassTensorOp   // Tensor Core (hardware matrix multiply)
cutlass::arch::OpClassWmmaTensorOp  // WMMA Tensor Core (SM70)
```

---

## 14.12 Conditional Compilation Macros

CUTLASS uses preprocessor macros for architecture-conditional compilation:

```cpp
// CUTLASS architecture detection macros
// These are defined based on the CUDA compiler target (--gpu-architecture)

// Check minimum compute capability
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // SM80+ (Ampere and later) code path
    // cp.async is available
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // SM90+ (Hopper and later) code path
    // TMA, WGMMA, clusters available
#endif

// CUTLASS-specific macros
// CUTLASS_ARCH_SM80_ENABLED - defined when targeting SM80+
// CUTLASS_ARCH_SM90_ENABLED - defined when targeting SM90+

// Use in CUTLASS code:
#if CUTLASS_ARCH_SM90_ENABLED
    // Use TMA-based pipeline
#else
    // Use cp.async-based pipeline
#endif
```

**Common CUDA architecture macros:**

| Macro | Description |
|---|---|
| `__CUDA_ARCH__` | Target architecture in decimal (e.g., 800, 900) |
| `__CUDA_ARCH_FEAT_SM80` | SM80 feature set available |
| `__CUDA_ARCH_FEAT_SM90` | SM90 feature set available |
| `CUTLASS_ENABLE_GDC_FOR_SM90` | Enable GDC for SM90 kernels |

---

## 14.13 Architecture Selection in CUTLASS 3.x

The `CollectiveBuilder` automatically selects the best implementation for the target architecture:

```cpp
// CollectiveBuilder dispatch table (simplified):
// SM90 + OpClassTensorOp -> TMA + WGMMA pipeline (warp-specialized)
// SM80 + OpClassTensorOp -> cp.async + mma.sync pipeline (multistage)
// SM75 + OpClassTensorOp -> mma.sync pipeline (2-stage)
// SM70 + OpClassTensorOp -> WMMA pipeline (2-stage)
// Any  + OpClassSimt     -> SIMT pipeline

using CollectiveOp = cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,          // Architecture tag determines pipeline type
    OpClass,          // SIMT vs TensorOp determines MMA type
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape,
    StageCount,
    KernelSchedule    // Auto lets builder choose best schedule
>::CollectiveOp;
```

**Automatic dispatch logic:**

```
ArchTag == Sm90 && OpClass == TensorOp:
  -> TMA-based warp-specialized pipeline
  -> WGMMA for compute
  -> Cluster support if cluster_dims > 1

ArchTag == Sm80 && OpClass == TensorOp:
  -> cp.async multistage pipeline
  -> mma.sync for compute
  -> 2-4 pipeline stages

ArchTag == Sm75 && OpClass == TensorOp:
  -> mma.sync 2-stage pipeline
  -> No async copy (uses ld.global)

ArchTag == Sm70 && OpClass == TensorOp:
  -> WMMA pipeline
  -> wmma.mma.sync for compute

OpClass == Simt (any architecture):
  -> SIMT multistage pipeline
  -> Scalar multiply-accumulate
```

---

## 14.14 Cross-Architecture Portability

### 14.14.1 Writing Portable CUTLASS Code

```cpp
// Use the CollectiveBuilder for architecture-portable code:
// The builder selects the best implementation for each architecture

// This single definition works across SM80, SM89, SM90, SM100:
template <typename ArchTag>
using PortableGemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<
        typename cutlass::gemm::collective::CollectiveBuilder<
            ArchTag,
            cutlass::arch::OpClassTensorOp,
            cutlass::half_t, cutlass::layout::RowMajor, 8,
            cutlass::half_t, cutlass::layout::ColumnMajor, 8,
            float,
            cutlass::gemm::GemmShape<128, 128, 64>,
            cutlass::gemm::collective::StageCountAutoCarveout<0>,
            cutlass::gemm::collective::KernelScheduleAuto
        >::CollectiveOp,
        cutlass::epilogue::collective::DefaultEpilogue<
            cutlass::layout::RowMajor,
            cutlass::layout::RowMajor,
            cutlass::epilogue::collective::EpilogueScheduleAuto
        >
    >
>;

// Instantiate for specific architectures:
using GemmAmpere = PortableGemm<cutlass::arch::Sm80>;
using GemmHopper = PortableGemm<cutlass::arch::Sm90>;
using GemmBlackwell = PortableGemm<cutlass::arch::Sm100>;
```

### 14.14.2 Architecture-Specific Tuning

For optimal performance, tile shapes and stage counts should be tuned per architecture:

```cpp
// Architecture-specific tile shape selection
template <typename ArchTag>
struct OptimalTileShape;

template <>
struct OptimalTileShape<cutlass::arch::Sm80> {
    using Shape = cutlass::gemm::GemmShape<128, 128, 64>;
    static constexpr int Stages = 3;
};

template <>
struct OptimalTileShape<cutlass::arch::Sm90> {
    using Shape = cutlass::gemm::GemmShape<128, 128, 64>;
    static constexpr int Stages = 7;
};

template <>
struct OptimalTileShape<cutlass::arch::Sm100> {
    using Shape = cutlass::gemm::GemmShape<128, 128, 128>;
    static constexpr int Stages = 7;
};
```

---

## 14.15 Key Header Files Reference

| Header | Purpose |
|---|---|
| `cutlass/arch/arch.h` | Architecture tag definitions |
| `cutlass/arch/mma.h` | MMA operation dispatch |
| `cutlass/arch/mma_sm50.h` | SM50 SIMT MMA |
| `cutlass/arch/mma_sm70.h` | SM70 WMMA MMA |
| `cutlass/arch/mma_sm75.h` | SM75 Tensor Core MMA |
| `cutlass/arch/mma_sm80.h` | SM80 Tensor Core MMA |
| `cutlass/arch/mma_sm89.h` | SM89 FP8 MMA |
| `cutlass/arch/mma_sm90.h` | SM90 WGMMA MMA |
| `cutlass/arch/wmma.h` | WMMA abstraction (SM70+) |
| `cutlass/arch/memory.h` | cp.async operations (SM80+) |
| `cutlass/arch/memory_sm90.h` | SM90 memory operations |
| `cutlass/gemm/collective/collective_builder.hpp` | Architecture-dispatched collective builder |
| `cute/arch/copy_sm90_tma.hpp` | TMA copy operations |
| `cute/arch/mma_sm90.hpp` | CuTe WGMMA operations |

---

## 14.16 Summary

CUTLASS supports a wide range of NVIDIA GPU architectures, from Maxwell (SM50) through future SM120:

1. **SM50-SM61 (Maxwell/Pascal)**: SIMT-only operations, no Tensor Cores. Pascal adds FP16 storage and DP4A.
2. **SM70 (Volta)**: First Tensor Cores via WMMA API, FP16 input with FP32 accumulation.
3. **SM75 (Turing)**: Second-gen Tensor Cores with `mma.sync`, INT8/INT4/INT1 support.
4. **SM80 (Ampere)**: TF32, BF16, `cp.async` for pipelined memory, multi-stage pipelines. The baseline for CUTLASS 3.x.
5. **SM89 (Ada)**: Adds FP8 (E4M3, E5M2) Tensor Core support.
6. **SM90 (Hopper)**: TMA, WGMMA, thread block clusters, warp specialization. The primary target for CUTLASS 3.x.
7. **SM100+ (Blackwell)**: UMMA, block-scaled types (NVFP4/MXFP4/6/8), distributed GEMM, green contexts.
8. **SM120**: Future architecture support through extensible API design.
9. The `CollectiveBuilder` provides automatic architecture dispatch, selecting the optimal pipeline and MMA type for each target.
