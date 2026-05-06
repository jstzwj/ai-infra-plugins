# CUTLASS Reference - Chapter 23: Blockwise Scaling

This reference covers blockwise scaling in CUTLASS, a technique that enables high-throughput GEMM with sub-byte data types by applying per-block scaling factors. Blockwise scaling is essential for FP8, NVFP4, and MX-format operations on Hopper (SM90) and Blackwell (SM100+) architectures.

---

## 23.1 FP8 Blockwise Scaling Concept

### 23.1.1 Motivation

FP8 data types (e4m3 and e5m2) have very limited dynamic range and precision compared to FP16 or BF16. Without scaling, FP8 GEMM can suffer from overflow, underflow, and significant numerical error. Blockwise scaling addresses this by:

1. Dividing the input tensors into fixed-size blocks (tiles).
2. Computing a per-block scaling factor that normalizes the values to the optimal range for the data type.
3. Applying the scaling factor during the GEMM computation.

This is analogous to how quantization scales work in INT8 inference, but applied to floating-point types.

### 23.1.2 Blockwise Scaling Pipeline

The blockwise scaling pipeline for a GEMM operation is:

```
A (FP8) * Scale_A (FP32) --> normalized A values (FP32 accumulator)
B (FP8) * Scale_B (FP32) --> normalized B values (FP32 accumulator)
                                     |
                                     v
                          FP32 Accumulator result
                                     |
                                     v
                          / (Scale_A * Scale_B)    (de-normalize)
                                     |
                                     v
                          Output (FP8, FP16, BF16, FP32)
```

The key insight is that the per-block scaling factors are applied element-wise during the MMA operation, and the reciprocal scaling is applied in the epilogue to produce the final result.

---

## 23.2 Scale Tensors: SFA and SFB

### 23.2.1 Scale Factor Definitions

In CUTLASS, block-scaled GEMM uses two scale tensors:

- **SFA (Scale Factor A)**: Per-block scaling factor for operand A. Shape depends on the scaling granularity.
- **SFB (Scale Factor B)**: Per-block scaling factor for operand B. Shape depends on the scaling granularity.

The GEMM operation with blockwise scaling computes:

```
D[i, j] = sum_k( A[i, k] * SFA[i, k_block] * B[k, j] * SFB[k_block, j] )
```

Where `k_block` is the block index corresponding to the K position. The SFA and SFB values broadcast across the block dimension.

### 23.2.2 Scale Factor Tensor Layout

The scale tensors have a specific layout determined by the scaling granularity:

```cpp
// For GEMM with M x K operand A and K x N operand B:
// If scaling granularity is (TM, TK) for A and (TK, TN) for B:

// SFA shape: (M / TM, K / TK) -- one scale per TM x TK block
// SFB shape: (K / TK, N / TN) -- one scale per TK x TN block

// Example with TM=128, TK=128 for a 1024x4096 matrix:
// SFA shape: (1024/128, 4096/128) = (8, 32) = 256 scale factors
```

### 23.2.3 Scale Factor Data Types

Scale factors are typically stored in FP32 for maximum precision:

```cpp
using ElementScale = float;  // Scale factors are FP32

// For the scaled GEMM API, scale tensors are passed as additional arguments:
// {ptr_SFA, stride_SFA} and {ptr_SFB, stride_SFB}
```

---

## 23.3 Scale Granularity Configuration

### 23.3.1 Granularity Options

The scaling granularity determines the size of the blocks over which each scale factor applies. CUTLASS supports several granularity configurations:

| Granularity | Block Size (A) | Block Size (B) | Use Case |
|---|---|---|---|
| 1x128 | 1 x 128 | 128 x 1 | Per-row scaling (common for inference) |
| 32x128 | 32 x 128 | 128 x 32 | Per-tile scaling (training) |
| 128x128 | 128 x 128 | 128 x 128 | Larger tile scaling |

### 23.3.2 Granularity and the CollectiveBuilder

The CollectiveBuilder accepts granularity parameters through the tile shape configuration:

```cpp
// FP8 blockwise GEMM with specific scaling granularity
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass::float_e4m3_t, cutlass::layout::RowMajor, 16,
    cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
    float,  // ElementAccumulator
    cutlass::gemm::GemmShape<128, 128, 128>,  // MMA tile shape
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

### 23.3.3 Granularity Impact on Accuracy

Finer granularity (smaller blocks) provides better numerical accuracy because the scale factor more closely matches the local value distribution:

- **1x128 (per-row)**: Each row gets its own scale. Good for activation matrices with varying magnitudes across rows.
- **32x128**: Each 32x128 tile gets its own scale. Balances accuracy and overhead.
- **128x128**: Larger tiles share a scale. More efficient but less adaptive.

---

## 23.4 Block-Scaled GEMM on Blackwell

### 23.4.1 Blackwell UMMA Instructions

Blackwell (SM100+) introduces **UMMA** (Universal MMA) instructions that natively support block-scaled data types. Unlike Hopper where scaling must be done in software, Blackwell hardware can apply scale factors during the MMA operation itself.

Key features of Blackwell block-scaled GEMM:
- **Native FP8 blockwise scaling**: Hardware applies scale factors during MMA.
- **NVFP4 support**: 4-bit floating-point with per-block scaling.
- **MX format support**: Microscaling formats (MXFP4, MXFP6, MXFP8).
- **Reduced software overhead**: Scale factors are loaded directly by the MMA instruction.

### 23.4.2 Block-Scaled GEMM API

```cpp
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

// Blackwell block-scaled FP8 GEMM
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementAccumulator = float;
using ElementOutput = cutlass::float_e4m3_t;
using ElementScale = float;

using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassTensorOp,
    ElementA, cutlass::layout::RowMajor, 16,
    ElementB, cutlass::layout::ColumnMajor, 16,
    ElementAccumulator,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelTmaWarpSpecializedFP8Blockwise
>::CollectiveOp;

using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::layout::RowMajor, cutlass::layout::RowMajor,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>;

using Kernel = cutlass::gemm::kernel::GemmUniversal<
    CollectiveOp, EpilogueOp
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
```

### 23.4.3 Launching Block-Scaled GEMM

```cpp
void run_block_scaled_gemm(
    int M, int N, int K,
    const cutlass::float_e4m3_t* A, int64_t lda,
    const cutlass::float_e4m3_t* B, int64_t ldb,
    const float* C, int64_t ldc,
    cutlass::float_e4m3_t* D, int64_t ldd,
    const float* SFA, int64_t stride_sfa,    // Scale factor A
    const float* SFB, int64_t stride_sfb,    // Scale factor B
    float alpha, float beta,
    cudaStream_t stream = 0
) {
    using Gemm = /* ... as defined above ... */;
    Gemm gemm_op;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmCoord{M, N, K},
        {A, lda},
        {B, ldb},
        {C, ldc},
        {D, ldd},
        {alpha, beta},
        {SFA, stride_sfa},   // Block scale for A
        {SFB, stride_sfb}    // Block scale for B
    };

    size_t workspace_size = gemm_op.get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    auto status = gemm_op.initialize(args, workspace.get(), stream);
    if (status == cutlass::Status::kSuccess) {
        status = gemm_op(stream);
    }
}
```

---

## 23.5 NVFP4 Data Format

### 23.5.1 NVFP4 Overview

NVFP4 is a 4-bit floating-point format introduced with Blackwell (SM100+):
- **1 sign bit**.
- **2 exponent bits** (or 3, depending on variant).
- **1 mantissa bit** (or 0, depending on variant).

The extremely small representation (4 bits) provides 8x density compared to FP32 and 4x compared to FP8, but requires blockwise scaling to maintain usable dynamic range.

### 23.5.2 NVFP4 Storage

NVFP4 elements are packed into bytes, with 2 elements per byte:

```cpp
// NVFP4: 2 elements per byte
// Element 0: bits [3:0]
// Element 1: bits [7:4]

// CUTLASS provides the nvfp4_t type
using ElementNVFP4 = cutlass::nvfp4_t;
// Each nvfp4_t value packs 2 FP4 elements
```

### 23.5.3 NVFP4 Blockwise Scaling

NVFP4 always requires blockwise scaling because 4 bits cannot represent a useful range without normalization:

```cpp
// NVFP4 GEMM with blockwise scaling
using ElementA = cutlass::nvfp4_t;
using ElementB = cutlass::nvfp4_t;
using ElementAccumulator = float;

// Scale factors are mandatory for NVFP4
using ElementScale = float;

// The CollectiveBuilder handles the NVFP4 + scaling combination
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassTensorOp,
    ElementA, cutlass::layout::RowMajor, 32,    // Alignment 32 for NVFP4 (16 bytes)
    ElementB, cutlass::layout::ColumnMajor, 32,
    ElementAccumulator,
    cutlass::gemm::GemmShape<128, 128, 256>,    // Larger K tile for sub-byte types
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

---

## 23.6 MXFP4, MXFP6, MXFP8 Formats

### 23.6.1 Microscaling (MX) Format Overview

The MX (Microscaling) format family is an open standard (OCP specification) for block-scaled floating-point types. CUTLASS supports the following MX formats on Blackwell:

| Format | Total Bits | Sign | Exponent | Mantissa | Block Size |
|---|---|---|---|---|---|
| MXFP8 (E5M2) | 8 | 1 | 5 | 2 | 32 |
| MXFP8 (E4M3) | 8 | 1 | 4 | 3 | 32 |
| MXFP6 (E3M2) | 6 | 1 | 3 | 2 | 32 |
| MXFP6 (E2M3) | 6 | 1 | 2 | 3 | 32 |
| MXFP4 (E2M1) | 4 | 1 | 2 | 1 | 32 |

All MX formats use a fixed block size of 32 elements with a shared FP8 scale factor (E8M0 format -- 8-bit exponent, 0 mantissa bits, representing a power of 2).

### 23.6.2 MXFP8 GEMM

MXFP8 is the most mature MX format, providing a balance between compression and accuracy:

```cpp
// MXFP8 GEMM with microscaling
using ElementA = cutlass::float_e4m3_t;  // MXFP8 E4M3
using ElementB = cutlass::float_e4m3_t;
using ElementAccumulator = float;
using ElementScale = float;               // E8M0 scale (power of 2)

// Scale granularity: 32 elements per scale factor
// For M x K matrix with MX scaling:
//   SFA shape: (M, K / 32)
//   SFB shape: (K / 32, N)

using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassTensorOp,
    ElementA, cutlass::layout::RowMajor, 16,
    ElementB, cutlass::layout::ColumnMajor, 16,
    ElementAccumulator,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

### 23.6.3 MXFP6 and MXFP4 GEMM

Sub-byte MX formats provide higher compression at the cost of precision:

```cpp
// MXFP6 GEMM
// Note: MXFP6 has 6-bit elements, packed 4 elements per 3 bytes
using ElementA = cutlass::mxfp6_e3m2_t;  // or cutlass::mxfp6_e2m3_t
using ElementB = cutlass::mxfp6_e3m2_t;
using ElementAccumulator = float;

// MXFP4 GEMM (most compact)
using ElementA = cutlass::mxfp4_e2m1_t;
using ElementB = cutlass::mxfp4_e2m1_t;
using ElementAccumulator = float;
```

### 23.6.4 MX Scale Factor Format

MX scale factors use the E8M0 format (also called FP8 E8M0):
- **8 exponent bits**: Representing powers of 2.
- **0 mantissa bits**: The scale is exactly a power of 2.
- **Range**: 2^(-127) to 2^(127), with subnormal support.

```cpp
// MX scale factors are powers of 2
// To compute the scale for a block of 32 elements:
float compute_mx_scale(const float* block, int block_size) {
    float max_abs = 0.0f;
    for (int i = 0; i < block_size; ++i) {
        max_abs = std::max(max_abs, std::abs(block[i]));
    }
    // Scale to use the full range of the target format
    float target_max = /* max representable value for the format */;
    float scale = target_max / max_abs;
    // Round to nearest power of 2 (E8M0 format)
    int exp = (int)std::round(std::log2(scale));
    return std::ldexp(1.0f, exp);
}
```

---

## 23.7 Software Scaling with Configurable Granularity

### 23.7.1 Hopper Software Scaling

On Hopper (SM90), blockwise scaling must be implemented in software because the GMMA instructions do not natively apply scale factors. CUTLASS provides the `KernelTmaWarpSpecializedFP8Blockwise` dispatch policy for this purpose:

```cpp
// Hopper FP8 blockwise scaling (software)
using DispatchPolicy = cutlass::gemm::collective::KernelTmaWarpSpecializedFP8Blockwise;

// This dispatch policy:
// 1. Loads FP8 data via TMA into shared memory
// 2. Loads scale factors via TMA
// 3. Applies scaling in registers before MMA
// 4. Accumulates in FP32
// 5. Applies reciprocal scaling in the epilogue
```

### 23.7.2 Configurable Granularity

The software scaling path supports configurable granularity through template parameters:

```cpp
// Specify block scaling granularity explicitly
template<
    int ScaleGranularityM,    // Block size in M dimension
    int ScaleGranularityN,    // Block size in N dimension
    int ScaleGranularityK     // Block size in K dimension
>
struct BlockScaleConfig {
    static constexpr int kGranularityM = ScaleGranularityM;
    static constexpr int kGranularityN = ScaleGranularityN;
    static constexpr int kGranularityK = ScaleGranularityK;
};

// Common configurations:
using PerRowScaling = BlockScaleConfig<1, 1, 128>;      // 1x128 granularity
using TileScaling32 = BlockScaleConfig<32, 32, 128>;    // 32x128 granularity
using TileScaling128 = BlockScaleConfig<128, 128, 128>; // 128x128 granularity
```

---

## 23.8 Naming Conventions for Scaled Operations

### 23.8.1 Kernel Naming

CUTLASS uses specific naming conventions for block-scaled GEMM kernels:

- `KernelTmaWarpSpecializedFP8Blockwise`: FP8 GEMM with software blockwise scaling on Hopper.
- `KernelTmaWarpSpecializedBlockScaled`: Block-scaled GEMM on Blackwell (hardware scaling).
- `KernelTmaWarpSpecializedMixed`: Mixed-type block-scaled GEMM.

### 23.8.2 Collective Naming

The collective operations for block-scaled GEMM follow a naming pattern:

```
CollectiveMma<
    ArchTag, OpClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape,
    StageCount,
    Schedule,
    ScaleGranularity   // Optional: for block-scaled variants
>
```

---

## 23.9 MMA Dimensions and Layout Swapping

### 23.9.1 MMA Tile Shape for Block-Scaled GEMM

Block-scaled GEMM often uses different tile shapes than dense GEMM because:
- Sub-byte types process more logical K elements per MMA instruction.
- Scale factors introduce additional memory traffic that must be overlapped with computation.
- The hardware MMA instruction has specific supported shapes for each data type.

```cpp
// Typical tile shapes for FP8 blockwise GEMM:
// Dense FP16:    GemmShape<128, 128, 64>    -- 64 K elements
// FP8 blockwise: GemmShape<128, 128, 128>   -- 128 K elements (2x due to smaller type)
// NVFP4:         GemmShape<128, 128, 256>   -- 256 K elements (8x vs FP32)
```

### 23.9.2 Layout Swapping

For block-scaled operations, CUTLASS may internally swap the M and N dimensions of the MMA operation to optimize for the hardware's preferred data layout. This is handled transparently by the CollectiveBuilder:

```cpp
// The CollectiveBuilder handles layout swapping automatically
// For example, on Blackwell, the UMMA instruction may prefer:
//   A: (K, M) with ColumnMajor instead of (M, K) with RowMajor
// This swap is internal and does not affect the user-visible API.
```

The layout swap is a performance optimization that aligns the data access pattern with the hardware's preferred traversal order. It is particularly important for sub-byte types where the memory access pattern significantly impacts throughput.

---

## 23.10 Accumulator Type Selection for Precision Control

### 23.10.1 Accumulator Options

The choice of accumulator type affects both precision and performance for block-scaled GEMM:

| Accumulator Type | Precision | Throughput Impact | Use Case |
|---|---|---|---|
| `float` (FP32) | Full single precision | Baseline | Training, high-accuracy inference |
| `cutlass::float_e4m3_t` | FP8 precision | Higher throughput | Inference with acceptable error |
| `cutlass::half_t` | FP16 precision | Moderate improvement | Memory-bound scenarios |

### 23.10.2 FP32 Accumulation (Recommended for Training)

```cpp
// Block-scaled GEMM with FP32 accumulation for training
using ElementAccumulator = float;  // Full precision accumulation

using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassTensorOp,
    cutlass::float_e4m3_t, cutlass::layout::RowMajor, 16,
    cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
    ElementAccumulator,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

### 23.10.3 Reduced Precision Accumulation

For inference workloads where some accuracy can be traded for throughput:

```cpp
// Block-scaled GEMM with FP16 accumulation for inference
using ElementAccumulator = cutlass::half_t;

// WARNING: FP16 accumulation can cause overflow for large K values.
// Use only when K is small or values are well-normalized.
```

---

## 23.11 KernelTmaWarpSpecializedFP8Blockwise Dispatch Policy

### 23.11.1 Overview

The `KernelTmaWarpSpecializedFP8Blockwise` dispatch policy implements FP8 blockwise scaling in software on Hopper (SM90). It extends the standard `KernelTmaWarpSpecialized` policy with additional logic for scale factor handling.

### 23.11.2 Policy Components

The dispatch policy consists of:

1. **TMA loads for operands A and B**: Load FP8 data tiles from global memory.
2. **TMA loads for scale tensors SFA and SFB**: Load scale factors for each block.
3. **Software scaling in registers**: Apply `A[i,k] * SFA[k_block]` and `B[k,j] * SFB[k_block]` before MMA.
4. **FP32 MMA accumulation**: The scaled values are accumulated in FP32.
5. **Epilogue with reciprocal scaling**: The final result is divided by `(SFA * SFB)` or the scaling is folded into the epilogue scaling factors.

```cpp
// Dispatch policy for Hopper FP8 blockwise GEMM
using DispatchPolicy = cutlass::gemm::collective::KernelTmaWarpSpecializedFP8Blockwise;

// The policy requires additional template arguments:
// - Scale granularity in K dimension (e.g., 128)
// - Scale granularity in M/N dimensions
```

### 23.11.3 Complete Hopper FP8 Blockwise GEMM Example

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

// Hopper FP8 blockwise GEMM
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementAccumulator = float;
using ElementOutput = cutlass::float_e4m3_t;
using ElementScale = float;

using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, cutlass::layout::RowMajor, 16,
    ElementB, cutlass::layout::ColumnMajor, 16,
    ElementAccumulator,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelTmaWarpSpecializedFP8Blockwise
>::CollectiveOp;

using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::layout::RowMajor, cutlass::layout::RowMajor,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>;

using Kernel = cutlass::gemm::kernel::GemmUniversal<
    CollectiveOp, EpilogueOp
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

void run_hopper_fp8_blockwise_gemm(
    int M, int N, int K,
    const cutlass::float_e4m3_t* A, int64_t lda,
    const cutlass::float_e4m3_t* B, int64_t ldb,
    const float* C, int64_t ldc,
    cutlass::float_e4m3_t* D, int64_t ldd,
    const float* SFA, int64_t stride_sfa,
    const float* SFB, int64_t stride_sfb,
    float alpha, float beta,
    cudaStream_t stream = 0
) {
    Gemm gemm_op;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmCoord{M, N, K},
        {A, lda},
        {B, ldb},
        {C, ldc},
        {D, ldd},
        {alpha, beta},
        {SFA, stride_sfa},   // Scale factors for A
        {SFB, stride_sfb}    // Scale factors for B
    };

    size_t workspace_size = gemm_op.get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    auto status = gemm_op.initialize(args, workspace.get(), stream);
    if (status == cutlass::Status::kSuccess) {
        status = gemm_op(stream);
    }
}
```

---

## 23.12 Computing Block Scale Factors

### 23.12.1 Scale Factor Computation

Block scale factors should be computed to maximize the utilization of the target format's dynamic range. The standard approach:

```cpp
// Compute FP8 E4M3 scale factors for a block
template <typename FP8Type, int BlockSize>
float compute_block_scale(const float* data, int block_idx) {
    constexpr float FP8_E4M3_MAX = 448.0f;    // Max representable value
    constexpr float FP8_E4M3_MIN_POS = 6e-8f; // Min positive normal

    const float* block = data + block_idx * BlockSize;

    // Find the maximum absolute value in the block
    float amax = 0.0f;
    for (int i = 0; i < BlockSize; ++i) {
        amax = std::max(amax, std::abs(block[i]));
    }

    // Compute the scale: target_max / amax
    if (amax < FP8_E4M3_MIN_POS) {
        return 0.0f;  // Block is too small, scale to zero
    }

    float scale = FP8_E4M3_MAX / amax;
    return scale;
}
```

### 23.12.2 GPU-Accelerated Scale Computation

For large tensors, compute scale factors on the GPU:

```cpp
// CUDA kernel to compute per-block scale factors
template <int BlockSize>
__global__ void compute_block_scales_kernel(
    const float* __restrict__ input,
    float* __restrict__ scales,
    int num_blocks
) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    const float* block = input + block_idx * BlockSize;

    // Find amax in the block
    float amax = 0.0f;
    for (int i = 0; i < BlockSize; ++i) {
        amax = fmaxf(amax, fabsf(block[i]));
    }

    // Compute scale
    constexpr float TARGET_MAX = 448.0f;  // FP8 E4M3 max
    scales[block_idx] = (amax > 1e-8f) ? (TARGET_MAX / amax) : 0.0f;
}

// Apply scaling and convert to FP8
template <int BlockSize>
__global__ void apply_scaling_and_convert_kernel(
    const float* __restrict__ input,
    const float* __restrict__ scales,
    cutlass::float_e4m3_t* __restrict__ output,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int block_idx = idx / BlockSize;
    float scale = scales[block_idx];

    output[idx] = cutlass::float_e4m3_t(input[idx] * scale);
}
```

---

## 23.13 Summary

Blockwise scaling is essential for achieving high accuracy with low-precision data types:

1. **FP8 blockwise scaling**: Applied in software on Hopper, hardware-assisted on Blackwell.
2. **NVFP4**: 4-bit floating-point that always requires blockwise scaling.
3. **MX formats**: Open standard microscaling with 32-element blocks and E8M0 scale factors.
4. **Scale granularity**: Controls the tradeoff between accuracy (finer) and overhead (coarser).
5. **Accumulator selection**: FP32 accumulation is recommended for training; reduced precision may suffice for inference.
6. **SFA/SFB tensors**: Per-block scale factors that normalize data to the optimal range for the target format.
7. **Hardware progression**: Blackwell UMMA instructions provide native block-scaled MMA, reducing software overhead compared to Hopper.
