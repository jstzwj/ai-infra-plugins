# CUTLASS - Chapter 32: Blackwell (SM100+) Features

This reference covers the Blackwell GPU architecture (SM100, SM101, SM103, SM120) features in CUTLASS, including UMMA (Unified Matrix Multiply-Accumulate), block-scaled data types (NVFP4, MXFP4/6/8), Green Contexts, distributed GEMM, FMHA with MLA, narrow precision GEMM, blockwise/groupwise GEMM, and SM variant details.

---

## 32.1 Blackwell Architecture Overview

The NVIDIA Blackwell architecture represents the next generation beyond Hopper, introducing several major innovations:

| Feature | Description | Impact |
|---------|-------------|--------|
| UMMA (Unified MMA) | Unified matrix multiply-accumulate supporting all data types | Single instruction interface for all precision levels |
| Block-scaled data types | NVFP4, MXFP4/6/8 with per-block scale factors | Enables sub-byte precision for inference |
| Green Contexts | SM resource partitioning for concurrent execution | Multiple independent kernels per SM |
| Persistent CLC Scheduler | Dynamic work scheduling across green contexts | Load balancing for persistent kernels |
| Distributed GEMM | Enhanced multi-GPU Tensor Parallelism | Improved NVLink utilization |

### SM Variants

| SM Version | Product | Key Features |
|------------|---------|-------------|
| SM100 | B100/B200 | Full Blackwell feature set |
| SM101 | Consumer Blackwell | Subset of SM100 features |
| SM103 | Variant | Specific configuration |
| SM120 | Future variant | Extended feature set |

---

## 32.2 UMMA (Unified Matrix Multiply-Accumulate)

### 32.2.1 Overview

UMMA is the Blackwell successor to Hopper's WGMMA. It provides a unified instruction interface for all data types, including the new block-scaled formats. Key improvements over WGMMA:

- **Unified interface**: Single instruction format for FP64, FP32, TF32, FP16, BF16, FP8, INT8, and block-scaled types.
- **Higher throughput**: Increased FLOPS per clock compared to WGMMA.
- **Scale factor integration**: Native support for scale factor tensors in block-scaled operations.
- **Flexible accumulator**: Wider accumulator support for high-precision narrow-input combinations.

### 32.2.2 UMMA Instruction Shapes

| Input Types | Accumulator | Instruction Shape (MxNxK) | Notes |
|-------------|-------------|---------------------------|-------|
| FP64 x FP64 | FP64 | 8x8x4 | Double precision |
| FP32 x FP32 | FP32 | 16x16x8 | Standard precision |
| TF32 x TF32 | FP32 | 16x16x16 | TensorFloat-32 |
| FP16 x FP16 | FP32 | 32x64x32, 32x128x16 | Half precision |
| BF16 x BF16 | FP32 | 32x64x32, 32x128x16 | Brain Float |
| E4M3 x E4M3 | FP32 | 32x128x64 | FP8 |
| E5M2 x E4M3 | FP32 | 32x128x64 | FP8 mixed |
| NVFP4 x NVFP4 | FP32 | 32x128x256 | Block-scaled FP4 |
| MXFP4 x MXFP4 | FP32 | 32x128x256 | Block-scaled MX FP4 |
| MXFP8 x MXFP8 | FP32 | 32x128x64 | Block-scaled MX FP8 |

### 32.2.3 UMMA in CUTLASS

```cpp
#include "cutlass/gemm/collective/collective_builder.hpp"

// UMMA GEMM on Blackwell
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 16,
    cutlass::half_t, cutlass::layout::ColumnMajor, 16,
    float,                              // Accumulator
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

// The CollectiveBuilder automatically selects UMMA for SM100+
```

### 32.2.4 UMMA with CuTe Atoms

```cpp
#include "cute/arch/mma_sm100.hpp"

// UMMA atom for FP16 GEMM
using UMMAAtom = cute::SM100_32x64x32_F16F16F32_TN;

// UMMA atom for block-scaled NVFP4
using UMMA_NVFP4 = cute::SM100_32x128x256_NVFP4_NVFP4_F32_SS_TN;

// Create tiled MMA from UMMA atom
auto tiled_mma = cute::make_tiled_mma(
    UMMAAtom{},
    cute::make_layout(cute::Shape<cute::_2, cute::_2, cute::_1>{})
);

// The tiled MMA wraps the UMMA atom for the threadblock-level computation
```

### 32.2.5 UMMA Execution Flow

The UMMA execution on Blackwell follows a similar pipeline to WGMMA on Hopper:

```
1. TMA Load: Global memory -> Shared memory (tiles of A and B)
2. Register Load: Shared memory -> Registers (partition by thread)
3. UMMA: Registers -> Registers (async matrix multiply)
4. Named Barrier: Wait for UMMA completion
5. Epilogue: Registers -> Shared memory -> Global memory (TMA Store)
```

---

## 32.3 Block-Scaled Data Types

### 32.3.1 NVFP4 Format

NVFP4 is NVIDIA's custom 4-bit floating-point format designed for efficient inference:

- **Bit layout**: 1 sign bit, 2 exponent bits, 1 mantissa bit
- **Dynamic range**: Controlled by per-block scale factors
- **Block size**: Typically 16 or 32 elements per scale factor
- **Quantization**: Uniform quantization within each block

```cpp
#include "cutlass/numeric_types.h"

// NVFP4 element type
using NVFP4 = cutlass::float_nvfp4_t;  // 4-bit floating point

// NVFP4 storage: 2 elements per byte
// Each byte stores two NVFP4 values (packed)
using PackedNVFP4 = uint8_t;  // 2 x NVFP4 per byte

// Scale factor type for NVFP4
using ScaleFactor = cutlass::float_e8m0_t;  // E8M0 power-of-two scale
```

### 32.3.2 MXFP4, MXFP6, and MXFP8 Formats

The MX (Microscaling) formats are industry-standard block-scaled formats:

```cpp
#include "cutlass/numeric_types.h"

// MXFP4: 4-bit floating point with microscaling
using MXFP4 = cutlass::float_mxfp4_t;

// MXFP6: 6-bit floating point with microscaling
// Two variants: E2M3 and E3M2
using MXFP6_E2M3 = cutlass::float_mxfp6_e2m3_t;
using MXFP6_E3M2 = cutlass::float_mxfp6_e3m2_t;

// MXFP8: 8-bit floating point with microscaling
// Two variants: E4M3 and E5M2
using MXFP8_E4M3 = cutlass::float_mxfp8_e4m3_t;
using MXFP8_E5M2 = cutlass::float_mxfp8_e5m2_t;

// All MX formats use block-level scale factors
// Block size is defined by the OCP (Open Compute Project) specification
// Default block size: 32 elements per scale factor
```

MX format comparison:

| Format | Bits | Exponent | Mantissa | Block Size | Scale Type |
|--------|------|----------|----------|------------|------------|
| MXFP4 | 4 | 2 | 1 | 32 | E8M0 |
| MXFP6 E2M3 | 6 | 2 | 3 | 32 | E8M0 |
| MXFP6 E3M2 | 6 | 3 | 2 | 32 | E8M0 |
| MXFP8 E4M3 | 8 | 4 | 3 | 32 | E8M0 |
| MXFP8 E5M2 | 8 | 5 | 2 | 32 | E8M0 |
| NVFP4 | 4 | 2 | 1 | 16/32 | E8M0 |

### 32.3.3 Scale Factor Tensors (SFA, SFB)

Block-scaled GEMM requires scale factor tensors alongside the narrow-precision data:

```cpp
// Scale factor tensors:
// SFA: Scale factors for matrix A, shape [M/BLK_M, K/BLK_K]
// SFB: Scale factors for matrix B, shape [N/BLK_N, K/BLK_K]
//
// The GEMM computes: D = (A * SFA) @ (B * SFB)^T
// Where A and B are narrow-precision, SFA and SFB are per-block scale factors

// Scale factor layout:
// For NVFP4 with block_size=16:
//   SFA shape: [ceil(M/16), ceil(K/16)]   (one scale per 16x16 block of A)
//   SFB shape: [ceil(N/16), ceil(K/16)]   (one scale per 16x16 block of B)

#include "cutlass/gemm/gemm.h"

// Arguments for block-scaled GEMM
typename GemmOp::Arguments args{
    {M, N, K},                  // Problem size
    {ptr_A, stride_A},          // Matrix A (NVFP4)
    {ptr_B, stride_B},          // Matrix B (NVFP4)
    {ptr_SFA, stride_SFA},      // Scale factors for A (E8M0)
    {ptr_SFB, stride_SFB},      // Scale factors for B (E8M0)
    {ptr_C, stride_C},          // Matrix C (source)
    {ptr_D, stride_D},          // Matrix D (output)
    {alpha, beta}               // Epilogue parameters
};
```

### 32.3.4 Block-Scaled GEMM Configuration

```cpp
#include "cutlass/gemm/collective/collective_builder.hpp"

// NVFP4 block-scaled GEMM on Blackwell
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass::float_nvfp4_t, cutlass::layout::RowMajor, 32,   // NVFP4 A
    cutlass::float_nvfp4_t, cutlass::layout::ColumnMajor, 32, // NVFP4 B
    float,                                                      // Accumulator
    cutlass::gemm::GemmShape<128, 128, 256>,                   // Larger K tile for narrow types
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

### 32.3.5 Quantization to Block-Scaled Types

```cpp
// Quantize FP16/BF16 tensors to NVFP4 with scale factors
#include "cutlass/transform/threadblock/predicated_scale_bias_vector_iterator.h"

// Quantization steps:
// 1. Divide the input tensor into blocks of BLK_SIZE elements
// 2. Find the maximum absolute value in each block
// 3. Compute the scale factor: scale = max_val / FP4_max
// 4. Quantize each element: fp4_value = round(element / scale)

// CUTLASS provides quantization utilities:
template <int BlockSize>
void quantize_to_nvfp4(
    const cutlass::half_t *input,    // FP16 input [M, K]
    uint8_t *output,                 // NVFP4 output (packed) [M, K/2]
    cutlass::float_e8m0_t *scales,   // Scale factors [M, K/BlockSize]
    int M, int K
) {
    int num_blocks = M * (K / BlockSize);

    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        int row = block_idx / (K / BlockSize);
        int col_block = block_idx % (K / BlockSize);

        // Find max in block
        float max_val = 0.0f;
        for (int i = 0; i < BlockSize; ++i) {
            float val = fabsf(float(input[row * K + col_block * BlockSize + i]));
            max_val = fmaxf(max_val, val);
        }

        // Compute scale (E8M0 = power of 2)
        float scale = max_val / 6.0f;  // NVFP4 max representable = 6.0
        // Round to nearest power of 2 for E8M0

        // Quantize elements
        for (int i = 0; i < BlockSize; i += 2) {
            float v0 = float(input[row * K + col_block * BlockSize + i]) / scale;
            float v1 = float(input[row * K + col_block * BlockSize + i + 1]) / scale;
            // Convert to NVFP4 and pack two values into one byte
            uint8_t packed = (nvfp4_from_float(v0) << 4) | nvfp4_from_float(v1);
            output[row * (K/2) + col_block * (BlockSize/2) + i/2] = packed;
        }

        scales[block_idx] = cutlass::float_e8m0_t(scale);
    }
}
```

---

## 32.4 Green Contexts

### 32.4.1 Overview

Green Contexts are a Blackwell feature that enables SM resource partitioning, allowing multiple independent kernels (or kernel phases) to execute concurrently on the same SM. This enables:

- **Concurrent kernel execution**: Multiple lightweight kernels share SM resources.
- **Persistent kernel scheduling**: A scheduler kernel manages work allocation dynamically.
- **Reduced launch overhead**: Persistent kernels stay resident on SMs.

### 32.4.2 SM Resource Partitioning

Each Green Context defines a partition of SM resources:

```cpp
#include "cutlass/arch/green_context.hpp"

// Green Context configuration
struct GreenContextConfig {
    // Number of warps allocated to this context
    int num_warps;

    // Shared memory allocation
    size_t smem_size;

    // Register allocation per warp
    int regs_per_warp;

    // Whether this context can yield to other contexts
    bool preemptible;
};

// Partition an SM into two green contexts:
// Context 0: GEMM kernel (main workload)
// Context 1: Scheduler (lightweight, manages work distribution)
GreenContextConfig ctx_gemm = {
    .num_warps = 8,
    .smem_size = 128 * 1024,
    .regs_per_warp = 256,
    .preemptible = false
};

GreenContextConfig ctx_scheduler = {
    .num_warps = 2,
    .smem_size = 4 * 1024,
    .regs_per_warp = 64,
    .preemptible = true
};
```

### 32.4.3 Dynamic Persistent CLC Scheduler

The Command List Controller (CLC) scheduler is a persistent kernel that dynamically assigns work to green contexts:

```cpp
#include "cutlass/gemm/kernel/sm100_gemm_scheduler.hpp"

// CLC Scheduler: a persistent kernel that manages work distribution
// across green contexts on each SM

// The scheduler maintains a work queue:
// 1. Each SM runs a persistent scheduler kernel
// 2. The scheduler reads work items from a global queue
// 3. It dispatches work to green contexts based on availability
// 4. When a green context completes its work, it signals the scheduler
// 5. The scheduler assigns the next work item

// CLC Scheduler configuration
using Scheduler = cutlass::gemm::kernel::Sm100PersistentScheduler<
    cutlass::gemm::GemmShape<128, 128, 64>,   // Tile shape
    2,                                          // Max concurrent tiles per SM
    cutlass::gemm::kernel::SchedulerMode::Dynamic
>;

// The scheduler is integrated into the kernel launch:
using Kernel = cutlass::gemm::kernel::GemmUniversal<
    CollectiveOp,
    EpilogueOp,
    Scheduler  // Persistent scheduler
>;
```

### 32.4.4 Static Persistent Scheduler

For predictable workloads, a static scheduler pre-assigns tiles to SMs:

```cpp
// Static scheduler: tiles are pre-assigned to SMs at launch time
using StaticScheduler = cutlass::gemm::kernel::Sm100PersistentScheduler<
    cutlass::gemm::GemmShape<128, 128, 64>,
    1,  // One tile per SM at a time
    cutlass::gemm::kernel::SchedulerMode::Static
>;

// The static scheduler is simpler but less adaptive:
// - Each SM gets a fixed set of tiles
// - No load balancing across SMs
// - Best for uniform tile sizes (square GEMM)
```

### 32.4.5 Partition Stream Management

Green contexts use partition streams to manage concurrent execution:

```cpp
// Partition streams allow multiple green contexts to submit work
// to the same SM concurrently

// Each partition stream is an independent execution context:
// - Separate register file allocation
// - Separate shared memory allocation
// - Separate program counter and execution state

// In CUTLASS, partition streams are managed implicitly by the
// kernel scheduler. The user does not need to manually manage streams.

// The number of partition streams per SM is configurable:
// - More streams = more concurrent contexts = smaller per-context resources
// - Fewer streams = fewer concurrent contexts = larger per-context resources

// Typical configuration:
// 1 partition stream: Full SM resources for the kernel (no green contexts)
// 2 partition streams: Split SM between kernel and scheduler
// 4 partition streams: Multiple concurrent kernels
```

---

## 32.5 Distributed GEMM on Blackwell

### 32.5.1 Enhanced Tensor Parallelism

Blackwell extends the Hopper distributed GEMM with improved communication primitives:

```cpp
#include "cutlass/gemm/collective/collective_builder.hpp"

// Distributed GEMM on Blackwell
using DistributedCollective = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccum,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelTmaWarpSpecializedPingpong
>::CollectiveOp;
```

### 32.5.2 Communication Patterns

```cpp
// Blackwell distributed GEMM supports:
//
// 1. Column Parallel (Megatron-LM style):
//    Y = X @ W, where W is split column-wise across N GPUs
//    Each GPU: Y_i = X @ W_i (no communication for forward)
//    Backward: All-Reduce for gradient of X
//
// 2. Row Parallel:
//    Y = W @ X, where W is split row-wise
//    Each GPU: Y_i = W_i @ X
//    Forward: All-Reduce(Y_0, Y_1, ..., Y_{N-1}) = Y
//
// 3. Ring Attention (sequence parallel):
//    Q, K, V are split across GPUs along the sequence dimension
//    Each GPU processes its local chunk and rotates K/V via NVLink

// Blackwell-specific optimizations:
// - Larger NVLink bandwidth (up to 1.8 TB/s per GPU)
// - TMA-based communication (hardware-managed transfers)
// - Overlap of UMMA computation with NVLink transfers
```

---

## 32.6 FMHA on Blackwell

### 32.6.1 FMHA Overview

Blackwell FMHA builds on the Hopper implementation with UMMA instructions and additional optimizations:

```cpp
#include "cutlass/gemm/gemm.h"

// FMHA on Blackwell uses:
// - UMMA for Q*K^T and P*V matrix multiplies
// - TMA for loading Q, K, V tiles from global memory
// - Online softmax (flash attention algorithm)
// - Optional causal masking
```

### 32.6.2 MLA (Multi-head Latent Attention)

MLA is a memory-efficient attention variant that compresses the Key and Value projections into a lower-dimensional latent space:

```
Standard Attention:
  Q = X @ W_Q  [B, S, H, D]
  K = X @ W_K  [B, S, H, D]
  V = X @ W_V  [B, S, H, D]
  O = softmax(Q @ K^T) @ V

MLA (Multi-head Latent Attention):
  C = X @ W_D  [B, S, R]  (latent compression, R << H*D)
  Q = X @ W_Q  [B, S, H, D]
  K = C @ W_UK [B, S, H, D]  (up-project from latent)
  V = C @ W_UV [B, S, H, D]  (up-project from latent)

  Key insight: K and V share the latent representation C
  Only C (not full K, V) is stored in KV cache -> massive memory savings
```

### 32.6.3 Weight Absorption

Weight absorption is a critical optimization for MLA that fuses the latent-to-key/value projection with the attention computation:

```
Standard MLA computation:
  K = C @ W_UK  -> S = Q @ K^T = Q @ (C @ W_UK)^T = Q @ W_UK^T @ C^T

Weight Absorption:
  Absorb W_UK^T into Q projection:
    Q' = Q @ W_UK^T   (absorb the weight matrix)
    S = Q' @ C^T       (now GEMM with compressed C, much smaller)

  This eliminates the materialization of full K matrix
  and reduces the GEMM size from [M, D] x [D, S] to [M, D] x [R, S]
  where R << D
```

```cpp
// MLA with weight absorption in CUTLASS
// The weight absorption is implemented as a fused GEMM:
//   Q' = Q @ W_UK^T  (GEMM 1: [B, H, M, D] x [H, D, D] -> [B, H, M, D])
//   S = Q' @ C^T     (GEMM 2: [B, H, M, D] x [B, S, R] -> [B, H, M, S])
//   P = softmax(S)
//   O = P @ V_absorbed (GEMM 3: using absorbed V weights)

// In CUTLASS, this can be implemented as a fused multi-GEMM kernel:
//   The output of GEMM 1 feeds directly into GEMM 2 without writing to memory
```

---

## 32.7 Narrow Precision GEMM

### 32.7.1 NVFP4 GEMM

```cpp
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

// Complete NVFP4 GEMM example
using NVFP4Gemm = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass::float_nvfp4_t, cutlass::layout::RowMajor, 32,
    cutlass::float_nvfp4_t, cutlass::layout::ColumnMajor, 32,
    float,                              // FP32 accumulator
    cutlass::gemm::GemmShape<128, 128, 256>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::layout::RowMajor,
    cutlass::layout::RowMajor,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>;

using Kernel = cutlass::gemm::kernel::GemmUniversal<NVFP4Gemm, EpilogueOp>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

// Launch
Gemm gemm_op;
typename Gemm::Arguments args{
    {M, N, K},
    {ptr_A, stride_A},          // NVFP4 data
    {ptr_B, stride_B},          // NVFP4 data
    {ptr_SFA, stride_SFA},      // Scale factors for A
    {ptr_SFB, stride_SFB},      // Scale factors for B
    {ptr_C, stride_C},          // Source matrix
    {ptr_D, stride_D},          // Output matrix
    {alpha, beta}
};
gemm_op(args);
```

### 32.7.2 MXFP8 GEMM

```cpp
// MXFP8 GEMM (8-bit with microscaling)
using MXFP8Gemm = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass::float_mxfp8_e4m3_t, cutlass::layout::RowMajor, 32,
    cutlass::float_mxfp8_e4m3_t, cutlass::layout::ColumnMajor, 32,
    float,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

### 32.7.3 Mixed Block-Scaled GEMM

```cpp
// Mixed block-scaled: NVFP4 input, FP16 output
// This is common for inference: weights in NVFP4, activations in FP16
using MixedBlockScaled = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass::float_nvfp4_t, cutlass::layout::RowMajor, 32,     // A: NVFP4 (weights)
    cutlass::half_t, cutlass::layout::ColumnMajor, 16,          // B: FP16 (activations)
    float,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

---

## 32.8 Blockwise/Groupwise GEMM

### 32.8.1 Blockwise Scaling

Blockwise GEMM applies scale factors at a block granularity rather than globally:

```cpp
// Blockwise GEMM:
// D[i,j] = alpha * sum_k( A[i,k] * SFA[i,k_block] * B[k,j] * SFB[j,k_block] ) + beta * C[i,j]
//
// Where:
//   SFA[i,k_block] is the scale factor for block (i, k_block) of matrix A
//   SFB[j,k_block] is the scale factor for block (j, k_block) of matrix B
//   k_block = k / BLOCK_SIZE_K
//
// The scale factors are applied during the UMMA instruction, not as separate operations

// Blockwise GEMM configuration
using BlockwiseGemm = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass::float_nvfp4_t, cutlass::layout::RowMajor, 32,
    cutlass::float_nvfp4_t, cutlass::layout::ColumnMajor, 32,
    float,
    cutlass::gemm::GemmShape<128, 128, 256>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

### 32.8.2 Groupwise Quantization Pattern

```cpp
// Groupwise quantization: quantize groups of elements independently
// Each group has its own scale factor and (optionally) zero-point

// Group configuration:
//   Group size: 16, 32, 64, or 128 elements
//   Scale type: E8M0 (power of 2) or E4M3 (FP8)
//   Data type: NVFP4, MXFP4, INT4

template <typename ElementInput, typename ElementQuantized, int GroupSize>
struct GroupwiseQuantizer {
    // Quantize a group of elements
    static void quantize_group(
        const ElementInput *input,
        ElementQuantized *output,
        cutlass::float_e8m0_t *scale,
        int num_groups
    ) {
        for (int g = 0; g < num_groups; ++g) {
            // Find max in group
            float max_val = find_max(input + g * GroupSize, GroupSize);

            // Compute scale
            float s = max_val / max_representable<ElementQuantized>();

            // Quantize
            for (int i = 0; i < GroupSize; ++i) {
                output[g * GroupSize + i] = quantize(input[g * GroupSize + i] / s);
            }

            // Store scale
            scale[g] = cutlass::float_e8m0_t(s);
        }
    }
};
```

---

## 32.9 SM101, SM103, SM120 Variations

### 32.9.1 SM101 (Consumer Blackwell)

SM101 is a consumer-grade variant of Blackwell with reduced feature set:

```cpp
// SM101 supports:
// - UMMA with FP16, BF16, INT8 (no block-scaled types)
// - TMA load/store
// - Thread block clusters
// - Warp specialization
// - No Green Contexts
// - No NVFP4/MXFP support

using SM101Collective = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm101, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 16,
    cutlass::half_t, cutlass::layout::ColumnMajor, 16,
    float,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

### 32.9.2 SM103

SM103 is a specialized variant with specific hardware configurations:

```cpp
// SM103 supports:
// - Full SM100 UMMA feature set
// - Enhanced TMA with larger descriptor support
// - Modified shared memory configuration
// - Specific register file layout changes

using SM103Collective = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm103, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccum,
    TileShape,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

### 32.9.3 SM120

SM120 is a future Blackwell variant with extended capabilities:

```cpp
// SM120 extends SM100 with:
// - Larger shared memory per SM
// - Enhanced UMMA instruction set
// - Extended block-scaled type support
// - Improved NVLink integration
// - Additional green context partition modes

using SM120Collective = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccum,
    TileShape,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

### 32.9.4 Architecture Feature Matrix

| Feature | SM100 | SM101 | SM103 | SM120 |
|---------|-------|-------|-------|-------|
| UMMA | Yes | Partial | Yes | Yes |
| NVFP4 | Yes | No | Yes | Yes |
| MXFP4/6/8 | Yes | No | Yes | Yes |
| Green Contexts | Yes | No | Partial | Yes |
| TMA Load/Store | Yes | Yes | Yes | Yes |
| Thread Block Clusters | Yes | Yes | Yes | Yes |
| Warp Specialization | Yes | Yes | Yes | Yes |
| Persistent Scheduler | Yes | No | Partial | Yes |
| Distributed GEMM | Yes | Yes | Yes | Yes |
| FP64 Tensor Core | Yes | No | Yes | Yes |
| Max Shared Memory/SM | 228 KB | 100 KB | 228 KB | 256 KB |

---

## 32.10 Complete Blackwell GEMM Example

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

// Blackwell SM100 NVFP4 GEMM
// D = alpha * (A * SFA) @ (B * SFB)^T + beta * C

namespace {

using ElementA = cutlass::float_nvfp4_t;   // NVFP4 input A
using ElementB = cutlass::float_nvfp4_t;   // NVFP4 input B
using ElementC = cutlass::half_t;          // FP16 source
using ElementD = cutlass::half_t;          // FP16 output
using ElementAccum = float;               // FP32 accumulator
using ElementScale = cutlass::float_e8m0_t; // E8M0 scale factor

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// Use CollectiveBuilder for automatic UMMA configuration
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 32,   // 32-byte alignment for NVFP4
    ElementB, LayoutB, 32,
    ElementAccum,
    cutlass::gemm::GemmShape<128, 128, 256>,  // Large K for narrow types
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
    LayoutD, LayoutC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>;

using Kernel = cutlass::gemm::kernel::GemmUniversal<
    CollectiveOp, EpilogueOp
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

} // namespace

void run_blackwell_nvfp4_gemm(
    // NVFP4 data (packed: 2 elements per byte)
    const uint8_t *ptr_A, int lda,
    const uint8_t *ptr_B, int ldb,
    // Scale factors
    const ElementScale *ptr_SFA, int stride_SFA,
    const ElementScale *ptr_SFB, int stride_SFB,
    // Source and output
    const ElementC *ptr_C, int ldc,
    ElementD *ptr_D, int ldd,
    // Problem dimensions
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream
) {
    Gemm gemm_op;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmCoord{M, N, K},
        {ptr_A, lda},
        {ptr_B, ldb},
        {ptr_SFA, stride_SFA},      // Scale factors for A
        {ptr_SFB, stride_SFB},      // Scale factors for B
        {ptr_C, ldc},
        {ptr_D, ldd},
        {alpha, beta}
    };

    size_t workspace_size = Gemm::get_workspace_size(args);
    void *workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    auto status = gemm_op(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "Blackwell GEMM failed: %d\n", (int)status);
    }

    if (workspace) {
        cudaFree(workspace);
    }
}
```

---

## 32.11 Green Context GEMM Example

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/kernel/sm100_gemm_scheduler.hpp"

// Green Context GEMM with persistent CLC scheduler

using CollectiveOp = /* ... as above ... */;

using EpilogueOp = /* ... as above ... */;

// Persistent scheduler for green contexts
using Scheduler = cutlass::gemm::kernel::Sm100PersistentScheduler<
    cutlass::gemm::GemmShape<128, 128, 256>,
    2,  // Max concurrent tiles per SM
    cutlass::gemm::kernel::SchedulerMode::Dynamic
>;

using Kernel = cutlass::gemm::kernel::GemmUniversal<
    CollectiveOp,
    EpilogueOp,
    Scheduler
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

// The green context configuration is handled internally by the scheduler.
// The user launches the kernel normally, and the persistent scheduler
// manages SM partitioning and work distribution automatically.

void run_green_context_gemm(/* ... */) {
    Gemm gemm_op;
    typename Gemm::Arguments args{ /* ... */ };

    // The kernel launch uses more CTAs than physical SMs
    // (over-subscription), and the persistent scheduler manages
    // which CTAs are active on each SM via green contexts.
    gemm_op(args);
}
```

---

## 32.12 Performance Tuning for Blackwell

### 32.12.1 Tile Size Selection

For block-scaled types, larger K tile sizes are recommended to amortize the scale factor overhead:

```cpp
// NVFP4: Use large K tile (256 or 512) because each NVFP4 element is only 4 bits
cutlass::gemm::GemmShape<128, 128, 256>  // Good for NVFP4

// MXFP8: K tile of 64-128 is sufficient
cutlass::gemm::GemmShape<128, 128, 128>  // Good for MXFP8

// FP16: Standard tile sizes apply
cutlass::gemm::GemmShape<128, 128, 64>   // Good for FP16
```

### 32.12.2 Shared Memory Considerations

```cpp
// Blackwell shared memory allocation:
// For NVFP4 with GemmShape<128, 128, 256>:
//   A tile: 128 * 256 * 0.5 bytes = 16 KB (NVFP4 = 4 bits)
//   B tile: 256 * 128 * 0.5 bytes = 16 KB
//   Scale factors: (128*256/16) * 1 byte + (128*256/16) * 1 byte = 4 KB
//   Total per stage: ~36 KB
//   With 4 stages: ~144 KB shared memory

// Ensure the tile shape + stage count fits within SM shared memory
// Use StageCountAutoCarveout to let CUTLASS choose the optimal stage count
```

### 32.12.3 Occupancy Guidelines

| Tile Shape | Stages | Shared Memory | Occupancy |
|------------|--------|---------------|-----------|
| 128x128x64 (FP16) | 3 | ~72 KB | 2-3 CTAs/SM |
| 128x128x256 (NVFP4) | 3 | ~108 KB | 1-2 CTAs/SM |
| 256x128x64 (FP16) | 2 | ~96 KB | 1-2 CTAs/SM |

---

## 32.13 Summary

Blackwell (SM100+) introduces several transformative features to CUTLASS:

- **UMMA (Unified MMA)**: A unified matrix multiply-accumulate instruction supporting all data types from FP64 down to NVFP4, with integrated scale factor support for block-scaled operations.

- **Block-Scaled Data Types**: NVFP4 (4-bit), MXFP4/6/8 formats with per-block scale factors (SFA, SFB), enabling high-throughput inference with minimal accuracy loss.

- **Green Contexts**: SM resource partitioning that enables multiple independent kernel contexts to execute concurrently on the same SM, managed by persistent CLC or static schedulers.

- **Distributed GEMM**: Enhanced Tensor Parallelism with improved NVLink bandwidth utilization and communication-computation overlap.

- **FMHA with MLA**: Fused Multi-Head Attention with Multi-head Latent Attention support, including weight absorption optimization that eliminates the need to materialize full KV matrices.

- **SM Variants**: SM100 (full features), SM101 (consumer, no block-scaled types or green contexts), SM103, and SM120, each with specific feature subsets and resource configurations.
