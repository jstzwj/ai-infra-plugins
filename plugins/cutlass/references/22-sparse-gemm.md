# CUTLASS Reference - Chapter 22: Sparse GEMM

This reference covers sparse GEMM operations in CUTLASS, focusing on structured sparsity support via NVIDIA Tensor Cores. The 2:4 structured sparsity pattern provides up to 2x throughput improvement for matrix multiplication by exploiting zero elements known at the hardware level.

---

## 22.1 Sparse Tensor Core Operations

### 22.1.1 Structured Sparsity Overview

Starting with Ampere (SM80), NVIDIA Tensor Cores support a **2:4 structured sparsity** pattern. In this pattern, for every group of 4 consecutive elements in a row (or column) of a matrix, exactly 2 elements are zero and 2 are non-zero. The Tensor Core hardware can skip the multiply-by-zero operations, effectively doubling the throughput for the non-zero elements.

Key properties of structured sparsity:
- **2:4 pattern**: Out of every 4 elements, exactly 2 must be zero.
- **50% sparsity**: Half the elements are zero.
- **2x throughput**: The Tensor Core processes the 2 non-zero elements at the same rate it would process 4 dense elements.
- **Hardware support**: SM80+ Tensor Cores (Ampere, Ada, Hopper, Blackwell).
- **Transparent to output**: The result is mathematically equivalent to dense GEMM on the pruned matrix.

### 22.1.2 OpClassSparseTensorOp

CUTLASS uses `cutlass::arch::OpClassSparseTensorOp` to indicate sparse Tensor Core operations. This is analogous to `OpClassTensorOp` for dense operations:

```cpp
// Dense Tensor Core operation
using DenseOpClass = cutlass::arch::OpClassTensorOp;

// Sparse Tensor Core operation
using SparseOpClass = cutlass::arch::OpClassSparseTensorOp;
```

When using `OpClassSparseTensorOp`, the GEMM expects:
1. A **sparse matrix** operand (A) with 2:4 structured sparsity.
2. A **metadata tensor** that encodes which elements are non-zero.
3. A **dense matrix** operand (B).

---

## 22.2 Sparse Meta-data Format

### 22.2.1 Encoding the Sparsity Pattern

The 2:4 sparsity pattern is encoded in a **metadata tensor** that runs alongside the sparse matrix. For every group of 4 elements, a 2-bit index indicates which 2 of the 4 original positions contain non-zero values.

Consider a row of a matrix with 8 elements:

```
Original:   [a0, a1, a2, a3, a4, a5, a6, a7]
Pruned:     [a0,  0, a2,  0,  0, a5, a6,  0]  (2:4 pattern applied)
Stored:     [a0, a2 | a5, a6]                    (only non-zero values)
Metadata:   [0b00 | 0b01]                        (indices of non-zero positions)
```

The metadata encoding for each group of 4:
- `0b00`: elements at positions 0 and 2 are non-zero
- `0b01`: elements at positions 0 and 3 are non-zero
- `0b10`: elements at positions 1 and 2 are non-zero
- `0b11`: elements at positions 1 and 3 are non-zero

Actually, the 2-bit encoding uses 4 possible values:

| Metadata Value | Non-zero Positions |
|---|---|
| 0 | Indices 0, 1 |
| 1 | Indices 0, 2 |
| 2 | Indices 0, 3 |
| 3 | Indices 1, 2 |

(Exact encoding may vary by architecture; CUTLASS handles this internally.)

### 22.2.2 Metadata Storage

The metadata is stored as a separate tensor with a compact representation:
- **2 bits per group of 4 elements** in the sparse dimension.
- For FP16 with a sparse K-dimension of 64, the metadata per row is `64/4 * 2 bits = 32 bits = 4 bytes`.
- CUTLASS packs the 2-bit values into 8-bit, 16-bit, or 32-bit words depending on the configuration.

The metadata tensor has shape:
- For GEMM with A sparse (M x K): metadata shape is (M, K/4) with 2-bit elements packed into larger words.

### 22.2.3 CUTLASS Sparse Metadata Types

CUTLASS provides the `cutlass::layout::SparseMetadata` layout for managing metadata tensors:

```cpp
// The sparse metadata layout encodes how 2-bit indices map to memory
// In CUTLASS 2.x:
using LayoutMeta = cutlass::layout::ColumnMajorInterleaved<4>;

// In CUTLASS 3.x, metadata is handled by the sparse collective builder
```

---

## 22.3 Sparse Matrix Storage Format

### 22.3.1 Compressed Storage

The sparse operand is stored in a **compressed format** where only non-zero elements are retained. Since 2:4 sparsity removes exactly half the elements, the sparse matrix has half the memory footprint of the dense equivalent:

```
Dense matrix A (M x K):    M * K elements
Sparse matrix A (M x K):   M * K/2 elements (only non-zero)
Metadata:                   M * K/4 * 2 bits
```

For example, an FP16 matrix with M=1024, K=4096:
- Dense storage: 1024 * 4096 * 2 bytes = 8 MB
- Sparse storage: 1024 * 2048 * 2 bytes = 4 MB (50% savings)
- Metadata: 1024 * 1024 * 0.25 bytes = 256 KB

### 22.3.2 Sparse Matrix Layout

The sparse matrix retains the same logical layout (RowMajor or ColumnMajor) but with a compressed stride. CUTLASS handles the stride adjustment:

```cpp
// Dense A: M x K with stride lda = K (for RowMajor)
// Sparse A: M x K/2 with stride lda_sparse = K/2

// When allocating memory for sparse A:
int M = 1024;
int K = 4096;
int K_sparse = K / 2;  // Only non-zero elements stored

cutlass::half_t* sparse_A;
cudaMalloc(&sparse_A, M * K_sparse * sizeof(cutlass::half_t));

// Metadata tensor
uint32_t* metadata_A;
cudaMalloc(&metadata_A, M * (K / 4) * sizeof(uint32_t));  // Packed 2-bit indices
```

---

## 22.4 Sparse GEMM Device API

### 22.4.1 CUTLASS 2.x Sparse GEMM

In CUTLASS 2.x, sparse GEMM is accessed via `cutlass::gemm::device::SparseGemm`:

```cpp
#include "cutlass/gemm/device/sparse_gemm.h"

using SparseGemm = cutlass::gemm::device::SparseGemm<
    cutlass::half_t, cutlass::layout::RowMajor,       // ElementA (sparse), LayoutA
    cutlass::half_t, cutlass::layout::ColumnMajor,     // ElementB (dense), LayoutB
    float, cutlass::layout::RowMajor,                  // ElementC, LayoutC
    float,                                              // ElementAccumulator
    cutlass::arch::OpClassSparseTensorOp,               // Sparse Tensor Op
    cutlass::arch::Sm80,                                // Architecture
    cutlass::gemm::GemmShape<128, 128, 64>,             // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 64>,               // WarpShape
    cutlass::gemm::GemmShape<16, 8, 32>,                // InstructionShape (sparse)
    cutlass::epilogue::thread::LinearCombination<
        float, 4, float, float>,                        // Epilogue
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4                                                   // Stages
>;

// Arguments for sparse GEMM
typename SparseGemm::Arguments args(
    {M, N, K},             // Problem dimensions (original K, not K/2)
    sparse_A, lda_sparse,  // Sparse A data (compressed)
    metadata_A, lda_meta,  // Metadata tensor
    B, ldb,                // Dense B data
    C, ldc,                // C data
    D, ldd,                // D output
    alpha, beta            // Scaling factors
);

SparseGemm sparse_gemm_op;
auto status = sparse_gemm_op(args);
```

Note that the `K` in `GemmCoord{M, N, K}` refers to the **original** K dimension (before compression). CUTLASS internally divides by 2 for the sparse operand.

### 22.4.2 Instruction Shape for Sparse GEMM

Sparse Tensor Core instructions have a different instruction shape than dense operations. The sparse MMA instruction processes twice as many output elements per instruction because it skips zeros:

| Data Type | Dense Instruction Shape | Sparse Instruction Shape |
|---|---|---|
| FP16 | 16x8x16 | 16x8x32 |
| BF16 | 16x8x16 | 16x8x32 |
| INT8 | 16x8x32 | 16x8x64 |
| TF32 | 16x8x8 | 16x8x16 |

The K dimension of the instruction is doubled because the hardware processes 2 non-zero elements for every 4 positions, effectively covering 2x the logical K range per instruction.

---

## 22.5 SM80 Sparse Operations (Ampere)

### 22.5.1 Ampere Sparse Tensor Core

Ampere (SM80) introduced sparse Tensor Core support with the following capabilities:

- **Supported types**: FP16, BF16, INT8, TF32
- **Sparsity pattern**: 2:4 structured sparsity (50% zeros)
- **Sparse operand**: Only operand A can be sparse
- **Throughput**: 2x compared to dense Tensor Core operations

```cpp
// SM80 Sparse FP16 GEMM
using SparseGemmFP16 = cutlass::gemm::device::SparseGemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSparseTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,    // Sparse instruction: K=32 (doubled from 16)
    cutlass::epilogue::thread::LinearCombination<
        float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4
>;
```

### 22.5.2 SM80 Sparse INT8 GEMM

```cpp
// SM80 Sparse INT8 GEMM for quantized inference
using SparseGemmINT8 = cutlass::gemm::device::SparseGemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassSparseTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,    // Sparse INT8 instruction
    cutlass::epilogue::thread::LinearCombination<
        int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4
>;
```

---

## 22.6 SM90 Sparse TMA GMMA (Hopper)

### 22.6.1 Hopper Sparse GEMM Architecture

Hopper (SM90) extends sparse GEMM support with TMA (Tensor Memory Accelerator) integration and GMMA (GeMM MMA Assembly) instructions. Key improvements:

- **TMA-based data movement**: Sparse data and metadata are loaded via TMA, reducing register pressure on the producer warp group.
- **Larger tile sizes**: Sparse GMMA supports larger instruction tiles.
- **Warp specialization**: Producer-consumer pattern with dedicated warp groups.
- **FP8 sparse support**: `float_e4m3_t` and `float_e5m2_t` sparse operations.

### 22.6.2 SM90 Sparse GEMM with CollectiveBuilder

```cpp
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

// SM90 Sparse FP16 GEMM
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementAccumulator = float;

using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassSparseTensorOp,       // Sparse operation
    ElementA, cutlass::layout::RowMajor, 8,
    ElementB, cutlass::layout::ColumnMajor, 8,
    ElementAccumulator,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::layout::RowMajor, cutlass::layout::RowMajor,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>;

using Kernel = cutlass::gemm::kernel::GemmUniversal<
    CollectiveOp, EpilogueOp,
    cutlass::gemm::kernel::GemmUniversal::GuestKernel<  // Sparse requires guest kernel
        cutlass::gemm::kernel::SparseGemmConfiguration
    >
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
```

### 22.6.3 SM90 Sparse FP8 GEMM

```cpp
// SM90 Sparse FP8 GEMM (e4m3 for both A and B)
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementAccumulator = float;

using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassSparseTensorOp,
    ElementA, cutlass::layout::RowMajor, 16,
    ElementB, cutlass::layout::ColumnMajor, 16,
    ElementAccumulator,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

### 22.6.4 Launching SM90 Sparse GEMM

```cpp
void run_sparse_gemm_sm90(
    int M, int N, int K,
    const cutlass::half_t* sparse_A, int64_t lda_sparse,
    const uint16_t* metadata_A, int64_t lda_meta,
    const cutlass::half_t* B, int64_t ldb,
    const float* C, int64_t ldc,
    float* D, int64_t ldd,
    float alpha, float beta,
    cudaStream_t stream = 0
) {
    using Gemm = /* ... as defined above ... */;
    Gemm gemm_op;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmCoord{M, N, K},
        {sparse_A, lda_sparse},     // Sparse A (compressed, M x K/2)
        {metadata_A, lda_meta},     // Metadata (M x K/4)
        {B, ldb},                   // Dense B (K x N)
        {C, ldc},                   // C (M x N)
        {D, ldd},                   // D output (M x N)
        {alpha, beta}
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

## 22.7 Sparse Conversion Utilities

### 22.7.1 Pruning to 2:4 Sparsity

CUTLASS provides utilities for converting a dense matrix to 2:4 sparse format. The pruning process selects which elements to zero out to achieve the best accuracy:

```cpp
#include "cutlass/util/reference/device/sparse_gemm.h"

// Host-side: Prune a dense matrix to 2:4 sparsity
// This selects the 2 largest magnitude elements in each group of 4
template <typename Element>
void prune_to_2_4_sparsity(
    const Element* dense_matrix,
    Element* sparse_matrix,
    uint32_t* metadata,
    int rows, int cols,
    cutlass::layout::RowMajor layout
) {
    for (int row = 0; row < rows; ++row) {
        for (int group = 0; group < cols / 4; ++group) {
            // Find the 2 largest elements in this group of 4
            int base = row * cols + group * 4;
            float max1 = std::abs(float(dense_matrix[base]));
            float max2 = std::abs(float(dense_matrix[base + 1]));
            int idx1 = 0, idx2 = 1;

            for (int i = 2; i < 4; ++i) {
                float val = std::abs(float(dense_matrix[base + i]));
                if (val > max1 || val > max2) {
                    if (max1 <= max2) {
                        max1 = val; idx1 = i;
                    } else {
                        max2 = val; idx2 = i;
                    }
                }
            }

            // Encode metadata
            // (Exact encoding depends on hardware format)
            metadata[row * (cols / 4) + group] = encode_sparse_meta(idx1, idx2);

            // Store non-zero elements
            int sparse_base = row * (cols / 2) + group * 2;
            sparse_matrix[sparse_base] = dense_matrix[base + idx1];
            sparse_matrix[sparse_base + 1] = dense_matrix[base + idx2];
        }
    }
}
```

### 22.7.2 CUTLASS Sparse Conversion Helper

CUTLASS provides reference implementations for sparse matrix conversion:

```cpp
#include "cutlass/util/reference/host/sparse_gemm.h"

// Convert dense matrix to sparse format using CUTLASS utilities
// The following uses the reference implementation
void convert_dense_to_sparse(
    cutlass::TensorRef<cutlass::half_t, cutlass::layout::RowMajor> dense_ref,
    cutlass::TensorRef<cutlass::half_t, cutlass::layout::RowMajor> sparse_ref,
    cutlass::TensorRef<uint32_t, cutlass::layout::RowMajor> meta_ref,
    int M, int K
) {
    // CUTLASS provides helper functions for this conversion
    // See examples/24_ampere_sparse_tensorop_gemm for a complete example
    cutlass::reference::host::SparseGemm<
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        float, cutlass::layout::RowMajor,
        float
    >::compress_to_sparse(dense_ref, sparse_ref, meta_ref, M, K);
}
```

### 22.7.3 GPU-Accelerated Sparse Conversion

For large matrices, GPU-accelerated conversion is recommended:

```cpp
#include "cutlass/util/device/transform_filter.h"

// GPU-accelerated sparse conversion
void gpu_sparse_convert(
    const cutlass::half_t* dense_A,
    cutlass::half_t* sparse_A,
    uint16_t* metadata_A,
    int M, int K,
    cudaStream_t stream
) {
    // Use CUTLASS's device-side transform utilities
    // for pruning and metadata generation
    cutlass::transform::device::CompressSparseMatrix<
        cutlass::half_t, cutlass::layout::RowMajor
    > compressor;

    compressor(dense_A, sparse_A, metadata_A, M, K, stream);
}
```

---

## 22.8 Performance Benefits of Sparsity

### 22.8.1 Throughput Gains

Sparse Tensor Core operations provide up to 2x throughput improvement compared to dense Tensor Core operations for the same data type:

| Operation | Dense Throughput (relative) | Sparse Throughput (relative) |
|---|---|---|
| FP16 Tensor Core | 1x | 2x |
| BF16 Tensor Core | 1x | 2x |
| INT8 Tensor Core | 1x | 2x |
| TF32 Tensor Core | 1x | 2x |

### 22.8.2 Memory Bandwidth Savings

Since only 50% of the sparse matrix elements are stored:
- **Matrix A memory traffic**: Reduced by 50% compared to dense.
- **Metadata overhead**: Typically small (2 bits per 4 elements = 6.25% overhead relative to dense).
- **Net memory savings for A**: ~43.75% (50% saved minus 6.25% metadata overhead).

### 22.8.3 Accuracy Impact

The 2:4 pruning process introduces approximation error. The impact depends on the weight distribution:

1. **Well-trained models**: Pruning typically causes < 0.1% accuracy loss after fine-tuning.
2. **Fine-tuning approach**: Prune weights, then fine-tune the model for a few epochs to recover accuracy.
3. **Magnitude-based pruning**: Zeroing the smallest-magnitude weights in each group of 4 minimizes accuracy impact.
4. **Global vs local pruning**: Per-group pruning (local) is required for the hardware pattern; however, global importance scoring can guide which weights to prioritize.

### 22.8.4 When to Use Sparse GEMM

Sparse GEMM is beneficial when:
- The model has been pruned to 2:4 sparsity (common for inference deployment).
- Matrix A (weights) is static across multiple GEMM calls (amortize the conversion cost).
- The K dimension is sufficiently large to saturate the Tensor Cores.
- Memory bandwidth is a bottleneck (sparse reduces A traffic by ~44%).

Sparse GEMM is NOT beneficial when:
- The matrix does not have 2:4 sparsity (the hardware cannot exploit unstructured sparsity).
- The conversion overhead outweighs the compute savings (small matrices, one-time use).
- Accuracy requirements cannot tolerate the pruning approximation.

---

## 22.9 Sparsity Pattern Requirements

### 22.9.1 2:4 Structured Sparsity Rules

The 2:4 sparsity pattern must satisfy these constraints:

1. **Group size**: Elements are grouped into contiguous blocks of 4 along the K dimension (for RowMajor A).
2. **Zero count**: Exactly 2 elements in each group of 4 must be zero.
3. **Position freedom**: Any 2 of the 4 positions can be non-zero; the metadata encodes which.
4. **Row independence**: The sparsity pattern can differ between rows (each row has its own pattern).

```
Valid 2:4 patterns for a group of 4:
  [nz, nz,  0,  0]  -- positions 0, 1 are non-zero
  [nz,  0, nz,  0]  -- positions 0, 2 are non-zero
  [nz,  0,  0, nz]  -- positions 0, 3 are non-zero
  [ 0, nz, nz,  0]  -- positions 1, 2 are non-zero
  [ 0, nz,  0, nz]  -- positions 1, 3 are non-zero
  [ 0,  0, nz, nz]  -- positions 2, 3 are non-zero
```

### 22.9.2 Dimension Constraints

Sparse GEMM has specific dimension constraints:

- **K must be a multiple of 4**: Each group requires 4 elements.
- **K must be a multiple of the instruction K dimension**: Typically 16 or 32 depending on the instruction shape.
- **Alignment**: The sparse matrix stride must satisfy alignment requirements (typically 128-bit / 16-byte boundaries).

### 22.9.3 Layout Considerations

The sparsity is applied along the K dimension of the sparse operand:

- **RowMajor A (M x K)**: Sparsity is along columns (within each row, groups of 4 in the K direction).
- **ColumnMajor A (K x M)**: Sparsity is along rows (within each column, groups of 4 in the K direction).

---

## 22.10 Complete Sparse GEMM Examples

### 22.10.1 Full SM80 Sparse FP16 GEMM Example

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/sparse_gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"

// Define the sparse GEMM type
using SparseGemm = cutlass::gemm::device::SparseGemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSparseTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<
        float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4
>;

int main() {
    int M = 1024;
    int N = 512;
    int K = 1024;

    // Allocate tensors
    // Dense A (for reference / pruning source)
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> dense_A({M, K});
    // Sparse A (compressed, M x K/2)
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> sparse_A({M, K / 2});
    // Metadata (M x K/4, packed 2-bit indices)
    cutlass::HostTensor<uint16_t, cutlass::layout::RowMajor> metadata_A({M, K / 16});
    // Dense B
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> tensor_B({K, N});
    // C and D
    cutlass::HostTensor<float, cutlass::layout::RowMajor> tensor_C({M, N});
    cutlass::HostTensor<float, cutlass::layout::RowMajor> tensor_D({M, N});

    // Initialize data
    cutlass::reference::host::BlockFillRandom(dense_A.host_data(), M * K, 42);
    cutlass::reference::host::BlockFillRandom(tensor_B.host_data(), K * N, 43);
    cutlass::reference::host::BlockFillRandom(tensor_C.host_data(), M * N, 44);

    // TODO: Prune dense_A to 2:4 sparsity and generate sparse_A + metadata_A
    // (Use CUTLASS conversion utilities or custom pruning code)

    // Copy to device
    sparse_A.sync_device();
    metadata_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();

    // Run sparse GEMM
    SparseGemm gemm_op;
    typename SparseGemm::Arguments args(
        {M, N, K},
        sparse_A.device_data(), K / 2,       // Sparse A stride
        metadata_A.device_data(), K / 16,     // Metadata stride
        tensor_B.device_data(), N,
        tensor_C.device_data(), N,
        tensor_D.device_data(), N,
        1.0f, 0.0f
    );

    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Sparse GEMM failed!" << std::endl;
        return -1;
    }

    tensor_D.sync_host();
    std::cout << "Sparse GEMM completed successfully." << std::endl;
    return 0;
}
```

### 22.10.2 Sparse GEMM with Mixed Precision

```cpp
// Sparse BF16 GEMM with FP32 accumulation
using SparseBF16Gemm = cutlass::gemm::device::SparseGemm<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,    // Sparse A: BF16
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,  // Dense B: BF16
    float, cutlass::layout::RowMajor,                   // C/D: FP32
    float,                                               // Accumulator: FP32
    cutlass::arch::OpClassSparseTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,                 // Sparse BF16 instruction
    cutlass::epilogue::thread::LinearCombination<
        float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4
>;
```

### 22.10.3 Sparse TF32 GEMM

```cpp
// Sparse TF32 GEMM on Ampere
using SparseTF32Gemm = cutlass::gemm::device::SparseGemm<
    cutlass::tfloat32_t, cutlass::layout::RowMajor,
    cutlass::tfloat32_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSparseTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,    // Sparse TF32 instruction
    cutlass::epilogue::thread::LinearCombination<
        float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4
>;
```

---

## 22.11 Summary

Sparse GEMM with 2:4 structured sparsity provides a powerful optimization for workloads where matrix sparsity can be exploited:

1. **2x compute throughput** over dense Tensor Core operations.
2. **~44% memory savings** for the sparse operand.
3. **Minimal accuracy impact** when combined with fine-tuning.
4. **Hardware support** on SM80+ (Ampere, Ada, Hopper, Blackwell).
5. **CUTLASS support** across both 2.x and 3.x APIs, with CollectiveBuilder integration for SM90+.

The key tradeoff is the requirement for 2:4 structured sparsity: not all matrices can be pruned to this pattern without significant accuracy loss. However, for well-trained neural network weights, the accuracy impact is typically minimal and can be recovered through a few epochs of fine-tuning.
