# CUTLASS Reference - Chapter 24: Distributed GEMM

This reference covers distributed GEMM operations in CUTLASS, which enable tensor parallelism across multiple GPUs connected via NVLink. Distributed GEMM combines communication (AllGather, ReduceScatter) with computation (GEMM) to achieve high efficiency in multi-GPU training and inference.

---

## 24.1 Tensor Parallel GEMM Concept

### 24.1.1 Motivation

Large language models (LLMs) and other large-scale neural networks often have weight matrices that are too large to fit on a single GPU. Tensor parallelism distributes the GEMM computation across multiple GPUs, with each GPU holding a shard (partition) of the weight matrices.

There are two primary patterns for tensor-parallel GEMM:

1. **Column Parallel (AllGather + GEMM)**: Gather the full activation tensor, then compute a partial GEMM with the local weight shard. Produces a partial result that is complete along the M dimension but partitioned along N.

2. **Row Parallel (GEMM + ReduceScatter)**: Compute a partial GEMM with the local activation shard and weight shard, then reduce-scatter the result so each GPU holds a different partition of the output.

### 24.1.2 Mathematical Formulation

For a GEMM `D = A * B` distributed across `P` GPUs:

**Column Parallel (AllGather + GEMM)**:
```
GPU p: D_p = A * B_p
Where B is partitioned along columns: B = [B_0 | B_1 | ... | B_{P-1}]
The result is D = [D_0 | D_1 | ... | D_{P-1}]
```

**Row Parallel (GEMM + ReduceScatter)**:
```
GPU p: D_partial_p = A_p * B_p
Where A is partitioned along rows: A = [A_0; A_1; ...; A_{P-1}]
And B is partitioned along columns: B = [B_0; B_1; ...; B_{P-1}]
Then: D = ReduceScatter(D_partial_0 + D_partial_1 + ... + D_partial_{P-1})
```

### 24.1.3 NVLink Requirements

Distributed GEMM relies on high-bandwidth inter-GPU communication via NVLink:
- **NVLink 3.0** (Hopper): 900 GB/s aggregate bidirectional bandwidth per GPU.
- **NVLink 4.0** (Blackwell): 1.8 TB/s aggregate bidirectional bandwidth per GPU.
- **Minimum configuration**: 2 GPUs with NVLink connections.
- **Recommended**: 4 or 8 GPUs within a single node (NVLink fully connected).

PCIe-connected GPUs cannot efficiently run distributed GEMM due to insufficient bandwidth (~64 GB/s for PCIe Gen5 x16).

---

## 24.2 AllGather + GEMM Pattern

### 24.2.1 Algorithm

The AllGather + GEMM pattern is used for column-parallel linear layers:

1. **AllGather**: Each GPU holds a local shard of the activation tensor A (partitioned along M). AllGather collects the full A tensor on every GPU.
2. **Local GEMM**: Each GPU computes `D_p = A * B_p` where `B_p` is the local shard of the weight matrix (partitioned along N).
3. **Output**: Each GPU holds `D_p`, a shard of the output along N. No further communication is needed.

```
GPU 0: A_0 --[AllGather]--> A_full --[GEMM with B_0]--> D_0
GPU 1: A_1 --[AllGather]--> A_full --[GEMM with B_1]--> D_1
...
GPU P: A_P --[AllGather]--> A_full --[GEMM with B_P]--> D_P
```

### 24.2.2 Communication Volume

The AllGather communication volume is:
- **Data transferred per GPU**: `M * K * sizeof(ElementA) * (P - 1) / P`
- **Total data across all GPUs**: `M * K * sizeof(ElementA) * (P - 1)`

For efficient overlap, the AllGather should be pipelined with the GEMM computation so that GEMM can start on the locally available data while the remote data is still being gathered.

---

## 24.3 GEMM + ReduceScatter Pattern

### 24.3.1 Algorithm

The GEMM + ReduceScatter pattern is used for row-parallel linear layers:

1. **Local GEMM**: Each GPU computes a partial result using its local shards of A and B.
2. **ReduceScatter**: The partial results are summed and scattered so each GPU receives a different partition of the final result.

```
GPU 0: A_0 * B_0 --> partial_0 --[ReduceScatter]--> D_0
GPU 1: A_1 * B_1 --> partial_1 --[ReduceScatter]--> D_1
...
GPU P: A_P * B_P --> partial_P --[ReduceScatter]--> D_P
```

### 24.3.2 Communication Volume

The ReduceScatter communication volume is:
- **Data transferred per GPU**: `M * N * sizeof(ElementD) * (P - 1) / P`
- **Total data across all GPUs**: `M * N * sizeof(ElementD) * (P - 1)`

---

## 24.4 Distributed GEMM API (Experimental)

### 24.4.1 Overview

CUTLASS provides experimental support for distributed GEMM starting with version 3.5. The API integrates communication operations directly into the GEMM kernel, enabling overlap between communication and computation.

The distributed GEMM API is in the `cutlass::gemm::distributed` namespace and requires the following additional headers:

```cpp
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
```

### 24.4.2 Distributed GEMM Arguments

The arguments for distributed GEMM extend the standard GEMM arguments with communication-related parameters:

```cpp
struct DistributedGemmArguments {
    // Standard GEMM arguments
    cutlass::gemm::GemmCoord problem_shape;
    typename Epilogue::Arguments output_op;

    // Distributed arguments
    int num_ranks;                          // Number of GPUs
    int rank;                               // This GPU's rank (0 to num_ranks-1)
    void* communication_workspace;          // Buffer for inter-GPU communication
    size_t communication_workspace_size;    // Size of the communication buffer
    cudaStream_t communication_stream;      // Stream for communication operations
};
```

---

## 24.5 Hopper Distributed GEMM (SM90)

### 24.5.1 SM90 Distributed Capabilities

Hopper (SM90) supports distributed GEMM with:
- **TMA-based data movement**: Efficient loading of both local and remote tensor shards.
- **Thread block clusters**: Enable cooperation across thread blocks within a GPC.
- **Warp specialization**: Overlap of communication and computation.
- **NVLink 3.0**: High-bandwidth inter-GPU communication at 900 GB/s.

### 24.5.2 AllGather1D Tiling Strategy

The `AllGather1D_TilingCD_RotatingA` schedule implements the AllGather + GEMM pattern:

```cpp
#include "cutlass/gemm/dispatch_policy.hpp"

// Hopper distributed GEMM with AllGather
using DispatchPolicy = cutlass::gemm::collective::AllGather1D_TilingCD_RotatingA<
    cutlass::gemm::GemmShape<128, 128, 64>,   // Tile shape
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    1,                                          // Number of pipeline stages
    cutlass::gemm::KernelTmaWarpSpecialized    // Base schedule
>;

// The schedule:
// 1. Tiles the output (C, D) across GPUs -- each GPU owns a different output tile range
// 2. Rotates the A operand across GPUs during AllGather
// 3. Each GPU computes its local output tiles as A shards arrive
```

### 24.5.3 ReduceScatter1D Tiling Strategy

The `ReduceScatter1D_TilingA_RotatingC` schedule implements the GEMM + ReduceScatter pattern:

```cpp
// Hopper distributed GEMM with ReduceScatter
using DispatchPolicy = cutlass::gemm::collective::ReduceScatter1D_TilingA_RotatingC<
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    1,
    cutlass::gemm::KernelTmaWarpSpecialized
>;

// The schedule:
// 1. Tiles A across GPUs -- each GPU owns a different row range of A
// 2. Computes partial GEMM with local A shard and local B shard
// 3. Rotates partial results across GPUs for ReduceScatter
// 4. Each GPU accumulates its final output partition
```

### 24.5.4 Complete SM90 Distributed GEMM Example

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

// Define types
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// Use CollectiveBuilder with distributed schedule
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 8,
    ElementB, LayoutB, 8,
    ElementAccumulator,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelTmaWarpSpecialized
>::CollectiveOp;

using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
    LayoutD, LayoutC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>;

using Kernel = cutlass::gemm::kernel::GemmUniversal<
    CollectiveOp, EpilogueOp
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

void run_distributed_gemm_sm90(
    int M, int N, int K,
    const cutlass::half_t* A, int64_t lda,
    const cutlass::half_t* B, int64_t ldb,
    const float* C, int64_t ldc,
    cutlass::half_t* D, int64_t ldd,
    float alpha, float beta,
    int rank, int num_ranks,
    void* comm_workspace, size_t comm_workspace_size,
    cudaStream_t stream = 0
) {
    Gemm gemm_op;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmCoord{M, N, K},
        {A, lda},
        {B, ldb},
        {C, ldc},
        {D, ldd},
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

## 24.6 Blackwell Distributed GEMM (SM100)

### 24.6.1 SM100 Distributed Enhancements

Blackwell (SM100+) extends distributed GEMM with:

- **NVLink 4.0**: 1.8 TB/s bidirectional bandwidth (2x over Hopper).
- **Hardware-assisted communication**: Direct memory access across GPU memory spaces via NVLink.
- **Larger cluster sizes**: Thread block clusters can span more thread blocks.
- **Improved overlap**: Better hardware support for overlapping communication and computation.

### 24.6.2 Blackwell Distributed Scheduling

The scheduling strategies on Blackwell follow the same patterns as Hopper but benefit from the increased bandwidth:

```cpp
// Blackwell distributed GEMM with AllGather
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassTensorOp,
    cutlass::float_e4m3_t, cutlass::layout::RowMajor, 16,
    cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
    float,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelTmaWarpSpecialized
>::CollectiveOp;
```

### 24.6.3 Grace Hopper Distributed GEMM

Grace Hopper (GH200) systems have a unified memory architecture that eliminates explicit data movement between CPU and GPU. Distributed GEMM on GH200 benefits from:

- **Unified memory**: No need for host-to-device copies; data can be accessed directly.
- **NVLink-C2C**: 900 GB/s bandwidth between Grace CPU and Hopper GPU.
- **NVLink interconnect**: Full NVLink connectivity between GPUs in the node.

---

## 24.7 Scheduling Strategies in Detail

### 24.7.1 AllGather1D_TilingCD_RotatingA

This strategy implements a 1D AllGather + GEMM pattern:

**Tiling strategy**:
- The C and D (output) tensors are tiled across the N dimension. Each GPU owns a contiguous range of N columns.
- The A operand is gathered from all GPUs. Each GPU starts with its local A shard and receives other GPUs' shards.

**Rotation pattern**:
```
Step 0: GPU 0 has A_0, computes A_0 * B_0  -> partial D_0
Step 1: A rotates: GPU 0 receives A_1, computes A_1 * B_0  -> accumulate into D_0
Step 2: A rotates: GPU 0 receives A_2, computes A_2 * B_0  -> accumulate into D_0
...
Step P-1: A rotates: GPU 0 receives A_{P-1}, computes A_{P-1} * B_0  -> finalize D_0
```

**Advantages**:
- Computation starts immediately on locally available data.
- Communication is overlapped with computation (pipeline).
- No synchronization barrier between steps.

### 24.7.2 AllGather1D_TilingCD_RotatingB

A variant where B rotates instead of A:

```cpp
using DispatchPolicy = cutlass::gemm::collective::AllGather1D_TilingCD_RotatingB<
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    1,
    cutlass::gemm::KernelTmaWarpSpecialized
>;
```

This is useful when B is the activation tensor and A is the weight (which stays local).

### 24.7.3 ReduceScatter1D_TilingA_RotatingC

This strategy implements a 1D GEMM + ReduceScatter pattern:

**Tiling strategy**:
- The A tensor is tiled along the K dimension. Each GPU owns a contiguous range of K rows of A.
- The C/D (output) partial results are accumulated via ReduceScatter.

**Rotation pattern**:
```
Step 0: GPU 0 computes A_0_local * B_0_local -> partial C_0
Step 1: Partial C rotates to next GPU for accumulation
Step 2: Partial C rotates again
...
Step P-1: Each GPU has accumulated all partial C values for its output partition
```

### 24.7.4 Scheduling Strategy Selection

| Strategy | Communication | Computation | Best For |
|---|---|---|---|
| `AllGather1D_TilingCD_RotatingA` | AllGather A | Full GEMM per GPU | Column-parallel layers |
| `AllGather1D_TilingCD_RotatingB` | AllGather B | Full GEMM per GPU | Row-parallel with B activation |
| `ReduceScatter1D_TilingA_RotatingC` | ReduceScatter C | Partial GEMM per GPU | Row-parallel layers |

---

## 24.8 Multi-GPU Coordination

### 24.8.1 Initialization

Distributed GEMM requires proper multi-GPU initialization:

```cpp
#include <cuda_runtime.h>
#include <nccl.h>

// Initialize NCCL for communication
ncclComm_t nccl_comm;
ncclCommInitRank(&nccl_comm, num_ranks, nccl_id, rank);

// Set the device
cudaSetDevice(rank);

// Allocate communication workspace
void* comm_workspace;
size_t comm_workspace_size = compute_comm_workspace_size(M, N, K, num_ranks);
cudaMalloc(&comm_workspace, comm_workspace_size);
```

### 24.8.2 Data Distribution

Each GPU needs its local shard of the input tensors:

```cpp
// Distribute B matrix across GPUs (column partition for column-parallel)
// B has shape (K, N), partition along N
int local_N = N / num_ranks;
int offset_N = rank * local_N;

// GPU `rank` holds B[:, offset_N:offset_N+local_N]
const cutlass::half_t* local_B = B + offset_N;  // ColumnMajor: offset by columns
int64_t local_ldb = ldb;  // Same leading dimension

// A is fully replicated on all GPUs (after AllGather)
// Each GPU starts with its local shard of A
int local_M = M / num_ranks;
const cutlass::half_t* local_A = A + rank * local_M * lda;
```

### 24.8.3 Launching Distributed GEMM

```cpp
// Launch distributed GEMM on each GPU
void launch_distributed_gemm(
    int rank, int num_ranks,
    int M, int N, int K,
    const cutlass::half_t* local_A, int64_t lda,
    const cutlass::half_t* local_B, int64_t ldb,
    const float* C, int64_t ldc,
    cutlass::half_t* D, int64_t ldd,
    float alpha, float beta,
    ncclComm_t nccl_comm,
    cudaStream_t stream
) {
    // Step 1: Allocate buffers for gathered A (AllGather)
    cutlass::half_t* gathered_A;
    cudaMalloc(&gathered_A, M * K * sizeof(cutlass::half_t));

    // Step 2: Perform AllGather on A
    ncclAllGather(
        local_A, gathered_A, M / num_ranks * K,
        ncclFloat16, nccl_comm, stream
    );

    // Step 3: Compute local GEMM with gathered A and local B
    int local_N = N / num_ranks;
    int offset_N = rank * local_N;

    using Gemm = /* ... CUTLASS GEMM type ... */;
    Gemm gemm_op;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmCoord{M, local_N, K},
        {gathered_A, lda},
        {local_B + offset_N, ldb},  // Local B shard
        {C, ldc},
        {D, ldd},
        {alpha, beta}
    };

    size_t workspace_size = gemm_op.get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    auto status = gemm_op.initialize(args, workspace.get(), stream);
    if (status == cutlass::Status::kSuccess) {
        status = gemm_op(stream);
    }

    cudaFree(gathered_A);
}
```

### 24.8.4 Fused Communication-Computation (Advanced)

For maximum performance, CUTLASS can fuse the communication and computation into a single kernel, avoiding the overhead of launching separate NCCL and GEMM kernels:

```cpp
// Fused AllGather + GEMM kernel (experimental)
// This overlaps data movement with computation
// Available in CUTLASS 3.5+ for Hopper

void run_fused_distributed_gemm(
    int rank, int num_ranks,
    int M, int N, int K,
    const cutlass::half_t* local_A, int64_t lda,
    const cutlass::half_t* local_B, int64_t ldb,
    const float* C, int64_t ldc,
    cutlass::half_t* D, int64_t ldd,
    float alpha, float beta,
    void* comm_workspace, size_t comm_workspace_size,
    cudaStream_t stream
) {
    // The fused kernel handles:
    // 1. AllGather of A across NVLink
    // 2. GEMM computation on each A shard as it arrives
    // 3. Accumulation of partial results

    // This requires CUTLASS's distributed dispatch policy
    // and proper communication workspace allocation
}
```

---

## 24.9 Performance Considerations

### 24.9.1 Communication Overhead

The communication overhead depends on the data type and matrix dimensions:

| Operation | Data per GPU | Time at 900 GB/s (Hopper) |
|---|---|---|
| AllGather A (FP16, M=4K, K=4K) | 4 * 4K * 4K * 2 / 8 = 16 MB | ~18 us |
| ReduceScatter D (FP16, M=4K, N=4K) | 4 * 4K * 4K * 2 / 8 = 16 MB | ~18 us |
| AllGather A (FP8, M=4K, K=4K) | 4 * 4K * 4K * 1 / 8 = 8 MB | ~9 us |

### 24.9.2 Overlap Efficiency

The key to performance in distributed GEMM is overlapping communication with computation. The ideal scenario is:

```
Timeline: |--compute A0*B--|--compute A1*B--|--compute A2*B--|--compute A3*B--|
          |--recv A1------|--recv A2------|--recv A3------|
```

If the compute time per A shard exceeds the communication time, the communication is fully hidden. This requires:
- Sufficient K dimension for compute-heavy tiles.
- Adequate pipeline depth (multiple stages of compute-communication overlap).
- Large enough problem sizes to amortize kernel launch overhead.

### 24.9.3 Scaling Efficiency

Theoretical scaling efficiency for distributed GEMM:

```
Efficiency = 1 / (1 + (communication_time / compute_time))
```

For a square GEMM (M = N = K) with FP16:
- Compute per GPU: M * (N/P) * K * 2 / (peak_tflops * 1e12)
- Communication: M * K * 2 / (nvlink_bandwidth * 1e9)

With 8 GPUs on Hopper (990 TFLOPS FP16 Tensor Core, 900 GB/s NVLink):
- M=N=K=8192: Communication ~1.2 ms, Compute per GPU ~0.3 ms -> Efficiency ~20%
- M=N=K=32768: Communication ~4.8 ms, Compute per GPU ~4.8 ms -> Efficiency ~50%
- M=N=K=131072: Communication ~19 ms, Compute per GPU ~77 ms -> Efficiency ~80%

**Key takeaway**: Distributed GEMM scales best with large problem sizes where computation dominates communication.

---

## 24.10 Complete Distributed GEMM Example

### 24.10.1 Multi-GPU Column Parallel GEMM

```cpp
#include <cuda_runtime.h>
#include <nccl.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

// GEMM type definition
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;

using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
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

using Kernel = cutlass::gemm::kernel::GemmUniversal<CollectiveOp, EpilogueOp>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

int main(int argc, char** argv) {
    int num_ranks = 8;
    int rank = 0; // Set from MPI or NCCL initialization

    int M = 4096;
    int N = 4096;
    int K = 4096;

    // Initialize NCCL
    ncclComm_t nccl_comm;
    // ... NCCL initialization code ...

    cudaSetDevice(rank);

    // Allocate local tensors
    int local_N = N / num_ranks;
    cutlass::half_t* local_A;
    cutlass::half_t* local_B_shard;
    float* tensor_C;
    cutlass::half_t* tensor_D;

    // A is partitioned along M for AllGather
    int local_M = M / num_ranks;
    cudaMalloc(&local_A, local_M * K * sizeof(cutlass::half_t));

    // B is partitioned along N (each GPU owns a column shard)
    cudaMalloc(&local_B_shard, K * local_N * sizeof(cutlass::half_t));

    // C and D: same size as local output
    cudaMalloc(&tensor_C, M * local_N * sizeof(float));
    cudaMalloc(&tensor_D, M * local_N * sizeof(cutlass::half_t));

    // Allocate gathered A buffer
    cutlass::half_t* gathered_A;
    cudaMalloc(&gathered_A, M * K * sizeof(cutlass::half_t));

    // AllGather A
    ncclAllGather(
        local_A, gathered_A, local_M * K,
        ncclFloat16, nccl_comm, 0
    );

    // Local GEMM: gathered_A (M x K) * local_B_shard (K x local_N) -> D (M x local_N)
    Gemm gemm_op;
    typename Gemm::Arguments args{
        cutlass::gemm::GemmCoord{M, local_N, K},
        {gathered_A, K},
        {local_B_shard, local_N},
        {tensor_C, local_N},
        {tensor_D, local_N},
        {1.0f, 0.0f}
    };

    size_t workspace_size = gemm_op.get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    auto status = gemm_op.initialize(args, workspace.get());
    if (status == cutlass::Status::kSuccess) {
        status = gemm_op();
    }

    // Cleanup
    cudaFree(gathered_A);
    cudaFree(local_A);
    cudaFree(local_B_shard);
    cudaFree(tensor_C);
    cudaFree(tensor_D);

    return 0;
}
```

---

## 24.11 Summary

Distributed GEMM enables scaling beyond single-GPU memory and compute limits:

1. **AllGather + GEMM**: Gather activations, compute local GEMM. Used for column-parallel layers.
2. **GEMM + ReduceScatter**: Compute partial GEMM, reduce-scatter results. Used for row-parallel layers.
3. **NVLink dependency**: Distributed GEMM requires NVLink interconnect; PCIe is insufficient.
4. **Overlap is critical**: Communication must be overlapped with computation for efficiency.
5. **Problem size matters**: Large matrices (M, N, K >= 8192) are needed for efficient scaling.
6. **CUTLASS integration**: Experimental support in CUTLASS 3.5+ with fused communication-computation kernels on Hopper and Blackwell.
7. **Future direction**: Tighter hardware integration on Blackwell (NVLink 4.0, UMMA) will reduce communication overhead further.
