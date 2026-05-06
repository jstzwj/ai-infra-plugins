# CUTLASS - Chapter 31: Hopper (SM90) Features

This reference covers the Hopper GPU architecture (SM90) features in CUTLASS in detail, including the Tensor Memory Accelerator (TMA), GMMA/WGMMA instructions, Thread Block Clusters, Warp Specialization, mixed dtype GEMM, weight prefetching, FMHA, distributed GEMM, and State Space Decomposition.

---

## 31.1 Hopper Architecture Overview

The NVIDIA Hopper architecture (SM90), introduced with the H100 GPU, represents a significant leap in GPU compute capabilities. Key hardware innovations include:

| Feature | Description | Impact |
|---------|-------------|--------|
| Tensor Memory Accelerator (TMA) | Hardware unit for asynchronous tensor transfers | Offloads address generation and data movement from threads |
| WGMMA | Warp Group Matrix Multiply-Accumulate (4 warps) | 4x throughput per instruction vs. Ampere MMA |
| Thread Block Clusters | Groups of CTAs with shared distributed memory | Enables cooperative computation across CTAs |
| Warp Specialization | Hardware support for producer-consumer warp roles | Overlaps data movement and computation |
| Dynamic Shared Memory | 227 KB max shared memory per SM | Larger working sets for deeper pipelines |

---

## 31.2 Tensor Memory Accelerator (TMA)

### 31.2.1 TMA Overview

The Tensor Memory Accelerator is a dedicated hardware unit on Hopper that performs asynchronous tensor transfers between global memory and shared memory. Unlike Ampere's `cp.async` (which is per-thread), TMA is a bulk transfer engine where a single thread can initiate a transfer for an entire tile.

Key advantages of TMA over `cp.async`:
- **Single-thread initiation**: One thread programs the TMA descriptor; all threads benefit.
- **Hardware address generation**: The TMA unit computes all element addresses, freeing threads for computation.
- **Multi-dimensional transfers**: Native support for 1D-5D tensor transfers with swizzling.
- **Multi-cast**: A single TMA load can deliver data to shared memory in multiple CTAs simultaneously.
- **Reduced register pressure**: No need for threads to hold pointers or strides.

### 31.2.2 TMA Load Operations

TMA load transfers data from global memory to shared memory:

```cpp
#include "cute/arch/copy_sm90_tma.hpp"

// TMA load: global memory -> shared memory
// The TMA descriptor encodes the tensor's layout, strides, and memory location

// Step 1: Create a TMA tensor (usually done by the kernel host side or first thread)
using TMALoad = cute::SM90_TMA_LOAD;

// Step 2: Execute TMA load
// tma_load(tma_desc, smem_ptr, coord)
cute::copy(TMALoad(), tma_tensor, smem_tensor);

// The copy is asynchronous - use a barrier to wait for completion
cute::wait_barrier(barrier);
```

In CUTLASS 3.x, TMA loads are typically configured via the CollectiveBuilder:

```cpp
#include "cutlass/gemm/collective/collective_builder.hpp"

// The CollectiveBuilder automatically configures TMA for SM90
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
// ^ For SM90, this automatically selects TMA-based data movement
```

### 31.2.3 TMA Store Operations

TMA store transfers data from shared memory to global memory:

```cpp
#include "cute/arch/copy_sm90_tma.hpp"

// TMA store: shared memory -> global memory
using TMAStore = cute::SM90_TMA_STORE;

// Execute TMA store
cute::copy(TMAStore(), smem_tensor, gmem_tensor);

// TMA stores are also asynchronous - synchronize with a barrier
cute::store_wait(barrier);
```

TMA stores are used in the SM90 epilogue for writing GEMM output:

```cpp
// SM90 epilogue using TMA store
using EpilogueOp = cutlass::epilogue::collective::Epilogue<
    cutlass::gemm::EpilogueSm90TmaWarpSpecialized,
    ...>;
```

### 31.2.4 TMA Descriptors and Tensors

TMA descriptors encode the tensor's memory layout for the TMA hardware:

```cpp
#include "cute/arch/copy_sm90_tma.hpp"

// Create a TMA descriptor from a CuTe tensor
auto tma_desc = cute::make_tma_copy(
    cute::SM90_TMA_LOAD{},
    gmem_tensor,              // Source tensor in global memory
    smem_layout,              // Layout in shared memory (SmemLayout)
    tile_shape,               // Size of each TMA transfer tile
    cluster_size              // Number of CTAs in the cluster (for multicast)
);

// The TMA descriptor contains:
// - Base address in global memory
// - Tensor dimensions and strides
// - Swizzling pattern for shared memory
// - Multicast mask (if applicable)
```

TMA descriptors support various tensor layouts:

```cpp
// TMA descriptor for a 2D row-major tensor
auto tma_2d = cute::make_tma_copy(
    cute::SM90_TMA_LOAD{},
    make_tensor(ptr, make_shape(M, K), make_stride(K, 1)),  // Row-major
    SmemLayoutA{},
    make_shape(TM, TK)  // Tile size
);

// TMA descriptor for a 4D NHWC tensor (for convolution)
auto tma_4d = cute::make_tma_copy(
    cute::SM90_TMA_LOAD{},
    make_tensor(ptr, make_shape(N, H, W, C), make_stride(H*W*C, W*C, C, 1)),
    SmemLayoutAct{},
    make_shape(TN, TH, TW, TC)
);
```

### 31.2.5 Multi-Cast TMA for Clusters

TMA multi-cast delivers the same data to shared memory across multiple CTAs in a cluster:

```cpp
// Multi-cast TMA: one TMA load delivers data to all CTAs in the cluster
// This is used when multiple CTAs need the same input tile (e.g., in Split-K)

auto tma_multicast = cute::make_tma_copy(
    cute::SM90_TMA_LOAD_MULTICAST{},
    gmem_tensor,
    smem_layout,
    tile_shape,
    cluster_shape_v  // e.g., make_shape(2, 1, 1) for 2-CTA cluster
);

// Each CTA receives the data in its own shared memory
// Only one CTA needs to initiate the TMA load
if (cute::elect_one_sync()) {
    cute::copy(tma_multicast, tAgA(_,_,_,cta_rank), tAsA);
}

// All CTAs wait for the transfer to complete
cute::cp_async_barrier_arrive(cluster_barrier);
```

---

## 31.3 GMMA (Global Matrix Multiply-Accumulate)

### 31.3.1 WGMMA Instructions

WGMMA (Warp Group Matrix Multiply-Accumulate) is the primary Tensor Core instruction on Hopper. A warp group consists of 4 warps (128 threads) cooperating on a single matrix multiply operation.

Key characteristics of WGMMA:
- **Warp group scope**: 128 threads (4 warps) participate in each instruction.
- **Async execution**: WGMMA is asynchronous; a barrier is needed to read results.
- **Register-to-register**: Inputs from registers, output to registers.
- **Large tile sizes**: Supports larger M and N dimensions per instruction than Ampere MMA.

WGMMA instruction shapes for various data types:

| Input Types | Accumulator | Instruction Shape (MxNxK) |
|-------------|-------------|---------------------------|
| FP16 x FP16 | FP32 | 64x128x32, 64x64x32, 64x256x16 |
| BF16 x BF16 | FP32 | 64x128x32, 64x64x32, 64x256x16 |
| TF32 x TF32 | FP32 | 64x128x16, 64x64x16 |
| FP8 E4M3 x FP8 E4M3 | FP32 | 64x128x64, 64x256x32 |
| FP8 E5M2 x FP8 E4M3 | FP32 | 64x128x64, 64x256x32 |
| INT8 x INT8 | INT32 | 64x128x64, 64x256x32 |

### 31.3.2 Async Matrix Multiply

WGMMA instructions are asynchronous. The warp group issues the instruction and continues execution. A named barrier is used to wait for completion:

```cpp
#include "cute/arch/mma_sm90.hpp"

// WGMMA instruction
// D = alpha * A * B + beta * D (in registers)
auto mma_result = cute::SM90_U32x4x4_XOR_WGMMA<64, 128, 32>();

// In CuTe, WGMMA is accessed through MMA atoms:
using XpuOp = cute::SM90_64x128x32_F16F16F16_TN;

// The async nature means you can issue multiple WGMMA instructions
// and wait for all of them together:
// Issue WGMMA #1
cute::gemm(mma_atom, accumulator, A_tile_1, B_tile_1);
// Issue WGMMA #2 (overlaps with #1)
cute::gemm(mma_atom, accumulator, A_tile_2, B_tile_2);
// Wait for all outstanding WGMMA to complete
cutlass::arch::NamedBarrier::sync(128); // 128 threads = warp group
```

### 31.3.3 Register-to-Register Variants

WGMMA also supports register-to-register operation where both inputs come from registers (rather than one from shared memory):

```cpp
// Register-to-register WGMMA variant
// Both A and B operands are in registers
// This enables pipelining of shared-memory-to-register loads with MMA

using RegToRegMMA = cute::SM90_64x128x32_G2R_F16F16F16_TN;
```

---

## 31.4 Thread Block Clusters

### 31.4.1 Multi-CTA Coordination

Thread Block Clusters allow multiple CTAs (thread blocks) to coordinate directly through hardware, without going through global memory:

```cpp
#include "cutlass/cluster_launch.hpp"

// Define cluster shape
using ClusterShape = cutlass::gemm::GemmShape<2, 1, 1>; // 2 CTAs per cluster

// Cluster launch configuration
dim3 grid_dims(
    problem_size.m() / TileShape::kM / ClusterShape::kM,
    problem_size.n() / TileShape::kN / ClusterShape::kN,
    1
);
dim3 cluster_dims(ClusterShape::kM, ClusterShape::kN, ClusterShape::kCluster);

// Launch kernel with cluster support
cutlass::ClusterLauncher::launch(
    grid_dims, cluster_dims, block_dims,
    kernel, args, stream
);
```

### 31.4.2 Distributed Shared Memory

CTAs within a cluster can access each other's shared memory through Distributed Shared Memory (DSMEM):

```cpp
// Access another CTA's shared memory within the cluster
#include "cute/arch/copy_sm90_tma.hpp"

// Each CTA has its own shared memory
extern __shared__ char shared_memory[];
float *my_smem = reinterpret_cast<float*>(shared_memory);

// Read from a neighboring CTA's shared memory
int neighbor_rank = (cta_rank + 1) % cluster_size;
float *neighbor_smem = my_smem + neighbor_rank * smem_per_cta;

// Direct DSMEM access (hardware-supported)
// This enables cooperative computation where CTAs share partial results
float value = neighbor_smem[offset]; // Hardware routes this to the correct SM
```

DSMEM use cases in CUTLASS:
- **Multi-cast input sharing**: One CTA loads data, shares with cluster via TMA multicast.
- **Cross-CTA reduction**: Combine partial results from neighboring CTAs.
- **Peer-to-peer data exchange**: Pass data between CTAs in a producer-consumer pattern.

---

## 31.5 Warp Specialization

### 31.5.1 Producer and Consumer Warp Groups

Warp specialization divides the threads in a CTA into distinct roles:

- **Producer warp group** (1 warp group = 4 warps = 128 threads): Responsible for loading data from global memory to shared memory using TMA.
- **Consumer warp group** (1 warp group = 4 warps = 128 threads): Responsible for computing the GEMM using WGMMA.

This division enables overlapping data movement with computation:

```cpp
// In a warp-specialized kernel:
if (cutlass::canonical_warp_group_idx() == 0) {
    // Producer warp group: load data via TMA
    for (int k = 0; k < K; k += TK) {
        // Initiate TMA loads for A and B tiles
        cute::copy(tma_load_a, gA_tile, sA_tile);
        cute::copy(tma_load_b, gB_tile, sB_tile);

        // Signal consumer that data is ready
        cutlass::arch::NamedBarrier::arrive(256); // 256 = 2 warp groups
    }
} else {
    // Consumer warp group: compute GEMM via WGMMA
    for (int k = 0; k < K; k += TK) {
        // Wait for producer to load data
        cutlass::arch::NamedBarrier::wait(256);

        // Load from shared memory to registers
        cute::copy(sA_tile, rA_tile);
        cute::copy(sB_tile, rB_tile);

        // WGMMA
        cute::gemm(mma_atom, accumulator, rA_tile, rB_tile);
    }
}
```

### 31.5.2 Named Barriers

Hopper introduces hardware Named Barriers that support synchronization between specific sets of threads:

```cpp
#include "cutlass/arch/barrier.h"

// Named barriers are identified by name (0-15) and expected arrival count
// Barrier name 0: Producer signals data availability
// Barrier name 1: Consumer signals buffer is consumed

// Producer: arrive at barrier (signal that data is ready)
cutlass::arch::NamedBarrier::arrive(
    256,    // Expected arrival count (2 warp groups = 256 threads)
    0       // Barrier name (0 = data ready)
);

// Consumer: wait at barrier (wait for data to be ready)
cutlass::arch::NamedBarrier::wait(
    256,    // Expected arrival count
    0       // Barrier name (0 = data ready)
);

// Barrier name 1: signal buffer consumption complete
// Consumer arrives
cutlass::arch::NamedBarrier::arrive(256, 1);
// Producer waits
cutlass::arch::NamedBarrier::wait(256, 1);
```

### 31.5.3 Producer-Consumer Synchronization Pattern

The complete synchronization pattern for a warp-specialized GEMM:

```
Timeline:
  Producer: [Load A0,B0] --wait1-- [Load A1,B1] --wait1-- [Load A2,B2] ...
  Consumer:     --wait0-- [MMA(A0,B0)] --wait0-- [MMA(A1,B1)] --wait0-- ...

Barriers:
  wait0: Consumer waits for Producer to load data
  wait1: Producer waits for Consumer to consume data (buffer free)
```

This double-buffering scheme ensures that the producer can load the next tile while the consumer computes on the current tile, achieving full overlap.

---

## 31.6 Warp-Specialized Schedules

CUTLASS provides three main warp-specialized kernel schedules for SM90:

### 31.6.1 KernelTmaWarpSpecialized

The basic warp-specialized schedule with single buffering:

```cpp
using KernelSchedule = cutlass::gemm::collective::KernelTmaWarpSpecialized;

// Uses:
// - 1 producer warp group for TMA loads
// - 1 consumer warp group for WGMMA
// - Single buffer for A and B in shared memory
// - Simple barrier synchronization

using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 16,
    ElementB, LayoutB, 16,
    ElementAccumulator,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    KernelSchedule
>::CollectiveOp;
```

### 31.6.2 KernelTmaWarpSpecializedPingpong

Double-buffered warp specialization for maximum overlap:

```cpp
using KernelSchedule = cutlass::gemm::collective::KernelTmaWarpSpecializedPingpong;

// Uses:
// - 1 producer warp group for TMA loads
// - 1 consumer warp group for WGMMA
// - Double buffer (ping-pong) for A and B in shared memory
// - Producer loads into buffer 0 while consumer computes on buffer 1
// - Eliminates pipeline bubbles for steady-state operation

// Best for: Large problems where compute and memory can fully overlap
```

### 31.6.3 KernelTmaWarpSpecializedCooperative

Multiple warp groups cooperate on a single output tile:

```cpp
using KernelSchedule = cutlass::gemm::collective::KernelTmaWarpSpecializedCooperative;

// Uses:
// - 1 producer warp group for TMA loads
// - 2 consumer warp groups for WGMMA (splitting the M dimension)
// - Cooperative accumulation across warp groups
// - Larger effective tile size (e.g., 256x128 instead of 128x128)

// Best for: Large M/N where a single warp group cannot saturate Tensor Cores
// Requires more shared memory for the larger tile
```

Schedule comparison:

| Schedule | Warp Groups | Buffering | Best For |
|----------|-------------|-----------|----------|
| WarpSpecialized | 1P + 1C | Single | General purpose |
| Pingpong | 1P + 1C | Double | Maximum throughput |
| Cooperative | 1P + 2C | Single | Large tiles, high occupancy |

---

## 31.7 Mixed Dtype GEMM

### 31.7.1 Mixed Data Type Support on SM90

Hopper's WGMMA instruction supports mixed data types where A and B can have different element types, and the accumulator can be a third type:

```cpp
#include "cutlass/gemm/collective/collective_builder.hpp"

// FP8 inputs with FP32 accumulation and FP16 output
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::float_e4m3_t, cutlass::layout::RowMajor, 16,    // A: FP8 E4M3
    cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16, // B: FP8 E4M3
    float,                                                      // Accumulator: FP32
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelTmaWarpSpecializedPingpong
>::CollectiveOp;
```

### 31.7.2 Supported Mixed Type Combinations

| A Type | B Type | Accumulator | WGMMA Shape |
|--------|--------|-------------|-------------|
| E4M3 | E4M3 | FP32 | 64x128x64, 64x256x32 |
| E5M2 | E4M3 | FP32 | 64x128x64 |
| FP16 | FP16 | FP32 | 64x128x32, 64x256x16 |
| BF16 | BF16 | FP32 | 64x128x32, 64x256x16 |
| TF32 | TF32 | FP32 | 64x128x16 |
| INT8 | INT8 | INT32 | 64x128x64 |
| E4M3 | FP16 | FP32 | Mixed format supported |

### 31.7.3 Type Conversion in the Pipeline

Mixed dtype GEMM involves type conversion at specific points:

```
Pipeline:
  1. Load A (E4M3) from global -> shared memory (TMA, keep as E4M3)
  2. Load B (E4M3) from global -> shared memory (TMA, keep as E4M3)
  3. WGMMA: E4M3 x E4M3 -> FP32 (hardware converts during MMA)
  4. Epilogue: FP32 -> FP16 (convert output type)
  5. Store FP16 output to global memory (TMA store)
```

---

## 31.8 Weight Prefetching

### 31.8.1 Motivation

In transformer inference (batched GEMM with shared weights), the same weight matrix is used across many input tokens. Weight prefetching pre-loads weights into shared memory before they are needed, hiding the memory latency:

```cpp
// Weight prefetch pattern:
// 1. Start loading weights for the NEXT tile while computing the CURRENT tile
// 2. This creates a pipeline where data movement is always ahead of computation

// CUTLASS 3.x SM90 weight prefetch example
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 16,
    ElementB, LayoutB, 16,
    ElementAccumulator,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelTmaWarpSpecialized
>::CollectiveOp;
```

### 31.8.2 Prefetch with Multi-Cast TMA

For grouped problems where weights are shared across CTAs:

```cpp
// Multi-cast TMA weight prefetch
// Multiple CTAs in a cluster receive the same weight tile via single TMA load
auto tma_multicast = cute::make_tma_copy(
    cute::SM90_TMA_LOAD_MULTICAST{},
    weight_tensor,
    weight_smem_layout,
    weight_tile_shape,
    cluster_shape  // e.g., Shape<4, 1, 1> = 4 CTAs
);

// Only the elected CTA initiates the TMA load
if (cute::elect_one_sync()) {
    cute::copy(tma_multicast, gW_tile, sW_tile);
}

// All CTAs wait for the load to complete
cute::cp_async_wait<0>();
```

---

## 31.9 FMHA (Fused Multi-Head Attention)

### 31.9.1 Attention Kernel Structure

CUTLASS provides a fused multi-head attention (FMHA) implementation for Hopper that combines the Q*K^T, softmax, and V*attention operations into a single kernel:

```
Attention computation:
  S = Q @ K^T                    (GEMM 1: [B, H, M, D] x [B, H, D, N] -> [B, H, M, N])
  P = softmax(S)                 (Online softmax: row-wise max + exp + sum + normalize)
  O = P @ V                      (GEMM 2: [B, H, M, N] x [B, H, N, D] -> [B, H, M, D])
```

### 31.9.2 FMHA Kernel Implementation

```cpp
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.h"

// FMHA on SM90 uses:
// - TMA for loading Q, K, V tiles
// - WGMMA for Q*K^T and P*V
// - Online softmax (flash attention algorithm)
// - Register-efficient accumulation

// The flash attention algorithm processes K/V in tiles along the N dimension:
// For each tile of K and V:
//   1. Compute S_tile = Q @ K_tile^T  (WGMMA)
//   2. Update running max and sum for softmax
//   3. Rescale previous O accumulation
//   4. Update O += softmax(S_tile) @ V_tile  (WGMMA)

// Online softmax rescaling:
//   m_new = max(m_old, max(S_tile))
//   l_new = l_old * exp(m_old - m_new) + sum(exp(S_tile - m_new))
//   O_new = O_old * (l_old * exp(m_old - m_new)) / l_new + ...
```

### 31.9.3 Causal Masking

FMHA supports causal masking where future positions are masked out:

```cpp
// Causal mask: only attend to positions j <= i
// This is handled by adjusting the tile boundaries for each row
// No explicit mask tensor is needed - the tile loop bounds encode the mask

// For tile (i_tile, j_tile):
//   if j_tile * TILE_N > i_tile * TILE_M + TILE_M:
//     Skip this tile (all elements are masked)
//   else:
//     Apply mask within the tile for boundary cases
```

### 31.9.4 Variable-Length Sequences

```cpp
// FMHA supports variable-length sequences (e.g., in LLM batched inference)
// Sequence lengths are passed as a cu_seqlens array:
//   cu_seqlens = [0, s1, s1+s2, s1+s2+s3, ...]
// Each attention head processes only up to its sequence length
```

---

## 31.10 Distributed GEMM

### 31.10.1 Tensor Parallel GEMM

CUTLASS supports distributed GEMM across multiple GPUs using Tensor Parallelism:

```
Column Parallel: Y = X @ A
  Split A column-wise across N GPUs:
    A = [A_0 | A_1 | ... | A_{N-1}]
    Y_i = X @ A_i  (each GPU computes a shard of Y)
    All-Gather to get full Y (if needed)

Row Parallel: Y = A @ X
  Split A row-wise across N GPUs:
    A = [A_0; A_1; ... ; A_{N-1}]
    Y_i = A_i @ X  (each GPU computes a partial Y)
    All-Reduce to sum partial results into full Y
```

### 31.10.2 Communication-Computation Overlap

```cpp
// Distributed GEMM overlaps NVLink communication with computation
// Pipeline:
//   Step 1: Start async all-reduce for partial result
//   Step 2: While all-reduce is in flight, compute next tile
//   Step 3: Wait for all-reduce completion, use result

// In CUTLASS, this is handled by the distributed GEMM schedule:
using DistributedSchedule = cutlass::gemm::collective::KernelTmaWarpSpecializedPingpong;

// The distributed GEMM adds NVLink communication stages:
//   1. TMA Load A, B tiles
//   2. WGMMA compute
//   3. NVLink All-Reduce (async)
//   4. Local reduction of all-reduce result
```

### 31.10.3 NVLink Integration

```cpp
// NVLink communication primitives used in distributed GEMM:
#include "cutlass/cluster_launch.hpp"

// All-reduce across NVLink-connected GPUs
// Uses NVIDIA's NCCL or custom communication primitives
// The key optimization is overlapping:
//   - WGMMA computation for tile T+1
//   - NVLink all-reduce for tile T

// Reduce-scatter variant for row-parallel GEMM:
//   Each GPU holds a shard of the output, reducing communication volume
```

---

## 31.11 State Space Decomposition (SSD)

### 31.11.1 SSD Overview

State Space Decomposition implements the core operation of selective state space models (SSMs) like Mamba. The SSD operation combines:

- **Discretization**: Converting continuous SSM parameters to discrete form.
- **Parallel scan**: Computing the recurrent state progression in parallel.
- **Matrix operations**: Matrix-vector multiply at each timestep.

```
SSD operation:
  For each timestep t:
    state_t = A_t * state_{t-1} + B_t * x_t
    y_t = C_t * state_t

  Where A_t, B_t, C_t are time-varying parameters derived from input
```

### 31.11.2 SSD Kernel on SM90

```cpp
// SSD kernel using TMA and WGMMA on Hopper
// The key insight is that the parallel scan can be reformulated as
// a matrix operation that maps to WGMMA:

// Chunk-level computation:
//   Y_chunk = (C * A^{L-1} * B) * X_chunk  (matrix form of the scan)
//
// Where A^{L-1} * B is a lower-triangular matrix that can be computed
// using a modified GEMM

// CUTLASS implements SSD as:
//   1. Load discretized A, B, C parameters (TMA)
//   2. Compute intra-chunk states (WGMMA)
//   3. Compute inter-chunk propagation (scan + WGMMA)
//   4. Store output (TMA store)
```

---

## 31.12 Dynamic Shared Memory Management

### 31.12.1 Shared Memory Allocation on SM90

Hopper supports up to 227 KB of shared memory per SM. CUTLASS 3.x kernels dynamically allocate shared memory:

```cpp
// SM90 kernel shared memory layout
// The shared memory is divided between:
//   - A tile buffers (multi-stage)
//   - B tile buffers (multi-stage)
//   - Epilogue workspace
//   - Barrier space (NamedBarriers)

// Calculate shared memory requirement
using SmemSizeA = cute::sizeof_bits<ElementA>::value * TileShape::kM * TileShape::kK / 8;
using SmemSizeB = cute::sizeof_bits<ElementB>::value * TileShape::kK * TileShape::kN / 8;

// For N stages:
constexpr size_t smem_size = (SmemSizeA + SmemSizeB) * Stages + EpilogueWorkspaceSize;

// Set shared memory carveout for the kernel
cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    smem_size
);
```

### 31.12.2 Shared Memory Carveout

```cpp
// SM90 allows configuring how much shared memory is reserved for the kernel
// vs. available for L1 cache

// Maximum shared memory configuration:
cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    227 * 1024  // 227 KB max
);

// Preferred carveout (shared memory vs L1)
cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared  // Maximize shared memory
);
```

---

## 31.13 Complete SM90 GEMM Example

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

// SM90 FP8 GEMM with warp-specialized pingpong schedule
namespace {

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccum = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// CollectiveBuilder automatically selects the optimal configuration
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 16,    // 16-byte alignment for TMA
    ElementB, LayoutB, 16,
    ElementAccum,
    cutlass::gemm::GemmShape<128, 128, 64>,  // Threadblock tile
    cutlass::gemm::collective::StageCountAutoCarveout<0>,  // Auto stages
    cutlass::gemm::collective::KernelTmaWarpSpecializedPingpong  // Schedule
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

// Launch the GEMM
void run_sm90_gemm(
    ElementA *ptr_A, ElementB *ptr_B,
    ElementC *ptr_C, ElementD *ptr_D,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream
) {
    Gemm gemm_op;
    typename Gemm::Arguments args{
        {M, N, K},                          // Problem size
        {ptr_A, M},                         // Tensor A (row-major: stride = M)
        {ptr_B, N},                         // Tensor B (col-major: stride = N)
        {ptr_C, M},                         // Tensor C
        {ptr_D, M},                         // Tensor D (output)
        {alpha, beta}                       // Epilogue parameters
    };

    // Get workspace size and allocate
    size_t workspace_size = Gemm::get_workspace_size(args);
    void *workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    // Run the GEMM
    auto status = gemm_op(args, workspace, stream);

    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "GEMM failed: %d\n", (int)status);
    }

    if (workspace) {
        cudaFree(workspace);
    }
}
```

---

## 31.14 Summary

Hopper (SM90) introduces several transformative features to CUTLASS:

- **TMA (Tensor Memory Accelerator)**: Offloads address generation and bulk data transfer from threads to dedicated hardware. Supports multi-cast for clusters.
- **WGMMA (Warp Group MMA)**: 128-thread cooperative matrix multiply instruction with async execution and support for FP8, FP16, BF16, TF32, and INT8.
- **Thread Block Clusters**: Groups of CTAs with distributed shared memory for cooperative computation.
- **Warp Specialization**: Hardware-enforced producer-consumer split between data movement (TMA) and computation (WGMMA) warp groups.
- **Kernel Schedules**: WarpSpecialized, Pingpong (double-buffer), and Cooperative (multi-consumer) schedules for different workload characteristics.
- **Mixed Dtype**: Native support for mixed input types (e.g., FP8 inputs, FP32 accumulation, FP16 output).
- **FMHA**: Fused Multi-Head Attention using flash attention algorithm with online softmax.
- **Distributed GEMM**: Tensor Parallelism with NVLink communication-computation overlap.
- **SSD**: State Space Decomposition for selective state space models.
