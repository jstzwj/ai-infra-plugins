# Collective Operations in CUTLASS

Collective operations in CUTLASS define the shared computation performed collaboratively by threads within a threadblock (or cluster). They are the central abstraction in CUTLASS 3.x for the GEMM mainloop, encapsulating data loading, MMA execution, and pipeline management into a single composable unit.

---

## Overview

### What is a Collective Operation?

A collective operation is the code executed by a group of threads (typically all threads in a threadblock) working together to:

1. **Load tiles** of input data from global memory to shared memory (or directly to registers).
2. **Execute MMA operations** on the loaded data, accumulating partial results.
3. **Manage the pipeline** that overlaps data movement with computation.

The collective operation abstracts away the details of how tiles are loaded and computed, allowing higher-level code to simply invoke the mainloop without worrying about TMA descriptors, barrier synchronization, or warp specialization.

### Position in the Hierarchy

```
GemmUniversalAdapter  (host interface)
  |
GemmUniversal         (kernel)
  |
  +-- CollectiveMainloop  <-- This is the collective operation
  |     |
  |     +-- TiledCopy (load A/B from global to shared)
  |     +-- TiledMMA  (compute MMA from shared to registers)
  |     +-- Pipeline  (manage staging and synchronization)
  |
  +-- CollectiveEpilogue (store results)
```

---

## CollectiveMma Class

The `CollectiveMma` class is the primary collective operation for matrix multiply-accumulate mainloops.

### Template Signature

```cpp
namespace cutlass::gemm::collective {

template <
  typename DispatchPolicy_,     // Kernel dispatch policy (determines the strategy)
  typename TileShape_,          // CTA tile shape
  typename ElementA_,           // Element type for operand A
  typename StrideA_,            // Stride type for A (CuTe Stride)
  typename ElementB_,           // Element type for operand B
  typename StrideB_,            // Stride type for B (CuTe Stride)
  typename TiledMma_,           // CuTe tiled MMA operation
  typename GmemTiledCopyA_,     // Tiled copy operation for loading A
  typename SmemLayoutA_,        // Shared memory layout for A
  typename GmemTiledCopyB_,     // Tiled copy operation for loading B
  typename SmemLayoutB_,        // Shared memory layout for B
  typename TransformA_ = cute::identity,    // Optional transform on A elements
  typename TransformB_ = cute::identity     // Optional transform on B elements
>
class CollectiveMma;

} // namespace cutlass::gemm::collective
```

### Key Type Aliases

```cpp
// Within CollectiveMma, the following types are available:
using TileShape = TileShape_;
using ElementA = ElementA_;
using ElementB = ElementB_;
using TiledMma = TiledMma_;
using SmemLayoutA = SmemLayoutA_;
using SmemLayoutB = SmemLayoutB_;

// Fragment types for register storage
using FragmentA = decltype(get<0>(partition_fragment_A(TiledMma{}, TileShape{})));
using FragmentB = decltype(get<0>(partition_fragment_B(TiledMma{}, TileShape{})));
using FragmentC = decltype(get<0>(partition_fragment_C(TiledMma{}, TileShape{})));

// Accumulator type
using ElementAccumulator = typename TiledMma::ValTypeC;
```

### Core Methods

```cpp
// Constructor: stores arguments for the mainloop
CollectiveMma(Params const& params);

// Returns the shared memory storage requirement
struct SharedStorage {
    cute::array_aligned<ElementA, cute::cosize(SmemLayoutA{})> smem_A;
    cute::array_aligned<ElementB, cute::cosize(SmemLayoutB{})> smem_B;
    // For multi-stage pipelines, there may be arrays of these
};

// Main operator: execute the mainloop
template <class... Args>
void operator()(
    Params const& params,
    cute::Tensor const& tCrA,     // Register tensor for A fragments
    cute::Tensor const& tCrB,     // Register tensor for B fragments
    cute::Tensor& tCrC,            // Register tensor for C accumulators
    cute::Tensor const& tSrA,     // Shared memory tensor for A
    cute::Tensor const& tSrB,     // Shared memory tensor for B
    SharedStorage& shared_storage, // Shared memory allocation
    int tile_m, int tile_n,        // Tile coordinates
    int k_tile_count,              // Number of K tiles to process
    Args&&... args
);
```

---

## CollectiveBuilder: Convenient Builder Pattern

The `CollectiveBuilder` provides a high-level interface for constructing the right `CollectiveMma` specialization based on architecture, data types, and desired kernel schedule. It uses template metaprogramming to select the optimal implementation.

### Builder Dispatch Logic

The CollectiveBuilder uses the following decision tree to select the implementation:

```
1. Check ArchTag
   |
   +-- SM70 (Volta)
   |     +-- OpClassSimt: TwoStage SIMT mainloop
   |     +-- OpClassTensorOp: TwoStage TensorOp mainloop (HMMA.1688)
   |
   +-- SM75 (Turing)
   |     +-- OpClassSimt: TwoStage SIMT mainloop
   |     +-- OpClassTensorOp: Multistage mainloop (HMMA.1688, IMMA.8816)
   |
   +-- SM80 (Ampere)
   |     +-- OpClassSimt: Multistage SIMT mainloop
   |     +-- OpClassTensorOp: Multistage TensorOp mainloop
   |           +-- HMMA.16816 (FP16, BF16)
   |           +-- IMMA (INT8)
   |           +-- TF32 MMA
   |
   +-- SM90 (Hopper)
   |     +-- OpClassTensorOp:
   |           +-- KernelTma: Basic TMA mainloop
   |           +-- KernelTmaWarpSpecialized: Warp-specialized TMA
   |           +-- KernelTmaWarpSpecializedPingpong: Ping-pong buffering
   |           +-- KernelTmaWarpSpecializedCooperative: Cooperative MMA
   |           +-- FP8 variants with blockwise scaling
   |
   +-- SM100+ (Blackwell)
         +-- UMMA-based collectives with various specializations
```

### Architecture-Specific Implementations

#### SM70: Two-Stage Pipeline (Volta)

```cpp
// SM70 uses a simple two-stage (double-buffer) pipeline
// with synchronous global-to-shared copies (no cp.async)

// Dispatch policy:
struct KernelMultistage {
    static constexpr int Stages = 2;
    // Uses __syncthreads() for barrier synchronization
    // Loads via normal global memory access patterns
};

// CollectiveBuilder for SM70 selects:
// - OpClassSimt: thread-level MMA with double buffering
// - OpClassTensorOp: HMMA.1688 with two shared memory buffers

// Key characteristics:
// - No cp.async (not available on SM70)
// - No TMA (not available on SM70)
// - Synchronous loads with __syncthreads() barrier
// - Shared memory: 2 buffers per operand
```

#### SM80: Multistage with cp.async (Ampere)

```cpp
// SM80 introduces cp.async for asynchronous global-to-shared copies
// enabling deeper pipelines with multiple stages

// Dispatch policy:
template <int Stages>
struct KernelMultistage {
    static constexpr int stages = Stages;
    // Uses cp.async with commit+wait for pipeline management
    // Typically 2-5 stages depending on shared memory budget
};

// CollectiveBuilder for SM80 selects:
// - Multistage pipeline with cp.async
// - Warp-level MMA: HMMA.16816 (FP16), BMMA (BF16), TF32-MMA, IMMA
// - Number of stages based on TileShape and shared memory capacity

// Stage count calculation:
// smem_per_stage = sizeof(ElementA) * tile_M * tile_K + sizeof(ElementB) * tile_N * tile_K
// available_smem = shared_memory_capacity - epilogue_smem
// stages = available_smem / smem_per_stage

// Example configuration:
using CollectiveSM80 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm80,
    cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    float,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cute::Int<3>,                             // 3 stages
    cutlass::gemm::KernelMultistage
>::CollectiveOp;
```

#### SM90: TMA GMMA (Hopper)

SM90 introduces TMA (Tensor Memory Accelerator) and GMMA (General Matrix Multiply-Accumulate), enabling fundamentally different data movement and computation patterns.

##### SS (Shared-to-Shared) Variant

```cpp
// TMA loads data from global memory to shared memory
// GMMA reads from shared memory and computes MMA

// This is the standard flow:
// Global Memory --[TMA]--> Shared Memory --[GMMA]--> Register Accumulators

// The "SS" designation means both operands go through shared memory
// This is the default and most general approach
```

##### RS (Register-to-Shared) Variant

```cpp
// One operand goes through registers, the other through shared memory
// Can reduce shared memory usage for one operand

// RS is used when one operand can be kept in registers across
// multiple K-tiles, reducing shared memory pressure
```

##### Sparse Variants

```cpp
// SM90 supports structured sparsity (2:4 pattern) in operand A
// Only 50% of A elements are non-zero, reducing memory traffic

// Sparse GMMA variants:
// - Operand A has a 2:4 sparsity pattern
// - Hardware automatically handles the sparse access
// - Index metadata describes which elements are non-zero

// CollectiveBuilder detects sparse inputs and selects sparse GMMA:
using CollectiveSparse = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    float,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto,
    cutlass::gemm::collective::SparsityPtr        // Enable sparse
>::CollectiveOp;
```

#### SM100+: UMMA (Blackwell)

```cpp
// SM100 introduces UMMA (Unified Matrix Multiply-Accumulate)
// with improved throughput and additional data type support

// UMMA features:
// - Higher throughput MMA operations
// - Support for new data types (FP8, etc.)
// - Improved TMA with gather/scatter capabilities
// - Larger cluster sizes

// CollectiveBuilder for SM100:
using CollectiveSM100 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassTensorOp,
    cutlass::float_e4m3_t, cutlass::layout::RowMajor, 16,
    cutlass::float_e4m3_t, cutlass::layout::RowMajor, 16,
    float,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

---

## Dispatch Policies in Detail

### KernelMultistage (SM80)

```cpp
// Full policy definition
template <int Stages>
struct KernelMultistage {
    static constexpr int stages = Stages;

    // Pipeline management using cp.async:
    // 1. Issue cp.async for all stages up to pipeline depth
    // 2. Commit the async group
    // 3. Wait for the async group to complete
    // 4. Compute MMA from the ready stage
    // 5. Advance to next stage
    //
    // Pipeline states:
    //   - BufferEmpty: shared memory buffer is available for loading
    //   - BufferFull: shared memory buffer has valid data for MMA
    //   - BufferReady: (transitional) buffer is being processed
};

// Pipeline flow for 3 stages:
// Iteration 0: cp.async load stage_0_A, stage_0_B
// Iteration 1: MMA stage_0; cp.async load stage_1_A, stage_1_B
// Iteration 2: MMA stage_1; cp.async load stage_2_A, stage_2_B
// Iteration 3: MMA stage_2; cp.async load stage_0_A, stage_0_B (reuse buffer)
// ...

// Shared memory layout:
// SmemLayoutA: [tile_M x tile_K] per stage, Stages copies
// SmemLayoutB: [tile_N x tile_K] per stage, Stages copies
// Total smem = Stages * (tile_M * tile_K * sizeof_A + tile_N * tile_K * sizeof_B)
```

### KernelTma (SM90, Basic TMA)

```cpp
// Basic TMA kernel: all warp groups participate in both loads and MMA
struct KernelTma {
    // Uses TMA bulk copy for global-to-shared memory transfers
    // Single warp group handles both load and compute
    //
    // Pipeline:
    // 1. TMA prefetch (issue TMA copy from global to shared)
    // 2. Wait for TMA completion
    // 3. Execute GMMA on shared memory data
    // 4. Repeat for all K tiles
    //
    // Simpler than warp-specialized, but less overlap potential
};

// Usage in CollectiveBuilder:
using Collective = typename CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccum,
    TileShape,
    StageCountAuto,
    cutlass::gemm::KernelTma  // Basic TMA
>::CollectiveOp;
```

### KernelTmaWarpSpecialized (SM90)

```cpp
// Warp-specialized: divides warps into specialized roles
struct KernelTmaWarpSpecialized {
    // Warp group assignment:
    //   Warp group 0 (warps 0-3): MMA warps
    //     - Execute GMMA operations
    //     - Wait for data in shared memory
    //   Warp group 1 (warps 4-7): DMA warps
    //     - Issue TMA copies from global to shared
    //     - Manage pipeline barriers
    //
    // Overlap pattern:
    //   Time step 1: DMA loads tile_K_0, MMA idle
    //   Time step 2: DMA loads tile_K_1, MMA computes tile_K_0
    //   Time step 3: DMA loads tile_K_2, MMA computes tile_K_1
    //   ...
    //
    // Barrier synchronization:
    //   - DMA signals "data ready" after TMA completion
    //   - MMA signals "buffer consumed" after GMMA completion
    //   - Uses named barriers or cutlass::Barrier

    static constexpr bool IsWarpSpecialized = true;
};

// The two warp groups synchronize via shared memory barriers:
// DMA warps: TMA_issue -> signal_mma_barriers_arrive
// MMA warps: wait_mma_barriers -> GMMA_execute -> signal_dma_barriers
```

### KernelTmaWarpSpecializedPingpong (SM90)

```cpp
// Ping-pong: double buffering with warp specialization
struct KernelTmaWarpSpecializedPingpong {
    // Two complete sets of shared memory buffers:
    //   Set A: smem_A[0], smem_B[0]
    //   Set B: smem_A[1], smem_B[1]
    //
    // Pattern:
    //   Phase 1: DMA loads into Set A; MMA computes from Set B (initially idle)
    //   Phase 2: DMA loads into Set B; MMA computes from Set A
    //   Phase 3: DMA loads into Set A; MMA computes from Set B
    //   ...
    //
    // Advantages:
    //   - Maximum overlap: DMA and MMA never wait for each other
    //   - Each set is fully consumed before being overwritten
    //   - No need for pipeline stage tracking
    //
    // Disadvantages:
    //   - 2x shared memory usage compared to single-buffer
    //   - May not fit in shared memory for large tiles

    static constexpr int PingPongBuffers = 2;
};

// Shared memory:
// SmemLayoutA: [tile_M x tile_K] x 2 (ping and pong)
// SmemLayoutB: [tile_N x tile_K] x 2
// Total smem = 2 * (tile_M * tile_K * sizeof_A + tile_N * tile_K * sizeof_B)
```

### KernelTmaWarpSpecializedCooperative (SM90)

```cpp
// Cooperative: multiple warp groups cooperate on MMA
struct KernelTmaWarpSpecializedCooperative {
    // Warp group assignment:
    //   Warp group 0 (warps 0-3): MMA warp group 0
    //   Warp group 1 (warps 4-7): MMA warp group 1
    //   Warp group 2 (warps 8-11): DMA warps
    //
    // The two MMA warp groups cooperate on the same tile:
    //   - Each MMA warp group handles a subset of the GMMA instructions
    //   - Combined, they cover the full tile in fewer cycles
    //
    // Use case:
    //   - Large tile shapes: 256x128, 128x256, 256x256
    //   - When a single warp group cannot cover the full tile efficiently
    //
    // Note: requires at least 3 warp groups (128 threads per CTA)

    static constexpr int NumMmaWarpGroups = 2;
    static constexpr int NumDmaWarpGroups = 1;
};
```

### KernelTmaWarpSpecializedFP8Blockwise

```cpp
// FP8-specific variant with blockwise scaling factors
struct KernelTmaWarpSpecializedFP8Blockwise {
    // Designed for FP8 (E4M3/E5M2) data types with:
    //   - Blockwise scale factors for dynamic quantization
    //   - Scale factors applied during the MMA mainloop
    //
    // Scale factor handling:
    //   - Scale factors stored separately from matrix data
    //   - Each block of FP8 elements has an associated scale factor
    //   - Scale applied during dequantization in the MMA pipeline
    //
    // Input format:
    //   ElementA = cutlass::float_e4m3_t (or float_e5m2_t)
    //   Block scaling factor: float (one per block of 16 or 32 elements)
    //
    // Additional shared memory for scale factors:
    //   smem_scale_A: [tile_M x ceil(tile_K / block_size)]
    //   smem_scale_B: [tile_N x ceil(tile_K / block_size)]
};
```

---

## Mainloop Policies

Mainloop policies define the internal structure of the mainloop execution within a collective operation.

### Pipeline Management

```cpp
// Pipeline manages the ordering and synchronization of load and compute operations

// SM80 Pipeline (cp.async based):
// PipelineState tracks which buffer is being loaded and which is being computed
// cp.async issued in order, committed and waited on before MMA

// SM90 Pipeline (TMA based):
// TMA pipeline uses hardware barriers (fences) and named barriers
// TMA issue is non-blocking; completion signaled via barrier arrival

// Pipeline states:
enum class PipelineState {
    Start,          // Initial state, no data loaded
    LoadingA,       // TMA/cp.async loading A tile
    LoadingB,       // TMA/cp.async loading B tile
    ComputeReady,   // Both A and B tiles are in shared memory
    Computing,      // GMMA/HMMA executing on current tiles
    Complete        // All K tiles processed
};
```

### Pipeline Barriers

```cpp
// SM90 TMA barriers:
// Named barriers are hardware-accelerated synchronization primitives
// Each barrier has:
//   - A phase bit (alternates between 0 and 1)
//   - An arrival count (tracks how many threads have arrived)
//   - A threshold (number of arrivals needed to release)

// DMA warp usage:
// tma_store_async(ptr, gmem_tensor, smem_tensor);
// barrier_arrive(mma_barrier);  // signal to MMA: data is ready

// MMA warp usage:
// barrier_wait(mma_barrier);    // wait for DMA to signal
// gmma_operand_a(smem_tensor);  // start GMMA with shared memory data
// barrier_arrive(dma_barrier);  // signal to DMA: buffer consumed
```

---

## Fragment Management and Tensor Partitions

### Tensor Partitioning

CUTLASS 3.x uses CuTe's tensor partitioning to divide work across threads:

```cpp
// Partition the global tensors for the TiledMma
// Each thread gets a slice of the accumulator tensor

// For Tiled MMA:
// auto tiled_mma = make_tiled_mma(atom_mma, tile_layout);
// auto tCrA = partition_fragment_A(tiled_mma, tile_shape);  // register tensor for A
// auto tCrB = partition_fragment_B(tiled_mma, tile_shape);  // register tensor for B
// auto tCrC = partition_fragment_C(tiled_mma, tile_shape);  // accumulator register tensor

// For Tiled Copy (TMA):
// auto tiled_copy = make_tma_copy(tma_atom, gmem_tensor, smem_layout);
// auto tAgA = partition(tiled_copy, gmem_tensor_A);  // global tensor partitioned for TMA
// auto tAsA = smem_tensor_A;                          // shared memory tensor

// The partition maps each thread/CTA to its portion of the data:
// - For MMA: each thread holds a register fragment (small tile)
// - For TMA: the CTA-level TMA handles the full tile copy
```

### Register Fragment Lifecycle

```cpp
// Register fragments go through these phases:

// 1. Allocation (compile-time size known)
FragmentA frag_A;  // Zero-initialized or uninitialized
FragmentB frag_B;
FragmentC accum;   // Must be initialized (usually to zero or loaded from C)

// 2. Load from shared memory (inside mainloop)
// For GMMA: the MMA instruction loads directly from shared memory
// frag_A is implicitly loaded by the GMMA instruction
// (unlike 2.x where warp-level iterators explicitly load fragments)

// 3. Compute (inside mainloop)
// cute::gemm(tiled_mma, frag_A, frag_B, accum);
// This executes the MMA: accum += frag_A * frag_B

// 4. Store to shared memory or global memory (epilogue)
// The epilogue reads from accum and writes the output
```

---

## Load/Store Operations Within Collectives

### Global-to-Shared Memory Loads

```cpp
// SM80 (cp.async based):
// The collective uses PredicatedTileAccessIterator or
// direct cp.async calls to load tiles from global to shared memory

// Example internal code (simplified):
template <int Stages>
__device__ void load_tile_multistage(
    ElementA* gmem_ptr, ElementA* smem_ptr,
    int tile_k_offset, int stage
) {
    // Issue cp.async for the entire tile
    for (int offset = threadIdx.x; offset < tile_size; offset += blockDim.x) {
        cp_async<sizeof(ElementA) * cache_line_elements>(
            smem_ptr + stage * stage_size + offset,
            gmem_ptr + tile_k_offset + offset
        );
    }
    cp_async_commit();
    cp_async_wait<Stages - 1>();
}

// SM90 (TMA based):
// The collective uses TMA (Tensor Memory Accelerator) for bulk copies

// Example internal code (simplified):
__device__ void load_tile_tma(
    cute::TmaDescriptor const& tma_desc,
    cute::Tensor const& smem_tensor,
    cute::Coord<3> const& coord  // (tile_M_offset, tile_N_offset, tile_K_offset)
) {
    // TMA copy: bulk transfer from global to shared
    // Only one thread per CTA needs to issue the TMA
    if (cute::elect_one_sync()) {
        cute::copy(tma_desc, smem_tensor, coord);
    }
    // Wait for TMA completion
    cute::cp_async_wait<0>();
    __syncthreads();
}
```

### Shared Memory Layout

```cpp
// Shared memory layout is critical for avoiding bank conflicts

// For GMMA on SM90:
// Operand A in shared memory must be in a GMMA-compatible layout
// The GMMA instruction expects data in a specific swizzled pattern

// CuTe layout for shared memory:
// auto smem_layout_A = GMMA::Layout_K_SW128_Atom<ElementA>{};
// This creates a layout that matches GMMA's expected access pattern

// For SM80 (cp.async + HMMA):
// Layout is typically ColumnMajor for A (K x tile_M) and RowMajor for B (K x tile_N)
// But may be swizzled to avoid bank conflicts during warp-level loads
```

### Register File Management

```cpp
// Register allocation for fragments:
// The number of registers used depends on:
//   - Tile size (larger tiles = more registers)
//   - Element type (FP32 uses more registers than FP16)
//   - Pipeline depth (more stages = more register pressure)

// Register pressure estimation:
// accum registers: tile_M * tile_N / (warps * threads_per_warp) elements
// frag_A registers: depends on MMA instruction and warp-level tiling
// frag_B registers: similar to frag_A

// CUTLASS tries to minimize register usage through:
//   - Fragment reuse across K iterations
//   - Accumulator register sharing across output tiles
//   - Warp specialization (separating register-heavy MMA from load)
```

---

## Architecture-Specific Optimizations

### SM80 Optimizations

```cpp
// 1. cp.async with multi-stage pipeline:
//    Overlap up to (Stages-1) loads with 1 compute
//    cp.async.commit() + cp.async_wait<Stages-1>()

// 2. Shared memory swizzling:
//    XOR-based swizzle to avoid bank conflicts
//    Applied in the SmemLayoutA/B types

// 3. L2 cache persistence:
//    Use cudaAccessPolicyWindow to hint L2 cache retention
//    Beneficial for operands accessed multiple times

// 4. Warp-level HMMA.16816:
//    16x8x16 MMA per instruction
//    Each warp executes multiple HMMA to cover the warp tile
```

### SM90 Optimizations

```cpp
// 1. TMA (Tensor Memory Accelerator):
//    Bulk async copy from global to shared memory
//    Descriptor-based addressing (no per-element addressing)
//    Hardware-managed address calculation and prefetch

// 2. GMMA (General Matrix Multiply-Accumulate):
//    Async MMA operation reading directly from shared memory
//    No explicit register loads for operands (hardware managed)
//    Larger effective instruction sizes (e.g., 128x256x32 per warp group)

// 3. Warp specialization:
//    Separate DMA and MMA warp groups
//    Producer-consumer pattern with barrier synchronization
//    Better overlap of memory and compute

// 4. Cluster-level cooperation:
//    CTAs in a cluster can access each other's shared memory
//    Enables cooperative loading and computation across CTAs
//    Useful for very large tiles or cross-tile data sharing

// 5. Hardware barriers:
//    Named barriers for fast inter-warp-group synchronization
//    fences for TMA completion notification
//    Much faster than __syncthreads() for producer-consumer patterns
```

---

## Code Examples

### Example 1: Building a Collective with CollectiveBuilder

```cpp
#include "cutlass/gemm/collective/collective_builder.hpp"

// FP16 GEMM for SM90 with automatic configuration
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,                            // Architecture
    cutlass::arch::OpClassTensorOp,                 // Operator class
    cutlass::half_t,                                // ElementA
    cutlass::layout::RowMajor,                      // LayoutA
    8,                                              // AlignmentA (128-bit = 8 x FP16)
    cutlass::half_t,                                // ElementB
    cutlass::layout::RowMajor,                      // LayoutB
    8,                                              // AlignmentB
    float,                                          // ElementAccumulator
    cutlass::gemm::GemmShape<128, 128, 64>,        // TileShape
    cutlass::gemm::collective::StageCountAuto,      // Auto stage count
    cutlass::gemm::collective::KernelScheduleAuto   // Auto kernel schedule
>::CollectiveOp;

// The builder selects:
// - For SM90 + FP16 + Auto: KernelTmaWarpSpecialized with 2-3 stages
// - TiledMma: GMMA operator for FP16 -> FP32 accumulation
// - GmemTiledCopy: TMA copy atoms
// - SmemLayout: GMMA-compatible swizzled layout
```

### Example 2: Explicit Dispatch Policy Selection

```cpp
// Manually select the cooperative warp-specialized policy
using CollectiveCooperative = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    float,
    cutlass::gemm::GemmShape<256, 128, 64>,        // Larger tile
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative  // Explicit cooperative
>::CollectiveOp;

// This is useful when:
// - You know your tile shape benefits from cooperative MMA
// - You want to override the auto-selected policy
// - You are benchmarking different policies
```

### Example 3: SM80 Multistage Collective

```cpp
// FP16 GEMM for SM80 with 4-stage pipeline
using CollectiveSM80 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm80,
    cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    float,
    cutlass::gemm::GemmShape<128, 128, 32>,        // K=32 for SM80 tile
    cute::Int<4>,                                    // Exactly 4 stages
    cutlass::gemm::KernelMultistage                  // Multistage policy
>::CollectiveOp;

// Shared memory usage:
// 4 stages x (128*32*2 + 128*32*2) = 4 * 16384 = 65536 bytes = 64 KB
// Fits within the 100 KB shared memory limit of SM80 (with carveout)
```

### Example 4: BFloat16 GEMM Collective

```cpp
using CollectiveBF16 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    float,                                           // Accumulate in FP32
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

// BF16 uses the same GMMA atom as FP16 (HMMA is shared for both types)
// The accumulation is always in FP32 for both FP16 and BF16
```

### Example 5: Mixed Precision Collective

```cpp
// FP16 inputs with FP32 output
using CollectiveMixed = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,    // A: FP16
    cutlass::half_t, cutlass::layout::RowMajor, 8,    // B: FP16
    float,                                              // Accumulator: FP32
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

// Note: ElementAccumulator controls the accumulation precision
// The epilogue can convert from float accumulators to any output type
```

### Example 6: INT8 Collective

```cpp
using CollectiveINT8 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    int8_t, cutlass::layout::RowMajor, 16,             // A: INT8, 16-element alignment
    int8_t, cutlass::layout::RowMajor, 16,             // B: INT8, 16-element alignment
    int32_t,                                            // Accumulator: INT32
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

// INT8 GMMA uses IMMA instructions
// Accumulation in INT32
// Output can be dequantized in the epilogue
```

---

## Summary

Collective operations are the heart of CUTLASS 3.x GEMM performance. Key takeaways:

1. **CollectiveBuilder** provides automatic selection of the best collective implementation based on architecture, data types, and tile shape.

2. **Dispatch policies** map to hardware capabilities:
   - `KernelMultistage` for SM80 with cp.async
   - `KernelTma*` variants for SM90 with TMA and warp specialization

3. **Pipeline management** handles the overlap of data loading and MMA computation, with architecture-specific synchronization primitives.

4. **Fragment and tensor partitioning** via CuTe ensures each thread knows exactly which data it owns and operates on.

5. **Architecture-specific optimizations** (TMA, GMMA, warp specialization, cluster cooperation) are automatically applied by the CollectiveBuilder based on the target architecture.
