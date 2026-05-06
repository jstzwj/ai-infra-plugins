# CUTLASS - Chapter 13: Pipeline Operations

This reference covers pipeline operations in CUTLASS, which implement multi-stage data movement patterns using producer-consumer synchronization. Pipelines are fundamental to achieving high performance in CUTLASS kernels by overlapping computation with data movement across the GPU memory hierarchy.

---

## 13.1 Pipeline Concept

A **pipeline** in CUTLASS is an abstraction for coordinating multi-stage data movement between global memory, shared memory, and register files. The pipeline manages a set of buffer stages (typically in shared memory) through which data flows, with producer operations (loads) and consumer operations (compute) operating concurrently on different stages.

The key insight behind pipelining is that data movement and computation can overlap:

- While the **producer** (load operation) fills stage `N+1`, the **consumer** (MMA operation) processes stage `N`.
- This hides memory latency by ensuring the consumer always has data ready to process.
- More pipeline stages allow more overlap, up to the limits of shared memory capacity.

**Visual representation of a 3-stage pipeline:**

```
Time ->   T0       T1       T2       T3       T4       T5
Stage 0: [LOAD]   [COMPUTE] ........ ........ ........
Stage 1: ........ [LOAD]   [COMPUTE] ........ ........
Stage 2: ........ ........ [LOAD]   [COMPUTE] ........
                    ^                   ^
                    |                   |
              Overlap: load           Overlap: load
              T1 overlaps             T3 overlaps
              compute T0              compute T2
```

---

## 13.2 SM80 Pipeline (Ampere)

The SM80 pipeline is based on the `cp.async` instruction family introduced in the Ampere architecture. It provides asynchronous copies from global memory to shared memory with barrier synchronization.

### 13.2.1 PipelineState Tracking

`PipelineState` tracks which stage of the pipeline is currently being processed. It wraps a simple integer index that cycles through the pipeline stages modulo the stage count.

```cpp
// PipelineState tracks the current stage index
struct PipelineState {
    int index;   // Current stage index (wraps around)

    // Advance to the next stage
    PIPELINE_STATE operator++() {
        index = (index + 1) % Stages;
        return *this;
    }

    // Access current stage
    int operator*() const { return index; }
};
```

In CUTLASS 3.x, `PipelineState` is a lightweight wrapper:

```cpp
#include "cutlass/pipeline/pipeline.hpp"

// Create pipeline state
auto producer_state = cutlass::PipelineState<Stages>{0};  // Starting at stage 0
auto consumer_state = cutlass::PipelineState<Stages>{0};

// Advance states
++producer_state;  // Move to next stage for next load
++consumer_state;  // Move to next stage for next compute
```

### 13.2.2 Barrier Synchronization

The SM80 pipeline uses hardware barriers to synchronize between producers and consumers. The barrier ensures that:

1. The consumer does not read from a stage until the producer has finished writing to it.
2. The producer does not overwrite a stage until the consumer has finished reading from it.

```cpp
#include "cutlass/arch/barrier.h"

// Shared memory barrier (using mbarrier instruction on SM80+)
// Each pipeline stage has an associated barrier

// Producer signals completion:
// cutlass::arch::PipelineBarrier::arrive_inc(barrier_ptr);
//   - Increments the barrier's arrival count

// Consumer waits for completion:
// cutlass::arch::PipelineBarrier::wait(barrier_ptr, phase);
//   - Blocks until the barrier's arrival count matches the expected phase

// Consumer signals completion (so producer can reuse the stage):
// cutlass::arch::PipelineBarrier::arrive_inc(barrier_ptr);
```

### 13.2.3 SM80 Pipeline Implementation

The SM80 pipeline uses `cp.async` for global-to-shared memory copies:

```cpp
#include "cutlass/conv/threadblock/implicit_gemm_multistage.h"

// The multistage pipeline for SM80 works as follows:
template <int Stages>
void sm80_pipeline_mainloop(/* ... */) {
    // Pipeline barriers (one per stage)
    cutlass::arch::PipelineBarrier barrier[Stages];

    // Shared memory buffers (one per stage, per operand)
    // smem_A[Stages][tile_size_A]
    // smem_B[Stages][tile_size_B]

    // --- Prologue: fill the pipeline stages ---
    for (int stage = 0; stage < Stages - 1; ++stage) {
        // Issue async copy for this stage
        cutlass::arch::cp_async<CacheOp::Global>(
            &smem_A[stage], gmem_ptr_A, sizeof(TileA));
        cutlass::arch::cp_async<CacheOp::Global>(
            &smem_B[stage], gmem_ptr_B, sizeof(TileB));

        // Commit async copies
        cutlass::arch::cp_async_fence();

        // Wait for this stage's copy to complete
        cutlass::arch::cp_async_wait<0>();
        __syncthreads();
    }

    // --- Main loop: overlapping load and compute ---
    PipelineState producer_state{0};
    PipelineState consumer_state{0};

    for (int k_iter = 0; k_iter < gemm_k_iterations; ++k_iter) {
        // Producer: load next stage
        int load_stage = *producer_state;
        cutlass::arch::cp_async<CacheOp::Global>(
            &smem_A[load_stage], gmem_ptr_A_next, sizeof(TileA));
        cutlass::arch::cp_async<CacheOp::Global>(
            &smem_B[load_stage], gmem_ptr_B_next, sizeof(TileB));
        cutlass::arch::cp_async_fence();

        // Consumer: compute on current stage
        int compute_stage = *consumer_state;
        load_from_smem(fragment_A, smem_A[compute_stage]);
        load_from_smem(fragment_B, smem_B[compute_stage]);
        mma_op(accumulators, fragment_A, fragment_B, accumulators);

        // Synchronize
        cutlass::arch::cp_async_wait<0>();
        __syncthreads();

        ++producer_state;
        ++consumer_state;
    }
}
```

### 13.2.4 cp_async Operations

The `cp.async` instruction family provides asynchronous memory copies:

```cpp
#include "cutlass/arch/memory.h"

// cp.async.ca: cache-all (can cache in L1 and L2)
cutlass::arch::cp_async<cutlass::arch::CacheOperation::Always>(
    smem_dst,      // Shared memory destination
    gmem_src,      // Global memory source
    sizeof_bytes   // Bytes to copy (must be 4, 8, or 16)
);

// cp.async.cg: cache-global (bypass L1, cache only in L2)
// Preferred for streaming data that will not be reused soon
cutlass::arch::cp_async<cutlass::arch::CacheOperation::Global>(
    smem_dst,
    gmem_src,
    16  // 16 bytes = 128 bits = 8 x half_t
);

// cp.async.cg with predication (for boundary handling)
cutlass::arch::cp_async<cutlass::arch::CacheOperation::Global>(
    smem_dst,
    gmem_src,
    16,
    predicate  // If false, the copy is skipped
);

// Fence: ensures all previously issued cp.async are committed
cutlass::arch::cp_async_fence();

// Wait: blocks until N or fewer pending cp.async operations remain
cutlass::arch::cp_async_wait<0>();   // Wait for all to complete
cutlass::arch::cp_async_wait<1>();   // Wait until 1 or fewer pending
cutlass::arch::cp_async_wait<N>();   // Wait until N or fewer pending
```

**Supported cp.async sizes:**

| Instruction | Size | Data Types |
|---|---|---|
| `cp.async.ca` | 4 bytes | INT32, FP32 |
| `cp.async.ca` | 8 bytes | FP32x2, FP16x4 |
| `cp.async.ca` | 16 bytes | FP16x8, BF16x8, TF32x4 |
| `cp.async.cg` | 16 bytes | FP16x8, BF16x8, TF32x4 |

---

## 13.3 SM90 Pipeline (Hopper)

The SM90 pipeline is built on top of TMA (Tensor Memory Accelerator) and provides significantly more efficient data movement. TMA handles address computation, bounds checking, and swizzling in hardware, freeing up threads for computation.

### 13.3.1 TMA-Based Pipeline Overview

The SM90 pipeline uses TMA for global-to-shared memory copies and WGMMA (Warp Group Matrix Multiply-Accumulate) for the compute stage. The pipeline can operate in two modes:

1. **Warp-specialized**: A dedicated warp group handles TMA loads (producer) while another warp group handles WGMMA (consumer).
2. **Non-warp-specialized**: All warps participate in both loading and computing, using barrier synchronization.

```cpp
#include "cutlass/gemm/collective/sm90_mma_tma_gmma_ss.hpp"

// SM90 TMA-based pipeline components:
// - TmaDescriptor: describes the tensor in global memory (shape, stride, etc.)
// - TmaLoad: issues a TMA copy from global to shared memory
// - PipelineBarrier: synchronizes between TMA loads and WGMMA operations
// - WGMMA: warp-group-level matrix multiply-accumulate
```

### 13.3.2 ProducerBarrier for Async Operations

The producer barrier is used by the TMA load operation to signal when a shared memory buffer is ready for consumption:

```cpp
#include "cutlass/pipeline/sm90_pipeline.hpp"

// The producer barrier is associated with each pipeline stage
// When a TMA load is issued, it targets the producer barrier
// The barrier is decremented when the TMA load completes

// Producer side: issue TMA load with barrier
auto producer_barrier = pipeline.producer_barrier(producer_state);
// The TMA load will signal completion by arriving on this barrier
tma_load(smem_buffer[stage], tma_desc, producer_barrier, coords);

// The producer can now advance to the next stage
++producer_state;
```

### 13.3.3 ConsumerBarrier for Completion Tracking

The consumer barrier is used by the compute side to wait for data to be ready and to signal when it has finished consuming the data:

```cpp
// Consumer side: wait for producer to complete
auto consumer_barrier = pipeline.consumer_barrier(consumer_state);

// Wait until the producer has finished loading this stage
pipeline.consumer_wait(consumer_state);

// ... perform WGMMA on this stage's data ...

// Signal that this stage's data has been consumed
pipeline.consumer_release(consumer_state);
++consumer_state;
```

### 13.3.4 TMA Load Operations

TMA provides hardware-accelerated tensor loading with built-in features:

```cpp
#include "cute/arch/copy_sm90_tma.hpp"

// TMA load features:
// - Multi-dimensional coordinate-based addressing
// - Automatic bounds checking in hardware
// - Swizzling for bank-conflict-free shared memory layout
// - No thread participation (only 1 thread issues the TMA)
// - Cluster multicast: one TMA load can fill shared memory in multiple CTAs

// Create a TMA descriptor
// auto tma_desc = make_tma_copy(
//     SM90_TMA_LOAD{},
//     tensor_G,        // Global tensor
//     smem_layout      // Shared memory layout
// );

// Issue TMA load
// copy(tma_desc.with(smem_ptr, barrier), tensor_G(coords));
```

**TMA load variants:**

| Variant | Description |
|---|---|
| `SM90_TMA_LOAD` | Standard TMA load to single CTA's shared memory |
| `SM90_TMA_LOAD_MULTICAST` | TMA load to multiple CTAs in a cluster |
| `SM90_TMA_STORE` | TMA store from shared memory to global memory |
| `SM90_TMA_LOAD_IM2COL` | TMA load with implicit im2col for convolution |

### 13.3.5 SM90 Pipeline Example (Warp-Specialized)

```cpp
#include "cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp"

// Simplified SM90 warp-specialized pipeline
template <int Stages, typename CollectiveOp>
void sm90_warpspecialized_mainloop(
    CollectiveOp collective,
    int k_tile_count)
{
    // The pipeline is split into producer (TMA) and consumer (WGMMA) warp groups
    // Producer warp group:
    if (is_producer_warp()) {
        auto producer_state = cutlass::PipelineState<Stages>{0};

        for (int k = 0; k < k_tile_count; ++k) {
            // Wait for consumer to release this stage
            pipeline.producer_acquire(producer_state);

            // Issue TMA load for operand A
            auto barrier_A = pipeline.producer_barrier(producer_state);
            collective.tma_load_A(smem_A[*producer_state], barrier_A, k);

            // Issue TMA load for operand B
            auto barrier_B = pipeline.producer_barrier_B(producer_state);
            collective.tma_load_B(smem_B[*producer_state], barrier_B, k);

            // Commit loads and advance
            ++producer_state;
        }
        // Signal end of production
        pipeline.producer_tail(producer_state);
    }

    // Consumer warp group:
    if (is_consumer_warp()) {
        auto consumer_state = cutlass::PipelineState<Stages>{0};

        for (int k = 0; k < k_tile_count; ++k) {
            // Wait for producer to complete this stage
            pipeline.consumer_wait(consumer_state);

            // Load from SMEM to registers (via TiledCopy)
            copy(tiled_copy_A, smem_A[*consumer_state], reg_A);
            copy(tiled_copy_B, smem_B[*consumer_state], reg_B);

            // WGMMA operation
            cute::gemm(tiled_mma, reg_A, reg_B, accumulators);

            // Signal that this stage is consumed
            pipeline.consumer_release(consumer_state);
            ++consumer_state;
        }
    }
}
```

---

## 13.4 SM100 Pipeline (Blackwell)

The SM100 pipeline extends the SM90 model with additional capabilities for the Blackwell architecture:

```cpp
// SM100 pipeline enhancements:
// 1. TC (Tensor Controller) managed pipelines for even lower latency
// 2. Cluster-level barriers with distributed shared memory support
// 3. Enhanced TMA with UMMA (Unified MMA) integration
// 4. Block-scaled data type support in the pipeline
// 5. Larger pipeline stage counts with better barrier management

// SM100 uses similar pipeline patterns as SM90 but with:
// - UMMA operations instead of WGMMA
// - Block-scaled type handling (NVFP4, MXFP4/6/8)
// - Distributed GEMM support across multiple CTAs

#include "cutlass/gemm/collective/sm100_mma_umma.hpp"

// SM100 pipeline follows the same producer-consumer pattern
// but with enhanced hardware support for:
// - Automatic scale factor loading
// - Cluster-level barrier synchronization
// - Distributed shared memory access
```

---

## 13.5 Pipeline Patterns

### 13.5.1 Single-Buffer (No Pipelining)

The simplest pattern with no overlap between load and compute:

```cpp
// Single-buffer: no pipelining
// Load -> Compute -> Load -> Compute -> ...
// Maximum latency, minimum shared memory

for (int k = 0; k < k_tiles; ++k) {
    // Load tile from global to shared memory (blocking)
    load_tile_gmem_to_smem(smem_A, gmem_A[k]);
    load_tile_gmem_to_smem(smem_B, gmem_B[k]);
    __syncthreads();

    // Compute on the tile
    compute_mma(accumulators, smem_A, smem_B);
    __syncthreads();
}
```

**Characteristics:**

- Simplest implementation.
- No overlap between load and compute.
- Minimum shared memory usage (1 stage).
- Maximum memory latency exposure.
- Rarely used in production due to poor performance.

### 13.5.2 Double-Buffer (2-Stage Pipeline)

The double-buffer pattern uses two shared memory buffers to overlap one load with one compute:

```cpp
// Double-buffer: 2-stage pipeline
// Buffer 0: [LOAD_0][COMPUTE_0][LOAD_2][COMPUTE_2]...
// Buffer 1: ........[LOAD_1]...[COMPUTE_1][LOAD_3]...
// Load of next tile overlaps with compute of current tile

constexpr int Stages = 2;

// Prologue: load first tile
load_tile(smem_A[0], gmem_A[0]);
load_tile(smem_B[0], gmem_B[0]);
commit_and_wait();

for (int k = 0; k < k_tiles - 1; ++k) {
    int current = k % 2;
    int next = (k + 1) % 2;

    // Start loading next tile (async)
    async_load(smem_A[next], gmem_A[k + 1]);
    async_load(smem_B[next], gmem_B[k + 1]);

    // Compute on current tile
    compute_mma(accumulators, smem_A[current], smem_B[current]);

    // Wait for next load to complete
    commit_and_wait();
    __syncthreads();
}

// Epilogue: compute last tile
compute_mma(accumulators, smem_A[(k_tiles - 1) % 2], smem_B[(k_tiles - 1) % 2]);
```

**Characteristics:**

- Moderate complexity.
- Overlaps one load with one compute.
- Requires 2x the shared memory of single-buffer.
- Good baseline for most workloads.
- The standard pattern for SM80 (Ampere) kernels.

### 13.5.3 Triple-Buffer (3+ Stage Pipeline)

The triple-buffer (or more generally, multi-stage) pattern extends the overlap further:

```cpp
// Triple-buffer: 3-stage pipeline
// Allows up to 2 loads in flight while computing
// Better overlap but more shared memory

constexpr int Stages = 3;

// Prologue: fill first 2 stages
for (int s = 0; s < Stages - 1; ++s) {
    async_load(smem_A[s], gmem_A[s]);
    async_load(smem_B[s], gmem_B[s]);
    commit_and_wait();
    __syncthreads();
}

// Main loop
for (int k = 0; k < k_tiles - (Stages - 1); ++k) {
    int compute_stage = k % Stages;
    int load_stage = (k + Stages - 1) % Stages;

    // Issue async load for the "load ahead" stage
    async_load(smem_A[load_stage], gmem_A[k + Stages - 1]);
    async_load(smem_B[load_stage], gmem_B[k + Stages - 1]);
    commit();

    // Compute on the current stage
    compute_mma(accumulators, smem_A[compute_stage], smem_B[compute_stage]);

    // Wait for the load we just issued
    wait();
    __syncthreads();
}

// Epilogue: drain remaining stages
for (int s = 0; s < Stages - 1; ++s) {
    int compute_stage = (k_tiles - Stages + 1 + s) % Stages;
    compute_mma(accumulators, smem_A[compute_stage], smem_B[compute_stage]);
}
```

**Characteristics:**

- Maximum overlap between load and compute.
- Higher shared memory requirements.
- Can fully hide memory latency when there are enough stages.
- The standard pattern for SM90 (Hopper) kernels (often 3-7 stages).

**Stage count selection guide:**

| Architecture | Typical Stages | Considerations |
|---|---|---|
| SM80 (Ampere) | 2-4 | Limited by 164KB shared memory per SM |
| SM90 (Hopper) | 2-10 | Up to 228KB shared memory; TMA is very efficient |
| SM100 (Blackwell) | 2-10+ | Even larger shared memory; UMMA integration |

---

## 13.6 Warp Specialization Pipeline

Warp specialization is a key feature of SM90 (Hopper) that assigns different warp groups to different roles within the kernel. This enables true concurrent execution of load and compute operations.

### 13.6.1 Producer Warp Group

The producer warp group is responsible for issuing TMA loads and managing the pipeline barriers:

```cpp
// Producer warp group (typically warp group 0 = warps 0-3)
// - Issues TMA load instructions
// - Manages producer barriers
// - Signals data availability to consumers

// Key responsibilities:
// 1. Acquire pipeline stage (wait for consumer to release)
// 2. Issue TMA load with barrier
// 3. Advance to next stage
// 4. Handle tail (signal end of production)

void producer_mainloop(Pipeline& pipeline, int k_tiles) {
    auto state = PipelineState<Stages>{0};

    for (int k = 0; k < k_tiles; ++k) {
        // Acquire: wait until consumer has released this stage
        pipeline.producer_acquire(state);

        // Issue TMA loads for operands A and B
        auto barrier = pipeline.producer_barrier(state);
        tma_load_A(smem_A[*state], barrier, k);
        tma_load_B(smem_B[*state], barrier, k);

        ++state;
    }

    // Tail: release any remaining stages
    pipeline.producer_tail(state);
}
```

### 13.6.2 Consumer Warp Group

The consumer warp group performs the WGMMA computation:

```cpp
// Consumer warp group (typically warp group 1 = warps 4-7)
// - Waits for data to be ready
// - Copies from SMEM to registers
// - Performs WGMMA
// - Releases pipeline stage

void consumer_mainloop(Pipeline& pipeline, int k_tiles) {
    auto state = PipelineState<Stages>{0};

    for (int k = 0; k < k_tiles; ++k) {
        // Wait: block until producer has loaded this stage
        pipeline.consumer_wait(state);

        // SMEM -> Register copy
        copy(tiled_copy_A, tAsA(smem_A[*state]), tCrA(reg_A));
        copy(tiled_copy_B, tBsB(smem_B[*state]), tCrB(reg_B));

        // WGMMA
        cute::gemm(tiled_mma, tCrA, tCrB, tCrC(accum));

        // Release: signal that this stage can be reused
        pipeline.consumer_release(state);
        ++state;
    }
}
```

### 13.6.3 Cooperative vs. Ping-Pong Scheduling

CUTLASS supports two scheduling modes for warp-specialized kernels:

**Cooperative Scheduling:**

Both warp groups cooperate on the same output tile. The accumulator is split between the two warp groups, and each warp group computes a portion of the output.

```cpp
// Cooperative: both warp groups compute the SAME output tile
// Warp group 0: computes rows [0, M/2) of the output tile
// Warp group 1: computes rows [M/2, M) of the output tile
// Both warp groups share the same accumulator (split across warp groups)
//
// Benefits:
// - Better utilization for small output tiles
// - Shared pressure on memory system
// - Reduced register pressure per warp group

using Schedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
```

**Ping-Pong Scheduling:**

Warp groups alternate between producer and consumer roles, each computing a different output tile.

```cpp
// Ping-Pong: warp groups alternate between producer and consumer
// Warp group 0: produces for tile 0, consumes tile 1, produces for tile 2, ...
// Warp group 1: consumes tile 0, produces for tile 1, consumes tile 2, ...
//
// Benefits:
// - Better utilization of TMA and WGMMA units
// - Higher throughput for large tiles
// - Reduced pipeline stalls

using Schedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
```

**Choosing between cooperative and ping-pong:**

| Factor | Cooperative | Ping-Pong |
|---|---|---|
| Output tile size | Small tiles | Large tiles |
| Register pressure | Lower (split accum) | Higher (full accum) |
| Memory throughput | Moderate | High |
| Best for | Batched small GEMMs | Large GEMMs |

---

## 13.7 Cluster-Level Pipelines

Thread block clusters (SM90+) allow multiple CTAs to coordinate their pipeline operations, enabling shared access to distributed shared memory and multicast TMA loads.

### 13.7.1 Multi-CTA Coordination

In a cluster, CTAs can coordinate through:

1. **Distributed shared memory (DSMEM)**: CTAs in a cluster can read each other's shared memory.
2. **Cluster-level barriers**: Synchronization across CTAs in a cluster.
3. **Multicast TMA**: One TMA load can write to multiple CTAs' shared memory simultaneously.

```cpp
#include "cutlass/cluster_launch.hpp"

// Cluster launch configuration
dim3 grid_dims(num_ctas_m, num_ctas_n, num_ctas_k);
dim3 cluster_dims(cluster_m, cluster_n, 1);  // e.g., 2x2 = 4 CTAs per cluster
dim3 block_dims(threads_per_cta);

cutlass::ClusterLaunchParams launch_params{
    grid_dims, block_dims, cluster_dims, smem_size, stream
};

// Within the kernel, CTAs coordinate:
// 1. CTA (0,0) issues TMA load that multicasts to all CTAs in cluster
// 2. All CTAs wait for the load to complete (cluster barrier)
// 3. Each CTA reads from its own SMEM (which has the multicast data)
// 4. Each CTA computes on its assigned portion of the output
```

### 13.7.2 Distributed Shared Memory

DSMEM allows CTAs in a cluster to access each other's shared memory:

```cpp
// Access another CTA's shared memory
// CTA (cx, cy) can read CTA (cx', cy')'s shared memory

// In CuTe:
// auto remote_smem = make_tensor(
//     make_smem_ptr(reinterpret_cast<Element*>(smem_ptr) + offset),
//     remote_layout
// );

// Use cluster barrier for synchronization
// cutlass::arch::NamedBarrier::sync();  // Synchronize across cluster

// Typical pattern for DSMEM usage:
// 1. Each CTA loads a different tile via TMA
// 2. CTAs exchange tiles through DSMEM
// 3. Each CTA computes on the combined data
```

### 13.7.3 TMA Multicast

TMA multicast allows a single TMA load to write to multiple CTAs' shared memory simultaneously:

```cpp
// TMA multicast: one load broadcasts to all CTAs in the cluster
// This saves global memory bandwidth by reading data only once

// Create a multicast TMA descriptor
// auto tma_desc_multicast = tma_desc.with(smem_ptr, barrier, multicast_mask);

// The multicast mask specifies which CTAs in the cluster receive the data
// For a 2x2 cluster:
//   mask = 0b1111 = all 4 CTAs
//   mask = 0b0001 = only CTA (0,0)

// In CUTLASS 3.x CollectiveBuilder:
// Multicast is enabled automatically when using cluster dimensions > 1
```

---

## 13.8 Async Copy Operations

### 13.8.1 cp.async (SM80+)

The `cp.async` instruction provides asynchronous global-to-shared memory copies:

```cpp
#include "cutlass/arch/memory.h"

// Basic cp.async usage
// - Source: global memory
// - Destination: shared memory
// - Size: 4, 8, or 16 bytes per thread
// - Async: returns immediately, data arrives later

// Issue async copy
cutlass::arch::cp_async_ca(smem_ptr, gmem_ptr, 16);   // 16 bytes, cache-all
cutlass::arch::cp_async_cg(smem_ptr, gmem_ptr, 16);   // 16 bytes, cache-global

// Commit: ensure all cp.async are visible to the memory system
cutlass::arch::cp_async_fence();

// Wait: block until N or fewer pending
cutlass::arch::cp_async_wait<0>();  // Wait for all

// Predicated (for boundary handling)
cutlass::arch::cp_async_cg(smem_ptr, gmem_ptr, 16, /*predicate=*/guard);
```

### 13.8.2 TMA Copy (SM90+)

TMA provides hardware-accelerated tensor copies with minimal thread participation:

```cpp
#include "cute/arch/copy_sm90_tma.hpp"

// TMA advantages over cp.async:
// 1. Only 1 thread issues the copy (vs all threads for cp.async)
// 2. Hardware handles address computation
// 3. Hardware handles bounds checking
// 4. Hardware handles swizzling
// 5. Supports multicast
// 6. Much higher throughput for large tiles

// TMA load
// cute::copy(SM90_TMA_LOAD{}, tma_desc, smem_dst, gmem_coords);

// TMA store (SMEM -> GMEM)
// cute::copy(SM90_TMA_STORE{}, tma_desc, smem_src, gmem_coords);

// TMA im2col (for convolution)
// cute::copy(SM90_TMA_LOAD_IM2COL{}, tma_desc, smem_dst, gmem_coords, im2col_coords);
```

### 13.8.3 cp.async.bulk (SM90+)

Bulk async copy provides large, efficient data movement:

```cpp
#include "cutlass/arch/memory_sm90.h"

// cp.async.bulk copies large contiguous blocks
// - Minimum size: 128 bytes
// - Aligned to 16 bytes
// - Can copy up to 256 bytes per instruction

// Used internally by TMA and for bulk data movement
// Not typically used directly in application code
```

---

## 13.9 Barrier Management

### 13.9.1 NamedBarrier

Named barriers provide synchronization between specific sets of threads within a CTA or cluster:

```cpp
#include "cutlass/arch/barrier.h"

// Named barriers are hardware primitives (SM90+)
// Each named barrier has:
// - A name (0-15 for SM90)
// - An expected arrival count
// - A phase bit for tracking completion

// Barrier names used in CUTLASS:
// Name 0: typically reserved for __syncthreads()
// Name 1-7: producer/consumer barriers for pipeline stages
// Name 8-15: user-defined barriers

// Arrive on a named barrier
cutlass::arch::NamedBarrier::arrive(BarrierName, ArrivalCount);

// Wait on a named barrier
cutlass::arch::NamedBarrier::wait(BarrierName, PhaseBit);

// Example: synchronize between two warp groups
// Warp group 0 (producer):
cutlass::arch::NamedBarrier::arrive(/*name=*/1, /*count=*/64);  // 64 = 2 warp groups
// Warp group 1 (consumer):
cutlass::arch::NamedBarrier::wait(/*name=*/1, /*phase=*/0);
```

### 13.9.2 LinearCombination Barrier

CUTLASS uses a combination of named barriers and software counters to implement the pipeline synchronization:

```cpp
// The pipeline uses a linear combination of:
// 1. Phase bits to track whether a stage is being produced or consumed
// 2. Arrival counts to know when all producers/consumers are done
// 3. Named barriers for inter-thread-group synchronization

// The synchronization protocol:
// Producer acquires: waits for consumer to release (phase flip)
// Producer commits: arrives on barrier (signals data is ready)
// Consumer waits: waits for producer commit (phase match)
// Consumer releases: arrives on barrier (signals data consumed)
```

### 13.9.3 SyncProvider

The `SyncProvider` concept abstracts different synchronization mechanisms:

```cpp
// CUTLASS pipeline sync providers:
// 1. CTA-level sync: __syncthreads() based (all threads in CTA participate)
// 2. Warp-group sync: named barrier based (specific warp groups)
// 3. Cluster sync: cluster barrier based (multiple CTAs)

// The pipeline template takes a SyncProvider to determine the sync mechanism:
template <int Stages, typename SyncProvider>
class Pipeline;

// Common sync providers:
using CTASync = cutlass::ArchSync;            // CTA-level __syncthreads()
using WarpGroupSync = cutlass::WarpGroupSync; // Warp group named barrier
using ClusterSync = cutlass::ClusterSync;      // Cluster barrier
```

---

## 13.10 CUTLASS 3.x Pipeline API

CUTLASS 3.x provides a unified pipeline API that works across architectures:

```cpp
#include "cutlass/pipeline/pipeline.hpp"

// Generic pipeline creation
template <int Stages, typename BarrierType>
auto pipeline = cutlass::make_pipeline<Stages, BarrierType>(barrier_storage);

// Pipeline operations (architecture-agnostic interface):
// pipeline.producer_acquire(state)  - Acquire a stage for production
// pipeline.producer_commit(state)   - Signal production complete
// pipeline.consumer_wait(state)     - Wait for production to complete
// pipeline.consumer_release(state)  - Signal consumption complete
// pipeline.producer_tail(state)     - Signal end of all production
```

### 13.10.1 Pipeline in CollectiveBuilder

The `CollectiveBuilder` automatically selects the appropriate pipeline configuration:

```cpp
using CollectiveOp = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape,
    StageCount,       // Controls number of pipeline stages
    ScheduleType      // Controls pipeline type (auto, cooperative, pingpong)
>::CollectiveOp;

// ScheduleType options for SM90:
// cutlass::gemm::collective::KernelScheduleAuto
// cutlass::gemm::collective::KernelTmaWarpSpecialized
// cutlass::gemm::collective::KernelTmaWarpSpecializedCooperative
// cutlass::gemm::collective::KernelTmaWarpSpecializedPingpong
// cutlass::gemm::collective::KernelCpAsyncWarpSpecialized
```

---

## 13.11 Complete Pipeline Example (SM80)

```cpp
// Complete SM80 multistage GEMM pipeline example
#include "cutlass/gemm/threadblock/mma_multistage.h"
#include "cutlass/arch/memory.h"

template <
    int Stages,
    typename MmaOp,
    int kThreadCount
>
void sm80_multistage_gemm(
    typename MmaOp::IteratorA iterator_A,
    typename MmaOp::IteratorB iterator_B,
    typename MmaOp::FragmentC& accumulators,
    int gemm_k_iterations)
{
    using FragmentA = typename MmaOp::FragmentA;
    using FragmentB = typename MmaOp::FragmentB;

    // Shared memory buffers (multi-stage)
    __shared__ FragmentA smem_A[Stages];
    __shared__ FragmentB smem_B[Stages];

    FragmentA frag_A;
    FragmentB frag_B;

    // Prologue: fill pipeline stages
    for (int s = 0; s < Stages - 1; ++s) {
        iterator_A.load(smem_A[s]);
        iterator_B.load(smem_B[s]);
        ++iterator_A;
        ++iterator_B;
        cutlass::arch::cp_async_fence();
        cutlass::arch::cp_async_wait<0>();
        __syncthreads();
    }

    // Main pipeline loop
    for (int k = 0; k < gemm_k_iterations - (Stages - 1); ++k) {
        int smem_write = (k + Stages - 1) % Stages;
        int smem_read = k % Stages;

        // Async load next tile (producer)
        iterator_A.load(smem_A[smem_write]);
        iterator_B.load(smem_B[smem_write]);
        cutlass::arch::cp_async_fence();

        // Compute on current tile (consumer)
        MmaOp::mma(accumulators, smem_A[smem_read], smem_B[smem_read]);

        // Wait for next load
        cutlass::arch::cp_async_wait<0>();
        __syncthreads();

        ++iterator_A;
        ++iterator_B;
    }

    // Epilogue: drain remaining stages
    for (int s = 0; s < Stages - 1; ++s) {
        int smem_read = (gemm_k_iterations - Stages + 1 + s) % Stages;
        MmaOp::mma(accumulators, smem_A[smem_read], smem_B[smem_read]);
    }
}
```

---

## 13.12 Complete Pipeline Example (SM90 Warp-Specialized)

```cpp
// Complete SM90 warp-specialized pipeline example
#include "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp"

// Using the CollectiveBuilder to set up the pipeline automatically:
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
    cutlass::gemm::collective::KernelTmaWarpSpecializedPingpong
>::CollectiveOp;

using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::layout::RowMajor,
    cutlass::layout::RowMajor,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>;

using Kernel = cutlass::gemm::kernel::GemmUniversal<
    CollectiveOp, EpilogueOp
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

// Launch
Gemm gemm_op;
typename Gemm::Arguments args{
    {M, N, K},                    // GemmCoord
    {ptr_A, stride_A},            // TensorA
    {ptr_B, stride_B},            // TensorB
    {ptr_C, stride_C},            // TensorC
    {ptr_D, stride_D},            // TensorD
    {alpha, beta}                 // Epilogue scalars
};

// The pipeline is managed internally by the CollectiveOp
// - Producer warp group issues TMA loads
// - Consumer warp group performs WGMMA
// - Pipeline barriers synchronize between them
// - Pingpong mode alternates roles for maximum throughput

gemm_op(args);
```

---

## 13.13 Key Header Files Reference

| Header | Purpose |
|---|---|
| `cutlass/pipeline/pipeline.hpp` | Core pipeline abstraction (3.x) |
| `cutlass/pipeline/sm90_pipeline.hpp` | SM90 TMA-based pipeline |
| `cutlass/pipeline/sm80_pipeline.hpp` | SM80 cp.async-based pipeline |
| `cutlass/arch/memory.h` | cp.async operations |
| `cutlass/arch/memory_sm90.h` | SM90 memory operations |
| `cutlass/arch/barrier.h` | Barrier and named barrier operations |
| `cute/arch/copy_sm90_tma.hpp` | TMA copy instructions |
| `cutlass/gemm/threadblock/mma_multistage.h` | SM80 multistage GEMM |
| `cutlass/gemm/collective/sm90_mma_tma_gmma_ss.hpp` | SM90 TMA+GMMA collective |
| `cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp` | SM90 warp-specialized |
| `cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp` | SM90 warp-specialized kernel |
| `cutlass/cluster_launch.hpp` | Cluster launch utilities |

---

## 13.14 Summary

Pipeline operations are the mechanism by which CUTLASS overlaps data movement with computation to achieve high performance:

1. **SM80 pipeline**: Uses `cp.async` for asynchronous global-to-shared memory copies with barrier synchronization. Supports 2-4 stages typically.
2. **SM90 pipeline**: Uses TMA for hardware-accelerated tensor copies with named barrier synchronization. Supports warp specialization and 2-10+ stages.
3. **SM100 pipeline**: Extends SM90 with UMMA integration, block-scaled types, and distributed GEMM support.
4. **Pipeline patterns**: Single-buffer (no overlap), double-buffer (1-level overlap), and multi-stage (maximum overlap).
5. **Warp specialization**: Dedicated producer/consumer warp groups for true concurrent load/compute.
6. **Cluster-level pipelines**: Multi-CTA coordination with distributed shared memory and TMA multicast.
7. **Barrier management**: Named barriers, phase bits, and arrival counts coordinate producer-consumer synchronization.
