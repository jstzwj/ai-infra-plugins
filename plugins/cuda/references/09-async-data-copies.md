# 9. Asynchronous Data Copies

Asynchronous data copy mechanisms allow GPUs to transfer data between memory hierarchies without occupying general-purpose registers or the compute pipeline. This enables overlap of data movement with computation, which is critical for hiding memory latency and achieving high throughput on modern NVIDIA GPUs.

This section covers four generations of async copy hardware:

1. **LDGSTS** (Compute Capability 8.0+) -- global-to-shared copies via load-global-store-shared instruction
2. **Tensor Memory Accelerator (TMA)** (Compute Capability 9.0+) -- multi-dimensional tensor copies with hardware acceleration
3. **STAS** (Compute Capability 9.0+) -- register-to-distributed-shared-memory async stores
4. **Cluster Launch Control** (Compute Capability 10.0) -- work stealing patterns across cluster CTAs

---

## 9.1 LDGSTS (CC 8.0+)

The LDGSTS (Load Global, Store Shared) instruction enables asynchronous copying of data from global memory directly to shared memory without routing through registers. This is the foundational async copy mechanism introduced with the NVIDIA Ampere architecture.

### 9.1.1 Instruction Basics

The LDGSTS instruction copies 4, 8, or 16 bytes from a global memory address to a shared memory address. The transfer is asynchronous: after issuing the instruction, the warp continues executing subsequent instructions without waiting for the copy to complete. Completion must be explicitly synchronized using barriers or pipeline primitives.

**Copy sizes and behavior:**

| Copy Size | Notes |
|-----------|-------|
| 4 bytes | Basic scalar transfer |
| 8 bytes | 64-bit value transfer |
| 16 bytes | Enables L1 BYPASS mode -- data bypasses the L1 cache, reducing cache pollution |

When performing 16-byte copies, the hardware automatically enters L1 BYPASS mode, meaning the copied data does not occupy space in the L1 data cache. This is beneficial when the copied data is only needed in shared memory and should not evict other useful L1 cache lines.

**Best alignment practice:** For optimal performance, align both the source (global memory) and destination (shared memory) addresses to 128-byte boundaries. This ensures the copies map cleanly to cache-line-sized transactions and minimizes memory transaction overhead.

### 9.1.2 API Overview

LDGSTS copies are initiated through several API layers, all of which ultimately generate the same hardware instruction. The choice of API depends on the synchronization and batching model in use.

| Source Memory | Destination Memory | Completion Mechanism | API |
|---|---|---|---|
| global | shared (CTA-scoped) | barrier or pipeline | `cuda::memcpy_async`, `cooperative_groups::memcpy_async`, `__pipeline_memcpy_async` |

The three API variants are:

1. **`cuda::memcpy_async`** -- C++ standard-library style API using `cuda::barrier` for synchronization
2. **`cooperative_groups::memcpy_async`** -- Cooperative groups API using `cg::thread_block` and group-based barriers
3. **C pipeline primitives** -- Low-level `__pipeline_*` intrinsics for fine-grained control

### 9.1.3 Using cuda::memcpy_async with Barriers

The `cuda::memcpy_async` API integrates with `cuda::barrier` for synchronization. A barrier tracks in-flight async copies; threads arrive on the barrier to indicate they have submitted their copies, and then wait for the barrier to complete all outstanding transfers.

```cpp
#include <cuda/barrier>

__global__ void asyncCopyKernel(const float* __restrict__ global_in,
                                float* __restrict__ shared_out,
                                int N)
{
    // Define shared memory barrier with thread count
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

    // Initialize the barrier once per block
    if (threadIdx.x == 0) {
        init(&barrier, blockDim.x); // Expect blockDim.x arrivals
    }
    __syncthreads();

    // Each thread issues its own async copy
    int tid = threadIdx.x;
    if (tid < N) {
        // Issue async copy: 4 bytes (float) from global to shared
        cuda::memcpy_async(barrier, &shared_out[tid], &global_in[tid], sizeof(float));
    }

    // Arrive on the barrier (indicates this thread has submitted all its copies)
    // and wait for all threads' copies to complete
    barrier.arrive_and_wait();

    // shared_out is now safe to use
    shared_out[tid] *= 2.0f;
}
```

For batched copies where each thread copies multiple elements:

```cpp
__global__ void batchedAsyncCopy(const float* __restrict__ global_in,
                                 float* __restrict__ shared_out,
                                 int elements_per_thread,
                                 int total_elements)
{
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

    if (threadIdx.x == 0) {
        init(&barrier, blockDim.x);
    }
    __syncthreads();

    int tid = threadIdx.x;
    int base = tid * elements_per_thread;

    // Issue multiple async copies per thread
    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = base + i;
        if (idx < total_elements) {
            cuda::memcpy_async(barrier, &shared_out[idx], &global_in[idx], sizeof(float));
        }
    }

    // All copies issued; arrive and wait
    barrier.arrive_and_wait();

    // All data is now in shared memory
}
```

### 9.1.4 Using Pipeline Primitives

Pipeline primitives provide finer-grained control over async copy scheduling. A pipeline consists of a producer side (which issues copies and commits them) and a consumer side (which waits for committed copies to complete).

```cpp
__global__ void pipelineAsyncCopy(const float* __restrict__ global_in,
                                  float* __restrict__ shared_out,
                                  int N)
{
    int tid = threadIdx.x;

    if (tid < N) {
        // Issue the async copy
        __pipeline_memcpy_async(&shared_out[tid], &global_in[tid], sizeof(float));
    }

    // Commit all pending async copies (makes them visible to the pipeline)
    __pipeline_commit();

    // Wait for all committed copies to complete
    __pipeline_wait_prior(0);

    // Data is now available in shared_out
    shared_out[tid] += 1.0f;
}
```

The `__pipeline_wait_prior(N)` primitive waits for all pipeline commits except the most recent N stages. Passing 0 means wait for everything. This enables multi-stage pipelining where you can overlap computation on stage K with data loading for stage K+1.

### 9.1.5 Batching in Conditional Code

A key consideration when using async copies in divergent control flow is that different threads may issue different numbers of copies. The three API variants handle this differently:

#### cuda::memcpy_async with barrier (recommended for conditional code)

```cpp
__global__ void conditionalCopyBarrier(const float* __restrict__ global_in,
                                       float* __restrict__ shared_out,
                                       const int* __restrict__ predicates,
                                       int N)
{
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

    if (threadIdx.x == 0) {
        init(&barrier, blockDim.x);
    }
    __syncthreads();

    int tid = threadIdx.x;

    // Conditionally issue copies -- some threads may skip
    if (tid < N && predicates[tid]) {
        cuda::memcpy_async(barrier, &shared_out[tid], &global_in[tid], sizeof(float));
    }

    // Even threads that did NOT issue a copy must arrive on the barrier
    // This is handled automatically by arrive_and_wait()
    barrier.arrive_and_wait();

    // All issued copies are guaranteed complete
}
```

With `cuda::barrier`, every thread must participate in `arrive_and_wait()` regardless of whether it issued copies. The barrier tracks the expected number of arrivals (set during initialization), not the number of copies.

#### cooperative_groups::memcpy_async

```cpp
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

__global__ void conditionalCopyCG(const float* __restrict__ global_in,
                                  float* __restrict__ shared_out,
                                  int N)
{
    cg::thread_block block = cg::this_thread_block();
    int tid = threadIdx.x;

    // cg::memcpy_async copies the entire range for the group
    if (tid < N) {
        // All threads in the block participate in the collective copy
        cg::memcpy_async(block, shared_out, global_in, N * sizeof(float));
    }

    // Synchronize: waits for the async copy to complete
    cg::wait(block);

    // Data is available
}
```

#### C Pipeline Primitives in Conditional Code

```cpp
__global__ void conditionalCopyPipeline(const float* __restrict__ global_in,
                                        float* __restrict__ shared_out,
                                        const int* __restrict__ predicates,
                                        int N)
{
    int tid = threadIdx.x;

    // Each thread conditionally issues copies
    if (tid < N && predicates[tid]) {
        __pipeline_memcpy_async(&shared_out[tid], &global_in[tid], sizeof(float));
    }

    // All threads must call __pipeline_commit(), even if they issued no copies
    __pipeline_commit();

    // All threads must call __pipeline_wait_prior(0)
    __pipeline_wait_prior(0);
}
```

**Important:** With pipeline primitives, every thread in the warp must call `__pipeline_commit()` and `__pipeline_wait_prior()` regardless of whether it issued any copies. Failure to do so results in undefined behavior.

### 9.1.6 Prefetching with Multi-Stage Pipeline

Multi-stage pipelining overlaps data loading with computation. While stage K is being processed, stage K+1 data is being loaded asynchronously.

```cpp
#include <cuda/pipeline>

static constexpr int STAGE_COUNT = 2;

__global__ void multiStagePipeline(const float* __restrict__ global_in,
                                   float* __restrict__ output,
                                   int chunk_size,
                                   int num_chunks)
{
    // Double-buffered shared memory
    __shared__ float shared_buf[STAGE_COUNT][1024]; // Adjust size as needed

    cuda::pipeline<cuda::thread_scope_block> pipeline = cuda::make_pipeline();

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * chunk_size;

    // Prologue: fill the first stage
    pipeline.producer_acquire();
    for (int i = tid; i < chunk_size; i += blockDim.x) {
        int src_idx = block_offset + i;
        if (src_idx < num_chunks * chunk_size) {
            cuda::memcpy_async(pipeline,
                               &shared_buf[0][i],
                               &global_in[src_idx],
                               sizeof(float));
        }
    }
    pipeline.producer_commit();

    for (int stage = 1; stage < num_chunks; ++stage) {
        int curr_stage = stage % STAGE_COUNT;
        int prev_stage = (stage - 1) % STAGE_COUNT;

        // Producer: start loading the next stage
        pipeline.producer_acquire();
        for (int i = tid; i < chunk_size; i += blockDim.x) {
            int src_idx = block_offset + stage * chunk_size + i;
            if (src_idx < num_chunks * chunk_size) {
                cuda::memcpy_async(pipeline,
                                   &shared_buf[curr_stage][i],
                                   &global_in[src_idx],
                                   sizeof(float));
            }
        }
        pipeline.producer_commit();

        // Consumer: wait for and process the previous stage
        pipeline.consumer_wait();
        for (int i = tid; i < chunk_size; i += blockDim.x) {
            // Process data from shared_buf[prev_stage]
            shared_buf[prev_stage][i] = shared_buf[prev_stage][i] * 2.0f + 1.0f;
        }
        pipeline.consumer_release();
    }

    // Epilogue: process the last stage
    int last_stage = (num_chunks - 1) % STAGE_COUNT;
    pipeline.consumer_wait();
    for (int i = tid; i < chunk_size; i += blockDim.x) {
        shared_buf[last_stage][i] = shared_buf[last_stage][i] * 2.0f + 1.0f;
    }
    pipeline.consumer_release();
}
```

**Pipeline state machine:**

```
producer_acquire()  -->  memcpy_async()  -->  producer_commit()
                                                    |
                                                    v
consumer_wait()  -->  [process data]  -->  consumer_release()
                                                    |
                                                    v
                                          producer_acquire()  (next stage)
```

### 9.1.7 Warp Specialization

Warp specialization divides a thread block into producer warps (responsible for loading data) and consumer warps (responsible for computation). This avoids the overhead of every thread participating in both the producer and consumer synchronization protocol.

In the typical pattern, the first warp (warp 0) acts as the producer, and all remaining warps act as consumers. A double-buffering scheme ensures the producer is always one stage ahead of the consumers.

```cpp
#include <cuda/barrier>
#include <cuda/pipeline>

static constexpr int NUM_STAGES = 2;

__global__ void warpSpecializedKernel(const float* __restrict__ global_in,
                                      float* __restrict__ global_out,
                                      int elements_per_block)
{
    // Double-buffered shared memory
    __shared__ float smem[NUM_STAGES][2048]; // Adjust as needed
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier[NUM_STAGES];

    int tid = threadIdx.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int block_offset = blockIdx.x * elements_per_block;

    // Initialize barriers
    if (tid < NUM_STAGES) {
        init(&barrier[tid], blockDim.x);
    }
    __syncthreads();

    bool is_producer = (warp_id == 0);
    int num_consumers = blockDim.x - warpSize;

    if (is_producer) {
        // Producer warp: issue async copies for all stages
        for (int stage = 0; stage < NUM_STAGES; ++stage) {
            // Arrive on barrier to indicate readiness
            barrier[stage].arrive();

            // Issue async copies
            for (int i = lane_id; i < elements_per_block; i += warpSize) {
                int src_idx = block_offset + stage * elements_per_block + i;
                cuda::memcpy_async(barrier[stage],
                                   &smem[stage][i],
                                   &global_in[src_idx],
                                   sizeof(float));
            }
        }
    } else {
        // Consumer warps: process data stage by stage
        for (int stage = 0; stage < NUM_STAGES; ++stage) {
            // Wait for this stage's data to be ready
            barrier[stage].arrive_and_wait();

            // Process data
            for (int i = tid - warpSize; i < elements_per_block; i += num_consumers) {
                smem[stage][i] = smem[stage][i] * 3.0f - 0.5f;
            }

            // Signal completion (implicit via next barrier arrive)
        }
    }

    // Write results back
    __syncthreads();
    for (int stage = 0; stage < NUM_STAGES; ++stage) {
        for (int i = tid; i < elements_per_block; i += blockDim.x) {
            int dst_idx = block_offset + stage * elements_per_block + i;
            global_out[dst_idx] = smem[stage][i];
        }
    }
}
```

**Warp specialization considerations:**

- The producer warp should be warp 0 (first warp in the block) for simplicity
- Use separate barriers for each pipeline stage to avoid false dependencies
- Consumer warps must participate in all barrier arrivals even if they process different data
- The double-buffer pattern (2 stages) is most common; triple-buffering (3 stages) can further hide latency but uses more shared memory

---

## 9.2 Tensor Memory Accelerator (TMA) (CC 9.0+)

The Tensor Memory Accelerator (TMA), introduced with the NVIDIA Hopper architecture, provides hardware-accelerated asynchronous data transfers between global memory and shared memory. Unlike LDGSTS, which is per-thread, TMA allows a single thread to initiate bulk transfers of multi-dimensional tensors, dramatically reducing the overhead of address computation and copy issuance.

### 9.2.1 Key Features

- Supports 1D and multi-dimensional transfers (up to 5 dimensions)
- Single-thread initiation for multi-dimensional transfers
- Hardware handles all address computation, bounds checking, and swizzling
- Supports tensor map descriptors for describing multi-dimensional memory layouts
- Compatible with thread-block clusters for distributed shared memory access

### 9.2.2 TMA for 1D Arrays

For 1D transfers, TMA provides a simplified interface that does not require explicit tensor map creation. A single thread can initiate the transfer, and completion is tracked via a barrier in shared memory.

**Single-thread initiation:** Only one thread should initiate a TMA transfer. Use `cuda::ptx::elect_sync` (via `is_elected()`) to select a single thread within a warp to issue the copy.

**Alignment requirements for 1D TMA:**

| Parameter | Alignment Requirement |
|-----------|----------------------|
| Global memory address | 16-byte aligned |
| Shared memory address | 16-byte aligned |
| Barrier address | 8-byte aligned |
| Transfer size | Multiple of 16 bytes |

**Basic 1D TMA copy:**

```cpp
#include <cuda/barrier>
#include <cuda/ptx>

__global__ void tmaCopy1D(const float* __restrict__ global_in,
                          float* __restrict__ shared_out,
                          int size_bytes)
{
    // Shared memory barrier for TMA completion
    __shared__ cuda::barrier<cuda::thread_scope_block> tma_barrier;

    // Initialize barrier with expected arrival count
    if (threadIdx.x == 0) {
        init(&tma_barrier, blockDim.x);
    }
    __syncthreads();

    // Elect a single thread to initiate the TMA transfer
    auto elect = cuda::ptx::elect_sync(0xFFFFFFFF);
    if (elect.threadIdx.x() == 0) {
        // Initiate 1D TMA copy from global to shared memory
        // The barrier arrives once per thread (not per copy)
        cuda::device::memcpy_async_tx(tma_barrier,
                                      &shared_out[0],
                                      &global_in[0],
                                      size_bytes);
    }

    // All threads wait for TMA completion
    tma_barrier.arrive_and_wait();

    // shared_out is now populated
}
```

**Using `is_elected()` pattern:**

```cpp
__global__ void tmaCopyElected(const float* __restrict__ global_in,
                               float* __restrict__ shared_out,
                               int size_bytes)
{
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

    if (threadIdx.x == 0) {
        init(&barrier, blockDim.x);
    }
    __syncthreads();

    // is_elected() returns true for exactly one thread per warp
    if (cuda::ptx::is_elected()) {
        cuda::device::memcpy_async_tx(barrier,
                                      shared_out,
                                      global_in,
                                      size_bytes);
    }

    barrier.arrive_and_wait();
}
```

**Important notes on 1D TMA:**

- The transfer size must be a multiple of 16 bytes
- The barrier is used for both synchronization and TMA completion tracking
- The elected thread's call to `memcpy_async_tx` performs an implicit arrive on the barrier
- All other threads must still call `arrive_and_wait()` (or equivalent) to participate in the barrier

### 9.2.3 TMA for Multi-dimensional Arrays

For transfers involving 2D to 5D arrays, TMA uses a tensor map descriptor that describes the layout of the source tensor in global memory. The tensor map encodes dimensions, strides, and swizzling patterns, allowing the hardware to compute addresses and handle boundary conditions automatically.

**Creating a tensor map:** Tensor maps are created on the host using the driver API function `cuTensorMapEncodeTiled` and then passed to the kernel as a `const __grid_constant__` parameter.

```cpp
#include <cudaTypedefs.h>

// Host-side tensor map creation
void createTensorMap(CUtensorMap* tensor_map,
                     const float* data,
                     int dim0, int dim1,
                     int stride1)
{
    // For a 2D tensor with dimensions [dim0][dim1]
    // globalDims[] = {dim0, dim1}
    // globalStrides[] = {stride1 * sizeof(float), sizeof(float)}

    uint64_t globalDims[2] = {static_cast<uint64_t>(dim0),
                               static_cast<uint64_t>(dim1)};
    uint64_t globalStrides[2] = {static_cast<uint64_t>(stride1 * sizeof(float)),
                                  static_cast<uint64_t>(sizeof(float))};
    int32_t tileDims[2] = {16, 16}; // Tile dimensions for the TMA transfer
    uint32_t elementStrides[2] = {1, 1};

    CUtensorMapResult result = cuTensorMapEncodeTiled(
        tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2,                      // rank (number of dimensions)
        (void*)data,            // global memory address
        globalDims,             // global dimensions
        globalStrides,          // global strides (in bytes)
        tileDims,               // box/tile dimensions
        elementStrides,         // element strides within tile
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_OOB_FILL_NONE
    );

    if (result != CU_TENSOR_MAP_RESULT_SUCCESS) {
        // Handle error
    }
}
```

**Passing the tensor map to a kernel:**

```cpp
// Kernel signature using __grid_constant__
__global__ void tmaCopy2D(cuda::barrier<cuda::thread_scope_block>* barrier,
                          const __grid_constant__ CUtensorMap tensor_map,
                          int tile_x, int tile_y,
                          float* shared_out)
{
    // Coordinates for this CTA's tile
    int coord[2] = {blockIdx.y, blockIdx.x};

    // Elect a single thread to initiate the TMA transfer
    if (cuda::ptx::is_elected()) {
        // Initiate multi-dimensional TMA copy
        cuda::ptx::cp_async_bulk_tensor_2d_global_to_shared(
            shared_out,
            &tensor_map,
            coord,
            barrier
        );
    }

    // All threads wait for TMA completion
    barrier->arrive_and_wait();
}
```

**Alignment requirements for multi-dimensional TMA:**

| Parameter | Alignment Requirement |
|-----------|----------------------|
| Global memory address | 16-byte aligned |
| Global memory strides | Multiple of 16 bytes |
| Shared memory address | 128-byte aligned |
| Barrier address | 8-byte aligned |

Note the stricter alignment requirements for multi-dimensional TMA compared to 1D TMA: shared memory must be 128-byte aligned (vs 16-byte for 1D), and global strides must be multiples of 16 bytes.

### 9.2.4 Tensor Map Encoding on Device

In some cases, it is useful to modify a tensor map on the device, for example to change the base pointer or adjust dimensions. The pattern for this is:

1. **Create a template tensor map on the host** with placeholder values
2. **Copy the tensor map to device memory** (constant or global memory)
3. **On the device, modify specific fields** using `cuda::ptx::tensormap_replace_*` functions
4. **Issue a fence** to ensure the modifications are visible
5. **Use the modified tensor map** for TMA transfers

```cpp
__global__ void deviceTensorMapModify(CUtensorMap* device_tensor_map,
                                      const float* new_base_ptr,
                                      /* other fields */)
{
    // Only one thread should modify the tensor map
    if (threadIdx.x == 0) {
        // Replace the global address (base pointer)
        cuda::ptx::tensormap_replace_global_address(device_tensor_map, new_base_ptr);

        // Replace global dimensions (example: set dimension 0 to 1024)
        uint64_t new_dim = 1024;
        cuda::ptx::tensormap_replace_global_dim(device_tensor_map, 0, &new_dim);

        // Replace global strides (example: set stride 0 to 4096 bytes)
        uint64_t new_stride = 4096;
        cuda::ptx::tensormap_replace_global_stride(device_tensor_map, 0, &new_stride);

        // Fence to ensure modifications are visible to TMA hardware
        cuda::ptx::fence_proxy_tensormap();
    }
    __syncthreads();

    // Now the modified tensor map can be used for TMA transfers
}
```

**Available tensormap_replace functions:**

```cpp
// Replace the global memory address
cuda::ptx::tensormap_replace_global_address(CUtensorMap* map, const void* addr);

// Replace a specific global dimension
cuda::ptx::tensormap_replace_global_dim(CUtensorMap* map, int dim, const uint64_t* size);

// Replace a specific global stride
cuda::ptx::tensormap_replace_global_stride(CUtensorMap* map, int dim, const uint64_t* stride);

// Replace the box/tile origin offset
cuda::ptx::tensormap_replace_box_origin(CUtensorMap* map, const int32_t* offset);
```

### 9.2.5 Shared Memory Bank Swizzling

TMA supports hardware-level shared memory bank swizzling to resolve bank conflicts when accessing tiled data patterns. Swizzling rearranges the shared memory layout to ensure that common access patterns avoid bank conflicts.

**Four swizzle modes:**

| Swizzle Mode | Swizzle Width | Inner Dimension | Shared Memory Alignment | Global Memory Alignment |
|---|---|---|---|---|
| `CU_TENSOR_MAP_SWIZZLE_128B` | 128 bytes | <= 128 bytes | 128 bytes | 128 bytes |
| `CU_TENSOR_MAP_SWIZZLE_64B` | 64 bytes | <= 64 bytes | 128 bytes | 128 bytes |
| `CU_TENSOR_MAP_SWIZZLE_32B` | 32 bytes | <= 32 bytes | 128 bytes | 128 bytes |
| `CU_TENSOR_MAP_SWIZZLE_NONE` | None (disabled) | N/A | 128 bytes | 16 bytes |

**Choosing a swizzle mode:**

- `SWIZZLE_128B` is best for inner dimensions up to 128 bytes (e.g., 32 float elements)
- `SWIZZLE_64B` is best for inner dimensions up to 64 bytes (e.g., 16 float elements)
- `SWIZZLE_32B` is best for inner dimensions up to 32 bytes (e.g., 8 float elements)
- `SWIZZLE_NONE` should be used when no swizzling is needed or for non-tiled access patterns

```cpp
// Example: Creating a tensor map with 128B swizzle for a matrix
CUtensorMap createSwizzledTensorMap(const half* data, int rows, int cols, int row_stride)
{
    CUtensorMap tmap;
    uint64_t globalDims[2] = {static_cast<uint64_t>(rows), static_cast<uint64_t>(cols)};
    uint64_t globalStrides[2] = {static_cast<uint64_t>(row_stride * sizeof(half)),
                                  static_cast<uint64_t>(sizeof(half))};
    int32_t tileDims[2] = {16, 32}; // 16 rows x 32 cols = 64 bytes (half)
    uint32_t elementStrides[2] = {1, 1};

    cuTensorMapEncodeTiled(
        &tmap,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        2,
        (void*)data,
        globalDims,
        globalStrides,
        tileDims,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,  // Enable 128B swizzle
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_OOB_FILL_NONE
    );
    return tmap;
}
```

When swizzling is enabled, the shared memory buffer receiving TMA data must account for the swizzled layout. Padding may be required to ensure 128-byte alignment of the shared memory buffer:

```cpp
// Ensure shared memory alignment for TMA with swizzling
__shared__ __align__(128) float smem_buffer[2048]; // 128-byte aligned
```

---

## 9.3 STAS -- Store Async to Distributed Shared Memory (CC 9.0+)

The STAS (Store Async) instruction enables asynchronous writes from registers to distributed shared memory (DSM) within a thread-block cluster. Unlike LDGSTS (global-to-shared) and TMA (global-to-shared with tensor descriptors), STAS handles shared-to-shared communication across CTAs in a cluster.

### 9.3.1 Overview

- Copies 4, 8, or 16 bytes from a register to distributed shared memory
- Only available via the `cuda::ptx::st_async` PTX-level API
- Used primarily in producer-consumer ring patterns within thread-block clusters
- Enables fine-grained inter-CTA communication without going through global memory

### 9.3.2 API

```cpp
// Store 4 bytes asynchronously to distributed shared memory
cuda::ptx::st_async(void* dsm_ptr, uint32_t value);

// Store 8 bytes asynchronously to distributed shared memory
cuda::ptx::st_async(void* dsm_ptr, uint64_t value);

// Store 16 bytes asynchronously to distributed shared memory
cuda::ptx::st_async(void* dsm_ptr, uint4 value);
```

### 9.3.3 Producer-Consumer Ring Pattern

The canonical use case for STAS is a ring-based producer-consumer pattern within a thread-block cluster. Each CTA in the cluster produces data for its neighbor and consumes data from its other neighbor.

```cpp
#include <cuda/ptx>
#include <cuda/barrier>

__global__ void __cluster_dims__(4)  // 4 CTAs per cluster
stasRingKernel(float* __restrict__ output,
               const float* __restrict__ input,
               int elements_per_cta)
{
    // Shared memory for local data
    __shared__ float local_smem[1024];

    // Distributed shared memory -- visible to all CTAs in the cluster
    // Each CTA has its own portion of DSM
    __shared__ float dsm_send[256];  // Data this CTA sends to its neighbor
    __shared__ float dsm_recv[256];  // Data this CTA receives from its neighbor

    // Cluster barrier
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();

    int cta_id = cluster.block_rank();
    int num_ctas = cluster.num_blocks();
    int next_cta = (cta_id + 1) % num_ctas;
    int prev_cta = (cta_id + num_ctas - 1) % num_ctas;

    int tid = threadIdx.x;

    // Phase 1: Load input data into local shared memory
    for (int i = tid; i < elements_per_cta; i += blockDim.x) {
        int global_idx = cta_id * elements_per_cta + i;
        local_smem[i] = input[global_idx];
    }
    cg::this_thread_block().sync();

    // Phase 2: Prepare data to send to the next CTA
    for (int i = tid; i < 256; i += blockDim.x) {
        dsm_send[i] = local_smem[i]; // Prepare subset for neighbor
    }
    cg::this_thread_block().sync();

    // Phase 3: Use STAS to asynchronously store data to the next CTA's DSM
    // Get a pointer to the next CTA's receive buffer
    float* remote_recv = cluster.map_shared_rank(dsm_recv, next_cta);

    for (int i = tid; i < 256; i += blockDim.x) {
        // Async store to remote CTA's distributed shared memory
        uint32_t val = __float_as_uint(dsm_send[i]);
        cuda::ptx::st_async(&remote_recv[i], val);
    }

    // Synchronize across the cluster to ensure all STAS operations complete
    cluster.sync();

    // Phase 4: Process received data from the previous CTA
    for (int i = tid; i < 256; i += blockDim.x) {
        local_smem[i] += dsm_recv[i]; // Combine with received data
    }
    cg::this_thread_block().sync();

    // Phase 5: Write results
    for (int i = tid; i < elements_per_cta; i += blockDim.x) {
        int global_idx = cta_id * elements_per_cta + i;
        output[global_idx] = local_smem[i];
    }
}
```

**Key considerations for STAS:**

- STAS targets distributed shared memory, which is only available within thread-block clusters
- The `map_shared_rank` function obtains a pointer to another CTA's shared memory within the cluster
- A cluster-level synchronization (`cluster.sync()`) is required to ensure STAS completion before consuming the data
- STAS is useful for halo exchange, pipeline communication, and data sharing between CTAs in a cluster

---

## 9.4 Work Stealing with Cluster Launch Control (CC 10.0)

Cluster Launch Control, introduced with Compute Capability 10.0 (NVIDIA Blackwell), enables work-stealing patterns where idle CTAs within a cluster can detect and cancel work assigned to other CTAs. This is useful for load-balancing irregular workloads.

### 9.4.1 Overview

Work stealing with Cluster Launch Control follows a five-step pattern:

1. **Declare** -- Set up the cluster launch control state
2. **Initialize barrier** -- Prepare synchronization primitives
3. **Submit request** -- CTAs submit work requests
4. **Synchronize** -- Wait for work assignment resolution
5. **Decode result** -- Determine which work was assigned or stolen

### 9.4.2 API

```cpp
#include <cuda/ptx>

// Step 1-3: Try to cancel a CTA's work assignment
// Returns true if the cancel was successful
bool canceled = cuda::ptx::cluster_launch_control_try_cancel(
    /* cluster dimension info */
);

// Step 4: Query whether a specific CTA's work was canceled
bool is_canceled = cuda::ptx::cluster_launch_control_query_cancel_is_canceled(
    /* target CTA ID */
);

// Step 5: Get the ID of the first CTA that was canceled
uint32_t first_canceled = cuda::ptx::cluster_launch_control_query_cancel_get_first_ctaid_x(
    /* parameters */
);
// Also available for y and z dimensions:
// cuda::ptx::cluster_launch_control_query_cancel_get_first_ctaid_y(...)
// cuda::ptx::cluster_launch_control_query_cancel_get_first_ctaid_z(...)
```

### 9.4.3 Work Stealing Pattern

```cpp
__global__ void __cluster_dims__(8)
workStealingKernel(/* parameters */)
{
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();

    int cta_id = cluster.block_rank();
    int num_ctas = cluster.num_blocks();

    // Each CTA is initially assigned a chunk of work
    int my_work_id = cta_id;

    // Step 1: Declare intent to participate in work stealing

    // Step 2: Initialize the cluster barrier for launch control
    // (barrier initialization code)

    // Step 3: Submit cancel request -- try to steal work from another CTA
    bool i_am_canceled = cuda::ptx::cluster_launch_control_try_cancel(/* ... */);

    // Step 4: Synchronize across the cluster
    cluster.sync();

    // Step 5: Decode the result
    if (i_am_canceled) {
        // This CTA's work was stolen; it can exit or take other action
        return;
    }

    // Check if any other CTA was canceled (meaning we may have stolen their work)
    for (int other = 0; other < num_ctas; ++other) {
        if (other == cta_id) continue;
        bool other_canceled = cuda::ptx::cluster_launch_control_query_cancel_is_canceled(/* other */);
        if (other_canceled) {
            // We can pick up 'other's work in addition to our own
            // Get the CTA ID that was canceled
            uint32_t stolen_cta = cuda::ptx::cluster_launch_control_query_cancel_get_first_ctaid_x(/* ... */);
            // Process stolen_cta's work chunk
        }
    }

    // Process this CTA's originally assigned work
    // ... kernel computation ...
}
```

### 9.4.4 Use Cases

- **Irregular workloads:** When different CTAs have vastly different amounts of work, work stealing allows fast CTAs to take over work from slow ones
- **Load balancing:** Dynamic assignment of work items across CTAs in a cluster
- **Adaptive algorithms:** Where the amount of work per CTA is not known at launch time

**Key constraints:**

- Only available within thread-block clusters (requires `__cluster_dims__` annotation)
- All CTAs in the cluster must participate in the launch control protocol
- The cancel and query operations are collective operations that require cluster-level synchronization
