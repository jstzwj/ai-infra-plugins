# 19. CUDA Device-Callable APIs and Intrinsics Reference

This section covers APIs that can be called from within device code (kernels), including memory barrier primitives, pipeline primitives for async memory operations, Cooperative Groups, CUDA Dynamic Parallelism (CDP), and cooperative kernel launches.

---

## 19.1 Memory Barrier Primitives (`<cuda_awbarrier_primitives.h>`)

Memory barrier primitives provide a flexible mechanism for synchronizing threads within a thread block using arrival-counting barriers. They are declared in `<cuda_awbarrier_primitives.h>`.

### 19.1.1 Types

```c
// Opaque barrier type. Must reside in __shared__ memory.
typedef /* implementation-defined */ __mbarrier_t;

// Opaque token returned by arrive operations.
typedef /* implementation-defined */ __mbarrier_token_t;
```

### 19.1.2 Function Reference

#### Initialization and Management

```c
// Returns the maximum supported arrival count for a barrier.
// The value depends on the compute capability.
uint32_t __mbarrier_maximum_count();

// Initialize a barrier with the given expected arrival count.
// bar must point to __shared__ memory.
// expected_count must be <= __mbarrier_maximum_count().
void __mbarrier_init(__mbarrier_t* bar, uint32_t expected_count);

// Invalidate a barrier so it can be re-initialized.
// Must be called before reusing a barrier that has completed.
void __mbarrier_init(__mbarrier_t* bar, uint32_t expected_count); // re-init is OK after invalidation
void __mbarrier_invalidate(__mbarrier_t* bar);
```

#### Arrival

```c
// Arrive at the barrier and return a token.
// The arrival increments the barrier's count.
// Does NOT block the calling thread.
__mbarrier_token_t __mbarrier_arrive(__mbarrier_t* bar);

// Arrive at the barrier and immediately drop (decrement expected count).
// Useful when a thread is done contributing and should not participate
// in future phases of the barrier.
__mbarrier_token_t __mbarrier_arrive_and_drop(__mbarrier_t* bar);
```

#### Waiting

```c
// Non-blocking test: returns true if all expected threads have arrived
// (i.e., the barrier has been satisfied for the phase associated with token).
bool __mbarrier_test_wait(__mbarrier_t* bar, __mbarrier_token_t token);

// Wait using phase parity instead of a specific token.
// phase_parity should alternate between true and false for successive phases.
bool __mbarrier_test_wait_parity(__mbarrier_t* bar, bool phase_parity);

// Try-wait with a bounded sleep: spins for at most max_sleep_ns,
// then returns false if the barrier has not been satisfied.
// Useful to avoid indefinite spinning.
bool __mbarrier_try_wait(__mbarrier_t* bar, __mbarrier_token_t token, uint32_t max_sleep_ns);

// Try-wait with phase parity and bounded sleep.
bool __mbarrier_try_wait_parity(__mbarrier_t* bar, bool phase_parity, uint32_t max_sleep_ns);
```

### 19.1.3 Usage Pattern

```cpp
#include <cuda_awbarrier_primitives.h>

__global__ void barrier_example(float* output, int n) {
    __shared__ __mbarrier_t barrier;

    // Thread 0 initializes the barrier
    if (threadIdx.x == 0) {
        __mbarrier_init(&barrier, blockDim.x);
    }
    __syncthreads(); // ensure barrier is initialized before use

    // Each thread does its work
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] *= 2.0f;
    }

    // Each thread arrives at the barrier
    __mbarrier_token_t token = __mbarrier_arrive(&barrier);

    // Spin-wait until all threads have arrived
    while (!__mbarrier_test_wait(&barrier, token)) {
        // busy-wait
    }

    // All threads have completed their work -- safe to proceed
    // Use output[] data written by other threads...

    // For a second phase, use arrive_and_drop if a thread is done:
    // __mbarrier_arrive_and_drop(&barrier);
}
```

### 19.1.4 Important Constraints

- `bar` must always point to **shared memory** (`__shared__`). Pointing to global, local, or constant memory results in undefined behavior.
- `expected_count` must not exceed `__mbarrier_maximum_count()`.
- After a barrier phase completes, you must either wait for the next phase or call `__mbarrier_invalidate()` before reinitializing.
- `__mbarrier_arrive_and_drop()` permanently reduces the expected count by 1. This is useful for producer-consumer patterns where producers finish early.

### 19.1.5 Producer-Consumer Pattern

```cpp
__global__ void producer_consumer(float* data, int n) {
    __shared__ __mbarrier_t barrier;
    __shared__ float shared_buf[256];

    constexpr int PRODUCER_COUNT = 128;
    constexpr int CONSUMER_COUNT = 128;

    if (threadIdx.x == 0) {
        __mbarrier_init(&barrier, PRODUCER_COUNT);
    }
    __syncthreads();

    if (threadIdx.x < PRODUCER_COUNT) {
        // Producer: load data into shared memory
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            shared_buf[threadIdx.x] = data[idx];
        }
        // Arrive and drop -- consumers will use a separate barrier
        __mbarrier_arrive_and_drop(&barrier);
    } else {
        // Consumer: wait for producers
        __mbarrier_token_t token = __mbarrier_arrive(&barrier);
        while (!__mbarrier_test_wait(&barrier, token)) {}
        // Now shared_buf is populated -- consume it
    }
}
```

---

## 19.2 Pipeline Primitives (`<cuda_pipeline.h>`)

Pipeline primitives enable asynchronous memory copies from global memory to shared memory, overlapping data movement with computation. They are declared in `<cuda_pipeline.h>`.

### 19.2.1 Function Reference

```c
// Initiate an asynchronous memory copy from global to shared memory.
// dst_shared: destination address in shared memory
// src_global: source address in global memory
// size_and_align: number of bytes to copy (must be 4, 8, or 16)
// zfill: number of trailing bytes to zero-fill (default 0)
//
// Constraints:
//   - dst_shared must be in shared memory
//   - src_global must be in global memory
//   - size_and_align must be 4, 8, or 16
//   - size_and_align must equal the alignment of both dst_shared and src_global
void __pipeline_memcpy_async(void* dst_shared,
                              const void* src_global,
                              size_t size_and_align,
                              size_t zfill = 0);

// Commit all pending async copies initiated by this thread since
// the last __pipeline_commit() or __pipeline_wait_prior().
void __pipeline_commit();

// Wait for the N-th most recent group of committed async copies to complete.
// N=0 means wait for ALL committed copies.
// N=1 means wait for all but the most recent commit group.
void __pipeline_wait_prior(size_t N);

// Arrive on an mbarrier when all pending async copies for this thread complete.
// The arrival is counted as a single arrival regardless of the number of copies.
void __pipeline_arrive_on(__mbarrier_t* bar);
```

### 19.2.2 Multi-Stage Pipeline Pattern

```cpp
#include <cuda_pipeline.h>

__global__ void pipeline_example(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int n)
{
    // Double-buffered shared memory for pipeline
    __shared__ float buffer[2][128];

    // Compute how many elements this block processes
    int block_offset = blockIdx.x * 128;
    int tid = threadIdx.x;

    // Stage 0: issue first async copy
    if (block_offset + tid < n) {
        __pipeline_memcpy_async(&buffer[0][tid],
                                 &input[block_offset + tid],
                                 sizeof(float));
    }
    __pipeline_commit();

    int stages = (n - block_offset + 127) / 128;
    stages = min(stages, 2); // we have 2 buffer slots

    for (int stage = 1; stage < stages; ++stage) {
        int src_stage = stage - 1;
        int buf_idx = stage % 2;
        int offset = block_offset + stage * 128;

        // Issue async copy for NEXT stage
        if (offset + tid < n) {
            __pipeline_memcpy_async(&buffer[buf_idx][tid],
                                     &input[offset + tid],
                                     sizeof(float));
        }
        __pipeline_commit();

        // Wait for the PREVIOUS stage's copy to complete
        __pipeline_wait_prior(1);

        // Compute on data from previous stage
        if (block_offset + src_stage * 128 + tid < n) {
            buffer[src_stage % 2][tid] *= 2.0f;
            output[block_offset + src_stage * 128 + tid] = buffer[src_stage % 2][tid];
        }
    }

    // Wait for last copy and process remaining data
    __pipeline_wait_prior(0);
    int last_stage = stages - 1;
    if (block_offset + last_stage * 128 + tid < n) {
        buffer[last_stage % 2][tid] *= 2.0f;
        output[block_offset + last_stage * 128 + tid] = buffer[last_stage % 2][tid];
    }
}
```

### 19.2.3 Pipeline with mbarrier Integration

```cpp
__global__ void pipeline_mbarrier_example(const float* input, float* output, int n) {
    __shared__ __mbarrier_t barrier;
    __shared__ float shared_buf[128];

    if (threadIdx.x == 0) {
        __mbarrier_init(&barrier, blockDim.x);
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initiate async copy and arrive on the barrier when done
    if (idx < n) {
        __pipeline_memcpy_async(&shared_buf[threadIdx.x],
                                 &input[idx],
                                 sizeof(float));
    }
    __pipeline_arrive_on(&barrier);

    // Wait for all copies to complete
    __mbarrier_token_t token = __mbarrier_arrive(&barrier);
    while (!__mbarrier_test_wait(&barrier, token)) {}

    // Safe to read shared_buf
    if (idx < n) {
        output[idx] = shared_buf[threadIdx.x] * 2.0f;
    }
}
```

### 19.2.4 Constraints and Notes

- `dst_shared` must reside in shared memory; `src_global` must reside in global memory.
- `size_and_align` must be 4, 8, or 16 bytes. It must equal the alignment of both source and destination pointers.
- The `zfill` parameter specifies how many of the last bytes of the transfer should be zero-initialized instead of copied. This is useful for padding.
- `__pipeline_memcpy_async` only initiates the transfer; it does not block. Use `__pipeline_wait_prior` or `__pipeline_arrive_on` to synchronize.
- On architectures without hardware async copy support (pre-CC 8.0), these primitives are emulated in software.

---

## 19.3 Cooperative Groups API Reference

Cooperative Groups (CG) provide a modern, composable mechanism for thread cooperation in CUDA. The namespace is `cuda::cooperative_groups`, often aliased as `cg`.

```cpp
#include <cooperative_groups.h>
namespace cg = cuda::cooperative_groups;
```

### 19.3.1 thread_block

Represents all threads in a CUDA thread block. This is the fundamental cooperative group.

```cpp
// Obtain the current thread block group
cg::thread_block g = cg::this_thread_block();

// Synchronization: equivalent to __syncthreads()
g.sync();

// Thread identification
unsigned int rank = g.thread_rank();    // thread rank in [0, num_threads)
dim3 block_idx = g.group_index();       // block index within the grid (dim3)
dim3 thread_idx = g.thread_index();     // thread index within the block (dim3)
dim3 dims = g.dim_threads();            // block dimensions (dim3)
unsigned int count = g.num_threads();   // total threads in block (int)

// Barrier with arrival/wait pattern (more flexible than sync())
auto token = g.barrier_arrive();        // arrive: returns arrival_token
// ... do independent work ...
g.barrier_wait(std::move(token));       // wait: blocks until all arrive
```

```cpp
__global__ void thread_block_example(int* data, int n) {
    cg::thread_block block = cg::this_thread_block();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes its element
    if (idx < n) {
        data[idx] += 1;
    }

    // Synchronize all threads in the block
    block.sync();

    // Now safe to read data written by any thread in the block
    if (idx < n && idx > 0) {
        data[idx] += data[idx - 1]; // use neighbor's result
    }
}
```

### 19.3.2 cluster_group (Compute Capability 9.0+)

Thread blocks can be grouped into clusters on GPUs with compute capability 9.0 and above. Clusters enable threads in different blocks to cooperate and access each other's shared memory.

```cpp
// Obtain the current cluster group
auto cluster = cg::this_cluster();

// Synchronize all threads in the cluster
cluster.sync();

// Block identification within cluster
unsigned int block_rank = cluster.block_rank();    // rank of this block [0, num_blocks)
unsigned int num_blocks = cluster.num_blocks();     // total blocks in cluster
dim3 dim_blocks = cluster.dim_blocks();             // cluster dimensions (dim3)

// Cross-block shared memory access
// query_shared_rank: get the rank of the block that owns the shared memory at addr
unsigned int owner_rank = cluster.query_shared_rank(addr);

// map_shared_rank: get a pointer in the shared memory of the block at given rank
// that corresponds to the same shared memory offset as addr in this block
void* remote_ptr = cluster.map_shared_rank(addr, rank);

// Get the multi-dimensional block index within the cluster
dim3 block_idx = cluster.block_index();
```

```cpp
__global__ void __cluster_dims__(2, 1, 1)  // 2 blocks per cluster
cluster_example(float* output, int n) {
    auto cluster = cg::this_cluster();
    __shared__ float shared_data[128];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each block writes to its own shared memory
    if (threadIdx.x < 128 && idx < n) {
        shared_data[threadIdx.x] = output[idx];
    }

    // Synchronize across the entire cluster
    cluster.sync();

    // Access neighboring block's shared memory
    unsigned int neighbor_rank = (cluster.block_rank() + 1) % cluster.num_blocks();
    float* neighbor_shared = (float*)cluster.map_shared_rank(shared_data, neighbor_rank);

    // Read from neighbor's shared memory
    if (threadIdx.x < 128 && idx < n) {
        output[idx] += neighbor_shared[threadIdx.x];
    }
}
```

**Launch with clusters**:

```cpp
// Kernel launch with cluster configuration
cudaLaunchConfig_t config = {0};
config.gridDim = gridDim;
config.blockDim = blockDim;

cudaLaunchAttribute cluster_attr;
cluster_attr.id = cudaLaunchAttributeClusterDimension;
cluster_attr.val.clusterDim.x = 2;
cluster_attr.val.clusterDim.y = 1;
cluster_attr.val.clusterDim.z = 1;

config.attrs = &cluster_attr;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, cluster_example, output, n);
```

### 19.3.3 grid_group

Represents all threads in the entire grid. Requires cooperative kernel launch.

```cpp
// Obtain the grid group
auto grid = cg::this_grid();

// Check if cooperative launch is being used (valid)
bool valid = grid.is_valid();

// Synchronize ALL threads in the grid
// Only valid if launched with cudaLaunchCooperativeKernel
grid.sync();

// Query properties
unsigned int num_threads = grid.num_threads();
unsigned int thread_rank = grid.thread_rank();
dim3 grid_dim = grid.dim_grid();
dim3 block_dim = grid.dim_block();
```

```cpp
__global__ void grid_sync_example(int* data, int n) {
    auto grid = cg::this_grid();

    if (!grid.is_valid()) {
        // Not launched cooperatively -- grid.sync() would be undefined
        return;
    }

    int idx = grid.thread_rank();

    // Phase 1: each thread processes its data
    if (idx < n) {
        data[idx] *= 2;
    }

    // Synchronize across the entire grid
    grid.sync();

    // Phase 2: now all data is updated, safe to read any element
    if (idx < n) {
        int neighbor = data[(idx + 1) % n]; // read data from another thread
        data[idx] += neighbor;
    }
}
```

### 19.3.4 thread_block_tile<Size>

A tiled partition of a thread block into groups of `Size` threads, where `Size` is a power of 2 (1, 2, 4, 8, 16, or 32). These groups support warp-level primitives like shuffle and vote.

```cpp
// Partition the thread block into tiles of the given size
auto tile = cg::tiled_partition<32>(cg::this_thread_block());

// For compile-time unknown size (must still be power-of-2 at runtime):
// auto tile = cg::tiled_partition(cg::this_thread_block(), tile_size);

// Synchronize within the tile
tile.sync();

// Warp shuffle operations
unsigned int rank = tile.thread_rank();   // rank within tile [0, Size)
unsigned int size = tile.size();          // number of threads in tile

// Direct shuffle: read var from src_rank
T result = tile.shfl(var, src_rank);

// Shuffle up: read from (rank - delta), clamped
T result = tile.shfl_up(var, delta);

// Shuffle down: read from (rank + delta), clamped
T result = tile.shfl_down(var, delta);

// Shuffle xor: read from (rank ^ lane_mask)
T result = tile.shfl_xor(var, lane_mask);

// Vote operations
int any_result = tile.any(predicate);    // true if any thread's predicate is true
int all_result = tile.all(predicate);    // true if all threads' predicates are true
unsigned int ballot = tile.ballot(predicate); // bitmask of predicates across tile
```

```cpp
__global__ void tile_example(float* data, int n) {
    auto block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (idx < n) {
        val = data[idx];
    }

    // Parallel reduction within each warp tile
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
        val += tile32.shfl_down(val, offset);
    }

    // Thread 0 of each tile has the partial sum
    if (tile32.thread_rank() == 0) {
        // Could use atomics or further reduction
        atomicAdd(&data[0], val);
    }
}
```

```cpp
__global__ void vote_example(int* results) {
    auto tile = cg::tiled_partition<32>(cg::this_thread_block());

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool is_even = (tid % 2 == 0);

    // Vote: check if any thread in the tile has an even tid
    int any_even = tile.any(is_even);
    int all_even = tile.all(is_even);
    unsigned int ballot = tile.ballot(is_even);

    if (tile.thread_rank() == 0) {
        results[0] = any_even;
        results[1] = all_even;
        results[2] = (int)ballot;
    }
}
```

### 19.3.5 coalesced_group

A group of active (converged) threads. Created via `coalesced_threads()`, it includes only threads that are currently active in the warp.

```cpp
// Create a group of currently active threads
auto active = cg::coalesced_threads();

// Synchronize within the coalesced group
active.sync();

// Same shuffle, vote, and match operations as thread_block_tile
unsigned int rank = active.thread_rank();
unsigned int size = active.size();

T result = active.shfl(var, src_rank);
T result_up = active.shfl_up(var, delta);
T result_down = active.shfl_down(var, delta);
T result_xor = active.shfl_xor(var, lane_mask);

int any_result = active.any(predicate);
int all_result = active.all(predicate);
unsigned int ballot_result = active.ballot(predicate);

// Match operations (compute capability 7.0+)
unsigned int match_any = active.match_any(value);   // bitmask of threads with same value
unsigned int match_all = active.match_all(value);   // bitmask if all threads have same value, else 0
```

```cpp
__global__ void coalesced_example(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only threads with valid indices participate
    if (idx < n) {
        auto active = cg::coalesced_threads();

        // Only the active threads participate in the shuffle
        float val = data[idx];
        float neighbor_val = active.shfl_xor(val, 1);

        data[idx] = val + neighbor_val;
        active.sync();
    }
    // Threads that did not enter the if-block are not part of the group
}
```

### 19.3.6 CG Synchronization Summary

| Group Type | Scope | Sync Method | Key Operations |
|---|---|---|---|
| `thread_block` | All threads in block | `.sync()` | Barrier, arrival/wait |
| `cluster_group` | All threads in cluster | `.sync()` | Cross-block shared mem |
| `grid_group` | All threads in grid | `.sync()` | Requires cooperative launch |
| `thread_block_tile<N>` | N threads (power of 2) | `.sync()` | Shuffle, vote, ballot |
| `coalesced_group` | Active threads | `.sync()` | Shuffle, vote, ballot, match |

---

## 19.4 Device Runtime (CUDA Dynamic Parallelism - CDP)

CUDA Dynamic Parallelism (CDP) allows kernels to launch new kernels from device code. Starting with CUDA 11.0+, CDP is supported on compute capability 3.5 and above. CDP v2 (improved) is available from compute capability 7.0.

### 19.4.1 Device-Side Memory Management

```cpp
// Allocate global memory from device code
// Returns cudaSuccess on success. Memory is accessible by all threads in the
// device-launched grid and its child grids.
cudaError_t cudaMalloc(void** ptr, size_t size);

// Free memory allocated by device-side cudaMalloc
cudaError_t cudaFree(void* ptr);
```

**Important distinction from host-side API**: Device-side `cudaMalloc` / `cudaFree` have distinct semantics:
- Device `cudaMalloc` allocates from the same global memory pool but the allocation is managed by the device runtime.
- Device `cudaFree` must free memory allocated by device-side `cudaMalloc` (or from a parent grid).
- Memory allocated by device `cudaMalloc` is visible to the allocating thread's grid and any child grids it launches.

```cpp
__device__ void device_malloc_example() {
    float* temp;
    cudaError_t err = cudaMalloc(&temp, 256 * sizeof(float));
    if (err != cudaSuccess) {
        // allocation failed
        return;
    }

    // Use temp...
    temp[threadIdx.x] = 1.0f;

    cudaFree(temp);
}
```

### 19.4.2 Device-Side Memory Operations

```cpp
// Asynchronous memory copy (device-to-device only)
// dst and src must both be in device global memory
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                             cudaMemcpyKind kind, cudaStream_t stream = 0);

// Kind must be cudaMemcpyDeviceToDevice for device-side calls

// Asynchronous memory set
cudaError_t cudaMemsetAsync(void* ptr, int value, size_t count,
                             cudaStream_t stream = 0);
```

```cpp
__global__ void memcpy_kernel(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        // Copy within device memory
        cudaMemcpyAsync(dst, src, n * sizeof(float),
                        cudaMemcpyDeviceToDevice);
    }
}
```

### 19.4.3 Device-Side Stream and Event Management

```cpp
// Create a device-side stream
// Must use cudaStreamNonBlocking for device-created streams
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream,
                                       unsigned int flags);
// flags must be cudaStreamNonBlocking for device-side streams

// Destroy a device-side stream
cudaError_t cudaStreamDestroy(cudaStream_t stream);

// Synchronize a device-side stream
cudaError_t cudaStreamSynchronize(cudaStream_t stream);

// Create a device-side event (must disable timing)
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event,
                                      unsigned int flags);
// flags should include cudaEventDisableTiming

// Record / wait on events
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                 unsigned int flags = 0);

// Destroy an event
cudaError_t cudaEventDestroy(cudaEvent_t event);
```

```cpp
__global__ void child_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

__global__ void parent_kernel(float* data, int n) {
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // Launch child kernel asynchronously
    child_kernel<<<(n + 255) / 256, 256, 0, stream>>>(data, n);

    // Wait for child to complete
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
}
```

### 19.4.4 Device-Side Error Handling and Query

```cpp
// Get the last error from a runtime call (resets to cudaSuccess)
cudaError_t cudaGetLastError();

// Get error string for a cudaError_t
const char* cudaGetErrorString(cudaError_t error);

// Query function attributes (usable in device code)
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr,
                                   const void* func);

// Query occupancy (usable in device code)
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize);
```

```cpp
__global__ void error_handling_example(float* data, int n) {
    float* temp;
    cudaError_t err = cudaMalloc(&temp, n * sizeof(float));
    if (err != cudaSuccess) {
        // Check last error
        cudaError_t last = cudaGetLastError();
        // Get description
        const char* desc = cudaGetErrorString(err);
        // desc might be "out of memory"
        return;
    }

    // Query child kernel occupancy
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)child_kernel);

    int max_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks, (const void*)child_kernel, 256, 0);

    cudaFree(temp);
}
```

### 19.4.5 Named Streams for CDP

CDP provides two special named streams that control the launch relationship between parent and child grids:

```cpp
// cudaStreamTailLaunch:
// Child grid begins execution AFTER the parent grid has completed.
// All blocks of the parent grid must finish before the child starts.
cudaStream_t cudaStreamTailLaunch;

// cudaStreamFireAndForget:
// Child grid begins execution as soon as possible, with no dependency
// on the parent grid's completion. No guarantee of ordering.
cudaStream_t cudaStreamFireAndForget;
```

```cpp
__global__ void tail_launch_example(float* data, int n) {
    // This child kernel will only start after the ENTIRE parent grid finishes
    child_kernel<<<1, 256, 0, cudaStreamTailLaunch>>>(data, n);
}

__global__ void fire_and_forget_example(float* data, int n) {
    // This child kernel starts immediately, no dependency on parent
    child_kernel<<<1, 256, 0, cudaStreamFireAndForget>>>(data, n);
}
```

**CDP v2 (Compute Capability 7.0+)** improvements:
- More efficient memory management
- Better synchronization semantics
- Reduced overhead for device-side kernel launches
- Support for `cudaLaunchKernel` and `<<<...>>>` syntax

### 19.4.6 CDP Limitations

- Device-side streams must use `cudaStreamNonBlocking`.
- Device-side events must use `cudaEventDisableTiming`.
- `cudaMemcpy` with `cudaMemcpyHostToDevice` or `cudaMemcpyDeviceToHost` is not supported in device code. Only `cudaMemcpyDeviceToDevice`.
- The device runtime maintains a limited pool of memory for launch buffering. Very deep nesting or many concurrent launches can exhaust this pool.
- `printf()` from device code has a limited buffer.
- Texture and surface references have limited support in CDP.

---

## 19.5 Grid Synchronization (Cooperative Launch)

Cooperative kernel launches enable synchronization across all threads in a grid using `cg::this_grid().sync()`. This requires special launch APIs.

### 19.5.1 Checking Device Support

```cpp
int device;
cudaGetDevice(&device);

int supports_cooperative = 0;
cudaDeviceGetAttribute(&supports_cooperative,
                        cudaDevAttrCooperativeLaunch,
                        device);

if (!supports_cooperative) {
    printf("Cooperative launch not supported on this device.\n");
    return;
}
```

### 19.5.2 Cooperative Kernel Launch (Runtime API)

```cpp
// Launch a kernel cooperatively using the runtime API
cudaError_t cudaLaunchCooperativeKernel(
    const void* func,       // kernel function pointer
    dim3 gridDim,           // grid dimensions
    dim3 blockDim,          // block dimensions
    void** args,            // kernel arguments
    size_t sharedMem = 0,   // dynamic shared memory per block
    cudaStream_t stream = 0 // stream
);
```

```cpp
__global__ void cooperative_kernel(int* data, int n) {
    auto grid = cg::this_grid();

    int idx = grid.thread_rank();

    // Phase 1: local computation
    if (idx < n) {
        data[idx] += 1;
    }

    // Synchronize ALL threads across ALL blocks in the grid
    grid.sync();

    // Phase 2: use results from phase 1
    if (idx < n) {
        int neighbor_idx = (idx + 1) % n;
        data[idx] += data[neighbor_idx];
    }
}

void launch_cooperative(int* d_data, int n) {
    int device;
    cudaGetDevice(&device);

    int supports_coop;
    cudaDeviceGetAttribute(&supports_coop,
                            cudaDevAttrCooperativeLaunch, device);
    if (!supports_coop) {
        fprintf(stderr, "Cooperative launch not supported\n");
        return;
    }

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);

    // For cooperative launches, grid size must not exceed max blocks per SM
    int blockSize = 256;
    int gridSize = numSMs; // typical: one block per SM

    void* args[] = { &d_data, &n };
    cudaLaunchCooperativeKernel(
        (const void*)cooperative_kernel,
        gridSize,
        blockSize,
        args,
        0,
        0
    );
}
```

### 19.5.3 Cooperative Kernel Launch (Driver API)

```c
CUresult cuLaunchCooperativeKernel(
    CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams
);
```

### 19.5.4 Multi-Device Cooperative Launch

For synchronizing across multiple GPUs in a cooperative launch:

```c
CUresult cuLaunchCooperativeKernelMultiDevice(
    CUDA_LAUNCH_PARAMS* launchParamsList,
    unsigned int numDevices,
    unsigned int flags  // 0 or CUDA_LAUNCH_PARAM_BUFFER_POINTER etc.
);
```

```cpp
// Multi-device cooperative launch example (driver API)
int num_gpus;
cudaGetDeviceCount(&num_gpus);

CUcontext contexts[MAX_GPUS];
CUfunction functions[MAX_GPUS];
CUDA_LAUNCH_PARAMS launch_params[MAX_GPUS];

// ... initialize contexts, load functions for each GPU ...

for (int i = 0; i < num_gpus; i++) {
    launch_params[i].function = functions[i];
    launch_params[i].gridDimX = gridSize;
    launch_params[i].gridDimY = 1;
    launch_params[i].gridDimZ = 1;
    launch_params[i].blockDimX = blockSize;
    launch_params[i].blockDimY = 1;
    launch_params[i].blockDimZ = 1;
    launch_params[i].sharedMemBytes = 0;
    launch_params[i].hStream = 0;
    launch_params[i].kernelParams = args;
}

cuLaunchCooperativeKernelMultiDevice(launch_params, num_gpus, 0);
```

### 19.5.5 Cooperative Launch Constraints

- The total number of blocks in the grid must be small enough that all blocks can run concurrently on the device. Specifically: `gridSize <= numSMs * maxBlocksPerSM`.
- Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to determine the maximum number of blocks per SM.
- All blocks in the grid must be able to reside on the device simultaneously for `grid.sync()` to work correctly.
- The kernel must not use more dynamic shared memory than would prevent full occupancy.
- `grid.sync()` in a non-cooperatively launched kernel results in undefined behavior.

### 19.5.6 Determining Grid Size for Cooperative Launches

```cpp
int calculate_cooperative_grid_size(CUfunction kernel, int blockSize,
                                     size_t dynamicSharedMem = 0) {
    int device;
    cudaGetDevice(&device);

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);

    int maxBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM, kernel, blockSize, dynamicSharedMem);

    // Maximum grid size for cooperative launch
    return numSMs * maxBlocksPerSM;
}
```

---

## 19.6 Device-Side Syncthreads and Explicit Synchronization

### 19.6.1 __syncthreads() and Variants

```cpp
// Full barrier: all threads in the block must reach this point
void __syncthreads();

// Conditional barrier with memory fence
void __syncthreads_and(int predicate);   // returns non-zero iff ALL predicates are non-zero
void __syncthreads_or(int predicate);    // returns non-zero iff ANY predicate is non-zero
void __syncthreads_count(int predicate); // returns count of non-zero predicates

// Thread count evaluation (no barrier)
int __syncthreads_and(int predicate);    // evaluates AND across block
int __syncthreads_or(int predicate);     // evaluates OR across block
int __syncthreads_count(int predicate);  // counts true predicates across block
```

### 19.6.2 Memory Fences

```cpp
// Fence within thread block
void __threadfence_block();  // ensures all writes to shared/global memory
                              // are visible to all threads in the block

// Fence within device (all blocks on the GPU)
void __threadfence();        // ensures all writes to global memory
                              // are visible to all threads on the device

// Fence across the system (host + all devices)
void __threadfence_system(); // ensures all writes are visible across
                              // the entire system (host + all GPUs)
```

```cpp
__device__ int count = 0;
__device__ int results[N];

__global__ void fence_example(int* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Write to global memory
    results[idx] = idx * 2;

    // Ensure the write is visible device-wide before incrementing count
    __threadfence();
    atomicAdd(&count, 1);

    // Another thread might read results[idx] after seeing count increment
    // __threadfence() above guarantees the write to results[idx] is visible
}
```
