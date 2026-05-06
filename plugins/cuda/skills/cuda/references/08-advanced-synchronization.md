# 8. Advanced Synchronization

This chapter covers advanced synchronization mechanisms available in CUDA beyond the basic `__syncthreads()` and Cooperative Groups `sync()`. These mechanisms enable finer-grained control over thread coordination, overlapping computation with communication, and enabling new execution models on modern GPU architectures.

---

## 8.1 Asynchronous Barriers

Asynchronous barriers separate the arrival and waiting phases of a barrier, enabling more flexible synchronization patterns. They are available starting from CC 7.0 and are hardware-accelerated on CC 8.0+.

### Header

```cpp
#include <cuda/barrier>
```

### Overview

Traditional barriers (`__syncthreads()`, `cg::sync()`) are monolithic: every thread arrives and blocks until all threads arrive. Asynchronous barriers decouple these phases:

- **Arrive**: A thread signals it has reached the barrier. This is non-blocking.
- **Wait**: A thread blocks until all expected threads have arrived (and the barrier phase has completed).

This separation enables threads to perform independent work between arriving and waiting, improving utilization.

### Declaration and Initialization

```cpp
#include <cuda/barrier>

// Declare a barrier with block scope
__shared__ cuda::barrier<cuda::thread_scope_block> barrier;

__global__ void kernel() {
    // Initialize the barrier (must be done by a SINGLE thread)
    if (threadIdx.x == 0) {
        init(&barrier, blockDim.x); // Expected arrival count = block size
    }
    __syncthreads(); // Ensure barrier is initialized before use

    // ... use barrier ...
}
```

**Scopes**:

| Scope | Description |
|---|---|
| `cuda::thread_scope_block` | Visible to all threads in the block |
| `cuda::thread_scope_device` | Visible to all threads on the device |
| `cuda::thread_scope_system` | Visible across the system (host + device) |

### Basic Arrive-Wait Pattern

```cpp
__global__ void arrive_wait_example(float* output, const float* input, int n) {
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    __shared__ float shared_data[256];

    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
    }
    __syncthreads();

    // Each thread loads data into shared memory
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        shared_data[threadIdx.x] = input[idx];
    }

    // Arrive at the barrier -- non-blocking
    auto token = bar.arrive();

    // Perform independent work while waiting
    float local_result = 0.0f;
    // ... some computation that doesn't depend on shared_data ...

    // Wait for all threads to arrive (blocking)
    bar.wait(std::move(token));

    // Now shared_data is fully populated by all threads
    float neighbor = shared_data[(threadIdx.x + 1) % blockDim.x];
    if (idx < n) {
        output[idx] = shared_data[threadIdx.x] + neighbor + local_result;
    }
}
```

### Barrier Phase

The barrier operates in phases. Understanding phases is essential for correct usage:

1. The barrier is initialized with an **expected count** (e.g., the number of threads).
2. As threads call `arrive()`, an internal counter decrements.
3. When the counter reaches zero, the phase is **complete**.
4. The barrier **auto-resets**: the counter returns to the expected count, and the phase flips (even/odd).
5. Tokens carry a phase value. `wait(token)` blocks until the phase associated with the token completes.

This means a single barrier object can be reused across multiple synchronization points without re-initialization.

```cpp
__shared__ cuda::barrier<cuda::thread_scope_block> bar;

// Initialization (once)
if (threadIdx.x == 0) init(&bar, blockDim.x);
__syncthreads();

// Phase 1
shared[threadIdx.x] = compute_phase1();
auto token1 = bar.arrive();
// ... independent work ...
bar.wait(std::move(token1)); // Waits for phase 1

// Phase 2 (barrier auto-reset, now in next phase)
shared[threadIdx.x] = compute_phase2();
auto token2 = bar.arrive();
// ... independent work ...
bar.wait(std::move(token2)); // Waits for phase 2
```

### Warp Entanglement

When multiple threads within the same warp arrive at a barrier, the hardware can optimize the arrival process. On CC 8.0+, **warp entanglement** allows converged threads (threads executing the same instruction) to be counted together, reducing the number of individual arrival operations.

To use warp entanglement effectively:

1. **Arrive-on by converged threads**: If multiple threads in a warp arrive simultaneously, the hardware may treat them as a single arrival. This is most effective when threads are converged.
2. **Re-converge before arrive**: Use `__syncwarp()` to re-converge divergent threads before arriving, ensuring maximum warp entanglement.

```cpp
// Ensure warp convergence before arriving
__syncwarp();
auto token = bar.arrive();
```

**Implications**:
- Warp entanglement reduces the number of individual atomic updates to the barrier counter.
- On CC 8.0+, the hardware counts arriving warps rather than individual threads.
- The expected count for the barrier should match the actual arrival pattern (e.g., number of warps if using warp-level arrival).

### Explicit Phase Tracking

For advanced use cases, you can manually track barrier phases using parity-based waiting:

```cpp
// Direct PTX-level barrier operations
// mbarrier_try_wait_parity: wait for a specific parity (0 or 1)
// The parity flips each phase

// Use the barrier's internal phase management
bool parity = 0; // Start at even parity

// After arrive:
parity = 1 - parity; // Flip parity for next phase

// Low-level PTX (CC 8.0+):
// mbarrier_try_wait_parity(bar, parity);
```

### Early Exit with `arrive_and_drop()`

Threads that need to exit the barrier early (e.g., after completing their work) can use `arrive_and_drop()`. This signals arrival and removes the thread from future expected counts.

```cpp
__shared__ cuda::barrier<cuda::thread_scope_block> bar;

if (threadIdx.x == 0) init(&bar, blockDim.x);
__syncthreads();

if (should_exit_early(threadIdx.x)) {
    // Arrive and remove self from future expected counts
    bar.arrive_and_drop();
    return; // Thread exits, won't participate in future phases
}

// Normal threads continue
shared[threadIdx.x] = data[threadIdx.x];
auto token = bar.arrive();
// ... work ...
bar.wait(std::move(token));

// Subsequent barrier uses will expect blockDim.x - num_exited threads
```

**Important**: `arrive_and_drop()` permanently modifies the barrier's expected count. Subsequent phases will expect fewer arrivals. This is useful for:
- Reducing the number of participating threads across iterations.
- Implementing early termination patterns.

### Completion Function

Barriers can be configured with a **completion function** that executes once per phase after the last thread arrives. This is useful for triggering a computation or notification after all threads have synchronized.

```cpp
// Custom completion function type
struct ClearSharedMemory {
    __device__ void operator()() {
        // This runs once after all threads arrive
        // Only one thread executes this (the last to arrive)
        memset(shared_buffer, 0, sizeof(shared_buffer));
    }
};

// Barrier with completion function
__shared__ cuda::barrier<cuda::thread_scope_block, ClearSharedMemory> bar;

if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
}
__syncthreads();
```

The completion function signature must be `void operator()()` and the type must be trivially constructible.

### Tracking Asynchronous Memory Operations

On CC 9.0+ (Hopper), barriers can track the completion of asynchronous memory operations (e.g., TMA copies). This allows the barrier to wait for both thread arrivals and pending memory transfers.

```cpp
#include <cuda/barrier>

// The barrier can count expected arrivals from both threads and async transactions
__shared__ cuda::barrier<cuda::thread_scope_block> bar;

if (threadIdx.x == 0) {
    init(&bar, blockDim.x); // Initialize with thread count
}
__syncthreads();

// Initiate async copy (e.g., TMA)
// The transaction count is automatically tracked by the barrier

// For explicit transaction tracking (CC 9.0+):
cuda::device::barrier_arrive_tx(bar, count, transaction_count);
```

**`barrier_arrive_tx`** (CC 9.0+):
- Arrives at the barrier on behalf of `count` threads AND registers `transaction_count` pending transactions.
- This is used when initiating asynchronous bulk copies (TMA operations).
- The barrier phase does not complete until all threads have arrived AND all tracked transactions have completed.

```cpp
// Example: Async copy with barrier tracking (conceptual)
__global__ void async_copy_kernel(float* output, const float* input, int n) {
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    __shared__ float shared[256];

    if (threadIdx.x == 0) {
        init(&bar, 1); // Only "1" logical arrival (the async copy)
    }
    __syncthreads();

    // Initiate async bulk copy
    // TMA operation here (conceptual)
    // cuda::memcpy_async(...);

    // Arrive with transaction tracking
    // cuda::device::barrier_arrive_tx(bar, 1, expected_transaction_bytes);

    // Wait for copy to complete
    bar.wait(bar.arrive());

    // Now shared memory is populated
}
```

### Producer-Consumer Pattern with Asynchronous Barriers

The classic pattern uses two barriers to implement double-buffered producer-consumer synchronization:

```cpp
__global__ void producer_consumer(float* output, const float* input, int n) {
    __shared__ float buffer[2][256];
    __shared__ cuda::barrier<cuda::thread_scope_block> bar[2];

    if (threadIdx.x == 0) {
        init(&bar[0], blockDim.x);
        init(&bar[1], blockDim.x);
    }
    __syncthreads();

    for (int base = 0; base < n; base += 256) {
        int stage = (base / 256) % 2;

        // Producer: load data into shared buffer
        if (threadIdx.x < min(256, n - base)) {
            buffer[stage][threadIdx.x] = input[base + threadIdx.x];
        }

        // Arrive at barrier for this stage
        auto token = bar[stage].arrive();

        // Consumer can do independent work
        // ...

        // Wait for producer to finish
        bar[stage].wait(std::move(token));

        // Process data
        if (threadIdx.x < min(256, n - base)) {
            output[base + threadIdx.x] = buffer[stage][threadIdx.x] * 2.0f;
        }
    }
}
```

**Spatial partitioning with double buffering**:

- Divide shared memory into two buffers (A and B).
- While the producer fills buffer A, the consumer processes buffer B.
- Two one-sided synchronizations per buffer swap:
  1. Producer signals buffer A is ready (arrive on barrier A).
  2. Consumer signals buffer B is consumed (arrive on barrier B).

```cpp
// Spatial partitioning pattern
__shared__ float bufA[256], bufB[256];
__shared__ cuda::barrier<cuda::thread_scope_block> prod_bar, cons_bar;

if (threadIdx.x == 0) {
    init(&prod_bar, blockDim.x);
    init(&cons_bar, blockDim.x);
}
__syncthreads();

for (int chunk = 0; chunk < num_chunks; chunk++) {
    float* src_buf = (chunk % 2 == 0) ? bufA : bufB;
    float* dst_buf = (chunk % 2 == 0) ? bufB : bufA;

    // Producer: load next chunk
    load_to_shared(src_buf, chunk * 256 + threadIdx.x);
    auto prod_token = prod_bar.arrive();

    // Consumer: process previous chunk (if not first iteration)
    if (chunk > 0) {
        // ... wait for previous production ...
        auto cons_token = cons_bar.arrive();
        // ... process ...
    }

    prod_bar.wait(std::move(prod_token));
    // ... consume src_buf ...
    auto cons_token2 = cons_bar.arrive();
    cons_bar.wait(std::move(cons_token2));
}
```

---

## 8.2 Pipelines

CUDA Pipelines provide a higher-level abstraction for managing staged asynchronous operations. They build on top of asynchronous barriers and provide a structured producer-consumer model with support for multiple pipeline stages.

### Header

```cpp
#include <cuda/pipeline>
```

### Creating a Pipeline

```cpp
// Thread-scope pipeline (each thread has its own pipeline state)
cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

// Block-scope pipeline (shared across threads in a block)
__shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, num_stages> shared_state;
auto pipe = cuda::make_pipeline(cg::this_thread_block(), &shared_state);
```

**Parameters**:
- `Scope`: `cuda::thread_scope_thread`, `cuda::thread_scope_block`, or `cuda::thread_scope_device`.
- `num_stages`: Number of pipeline stages (for shared state).

### Producer-Consumer Operations

The pipeline API provides explicit acquire/commit/wait/release operations:

```cpp
// Producer side:
pipe.producer_acquire();   // Acquire a pipeline stage for producing
// ... submit async copies or other operations ...
pipe.producer_commit();    // Commit the async operations

// Consumer side:
pipe.consumer_wait();      // Wait for the current stage to be ready
// ... consume the data ...
pipe.consumer_release();   // Release the pipeline stage
```

### Single-Threaded Pipeline Example

```cpp
__global__ void pipeline_basic(float* output, const float* input, int n) {
    auto pipe = cuda::make_pipeline();
    constexpr int stages = 2;
    __shared__ float buffer[stages][256];

    for (int base = -stages * 256; base < n; base += 256) {
        // Determine current pipeline stage
        int stage = (base / 256) % stages;

        // Producer: async copy input to shared memory
        pipe.producer_acquire();

        int src_idx = base + stages * 256; // Look ahead
        if (src_idx >= 0 && src_idx < n) {
            cuda::memcpy_async(pipe, buffer[stage], &input[src_idx],
                               min(256, n - src_idx) * sizeof(float));
        }
        pipe.producer_commit();

        // Consumer: process data from previous stage
        pipe.consumer_wait();

        int cons_idx = base; // Current chunk to process
        if (cons_idx >= 0 && cons_idx < n) {
            int cons_stage = ((cons_idx / 256) % stages);
            int count = min(256, n - cons_idx);
            if (threadIdx.x < count) {
                output[cons_idx + threadIdx.x] = buffer[cons_stage][threadIdx.x] * 2.0f;
            }
        }

        pipe.consumer_release();
    }
}
```

### Block-Scope Pipeline with Multiple Stages

```cpp
__global__ void block_pipeline(float* output, const float* input, int n) {
    constexpr int stages = 2;
    __shared__ float buffer[stages][256];
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, stages> shared_state;

    auto pipe = cuda::make_pipeline(cg::this_thread_block(), &shared_state);

    // Cooperatively load and process data
    for (int base = -stages * 256; base < n; base += 256) {
        int stage = ((base / 256) % stages + stages) % stages;

        // All threads in block cooperatively acquire and commit
        pipe.producer_acquire();
        if (base + stages * 256 >= 0 && base + stages * 256 < n) {
            // Each thread copies its portion
            cuda::memcpy_async(pipe, &buffer[stage][threadIdx.x],
                               &input[base + stages * 256 + threadIdx.x],
                               sizeof(float));
        }
        pipe.producer_commit();

        // All threads wait for data
        pipe.consumer_wait();

        int cons_base = base;
        if (cons_base >= 0 && cons_base < n && threadIdx.x < min(256, n - cons_base)) {
            int cons_stage = ((cons_base / 256) % stages + stages) % stages;
            output[cons_base + threadIdx.x] = buffer[cons_stage][threadIdx.x] + 1.0f;
        }
        pipe.consumer_release();
    }
}
```

### Partial Wait with `consumer_wait_prior`

When using multiple pipeline stages, you can wait for a specific number of prior stages rather than the most recent one:

```cpp
// Wait for all stages older than N
cuda::pipeline_consumer_wait_prior<N>(pipe);

// Example: with 3 stages, wait_prior<1> waits for stages older than the current
// This allows overlapping with the most recent stage
```

**`wait_prior<N>`** signals that the consumer only needs data from stages committed more than `N` commits ago. This enables finer-grained overlapping:

```cpp
for (int base = 0; base < n; base += chunk_size) {
    // Producer
    pipe.producer_acquire();
    cuda::memcpy_async(pipe, buffer[stage], &input[base], copy_size);
    pipe.producer_commit();

    // Consumer: only wait for stages > N commits old
    cuda::pipeline_consumer_wait_prior<1>(pipe);
    // This means we don't wait for the MOST RECENT commit,
    // allowing overlap between copy and compute

    // Process data from an older stage
    process(buffer[old_stage]);

    pipe.consumer_release();
}
```

### Pipeline Early Exit

Threads can exit a pipeline early by calling `quit()`:

```cpp
pipe.quit();
```

This signals that the calling thread will no longer participate in pipeline operations. Other threads in the pipeline can continue. This is useful for:
- Handling boundary conditions where some threads run out of work.
- Early termination when a condition is met.

```cpp
for (int base = 0; base < n; base += 256) {
    if (threadIdx.x >= min(256, n - base)) {
        // This thread has no more work
        pipe.quit();
        break;
    }

    pipe.producer_acquire();
    // ... async copy ...
    pipe.producer_commit();
    pipe.consumer_wait();
    // ... process ...
    pipe.consumer_release();
}
```

### Pipeline API Summary

| Operation | Description |
|---|---|
| `cuda::make_pipeline()` | Create a thread-scope pipeline |
| `cuda::make_pipeline(group, &shared_state)` | Create a group-scope pipeline |
| `pipe.producer_acquire()` | Acquire a pipeline stage for async operations |
| `pipe.producer_commit()` | Commit submitted async operations |
| `pipe.consumer_wait()` | Wait for current stage to complete |
| `pipe.consumer_release()` | Release the pipeline stage |
| `cuda::pipeline_consumer_wait_prior<N>(pipe)` | Partial wait for stages older than N |
| `pipe.quit()` | Exit the pipeline early |

---

## 8.3 Programmatic Dependent Launch (CC 9.0+)

Programmatic Dependent Launch (PDL) is a feature introduced with the Hopper architecture (CC 9.0) that allows two kernels to overlap execution while maintaining a dependency relationship. Unlike traditional stream dependencies (which are sequential), PDL allows the primary kernel to signal completion while still executing, enabling the secondary kernel to begin earlier.

### Overview

Traditional CUDA stream serialization:
```
[Kernel A runs completely] -> [Kernel B starts]
```

Programmatic Dependent Launch:
```
[Kernel A runs] --> signal completion --> [Kernel B starts dependent work]
[Kernel A continues independent work]  --overlap--  [Kernel B independent work]
```

### API

#### Primary Kernel

The primary kernel signals that its dependent data is ready by calling `cudaTriggerProgrammaticLaunchCompletion()`. After this call, the kernel can continue executing independent work that overlaps with the secondary kernel.

```cpp
__global__ void primary_kernel(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Phase 1: Produce data that the secondary kernel depends on
    if (idx < n) {
        data[idx] = compute_value(idx);
    }

    // Signal that dependent data is ready
    // All threads in the grid must reach this point (effectively a grid sync)
    cudaTriggerProgrammaticLaunchCompletion();

    // Phase 2: Independent work that can overlap with the secondary kernel
    // This work does not modify data that the secondary kernel reads
    if (idx < n) {
        some_other_computation(idx); // Can overlap with secondary kernel
    }
}
```

#### Secondary Kernel

The secondary kernel calls `cudaGridDependencySynchronize()` to wait for the primary kernel to complete its dependent work. Before this call, the secondary kernel can perform independent work.

```cpp
__global__ void secondary_kernel(const float* data, float* result, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Phase 1: Independent work (can overlap with primary kernel)
    float local = precompute(idx);

    // Wait for the primary kernel to signal completion
    cudaGridDependencySynchronize();

    // Phase 2: Dependent work (uses data from primary kernel)
    if (idx < n) {
        result[idx] = data[idx] + local;
    }
}
```

#### Launch Configuration

The secondary kernel must be launched with the `cudaLaunchAttributeProgrammaticStreamSerialization` attribute to indicate it uses PDL:

```cpp
// Launch primary kernel normally
primary_kernel<<<gridSize, blockSize>>>(data, n);

// Configure secondary kernel for PDL
cudaLaunchAttribute attr;
attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
// Value of 0: default (no tail launch optimization)
// Value of 1: enable programmatic serialization

cudaLaunchConfig_t config = {0};
config.gridDim = gridSize;
config.blockDim = blockSize;
config.attrs = &attr;
config.numAttrs = 1;

// Launch secondary kernel with PDL
cudaLaunchKernelEx(&config, secondary_kernel, data, result, n);
```

#### Complete Example

```cpp
// Primary kernel: produces data in global memory
__global__ void producer_kernel(float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Produce data
    if (idx < n) {
        output[idx] = sqrtf((float)idx) * 2.0f;
    }

    // Signal that output[] is ready for the consumer
    cudaTriggerProgrammaticLaunchCompletion();

    // Continue with independent work (overlaps with consumer)
    if (idx < n) {
        output[idx + n] = output[idx] * output[idx]; // Writes to different region
    }
}

// Secondary kernel: consumes data from primary kernel
__global__ void consumer_kernel(const float* input, float* result, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Independent pre-processing (overlaps with primary kernel)
    float scale = 1.0f / (1.0f + (float)idx);

    // Wait for primary kernel's data to be ready
    cudaGridDependencySynchronize();

    // Now safe to read input[]
    if (idx < n) {
        result[idx] = input[idx] * scale;
    }
}

// Host-side launch
void launch_pdl(float* data, float* result, int n) {
    dim3 gridSize((n + 255) / 256);
    dim3 blockSize(256);

    // Primary kernel
    producer_kernel<<<gridSize, blockSize>>>(data, n);

    // Secondary kernel with PDL attribute
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config = {0};
    config.gridDim = gridSize;
    config.blockDim = blockSize;
    config.stream = 0;
    config.attrs = attrs;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, consumer_kernel, data, result, n);
}
```

### Performance Considerations

- **Overlap opportunity**: The more independent work in both kernels after the synchronization points, the greater the potential overlap and performance benefit.
- **Grid synchronization**: `cudaTriggerProgrammaticLaunchCompletion()` implies a grid-wide synchronization. All threads must reach this point before the signal is sent.
- **Memory visibility**: After `cudaGridDependencySynchronize()` in the secondary kernel, all writes by the primary kernel before `cudaTriggerProgrammaticLaunchCompletion()` are guaranteed to be visible.
- **Tail launch**: The secondary kernel may begin execution as a "tail launch" -- starting while the primary kernel's final waves are still executing on other SMs.

---

## 8.4 Memory Synchronization Domains (CC 9.0+)

Memory Synchronization Domains, introduced with the Hopper architecture (CC 9.0), provide a mechanism to reduce unnecessary memory ordering overhead when multiple kernels execute concurrently on the same GPU.

### The Problem

Without domains, a memory fence (e.g., `__threadfence()`) orders ALL writes by the issuing thread, regardless of which concurrent kernel will consume those writes. This can cause unnecessary stalls when concurrent kernels are independent.

### Solution

Each kernel launch can be assigned a **domain ID**. Memory fences then only order writes that are relevant to the same domain, reducing cross-kernel interference.

### Default Behavior

By default, each kernel launch is assigned a domain ID automatically. Fences within a kernel only order memory operations relative to threads in the same domain.

### Explicit Domain Configuration

```cpp
// Set the memory synchronization domain for a kernel launch
cudaLaunchAttribute attr;
attr.id = cudaLaunchAttributeMemSyncDomain;

// cudaLaunchMemSyncDomainDefault: default domain
// cudaLaunchMemSyncDomainRemote: remote domain (for cross-domain access)
attr.val.memSyncDomain = cudaLaunchMemSyncDomainDefault;

cudaLaunchConfig_t config = {0};
config.attrs = &attr;
config.numAttrs = 1;
// ... set other config fields ...
cudaLaunchKernelEx(&config, my_kernel, args...);
```

### Custom Domain Mapping

For fine-grained control over which domains interact, use `cudaLaunchAttributeMemSyncDomainMap`:

```cpp
// Define a custom domain mapping
cudaLaunchAttributeValue domainMap;
domainMap.memSyncDomainMap.shared = 0; // Shared domain ID
domainMap.memSyncDomainMap.local = 1;  // Local domain ID

cudaLaunchAttribute attr;
attr.id = cudaLaunchAttributeMemSyncDomainMap;
attr.val = domainMap;

cudaLaunchConfig_t config = {0};
config.attrs = &attr;
config.numAttrs = 1;
// ... launch ...
```

### Use Cases

- **Concurrent independent kernels**: Assign different domains to avoid unnecessary memory ordering between them.
- **Producer-consumer on same GPU**: Use matching domains when kernels need to communicate through global memory.
- **Reducing fence overhead**: Domains allow fences to be more targeted, reducing the performance impact of memory ordering.

### Example: Concurrent Kernels with Separate Domains

```cpp
void launch_independent_kernels(float* dataA, float* dataB, int n) {
    dim3 grid((n + 255) / 256);
    dim3 block(256);

    // Kernel A uses default domain
    cudaLaunchAttribute attrA;
    attrA.id = cudaLaunchAttributeMemSyncDomain;
    attrA.val.memSyncDomain = cudaLaunchMemSyncDomainDefault;

    cudaLaunchConfig_t configA = {0};
    configA.gridDim = grid;
    configA.blockDim = block;
    configA.attrs = &attrA;
    configA.numAttrs = 1;
    cudaLaunchKernelEx(&configA, kernel_A, dataA, n);

    // Kernel B uses a different domain setting
    cudaLaunchAttribute attrB;
    attrB.id = cudaLaunchAttributeMemSyncDomain;
    attrB.val.memSyncDomain = cudaLaunchMemSyncDomainRemote;

    cudaLaunchConfig_t configB = {0};
    configB.gridDim = grid;
    configB.blockDim = block;
    configB.stream = stream1; // Different stream
    configB.attrs = &attrB;
    configB.numAttrs = 1;
    cudaLaunchKernelEx(&configB, kernel_B, dataB, n);
}
```

---

## 8.5 Memory Fence Functions

CUDA provides built-in memory fence functions that ensure ordering of memory operations at different scopes. These are lighter-weight than full barriers and do not synchronize thread execution -- they only guarantee memory ordering.

### Available Fence Functions

| Function | Scope | Description |
|---|---|---|
| `__threadfence_block()` | Block | All writes to all memory before the fence are visible to all threads in the block |
| `__threadfence()` | Device | All writes to all memory before the fence are visible to all threads on the device |
| `__threadfence_system()` | System | All writes to all memory before the fence are visible to all threads and the host |

### How Fences Work

A memory fence guarantees that:

1. All memory writes (to global, shared, or local memory) issued by the calling thread **before** the fence are committed and visible at the fence's scope.
2. No memory read issued **after** the fence can be reordered before the fence.

Fences do **not**:
- Synchronize threads (no thread waits at a fence).
- Guarantee that writes by OTHER threads are visible (only the calling thread's writes).

### `__threadfence_block()`

Ensures the calling thread's memory writes are visible to all threads in the same block.

```cpp
__global__ void block_fence_example(int* data, int* flags) {
    // Write data
    data[threadIdx.x] = compute();

    // Ensure data is visible to all threads in this block
    __threadfence_block();

    // Signal that data is ready
    flags[threadIdx.x] = 1;
}
```

### `__threadfence()`

Ensures the calling thread's memory writes are visible to all threads on the device (across all blocks). This is essential for inter-block communication without cooperative launch.

```cpp
__global__ void device_fence_example(int* data, int* flags, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Write data to global memory
    if (idx < n) {
        data[idx] = compute();
    }

    // Ensure data write is visible device-wide before signaling
    __threadfence();

    // Signal completion (other blocks may read data[])
    if (threadIdx.x == 0) {
        atomicExch(&flags[blockIdx.x], 1);
    }
}
```

**Common pattern**: Atomic flag signaling with device fence:

```cpp
// Producer block
data[producer_idx] = value;
__threadfence();            // Ensure data is visible before flag
atomicExch(&flag, 1);       // Signal that data is ready

// Consumer block (different kernel or same kernel on different block)
while (atomicExch(&flag, 0) == 0) { /* spin */ }  // Wait for flag
// Now data[] is guaranteed to be visible
float val = data[producer_idx];
```

### `__threadfence_system()`

Ensures the calling thread's memory writes are visible across the entire system, including the host CPU and other GPUs (in multi-GPU configurations with unified memory or peer access).

```cpp
__global__ void system_fence_example(int* data, int* flag) {
    data[threadIdx.x] = compute();

    // Ensure data is visible to host and other GPUs
    __threadfence_system();

    if (threadIdx.x == 0) {
        atomicExch_system(flag, 1); // System-wide atomic
    }
}
```

### Fence Comparison

| Property | `__threadfence_block()` | `__threadfence()` | `__threadfence_system()` |
|---|---|---|---|
| Visibility scope | Block | Device | System (host + all GPUs) |
| Ordering | All memory types | All memory types | All memory types |
| Performance cost | Low | Medium | High |
| Use with atomics | Not typically needed | Yes (inter-block signaling) | Yes (system-wide signaling) |
| Memory types ordered | Global, shared, local | Global, shared, local | Global, shared, local |

### Fences vs. Barriers

| Aspect | Fence | Barrier |
|---|---|---|
| Thread synchronization | No (threads don't wait) | Yes (all threads must reach) |
| Memory ordering | Yes (ordering guarantee) | Yes (implied by sync) |
| Use case | Signaling between blocks/agents | Coordinating threads within a group |
| Overhead | Lower | Higher |
| Deadlock potential | None (non-blocking) | Possible if misused |

### Example: Lock-Free Queue with Fences

```cpp
__device__ void enqueue(int* queue, int* tail, int value) {
    // Atomically reserve a slot
    int pos = atomicAdd(tail, 1);

    // Write the value
    queue[pos] = value;

    // Ensure the write is visible before any reader sees the updated tail
    __threadfence();
}

__device__ int dequeue(int* queue, int* head) {
    // Read the head
    int pos = atomicAdd(head, 1);
    // The value at queue[pos] is guaranteed visible because
    // the producer used __threadfence() before signaling
    return queue[pos];
}
```

### Example: Inter-Block Reduction with Fences

```cpp
__global__ void fence_reduction(float* data, float* result, int* counter, int n) {
    __shared__ float block_sum;

    // Each thread computes partial sum
    float sum = 0.0f;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
         i += blockDim.x * gridDim.x) {
        sum += data[i];
    }

    // Block-level reduction using shared memory
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial[threadIdx.x] += partial[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_sum = partial[0];

        // Ensure block_sum is written before we increment the counter
        __threadfence();

        // Signal this block is done
        int idx = atomicAdd(counter, 1);

        // Last block performs final reduction
        if (idx == gridDim.x - 1) {
            // All blocks have written their results
            // (fence in each block ensured ordering)
            float total = 0.0f;
            // Read all block results (from block_sum of other blocks)
            // This pattern requires each block's result to be in global memory
            // For simplicity, using atomicAdd:
            atomicAdd(result, block_sum);
        }
    }
}
```

---

## 8.6 Summary of Synchronization Primitives

| Primitive | Scope | Blocking | Use Case |
|---|---|---|---|
| `__syncthreads()` | Block | Yes | Simple block-level synchronization |
| `cg::sync()` | Any group | Yes | Cooperative Groups synchronization |
| `cuda::barrier` | Configurable | Split (arrive/wait) | Flexible barrier with phase management |
| `cuda::pipeline` | Configurable | Split (acquire/commit/wait/release) | Staged async copy with compute overlap |
| PDL (CC 9.0+) | Grid | Signal-based | Inter-kernel overlap with dependency |
| `__threadfence_block()` | Block | No | Block-scope memory ordering |
| `__threadfence()` | Device | No | Device-scope memory ordering |
| `__threadfence_system()` | System | No | System-scope memory ordering |
| Memory Sync Domains (CC 9.0+) | Kernel | N/A | Reduce fence overhead between concurrent kernels |

### Choosing the Right Synchronization

1. **All threads in a block need to synchronize**: Use `cg::this_thread_block().sync()` or `__syncthreads()`.
2. **Threads need to do independent work while waiting**: Use `cuda::barrier` with split arrive/wait.
3. **Staged async copy with compute overlap**: Use `cuda::pipeline`.
4. **Inter-block synchronization without cooperative launch**: Use `__threadfence()` with atomic flags.
5. **Cross-grid synchronization**: Use `cg::this_grid().sync()` with cooperative launch, or PDL (CC 9.0+).
6. **Overlapping two dependent kernels**: Use Programmatic Dependent Launch (CC 9.0+).
7. **Reducing fence overhead with concurrent kernels**: Use Memory Synchronization Domains (CC 9.0+).
8. **System-wide memory visibility**: Use `__threadfence_system()` with system-scope atomics.
