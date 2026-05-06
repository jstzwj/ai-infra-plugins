# 4. Asynchronous Execution

This document covers CUDA's asynchronous execution model, including streams, events, callbacks, synchronization mechanisms, and environment variables that control asynchronous behavior. These features are fundamental to achieving overlapping computation and data transfers on NVIDIA GPUs.

---

## Table of Contents

1. [Asynchronous Concurrent Execution](#41-asynchronous-concurrent-execution)
2. [CUDA Streams](#42-cuda-streams)
3. [CUDA Events](#43-cuda-events)
4. [Callbacks](#44-callbacks)
5. [Default Stream Behavior](#45-default-stream-behavior)
6. [Stream Priorities](#46-stream-priorities)
7. [Explicit Synchronization](#47-explicit-synchronization)
8. [Implicit Synchronization](#48-implicit-synchronization)
9. [Cross-Stream Dependencies](#49-cross-stream-dependencies)
10. [Environment Variables](#410-environment-variables)

---

## 4.1 Asynchronous Concurrent Execution

CUDA allows concurrent, or overlapping, execution of multiple tasks:

- Computation on the host (CPU)
- Computation on the device (GPU)
- Host-to-device (H2D) memory transfers
- Device-to-host (D2H) memory transfers
- Memory transfers within a given device (device-to-device)
- Memory transfers among multiple devices

The concurrency is expressed via an asynchronous interface where a dispatching function call or kernel launch returns immediately. Asynchronous calls usually return before the dispatched operation has completed and may even return before the operation has started. When the final results are needed, the application must perform some form of synchronization.

A typical pattern is overlapping host-device memory transfers with computation, thereby reducing or eliminating transfer overhead.

### Three Synchronization Approaches

CUDA provides three main ways to synchronize with dispatched operations:

| Approach | Description |
|----------|-------------|
| **Blocking** | The application calls a function that blocks (waits) until the operation has completed. Example: `cudaStreamSynchronize()`. |
| **Non-blocking (polling)** | The application calls a function that returns immediately, reporting the operation's status. Example: `cudaStreamQuery()`. |
| **Callback** | A pre-registered function is executed on the host when the operation has completed. Example: `cudaLaunchHostFunc()`. |

The actual ability to carry out operations concurrently depends on the CUDA version and the compute capability of the hardware. The core API components for asynchronous execution are **CUDA Streams** and **CUDA Events**.

---

## 4.2 CUDA Streams

A CUDA stream is an abstraction that represents a sequence of operations. A stream operates like a **work queue** into which operations (kernel launches, memory copies, etc.) are enqueued for ordered execution. Operations in a stream are guaranteed to execute **in the order they were enqueued**. An operation in a stream cannot leapfrog other operations.

An application may use multiple streams simultaneously. The runtime selects tasks from streams that have work available, depending on GPU resource state. Streams may be assigned a **priority** that acts as a scheduling hint.

All API function calls and kernel launches operating in a stream are **asynchronous with respect to the host thread**. The host code continues executing immediately after enqueueing work.

### 4.2.1 Creating and Destroying Streams

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

// ... enqueue work into stream ...

cudaStreamDestroy(stream);
```

If the device is still doing work in the stream when `cudaStreamDestroy()` is called, the function will return immediately and the stream will be destroyed once all queued work completes.

### 4.2.2 Launching Kernels in Streams

The stream is specified as the fourth execution configuration parameter:

```cpp
//                        grid  block  shmem  stream
kernel<<<grid, block, 0, stream>>>(...);
```

The kernel launch is asynchronous -- the function call returns immediately. The kernel executes in the specified stream, and the application is free to perform other CPU work or enqueue GPU work in other streams.

### 4.2.3 Launching Memory Transfers in Streams

Use `cudaMemcpyAsync()` to perform asynchronous memory transfers in a stream:

```cpp
// Copy `size` bytes from host to device in `stream`
cudaMemcpyAsync(d_dst, h_src, size, cudaMemcpyHostToDevice, stream);
```

Like kernel launches, this function returns immediately. In order to access the results safely, the application must synchronize.

Other transfer functions such as `cudaMemcpy2D()` and `cudaMemcpy3D()` also have asynchronous variants (`cudaMemcpy2DAsync()`, `cudaMemcpy3DAsync()`).

**Important**: For memory copies involving CPU memory to execute asynchronously, the host buffers must be **pinned** (page-locked). Use `cudaMallocHost()` or `cudaHostAlloc()` to allocate pinned host memory. If non-pinned host memory is used, `cudaMemcpyAsync()` reverts to synchronous behavior and will not overlap with other work.

### 4.2.4 Stream Synchronization

#### Blocking Synchronization

```cpp
// Blocks the host thread until all work in the stream has completed
cudaStreamSynchronize(stream);
// At this point, all stream operations are guaranteed complete
```

#### Non-Blocking Query

```cpp
// Returns cudaSuccess if stream is empty, cudaErrorNotReady if not
cudaError_t status = cudaStreamQuery(stream);
switch (status) {
    case cudaSuccess:
        printf("Stream is empty (all work done)\n");
        break;
    case cudaErrorNotReady:
        printf("Stream still has pending work\n");
        break;
    default:
        // An error occurred
        printf("Error: %s\n", cudaGetErrorString(status));
        break;
}
```

Note: `cudaErrorNotReady` returned by `cudaStreamQuery()` is not considered an error and is not reported by `cudaPeekAtLastError()` or `cudaGetLastError()`.

### 4.2.5 Multi-Stream Overlap Example

```cpp
const int N = 2;
cudaStream_t streams[N];
float *d_data[N], *h_data[N];

for (int i = 0; i < N; i++) {
    cudaStreamCreate(&streams[i]);
    cudaMalloc(&d_data[i], bytes);
    cudaMallocHost(&h_data[i], bytes);
    initializeData(h_data[i], elements);
}

// Overlap transfers and computation across streams
for (int i = 0; i < N; i++) {
    cudaMemcpyAsync(d_data[i], h_data[i], bytes,
                    cudaMemcpyHostToDevice, streams[i]);
    kernel<<<grid, block, 0, streams[i]>>>(d_data[i], elements);
    cudaMemcpyAsync(h_data[i], d_data[i], bytes,
                    cudaMemcpyDeviceToHost, streams[i]);
}

for (int i = 0; i < N; i++) {
    cudaStreamSynchronize(streams[i]);
}

// Cleanup
for (int i = 0; i < N; i++) {
    cudaStreamDestroy(streams[i]);
    cudaFree(d_data[i]);
    cudaFreeHost(h_data[i]);
}
```

### 4.2.6 Stream Ordering Semantics

CUDA streams are **in-order streams**: the order of execution of operations is the same as the order in which they were enqueued. An operation in a stream cannot leapfrog other operations. Memory operations (such as copies) are tracked by the runtime and will always complete before the next operation in order, allowing dependent kernels safe access to transferred data.

Exceptions to strict in-order semantics exist for specific optimizations such as **Programmatic Dependent Launch** (which allows overlap of two kernels through special attributes) and **batched memory transfers** (`cudaMemcpyBatchAsync()`).

---

## 4.3 CUDA Events

CUDA events are markers inserted into a stream. They serve two primary purposes:

1. **Dependency tracking** -- events enable fine-grained synchronization between streams without requiring the entire stream to drain.
2. **Timing** -- events record timestamps that can be used to accurately measure kernel execution and transfer durations.

### 4.3.1 Creating and Destroying Events

```cpp
cudaEvent_t event;
cudaEventCreate(&event);
// ... use event ...
cudaEventDestroy(event);
```

### 4.3.2 Recording Events

```cpp
cudaEventRecord(event, stream);
```

The event is enqueued into the stream at the current position. It completes (is "reached") when all preceding operations in the stream have finished.

### 4.3.3 Timing with Events

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Record start before kernel
cudaEventRecord(start, stream);

// Launch kernel
kernel<<<grid, block, 0, stream>>>(...);

// Record stop after kernel
cudaEventRecord(stop, stream);

// Wait for stop event to complete
cudaEventSynchronize(stop);

// Compute elapsed time in milliseconds
float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel execution time: %.3f ms\n", ms);

// Cleanup
cudaEventDestroy(start);
cudaEventDestroy(stop);
cudaStreamDestroy(stream);
```

### 4.3.4 Dependency-Only Events

For events used solely for synchronization (not timing), disable timing to improve performance:

```cpp
cudaEvent_t event;
cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
```

Events created with `cudaEventDisableTiming` have lower overhead and should be used whenever timing is not needed.

### 4.3.5 Checking Event Status

#### Blocking Check

```cpp
cudaEventSynchronize(event);  // Blocks until event completes
```

#### Non-Blocking Check

```cpp
cudaError_t status = cudaEventQuery(event);
if (status == cudaSuccess) {
    // Event has been reached (all preceding work completed)
} else if (status == cudaErrorNotReady) {
    // Event has not been reached yet
}
```

### 4.3.6 Polling for Completion Example

This pattern shows how to overlap CPU work with GPU execution by polling an event:

```cpp
cudaEvent_t event;
cudaStream_t stream1, stream2;

cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
cudaEventCreate(&event);

// Launch kernel1, record event, then launch kernel2
kernel1<<<grid, block, 0, stream1>>>(d_data, size);
cudaEventRecord(event, stream1);
kernel2<<<grid, block, 0, stream1>>>();

// Do CPU work while checking if kernel1 is done
bool copyStarted = false;
while (!allCPUWorkDone() || !copyStarted) {
    doNextChunkOfCPUWork();
    if (!copyStarted && cudaEventQuery(event) == cudaSuccess) {
        // kernel1 done; launch async copy in stream2
        cudaMemcpyAsync(h_data, d_data, size,
                        cudaMemcpyDeviceToHost, stream2);
        copyStarted = true;
    }
}
cudaDeviceSynchronize();
```

---

## 4.4 Callbacks

CUDA provides a mechanism for launching host-side functions from within a stream. There are two APIs:

- **`cudaLaunchHostFunc()`** -- the recommended API.
- **`cudaStreamAddCallback()`** -- deprecated; included for legacy code compatibility.

### 4.4.1 cudaLaunchHostFunc

```cpp
void CUDART_CB myHostFunc(void *userData) {
    // This function runs on the host when all preceding
    // work in the stream has completed.
    // IMPORTANT: May NOT call any CUDA APIs.
    MyData *data = static_cast<MyData*>(userData);
    printf("Callback executed with value: %d\n", data->value);
}

// Usage
MyData userData = {42};
cudaLaunchHostFunc(stream, myHostFunc, &userData);
```

**Signature:**

```cpp
cudaError_t cudaLaunchHostFunc(cudaStream_t stream,
                               void (*func)(void*),
                               void *userData);
```

**Restrictions:**

- The host function **may NOT call any CUDA APIs**. This includes all runtime and driver API functions.
- The function must not block or perform long-running operations, as it runs on an internal CUDA thread.

**Unified Memory guarantees for callbacks:**

- The stream is considered **idle** for the duration of the function's execution.
- The start of execution has the same effect as synchronizing an event recorded immediately prior.
- Adding device work to any stream does not take effect until all preceding host functions and callbacks have executed.
- The stream remains idle across consecutive host functions unless device work follows.

### 4.4.2 cudaStreamAddCallback (Deprecated)

```cpp
void CUDART_CB myCallback(cudaStream_t stream, cudaError_t status,
                           void *userData) {
    if (status != cudaSuccess) {
        printf("Callback received error: %s\n",
               cudaGetErrorString(status));
    }
    // Also may NOT call any CUDA APIs
}

cudaStreamAddCallback(stream, myCallback, &userData, 0);
```

This API is slated for deprecation and removal. Applications should use `cudaLaunchHostFunc()` instead. The callback receives the stream handle and error status as additional parameters compared to `cudaLaunchHostFunc`.

---

## 4.5 Default Stream Behavior

CUDA has a default stream used when no stream is specified. The behavior of this default stream depends on configuration.

### 4.5.1 Legacy Default Stream (NULL / 0)

By default (without special compiler flags), CUDA uses a **legacy default stream** (also called the NULL stream or stream 0). This stream:

- Is **shared among all host threads** in a process.
- Is a **blocking stream** -- operations in the legacy default stream synchronize with all other blocking streams.

```cpp
cudaStream_t stream1;
cudaStreamCreate(&stream1);  // blocking stream by default

kernel1<<<grid, block, 0, stream1>>>(...);  // launched in stream1
kernel2<<<grid, block>>>(...);              // launched in legacy default stream
kernel3<<<grid, block, 0, stream1>>>(...);  // launched in stream1

// kernel2 WAITS for kernel1, kernel3 WAITS for kernel2
// No concurrency between the three kernels!
```

In the above code, `kernel2` (in the legacy default stream) blocks until `kernel1` completes, and `kernel3` blocks until `kernel2` completes -- even though `stream1` is independent.

### 4.5.2 Per-Thread Default Stream

Since CUDA 7, each host thread can have its own **independent default stream** that does not synchronize with other streams. Enable with either:

- **Compiler flag:** `--default-stream per-thread` passed to `nvcc`
- **Preprocessor macro:** `#define CUDA_API_PER_THREAD_DEFAULT_STREAM` before including CUDA headers

```cpp
// With per-thread default stream enabled:
cudaStream_t stream1;
cudaStreamCreate(&stream1);

kernel1<<<grid, block, 0, stream1>>>(...);  // stream1
kernel2<<<grid, block>>>(...);              // per-thread default stream
kernel3<<<grid, block, 0, stream1>>>(...);  // stream1

// All three kernels CAN execute concurrently
```

With per-thread default stream, each host thread's default stream acts like a non-blocking stream.

### 4.5.3 Non-Blocking Streams

Regardless of default stream configuration, streams can explicitly be created as non-blocking:

```cpp
cudaStream_t stream;
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
```

Non-blocking streams never synchronize with the legacy default stream. They can execute concurrently with work in any default stream.

---

## 4.6 Stream Priorities

Streams can be assigned priorities to influence scheduling. Lower numbers correspond to higher priorities. The default priority is 0.

```cpp
int leastPriority, greatestPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

// Create streams with different priorities
cudaStream_t stream1, stream2;
cudaStreamCreateWithPriority(&stream1, cudaStreamDefault, leastPriority);
cudaStreamCreateWithPriority(&stream2, cudaStreamDefault, greatestPriority);

// Or with non-blocking flag:
cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, greatestPriority);
```

**Notes:**

- Stream priority is only a **hint** to the runtime. It does not guarantee a specific execution order.
- Priority generally applies primarily to kernel launches and may not be respected for memory transfers.
- Priority does not preempt already executing work.

---

## 4.7 Explicit Synchronization

CUDA provides several mechanisms for explicit synchronization:

| API | Scope | Behavior |
|-----|-------|----------|
| `cudaDeviceSynchronize()` | Device | Blocks until **all** preceding commands in **all** streams of **all** host threads have completed. |
| `cudaStreamSynchronize(stream)` | Stream | Blocks until all preceding commands in the specified stream have completed. Allows other streams to continue. |
| `cudaStreamWaitEvent(stream, event, flags)` | Stream-Event | All commands added to `stream` **after** this call will delay execution until `event` has completed. `flags` must be 0. |
| `cudaStreamQuery(stream)` | Stream (non-blocking) | Returns `cudaSuccess` if all preceding commands have completed, `cudaErrorNotReady` otherwise. |
| `cudaEventSynchronize(event)` | Event | Blocks until the event has been reached. |
| `cudaEventQuery(event)` | Event (non-blocking) | Returns `cudaSuccess` if the event has been reached, `cudaErrorNotReady` otherwise. |

---

## 4.8 Implicit Synchronization

Certain operations can implicitly prevent inter-stream concurrency. Two operations from different streams **cannot run concurrently** if any of the following occurs between them:

- A page-locked host memory allocation (`cudaMallocHost()`, `cudaHostAlloc()`)
- A device memory allocation (`cudaMalloc()`, `cudaMallocPitch()`, `cudaMallocArray()`)
- A device memory set (`cudaMemset()`, `cudaMemset2D()`, `cudaMemset3D()`)
- A memory copy between two regions on the same device
- Any operation on the **NULL (legacy default) stream** (unless the other streams are non-blocking)
- A switch between L1/shared memory configurations (`cudaDeviceSetCacheConfig()`)

**Best practices to maximize concurrency:**

- Issue all independent operations before dependent ones.
- Delay synchronization of any kind as long as possible.
- Use non-blocking streams (`cudaStreamNonBlocking`) if the legacy default stream synchronization is not desired.
- Use `cudaMallocAsync()`/`cudaFreeAsync()` instead of `cudaMalloc()`/`cudaFree()` to avoid synchronization.

---

## 4.9 Cross-Stream Dependencies

Events enable cross-stream dependencies: one stream can wait for operations in another stream to complete.

```cpp
cudaStream_t s1, s2;
cudaEvent_t event;

cudaEventCreate(&event);
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);

// Enqueue work in s1, then record event
kernel1<<<grid, block, 0, s1>>>();
cudaEventRecord(event, s1);

// s2 waits for s1's event before executing kernel2
cudaStreamWaitEvent(s2, event, 0);
kernel2<<<grid, block, 0, s2>>>();  // guaranteed to wait for kernel1

// More complex: fan-in / fan-out
// s1: A -> event1
// s2: B -> event2
// s3: waits for event1 AND event2, then C
cudaStreamWaitEvent(s3, event1, 0);
cudaStreamWaitEvent(s3, event2, 0);
kernelC<<<grid, block, 0, s3>>>();
```

The `cudaStreamWaitEvent()` call makes all commands subsequently added to the stream wait until the event completes. The event can be recorded in a different stream, enabling cross-stream ordering without blocking the host.

---

## 4.10 Environment Variables

### CUDA_LAUNCH_BLOCKING

```bash
export CUDA_LAUNCH_BLOCKING=1
```

When set to `1`, all kernel launches become **synchronous** -- the host blocks until each kernel completes. This is useful for debugging because:

- Errors are reported immediately at the launch site rather than at a later synchronization point.
- It helps identify which specific kernel caused an error.
- It serializes execution, making behavior more deterministic.

**Warning:** This causes significant performance degradation and should only be used for debugging.

### CUDA_DEVICE_MAX_CONNECTIONS

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=16
```

Sets the number of concurrent work queues (connections) between the host and device. Valid range is 1 to 32; default is 8. Higher values enable more concurrent operations across streams but consume more resources.

This affects how many streams can truly execute concurrently at the hardware level. If more streams are used than available connections, some streams will share connections, reducing potential overlap.

### CUDA_DEVICE_MAX_COPY_CONNECTIONS

```bash
export CUDA_DEVICE_MAX_COPY_CONNECTIONS=8
```

Controls the number of concurrent copy engine connections. This is separate from compute connections and affects how many concurrent memory transfers can occur.

### Related: Asynchronous Error Handling

When errors occur in asynchronous operations, they may not be reported until the next CUDA API call that returns `cudaError_t`. Use:

```cpp
// After synchronization:
cudaError_t err = cudaGetLastError();   // Returns AND clears last error
cudaError_t err2 = cudaPeekAtLastError(); // Returns but does NOT clear

if (err != cudaSuccess) {
    printf("Error: %s (%s)\n",
           cudaGetErrorName(err), cudaGetErrorString(err));
}
```

---

## Summary

| Concept | Key Point |
|---------|-----------|
| Streams | Ordered work queues; multiple streams enable concurrency |
| Events | Stream markers for timing and cross-stream dependencies |
| Callbacks | Host functions invoked from stream order; cannot call CUDA APIs |
| Default stream | Legacy (NULL) is blocking and shared; per-thread is independent |
| Priorities | Hints to scheduler; lower number = higher priority |
| Explicit sync | `cudaDeviceSynchronize`, `cudaStreamSynchronize`, `cudaStreamWaitEvent` |
| Implicit sync | Allocations, memset, NULL stream ops can prevent concurrency |
| Cross-stream deps | Use `cudaEventRecord` + `cudaStreamWaitEvent` |
| Debug env vars | `CUDA_LAUNCH_BLOCKING=1` for synchronous debugging |
