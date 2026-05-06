# 6. Stream-Ordered Memory Allocator

This document covers the CUDA stream-ordered memory allocator, which enables applications to order memory allocation and deallocation with other work launched into a CUDA stream. This avoids the GPU-wide synchronization inherent in `cudaMalloc`/`cudaFree` and enables efficient memory pool management, inter-process sharing, and fine-grained caching control.

**CUDA Toolkit Version:** 13.2 (March 2026)

---

## Table of Contents

1. [Introduction](#61-introduction)
2. [API](#62-api)
3. [Memory Pools](#63-memory-pools)
4. [Multi-GPU Support](#64-multi-gpu-support)
5. [IPC Memory Pools](#65-ipc-memory-pools)
6. [Tuning and Statistics](#66-tuning-and-statistics)
7. [Addendums and Limitations](#67-addendums-and-limitations)

---

## 6.1 Introduction

Managing memory allocations using `cudaMalloc` and `cudaFree` causes the GPU to **synchronize across all executing CUDA streams**. This is because the driver must ensure no kernel on any stream is using the memory before it can be allocated or freed.

The stream-ordered memory allocator solves this by allowing memory allocation and deallocation to be **ordered within a CUDA stream**, just like kernel launches and async copies. This means:

- **No cross-stream synchronization** -- allocations and frees are ordered relative to stream work.
- **Memory reuse** -- the allocator can reuse memory from frees that are known to have completed in stream order.
- **Driver-managed caching** -- the driver can hold and reuse physical memory blocks, reducing OS allocation calls.
- **IPC support** -- allocations can be shared between processes with OS-level security.

### Benefits

- Reduces the need for custom memory management abstractions.
- Enables multiple libraries to share a common memory pool managed by the driver, reducing excess memory consumption.
- Allows the driver to perform optimizations based on awareness of the allocator and stream management APIs.

### Querying Support

```cpp
int deviceSupportsMemoryPools = 0;
int driverVersion = 0;
cudaDriverGetVersion(&driverVersion);

if (driverVersion >= 11020) {  // CUDA 11.2+
    cudaDeviceGetAttribute(&deviceSupportsMemoryPools,
                           cudaDevAttrMemoryPoolsSupported, device);
}
if (deviceSupportsMemoryPools) {
    // Stream-ordered memory allocator is available
}
```

---

## 6.2 API

### 6.2.1 Allocating Memory

```cpp
void* ptr;
size_t size = 512;
cudaMallocAsync(&ptr, size, stream);
// ptr is available for use after all preceding work in stream
kernel<<<..., stream>>>(ptr, ...);
cudaFreeAsync(ptr, stream);
```

`cudaMallocAsync` determines the device based on the specified memory pool or the supplied stream (it **ignores** the current device/context).

**Key behavior:**

- The allocation becomes usable after all preceding work in the stream completes.
- When accessing the allocation from a different stream, the user must guarantee the access occurs after the allocation (e.g., via events).

### 6.2.2 Freeing Memory

```cpp
cudaFreeAsync(ptr, stream);
```

Frees memory asynchronously in stream order. The memory is reclaimed after all preceding work in the stream completes.

**Cross-stream free example:**

```cpp
cudaMallocAsync(&ptr, size, stream1);
cudaEventRecord(event1, stream1);

// stream2 waits for allocation
cudaStreamWaitEvent(stream2, event1);
kernel<<<..., stream2>>>(ptr, ...);
cudaEventRecord(event2, stream2);

// stream3 waits for all uses, then frees
cudaStreamWaitEvent(stream3, event2);
cudaFreeAsync(ptr, stream3);
```

### 6.2.3 Mixing Allocation and Free APIs

The stream-ordered allocator is compatible with the legacy allocation APIs, with appropriate synchronization:

#### cudaMalloc memory freed with cudaFreeAsync

```cpp
cudaMalloc(&ptr, size);
kernel<<<..., stream>>>(ptr, ...);
cudaFreeAsync(ptr, stream);  // All accesses must be complete before free
```

#### cudaMallocAsync memory freed with cudaFree

```cpp
cudaMallocAsync(&ptr, size, stream);
kernel<<<..., stream>>>(ptr, ...);
cudaStreamSynchronize(stream);  // Must sync before synchronous free
cudaFree(ptr);
```

When using `cudaFree()` on a `cudaMallocAsync` allocation, the user must ensure all GPU accesses are complete (via `cudaStreamSynchronize`, `cudaEventSynchronize`, `cudaDeviceSynchronize`, etc.) before calling `cudaFree()`.

---

## 6.3 Memory Pools

Memory pools encapsulate virtual address and physical memory resources. All `cudaMallocAsync` calls use resources from a memory pool.

### 6.3.1 How Pool Selection Works

1. If a pool is explicitly specified (via `cudaMallocFromPoolAsync` or C++ overloads), that pool is used.
2. Otherwise, the **current memory pool** for the stream's device is used.
3. If `cudaDeviceSetMempool` has not been called, the **default memory pool** is used.

```cpp
// Get/set the current pool for a device
cudaMemPool_t currentPool;
cudaDeviceGetMempool(&currentPool, device);
cudaDeviceSetMempool(device, newPool);

// Allocate from a specific pool
cudaMallocFromPoolAsync(&ptr, size, pool, stream);
```

**Note:** The pool current to a device is always local to that device. Allocating without specifying a pool always yields a local allocation.

### 6.3.2 Default/Implicit Pools

```cpp
cudaMemPool_t defaultPool;
cudaDeviceGetDefaultMempool(&defaultPool, device);
```

**Characteristics:**

- Allocations are **non-migratable device allocations** located on the device.
- Always accessible from the resident device.
- Accessibility can be modified with `cudaMemPoolSetAccess` and queried with `cudaMemPoolGetAccess`.
- **Does not support IPC.**
- Do not need to be explicitly created (hence "implicit").

### 6.3.3 Explicit Pools

```cpp
cudaMemPool_t memPool;
cudaMemPoolProps poolProps = {};
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = device;
poolProps.location.type = cudaMemLocationTypeDevice;
cudaMemPoolCreate(&memPool, &poolProps);
```

Explicit pools support additional properties:

- **IPC capability** -- via `handleTypes` in pool properties.
- **NUMA residency** -- allocations can be resident on a specific CPU NUMA node.
- **Custom handle types** -- e.g., POSIX file descriptors for IPC.

**IPC-capable pool on a CPU NUMA node:**

```cpp
int cpu_numa_id = 0;
cudaMemPoolProps poolProps = {};
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = cpu_numa_id;
poolProps.location.type = cudaMemLocationTypeHostNuma;
poolProps.handleTypes = cudaMemHandleTypePosixFileDescriptor;
cudaMemPoolCreate(&ipcMemPool, &poolProps);
```

### 6.3.4 Pool Destruction

```cpp
cudaMemPoolDestroy(pool);
```

Destroy a pool created with `cudaMemPoolCreate()`. Default pools cannot be destroyed.

---

## 6.4 Multi-GPU Support

Memory pool allocation accessibility **does not follow** `cudaDeviceEnablePeerAccess()` / `cuCtxEnablePeerAccess()`. Instead, accessibility is controlled explicitly via `cudaMemPoolSetAccess()`.

```cpp
cudaError_t setAccessOnDevice(cudaMemPool_t memPool, int residentDevice,
                               int accessingDevice) {
    cudaMemAccessDesc accessDesc = {};
    accessDesc.location.type = cudaMemLocationTypeDevice;
    accessDesc.location.id = accessingDevice;
    accessDesc.flags = cudaMemAccessFlagsProtReadWrite;

    int canAccess = 0;
    cudaError_t error = cudaDeviceCanAccessPeer(&canAccess,
                                                 accessingDevice,
                                                 residentDevice);
    if (error != cudaSuccess) return error;
    if (canAccess == 0) return cudaErrorPeerAccessUnsupported;

    return cudaMemPoolSetAccess(memPool, &accessDesc, 1);
}
```

**Key points:**

- `cudaMemPoolSetAccess` affects **all** allocations from the pool, not just future ones.
- Accessibility reported by `cudaMemPoolGetAccess` applies to all allocations.
- Once a pool is made accessible from a GPU, it should remain accessible for the pool's lifetime (changing accessibility frequently is not recommended).
- Default accessibility is only from the device where allocations are located (resident device).

---

## 6.5 IPC Memory Pools

Memory pools can be enabled for interprocess communication (IPC), allowing GPU memory sharing between processes with OS-level security.

### 6.5.1 Sharing the Pool

IPC requires two steps: share the pool, then share specific allocations.

#### Step 1: Export the Pool (Exporting Process)

```cpp
// Create an exportable IPC-capable pool
cudaMemPoolProps poolProps = {};
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = 0;
poolProps.location.type = cudaMemLocationTypeDevice;
poolProps.handleTypes = cudaMemHandleTypePosixFileDescriptor;
cudaMemPoolCreate(&memPool, &poolProps);

// Export to OS-native handle
int fdHandle = 0;
cudaMemPoolExportToShareableHandle(&fdHandle, memPool,
                                    cudaMemHandleTypePosixFileDescriptor, 0);
// Transfer fdHandle to importing process via OS IPC (shared memory, socket, etc.)
```

#### Step 2: Import the Pool (Importing Process)

```cpp
int fdHandle;  // received from exporting process
cudaMemPool_t importedMemPool;
cudaMemPoolImportFromShareableHandle(&importedMemPool,
                                      (void*)fdHandle,
                                      cudaMemHandleTypePosixFileDescriptor,
                                      0);
```

### 6.5.2 Sharing Allocations

Once the pool is shared, allocations from the pool can be shared between processes.

#### Export Allocation (Exporting Process)

```cpp
cudaMemPoolPtrExportData exportData;
cudaEvent_t readyIpcEvent;
cudaIpcEventHandle_t readyIpcEventHandle;

// Create IPC event for coordination
cudaEventCreate(&readyIpcEvent,
                cudaEventDisableTiming | cudaEventInterprocess);

// Allocate and record ready event
cudaMallocAsync(&ptr, size, exportMemPool, stream);
cudaEventRecord(readyIpcEvent, stream);

// Export pointer data
cudaMemPoolExportPointer(&exportData, ptr);
cudaIpcGetEventHandle(&readyIpcEventHandle, readyIpcEvent);

// Share exportData and readyIpcEventHandle with importing process
```

#### Import Allocation (Importing Process)

```cpp
cudaIpcOpenEventHandle(&readyIpcEvent, readyIpcEventHandle);

// Import the allocation (does not block on allocation readiness)
cudaMemPoolImportPointer(&ptr, importedMemPool, &importData);

// Wait for allocation to be ready
cudaStreamWaitEvent(stream, readyIpcEvent);
kernel<<<..., stream>>>(ptr, ...);
```

### 6.5.3 Freeing Shared Allocations

The free must happen in the importing process before the exporting process:

```cpp
// Importing process
kernel<<<..., stream>>>(ptr, ...);
cudaFreeAsync(ptr, stream);
cudaIpcEventRecord(finishedIpcEvent, stream);

// Exporting process
cudaStreamWaitEvent(stream, finishedIpcEvent);
kernel<<<..., stream>>>(ptrInExportingProcess, ...);
cudaFreeAsync(ptrInExportingProcess, stream);
```

### 6.5.4 IPC Limitations

**Export pool limitations:**
- IPC pools do not currently support releasing physical blocks back to the OS.
- `cudaMemPoolTrimTo()` has no effect on IPC export pools.
- `cudaMemPoolAttrReleaseThreshold` is effectively ignored.

**Import pool limitations:**
- Allocating from an import pool is not allowed.
- Import pools cannot be set current or used with `cudaMallocFromPoolAsync`.
- Reuse policy attributes have no meaning for import pools.
- Resource usage statistics reflect only imported allocations.

---

## 6.6 Tuning and Statistics

### 6.6.1 Release Threshold

The release threshold controls how much memory the pool retains before attempting to release memory back to the OS on synchronization:

```cpp
// Set to UINT64_MAX to prevent automatic shrinking
cuuint64_t threshold = UINT64_MAX;
cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &threshold);
```

When the pool holds more than the threshold bytes, the allocator will try to release memory on the next `cudaStreamSynchronize`, `cudaEventSynchronize`, or `cudaDeviceSynchronize` call.

### 6.6.2 Explicit Trimming

```cpp
// Trim pool to retain at least minBytesToKeep bytes
cudaMemPoolTrimTo(memPool, minBytesToKeep);
```

```cpp
// Example: high threshold during compute phase, then trim
cuuint64_t threshold = UINT64_MAX;
cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &threshold);

for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
        cudaMallocAsync(&ptrs[j], size[j], stream);
    }
    kernel<<<..., stream>>>(ptrs, ...);
    for (int j = 0; j < 10; j++) {
        cudaFreeAsync(ptrs[j], stream);
    }
}

// Phase done; release memory
cudaStreamSynchronize(stream);
cudaMemPoolTrimTo(memPool, 0);
```

### 6.6.3 Resource Usage Statistics

| Attribute | Description |
|-----------|-------------|
| `cudaMemPoolAttrReservedMemCurrent` | Current total physical GPU memory consumed by the pool |
| `cudaMemPoolAttrReservedMemHigh` | High watermark of `ReservedMemCurrent` since last reset |
| `cudaMemPoolAttrUsedMemCurrent` | Total size of memory allocated from pool and not available for reuse |
| `cudaMemPoolAttrUsedMemHigh` | High watermark of `UsedMemCurrent` since last reset |

```cpp
struct UsageStatistics {
    cuuint64_t reserved;
    cuuint64_t reservedHigh;
    cuuint64_t used;
    cuuint64_t usedHigh;
};

void getUsageStatistics(cudaMemPool_t memPool, UsageStatistics* stats) {
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemCurrent,
                            &stats->reserved);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemHigh,
                            &stats->reservedHigh);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemCurrent,
                            &stats->used);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemHigh,
                            &stats->usedHigh);
}

// Reset watermarks to current value
void resetStatistics(cudaMemPool_t memPool) {
    cuuint64_t value = 0;
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, &value);
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, &value);
}
```

### 6.6.4 Memory Reuse Policies

The allocator attempts to reuse memory freed via `cudaFreeAsync()` before allocating more from the OS. Three controllable policies govern reuse:

#### cudaMemPoolReuseFollowEventDependencies (enabled by default)

Before allocating new memory, the allocator examines dependency information from CUDA events and tries to reuse memory freed in another stream:

```cpp
cudaMallocAsync(&ptr, size, originalStream);
kernel<<<..., originalStream>>>(ptr, ...);
cudaFreeAsync(ptr, originalStream);
cudaEventRecord(event, originalStream);

// Waiting on the event that captures the free
cudaStreamWaitEvent(otherStream, event);

// The allocator can reuse ptr's memory for this allocation
cudaMallocAsync(&ptr2, size, otherStream);
```

#### cudaMemPoolReuseAllowOpportunistic (enabled by default)

The allocator examines freed allocations to see if the GPU has already passed the free operation's point in execution (even without an explicit event dependency):

```cpp
cudaMallocAsync(&ptr, size, originalStream);
kernel<<<..., originalStream>>>(ptr, ...);
cudaFreeAsync(ptr, originalStream);

// If originalStream has progressed past the free on the GPU,
// the allocator may reuse the memory opportunistically
cudaMallocAsync(&ptr2, size, otherStream);
```

Disabling this policy does not prevent reuse from `cudaMemPoolReuseFollowEventDependencies` or from stream synchronization.

#### cudaMemPoolReuseAllowInternalDependencies (enabled by default)

When the driver fails to allocate more physical memory from the OS, it looks for memory whose availability depends on another stream's pending progress. If found, the driver inserts the required dependency:

```cpp
// If the driver fails to allocate more physical memory,
// it may effectively insert a cudaStreamWaitEvent to ensure
// the allocating stream waits for the original stream to
// complete its use of the memory.
cudaMallocAsync(&ptr2, size, otherStream);
```

#### Disabling Reuse Policies

```cpp
int disable = 0;
cudaMemPoolSetAttribute(memPool, cudaMemPoolReuseAllowOpportunistic, &disable);
cudaMemPoolSetAttribute(memPool, cudaMemPoolReuseAllowInternalDependencies, &disable);
```

Reasons to disable:
- **Opportunistic reuse** introduces run-to-run variance based on CPU/GPU execution interleaving.
- **Internal dependency insertion** can serialize work in unexpected and non-deterministic ways.

### 6.6.5 Synchronization API Integration

When the user calls any CUDA synchronization API (`cudaStreamSynchronize`, `cudaEventSynchronize`, `cudaDeviceSynchronize`), the driver:

1. Waits for asynchronous work to complete.
2. Determines which frees are guaranteed complete.
3. Makes those freed allocations available for reuse regardless of stream or disabled policies.
4. Checks `cudaMemPoolAttrReleaseThreshold` and releases excess physical memory.

---

## 6.7 Addendums and Limitations

### 6.7.1 cudaMemcpyAsync Device Sensitivity

Any async memcpy involving memory from `cudaMallocAsync` should be done using the specified stream's context as the calling thread's current context. This is not necessary for `cudaMemcpyPeerAsync`, which references the device primary contexts directly.

### 6.7.2 cudaPointerGetAttributes Query

Invoking `cudaPointerGetAttributes()` on an allocation after `cudaFreeAsync()` has been called on it results in **undefined behavior**, even if the allocation is still accessible from a stream.

### 6.7.3 cudaGraphAddMemsetNode

`cudaGraphAddMemsetNode()` does not work with memory allocated via the stream-ordered allocator. However, memsets of such allocations can be captured via stream capture.

### 6.7.4 Pointer Attributes

`cudaPointerGetAttributes()` works on stream-ordered allocations, but:

- Since stream-ordered allocations are not context-associated, querying `CU_POINTER_ATTRIBUTE_CONTEXT` succeeds but returns NULL.
- Use `CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL` to determine the allocation's device.
- `CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE` (CUDA 11.3+) can be used to identify which pool an allocation came from (useful for IPC debugging).

### 6.7.5 CPU Virtual Memory

Avoid setting VRAM limitations with `ulimit -v` when using the stream-ordered memory allocator -- this is not supported.

### 6.7.6 IPC Handle Type Query

```cpp
int poolSupportedHandleTypes = 0;
if (driverVersion >= 11030) {  // CUDA 11.3+
    cudaDeviceGetAttribute(&poolSupportedHandleTypes,
                           cudaDevAttrMemoryPoolSupportedHandleTypes, device);
}
if (poolSupportedHandleTypes & cudaMemHandleTypePosixFileDescriptor) {
    // Pools on this device can be created with POSIX fd-based IPC
}
```

---

## Quick Reference: API Summary

| API | Purpose |
|-----|---------|
| `cudaMallocAsync(&ptr, size, stream)` | Allocate memory in stream order |
| `cudaFreeAsync(ptr, stream)` | Free memory in stream order |
| `cudaMallocFromPoolAsync(&ptr, size, pool, stream)` | Allocate from a specific pool |
| `cudaDeviceGetDefaultMempool(&pool, device)` | Get the default pool for a device |
| `cudaMemPoolCreate(&pool, &props)` | Create an explicit pool |
| `cudaMemPoolDestroy(pool)` | Destroy an explicit pool |
| `cudaDeviceSetMempool(device, pool)` | Set the current pool for a device |
| `cudaDeviceGetMempool(&pool, device)` | Get the current pool for a device |
| `cudaMemPoolSetAccess(pool, &desc, count)` | Set device accessibility |
| `cudaMemPoolGetAccess(&flags, pool, &location)` | Query device accessibility |
| `cudaMemPoolSetAttribute(pool, attr, &value)` | Set pool attribute |
| `cudaMemPoolGetAttribute(pool, attr, &value)` | Get pool attribute |
| `cudaMemPoolTrimTo(pool, minBytesToKeep)` | Release unused memory |
| `cudaMemPoolExportToShareableHandle(...)` | Export pool for IPC |
| `cudaMemPoolImportFromShareableHandle(...)` | Import pool for IPC |
| `cudaMemPoolExportPointer(&data, ptr)` | Export pointer for IPC |
| `cudaMemPoolImportPointer(&ptr, pool, &data)` | Import pointer for IPC |
