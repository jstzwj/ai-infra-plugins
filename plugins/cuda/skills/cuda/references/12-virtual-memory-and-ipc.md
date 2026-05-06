# Chapter 12: Virtual Memory Management and Interprocess Communication

CUDA provides low-level virtual memory management APIs (Driver API) that give fine-grained control over address space layout, memory sharing between processes, and advanced features such as multicast memory and compressible allocations. The Interprocess Communication (IPC) APIs enable sharing of device memory between host processes.

## 12.1 Virtual Memory Management

The Virtual Memory Management API allows applications to manage GPU virtual address (VA) space explicitly. This is useful for sparse allocations, memory sharing across processes, and scalable peer-to-peer access patterns that exceed the limitations of the legacy `cudaDeviceEnablePeerAccess` mechanism (which supports at most 8 peer connections on non-NVSwitch topologies).

### 12.1.1 Querying Support

Before using virtual memory management APIs, verify that the device supports them:

```cpp
// Check basic virtual address management support
int supported;
cuDeviceGetAttribute(&supported,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device);
if (!supported) {
    // Device does not support virtual memory management
}

// Check fabric support (NVLink / PCIe fabric for multicast and cross-node)
int fabricSupported;
cuDeviceGetAttribute(&fabricSupported,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device);

// Check multicast support
int mcSupported;
cuDeviceGetAttribute(&mcSupported,
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device);
```

You can also query allocation granularity requirements:

```cpp
// Query allocation granularity for a given allocation type
CUmemAllocationProp prop = {};
prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
prop.location.id = device;

size_t granularity;
cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
// size must be a multiple of granularity for cuMemCreate
```

### 12.1.2 Unicast Memory Sharing (5 Steps)

Unicast memory sharing allows one allocation to be mapped into the virtual address space of one or more devices. The process involves creating a shareable allocation handle, exchanging it between processes, reserving VA space, mapping the allocation, and setting access permissions.

#### Step 1: Allocate and Export

Create a physical allocation and export it as a shareable OS handle.

```cpp
#include <cuda.h>
#include <fcntl.h>
#include <unistd.h>

const size_t size = 1024 * 1024; // 1 MB
const size_t alignment = 0;       // default alignment

// Describe the allocation properties
CUmemAllocationProp allocProp = {};
allocProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
allocProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
allocProp.location.id = 0; // device 0

// Optionally request a specific handle type
allocProp.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

// Create the physical allocation
CUmemGenericAllocationHandle handle;
CUresult res = cuMemCreate(&handle, size, &allocProp, 0 /* flags */);
if (res != CUDA_SUCCESS) {
    fprintf(stderr, "cuMemCreate failed: %d\n", res);
}

// Export to a shareable OS handle
CUmemGenericAllocationHandle shHandle;
int osHandle = -1;
res = cuMemExportToShareableHandle(
    &osHandle,
    handle,
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
    0 /* flags */
);
// osHandle is now a file descriptor that can be sent to another process
// via UNIX domain socket, fork(), etc.
```

For Windows, use `CU_MEM_HANDLE_TYPE_WIN32` or `CU_MEM_HANDLE_TYPE_WIN32_KMT`:

```cpp
// Windows variant
HANDLE winHandle = NULL;
cuMemExportToShareableHandle(
    &winHandle,
    handle,
    CU_MEM_HANDLE_TYPE_WIN32,
    0
);
```

#### Step 2: Share and Import

In the receiving process, import the shareable handle back into a CUDA allocation handle.

```cpp
// Receive osHandle via IPC mechanism (e.g., recvmsg on UNIX domain socket)
// Then import it:

CUmemGenericAllocationHandle importHandle;
CUresult res = cuMemImportFromShareableHandle(
    &importHandle,
    (void*)(intptr_t)osHandle,
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
);
// Close the OS file descriptor after import; CUDA has its own reference
close(osHandle);
```

#### Step 3: Reserve Virtual Address Space and Map

Reserve a range of virtual address space on the target device, then map the physical allocation into it.

```cpp
CUdeviceptr dptr;
CUresult res = cuMemAddressReserve(&dptr, size, alignment, 0 /* addr */, 0 /* flags */);
if (res != CUDA_SUCCESS) {
    fprintf(stderr, "cuMemAddressReserve failed\n");
}

// Map the imported physical allocation at the reserved VA range
res = cuMemMap(dptr, size, 0 /* offset */, importHandle, 0 /* flags */);
if (res != CUDA_SUCCESS) {
    fprintf(stderr, "cuMemMap failed\n");
}
```

The `offset` parameter allows mapping a sub-range of the physical allocation. This is useful for sparse residency patterns.

#### Step 4: Set Access Permissions

By default, mapped memory is inaccessible. You must explicitly grant access to specific devices.

```cpp
CUmemAccessDesc accessDesc = {};
accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
accessDesc.location.id = 0; // grant access to device 0
accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

CUresult res = cuMemSetAccess(dptr, size, &accessDesc, 1 /* count */);
if (res != CUDA_SUCCESS) {
    fprintf(stderr, "cuMemSetAccess failed\n");
}
```

You can grant access to multiple locations at once:

```cpp
// Grant read-write access to two devices
CUmemAccessDesc accessDescs[2];
accessDescs[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
accessDescs[0].location.id = 0;
accessDescs[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
accessDescs[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
accessDescs[1].location.id = 1;
accessDescs[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

cuMemSetAccess(dptr, size, accessDescs, 2);
```

Other access flags:
- `CU_MEM_ACCESS_FLAGS_PROT_NONE` -- no access (default after mapping)
- `CU_MEM_ACCESS_FLAGS_PROT_READWRITE` -- full read-write access
- `CU_MEM_ACCESS_FLAGS_PROT_READ` -- read-only access

You can also retrieve the current access descriptor for a mapped region:

```cpp
CUmemAllocationProp queryProp = {};
cuMemGetAllocationInfoFromHandle(&queryProp, handle);
```

#### Step 5: Release and Cleanup

When done, unmap the VA range, release the physical allocation handle, and free the reserved VA space. Order matters: unmap before releasing the handle.

```cpp
// Unmap the VA range
cuMemUnmap(dptr, size);

// Release the physical allocation handle (does not free VA)
cuMemRelease(handle);
cuMemRelease(importHandle);

// Free the reserved virtual address range
cuMemAddressFree(dptr, size);
```

#### Complete Example: Inter-Process Memory Sharing via fork()

```cpp
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

__global__ void writeKernel(float* ptr, float val, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) ptr[idx] = val;
}

__global__ void readKernel(float* ptr, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) printf("ptr[%d] = %f\n", idx, ptr[idx]);
}

int main() {
    cuInit(0);

    const size_t N = 256;
    const size_t size = N * sizeof(float);
    size_t granularity;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    const size_t allocSize = ((size + granularity - 1) / granularity) * granularity;

    // Create allocation
    CUmemGenericAllocationHandle handle;
    cuMemCreate(&handle, allocSize, &prop, 0);

    // Export
    int fd;
    cuMemExportToShareableHandle(&fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);

    pid_t pid = fork();
    if (pid == 0) {
        // Child process: import and read
        CUmemGenericAllocationHandle childHandle;
        cuMemImportFromShareableHandle(&childHandle, (void*)(intptr_t)fd,
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
        close(fd);

        CUdeviceptr childPtr;
        cuMemAddressReserve(&childPtr, allocSize, 0, 0, 0);
        cuMemMap(childPtr, allocSize, 0, childHandle, 0);

        CUmemAccessDesc access = {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = 0;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        cuMemSetAccess(childPtr, allocSize, &access, 1);

        // Read data written by parent
        readKernel<<<1, 256>>>((float*)childPtr, N);
        cudaDeviceSynchronize();

        cuMemUnmap(childPtr, allocSize);
        cuMemRelease(childHandle);
        cuMemAddressFree(childPtr, allocSize);
        exit(0);
    } else {
        // Parent process: map and write
        CUdeviceptr parentPtr;
        cuMemAddressReserve(&parentPtr, allocSize, 0, 0, 0);
        cuMemMap(parentPtr, allocSize, 0, handle, 0);

        CUmemAccessDesc access = {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = 0;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        cuMemSetAccess(parentPtr, allocSize, &access, 1);

        writeKernel<<<1, 256>>>((float*)parentPtr, 42.0f, N);
        cudaDeviceSynchronize();

        // Signal child (in practice, use a synchronization primitive)
        waitpid(pid, NULL, 0);

        cuMemUnmap(parentPtr, allocSize);
        cuMemRelease(handle);
        cuMemAddressFree(parentPtr, allocSize);
        close(fd);
    }
    return 0;
}
```

### 12.1.3 Multicast Memory Sharing

Multicast memory leverages NVLink SHARP (Scalable Hierarchical Aggregation Reduction Protocol) to enable efficient broadcast and reduction operations across multiple devices. This avoids the need for separate unicast copies to each device.

Multicast memory is particularly useful for:
- Broadcasting the same read-only dataset to all GPUs (e.g., embedding tables)
- Performing reductions across GPUs with hardware support
- Reducing memory footprint when all devices need the same data

#### Creating and Using Multicast Memory

```cpp
#include <cuda.h>

void setupMulticast(int* devices, int numDevices, size_t size) {
    size_t granularity;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = devices[0];
    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    const size_t allocSize = ((size + granularity - 1) / granularity) * granularity;

    // Step 1: Create the multicast object
    CUmemGenericAllocationHandle mcHandle;
    CUresult res = cuMulticastCreate(&mcHandle, numDevices, allocSize);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuMulticastCreate failed: %d\n", res);
        return;
    }

    // Step 2: Add each participating device
    for (int i = 0; i < numDevices; i++) {
        res = cuMulticastAddDevice(mcHandle, devices[i]);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "cuMulticastAddDevice(%d) failed\n", devices[i]);
        }
    }

    // Step 3: Create per-device allocations and bind to multicast
    CUmemGenericAllocationHandle* handles =
        (CUmemGenericAllocationHandle*)malloc(numDevices * sizeof(CUmemGenericAllocationHandle));
    CUdeviceptr* ptrs = (CUdeviceptr*)malloc(numDevices * sizeof(CUdeviceptr));

    for (int i = 0; i < numDevices; i++) {
        CUmemAllocationProp devProp = {};
        devProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        devProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        devProp.location.id = devices[i];

        cuMemCreate(&handles[i], allocSize, &devProp, 0);

        // Reserve VA on each device
        cuMemAddressReserve(&ptrs[i], allocSize, 0, 0, 0);
        cuMemMap(ptrs[i], allocSize, 0, handles[i], 0);

        // Grant access
        CUmemAccessDesc access = {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = devices[i];
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        cuMemSetAccess(ptrs[i], allocSize, &access, 1);

        // Bind this allocation into the multicast object at offset 0
        cuMulticastBindMem(mcHandle, 0 /* mcOffset */, handles[i], 0 /* memOffset */,
                           allocSize, 0 /* flags */);
    }

    // Now each device can access the multicast memory via its own ptrs[i].
    // Writes from any device are visible to all others via the multicast fabric.

    // ... use ptrs[i] on devices[i] ...

    // Cleanup: unbind, unmap, release
    for (int i = 0; i < numDevices; i++) {
        cuMulticastUnbind(mcHandle, devices[i], 0, allocSize);
        cuMemUnmap(ptrs[i], allocSize);
        cuMemRelease(handles[i]);
        cuMemAddressFree(ptrs[i], allocSize);
    }
    cuMemRelease(mcHandle);
    free(handles);
    free(ptrs);
}
```

#### Multicast Granularity

Query the minimum granularity for multicast allocations:

```cpp
size_t mcGranularity;
CUmemAllocationProp mcProp = {};
mcProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
mcProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
mcProp.location.id = device;

cuMemGetAllocationGranularity(&mcGranularity, &mcProp,
    CU_MEM_ALLOC_GRANULARITY_MINIMUM);
// All multicast sizes and offsets must be multiples of mcGranularity
```

#### Important Notes on Multicast

- All participating devices must be added before any binding.
- Once a device is added, it cannot be removed.
- The multicast handle can be exported and imported across processes using the same shareable handle mechanism as unicast memory.
- Multicast memory requires NVLink connectivity between participating devices.
- The `cuMulticastBindMem` call associates a physical allocation with the multicast object at a specified offset. All devices must bind the same offset range for the multicast to be coherent.
- Use `cuMulticastBindHandle` to bind a multicast handle into a VA range for direct access.

### 12.1.4 Compressible Memory

Devices that support memory compression can benefit from reduced memory bandwidth usage. You can request compression when creating an allocation:

```cpp
CUmemAllocationProp prop = {};
prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
prop.location.id = device;

// Request generic compression
prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;

CUmemGenericAllocationHandle handle;
CUresult res = cuMemCreate(&handle, size, &prop, 0);
```

To check if a given allocation is actually compressed (compression is a request, not a guarantee):

```cpp
CUmemAllocationProp queryProp = {};
cuMemGetAllocationInfoFromHandle(&queryProp, handle);
if (queryProp.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_GENERIC) {
    // Allocation is compressed
} else {
    // Compression was not applied (e.g., unsupported format)
}
```

Compression benefits:
- Reduced DRAM bandwidth consumption for compressible data patterns
- Higher effective bandwidth for memory-bound workloads
- The GPU hardware handles compression/decompression transparently

### 12.1.5 Virtual Aliasing

Virtual aliasing allows multiple virtual addresses to map to the same physical allocation. This can be useful for:
- Implementing circular buffers with different access patterns
- Aligning the same data with different base addresses for different algorithms
- Sharing read-only data with different view offsets

```cpp
// Reserve two VA ranges
CUdeviceptr alias1, alias2;
cuMemAddressReserve(&alias1, size, 0, 0, 0);
cuMemAddressReserve(&alias2, size, 0, 0, 0);

// Map the same physical allocation to both VA ranges
cuMemMap(alias1, size, 0, handle, 0);
cuMemMap(alias2, size, 0, handle, 0);

// Set access on both
CUmemAccessDesc access = {};
access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
access.location.id = device;
access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
cuMemSetAccess(alias1, size, &access, 1);
cuMemSetAccess(alias2, size, &access, 1);

// Now alias1 and alias2 point to the same physical memory
```

#### Fence Requirements for Virtual Aliasing

When the same kernel accesses the same physical memory through different virtual aliases, the memory model requires explicit fencing to ensure visibility:

```cpp
// When writing through one alias and reading through another in the same kernel:
// Use fence.proxy.alias to ensure cross-alias visibility
asm volatile("fence.proxy.alias;");

// Or use the cooperative groups API
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void aliasKernel(float* writeAlias, float* readAlias, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        writeAlias[idx] = 42.0f;

        // Ensure the write is visible through the other alias
        asm volatile("fence.proxy.alias;");

        // Now safe to read from the other alias
        float val = readAlias[idx]; // Guaranteed to see 42.0f
    }
}
```

Key rules for virtual aliasing:
- Different kernels accessing different aliases do not require `fence.proxy.alias` (normal kernel completion ordering applies).
- Within the same kernel, any write-then-read across aliases requires `fence.proxy.alias`.
- This fence applies to the `proxy::generic` memory proxy and ensures visibility across the virtual aliasing layer.

### 12.1.6 Memory Migration with Virtual Memory API

You can migrate physical backing of a VA range from one device to another by unmapping and remapping:

```cpp
// Unmap from source device's allocation
cuMemUnmap(dptr, size);

// Map a new allocation from the destination device
CUmemAllocationProp dstProp = {};
dstProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
dstProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
dstProp.location.id = dstDevice;

CUmemGenericAllocationHandle dstHandle;
cuMemCreate(&dstHandle, size, &dstProp, 0);
cuMemMap(dptr, size, 0, dstHandle, 0);

// Update access permissions
CUmemAccessDesc access = {};
access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
access.location.id = dstDevice;
access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
cuMemSetAccess(dptr, size, &access, 1);
```

Note that this does not copy data. For data-preserving migration, you must copy the data before unmapping or use `cuMemcpy` between the old and new mappings.

### 12.1.7 Sparse Residency

The virtual memory API enables sparse residency, where only parts of a large VA range are backed by physical memory:

```cpp
// Reserve a large VA range (e.g., 10 GB)
CUdeviceptr sparsePtr;
cuMemAddressReserve(&sparsePtr, 10ULL * 1024 * 1024 * 1024, 0, 0, 0);

// Only map and back a small portion (e.g., 64 MB at offset 1 GB)
size_t chunkSize = 64 * 1024 * 1024;
CUmemGenericAllocationHandle chunkHandle;
CUmemAllocationProp prop = {};
prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
prop.location.id = device;
cuMemCreate(&chunkHandle, chunkSize, &prop, 0);

// Map at offset 1 GB within the VA range
CUdeviceptr mapAddr = sparsePtr + (1ULL * 1024 * 1024 * 1024);
cuMemMap(mapAddr, chunkSize, 0, chunkHandle, 0);

CUmemAccessDesc access = {};
access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
access.location.id = device;
access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
cuMemSetAccess(mapAddr, chunkSize, &access, 1);

// Unmapped regions will fault on access (or return zero depending on access flags)
// This is useful for sparse data structures, sparse tensors, etc.
```

### 12.1.8 Querying Allocation Properties

You can retrieve properties of existing allocations:

```cpp
// Get properties from a handle
CUmemAllocationProp prop = {};
cuMemGetAllocationInfoFromHandle(&prop, handle);
// prop.type, prop.location, prop.allocFlags are now populated

// Get properties from a VA address
CUmemAllocationProp vaProp = {};
cuMemGetAllocationInfo(&vaProp, dptr);
```

---

## 12.2 Interprocess Communication (IPC)

CUDA IPC enables sharing of device memory pointers and synchronization objects between processes on the same machine. This is essential for multi-process training frameworks, data loading pipelines, and inference serving systems where different processes need to operate on the same GPU data.

### 12.2.1 Legacy IPC

The legacy IPC API provides a simple mechanism for sharing `cudaMalloc`-allocated memory between processes. It is the most commonly used IPC mechanism.

#### Exporting and Importing Memory

```cpp
// Process A: Export
float* devPtr;
cudaMalloc(&devPtr, size);

// Initialize data
cudaMemset(devPtr, 0, size);

cudaIpcMemHandle_t ipcHandle;
cudaError_t err = cudaIpcGetMemHandle(&ipcHandle, devPtr);
if (err != cudaSuccess) {
    fprintf(stderr, "cudaIpcGetMemHandle failed: %s\n",
            cudaGetErrorString(err));
}

// Send ipcHandle to Process B via IPC mechanism
// (shared memory, socket, pipe, etc.)
// The handle is a fixed-size struct that can be copied bytewise.
```

```cpp
// Process B: Import
cudaIpcMemHandle_t ipcHandle;
// Receive ipcHandle from Process A

float* devPtr;
cudaError_t err = cudaIpcOpenMemHandle(&devPtr, ipcHandle,
    cudaIpcMemLazyEnablePeerAccess);
if (err != cudaSuccess) {
    fprintf(stderr, "cudaIpcOpenMemHandle failed: %s\n",
            cudaGetErrorString(err));
}

// devPtr is now usable on Process B's GPU
// Reads and writes are coherent with Process A's view

// When done:
cudaIpcCloseMemHandle(devPtr);
```

#### IPC Event Sharing

CUDA events can also be shared across processes for cross-process synchronization:

```cpp
// Process A: Create and record event
cudaEvent_t event;
cudaEventCreateWithFlags(&event, cudaEventInterprocess | cudaEventDisableTiming);
cudaEventRecord(event, stream);

cudaIpcEventHandle_t eventHandle;
cudaIpcGetEventHandle(&eventHandle, event);

// Send eventHandle to Process B
```

```cpp
// Process B: Import and wait on event
cudaIpcEventHandle_t eventHandle;
// Receive eventHandle from Process A

cudaEvent_t event;
cudaIpcOpenEventHandle(&event, eventHandle);
cudaStreamWaitEvent(stream, event, 0);
// Stream B now waits until Process A records the event
```

#### Limitations of Legacy IPC

1. **Linux only** -- Not supported on Windows.
2. **Not compatible with `cudaMallocManaged`** -- Unified Memory pointers cannot be shared via legacy IPC. Use the Virtual Memory Management API instead.
3. **Not compatible with `cudaMallocAsync`** -- Stream-ordered allocations from memory pools require the IPC pool mechanism (see below).
4. **Limited to 1-to-1 sharing** -- The same handle can be imported by multiple processes, but each import creates a separate mapping.
5. **No fine-grained access control** -- Imported memory is always fully readable and writable.
6. **`cudaIpcMemLazyEnablePeerAccess`** -- The flag automatically enables peer access if the importing device is different from the exporting device. Use `cudaIpcMemEnablePeerAccess` to require explicit peer access setup.

### 12.2.2 IPC with Memory Pools (Stream-Ordered Allocator)

CUDA 11.3+ introduces IPC support for memory pool allocations created with `cudaMallocAsync`. This enables sharing of stream-ordered memory between processes.

#### Exporting and Importing Memory Pools

```cpp
// Process A: Export memory pool
cudaMemPool_t pool;
cudaDeviceGetDefaultMemPool(&pool, 0); // or a custom pool

// Export the pool to a shareable handle
cudaMemPoolProps poolProps;
cudaMemPoolGetAttribute(pool, cudaMemPoolHandle, &poolProps);

cudaMemPool_t exportPoolHandle;
// On Linux:
int shareableFd;
cudaMemPoolExportToShareableHandle(&shareableFd, pool,
    cudaMemHandleTypePosixFileDescriptor, 0);

// On Windows:
// HANDLE shareableHandle;
// cudaMemPoolExportToShareableHandle(&shareableHandle, pool,
//     cudaMemHandleTypeWin32, 0);

// Send shareableFd to Process B
```

```cpp
// Process B: Import memory pool
int shareableFd;
// Receive shareableFd from Process A

cudaMemPool_t importedPool;
cudaMemPoolImportFromShareableHandle(&importedPool, (void*)(intptr_t)shareableFd,
    cudaMemHandleTypePosixFileDescriptor, 0);

// Now allocations from importedPool are accessible in Process B
```

#### Exporting and Importing Individual Pointers

Even after a pool is shared, individual pointers allocated from it need to be explicitly exported and imported:

```cpp
// Process A: Export a specific pointer
float* devPtr;
cudaMallocAsync(&devPtr, size, stream);
// ... fill data ...

cudaMemPoolPtrExportData exportData;
cudaMemPoolExportPointer(&exportData, devPtr);

// Send exportData (fixed-size struct) to Process B
```

```cpp
// Process B: Import the pointer
cudaMemPoolPtrExportData exportData;
// Receive exportData from Process A

float* devPtr;
cudaMemPoolImportPointer(&devPtr, importedPool, &exportData);

// devPtr is now usable in Process B
// When done, free it:
cudaFreeAsync(devPtr, stream);
```

#### Complete Example: IPC Memory Pool Sharing

```cpp
// === Process A (Producer) ===
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>

int main() {
    cudaSetDevice(0);

    // Get default memory pool
    cudaMemPool_t pool;
    cudaDeviceGetDefaultMemPool(&pool, 0);

    // Export pool
    int poolFd;
    cudaMemPoolExportToShareableHandle(&poolFd, pool,
        cudaMemHandleTypePosixFileDescriptor, 0);

    // Allocate from pool
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    float* data;
    size_t N = 1024;
    cudaMallocAsync(&data, N * sizeof(float), stream);

    // Fill data
    for (int i = 0; i < N; i++) {
        float val = (float)i;
        cudaMemcpyAsync(&data[i], &val, sizeof(float),
                         cudaMemcpyHostToDevice, stream);
    }

    // Export pointer
    cudaMemPoolPtrExportData ptrExportData;
    cudaMemPoolExportPointer(&ptrExportData, data);

    // Send poolFd and ptrExportData to Process B via IPC
    // ... (application-specific IPC mechanism) ...

    cudaStreamSynchronize(stream);

    // Keep alive until Process B signals done
    sleep(10);

    cudaFreeAsync(data, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    close(poolFd);
    return 0;
}
```

```cpp
// === Process B (Consumer) ===
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    cudaSetDevice(0);

    // Receive poolFd and ptrExportData from Process A
    int poolFd; // ... received via IPC
    cudaMemPoolPtrExportData ptrExportData; // ... received via IPC

    // Import pool
    cudaMemPool_t pool;
    cudaMemPoolImportFromShareableHandle(&pool, (void*)(intptr_t)poolFd,
        cudaMemHandleTypePosixFileDescriptor, 0);

    // Import pointer
    float* data;
    cudaMemPoolImportPointer(&data, pool, &ptrExportData);

    // Read data
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float hostData[1024];
    cudaMemcpyAsync(hostData, data, 1024 * sizeof(float),
                     cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (int i = 0; i < 10; i++) {
        printf("data[%d] = %f\n", i, hostData[i]);
    }

    cudaFreeAsync(data, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    return 0;
}
```

### 12.2.3 IPC Synchronization Patterns

#### Cross-Process Fence with Events

```cpp
// Process A: Signal completion
cudaEvent_t doneEvent;
cudaEventCreateWithFlags(&doneEvent,
    cudaEventInterprocess | cudaEventDisableTiming);
// ... launch kernels, copies ...
cudaEventRecord(doneEvent, stream);

cudaIpcEventHandle_t eventHandle;
cudaIpcGetEventHandle(&eventHandle, doneEvent);
// Send eventHandle to Process B
```

```cpp
// Process B: Wait for Process A
cudaIpcEventHandle_t eventHandle;
// Receive from Process A

cudaEvent_t doneEvent;
cudaIpcOpenEventHandle(&doneEvent, eventHandle);

cudaStream_t stream;
cudaStreamCreate(&stream);
cudaStreamWaitEvent(stream, doneEvent, 0);

// Now safe to use shared memory
kernel<<<grid, block, 0, stream>>>(sharedData);
```

#### Spin-Wait on Shared Memory Flag (Polling)

For ultra-low-latency synchronization when event overhead is too high:

```cpp
// Shared flag in IPC memory
__global__ void setFlag(volatile int* flag) {
    *flag = 1;
    __threadfence_system(); // Ensure visibility across devices
}

__global__ void waitOnFlag(volatile int* flag) {
    while (*flag == 0) {
        // Spin-wait; use __ldcv to reduce memory pressure
    }
}
```

Note: Spin-waiting consumes GPU resources. Use this pattern only when latency is critical and the wait duration is expected to be very short.

### 12.2.4 Choosing Between IPC Mechanisms

| Feature | Legacy IPC | Virtual Memory API | IPC Memory Pools |
|---------|-----------|--------------------|------------------|
| Allocation type | `cudaMalloc` | `cuMemCreate` | `cudaMallocAsync` |
| Access control | None | Per-device read/write | Via pool properties |
| Sparse residency | No | Yes | No |
| Multicast | No | Yes | No |
| Compressible | No | Yes | No |
| Virtual aliasing | No | Yes | No |
| Linux only | Yes | Yes | Yes |
| Stream-ordered | No | No | Yes |
| Max peer connections | 8 (non-NVSwitch) | Unlimited | Unlimited |

### 12.2.5 Security Considerations

- File descriptors (POSIX) and Windows handles must be transmitted securely. On Linux, use UNIX domain sockets with `SCM_RIGHTS` to pass file descriptors.
- Imported memory inherits the access permissions set by the exporting process.
- The importing process must run on the same machine. CUDA IPC does not work across network boundaries.
- Always close imported handles when done to avoid resource leaks.
- On multi-tenant systems, ensure that IPC handles are not leaked to untrusted processes.
