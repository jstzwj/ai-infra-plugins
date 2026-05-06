# Chapter 29: Extended GPU Memory (EGM)

Extended GPU Memory (EGM) enables GPUs to access all system memory through NVLink-C2C interconnect technology, allowing workloads to transparently utilize host memory as if it were device memory. This capability is essential for applications whose working sets exceed device memory capacity, or for architectures where the GPU and CPU share a unified memory domain.

## 29.1 Overview

EGM leverages NVLink-C2C (Chip-to-Chip) to provide the GPU with high-bandwidth, low-latency access to the full system memory. Rather than copying data between host and device explicitly, the GPU can directly address and read/write host-side memory. This eliminates the need for manual `cudaMemcpy` transfers in many scenarios and simplifies programming models for large-scale workloads.

### Supported Platforms

EGM operates across three deployment topologies:

| Topology | Description |
|---|---|
| Single-node, single-GPU | One GPU accesses host memory on the same node via NVLink-C2C. Useful for workloads that exceed a single GPU's memory. |
| Single-node, multi-GPU | Multiple GPUs within one node each access shared host memory. NUMA placement becomes critical for performance. |
| Multi-node, multi-GPU | GPUs across multiple nodes access their respective local host memory. Each node's GPUs see only local host memory, but cross-node access can be orchestrated through MPI or NCCL. |

### Key Characteristics

- **Transparent access**: The GPU addresses host memory using the same load/store instructions as device memory, albeit with higher latency.
- **Cache-coherent**: NVLink-C2C maintains cache coherence between CPU and GPU accesses to the same memory regions.
- **NUMA-aware**: Performance depends heavily on which NUMA node the memory is allocated on relative to the GPU's socket affinity.
- **No pinned memory requirement**: Unlike traditional zero-copy memory (`cudaHostAlloc` with `cudaHostAllocMapped`), EGM uses the virtual memory management API and does not require the host memory to be pinned through the CUDA runtime.

### Enabling EGM Support

Verify that the device supports EGM by querying the relevant attributes:

```cpp
CUdevice device;
cuDeviceGet(&device, 0);

// Check whether the GPU can access host memory directly
int egmSupported = 0;
cuDeviceGetAttribute(&egmSupported,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device);

// A valid NUMA node ID (>= 0) indicates the GPU is NUMA-affine
// and EGM is available on this platform.
if (egmSupported >= 0) {
    printf("GPU is on NUMA node %d -- EGM is available\n", egmSupported);
} else {
    printf("EGM not available on this platform\n");
}
```

## 29.2 Socket Identifiers

In multi-socket systems, the GPU is physically connected to a specific CPU socket (NUMA node). Knowing which socket the GPU is attached to is critical for memory placement decisions. If host memory is allocated on a remote NUMA node relative to the GPU, the GPU's accesses must traverse an inter-socket link (e.g., UPI on Intel, Infinity Fabric on AMD), which adds latency and reduces bandwidth.

### Querying the NUMA Node

Use `CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID` (Driver API) or `cudaDevAttrHostNumaId` (Runtime API) to determine the GPU's NUMA affinity:

```cpp
// Driver API
int numaNodeId;
cuDeviceGetAttribute(&numaNodeId,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device);
printf("GPU %d is on NUMA node %d\n", device, numaNodeId);
```

```cpp
// Runtime API
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, deviceId);
printf("GPU %d is on NUMA node %d\n", deviceId, prop.hostNumaId);
```

### Multi-Socket Example

On a dual-socket system with 4 GPUs (2 per socket), the topology might look like this:

```
Socket 0 (NUMA node 0)     Socket 1 (NUMA node 1)
  +----+  +----+             +----+  +----+
  |GPU0|  |GPU1|             |GPU2|  |GPU3|
  +----+  +----+             +----+  +----+
```

In this scenario, allocating host memory on NUMA node 0 for GPU 2 and GPU 3 would result in cross-socket traffic. The correct approach is to allocate host memory on NUMA node 1 for those GPUs.

```cpp
// Discover the topology and report affinity
int deviceCount;
cuDeviceGetCount(&deviceCount);

for (int i = 0; i < deviceCount; i++) {
    CUdevice dev;
    cuDeviceGet(&dev, i);

    int numaId;
    cuDeviceGetAttribute(&numaId,
        CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, dev);

    printf("GPU %d -> NUMA node %d\n", i, numaId);
}
```

## 29.3 Allocators

EGM allocations are created using the CUDA virtual memory management API or the memory pool API. Both approaches require specifying that the allocation's location is on the host (`CU_MEM_LOCATION_TYPE_HOST` / `cudaMemLocationTypeHost`) and providing the NUMA node ID.

### Using cuMemCreate (Driver API)

The `cuMemCreate` function creates a physical memory allocation with explicit placement control. The allocation handle can then be mapped into the GPU's virtual address space.

```cpp
#include <cuda.h>
#include <numa.h>
#include <numaif.h>
#include <stdio.h>

CUmemGenericAllocationHandle allocateEgMemory(size_t size, int numaNodeId) {
    // Align size to allocation granularity
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_HOST;
    prop.location.id = numaNodeId;

    size_t granularity;
    cuMemGetAllocationGranularity(&granularity, &prop,
        CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    size_t alignedSize = ((size + granularity - 1) / granularity) * granularity;

    // Create the physical allocation on the host NUMA node
    CUmemGenericAllocationHandle handle;
    CUresult res = cuMemCreate(&handle, alignedSize, &prop, 0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemCreate failed: %d\n", res);
        return CUmemGenericAllocationHandle(0);
    }

    printf("Allocated %zu bytes on NUMA node %d (granularity: %zu)\n",
           alignedSize, numaNodeId, granularity);
    return handle;
}
```

Once the allocation handle is created, map it into the GPU address space:

```cpp
void* mapEgMemoryToDevice(CUmemGenericAllocationHandle handle,
                           size_t size, CUdevice device) {
    // Reserve virtual address space on the device
    CUdeviceptr dptr;
    cuMemAddressReserve(&dptr, size, 0, 0, 0);

    // Map the physical allocation into the reserved VA range
    cuMemMap(dptr, size, 0, handle, 0);

    // Set access permissions -- allow the device to read/write
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    cuMemSetAccess(dptr, size, &accessDesc, 1);

    return (void*)dptr;
}
```

Full allocation and usage example:

```cpp
int main() {
    cuInit(0);
    CUdevice device;
    cuDeviceGet(&device, 0);
    CUcontext ctx;
    cuCtxCreate(&ctx, 0, device);

    // Determine GPU's NUMA affinity
    int numaNodeId;
    cuDeviceGetAttribute(&numaNodeId,
        CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device);

    const size_t size = 256ULL * 1024 * 1024; // 256 MB

    // Step 1: Create the physical allocation on host memory
    CUmemGenericAllocationHandle handle = allocateEgMemory(size, numaNodeId);

    // Step 2: Map into device address space
    void* devPtr = mapEgMemoryToDevice(handle, size, device);

    // Step 3: Use the memory in kernels -- GPU directly accesses host memory
    // (Kernel launch omitted for brevity)

    // Step 4: Cleanup
    cuMemUnmap((CUdeviceptr)devPtr, size);
    cuMemAddressFree((CUdeviceptr)devPtr, size);
    cuMemRelease(handle);
    cuCtxDestroy(ctx);

    return 0;
}
```

### Using cudaMemPoolCreate (Runtime API)

The memory pool API provides a higher-level interface. Pools with host-side placement enable EGM through the stream-ordered memory allocator.

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

cudaError_t createEgMemoryPool(cudaMemPool_t* pool, int numaNodeId) {
    cudaMemPoolProps poolProps = {};
    poolProps.allocType = cudaMemAllocationTypePinned;
    poolProps.location.type = cudaMemLocationTypeHost;
    poolProps.location.id = numaNodeId;

    // Optionally set handle type for IPC sharing
    poolProps.handleTypes = cudaMemHandleTypeNone;

    cudaError_t err = cudaMemPoolCreate(pool, &poolProps);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolCreate failed: %s\n",
                cudaGetErrorString(err));
    }
    return err;
}
```

Usage with stream-ordered allocation:

```cpp
int main() {
    int deviceId = 0;
    cudaSetDevice(deviceId);

    // Query NUMA affinity
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    int numaNodeId = prop.hostNumaId;

    // Create an EGM memory pool
    cudaMemPool_t pool;
    createEgMemoryPool(&pool, numaNodeId);

    // Allocate from the pool (stream-ordered)
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const size_t size = 128ULL * 1024 * 1024; // 128 MB
    void* devPtr = nullptr;
    cudaMallocFromPoolAsync(&devPtr, size, pool, stream);

    // Use the memory -- GPU accesses host memory transparently
    // myKernel<<<grid, block, 0, stream>>>((float*)devPtr, ...);

    cudaStreamSynchronize(stream);

    // Free back to pool
    cudaFreeAsync(devPtr, stream);

    // Cleanup
    cudaStreamDestroy(stream);
    cudaMemPoolDestroy(pool);

    return 0;
}
```

### Setting Pool Properties for NUMA-Aware Allocation

You can configure pool behavior to optimize for EGM workloads:

```cpp
void configureEgPool(cudaMemPool_t pool) {
    // Disable reuse across streams to enforce NUMA locality
    int64_t threshold = 0;
    cudaMemPoolSetAttribute(pool,
        cudaMemPoolReuseFollowEventDependencies, &threshold);

    // Set the release threshold -- keep freed blocks available
    // for reuse rather than returning them to the OS immediately
    uint64_t releaseThreshold = 256ULL * 1024 * 1024; // 256 MB
    cudaMemPoolSetAttribute(pool,
        cudaMemPoolReleaseThreshold, &releaseThreshold);
}
```

### Comparing Allocation Approaches

| Feature | `cuMemCreate` (Driver) | `cudaMemPoolCreate` (Runtime) |
|---|---|---|
| Placement control | Full control via `CUmemAllocationProp` | Via `cudaMemPoolProps` |
| VA management | Manual (`cuMemAddressReserve`, `cuMemMap`, etc.) | Automatic |
| Stream-ordered | No (explicit) | Yes (`cudaMallocFromPoolAsync`) |
| IPC support | Yes (export/import handles) | Yes (with appropriate `handleTypes`) |
| Granularity alignment | Manual | Automatic |
| Complexity | Low-level, verbose | High-level, simpler |

## 29.4 NUMA Best Practices

Correct NUMA placement is the single most important factor for EGM performance. When host memory is allocated on a NUMA node that is remote from the GPU, every memory access traverses the inter-socket interconnect, which can reduce effective bandwidth by 30-50% and add significant latency.

### Use CUDA_VISIBLE_DEVICES for Device Limiting

When running on a multi-GPU, multi-socket system, restrict GPU visibility to only the GPUs on the local socket. Avoid using Linux cgroups to limit device access, as they do not influence NUMA memory placement and can lead to subtle performance bugs.

```bash
# Good: restrict to GPUs on NUMA node 0
CUDA_VISIBLE_DEVICES=0,1 ./my_application

# Bad: cgroups do not help with NUMA affinity
# echo "0,1" > /sys/fs/cgroup/devices/gpu/cgroup.devices
```

When `CUDA_VISIBLE_DEVICES` is set, CUDA renumbers visible devices starting from 0, so the application code does not need to change:

```cpp
// Application uses logical device indices 0, 1, ...
// which map to physical devices listed in CUDA_VISIBLE_DEVICES
cudaSetDevice(0); // maps to first visible GPU
```

### Disable Automatic NUMA Balancing

The Linux kernel's automatic NUMA balancing daemon (`numad` / `kernel.numa_balancing`) can migrate memory pages between NUMA nodes at runtime. While this helps general-purpose workloads, it is harmful for EGM because:

1. The GPU's NVLink-C2C connection is statically routed to a specific socket's memory controller.
2. If the kernel migrates a page to a remote node, subsequent GPU accesses become cross-socket, defeating the purpose of EGM placement.
3. Page migration causes TLB shootdowns that stall both CPU and GPU accesses.

Disable automatic NUMA balancing system-wide:

```bash
# Disable via sysctl (takes effect immediately, not persistent across reboot)
sudo sysctl -w kernel.numa_balancing=0

# Make it persistent
echo "kernel.numa_balancing = 0" | sudo tee -a /etc/sysctl.d/99-disable-numa-balancing.conf
sudo sysctl --system
```

Or disable it at the kernel command line:

```
# Add to GRUB_CMDLINE_LINUX in /etc/default/grub
numa_balancing=disable

# Rebuild grub config
sudo update-grub
sudo reboot
```

### Use numactl for Binding

Use `numactl` to bind both the CPU and memory allocation to the correct NUMA node. This ensures that:
- CPU threads accessing EGM-allocated memory do so locally.
- `malloc` and `mmap` calls allocate memory on the GPU's local NUMA node.

```bash
# Bind to NUMA node 0 (matching GPUs 0 and 1)
numactl --cpunodebind=0 --membind=0 ./my_application

# Bind to NUMA node 1 (matching GPUs 2 and 3)
numactl --cpunodebind=1 --membind=1 ./my_application

# Combine with CUDA_VISIBLE_DEVICES for a complete setup
CUDA_VISIBLE_DEVICES=0,1 numactl --cpunodebind=0 --membind=0 ./my_application
```

Programmatic NUMA binding using `libnuma`:

```cpp
#include <numa.h>
#include <numaif.h>

void bindToNumaNode(int nodeId) {
    if (numa_available() == -1) {
        fprintf(stderr, "NUMA not available on this system\n");
        return;
    }

    // Bind the current thread's memory allocations to the specified node
    struct bitmask* mask = numa_allocate_nodemask();
    numa_bitmask_setbit(mask, nodeId);
    numa_set_membind(mask);
    numa_free_nodemask(mask);

    // Also bind the current thread to CPUs on that node
    numa_run_on_node(nodeId);
}
```

### Combined Best Practice Pattern

The recommended pattern for EGM applications combines all three techniques:

```bash
#!/bin/bash
# launch_egm_app.sh -- launch an EGM-aware application

NUMA_NODE=${1:-0}

# Map GPUs to NUMA nodes (adjust for your system topology)
case $NUMA_NODE in
    0) GPUS="0,1" ;;
    1) GPUS="2,3" ;;
    *) echo "Unknown NUMA node"; exit 1 ;;
esac

# 1. Disable NUMA balancing for this session
sudo sysctl -w kernel.numa_balancing=0 2>/dev/null

# 2. Set visible devices to GPUs on this NUMA node
export CUDA_VISIBLE_DEVICES=$GPUS

# 3. Bind CPU and memory with numactl
exec numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE \
    ./my_egm_application
```

### NUMA-Aware Memory Allocation with EGM

When allocating EGM from within the application, always match the NUMA node to the GPU's affinity:

```cpp
CUmemGenericAllocationHandle createNumaAwareEgAllocation(
    CUdevice device, size_t size) {

    // Step 1: Determine the GPU's NUMA node
    int numaNodeId;
    cuDeviceGetAttribute(&numaNodeId,
        CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device);

    // Step 2: Create the allocation on the matching NUMA node
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_HOST;
    prop.location.id = numaNodeId;

    size_t granularity;
    cuMemGetAllocationGranularity(&granularity, &prop,
        CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    size_t alignedSize = ((size + granularity - 1) / granularity) * granularity;

    CUmemGenericAllocationHandle handle;
    cuMemCreate(&handle, alignedSize, &prop, 0);

    return handle;
}
```

### Performance Considerations

| Factor | Recommendation |
|---|---|
| Memory placement | Always allocate on the GPU's local NUMA node |
| Access pattern | Prefer sequential / coalesced access from GPU kernels |
| Page size | Consider using huge pages (2 MB or 1 GB) to reduce TLB pressure |
| Concurrent access | Avoid simultaneous CPU and GPU writes to the same cache line |
| Bandwidth | Expect 20-40% of device memory bandwidth for local NUMA EGM accesses |
| Latency | EGM latency is ~2-5x higher than device memory; hide with occupancy |

### Monitoring NUMA Locality

Use Linux tools to verify that memory is allocated on the correct NUMA node:

```bash
# Check the NUMA topology
numactl --hardware

# Check page placement for a running process (PID)
numastat -p <PID>

# Use hwloc to visualize the hardware topology
lstopo --of txt
```

Programmatic verification using `move_pages`:

```cpp
#include <numa.h>
#include <numaif.h>
#include <stdio.h>

void checkPagePlacement(void* ptr, size_t size, int expectedNode) {
    size_t pageSize = 4096; // standard page size
    int numPages = (size + pageSize - 1) / pageSize;
    void** pages = (void**)malloc(numPages * sizeof(void*));
    int* status = (int*)malloc(numPages * sizeof(int));

    for (int i = 0; i < numPages; i++) {
        pages[i] = (char*)ptr + i * pageSize;
    }

    // Query current NUMA node for each page (no migration)
    numa_move_pages(0, numPages, pages, NULL, status, 0);

    int mismatchCount = 0;
    for (int i = 0; i < numPages; i++) {
        if (status[i] != expectedNode) {
            mismatchCount++;
        }
    }

    if (mismatchCount > 0) {
        printf("WARNING: %d/%d pages not on expected NUMA node %d\n",
               mismatchCount, numPages, expectedNode);
    } else {
        printf("All %d pages correctly placed on NUMA node %d\n",
               numPages, expectedNode);
    }

    free(pages);
    free(status);
}
```
