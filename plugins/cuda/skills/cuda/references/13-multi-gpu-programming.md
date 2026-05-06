# Chapter 13: Multi-GPU Programming

CUDA supports systems with multiple GPUs, enabling applications to scale across devices within a single node. Multi-GPU programming involves device enumeration, multi-device execution, peer-to-peer (P2P) memory access, and cross-device synchronization. Understanding the behavior of streams, events, and memory operations across device boundaries is essential for correct and performant multi-GPU code.

## 13.1 Device Enumeration and Selection

### 13.1.1 Querying Device Count

```cpp
int deviceCount;
cudaError_t err = cudaGetDeviceCount(&deviceCount);
if (err != cudaSuccess) {
    fprintf(stderr, "cudaGetDeviceCount failed: %s\n",
            cudaGetErrorString(err));
    return;
}
if (deviceCount == 0) {
    fprintf(stderr, "No CUDA-capable devices found\n");
    return;
}
printf("Found %d CUDA devices\n", deviceCount);
```

### 13.1.2 Querying Device Properties

Iterate over all devices to select the best one or to understand the system topology:

```cpp
for (int dev = 0; dev < deviceCount; dev++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    printf("Device %d: %s\n", dev, prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total global memory: %.2f GB\n",
           (double)prop.totalGlobalMem / (1024 * 1024 * 1024));
    printf("  SM count: %d\n", prop.multiProcessorCount);
    printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Clock rate: %.2f MHz\n", (double)prop.clockRate / 1000.0);
    printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
    printf("  L2 cache size: %d bytes\n", prop.l2CacheSize);
    printf("  PCI bus ID: %04x:%02x:%02x\n",
           prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

    // NVLink information (CUDA 9.0+)
    for (int i = 0; i < prop.numaNodeId; i++) {
        // Use cuDeviceGetNvSciPCIEBarInfo or cuDeviceGetP2PAttribute
        // for detailed interconnect info
    }

    // Check if device supports cooperative kernel launch
    if (prop.cooperativeLaunch) {
        printf("  Supports cooperative launch\n");
    }
    if (prop.cooperativeMultiDeviceLaunch) {
        printf("  Supports multi-device cooperative launch\n");
    }
}
```

### 13.1.3 Selecting and Setting the Current Device

CUDA uses a per-thread current device. All subsequent runtime API calls (memory allocation, kernel launches, stream operations) operate on the current device unless otherwise specified.

```cpp
int device = 0; // Select first device
cudaError_t err = cudaSetDevice(device);
if (err != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice(%d) failed: %s\n",
            device, cudaGetErrorString(err));
}

// Query the current device
int currentDevice;
cudaGetDevice(&currentDevice);
printf("Current device: %d\n", currentDevice);
```

### 13.1.4 Device Selection Heuristics

Common strategies for choosing which GPU(s) to use:

```cpp
// Strategy 1: Select device with most free memory
int bestDevice = -1;
size_t maxFreeMem = 0;
for (int dev = 0; dev < deviceCount; dev++) {
    cudaSetDevice(dev);
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    if (freeMem > maxFreeMem) {
        maxFreeMem = freeMem;
        bestDevice = dev;
    }
}
cudaSetDevice(bestDevice);
```

```cpp
// Strategy 2: Select device with highest compute capability
int bestDevice = 0;
int bestMajor = 0, bestMinor = 0;
for (int dev = 0; dev < deviceCount; dev++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    if (prop.major > bestMajor ||
        (prop.major == bestMajor && prop.minor > bestMinor)) {
        bestMajor = prop.major;
        bestMinor = prop.minor;
        bestDevice = dev;
    }
}
cudaSetDevice(bestDevice);
```

### 13.1.5 Matching CUDA and PCI Device Order

On systems with multiple GPUs, the CUDA device ordering may not match the PCI bus order. To match a specific PCI address:

```cpp
// Find CUDA device for a specific PCI address
int pciBusId = 0x03;
int pciDeviceId = 0x00;
int pciDomainId = 0x0000;

int cudaDevice;
cudaError_t err = cudaDeviceGetByPCIBusId(&cudaDevice,
    pciDomainId, pciBusId, pciDeviceId);
if (err == cudaSuccess) {
    cudaSetDevice(cudaDevice);
}
```

## 13.2 Multi-Device Execution

### 13.2.1 Basic Multi-Device Pattern

Each device operates independently. Set the current device before performing any device-specific operations:

```cpp
const int N = 1024 * 1024;
const size_t size = N * sizeof(float);

// Device 0
float *d_A0, *d_B0, *d_C0;
cudaSetDevice(0);
cudaMalloc(&d_A0, size);
cudaMalloc(&d_B0, size);
cudaMalloc(&d_C0, size);

cudaStream_t stream0;
cudaStreamCreate(&stream0);
vecAdd<<<grid, block, 0, stream0>>>(d_A0, d_B0, d_C0, N);

// Device 1
float *d_A1, *d_B1, *d_C1;
cudaSetDevice(1);
cudaMalloc(&d_A1, size);
cudaMalloc(&d_B1, size);
cudaMalloc(&d_C1, size);

cudaStream_t stream1;
cudaStreamCreate(&stream1);
vecAdd<<<grid, block, 0, stream1>>>(d_A1, d_B1, d_C1, N);

// Both kernels execute concurrently on their respective devices
// Synchronize both devices
cudaSetDevice(0);
cudaDeviceSynchronize();
cudaSetDevice(1);
cudaDeviceSynchronize();
```

### 13.2.2 Per-Device Resource Management

Memory allocations, streams, events, and other resources are associated with the device that was current at creation time. Key rules:

1. **`cudaMalloc`** always allocates on the current device.
2. **`cudaFree`** can free memory on any device; the pointer encodes the device.
3. **`cudaStreamCreate`** creates a stream on the current device.
4. **`cudaEventCreate`** creates an event on the current device.
5. **Kernel launches** execute on the device associated with the stream (or the current device for default stream).

```cpp
// Safe pattern: always set device before creating resources
void setupDevice(int deviceId, DeviceResources* res) {
    cudaSetDevice(deviceId);

    cudaMalloc(&res->data, res->size);
    cudaStreamCreate(&res->stream);
    cudaEventCreate(&res->startEvent);
    cudaEventCreate(&res->stopEvent);
}

// Cleanup can happen without switching device
void cleanupDevice(DeviceResources* res) {
    // These work regardless of current device
    cudaFree(res->data);
    cudaStreamDestroy(res->stream);
    cudaEventDestroy(res->startEvent);
    cudaEventDestroy(res->stopEvent);
}
```

### 13.2.3 Multi-Device Data Partitioning

For embarrassingly parallel workloads, partition data across devices and merge results:

```cpp
// Partition a vector addition across N GPUs
void multiGpuVecAdd(float* h_A, float* h_B, float* h_C, int totalElements) {
    int numDevices;
    cudaGetDeviceCount(&numDevices);

    int elementsPerDevice = (totalElements + numDevices - 1) / numDevices;

    float** d_A = new float*[numDevices];
    float** d_B = new float*[numDevices];
    float** d_C = new float*[numDevices];
    cudaStream_t* streams = new cudaStream_t[numDevices];

    for (int dev = 0; dev < numDevices; dev++) {
        cudaSetDevice(dev);
        int offset = dev * elementsPerDevice;
        int count = min(elementsPerDevice, totalElements - offset);
        if (count <= 0) break;

        size_t bytes = count * sizeof(float);
        cudaMalloc(&d_A[dev], bytes);
        cudaMalloc(&d_B[dev], bytes);
        cudaMalloc(&d_C[dev], bytes);
        cudaStreamCreate(&streams[dev]);

        cudaMemcpyAsync(d_A[dev], h_A + offset, bytes,
                         cudaMemcpyHostToDevice, streams[dev]);
        cudaMemcpyAsync(d_B[dev], h_B + offset, bytes,
                         cudaMemcpyHostToDevice, streams[dev]);

        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        vecAdd<<<blocks, threads, 0, streams[dev]>>>(
            d_A[dev], d_B[dev], d_C[dev], count);

        cudaMemcpyAsync(h_C + offset, d_C[dev], bytes,
                         cudaMemcpyDeviceToHost, streams[dev]);
    }

    // Synchronize all devices
    for (int dev = 0; dev < numDevices; dev++) {
        cudaSetDevice(dev);
        cudaStreamSynchronize(streams[dev]);
    }

    // Cleanup
    for (int dev = 0; dev < numDevices; dev++) {
        cudaSetDevice(dev);
        cudaFree(d_A[dev]);
        cudaFree(d_B[dev]);
        cudaFree(d_C[dev]);
        cudaStreamDestroy(streams[dev]);
    }
    delete[] d_A;
    delete[] d_B;
    delete[] d_C;
    delete[] streams;
}
```

### 13.2.4 Multi-Device Cooperative Kernels

For workloads requiring all GPUs to cooperate (e.g., global reductions with inter-device synchronization), use cooperative multi-device launch:

```cpp
// Requires devices that support cooperative multi-device launch
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
if (!prop.cooperativeMultiDeviceLaunch) {
    fprintf(stderr, "Multi-device cooperative launch not supported\n");
    return;
}

// Create launch params for each device
int numDevices = 2;
cudaLaunchParams* launchParams = new cudaLaunchParams[numDevices];
CUlaunchParams* driverLaunchParams = new CUlaunchParams[numDevices];

void* kernelParams[] = { ... };

for (int dev = 0; dev < numDevices; dev++) {
    launchParams[dev].function = (void*)myCooperativeKernel;
    launchParams[dev].gridDim = make_cudaInt3(gridX, gridY, gridZ);
    launchParams[dev].blockDim = make_cudaInt3(blockX, blockY, blockZ);
    launchParams[dev].sharedMem = 0;
    launchParams[dev].stream = streams[dev];
    launchParams[dev].kernelParams = kernelParams;
}

// Launch cooperatively across all devices
cudaLaunchCooperativeKernelMultiDevice(launchParams, numDevices,
    0 /* flags */);

// All devices synchronize internally via cooperative groups grid sync
for (int dev = 0; dev < numDevices; dev++) {
    cudaSetDevice(dev);
    cudaDeviceSynchronize();
}
```

## 13.3 Multi-Device Stream and Event Behavior

### 13.3.1 Streams Are Device-Scoped

A CUDA stream is associated with a specific device. Launching a kernel into a stream that belongs to a different device than the current device results in an error:

```cpp
cudaSetDevice(0);
cudaStream_t stream0;
cudaStreamCreate(&stream0);

cudaSetDevice(1);
// ERROR: stream0 belongs to device 0, but current device is 1
myKernel<<<grid, block, 0, stream0>>>(...);
// Returns cudaErrorInvalidResourceHandle

// Correct: create a stream on each device
cudaStream_t stream1;
cudaStreamCreate(&stream1);
myKernel<<<grid, block, 0, stream1>>>(...);  // OK
```

### 13.3.2 Memory Copies Work Across Devices

`cudaMemcpy` and `cudaMemcpyAsync` can copy between different devices' memory without explicitly enabling peer access. The runtime handles the routing internally:

```cpp
float *d_src, *d_dst;
cudaSetDevice(0);
cudaMalloc(&d_src, size);
cudaSetDevice(1);
cudaMalloc(&d_dst, size);

// Works regardless of peer access state
// The runtime uses an internal staging buffer if P2P is not available
cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice);
```

However, this cross-device copy is most efficient when peer access is enabled (see Section 13.4).

### 13.3.3 Cross-Device Event Synchronization

`cudaStreamWaitEvent()` works across devices, providing a powerful cross-device synchronization mechanism:

```cpp
// Device 0 produces data
cudaSetDevice(0);
cudaStream_t stream0;
cudaStreamCreate(&stream0);

cudaEvent_t readyEvent;
cudaEventCreate(&readyEvent);

// Launch work on device 0
producerKernel<<<grid, block, 0, stream0>>>(d_data0);
cudaEventRecord(readyEvent, stream0);

// Device 1 consumes data
cudaSetDevice(1);
cudaStream_t stream1;
cudaStreamCreate(&stream1);

// Make stream1 on device 1 wait for readyEvent on device 0
cudaStreamWaitEvent(stream1, readyEvent, 0);

// This kernel on device 1 will not execute until device 0 finishes
consumerKernel<<<grid, block, 0, stream1>>>(d_data1);
```

This pattern is the foundation for pipeline parallelism across multiple GPUs.

### 13.3.4 Default Stream Behavior Across Devices

The default stream (stream 0) is per-device. Using the default stream on device 0 does not synchronize with the default stream on device 1:

```cpp
cudaSetDevice(0);
kernelA<<<grid, block>>>(...);  // Device 0, default stream

cudaSetDevice(1);
kernelB<<<grid, block>>>(...);  // Device 1, default stream

// kernelA and kernelB execute concurrently on their respective devices
```

With per-thread default stream (`-default-stream per-thread`), each host thread has its own default stream per device, enabling more concurrency.

## 13.4 Peer-to-Peer Memory Access

Peer-to-peer (P2P) memory access allows one GPU to directly read and write another GPU's memory without staging through host memory. This requires NVLink or PCIe connectivity and must be explicitly enabled.

### 13.4.1 Checking P2P Support

```cpp
int canAccessPeer;
cudaError_t err = cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1);
if (err != cudaSuccess) {
    fprintf(stderr, "cudaDeviceCanAccessPeer query failed\n");
}
if (canAccessPeer) {
    printf("Device 0 can directly access device 1's memory\n");
} else {
    printf("P2P access not available; copies will use staging buffer\n");
}
```

You can also query P2P performance attributes:

```cpp
// Query P2P bandwidth attributes (Driver API)
CUdevice cuDev0, cuDev1;
cuDeviceGet(&cuDev0, 0);
cuDeviceGet(&cuDev1, 1);

int p2pLinkPerformance;
cuDeviceGetP2PAttribute(&p2pLinkPerformance,
    CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK, cuDev0, cuDev1);

int p2pAccessSupported;
cuDeviceGetP2PAttribute(&p2pAccessSupported,
    CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED, cuDev0, cuDev1);

int p2pNativeAtomicSupported;
cuDeviceGetP2PAttribute(&p2pNativeAtomicSupported,
    CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED, cuDev0, cuDev1);

printf("P2P performance rank: %d\n", p2pLinkPerformance);
printf("P2P access supported: %d\n", p2pAccessSupported);
printf("P2P native atomics: %d\n", p2pNativeAtomicSupported);
```

### 13.4.2 Enabling P2P Access

```cpp
int canAccess;
cudaDeviceCanAccessPeer(&canAccess, 0, 1);
if (canAccess) {
    // Enable unidirectional P2P: device 0 can access device 1's memory
    cudaSetDevice(0);
    cudaError_t err = cudaDeviceEnablePeerAccess(1, 0 /* flags */);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        // Already enabled, this is fine
        cudaGetLastError(); // Clear the error
    } else if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceEnablePeerAccess failed: %s\n",
                cudaGetErrorString(err));
    }

    // For bidirectional P2P, also enable the reverse direction
    cudaSetDevice(1);
    err = cudaDeviceEnablePeerAccess(0, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
    }
}
```

Once P2P access is enabled, kernels on device 0 can directly dereference pointers allocated on device 1:

```cpp
float* d_data_on_device1;
cudaSetDevice(1);
cudaMalloc(&d_data_on_device1, size);

// Initialize data on device 1
initializeKernel<<<grid, block>>>(d_data_on_device1, N);
cudaDeviceSynchronize();

// Now use the pointer from device 0 (P2P access enabled)
cudaSetDevice(0);
processKernel<<<grid, block>>>(d_data_on_device1, N);
// processKernel reads/writes device 1's memory directly over NVLink/PCIe
```

### 13.4.3 P2P Memory Transfers

Explicit P2P memory copies transfer data directly between two devices' memory:

```cpp
float *d_src, *d_dst;
cudaSetDevice(0);
cudaMalloc(&d_src, size);

cudaSetDevice(1);
cudaMalloc(&d_dst, size);

// Direct P2P copy (does not require peer access to be enabled)
cudaError_t err = cudaMemcpyPeer(d_dst, 1, d_src, 0, size);
if (err != cudaSuccess) {
    fprintf(stderr, "cudaMemcpyPeer failed: %s\n",
            cudaGetErrorString(err));
}
```

Asynchronous P2P copy:

```cpp
cudaSetDevice(0);
cudaStream_t stream0;
cudaStreamCreate(&stream0);

// Async P2P copy
cudaError_t err = cudaMemcpyPeerAsync(d_dst, 1, d_src, 0, size, stream0);
cudaStreamSynchronize(stream0);
```

When P2P access is enabled, `cudaMemcpy` with `cudaMemcpyDeviceToDevice` also uses direct P2P paths:

```cpp
// If peer access is enabled between device 0 and device 1:
cudaSetDevice(0); // or 1, doesn't matter for cudaMemcpy
cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice);
// Uses direct P2P path (NVLink or PCIe)
```

### 13.4.4 P2P Topology and Interconnect

Understanding the physical interconnect between GPUs is important for optimizing multi-GPU applications:

```cpp
// Query interconnect topology (Driver API)
for (int i = 0; i < deviceCount; i++) {
    for (int j = 0; j < deviceCount; j++) {
        if (i == j) continue;
        int canAccess;
        cudaDeviceCanAccessPeer(&canAccess, i, j);
        if (canAccess) {
            printf("Device %d -> Device %d: P2P available\n", i, j);
        }
    }
}
```

Common topologies:
- **NVLink** -- High-bandwidth (up to 600 GB/s aggregate on H100), low-latency direct GPU-to-GPU links. Up to 6 NVLink connections per GPU on H100.
- **PCIe** -- Standard PCIe Gen4/Gen5. Bandwidth limited to ~64 GB/s (Gen5 x16) per direction. P2P over PCIe requires IOMMU to be disabled on Linux.
- **NVSwitch** -- A switching fabric that connects all GPUs in a node with full bandwidth. Removes the 8-peer-access limitation. Systems like DGX/HGX use NVSwitch.
- **PCIe Switch** -- Some multi-GPU systems use PCIe switches for better P2P connectivity.

### 13.4.5 P2P Access Limitations

1. **Maximum peer connections**: On non-NVSwitch systems, a single GPU can have direct peer access to at most 8 other GPUs. For systems with more than 8 GPUs per node, use the Virtual Memory Management API (Section 12.1) which does not have this limitation.

2. **Unified Memory + P2P**: When using `cudaMallocManaged`, the CUDA runtime automatically handles P2P access for page migrations. Explicit P2P enablement is not needed for managed memory.

3. **Atomic operations**: Cross-device atomics are supported over NVLink but may have higher latency than local atomics. Use `cudaDeviceGetP2PAttribute` with `CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED` to check.

4. **Disabling P2P access**:
```cpp
cudaSetDevice(0);
cudaDeviceDisablePeerAccess(1);
```

### 13.4.6 Cross-Device Synchronization with thread_scope_system

When multiple GPUs access the same memory (via P2P or unified memory), standard atomics and memory fences use `thread_scope_system` to ensure visibility across devices:

```cpp
#include <cuda/atomic>

__global__ void crossDeviceSync(cuda::atomic<int, cuda::thread_scope_system>* flag,
                                 float* data, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        // Write data
        data[idx] = compute(idx);
    }

    // Ensure all writes are visible system-wide (across all GPUs)
    __threadfence_system();

    // Or use cuda::atomic with system scope
    if (idx == 0) {
        flag->store(1, cuda::memory_order_release);
    }
}

__global__ void crossDeviceWait(cuda::atomic<int, cuda::thread_scope_system>* flag,
                                 float* data, int N) {
    // Spin-wait with system scope
    while (flag->load(cuda::memory_order_acquire) == 0) {
        // Wait for the other GPU to set the flag
    }

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        float val = data[idx]; // Safe to read
    }
}
```

Memory fence scopes:
- `__threadfence()` -- All threads on the same GPU
- `__threadfence_system()` -- All threads across all GPUs and the host (system-wide)
- `__threadfence_block()` -- Only threads in the same block

Atomic scopes in `cuda::atomic`:
- `cuda::thread_scope_thread` -- Single thread
- `cuda::thread_scope_block` -- Thread block
- `cuda::thread_scope_device` -- Single device
- `cuda::thread_scope_system` -- All devices + host

## 13.5 Host IOMMU and Virtual Machines

The IOMMU (Input/Output Memory Management Unit) on the host system can affect P2P memory access between GPUs. The behavior varies by platform and configuration.

### 13.5.1 Linux Bare-Metal Systems

On Linux bare-metal systems, the IOMMU must be **disabled** for PCIe-based peer-to-peer transfers to work:

```bash
# Check if IOMMU is enabled
dmesg | grep -i iommu

# To disable IOMMU, add to kernel boot parameters:
# intel_iommu=off
# or
# amd_iommu=off

# Verify after reboot
cat /proc/cmdline
```

When IOMMU is enabled on Linux bare-metal:
- PCIe P2P `cudaMemcpyPeer` falls back to staging through host memory (slower)
- NVLink P2P is not affected by IOMMU settings
- `cudaDeviceCanAccessPeer` may report 0 for PCIe-connected GPU pairs

### 13.5.2 Virtual Machines

In virtualized environments, IOMMU (VFIO) is typically required for GPU pass-through:

```bash
# Enable IOMMU and VFIO for GPU pass-through in VM
# Kernel boot parameters:
# intel_iommu=on iommu=pt vfio-pci.ids=10de:XXXX
```

- PCIe pass-through with VFIO allows a VM to have direct access to a physical GPU.
- P2P between GPUs passed through to the same VM depends on the hypervisor and hardware support.
- Some hypervisors (e.g., libvirt with `<pci>` device assignment) support P2P between passed-through GPUs.
- SR-IOV (Single Root I/O Virtualization) for GPUs is emerging but has limited P2P support.

### 13.5.3 Windows

On Windows, the IOMMU does not impose limitations on P2P access:

- PCIe P2P works regardless of IOMMU/IOMMU settings.
- The Windows Display Driver Model (WDDM) handles P2P transparently.
- TCC (Tesla Compute Cluster) mode is recommended for compute-only GPUs on Windows for best P2P performance.

```bash
# Enable TCC mode on Windows (requires nvidia-smi)
nvidia-smi -g 0 -dm 1  # Set device 0 to TCC mode
nvidia-smi -g 0 -dm 0  # Set device 0 to WDDM mode
```

### 13.5.4 WSL (Windows Subsystem for Linux)

- CUDA on WSL2 supports multi-GPU but with some limitations.
- P2P access depends on the underlying Windows GPU driver.
- NVLink P2P is supported when running on native Windows GPUs with NVLink.

### 13.5.5 Summary Table

| Platform | IOMMU Setting | PCIe P2P | NVLink P2P |
|----------|--------------|----------|------------|
| Linux bare-metal | Disabled | Supported | Supported |
| Linux bare-metal | Enabled | Not supported (staging) | Supported |
| Linux VM (VFIO) | Enabled | Depends on hypervisor | Supported (if NVLink passed through) |
| Windows (WDDM) | Any | Supported | Supported |
| Windows (TCC) | Any | Supported | Supported |
| WSL2 | Any | Limited | Limited |

## 13.6 Unified Memory Across Multiple GPUs

Unified Memory (`cudaMallocManaged`) transparently handles multi-GPU data movement. When a kernel on device 0 accesses a page that resides on device 1, the UM driver migrates the page:

```cpp
float* data;
cudaMallocManaged(&data, N * sizeof(float));

// Initialize on device 0
cudaSetDevice(0);
initKernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();

// Access from device 1 -- page migration occurs automatically
cudaSetDevice(1);
processKernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();
```

### 13.6.1 Memory Advise for Multi-GPU

Use `cudaMemPrefetchAsync` and `cudaMemAdvise` to optimize data placement:

```cpp
// Prefetch data to a specific device
cudaMemPrefetchAsync(data, N * sizeof(float), 0, stream0);
// data is now resident on device 0

cudaSetDevice(0);
processOnDevice0<<<grid, block, 0, stream0>>>(data, N);

// Prefetch to device 1
cudaMemPrefetchAsync(data, N * sizeof(float), 1, stream1);
cudaSetDevice(1);
processOnDevice1<<<grid, block, 0, stream1>>(data, N);
```

```cpp
// Advise the driver about expected access pattern
cudaMemAdvise(data, N * sizeof(float),
              cudaMemAdviseSetPreferredLocation, 0);
// Prefer keeping pages on device 0

cudaMemAdvise(data, N * sizeof(float),
              cudaMemAdviseSetAccessedBy, 1);
// Device 1 will read this data; driver may set up read-only mapping
```

### 13.6.2 Multi-GPU UM Performance Considerations

- **Page migration overhead**: Each page fault triggers a migration. Use prefetching to batch migrations.
- **Thrashing**: If two devices repeatedly access the same pages, performance degrades. Use `cudaMemAdviseSetPreferredLocation` to pin pages or `cudaMemAdviseSetReadMostly` for read-shared data.
- **Over-subscription**: UM allows allocating more memory than the GPU has, spilling to host memory. This can cause excessive migration. Monitor with `cudaMemGetInfo`.
- **Accessibility hints**: `cudaMemAdviseSetAccessedBy` avoids page faults for remote access by setting up direct mappings (especially useful with NVLink).

## 13.7 Best Practices for Multi-GPU Programming

1. **Minimize cross-device data transfers** -- Keep data local to the GPU that processes it.
2. **Use NVLink over PCIe** -- When available, NVLink provides much higher bandwidth and lower latency.
3. **Overlap computation and communication** -- Use streams and async copies to pipeline data transfer with computation.
4. **Enable P2P access early** -- Call `cudaDeviceEnablePeerAccess` at application startup, not in the hot path.
5. **Use events for cross-device sync** -- `cudaStreamWaitEvent` is the preferred mechanism; avoid `cudaDeviceSynchronize` in performance-critical paths.
6. **Check P2P availability** -- Always query `cudaDeviceCanAccessPeer` before assuming P2P is available.
7. **Consider Virtual Memory Management API for large-scale P2P** -- When you need more than 8 peer connections, use `cuMemCreate`/`cuMemMap` instead of `cudaDeviceEnablePeerAccess`.
8. **Balance workload across GPUs** -- Partition work evenly, accounting for different GPU capabilities or memory sizes.
9. **Use `cudaMemPrefetchAsync` with UM** -- Avoid page fault storms by explicitly prefetching data to the device that needs it.
10. **Profile with Nsight Systems** -- Use `nsys profile --trace=cuda,nvtx` to visualize multi-GPU timelines and identify transfer bottlenecks.
