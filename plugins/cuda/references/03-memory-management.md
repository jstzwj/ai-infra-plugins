# CUDA Programming Guide - Chapter 3: Memory Management

This reference covers all aspects of CUDA memory management, including the unified virtual address space, Unified Memory, page-locked host memory, device memory spaces (global, shared, constant, registers, local), coalesced access patterns, and shared memory bank conflicts.

---

## 3.1 Unified Virtual Address Space

On 64-bit systems with compute capability 2.0 or higher, CUDA provides a **unified virtual address space** (UVA) where host memory and all device memory are mapped into a single virtual address range.

### Key Properties

- Every memory allocation (host or device) has a unique virtual address within the same address space.
- The runtime can automatically determine the physical location (host or device) of any pointer.
- `cudaMemcpyDefault` can be used as the copy direction, and the runtime determines the source and destination automatically.

### Querying Pointer Attributes

```cpp
cudaPointerAttributes attrs;
cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
if (err == cudaSuccess) {
    switch (attrs.type) {
        case cudaMemoryTypeHost:
            printf("Pointer is in host memory\n");
            break;
        case cudaMemoryTypeDevice:
            printf("Pointer is in device memory (device %d)\n", attrs.device);
            break;
        case cudaMemoryTypeManaged:
            printf("Pointer is managed (Unified Memory)\n");
            break;
        default:
            printf("Pointer type unknown\n");
            break;
    }
}
```

### Auto-Detect Copy Direction

```cpp
// With UVA, cudaMemcpyDefault auto-detects direction
cudaMemcpy(dst, src, size, cudaMemcpyDefault);

// Works regardless of whether src and dst are host or device pointers
// The runtime inspects the virtual addresses to determine the transfer type
```

---

## 3.2 Unified Memory

Unified Memory (UM) creates a **single pool of managed memory** that is accessible from both CPU and GPU. The CUDA runtime and hardware automatically handle data migration between host and device memory.

### 3.2.1 Allocation Methods

#### cudaMallocManaged

```cpp
float* data;
cudaError_t err = cudaMallocManaged(&data, N * sizeof(float));
if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate managed memory: %s\n",
            cudaGetErrorString(err));
    return;
}

// Accessible from host
for (int i = 0; i < N; i++) data[i] = float(i);

// Accessible from device
kernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();

// Still accessible from host
printf("data[0] = %f\n", data[0]);

cudaFree(data);
```

#### __managed__ Specifier

```cpp
// File-scope managed variable
__managed__ int counter = 0;
__managed__ float buffer[4096];

__global__ void increment() {
    atomicAdd(&counter, 1);
}

int main() {
    counter = 0;        // host write
    increment<<<1, 256>>>();
    cudaDeviceSynchronize();
    printf("Counter = %d\n", counter);  // host read
    return 0;
}
```

#### Implicit Unified Memory (HMM/ATS)

On systems with **Heterogeneous Memory Management (HMM)** or **Address Translation Services (ATS)** support, all system memory is implicitly managed:

- **HMM**: Available on Linux with recent kernels and Grace Hopper / PCIe-connected GPUs. All `malloc`/`new`/`mmap` allocations are automatically accessible from the GPU.
- **ATS**: Available on Grace Hopper Superchip. CPU and GPU share the same physical memory with coherent caches.

```cpp
// On HMM/ATS systems, regular allocations work
float* data = (float*)malloc(N * sizeof(float));
// data is directly accessible from GPU kernels
kernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();
free(data);
```

### 3.2.2 Unified Memory Paradigms

| Attribute | Full Support | HMM | ATS | Limited (Pre-6.0) |
|---|---|---|---|---|
| `concurrentManagedAccess` | 1 | 1 | 1 | 0 |
| `pageableMemoryAccess` | 0 | 1 | 1 | 0 |
| `pageableMemoryAccessUsesHostPageTables` | 0 | 0 | 1 | 0 |

Query UM capabilities:

```cpp
int concurrent, pageable, pageTables;
cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, device);
cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, device);
cudaDeviceGetAttribute(&pageTables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, device);

printf("Concurrent managed access: %d\n", concurrent);
printf("Pageable memory access (HMM): %d\n", pageable);
printf("Uses host page tables (ATS): %d\n", pageTables);
```

### 3.2.3 Performance Hints

Unified Memory provides advisory hints to guide the runtime's migration decisions. These hints do not change program correctness but can significantly improve performance.

#### Prefetching

Move data to a specific device before it is needed, avoiding on-demand page fault latency:

```cpp
// Prefetch to GPU (deviceId) asynchronously on a stream
cudaMemPrefetchAsync(data, size, deviceId, 0, stream);

// Prefetch to CPU (cudaCpuDeviceId)
cudaMemPrefetchAsync(data, size, cudaCpuDeviceId, 0, stream);

// Example: prefetch before kernel launch
cudaMemPrefetchAsync(d_data, N * sizeof(float), deviceId, 0, stream);
kernel<<<grid, block, 0, stream>>>(d_data, N);
// No stream synchronize needed -- kernel will wait for prefetch
```

#### Memory Advise

```cpp
// Hint that data will be mostly read (enables replication)
cudaMemAdvise(data, size, cudaMemAdviseSetReadMostly, deviceId);

// Hint preferred physical location (data will be migrated there)
cudaMemAdvise(data, size, cudaMemAdviseSetPreferredLocation, deviceId);
// Use cudaCpuDeviceId for CPU as preferred location

// Hint that data will be accessed by a device (prefetches on first access)
cudaMemAdvise(data, size, cudaMemAdviseSetAccessedBy, deviceId);

// Clear hints (restore default behavior)
cudaMemAdvise(data, size, cudaMemAdviseUnsetReadMostly, deviceId);
cudaMemAdvise(data, size, cudaMemAdviseUnsetPreferredLocation, deviceId);
cudaMemAdvise(data, size, cudaMemAdviseUnsetAccessedBy, deviceId);
```

Practical example combining prefetch and advise:

```cpp
void processOnGPU(float* data, int N, int deviceId) {
    size_t bytes = N * sizeof(float);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Advise that data is read-mostly (allows replication)
    cudaMemAdvise(data, bytes, cudaMemAdviseSetReadMostly, deviceId);

    // Prefetch to GPU
    cudaMemPrefetchAsync(data, bytes, deviceId, 0, stream);

    // Set GPU as preferred location for writes
    cudaMemAdvise(data, bytes, cudaMemAdviseSetPreferredLocation, deviceId);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    kernel<<<gridSize, blockSize, 0, stream>>>(data, N);

    // Prefetch result back to CPU
    cudaMemPrefetchAsync(data, bytes, cudaCpuDeviceId, 0, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}
```

### 3.2.4 Memory Allocators Overview

| API | Placement | Accessible From | Migrates | Page Sizes |
|-----|-----------|-----------------|----------|------------|
| `malloc` / `new` / `mmap` | First touch (CPU) | CPU, GPU (HMM/ATS) | Yes | System/huge pages |
| `cudaMallocManaged` | First touch | CPU, GPU | Yes | CPU: system pages, GPU: 2 MB |
| `cudaMalloc` | GPU | GPU only | No | 2 MB |
| `cudaMallocHost` | CPU (pinned) | CPU, GPU (DMA) | No | System/2 MB |

Migration behavior details:

- **`cudaMallocManaged`**: Pages start in an unpopulated state. On first access by either CPU or GPU, the page is allocated in that processor's memory. Subsequent accesses by the other processor trigger page migration.
- **`cudaMalloc`**: Always allocated on the GPU. Not accessible from the CPU (except via UVA mapping with `cudaHostGetDevicePointer` on very old architectures).
- **HMM allocations**: Regular `malloc` allocations behave similarly to `cudaMallocManaged` -- the runtime transparently migrates pages between CPU and GPU on demand.

### 3.2.5 Asynchronous Memory Operations

```cpp
// Async allocation (CUDA 11.3+)
cudaMemPool_t pool;
cudaMemPoolCreate(&pool, &poolProps);
void* ptr;
cudaMallocAsync(&ptr, size, stream);

// Async free
cudaFreeAsync(ptr, stream);

// Memset
cudaMemsetAsync(d_ptr, 0, size, stream);

// 2D memset
cudaMemset2DAsync(d_ptr, pitch, 0, width, height, stream);

// 3D memset
cudaMemset3DAsync(&memsetParams, 0, stream);
```

---

## 3.3 Page-Locked Host Memory

Page-locked (pinned) host memory is host memory whose physical pages are guaranteed to remain in RAM and not be swapped to disk by the operating system.

### Benefits

- **Faster transfers**: DMA engine can transfer directly between pinned memory and device memory without staging through an intermediate buffer.
- **Asynchronous transfers**: `cudaMemcpyAsync` with pinned memory truly overlaps with kernel execution.
- **Mapped memory**: Pinned memory can be mapped into the device's address space for zero-copy access.
- **Write-combined memory**: Optional flag that can improve PCIe write bandwidth for host-to-device transfers.

### Allocation APIs

```cpp
// Method 1: Simple pinned allocation
float* h_data;
cudaMallocHost(&h_data, N * sizeof(float));
// Use h_data on host...
cudaFreeHost(h_data);

// Method 2: Pinned allocation with flags
float* h_data2;
unsigned int flags = cudaHostAllocDefault;
cudaHostAlloc(&h_data2, N * sizeof(float), flags);
cudaFreeHost(h_data2);

// Method 3: Pin existing allocation
float* h_existing = (float*)malloc(N * sizeof(float));
cudaHostRegister(h_existing, N * sizeof(float), cudaHostRegisterDefault);
// ... use for async transfers ...
cudaHostUnregister(h_existing);
free(h_existing);
```

### Flags for cudaHostAlloc

| Flag | Description |
|------|-------------|
| `cudaHostAllocDefault` | Default behavior (pinned, not portable, not mapped) |
| `cudaHostAllocPortable` | Memory is accessible from all CUDA contexts (not just the one that allocated it) |
| `cudaHostAllocMapped` | Map the allocation into the device's address space (zero-copy access) |
| `cudaHostAllocWriteCombined` | Write-combined memory; faster for host-to-device transfers, slower for host reads |

### Mapped (Zero-Copy) Memory

```cpp
// Allocate mapped pinned memory
float* h_mapped;
cudaHostAlloc(&h_mapped, N * sizeof(float), cudaHostAllocMapped);

// Get device-accessible pointer
float* d_mapped;
cudaHostGetDevicePointer(&d_mapped, h_mapped, 0);

// Kernel accesses device memory directly -- no cudaMemcpy needed
// Data goes over PCIe on each access (high latency, use only when data is accessed once)
kernel<<<grid, block>>>(d_mapped, N);
cudaDeviceSynchronize();

// Host reads the same memory
printf("Result: %f\n", h_mapped[0]);

cudaFreeHost(h_mapped);
```

**Warning**: Mapped/zero-copy memory is appropriate only when data is accessed sparsely or once. Every device access goes over PCIe, which has much higher latency and lower bandwidth than device global memory.

### Portable Memory

```cpp
// Allocate portable pinned memory (accessible from all GPU contexts)
float* h_portable;
cudaHostAlloc(&h_portable, N * sizeof(float),
              cudaHostAllocPortable | cudaHostAllocDefault);

// Can be used with cudaSetDevice() for any GPU
cudaSetDevice(0);
cudaMemcpy(d0, h_portable, size, cudaMemcpyHostToDevice);

cudaSetDevice(1);
cudaMemcpy(d1, h_portable, size, cudaMemcpyHostToDevice);

cudaFreeHost(h_portable);
```

---

## 3.4 Device Memory Spaces

CUDA provides several distinct memory spaces on the device, each with different characteristics:

| Memory | Location | Cached | Access | Scope | Lifetime |
|--------|----------|--------|--------|-------|----------|
| **Register** | On-chip | N/A | R/W | Single thread | Kernel |
| **Local** | Off-chip (global) | L1+L2 | R/W | Single thread | Kernel |
| **Shared** | On-chip | N/A | R/W | Thread block (or cluster) | Kernel |
| **Global** | Off-chip (DRAM) | L1+L2 | R/W | All threads + host | Application |
| **Constant** | Off-chip (DRAM) | Yes (const cache) | R only in kernel | All threads + host | Application |
| **Texture** | Off-chip (DRAM) | Yes (tex cache) | R only in kernel | All threads + host | Application |
| **Surface** | Off-chip (DRAM) | Yes (surf cache) | R/W | All threads + host | Application |

### 3.4.1 Global Memory

Global memory is the largest but highest-latency memory space. All thread blocks and the host can read and write global memory.

```cpp
// Allocation
float* d_data;
cudaMalloc(&d_data, N * sizeof(float));

// Initialization
cudaMemset(d_data, 0, N * sizeof(float));  // set to zero

// Host-to-device copy
cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

// Device-to-host copy
cudaMemcpy(h_result, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

// Free
cudaFree(d_data);
```

#### 2D and 3D Global Memory Allocation

For multidimensional data, CUDA provides pitched allocation APIs that ensure proper alignment for coalesced access:

```cpp
// 2D allocation
size_t pitch;
float* d_matrix;
cudaMallocPitch(&d_matrix, &pitch, width * sizeof(float), height);
// pitch is the byte offset between consecutive rows (may be > width * sizeof(float))

// Access in kernel
__global__ void process2D(float* matrix, size_t pitch, int width, int height) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x < width && y < height) {
        float* row = (float*)((char*)matrix + y * pitch);
        row[x] *= 2.0f;
    }
}

// 3D allocation
cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
cudaPitchedPtr d_volume;
cudaMalloc3D(&d_volume, extent);

// Access in kernel using cudaMemcpy3DParms struct for positioning
```

#### 3D Array and Surface/Texture Memory

```cpp
// Create a 3D CUDA array for texture/surface access
cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
cudaArray* d_array;
cudaExtent extent = make_cudaExtent(width, height, depth);
cudaMalloc3DArray(&d_array, &desc, extent);

// Copy data to array
cudaMemcpy3DParms copyParams = {0};
copyParams.srcPtr = make_cudaPitchedPtr(h_data, width * sizeof(float), width, height);
copyParams.dstArray = d_array;
copyParams.extent = extent;
copyParams.kind = cudaMemcpyHostToDevice;
cudaMemcpy3D(&copyParams);

// Bind to texture or surface
cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeArray;
resDesc.res.array.array = d_array;

cudaSurfaceObject_t surface;
cudaCreateSurfaceObject(&surface, &resDesc);

// Use in kernel
__global__ void surfKernel(cudaSurfaceObject_t surf, int width, int height) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x < width && y < height) {
        float val = surf2Dread<float>(surf, x * sizeof(float), y);
        surf2Dwrite(val * 2.0f, surf, x * sizeof(float), y);
    }
}

// Cleanup
cudaDestroySurfaceObject(surface);
cudaFreeArray(d_array);
```

### 3.4.2 Shared Memory

Shared memory is a fast, user-managed on-chip memory shared among all threads in a thread block. It is typically 10-100x faster than global memory.

#### Static Shared Memory

```cpp
__global__ void staticSharedKernel(float* input, float* output, int N) {
    // Static shared memory -- size known at compile time
    __shared__ float tile[16][16];

    int x = threadIdx.x;
    int y = threadIdx.y;
    int idx = x + y * 16 + blockIdx.x * 256;

    // Load from global to shared
    if (idx < N) tile[y][x] = input[idx];
    __syncthreads();

    // Process in shared memory
    tile[y][x] *= 2.0f;
    __syncthreads();

    // Write back to global
    if (idx < N) output[idx] = tile[y][x];
}

dim3 block(16, 16);
dim3 grid(N / 256);
staticSharedKernel<<<grid, block>>>(input, output, N);
```

#### Dynamic Shared Memory

```cpp
__global__ void dynamicSharedKernel(float* input, float* output, int N) {
    // Dynamic shared memory -- size specified at kernel launch
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = tid + blockDim.x * blockIdx.x;

    // Load
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Specify dynamic shared memory size as third launch parameter
int sharedMemBytes = blockSize * sizeof(float);
dynamicSharedKernel<<<gridSize, blockSize, sharedMemBytes>>>(input, output, N);
```

#### Multiple Dynamic Shared Memory Arrays

When multiple dynamically-sized shared memory arrays are needed, use pointer arithmetic on a single `extern __shared__` allocation:

```cpp
__global__ void multiSharedKernel(float* input, int N,
                                   int arr0Size, int arr1Size) {
    // Single dynamic allocation
    extern __shared__ char sharedBuffer[];

    // Partition into typed sub-arrays via pointer arithmetic
    short* arr0 = (short*)sharedBuffer;
    float* arr1 = (float*)&arr0[arr0Size];
    int*   arr2 = (int*)&arr1[arr1Size];

    // Use arr0, arr1, arr2...
    int tid = threadIdx.x;
    arr0[tid] = (short)input[tid];
    arr1[tid] = input[tid] * 2.0f;
    arr2[tid] = (int)input[tid];

    __syncthreads();
    // ...
}

// Calculate total shared memory size
size_t sharedBytes = arr0Size * sizeof(short)
                   + arr1Size * sizeof(float)
                   + arr2Size * sizeof(int);
multiSharedKernel<<<grid, block, sharedBytes>>>(input, N, arr0Size, arr1Size);
```

#### Configuring Shared Memory / L1 Cache Split

```cpp
// Set shared memory carveout preference
// cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, percentage);
// percentage: 0 (default) to 100 (max shared memory)

// Or set maximum dynamic shared memory
cudaFuncSetAttribute(myKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);

// Or set preferred shared memory configuration for the device
// On Ampere/Hopper, SM has 164 KB shared memory + L1 cache combined
cudaFuncSetAttribute(myKernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                     cudaSharedmemCarveoutMaxShared);  // maximize shared memory
```

### 3.4.3 Constant Memory

Constant memory is a read-only (in kernel) memory space of up to 64 KB that is cached by the constant cache. It is optimized for the case where all threads in a warp read the same address.

```cpp
// Declare constant memory variable (file scope)
__constant__ float c_transform[3][3];
__constant__ int c_params[8];
__constant__ float c_weights[256];  // up to 64 KB total

// Initialize from host
float h_transform[3][3] = {
    {1.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f},
    {0.0f, 0.0f, 1.0f}
};
cudaMemcpyToSymbol(c_transform, h_transform, sizeof(h_transform));

// Use in kernel
__global__ void transformKernel(float* points, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        float x = points[idx * 3 + 0];
        float y = points[idx * 3 + 1];
        float z = points[idx * 3 + 2];

        // All threads read same c_transform values -- broadcast, very fast
        float nx = c_transform[0][0] * x + c_transform[0][1] * y + c_transform[0][2] * z;
        float ny = c_transform[1][0] * x + c_transform[1][1] * y + c_transform[1][2] * z;
        float nz = c_transform[2][0] * x + c_transform[2][1] * y + c_transform[2][2] * z;

        points[idx * 3 + 0] = nx;
        points[idx * 3 + 1] = ny;
        points[idx * 3 + 2] = nz;
    }
}

// Read back on host (optional)
float h_result[3][3];
cudaMemcpyFromSymbol(h_result, c_transform, sizeof(h_result));
```

**Performance notes**:

- When all threads in a warp read the same constant memory address, the value is **broadcast** in a single cycle -- very fast.
- When threads in a warp read different constant memory addresses, the reads are **serialized** -- one per cycle. In this case, global memory with L1 caching may be faster.

### 3.4.4 Registers and Local Memory

#### Registers

Registers are the fastest storage available to CUDA threads. The compiler automatically maps local scalar variables to registers when possible.

```cpp
__global__ void registerExample(float* output, int N) {
    // These are all register variables (compiled by the compiler)
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float temp = 0.0f;
    float sum = 0.0f;
    int count = 0;

    for (int i = 0; i < 100; i++) {
        sum += sinf((float)i);  // sum and i in registers
    }

    if (idx < N) output[idx] = sum;
}
```

- Maximum registers per thread: 255.
- Register allocation affects occupancy: more registers per thread means fewer concurrent threads per SM.
- Use `__launch_bounds__` to control register allocation:

```cpp
// Tell compiler: expect at least 256 threads per block, 2 blocks per SM
__global__ void __launch_bounds__(256, 2) myKernel(...) {
    // Compiler will limit register usage to fit 2 blocks of 256 threads
}
```

- Inspect register usage: `--ptxas-options=-v` reports registers per kernel.

#### Local Memory

**Local memory** is a logical per-thread memory space that is physically stored in global memory (off-chip). Variables that cannot fit in registers are placed in local memory ("register spills"). Common causes:

- Arrays indexed with variable (runtime-computed) indices.
- Large structures that exceed the register file.
- Register spills when the compiler runs out of registers.

```cpp
__global__ void localMemoryExample(float* output, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Variable-indexed array -> local memory
    float coeffs[256];
    for (int i = 0; i < 256; i++) coeffs[i] = (float)i;
    int j = idx % 256;
    float val = coeffs[j];  // j is not a compile-time constant -> local memory

    // Large structure -> local memory
    struct Big { float data[1024]; };
    Big big;  // too large for registers -> local memory

    if (idx < N) output[idx] = val;
}
```

Checking for local memory usage:

```bash
# Compile and show resource usage
nvcc --ptxas-options=-v mykernel.cu -arch=sm_80

# Output will include:
#   ptxas info: Used 32 registers, 4096 bytes smem, 128 bytes lmem
#                                                        ^^^^ local memory
```

In PTX code, local memory uses the `.local` directive and `ld.local` / `st.local` instructions:

```ptx
.reg .f32 %f<256>;
.local .align 4 .b8 __local_depot0[1024];
ld.local.f32 %f1, [%r1];
st.local.f32 [%r1], %f2;
```

---

## 3.5 Coalesced Global Memory Access

Coalesced memory access is one of the most critical performance optimizations in CUDA programming. The GPU hardware coalesces (combines) memory requests from threads in a warp into fewer, larger transactions.

### How Coalescing Works

When a warp executes a global memory load or store instruction:

1. Each thread provides a memory address.
2. The hardware groups addresses into **segments** (32-byte, 64-byte, or 128-byte aligned).
3. The hardware issues the minimum number of memory transactions to service all addresses.

**Perfectly coalesced**: consecutive threads access consecutive 4-byte words within a single 128-byte aligned segment. This results in a single 128-byte transaction for the entire warp.

**Uncoalesced**: strided, random, or misaligned access patterns result in multiple transactions, wasting bandwidth.

### Coalescing Rules (Simplified)

| Access Pattern | Transactions per Warp | Efficiency |
|---|---|---|
| Consecutive threads, consecutive words, aligned | 1 | 100% |
| Consecutive threads, consecutive words, misaligned | 2 (for small offsets) | ~50% |
| Strided access (stride = 2) | 2 | 50% |
| Strided access (stride = 4) | 4 | 25% |
| Random access | Up to 32 | ~3% |

### Example: Coalesced vs. Uncoalesced

```cpp
// GOOD: Coalesced read -- thread i reads data[i]
__global__ void coalescedRead(float* data, float* output, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) output[idx] = data[idx];
}

// BAD: Strided read -- thread i reads data[i * stride]
__global__ void stridedRead(float* data, float* output, int N, int stride) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx * stride < N) output[idx] = data[idx * stride];
    // Warp reads: data[0], data[stride], data[2*stride], ...
    // Each access hits a different cache line -> 32 separate transactions
}

// BAD: Column-major read of row-major matrix
__global__ void badTranspose(float* input, float* output, int width, int height) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x < width && y < height) {
        // output[y + x * height] = input[x + y * width];
        // Threads in warp have same y, consecutive x -> strided reads of input
        output[y + x * height] = input[x + y * width];
    }
}
```

### Optimizing with Shared Memory: Matrix Transpose

A classic example where shared memory enables coalesced access:

```cpp
// Naive transpose: coalesced reads but strided writes
__global__ void naiveTranspose(const float* input, float* output,
                                int width, int height) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x < width && y < height) {
        output[x * height + y] = input[y * width + x];
        // Coalesced read: input[y*width+0], input[y*width+1], ...
        // Strided write: output[0*height+y], output[1*height+y], ... (stride = height)
    }
}

// Optimized transpose using shared memory
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void optimizedTranspose(const float* input, float* output,
                                    int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts

    // Read from global (coalesced) into shared memory
    int x = threadIdx.x + blockIdx.x * TILE_DIM;
    int y = threadIdx.y + blockIdx.y * TILE_DIM;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
        }
    }
    __syncthreads();

    // Write from shared memory to global (coalesced)
    x = threadIdx.x + blockIdx.y * TILE_DIM;  // transposed block indices
    y = threadIdx.y + blockIdx.x * TILE_DIM;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            output[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

dim3 block(TILE_DIM, BLOCK_ROWS);
dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
optimizedTranspose<<<grid, block>>>(d_input, d_output, width, height);
```

### Structure of Arrays (SoA) vs. Array of Structures (AoS)

For coalesced access, prefer SoA layout:

```cpp
// BAD: AoS -- threads access non-contiguous memory
struct Particle {
    float x, y, z;       // 12 bytes
    float vx, vy, vz;    // 12 bytes
    float mass;           // 4 bytes
};
Particle* particles;  // array of structures

__global__ void updateAoS(Particle* particles, int N, float dt) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) {
        // Each thread reads 7 floats, but consecutive threads read
        // x at offsets 0, 28, 56, ... -> stride = 7 (uncolalesced)
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
    }
}

// GOOD: SoA -- threads access contiguous memory
struct Particles {
    float* x; float* y; float* z;
    float* vx; float* vy; float* vz;
    float* mass;
};

__global__ void updateSoA(Particles p, int N, float dt) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) {
        // Consecutive threads read consecutive x values -> coalesced
        p.x[i] += p.vx[i] * dt;
        p.y[i] += p.vy[i] * dt;
        p.z[i] += p.vz[i] * dt;
    }
}
```

---

## 3.6 Shared Memory Banks

### Bank Architecture

Shared memory is divided into **32 memory banks** that can be accessed simultaneously. The mapping of addresses to banks depends on the data width:

- **4-byte words** (int, float): `bank = (address / 4) % 32`
- **8-byte words** (double, double2): `bank = (address / 8) % 32` (on CC 2.0+)

Successive 4-byte words map to successive banks:

```
Address (bytes)   0-3   4-7   8-11  12-15  ...  124-127  128-131  ...
Bank               0     1     2     3     ...    31       0       ...
```

### Bank Conflicts

A **bank conflict** occurs when two or more threads in a warp access different addresses within the same bank in the same memory transaction. Bank conflicts cause the accesses to be **serialized**:

- **No conflict**: all threads access different banks (or same address = broadcast). Single-cycle access.
- **2-way conflict**: two threads access different addresses in the same bank. Takes 2 cycles.
- **N-way conflict**: N threads access different addresses in the same bank. Takes N cycles.
- **Broadcast**: all threads access the **same address** in the same bank. Single-cycle access (broadcast).

### Common Conflict Patterns

```cpp
// NO CONFLICT: Each thread accesses its own element (consecutive words)
__shared__ float data[256];
float val = data[threadIdx.x];  // Thread i -> bank i, no conflict

// 2-WAY CONFLICT: Stride of 2
__shared__ float data[256];
float val = data[threadIdx.x * 2];  // Thread 0 -> bank 0, thread 1 -> bank 2
                                      // Thread 16 -> bank 0 (conflict with thread 0)

// NO CONFLICT: Each thread reads the same value (broadcast)
__shared__ float data[256];
float val = data[0];  // All threads read bank 0, same address -> broadcast, no conflict
```

### Padding to Avoid Bank Conflicts

A common technique is to pad shared memory arrays so that column accesses in a 2D tile avoid bank conflicts:

```cpp
// WITH CONFLICTS: 32x32 tile, column access
__shared__ float tile[32][32];
float val = tile[threadIdx.x][threadIdx.y];
// tile[0][y] and tile[1][y] both map to bank y (column y has stride 32 = 32 banks)
// -> 32-way bank conflict!

// WITHOUT CONFLICTS: Pad by 1 element
__shared__ float tile[32][33];  // 33 = 32 + 1 padding
float val = tile[threadIdx.x][threadIdx.y];
// tile[0][y] maps to bank y
// tile[1][y] maps to bank (y + 33) % 32 = (y + 1) % 32
// tile[2][y] maps to bank (y + 66) % 32 = (y + 2) % 32
// No two rows map to the same bank -> no conflict
```

### Practical Example: Reduction with Bank Conflict Avoidance

```cpp
__global__ void reduceSharedMem(const float* input, float* output, int N) {
    // Pad to avoid bank conflicts in reduction
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = tid + blockDim.x * blockIdx.x;

    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // Sequential addressing reduction (avoids bank conflicts)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// For interleaved addressing: sdata[tid] += sdata[tid + stride]
// stride = blockDim.x/2 = 128: thread 0 reads sdata[0] and sdata[128]
//   bank 0 and bank (128 % 32) = bank 0 -> CONFLICT
// But since only first half of threads are active, and stride is large,
// the conflict pattern varies. Sequential addressing avoids this more cleanly.
```

### Matrix Transpose with Padding

```cpp
#define TILE 32

__global__ void transposeWithPadding(const float* input, float* output,
                                      int width, int height) {
    // Padding (+1) eliminates bank conflicts during the transpose
    __shared__ float tile[TILE][TILE + 1];

    int x = threadIdx.x + blockIdx.x * TILE;
    int y = threadIdx.y + blockIdx.y * TILE;

    // Coalesced global read -> shared memory
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();

    // Shared memory -> coalesced global write
    // Without padding: tile[threadIdx.x][threadIdx.y] has bank conflicts
    // because column stride is 32 (equals number of banks)
    // With padding: stride is 33, so bank = (threadIdx.y * 33 + threadIdx.x) % 32
    x = threadIdx.x + blockIdx.y * TILE;
    y = threadIdx.y + blockIdx.x * TILE;
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

---

## Summary: Memory Hierarchy Performance Characteristics

| Memory Type | Latency (cycles) | Bandwidth | Scope | Programmer Managed |
|---|---|---|---|---|
| Registers | ~1 | Highest | Thread | No (compiler) |
| Shared Memory | ~20-30 | ~19 TB/s (per SM) | Block | Yes |
| L1 Cache | ~20-30 | ~19 TB/s (per SM) | SM | Partially (carveout) |
| L2 Cache | ~200-300 | ~3-5 TB/s | GPU | Partially (policies) |
| Global Memory | ~400-800 | ~1-3 TB/s | Grid + Host | Yes |
| Host Memory | ~1000+ (over PCIe) | ~32-64 GB/s (PCIe) | Host | Yes |

### Memory Optimization Priority

1. **Minimize global memory accesses** -- use shared memory as a scratchpad.
2. **Ensure coalesced global memory access** -- consecutive threads access consecutive addresses.
3. **Avoid shared memory bank conflicts** -- pad arrays or use access patterns that distribute across banks.
4. **Use constant memory** for read-only data accessed uniformly by all threads.
5. **Use prefetching and memory advise** with Unified Memory to reduce page fault overhead.
6. **Prefer SoA over AoS** for data accessed in a data-parallel fashion.
