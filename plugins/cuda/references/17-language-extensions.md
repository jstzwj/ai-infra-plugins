# 17. CUDA C++ Language Extensions

This document covers CUDA C++ language extensions, which provide the constructs for defining GPU kernels, managing memory spaces, synchronizing threads, performing atomic operations, and using warp-level primitives. These extensions are the core API surface for CUDA kernel programming.

---

## Table of Contents

1. [Execution Space Specifiers](#171-execution-space-specifiers)
2. [Memory Space Specifiers](#172-memory-space-specifiers)
3. [Built-in Variables](#173-built-in-variables)
4. [Vector Types](#174-vector-types)
5. [Kernel Launch Configuration](#175-kernel-launch-configuration)
6. [Launch Bounds](#176-launch-bounds)
7. [Synchronization](#177-synchronization)
8. [Atomic Functions](#178-atomic-functions)
9. [Warp Functions](#179-warp-functions)
10. [Other Intrinsics](#1710-other-intrinsics)
11. [Compiler Hints](#1711-compiler-hints)
12. [__grid_constant__ Parameters](#1712-__grid_constant__-parameters)
13. [Annotation Summary](#1713-annotation-summary)

---

## 17.1 Execution Space Specifiers

Execution space specifiers define where a function executes and from where it can be called:

| Specifier | Executes On | Callable From Host | Callable From Device |
|---|---|---|---|
| `__host__` | Host (CPU) | Yes | No |
| `__device__` | Device (GPU) | No | Yes |
| `__global__` | Device (GPU) | Yes | Yes (CC 3.2+, dynamic parallelism) |
| `__host__ __device__` | Both | Yes | Yes |

### Detailed Semantics

```cpp
// __host__: Executes on the host CPU only
// Can only be called from host code
// This is the default for functions without any specifier
__host__ void hostFunction() {
    printf("Running on CPU\n");
}

// __device__: Executes on the device GPU only
// Can only be called from device code (or __global__ functions)
__device__ float deviceFunction(float x) {
    return x * x + 2.0f * x + 1.0f;
}

// __global__: Kernel function that runs on the device
// Must return void
// Callable from host code, and from device code on CC 3.2+
__global__ void myKernel(int* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        data[tid] = deviceFunction(data[tid]);  // Calling __device__ from __global__
    }
}

// __host__ __device__: Runs on both host and device
// Compiler generates two versions of the function
__host__ __device__ int clamp(int val, int lo, int hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

__global__ void useClamp(int* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        data[tid] = clamp(data[tid], 0, 255);  // Calls device version
    }
}

void hostUseClamp() {
    int result = clamp(300, 0, 255);  // Calls host version, result = 255
}
```

### __global__ Function Details

```cpp
// __global__ functions must:
// - Return void
// - Have a fixed number of arguments (no variadic)
// - Not be overloaded (before CUDA 11.5)
// - Not be a member function (unless static)
// - Max 32764 bytes of parameters

// Launching from host
myKernel<<<gridSize, blockSize>>>(d_data, N);

// Launching from device (dynamic parallelism, CC 3.2+)
__global__ void parentKernel(int* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        // Device-side kernel launch
        childKernel<<<1, 256>>>(data, N);
        cudaDeviceSynchronize();  // Wait for child
    }
}

__global__ void childKernel(int* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) data[tid] *= 2;
}
```

### Conditional Compilation for Host/Device

```cpp
// Use __CUDA_ARCH__ to differentiate compilation
__host__ __device__ void conditionalCode() {
#if defined(__CUDA_ARCH__)
    // Device path
    printf("Running on GPU, SM %d\n", __CUDA_ARCH__);
#else
    // Host path
    printf("Running on CPU\n");
#endif
}

// Preventing host compilation of device-only code
__host__ __device__ void maybeDeviceOnly() {
#if defined(__CUDA_ARCH__)
    __syncthreads();  // Only valid on device
#endif
}
```

---

## 17.2 Memory Space Specifiers

CUDA defines distinct memory spaces with different accessibility, lifetime, and performance characteristics:

| Specifier | Location | Accessible By | Lifetime |
|---|---|---|---|
| `__device__` (variable) | Global memory | All threads in grid + host (via API) | CUDA context (application) |
| `__constant__` | Constant memory | All threads in grid + host (via API) | CUDA context (application) |
| `__managed__` | Host + Device (Unified Memory) | Host and device threads | CUDA context (application) |
| `__shared__` | On-chip shared memory (SM) | All threads in same block | Thread block |
| (no specifier, local) | Local memory (per-thread) | Owning thread only | Thread |

### Global Memory (__device__)

```cpp
// Global variable accessible by all threads, persists across kernel launches
__device__ int globalCounter = 0;
__device__ float globalBuffer[1024];

__global__ void useGlobal() {
    // All threads can read/write
    atomicAdd(&globalCounter, 1);

    // Thread 0 can print value
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Counter: %d\n", globalCounter);
    }
}

// Access from host via cudaMemcpy
void hostAccess() {
    int h_counter;
    cudaMemcpyFromSymbol(&h_counter, globalCounter, sizeof(int),
                         0, cudaMemcpyDeviceToHost);
    printf("Global counter from host: %d\n", h_counter);

    // Modify from host
    int new_val = 100;
    cudaMemcpyToSymbol(globalCounter, &new_val, sizeof(int),
                       0, cudaMemcpyHostToDevice);
}
```

### Constant Memory (__constant__)

```cpp
// Constant memory: 64 KB, cached, read-only from device
// Read-only access provides broadcast capability for warp-wide reads
__constant__ float constWeights[256];
__constant__ int constParams[16];

__global__ void useConstant(float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        // Reads are cached and broadcast-efficient when all threads
        // in a warp read the same address
        output[tid] = constWeights[tid % 256] * 2.0f;
    }
}

// Initialize from host
void initConstant() {
    float h_weights[256];
    for (int i = 0; i < 256; ++i) h_weights[i] = 1.0f / (i + 1);
    cudaMemcpyToSymbol(constWeights, h_weights, sizeof(h_weights));
}
```

### Managed Memory (__managed__)

```cpp
// Unified memory: accessible from both host and device
// Runtime migrates data automatically
__managed__ int managedArray[1024];
__managed__ float managedScalar = 3.14f;

__global__ void useManaged() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 1024) {
        managedArray[tid] = tid * 2;
    }
}

void managedExample() {
    // Access from host before kernel launch
    for (int i = 0; i < 1024; ++i) managedArray[i] = i;

    // Launch kernel that modifies managed memory
    useManaged<<<1, 1024>>>();
    cudaDeviceSynchronize();

    // Access from host after kernel completion
    printf("managedArray[500] = %d\n", managedArray[500]);
}
```

### Shared Memory (__shared__)

```cpp
// Shared memory: fast on-chip memory shared by all threads in a block
// Two forms: static and dynamic

// Static shared memory (size known at compile time)
__global__ void staticShared(const float* input, float* output, int N) {
    __shared__ float tile[256];  // Static: 256 floats = 1024 bytes

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load into shared memory
    if (gid < N) tile[tid] = input[gid];
    __syncthreads();

    // Process data in shared memory (e.g., reverse)
    if (gid < N) output[gid] = tile[255 - tid];
}

// Dynamic shared memory (size specified at launch)
__global__ void dynamicShared(const float* input, float* output, int N) {
    extern __shared__ float dynTile[];  // Size determined at launch

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < N) dynTile[tid] = input[gid];
    __syncthreads();

    if (gid < N) output[gid] = dynTile[tid] * 2.0f;
}

// Launch with dynamic shared memory size
void launchDynamic() {
    int smemSize = 256 * sizeof(float);  // 1024 bytes
    dynamicShared<<<4, 256, smemSize>>>(d_input, d_output, 1024);
}
```

---

## 17.3 Built-in Variables

CUDA provides several built-in variables that are automatically available in `__global__` and `__device__` functions:

```cpp
// Grid and block dimensions
dim3 gridDim;    // Number of blocks in the grid (x, y, z)
dim3 blockDim;   // Number of threads per block (x, y, z)

// Block and thread indices
uint3 blockIdx;    // Block index within the grid (x, y, z)
uint3 threadIdx;   // Thread index within the block (x, y, z)

// Warp size
int warpSize;      // Number of threads per warp (32 on all current architectures)
```

### Usage Patterns

```cpp
__global__ void indexingExamples(int* output, int N) {
    // 1D grid, 1D block: most common pattern
    int tid1d = blockIdx.x * blockDim.x + threadIdx.x;

    // 2D grid, 2D block: for image/matrix processing
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    int width = gridDim.x * blockDim.x;
    int tid2d = ty * width + tx;

    // 3D grid, 3D block: for volumetric data
    int tz = threadIdx.z + blockIdx.z * blockDim.z;
    int depth = gridDim.z * blockDim.z;
    int tid3d = tz * width * (gridDim.y * blockDim.y) + ty * width + tx;

    // Compute flattened index
    if (tid1d < N) {
        output[tid1d] = tid1d;
    }

    // Warp-relative operations
    int laneId = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
}
```

### Computing Global Thread Index

```cpp
// Common helper for computing 1D global thread index
__device__ __forceinline__ int getGlobalIdx_1D_1D() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ int getGlobalIdx_2D_2D() {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int width = gridDim.x * blockDim.x;
    return y * width + x;
}

// Helper to compute total number of threads
__device__ __forceinline__ int getGlobalSize() {
    return gridDim.x * blockDim.x * gridDim.y * blockDim.y * gridDim.z * blockDim.z;
}

// Stride loop pattern for processing large arrays
__global__ void strideLoop(int* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;  // Total threads in grid

    for (int i = tid; i < N; i += stride) {
        data[i] *= 2;
    }
}
```

---

## 17.4 Vector Types

CUDA provides vector types for 1, 2, 3, and 4-component vectors of scalar types:

### Available Vector Types

| Type | Components | Base Type | Size |
|---|---|---|---|
| `char1`, `char2`, `char3`, `char4` | 1-4 | `char` (8-bit signed) | 1-4 bytes |
| `uchar1`, `uchar2`, `uchar3`, `uchar4` | 1-4 | `unsigned char` (8-bit unsigned) | 1-4 bytes |
| `short1`, `short2`, `short3`, `short4` | 1-4 | `short` (16-bit signed) | 2-8 bytes |
| `ushort1`, `ushort2`, `ushort3`, `ushort4` | 1-4 | `unsigned short` (16-bit unsigned) | 2-8 bytes |
| `int1`, `int2`, `int3`, `int4` | 1-4 | `int` (32-bit signed) | 4-16 bytes |
| `uint1`, `uint2`, `uint3`, `uint4` | 1-4 | `unsigned int` (32-bit unsigned) | 4-16 bytes |
| `long1`, `long2`, `long3`, `long4` | 1-4 | `long` (32/64-bit) | varies |
| `ulong1`, `ulong2`, `ulong3`, `ulong4` | 1-4 | `unsigned long` | varies |
| `longlong1`, `longlong2` | 1-2 | `long long` (64-bit signed) | 8-16 bytes |
| `float1`, `float2`, `float3`, `float4` | 1-4 | `float` (32-bit) | 4-16 bytes |
| `double1`, `double2` | 1-2 | `double` (64-bit) | 8-16 bytes |
| `dim3` | 3 | `unsigned int` | 12 bytes |

### Field Access

```cpp
// Vector types have named fields: x, y, z, w
int4 v4 = make_int4(1, 2, 3, 4);
int x = v4.x;  // 1
int y = v4.y;  // 2
int z = v4.z;  // 3
int w = v4.w;  // 4

float2 f2 = make_float2(1.0f, 2.0f);
float a = f2.x;  // 1.0f
float b = f2.y;  // 2.0f
```

### Factory Functions

```cpp
// Each vector type has a make_ function
char4 c4 = make_char4('a', 'b', 'c', 'd');
uchar4 uc4 = make_uchar4(0, 128, 200, 255);
short2 s2 = make_short2(-100, 200);
ushort4 us4 = make_ushort4(100, 200, 300, 400);
int3 i3 = make_int3(1, 2, 3);
uint4 ui4 = make_uint4(0, 1, 2, 3);
float4 f4 = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
double2 d2 = make_double2(1.0, 2.0);
```

### Usage Examples

```cpp
// Image processing with uchar4 (RGBA pixel)
__global__ void invertColors(uchar4* pixels, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        uchar4 pixel = pixels[idx];
        pixels[idx] = make_uchar4(
            255 - pixel.x,  // R
            255 - pixel.y,  // G
            255 - pixel.z,  // B
            pixel.w         // Alpha unchanged
        );
    }
}

// Vector math operations
__device__ float3 add3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 cross3(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ float length3(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ float3 normalize3(float3 v) {
    float len = length3(v);
    float invLen = (len > 0.0f) ? 1.0f / len : 0.0f;
    return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

// dim3 for grid/block dimensions
void setupLaunch() {
    dim3 blockSize(16, 16, 1);       // 16x16 = 256 threads per block
    dim3 gridSize(
        (width + 15) / 16,           // Ceiling division
        (height + 15) / 16,
        1
    );
    myKernel<<<gridSize, blockSize>>>(...);
}
```

### Aligned Memory Access with Vector Types

```cpp
// Using vector types for coalesced memory access
__global__ void vectorCopy(const float4* input, float4* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread reads/writes 4 floats (16 bytes) at once
    // This is coalesced when threads in a warp access consecutive 16-byte
    // addresses
    if (tid < N) {
        float4 val = input[tid];
        val.x *= 2.0f;
        val.y *= 2.0f;
        val.z *= 2.0f;
        val.w *= 2.0f;
        output[tid] = val;
    }
}

// Equivalent scalar version processes 4x more elements
__global__ void scalarCopy(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) output[tid] = input[tid] * 2.0f;
}

// Launch: vectorCopy<<<grid, block>>>(..., N/4);  // 4x fewer elements
```

---

## 17.5 Kernel Launch Configuration

Kernel launches use the triple-chevron `<<<>>>` syntax to specify execution configuration:

```cpp
kernel<<<gridDim, blockDim, dynamicSmemSize, stream>>>(kernelParams);
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `gridDim` | `dim3` or `int` | Number of blocks in grid (or x-dimension) |
| `blockDim` | `dim3` or `int` | Threads per block (or x-dimension) |
| `dynamicSmemSize` | `size_t` | Dynamic shared memory per block in bytes (default 0) |
| `stream` | `cudaStream_t` | Stream for asynchronous execution (default 0 = default stream) |

### Launch Variants

```cpp
// 1D grid, 1D block (most common)
kernel<<<numBlocks, threadsPerBlock>>>(params);

// 1D with dynamic shared memory
kernel<<<numBlocks, threadsPerBlock, smemBytes>>>(params);

// 1D with shared memory and stream
kernel<<<numBlocks, threadsPerBlock, smemBytes, stream>>>(params);

// 2D grid, 2D block
dim3 blockSize(16, 16);
dim3 gridSize((width + 15) / 16, (height + 15) / 16);
kernel<<<gridSize, blockSize>>>(params);

// 3D grid, 3D block
dim3 blockSize(8, 8, 8);
dim3 gridSize((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
kernel<<<gridSize, blockSize>>>(params);
```

### Error Handling

```cpp
// Always check for launch errors
#define CUDA_CHECK(call) \
    do { \
        call; \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Usage
CUDA_CHECK(myKernel<<<grid, block>>>(d_data, N));

// Check async errors after synchronization
cudaError_t err = cudaDeviceSynchronize();
if (err != cudaSuccess) {
    fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(err));
}
```

### Cooperative Launch (CC 6.0+)

```cpp
// Launch a kernel that can synchronize across all blocks in the grid
void cooperativeLaunch() {
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    int blockSize = 256;
    int numBlocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, myKernel, blockSize, 0);
    numBlocks *= prop.multiProcessorCount;

    // Cooperative launch: all blocks can participate in grid sync
    void* args[] = { &d_data, &N };
    cudaLaunchCooperativeKernel(
        (void*)myKernel,
        numBlocks,
        blockSize,
        args,
        0,      // shared memory
        0       // stream
    );
}

// Inside kernel: use cooperative_groups for grid synchronization
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void myKernel(int* data, int N) {
    cg::grid_group grid = cg::this_grid();
    int tid = grid.thread_rank();

    if (tid < N) data[tid] = tid;

    grid.sync();  // All blocks synchronize here

    if (tid < N) data[tid] += 1000;
}
```

### Cluster Launch (CC 9.0+)

```cpp
// Launch with thread block clusters
__global__ void __cluster_dims__(2, 2, 1)
clusterKernel(int* data, int N) {
    namespace cg = cooperative_groups;
    auto cluster = cg::this_cluster();
    // ...
}

// Launch cluster kernel
void launchCluster() {
    dim3 blockSize(128);
    dim3 gridSize(8);
    dim3 clusterSize(2);  // 2 blocks per cluster

    cudaLaunchConfig_t config = {0};
    cudaLaunchDimension dimension = {0};
    dimension.gridDim = gridSize;
    dimension.blockDim = blockSize;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = clusterSize.x;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;

    cudaLaunchKernelEx(&config, clusterKernel, data, N);
}
```

---

## 17.6 Launch Bounds

Launch bounds allow developers to guide the compiler's register allocation and occupancy optimization:

```cpp
__global__ void __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor, maxBlocksPerCluster)
kernelName(...);
```

### Parameters

| Parameter | Description | Required |
|---|---|---|
| `maxThreadsPerBlock` | Maximum threads per block this kernel will be launched with | Yes |
| `minBlocksPerMultiprocessor` | Minimum number of blocks per SM desired | No (default: compiler chooses) |
| `maxBlocksPerCluster` | Maximum number of blocks per cluster (CC 9.0+) | No |

### Usage Examples

```cpp
// Specify max threads per block only
__global__ void __launch_bounds__(256)
kernelA(int* data, int N) {
    // Compiler optimizes for 256 threads per block
    // May use more registers per thread
}

// Specify max threads and minimum blocks for higher occupancy
__global__ void __launch_bounds__(256, 8)
kernelB(int* data, int N) {
    // Compiler limits registers to allow at least 8 blocks/SM
    // Max registers per thread = 65536 / (256 * 8) = 32
}

// Full specification with cluster bounds (CC 9.0+)
__global__ void __launch_bounds__(256, 4, 2)
__cluster_dims__(2, 1, 1)
kernelC(int* data, int N) {
    // 256 threads/block, min 4 blocks/SM, max 2 blocks/cluster
}

// __maxnreg__: explicit register limit (CC 9.0+)
__global__ void __maxnreg__(64)
kernelD(int* data, int N) {
    // Each thread uses at most 64 registers
    // If compiler needs more, it spills to local memory
}
```

### Occupancy Calculation

```cpp
// Calculate optimal launch bounds
void optimizeLaunch() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int blockSize = 256;
    int regsPerThread = 32;  // Estimate or get from profiler

    // Max blocks per SM based on registers
    int regsPerSM = prop.regsPerMultiprocessor;  // 65536
    int maxBlocksByRegs = regsPerSM / (blockSize * regsPerThread);

    // Max blocks per SM based on threads
    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;  // 2048
    int maxBlocksByThreads = maxThreadsPerSM / blockSize;

    // Max blocks per SM based on hardware limit
    int maxBlocksHardware = prop.maxBlocksPerMultiProcessor;  // 32

    // Actual occupancy
    int occupancy = min(maxBlocksByRegs, min(maxBlocksByThreads, maxBlocksHardware));
    printf("Blocks/SM: %d, Threads/SM: %d, Occupancy: %.1f%%\n",
           occupancy, occupancy * blockSize,
           100.0 * occupancy * blockSize / maxThreadsPerSM);
}
```

---

## 17.7 Synchronization

CUDA provides multiple levels of synchronization primitives:

### __syncthreads()

```cpp
// Block-level barrier: all threads in a block must reach this point
// before any can proceed
__device__ void blockSync() {
    __syncthreads();
}

// Warning: __syncthreads() must be called by ALL threads in the block
// or NONE. Conditional __syncthreads() causes undefined behavior.

// BAD: conditional __syncthreads__
__global__ void badSync() {
    if (threadIdx.x < 128) {
        __syncthreads();  // DEADLOCK or UNDEFINED BEHAVIOR
    }
}

// GOOD: unconditional __syncthreads__
__global__ void goodSync(float* data, int N) {
    __shared__ float tile[256];
    int tid = threadIdx.x;

    if (tid < N) tile[tid] = data[blockIdx.x * 256 + tid];
    __syncthreads();  // All threads reach here

    // Safe to read any element of tile
    if (tid < N) data[blockIdx.x * 256 + tid] = tile[255 - tid];
    __syncthreads();
}
```

### Counting/Logical Sync

```cpp
// __syncthreads_count: returns count of threads where predicate is true
__device__ void syncCount() {
    int tid = threadIdx.x;
    int count = __syncthreads_count(tid % 2 == 0);
    // count = number of threads where (tid % 2 == 0)
    if (tid == 0) printf("Even threads: %d\n", count);
}

// __syncthreads_and: returns non-zero iff ALL threads have predicate true
__device__ void syncAnd() {
    int tid = threadIdx.x;
    int allPositive = __syncthreads_and(tid >= 0);
    // allPositive != 0 only if every thread has tid >= 0
}

// __syncthreads_or: returns non-zero iff ANY thread has predicate true
__device__ void syncOr() {
    int tid = threadIdx.x;
    int anyNegative = __syncthreads_or(tid < 0);
    // anyNegative != 0 if at least one thread has tid < 0
}
```

### Warp-level Sync

```cpp
// __syncwarp: synchronize threads within a warp
// mask specifies which threads participate
__device__ void warpSync() {
    unsigned mask = 0xffffffff;  // All 32 threads
    __syncwarp(mask);            // All threads in warp wait

    // With partial mask: only lanes 0-15 participate
    unsigned half_mask = 0x0000ffff;
    if (threadIdx.x < 16) {
        __syncwarp(half_mask);
    }
}

// __syncwarp with default mask (all lanes)
__device__ void defaultWarpSync() {
    __syncwarp();  // Equivalent to __syncwarp(0xffffffff)
}
```

### Memory Fences

```cpp
// Thread fence: ensures all writes to shared memory are visible
__device__ void threadfence_block() {
    __threadfence_block();  // All writes to shared memory before this call
                             // are visible to all threads in the block
}

// Device fence: ensures writes to global memory are visible to all
// threads on the device
__device__ void threadfence() {
    __threadfence();  // All writes to global memory before this call
                       // are visible to all threads on the device
}

// System fence: ensures writes are visible to host and all devices
__device__ void threadfence_system() {
    __threadfence_system();  // All writes visible to host + all GPUs
}
```

### Cooperative Group Synchronization

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void cgSync() {
    // Thread block group
    cg::thread_block block = cg::this_thread_block();
    block.sync();  // Equivalent to __syncthreads()

    // Thread block tile (subset of block)
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    warp.sync();  // Equivalent to __syncwarp()

    // Warp-level grouped operations
    int val = threadIdx.x;
    int sum = cg::reduce(warp, val, cg::plus<int>());
    int maxVal = cg::reduce(warp, val, cg::greater<int>());

    // Grid sync (requires cooperative launch)
    // cg::grid_group grid = cg::this_grid();
    // grid.sync();
}
```

---

## 17.8 Atomic Functions

CUDA provides several categories of atomic operations:

### Categories Overview

| Category | Standard | Scope Support | Notes |
|---|---|---|---|
| Extended CUDA C++ atomics | CUDA | Thread scope suffix | Modern, recommended |
| Standard C++ atomics | C++11 | Memory order + scope | `cuda::std::atomic` |
| Compiler built-in atomics | CUDA 12.8+ | Order + scope params | Lowest level |
| Legacy atomics | CUDA | Thread scope suffix | Original CUDA atomics |

### Legacy Atomic Functions

```cpp
// Arithmetic operations
int atomicAdd(int* address, int val);
unsigned int atomicAdd(unsigned int* address, unsigned int val);
unsigned long long int atomicAdd(unsigned long long int* address, unsigned long long int val);
float atomicAdd(float* address, float val);
double atomicAdd(double* address, double val);          // CC 6.0+
__half2 atomicAdd(__half2* address, __half2 val);       // CC 6.0+
__half atomicAdd(__half* address, __half val);          // CC 7.0+
__nv_bfloat16 atomicAdd(__nv_bfloat16* addr, __nv_bfloat16 val);  // CC 8.0+

int atomicSub(int* address, int val);
unsigned int atomicSub(unsigned int* address, unsigned int val);

// Bitwise operations
int atomicAnd(int* address, int val);
unsigned int atomicAnd(unsigned int* address, unsigned int val);
unsigned long long int atomicAnd(unsigned long long int* address, unsigned long long int val);

int atomicOr(int* address, int val);
unsigned int atomicOr(unsigned int* address, unsigned int val);
unsigned long long int atomicOr(unsigned long long int* address, unsigned long long int val);

int atomicXor(int* address, int val);
unsigned int atomicXor(unsigned int* address, unsigned int val);
unsigned long long int atomicXor(unsigned long long int* address, unsigned long long int val);

// Min/Max
int atomicMin(int* address, int val);
unsigned int atomicMin(unsigned int* address, unsigned int val);
unsigned long long int atomicMin(unsigned long long int* address, unsigned long long int val);
long long int atomicMin(long long int* address, long long int val);  // CC 9.0+

int atomicMax(int* address, int val);
unsigned int atomicMax(unsigned int* address, unsigned int val);
unsigned long long int atomicMax(unsigned long long int* address, unsigned long long int val);
long long int atomicMax(long long int* address, long long int val);  // CC 9.0+

// Exchange
int atomicExch(int* address, int val);
unsigned int atomicExch(unsigned int* address, unsigned int val);
unsigned long long int atomicExch(unsigned long long int* address, unsigned long long int val);
float atomicExch(float* address, float val);

// Compare-and-swap
int atomicCAS(int* address, int compare, int val);
unsigned int atomicCAS(unsigned int* address, unsigned int compare, unsigned int val);
unsigned long long int atomicCAS(unsigned long long int* address,
                                  unsigned long long int compare,
                                  unsigned long long int val);
__half atomicCAS(__half* address, __half compare, __half val);
```

### Thread Scope Suffixes

```cpp
// Scoped variants: _block, _device (default), _system
// Example with atomicAdd:

// Block scope: only visible to threads in the same block
int atomicAdd_block(int* address, int val);

// Device scope (default): visible to all threads on the device
int atomicAdd(int* address, int val);              // Same as atomicAdd_device
int atomicAdd_device(int* address, int val);

// System scope: visible to host and all devices
int atomicAdd_system(int* address, int val);

// All legacy atomics support scope suffixes
float atomicAdd_block(float* address, float val);
unsigned int atomicCAS_system(unsigned int* address, unsigned int compare, unsigned int val);
int atomicMax_device(int* address, int val);
```

### Standard C++ Atomics (libcu++)

```cpp
#include <cuda/std/atomic>

__global__ void stdAtomicExample() {
    // cuda::std::atomic works on device
    __shared__ cuda::std::atomic<int> sharedCounter;
    __shared__ cuda::std::atomic<int> atomicArray[256];

    if (threadIdx.x == 0) {
        sharedCounter.store(0, cuda::std::memory_order_relaxed);
    }
    __syncthreads();

    // Atomic fetch and add with memory ordering
    int old = sharedCounter.fetch_add(1, cuda::std::memory_order_relaxed);

    // Compare exchange
    int expected = 10;
    bool success = sharedCounter.compare_exchange_weak(
        expected, 20,
        cuda::std::memory_order_acq_rel,
        cuda::std::memory_order_relaxed
    );

    // Load and store with memory ordering
    int val = sharedCounter.load(cuda::std::memory_order_acquire);
    sharedCounter.store(val + 1, cuda::std::memory_order_release);
}
```

### Compiler Built-in Atomics (CUDA 12.8+)

```cpp
// Lowest-level atomic operations with explicit memory order and scope
// Available in CUDA 12.8 and later

// Template: __nv_atomic_<operation>(ptr, val, order, scope)

// Memory orders:
//   __NV_ATOMIC_RELAXED   - No ordering guarantees
//   __NV_ATOMIC_ACQUIRE   - Subsequent reads see prior writes
//   __NV_ATOMIC_RELEASE   - Prior writes visible to acquiring threads
//   __NV_ATOMIC_ACQ_REL   - Both acquire and release
//   __NV_ATOMIC_SEQ_CST   - Sequentially consistent

// Scopes:
//   __NV_THREAD_SCOPE_BLOCK    - Visible within thread block
//   __NV_THREAD_SCOPE_CLUSTER  - Visible within cluster (CC 9.0+)
//   __NV_THREAD_SCOPE_DEVICE   - Visible within device
//   __NV_THREAD_SCOPE_SYSTEM   - Visible across system

__global__ void builtinAtomicExample(int* global_ptr,
                                      int* shared_ptr,
                                      int* cluster_ptr) {
    // Block-scoped relaxed atomic add
    int old = __nv_atomic_fetch_add(
        shared_ptr, 1,
        __NV_ATOMIC_RELAXED,
        __NV_THREAD_SCOPE_BLOCK
    );

    // Device-scoped release atomic exchange
    int prev = __nv_atomic_exchange(
        global_ptr, threadIdx.x,
        __NV_ATOMIC_RELEASE,
        __NV_THREAD_SCOPE_DEVICE
    );

    // System-scoped sequentially consistent CAS
    int expected = 0;
    bool success = __nv_atomic_compare_exchange(
        global_ptr, &expected, 1,
        __NV_ATOMIC_SEQ_CST, __NV_ATOMIC_SEQ_CST,
        __NV_THREAD_SCOPE_SYSTEM
    );

    // Cluster-scoped acquire atomic load (CC 9.0+)
    int val = __nv_atomic_load(
        cluster_ptr,
        __NV_ATOMIC_ACQUIRE,
        __NV_THREAD_SCOPE_CLUSTER
    );

    // Device-scoped release atomic store
    __nv_atomic_store(
        global_ptr, 42,
        __NV_ATOMIC_RELEASE,
        __NV_THREAD_SCOPE_DEVICE
    );
}
```

### Common Atomic Patterns

```cpp
// Atomic counter
__global__ void atomicCounter(unsigned int* counter, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        atomicAdd(counter, 1);
    }
}

// Atomic histogram
__global__ void histogram(const int* data, int* bins, int N, int numBins) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int bin = data[tid] % numBins;
        atomicAdd(&bins[bin], 1);
    }
}

// Atomic max tracking
__global__ void findMax(const float* data, float* maxVal, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        // Loop until we've set the maximum
        float val = data[tid];
        float old = atomicMax(reinterpret_cast<unsigned int*>(maxVal),
                              __float_as_uint(val));
        // Note: float atomicMax requires bit reinterpretation for CC < 9.0
    }
}

// Lock-free stack with atomicCAS
struct Node {
    int data;
    Node* next;
};

__device__ void push(Node** head, Node* newNode) {
    Node* oldHead;
    do {
        oldHead = *head;
        newNode->next = oldHead;
    } while (atomicCAS(reinterpret_cast<unsigned long long*>(head),
                        reinterpret_cast<unsigned long long>(oldHead),
                        reinterpret_cast<unsigned long long>(newNode))
             != reinterpret_cast<unsigned long long>(oldHead));
}
```

---

## 17.9 Warp Functions

Warp-level functions operate on groups of 32 threads (a warp) without requiring shared memory or barriers:

### Warp Mask and Membership

```cpp
// Get the mask of all currently active threads in the warp
unsigned __activemask();
// Returns a 32-bit mask where bit i is 1 if lane i is active

__global__ void activeMaskExample() {
    unsigned mask = __activemask();
    // If only lanes 0-7 are active: mask = 0x000000FF
}
```

### Warp Vote Functions

```cpp
// __all_sync: return 1 iff ALL threads in mask have predicate true
int __all_sync(unsigned mask, int predicate);

// __any_sync: return 1 iff ANY thread in mask has predicate true
int __any_sync(unsigned mask, int predicate);

// __ballot_sync: return mask of threads where predicate is true
unsigned __ballot_sync(unsigned mask, int predicate);

// Usage
__global__ void voteExample() {
    unsigned mask = 0xffffffff;  // All lanes participate
    int tid = threadIdx.x;

    // Check if all threads in warp have positive values
    int allPositive = __all_sync(mask, tid > 0);  // true if all tid > 0

    // Check if any thread has a negative value
    int anyNegative = __any_sync(mask, tid < 0);  // true if any tid < 0

    // Get bitmask of even-numbered threads
    unsigned evenMask = __ballot_sync(mask, tid % 2 == 0);
    // e.g., if lanes 0,2,4,6... are active: evenMask = 0x55555555
}
```

### Warp Match Functions

```cpp
// __match_any_sync: return mask of threads that have the same value
unsigned __match_any_sync(unsigned mask, T value);

// __match_all_sync: return mask if ALL threads have the same value, else 0
unsigned __match_all_sync(unsigned mask, T value, int* predicate);

// Usage
__global__ void matchExample() {
    unsigned mask = 0xffffffff;
    int tid = threadIdx.x;
    int val = tid % 4;  // Values: 0,1,2,3,0,1,2,3,...

    // Find all threads with the same value as this thread
    unsigned sameValMask = __match_any_sync(mask, val);
    // If tid=0 (val=0): sameValMask = bit mask of all lanes where val==0

    // Check if ALL threads have the same value
    int pred;
    unsigned allSameMask = __match_all_sync(mask, val, &pred);
    // pred = 1 if all threads have same val, 0 otherwise
}
```

### Warp Shuffle Functions

```cpp
// __shfl_sync: read value from a specific lane
T __shfl_sync(unsigned mask, T value, int srcLane, int width = 32);

// __shfl_up_sync: read value from lane with lower index (delta lanes below)
T __shfl_up_sync(unsigned mask, T value, unsigned int delta, int width = 32);

// __shfl_down_sync: read value from lane with higher index (delta lanes above)
T __shfl_down_sync(unsigned mask, T value, unsigned int delta, int width = 32);

// __shfl_xor_sync: read value from lane determined by XOR with laneMask
T __shfl_xor_sync(unsigned mask, T value, int laneMask, int width = 32);
```

### Shuffle Examples

```cpp
__global__ void shuffleExamples() {
    unsigned mask = 0xffffffff;
    int tid = threadIdx.x;
    int val = tid;

    // Direct lane read: get value from lane 0
    int fromLane0 = __shfl_sync(mask, val, 0);  // All threads get value of lane 0

    // Broadcast from lane 0
    int broadcast = __shfl_sync(mask, val, 0);

    // Shuffle up: each lane reads from (lane - delta)
    // Lane 0 gets its own value, lane i gets value from lane (i-delta)
    int upVal = __shfl_up_sync(mask, val, 4);
    // Lane 4 reads lane 0's value, lane 5 reads lane 1's value, etc.

    // Shuffle down: each lane reads from (lane + delta)
    int downVal = __shfl_down_sync(mask, val, 4);
    // Lane 0 reads lane 4's value, lane 27 reads lane 31's value

    // XOR shuffle: butterfly exchange pattern
    int xorVal = __shfl_xor_sync(mask, val, 16);
    // Lane 0 swaps with lane 16, lane 1 with lane 17, etc.
}

// Warp-level reduction using shuffle
__device__ int warpReduceSum(int val) {
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;  // Only lane 0 has the full sum
}

__global__ void blockReduceSum(const int* input, int* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (tid < N) ? input[tid] : 0;

    int warpSum = warpReduceSum(val);

    // First thread of each warp writes partial sum to shared memory
    __shared__ int warpSums[32];  // Max 32 warps per block
    int laneId = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;

    if (laneId == 0) warpSums[warpId] = warpSum;
    __syncthreads();

    // First warp reduces the partial sums
    if (warpId == 0) {
        int warpVal = (laneId < (blockDim.x + warpSize - 1) / warpSize)
                       ? warpSums[laneId] : 0;
        warpVal = warpReduceSum(warpVal);
        if (laneId == 0) {
            atomicAdd(output, warpVal);
        }
    }
}

// Warp scan (prefix sum) using shuffle
__device__ int warpScanSum(int val) {
    unsigned mask = 0xffffffff;
    int laneId = threadIdx.x % warpSize;

    // Inclusive scan using XOR shuffle
    for (int offset = 1; offset < warpSize; offset *= 2) {
        int tmp = __shfl_up_sync(mask, val, offset);
        if (laneId >= offset) val += tmp;
    }
    return val;  // Inclusive prefix sum
}
```

### Warp Reduce Functions (CC 8.0+)

```cpp
// Hardware-accelerated warp reductions (Ampere and later)
T __reduce_add_sync(unsigned mask, T value);
T __reduce_min_sync(unsigned mask, T value);
T __reduce_max_sync(unsigned mask, T value);

// These are single-instruction reductions across the warp
__global__ void warpReduceHardware() {
    unsigned mask = 0xffffffff;
    int tid = threadIdx.x;
    int val = tid * 2 + 1;  // 1, 3, 5, 7, ...

    int sum = __reduce_add_sync(mask, val);  // Sum across all 32 lanes
    int min_val = __reduce_min_sync(mask, val);  // Min across all 32 lanes
    int max_val = __reduce_max_sync(mask, val);  // Max across all 32 lanes

    if (tid == 0) {
        printf("Warp sum: %d, min: %d, max: %d\n", sum, min_val, max_val);
    }
}

// Supported types for warp reduce:
// int, unsigned int, long long, unsigned long long
// float, double
```

---

## 17.10 Other Intrinsics

### Address Space Predicates

```cpp
// Check which memory space a pointer points to
unsigned __isGlobal(const void* ptr);     // 1 if in global memory
unsigned __isShared(const void* ptr);     // 1 if in shared memory
unsigned __isConstant(const void* ptr);   // 1 if in constant memory
unsigned __isLocal(const void* ptr);      // 1 if in local memory

__global__ void addressSpaceCheck() {
    __shared__ float smem[256];
    extern __device__ float gmem[256];

    printf("smem is shared: %d\n", __isShared(smem));       // 1
    printf("smem is global: %d\n", __isGlobal(smem));       // 0
    printf("gmem is global: %d\n", __isGlobal(gmem));       // 1
    printf("gmem is shared: %d\n", __isShared(gmem));       // 0

    float local;
    printf("local is local: %d\n", __isLocal(&local));      // 1
}
```

### Address Space Conversion

```cpp
// Convert between generic and specific address spaces
// These are low-level operations for PTX-level programming

// Generic pointer to shared memory pointer
size_t __cvta_generic_to_shared(const void* ptr);

// Shared memory pointer to generic pointer
void* __cvta_shared_to_generic(size_t raw_ptr);

__global__ void addressConversion() {
    __shared__ float smem[256];

    // Convert to shared-space pointer for PTX inline assembly
    size_t smem_ptr = __cvta_generic_to_shared(smem);

    // Convert back to generic
    void* gen_ptr = __cvta_shared_to_generic(smem_ptr);
}
```

### Cache Control for Loads

```cpp
// __ldg: load through read-only/texture cache (evicts from L1, caches in L2)
T __ldg(const T* ptr);

// __ldcg: load into L1 and L2 cache (cached at all levels)
T __ldcg(const T* ptr);

// __ldca: load with cache-all policy (L1 + L2)
T __ldca(const T* ptr);

// __ldcs: load with cache-streaming policy (evict-first in L1/L2)
T __ldcs(const T* ptr);

// __ldlu: load with last-use hint (CC 8.0+)
T __ldlu(const T* ptr);

// __ldcv: load as volatile (bypass cache)
T __ldcv(const T* ptr);

__global__ void loadCacheExample(const float* readOnly,
                                  const float* streamData,
                                  float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        // Read-only data: use texture cache
        float a = __ldg(&readOnly[tid]);

        // Streaming data: read once, don't pollute cache
        float b = __ldcs(&streamData[tid]);

        output[tid] = a + b;
    }
}
```

### Cache Control for Stores

```cpp
// __stwb: store with write-back policy (L1 only, write-back to L2)
void __stwb(T* ptr, T val);

// __stcg: store with cache-global policy (L1 + L2)
void __stcg(T* ptr, T val);

// __stcs: store with cache-streaming policy (evict-first)
void __stcs(T* ptr, T val);

// __stwt: store with write-through policy (write-through to L2)
void __stwt(T* ptr, T val);

__global__ void storeCacheExample(float* output,
                                   float* streamOutput, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        // Normal write: cache in L1 and L2
        __stcg(&output[tid], tid * 2.0f);

        // Streaming write: write but don't keep in cache
        __stcs(&streamOutput[tid], tid * 3.0f);
    }
}
```

### Other Intrinsics

```cpp
// __trap: abort kernel execution (all threads terminate)
__device__ void fatal() {
    printf("Fatal error!\n");
    __trap();  // Kernel aborts
}

// __nanosleep: put thread to sleep for approximately ns nanoseconds
// Useful for reducing contention in spin-wait loops
__device__ void spinWait(volatile int* flag) {
    while (*flag == 0) {
        __nanosleep(100);  // Sleep ~100 ns between polls
    }
}

// __brkpt: trigger a breakpoint (for debugging)
__device__ void debugPoint() {
    if (threadIdx.x == 0) __brkpt();
}

// __prof_trigger: trigger a profiler event
__device__ void profilePoint() {
    __prof_trigger(0);  // Trigger profiler event 0
}
```

### DPX Instructions (CC 8.9+)

DPX instructions accelerate dynamic programming algorithms by performing fused operations in a single instruction:

```cpp
// Minimum/Maximum of three values
int __vimin3_s32(int a, int b, int c);          // min(a, b, c)
unsigned __vimin3_u32(unsigned a, unsigned b, unsigned c);
int __vimax3_s32(int a, int b, int c);          // max(a, b, c)
unsigned __vimax3_u32(unsigned a, unsigned b, unsigned c);

// Fused add-min: min(a + b, c)
int __viaddmin_s32(int a, int b, int c);        // min(a + b, c)
unsigned __viaddmin_u32(unsigned a, unsigned b, unsigned c);
long long __viaddmin_s64(long long a, long long b, long long c);

// Fused add-max: max(a + b, c)
int __viaddmax_s32(int a, int b, int c);        // max(a + b, c)
unsigned __viaddmax_u32(unsigned a, unsigned b, unsigned c);
long long __viaddmax_s64(long long a, long long b, long long c);

// Usage: Smith-Waterman alignment score
__global__ void smithWaterman(int* score, const int* seqA, const int* seqB,
                               int M, int N, int match, int mismatch, int gap) {
    for (int i = 1; i <= M; ++i) {
        for (int j = 1; j <= N; ++j) {
            int s = (seqA[i-1] == seqB[j-1]) ? match : mismatch;
            int diag = score[(i-1) * (N+1) + (j-1)] + s;
            int up   = score[(i-1) * (N+1) + j] + gap;
            int left = score[i * (N+1) + (j-1)] + gap;

            // DPX: max of three values in one instruction
            score[i * (N+1) + j] = __vimax3_s32(diag, up, left);
        }
    }
}
```

---

## 17.11 Compiler Hints

### Loop Unrolling

```cpp
// #pragma unroll: hint to unroll loops
__global__ void unrollExample(float* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Fully unroll (compiler decides factor)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        data[tid + i * N] *= 2.0f;
    }

    // Unroll with specified factor
    #pragma unroll 4
    for (int i = 0; i < 32; ++i) {
        data[tid + i * N] += 1.0f;
    }

    // Prevent unrolling
    #pragma unroll 1
    for (int i = 0; i < 1000; ++i) {
        // Large loop: don't unroll to avoid code bloat
    }
}
```

### Alignment and Assume

```cpp
// __builtin_assume_aligned: tell compiler the pointer alignment
__global__ void alignedAccess(float* data, int N) {
    // Hint: data is 16-byte aligned
    float* aligned = (float*)__builtin_assume_aligned(data, 16);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) aligned[tid] *= 2.0f;
}

// __builtin_assume: tell compiler a condition is always true
__global__ void assumeExample(int* data, int N) {
    // Hint: N is always a multiple of blockDim.x
    __builtin_assume(N % blockDim.x == 0);

    // Compiler can optimize away boundary checks
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = tid;  // No if (tid < N) needed
}

// __builtin_expect: branch prediction hint
__global__ void expectExample(int* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int val = data[tid];

        // Hint: val > 0 is the common case
        if (__builtin_expect(val > 0, 1)) {
            data[tid] = val * 2;  // Likely path
        } else {
            data[tid] = 0;        // Unlikely path
        }
    }
}

// __builtin_unreachable: tell compiler this point is unreachable
__global__ void unreachableExample(int mode) {
    switch (mode) {
        case 0: /* ... */ break;
        case 1: /* ... */ break;
        default: __builtin_unreachable();  // mode is always 0 or 1
    }
}
```

---

## 17.12 __grid_constant__ Parameters

`__grid_constant__` is a parameter annotation for `__global__` functions that prevents per-thread copies of kernel parameters, providing both memory efficiency and a guaranteed read-only guarantee.

### Overview

When a kernel receives a parameter by value, each thread may make its own copy in local memory. `__grid_constant__` prevents this by keeping a single copy that is:

- Read-only for the entire grid
- At the same address for all threads
- Guaranteed to have kernel lifetime (persists until kernel completes)

### Requirements

- Must be used on `__global__` function parameters only
- The parameter must be `const`-qualified
- The parameter must be a non-reference type (passed by value)
- The parameter must not be modified within the kernel

### Syntax and Usage

```cpp
// Basic usage: const-qualified struct parameter
struct KernelParams {
    int width;
    int height;
    float scale;
    float offset;
};

__global__ void kernelWithGridConstant(
    const __grid_constant__ KernelParams params,
    float* output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // params is read-only and shared by all threads
    if (x < params.width && y < params.height) {
        int idx = y * params.width + x;
        output[idx] = idx * params.scale + params.offset;
    }
}

// Launch as usual
void launchKernel() {
    KernelParams params = {1920, 1080, 0.5f, 1.0f};
    dim3 blockSize(16, 16);
    dim3 gridSize((params.width + 15) / 16, (params.height + 15) / 16);
    kernelWithGridConstant<<<gridSize, blockSize>>>(params, d_output);
}
```

### Benefits

```cpp
// WITHOUT __grid_constant__: each thread gets its own copy
__global__ void withoutGridConstant(KernelParams params, float* output) {
    // 'params' may be copied per-thread into local memory
    // This wastes registers/local memory for large structs
    // No read-only guarantee: any thread could modify its copy
    output[threadIdx.x] = params.scale;  // params is in local memory
}

// WITH __grid_constant__: single shared read-only copy
__global__ void withGridConstant(const __grid_constant__ KernelParams params,
                                  float* output) {
    // 'params' is in constant/parameter memory, shared by all threads
    // Read-only: compiler can optimize loads aggressively
    // No per-thread memory overhead
    output[threadIdx.x] = params.scale;  // params is in parameter space
}
```

### Large Parameter Example

```cpp
struct LargeConfig {
    float weights[256];   // 1 KB
    int indices[64];      // 256 bytes
    float bias;
    int layers;
};

// Without __grid_constant__: 1280+ bytes per thread -> excessive local memory
// With __grid_constant__: single copy, all threads share
__global__ void largeParamKernel(
    const __grid_constant__ LargeConfig config,
    float* input, float* output, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        float sum = config.bias;
        for (int i = 0; i < 64; ++i) {
            sum += config.weights[config.indices[i]] * input[tid + i];
        }
        output[tid] = sum / config.layers;
    }
}
```

---

## 17.13 Annotation Summary

The following table summarizes all CUDA C++ function and parameter annotations:

| Annotation | Applicable To | __host__/__device__ | __global__ | Purpose |
|---|---|---|---|---|
| `__noinline__` | Function | Yes | No | Prevent inlining |
| `__forceinline__` | Function | Yes | No | Force inlining |
| `__restrict__` | Pointer parameter | Yes | Yes | No aliasing guarantee |
| `__grid_constant__` | Parameter | No | Yes (const param) | Prevent per-thread copy |
| `__launch_bounds__` | Function | No | Yes | Guide register allocation |
| `__maxnreg__` | Function | No | Yes (CC 9.0+) | Set max registers per thread |
| `__cluster_dims__` | Function | No | Yes (CC 9.0+) | Set cluster dimensions |

### Inline Control

```cpp
// __noinline__: prevent function from being inlined
__device__ __noinline__ int expensiveFunction(int x) {
    // Large function: don't inline to save code size
    int result = 0;
    for (int i = 0; i < 1000; ++i) result += x * i;
    return result;
}

// __forceinline__: force function to be inlined
__device__ __forceinline__ int fastFunction(int x) {
    // Small hot function: always inline for performance
    return x * x + 2 * x + 1;
}
```

### Restrict Pointers

```cpp
// __restrict__: tell compiler that pointers do not alias
// Allows more aggressive optimization (vectorization, reordering)
__global__ void addKernel(
    float* __restrict__ output,
    const float* __restrict__ inputA,
    const float* __restrict__ inputB,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        // Compiler knows output, inputA, inputB don't overlap
        output[tid] = inputA[tid] + inputB[tid];
    }
}
```

### __maxnreg__ (CC 9.0+)

```cpp
// Explicit register limit per thread
__global__ void __maxnreg__(64)
registerLimitedKernel(float* data, int N) {
    // Each thread uses at most 64 registers
    // Excess spills to local memory
    // Useful for fine-grained occupancy control

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        data[tid] *= 2.0f;
    }
}

// Combining __launch_bounds__ and __maxnreg__
__global__ void __launch_bounds__(256, 8) __maxnreg__(32)
fullyControlledKernel(float* data, int N) {
    // 256 threads, min 8 blocks/SM, max 32 registers/thread
    // 65536 / (256 * 8) = 32 regs/thread, so __maxnreg__(32) is consistent
}
```

### __cluster_dims__ (CC 9.0+)

```cpp
// Specify the number of thread blocks per cluster
__global__ void __cluster_dims__(X, Y, Z)
clusterKernel(...) { }

// 1D cluster: 4 blocks per cluster in x-dimension
__global__ void __cluster_dims__(4, 1, 1)
cluster4x() {
    namespace cg = cooperative_groups;
    auto cluster = cg::this_cluster();
    printf("Cluster size: %d blocks\n",
           cluster.dim_blocks().x * cluster.dim_blocks().y * cluster.dim_blocks().z);
}

// 2D cluster: 2x2 blocks
__global__ void __cluster_dims__(2, 2, 1)
cluster2x2() {
    namespace cg = cooperative_groups;
    auto cluster = cg::this_cluster();
    dim3 clusterDim = cluster.dim_blocks();
    dim3 clusterRank = cluster.block_index();

    __shared__ int smem[256];
    smem[threadIdx.x] = threadIdx.x;
    cluster.sync();  // Synchronize all blocks in cluster

    // Access shared memory from another block in the cluster
    int* remote = cluster.map_shared_rank(smem, (clusterRank.x + 1) % clusterDim.x);
    int remoteVal = remote[threadIdx.x];
}
```

### Combined Annotation Example

```cpp
// Production kernel with all annotations
__global__ void __launch_bounds__(256, 6)
               __maxnreg__(40)
               __cluster_dims__(2, 1, 1)
productionKernel(
    const __grid_constant__ KernelConfig config,
    const float* __restrict__ input,
    float* __restrict__ output)
{
    __shared__ float tile[256];
    namespace cg = cooperative_groups;
    auto cluster = cg::this_cluster();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load with cache hint
    float val = __ldcs(&input[tid]);

    // Process using grid constant config
    tile[threadIdx.x] = val * config.scale + config.offset;
    cg::this_thread_block().sync();

    // Distributed shared memory access within cluster
    if (blockIdx.x % 2 == 0) {
        float* neighborTile = cluster.map_shared_rank(tile, blockIdx.x + 1);
        val = (tile[threadIdx.x] + neighborTile[threadIdx.x]) * 0.5f;
    }

    // Store with streaming hint
    __stcs(&output[tid], val);
}
```
