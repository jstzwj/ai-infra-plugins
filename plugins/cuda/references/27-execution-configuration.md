# 27. Execution Configuration

This document covers best practices for configuring kernel execution parameters in CUDA, including occupancy calculation, thread and block size heuristics, concurrent kernel execution, shared memory configuration effects, and multiple context management. Proper execution configuration is essential for fully utilizing GPU hardware resources.

---

## Table of Contents

1. [Occupancy Calculation](#271-occupancy-calculation)
2. [Thread and Block Heuristics](#272-thread-and-block-heuristics)
3. [Concurrent Kernel Execution](#273-concurrent-kernel-execution)
4. [Shared Memory Effects](#274-shared-memory-effects)
5. [Multiple Contexts](#275-multiple-contexts)

---

## 27.1 Occupancy Calculation

Occupancy is the ratio of active warps per streaming multiprocessor (SM) to the maximum number of active warps that the SM supports. Higher occupancy generally means better latency hiding, since the SM can switch between warps to cover memory access latencies. However, maximum occupancy does not always guarantee maximum performance -- other factors such as instruction-level parallelism, memory bandwidth, and register count per thread also matter.

### 27.1.1 Occupancy Formula

```
Occupancy = Active Warps / Max Warps per SM
```

The number of active warps is limited by three hardware constraints. The most restrictive constraint determines the actual occupancy:

1. **Maximum threads per SM** (limits blocks based on threads)
2. **Maximum blocks per SM** (limits blocks based on count)
3. **Maximum registers per SM** (limits blocks based on register usage)

```
Blocks per SM = min(
    MaxThreads / ThreadsPerBlock,                    // Thread limit
    MaxBlocks,                                        // Block count limit
    floor(MaxRegisters / (ThreadsPerBlock * RegsPerThread))  // Register limit
)

Active Warps = Blocks per SM * (ThreadsPerBlock / WarpSize)
Occupancy = Active Warps / Max Warps
```

### 27.1.2 Hardware Limits by Compute Capability

| Resource | CC 7.5 | CC 8.0 | CC 8.6/8.9 | CC 9.0 | CC 10.0/11.0 |
|----------|--------|--------|------------|--------|--------------|
| Max threads/SM | 1024 | 2048 | 1536 | 2048 | 2048 |
| Max blocks/SM | 16 | 32 | 16/24 | 32 | 32 |
| Max 32-bit regs/SM | 65536 | 65536 | 65536 | 65536 | 65536 |
| Max regs/thread | 255 | 255 | 255 | 255 | 255 |
| Warp size | 32 | 32 | 32 | 32 | 32 |
| Max warps/SM | 32 | 64 | 48 | 64 | 64 |

### 27.1.3 Register Limit Is the Key Constraint

Register availability is often the primary limiter of occupancy. Each thread uses a certain number of registers (determined at compile time), and the SM has a finite register file. More registers per thread means fewer concurrent threads, reducing the SM's ability to hide latency.

**Example for CC 7.0:**

- 65,536 registers per SM
- 2,048 max threads per SM
- For 100% occupancy (2,048 threads): max 32 registers per thread (65536 / 2048 = 32)
- For 64 registers per thread: max 1,024 threads (65536 / 64 / 32 = 32 warps, i.e. 50% occupancy)
- For 128 registers per thread: max 512 threads (25% occupancy)

```cpp
// Check register usage per thread at compile time
// nvcc --ptxas-options=-v mykernel.cu
// Output example:
// ptxas info: Used 32 registers, 1024 bytes smem, 68 bytes cmem[0]

// Runtime occupancy calculation
#include <cuda_profiler_api.h>

void computeOccupancy(int regsPerThread, int threadsPerBlock, int sharedMemPerBlock) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks,
        myKernel,           // Kernel function
        threadsPerBlock,    // Block size
        sharedMemPerBlock   // Dynamic shared memory per block
    );

    float occupancy = (maxActiveBlocks * threadsPerBlock / prop.warpSize)
                    / (float)(prop.maxThreadsPerMultiProcessor / prop.warpSize);

    printf("Threads/block:     %d\n", threadsPerBlock);
    printf("Registers/thread:  %d\n", regsPerThread);
    printf("Active blocks/SM:  %d\n", maxActiveBlocks);
    printf("Active warps/SM:   %d\n",
           maxActiveBlocks * threadsPerBlock / prop.warpSize);
    printf("Occupancy:         %.1f%%\n", occupancy * 100.0f);
}
```

### 27.1.4 Occupancy Calculation Examples

```cpp
// Example 1: CC 7.0 kernel with 128 threads/block, 32 regs/thread
// MaxThreads/SM = 2048, MaxBlocks/SM = 32, MaxRegs/SM = 65536
//
// Thread limit: 2048 / 128 = 16 blocks
// Block limit:  32 blocks
// Reg limit:    floor(65536 / (128 * 32)) = floor(16) = 16 blocks
// Active blocks = min(16, 32, 16) = 16
// Active warps = 16 * (128/32) = 64
// Max warps = 2048/32 = 64
// Occupancy = 64/64 = 100%

// Example 2: CC 7.0 kernel with 256 threads/block, 48 regs/thread
// Thread limit: 2048 / 256 = 8 blocks
// Block limit:  32 blocks
// Reg limit:    floor(65536 / (256 * 48)) = floor(5.33) = 5 blocks
// Active blocks = min(8, 32, 5) = 5
// Active warps = 5 * (256/32) = 40
// Occupancy = 40/64 = 62.5%

// Example 3: CC 9.0 kernel with 128 threads/block, 40 regs/thread
// MaxThreads/SM = 2048, MaxBlocks/SM = 32, MaxRegs/SM = 65536
// Thread limit: 2048 / 128 = 16 blocks
// Block limit:  32 blocks
// Reg limit:    floor(65536 / (128 * 40)) = floor(12.8) = 12 blocks
// Active blocks = min(16, 32, 12) = 12
// Active warps = 12 * 4 = 48
// Max warps = 2048/32 = 64
// Occupancy = 48/64 = 75%
```

### 27.1.5 Controlling Register Usage

```cpp
// Method 1: __launch_bounds__ directive (compile-time)
// __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
// The compiler adjusts register allocation to satisfy the constraint

__global__ void __launch_bounds__(256, 8)   // 256 threads, at least 8 blocks/SM
myKernel(...) {
    // Compiler will limit registers to allow 8 blocks/SM on the target
    // For CC 7.0: maxRegs = 65536 / (256 * 8) = 32 regs/thread
}

// Method 2: __maxnreg__ directive (CC 9.0+)
// Explicitly sets the maximum number of registers per thread

__global__ void __maxnreg__(64)   // Limit to 64 registers per thread
myKernel2(...) {
    // Compiler will not allocate more than 64 registers per thread
    // Spills to local memory if more are needed
}

// Method 3: Compiler flag --maxrregcount
// nvcc --maxrregcount=32 mykernel.cu
// Applies globally to all kernels in the file
// Less flexible than per-kernel __launch_bounds__
```

### 27.1.6 Occupancy API

```cpp
// Comprehensive occupancy analysis at runtime
void analyzeOccupancy() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Calculate max potential block size
    int minGridSize, optimalBlockSize;
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &optimalBlockSize,
        myKernel,
        0,    // dynamic shared memory per block
        0     // block size limit (0 = no limit)
    );

    printf("Optimal block size: %d threads\n", optimalBlockSize);
    printf("Minimum grid size:  %d blocks\n", minGridSize);

    // Calculate with shared memory constraint
    int blockSize = 256;
    size_t dynamicSharedMem = 48 * 1024;  // 48 KB

    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, myKernel, blockSize, dynamicSharedMem);

    float occupancy = (maxActiveBlocks * blockSize / (float)prop.maxThreadsPerMultiProcessor) * 100.0f;

    printf("Block size:         %d\n", blockSize);
    printf("Dynamic shared mem: %zu bytes\n", dynamicSharedMem);
    printf("Active blocks/SM:   %d\n", maxActiveBlocks);
    printf("Occupancy:          %.1f%%\n", occupancy);

    // Find the optimal block size for maximum occupancy
    printf("\nOccupancy by block size:\n");
    for (int bs = 32; bs <= 1024; bs += 32) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks, myKernel, bs, 0);
        float occ = (maxActiveBlocks * bs / (float)prop.maxThreadsPerMultiProcessor) * 100.0f;
        printf("  %4d threads: %2d blocks, %.1f%% occupancy\n",
               bs, maxActiveBlocks, occ);
    }
}
```

---

## 27.2 Thread and Block Heuristics

### 27.2.1 Block Size Guidelines

- Block size must be a multiple of the warp size (32) to avoid wasted threads and ensure efficient scheduling.
- Start with 128 or 256 threads per block and tune from there.
- 256 threads per block is a good default for most memory-bound kernels.
- 128 threads per block may be better for compute-bound kernels that need more registers per thread.

```cpp
// Safe block size selection
int blockSize = 256;  // Good starting point

// Ensure multiple of warp size
blockSize = ((blockSize + 31) / 32) * 32;

// Ensure within hardware limit
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
if (blockSize > prop.maxThreadsPerBlock) {
    blockSize = prop.maxThreadsPerBlock;
}

int gridSize = (N + blockSize - 1) / blockSize;
```

### 27.2.2 Grid Size Guidelines

- The total number of blocks should significantly exceed the number of SMs to enable load balancing and latency hiding.
- Aim for thousands of blocks for large problems.
- At minimum, use at least 2-4 blocks per SM to allow the scheduler to switch between blocks.

```cpp
// Calculate grid size for full GPU utilization
int numSMs;
cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

// Target: at least 10-20 blocks per SM
int targetBlocksPerSM = 10;
int minBlocks = numSMs * targetBlocksPerSM;

int blockSize = 256;
int gridSize = (N + blockSize - 1) / blockSize;

// Ensure enough blocks for full utilization
if (gridSize < minBlocks) {
    // If the problem is too small, reduce block size to create more blocks
    // (as long as it stays a multiple of 32)
    blockSize = max(32, (N + minBlocks - 1) / minBlocks);
    blockSize = min(blockSize, 256);
    blockSize = ((blockSize + 31) / 32) * 32;  // Round up to warp multiple
    gridSize = (N + blockSize - 1) / blockSize;
}

printf("Launch config: %d blocks x %d threads = %d total threads\n",
       gridSize, blockSize, gridSize * blockSize);
```

### 27.2.3 Choosing Between 1D, 2D, and 3D Grids

```cpp
// 1D grid: best for linear data (vectors, flat arrays)
int blockSize1D = 256;
int gridSize1D = (N + blockSize1D - 1) / blockSize1D;
kernel1D<<<gridSize1D, blockSize1D>>>(d_data, N);

// 2D grid: best for 2D data (images, matrices)
dim3 blockSize2D(16, 16);  // 256 threads per block
dim3 gridSize2D((width + 15) / 16, (height + 15) / 16);
kernel2D<<<gridSize2D, blockSize2D>>>(d_image, width, height);

// 3D grid: best for volumetric data (3D simulations, medical imaging)
dim3 blockSize3D(8, 8, 8);  // 512 threads per block (stay under 1024)
dim3 gridSize3D((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
kernel3D<<<gridSize3D, blockSize3D>>>(d_volume, nx, ny, nz);
```

### 27.2.4 Adaptive Launch Configuration

```cpp
// Auto-tune block size at runtime for a specific kernel
template <typename KernelFunc>
int autoTuneBlockSize(KernelFunc kernel, int N) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int bestBlockSize = 256;
    float bestOccupancy = 0.0f;

    // Test common block sizes
    int blockSizes[] = {32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 512, 768, 1024};

    for (int bs : blockSizes) {
        if (bs > prop.maxThreadsPerBlock) continue;

        int maxActiveBlocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks, kernel, bs, 0);

        float occupancy = (maxActiveBlocks * bs) /
                          (float)prop.maxThreadsPerMultiProcessor;

        if (occupancy > bestOccupancy) {
            bestOccupancy = occupancy;
            bestBlockSize = bs;
        }
    }

    printf("Auto-tuned block size: %d (occupancy: %.1f%%)\n",
           bestBlockSize, bestOccupancy * 100.0f);
    return bestBlockSize;
}

// Usage
int blockSize = autoTuneBlockSize(myKernel, N);
int gridSize = (N + blockSize - 1) / blockSize;
myKernel<<<gridSize, blockSize>>>(d_data, N);
```

---

## 27.3 Concurrent Kernel Execution

Multiple kernels can execute concurrently on the GPU if they are launched in different non-default streams. This allows small kernels that cannot fill the GPU individually to share SM resources.

### 27.3.1 Basic Concurrent Execution

```cpp
// Kernels in different streams execute concurrently
cudaStream_t stream1, stream2, stream3;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
cudaStreamCreate(&stream3);

// These three kernels run concurrently on different SMs
kernelA<<<gridA, blockA, 0, stream1>>>(d_dataA);
kernelB<<<gridB, blockB, 0, stream2>>>(d_dataB);
kernelC<<<gridC, blockC, 0, stream3>>>(d_dataC);

cudaDeviceSynchronize();
```

### 27.3.2 Dependencies Between Concurrent Kernels

```cpp
// Use events to create dependencies between streams
cudaStream_t s1, s2;
cudaEvent_t e1;

cudaStreamCreate(&s1);
cudaStreamCreate(&s2);
cudaEventCreateWithFlags(&e1, cudaEventDisableTiming);

// Kernel A in stream 1, then record event
kernelA<<<grid, block, 0, s1>>>(d_A);
cudaEventRecord(e1, s1);

// Stream 2 waits for stream 1's event before launching kernel B
cudaStreamWaitEvent(s2, e1, 0);
kernelB<<<grid, block, 0, s2>>>(d_B);  // Waits for kernelA

// Stream 1 can do other work concurrently with kernelB
kernelC<<<grid, block, 0, s1>>>(d_C);  // Runs concurrently with kernelB

cudaDeviceSynchronize();
```

### 27.3.3 Limitations of Concurrent Execution

- The default (legacy) stream serializes with all other streams. Always use non-default streams for concurrency.
- Kernels that saturate all SMs leave no room for concurrent execution.
- Concurrent kernels share the register file, shared memory, and L1 cache, which may reduce per-kernel performance.
- The number of concurrent connections is limited by `CUDA_DEVICE_MAX_CONNECTIONS` (default: 8).

```cpp
// Create non-blocking streams for maximum concurrency
cudaStream_t streams[NUM_STREAMS];
for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
}

// Launch multiple independent kernels concurrently
for (int i = 0; i < NUM_STREAMS; i++) {
    kernel<<<smallGrid, block, 0, streams[i]>>>(d_data[i], chunkSize);
}

cudaDeviceSynchronize();
```

---

## 27.4 Shared Memory Effects

The amount of shared memory used by a kernel affects occupancy because shared memory is a finite resource. On GPUs with unified data cache (CC 7.0+), the split between shared memory and L1 cache is configurable.

### 27.4.1 Shared Memory Impact on Occupancy

```cpp
// Shared memory limits occupancy
// Example: CC 8.0 with 164 KB shared memory per SM

// Kernel using 48 KB shared memory per block
// Max blocks = floor(164 KB / 48 KB) = 3 blocks
// With 256 threads/block: 3 * 256 = 768 threads -> 24 warps -> 37.5% occupancy

// Kernel using 16 KB shared memory per block
// Max blocks = floor(164 KB / 16 KB) = 10 blocks (limited by other constraints)
// With 256 threads/block: 10 * 256 = 2560 -> clamped to 2048 -> 64 warps -> 100% occupancy

void checkSharedMemImpact(size_t sharedMemPerBlock, int threadsPerBlock) {
    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, myKernel, threadsPerBlock, sharedMemPerBlock);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float occupancy = (maxActiveBlocks * threadsPerBlock) /
                      (float)prop.maxThreadsPerMultiProcessor * 100.0f;

    printf("Shared mem/block: %zu bytes -> %d blocks -> %.1f%% occupancy\n",
           sharedMemPerBlock, maxActiveBlocks, occupancy);
}
```

### 27.4.2 Configuring Cache Preference

```cpp
// Prefer shared memory (less L1 cache)
// Use when kernel heavily uses shared memory
cudaFuncSetCacheConfig(myKernel, cudaFuncCachePreferShared);

// Prefer L1 cache (less shared memory)
// Use when kernel doesn't use shared memory but benefits from caching
cudaFuncSetCacheConfig(myKernel, cudaFuncCachePreferL1);

// Equal split
cudaFuncSetCacheConfig(myKernel, cudaFuncCachePreferEqual);

// Prefer no L1 cache (maximum shared memory)
cudaFuncSetCacheConfig(myKernel, cudaFuncCachePreferNone);
```

### 27.4.3 Dynamic Shared Memory

Dynamic shared memory allows the amount of shared memory to be specified at kernel launch time rather than compile time, enabling the same kernel to adapt to different problem sizes.

```cpp
// Declare dynamic shared memory with extern
__global__ void dynamicSharedKernel(const float* input, float* output, int N) {
    // Dynamic shared memory: size specified at launch time
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    if (idx < N) {
        sdata[tid] = input[idx];
    }
    __syncthreads();

    // Process with shared memory
    if (idx < N) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            sum += sdata[i];
        }
        output[idx] = sum;
    }
}

// Multiple dynamic shared memory arrays using manual offset
__global__ void multiDynamicShared(int N) {
    // All dynamic shared memory is a single contiguous block
    extern __shared__ char dynamicSmem[];

    // Partition it manually
    float* arrayA = (float*)dynamicSmem;
    float* arrayB = (float*)(dynamicSmem + N * sizeof(float));

    arrayA[threadIdx.x] = threadIdx.x;
    arrayB[threadIdx.x] = threadIdx.x * 2.0f;
    __syncthreads();
}

// Launch with dynamic shared memory
int smemSize = N * sizeof(float);  // For single array
dynamicSharedKernel<<<grid, block, smemSize, stream>>>(d_in, d_out, N);

// For multiple arrays, sum all sizes
int totalSmem = 2 * N * sizeof(float);  // arrayA + arrayB
multiDynamicShared<<<grid, block, totalSmem, stream>>>(N);
```

### 27.4.4 Shared Memory Bank Size Configuration

```cpp
// Configure bank size: 4 bytes (default) or 8 bytes
// 8-byte mode helps with double-precision access patterns

// Set to 4-byte banks (default, best for float/int)
cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

// Set to 8-byte banks (better for double)
cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

// When using 8-byte banks:
// __shared__ double data[32];
// Thread i reads data[i] -> 8-byte bank i -> no conflict
// But the effective number of banks is still 32
```

---

## 27.5 Multiple Contexts

A CUDA context is analogous to a CPU process -- it encapsulates all GPU state including memory allocations, kernel modules, and streams. Each host thread that uses CUDA has an associated context for each GPU.

### 27.5.1 The Primary Context

Every GPU has a primary context that is shared by default across all host threads using the Runtime API. The Runtime API implicitly creates and manages this context.

```cpp
// The Runtime API automatically uses the primary context
// No explicit context management needed for most applications
cudaSetDevice(0);  // Selects GPU 0's primary context
cudaMalloc(&d_ptr, size);  // Allocates in the primary context
kernel<<<grid, block>>>(d_ptr);  // Launches in the primary context
```

### 27.5.2 Avoid Creating Multiple Contexts

Creating multiple contexts on the same GPU should be avoided because:

1. **Resource fragmentation**: Each context has its own memory space. Memory allocated in one context is not accessible from another.
2. **Context switching overhead**: The GPU must save and restore state when switching between contexts, causing stalls.
3. **Reduced occupancy**: Multiple active contexts share SM resources, reducing the number of concurrent threads each context can use.

```cpp
// BAD: Creating multiple contexts on the same GPU
// (Using the Driver API)
CUcontext ctx1, ctx2;
cuCtxCreate(&ctx1, 0, device);
cuCtxCreate(&ctx2, 0, device);  // Second context on same GPU!
// Both contexts compete for GPU resources

// GOOD: Use the primary context, share across threads
// Thread 1:
cudaSetDevice(0);
kernelA<<<grid, block>>>(d_dataA);

// Thread 2:
cudaSetDevice(0);  // Same primary context
kernelB<<<grid, block>>>(d_dataB);

// Or use streams for concurrency within a single context
cudaStream_t s1, s2;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);
kernelA<<<grid, block, 0, s1>>>(d_dataA);
kernelB<<<grid, block, 0, s2>>>(d_dataB);
```

### 27.5.3 Primary Context Management

```cpp
// Explicitly manage the primary context (Driver API interoperability)
CUcontext primaryCtx;
cuDevicePrimaryCtxRetain(&primaryCtx, 0);  // Retain primary context for device 0
cuCtxPushCurrent(primaryCtx);

// Use CUDA Driver API or Runtime API interchangeably
// Both operate on the same context

cuCtxPopCurrent(NULL);
cuDevicePrimaryCtxRelease(0);  // Release when done

// Reset primary context (frees all resources)
cuDevicePrimaryCtxReset(0);  // Destroys all allocations, modules, etc.
```

### 27.5.4 Multi-Device Contexts

Using multiple GPUs is different from using multiple contexts on the same GPU. Each GPU naturally has its own context, and work is distributed across them.

```cpp
// Correct multi-GPU usage: one context per GPU
int numDevices;
cudaGetDeviceCount(&numDevices);

// Each device has its own primary context
for (int dev = 0; dev < numDevices; dev++) {
    cudaSetDevice(dev);
    cudaMalloc(&d_ptrs[dev], size);
    kernel<<<grid, block, 0, streams[dev]>>>(d_ptrs[dev], chunkSize);
}

// Synchronize all devices
for (int dev = 0; dev < numDevices; dev++) {
    cudaSetDevice(dev);
    cudaDeviceSynchronize();
}
```

### 27.5.5 Stream-Ordered Memory Allocator and Contexts

Using `cudaMallocAsync` / `cudaFreeAsync` avoids some context-related synchronization issues because the allocations are stream-ordered and do not implicitly synchronize.

```cpp
// Stream-ordered allocation avoids implicit sync across streams
cudaStream_t s1, s2;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);

// Allocate in stream order (no implicit device-wide sync)
float* d1;
cudaMallocAsync(&d1, size, s1);

float* d2;
cudaMallocAsync(&d2, size, s2);

// Both allocations can proceed concurrently
kernel1<<<grid, block, 0, s1>>>(d1);
kernel2<<<grid, block, 0, s2>>>(d2);

// Free in stream order
cudaFreeAsync(d1, s1);
cudaFreeAsync(d2, s2);

cudaDeviceSynchronize();
```

---

## Summary

| Topic | Key Takeaway |
|---|---|
| **Occupancy** | Ratio of active warps to max warps per SM; register usage is often the limiting factor |
| **Register control** | Use `__launch_bounds__` or `--ptxas-options=-v` to manage register pressure |
| **Block size** | Must be a multiple of 32; start with 128-256 and tune |
| **Grid size** | Thousands of blocks preferred; at least 2-4 blocks per SM |
| **Concurrent kernels** | Use non-default streams; beware of resource sharing overhead |
| **Shared memory** | Affects occupancy; use dynamic shared memory for flexibility |
| **Cache config** | `cudaFuncSetCacheConfig()` controls shared/L1 split |
| **Multiple contexts** | Avoid on same GPU; use primary context and streams instead |
| **Multi-GPU** | Each GPU gets its own context; use `cudaSetDevice()` to switch |
