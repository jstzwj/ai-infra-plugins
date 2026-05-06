# CUDA Programming Guide - Chapter 2: Getting Started with CUDA C++

This reference covers the practical fundamentals of writing, compiling, and debugging CUDA C++ programs, including kernel syntax, memory management, synchronization, error handling, and the NVCC compilation pipeline.

---

## 2.1 Kernels

### Declaration

A **kernel** is a function that executes on the GPU. Kernels are declared using the `__global__` specifier and must have a `void` return type:

```cpp
__global__ void kernelName(parameters...) {
    // Code that executes on the GPU
}
```

Rules for kernel functions:

- Must return `void`.
- Can be called from host code (or from device code on CC 3.5+ with dynamic parallelism).
- Support most C++ features: classes, templates, lambdas (device), operator overloading.
- Cannot use variable argument lists (`...`), static variables inside the function, or function pointers as kernel parameters (with exceptions).

### Launch Syntax

Kernels are launched using **triple-chevron** (`<<<>>>`) notation:

```cpp
kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args...);
```

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `gridDim` | `dim3` or `int` | Number of thread blocks in the grid | Yes |
| `blockDim` | `dim3` or `int` | Number of threads per block | Yes |
| `sharedMemBytes` | `unsigned int` | Bytes of dynamic shared memory per block | No (default 0) |
| `stream` | `cudaStream_t` | CUDA stream for asynchronous execution | No (default stream 0) |

### Kernel Launch Constraints

- Maximum threads per block: **1024** (hardware limit).
- Maximum block dimensions: `(1024, 1024, 64)`, with `x * y * z <= 1024`.
- Maximum grid dimensions: `(2^31 - 1, 65535, 65535)` for CC 3.0+.
- Kernel launches are **asynchronous** with respect to the host thread. Control returns to the host immediately, and the kernel begins executing on the GPU.

```cpp
// Basic vector addition kernel
__global__ void vecAdd(float* A, float* B, float* C, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Launch with 1 block of 256 threads
vecAdd<<<1, 256>>>(d_A, d_B, d_C, N);

// Launch with enough blocks to cover all elements
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

// Launch with 2D grid and 2D blocks (e.g., for image processing)
dim3 blockDim(16, 16);  // 256 threads per block
dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
             (height + blockDim.y - 1) / blockDim.y);
imageKernel<<<gridDim, blockDim>>>(d_image, width, height);
```

---

## 2.2 Thread and Grid Index Intrinsics

CUDA provides built-in variables inside kernel code that allow each thread to identify its position within the block and grid.

### Built-in Variables

| Variable | Type | Description |
|----------|------|-------------|
| `threadIdx.x/y/z` | `int` | Thread index within its block (0-based) |
| `blockDim.x/y/z` | `int` | Dimensions of the block (threads per dimension) |
| `blockIdx.x/y/z` | `int` | Block index within the grid (0-based) |
| `gridDim.x/y/z` | `int` | Dimensions of the grid (blocks per dimension) |
| `warpSize` | `int` | Number of threads per warp (32) |

### Common Indexing Patterns

```cpp
// 1D index for 1D grid of 1D blocks
int idx1D = threadIdx.x + blockDim.x * blockIdx.x;

// 2D index for 2D grid of 2D blocks (row-major)
int row = threadIdx.y + blockDim.y * blockIdx.y;
int col = threadIdx.x + blockDim.x * blockIdx.x;
int idx2D = row * width + col;

// Linear index for 3D grid of 3D blocks
int idx3D = threadIdx.x + blockDim.x * blockIdx.x
          + (threadIdx.y + blockDim.y * blockIdx.y) * blockDim.x * gridDim.x
          + (threadIdx.z + blockDim.z * blockIdx.z) * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

// Flat index for arbitrary dimensionality
int flatIdx()
{
    // For 1D: just threadIdx.x + blockDim.x * blockIdx.x
    // For 2D/3D: compute as above
}
```

### Grid-Stride Loop

For processing arrays larger than what a single grid launch can cover, or for better load balancing, use the **grid-stride loop** pattern:

```cpp
__global__ void gridStrideKernel(float* data, int N, float alpha) {
    int stride = blockDim.x * gridDim.x;
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < N; i += stride) {
        data[i] *= alpha;
    }
}

// Launch with a reasonable grid size (e.g., number of SMs * occupancy)
int blockSize = 256;
int numSMs;
cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
int gridSize = numSMs * 16;  // oversubscribe for better occupancy
gridStrideKernel<<<gridSize, blockSize>>>(d_data, N, 2.0f);
```

Benefits of grid-stride loops:

- Works with any grid size (no need to compute exact block count).
- Better cache utilization due to sequential access pattern.
- Scalable across GPU generations with different SM counts.

### Bounds Checking

Always use bounds checking when the data size is not an exact multiple of the grid:

```cpp
int threads = 256;
int blocks = (N + threads - 1) / threads;  // ceiling division
kernel<<<blocks, threads>>>(d_data, N);

// Inside kernel:
__global__ void kernel(float* data, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        data[idx] *= 2.0f;
    }
    // Threads beyond N do nothing
}
```

---

## 2.3 Memory in GPU Computing

### 2.3.1 Unified Memory

Unified Memory provides a single address space accessible from both host and device code. It simplifies memory management by eliminating the need for explicit `cudaMemcpy` calls.

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void saxpy(int n, float alpha, float* x, float* y) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) y[i] = alpha * x[i] + y[i];
}

int main() {
    int N = 1 << 20;
    float *x, *y;

    // Allocate unified memory
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // Initialize on host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    saxpy<<<numBlocks, blockSize>>>(N, 3.0f, x, y);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Access results on host -- no explicit copy needed
    printf("y[0] = %f (expected 5.0)\n", y[0]);

    cudaFree(x);
    cudaFree(y);
    return 0;
}
```

The `__managed__` specifier provides an alternative syntax for global variables:

```cpp
__managed__ float data[1024];  // Unified memory global variable

__global__ void process() {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < 1024) data[idx] *= 2.0f;
}

int main() {
    for (int i = 0; i < 1024; i++) data[i] = float(i);  // host access
    process<<<1, 1024>>>();
    cudaDeviceSynchronize();
    printf("%f\n", data[0]);  // host access after kernel
    return 0;
}
```

### 2.3.2 Explicit Memory Management

For maximum control and performance, CUDA provides explicit device memory allocation and copy APIs:

```cpp
// Allocate device memory
float* d_data;
cudaMalloc(&d_data, N * sizeof(float));

// Copy data from host to device
cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

// Copy data from device to host
cudaMemcpy(h_result, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

// Copy between devices (in multi-GPU systems)
cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice);

// Free device memory
cudaFree(d_data);
```

Key `cudaMemcpy` transfer types:

| Enum Value | Direction |
|------------|-----------|
| `cudaMemcpyHostToDevice` | Host -> Device |
| `cudaMemcpyDeviceToHost` | Device -> Host |
| `cudaMemcpyDeviceToDevice` | Device -> Device |
| `cudaMemcpyHostToHost` | Host -> Host |
| `cudaMemcpyDefault` | Auto-detect (requires unified VA) |

Asynchronous copy (overlaps with kernel execution on a stream):

```cpp
cudaMemcpyAsync(d_data, h_pinned, size, cudaMemcpyHostToDevice, stream);
// h_pinned MUST be page-locked host memory for true async behavior
```

### 2.3.3 Page-Locked (Pinned) Host Memory

Page-locked (pinned) host memory is guaranteed not to be swapped out by the OS. This enables:

- Faster `cudaMemcpy` transfers (DMA directly to/from device).
- True asynchronous `cudaMemcpyAsync` (overlaps with kernel execution).
-Mapped device access (zero-copy).

```cpp
// Allocate pinned host memory
float* h_data;
cudaMallocHost(&h_data, N * sizeof(float));  // or cudaHostAlloc

// Use h_data on host...
for (int i = 0; i < N; i++) h_data[i] = float(i);

// Async copy to device (truly async with pinned memory)
cudaMemcpyAsync(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice, stream);

// Free pinned memory
cudaFreeHost(h_data);
```

**Important**: Pinned memory is a scarce resource. Over-allocating pinned memory can reduce overall system performance because the OS has fewer pages available for other processes. Use it judiciously for buffers involved in frequent host-device transfers.

---

## 2.4 Synchronization

CUDA provides synchronization at multiple levels:

### Host-Device Synchronization

```cpp
// Block host until ALL previously submitted GPU work completes
cudaError_t err = cudaDeviceSynchronize();
if (err != cudaSuccess) {
    fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));
}
```

`cudaDeviceSynchronize()` is the broadest synchronization primitive. It blocks the calling host thread until all previously issued commands in all streams on the current device have completed. Use sparingly, as it prevents CPU-GPU overlap.

### Stream-Level Synchronization

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

// Submit work to stream
kernel<<<grid, block, 0, stream>>>(...);
cudaMemcpyAsync(d_out, d_in, size, cudaMemcpyDeviceToDevice, stream);

// Block host until all work in this stream completes
cudaStreamSynchronize(stream);

// Or use an event for finer-grained synchronization
cudaEvent_t event;
cudaEventCreate(&event);
cudaEventRecord(event, stream);
cudaEventSynchronize(event);  // block until event completes

cudaStreamDestroy(stream);
cudaEventDestroy(event);
```

### Block-Level Synchronization

```cpp
__global__ void reductionKernel(float* input, float* output, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Load data into shared memory
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();  // Ensure all threads have loaded their data

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();  // Ensure all threads have updated before next iteration
    }

    // Thread 0 writes the block's result
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

`__syncthreads()` acts as a barrier: all threads in the block must reach the barrier before any thread can proceed. It also ensures that all writes to shared memory and global memory before the barrier are visible to all threads after the barrier.

**Warning**: `__syncthreads()` must be called by all threads in the block. Placing it inside a conditional that only some threads execute results in undefined behavior (deadlock or data corruption).

### Synchronization Summary

| Mechanism | Scope | Callable From |
|-----------|-------|---------------|
| `cudaDeviceSynchronize()` | All streams on device | Host code |
| `cudaStreamSynchronize(stream)` | Single stream | Host code |
| `cudaEventSynchronize(event)` | Until event is recorded | Host code |
| `__syncthreads()` | All threads in a block | Device code |
| `__syncwarp(mask)` | Threads in a warp | Device code |

---

## 2.5 Error Checking

CUDA errors can occur at two stages:

1. **Launch errors**: occur when the kernel is launched (e.g., invalid configuration).
2. **Execution errors**: occur during kernel execution (e.g., out-of-bounds memory access).

### Error Checking Macro

```cpp
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do { \
    cudaError_t result = (expr); \
    if (result != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s:%d: %s (%s)\n", \
                __FILE__, __LINE__, \
                cudaGetErrorString(result), \
                #expr); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
```

### Checking Kernel Errors

Because kernel launches are asynchronous, errors are reported in two phases:

```cpp
// Launch kernel
myKernel<<<grid, block>>>(d_data, N);

// Phase 1: Check for launch errors (invalid config, etc.)
CUDA_CHECK(cudaGetLastError());

// Phase 2: Check for execution errors (requires synchronization)
CUDA_CHECK(cudaDeviceSynchronize());

// Alternative: check last error without synchronizing (catches launch errors only)
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "Launch failed: %s\n", cudaGetErrorString(err));
}
```

### Runtime Error Checking Helper Class (C++ RAII)

```cpp
// RAII wrapper for checking CUDA errors after a scope
class CudaCheck {
public:
    CudaCheck(const char* file, int line) : file_(file), line_(line) {}
    ~CudaCheck() {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error: %s:%d: %s\n",
                    file_, line_, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
    // Prevent copying
    CudaCheck(const CudaCheck&) = delete;
    CudaCheck& operator=(const CudaCheck&) = delete;
private:
    const char* file_;
    int line_;
};

#define CUDA_CHECK_KERNEL() CudaCheck _cuda_check(__FILE__, __LINE__)

// Usage:
// myKernel<<<grid, block>>>(args);
// CUDA_CHECK_KERNEL();
```

### Assertion in Device Code

```cpp
__global__ void safeKernel(float* data, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    assert(idx < N);  // Device-side assertion (CC 3.5+)
    data[idx] = 1.0f / data[idx];
}
```

Device-side assertions cause the kernel to abort and the subsequent host-side synchronization call or `cudaGetLastError()` to return `cudaErrorAssert`.

---

## 2.6 Device and Host Functions

CUDA extends C++ with function type qualifiers that specify where a function is compiled for and callable from:

| Qualifier | Executes On | Callable From | Notes |
|-----------|-------------|---------------|-------|
| `__global__` | Device | Host (or Device with dynamic parallelism) | Kernel entry point. Must return `void`. |
| `__device__` | Device | Device only | Helper function for GPU code. |
| `__host__` | Host | Host only | Default; can be omitted. |
| `__host__ __device__` | Both | Both | Compiled twice; useful for reusable utility functions. |

```cpp
// Device-only function
__device__ float square(float x) {
    return x * x;
}

// Callable from both host and device
__host__ __device__ float clampf(float x, float lo, float hi) {
    return fmaxf(lo, fminf(hi, x));
}

// Kernel calls __device__ functions
__global__ void computeKernel(float* out, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) {
        out[i] = square(clampf(out[i], -1.0f, 1.0f));
    }
}

// Host code can call __host__ __device__ functions
int main() {
    float val = clampf(2.5f, 0.0f, 1.0f);  // OK, calls host version
    // float val2 = square(3.0f);  // ERROR: __device__ function not callable from host
    ...
}
```

Key notes:

- `__global__` functions must have `void` return type.
- `__device__` functions cannot be called directly from host code.
- `__global__` functions cannot call other `__global__` functions (except via CUDA dynamic parallelism, CC 3.5+).
- `__host__` and `__device__` can be combined; the compiler generates two versions of the function.
- Function pointers to `__device__` functions can be used in device code.
- `__device__` functions are always inlined by default (no recursion in earlier architectures; recursion supported from CC 2.0+).

---

## 2.7 Variable Specifiers

CUDA provides memory specifier qualifiers for variables declared at file or function scope:

| Qualifier | Memory | Lifetime | Scope | Notes |
|-----------|--------|----------|-------|-------|
| `__device__` | Global memory | Application | Grid + host | Read/write from all threads and host (via `cudaMemcpyToSymbol`) |
| `__constant__` | Constant memory | Application | Grid + host | 64 KB limit, read-only in kernels, cached |
| `__managed__` | Unified Memory | Application | Grid + host | Managed by UM runtime |
| `__shared__` | Shared memory | Kernel launch | Block | On-chip, very fast, user-managed |

```cpp
// File-scope device variable (in global memory)
__device__ int d_counter = 0;

// File-scope constant variable (64 KB limit)
__constant__ float c_coeffs[4] = {1.0f, 0.5f, 0.25f, 0.125f};

// File-scope managed variable (unified memory)
__managed__ float m_buffer[1024];

// Block-scope shared variable (inside kernel)
__global__ void myKernel(float* out, int N) {
    __shared__ float tile[16][16];  // Static shared memory
    // ...
}

// Dynamic shared memory
__global__ void dynamicSharedKernel(float* out, int N) {
    extern __shared__ float dynamicData[];  // Size specified at launch
    // ...
}
dynamicSharedKernel<<<grid, block, sharedMemBytes>>>(out, N);
```

### Setting device/constant variables from host

```cpp
// Write to __device__ variable from host
int h_counter = 42;
cudaMemcpyToSymbol(d_counter, &h_counter, sizeof(int));

// Read from __device__ variable on host
int result;
cudaMemcpyFromSymbol(&result, d_counter, sizeof(int));

// Same for __constant__ variables
float h_coeffs[4] = {2.0f, 1.0f, 0.5f, 0.25f};
cudaMemcpyToSymbol(c_coeffs, h_coeffs, sizeof(h_coeffs));
```

---

## 2.8 Runtime Initialization

The CUDA runtime initializes **lazily** -- the first CUDA API call creates the **primary context** for the device. This means:

- No explicit initialization call is needed.
- The first call to any runtime API function (e.g., `cudaMalloc`, `cudaMemcpy`, kernel launch) triggers initialization.
- Initialization has non-trivial cost (driver loading, context creation, JIT compilation of embedded PTX). Expect 100ms-1s for the first CUDA call.

### Device Selection

```cpp
// Set device before any other CUDA calls (multi-GPU systems)
cudaSetDevice(0);  // Select GPU 0

// CUDA 12.0+ introduces cudaInitDevice for explicit initialization
cudaInitDevice(0, 0, 0);  // (deviceId, flags, reserved)

// Query device count
int deviceCount;
cudaGetDeviceCount(&deviceCount);

// Query current device
int currentDevice;
cudaGetDevice(&currentDevice);

// Query device properties
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
```

### Context Management

```cpp
// Explicitly destroy the primary context and release all resources
cudaDeviceReset();

// After cudaDeviceReset(), the next CUDA call will create a fresh primary context.
// Useful for:
// - Ensuring clean shutdown
// - Releasing GPU memory between benchmarking runs
// - Resetting device state
```

**Important**: `cudaDeviceReset()` destroys ALL allocations, streams, events, and other runtime state for the current device. Call it only when you are completely done with the device.

---

## 2.9 NVCC Compilation

### Basic Compilation

```bash
# Compile a CUDA source file
nvcc mykernel.cu -o myapp

# Run the application
./myapp
```

### Specifying Target Architecture

The `-arch` flag tells NVCC what GPU architecture to target:

```bash
# Target a specific architecture
nvcc -arch=sm_80 mykernel.cu -o myapp          # A100
nvcc -arch=sm_90 mykernel.cu -o myapp           # H100
nvcc -arch=sm_89 mykernel.cu -o myapp           # RTX 4090

# Generate code for multiple architectures (fat binary)
nvcc -gencode=arch=compute_70,code=sm_70 \
     -gencode=arch=compute_80,code=sm_80 \
     -gencode=arch=compute_90,code=sm_90 \
     -gencode=arch=compute_90,code=compute_90 \
     mykernel.cu -o myapp

# This produces:
#   - Cubin for sm_70 (runs on V100)
#   - Cubin for sm_80 (runs on A100)
#   - Cubin for sm_90 (runs on H100)
#   - PTX for compute_90 (JIT for future GPUs >= 9.0)
```

### Key NVCC Options

| Option | Description |
|--------|-------------|
| `-arch=sm_XY` | Target a specific SM architecture (e.g., `sm_80`) |
| `-gencode=arch=compute_XY,code=sm_XY` | Generate cubin for specific architecture |
| `-gencode=arch=compute_XY,code=compute_XY` | Generate PTX for specific architecture |
| `-O2` / `-O3` | Optimization level |
| `-rdc=true` or `-dc` | Enable relocatable device code (separate compilation) |
| `--std=c++17` | Set C++ standard |
| `-maxrregcount=N` | Limit registers per thread (affects occupancy) |
| `--ptxas-options=-v` | Show register, shared memory, and local memory usage per kernel |
| `-lineinfo` | Generate line information for profiling (no debug overhead) |
| `-G` | Generate device debug info (disables most optimizations) |
| `-Xcompiler -fPIC` | Pass flags to the host compiler |
| `--expt-relaxed-constexpr` | Allow `__host__ __device__` functions in constexpr contexts |
| `--expt-extended-lambda` | Enable device lambdas |
| `-I <path>` | Add include path |
| `-l <lib>` | Link library |
| `-L <path>` | Add library search path |

### Separating Host and Device Compilation

For larger projects, CUDA supports separate compilation of device code:

```bash
# Step 1: Compile device code to object files
nvcc -dc -arch=sm_80 kernel1.cu -o kernel1.o
nvcc -dc -arch=sm_80 kernel2.cu -o kernel2.o

# Step 2: Link device code
nvcc -arch=sm_80 kernel1.o kernel2.o -o myapp

# Or: create a device link library
nvcc -lib kernel1.o kernel2.o -o mykernels.a
nvcc main.cu mykernels.a -o myapp
```

### Integration with Build Systems

```cmake
# CMake example
cmake_minimum_required(VERSION 3.18)
project(my_cuda_app LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 70 80 90)

add_executable(myapp main.cu kernel.cu)
target_compile_options(myapp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
```

```makefile
# Makefile example
NVCC = nvcc
NVCC_FLAGS = -arch=sm_80 -O2 --std=c++17 --ptxas-options=-v

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

myapp: main.o kernel.o
	$(NVCC) $(NVCC_FLAGS) $^ -o $@
```

### Inspecting Compiled Code

```bash
# View PTX output
nvcc -arch=sm_80 mykernel.cu --ptx -o mykernel.ptx

# View cubin (binary disassembly)
cuobjdump -sass myapp

# View resource usage
nvcc -arch=sm_80 --ptxas-options=-v mykernel.cu
# Output example:
#   ptxas info: 0 bytes gmem, 8 bytes cmem[2]
#   ptxas info: Compiling entry function '_Z6myKernelPfi' for 'sm_80'
#   ptxas info: Function properties for _Z6myKernelPfi
#     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
#   ptxas info: Used 8 registers, 1024 bytes smem, 368 bytes cmem[0]
```

---

## 2.10 Thread Block Clusters (Compute Capability 9.0+)

Thread block clusters extend the thread block hierarchy by grouping blocks together with stronger cooperation guarantees.

### Declaring Cluster Dimensions

```cpp
// Method 1: Compile-time cluster dimensions via annotation
__global__ void __cluster_dims__(2, 1, 1)
clusterKernel(float* input, float* output, int N) {
    // This kernel's blocks are grouped into clusters of 2x1x1 blocks
    // ...
}

// Method 2: Runtime launch configuration
#include <cuda_runtime.h>

void launchClusterKernel(float* input, float* output, int N) {
    dim3 gridDim(64, 1, 1);
    dim3 blockDim(128, 1, 1);

    cudaLaunchConfig_t config = {0};
    config.gridDim = gridDim;
    config.blockDim = blockDim;

    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = 2;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;

    config.attrs = &attr;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, clusterKernel, input, output, N);
}
```

### Distributed Shared Memory

The key feature of clusters is **distributed shared memory**: blocks within a cluster can read each other's shared memory.

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(2, 1, 1)
clusterSharedMemKernel(float* input, float* output, int N) {
    // Each block has its own shared memory
    __shared__ float shared_data[128];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Load into local shared memory
    if (idx < N) shared_data[tid] = input[idx];
    cg::this_cluster().sync();  // Cluster-wide synchronization

    // Access another block's shared memory
    auto cluster = cg::this_cluster();
    int neighbor_rank = (cluster.block_rank() + 1) % cluster.num_blocks();

    // Map neighbor's shared memory into our address space
    float* neighbor_shared = cluster.map_shared_rank(shared_data, neighbor_rank);

    // Read from neighbor's shared memory
    if (tid < 128 && idx < N) {
        output[idx] = shared_data[tid] + neighbor_shared[tid];
    }

    cg::this_cluster().sync();
}
```

### Cluster Query Functions

```cpp
// Query maximum cluster size for a kernel
int maxClusterSize;
cudaOccupancyMaxPotentialClusterSize(&maxClusterSize, myKernel, blockDim);
printf("Max cluster size: %d blocks\n", maxClusterSize);

// Query non-portable cluster size (hardware-specific, may be larger)
int maxSize;
cudaOccDeviceMaxClusterSize(&maxSize, prop, blockDim);
```

### Cluster Synchronization

```cpp
__global__ void clusterSyncKernel(int* data, int N) {
    namespace cg = cooperative_groups;
    auto cluster = cg::this_cluster();

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) data[idx] = blockIdx.x;

    // Synchronize all blocks in the cluster
    cluster.sync();

    // After sync, all blocks' writes are visible across the cluster
    ...
}
```

---

## Quick Reference: Common Patterns

### Complete Minimal CUDA Program

```cpp
#include <cuda_runtime.h>
#include <cstdio>

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return 1; \
    } \
} while(0)

__global__ void addOne(float* data, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) data[idx] += 1.0f;
}

int main() {
    const int N = 1024;
    const size_t bytes = N * sizeof(float);

    // Allocate unified memory
    float* data;
    CUDA_CHECK(cudaMallocManaged(&data, bytes));

    // Initialize on host
    for (int i = 0; i < N; i++) data[i] = float(i);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    addOne<<<gridSize, blockSize>>>(data, N);

    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify on host
    printf("data[0] = %f (expected 1.0)\n", data[0]);
    printf("data[999] = %f (expected 1000.0)\n", data[999]);

    CUDA_CHECK(cudaFree(data));
    return 0;
}
```

### Compilation

```bash
nvcc -arch=sm_80 -O2 --ptxas-options=-v minimal.cu -o minimal
./minimal
```
