# CUDA Programming Guide - Chapter 1: Introduction and Programming Model

This reference covers the foundational concepts of CUDA, including the GPU architecture overview, the CUDA programming model, memory hierarchy, and the CUDA platform toolchain.

---

## 1.1 The GPU

### Historical Context

Graphics Processing Units (GPUs) originated as fixed-function hardware accelerators designed exclusively for rendering 2D and 3D graphics. Their highly parallel structure made them effective at processing large batches of geometric primitives and pixel operations simultaneously.

By 2003, GPU vendors began exposing programmable shading units, allowing developers to write custom vertex and fragment programs. Researchers quickly recognized that the same parallel throughput hardware could accelerate general-purpose computation workloads -- a practice known as **GPGPU** (General-Purpose computing on GPUs). However, early GPGPU programming required mapping computational problems onto graphics primitives (textures, triangles, and framebuffer operations), which was cumbersome and error-prone.

### CUDA: A General-Purpose Parallel Computing Platform

In 2006, NVIDIA introduced **CUDA** (Compute Unified Device Architecture), a software and hardware architecture that exposed the GPU's parallel compute engines through a straightforward C/C++ language extension and API. CUDA eliminated the need to cast computations in graphics terms and provided:

- A hierarchy of threads, blocks, and grids that maps naturally to GPU hardware.
- Dedicated device memory management APIs.
- Synchronization primitives and atomic operations.
- Libraries (cuBLAS, cuFFT, cuRAND, Thrust, etc.) for common workloads.

### Throughput vs. Latency

CPUs are optimized for **latency** -- minimizing the time to complete a single task. They dedicate substantial silicon to branch predictors, out-of-order execution engines, deep caches, and large register rename files.

GPUs are optimized for **throughput** -- maximizing the total number of operations completed per unit of time. They dedicate the majority of silicon to arithmetic logic units (ALUs) and rely on massive multithreading to hide memory latency. A modern GPU can deliver:

- **Instruction throughput**: tens of TFLOPS of floating-point performance.
- **Memory bandwidth**: multiple TB/s of off-chip memory bandwidth.

This tradeoff makes GPUs ideal for data-parallel workloads where the same operation is applied across large datasets -- image processing, scientific simulation, deep learning inference and training, signal processing, and many others.

---

## 1.2 Programming Model

### 1.2.1 Heterogeneous Systems

CUDA programs execute across a **heterogeneous system** consisting of one or more CPUs (the **host**) and one or more GPUs (the **device**). The key components are:

| Component | Name | Description |
|-----------|------|-------------|
| CPU + host memory | **Host** | Executes the sequential portions of the application. |
| GPU + device memory | **Device** | Executes the massively parallel portions of the application. |

In discrete GPU configurations, the host and device have physically separate memory subsystems connected via PCIe or NVLink. In **System-on-Chip (SoC)** designs (such as NVIDIA Tegra, Grace Hopper), the CPU and GPU may be integrated into a single silicon package and may share a unified physical memory.

The typical flow of a CUDA application is:

1. The host application allocates memory on the device (or uses Unified Memory).
2. The host copies input data from host memory to device memory.
3. The host launches one or more **kernels** -- functions that execute on the GPU.
4. The host copies results back from device memory to host memory.
5. The host frees device memory allocations.

```cpp
// Typical CUDA application flow
#include <cuda_runtime.h>

int main() {
    int N = 1 << 20;  // 1M elements
    size_t bytes = N * sizeof(float);

    // 1. Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 2. Copy input data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 3. Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vecAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    // 4. Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 5. Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```

### 1.2.2 GPU Hardware Model

Understanding the GPU hardware model is essential for writing efficient CUDA code. The high-level structure is:

**GPU** = collection of **Graphics Processing Clusters (GPCs)**, each containing multiple **Streaming Multiprocessors (SMs)**.

Each **SM** contains:

- A large **register file** (e.g., 256 KB on NVIDIA Ampere) partitioned among active threads.
- A **unified data cache** that serves as both shared memory and L1 data cache (configurable partitioning).
- Multiple **functional units**: FP32 CUDA cores, FP64 DPUs, INT32 ALUs, tensor cores, load/store units, and special function units (SFUs).
- Warp schedulers that issue instructions to the functional units.

The SM is the fundamental compute unit. Thread blocks are scheduled onto SMs, and each SM can concurrently execute multiple thread blocks (subject to resource constraints -- registers, shared memory, and maximum thread count).

```
GPU
+-- GPC 0
|   +-- SM 0
|   |   +-- Register File (256 KB)
|   |   +-- Shared Memory / L1 Cache (up to 164 KB)
|   |   +-- Warp Schedulers (4)
|   |   +-- FP32 Cores (128)
|   |   +-- FP64 DPUs (64)
|   |   +-- Tensor Cores (4)
|   |   +-- Load/Store Units (32)
|   |   +-- SFU (16)
|   +-- SM 1
|   +-- ...
+-- GPC 1
+-- ...
+-- L2 Cache (shared across all SMs)
+-- Memory Controllers
+-- DRAM (Global Memory)
```

### 1.2.2.1 Thread Blocks and Grids

CUDA organizes parallel work into a two-level hierarchy:

- A **thread block** is a group of threads that cooperate via shared memory and synchronization.
- A **grid** is a collection of thread blocks that execute the same kernel.

Both thread blocks and grids can be organized in 1, 2, or 3 dimensions, which allows natural mapping to problem domains (1D arrays, 2D images, 3D volumes).

#### Execution Configuration

When launching a kernel, the programmer specifies the **execution configuration** using triple-chevron notation:

```cpp
kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(...);
```

| Parameter | Description |
|-----------|-------------|
| `gridDim` | Dimensions of the grid (number of blocks in each dimension). Type: `dim3` or `int`. |
| `blockDim` | Dimensions of each thread block (number of threads per block in each dimension). Type: `dim3` or `int`. |
| `sharedMemBytes` | (Optional) Number of bytes of dynamically allocated shared memory per block. Default: 0. |
| `stream` | (Optional) CUDA stream on which to execute. Default: 0 (default stream). |

#### Built-in Variables

Within a kernel, the following built-in variables are available:

| Variable | Type | Description |
|----------|------|-------------|
| `threadIdx` | `uint3` | Thread index within its block (0-based). |
| `blockDim` | `dim3` | Dimensions of the block (number of threads per dimension). |
| `blockIdx` | `uint3` | Block index within the grid (0-based). |
| `gridDim` | `dim3` | Dimensions of the grid (number of blocks per dimension). |
| `warpSize` | `int` | Number of threads per warp (always 32 on current hardware). |

#### Computing Global Thread Index

The most common pattern for computing a unique global thread index:

```cpp
// 1D grid, 1D blocks
int idx = threadIdx.x + blockDim.x * blockIdx.x;

// 2D grid, 2D blocks (row-major layout)
int row = threadIdx.y + blockDim.y * blockIdx.y;
int col = threadIdx.x + blockDim.x * blockIdx.x;
int idx = row * width + col;

// 3D grid, 3D blocks
int idx = threadIdx.x + blockDim.x * blockIdx.x
        + (threadIdx.y + blockDim.y * blockIdx.y) * blockDim.x * gridDim.x
        + (threadIdx.z + blockDim.z * blockIdx.z) * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
```

#### Scheduling and Execution Guarantees

- A thread block executes on a **single SM**; it does not migrate to other SMs during execution.
- Multiple thread blocks can execute concurrently on a single SM, subject to resource constraints (registers, shared memory, maximum thread count per SM).
- There is **no guaranteed ordering** of thread block execution. The programmer must not assume any specific order in which blocks are scheduled.
- Thread blocks within a grid are **independent** -- they cannot safely communicate through global memory without explicit synchronization mechanisms (e.g., cooperative groups or separate kernel launches).

```cpp
// Example: Thread indexing with bounds checking
__global__ void scaleKernel(float* data, int N, float factor) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        data[idx] *= factor;
    }
}

// Launch configuration
int N = 1000000;
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
scaleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N, 2.0f);
```

#### Thread Block Clusters (Compute Capability 9.0+, e.g., Hopper H100)

Thread block clusters are an optional grouping mechanism that provides additional cooperation capabilities beyond thread blocks. Key properties:

- A **cluster** is a group of thread blocks (1, 2, or 3 dimensional) that are guaranteed to execute concurrently on a single **GPC** (Graphics Processing Cluster).
- All thread blocks within a cluster can access the combined shared memory of all blocks in the cluster, referred to as **distributed shared memory**.
- The maximum portable cluster size is **8 thread blocks**. Hardware-specific limits may be larger.
- Query the maximum cluster size with: `cudaOccupancyMaxPotentialClusterSize`.

```cpp
// Query maximum cluster size for a kernel
int clusterSize;
cudaOccupancyMaxPotentialClusterSize(&clusterSize, myKernel, threadsPerBlock);
printf("Max cluster size: %d blocks\n", clusterSize);

// Launch with cluster configuration
cudaLaunchConfig_t config = {0};
config.gridDim = gridDim;
config.blockDim = blockDim;
config.dynamicSmemBytes = 0;

cudaLaunchAttribute clusterAttr;
clusterAttr.id = cudaLaunchAttributeClusterDimension;
clusterAttr.val.clusterDim.x = 2;
clusterAttr.val.clusterDim.y = 1;
clusterAttr.val.clusterDim.z = 1;
config.attrs = &clusterAttr;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, myKernel, ...);
```

### 1.2.2.2 Warps and SIMT Architecture

#### Warp Organization

Threads within a thread block are organized into groups of 32 called **warps**. The warp is the fundamental scheduling unit on the GPU. Threads within a warp are assigned consecutive `threadIdx` values (first along the x dimension, then y, then z).

For a thread block with dimensions `(Dx, Dy, Dz)`:
- Total threads = `Dx * Dy * Dz`
- Number of warps = `ceil(total_threads / 32)`
- Warp 0 contains threads 0-31, warp 1 contains threads 32-63, and so on.

#### SIMT Execution Model

CUDA GPUs follow the **Single Instruction, Multiple Thread (SIMT)** model:

- All 32 threads in a warp execute the **same instruction** simultaneously on different data.
- Each thread has its own register state and can follow its own control flow, but execution is serialized when threads diverge.
- The warp scheduler selects a ready warp and issues its next instruction to the functional units.

#### Warp Divergence

When threads within a warp follow different execution paths (due to conditional branches), this is called **warp divergence**:

- The warp executes each branch path serially, disabling threads that do not take that path.
- **Masked-off threads** are inactive and simply wait; they do not contribute useful work during that cycle.
- After all paths complete, threads reconverge and continue in lockstep.

Warp divergence is one of the most common performance pitfalls. To maximize throughput:

- Structure code so that threads within a warp follow the **same control flow** as much as possible.
- Ensure thread block sizes are multiples of 32 so that no warp is partially populated.
- When branching is unavoidable, prefer to branch on warp-aligned boundaries (e.g., `if (warpId == ...)` rather than `if (threadIdx.x == ...)`).

```cpp
// BAD: Divergent warp -- threads in same warp take different paths
__global__ void divergentKernel(int* data, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx % 2 == 0) {
        data[idx] = data[idx] * 2;   // half of warp does this
    } else {
        data[idx] = data[idx] + 1;   // other half does this
    }
    // Both paths are executed serially for each divergent warp
}

// BETTER: Warp-aligned branch -- full warps take same path
__global__ void warpAlignedKernel(int* data, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if ((idx / 32) % 2 == 0) {
        data[idx] = data[idx] * 2;   // entire warp does this
    } else {
        data[idx] = data[idx] + 1;   // entire warp does this
    }
}
```

#### Thread Block Sizing Guidelines

- Maximum threads per block: **1024**.
- Maximum dimensions per block: `(1024, 1024, 64)`, with the constraint that `Dx * Dy * Dz <= 1024`.
- For best occupancy and to avoid wasted warp slots, total threads per block should be a **multiple of 32**.
- Common block sizes: 128, 256, 512, 1024. The optimal choice depends on register and shared memory usage.

---

## 1.2.3 GPU Memory

CUDA provides a rich memory hierarchy with different scopes, latencies, and access patterns. Understanding this hierarchy is critical for performance.

### DRAM Memory

#### Global Memory

GPU DRAM is referred to as **global memory**. It is the largest memory space on the device (typically 8 GB to 80+ GB depending on the GPU) but has the highest latency (hundreds of clock cycles). Global memory is:

- Accessible by all threads in a grid, plus the host via CUDA API calls.
- Persistent for the lifetime of the application (until explicitly freed).
- Cached by the L2 cache and (optionally) the L1 cache.
- The primary location for data that must be shared across thread blocks.

CPU DRAM is referred to as **system memory** or **host memory**.

#### Unified Virtual Address Space

On 64-bit systems, CUDA provides a **unified virtual address space** where CPU and GPU memory share a single virtual address range. This means:

- Every allocation (host or device) has a unique virtual address.
- `cudaMemcpyDefault` can automatically determine the direction of a copy based on the address.
- `cudaPointerGetAttributes()` can determine whether a pointer refers to host or device memory.

### On-Chip Memory

On-chip memories are physically located on the GPU die, providing much lower latency and higher bandwidth than global memory.

#### Register File

- The fastest memory available to a CUDA thread.
- **Thread-local**: each thread has its own private set of registers.
- Allocated by the compiler from the SM's register file (e.g., 256 KB per SM, up to 255 registers per thread).
- Register pressure limits occupancy: if a kernel uses many registers per thread, fewer threads (and fewer blocks) can be active on an SM simultaneously.
- Use `--ptxas-options=-v` to see register usage per kernel.
- Use `__launch_bounds__()` to hint the compiler about desired occupancy.

#### Shared Memory

- **On-chip memory** shared among all threads in a thread block (or cluster).
- Much faster than global memory (comparable to L1 cache latency).
- User-managed: programmers explicitly load data into shared memory, process it, and write results back.
- Declared with the `__shared__` qualifier.
- Can be statically sized or dynamically allocated at kernel launch.
- Physically backed by the same memory as the L1 cache; the partition between shared memory and L1 is configurable via `cudaFuncSetAttribute()`.

#### L1 and L2 Caches

- **L1 cache**: per-SM, caches global memory accesses. Can be configured for preferential caching of global or local data.
- **L2 cache**: shared across all SMs, provides a larger secondary cache for global memory accesses.
- **Constant cache**: optimized for broadcast reads of `__constant__` data to all threads in a warp.

### Unified Memory

Unified Memory provides a **single address space** accessible from both CPU and GPU code. It is managed by the CUDA runtime and the GPU hardware/driver:

- Allocations made with `cudaMallocManaged()` or the `__managed__` specifier are accessible from both host and device code.
- The runtime and hardware automatically manage **data migration** and **access** -- pages are moved between host and device memory on demand.
- On systems with hardware support (HMM -- Heterogeneous Memory Management, or ATS -- Address Translation Services), Unified Memory can extend to any system memory allocation.

```cpp
// Unified Memory example
#include <cuda_runtime.h>
#include <cstdio>

__global__ void multiplyByTwo(float* data, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) data[idx] *= 2.0f;
}

int main() {
    const int N = 1024;
    float* data;

    // Allocate unified memory -- accessible from both CPU and GPU
    cudaMallocManaged(&data, N * sizeof(float));

    // Initialize on the CPU
    for (int i = 0; i < N; i++) data[i] = float(i);

    // Launch kernel on GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    multiplyByTwo<<<numBlocks, blockSize>>>(data, N);
    cudaDeviceSynchronize();

    // Read back on CPU -- no explicit cudaMemcpy needed
    printf("data[0]=%f, data[1]=%f\n", data[0], data[1]);

    cudaFree(data);
    return 0;
}
```

---

## 1.3 The CUDA Platform

### Compute Capability

Every NVIDIA GPU has a **compute capability** that indicates its hardware features and specification limits. The format is **X.Y**, where:

- **X** indicates the major architecture generation.
- **Y** indicates the incremental improvement within that generation.

The compute capability directly corresponds to the **SM version**. For example:

| Compute Capability | Architecture | SM Version | Example GPU |
|---|---|---|---|
| 7.0 | Volta | sm_70 | V100 |
| 7.5 | Turing | sm_75 | RTX 2080, T4 |
| 8.0 | Ampere | sm_80 | A100 |
| 8.6 | Ampere | sm_86 | RTX 3080 |
| 8.9 | Ada Lovelace | sm_89 | RTX 4090, L4 |
| 9.0 | Hopper | sm_90 | H100 |
| 10.0 | Blackwell | sm_100 | B100 |
| 12.0 | Next-gen | sm_120 | (future) |

Compute capability determines:

- Maximum threads per block, block dimensions, grid dimensions.
- Amount of shared memory, registers, and L1 cache per SM.
- Supported instructions and features (e.g., thread block clusters require CC 9.0+).
- Warp-level primitives available.

Query at runtime:

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("Compute capability: %d.%d\n", prop.major, prop.minor);
printf("SMs: %d\n", prop.multiProcessorCount);
printf("Global memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
printf("Max shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
```

### CUDA Toolkit and Driver

The CUDA platform consists of several software layers:

#### NVIDIA Driver

- The GPU **driver** manages the hardware, provides kernel-mode interfaces, and implements the CUDA Driver API.
- Versioned independently (e.g., r580, r590). Each driver version supports a range of CUDA toolkit versions.
- The driver handles PTX JIT compilation, memory management, and context management.

#### CUDA Toolkit

- The **CUDA Toolkit** is the SDK distributed by NVIDIA. It includes:
  - **CUDA runtime** (libcudart): the high-level C++ API.
  - **CUDA driver API** (libcuda): the low-level API.
  - **NVCC** compiler: compiles CUDA C++ to device code.
  - **Libraries**: cuBLAS, cuFFT, cuRAND, cuDNN, Thrust, CUB, etc.
  - **Tools**: nvprof, Nsight Systems, Nsight Compute, cuda-gdb, cuda-memcheck.
  - **Headers and source**: CUDA runtime API headers, device intrinsic headers.

#### CUDA Runtime

- The CUDA runtime API is the primary interface for most CUDA applications.
- Provides implicit initialization (creates primary context on first API call).
- Manages device memory, streams, events, and kernel launches.
- Implemented as a thin layer over the CUDA driver API.

### PTX and Cubins

CUDA code goes through several compilation stages:

#### PTX (Parallel Thread Execution)

- A **high-level virtual assembly language** (virtual ISA) for CUDA.
- PTX is **forward-compatible**: PTX generated for an older architecture can be JIT-compiled by newer drivers for newer hardware.
- PTX uses a register-based instruction set with virtual registers.
- Provides a stable compilation target across GPU generations.

#### Cubins (CUDA Binaries)

- A **cubin** is a CUDA binary containing native machine code for a specific SM version.
- Cubins are **not forward-compatible**: a cubin for sm_70 will not run on sm_80 hardware.
- Cubins provide the best runtime performance because no JIT compilation is needed.

#### Fatbinaries

- A **fatbin** (fat binary) is a container format that embeds multiple cubins and/or PTX for different architectures.
- At runtime, the driver selects the most appropriate cubin for the current GPU, falling back to JIT-compiling embedded PTX if needed.
- NVCC produces fatbins by default when multiple `-gencode` options are specified.

#### JIT Compilation

- When a kernel is launched and only PTX is available for the current GPU, the driver performs **Just-In-Time (JIT)** compilation to produce a cubin.
- JIT compilation adds latency at first kernel launch but enables forward compatibility.
- The driver caches JIT-compiled cubins in a local cache to avoid recompilation across runs.

```
CUDA C++ Source (.cu)
       |
       v
    NVCC Compiler
       |
       +---> PTX (virtual ISA, forward compatible)
       |
       +---> Cubin (native binary, specific SM)
       |
       v
    Fat Binary (embeds PTX + Cubins for multiple architectures)
       |
       v
    Embedded in Host Binary
       |
       v
    Runtime: Select best cubin, or JIT-compile PTX
```

```cpp
// Example: Full compilation pipeline with multiple architectures
// Compile command:
//   nvcc -gencode=arch=compute_70,code=sm_70 \
//        -gencode=arch=compute_80,code=sm_80 \
//        -gencode=arch=compute_90,code=sm_90 \
//        -gencode=arch=compute_90,code=compute_90 \
//        mykernel.cu -o myapp
//
// This produces a fatbin with:
//   - Cubin for sm_70 (V100)
//   - Cubin for sm_80 (A100)
//   - Cubin for sm_90 (H100)
//   - PTX for compute_90 (enables JIT for future GPUs >= 9.0)
```

---

## Summary of Key Concepts

| Concept | Description |
|---------|-------------|
| Host | CPU + host memory |
| Device | GPU + device memory |
| Kernel | Function executed on the GPU by many threads |
| Thread block | Group of cooperating threads on one SM |
| Grid | Collection of thread blocks executing a kernel |
| Warp | Group of 32 threads executing in SIMT lockstep |
| SM | Streaming Multiprocessor -- the GPU's compute unit |
| Global memory | Large, high-latency off-chip DRAM |
| Shared memory | Fast, on-chip memory shared within a block |
| Registers | Fastest, thread-local on-chip storage |
| Unified Memory | Single address space accessible from host and device |
| Compute capability | Version number indicating GPU feature support |
| PTX | Virtual assembly language (forward compatible) |
| Cubin | Native binary for a specific SM architecture |
