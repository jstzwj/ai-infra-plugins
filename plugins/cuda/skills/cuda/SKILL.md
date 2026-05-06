---
name: cuda
description: >
  Comprehensive reference documentation and skill for NVIDIA CUDA C++ - the parallel computing platform
  and programming model for GPU acceleration. Covers CUDA Programming Guide (Release 13.2) and
  CUDA C++ Best Practices Guide (Release 13.2). Includes programming model, memory management,
  asynchronous execution, CUDA graphs, cooperative groups, advanced synchronization, async data copies,
  TMA unit, L2 cache control, green contexts, virtual memory, IPC, multi-GPU, driver API, math functions,
  device-callable APIs, compute capabilities, C++ language support, deployment, and performance optimization.
version: 13.2
---

# CUDA C++ - Parallel Computing Platform & Programming Model

## Overview

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model. It enables dramatic increases in computing performance by harnessing the power of GPUs for general-purpose processing (GPGPU). CUDA C++ extends C++ with GPU-specific keywords, built-in variables, and runtime APIs.

**Supported Hardware:** NVIDIA GPUs with Compute Capability 5.0+ (full feature set requires CC 7.0+)

**Supported Platforms:** Linux, Windows, WSL

**CUDA Toolkit Version:** 13.2 (March 2026)

## Key Architecture Concepts

- **Host**: CPU + system memory
- **Device**: GPU + device memory
- **Kernel**: Function launched for parallel GPU execution
- **Thread Block**: Group of threads executing on a single SM, can cooperate via shared memory
- **Grid**: Collection of thread blocks executing a kernel
- **Warp**: Group of 32 threads executing in SIMT fashion
- **SM (Streaming Multiprocessor)**: GPU compute unit with registers, shared memory, and functional units
- **Thread Block Cluster** (CC 9.0+): Group of thread blocks on a single GPC with distributed shared memory

## Quick Reference

### Minimal Kernel Example
```cpp
__global__ void vecAdd(float* A, float* B, float* C, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}

// Launch
int threads = 256;
int blocks = (N + threads - 1) / threads;
vecAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
cudaDeviceSynchronize();
```

### Memory Allocation
```cpp
// Unified Memory
float* data;
cudaMallocManaged(&data, N * sizeof(float));
cudaFree(data);

// Explicit Device Memory
cudaMalloc(&d_data, N * sizeof(float));
cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
cudaFree(d_data);

// Stream-Ordered (CUDA 11.0+)
cudaMallocAsync(&d_data, N * sizeof(float), stream);
cudaFreeAsync(d_data, stream);
```

### GPU Timing
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, 0);
kernel<<<grid, block>>>(...);
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
```

### Error Checking Macro
```cpp
#define CUDA_CHECK(expr) do { \
    cudaError_t err = expr; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
```

### Compilation
```bash
nvcc -arch=sm_80 -O2 mykernel.cu -o myapp
nvcc -gencode=arch=compute_70,code=sm_70 \
     -gencode=arch=compute_80,code=sm_80 \
     -gencode=arch=compute_90,code=sm_90 \
     mykernel.cu -o myapp
```

## Memory Hierarchy at a Glance

| Memory | Location | Cached | Access | Scope | Lifetime |
|--------|----------|--------|--------|-------|----------|
| Register | On-chip (SM) | N/A | R/W | 1 thread | Kernel |
| Shared | On-chip (SM) | N/A | R/W | Thread block | Kernel |
| L1 Cache | On-chip (SM) | Yes | R/W | Thread block | Kernel |
| L2 Cache | On-chip (GPU) | Yes | R/W | All threads | Kernel |
| Local | Off-chip (DRAM) | L1+L2 | R/W | 1 thread | Kernel |
| Global | Off-chip (DRAM) | L1+L2 | R/W | All threads + host | Application |
| Constant | Off-chip (DRAM) | Yes | R | All threads + host | Application |
| Texture | Off-chip (DRAM) | Yes | R | All threads + host | Application |

## High-Priority Best Practices

1. **Profile first** - Use Nsight Compute/Systems to identify bottlenecks
2. **Coalesce global memory accesses** - Consecutive threads access consecutive addresses
3. **Minimize host-device transfers** - Keep data on device as long as possible
4. **Use shared memory** - Avoid redundant global memory loads
5. **Avoid warp divergence** - Keep threads in a warp on the same control flow path
6. **Use effective bandwidth** as a performance metric
7. **Overlap computation with transfers** using streams and async operations

## Documentation Structure

### Core Programming Model
- [01-introduction-and-programming-model](references/01-introduction-and-programming-model.md) - CUDA overview, GPU hardware model, threads/blocks/grids/warps, SIMT, compute capability
- [02-getting-started](references/02-getting-started.md) - CUDA C++ basics, kernels, NVCC compilation, memory, error handling, device/host functions
- [03-memory-management](references/03-memory-management.md) - Unified memory, explicit memory, page-locked memory, memory spaces, managed memory paradigms

### Asynchronous Execution
- [04-asynchronous-execution](references/04-asynchronous-execution.md) - CUDA streams, events, callbacks, stream ordering, default stream, priorities
- [05-cuda-graphs](references/05-cuda-graphs.md) - Graph structure, building (API + stream capture), instantiation, execution, updating, conditional nodes, memory nodes, device graph launch

### Memory & Synchronization
- [06-stream-ordered-memory-allocator](references/06-stream-ordered-memory-allocator.md) - cudaMallocAsync, cudaFreeAsync, memory pools, IPC pools
- [07-cooperative-groups](references/07-cooperative-groups.md) - CG handles, thread_block, cluster_group, grid_group, tiled_partition, collective operations
- [08-advanced-synchronization](references/08-advanced-synchronization.md) - Async barriers, pipelines, programmatic dependent launch, memory sync domains

### Data Movement
- [09-async-data-copies](references/09-async-data-copies.md) - LDGSTS, TMA (1D + multi-dim), STAS, tensor maps, bank swizzling, work stealing
- [10-l2-cache-control](references/10-l2-cache-control.md) - L2 persistence, access policy windows, hit ratio tuning

### Execution Contexts
- [11-green-contexts](references/11-green-contexts.md) - SM partitioning, resource descriptors, work queues, execution context streams
- [12-virtual-memory-and-ipc](references/12-virtual-memory-and-ipc.md) - VM management, unicast/multicast sharing, IPC, compressible memory
- [13-multi-gpu-programming](references/13-multi-gpu-programming.md) - Device enumeration, P2P transfers, peer access, consistency

### APIs & Language
- [14-cuda-driver-api](references/14-cuda-driver-api.md) - Context, module, kernel execution, runtime/driver interop
- [15-compute-capabilities](references/15-compute-capabilities.md) - CC specs, features, device/SM/memory info per CC, tensor core types
- [16-cpp-language-support](references/16-cpp-language-support.md) - C++11/14/17/20 features, restrictions, libcu++, lambda expressions
- [17-language-extensions](references/17-language-extensions.md) - Execution/memory space specifiers, built-in types, atomics, warp functions, vector types, DPX
- [18-math-functions](references/18-math-functions.md) - Standard math, intrinsic functions, non-standard functions, half precision math
- [19-device-callable-apis](references/19-device-callable-apis.md) - Memory barrier primitives, pipeline primitives, CG API reference, device runtime (CDP)

### Memory Model & Execution
- [20-cuda-memory-model](references/20-cuda-memory-model.md) - Thread scopes, atomicity, data races, forward progress, execution model
- [21-environment-variables](references/21-environment-variables.md) - All CUDA env vars: device enumeration, JIT, execution, module loading, error log

### Interoperability
- [22-interop-and-external-resources](references/22-interop-and-external-resources.md) - Vulkan, Direct3D, OpenGL, NVSCI interop, external semaphores
- [23-driver-entry-point-access](references/23-driver-entry-point-access.md) - cuGetProcAddress, versioning, typedefs, per-thread default stream

### Deployment
- [24-deployment-and-compatibility](references/24-deployment-and-compatibility.md) - Building for compatibility, runtime redistribution, nvidia-smi, NVML, lazy loading

### Best Practices Guide
- [25-best-practices-overview](references/25-best-practices-overview.md) - APOD cycle, profiling, assessing applications, verification, debugging
- [26-memory-optimizations](references/26-memory-optimizations.md) - Host-device transfers, coalescing, shared memory optimization, L2 cache, allocation
- [27-execution-configuration](references/27-execution-configuration.md) - Occupancy calculation, thread/block heuristics, concurrent kernels, multiple contexts
- [28-instruction-optimization](references/28-instruction-optimization.md) - Arithmetic throughput table, control flow, math libraries, compiler flags, nvcc switches

### Extended Features
- [29-extended-gpu-memory](references/29-extended-gpu-memory.md) - EGM, NVLink-C2C, NUMA considerations
- [30-lazy-loading-and-error-log](references/30-lazy-loading-and-error-log.md) - Module lazy loading, CUDA error log management
