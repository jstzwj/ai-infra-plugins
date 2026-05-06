# 15. Compute Capabilities

This document covers CUDA Compute Capabilities, which define the hardware feature set and performance specifications of NVIDIA GPU architectures. Each GPU has a compute capability version (e.g., 7.5, 8.0, 9.0) that determines available instructions, memory limits, and architectural features.

---

## Table of Contents

1. [Obtaining Compute Capability](#151-obtaining-compute-capability)
2. [Feature Availability](#152-feature-availability)
3. [Device and SM Specifications](#153-device-and-sm-specifications)
4. [Memory Specifications](#154-memory-specifications)
5. [Shared Memory Capacity Options](#155-shared-memory-capacity-options)
6. [Tensor Core Input Types](#156-tensor-core-input-types)
7. [Feature Support Matrix](#157-feature-support-matrix)

---

## 15.1 Obtaining Compute Capability

Compute capability can be queried using several methods:

### Command Line

```bash
# Show compute capability for all GPUs
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Example output:
# name, compute_cap
# NVIDIA A100-SXM4-80GB, 8.0
# NVIDIA H100-SXM5, 9.0
```

### Runtime API

```cpp
// Method 1: Full device properties
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device);
int cc = prop.major * 10 + prop.minor;
printf("Device %d: %s, CC %d.%d (compute capability %d)\n",
       device, prop.name, prop.major, prop.minor, cc);

// Method 2: Individual attributes
int major, minor;
cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
printf("Compute capability: %d.%d\n", major, minor);
```

### Querying Multiple Devices

```cpp
int deviceCount;
cudaGetDeviceCount(&deviceCount);

for (int dev = 0; dev < deviceCount; dev++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    printf("Device %d: \"%s\"\n", dev, prop.name);
    printf("  Compute capability:        %d.%d\n", prop.major, prop.minor);
    printf("  Total global memory:       %.0f MB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("  Multiprocessors (SMs):     %d\n", prop.multiProcessorCount);
    printf("  Max threads per SM:        %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Max threads per block:     %d\n", prop.maxThreadsPerBlock);
    printf("  Warp size:                 %d\n", prop.warpSize);
    printf("  Max shared memory per SM:  %lu bytes\n", prop.sharedMemPerMultiprocessor);
    printf("  Max shared memory per blk: %lu bytes\n", prop.sharedMemPerBlock);
    printf("  Registers per SM:          %d\n", prop.regsPerMultiprocessor);
    printf("  Registers per block:       %d\n", prop.regsPerBlock);
    printf("  Memory clock rate:         %d kHz\n", prop.memoryClockRate);
    printf("  Memory bus width:          %d bits\n", prop.memoryBusWidth);
    printf("  L2 cache size:             %d bytes\n", prop.l2CacheSize);
    printf("  Concurrent kernels:        %s\n",
           prop.concurrentKernels ? "Yes" : "No");
}
```

### Conditional Compilation

```cpp
// Architecture-specific code using preprocessor macros
#if defined(__CUDA_ARCH__)
    #if __CUDA_ARCH__ >= 900
        // Hopper (CC 9.0) and later specific code
        asm volatile("bar.sync");
    #elif __CUDA_ARCH__ >= 800
        // Ampere (CC 8.x) specific code
    #elif __CUDA_ARCH__ >= 750
        // Turing (CC 7.5) specific code
    #else
        // Fallback for older architectures
    #endif
#endif
```

### nvcc Compilation Flags

```bash
# Generate code for specific compute capability
nvcc -arch=sm_80         # CC 8.0 (A100)
nvcc -arch=sm_90         # CC 9.0 (H100)

# Generate PTX and SASS for multiple architectures
nvcc -gencode arch=compute_80,code=sm_80 \
      -gencode arch=compute_90,code=sm_90

# Fat binary with both PTX and SASS
nvcc -gencode arch=compute_90,code=compute_90 \
      -gencode arch=compute_90,code=sm_90
```

---

## 15.2 Feature Availability

NVIDIA provides three levels of feature availability classification when specifying compute capability targets:

### Architecture-Specific: `compute_XXa` (Full Feature Set)

The `a` suffix targets the **exact compute capability** with its full feature set. Code compiled for `compute_90a` will use all features available on CC 9.0, including optional features. This binary will **not** run on any other compute capability.

```bash
# Compile for exact CC 9.0 with all features
nvcc -arch=sm_90a my_kernel.cu
```

Use this when:
- You know the exact target hardware
- You need optional architectural features (e.g., sparse tensor cores)
- Maximum performance is the priority

### Family-Specific: `compute_XXf` (Common Subset)

The `f` suffix targets the **common feature subset** across all devices in the family. Code compiled for `compute_90f` uses only features present on all CC 9.x devices. This binary will run on any CC 9.x device.

```bash
# Compile for the CC 9.x family common subset
nvcc -arch=sm_90f my_kernel.cu
```

Use this when:
- You need portability within a compute capability family
- Optional features are not required
- Deploying across multiple GPU variants in the same generation

### Baseline: `compute_XX` (Compatible with All Later Devices)

The baseline `compute_XX` (no suffix) targets the minimum feature set for that CC and is **forward-compatible** with all later compute capabilities. Code compiled for `compute_80` will run on CC 8.0, 8.6, 8.9, 9.0, 10.0, and later.

```bash
# Compile for baseline CC 8.0 (forward compatible)
nvcc -arch=compute_80 -code=sm_80 my_kernel.cu
```

Use this when:
- You need maximum forward compatibility
- Deploying across multiple GPU generations
- The fat binary approach is not feasible

### Feature Availability Summary

| Suffix | Scope | Example | Runs On |
|--------|-------|---------|---------|
| `a` | Exact CC full features | `compute_90a` | CC 9.0 only |
| `f` | Family common subset | `compute_90f` | All CC 9.x |
| (none) | Baseline + forward | `compute_90` | CC 9.0 and later |

---

## 15.3 Device and SM Specifications

The following table lists key device and SM (Streaming Multiprocessor) specifications across compute capabilities:

| Specification | 7.5 | 8.0 | 8.6 | 8.9 | 9.0 | 10.0 | 11.0 | 12.x |
|---|---|---|---|---|---|---|---|---|
| **Architecture** | Turing | Ampere | Ampere | Ada Lovelace | Hopper | Blackwell | Blackwell | Next-gen |
| **FP32:FP64 Ratio** | 32:1 | 2:1 | 64:1 | 64:1 | 2:1 | 2:1 | 2:1 | 2:1 |
| **Max grid dimensionality** | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| **Max grid x-dim** | 2^31-1 | 2^31-1 | 2^31-1 | 2^31-1 | 2^31-1 | 2^31-1 | 2^31-1 | 2^31-1 |
| **Max grid y-dim** | 65535 | 65535 | 65535 | 65535 | 65535 | 65535 | 65535 | 65535 |
| **Max grid z-dim** | 65535 | 65535 | 65535 | 65535 | 65535 | 65535 | 65535 | 65535 |
| **Max threads/block** | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 |
| **Max thread dimensions (x,y,z)** | (1024,1024,64) | (1024,1024,64) | (1024,1024,64) | (1024,1024,64) | (1024,1024,64) | (1024,1024,64) | (1024,1024,64) | (1024,1024,64) |
| **Warp size** | 32 | 32 | 32 | 32 | 32 | 32 | 32 | 32 |
| **Max resident blocks/SM** | 16 | 32 | 16 | 24 | 32 | 32 | 32 | 24 |
| **Max resident threads/SM** | 1024 | 2048 | 1536 | 1536 | 2048 | 2048 | 2048 | 2048 |
| **Max resident warps/SM** | 32 | 64 | 48 | 48 | 64 | 64 | 64 | 64 |
| **FP32 CUDA cores/SM** | 64 | 64 | 128 | 128 | 128 | 128 | 128 | 128 |
| **FP64 CUDA cores/SM** | 2 | 32 | 2 | 2 | 64 | 64 | 64 | 64 |
| **INT8 TOPS/SM (per clock)** | 128 | 128 | 256 | 256 | 256 | 256 | 256 | 256 |
| **Tensor cores/SM** | 8 | 4 (3rd gen) | 4 (3rd gen) | 4 (4th gen) | 4 (4th gen) | 4 (5th gen) | 4 (5th gen) | 4 (5th gen) |
| **Max CP clusters/SM** | N/A | N/A | N/A | N/A | 8 | 8 | 8 | 8 |
| **Max threads/cluster** | N/A | N/A | N/A | N/A | 2048 | 2048 | 2048 | 2048 |

### Key Architectural Changes

**Turing (CC 7.5):**
- First consumer tensor cores
- Unified data cache architecture for shared memory and L1
- Independent thread scheduling within warps

**Ampere (CC 8.0/8.6):**
- 3rd generation tensor cores with TF32 and BF16 support
- CC 8.0 (A100): 2:1 FP32:FP64 ratio, 32 resident blocks/SM
- CC 8.6 (GeForce 30): 128 FP32 cores/SM (double throughput), 64:1 FP32:FP64

**Ada Lovelace (CC 8.9):**
- 4th generation tensor cores
- DPX instructions for dynamic programming acceleration
- 24 resident blocks/SM

**Hopper (CC 9.0):**
- Thread block clusters for cross-SM cooperation
- Distributed shared memory
- Tensor memory accelerator (TMA)
- Asynchronous transactional barrier
- 4th generation tensor cores with FP8 support
- 128KB L1 data cache per SM

**Blackwell (CC 10.0/11.0):**
- 5th generation tensor cores with FP4 and FP6 support
- Second-generation thread block clusters
- CC 11.0: B200 variant with enhanced memory subsystem

---

## 15.4 Memory Specifications

| Specification | 7.5 | 8.0 | 8.6 | 8.9 | 9.0 | 10.0 | 11.0 | 12.x |
|---|---|---|---|---|---|---|---|---|
| **32-bit registers/SM** | 65536 | 65536 | 65536 | 65536 | 65536 | 65536 | 65536 | 65536 |
| **Max registers/thread** | 255 | 255 | 255 | 255 | 255 | 255 | 255 | 255 |
| **Max shared memory/SM** | 64 KB | 164 KB | 100 KB | 100 KB | 228 KB | 228 KB | 228 KB | 100 KB |
| **Max shared memory/block** | 48 KB | 163 KB | 99 KB | 99 KB | 227 KB | 227 KB | 227 KB | 99 KB |
| **Shared memory banks** | 32 | 32 | 32 | 32 | 32 | 32 | 32 | 32 |
| **Constant memory** | 64 KB | 64 KB | 64 KB | 64 KB | 64 KB | 64 KB | 64 KB | 64 KB |
| **L1 cache size/SM** | 96 KB | 192 KB | 128 KB | 128 KB | 256 KB | 256 KB | 256 KB | 128 KB |
| **L2 cache size** | Varies | 40 MB (A100) | Varies | Varies | 50 MB (H100) | Varies | Varies | Varies |
| **Global memory bandwidth** | Varies | 2.0 TB/s (A100) | Varies | Varies | 3.35 TB/s (H100) | 8.0 TB/s (B200) | 8.0 TB/s | Varies |
| **Memory address size** | 64-bit | 64-bit | 64-bit | 64-bit | 64-bit | 64-bit | 64-bit | 64-bit |
| **Max pitch (2D memcpy)** | 2 GB | 2 GB | 2 GB | 2 GB | 2 GB | 2 GB | 2 GB | 2 GB |
| **Texture alignment** | 512 B | 512 B | 512 B | 512 B | 512 B | 512 B | 512 B | 512 B |
| **Uniform memory** | No | No | No | No | Yes | Yes | Yes | Yes |

### Register Allocation

```cpp
// Control register allocation with launch bounds
__global__ void __launch_bounds__(256, 8)   // 256 threads, min 8 blocks/SM
myKernel(...) {
    // Compiler limits registers per thread to allow 8 blocks/SM
    // MaxRegsPerThread = 65536 / (256 * 8) = 32
}

// Explicit register cap via maxnreg (CC 9.0+)
__global__ void __maxnreg__(64)              // max 64 registers per thread
myKernel2(...) { }
```

### Shared Memory Configuration

```cpp
// Configure shared memory vs L1 cache split
cudaFuncSetAttribute(
    myKernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared    // maximize shared memory
);

// Or set a specific percentage
cudaFuncSetAttribute(
    myKernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    75    // 75% shared memory, 25% L1 cache
);
```

---

## 15.5 Shared Memory Capacity Options

Shared memory capacity is carved from the unified data cache (L1 + shared memory). Not all sizes are valid; only specific sizes are supported based on compute capability:

| CC | Unified Data Cache (KB) | SMEM Capacity Sizes (KB) |
|---|---|---|
| 7.5 | 96 | 32, 64 |
| 8.0 | 192 | 0, 8, 16, 32, 64, 100, 132, 164 |
| 8.6 | 128 | 0, 8, 16, 32, 64, 100 |
| 8.9 | 128 | 0, 8, 16, 32, 64, 100 |
| 9.0 | 256 | 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 |
| 10.0 | 256 | 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 |
| 10.1 | 256 | 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 |
| 11.0 | 256 | 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 |
| 12.x | 128 | 0, 8, 16, 32, 64, 100 |

### Setting Shared Memory Carveout

```cpp
// Method 1: Runtime API
cudaFuncSetAttribute(
    myKernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared    // request maximum shared memory
);

// Method 2: Specify exact carveout in bytes
cudaFuncSetAttribute(
    myKernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    100 * 1024    // request 100 KB shared memory
);

// Method 3: Use max dynamic shared memory
cudaFuncSetAttribute(
    myKernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    100 * 1024    // allow up to 100 KB dynamic shared memory
);

// Launch with dynamic shared memory
myKernel<<<grid, block, 100 * 1024, stream>>>(...);  // 100 KB dynamic SMEM
```

### Querying Shared Memory Configuration

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

printf("Shared memory per SM:          %lu bytes\n", prop.sharedMemPerMultiprocessor);
printf("Shared memory per block:       %lu bytes\n", prop.sharedMemPerBlockOptin);
printf("Shared memory per block (max): %lu bytes\n", prop.sharedMemPerBlock);

// Query current shared memory config
cudaSharedMemConfig config;
cudaDeviceGetSharedMemConfig(&config);
if (config == cudaSharedMemBankSizeFourByte) {
    printf("Shared memory bank size: 4 bytes\n");
} else if (config == cudaSharedMemBankSizeEightByte) {
    printf("Shared memory bank size: 8 bytes\n");
}

// Set bank size for conflict optimization
cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
```

### Shared Memory Bank Conflicts

Shared memory has 32 banks, each 4 bytes wide (default) or 8 bytes wide (configurable). Simultaneous accesses to the same bank by different threads in a warp cause bank conflicts:

```
Bank mapping (4-byte mode):
  Address       Bank
  0-3 bytes     Bank 0
  4-7 bytes     Bank 1
  ...
  124-127 bytes Bank 31
  128-131 bytes Bank 0  (repeats)

Bank mapping (8-byte mode):
  Address       Bank
  0-7 bytes     Bank 0
  8-15 bytes    Bank 1
  ...
  248-255 bytes Bank 31
  256-263 bytes Bank 0  (repeats)
```

```cpp
// Padding to avoid bank conflicts for float arrays
// Without padding: consecutive threads access consecutive floats -> no conflict
// With stride-32 access: padding avoids conflicts
__shared__ float tile[32][33];  // 33 instead of 32 to avoid bank conflicts
                                // in column-major access patterns
```

---

## 15.6 Tensor Core Input Types

Tensor cores provide hardware-accelerated matrix multiply-accumulate (MMA) operations. The supported input data types depend on the compute capability:

| CC | FP64 | TF32 | BF16 | FP16 | FP8 (E4M3/E5M2) | FP6 (E3M2/E2M3) | FP4 (E2M1) | INT8 | INT4 |
|---|---|---|---|---|---|---|---|---|---|
| 7.0 | | | | Yes | | | | Yes | |
| 7.5 | | | | Yes | | | | Yes | Yes |
| 8.0 | Yes | Yes | Yes | Yes | | | | Yes | Yes |
| 8.6 | Yes | Yes | Yes | Yes | | | | Yes | Yes |
| 8.9 | Yes | Yes | Yes | Yes | Yes | | | Yes | |
| 9.0 | Yes | Yes | Yes | Yes | Yes | | | Yes | |
| 9.2 | Yes | Yes | Yes | Yes | Yes | | | Yes | |
| 10.0 | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| 11.0 | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

### Using Tensor Cores via WMMA (Warp-level Matrix Multiply-Accumulate)

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void tensorCoreMatMul(
    const half* A, const half* B, float* C,
    int M, int N, int K)
{
    // Tile dimensions for FP16 tensor cores
    // WMMA fragment sizes: 16x16x16 for FP16
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    fill_fragment(c_frag, 0.0f);

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y) / warpSize;

    for (int i = 0; i < K; i += 16) {
        load_matrix_sync(a_frag, A + warpM * 16 * K + i, K);
        load_matrix_sync(b_frag, B + i * N + warpN * 16, N);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    store_matrix_sync(C + warpM * 16 * N + warpN * 16, c_frag, N, mem_row_major);
}
```

### Using Tensor Cores via PTX MMA (Lower-level)

```cpp
// Hopper (CC 9.0+) MMA instruction for FP16, 16x8x16 tile
__global__ void mmaFP16(const half* A, const half* B, float* C, int K) {
    // Each warp computes a 16x8 output tile
    // A is 16xK (row-major), B is Kx8 (col-major)

    uint32_t a_frag[4];  // 4 registers for 16x16 FP16 values
    uint32_t b_frag[2];  // 2 registers for 16x8 FP16 values
    uint32_t c_frag[4] = {0, 0, 0, 0};  // accumulator in FP32

    for (int k = 0; k < K; k += 16) {
        // Load A fragment (row-major, 16x16)
        // Load B fragment (col-major, 16x8)
        // ... PTX inline assembly for load ...

        // MMA operation
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%0, %1, %2, %3};"
            : "+f"(c_frag[0]), "+f"(c_frag[1]),
              "+f"(c_frag[2]), "+f"(c_frag[3])
            : "r"(a_frag[0]), "r"(a_frag[1]),
              "r"(a_frag[2]), "r"(a_frag[3]),
              "r"(b_frag[0]), "r"(b_frag[1])
        );
    }
}
```

### TF32 Mode (CC 8.0+)

```cpp
// Enable TF32 tensor cores for cuBLAS
cublasHandle_t handle;
cublasCreate(&handle);

// Enable TF32 (reduced precision: 10-bit mantissa, 8-bit exponent)
cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

// Or enable math to allow TF32 by default
cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
// Note: TF32 is used by default for single-precision GEMM on Ampere+
```

### FP8 (CC 8.9+)

```cpp
#include <cuda_fp8.h>

// FP8 types: __nv_fp8_e4m3 (for inputs) and __nv_fp8_e5m2 (for gradients)
__global__ void fp8MatMul(
    const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B,
    float* C, int M, int N, int K)
{
    // FP8 tensor cores provide 2x throughput vs FP16
    // Use WMMA or PTX MMA with FP8 operands
}
```

---

## 15.7 Feature Support Matrix

Key architectural features and their compute capability requirements:

| Feature | Min CC | Introduced | Notes |
|---|---|---|---|
| **Dynamic Parallelism** | 3.5 | Kepler | Device-side kernel launch |
| **Unified Memory** | 6.0 | Pascal | Single address space |
| **Pageable memory access** | 7.0 | Volta | No pinned memory required |
| **Independent thread scheduling** | 7.0 | Volta | Threads in warp can diverge/sync independently |
| **Cooperative groups** | 7.0 | Volta | Flexible synchronization primitives |
| **Tensor cores (1st gen)** | 7.0 | Volta | FP16 only, Volta-specific |
| **FP16 tensor cores** | 7.5 | Turing | Consumer tensor cores |
| **INT4/INT8 tensor cores** | 7.5 | Turing | Quantized inference |
| **128-bit atomics** | 7.5 | Turing | `atomicAdd` for `__int128` etc. |
| **Bfloat16** | 8.0 | Ampere | `__nv_bfloat16` type |
| **TF32 tensor cores** | 8.0 | Ampere | 19-bit floating point for ML |
| **FP64 tensor cores** | 8.0 | Ampere | `__nv_fp64` in tensor ops |
| **L2 cache residency control** | 8.0 | Ampere | `cudaAccessPolicyWindow` |
| **Async copy (cp.async)** | 8.0 | Ampere | Global-to-shared async copy |
| **Hardware barriers** | 8.0 | Ampere | `barrier` and `fence` instructions |
| **Thread block clusters** | 9.0 | Hopper | Cross-SM cooperation |
| **Distributed shared memory** | 9.0 | Hopper | Remote SMEM access in cluster |
| **Tensor memory accelerator (TMA)** | 9.0 | Hopper | Hardware-assisted bulk transfer |
| **DPX instructions** | 8.9 | Ada Lovelace | Dynamic programming acceleration |
| **FP8 tensor cores** | 8.9 | Ada Lovelace | `__nv_fp8_e4m3`, `__nv_fp8_e5m2` |
| **Uniform memory** | 9.0 | Hopper | Read-only data cached uniformly |
| **FP6 tensor cores** | 10.0 | Blackwell | `__nv_fp6_e3m2`, `__nv_fp6_e2m3` |
| **FP4 tensor cores** | 10.0 | Blackwell | `__nv_fp4_e2m1` |
| **INT4 tensor cores** | 10.0 | Blackwell | Quantized inference acceleration |
| **Enhanced hardware memcpy_async** | 9.0 | Hopper | TMA-based bulk async copy |
| **Cluster barrier** | 9.0 | Hopper | Synchronization across cluster |

### 128-bit Atomics (CC 7.5+)

```cpp
// 128-bit atomic operations on CC 7.5+
__global__ void atomic128Example(unsigned long long* addr) {
    // atomicAdd for unsigned long long (64-bit)
    unsigned long long old = atomicAdd(addr, 1ULL);

    // 128-bit CAS requires inline PTX
    // Available on CC 7.5+ with 128-bit memory operations
}
```

### L2 Cache Residency Control (CC 8.0+)

```cpp
// Persist data in L2 cache for repeated access (CC 8.0+)
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

// Set aside a portion of L2 cache for persistent access
cudaAccessPolicyWindow window;
window.base_ptr = nullptr;
window.num_bytes = prop.persistingL2CacheMaxSize;  // max persisting size
window.hitRatio = 0.5;   // 50% of accesses go to persisting region
window.hitProp = cudaAccessPropertyPersisting;
window.missProp = cudaAccessPropertyStreaming;

// Apply to a memory range
cudaCtxPersistAccess::SetAccessPolicyWindow(window);
```

### Thread Block Clusters (CC 9.0+)

```cpp
#include <cuda/barrier>

// Launch a kernel with thread block clusters (CC 9.0+)
__global__ void __cluster_dims__(2, 1, 1)  // 2 blocks per cluster
clusterKernel(...) {
    // Access shared memory of other blocks in the cluster
    namespace cg = cooperative_groups;
    auto cluster = cg::this_cluster();

    // Get dimensions of the cluster
    dim3 cluster_dim = cluster.dim_blocks();
    dim3 cluster_rank = cluster.block_index();

    // Distributed shared memory: read from another block's SMEM
    __shared__ int my_data[256];
    my_data[threadIdx.x] = threadIdx.x;
    cluster.sync();  // synchronize all blocks in the cluster

    // Read from block 0's shared memory
    int* remote_smem = cluster.map_shared_rank(my_data, 0);
    int val = remote_smem[threadIdx.x];
}
```

### Tensor Memory Accelerator / TMA (CC 9.0+)

```cpp
#include <cuda/barrier>

// TMA: hardware-accelerated bulk transfer from global to shared memory
__global__ void __cluster_dims__(1, 1, 1)
tmaKernel(cudaTmaDescriptor tma_desc, ...) {
    // TMA descriptor was created on host with cudaCreateTmaDescriptor

    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
    // Initialize barrier with expected arrival count = 1 (the TMA unit)
    init(&barrier, 1);

    // Issue TMA copy (single thread in block initiates)
    if (threadIdx.x == 0) {
        cuda::memcpy_async(barrier, tma_desc, smem_ptr, coord);
    }

    // All threads wait for TMA copy completion
    barrier.arrive_and_wait();
}
```

### DPX Instructions (CC 8.9+)

```cpp
// DPX: Dynamic programming acceleration instructions
// Available on Ada Lovelace (CC 8.9) and later
__global__ void dpxExample() {
    int a = 10, b = 20, c = 15;

    // Minimum of 3 values
    int min3 = __vimin3_s32(a, b, c);      // min(a, b, c) = 10

    // Maximum of 3 values
    int max3 = __vimax3_s32(a, b, c);      // max(a, b, c) = 20

    // Add and minimize/maximize
    int addmin = __viaddmin_s32(a, b, c);  // min(a+b, c) = min(30, 15) = 15
    int addmax = __viaddmax_s32(a, b, c);  // max(a+b, c) = max(30, 15) = 30
}
```

### Hardware memcpy_async (CC 8.0+, Enhanced in 9.0+)

```cpp
// Ampere (CC 8.0+): cp.async instruction
__global__ void asyncCopyAmpere(const float* src, float* dst, int N) {
    __shared__ float smem[256];

    // Asynchronous copy from global to shared memory
    // Single 4-byte or 8-byte or 16-byte copy per thread
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        // PTX-level async copy
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
            :: "r"((unsigned)&smem[i % 256]),
               "l"((unsigned long long)&src[i]),
               "n"(4));
    }

    // Wait for all pending async copies to complete
    asm volatile("cp.async.wait_all;");
    __syncthreads();
}

// Hopper (CC 9.0+): TMA-based bulk async
__global__ void asyncCopyHopper(cudaTmaDescriptor tma_desc) {
    // Uses TMA for multi-dimensional bulk transfers
    // See TMA section above
}
```

### Hardware Barriers (CC 8.0+)

```cpp
#include <cuda/barrier>

__global__ void barrierExample() {
    // Block-scope barrier (CC 8.0+)
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> bar;
    init(&bar, blockDim.x);
    bar.arrive_and_wait();

    // Cluster-scope barrier (CC 9.0+)
    // Requires cluster launch
    namespace cg = cooperative_groups;
    auto cluster = cg::this_cluster();
    cluster.sync();
}
```

### Feature Query at Runtime

```cpp
// Check specific features at runtime
void checkFeatures(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("Device: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);

    // Check for various features
    printf("Cooperative launch:   %s\n",
        prop.cooperativeLaunch ? "Yes" : "No");
    printf("Cooperative multi-device: %s\n",
        prop.cooperativeMultiDeviceLaunch ? "Yes" : "No");
    printf("Concurrent kernels:   %s\n",
        prop.concurrentKernels ? "Yes" : "No");
    printf("Unified addressing:   %s\n",
        prop.unifiedAddressing ? "Yes" : "No");
    printf("Managed memory:       %s\n",
        prop.managedMemory ? "Yes" : "No");
    printf("Compute preemption:   %s\n",
        prop.computePreemptionSupported ? "Yes" : "No");
    printf("Can use host pointers: %s\n",
        prop.canUseHostPointerForRegisteredMem ? "Yes" : "No");

    // Cluster launch support (CC 9.0+)
    printf("Cluster launch:       %s\n",
        prop.clusterLaunch ? "Yes" : "No");

    // Check specific attributes
    int accessPolicyWin, persistingL2;
    cudaDeviceGetAttribute(&accessPolicyWin,
        cudaDevAttrAccessPolicyMaxWindowSize, device);
    cudaDeviceGetAttribute(&persistingL2,
        cudaDevAttrMaxPersistingL2CacheSize, device);
    printf("L2 persisting cache:  %d bytes\n", persistingL2);
}
```
