# 26. Memory Optimizations

This document covers best practices for optimizing memory usage in CUDA applications. Memory performance is typically the primary bottleneck in GPU computing, and effective memory optimization often yields the largest performance gains. Topics include minimizing host-device transfers, using pinned memory, overlapping transfers with computation, achieving coalesced global memory access, optimizing shared memory usage, and NUMA considerations.

---

## Table of Contents

1. [Minimize Host-Device Transfers](#261-minimize-host-device-transfers)
2. [Pinned Memory](#262-pinned-memory)
3. [Overlapping Transfers with Computation](#263-overlapping-transfers-with-computation)
4. [Staged Concurrent Copy and Execute](#264-staged-concurrent-copy-and-execute)
5. [Zero Copy and Unified Virtual Addressing](#265-zero-copy-and-unified-virtual-addressing)
6. [Coalesced Global Memory Access](#266-coalesced-global-memory-access)
7. [Shared Memory Optimization](#267-shared-memory-optimization)
8. [NUMA Best Practices](#268-numa-best-practices)

---

## 26.1 Minimize Host-Device Transfers

The single most important memory optimization is minimizing the volume of data transferred between host and device. PCIe bandwidth (16-32 GB/s for PCIe 4.0 x16) is orders of magnitude lower than GPU device memory bandwidth (1-8 TB/s). Every unnecessary transfer creates a bottleneck.

### 26.1.1 Bandwidth Comparison

| Path | Typical Bandwidth | Notes |
|------|-------------------|-------|
| GPU Global Memory (H100) | 3,350 GB/s | HBM3 |
| GPU Global Memory (A100) | 2,039 GB/s | HBM2e |
| GPU Global Memory (RTX 4090) | 1,008 GB/s | GDDR6X |
| PCIe 4.0 x16 | 25-32 GB/s | Bidirectional |
| PCIe 5.0 x16 | 50-64 GB/s | Bidirectional |
| NVLink (H100) | 900 GB/s | GPU-to-GPU |

A kernel that reads 1 GB of data from device memory takes approximately 0.5 ms on an H100 but transferring that same 1 GB over PCIe 4.0 takes approximately 40 ms -- an 80x difference.

### 26.1.2 Keep Data on Device

Leave data on the GPU across multiple kernel launches. Intermediate results should never make a round-trip to the host.

```cpp
// BAD: Transfer data back and forth between kernels
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
kernel1<<<grid, block>>>(d_A, d_B);
cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);  // Unnecessary transfer!
// ... CPU does nothing with h_B ...
cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);   // Send it right back!
kernel2<<<grid, block>>>(d_B, d_C);
cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

// GOOD: Keep data on device
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
kernel1<<<grid, block>>>(d_A, d_B);      // d_B stays on device
kernel2<<<grid, block>>>(d_B, d_C);      // Use d_B directly
cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);  // Only final result
```

### 26.1.3 Batch Small Transfers

Many small transfers are much less efficient than one large transfer due to per-transfer overhead (driver dispatch, command processing). Batch small buffers into a single large transfer.

```cpp
// BAD: Many small transfers
for (int i = 0; i < 1000; i++) {
    cudaMemcpy(d_arrays[i], h_arrays[i], smallSize, cudaMemcpyHostToDevice);
}

// GOOD: Pack into one contiguous buffer and transfer once
// Option 1: Pack into a single contiguous host buffer
size_t totalSize = 1000 * smallSize;
cudaMemcpy(d_buffer, h_packed, totalSize, cudaMemcpyHostToDevice);

// Option 2: Use cudaMemcpy3D or cudaMemcpy2D for strided data
// Option 3: Use a batched transfer approach
```

### 26.1.4 Compute on Device, Even for Intermediate Steps

Move computation to the data rather than data to the computation. If a small amount of processing is needed between kernel launches, write a GPU kernel for it rather than transferring data to the CPU.

```cpp
// BAD: Transfer to CPU for a simple operation
cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
for (int i = 0; i < N; i++) h_data[i] *= 2.0f;  // Simple scale on CPU
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
processKernel<<<grid, block>>>(d_data);

// GOOD: Do the scale on the GPU
scaleKernel<<<grid, block>>>(d_data, 2.0f, N);  // Trivial GPU kernel
processKernel<<<grid, block>>>(d_data);
```

---

## 26.2 Pinned Memory

Pinned (page-locked) host memory is guaranteed by the operating system to remain at a fixed physical address and never be paged out to disk. This is critical for high-performance CUDA transfers because the GPU DMA engine can directly access pinned memory without staging through an intermediate buffer.

### 26.2.1 Allocating Pinned Memory

```cpp
// Method 1: cudaMallocHost (allocates pinned host memory)
float* h_pinned;
cudaMallocHost(&h_pinned, N * sizeof(float));

// Use h_pinned like normal host memory
for (int i = 0; i < N; i++) h_pinned[i] = (float)i;

// Transfer (faster than pageable memory)
cudaMemcpy(d_data, h_pinned, N * sizeof(float), cudaMemcpyHostToDevice);

// Free with cudaFreeHost (not free)
cudaFreeHost(h_pinned);

// Method 2: cudaHostAlloc (more control over flags)
float* h_pinned2;
cudaHostAlloc(&h_pinned2, N * sizeof(float), cudaHostAllocDefault);
cudaFreeHost(h_pinned2);
```

### 26.2.2 cudaHostAlloc Flags

| Flag | Description |
|------|-------------|
| `cudaHostAllocDefault` | Default behavior; page-locked, accessible from GPU |
| `cudaHostAllocPortable` | Memory is accessible by all CUDA contexts (not just the one that allocated it) |
| `cudaHostAllocMapped` | Maps the allocation into the GPU address space; enables zero-copy access |
| `cudaHostAllocWriteCombined` | Write-combining; faster GPU reads, slower CPU reads; good for host-to-device streaming data |
| `cudaHostAllocCoherent` (CC 9.0+) | Coherent between host and device; eliminates need for explicit synchronization |

```cpp
// Write-combined memory for host-to-device streaming
float* h_wc;
cudaHostAlloc(&h_wc, size, cudaHostAllocWriteCombined);
// CPU writes to h_wc are buffered for faster GPU reads via DMA
// Trade-off: CPU reads from h_wc are very slow

// Mapped memory for zero-copy (see Section 26.5)
float* h_mapped;
cudaHostAlloc(&h_mapped, size, cudaHostAllocMapped);
float* d_mapped;
cudaHostGetDevicePointer(&d_mapped, h_mapped, 0);
// d_mapped points to the same physical memory as h_mapped
```

### 26.2.3 Pinning Existing Memory

For cases where the host memory is already allocated (e.g., by a library or framework), use `cudaHostRegister` to pin it in place.

```cpp
// Allocate normal pageable memory
float* h_existing = (float*)malloc(N * sizeof(float));

// ... fill h_existing ...

// Pin the existing allocation
cudaHostRegister(h_existing, N * sizeof(float), cudaHostRegisterDefault);

// Now transfers using h_existing are fast and can be async
cudaMemcpyAsync(d_data, h_existing, N * sizeof(float),
                cudaMemcpyHostToDevice, stream);

// Unpin when done (must unpin before freeing)
cudaHostRegister(h_existing);
free(h_existing);
```

### 26.2.4 When to Use Pinned Memory

| Scenario | Use Pinned? | Reason |
|----------|-------------|--------|
| Async transfers (cudaMemcpyAsync) | Yes | Required for true asynchronous behavior |
| Overlapping transfers with kernels | Yes | DMA engine needs pinned memory |
| Large streaming data | Yes | Higher bandwidth, less CPU overhead |
| Small one-time transfers | Optional | Overhead of pinning may not be worth it |
| Very large allocations | Use caution | Pinned memory is not swappable; reduces available system RAM |

**Caution**: Pinned memory is not pageable. Allocating too much pinned memory can starve the OS and other processes of physical RAM. As a rule of thumb, keep pinned allocations well below total system memory (typically under 50%).

---

## 26.3 Overlapping Transfers with Computation

Overlapping host-device memory transfers with kernel execution is one of the most effective techniques for hiding transfer latency. This requires the GPU to have a dedicated copy engine (available on all modern GPUs) and pinned host memory.

### 26.3.1 Basic Overlap Pattern

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Allocate pinned host memory (required for async overlap)
float *h_a, *h_b;
cudaMallocHost(&h_a, size);
cudaMallocHost(&h_b, size);

// Device memory
float *d_a, *d_otherData, *d_result;
cudaMalloc(&d_a, size);
cudaMalloc(&d_otherData, size);
cudaMalloc(&d_result, size);

// Overlap: transfer in stream1, compute in stream2
cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream1);
kernel<<<grid, block, 0, stream2>>>(d_otherData, d_result);

// Both operations execute concurrently on different hardware units
// (DMA engine handles the transfer, SMs handle the kernel)

cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
```

### 26.3.2 Bidirectional Overlap

Modern GPUs have separate DMA engines for host-to-device and device-to-host transfers, enabling bidirectional overlap.

```cpp
cudaStream_t h2d_stream, d2h_stream, compute_stream;
cudaStreamCreate(&h2d_stream);
cudaStreamCreate(&d2h_stream);
cudaStreamCreate(&compute_stream);

// Three concurrent operations:
// 1. Upload new data (H2D)
cudaMemcpyAsync(d_input, h_input, size, cudaMemcpyHostToDevice, h2d_stream);

// 2. Download previous results (D2H)
cudaMemcpyAsync(h_output, d_output, size, cudaMemcpyDeviceToHost, d2h_stream);

// 3. Compute on different data
processKernel<<<grid, block, 0, compute_stream>>>(d_buffer);

// All three can execute concurrently
cudaDeviceSynchronize();
```

---

## 26.4 Staged Concurrent Copy and Execute

For large datasets, the staged (or double-buffered) pattern splits the data into chunks and processes them in a pipeline, overlapping each chunk's transfer with the previous chunk's computation.

### 26.4.1 Basic Staged Pattern

```cpp
const int nStreams = 4;
int chunkSize = N / nStreams;
size_t chunkBytes = chunkSize * sizeof(float);

cudaStream_t streams[nStreams];
for (int i = 0; i < nStreams; i++) {
    cudaStreamCreate(&streams[i]);
}

// Stage 0: Transfer chunk 0
// Stage 1: Transfer chunk 1, compute chunk 0
// Stage 2: Transfer chunk 2, compute chunk 1, transfer result 0
// ...

for (int i = 0; i < nStreams; i++) {
    int offset = i * chunkSize;

    // Transfer input chunk to device
    cudaMemcpyAsync(d_data + offset, h_data + offset, chunkBytes,
                    cudaMemcpyHostToDevice, streams[i]);

    // Process chunk on device (overlaps with next transfer)
    kernel<<<grid, block, 0, streams[i]>>>(d_data + offset, d_result + offset,
                                            chunkSize);

    // Transfer result chunk back to host
    cudaMemcpyAsync(h_result + offset, d_result + offset, chunkBytes,
                    cudaMemcpyDeviceToHost, streams[i]);
}

// Wait for all stages to complete
cudaDeviceSynchronize();
```

### 26.4.2 Double-Buffered Pattern

```cpp
// Double-buffered: process one chunk while transferring the next
const int N = 1 << 24;  // 16M elements
const int chunkSize = N / 2;
size_t chunkBytes = chunkSize * sizeof(float);

float *d_buf[2], *h_buf[2];
for (int i = 0; i < 2; i++) {
    cudaMalloc(&d_buf[i], chunkBytes);
    cudaMallocHost(&h_buf[i], chunkBytes);
}

cudaStream_t copyStream, computeStream;
cudaStreamCreate(&copyStream);
cudaStreamCreate(&computeStream);

// Prefetch first chunk
cudaMemcpyAsync(d_buf[0], h_buf[0], chunkBytes,
                cudaMemcpyHostToDevice, copyStream);
cudaStreamSynchronize(copyStream);

for (int chunk = 0; chunk < 2; chunk++) {
    int cur = chunk % 2;
    int next = (chunk + 1) % 2;

    // Start transfer of next chunk (if there is one)
    if (chunk + 1 < 2) {
        cudaMemcpyAsync(d_buf[next], h_buf[next], chunkBytes,
                        cudaMemcpyHostToDevice, copyStream);
    }

    // Compute current chunk (overlaps with next transfer)
    kernel<<<grid, block, 0, computeStream>>>(d_buf[cur], chunkSize);

    // Start result transfer
    cudaMemcpyAsync(h_buf[cur], d_buf[cur], chunkBytes,
                    cudaMemcpyDeviceToHost, copyStream);

    cudaStreamSynchronize(computeStream);
}
cudaDeviceSynchronize();
```

### 26.4.3 Circular Pipeline with Events

```cpp
const int PIPELINE_DEPTH = 3;
cudaStream_t streams[PIPELINE_DEPTH];
cudaEvent_t  events[PIPELINE_DEPTH];

for (int i = 0; i < PIPELINE_DEPTH; i++) {
    cudaStreamCreate(&streams[i]);
    cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
}

for (int chunk = 0; chunk < numChunks; chunk++) {
    int s = chunk % PIPELINE_DEPTH;
    int offset = chunk * chunkSize;

    // Wait for this stream to finish previous iteration
    // (pipeline depth limits in-flight chunks)
    cudaEventSynchronize(events[s]);

    // Record event at start of this iteration
    cudaEventRecord(events[s], streams[s]);

    // H2D transfer
    cudaMemcpyAsync(d_data + offset, h_data + offset, chunkBytes,
                    cudaMemcpyHostToDevice, streams[s]);

    // Compute
    kernel<<<grid, block, 0, streams[s]>>>(d_data + offset, chunkSize);

    // D2H transfer
    cudaMemcpyAsync(h_result + offset, d_result + offset, chunkBytes,
                    cudaMemcpyDeviceToHost, streams[s]);
}

cudaDeviceSynchronize();
```

---

## 26.5 Zero Copy and Unified Virtual Addressing

### 26.5.1 Zero Copy Memory

Zero copy provides direct GPU access to host memory without an explicit `cudaMemcpy`. The GPU accesses host memory over PCIe on-demand, caching it locally. Zero copy eliminates explicit transfer latency but has lower bandwidth than device memory.

**When to use zero copy:**

- Data is read or written only once (no benefit to staging in device memory).
- The system has an integrated GPU (shared physical memory).
- Memory is too large to fit in device memory.
- Sporadic, random access patterns where prefetching is ineffective.

```cpp
// Allocate mapped pinned host memory
float* h_data;
cudaHostAlloc(&h_data, N * sizeof(float), cudaHostAllocMapped);

// Initialize on host
for (int i = 0; i < N; i++) h_data[i] = (float)i;

// Get device pointer to the same physical memory
float* d_data;
cudaHostGetDevicePointer(&d_data, h_data, 0);

// Kernel accesses host memory directly over PCIe
kernel<<<grid, block>>>(d_data, N);

// No explicit cudaMemcpy needed -- but kernel is slower than
// if data were in device memory due to PCIe latency per access
```

**When NOT to use zero copy:**

- Data is accessed multiple times (use explicit copy to device memory instead).
- Bandwidth-sensitive kernels (PCIe is 50-100x slower than HBM).
- Random write patterns from many threads (PCIe contention).

### 26.5.2 Unified Virtual Addressing (UVA)

On 64-bit systems with compute capability 2.0 and later, CUDA uses a Unified Virtual Address (UVA) space. Host and device memory share a single address space, and the CUDA runtime can determine from a pointer's address whether it refers to host or device memory.

```cpp
// UVA: The runtime knows where each pointer lives
float *h_ptr, *d_ptr;
cudaMallocHost(&h_ptr, size);  // Host pinned memory
cudaMalloc(&d_ptr, size);       // Device memory

// cudaMemcpy direction can be "cudaMemcpyDefault" with UVA
cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyDefault);  // Runtime figures out direction
cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDefault);  // Same call, reverse direction

// Works for peer-to-peer copies between GPUs too
cudaMemcpy(d_ptr_gpu0, d_ptr_gpu1, size, cudaMemcpyDefault);
```

### 26.5.3 Unified Memory (cudaMallocManaged)

Unified Memory provides a single, coherent memory space accessible from both CPU and GPU. The CUDA runtime handles data movement transparently through page migration and caching.

```cpp
// Allocate unified memory
float* data;
cudaMallocManaged(&data, N * sizeof(float));

// Access from CPU
for (int i = 0; i < N; i++) data[i] = (float)i;

// Prefetch to GPU before kernel launch (reduces page faults)
int device = 0;
cudaMemPrefetchAsync(data, N * sizeof(float), device, stream);

// Use in kernel -- same pointer
kernel<<<grid, block, 0, stream>>>(data, N);

// Access from CPU after kernel completes
cudaStreamSynchronize(stream);
printf("Result: %f\n", data[0]);  // Runtime migrates pages back to CPU

cudaFree(data);
```

**Unified Memory hints (CC 7.0+):**

```cpp
// Advise the driver about access patterns
cudaMemAdvise(data, N * sizeof(float), cudaMemAdviseSetReadMostly, device);
cudaMemAdvise(data, N * sizeof(float), cudaMemAdviseSetPreferredLocation, device);
cudaMemAdvise(data, N * sizeof(float), cudaMemAdviseSetAccessedBy, device);
```

---

## 26.6 Coalesced Global Memory Access

Coalesced access is the single most important factor for global memory performance. When threads in a warp access contiguous, aligned global memory locations, the hardware combines (coalesces) these accesses into the minimum number of memory transactions.

### 26.6.1 How Coalescing Works

Global memory is accessed in 32-byte segments (transactions). For a warp of 32 threads:

- **Best case**: All 32 threads access consecutive, aligned 4-byte words (128 bytes total). This results in 4 transactions of 32 bytes each (or fewer with caching).
- **Worst case**: All 32 threads access scattered addresses, resulting in up to 32 separate transactions.

```
Coalesced access (ideal):
Thread:  0   1   2   3  ...  31
Address: 0   4   8   12 ... 124
         |----128 bytes contiguous----|
         -> 4 x 32-byte transactions

Strided access (poor):
Thread:  0    1    2    3  ...  31
Address: 0   128  256  384... 3968
         |---scattered, non-contiguous---|
         -> up to 32 transactions
```

### 26.6.2 Access Pattern Examples

```cpp
// Pattern 1: Sequential access -- FULLY COALESCED (best)
__global__ void sequentialAccess(const float* in, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = in[idx];  // Consecutive threads -> consecutive addresses
    }
}

// Pattern 2: Strided access -- NOT COALESCED (bad)
__global__ void stridedAccess(const float* in, float* out, int N, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int address = idx * stride;
    if (address < N) {
        out[address] = in[address];  // stride > 1 wastes bandwidth
    }
}

// Pattern 3: Structure of Arrays (SoA) -- COALESCED
struct ParticleSoA {
    float* x;  // N consecutive floats
    float* y;  // N consecutive floats
    float* z;  // N consecutive floats
};

__global__ void processSoA(ParticleSoA p, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float dist = sqrtf(p.x[idx]*p.x[idx] +
                           p.y[idx]*p.y[idx] +
                           p.z[idx]*p.z[idx]);
        // Each array access is coalesced (consecutive threads -> consecutive floats)
    }
}

// Pattern 4: Array of Structures (AoS) -- NOT FULLY COALESCED
struct ParticleAoS {
    float x, y, z;
};

__global__ void processAoS(const ParticleAoS* p, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Access pattern: x0,y0,z0, x1,y1,z1, x2,y2,z2, ...
        // threadIdx.x=0 accesses offset 0 (x0)
        // threadIdx.x=1 accesses offset 3 (x1)
        // threadIdx.x=2 accesses offset 6 (x2)
        // Stride of 3 floats between threads for same field -> not coalesced
        float dist = sqrtf(p[idx].x*p[idx].x +
                           p[idx].y*p[idx].y +
                           p[idx].z*p[idx].z);
    }
}

// Pattern 5: AoS to SoA conversion for coalesced access
__global__ void aosToSoA(const ParticleAoS* aos, ParticleSoA soa, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        soa.x[idx] = aos[idx].x;
        soa.y[idx] = aos[idx].y;
        soa.z[idx] = aos[idx].z;
    }
}
```

### 26.6.3 Matrix Multiply Optimization Through Coalescing

The classic matrix multiplication example demonstrates how memory access patterns directly affect bandwidth utilization. The progression below shows how coalescing improvements translate to measurable performance gains.

**Version 1: Naive (poor coalescing on matrix B)**

```cpp
// Each thread computes one element of C
// C[row][col] = sum(A[row][k] * B[k][col])
__global__ void matMulNaive(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // A[row*K+k]: sequential in k -> OK for threads in same row
            // B[k*N+col]: strided by N for consecutive k -> poor for B
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
// Bandwidth: ~119.9 GB/s (on a reference GPU)
```

**Version 2: Transposed B (improved coalescing)**

```cpp
// Transpose B first so that B^T[col][k] is sequential in k
__global__ void matMulTransposed(const float* A, const float* BT, float* C,
                                  int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // A[row*K+k]: sequential in k
            // BT[col*K+k]: sequential in k (transposed!)
            // Both accesses are coalesced within the same k loop iteration
            sum += A[row * K + k] * BT[col * K + k];
        }
        C[row * N + col] = sum;
    }
}
// Bandwidth: ~144.4 GB/s
```

**Version 3: Shared memory tiling (best coalescing + reuse)**

```cpp
#define TILE_SIZE 16

__global__ void matMulTiled(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Cooperative load: each thread loads one element
        // Fully coalesced because consecutive threads load consecutive addresses
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
// Bandwidth: ~195.5 GB/s (shared memory eliminates redundant global reads)
```

### 26.6.4 L1 Cache Line Size Considerations

On modern GPUs, global memory loads first go through the L1 cache, which uses 128-byte cache lines. A single thread reading a 4-byte float may fetch 128 bytes. If neighboring threads access nearby data, they benefit from this prefetch.

```cpp
// Understanding cache line behavior:
// - L1 cache line = 128 bytes
// - One float = 4 bytes -> 32 floats per cache line
// - A warp of 32 threads accessing 32 consecutive floats -> 1 cache line (ideal)
// - A warp accessing scattered floats -> many cache lines (wasteful)

// Recommendation: structure data layouts so that threads in a warp
// access addresses within the same 128-byte aligned segment
```

---

## 26.7 Shared Memory Optimization

Shared memory is a fast, user-managed on-chip memory that serves as a programmable cache. It is approximately 100x lower latency than global memory and provides high bandwidth. Effective use of shared memory is critical for performance-critical kernels.

### 26.7.1 Bank Conflicts

Shared memory is divided into 32 memory banks. Each bank services one address per clock cycle. When multiple threads in a warp access different addresses within the same bank simultaneously, a bank conflict occurs, and the accesses are serialized.

```
Bank layout (4-byte mode, default):
Address        Bank
0, 128, 256...  0
4, 132, 260...  1
8, 136, 264...  2
...
124,252,380...  31
```

**Conflict patterns:**

| Access Pattern | Conflict Level | Multiplier |
|---------------|----------------|------------|
| All threads access different banks | No conflict | 1x (fastest) |
| 2 threads access same bank | 2-way conflict | 2x slower |
| All 32 threads access same bank | 32-way conflict | 32x slower |
| All threads access same address | Broadcast (no conflict) | 1x |

```cpp
// Example: Bank conflict in matrix transpose
// Naive 2D shared memory layout
__shared__ float tile[32][32];  // 32 banks, 4-byte mode

// Reading a row: tile[row][0..31]
// Thread i reads tile[row][i] -> bank i -> no conflict

// Reading a column: tile[0..31][col]
// Thread i reads tile[i][col] -> bank (i*32 + col) % 32 = col
// All threads access bank 'col' -> 32-way conflict!

// Solution: Pad the shared memory array
__shared__ float tile_padded[32][33];  // Extra column breaks the stride
// Reading a column: tile_padded[i][col]
// Thread i reads at offset (i * 33 + col) -> bank (i*33 + col) % 32
// = (i + col) % 32 (since 33 mod 32 = 1)
// Each thread accesses a different bank -> no conflict!
```

### 26.7.2 Padding to Avoid Bank Conflicts

```cpp
// Padding technique for common patterns

// Pattern 1: 2D array with column access
// BAD: column access causes conflicts
__shared__ float matrix[16][16];

// GOOD: pad each row by 1 element
__shared__ float matrix_padded[16][17];  // 17 = 16 + 1 padding

// Pattern 2: Power-of-2 stride access
// BAD: accessing every 2nd element causes 2-way conflicts
__shared__ float data[64];
// Thread i accesses data[i * 2] -> bank (i*2) % 32 -> 2-way conflict

// GOOD: use odd stride or pad
__shared__ float data_padded[65];  // 64 + 1 padding
// Thread i accesses data_padded[i * 2] -> bank (i*2) % 32 with offset
// Still may conflict; better to restructure the access pattern

// Pattern 3: Configurable padding with template
template <int DIM, int PAD = 1>
struct SharedTile {
    float data[DIM][DIM + PAD];  // Always padded
    __device__ float& operator()(int row, int col) {
        return data[row][col];
    }
};

__global__ void kernelWithTile() {
    __shared__ SharedTile<32> tile;  // 32x33 padded array
    // Column access: tile(i, j) -> bank (i * 33 + j) % 32 = no conflict
}
```

### 26.7.3 Asynchronous Copy from Global to Shared Memory

Modern GPUs (CC 8.0+) provide hardware-accelerated asynchronous copy from global to shared memory, bypassing registers and reducing register pressure.

**Ampere (CC 8.0) -- cp.async:**

```cpp
// Use cooperative groups memcpy_async
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

__global__ void asyncCopyCG(const float* global_in, float* global_out, int N) {
    __shared__ float smem[256];

    cg::thread_block block = cg::this_thread_block();

    // Async copy from global to shared memory
    cg::memcpy_async(block, smem, global_in + blockIdx.x * 256,
                     256 * sizeof(float));

    // Wait for copy to complete
    cg::wait(block);

    // Process data in shared memory
    int tid = threadIdx.x;
    smem[tid] *= 2.0f;

    // Copy result back
    cg::memcpy_async(block, global_out + blockIdx.x * 256, smem,
                     256 * sizeof(float));
    cg::wait(block);
}
```

**Hopper (CC 9.0) -- TMA (Tensor Memory Accelerator):**

```cpp
#include <cuda/barrier>

// TMA bulk copy: a single thread initiates the transfer for the entire block
__global__ void asyncCopyTMA(cudaTmaDescriptor tma_desc,
                              float* global_out, int N) {
    __shared__ float smem[256];
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

    // Initialize barrier: expect 1 arrival (the TMA unit)
    init(&barrier, 1);

    // Single thread initiates TMA copy
    if (threadIdx.x == 0) {
        cuda::memcpy_async(barrier, tma_desc, smem,
                          {blockIdx.x * 256});  // coordinate in tensor
    }

    // All threads wait for copy to complete
    barrier.arrive_and_wait();

    // Process data
    int tid = threadIdx.x;
    if (blockIdx.x * 256 + tid < N) {
        smem[tid] *= 2.0f;
    }

    // Write back to global memory
    __syncthreads();
    if (blockIdx.x * 256 + tid < N) {
        global_out[blockIdx.x * 256 + tid] = smem[tid];
    }
}
```

### 26.7.4 Shared Memory Carveout Configuration

Shared memory is carved from the unified L1 data cache. The split between shared memory and L1 cache can be configured per-kernel.

```cpp
// Request maximum shared memory for a kernel
cudaFuncSetAttribute(
    myKernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared
);

// Request maximum L1 cache (minimum shared memory)
cudaFuncSetAttribute(
    myKernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxL1
);

// Set a specific percentage (0-100)
cudaFuncSetAttribute(
    myKernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    75  // 75% shared memory, 25% L1 cache
);

// Allow dynamic shared memory to exceed the default per-block limit
cudaFuncSetAttribute(
    myKernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    100 * 1024  // Allow up to 100 KB dynamic shared memory
);

// Launch with dynamic shared memory size
myKernel<<<grid, block, dynamicSharedBytes, stream>>>(...);
```

---

## 26.8 NUMA Best Practices

On multi-socket systems with Non-Uniform Memory Access (NUMA), the physical location of CPU memory relative to the GPU affects PCIe transfer performance. A GPU connected to a PCIe slot on CPU socket 0 has higher bandwidth and lower latency to memory local to socket 0 than to memory on socket 1.

### 26.8.1 NUMA-Aware Memory Allocation

```cpp
// Linux: Use numactl to bind process to a NUMA node
// numactl --cpunodebind=0 --membind=0 ./myapp

// Programmatically bind memory allocation
#include <numa.h>
#include <numaif.h>

void* allocateNumaMemory(size_t size, int numaNode) {
    void* ptr = numa_alloc_onnode(size, numaNode);
    if (!ptr) {
        fprintf(stderr, "Failed to allocate %zu bytes on NUMA node %d\n",
                size, numaNode);
        return nullptr;
    }
    return ptr;
}

// Then pin the NUMA-local memory for CUDA transfers
void* h_numa = allocateNumaMemory(size, 0);  // Allocate on NUMA node 0
cudaHostRegister(h_numa, size, cudaHostRegisterDefault);
// GPU on PCIe of CPU socket 0 will have optimal transfer performance
```

### 26.8.2 GPU-CPU Affinity

```cpp
// Determine which NUMA node a GPU is connected to
int getGpuNumaNode(int gpuDevice) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpuDevice);

    // On Linux, the NUMA node can be read from sysfs
    // /sys/class/drm/card0/device/numa_node
    // Or use the PCI bus ID to determine affinity

    printf("GPU %d (%s) PCI bus: %02x:%02x.%x\n",
           gpuDevice, prop.name,
           prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

    // Match PCI bus to CPU socket via system topology
    // Use hwloc or custom topology discovery
    return -1;  // Return NUMA node (implementation-dependent)
}

// Set thread affinity to match GPU NUMA node
void setThreadAffinityForGpu(int gpuDevice) {
    int numaNode = getGpuNumaNode(gpuDevice);
    if (numaNode >= 0) {
        // Bind the current thread to the NUMA node
        numa_run_on_node(numaNode);
        // Set memory allocation policy to prefer local node
        numa_set_preferred(numaNode);
    }
}
```

### 26.8.3 Multi-GPU NUMA Strategy

```cpp
// For multi-GPU systems with NUMA topology:
// 1. Assign each host thread to the NUMA node of its GPU
// 2. Allocate pinned memory on the local NUMA node
// 3. Use peer-to-peer (P2P) access for GPU-to-GPU transfers
//    instead of routing through host memory

void setupMultiGpuNuma(int numGpus) {
    for (int gpu = 0; gpu < numGpus; gpu++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, gpu);

        // Enable P2P access between GPUs (avoids host memory bounce)
        for (int peer = 0; peer < numGpus; peer++) {
            if (gpu != peer) {
                int canAccessPeer;
                cudaDeviceCanAccessPeer(&canAccessPeer, gpu, peer);
                if (canAccessPeer) {
                    cudaDeviceEnablePeerAccess(peer, 0);
                    printf("Enabled P2P: GPU %d <-> GPU %d\n", gpu, peer);
                }
            }
        }
    }
}
```

### 26.8.4 NUMA Quick Reference

| Practice | Benefit |
|----------|---------|
| Allocate host memory on the NUMA node closest to the GPU | Minimizes PCIe latency and maximizes bandwidth |
| Bind host threads to the NUMA node of their GPU | Avoids cross-socket memory access |
| Use NVLink for GPU-to-GPU transfers | Bypasses host memory entirely |
| Enable P2P access between GPUs | Avoids staging through host memory |
| Use `numactl --membind` for process launch | Ensures all allocations are NUMA-local |
| Use hwloc library for topology discovery | Portable NUMA topology queries |

---

## Summary

| Optimization | Key Benefit | Typical Speedup |
|---|---|---|
| Minimize transfers | Eliminates the largest bottleneck | Variable (up to 10-100x) |
| Pinned memory | Enables async transfers, higher bandwidth | 2-3x for transfers |
| Overlap transfer + compute | Hides transfer latency | Up to 2x for transfer-bound apps |
| Staged pipeline | Overlaps all phases of computation | Up to 2x for large datasets |
| Zero copy | Eliminates explicit copy for one-time access | Useful for specific patterns |
| Coalesced access | Minimizes global memory transactions | 2-10x for memory-bound kernels |
| Shared memory tiling | Data reuse, reduces global memory traffic | 2-5x for data-reuse patterns |
| Bank conflict avoidance | Prevents shared memory serialization | 2-32x within shared memory phase |
| Async global-to-shared copy | Hides copy latency, reduces register pressure | 10-30% for copy-heavy kernels |
| NUMA-aware allocation | Optimal PCIe bandwidth on multi-socket | 10-30% on NUMA systems |
