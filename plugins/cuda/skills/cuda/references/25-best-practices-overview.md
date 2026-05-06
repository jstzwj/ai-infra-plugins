# 25. Best Practices Overview

This document covers the foundational best practices for CUDA application development, including the APOD design cycle for systematic GPU acceleration, scaling laws that govern parallel speedup, verification and debugging strategies, and numerical accuracy considerations.

---

## Table of Contents

1. [APOD Design Cycle](#251-apod-design-cycle)
2. [Scaling Laws](#252-scaling-laws)
3. [Verification and Debugging](#253-verification-and-debugging)
4. [Numerical Accuracy](#254-numerical-accuracy)

---

## 25.1 APOD Design Cycle

The APOD (Assess, Parallelize, Optimize, Deploy) cycle is a structured, iterative methodology for accelerating applications with CUDA. It is designed to be applied incrementally: each pass through the cycle yields measurable improvements, and the process repeats until performance goals are met.

### 25.1.1 Assess

The first phase identifies the most impactful targets for GPU acceleration by analyzing the existing CPU application.

**Profile to find hotspots:**

- Use profiling tools such as `gprof`, Nsight Systems (`nsys`), or Nsight Compute (`ncu`) to determine where the application spends the most time.
- Focus on functions or loops that consume a significant fraction of total runtime. A function accounting for 5% of runtime is rarely worth accelerating; one accounting for 40% is a prime candidate.
- Identify data dependencies within hotspots: loops with independent iterations are the easiest to parallelize.

```bash
# Profile with gprof (CPU-side)
gcc -pg myapp.c -o myapp
./myapp
gprof myapp gmon.out > analysis.txt

# Profile with Nsight Systems (GPU-side)
nsys profile --trace=cuda,nvtx --output=myapp_report ./myapp
```

**Evaluate parallelism potential:**

- Use Amdahl's Law and Gustafson's Law (see Section 25.2) to estimate the theoretical maximum speedup.
- Determine the serial fraction `P` of the workload -- the portion that cannot be parallelized. This directly bounds achievable speedup.
- Consider data transfer overhead: if the input/output data must traverse PCIe for every kernel, the transfer cost can dominate unless managed carefully.

```cpp
// Quick assessment: measure baseline CPU performance
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();
cpuHotspotFunction(data, N);
auto end = std::chrono::high_resolution_clock::now();
double ms = std::chrono::duration<double, std::milli>(end - start).count();
printf("CPU baseline: %.3f ms for %d elements\n", ms, N);
```

### 25.1.2 Parallelize

Once hotspots are identified, parallelize them using the most appropriate approach. CUDA provides a spectrum of options ranging from high-level libraries to custom kernels.

**Use GPU-accelerated libraries when possible:**

Libraries are heavily optimized and often provide near-peak performance with minimal development effort. Prefer them over custom implementations unless the operation is highly specialized.

| Library | Domain | Key Operations |
|---------|--------|---------------|
| cuBLAS | Linear algebra | GEMM, GEMV, triangular solves, batched operations |
| cuFFT | Fourier transforms | 1D/2D/3D FFT, batched FFT, real-to-complex |
| cuRAND | Random numbers | Philox, XORWOW, MRG32k3a generators |
| cuSPARSE | Sparse linear algebra | SpMV, SpGEMM, triangular solves |
| Thrust | Parallel algorithms | sort, reduce, scan, transform, unique |
| cuDNN | Deep learning | Convolution, pooling, normalization, attention |

```cpp
// Example: replace CPU matrix multiply with cuBLAS
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

float alpha = 1.0f, beta = 0.0f;
cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    M, N, K,
    &alpha,
    d_A, M,
    d_B, K,
    &beta,
    d_C, M);
```

**Use directive-based approaches for quick porting:**

OpenACC and OpenMP target offloading allow incremental parallelization with pragmas, useful for large legacy codebases.

```c
// OpenACC parallelization of a loop
#pragma acc parallel loop copyin(a[0:N], b[0:N]) copyout(c[0:N])
for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
}
```

**Write custom CUDA kernels for specialized operations:**

When no library covers the use case, write custom kernels. Start simple and optimize iteratively.

```cpp
// Simple custom kernel as a starting point
__global__ void vectorAdd(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// Launch with basic configuration
int blockSize = 256;
int gridSize = (N + blockSize - 1) / blockSize;
vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
```

### 25.1.3 Optimize

After parallelization produces a correct result, optimize iteratively. Use profiling data to guide each optimization step -- never optimize blindly.

**Profiling-driven optimization workflow:**

1. Profile the GPU-accelerated code with Nsight Systems to identify bottlenecks.
2. If kernel execution time dominates, use Nsight Compute to drill into individual kernel metrics.
3. Apply targeted optimizations based on the bottleneck type:
   - **Memory-bound**: improve coalescing, use shared memory, reduce transfers.
   - **Compute-bound**: use faster math functions, reduce redundant computation.
   - **Latency-bound**: increase occupancy, reduce warp divergence.
4. Re-profile after each optimization to measure improvement.
5. Repeat until performance goals are met or diminishing returns set in.

```bash
# Step 1: System-wide profile
nsys profile --trace=cuda,nvtx --stats=true ./myapp

# Step 2: Detailed kernel profile
ncu --set full --target-processes all --launch-skip 5 --launch-count 1 \
    -o kernel_profile ./myapp

# Step 3: Focus on specific metrics
ncu --metrics gpu__time_duration.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
lts__t_sectors_op_read.sum,\
sm__warps_active.avg.pct_of_peak \
    ./myapp
```

**Key optimization areas (covered in subsequent documents):**

- Memory optimizations (Section 26): coalesced access, shared memory, pinned transfers.
- Execution configuration (Section 27): occupancy, block size, concurrent kernels.
- Instruction optimization (Section 28): math intrinsics, warp-level primitives, tensor cores.

### 25.1.4 Deploy

Deploy the accelerated application even if only partially optimized. Early deployment provides real-world feedback and validates correctness under production conditions.

**Deployment considerations:**

- **Forward compatibility**: Compile fat binaries with multiple `-gencode` flags to support a range of GPU architectures.

```bash
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_90,code=compute_90 \
     myapp.cu -o myapp
```

- **Error handling**: Wrap all CUDA calls with error checking. Provide meaningful fallbacks when GPU resources are unavailable.

```cpp
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n",             \
                    __FILE__, __LINE__,                                     \
                    cudaGetErrorName(err), cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

CUDA_CHECK(cudaMalloc(&d_ptr, size));
CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));
```

- **Runtime capability checks**: Query the GPU at startup and select appropriate code paths.

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
int cc = prop.major * 10 + prop.minor;

if (cc >= 90) {
    // Use Hopper-optimized path (thread block clusters, TMA)
    runHopperKernel<<<...>>>(...);
} else if (cc >= 80) {
    // Use Ampere-optimized path (async copy, tensor cores)
    runAmpereKernel<<<...>>>(...);
} else {
    // Fallback path
    runBasicKernel<<<...>>>(...);
}
```

- **Iterative evolution**: APOD is a cycle, not a linear process. After deployment, collect production performance data and feed it back into the Assess phase for the next round of optimization.

---

## 25.2 Scaling Laws

Scaling laws provide theoretical bounds on the speedup achievable through parallelization. They are essential for setting realistic expectations during the Assess phase of APOD.

### 25.2.1 Amdahl's Law

Amdahl's Law models speedup when a fixed-size problem is parallelized. It states that the maximum speedup is bounded by the serial fraction of the workload.

**Formula:**

```
S = 1 / ((1 - P) + P / N)
```

Where:
- `S` = speedup (serial time / parallel time)
- `P` = fraction of the workload that is parallelizable (0 to 1)
- `N` = number of processors
- `(1 - P)` = serial fraction that cannot be parallelized

**Maximum speedup** (as N approaches infinity):

```
S_max = 1 / (1 - P)
```

**Implications:**

| Serial Fraction (1-P) | Max Speedup | Notes |
|------------------------|-------------|-------|
| 1% | 100x | Exceptional parallelization needed |
| 5% | 20x | Good parallelization achievable |
| 10% | 10x | Reasonable target |
| 25% | 4x | Limited benefit |
| 50% | 2x | Modest gains |

```cpp
// Calculate Amdahl's Law speedup
double amdahl_speedup(double P, int N) {
    return 1.0 / ((1.0 - P) + P / (double)N);
}

// Example: 95% parallelizable, 1000 processors
double P = 0.95;
int N = 1000;
printf("Speedup: %.2fx\n", amdahl_speedup(P, N));  // 16.81x
printf("Max speedup (N->inf): %.2fx\n", 1.0 / (1.0 - P));  // 20.00x
```

**Key insight**: Even a small serial fraction severely limits maximum speedup. If 5% of the workload is serial, no amount of additional GPU hardware can exceed 20x speedup. This makes eliminating serial bottlenecks critical.

### 25.2.2 Gustafson's Law

Gustafson's Law takes a different perspective: instead of keeping the problem size fixed, it scales the problem size proportionally with the number of processors. This better models real-world scenarios where more compute power is used to solve larger problems.

**Formula:**

```
S = N + (1 - P) * (1 - N)

Equivalently: S = N - (N - 1) * (1 - P) = N * P + (1 - P)
```

Where:
- `S` = scaled speedup
- `P` = parallel fraction of the workload (measured on the parallel system)
- `N` = number of processors

**Implications:**

Gustafson's Law is more optimistic than Amdahl's Law because it recognizes that larger problem sizes typically expose more parallelism. A workload that appears 95% parallel on a small problem might be 99.9% parallel on a problem 100x larger.

```cpp
// Calculate Gustafson's Law speedup
double gustafson_speedup(double P, int N) {
    return (double)N + (1.0 - P) * (1.0 - (double)N);
    // Equivalently: return (double)N * P + (1.0 - P);
}

// Example: 95% parallel fraction, 1000 processors
double P = 0.95;
int N = 1000;
printf("Gustafson speedup: %.2fx\n", gustafson_speedup(P, N));  // 950.05x
```

### 25.2.3 Practical Application of Scaling Laws

```cpp
// Comprehensive speedup analysis
void analyzeSpeedup(double serialFraction, int maxProcessors) {
    double P = 1.0 - serialFraction;

    printf("Serial fraction: %.1f%%, Parallel fraction: %.1f%%\n",
           serialFraction * 100, P * 100);
    printf("Amdahl max speedup: %.2fx\n\n", 1.0 / serialFraction);

    printf("%-10s %-15s %-15s\n", "Processors", "Amdahl", "Gustafson");
    printf("%-10s %-15s %-15s\n", "----------", "-------", "---------");

    int procs[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    for (int i = 0; i < 11; i++) {
        int N = procs[i];
        if (N > maxProcessors) break;
        double amdahl = 1.0 / (serialFraction + P / N);
        double gustafson = (double)N * P + serialFraction;
        printf("%-10d %-15.2f %-15.2f\n", N, amdahl, gustafson);
    }
}

// For a typical CUDA application with 2% serial fraction:
analyzeSpeedup(0.02, 1024);
```

---

## 25.3 Verification and Debugging

Correctness must be established before and maintained during optimization. CUDA introduces additional debugging challenges due to asynchronous execution, massive parallelism, and limited error reporting.

### 25.3.1 Reference Comparison

The most reliable verification method is comparing GPU results against a trusted CPU reference implementation.

```cpp
// Reference CPU implementation
void cpuReference(const float* input, float* output, int N) {
    for (int i = 0; i < N; i++) {
        output[i] = sqrtf(input[i]) * expf(-input[i]);
    }
}

// GPU kernel
__global__ void gpuKernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = sqrtf(input[idx]) * expf(-input[idx]);
    }
}

// Verification with tolerance
bool verifyResults(const float* ref, const float* gpu, int N,
                   float tolerance = 1e-5f) {
    float maxError = 0.0f;
    int maxErrorIdx = -1;
    for (int i = 0; i < N; i++) {
        float error = fabsf(ref[i] - gpu[i]);
        if (error > maxError) {
            maxError = error;
            maxErrorIdx = i;
        }
    }

    bool passed = (maxError < tolerance);
    printf("Verification: %s (max error = %e at index %d, tol = %e)\n",
           passed ? "PASSED" : "FAILED",
           maxError, maxErrorIdx, tolerance);
    return passed;
}

// Usage
float *h_ref = new float[N];
cpuReference(h_input, h_ref, N);

float *h_gpu = new float[N];
cudaMemcpy(h_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

verifyResults(h_ref, h_gpu, N);
```

### 25.3.2 Dual Testing with `__host__ __device__`

Functions marked with both `__host__` and `__device__` can be called from both CPU and GPU code. This enables running the same logic on both sides for comparison without code duplication.

```cpp
// Shared function callable from both host and device
__host__ __device__ float computeElement(float x) {
    return sqrtf(x) * expf(-x);
}

// CPU wrapper
void cpuCompute(const float* input, float* output, int N) {
    for (int i = 0; i < N; i++) {
        output[i] = computeElement(input[i]);
    }
}

// GPU kernel using the same function
__global__ void gpuCompute(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = computeElement(input[idx]);
    }
}

// Unit test: compare CPU and GPU results from the same source function
void unitTest() {
    const int N = 1024;
    float h_input[N], h_cpu_out[N], h_gpu_out[N];
    float *d_input, *d_output;

    // Initialize test data
    for (int i = 0; i < N; i++) h_input[i] = (float)i * 0.001f;

    // CPU computation
    cpuCompute(h_input, h_cpu_out, N);

    // GPU computation
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    gpuCompute<<<(N + 255) / 256, 256>>>(d_input, d_output, N);
    cudaMemcpy(h_gpu_out, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    verifyResults(h_cpu_out, h_gpu_out, N);

    cudaFree(d_input);
    cudaFree(d_output);
}
```

### 25.3.3 Debugging Tools

**CUDA-GDB** (Linux):

The CUDA-aware debugger supports breakpoints in device code, inspection of GPU memory and registers, and stepping through individual threads.

```bash
# Compile with debug symbols (-G disables optimizations, -lineinfo preserves source info)
nvcc -G -g myapp.cu -o myapp_debug

# Debug with cuda-gdb
cuda-gdb ./myapp_debug

# Common cuda-gdb commands:
# break myKernel           - set breakpoint in kernel
# break myfile.cu:42       - set breakpoint at line 42
# run                      - start execution
# cuda thread (0,0,0)      - focus on specific thread
# print myArray[0]         - print device variable
# cuda launch              - continue to next kernel launch
```

**Nsight Compute (`ncu`)**:

Nsight Compute provides detailed per-kernel profiling, including register usage, memory throughput, occupancy, and warp stall reasons. It is invaluable for identifying performance bottlenecks.

```bash
# Profile a specific kernel with full metrics
ncu --set full -o profile.ncu-rep ./myapp

# Quick profile with summary metrics
ncu --set quick ./myapp

# Profile with kernel name filter
ncu --kernel-name "myKernel*" --launch-count 1 ./myapp

# Compare two profiling sessions
ncu --import baseline.ncu-rep --import optimized.ncu-rep --compare
```

**Nsight Systems (`nsys`)**:

Nsight Systems provides a system-wide timeline view showing CPU activity, GPU kernels, memory transfers, and API calls. It is the primary tool for identifying concurrency opportunities and transfer bottlenecks.

```bash
# Generate timeline report
nsys profile --trace=cuda,nvtx,osrt --output=timeline ./myapp

# Open in the Nsight Systems GUI
nsys-ui timeline.qdrep

# Generate a text report
nsys stats timeline.qdrep
```

**Compute Sanitizer**:

Compute Sanitizer (formerly cuda-memcheck) detects memory errors, race conditions, and synchronization issues at runtime.

```bash
# Check for memory access errors
compute-sanitizer ./myapp

# Race condition detection with Thread Sanitizer
compute-sanitizer --tool racecheck ./myapp

# Memory access validation with Memcheck
compute-sanitizer --tool memcheck ./myapp

# Sync check for synchronization issues
compute-sanitizer --tool synccheck ./myapp
```

### 25.3.4 Common Debugging Pitfalls

**Asynchronous error masking:**

CUDA errors from kernel launches are reported asynchronously. An error from one kernel may appear at a completely unrelated API call. Use `CUDA_LAUNCH_BLOCKING=1` to make errors synchronous.

```bash
# Make all kernel launches synchronous for debugging
export CUDA_LAUNCH_BLOCKING=1
./myapp
```

```cpp
// Always check for errors after kernel launches
myKernel<<<grid, block>>>(d_data, N);

// Method 1: Check launch error immediately
cudaError_t launchErr = cudaGetLastError();
if (launchErr != cudaSuccess) {
    printf("Launch error: %s\n", cudaGetErrorString(launchErr));
}

// Method 2: Wait and check execution error
cudaError_t execErr = cudaDeviceSynchronize();
if (execErr != cudaSuccess) {
    printf("Execution error: %s\n", cudaGetErrorString(execErr));
}
```

**Uninitialized memory:**

Device memory allocated with `cudaMalloc` is not initialized. Reading uninitialized values produces undefined behavior that can be difficult to reproduce.

```cpp
// Always initialize device memory
float* d_data;
cudaMalloc(&d_data, N * sizeof(float));
cudaMemset(d_data, 0, N * sizeof(float));  // Zero-initialize

// Or copy from initialized host memory
float* h_data = new float[N]();  // Zero-initialized
cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
```

**Out-of-bounds access:**

GPU out-of-bounds accesses may not crash immediately but corrupt data silently. Use Compute Sanitizer to detect them.

```bash
compute-sanitizer --tool memcheck ./myapp
```

---

## 25.4 Numerical Accuracy

Floating-point arithmetic on GPUs follows the IEEE 754 standard for individual operations, but differences in operation ordering, compiler optimizations, and hardware-specific behavior can produce results that differ from CPU computation.

### 25.4.1 Single vs Double Precision

| Property | Single (float/FP32) | Double (double/FP64) |
|---|---|---|
| Bits | 32 | 64 |
| Significand | 23 bits (~7 decimal digits) | 52 bits (~15 decimal digits) |
| Exponent | 8 bits | 11 bits |
| Range | ~1.2e-38 to ~3.4e+38 | ~2.2e-308 to ~1.8e+308 |
| GPU throughput | Higher (varies by CC) | Lower (varies by CC) |
| Memory usage | 4 bytes | 8 bytes |

```cpp
// Choose precision based on accuracy requirements
// Single precision: sufficient for graphics, ML inference, many simulations
// Double precision: required for scientific computing, financial modeling

// Use single precision for performance when possible
__global__ void singlePrecisionKernel(const float* data, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        result[idx] = data[idx] * 2.0f;  // 'f' suffix for float literal
    }
}

// Use double precision when accuracy demands it
__global__ void doublePrecisionKernel(const double* data, double* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        result[idx] = data[idx] * 2.0;   // No suffix = double literal
    }
}
```

### 25.4.2 IEEE 754 Compliance

CUDA GPUs are IEEE 754 compliant for individual floating-point operations of the same precision. This means:

- Addition, subtraction, multiplication, division, and square root produce correctly rounded results.
- The hardware follows the same rounding rules as compliant CPUs.

However, **compound operations may differ** from CPU results because:

1. **Floating-point addition is not associative**: `(a + b) + c != a + (b + c)` in floating-point arithmetic. Parallel reductions on the GPU sum elements in a different order than a sequential CPU loop, producing different rounding.

```cpp
// CPU: sequential sum
float cpuSum = 0.0f;
for (int i = 0; i < N; i++) {
    cpuSum += data[i];  // Sequential accumulation order
}

// GPU: parallel reduction (different accumulation order)
// This will produce a different result even with identical inputs
__global__ void parallelReduce(const float* data, float* result, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < N) ? data[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];  // Different accumulation order
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(result, sdata[0]);
}
```

2. **FMA (Fused Multiply-Add) differences**: GPUs heavily use FMA operations (`a * b + c`) which perform the multiply and add in a single step with only one rounding. This is more accurate than separate multiply and add (which round twice), but produces different results from CPUs that do not use FMA.

```cpp
// FMA: more accurate (one rounding)
// result = round(a * b + c)           -- FMA path

// Separate: less accurate (two roundings)
// result = round(round(a * b) + c)    -- non-FMA path

// Control FMA behavior with compiler flags
// -fma=true   (default): use FMA for better accuracy
// -fma=false: disable FMA for bit-exact CPU matching

// Explicit FMA control in code
float fma_result  = fmaf(a, b, c);       // Force FMA
float sep_result  = a * b + c;            // May or may not use FMA depending on flags
```

3. **Compiler optimizations**: Flags like `-use_fast_math` relax IEEE compliance for speed.

### 25.4.3 Float Literal Suffixes

A common source of precision errors is inadvertently promoting expressions to double precision by omitting the `f` suffix on float literals. In CUDA, this can cause unexpected double-precision arithmetic inside single-precision kernels.

```cpp
// WRONG: literal 0.5 is double, promotes entire expression to double
float half_val = x * 0.5;  // Mixed precision: float * double -> double -> float

// CORRECT: use 'f' suffix for float literals
float half_val = x * 0.5f;  // float * float -> float

// Other common float literals
float pi     = 3.14159265f;
float inv3   = 1.0f / 3.0f;
float thresh = 1e-6f;

// Double literals for double-precision code
double dpi   = 3.141592653589793;  // No suffix needed (double by default)
```

### 25.4.4 Kahan Summation for Accurate Reductions

When the order of summation cannot be controlled (as in parallel reduction), Kahan summation compensates for accumulated rounding error.

```cpp
// CPU: Kahan summation
float kahanSum(const float* data, int N) {
    float sum = 0.0f;
    float compensation = 0.0f;
    for (int i = 0; i < N; i++) {
        float y = data[i] - compensation;
        float t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    return sum;
}

// GPU: Kahan summation in shared memory reduction
__global__ void kahanReduce(const float* data, float* result, int N) {
    __shared__ float s_sum[256];
    __shared__ float s_comp[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    s_sum[tid] = 0.0f;
    s_comp[tid] = 0.0f;
    __syncthreads();

    // Load and apply Kahan compensation per thread
    float val = (idx < N) ? data[idx] : 0.0f;
    float y = val - s_comp[tid];
    float t = s_sum[tid] + y;
    s_comp[tid] = (t - s_sum[tid]) - y;
    s_sum[tid] = t;
    __syncthreads();

    // Standard reduction on compensated sums
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(result, s_sum[0]);
}
```

### 25.4.5 Compiler Flags for Numerical Accuracy

| Flag | Effect | Accuracy Impact |
|------|--------|----------------|
| `-fma=true` (default) | Enable FMA operations | More accurate (one rounding vs two) |
| `-fma=false` | Disable FMA | Less accurate, matches non-FMA CPUs |
| `-ftz=true` | Flush denormals to zero | Less accurate for very small values, faster |
| `-ftz=false` (default) | Preserve denormals | IEEE compliant, slower |
| `-prec-div=true` (default) | IEEE-accurate division | Full accuracy, slower |
| `-prec-div=false` | Approximate division | Less accurate, faster |
| `-prec-sqrt=true` (default) | IEEE-accurate sqrt | Full accuracy, slower |
| `-prec-sqrt=false` | Approximate sqrt | Less accurate, faster |
| `-use_fast_math` | All of: ftz, imprecise div/sqrt, fast intrinsics | Least accurate, fastest |

```bash
# Maximum accuracy
nvcc -fma=true -ftz=false -prec-div=true -prec-sqrt=true myapp.cu

# Maximum speed (reduced accuracy)
nvcc -use_fast_math myapp.cu

# Balanced: FMA on, denormal flush, accurate div/sqrt
nvcc -fma=true -ftz=true -prec-div=true -prec-sqrt=true myapp.cu
```

---

## Summary

| Topic | Key Takeaway |
|-------|-------------|
| **APOD** | Iterative cycle: Assess hotspots, Parallelize with libraries first, Optimize with profiling, Deploy early |
| **Amdahl's Law** | `S = 1/((1-P) + P/N)`; serial fraction bounds maximum speedup |
| **Gustafson's Law** | `S = N*P + (1-P)`; scaled problem size yields more optimistic speedup |
| **Verification** | Compare against CPU reference; use `__host__ __device__` for shared logic |
| **Debugging** | Use CUDA-GDB, Nsight Systems/Compute, Compute Sanitizer; enable `CUDA_LAUNCH_BLOCKING` for errors |
| **Numerical accuracy** | IEEE 754 compliant per-operation; differences arise from FMA, non-associativity, compiler flags |
| **Float literals** | Always use `f` suffix (`0.5f`) in single-precision code to avoid implicit double promotion |
