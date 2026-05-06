# 28. Instruction Optimizations

This document covers best practices for optimizing instruction-level execution in CUDA kernels, including arithmetic throughput considerations, math library usage, compiler optimization flags, control flow optimization, and warp-level matrix functions for tensor core utilization.

---

## Table of Contents

1. [Arithmetic Instruction Throughput](#281-arithmetic-instruction-throughput)
2. [Math Library Tips](#282-math-library-tips)
3. [Exponentiation Formulas](#283-exponentiation-formulas)
4. [Compiler Flags](#284-compiler-flags)
5. [Control Flow Optimization](#285-control-flow-optimization)
6. [Warp Matrix Functions (Tensor Cores)](#286-warp-matrix-functions-tensor-cores)

---

## 28.1 Arithmetic Instruction Throughput

Understanding the throughput of different arithmetic instructions across GPU architectures is essential for identifying compute bottlenecks and selecting optimal operations. Throughput is measured in operations per clock cycle per SM.

### 28.1.1 Throughput Table (Operations per SM per Clock)

| Instruction | CC 7.5 | CC 8.0 | CC 8.6 | CC 8.9 | CC 9.0 | CC 10.0 |
|---|---|---|---|---|---|---|
| **FP16 x2 (half2)** | 64 | 128 | 256 | 256 | 128 | 64 |
| **FP32 (single)** | 64 | 64 | 128 | 128 | 128 | 128 |
| **FP32 (tensor, TF32)** | N/A | 256 | 256 | 512 | 512 | 512 |
| **FP64 (double)** | 2 | 32 | 2 | 2 | 64 | 64 |
| **INT32 add/logical** | 128 | 128 | 128 | 128 | 128 | 128 |
| **INT32 mul** | 128 | 128 | 128 | 128 | 128 | 128 |
| **INT32 div/mod** | Lower | Lower | Lower | Lower | Lower | Lower |
| **INT64** | 1 | 16 | 16 | 16 | 32 | 32 |
| **Warp shuffle** | 16 | 32 | 32 | 32 | 32 | 32 |
| **SFU (sin, cos, etc.)** | 8 | 8 | 16 | 16 | 16 | 16 |
| **FP64 (tensor)** | N/A | 64 | N/A | N/A | 128 | 128 |
| **FP16 (tensor)** | 256 | 512 | 512 | 1024 | 1024 | 1024 |
| **INT8 (tensor)** | 256 | 512 | 1024 | 1024 | 1024 | 1024 |
| **FP8 (tensor)** | N/A | N/A | N/A | 2048 | 2048 | 2048 |

**Key observations:**

- FP16 operations provide 2x throughput over FP32 on most architectures, and even more through tensor cores.
- FP64 throughput varies dramatically: only 2 ops/clock on consumer GPUs (CC 8.6/8.9) vs 32-64 on data center GPUs (CC 8.0/9.0).
- INT32 add/mul is 128 ops/clock across all architectures -- one of the highest throughput operations.
- INT64 operations are significantly slower than INT32; avoid 64-bit integers in performance-critical code when possible.
- Special function unit (SFU) operations (sin, cos, exp, log, rcp, sqrt) have limited throughput.

### 28.1.2 Implications for Kernel Design

```cpp
// Prefer FP16 when precision allows (2x throughput)
#include <cuda_fp16.h>
__global__ void fp16Compute(const half* a, const half* b, half* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // half2 processes two elements per instruction
        // Use half2 for even higher throughput
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

// Prefer INT32 arithmetic over INT64
__global__ void intCompute(const int* a, const int* b, int* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];   // INT32 add: 128 ops/clock
        // Avoid: (long long)c[idx] = (long long)a[idx] + (long long)b[idx];
        // INT64 add: only 1-32 ops/clock depending on architecture
    }
}

// Avoid expensive SFU operations in inner loops
__global__ void avoidSFU(const float* angle, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // BAD: sin/cos in inner loop (8-16 ops/clock)
        result[idx] = sinf(angle[idx]) + cosf(angle[idx]);

        // BETTER: Use fast math intrinsics (see Section 28.2)
        result[idx] = __sinf(angle[idx]) + __cosf(angle[idx]);

        // BEST: Precompute on CPU or use lookup table if angles are reused
    }
}
```

---

## 28.2 Math Library Tips

CUDA provides two categories of math functions: standard C library functions (e.g., `sinf`, `sqrtf`) and CUDA intrinsics (e.g., `__sinf`, `__fsqrt_rn`). The intrinsics are faster but have lower precision or narrower input ranges.

### 28.2.1 Fast Math Intrinsics

Intrinsic functions with the `__` prefix are lower precision but significantly faster than their standard counterparts. They are implemented directly in hardware or through optimized instruction sequences.

| Standard Function | Intrinsic | Speedup | Accuracy | Notes |
|---|---|---|---|---|
| `sinf(x)` | `__sinf(x)` | ~10x | ~2-3 ulp | Input range: [-pi, pi] for best accuracy |
| `cosf(x)` | `__cosf(x)` | ~10x | ~2-3 ulp | Input range: [-pi, pi] |
| `sincosf(x,&s,&c)` | `__sincosf(x,&s,&c)` | ~10x | ~2-3 ulp | Computes both sin and cos |
| `tanf(x)` | `__tanf(x)` | ~5x | ~2-3 ulp | |
| `logf(x)` | `__logf(x)` | ~3x | ~2-3 ulp | Natural log |
| `log2f(x)` | `__log2f(x)` | ~3x | ~2-3 ulp | Base-2 log |
| `log10f(x)` | `__log10f(x)` | ~3x | ~2-3 ulp | Base-10 log |
| `expf(x)` | `__expf(x)` | ~3x | ~2-3 ulp | |
| `exp2f(x)` | `__exp2f(x)` | ~3x | ~2-3 ulp | Often faster than expf |
| `exp10f(x)` | `__exp10f(x)` | ~3x | ~2-3 ulp | Often faster than expf |
| `powf(x,y)` | `__powf(x,y)` | ~3x | ~2-3 ulp | |
| `fdividef(x,y)` | `__fdividef(x,y)` | ~5x | ~2-3 ulp | Use for division in hot paths |
| `sqrtf(x)` | `__fsqrt_rn(x)` | ~2x | IEEE | IEEE rounded sqrt |
| `rsqrtf(x)` | `__frsqrt_rn(x)` | ~5x | ~1-2 ulp | 1/sqrt(x) -- very fast |
| `rcpf(x)` | `__frcp_rn(x)` | ~5x | ~1-2 ulp | 1/x |
| `fmaf(a,b,c)` | `__fmaf_rn(a,b,c)` | same | IEEE | Fused multiply-add |

```cpp
// Standard math functions (accurate, slower)
float a = sinf(x);
float b = cosf(x);
float c = expf(x);
float d = logf(x);
float e = sqrtf(x);

// Intrinsic math functions (faster, lower accuracy)
float a_fast = __sinf(x);
float b_fast = __cosf(x);
float c_fast = __expf(x);
float d_fast = __logf(x);
float e_fast = __fsqrt_rn(x);
```

### 28.2.2 Preferred Functions for Common Operations

```cpp
// 1/sqrt(x): use rsqrtf instead of 1.0f/sqrtf(x)
float val = rsqrtf(x);  // Single hardware instruction, very fast

// Cube root: use cbrtf instead of powf(x, 1.0f/3.0f)
float val = cbrtf(x);  // Faster and more accurate than pow(x, 1/3)

// sin(pi*x): use sinpif instead of sinf(x * PI)
float val = sinpif(x);  // Better accuracy for multiples of pi

// exp2/exp10: use base-2 or base-10 when possible
float val = exp2f(x);   // Often faster than expf(x * log2(e))
float val = exp10f(x);  // Often faster than expf(x * log10(e))

// Division: use fdividef for non-critical paths
float val = fdividef(numerator, denominator);  // Much faster than operator/

// Absolute value: use fabsf instead of conditional
float val = fabsf(x);   // Single instruction

// Minimum/Maximum: use fminf/fmaxf instead of conditional
float val = fmaxf(a, b);  // Single instruction, no branch
```

### 28.2.3 Integer Optimization Tips

```cpp
// Integer division by power of 2: use right shift
int quotient = value >> 2;       // value / 4 (much faster than div)
int remainder = value & 3;       // value % 4

// Integer division by constant: let compiler optimize
// Modern compilers convert integer division by constants to multiply+shift
int q = value / 10;  // Compiler generates: (value * 0xCCCCCCCD) >> 35

// Bit manipulation for power-of-2 operations
bool isPowerOf2 = (n & (n - 1)) == 0;
int nextPowerOf2 = 1 << (32 - __clz(n - 1));  // __clz: count leading zeros

// Use __popc for population count (number of set bits)
int bitsSet = __popc(value);

// Use __ffs for find first set bit (1-indexed, 0 if none)
int firstSet = __ffs(value);

// Avoid modulo with non-power-of-2 in inner loops
// BAD
for (int i = 0; i < N; i++) {
    int slot = i % TABLE_SIZE;  // Expensive division
    table[slot] += data[i];
}
// GOOD (if TABLE_SIZE can be made a power of 2)
const int TABLE_MASK = TABLE_SIZE - 1;  // TABLE_SIZE must be power of 2
for (int i = 0; i < N; i++) {
    int slot = i & TABLE_MASK;  // Cheap bitmask
    table[slot] += data[i];
}
```

---

## 28.3 Exponentiation Formulas

Many common exponentiation operations can be expressed using faster CUDA intrinsics instead of the general `powf()` function. The table below lists the optimal replacement for each case.

### 28.3.1 Fast Exponentiation Reference

| Expression | Fast Replacement | Notes |
|---|---|---|
| `x^(1/2)` | `sqrtf(x)` | Standard sqrt |
| `x^(1/3)` | `cbrtf(x)` | Cube root |
| `x^(1/4)` | `rsqrtf(rsqrtf(x))` | Two rsqrt instructions |
| `x^(-1/2)` | `rsqrtf(x)` | Single instruction |
| `x^(-1/4)` | `sqrtf(rsqrtf(x))` | rsqrt + sqrt |
| `x^(-3/4)` | `rsqrtf(x) * rsqrtf(rsqrtf(x))` | Multiple rsqrt |
| `x^2` | `x * x` | Simple multiply |
| `x^3` | `x * x * x` | Two multiplies |
| `x^(-1)` | `1.0f / x` or `__frcp_rn(x)` | Reciprocal |
| `x^(-2)` | `1.0f / (x * x)` | Multiply then reciprocal |
| `1 / sqrt(x)` | `rsqrtf(x)` | Single instruction |
| `1 / cbrt(x)` | `__frcp_rn(cbrtf(x))` | Cube root then reciprocal |
| `2^x` | `exp2f(x)` | Base-2 exponential |
| `10^x` | `exp10f(x)` | Base-10 exponential |
| `e^x` | `expf(x)` | Natural exponential |
| `log2(x)` | `log2f(x)` | Base-2 log |
| `log10(x)` | `log10f(x)` | Base-10 log |

```cpp
// Practical examples
__global__ void exponentiationExamples(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = input[idx];

    // Instead of powf(x, -0.5f)
    output[idx * 8 + 0] = rsqrtf(x);

    // Instead of powf(x, 0.25f)
    output[idx * 8 + 1] = rsqrtf(rsqrtf(x));

    // Instead of powf(x, 1.0f/3.0f)
    output[idx * 8 + 2] = cbrtf(x);

    // Instead of powf(x, -0.75f)
    float inv_sqrt_x = rsqrtf(x);
    float inv_4th_x = rsqrtf(inv_sqrt_x);
    output[idx * 8 + 3] = inv_sqrt_x * inv_4th_x;

    // Instead of powf(x, -0.25f)
    output[idx * 8 + 4] = sqrtf(rsqrtf(x));

    // Instead of powf(2.0f, x)
    output[idx * 8 + 5] = exp2f(x);

    // Instead of powf(10.0f, x)
    output[idx * 8 + 6] = exp10f(x);

    // Instead of powf(x, -2.0f)
    float x2 = x * x;
    output[idx * 8 + 7] = __frcp_rn(x2);
}
```

### 28.3.2 Half Precision Exponentiation

```cpp
#include <cuda_fp16.h>

__global__ void fp16Exponentiation(const half* input, half* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    half x = input[idx];
    float xf = __half2float(x);

    // FP16 doesn't have native exponentiation; promote to float, compute, convert back
    float result = rsqrtf(xf);  // 1/sqrt(x) in float
    output[idx] = __float2half(result);

    // For half2: process two elements at once
    // half2 x2 = reinterpret_cast<const half2*>(input)[idx / 2];
    // float2 f2 = __half22float2(x2);
    // f2.x = rsqrtf(f2.x);
    // f2.y = rsqrtf(f2.y);
    // reinterpret_cast<half2*>(output)[idx / 2] = __float22half2_rn(f2);
}
```

---

## 28.4 Compiler Flags

NVCC provides several compiler flags that control the trade-off between numerical accuracy and performance. Understanding these flags is essential for extracting maximum performance from compute-bound kernels.

### 28.4.1 Optimization Flags Reference

| Flag | Default | Effect |
|------|---------|--------|
| `-O3` | Yes (device) | Maximum optimization level |
| `-fma=true` | `true` | Enable FMA (fused multiply-add); more accurate and faster |
| `-ftz=true` | `false` | Flush denormal floats to zero; faster but loses subnormal values |
| `-prec-div=true` | `true` | IEEE-accurate division; set to `false` for faster approximate division |
| `-prec-sqrt=true` | `true` | IEEE-accurate sqrt; set to `false` for faster approximate sqrt |
| `-use_fast_math` | N/A | Implies: `-ftz=true`, `-prec-div=false`, `-prec-sqrt=false`, `-fma=true`, uses `__` intrinsics |
| `--ptxas-options=-v` | N/A | Verbose: prints register, shared memory, and stack usage per kernel |
| `--maxrregcount=N` | N/A | Global register cap per thread for all kernels |
| `-Xptxas -dlcm=cg` | N/A | Cache load as cache-global (bypass L1, use L2) |
| `-Xptxas -dlcm=ca` | N/A | Cache load as cache-all (use both L1 and L2, default) |

### 28.4.2 Detailed Flag Usage

**`-use_fast_math`:**

The most aggressive optimization flag. It enables all fast-math options simultaneously and replaces standard math functions with their intrinsic counterparts.

```bash
nvcc -use_fast_math mykernel.cu
# Equivalent to:
# nvcc -fma=true -ftz=true -prec-div=false -prec-sqrt=false mykernel.cu
# Plus: sinf -> __sinf, cosf -> __cosf, etc.
```

```cpp
// With -use_fast_math, these are automatically used:
// sinf(x)   -> __sinf(x)    (faster, less accurate)
// cosf(x)   -> __cosf(x)
// expf(x)   -> __expf(x)
// logf(x)   -> __logf(x)
// powf(x,y) -> __powf(x,y)
// etc.
```

**Warning:** `-use_fast_math` can significantly reduce numerical accuracy. Use it only when the application tolerates reduced precision (e.g., neural network inference, graphics rendering).

**`--ptxas-options=-v`:**

Essential for understanding register pressure. Print this information for every kernel to identify occupancy bottlenecks.

```bash
nvcc --ptxas-options=-v mykernel.cu
# Output:
# ptxas info:    48 bytes gmem, 8 bytes cmem[0]
# ptxas info:    Compiling entry function '_Z8myKernelPKfPfi' for 'sm_80'
# ptxas info:    Used 32 registers, 1024 bytes smem, 68 bytes cmem[0], 0 bytes lmem
```

The output shows:
- **registers**: Number of registers per thread (affects occupancy)
- **smem**: Static shared memory usage per block
- **cmem[0]**: Constant memory usage
- **lmem**: Local memory usage (register spills to global memory -- bad for performance)

**`--maxrregcount=N`:**

```bash
# Limit registers to 32 per thread for all kernels
nvcc --maxrregcount=32 mykernel.cu

# Trade-off: lower register count -> higher occupancy -> more register spills to local memory
# Profile to find the sweet spot
```

### 28.4.3 Selective Fast Math

For finer control, apply fast-math flags to specific functions rather than globally.

```cpp
// Method 1: Use intrinsic functions explicitly in hot code paths
__global__ void hybridKernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Use intrinsic in the hot inner loop
    float val = __sinf(input[idx]);  // Fast, ~2-3 ulp error
    output[idx] = val;
}

// Method 2: Use #pragma to control precision per function
#pragma hd_warning_disable  // Suppress warnings for __host__ __device__ functions

// Method 3: Compile different .cu files with different flags
// precise_math.cu: nvcc -prec-div=true -prec-sqrt=true ...
// fast_math.cu:    nvcc -use_fast_math ...
```

### 28.4.4 Architecture-Specific Compilation

```bash
# Always specify the target architecture for optimal code generation
nvcc -arch=sm_80 mykernel.cu                # For A100
nvcc -arch=sm_90 mykernel.cu                # For H100
nvcc -arch=native mykernel.cu               # Auto-detect current GPU

# Fat binary for multiple architectures
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_90,code=compute_90 \
     mykernel.cu -o myapp

# With optimization flags
nvcc -O3 -arch=sm_80 --ptxas-options=-v \
     -use_fast_math \
     mykernel.cu -o myapp
```

---

## 28.5 Control Flow Optimization

Efficient control flow on GPUs requires minimizing warp divergence, where threads within a warp take different execution paths. When divergence occurs, the hardware serializes execution of each path, reducing effective throughput.

### 28.5.1 Understanding Warp Divergence

```
Warp with divergence:
Thread:   0  1  2  3  4  5  6  7  ... 31
Condition: T  T  F  T  F  F  T  T  ... T

Path A (condition=true):  0,1,3,6,7,...31 execute
Path B (condition=false): 2,4,5,... execute
Total time = time(A) + time(B)  (serialized)

Warp without divergence:
Thread:   0  1  2  3  4  5  6  7  ... 31
Condition: T  T  T  T  T  T  T  T  ... T
All threads execute Path A only
Total time = time(A)
```

### 28.5.2 Avoiding Divergence

```cpp
// BAD: Divergent conditional in inner loop
__global__ void divergentKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    for (int i = 0; i < 100; i++) {
        if (data[idx] > 0.0f) {        // May diverge within warp
            data[idx] = sqrtf(data[idx]);
        } else {
            data[idx] = -sqrtf(-data[idx]);
        }
    }
}

// GOOD: Use branch predication or math to eliminate the branch
__global__ void predicatedKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    for (int i = 0; i < 100; i++) {
        float val = data[idx];
        float sign = copysignf(1.0f, val);  // +1 or -1
        data[idx] = sign * sqrtf(fabsf(val));
        // No branch needed -- all threads execute same instructions
    }
}

// GOOD: Sort data so threads in a warp take the same path
// Use Thrust or CUB to sort by predicate before processing
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

void sortForCoherentAccess(float* d_data, int N) {
    thrust::device_ptr<float> dev_ptr(d_data);
    // Sort so that all positive values come first, then negative
    // This reduces warp divergence in subsequent kernel
}
```

### 28.5.3 Branch Predication

For short conditional blocks, the compiler automatically uses predication instead of actual branching. Predication executes both paths but only writes results for the correct path, avoiding the cost of branch misprediction.

```cpp
// Short conditionals are auto-predicated by the compiler
__global__ void autoPredication(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // This short if-else is predicated by the compiler
    // Both paths execute, but results are masked
    float val = data[idx];
    float result = (val > 0.0f) ? sqrtf(val) : 0.0f;  // Predicated
    data[idx] = result;
}

// For longer conditionals, predication is NOT used (branches instead)
__global__ void longBranch(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // This long if-else will use actual branching
    if (data[idx] > threshold) {
        // Many operations...
        for (int i = 0; i < 100; i++) { /* ... */ }
    } else {
        // Many operations...
        for (int i = 0; i < 50; i++) { /* ... */ }
    }
    // Warp divergence is costly here
}
```

### 28.5.4 Loop Unrolling

```cpp
// #pragma unroll: hint to the compiler to unroll loops
// Reduces loop overhead and enables better instruction scheduling

// Full unroll (compiler decides the factor)
#pragma unroll
for (int i = 0; i < 4; i++) {
    sum += data[offset + i];
}

// Specify unroll factor
#pragma unroll 4
for (int i = 0; i < 32; i++) {
    sum += data[offset + i];
}
// Generates 8 iterations of 4 unrolled operations each

// Disable unrolling (prevent code bloat for large loops)
#pragma unroll 1
for (int i = 0; i < 1024; i++) {
    // Large loop -- don't unroll
}

// Unroll with template parameter for compile-time loop count
template <int UNROLL_FACTOR>
__global__ void unrolledKernel(const float* data, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) {
        int accessIdx = idx * UNROLL_FACTOR + i;
        if (accessIdx < N) {
            sum += data[accessIdx];
        }
    }
    result[idx] = sum;
}

// Instantiate with different unroll factors
template __global__ void unrolledKernel<4>(const float*, float*, int);
template __global__ void unrolledKernel<8>(const float*, float*, int);
template __global__ void unrolledKernel<16>(const float*, float*, int);
```

### 28.5.5 Warp-Level Synchronization

On Volta (CC 7.0) and later, threads within a warp can diverge and reconverge independently. Use `__syncwarp()` to ensure all threads in a warp have reached a synchronization point before proceeding.

```cpp
// Warp-level synchronization (lighter than __syncthreads)
__global__ void warpSyncExample(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;

    float val = (idx < N) ? data[idx] : 0.0f;

    // Warp shuffle: exchange data between threads in the same warp
    float left  = __shfl_up_sync(0xFFFFFFFF, val, 1);   // Get value from lane-1
    float right = __shfl_down_sync(0xFFFFFFFF, val, 1);  // Get value from lane+1
    float any   = __shfl_sync(0xFFFFFFFF, val, 5);       // Get value from lane 5

    // Warp reduction using shuffles (no shared memory needed)
    float sum = val;
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    // Lane 0 now has the warp-wide sum
    if (lane == 0) {
        data[blockIdx.x * blockDim.x / 32 + threadIdx.x / 32] = sum;
    }

    // Explicit warp sync (needed when warps may have diverged)
    __syncwarp();
}

// Warp vote functions
__global__ void warpVote(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;

    float val = (idx < N) ? data[idx] : 0.0f;

    // all: true if predicate is true for ALL threads in warp
    int allPositive = __all_sync(0xFFFFFFFF, val > 0.0f);

    // any: true if predicate is true for ANY thread in warp
    int anyNegative = __any_sync(0xFFFFFFFF, val < 0.0f);

    // ballot: returns a 32-bit mask where bit i = 1 if predicate is true for lane i
    unsigned int mask = __ballot_sync(0xFFFFFFFF, val > threshold);

    if (lane == 0) {
        printf("All positive: %d, Any negative: %d, Mask: 0x%08X\n",
               allPositive, anyNegative, mask);
    }
}
```

---

## 28.6 Warp Matrix Functions (Tensor Cores)

Tensor cores provide hardware-accelerated matrix multiply-accumulate (MMA) operations that are dramatically faster than scalar CUDA core implementations. The Warp Matrix Functions (WMMA) API provides a C++ interface for accessing tensor cores at the warp level.

### 28.6.1 WMMA Basics

WMMA operates at the warp level: all 32 threads in a warp cooperate to compute a matrix tile. The API manages data distribution across lanes transparently.

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// WMMA tile sizes for different data types:
// FP16: 16x16x16, 32x8x16, 8x32x16
// INT8: 16x16x16, 32x8x16, 8x32x16
// TF32: 16x16x8 (CC 8.0+)
// FP64: 8x8x4 (CC 8.0+)
// FP8:  16x16x32 (CC 8.9+)
```

### 28.6.2 FP16 Matrix Multiply with WMMA

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// Tile dimensions
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void wmmaMatMul(
    const half* A, const half* B, float* C,
    int M, int N, int K)
{
    // Leading dimensions (assume row-major)
    int lda = K;  // A is M x K
    int ldb = N;  // B is K x N
    int ldc = N;  // C is M x N

    // Warp-level tile coordinates
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y) / warpSize;

    // Declare WMMA fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator to zero
    fill_fragment(c_frag, 0.0f);

    // Loop over K dimension in WMMA_K tiles
    for (int k = 0; k < K; k += WMMA_K) {
        // Bounds check
        if (warpM * WMMA_M < M && warpN * WMMA_N < N && k + WMMA_K <= K) {
            // Load A tile: M x K (row-major)
            load_matrix_sync(a_frag, A + warpM * WMMA_M * lda + k, lda);

            // Load B tile: K x N (column-major layout for optimal access)
            load_matrix_sync(b_frag, B + k + warpN * WMMA_N * K, ldb);

            // Matrix multiply-accumulate: C += A * B
            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // Store C tile to global memory
    if (warpM * WMMA_M < M && warpN * WMMA_N < N) {
        store_matrix_sync(C + warpM * WMMA_M * ldc + warpN * WMMA_N,
                          c_frag, ldc, mem_row_major);
    }
}

// Launch configuration
dim3 blockDim(128, 4);  // 4 warps in x, 4 in y -> 512 threads
dim3 gridDim(
    (M + WMMA_M * (blockDim.x / warpSize) - 1) / (WMMA_M * (blockDim.x / warpSize)),
    (N + WMMA_N * (blockDim.y / warpSize) - 1) / (WMMA_N * (blockDim.y / warpSize))
);
wmmaMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
```

### 28.6.3 TF32 Matrix Multiply (CC 8.0+)

```cpp
// TF32: same range as FP32 (8-bit exponent) but reduced mantissa (10 bits)
// Provides FP32-level dynamic range with FP16-like throughput

__global__ void wmmaTF32MatMul(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    // TF32 uses 16x16x8 tile size
    fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
    fragment<matrix_b, 16, 16, 8, precision::tf32, col_major> b_frag;
    fragment<accumulator, 16, 16, 8, float> c_frag;

    fill_fragment(c_frag, 0.0f);

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y) / warpSize;

    for (int k = 0; k < K; k += 8) {
        load_matrix_sync(a_frag, A + warpM * 16 * K + k, K);
        load_matrix_sync(b_frag, B + k * N + warpN * 16, N);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    store_matrix_sync(C + warpM * 16 * N + warpN * 16,
                      c_frag, N, mem_row_major);
}
```

### 28.6.4 FP64 Matrix Multiply (CC 8.0+)

```cpp
// FP64 tensor cores on data center GPUs (A100, H100)
__global__ void wmmaFP64MatMul(
    const double* A, const double* B, double* C,
    int M, int N, int K)
{
    // FP64 uses 8x8x4 tile size
    fragment<matrix_a, 8, 8, 4, double, row_major> a_frag;
    fragment<matrix_b, 8, 8, 4, double, col_major> b_frag;
    fragment<accumulator, 8, 8, 4, double> c_frag;

    fill_fragment(c_frag, 0.0);

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y) / warpSize;

    for (int k = 0; k < K; k += 4) {
        load_matrix_sync(a_frag, A + warpM * 8 * K + k, K);
        load_matrix_sync(b_frag, B + k * N + warpN * 8, N);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    store_matrix_sync(C + warpM * 8 * N + warpN * 8,
                      c_frag, N, mem_row_major);
}
```

### 28.6.5 Shared Memory Tiled WMMA

```cpp
// Optimal: tile matrix multiply with shared memory + WMMA
#define TILE_M 128
#define TILE_N 128
#define TILE_K 32

__global__ void wmmaTiledMatMul(
    const half* A, const half* B, float* C,
    int M, int N, int K)
{
    // Shared memory tiles
    __shared__ half sA[TILE_M][TILE_K];
    __shared__ half sB[TILE_K][TILE_N];

    int warpId = (threadIdx.x + threadIdx.y * blockDim.x) / warpSize;
    int laneId = (threadIdx.x + threadIdx.y * blockDim.x) % warpSize;

    // Warp tile coordinates within the block
    int warpM = warpId / (TILE_N / WMMA_N);  // Number of warps in M direction
    int warpN = warpId % (TILE_N / WMMA_N);

    // Global tile coordinates
    int blockM = blockIdx.y * TILE_M;
    int blockN = blockIdx.x * TILE_N;

    // WMMA fragments
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    fill_fragment(c_frag, 0.0f);

    // Loop over K in tiles
    for (int kTile = 0; kTile < K; kTile += TILE_K) {
        // Cooperative load into shared memory
        // Each thread loads multiple elements to fill the tile
        for (int i = threadIdx.y; i < TILE_M; i += blockDim.y) {
            for (int j = threadIdx.x; j < TILE_K; j += blockDim.x) {
                int globalRow = blockM + i;
                int globalCol = kTile + j;
                sA[i][j] = (globalRow < M && globalCol < K)
                          ? A[globalRow * K + globalCol] : __float2half(0.0f);
            }
        }

        for (int i = threadIdx.y; i < TILE_K; i += blockDim.y) {
            for (int j = threadIdx.x; j < TILE_N; j += blockDim.x) {
                int globalRow = kTile + i;
                int globalCol = blockN + j;
                sB[i][j] = (globalRow < K && globalCol < N)
                          ? B[globalRow * N + globalCol] : __float2half(0.0f);
            }
        }

        __syncthreads();

        // WMMA accumulate over the shared memory tile
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;

        for (int k = 0; k < TILE_K; k += WMMA_K) {
            int aRow = warpM * WMMA_M;
            int bCol = warpN * WMMA_N;

            load_matrix_sync(a_frag, &sA[aRow][k], TILE_K);
            load_matrix_sync(b_frag, &sB[k][bCol], TILE_N);
            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
    }

    // Store result
    int cRow = blockM + warpM * WMMA_M;
    int cCol = blockN + warpN * WMMA_N;
    if (cRow < M && cCol < N) {
        store_matrix_sync(&C[cRow * N + cCol], c_frag, N, mem_row_major);
    }
}
```

### 28.6.6 PTX-Level MMA (Lower-Level Control)

For maximum performance, use PTX inline assembly for MMA operations. This provides access to all tile sizes and data types, including those not exposed by the WMMA API.

```cpp
// Hopper (CC 9.0+) MMA: 16x8x16 FP16 -> FP32
__global__ void ptxMMAF16(
    const half* A, const half* B, float* C, int K)
{
    // Each warp computes a 16x8 output tile from 16xK and Kx8 inputs
    uint32_t a_frag[4];  // 4 registers for 16x16 FP16 = 128 bits in 4x32-bit regs
    uint32_t b_frag[2];  // 2 registers for 16x8 FP16 = 64 bits
    uint32_t c_frag[4] = {0, 0, 0, 0};  // FP32 accumulators

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y) / 32;

    for (int k = 0; k < K; k += 16) {
        // Load A fragment: 16x16 block of FP16 starting at row warpM*16, col k
        // Each lane loads specific elements based on its lane ID
        const half* a_ptr = A + warpM * 16 * K + k;
        const half* b_ptr = B + k * 8 + warpN * 8;

        // Load A: 128 bytes = 64 FP16 values, distributed across 32 lanes
        // Each lane loads 2 FP16 values (4 bytes) via ldmatrix instruction
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16"
            "{%0, %1, %2, %3}, [%4];"
            : "=r"(a_frag[0]), "=r"(a_frag[1]),
              "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(a_ptr));

        // Load B: 64 bytes = 32 FP16 values, distributed across 32 lanes
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16"
            "{%0, %1}, [%2];"
            : "=r"(b_frag[0]), "=r"(b_frag[1])
            : "r"(b_ptr));

        // MMA: C += A * B
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

    // Store results
    int row = warpM * 16;
    int col = warpN * 8;
    for (int i = 0; i < 4; i++) {
        int r = row + (i / 2) * 8 + (threadIdx.x % 4) * 2;
        int c = col + (i % 2) * 4 + (threadIdx.x / 4);
        if (r < /* M */ && c < /* N */) {
            C[r * /* N */ + c] = *((float*)&c_frag[i]);
        }
    }
}
```

### 28.6.7 WMMA API Quick Reference

| Operation | Function | Description |
|---|---|---|
| Load matrix A | `load_matrix_sync(a_frag, ptr, stride)` | Load from global or shared memory |
| Load matrix B | `load_matrix_sync(b_frag, ptr, stride)` | Load from global or shared memory |
| Initialize accumulator | `fill_fragment(c_frag, value)` | Set all elements to a value |
| Multiply-accumulate | `mma_sync(c_frag, a_frag, b_frag, c_frag)` | C += A * B |
| Store result | `store_matrix_sync(ptr, c_frag, stride, layout)` | Store to global or shared memory |
| Access fragment elements | `a_frag.x[i]` | Direct element access (fragments are structs with `x[]` array) |

### 28.6.8 cuBLAS Integration for Tensor Cores

For most applications, using cuBLAS with tensor cores enabled is simpler and often as performant as custom WMMA code.

```cpp
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

// Enable tensor cores
cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
// Or for maximum tensor core utilization (may reduce accuracy):
cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

// FP16 GEMM with FP32 accumulation
half alpha = __float2half(1.0f);
half beta  = __float2half(0.0f);
cublasGemmEx(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    M, N, K,
    &alpha,
    d_A, CUDA_R_16F, K,     // A: MxK, FP16, leading dim K
    d_B, CUDA_R_16F, N,     // B: KxN, FP16, leading dim N
    &beta,
    d_C, CUDA_R_16F, N,     // C: MxN, FP16, leading dim N
    CUBLAS_COMPUTE_32F,      // Compute type: FP32 accumulation
    CUBLAS_GEMM_DEFAULT_TENSOR_OP  // Use tensor cores
);

// FP32 GEMM with TF32 tensor cores
float alpha_f = 1.0f;
float beta_f  = 0.0f;
cublasGemmEx(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    M, N, K,
    &alpha_f,
    d_A, CUDA_R_32F, K,
    d_B, CUDA_R_32F, N,
    &beta_f,
    d_C, CUDA_R_32F, N,
    CUBLAS_COMPUTE_32F_FAST_TF32,  // Use TF32 tensor cores for FP32 input
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
);

cublasDestroy(handle);
```

---

## Summary

| Optimization | Key Takeaway |
|---|---|
| **Instruction throughput** | FP16 is 2x FP32; INT64 is 1/4 to 1/128 of INT32; SFU ops are slow |
| **Math intrinsics** | `__sinf`, `rsqrtf`, `cbrtf`, `sinpif`, `exp2f` are 3-10x faster than standard functions |
| **Exponentiation** | `rsqrtf(x)` for x^(-1/2), `cbrtf(x)` for x^(1/3), avoid `powf()` |
| **Compiler flags** | `-use_fast_math` for speed, `--ptxas-options=-v` for register info |
| **Register control** | `__launch_bounds__` and `--maxrregcount` to manage occupancy |
| **Warp divergence** | Use predication, sorting, or math tricks to avoid branches in warps |
| **Loop unrolling** | `#pragma unroll` for small fixed-count loops |
| **Warp shuffle** | `__shfl_sync`, `__all_sync`, `__any_sync`, `__ballot_sync` for warp-level operations |
| **Tensor cores (WMMA)** | `load_matrix_sync`, `mma_sync`, `store_matrix_sync` for hardware-accelerated matrix ops |
| **cuBLAS tensor cores** | Use `cublasGemmEx` with `CUBLAS_GEMM_DEFAULT_TENSOR_OP` for production GEMM |
