# CUTLASS - Chapter 28: Profiler

This reference covers the CUTLASS Profiler, a command-line tool for testing, profiling, and benchmarking CUTLASS kernels. The profiler supports GEMM, sparse GEMM, and convolution operations across all supported architectures and data types.

---

## 28.1 Overview

The CUTLASS Profiler (`cutlass_profiler`) is a comprehensive performance measurement tool built into the CUTLASS project. It serves multiple purposes:

- **Functional verification**: Validate CUTLASS kernels against reference implementations.
- **Performance benchmarking**: Measure kernel runtime, throughput (GFLOPS), and memory bandwidth utilization.
- **Kernel enumeration**: Discover all available kernel configurations for a given operation.
- **Comparison with vendor libraries**: Benchmark CUTLASS kernels against cuBLAS and cuDNN.
- **Architecture exploration**: Test kernels across different GPU architectures (Volta through Blackwell).

The profiler generates consistent, reproducible results by controlling warmup iterations, measurement iterations, and GPU synchronization.

---

## 28.2 Building the Profiler

### 28.2.1 Standard Build

```bash
# Clone CUTLASS
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass

# Build with profiler enabled (default)
mkdir build && cd build
cmake .. -DCUTLASS_ENABLE_TESTS=ON
make cutlass_profiler -j$(nproc)
```

### 28.2.2 Build Options

```bash
# Enable specific architectures
cmake .. \
  -DCUTLASS_NVCC_ARCHS="80;90;100" \
  -DCUTLASS_ENABLE_TESTS=ON

# Disable certain operations for faster build
cmake .. \
  -DCUTLASS_LIBRARY_KERNELS=cutlass_simt*gemm* \
  -DCUTLASS_UNORDERED_TEST_SETS=ON

# Build only the profiler (not all examples)
make cutlass_profiler -j$(nproc)
```

### 28.2.3 Hopper-Specific Build

```bash
# Build with SM90 (Hopper) support including TMA and warp specialization
cmake .. \
  -DCUTLASS_NVCC_ARCHS="90" \
  -DENABLE_CUBLAS=ON \
  -DENABLE_CUDNN=ON

make cutlass_profiler -j$(nproc)
```

### 28.2.4 Blackwell-Specific Build

```bash
# Build with SM100 (Blackwell) support
cmake .. \
  -DCUTLASS_NVCC_ARCHS="100;101;103;120" \
  -DENABLE_CUBLAS=ON

make cutlass_profiler -j$(nproc)
```

After building, the profiler binary is located at:

```bash
./tools/library/bin/cutlass_profiler    # Release build
./bin/cutlass_profiler                  # Alternative location
```

---

## 28.3 Execution Modes

The profiler supports four primary execution modes:

### 28.3.1 Profile Mode (Default)

Executes the kernel and measures performance:

```bash
./cutlass_profiler --kernels=cutlass_simt_sgemm --m=1024 --n=1024 --k=1024
```

This runs the matching kernels, warms up the GPU, and then measures performance over multiple iterations.

### 28.3.2 Dry Run Mode

Validates arguments and shows what would be executed without actually running kernels:

```bash
./cutlass_profiler --mode=dry_run --kernels=cutlass_simt_sgemm
```

Output shows all kernel configurations that match the filter, along with their problem sizes and parameters.

### 28.3.3 Enumerate Mode

Lists all available kernel configurations without running them:

```bash
./cutlass_profiler --mode=enumerate --kernels=cutlass_simt_sgemm
```

This is useful for discovering available kernels and their parameterizations before running benchmarks.

### 28.3.4 Trace Mode

Produces a detailed trace of each kernel execution:

```bash
./cutlass_profiler --mode=trace --kernels=cutlass_simt_sgemm --m=512 --n=512 --k=512
```

Trace mode outputs per-iteration timing data, enabling analysis of variance in execution times.

---

## 28.4 Command-Line Options

### 28.4.1 Operation Selection

The `--operation` flag selects which type of operation to profile:

| Flag Value | Description |
|------------|-------------|
| `gemm` | Dense General Matrix-Matrix Multiply |
| `sparse_gem` | Sparse GEMM (2:4 structured sparsity) |
| `conv2d` | 2D Convolution |
| `conv3d` | 3D Convolution |

```bash
# Profile GEMM operations
./cutlass_profiler --operation=gemm --kernels=cutlass_simt_sgemm

# Profile sparse GEMM
./cutlass_profiler --operation=sparse_gem --kernels=cutlass_sm80_sp*sparse

# Profile Conv2D
./cutlass_profiler --operation=conv2d --kernels=cutlass_simt_conv2d

# Profile Conv3D
./cutlass_profiler --operation=conv3d --kernels=cutlass_simt_conv3d
```

### 28.4.2 Kernel Filter Options

The `--kernels` flag supports glob-style pattern matching:

```bash
# Exact kernel name
./cutlass_profiler --kernels=cutlass_simt_sgemm_nn

# Wildcard patterns
./cutlass_profiler --kernels=cutlass_sm80*e*tensor_op*gemm*

# Multiple patterns (comma-separated)
./cutlass_profiler --kernels=cutlass_simt_sgemm*,cutlass_simt_dgemm*

# Regex-style patterns
./cutlass_profiler --kernels=cutlass_sm80.*tf32.*gemm
```

### 28.4.3 Data Type Filters

Filter kernels by data types:

```bash
# Float (SGEMM)
./cutlass_profiler --kernels=cutlass_simt_sgemm --A_type=f32

# Half precision (HGEMM)
./cutlass_profiler --kernels=cutlass_sm80*hgemm --A_type=f16

# BF16
./cutlass_profiler --kernels=cutlass_sm80*bf16 --A_type=bf16

# TF32
./cutlass_profiler --kernels=cutlass_sm80*tf32 --A_type=tf32

# INT8
./cutlass_profiler --kernels=cutlass_sm80*i8 --A_type=s8

# FP8 (E4M3, E5M2)
./cutlass_profiler --kernels=cutlass_sm90*e4m3 --A_type=e4m3
./cutlass_profiler --kernels=cutlass_sm90*e5m2 --A_type=e5m2
```

### 28.4.4 Layout Filters

Filter by matrix/tensor layout:

```bash
# Row-major A, Column-major B (NN layout)
./cutlass_profiler --kernels=cutlass_simt_sgemm*nn

# All layout combinations
./cutlass_profiler --kernels=cutlass_simt_sgemm*tn

# Column-major output
./cutlass_profiler --kernels=cutlass_simt_sgemm* --layout_C=column
```

### 28.4.5 Problem Size Options

GEMM problem sizes:

```bash
# Single problem size
./cutlass_profiler --m=1024 --n=1024 --k=1024

# Range of problem sizes
./cutlass_profiler --m=128:1024:128 --n=1024 --k=512

# Specific problem sizes (comma-separated)
./cutlass_profiler --m=256,512,1024,2048 --n=256,512,1024,2048 --k=128,256,512

# Batched GEMM
./cutlass_profiler --batch_count=16 --m=512 --n=512 --k=512
```

Convolution problem sizes:

```bash
# Conv2D problem configuration
./cutlass_profiler --operation=conv2d \
  --n=32 --h=56 --w=56 --c=64 \
  --k=64 --r=3 --s=3 \
  --pad_h=1 --pad_w=1 \
  --stride_h=1 --stride_w=1 \
  --dilation_h=1 --dilation_w=1

# Conv3D problem configuration
./cutlass_profiler --operation=conv3d \
  --n=8 --d=32 --h=32 --w=32 --c=64 \
  --k=64 --t=3 --r=3 --s=3 \
  --pad_d=1 --pad_h=1 --pad_w=1
```

### 28.4.6 Verification Options

Control correctness verification:

```bash
# Enable verification (default for supported operations)
./cutlass_profiler --verification-enabled=true

# Disable verification (faster profiling)
./cutlass_profiler --verification-enabled=false

# Set verification tolerance
./cutlass_profiler --epsilon=0.001    # Relative error tolerance
./cutlass_profiler --nonzero-floor=1.0e-6  # Minimum absolute value

# Number of verification iterations
./cutlass_profiler --verification-iters=3

# Save workspace for debugging
./cutlass_profiler --save-workspace=true
```

### 28.4.7 Profiling Options

Control measurement behavior:

```bash
# Number of warmup iterations
./cutlass_profiler --warmup-iterations=10

# Number of profiling iterations
./cutlass_profiler --profiling-iterations=100

# Sleep between kernels (ms) to allow GPU to cool
./cutlass_profiler --sleep-duration=50

# Provision for sleep between profiling runs
./cutlass_profiler --profiling-enabled=true
```

### 28.4.8 Output Options

Control output format and destination:

```bash
# Output to CSV file
./cutlass_profiler --output=default.csv

# Append to existing file
./cutlass_profiler --output=results.csv --append=true

# JSON output
./cutlass_profiler --output=default.json

# Human-readable console output (default)
./cutlass_profiler --output=/dev/stdout
```

---

## 28.5 Supported Operations

### 28.5.1 GEMM Operations

The profiler supports all GEMM variants:

| Operation | Description | Key Parameters |
|-----------|-------------|----------------|
| SGEMM | FP32 GEMM | `--A_type=f32` |
| DGEMM | FP64 GEMM | `--A_type=f64` |
| HGEMM | FP16 GEMM | `--A_type=f16` |
| IGEMM | INT8 GEMM | `--A_type=s8` |
| HGEMM BF16 | BF16 GEMM | `--A_type=bf16` |
| TF32 GEMM | TF32 Tensor Core GEMM | `--A_type=tf32` |
| FP8 E4M3 | FP8 E4M3 GEMM | `--A_type=e4m3` |
| FP8 E5M2 | FP8 E5M2 GEMM | `--A_type=e5m2` |
| Mixed precision | Mixed A/B/C/D types | `--A_type=f16 --accum=f32` |
| Batched GEMM | Batched matrix multiply | `--batch_count=N` |
| Split-K GEMM | Split-K parallel reduction | `--split_k_slices=N` |

### 28.5.2 Sparse GEMM Operations

```bash
# Profile sparse GEMM with 2:4 structured sparsity
./cutlass_profiler --operation=sparse_gem \
  --kernels=cutlass_sm80*sparse \
  --m=1024 --n=1024 --k=1024 \
  --A_type=tf32 --B_type=tf32
```

Sparse GEMM operations use 2:4 structured sparsity where 50% of elements are zero in a structured pattern.

### 28.5.3 Convolution Operations

```bash
# Conv2D forward
./cutlass_profiler --operation=conv2d \
  --kernels=cutlass_sm80*conv2d \
  --n=32 --h=56 --w=56 --c=64 \
  --k=64 --r=3 --s=3

# Conv2D backward (weight gradient)
./cutlass_profiler --operation=conv2d \
  --kernels=cutlass_sm80*conv2d*wgrad \
  --n=32 --h=56 --w=56 --c=64 \
  --k=64 --r=3 --s=3

# Conv2D backward (data gradient)
./cutlass_profiler --operation=conv2d \
  --kernels=cutlass_sm80*conv2d*dgrad \
  --n=32 --h=56 --w=56 --c=64 \
  --k=64 --r=3 --s=3
```

---

## 28.6 Output Format

### 28.6.1 Console Output

The default console output includes comprehensive information about each kernel:

```
cutlass_profiler : GEMM operation
=============================

  ID   Name                                      Status   CompTime  GFLOPS   Runtime(ms)  Bandwidth(GB/s)
  1    cutlass_simt_sgemm_nn                     Passed   0.012     4056.3   0.532         156.2
  2    cutlass_sm80_tensorop_sgemm_nn            Passed   0.008     8432.1   0.256         312.4
  3    cutlass_sm80_tensorop_sgemm_tf32_nn       Passed   0.006     11234.7  0.192         416.7
```

### 28.6.2 CSV Output Format

When writing to CSV, each row contains detailed kernel information:

```csv
Problem,Provider,OperationKind,Operation,Disposition,Status,Verificaton,Alpha,Beta,SplitK,BatchCount,M,N,K,QuantizationOpID, warmed_up,provisioned,iters,compute_time,compute_flops,gpu_clock,gpu_time,gpu_flops,operand_bytes,bw_utilization
GEMM,cutlass,gemm,cutlass_simt_sgemm_nn,passed,success,true,1,0,1,1,1024,1024,1024,,10,100,100,0.532,2147483648,,0.532,4056.3,,,
```

### 28.6.3 Key Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| GFLOPS | Giga floating-point operations per second | `2 * M * N * K / (runtime * 1e6)` |
| Runtime | Kernel execution time in milliseconds | Measured via CUDA events |
| Bandwidth | Effective memory bandwidth in GB/s | `(bytes_read + bytes_written) / (runtime * 1e6)` |
| Compute Time | End-to-end time including launch overhead | Wall-clock measurement |

### 28.6.4 Interpreting Profiler Output

```bash
# Typical output interpretation
cutlass_simt_sgemm_nn :
  Problem: M=1024, N=1024, K=1024
  Runtime: 0.532 ms
  GFLOPS: 4056.3

# Compare against theoretical peak
# A100 FP32 peak: 19,500 GFLOPS (with Tensor Cores)
# A100 FP32 peak: 9,700 GFLOPS (without Tensor Cores)
# Achieved 4056 GFLOPS = ~42% of non-Tensor Core peak
```

---

## 28.7 Kernel Enumeration

### 28.7.1 Listing All Available Kernels

```bash
# List all GEMM kernels
./cutlass_profiler --mode=enumerate --operation=gemm

# List kernels matching a pattern
./cutlass_profiler --mode=enumerate --kernels=cutlass_sm80*tf32*

# Count available kernels
./cutlass_profiler --mode=enumerate --operation=gemm | wc -l
```

### 28.7.2 Filtering by Architecture

```bash
# Ampere SM80 kernels only
./cutlass_profiler --mode=enumerate --kernels=cutlass_sm80*

# Hopper SM90 kernels
./cutlass_profiler --mode=enumerate --kernels=cutlass_sm90*

# Blackwell SM100 kernels
./cutlass_profiler --mode=enumerate --kernels=cutlass_sm100*

# Simt (non-Tensor Core) kernels
./cutlass_profiler --mode=enumerate --kernels=cutlass_simt*
```

### 28.7.3 Filtering by Kernel Schedule (SM90+)

```bash
# Warp-specialized kernels
./cutlass_profiler --mode=enumerate --kernels=cutlass_sm90*warp_specialized*

# TMA-based kernels
./cutlass_profiler --mode=enumerate --kernels=cutlass_sm90*tma*

# Cooperative kernels
./cutlass_profiler --mode=enumerate --kernels=cutlass_sm90*cooperative*
```

---

## 28.8 Performance Ranking

### 28.8.1 Finding the Fastest Kernel

```bash
# Profile all TF32 kernels and sort by performance
./cutlass_profiler --kernels=cutlass_sm80*tf32* \
  --m=4096 --n=4096 --k=4096 \
  --profiling-iterations=100 \
  --warmup-iterations=10

# The output automatically ranks kernels by GFLOPS
```

### 28.8.2 Sweep Across Problem Sizes

```bash
# Sweep GEMM sizes
./cutlass_profiler --kernels=cutlass_sm80*tf32*gemm \
  --m=256:8192:256 --n=256:8192:256 --k=256:8192:256 \
  --output=gemm_sweep.csv
```

### 28.8.3 Roofline Analysis

The profiler output provides enough information to construct a roofline model:

```
# For GEMM: C = A * B + C
# Arithmetic intensity = 2*M*N*K / (M*K + N*K + M*N + M*N) bytes
# For M=N=K=1024, FP16: intensity ~ 80 FLOPs/Byte
# A100 FP16 Tensor Core peak: 312 TFLOPS
# A100 HBM bandwidth: 2 TB/s
# Ridge point: 312/2 = 156 FLOPs/Byte
# Since 80 < 156, this is memory-bound for FP16 at this size
```

---

## 28.9 cuBLAS/cuDNN Comparison

### 28.9.1 Enabling cuBLAS Comparison

```bash
# Compare CUTLASS against cuBLAS for SGEMM
./cutlass_profiler --kernels=cutlass_simt_sgemm* \
  --providers=cutlass,cublas \
  --m=4096 --n=4096 --k=4096

# cuBLAS-only profiling
./cutlass_profiler --kernels=cublas_sgemm \
  --providers=cublas \
  --m=4096 --n=4096 --k=4096
```

### 28.9.2 Enabling cuDNN Comparison

```bash
# Compare CUTLASS Conv2D against cuDNN
./cutlass_profiler --operation=conv2d \
  --kernels=cutlass_sm80*conv2d*fprop \
  --providers=cutlass,cudnn \
  --n=32 --h=56 --w=56 --c=64 \
  --k=64 --r=3 --s=3 \
  --pad_h=1 --pad_w=1 \
  --stride_h=1 --stride_w=1
```

### 28.9.3 Comparison Output Format

When multiple providers are enabled, the output shows side-by-side comparison:

```
Provider       Kernel                                      GFLOPS    Runtime(ms)
cutlass        cutlass_sm80_tensorop_tf32_gemm_nn          18432.5   0.094
cublas         cublas_tf32_gemm_nn                         18944.2   0.091
```

### 28.9.4 Verification Against cuBLAS

```bash
# Use cuBLAS as reference for verification
./cutlass_profiler --kernels=cutlass_sm80*tf32* \
  --verification-enabled=true \
  --reference-provider=cublas \
  --epsilon=0.001
```

---

## 28.10 Hopper-Specific Profiling Options

### 28.10.1 Warp-Specialized Kernels

```bash
# Profile SM90 warp-specialized GEMM
./cutlass_profiler --kernels=cutlass_sm90*e4m3*warpspecialized \
  --m=8192 --n=8192 --k=8192

# Profile SM90 warp-specialized pingpong schedule
./cutlass_profiler --kernels=cutlass_sm90*e4m3*pingpong \
  --m=8192 --n=8192 --k=8192

# Profile SM90 cooperative schedule
./cutlass_profiler --kernels=cutlass_sm90*e4m3*cooperative \
  --m=8192 --n=8192 --k=8192
```

### 28.10.2 TMA-Based Operations

```bash
# Profile TMA-based GEMM
./cutlass_profiler --kernels=cutlass_sm90*tma*e4m3 \
  --m=4096 --n=4096 --k=4096

# Profile cluster-based kernels
./cutlass_profiler --kernels=cutlass_sm90*cluster*e4m3 \
  --m=4096 --n=4096 --k=4096
```

### 28.10.3 Mixed-Dtype Kernels

```bash
# Profile mixed dtype GEMM (e.g., FP8 inputs, FP16 output)
./cutlass_profiler --kernels=cutlass_sm90*mixed*e4m3* \
  --A_type=e4m3 --B_type=e4m3 --C_type=f16 --accum=f32 \
  --m=4096 --n=4096 --k=4096
```

### 28.10.4 Grouped GEMM

```bash
# Profile grouped GEMM
./cutlass_profiler --kernels=cutlass_sm90*grouped*gemm \
  --problem_count=100 \
  --m=128:512:64 --n=128:512:64 --k=128:512:64
```

---

## 28.11 Blackwell-Specific Profiling Options

### 28.11.1 UMMA Operations

```bash
# Profile Blackwell UMMA (Unified MMA) operations
./cutlass_profiler --kernels=cutlass_sm100*umma* \
  --m=8192 --n=8192 --k=8192

# Profile with FP8 and block-scaled types
./cutlass_profiler --kernels=cutlass_sm100*nvfp4* \
  --A_type=nvfp4 --B_type=nvfp4 --C_type=f16 \
  --m=8192 --n=8192 --k=8192
```

### 28.11.2 Block-Scaled GEMM

```bash
# Profile NVFP4 GEMM with scale factors
./cutlass_profiler --kernels=cutlass_sm100*nvfp4*blockwise \
  --m=4096 --n=4096 --k=4096

# Profile MXFP4 format
./cutlass_profiler --kernels=cutlass_sm100*mxfp4* \
  --m=4096 --n=4096 --k=4096

# Profile MXFP8 format
./cutlass_profiler --kernels=cutlass_sm100*mxfp8* \
  --m=4096 --n=4096 --k=4096
```

### 28.11.3 Green Context Profiling

```bash
# Profile kernels using green contexts
./cutlass_profiler --kernels=cutlass_sm100*green_context* \
  --m=8192 --n=8192 --k=8192

# Profile with persistent scheduler
./cutlass_profiler --kernels=cutlass_sm100*persistent* \
  --m=8192 --n=8192 --k=8192
```

### 28.11.4 SM Architecture Variants

```bash
# Profile SM100 (Blackwell base)
./cutlass_profiler --kernels=cutlass_sm100* --m=4096 --n=4096 --k=4096

# Profile SM101 variant
./cutlass_profiler --kernels=cutlass_sm101* --m=4096 --n=4096 --k=4096

# Profile SM103 variant
./cutlass_profiler --kernels=cutlass_sm103* --m=4096 --n=4096 --k=4096

# Profile SM120 variant
./cutlass_profiler --kernels=cutlass_sm120* --m=4096 --n=4096 --k=4096
```

---

## 28.12 Example Commands and Workflows

### 28.12.1 Quick Performance Check

```bash
# Single kernel, single problem size
./cutlass_profiler --kernels=cutlass_sm80_tensorop_tf32_gemm_f16 \
  --m=1024 --n=1024 --k=1024 \
  --warmup-iterations=5 --profiling-iterations=50
```

### 28.12.2 Comprehensive GEMM Sweep

```bash
# Sweep all TF32 kernels across multiple problem sizes
./cutlass_profiler --operation=gemm \
  --kernels=cutlass_sm80*tf32* \
  --m=256,512,1024,2048,4096,8192 \
  --n=256,512,1024,2048,4096,8192 \
  --k=256,512,1024,2048,4096,8192 \
  --warmup-iterations=5 \
  --profiling-iterations=100 \
  --output=tf32_sweep.csv
```

### 28.12.3 Head-to-Head Comparison

```bash
# Compare CUTLASS vs cuBLAS across multiple sizes
./cutlass_profiler --operation=gemm \
  --kernels=cutlass_sm80*hgemm* \
  --providers=cutlass,cublas \
  --m=256:8192:512 --n=256:8192:512 --k=512 \
  --verification-enabled=true \
  --output=comparison.csv
```

### 28.12.4 Convolution Profiling

```bash
# Profile ResNet-like convolutions
for c in 64 128 256 512; do
  for size in 56 28 14 7; do
    ./cutlass_profiler --operation=conv2d \
      --kernels=cutlass_sm80*tf32*fprop \
      --n=64 --h=$size --w=$size --c=$c \
      --k=$c --r=3 --s=3 \
      --pad_h=1 --pad_w=1 \
      --stride_h=1 --stride_w=1 \
      --output=conv2d_resnet.csv --append=true
  done
done
```

### 28.12.5 FP8 Profiling on Hopper

```bash
# Profile FP8 GEMM on SM90
./cutlass_profiler --operation=gemm \
  --kernels=cutlass_sm90*e4m3*warpspecialized \
  --A_type=e4m3 --B_type=e4m3 \
  --m=256:16384:256 --n=256:16384:256 --k=1024 \
  --warmup-iterations=10 \
  --profiling-iterations=100 \
  --output=fp8_sm90_sweep.csv
```

### 28.12.6 Batch Processing Script

```bash
#!/bin/bash
# batch_profile.sh - Profile multiple configurations

SIZES="128 256 512 1024 2048 4096 8192"
DTYPES="f16 bf16 tf32"
OUTPUT="results_$(date +%Y%m%d_%H%M%S).csv"

for dtype in $DTYPES; do
  for size in $SIZES; do
    echo "Profiling ${dtype} GEMM ${size}x${size}x${size}..."
    ./cutlass_profiler --operation=gemm \
      --kernels=cutlass_sm80*${dtype}* \
      --m=$size --n=$size --k=$size \
      --profiling-iterations=50 \
      --warmup-iterations=5 \
      --output=$OUTPUT --append=true
  done
done
echo "Results saved to $OUTPUT"
```

---

## 28.13 Interpreting Profiler Output

### 28.13.1 Understanding GFLOPS

The GFLOPS metric for GEMM is calculated as:

```
GFLOPS = 2 * M * N * K / (runtime_ms * 1e6)
```

The factor of 2 accounts for the multiply-add operation. For example, for M=N=K=4096:

```
GFLOPS = 2 * 4096 * 4096 * 4096 / (runtime_ms * 1e6)
       = 137,438,953,472 / (runtime_ms * 1e6)
```

A runtime of 0.5ms would yield ~274,878 GFLOPS (275 TFLOPS).

### 28.13.2 Bandwidth Utilization

Effective bandwidth measures how well the kernel utilizes memory:

```
Bandwidth = (bytes_A + bytes_B + bytes_C + bytes_D) / (runtime_ms * 1e6)
```

For FP16 GEMM with M=N=K=1024:

```
bytes_A = M * K * 2 = 2,097,152
bytes_B = K * N * 2 = 2,097,152
bytes_C = M * N * 4 = 4,194,304  (FP32 accumulator output)
bytes_D = M * N * 2 = 2,097,152  (FP16 output)
Total   = 10,485,760 bytes
```

### 28.13.3 Performance Classification

| Runtime Characteristic | Typical Cause |
|-----------------------|---------------|
| High GFLOPS, high bandwidth | Compute-bound kernel (expected for large GEMM) |
| Low GFLOPS, high bandwidth | Memory-bound kernel (expected for small K) |
| Low GFLOPS, low bandwidth | Suboptimal kernel configuration or low occupancy |
| High variance between iterations | Thermal throttling or GPU frequency instability |

### 28.13.4 Common Performance Issues

```bash
# Check if kernel is compute-bound
# Compare achieved GFLOPS against theoretical peak
# A100 FP16 Tensor Core peak: ~312 TFLOPS
# If achieving >70% peak, the kernel is well-tuned

# Check if kernel is memory-bound
# Compare achieved bandwidth against HBM bandwidth
# A100 HBM2e: ~2 TB/s
# If achieving >80% bandwidth, memory utilization is good

# Diagnose low occupancy
# Look at threadblock configuration in kernel name
# Small tile sizes may lead to low occupancy
```

---

## 28.14 Summary

The CUTLASS Profiler is an essential tool for:

- **Benchmarking**: Measuring kernel performance with controlled iterations and warmup.
- **Verification**: Ensuring numerical correctness against reference implementations.
- **Discovery**: Enumerating available kernels for a given architecture and operation.
- **Comparison**: Side-by-side performance comparison with cuBLAS and cuDNN.
- **Analysis**: Understanding performance through GFLOPS, bandwidth, and runtime metrics.

Key workflow:

1. Build the profiler with target architectures enabled.
2. Use enumerate mode to discover available kernels.
3. Profile specific kernels with appropriate problem sizes.
4. Compare against vendor libraries (cuBLAS/cuDNN).
5. Sweep problem sizes to find performance sweet spots.
6. Export results to CSV/JSON for further analysis.
