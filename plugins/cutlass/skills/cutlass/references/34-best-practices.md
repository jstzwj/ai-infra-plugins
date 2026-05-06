# CUTLASS Best Practices - Chapter 34: Performance Tuning, Optimization, and Common Patterns

This reference covers performance tuning guidelines, memory optimization, kernel selection strategies, debugging techniques, and migration advice for CUTLASS.

---

## 34.1 Performance Tuning Guidelines

### 34.1.1 Tile Shape Selection

The tile shape (threadblock tile) determines how much work each threadblock performs. Selecting the right tile shape is critical for achieving high performance.

**General Principles:**

- Larger tiles mean more work per threadblock but fewer threadblocks, potentially underutilizing the GPU.
- Smaller tiles mean more threadblocks but more overhead from redundant data loading.
- The optimal tile shape depends on the problem dimensions, data types, and architecture.

```cpp
// Common tile shapes and when to use them:

// Large tiles for large matrices (M, N >= 2048)
using TileShape = cutlass::gemm::GemmShape<128, 256, 64>;  // High occupancy

// Medium tiles for medium matrices (256 <= M, N < 2048)
using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;  // Balanced

// Small tiles for small matrices or tall-skinny/short-wide shapes
using TileShape = cutlass::gemm::GemmShape<64, 64, 64>;    // Low latency

// Very small tiles for extremely small problems
using TileShape = cutlass::gemm::GemmShape<32, 32, 32>;    // Avoid SM underutilization
```

**Architecture-Specific Recommendations:**

| Architecture | Recommended Tile Shape | Notes |
|---|---|---|
| SM80 (Ampere) | `128x128x32` or `128x64x64` | 4-8 stages typical |
| SM90 (Hopper) | `128x128x64` or `256x128x64` | TMA handles large tiles efficiently |
| SM100 (Blackwell) | `128x256x64` or `256x128x128` | UMMA benefits from wider tiles |

```cpp
// For Hopper with TMA, larger tiles are often better because TMA amortizes
// the cost of large transfers. Use StageCountAutoCarveout to let the compiler
// determine the optimal stage count.
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 128 / cutlass::sizeof_bits<ElementA>::value,  // Alignment
    ElementB, LayoutB, 128 / cutlass::sizeof_bits<ElementB>::value,
    ElementAccumulator,
    cutlass::gemm::GemmShape<128, 128, 64>,                        // Tile shape
    cutlass::gemm::collective::StageCountAutoCarveout<0>,           // Auto stages
    cutlass::gemm::collective::KernelScheduleAuto                   // Auto schedule
>::CollectiveOp;
```

### 34.1.2 Stage Count Optimization

The stage count determines how many tiles of the K dimension are prefetched into shared memory. More stages overlap computation with data loading but increase shared memory consumption.

```cpp
// Manual stage count selection
using StageCount = cutlass::gemm::collective::StageCount<4>;  // Exactly 4 stages

// Automatic stage count (recommended for most cases)
using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<0>;

// Auto with carveout: reserve shared memory for epilogue
using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<sizeof(ElementD) * 128 * 8>;

// Stage count guidelines:
// - SM80 (Ampere): 3-8 stages typical, limited by 164 KB shared memory
// - SM90 (Hopper): 2-10 stages, TMA reduces pressure on register usage
// - More stages help when K is large (>256) and memory bandwidth is bottleneck
// - Fewer stages (2-4) are better for small K or compute-bound scenarios
```

### 34.1.3 Alignment Requirements

Proper alignment of data pointers and strides is essential for performance and correctness.

```cpp
// Required alignment for Tensor Core operations:
// FP16/BF16: 128-bit alignment (8 elements)
// TF32:      128-bit alignment (4 elements)
// FP8:       128-bit alignment (16 elements)
// INT8:      128-bit alignment (16 elements)
// INT4:      128-bit alignment (32 elements)

// Alignment is specified in the CollectiveBuilder:
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 8,   // Alignment for A (must match pointer alignment)
    ElementB, LayoutB, 8,   // Alignment for B (must match pointer alignment)
    ElementAccumulator,
    TileShape, StageCount, Schedule
>::CollectiveOp;

// Ensure runtime pointers are properly aligned
void* ptr_A;
cudaMalloc(&ptr_A, M * K * sizeof(ElementA));  // cudaMalloc returns 256-byte aligned

// Check stride alignment
int lda = K;  // Leading dimension
// For 128-bit alignment with FP16 (2 bytes): lda must be divisible by 8
// For 128-bit alignment with FP32 (4 bytes): lda must be divisible by 4

// The alignment parameter in CollectiveBuilder must not exceed the
// actual alignment of your data. If alignment is 1, only 1 is safe:
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA, 1,   // Unaligned (fallback to slower path)
    ElementB, LayoutB, 1,
    ElementAccumulator,
    TileShape, StageCount, Schedule
>::CollectiveOp;
```

### 34.1.4 Layout Selection for Optimal Performance

The memory layout of input and output matrices affects whether additional shared memory transpose operations are needed.

```cpp
// Optimal layout combinations for Tensor Cores:
// - A is RowMajor + B is ColumnMajor: optimal for most Tensor Core ops
// - A is ColumnMajor + B is RowMajor: also good
// - A is RowMajor + B is RowMajor: requires transpose in shared memory
// - A is ColumnMajor + B is ColumnMajor: requires transpose in shared memory

// CUTLASS handles all layout combinations correctly, but some are faster:
using LayoutA = cutlass::layout::RowMajor;    // or ColumnMajor
using LayoutB = cutlass::layout::ColumnMajor;  // or RowMajor

// For best performance with Tensor Core operations:
// On SM80+ (Ampere+), the layout matters less due to shared memory transpose
// On SM90+ (Hopper), TMA handles layout transformation efficiently
```

---

## 34.2 Memory Optimization

### 34.2.1 Shared Memory Usage

Shared memory is a scarce resource. Each SM has a limited amount (up to 228 KB on SM90, 164 KB on SM80). The stage count and tile shape directly determine shared memory consumption.

```cpp
// Estimate shared memory per threadblock:
// SMEM = stages * (tile_M * tile_K * sizeof(ElementA) + tile_K * tile_N * sizeof(ElementB))
//
// Example: 4 stages, 128x128x64 tile, FP16
// SMEM = 4 * (128 * 64 * 2 + 64 * 128 * 2) = 4 * (16384 + 16384) = 131,072 bytes = 128 KB

// Check shared memory usage at runtime:
cudaFuncAttributes attr;
cudaFuncGetAttributes(&attr, kernel_function);
printf("Shared memory per block: %zu bytes\n", attr.sharedSizeBytes);

// Reduce shared memory usage:
// 1. Use fewer stages
using StageCount = cutlass::gemm::collective::StageCount<2>;  // Instead of 4

// 2. Use smaller tile shapes
using TileShape = cutlass::gemm::GemmShape<64, 64, 32>;

// 3. Use smaller data types (FP16 instead of FP32 inputs)
using ElementA = cutlass::half_t;  // Instead of float
```

### 34.2.2 Register Pressure Management

Register pressure affects occupancy. Each SM has a fixed register file (65536 registers on SM80+).

```cpp
// Control register usage with maxrregcount
// Lower register count = higher occupancy but potentially more spills
nvcc --maxrregcount=128 my_kernel.cu

// The tradeoff:
// - More registers per thread: faster computation, lower occupancy
// - Fewer registers per thread: higher occupancy, potential register spills to local memory

// Use CUTLASS_DEBUG_TRACE_LEVEL to check register usage:
// Set CUTLASS_DEBUG_TRACE_LEVEL >= 1 and examine kernel launch info

// For SM90 with warp specialization:
// Producer warps and consumer warps have different register requirements
// WGMMA consumer warps typically need 240+ registers
```

### 34.2.3 Bank Conflict Avoidance

Shared memory is organized into 32 banks. Bank conflicts occur when multiple threads in a warp access the same bank simultaneously, serializing access.

```cpp
// CUTLASS handles bank conflict avoidance internally through:
// 1. Padding in shared memory layouts
// 2. Careful access patterns in the MMA mainloop
// 3. Crosswise layouts that stagger access

// If you implement custom shared memory operations, be aware of bank conflicts:
// - Consecutive 4-byte words map to consecutive banks
// - Bank index = (byte_address / 4) % 32
// - Multiple accesses to the same bank in the same instruction are serialized

// Common mitigation: pad the shared memory stride
// Instead of: __shared__ float tile[128][128];
// Use:       __shared__ float tile[128][128 + 4];  // 4-element padding avoids bank conflicts
```

---

## 34.3 GEMM Kernel Selection

### 34.3.1 SIMT vs TensorOp

```cpp
// SIMT (Single Instruction Multiple Thread): scalar CUDA cores
// - Works on all architectures
// - Supports arbitrary data types
// - Slower for dense matrix multiply
// - Better for very small problem sizes

// TensorOp: Tensor Cores
// - Much higher throughput (4-16x over SIMT)
// - Limited data type support per architecture
// - Requires alignment (128-bit)
// - Better for medium-to-large problem sizes

// When to use SIMT:
// - Very small matrices where Tensor Core overhead dominates
// - Custom data types not supported by Tensor Cores
// - Debugging (easier to understand and verify)

// When to use TensorOp:
// - Performance-critical code
// - Matrices of size 128x128 or larger
// - Supported data types (FP16, BF16, TF32, FP8, INT8)

// Specify via CollectiveBuilder:
using CollectiveOp_SIMT = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassSimt,     // SIMT
    ElementA, LayoutA, 1,
    ElementB, LayoutB, 1,
    ElementAccumulator,
    TileShape, StageCount, Schedule
>::CollectiveOp;

using CollectiveOp_TensorOp = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassTensorOp, // TensorOp
    ElementA, LayoutA, 8,
    ElementB, LayoutB, 8,
    ElementAccumulator,
    TileShape, StageCount, Schedule
>::CollectiveOp;
```

### 34.3.2 Architecture-Appropriate Kernels

```cpp
// SM70 (Volta): WMMA FP16 only
// OpClass: OpClassTensorOp with WMMA instruction shape 16x16x16

// SM75 (Turing): MMA PTX instructions, INT8/INT4/INT1 support
// OpClass: OpClassTensorOp with instruction shapes 16x8x8 (FP16), 8x8x16 (INT8)

// SM80 (Ampere): cp.async, TF32, BF16, async copy
// OpClass: OpClassTensorOp with instruction shapes 16x8x16 (TF32), 16x8x16 (FP16/BF16)

// SM90 (Hopper): TMA, GMMA, WGMMA, warp specialization
// OpClass: OpClassTensorOp with GMMA instructions
// Schedule: KernelTmaWarpSpecialized, KernelCpAsyncWarpSpecialized

// SM100 (Blackwell): UMMA, block-scaled types
// OpClass: OpClassTensorOp with UMMA instructions

// Use CollectiveBuilder to automatically select the best kernel:
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,              // Architecture tag
    cutlass::arch::OpClassTensorOp,   // Operation class
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, StageCount, Schedule
>::CollectiveOp;
// CollectiveBuilder selects the appropriate MMA atom, mainloop, and pipeline
// based on the architecture and operation class.
```

### 34.3.3 Batched vs Grouped GEMM

```cpp
// Batched GEMM: All problems have the same dimensions, runs as a single kernel
// Use when: all batches have identical M, N, K
using GemmBatched = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<CollectiveOp, EpilogueOp>
>;

GemmBatched gemm_op;
typename GemmBatched::Arguments args{
    cutlass::gemm::GemmUniversalMode::kBatched,  // Batched mode
    {M, N, K},                                    // All batches same size
    {ptr_A, stride_A, batch_stride_A},            // Batch-strided A
    {ptr_B, stride_B, batch_stride_B},            // Batch-strided B
    {ptr_C, stride_C, batch_stride_C},            // Batch-strided C
    {ptr_D, stride_D, batch_stride_D},            // Batch-strided D
    {alpha, beta},
    batch_count                                   // Number of batches
};
gemm_op(args);

// Grouped GEMM: Each problem can have different dimensions, runs as a single kernel
// Use when: each group has different M, N, K or layouts
using GemmGrouped = cutlass::gemm::device::GemmGrouped<
    cutlass::gemm::kernel::GemmGrouped<CollectiveOp, EpilogueOp>
>;

// Grouped GEMM requires an array of problem descriptors
std::vector<cutlass::gemm::GemmCoord> problem_sizes;
std::vector<int64_t> ptr_A, ptr_B, ptr_C, ptr_D;
std::vector<typename LayoutA::Stride> lda;
std::vector<typename LayoutB::Stride> ldb;
std::vector<typename LayoutC::Stride> ldc, ldd;

// ... fill problem_sizes, pointers, and strides for each group ...

typename GemmGrouped::Arguments args{
    problem_sizes.size(),
    problem_sizes.data(),
    problem_sizes.size(),  // Problem count
    ptr_A.data(), lda.data(),
    ptr_B.data(), ldb.data(),
    ptr_C.data(), ldc.data(),
    ptr_D.data(), ldd.data(),
    alpha, beta
};
```

---

## 34.4 Common Patterns

### 34.4.1 Workspace Allocation

Some CUTLASS kernels require a workspace buffer for intermediate results or split-K reduction.

```cpp
// Get workspace size requirement
typename Gemm::Arguments args{...};
size_t workspace_size = Gemm::get_workspace_size(args);

// Allocate workspace
void* workspace;
cudaMalloc(&workspace, workspace_size);

// Pass workspace to the kernel
args.workspace = workspace;

// Alternative: use the Gemm device adapter which manages workspace
Gemm gemm_op;
auto status = gemm_op.initialize(args, workspace, stream);
status = gemm_op.run(stream);

// Always check workspace requirements for split-K:
// Split-K requires workspace for partial reduction results
// Size = split_k_slices * M * N * sizeof(ElementAccumulator)
```

### 34.4.2 Stream Management

```cpp
// CUTLASS kernels respect CUDA streams
cudaStream_t stream;
cudaStreamCreate(&stream);

// Run on a specific stream
Gemm gemm_op;
auto status = gemm_op.initialize(args, workspace, stream);
status = gemm_op.run(stream);

// Multiple independent GEMMs on different streams
for (int i = 0; i < num_streams; ++i) {
    gemm_ops[i].initialize(args[i], workspace[i], streams[i]);
}
for (int i = 0; i < num_streams; ++i) {
    gemm_ops[i].run(streams[i]);
}

// Synchronize all streams
for (int i = 0; i < num_streams; ++i) {
    cudaStreamSynchronize(streams[i]);
}
```

### 34.4.3 Error Handling

```cpp
#include "cutlass/cutlass.h"

// CUTLASS uses cutlass::Status for error reporting
// Always check return values:

// Initialize
cutlass::Status status = gemm_op.initialize(args, workspace, stream);
if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS initialize failed: "
              << cutlassGetStatusString(status) << std::endl;
    return EXIT_FAILURE;
}

// Run
status = gemm_op.run(stream);
if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS run failed: "
              << cutlassGetStatusString(status) << std::endl;
    return EXIT_FAILURE;
}

// Common error statuses:
// cutlass::Status::kSuccess             - Operation succeeded
// cutlass::Status::kErrorMisalignedOperand   - Data alignment mismatch
// cutlass::Status::kErrorInvalidProblem      - Problem dimensions invalid
// cutlass::Status::kErrorNotSupported        - Operation not supported on this arch
// cutlass::Status::kErrorWorkspaceNull       - Workspace pointer is null
// cutlass::Status::kErrorInternal            - Internal error

// Also check the can_implement method before running:
status = Gemm::can_implement(args);
if (status != cutlass::Status::kSuccess) {
    std::cerr << "Kernel cannot implement this problem configuration: "
              << cutlassGetStatusString(status) << std::endl;
    return EXIT_FAILURE;
}
```

---

## 34.5 Debugging Tips

### 34.5.1 CUTLASS_DEBUG_TRACE_LEVEL

```cpp
// Compile with debug trace level
// Level 0: No tracing (default for release)
// Level 1: Basic kernel launch info
// Level 2: Per-stage data movement info
// Level 3: Detailed per-thread tracing

// In CMake:
cmake .. -DCUTLASS_DEBUG_TRACE_LEVEL=2

// Or define at compile time:
// #define CUTLASS_DEBUG_TRACE_LEVEL 2

// At level 1, CUTLASS prints:
// - Grid dimensions
// - Block dimensions
// - Problem size
// - Tile shape
// - Shared memory allocation size
```

### 34.5.2 Kernel can_implement Checks

```cpp
// Before running a kernel, verify it can handle your configuration:
auto status = Gemm::can_implement(args);
if (status != cutlass::Status::kSuccess) {
    // Common reasons for failure:
    // 1. Alignment mismatch (data not aligned to required boundary)
    // 2. Layout incompatibility
    // 3. Data type not supported on target architecture
    // 4. Tile shape not divisible by instruction shape
    // 5. Problem dimensions too small for the tile shape

    std::cerr << "Cannot implement: " << cutlassGetStatusString(status) << std::endl;
}

// The can_implement check verifies:
// - Element types are supported
// - Layouts are compatible
// - Alignment requirements are met
// - Tile shapes are valid
// - Architecture supports the requested operation
```

### 34.5.3 Layout Compatibility

```cpp
// Common layout issues:

// 1. Wrong stride computation for RowMajor vs ColumnMajor
// RowMajor: stride = N (columns), element (i,j) at offset i*stride + j
// ColumnMajor: stride = M (rows), element (i,j) at offset j*stride + i

// 2. Using wrong LayoutA/LayoutB in the kernel definition
// Always double-check that the layout template parameter matches
// the actual data layout in memory

// 3. Mixed layouts
// A: RowMajor, B: ColumnMajor -> NN (non-transpose, non-transpose)
// A: ColumnMajor, B: RowMajor -> TT (transpose, transpose)
// A: RowMajor, B: RowMajor -> NT (non-transpose, transpose)
// A: ColumnMajor, B: ColumnMajor -> TN (transpose, non-transpose)

// Verify by checking a few elements:
// If A is RowMajor with leading dimension lda:
//   A[i*K + k] should equal the element at row i, column k
// If A is ColumnMajor with leading dimension lda:
//   A[k*lda + i] should equal the element at row i, column k
```

---

## 34.6 Profiling Best Practices

### 34.6.1 Using the CUTLASS Profiler

```bash
# Profile GEMM with specific parameters
./cutlass_profiler --kinds=gemm --m=1024 --n=1024 --k=1024 \
  --A=f16:row --B=f16:col --C=f32:row

# Profile all supported GEMM configurations
./cutlass_profiler --kinds=gemm --m=1024 --n=1024 --k=1024

# Profile only specific architectures
./cutlass_profiler --kinds=gemm --devices=0 --m=2048 --n=2048 --k=2048

# Warmup and iteration control
./cutlass_profiler --kinds=gemm --warmup-iterations=10 --profiling-iterations=100

# Output formats
./cutlass_profiler --kinds=gemm --output=default
./cutlass_profiler --kinds=gemm --output=csv --j=2048
```

### 34.6.2 Custom Benchmarking

```cpp
// Use CUDA events for accurate timing
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Warmup
for (int i = 0; i < 5; ++i) {
    gemm_op.run(stream);
}

// Benchmark
int num_iters = 100;
cudaEventRecord(start, stream);
for (int i = 0; i < num_iters; ++i) {
    gemm_op.run(stream);
}
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);

float elapsed_ms;
cudaEventElapsedTime(&elapsed_ms, start, stop);
double avg_ms = elapsed_ms / num_iters;

// Compute TFLOPS:
// FLOPS for GEMM(M,N,K) = 2 * M * N * K
// TFLOPS = FLOPS / (avg_ms * 1e-3) / 1e12
double tflops = (2.0 * M * N * K) / (avg_ms * 1e-3) / 1e12;
printf("Performance: %.2f TFLOPS (%.3f ms)\n", tflops, avg_ms);
```

### 34.6.3 Roofline Analysis

```cpp
// Compute arithmetic intensity to understand bottlenecks:
// Arithmetic intensity = FLOPS / bytes_transferred
// For GEMM(M,N,K) with FP16:
//   FLOPS = 2 * M * N * K
//   Bytes = M*K*2 + N*K*2 + M*N*4 + M*N*2 (A + B + C + D, assuming FP16 A,B,D and FP32 C)
//   AI = 2*M*N*K / (2*M*K + 2*N*K + 4*M*N + 2*M*N)

// If AI > ridge_point: compute-bound (optimize for throughput)
// If AI < ridge_point: memory-bound (optimize for bandwidth)

// For large K, GEMM is typically compute-bound
// For small K, GEMM is typically memory-bound
```

---

## 34.7 Migration from CUTLASS 2.x to 3.x

### 34.7.1 API Changes Overview

CUTLASS 3.x introduced significant API changes while maintaining backward compatibility for most 2.x code.

```cpp
// CUTLASS 2.x style (still supported):
#include "cutlass/gemm/device/gemm.h"
using Gemm2x = cutlass::gemm::device::Gemm<
    ElementA, LayoutA, ElementB, LayoutB,
    ElementC, LayoutC, ElementAccumulator,
    OpClass, ArchTag,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOp
>;

// CUTLASS 3.x style (recommended):
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator, TileShape, StageCount, Schedule
>::CollectiveOp;
using Gemm3x = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<CollectiveOp, EpilogueOp>
>;
```

### 34.7.2 Key Differences

| Feature | CUTLASS 2.x | CUTLASS 3.x |
|---|---|---|
| Kernel definition | Nested template parameters | CollectiveBuilder + GemmUniversal |
| Tile hierarchy | Threadblock > Warp > Thread | Single TileShape (CuTe tiling) |
| Mainloop | Explicit stage management | Collective handles stages automatically |
| Epilogue | Thread-level epilogue functors | Collective epilogue with fusion support |
| Data movement | Tile iterators | CuTe copy atoms |
| Layout | Pre-defined layout classes | CuTe layout algebra |
| Warp specialization | Not available | Producer/consumer warp groups (SM90+) |

### 34.7.3 Migration Steps

1. **Replace device GEMM with GemmUniversalAdapter**: The `GemmUniversalAdapter` is the entry point for 3.x-style kernels.
2. **Use CollectiveBuilder instead of manual template assembly**: CollectiveBuilder automatically selects the best MMA atom, mainloop, and pipeline.
3. **Update argument passing**: 3.x uses a structured `Arguments` type with named fields.
4. **Update epilogue**: Use `DefaultEpilogue` or custom collective epilogue.
5. **Update stride types**: Strides are now specified as CuTe strides (int64_t typically).

```cpp
// Step-by-step migration example:

// BEFORE (2.x):
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>
>;

Gemm gemm_op;
gemm_op({M, N, K}, alpha, ptr_A, lda, ptr_B, ldb, beta, ptr_C, ldc, ptr_D, ldd);

// AFTER (3.x):
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm80, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::layout::RowMajor, cutlass::layout::RowMajor,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>;

using Kernel = cutlass::gemm::kernel::GemmUniversal<CollectiveOp, EpilogueOp>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

Gemm gemm_op;
typename Gemm::Arguments args{
    {M, N, K},
    {ptr_A, lda}, {ptr_B, ldb},
    {ptr_C, ldc}, {ptr_D, ldd},
    {alpha, beta}
};
gemm_op(args);
```

---

## 34.8 Common Pitfalls and How to Avoid Them

### 34.8.1 Alignment Mismatch

```cpp
// WRONG: Specify alignment 8 but data is only 4-byte aligned
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA, 8,   // Claims 8-element alignment
    ElementB, LayoutB, 8,
    ElementAccumulator, TileShape, StageCount, Schedule
>::CollectiveOp;
// But lda = 127 (not divisible by 8) -> incorrect results or crash

// RIGHT: Match alignment to actual data layout
int alignment_A = 8;  // If you can guarantee it
// Or use alignment 1 for safety (slower but correct):
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA, 1,   // Safe but may be slower
    ElementB, LayoutB, 1,
    ElementAccumulator, TileShape, StageCount, Schedule
>::CollectiveOp;
```

### 34.8.2 Forgetting to Check can_implement

```cpp
// WRONG: Just run and hope for the best
gemm_op(args);

// RIGHT: Always check first
auto status = Gemm::can_implement(args);
if (status != cutlass::Status::kSuccess) {
    // Handle error gracefully
    fprintf(stderr, "Configuration not supported: %s\n",
            cutlassGetStatusString(status));
    return false;
}
gemm_op.initialize(args, workspace, stream);
gemm_op.run(stream);
```

### 34.8.3 Wrong Stride Convention

```cpp
// WRONG: Using column-major stride for row-major data
// For RowMajor matrix of shape M x K:
int64_t stride_A = K;  // CORRECT: stride between consecutive rows

// WRONG: For ColumnMajor matrix of shape M x K:
int64_t stride_A = M;  // CORRECT: stride between consecutive columns

// Common mistake: confusing the stride direction
// RowMajor stride = number of columns (K for A in GEMM)
// ColumnMajor stride = number of rows (M for A in GEMM)
```

### 34.8.4 Not Allocating Workspace for Split-K

```cpp
// WRONG: Use split-k without workspace
args.mode = cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel;
args.k_partition = 4;
// But workspace is nullptr -> crash or garbage output

// RIGHT: Always allocate workspace when using split-K
size_t workspace_size = Gemm::get_workspace_size(args);
void* workspace;
cudaMalloc(&workspace, workspace_size);
auto status = gemm_op.initialize(args, workspace, stream);
```

### 34.8.5 Uninitialized Output

```cpp
// WRONG: Assume output D is initialized
// CUTLASS computes D = alpha * A * B + beta * C
// If beta is 0, C is not read, but D must still be writable
// If beta is non-zero, C must contain valid data

// RIGHT: Initialize C when using beta != 0
if (beta != 0) {
    // Ensure C contains valid data
    // The epilogue reads C and writes D = alpha*AB + beta*C
}

// For initializing D without reading C, set beta = 0
// D = alpha * A * B + 0 * C (C is not read)
```

### 34.8.6 Tile Shape Not Divisible by Instruction Shape

```cpp
// WRONG: Tile K dimension not divisible by instruction K
using TileShape = cutlass::gemm::GemmShape<128, 128, 48>;
// For FP16 TensorOp with instruction shape 16x8x16:
// 48 % 16 != 0 -> compilation error or incorrect results

// RIGHT: Ensure tile dimensions are divisible by instruction dimensions
using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;  // 64 % 16 = 0

// Common instruction shapes:
// FP16 TensorOp SM80: 16x8x16 (MxNxK)
// TF32 TensorOp SM80: 16x8x8
// FP16 WGMMA SM90:    64xNnx16
// BF16 WGMMA SM90:    64xNnx16
```

---

## 34.9 Summary Checklist

- [ ] Use CollectiveBuilder for automatic kernel selection
- [ ] Match alignment parameters to actual data alignment
- [ ] Call `can_implement()` before running
- [ ] Allocate workspace when using split-K
- [ ] Use appropriate tile shapes for problem sizes
- [ ] Profile with CUTLASS Profiler before and after changes
- [ ] Check stride conventions match your data layout
- [ ] Use StageCountAutoCarveout for optimal stage count
- [ ] Prefer 3.x API (GemmUniversalAdapter) for new code
- [ ] Set CUTLASS_DEBUG_TRACE_LEVEL for debugging
