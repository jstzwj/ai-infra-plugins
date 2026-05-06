# CUTLASS 2.x GEMM API

CUTLASS 2.x organizes GEMM computation into a strict hierarchy that mirrors the GPU hardware architecture. Each level of the hierarchy corresponds to a level of parallelism: device-level orchestration, threadblock-level tiling, warp-level MMA, and thread-level element operations. This document covers the complete API at each level.

---

## GEMM Hierarchy Overview

```
Device API          -- Host-facing, manages kernel launch, workspace, API stream
  |
  Threadblock API   -- Maps to one CUDA threadblock; manages tile loading, MMA, epilogue
    |
    Warp API        -- Maps to one CUDA warp (32 threads); executes warp-level MMA
      |
      Thread API    -- Individual thread operations; element-wise compute
        |
        Instruction API -- Maps to hardware MMA instructions (HMMA, IMMA, etc.)
```

Each level is templatized and composable. The user typically interacts only with the Device API, but understanding the lower levels is essential for customization and optimization.

---

## Layout Notation: NN, NT, TN, TT

CUTLASS uses a concise notation to specify the data layout of operands A and B:

| Notation | Layout A | Layout B | Meaning |
|---|---|---|---|
| **NN** | Row-Major | Row-Major | Both A and B are row-major (non-transposed) |
| **NT** | Row-Major | Column-Major | A is row-major, B is column-major |
| **TN** | Column-Major | Row-Major | A is column-major, B is row-major |
| **TT** | Column-Major | Column-Major | Both A and B are column-major (both transposed) |

In terms of GEMM: `D = alpha * A * B + beta * C`

- Row-major A (NN/TN): A[i][k], contiguous along K
- Column-major A (NT/TT): A[k][i], contiguous along M
- Row-major B (NN/TN): B[k][j], contiguous along N
- Column-major B (NT/TT): B[j][k], contiguous along K

```cpp
// Layout selection example
using LayoutA = cutlass::layout::RowMajor;    // NN or TN
using LayoutB = cutlass::layout::RowMajor;    // NN or TN
using LayoutC = cutlass::layout::RowMajor;    // Output layout

// For NT layout:
// using LayoutA = cutlass::layout::RowMajor;
// using LayoutB = cutlass::layout::ColumnMajor;
```

---

## Device API

The Device API is the top-level interface for launching GEMM kernels. It handles workspace allocation, kernel configuration, grid launch, and error checking.

### cutlass::gemm::device::Gemm Template

```cpp
namespace cutlass::gemm::device {

template <
  typename ElementA_,              // Data type of operand A (e.g., half_t, float, int8_t)
  typename LayoutA_,               // Layout of A (RowMajor or ColumnMajor)
  typename ElementB_,              // Data type of operand B
  typename LayoutB_,               // Layout of B (RowMajor or ColumnMajor)
  typename ElementC_,              // Data type of operand C (output)
  typename LayoutC_,               // Layout of C (RowMajor or ColumnMajor)
  typename ElementAccumulator_,    // Accumulator type (typically float or int32_t)
  typename OperatorClass_ = cutlass::arch::OpClassSimt,  // SIMT or TensorOp
  typename ArchTag_ = cutlass::arch::Sm70,               // Target SM architecture
  typename ThreadblockShape_ = GemmShape<128, 128, 8>,   // Tile dimensions (M, N, K)
  typename WarpShape_ = GemmShape<32, 64, 8>,            // Warp-level tile shape
  typename InstructionShape_ = GemmShape<1, 1, 1>,       // Instruction shape (SIMT) or MMA shape
  typename EpilogueOutputOp_ = epilogue::thread::LinearCombination<
      ElementC_, 128 / sizeof_bits<ElementC_>::value,
      ElementAccumulator_, ElementAccumulator_>,
  typename ThreadblockSwizzle_ = threadblock::GemmIdentityThreadblockSwizzle<>,
  int Stages = 2,                                        // Pipeline stages
  typename Operator_ = typename device::DefaultGemmConfiguration<
      ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::Operator,
  bool SplitKSerial = false                              // Split-K serial reduction
>
class Gemm;

} // namespace cutlass::gemm::device
```

### Key Template Parameters Explained

#### Element Types

```cpp
// ElementA, ElementB: input operand data types
using ElementA = cutlass::half_t;   // FP16 input A
using ElementB = cutlass::half_t;   // FP16 input B
using ElementC = cutlass::half_t;   // FP16 output C
using ElementAccumulator = float;   // FP32 accumulation

// For integer GEMM:
// using ElementA = int8_t;
// using ElementB = int8_t;
// using ElementAccumulator = int32_t;

// For mixed precision:
// using ElementA = cutlass::half_t;
// using ElementB = cutlass::half_t;
// using ElementC = float;            // higher precision output
// using ElementAccumulator = float;
```

#### Operator Class

```cpp
// OpClassSimt: use SIMT (scalar) per-thread math pipeline
//   - Works on all architectures (SM50+)
//   - No Tensor Core usage
//   - InstructionShape = GemmShape<1, 1, 1>
using OpClass = cutlass::arch::OpClassSimt;

// OpClassTensorOp: use Tensor Core MMA instructions
//   - SM70: HMMA.1688 (FP16, 16x8x8)
//   - SM75: HMMA.1688 (FP16), IMMA.8816 (INT8)
//   - SM80: HMMA.16816 (FP16), BMMA (BFloat16), IMMA, TF32
using OpClass = cutlass::arch::OpClassTensorOp;
```

#### Architecture Tags

```cpp
using ArchSm70 = cutlass::arch::Sm70;   // Volta (V100)
using ArchSm75 = cutlass::arch::Sm75;   // Turing (T4, RTX 2080)
using ArchSm80 = cutlass::arch::Sm80;   // Ampere (A100)
using ArchSm86 = cutlass::arch::Sm86;   // Ampere (GA10x)
using ArchSm89 = cutlass::arch::Sm89;   // Ada (RTX 4090)
```

#### Tile Shapes

```cpp
// ThreadblockShape: the tile size processed by one threadblock
//   (M, N, K) -- the MxN tile from the output, with K being the
//   dimension loaded per pipeline stage
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;

// WarpShape: the tile size processed by one warp within a threadblock
//   Must evenly divide the ThreadblockShape
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;

// InstructionShape: the shape of one hardware MMA instruction
//   For SM80 FP16 TensorOp: 16x8x16
//   For SM80 SIMT: 1x1x1
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

// Constraints:
// ThreadblockShape::kM % WarpShape::kM == 0
// ThreadblockShape::kN % WarpShape::kN == 0
// WarpShape::kM % InstructionShape::kM == 0
// WarpShape::kN % InstructionShape::kN == 0
// WarpShape::kK % InstructionShape::kK == 0
```

#### Stage Count

```cpp
// Stages: number of pipeline stages for double/multi-buffering
//   Stages = 2: double buffering (load + compute overlap)
//   Stages = 3+: multi-stage pipeline (SM80+ with cp.async)
//   Higher stages use more shared memory but improve overlap
int Stages = 3;  // Typical for SM80 with cp.async
int Stages = 2;  // Typical for SM70/SIMT
```

### Complete GEMM Type Definition

```cpp
#include "cutlass/gemm/device/gemm.h"

// Define the GEMM type
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                           // ElementA
    cutlass::layout::RowMajor,                 // LayoutA
    cutlass::half_t,                           // ElementB
    cutlass::layout::RowMajor,                 // LayoutB
    cutlass::half_t,                           // ElementC
    cutlass::layout::RowMajor,                 // LayoutC
    float,                                     // ElementAccumulator
    cutlass::arch::OpClassTensorOp,            // OperatorClass
    cutlass::arch::Sm80,                       // ArchTag
    cutlass::gemm::GemmShape<128, 128, 32>,   // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 32>,     // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,      // InstructionShape
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t,
        128 / cutlass::sizeof_bits<cutlass::half_t>::value,
        float,
        float
    >,                                         // EpilogueOutputOp
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, // Swizzle
    3,                                         // Stages
    cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::Sm80,
        cutlass::half_t,
        cutlass::half_t,
        cutlass::half_t,
        float
    >::Operator,                               // Operator
    false                                      // SplitKSerial
>;
```

### Running the GEMM

```cpp
// Initialize the GEMM object
Gemm gemm_op;

// Define the problem size
int M = 1024;
int N = 512;
int K = 256;

// Create the arguments structure
typename Gemm::Arguments args(
    {M, N, K},        // Problem size (GemmCoord)
    {d_A, K},         // TensorRef for A: {pointer, leading dimension}
    {d_B, N},         // TensorRef for B: {pointer, leading dimension}
    {d_C, N},         // TensorRef for C: {pointer, leading dimension}
    {d_D, N},         // TensorRef for D: {pointer, leading dimension}
    {alpha, beta}     // Epilogue params: {alpha, beta}
);

// Check if the GEMM can be implemented for the given arguments
cutlass::Status status = gemm_op.can_implement(args);
if (status != cutlass::Status::kSuccess) {
    std::cerr << "GEMM cannot be implemented for these arguments" << std::endl;
    return;
}

// Initialize the GEMM (allocates workspace, configures kernel)
status = gemm_op.initialize(args);
if (status != cutlass::Status::kSuccess) {
    std::cerr << "GEMM initialization failed" << std::endl;
    return;
}

// Run the GEMM
status = gemm_op();
if (status != cutlass::Status::kSuccess) {
    std::cerr << "GEMM execution failed" << std::endl;
    return;
}

// Alternatively, initialize and run in one step with a stream:
status = gemm_op(args, nullptr, stream);
```

### Workspace Management

```cpp
// Get the required workspace size
size_t workspace_size = Gemm::get_workspace_size(args);

// Allocate workspace
void* workspace = nullptr;
if (workspace_size > 0) {
    cudaMalloc(&workspace, workspace_size);
}

// Initialize with workspace
status = gemm_op.initialize(args, workspace);

// Run
status = gemm_op();

// Clean up
cudaFree(workspace);
```

---

## Device API Variants

### GemmArray

Processes an array of GEMM problems with the same dimensions but different data pointers.

```cpp
#include "cutlass/gemm/device/gemm_array.h"

using GemmArray = cutlass::gemm::device::GemmArray<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,          cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>
>;

// Prepare array of pointers
int batch_count = 10;
cutlass::half_t* ptr_A[10];
cutlass::half_t* ptr_B[10];
float* ptr_C[10];
float* ptr_D[10];

// Fill ptr_A, ptr_B, ptr_C, ptr_D with device pointers...

GemmArray gemm_op;
typename GemmArray::Arguments args(
    {M, N, K},        // Problem size
    ptr_A, K,          // Array of A pointers + leading dim
    ptr_B, N,          // Array of B pointers + leading dim
    ptr_C, N,          // Array of C pointers + leading dim
    ptr_D, N,          // Array of D pointers + leading dim
    {alpha, beta},     // Epilogue params
    batch_count        // Number of GEMMs
);

gemm_op.initialize(args);
gemm_op();
```

### GemmBatched

Processes a batch of GEMMs with strided access (batch stride between matrices).

```cpp
#include "cutlass/gemm/device/gemm_batched.h"

using GemmBatched = cutlass::gemm::device::GemmBatched<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,          cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>
>;

int64_t batch_stride_A = M * K;  // stride between consecutive A matrices
int64_t batch_stride_B = K * N;
int64_t batch_stride_C = M * N;
int64_t batch_stride_D = M * N;

typename GemmBatched::Arguments args(
    {M, N, K},
    {d_A, K, batch_stride_A},    // ptr, leading dim, batch stride
    {d_B, N, batch_stride_B},
    {d_C, N, batch_stride_C},
    {d_D, N, batch_stride_D},
    {alpha, beta},
    batch_count
);
```

### GemmSplitKParallel

Splits the K dimension across multiple threadblocks and performs parallel reduction.

```cpp
#include "cutlass/gemm/device/gemm_splitk_parallel.h"

using GemmSplitK = cutlass::gemm::device::GemmSplitKParallel<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,          cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>
>;

int split_k_slices = 4;  // Split K into 4 parts

typename GemmSplitK::Arguments args(
    {M, N, K},
    {d_A, K},
    {d_B, N},
    {d_C, N},
    {d_D, N},
    {alpha, beta},
    split_k_slices  // Number of split-K slices
);

// Split-K requires workspace for partial results
size_t ws_size = GemmSplitK::get_workspace_size(args);
void* workspace;
cudaMalloc(&workspace, ws_size);

gemm_op.initialize(args, workspace);
gemm_op();
```

---

## Threadblock API

The Threadblock API defines the computation performed by a single CUDA threadblock. It is responsible for loading tiles of A and B from global memory to shared memory, computing MMA operations at the warp level, and storing results.

### MmaPipelined (Double-Buffered)

```cpp
#include "cutlass/gemm/threadblock/mma_pipelined.h"

// MmaPipelined uses two shared memory buffers to overlap
// global memory loads with MMA computation.
//
// Pipeline stages: 2 (double buffering)
//
// Stage 0: Load tile from global memory to smem buffer 0
// Stage 1: Compute MMA from smem buffer 0, Load next tile to smem buffer 1
// ... alternating

template <
  typename Shape_,           // Threadblock tile shape (M, N, K)
  typename IteratorA_,       // Iterator for loading A tiles
  typename SmemIteratorA_,   // Shared memory iterator for A
  typename IteratorB_,       // Iterator for loading B tiles
  typename SmemIteratorB_,   // Shared memory iterator for B
  typename Policy_           // Warp-level MMA policy
>
class MmaPipelined;
```

### MmaMultistage (Multi-Stage Pipeline)

```cpp
#include "cutlass/gemm/threadblock/mma_multistage.h"

// MmaMultistage uses multiple shared memory buffers with cp.async
// for deeper pipelines (SM80+).
//
// Stage count > 2: multi-stage pipelining
// Uses cp.async for asynchronous global-to-shared copies
// Better overlap of memory and computation on Ampere+

template <
  typename Shape_,           // Threadblock tile shape
  typename IteratorA_,       // Iterator for loading A tiles
  typename SmemIteratorA_,   // Shared memory iterator for A
  typename IteratorB_,       // Iterator for loading B tiles
  typename SmemIteratorB_,   // Shared memory iterator for B
  typename Policy_,          // Warp-level MMA policy
  int Stages_,               // Number of pipeline stages
  typename Enable = void
>
class MmaMultistage;
```

### IteratorA and IteratorB

These iterators handle loading tiles from global memory into shared memory:

```cpp
// Iterator for loading operand A tiles from global memory
// Transforms global memory layout to shared memory layout
//
// Key types:
//   - AccessType: the memory access width (e.g., 128 bits)
//   - ThreadMap: maps threads to elements in the tile
//   - Shape: the tile shape loaded per iteration

// Default iterator configuration for SM80:
using IteratorA = cutlass::gemm::threadblock::PredicatedTileAccessIterator<
    cutlass::gemm::GemmShape<128, 128, 32>,  // ThreadblockShape
    cutlass::half_t,                          // ElementA
    cutlass::layout::RowMajor,                // LayoutA
    0,                                        // Advance rank (0 = along M)
    cutlass::gemm::threadblock::PredicatedTileAccessIteratorParams<
        cutlass::gemm::threadblock::OutputTileOptimalThreadMap<
            cutlass::gemm::GemmShape<128, 32>,  // warp tile
            8,                                    // warps per threadblock
            4                                     // threads per warp
        >
    >
>;
```

### Shared Memory Management

```cpp
// Shared memory layout optimization:
// CUTLASS uses layout::ColumnMajorInterleaved or PitchLinear for shared
// memory to avoid bank conflicts during warp-level matrix loads.

// Shared memory is allocated as a union to allow double/multi-buffering:
union SharedStorage {
    typename Mma::SharedStorage mma_shared_storage;
    typename Epilogue::SharedStorage epilogue_shared_storage;
    // For multi-stage, there are multiple buffers:
    // typename Mma::SmemStorageA smem_A[Stages];
    // typename Mma::SmemStorageB smem_B[Stages];
};
```

---

## Warp API

The Warp API implements the per-warp matrix multiply-accumulate using either Tensor Core instructions or SIMT math.

### MmaTensorOp (Tensor Core)

```cpp
#include "cutlass/gemm/warp/mma_tensorop.h"

// MmaTensorOp: maps to Tensor Core MMA instructions
// Handles fragment loading, MMA execution, and accumulator management

template <
  typename Shape_,              // Warp-level GEMM shape (e.g., 64x64x32)
  typename Policy_,             // MMA instruction policy
  typename MmaOp_,              // Base MMA operation (e.g., ArchMmaOperator)
  typename FragmentIteratorA_,  // Iterator over A fragments
  typename FragmentIteratorB_,  // Iterator over B fragments
  typename AccumulatorLayout_   // Layout of accumulator fragments
>
class MmaTensorOp;

// Fragment types:
// FragmentA: holds a subset of the A tile in registers for one warp
// FragmentB: holds a subset of the B tile in registers for one warp
// FragmentC: accumulator fragment (typically float or int32_t)

// Fragment sizes depend on the MMA instruction:
// For HMMA.16816 (FP16 on SM80):
//   FragmentA = 4 x half_t (8 bytes) per instruction
//   FragmentB = 4 x half_t (8 bytes) per instruction
//   FragmentC = 4 x float (16 bytes) per instruction
```

### MmaSimt (SIMT)

```cpp
#include "cutlass/gemm/warp/mma_simt.h"

// MmaSimt: uses scalar (per-thread) FP32/FP64 math
// No Tensor Core usage; works on all architectures
// Each thread computes a small tile of the output

template <
  typename Shape_,           // Warp-level GEMM shape
  typename ElementA_,        // Element type of A
  typename LayoutA_,         // Layout of A in shared memory
  typename ElementB_,        // Element type of B
  typename LayoutB_,         // Layout of B in shared memory
  typename ElementC_,        // Accumulator type
  typename LayoutC_,         // Layout of C
  typename Policy_           // SIMT MMA policy
>
class MmaSimt;
```

### Warp-Level MMA Execution

```cpp
// Typical warp-level MMA usage (simplified):

__device__ void warp_mma_loop(
    // Fragments loaded from shared memory
    cutlass::Array<cutlass::half_t, 8> const& frag_A,
    cutlass::Array<cutlass::half_t, 8> const& frag_B,
    // Accumulator
    cutlass::Array<float, 4>& accum
) {
    // Execute one MMA instruction: accum += A * B
    // For HMMA.16816 on SM80:
    //   Inputs: 16x8 (A) x 8x16 (B) = 16x16 (C)
    //   Each thread holds a slice of A, B, and C

    // The actual MMA call (simplified representation):
    nvcuda::wmma::load_matrix_sync(frag_A, smem_ptr_A, stride_A);
    nvcuda::wmma::load_matrix_sync(frag_B, smem_ptr_B, stride_B);
    nvcuda::wmma::mma_sync(accum, frag_A, frag_B, accum);
}
```

### Fragment Types in Detail

```cpp
// FragmentA: register storage for A tile elements
// For TensorOp with HMMA.16816:
using FragmentA = cutlass::Array<cutlass::half_t, 4>;
// Contains elements of A needed by one thread for one MMA instruction

// FragmentB: register storage for B tile elements
using FragmentB = cutlass::Array<cutlass::half_t, 4>;

// FragmentC / Accumulator: register storage for accumulated results
using FragmentC = cutlass::Array<float, 4>;
// Each thread accumulates partial results across all K iterations

// The fragments are iterated over within the warp-level tile:
// WarpShape = GemmShape<64, 64, 32>
// InstructionShape = GemmShape<16, 8, 16>
// Number of instructions along M: 64 / 16 = 4
// Number of instructions along N: 64 / 8 = 8
// Number of instructions along K: 32 / 16 = 2
// Total instructions per warp: 4 * 8 * 2 = 64
```

---

## Thread API

The Thread API implements per-thread GEMM operations for SIMT mode or for very small tile sizes.

```cpp
#include "cutlass/gemm/thread/mma.h"

// Thread-level MMA: each thread computes a small GEMM
template <
  typename Shape_,           // Thread-level GEMM shape (e.g., 2x2x4)
  typename ElementA_,        // Element type of A
  typename LayoutA_,         // Layout of A
  typename ElementB_,        // Element type of B
  typename LayoutB_,         // Layout of B
  typename ElementC_,        // Accumulator type
  typename LayoutC_          // Layout of C
>
class Mma;

// Usage example (SIMT mode):
using ThreadMma = cutlass::gemm::thread::Mma<
    cutlass::gemm::GemmShape<2, 2, 4>,    // Shape: each thread computes 2x2 output
    cutlass::half_t,                       // ElementA
    cutlass::layout::RowMajor,             // LayoutA
    cutlass::half_t,                       // ElementB
    cutlass::layout::RowMajor,             // LayoutB
    float,                                 // ElementC
    cutlass::layout::RowMajor              // LayoutC
>;

// Each thread:
// 1. Loads its portion of A and B from registers/shared memory
// 2. Performs the dot product
// 3. Accumulates into its output fragment
```

---

## Epilogue Integration

The epilogue processes the accumulated results after the MMA loop completes. It handles scaling, bias addition, activation functions, and writing results to global memory.

### Standard Epilogue Pattern

```cpp
#include "cutlass/epilogue/threadblock/epilogue.h"

// The epilogue is templated on:
// - Output tile shape
// - Warp-level MMA accumulator layout
// - Output operator (e.g., LinearCombination)
// - Output element type

using Epilogue = cutlass::epilogue::threadblock::Epilogue<
    cutlass::gemm::GemmShape<128, 128, 32>,    // ThreadblockShape
    typename Gemm::WarpMma,                     // Warp-level MMA type
    128 / sizeof_bits<float>::value,             // Elements per access
    cutlass::half_t,                             // Output element type
    cutlass::layout::RowMajor                    // Output layout
>;
```

### Epilogue Execution Flow

```cpp
// In the kernel, after the MMA loop completes:

// 1. Warp-level results are in accumulator fragments
// FragmentC accum[M][N]; // accumulated results per warp

// 2. Epilogue processes these:
//    a. Loads C (if beta != 0)
//    b. Computes D = alpha * accum + beta * C
//    c. Optionally applies activation function
//    d. Writes D to global memory

// Simplified kernel structure:
__global__ void gemm_kernel(/* params */) {
    // ... MMA loop accumulates into accumulators ...

    // Epilogue: transform and store
    epilogue(output_ptr, output_layout, accumulators, source_ptr, source_layout);
}
```

---

## Threadblock Rasterization Strategies

Threadblock rasterization determines how threadblocks are mapped to output tiles. CUTLASS provides several swizzling strategies.

### GemmIdentityThreadblockSwizzle

```cpp
// Default: linear mapping, threadblock (i,j) maps to output tile (i,j)
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Grid dimensions:
// grid_dim.x = ceil(M / ThreadblockShape::kM)
// grid_dim.y = ceil(N / ThreadblockShape::kN)
```

### GemmBatchedIdentityThreadblockSwizzle

```cpp
// Adds batch dimension to the grid
using BatchSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle<>;

// Grid dimensions:
// grid_dim.x = ceil(M / ThreadblockShape::kM)
// grid_dim.y = ceil(N / ThreadblockShape::kN)
// grid_dim.z = batch_count
```

### GemmSplitKIdentityThreadblockSwizzle

```cpp
// Adds split-K dimension
using SplitKSwizzle = cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>;

// Grid dimensions:
// grid_dim.x = ceil(M / ThreadblockShape::kM) * ceil(N / ThreadblockShape::kN)
// grid_dim.y = split_k_slices
```

### Custom Swizzle for Bank Conflict Avoidance

```cpp
// Some swizzles reorder threadblocks to improve L2 cache locality
// or shared memory bank conflict patterns

// The swizzle pattern remaps the threadblock index:
// tb_idx = swizzle(grid_idx)
// where grid_idx is the linear threadblock index
```

---

## Split-K Parallel Reduction

Split-K divides the K dimension across multiple threadblocks, each computing a partial result that is then reduced.

### How Split-K Works

```
Full GEMM:    C += A[M,K] * B[K,N]

Split-K (2):  C += A[M,K/2] * B[K/2,N]    (threadblock group 0)
            + A[M,K/2] * B[K/2,N]    (threadblock group 1, offset K/2)

Reduction:    D[i,j] = sum over all split-K slices of partial_C[i,j]
```

### Using Split-K

```cpp
// Enable split-K serial in the Gemm template
using GemmSplitK = cutlass::gemm::device::Gemm<
    /* ... same as before ... */,
    /* SplitKSerial = */ true
>;

// Or use GemmSplitKParallel which includes reduction:
using GemmSK = cutlass::gemm::device::GemmSplitKParallel<
    /* ... element/layout types ... */,
    /* ... accumulator type ... */,
    /* ... op class, arch, shapes ... */
>;

// Arguments include split_k_slices:
typename GemmSK::Arguments args(
    {M, N, K},
    {d_A, K},
    {d_B, N},
    {d_C, N},
    {d_D, N},
    {alpha, beta},
    4  // split_k_slices: divide K into 4 parts
);
```

### Split-K Reduction Kernel

```cpp
// The reduction kernel accumulates partial results:
// D = alpha * sum(partial_C[slice]) + beta * C

// CUTLASS provides:
// cutlass::reduction::device::ReduceSplitK
// cutlass::reduction::thread::ReduceAdd

// Workspace needed:
// sizeof(ElementC) * M * N * split_k_slices
```

---

## Complete GEMM Example

### Full Working Example: FP16 GEMM on SM80

```cpp
#include <iostream>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"

// Type definitions
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

// GEMM configuration
using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementC,
        128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator,
        ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
>;

int main() {
    // Problem dimensions
    int M = 5120;
    int N = 4096;
    int K = 2048;

    // Scalar parameters
    float alpha = 1.0f;
    float beta = 0.0f;

    // Allocate tensors
    cutlass::HostTensor<ElementA, LayoutA> tensor_a({M, K});
    cutlass::HostTensor<ElementB, LayoutB> tensor_b({K, N});
    cutlass::HostTensor<ElementC, LayoutC> tensor_c({M, N});
    cutlass::HostTensor<ElementC, LayoutC> tensor_d({M, N});

    // Initialize input data
    cutlass::reference::device::BlockFillSequential(
        tensor_a.device_data(), tensor_a.capacity(), ElementA(1), ElementA(2));
    cutlass::reference::device::BlockFillSequential(
        tensor_b.device_data(), tensor_b.capacity(), ElementB(1), ElementB(2));
    cutlass::reference::device::BlockFillSequential(
        tensor_c.device_data(), tensor_c.capacity(), ElementC(0), ElementC(0));

    // Create arguments
    typename Gemm::Arguments args(
        {M, N, K},
        {tensor_a.device_ref()},     // TensorRef for A (uses packed stride)
        {tensor_b.device_ref()},     // TensorRef for B
        {tensor_c.device_ref()},     // TensorRef for C (source)
        {tensor_d.device_ref()},     // TensorRef for D (destination)
        {alpha, beta}                // Epilogue parameters
    );

    // Verify the GEMM can be implemented
    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "This GEMM configuration is not supported." << std::endl;
        return -1;
    }

    // Initialize and run
    status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to initialize GEMM." << std::endl;
        return -1;
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to run GEMM." << std::endl;
        return -1;
    }

    // Wait for completion
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "GEMM completed successfully." << std::endl;
    std::cout << "  Problem: " << M << " x " << N << " x " << K << std::endl;
    std::cout << "  A: " << sizeof(ElementA) * 8 << "-bit, Row-Major" << std::endl;
    std::cout << "  B: " << sizeof(ElementB) * 8 << "-bit, Row-Major" << std::endl;
    std::cout << "  C/D: " << sizeof(ElementC) * 8 << "-bit, Row-Major" << std::endl;

    return 0;
}
```

### BFloat16 GEMM Example (SM80)

```cpp
// BFloat16 uses the same structure but with bfloat16 element types
using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = float;
using ElementAccumulator = float;

// Instruction shape for BF16 on SM80 is the same as FP16:
// GemmShape<16, 8, 16>

using GemmBF16 = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementC,
        128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator,
        ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
>;
```

### INT8 GEMM Example (SM80)

```cpp
using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = int32_t;
using ElementAccumulator = int32_t;

// INT8 uses different instruction shapes:
// IMMA.8816: GemmShape<8, 8, 16> or GemmShape<16, 8, 32>

using GemmINT8 = cutlass::gemm::device::Gemm<
    ElementA, cutlass::layout::ColumnMajorInterleaved<32>,  // Interleaved for INT8
    ElementB, cutlass::layout::RowMajorInterleaved<32>,
    ElementC, cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<
        ElementC,
        128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator,
        ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
>;
```

---

## Summary: 2.x API Quick Reference

### Device API Class Hierarchy

| Class | Purpose |
|---|---|
| `Gemm` | Single GEMM operation |
| `GemmArray` | Array of GEMMs with different data pointers |
| `GemmBatched` | Batched GEMM with strided access |
| `GemmSplitKParallel` | GEMM with split-K and parallel reduction |
| `GemmComplex` | Complex number GEMM |
| `GemmGrouped` | Grouped GEMM (variable problem sizes) |

### Key Operations

| Method | Description |
|---|---|
| `can_implement(args)` | Check if arguments are supported |
| `get_workspace_size(args)` | Get required workspace bytes |
| `initialize(args, workspace, stream)` | Initialize kernel state |
| `run(stream)` | Launch the kernel |
| `operator()(args, workspace, stream)` | Initialize + run |
| `get_gemm_k_split_lhs()` | Get LHS split-K factor |
| `get_gemm_k_split_rhs()` | Get RHS split-K factor |

### Shape Hierarchy Constraints

```
ThreadblockShape must be divisible by WarpShape
WarpShape must be divisible by InstructionShape
InstructionShape is determined by the architecture and element type
```

The CUTLASS 2.x API provides a clean hierarchical decomposition of GEMM that closely maps to GPU hardware, enabling users to customize at any level while providing sensible defaults at the Device API level.
