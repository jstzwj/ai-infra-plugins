# CUTLASS 3.x GEMM API

CUTLASS 3.x introduces a fundamentally redesigned GEMM API built on the CuTe layout and tensor algebra library. The 3.x API replaces the rigid 2.x hierarchy with a more flexible, composable architecture that supports new hardware features like TMA (Tensor Memory Accelerator), warp specialization, and cluster-level cooperation on Hopper (SM90) and later architectures.

---

## Programming Model Hierarchy

CUTLASS 3.x reorganizes the GEMM hierarchy into a cleaner, more composable structure:

```
Device / Host API
  GemmUniversalAdapter  -- Host-facing launcher and configuration
    |
  Kernel
    GemmUniversal       -- Device kernel entry point
      |
    Collective           -- Mainloop + Epilogue combined logic
      |
    Tiled MMA / Copy     -- CuTe-based tiled operations
      |
    Atom                 -- Hardware instruction mapping (GMMA, TMA, etc.)
```

### Comparison with 2.x

| Concept | CUTLASS 2.x | CUTLASS 3.x |
|---|---|---|
| Host interface | `device::Gemm<...>` | `GemmUniversalAdapter<...>` |
| Kernel | Implicit in device class | `GemmUniversal<...>` |
| Mainloop | Threadblock Mma (MmaPipelined/Multistage) | Collective operation |
| Data movement | Iterators (PredicatedTileIterator) | CuTe Copy Atoms + TMA |
| MMA | Warp MmaTensorOp | CuTe MMA Atoms (GMMA) |
| Layout | Layout policies | CuTe Layouts (Shape + Stride) |
| Epilogue | Epilogue threadblock | Collective Epilogue |

---

## CollectiveBuilder: Convenient Kernel Assembly

The `CollectiveBuilder` is the primary interface for assembling GEMM kernels in CUTLATCH 3.x. It selects the optimal collective operation (mainloop + epilogue) based on architecture, data types, and user preferences.

### Template Signature

```cpp
namespace cutlass::gemm::collective {

template <
  typename ArchTag_,           // Target architecture (e.g., arch::Sm90)
  typename OpClass_,           // Operator class (OpClassTensorOp)
  typename ElementA_,          // Element type for operand A
  typename LayoutA_,           // Layout for A (CuTe layout or CUTLASS layout)
  int AlignmentA,              // Memory alignment for A (in elements)
  typename ElementB_,          // Element type for operand B
  typename LayoutB_,           // Layout for B
  int AlignmentB,              // Memory alignment for B (in elements)
  typename ElementAccumulator_,// Accumulator element type
  typename TileShape_,         // CTA tile shape (GemmShape or cute::Shape)
  typename StageCount_,        // Number of pipeline stages (int or cute::Int<N>)
  typename KernelSchedule_     // Kernel dispatch policy
>
struct CollectiveBuilder;
};
```

### Parameters in Detail

#### ArchTag

```cpp
// Specifies the target SM architecture
using ArchTag = cutlass::arch::Sm80;   // Ampere
using ArchTag = cutlass::arch::Sm90;   // Hopper
using ArchTag = cutlass::arch::Sm100;  // Blackwell
```

#### OpClass

```cpp
// Only OpClassTensorOp is supported in 3.x for SM90+
using OpClass = cutlass::arch::OpClassTensorOp;
```

#### Element and Layout Types

```cpp
// Element types
using ElementA = cutlass::half_t;       // FP16
using ElementB = cutlass::half_t;       // FP16
using ElementC = cutlass::half_t;       // Output type
using ElementAccum = float;             // Accumulator type

// Other supported types:
// cutlass::bfloat16_t    -- BFloat16
// cutlass::tfloat32_t    -- TF32
// float                  -- FP32
// int8_t                 -- INT8
// cutlass::float_e4m3_t  -- FP8 (E4M3)
// cutlass::float_e5m2_t  -- FP8 (E5M2)

// Layout types
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
```

#### Alignment

```cpp
// Alignment specifies the guaranteed memory alignment in elements
// This must be a power of 2 and determines the minimum contiguity
// of the innermost dimension.

// For FP16 with 128-bit TMA access: 128 / 16 = 8 elements
int AlignmentA = 8;
int AlignmentB = 8;

// For FP32 with 128-bit access: 128 / 32 = 4 elements
// int AlignmentA = 4;

// For FP8 with 128-bit access: 128 / 8 = 16 elements
// int AlignmentA = 16;

// Use 1 if alignment is unknown (may reduce performance)
// int AlignmentA = 1;
```

#### TileShape

```cpp
// TileShape defines the CTA-level tile dimensions
// Can be specified as cutlass::gemm::GemmShape or cute::Shape

// Using GemmShape (M, N, K):
using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;

// Using cute::Shape (more flexible, can encode compile-time shapes):
// auto tile_shape = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<64>>{};

// Common tile shapes for SM90:
// GemmShape<128, 128, 64>   -- small tile, fits in small shared memory
// GemmShape<128, 256, 64>   -- larger N dimension
// GemmShape<256, 128, 64>   -- larger M dimension
// GemmShape<64, 64, 128>    -- small M/N, large K for reduction-heavy workloads
```

#### StageCount

```cpp
// StageCount can be specified in several ways:

// 1. Compile-time constant (most efficient)
using StageCount = cutlass::gemm::collective::StageCountAuto;  // Let CUTLASS decide

// 2. Explicit compile-time value
using StageCount = cute::Int<3>;  // 3 stages

// 3. Carveout-based (reserve shared memory for epilogue)
// StageCountAutoCarveout<bytes>: compute stages from remaining smem
using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<sizeof(Epilogue)>;
```

#### KernelSchedule

```cpp
// KernelSchedule determines the mainloop dispatch policy:

// Automatic selection (recommended for most users):
using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;

// SM90-specific schedules:
using KernelSchedule = cutlass::gemm::KernelTma;                              // Basic TMA
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;               // Warp-specialized
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;       // Ping-pong buffering
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;    // Cooperative across warps

// SM80 schedules:
using KernelSchedule = cutlass::gemm::KernelMultistage;
```

### CollectiveBuilder Usage

```cpp
#include "cutlass/gemm/collective/collective_builder.hpp"

// Define the collective operation
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,                        // ArchTag
    cutlass::arch::OpClassTensorOp,             // OpClass
    cutlass::half_t,                            // ElementA
    cutlass::layout::RowMajor,                  // LayoutA
    8,                                          // AlignmentA
    cutlass::half_t,                            // ElementB
    cutlass::layout::RowMajor,                  // LayoutB
    8,                                          // AlignmentB
    float,                                      // ElementAccumulator
    cutlass::gemm::GemmShape<128, 128, 64>,    // TileShape
    cutlass::gemm::collective::StageCountAuto,  // StageCount
    cutlass::gemm::collective::KernelScheduleAuto // KernelSchedule
>::CollectiveOp;

// The builder selects the optimal implementation based on the above parameters
// For SM90 with FP16 inputs: likely KernelTmaWarpSpecialized with GMMA
```

### Automatic vs Manual Configuration

```cpp
// AUTOMATIC (recommended):
// Use StageCountAuto and KernelScheduleAuto to let CUTLASS pick the best configuration
using CollectiveAuto = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccum,
    TileShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

// MANUAL (for expert tuning):
// Specify exact stage count and kernel schedule
using CollectiveManual = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccum,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cute::Int<3>,                                  // Exactly 3 stages
    cutlass::gemm::KernelTmaWarpSpecializedCooperative  // Cooperative schedule
>::CollectiveOp;
```

---

## GemmUniversal Kernel

`GemmUniversal` is the device kernel that orchestrates the GEMM computation. It handles grid mapping, tile distribution, and invokes the collective operation.

### Template Signature

```cpp
namespace cutlass::gemm::kernel {

template <
  typename ProblemShape_,         // Shape of the full problem
  typename CollectiveMainloop_,   // Collective mainloop operation
  typename CollectiveEpilogue_,   // Collective epilogue operation
  typename TileScheduler_         // How tiles are scheduled to CTAs
>
struct GemmUniversal;

} // namespace cutlass::gemm::kernel
```

### Operation Modes

GemmUniversal supports multiple modes of operation:

```cpp
enum class GemmUniversalMode {
    kGemm,                  // Standard GEMM: D = alpha * A * B + beta * C
    kGemmSplitKParallel,    // Split-K: parallel reduction across K slices
    kBatched,               // Batched GEMM: same dims, different data
    kArray,                 // Array GEMM: array of pointers
    kGrouped,               // Grouped GEMM: variable problem sizes
};
```

### kGemm Mode

```cpp
// Standard single GEMM
// Problem shape: cute::Shape<M, N, K>
auto problem_shape = cute::make_shape(M, N, K);

// Arguments
typename GemmKernel::Arguments args {
    problem_shape,
    mainloop_args,    // {ptr_A, ptr_B, ...}
    epilogue_args,    // {ptr_C, ptr_D, alpha, beta, ...}
    /* mode = */ cutlass::gemm::kernel::GemmUniversalMode::kGemm
};
```

### kBatched Mode

```cpp
// Batched GEMM with strided access
// Problem shape: cute::Shape<M, N, K, BatchCount>
auto problem_shape = cute::make_shape(M, N, K, batch_count);

// Strides include batch dimension
typename GemmKernel::Arguments args {
    cute::make_shape(M, N, K, batch_count),
    {
        ptr_A, stride_a, batch_stride_a,  // A with batch stride
        ptr_B, stride_b, batch_stride_b,  // B with batch stride
    },
    {
        ptr_C, stride_c, batch_stride_c,  // C with batch stride
        ptr_D, stride_d, batch_stride_d,  // D with batch stride
        alpha, beta
    },
    cutlass::gemm::kernel::GemmUniversalMode::kBatched
};
```

### kGemmSplitKParallel Mode

```cpp
// Split-K parallel reduction
auto problem_shape = cute::make_shape(M, N, K);

typename GemmKernel::Arguments args {
    problem_shape,
    mainloop_args,
    epilogue_args,
    cutlass::gemm::kernel::GemmUniversalMode::kGemmSplitKParallel,
    /* split_k_slices = */ 4
};
```

### kArray Mode

```cpp
// Array of GEMM pointers
// Problem shape includes batch count
auto problem_shape = cute::make_shape(M, N, K, batch_count);

// Pass arrays of pointers
// ptr_A is ElementA**, ptr_B is ElementB**, etc.
```

### kGrouped Mode

```cpp
// Grouped GEMM with variable problem sizes
// Each problem in the group can have different M, N, K
// Problem shape: GroupProblemShape (array of shapes)
```

### Arguments Structure

```cpp
// The Arguments structure for GemmUniversal
struct Arguments {
    ProblemShape problem_shape;          // Full problem dimensions
    typename CollectiveMainloop::Arguments mainloop;   // Mainloop params
    typename CollectiveEpilogue::Arguments epilogue;    // Epilogue params
    GemmUniversalMode mode;             // Operation mode
    int split_k_slices = 1;             // Split-K factor (default 1)
};
```

### Grid Tiling and Launch Configuration

```cpp
// GemmUniversal computes the grid dimensions from the tile shape and problem shape
// Grid dim = ceil(problem_M / tile_M) x ceil(problem_N / tile_N) x [extra dims]

// For standard GEMM:
// grid.x = ceil(M / TileShape::kM)
// grid.y = ceil(N / TileShape::kN)
// grid.z = 1

// For batched GEMM:
// grid.z = batch_count

// For split-K:
// grid.y *= split_k_slices  (or grid.z)
```

---

## GemmUniversalAdapter: Host-Facing Interface

`GemmUniversalAdapter` is the host-facing wrapper around `GemmUniversal`. It provides the same interface pattern as CUTLASS 2.x Device GEMM classes for familiarity, while internally managing 3.x-specific features.

### Template Signature

```cpp
namespace cutlass::gemm::device {

template <typename GemmKernel_>
class GemmUniversalAdapter;

} // namespace cutlass::gemm::device
```

### Static Methods

```cpp
// can_implement: check if the given arguments are supported
static cutlass::Status can_implement(Arguments const& args);

// get_workspace_size: compute required workspace in bytes
static size_t get_workspace_size(Arguments const& args);

// get_grid_shape: compute the kernel launch grid dimensions
static dim3 get_grid_shape(Arguments const& args);

// get_block_shape: get the CTA (threadblock) dimensions
static dim3 get_block_shape();
```

### Stateful Methods

```cpp
// Constructor: optionally takes the CUDA stream
GemmUniversalAdapter(cudaStream_t stream = nullptr);

// initialize: set up the kernel with arguments and workspace
cutlass::Status initialize(
    Arguments const& args,
    void* workspace = nullptr,
    cudaStream_t stream = nullptr
);

// run: launch the kernel
cutlass::Status run(cudaStream_t stream = nullptr);

// operator(): convenience method for initialize + run
cutlass::Status operator()(
    Arguments const& args,
    void* workspace = nullptr,
    cudaStream_t stream = nullptr
);

// update: update pointers/arguments without full reinitialization
// Useful for changing input/output pointers between runs
cutlass::Status update(
    Arguments const& args,
    void* workspace = nullptr
);
```

### Complete Example: GemmUniversalAdapter

```cpp
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/epilogue_builder.hpp"

// Step 1: Define the collective mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,   // A: type, layout, alignment
    cutlass::half_t, cutlass::layout::RowMajor, 8,   // B: type, layout, alignment
    float,                                             // Accumulator type
    cutlass::gemm::GemmShape<128, 128, 64>,          // Tile shape
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

// Step 2: Define the collective epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::EpilogueBuilder<
    cutlass::arch::Sm90,
    cutlass::gemm::GemmShape<128, 128, 64>,  // Must match TileShape
    CollectiveMainloop,                        // Mainloop type for accumulator layout
    cutlass::half_t,                           // ElementC (source)
    cutlass::layout::RowMajor,                 // LayoutC
    8,                                         // AlignmentC
    cutlass::half_t,                           // ElementD (output)
    cutlass::layout::RowMajor,                 // LayoutD
    8,                                         // AlignmentD
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

// Step 3: Define the kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,   // ProblemShape: M, N, K, L (L=1 for non-batched)
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::PersistentTileScheduler  // or AutoTileScheduler
>;

// Step 4: Define the adapter (host interface)
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Step 5: Use it
int M = 4096, N = 3072, K = 2048;

typename Gemm::Arguments args {
    cute::make_shape(M, N, K, 1),              // problem_shape (L=1 for non-batched)
    { d_A, stride_A, d_B, stride_B },          // mainloop args
    { d_C, stride_C, d_D, stride_D, alpha, beta } // epilogue args
};

Gemm gemm_op;
cutlass::Status status = gemm_op.can_implement(args);
status = gemm_op.initialize(args);
status = gemm_op.run();
```

### 3.x-Specific Features: TileShape and ClusterShape

```cpp
// TileShape: defines the per-CTA tile dimensions
// This replaces ThreadblockShape from 2.x
using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;

// ClusterShape: defines the number of CTAs that cooperate in a cluster (SM90+)
// Clusters enable cooperative sharing of data through distributed shared memory
using ClusterShape = cutlass::gemm::GemmShape<1, 1, 1>;   // No clustering (default)
using ClusterShape = cutlass::gemm::GemmShape<2, 1, 1>;   // 2 CTAs along M
using ClusterShape = cutlass::gemm::GemmShape<1, 2, 1>;   // 2 CTAs along N
using ClusterShape = cutlass::gemm::GemmShape<2, 2, 1>;   // 2x2 cluster

// ClusterShape affects:
// - Grid launch configuration (must align to cluster dimensions)
// - Shared memory layout (distributed across cluster)
// - Synchronization barriers
```

---

## Dispatch Policies

Dispatch policies determine how the kernel mainloop is implemented, including data movement strategy, pipeline depth, and warp specialization.

### KernelMultistage (SM80)

```cpp
// Multi-stage pipeline using cp.async for global-to-shared copies
// Compatible with SM80 Ampere architecture
//
// Features:
// - Multiple shared memory buffers (stages)
// - cp.async for asynchronous global-to-shared memory copies
// - warp-level MMA using HMMA/IMMA instructions
// - No TMA support

using KernelSchedule = cutlass::gemm::KernelMultistage;
// Typical configuration:
//   ArchTag = Sm80
//   TileShape = GemmShape<128, 128, 32>
//   Stages = 3-5
//   InstructionShape = GemmShape<16, 8, 16> (for FP16)
```

### KernelTma (SM90, Basic TMA)

```cpp
// Basic TMA-based kernel
// Uses Tensor Memory Accelerator for global-to-shared copies
//
// Features:
// - TMA bulk copy for efficient data movement
// - Single-warp-group operation (no specialization)
// - Simpler control flow

using KernelSchedule = cutlass::gemm::KernelTma;
// Usage: all warps participate in both loads and MMA
```

### KernelTmaWarpSpecialized (SM90)

```cpp
// Warp-specialized TMA kernel
// Divides warps into two groups:
//   - DMA warps: handle TMA loads from global to shared memory
//   - MMA warps: handle GMMA computation
//
// Features:
// - Overlaps TMA loads with GMMA computation
// - Higher throughput than basic KernelTma
// - Standard buffering (single buffer per stage)

using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
```

### KernelTmaWarpSpecializedPingpong (SM90)

```cpp
// Ping-pong warp-specialized kernel
// Alternates between two sets of buffers for maximum overlap
//
// Features:
// - Double buffering with ping-pong pattern
// - DMA warps fill buffer A while MMA warps compute from buffer B
// - Then swap: DMA fills B, MMA computes from A
// - Maximum overlap of memory and compute

using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
// Best for compute-bound workloads with enough shared memory
```

### KernelTmaWarpSpecializedCooperative (SM90)

```cpp
// Cooperative warp-specialized kernel
// Multiple warp groups cooperate on the same MMA tile
//
// Features:
// - 2 or more warp groups cooperate on MMA
// - Useful for large tile shapes that exceed single warp-group capacity
// - Enables larger effective tile sizes without increasing shared memory proportionally

using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
// Best for large tiles: GemmShape<256, 128, 64> or GemmShape<128, 256, 64>
```

### KernelTmaWarpSpecializedFP8Blockwise

```cpp
// FP8-specific warp-specialized kernel with blockwise scaling
// Designed for FP8 (E4M3/E5M2) data types
//
// Features:
// - Blockwise scaling factors for FP8 quantization
// - TMA-based loads with scale factor handling
// - Optimized for the reduced dynamic range of FP8

using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8Blockwise;
// For FP8 inputs: ElementA = cutlass::float_e4m3_t, ElementB = cutlass::float_e4m3_t
```

---

## Epilogue Integration in 3.x

CUTLASS 3.x uses a collective epilogue design that mirrors the collective mainloop.

### EpilogueBuilder

```cpp
#include "cutlass/epilogue/collective/epilogue_builder.hpp"

using CollectiveEpilogue = typename cutlass::epilogue::collective::EpilogueBuilder<
    cutlass::arch::Sm90,                        // ArchTag
    cutlass::gemm::GemmShape<128, 128, 64>,    // TileShape (must match mainloop)
    CollectiveMainloop,                          // Mainloop type
    cutlass::half_t,                             // ElementC (source)
    cutlass::layout::RowMajor,                   // LayoutC
    8,                                           // AlignmentC
    cutlass::half_t,                             // ElementD (destination/output)
    cutlass::layout::RowMajor,                   // LayoutD
    8,                                           // AlignmentD
    cutlass::epilogue::collective::EpilogueScheduleAuto  // Schedule policy
>::CollectiveOp;
```

### Epilogue Schedules

```cpp
// Auto: let CUTLASS pick the best schedule
using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;

// SM90-specific epilogue schedules:
// - TMA store-based epilogue
// - Warp-specialized epilogue
// - Sub-byte compression epilogue (for FP8, INT8)
```

### Epilogue Arguments

```cpp
// The epilogue arguments are embedded in the kernel arguments
struct EpilogueArgs {
    ElementC const* ptr_C;      // Source matrix C pointer
    typename LayoutC::Stride stride_C;  // C stride
    ElementD* ptr_D;            // Destination matrix D pointer
    typename LayoutD::Stride stride_D;  // D stride
    ElementCompute alpha;        // Scaling factor alpha
    ElementCompute beta;         // Scaling factor beta
};
```

---

## Batched and Grouped GEMM Support

### Batched GEMM

```cpp
// Batched GEMM: same M, N, K for all problems in the batch
int M = 1024, N = 1024, K = 512;
int batch_count = 16;

// Problem shape includes batch dimension
auto problem_shape = cute::make_shape(M, N, K, batch_count);

typename Gemm::Arguments args {
    problem_shape,
    {
        ptr_A, dA, batch_stride_A,    // mainloop A args
        ptr_B, dB, batch_stride_B     // mainloop B args
    },
    {
        ptr_C, dC, batch_stride_C,    // epilogue C args
        ptr_D, dD, batch_stride_D,    // epilogue D args
        alpha, beta
    },
    cutlass::gemm::kernel::GemmUniversalMode::kBatched
};
```

### Grouped GEMM

```cpp
// Grouped GEMM: variable M, N, K for each problem
// Each problem has its own dimensions

// Problem shape type: GroupProblemShape
// Contains arrays of (M, N, K) for each problem

using GroupProblemShape = cutlass::gemm::GroupProblemShape<
    cute::Shape<int, int, int>  // (M, N, K) per problem
>;

// Arguments include:
// - problem_shapes: array of (M, N, K)
// - ptr_A, ptr_B, ptr_C, ptr_D: arrays of pointers (one per problem)
// - lda, ldb, ldc, ldd: arrays of leading dimensions
```

### MoE (Mixture of Experts) GEMM

```cpp
// MoE GEMM: specialized grouped GEMM where M can vary per expert
// but N and K are fixed across all experts

using MoEProblemShape = cutlass::gemm::MoEProblemShape<
    cute::Shape<int, int, int>  // (M_i, N, K) where M_i varies
>;

// Arguments:
// - problem_shapes: (M_i, N, K) for each expert
// - All N and K values must be the same
// - M can differ across experts
```

---

## Problem Shapes

### GemmShape

```cpp
// Standard GEMM problem shape
using ProblemShape = cute::Shape<int, int, int>;  // (M, N, K)

// Create at runtime
auto problem = cute::make_shape(M, N, K);

// Access dimensions
auto m = cute::get<0>(problem);
auto n = cute::get<1>(problem);
auto k = cute::get<2>(problem);
```

### GroupProblemShape

```cpp
// Grouped GEMM: each problem has its own (M, N, K)
// Stored as separate arrays: M_array, N_array, K_array
struct GroupProblemShape {
    int num_problems;
    int const* M;  // device array of M values
    int const* N;  // device array of N values
    int const* K;  // device array of K values
};
```

### MoEProblemShape

```cpp
// MoE: variable M, fixed N and K
struct MoEProblemShape {
    int num_experts;
    int N, K;              // Fixed across all experts
    int const* M;          // device array of M values (one per expert)
    int64_t const* offset_C; // device array of output offsets
};
```

---

## Complete Code Examples

### Example 1: Basic FP16 GEMM on SM90

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/epilogue_builder.hpp"

// Define types
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccum = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
using ClusterShape = cutlass::gemm::GemmShape<1, 1, 1>;

// Build the collective mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 8,
    ElementB, LayoutB, 8,
    ElementAccum,
    TileShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

// Build the collective epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::EpilogueBuilder<
    cutlass::arch::Sm90,
    TileShape,
    CollectiveMainloop,
    ElementC, LayoutC, 8,
    ElementD, LayoutD, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

// Define the kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::PersistentTileScheduler
>;

// Define the adapter
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Run the GEMM
int M = 4096, N = 3072, K = 2048;
float alpha = 1.0f, beta = 0.0f;

typename Gemm::Arguments args {
    cute::make_shape(M, N, K, 1),
    { d_A, LayoutA::packed({M, K}).stride(0),
      d_B, LayoutB::packed({K, N}).stride(0) },
    { d_C, LayoutC::packed({M, N}).stride(0),
      d_D, LayoutD::packed({M, N}).stride(0),
      alpha, beta }
};

Gemm gemm_op;
auto status = gemm_op(args);
```

### Example 2: Batched FP16 GEMM

```cpp
int M = 1024, N = 1024, K = 512;
int batch_count = 8;
float alpha = 1.0f, beta = 0.0f;

int64_t batch_stride_A = M * K;
int64_t batch_stride_B = K * N;
int64_t batch_stride_C = M * N;
int64_t batch_stride_D = M * N;

typename Gemm::Arguments args {
    cute::make_shape(M, N, K, batch_count),
    {
        d_A, LayoutA::packed({M, K}).stride(0), batch_stride_A,
        d_B, LayoutB::packed({K, N}).stride(0), batch_stride_B
    },
    {
        d_C, LayoutC::packed({M, N}).stride(0), batch_stride_C,
        d_D, LayoutD::packed({M, N}).stride(0), batch_stride_D,
        alpha, beta
    },
    cutlass::gemm::kernel::GemmUniversalMode::kBatched
};

Gemm gemm_op;
gemm_op.initialize(args);
gemm_op.run();
```

### Example 3: BF16 GEMM with Bias Epilogue

```cpp
using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccum = float;
using ElementBias = cutlass::bfloat16_t;

// Use a bias epilogue by selecting the appropriate epilogue visitor
// In 3.x, bias addition is handled through epilogue fusion

// The epilogue arguments would include a bias pointer:
// { ptr_C, stride_C, ptr_D, stride_D, alpha, beta, ptr_bias, stride_bias }
```

---

## Migration Guide from 2.x to 3.x

### Template Parameter Mapping

| 2.x Parameter | 3.x Equivalent |
|---|---|
| `device::Gemm<...>` | `GemmUniversalAdapter<GemmUniversal<...>>` |
| `ThreadblockShape` | `TileShape` |
| `WarpShape` | Determined automatically by CollectiveBuilder |
| `InstructionShape` | Determined automatically (GMMA atom selection) |
| `Stages` | `StageCount` or `StageCountAuto` |
| `EpilogueOutputOp` | `CollectiveEpilogue` via EpilogueBuilder |
| `ThreadblockSwizzle` | `TileScheduler` (PersistentTileScheduler, etc.) |
| `SplitKSerial` | `GemmUniversalMode::kGemmSplitKParallel` |

### API Call Mapping

```cpp
// 2.x:
cutlass::gemm::device::Gemm<...> gemm_op;
typename Gemm::Arguments args({M, N, K}, {ptr_A, lda}, {ptr_B, ldb},
                               {ptr_C, ldc}, {ptr_D, ldd}, {alpha, beta});
gemm_op.initialize(args);
gemm_op();

// 3.x:
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
Gemm gemm_op;
typename Gemm::Arguments args {
    cute::make_shape(M, N, K, 1),
    { ptr_A, lda, ptr_B, ldb },
    { ptr_C, ldc, ptr_D, ldd, alpha, beta }
};
gemm_op.initialize(args);
gemm_op();
```

### Key Differences

1. **No WarpShape**: CUTLASS 3.x infers warp-level shapes automatically from the TileShape and architecture.

2. **No InstructionShape**: The GMMA atom selection is handled by the CollectiveBuilder based on element types and architecture.

3. **CuTe Layouts**: Internal data layout uses CuTe tensor algebra instead of the older Layout policies.

4. **TMA instead of cp.async**: SM90+ uses TMA for data movement, which is fundamentally different from the SM80 cp.async pipeline.

5. **Cluster-level cooperation**: New in 3.x, CTAs can form clusters and share distributed shared memory.

6. **Warp specialization**: SM90 divides warps into specialized roles (DMA warps vs MMA warps) for better overlap.

7. **Persistent tiles**: The 3.x tile scheduler supports persistent kernels where CTAs dynamically pick up tiles, improving load balancing.

8. **Arguments structure**: The arguments are more structured with separate mainloop and epilogue parameter bundles.

---

## Summary

The CUTLASS 3.x API represents a significant evolution from 2.x:

- **CollectiveBuilder** handles automatic kernel configuration, selecting the best dispatch policy based on architecture and data types.
- **GemmUniversal** provides a unified kernel supporting multiple modes (GEMM, batched, split-K, array, grouped).
- **GemmUniversalAdapter** provides a familiar host interface while leveraging all 3.x features.
- **Dispatch policies** (KernelMultistage, KernelTma*, etc.) map directly to hardware capabilities.
- **CuTe tensors and layouts** provide a more expressive and composable foundation for all data access patterns.

For new projects targeting SM90+ hardware, CUTLASS 3.x is the recommended API. For SM80 and earlier, CUTLASS 2.x remains the appropriate choice, though 3.x also supports SM80 via the KernelMultistage dispatch policy.
