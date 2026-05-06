# CUTLASS Terminology - Chapter 38: Comprehensive Glossary

This reference provides a comprehensive glossary of all CUTLASS terminology, including core concepts, architecture-specific terms, CuTe terms, and deprecated 2.x terminology with their 3.x equivalents.

---

## 38.1 Core Concepts

### Capacity

**Definition**: The physical storage requirement (in bytes or elements) needed to hold a tensor or data structure.

Capacity differs from the logical size of a tensor because of alignment requirements, padding, and stride considerations. For a tensor with shape `(M, N)` and stride `(s0, s1)`, the capacity is the total number of elements that must be allocated, which is at least `M * N` but may be larger if the stride implies gaps.

```cpp
// In CuTe, capacity() returns the total number of elements in a tensor's storage
// For a contiguous tensor of shape (M, N):
//   capacity = M * N
// For a tensor with padded stride:
//   capacity = s0 * (M - 1) + s1 * (N - 1) + 1

// Example:
auto tensor = make_tensor(ptr, make_shape(128, 256));
size_t cap = size(tensor);  // Logical size: 128 * 256 = 32768
// capacity(tensor) >= 32768
```

---

### Cluster

**Definition**: A group of CTAs (Cooperative Thread Arrays / threadblocks) that can coordinate and share data through distributed shared memory. Clusters are a hardware feature introduced in the NVIDIA Hopper architecture (SM90).

A cluster typically contains 1 to 8 CTAs that execute simultaneously on the same GPC (Graphics Processing Cluster). CTAs within a cluster can:
- Access each other's shared memory via distributed shared memory (DSMEM)
- Perform barrier synchronization across CTAs
- Coordinate TMA operations

```cpp
// Cluster dimensions are specified in the kernel launch configuration:
dim3 cluster_dims(2, 1, 1);  // 2 CTAs per cluster
dim3 grid_dims(M / tile_m, N / tile_n, 1);

// In CUTLASS 3.x kernels, cluster size is part of the dispatch policy:
using DispatchPolicy = cutlass::gemm::MainloopSm90TmaGmmaWarpSpecialized<
    TileShape, StageCount,
    cutlass::gemm::KernelTmaWarpSpecialized,
    2  // Cluster size (2 CTAs per cluster)
>;

// Cluster synchronization:
// CTAs within a cluster can synchronize via cluster_barrier
// This enables cooperative data loading and computation
```

---

### Collective

**Definition**: The main computation component that spans a threadblock (and potentially a warp group in warp-specialized kernels). A collective encompasses the mainloop of a GEMM operation: loading data from global memory, storing to shared memory, and performing the matrix multiply-accumulate.

In CUTLASS 3.x, collectives replace the 2.x concept of threadblock-level and warp-level operation layers with a single, unified abstraction.

```cpp
// A collective is typically defined via CollectiveBuilder:
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, StageCount, Schedule
>::CollectiveOp;

// The collective encapsulates:
// 1. Data loading from global memory to shared memory (TMA or cp.async)
// 2. Shared memory layout
// 3. MMA operation (WGMMA, MMA, WMMA, or SIMT)
// 4. Pipeline management (multi-stage prefetching)
// 5. Warp specialization (on SM90+)
```

---

### CuTe

**Definition**: CuTe is CUTLASS's modern C++ tensor abstraction library. It provides a formalized layout algebra, tensor objects, and composable algorithms for GPU programming. CuTe is the foundation of CUTLASS 3.x.

CuTe provides:
- **Layout**: A mathematical function mapping coordinates (logical indices) to linear offsets, represented as a pair of shape and stride.
- **Tensor**: A pairing of an engine (pointer or register array) with a layout, providing a unified abstraction for data access.
- **Algorithms**: Composable operations like `copy`, `gemm`, `fill`, and `prefetch` that operate on tensors.
- **Atoms**: Architecture-specific MMA and copy operations that serve as building blocks.

```cpp
#include "cute/tensor.hpp"

// CuTe layout: maps (i, j) to offset i * stride0 + j * stride1
auto layout = make_layout(make_shape(128, 64), make_stride(64, 1));

// CuTe tensor: pairs a pointer with a layout
auto tensor = make_tensor(ptr, layout);

// Access element at coordinate (i, j):
float val = tensor(i, j);

// CuTe GEMM algorithm:
gemm(tiled_mma, a_tensor, b_tensor, c_tensor);  // C += A * B
```

---

### Dispatch Policy

**Definition**: A tag-dispatch mechanism used in CUTLASS 3.x to select kernel specializations at compile time. Dispatch policies are empty structs (tags) that direct the collective builder to choose specific implementations.

```cpp
// Dispatch policies for SM90 GEMM mainloops:

// TMA-based with warp specialization (recommended for SM90)
struct MainloopSm90TmaGmmaWarpSpecialized;

// cp.async-based with warp specialization
struct MainloopSm90CpAsyncGmmaWarpSpecialized;

// TMA-based with cooperative handling
struct MainloopSm90TmaGmma;

// Dispatch policies are typically used through KernelScheduleAuto,
// which selects the best policy for the given architecture:
using Schedule = cutlass::gemm::collective::KernelScheduleAuto;
// Auto-selects based on:
// - Architecture (SM80 vs SM90)
// - Operation class (TensorOp vs SIMT)
// - Data types and alignment
```

---

### Epilogue

**Definition**: The post-GEMM processing stage that transforms the accumulator results into the final output. The epilogue applies scaling (`alpha * AB + beta * C`), type conversion, activation functions, and writes results to global memory.

```cpp
// CUTLASS 3.x epilogue:
using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
    LayoutD, LayoutC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>;

// The epilogue performs:
// 1. Load accumulator values from registers
// 2. Apply scaling: scaled = alpha * accumulator + beta * source
// 3. Apply activation function (ReLU, GELU, etc.)
// 4. Convert from accumulator type to output type (float -> half_t)
// 5. Write output to global memory

// CUTLASS 2.x epilogue (thread-level functor):
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,    // Output element type
    ElementsPerAccess, // Elements per memory access
    ElementAccumulator, // Accumulator type
    ElementCompute,    // Computation type for alpha/beta
    cutlass::epilogue::thread::Identity<ElementCompute>  // Activation
>;
```

---

### Fragment

**Definition**: A register-backed array that holds a thread's (or warp's) portion of a tile. Fragments are the primary data structure for holding data during computation in registers.

In CUTLASS 2.x, fragments are explicitly managed as `Array<T, N>` objects. In CUTLASS 3.x with CuTe, fragments are represented as CuTe tensors with register engines.

```cpp
// CUTLASS 2.x fragment:
using FragmentA = cutlass::Array<half_t, 32>;  // 32 FP16 values in registers
FragmentA frag_a;
// Fill from shared memory iterator
iterator_a.load(frag_a);

// CUTLASS 3.x / CuTe fragment:
// A tensor with a register engine, holding the thread's tile portion
auto frag_a = make_tensor<float>(Shape<Shape<_4, _2>, _2>{});
// This is a CuTe tensor stored in registers
// The layout determines how elements map to register indices
```

---

### GEMM

**Definition**: General Matrix Multiply. The fundamental operation `D = alpha * A * B + beta * C`, where A is MxK, B is KxN, C and D are MxN. GEMM is the core operation that CUTLASS optimizes.

```
  A (M x K)  *  B (K x N)  =  C (M x N)
  [a00 a01]     [b00 b01]     [c00 c01]
  [a10 a11]  *  [b10 b11]  =  [c10 c11]
  [a20 a21]     [b20 b21]     [c20 c21]

  c_ij = sum_k(a_ik * b_kj)
```

CUTLASS implements GEMM with a hierarchical tiling strategy:
1. **Device level**: Grid of threadblocks, each handling one output tile
2. **Threadblock level**: Each CTA processes a tile of the output matrix
3. **Warp level**: Each warp processes a sub-tile within the threadblock tile
4. **Thread level**: Each thread performs MMA operations on instruction-sized tiles

---

### Kernel Schedule

**Definition**: An execution policy that determines how a kernel orchestrates its computation, including data movement patterns, synchronization strategies, and warp roles.

```cpp
// Kernel schedules in CUTLASS 3.x:
using KernelScheduleAuto = ...;  // Auto-select best schedule

// SM90 schedules:
using KernelTmaWarpSpecialized = ...;      // TMA + warp-specialized (producer/consumer)
using KernelCpAsyncWarpSpecialized = ...;  // cp.async + warp-specialized
using KernelTma = ...;                     // TMA without warp specialization

// SM80 schedules:
using KernelMultistage = ...;   // Multi-stage cp.async pipeline
using KernelPingpong = ...;     // Double-buffered pipeline
```

---

### Layout

**Definition**: A functor that maps a logical coordinate (N-dimensional index) to a linear offset. Layouts encode the memory arrangement of tensor data.

In CUTLASS 2.x, layouts are classes like `RowMajor`, `ColumnMajor`, `PitchLinear`. In CUTLASS 3.x with CuTe, a layout is a mathematical function represented as `(shape, stride)`.

```cpp
// CUTLASS 2.x layout:
using LayoutA = cutlass::layout::RowMajor;     // stride = {N, 1}
using LayoutB = cutlass::layout::ColumnMajor;   // stride = {1, K}
using LayoutC = cutlass::layout::RowMajor;     // stride = {N, 1}

// CuTe layout:
auto row_major = make_layout(make_shape(M, N), make_stride(N, 1));
// Maps (i, j) -> i*N + j

auto col_major = make_layout(make_shape(M, N), make_stride(1, M));
// Maps (i, j) -> i + j*M

// General strided layout:
auto general = make_layout(make_shape(M, N), make_stride(stride_m, stride_n));
// Maps (i, j) -> i*stride_m + j*stride_n
```

---

### Mainloop

**Definition**: The primary computation loop in a GEMM kernel that iterates over the K dimension, loading tiles of A and B, and accumulating partial results.

The mainloop is the performance-critical section of a GEMM kernel. It overlaps data loading from global memory to shared memory with MMA computation using a multi-stage pipeline.

```cpp
// Pseudocode for a typical mainloop:
for (int k_tile = 0; k_tile < K / tile_k; ++k_tile) {
    // 1. Load A tile from global memory to shared memory (async)
    // 2. Load B tile from global memory to shared memory (async)
    // 3. Wait for previous load to complete (arrive-wait on pipeline)
    // 4. For each warp-level sub-tile:
    //    a. Load A fragment from shared memory to registers
    //    b. Load B fragment from shared memory to registers
    //    c. MMA: accumulator += A_fragment * B_fragment
    // 5. Advance pipeline to next stage
}

// In warp-specialized kernels (SM90+):
// Producer warp group: handles steps 1-2 (data loading)
// Consumer warp group: handles steps 3-4 (computation)
// Producer and consumer run concurrently via warp specialization
```

---

### MMA (Matrix Multiply-Accumulate)

**Definition**: The fundamental matrix multiplication operation that computes a matrix product and adds it to an accumulator. MMA operations execute on Tensor Cores (for TensorOp) or CUDA cores (for SIMT).

MMA operations are defined at multiple levels:
- **MMA Atom**: The hardware-level instruction (e.g., `wmma.mma.sync`, `mma.sync`, `wgmma.mma_async`)
- **Tiled MMA**: A tiled version of the atom that covers a larger tile

```cpp
// MMA atom sizes by architecture:
// SM70 (Volta):    WMMA 16x16x16 (FP16)
// SM75 (Turing):   MMA 16x8x8 (FP16), 8x8x16 (INT8)
// SM80 (Ampere):   MMA 16x8x16 (TF32, FP16, BF16)
// SM90 (Hopper):   WGMMA 64xNnx16 (FP16, BF16, TF32, FP8, INT8)
// SM100 (Blackwell): UMMA (unified MMA, wider tiles)
```

---

### Operand

**Definition**: A matrix input or output in a GEMM operation. The standard operands are:

| Operand | Matrix | Dimensions | Role |
|---|---|---|---|
| A | Left matrix | M x K | Multiplied with B |
| B | Right matrix | K x N | Multiplied with A |
| C | Source matrix | M x N | Added to the product (with beta scaling) |
| D | Destination matrix | M x N | Final output: D = alpha * A * B + beta * C |

```cpp
// Operand is also used as an enum for identifying which matrix is being accessed:
enum class Operand {
    kA = 0,  // Matrix A
    kB = 1,  // Matrix B
    kC = 2,  // Matrix C (source)
    kD = 3   // Matrix D (destination)
};

// Used in iterators and tile accessors methods:
template <Operand Operand_>
struct TileIterator;
```

---

### Pipeline

**Definition**: A multi-stage data movement pattern that overlaps computation with memory transfers. A pipeline consists of multiple stages (buffers) where data at different stages of processing is held simultaneously.

Pipelines enable latency hiding: while one stage's data is being computed, the next stage's data is being loaded.

```cpp
// Pipeline types in CUTLASS 3.x:
// 1. TmaPipeline: Uses TMA for async loading from global to shared memory
// 2. Sm80Pipeline: Uses cp.async for async loading
// 3. Sm75Pipeline: Uses synchronous loading (no async copy)

// Pipeline stages:
// A pipeline with N stages has N buffers in shared memory.
// While stage i is being consumed (computed with), stage (i+1) % N is being produced (loaded into).

// Pipeline synchronization uses barrier operations:
// Producer: issue async loads, then commit and signal barrier
// Consumer: wait on barrier, then consume data

// In warp-specialized pipelines (SM90+):
// Producer warp group: issues TMA copies, signals pipeline barriers
// Consumer warp group: waits on pipeline barriers, performs WGMMA
```

---

### SMEM (Shared Memory)

**Definition**: Fast on-chip memory shared among all threads within a CTA (threadblock). Shared memory sits between global memory (slow, large) and registers (fast, small). It is used for staging data between global memory and registers.

```cpp
// SMEM characteristics:
// - Per-CTA allocation: each threadblock has its own SMEM
// - Size: up to 164 KB (SM80), up to 228 KB (SM90) per threadblock
// - Latency: ~20-30 cycles (vs. ~200-400 cycles for global memory)
// - Banked architecture: 32 banks, 4 bytes per bank
// - Access: via load/store instructions, async copy (cp.async, TMA)

// SMEM allocation in CUTLASS:
// - Static: __shared__ keyword in CUDA
// - Dynamic: allocated via the kernel launch's dynamic shared memory parameter

// In CUTLASS 3.x, SMEM layout is managed by CuTe:
// auto smem_tensor = make_tensor(make_smem_ptr(smem_ptr), smem_layout);
```

---

### Split-K

**Definition**: A parallelization strategy that splits the K dimension across multiple threadblocks, each computing a partial product. The partial results are then combined using a reduction step.

Split-K is beneficial when the GEMM output matrix is small (few output tiles) but K is large, leaving the GPU underutilized.

```cpp
// Without Split-K: one threadblock per output tile, iterates over all K
// Grid size = ceil(M/tile_m) * ceil(N/tile_n)

// With Split-K: multiple threadblocks per output tile, each handles K/split_k tiles
// Grid size = ceil(M/tile_m) * ceil(N/tile_n) * split_k_slices

// Usage in CUTLASS:
typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,
    {M, N, K},
    {ptr_A, stride_A}, {ptr_B, stride_B},
    {ptr_C, stride_C}, {ptr_D, stride_D},
    {alpha, beta},
    split_k_slices  // Number of splits along K
};

// Split-K requires workspace for partial results:
// workspace_size = split_k_slices * M * N * sizeof(ElementAccumulator)
size_t workspace_size = Gemm::get_workspace_size(args);
```

---

### Stage

**Definition**: One iteration of the pipeline buffer. A stage represents a complete tile of data (for both A and B operands) that is being processed. Multi-stage pipelines allow overlapping load and compute operations.

```cpp
// A pipeline with N stages has N sets of shared memory buffers:
// Stage 0: SMEM buffer 0 for A tile, SMEM buffer 0 for B tile
// Stage 1: SMEM buffer 1 for A tile, SMEM buffer 1 for B tile
// ...
// Stage N-1: SMEM buffer N-1 for A tile, SMEM buffer N-1 for B tile

// SMEM per stage = tile_m * tile_k * sizeof(ElementA) + tile_k * tile_n * sizeof(ElementB)
// Total SMEM = stages * SMEM_per_stage

// Stage count tradeoffs:
// More stages: better latency hiding, more SMEM usage, fewer concurrent CTAs
// Fewer stages: less SMEM, more concurrent CTAs, potentially worse latency hiding

// In CUTLASS 3.x:
using StageCount = cutlass::gemm::collective::StageCount<4>;  // 4 stages
using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<0>;  // Auto
```

---

### Tiled MMA

**Definition**: A tiled version of an MMA atom that covers a larger tile by replicating the atom across threads and data. The tiling determines how threads within a warp (or warp group) cooperate to compute a larger matrix product.

```cpp
// CuTe tiled MMA construction:
auto mma_atom = SM90_64x128x16_F16F16F16_SS {};  // WGMMA atom
auto tiled_mma = make_tiled_mma(mma_atom,
    make_layout(make_shape(_2{}, _2{})),  // 2x2 replication
    Tile<_128, _128, _16>{}               // Logical tile size
);

// The tiled MMA maps:
// - Threads to MMA atom instances
// - Data tiles to atom tile positions
// - Accumulator tiles to warp-level register layout
```

---

### Tiled Copy

**Definition**: A tiled version of a copy atom that defines how threads cooperate to copy a tile of data between memory spaces (e.g., global to shared, shared to register).

```cpp
// CuTe tiled copy construction:
auto copy_atom = SM90_TMA_LOAD {};  // TMA copy atom
auto tiled_copy = make_tiled_copy_A(copy_atom,
    smem_layout,  // Layout in shared memory
    tiled_mma     // Associated tiled MMA for thread mapping
);

// Or for cp.async:
auto copy_atom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS, half_t>{};
auto tiled_copy = make_tiled_copy(copy_atom,
    make_layout(make_shape(_32{}, _8{})),  // 32x8 thread layout
    Layout<_8>{}                            // Vector length
);
```

---

### Tile

**Definition**: A constant-sized partition of a tensor that is processed by a single CTA (at the threadblock level) or a single warp (at the warp level). Tiles are the fundamental unit of work distribution in CUTLASS.

```cpp
// Tile shapes are compile-time constants:
using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
// Tile_M = 128, Tile_N = 128, Tile_K = 64

// Grid dimensions determine how many tiles are processed:
// grid_x = ceil(M / Tile_M)
// grid_y = ceil(N / Tile_N)

// Each CTA processes one output tile of size (Tile_M x Tile_N)
// and iterates over K in chunks of Tile_K
```

---

## 38.2 Architecture-Specific Terms

### TMA (Tensor Memory Accelerator)

**Definition**: A hardware unit introduced in the NVIDIA Hopper architecture (SM90) that performs asynchronous multi-dimensional tensor copy operations from global memory to shared memory. TMA handles address computation, boundary checking, and swizzling autonomously, freeing threads from data movement overhead.

```cpp
// TMA features:
// - Multi-dimensional address generation (1D to 5D)
// - Automatic boundary handling (out-of-bounds fills with zero)
// - Hardware swizzling for bank-conflict-free shared memory layout
// - Cluster-level broadcast (one TMA load, multiple CTAs receive)
// - Async operation with completion signals

// TMA in CUTLASS is accessed through copy atoms:
auto tma_copy = SM90_TMA_LOAD {};
copy(tma_copy, gmem_tensor, smem_tensor);

// Or through the CollectiveBuilder which selects TMA automatically for SM90
```

---

### GMMA (Global Matrix Multiply-Accumulate)

**Definition**: An alternate name for operations that combine global memory access with matrix multiply-accumulate on SM90. In practice, GMMA typically refers to the WGMMA instruction in the context of data loading from shared memory to registers.

GMMA is used informally in CUTLASS to describe the SM90 Tensor Core instruction path.

---

### WGMMA (Warp Group Matrix Multiply-Accumulate)

**Definition**: The SM90 (Hopper) Tensor Core instruction that operates across an entire warp group (4 warps = 128 threads). WGMMA computes a matrix product of larger dimensions than previous MMA instructions.

```cpp
// WGMMA characteristics:
// - Operates on a warp group (128 threads, 4 warps)
// - A matrix in shared memory (up to 64xK)
// - B matrix in shared memory or register (up to KxNn)
// - Accumulator in registers (up to 64x256 for FP16)
// - Async execution: overlaps with other warp group operations
// - Supported types: FP16, BF16, TF32, FP8 (e4m3, e5m2), INT8

// WGMMA instruction shapes (FP16 example):
// A: 64 x K (shared memory)
// B: K x Nn (shared memory, where Nn is 8, 16, ..., 256)
// C: 64 x Nn (registers, accumulator)

// In CUTLASS 3.x, WGMMA is used through CuTe MMA atoms:
auto wgmma_atom = SM90_64x128x16_F16F16F16_SS_TN{};
```

---

### WMMA (Warp Matrix Multiply-Accumulate)

**Definition**: The SM70 (Volta) Tensor Core instruction that operates at the warp level (32 threads). WMMA was the first Tensor Core programming interface, introduced with the Volta architecture.

```cpp
// WMMA characteristics:
// - Operates on a warp (32 threads)
// - Fixed 16x16x16 tile size (FP16)
// - Synchronous API: wmma::load_matrix_sync, wmma::store_matrix_sync, wmma::mma_sync
// - Available on SM70 and later architectures

// WMMA tile sizes:
// FP16: 16x16x16
// INT8 (SM75+): 16x16x16, 8x32x16, 32x8x16
// INT4 (SM75+): 8x32x32

// In CUTLASS, WMMA is the lowest-level Tensor Core interface for SM70.
// For SM75+, CUTLASS uses the more flexible MMA PTX instructions instead.
```

---

### UMMA (Unified Matrix Multiply-Accumulate)

**Definition**: The SM100+ (Blackwell) unified MMA instruction that provides a single interface for all matrix multiply operations. UMMA unifies and extends the WGMMA instruction with support for block-scaled types and larger tile sizes.

```cpp
// UMMA characteristics:
// - Operates on a warp group (128 threads)
// - Supports block-scaled types: NVFP4, MXFP4, MXFP6, MXFP8
// - Unified interface for all data types
// - Larger accumulator tiles
// - Distributed shared memory support

// UMMA is used in CUTLASS 3.x for Blackwell:
auto umma_atom = SM100_64x192x32_F16F16F16_SS_TN{};
```

---

### Warp Specialization

**Definition**: A programming model introduced in SM90 (Hopper) where different warp groups within a CTA assume different roles: producer (data loading) and consumer (computation). This enables true concurrent execution of data movement and computation within the same threadblock.

```cpp
// In a warp-specialized kernel:
// CTA has 128 threads = 4 warp groups (on SM90):
// - Warp Group 0: Producer (loads data from global to shared memory via TMA)
// - Warp Group 1, 2, 3: Consumer (performs WGMMA computation)

// The producer-consumer model:
// Producer:
//   1. Issue TMA loads for A and B tiles
//   2. Signal pipeline barriers
//   3. Repeat for all K tiles
//
// Consumer:
//   1. Wait on pipeline barrier
//   2. Perform WGMMA on loaded tiles
//   3. Repeat for all K tiles
//
// Both run concurrently, hiding memory latency with computation.

// Warp specialization is enabled by the dispatch policy:
using DispatchPolicy = cutlass::gemm::collective::KernelTmaWarpSpecialized;
```

---

## 38.3 CuTe-Specific Terms

### Atom

**Definition**: The smallest hardware-level operation that CuTe can tile. An atom represents a single MMA instruction or a single copy operation. Atoms are architecture-specific and are tiled to cover larger logical tiles.

```cpp
// MMA atoms:
// SM90: SM90_64x128x16_F16F16F16_SS_TN (WGMMA)
// SM80: SM80_16x8x16_F32F16F16F32_TN (MMA)
// SM75: SM75_16x8x8_S32S8S8S32_TN (MMA)
// SM70: SM70_16x16x16_F16F16F16F16_TN (WMMA)

// Copy atoms:
// SM90: SM90_TMA_LOAD, SM90_TMA_STORE
// SM80: SM80_CP_ASYNC_CACHEALWAYS, SM80_CP_ASYNC_CACHEGLOBAL
// Generic: AutoVectorizingCopy, UniversalCopy

// Atoms are used as building blocks:
auto tiled_mma = make_tiled_mma(mma_atom, ...);
auto tiled_copy = make_tiled_copy(copy_atom, ...);
```

### Compose

**Definition**: A CuTe operation that combines two layouts (or a layout and a shape) to create a new layout. Composition is the fundamental operation for mapping thread layouts to data layouts.

```cpp
// Compose: given layout L and shape S, create layout L' = L o S
// This maps the logical coordinates in S through L

auto layout = make_layout(make_shape(128, 64), make_stride(64, 1));
auto shape = make_shape(32, 4);

auto composed = composition(layout, shape);
// Creates a new layout that selects 32 elements in the first dimension
// and 4 in the second from the original layout

// Composition is used to create thread-to-data mappings:
// auto thread_layout = make_layout(make_shape(_32{}, _4{}), make_stride(_4{}, _1{}));
// auto data_layout = composition(tensor_layout, thread_layout);
```

### Shape

**Definition**: A compile-time or runtime multi-dimensional extent in CuTe. Shapes can be hierarchical (nested) and may contain both static and dynamic dimensions.

```cpp
// Static shapes (compile-time known):
auto shape_static = Shape<_128, _64>{};  // 128 x 64

// Dynamic shapes (runtime known):
auto shape_dynamic = make_shape(128, 64);

// Hierarchical shapes:
auto shape_hier = Shape<Shape<_4, _8>, _64>{};  // (4x8) x 64

// Mixed static/dynamic:
auto shape_mixed = make_shape(Int<128>{}, 64);
```

### Stride

**Definition**: A compile-time or runtime multi-dimensional stride in CuTe. Strides define the memory step between consecutive elements in each dimension of a shape.

```cpp
// Strides for a row-major 128x64 matrix:
auto stride_rm = make_stride(_64{}, _1{});  // Static: step 64 in dim 0, step 1 in dim 1

// Strides for a column-major 128x64 matrix:
auto stride_cm = make_stride(_1{}, _128{});  // Step 1 in dim 0, step 128 in dim 1

// General stride:
auto stride_gen = make_stride(64, 1);  // Dynamic strides

// Zero stride (broadcast):
auto stride_bc = make_stride(_1{}, _0{});  // Second dimension broadcasts (always same value)
```

---

## 38.4 Data Type Terms

### Element

**Definition**: The data type of a single element in a tensor. CUTLASS supports a wide range of element types.

| Type | Bits | Range | Description |
|---|---|---|---|
| `float` | 32 | IEEE 754 | Standard single precision |
| `double` | 64 | IEEE 754 | Standard double precision |
| `half_t` | 16 | IEEE 754 FP16 | Half precision |
| `bfloat16_t` | 16 | BF16 | Brain floating point |
| `tfloat32_t` | 19 | TF32 | TensorFloat-32 |
| `float_e4m3_t` | 8 | FP8 E4M3 | FP8 with 4 exponent bits |
| `float_e5m2_t` | 8 | FP8 E5M2 | FP8 with 5 exponent bits |
| `int8_t` | 8 | -128 to 127 | Signed 8-bit integer |
| `uint8_t` | 8 | 0 to 255 | Unsigned 8-bit integer |
| `int4b_t` | 4 | -8 to 7 | Signed 4-bit integer |
| `uint4b_t` | 4 | 0 to 15 | Unsigned 4-bit integer |
| `int2b_t` | 2 | -2 to 1 | Signed 2-bit integer |
| `uint2b_t` | 2 | 0 to 3 | Unsigned 2-bit integer |
| `bin1_t` | 1 | 0 to 1 | Binary value |

---

## 38.5 Deprecated CUTLASS 2.x Terms

The following terms from CUTLASS 2.x are deprecated in 3.x. The table shows the 2.x term and its 3.x equivalent:

| CUTLASS 2.x Term | CUTLASS 3.x Equivalent | Notes |
|---|---|---|
| `Operator` (thread/warp level) | **Atom** | The hardware-level MMA or copy operation is now called an "atom" |
| `Tile Iterator` | **CuTe Layout + Tensor** | Data access patterns are now expressed as CuTe layouts and tensors |
| `Thread Map` | **CuTe Layout** | The mapping of threads to data elements is now a CuTe layout |
| `MmaPolicy` | **Dispatch Policy** | Compile-time configuration is now a tag-dispatch policy |
| `EpilogueOutputOp` | **Collective Epilogue** | Epilogue is now a collective operation |
| `WarpIterator` | **CuTe Tensor (register engine)** | Warp-level data access uses CuTe tensors |
| `SmemIterator` | **CuTe Tensor (smem engine)** | Shared memory access uses CuTe tensors |
| `GmemIterator` | **CuTe Tensor (gmem engine)** | Global memory access uses CuTe tensors |
| `TransformedIterator` | **CuTe composition** | Data transformations are expressed as layout composition |
| `ThreadblockShape` | **TileShape** | Now expressed as a single `GemmShape` parameter |
| `WarpShape` | **(implicit in TiledMMA)** | Warp-level tiling is implicit in the MMA atom tiling |
| `InstructionShape` | **(implicit in MMA Atom)** | The instruction shape is determined by the MMA atom |
| `OperatorClass` | **OpClass** | `OpClassTensorOp` or `OpClassSimt`, passed to CollectiveBuilder |
| `MmaCore` | **Collective** | The core MMA logic is now in the collective |
| `SharedMemory` | **CuTe SMEM Tensor** | Shared memory is managed as a CuTe tensor |

### Migration Notes

```cpp
// 2.x: Explicit 3-level hierarchy
using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA, ElementB, LayoutB,
    ElementC, LayoutC, ElementAccumulator,
    OpClass, ArchTag,
    ThreadblockShape,   // Level 1: threadblock tile
    WarpShape,          // Level 2: warp tile
    InstructionShape,   // Level 3: MMA instruction
    EpilogueOp
>;

// 3.x: CollectiveBuilder selects the hierarchy
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape,          // Single tile shape (no separate warp/instruction)
    StageCount,
    Schedule
>::CollectiveOp;
// WarpShape and InstructionShape are inferred from the MMA atom
// selected by the CollectiveBuilder for the given architecture.
```

---

## 38.6 Quick Reference: Acronym Expansion

| Acronym | Full Name |
|---|---|
| CTA | Cooperative Thread Array (threadblock) |
| CuTe | C++ Unified Tensor Expression |
| DSMEM | Distributed Shared Memory (across cluster) |
| FMHA | Fused Multi-Head Attention |
| GEMM | General Matrix Multiply |
| GMMA | Global Matrix Multiply-Accumulate |
| GMEM | Global Memory |
| MMA | Matrix Multiply-Accumulate |
| PTX | Parallel Thread Execution (NVIDIA's low-level GPU instruction set) |
| REG | Register (per-thread storage) |
| SIMT | Single Instruction Multiple Thread |
| SMEM | Shared Memory |
| SASS | Streaming Assembly (NVIDIA's GPU machine code) |
| TMA | Tensor Memory Accelerator |
| TF32 | TensorFloat-32 |
| UMMA | Unified Matrix Multiply-Accumulate |
| WGMMA | Warp Group Matrix Multiply-Accumulate |
| WMMA | Warp Matrix Multiply-Accumulate |
