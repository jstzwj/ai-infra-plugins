# CUTLASS - Chapter 11: Transform Operations

This reference covers transform operations in CUTLASS, including tile iterators for loading and storing data, thread-level and warp-level transforms, layout transformations for Tensor Core input, and data movement patterns between memory hierarchies.

---

## 11.1 Overview

Transform operations in CUTLASS are responsible for moving data between different memory hierarchies (global memory, shared memory, registers) and transforming data layouts to match the requirements of hardware-accelerated operations such as Tensor Core MMA (Matrix Multiply-Accumulate) instructions. These operations are essential building blocks in the GEMM hierarchy, bridging the gap between the high-level mathematical operation and the low-level hardware instructions.

The transform layer in CUTLASS provides:

- **Tile iterators** that load and store rectangular tiles of data with bounds checking and predicate computation.
- **Thread-level transforms** that reshape, transpose, or otherwise modify data within a single thread's register file.
- **Warp-level transforms** that coordinate data movement across threads in a warp, particularly for shared memory access patterns.
- **Tensor Core input transforms** that rearrange data into the specific layouts required by MMA hardware instructions.
- **Copy operations** that handle data movement between global memory, shared memory, and registers with proper vectorization and alignment.

These components are primarily found in the following CUTLASS header directories:

- `include/cutlass/transform/` - Core transform operations and tile iterators
- `include/cutlass/transform/thread/` - Thread-level transforms
- `include/cutlass/transform/warp/` - Warp-level transforms
- `include/cutlass/transform/kernel/` - Kernel-level transform operations

---

## 11.2 Tile Iterators

Tile iterators are the primary mechanism for loading and storing tiles (small rectangular blocks) of data from global memory into shared memory or registers, and vice versa. They abstract the complex indexing arithmetic needed to map thread IDs to the correct memory locations within a tile.

### 11.2.1 PredicatedTileIterator

`PredicatedTileIterator` is the workhorse tile iterator for CUTLASS 2.x kernels. It loads tiles from global memory into a shared memory scratchpad, performing bounds checking via predicates to handle edge cases where the tile extends beyond the valid data region.

**Key characteristics:**

- **Bounds-checked**: Generates predicates for each thread to guard against out-of-bounds accesses.
- **Vectorized**: Uses vector memory instructions (e.g., `ld.global.nc.v4`) for coalesced access.
- **Configurable tile shape**: The tile dimensions are template parameters, allowing the same iterator to be used at different levels of the GEMM hierarchy.
- **Residual tile handling**: When the problem dimensions are not multiples of the tile dimensions, the iterator computes residual (partial) tiles.

**Template parameters:**

```cpp
template <
    typename Shape_,               // Tile shape (e.g., GemmShape<128, 128>)
    typename Element_,             // Data type of elements
    typename Layout_,              // Layout of source data (RowMajor, ColumnMajor, etc.)
    int AdvanceRank,               // Which dimension to advance along (0 or 1)
    typename ThreadMap_,           // Maps threads to data elements
    int AccessSize = ThreadMap_::kElementsPerAccess  // Vector access width
>
class PredicatedTileIterator;
```

**Example: Using PredicatedTileIterator to load a tile from global memory**

```cpp
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

// Define the tile shape for the A operand (K x M tile, row-major)
using TileShape = cutlass::gemm::GemmShape<128, 64>;
using ElementA = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;

// Define a thread map: how threads map onto the tile
using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<TileShape::kColumn, TileShape::kRow>,
    128,   // number of threads
    4      // elements per access (vector width)
>;

// Iterator type
using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
    TileShape, ElementA, LayoutA, 1, ThreadMap
>;

// Inside the kernel:
// typename IteratorA::Params params(layout_A);
// IteratorA iterator(params, ptr_A, {M, N, K}, thread_idx, block_idx);

// // Load a tile into shared memory
// typename IteratorA::Fragment fragment;
// iterator.load(fragment);
// // Store fragment to shared memory
// // ... (typically using a shared memory iterator)
```

**AdvanceRank parameter explained:**

The `AdvanceRank` parameter controls which dimension of the GEMM the iterator advances along when iterating through tiles:

- `AdvanceRank = 0`: The iterator advances along the rows (K dimension for operand A in a row-major GEMM). Used for the A operand in the K-dimension iteration.
- `AdvanceRank = 1`: The iterator advances along the columns. Used for the B operand in the K-dimension iteration.

**ThreadMap concept:**

The `ThreadMap` defines how threads in a threadblock map onto the tile's elements. CUTLASS provides several thread map strategies:

- `PitchLinearStripminedThreadMap` - Strips elements along the contiguous dimension across threads, ensuring coalesced memory access.
- `TransposePitchLinearThreadMap` - Transposes the thread mapping for column-major access patterns.
- `PitchLinearWarpRakedThreadMap` - Organizes threads within warps for bank-conflict-free shared memory stores.

```cpp
// ThreadMap for loading operand A (M x K tile, row-major)
using ThreadMapA = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<128, 8>,  // contiguous x strided
    128,   // threads
    4      // elements per access
>;
```

### 11.2.2 RegularTileIterator

`RegularTileIterator` is used for iterating over tiles in dense, regularly-strided data without bounds checking. It is typically used for moving data between shared memory and registers where bounds are already guaranteed.

**Key characteristics:**

- **No predication**: Assumes all accesses are in bounds, avoiding predicate overhead.
- **Optimized for shared memory**: Uses shared memory access patterns that minimize bank conflicts.
- **SUPPORTED by all layouts**: Works with RowMajor, ColumnMajor, and interleaved layouts.

**Example:**

```cpp
#include "cutlass/transform/threadblock/regular_tile_iterator.h"

// Define shapes
using Shape = cutlass::gemm::GemmShape<64, 64>;
using Element = cutlass::half_t;
using Layout = cutlass::layout::RowMajor;

// ThreadMap for the regular iterator
using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
    128, 4
>;

using RegularIterator = cutlass::transform::threadblock::RegularTileIterator<
    Shape, Element, Layout, 1, ThreadMap
>;

// Usage in kernel:
// RegularIterator iterator(smem_ptr, thread_idx);
// RegularIterator::Fragment fragment;
// iterator.load(fragment);  // Load from shared memory to registers
// iterator.store(fragment); // Store from registers to shared memory
```

### 11.2.3 PredicatedTileIterator2dThreadTile

`PredicatedTileIterator2dThreadTile` extends the predicated tile iterator concept to support 2D thread tiles, where each thread is responsible for a small 2D block of elements rather than a 1D strip. This is useful for certain epilogue operations and output writing patterns.

**Key characteristics:**

- **2D thread assignment**: Each thread owns a 2D sub-tile of the output.
- **Predicated**: Handles boundary conditions for non-aligned problem sizes.
- **Efficient output writing**: Reduces the number of strided accesses when writing output tiles.

```cpp
#include "cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h"

// Define for output writing with 2D thread tiles
using OutputIterator = cutlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
    cutlass::gemm::GemmShape<128, 128>,   // Threadblock tile shape
    float,                                 // Element type
    cutlass::layout::RowMajor,             // Output layout
    1,                                     // Advance rank
    cutlass::transform::PitchLinearWarpRakedThreadMap<
        cutlass::layout::PitchLinearShape<8, 32>,
        128, 4
    >,
    4,                                     // Elements per access
    cutlass::gemm::GemmShape<4, 4>         // Thread tile shape (2D)
>;
```

---

## 11.3 Thread-Level Transforms

Thread-level transforms operate on data within a single thread's register file. They reshape, transpose, or modify data layouts to prepare for warp-level operations.

### 11.3.1 Transpose

The `transpose` transform performs matrix transposition on data held in registers. This is commonly needed when the operand layout does not match the expected Tensor Core input layout.

```cpp
#include "cutlass/transform/thread/transpose.h"

// Transpose a fragment of data
// Fragment is typically an array<array<Element, N>, M> representing an M x N tile
using TransposeOp = cutlass::transform::thread::Transpose<
    cutlass::gemm::GemmShape<4, 4>,  // Shape of the tile to transpose
    cutlass::half_t,                  // Element type
    4                                 // Elements per access (vector width)
>;

// Usage:
// typename TransposeOp::Fragment fragment;
// // ... fill fragment ...
// TransposeOp transpose_op;
// transpose_op.transform(fragment, fragment);  // In-place transpose
```

The transpose operation is critical for:
- Converting row-major data to column-major format (or vice versa) before feeding into Tensor Core operations.
- Rearranging data within registers to match the expected storage order of MMA instructions.
- Handling the transposition of operand B in GEMM (which is often stored transposed in shared memory for efficient access).

### 11.3.2 ScaleBias

The `ScaleBias` transform applies per-channel scaling and bias to data elements. This is used in quantized inference and mixed-precision training where data needs to be rescaled before or after multiplication.

```cpp
#include "cutlass/transform/thread/scale_bias.h"

using ScaleBiasOp = cutlass::transform::thread::ScaleBias<
    cutlass::gemm::GemmShape<1, 128>,  // Shape (typically 1 x N for per-column)
    float,                              // Element type
    float                               // Scale/bias element type
>;

// Usage:
// ScaleBiasOp scale_bias(scale_ptr, bias_ptr);
// typename ScaleBiasOp::Fragment fragment;
// // ... load fragment ...
// scale_bias.transform(fragment, fragment);
```

**Typical use cases:**

- **Quantized GEMM**: Scaling INT8 products back to FP32 or FP16.
- **Layer normalization**: Applying scale and bias in the epilogue.
- **Batch normalization**: Per-channel scaling during inference.

### 11.3.3 planarComplexToArray

The `planarComplexToArray` transform converts data stored in planar complex format (real and imaginary parts in separate arrays) to the interleaved array format expected by complex-valued Tensor Core operations.

```cpp
#include "cutlass/transform/thread/planar_complex.h"

// Convert planar complex to interleaved complex
// Planar format: [Re0, Re1, Re2, ...] [Im0, Im1, Im2, ...]
// Array format:  [Re0, Im0, Re1, Im1, Re2, Im2, ...]
using PlanarToArray = cutlass::transform::thread::planarComplexToArray<
    cutlass::half_t,   // Real element type
    4                  // Elements per access
>;

// Usage:
// typename PlanarToArray::Fragment real_part, imag_part, result;
// // ... load real_part and imag_part from separate arrays ...
// PlanarToArray transform;
// transform(result, real_part, imag_part);
```

### 11.3.4 arrayToPlanarComplex

The `arrayToPlanarComplex` transform performs the inverse operation: converting interleaved complex data back to planar format where real and imaginary components are stored in separate memory regions.

```cpp
#include "cutlass/transform/thread/planar_complex.h"

using ArrayToPlanar = cutlass::transform::thread::arrayToPlanarComplex<
    cutlass::half_t,   // Real element type
    4                  // Elements per access
>;

// Usage:
// typename ArrayToPlanar::Fragment interleaved, real_out, imag_out;
// // ... fill interleaved ...
// ArrayToPlanar transform;
// transform(real_out, imag_out, interleaved);
```

---

## 11.4 Warp-Level Transforms

Warp-level transforms coordinate data movement and layout changes across all threads in a warp (32 threads). These are particularly important for the interaction between shared memory and the register file in preparation for Tensor Core MMA instructions.

### 11.4.1 WarpTileIterator

`WarpTileIterator` is the primary warp-level transform component. It loads data from shared memory into warp-level fragments (register files distributed across warp threads) in the specific layout required by MMA instructions.

**Template parameters:**

```cpp
template <
    typename WarpShape_,          // Shape of the warp-level tile
    typename Element_,            // Data element type
    typename Layout_,             // Shared memory layout
    typename InstructionShape_,   // MMA instruction shape (e.g., 16x8x16)
    int AccessSize = 128          // Access size in bits
>
class WarpTileIterator;
```

**Layout-specific specializations:**

CUTLASS provides specializations of `WarpTileIterator` for different shared memory layouts:

```cpp
// For row-major shared memory layout
using WarpIteratorA = cutlass::transform::warp::WarpTileIterator<
    cutlass::gemm::GemmShape<64, 32>,     // Warp shape
    cutlass::half_t,                       // Element type
    cutlass::layout::RowMajor,             // Shared memory layout
    cutlass::gemm::GemmShape<16, 8, 16>,  // Instruction shape
    128                                    // Access size in bits
>;

// For column-major shared memory layout
using WarpIteratorB = cutlass::transform::warp::WarpTileIterator<
    cutlass::gemm::GemmShape<32, 64>,     // Warp shape
    cutlass::half_t,                       // Element type
    cutlass::layout::ColumnMajor,          // Shared memory layout
    cutlass::gemm::GemmShape<16, 8, 16>,  // Instruction shape
    128                                    // Access size in bits
>;
```

### 11.4.2 Shared Memory Loading Patterns

The warp-level iterators implement specific shared memory loading patterns designed to:

1. **Avoid bank conflicts**: Shared memory has 32 banks, and simultaneous accesses to the same bank by different threads result in bank conflicts. The iterators arrange access patterns to minimize these conflicts.

2. **Vectorize loads**: Use 128-bit load instructions to maximize memory throughput.

3. **Match MMA input requirements**: The data in registers must be in the exact layout expected by the hardware MMA instruction.

```cpp
// Example: Warp-level shared memory load pattern for operand A
// Each thread in the warp loads elements according to the thread map
// defined by the WarpTileIterator specialization.

// Shared memory layout for operand A (row-major):
// SMEM[0][0..K]   row 0
// SMEM[1][0..K]   row 1
// ...
// SMEM[M-1][0..K] row M-1

// The warp iterator maps threads to SMEM locations such that:
// - Threads 0-31 each load a portion of the M x K tile
// - The loaded data is arranged in registers to match the MMA instruction format
// - Bank conflicts are minimized through padding and access pattern design
```

### 11.4.3 Warp-Level Matrix Fragment

The output of a warp-level load is a `Fragment` -- an array of elements held in registers across the threads of the warp. The fragment layout is hardware-specific and depends on the MMA instruction shape.

```cpp
// Fragment type for a warp-level MMA operation
// For a 16x8x16 MMA with FP16:
// Each thread holds a specific set of elements from the A, B, and C matrices
// FragmentA: array<half_t, 4>  (for 16x8x16, each thread holds 4 FP16 values of A)
// FragmentB: array<half_t, 2>  (each thread holds 2 FP16 values of B)
// FragmentC: array<float, 4>   (each thread holds 4 FP32 accumulator values)

using FragmentA = cutlass::Array<cutlass::half_t, 4>;
using FragmentB = cutlass::Array<cutlass::half_t, 2>;
using FragmentC = cutlass::Array<float, 4>;

// The warp-level MMA operation:
// mma_op(accumulators, fragment_A, fragment_B, accumulators);
```

---

## 11.5 Transform for Tensor Core Input

Tensor Core MMA instructions require data in very specific layouts. The transform layer is responsible for converting from the natural data layout (e.g., row-major or column-major) to the hardware-required layout.

### 11.5.1 Data Layout Transformation for MMA Input

For Ampere and later architectures, the hardware MMA instructions (e.g., `mma.sync`) expect:

- **Operand A**: Data arranged in a specific pattern where consecutive threads hold consecutive elements along the M dimension.
- **Operand B**: Data arranged such that consecutive threads hold consecutive elements along the N dimension.
- **Operand C**: Accumulator data distributed across warp threads in a specific pattern.

The transformation from shared memory to the MMA input layout involves:

1. **Loading from shared memory** with the correct access pattern.
2. **Optionally transposing** the data if the natural layout differs from the MMA expectation.
3. **Rearranging elements** within each thread's registers to match the MMA fragment format.

```cpp
// CUTLASS 2.x: Transform chain for Tensor Core input
//
// Global Memory
//      |
//      | (PredicatedTileIterator - vectorized, predicated load)
//      v
// Shared Memory (with padding for bank conflict avoidance)
//      |
//      | (WarpTileIterator - bank-conflict-free load)
//      v
// Register File (Fragment in MMA layout)
//      |
//      | (mma_sync operation)
//      v
// Accumulator Registers
```

### 11.5.2 Crosswise Layout Handling

The **crosswise** layout is a specific shared memory arrangement used for the K dimension of the GEMM. In this layout, elements along the K dimension are stored contiguously, and elements along the M (or N) dimension are stored with a stride that crosses the shared memory banks in a conflict-free manner.

```cpp
// Crosswise layout for operand A in shared memory
// For a 128x32 tile (M=128, K=32) with FP16 elements:
// Elements along K (contiguous dimension) are stored in groups
// The stride is chosen so that concurrent warp accesses avoid bank conflicts

// CUTLASS defines crosswise layouts through the layout templates:
using SmemLayoutA = cutlass::layout::ColumnMajorInterleaved<4>;
// Or for specific crosswise stride:
using SmemLayoutB = typename cutlass::layout::PitchLinearShape<
    cutlass::gemm::GemmShape<128, 32>::kRow,
    cutlass::gemm::GemmShape<128, 32>::kColumn
>;
```

**Crosswise vs. Congruous layouts:**

| Layout Type | Description | Use Case |
|---|---|---|
| **Congruous** | Contiguous elements along the M or N dimension; stride crosses banks | Operand A (row-major GEMM) or Operand B (column-major GEMM) |
| **Crosswise** | Contiguous elements along the K dimension; stride crosses banks | The other operand that requires K-dimension contiguity |

### 11.5.3 Interleaved Layout Transformation

**Interleaved** layouts store elements in small interleaved groups (e.g., 4 or 8 elements) to optimize for shared memory bank conflict avoidance. The transformation between natural and interleaved layouts is handled by specialized iterators.

```cpp
#include "cutlass/layout/tensor_interleaved.h"

// Interleaved layout with group size 4
// Elements are stored as: [a0,a1,a2,a3, b0,b1,b2,b3, ...]
// rather than: [a0,b0,c0,d0, ...]
using InterleavedLayout = cutlass::layout::TensorNCxHWx<4>;

// Transform iterator for interleaved layout
using InterleavedIterator = cutlass::transform::threadblock::PredicatedTileIterator<
    cutlass::gemm::GemmShape<128, 128>,
    cutlass::half_t,
    InterleavedLayout,
    1,
    ThreadMap
>;
```

---

## 11.6 Copy Operations and Data Movement

### 11.6.1 Global Memory to Shared Memory

The fundamental data movement in CUTLASS is loading tiles from global memory into shared memory. On SM80+ (Ampere), this uses `cp.async` instructions for asynchronous copies.

**SM80+ Asynchronous Copy (cp.async):**

```cpp
#include "cutlass/arch/memory.h"
#include "cutlass/arch/cache_operation.h"

// Asynchronous copy from global to shared memory (SM80+)
// Uses cp.async.cg (cache-global) or cp.async.ca (cache-all) instructions

// CUTLASS wraps these in the cp_async operations:
cutlass::arch::cp_async<cutlass::arch::CacheOperation::Global>(
    smem_ptr,    // Destination (shared memory)
    gmem_ptr,    // Source (global memory)
    sizeof(cutlass::Array<cutlass::half_t, 8>)  // Bytes to copy (16 bytes = 128 bits)
);

// Wait for all pending async copies to complete
cutlass::arch::cp_async_fence();
cutlass::arch::cp_async_wait<0>();  // Wait for all (count=0 means wait all)
```

**SM90+ TMA (Tensor Memory Accelerator) Copy:**

On Hopper and later, TMA provides hardware-accelerated tensor copying with built-in bounds checking and swizzling:

```cpp
// TMA copy is used through the CollectiveBuilder or directly through CuTe
// In CUTLASS 3.x, TMA is the primary copy mechanism for SM90+

// TMA handles:
// - Multi-dimensional address computation
// - Bounds checking in hardware
// - Swizzling for bank-conflict-free shared memory layout
// - Cluster-level multicast (one load broadcasts to multiple CTAs)
```

### 11.6.2 Shared Memory to Register File

Loading from shared memory to registers uses the warp-level iterators described in Section 11.4. The access patterns are carefully designed to avoid bank conflicts.

```cpp
// Shared memory to register file load (SMEM -> RF)
// This is the WarpTileIterator's primary operation

// In CUTLASS 2.x:
WarpTileIteratorA warp_iterator_A(smem_ptr_A, warp_idx);
FragmentA fragment_A;
warp_iterator_A.load(fragment_A);

// In CUTLASS 3.x with CuTe:
// auto tiled_copy = make_tiled_copy(atom, layout, size);
// copy(tiled_copy, tSrA(smem_tensor_A), tSrA(reg_tensor_A));
```

### 11.6.3 Register File to Global Memory

Writing results from registers back to global memory typically occurs in the epilogue phase. This uses predicated iterators similar to the load path but in reverse.

```cpp
// Register to global memory store (via epilogue)
// The epilogue iterator handles:
// 1. Reading from accumulator registers
// 2. Applying epilogue transformations (scale, bias, activation)
// 3. Writing to global memory with predication for boundary handling

using EpilogueIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    cutlass::gemm::GemmShape<128, 128>,
    float,
    cutlass::layout::RowMajor,
    1,
    EpilogueThreadMap
>;
```

---

## 11.7 Padding and Alignment Handling

### 11.7.1 Shared Memory Padding

Shared memory padding is critical for avoiding bank conflicts. CUTLASS automatically adds padding to shared memory allocations based on the layout and access pattern requirements.

```cpp
// Shared memory allocation with padding
// The padding amount is typically 4 bytes (2 FP16 elements or 1 FP32 element)
// This ensures that consecutive rows in shared memory start in different banks

// CUTLASS defines the padding through layout templates:
using SmemLayoutA = cutlass::layout::ColumnMajor;  // May include implicit padding

// In CUTLASS 2.x, the shared memory storage includes padding:
using SmemStorageA = cutlass::transform::threadblock::SharedStorage<
    ElementA, SmemLayoutA, ShapeA
>;
// The storage accounts for padding internally

// In CUTLASS 3.x with CuTe, padding is explicit in the layout:
// auto smem_layout = make_layout(shape, stride);
// where stride includes the padding offset
```

### 11.7.2 Alignment Requirements

CUTLASS requires that global memory pointers are aligned to specific boundaries for efficient vector memory access:

| Access Width | Alignment Requirement | Typical Types |
|---|---|---|
| 32 bits (4 bytes) | 4-byte aligned | FP32, INT32 |
| 64 bits (8 bytes) | 8-byte aligned | FP16 x 4, FP32 x 2 |
| 128 bits (16 bytes) | 16-byte aligned | FP16 x 8, FP32 x 4, TF32 x 4 |
| 256 bits (32 bytes) | 32-byte aligned | FP16 x 16 |

```cpp
// Ensure alignment when allocating tensors
// CUTLASS provides alignment checking utilities:

// The CollectiveBuilder automatically selects the appropriate alignment
// based on the data type and architecture:

using CollectiveOp = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm80, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 8,   // 8 = alignment of A in elements (e.g., 16 bytes / sizeof(half))
    ElementB, LayoutB, 8,   // 8 = alignment of B
    ElementC,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

### 11.7.3 Handling Non-Aligned Problem Sizes

When problem dimensions are not multiples of the tile dimensions, CUTLASS handles the boundary through:

1. **Predication**: Out-of-bounds threads are disabled via predicates.
2. **Residual tiles**: Special handling for the last partial tile in each dimension.
3. **Masked MMA**: On SM90+, WGMMA instructions can be partially masked.

```cpp
// PredicatedTileIterator handles boundary conditions automatically
// by computing predicates based on the thread's position and the problem size:

// Internal predicate computation (simplified):
// for each element in thread's portion:
//   if (element_row < M && element_col < K):
//     enable load/store
//   else:
//     disable (zero-fill or skip)

// The iterator's load() method applies these predicates:
typename IteratorA::Fragment fragment;
fragment.clear();  // Zero-initialize (for safety)
iterator.load(fragment);  // Only loads valid elements
```

---

## 11.8 SMEM to Register Transforms in Detail

### 11.8.1 The Transform Chain

The complete data movement chain for a GEMM operand involves multiple transform stages:

```
Stage 1: Global Memory -> Shared Memory
  - PredicatedTileIterator or TMA
  - Vectorized load (128-bit or 256-bit)
  - Bounds-checked (predicated or TMA bounds)

Stage 2: Shared Memory -> Register File (Warp-level)
  - WarpTileIterator
  - Bank-conflict-free access pattern
  - Layout transformation to MMA input format

Stage 3: Register File -> MMA Unit
  - Hardware MMA instruction (mma.sync or WGMMA)
  - No additional transform (layout already correct)

Stage 4: Accumulator -> Shared Memory (Epilogue)
  - Epilogue warp iterator
  - Optional element-wise operations (scale, bias, activation)

Stage 5: Shared Memory -> Global Memory (Epilogue)
  - PredicatedTileIterator (output)
  - Vectorized store
  - Bounds-checked
```

### 11.8.2 Fragment Layout Convention

The `Fragment` type represents data in registers. Its internal layout follows hardware conventions:

```cpp
// For mma.sync with shape 16x8x16 and FP16 inputs:
// FragmentA (for operand A):
//   - 4 x half_t values per thread
//   - Arranged such that the MMA instruction accesses them correctly
//
// The mapping from logical (row, col) to fragment index is:
//   thread_id = lane_id within the warp
//   fragment[i] corresponds to specific (row, col) based on the MMA instruction spec

// CUTLASS handles this mapping internally. The user's contract is:
// 1. Load fragment using WarpTileIterator (which knows the correct mapping)
// 2. Pass fragment to mma() operation (which knows the expected layout)
// 3. Never manually index into fragments
```

---

## 11.9 CUTLASS 3.x Transform Approach (CuTe-Based)

In CUTLASS 3.x, many of the named iterator types from CUTLASS 2.x are replaced by CuTe-based generic operations. The core transform operations use CuTe's `copy` algorithm with `TiledCopy` atoms.

```cpp
#include "cute/algorithm/copy.hpp"
#include "cute/atom/copy_atom.hpp"

// CuTe-based copy from global memory to shared memory (SM90 TMA)
// Step 1: Define a TMA copy atom
using TmaCopyAtom = decltype(make_tma_copy_atom(
    SM90_TMA_LOAD{},              // TMA load instruction
    make_layout(make_shape(128, 64))  // Tensor shape in SMEM
));

// Step 2: Create a tiled copy (tiling the atom over the full tensor)
// auto tiled_copy = make_tiled_copy(TmaCopyAtom, ...);

// Step 3: Use cute::copy to perform the data movement
// copy(tiled_copy, source_tensor, destination_tensor);
```

### 11.9.1 TiledCopy Concept

A `TiledCopy` represents a tiled version of a copy atom, distributing the copy operation across threads (or across the TMA hardware unit):

```cpp
// Define a copy atom for cp.async (SM80+)
using CopyAtomCpAsync = decltype(make_copy_atom(
    CacheAligned{},                // Copy instruction type
    decltype(make_layout(make_shape(Int<8>{}, Int<1>{})))  // Access shape
));

// Tile it for the threadblock
// auto tiled_copy = make_tiled_copy(CopyAtomCpAsync, thread_layout, access_layout);
```

### 11.9.2 Comparison: CUTLASS 2.x vs 3.x Transforms

| Concept | CUTLASS 2.x | CUTLASS 3.x |
|---|---|---|
| GMEM -> SMEM load | `PredicatedTileIterator` | CuTe `copy()` with `TiledCopy` or TMA |
| SMEM -> RF load | `WarpTileIterator` | CuTe `copy()` with warp-level `TiledCopy` |
| Layout transform | Named transform types | CuTe Layout composition |
| Bounds checking | Manual predicates | TMA hardware bounds / CuTe predicates |
| Shared memory padding | Hardcoded in layouts | CuTe swizzle and padding layouts |

---

## 11.10 Complete Example: Transform Operations in a GEMM Kernel

This example shows how all the transform components fit together in a CUTLASS 2.x GEMM kernel mainloop:

```cpp
// Simplified CUTLASS 2.x GEMM mainloop showing transform operations
template <
    typename Mma,                  // MMA operation type
    typename IteratorA,            // Global memory iterator for A
    typename IteratorB,            // Global memory iterator for B
    typename SmemIteratorA,        // Shared memory iterator for A
    typename SmemIteratorB,        // Shared memory iterator for B
    typename WarpIteratorA,        // Warp iterator for A
    typename WarpIteratorB,        // Warp iterator for B
    int Stages                     // Number of pipeline stages
>
struct GemmMainloop {
    static void run(
        IteratorA iterator_A, IteratorB iterator_B,
        SmemIteratorA smem_iterator_A, SmemIteratorB smem_iterator_B,
        typename Mma::FragmentC& accumulators,
        int gemm_k_iterations)
    {
        // Fragment storage
        typename WarpIteratorA::Fragment warp_fragment_A;
        typename WarpIteratorB::Fragment warp_fragment_B;
        typename Mma::FragmentC frag_C;
        frag_C.clear();

        // Stage 1: Load first tile from global to shared memory
        typename IteratorA::Fragment gm_fragment_A;
        typename IteratorB::Fragment gm_fragment_B;
        iterator_A.load(gm_fragment_A);
        iterator_B.load(gm_fragment_B);
        smem_iterator_A.store(gm_fragment_A);
        smem_iterator_B.store(gm_fragment_B);
        __syncthreads();

        for (int k_iter = 0; k_iter < gemm_k_iterations; ++k_iter) {
            // Stage 2: Load from shared memory to registers (warp-level transform)
            WarpIteratorA warp_iter_A(smem_iterator_A);
            WarpIteratorB warp_iter_B(smem_iterator_B);
            warp_iter_A.load(warp_fragment_A);
            warp_iter_B.load(warp_fragment_B);

            // Stage 3: MMA operation (registers -> accumulator)
            Mma::mma(accumulators, warp_fragment_A, warp_fragment_B, accumulators);
        }
    }
};
```

### CUTLASS 3.x Equivalent (with CuTe):

```cpp
// CUTLASS 3.x mainloop using CuTe transforms (simplified)
template <typename CollectiveOp>
void gemm_mainloop(CollectiveOp collective,
                   typename CollectiveOp::TensorA tCrA,   // Register tensor A
                   typename CollectiveOp::TensorB tCrB,   // Register tensor B
                   typename CollectiveOp::TensorC tCrC,   // Accumulator tensor C
                   typename CollectiveOp::TmaLoaderA tma_load_A,
                   typename CollectiveOp::TmaLoaderB tma_load_B,
                   int k_tiles)
{
    // TMA load handles GMEM -> SMEM transform
    // CuTe copy handles SMEM -> RF transform
    // GMMA handles RF -> Accumulator transform

    for (int k = 0; k < k_tiles; ++k) {
        // Issue TMA load for next tile
        collective.tma_load_A(tma_load_A, k);
        collective.tma_load_B(tma_load_B, k);

        // Wait for current tile
        collective.barrier_arrive();
        collective.barrier_wait();

        // Copy from SMEM to RF (CuTe copy)
        copy(collective.smem_layout_A(), collective.smem_tensor_A(k), tCrA);
        copy(collective.smem_layout_B(), collective.smem_tensor_B(k), tCrB);

        // GMMA operation
        cute::gemm(collective.tiled_mma(), tCrA, tCrB, tCrC);
    }
}
```

---

## 11.11 Key Header Files Reference

| Header | Purpose |
|---|---|
| `cutlass/transform/threadblock/predicated_tile_iterator.h` | Bounds-checked tile loading/storing |
| `cutlass/transform/threadblock/regular_tile_iterator.h` | Dense tile iteration without predication |
| `cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h` | 2D thread tile output iterator |
| `cutlass/transform/threadblock/predicated_tile_iterator_residual.h` | Residual tile handling |
| `cutlass/transform/thread/transpose.h` | Thread-level matrix transposition |
| `cutlass/transform/thread/scale_bias.h` | Thread-level scale and bias |
| `cutlass/transform/thread/planar_complex.h` | Complex number layout transforms |
| `cutlass/transform/warp/warp_tile_iterator.h` | Warp-level shared memory to register load |
| `cutlass/transform/warp/warp_tile_iterator_mma.h` | Warp tile iterator for MMA input |
| `cutlass/transform/warp/warp_tile_iterator_mma_tensor_op.h` | Tensor Core specific warp iterator |
| `cutlass/transform/warp/warp_tile_iterator_mma_multistage.h` | Multi-stage pipeline warp iterator |
| `cute/algorithm/copy.hpp` | CuTe generic copy algorithm |
| `cute/atom/copy_atom.hpp` | CuTe copy atom definitions |
| `cute/atom/mma_atom.hpp` | CuTe MMA atom definitions |

---

## 11.12 Summary

Transform operations form the data movement backbone of CUTLASS. They handle:

1. **Loading tiles** from global memory to shared memory with predication and vectorization.
2. **Transforming layouts** between the natural data format and the hardware-required format for Tensor Core operations.
3. **Moving data** between shared memory and register files with bank-conflict-free access patterns.
4. **Storing results** from registers back to global memory in the epilogue.
5. **Handling edge cases** through predication, residual tiles, and alignment management.

In CUTLASS 3.x, many of these operations are unified under the CuTe library, which provides a generic `copy()` algorithm parameterized by copy atoms and tiled copies. This approach replaces the named iterator types of CUTLASS 2.x with a more composable and flexible system.
