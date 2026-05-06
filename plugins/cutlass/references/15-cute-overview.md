# CUTLASS - Chapter 15: CuTe Library Overview

This reference provides a comprehensive overview of the CuTe library, a modern C++ tensor abstraction for GPU computing that forms the foundation of CUTLASS 3.x. CuTe provides composable, rank-agnostic, and compile-time-optimized abstractions for tensors, layouts, and hardware-accelerated operations.

---

## 15.1 What is CuTe?

CuTe (pronounced "cute") is a C++ template library within CUTLASS that provides a formal, composable abstraction for GPU tensor programming. It was introduced with CUTLASS 3.0 as a replacement for the numerous named iterator types and ad-hoc indexing schemes used in CUTLASS 2.x. CuTe enables developers to write GPU micro-kernels (small, composable computational units) that are:

- **Rank-agnostic**: Operations work on tensors of any dimensionality (1D, 2D, 3D, etc.).
- **Layout-composable**: Layouts can be composed, sliced, and transformed algebraically.
- **Compile-time optimized**: Extensive use of C++ template metaprogramming enables the compiler to eliminate abstraction overhead entirely.
- **Hardware-aware**: Direct mapping to GPU hardware primitives (shared memory banks, Tensor Core instructions, TMA descriptors).

CuTe replaces approximately 200+ named iterator and transform types from CUTLASS 2.x with a small set of composable primitives:

| CUTLASS 2.x | CuTe (3.x) |
|---|---|
| `PredicatedTileIterator` | `copy()` with `TiledCopy` and predicates |
| `WarpTileIterator` | `copy()` with warp-level `TiledCopy` |
| `RegularTileIterator` | `copy()` with `TiledCopy` |
| `Mma` (warp-level) | `gemm()` with `TiledMMA` |
| `Transform` types | Layout composition |
| Epilogue iterators | `copy()` with epilogue `TiledCopy` |

---

## 15.2 Design Philosophy

CuTe is built on three core design principles:

### 15.2.1 Composability over Naming

Instead of creating a new named type for each hardware configuration (as in CUTLASS 2.x), CuTe provides a small set of composable building blocks:

```cpp
// CUTLASS 2.x: Different named types for each configuration
// WarpTileIterator<Shape, Element, RowMajor, InstructionShape, 128>
// WarpTileIterator<Shape, Element, ColumnMajor, InstructionShape, 128>
// PredicatedTileIterator<Shape, Element, RowMajor, 0, ThreadMap>
// PredicatedTileIterator<Shape, Element, ColumnMajor, 1, ThreadMap>

// CuTe: Same operations expressed through composition
// auto tiled_copy = make_tiled_copy(copy_atom, thread_layout, access_layout);
// copy(tiled_copy, source_tensor, destination_tensor);
// Works for any layout, any dimensionality, any hardware target
```

### 15.2.2 Rank-Agnostic Design

CuTe operations work uniformly across tensors of any rank. A `copy()` operation handles 1D, 2D, 3D, and higher-dimensional tensors identically:

```cpp
// 1D copy
auto tensor_1d = make_tensor(ptr, make_layout(make_shape(128)));
copy(src_1d, dst_1d);

// 2D copy
auto tensor_2d = make_tensor(ptr, make_layout(make_shape(128, 64), make_stride(64, 1)));
copy(src_2d, dst_2d);

// 3D copy (same operation, different tensor rank)
auto tensor_3d = make_tensor(ptr, make_layout(make_shape(32, 16, 8), make_stride(128, 8, 1)));
copy(src_3d, dst_3d);

// The copy() algorithm handles all ranks uniformly
```

### 15.2.3 Compile-Time Optimization

CuTe leverages C++ template metaprogramming to push as much computation as possible to compile time. Layout computations, index arithmetic, and loop bounds are often resolved entirely by the compiler:

```cpp
// Static shapes and strides are encoded in the type system
using Shape = cute::Shape<cute::Int<128>, cute::Int<64>>;  // Compile-time constants
using Stride = cute::Stride<cute::Int<64>, cute::Int<1>>;

// The compiler can:
// 1. Unroll all loops over static dimensions
// 2. Eliminate dead code (unused dimensions)
// 3. Precompute all index arithmetic
// 4. Generate optimal memory access patterns

// Compare with dynamic (runtime) values:
auto dynamic_shape = cute::make_shape(128, 64);  // Values known at runtime
// Still works, but the compiler cannot optimize as aggressively
```

---

## 15.3 Core Abstractions

CuTe provides four fundamental abstractions that together enable composable GPU programming:

### 15.3.1 Layout

A **Layout** is a pair of a Shape and a Stride that maps multi-dimensional indices to flat memory offsets. It is the mathematical foundation of CuTe's index space.

**Formal definition:**

```
Layout: Shape x Stride -> Int
layout(coords) = inner_product(flatten(coords), flatten(stride))
```

**Creating layouts:**

```cpp
#include "cute/layout.hpp"

// Simple 1D layout: 128 elements with stride 1
auto layout_1d = make_layout(make_shape(128));
// Equivalent to: shape=(128,), stride=(1,)
// layout(i) = i * 1 = i  (for i in [0, 128))

// 2D layout: 128x64 matrix, row-major
auto layout_rm = make_layout(make_shape(128, 64), make_stride(64, 1));
// layout(i, j) = i * 64 + j * 1

// 2D layout: 128x64 matrix, column-major
auto layout_cm = make_layout(make_shape(128, 64), make_stride(1, 128));
// layout(i, j) = i * 1 + j * 128

// Compile-time static layout (fully known at compile time)
auto static_layout = make_layout(
    make_shape(Int<128>{}, Int<64>{}),
    make_stride(Int<64>{}, Int<1>{})
);

// Mixed static/dynamic layout
auto mixed_layout = make_layout(
    make_shape(Int<128>{}, 64),   // Static M, dynamic N
    make_stride(Int<64>{}, Int<1>{})
);

// Hierarchical (nested) layout: groups of elements
auto hier_layout = make_layout(
    make_shape(make_shape(4, 8), make_shape(16, 2))
    // Shape is (4,8) x (16,2) = a 2D layout with nested shapes
);
```

**Layout operations:**

```cpp
// Composition: create a new layout by composing two layouts
auto layout_a = make_layout(make_shape(128, 64), make_stride(64, 1));
auto layout_b = make_layout(make_shape(32, 4));       // Default stride: (4, 1)
auto composed = composition(layout_a, layout_b);
// Composed layout maps layout_b's coordinates through layout_a

// Slice: take a sub-range of a layout
auto sliced = layout_a(make_shape(32, _), make_shape(_, 16));
// Takes rows [0, 32) and columns [0, 16) of the original layout

// Coalesce: merge dimensions with unit stride
auto coalesced = coalesce(layout);
// Removes dimensions of size 1 and merges adjacent unit-stride dimensions

// Filter: remove dimensions of size 1
auto filtered = filter(layout);

// Flatten: convert to 1D layout
auto flat = flatten(layout);
```

### 15.3.2 Tensor

A **Tensor** is a pointer backed by a Layout. It associates data in memory (via a pointer) with an index space (via a Layout).

**Creating tensors:**

```cpp
#include "cute/tensor.hpp"

// Create a tensor from a raw pointer and a layout
float* ptr = /* ... */;
auto layout = make_layout(make_shape(128, 64), make_stride(64, 1));
auto tensor = make_tensor(ptr, layout);

// Access elements (for device code)
// float val = tensor(i, j);   // Read element at (i, j)
// tensor(i, j) = val;          // Write element at (i, j)

// Create a tensor with packed layout (inferred from shape)
auto packed_tensor = make_tensor(ptr, make_shape(128, 64));
// Row-major by default: stride = (64, 1)

// Create a tensor for shared memory
__shared__ float smem[128 * 64];
auto smem_tensor = make_tensor(make_smem_ptr(smem),
                               make_layout(make_shape(128, 64), make_stride(64, 1)));

// Create a tensor for register file
auto reg_tensor = make_tensor<float>(make_layout(make_shape(16, 8)));
// Allocates 128 floats in registers (if used in device code)
```

**Tensor engines:**

CuTe tensors can be backed by different memory spaces, distinguished by their **engine** type:

```cpp
// Global memory tensor
auto gmem_tensor = make_tensor(make_gmem_ptr(ptr), layout);

// Shared memory tensor
auto smem_tensor = make_tensor(make_smem_ptr(smem_ptr), layout);

// Register tensor (statically allocated)
auto reg_tensor = make_tensor<float>(layout);

// The engine type determines the memory space and affects:
// - Which operations are legal (e.g., TMA only from global to shared)
// - Access patterns and synchronization requirements
// - Copy atom selection
```

**Tensor slicing and partitioning:**

```cpp
// Slice: create a sub-tensor by fixing some dimensions
auto row_slice = tensor(32, _);     // Row 32, all columns (1D tensor)
auto col_slice = tensor(_, 16);     // All rows, column 16 (1D tensor)
auto block_slice = tensor(make_coord(32, 64), make_coord(16, 16));
// 16x16 sub-block starting at (32, 64)

// Partition: divide a tensor among threads/warps
// This is the key operation for distributing work across threads
auto tiled_copy = make_tiled_copy(copy_atom, thread_layout, access_layout);
auto thr_tensor = tiled_copy.partition_S(tensor);  // Partition source
auto thr_dst = tiled_copy.partition_D(dst_tensor); // Partition destination
// Each thread gets a local view of its portion of the tensor
```

### 15.3.3 Atom

An **Atom** is a hardware-accelerated primitive operation. CuTe defines atoms for copy operations and MMA (Matrix Multiply-Accumulate) operations that directly map to GPU hardware instructions.

**Copy atoms:**

```cpp
#include "cute/atom/copy_atom.hpp"

// Copy atoms represent hardware copy instructions:

// SM90 TMA load (Tensor Memory Accelerator)
auto tma_atom = SM90_TMA_LOAD{};
// Hardware: TMA unit on Hopper+
// Capable of multi-dimensional, bounds-checked, swizzled copies

// SM90 TMA multicast load
auto tma_mcast_atom = SM90_TMA_LOAD_MULTICAST{};

// SM90 TMA store
auto tma_store_atom = SM90_TMA_STORE{};

// SM80 cp.async (asynchronous global-to-shared copy)
auto cp_async_atom = SM80_CP_ASYNC_CACHEALWAYS{};
auto cp_async_global_atom = SM80_CP_ASYNC_CACHEGLOBAL{};

// Universal copy (fallback for any memory space pair)
auto universal_copy = AutoVectorizingCopyWithAssumedAlignment<128>{};

// Warp-level shared memory to register copy
auto smem_to_reg_atom = SM75_U32x4_LDSM_N{};
```

**MMA atoms:**

```cpp
#include "cute/atom/mma_atom.hpp"

// MMA atoms represent hardware matrix multiply instructions:

// SM90 WGMMA (Warp Group MMA) for FP16
auto wgmma_f16 = SM90_64x64x16_F16F16F16_SS{};
// Shape: 64x64x16, types: F16xF16->F16, source: SMEMxSMEM

// SM90 WGMMA for TF32
auto wgmma_tf32 = SM90_64x64x16_TF32F32F32_SS{};

// SM90 WGMMA for FP8
auto wgmma_fp8 = SM90_64x128x32_F8F8F32_SS_TN{};

// SM80 mma.sync for FP16
auto mma_f16 = SM80_16x8x16_F16F16F16F16TNTN{};
// Shape: 16x8x16, types: F16xF16->F16, layout: TN (transposed, non-transposed)

// SM80 mma.sync for TF32
auto mma_tf32 = SM80_16x8x8_TF32F32F32F32TNTN{};

// SM75 mma.sync for INT8
auto mma_int8 = SM75_16x8x16_S32S32S32S32TNTN{};
```

### 15.3.4 TiledMMA and TiledCopy

**TiledMMA** and **TiledCopy** are tiled versions of atoms that distribute the atom's operation across multiple threads or warp groups:

**TiledCopy:**

```cpp
#include "cute/atom/copy_atom.hpp"

// Create a tiled copy: distributes a copy atom across threads
// auto tiled_copy = make_tiled_copy(
//     copy_atom,           // The hardware copy primitive
//     thread_layout,       // How threads are mapped to the copy
//     access_layout        // How accesses are organized within each thread
// );

// Example: Tiled copy for SM80 global-to-shared
auto tiled_copy = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL, half_t>{},
    make_layout(make_shape(Int<32>{}, Int<4>{})),   // 32x4 thread mapping
    make_layout(make_shape(Int<4>{}))                // 4 elements per access
);

// Use the tiled copy to partition tensors
auto thr_src = tiled_copy.partition_S(gmem_tensor);
auto thr_dst = tiled_copy.partition_D(smem_tensor);

// Execute the copy
copy(tiled_copy, thr_src, thr_dst);
```

**TiledMMA:**

```cpp
#include "cute/atom/mma_atom.hpp"

// Create a tiled MMA: distributes MMA atom across threads in a warp (group)
auto tiled_mma = make_tiled_mma(
    SM90_64x64x16_F16F16F32_SS{},
    make_layout(make_shape(Int<2>{}, Int<2>{}, Int<1>{}))  // Tiling: 2x2x1
);
// This creates a tiled MMA that computes a 128x128x16 GEMM
// by tiling the 64x64x16 atom in a 2x2 pattern

// Partition tensors for the MMA
auto thr_A = tiled_mma.partition_A(smem_tensor_A);  // A operand partitioned
auto thr_B = tiled_mma.partition_B(smem_tensor_B);  // B operand partitioned
auto thr_C = tiled_mma.partition_C(reg_tensor_C);   // C accumulator partitioned

// Execute the MMA
gemm(tiled_mma, thr_A, thr_B, thr_C);
```

---

## 15.4 Hierarchy: Atom > Tiled Op > Collective > Kernel > Device

CuTe operations are organized in a clear hierarchy that maps to the GPU execution model:

```
Device Level (cutlass::gemm::device::GemmUniversalAdapter)
  |  Manages kernel launch, workspace, and device-level coordination
  |
Kernel Level (cutlass::gemm::kernel::GemmUniversal)
  |  Manages grid-level programming, tile scheduling, epilogue
  |
Collective Level (cutlass::gemm::collective::*)
  |  Manages threadblock-level mainloop, shared memory, pipeline
  |  Composes TiledMMA and TiledCopy operations
  |
Tiled Operation Level (CuTe: TiledMMA, TiledCopy)
  |  Distributes an atom across threads/warps
  |  Handles partitioning of tensors
  |
Atom Level (CuTe: Copy_Atom, MMA_Atom)
     Hardware-accelerated primitive instruction
     (WGMMA, TMA, mma.sync, cp.async, etc.)
```

**Each level is independently composable:**

```cpp
// Level 1: Define an atom (hardware primitive)
auto mma_atom = SM90_64x64x16_F16F16F32_SS{};

// Level 2: Tile it for the threadblock
auto tiled_mma = make_tiled_mma(mma_atom, tile_layout);

// Level 3: Use it in a collective (mainloop)
// The collective manages the pipeline, shared memory, and calls gemm()

// Level 4: Wrap in a kernel (handles grid programming)
// The kernel manages tile assignment and epilogue

// Level 5: Wrap in a device operation (handles launch)
// The device operation manages CUDA launch configuration
```

---

## 15.5 CuTe vs CUTLASS 2.x Iterators

### 15.5.1 Conceptual Mapping

| CUTLASS 2.x Concept | CuTe Equivalent |
|---|---|
| `PredicatedTileIterator` | `TiledCopy` with predicate tensor |
| `RegularTileIterator` | `TiledCopy` (unpredicated) |
| `WarpTileIterator` | `TiledCopy` (warp-level) |
| `Mma` (warp-level) | `TiledMMA` + `gemm()` |
| `Fragment` | CuTe register `Tensor` |
| Thread map | Layout composition |
| SMEM padding | CuTe swizzle layout |
| Boundary predicates | CuTe predicate tensor or TMA bounds |

### 15.5.2 Code Comparison: Loading a Tile

**CUTLASS 2.x:**

```cpp
// CUTLASS 2.x: Named iterator type with hardcoded logic
using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
    cutlass::gemm::GemmShape<128, 32>,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    1,
    ThreadMapA
>;

IteratorA iterator(params, ptr_A, problem_size, thread_idx, block_idx);
typename IteratorA::Fragment fragment;
iterator.load(fragment);  // Loads predicated tile into fragment
```

**CuTe (CUTLASS 3.x):**

```cpp
// CuTe: Generic copy with tiled copy atom
auto tiled_copy = make_tiled_copy(
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, half_t>{},
    thread_layout,
    access_layout
);

auto thr_gmem = tiled_copy.partition_S(gmem_tensor);
auto thr_smem = tiled_copy.partition_D(smem_tensor);

// Compute predicates for boundary handling
auto predicates = make_predicate_tensor(shape(thr_gmem), problem_bounds);

// Execute copy (with predication)
copy(tiled_copy, thr_gmem, thr_smem, predicates);
```

### 15.5.3 Advantages of CuTe

1. **Fewer types**: One `copy()` function replaces dozens of iterator types.
2. **Better composability**: Layout operations compose mathematically.
3. **Compile-time optimization**: More information available to the compiler.
4. **Easier debugging**: CuTe layouts can be printed and inspected at compile time.
5. **Forward compatibility**: New hardware only needs new atoms, not new iterators.

---

## 15.6 Layout Algebra Fundamentals

CuTe provides a formal layout algebra that enables precise manipulation of index-to-offset mappings.

### 15.6.1 Shape and Stride

```cpp
// A Layout is defined by its Shape and Stride:
// Shape: defines the extent of each dimension
// Stride: defines the offset multiplier for each dimension

// 1D layout
auto L1 = make_layout(make_shape(128));
// Shape: (128), Stride: (1) [implicit]
// L1(i) = i for i in [0, 128)

// 2D layout (row-major)
auto L2 = make_layout(make_shape(32, 64), make_stride(64, 1));
// Shape: (32, 64), Stride: (64, 1)
// L2(i, j) = i * 64 + j * 1

// 3D layout
auto L3 = make_layout(make_shape(4, 8, 16), make_stride(128, 16, 1));
// Shape: (4, 8, 16), Stride: (128, 16, 1)
// L3(i, j, k) = i * 128 + j * 16 + k * 1

// The size of a layout:
auto sz = size(L2);  // 32 * 64 = 2048
```

### 15.6.2 Composition

Layout composition creates a new layout by applying one layout's index space through another:

```cpp
// Composition: compose(A, B) creates layout C such that
// C(x) = A(B(x))

auto A = make_layout(make_shape(128), make_stride(1));      // Identity-like: A(i) = i
auto B = make_layout(make_shape(32, 4), make_stride(4, 1)); // B(i,j) = 4i + j

auto C = composition(A, B);
// C(i, j) = A(B(i, j)) = 4i + j
// C has Shape (32, 4) and Stride (4, 1)

// Common use: tiling a tensor
// The tile layout B "selects" elements from the base layout A
```

### 15.6.3 Complement

The complement finds a layout that covers the remaining elements not covered by a sub-layout:

```cpp
// complement(A, size) finds a layout B such that
// composition(A, B) covers [0, size)

// Used for finding "the rest of the tensor" after a tile
auto tile = make_layout(make_shape(32));      // A tile of 32 elements
auto rest = complement(tile, 128);            // The rest: 4 tiles of 32
// rest has Shape (4) and maps to offsets [32, 64, 96, 128)
```

### 15.6.4 Zipped and Tiled Layouts

```cpp
// Zip: combine inner dimensions into groups
// Unzip: separate grouped dimensions

// These are used to create the tiled layout from an atom layout:
// auto tiled_layout = zipped_divide(layout, tile_shape);
// Divides the layout into tiles of tile_shape, creating a
// (num_tiles, tile_elements) layout
```

---

## 15.7 Rank-Agnostic Design

CuTe operations are designed to work with tensors of any rank. This is achieved through recursive layout handling and variadic template programming.

```cpp
// Rank-agnostic copy: works for any tensor rank
template <class SrcTensor, class DstTensor>
void my_copy(SrcTensor const& src, DstTensor& dst) {
    // cute::copy handles any rank automatically
    copy(src, dst);
}

// Rank-agnostic GEMM: works for any tile dimensions
template <class MMA, class TensorA, class TensorB, class TensorC>
void my_gemm(MMA mma, TensorA A, TensorB B, TensorC C) {
    // cute::gemm handles any tiled MMA configuration
    gemm(mma, A, B, C);
}

// Rank-agnostic fill: initialize any tensor
template <class Tensor, class Value>
void my_fill(Tensor tensor, Value val) {
    // cute::fill handles any tensor shape
    fill(tensor, val);
}
```

**How rank-agnosticism works internally:**

```cpp
// CuTe flattens hierarchical shapes into a 1D index space internally
// A Shape of ((4, 8), (16, 2)) flattens to a single 1D coordinate
// The Layout maps this 1D coordinate to a memory offset

// This means:
// 1. All operations are ultimately 1D in the index space
// 2. Multi-dimensional access is syntactic sugar
// 3. The compiler can optimize the flat index computation

// Example: 2D access tensor(i, j) is equivalent to tensor(i * stride_j + j * stride_i)
// which is tensor(make_coord(i, j)) which is tensor(i * stride<0> + j * stride<1>)
```

---

## 15.8 Static vs Dynamic Layouts

CuTe distinguishes between static (compile-time) and dynamic (runtime) layout components.

### 15.8.1 Static Layouts

```cpp
// All dimensions and strides are compile-time constants
// Uses cute::Int<N> for static values
using StaticShape = cute::Shape<cute::Int<128>, cute::Int<64>>;
using StaticStride = cute::Stride<cute::Int<64>, cute::Int<1>>;

auto static_layout = make_layout(
    cute::Shape<cute::Int<128>, cute::Int<64>>{},
    cute::Stride<cute::Int<64>, cute::Int<1>>{}
);

// Benefits:
// 1. Compiler can unroll all loops
// 2. All index arithmetic is constant-folded
// 3. Dead code elimination for unused dimensions
// 4. Register allocation is fully determined
// 5. Zero runtime overhead
```

### 15.8.2 Dynamic Layouts

```cpp
// Dimensions or strides are runtime values
int M = problem_size.m();
int N = problem_size.n();

auto dynamic_layout = make_layout(
    make_shape(M, N),           // Dynamic shape
    make_stride(N, Int<1>{})    // Mixed: dynamic N stride, static column stride
);

// Benefits:
// 1. Supports variable problem sizes
// 2. Same code handles multiple configurations
// 3. Useful for host-side tensor setup

// Drawbacks:
// 1. Compiler cannot unroll loops over dynamic dimensions
// 2. Some index arithmetic happens at runtime
// 3. Less aggressive optimization
```

### 15.8.3 Best Practice: Maximize Static Information

```cpp
// Prefer static layouts for kernel-internal operations:
// Good: tile shapes and thread maps are fully static
auto tile_layout = make_layout(
    make_shape(Int<128>{}, Int<64>{}),
    make_stride(Int<64>{}, Int<1>{})
);

// Only use dynamic layouts for problem-size-dependent tensors:
// The problem size is inherently dynamic
auto problem_layout = make_layout(
    make_shape(M, N, K),
    make_stride(N * K, K, Int<1>{})
);

// CuTe automatically optimizes the static portions
// of mixed static/dynamic layouts
```

---

## 15.9 Compile-Time Optimization

CuTe achieves zero-overhead abstraction through several compile-time techniques:

### 15.9.1 Static Loop Unrolling

```cpp
// When shapes are static, CuTe automatically unrolls loops:
template <int N>
struct UnrolledCopy {
    static void copy(float* dst, float const* src) {
        // The compiler unrolls this completely when N is a compile-time constant
        CUTE_UNROLL
        for (int i = 0; i < N; ++i) {
            dst[i] = src[i];
        }
    }
};

// CuTe uses CUTE_UNROLL and #pragma unroll directives
// extensively to ensure static loops are fully unrolled
```

### 15.9.2 Type-Level Layout Encoding

```cpp
// Layouts are encoded in the C++ type system:
// make_layout(make_shape(Int<128>{}), make_stride(Int<1>{}))
// has type: Layout<Shape<Int<128>>, Stride<Int<1>>>

// This means:
// 1. The compiler knows the exact layout at compile time
// 2. No runtime dispatch is needed
// 3. The generated code is equivalent to hand-written indexing
// 4. Type checking catches layout errors at compile time
```

### 15.9.3 Expression Templates

```cpp
// CuTe uses expression templates to defer computation:
// tensor(i, j) does not immediately compute the offset
// Instead, it creates an expression that the compiler can optimize

// When combined with static layouts, the compiler can often
// reduce the entire expression to a single register offset:
// tensor(Int<3>{}, Int<7>{}) with stride (64, 1)
//   -> 3 * 64 + 7 * 1 = 199
//   -> Compiler generates: ld.shared.f32 %r0, [smem + 199*4]
```

---

## 15.10 CuTe Subdirectories

The CuTe library is organized into the following subdirectories within `include/cute/`:

### 15.10.1 algorithm/

Core algorithms that operate on CuTe tensors:

```
cute/algorithm/
  copy.hpp         - Generic copy algorithm (GMEM<->SMEM<->RF)
  gemm.hpp         - Generic GEMM algorithm (tiled MMA)
  fill.hpp         - Fill a tensor with a value
  prefetch.hpp     - Prefetch tensor data
  axpby.hpp        - A*x + B*y (element-wise)
  cooperative.hpp  - Cooperative algorithms across thread groups
```

```cpp
#include "cute/algorithm/copy.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/fill.hpp"

// copy(src_tensor, dst_tensor) or copy(tiled_copy, src, dst)
// gemm(tiled_mma, A, B, C)
// fill(tensor, value)
```

### 15.10.2 arch/

Hardware architecture-specific operations and atom definitions:

```
cute/arch/
  copy_sm75.hpp      - SM75 copy atoms (LDSM)
  copy_sm80.hpp      - SM80 copy atoms (cp.async)
  copy_sm90_tma.hpp  - SM90 TMA copy atoms
  mma_sm75.hpp       - SM75 MMA atoms
  mma_sm80.hpp       - SM80 MMA atoms
  mma_sm90.hpp       - SM90 WGMMA atoms
  mma_sm100.hpp      - SM100 UMMA atoms
  util.hpp           - Architecture utility functions
```

### 15.10.3 atom/

High-level atom abstractions and tiling:

```
cute/atom/
  copy_atom.hpp    - Copy atom definition and make_tiled_copy
  mma_atom.hpp     - MMA atom definition and make_tiled_mma
  mma_traits.hpp   - MMA atom traits (types, shapes, threads)
  copy_traits.hpp  - Copy atom traits
```

### 15.10.4 container/

Container types used throughout CuTe:

```
cute/container/
  array.hpp     - Static array with compile-time size
  tuple.hpp     - Heterogeneous tuple (CUTLASS's own implementation)
  alignment.hpp - Alignment utilities
```

### 15.10.5 numeric/

Numeric types and operations:

```
cute/numeric/
  integer.hpp      - Compile-time integer types (Int<N>)
  math.hpp         - Math utilities (max, min, abs, etc.)
  numeric_types.hpp - Numeric type definitions
  complex.hpp      - Complex number support
  real.hpp         - Real number operations
```

### 15.10.6 util/

Utility functions and debugging tools:

```
cute/util/
  debug.hpp      - Print and debug utilities for layouts and tensors
  type_traits.hpp - Type trait utilities
  print.hpp      - Pretty-printing for CuTe types (host-side)
```

---

## 15.11 Integration with CUTLASS 3.x

CuTe is the foundation of CUTLASS 3.x. All CUTLASS 3.x operations are built on CuTe abstractions:

### 15.11.1 Collective Operations

The collective (mainloop) in CUTLASS 3.x is composed from CuTe operations:

```cpp
// CUTLASS 3.x collective mainloop (simplified structure):
template <class... Args>
struct CollectiveMma {
    // CuTe types for the mainloop
    using TiledMma = decltype(make_tiled_mma(mma_atom, tile_layout));
    using TiledCopyA = decltype(make_tiled_copy(copy_atom_A, ...));
    using TiledCopyB = decltype(make_tiled_copy(copy_atom_B, ...));

    // Storage: shared memory tensors defined by CuTe layouts
    struct SharedStorage {
        cute::ArrayEngine<ElementA, size(smem_layout_A)> smem_A;
        cute::ArrayEngine<ElementB, size(smem_layout_B)> smem_B;
    };

    // Mainloop using CuTe operations
    void operator()(/* params */) {
        // 1. TMA load (GMEM -> SMEM) via CuTe copy
        copy(tma_copy, gmem_tensor, smem_tensor);

        // 2. SMEM -> RF copy via CuTe copy
        copy(tiled_copy_A, thr_smem_A, thr_reg_A);
        copy(tiled_copy_B, thr_smem_B, thr_reg_B);

        // 3. MMA via CuTe gemm
        gemm(tiled_mma, thr_reg_A, thr_reg_B, thr_reg_C);
    }
};
```

### 15.11.2 Epilogue Operations

CUTLASS 3.x epilogues use CuTe for output writing:

```cpp
// Epilogue: accumulator -> output
// Uses CuTe copy with epilogue-specific tiled copy
// that applies element-wise operations (scale, bias, activation)

// The epilogue performs:
// 1. Read accumulators (CuTe tensor partitioned by threads)
// 2. Apply element-wise operations (scale, bias, activation)
// 3. Convert to output type
// 4. Write to global memory (CuTe copy with predication)
```

---

## 15.12 CuTe DSL (Python Interface)

CuTe DSL provides a Python interface for writing CuTe kernels, enabling rapid prototyping and testing of GPU kernels in Python before porting to C++.

### 15.12.1 Overview

```python
# CuTe DSL allows writing CuTe operations in Python
# with direct mapping to the C++ CuTe API

from cutlass.cute import *

# Define layouts (same as C++ CuTe)
layout = Layout(Shape(128, 64), Stride(64, 1))

# Define tensors
# tensor = Tensor(ptr, layout)

# Define copy and MMA operations
# tiled_copy = make_tiled_copy(...)
# tiled_mma = make_tiled_mma(...)

# Write kernel logic using the same abstractions
```

### 15.12.2 Integration with CUTLASS Python Interface

```python
# CuTe DSL integrates with the CUTLASS Python interface
# for defining and launching GEMM, convolution, and other operations

import cutlass
from cutlass import LayoutType, DataType

# Define a GEMM operation using the Python interface
gemm = cutlass.Gemm(
    element_A=DataType.float16,
    element_B=DataType.float16,
    element_C=DataType.float32,
    layout_A=LayoutType.RowMajor,
    layout_B=LayoutType.ColumnMajor,
    # CuTe DSL can be used to customize internal operations
)
```

---

## 15.13 Complete Code Examples

### 15.13.1 Simple Copy Kernel Using CuTe

```cpp
#include "cute/cute.hpp"
#include "cute/algorithm/copy.hpp"

// A simple kernel that copies data from global to global memory
// using CuTe's copy algorithm
template <class TensorSrc, class TensorDst>
__global__ void cute_copy_kernel(TensorSrc src, TensorDst dst) {
    // Get the thread's index within the CTA
    auto tid = threadIdx.x;
    auto bid = blockIdx.x;

    // Create a tiled copy: each thread copies multiple elements
    auto tiled_copy = make_tiled_copy(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, typename TensorSrc::value_type>{},
        make_layout(make_shape(Int<128>{})),    // 128 threads
        make_layout(make_shape(Int<4>{}))       // 4 elements per thread per access
    );

    // Partition the tensors for this thread
    auto thr_src = tiled_copy.partition_S(src);
    auto thr_dst = tiled_copy.partition_D(dst);

    // Execute the copy
    copy(tiled_copy, thr_src, thr_dst);
}
```

### 15.13.2 GEMM Kernel Using CuTe

```cpp
#include "cute/cute.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"

// A simple GEMM kernel using CuTe directly
template <class TiledMMA, class TiledCopyA, class TiledCopyB,
          class GmemTensorA, class GmemTensorB, class GmemTensorC,
          class SmemLayoutA, class SmemLayoutB>
__global__ void cute_gemm_kernel(
    TiledMMA tiled_mma,
    TiledCopyA tiled_copy_a, TiledCopyB tiled_copy_b,
    GmemTensorA gmem_A, GmemTensorB gmem_B, GmemTensorC gmem_C,
    SmemLayoutA smem_layout_a, SmemLayoutB smem_layout_b,
    int k_tiles)
{
    using ElementA = typename GmemTensorA::value_type;
    using ElementB = typename GmemTensorB::value_type;
    using ElementC = typename GmemTensorC::value_type;

    // Allocate shared memory
    extern __shared__ char smem[];
    auto smem_A = make_tensor(make_smem_ptr<ElementA>(smem), smem_layout_a);
    auto smem_B = make_tensor(make_smem_ptr<ElementB>(smem + sizeof(ElementA) * size(smem_layout_a)),
                              smem_layout_b);

    // Partition tensors for this thread
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto thr_A = thr_mma.partition_A(smem_A);    // Thread's portion of A
    auto thr_B = thr_mma.partition_B(smem_B);    // Thread's portion of B
    auto thr_C = thr_mma.partition_C(gmem_C);    // Thread's portion of C

    // Allocate registers for the MMA
    auto reg_A = make_tensor<ElementA>(shape(thr_A));
    auto reg_B = make_tensor<ElementB>(shape(thr_B));
    auto accum = make_tensor<ElementC>(shape(thr_C));
    clear(accum);

    // Main GEMM loop
    for (int k = 0; k < k_tiles; ++k) {
        // Load A and B tiles from global to shared memory
        auto gmem_tile_A = gmem_A(_, _, k);
        auto gmem_tile_B = gmem_B(_, _, k);

        auto thr_gA = tiled_copy_a.partition_S(gmem_tile_A);
        auto thr_gB = tiled_copy_b.partition_S(gmem_tile_B);
        auto thr_sA = tiled_copy_a.partition_D(smem_A);
        auto thr_sB = tiled_copy_b.partition_D(smem_B);

        copy(tiled_copy_a, thr_gA, thr_sA);
        copy(tiled_copy_b, thr_gB, thr_sB);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // Load from SMEM to registers
        copy(thr_A, reg_A);
        copy(thr_B, reg_B);

        // MMA
        gemm(tiled_mma, reg_A, reg_B, accum);
    }

    // Store results
    copy(thr_C, accum);
}
```

### 15.13.3 Layout Manipulation Example

```cpp
#include "cute/layout.hpp"
#include "cute/print.hpp"  // For print_layout (host-side debugging)

void layout_examples() {
    // Create a 128x64 row-major layout
    auto layout = make_layout(make_shape(128, 64), make_stride(64, 1));

    // Print the layout (host-side only)
    // print_layout(layout);

    // Slice: get row 5
    auto row = layout(5, _);
    // row is a 1D layout with shape (64) and stride (1)

    // Slice: get column 3
    auto col = layout(_, 3);
    // col is a 1D layout with shape (128) and stride (64)

    // Compose with a tile layout
    auto tile_shape = make_shape(16, 16);
    auto tiled = zipped_divide(layout, tile_shape);
    // tiled has shape ((8, 4), (16, 16))
    // Outer shape: (8, 4) = number of 16x16 tiles
    // Inner shape: (16, 16) = tile size

    // Coalesce: merge unit-stride dimensions
    auto flat = coalesce(layout);
    // flat has shape (128*64,) = (8192,) with stride (1,)
    // Because the layout is fully dense and contiguous

    // Create a swizzled layout for shared memory bank conflict avoidance
    // Swizzle: XOR-based index transformation
    // auto swizzled = composition(swizzle<3, 3, -1>{}, layout);
    // The swizzle rearranges elements to avoid bank conflicts
}
```

---

## 15.14 Key Header Files Reference

| Header | Purpose |
|---|---|
| `cute/cute.hpp` | Master include for all CuTe functionality |
| `cute/layout.hpp` | Layout, Shape, Stride definitions |
| `cute/tensor.hpp` | Tensor abstraction (pointer + layout) |
| `cute/algorithm/copy.hpp` | Generic copy algorithm |
| `cute/algorithm/gemm.hpp` | Generic GEMM algorithm |
| `cute/algorithm/fill.hpp` | Fill algorithm |
| `cute/algorithm/prefetch.hpp` | Prefetch algorithm |
| `cute/atom/copy_atom.hpp` | Copy atom and TiledCopy |
| `cute/atom/mma_atom.hpp` | MMA atom and TiledMMA |
| `cute/atom/copy_traits.hpp` | Copy atom trait definitions |
| `cute/atom/mma_traits.hpp` | MMA atom trait definitions |
| `cute/arch/copy_sm90_tma.hpp` | SM90 TMA copy atoms |
| `cute/arch/mma_sm90.hpp` | SM90 WGMMA atoms |
| `cute/arch/mma_sm80.hpp` | SM80 mma.sync atoms |
| `cute/container/array.hpp` | Static array container |
| `cute/numeric/integer.hpp` | Compile-time integer (Int<N>) |
| `cute/util/debug.hpp` | Debug and print utilities |
| `cute/swizzle.hpp` | Swizzle layout utilities |

---

## 15.15 Summary

CuTe is the foundational abstraction layer of CUTLASS 3.x, providing:

1. **Layout**: A formal algebra for mapping multi-dimensional indices to memory offsets, supporting both static (compile-time) and dynamic (runtime) values.
2. **Tensor**: A pointer backed by a Layout, providing type-safe access to data in any memory space (global, shared, register).
3. **Atom**: Hardware-accelerated primitive operations (copy atoms and MMA atoms) that directly map to GPU instructions.
4. **TiledMMA / TiledCopy**: Tiled versions of atoms that distribute work across threads and warp groups.
5. **Composable hierarchy**: Atom > Tiled Op > Collective > Kernel > Device, where each level is independently composable.
6. **Rank-agnostic design**: All operations work uniformly across tensors of any dimensionality.
7. **Compile-time optimization**: Extensive use of C++ template metaprogramming eliminates abstraction overhead.
8. **CuTe DSL**: Python interface for rapid prototyping of GPU kernels using the same abstractions.

CuTe replaces the hundreds of named iterator types from CUTLASS 2.x with a small set of composable primitives, making GPU kernel code more maintainable, debuggable, and portable across architectures.
