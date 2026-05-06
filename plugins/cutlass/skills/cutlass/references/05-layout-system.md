# CUTLASS: Layout System

## Overview

The layout system is one of the most critical aspects of CUTLASS. A layout defines the mapping from a logical index space (e.g., matrix coordinates (i, j)) to physical memory offsets. Understanding how CUTLASS represents and uses layouts is essential for achieving optimal performance and correctness.

CUTLASS has two layout systems:
1. **CUTLASS 2.x layout tags** -- Tag types that describe matrix/tensor memory ordering (e.g., `RowMajor`, `ColumnMajor`)
2. **CuTe layout algebra** (CUTLASS 3.x) -- A powerful, composable layout algebra based on hierarchical shapes and strides

Both systems coexist in the codebase. CUTLASS 2.x code uses tag-based layouts exclusively, while CUTLASS 3.x uses CuTe layouts internally but still accepts 2.x layout tags at the device API level for convenience.

---

## Layout Concept

### What is a Layout?

A layout is a function that maps logical coordinates to linear memory offsets:

```
offset = layout(i, j, k, ...)
```

For a 2D matrix, the layout determines how rows and columns are arranged in memory:

- **Row-major**: Elements in the same row are contiguous; stride is `(N, 1)`
- **Column-major**: Elements in the same column are contiguous; stride is `(1, M)`

In CUTLASS 2.x, layouts are represented by tag types:
```cpp
cutlass::layout::RowMajor       // Row-major storage
cutlass::layout::ColumnMajor    // Column-major storage
```

In CuTe (CUTLASS 3.x), layouts are represented as composable shape-stride pairs:
```cpp
auto layout = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));  // Row-major
auto layout = make_layout(make_shape(M, N), make_stride(Int<1>{}, M));  // Column-major
```

### Why Layouts Matter

The choice of layout affects:
1. **Tensor Core compatibility** -- Certain layouts are required for specific Tensor Core instructions
2. **Memory access efficiency** -- Coalesced access patterns depend on layout alignment with GPU memory
3. **Shared memory bank conflicts** -- Layout determines which shared memory banks are accessed simultaneously
4. **TMA (Tensor Memory Accelerator) efficiency** -- TMA descriptors encode tensor layouts directly
5. **Epilogue efficiency** -- Output writes must match the expected output layout

---

## Matrix Layouts

### RowMajor

Elements in the same row are stored contiguously in memory. This is the C/C++ default memory ordering.

```
Logical view (M=3, N=4):
  (0,0) (0,1) (0,2) (0,3)
  (1,0) (1,1) (1,2) (1,3)
  (2,0) (2,1) (2,2) (2,3)

Memory (row-major):
  offset: 0   1   2   3   4   5   6   7   8   9  10  11
  value:  A   B   C   D   E   F   G   H   I   J   K   L

Stride: (N, 1) -- advance by N elements to move to next row, 1 element for next column
```

```cpp
#include <cutlass/layout/matrix.h>
using LayoutA = cutlass::layout::RowMajor;

// Stride for RowMajor: leading dimension = N (number of columns)
int lda = N;  // For an M x N matrix
// offset(i, j) = i * lda + j
```

### ColumnMajor

Elements in the same column are stored contiguously in memory. This is the Fortran/MATLAB default.

```
Logical view (M=3, N=4):
  (0,0) (0,1) (0,2) (0,3)
  (1,0) (1,1) (1,2) (1,3)
  (2,0) (2,1) (2,2) (2,3)

Memory (column-major):
  offset: 0   1   2   3   4   5   6   7   8   9  10  11
  value:  A   E   I   B   F   J   C   G   K   D   H   L

Stride: (1, M) -- advance by 1 element to move to next row, M elements for next column
```

```cpp
using LayoutA = cutlass::layout::ColumnMajor;

// Stride for ColumnMajor: leading dimension = M (number of rows)
int lda = M;
// offset(i, j) = i + j * lda
```

### RowMajorInterleaved

An interleaved layout where elements are grouped into fixed-size interleaved blocks for improved Tensor Core efficiency. This layout stores `K` consecutive elements from the same row together before moving to the next group.

```cpp
#include <cutlass/layout/row_major_interleaved.h>

// Row-major interleaved with interleaved size of 32
using LayoutA = cutlass::layout::RowMajorInterleaved<32>;

// Memory pattern: groups of 32 elements from each row are interleaved
// offset(i, j) = (i / InterleavedK) * InterleavedK * N +
//                (j / InterleavedK) * InterleavedK +
//                (i % InterleavedK)
```

**Purpose:** Interleaved layouts improve data locality for Tensor Core operations by ensuring that the K elements consumed by a single MMA instruction come from contiguous memory locations. This reduces the number of memory transactions and improves cache utilization.

---

## Layout Tags and Stride Types

### Tag Types (CUTLASS 2.x)

CUTLASS 2.x uses tag types to specify layouts. Tags are empty structs that carry type-level layout information:

```cpp
namespace cutlass::layout {
struct RowMajor {
    using Index = int;
    using LongIndex = int64_t;
    using Stride = Coord<2, Index>;  // (row_stride, col_stride)

    static Stride stride(LongIndex lda) {
        return Stride({lda, 1});  // Row-major: (N, 1)
    }
};

struct ColumnMajor {
    using Index = int;
    using LongIndex = int64_t;
    using Stride = Coord<2, Index>;

    static Stride stride(LongIndex lda) {
        return Stride({1, lda});  // Column-major: (1, M)
    }
};
}
```

### TagToStrideA / TagToStrideB / TagToStrideC

CUTLASS provides utility mappings from layout tags to stride tuples used internally:

```cpp
// TagToStrideA maps layout tags to stride types for matrix A
template <typename LayoutTag>
struct TagToStrideA;

template <>
struct TagToStrideA<cutlass::layout::RowMajor> {
    using Stride = cutlass::gemm::GemmCoord::Index;  // single stride value = K
    // For A (M x K): RowMajor stride is K
};

template <>
struct TagToStrideA<cutlass::layout::ColumnMajor> {
    using Stride = cutlass::gemm::GemmCoord::Index;  // single stride value = M
    // For A (M x K): ColumnMajor stride is M
};

// Similarly for B and C
template <typename LayoutTag> struct TagToStrideB;
template <typename LayoutTag> struct TagToStrideC;
```

### CUTLASS 3.x Stride Encoding

In CUTLASS 3.x, strides are encoded as CuTe integer tuples. The `cutlass::make_cute_packed_stride` function converts layout tags to CuTe strides:

```cpp
#include <cutlass/util/packed_stride.hpp>

// Convert layout tag to CuTe stride for matrix A
// For RowMajor A (M x K): stride = (K, Int<1>{})
auto stride_a = cutlass::make_cute_packed_stride(
    cutlass::layout::RowMajor{}, make_shape(M, K)
);

// For ColumnMajor A (M x K): stride = (Int<1>{}, M)
auto stride_a_col = cutlass::make_cute_packed_stride(
    cutlass::layout::ColumnMajor{}, make_shape(M, K)
);
```

---

## PitchLinear Layout

The `PitchLinear` layout is a generalized 2D layout used internally by CUTLASS for shared memory and register file access patterns. It abstracts a 2D memory region as (contiguous, strided) dimensions.

```cpp
#include <cutlass/layout/pitch_linear.h>

// PitchLinear layout: (contiguous_dim, strided_dim)
// contiguous: elements that are adjacent in memory
// strided: elements separated by a fixed stride (pitch)
using PitchLinearShape = cutlass::layout::PitchLinearShape<128, 8>;
// 128 contiguous elements, 8 strided elements (stride = pitch)

// PitchLinear coordinate
using PitchLinearCoord = cutlass::layout::PitchLinearCoord;
PitchLinearCoord coord(32, 4);  // (contiguous=32, strided=4)
```

### PitchLinear to Matrix Layout Conversion

CUTLASS internally maps between matrix layouts (RowMajor/ColumnMajor) and PitchLinear layouts:

```
RowMajor:    PitchLinear(contiguous=N, strided=M, pitch=N)
  contiguous dimension = columns (N)
  strided dimension = rows (M)
  pitch = N (stride between rows)

ColumnMajor: PitchLinear(contiguous=M, strided=N, pitch=M)
  contiguous dimension = rows (M)
  strided dimension = columns (N)
  pitch = M (stride between columns)
```

---

## Tensor Layouts

CUTLASS defines layout types for multi-dimensional tensors used in convolution and other operations:

### TensorNHWC

Standard 4D tensor layout for image data in deep learning (Batch, Height, Width, Channels):

```cpp
#include <cutlass/layout/tensor.h>
using Layout = cutlass::layout::TensorNHWC;

// Shape: (N, H, W, C)
// Stride: (H*W*C, W*C, C, 1)
// offset(n, h, w, c) = n * (H*W*C) + h * (W*C) + w * C + c

// Example: N=1, H=224, W=224, C=3
// Total elements: 1 * 224 * 224 * 3 = 150528
// Stride: (150528, 672, 3, 1)
```

### TensorNCxHWx

Interleaved tensor layout used for implicit GEMM convolution on Tensor Cores. The channel dimension is split into groups of `x` elements:

```cpp
using Layout = cutlass::layout::TensorNCxHWx<32>;

// Shape: (N, C/x, H, W, x)
// where x is the interleave factor (typically 32 for INT8, 8 for FP16)
// The x elements from the channel dimension are packed contiguously
```

### TensorCxRSKx

Weight tensor layout for convolution with interleaved channel groups:

```cpp
using Layout = cutlass::layout::TensorCxRSKx<32>;

// Shape: (C/x, R, S, K, x)
// C: output channels, R: filter height, S: filter width, K: input channels
// x elements of output channels are packed contiguously
```

---

## TensorOp Multiplicand Layouts

Tensor Core instructions require input data in specific layouts. CUTLASS provides `TensorOpMultiplicand` layouts that describe how data must be arranged in shared memory for efficient Tensor Core access.

### Per-Architecture Multiplicand Layouts

```cpp
// SM70 (Volta) - WMMA-based
// ColumnMajorInterleaved<32> for both A and B
// 32-element interleaving for wmma::load_matrix_sync

// SM75 (Turing) - mma.sync based
// TensorOpMultiplicand for SM75
using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicand<
    32,   // ElementsPerAccess
    cutlass::layout::TensorOpMultiplicandCongruous<32, 4>
>;

// SM80 (Ampere) - mma.sync based
// TensorOpMultiplicand with various crosswise/congruous configurations
using LayoutA_SM80 = cutlass::layout::TensorOpMultiplicandCongruous<
    64,   // Shape::kContiguous
    4     // Shape::kStrided / ElementsPerAccess
>;

// Crosswise layout for K-dimension access
using LayoutA_Crosswise = cutlass::layout::TensorOpMultiplicandCrosswise<
    32,   // Shape (K dimension tile)
    4     // ElementsPerAccess
>;
```

### Congruous vs Crosswise

These terms describe how the K-dimension (reduction dimension) is laid out in shared memory:

- **Congruous**: The K dimension elements are contiguous in memory, and the M/N dimension elements are strided. This is efficient when the K tile is small.

- **Crosswise**: The M/N dimension elements are contiguous in memory, and the K dimension elements are strided. This is efficient when the M/N tile is small and reduces bank conflicts for certain tile sizes.

```cpp
// Congruous layout: K is contiguous
// Memory: [k0_m0, k0_m1, k0_m2, ..., k1_m0, k1_m1, k1_m2, ...]
// Good for: small K tiles, large M/N tiles

// Crosswise layout: M/N is contiguous (K strides across)
// Memory: [m0_k0, m1_k0, m2_k0, ..., m0_k1, m1_k1, m2_k1, ...]
// Good for: large K tiles, small M/N tiles, fewer bank conflicts
```

---

## Interleaved Layouts for Tensor Core Efficiency

Interleaved layouts rearrange elements so that groups consumed by Tensor Core instructions are contiguous in memory. This reduces the number of shared memory bank conflicts and improves Tensor Core throughput.

### Interleaved Layout Types

```cpp
// Row-major interleaved: groups of InterleavedK elements from each row
using Layout = cutlass::layout::RowMajorInterleaved<32>;

// Column-major interleaved: groups of InterleavedK elements from each column
using Layout = cutlass::layout::ColumnMajorInterleaved<32>;
```

### Interleaved Layout Example (RowMajor, InterleavedK=4)

```
Original RowMajor (M=4, N=8):
Row 0: a00 a01 a02 a03 a04 a05 a06 a07
Row 1: a10 a11 a12 a13 a14 a15 a16 a17
Row 2: a20 a21 a22 a23 a24 a25 a26 a27
Row 3: a30 a31 a32 a33 a34 a35 a36 a37

Interleaved RowMajor (InterleavedK=4):
Block 0: a00 a01 a02 a03 a10 a11 a12 a13 a20 a21 a22 a23 a30 a31 a32 a33
Block 1: a04 a05 a06 a07 a14 a15 a16 a17 a24 a25 a26 a27 a34 a35 a36 a37
```

The interleave factor is chosen based on the Tensor Core instruction width and the data type size. Typical values:
- FP16: InterleavedK = 32 or 64
- INT8: InterleavedK = 32
- INT4: InterleavedK = 64

---

## Layout Algebra (CuTe)

CuTe provides a powerful layout algebra that is the foundation of CUTLASS 3.x. CuTe layouts are composable, inspectable, and architecture-portable.

### Core CuTe Layout Types

```cpp
#include <cute/layout.hpp>
using namespace cute;

// A CuTe Layout is a pair of a Shape and a Stride
// Layout<Shape, Stride>

// 1D layout: 128 contiguous elements
auto layout_1d = Layout<Shape<Int<128>>, Stride<Int<1>>>{};
// Equivalently:
auto layout_1d = make_layout(Int<128>{}, Int<1>{});

// 2D row-major layout: 128 rows x 64 columns
auto layout_rm = make_layout(
    make_shape(Int<128>{}, Int<64>{}),   // Shape: 128 x 64
    make_stride(Int<64>{}, Int<1>{})     // Stride: (64, 1) -- row-major
);

// 2D column-major layout: 128 rows x 64 columns
auto layout_cm = make_layout(
    make_shape(Int<128>{}, Int<64>{}),   // Shape: 128 x 64
    make_stride(Int<1>{}, Int<128>{})    // Stride: (1, 128) -- column-major
);
```

### Shape and Stride Hierarchy

CuTe shapes and strides can be nested hierarchically, enabling complex tensor layouts:

```cpp
// Hierarchical shape: 2 clusters of (128 elements each in 4 groups of 32)
auto shape = make_shape(Int<2>{}, make_shape(Int<4>{}, Int<32>{}));
// Shape: (2, (4, 32))

// Corresponding hierarchical stride
auto stride = make_stride(Int<128>{}, make_stride(Int<32>{}, Int<1>{}));
// Stride: (128, (32, 1))

auto layout = make_layout(shape, stride);
```

### Layout Operations

```cpp
// Size of a layout (total number of elements)
auto sz = size(layout);  // Product of all shape elements

// Access element at coordinate
auto offset = layout(0, 0);          // 2D coordinate
auto offset2 = layout(make_coord(3, 7));  // Coordinate object

// Slice a layout
auto sub_layout = slice(make_coord(_, 3), layout);  // Take row 3

// Compose layouts
auto composed = composition(layout_a, layout_b);

// Coalesce layout (merge contiguous dimensions)
auto coalesced = coalesce(layout);

// Flatten to 1D
auto flat = flatten(layout);

// Filter out size-1 dimensions
auto filtered = filter(layout);

// Compute the shape that a tensor of this layout would have
auto footprint_shape = shape(layout);

// Compute the minimum memory needed
auto footprint_size = cosize(layout);  // Maximum offset + 1
```

### Layout Composition

Composition allows combining two layouts to create a new one:

```cpp
// Outer layout: maps logical indices to intermediate indices
auto outer = make_layout(make_shape(Int<128>{}), make_stride(Int<1>{}));

// Inner layout: maps intermediate indices to physical offsets
auto inner = make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<1>{}, Int<32>{}));

// Composed: maps logical indices directly to physical offsets
auto composed = composition(inner, outer);
```

### Layout Partitioning for Thread-Level Access

CuTe layouts are used to partition data across threads in a warp or threadblock:

```cpp
// Create a tiled layout that maps threads to data elements
auto tiled_layout = make_layout(
    make_shape(Int<128>{}),           // 128 elements
    make_stride(Int<1>{})             // Contiguous
);

// Partition for a warp of 32 threads
auto thr_layout = zipped_divide(tiled_layout, make_shape(Int<32>{}));
// Result: each thread owns 4 contiguous elements

// Create tensor with this layout
auto tensor = make_tensor(ptr, tiled_layout);

// Partition tensor for a specific thread
auto thr_tensor = local_partition(tensor, thr_layout, thread_id);
```

---

## K-Major vs MN-Major Layouts

The layout of the K dimension (reduction dimension) relative to the M/N dimensions has a significant impact on Tensor Core performance, especially for TMA-based kernels on SM90+.

### K-Major Layout

A K-major layout has the K dimension as the contiguous (stride-1) dimension:

```
For A (M x K):
  K-major RowMajor means: stride = (K, 1) -- K is innermost
  This is the same as standard RowMajor

For B (K x N):
  K-major ColumnMajor means: stride = (1, K) -- K is innermost
  This is the same as standard ColumnMajor
```

**Why K-major matters:** TMA (Tensor Memory Accelerator) on Hopper/Blackwell works best with K-major layouts because:
1. TMA can load tiles along the K dimension efficiently
2. The innermost K dimension enables contiguous memory transfers
3. Warp-specialized kernels prefetch K-major tiles with minimal overhead

### MN-Major Layout

An MN-major layout has the M or N dimension as the contiguous dimension:

```
For A (M x K):
  M-major means: stride = (1, M) -- M is innermost
  This is ColumnMajor for A

For B (K x N):
  N-major means: stride = (N, 1) -- N is innermost
  This is RowMajor for B
```

### Layout Selection for GEMM

The optimal layout depends on the kernel schedule and architecture:

```cpp
// For SM90 TMA warp-specialized kernels:
// A should be K-major: RowMajor A (stride = (K, 1)) or
//                      ColumnMajor transpose A (stride = (1, M))
// B should be K-major: ColumnMajor B (stride = (1, K)) or
//                      RowMajor transpose B (stride = (N, 1))

// Configuration 1: A is RowMajor (K-major), B is ColumnMajor (K-major)
using LayoutA = cutlass::layout::RowMajor;     // K-major for A
using LayoutB = cutlass::layout::ColumnMajor;   // K-major for B

// Configuration 2: A is ColumnMajor (M-major), B is RowMajor (N-major)
// Less optimal for TMA, but may be required by data format
using LayoutA = cutlass::layout::ColumnMajor;   // M-major for A
using LayoutB = cutlass::layout::RowMajor;       // N-major for B
```

---

## Capacity Calculation for Memory Allocation

When allocating memory for tensors, the size must account for the layout's footprint:

### CUTLASS 2.x Capacity

```cpp
// For a matrix of shape M x N:
// RowMajor capacity: M * N
// ColumnMajor capacity: M * N
// The capacity is simply the product of dimensions for standard layouts

size_t capacity = M * N;

// For interleaved layouts:
// RowMajorInterleaved<InterleavedK> capacity: M * N
// (interleaving doesn't change total element count, just ordering)
```

### CuTe Capacity

```cpp
// CuTe provides cosize() for computing the maximum offset + 1
auto layout = make_layout(make_shape(Int<128>{}, Int<64>{}), make_stride(Int<64>{}, Int<1>{}));
size_t capacity = cosize(layout);  // 128 * 64 = 8192

// For general layouts with non-unit strides, cosize accounts for stride:
auto strided_layout = make_layout(
    make_shape(Int<128>{}, Int<64>{}),
    make_stride(Int<128>{}, Int<1>{})
);
size_t cap = cosize(strided_layout);  // (128-1)*128 + (64-1)*1 + 1 = 16320
```

### Practical Allocation

```cpp
// Allocate memory for a GEMM operation
int M = 1024, N = 1024, K = 512;

// A: M x K, RowMajor -> capacity = M * K
size_t size_A = M * K * sizeof(cutlass::half_t);
cutlass::half_t* d_A;
cudaMalloc(&d_A, size_A);

// B: K x N, ColumnMajor -> capacity = K * N
size_t size_B = K * N * sizeof(cutlass::half_t);
cutlass::half_t* d_B;
cudaMalloc(&d_B, size_B);

// C: M x N, RowMajor -> capacity = M * N
size_t size_C = M * N * sizeof(cutlass::half_t);
cutlass::half_t* d_C;
cudaMalloc(&d_C, size_C);
```

---

## Layout Selection Guide

### By Operation

| Operation | Recommended A Layout | Recommended B Layout | Recommended C Layout | Rationale |
|---|---|---|---|---|
| GEMM (SM90 TMA) | RowMajor (K-major) | ColumnMajor (K-major) | RowMajor | Optimal TMA prefetching |
| GEMM (SM80) | RowMajor or ColumnMajor | ColumnMajor or RowMajor | RowMajor | Depends on transpose configuration |
| GEMM (SM70/75) | ColumnMajor | RowMajor | RowMajor | Default for wmma and mma.sync |
| Convolution (fprop) | TensorNHWC | TensorCxRSKx | TensorNHWC | Standard NCHW format |
| Convolution (dgrad) | TensorNHWC | TensorCxRSKx | TensorNHWC | Data gradient layout |
| Convolution (wgrad) | TensorNHWC | TensorKxRSCx | TensorCxRSKx | Weight gradient layout |

### By Transpose Configuration

GEMM operation `D = alpha * A * B + beta * C` where A is M x K and B is K x N:

| Layout A | Layout B | TransA | TransB | Notes |
|---|---|---|---|---|
| RowMajor | RowMajor | N | Y | B is (N x K), A is (M x K). Both K-major. |
| RowMajor | ColumnMajor | N | N | Standard configuration, both K-major |
| ColumnMajor | RowMajor | Y | Y | Both transposed |
| ColumnMajor | ColumnMajor | Y | N | A transposed, B not |

### Layout Compatibility with Architecture

| Architecture | Layout Support | Notes |
|---|---|---|
| SM70 (Volta) | RowMajor, ColumnMajor | WMMA requires specific layout combinations |
| SM75 (Turing) | RowMajor, ColumnMajor, Interleaved | More flexible layout support |
| SM80 (Ampere) | RowMajor, ColumnMajor, Interleaved, TensorOp | Full layout support for mma.sync |
| SM90 (Hopper) | CuTe layouts, TMA-compatible | TMA encodes layouts in hardware descriptors |
| SM100 (Blackwell) | CuTe layouts, TMA-compatible, block-scaled | Block-scaled MMA requires specific CuTe layouts |

---

## Layout Swizzling for Shared Memory

Shared memory on NVIDIA GPUs has 32 memory banks. When multiple threads in a warp access the same bank simultaneously, a bank conflict occurs, serializing the access. CUTLASS applies swizzling transformations to shared memory layouts to avoid bank conflicts.

### Swizzle in CuTe

CuTe provides built-in swizzle operations:

```cpp
#include <cute/swizzle.hpp>

// 3-2-1 swizzle: XOR-based bank conflict elimination
auto swizzle = Swizzle<3, 2, 1>{};

// Apply swizzle to a layout
auto swizzled_layout = composition(swizzle, base_layout);

// The swizzle remaps addresses so that concurrent accesses
// from different threads hit different banks
```

### Swizzle Patterns

```
Without swizzle (bank conflicts possible):
Thread 0: addr 0 -> bank 0
Thread 1: addr 1 -> bank 1
Thread 2: addr 2 -> bank 2
...
Thread 8: addr 8 -> bank 8 (same row, bank conflict if threads 0 and 8 in same warp)

With 3-2-1 swizzle:
Thread 0: addr 0 -> bank 0
Thread 1: addr 1 -> bank 1
...
Thread 8: addr 8 ^ 0b100 -> bank 12  (different bank, no conflict)
```

### Using Swizzled SMEM Layouts in CUTLASS 3.x

```cpp
// Define a swizzled shared memory layout for A tile
auto smem_layout_a = composition(
    Swizzle<3, 4, 3>{},            // Swizzle transformation
    make_layout(
        make_shape(Int<128>{}, Int<8>{}),    // 128 x 8 tile
        make_stride(Int<8>{}, Int<1>{})      // Stride
    )
);

// Create a tensor in shared memory with this layout
extern __shared__ char smem_buf[];
auto sA = make_tensor(make_smem_ptr<ElementType>(smem_buf), smem_layout_a);
```

---

## Layout Debugging

CuTe provides powerful debugging utilities for inspecting layouts at compile time and runtime:

```cpp
#include <cute/util/debug.hpp>

// Print a layout's shape, stride, and size
CUTE_STATIC_ASSERT_V(size(layout) == Int<128>{});
print_layout(layout);  // Prints the layout to stdout

// Print a tensor's layout and data
print_tensor(tensor);  // Prints shape, stride, and element values

// Check if two layouts are congruent
CUTE_STATIC_ASSERT_V(congruent(layout_a, layout_b));

// Assert layout compatibility
CUTE_STATIC_ASSERT_V(size<0>(layout) == Int<128>{});
CUTE_STATIC_ASSERT_V(size<1>(layout) == Int<64>{});
```

### Common Layout Errors

1. **Stride mismatch**: The stride values do not produce correct offsets for the expected shape
2. **Alignment violation**: The layout requires alignment that the data does not satisfy
3. **Transpose confusion**: Using RowMajor when ColumnMajor is expected (or vice versa)
4. **Leading dimension error**: Setting the leading dimension incorrectly (e.g., lda = M for RowMajor A instead of lda = K)

---

## Summary

The CUTLASS layout system spans two paradigms:

1. **CUTLASS 2.x layout tags** (`RowMajor`, `ColumnMajor`, `RowMajorInterleaved`, etc.) provide simple, tag-based layout specification for the device API. These tags encode the memory ordering of matrix elements and determine stride computation.

2. **CuTe layout algebra** (CUTLASS 3.x) provides a composable, hierarchical layout system based on shape-stride pairs. CuTe layouts can be composed, sliced, partitioned, and swizzled to create optimal memory access patterns for any architecture.

Key layout concepts to remember:
- K-major layouts (where the K dimension is contiguous) are preferred for TMA-based kernels on SM90+
- Interleaved layouts improve Tensor Core efficiency by grouping elements consumed by MMA instructions
- Shared memory layouts must be swizzled to avoid bank conflicts
- Layout selection must match the expected transpose configuration of the GEMM operation
- The leading dimension must correctly reflect the layout (RowMajor: lda = K for A; ColumnMajor: lda = M for A)
