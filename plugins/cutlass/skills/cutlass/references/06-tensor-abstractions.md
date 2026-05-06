# Tensor Abstractions in CUTLASS

CUTLASS provides a layered system of tensor abstractions that serve as the foundation for all data movement and computation within the library. These abstractions decouple raw pointer arithmetic from logical tensor indexing, enabling portable, efficient, and type-safe access to multi-dimensional data across all levels of the GPU memory hierarchy.

## Overview

The core tensor abstractions in CUTLASS are:

- **Coord**: A compile-time dimensional coordinate for indexing into tensors.
- **TensorRef**: A lightweight, non-owning reference to tensor data composed of a base pointer and a layout object.
- **TensorView**: Extends TensorRef with an extent (bounds) to support bounds-checked access.
- **Layout**: A policy object that maps logical coordinates to linear offsets (e.g., pitch-linear, column-major, row-major, and more).

These abstractions are templated on element type, rank (number of dimensions), and layout policy, allowing the compiler to fully inline and optimize all index calculations.

---

## Coordinate System: Coord<N, Index>

### Template Signature

```cpp
template <int N, typename Index = int>
struct Coord;
```

`Coord<N, Index>` represents a point in an N-dimensional index space. It is used pervasively throughout CUTLASS to express tile origins, tensor extents, iteration ranges, and stride vectors.

### Key Properties

- **N** (template parameter): The rank / number of dimensions. Common values are 2, 3, 4.
- **Index** (template parameter): The integer type used for each component. Defaults to `int`, but `int64_t` or `long long` may be used for large tensors.

### Construction

```cpp
// Construct a 2D coordinate
cutlass::Coord<2> coord_2d = cutlass::make_Coord(32, 64);

// Construct a 3D coordinate
cutlass::Coord<3> coord_3d = cutlass::make_Coord(128, 256, 512);

// Construct with explicit index type
cutlass::Coord<2, int64_t> large_coord = cutlass::make_Coord<int64_t>(1000000, 2000000);

// Access individual dimensions
int m = coord_2d[0]; // 32
int n = coord_2d[1]; // 64
```

### Factory Functions

```cpp
// make_Coord: creates a Coord from individual values
auto c2 = cutlass::make_Coord(10, 20);
auto c3 = cutlass::make_Coord(10, 20, 30);
auto c4 = cutlass::make_Coord(10, 20, 30, 40);
```

### Arithmetic Operations

`Coord` supports element-wise arithmetic and comparison operations:

```cpp
cutlass::Coord<2> a = cutlass::make_Coord(10, 20);
cutlass::Coord<2> b = cutlass::make_Coord(5, 10);

// Addition
auto sum = a + b;  // (15, 30)

// Subtraction
auto diff = a - b; // (5, 10)

// Scalar multiplication
auto scaled = a * 2; // (20, 40)

// Element-wise min/max
auto mn = cutlass::min(a, b);  // (5, 10)
auto mx = cutlass::max(a, b);  // (10, 20)

// Comparison
bool eq = (a == a);   // true
bool neq = (a != b);  // true
```

### Coordinate Clamping

```cpp
cutlass::Coord<2> coord = cutlass::make_Coord(100, 200);
cutlass::Coord<2> upper = cutlass::make_Coord(64, 128);

// Clamp coord to be at most upper
auto clamped = cutlass::min(coord, upper); // (64, 128)
```

---

## TensorRef: Lightweight Tensor Reference

`TensorRef` is the most fundamental tensor abstraction in CUTLASS. It combines a raw data pointer with a layout object, providing a typed and layout-aware interface for accessing tensor elements without owning or managing the underlying memory.

### Template Signature

```cpp
template <
  typename Element_,     // Data type of tensor elements (e.g., half_t, float, int8_t)
  typename Layout_       // Layout policy (e.g., layout::ColumnMajor, layout::RowMajor)
>
class TensorRef;
```

The rank of the tensor is inferred from the `Layout_` type. For example, `layout::RowMajor` implies a 2D tensor (rank = 2), while `layout::ColumnMajorInterleaved<4>` also implies rank 2 but with a different stride mapping.

### Core Members

A `TensorRef` holds exactly two members:
1. **A pointer to the data**: `Element_* ptr_`
2. **A layout object**: `Layout_ layout_`

The total size of a `TensorRef` is typically just `sizeof(Element_*) + sizeof(Layout_)`, making it cheap to pass by value into kernels and device functions.

### Construction Patterns

```cpp
using Layout = cutlass::layout::RowMajor;
using Element = float;
using TensorRef = cutlass::TensorRef<Element, Layout>;

// Method 1: Construct from pointer and layout object
Element* ptr = ...;  // device or host pointer
Layout layout = Layout::packed({M, N});  // stride computed for packed layout
TensorRef ref(ptr, layout);

// Method 2: Construct from pointer and stride
typename Layout::Stride stride;
stride[0] = N;   // stride along rows for row-major: N elements between rows
Layout layout_with_stride(stride);
TensorRef ref2(ptr, layout_with_stride);

// Method 3: Construct from pointer and individual stride values
cutlass::Coord<2> extent = cutlass::make_Coord(M, N);
Layout packed = Layout::packed(extent);
TensorRef ref3(ptr, packed);
```

### Access Methods

```cpp
// data(): returns the raw base pointer
Element* base_ptr = ref.data();

// at(Coord): returns a reference to the element at the given coordinate
// This performs the layout mapping internally: offset = layout(coord)
cutlass::Coord<2> coord = cutlass::make_Coord(i, j);
Element& elem = ref.at(coord);

// operator()(int, int, ...): variadic access for convenience
Element& elem2 = ref(i, j);  // equivalent to at(make_Coord(i, j))

// operator[](Index): direct linear offset access (bypasses layout)
Element& elem3 = ref[linear_offset];
```

### Layout Mapping

The key insight of TensorRef is that coordinate-to-offset mapping is delegated entirely to the layout:

```cpp
// Internally, operator() works like:
Element& operator()(int i, int j) const {
    Coord<2> coord = make_Coord(i, j);
    typename Layout::Index offset = layout_(coord);
    return ptr_[offset];
}
```

For `layout::RowMajor`:
- offset = `i * stride[0] + j`

For `layout::ColumnMajor`:
- offset = `i + j * stride[0]`

For `layout::PitchLinear`:
- offset = `coord.contiguous() + coord.strided() * stride[0]`

### Const TensorRef

```cpp
// Creating a const TensorRef from a non-const one
cutlass::TensorRef<float, Layout> ref(ptr, layout);

// Implicit conversion to const
cutlass::TensorRef<float const, Layout> const_ref = ref;

// Explicit access to const version
auto cref = ref.const_tensor_ref();
// cref has type TensorRef<float const, Layout>
```

### TensorRef for Different Layouts

```cpp
// Row-major 2D tensor
using RM_Ref = cutlass::TensorRef<float, cutlass::layout::RowMajor>;
RM_Ref rm_ref(ptr_rm, cutlass::layout::RowMajor::packed({M, N}));

// Column-major 2D tensor
using CM_Ref = cutlass::TensorRef<float, cutlass::layout::ColumnMajor>;
CM_Ref cm_ref(ptr_cm, cutlass::layout::ColumnMajor::packed({M, N}));

// Row-major interleaved layout (e.g., for mixed-precision interleaving)
using InterleavedRef = cutlass::TensorRef<
    float,
    cutlass::layout::RowMajorInterleaved<4>
>;

// Pitch-linear layout (used internally for shared memory tiles)
using PL_Ref = cutlass::TensorRef<
    float,
    cutlass::layout::PitchLinear
>;
```

---

## TensorView: Bounded Tensor Reference

`TensorView` extends `TensorRef` by adding an extent (the logical size of the tensor). This enables bounds-checked access and provides methods for querying tensor shape. Like `TensorRef`, it is non-owning and lightweight.

### Template Signature

```cpp
template <
  typename Element_,   // Data type of tensor elements
  typename Layout_     // Layout policy
>
class TensorView;
```

### Core Members

A `TensorView` contains:
1. **A TensorRef**: provides pointer + layout
2. **An extent**: `Coord<Rank>` describing the logical size of each dimension

### Construction

```cpp
using Layout = cutlass::layout::RowMajor;
using Element = cutlass::half_t;
using TensorView = cutlass::TensorView<Element, Layout>;

// Construct from pointer and extent (packed stride assumed)
Element* ptr = ...;
cutlass::Coord<2> extent = cutlass::make_Coord(M, N);
Layout layout = Layout::packed(extent);
TensorView view(ptr, layout, extent);

// Construct from pointer, stride, and extent
typename Layout::Stride stride;
stride[0] = N + padding;
Layout layout_with_stride(stride);
TensorView view2(ptr, layout_with_stride, extent);

// Construct from a TensorRef + extent
cutlass::TensorRef<Element, Layout> ref(ptr, layout);
TensorView view3(ref, extent);
```

### Key Methods

```cpp
// extent(): returns the logical size of the tensor
cutlass::Coord<2> ext = view.extent();
int rows = ext[0];  // M
int cols = ext[1];  // N

// contains(Coord): checks if a coordinate is within bounds
bool in_bounds = view.contains(make_Coord(i, j));
// Returns true if 0 <= i < M and 0 <= j < N

// reference(): returns the underlying TensorRef
cutlass::TensorRef<Element, Layout> ref = view.reference();

// data(): returns the raw base pointer
Element* ptr = view.data();

// Access elements (same interface as TensorRef)
Element& elem = view(i, j);
Element& elem_at = view.at(make_Coord(i, j));

// const_tensor_view(): convert to const view
auto const_view = view.const_tensor_view();
// Type: TensorView<Element const, Layout>
```

### Bounds-Checked Access Pattern

```cpp
template <typename View>
__device__ void safe_access(View& view, int i, int j) {
    if (view.contains(cutlass::make_Coord(i, j))) {
        // Safe: coordinate is within bounds
        auto& elem = view(i, j);
        // ... process elem
    }
}
```

### TensorView from Raw Pointers and Dimensions

```cpp
// Common pattern: wrapping a device allocation
float* device_ptr;
int M = 128, N = 256;
cudaMalloc(&device_ptr, M * N * sizeof(float));

cutlass::Coord<2> extent = cutlass::make_Coord(M, N);
auto layout = cutlass::layout::RowMajor::packed(extent);
cutlass::TensorView<float, cutlass::layout::RowMajor> view(
    device_ptr, layout, extent
);

// Now view can be used for bounds-checked access
float val = view(10, 20);  // access element at row 10, col 20
```

---

## Capacity Calculation for Memory Allocation

When allocating memory for tensors, it is important to compute the correct capacity (total number of elements) rather than just the product of dimensions, because layouts with padding or non-trivial strides may require more memory.

```cpp
// Using layout's capacity method
cutlass::Coord<2> extent = cutlass::make_Coord(M, N);
auto layout = cutlass::layout::RowMajor::packed(extent);

// capacity() returns the minimum number of elements needed
typename Layout::Index capacity = layout.capacity(extent);

// Allocate
float* ptr;
cudaMalloc(&ptr, capacity * sizeof(float));
```

### Packed vs Padded Layouts

```cpp
// Packed layout: stride is computed to be minimal
auto packed = cutlass::layout::RowMajor::packed({128, 256});
// stride[0] = 256, capacity = 128 * 256 = 32768

// Padded layout: stride may exceed dimension for alignment
typename cutlass::layout::RowMajor::Stride stride;
stride[0] = 256 + 16;  // 16 elements of padding per row
auto padded_layout(stride);
int64_t padded_capacity = padded_layout.capacity({128, 256});
// = 128 * (256 + 16) = 34816

// Always use capacity() for allocation, not M*N
size_t bytes = padded_capacity * sizeof(float);
cudaMalloc(&ptr, bytes);
```

---

## Helper Functions

### make_TensorRef

```cpp
// Create a TensorRef from a raw pointer and layout
template <typename Element, typename Layout>
cutlass::TensorRef<Element, Layout>
make_TensorRef(Element* ptr, Layout const& layout) {
    return cutlass::TensorRef<Element, Layout>(ptr, layout);
}

// Usage
float* ptr = ...;
auto layout = cutlass::layout::RowMajor::packed({M, N});
auto ref = cutlass::make_TensorRef(ptr, layout);
```

### make_TensorView

```cpp
// Create a TensorView from a raw pointer, layout, and extent
template <typename Element, typename Layout>
cutlass::TensorView<Element, Layout>
make_TensorView(
    Element* ptr,
    Layout const& layout,
    cutlass::Coord<Layout::kRank> extent
) {
    return cutlass::TensorView<Element, Layout>(ptr, layout, extent);
}

// Usage
float* ptr = ...;
auto layout = cutlass::layout::RowMajor::packed({M, N});
auto extent = cutlass::make_Coord(M, N);
auto view = cutlass::make_TensorView(ptr, layout, extent);
```

### make_ConstTensorRef / make_ConstTensorView

```cpp
// Convert to const versions
template <typename Element, typename Layout>
cutlass::TensorRef<Element const, Layout>
make_ConstTensorRef(cutlass::TensorRef<Element, Layout> ref) {
    return ref.const_tensor_ref();
}

template <typename Element, typename Layout>
cutlass::TensorView<Element const, Layout>
make_ConstTensorView(cutlass::TensorView<Element, Layout> view) {
    return view.const_tensor_view();
}
```

---

## Alignment and Stride Requirements

CUTLASS enforces alignment requirements for performance. The leading dimension (stride) must typically be a multiple of the memory access width.

### Alignment Checking

```cpp
// Check if a pointer is aligned to a given boundary
template <typename Element>
bool is_aligned(Element* ptr, int alignment_in_elements) {
    return (reinterpret_cast<uintptr_t>(ptr) % (alignment_in_elements * sizeof(Element))) == 0;
}

// Check if stride satisfies alignment requirements
template <typename Layout>
bool is_stride_aligned(Layout const& layout, int alignment) {
    return (layout.stride(0) % alignment) == 0;
}
```

### Alignment in CUTLASS GEMM

For GEMM operations, alignment requirements come from:
- **Operand A**: The K-dimension stride must be aligned to the access width of the MMA instruction.
- **Operand B**: Similarly, the K-dimension stride must be aligned.
- **Output C**: The N-dimension stride must be aligned for efficient epilogue stores.

```cpp
// Typical alignment requirements
// FP16 with 128-bit memory access: alignment = 8 elements
// FP32 with 128-bit memory access: alignment = 4 elements
// INT8 with 128-bit memory access: alignment = 16 elements

int alignment_A = 128 / (sizeof(ElementA) * 8);  // elements per 128-bit access
int alignment_B = 128 / (sizeof(ElementB) * 8);
int alignment_C = 128 / (sizeof(ElementC) * 8);

// CUTLASS GEMM will check alignment at runtime:
// if (problem_size.n() % alignment_B != 0) { /* error */ }
```

### Layout Stride and Contiguity

```cpp
using RowMajor = cutlass::layout::RowMajor;

// Packed (contiguous) layout
auto packed = RowMajor::packed({128, 256});
// packed.stride() = (256,)  -- stride between rows = N columns

// The layout stride tells CUTLASS how elements are arranged in memory
// For RowMajor, stride(0) = number of columns (including padding)

// When the stride equals the dimension, the tensor is "packed"
bool is_packed = (packed.stride(0) == 256);  // true
```

---

## Layout Types in Detail

### Standard Layouts

```cpp
// Row-major (C-style): rightmost index varies fastest
// offset(i, j) = i * N + j
namespace cutlass::layout {
struct RowMajor;       // stride(0) = N
struct ColumnMajor;    // stride(0) = M
}

// Column-major (Fortran-style): leftmost index varies fastest
// offset(i, j) = i + j * M
```

### PitchLinear Layout

```cpp
// Used internally for shared memory tiles and tensor core access patterns
// Separates dimensions into "contiguous" and "strided"
namespace cutlass::layout {
struct PitchLinear {
    struct Stride {
        int contiguous;  // stride in the contiguous dimension
        int strided;     // stride in the strided dimension
    };
};
}

// offset(c, s) = c + s * stride.strided
```

### Interleaved Layouts

```cpp
// Row-major with interleaved access pattern
// Used for certain memory access optimizations
namespace cutlass::layout {
template <int InterleavedK>
struct RowMajorInterleaved;
// InterleavedK elements from each row are stored contiguously
}

// Example: RowMajorInterleaved<32>
// Data layout: [row0_col0..col31, row1_col0..col31, row2_col0..col31, ...]
```

### TensorNHWC Layout

```cpp
// 4D tensor layout for convolution
namespace cutlass::layout {
struct TensorNHWC;
// Stride: (W*C, C, 1) for N, H, W, C dimensions
}

using NHWC_Ref = cutlass::TensorRef<float, cutlass::layout::TensorNHWC>;
```

---

## Interoperability with CuTe Tensors

CUTLASS 3.x introduces the CuTe library, which provides a more powerful tensor abstraction system. CUTLASS 2.x tensor types (TensorRef, TensorView) can coexist with CuTe tensors.

### CuTe Tensor Overview

```cpp
#include <cute/tensor.hpp>

// CuTe tensor: combines a pointer/engine with a layout (shape + stride)
// Template parameters: Engine, Layout
auto shape = cute::Shape<cute::Int<128>, cute::Int<256>>{};
auto stride = cute::Stride<cute::Int<1>, cute::Int<128>>{};
auto tensor = cute::make_tensor<float*>(ptr, cute::make_layout(shape, stride));
```

### Conversion Between CUTLASS 2.x and CuTe

```cpp
// CuTe provides compatibility with CUTLASS 2.x layout types
// The conversion is handled through layout adapters

// In practice, CUTLASS 3.x code uses CuTe tensors directly
// while CUTLASS 2.x code uses TensorRef/TensorView

// CUTLASS 3.x internal usage:
// auto gA = cute::make_tensor<ElementA>(ptr_A, layout_A);
// auto sA = cute::make_tensor<ElementA>(smem_ptr, layout_sA);
// auto tAgA = cute::partition_A(gA, tiled_mma);  // partition for thread
```

### When to Use Each

| Feature | TensorRef / TensorView (2.x) | CuTe Tensor (3.x) |
|---|---|---|
| Rank | Fixed at compile time | Compile-time or runtime |
| Layout flexibility | One layout per type | Compose layouts freely |
| Shape encoding | Runtime Coord | Compile-time shapes via cute::Shape |
| Bounds checking | TensorView only | Optional |
| Integration | CUTLASS 2.x GEMM API | CUTLASS 3.x GEMM API |
| Memory space | Implicit (host/device) | Explicit tags (gm, smem, rm) |

---

## Complete Usage Examples

### Example 1: Creating TensorRef for GEMM Input

```cpp
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/matrix.h"

// Allocate device memory for a MxK row-major matrix
int M = 128, K = 64;
cutlass::half_t* d_A;
cudaMalloc(&d_A, M * K * sizeof(cutlass::half_t));

// Create a TensorRef for operand A
using LayoutA = cutlass::layout::RowMajor;
using TensorRefA = cutlass::TensorRef<cutlass::half_t, LayoutA>;

LayoutA layout_A = LayoutA::packed({M, K});
TensorRefA ref_A(d_A, layout_A);

// Access elements in device code
__global__ void kernel(TensorRefA ref_A, int M, int K) {
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < M && j < K) {
        cutlass::half_t val = ref_A(i, j);
        // ... process val
    }
}
```

### Example 2: TensorView with Bounds Checking

```cpp
#include "cutlass/tensor_view.h"
#include "cutlass/layout/matrix.h"

void process_tensor() {
    int M = 64, N = 128;
    float* data;
    cudaMallocManaged(&data, M * N * sizeof(float));

    using Layout = cutlass::layout::ColumnMajor;
    using View = cutlass::TensorView<float, Layout>;

    Layout layout = Layout::packed({M, N});
    View view(data, layout, cutlass::make_Coord(M, N));

    // Bounds-checked access (host side, unified memory)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (view.contains(cutlass::make_Coord(i, j))) {
                view.at(cutlass::make_Coord(i, j)) = static_cast<float>(i * N + j);
            }
        }
    }

    // Get the extent
    auto ext = view.extent();
    printf("Tensor dimensions: %d x %d\n", ext[0], ext[1]);

    cudaFree(data);
}
```

### Example 3: TensorRef with Non-Standard Strides

```cpp
// A sub-matrix view (pointer into the middle of a larger allocation)
int big_M = 1024, big_N = 1024;
float* big_matrix;
cudaMalloc(&big_matrix, big_M * big_N * sizeof(float));

// We want to work with rows [100..227] x cols [200..455]
int start_row = 100, rows = 128;
int start_col = 200, cols = 256;

// Stride for the big matrix (row-major)
int big_stride = big_N;

// Pointer to the sub-matrix origin
float* sub_ptr = big_matrix + start_row * big_stride + start_col;

// Create a TensorRef for the sub-matrix with the big matrix's stride
using Layout = cutlass::layout::RowMajor;
cutlass::TensorRef<float, Layout> sub_ref(
    sub_ptr,
    Layout(typename Layout::Stride({big_stride}))
);

// Access sub_ref(i, j) for i in [0, rows), j in [0, cols)
// sub_ref(0, 0) == big_matrix[100 * 1024 + 200]
// sub_ref(1, 0) == big_matrix[101 * 1024 + 200]
```

### Example 4: Device Kernel Using TensorRef

```cpp
template <typename Element, typename Layout>
__global__ void tensor_fill_kernel(
    cutlass::TensorRef<Element, Layout> ref,
    cutlass::Coord<Layout::kRank> extent,
    Element value
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total = extent[0] * extent[1];

    for (int pos = idx; pos < total; pos += blockDim.x * gridDim.x) {
        int i = pos / extent[1];
        int j = pos % extent[1];
        ref(i, j) = value;
    }
}

// Launch
template <typename Element, typename Layout>
void tensor_fill(
    cutlass::TensorView<Element, Layout> view,
    Element value,
    cudaStream_t stream = 0
) {
    auto extent = view.extent();
    int total = extent[0] * extent[1];
    int block = 256;
    int grid = (total + block - 1) / block;

    tensor_fill_kernel<<<grid, block, 0, stream>>>(
        view.reference(), extent, value
    );
}
```

### Example 5: Multiple Layouts for Same Data

```cpp
// Sometimes the same data needs to be interpreted with different layouts
// For example, the shared memory tile in GEMM uses PitchLinear layout
// for efficient access, while the global memory uses RowMajor.

float* smem_ptr = shared_memory;  // pointer to shared memory

// Pitch-linear view for shared memory access (contiguous + strided)
using PL = cutlass::layout::PitchLinear;
cutlass::TensorRef<float, PL> pl_ref(smem_ptr, PL::packed({K, M}));

// Row-major view for the same data
using RM = cutlass::layout::RowMajor;
cutlass::TensorRef<float, RM> rm_ref(smem_ptr, RM::packed({M, K}));

// Both views access the same memory but with different index calculations
// pl_ref(c, s) -> offset = c + s * K
// rm_ref(i, j) -> offset = i * K + j
// When K is the same, these produce the same mapping for (c=i, s=j)
```

---

## Summary of Tensor Abstraction API

### TensorRef Quick Reference

| Method | Description |
|---|---|
| `data()` | Returns the raw base pointer |
| `data(Coord)` | Returns pointer to element at coordinate |
| `at(Coord)` | Returns reference to element at coordinate |
| `operator()(indices...)` | Variadic element access |
| `operator[](offset)` | Direct linear offset access |
| `layout()` | Returns a copy of the layout object |
| `stride(dim)` | Returns the stride for a given dimension |
| `const_tensor_ref()` | Returns a const version of this ref |

### TensorView Quick Reference

| Method | Description |
|---|---|
| All TensorRef methods | Inherited from TensorRef |
| `extent()` | Returns the logical extent of the tensor |
| `contains(Coord)` | Checks if coordinate is within bounds |
| `reference()` | Returns the underlying TensorRef |
| `const_tensor_view()` | Returns a const version of this view |

### Layout Quick Reference

| Layout | Stride | offset(coord) |
|---|---|---|
| `RowMajor` | `{N}` | `coord[0]*N + coord[1]` |
| `ColumnMajor` | `{M}` | `coord[0] + coord[1]*M` |
| `PitchLinear` | `{stride}` | `c + s*stride` |
| `RowMajorInterleaved<K>` | complex | interleaved access pattern |
| `TensorNHWC` | `{HWC, WC, C, 1}` | N-dimensional offset |

These tensor abstractions form the backbone of CUTLASS, enabling efficient and type-safe data access across all levels of the GPU computation hierarchy. Understanding them is essential for working with both CUTLASS 2.x and 3.x APIs.
