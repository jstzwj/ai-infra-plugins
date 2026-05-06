# CUTLASS Coordinate System - Chapter 36: Coord Template, Tensor Coordinates, and Helper Functions

This reference covers the CUTLASS coordinate system, including the `Coord` template, coordinate operations, helper functions for construction, and integration with tensor abstractions.

---

## 36.1 Overview

CUTLASS represents tensor coordinates using the `Coord<N, Index>` template, where `N` is the rank (number of dimensions) and `Index` is the integer type used for each dimension. Coordinates are the foundation for indexing into tensors, defining tile shapes, specifying strides, and describing problem dimensions.

The `Coord` template is defined in `include/cutlass/coord.h`.

---

## 36.2 Coord<N, Index> Template

### 36.2.1 Template Parameters

```cpp
template <int N, typename Index = int>
struct Coord;
```

| Parameter | Type | Description |
|---|---|---|
| `N` | `int` (compile-time) | The rank of the coordinate (number of dimensions). Valid values: 1-5 for explicit specializations, arbitrary via recursive definition. |
| `Index` | `int` (default), `int64_t`, `long` | The integer type used for each dimension's index value. |

### 36.2.2 Supported Ranks

CUTLASS provides explicit specializations for ranks 1 through 5, plus a generic template for arbitrary ranks:

```cpp
// Common coordinate types:
using Coord1D = cutlass::Coord<1, int>;       // 1D coordinate (e.g., linear index)
using Coord2D = cutlass::Coord<2, int>;       // 2D coordinate (e.g., matrix MxN)
using Coord3D = cutlass::Coord<3, int>;       // 3D coordinate (e.g., GEMM MxNxK)
using Coord4D = cutlass::Coord<4, int>;       // 4D coordinate (e.g., convolution NCHW)
using Coord5D = cutlass::Coord<5, int>;       // 5D coordinate (e.g., conv NCDHW)

// 64-bit index type for large tensors:
using Coord2D_64 = cutlass::Coord<2, int64_t>;
using Coord3D_64 = cutlass::Coord<3, int64_t>;
```

---

## 36.3 Construction

### 36.3.1 Default Construction

```cpp
// All dimensions initialized to 0
cutlass::Coord<3> coord;        // coord = {0, 0, 0}
cutlass::Coord<2, int64_t> c2;  // c2 = {0, 0}
```

### 36.3.2 Construction from Values

```cpp
// Direct initialization with values
cutlass::Coord<3> coord3d{128, 256, 64};  // {128, 256, 64}
cutlass::Coord<2> coord2d{32, 64};         // {32, 64}
cutlass::Coord<1> coord1d{1024};           // {1024}
cutlass::Coord<4> coord4d{1, 3, 224, 224}; // {1, 3, 224, 224}

// Using uniform initialization (braces)
cutlass::Coord<3> gemm_shape = {M, N, K};
cutlass::Coord<2> tile_shape = {128, 128};
```

### 36.3.3 Construction from Other Coords

```cpp
// Copy construction
cutlass::Coord<3> original{128, 256, 64};
cutlass::Coord<3> copy(original);  // copy = {128, 256, 64}

// Assignment
cutlass::Coord<3> assigned;
assigned = original;  // assigned = {128, 256, 64}

// Slicing to reduce rank (see Section 36.4.2)
cutlass::Coord<3> full{10, 20, 30};
cutlass::Coord<2> slice = full.slice();  // {10, 20} (drops last dimension)
```

### 36.3.4 Construction via make_Coord Helper

```cpp
// make_Coord provides a convenient factory function:
auto c1 = cutlass::make_Coord(128);                      // Coord<1>{128}
auto c2 = cutlass::make_Coord(128, 256);                 // Coord<2>{128, 256}
auto c3 = cutlass::make_Coord(128, 256, 64);             // Coord<3>{128, 256, 64}
auto c4 = cutlass::make_Coord(1, 3, 224, 224);           // Coord<4>{1, 3, 224, 224}
auto c5 = cutlass::make_Coord(1, 3, 32, 224, 224);       // Coord<5>{1, 3, 32, 224, 224}

// Explicit type specification
auto c3_64 = cutlass::make_Coord<int64_t>(128, 256, 64);  // Coord<3, int64_t>

// Use in GEMM problem definition
auto problem_size = cutlass::make_Coord(M, N, K);
```

### 36.3.5 Construction via make_Coord_with_padding

```cpp
// make_Coord_with_padding creates a higher-rank coordinate by padding
// remaining dimensions with a fill value (default 0).

// Create a 4D coordinate from 2D, padding extra dims with 1:
auto padded = cutlass::make_Coord_with_padding<4>(128, 256);
// Result: Coord<4>{128, 256, 0, 0}

// With explicit fill value:
auto padded2 = cutlass::make_Coord_with_padding<4>(128, 256, 1);
// Result: Coord<4>{128, 256, 1, 1}  (padded with 1)

// Useful for converting GEMM problem size to a uniform coordinate:
// GEMM problem: M=128, N=256, K=512
auto gemm_coord = cutlass::make_Coord_with_padding<4>(128, 256, 512);
// Coord<4>{128, 256, 512, 0}
```

---

## 36.4 Operations

### 36.4.1 Element Access

```cpp
// Operator[] provides indexed access to dimensions
cutlass::Coord<3> coord{128, 256, 64};

int m = coord[0];  // 128 (first dimension)
int n = coord[1];  // 256 (second dimension)
int k = coord[2];  // 64  (third dimension)

// Mutable access
coord[0] = 512;    // coord = {512, 256, 64}
coord[2] = 128;    // coord = {512, 256, 128}

// Named access helpers (for GemmCoord specialization):
cutlass::gemm::GemmCoord gemm_size{128, 256, 64};
int m = gemm_size.m();  // 128
int n = gemm_size.n();  // 256
int k = gemm_size.k();  // 64
```

### 36.4.2 Slicing

The `slice()` method reduces or increases the rank of a coordinate by removing or adding dimensions.

```cpp
// Reduce rank by removing trailing dimensions:
cutlass::Coord<4> c4{10, 20, 30, 40};

// Slice to lower rank (removes trailing dimensions)
cutlass::Coord<3> c3 = c4.slice<3>();   // {10, 20, 30}
cutlass::Coord<2> c2 = c4.slice<2>();   // {10, 20}
cutlass::Coord<1> c1 = c4.slice<1>();   // {10}

// Default slice removes the last dimension:
auto default_slice = c4.slice();  // Returns Coord<3>{10, 20, 30}

// Slice with starting offset:
// Takes dimensions [start, start + N_out)
cutlass::Coord<2> mid = c4.slice<2>(1);  // {20, 30} (dimensions 1 and 2)

// Increase rank by padding with zeros:
cutlass::Coord<2> small{10, 20};
cutlass::Coord<4> large = small.slice<4>();  // {10, 20, 0, 0}
```

### 36.4.3 Arithmetic Operations

All arithmetic operations are element-wise:

```cpp
// Addition
cutlass::Coord<3> a{10, 20, 30};
cutlass::Coord<3> b{1, 2, 3};
cutlass::Coord<3> sum = a + b;    // {11, 22, 33}

// Subtraction
cutlass::Coord<3> diff = a - b;   // {9, 18, 27}

// Multiplication (element-wise, both coords must have same rank)
cutlass::Coord<3> prod = a * b;   // {10, 40, 90}

// Division (element-wise, integer division)
cutlass::Coord<3> quot = a / b;   // {10, 10, 10}

// Scalar multiplication
cutlass::Coord<3> scaled = a * 2;  // {20, 40, 60}
cutlass::Coord<3> scaled2 = 3 * a; // {30, 60, 90}

// Scalar division
cutlass::Coord<3> halved = a / 2;  // {5, 10, 15}

// In-place operations
a += b;  // a = {11, 22, 33}
a -= b;  // a = {10, 20, 30}
a *= 2;  // a = {20, 40, 60}
```

### 36.4.4 Comparison Operations

```cpp
// Equality (all elements must match)
cutlass::Coord<3> a{10, 20, 30};
cutlass::Coord<3> b{10, 20, 30};
cutlass::Coord<3> c{10, 20, 31};

bool eq = (a == b);   // true
bool neq = (a != c);  // true

// Element-wise comparisons (return Coord<bool, N>)
auto lt_result = a < cutlass::Coord<3>{20, 30, 40};
// Returns Coord<3>{true, true, true}

auto le_result = a <= cutlass::Coord<3>{10, 20, 30};
// Returns Coord<3>{true, true, true}

auto gt_result = a > cutlass::Coord<3>{5, 10, 15};
// Returns Coord<3>{true, true, true}

auto ge_result = a >= cutlass::Coord<3>{10, 20, 30};
// Returns Coord<3>{true, true, true}

// Note: <, <=, >, >= compare element-wise and return Coord<bool, N>
// To check if ALL elements satisfy a condition, use:
bool all_less = (a < b).dot(true);  // true if all dimensions satisfy a[i] < b[i]
```

### 36.4.5 dot() - Dot Product

Computes the dot product of the coordinate elements.

```cpp
// Dot product between two coordinates
cutlass::Coord<3> a{2, 3, 4};
cutlass::Coord<3> b{5, 6, 7};

int dot = a.dot(b);  // 2*5 + 3*6 + 4*7 = 10 + 18 + 28 = 56

// Commonly used for computing linear offsets from coordinates and strides:
cutlass::Coord<3> coord{2, 3, 4};       // Position in tensor
cutlass::Coord<3> stride{64, 8, 1};     // Strides
int linear_offset = coord.dot(stride);   // 2*64 + 3*8 + 4*1 = 128 + 24 + 4 = 156

// This is the fundamental operation for layout mapping:
// offset = coord[0]*stride[0] + coord[1]*stride[1] + ... + coord[N-1]*stride[N-1]
```

### 36.4.6 sum() - Sum of Elements

```cpp
// Sum of all dimension values
cutlass::Coord<4> coord{10, 20, 30, 40};
int s = coord.sum();  // 10 + 20 + 30 + 40 = 100

// Useful for computing total rank or dimension count:
// Also used in convolution parameter calculations
```

### 36.4.7 prod() - Product of Elements

```cpp
// Product of all dimension values
cutlass::Coord<3> shape{128, 256, 64};
int64_t total = shape.prod();  // 128 * 256 * 64 = 2,097,152

// Commonly used for computing total element count:
cutlass::Coord<4> tensor_shape{1, 3, 224, 224};
int64_t num_elements = tensor_shape.prod();  // 150,528

// For GEMM: total FLOPS = 2 * problem_size.prod()
cutlass::Coord<3> gemm_size{1024, 1024, 1024};
int64_t flops = 2LL * gemm_size.prod();  // 2,147,483,648 = ~2.15 GFLOPS
```

### 36.4.8 clamp() - Clamp to Range

```cpp
// Clamp each dimension to a range [lower, upper]
cutlass::Coord<3> value{150, 300, 50};
cutlass::Coord<3> lower{0, 0, 0};
cutlass::Coord<3> upper{128, 256, 64};

cutlass::Coord<3> clamped = value.clamp(lower, upper);
// Result: {128, 256, 50}  (clamped to upper bounds)

// Useful for boundary checking in tiled operations:
cutlass::Coord<2> tile_coord{7, 3};  // Tile coordinate
cutlass::Coord<2> tile_size{128, 128};  // Tile dimensions
cutlass::Coord<2> problem_size{900, 384};  // Problem dimensions

// Compute actual extent of this tile:
cutlass::Coord<2> tile_begin = tile_coord * tile_size;  // {896, 384}
cutlass::Coord<2> tile_end = (tile_coord + cutlass::Coord<2>{1, 1}) * tile_size;  // {1024, 512}
cutlass::Coord<2> actual_end = tile_end.clamp(
    cutlass::Coord<2>{0, 0}, problem_size
);  // {900, 384}
```

---

## 36.5 Specialized Coordinate Types

### 36.5.1 GemmCoord

A specialized 3D coordinate for GEMM problem sizes with named accessors:

```cpp
namespace cutlass::gemm {

struct GemmCoord : public cutlass::Coord<3, int> {
    // Named accessors for GEMM dimensions
    int& m() { return (*this)[0]; }  // Number of rows of A and C
    int& n() { return (*this)[1]; }  // Number of columns of B and C
    int& k() { return (*this)[2]; }  // Inner dimension (columns of A, rows of B)

    int m() const { return (*this)[0]; }
    int n() const { return (*this)[1]; }
    int k() const { return (*this)[2]; }

    // Total multiply-add operations
    int64_t mno() const { return m() * n() * k(); }

    // Construction
    GemmCoord(int m_, int n_, int k_) : Coord<3>({m_, n_, k_}) {}
    GemmCoord() : Coord<3>({0, 0, 0}) {}
};

} // namespace cutlass::gemm

// Usage:
cutlass::gemm::GemmCoord problem_size{1024, 512, 256};
printf("M=%d, N=%d, K=%d\n", problem_size.m(), problem_size.n(), problem_size.k());
printf("Total MADs: %lld\n", (long long)problem_size.mno());
```

### 36.5.2 Tensor4DCoord

A specialized 4D coordinate for tensor dimensions (NCHW format):

```cpp
namespace cutlass {

struct Tensor4DCoord : public Coord<4, int> {
    int& n() { return (*this)[0]; }  // Batch dimension
    int& c() { return (*this)[1]; }  // Channels
    int& h() { return (*this)[2]; }  // Height
    int& w() { return (*this)[3]; }  // Width

    int n() const { return (*this)[0]; }
    int c() const { return (*this)[1]; }
    int h() const { return (*this)[2]; }
    int w() const { return (*this)[3]; }

    Tensor4DCoord(int n_, int c_, int h_, int w_)
        : Coord<4>({n_, c_, h_, w_}) {}
};

} // namespace cutlass

// Usage:
cutlass::Tensor4DCoord activation_shape{1, 256, 14, 14};
cutlass::Tensor4DCoord filter_shape{512, 256, 3, 3};
```

### 36.5.3 Tensor5DCoord

A 5D coordinate for 3D tensors (NCDHW format):

```cpp
namespace cutlass {

struct Tensor5DCoord : public Coord<5, int> {
    int& n() { return (*this)[0]; }  // Batch
    int& c() { return (*this)[1]; }  // Channels
    int& d() { return (*this)[2]; }  // Depth
    int& h() { return (*this)[3]; }  // Height
    int& w() { return (*this)[4]; }  // Width

    int n() const { return (*this)[0]; }
    int c() const { return (*this)[1]; }
    int d() const { return (*this)[2]; }
    int h() const { return (*this)[3]; }
    int w() const { return (*this)[4]; }
};

} // namespace cutlass
```

---

## 36.6 Integration with Tensor Abstractions

### 36.6.1 TensorRef Coordinate Indexing

`TensorRef` uses coordinates to access elements through a layout:

```cpp
#include "cutlass/tensor_ref.h"

// Create a TensorRef with a 2D layout
cutlass::TensorRef<float, cutlass::layout::RowMajor> ref(
    ptr, cutlass::layout::RowMajor::Stride(N)
);

// Access element at coordinate (i, j)
cutlass::Coord<2> coord{i, j};
float value = ref.at(coord);

// Using operator() with Coord:
ref.at(coord) = 3.14f;

// The layout maps Coord to linear offset:
// offset = coord[0] * stride + coord[1]
// For RowMajor: offset = i * N + j
// For ColumnMajor: offset = j * M + i
```

### 36.6.2 TensorView Coordinate Bounds

```cpp
#include "cutlass/tensor_view.h"

// Create a TensorView with extent
float *data;
cutlass::TensorView<float, cutlass::layout::RowMajor> view(
    data, cutlass::layout::RowMajor(M),
    cutlass::make_Coord(M, N)
);

// Get extent (size of each dimension)
cutlass::Coord<2> extent = view.extent();  // {M, N}

// Iterate over all coordinates
for (int i = 0; i < extent[0]; ++i) {
    for (int j = 0; j < extent[1]; ++j) {
        view.at(cutlass::make_Coord(i, j)) = float(i * N + j);
    }
}

// Bounds checking (debug mode)
cutlass::Coord<2> valid_coord{M - 1, N - 1};
cutlass::Coord<2> invalid_coord{M, N};
view.at(valid_coord);    // OK
// view.at(invalid_coord);  // Assert fails in debug mode
```

### 36.6.3 Layout Mapping

Layouts are functors that map coordinates to linear offsets using the dot product:

```cpp
// A layout maps Coord<N> to a linear offset
// The fundamental operation is:
// offset = coord.dot(stride)

// RowMajor layout for M x N matrix:
// stride = {N, 1}
// offset = row * N + col

// ColumnMajor layout for M x N matrix:
// stride = {1, M}
// offset = row + col * M

// RowMajor Interleaved layout (interleave = 4):
// stride = {N, 1} but with special tile access pattern

// PitchLinear layout (used in internal representations):
// stride = {contiguous_stride, stride}

// Using layout to compute offsets:
cutlass::layout::RowMajor layout(N);  // Stride = N
cutlass::Coord<2> coord{3, 7};        // Row 3, Column 7
int64_t offset = layout(coord);       // 3 * N + 7
```

### 36.6.4 Tile Shape Coordinates

In GEMM operations, tile shapes are represented as Coord<3>:

```cpp
// Threadblock tile shape
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
// Internally: Coord<3>{128, 128, 32}

// Warp tile shape (subset of threadblock tile)
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
// Internally: Coord<3>{64, 64, 32}

// Instruction shape (MMA atom shape)
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
// Internally: Coord<3>{16, 8, 16}

// Computing grid dimensions from tile shapes:
cutlass::gemm::GemmCoord problem_size{1024, 1024, 1024};
cutlass::gemm::GemmShape<128, 128, 32> tile_shape;

cutlass::Coord<2> grid_extent{
    (problem_size.m() + tile_shape.m() - 1) / tile_shape.m(),
    (problem_size.n() + tile_shape.n() - 1) / tile_shape.n()
};
// grid_extent = {8, 8}
```

---

## 36.7 Helper Functions Reference

### 36.7.1 make_Coord

```cpp
// 1D coordinate
template <typename Index = int>
CUTLASS_HOST_DEVICE
Coord<1, Index> make_Coord(Index idx0);

// 2D coordinate
template <typename Index = int>
CUTLASS_HOST_DEVICE
Coord<2, Index> make_Coord(Index idx0, Index idx1);

// 3D coordinate
template <typename Index = int>
CUTLASS_HOST_DEVICE
Coord<3, Index> make_Coord(Index idx0, Index idx1, Index idx2);

// 4D coordinate
template <typename Index = int>
CUTLASS_HOST_DEVICE
Coord<4, Index> make_Coord(Index idx0, Index idx1, Index idx2, Index idx3);

// 5D coordinate
template <typename Index = int>
CUTLASS_HOST_DEVICE
Coord<5, Index> make_Coord(Index idx0, Index idx1, Index idx2, Index idx3, Index idx4);

// With explicit index type:
auto c64 = cutlass::make_Coord<int64_t>(1000LL, 2000LL, 3000LL);
// Returns Coord<3, int64_t>
```

### 36.7.2 make_Coord_with_padding

```cpp
// Create a coordinate of rank N from fewer values, padding with fill_value
template <int N, typename Index = int>
CUTLASS_HOST_DEVICE
Coord<N, Index> make_Coord_with_padding(Index fill_value = Index(0));

// Examples:
auto c4 = cutlass::make_Coord_with_padding<4>(10, 20);
// Coord<4>{10, 20, 0, 0}

auto c4_filled = cutlass::make_Coord_with_padding<4>(1)(10, 20, 30);
// Coord<4>{10, 20, 30, 1}

auto c5 = cutlass::make_Coord_with_padding<5>(0)(100, 200);
// Coord<5>{100, 200, 0, 0, 0}
```

### 36.7.3 clamp

```cpp
// Free function clamp
template <int N, typename Index>
CUTLASS_HOST_DEVICE
Coord<N, Index> clamp(
    Coord<N, Index> const &value,
    Coord<N, Index> const &lower,
    Coord<N, Index> const &upper
);

// Usage:
cutlass::Coord<2> value{150, 300};
cutlass::Coord<2> lower{0, 0};
cutlass::Coord<2> upper{128, 256};
auto result = cutlass::clamp(value, lower, upper);  // {128, 256}
```

---

## 36.8 Code Examples

### 36.8.1 GEMM Problem Setup with Coordinates

```cpp
#include "cutlass/coord.h"
#include "cutlass/gemm/gemm.h"

void setup_gemm() {
    // Define problem dimensions using GemmCoord
    cutlass::gemm::GemmCoord problem_size{2048, 1024, 512};

    // Define tile shapes using Coord-like GemmShape
    cutlass::gemm::GemmShape<128, 128, 64> threadblock_tile;
    cutlass::gemm::GemmShape<64, 64, 64> warp_tile;
    cutlass::gemm::GemmShape<16, 8, 16> instruction_tile;

    // Compute grid dimensions
    dim3 grid(
        (problem_size.m() + threadblock_tile.m() - 1) / threadblock_tile.m(),
        (problem_size.n() + threadblock_tile.n() - 1) / threadblock_tile.n(),
        1  // Can be >1 for batched GEMM
    );

    printf("Problem: M=%d, N=%d, K=%d\n",
           problem_size.m(), problem_size.n(), problem_size.k());
    printf("Grid: (%d, %d)\n", grid.x, grid.y);
    printf("Total FLOPs: %lld\n", 2LL * problem_size.mno());
}
```

### 36.8.2 Convolution Tensor Dimensions

```cpp
#include "cutlass/coord.h"

void setup_convolution() {
    // Activation tensor: N x C x H x W
    cutlass::Tensor4DCoord activation{1, 256, 56, 56};

    // Filter tensor: K x C x R x S
    cutlass::Tensor4DCoord filter{512, 256, 3, 3};

    // Output tensor: N x K x P x Q
    int padding = 1;
    int stride = 1;
    int P = (activation.h() + 2 * padding - filter.h()) / stride + 1;
    int Q = (activation.w() + 2 * padding - filter.w()) / stride + 1;
    cutlass::Tensor4DCoord output{activation.n(), filter.n(), P, Q};

    // Compute total elements
    printf("Activation elements: %lld\n", (long long)activation.n() * activation.c() * activation.h() * activation.w());
    printf("Output elements: %lld\n", (long long)output.n() * output.c() * output.h() * output.w());

    // Implicit GEMM problem size
    cutlass::gemm::GemmCoord implicit_gemm_size{
        output.n() * output.h() * output.w(),  // M dimension
        filter.n(),                              // N dimension
        filter.c() * filter.h() * filter.w()    // K dimension
    };
}
```

### 36.8.3 Coordinate-Based Tensor Iteration

```cpp
#include "cutlass/coord.h"
#include "cutlass/tensor_view.h"

template <typename Element, typename Layout>
void fill_tensor(cutlass::TensorView<Element, Layout> view, Element value) {
    cutlass::Coord<Layout::kRank> extent = view.extent();

    // Generic iteration over arbitrary-rank tensors
    // For a 2D tensor:
    for (int i = 0; i < extent[0]; ++i) {
        for (int j = 0; j < extent[1]; ++j) {
            auto coord = cutlass::make_Coord(i, j);
            view.at(coord) = value;
        }
    }
}

// Using dot product for custom linear indexing:
template <int N>
int64_t compute_offset(cutlass::Coord<N> const &coord,
                       cutlass::Coord<N, int64_t> const &stride) {
    return coord.dot(stride);
}

// Example: 3D tensor with custom strides
auto strides = cutlass::make_Coord<int64_t>(256 * 64, 64, 1);
auto coord = cutlass::make_Coord(2, 3, 5);
int64_t offset = compute_offset(coord, strides);
// offset = 2 * 256*64 + 3 * 64 + 5 = 32805
```

### 36.8.4 Tile Coordinate Computation

```cpp
// Compute per-thread tile coordinates within a threadblock
__global__ void tiled_operation(
    float* output,
    int M, int N,
    int TILE_M, int TILE_N
) {
    // Threadblock tile coordinates
    cutlass::Coord<2> tb_coord{
        blockIdx.x,  // Tile row
        blockIdx.y   // Tile column
    };

    // Thread tile coordinates within threadblock
    cutlass::Coord<2> thread_coord{
        threadIdx.x,
        threadIdx.y
    };

    // Global element coordinates
    cutlass::Coord<2> tile_size{TILE_M, TILE_N};
    cutlass::Coord<2> global_coord = tb_coord * tile_size + thread_coord;

    // Clamp to problem bounds
    cutlass::Coord<2> problem_size{M, N};
    cutlass::Coord<2> zero{0, 0};
    cutlass::Coord<2> clamped = global_coord.clamp(zero, problem_size - cutlass::Coord<2>{1, 1});

    // Check if this thread has valid work
    bool valid = (global_coord[0] < M) && (global_coord[1] < N);

    if (valid) {
        int64_t offset = global_coord[0] * N + global_coord[1];
        output[offset] = static_cast<float>(global_coord[0] + global_coord[1]);
    }
}
```

---

## 36.9 Summary

The `Coord<N, Index>` template is a fundamental building block in CUTLASS for:

| Use Case | Rank | Example |
|---|---|---|
| GEMM problem size | 3 | `GemmCoord{M, N, K}` |
| Matrix/tensor dimensions | 2-5 | `Coord<4>{N, C, H, W}` |
| Tile shapes | 3 | `GemmShape<128, 128, 64>` |
| Stride vectors | N | `Coord<3, int64_t>{stride_m, stride_n, stride_k}` |
| Grid dimensions | 2-3 | `Coord<2>{grid_m, grid_n}` |
| Tensor coordinates | N | `make_Coord(i, j, k)` |
| Linear offset computation | N | `coord.dot(stride)` |
