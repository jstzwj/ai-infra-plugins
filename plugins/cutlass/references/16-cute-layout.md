# CuTe Layout System

## Table of Contents

- [1. Overview](#1-overview)
- [2. Layout Structure: Shape + Stride](#2-layout-structure-shape--stride)
- [3. Shape: Hierarchical Tuple of Integers](#3-shape-hierarchical-tuple-of-integers)
- [4. Stride: Hierarchical Tuple for Offset Computation](#4-stride-hierarchical-tuple-for-offset-computation)
- [5. Layout Factory Functions](#5-layout-factory-functions)
- [6. Layout Algebra](#6-layout-algebra)
- [7. Layout Operations](#7-layout-operations)
- [8. Shape Operations](#8-shape-operations)
- [9. Stride Operations](#9-stride-operations)
- [10. Coordinate Systems](#10-coordinate-systems)
- [11. Layout Predicates and Static Assertions](#11-layout-predicates-and-static-assertions)
- [12. Tiled Layout Construction](#12-tiled-layout-construction)
- [13. Swizzled Layouts](#13-swizzled-layouts)
- [14. Code Examples](#14-code-examples)

---

## 1. Overview

The CuTe layout system is the foundational abstraction in CUTLASS's CuTe library. A layout defines a mapping from a coordinate space (the "logical" index) to a linear offset space (the "physical" index). This mapping enables CuTe to express how threads map to data elements, how data is arranged in shared memory, and how tensor indices translate to memory addresses.

The layout system is designed around compile-time algebra: layouts are parameterized by compile-time integers (via `cute::Int<N>`) and compile-time tuples (via `cute::tuple`), enabling the compiler to reason about and optimize data access patterns. This compile-time representation is critical for achieving peak GPU performance, as the compiler can eliminate index arithmetic and emit specialized load/store instructions.

Key design goals:
- **Composability**: Layouts can be combined, sliced, and tiled to build complex access patterns from simple primitives.
- **Static analysis**: The shape and stride are known at compile time whenever possible, enabling static assertions and compile-time optimization.
- **Hierarchical**: Layouts support nested tuple structures that naturally represent multi-dimensional and hierarchical data access patterns.
- **Hardware mapping**: Layouts serve as the bridge between logical tensor indices and physical memory addresses, including support for swizzling and bank conflict avoidance.

All layout types live in the `cute::` namespace. The primary header is `include/cute/layout.hpp`.

---

## 2. Layout Structure: Shape + Stride

A CuTe layout is a pair `(Shape, Stride)` where:
- **Shape** defines the size of each dimension as a hierarchical tuple of integers.
- **Stride** defines the step size for each dimension as a hierarchical tuple of integers.

The layout maps a coordinate `c` (which may be a hierarchical tuple) to a linear offset via the formula:

```
offset = sum(coord_i * stride_i for each dimension i)
```

The layout type is:

```cpp
template <class Shape, class Stride>
struct Layout : private cute::tuple<Shape, Stride> {
    // Accessors
    using ShapeType = Shape;
    using StrideType = Stride;

    // The shape and stride can be accessed via .shape() and .stride()
    CUTE_HOST_DEVICE constexpr decltype(auto) shape() const;
    CUTE_HOST_DEVICE constexpr decltype(auto) stride() const;

    // Subscript operator: map coordinate to linear offset
    template <class Coord>
    CUTE_HOST_DEVICE constexpr auto operator()(Coord const& coord) const;
};
```

The layout stores no data -- it is purely a mapping function. The data itself lives in a tensor (see chapter 17), which pairs a layout with a data pointer or engine.

### Invariants

1. `rank(shape) == rank(stride)` -- the shape and stride must have matching tuple structure.
2. `size(layout) == size(shape)` -- the total number of elements mapped by the layout equals the product of the shape.
3. All coordinates in `[0, shape)` are valid inputs to `operator()`.

### Example: Simple 1D Layout

```cpp
// A 1D layout mapping 128 elements with stride 1 (contiguous)
auto layout = Layout<Shape<_128>, Stride<_1>>{};
// layout(0) == 0, layout(1) == 1, ..., layout(127) == 127
```

### Example: 2D Layout (Row-Major)

```cpp
// A 128x64 row-major layout
// Shape: (128, 64), Stride: (1, 128)
auto layout = Layout<Shape<_128, _64>, Stride<_1, _128>>{};
// layout(make_coord(0, 0)) == 0
// layout(make_coord(0, 1)) == 1
// layout(make_coord(1, 0)) == 128
// layout(make_coord(1, 1)) == 129
```

### Example: 2D Layout (Column-Major)

```cpp
// A 128x64 column-major layout
// Shape: (128, 64), Stride: (64, 1)
auto layout = Layout<Shape<_128, _64>, Stride<_64, _1>>{};
// layout(make_coord(0, 0)) == 0
// layout(make_coord(0, 1)) == 1
// layout(make_coord(1, 0)) == 64
// layout(make_coord(1, 1)) == 65
```

---

## 3. Shape: Hierarchical Tuple of Integers

The shape of a layout describes the extent of each dimension. Shapes can be:
- A single integer (1D layout): `Int<N>` or dynamic `int`
- A tuple of integers (multi-dimensional): `tuple<Int<M>, Int<N>>`
- A nested tuple (hierarchical): `tuple<tuple<Int<M>, Int<N>>, Int<K>>`

### Static Integers

CuTe uses `cute::Int<N>` as a compile-time integer constant, similar to `std::integral_constant<int, N>`. Common aliases:

```cpp
using _1  = Int<1>;
using _2  = Int<2>;
using _4  = Int<4>;
using _8  = Int<8>;
using _16 = Int<16>;
using _32 = Int<32>;
using _64 = Int<64>;
using _128 = Int<128>;
using _256 = Int<256>;
```

These are used to construct compile-time shapes and strides. The underscore prefix convention (e.g., `_128`) distinguishes static integers from runtime values.

### Dynamic Integers

When a dimension size is only known at runtime, use a plain `int` or `int64_t`:

```cpp
int M = problem_M;
auto shape = make_shape(M, _64);  // (dynamic, 64)
```

### Shape Construction

```cpp
// 1D shape
auto s1 = Shape<_128>{};             // Static 1D shape of 128
auto s2 = Shape<int>{};              // Dynamic 1D shape (value set at construction)

// 2D shape
auto s3 = Shape<_128, _64>{};        // Static 2D shape 128x64
auto s4 = Shape<int, _64>{};         // Mixed: dynamic M, static 64

// 3D shape
auto s5 = Shape<_128, _64, _32>{};   // Static 3D shape

// Hierarchical shape: ((M, N), K)
auto s6 = Shape<Shape<_128, _64>, _32>{};
```

### make_shape()

The `make_shape()` function constructs shapes with automatic type deduction:

```cpp
auto shape_1d = make_shape(128);                         // Shape<int>{128}
auto shape_2d = make_shape(_128{}, _64{});               // Shape<_128, _64>{}
auto shape_3d = make_shape(_128{}, _64{}, _32{});        // Shape<_128, _64, _32>{}
auto shape_hier = make_shape(make_shape(_128{}, _64{}),  // Shape<Shape<_128, _64>, _32>{}
                             _32{});
```

### Shape Properties

```cpp
// Get the total number of elements (product of all dimensions)
auto s = make_shape(_128{}, _64{});
auto total = size(s);    // Int<8192>{}

// Get the rank (number of dimensions at the top level)
auto r = rank(s);        // Int<2>{}

// Get the depth (maximum nesting depth)
auto d = depth(s);       // Int<1>{}

// Access individual dimensions via get<I>()
auto dim0 = get<0>(s);   // _128{}
auto dim1 = get<1>(s);   // _64{}
```

---

## 4. Stride: Hierarchical Tuple for Offset Computation

The stride determines how a change in each coordinate dimension maps to a change in the linear offset. Like shapes, strides can be static integers, dynamic integers, or hierarchical tuples.

### Stride Semantics

For a layout `L = (Shape, Stride)` and a coordinate `(i, j, k, ...)`:

```
L(i, j, k, ...) = i * stride_0 + j * stride_1 + k * stride_2 + ...
```

where `stride_i` is the i-th element of the Stride tuple.

### Common Stride Patterns

```cpp
// Row-major (C-order): innermost dimension has stride 1
// For shape (M, N): stride = (1, M)
auto rm_stride = make_stride(_1{}, _128{});

// Column-major (F-order): outermost dimension has stride 1
// For shape (M, N): stride = (N, 1)
auto cm_stride = make_stride(_64{}, _1{});

// Generalized strides for non-contiguous access
// E.g., accessing every other element: stride = 2
auto strided = make_stride(_2{});

// Negative strides for reverse traversal
auto reverse_stride = make_stride(Int<-1>{});
```

### Stride Construction

```cpp
auto stride_1d = make_stride(1);                    // Stride<int>{1}
auto stride_2d = make_stride(_1{}, _128{});          // Stride<_1, _128>{}
auto stride_3d = make_stride(_1{}, _128{}, _8192{}); // Stride<_1, _128, _8192>{}
auto stride_hier = make_stride(make_stride(_1{}, _8{}), _128{});
```

### Zero Stride

A stride of zero means the dimension does not contribute to the offset. This is used for broadcasting:

```cpp
// A layout that "broadcasts" the second dimension
// Shape (M, N) but stride (1, 0) -- all N elements map to the same offset
auto broadcast_layout = make_layout(make_shape(_128{}, _64{}), make_stride(_1{}, _0{}));
// broadcast_layout(i, j) == i for all j in [0, 64)
```

---

## 5. Layout Factory Functions

### make_layout()

The primary factory for constructing layouts:

```cpp
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto make_layout(Shape const& shape, Stride const& stride);

// Overload with default stride (compact/left-aligned)
template <class Shape>
CUTE_HOST_DEVICE constexpr auto make_layout(Shape const& shape);
```

When no stride is provided, `make_layout()` computes a compact (right-major) stride:

```cpp
auto layout = make_layout(make_shape(_128{}, _64{}));
// Equivalent to: make_layout(make_shape(_128{}, _64{}), make_stride(_1{}, _128{}))
```

```cpp
// Explicit stride specification
auto col_major = make_layout(make_shape(_128{}, _64{}), make_stride(_64{}, _1{}));

// 3D layout
auto layout_3d = make_layout(make_shape(_32{}, _64{}, _128{}));
// Stride defaults to: (1, 32, 32*64) = (1, 32, 2048)
```

### make_ordered_layout()

Creates a layout with a specified major dimension ordering:

```cpp
template <class Shape, class Order>
CUTE_HOST_DEVICE constexpr auto make_ordered_layout(Shape const& shape, Order const& order);
```

The `order` parameter is a permutation that specifies the dimension ordering from fastest-varying to slowest-varying.

```cpp
// Row-major 2D (M, N): N is fastest-varying, M is slowest
auto row_major = make_ordered_layout(make_shape(_128{}, _64{}), Step<_2, _1>{});
// Stride: (64, 1)

// Column-major 2D (M, N): M is fastest-varying, N is slowest
auto col_major = make_ordered_layout(make_shape(_128{}, _64{}), Step<_1, _2>{});
// Stride: (1, 128)
```

### make_fragment_like()

Creates a fragment layout that mirrors the storage pattern of a tensor's fragment for a given MMA atom:

```cpp
template <class Engine, class Layout, class MMA_Op>
CUTE_HOST_DEVICE constexpr auto make_fragment_like(Tensor<Engine, Layout> const& tensor, MMA_Op const& mma_op);
```

This is typically used to allocate register tensors that match the data layout expected by a hardware MMA instruction:

```cpp
// Create a fragment layout matching the A operand of an MMA atom
auto frag_layout = make_fragment_like(gA, tiled_mma);
```

### make_layout_like()

Creates a layout with the same shape as another layout but with a compact stride:

```cpp
template <class Layout>
CUTE_HOST_DEVICE constexpr auto make_layout_like(Layout const& layout);
```

---

## 6. Layout Algebra

### LayoutLeft (Column-Major)

`LayoutLeft` represents a column-major layout where the leftmost (first) dimension is contiguous:

```cpp
template <class Shape>
using LayoutLeft = Layout<Shape, decltype(make_layout(declval<Shape>()).stride())>;
```

Usage:

```cpp
// Column-major 128x64 layout: stride = (1, M)
auto left_layout = make_layout(make_shape(_128{}, _64{}), make_stride(_1{}, _128{}));
```

### LayoutRight (Row-Major)

`LayoutRight` represents a row-major layout where the rightmost (last) dimension is contiguous:

```cpp
// Row-major 128x64 layout: stride = (N, 1)
auto right_layout = make_layout(make_shape(_128{}, _64{}), make_stride(_64{}, _1{}));
```

### LayoutLeftProjected

A column-major layout where the first dimension is projected (collapsed) to stride 0, effectively broadcasting:

```cpp
// Projected layout: first dimension has stride 0
auto proj_left = make_layout(make_shape(_128{}, _64{}), make_stride(_0{}, _1{}));
```

### LayoutRightProjected

A row-major layout where the last dimension is projected to stride 0:

```cpp
auto proj_right = make_layout(make_shape(_128{}, _64{}), make_stride(_1{}, _0{}));
```

### Layout Algebra Properties

The layout algebra supports the following operations on stride patterns:

| Pattern | Stride Formula | Description |
|---------|---------------|-------------|
| LayoutLeft | stride = (1, M, M*N, ...) | First dimension contiguous |
| LayoutRight | stride = (N*K, N, 1) | Last dimension contiguous |
| LayoutLeftProjected | stride = (0, 1, M, ...) | First dimension broadcast |
| LayoutRightProjected | stride = (..., N, 0) | Last dimension broadcast |

---

## 7. Layout Operations

### composition()

Combines two layouts `A` and `B` such that the result maps coordinates through `B` first, then through `A`. This is the fundamental composition operation:

```cpp
template <class LayoutA, class LayoutB>
CUTE_HOST_DEVICE constexpr auto composition(LayoutA const& layoutA, LayoutB const& layoutB);
```

The composition `composition(A, B)` produces a layout `C` such that:
```
C(coord) = A(B(coord))
```

```cpp
// Layout A: 128 contiguous elements
auto A = make_layout(_128{});

// Layout B: 32 elements with stride 4 (every 4th element)
auto B = make_layout(make_shape(_32{}), make_stride(_4{}));

// Composition: 32 elements, each 4 apart within a space of 128
auto C = composition(A, B);
// C(i) = A(B(i)) = 4*i, for i in [0, 32)
```

Composition with multi-dimensional layouts:

```cpp
auto A = make_layout(make_shape(_128{}, _64{}), make_stride(_1{}, _128{}));
auto B = make_layout(make_shape(_32{}, _16{}), make_stride(_4{}, _4{}));
auto C = composition(A, B);
// Maps (i, j) to (4*i + 4*j * 128) = 4*i + 512*j
```

### complementary()

Given a layout `A` and a target shape, finds the complementary layout `B` such that `composition(A, B)` covers the target shape without overlap:

```cpp
template <class Layout, class Shape>
CUTE_HOST_DEVICE constexpr auto complementary(Layout const& layout, Shape const& shape);
```

```cpp
auto layout = make_layout(_32{}, make_stride(_4{}));   // 32 elements, stride 4
auto comp = complementary(layout, _128{});
// comp maps 4 contiguous elements (fills gaps between the stride-4 elements)
```

### tile()

Tiles a layout with another layout, creating a hierarchical layout where the tiling layout is applied at each position of the original layout:

```cpp
template <class LayoutA, class LayoutB>
CUTE_HOST_DEVICE constexpr auto tile(LayoutA const& layoutA, LayoutB const& layoutB);
```

```cpp
// Original layout: 128x64
auto A = make_layout(make_shape(_128{}, _64{}), make_stride(_1{}, _128{}));

// Tile layout: 32x16
auto B = make_layout(make_shape(_32{}, _16{}), make_stride(_1{}, _32{}));

auto tiled = tile(A, B);
// Creates a tiled layout with the tile pattern repeated across A
```

### flatten()

Converts a hierarchical layout into a flat layout by collapsing all nested tuples:

```cpp
template <class Layout>
CUTE_HOST_DEVICE constexpr auto flatten(Layout const& layout);
```

```cpp
auto hier = make_layout(make_shape(make_shape(_4{}, _8{}), _16{}));
auto flat = flatten(hier);
// Result: Layout<Shape<_4, _8, _16>, Stride<...>>
```

### coalesce()

Merges adjacent layout levels that are compatible (i.e., their dimensions can be fused without changing the mapping):

```cpp
template <class Layout>
CUTE_HOST_DEVICE constexpr auto coalesce(Layout const& layout);
```

```cpp
// A layout with two consecutive contiguous dimensions
auto layout = make_layout(make_shape(_32{}, _4{}), make_stride(_1{}, _32{}));
auto coalesced = coalesce(layout);
// Result: Layout<Shape<_128>, Stride<_1>> -- the two dims fuse into one
```

Coalescing only merges dimensions when the stride of the next dimension equals the size of the current dimension times the stride of the current dimension (i.e., they are contiguous).

### filter_zero()

Removes dimensions with zero stride from the layout, which corresponds to removing broadcast dimensions:

```cpp
template <class Layout>
CUTE_HOST_DEVICE constexpr auto filter_zero(Layout const& layout);
```

```cpp
auto layout = make_layout(make_shape(_128{}, _64{}), make_stride(_1{}, _0{}));
auto filtered = filter_zero(layout);
// Result: Layout<Shape<_128>, Stride<_1>> -- the broadcast dimension is removed
```

### group()

Groups layout dimensions, reorganizing the tuple structure:

```cpp
template <int... Is, class Layout>
CUTE_HOST_DEVICE constexpr auto group(Layout const& layout);
```

```cpp
auto layout = make_layout(make_shape(_32{}, _8{}, _64{}));
auto grouped = group<2>(layout);
// Groups first 2 dimensions: Shape<Shape<_32, _8>, _64>
```

### logical_product()

Computes the logical product of two layouts, producing a layout that maps all combinations:

```cpp
template <class LayoutA, class LayoutB>
CUTE_HOST_DEVICE constexpr auto logical_product(LayoutA const& a, LayoutB const& b);
```

### logical_divide()

Computes the logical division of a layout by another, effectively partitioning:

```cpp
template <class LayoutA, class LayoutB>
CUTE_HOST_DEVICE constexpr auto logical_divide(LayoutA const& a, LayoutB const& b);
```

```cpp
auto A = make_layout(make_shape(_128{}));
auto B = make_layout(make_shape(_32{}));
auto div = logical_divide(A, B);
// Result: Shape<Shape<_32>, Shape<_4>> -- 128 / 32 = 4 tiles of 32
```

### zipped_divide()

Like logical_divide but returns a layout where the inner and outer dimensions are zipped together:

```cpp
template <class LayoutA, class LayoutB>
CUTE_HOST_DEVICE constexpr auto zipped_divide(LayoutA const& a, LayoutB const& b);
```

### tiled_divide()

Divides a layout into tiles:

```cpp
template <class LayoutA, class LayoutB>
CUTE_HOST_DEVICE constexpr auto tiled_divide(LayoutA const& a, LayoutB const& b);
```

### left_inverse() / right_inverse()

Computes the inverse of a layout when it is bijective:

```cpp
template <class Layout>
CUTE_HOST_DEVICE constexpr auto left_inverse(Layout const& layout);

template <class Layout>
CUTE_HOST_DEVICE constexpr auto right_inverse(Layout const& layout);
```

---

## 8. Shape Operations

### size()

Returns the total number of elements (product of all dimensions):

```cpp
template <class Shape>
CUTE_HOST_DEVICE constexpr auto size(Shape const& shape);

// Overload for Layout: returns size of the layout's shape
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto size(Layout<Shape, Stride> const& layout);
```

```cpp
auto s = make_shape(_128{}, _64{});
auto total = size(s);         // Int<8192>{}
auto runtime_total = size(make_shape(M, N));  // int: M*N
```

### rank()

Returns the number of top-level dimensions:

```cpp
template <class Shape>
CUTE_HOST_DEVICE constexpr auto rank(Shape const& shape);
```

```cpp
auto s = make_shape(_128{}, _64{}, _32{});
auto r = rank(s);             // Int<3>{}

auto hier = make_shape(make_shape(_4{}, _8{}), _16{});
auto r2 = rank(hier);         // Int<2>{} (two top-level elements: tuple and _16)
```

### depth()

Returns the maximum nesting depth of a shape:

```cpp
template <class Shape>
CUTE_HOST_DEVICE constexpr auto depth(Shape const& shape);
```

```cpp
auto flat = make_shape(_128{}, _64{});
auto d1 = depth(flat);        // Int<0>{} -- no nesting

auto hier = make_shape(make_shape(_4{}, _8{}), _16{});
auto d2 = depth(hier);        // Int<1>{} -- one level of nesting
```

### get()

Accesses individual dimensions of a shape:

```cpp
template <int I, class Shape>
CUTE_HOST_DEVICE constexpr auto get(Shape const& shape);
```

```cpp
auto s = make_shape(_128{}, _64{}, _32{});
auto dim0 = get<0>(s);        // _128{}
auto dim1 = get<1>(s);        // _64{}
auto dim2 = get<2>(s);        // _32{}
```

### shape()

Returns the shape of a layout:

```cpp
auto layout = make_layout(make_shape(_128{}, _64{}), make_stride(_1{}, _128{}));
auto s = shape(layout);        // Shape<_128, _64>{}
```

### product(), ceil_div()

Utility operations on shapes:

```cpp
auto prod = size(shape);                       // Total product
auto ceil = ceil_div(_128{}, _32{});           // Int<4>{}
```

---

## 9. Stride Operations

### stride()

Returns the stride of a layout:

```cpp
auto layout = make_layout(make_shape(_128{}, _64{}), make_stride(_1{}, _128{}));
auto s = stride(layout);       // Stride<_1, _128>{}
```

### stride comparison

Strides can be compared to determine layout types:

```cpp
template <class Layout>
CUTE_HOST_DEVICE constexpr bool is_column_major(Layout const& layout);

template <class Layout>
CUTE_HOST_DEVICE constexpr bool is_row_major(Layout const& layout);
```

### compact_col_major() / compact_row_major()

Compute compact strides for a given shape:

```cpp
auto cm_stride = compact_col_major(make_shape(_128{}, _64{}));
// Result: (1, 128)

auto rm_stride = compact_row_major(make_shape(_128{}, _64{}));
// Result: (64, 1)
```

---

## 10. Coordinate Systems

CuTe supports two coordinate systems for accessing layouts and tensors.

### Natural (Hierarchical) Coordinates

Natural coordinates mirror the hierarchical structure of the shape. For a layout with shape `Shape<Shape<_4, _8>, _16>`, a natural coordinate is `tuple<tuple<int, int>, int>`:

```cpp
auto layout = make_layout(make_shape(make_shape(_4{}, _8{}), _16{}));

// Natural (hierarchical) coordinate
auto coord = make_coord(make_coord(2, 3), 7);
auto offset = layout(coord);  // Computes 2*stride_00 + 3*stride_01 + 7*stride_1
```

### Flat Coordinates

Flat coordinates treat the layout as a 1D range. A flat coordinate is a single integer:

```cpp
// Flat coordinate (0 to size(layout)-1)
for (int i = 0; i < size(layout); ++i) {
    auto offset = layout(i);  // Linear offset
}
```

### make_coord()

Constructs coordinates:

```cpp
// 1D coordinate
auto c1 = 42;

// 2D coordinate
auto c2 = make_coord(10, 20);

// 3D coordinate
auto c3 = make_coord(10, 20, 30);

// Hierarchical coordinate
auto c4 = make_coord(make_coord(2, 3), 7);
```

### Coordinate Conversion

`idx2crd()` converts a flat index to a hierarchical coordinate:

```cpp
template <class Idx, class Shape>
CUTE_HOST_DEVICE constexpr auto idx2crd(Idx const& idx, Shape const& shape);
```

```cpp
auto shape = make_shape(_128{}, _64{});
auto coord = idx2crd(65, shape);  // make_coord(1, 1)
```

`crd2idx()` converts a hierarchical coordinate back to a flat index:

```cpp
template <class Coord, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto crd2idx(Coord const& coord, Shape const& shape, Stride const& stride);
```

---

## 11. Layout Predicates and Static Assertions

CuTe provides compile-time predicates for checking layout properties:

### is_layout

```cpp
template <class T>
struct is_layout : false_type {};

template <class Shape, class Stride>
struct is_layout<Layout<Shape, Stride>> : true_type {};
```

### is_static / is_dynamic

```cpp
template <class Layout>
struct is_static;   // true if all shape and stride values are compile-time constants

template <class Layout>
struct is_dynamic;  // true if any shape or stride value is runtime
```

### is_compatible

```cpp
// Check if two layouts have compatible shapes
template <class LayoutA, class LayoutB>
struct is_compatible;
```

### Static assertions

CuTe uses `CUTE_STATIC_ASSERT` for compile-time checks:

```cpp
// Ensure layout sizes match
CUTE_STATIC_ASSERT(size(layout_a) == size(layout_b), "Layout sizes must match");

// Ensure layout is static
static_assert(cute::is_static_v<decltype(layout)>, "Layout must be static");

// Ensure correct rank
static_assert(rank(layout) == 2, "Layout must be 2D");
```

### Layout predicates

```cpp
// Check if a layout is a bijection (one-to-one mapping)
template <class Layout>
CUTE_HOST_DEVICE constexpr bool is_bijective(Layout const& layout);

// Check if a layout is compact (contiguous without gaps)
template <class Layout>
CUTE_HOST_DEVICE constexpr bool is_compact(Layout const& layout);

// Check if a layout has zero strides (broadcast dimensions)
template <class Layout>
CUTE_HOST_DEVICE constexpr bool has_zero_stride(Layout const& layout);
```

---

## 12. Tiled Layout Construction

Tiled layouts are used to represent how a tensor is partitioned across threads in a cooperative operation (like a tiled MMA or tiled copy).

### TiledCopy Layout

A `TiledCopy` pairs a `Copy_Atom` (hardware copy instruction) with a thread layout that tiles the atom across a thread block:

```cpp
// Create a tiled copy that copies 128x64 elements with 128 threads
auto tiled_copy = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::half_t>, cute::half_t>{},
    Layout<Shape<_128, _1>, Stride<_1, _0>>{},     // Thread layout
    Layout<Shape<_1, _8>, Stride<_1, _0>>{}         // Value layout per thread
);
```

### TiledMMA Layout

A `TiledMMA` pairs an `MMA_Atom` (hardware MMA instruction) with a thread layout:

```cpp
// Create a tiled MMA that tiles a 16x8x16 MMA across a warp
auto tiled_mma = make_tiled_mma(
    MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{},
    Layout<Shape<_2, _1, _1>>{},                    // 2x replication along M
    Tile<_32, _8, _16>{}                            // Resulting tile size
);
```

### Partitioning with Tiled Layouts

Tiled layouts are used to partition tensors among threads:

```cpp
// Partition tensor A for the MMA
auto tPgA = tiled_mma.get_slice(thread_id).partition_A(gA);
// tPgA is a tensor representing thread's view of A for the MMA

// Partition tensor B
auto tPgB = tiled_mma.get_slice(thread_id).partition_B(gB);

// Partition tensor C (accumulator)
auto tCrC = tiled_mma.get_slice(thread_id).partition_C(gC);
```

---

## 13. Swizzled Layouts

Swizzled layouts apply a permutation to addresses to avoid shared memory bank conflicts. A swizzle function XORs bits of the address to redistribute accesses across memory banks.

### Swizzle Basics

```cpp
// A 3-2 swizzle (3 bits from row, 2 bits from column)
// The swizzle function XORs selected address bits
template <class Layout>
CUTE_HOST_DEVICE constexpr auto compose_swizzle(Layout const& layout);
```

### Built-in Swizzle Types

```cpp
// Common swizzle patterns for shared memory layouts
// Swizzle<3, 3, 3>: XOR lower 3 bits, useful for 32-bit data
using Swizzle3 = Swizzle<3, 3, 3>;

// Swizzle<3, 4, 3>: wider swizzle for 16-bit data
using Swizzle4 = Swizzle<3, 4, 3>;

// Apply swizzle to a layout
auto swizzled_layout = compose_swizzle(basil_layout, Swizzle<3, 3, 3>{});
```

### Swizzle in Shared Memory

When constructing shared memory layouts for GEMM, swizzling is critical for avoiding bank conflicts:

```cpp
// Shared memory layout for matrix A with swizzle to avoid bank conflicts
auto smem_layout_a = composition(
    Swizzle<3, 4, 3>{},
    make_layout(make_shape(_128{}, _16{}), make_stride(_1{}, _128{}))
);

// The swizzle permutes the address bits, distributing
// thread accesses across 32 shared memory banks
```

### Swizzle Behavior

The swizzle operation XORs selected bits of the address:

```
swizzled_addr = base_addr XOR (select_bits << shift)
```

Where:
- `select_bits` are extracted from specific bit positions
- `shift` determines where the XOR is applied

This effectively reorders data elements so that consecutive threads (which would otherwise access the same bank) instead access different banks.

---

## 14. Code Examples

### Example 1: Basic Layout Construction and Access

```cpp
#include "cute/layout.hpp"

using namespace cute;

// Create a 128x64 row-major layout
auto layout = make_layout(make_shape(_128{}, _64{}), make_stride(_64{}, _1{}));

// Access elements
auto offset_00 = layout(make_coord(0, 0));   // 0
auto offset_10 = layout(make_coord(1, 0));   // 64
auto offset_01 = layout(make_coord(0, 1));   // 1
auto offset_11 = layout(make_coord(1, 1));   // 65

// Size and rank
static_assert(size(layout) == 8192);
static_assert(rank(layout) == 2);
```

### Example 2: Layout Composition for Tiling

```cpp
using namespace cute;

// Original 256-element contiguous layout
auto full = make_layout(_256{});

// Tile of 32 elements
auto tile = make_layout(_32{});

// Divide into tiles
auto tiled = logical_divide(full, tile);
// tiled.shape() = (32, 8) -- 8 tiles of 32 elements

// Get a specific tile
auto tile_3 = tiled(make_coord(_, 3));
// This gives the 3rd tile's layout
```

### Example 3: Hierarchical Layout for Thread Mapping

```cpp
using namespace cute;

// Warp-level thread layout: 32 threads arranged as (4, 8)
auto thread_layout = make_layout(make_shape(_4{}, _8{}));

// Each thread handles a 2x4 block of elements
auto value_layout = make_layout(make_shape(_2{}, _4{}));

// Combined: each thread handles its own block, threads arranged in a grid
auto combined = make_layout(
    make_shape(make_shape(_2{}, _4{}), make_shape(_4{}, _8{})),
    make_stride(make_stride(_1{}, _32{}), make_stride(_8{}, _64{}))
);
// Maps (thread_row, thread_col, value_row, value_col) to offset
```

### Example 4: Layout for Shared Memory with Swizzle

```cpp
using namespace cute;

// Base shared memory layout for A matrix in GEMM
// Shape: (BM, BK) = (128, 32) column-major
auto smem_shape = make_shape(_128{}, _32{});
auto smem_layout_base = make_layout(smem_shape, make_stride(_1{}, _128{}));

// Apply swizzle to avoid bank conflicts
// Swizzle<3, 3, 3> XORs 3 bits to spread accesses across banks
auto smem_layout = composition(
    Swizzle<3, 3, 3>{},
    smem_layout_base
);

// The resulting layout still has shape (128, 32) but with permuted addresses
// When threads read along the M dimension, they access different banks
```

### Example 5: Coalesce and Flatten

```cpp
using namespace cute;

// A hierarchical layout from tiling
auto hier = make_layout(
    make_shape(make_shape(_4{}, _8{}), make_shape(_2{}, _16{})),
    make_stride(make_stride(_1{}, _4{}), make_stride(_32{}, _64{}))
);

// Flatten to a single-level tuple
auto flat = flatten(hier);
// Shape: (_4, _8, _2, _16), Stride: (_1, _4, _32, _64)

// Coalesce contiguous dimensions
auto coal = coalesce(flat);
// Since _4 * _1 == _4 (contiguous), first two dims merge:
// Shape: (_32, _2, _16), Stride: (_1, _32, _64)
// And _32 * _1 != _32, _2 * _32 == 64:
// Shape: (_32, _32), Stride: (_1, _32)
```

### Example 6: Complementary Layout for Data Partitioning

```cpp
using namespace cute;

// Thread layout: 4 threads access every 4th element (stride 4)
auto thread_layout = make_layout(make_shape(_32{}), make_stride(_4{}));

// Find the complementary layout that covers the remaining elements
auto comp = complementary(thread_layout, _128{});
// comp has 4 elements with stride 1, covering the gaps

// Together, thread_layout and comp cover all 128 elements
auto combined = composition(thread_layout, comp);
// This effectively partitions the 128 elements among the thread group
```

### Example 7: Layout Predicates

```cpp
using namespace cute;

auto layout = make_layout(make_shape(_128{}, _64{}), make_stride(_1{}, _128{}));

// Check layout properties at compile time
static_assert(is_layout_v<decltype(layout)>);
static_assert(is_static_v<decltype(layout)>);
static_assert(rank(layout) == 2);
static_assert(size(layout) == 8192);
static_assert(!has_zero_stride(layout));

// Runtime check for bijectivity
if (is_bijective(layout)) {
    // All offsets are unique, can safely invert
    auto inv = right_inverse(layout);
}
```

### Example 8: Layout for MMA Atom Fragment

```cpp
using namespace cute;

// The layout for an MMA 16x8x16 F16 fragment
// For operand A (M x K): 16 rows, 16 columns
// The MMA instruction expects data in specific register positions

// Thread layout within a warp for the MMA
auto thr_layout = make_layout(make_shape(_4{}, _8{}));

// Fragment layout per thread for A operand
auto frag_A = make_layout(
    make_shape(make_shape(_2{}, _2{})),   // 4 elements per thread
    make_stride(make_stride(_1{}, _8{}))
);

// Combined with thread layout gives the full 16x16 mapping
```

---

## Summary

The CuTe layout system provides a powerful algebra for describing data access patterns on GPUs. Key takeaways:

1. **Layouts are pure functions** from coordinates to offsets, defined by a (Shape, Stride) pair.
2. **Hierarchical shapes and strides** enable natural expression of multi-dimensional data arrangements.
3. **Compile-time integers** (`Int<N>`) allow the compiler to optimize layout arithmetic away.
4. **Composition, tiling, and division** are the primary algebraic operations for building complex layouts from simple ones.
5. **Swizzling** provides hardware-aware address permutation for shared memory bank conflict avoidance.
6. **Predicates and static assertions** ensure correctness at compile time.

The layout system is the backbone of CuTe and enables the tensor abstraction (covered in chapter 17) and the MMA atom system (covered in chapter 18).
