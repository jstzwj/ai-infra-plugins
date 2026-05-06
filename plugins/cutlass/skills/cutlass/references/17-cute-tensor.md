# CuTe Tensor System

## Table of Contents

- [1. Overview](#1-overview)
- [2. Tensor Structure: Engine + Layout](#2-tensor-structure-engine--layout)
- [3. Engine Types](#3-engine-types)
- [4. Tensor Creation](#4-tensor-creation)
- [5. Tensor Operations](#5-tensor-operations)
- [6. Tensor Concepts and Type Traits](#6-tensor-concepts-and-type-traits)
- [7. Tensor Transformations](#7-tensor-transformations)
- [8. Sparse Tensor Support](#8-sparse-tensor-support)
- [9. Tensor Iteration Patterns](#9-tensor-iteration-patterns)
- [10. Register and Shared Memory Tensors](#10-register-and-shared-memory-tensors)
- [11. Fragment Tensors for MMA](#11-fragment-tensors-for-mma)
- [12. Code Examples](#12-code-examples)

---

## 1. Overview

The CuTe tensor is the central data structure in CUTLASS's CuTe library. A tensor pairs a **layout** (which maps coordinates to offsets, as described in chapter 16) with an **engine** (which provides access to the underlying data storage). Together, the engine and layout form a complete tensor abstraction that supports:

- Owning and non-owning data access
- Hierarchical multi-dimensional indexing
- Subscript access with natural (hierarchical) coordinates
- Slicing, partitioning, and composition operations
- Register file, shared memory, and global memory tensors
- Fragment tensors for hardware MMA operations

The tensor abstraction is designed to be zero-cost: all layout computations are performed at compile time when possible, and the engine provides direct access to the underlying memory without indirection.

All tensor types live in the `cute::` namespace. The primary header is `include/cute/tensor.hpp`.

### Key Design Principles

1. **Separation of concerns**: The layout handles index-to-offset mapping; the engine handles data storage and access.
2. **Zero overhead**: Tensors compile down to raw pointer arithmetic with no runtime cost.
3. **Composability**: Tensors can be sliced, partitioned, and composed to build complex access patterns.
4. **Type safety**: The template system enforces correct dimensionality and type compatibility.

---

## 2. Tensor Structure: Engine + Layout

A CuTe tensor is defined as:

```cpp
template <class Engine, class Layout>
struct Tensor {
    // The engine provides data access
    using engine_type = Engine;
    // The layout provides coordinate-to-offset mapping
    using layout_type = Layout;

    // Underlying storage
    Engine engine_;
    Layout layout_;

    // Access the engine
    CUTE_HOST_DEVICE constexpr Engine& engine() { return engine_; }
    CUTE_HOST_DEVICE constexpr Engine const& engine() const { return engine_; }

    // Access the layout
    CUTE_HOST_DEVICE constexpr Layout& layout() { return layout_; }
    CUTE_HOST_DEVICE constexpr Layout const& layout() const { return layout_; }

    // Subscript operator: maps coordinate to reference
    template <class Coord>
    CUTE_HOST_DEVICE constexpr decltype(auto) operator()(Coord const& coord);

    // Subscript with integer (flat index)
    CUTE_HOST_DEVICE constexpr decltype(auto) operator[](int idx);
};
```

The tensor does not own data unless the engine is an owning type (like `ArrayEngine`). For non-owning engines, the tensor is just a view over external memory.

### Tensor Properties

```cpp
// Get the shape of a tensor (from its layout)
template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr auto shape(Tensor<Engine, Layout> const& tensor);

// Get the total number of elements
template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr auto size(Tensor<Engine, Layout> const& tensor);

// Get the rank (number of top-level dimensions)
template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr auto rank(Tensor<Engine, Layout> const& tensor);

// Get the stride
template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr auto stride(Tensor<Engine, Layout> const& tensor);
```

---

## 3. Engine Types

The engine type determines how data is stored and accessed. CuTe provides several engine types for different use cases.

### ArrayEngine

`ArrayEngine` wraps a fixed-size array and provides owning or non-owning access:

```cpp
template <class T, size_t N>
struct ArrayEngine {
    using value_type = T;
    using element_type = T;
    using reference = T&;
    using const_reference = T const&;
    static constexpr int elements = N;

    // Storage
    T storage_[N];

    // Access
    CUTE_HOST_DEVICE constexpr T& operator[](int i) { return storage_[i]; }
    CUTE_HOST_DEVICE constexpr T const& operator[](int i) const { return storage_[i]; }

    // Begin/end iterators
    CUTE_HOST_DEVICE constexpr T* begin() { return storage_; }
    CUTE_HOST_DEVICE constexpr T const* begin() const { return storage_; }
    CUTE_HOST_DEVICE constexpr T* end() { return storage_ + N; }
    CUTE_HOST_DEVICE constexpr T const* end() const { return storage_ + N; }

    // Data pointer
    CUTE_HOST_DEVICE constexpr T* data() { return storage_; }
    CUTE_HOST_DEVICE constexpr T const* data() const { return storage_; }
};
```

`ArrayEngine` is used for register tensors (where data is stored in registers):

```cpp
// Register tensor: 128 floats owned by the thread
auto reg_tensor = make_tensor<float, Layout<Shape<_128>, Stride<_1>>>();
// Uses ArrayEngine<float, 128> internally
```

### ViewEngine

`ViewEngine` provides a non-owning view over data accessed through a pointer:

```cpp
template <class Iterator>
struct ViewEngine {
    using iterator = Iterator;
    using value_type = remove_reference_t<decltype(*declval<Iterator>())>;
    using reference = decltype(*declval<Iterator>());
    using const_reference = decltype(*declval<Iterator const>());

    Iterator ptr_;

    // Access
    CUTE_HOST_DEVICE constexpr reference operator[](int i) { return ptr_[i]; }
    CUTE_HOST_DEVICE constexpr const_reference operator[](int i) const { return ptr_[i]; }

    // Begin/end
    CUTE_HOST_DEVICE constexpr Iterator begin() { return ptr_; }
    CUTE_HOST_DEVICE constexpr Iterator const begin() const { return ptr_; }
};
```

Used for global memory and shared memory tensors:

```cpp
// Global memory tensor: non-owning view of device memory
float* d_ptr;  // Device pointer
auto gmem_tensor = make_tensor(make_gmem_ptr(d_ptr), make_layout(make_shape(_128{}, _64{})));
// Uses ViewEngine<float*> internally
```

### ConstViewEngine

`ConstViewEngine` is like `ViewEngine` but provides const-only access:

```cpp
template <class Iterator>
struct ConstViewEngine {
    using iterator = Iterator;
    using value_type = remove_reference_t<decltype(*declval<Iterator>())>;
    using reference = value_type const&;

    Iterator ptr_;

    CUTE_HOST_DEVICE constexpr reference operator[](int i) const { return ptr_[i]; }
    CUTE_HOST_DEVICE constexpr Iterator begin() const { return ptr_; }
};
```

### Engine Type Aliases

```cpp
// Tag types for engine construction
struct GlobalMem {};   // Tag for global memory
struct SharedMem {};   // Tag for shared memory
struct RegisterMem {}; // Tag for register memory
```

### Engine Selection

The choice of engine is determined by the memory space:

| Memory Space | Engine Type | Ownership | Example |
|---|---|---|---|
| Register file | `ArrayEngine<T, N>` | Owning | Accumulator tensor |
| Shared memory | `ViewEngine<T*>` | Non-owning | SMEM tile tensor |
| Global memory | `ViewEngine<T*>` or `ViewEngine<GmemPtr<T>>` | Non-owning | Input/output tensor |

---

## 4. Tensor Creation

### make_tensor() - Factory Functions

The `make_tensor()` family of functions is the primary way to create tensors.

#### From pointer and layout (non-owning view):

```cpp
template <class Pointer, class Layout>
CUTE_HOST_DEVICE constexpr auto make_tensor(Pointer pointer, Layout const& layout);
```

```cpp
// Global memory tensor from raw pointer
float* d_data;
auto layout = make_layout(make_shape(_128{}, _64{}), make_stride(_64{}, _1{}));
auto tensor = make_tensor(d_data, layout);
```

#### From pointer and shape (compact stride):

```cpp
template <class Pointer, class Shape>
CUTE_HOST_DEVICE constexpr auto make_tensor(Pointer pointer, Shape const& shape);
```

```cpp
// Auto-computed compact stride
auto tensor = make_tensor(d_data, make_shape(_128{}, _64{}));
// Stride defaults to (1, 128) for compact layout
```

#### Owning tensor (register allocation):

```cpp
template <class T, class Layout>
CUTE_HOST_DEVICE constexpr auto make_tensor(Layout const& layout);
```

```cpp
// Allocate register tensor with 128 elements
auto reg_tensor = make_tensor<float>(make_layout(_128{}));
// Uses ArrayEngine<float, 128>

// Allocate 2D register tensor
auto reg_2d = make_tensor<float>(make_layout(make_shape(_16{}, _8{})));
// 128 floats in registers with layout (16, 8), stride (1, 16)
```

#### Shared memory tensor:

```cpp
template <class T, class Layout>
CUTE_HOST_DEVICE auto make_tensor(make_gmem_ptr(T* ptr), Layout const& layout);

template <class T, class Layout>
CUTE_HOST_DEVICE auto make_tensor(make_smem_ptr(T* ptr), Layout const& layout);
```

```cpp
// Shared memory tensor
extern __shared__ float smem[];
auto smem_tensor = make_tensor(make_smem_ptr(smem), smem_layout);
```

### make_tensor() with pointer wrappers

CuTe uses pointer wrappers to encode memory space information:

```cpp
// Global memory pointer wrapper
auto gmem_ptr = make_gmem_ptr(d_data);
auto gmem_tensor = make_tensor(gmem_ptr, layout);

// Shared memory pointer wrapper
auto smem_ptr = make_smem_ptr(smem_data);
auto smem_tensor = make_tensor(smem_ptr, layout);

// Register data (array)
auto reg_tensor = make_tensor<float>(layout);  // Allocates ArrayEngine
```

### make_fragment_like()

Creates a tensor with the same layout as an existing tensor's fragment for a given MMA atom:

```cpp
template <class Engine, class Layout, class MMA_Op>
CUTE_HOST_DEVICE constexpr auto make_fragment_like(Tensor<Engine, Layout> const& tensor, MMA_Op const& mma_op);
```

```cpp
// Create accumulator fragment matching the MMA atom's C layout
auto tCrC = make_fragment_like(gC, tiled_mma);
// tCrC is a register tensor with the layout expected by the MMA for C
```

### make_tensor_like()

Creates a tensor with the same layout as another tensor but using a different engine:

```cpp
template <class T, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr auto make_tensor_like(Tensor<Engine, Layout> const& tensor);
```

```cpp
// Create a float accumulator tensor matching the shape of a half tensor
auto acc_tensor = make_tensor_like<float>(half_tensor);
```

### Construction from shared memory allocation

For managed shared memory allocation, CuTe provides helpers:

```cpp
// Allocate and create a shared memory tensor
auto [smem_ptr, smem_layout] = make_smem_tensor<T>(shape);
// or using the SM90+ allocator:
auto [ptr_A, layout_A] = SM90_TMA_ALLOC::allocate(smem_allocator, shape_A);
```

---

## 5. Tensor Operations

### Subscript Operator with Coordinates

The subscript operator `operator()` maps a coordinate to a reference:

```cpp
// 2D tensor access
auto tensor = make_tensor(ptr, make_layout(make_shape(_128{}, _64{})));
float& val = tensor(make_coord(32, 16));  // Reference to element at (32, 16)

// Flat index access
float& val2 = tensor(32 * 64 + 16);       // Same element via flat index

// Hierarchical coordinate access
auto hier_tensor = make_tensor(ptr, make_layout(make_shape(make_shape(_4{}, _8{}), _16{})));
float& val3 = hier_tensor(make_coord(make_coord(2, 3), 7));
```

### Slice Operations

Slicing creates a sub-tensor by fixing some dimensions:

```cpp
// Slice a 3D tensor along the first dimension
auto tensor_3d = make_tensor(ptr, make_layout(make_shape(_32{}, _64{}, _128{})));

// Fix first dimension to 5: gives a 2D view (64, 128)
auto slice_2d = tensor_3d(5, _, _);

// Fix first two dimensions: gives a 1D view (128)
auto slice_1d = tensor_3d(5, 10, _);

// Range slice: keep a subrange
auto sub_tensor = tensor_3d(make_coord(8, _, 16), _, _);
```

The underscore `_` is a placeholder meaning "keep this dimension".

### Partition Operations

Partitioning distributes tensor elements across threads for cooperative operations.

#### tiled_copy_partition

Partitions a tensor for a tiled copy operation:

```cpp
template <class TiledCopy, class Engine, class Layout, class ThrCoord>
CUTE_HOST_DEVICE constexpr auto
tiled_copy_partition(TiledCopy const& tiled_copy,
                     Tensor<Engine, Layout> const& tensor,
                     ThrCoord const& thr_coord);
```

```cpp
// Partition a source tensor for copying
auto tiled_copy = make_tiled_copy(copy_atom, thr_layout, val_layout);
auto thr_coord = canonicalize_thr_idx(thread_id);
auto tSrc = tiled_copy_partition(tiled_copy, src_tensor, thr_coord);
// tSrc is the thread's partition of src_tensor
```

#### tiled_mma_partition

Partitions a tensor for a tiled MMA operation:

```cpp
template <class TiledMMA, class Engine, class Layout, class ThrCoord>
CUTE_HOST_DEVICE constexpr auto
tiled_mma_partition_A(TiledMMA const& tiled_mma,
                      Tensor<Engine, Layout> const& tensor_A,
                      ThrCoord const& thr_coord);

template <class TiledMMA, class Engine, class Layout, class ThrCoord>
CUTE_HOST_DEVICE constexpr auto
tiled_mma_partition_B(TiledMMA const& tiled_mma,
                      Tensor<Engine, Layout> const& tensor_B,
                      ThrCoord const& thr_coord);

template <class TiledMMA, class Engine, class Layout, class ThrCoord>
CUTE_HOST_DEVICE constexpr auto
tiled_mma_partition_C(TiledMMA const& tiled_mma,
                      Tensor<Engine, Layout> const& tensor_C,
                      ThrCoord const& thr_coord);
```

```cpp
// Partition A, B, C for the MMA
auto tPgA = tiled_mma_partition_A(tiled_mma, gA, thr_coord);  // A partition (in GMEM/SMEM)
auto tPgB = tiled_mma_partition_B(tiled_mma, gB, thr_coord);  // B partition
auto tCrC = tiled_mma_partition_C(tiled_mma, gC, thr_coord);  // C partition (accumulator)
```

#### get_slice()

An alternative interface using `get_slice()` on the tiled operation:

```cpp
// Get a "slice" object that represents a single thread's view
auto thr_mma = tiled_mma.get_slice(thread_id);

// Partition using the slice
auto tCrA = thr_mma.partition_A(gA);   // Thread's partition of A
auto tCrB = thr_mma.partition_B(gB);   // Thread's partition of B
auto tCrC = thr_mma.partition_C(gC);   // Thread's partition of C (accumulator)

// Get the MMA fragment layouts
auto tCrA_layout = thr_mma.partition_A(gA.layout());
```

### Composition with Layouts

Tensors can be composed with layouts to remap their access patterns:

```cpp
template <class Engine, class Layout, class OtherLayout>
CUTE_HOST_DEVICE constexpr auto
composition(Tensor<Engine, Layout> const& tensor, OtherLayout const& other_layout);
```

```cpp
// Original tensor
auto gmem_tensor = make_tensor(ptr, make_layout(make_shape(_128{}, _64{})));

// Compose with a tile layout to get a tiled view
auto tile_layout = make_layout(make_shape(_32{}, _16{}));
auto tiled_view = composition(gmem_tensor, tile_layout);
// tiled_view has shape (32, 16) within the original (128, 64) space
```

---

## 6. Tensor Concepts and Type Traits

### is_tensor

Checks if a type is a CuTe tensor:

```cpp
template <class T>
struct is_tensor : cute::false_type {};

template <class Engine, class Layout>
struct is_tensor<Tensor<Engine, Layout>> : cute::true_type {};

template <class T>
constexpr bool is_tensor_v = is_tensor<T>::value;
```

```cpp
static_assert(is_tensor_v<decltype(my_tensor)>);
static_assert(!is_tensor_v<float>);
```

### tensor_rank_v

Gets the rank of a tensor type at compile time:

```cpp
template <class Tensor>
constexpr int tensor_rank_v = rank_v<typename Tensor::layout_type>;
```

```cpp
auto tensor = make_tensor(ptr, make_layout(make_shape(_128{}, _64{}, _32{})));
static_assert(tensor_rank_v<decltype(tensor)> == 3);
```

### tensor_size_v

Gets the total size of a tensor type:

```cpp
template <class Tensor>
constexpr int tensor_size_v = size_v<typename Tensor::layout_type>;
```

```cpp
static_assert(tensor_size_v<decltype(tensor)> == 128 * 64 * 32);
```

### is_gmem, is_smem, is_rmem

Checks the memory space of a tensor:

```cpp
template <class Tensor>
struct is_gmem;   // True if tensor is in global memory

template <class Tensor>
struct is_smem;   // True if tensor is in shared memory

template <class Tensor>
struct is_rmem;   // True if tensor is in register memory
```

### Other Type Traits

```cpp
// Get the element type of a tensor
template <class Tensor>
using tensor_element_t = typename Tensor::engine_type::value_type;

// Check if tensor has static layout
template <class Tensor>
struct has_static_layout;

// Check if tensor data is const-qualified
template <class Tensor>
struct is_const_tensor;
```

---

## 7. Tensor Transformations

### recast<DstType>()

Reinterprets the tensor data as a different type, similar to `reinterpret_cast`:

```cpp
template <class DstType, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr auto
recast(Tensor<Engine, Layout> const& tensor);
```

```cpp
// Recast a half tensor to a uint32_t tensor (packing 2 halfs per uint32)
auto half_tensor = make_tensor(ptr_half, make_layout(_128{}));
auto uint_tensor = recast<uint32_t>(half_tensor);
// Now has 64 uint32_t elements (128 halfs / 2)

// Recast accumulator from float to half for store
auto acc_tensor = make_tensor<float>(acc_layout);
auto half_acc = recast<half_t>(acc_tensor);
```

The recast adjusts the layout shape to account for the type size change:

```
new_size = old_size * sizeof(OldType) / sizeof(NewType)
```

### reshape()

Changes the shape of a tensor while keeping the same data:

```cpp
template <class NewShape, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr auto
reshape(Tensor<Engine, Layout> const& tensor, NewShape const& new_shape);
```

```cpp
auto tensor_2d = make_tensor(ptr, make_layout(make_shape(_128{}, _64{})));
auto tensor_1d = reshape(tensor_2d, _8192{});           // Flatten to 1D
auto tensor_3d = reshape(tensor_2d, make_shape(_32{}, _64{}, _4{}));  // Reshape to 3D
```

The total number of elements must remain the same after reshaping.

### zip()

Zips multiple tensors together along their innermost dimensions:

```cpp
template <class... Tensors>
CUTE_HOST_DEVICE constexpr auto
zip(Tensors&&... tensors);
```

```cpp
auto tensor_a = make_tensor(ptr_a, make_layout(_128{}));
auto tensor_b = make_tensor(ptr_b, make_layout(_128{}));
auto zipped = zip(tensor_a, tensor_b);
// zipped has shape (128, 2) -- the second dimension indexes the two tensors
```

### group()

Groups tensor dimensions:

```cpp
template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr auto
group(Tensor<Engine, Layout> const& tensor);
```

### filter()

Filters out dimensions of size 1 or with stride 0:

```cpp
template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr auto
filter(Tensor<Engine, Layout> const& tensor);
```

### as_position_independent()

Creates a position-independent tensor view that only tracks offsets:

```cpp
template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr auto
as_position_independent(Tensor<Engine, Layout> const& tensor);
```

---

## 8. Sparse Tensor Support

CuTe provides support for sparse tensor operations through the `sparse_elem` wrapper.

### sparse_elem<S, T>

A `sparse_elem` wraps a tensor element with associated metadata for 2:4 structured sparsity:

```cpp
template <int S, class T>
struct sparse_elem {
    static constexpr int sparse = S;  // Sparsity pattern (e.g., 2 means 2:4)
    T value_;

    CUTE_HOST_DEVICE constexpr T& value() { return value_; }
    CUTE_HOST_DEVICE constexpr T const& value() const { return value_; }
};
```

### Sparse Tensor Creation

```cpp
// Create a sparse tensor for 2:4 structured sparsity
// Operand B can be sparse with 2 non-zero elements per 4-element group
auto sparse_b = make_tensor(make_gmem_ptr(ptr_b), sparse_layout);
// The layout accounts for the 2:4 sparsity pattern
```

### Sparse Tensor Layout

For 2:4 structured sparsity, the layout compresses each group of 4 elements to 2:

```cpp
// Dense B shape: (K, N)
// Sparse B shape: (K/2, N) with metadata indicating which 2 of 4 are non-zero
// The layout includes a stride pattern that maps to the compressed storage
```

---

## 9. Tensor Iteration Patterns

CuTe tensors support several iteration patterns for traversing elements.

### Sequential Iteration

```cpp
// Iterate over all elements of a tensor
auto tensor = make_tensor(ptr, make_layout(make_shape(_128{}, _64{})));

// Flat iteration
for (int i = 0; i < size(tensor); ++i) {
    tensor(i) = 0.0f;
}

// Coordinate-based iteration using for_each
cute::for_each(tensor, [](auto& elem) {
    elem = 0.0f;
});
```

### Hierarchical Iteration with for_each

```cpp
// Iterate over all coordinates in a tensor's shape
cute::for_each_idx(tensor, [](auto& elem, auto coord) {
    // coord is the hierarchical coordinate
    // elem is the reference to the element
    printf("coord: "); print(coord); printf(" value: %f\n", (float)elem);
});
```

### Tile-based Iteration

```cpp
// Iterate over tiles of a tensor
auto tensor = make_tensor(ptr, make_layout(make_shape(_128{}, _64{})));
auto tile_shape = make_shape(_32{}, _16{});

// Create a tiled view and iterate
auto tiled = zipped_divide(tensor.layout(), tile_shape);
for (int tile_m = 0; tile_m < size<0>(tiled); ++tile_m) {
    for (int tile_n = 0; tile_n < size<1>(tiled); ++tile_n) {
        auto tile = tensor(make_coord(tile_m, tile_n), _);
        // Process tile
    }
}
```

### SIMD-friendly Iteration

For vectorized access patterns:

```cpp
// Access elements in vector-width groups
auto tensor = make_tensor(ptr, make_layout(_128{}));
constexpr int vec_size = 4;

for (int i = 0; i < size(tensor); i += vec_size) {
    // The engine may auto-vectorize these accesses
    float vec[vec_size] = { tensor(i), tensor(i+1), tensor(i+2), tensor(i+3) };
}
```

---

## 10. Register and Shared Memory Tensors

### Register Tensors

Register tensors use `ArrayEngine` and store data in GPU registers. They are created by `make_tensor<T>(layout)`:

```cpp
// Allocate a register tensor for MMA accumulator
// Shape matches the MMA atom's output: 16x8 for a 16x8x16 MMA
auto acc = make_tensor<float>(make_layout(make_shape(_16{}, _8{})));
// acc uses ArrayEngine<float, 128>
// Each thread has 128 float registers for the accumulator

// Access register tensor elements
acc(make_coord(0, 0)) = 0.0f;
acc(make_coord(15, 7)) = result;
```

Register tensors are:
- **Owning**: The data is stored in the `ArrayEngine` member.
- **Fast**: Access to register data is as fast as any register variable.
- **Small**: Limited by register file size (typically 255 registers per thread on modern GPUs).

### Shared Memory Tensors

Shared memory tensors use `ViewEngine` with a shared memory pointer:

```cpp
// Declare shared memory
extern __shared__ char smem_buf[];

// Create shared memory tensor
float* smem_ptr = reinterpret_cast<float*>(smem_buf);
auto smem_tensor = make_tensor(
    make_smem_ptr(smem_ptr),
    make_layout(make_shape(_128{}, _32{}), make_stride(_1{}, _128{}))
);

// Access shared memory tensor
smem_tensor(make_coord(threadIdx.x, 0)) = gmem_data;
__syncwarp();  // Or appropriate synchronization
float val = smem_tensor(make_coord(threadIdx.y, threadIdx.z));
```

Shared memory tensors are:
- **Non-owning**: They view externally managed shared memory.
- **Shared**: All threads in a block can access the same shared memory tensor.
- **Fast**: Shared memory has much lower latency than global memory.
- **Banked**: 32 banks; swizzled layouts help avoid bank conflicts.

### Global Memory Tensors

Global memory tensors use `ViewEngine` with a device pointer:

```cpp
// Create a global memory tensor
float* d_data;  // Allocated with cudaMalloc
auto gmem_tensor = make_tensor(
    make_gmem_ptr(d_data),
    make_layout(make_shape(M, N), make_stride(N, 1))
);

// Access (typically done in cooperative fashion across a thread block)
float val = gmem_tensor(make_coord(row, col));
```

Global memory tensors are:
- **Non-owning**: They view externally allocated global memory.
- **Slow**: Global memory has high latency (hundreds of cycles).
- **Coalesced**: Proper layout ensures coalesced access by adjacent threads.

---

## 11. Fragment Tensors for MMA

Fragment tensors are specialized register tensors that match the data layout expected by hardware MMA instructions.

### Fragment for Operand A

```cpp
// Partition A matrix for the MMA
auto thr_mma = tiled_mma.get_slice(thread_id);
auto tCrA = thr_mma.partition_A(gA);

// tCrA is a register tensor with layout matching the MMA's A fragment
// Its shape and stride are determined by the MMA atom's A traits
```

### Fragment for Operand B

```cpp
auto tCrB = thr_mma.partition_B(gB);
// Layout matches the MMA atom's B fragment
```

### Fragment for Accumulator C

```cpp
auto tCrC = thr_mma.partition_C(gC);
// Accumulator fragment layout determined by MMA atom's C traits
```

### Fragment Allocation

```cpp
// Allocate an accumulator fragment
auto tCrC = make_tensor<float>(thr_mma.partition_C(gC).layout());
// or equivalently:
auto tCrC = make_fragment_like(gC, tiled_mma);
```

### Fragment Access Patterns

The layout of a fragment depends on the MMA atom:

```cpp
// For SM80 16x8x16 FP16 MMA:
// A fragment: 4 registers per thread (2 rows x 2 cols of uint32_t packed FP16)
// B fragment: 2 registers per thread
// C fragment: 4 registers per thread (2 rows x 2 cols of float)

// Access pattern matches the MMA instruction's register assignment
for (int k = 0; k < K / k_tile; ++k) {
    // Load A and B fragments
    copy(tiled_copy, tAgA(_, _, k), tAsA);
    copy(tiled_copy, tBgB(_, _, k), tBsB);

    // MMA accumulate
    gemm(tiled_mma, tCrA, tCrB, tCrC);
}
```

---

## 12. Code Examples

### Example 1: Create and Use a Global Memory Tensor

```cpp
#include "cute/tensor.hpp"
using namespace cute;

__global__ void kernel(float* d_data, int M, int N) {
    // Create a global memory tensor
    auto gmem = make_tensor(
        make_gmem_ptr(d_data),
        make_layout(make_shape(M, N), make_stride(N, 1))
    );

    // Access elements
    int row = threadIdx.x;
    int col = threadIdx.y;
    if (row < M && col < N) {
        gmem(make_coord(row, col)) = row * N + col;
    }
}
```

### Example 2: Shared Memory Tensor with Swizzle

```cpp
#include "cute/tensor.hpp"
using namespace cute;

__global__ void kernel(half_t const* d_A, half_t* d_C, int M, int N, int K) {
    extern __shared__ char smem_buf[];

    // Shared memory layout with swizzle for bank conflict avoidance
    auto smem_layout = composition(
        Swizzle<3, 4, 3>{},
        make_layout(make_shape(_128{}, _32{}), make_stride(_1{}, _128{}))
    );

    // Create shared memory tensor
    auto sA = make_tensor(
        make_smem_ptr(reinterpret_cast<half_t*>(smem_buf)),
        smem_layout
    );

    // Create register tensor for accumulator
    auto rC = make_tensor<float>(make_layout(make_shape(_128{}, _32{})));

    // Load from global to shared
    // ... (copy operation)

    // Use shared memory tensor
    __syncthreads();
    float val = sA(make_coord(threadIdx.x, 0));
}
```

### Example 3: Tensor Partitioning for Tiled MMA

```cpp
#include "cute/tensor.hpp"
#include "cute/arch/mma_sm80.hpp"
using namespace cute;

__global__ void gemm_kernel(
    half_t const* d_A, half_t const* d_B, float* d_C,
    int M, int N, int K
) {
    // Define the tiled MMA
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{},
        Layout<Shape<_2, _2, _1>>{}  // 2x2 tiling
    );

    // Create global memory tensors
    auto gA = make_tensor(make_gmem_ptr(d_A), make_layout(make_shape(M, K), make_stride(K, 1)));
    auto gB = make_tensor(make_gmem_ptr(d_B), make_layout(make_shape(K, N), make_stride(N, 1)));
    auto gC = make_tensor(make_gmem_ptr(d_C), make_layout(make_shape(M, N), make_stride(N, 1)));

    // Get this thread's slice of the MMA
    int thread_id = threadIdx.x;
    auto thr_mma = tiled_mma.get_slice(thread_id);

    // Partition A, B, C for this thread
    auto tCrA = thr_mma.partition_A(gA);  // Thread's A fragment view
    auto tCrB = thr_mma.partition_B(gB);  // Thread's B fragment view
    auto tCrC = thr_mma.partition_C(gC);  // Thread's C accumulator

    // Allocate actual register tensors for the fragments
    auto rA = make_tensor<half_t>(tCrA.layout());
    auto rB = make_tensor<half_t>(tCrB.layout());
    auto rC = make_tensor<float>(tCrC.layout());

    // Clear accumulator
    clear(rC);

    // Main GEMM loop
    for (int k = 0; k < K; k += size<2>(tCrA)) {
        // Load A and B
        copy(tCrA(_, _, k), rA);
        copy(tCrB(_, _, k), rB);

        // Perform MMA
        gemm(tiled_mma, rA, rB, rC);
    }

    // Store result
    copy(rC, tCrC);
}
```

### Example 4: Tensor Recast for Mixed Precision

```cpp
#include "cute/tensor.hpp"
using namespace cute;

__global__ void mixed_precision_kernel(float* d_out, half_t* d_acc, int N) {
    // Create a float tensor view
    auto float_tensor = make_tensor(
        make_gmem_ptr(d_out),
        make_layout(make_shape(N))
    );

    // Recast to half_t view for storage
    auto half_view = recast<half_t>(float_tensor);
    // Now has N/2 half_t elements (2 halfs per float)

    // Or the other way: recast accumulator from half to float for computation
    auto acc_half = make_tensor(make_gmem_ptr(d_acc), make_layout(_128{}));
    auto acc_float = recast<float>(acc_half);
    // Now has 64 float elements (128 halfs / 2)

    // Perform float computation
    for (int i = 0; i < size(acc_float); ++i) {
        acc_float(i) *= 2.0f;
    }
}
```

### Example 5: Tensor Slicing and Subtensors

```cpp
#include "cute/tensor.hpp"
using namespace cute;

__global__ void slice_kernel(float* d_data, int M, int N, int K) {
    // 3D tensor (M, N, K)
    auto tensor_3d = make_tensor(
        make_gmem_ptr(d_data),
        make_layout(make_shape(M, N, K), make_stride(N*K, K, 1))
    );

    // Slice: fix M dimension, get (N, K) subtensor
    auto slice_nk = tensor_3d(5, _, _);

    // Slice: fix M and N, get (K) subtensor
    auto slice_k = tensor_3d(5, 10, _);

    // Range slice: get first 32 elements of K
    auto sub_k = tensor_3d(5, 10, make_coord(_, 32));
    // sub_k has shape (32) covering K indices [0, 32)

    // Access elements
    float val = slice_nk(make_coord(10, 20));  // tensor_3d(5, 10, 20)
}
```

### Example 6: Tensor Type Traits

```cpp
#include "cute/tensor.hpp"
using namespace cute;

void type_traits_example() {
    // Create tensors of different types
    auto gmem = make_tensor(make_gmem_ptr((float*)nullptr), make_layout(_128{}));
    auto reg = make_tensor<float>(make_layout(_128{}));

    // Check type traits
    static_assert(is_tensor_v<decltype(gmem)>);
    static_assert(is_tensor_v<decltype(reg)>);
    static_assert(!is_tensor_v<float>);

    static_assert(tensor_rank_v<decltype(gmem)> == 1);
    static_assert(tensor_size_v<decltype(gmem)> == 128);

    // Element type
    using elem_t = tensor_element_t<decltype(gmem)>;
    static_assert(std::is_same_v<elem_t, float>);
}
```

---

## Summary

The CuTe tensor system provides a powerful and flexible abstraction for GPU data:

1. **Engine + Layout decomposition** separates data storage from index mapping.
2. **Multiple engine types** support register, shared memory, and global memory tensors.
3. **Partitioning** distributes tensor elements across threads for cooperative operations.
4. **Transformations** like recast, reshape, and zip enable type reinterpretation and layout changes.
5. **Fragment tensors** match hardware MMA instruction layouts for zero-overhead integration.
6. **Type traits** provide compile-time introspection for correctness and optimization.
