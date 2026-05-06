# XLA Reference - Chapter 2: Shapes and Layout

This reference provides comprehensive documentation about XLA's shape system, layout conventions, memory representation, and related utilities. Shapes and layouts are fundamental to XLA's operation -- they define the type, dimensionality, memory ordering, and storage properties of every tensor in the compilation graph. Understanding shapes and layouts is essential for writing efficient XLA programs, debugging layout-related issues, and implementing custom XLA operations.

---

## 2.1 Structure of an XLA Operation

Every XLA operation (also called an HLO instruction) is characterized by four core attributes:

| Attribute | Description |
|-----------|-------------|
| **Op name** (opcode) | The operation identifier, such as `add`, `dot`, `convolution`, `fusion`. This determines what computation the instruction performs. |
| **Shape** | Describes the output of the operation -- its element type, dimensions (rank and size of each dimension), and layout. Also called the "result shape." |
| **Operands (arguments)** | Zero or more input HLO instructions whose outputs feed into this operation. Each operand has its own shape. |
| **Operation-specific attributes** | Additional parameters that configure the operation. For example, a `dot` operation has `lhs_contracting_dims` and `rhs_contracting_dims`; a `convolution` has window attributes, padding, and stride information. |

In textual HLO representation, an instruction is written as:

```
result_name = shape opcode(operand1, operand2, ...), attribute1=value1, attribute2=value2
```

Example:

```
dot.0 = f32[128,512] dot(parameter.0, parameter.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
```

Here:
- `dot.0` is the instruction name.
- `f32[128,512]` is the shape (32-bit float, 2D array with dimensions 128 and 512).
- `dot` is the opcode.
- `parameter.0` and `parameter.1` are operands.
- `lhs_contracting_dims={1}` and `rhs_contracting_dims={0}` are operation-specific attributes.

---

## 2.2 ShapeProto and Shape Representation

### 2.2.1 ShapeProto Definition

In XLA's protocol buffer representation, shapes are described by the `ShapeProto` message. The core definition is:

```protobuf
message ShapeProto {
  // Element type (e.g., F32, S32, U8, BOOL, etc.)
  PrimitiveType element_type = 1;

  // For array shapes: the size of each dimension.
  repeated int64 dimensions = 2;

  // For tuple shapes: the sub-shapes.
  repeated ShapeProto tuple_shapes = 3;

  // Layout information for array shapes.
  optional LayoutProto layout = 4;

  // Whether this is a dynamic dimension.
  repeated bool is_dynamic_dimension = 5;
}
```

### 2.2.2 Shape Types

XLA shapes fall into several categories:

| Shape Type | Description | Example |
|------------|-------------|---------|
| **Array shape** | An N-dimensional array of a primitive type | `f32[128, 256]` -- a 2D array of 32-bit floats |
| **Tuple shape** | A heterogeneous collection of sub-shapes | `(f32[128], s32[])` -- a tuple of a 2D array and a scalar |
| **Opaque shape** | A shape with hidden internal structure (used for custom calls) | `opaque()` |
| **Token shape** | A shape used for enforcing execution ordering without carrying data | `token` |
| **Scalar shape** | A zero-dimensional array | `f32[]` -- a single 32-bit float |

### 2.2.3 Primitive Types

XLA supports the following primitive element types:

| Type | Description | Size (bytes) |
|------|-------------|--------------|
| `PRED` | Boolean (predicate) | 1 |
| `S8` | Signed 8-bit integer | 1 |
| `S16` | Signed 16-bit integer | 2 |
| `S32` | Signed 32-bit integer | 4 |
| `S64` | Signed 64-bit integer | 8 |
| `U8` | Unsigned 8-bit integer | 1 |
| `U16` | Unsigned 16-bit integer | 2 |
| `U32` | Unsigned 32-bit integer | 4 |
| `U64` | Unsigned 64-bit integer | 8 |
| `F16` | 16-bit floating point (IEEE 754 half) | 2 |
| `F32` | 32-bit floating point (IEEE 754 single) | 4 |
| `F64` | 64-bit floating point (IEEE 754 double) | 8 |
| `BF16` | 16-bit brain floating point (bfloat16) | 2 |
| `C64` | 64-bit complex (two F32) | 8 |
| `C128` | 128-bit complex (two F64) | 16 |
| `F8E4M3FN` | 8-bit float (E4M3 format, NaN on overflow) | 1 |
| `F8E4M3FNUZ` | 8-bit float (E4M3 format, unsigned zero) | 1 |
| `F8E5M2` | 8-bit float (E5M2 format) | 1 |
| `F8E5M2FNUZ` | 8-bit float (E5M2 format, unsigned zero) | 1 |
| `TUPLE` | Tuple type (container for sub-shapes) | N/A |
| `OPAQUE_TYPE` | Opaque type | N/A |
| `TOKEN` | Token type (ordering only) | 0 |

### 2.2.4 Dynamic Dimensions

XLA supports dynamic dimensions where the size of a dimension is not known at compile time. A shape can mark specific dimensions as dynamic:

```
f32[<=128, 256]  // First dimension is dynamically sized up to 128
```

Dynamic dimensions enable:
- Variable batch sizes without recompilation
- Variable sequence lengths in NLP models
- Dynamic shapes in data-dependent control flow

The `is_dynamic_dimension` field in `ShapeProto` tracks which dimensions are dynamic. Dynamic dimensions have an upper bound (the value in `dimensions`) and an actual runtime value that is less than or equal to this bound.

---

## 2.3 Dimension Numbering Conventions

### 2.3.1 N-Dimensional Array Dimensions

XLA represents arrays as N-dimensional structures. Each dimension has an index numbered from `0` to `rank-1`, where `rank` is the number of dimensions.

For an array shape `T[d0, d1, ..., dN-1]`:
- The rank is `N` (the number of dimensions).
- `d0` is dimension 0, `d1` is dimension 1, and so on.
- The total number of elements is `d0 * d1 * ... * dN-1`.

### 2.3.2 Dimension Naming Conventions

XLA uses specific naming conventions for dimensions in commonly used array ranks. These names originate from matrix and tensor conventions:

#### 2D Arrays: `T[y, x]`

For a 2D array (matrix), the dimensions are named:

| Dimension Index | Name | Description |
|-----------------|------|-------------|
| 0 | **y** | Row dimension (vertical) |
| 1 | **x** | Column dimension (horizontal) |

Example: `f32[3, 4]` is a 3-row by 4-column matrix.

```
     x=0  x=1  x=2  x=3
y=0  [    a    b    c    d  ]
y=1  [    e    f    g    h  ]
y=2  [    i    j    k    l  ]
```

#### 3D Arrays: `T[z, y, x]`

For a 3D array, the dimensions are named:

| Dimension Index | Name | Description |
|-----------------|------|-------------|
| 0 | **z** | Depth dimension |
| 1 | **y** | Row dimension (vertical) |
| 2 | **x** | Column dimension (horizontal) |

Example: `f32[2, 3, 4]` is a 2-depth by 3-row by 4-column 3D tensor.

#### 4D Arrays: `T[n, z, y, x]` or `T[p, z, y, x]`

For 4D arrays (common in convolutions), the dimensions are named:

| Dimension Index | Name | Common ML Name | Description |
|-----------------|------|---------------|-------------|
| 0 | **n** or **p** | batch | Batch / planar dimension |
| 1 | **z** | channels / features | Channel / depth dimension |
| 2 | **y** | height | Spatial row dimension |
| 3 | **x** | width | Spatial column dimension |

Example: `f32[32, 3, 224, 224]` is a batch of 32 images with 3 color channels at 224x224 resolution (NHWC or NCHW format depending on layout).

#### General N-Dimensional Arrays

For higher-dimensional arrays, dimensions are named `d0, d1, d2, ...` using their numeric indices. The numbering convention extends naturally: the first listed dimension is dimension 0, and each subsequent dimension increments the index.

### 2.3.3 Dimension Ordering in Operations

Many XLA operations reference dimensions by index. For example:

- **Reduce**: `reduce(f32[3, 4], f32[], dimensions={1})` reduces dimension 1 (the `x` dimension), producing `f32[3]`.
- **Transpose**: `transpose(f32[3, 4], {1, 0})` swaps dimensions 0 and 1, producing `f32[4, 3]`.
- **Dot**: `dot(f32[M, K], f32[K, N]), lhs_contracting_dims={1}, rhs_contracting_dims={0}` contracts dimension 1 of the LHS with dimension 0 of the RHS.

---

## 2.4 Layout

Layout is one of the most important concepts in XLA. It defines how multi-dimensional array elements are mapped to linear (1D) memory addresses. The same logical array can be stored in memory in many different orderings, and choosing the right layout has a profound impact on performance.

### 2.4.1 LayoutProto Definition

```protobuf
message LayoutProto {
  // Minor-to-major dimension ordering.
  repeated int64 minor_to_major = 1;

  // Optional: tiles for tiled layouts.
  optional TileProto tiles = 3;

  // Optional: tail padding alignment.
  optional int64 tail_padding_alignment_in_elements = 2;

  // Memory space identifier.
  optional int64 memory_space = 4;

  // Physical shape for physical layouts.
  // (used when logical and physical shapes differ due to tiling)
}
```

### 2.4.2 Minor-to-Major Dimension Ordering

The fundamental concept in XLA layout is the **minor-to-major** ordering. This is a permutation of the dimension indices that specifies the order in which dimensions change as we step through the array's memory.

- The **minor** dimension (last in the list) is the dimension whose index changes fastest -- adjacent elements along this dimension are contiguous in memory.
- The **major** dimension (first in the list) is the dimension whose index changes slowest -- stepping along this dimension requires the largest memory stride.

For a 2D array `T[y, x]` with shape `f32[3, 4]`:

- **Column-major layout** (Fortran order): `minor_to_major = {1, 0}` means x (dimension 1) is minor (changes fastest) and y (dimension 0) is major (changes slowest).
- **Row-major layout** (C order): `minor_to_major = {0, 1}` means y (dimension 0) is minor (changes fastest) and x (dimension 1) is major (changes slowest).

### 2.4.3 Column-Major Layout

In column-major layout (also called Fortran order), the **rightmost logical dimension** (dimension 1, the `x` dimension for 2D) is the **major** dimension (changes slowest), and the **leftmost logical dimension** (dimension 0, the `y` dimension for 2D) is the **minor** dimension (changes fastest).

For `f32[3, 4]` with column-major layout `minor_to_major = {0, 1}`:

```
Memory offset = y * 1 + x * 3

Memory layout:
Offset 0: T[0,0]    Offset 3: T[0,1]    Offset 6: T[0,2]    Offset 9: T[0,3]
Offset 1: T[1,0]    Offset 4: T[1,1]    Offset 7: T[1,2]    Offset 10: T[1,3]
Offset 2: T[2,0]    Offset 5: T[2,1]    Offset 8: T[2,2]    Offset 11: T[2,3]
```

Notice that elements within the same column (same `x`, varying `y`) are contiguous in memory.

**Column-major is the default layout in XLA.** When no explicit layout is specified, XLA defaults to column-major (minor-to-major = `{0, 1, 2, ..., N-1}`).

### 2.4.4 Row-Major Layout

In row-major layout (also called C order), the **leftmost logical dimension** (dimension 0, the `y` dimension for 2D) is the **major** dimension (changes slowest), and the **rightmost logical dimension** (dimension 1, the `x` dimension for 2D) is the **minor** dimension (changes fastest).

For `f32[3, 4]` with row-major layout `minor_to_major = {1, 0}`:

```
Memory offset = y * 4 + x * 1

Memory layout:
Offset 0: T[0,0]    Offset 1: T[0,1]    Offset 2: T[0,2]    Offset 3: T[0,3]
Offset 4: T[1,0]    Offset 5: T[1,1]    Offset 6: T[1,2]    Offset 7: T[1,3]
Offset 8: T[2,0]    Offset 9: T[2,1]    Offset 10: T[2,2]   Offset 11: T[2,3]
```

Notice that elements within the same row (same `y`, varying `x`) are contiguous in memory.

### 2.4.5 General Minor-to-Major Ordering

For higher-dimensional arrays, the minor-to-major ordering can be any permutation. For a 4D array `T[n, z, y, x]` with shape `f32[32, 3, 224, 224]`:

| Layout | minor_to_major | Description |
|--------|---------------|-------------|
| Column-major (default) | `{0, 1, 2, 3}` | n is minor, x is major. Elements with same z,y,x and consecutive n are contiguous. |
| Row-major | `{3, 2, 1, 0}` | x is minor, n is major. Elements with same n,z,y and consecutive x are contiguous. |
| NHWC (TensorFlow-style) | `{3, 2, 1, 0}` | Same as row-major. Width (x) is minor, batch (n) is major. |
| NCHW (PyTorch-style) | `{1, 0, 2, 3}` | A mixed ordering. |

The general formula for computing the memory offset of element `(d0, d1, ..., dN-1)` is:

```
offset = sum(di * stride_i for i in 0..N-1)

where stride_i = product(dj for j in minor_to_major after i)
```

More precisely, given `minor_to_major = [m0, m1, ..., mN-1]` where `m0` is the most minor dimension:

```
stride_{m0} = 1
stride_{m1} = dim_size[m0]
stride_{m2} = dim_size[m0] * dim_size[m1]
...
stride_{mK} = product(dim_size[m0] * ... * dim_size[mK-1])
```

### 2.4.6 Default Layout Conventions

XLA applies the following default layouts:

- **Default array layout**: `minor_to_major = {0, 1, 2, ..., rank-1}`. This is column-major order where dimension 0 is the most minor (changes fastest).
- **Default tuple layout**: For tuple shapes, each sub-shape has its own default layout.
- **Default scalar layout**: Scalars (rank-0 arrays) have an empty `minor_to_major` list.

These defaults are overridden by the **layout assignment** optimization pass, which assigns optimal layouts based on:
- The target hardware backend
- The operations in the computation (e.g., convolutions often prefer NHWC on GPUs)
- Fusion patterns
- Memory access patterns

### 2.4.7 Layout Assignment

The layout assignment pass (`LayoutAssignment`) is a critical optimization pass that assigns layouts to all instructions in the HLO module. It operates as follows:

1. **Start from outputs and parameters**: The layouts of parameters (inputs) and the root instruction (output) may be constrained by the calling convention (e.g., the framework may specify NHWC for convolution inputs).
2. **Propagate backward and forward**: The pass propagates layout constraints through the graph, attempting to find a consistent assignment that minimizes the total number of copy operations needed.
3. **Copy insertion**: If two instructions require different layouts for the same tensor, a `copy` instruction is inserted to convert between layouts. The pass minimizes the number of such copies.
4. **Backend-specific preferences**: Each backend can provide layout preferences for specific operations. For example, the GPU backend prefers NHWC layout for convolution inputs and outputs because cuDNN is optimized for this layout.

The layout assignment pass produces an HLO module where every array-shaped instruction has a fully specified layout.

---

## 2.5 Padding: tail_padding_alignment_in_elements

The `tail_padding_alignment_in_elements` field in the layout specifies that the last (most major) dimension of the array should be padded so that the total number of elements is a multiple of this value.

### Purpose

Tail padding serves several purposes:

1. **Memory alignment**: Ensures that the total allocation size is aligned to a specific boundary, enabling vectorized memory accesses and DMA transfers.
2. **Shared memory bank conflict avoidance**: On GPUs, padding can prevent shared memory bank conflicts by ensuring that accessed elements do not map to the same memory bank.
3. **Hardware requirements**: Some hardware requires allocations to be aligned to specific boundaries (e.g., 16 bytes for TPU HBM accesses).

### Example

For an array `f32[3, 4]` with `tail_padding_alignment_in_elements = 8`:

- Without padding: total elements = 3 * 4 = 12
- With padding: total elements is rounded up to the next multiple of 8 = 16

The extra 4 elements are unused padding at the end of the allocation.

### When Tail Padding Is Applied

Tail padding is typically set during layout assignment when:
- The backend specifies alignment requirements (e.g., GPU shared memory alignment).
- The operation emitter benefits from aligned memory accesses.
- The allocation is used as an input to a library call that requires alignment.

---

## 2.6 Indexing into Arrays (IndexUtil)

XLA provides the `IndexUtil` utility class for converting between multi-dimensional indices and linear (flat) memory offsets, taking into account the array's layout.

### Key Functions

#### `IndexUtil::LinearIndex`

Computes the linear memory offset for a given multi-dimensional index:

```cpp
// Given a shape and a multi_index, compute the linear offset
int64_t LinearIndex(const Shape& shape,
                    absl::Span<const int64_t> multi_index);
```

The computation follows the layout's minor-to-major ordering:

```
linear_index = sum(multi_index[dim] * stride[dim] for all dimensions)

where stride[dim] = product(shape.dimensions(minor_to_major[j]) for all j < position_of_dim_in_minor_to_major)
```

Example for `f32[3, 4]` with `minor_to_major = {0, 1}` (column-major):

```
multi_index = (2, 1)  // y=2, x=1
stride[0] = 1  (dimension 0 is most minor, stride = 1)
stride[1] = 3  (dimension 1 is next, stride = dim_size[0] = 3)
linear_index = 2 * 1 + 1 * 3 = 5
```

#### `IndexUtil::MultidimensionalIndex`

The inverse operation: converts a linear offset to a multi-dimensional index:

```cpp
// Given a shape and a linear_index, compute the multi-dimensional index
std::vector<int64_t> MultidimensionalIndex(const Shape& shape,
                                            int64_t linear_index);
```

This iterates through the minor-to-major ordering in reverse (from most major to most minor), extracting each dimension's index by division and modulus:

```
for dim in minor_to_major reversed:
    multi_index[dim] = linear_index % shape.dimensions(dim)
    linear_index /= shape.dimensions(dim)
```

#### `IndexUtil::IsValidIndex`

Checks whether a given multi-dimensional index is within the bounds of a shape:

```cpp
bool IsValidIndex(const Shape& shape,
                  absl::Span<const int64_t> multi_index);
```

Returns true if and only if `0 <= multi_index[d] < shape.dimensions(d)` for all dimensions `d`.

#### `IndexUtil::BumpIndices`

Advances a multi-dimensional index by one position in minor-to-major order (effectively incrementing the multi-dimensional index as if it were a counter):

```cpp
bool BumpIndices(const Shape& shape,
                 absl::Span<int64_t> multi_index);
```

Returns `true` if the index was successfully bumped, or `false` if the index has overflowed (i.e., the index has wrapped around to all-zeros).

---

## 2.7 Memory Space Identifiers

XLA's layout system includes a **memory space** identifier that specifies which memory region an allocation resides in. This is critical for hardware with multiple memory hierarchies, such as GPUs with global memory, shared memory, and registers.

### Memory Space Values

The memory space is specified as an integer in the layout. The semantics are backend-specific:

| Identifier | GPU Meaning | Description |
|-----------|-------------|-------------|
| **S(0)** | Default (global) memory | The device's main memory (e.g., GPU HBM). This is the default for all allocations. |
| **S(1)** | Alternate memory (shared memory on GPU) | Fast, small, on-chip memory. On GPUs, this maps to CUDA shared memory. Used for intermediate results in fused operations. |
| **S(2)** | Alternate memory 2 | Additional memory space. Backend-specific usage. |
| **S(3)** | Alternate memory 3 | Additional memory space. Backend-specific usage. |
| **S(5)** | Temporary or scratch space | Used for temporary allocations during computation. May alias with other buffers. |

### Default Memory Space

If no memory space is specified in the layout, the default is `S(0)` (global/default memory). This is where all input, output, and intermediate tensors are allocated by default.

### Usage in Fusion

During fusion optimization, the compiler may assign intermediate buffers to alternate memory spaces. For example, on GPUs:

1. Input buffers to a fusion kernel are in `S(0)` (global memory).
2. Intermediate buffers used within the fusion may be assigned to `S(1)` (shared memory) for fast access.
3. Output buffers are in `S(0)` (global memory).

The memory space assignment is done by the **memory space assignment** pass, which considers:
- The size of the alternate memory
- The access patterns of the computation
- The cost of copying between memory spaces

### Example with Memory Spaces

```
// A fusion where intermediate results use shared memory
fusion.1 = f32[128,512] fusion(parameter.0, parameter.1, parameter.2),
    kind=kCustom, calls=fused_computation,
    frontend_attributes={
        _xla_other_memory_space="1"
    }
```

---

## 2.8 Tiled Layouts

Tiled layouts are an advanced layout feature that organizes array elements into fixed-size tiles (blocks). This is particularly useful for hardware that operates on data in blocks, such as GPUs with tensor cores and TPUs with systolic arrays.

### 2.8.1 TileProto Definition

```protobuf
message TileProto {
    // The number of elements in each dimension of the tile.
    // For example, {16, 16} represents a 16x16 tile for a 2D array.
    repeated int64 dimensions = 1;
}
```

### 2.8.2 How Tiled Layouts Work

In a tiled layout, the array is divided into fixed-size blocks (tiles). Elements within each tile are stored contiguously, and tiles are arranged in the order specified by the minor-to-major dimension ordering.

For example, consider a `f32[128, 128]` array with a tiled layout using `tile = {16, 16}` and `minor_to_major = {0, 1}`:

1. The array is divided into 8x8 = 64 tiles, each of size 16x16.
2. Within each tile, elements are stored in column-major order: elements with consecutive row indices are contiguous.
3. Tiles themselves are arranged in column-major order: tiles with consecutive column indices are contiguous.

The physical layout in memory would be:

```
Tile(0,0) elements: T[0..15, 0..15]   (256 elements, column-major within tile)
Tile(0,1) elements: T[0..15, 16..31]  (256 elements)
...
Tile(7,7) elements: T[112..127, 112..127] (256 elements)
```

### 2.8.3 Benefits of Tiled Layouts

Tiled layouts provide several performance benefits:

1. **Improved spatial locality**: Accessing elements within a tile accesses a contiguous block of memory, improving cache and memory coalescing behavior.
2. **Hardware alignment**: Tiles can be sized to match hardware block sizes (e.g., 16x16 for GPU tensor cores, 8x8 for certain TPU operations).
3. **Vectorized access**: Tile boundaries provide natural boundaries for vectorized load/store operations.
4. **Reduced bank conflicts**: On GPUs, tiled access patterns can be designed to avoid shared memory bank conflicts.

### 2.8.4 Padded Physical Dimensions

When the array dimensions are not multiples of the tile size, the physical dimensions are padded up to the next tile boundary. For example:

- Logical shape: `f32[100, 100]` with tile `{16, 16}`
- Physical shape: `f32[112, 112]` (100 rounded up to next multiple of 16)

The padded elements are unused. The physical shape is tracked separately from the logical shape.

---

## 2.9 ShapeUtil and LayoutUtil Utilities

XLA provides two important utility classes for working with shapes and layouts programmatically.

### 2.9.1 ShapeUtil

`ShapeUtil` (defined in `xla/shape_util.h`) provides functions for creating, querying, and manipulating shapes.

#### Construction Functions

| Function | Description |
|----------|-------------|
| `MakeShape(element_type, dimensions)` | Creates an array shape with default layout |
| `MakeShapeWithLayout(element_type, dimensions, minor_to_major)` | Creates an array shape with specified layout |
| `MakeTupleShape(subshapes)` | Creates a tuple shape from a list of sub-shapes |
| `MakeTokenShape()` | Creates a token shape |
| `MakeValidShape(element_type, dimensions)` | Creates and validates an array shape |

#### Query Functions

| Function | Description |
|----------|-------------|
| `GetDimension(const Shape& shape, int64_t dim)` | Returns the size of a dimension |
| `Rank(const Shape& shape)` | Returns the rank (number of dimensions) |
| `Dimensions(const Shape& shape)` | Returns all dimension sizes |
| `ElementSizeBytes(const Shape& shape)` | Returns the size of one element in bytes |
| `ByteSizeOf(const Shape& shape)` | Returns the total byte size of the shape |
| `ByteSizeOfElements(const Shape& shape)` | Returns the byte size considering only elements (no padding) |
| `IsArray(const Shape& shape)` | Returns true if the shape is an array |
| `IsTuple(const Shape& shape)` | Returns true if the shape is a tuple |
| `IsScalar(const Shape& shape)` | Returns true if the shape is a scalar (rank 0) |
| `IsEmptyTuple(const Shape& shape)` | Returns true if the shape is an empty tuple |
| `IsEffectiveScalar(const Shape& shape)` | Returns true if the shape has 1 element |
| `ElementsIn(const Shape& shape)` | Returns the total number of elements |
| `GetSubshape(const Shape& shape, ShapeIndexView index)` | Returns a sub-shape at the given index path |

#### Comparison Functions

| Function | Description |
|----------|-------------|
| `Equal(const Shape& a, const Shape& b)` | Returns true if shapes are identical (including layout) |
| `Compatible(const Shape& a, const Shape& b)` | Returns true if shapes have the same element type and dimensions (ignoring layout) |
| `CompatibleIgnoringElementType(const Shape& a, const Shape& b)` | Returns true if shapes have the same dimensions (ignoring element type and layout) |

#### Mutation Functions

| Function | Description |
|----------|-------------|
| `AppendMajorDimension(int64_t size, Shape* shape)` | Appends a new major dimension |
| `AppendMinorDimension(int64_t size, Shape* shape)` | Appends a new minor dimension |
| `DeleteDimension(int64_t dim, Shape* shape)` | Deletes a dimension |
| `UpdateDynamicDimension(Shape* shape, int64_t dim, bool is_dynamic)` | Sets whether a dimension is dynamic |

#### Transposition and Reshaping

| Function | Description |
|----------|-------------|
| `PermuteDimensions(absl::Span<const int64_t> permutation, const Shape& shape)` | Returns a shape with permuted dimensions |
| `TransposeDimensions(const Shape& shape, int64_t dim_a, int64_t dim_b)` | Swaps two dimensions |
| `ReshapeShape(const Shape& shape, absl::Span<const int64_t> new_dimensions)` | Returns a reshaped shape |

### 2.9.2 LayoutUtil

`LayoutUtil` (defined in `xla/layout_util.h`) provides functions for creating, querying, and manipulating layouts.

#### Construction Functions

| Function | Description |
|----------|-------------|
| `MakeLayout(minor_to_major)` | Creates a layout with specified minor-to-major ordering |
| `MakeLayout(minor_to_major, tiles, memory_space)` | Creates a layout with tiles and memory space |
| `GetDefaultLayoutForShape(const Shape& shape)` | Returns the default layout for a given shape |
| `GetDefaultLayoutForRank(int rank)` | Returns the default layout for a given rank |

#### Query Functions

| Function | Description |
|----------|-------------|
| `Minor(const Layout& layout)` | Returns the most minor dimension |
| `Major(const Layout& layout)` | Returns the most major dimension |
| `GetDimensionOrder(const Layout& layout)` | Returns the minor-to-major ordering as a vector |
| `MinorToMajor(const Layout& layout)` | Returns the minor-to-major ordering |
| `MajorToMinor(const Layout& layout)` | Returns the major-to-minor ordering (reverse of minor-to-major) |
| `HasTile(const Layout& layout)` | Returns true if the layout has tiling |
| `GetTile(const Layout& layout)` | Returns the tile dimensions |
| `TailPaddingAlignment(const Layout& layout)` | Returns the tail padding alignment |
| `MemorySpace(const Layout& layout)` | Returns the memory space identifier |
| `IsDefaultLayout(const Shape& shape)` | Returns true if the shape has the default layout |
| `IsMonotonicWithDim0Minor(const Layout& layout)` | Returns true if `minor_to_major = {0, 1, ..., rank-1}` |
| `IsMonotonicWithDim0Major(const Layout& layout)` | Returns true if `minor_to_major = {rank-1, ..., 1, 0}` |

#### Validation Functions

| Function | Description |
|----------|-------------|
| `ValidateLayoutForShape(const Layout& layout, const Shape& shape)` | Validates that a layout is valid for a shape |
| `LayoutValidForShape(const Layout& layout, const Shape& shape)` | Returns true if layout is valid for shape |

#### Layout Comparison

| Function | Description |
|----------|-------------|
| `Equal(const Layout& a, const Layout& b)` | Returns true if layouts are identical |
| `DimensionsImplicitlyPermutation(const Layout& a, const Layout& b)` | Checks if two layouts are compatible under a permutation |

---

## 2.10 Example HLO with Layout Annotations

This section provides complete examples of HLO modules with explicit layout annotations, demonstrating how layouts affect computation.

### Example 1: Matrix Multiplication with Explicit Layouts

```
HloModule matmul_with_layouts

ENTRY main {
  // Input parameters with specific layouts
  // parameter.0: f32[128, 256] with column-major layout (minor_to_major={0,1})
  parameter.0 = f32[128, 256]{0,1} parameter(0)
  
  // parameter.1: f32[256, 512] with column-major layout (minor_to_major={0,1})
  parameter.1 = f32[256, 512]{0,1} parameter(1)

  // Matrix multiplication
  // Contracting dimension 1 of LHS with dimension 0 of RHS
  dot.0 = f32[128, 512]{0,1} dot(parameter.0, parameter.1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}

  ROOT result = f32[128, 512]{0,1} copy(dot.0)
}
```

The `{0,1}` suffix on shapes indicates the minor-to-major layout. In this case, both inputs and the output use column-major layout (dimension 0 is most minor).

### Example 2: Convolution with NHWC Layout

```
HloModule conv_nhwc

ENTRY main {
  // Input: f32[batch, height, width, channels] = f32[1, 224, 224, 3]
  // Layout {3,2,1,0} = row-major = NHWC (width is most minor)
  input = f32[1, 224, 224, 3]{3,2,1,0} parameter(0)
  
  // Kernel: f32[kernel_h, kernel_w, in_channels, out_channels] = f32[3, 3, 3, 64]
  // Layout {3,2,1,0} = row-major
  kernel = f32[3, 3, 3, 64]{3,2,1,0} parameter(1)
  
  // Convolution with appropriate window attributes
  conv = f32[1, 224, 224, 64]{3,2,1,0} convolution(input, kernel),
      window={size=3x3 pad=1_1x1_1},
      dim_labels=b01f_01io->b01f,
      feature_group_count=1

  ROOT result = f32[1, 224, 224, 64]{3,2,1,0} copy(conv)
}
```

The `dim_labels=b01f_01io->b01f` attribute specifies:
- Input dimensions: `b` (batch), `0/1` (spatial), `f` (features/channels)
- Kernel dimensions: `0/1` (spatial), `i` (input channels), `o` (output channels)
- Output dimensions: `b` (batch), `0/1` (spatial), `f` (features/channels)

### Example 3: Tuple Shape with Mixed Layouts

```
HloModule tuple_example

ENTRY main {
  parameter.0 = f32[128, 256]{0,1} parameter(0)
  parameter.1 = f32[64, 64]{1,0} parameter(1)
  parameter.2 = f32[] parameter(2)
  
  // Create a tuple containing arrays with different layouts
  tuple.0 = (f32[128, 256]{0,1}, f32[64, 64]{1,0}, f32[]) tuple(parameter.0, parameter.1, parameter.2)

  ROOT result = (f32[128, 256]{0,1}, f32[64, 64]{1,0}, f32[]) tuple(parameter.0, parameter.1, parameter.2)
}
```

### Example 4: Layout Conversion (Copy Operation)

```
HloModule layout_conversion

ENTRY main {
  // Input with row-major layout
  input = f32[3, 4]{1,0} parameter(0)
  
  // Copy to column-major layout
  // This generates a transpose/copy operation at runtime
  output = f32[3, 4]{0,1} copy(input)
  
  ROOT result = f32[3, 4]{0,1} copy(output)
}
```

The `copy` operation converts between layouts. When the source and destination layouts differ, this is a non-trivial operation that reorders elements in memory.

### Example 5: Tiled Layout

```
HloModule tiled_layout_example

ENTRY main {
  // f32[128, 128] with 16x16 tiles
  // Layout annotation: (T=(16,16))
  input = f32[128, 128]{0,1:T(16,16)} parameter(0)
  
  // Element-wise operation preserves tiled layout
  output = f32[128, 128]{0,1:T(16,16)} sin(input)
  
  ROOT result = f32[128, 128]{0,1:T(16,16)} copy(output)
}
```

---

## 2.11 Layout Constraints and Copy Insertion

When an operation requires a specific layout for its inputs or outputs but the producer or consumer provides a different layout, XLA inserts a `copy` instruction to convert between layouts.

### Automatic Copy Insertion

The copy insertion pass (`CopyInsertion`) runs after layout assignment to ensure layout consistency:

1. **Parameter layout mismatches**: If the calling convention specifies a layout for a parameter but the computation uses a different layout internally, a copy is inserted at the entry point.
2. **Operation layout constraints**: If an operation requires a specific layout for its operands (e.g., some GPU operations require column-major), a copy is inserted before the operation.
3. **Fusion layout mismatches**: If the producer and consumer in a fusion have incompatible layouts, a copy is inserted.

### Cost of Copies

Copy operations are not free -- they require:
- Additional memory bandwidth (reading the entire tensor and writing it back).
- Additional memory allocation (temporary storage for the copy).
- Additional kernel launch overhead.

The layout assignment pass minimizes the number and size of copies by choosing layouts that are compatible with as many operations as possible.

---

## 2.12 Layout in Practice

### Debugging Layout Issues

Common layout-related issues and their symptoms:

| Symptom | Possible Cause |
|---------|---------------|
| Unexpectedly slow execution | Layout mismatch causing unnecessary copies |
| Incorrect numerical results | Layout misinterpretation (e.g., treating row-major data as column-major) |
| Excessive memory usage | Poor layout assignment causing large temporary allocations |
| Compilation failure | Incompatible layout constraints that cannot be resolved |

### Controlling Layout from Frameworks

**JAX**: JAX does not expose direct layout control, but you can influence layout through:
- Using `jax.jit` with `compiler_kwargs` to pass layout hints.
- Examining compiled HLO to verify layout decisions.

**TensorFlow**: TensorFlow provides layout hints through:
- `tf.function(jit_compile=True)` with data format specifications (`"NHWC"` vs `"NCHW"`).
- `tf.nn.conv2d(..., data_format="NHWC")` which sets layout preferences.

**PyTorch/XLA**: PyTorch/XLA respects PyTorch's memory format conventions:
- `torch.contiguous_format` maps to row-major layout.
- `torch.channels_last_format` maps to NHWC layout.

---

## 2.13 Summary

XLA's shape and layout system is a comprehensive framework for describing the type, dimensionality, and memory organization of tensors. Shapes define what data is stored (element type, dimensions, dynamic dimensions), while layouts define how data is stored in memory (minor-to-major ordering, tiling, padding, memory space). The layout assignment pass optimizes layouts for the target hardware, and copy insertion ensures consistency. Understanding these concepts is essential for diagnosing performance issues, writing custom operations, and effectively using XLA as a compilation backend.
