# Operations: Views

Views are a structured way to interact with tensors in memory. They are described in both the types section (Tensor View) and the semantics section (Views).

Views are the primary way to interact with global memory in Tile IR. A common pattern is to construct a Tensor View from a pointer with `cuda_tile.make_tensor_view` and then use the `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko` operations to read and write to them. For larger tensors, loading the entire tensor is not efficient and therefore we have a sub-view Partition View which allows a user to tile a tensor_view.

---

## `cuda_tile.get_index_space_shape`

Query the index space dimension size.

```
cuda_tile.get_index_space_shape %src
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| src | `view_type` | The source view type. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `Variadic<tile<any>>` | The shape of the index space, each value representing the size of the corresponding dimension. |

**Description:**

The get_index_space_shape operation returns the shape of the index space of src. The result tile has the same rank as the view's index space with the elements representing the size of the corresponding dimension. The result values should be interpreted as unsigned integers.

> **Warning:** If the individual index space dimension does not fit in the result tile's element type the behavior is undefined.

**Constraints:**

- Operation must not perform any memory side effects.

**Examples:**

```cuda_tile
%tensor_view = make_tensor_view %base,
    shape = [2, 2, 4], strides = [2, 2, 1]
    : tensor_view<2x2x4xf32, strides=[2,2,1]>
%partition_view = make_partition_view %tensor_view :
  partition_view<tile=(2x2x4), tensor_view<2x2x4xf32, strides=[2,2,1]>>
%dim0, %dim1, %dim2 = get_index_space_shape %partition_view :
  partition_view<tile=(2x2x4), tensor_view<2x2x4xf32, strides=[2,2,1]>> -> tile<i64>
```

---

## `cuda_tile.get_tensor_shape`

Query the shape of a tensor view.

```
cuda_tile.get_tensor_shape %src
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| src | `tensor_view` | The source tensor view. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `Variadic<tile<any>>` | The shape of the tensor, each value representing the size of the corresponding dimension. |

**Description:**

The get_tensor_shape operation returns the shape of the tensor backing the provided tensor view. The result values should be interpreted as unsigned integers.

> **Warning:** If the tensor dimensions do not fit in the result tile's element type the behavior is undefined.

**Constraints:**

- Operation must not perform any memory side effects.

**Examples:**

```cuda_tile
%dim0, %dim1 = get_tensor_shape %tensor_view : tensor_view<32x32xf32, strides=[32,1]> -> tile<i64>
```

---

## `cuda_tile.load_view_tko`

Load a tile from a tile view.

```
cuda_tile.load_view_tko %memory_ordering_semantics %memory_scope %view %index %token %optimization_hints
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| memory_ordering_semantics | `MemoryOrderingSemantics` | The memory ordering semantics for the load operation. |
| memory_scope | `MemoryScope` | The memory scope for the atomic operation. |
| view | `view_type` | The view from which the tile will be loaded. |
| index | `Variadic<tile<any>>` | The n-dimensional index of the desired element to load from the view. |
| token | `token` | The optional token for the load operation. |
| optimization_hints | `OptimizationHints` | Optimization hints for operation. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| tile | `tile` | The loaded tile. |
| result_token | `token` | The result token. |

**Description:**

The load_view_tko operation loads a tile from a tile view. A view is a mapping from view-space indices to a particular element in the view; each view type has a defined mapping from view-space indices to tiles produced from elements of the view.

For example, the Partition View partitions a Tensor View into a grid of equally sized tiles. The view indexes one of the partitioned tiles in the grid.

For a given view the rank of the indices must match the rank of the view's index space. The space of valid indices depends on which view is passed to the operation. For example the index space of a Partition View is equal to the rank of the partitioned tiles.

The index operands are interpreted as unsigned integers. Out of bounds accesses are handled according to the semantics of Partition View.

**Memory ordering semantics:**

| Ordering | Description |
|----------|-------------|
| `weak` | No concurrent accesses to the source/destination location. |
| `relaxed` | There may be concurrent access to the location, but this access does not establish a happens-before relationship. |
| `acquire` | There may be concurrent accesses to the location. If this acquire observes a release operation, then happens before is established. |

Note: `release` and `acq_rel` are not supported by this operation.

**Memory scope:**

| Scope | Description |
|-------|-------------|
| `tl_blk` | There may be concurrent accesses from within the same tile block. |
| `device` | There may be concurrent accesses from within the same device (i.e., GPU). |
| `sys` | There may be concurrent accesses from anywhere within the system (i.e., all devices). |

**Optimization hints:**

The optimization_hints attribute provides architecture-specific compiler hints in the form of nested dictionaries. The hints are specified for each architecture (e.g., sm_100, sm_120) and for each architecture the user can specify specific hints for each operation:

- `num_cta_in_cga` - suggest the number of CTAs in a CGA for `cuda_tile.entry`.
- `allow_tma` - suggest whether to use TMA for `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko`.
- `latency` - latency hint for `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko`.

**Constraints:**

- Operation must encode variadic operand segment sizes in attributes.

**Examples:**

```cuda_tile
%tensor_view = make_tensor_view %ptr, shape=[8192, 128], strides=[128, 1]
  : tensor_view<8192x128xf32, strides=[128,1]>

// This example uses the PartitionView on a 8192x128xf32 tensor_view,
// dividing the tensor_view in tiles of 64x64.

%view = make_partition_view %tensor_view : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>

%c0 = constant <i32: 0> : tile<i32>
%c1 = constant <i32: 1> : tile<i32>

// Load a tile at index (0, 0) in the view's index space.
%tile0, %res_token0 = load_view_tko weak %view[%c0, %c0]
  : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> tile<64x64xf32>, token

// Load a tile at index (0, 1) in the view's index space.
%tile1, %res_token1 = load_view_tko weak %view[%c0, %c1]
  : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> tile<64x64xf32>, token

// Same example as above but with memory token as input.
%token = make_token : token
%tile2, %res_token2 = load_view_tko weak %view[%c0, %c1] token = %token
  : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> tile<64x64xf32>, token

// Loads a tile at the dynamic index (%index, %index) in the view's index space.
%tile3, %res_token3 = load_view_tko weak %view[%index, %index]
  : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> tile<64x64xf32>, token
```

---

## `cuda_tile.make_partition_view`

Create a partition view from a tensor view.

```
cuda_tile.make_partition_view %tensor_view
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| tensor_view | `tensor_view` | The source tensor view to create a partition view from. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `partition_view` | The created partition view. |

**Description:**

The make_partition_view operation creates a partition_view from a tensor_view. The operation uses the type constraints of the input tensor view and the annotated return type to perform the partitioning. The tensor view's type contains its physical layout in the form of shapes and strides and the partition view contains the logical size of a single tile.

The resulting partition view can be loaded from using `cuda_tile.load_view_tko` and stored to using `cuda_tile.store_view_tko`. The view memory options act on the computed index space of the partition view.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.

**Examples:**

```cuda_tile
%tensor_view0 = make_tensor_view %ptr, shape=[8192, 8192, 64], strides=[524288,64,1]
  : tensor_view<8192x8192x64xf32, strides=[524288,64,1]>

// Creates a partition with 32-bit-indexed tiles of size (1024x1x32) over
// the provided tensor_view.
make_partition_view %tensor_view0 :
  partition_view<
    tile=(1024x1x32),
    tensor_view<8192x8192x64xf32, strides=[524288,64,1]>
  >

%s0 = constant <i32: 8192> : tile<i32>
%str0 = constant <i32: 524288> : tile<i32>

%tensor_view1 = make_tensor_view %ptr, shape=[%s0, 8192, 64], strides=[%str0, 64, 1]
  : tile<i32> -> tensor_view<?x8192x64xf32, strides=[?,64,1]>

// Creates a partition with 32-bit-indexed tiles of size (1024x1x32) over
// the provided tensor_view. The provided tensor_view has a
// dynamically-sized dimension.
make_partition_view %tensor_view1 :
  partition_view<tile=(1024x1x32), tensor_view<?x8192x64xf32, strides=[?,64,1]>>
```

---

## `cuda_tile.make_tensor_view`

Create `tensor_view` from a pointer to global memory.

```
cuda_tile.make_tensor_view %base %dynamicShape %dynamicStrides
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| base | `tile<ptr>` | The scalar base pointer to a portion of global memory. |
| dynamicShape | `Variadic<tile<any>>` | The array of values representing the shape of the view, may be fully dynamic. |
| dynamicStrides | `Variadic<tile<any>>` | The array of values representing the strides of the view, may be fully dynamic. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tensor_view` | The constructed tensor_view. |

**Description:**

The make_tensor_view operation constructs a tensor_view from a global memory pointer, a dynamic shape and dynamic strides. The constructor supports taking dynamic arrays for shapes and strides as part of the constructor enabling workloads to take global memory tensors of dynamic shape and strides. If these arguments are static they will be statically reflected in the type of the resulting tensor_view, if they are dynamic they will appear as `?` in the type.

The dynamicShape and dynamicStrides operands are interpreted as unsigned integers.

**Constraints:**

- Operation must encode variadic operand segment sizes in attributes.
- Operation must not perform any memory side effects.

**Examples:**

```cuda_tile
  // tensor_view to a scalar tile of f32
  %a0 = make_tensor_view %base,
      shape = [], strides = [] : tensor_view<f32>

  // tensor_view to a tile of static shape and strides
  %a1 = make_tensor_view %base,
      shape = [32, 32], strides = [32, 1]
      : tensor_view<32x32xf32, strides=[32,1]>

%sh0 = constant <i32: 32> : tile<i32>
%sh1 = constant <i32: 32> : tile<i32>
%st0 = constant <i32: 32> : tile<i32>
%st1 = constant <i32: 1> : tile<i32>

  // tensor_view to a tile with partially dynamic shape and strides
  // all dynamic values must be of the same type, here tile<i32>
  %a2 = make_tensor_view %base,
          shape = [%sh0, %sh1], strides = [%st0, %st1]
          : tile<i32> -> tensor_view<?x?xf32, strides=[?,?]>
```

---

## `cuda_tile.store_view_tko`

Stores a tile into a tile view.

```
cuda_tile.store_view_tko %memory_ordering_semantics %memory_scope %tile %view %index %token %optimization_hints
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| memory_ordering_semantics | `MemoryOrderingSemantics` | The memory ordering semantics for the store operation. |
| memory_scope | `MemoryScope` | The memory scope for the store operation. |
| tile | `tile` | The tile to store. |
| view | `view_type` | The view to store the tile to. |
| index | `Variadic<tile<any>>` | The indices of the desired target tile within the view. |
| token | `token` | The optional token for operation ordering. |
| optimization_hints | `OptimizationHints` | Optimization hints for operation. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result_token | `token` | The result token for synchronization. |

**Description:**

The store_view_tko operation stores a tile to a view indexing into a tile view. A view is a mapping from view-space indices to a particular element in the view; each view type has a defined mapping from view-space indices to tiles produced from elements of the view.

For example, the Partition View partitions a Tensor View into a grid of equally sized tiles. The view indexes one of the partitioned tiles in the grid.

For a given view the rank of the indices must match the rank of the view's index space. The space of valid indices depends on which view is passed to the operation. The index space of the view is computed as a function of the requested tile size and the shape of the view.

The index operands are interpreted as unsigned integers. Out of bounds accesses are handled according to the semantics of Partition View.

**Memory ordering semantics:**

| Ordering | Description |
|----------|-------------|
| `weak` | No concurrent accesses to the source/destination location. |
| `relaxed` | There may be concurrent access to the location, but this access does not establish a happens-before relationship. |
| `release` | There may be concurrent access to the location. If this release is observed with an acquire operation, then happens before is established. |

Note: `acquire` and `acq_rel` are not supported by this operation.

**Memory scope:**

| Scope | Description |
|-------|-------------|
| `tl_blk` | There may be concurrent accesses from within the same tile block. |
| `device` | There may be concurrent accesses from within the same device (i.e., GPU). |
| `sys` | There may be concurrent accesses from anywhere within the system (i.e., all devices). |

**Optimization hints:**

The optimization_hints attribute provides architecture-specific compiler hints in the form of nested dictionaries:

- `num_cta_in_cga` - suggest the number of CTAs in a CGA for `cuda_tile.entry`.
- `allow_tma` - suggest whether to use TMA for `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko`.
- `latency` - latency hint for `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko`.

**Constraints:**

- Operation must encode variadic operand segment sizes in attributes.
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
%tensor_view = make_tensor_view %ptr, shape=[8192, 128], strides=[128,1] :
  tensor_view<8192x128xf32, strides=[128,1]>

// This example uses the PartitionView on a 8192x128xf32 tensor_view,
// dividing the tensor_view in tiles of 64x64.
%view = make_partition_view %tensor_view :
  partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>

%c0 = constant <i32: 0> : tile<i32>
%c1 = constant <i32: 1> : tile<i32>

%tile = constant <f32: 0.0> : tile<64x64xf32>

// Store a tile at index (0, 0) in the view's index space.
%res_token0 = store_view_tko weak %tile, %view[%c0, %c0]
  : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> token

// Store a tile at index (0, 1) in the view's index space.
%res_token1 = store_view_tko weak %tile, %view[%c0, %c1]
  : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> token

// Same example as above but with input token.
%token = make_token : token
%res_token2 = store_view_tko weak %tile, %view[%c0, %c1] token = %token
  : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32> -> token
```

---

## Miscellaneous Operations

The set of miscellaneous operations in Tile IR are operations which do not have a specific category.

### `cuda_tile.assume`

Attach static information to an SSA value.

```
cuda_tile.assume %value %predicate
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| value | `Any` | The value to attach the predicate to. |
| predicate | `AssumePredicate` | The predicate to attach to the value. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `Any` | The value with the attached predicate. |

**Description:**

The assume operation passes through value as the result and attaches a predicate to it. The assumed predicate is a property of result. This operation can be used to inject static information into the compiler, potentially resulting in more efficient code generation. The predicate must implement the AssumePredicateAttrInterface.

> **Note:** assume does not check the correctness of the predicate. Incorrect predicates may inject incorrect static information and cause miscompilation. If an incorrect predicate is attached to an SSA value, the behavior of the program is undefined.

**AssumePredicate Implementers:**

#### Bounded

```
#bounded<(lb|?), (ub|?)>
```

The bounded attribute must be used as a predicate for `cuda_tile.assume`. The predicated value must be a tile of integers. bounded specifies a lower and upper bound for all elements of the predicated tile when interpreted as signed integers. Bounds are optional: it is possible to leave a bound unspecified, as indicated by "?" in the assembly format. E.g., `#bounded<0, ?>`. Both lower bound and upper bound are inclusive.

The lower bounds must be less than or equal to the upper bound. A lower/upper bound that exceeds the range of valid values of the predicated value is invalid.

```cuda_tile
%1 = cuda_tile.assume #cuda_tile.bounded<0, ?>, %0
    : !cuda_tile.tile<4x8xi16>
```

#### DivBy

```
div_by< $divisor (, every $every^ along $along)?>
```

The div_by attribute must be used as a predicate for cuda_tile.assume ops. The predicated value must be a tile of integers or pointers, or a tensor_view.

If the predicated value is a tile, the attribute indicates that some elements of the tile are divisible by divisor. If the predicated value is a tensor_view the attribute indicates that the base address of the tensor_view is divisible by divisor. divisor must be a positive power of 2.

The `every` and `along` attributes control which elements are assumed to satisfy the divisibility property. When splitting the tensor in groups of size `every` along dimension `along`, the first element of each group is assumed to satisfy the divisibility property. The other elements are assumed to be monotonically increasing by 1 within the group. In case of a tile of pointers, the elements are assumed to be monotonically increasing by the byte width of the pointee type. The size of the last group may be smaller than `every`.

The `every` and `along` attributes are optional. When missing, they are assumed to have a default value of 1 and 0 in case of a tile. I.e., all elements of the tile are assumed to satisfy the divisibility property. (The value of `along` does not matter in that case.) If the predicated value is a tensor_view or a 0D tile, `every` and `along` cannot be used. `every` and `along` must be used together. If one is specified, so must be the other.

> **Note:** If the predicated value is a tile of integers, `every` is a property of the signed interpretation of the integer values. Otherwise, it is a property of the unsigned integer interpretation.

#### SameElements

```
#same_elements< $values >
```

The same_elements attribute must be used as a predicate for cuda_tile.assume. The predicated value must be a tensor of integers or pointers.

same_elements is specified for each dimension. A value of C for a dimension of size N indicates that, after dividing the respective dimension into N/C groups of size C, each group consists of the same elements. As N/C may not divide evenly, the last group may have fewer than C elements.

If the "same elements" property does not hold along a dimension, the respective value should be set to 1. `#cuda_tile.same_elements<[1, 1, ..., 1]>` is a correct predicate for any tensor of integers or pointers, where the number of ones matches the rank of the tensor. (Size-1 groups always have the same elements.)

```cuda_tile
%1 = cuda_tile.assume #cuda_tile.same_elements<[2, 4]>, %0
    : !cuda_tile.tile<4x8xi16>
```

**Constraints:**

- value and result must have the same shape and element type (Any).
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
// Assume that all integers are divisible by 32.
%int_tile = constant <i16: [32, 64, 0, 0, 32, -32, 1024, 0]> : tile<8xi16>
%div_by_1 = assume div_by<32>, %int_tile : tile<8xi16>

// Assume that every 4th element (starting with element 0) along
// dimension 0 is divisible by 32 that and all integers are
// monotonically increasing by 1 within each group of 4.
%int_tile_2 = constant <i16: [96, 97, 98, 99, 64, 65, 66, 67]> : tile<8xi16>
%div_by_2 = assume div_by<32, every 4 along 0>, %int_tile_2 : tile<8xi16>

// Assume that every rectangular chunk of size [1, 4, 2] has the same
// values.
%same_elem = assume same_elements<[1, 4, 2]>, %ptr_3d : tile<1x8x8xptr<f32>>

// Assume that every value is greater or equal to 5.
%int_tile_3 = constant <i16: [5, 9, 10, 11, 6, 5, 5, 7]> : tile<8xi16>
%bounded = assume bounded<5, ?>, %int_tile_3 : tile<8xi16>
```

---

### `cuda_tile.print_tko`

Print a formatted string (token-ordered).

```
cuda_tile.print_tko %str %args %token
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| str | `String` | The format string. |
| args | `Variadic<tile>` | The arguments to format and print. |
| token | `token` | The optional token for operation ordering. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result_token | `token` | The result token for synchronization. |

**Description:**

The print_tko operation prints a C-printf-style format string, interleaved with the given operands. The number of format expressions (starting with the % character) must match the number of operands. If a format expression is not applicable to its respective operand, then the output is undefined.

Token-ordered print operations are not constrained by program order. The compiler may reorder them (i.e., move them earlier or later in the program) unless further constrained by tokens.

This operation is meant for debugging. Its implementation is not optimized for performance, so it should not be used in production mode. Prints are not guaranteed to be atomic. I.e., the output of prints that execute simultaneously may be interleaved.

> **Note:** This op was renamed from `print` to `print_tko` in 13.2. The op code did not change.

**Constraints:**

- Operation must encode variadic operand segment sizes in attributes.
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
print_tko "Hello world: %f\n", %arg : tile<4xf32> -> token
print_tko "%+08.3f", %arg : tile<4xf32> -> token
```
