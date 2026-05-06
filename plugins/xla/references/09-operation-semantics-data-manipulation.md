# XLA Operation Semantics: Data Manipulation Operations

This reference provides comprehensive documentation of all XLA data manipulation operations. These operations reshape, slice, broadcast, concatenate, permute, and otherwise rearrange tensor data without performing mathematical computations. They are fundamental building blocks used in virtually every XLA computation.

---

## Table of Contents

1. [Reshape and DynamicReshape](#reshape-and-dynamicreshape)
2. [Collapse](#collapse)
3. [Transpose](#transpose)
4. [Slice and DynamicSlice](#slice-and-dynamicslice)
5. [DynamicUpdateSlice](#dynamicupdateslice)
6. [ConcatInDim](#concatindim)
7. [Broadcast and BroadcastInDim](#broadcast-and-broadcastindim)
8. [Pad](#pad)
9. [Reverse](#reverse)
10. [Gather](#gather)
11. [DynamicGather](#dynamicgather)
12. [BitcastConvertType](#bitcastconverttype)
13. [ConvertElementType](#convertelementtype)
14. [Copy](#copy)
15. [StableHLO Cross-References](#stablehlo-cross-references)

---

## Reshape and DynamicReshape

### Reshape

`Reshape` changes the shape of a tensor without changing its data. The total number of elements must remain the same.

#### Signature

```
Reshape(operand, new_shape)
```

#### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor. |
| `new_shape` | `Shape` | The desired output shape. The product of all dimensions must equal the product of all input dimensions. |

#### Semantics

The elements are read from the operand in **lexicographic order** (row-major, from the first dimension to the last) and placed into the output shape in the same order. No data is copied or rearranged in memory -- only the shape metadata changes.

Formally, if the input has shape `[d0, d1, ..., dn]` and the output has shape `[e0, e1, ..., em]`:
```
prod(d0, d1, ..., dn) == prod(e0, e1, ..., em)
```

#### Examples

```
// f32[2, 3] -> f32[6]
%result = f32[6] reshape(f32[2, 3] %operand)
// [[1,2,3],[4,5,6]] -> [1,2,3,4,5,6]

// f32[2, 3] -> f32[3, 2]
%result = f32[3, 2] reshape(f32[2, 3] %operand)
// [[1,2,3],[4,5,6]] -> [[1,2],[3,4],[5,6]]

// f32[2, 3, 4] -> f32[6, 4]
%result = f32[6, 4] reshape(f32[2, 3, 4] %operand)

// f32[6] -> f32[2, 1, 3]
%result = f32[2, 1, 3] reshape(f32[6] %operand)
```

#### HLO Text Format

```
%result = f32[6]{0} reshape(f32[2,3]{1,0} %operand)
%result = f32[3,2]{1,0} reshape(f32[2,3]{1,0} %operand)
```

#### Constraints

- The total number of elements must be preserved.
- A dimension of size `-1` in `new_shape` is not supported in XLA HLO (unlike some frameworks). All dimensions must be explicitly specified.

---

### DynamicReshape

`DynamicReshape` is similar to `Reshape` but supports output shapes that are not statically known. The dimension sizes may be computed at runtime.

#### Signature

```
DynamicReshape(operand, dim_sizes, shape)
```

#### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor. |
| `dim_sizes` | `std::vector<XlaOp>` | Runtime-computed dimension sizes. Each element is a scalar integer tensor specifying the size of one output dimension. |
| `shape` | `Shape` | The statically known shape with symbolic dimensions (if supported by the backend) or the upper-bound shape. |

#### Semantics

Same as `Reshape` but the output dimensions are determined at runtime by `dim_sizes`. The total number of elements must still be preserved, but this is verified at runtime rather than compile time.

#### Example

```
// Reshape a tensor where the output shape depends on runtime values
%d0 = s32[] ...  // runtime-computed dimension 0
%d1 = s32[] ...  // runtime-computed dimension 1

%result = f32[?, ?] dynamic-reshape(f32[6] %operand, {%d0, %d1})
// At runtime: d0 * d1 must equal 6
```

---

## Collapse

`Collapse` collapses (merges) a range of dimensions into a single dimension.

#### Signature

```
Collapse(operand, dimensions)
```

#### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor. |
| `dimensions` | `std::vector<int64>` | A contiguous range of dimensions to collapse: `{d_start, d_start+1, ..., d_end}`. Can be specified as `{d_start, d_end}` in some APIs. |

#### Semantics

The specified range of dimensions is merged into a single dimension whose size is the product of the merged dimension sizes. The result is equivalent to a `Reshape` that combines the specified dimensions.

The dimensions must form a contiguous range (e.g., `{1, 2, 3}` is valid but `{0, 2}` is not).

#### Example

```
// f32[2, 3, 4, 5] -> f32[2, 12, 5]  (collapse dims 1 and 2)
%result = f32[2, 12, 5] collapse(f32[2, 3, 4, 5] %operand), dimensions={1, 2}

// Equivalent to:
%result = f32[2, 12, 5] reshape(f32[2, 3, 4, 5] %operand)
```

---

## Transpose

`Transpose` permutes the dimensions of a tensor according to a specified permutation.

#### Signature

```
Transpose(operand, permutation)
```

#### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor. |
| `permutation` | `std::vector<int64>` | A permutation of `[0, 1, ..., rank-1]`. The output dimension `i` corresponds to the input dimension `permutation[i]`. |

#### Semantics

For output element at index `(o0, o1, ..., on)`, the value comes from the input element at index `(o_{permutation^{-1}(0)}, o_{permutation^{-1}(1)}, ..., o_{permutation^{-1}(n)})`.

More concretely, if `permutation = [p0, p1, ..., pn]`, then:
```
output[o0, o1, ..., on] = input[o_{p0}, o_{p1}, ..., o_{pn}]
```

And the output shape is:
```
output_shape[i] = input_shape[permutation[i]]
```

#### Example: Matrix Transpose

```
// Input: f32[3, 4] (3 rows, 4 columns)
// permutation: {1, 0}
// Output: f32[4, 3] (4 rows, 3 columns)

%result = f32[4, 3] transpose(f32[3, 4] %operand), permutation={1, 0}
```

Input:
```
[[1, 2, 3, 4],
 [5, 6, 7, 8],
 [9, 10, 11, 12]]
```

Output:
```
[[1, 5, 9],
 [2, 6, 10],
 [3, 7, 11],
 [4, 8, 12]]
```

#### Example: 3D Transpose (NCHW to NHWC)

```
// Input: f32[2, 3, 8, 8] (NCHW)
// permutation: {0, 2, 3, 1}
// Output: f32[2, 8, 8, 3] (NHWC)

%result = f32[2, 8, 8, 3] transpose(f32[2, 3, 8, 8] %operand),
  permutation={0, 2, 3, 1}
```

#### HLO Text Format

```
%result = f32[4,3]{1,0} transpose(f32[3,4]{1,0} %operand), dimensions={1,0}
%result = f32[2,8,8,3]{3,2,1,0} transpose(f32[2,3,8,8]{3,2,1,0} %operand),
  dimensions={0,2,3,1}
```

#### Constraints

- `permutation` must be a valid permutation of `[0, rank)`.
- Each element of `permutation` must be unique.
- `permutation.size()` must equal `operand.rank()`.

---

## Slice and DynamicSlice

### Slice

`Slice` extracts a subarray from the input tensor using statically known start indices and limit indices.

#### Signature

```
Slice(operand, start_indices, limit_indices, strides)
```

#### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor. |
| `start_indices` | `std::vector<int64>` | The start index (inclusive) for each dimension. |
| `limit_indices` | `std::vector<int64>` | The limit index (exclusive) for each dimension. |
| `strides` | `std::vector<int64>` | The stride for each dimension. Default is all 1s. |

#### Semantics

For each dimension `i`:
```
output_size[i] = ceil((limit_indices[i] - start_indices[i]) / strides[i])
```

The output element at `(o0, ..., on)` is:
```
output[o0, ..., on] = input[start[0] + o0 * stride[0], ..., start[n] + on * stride[n]]
```

#### Examples

```
// Extract elements 1, 2, 3 from a f32[6]
%result = f32[3] slice(f32[6] %operand),
  start_indices={1}, limit_indices={4}, strides={1}

// Extract top-left 2x2 submatrix from f32[4, 4]
%result = f32[2, 2] slice(f32[4, 4] %operand),
  start_indices={0, 0}, limit_indices={2, 2}, strides={1, 1}

// Strided slice: take every other element
%result = f32[3] slice(f32[6] %operand),
  start_indices={0}, limit_indices={6}, strides={2}

// 2D strided slice
%result = f32[2, 2] slice(f32[4, 4] %operand),
  start_indices={0, 0}, limit_indices={4, 4}, strides={2, 2}
```

#### HLO Text Format

```
%result = f32[3]{0} slice(f32[6]{0} %operand),
  slice={[1:4:1]}

%result = f32[2,2]{1,0} slice(f32[4,4]{1,0} %operand),
  slice={[0:2:1], [0:2:1]}

%result = f32[3]{0} slice(f32[6]{0} %operand),
  slice={[0:6:2]}
```

---

### DynamicSlice

`DynamicSlice` extracts a subarray with **runtime-computed** start indices. The slice size is statically known.

#### Signature

```
DynamicSlice(operand, start_indices, slice_sizes)
```

#### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor. |
| `start_indices` | `std::vector<XlaOp>` | A list of scalar integer tensors, one per dimension, specifying the start index. These are runtime values. |
| `slice_sizes` | `std::vector<int64>` | The size of the slice in each dimension. Statically known. |

#### Semantics

Similar to `Slice`, but start indices are computed at runtime. If `start_indices[i]` would cause the slice to extend beyond the operand, it is clamped to ensure the slice stays within bounds.

The clamping behavior:
```
effective_start[i] = clamp(start_indices[i], 0, operand_size[i] - slice_sizes[i])
```

#### Example

```
// Dynamically slice a 4x4 matrix starting at runtime offset (r, c)
%r = s32[] ...  // runtime row start
%c = s32[] ...  // runtime column start

%result = f32[2, 2] dynamic-slice(f32[4, 4] %operand, {%r, %c}),
  dynamic_slice_sizes={2, 2}
```

If `%r = 2` and `%c = 1`, the result is the 2x2 subarray starting at row 2, column 1.

If `%r = 3` and `%c = 3`, the slice would go out of bounds. After clamping: `effective_start = (2, 2)` (clamped to `4 - 2 = 2`).

#### HLO Text Format

```
%result = f32[2,2]{1,0} dynamic-slice(f32[4,4]{1,0} %operand, s32[] %r, s32[] %c),
  dynamic_slice_sizes={2, 2}
```

---

## DynamicUpdateSlice

`DynamicUpdateSlice` produces a result that is the operand with a slice overwritten by the update values at runtime-specified start indices.

#### Signature

```
DynamicUpdateSlice(operand, update, start_indices)
```

#### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The base tensor. |
| `update` | `XlaOp` | The tensor containing the update values. Must have the same rank as `operand`. The size of `update` in each dimension must be less than or equal to the corresponding dimension of `operand`. |
| `start_indices` | `std::vector<XlaOp>` | Runtime scalar integers specifying where to place the update. One scalar per dimension. |

#### Semantics

The output is a copy of `operand` with a slice overwritten:
```
output = copy(operand)
output[start[i]:start[i]+update_size[i], ...] = update
```

Start indices are clamped similarly to `DynamicSlice`:
```
effective_start[i] = clamp(start_indices[i], 0, operand_size[i] - update_size[i])
```

#### Example

```
operand: f32[4] = [0, 0, 0, 0]
update:  f32[2] = [5, 10]
start:   s32[] = 1

%result = f32[4] dynamic-update-slice(f32[4] %operand, f32[2] %update, s32[] %start)
// Result: [0, 5, 10, 0]
```

#### Example with Clamping

```
operand: f32[4] = [1, 2, 3, 4]
update:  f32[2] = [9, 8]
start:   s32[] = 3

// effective_start = clamp(3, 0, 4-2) = clamp(3, 0, 2) = 2
// Result: [1, 2, 9, 8]
```

#### HLO Text Format

```
%result = f32[4]{0} dynamic-update-slice(f32[4]{0} %operand, f32[2]{0} %update),
  s32[] %start
```

---

## ConcatInDim

`ConcatInDim` concatenates a sequence of tensors along a specified dimension.

#### Signature

```
ConcatInDim(operands, dimension)
```

#### Arguments

| Argument | Type | Description |
|---|---|---|
| `operands` | `std::vector<XlaOp>` | A list of tensors to concatenate. All must have the same rank, and all dimensions except `dimension` must match. |
| `dimension` | `int64` | The dimension along which to concatenate. Must be in `[0, rank)`. |

#### Semantics

The tensors are joined along `dimension`. The output size along `dimension` is the sum of the input sizes along `dimension`. All other dimensions remain unchanged.

```
output_size[dimension] = sum(operand_i.size[dimension] for all i)
output_size[other_dim] = operand_0.size[other_dim]  // must be same for all
```

#### Example: 1D Concatenation

```
%a = f32[3] constant([1, 2, 3])
%b = f32[4] constant([4, 5, 6, 7])

%result = f32[7] concat(f32[3] %a, f32[4] %b), dimensions={0}
// Result: [1, 2, 3, 4, 5, 6, 7]
```

#### Example: 2D Concatenation Along Rows

```
%a = f32[2, 3] constant([[1, 2, 3], [4, 5, 6]])
%b = f32[2, 3] constant([[7, 8, 9], [10, 11, 12]])

%result = f32[4, 3] concat(f32[2, 3] %a, f32[2, 3] %b), dimensions={0}
// Result:
// [[1, 2, 3],
//  [4, 5, 6],
//  [7, 8, 9],
//  [10, 11, 12]]
```

#### Example: 2D Concatenation Along Columns

```
%result = f32[2, 6] concat(f32[2, 3] %a, f32[2, 3] %b), dimensions={1}
// Result:
// [[1, 2, 3, 7, 8, 9],
//  [4, 5, 6, 10, 11, 12]]
```

#### HLO Text Format

```
%result = f32[7]{0} concat(f32[3]{0} %a, f32[4]{0} %b), dimensions={0}
%result = f32[4,3]{1,0} concat(f32[2,3]{1,0} %a, f32[2,3]{1,0} %b), dimensions={0}
```

#### Constraints

- All operands must have the same rank.
- All non-concatenated dimensions must have the same size across operands.
- At least one operand must be provided.

---

## Broadcast and BroadcastInDim

### Broadcast

`Broadcast` adds new dimensions to a tensor by replicating its data. The sizes of the new dimensions are specified by `broadcast_sizes`.

#### Signature

```
Broadcast(operand, broadcast_sizes)
```

#### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor. |
| `broadcast_sizes` | `std::vector<int64>` | Sizes of the new dimensions to add as leading (leftmost) dimensions. |

#### Semantics

The operand is replicated along the new leading dimensions. The output shape is `broadcast_sizes ++ operand_shape` (concatenation).

For output element at index `(b0, b1, ..., bm, i0, i1, ..., in)`:
```
output[b0, b1, ..., bm, i0, i1, ..., in] = input[i0, i1, ..., in]
```

The `b0, ..., bm` indices are ignored; the input is replicated for every combination of these indices.

#### Example

```
// Broadcast a f32[3] into f32[2, 4, 3]
%result = f32[2, 4, 3] broadcast(f32[3] %operand), dimensions={2, 4}
// Each of the 2*4 = 8 positions gets the same f32[3] vector
```

#### HLO Text Format

```
%result = f32[2,4,3]{2,1,0} broadcast(f32[3]{0} %operand), sizes={2,4}
```

---

### BroadcastInDim

`BroadcastInDim` is a more flexible broadcast that allows specifying how input dimensions map to output dimensions. This enables broadcasting to a different shape where the input dimensions can appear at arbitrary positions in the output.

#### Signature

```
BroadcastInDim(operand, out_dim_bounds, broadcast_dimensions)
```

#### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor. |
| `out_dim_bounds` | `std::vector<int64>` | The desired output shape. |
| `broadcast_dimensions` | `std::vector<int64>` | Maps each input dimension to an output dimension. `broadcast_dimensions[i]` is the output dimension that corresponds to input dimension `i`. |

#### Semantics

For each input dimension `i`, the operand is aligned with output dimension `broadcast_dimensions[i]`. Input dimensions that have size 1 are broadcast to match the corresponding output dimension size. Dimensions of the output that do not correspond to any input dimension (i.e., are not in `broadcast_dimensions`) are fully broadcast (the entire input is replicated along these dimensions).

Constraints:
- `broadcast_dimensions` must be strictly increasing.
- Each `broadcast_dimensions[i]` must be in `[0, output_rank)`.
- For each input dimension `i`: `operand_shape[i] == 1` or `operand_shape[i] == out_dim_bounds[broadcast_dimensions[i]]`.

#### Example: Scalar to Tensor

```
// Broadcast scalar to f32[3, 4]
%scalar = f32[] constant(5.0)
%result = f32[3, 4] broadcast-in-dim(f32[] %scalar),
  out_dim_bounds={3, 4}, broadcast_dimensions={}
// Result: f32[3, 4] all elements = 5.0
```

#### Example: 1D to 2D (Column Vector)

```
// Input: f32[3] = [1, 2, 3]
// Map input dim 0 to output dim 0
%result = f32[3, 4] broadcast-in-dim(f32[3] %operand),
  out_dim_bounds={3, 4}, broadcast_dimensions={0}
// Result:
// [[1, 1, 1, 1],
//  [2, 2, 2, 2],
//  [3, 3, 3, 3]]
```

#### Example: 1D to 2D (Row Vector)

```
// Input: f32[4] = [1, 2, 3, 4]
// Map input dim 0 to output dim 1
%result = f32[3, 4] broadcast-in-dim(f32[4] %operand),
  out_dim_bounds={3, 4}, broadcast_dimensions={1}
// Result:
// [[1, 2, 3, 4],
//  [1, 2, 3, 4],
//  [1, 2, 3, 4]]
```

#### Example: Broadcasting a Matrix

```
// Input: f32[3, 1]
// Map input dims {0, 1} to output dims {0, 2}
%result = f32[3, 4, 1] broadcast-in-dim(f32[3, 1] %operand),
  out_dim_bounds={3, 4, 1}, broadcast_dimensions={0, 2}
// Dimension 1 of input (size 1) maps to dimension 2 of output (size 1)
// Output dimension 1 (size 4) is a fully broadcast dimension
```

#### HLO Text Format

```
%result = f32[3,4]{1,0} broadcast-in-dim(f32[3]{0} %operand),
  out_dim_bounds={3, 4}, broadcast_dimensions={0}
```

---

## Pad

`Pad` adds padding to a tensor by inserting values at the edges of each dimension and between elements (interior padding).

#### Signature

```
Pad(operand, padding_value, padding_config)
```

#### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor. |
| `padding_value` | `XlaOp` | A scalar value used for padding. Must be the same element type as the operand. |
| `padding_config` | `PaddingConfig` | Configuration specifying edge and interior padding for each dimension. |

#### PaddingConfig

`PaddingConfig` is a repeated `PaddingConfigDimension` message, one per operand dimension:

| Field | Type | Description |
|---|---|---|
| `edge_padding_low` | `int64` | Number of padding elements to add before the first element. |
| `edge_padding_high` | `int64` | Number of padding elements to add after the last element. |
| `interior_padding` | `int64` | Number of padding elements to insert between each pair of elements. |

#### Semantics

The output size for each dimension:
```
output_size[i] = operand_size[i]
               + edge_padding_low[i]
               + edge_padding_high[i]
               + max(0, operand_size[i] - 1) * interior_padding[i]
```

Edge padding can be negative, which trims elements from the operand. Interior padding must be non-negative.

#### Example: Edge Padding Only

```
// Input: f32[3] = [1, 2, 3]
// Padding: low=2, high=1, interior=0, value=0
%config = PaddingConfig({edge_padding_low=2, edge_padding_high=1, interior_padding=0})
%result = f32[6] pad(f32[3] %operand, f32[] 0.0), padding=%config
// Result: [0, 0, 1, 2, 3, 0]
```

#### Example: Interior Padding

```
// Input: f32[3] = [1, 2, 3]
// Padding: low=0, high=0, interior=1, value=0
%result = f32[5] pad(f32[3] %operand, f32[] 0.0),
  padding={{0, 0, 1}}
// Result: [1, 0, 2, 0, 3]
```

#### Example: 2D Padding

```
// Input: f32[2, 3]
// Padding: {(1, 1, 0), (0, 0, 1)}, value=0

%result = f32[4, 5] pad(f32[2, 3] %operand, f32[] 0.0),
  padding={{1, 1, 0}, {0, 0, 1}}
// Output shape: 2+1+1+0=4 x 3+0+0+2*1=5
// Row 0: [0, 0, 0, 0, 0]
// Row 1: [a, 0, b, 0, c]   (original row 0: a,b,c)
// Row 2: [d, 0, e, 0, f]   (original row 1: d,e,f)
// Row 3: [0, 0, 0, 0, 0]
```

#### Example: Negative Padding (Trimming)

```
// Input: f32[6] = [1, 2, 3, 4, 5, 6]
// Padding: low=-1, high=-2, interior=0
%result = f32[3] pad(f32[6] %operand, f32[] 0.0),
  padding={{-1, -2, 0}}
// Result: [2, 3, 4]  (trimmed 1 from start, 2 from end)
```

#### HLO Text Format

```
%result = f32[6]{0} pad(f32[3]{0} %operand, f32[] %padding_value),
  padding={2, 1, 0}
```

---

## Reverse

`Reverse` reverses the elements of a tensor along specified dimensions.

#### Signature

```
Reverse(operand, dimensions)
```

#### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor. |
| `dimensions` | `std::vector<int64>` | The dimensions along which to reverse. |

#### Semantics

For each dimension in `dimensions`, the elements are reversed (flipped) along that dimension. Other dimensions remain unchanged.

#### Example: Reverse Along One Dimension

```
// Input: f32[3] = [1, 2, 3]
%result = f32[3] reverse(f32[3] %operand), dimensions={0}
// Result: [3, 2, 1]
```

#### Example: Reverse Along One Dimension of 2D

```
// Input: f32[2, 3] = [[1, 2, 3], [4, 5, 6]]
%result = f32[2, 3] reverse(f32[2, 3] %operand), dimensions={1}
// Result: [[3, 2, 1], [6, 5, 4]]
```

#### Example: Reverse Along Multiple Dimensions

```
// Input: f32[2, 3] = [[1, 2, 3], [4, 5, 6]]
%result = f32[2, 3] reverse(f32[2, 3] %operand), dimensions={0, 1}
// Result: [[6, 5, 4], [3, 2, 1]]
```

#### HLO Text Format

```
%result = f32[2,3]{1,0} reverse(f32[2,3]{1,0} %operand), dimensions={0,1}
```

---

## Gather

`Gather` collects slices from `operand` at positions specified by `start_indices`. It is a highly configurable indexing operation that generalizes advanced indexing, gathering, and embedding lookups.

#### Signature

```
Gather(operand, start_indices, dimension_numbers, slice_sizes,
       indices_are_sorted)
```

#### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The tensor from which to gather. Shape: `[D0, D1, ..., Dn]`. |
| `start_indices` | `XlaOp` | Tensor of starting positions for each gather. Shape: `[I0, I1, ..., Im, index_vector_dim_size]`. |
| `dimension_numbers` | `GatherDimensionNumbers` | Configuration specifying the index mapping. |
| `slice_sizes` | `std::vector<int64>` | The size of the slice gathered at each position. Must have the same rank as `operand`. |
| `indices_are_sorted` | `bool` | Hint that indices are sorted, enabling optimization. Default `false`. |

#### GatherDimensionNumbers

| Field | Type | Description |
|---|---|---|
| `offset_dims` | `std::vector<int64>` | Dimensions in the output that correspond to the window (gathered slice) dimensions. These dimensions hold the gathered data. |
| `collapsed_slice_dims` | `std::vector<int64>` | Dimensions of the slice that are collapsed (must have size 1 in `slice_sizes`). These dimensions are removed from the output. |
| `start_index_map` | `std::vector<int64>` | Maps each index in the index vector to a dimension of `operand`. `start_index_map[i]` maps the i-th element of the index vector to `operand` dimension `start_index_map[i]`. |
| `index_vector_dim` | `int64` | The dimension in `start_indices` that contains the index vector. All other dimensions are "batch" dimensions. |

#### Semantics

The gather operation can be decomposed as follows:

1. **Batch dimensions**: The output has the same "batch" shape as `start_indices` minus the `index_vector_dim`. These dimensions iterate over different gather operations.

2. **Index vector**: For each position in the batch dimensions, an index vector is extracted from `start_indices`. This vector (of length `start_index_map.size()`) specifies the starting position in `operand` for the gather.

3. **Slice extraction**: A slice of size `slice_sizes` is extracted from `operand` starting at the position specified by the index vector.

4. **Collapsed dimensions**: Dimensions listed in `collapsed_slice_dims` (which must have size 1) are removed from the slice.

5. **Offset dimensions**: The remaining slice dimensions are placed into the output at positions specified by `offset_dims`.

#### Output Shape

```
output_rank = batch_dims_count + offset_dims_count
batch_dims_count = start_indices.rank - 1  (if index_vector_dim is the last dim)
offset_dims_count = operand.rank - collapsed_slice_dims.size()
```

#### Example 1: Simple 1D Gather (Embedding Lookup)

```
// operand: f32[5, 3] (5 embeddings of size 3)
// start_indices: s32[4] (4 indices)
// Gather 4 embeddings

dim_nums = GatherDimensionNumbers(
  offset_dims = {1},              // output dim 1 holds the embedding
  collapsed_slice_dims = {0},     // collapse dim 0 (must be size 1)
  start_index_map = {0},          // index maps to operand dim 0
  index_vector_dim = 1            // dim 1 of start_indices is the index
)

%result = f32[4, 3] gather(f32[5, 3] %operand, s32[4, 1] %indices),
  dim_nums=dim_nums, slice_sizes={1, 3}
```

If `start_indices = [[2], [0], [4], [1]]`:
- Output[0] = operand[2, :] (3rd embedding)
- Output[1] = operand[0, :] (1st embedding)
- Output[2] = operand[4, :] (5th embedding)
- Output[3] = operand[1, :] (2nd embedding)

#### Example 2: 2D Slice Gather

```
// operand: f32[10, 10]
// start_indices: s32[3, 2] (3 positions, each a 2D index)
// Gather 3x3 patches

dim_nums = GatherDimensionNumbers(
  offset_dims = {1, 2},
  collapsed_slice_dims = {},
  start_index_map = {0, 1},
  index_vector_dim = 1
)

%result = f32[3, 3, 3] gather(f32[10, 10] %operand, s32[3, 2] %indices),
  dim_nums=dim_nums, slice_sizes={3, 3}
```

#### Example 3: Batched Gather (Like tf.gather)

```
// operand: f32[4, 6]
// start_indices: s32[2, 3] (batch of 2, each with 3 indices)
// Gather individual elements from operand

dim_nums = GatherDimensionNumbers(
  offset_dims = {},               // no window dims (gather scalars)
  collapsed_slice_dims = {0, 1},  // both dims collapsed (size 1 each)
  start_index_map = {0},          // index maps to operand dim 0
  index_vector_dim = 1
)

// Actually, to gather along dim 0 with batch:
dim_nums = GatherDimensionNumbers(
  offset_dims = {1},              // output dim 1 = operand dim 1 (features)
  collapsed_slice_dims = {0},     // operand dim 0 collapsed
  start_index_map = {0},
  index_vector_dim = 2
)

%result = f32[2, 3, 6] gather(f32[4, 6] %operand, s32[2, 3, 1] %indices),
  dim_nums=dim_nums, slice_sizes={1, 6}
```

#### HLO Text Format

```
%result = f32[4,3]{1,0} gather(f32[5,3]{1,0} %operand, s32[4,1]{1,0} %indices),
  offset_dims={1}, collapsed_slice_dims={0},
  start_index_map={0}, index_vector_dim=1,
  slice_sizes={1, 3}, indices_are_sorted=false
```

---

## DynamicGather

`DynamicGather` is a variant of `Gather` where the slice sizes are determined at runtime rather than being statically known.

#### Signature

```
DynamicGather(operand, start_indices, slice_sizes, dimension_numbers,
              indices_are_sorted)
```

#### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The tensor from which to gather. |
| `start_indices` | `XlaOp` | Runtime-computed starting positions. |
| `slice_sizes` | `XlaOp` | A 1D integer tensor of runtime-computed slice sizes (one per operand dimension). |
| `dimension_numbers` | `GatherDimensionNumbers` | Same configuration as `Gather`. |
| `indices_are_sorted` | `bool` | Hint for optimization. |

#### Semantics

Identical to `Gather` but the slice sizes are determined at runtime. This enables gather operations where the extent of the gathered region is data-dependent.

#### Example

```
%operand = f32[10, 10] ...
%indices = s32[3, 2] ...
%slice_sz = s32[2] ...  // runtime slice sizes, e.g., [2, 3]

%result = dynamic-gather(%operand, %indices, %slice_sz),
  offset_dims={1, 2}, collapsed_slice_dims={},
  start_index_map={0, 1}, index_vector_dim=1
```

---

## BitcastConvertType

`BitcastConvertType` converts a tensor to a different element type by reinterpreting the bit pattern. No data conversion is performed; the raw bits are treated as the new type.

#### Signature

```
BitcastConvertType(operand, new_element_type)
```

#### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor. |
| `new_element_type` | `PrimitiveType` | The target element type. |

#### Semantics

Each element of the operand is reinterpreted as the new type by preserving the bit pattern. The source and target types must have the same bit width.

Common conversions:

| Source | Target | Bits | Notes |
|---|---|---|---|
| `F32` | `U32` | 32 | View float bits as unsigned int |
| `F32` | `S32` | 32 | View float bits as signed int |
| `U32` | `F32` | 32 | View unsigned int bits as float |
| `F16` | `U16` | 16 | Half-precision reinterpret |
| `BF16` | `U16` | 16 | Bfloat16 reinterpret |
| `S64` | `F64` | 64 | Signed int to double |

When converting between types of different bit widths, the last dimension may change size:
- `f32[4]` -> `u8[16]` (each f32 becomes 4 u8 values)
- `u8[4]` -> `f32[1]` (4 u8 values combine into 1 f32)

#### Example

```
// View float bits as uint32
%result = u32[4] bitcast-convert(f32[4] %operand)

// Interpret uint32 as float
%result = f32[4] bitcast-convert(u32[4] %operand)
```

#### HLO Text Format

```
%result = u32[4]{0} bitcast-convert(f32[4]{0} %operand)
```

#### Important Notes

- This is a zero-copy operation; it only changes the type interpretation.
- The bit endianness is platform-dependent.
- NaN payloads and signed zero representations are preserved exactly.

---

## ConvertElementType

`ConvertElementType` converts a tensor to a different element type using proper numeric conversion (not bit reinterpretation).

#### Signature

```
ConvertElementType(operand, new_element_type)
```

#### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor. |
| `new_element_type` | `PrimitiveType` | The target element type. |

#### Semantics

Each element is converted using standard numeric conversion rules:

- **Float to Float**: Precision may be lost (e.g., F64 -> F32). Values are rounded to the nearest representable value.
- **Float to Integer**: Truncation toward zero. Values out of range are saturating.
- **Integer to Float**: Exact for representable values, otherwise rounded.
- **Integer to Integer**: Sign-extended or truncated. Out-of-range values may saturate or wrap, depending on the backend.
- **Bool to numeric**: `true` -> `1`, `false` -> `0`.
- **Numeric to Bool**: Non-zero -> `true`, zero -> `false`.

#### Examples

```
// F32 to BF16 (lossy)
%result = bf16[4] convert(f32[4] %operand)

// F32 to S32 (truncation toward zero)
%result = s32[4] convert(f32[4] %operand)  // 3.7 -> 3, -2.9 -> -2

// S32 to F32
%result = f32[4] convert(s32[4] %operand)

// U8 to F32
%result = f32[4] convert(u8[4] %operand)
```

#### HLO Text Format

```
%result = bf16[4]{0} convert(f32[4]{0} %operand)
%result = s32[4]{0} convert(f32[4]{0} %operand)
```

#### Conversion Precision

| Conversion | Behavior |
|---|---|
| `F32 -> BF16` | Rounds to nearest even; loses ~3 decimal digits |
| `F32 -> F16` | Rounds to nearest even; loses ~3 decimal digits |
| `F64 -> F32` | Rounds to nearest even |
| `F32 -> S32` | Truncates toward zero; saturates on overflow |
| `S32 -> F32` | Exact for |x| <= 2^24; rounded otherwise |
| `S8 -> S32` | Sign-extended; exact |
| `U8 -> S32` | Zero-extended; exact |

---

## Copy

`Copy` copies its operand. It is primarily used to change the layout of a tensor (the in-memory ordering of dimensions).

### Signature

```
Copy(operand)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor. |

### Semantics

`Copy` produces a logically identical copy of the operand. When the operand and result have different layouts, the copy performs a physical data rearrangement.

If the operand and result have the same layout, the copy may be optimized away (it becomes a no-op).

### CopyStart / CopyDone Decomposition

For asynchronous execution, `Copy` decomposes into:

1. **CopyStart**: Initiates the asynchronous copy. Returns a target future.

   ```
   %target = copy-start(f32[4,4]{1,0} %operand)
   ```

2. **CopyDone**: Waits for the copy to complete. Returns the result.

   ```
   %result = copy-done(f32[4,4]{0,1} %target)
   ```

This decomposition enables overlapping copies with computation, particularly useful for host-device transfers and layout conversions.

### Example: Layout Change

```
// Copy from row-major {1,0} to column-major {0,1}
%result = f32[4,4]{0,1} copy(f32[4,4]{1,0} %operand)
```

### HLO Text Format

```
%result = f32[4,4]{0,1} copy(f32[4,4]{1,0} %operand)
```

---

## StableHLO Cross-References

| XLA Operation | StableHLO Operation | Notes |
|---|---|---|
| Reshape | `stablehlo.reshape` | Same semantics |
| DynamicReshape | `stablehlo.dynamic_reshape` | Same semantics |
| Transpose | `stablehlo.transpose` | Same semantics |
| Slice | `stablehlo.slice` | Same semantics |
| DynamicSlice | `stablehlo.dynamic_slice` | Same semantics |
| DynamicUpdateSlice | `stablehlo.dynamic_update_slice` | Same semantics |
| ConcatInDim | `stablehlo.concatenate` | Same semantics |
| Broadcast | `stablehlo.broadcast_in_dim` | When using `broadcast_dimensions` |
| Pad | `stablehlo.pad` | Same semantics |
| Reverse | `stablehlo.reverse` | Same semantics |
| Gather | `stablehlo.gather` | Same dimension numbers |
| BitcastConvertType | `stablehlo.bitcast_convert` | Same semantics |
| ConvertElementType | `stablehlo.convert` | Same semantics |
| Copy | No direct equivalent | Layout management is implicit |

### StableHLO Example: Gather

```mlir
%result = stablehlo.gather(%operand, %indices) {
  dimension_numbers = #stablehlo.gather<
    offset_dims = [1],
    collapsed_slice_dims = [0],
    start_index_map = [0],
    index_vector_dim = 1
  >,
  slice_sizes = dense<[1, 3]> : tensor<2xi64>,
  indices_are_sorted = false
} : (tensor<5x3xf32>, tensor<4x1xi32>) -> tensor<4x3xf32>
```

### StableHLO Example: BroadcastInDim

```mlir
%result = stablehlo.broadcast_in_dim(%operand) {
  broadcast_dimensions = dense<[0]> : tensor<1xi64>
} : (tensor<3xf32>) -> tensor<3x4xf32>
```

### StableHLO Example: Pad

```mlir
%result = stablehlo.pad(%operand, %padding_value) {
  edge_padding_low = dense<[1, 0]> : tensor<2xi64>,
  edge_padding_high = dense<[1, 0]> : tensor<2xi64>,
  interior_padding = dense<[0, 1]> : tensor<2xi64>
} : (tensor<2x3xf32>, tensor<f32>) -> tensor<4x5xf32>
```

### StableHLO Example: Slice

```mlir
%result = stablehlo.slice(%operand) {
  start_indices = dense<[1, 0]> : tensor<2xi64>,
  limit_indices = dense<[3, 3]> : tensor<2xi64>,
  strides = dense<[1, 1]> : tensor<2xi64>
} : (tensor<4x4xf32>) -> tensor<2x3xf32>
```

---

## Appendix: Operation Quick Reference

| Operation | Changes Shape? | Changes Data? | Key Use |
|---|---|---|---|
| Reshape | Yes | No | Redimensioning |
| DynamicReshape | Yes (runtime) | No | Dynamic shapes |
| Collapse | Yes | No | Merge dimensions |
| Transpose | Yes (permute) | Yes (reorder) | Layout conversion |
| Slice | Yes (smaller) | No (subset) | Extract subarray |
| DynamicSlice | Yes (smaller) | No (subset) | Runtime-indexed extraction |
| DynamicUpdateSlice | No | Yes (partial) | Runtime-indexed write |
| ConcatInDim | Yes (larger) | No (join) | Merge tensors |
| Broadcast | Yes (larger) | No (replicate) | Expand dimensions |
| BroadcastInDim | Yes (larger) | No (replicate) | Flexible expansion |
| Pad | Yes (larger) | Yes (fill) | Add padding |
| Reverse | No | Yes (flip) | Reverse along dim |
| Gather | Yes | No (subset) | Advanced indexing |
| DynamicGather | Yes (runtime) | No (subset) | Runtime-sized gather |
| BitcastConvertType | Maybe | No (reinterpret) | Type punning |
| ConvertElementType | No | Yes (convert) | Type conversion |
| Copy | No | No (layout only) | Layout change |
