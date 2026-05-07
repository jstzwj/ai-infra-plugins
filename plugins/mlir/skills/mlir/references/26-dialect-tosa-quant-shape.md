# MLIR TOSA, Quant & Shape Dialects

## TOSA Dialect

Tensor Operator Set Architecture - a standard set of tensor operators for ML workloads.

### Arithmetic Operations

```mlir
%r = tosa.abs %input : (tensor<10xf32>) -> tensor<10xf32>
%r = tosa.add %a, %b : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
%r = tosa.sub %a, %b : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
%r = tosa.mul %a, %b {shift = 0 : i8} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
%r = tosa.negate %input {mode = "UNKNOWN"} : (tensor<10xf32>) -> tensor<10xf32>
%r = tosa.pow %a, %b : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
%r = tosa.logical_left_shift %a, %b : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
%r = tosa.logical_right_shift %a, %b : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
%r = tosa.arithmetic_right_shift %a, %b {round = false} : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
```

### Comparison

```mlir
%r = tosa.equal %a, %b : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
%r = tosa.greater %a, %b : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
%r = tosa.greater_equal %a, %b : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
```

### Unary Math

```mlir
%r = tosa.ceil %a : (tensor<10xf32>) -> tensor<10xf32>
%r = tosa.floor %a : (tensor<10xf32>) -> tensor<10xf32>
%r = tosa.clamp %a {min_val = 0.0 : f32, max_val = 1.0 : f32} : (tensor<10xf32>) -> tensor<10xf32>
%r = tosa.exp %a : (tensor<10xf32>) -> tensor<10xf32>
%r = tosa.log %a : (tensor<10xf32>) -> tensor<10xf32>
%r = tosa.sin %a : (tensor<10xf32>) -> tensor<10xf32>
%r = tosa.cos %a : (tensor<10xf32>) -> tensor<10xf32>
%r = tosa.tanh %a : (tensor<10xf32>) -> tensor<10xf32>
%r = tosa.erf %a : (tensor<10xf32>) -> tensor<10xf32>
%r = tosa.identity %a : (tensor<10xf32>) -> tensor<10xf32>
%r = tosa.reciprocal %a : (tensor<10xf32>) -> tensor<10xf32>
%r = tosa.rsqrt %a : (tensor<10xf32>) -> tensor<10xf32>
%r = tosa.sqrt %a : (tensor<10xf32>) -> tensor<10xf32>
```

### Neural Network Operations

```mlir
// Conv2D
%r = tosa.conv2d %input, %weight, %bias
    {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1]}
    : (tensor<1x8x8x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<1x6x6x16xf32>

// Depthwise conv2d
%r = tosa.depthwise_conv2d %input, %weight, %bias
    {pad = [0, 0], stride = [1, 1], dilation = [1, 1]}
    : (...) -> (...)

// Fully connected
%r = tosa.fully_connected %input, %weight, %bias
    : (tensor<1x128xf32>, tensor<256x128xf32>, tensor<256xf32>) -> tensor<1x256xf32>

// Matmul
%r = tosa.matmul %a, %b, %c
    : (tensor<1x4x8xf32>, tensor<1x8x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xf32>

// Max pool2d
%r = tosa.max_pool2d %input
    {kernel = [2, 2], stride = [2, 2], pad = [0, 0, 0, 0]}
    : (tensor<1x8x8x3xf32>) -> tensor<1x4x4x3xf32>

// Avg pool2d
%r = tosa.avg_pool2d %input
    {kernel = [2, 2], stride = [2, 2], pad = [0, 0, 0, 0]}
    : (tensor<1x8x8x3xf32>) -> tensor<1x4x4x3xf32>

// Transpose conv2d
%r = tosa.transpose_conv2d %input, %weight, %bias
    {out_pad = [0, 0, 0, 0], stride = [2, 2]}
    : (...) -> (...)

// Reshape
%r = tosa.reshape %input {new_shape = [2, 8]} : (tensor<16xf32>) -> tensor<2x8xf32>

// Concat
%r = tosa.concat %a, %b {axis = 0} : (tensor<8xf32>, tensor<8xf32>) -> tensor<16xf32>

// Pad
%r = tosa.pad %input, %padding : (tensor<8xf32>, tensor<2x2xi32>) -> tensor<?xf32>

// Slice
%r = tosa.slice %input {start = [0], size = [4]} : (tensor<10xf32>) -> tensor<4xf32>

// Gather
%r = tosa.gather %values, %indices {axis = 0} : (tensor<10x5xf32>, tensor<3xi32>) -> tensor<3x5xf32>

// Scatter
%r = tosa.scatter %values, %indices, %input : (...) -> (...)

// Transpose
%r = tosa.transpose %input, %perms : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>

// Reduce
%r = tosa.reduce_sum %input {axis = 1} : (tensor<2x3xf32>) -> tensor<2xf32>
%r = tosa.reduce_max %input {axis = 0} : (tensor<2x3xf32>) -> tensor<3xf32>
%r = tosa.reduce_min %input {axis = 0} : (tensor<2x3xf32>) -> tensor<3xf32>
%r = tosa.reduce_prod %input {axis = 0} : (tensor<2x3xf32>) -> tensor<3xf32>

// Activation functions
%r = tosa.relu %input : (tensor<10xf32>) -> tensor<10xf32>
%r = tosa.sigmoid %input : (tensor<10xf32>) -> tensor<10xf32>
%r = tosa.tanh %input : (tensor<10xf32>) -> tensor<10xf32>

// Select
%r = tosa.select %pred, %true, %false : (tensor<10xi1>, tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>

// Conditional
%r = tosa.cond_if %pred {then_branch = @then_func, else_branch = @else_func} %input : ...

// While loop
%r = tosa.while %input {cond_branch = @cond, body_branch = @body} : ...
```

### Image Operations

```mlir
// Resize (bilinear)
%r = tosa.resize %input
    {scale = [2, 2], offset = [0, 0], border = [0, 0], mode = "BILINEAR"}
    : (tensor<1x8x8x3xf32>) -> tensor<1x16x16x3xf32>

// Reverse
%r = tosa.reverse %input {axis = 0} : (tensor<10xf32>) -> tensor<10xf32>
```

### Complete TOSA Operations

| Category | Operations |
|----------|-----------|
| Arithmetic | abs, add, sub, mul, negate, pow, logical_left_shift, logical_right_shift, arithmetic_right_shift |
| Comparison | equal, greater, greater_equal |
| Unary | ceil, floor, clamp, exp, log, sin, cos, tanh, erf, identity, reciprocal, rsqrt, sqrt |
| NN | conv2d, depthwise_conv2d, conv3d, fully_connected, matmul, max_pool2d, avg_pool2d, transpose_conv2d |
| Data | reshape, concat, pad, slice, gather, scatter, transpose, tile, reverse, resize |
| Reduction | reduce_sum, reduce_max, reduce_min, reduce_prod, reduce_any, reduce_all |
| Activation | relu, sigmoid, tanh |
| Control | select, cond_if, while, const |
| Casting | cast, rescale |
| Scatter | scatter |

## Quant Dialect

Quantization operations for ML model compression:

```mlir
// Quantize (float -> quantized)
%q = quant.qcast %input : tensor<10xf32> to tensor<10x!quant.uniform<i8:f32, 1.0:0>>

// Dequantize (quantized -> float)
%d = quant.dcast %qinput : tensor<10x!quant.uniform<i8:f32, 1.0:0>> to tensor<10xf32>

// Quantized types
!quant.uniform<i8:f32, 1.0:0>           // Uniform affine quantized
!quant.uniform<i8:f32, 1.0:128>         // With zero point
!quant.per_tensor<i8:f32, 1.0:0>        // Per-tensor
!quant.per_axis<i8:f32, {1.0, 2.0}:0>   // Per-axis
```

## Shape Dialect

Shape inference and computation:

```mlir
// Get shape
%shape = shape.shape_of %tensor : tensor<10x?xf32> -> !shape.shape

// Assuming (constraint)
%result = shape.assuming %witness {
  shape.yield %val : !shape.shape
} : !shape.shape

// Assuming all (multiple constraints)
%result = shape.assuming_all %w1, %w2, %w3 {
  shape.yield %val : !shape.shape
}

// Broadcast
%result = shape.broadcast %shape1, %shape2 : (!shape.shape, !shape.shape) -> !shape.shape

// Meet
%result = shape.meet %shape1, %shape2 : (!shape.shape, !shape.shape) -> !shape.shape

// Number of elements
%num = shape.num_elements %shape : !shape.shape -> index

// Get dimension
%dim = shape.dim %shape, %idx : (!shape.shape, index) -> index

// Reduce
shape.reduce %shape init(%init) {
  ^bb0(%dim: index, %acc: index):
    %new = arith.muli %dim, %acc : index
    shape.yield %new : index
} : !shape.shape -> index

// Equality
%eq = shape.eq %shape1, %shape2 : (!shape.shape, !shape.shape) -> i1

// Constraint (assertion)
%cstr = shape.cstr_broadcastable %shape1, %shape2 : (!shape.shape, !shape.shape)
%cstr_eq = shape.cstr_eq %shape1, %shape2 : (!shape.shape, !shape.shape)

// To extent tensor
%tensor = shape.to_extent_tensor %shape : !shape.shape -> tensor<2xindex>

// From extent tensor
%shape = shape.from_extent_tensor %tensor : tensor<2xindex> -> !shape.shape

// Value of (get runtime dimension)
%val = shape.value_of %const_shape : !shape.shape -> index

// Size to index
%idx = shape.size_to_index %val : index -> index

// Index to size
%sz = shape.index_to_size %idx : index -> !shape.size

// Join
%joined = shape.join %shape1, %shape2 : (!shape.shape, !shape.shape) -> !shape.shape

// Div
%div = shape.div %a, %b : (index, index) -> index

// Mul
%mul = shape.mul %a, %b : (index, index) -> index

// Add
%add = shape.add %a, %b : (index, index) -> index

// Any (unknown shape)
%any = shape.any : !shape.shape
```

## Shard Dialect

Tensor sharding for distributed computation:

```mlir
%sharded = tensor.empty() : tensor<100xf32>
%result = sharding.shard %tensor to %sharded : tensor<100xf32>
```
