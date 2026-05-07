# TensorFlow Lite Built-in Operators Reference

## Table of Contents

1. [Operator System Overview](#operator-system-overview)
2. [Activation Operations](#activation-operations)
3. [Arithmetic Operations](#arithmetic-operations)
4. [Array Operations](#array-operations)
5. [Convolution Operations](#convolution-operations)
6. [Pooling Operations](#pooling-operations)
7. [Reduction Operations](#reduction-operations)
8. [Neural Network Operations](#neural-network-operations)
9. [Comparison and Logic Operations](#comparison-and-logic-operations)
10. [Control Flow Operations](#control-flow-operations)
11. [Image and Resize Operations](#image-and-resize-operations)
12. [Quantization Operations](#quantization-operations)
13. [Hash Table Operations](#hash-table-operations)
13. [Other Built-in Operations](#other-built-in-operations)
14. [Custom Operations](#custom-operations)
15. [Operator Versioning](#operator-versioning)
16. [OpResolver System](#opresolver-system)

---

## Operator System Overview

### TfLiteRegistration

Every TFLite operator is represented by a `TfLiteRegistration` structure that
contains the function pointers implementing the operator's lifecycle:

```c
typedef struct TfLiteRegistration {
  // Called once per node. Returns opaque user_data.
  void* (*init)(TfLiteContext* context, const char* buffer, size_t length);

  // Called once per node to free user_data.
  void (*free)(TfLiteContext* context, void* buffer);

  // Called when inputs have been resized. Can be called multiple times.
  TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node);

  // Execute the node, reading inputs and writing outputs.
  TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node);

  // Profiling string (optional).
  const char* (*profiling_string)(const TfLiteContext* context,
                                  const TfLiteNode* node);

  // Builtin code for this op.
  int32_t builtin_code;

  // Custom op name (for custom ops).
  const char* custom_name;

  // Version of the op.
  int version;

  // In-place operation flags.
  uint64_t inplace_operator;
} TfLiteRegistration;
```

### Builtin Operator Enum

All built-in operators have unique codes defined in
`tensorflow/lite/builtin_ops.h`. The current operator list (up to code 209)
includes over 200 operators spanning:
- Basic arithmetic and activation functions
- Convolution and pooling
- Array manipulation
- Neural network layers (LSTM, RNN, etc.)
- Control flow (IF, WHILE)
- Quantization ops
- StableHLO compatibility ops

### Data Types

TFLite operators support the following data types:

```c
typedef enum {
  kTfLiteNoType = 0,
  kTfLiteFloat32 = 1,
  kTfLiteInt32 = 2,
  kTfLiteUInt8 = 3,
  kTfLiteInt64 = 4,
  kTfLiteString = 5,
  kTfLiteBool = 6,
  kTfLiteInt16 = 7,
  kTfLiteComplex64 = 8,
  kTfLiteInt8 = 9,
  kTfLiteFloat16 = 10,
  kTfLiteFloat64 = 11,
  kTfLiteComplex128 = 12,
  kTfLiteUInt64 = 13,
  kTfLiteResource = 14,
  kTfLiteVariant = 15,
  kTfLiteUInt32 = 16,
  kTfLiteUInt16 = 17,
  kTfLiteInt4 = 18,
  kTfLiteBFloat16 = 19,
} TfLiteType;
```

### Quantization Parameters

Quantized tensors carry quantization metadata:

```c
typedef struct TfLiteQuantizationParams {
  float scale;
  int32_t zero_point;
} TfLiteQuantizationParams;

// Per-channel quantization
typedef struct TfLiteAffineQuantization {
  TfLiteFloatArray* scale;
  TfLiteIntArray* zero_point;
  int32_t quantized_dimension;
} TfLiteAffineQuantization;
```

---

## Activation Operations

### RELU (kTfLiteBuiltinRelu, code 19)

Rectified Linear Unit activation: `max(0, x)`

- **Inputs**: tensor of any numeric type
- **Outputs**: same shape and type as input
- **Supported types**: FP32, FP16, INT8, UINT8, INT16
- **Quantization**: zero_point represents the 0 value
- **Version**: 1 (original), 2 (added INT8 support)

### RELU_N1_TO_1 (kTfLiteBuiltinReluN1To1, code 20)

Clipped ReLU with bounds [-1, 1]: `max(-1, min(1, x))`

- **Inputs**: tensor
- **Outputs**: same shape and type as input
- **Supported types**: FP32, FP16, INT8, UINT8
- **Quantization**: zero_point represents 0, scale determines -1 and 1 mapping

### RELU6 (kTfLiteBuiltinRelu6, code 21)

Clipped ReLU with upper bound 6: `min(max(0, x), 6)`

- **Inputs**: tensor
- **Outputs**: same shape and type as input
- **Supported types**: FP32, FP16, INT8, UINT8, INT16
- **Commonly used in**: MobileNet, EfficientNet architectures
- **Version**: 1 (original), 2 (INT8 support)

### TANH (kTfLiteBuiltinTanh, code 28)

Hyperbolic tangent: `(e^x - e^(-x)) / (e^x + e^(-x))`

- **Inputs**: tensor
- **Outputs**: same shape, values in [-1, 1]
- **Supported types**: FP32, FP16, INT8, UINT8

### LOGISTIC (kTfLiteBuiltinLogistic, code 14)

Sigmoid activation: `1 / (1 + exp(-x))`

- **Inputs**: tensor
- **Outputs**: same shape, values in [0, 1]
- **Supported types**: FP32, FP16, INT8, UINT8
- **Used for**: binary classification output, gating mechanisms

### HARD_SWISH (kTfLiteBuiltinHardSwish, code 117)

Hard Swish activation: `x * relu6(x + 3) / 6`

- **Inputs**: FP32 tensor
- **Outputs**: same shape as input
- **Supported types**: FP32, FP16, UINT8 (with specific quantization)
- **Used in**: MobileNetV3, EfficientNet-EdgeTPU

### LEAKY_RELU (kTfLiteBuiltinLeakyRelu, code 98)

Leaky ReLU: `x >= 0 ? x : alpha * x`

- **Attributes**:
  - `alpha` (float): slope for negative values (default: 0.01)
- **Inputs**: tensor
- **Outputs**: same shape and type
- **Supported types**: FP32, FP16, INT8, UINT8

### PRELU (kTfLiteBuiltinPrelu, code 54)

Parametric ReLU: `x >= 0 ? x : alpha * x` where alpha is learned

- **Inputs**:
  - Input tensor
  - Alpha tensor (slope coefficients, broadcastable to input)
- **Outputs**: same shape as input
- **Supported types**: FP32, FP16, INT8, UINT8

### ELU (kTfLiteBuiltinElu, code 111)

Exponential Linear Unit: `x >= 0 ? x : exp(x) - 1`

- **Inputs**: tensor
- **Outputs**: same shape and type
- **Supported types**: FP32, FP16, INT8, UINT8

### GELU (kTfLiteBuiltinGelu, code 150)

Gaussian Error Linear Unit approximation

- **Attributes**:
  - `approximate` (bool): use tanh approximation (default: true)
- **Inputs**: tensor
- **Outputs**: same shape and type
- **Supported types**: FP32, FP16
- **Formula (exact)**: `x * Phi(x)` where Phi is the standard normal CDF
- **Formula (approx)**: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
- **Used in**: BERT, GPT, transformer models

### RELU0_TO_1 (kTfLiteBuiltinRelu0To1, code 152)

Clipped ReLU: `min(max(0, x), 1)`

- **Inputs**: tensor
- **Outputs**: same shape and type
- **Supported types**: FP32

### SIGN (kTfLiteBuiltinSign, code 158)

Sign function: `x > 0 ? 1 : (x < 0 ? -1 : 0)`

- **Inputs**: tensor
- **Outputs**: same shape, values in {-1, 0, 1}
- **Supported types**: FP32, FP16

---

## Arithmetic Operations

### ADD (kTfLiteBuiltinAdd, code 0)

Element-wise addition: `output = input1 + input2`

- **Attributes**:
  - `fused_activation_function` (ActivationFunctionType): post-add activation
    (NONE, RELU, RELU_N1_TO_1, RELU6)
- **Inputs**:
  - input1: tensor
  - input2: tensor (must be broadcastable with input1)
- **Outputs**: tensor
- **Broadcasting**: supports NumPy-style broadcasting
- **Supported types**: FP32, FP16, INT8, UINT8, INT16, INT32, INT64
- **Quantization**: requires compatible scale and zero_point; if both inputs
  are quantized, the output quantization must be specified
- **Version**: 1 (original), 2 (INT8 support)

### SUB (kTfLiteBuiltinSub, code 41)

Element-wise subtraction: `output = input1 - input2`

- **Attributes**:
  - `fused_activation_function`: post-sub activation
- **Inputs**: input1, input2 (broadcastable)
- **Outputs**: tensor
- **Supported types**: FP32, FP16, INT8, UINT8, INT32

### MUL (kTfLiteBuiltinMul, code 18)

Element-wise multiplication: `output = input1 * input2`

- **Attributes**:
  - `fused_activation_function`: post-mul activation
- **Inputs**: input1, input2 (broadcastable)
- **Outputs**: tensor
- **Supported types**: FP32, FP16, INT8, UINT8, INT32
- **Version**: 1 (original), 2 (INT8 support), 3 (complex64)

### DIV (kTfLiteBuiltinDiv, code 42)

Element-wise division: `output = input1 / input2`

- **Attributes**:
  - `fused_activation_function`: post-div activation
- **Inputs**: input1, input2 (broadcastable)
- **Outputs**: tensor
- **Supported types**: FP32, FP16, INT8, UINT8

### POW (kTfLiteBuiltinPow, code 78)

Element-wise power: `output = input1 ^ input2`

- **Inputs**: base tensor, exponent tensor (broadcastable)
- **Outputs**: tensor
- **Supported types**: FP32

### MAX (kTfLiteBuiltinMaximum, code 55)

Element-wise maximum: `output = max(input1, input2)`

- **Inputs**: input1, input2 (broadcastable)
- **Outputs**: tensor
- **Supported types**: FP32, FP16, INT8, UINT8, INT32, INT64

### MIN (kTfLiteBuiltinMinimum, code 57)

Element-wise minimum: `output = min(input1, input2)`

- **Inputs**: input1, input2 (broadcastable)
- **Outputs**: tensor
- **Supported types**: FP32, FP16, INT8, UINT8, INT32, INT64

### SQUARED_DIFF (kTfLiteBuiltinSquaredDifference, code 99)

Element-wise squared difference: `output = (input1 - input2)^2`

- **Inputs**: input1, input2 (broadcastable)
- **Outputs**: tensor
- **Supported types**: FP32, FP16, INT8, UINT8

### FLOOR_DIV (kTfLiteBuiltinFloorDiv, code 90)

Element-wise floor division: `output = floor(input1 / input2)`

- **Inputs**: input1, input2 (broadcastable)
- **Outputs**: tensor
- **Supported types**: FP32, INT32

### FLOOR_MOD (kTfLiteBuiltinFloorMod, code 95)

Element-wise floor modulo: `output = input1 - floor_div(input1, input2) * input2`

- **Inputs**: input1, input2 (broadcastable)
- **Outputs**: tensor
- **Supported types**: FP32, INT32

### NEG (kTfLiteBuiltinNeg, code 59)

Element-wise negation: `output = -input`

- **Inputs**: tensor
- **Outputs**: same shape and type
- **Supported types**: FP32, FP16, INT8, UINT8, INT32, INT64

### ABS (kTfLiteBuiltinAbs, code 101)

Element-wise absolute value: `output = |input|`

- **Inputs**: tensor
- **Outputs**: same shape and type
- **Supported types**: FP32, FP16, INT8, UINT8, INT32, INT64

### SQUARE (kTfLiteBuiltinSquare, code 92)

Element-wise square: `output = input^2`

- **Inputs**: tensor
- **Outputs**: same shape and type
- **Supported types**: FP32, INT32

### EXP (kTfLiteBuiltinExp, code 47)

Element-wise exponential: `output = e^input`

- **Inputs**: tensor
- **Outputs**: same shape
- **Supported types**: FP32

### SQRT (kTfLiteBuiltinSqrt, code 75)

Element-wise square root: `output = sqrt(input)`

- **Inputs**: tensor
- **Outputs**: same shape
- **Supported types**: FP32

### RSQRT (kTfLiteBuiltinRsqrt, code 76)

Element-wise reciprocal square root: `output = 1 / sqrt(input)`

- **Inputs**: tensor
- **Outputs**: same shape
- **Supported types**: FP32

### LOG (kTfLiteBuiltinLog, code 73)

Element-wise natural logarithm: `output = ln(input)`

- **Inputs**: tensor
- **Outputs**: same shape
- **Supported types**: FP32

### SIN (kTfLiteBuiltinSin, code 66)

Element-wise sine: `output = sin(input)`

- **Inputs**: tensor
- **Outputs**: same shape
- **Supported types**: FP32

### COS (kTfLiteBuiltinCos, code 108)

Element-wise cosine: `output = cos(input)`

- **Inputs**: tensor
- **Outputs**: same shape
- **Supported types**: FP32

### LOGICAL_NOT (kTfLiteBuiltinLogicalNot, code 87)

Element-wise logical NOT: `output = !input`

- **Inputs**: bool tensor
- **Outputs**: bool tensor, same shape

### LOGICAL_AND (kTfLiteBuiltinLogicalAnd, code 86)

Element-wise logical AND: `output = input1 && input2`

- **Inputs**: two bool tensors (broadcastable)
- **Outputs**: bool tensor

### LOGICAL_OR (kTfLiteBuiltinLogicalOr, code 84)

Element-wise logical OR: `output = input1 || input2`

- **Inputs**: two bool tensors (broadcastable)
- **Outputs**: bool tensor

### CAST (kTfLiteBuiltinCast, code 53)

Cast tensor to a different data type

- **Attributes**: output data type (implicit from output tensor type)
- **Inputs**: tensor of any type
- **Outputs**: tensor with the target type, same shape
- **Supported conversions**: Between most numeric types

### ROUND (kTfLiteBuiltinRound, code 116)

Element-wise rounding: `output = round(input)` (banker's rounding)

- **Inputs**: tensor
- **Outputs**: same shape
- **Supported types**: FP32

### CEIL (kTfLiteBuiltinCeil, code 104)

Element-wise ceiling: `output = ceil(input)`

- **Inputs**: tensor
- **Outputs**: same shape
- **Supported types**: FP32

### FLOOR (kTfLiteBuiltinFloor, code 8)

Element-wise floor: `output = floor(input)`

- **Inputs**: tensor
- **Outputs**: same shape
- **Supported types**: FP32

### FILL (kTfLiteBuiltinFill, code 94)

Create tensor filled with a scalar value

- **Inputs**:
  - dims: 1-D int32 tensor specifying output shape
  - value: scalar value to fill with
- **Outputs**: tensor of the specified shape filled with value

### ZEROS_LIKE (kTfLiteBuiltinZerosLike, code 93)

Create tensor of zeros with same shape as input

- **Inputs**: tensor
- **Outputs**: tensor of zeros with same shape

### ATAN2 (kTfLiteBuiltinAtan2, code 156)

Element-wise arc tangent of two variables: `output = atan2(y, x)`

- **Inputs**: y tensor, x tensor (broadcastable)
- **Outputs**: tensor with arctangent values
- **Supported types**: FP32

---

## Array Operations

### RESHAPE (kTfLiteBuiltinReshape, code 22)

Reshape tensor to a new shape

- **Inputs**:
  - tensor: input tensor
  - shape: 1-D int32 target shape (can contain at most one -1)
- **Outputs**: reshaped tensor (shares data with input)
- **Notes**: Total number of elements must remain the same. A -1 dimension
  infers the size from the remaining dimensions.
- **In-place**: Yes (kTfLiteInplaceOpDataUnmodified)

### CONCATENATION (kTfLiteBuiltinConcatenation, code 2)

Concatenate tensors along a specified axis

- **Attributes**:
  - `axis` (int): dimension to concatenate along
  - `fused_activation_function`: optional post-concat activation
- **Inputs**: variable number of tensors (all same shape except concat axis)
- **Outputs**: concatenated tensor
- **Supported types**: FP32, FP16, INT8, UINT8, INT32, INT64, INT16, BOOL,
  STRING, COMPLEX64

### GATHER (kTfLiteBuiltinGather, code 36)

Gather slices from a tensor along an axis

- **Attributes**:
  - `axis` (int): dimension to gather from (default: 0)
  - `batch_dims` (int): number of batch dimensions (default: 0)
- **Inputs**:
  - params: tensor to gather from
  - indices: index tensor (INT32 or INT64)
- **Outputs**: gathered tensor
- **Supported types**: FP32, FP16, INT8, UINT8, INT32, INT64, INT16, BOOL,
  STRING, COMPLEX64

### EXPAND_DIMS (kTfLiteBuiltinExpandDims, code 70)

Insert a dimension of size 1 at the specified axis

- **Inputs**:
  - input: tensor
  - axis: int32/int64 scalar specifying where to insert the dimension
- **Outputs**: tensor with an additional dimension

### SQUEEZE (kTfLiteBuiltinSqueeze, code 43)

Remove dimensions of size 1

- **Attributes**:
  - `squeeze_dims` (list of int): specific dimensions to squeeze
- **Inputs**: tensor
- **Outputs**: tensor with specified dimensions removed

### TRANSPOSE (kTfLiteBuiltinTranspose, code 39)

Transpose tensor according to a permutation

- **Inputs**:
  - data: input tensor
  - perm: int32/int64 permutation vector
- **Outputs**: transposed tensor
- **Supported types**: FP32, FP16, INT8, UINT8, INT32, INT64, BOOL, STRING,
  COMPLEX64

### REVERSE_SEQUENCE (kTfLiteBuiltinReverseSequence, code 112)

Reverse variable-length slices along a specified dimension

- **Attributes**:
  - `seq_dim` (int): dimension to reverse along
  - `batch_dim` (int): batch dimension (default: 0)
- **Inputs**:
  - input: tensor
  - seq_lengths: 1-D int32/int64 tensor of sequence lengths
- **Outputs**: tensor with reversed sequences

### TILE (kTfLiteBuiltinTile, code 69)

Tile a tensor by repeating it along each dimension

- **Inputs**:
  - input: tensor
  - multiples: 1-D int32/int64 tensor specifying replication count per dim
- **Outputs**: tiled tensor

### PAD (kTfLiteBuiltinPad, code 34)

Pad a tensor with zeros

- **Inputs**:
  - input: tensor
  - paddings: 2-column matrix [N, 2] of int32/int64 padding amounts
- **Outputs**: padded tensor
- **Supported types**: FP32, FP16, INT32, INT8, UINT8, INT64

### PADV2 (kTfLiteBuiltinPadv2, code 60)

Pad a tensor with a constant value

- **Inputs**:
  - input: tensor
  - paddings: 2-column matrix [N, 2] of padding amounts
  - constant_value: scalar value to pad with
- **Outputs**: padded tensor

### MIRROR_PAD (kTfLiteBuiltinMirrorPad, code 100)

Pad a tensor with mirrored values

- **Attributes**:
  - `mode` (MirrorPadMode): REFLECT or SYMMETRIC
- **Inputs**:
  - input: tensor
  - paddings: 2-column matrix of padding amounts
- **Outputs**: padded tensor

### PACK (kTfLiteBuiltinPack, code 83)

Pack (stack) tensors along a new axis

- **Attributes**:
  - `values_count` (int): number of tensors to pack
  - `axis` (int): dimension to pack along
- **Inputs**: `values_count` tensors of the same shape
- **Outputs**: packed tensor

### UNPACK (kTfLiteBuiltinUnpack, code 88)

Unpack (unstack) a tensor along a dimension

- **Attributes**:
  - `num` (int): number of tensors to unpack
  - `axis` (int): dimension to unpack along
- **Inputs**: tensor
- **Outputs**: `num` tensors

### STRIDED_SLICE (kTfLiteBuiltinStridedSlice, code 45)

Extract a strided slice from a tensor

- **Attributes**:
  - `begin_mask` (int): bitmask for begin indices
  - `end_mask` (int): bitmask for end indices
  - `ellipsis_mask` (int): bitmask for ellipsis positions
  - `new_axis_mask` (int): bitmask for new axis positions
  - `shrink_axis_mask` (int): bitmask for shrinking axes
- **Inputs**:
  - input: tensor
  - begin: 1-D int32/int64 tensor of start indices
  - end: 1-D int32/int64 tensor of end indices
  - strides: 1-D int32/int64 tensor of strides
- **Outputs**: sliced tensor

### SPLIT (kTfLiteBuiltinSplit, code 49)

Split a tensor along a dimension into sub-tensors

- **Attributes**:
  - `num_splits` (int): number of output tensors
- **Inputs**:
  - axis: int32 scalar
  - input: tensor
- **Outputs**: `num_splits` tensors

### SPLIT_V (kTfLiteBuiltinSplitV, code 102)

Split a tensor with explicit size splits

- **Inputs**:
  - input: tensor
  - size_splits: 1-D int32/int64 tensor of split sizes (can contain -1)
  - axis: int32 scalar
- **Outputs**: variable number of tensors

### SELECT (kTfLiteBuiltinSelect, code 64)

Element-wise conditional selection: `output = condition ? true_val : false_val`

- **Inputs**:
  - condition: bool tensor
  - true_value: tensor (broadcastable)
  - false_value: tensor (broadcastable)
- **Outputs**: selected tensor

### SELECT_V2 (kTfLiteBuiltinSelectV2, code 123)

Same as SELECT with improved broadcasting support

### BROADCAST_TO (kTfLiteBuiltinBroadcastTo, code 130)

Broadcast tensor to a given shape

- **Inputs**:
  - input: tensor
  - shape: 1-D int32/int64 target shape
- **Outputs**: broadcasted tensor

### WHERE (kTfLiteBuiltinWhere, code 109)

Return coordinates of non-zero / true elements

- **Inputs**: bool tensor (condition)
- **Outputs**: 2-D int64 tensor of coordinates [num_true, rank]

### SLICE (kTfLiteBuiltinSlice, code 65)

Extract a slice from a tensor

- **Inputs**:
  - input: tensor
  - begin: 1-D int32/int64 start indices
  - size: 1-D int32/int64 sizes (-1 means all remaining)
- **Outputs**: sliced tensor

### SHAPE (kTfLiteBuiltinShape, code 77)

Return the shape of a tensor

- **Attributes**:
  - `out_type` (int): output type (INT32 or INT64, default: INT32)
- **Inputs**: tensor
- **Outputs**: 1-D int32 or int64 tensor of shape dimensions

### RANK (kTfLiteBuiltinRank, code 110)

Return the rank (number of dimensions) of a tensor

- **Inputs**: tensor
- **Outputs**: int32 scalar

### SIZE (related to shape ops)

Return the total number of elements in a tensor

### BATCH_TO_SPACE_ND (kTfLiteBuiltinBatchToSpaceNd, code 37)

Rearrange data from batch into spatial dimensions

- **Inputs**:
  - input: N-D tensor
  - block_shape: 1-D int32 tensor
  - crops: 2-D int32 tensor [M, 2]
- **Outputs**: rearranged tensor

### SPACE_TO_BATCH_ND (kTfLiteBuiltinSpaceToBatchNd, code 38)

Rearrange data from spatial into batch dimension

- **Inputs**:
  - input: N-D tensor
  - block_shape: 1-D int32 tensor
  - paddings: 2-D int32 tensor [M, 2]
- **Outputs**: rearranged tensor

### DEPTH_TO_SPACE (kTfLiteBuiltinDepthToSpace, code 5)

Rearrange data from depth into spatial blocks

- **Attributes**:
  - `block_size` (int): size of spatial blocks
- **Inputs**: 4-D tensor [batch, height, width, depth]
- **Outputs**: 4-D tensor [batch, height*block_size, width*block_size, depth/(block_size^2)]

### SPACE_TO_DEPTH (kTfLiteBuiltinSpaceToDepth, code 26)

Rearrange data from spatial blocks into depth

- **Attributes**:
  - `block_size` (int): size of spatial blocks
- **Inputs**: 4-D tensor
- **Outputs**: 4-D tensor with rearranged dimensions

### REVERSE_V2 (kTfLiteBuiltinReverseV2, code 105)

Reverse tensor along specified axes

- **Inputs**:
  - input: tensor
  - axes: 1-D int32 tensor of axes to reverse
- **Outputs**: reversed tensor

### UNIQUE (kTfLiteBuiltinUnique, code 103)

Find unique elements in a 1-D tensor

- **Attributes**:
  - `idx_out_type` (int): INT32 or INT64 for output indices
- **Inputs**: 1-D tensor
- **Outputs**:
  - unique values tensor
  - indices tensor mapping input to unique values

### GATHER_ND (kTfLiteBuiltinGatherNd, code 107)

Gather slices from a tensor using N-D indices

- **Inputs**:
  - params: tensor to gather from
  - indices: N-D index tensor
- **Outputs**: gathered tensor

### SCATTER_ND (kTfLiteBuiltinScatterNd, code 122)

Scatter updates into a tensor using N-D indices

- **Inputs**:
  - indices: N-D index tensor
  - updates: tensor of values to scatter
  - shape: 1-D int32/int64 tensor of output shape
- **Outputs**: tensor with scattered values

### ONE_HOT (kTfLiteBuiltinOneHot, code 85)

Convert indices to one-hot encoding

- **Attributes**: (none in flatbuffer, values passed as inputs)
- **Inputs**:
  - indices: int32 tensor of indices
  - depth: int32 scalar
  - on_value: scalar value for "on" positions
  - off_value: scalar value for "off" positions
- **Attributes**:
  - `axis` (int): dimension for the new one-hot dimension (default: -1)
- **Outputs**: one-hot encoded tensor

### RANGE (kTfLiteBuiltinRange, code 96)

Generate a sequence of numbers

- **Inputs**:
  - start: scalar
  - limit: scalar
  - delta: scalar (step)
- **Outputs**: 1-D tensor of the sequence

---

## Convolution Operations

### CONV_2D (kTfLiteBuiltinConv2d, code 3)

2D convolution

- **Attributes**:
  - `padding` (PaddingType): SAME or VALID
  - `stride_w` (int): horizontal stride
  - `stride_h` (int): vertical stride
  - `dilation_w_factor` (int): horizontal dilation (default: 1)
  - `dilation_h_factor` (int): vertical dilation (default: 1)
  - `fused_activation_function`: post-conv activation
- **Inputs**:
  - input: 4-D tensor [batch, height, width, channels]
  - filter: 4-D tensor [out_channels, filter_h, filter_w, in_channels]
  - bias (optional): 1-D tensor [out_channels]
- **Outputs**: 4-D tensor [batch, out_h, out_w, out_channels]
- **Weight formats**: Standard (HWIO). For per-channel quantization, filter
  must have per-channel scale and zero_point arrays.
- **Supported types**: FP32, FP16, INT8 (per-tensor and per-channel), UINT8
- **Version**: 1 (original), 2 (dilation), 3 (INT8 per-channel)

### DEPTHWISE_CONV_2D (kTfLiteBuiltinDepthwiseConv2d, code 4)

Depthwise separable 2D convolution

- **Attributes**: Same as CONV_2D plus:
  - `depth_multiplier` (int): number of output channels per input channel
- **Inputs**:
  - input: 4-D tensor [batch, height, width, channels]
  - filter: 4-D tensor [1, filter_h, filter_w, channels * depth_multiplier]
  - bias (optional): 1-D tensor [channels * depth_multiplier]
- **Outputs**: 4-D tensor
- **Supported types**: FP32, FP16, INT8, UINT8

### TRANSPOSE_CONV (kTfLiteBuiltinTransposeConv, code 67)

Transposed (deconvolution) 2D convolution

- **Attributes**:
  - `padding` (PaddingType): SAME or VALID
  - `stride_w`, `stride_h` (int): strides
- **Inputs**:
  - output_shape: 1-D int32 tensor [batch, height, width, channels]
  - filter: 4-D tensor [out_channels, filter_h, filter_w, in_channels]
  - input: 4-D tensor
- **Outputs**: 4-D tensor
- **Supported types**: FP32, FP16, INT8

### CONV_3D (kTfLiteBuiltinConv3d, code 132)

3D convolution

- **Attributes**:
  - `padding` (PaddingType): SAME or VALID
  - `stride_d`, `stride_h`, `stride_w` (int)
  - `dilation_d_factor`, `dilation_h_factor`, `dilation_w_factor` (int)
- **Inputs**:
  - input: 5-D tensor [batch, depth, height, width, channels]
  - filter: 5-D tensor
  - bias (optional)
- **Outputs**: 5-D tensor

### CONV_3D_TRANSPOSE (kTfLiteBuiltinConv3dTranspose, code 141)

Transposed 3D convolution

- Similar to CONV_3D but with output shape specification

---

## Pooling Operations

### AVERAGE_POOL_2D (kTfLiteBuiltinAveragePool2d, code 1)

2D average pooling

- **Attributes**:
  - `padding` (PaddingType): SAME or VALID
  - `stride_w`, `stride_h` (int)
  - `filter_width`, `filter_height` (int): pooling window size
  - `fused_activation_function`: post-pool activation
- **Inputs**: 4-D tensor [batch, height, width, channels]
- **Outputs**: 4-D tensor [batch, out_h, out_w, channels]
- **Supported types**: FP32, FP16, INT8, UINT8

### MAX_POOL_2D (kTfLiteBuiltinMaxPool2d, code 17)

2D max pooling

- **Attributes**: Same as AVERAGE_POOL_2D
- **Inputs**: 4-D tensor
- **Outputs**: 4-D tensor
- **Supported types**: FP32, FP16, INT8, UINT8

### L2_POOL_2D (kTfLiteBuiltinL2Pool2d, code 12)

2D L2 pooling: `sqrt(sum(x^2) / N)` over the pooling window

- **Attributes**: Same as AVERAGE_POOL_2D
- **Inputs**: 4-D tensor
- **Outputs**: 4-D tensor
- **Supported types**: FP32

---

## Reduction Operations

### MEAN (kTfLiteBuiltinMean, code 40)

Compute the mean of elements along specified axes

- **Attributes**:
  - `keep_dims` (bool): whether to retain reduced dimensions
- **Inputs**:
  - input: tensor
  - axis: 1-D int32 tensor of reduction axes
- **Outputs**: reduced tensor
- **Supported types**: FP32, FP16, INT8, UINT8, INT32

### REDUCE_SUM (kTfLiteBuiltinSum, code 74)

Sum of elements along axes

- **Attributes**: `keep_dims`
- **Inputs**: input, axis
- **Outputs**: reduced tensor
- **Supported types**: FP32, INT32

### REDUCE_PROD (kTfLiteBuiltinReduceProd, code 81)

Product of elements along axes

- **Attributes**: `keep_dims`
- **Inputs**: input, axis
- **Outputs**: reduced tensor
- **Supported types**: FP32, INT32

### REDUCE_MAX (kTfLiteBuiltinReduceMax, code 82)

Maximum of elements along axes

- **Attributes**: `keep_dims`
- **Inputs**: input, axis
- **Outputs**: reduced tensor
- **Supported types**: FP32, INT8, UINT8, INT32

### REDUCE_MIN (kTfLiteBuiltinReduceMin, code 89)

Minimum of elements along axes

- **Attributes**: `keep_dims`
- **Inputs**: input, axis
- **Outputs**: reduced tensor
- **Supported types**: FP32, INT8, UINT8, INT32

### REDUCE_ANY (kTfLiteBuiltinReduceAny, code 91)

Logical OR of elements along axes

- **Attributes**: `keep_dims`
- **Inputs**: bool tensor, axis
- **Outputs**: bool tensor

### REDUCE_ALL (kTfLiteBuiltinReduceAll, code 140)

Logical AND of elements along axes

- **Attributes**: `keep_dims`
- **Inputs**: bool tensor, axis
- **Outputs**: bool tensor

---

## Neural Network Operations

### FULLY_CONNECTED (kTfLiteBuiltinFullyConnected, code 9)

Fully connected (dense) layer: `output = activation(input * weights^T + bias)`

- **Attributes**:
  - `fused_activation_function`: post-FC activation
  - `weights_format`: format of weights tensor
    - `DEFAULT` (0): weights shape [num_units, input_size]
    - `SHUFFLED4x16INT8` (1): shuffled for optimized INT8 inference
  - `keep_num_dims` (bool): preserve input rank (default: false)
  - `asymmetric_quantize_inputs` (bool): dynamically quantize inputs (default: false)
- **Inputs**:
  - input: tensor (2-D or higher)
  - weights: 2-D tensor [num_units, input_size]
  - bias (optional): 1-D tensor [num_units]
- **Outputs**: tensor
- **Supported types**: FP32, FP16, INT8, UINT8, INT4
- **Version**: 1-7 (various quantization and format improvements)
- **Quantization details**:
  - INT8 per-tensor quantization: input and weights both quantized
  - INT8 hybrid: weights quantized, input stays FP32 (dynamic quantization)
  - INT8 per-channel: per-channel scale for weights

### SOFTMAX (kTfLiteBuiltinSoftmax, code 25)

Softmax activation: `exp(x_i) / sum(exp(x_j))`

- **Attributes**:
  - `beta` (float): temperature parameter (default: 1.0)
- **Inputs**: tensor
- **Outputs**: same shape tensor with softmax probabilities
- **Supported types**: FP32, FP16, INT8, UINT8
- **Quantization**: For quantized types, the output is quantized to [0, 255]
  with zero_point at 0 (unsigned) or specific mapping

### LOG_SOFTMAX (kTfLiteBuiltinLogSoftmax, code 50)

Log-softmax: `log(softmax(x))` computed in a numerically stable manner

- **Inputs**: tensor
- **Outputs**: same shape tensor
- **Supported types**: FP32

### L2_NORMALIZATION (kTfLiteBuiltinL2Normalization, code 11)

L2 normalize along the last dimension

- **Attributes**:
  - `fused_activation_function`: optional post-norm activation
- **Inputs**: tensor
- **Outputs**: L2-normalized tensor
- **Supported types**: FP32, INT8, UINT8

### LOCAL_RESPONSE_NORMALIZATION (kTfLiteBuiltinLocalResponseNormalization, code 13)

Local Response Normalization across channels

- **Attributes**: (stored in builtin_data)
  - `radius` (int): number of adjacent channels to normalize over
  - `bias` (float): offset (usually 1.0)
  - `alpha` (float): scale factor
  - `beta` (float): exponent
- **Inputs**: 4-D tensor
- **Outputs**: normalized 4-D tensor

### LSTM (kTfLiteBuiltinLstm, code 16)

Standard LSTM layer (unrolled by time step)

- **Attributes** (TfLiteLSTMParams):
  - `activation` (ActivationFunctionType): activation for gate outputs
  - `cell_clip` (float): clipping value for cell state
  - `proj_clip` (float): clipping value for projection
  - `kernel_type` (LSTMKernelType): `FULL_KERNEL` or `BASIC_KERNEL`
  - `asymmetric_quantize_inputs` (bool): dynamic input quantization
- **Inputs** (up to 24 tensors for full LSTM):
  - Input tensor
  - Input-to-input weights, Input-to-forget weights, Input-to-cell weights,
    Input-to-output weights
  - Recurrent-to-input weights, Recurrent-to-forget, Recurrent-to-cell,
    Recurrent-to-output
  - Peephole weights (optional): cell-to-input, cell-to-forget, cell-to-output
  - Layer norm weights (optional): input, forget, cell, output
  - Bias vectors: input, forget, cell, output, projection
  - State tensors: activation state, cell state
- **Outputs**:
  - Output tensor
  - Updated activation state (if variable)
  - Updated cell state (if variable)

### UNIDIRECTIONAL_SEQUENCE_LSTM (kTfLiteBuiltinUnidirectionalSequenceLstm, code 44)

Sequence LSTM that processes an entire time sequence

- Similar parameters to LSTM but operates on a sequence
- Inputs include a time-major or batch-major sequence
- Outputs the full output sequence

### BIDIRECTIONAL_SEQUENCE_LSTM (kTfLiteBuiltinBidirectionalSequenceLstm, code 52)

Bidirectional LSTM processing sequences in both forward and backward directions

- Contains two sets of LSTM weights (forward and backward)
- Merges outputs based on merge mode

### UNIDIRECTIONAL_SEQUENCE_RNN (kTfLiteBuiltinUnidirectionalSequenceRnn, code 35)

Sequence RNN (simpler than LSTM)

### BIDIRECTIONAL_SEQUENCE_RNN (kTfLiteBuiltinBidirectionalSequenceRnn, code 46)

Bidirectional sequence RNN

### EMBEDDING_LOOKUP (kTfLiteBuiltinEmbeddingLookup, code 7)

Look up embeddings from a lookup table

- **Inputs**:
  - lookup: int32 tensor of indices
  - value: 2-D embedding matrix [vocab_size, embedding_dim]
- **Outputs**: embedded tensor [num_indices, embedding_dim]
- **Supported types**: FP32, INT8 (hybrid quantization)

### EMBEDDING_LOOKUP_SPARSE (kTfLiteBuiltinEmbeddingLookupSparse, code 33)

Sparse embedding lookup with weighted aggregation

- **Inputs**:
  - embeddings: 2-D matrix
  - indices: sparse indices
  - weights: aggregation weights
- **Outputs**: aggregated embedding tensor

### SVDF (kTfLiteBuiltinSvdf, code 27)

Singular Value Decomposition Filter op

- Used for keyword spotting models
- Combines a static rank-1 approximation with recurrent state

### LSH_PROJECTION (kTfLiteBuiltinLshProjection, code 15)

Locality-Sensitive Hashing projection

### TOPK_V2 (kTfLiteBuiltinTopkV2, code 48)

Find top-K elements and their indices

- **Inputs**:
  - input: tensor
  - k: int32 scalar
- **Outputs**:
  - values: top-K values
  - indices: top-K indices (int32)

### ARG_MAX (kTfLiteBuiltinArgMax, code 56)

Return the index of the maximum value along an axis

- **Inputs**: input tensor, axis (int32/int64 scalar)
- **Outputs**: index tensor
- **Output type**: INT32 or INT64 based on output_index_type attribute

### ARG_MIN (kTfLiteBuiltinArgMin, code 79)

Return the index of the minimum value along an axis

### BATCH_MATMUL (kTfLiteBuiltinBatchMatmul, code 126)

Batched matrix multiplication

- **Attributes**:
  - `adj_x` (bool): transpose input1 (default: false)
  - `adj_y` (bool): transpose input2 (default: false)
- **Inputs**:
  - input1: tensor [..., M, K]
  - input2: tensor [..., K, N]
- **Outputs**: tensor [..., M, N]
- **Supported types**: FP32, FP16, INT8

---

## Comparison and Logic Operations

### EQUAL (kTfLiteBuiltinEqual, code 71)

Element-wise equality: `output = (input1 == input2)`

- **Inputs**: two tensors (broadcastable)
- **Outputs**: bool tensor
- **Supported types**: FP32, FP16, INT32, INT64, INT8, UINT8, STRING, BOOL

### NOT_EQUAL (kTfLiteBuiltinNotEqual, code 72)

Element-wise inequality: `output = (input1 != input2)`

- **Inputs**: two tensors (broadcastable)
- **Outputs**: bool tensor

### GREATER (kTfLiteBuiltinGreater, code 61)

Element-wise greater than: `output = (input1 > input2)`

- **Inputs**: two tensors (broadcastable)
- **Outputs**: bool tensor

### GREATER_EQUAL (kTfLiteBuiltinGreaterEqual, code 62)

Element-wise greater-or-equal: `output = (input1 >= input2)`

- **Inputs**: two tensors (broadcastable)
- **Outputs**: bool tensor

### LESS (kTfLiteBuiltinLess, code 58)

Element-wise less than: `output = (input1 < input2)`

- **Inputs**: two tensors (broadcastable)
- **Outputs**: bool tensor

### LESS_EQUAL (kTfLiteBuiltinLessEqual, code 63)

Element-wise less-or-equal: `output = (input1 <= input2)`

- **Inputs**: two tensors (broadcastable)
- **Outputs**: bool tensor

---

## Control Flow Operations

### IF (kTfLiteBuiltinIf, code 118)

Conditional execution

- **Inputs**:
  - condition: bool scalar
  - args: variable-length list of input tensors
- **Attributes** (in builtin_data):
  - `then_subgraph_index` (int): subgraph index for true branch
  - `else_subgraph_index` (int): subgraph index for false branch
- **Outputs**: output tensors from the executed branch

### WHILE (kTfLiteBuiltinWhile, code 119)

Loop execution

- **Inputs**: loop variable tensors
- **Attributes**:
  - `cond_subgraph_index` (int): subgraph for loop condition
  - `body_subgraph_index` (int): subgraph for loop body
- **Outputs**: final loop variable tensors
- **Execution**: repeatedly executes body subgraph while cond subgraph returns true

### CALL_ONCE (kTfLiteBuiltinCallOnce, code 129)

Execute a subgraph exactly once (initialization pattern)

- **Attributes**:
  - `init_subgraph_index` (int): subgraph to call once
- **Used for**: initializing stateful variables, random seeds

---

## Image and Resize Operations

### RESIZE_BILINEAR (kTfLiteBuiltinResizeBilinear, code 23)

Resize images using bilinear interpolation

- **Attributes**:
  - `align_corners` (bool): align corner pixels
  - `half_pixel_centers` (bool): use half-pixel-centered coordinates
- **Inputs**:
  - input: 4-D tensor [batch, height, width, channels]
  - size: 1-D int32 tensor [new_height, new_width]
- **Outputs**: 4-D resized tensor

### RESIZE_NEAREST_NEIGHBOR (kTfLiteBuiltinResizeNearestNeighbor, code 97)

Resize using nearest neighbor interpolation

- **Attributes**:
  - `align_corners` (bool)
  - `half_pixel_centers` (bool)
- **Inputs**: input, size
- **Outputs**: resized tensor

---

## Quantization Operations

### DEQUANTIZE (kTfLiteBuiltinDequantize, code 6)

Convert quantized tensor to FP32

- **Inputs**: quantized tensor (INT8, UINT8, INT16, FP16)
- **Outputs**: FP32 tensor
- **Formula**: `float_value = scale * (quantized_value - zero_point)`
- **Version**: 1 (per-tensor), 2 (per-channel), 3 (FP16)

### QUANTIZE (kTfLiteBuiltinQuantize, code 114)

Convert FP32 tensor to quantized

- **Inputs**: FP32 tensor
- **Outputs**: quantized tensor (type determined by output tensor)
- **Supported output types**: INT8, UINT8, INT16, FP16

### FAKE_QUANT (kTfLiteBuiltinFakeQuant, code 80)

Fake quantization for quantization-aware training

- **Attributes**:
  - `min` (float): minimum quantized value
  - `max` (float): maximum quantized value
  - `num_bits` (int): quantization bit width (default: 8)
  - `narrow_range` (bool): narrow range quantization
- **Inputs**: FP32 tensor
- **Outputs**: FP32 tensor (with fake-quantized values)

---

## Hash Table Operations

### HASHTABLE (kTfLiteBuiltinHashtable, code 136)

Create a hash table resource

### HASHTABLE_FIND (kTfLiteBuiltinHashtableFind, code 137)

Look up values in a hash table

### HASHTABLE_IMPORT (kTfLiteBuiltinHashtableImport, code 138)

Import key-value pairs into a hash table

### HASHTABLE_SIZE (kTfLiteBuiltinHashtableSize, code 139)

Return the number of entries in a hash table

---

## Other Built-in Operations

### Variable Operations

- **VAR_HANDLE** (code 142): Create a variable resource handle
- **READ_VARIABLE** (code 143): Read from a variable
- **ASSIGN_VARIABLE** (code 144): Write to a variable

### Sparse Operations

- **DENSIFY** (code 124): Convert sparse tensor to dense
- **SPARSE_TO_DENSE** (code 68): Convert sparse representation to dense tensor

### Segment Operations

- **SEGMENT_SUM** (code 125): Unsorted segment sum
- **UNSORTED_SEGMENT_PROD** (code 153): Unsorted segment product
- **UNSORTED_SEGMENT_MAX** (code 154): Unsorted segment maximum
- **UNSORTED_SEGMENT_SUM** (code 155): Unsorted segment sum
- **UNSORTED_SEGMENT_MIN** (code 157): Unsorted segment minimum

### Complex Number Operations

- **IMAG** (code 133): Extract imaginary part
- **REAL** (code 134): Extract real part
- **COMPLEX_ABS** (code 135): Absolute value of complex tensor
- **RFFT2D** (code 131): 2D real-valued Fast Fourier Transform

### Matrix Operations

- **MATRIX_DIAG** (code 113): Create diagonal matrix from vector
- **MATRIX_SET_DIAG** (code 115): Set diagonal of matrix

### Other Operations

- **ADD_N** (code 106): Element-wise sum of N tensors
- **CUMSUM** (code 128): Cumulative sum along axis
- **BUCKETIZE** (code 147): Bucketize values based on boundaries
- **NON_MAX_SUPPRESSION_V4** (code 120): NMS algorithm
- **NON_MAX_SUPPRESSION_V5** (code 121): NMS with scores
- **DYNAMIC_UPDATE_SLICE** (code 151): Update slice of tensor
- **BITCAST** (code 159): Bitcast tensor to different type
- **BITWISE_XOR** (code 160): Bitwise XOR
- **RIGHT_SHIFT** (code 161): Right bit shift
- **DILATE** (code 203): Tensor dilation

### Random Operations

- **RANDOM_STANDARD_NORMAL** (code 146): Generate random normal values
- **RANDOM_UNIFORM** (code 148): Generate random uniform values
- **MULTINOMIAL** (code 149): Sample from multinomial distribution

---

## Custom Operations

### Registration

Custom operations are registered using the `OpResolver`:

```c++
class MyOpResolver : public tflite::MutableOpResolver {
 public:
  MyOpResolver() {
    // Register built-in ops
    AddBuiltin(BuiltinOperator_ADD, Register_ADD());
    // Register custom op
    AddCustom("MyCustomOp", GetMyCustomOpRegistration());
  }
};
```

### Custom Op Implementation

```c++
// Registration structure
TfLiteRegistration* GetMyCustomOpRegistration() {
  static TfLiteRegistration reg = {
      .init = MyCustomInit,
      .free = MyCustomFree,
      .prepare = MyCustomPrepare,
      .invoke = MyCustomInvoke,
      .builtin_code = kTfLiteBuiltinCustom,
      .custom_name = "MyCustomOp",
      .version = 1,
  };
  return &reg;
}

void* MyCustomInit(TfLiteContext* context, const char* buffer,
                   size_t length) {
  // Parse custom op parameters from buffer
  auto* params = new MyCustomParams();
  // ... parse ...
  return params;
}

void MyCustomFree(TfLiteContext* context, void* buffer) {
  delete static_cast<MyCustomParams*>(buffer);
}

TfLiteStatus MyCustomPrepare(TfLiteContext* context, TfLiteNode* node) {
  // Set output tensor shape based on input shapes
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
      context->GetTensor(context, node->outputs->data[0], &output));

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context,
      context->GetTensor(context, node->inputs->data[0], &input));

  // Set output shape
  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input->dims);
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus MyCustomInvoke(TfLiteContext* context, TfLiteNode* node) {
  // Execute the operation
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  // ... computation ...

  return kTfLiteOk;
}
```

### Using Custom Ops in Model Conversion

```python
import tensorflow as tf

# Define custom op in TF
@tf.function
def my_custom_op(x):
    # Define the op using TF operations
    return tf.raw_ops.MyCustomOp(input=x)

# Convert with custom ops
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [my_custom_op.get_concrete_function(tf.TensorSpec([1, 10], tf.float32))]
)
converter.allow_custom_ops = True
tflite_model = converter.convert()
```

---

## Operator Versioning

### Version Numbers

Each operator has a version number that indicates the minimum TFLite runtime
version required. The versioning system ensures forward compatibility:

- **Version 1**: Original implementation
- **Version 2+**: Added features (e.g., new data types, attributes)

When a model is loaded, the runtime checks that it supports the required op
versions. If a newer version is required, the runtime will return an error.

### Version Compatibility

```
Model op versions <= Runtime op versions => Compatible
Model op versions > Runtime op versions => Incompatible (error)
```

### Adding New Op Versions

When adding a new feature to an existing op:

1. Increment the version in `TfLiteRegistration::version`.
2. Update the op's `Prepare` or `Invoke` to handle the new version.
3. Update the converter to produce the new version for models that use the
   new feature.
4. Document the version requirements.

### Op Version Examples

| Operator | Version 1 | Version 2 | Version 3 |
|---|---|---|---|
| ADD | FP32 only | INT8 support | Complex64 |
| CONV_2D | Basic | Dilation | Per-channel INT8 |
| FULLY_CONNECTED | FP32 | INT8 | INT4, dynamic quant |
| MUL | FP32 | INT8 | Complex64 |

---

## OpResolver System

### Built-in OpResolver

```c++
// Register all built-in ops
tflite::ops::builtin::BuiltinOpResolver resolver;
```

### MutableOpResolver

```c++
class MyResolver : public tflite::MutableOpResolver {
 public:
  MyResolver() {
    // Add specific built-in ops
    AddBuiltin(tflite::BuiltinOperator_CONV_2D,
               tflite::ops::builtin::Register_CONV_2D());
    AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
               tflite::ops::builtin::Register_FULLY_CONNECTED());
    // Add custom op
    AddCustom("MyOp", Register_MyOp());
  }
};
```

### OpResolver Methods

```c++
class OpResolver {
 public:
  // Find registration for a built-in op
  virtual const TfLiteRegistration* FindOp(tflite::BuiltinOperator op,
                                           int version) const = 0;

  // Find registration for a custom op
  virtual const TfLiteRegistration* FindOp(const char* op,
                                           int version) const = 0;
};
```

### Minimal OpResolver for Embedded

For resource-constrained environments, only register the ops needed by the
specific model:

```c++
class MinimalOpResolver : public tflite::MutableOpResolver {
 public:
  MinimalOpResolver() {
    AddBuiltin(tflite::BuiltinOperator_ADD,
               tflite::Register_ADD());
    AddBuiltin(tflite::BuiltinOperator_MUL,
               tflite::Register_MUL());
    AddBuiltin(tflite::BuiltinOperator_RESHAPE,
               tflite::Register_RESHAPE());
    AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
               tflite::Register_SOFTMAX());
    // Only add what your model needs
  }
};
```

---

## StableHLO Operations

TFLite includes a set of StableHLO (Stable High-Level Operations) compatibility
ops (codes 162-209). These enable running StableHLO programs through the TFLite
runtime:

- StableHloLogistic, StablehloAdd, StablehloDivide, StablehloMultiply
- StablehloMaximum, StablehloReshape, StablehloClamp, StablehloConcatenate
- StablehloBroadcastInDim, StablehloConvolution, StablehloSlice
- StablehloCustomCall, StablehloReduce, StablehloAbs, StablehloAnd
- StablehloCosine, StablehloExponential, StablehloFloor, StablehloLog
- StablehloMinimum, StablehloNegate, StablehloOr, StablehloPower
- StablehloRemainder, StablehloRsqrt, StablehloSelect, StablehloSubtract
- StablehloTanh, StablehloScatter, StablehloCompare, StablehloConvert
- StablehloDynamicSlice, StablehloDynamicUpdateSlice, StablehloPad
- StablehloIota, StablehloDotGeneral, StablehloReduceWindow
- StablehloSort, StablehloWhile, StablehloGather, StablehloTranspose
- StablehloRngBitGenerator, StablehloComposite, StablehloShiftLeft
- StablehloCbrt, StablehloCase

These operations provide a bridge between the XLA/StableHLO ecosystem and
TFLite runtime, enabling models compiled with XLA to run on TFLite-supported
hardware.

---

## Summary

TFLite provides over 200 built-in operators covering the full range of
functionality needed for machine learning inference:

1. **Core math**: ADD, SUB, MUL, DIV and element-wise operations with
   broadcasting support.
2. **Neural network layers**: CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED,
   LSTM variants.
3. **Array manipulation**: RESHAPE, CONCATENATION, GATHER, SCATTER_ND,
   STRIDED_SLICE.
4. **Activation functions**: RELU variants, SIGMOID, TANH, HARD_SWISH, GELU.
5. **Control flow**: IF, WHILE, CALL_ONCE enabling dynamic computation.
6. **Quantization**: DEQUANTIZE, QUANTIZE, FAKE_QUANT for efficient inference.
7. **Custom ops**: Full support via OpResolver and custom registrations.
8. **StableHLO**: Bridge to XLA ecosystem for broader model compatibility.

Each operation is versioned for forward compatibility, and the OpResolver
system allows selective registration for minimal binary size on embedded
platforms.
