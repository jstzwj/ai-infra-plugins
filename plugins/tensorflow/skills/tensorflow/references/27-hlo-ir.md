# XLA HLO IR (High-Level Optimizer IR) Reference

This document provides a comprehensive reference for XLA's HLO (High-Level
Optimizer) intermediate representation. HLO is the core IR used by XLA for
optimization and code generation.

## Table of Contents

1. [HLO IR Overview](#hlo-ir-overview)
2. [Shape and Layout](#shape-and-layout)
3. [HloModule](#hlomodule)
4. [HloComputation](#hlocomputation)
5. [HloInstruction](#hloinstruction)
6. [Data Movement Instructions](#data-movement-instructions)
7. [Arithmetic Instructions](#arithmetic-instructions)
8. [Comparison Instructions](#comparison-instructions)
9. [Reduction Instructions](#reduction-instructions)
10. [Convolution Instructions](#convolution-instructions)
11. [Control Flow Instructions](#control-flow-instructions)
12. [Sort Instructions](#sort-instructions)
13. [Random Number Generation](#random-number-generation)
14. [DotGeneral Instruction](#dotgeneral-instruction)
15. [Collective Operations](#collective-operations)
16. [Layout and Tiling](#layout-and-tiling)
17. [HLO Text Format](#hlo-text-format)
18. [HloPass Interface](#hlopass-interface)
19. [HLO Verification](#hlo-verification)

---

## HLO IR Overview

HLO (High-Level Optimizer) is XLA's intermediate representation. It operates
at a level above LLVM IR but below TensorFlow's graph representation. HLO
represents computations as dataflow graphs of typed instructions.

### HLO Hierarchy

```
HloModule (compilation unit)
  |-- HloComputation (function)
  |     |-- HloInstruction (operation)
  |     |     |-- operands (inputs)
  |     |     |-- shape (output type)
  |     |     |-- attributes (op-specific config)
  |     |-- ...
  |-- HloComputation (called computation)
  |-- ...
```

### Key Properties

1. **Typed**: Every instruction has a known output Shape
2. **SSA**: Each instruction produces a value; values are never mutated
3. **Explicit Layout**: Memory layout is part of the type system
4. **Side Effects**: Some instructions (Rng, Infeed, Outfeed) have side effects
5. **Fusion**: Multiple instructions can be grouped into fusion nodes

---

## Shape and Layout

### Shape

Shapes describe the type, dimensions, and layout of HLO values.

```
// Array shapes
f32[128]                  // 1D array of 128 floats
f32[128,64]               // 2D array (128x64 matrix)
bf16[1,224,224,3]         // 4D batch-NHWC image

// Tuple shapes
(f32[128], s32[])          // Tuple of array and scalar

// Token shape
token                      // Used for sequencing side-effecting ops

// Dynamic shapes
f32[<=128,64]              // First dimension is dynamic (max 128)
```

### ShapeProto

From `xla/xla_data.proto`:

```protobuf
message ShapeProto {
  PrimitiveType element_type = 2;
  repeated int64 dimensions = 3;
  repeated bool is_dynamic_dimension = 6;
  repeated ShapeProto tuple_shapes = 4;
  LayoutProto layout = 5;
}
```

### Layout

Layout describes memory ordering:

```
f32[128,64]{0,1}          // Dim 0 is minor (fastest varying)
f32[128,64]{1,0}          // Dim 1 is minor (column-major-like)
f32[128,64]{0,1}(2,4)     // Tiled layout with 2x4 tiles
```

### LayoutProto

```protobuf
message LayoutProto {
  repeated int64 minor_to_major = 1;
  repeated DimLevelType dim_level_types = 9;
  repeated TileProto tiles = 6;
  int64 element_size_in_bits = 7;
  int64 memory_space = 8;
  int64 tail_padding_alignment_in_elements = 16;
  repeated SplitConfigProto split_configs = 17;
}
```

### TileProto

```protobuf
message TileProto {
  repeated int64 dimensions = 1;
}
```

Tiles describe block layouts where data is organized in small blocks for
better cache behavior:

```
// 2D tiled layout with 8x16 tiles
f32[128,256]{1,0}(8,16)
// Data is stored as: tile[0,0] tile[0,1] ... tile[0,15] tile[1,0] ...
```

### PrimitiveType

All supported data types in HLO:

| Type | Name | Description |
|------|------|-------------|
| 1 | `PRED` | Boolean (1 bit) |
| 2 | `S8` | Signed 8-bit integer |
| 3 | `S16` | Signed 16-bit integer |
| 4 | `S32` | Signed 32-bit integer |
| 5 | `S64` | Signed 64-bit integer |
| 6 | `U8` | Unsigned 8-bit integer |
| 7 | `U16` | Unsigned 16-bit integer |
| 8 | `U32` | Unsigned 32-bit integer |
| 9 | `U64` | Unsigned 64-bit integer |
| 10 | `F16` | IEEE 16-bit float |
| 11 | `F32` | IEEE 32-bit float |
| 12 | `F64` | IEEE 64-bit float |
| 16 | `BF16` | Brain float 16-bit |
| 15 | `C64` | Complex (2xF32) |
| 18 | `C128` | Complex (2xF64) |
| 19 | `F8E5M2` | FP8: 5 exp, 2 mantissa |
| 20 | `F8E4M3FN` | FP8: 4 exp, 3 mantissa, finite+NaN |
| 23 | `F8E4M3B11FNUZ` | FP8: 4 exp, 3 mantissa, bias=11 |
| 24 | `F8E5M2FNUZ` | FP8: 5 exp, 2 mantissa, unsigned zero |
| 25 | `F8E4M3FNUZ` | FP8: 4 exp, 3 mantissa, unsigned zero |
| 28 | `F8E4M3` | FP8: 4 exp, 3 mantissa (IEEE-like) |
| 29 | `F8E3M4` | FP8: 3 exp, 4 mantissa |
| 32 | `F4E2M1FN` | FP4: 2 exp, 1 mantissa, finite |
| 33 | `F8E8M0FNU` | FP8: 8 exp, 0 mantissa, finite |
| 13 | `TUPLE` | Tuple type |
| 14 | `OPAQUE_TYPE` | Opaque type |
| 17 | `TOKEN` | Token type |
| 34 | `BUFFER` | Buffer type |

Additional sub-byte types: `S1`, `S2`, `S4`, `U1`, `U2`, `U4`

---

## HloModule

The top-level compilation unit. Contains the entry computation and
all called computations.

### HloModule Properties

- **Name**: Unique identifier for the module
- **Entry Computation**: The main computation to execute
- **Config**: Compilation parameters (replica count, debug options, etc.)
- **Program Shape**: Parameter and result types

### HloModuleConfig

```cpp
struct HloModuleConfig {
  int64_t replica_count = 1;
  int64_t num_partitions = 1;
  DebugOptions debug_options;
  std::optional<ProgramShape> entry_computation_layout;
  bool has_static_device_assignment = false;
  DeviceAssignment device_assignment;
};
```

### Example HloModule

```
HloModule my_module

ENTRY main {
  p0 = f32[128,64] parameter(0)
  p1 = f32[64,32] parameter(1)
  dot = f32[128,32] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT result = f32[128,32] copy(dot)
}
```

---

## HloComputation

A function-level unit containing a dataflow graph of HLO instructions.

### Properties

- **Name**: Unique computation name
- **Instructions**: List of all instructions in the computation
- **Root Instruction**: The instruction producing the computation result
- **Parameters**: Input parameter instructions
- **Parent Module**: The containing HloModule

### Computation Types

1. **Entry Computation**: The main computation invoked by the executable
2. **Called Computation**: Invoked by `Call`, `While`, `Conditional`, `Map`, etc.
3. **Reduce Computation**: Applied by `Reduce` and `ReduceWindow`
4. **Select Computation**: Applied by `SelectAndScatter`
5. **Sort Computation**: Comparator for `Sort`

---

## HloInstruction

The fundamental operation in HLO. Each instruction has:
- A unique name within its computation
- An opcode (operation type)
- Operands (input instructions)
- A result shape
- Optional attributes

### HloInstruction Categories

| Category | Opcodes |
|----------|---------|
| Data movement | Parameter, Constant, Tuple, GetTupleElement, Reshape, Transpose, Broadcast, Pad, Slice, DynamicSlice, DynamicUpdateSlice, Concatenate, Reverse, Gather, Scatter, Copy, BitcastConvert, Convert |
| Arithmetic | Add, Sub, Mul, Div, Rem, Negate, Exp, Log, Sqrt, Power, Floor, Ceil, Abs, Sign, Min, Max, Clamp, Not, And, Or, Xor, ShiftLeft, ShiftRightArithmetic, ShiftRightLogical |
| Comparison | Compare |
| Reduction | Reduce, ReduceWindow, SelectAndScatter, ReduceScatter |
| Convolution | Convolution, CustomCall |
| Control flow | Call, While, Conditional, Select, TupleSelect |
| Sort | Sort, TopK |
| RNG | Rng, RngBitGenerator, RngGetAndUpdateState |
| Dot | DotGeneral |
| Collective | AllReduce, AllGather, CollectivePermute, CollectiveBroadcast, ReduceScatter |
| Other | Infeed, Outfeed, Send, Recv, CustomCall, AfterAll, Token, OptBarrier |

---

## Data Movement Instructions

### Parameter

Reads an input argument of the computation.

```
%p0 = f32[128,64] parameter(0)
```

| Attribute | Description |
|-----------|-------------|
| Parameter number | Index of the parameter (0-based) |

### Constant

Produces a constant literal value.

```
%c0 = f32[] constant(1.0)
%c1 = s32[3] constant({1, 2, 3})
```

### Tuple / GetTupleElement

Creates and accesses tuple values.

```
%tuple = (f32[10], s32[]) tuple(%x, %y)
%elem = f32[10] get-tuple-element(%tuple), index=0
```

### Reshape

Changes the shape of a tensor without modifying data.

```
%x = f32[2,3,4] ...
%y = f32[6,4] reshape(%x)
```

The total number of elements must remain the same. A single dimension may be
-1 for inference.

### Transpose

Permutes the dimensions of a tensor.

```
%x = f32[128,64] ...
%y = f32[64,128] transpose(%x), dimensions={1,0}
```

| Attribute | Description |
|-----------|-------------|
| dimensions | Permutation of dimension indices |

### Broadcast

Broadcasts a tensor along specified dimensions.

```
%x = f32[64] ...
%y = f32[128,64] broadcast(%x), dimensions={1}
```

| Attribute | Description |
|-----------|-------------|
| broadcast_dimensions | Mapping from source to destination dimensions |

### Pad

Pads a tensor with a value.

```
%x = f32[10,20] ...
%val = f32[] constant(0.0)
%y = f32[14,24] pad(%x, %val), padding=2_2x2_2
```

Uses `PaddingConfig`:
```protobuf
message PaddingConfig {
  message PaddingConfigDimension {
    int64 edge_padding_low = 1;
    int64 edge_padding_high = 2;
    int64 interior_padding = 3;
  }
  repeated PaddingConfigDimension dimensions = 1;
}
```

### Slice

Extracts a sub-tensor with static bounds.

```
%x = f32[10,20] ...
%y = f32[5,10] slice(%x), slice={[2:7, 5:15]}
```

### DynamicSlice

Extracts a sub-tensor with dynamic start indices.

```
%x = f32[10,20] ...
%start = s32[2] ...
%y = f32[3,5] dynamic-slice(%x, %start), dynamic_slice_sizes={3,5}
```

### DynamicUpdateSlice

Writes a sub-tensor at dynamic positions.

```
%x = f32[10,20] ...
%update = f32[3,5] ...
%start = s32[2] ...
%y = f32[10,20] dynamic-update-slice(%x, %update, %start)
```

### Concatenate

Concatenates tensors along a dimension.

```
%x = f32[10,20] ...
%y = f32[5,20] ...
%z = f32[15,20] concatenate(%x, %y), dimensions={0}
```

### Reverse

Reverses tensor along specified dimensions.

```
%x = f32[10,20] ...
%y = f32[10,20] reverse(%x), dimensions={0,1}
```

### Gather

Gathers elements from a tensor using an index tensor.

```
%operand = f32[10,20] ...
%indices = s32[5,1] ...
%result = f32[5,3] gather(%operand, %indices),
  offset_dims={1},
  collapsed_slice_dims={0},
  start_index_map={0},
  index_vector_dim=1,
  slice_sizes={1,3}
```

Uses `GatherDimensionNumbers`:
```protobuf
message GatherDimensionNumbers {
  repeated int64 offset_dims = 1;
  repeated int64 collapsed_slice_dims = 2;
  repeated int64 start_index_map = 3;
  int64 index_vector_dim = 4;
  repeated int64 operand_batching_dims = 5;
  repeated int64 start_indices_batching_dims = 6;
}
```

### Scatter

Scatters values into a tensor using an index tensor.

```
%operand = f32[10,20] ...
%indices = s32[5,1] ...
%updates = f32[5,3] ...
%result = f32[10,20] scatter(%operand, %indices, %updates),
  to_apply=add_computation,
  update_window_dims={1},
  inserted_window_dims={0},
  scatter_dims_to_operand_dims={0},
  index_vector_dim=1
```

### Copy

Copies a tensor (may change layout).

```
%y = f32[128,64] copy(%x)
```

### BitcastConvert

Reinterprets the bit pattern of data as a different type.

```
%x = f32[10] ...
%y = u32[10] bitcast-convert(%x)
```

### Convert

Converts data between types (with value conversion).

```
%x = f32[10] ...
%y = bf16[10] convert(%x)
```

---

## Arithmetic Instructions

### Element-wise Binary Operations

| Opcode | Syntax | Description |
|--------|--------|-------------|
| `Add` | `add(%a, %b)` | Element-wise addition |
| `Sub` | `subtract(%a, %b)` | Element-wise subtraction |
| `Mul` | `multiply(%a, %b)` | Element-wise multiplication |
| `Div` | `divide(%a, %b)` | Element-wise division |
| `Rem` | `remainder(%a, %b)` | Element-wise remainder |
| `Min` | `minimum(%a, %b)` | Element-wise minimum |
| `Max` | `maximum(%a, %b)` | Element-wise maximum |
| `And` | `and(%a, %b)` | Bitwise AND (integer/pred) |
| `Or` | `or(%a, %b)` | Bitwise OR (integer/pred) |
| `Xor` | `xor(%a, %b)` | Bitwise XOR (integer/pred) |
| `ShiftLeft` | `shift-left(%a, %b)` | Left shift |
| `ShiftRightArithmetic` | `shift-right-arithmetic(%a, %b)` | Arithmetic right shift (sign-extending) |
| `ShiftRightLogical` | `shift-right-logical(%a, %b)` | Logical right shift (zero-extending) |
| `Power` | `power(%a, %b)` | Element-wise power |
| `Atan2` | `atan2(%a, %b)` | Element-wise atan2 |

### Element-wise Unary Operations

| Opcode | Syntax | Description |
|--------|--------|-------------|
| `Negate` | `negate(%x)` | Negation (-x) |
| `Exp` | `exp(%x)` | Exponential (e^x) |
| `Expm1` | `expm1(%x)` | Exponential minus 1 (e^x - 1) |
| `Log` | `log(%x)` | Natural logarithm |
| `Log1p` | `log1p(%x)` | Log of (1+x) |
| `Sqrt` | `sqrt(%x)` | Square root |
| `Rsqrt` | `rsqrt(%x)` | Reciprocal square root (1/sqrt(x)) |
| `Cbrt` | `cbrt(%x)` | Cube root |
| `Sin` | `sin(%x)` | Sine |
| `Cos` | `cos(%x)` | Cosine |
| `Tan` | `tan(%x)` | Tangent |
| `Sinh` | `sinh(%x)` | Hyperbolic sine |
| `Cosh` | `cosh(%x)` | Hyperbolic cosine |
| `Tanh` | `tanh(%x)` | Hyperbolic tangent |
| `Asin` | `asin(%x)` | Arc sine |
| `Acos` | `acos(%x)` | Arc cosine |
| `Atan` | `atan(%x)` | Arc tangent |
| `Floor` | `floor(%x)` | Floor |
| `Ceil` | `ceil(%x)` | Ceiling |
| `Round` | `round(%x)` | Round to nearest even |
| `RoundNearestEven` | `round-nearest-even(%x)` | Round to nearest even |
| `Abs` | `abs(%x)` | Absolute value |
| `Sign` | `sign(%x)` | Sign function (-1, 0, or 1) |
| `Not` | `not(%x)` | Bitwise NOT |
| `Clz` | `clz(%x)` | Count leading zeros |
| `Ctz` | `ctz(%x)` | Count trailing zeros |
| `PopulationCount` | `popcount(%x)` | Population count |
| `Logistic` | `logistic(%x)` | Logistic (sigmoid: 1/(1+e^-x)) |
| `Erf` | `erf(%x)` | Error function |
| `Erfc` | `erfc(%x)` | Complementary error function |
| `IsFinite` | `is-finite(%x)` | Check if finite (result: pred) |
| `Real` | `real(%x)` | Real part of complex |
| `Imag` | `imag(%x)` | Imaginary part of complex |
| `Complex` | `complex(%re, %im)` | Construct complex number |
| `Conj` | `conj(%x)` | Complex conjugate |
| `CountLeadingZeros` | `count-leading-zeros(%x)` | Count leading zeros |

### Clamp

Clamps values between a minimum and maximum.

```
%result = f32[10] clamp(%min, %x, %max)
```

Broadcasting is applied: each operand can be a scalar broadcast to the shape.

---

## Comparison Instructions

### Compare

Compares two tensors element-wise.

```
%result = pred[10] compare(%a, %b), direction=EQ
%result = pred[10] compare(%a, %b), direction=GT
```

| Direction | Meaning |
|-----------|---------|
| `EQ` | Equal |
| `NE` | Not equal |
| `LT` | Less than |
| `LE` | Less than or equal |
| `GT` | Greater than |
| `GE` | Greater than or equal |

Result type is always `pred` (boolean). Floating-point comparisons follow
IEEE 754 semantics (NaN comparisons return false).

---

## Reduction Instructions

### Reduce

Reduces a tensor along specified dimensions using a reduction function.

```
%input = f32[128,64] ...
%init = f32[] constant(0.0)
%result = f32[64] reduce(%input, %init), dimensions={0}, to_apply=add_scalar

// add_scalar computation:
add_scalar(a, b) -> a + b
```

| Attribute | Description |
|-----------|-------------|
| dimensions | Dimensions to reduce over |
| to_apply | Reduction computation (binary function) |

### ReduceWindow

Reduces over a sliding window.

```
%input = f32[10,10] ...
%init = f32[] constant(0.0)
%result = f32[8,8] reduce-window(%input, %init), window={size=3x3 stride=1x1 pad=0_0x0_0}, to_apply=add
```

Uses Window configuration:
```
Window {
  size: 3x3
  stride: 1x1
  padding_low: 0x0
  padding_high: 0x0
  window_dilation: 1x1
  base_dilation: 1x1
}
```

### SelectAndScatter

Selects elements and scatters values (used for max pooling backprop).

```
%operand = f32[10,10] ...
%source = f32[5,5] ...
%init = f32[] constant(-inf)
%result = f32[10,10] select-and-scatter(%operand, %source, %init),
  window={size=2x2 stride=2x2},
  select=ge_computation,
  scatter=add_computation
```

### ReduceScatter

Reduces then scatters chunks across replicas.

```
%input = f32[100] ...
%result = f32[50] reduce-scatter(%input), dimensions={0}, to_apply=add,
  replica_groups={{0,1}}
```

---

## Convolution Instructions

### Convolution

General n-dimensional convolution.

```
%lhs = f32[1,224,224,3] ...
%rhs = f32[3,3,3,64] ...
%result = f32[1,112,112,64] convolution(%lhs, %rhs),
  window={size=3x3 stride=2x2 pad=1_1x1_1},
  dim_labels=b01f_01io_b01f,
  feature_group_count=1,
  batch_group_count=1
```

**Dimension labels format**: `input_dims_kernel_dims_output_dims`
- `b` = batch, `f` = feature, `0-9` = spatial dimensions
- `i` = input features (kernel), `o` = output features (kernel)

**Example dimension labels**:
- `b01f_01io_b01f`: NHWC * HWIO -> NHWC (standard conv)
- `bf01_01oi_bf01`: NCHW * OIHW -> NCHW

### ConvolutionDimensionNumbers

```protobuf
message ConvolutionDimensionNumbers {
  int64 input_batch_dimension = 7;
  int64 input_feature_dimension = 8;
  repeated int64 input_spatial_dimensions = 11;
  int64 kernel_input_feature_dimension = 3;
  int64 kernel_output_feature_dimension = 4;
  repeated int64 kernel_spatial_dimensions = 6;
  int64 output_batch_dimension = 9;
  int64 output_feature_dimension = 10;
  repeated int64 output_spatial_dimensions = 12;
}
```

### Feature Groups

- `feature_group_count=1`: Standard convolution
- `feature_group_count=N`: Depthwise convolution (N = input features)
- `batch_group_count=N`: Grouped convolution

---

## Control Flow Instructions

### Call

Invokes another computation.

```
%result = f32[10] call(%x), to_apply=my_function
```

### While

Executes a body computation repeatedly while a condition is true.

```
%initial = (s32[], f32[10]) tuple(%i, %state)
%result = (s32[], f32[10]) while(%initial), condition=loop_cond, body=loop_body

// loop_cond: (s32[], f32[10]) -> pred[]
// loop_body: (s32[], f32[10]) -> (s32[], f32[10])
```

### Conditional

Branches based on a predicate.

```
%pred = pred[] ...
%true_val = f32[10] ...
%false_val = f32[10] ...
%result = f32[10] conditional(%pred, %true_val, %false_val),
  true_computation=true_branch,
  false_computation=false_branch
```

### Select

Element-wise conditional selection.

```
%pred = pred[10] ...
%on_true = f32[10] ...
%on_false = f32[10] ...
%result = f32[10] select(%pred, %on_true, %on_false)
```

### TupleSelect

Selects one of two tuple values based on a predicate.

```
%pred = pred[] ...
%on_true = (f32[10], s32[]) ...
%on_false = (f32[10], s32[]) ...
%result = (f32[10], s32[]) tuple-select(%pred, %on_true, %on_false)
```

---

## Sort Instructions

### Sort

Sorts a tensor along a dimension.

```
%input = f32[10,20] ...
%result = f32[10,20] sort(%input), dimension=1, is_stable=true,
  to_apply=compare_computation
```

| Attribute | Description |
|-----------|-------------|
| dimension | Dimension to sort along |
| is_stable | Whether to use stable sort |
| to_apply | Comparator computation (returns pred) |

Multiple operands can be sorted jointly:
```
%keys = f32[10,20] ...
%values = s32[10,20] ...
%result = (f32[10,20], s32[10,20]) sort(%keys, %values),
  dimension=1, to_apply=compare_lt
```

### TopK

Finds the top K elements along the last dimension.

```
%input = f32[10,100] ...
%values = f32[10,10] ...
%indices = s32[10,10] ...
%result = (f32[10,10], s32[10,10]) top-k(%input), k=10
```

---

## Random Number Generation

### Rng

Generates random numbers with specified distribution.

```
%a = f32[] constant(0.0)
%b = f32[] constant(1.0)
%result = f32[100,50] rng(%a, %b), distribution=uniform, shape=f32[100,50]
```

| Distribution | Description |
|-------------|-------------|
| `uniform` | Uniform on [a, b) |
| `normal` | Normal with mean a, stddev b |

### RngBitGenerator

Generates random bits using a specific algorithm.

```
%state = u32[4] ...
%result = (u32[4], u32[100,50]) rng-bit-generator(%state), algorithm=philox
```

| Algorithm | Description |
|-----------|-------------|
| `rng_default` | Backend-dependent default |
| `rng_three_fry` | ThreeFry counter-based PRNG |
| `rng_philox` | Philox counter-based PRNG |

### RngGetAndUpdateState

Gets and updates the global PRNG state.

```
%state = u32[4] rng-get-and-update-state()
```

---

## DotGeneral Instruction

Generalized matrix multiplication supporting batch and contracting dimensions.

```
%lhs = f32[2,128,64] ...
%rhs = f32[2,64,32] ...
%result = f32[2,128,32] dot(%lhs, %rhs),
  lhs_batch_dims={0}, rhs_batch_dims={0},
  lhs_contracting_dims={2}, rhs_contracting_dims={1}
```

### DotDimensionNumbers

```protobuf
message DotDimensionNumbers {
  repeated int64 lhs_contracting_dimensions = 1;
  repeated int64 rhs_contracting_dimensions = 2;
  repeated int64 lhs_batch_dimensions = 3;
  repeated int64 rhs_batch_dimensions = 4;
}
```

### Dot Variants

| Pattern | Description |
|---------|-------------|
| Vector dot | `f32[N] dot f32[N]` -> `f32[]` |
| Matrix multiply | `f32[M,K] dot f32[K,N]` -> `f32[M,N]` |
| Batched matmul | `f32[B,M,K] dot f32[B,K,N]` -> `f32[B,M,N]` |
| Batched + contract | Multiple batch and contracting dims |

### Precision Config

```
%dot = f32[128,32] dot(%lhs, %rhs), precision_config={HIGH,HIGH},
  lhs_contracting_dims={1}, rhs_contracting_dims={0}
```

Precision levels:
- `DEFAULT`: Backend default
- `HIGH`: Higher precision (slower)
- `HIGHEST`: Highest precision available

---

## Collective Operations

### AllReduce

Reduces values across all replicas.

```
%input = f32[100] ...
%result = f32[100] all-reduce(%input), to_apply=add,
  replica_groups={{0,1,2,3}},
  channel_id=1,
  use_global_device_ids=false
```

### AllGather

Gathers values from all replicas along a dimension.

```
%input = f32[25] ...
%result = f32[100] all-gather(%input), dimensions={0},
  replica_groups={{0,1,2,3}},
  channel_id=2
```

### CollectivePermute

Permutes data across replicas according to a source-target mapping.

```
%input = f32[100] ...
%result = f32[100] collective-permute(%input),
  source_target_pairs={{0,1}, {1,2}, {2,3}, {3,0}}
```

### CollectiveBroadcast

Broadcasts from one replica to all others.

```
%input = f32[100] ...
%result = f32[100] collective-broadcast(%input),
  replica_groups={{0,1,2,3}}
```

### CollectiveOpGroupMode

| Mode | Condition | Description |
|------|-----------|-------------|
| `CROSS_REPLICA` | No channel_id | Groups by replica within partition |
| `CROSS_PARTITION` | channel_id, no use_global_device_ids | Groups by partition within replica |
| `CROSS_REPLICA_AND_PARTITION` | channel_id, use_global_device_ids=false | All replicas, all partitions |
| `FLATTENED_ID` | channel_id, use_global_device_ids=true | Custom device ID groups |

---

## Layout and Tiling

### Layout Assignment

The layout assignment pass determines the optimal memory layout for each
instruction:

```
Before: f32[128,64]{0,1}  // Default: dim 0 minor
After:  f32[128,64]{1,0}  // Optimized: dim 1 minor for dot
```

### Tile Layout

Tiled layouts improve memory access patterns:

```
f32[128,64]{0,1}(2,8)  // 2x8 tiles, dim 0 minor within tile
```

TileProto:
```protobuf
message TileProto {
  repeated int64 dimensions = 1;
}
```

### SplitConfig

For multi-memory architectures:

```protobuf
message SplitConfigProto {
  int64 dimension = 1;
  repeated int64 split_indices = 2;
}
```

---

## HLO Text Format

### Complete Example

```
HloModule ResNet_Block

// Convolution + BatchNorm + ReLU
add_scalar.1 (a: f32[], b: f32[]) -> f32[] {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

max_scalar (a: f32[], b: f32[]) -> f32[] {
  a = f32[] parameter(0)
  b = f32[] parameter(0)
  ROOT max = f32[] maximum(a, b)
}

ENTRY main {
  // Inputs
  p0 = f32[1,224,224,3] parameter(0)               // Input image
  p1 = f32[3,3,3,64] parameter(1)                   // Conv kernel
  p2 = f32[64] parameter(2)                          // Bias

  // Convolution
  conv = f32[1,112,112,64] convolution(p0, p1),
    window={size=3x3 stride=2x2 pad=1_1x1_1},
    dim_labels=b01f_01io_b01f

  // Bias addition
  bias = f32[1,112,112,64] broadcast(p2), dimensions={3}
  add = f32[1,112,112,64] add(conv, bias)

  // ReLU
  zero = f32[] constant(0.0)
  bcast_zero = f32[1,112,112,64] broadcast(zero), dimensions={}
  ROOT relu = f32[1,112,112,64] maximum(add, bcast_zero)
}
```

### While Loop Example

```
HloModule WhileLoop

body (param: (s32[], f32[10])) -> (s32[], f32[10]) {
  %param = (s32[], f32[10]) parameter(0)
  %i = s32[] get-tuple-element(%param), index=0
  %data = f32[10] get-tuple-element(%param), index=1
  %one = s32[] constant(1)
  %new_i = s32[] add(%i, %one)
  %factor = f32[] constant(0.9)
  %bcast = f32[10] broadcast(%factor), dimensions={}
  %new_data = f32[10] multiply(%data, %bcast)
  ROOT result = (s32[], f32[10]) tuple(%new_i, %new_data)
}

condition (param: (s32[], f32[10])) -> pred[] {
  %param = (s32[], f32[10]) parameter(0)
  %i = s32[] get-tuple-element(%param), index=0
  %limit = s32[] constant(100)
  ROOT cmp = pred[] compare(%i, %limit), direction=LT
}

ENTRY main {
  %i0 = s32[] constant(0)
  %data0 = f32[10] ...
  %init = (s32[], f32[10]) tuple(%i0, %data0)
  ROOT %while = (s32[], f32[10]) while(%init),
    condition=condition, body=body
}
```

### Fusion Example

```
HloModule FusionExample

fused_computation (p0: f32[128]) -> f32[128] {
  %p0 = f32[128] parameter(0)
  %c = f32[] constant(1.0)
  %bcast = f32[128] broadcast(%c), dimensions={}
  %add = f32[128] add(%p0, %bcast)
  ROOT %relu = f32[128] maximum(%add, %bcast)
}

ENTRY main {
  %p0 = f32[128] parameter(0)
  ROOT %fusion = f32[128] fusion(%p0), kind=kLoop,
    calls=fused_computation
}
```

### Fusion Kinds

| Kind | Description |
|------|-------------|
| `kLoop` | Loop fusion (element-wise ops) |
| `kInput` | Input fusion (reduce from fusible producer) |
| `kOutput` | Output fusion (reduce into fusible consumer) |
| `kCustom` | Custom fusion (backend-specific) |

---

## HloPass Interface

### HloPassInterface

```cpp
class HloPassInterface {
 public:
  virtual ~HloPassInterface() = default;
  virtual string_view name() const = 0;

  // Run the pass on a module
  virtual StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) = 0;
};
```

Returns `true` if the pass modified the module.

### HloPassPipeline

Runs multiple passes in sequence:

```cpp
class HloPassPipeline : public HloPassInterface {
 public:
  explicit HloPassPipeline(const string& name);

  // Add a pass to the pipeline
  void AddPass(std::unique_ptr<HloPassInterface> pass);

  // Add an invariant checker
  void AddInvariantChecker(std::function<Status(const HloModule&)> checker);

  // Run all passes
  StatusOr<bool> Run(HloModule* module, ...) override;
};
```

### Standard Pass Pipeline

```
HloPassPipeline "optimization"
  |
  |-- CallInliner
  |-- HloConstantFolding
  |-- AlgebraicSimplifier
  |-- HloCSE
  |-- HloDCE
  |-- WhileLoopConstantSinking
  |-- WhileLoopSimplifier
  |-- TupleSimplifier
  |-- Fusion (backend-specific)
  |-- LayoutAssignment
  |-- HloSchedule
  |-- BufferAssignment
```

### Common Optimization Passes

| Pass | Description |
|------|-------------|
| `CallInliner` | Inline called computations |
| `HloConstantFolding` | Evaluate constant expressions |
| `AlgebraicSimplifier` | Simplify arithmetic |
| `HloCSE` | Common subexpression elimination |
| `HloDCE` | Dead code elimination |
| `WhileLoopConstantSinking` | Move constants into loops |
| `WhileLoopSimplifier` | Simplify while loops |
| `WhileLoopInvariantCodeMotion` | Move invariant ops out of loops |
| `TupleSimplifier` | Simplify tuple operations |
| `BroadcastFolding` | Fold broadcasts into preceding ops |
| `ConvolutionFolding` | Fold ops into convolutions |
| `SortSimplifier` | Simplify sort operations |
| `TransposeFolding` | Fold transposes into dots/convs |
| `ReshapeMover` | Move reshapes to simplify |
| `PadSimplifier` | Simplify pad operations |
| `SliceSimplifier` | Simplify slice operations |
| `GatherSimplifier` | Simplify gather operations |
| `ScatterSimplifier` | Simplify scatter operations |
| `ConditionalSimplifier` | Simplify conditional ops |
| `RealImagExpander` | Expand complex operations |
| `ZeroSimplifier` | Simplify operations involving zero |
| `ConvertSimplifier` | Simplify type conversions |
| `BroadcastIdempotentLaw` | Remove redundant broadcasts |
| `ChainedCallOptimization` | Optimize chained calls |
| `AllReduceSimplifier` | Simplify all-reduce operations |
| `AllGatherSimplifier` | Simplify all-gather operations |
| `Defuser` | Break apart fusion nodes |
| `GpuInstructionFusion` | GPU-specific fusion |
| `CpuInstructionFusion` | CPU-specific fusion |
| `MultiOutputFusion` | Fuse ops with multiple outputs |
| `FusionMerger` | Merge compatible fusion nodes |
| `LayoutAssignment` | Assign memory layouts |
| `HloSchedule` | Schedule instruction execution |
| `BufferAssignment` | Allocate memory buffers |
| `MemorySpaceAssignment` | Assign buffers to memory spaces |

### Pass Invariant Checking

```cpp
// Verify module invariants between passes
pipeline.AddInvariantChecker([](const HloModule& module) {
  return HloVerifier().VerifyModule(module);
});
```

---

## HLO Verification

### HloVerifier

Verifies the correctness of HLO modules:

```cpp
class HloVerifier : public HloPassInterface {
 public:
  StatusOr<bool> Run(HloModule* module, ...) override;

 private:
  // Verify instruction properties
  Status VerifyInstruction(const HloInstruction* instruction);
  // Verify computation properties
  Status VerifyComputation(const HloComputation* computation);
  // Verify module properties
  Status VerifyModule(const HloModule* module);
};
```

### Verification Checks

1. **Shape consistency**: Operand shapes match instruction expectations
2. **Parameter count**: Computations have correct number of parameters
3. **Name uniqueness**: Instruction names are unique within computation
4. **Root instruction**: Root instruction is in the computation
5. **Operand locality**: All operands are in the same computation
6. **Layout consistency**: Layouts are compatible with instruction semantics
7. **Control flow**: While/Conditional have valid body/branch computations

---

## Additional HLO Instructions

### CustomCall

Invokes a backend-specific custom operation.

```
%result = f32[128] custom-call(%input), custom_call_target="my_custom_op",
  backend_config="{\"config_key\": \"config_value\"}"
```

| Attribute | Description |
|-----------|-------------|
| custom_call_target | Name of the custom operation |
| backend_config | Backend-specific configuration string |
| has_side_effect | Whether the call has side effects |
| api_version | API version for the custom call |

### Infeed / Outfeed

Transfer data to/from the device.

```
%token = token[] after-all()
%data = (f32[100], token[]) infeed(%token), infeed_config=""

%token_out = token[] outfeed(%data_out, %token), outfeed_config=""
```

### Send / Recv

Send and receive data via channels.

```
%token = token[] after-all()
%sent = (f32[100], token[]) send(%data, %token), channel_id=1

%recv = (f32[100], token[]) recv(%token), channel_id=1
%data = f32[100] get-tuple-element(%recv), index=0
```

### AfterAll

Creates a token that depends on multiple input tokens.

```
%t1 = token[] ...
%t2 = token[] ...
%result = token[] after-all(%t1, %t2)
```

### OptBarrier

Optimization barrier preventing passes from moving operations across it.

```
%result = f32[100] optimization-barrier(%input)
```

### FFT

Fast Fourier Transform.

```
%result = c64[64] fft(%input), fft_type=FFT, fft_length={64}
```

| FFT Type | Description |
|----------|-------------|
| `FFT` | Forward complex FFT |
| `IFFT` | Inverse complex FFT |
| `RFFT` | Forward real FFT |
| `IRFFT` | Inverse real FFT |

### Cholesky

Cholesky decomposition.

```
%result = f32[64,64] cholesky(%input), lower=true
```

### TriangularSolve

Solves triangular linear systems.

```
%result = f32[64,10] triangular-solve(%a, %b), left_side=true, lower=true, transpose_a=NO_TRANSPOSE
```

### BitcastConvertType / ConvertType

```
%result = u32[10] bitcast-convert(%f32_input)  // Reinterpret bits
%result = bf16[10] convert(%f32_input)          // Value conversion
```

### Domain

Marks a boundary for layout or other domain-specific properties.

```
%result = f32[10] domain(%input), domain={kind="layout"}
```

### Map

Applies a scalar function to elements.

```
%result = f32[10] map(%x, %y), to_apply=add_scalar, dimensions={0}
```

### GetDimensionSize

Returns the size of a dynamic dimension.

```
%result = s32[] get-dimension-size(%x), dimension=0
```

### SetDimensionSize

Sets the dynamic size of a dimension.

```
%result = f32[<=128] set-dimension-size(%x, %size), dimension=0
```

---

## HLO Text Format Reference

### Syntax Summary

```
// Module
HloModule name <config>

// Computation
computation_name (param_name: type, ...) -> type {
  instruction_name = type opcode(operand, ...), attributes
  ROOT result_name = type opcode(operand, ...), attributes
}

// Entry computation
ENTRY name { ... }
```

### Common Attribute Syntax

```
// Dimensions
dimensions={0, 2}

// Window
window={size=3x3 stride=2x2 pad=1_1x1_1}

// Convolution dimension labels
dim_labels=b01f_01io_b01f

// Shaped types
f32[128,64]
(f32[128], s32[])
pred[]
token[]

// Layout (in braces)
f32[128,64]{0,1}
f32[128,64]{0,1}(2,4)

// Replica groups
replica_groups={{0,1}, {2,3}}

// Precision config
precision_config={HIGH,HIGH}

// Distribution
distribution=uniform
distribution=normal
```
