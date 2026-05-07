# TensorFlow Protocol Buffer Definitions Reference

This document provides a comprehensive reference for all TensorFlow protocol buffer
definitions used throughout the framework. These protobuf messages form the backbone
of TensorFlow's serialization, communication, and configuration systems.

## Table of Contents

1. [DataType Enum](#datatype-enum)
2. [TensorShapeProto](#tensorshapeproto)
3. [TensorProto](#tensorproto)
4. [AttrValue](#attrvalue)
5. [NameAttrList](#nameattrlist)
6. [OpDef](#opdef)
7. [OpDeprecation](#opdeprecation)
8. [OpList](#oplist)
9. [NodeDef](#nodedef)
10. [FunctionDef](#functiondef)
11. [FunctionDefLibrary](#functiondeflibrary)
12. [GradientDef](#gradientdef)
13. [RegisteredGradient](#registeredgradient)
14. [GraphDef](#graphdef)
15. [VersionDef](#versiondef)
16. [AllocationDescription](#allocationdescription)
17. [TensorDescription](#tensordescription)
18. [StepStats](#stepstats)
19. [SerializedDType](#serializeddtype)

---

## DataType Enum

**File**: `tensorflow/core/framework/types.proto`
**Package**: `tensorflow`

The `DataType` enum enumerates all supported element types for tensors in TensorFlow.
Values with the `_REF` suffix are reference types used internally for mutable variables
(TF1 legacy). Every base type has a corresponding reference type at offset +100.

### Base Data Types

| Value | Name | Description |
|-------|------|-------------|
| 0 | `DT_INVALID` | Not a legal value. Used to indicate a DataType field has not been set. |
| 1 | `DT_FLOAT` | 32-bit IEEE floating point. The most commonly used type for neural networks. |
| 2 | `DT_DOUBLE` | 64-bit IEEE floating point (double precision). |
| 3 | `DT_INT32` | 32-bit signed integer. |
| 4 | `DT_UINT8` | 8-bit unsigned integer. |
| 5 | `DT_INT16` | 16-bit signed integer. |
| 6 | `DT_INT8` | 8-bit signed integer. |
| 7 | `DT_STRING` | Variable-length byte string. Not a numeric type. |
| 8 | `DT_COMPLEX64` | Single-precision complex number (two 32-bit floats for real/imag). |
| 9 | `DT_INT64` | 64-bit signed integer. |
| 10 | `DT_BOOL` | Boolean value. |
| 11 | `DT_QINT8` | Quantized 8-bit signed integer. Used for quantized inference. |
| 12 | `DT_QUINT8` | Quantized 8-bit unsigned integer. |
| 13 | `DT_QINT32` | Quantized 32-bit signed integer. |
| 14 | `DT_BFLOAT16` | Brain floating point 16-bit. Float32 truncated to 16 bits (1 sign, 8 exponent, 7 mantissa). Popular in deep learning training. |
| 15 | `DT_QINT16` | Quantized 16-bit signed integer. |
| 16 | `DT_QUINT16` | Quantized 16-bit unsigned integer. |
| 17 | `DT_UINT16` | 16-bit unsigned integer. |
| 18 | `DT_COMPLEX128` | Double-precision complex number (two 64-bit floats). |
| 19 | `DT_HALF` | IEEE 16-bit floating point (FP16). 1 sign, 5 exponent, 10 mantissa bits. |
| 20 | `DT_RESOURCE` | Handle to a mutable resource (e.g., variable, queue). Passed by reference. |
| 21 | `DT_VARIANT` | Arbitrary C++ data types. Used for custom/extension types. |
| 22 | `DT_UINT32` | 32-bit unsigned integer. |
| 23 | `DT_UINT64` | 64-bit unsigned integer. |
| 24 | `DT_FLOAT8_E5M2` | 8-bit float: 5 exponent bits, 2 mantissa bits (IEEE-like). |
| 25 | `DT_FLOAT8_E4M3FN` | 8-bit float: 4 exponent bits, 3 mantissa bits, finite-only, with 2 NaN representations. |
| 26 | `DT_FLOAT8_E4M3FNUZ` | 8-bit float: 4 exponent bits, 3 mantissa bits, finite-only, unsigned zero, with NaN. |
| 27 | `DT_FLOAT8_E4M3B11FNUZ` | 8-bit float: 4 exponent bits, 3 mantissa bits, 11-bit bias, finite-only, with NaN. |
| 28 | `DT_FLOAT8_E5M2FNUZ` | 8-bit float: 5 exponent bits, 2 mantissa bits, finite-only, unsigned zero, with NaN. |
| 29 | `DT_INT4` | 4-bit signed integer. |
| 30 | `DT_UINT4` | 4-bit unsigned integer. |
| 31 | `DT_INT2` | 2-bit signed integer. |
| 32 | `DT_UINT2` | 2-bit unsigned integer. |
| 33 | `DT_FLOAT4_E2M1FN` | 4-bit float: 2 exponent bits, 1 mantissa bit, finite-only. |

### Reference Data Types (Legacy TF1)

Reference types are used internally for TF1's obsolete reference Variables. Each
base type has a corresponding `_REF` variant at value = base_value + 100.

| Value | Name |
|-------|------|
| 101 | `DT_FLOAT_REF` |
| 102 | `DT_DOUBLE_REF` |
| 103 | `DT_INT32_REF` |
| 104 | `DT_UINT8_REF` |
| 105 | `DT_INT16_REF` |
| 106 | `DT_INT8_REF` |
| 107 | `DT_STRING_REF` |
| 108 | `DT_COMPLEX64_REF` |
| 109 | `DT_INT64_REF` |
| 110 | `DT_BOOL_REF` |
| 111 | `DT_QINT8_REF` |
| 112 | `DT_QUINT8_REF` |
| 113 | `DT_QINT32_REF` |
| 114 | `DT_BFLOAT16_REF` |
| 115 | `DT_QINT16_REF` |
| 116 | `DT_QUINT16_REF` |
| 117 | `DT_UINT16_REF` |
| 118 | `DT_COMPLEX128_REF` |
| 119 | `DT_HALF_REF` |
| 120 | `DT_RESOURCE_REF` |
| 121 | `DT_VARIANT_REF` |
| 122 | `DT_UINT32_REF` |
| 123 | `DT_UINT64_REF` |
| 124 | `DT_FLOAT8_E5M2_REF` |
| 125 | `DT_FLOAT8_E4M3FN_REF` |
| 126 | `DT_FLOAT8_E4M3FNUZ_REF` |
| 127 | `DT_FLOAT8_E4M3B11FNUZ_REF` |
| 128 | `DT_FLOAT8_E5M2FNUZ_REF` |
| 129 | `DT_INT4_REF` |
| 130 | `DT_UINT4_REF` |
| 131 | `DT_INT2_REF` |
| 132 | `DT_UINT2_REF` |
| 133 | `DT_FLOAT4_E2M1FN_REF` |

---

## TensorShapeProto

**File**: `tensorflow/core/framework/tensor_shape.proto`
**Package**: `tensorflow`

Describes the shape (dimensionality) of a tensor. A tensor shape is represented
as a list of dimensions, where each dimension has a size and an optional name.

### Message: `TensorShapeProto`

Represents the full shape of a tensor.

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `dim` | `repeated Dim` | 2 | Dimensions of the tensor. The order matters: the first entry is the outermost dimension, the last entry is the innermost dimension. This matches the in-memory layout of RowMajor Eigen tensors. |
| `unknown_rank` | `bool` | 3 | If true, the number of dimensions in the shape is unknown. If true, `dim.size()` must be 0. |

**Constraints**:
- If `dim.size() > 0`, then `unknown_rank` must be `false`.
- A dimension value of -1 indicates an unknown size.
- Dimension values must be >= -1.

### Message: `TensorShapeProto.Dim`

Represents a single dimension of a tensor.

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `size` | `int64` | 1 | Size of the tensor in that dimension. Must be >= -1. A value of -1 means "unknown" dimension. |
| `name` | `string` | 2 | Optional name of the tensor dimension (e.g., "batch", "height", "width"). |

**Example representations**:
- Scalar: `dim` is empty, `unknown_rank` is false
- Vector of length 10: `dim = [{size: 10}]`
- 30x40 matrix: `dim = [{size: 30, name: "input"}, {size: 40, name: "output"}]`
- Unknown rank: `unknown_rank = true`, `dim` is empty
- Partially known shape: `dim = [{size: -1}, {size: 128}]` (batch size unknown)

---

## TensorProto

**File**: `tensorflow/core/framework/tensor.proto`
**Package**: `tensorflow`

Protocol buffer representing a tensor. This is the serialized form of a tensor
used for storage, RPC transfer, and graph definition constants.

### Message: `TensorProto`

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `dtype` | `DataType` | 1 | Data type of the tensor elements. |
| `tensor_shape` | `TensorShapeProto` | 2 | Shape of the tensor. |
| `version_number` | `int32` | 3 | Version number. In version 0, if the "repeated xxx" representations contain only one element, that element is repeated to fill the shape. |
| `tensor_content` | `bytes` | 4 | Serialized raw tensor content. This representation reduces serialization overhead during RPC calls by avoiding serialization of many repeated small items. Can be used for all tensor types. |
| `half_val` | `repeated int32` (packed) | 13 | Values for `DT_HALF` and `DT_BFLOAT16`. Note: since protobuf has no int16 type, there is zero padding per value. |
| `float_val` | `repeated float` (packed) | 5 | Values for `DT_FLOAT`. |
| `double_val` | `repeated double` (packed) | 6 | Values for `DT_DOUBLE`. |
| `int_val` | `repeated int32` (packed) | 7 | Values for `DT_INT32`, `DT_INT16`, `DT_UINT16`, `DT_INT8`, `DT_UINT8`. |
| `string_val` | `repeated bytes` | 8 | Values for `DT_STRING`. Each element is a separate bytes field. |
| `scomplex_val` | `repeated float` (packed) | 9 | Values for `DT_COMPLEX64`. Elements at indices `(2*i)` and `(2*i+1)` are the real and imaginary parts of the i-th complex number. |
| `int64_val` | `repeated int64` (packed) | 10 | Values for `DT_INT64`. |
| `bool_val` | `repeated bool` (packed) | 11 | Values for `DT_BOOL`. |
| `dcomplex_val` | `repeated double` (packed) | 12 | Values for `DT_COMPLEX128`. Elements at indices `(2*i)` and `(2*i+1)` are real and imaginary parts. |
| `resource_handle_val` | `repeated ResourceHandleProto` | 14 | Values for `DT_RESOURCE`. |
| `variant_val` | `repeated VariantTensorDataProto` | 15 | Values for `DT_VARIANT`. |
| `uint32_val` | `repeated uint32` (packed) | 16 | Values for `DT_UINT32`. |
| `uint64_val` | `repeated uint64` (packed) | 17 | Values for `DT_UINT64`. |
| `float8_val` | `bytes` | 18 | Values for `DT_FLOAT8_*` types. Variable-sized set of bytes. |

### Value Representation Rules

Only one representation should be set at a time. Either `tensor_content` is used
(for raw binary data) or the type-specific `xxx_val` field corresponding to `dtype`.

The values in repeated fields represent the flattened tensor in row-major order.

### Message: `VariantTensorDataProto`

Protocol buffer representing the serialization format of `DT_VARIANT` tensors.

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `type_name` | `string` | 1 | Name of the type of objects being serialized. |
| `metadata` | `bytes` | 2 | Portions of the object that are not Tensors. |
| `tensors` | `repeated TensorProto` | 3 | Tensors contained within the serialized objects. |

---

## AttrValue

**File**: `tensorflow/core/framework/attr_value.proto`
**Package**: `tensorflow`

Protocol buffer representing the value for an attribute used to configure an Op.
The `oneof value` field ensures that only one type of value is set, matching the
corresponding attribute type.

### Message: `AttrValue`

| Field | Type | Number | Attr Type String | Description |
|-------|------|--------|-------------------|-------------|
| `list` | `ListValue` | 1 | `"list(...)"` | A list of values. Any `"list(...)"` type. |
| `s` | `bytes` | 2 | `"string"` | String value. |
| `i` | `int64` | 3 | `"int"` | Integer value. |
| `f` | `float` | 4 | `"float"` | Float value. |
| `b` | `bool` | 5 | `"bool"` | Boolean value. |
| `type` | `DataType` | 6 | `"type"` | Data type value. |
| `shape` | `TensorShapeProto` | 7 | `"shape"` | Tensor shape value. |
| `tensor` | `TensorProto` | 8 | `"tensor"` | Tensor value. |
| `placeholder` | `string` | 9 | `"placeholder"` | Placeholder used in function definitions. The attr value will be supplied when the function is instantiated. |
| `func` | `NameAttrList` | 10 | `"func"` | Function reference with associated attributes. |

### Message: `AttrValue.ListValue`

Represents a list of values of a homogeneous type.

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `s` | `repeated bytes` | 2 | List of strings. |
| `i` | `repeated int64` (packed) | 3 | List of integers. |
| `f` | `repeated float` (packed) | 4 | List of floats. |
| `b` | `repeated bool` (packed) | 5 | List of booleans. |
| `type` | `repeated DataType` (packed) | 6 | List of data types. |
| `shape` | `repeated TensorShapeProto` | 7 | List of tensor shapes. |
| `tensor` | `repeated TensorProto` | 8 | List of tensors. |
| `func` | `repeated NameAttrList` | 9 | List of function references. |

---

## NameAttrList

**File**: `tensorflow/core/framework/attr_value.proto`
**Package**: `tensorflow`

A list of attribute names and their values, attached with a string name.
Used to represent a function or op reference with its configuration attributes.

### Message: `NameAttrList`

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `name` | `string` | 1 | The name of the function or op. |
| `attr` | `map<string, AttrValue>` | 2 | Attribute key-value pairs for the function/op instantiation. |

**Example**: `MatMul[T=float, transpose_b=true]` is represented as:
```
name: "MatMul"
attr: { key: "T", value: { type: DT_FLOAT } }
attr: { key: "transpose_b", value: { b: true } }
```

---

## OpDef

**File**: `tensorflow/core/framework/op_def.proto`
**Package**: `tensorflow`

Defines an operation. A `NodeDef` in a `GraphDef` specifies an Op by using the
`op` field which should match the name of an `OpDef`.

### Message: `OpDef`

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `name` | `string` | 1 | Op name. Names starting with underscore are reserved for internal use. Should be CamelCase matching `[A-Z][a-zA-Z0-9>_]*`. |
| `input_arg` | `repeated ArgDef` | 2 | Description of the input(s). |
| `output_arg` | `repeated ArgDef` | 3 | Description of the output(s). |
| `control_output` | `repeated string` | 20 | Named control outputs for composite operations (functions). |
| `attr` | `repeated AttrDef` | 4 | Description of graph-construction-time configuration attributes. |
| `deprecation` | `OpDeprecation` | 8 | Optional deprecation based on GraphDef versions. |
| `summary` | `string` | 5 | One-line human-readable description. |
| `description` | `string` | 6 | Additional, longer human-readable description. |
| `is_commutative` | `bool` | 18 | True if the operation is commutative: `op(a,b) == op(b,a)`. |
| `is_aggregate` | `bool` | 16 | True if the op accepts N >= 2 inputs and produces 1 output all of the same type. Should be associative and commutative. |
| `is_stateful` | `bool` | 17 | True if the op depends on state beyond inputs or has side effects. Stateful ops are never optimized away by CSE. |
| `allows_uninitialized_input` | `bool` | 19 | True if the op may accept uninitialized tensors as input (e.g., Assign). |
| `is_distributed_communication` | `bool` | 21 | True if the op uses distributed communication and may return network errors. |

### Message: `OpDef.ArgDef`

Describes an input or output argument of an operation.

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `name` | `string` | 1 | Name for the input/output. Should match `[a-z][a-z0-9_]*`. |
| `description` | `string` | 2 | Human readable description. |
| `type` | `DataType` | 3 | Fixed type for a single tensor. |
| `type_attr` | `string` | 4 | Name of an attr with type `"type"`. |
| `number_attr` | `string` | 5 | Name of an attr with type `"int"` for sequence length. |
| `type_list_attr` | `string` | 6 | Name of an attr with type `"list(type)"`. |
| `handle_data` | `repeated ResourceHandleProto.DtypeAndShape` | 7 | Handle data for resource inputs. |
| `is_ref` | `bool` | 16 | For inputs: if true, inputs must be refs. For outputs: if true, outputs are refs. |
| `experimental_full_type` | `FullTypeDef` | 17 | Experimental full type declaration. |

**Type specification rules**:
- Single tensor: set either `type` or `type_attr`
- Sequence of same-type tensors: set `number_attr` plus `type` or `type_attr`
- Sequence of different-type tensors: set `type_list_attr`

### Message: `OpDef.AttrDef`

Describes a graph-construction-time attribute of an operation.

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `name` | `string` | 1 | Descriptive name matching `[a-z][a-z0-9_]+`. Used as keyword argument name. |
| `type` | `string` | 2 | Type name from attr_value.proto: `"string"`, `"list(string)"`, `"int"`, `"float"`, `"bool"`, `"type"`, `"list(type)"`, `"shape"`, `"list(shape)"`, `"tensor"`, `"list(tensor)"`, `"func"`, `"list(func)"`. |
| `default_value` | `AttrValue` | 3 | Default value if user does not supply one. |
| `description` | `string` | 4 | Human-readable description. |
| `has_minimum` | `bool` | 5 | For `"int"` types: indicates a minimum value constraint. For `"list(___)"` types: indicates a minimum length. |
| `minimum` | `int64` | 6 | The minimum value/length. |
| `allowed_values` | `AttrValue` | 7 | Set of allowed values. Uses the "list" version of the type field. |

---

## OpDeprecation

**File**: `tensorflow/core/framework/op_def.proto`
**Package**: `tensorflow`

Information about version-dependent deprecation of an op.

### Message: `OpDeprecation`

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `version` | `int32` | 1 | First GraphDef version at which the op is disallowed. |
| `explanation` | `string` | 2 | Explanation of why it was deprecated and what to use instead. |

---

## OpList

**File**: `tensorflow/core/framework/op_def.proto`
**Package**: `tensorflow`

A collection of OpDefs, typically used to list all registered operations.

### Message: `OpList`

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `op` | `repeated OpDef` | 1 | List of operation definitions. |

---

## NodeDef

**File**: `tensorflow/core/framework/node_def.proto`
**Package**: `tensorflow`

Defines a node in the computation graph. Each node represents a single operation
with its inputs, device placement, and configuration attributes.

### Message: `NodeDef`

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `name` | `string` | 1 | Unique name for this node within a single GraphDef. Must match `[A-Za-z0-9.][A-Za-z0-9_>./]*`. Used for naming inputs, logging, and visualization. |
| `op` | `string` | 2 | The operation name. Should match a registered OpDef name. Names starting with underscore are reserved for internal use. |
| `input` | `repeated string` | 3 | Input specifications. Format: `"node:src_output"` where `node` is a string name and `src_output` indicates which output tensor to use. If `src_output` is 0, the `:0` suffix can be omitted. Control inputs have format `"^node"`. |
| `device` | `string` | 4 | A (possibly partial) specification for device placement. |
| `attr` | `map<string, AttrValue>` | 5 | Operation-specific configuration attributes. Keys must match the attr names from the corresponding OpDef. |
| `experimental_debug_info` | `ExperimentalDebugInfo` | 6 | Debug information associated with the node. |
| `experimental_type` | `FullTypeDef` | 7 | Complete type of this node (experimental). Currently contains return types only. |

### Device Specification Syntax

```
DEVICE_SPEC ::= PARTIAL_SPEC
PARTIAL_SPEC ::= ("/" CONSTRAINT) *
CONSTRAINT ::= ("job:" JOB_NAME)
             | ("replica:" [1-9][0-9]*)
             | ("task:" [1-9][0-9]*)
             | ("device:" [A-Za-z]* ":" ([1-9][0-9]* | "*"))
```

**Examples**:
- Full: `"/job:worker/replica:0/task:1/device:GPU:3"`
- Partial: `"/job:worker/device:GPU:3"`
- No specification: `""`

### Input Format Details

- Regular input: `"node_name"` or `"node_name:output_index"`
- Control input: `"^node_name"` (control dependencies)
- Multiple inputs are ordered: data inputs first, then control inputs

### Message: `NodeDef.ExperimentalDebugInfo`

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `original_node_names` | `repeated string` | 1 | Names of original graph nodes that this node was derived from (e.g., from fusion). |
| `original_func_names` | `repeated string` | 2 | Names of functions from the original graph that this node was derived from. |

---

## FunctionDef

**File**: `tensorflow/core/framework/function.proto`
**Package**: `tensorflow`

Defines a TensorFlow function. A function can be instantiated when the runtime
can bind every attr with a value.

### Message: `FunctionDef`

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `signature` | `OpDef` | 1 | The function's name, arguments, return values, attrs etc. |
| `attr` | `map<string, AttrValue>` | 5 | Attributes specific to this function definition (not bound to a specific call). |
| `arg_attr` | `map<uint32, ArgAttrs>` | 7 | Attributes for function arguments. Keyed by argument index. |
| `resource_arg_unique_id` | `map<uint32, uint32>` | 8 | Unique IDs for resource arguments to track aliasing. If arguments A and B alias each other, their unique IDs are equal. |
| `node_def` | `repeated NodeDef` | 3 | The body of the function. These are the nodes that implement the function's computation. |
| `ret` | `map<string, string>` | 4 | Mapping from output arg names (from `signature`) to the outputs from `node_def` that should be returned. |
| `control_ret` | `map<string, string>` | 6 | Mapping from control output names (from `signature`) to node names in `node_def` which should be control outputs. |

### Message: `FunctionDef.ArgAttrs`

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `attr` | `map<string, AttrValue>` | 1 | Attributes for a function argument. |

### Output Reference Format

Output references in `ret` and node inputs use these formats:
- `"fun_in"` -- a function input argument (single or list)
- `"fun_in:0"` -- first element of a function input arg
- `"node:out"` -- output from a node in `node_def`
- `"node:out:0"` -- first element of a node output arg

---

## FunctionDefLibrary

**File**: `tensorflow/core/framework/function.proto`
**Package**: `tensorflow`

A library is a set of named functions, their gradients, and registered gradients.

### Message: `FunctionDefLibrary`

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `function` | `repeated FunctionDef` | 1 | User-defined functions. |
| `gradient` | `repeated GradientDef` | 2 | Gradient function definitions. |
| `registered_gradients` | `repeated RegisteredGradient` | 3 | Registered gradient functions for specific op types. |

---

## GradientDef

**File**: `tensorflow/core/framework/function.proto`
**Package**: `tensorflow`

Defines the gradient function for a named function.

### Message: `GradientDef`

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `function_name` | `string` | 1 | The forward function name. |
| `gradient_func` | `string` | 2 | The gradient function's name. |

**Semantics**: If function `f` takes N inputs and produces M outputs:
```
(y1, y2, ..., y_M) = f(x1, x2, ..., x_N)
```
Then gradient function `g` takes N + M inputs and produces N outputs:
```
(dL/dx1, dL/dx2, ..., dL/dx_N) = g(x1, x2, ..., x_N, dL/dy1, ..., dL/dy_M)
```

---

## RegisteredGradient

**File**: `tensorflow/core/framework/function.proto`
**Package**: `tensorflow`

Stores a gradient function registered in the gradients library, identified by op type.

### Message: `RegisteredGradient`

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `gradient_func` | `string` | 1 | The gradient function's name. |
| `registered_op_type` | `string` | 2 | The op type this gradient is registered for. |

---

## GraphDef

**File**: `tensorflow/core/framework/graph.proto`
**Package**: `tensorflow`

Represents the graph of operations. This is the top-level serialization format
for TensorFlow computation graphs.

### Message: `GraphDef`

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `node` | `repeated NodeDef` | 1 | The nodes (operations) in the graph. |
| `versions` | `VersionDef` | 4 | Compatibility version information. |
| `version` | `int32` | 3 | **Deprecated**. Use `versions` instead. |
| `library` | `FunctionDefLibrary` | 2 | User-defined functions. Nodes whose `op` matches a function name are function calls. |
| `debug_info` | `GraphDebugInfo` | 5 | Stack traces for the nodes in this graph. |

### Function Call Semantics

When a node's `op` field matches a function name in `library`:
1. The callee may start execution as soon as some of its inputs are ready.
2. The consumer of return values may start executing as soon as the needed
   return values are ready.
3. Use `Tuple()` mechanism to synchronize when all inputs or all outputs
   must be ready simultaneously.

---

## VersionDef

**File**: `tensorflow/core/framework/versions.proto`
**Package**: `tensorflow`

Version information for serialized data. Each consumer has "consumer" and
"min_producer" versions. A consumer is allowed to consume data if:

1. `producer >= min_producer`
2. `consumer >= min_consumer`
3. `consumer` not in `bad_consumers`

### Message: `VersionDef`

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `producer` | `int32` | 1 | The version of the code that produced this data. |
| `min_consumer` | `int32` | 2 | Any consumer below this version is not allowed to consume this data. |
| `bad_consumers` | `repeated int32` | 3 | Specific consumer versions which are disallowed (e.g., due to bugs). |

---

## AllocationDescription

**File**: `tensorflow/core/framework/allocation_description.proto`
**Package**: `tensorflow`

Describes a memory allocation made by a tensor allocator.

### Message: `AllocationDescription`

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `requested_bytes` | `int64` | 1 | Total number of bytes requested. |
| `allocated_bytes` | `int64` | 2 | Total number of bytes allocated if known. |
| `allocator_name` | `string` | 3 | Name of the allocator used (e.g., "cpu", "GPU_0_bfc"). |
| `allocation_id` | `int64` | 4 | Identifier of the allocated buffer if known. |
| `has_single_reference` | `bool` | 5 | True if this tensor only has one remaining reference. |
| `ptr` | `uint64` | 6 | Memory address of the allocation. |

---

## TensorDescription

**File**: `tensorflow/core/framework/tensor_description.proto`
**Package**: `tensorflow`

Describes a tensor including its type, shape, and allocation information.

### Message: `TensorDescription`

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `dtype` | `DataType` | 1 | Data type of tensor elements. |
| `shape` | `TensorShapeProto` | 2 | Shape of the tensor. |
| `allocation_description` | `AllocationDescription` | 4 | Information about the size and allocator used. |

---

## StepStats

**File**: `tensorflow/core/framework/step_stats.proto`
**Package**: `tensorflow`

Performance and memory statistics for a single execution step. Collected
when profiling is enabled.

### Message: `StepStats`

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `dev_stats` | `repeated DeviceStepStats` | 1 | Per-device statistics. |

### Message: `DeviceStepStats`

Statistics for all nodes executed on a single device during a step.

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `device` | `string` | 1 | Device name (e.g., "/job:worker/task:0/device:GPU:0"). |
| `node_stats` | `repeated NodeExecStats` | 2 | Per-node execution statistics. |
| `thread_names` | `map<uint32, string>` | 3 | Map from thread ID to thread name. |

### Message: `NodeExecStats`

Time and size statistics for a single execution of a graph node.

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `node_name` | `string` | 1 | Full string name of the node. |
| `all_start_micros` | `int64` | 2 | Timestamp when the node started executing (microseconds since epoch). |
| `op_start_rel_micros` | `int64` | 3 | Microseconds between all_start and op start. |
| `op_end_rel_micros` | `int64` | 4 | Microseconds between all_start and op end. |
| `all_end_rel_micros` | `int64` | 5 | Microseconds between all_start and all end. |
| `memory` | `repeated AllocatorMemoryUsed` | 6 | Per-allocator memory statistics. |
| `output` | `repeated NodeOutput` | 7 | Output tensor descriptions. |
| `timeline_label` | `string` | 8 | Label for timeline visualization. |
| `scheduled_micros` | `int64` | 9 | Timestamp when the node was scheduled. |
| `thread_id` | `uint32` | 10 | Thread ID that executed the node. |
| `referenced_tensor` | `repeated AllocationDescription` | 11 | Descriptions of referenced tensors. |
| `memory_stats` | `MemoryStats` | 12 | Device memory statistics. |
| `all_start_nanos` | `int64` | 13 | Nanosecond version of all_start_micros. |
| `op_start_rel_nanos` | `int64` | 14 | Nanosecond version of op_start_rel_micros. |
| `op_end_rel_nanos` | `int64` | 15 | Nanosecond version of op_end_rel_micros. |
| `all_end_rel_nanos` | `int64` | 16 | Nanosecond version of all_end_rel_micros. |
| `scheduled_nanos` | `int64` | 17 | Nanosecond version of scheduled_micros. |

### Message: `NodeOutput`

Output sizes for a single output of a node execution.

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `slot` | `int32` | 1 | Output slot index. |
| `tensor_description` | `TensorDescription` | 3 | Description of the output tensor. |

### Message: `AllocatorMemoryUsed`

Memory usage statistics for a specific allocator during node execution.

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `allocator_name` | `string` | 1 | Name of the allocator. |
| `total_bytes` | `int64` | 2 | Total bytes used (per-node). |
| `peak_bytes` | `int64` | 3 | Peak memory usage. |
| `live_bytes` | `int64` | 4 | Bytes not yet deallocated. |
| `allocation_records` | `repeated AllocationRecord` | 6 | Allocation/deallocation timeline. |
| `allocator_bytes_in_use` | `int64` | 5 | Overall allocator bytes currently in use (snapshot). |

### Message: `AllocationRecord`

A single allocation or deallocation event.

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `alloc_micros` | `int64` | 1 | Timestamp of the operation (microseconds since epoch). |
| `alloc_bytes` | `int64` | 2 | Bytes allocated (positive) or deallocated (negative). |

### Message: `MemoryStats`

Memory statistics for a node execution.

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `temp_memory_size` | `int64` | 1 | Temporary memory size in bytes. |
| `persistent_memory_size` | `int64` | 3 | Persistent memory size in bytes. |
| `persistent_tensor_alloc_ids` | `repeated int64` | 5 | Allocation IDs for persistent tensors. |
| `device_temp_memory_size` | `int64` | 2 | **Deprecated**. |
| `device_persistent_memory_size` | `int64` | 4 | **Deprecated**. |
| `device_persistent_tensor_alloc_ids` | `repeated int64` | 6 | **Deprecated**. |

---

## SerializedDType

**File**: `tensorflow/core/framework/types.proto`
**Package**: `tensorflow`

Represents a serialized `tf.dtypes.Dtype`.

### Message: `SerializedDType`

| Field | Type | Number | Description |
|-------|------|--------|-------------|
| `datatype` | `DataType` | 1 | The serialized data type value. |

---

## ConfigProto (Reference)

**File**: `tensorflow/core/framework/config.proto`
**Package**: `tensorflow`

The main configuration message for TensorFlow sessions and runtime. Contains
all configurable parameters for the session, including parallelism settings,
device placement, and optimizer configuration.

This message is extensive and controls:

### Key Configuration Areas

1. **Session Configuration**: Controls how sessions are created and managed
2. **Device Placement**: Automatic vs manual device placement policies
3. **Execution Options**: Inter/intra-op parallelism, timeout settings
4. **Grappler Configuration**: Graph optimization settings via `RewriterConfig`
5. **RPC Options**: Communication settings for distributed training
6. **Cluster Definition**: Job and task configuration for distributed execution

### Common Usage Patterns

```python
# Python API
config = tf.ConfigProto(
    intra_op_parallelism_threads=4,
    inter_op_parallelism_threads=4,
    device_count={'GPU': 2}
)
session = tf.Session(config=config)
```

---

## Cross-Reference: Proto Dependencies

The protobuf messages have the following dependency relationships:

```
types.proto          (standalone)
tensor_shape.proto   (standalone)
allocation_description.proto (standalone)

tensor.proto
  ├── types.proto
  ├── tensor_shape.proto
  └── resource_handle.proto

attr_value.proto
  ├── tensor.proto
  ├── tensor_shape.proto
  └── types.proto

op_def.proto
  ├── attr_value.proto
  ├── full_type.proto
  ├── resource_handle.proto
  └── types.proto

node_def.proto
  ├── attr_value.proto
  └── full_type.proto

function.proto
  ├── attr_value.proto
  ├── node_def.proto
  └── op_def.proto

versions.proto      (standalone)

graph.proto
  ├── function.proto
  ├── graph_debug_info.proto
  ├── node_def.proto
  └── versions.proto

tensor_description.proto
  ├── allocation_description.proto
  ├── tensor_shape.proto
  └── types.proto

step_stats.proto
  ├── allocation_description.proto
  └── tensor_description.proto

config.proto
  └── (various other protos)
```

---

## Common Patterns

### Creating a Simple GraphDef

```
GraphDef:
  node: [
    NodeDef { name: "a", op: "Placeholder", attr: { dtype: DT_FLOAT } },
    NodeDef { name: "b", op: "Placeholder", attr: { dtype: DT_FLOAT } },
    NodeDef { name: "c", op: "Add", input: ["a", "b"], attr: { T: DT_FLOAT } }
  ]
  versions: { producer: 27 }
```

### Function Definition Pattern

```
FunctionDef:
  signature: { name: "MyFunc", input_arg: [{name: "x", type: DT_FLOAT}],
               output_arg: [{name: "y", type: DT_FLOAT}] }
  node_def: [ NodeDef { name: "scale", op: "Mul", input: ["x", "const"] } ]
  ret: { "y": "scale:z" }
```

### TensorProto for a Constant

```
TensorProto:
  dtype: DT_FLOAT
  tensor_shape: { dim: [{size: 2}, {size: 2}] }
  float_val: [1.0, 2.0, 3.0, 4.0]
```

---

## Version History Notes

- **GraphDef version**: Distinct from the TensorFlow version. Each TensorFlow
  release supports a range of GraphDef versions.
- **Field 3 in GraphDef** (`version`): Deprecated. Use `versions` field instead.
- **Field 2 in FunctionDef**: Deleted in January 2017, GraphDef version 21.
- Reference data types (`_REF` suffix) are legacy from TF1's obsolete reference
  Variables and should not be used in new code.

---

## Proto Encoding Notes

1. **Packed repeated fields**: Integer and float repeated fields use packed encoding
   (`[packed = true]`) for more efficient wire format.
2. **Oneof fields**: `AttrValue.value` uses `oneof` to ensure only one value type
   is set at a time.
3. **Map fields**: `NodeDef.attr`, `FunctionDef.ret`, and `FunctionDef.control_ret`
   use protobuf `map` syntax for key-value pairs.
4. **Arena allocation**: All framework protos use `cc_enable_arenas = true` for
   improved C++ allocation performance.
5. **Java/Go packages**: Each proto specifies Java outer class name, package, and
   Go import path for cross-language compatibility.
