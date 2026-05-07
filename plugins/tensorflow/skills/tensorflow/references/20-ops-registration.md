# TensorFlow Op Registration

This reference covers TensorFlow's op registration system: the `REGISTER_OP`
macro, `OpDefBuilder`, the `OpDef` and `AttrDef` protocol buffers, shape
inference, type constraints, gradient registration, and the complete process
of defining and registering custom operations.

---

## Table of Contents

1. [REGISTER_OP Macro](#register_op-macro)
2. [OpDefBuilder](#opdefbuilder)
3. [OpDef Protocol Buffer](#opdef-protocol-buffer)
4. [AttrDef](#attrdef)
5. [Shape Inference](#shape-inference)
6. [Input/Output Specification](#inputoutput-specification)
7. [Type Constraints](#type-constraints)
8. [Gradient Registration](#gradient-registration)
9. [Op Registration Patterns](#op-registration-patterns)
10. [Op Implementation Files](#op-implementation-files)
11. [Custom Op Registration](#custom-op-registration)
12. [Op Versioning](#op-versioning)

---

## REGISTER_OP Macro

**Header:** `tensorflow/core/framework/op.h`

The `REGISTER_OP` macro is the primary mechanism for defining new TensorFlow
operations. It registers an op definition with the global `OpRegistry`.

### Basic Syntax

```cpp
REGISTER_OP("OpName")
    .Attr("attr_name:attr_type")
    .Attr("attr_name:attr_type=default_value")
    .Input("input_name:type_spec")
    .Output("output_name:type_spec")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // Shape inference logic.
      return absl::OkStatus();
    })
    .Doc(R"(
Short description.

attr_name: Description of attr_name.
input_name: Description of input_name.
output_name: Description of output_name.
)");
```

### Available Builder Methods

```cpp
// OpDefBuilderWrapper methods (called after REGISTER_OP):

// Add an attribute.
OpDefBuilderWrapper& Attr(std::string spec);

// Add an input.
OpDefBuilderWrapper& Input(std::string spec);

// Add an output.
OpDefBuilderWrapper& Output(std::string spec);

// Mark as commutative (order of inputs doesn't matter).
OpDefBuilderWrapper& SetIsCommutative();

// Mark as aggregate (reduction-like: AddN).
OpDefBuilderWrapper& SetIsAggregate();

// Mark as stateful (has side effects, prevents CSE/constant folding).
OpDefBuilderWrapper& SetIsStateful();

// Prevent optimization (reuses stateful flag).
OpDefBuilderWrapper& SetDoNotOptimize();

// Allow uninitialized inputs.
OpDefBuilderWrapper& SetAllowsUninitializedInput();

// Mark as deprecated.
OpDefBuilderWrapper& Deprecated(int version, std::string explanation);

// Add documentation.
OpDefBuilderWrapper& Doc(std::string text);

// Set shape inference function.
OpDefBuilderWrapper& SetShapeFn(OpShapeInferenceFn fn);

// Mark as distributed communication.
OpDefBuilderWrapper& SetIsDistributedCommunication();

// Set type constructor (for full type inference).
OpDefBuilderWrapper& SetTypeConstructor(OpTypeConstructor fn);

// Set forward type function.
OpDefBuilderWrapper& SetForwardTypeFn(TypeInferenceFn fn);

// Set reverse type function.
OpDefBuilderWrapper& SetReverseTypeFn(int input_number, TypeInferenceFn fn);
```

### Example: Complete Op Registration

```cpp
REGISTER_OP("MatMul")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("T: {float, double, int32, complex64, complex128}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle a_shape = c->input(0);
      shape_inference::ShapeHandle b_shape = c->input(1);

      bool transpose_a, transpose_b;
      TF_RETURN_IF_ERROR(c->GetAttr("transpose_a", &transpose_a));
      TF_RETURN_IF_ERROR(c->GetAttr("transpose_b", &transpose_b));

      // Validate ranks.
      TF_RETURN_IF_ERROR(c->WithRank(a_shape, 2, &a_shape));
      TF_RETURN_IF_ERROR(c->WithRank(b_shape, 2, &b_shape));

      // Determine output dimensions.
      shape_inference::DimensionHandle output_rows =
          transpose_a ? c->Dim(a_shape, 1) : c->Dim(a_shape, 0);
      shape_inference::DimensionHandle output_cols =
          transpose_b ? c->Dim(b_shape, 0) : c->Dim(b_shape, 1);

      // Verify inner dimensions match.
      shape_inference::DimensionHandle inner_a =
          transpose_a ? c->Dim(a_shape, 0) : c->Dim(a_shape, 1);
      shape_inference::DimensionHandle inner_b =
          transpose_b ? c->Dim(b_shape, 1) : c->Dim(b_shape, 0);
      TF_RETURN_IF_ERROR(c->Merge(inner_a, inner_b, &inner_a));

      c->set_output(0, c->Matrix(output_rows, output_cols));
      return absl::OkStatus();
    })
    .Doc(R"doc(
Multiply matrix "a" by matrix "b".

The inputs must be two-dimensional matrices and the inner dimension of
"a" (after, optionally, being transposed) must match the outer dimension
of "b" (after optional transposition).

a: A matrix.
b: A matrix.
transpose_a: If true, "a" is transposed before multiplication.
transpose_b: If true, "b" is transposed before multiplication.
product: The product matrix.
)doc");
```

---

## OpDefBuilder

**Header:** `tensorflow/core/framework/op_def_builder.h`

`OpDefBuilder` is the underlying builder class used by `REGISTER_OP`. It
constructs an `OpDef` protocol buffer.

### Construction

```cpp
// Start building an op definition.
explicit OpDefBuilder(const string& op_name);
```

### Methods

```cpp
// Add an attribute specification.
// Format: "name:type" or "name:type=default"
OpDefBuilder& Attr(StringPiece spec);

// Add an input specification.
// Format: "name:type-expr"
OpDefBuilder& Input(StringPiece spec);

// Add an output specification.
OpDefBuilder& Output(StringPiece spec);

// Set various flags.
OpDefBuilder& SetIsCommutative();
OpDefBuilder& SetIsAggregate();
OpDefBuilder& SetIsStateful();
OpDefBuilder& SetAllowsUninitializedInput();

// Set documentation.
OpDefBuilder& Doc(StringPiece text);

// Set shape inference function.
OpDefBuilder& SetShapeFn(OpShapeInferenceFn fn);

// Set deprecation.
OpDefBuilder& Deprecated(int version, StringPiece explanation);

// Finalize and validate the op definition.
absl::Status Finalize(OpDef* op_def) const;
```

### Attribute Specification Format

The `Attr` method accepts specifications in these formats:

```
"name:type"                   // Required attribute
"name:type=default"           // Attribute with default value
"name:{type1,type2,...}"      // Type constraint (allowed types)
"name:{type1,type2,...}=default"  // Constrained with default
```

### Input/Output Specification Format

The `Input` and `Output` methods accept specifications in these formats:

```
"name:type"                   // Single tensor
"name:Ref(type)"              // Reference tensor
"name:N * type"               // N tensors of the same type (N is an int attr)
"name:type_attr"              // Type determined by an attribute
"name:N * type_attr"          // N tensors of type determined by attribute
"name:type_list_attr"         // List of tensors with types from an attribute
```

---

## OpDef Protocol Buffer

**Header:** `tensorflow/core/framework/op_def.proto`

`OpDef` is the protocol buffer representation of an op definition.

```protobuf
message OpDef {
  string name = 1;                        // Op name

  repeated ArgDef input_arg = 2;          // Input specifications
  repeated ArgDef output_arg = 3;         // Output specifications

  repeated AttrDef attr = 4;              // Attribute definitions

  // Op properties.
  bool is_commutative = 5;
  bool is_aggregate = 6;
  bool is_stateful = 7;
  bool allows_uninitialized_input = 8;

  // Deprecation.
  OpDeprecation deprecation = 12;

  // Summary and description.
  string summary = 9;
  string description = 10;

  // Whether this is a distributed communication op.
  bool is_distributed_communication = 13;
}
```

### ArgDef (Input/Output Arguments)

```protobuf
message ArgDef {
  string name = 1;                // Argument name
  string description = 2;         // Human-readable description
  string type_attr = 3;           // Name of attr that determines type
  string number_attr = 4;         // Name of attr that determines count
  string type_list_attr = 6;      // Name of attr with list of types
  DataType type = 5;              // Fixed type (if not using attrs)
  bool is_ref = 16;               // Whether this is a reference argument

  // Full type information.
  FullTypeDef experimental_full_type = 17;
}
```

### OpDeprecation

```protobuf
message OpDeprecation {
  int32 version = 1;        // First version where deprecated
  string explanation = 2;   // Explanation and suggested replacement
}
```

### OpList

```protobuf
message OpList {
  repeated OpDef op = 1;
}
```

---

## AttrDef

**Header:** `tensorflow/core/framework/op_def.proto`

`AttrDef` defines an attribute for an operation.

```protobuf
message AttrDef {
  string name = 1;                          // Attribute name
  string type = 2;                          // Type specification
  AttrValue default_value = 3;              // Default value
  string description = 4;                   // Human-readable description
  bool has_minimum = 5;                     // Whether minimum is set
  int64 minimum = 6;                        // Minimum value (for int attrs)
  AttrValue allowed_values = 7;             // Allowed values list
}
```

### Attribute Types

| Type String         | C++ Type / Description                            |
|--------------------|---------------------------------------------------|
| `"int"`            | `int64_t`                                         |
| `"float"`          | `float`                                           |
| `"bool"`           | `bool`                                            |
| `"string"`         | `std::string`                                     |
| `"type"`           | `DataType` enum value                             |
| `"shape"`          | `TensorShapeProto`                                |
| `"tensor"`         | `TensorProto`                                     |
| `"list(int)"`      | `std::vector<int64_t>`                            |
| `"list(float)"`    | `std::vector<float>`                              |
| `"list(bool)"`     `std::vector<bool>`                                |
| `"list(string)"`   | `std::vector<std::string>`                        |
| `"list(type)"`     | `std::vector<DataType>`                           |
| `"list(shape)"`    | `std::vector<TensorShapeProto>`                   |
| `"list(tensor)"`   | `std::vector<TensorProto>`                        |
| `"list(attr)"`     | List of AttrValue                                 |
| `"func"`           | `NameAttrList` (function with attributes)         |
| `"list(func)"`     | List of NameAttrList                              |
| `"{type1,type2}"`  | Type constraint (union of allowed types)           |

### Attribute Examples

```cpp
REGISTER_OP("ExampleOp")
    // Simple attributes.
    .Attr("axis: int")                          // Required int attribute
    .Attr("rate: float = 0.1")                  // Float with default
    .Attr("training: bool = false")             // Bool with default
    .Attr("name: string = 'default'")           // String with default

    // Type attributes.
    .Attr("T: type")                            // Any DataType
    .Attr("T: {float, double}")                 // Float or double only
    .Attr("T: numbertype")                      // Predefined type set

    // Type list attributes.
    .Attr("Tlist: list(type)")                  // List of types
    .Attr("Tlist: list({float, double})")       // Constrained list

    // List attributes.
    .Attr("strides: list(int)")                 // List of ints
    .Attr("paddings: list(int)")                // List of ints

    // Shape attribute.
    .Attr("shape: shape")                       // TensorShapeProto

    // Int with minimum.
    .Attr("N: int >= 1")                        // N >= 1
    .Attr("depth: int >= 0")                    // depth >= 0

    // Function attribute.
    .Attr("f: func")                            // Function reference
```

### Predefined Type Sets

TensorFlow provides several predefined type constraint names:

| Name              | Types Included                                   |
|-------------------|--------------------------------------------------|
| `numbertype`      | `{float, double, int32, int64, uint8, uint16, uint32, uint64, int16, int8, complex64, complex128, half, bfloat16}` |
| `realnumbertype`  | Same as numbertype minus complex types            |
| `quantizedtype`   | `{qint8, quint8, qint16, quint16, qint32}`       |
| `float`           | `{float, double, half, bfloat16}`                 |
| `int`             | `{int8, int16, int32, int64}`                     |

---

## Shape Inference

**Header:** `tensorflow/core/framework/shape_inference.h`

Shape inference functions determine the output shapes of an op based on input
shapes and attribute values.

### InferenceContext

```cpp
namespace shape_inference {
class InferenceContext {
 public:
  // Construction.
  InferenceContext(int graph_def_version, const OpDef* op_def,
                   const std::vector<ShapeHandle>& input_shapes,
                   ...);

  // --- Input Access ---

  ShapeHandle input(int idx) const;
  int num_inputs() const;
  int num_outputs() const;

  // --- Output Setting ---

  void set_output(int idx, ShapeHandle shape);

  // --- Shape Constructors ---

  ShapeHandle UnknownShape();       // Completely unknown shape
  ShapeHandle Scalar();             // Shape {} (0 dimensions)
  ShapeHandle Vector(DimensionHandle dim);  // Shape {dim}
  ShapeHandle Matrix(DimensionHandle d1,
                     DimensionHandle d2);   // Shape {d1, d2}
  ShapeHandle ShapeFromShapeProto(
      const TensorShapeProto& proto);

  // --- Dimension Constructors ---

  DimensionHandle UnknownDim();     // Unknown dimension
  DimensionHandle MakeDim(int64_t value);  // Known dimension

  // --- Shape Queries ---

  int Rank(ShapeHandle shape);
  bool RankKnown(ShapeHandle shape);
  DimensionHandle Dim(ShapeHandle shape, int idx);
  int64_t Value(DimensionHandle dim);
  bool ValueKnown(DimensionHandle dim);
  size_t num_elements(ShapeHandle shape);

  // --- Shape Manipulation ---

  // Enforce rank constraint.
  absl::Status WithRank(ShapeHandle shape, int rank,
                        ShapeHandle* out);
  absl::Status WithRankAtLeast(ShapeHandle shape, int rank,
                               ShapeHandle* out);
  absl::Status WithRankAtMost(ShapeHandle shape, int rank,
                              ShapeHandle* out);

  // Merge shapes (must be compatible).
  absl::Status Merge(ShapeHandle s0, ShapeHandle s1, ShapeHandle* out);
  absl::Status Merge(DimensionHandle d0, DimensionHandle d1,
                     DimensionHandle* out);

  // Subshape extraction.
  absl::Status Subshape(ShapeHandle shape, int64_t start,
                        ShapeHandle* out);
  absl::Status Subshape(ShapeHandle shape, int64_t start, int64_t end,
                        ShapeHandle* out);

  // Concatenate shapes.
  absl::Status Concatenate(ShapeHandle s0, ShapeHandle s1,
                           ShapeHandle* out);

  // --- Attribute Access ---

  template <class T>
  absl::Status GetAttr(absl::string_view attr_name, T* value) const;

  // --- Handle Shapes (for Resource/Variant types) ---

  const std::vector<ShapeAndType>* input_handle_shapes_and_types(
      int idx) const;
  void set_output_handle_shapes_and_types(
      int idx, const std::vector<ShapeAndType>& shapes_and_types);
  std::vector<ShapeAndType>* output_handle_shapes_and_types(int idx);

  // --- Make Shape from Dimensions ---

  ShapeHandle MakeShape(std::vector<DimensionHandle> dims);
  ShapeHandle MakeShapeFromShapeProto(const TensorShapeProto& proto);
  ShapeHandle MakeShapeFromTensorShape(const TensorShape& shape);

  // --- Dimension Operations ---

  DimensionHandle Add(DimensionHandle a, DimensionHandle b);
  DimensionHandle Sub(DimensionHandle a, DimensionHandle b);
  DimensionHandle Mul(DimensionHandle a, DimensionHandle b);
  DimensionHandle Div(DimensionHandle a, DimensionHandle b);
  DimensionHandle Min(DimensionHandle a, DimensionHandle b);
  DimensionHandle Max(DimensionHandle a, DimensionHandle b);
};
}
```

### ShapeHandle and DimensionHandle

```cpp
namespace shape_inference {
class ShapeHandle {
 public:
  ShapeHandle();
  bool SameHandle(ShapeHandle h) const;
  // Managed by InferenceContext
};

class DimensionHandle {
 public:
  DimensionHandle();
  bool SameHandle(DimensionHandle h) const;
  // Managed by InferenceContext
};

struct ShapeAndType {
  ShapeHandle shape;
  DataType dtype = DT_INVALID;
};
}
```

### Common Shape Functions

```cpp
// Keep shape unchanged.
absl::Status UnchangedShape(shape_inference::InferenceContext* c);

// Scalar output.
absl::Status ScalarShape(shape_inference::InferenceContext* c);

// Shape same as input 0.
absl::Status UnchangedShape(shape_inference::InferenceContext* c);
```

### Shape Function Examples

```cpp
// Output shape same as input shape.
.SetShapeFn([](shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(0));
  return absl::OkStatus();
})

// Scalar output.
.SetShapeFn([](shape_inference::InferenceContext* c) {
  c->set_output(0, c->Scalar());
  return absl::OkStatus();
})

// 2D output with specified dimensions.
.SetShapeFn([](shape_inference::InferenceContext* c) {
  ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
  c->set_output(0, input);
  return absl::OkStatus();
})

// MatMul shape function.
.SetShapeFn([](shape_inference::InferenceContext* c) {
  ShapeHandle a, b;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &b));

  bool transpose_a, transpose_b;
  TF_RETURN_IF_ERROR(c->GetAttr("transpose_a", &transpose_a));
  TF_RETURN_IF_ERROR(c->GetAttr("transpose_b", &transpose_b));

  DimensionHandle output_rows = c->Dim(a, transpose_a ? 1 : 0);
  DimensionHandle output_cols = c->Dim(b, transpose_b ? 0 : 1);
  c->set_output(0, c->Matrix(output_rows, output_cols));
  return absl::OkStatus();
})

// Broadcast shape (element-wise binary op).
.SetShapeFn([](shape_inference::InferenceContext* c) {
  ShapeHandle a = c->input(0);
  ShapeHandle b = c->input(1);
  ShapeHandle result;
  TF_RETURN_IF_ERROR(c->Merge(a, b, &result));
  c->set_output(0, result);
  return absl::OkStatus();
})

// Reduction shape.
.SetShapeFn([](shape_inference::InferenceContext* c) {
  ShapeHandle input = c->input(0);
  int32_t axis;
  TF_RETURN_IF_ERROR(c->GetAttr("axis", &axis));
  bool keep_dims;
  TF_RETURN_IF_ERROR(c->GetAttr("keep_dims", &keep_dims));

  if (keep_dims) {
    std::vector<DimensionHandle> dims;
    for (int i = 0; i < c->Rank(input); ++i) {
      if (i == axis) {
        dims.push_back(c->MakeDim(1));
      } else {
        dims.push_back(c->Dim(input, i));
      }
    }
    c->set_output(0, c->MakeShape(dims));
  } else {
    // Remove the reduced dimension.
    std::vector<DimensionHandle> dims;
    for (int i = 0; i < c->Rank(input); ++i) {
      if (i != axis) {
        dims.push_back(c->Dim(input, i));
      }
    }
    c->set_output(0, c->MakeShape(dims));
  }
  return absl::OkStatus();
})
```

---

## Input/Output Specification

### Single Tensor

```cpp
// Fixed type.
.Input("x: float")        // Single float input

// Type determined by attribute.
.Input("x: T")            // Type from attr "T"
.Input("x: T")            // .Attr("T: type") must be defined

// Reference type.
.Input("x: Ref(T)")       // Mutable reference to tensor of type T
```

### N Tensors (Repeating Input)

```cpp
// N tensors of the same type.
.Input("inputs: N * T")   // N inputs of type T
// Requires: .Attr("N: int >= 0")
// Requires: .Attr("T: type")

// N tensors with types from a list.
.Input("inputs: Tlist")   // N inputs with types from Tlist
// Requires: .Attr("Tlist: list(type)")
```

### Number Attribute

When an input specification uses `N *`, the `N` must be defined as an integer
attribute:

```cpp
REGISTER_OP("AddN")
    .Input("inputs: N * T")
    .Output("sum: T")
    .Attr("N: int >= 1")
    .Attr("T: numbertype")
```

### Output Examples

```cpp
// Single output.
.Output("result: T")

// Multiple outputs.
.Output("values: T")
.Output("indices: int32")

// N outputs.
.Output("outputs: N * T")
```

### ArgDef Fields

Each input/output argument maps to these `ArgDef` fields:

| Spec Format           | Fields Set                          |
|----------------------|-------------------------------------|
| `"name:float"`       | `name`, `type=DT_FLOAT`            |
| `"name:T"`           | `name`, `type_attr="T"`            |
| `"name:N * T"`       | `name`, `number_attr="N"`, `type_attr="T"` |
| `"name:Ref(T)"`      | `name`, `type_attr="T"`, `is_ref=true` |
| `"name:Tlist"`       | `name`, `type_list_attr="Tlist"`   |

---

## Type Constraints

Type constraints restrict the allowed types for a type attribute.

### Basic Type Constraint

```cpp
// Restrict to specific types.
.Attr("T: {float, double}")
```

### Multiple Type Constraints

```cpp
// Multiple allowed type sets.
.Attr("T: {float, double, int32}")
```

### Using Predefined Type Sets

```cpp
// Use predefined type set names.
.Attr("T: numbertype")           // All numeric types
.Attr("T: realnumbertype")       // All real numeric types
.Attr("T: quantizedtype")        // All quantized types
```

### Multiple Type Attributes

```cpp
// Different types for inputs and outputs.
.Input("a: T")
.Input("b: T")
.Output("result: T")
.Attr("T: {float, double}")

// SrcT and DstT for type conversion.
.Input("input: SrcT")
.Output("output: DstT")
.Attr("SrcT: numbertype")
.Attr("DstT: numbertype")
```

### Allowed Values

The `allowed_values` field in `AttrDef` restricts the values an attribute can
take:

```cpp
// Int with minimum.
.Attr("N: int >= 1")

// This sets has_minimum=true, minimum=1 in the AttrDef.

// List of allowed strings.
.Attr("padding: {'SAME', 'VALID'}")

// This sets allowed_values with a list of strings.
```

---

## Gradient Registration

**Header:** `tensorflow/core/framework/gradients.h`

### REGISTER_OP_GRADIENT

```cpp
REGISTER_OP_GRADIENT("OpName", GradientFunction);
```

### Gradient Function Signature

```cpp
// Gradient function type.
typedef std::function<absl::Status(
    const Scope& scope,
    const Operation& op,
    const std::vector<Output>& grad_inputs,
    std::vector<Output>* grad_outputs)>
    GradientFunc;
```

### Example Gradient Registration

```cpp
// Register gradient for Add.
REGISTER_OP_GRADIENT("Add", AddGrad);

// Implement the gradient.
absl::Status AddGrad(const Scope& scope, const Operation& op,
                     const std::vector<Output>& grad_inputs,
                     std::vector<Output>* grad_outputs) {
  // grad_inputs[0] is the gradient with respect to the output of Add.
  // The gradient of Add with respect to each input is the same as the
  // output gradient (with broadcasting if needed).

  grad_outputs->push_back(grad_inputs[0]);  // d(a+b)/da = 1
  grad_outputs->push_back(grad_inputs[0]);  // d(a+b)/db = 1

  return absl::OkStatus();
}
```

### Complex Gradient Example

```cpp
REGISTER_OP_GRADIENT("MatMul", MatMulGrad);

absl::Status MatMulGrad(const Scope& scope, const Operation& op,
                         const std::vector<Output>& grad_inputs,
                         std::vector<Output>* grad_outputs) {
  auto grad = grad_inputs[0];
  auto a = op.input(0);
  auto b = op.input(1);

  bool transpose_a, transpose_b;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->def(), "transpose_a", &transpose_a));
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->def(), "transpose_b", &transpose_b));

  if (!transpose_a && !transpose_b) {
    // C = A * B
    // dC/dA = dC * B^T
    // dC/dB = A^T * dC
    grad_outputs->push_back(MatMul(scope, grad, b, MatMul::TransposeB(true)));
    grad_outputs->push_back(MatMul(scope, a, grad, MatMul::TransposeA(true)));
  } else if (transpose_a && !transpose_b) {
    // C = A^T * B
    grad_outputs->push_back(MatMul(scope, grad, b, MatMul::TransposeB(true)));
    grad_outputs->push_back(MatMul(scope, grad, a, MatMul::TransposeA(true)));
  }
  // ... other transposition cases

  return absl::OkStatus();
}
```

### No Gradient

Some ops do not have meaningful gradients (e.g., integer ops):

```cpp
REGISTER_OP_NO_GRADIENT("ArgMax");
REGISTER_OP_NO_GRADIENT("Shape");
```

---

## Op Registration Patterns

### Pattern 1: Stateless Unary Op

```cpp
REGISTER_OP("Relu")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Computes Rectified Linear: `max(features, 0)`.
)doc");
```

### Pattern 2: Stateless Binary Op

```cpp
REGISTER_OP("Add")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: numbertype")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn)
    .SetIsCommutative()
    .SetIsAggregate()
    .Doc(R"doc(
Returns x + y element-wise.
)doc");
```

### Pattern 3: Reduction Op

```cpp
REGISTER_OP("Sum")
    .Input("input: T")
    .Input("axis: Tidx")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("keep_dims: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // Reduction shape inference.
      return shape_inference::ReductionShape(c);
    });
```

### Pattern 4: Op with Multiple Type Outputs

```cpp
REGISTER_OP("Unique")
    .Input("x: T")
    .Output("y: T")
    .Output("idx: Tidx")
    .Attr("T: type")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(c->UnknownDim()));
      c->set_output(1, c->input(0));
      return absl::OkStatus();
    });
```

### Pattern 5: Stateful Op

```cpp
REGISTER_OP("VariableV2")
    .Output("ref: Ref(T)")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      PartialTensorShape shape;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &shape));
      ShapeHandle output_shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(shape, &output_shape));
      c->set_output(0, output_shape);
      return absl::OkStatus();
    });
```

### Pattern 6: Op with N Inputs

```cpp
REGISTER_OP("ConcatV2")
    .Input("values: N * T")
    .Input("axis: Tidx")
    .Output("output: T")
    .Attr("N: int >= 2")
    .Attr("T: type")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return shape_inference::ConcatV2Shape(c);
    });
```

### Pattern 7: Convolution Op

```cpp
REGISTER_OP("Conv2D")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {float, double, int32, uint8, int16, int8, int64, bfloat16, half}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr("padding: {'SAME', 'VALID'}")
    .Attr("explicit_paddings: list(int) = []")
    .Attr("data_format: {'NHWC', 'NCHW'} = 'NHWC'")
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return shape_inference::Conv2DShape(c);
    });
```

---

## Op Implementation Files

### Core Op Definitions

TensorFlow's core op definitions are located in
`tensorflow/core/ops/`. Each file groups related operations:

| File                        | Ops                                              |
|----------------------------|--------------------------------------------------|
| `array_ops.cc`             | Concat, Split, Reshape, ExpandDims, Slice, etc.  |
| `math_ops.cc`              | Add, Sub, Mul, Div, MatMul, etc.                 |
| `nn_ops.cc`                | Conv2D, Pooling, Softmax, Relu, etc.             |
| `control_flow_ops.cc`      | Switch, Merge, Enter, Exit, NextIteration        |
| `data_flow_ops.cc`         | Queue ops, DynamicStitch, etc.                   |
| `state_ops.cc`             | Variable, Assign, AssignAdd, etc.                |
| `sparse_ops.cc`            | SparseTensor ops                                 |
| `training_ops.cc`          | GradientDescent, Adam, RMSProp, etc.             |
| `string_ops.cc`            | String processing ops                            |
| `spectral_ops.cc`          | FFT, DCT, etc.                                   |
| `bitwise_ops.cc`           | Bitwise operations                               |
| `candidate_sampling_ops.cc`| Candidate sampling for training                  |
| `collective_ops.cc`        | Collective communication ops                     |
| `functional_ops.cc`        | PartitionedCall, StatefulPartitionedCall         |
| `random_ops.cc`            | Random number generation                         |
| `stateless_random_ops.cc`  | Stateless random ops                             |
| `audio_ops.cc`             | Audio decoding/encoding                          |
| `image_ops.cc`             | Image processing ops                             |
| `io_ops.cc`                | File I/O ops                                     |
| `parsing_ops.cc`           | Example parsing ops                              |
| `lookup_ops.cc`            | Lookup table ops                                 |
| `set_ops.cc`               | Set operations                                   |
| `linalg_ops.cc`            | Linear algebra ops (Cholesky, SVD, etc.)         |
| `special_math_ops.cc`      | Special functions (Bessel, Beta, etc.)           |
| `legacy_math_ops.cc`       | Deprecated math ops                              |
| `debug_ops.cc`             | Debugging ops                                    |
| `summary_ops.cc`           | Summary (TensorBoard) ops                        |
| `sync_ops.cc`              | Synchronization ops                              |
| `checkpoint_ops.cc`        | Checkpoint operations                            |
| `dataset_ops.cc`           | tf.data pipeline ops                             |
| `composite_tensor_ops.cc`  | Composite tensor ops                             |
| `quantization_ops.cc`      | Quantization ops                                 |
| `uniform_quant_ops.cc`     | Uniform quantization ops                         |
| `boosted_trees_ops.cc`     | Boosted trees ops                                |
| `clustering_ops.cc`        | Clustering ops                                   |
| `ctc_ops.cc`               | CTC loss ops                                     |
| `cudnn_rnn_ops.cc`         | cuDNN RNN ops                                    |
| `decode_proto_ops.cc`      | Protobuf decoding ops                            |
| `encode_proto_ops.cc`      | Protobuf encoding ops                            |
| `filesystem_ops.cc`        | Filesystem ops                                   |
| `word2vec_ops`             | Word2Vec ops                                     |
| `stochastic_cast_op.cc`    | Stochastic casting ops                           |
| `batch_ops.cc`             | Batching ops                                     |
| `count_ops.cc`             | Counting ops                                     |

### Gradient Definition Files

| File                     | Gradients For                    |
|-------------------------|----------------------------------|
| `array_grad.cc`         | Array ops gradients              |
| `math_grad.cc`          | Math ops gradients               |
| `nn_grad.cc`            | Neural network ops gradients     |
| `functional_grad.cc`    | Functional ops gradients         |
| `stateless_random_grad.cc` | Stateless random ops gradients |
| `data_flow_grad.cc`     | Data flow ops gradients          |
| `image_grad.cc`         | Image ops gradients              |
| `linalg_grad.cc`        | Linear algebra ops gradients     |
| `spectral_grad.cc`      | Spectral ops gradients           |
| `sparse_grad.cc`        | Sparse ops gradients             |

---

## Custom Op Registration

### Step-by-Step Guide

#### Step 1: Define the Op

Create a file `my_ops.cc`:

```cpp
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

REGISTER_OP("MyCustomOp")
    .Input("input: T")
    .Input("weights: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("stride: int >= 1")
    .Attr("padding: {'SAME', 'VALID'} = 'VALID'")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
      tensorflow::shape_inference::ShapeHandle weights;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &weights));
      // ... compute output shape ...
      c->set_output(0, input);
      return absl::OkStatus();
    })
    .Doc(R"doc(
Applies a custom operation to the input tensor.

input: A 4-D tensor.
weights: A 4-D weight tensor.
output: The output tensor.
stride: The stride for the operation.
padding: The padding algorithm.
)doc");
```

#### Step 2: Implement the Kernel

```cpp
template <typename Device, typename T>
class MyCustomOpOp : public tensorflow::OpKernel {
 public:
  explicit MyCustomOpOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("stride", &stride_));
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    const tensorflow::Tensor& input = context->input(0);
    const tensorflow::Tensor& weights = context->input(1);

    // Validate shapes.
    OP_REQUIRES(context, input.dims() == 4,
                tensorflow::errors::InvalidArgument(
                    "Input must be 4-D, got shape: ",
                    input.shape().DebugString()));

    // Allocate output.
    tensorflow::Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
        context->allocate_output(0, input.shape(), &output));

    // Compute.
    const auto& d = context->eigen_device<Device>();
    // ... Eigen or custom computation ...
  }

 private:
  int stride_;
};
```

#### Step 3: Register the Kernel

```cpp
// CPU kernel.
REGISTER_KERNEL_BUILDER(
    Name("MyCustomOp").Device(tensorflow::DEVICE_CPU),
    MyCustomOpOp<tensorflow::CPUDevice, float>);

// GPU kernel.
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("MyCustomOp").Device(tensorflow::DEVICE_GPU),
    MyCustomOpOp<tensorflow::GPUDevice, float>);
#endif
```

#### Step 4: Register the Gradient (Optional)

```cpp
#include "tensorflow/core/framework/gradients.h"

REGISTER_OP_GRADIENT("MyCustomOp", MyCustomOpGrad);

tensorflow::Status MyCustomOpGrad(
    const tensorflow::Scope& scope,
    const tensorflow::Operation& op,
    const std::vector<tensorflow::Output>& grad_inputs,
    std::vector<tensorflow::Output>* grad_outputs) {
  // Implement gradient computation.
  // ...
  return absl::OkStatus();
}
```

#### Step 5: Build and Use

```python
# Python usage (after building and installing the custom op library).
import tensorflow as tf

# Load the custom op library.
custom_ops = tf.load_op_library('./my_custom_ops.so')

# Use the custom op.
result = custom_ops.my_custom_op(input_tensor, weights_tensor,
                                  stride=2, padding='SAME')
```

---

## Op Versioning

### OpDef Versioning

Ops can be versioned using the `Deprecated` method:

```cpp
REGISTER_OP("OldOp")
    .Deprecated(12, "Use NewOp instead.")
    .Input("input: float")
    .Output("output: float");
```

### GraphDef Versioning

The `VersionDef` message tracks producer and consumer versions:

```protobuf
message VersionDef {
  int32 producer = 1;          // Version of the producer
  int32 min_consumer = 2;      // Minimum consumer version
  repeated int32 bad_consumers = 3;  // Known-bad consumer versions
}
```

### Version Compatibility

When adding a new optional attribute to an existing op:

```cpp
// Original op (version 1):
REGISTER_OP("MyOp")
    .Input("x: float")
    .Output("y: float");

// Updated op (version 2) with new optional attribute:
REGISTER_OP("MyOp")
    .Input("x: float")
    .Output("y: float")
    .Attr("new_option: bool = false");  // Default makes it backward compatible
```

### OpDef Upgrade

TensorFlow can upgrade GraphDefs from older versions by adding default values
for new attributes:

```cpp
// In op_def_util.cc.
absl::Status AddDefaultsToNodeDef(const OpDef& op_def, NodeDef* node_def);
```

### Backward Compatibility Rules

1. **Adding a new optional attribute** (with default): Always safe
2. **Adding a new required attribute**: Breaking change
3. **Removing an attribute**: Breaking change (unless all uses had default)
4. **Changing an attribute type**: Breaking change
5. **Adding a new optional input**: Safe if kernel handles missing input
6. **Removing an input**: Breaking change
7. **Changing an output shape**: May break dependent ops

---

## OpRegistry

**Header:** `tensorflow/core/framework/op.h`

### OpRegistryInterface

```cpp
class OpRegistryInterface {
 public:
  virtual ~OpRegistryInterface() = default;

  // Look up an OpDef by name.
  virtual absl::Status LookUp(const std::string& op_type_name,
                              const OpRegistrationData** op_reg_data) const = 0;

  // Convenience: get just the OpDef.
  absl::Status LookUpOpDef(const std::string& op_type_name,
                           const OpDef** op_def) const;
};
```

### OpRegistry

```cpp
class OpRegistry : public OpRegistryInterface {
 public:
  typedef std::function<absl::Status(OpRegistrationData*)>
      OpRegistrationDataFactory;

  // Register an op definition factory.
  void Register(const OpRegistrationDataFactory& op_data_factory);

  // Look up by name.
  absl::Status LookUp(const std::string& op_type_name,
                      const OpRegistrationData** op_reg_data) const override;

  // Get by name (returns nullptr if not found).
  const OpRegistrationData* LookUp(const std::string& op_type_name) const;

  // Export all registered ops.
  void Export(bool include_internal, OpList* ops) const;

  // Debug string.
  std::string DebugString(bool include_internal) const;

  // Global singleton.
  static OpRegistry* Global();

  // Get all registered ops.
  void GetRegisteredOps(std::vector<OpDef>* op_defs);
  void GetOpRegistrationData(std::vector<OpRegistrationData>* op_data);

  // Deferred registration control.
  void DeferRegistrations();
  absl::Status ProcessRegistrations() const;
  void ClearDeferredRegistrations();

  // Watcher for registration events.
  typedef std::function<absl::Status(const absl::Status&, const OpDef&)> Watcher;
  absl::Status SetWatcher(const Watcher& watcher);
};
```

### OpRegistrationData

```cpp
struct OpRegistrationData {
  OpDef op_def;                    // The op definition
  OpShapeInferenceFn shape_inference_fn;  // Shape inference function
  // ... other registration data
};
```

### OpListOpRegistry

```cpp
// Adapter to use an OpList as an OpRegistryInterface.
class OpListOpRegistry : public OpRegistryInterface {
 public:
  explicit OpListOpRegistry(const OpList* op_list);
  absl::Status LookUp(const std::string& op_type_name,
                      const OpRegistrationData** op_reg_data) const override;
};
```

---

## Summary

### Op Definition Checklist

When defining a new TensorFlow operation:

1. [ ] Define the op with `REGISTER_OP`
2. [ ] Specify all inputs with `.Input()`
3. [ ] Specify all outputs with `.Output()`
4. [ ] Define all attributes with `.Attr()` including defaults
5. [ ] Add shape inference with `.SetShapeFn()`
6. [ ] Add documentation with `.Doc()`
7. [ ] Mark stateful ops with `.SetIsStateful()`
8. [ ] Mark deprecated ops with `.Deprecated()`
9. [ ] Implement the kernel class inheriting `OpKernel`
10. [ ] Register the kernel with `REGISTER_KERNEL_BUILDER`
11. [ ] Implement gradient with `REGISTER_OP_GRADIENT` (if differentiable)
12. [ ] Add tests for the op, kernel, and gradient
