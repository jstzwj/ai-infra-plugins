# TensorFlow C++ API Reference

## Table of Contents

1. [C++ API Overview](#c-api-overview)
2. [Scope](#scope)
3. [Input and Output](#input-and-output)
4. [Source Operations](#source-operations)
5. [Arithmetic Operations](#arithmetic-operations)
6. [Array Operations](#array-operations)
7. [Neural Network Operations](#neural-network-operations)
8. [Math Operations](#math-operations)
9. [Control Flow Operations](#control-flow-operations)
10. [IO Operations](#io-operations)
11. [Random Operations](#random-operations)
12. [ClientSession](#clientsession)
13. [Example Programs](#example-programs)

---

## C++ API Overview

The TensorFlow C++ API (`tensorflow/cc`) provides a C++ interface for building
and executing computation graphs. It is organized around the `Scope` class for
graph construction and the `ClientSession` class for execution.

### Header Files

```c++
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/const_op.h"
```

### Namespace

All operations are in the `tensorflow::ops` namespace:

```c++
using namespace tensorflow;
using namespace tensorflow::ops;
```

### Design Philosophy

The C++ API is designed around the following patterns:

- **Scope-based construction**: Operations are built within a `Scope` that
  manages naming, device placement, and error handling.
- **Return-value chaining**: Operations return `Output` objects that can be
  passed directly as inputs to subsequent operations.
- **Automatic type deduction**: Operation constructors deduce input types
  from their `Input` arguments.
- **Deferred error handling**: Errors are stored in the scope and checked
  at execution time, allowing construction to continue after failures.

---

## Scope

### Overview

`Scope` is the primary context for building TensorFlow graphs. Every operation
requires a `Scope` as its first argument. Scopes form a hierarchy, with child
scopes inheriting properties from parents.

### Creating Scopes

```c++
// Root scope - creates a new Graph
Scope root = Scope::NewRootScope();

// Sub-scope - adds name prefix
Scope linear = root.NewSubScope("linear");

// The sub-scope adds "linear/" prefix to all op names
auto W = Variable(linear.WithOpName("W"), {2, 2}, DT_FLOAT);
// W is named "linear/W"
```

### Scope Modifiers

Each modifier returns a new scope with the specified property changed:

#### WithOpName

```c++
// Set the operation name
Scope s = root.WithOpName("my_matmul");
// Operations created with s will have name "my_matmul"
// If already used, TensorFlow appends a suffix: "my_matmul_1"

// WithOpName supports variadic arguments for formatted names
int idx = 3;
Scope s = linear.WithOpName("b_", idx);
// Name: "linear/b_3"
```

#### WithDevice

```c++
// Set device placement
Scope gpu_scope = root.WithDevice("/gpu:0");
auto gpu_result = MatMul(gpu_scope, a, b);

// Empty string means unconstrained
Scope unconstrained = root.WithDevice("");

// CPU device
Scope cpu_scope = root.WithDevice("/cpu:0");
```

#### WithAssignedDevice

```c++
// Set assigned device (used for distributed training)
Scope assigned = root.WithAssignedDevice("/job:worker/task:0/gpu:0");
```

#### WithControlDependencies

```c++
// Add control dependencies
std::vector<Operation> deps = {op1, op2};
Scope controlled = root.WithControlDependencies(deps);
// Operations in this scope will execute only after op1 and op2 complete

// Control dependency from an Output
Scope controlled = root.WithControlDependencies(output);
```

#### WithNoControlDependencies

```c++
// Clear all control dependencies
Scope clean = parent_scope.WithNoControlDependencies();
```

#### WithKernelLabel

```c++
// Set kernel label (for custom kernel selection)
Scope labeled = root.WithKernelLabel("my_custom_kernel");
```

#### WithXlaCluster

```c++
// Assign operations to an XLA cluster
Scope clustered = root.WithXlaCluster("my_cluster");
```

#### ExitOnError

```c++
// Exit immediately on error instead of storing status
Scope strict = root.ExitOnError();
// Construction errors will LOG(FATAL) immediately
```

#### ColocateWith

```c++
// Co-locate with another operation
Scope colocated = root.ColocateWith(existing_op);

// Co-locate from an Output
Scope colocated = root.ColocateWith(output);
```

#### ClearColocation

```c++
// Clear colocation constraints
Scope free = scope.ClearColocation();
```

### Scope Properties

```c++
// Check if scope is valid (no errors)
bool is_ok = scope.ok();

// Get current status
absl::Status status = scope.status();

// Get underlying Graph object
Graph* graph = scope.graph();

// Get shared pointer to graph
std::shared_ptr<Graph> graph_ptr = scope.graph_as_shared_ptr();

// Get control dependencies
const std::vector<Operation>& deps = scope.control_deps();
```

### Exporting Graph

```c++
// Convert graph to GraphDef proto
GraphDef graph_def;
absl::Status s = root.ToGraphDef(&graph_def);
if (!s.ok()) {
    // Handle error
}

// Convert to a new Graph (validates and cleans up)
Graph g(GraphDef());
absl::Status s = root.ToGraph(&g);

// With debug info
GraphDef graph_def;
absl::Status s = root.ToGraphDef(&graph_def, /*include_debug_info=*/true);
```

### CompositeOpScopes

```c++
// Get scopes for composite operations
CompositeOpScopes scopes = scope.GetCompositeOpScopes("my_composite");
// scopes.child: for internal operations
// scopes.last: for the final output operation
```

---

## Input and Output

### Input

`Input` represents a tensor value that can be passed to an operation:

```c++
class Input {
public:
    // From a tensor
    Input(const Tensor& tensor);

    // From an Output (connecting graph operations)
    Input(const Output& output);

    // From a scalar (creates a Const op)
    Input(const float& scalar_value);
    Input(const double& scalar_value);
    Input(const int32& scalar_value);
    Input(const int64_t& scalar_value);
    Input(const bool& scalar_value);
    Input(const string& scalar_value);

    // From initializer list (creates a Const op)
    Input(std::initializer_list<float> v);
    Input(std::initializer_list<int32> v);

    // Shared pointer to a tensor
    Input(const std::shared_ptr<Tensor>& tensor);
};
```

### Output

`Output` represents a tensor produced by an operation:

```c++
class Output {
public:
    // Default constructor
    Output() = default;

    // From an Operation and output index
    Output(Operation op, int index);

    // Get the operation
    Operation op() const;

    // Get the output index
    int index() const;

    // Get the data type
    DataType type() const;

    // Get the shape (requires graph execution)
    TensorShape shape() const;

    // Get the node name
    string name() const;
};
```

### InputList

`InputList` represents a list of inputs:

```c++
class InputList {
public:
    InputList(const std::vector<Input>& inputs);
    InputList(std::initializer_list<Input> inputs);

    // Iterate
    std::vector<Input>::iterator begin();
    std::vector<Input>::iterator end();
    size_t size() const;
};
```

### Operation Chaining

```c++
// Operations return Output, which can be passed as Input
Scope root = Scope::NewRootScope();

auto a = Const(root, {1.0f, 2.0f});          // Output
auto b = Const(root, {3.0f, 4.0f});          // Output
auto c = Add(root, a, b);                     // Output from Output inputs
auto d = Mul(root, c, c);                     // Output from Output input

// Can also pass raw scalars (creates Const ops automatically)
auto e = Add(root, a, 10.0f);                // 10.0f creates a Const
```

---

## Source Operations

### Const

```c++
// Create a constant tensor
auto c1 = Const(root, { {1.0f, 2.0f}, {3.0f, 4.0f} });
// Shape: [2, 2], Type: DT_FLOAT

// Scalar constant
auto c2 = Const(root, 42.0f);
// Shape: [], Type: DT_FLOAT

// From a Tensor
Tensor tensor(DT_FLOAT, TensorShape({3, 2}));
tensor.matrix<float>() << 1, 2, 3, 4, 5, 6;
auto c3 = Const(root, tensor);

// With explicit shape
auto c4 = Const(root, {1.0f, 2.0f, 3.0f}, TensorShape({3}));

// With type specification
auto c5 = Const(root, {1, 2, 3}, DT_INT32);

// Integer constants
auto c6 = Const(root, 42);
auto c7 = Const(root, {1, 2, 3, 4});

// String constant
auto c8 = Const(root, "hello world");
auto c9 = Const(root, {"hello", "world"});

// Boolean constant
auto c10 = Const(root, true);
```

### Placeholder

```c++
// Create a placeholder for feeding values at runtime
auto x = Placeholder(root, DT_FLOAT);
// Shape: unknown

// With specific shape
auto y = Placeholder(root, DT_FLOAT,
    Placeholder::Shape({-1, 10}));  // batch_size x 10

// With fully defined shape
auto z = Placeholder(root, DT_FLOAT,
    Placeholder::Shape({32, 100}));
```

### PlaceholderWithDefault

```c++
// Placeholder that uses a default value when no feed is provided
auto default_val = Const(root, {1.0f, 2.0f, 3.0f});
auto x = PlaceholderWithDefault(root, default_val, {3});
```

---

## Arithmetic Operations

### Add

```c++
auto result = Add(root, a, b);

// With optional attributes
auto result = Add(root, a, b, Add::Attrs().NoopIfCompatibleTypes(true));
```

### Sub

```c++
auto result = Sub(root, a, b);
```

### Mul

```c++
auto result = Mul(root, a, b);
```

### Div

```c++
auto result = Div(root, a, b);

// Floor division
auto result = FloorDiv(root, a, b);

// Truncated division
auto result = TruncateDiv(root, a, b);

// Real division (Python 3 semantics)
auto result = RealDiv(root, a, b);
```

### Mod

```c++
auto result = Mod(root, a, b);      // Floor mod
auto result = FloorMod(root, a, b);
auto result = TruncateMod(root, a, b);
```

### Negate

```c++
auto result = Neg(root, a);
```

### Abs

```c++
auto result = Abs(root, a);
```

### Square

```c++
auto result = Square(root, a);
```

### Sqrt

```c++
auto result = Sqrt(root, a);
auto result = Rsqrt(root, a);  // 1/sqrt(a)
```

### Pow

```c++
auto result = Pow(root, base, exponent);
```

### Exp and Log

```c++
auto result = Exp(root, a);
auto result = Log(root, a);
auto result = Log1p(root, a);     // log(1 + a)
auto result = Expm1(root, a);     // exp(a) - 1
```

### Min and Max

```c++
// Element-wise min/max
auto result = Minimum(root, a, b);  // or Min
auto result = Maximum(root, a, b);  // or Max

// Reduction along axis
auto min_val = Min(root, input, axis);
auto max_val = Max(root, input, axis);
```

### Floor, Ceil, Round

```c++
auto result = Floor(root, a);
auto result = Ceil(root, a);
auto result = Round(root, a);
```

### Sign

```c++
auto result = Sign(root, a);  // -1, 0, or 1
```

### SquaredDifference

```c++
auto result = SquaredDifference(root, a, b);  // (a - b)^2
```

---

## Array Operations

### Reshape

```c++
auto reshaped = Reshape(root, tensor, {2, 3});

// With computed shape
auto shape = Const(root, {4, 5});
auto reshaped = Reshape(root, tensor, shape);
```

### ExpandDims

```c++
auto expanded = ExpandDims(root, input, axis);
// axis=0: add batch dimension
// axis=-1: add trailing dimension
```

### Squeeze

```c++
auto squeezed = Squeeze(root, input);
// Remove all size-1 dimensions

auto squeezed = Squeeze(root, input, Squeeze::Attrs().Axis({0, 2}));
// Remove specific dimensions
```

### Transpose

```c++
auto perm = Const(root, {1, 0});  // Transpose 2D
auto transposed = Transpose(root, matrix, perm);

// Conjugate transpose for complex tensors
auto ct = ConjugateTranspose(root, complex_tensor, perm);
```

### Concat

```c++
// Concatenate along axis 0
auto result = Concat(root, {tensor1, tensor2, tensor3}, 0);

// Concatenate along axis 1
auto result = Concat(root, InputList({a, b}), 1);
```

### Split

```c++
// Split into 3 parts along axis 0
auto split_result = Split(root, 0, tensor, 3);
// split_result is a std::vector<Output> of 3 outputs

// Split with explicit sizes
auto split_sizes = Const(root, {2, 3, 5});
auto split_result = SplitV(root, tensor, split_sizes, 0);
```

### Stack and Unstack

```c++
// Stack (Pack) - join tensors along a new axis
auto stacked = Stack(root, {a, b, c}, Stack::Attrs().Axis(0));
// Equivalent to tf.stack([a, b, c], axis=0)

// Unstack (Unpack) - split tensor along an axis
auto unstacked = Unstack(root, tensor, 3, Unstack::Attrs().Axis(0));
// Returns 3 tensors
```

### Slice

```c++
auto begin = Const(root, {0, 0});
auto size = Const(root, {2, 3});
auto sliced = Slice(root, tensor, begin, size);
```

### StridedSlice

```c++
auto begin = Const(root, {0, 0});
auto end = Const(root, {4, 4});
auto strides = Const(root, {1, 2});

auto result = StridedSlice(root, tensor, begin, end, strides,
    StridedSlice::Attrs()
        .BeginMask(0)
        .EndMask(0)
        .EllipsisMask(0)
        .NewAxisMask(0)
        .ShrinkAxisMask(0));
```

### Gather

```c++
auto gathered = Gather(root, params, indices);

// Gather with axis
auto gathered = Gather(root, params, indices,
    Gather::Attrs().Axis(1));

// GatherNd
auto gathered = GatherNd(root, params, indices);
```

### ScatterNd

```c++
auto scattered = ScatterNd(root, indices, updates, shape);
```

### Tile

```c++
auto multiples = Const(root, {2, 3});
auto tiled = Tile(root, tensor, multiples);
```

### Pad

```c++
// Constant padding
auto paddings = Const(root, {{1, 1}, {2, 2}});
auto padded = Pad(root, tensor, paddings);

// Padding with specific value
auto pad_value = Const(root, 0.0f);
auto padded = PadV2(root, tensor, paddings, pad_value);

// Mirror padding
auto mirrored = MirrorPad(root, tensor, paddings,
    MirrorPad::REFLECT);
```

### Reverse

```c++
auto axes = Const(root, {0});
auto reversed = Reverse(root, tensor, axes);

// Reverse sequence
auto seq_lengths = Const(root, {3, 2, 4});
auto reversed = ReverseSequence(root, tensor, seq_lengths,
    ReverseSequence::Attrs().SeqDim(1).BatchDim(0));
```

### Shape and Size

```c++
// Get shape as tensor
auto shape = Shape(root, tensor);           // DT_INT32
auto shape_i64 = Shape(root, tensor,
    Shape::Attrs().OutType(DT_INT64));

// Get total number of elements
auto size = Size(root, tensor);

// Get rank
auto rank = Rank(root, tensor);

// Get shape as list
auto shape_n = ShapeN(root, {tensor1, tensor2, tensor3});
```

### BroadcastTo

```c++
auto shape = Const(root, {3, 4});
auto broadcasted = BroadcastTo(root, tensor, shape);
```

### Where

```c++
// Get indices of true elements
auto indices = Where(root, condition);

// Select from two tensors
auto result = Where3(root, condition, x, y);
```

---

## Neural Network Operations

### Conv2D

```c++
// 2D convolution
auto conv = Conv2D(root, input, filter,
    {1, 1, 1, 1},  // stride [batch, height, width, channels]
    "SAME");        // padding

// With dilation
auto conv = Conv2D(root, input, filter,
    {1, 1, 1, 1}, "SAME",
    Conv2D::Attrs()
        .Dilations({1, 2, 2, 1})
        .DataFormat("NHWC"));

// Explicit padding
auto conv = Conv2D(root, input, filter,
    {1, 2, 2, 1}, "VALID");
```

### Conv2DBackpropInput (Transpose Convolution)

```c++
// Transposed convolution (deconvolution)
auto output_shape = Const(root, {1, 28, 28, 3});
auto deconv = Conv2DBackpropInput(root, output_shape, filter, input,
    {1, 2, 2, 1}, "SAME");
```

### Conv2DBackpropFilter

```c++
// Filter gradient
auto filter_grad = Conv2DBackpropFilter(root, input, filter_shape, out_backprop,
    {1, 1, 1, 1}, "SAME");
```

### DepthwiseConv2dNative

```c++
// Depthwise separable convolution
auto depthwise = DepthwiseConv2dNative(root, input, filter,
    {1, 1, 1, 1}, "SAME");
```

### BiasAdd

```c++
// Add bias to the last dimension
auto biased = BiasAdd(root, conv_output, bias);

// With specific data format
auto biased = BiasAdd(root, conv_output, bias,
    BiasAdd::Attrs().DataFormat("NCHW"));
```

### Activation Functions

```c++
auto relu = Relu(root, input);
auto relu6 = Relu6(root, input);
auto elu = Elu(root, input);
auto selu = Selu(root, input);
auto sigmoid = Sigmoid(root, input);
auto tanh = Tanh(root, input);
auto softplus = Softplus(root, input);
auto softsign = Softsign(root, input);
auto leaky_relu = LeakyRelu(root, input,
    LeakyRelu::Attrs().Alpha(0.1f));
```

### Softmax

```c++
auto softmax = Softmax(root, logits);
auto log_softmax = LogSoftmax(root, logits);
```

### LRN (Local Response Normalization)

```c++
auto lrn = LRN(root, input,
    LRN::Attrs()
        .DepthRadius(5)
        .Bias(1.0f)
        .Alpha(1.0f)
        .Beta(0.5f));
```

### L2Loss

```c++
// Sum of squares / 2 (for regularization)
auto loss = L2Loss(root, weights);
```

### Dropout

```c++
auto dropped = Dropout(root, input,
    Dropout::Attrs()
        .Rate(0.5f)
        .Seed(42)
        .Seed2(0));
```

### TopK

```c++
auto topk = TopK(root, input, k);
// topk.values: top k values
// topk.indices: top k indices

// InTopK (check if predictions are in top k)
auto in_topk = InTopK(root, predictions, targets, k);
```

---

## Math Operations

### MatMul

```c++
// Matrix multiplication
auto result = MatMul(root, a, b);

// Transpose inputs
auto result = MatMul(root, a, b,
    MatMul::Attrs()
        .TransposeA(false)
        .TransposeB(true));

// Batch matrix multiplication
auto result = BatchMatMul(root, a, b);
auto result_v2 = BatchMatMulV2(root, a, b,
    BatchMatMulV2::Attrs().AdjX(false).AdjY(false));
auto result_v3 = BatchMatMulV3(root, a, b);
```

### Einsum

```c++
// Einstein summation
auto result = Einsum(root, {a, b}, "ij,jk->ik");

// Batch matrix multiply
auto result = Einsum(root, {a, b}, "bij,bjk->bik");
```

### Matrix Operations

```c++
// Determinant
auto det = MatrixDeterminant(root, matrix);

// Inverse
auto inv = MatrixInverse(root, matrix);

// Cholesky decomposition
auto chol = Cholesky(root, matrix);

// Cholesky solve
auto solution = CholeskySolve(root, chol, rhs);

// QR decomposition
auto qr = Qr(root, matrix);

// SVD
auto svd = Svd(root, matrix);

// Norm
auto norm = Norm(root, tensor,
    Norm::Attrs().Axis({0}).Ord("euclidean"));

// Trace
auto trace = Trace(root, matrix);

// Matrix band part (extract triangular)
auto triu = MatrixBandPart(root, matrix, -1, 0);  // Upper triangle
auto tril = MatrixBandPart(root, matrix, 0, -1);  // Lower triangle

// Matrix diagonal
auto diag = MatrixDiag(root, diagonal);
auto diag_part = MatrixDiagPart(root, matrix);

// Matrix set diagonal
auto updated = MatrixSetDiag(root, matrix, diagonal);
```

### Cross Product

```c++
auto cross = Cross(root, a, b);
```

### Cumulative Operations

```c++
auto cumsum = Cumsum(root, input, axis);
auto cumprod = Cumprod(root, input, axis);
```

### Segment Operations

```c++
auto seg_sum = UnsortedSegmentSum(root, data, segment_ids, num_segments);
auto seg_max = UnsortedSegmentMax(root, data, segment_ids, num_segments);
auto seg_min = UnsortedSegmentMin(root, data, segment_ids, num_segments);
auto seg_prod = UnsortedSegmentProd(root, data, segment_ids, num_segments);
```

### Reduction Operations

```c++
// Sum along all axes
auto total = Sum(root, input, Const(root, {-1}), Sum::Attrs().KeepDims(false));

// Mean
auto mean = Mean(root, input, axis);

// Product
auto prod = Prod(root, input, axis);

// Min/Max
auto min = Min(root, input, axis);
auto max = Max(root, input, axis);

// All / Any (boolean)
auto all = All(root, bool_input, axis);
auto any = Any(root, bool_input, axis);
```

---

## Control Flow Operations

### Switch and Merge

```c++
// Switch: route tensor to one of two outputs based on predicate
auto switch_out = Switch(root, input, predicate);
// switch_out.output_true: tensor if predicate is true
// switch_out.output_false: tensor if predicate is false

// Merge: forward the first available input
auto merged = Merge(root, {tensor1, tensor2, tensor3});
```

### Enter and Exit

```c++
// Enter: push tensor into a frame (for loops)
auto entered = Enter(root, input, "frame_name");

// Exit: pop tensor from a frame
auto exited = Exit(root, entered);
```

### NextIteration

```c++
// Pass value to the next iteration of a loop
auto next = NextIteration(root, loop_value);
```

### LoopCond

```c++
// Loop condition (for While loops)
auto loop_cond = LoopCond(root, condition);
```

### NoOp

```c++
// No-op operation (useful for control dependencies)
auto noop = NoOp(root);
```

### Identity

```c++
// Identity (useful for adding control dependencies)
auto identity = Identity(root, input);
```

### StopGradient

```c++
// Prevent gradient computation through this tensor
auto stopped = StopGradient(root, input);
```

### While Loop

```c++
#include "tensorflow/cc/ops/while_loop.h"

// Build a while loop
Scope root = Scope::NewRootScope();

// Define loop condition
auto cond = [](const Scope& scope, const std::vector<Output>& inputs,
               Output* output) -> Status {
    auto i = inputs[0];
    auto limit = Const(scope, 10);
    *output = Less(scope, i, limit);
    return scope.status();
};

// Define loop body
auto body = [](const Scope& scope, const std::vector<Output>& inputs,
               std::vector<Output>* outputs) -> Status {
    auto i = inputs[0];
    auto accumulator = inputs[1];
    auto one = Const(scope, 1);
    auto new_i = Add(scope, i, one);
    auto new_acc = Add(scope, accumulator, i);
    *outputs = {new_i, new_acc};
    return scope.status();
};

// Create the while loop
auto initial_i = Const(root, 0);
auto initial_acc = Const(root, 0.0f);
std::vector<Output> loop_outputs;
TF_CHECK_OK(BuildWhileLoop(root, {initial_i, initial_acc}, cond, body,
                            "my_loop", &loop_outputs));
// loop_outputs[0]: final i (10)
// loop_outputs[1]: sum 0+1+2+...+9
```

---

## IO Operations

### File Operations

```c++
// Read entire file
auto content = ReadFile(root, filename);

// Write to file
auto write = WriteFile(root, filename, content);
```

### Image Operations

```c++
// Decode images
auto jpeg = DecodeJpeg(root, content,
    DecodeJpeg::Attrs().Channels(3));

auto png = DecodePng(root, content,
    DecodePng::Attrs().Channels(3));

// Encode images
auto encoded_jpeg = EncodeJpeg(root, image);
auto encoded_png = EncodePng(root, image);

// Resize images
auto resized = ResizeBilinear(root, images, size);
auto resized_nn = ResizeNearestNeighbor(root, images, size);

// Crop and resize
auto crops = CropAndResize(root, image, boxes, box_ind,
    CropAndResize::Attrs().CropSize({224, 224}));
```

### Text Operations

```c++
// Decode CSV
auto csv = DecodeCSV(root, content,
    DecodeCSV::Attrs().RecordDefaults({""}));

// String operations
auto joined = StringJoin(root, {a, b});
auto split = StringSplit(root, input, delimiter);
auto to_number = StringToNumber(root, input);
```

---

## Random Operations

### RandomStandardNormal

```c++
// Standard normal distribution
auto result = RandomStandardNormal(root, shape, DT_FLOAT);

// With seed
auto result = RandomStandardNormal(root, shape, DT_FLOAT,
    RandomStandardNormal::Attrs().Seed(42).Seed2(0));
```

### RandomUniform

```c++
// Uniform distribution [0, 1)
auto result = RandomUniform(root, shape, DT_FLOAT);

// Uniform integers [minval, maxval)
auto result = RandomUniformInt(root, shape, minval, maxval);
```

### TruncatedNormal

```c++
// Truncated normal (values beyond 2 std devs are rejected)
auto result = TruncatedNormal(root, shape, DT_FLOAT);
```

### RandomShuffle

```c++
// Randomly shuffle along first dimension
auto shuffled = RandomShuffle(root, input);
```

### StatelessRandom

```c++
// Stateless random (deterministic with seed)
auto result = StatelessRandomNormal(root, shape, seed);

auto result = StatelessRandomUniform(root, shape, seed);

auto result = StatelessTruncatedNormal(root, shape, seed);
```

---

## ClientSession

### Overview

`ClientSession` executes the graph built with `Scope`. It manages the
connection to the TensorFlow runtime and handles tensor feeding.

### Creation

```c++
// Create with default options
ClientSession session(root);

// With custom options
SessionOptions options;
options.config.set_allow_soft_placement(true);
options.config.mutable_gpu_options()->set_allow_growth(true);
ClientSession session(root, options);

// With target
ClientSession session(root, "grpc://localhost:2222");
```

### Running Operations

```c++
// Simple run (no feeds)
std::vector<Tensor> outputs;
TF_CHECK_OK(session.Run({}, {output_op}, &outputs));

// Run with feeds
ClientSession::FeedType feed;
feed.insert({placeholder_op, input_tensor});
std::vector<Tensor> outputs;
TF_CHECK_OK(session.Run(feed, {result_op}, &outputs));

// Run with targets (operations to execute but not fetch)
TF_CHECK_OK(session.Run(feed, {output_op}, {target_op}, &outputs));
```

### RunWithFetchOutput

```c++
// Run and get a specific output
Output result;
TF_CHECK_OK(session.Run(feed, &result));
```

### Close

```c++
// Close the session
TF_CHECK_OK(session.Close());
```

---

## Example Programs

### Example 1: Basic Arithmetic

```c++
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

using namespace tensorflow;
using namespace tensorflow::ops;

int main() {
    Scope root = Scope::NewRootScope();

    // Create constants
    auto a = Const(root, { {1.0f, 2.0f}, {3.0f, 4.0f} });
    auto b = Const(root, { {5.0f, 6.0f}, {7.0f, 8.0f} });

    // Arithmetic operations
    auto sum = Add(root, a, b);
    auto product = Mul(root, a, b);
    auto matmul = MatMul(root, a, b);

    // Execute
    ClientSession session(root);
    std::vector<Tensor> outputs;
    TF_CHECK_OK(session.Run({}, {sum, product, matmul}, &outputs));

    // Print results
    std::cout << "Sum:\n" << outputs[0].matrix<float>() << "\n";
    std::cout << "Product:\n" << outputs[1].matrix<float>() << "\n";
    std::cout << "MatMul:\n" << outputs[2].matrix<float>() << "\n";

    return 0;
}
```

### Example 2: Feed Dict with Placeholder

```c++
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"

using namespace tensorflow;
using namespace tensorflow::ops;

int main() {
    Scope root = Scope::NewRootScope();

    // Create placeholders
    auto x = Placeholder(root, DT_FLOAT);
    auto y = Placeholder(root, DT_FLOAT);

    // Build computation
    auto z = MatMul(root, x, y);

    // Prepare input data
    Tensor x_val(DT_FLOAT, TensorShape({2, 3}));
    x_val.matrix<float>() << 1, 2, 3, 4, 5, 6;

    Tensor y_val(DT_FLOAT, TensorShape({3, 2}));
    y_val.matrix<float>() << 7, 8, 9, 10, 11, 12;

    // Run with feeds
    ClientSession session(root);
    std::vector<Tensor> outputs;
    TF_CHECK_OK(session.Run({{x, x_val}, {y, y_val}}, {z}, &outputs));

    std::cout << "Result:\n" << outputs[0].matrix<float>() << "\n";
    return 0;
}
```

### Example 3: Linear Regression

```c++
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"

using namespace tensorflow;
using namespace tensorflow::ops;

int main() {
    Scope root = Scope::NewRootScope();

    // Model parameters
    auto W = Variable(root, {1, 1}, DT_FLOAT);
    auto b = Variable(root, {1}, DT_FLOAT);

    // Initialize
    auto assign_w = Assign(root, W, Const(root, 0.0f));
    auto assign_b = Assign(root, b, Const(root, 0.0f));

    // Input
    auto x = Placeholder(root, DT_FLOAT);
    auto y_true = Placeholder(root, DT_FLOAT);

    // Linear model: y = W * x + b
    auto y_pred = Add(root, MatMul(root, x, W), b);

    // Loss: mean squared error
    auto error = SquaredDifference(root, y_pred, y_true);
    auto loss = Mean(root, error, Const(root, 0));

    // Training: gradient descent
    float learning_rate = 0.01f;
    auto grad_w = Const(root, learning_rate);
    auto grad_b = Const(root, learning_rate);

    // Compute gradients manually
    // For simplicity, we use ApplyGradientDescent
    // In practice, use the gradients API
    auto train_w = ApplyGradientDescent(root, W, grad_w,
        Const(root, {1.0f}));  // Placeholder for actual gradient
    auto train_b = ApplyGradientDescent(root, b, grad_b,
        Const(root, {1.0f}));

    ClientSession session(root);
    std::vector<Tensor> outputs;

    // Initialize variables
    TF_CHECK_OK(session.Run({}, {assign_w, assign_b}, &outputs));

    // Training data
    Tensor x_data(DT_FLOAT, TensorShape({4, 1}));
    x_data.matrix<float>() << 1, 2, 3, 4;

    Tensor y_data(DT_FLOAT, TensorShape({4, 1}));
    y_data.matrix<float>() << 3, 5, 7, 9;  // y = 2x + 1

    // Training loop
    for (int i = 0; i < 100; i++) {
        TF_CHECK_OK(session.Run(
            {{x, x_data}, {y_true, y_data}},
            {loss, train_w, train_b},
            &outputs));
    }

    // Get final parameters
    TF_CHECK_OK(session.Run({}, {W, b}, &outputs));
    std::cout << "W: " << outputs[0].scalar<float>() << "\n";
    std::cout << "b: " << outputs[1].scalar<float>() << "\n";

    return 0;
}
```

### Example 4: Neural Network (Simple MLP)

```c++
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"

using namespace tensorflow;
using namespace tensorflow::ops;

// Helper: dense layer with ReLU
Output DenseLayer(const Scope& scope, Input input, int input_size,
                  int output_size, const string& name) {
    auto w = Variable(scope.WithOpName(name + "/weights"),
        {input_size, output_size}, DT_FLOAT);
    auto b = Variable(scope.WithOpName(name + "/bias"),
        {output_size}, DT_FLOAT);

    // Random initialization
    auto init_w = Assign(scope, w,
        RandomStandardNormal(scope.WithOpName("init_w"),
            {input_size, output_size}, DT_FLOAT));
    auto init_b = Assign(scope, b,
        ZerosLike(scope, Const(scope, {0.0f})));

    auto linear = Add(scope, MatMul(scope, input, w), b);
    return Relu(scope, linear);
}

int main() {
    Scope root = Scope::NewRootScope();

    // Input placeholder
    auto x = Placeholder(root, DT_FLOAT);
    auto labels = Placeholder(root, DT_FLOAT);

    // Build MLP
    auto hidden = DenseLayer(root, x, 784, 256, "hidden1");
    auto output = DenseLayer(root, hidden, 256, 10, "output");
    auto logits = Softmax(root, output);

    // Loss
    auto loss = ReduceMean(root,
        Neg(root, ReduceSum(root,
            Mul(root, labels, Log(root, logits)),
            Const(root, {1}))),
        Const(root, {0}));

    ClientSession session(root);

    // Create dummy input
    Tensor input_tensor(DT_FLOAT, TensorShape({32, 784}));
    input_tensor.flat<float>().setRandom();

    Tensor label_tensor(DT_FLOAT, TensorShape({32, 10}));
    label_tensor.flat<float>().setZero();

    std::vector<Tensor> outputs;
    TF_CHECK_OK(session.Run(
        {{x, input_tensor}, {labels, label_tensor}},
        {logits, loss},
        &outputs));

    std::cout << "Loss: " << outputs[1].scalar<float>() << "\n";
    return 0;
}
```

### Example 5: Save and Export Graph

```c++
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"

using namespace tensorflow;
using namespace tensorflow::ops;

int main() {
    Scope root = Scope::NewRootScope();

    auto x = Placeholder(root, DT_FLOAT, Placeholder::Shape({-1, 10}));
    auto w = Const(root, { /* weights */ }, TensorShape({10, 5}));
    auto b = Const(root, { /* bias */ }, TensorShape({5}));
    auto y = Add(root.WithOpName("output"), MatMul(root, x, w), b);

    // Export to GraphDef
    GraphDef graph_def;
    TF_CHECK_OK(root.ToGraphDef(&graph_def));

    // Save to file
    string serialized;
    graph_def.SerializeToString(&serialized);
    // Write serialized to file...

    return 0;
}
```

---

## Summary

The TensorFlow C++ API provides a comprehensive interface for building and
executing computation graphs:

1. **Scope**: The central context for graph construction with hierarchical
   naming, device placement, and error handling.
2. **Operations**: Complete set of operations in `tensorflow::ops` namespace,
  covering arithmetic, array manipulation, neural network layers, math
   functions, control flow, IO, and random number generation.
3. **Input/Output**: Flexible type system supporting tensors, scalars,
   initializer lists, and operation chaining.
4. **ClientSession**: Graph execution with feed dict support for runtime
   tensor values.
5. **Graph Export**: Convert built graphs to GraphDef for saving and
   deployment.
