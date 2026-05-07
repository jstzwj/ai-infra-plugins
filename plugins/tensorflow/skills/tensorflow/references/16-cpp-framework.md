# TensorFlow C++ Framework: Tensors, Types, Shapes, and Scope

This reference covers the foundational C++ building blocks of TensorFlow's core
framework. These classes and types are used throughout the runtime, kernel
implementations, graph construction APIs, and the high-level C++ API.

---

## Table of Contents

1. [Tensor Class](#tensor-class)
2. [TensorBuffer](#tensorbuffer)
3. [DataType Enum](#datatype-enum)
4. [TensorShape](#tensorshape)
5. [PartialTensorShape](#partialtensorshape)
6. [TensorShapeProto](#tensorshapeproto)
7. [Scope](#scope)
8. [Output and Input](#output-and-input)
9. [ClientSession](#clientsession)
10. [Status](#status)
11. [Allocator](#allocator)
12. [StringPiece](#stringpiece)
13. [RefCount](#refcount)

---

## Tensor Class

**Header:** `tensorflow/core/framework/tensor.h`
**Namespace:** `tensorflow`

The `Tensor` class is the central data structure in TensorFlow. It represents an
n-dimensional array of values with a specified data type and shape. Tensors are
immutable once created (their shape and dtype cannot change), though the
underlying buffer data can be modified via the Eigen tensor accessors.

### Construction

```cpp
// Default constructor: creates a 1-dimensional, 0-element float tensor.
// Shape is {0}, NumElements() == 0, IsInitialized() is true.
Tensor();

// Creates a Tensor with the given type and shape using CPUAllocator.
Tensor(DataType type, const TensorShape& shape);

// Creates a Tensor with a specific allocator.
Tensor(Allocator* a, DataType type, const TensorShape& shape);

// Creates a Tensor with allocator and allocation attributes.
Tensor(Allocator* a, DataType type, const TensorShape& shape,
       const AllocationAttributes& allocation_attr);

// Creates a Tensor from an existing buffer (acquires a ref).
Tensor(DataType type, const TensorShape& shape, TensorBuffer* buf);

// Creates a Tensor taking ownership of a RefCountPtr<TensorBuffer>.
Tensor(DataType type, TensorShape shape, core::RefCountPtr<TensorBuffer> buf);

// Creates an empty Tensor of the given data type (shape {0}).
explicit Tensor(DataType type);

// Scalar constructors (host memory).
explicit Tensor(float scalar_value);
explicit Tensor(double scalar_value);
explicit Tensor(int32_t scalar_value);
explicit Tensor(uint32_t scalar_value);
explicit Tensor(uint16_t scalar_value);
explicit Tensor(uint8_t scalar_value);
explicit Tensor(int16_t scalar_value);
explicit Tensor(int8_t scalar_value);
explicit Tensor(tstring scalar_value);
explicit Tensor(complex64 scalar_value);
explicit Tensor(complex128 scalar_value);
explicit Tensor(int64_t scalar_value);
explicit Tensor(uint64_t scalar_value);
explicit Tensor(bool scalar_value);
explicit Tensor(qint8 scalar_value);
explicit Tensor(quint8 scalar_value);
explicit Tensor(qint16 scalar_value);
explicit Tensor(quint16 scalar_value);
explicit Tensor(qint32 scalar_value);
explicit Tensor(bfloat16 scalar_value);
explicit Tensor(Eigen::half scalar_value);
explicit Tensor(ResourceHandle scalar_value);
explicit Tensor(const char* scalar_value);  // Converts to tstring

// Copy constructor (shares underlying storage).
Tensor(const Tensor& other);

// Move constructor.
Tensor(Tensor&& other);

// Factory method that validates DataType before construction.
static absl::Status BuildTensor(DataType type, const TensorShape& shape,
                                Tensor* out_tensor);
```

### Accessors

```cpp
// Data type of the tensor.
DataType dtype() const;  // Returns shape_.data_type()

// Shape of the tensor.
const TensorShape& shape() const;

// Convenience shape accessors.
int dims() const;              // Number of dimensions
int64_t dim_size(int d) const; // Size of dimension d
int64_t NumElements() const;   // Total number of elements

// Memory usage estimation.
size_t TotalBytes() const;     // Estimated bytes for this tensor
size_t GetBufferSize() const;  // Size of the underlying TensorBuffer
size_t AllocatedBytes() const; // Size of allocated memory

// Alignment check.
bool IsAligned() const;

// Initialization check (zero-element tensors are always initialized).
bool IsInitialized() const;

// Buffer sharing check.
bool SharesBufferWith(const Tensor& b) const;
bool IsSameSize(const Tensor& b) const;
```

### Data Access via Eigen Tensors

The `Tensor` class provides typed access to its data through Eigen tensor
mapping methods. These return Eigen `TensorMap` objects that can be used like
regular Eigen tensors for computation.

```cpp
// 1D access (flattened).
template <typename T>
typename TTypes<T>::Vec vec();

// 2D access (matrix).
template <typename T>
typename TTypes<T>::Matrix matrix();

// N-dimensional access (must match tensor rank).
template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor tensor();

// Scalar access.
template <typename T>
typename TTypes<T>::Scalar scalar();

// 1D flat access (any tensor can be flattened).
template <typename T>
typename TTypes<T>::Flat flat();

// Unaligned flat access (no alignment requirement).
template <typename T>
typename TTypes<T>::UnalignedFlat unaligned_flat();

// Bit-cast tensor access (reinterpret bits as another type).
template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor bit_casted_tensor();

// Reinterpret last dimension (e.g., NCHW_VECT_C int8 to NCHW int32).
template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor reinterpret_last_dimension();

// Reshape access: returns data with a user-specified shape.
template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor shaped(gtl::ArraySlice<int64_t> new_sizes);

// Flat inner dimensions: collapse all but last NDIMS-1 dims.
template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor flat_inner_dims();

// Flat outer dimensions: collapse all but first NDIMS-1 dims.
template <typename T, size_t NDIMS>
typename TTypes<T, NDIMS>::Tensor flat_outer_dims();
```

### Example: Data Access

```cpp
Tensor t(DT_FLOAT, TensorShape({3, 5}));
auto matrix = t.matrix<float>();     // 3x5 Eigen::Tensor
matrix(0, 0) = 1.0f;

auto flat = t.flat<float>();         // 15-element flat view
flat(0) = 42.0f;

auto reshaped = t.shaped<float, 3>({1, 3, 5}); // 1x3x5 view
```

### Slicing and Sub-slicing

```cpp
// Slice along the first dimension.
// returned[i, ...] == this[dim0_start + i, ...]
// REQUIRES: dims() >= 1
// REQUIRES: 0 <= dim0_start <= dim0_limit <= dim_size(0)
Tensor Slice(int64_t dim0_start, int64_t dim0_limit) const;

// Sub-slice: returns N-1 dimensional tensor at given index.
// REQUIRES: dims() >= 1
// REQUIRES: 0 <= index < dim_size(0)
Tensor SubSlice(int64_t index) const;
```

### Serialization

```cpp
// Parse from protobuf.
bool FromProto(const TensorProto& other);
bool FromProto(Allocator* a, const TensorProto& other);

// Serialize to protobuf.
// AsProtoField fills repeated field for proto.dtype()
void AsProtoField(TensorProto* proto) const;
// AsProtoTensorContent encodes content in proto.tensor_content()
void AsProtoTensorContent(TensorProto* proto) const;
```

### Assignment

```cpp
// Copy assignment (shares underlying storage).
Tensor& operator=(const Tensor& other);

// Move assignment.
Tensor& operator=(Tensor&& other);

// Copy with reshape (shares storage, returns false if element count mismatch).
bool CopyFrom(const Tensor& other, const TensorShape& shape);
```

---

## TensorBuffer

**Header:** `tensorflow/core/framework/tensor.h`

`TensorBuffer` is the interface to access the raw ref-counted data buffer
underlying a `Tensor`. It inherits from `core::RefCounted`.

```cpp
class TensorBuffer : public core::RefCounted {
 public:
  explicit TensorBuffer(void* data_ptr);

  // Points to a memory region of size() bytes.
  void* data() const;  // Non-virtual for performance (inlined).

  // Size of the buffer in bytes.
  virtual size_t size() const = 0;

  // Returns the root buffer (this or parent if sub-buffer).
  virtual TensorBuffer* root_buffer() = 0;

  // Fills allocation description proto.
  virtual void FillAllocationDescription(AllocationDescription* proto) const = 0;

  // Gets allocated bytes (returns false if not supported).
  virtual bool GetAllocatedBytes(size_t* out_bytes) const;

  // Reinterpret buffer as array of T.
  template <typename T>
  T* base() const;

  // Whether this buffer owns the underlying memory.
  virtual bool OwnsMemory() const;

  // The type of the underlying memory.
  virtual AllocatorMemoryType GetMemoryType() const;

 private:
  void* const data_;
};
```

The `data()` method is intentionally non-virtual because it may be called
multiple times during tensor data access. This design decision prioritizes
performance over extensibility at this level.

### Reference Counting

`TensorBuffer` extends `core::RefCounted`, which provides:

```cpp
void Ref() const;     // Increment reference count
void Unref() const;   // Decrement; delete if zero
bool HasOneRef() const; // True if count == 1
```

Multiple `Tensor` objects can share the same `TensorBuffer` (e.g., through
slicing or copy construction). The buffer is automatically freed when the last
reference is dropped.

---

## DataType Enum

**Header:** `tensorflow/core/framework/types.pb.h` (protobuf-generated)
**Referenced from:** `tensorflow/core/framework/types.h`

The `DataType` enum specifies the type of elements stored in a `Tensor`. It is
defined in the `tensorflow::DataType` namespace and generated from the
`types.proto` protocol buffer definition.

### Core Data Types

| Enum Value           | C++ Type          | Description                        |
|---------------------|-------------------|------------------------------------|
| `DT_INVALID`        | (none)            | Invalid/undefined type             |
| `DT_FLOAT`          | `float`           | 32-bit floating point              |
| `DT_DOUBLE`         | `double`          | 64-bit floating point              |
| `DT_INT32`          | `int32_t`         | 32-bit signed integer              |
| `DT_UINT8`          | `uint8_t`         | 8-bit unsigned integer             |
| `DT_INT16`          | `int16_t`         | 16-bit signed integer              |
| `DT_INT8`           | `int8_t`          | 8-bit signed integer               |
| `DT_STRING`         | `tstring`         | Variable-length string             |
| `DT_COMPLEX64`      | `complex64`       | 64-bit complex (two 32-bit floats) |
| `DT_INT64`          | `int64_t`         | 64-bit signed integer              |
| `DT_BOOL`           | `bool`            | Boolean                            |
| `DT_QINT8`          | `qint8`           | Quantized 8-bit signed integer     |
| `DT_QUINT8`         | `quint8`          | Quantized 8-bit unsigned integer   |
| `DT_QINT32`         | `qint32`          | Quantized 32-bit signed integer    |
| `DT_BFLOAT16`       | `bfloat16`        | Brain floating point 16-bit        |
| `DT_QINT16`         | `qint16`          | Quantized 16-bit signed integer    |
| `DT_QUINT16`        | `quint16`         | Quantized 16-bit unsigned integer  |
| `DT_UINT16`         | `uint16_t`        | 16-bit unsigned integer            |
| `DT_COMPLEX128`     | `complex128`      | 128-bit complex (two 64-bit floats)|
| `DT_HALF`           | `Eigen::half`     | 16-bit floating point              |
| `DT_UINT32`         | `uint32_t`        | 32-bit unsigned integer            |
| `DT_UINT64`         | `uint64_t`        | 64-bit unsigned integer            |
| `DT_RESOURCE`       | `ResourceHandle`  | Handle to a resource               |
| `DT_VARIANT`        | `Variant`         | Arbitrary C++ data type            |
| `DT_FLOAT_REF`      | (ref to float)    | Reference to DT_FLOAT              |
| `DT_DOUBLE_REF`     | (ref to double)   | Reference to DT_DOUBLE             |
| ..._REF variants    |                   | Reference variants of above types  |

### Utility Functions

```cpp
// Convert DataType to string representation.
std::string DataTypeString(DataType dtype);

// Convert a slice of DataTypes to string.
std::string DataTypeSliceString(DataTypeSlice dtypes);
std::string DataTypeVectorString(const DataTypeVector& dtypes);
```

### Type Collections

```cpp
// Vector of DataTypes.
typedef absl::InlinedVector<DataType, 4UL> DataTypeVector;
typedef absl::Span<const DataType> DataTypeSlice;

// DataTypeSet represents a set of DataType values as a bitmask.
// Cannot represent DT_*_REF values.
class DataTypeSet {
  bool Contains(DataType dt) const;
  void Insert(DataType dt);
  void Remove(DataType dt);
  // ... iteration support
};
```

### MemoryType

```cpp
enum MemoryType {
  DEVICE_MEMORY = 0,  // GPU memory for GPU devices, CPU for CPU
  HOST_MEMORY = 1,    // Always CPU memory
};

typedef absl::InlinedVector<MemoryType, 4UL> MemoryTypeVector;
typedef absl::Span<const MemoryType> MemoryTypeSlice;
```

### Device Type Constants

```cpp
extern const char* const DEVICE_DEFAULT;     // "DEFAULT"
extern const char* const DEVICE_CPU;         // "CPU"
extern const char* const DEVICE_GPU;         // "GPU"
extern const char* const DEVICE_TPU;         // "TPU"
extern const char* const DEVICE_TPU_SYSTEM;  // "TPU_SYSTEM"
```

### Template Traits

TensorFlow provides compile-time mapping from C++ types to DataType enums:

```cpp
// Example: DataTypeToEnum<float>::value == DT_FLOAT
template <typename T>
struct DataTypeToEnum {
  static constexpr DataType value = ...;
};

// Reverse: EnumToDataType<DT_FLOAT>::Type == float
template <DataType dt>
struct EnumToDataType {
  typedef ... Type;
};

// Check if a type is valid.
template <typename T>
struct IsValidDataType;

// Size of a DataType in bytes.
int DataTypeSize(DataType dt);
```

---

## TensorShape

**Header:** `tensorflow/core/framework/tensor_shape.h`
**Namespace:** `tensorflow`

`TensorShape` represents the shape of a tensor: the number of dimensions and the
size of each dimension. It is a fully-defined shape where every dimension size
is known. `TensorShape` inherits from `TensorShapeRep`.

### Internal Representation

`TensorShapeRep` uses a compact 16-byte representation with three possible
formats:

- **Rep16**: Up to 6 dimensions, each dimension < 2^16 - 1 (stored as `uint16_t[6]`)
- **Rep32**: Up to 3 dimensions, each dimension < 2^32 - 1 (stored as `uint32_t[3]`)
- **Rep64**: Arbitrary dimensionality with 64-bit dimensions (stored as
  `absl::InlinedVector<int64_t, 4>*`)

This optimization ensures that common tensor shapes (1D, 2D, 3D with small
dimensions) are stored inline without heap allocation.

### Maximum Dimensions

```cpp
static constexpr int MaxDimensions() { return 254; }
// 255 = kUnknownRank, used for PartialTensorShape
```

### Construction

```cpp
// Default: scalar shape (0 dimensions).
TensorShape();

// From dimension sizes.
explicit TensorShape(gtl::ArraySlice<int64_t> dim_sizes);
// Example: TensorShape({3, 5}) for a 3x5 tensor.

// From a TensorShapeProto protocol buffer.
explicit TensorShape(const TensorShapeProto& proto);

// Move and copy constructors.
TensorShape(const TensorShapeRep& b);
TensorShape(TensorShapeRep&& b);
```

### Dimension Access

```cpp
// Number of dimensions (rank).
int dims() const;

// Size of a specific dimension (0-indexed).
int64_t dim_size(int d) const;

// Total number of elements across all dimensions.
// Returns -1 for PartialTensorShape if not fully defined.
int64_t num_elements() const;

// All dimension sizes as a vector.
gtl::InlinedVector<int64_t, 4> dim_sizes() const;

// Check if fully defined (all dimensions known).
bool IsFullyDefined() const;  // Always true for TensorShape

// Check if two shapes are the same.
bool IsSameSize(const TensorShape& b) const;
```

### Modification

```cpp
// Add a dimension at the end.
void AddDim(int64_t size);

// Append all dimensions from another shape.
void AppendShape(const TensorShape& shape);

// Insert a dimension at position d.
void InsertDim(int d, int64_t size);

// Remove dimension d.
void RemoveDim(int d);

// Set dimension d to size.
void set_dim(int d, int64_t size);

// Clear to scalar shape.
void Clear();
```

### Iteration

```cpp
// Iterator support for dimension sizes.
typedef TensorShapeIter<TensorShape> Iterator;
Iterator begin() const;
Iterator end() const;
```

### Debugging

```cpp
// Human-readable string representation.
std::string DebugString() const;
static std::string DebugString(const TensorShapeProto& proto);
```

### Serialization

```cpp
// Convert to/from protocol buffer.
void AsProto(TensorShapeProto* proto) const;
// For PartialTensorShape, unknown dims get size -1.
```

### Example

```cpp
TensorShape shape({3, 5, 7});
CHECK_EQ(shape.dims(), 3);
CHECK_EQ(shape.dim_size(0), 3);
CHECK_EQ(shape.dim_size(1), 5);
CHECK_EQ(shape.dim_size(2), 7);
CHECK_EQ(shape.num_elements(), 3 * 5 * 7);  // 105

shape.AddDim(2);
CHECK_EQ(shape.dims(), 4);
CHECK_EQ(shape.num_elements(), 210);

TensorShapeProto proto;
shape.AsProto(&proto);
```

---

## PartialTensorShape

**Header:** `tensorflow/core/framework/tensor_shape.h`

`PartialTensorShape` extends `TensorShapeRep` to support shapes where some
dimension sizes may be unknown (represented as -1). This is used during graph
construction when shapes are not yet fully determined.

### Construction

```cpp
// From dimension sizes (use -1 for unknown dimensions).
explicit PartialTensorShape(gtl::ArraySlice<int64_t> dim_sizes);
// Example: PartialTensorShape({-1, 5}) -- batch size unknown.

// From a TensorShapeProto.
explicit PartialTensorShape(const TensorShapeProto& proto);

// Unknown rank constructor.
static PartialTensorShape Unknown();  // Rank unknown entirely
```

### Key Methods

```cpp
// Check if every dimension is known.
bool IsFullyDefined() const;

// Check if the rank is unknown.
bool unknown_rank() const;

// Convert to a fully-defined TensorShape.
// Returns error if any dimension is unknown.
absl::Status AsTensorShape(TensorShape* shape) const;

// Check compatibility with another PartialTensorShape.
bool IsCompatibleWith(const PartialTensorShape& b) const;

// Merge with another PartialTensorShape.
// Returns error if incompatible.
absl::Status MergeWith(const PartialTensorShape& b,
                       PartialTensorShape* result) const;
```

### Example

```cpp
PartialTensorShape partial({-1, 5, 7});
CHECK_EQ(partial.dims(), 3);
CHECK(!partial.IsFullyDefined());
CHECK_EQ(partial.dim_size(0), -1);  // Unknown

PartialTensorShape other({3, -1, 7});
PartialTensorShape merged;
Status s = partial.MergeWith(other, &merged);
// merged is {3, 5, 7}
```

---

## TensorShapeProto

**Header:** `tensorflow/core/framework/tensor_shape.proto`

The protocol buffer representation of a tensor shape.

```protobuf
message TensorShapeProto {
  message Dim {
    int64 size = 1;
    string name = 2;  // Optional human-readable name
  }
  repeated Dim dim = 2;
  bool unknown_rank = 3;  // If true, the rank is unknown
}
```

Usage:

```cpp
// From TensorShape to proto.
TensorShape shape({3, 5});
TensorShapeProto proto;
shape.AsProto(&proto);

// From proto to TensorShape.
TensorShape from_proto(proto);

// From proto to PartialTensorShape.
PartialTensorShape partial(proto);
```

---

## Scope

**Header:** `tensorflow/cc/framework/scope.h`
**Namespace:** `tensorflow`

The `Scope` class is the primary entry point for TensorFlow's C++ API. A
`Scope` object represents a set of related TensorFlow ops that share common
properties such as a name prefix. Every op constructor takes a `Scope` as its
first argument.

### Key Properties

- **Thread Safety**: `Scope` is NOT thread-safe. Concurrent op construction on
  the same `Scope` is not allowed.
- **Lifetime**: A root scope creates shared resources (Graph, Status) that are
  inherited by all child scopes.
- **Ownership**: The root scope owns the `Graph` object to which all operations
  are added.

### Creating Root Scopes

```cpp
// Create a new root scope with a fresh Graph.
static Scope NewRootScope();

// Create a scope with disabled shape inference (for testing).
static Scope DisabledShapeInferenceScope();
```

### Sub-scopes and Naming

```cpp
// Create a child scope with name prefix "child_scope_name".
// Ops will be named "parent_prefix/child_scope_name/OpType".
Scope NewSubScope(const std::string& child_scope_name) const;

// Set the op name (suffix) for ops created in this scope.
template <typename... Ty>
Scope WithOpName(Ty... fragments) const;

// Get a unique name for an op (uses default_name if none specified).
std::string GetUniqueNameForOp(const std::string& default_name) const;
```

### Naming Example

```cpp
Scope root = Scope::NewRootScope();
Scope linear = root.NewSubScope("linear");

// W will be named "linear/W"
auto W = Variable(linear.WithOpName("W"), {2, 2}, DT_FLOAT);

// b will be named "linear/b_3"
int idx = 3;
auto b = Variable(linear.WithOpName("b_", idx), {2}, DT_FLOAT);

// auto-named: "linear/Const"
auto x = Const(linear, {...});

// auto-named: "linear/MatMul"
auto m = MatMul(linear, x, W);
```

### Device Placement

```cpp
// Set the device for ops in this scope.
Scope WithDevice(const std::string& device) const;

// Set the assigned device (runtime placement).
Scope WithAssignedDevice(const std::string& assigned_device) const;

// Set XLA cluster attribute.
Scope WithXlaCluster(const std::string& xla_cluster) const;

// Co-locate ops with a given operation.
Scope ColocateWith(const Operation& op) const;
Scope ColocateWith(const Output& out) const;  // Convenience

// Clear colocation constraints.
Scope ClearColocation() const;
```

### Control Dependencies

```cpp
// Add control dependencies to ops in this scope.
Scope WithControlDependencies(
    absl::Span<const Operation> control_deps) const;
Scope WithControlDependencies(const Output& control_dep) const;

// Remove all control dependencies.
Scope WithNoControlDependencies() const;
```

### Error Handling

```cpp
// Exit (LOG(FATAL)) on error instead of setting status.
Scope ExitOnError() const;

// Check if scope is in a valid state.
bool ok() const;

// Get the current status.
absl::Status status() const;

// Update the status (shared between all children).
void UpdateStatus(const absl::Status& s) const;
```

### Kernel Label

```cpp
// Set the _kernel attribute for ops in this scope.
Scope WithKernelLabel(const std::string& kernel_label) const;
```

### Graph Access

```cpp
// Access the underlying Graph object.
Graph* graph() const;
std::shared_ptr<Graph> graph_as_shared_ptr() const;

// Convert the graph to a GraphDef proto.
absl::Status ToGraphDef(GraphDef* gdef,
                        bool include_debug_info = false) const;

// Convert to a Graph object.
absl::Status ToGraph(Graph* g,
                     GraphConstructorOptions opts = {}) const;

// Get control dependencies.
const std::vector<Operation>& control_deps() const;
```

### Shape Inference

```cpp
// Run shape inference on a node.
absl::Status DoShapeInference(Node* node) const;
```

### CompositeOpScopes

```cpp
// Helper for building composite operations.
struct CompositeOpScopes {
  Scope child;  // For creating local ops
  Scope last;   // For the final op
};

CompositeOpScopes GetCompositeOpScopes(
    const std::string& composite_op_name) const;
```

### Complete Example

```cpp
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

using namespace tensorflow;
using namespace tensorflow::ops;

Scope root = Scope::NewRootScope();

// Build a simple computation graph.
auto a = Const(root, { {1.f, 2.f}, {3.f, 4.f} });
auto b = Const(root, { {5.f, 6.f}, {7.f, 8.f} });
auto c = MatMul(root, a, b);

// Convert to GraphDef.
GraphDef graph_def;
Status s = root.ToGraphDef(&graph_def);
if (!s.ok()) {
  LOG(ERROR) << s.ToString();
  return;
}
```

---

## Output and Input

**Headers:**
- `tensorflow/cc/framework/ops.h` (Output, Input)
- `tensorflow/core/framework/tensor.h` (Tensor)

### Output

`Output` represents a specific output of a TensorFlow operation (a tensor
produced by a node at a given output index).

```cpp
class Output {
 public:
  Output() = default;
  Output(Node* n, int32_t index);
  Output(const Operation& op, int32_t index);
  Output(const Tensor& tensor);  // Creates a Const op

  // Accessors.
  Node* node() const;
  int32_t index() const;
  DataType type() const;
  Operation op() const;

  // Get the tensor shape (may require shape inference).
  TensorShape shape() const;
};
```

### Input

`Input` is a type that can be implicitly constructed from various sources,
serving as the argument type for op inputs.

```cpp
class Input {
 public:
  Input() = default;
  Input(const Output& output);
  Input(const Tensor& tensor);      // Creates a Const op
  Input(const std::initializer_list<float>& v);

  // Accessors.
  Node* node() const;
  int32_t index() const;
  DataType type() const;
};
```

### InputList and OutputList

```cpp
// For ops that take a list of tensors as input.
class InputList {
 public:
  InputList(const std::initializer_list<Input>& inputs);
  InputList(const std::vector<Output>& outputs);
  // ...
};

class OutputList {
 public:
  OutputList(const std::vector<Output>& outputs);
  // ...
};
```

---

## ClientSession

**Header:** `tensorflow/cc/client/client_session.h`

`ClientSession` is the C++ equivalent of Python's `tf.Session`. It drives the
execution of a graph built with the C++ API.

### Construction

```cpp
// Create with default options.
explicit ClientSession(const Scope& scope);

// Create with custom options.
ClientSession(const Scope& scope, const std::string& target);
ClientSession(const Scope& scope, const SessionOptions& options);
```

### Running Operations

```cpp
// Run operations, fetching outputs.
absl::Status Run(const std::vector<Output>& fetch_outputs,
                 std::vector<Tensor>* outputs) const;

// Run with feed dictionaries.
absl::Status Run(const FeedType& feeds,
                 const std::vector<Output>& fetch_outputs,
                 std::vector<Tensor>* outputs) const;

// Run with feeds and targets (operations to run but not fetch).
absl::Status Run(const FeedType& feeds,
                 const std::vector<Output>& fetch_outputs,
                 const std::vector<Operation>& run_outputs,
                 std::vector<Tensor>* outputs) const;

// Run with RunOptions and get RunMetadata.
absl::Status Run(const RunOptions& run_options,
                 const FeedType& feeds,
                 const std::vector<Output>& fetch_outputs,
                 const std::vector<Operation>& run_outputs,
                 std::vector<Tensor>* outputs,
                 RunMetadata* run_metadata) const;
```

### FeedType

```cpp
// FeedType is a mapping from Output to Tensor.
// Can be constructed from:
//   { {output1, tensor1}, {output2, tensor2} }
typedef std::unordered_map<Output, Tensor, OutputHash> FeedType;
```

### Example

```cpp
Scope root = Scope::NewRootScope();
auto a = Const(root, { {1.f, 2.f} });
auto b = Const(root, { {3.f}, {4.f} });
auto c = MatMul(root, a, b);

ClientSession session(root);
std::vector<Tensor> outputs;
TF_CHECK_OK(session.Run({c}, &outputs));

// outputs[0] is a 1x1 matrix: [[11.0]]
LOG(INFO) << outputs[0].matrix<float>();
```

### Callable API

```cpp
// For more efficient repeated execution.
typedef int64_t CallableHandle;

absl::Status MakeCallable(const CallableOptions& options,
                          CallableHandle* handle);
absl::Status RunCallable(CallableHandle handle,
                         const std::vector<Tensor>& feed_tensors,
                         std::vector<Tensor>* fetch_tensors,
                         RunMetadata* run_metadata);
absl::Status ReleaseCallable(CallableHandle handle);
```

---

## Status

**Header:** `tensorflow/core/lib/core/status.h` / `absl/status/status.h`

TensorFlow uses `absl::Status` for error reporting throughout the C++ API.
Historically it used a custom `tensorflow::Status` class, but this has been
migrated to Abseil's `absl::Status`.

### Usage

```cpp
// Create a success status.
absl::Status ok = absl::OkStatus();

// Create an error status.
absl::Status error = absl::InvalidArgumentError("Bad argument");
absl::Status error = absl::NotFoundError("Not found");
absl::Status error = absl::InternalError("Internal failure");

// Check status.
if (!status.ok()) {
  LOG(ERROR) << status.ToString();
}

// Common pattern in kernel code.
TF_RETURN_IF_ERROR(some_operation());
OP_REQUIRES_OK(context, another_operation());
```

### Common Error Factories

```cpp
absl::Status absl::OkStatus();
absl::Status absl::CancelledError(...);
absl::Status absl::UnknownError(...);
absl::Status absl::InvalidArgumentError(...);
absl::Status absl::DeadlineExceededError(...);
absl::Status absl::NotFoundError(...);
absl::Status absl::AlreadyExistsError(...);
absl::Status absl::PermissionDeniedError(...);
absl::Status absl::ResourceExhaustedError(...);
absl::Status absl::FailedPreconditionError(...);
absl::Status absl::AbortedError(...);
absl::Status absl::OutOfRangeError(...);
absl::Status absl::UnimplementedError(...);
absl::Status absl::InternalError(...);
absl::Status absl::UnavailableError(...);
absl::Status absl::DataLossError(...);
absl::Status absl::UnauthenticatedError(...);
```

### StatusOr

```cpp
// Abseil's StatusOr for returning either a value or an error.
absl::StatusOr<T> result = some_function();
if (!result.ok()) {
  return result.status();
}
T value = std::move(result.value());
```

### TensorFlow Error Macros

```cpp
// OP_REQUIRES: Check a condition in a kernel, set status on failure.
OP_REQUIRES(context, condition, errors::InvalidArgument("msg"));

// OP_REQUIRES_OK: Check that a status is OK.
OP_REQUIRES_OK(context, status);

// TF_RETURN_IF_ERROR: Return status if not OK.
TF_RETURN_IF_ERROR(status);

// TF_CHECK_OK: Crash if status is not OK.
TF_CHECK_OK(status);
```

---

## Allocator

**Header:** `tensorflow/core/framework/allocator.h`
**Actual implementation:** `xla/tsl/framework/allocator.h` (via TSL layer)

TensorFlow abstracts memory allocation through the `Allocator` interface. This
allows tensors to be allocated on different devices (CPU, GPU, TPU) with
different memory types and allocation strategies.

### Allocator Interface

```cpp
class Allocator {
 public:
  // Allocate memory of size bytes.
  virtual void* AllocateRaw(size_t alignment, size_t num_bytes) = 0;

  // Deallocate memory previously allocated.
  virtual void DeallocateRaw(void* ptr) = 0;

  // Allocate and return a typed buffer.
  template <typename T>
  T* Allocate(size_t num_elements);

  // Characteristics.
  virtual absl::string_view Name() = 0;
  virtual bool AllocatesOpaqueHandle() const;

  // Statistics.
  virtual bool TracksAllocationSizes() const;
  virtual size_t RequestedSize(const void* ptr) const;
  virtual size_t AllocatedSize(const void* ptr) const;
  virtual int64_t AllocationId(const void* ptr) const;
  virtual absl::optional<AllocatorStats> GetStats();
  virtual void ClearStats();
};
```

### AllocatorAttributes

```cpp
class AllocatorAttributes {
 public:
  // Set if allocation should be on host (CPU) memory.
  void set_on_host(bool value);
  bool on_host() const;

  // Set the NIC compatible flag.
  void set_nic_compatible(bool value);

  // Combination of flags.
  uint32_t value = 0;
};
```

### AllocationAttributes

```cpp
struct AllocationAttributes {
  // If non-zero, this is an allocation that may be revised later.
  int64_t allocation_id = 0;

  // Whether the allocation will be logged.
  bool allocation_will_be_logged = false;

  // Whether to retry allocation on failure.
  bool no_retry_on_failure = false;

  // If the allocation is for a tensor that may be in use by other ops.
  bool maybe_reallocated = false;
};
```

### AllocatorStats

```cpp
struct AllocatorStats {
  int64_t num_allocs;           // Number of allocations
  int64_t bytes_in_use;         // Bytes currently in use
  int64_t peak_bytes_in_use;    // Peak bytes in use
  int64_t largest_alloc_size;   // Size of largest allocation
  int64_t bytes_limit;          // Upper limit on bytes (0 = no limit)
  int64_t bytes_reserved;       // Bytes reserved (e.g., GPU memory pool)
  int64_t peak_bytes_reserved;
};
```

### CPU Allocator

```cpp
// Get the default CPU allocator.
Allocator* cpu_allocator();

// Get a CPU allocator for a specific NUMA node.
Allocator* cpu_allocator_base(int numa_node);

// Enable/disable CPU allocator statistics.
void EnableCPUAllocatorStats();
void DisableCPUAllocatorStats();
void EnableCPUAllocatorFullStats();
```

### SubAllocator

```cpp
class SubAllocator {
 public:
  virtual void* Alloc(size_t alignment, size_t num_bytes) = 0;
  virtual void Free(void* ptr, size_t num_bytes) = 0;
};
```

### AllocatorMemoryType

```cpp
enum class AllocatorMemoryType {
  kUnknown = 0,
  kDevice = 1,  // Device memory (GPU, etc.)
  kHost = 2,    // Host memory (CPU)
};
```

---

## StringPiece

**Header:** `tensorflow/core/lib/core/stringpiece.h`

`StringPiece` is a read-only string view type (similar to `absl::string_view`).
It provides a lightweight way to pass around string data without copying.

```cpp
class StringPiece {
 public:
  StringPiece();
  StringPiece(const char* str);
  StringPiece(const std::string& str);
  StringPiece(const char* data, size_t len);

  // Accessors.
  const char* data() const;
  size_t size() const;
  bool empty() const;

  // Iteration.
  const char* begin() const;
  const char* end() const;

  // Comparison.
  bool operator==(StringPiece x) const;
  bool operator!=(StringPiece x) const;
  bool operator<(StringPiece x) const;

  // Substring.
  StringPiece substr(size_t pos, size_t len = npos) const;

  // Find.
  size_t find(StringPiece s, size_t pos = 0) const;

  // Conversion.
  std::string ToString() const;
};
```

Note: Modern TensorFlow code uses `absl::string_view` directly in many places,
but `StringPiece` is still prevalent in the codebase.

---

## RefCount

**Header:** `tensorflow/core/lib/core/refcount.h`

TensorFlow uses reference counting for managing shared resources like tensor
buffers. The `RefCounted` base class and `RefCountPtr` smart pointer provide
this functionality.

### RefCounted

```cpp
class RefCounted {
 public:
  RefCounted();
  virtual ~RefCounted();

  void Ref() const;
  void Unref() const;
  bool HasOneRef() const;

 private:
  mutable std::atomic<int32_t> ref_;
};
```

### RefCountPtr

```cpp
template <typename T>
class RefCountPtr {
 public:
  RefCountPtr();
  explicit RefCountPtr(T* obj);  // Takes ownership (does not Ref)
  ~RefCountPtr();                // Calls Unref()

  // Move-only.
  RefCountPtr(RefCountPtr&& other);
  RefCountPtr& operator=(RefCountPtr&& other);

  // Access.
  T* get() const;
  T& operator*() const;
  T* operator->() const;
  explicit operator bool() const;

  // Release ownership.
  T* release();

  // Reset.
  void reset(T* obj = nullptr);
};
```

### Usage with TensorBuffer

```cpp
// RefCountPtr is used extensively with TensorBuffer.
core::RefCountPtr<TensorBuffer> buf = ...;
// When buf goes out of scope, the buffer is automatically freed
// (via Unref()) if no other references exist.
```

### WeakPtr

```cpp
// TensorFlow also provides a weak reference mechanism.
template <typename T>
class WeakPtr {
 public:
  WeakPtr();
  explicit WeakPtr(T* obj);
  absl::optional<T*> GetNewRef() const;
  bool HasRef() const;
};
```

---

## Common Patterns

### Creating and Using Tensors

```cpp
// Create a 2D float tensor.
Tensor t(DT_FLOAT, TensorShape({3, 4}));
auto matrix = t.matrix<float>();
for (int i = 0; i < 3; ++i) {
  for (int j = 0; j < 4; ++j) {
    matrix(i, j) = i * 4 + j;
  }
}
```

### Building a Graph with Scope

```cpp
Scope root = Scope::NewRootScope();

// With device specification.
Scope gpu_scope = root.WithDevice("/device:GPU:0");
auto gpu_matmul = MatMul(gpu_scope, input_a, input_b);

// With control dependencies.
Scope deps_scope = root.WithControlDependencies({some_op});
auto result = Add(deps_scope, x, y);
```

### Error Handling in Op Kernels

```cpp
void Compute(OpKernelContext* ctx) override {
  const Tensor& input = ctx->input(0);
  OP_REQUIRES(ctx, input.dims() == 2,
              errors::InvalidArgument("Input must be 2-D, got shape: ",
                                       input.shape().DebugString()));

  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

  // Use Eigen for computation.
  auto in = input.matrix<float>();
  auto out = output->matrix<float>();
  out = in * 2.0f;
}
```

---

## Type Compatibility Reference

### DataType to C++ Type Mapping

| DataType       | C++ Type        | Eigen Type              |
|---------------|-----------------|------------------------|
| DT_FLOAT      | `float`         | `Eigen::half` / float  |
| DT_DOUBLE     | `double`        | `double`               |
| DT_INT32      | `int32_t`       | `int32_t`              |
| DT_INT64      | `int64_t`       | `int64_t`              |
| DT_UINT8      | `uint8_t`       | `uint8_t`              |
| DT_UINT16     | `uint16_t`      | `uint16_t`             |
| DT_UINT32     | `uint32_t`      | `uint32_t`             |
| DT_UINT64     | `uint64_t`      | `uint64_t`             |
| DT_INT8       | `int8_t`        | `int8_t`               |
| DT_INT16      | `int16_t`       | `int16_t`              |
| DT_BOOL       | `bool`          | `bool`                 |
| DT_STRING     | `tstring`       | N/A                    |
| DT_HALF       | `Eigen::half`   | `Eigen::half`          |
| DT_BFLOAT16   | `bfloat16`      | `bfloat16`             |
| DT_COMPLEX64  | `complex64`     | `std::complex<float>`  |
| DT_COMPLEX128 | `complex128`    | `std::complex<double>` |
| DT_RESOURCE   | `ResourceHandle`| N/A                    |
| DT_VARIANT    | `Variant`       | N/A                    |

### TTypes Template Aliases

```cpp
// TTypes provides Eigen TensorMap type aliases:
template <typename T, int NDIMS = 1, typename IndexType = Eigen::Index>
struct TTypes {
  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned> Tensor;

  typedef Eigen::TensorMap<Eigen::Tensor<const T, NDIMS, ...>> ConstTensor;

  typedef ... Flat;        // 1D
  typedef ... Scalar;      // 0D
  typedef ... Vec;         // 1D (alias for Flat)
  typedef ... Matrix;      // 2D
  typedef ... UnalignedFlat;     // No alignment requirement
  typedef ... UnalignedVec;
  typedef ... UnalignedMatrix;
  typedef ... uint32;      // Reinterpret as uint32
  typedef ... uint16;
  typedef ... uint8;
};
```

---

## Thread Safety Summary

| Class              | Thread Safety                                  |
|--------------------|------------------------------------------------|
| `Tensor`           | Thread-safe for read access; write access must be synchronized |
| `TensorBuffer`     | Reference counting is atomic; data access must be synchronized |
| `TensorShape`      | Not thread-safe (copy if sharing)              |
| `Scope`            | NOT thread-safe                                |
| `ClientSession`    | Thread-safe for concurrent Run() calls         |
| `Allocator`        | Implementation-dependent (usually thread-safe) |
| `Status`           | Thread-safe (immutable after creation)         |

---

## Header Dependency Graph

```
tensor.h
  +-- tensor_shape.h
  +-- types.h / types.pb.h
  +-- allocator.h
  +-- refcount.h
  +-- stringpiece.h
  +-- tensor_types.h (TTypes)

scope.h
  +-- ops.h (Output, Input)
  +-- graph.h
  +-- status.h

session.h
  +-- tensor.h
  +-- graph.pb.h
  +-- config.pb.h
```
