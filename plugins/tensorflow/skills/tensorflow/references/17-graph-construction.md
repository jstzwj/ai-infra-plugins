# TensorFlow C++ Graph Construction

This reference covers TensorFlow's graph construction internals: the Graph,
Node, and Edge classes, the NodeBuilder utility, graph serialization via
GraphDef/NodeDef, and the infrastructure for graph optimization, validation,
and partitioning.

---

## Table of Contents

1. [Graph Class](#graph-class)
2. [Node Class](#node-class)
3. [Edge Class](#edge-class)
4. [GraphDef Protocol Buffer](#graphdef-protocol-buffer)
5. [NodeDef Protocol Buffer](#nodedef-protocol-buffer)
6. [NodeBuilder](#nodebuilder)
7. [Shape Refinement](#shape-refinement)
8. [Graph Optimization Passes](#graph-optimization-passes)
9. [Graph Partitioning](#graph-partitioning)
10. [Control Flow Edges](#control-flow-edges)
11. [Graph Validation](#graph-validation)
12. [Graph to GraphDef Conversion](#graph-to-graphdef-conversion)

---

## Graph Class

**Header:** `tensorflow/core/graph/graph.h`
**Namespace:** `tensorflow`

The `Graph` class represents a computation graph as a directed acyclic graph
(DAG). It contains:

- **Internal nodes** representing computational operations
- **Edges** representing data and control dependencies
- **Special nodes**: a source node (id 0) and a sink node (id 1)

The source node is the only node with no incoming dependencies; the sink node
is the only node with no outgoing dependencies. All other nodes should have
both incoming and outgoing edges.

### Construction

```cpp
// Construct with an op registry (looks up ops in the registry).
explicit Graph(const OpRegistryInterface* ops);

// Construct with a function library definition.
explicit Graph(const FunctionLibraryDefinition& flib_def);

// Clone the entire graph.
std::unique_ptr<Graph> Clone();
```

The constructor creates a graph with two special nodes:
- **Source** (id = `kSourceId` = 0): Entry point, no inputs
- **Sink** (id = `kSinkId` = 1): Exit point, no outputs

An edge from Source to Sink is automatically added.

### Node Management

```cpp
// Add a new node from a NodeDef.
// Returns nullptr and sets status on error.
Node* AddNode(NodeDef node_def, absl::Status* status);

// StatusOr variant (preferred).
absl::StatusOr<Node*> AddNode(NodeDef node_def);

// Copy a node from another graph (no edges copied).
Node* CopyNode(const Node* node);

// Remove a node and all its edges.
// REQUIRES: node->IsOp()
void RemoveNode(Node* node);

// Remove all nodes and edges.
void Clear();

// Copy from another graph.
void Copy(const Graph& src);
```

### Edge Management

```cpp
// Add a data edge from source's x-th output to dest's y-th input.
// Does not update dest's NodeDef.
const Edge* AddEdge(Node* source, int x, Node* dest, int y);

// Add a control edge from source to dest.
// Updates dest's NodeDef with the control input.
const Edge* AddControlEdge(Node* source, Node* dest,
                           bool allow_duplicates = false);

// Remove a data edge.
// REQUIRES: edge must exist.
void RemoveEdge(const Edge* edge);

// Remove a control edge (also updates NodeDef).
// REQUIRES: control edge must exist.
void RemoveControlEdge(const Edge* e);

// Update an input edge (replaces existing edge to dst).
absl::Status UpdateEdge(Node* new_src, int new_src_index,
                        Node* dst, int dst_index);

// Special hack for While op gradient construction.
absl::Status AddWhileInputHack(Node* new_src, int new_src_index, Node* dst);
```

### Query Methods

```cpp
// Number of live nodes (including Source and Sink).
int num_nodes() const;

// Number of live op nodes (excluding Source and Sink).
int num_op_nodes() const;

// Number of live edges.
int num_edges() const;

// Maximum node id (>= num_nodes because of removed nodes).
int num_node_ids() const;

// Maximum edge id (>= num_edges because of removed edges).
int num_edge_ids() const;

// Find node by id (returns nullptr if removed).
Node* FindNodeId(int id) const;

// Find edge by id (returns nullptr if removed).
const Edge* FindEdgeId(int id) const;
```

### Iteration

```cpp
// Iterate over all nodes (including Source and Sink).
gtl::iterator_range<NodeIter> nodes() const;
// for (Node* node : graph.nodes()) { ... }

// Iterate over op nodes only (excluding Source and Sink).
gtl::iterator_range<NodeIter> op_nodes() const;
// for (Node* node : graph.op_nodes()) { ... }

// Iterate over all edges.
GraphEdgesIterable edges() const;
// for (const Edge* e : graph.edges()) { ... }
```

### Special Nodes

```cpp
enum { kSourceId = 0, kSinkId = 1 };

Node* source_node() const;  // Always id 0
Node* sink_node() const;    // Always id 1
```

### Version Management

```cpp
// The GraphDef version range.
const VersionDef& versions() const;
void set_versions(const VersionDef& versions);
```

### Function Library

```cpp
// Access the op registry and function library.
const OpRegistryInterface* op_registry() const;
const FunctionLibraryDefinition& flib_def() const;
FunctionLibraryDefinition* mutable_flib_def();

// Add function/gradient definitions.
absl::Status AddFunctionLibrary(const FunctionDefLibrary& fdef_lib);
absl::Status AddFunctionLibrary(FunctionDefLibrary&& fdef_lib);
absl::Status AddFunctionDef(const FunctionDef& fdef,
                            const StackTracesMap& stack_traces);
absl::Status AddGradientDef(const GradientDef& gdef);
```

### Device Name Management

```cpp
// Intern a device name string (returns index).
int InternDeviceName(const std::string& device_name);

// Get the assigned device name for a node.
const std::string& get_assigned_device_name(const Node& node) const;

// Set assigned device for a node.
void set_assigned_device_name_index(Node* node, int device_name_index);
void set_assigned_device_name(Node* node, const std::string& device_name);
```

### Utility Methods

```cpp
// Generate a unique node name with the given prefix.
std::string NewName(absl::string_view prefix);

// Validate a node belongs to this graph.
absl::Status IsValidNode(const Node* node) const;
absl::Status IsValidOutputTensor(const Node* node, int idx) const;
absl::Status IsValidInputTensor(const Node* node, int idx) const;

// Build a name-to-node index.
std::unordered_map<std::string, Node*> BuildNodeNameIndex() const;

// While loop context creation.
absl::Status AddWhileContext(absl::string_view frame_name,
                            std::vector<Node*> enter_nodes,
                            std::vector<Node*> exit_nodes,
                            OutputTensor cond_output,
                            std::vector<OutputTensor> body_inputs,
                            std::vector<OutputTensor> body_outputs,
                            WhileContext** result);

// Full type management.
void SetNodeType(absl::string_view name, const FullTypeDef& type);
```

### Construction Context

```cpp
// Indicates where the graph instance originated from.
enum class ConstructionContext {
  kNotTracked,     // Not tracked.
  kDirectSession,  // From DirectSession (TF1 session API).
  kEagerRuntime,   // Registered from TF2 eager runtime.
};

void SetConstructionContext(ConstructionContext ctx);
ConstructionContext GetConstructionContextInternal() const;
```

### Control Flow Slot Constant

```cpp
static constexpr int kControlSlot = -1;
// Edges with src_output == kControlSlot or dst_input == kControlSlot
// are control edges (no data flows).
```

### Thread Safety

The `Graph` class is thread-compatible but NOT thread-safe. External
synchronization is required for concurrent access.

---

## Node Class

**Header:** `tensorflow/core/graph/graph.h`

`Node` represents a single operation in the computation graph. Each node has an
id, name, op type, input/output types, attributes, and device assignment.

### Identity

```cpp
// Unique integer id (dense in 0..max_id range, but may have gaps).
int id() const;

// Cost accounting id (-1 if no corresponding cost accounting node).
int cost_id() const;

// Node name (unique within the graph).
const std::string& name() const;
void set_name(std::string name);

// Op type string (e.g., "MatMul", "Add", "Const").
const std::string& type_string() const;
```

### NodeDef Access

```cpp
// The NodeDef (user-supplied definition).
// Note: def().input() is NOT reliable; use in_edges() instead.
// def().device() is the requested device, not the assigned device.
// def().attr() is authoritative.
const NodeDef& def() const;
NodeDef* mutable_def();

// The OpDef (operation specification from the op registry).
const OpDef& op_def() const;

// Attribute access (read-only).
AttrSlice attrs() const;

// Requested inputs from NodeDef (unreliable for actual edges).
const protobuf::RepeatedPtrField<std::string>& requested_inputs() const;
```

### Input/Output Types

```cpp
// Number of inputs/outputs.
int32_t num_inputs() const;
int32_t num_outputs() const;

// Type of a specific input/output.
DataType input_type(int32_t i) const;
DataType output_type(int32_t o) const;

// All input/output types.
const DataTypeVector& input_types() const;
const DataTypeVector& output_types() const;
```

### Device Management

```cpp
// The device requested by the user (from NodeDef).
const std::string& requested_device() const;
void set_requested_device(const std::string& device);

// The device assigned by the runtime (actual placement).
const std::string& assigned_device_name() const;
void set_assigned_device_name(const std::string& device_name);
bool has_assigned_device_name() const;
int assigned_device_name_index() const;
void set_assigned_device_name_index(int index);
```

### Edge Access

```cpp
// All incoming/outgoing edges (including control edges).
const EdgeSet& in_edges() const;
const EdgeSet& out_edges() const;

// Neighboring nodes via edges.
gtl::iterator_range<NeighborIter> in_nodes() const;
gtl::iterator_range<NeighborIter> out_nodes() const;

// Get the edge connecting to the 'idx' input.
absl::Status input_edge(int idx, const Edge** e) const;

// Get all input data edges (not control edges).
absl::Status input_edges(std::vector<const Edge*>* edges) const;

// Get the source node for the 'idx' input.
absl::Status input_node(int idx, const Node** n) const;
absl::Status input_node(int idx, Node** n) const;

// Get the idx-th input tensor (as OutputTensor of input_node).
absl::Status input_tensor(int idx, OutputTensor* t) const;
```

### Attribute Management

```cpp
// Add an attribute.
template <typename T>
void AddAttr(const std::string& name, const T& val);

// Clear an attribute.
void ClearAttr(const std::string& name);
```

### Node Type Checks

```cpp
// Special node checks.
bool IsSource() const;           // id() == 0
bool IsSink() const;             // id() == 1
bool IsOp() const;               // id() > 1

// Control flow nodes.
bool IsSwitch() const;
bool IsMerge() const;
bool IsEnter() const;
bool IsExit() const;
bool IsNextIteration() const;
bool IsLoopCond() const;
bool IsControlFlow() const;      // Switch, Merge, Enter, Exit, NextIteration
bool IsControlTrigger() const;

// Communication nodes.
bool IsSend() const;
bool IsRecv() const;
bool IsHostSend() const;
bool IsHostRecv() const;

// Other special nodes.
bool IsConstant() const;
bool IsVariable() const;
bool IsIdentity() const;
bool IsGetSessionHandle() const;
bool IsGetSessionTensor() const;
bool IsDeleteSessionTensor() const;
bool IsScopedAllocator() const;
bool IsCollective() const;
bool IsMetadata() const;
bool IsFakeParam() const;

// Function-related nodes.
bool IsPartitionedCall() const;
bool IsFunctionCall() const;     // Includes function ops, symbolic gradients
bool IsIfNode() const;
bool IsWhileNode() const;
bool IsCaseNode() const;
bool IsArg() const;              // Function input
bool IsRetval() const;           // Function output

// Distributed communication check.
bool IsDistributedCommunication() const;
```

### Debug Information

```cpp
// Set/get original node names (for graph rewriting).
void set_original_node_names(const std::vector<std::string>& names);
void set_original_func_names(const std::vector<std::string>& names);

// Stack trace from node instantiation.
void SetStackTrace(const std::shared_ptr<AbstractStackTrace>& stack_trace);
const std::shared_ptr<AbstractStackTrace>& GetStackTrace() const;

// Debug string.
std::string DebugString() const;

// Access the shared properties.
std::shared_ptr<NodeProperties> properties() const;
```

### While Loop Context

```cpp
// For exit nodes of while loops.
WhileContext* while_ctx() const;
void set_while_ctx(WhileContext* while_ctx);
```

### NodeClass Enumeration

```cpp
enum NodeClass {
  NC_UNINITIALIZED,
  NC_SWITCH,
  NC_MERGE,
  NC_ENTER,
  NC_EXIT,
  NC_NEXT_ITERATION,
  NC_LOOP_COND,
  NC_CONTROL_TRIGGER,
  NC_SEND,
  NC_HOST_SEND,
  NC_RECV,
  NC_HOST_RECV,
  NC_CONSTANT,
  NC_VARIABLE,
  NC_IDENTITY,
  NC_GET_SESSION_HANDLE,
  NC_GET_SESSION_TENSOR,
  NC_DELETE_SESSION_TENSOR,
  NC_METADATA,
  NC_SCOPED_ALLOCATOR,
  NC_COLLECTIVE,
  NC_FAKE_PARAM,
  NC_PARTITIONED_CALL,
  NC_FUNCTION_OP,
  NC_SYMBOLIC_GRADIENT,
  NC_IF,
  NC_WHILE,
  NC_CASE,
  NC_ARG,
  NC_RETVAL,
  NC_OTHER  // Not a special kind of node
};
```

---

## Edge Class

**Header:** `tensorflow/core/graph/graph.h`

`Edge` represents a dependency between two nodes in the graph. Edges carry
either data (from a specific output to a specific input) or control flow
information.

### Accessors

```cpp
class Edge {
 public:
  Node* src() const;       // Source node
  Node* dst() const;       // Destination node
  int id() const;          // Unique edge id

  // Output index on source node.
  // kControlSlot (-1) for control dependencies.
  int src_output() const;

  // Input index on destination node.
  // kControlSlot (-1) for control dependencies.
  int dst_input() const;

  // Check if this is a control edge (no data flow).
  bool IsControlEdge() const;

  // Debug string.
  std::string DebugString() const;
};
```

### EdgeSet

The `EdgeSet` class stores the set of edges connected to a node. It provides
efficient iteration and lookup.

```cpp
class EdgeSet {
 public:
  iterator begin() const;
  iterator end() const;
  size_t size() const;
  bool empty() const;
  // ...
};
```

### OutputTensor and InputTensor

```cpp
// Represents a specific output of a node.
struct OutputTensor {
  Node* node;
  int index;

  OutputTensor(Node* n, int i);
  bool operator==(const OutputTensor& other) const;
  struct Hash { uint64_t operator()(OutputTensor const& s) const; };
};

// Represents a specific input of a node.
struct InputTensor {
  Node* node;
  int index;

  InputTensor(Node* n, int i);
  bool operator==(const InputTensor& other) const;
  struct Hash { uint64_t operator()(InputTensor const& s) const; };
};
```

### Edge Semantics

- **Data edge**: `src_output >= 0` and `dst_input >= 0`. The output tensor at
  index `src_output` of `src` is consumed by the input at index `dst_input` of
  `dst`.
- **Control edge**: `src_output == kControlSlot (-1)` and
  `dst_input == kControlSlot (-1)`. Ensures `src` executes before `dst` but
  no data flows.

---

## GraphDef Protocol Buffer

**Header:** `tensorflow/core/framework/graph.proto`

`GraphDef` is the serialized representation of a `Graph`. It contains all the
information needed to recreate the graph.

```protobuf
message GraphDef {
  repeated NodeDef node = 1;

  // Version information.
  VersionDef versions = 4;

  // Deprecated: use node.name instead.
  // repeated string library = 2;

  // Function library.
  FunctionDefLibrary library = 2;

  // Debug information.
  GraphDebugInfo debug_info = 5;
}
```

### VersionDef

```protobuf
message VersionDef {
  int32 producer = 1;
  int32 min_consumer = 2;
  repeated int32 bad_consumers = 3;
}
```

### FunctionDefLibrary

```protobuf
message FunctionDefLibrary {
  repeated FunctionDef function = 1;
  repeated GradientDef gradient = 2;
  repeated_registered_gradient = 3;
}
```

### Usage

```cpp
// Serialize a Graph to GraphDef.
Graph graph(OpRegistry::Global());
// ... add nodes ...
GraphDef graph_def;
graph.ToGraphDef(&graph_def);

// Deserialize a GraphDef to a Graph.
Graph graph(OpRegistry::Global());
absl::Status status = ImportGraphDef(options, graph_def, &graph);
```

---

## NodeDef Protocol Buffer

**Header:** `tensorflow/core/framework/node_def.proto`

`NodeDef` represents a single node in the serialized graph.

```protobuf
message NodeDef {
  string name = 1;       // Unique name of the node
  string op = 2;         // Operation type (e.g., "MatMul")
  repeated string input = 3;  // Input specifications

  // Device specification.
  string device = 4;

  // Attributes.
  map<string, AttrValue> attr = 5;
}
```

### Input Specification Format

The `input` field uses a specific format:

```
"node_name"           -> Data input from output 0 of node_name
"node_name:output_idx" -> Data input from output_idx of node_name
"^node_name"          -> Control dependency on node_name
```

### Attributes

Attributes are stored as key-value pairs where values are `AttrValue` protocol
buffers:

```protobuf
message AttrValue {
  oneof value {
    bytes s = 2;            // DT_STRING
    int64 i = 3;            // DT_INT64
    float f = 4;            // DT_FLOAT
    bool b = 5;             // DT_BOOL
    DataType type = 6;      // DataType enum
    ShapeProto shape = 7;   // TensorShape
    TensorProto tensor = 8; // Tensor value
    list = 1;               // ListValue
    string placeholder = 9; // Placeholder name
    NameAttrList func = 10; // Function reference
  }

  message ListValue {
    repeated bytes s = 2;
    repeated int64 i = 3;
    repeated float f = 4;
    repeated bool b = 5;
    repeated DataType type = 6;
    repeated ShapeProto shape = 7;
    repeated TensorProto tensor = 8;
    repeated NameAttrList func = 10;
  }
}
```

### NodeDefBuilder

**Header:** `tensorflow/core/framework/node_def_builder.h`

`NodeDefBuilder` is a helper for constructing `NodeDef` objects:

```cpp
NodeDefBuilder builder("node_name", "OpType");
builder.Input({"input_node", 0, DT_FLOAT});
builder.Attr("key", value);
builder.Device("/device:GPU:0");

NodeDef node_def;
absl::Status status = builder.Finalize(&node_def);
```

---

## NodeBuilder

**Header:** `tensorflow/core/graph/node_builder.h`

`NodeBuilder` is a higher-level helper for creating nodes and adding them
directly to a `Graph`. It uses `NodeDefBuilder` internally but also handles
edge creation.

### Construction

```cpp
// Specify name and op type (looks up in OpRegistry::Global()).
NodeBuilder(absl::string_view name, absl::string_view op_name,
            const OpRegistryInterface* op_registry = OpRegistry::Global(),
            const NodeDebugInfo* debug = nullptr);

// Specify name and OpDef directly.
NodeBuilder(absl::string_view name, const OpDef* op_def);

// From an existing NodeDefBuilder.
NodeBuilder(const NodeDefBuilder& def_builder);
```

### NodeOut (Input Specification)

```cpp
struct NodeOut {
  // Reference an existing node.
  NodeOut(Node* n, int32_t i = 0);
  NodeOut(OutputTensor t);

  // Reference a node not yet in the graph (by name).
  NodeOut(absl::string_view name, int32_t i, DataType t);

  NodeOut();  // Default (error state)

  Node* node;
  bool error;        // True if construction failed
  std::string name;  // For nodes not in the graph
  int32_t index;
  DataType dt;
};
```

### Adding Inputs

```cpp
// Single tensor input.
NodeBuilder& Input(Node* src_node, int src_index = 0);
NodeBuilder& Input(NodeOut src);

// List of tensors input.
NodeBuilder& Input(absl::Span<const NodeOut> src_list);
```

### Control Inputs

```cpp
// Require this node to run after src_node(s).
NodeBuilder& ControlInput(Node* src_node);
NodeBuilder& ControlInputs(absl::Span<Node* const> src_nodes);
```

### Device Specification

```cpp
// Set requested device (in NodeDef).
NodeBuilder& Device(absl::string_view device_spec);

// Set assigned device (in the Node object, not NodeDef).
NodeBuilder& AssignedDevice(absl::string_view device);

// Set XLA cluster attribute.
NodeBuilder& XlaCluster(absl::string_view xla_cluster);
```

### Attributes

```cpp
// Set an attribute value.
template <class T>
NodeBuilder& Attr(absl::string_view attr_name, T&& value);

// Set a list attribute.
template <class T>
NodeBuilder& Attr(absl::string_view attr_name,
                  std::initializer_list<T> value);
```

### Finalization

```cpp
// Validate and add the node to the graph.
// Creates edges for all non-back inputs.
absl::Status Finalize(Graph* graph, Node** created_node,
                      bool consume = false);

// StatusOr variant (preferred).
absl::StatusOr<Node*> Finalize(Graph* graph, bool consume = false);

// Accessors.
const std::string& node_name() const;
const OpDef& op_def() const;
```

### Complete Example

```cpp
Graph graph(OpRegistry::Global());

// Create input nodes.
Node* a;
Status s = NodeBuilder("a", "Const")
    .Attr("dtype", DT_FLOAT)
    .Attr("value", Tensor(DT_FLOAT, TensorShape({2, 2})))
    .Finalize(&graph, &a);

Node* b;
s = NodeBuilder("b", "Const")
    .Attr("dtype", DT_FLOAT)
    .Attr("value", Tensor(DT_FLOAT, TensorShape({2, 2})))
    .Finalize(&graph, &b);

// Create a MatMul node.
Node* matmul;
s = NodeBuilder("matmul", "MatMul")
    .Input(a)
    .Input(b)
    .Attr("transpose_a", false)
    .Attr("transpose_b", false)
    .Device("/device:CPU:0")
    .Finalize(&graph, &matmul);
```

---

## Shape Refinement

**Header:** `tensorflow/core/common_runtime/shape_refiner.h`

`ShapeRefiner` performs static shape inference on the graph. It uses the
registered shape functions for each op to propagate shape information from
inputs to outputs.

### Key Concepts

- **ShapeHandle**: Represents a shape in the inference context (may be unknown)
- **DimensionHandle**: Represents a single dimension (may be unknown)
- **InferenceContext**: Provides shape inference utilities for a single op

### Shape Refiner Usage

```cpp
ShapeRefiner refiner(graph.op_registry(), graph.versions());

// Add a node to the refiner (runs shape inference).
absl::Status status = refiner.AddNode(node);

// Update shapes after graph modification.
absl::Status status = refiner.UpdateNode(node, /*relax=*/false);

// Set the shape of a node's output.
refiner.SetShape(node, output_index, shape_handle);
```

### InferenceContext

**Header:** `tensorflow/core/framework/shape_inference.h`

```cpp
namespace shape_inference {
class InferenceContext {
 public:
  // Input access.
  ShapeHandle input(int idx) const;
  int num_inputs() const;

  // Output setting.
  void set_output(int idx, ShapeHandle shape);
  int num_outputs() const;

  // Shape operations.
  ShapeHandle UnknownShape();
  ShapeHandle Scalar();
  ShapeHandle Vector(DimensionHandle dim);
  ShapeHandle Matrix(DimensionHandle dim1, DimensionHandle dim2);
  ShapeHandle ShapeFromShapeProto(const TensorShapeProto& proto);

  // Dimension operations.
  DimensionHandle UnknownDim();
  DimensionHandle MakeDim(int64_t value);
  int64_t Value(DimensionHandle dim);
  bool ValueKnown(DimensionHandle dim);

  // Shape merging.
  absl::Status Merge(ShapeHandle s0, ShapeHandle s1, ShapeHandle* out);
  absl::Status Merge(DimensionHandle d0, DimensionHandle d1,
                     DimensionHandle* out);

  // Rank and dimension access.
  int Rank(ShapeHandle shape);
  DimensionHandle Dim(ShapeHandle shape, int idx);

  // WithRank: enforce a rank constraint.
  absl::Status WithRank(ShapeHandle shape, int rank, ShapeHandle* out);
  absl::Status WithRankAtLeast(ShapeHandle shape, int rank, ShapeHandle* out);
  absl::Status WithRankAtMost(ShapeHandle shape, int rank, ShapeHandle* out);

  // Attribute access.
  template <class T>
  absl::Status GetAttr(absl::string_view attr_name, T* value) const;

  // Handle shapes and types for Variant/Resource types.
  const std::vector<ShapeAndType>* input_handle_shapes_and_types(int idx);
  void set_output_handle_shapes_and_types(
      int idx, const std::vector<ShapeAndType>& shapes_and_types);
};
}
```

### ShapeHandle and DimensionHandle

```cpp
namespace shape_inference {
class ShapeHandle {
 public:
  bool SameHandle(ShapeHandle shape) const;
  // Managed by InferenceContext; do not free.
};

class DimensionHandle {
 public:
  bool SameHandle(DimensionHandle dim) const;
};

struct ShapeAndType {
  ShapeHandle shape;
  DataType dtype = DT_INVALID;
};
}
```

---

## Graph Optimization Passes

**Headers:**
- `tensorflow/core/common_runtime/optimization_registry.h`
- `tensorflow/core/grappler/optimizers/*`

TensorFlow applies a series of optimization passes to the graph before
execution. These are registered and executed in a specific order.

### GraphOptimizer Interface

```cpp
class GraphOptimizer {
 public:
  virtual ~GraphOptimizer() = default;
  virtual absl::Status Optimize(const GrapplerItem& item,
                                GraphDef* optimized_graph) = 0;
  virtual void Feedback(const GrapplerItem& item,
                       const GraphDef& optimized_graph,
                       double result) = 0;
};
```

### Optimization Pass Types

| Pass                       | Description                                       |
|---------------------------|---------------------------------------------------|
| Constant Folding          | Evaluates constant expressions at graph build time |
| Layout Optimizer          | Rewrites graph for optimal data layout (NHWC/NCHW) |
| Memory Optimizer          | Optimizes memory usage and tensor placement        |
| Auto Mixed Precision      | Converts float32 ops to float16 where safe        |
| Arithmetic Optimizer      | Simplifies arithmetic expressions                  |
| Dependency Optimizer      | Removes unnecessary control dependencies           |
| Shape Optimizer           | Simplifies shape-related operations                |
| Remapper                  | Fuses common op patterns                           |
| Loop Optimizer            | Optimizes while loop execution                    |
| Function Optimizer        | Inlines and optimizes function calls               |
| Debug Stripper            | Removes debug operations                           |
| Scoped Allocator Optimizer| Optimizes memory allocation for multi-output ops   |

### Grappler Optimization Pipeline

The Grappler optimizer applies optimizations in a configurable order:

1. **Pre-optimization**: Model pruning, debug stripping
2. **Core optimization**: Constant folding, arithmetic optimization, layout
3. **Post-optimization**: Memory optimization, remapping

### MetaOptimizer

The `MetaOptimizer` is the top-level optimizer that orchestrates all
optimization passes. It is configured via `ConfigProto.RewriterConfig`:

```protobuf
message RewriterConfig {
  enum Toggle {
    DEFAULT = 0;
    ON = 1;
    OFF = 2;
    AGGRESSIVE = 3;
  }
  Toggle layout_optimizer = 1;
  Toggle constant_folding = 2;
  Toggle arithmetic_optimization = 5;
  Toggle dependency_optimization = 8;
  // ... many more toggles
  int64 min_graph_nodes = 17;
  repeated string optimizers = 100;
}
```

---

## Graph Partitioning

**Header:** `tensorflow/core/graph/graph_partition.h`

Graph partitioning splits a computation graph into subgraphs for multi-device
execution. Each subgraph contains nodes assigned to a specific device.

### Partition Function

```cpp
// Partition the graph into subgraphs by device placement.
// Each partition contains nodes assigned to the same device.
// Send/Recv ops are inserted at partition boundaries.
absl::Status Partition(const PartitionOptions& opts, const Graph* g,
                       std::unordered_map<string, GraphDef>* partitions);
```

### PartitionOptions

```cpp
struct PartitionOptions {
  // Returns the device name for a node.
  typedef std::function<string(const Node*)> NodeToNameFunc;
  NodeToNameFunc node_to_loc_func;

  // Returns a unique name for new nodes.
  typedef std::function<string()> NewNameFunc;
  NewNameFunc new_name_func;

  // Whether to include the library in each partition.
  bool include_cvt_funcs = false;

  // Returns true if a node should be placed on a specific device.
  typedef std::function<bool(const Edge*)> ShouldCopyFunc;
};
```

### Send/Recv Insertion

When a tensor needs to cross device boundaries, TensorFlow inserts:
- **Send op** (`_Send` / `_HostSend`): On the source device, sends the tensor
- **Recv op** (`_Recv` / `_HostRecv`): On the destination device, receives the tensor

The `_HostSend` and `_HostRecv` variants ensure data passes through host memory,
which is required for certain operations.

### Cross-Device Execution Flow

```
Device A                    Device B
+------------------+        +------------------+
| ... -> MatMul    |        | Add -> ...       |
|        |         |        |   ^              |
|        v         |        |   |              |
|      _Send ------|------->| _Recv            |
+------------------+        +------------------+
```

---

## Control Flow Edges

Control flow edges ensure execution ordering without carrying data. They are
represented by edges where `src_output == kControlSlot (-1)` and
`dst_input == kControlSlot (-1)`.

### Adding Control Dependencies

```cpp
// Via NodeBuilder.
NodeBuilder& ControlInput(Node* src_node);

// Via Scope (C++ API).
Scope scope = root.WithControlDependencies({op1, op2});

// Via Graph directly.
const Edge* AddControlEdge(Node* source, Node* dest,
                           bool allow_duplicates = false);
```

### Control Flow Operations

TensorFlow's control flow is implemented using special ops:

| Op              | Description                           |
|-----------------|---------------------------------------|
| `Switch`        | Routes input to one of two outputs based on a predicate |
| `Merge`         | Passes the first available input to output |
| `Enter`         | Passes input into a loop frame        |
| `Exit`          | Passes input out of a loop frame      |
| `NextIteration` | Makes input available for next loop iteration |
| `LoopCond`      | Boolean loop continuation predicate   |

### While Loop Structure

```
Enter -> Merge -> (body ops) -> NextIteration
              \                    ^
               -> Switch -> Exit
                   ^
               LoopCond
```

---

## Graph Validation

**Header:** `tensorflow/core/common_runtime/graph_constructor.h`

### VerifyGraphStructure

```cpp
// Validates the graph structure.
// Checks for:
// - Cycles (graph must be a DAG)
// - Invalid node references
// - Type mismatches on edges
// - Missing inputs
absl::Status ValidateGraphDef(const GraphDef& graph_def,
                              const OpRegistryInterface& registry);
```

### GraphConstructorOptions

```cpp
struct GraphConstructorOptions {
  bool allow_internal_ops = false;  // Allow ops starting with '_'
  bool expect_device_spec = false;  // Require device specification
  bool validate_nodes = true;       // Validate node definitions
};
```

### ImportGraphDef

```cpp
// Import a GraphDef into an existing Graph.
absl::Status ImportGraphDef(const GraphConstructorOptions& options,
                            const GraphDef& graph_def,
                            Graph* graph,
                            ShapeRefiner* refiner = nullptr);

// Scoped import (returns a cleanup object).
absl::Status ImportGraphDefToScope(const GraphDef& graph_def,
                                  const GraphConstructorOptions& options,
                                  Scope* scope);
```

### Common Validation Errors

1. **Cycle detected**: Graph contains a cycle, violating DAG requirement
2. **Unregistered op**: Op type not found in the registry
3. **Type mismatch**: Edge connects incompatible dtypes
4. **Invalid input reference**: Input references non-existent node or output
5. **Missing required attributes**: Required attrs not set on a node
6. **Invalid attribute values**: Attribute values don't match expected type/range

---

## Graph to GraphDef Conversion

### ToGraphDef

```cpp
// Serialize a Graph to a GraphDef.
void Graph::ToGraphDef(GraphDef* graph_def,
                       bool include_flib_def = true,
                       bool include_debug_info = false) const;

// Serialize starting from a specific node id.
void Graph::ToGraphDefSubRange(GraphDef* graph_def,
                               int from_node_id,
                               bool include_flib_def = true,
                               bool include_debug_info = false) const;

// Debug version (uses full serialization, not optimized).
GraphDef Graph::ToGraphDefDebug() const;
```

### ImportGraphDef

```cpp
// Deserialize a GraphDef into a Graph.
absl::Status ImportGraphDef(const GraphConstructorOptions& options,
                            const GraphDef& def,
                            Graph* g,
                            ShapeRefiner* refiner);
```

### Conversion Details

When converting `Graph` to `GraphDef`:

1. Each `Node` becomes a `NodeDef`
2. Data edges become `input` strings in the format `"src_name:src_output"`
3. Control edges become `input` strings in the format `"^src_name"`
4. The `versions` field is populated
5. The `library` field is populated with all function definitions
6. If `include_debug_info` is true, stack traces are included

When converting `GraphDef` to `Graph`:

1. Each `NodeDef` becomes a `Node` (with OpDef looked up from registry)
2. `input` strings are parsed to create `Edge` objects
3. Control edges use `kControlSlot` for src_output and dst_input
4. Shape inference may be run (if a `ShapeRefiner` is provided)
5. The graph is validated for structural correctness

### Node Input Ordering

In `NodeDef.input`, inputs are ordered:
1. **Positional data inputs**: In the order defined by the OpDef
2. **Control inputs**: After all data inputs, prefixed with `^`

Example:
```
inputs: ["a:0", "b:0", "^c"]  // Data from a, b; control from c
```

---

## GraphEdgesIterable

The `GraphEdgesIterable` class provides filtered iteration over a graph's edges,
skipping null entries (from removed edges).

```cpp
class GraphEdgesIterable {
 public:
  explicit GraphEdgesIterable(const std::vector<Edge*>& edges);

  class const_iterator {
    // Skips nullptr entries automatically.
    const_iterator& operator++();
    Edge* operator*();
    bool operator==(const const_iterator& other) const;
    bool operator!=(const const_iterator& other) const;
  };

  const_iterator begin();
  const_iterator end();
};
```

---

## NodeDebugInfo

```cpp
struct NodeDebugInfo {
  const std::string name;
  std::vector<std::string> original_node_names;
  std::vector<std::string> original_func_names;

  NodeDebugInfo(const Node& n);
  NodeDebugInfo(const NodeDef& ndef);
  NodeDebugInfo(absl::string_view node_name,
                bool has_experimental_debug_info,
                const NodeDef_ExperimentalDebugInfo& experimental_debug_info);
};
```

---

## Complete Graph Construction Example

```cpp
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/framework/op.h"

using namespace tensorflow;

// Create a graph.
Graph graph(OpRegistry::Global());

// Build a simple computation: c = a + b
Node* a;
auto status = NodeBuilder("a", "Placeholder")
    .Attr("dtype", DT_FLOAT)
    .Attr("shape", TensorShape({}))
    .Finalize(&graph, &a);
TF_CHECK_OK(status);

Node* b;
status = NodeBuilder("b", "Placeholder")
    .Attr("dtype", DT_FLOAT)
    .Attr("shape", TensorShape({}))
    .Finalize(&graph, &b);
TF_CHECK_OK(status);

Node* c;
status = NodeBuilder("c", "Add")
    .Input(a)
    .Input(b)
    .Finalize(&graph, &c);
TF_CHECK_OK(status);

// Add control edge: a must complete before c.
graph.AddControlEdge(a, c);

// Serialize to GraphDef.
GraphDef graph_def;
graph.ToGraphDef(&graph_def);

// Print graph structure.
for (const auto& node : graph_def.node()) {
  LOG(INFO) << "Node: " << node.name()
            << " op: " << node.op()
            << " inputs: " << node.input_size();
}
```

---

## Performance Considerations

### Node ID Allocation

- Node IDs are allocated sequentially but never reused after removal
- `num_node_ids()` >= `num_nodes()` due to gaps from removed nodes
- When creating large arrays indexed by node ID, use `num_node_ids()` as size

### Edge Storage

- Edges are stored in a vector; removed edges leave null entries
- `num_edge_ids()` >= `num_edges()` due to gaps from removed edges
- The `GraphEdgesIterable` class handles null filtering for iteration

### Shape Inference

- Shape inference is performed when nodes are added to a `ShapeRefiner`
- Can be expensive for large graphs
- Consider using `DisabledShapeInferenceScope()` for testing

### Graph Cloning

- `graph.Clone()` creates a deep copy of the entire graph
- All nodes, edges, attributes, and function definitions are copied
- Node pointers in the original graph are NOT valid in the clone

---

## Summary of Key Relationships

```
GraphDef (protobuf)
    |
    | ImportGraphDef()
    v
Graph (in-memory)
    +-- Node[] (owned, by id)
    +-- Edge[] (owned, by id)
    +-- FunctionLibraryDefinition
    +-- VersionDef
    +-- Device names table
    |
    | ToGraphDef()
    v
GraphDef (protobuf)

NodeBuilder -> Graph (adds Node + Edges)
Scope -> Graph (via C++ API ops)
```
