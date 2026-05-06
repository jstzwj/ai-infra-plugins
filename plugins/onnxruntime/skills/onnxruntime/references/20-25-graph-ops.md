# ONNX Runtime Reference - Chapter 20: Graph System Architecture

---

## 20.1 Graph Class

The Graph class is the core IR representation of ONNX models.

```cpp
class Graph {
public:
    // Construction
    Graph(const Model& model, const GraphProto& graph_proto,
          const std::unordered_map<std::string, const NodeProto*>& funcProtos,
          Version ir_version, const logging::Logger& logger);

    // Node management
    Node& AddNode(const std::string& name, const std::string& op_type,
                  const std::vector<std::string>& inputs,
                  const std::vector<std::string>& outputs);
    bool RemoveNode(NodeIndex node_index);

    // Accessors
    const std::vector<const Node*>& Nodes() const;
    size_t NumberOfNodes() const;
    const Node* GetNode(NodeIndex index) const;

    // Graph operations
    Status Resolve();
    bool GraphResolveNeeded() const;
    void SetGraphResolveNeeded();

    // Input/Output
    const std::vector<const NodeArg*>& GetInputs() const;
    const std::vector<const NodeArg*>& GetOutputs() const;
    const std::vector<const NodeArg*>& GetInputsIncludingInitializers() const;

    // Initializers (weights)
    const InitializedTensorSet& GetAllInitializedTensors() const;
    Status AddInitializedTensor(const TensorProto& tensor);
    bool RemoveInitializedTensor(const std::string& name);

    // Viewer
    GraphViewer CreateGraphViewer() const;

    // Subgraphs (If, Loop, Scan)
    Graph* MutableSubgraph(const Node& node, int index);
    const Graph* GetSubgraph(const Node& node, int index) const;

    // Functions
    const FunctionContainer& GetFunctionContainer() const;
};
```

## 20.2 GraphViewer (Read-only)

```cpp
class GraphViewer {
public:
    const std::vector<const Node*>& Nodes() const;
    const Node* GetNode(NodeIndex index) const;
    size_t NumberOfNodes() const;

    const std::vector<const NodeArg*>& GetInputs() const;
    const std::vector<const NodeArg*>& GetOutputs() const;
    const InitializedTensorSet& GetAllInitializedTensors() const;

    std::vector<NodeIndex> GetNodesInTopologicalOrder() const;
    const std::vector<NodeIndex>& GetRootNodes() const;
};
```

## 20.3 Node Class

```cpp
class Node {
public:
    // Identity
    const std::string& Name() const;
    const std::string& OpType() const;
    const std::string& Domain() const;
    NodeIndex Index() const;

    // Execution Provider
    const std::string& GetExecutionProviderType() const;
    void SetExecutionProviderType(const std::string& type);

    // I/O
    ConstPointerContainer<std::vector<NodeArg*>> InputDefs() const;
    ConstPointerContainer<std::vector<NodeArg*>> OutputDefs() const;
    ConstPointerContainer<std::vector<NodeArg*>> ImplicitInputDefs() const;

    // Attributes
    const NodeAttributes& GetAttributes() const;
    const AttributeProto* GetAttribute(const std::string& name) const;

    // Edges
    const std::vector<Edge>& GetEdges() const;
    std::vector<const Node*> InputNodes() const;
    std::vector<const Node*> OutputNodes() const;

    // Subgraphs
    const std::vector<std::reference_wrapper<Graph>>& GetSubgraphs() const;
};
```

## 20.4 NodeArg Class

```cpp
class NodeArg {
public:
    const std::string& Name() const;
    const TypeProto* Type() const;
    const TensorShapeProto* Shape() const;
    ONNXDataType DataType() const;
    bool Exists() const;
};
```

## 20.5 Model Class

```cpp
class Model {
public:
    explicit Model(const std::string& producer_name, Version ir_version);
    Model(const ModelProto& model_proto, const logging::Logger& logger);

    Graph& MainGraph();
    const Graph& MainGraph() const;

    const std::string& ProducerName() const;
    void SetProducerName(const std::string& name);

    Version IrVersion() const;
    Version ModelVersion() const;

    const ModelMetaData& MetaData() const;
    ModelMetaData& MutableMetaData();

    // Serialization
    Status Save(const ModelSavingOptions& options,
                const PathString& file_path) const;
    Status SaveToByteStream(const ModelSavingOptions& options,
                             std::ostream& stream) const;
};
```
