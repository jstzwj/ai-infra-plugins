# ONNX Runtime Reference - Chapter 44: ONNX Format and Model Loading

This chapter covers the ONNX protobuf format, ORT FlatBuffers format, the complete model loading pipeline, external data handling, serialization, and large model support in ONNX Runtime.

---

## 44.1 ONNX Protobuf Format

### 44.1.1 Top-Level Model Structure

The ONNX model is defined as a Protocol Buffers message in `onnx.proto3`:

```protobuf
// onnx/onnx.proto3
syntax = "proto3";
package onnx;

message ModelProto {
    optional int64 ir_version = 1;
    repeated OperatorSetIdProto opset_import = 8;
    optional string producer_name = 2;
    optional string producer_version = 3;
    optional string domain = 7;
    optional int64 model_version = 5;
    optional string doc_string = 6;
    optional GraphProto graph = 7;
    repeated NodeProto metadata_props = 14;
    repeated TrainingInfoProto training_info = 20;
    repeated FunctionProto functions = 25;
    optional string[] extra_meta_data = 26;  // reserved
}

message GraphProto {
    repeated NodeProto node = 1;
    optional string name = 2;       // Graph name
    repeated TensorProto initializer = 5;
    repeated SparseTensorProto sparse_initializer = 17;
    optional string doc_string = 9;
    repeated ValueInfoProto input = 11;
    repeated ValueInfoProto output = 12;
    repeated ValueInfoProto value_info = 13;
    repeated TensorAnnotation quantization_annotation = 14;
    optional int64 ir_version = 20;  // (reserved)
}

message NodeProto {
    repeated string input = 1;
    repeated string output = 2;
    optional string name = 3;
    optional string op_type = 4;
    optional string domain = 7;
    repeated AttributeProto attribute = 5;
    optional string doc_string = 6;
}

message AttributeProto {
    optional string name = 1;       // attribute name
    optional string ref_attr_name = 21;  // reference to sub-graph attribute
    optional string doc_string = 13;
    optional AttributeType type = 20;    // data type

    // Value fields (oneof based on type)
    optional float f = 2;
    optional int64 i = 3;
    optional bytes s = 4;
    optional TensorProto t = 5;
    optional GraphProto g = 6;
    optional SparseTensorProto sparse_tensor = 22;

    // List value fields
    repeated float floats = 7;
    repeated int64 ints = 8;
    repeated bytes strings = 9;
    repeated TensorProto tensors = 10;
    repeated GraphProto graphs = 11;
    repeated SparseTensorProto sparse_tensors = 23;
    // Type proto for type attributes
    optional TypeProto tp = 14;
    repeated TypeProto type_protos = 15;
}
```

### 44.1.2 TensorProto

```protobuf
message TensorProto {
    repeated int64 dims = 1;
    optional DataType data_type = 2;

    // For small tensors, data is stored inline
    // For large tensors, data may be in external files
    optional bytes raw_data = 13;

    // Typed data fields (legacy, prefer raw_data)
    repeated float float_data = 4 [packed = true];
    repeated int64 int64_data = 8 [packed = true];
    repeated int32 int32_data = 5 [packed = true];
    repeated uint8 byte_data = 0;    // bytes for uint8
    // ... other typed fields

    optional string name = 7;
    optional StringStringEntryProto external_data = 12;

    // Data location
    enum DataLocation {
        DEFAULT = 0;
        EXTERNAL = 1;
    }
    optional DataLocation data_location = 14;

    // External data details
    repeated StringStringEntryProto external_data = 12;
}

message StringStringEntryProto {
    optional string key = 1;
    optional string value = 2;
}
```

### 44.1.3 ValueInfoProto and TypeProto

```protobuf
message ValueInfoProto {
    optional string name = 1;
    optional TypeProto type = 2;
    optional string doc_string = 3;
}

message TypeProto {
    oneof value {
        Tensor tensor_type = 1;
        Sequence sequence_type = 4;
        Map map_type = 5;
        Optional optional_type = 9;
        // SparseTensor, Opaque, etc.
    }

    message Tensor {
        optional int32 elem_type = 1;
        optional TensorShapeProto shape = 2;
    }

    message TensorShapeProto {
        message Dimension {
            oneof value {
                int64 dim_value = 1;
                string dim_param = 2;   // symbolic dimension
            }
            optional string denotation = 3;
        }
        repeated Dimension dim = 1;
    }
}
```

### 44.1.4 OperatorSetIdProto

```protobuf
message OperatorSetIdProto {
    optional string domain = 1;
    optional int64 version = 2;
}
```

### 44.1.5 ONNX Data Types

```protobuf
enum DataType {
    UNDEFINED = 0;
    FLOAT = 1;           // IEEE 754 float32
    UINT8 = 2;
    INT8 = 3;
    UINT16 = 4;
    INT16 = 5;
    INT32 = 6;
    INT64 = 7;
    STRING = 8;
    BOOL = 9;
    FLOAT16 = 10;        // IEEE 754 float16
    DOUBLE = 11;
    UINT32 = 12;
    UINT64 = 13;
    COMPLEX64 = 14;      // complex with float32 real and imaginary
    COMPLEX128 = 15;     // complex with float64 real and imaginary
    BFLOAT16 = 16;       // Non-IEEE float16 (bfloat16)
    FLOAT8E4M3FN = 17;   // 8-bit floating point (E4M3)
    FLOAT8E4M3FNUZ = 18; // 8-bit floating point (E4M3, unsigned zero)
    FLOAT8E5M2 = 19;     // 8-bit floating point (E5M2)
    FLOAT8E5M2FNUZ = 20; // 8-bit floating point (E5M2, unsigned zero)
    UINT4 = 21;          // 4-bit unsigned integer
    INT4 = 22;           // 4-bit signed integer
}
```

---

## 44.2 ORT Format (FlatBuffers-Based)

### 44.2.1 Overview

The ORT format is an alternative serialization format based on FlatBuffers that provides faster loading times and lower memory overhead compared to protobuf-based ONNX format.

| Feature | ONNX (Protobuf) | ORT (FlatBuffers) |
|---------|-----------------|-------------------|
| Loading speed | Slower (parsing) | Faster (zero-copy) |
| Memory usage | Higher (copy needed) | Lower (memory-mapped) |
| File size | Larger (protobuf overhead) | Smaller |
| Compatibility | Universal | ORT-specific |
| Mutability | Editable | Read-optimized |

### 44.2.2 FlatBuffers Schema

```flatbuffers
// onnxruntime/core/flatbuffers/schema.fbs
file_identifier "ORTM";

table OrtModel {
    ort_version: uint64;
    model_version: uint64;
    producer_name: string;
    producer_version: string;
    description: string;
    domain: string;
    graph: Graph;
    opset_import: [OperatorSetId];
    metadata_props: [StringStringEntry];
    rt_info: [StringStringEntry];
}

table Graph {
    name: string;
    nodes: [Node];
    initializers: [Tensor];
    sparse_initializers: [SparseTensor];
    inputs: [ValueInfo];
    outputs: [ValueInfo];
    value_info: [ValueInfo];
    node_args: [NodeArg];
}

table Node {
    name: string;
    op_type: string;
    domain: string;
    description: string;
    input: [string];
    output: [string];
    attributes: [Attribute];
    since_version: uint32;
    doc_string: string;
}

table Tensor {
    name: string;
    data_type: uint32;
    shape: [int64];
    data: [ubyte];
    external_data_key: string;
    data_location: uint32;
}

table ValueInfo {
    name: string;
    type: uint32;
    elem_type: uint32;
    shape: [int64];
    denotation: string;
    doc_string: string;
}

table Attribute {
    name: string;
    type: uint32;
    f: float;
    i: int64;
    s: [ubyte];
    floats: [float];
    ints: [int64];
    strings: [[ubyte]];
    sub_graph: Graph;
}

table OperatorSetId {
    domain: string;
    version: uint64;
}

table StringStringEntry {
    key: string;
    value: string;
}

root_type OrtModel;
```

### 44.2.3 ORT Format Advantages

1. **Zero-copy access**: FlatBuffers allows direct access to data without parsing/deserialization
2. **Memory-mapped**: Models can be loaded via mmap without reading the entire file into memory
3. **Pre-optimized**: Graph optimizations can be baked into the ORT file
4. **External initializers**: Large tensors can reference external data files
5. **Pre-packed weights**: Weights can be pre-packed (e.g., for Conv) during serialization

---

## 44.3 Model Loading Pipeline

### 44.3.1 Complete Loading Flow

```
User: Ort::Session(env, "model.onnx", options)
   │
   ├── 1. Format Detection
   │     ├── Check file magic bytes
   │     ├── .onnx → protobuf format
   │     └── .ort → FlatBuffers format
   │
   ├── 2. File Reading
   │     ├── Read entire file (small models)
   │     └── Memory-map (large models, ORT format)
   │
   ├── 3. Deserialization
   │     ├── Protobuf: Parse ModelProto → onnx::ModelProto
   │     └── FlatBuffers: GetOrtModel() → direct access
   │
   ├── 4. Model Validation
   │     ├── Check IR version compatibility
   │     ├── Check opset versions
   │     ├── Validate graph structure
   │     └── Check for unsupported ops
   │
   ├── 5. Graph Construction
   │     ├── Create Graph IR objects (Node, NodeArg, Edge)
   │     ├── Resolve graph (topological sort, type/shape inference)
   │     └── Handle sub-graphs (If, Loop, Scan)
   │
   ├── 6. Graph Optimization
   │     ├── Level 1: Basic (constant folding, dead code elimination)
   │     ├── Level 2: Extended (node fusion, layout transform)
   │     ├── Level 3: Layout (NHWC conversion)
   │     └── Level 4: EP-specific (CUDA fusion, TensorRT compilation)
   │
   ├── 7. Graph Partitioning
   │     ├── Query each EP for supported nodes (GetCapability)
   │     ├── Assign nodes to EPs
   │     └── Create fused sub-graphs for EP compilation
   │
   ├── 8. Kernel Registration
   │     ├── Look up kernel implementations for each node
   │     ├── Compile fused sub-graphs (EP.Compile())
   │     └── Create OpKernel instances
   │
   ├── 9. Memory Planning
   │     ├── Compute memory requirements
   │     ├── Plan buffer reuse (memory pattern optimization)
   │     └── Pre-pack weights (Conv, MatMul)
   │
   └── 10. Session Ready
         └── Model is ready for inference
```

### 44.3.2 Format Detection

```cpp
// onnxruntime/core/session/environment.cc
ModelFormat DetectModelFormat(const std::string& model_path) {
    // Read first few bytes for magic number detection
    std::ifstream file(model_path, std::ios::binary);
    char magic[8] = {0};
    file.read(magic, sizeof(magic));

    // FlatBuffers files start with the FlatBuffers identifier
    // ORT files have "ORTM" identifier at offset 4
    if (magic[4] == 'O' && magic[5] == 'R' &&
        magic[6] == 'T' && magic[7] == 'M') {
        return ModelFormat::ORT_FLATBUFFERS;
    }

    // Protobuf files have varying magic bytes
    // Check for common ONNX protobuf patterns
    // Field 1 (ir_version) is varint, field 7 (graph) is length-delimited
    // A protobuf message with field 7 = length-delimited starts with 0x3A
    // followed by the graph length

    // Also check file extension as a fallback
    if (model_path.size() >= 4 &&
        model_path.substr(model_path.size() - 4) == ".ort") {
        return ModelFormat::ORT_FLATBUFFERS;
    }

    return ModelFormat::ONNX_PROTOBUF;
}
```

### 44.3.3 Protobuf Model Loading

```cpp
// onnxruntime/core/graph/model.cc
Status Model::Load(const std::string& model_path,
                   IOnnxModelMetadefIdGenerator* metadef_id_generator,
                   const ModelLoadOptions& options,
                   std::shared_ptr<Model>& model) {
    // Read the file
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    // Parse protobuf
    ONNX_NAMESPACE::ModelProto model_proto;
    if (!model_proto.ParseFromArray(buffer.data(), static_cast<int>(size))) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_PROTOBUF,
            "Failed to parse ONNX model from: ", model_path);
    }

    // Validate
    ORT_RETURN_IF_ERROR(ValidateModel(model_proto));

    // Create Model object
    model = std::make_shared<Model>(model_proto, model_path,
                                     metadef_id_generator, options);
    return Status::OK();
}

Status Model::Load(const void* model_data, size_t model_size,
                   const ModelLoadOptions& options,
                   std::shared_ptr<Model>& model) {
    ONNX_NAMESPACE::ModelProto model_proto;
    if (!model_proto.ParseFromArray(model_data,
                                     static_cast<int>(model_size))) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_PROTOBUF,
            "Failed to parse ONNX model from memory");
    }

    model = std::make_shared<Model>(model_proto, "",
                                     nullptr, options);
    return Status::OK();
}
```

### 44.3.4 FlatBuffers Model Loading

```cpp
// onnxruntime/core/flatbuffers/flatbuffers_utils.cc
Status Model::LoadFromOrtFormat(const std::string& model_path,
                                const ModelLoadOptions& options,
                                std::shared_ptr<Model>& model) {
    // Memory-map the file for zero-copy access
    MappedMemory mapped;
    ORT_RETURN_IF_ERROR(MapFileToMemory(model_path, mapped));

    // Verify FlatBuffers format
    flatbuffers::Verifier verifier(
        reinterpret_cast<const uint8_t*>(mapped.data()),
        mapped.size());
    if (!fbs::VerifyOrtModelBuffer(verifier)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH,
            "Invalid ORT format file: ", model_path);
    }

    // Get root object (zero-copy)
    auto ort_model = fbs::GetOrtModel(mapped.data());

    // Convert FlatBuffers model to internal representation
    model = std::make_shared<Model>();
    ORT_RETURN_IF_ERROR(
        model->LoadFromOrtFormatImpl(ort_model, options));

    return Status::OK();
}

Status Model::LoadFromOrtFormatImpl(const fbs::OrtModel* ort_model,
                                     const ModelLoadOptions& options) {
    // Read metadata
    producer_name_ = ort_model->producer_name()->str();
    producer_version_ = ort_model->producer_version()->str();
    model_version_ = ort_model->model_version();
    ir_version_ = ort_model->ort_version();

    // Read opset imports
    if (auto opsets = ort_model->opset_import()) {
        for (auto opset : *opsets) {
            opset_imports_.push_back({opset->domain()->str(),
                                      opset->version()});
        }
    }

    // Build graph from FlatBuffers
    auto graph = ort_model->graph();
    ORT_RETURN_IF_ERROR(
        graph_->LoadFromOrtFormat(graph, options));

    return Status::OK();
}
```

### 44.3.5 Graph Construction from Protobuf

```cpp
// onnxruntime/core/graph/graph.cc
Status Graph::LoadFromModelProto(const ONNX_NAMESPACE::ModelProto& model_proto) {
    const auto& graph_proto = model_proto.graph();

    // 1. Create NodeArg objects for all value_info
    //    (named tensors with type/shape info)
    for (const auto& value_info : graph_proto.value_info()) {
        auto& node_arg = GetOrCreateNodeArg(value_info.name(),
                                             &value_info.type());
        node_arg.SetShape(value_info.type().tensor_type().shape());
    }

    // 2. Create NodeArg objects for graph inputs
    for (const auto& input : graph_proto.input()) {
        auto& node_arg = GetOrCreateNodeArg(input.name(), &input.type());
        graph_inputs_.push_back(&node_arg);
    }

    // 3. Create NodeArg objects for graph outputs
    for (const auto& output : graph_proto.output()) {
        auto& node_arg = GetOrCreateNodeArg(output.name(), &output.type());
        graph_outputs_.push_back(&node_arg);
    }

    // 4. Load initializers (weights, biases)
    for (const auto& initializer : graph_proto.initializer()) {
        auto tensor = std::make_unique<TensorProto>(initializer);
        auto name = initializer.name();
        auto& node_arg = GetOrCreateNodeArg(name,
                                             &GetTypeProto(initializer));

        initializers_[name] = std::move(tensor);
        node_arg.SetInitializer(initializers_[name].get());
    }

    // 5. Create nodes
    for (const auto& node_proto : graph_proto.node()) {
        auto& node = CreateNode(node_proto);
        nodes_.push_back(&node);
    }

    // 6. Resolve graph (topological sort, type inference, etc.)
    ORT_RETURN_IF_ERROR(Resolve());

    return Status::OK();
}
```

### 44.3.6 Graph Resolution

```cpp
Status Graph::Resolve() {
    // 1. Topological sort
    ORT_RETURN_IF_ERROR(PerformTopologicalSort());

    // 2. Type and shape inference
    ORT_RETURN_IF_ERROR(PerformTypeShapeInference());

    // 3. Build node-to-output-edge mapping
    ORT_RETURN_IF_ERROR(BuildNodeToOutputEdgeMap());

    // 4. Validate graph integrity
    ORT_RETURN_IF_ERROR(ValidateGraph());

    // 5. Set resolved flag
    graph_resolved_ = true;

    return Status::OK();
}

Status Graph::PerformTopologicalSort() {
    // Kahn's algorithm for topological sorting
    std::unordered_set<NodeIndex> visited;
    std::queue<NodeIndex> queue;

    // Find nodes with no inputs (source nodes)
    for (const auto& node : Nodes()) {
        if (node.GetInputEdgesCount() == 0) {
            queue.push(node.Index());
        }
    }

    // Process nodes in topological order
    while (!queue.empty()) {
        NodeIndex current = queue.front();
        queue.pop();

        if (visited.count(current)) continue;
        visited.insert(current);

        // Add to sorted list
        topo_sorted_nodes_.push_back(current);

        // Process output edges
        for (auto it = GetNode(current)->OutputEdgesBegin();
             it != GetNode(current)->OutputEdgesEnd(); ++it) {
            NodeIndex next = it->GetNode().Index();
            bool all_inputs_visited = true;

            for (auto input_edge : GetNode(next)->GetInputEdges()) {
                if (!visited.count(input_edge.GetNode().Index())) {
                    all_inputs_visited = false;
                    break;
                }
            }

            if (all_inputs_visited) {
                queue.push(next);
            }
        }
    }

    if (visited.size() != NumberOfNodes()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH,
            "Graph contains a cycle");
    }

    return Status::OK();
}
```

---

## 44.4 External Data Files

### 44.4.1 Overview

For large models that exceed the 2GB protobuf limit, ONNX supports storing tensor data in external files. This is critical for LLMs (Large Language Models) with billions of parameters.

### 44.4.2 External Data Configuration

```protobuf
// In TensorProto, when data_location = EXTERNAL:
// The external_data field contains key-value pairs:
message TensorProto {
    // ...
    repeated StringStringEntryProto external_data = 12;
    optional DataLocation data_location = 14;
}

// Key-value pairs:
// "location"    - Relative path to external data file
// "offset"      - Byte offset within the file (optional, default 0)
// "length"      - Number of bytes to read (optional, default: all)
// "checksum"    - SHA256 checksum of the data (optional)
```

### 44.4.3 External Data Example

```python
import onnx
import numpy as np
from onnx import numpy_helper, TensorProto

# Create a large tensor stored externally
large_weight = np.random.randn(4096, 4096).astype(np.float32)
tensor = numpy_helper.from_array(large_weight, name="large_weight")

# Set up external data
tensor.data_location = TensorProto.EXTERNAL
tensor.external_data.append(
    onnx.StringStringEntryProto(key="location", value="weights/large_weight.bin"))
tensor.external_data.append(
    onnx.StringStringEntryProto(key="offset", value="0"))
tensor.external_data.append(
    onnx.StringStringEntryProto(key="length",
                                value=str(large_weight.nbytes))

# Clear inline data
tensor.ClearField("raw_data")

# Save external data file
with open("weights/large_weight.bin", "wb") as f:
    f.write(large_weight.tobytes())
```

### 44.4.4 Loading External Data in ONNX Runtime

```cpp
// onnxruntime/core/graph/graph.cc
Status Graph::LoadExternalInitializers(const std::string& model_dir) {
    for (auto& [name, tensor_proto] : initializers_) {
        if (tensor_proto->data_location() ==
            ONNX_NAMESPACE::TensorProto::EXTERNAL) {

            // Parse external data info
            std::string location;
            size_t offset = 0;
            size_t length = 0;
            std::string checksum;

            for (const auto& entry : tensor_proto->external_data()) {
                if (entry.key() == "location") {
                    location = entry.value();
                } else if (entry.key() == "offset") {
                    offset = std::stoull(entry.value());
                } else if (entry.key() == "length") {
                    length = std::stoull(entry.value());
                } else if (entry.key() == "checksum") {
                    checksum = entry.value();
                }
            }

            // Resolve path relative to model directory
            std::string external_path =
                PathString(model_dir).ParentPath() + PathString(location);

            // Read external data
            std::ifstream file(external_path, std::ios::binary);
            file.seekg(offset);

            std::vector<uint8_t> data(length);
            file.read(reinterpret_cast<char*>(data.data()), length);

            // Verify checksum if provided
            if (!checksum.empty()) {
                std::string computed = ComputeSHA256(data);
                ORT_RETURN_IF_NOT(computed == checksum,
                    "External data checksum mismatch for tensor: ", name);
            }

            // Set the tensor data
            tensor_proto->set_raw_data(data.data(), length);
            tensor_proto->set_data_location(
                ONNX_NAMESPACE::TensorProto::DEFAULT);
            tensor_proto->clear_external_data();
        }
    }

    return Status::OK();
}
```

### 44.4.5 External Data Save Options

```python
# Python API for saving with external data
import onnx

model = onnx.load("model.onnx")

# Save with external data (all tensors > 1GB)
onnx.save_model(
    model,
    "model_ext.onnx",
    save_as_external_data=True,
    size_threshold=1024 * 1024 * 1024,  # 1GB threshold
    all_tensors_to_one_file=True,
    external_data_location="weights.bin",
    external_data_size_threshold=0,  # 0 = use size_threshold
    convert_attribute=True
)

# Save each tensor to a separate file
onnx.save_model(
    model,
    "model_ext.onnx",
    save_as_external_data=True,
    all_tensors_to_one_file=False
)
```

---

## 44.5 Memory-Mapped Model Loading

### 44.5.1 Overview

Memory-mapped (mmap) loading allows ONNX Runtime to access model data without reading the entire file into memory. This is especially important for large models.

### 44.5.2 Mmap Implementation

```cpp
// onnxruntime/core/platform/file_system.cc
class MappedMemory {
public:
    MappedMemory() = default;
    ~MappedMemory() { Unmap(); }

    Status Map(const std::string& path) {
        // Open file
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                "Failed to open file: ", path);
        }

        // Get file size
        struct stat st;
        fstat(fd_, &st);
        size_ = st.st_size;

        // Memory map
        data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) {
            close(fd_);
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                "Failed to mmap file: ", path);
        }

        // Advise kernel about access pattern
        madvise(data_, size_, MADV_SEQUENTIAL);

        return Status::OK();
    }

    const void* data() const { return data_; }
    size_t size() const { return size_; }

private:
    void Unmap() {
        if (data_ && data_ != MAP_FAILED) {
            munmap(data_, size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    int fd_ = -1;
    void* data_ = nullptr;
    size_t size_ = 0;
};

// Windows implementation using CreateFileMapping/MapViewOfFile
class MappedMemory {
public:
    Status Map(const std::string& path) {
        HANDLE file = CreateFileA(path.c_str(), GENERIC_READ,
                                   FILE_SHARE_READ, nullptr,
                                   OPEN_EXISTING,
                                   FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                "Failed to open file: ", path);
        }

        LARGE_INTEGER file_size;
        GetFileSizeEx(file, &file_size);
        size_ = static_cast<size_t>(file_size.QuadPart);

        HANDLE mapping = CreateFileMappingA(file, nullptr,
                                             PAGE_READONLY, 0, 0, nullptr);
        data_ = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, size_);

        CloseHandle(mapping);
        CloseHandle(file);
        return Status::OK();
    }

private:
    void* data_ = nullptr;
    size_t size_ = 0;
};
```

### 44.5.3 Mmap Usage in ORT Format

```cpp
// The ORT FlatBuffers format is designed for memory-mapped loading
Status LoadOrtModelMmap(const std::string& path,
                         std::unique_ptr<Model>& model) {
    auto mapped = std::make_unique<MappedMemory>();
    ORT_RETURN_IF_ERROR(mapped->Map(path));

    // Zero-copy access to FlatBuffers data
    auto ort_model = fbs::GetOrtModel(mapped->data());

    // FlatBuffers allows direct access without deserialization
    // We keep the mapped memory alive as long as the model is in use
    model = std::make_unique<Model>(std::move(mapped), ort_model);

    return Status::OK();
}
```

---

## 44.6 Model Serialization

### 44.6.1 ONNX Format Serialization

```cpp
// onnxruntime/core/graph/model.cc
Status Model::Save(const std::string& model_path,
                   SaveOption save_option) const {
    // Convert Graph IR to ONNX protobuf
    ONNX_NAMESPACE::ModelProto model_proto;

    // Set metadata
    model_proto.set_ir_version(IR_VERSION);
    model_proto.set_producer_name(producer_name_.c_str());
    model_proto.set_producer_version(producer_version_.c_str());
    model_proto.set_model_version(model_version_);

    // Set opset imports
    for (const auto& opset : opset_imports_) {
        auto* opset_proto = model_proto.add_opset_import();
        opset_proto->set_domain(opset.domain);
        opset_proto->set_version(opset.version);
    }

    // Serialize graph
    auto* graph_proto = model_proto.mutable_graph();
    ORT_RETURN_IF_ERROR(
        graph_->SaveToProto(*graph_proto, save_option));

    // Set metadata properties
    for (const auto& [key, value] : metadata_props_) {
        auto* entry = model_proto.add_metadata_props();
        entry->set_key(key);
        entry->set_value(value);
    }

    // Write to file
    std::string serialized;
    model_proto.SerializeToString(&serialized);

    std::ofstream file(model_path, std::ios::binary);
    file.write(serialized.data(), serialized.size());

    return Status::OK();
}
```

### 44.6.2 ORT Format Serialization

```cpp
// onnxruntime/core/flatbuffers/flatbuffers_utils.cc
Status Model::SaveToOrtFormat(const std::string& model_path) const {
    flatbuffers::FlatBufferBuilder builder(1024);

    // Build opset imports
    std::vector<flatbuffers::Offset<fbs::OperatorSetId>> opset_offsets;
    for (const auto& opset : opset_imports_) {
        auto domain = builder.CreateString(opset.domain);
        opset_offsets.push_back(
            fbs::CreateOperatorSetId(builder, domain, opset.version));
    }
    auto opsets = builder.CreateVector(opset_offsets);

    // Build graph
    auto graph = BuildFlatBuffersGraph(builder);

    // Build model
    auto producer_name = builder.CreateString(producer_name_);
    auto producer_version = builder.CreateString(producer_version_);
    auto description = builder.CreateString(description_);
    auto domain = builder.CreateString(domain_);

    auto model = fbs::CreateOrtModel(
        builder,
        ORT_VERSION,
        model_version_,
        producer_name,
        producer_version,
        description,
        domain,
        graph,
        opsets
    );

    // Finish and write
    fbs::FinishOrtModelBuffer(builder, model);

    auto buf = builder.GetBufferSpan();
    std::ofstream file(model_path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(buf.data()), buf.size());

    return Status::OK();
}
```

### 44.6.3 Save Options

```cpp
enum class SaveOption {
    // Save as standard ONNX protobuf
    ONNX_PROTOBUF,

    // Save as ORT FlatBuffers format
    ORT_FLATBUFFERS,

    // Save with external data for large tensors
    ONNX_PROTOBUF_WITH_EXTERNAL_DATA,

    // Save as ORT format with pre-packed weights
    ORT_FLATBUFFERS_WITH_PREPACKED_WEIGHTS,
};

// Python API
// ort.save_model(session, "model.ort", format="ort")
// ort.save_model(session, "model.onnx", format="onnx")
```

---

## 44.7 External Initializers

### 44.7.1 Overview

External initializers allow model weights to be loaded on-demand rather than all at once during model loading. This is essential for large models.

### 44.7.2 External Initializer Configuration

```python
# Python API for external initializers
import onnxruntime as ort

# Configure session to use external initializers
options = ort.SessionOptions()
options.add_config_entry("session.load_external_initializers", "1")
options.add_config_entry("session.external_initializer_dir", "/path/to/weights")

session = ort.InferenceSession("model.onnx", options)
```

### 44.7.3 Lazy Loading

```cpp
// onnxruntime/core/session/inference_session.cc
Status InferenceSession::Initialize() {
    // ...

    // Option: Load initializers on-demand
    if (session_options_.config_options.GetConfigEntry(
            "session.load_external_initializers") == "1") {
        // Mark initializers as external (don't load immediately)
        for (auto& [name, initializer] : graph_->GetAllInitializers()) {
            if (IsLargeTensor(initializer)) {
                // Register as externally managed
                external_initializers_[name] = {
                    .path = GetExternalDataPath(initializer),
                    .offset = GetExternalDataOffset(initializer),
                    .size = GetExternalDataSize(initializer),
                    .loaded = false
                };
            }
        }
    }

    // ...
}

// Load external initializer on first use
Status InferenceSession::EnsureInitializerLoaded(const std::string& name) {
    auto it = external_initializers_.find(name);
    if (it == external_initializers_.end() || it->second.loaded) {
        return Status::OK();
    }

    // Load the tensor data
    auto& ext = it->second;
    std::ifstream file(ext.path, std::ios::binary);
    file.seekg(ext.offset);

    std::vector<uint8_t> data(ext.size);
    file.read(reinterpret_cast<char*>(data.data()), ext.size);

    // Update the initializer in the graph
    auto& tensor = graph_->GetInitializer(name);
    tensor.set_raw_data(data.data(), ext.size);
    ext.loaded = true;

    return Status::OK();
}
```

---

## 44.8 Large Model Support

### 44.8.1 Challenges with Large Models

Large Language Models (LLMs) present unique challenges:

- **Model size**: Models like LLaMA-70B are ~130GB in FP32, ~65GB in FP16
- **Protobuf limit**: Protobuf has a 2GB maximum message size
- **Memory pressure**: Loading entire model into RAM may not be feasible
- **Weight sharing**: Multiple sessions may share the same weights

### 44.8.2 Large Model Loading Strategies

```
Strategy 1: External Data Files
├── model.onnx (small, contains graph structure only)
├── weights_part0.bin (external data file)
├── weights_part1.bin
└── weights_part2.bin

Strategy 2: ORT Format with Memory Mapping
├── model.ort (FlatBuffers, mmap-able)
└── weights.bin (external data, mmap-able)

Strategy 3: EP Context (Pre-compiled)
├── model.onnx (contains EPContext nodes)
└── compiled_cache.bin (EP-specific compiled binary)
```

### 44.8.3 Session Options for Large Models

```python
import onnxruntime as ort

options = ort.SessionOptions()

# Enable memory-mapped model loading
options.add_config_entry("session.enable_mem_pattern", "0")  # Disable for large models
options.add_config_entry("session.enable_mem_reuse", "1")

# Configure external data
options.add_config_entry("session.load_external_initializers_lazily", "1")

# Limit memory allocation
options.add_config_entry("session.memory_arena_config", "max_memory:8589934592")  # 8GB

# Enable inter-op parallelism for model partitioning
options.intra_op_num_threads = 4
options.inter_op_num_threads = 4

# Use EP context for pre-compiled models
options.add_config_entry("ep.context_enable", "1")
options.add_config_entry("ep.context_file_path", "compiled_model.bin")

session = ort.InferenceSession("large_model.onnx", options)
```

### 44.8.4 Model Splitting for Multi-GPU

```python
import onnx
from onnx import helper, TensorProto
import numpy as np

def split_model_for_multi_gpu(model_path, num_gpus):
    """Split a large model across multiple GPUs."""
    model = onnx.load(model_path, load_external_data=False)
    graph = model.graph

    # Get all initializers (weights)
    initializers = list(graph.initializer)
    total_params = sum(np.frombuffer(init.raw_data, dtype=np.float32).size
                       for init in initializers)
    params_per_gpu = total_params // num_gpus

    # Split initializers across GPUs
    gpu_models = []
    current_params = 0
    current_gpu = 0
    gpu_initializers = [[] for _ in range(num_gpus)]

    for init in initializers:
        size = np.frombuffer(init.raw_data, dtype=np.float32).size
        gpu_initializers[current_gpu].append(init)
        current_params += size

        if current_params >= params_per_gpu and current_gpu < num_gpus - 1:
            current_gpu += 1
            current_params = 0

    # Create per-GPU models
    for gpu_id in range(num_gpus):
        gpu_model = onnx.ModelProto()
        gpu_model.CopyFrom(model)
        gpu_model.graph.ClearField('initializer')
        for init in gpu_initializers[gpu_id]:
            gpu_model.graph.initializer.append(init)
        gpu_models.append(gpu_model)

    return gpu_models
```

---

## 44.9 Model Version Compatibility

### 44.9.1 IR Version Handling

```cpp
// onnxruntime/core/graph/model.cc
Status ValidateModel(const ONNX_NAMESPACE::ModelProto& model_proto) {
    int64_t ir_version = model_proto.ir_version();

    // ORT supports IR versions 3-10 (as of ORT 1.20)
    if (ir_version < 3 || ir_version > 10) {
        if (ir_version > 10) {
            LOGS_DEFAULT(WARNING)
                << "Model IR version " << ir_version
                << " is newer than maximum supported version 10. "
                << "Attempting to load anyway.";
        } else {
            return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH,
                "Model IR version ", ir_version,
                " is not supported. Minimum supported version is 3.");
        }
    }

    return Status::OK();
}
```

### 44.9.2 ONNX IR Version History

| IR Version | ONNX Release | Key Changes |
|-----------|-------------|-------------|
| 3 | 1.0 | Initial stable release |
| 4 | 1.1 | Optional inputs and outputs |
| 5 | 1.2 | Operator functions |
| 6 | 1.3 | Training info, quantization annotations |
| 7 | 1.5 | Sparse tensors |
| 8 | 1.7 | Type annotation on formal parameters |
| 9 | 1.10 | Functions in opset imports |
| 10 | 1.15 | Extended data types (uint4, int4, float8) |

### 44.9.3 Opset Version Handling

```cpp
// onnxruntime/core/graph/graph.cc
Status Graph::ValidateOpsetVersions(
    const ONNX_NAMESPACE::ModelProto& model_proto) {

    // Determine the effective opset for each domain
    for (const auto& opset : model_proto.opset_import()) {
        std::string domain = opset.domain().empty() ? "" : opset.domain();
        int64_t version = opset.version();

        // Check if this opset version is supported
        auto supported_versions = GetSupportedOpsetVersions(domain);
        if (supported_versions.find(version) == supported_versions.end()) {
            // Try to find the highest supported version
            int64_t best_version = 0;
            for (auto v : supported_versions) {
                if (v <= version && v > best_version) {
                    best_version = v;
                }
            }

            if (best_version == 0) {
                return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH,
                    "Unsupported opset version ", version,
                    " for domain '", domain, "'");
            }

            if (best_version < version) {
                LOGS_DEFAULT(WARNING)
                    << "Model requests opset version " << version
                    << " for domain '" << domain
                    << "', but ORT only supports up to version "
                    << best_version << ". Using version " << best_version;
            }
        }
    }

    return Status::OK();
}
```

### 44.9.4 Supported Opset Versions

```cpp
// Domain: "" (default ONNX domain)
// Supported versions: 1-21 (as of ORT 1.20)
//
// Domain: "ai.onnx.ml" (ML domain)
// Supported versions: 1-4
//
// Domain: "ai.onnx.training" (Training domain)
// Supported versions: 1-1
//
// Domain: "com.microsoft" (Microsoft contrib domain)
// Supported versions: 1-1
//
// Key opset version changes:
// Opset 1-5:  Basic operators
// Opset 6:    Add quantization ops (QuantizeLinear, DequantizeLinear)
// Opset 7:    Add broadcast support to more ops
// Opset 8:   Add string tensor support
// Opset 9:   Add EyeLike, Compress, OneHot
// Opset 10:  Add StringNormalizer, ThresholdedRelu changes
// Opset 11:  Add Pad, Resize, changes to Gather, ScatterElements
// Opset 12:  Add Einsum, Det, Hardmax changes
// Opset 13:  Add Squeeze/Unsqueeze changes, CumSum, Round
// Opset 14:  Add CumSum, BitShift, Det
// Opset 15:  Add Bernoulli, CastLike
// Opset 16:  Add GridSample, Trilu, Optional
// Opset 17:  Add LayerNormalization, NegativeLogLikelihoodLoss
// Opset 18:  Add AffineGrid, DFT, SequenceMap
// Opset 19:  Add Summary statistics ops
// Opset 20:  Add more ops
// Opset 21:  Latest additions
```

### 44.9.5 Opset Version Compatibility Check

```python
import onnxruntime as ort

def check_model_compatibility(model_path):
    """Check if a model is compatible with the current ORT version."""
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3  # VERBOSE

    try:
        session = ort.InferenceSession(model_path, session_options)
        print("Model is fully compatible!")

        # Print model info
        print(f"  Inputs: {session.get_inputs()}")
        print(f"  Outputs: {session.get_outputs()}")
        print(f"  Overridable initializers: {session.get_overridable_initializers()}")

        # Get model metadata
        model_meta = session.get_modelmeta()
        print(f"  Producer: {model_meta.producer_name}")
        print(f"  Graph name: {model_meta.graph_name}")
        print(f"  Description: {model_meta.description}")
        print(f"  Domain: {model_meta.domain}")
        print(f"  Version: {model_meta.version}")
        print(f"  Custom metadata: {model_meta.custom_metadata_map}")

        return True
    except ort.RuntimeException as e:
        print(f"Compatibility error: {e}")
        return False
```

### 44.9.6 Version-Compatible Model Saving

```python
import onnx
from onnx import helper, optimizer

def save_model_compatible(model, output_path, target_opset=14):
    """Save model with a specific opset version for compatibility."""
    # Update opset version
    opset_imports = [helper.make_opsetid("", target_opset)]
    model.opset_import[:] = opset_imports

    # Run shape inference to ensure completeness
    from onnx import shape_inference
    model = shape_inference.infer_shapes(model)

    # Validate
    onnx.checker.check_model(model)

    # Save
    onnx.save(model, output_path)
```

---

## 44.10 Model Loading API Reference

### 44.10.1 C++ API

```cpp
// Create session from file
Ort::Session session(env, "model.onnx", session_options);

// Create session from memory
std::vector<uint8_t> model_data = ReadFile("model.onnx");
Ort::Session session(env, model_data.data(), model_data.size(),
                      session_options);

// Create session from pre-compiled ORT format
Ort::Session session(env, "model.ort", session_options);
```

### 44.10.2 C API

```c
// Load model from file path
OrtStatus* OrtSessionCreate(
    const OrtEnv* env,
    const char* model_path,
    const OrtSessionOptions* options,
    OrtSession** out);

// Load model from memory
OrtStatus* OrtSessionCreateFromArray(
    const OrtEnv* env,
    const void* model_data,
    size_t model_data_length,
    const OrtSessionOptions* options,
    OrtSession** out);
```

### 44.10.3 Python API

```python
import onnxruntime as ort

# From file
session = ort.InferenceSession("model.onnx")

# From bytes
with open("model.onnx", "rb") as f:
    model_bytes = f.read()
session = ort.InferenceSession(model_bytes)

# With session options
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession("model.onnx", options)

# With execution providers
session = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    provider_options=[{"device_id": 0}, {}]
)
```

### 44.10.4 Model Loading Configuration Options

```python
# All available session config options for model loading
options = ort.SessionOptions()

# Model format detection
options.add_config_entry("session.force_ort_format", "0")  # Auto-detect
options.add_config_entry("session.force_ort_format", "1")  # Force ORT format

# External data
options.add_config_entry("session.load_external_data", "1")
options.add_config_entry("session.external_data_dir", "/path/to/data")

# Memory-mapped loading (ORT format only)
options.add_config_entry("session.enable_mem_pattern", "1")
options.add_config_entry("session.enable_mem_reuse", "1")

# Pre-packed weights
options.add_config_entry("session.enable_prepacking", "1")

# Graph optimization level during loading
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL     # 0
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC    # 1
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED # 2
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL      # 99

# Serialized model cache (for faster subsequent loads)
options.add_config_entry("session.use_per_session_threads", "1")
options.add_config_entry("session.use_deterministic_compute", "0")
```

---

## 44.11 Summary

| Topic | Key Points |
|-------|-----------|
| ONNX Protobuf | Standard format, `ModelProto` → `GraphProto` → `NodeProto`, tensor data inline or external |
| ORT FlatBuffers | Faster loading, zero-copy, memory-mapped, `OrtModel` → `Graph` → `Node` |
| Model Loading | Format detection → deserialization → validation → Graph IR → optimization → partitioning → kernel registration → memory planning |
| External Data | Large tensors stored in separate files, `data_location = EXTERNAL`, key-value metadata |
| Memory Mapping | mmap for ORT format, zero-copy access, essential for large models |
| Serialization | Save as ONNX protobuf or ORT FlatBuffers, external data options |
| Version Compatibility | IR version 3-10, opset version 1-21, forward-compatible with warnings |
| Large Models | External data files, lazy loading, multi-GPU splitting, EP context pre-compilation |
