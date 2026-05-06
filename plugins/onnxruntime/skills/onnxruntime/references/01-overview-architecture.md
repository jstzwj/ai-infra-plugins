# ONNX Runtime Reference - Chapter 1: Overview and Architecture

This chapter provides a comprehensive overview of ONNX Runtime's design philosophy, system architecture, code layout, and the core inference pipeline.

---

## 1.1 Design Philosophy

ONNX Runtime is designed as a **cross-platform, high-performance inference and training accelerator** for ONNX models. The key design principles are:

1. **Hardware Agnostic**: Abstract execution providers allow the same model to run on CPUs, GPUs, NPUs, and edge devices
2. **Extensible**: Plugin architecture for custom operators and execution providers
3. **Optimized**: Multi-level graph optimization pipeline with hardware-specific backends
4. **Cross-Language**: C API foundation with bindings for Python, C#, Java, JavaScript, Rust, Objective-C
5. **Standards-Based**: Full ONNX specification compliance with support for all standard opsets

---

## 1.2 System Architecture

### 1.2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Language Bindings Layer                          │
│  Python │ C# │ Java │ JavaScript/TS │ Rust │ Objective-C │ C/C++   │
├─────────────────────────────────────────────────────────────────────┤
│              C API Layer (onnxruntime_c_api.h)                      │
│              OrtApi structure - all C API function pointers          │
├─────────────────────────────────────────────────────────────────────┤
│              Session Layer                                          │
│  InferenceSession │ TrainingSession │ IOBinding │ RunOptions        │
├─────────────────────────────────────────────────────────────────────┤
│              Graph Processing Layer                                  │
│  Graph IR │ Optimizer (L1-L4) │ Partitioner │ Shape Inference       │
├─────────────────────────────────────────────────────────────────────┤
│              Execution Layer                                        │
│  KernelRegistry │ OpKernel │ ExecutionProviders │ Executor          │
├─────────────────────────────────────────────────────────────────────┤
│              Runtime Infrastructure                                  │
│  MLAS │ Allocators │ ThreadPools │ Platform │ Logging │ Profiling  │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2.2 Core Inference Pipeline

The inference pipeline follows a strict sequence:

```
1. Environment Creation
   Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "app_name");
   ↓
2. Session Options Configuration
   Ort::SessionOptions opts;
   opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
   ↓
3. Execution Provider Registration
   opts.AppendExecutionProvider_CUDA(cuda_options);
   ↓
4. Session Creation (Load + Initialize)
   Ort::Session session(env, model_path, opts);
   │
   ├── 4a. Load Model (ONNX protobuf → Graph IR)
   ├── 4b. Graph Optimization (Level 1 → Level 2 → Level 3 → Level 4)
   ├── 4c. Graph Partitioning (Assign nodes to EPs)
   ├── 4d. Kernel Registration (Map ops → kernel implementations)
   └── 4e. Memory Planning (Allocate buffers, pre-pack weights)
   ↓
5. Input Preparation
   auto input_tensor = Ort::Value::CreateTensor<float>(...);
   ↓
6. Inference Execution
   auto outputs = session.Run(run_options, input_names, &input, 1, output_names, 1);
   │
   ├── 6a. Input validation and tensor conversion
   ├── 6b. Sequential/parallel execution schedule
   ├── 6c. OpKernel::Compute() for each node
   ├── 6d. EP-specific execution (CUDA, TensorRT, etc.)
   └── 6e. Output gathering and memory management
   ↓
7. Output Processing
   float* output_data = outputs[0].GetTensorData<float>();
```

### 1.2.3 Session Lifecycle States

```
    [Uninitialized]
          │ Load()
          ↓
    [Loaded]  ─── Model parsed, Graph IR built
          │ Initialize()
          ↓
    [Initialized] ─── Graph optimized, EPs assigned, kernels registered
          │ Run()
          ↓
    [Ready for Inference]
```

---

## 1.3 Source Code Layout

### 1.3.1 Top-Level Directory Structure

```
onnxruntime/
├── cmake/                     # Build system configuration
│   ├── CMakeLists.txt         # Main CMake entry point
│   ├── onnxruntime.cmake      # Core ORT build rules
│   ├── onnxruntime_*.cmake    # Per-component build files
│   ├── external/              # External dependencies (protobuf, ONNX, etc.)
│   └── patches/               # Patches for third-party dependencies
├── include/                   # Public API headers
│   └── onnxruntime/
│       ├── core/              # Core framework headers
│       │   ├── session/       # C/C++ API headers
│       │   ├── framework/     # Tensor, Allocator, OpKernel headers
│       │   ├── graph/         # Graph, Node, Model headers
│       │   ├── common/        # Status, logging, containers
│       │   ├── providers/     # Execution provider factory headers
│       │   ├── platform/      # Thread pool, OS abstraction
│       │   └── optimizer/     # Graph transformer headers
│       └── ep/                # New EP adapter API
├── onnxruntime/               # Core implementation
│   ├── core/                  # Core C++ implementation
│   │   ├── common/            # Utilities, Status, logging, containers
│   │   ├── framework/         # OpKernel, Tensor, Allocator, Session
│   │   ├── graph/             # Graph IR, Model, Node, NodeArg
│   │   ├── optimizer/         # Graph optimization passes
│   │   ├── providers/         # Execution provider implementations
│   │   │   ├── cpu/           # CPU EP and all CPU kernel implementations
│   │   │   ├── cuda/          # CUDA EP (if built with CUDA)
│   │   │   ├── tensorrt/      # TensorRT EP
│   │   │   ├── dnnl/          # oneDNN EP
│   │   │   ├── openvino/      # OpenVINO EP
│   │   │   ├── coreml/        # CoreML EP
│   │   │   ├── nnapi/         # NNAPI EP
│   │   │   ├── dml/           # DirectML EP
│   │   │   ├── webgpu/        # WebGPU EP
│   │   │   ├── qnn/           # QNN EP
│   │   │   └── shared/        # Shared EP utilities
│   │   ├── session/           # InferenceSession, Environment
│   │   ├── platform/          # OS abstraction (threading, file I/O)
│   │   ├── mlas/              # Machine Learning Acceleration System
│   │   ├── eager/             # Eager mode execution
│   │   ├── quantization/      # Quantization utilities
│   │   ├── util/              # Math utilities
│   │   ├── flatbuffers/       # ORT format serialization
│   │   └── dlpack/            # DLPack tensor integration
│   ├── contrib_ops/           # Contributed (non-standard) operators
│   │   ├── cpu/               # CPU contrib kernels
│   │   ├── cuda/              # CUDA contrib kernels
│   │   ├── js/                # JavaScript contrib kernels
│   │   └── webgpu/            # WebGPU contrib kernels
│   ├── lora/                  # LoRA adapter support
│   ├── python/                # Python bindings (pybind11)
│   ├── test/                  # Test suite
│   ├── tool/                  # Command-line tools
│   └── wasm/                  # WebAssembly build support
├── orttraining/               # Training-specific code
│   └── orttraining/
│       ├── orttraining/       # Training session, gradient ops, optimizers
│       └── tools/             # Training utilities
├── csharp/                    # C# bindings
│   ├── src/                   # Source code (Microsoft.ML.OnnxRuntime)
│   ├── test/                  # Tests
│   └── sample/                # Samples
├── java/                      # Java bindings
│   └── src/                   # Source code
├── js/                        # JavaScript/TypeScript bindings
│   ├── common/                # Shared code
│   ├── node/                  # Node.js bindings
│   ├── web/                   # Web/browser bindings
│   └── react_native/          # React Native bindings
├── rust/                      # Rust bindings
├── objectivec/                # Objective-C bindings
├── plugin-ep-cuda/            # CUDA EP plugin (separate shared library)
├── plugin-ep-webgpu/          # WebGPU EP plugin
├── docs/                      # Documentation
│   ├── c_cxx/                 # C/C++ API docs
│   ├── python/                # Python API docs
│   ├── execution_providers/   # EP documentation
│   └── cuda_plugin_ep/        # CUDA plugin EP docs
├── tools/                     # Build and development tools
├── samples/                   # Sample code
├── dockerfiles/               # Docker configurations
└── winml/                     # WinML integration
```

### 1.3.2 Key Header Files

| Header | Location | Purpose |
|--------|----------|---------|
| `onnxruntime_c_api.h` | `include/onnxruntime/core/session/` | Complete C API |
| `onnxruntime_cxx_api.h` | `include/onnxruntime/core/session/` | C++ wrapper API |
| `onnxruntime_cxx_inline.h` | `include/onnxruntime/core/session/` | C++ inline implementations |
| `onnxruntime_session_options_config_keys.h` | `include/onnxruntime/core/session/` | Session config keys |
| `onnxruntime_env_config_keys.h` | `include/onnxruntime/core/session/` | Environment config keys |
| `onnxruntime_run_options_config_keys.h` | `include/onnxruntime/core/session/` | Run options config keys |
| `onnxruntime_ep_c_api.h` | `include/onnxruntime/core/session/` | EP C API |
| `onnxruntime_float16.h` | `include/onnxruntime/core/session/` | Float16 type |
| `onnxruntime_lite_custom_op.h` | `include/onnxruntime/core/session/` | Simplified custom op API |
| `tensor.h` | `include/onnxruntime/core/framework/` | Tensor class |
| `tensor_shape.h` | `include/onnxruntime/core/framework/` | TensorShape class |
| `allocator.h` | `include/onnxruntime/core/framework/` | Allocator interfaces |
| `op_kernel.h` | `include/onnxruntime/core/framework/` | OpKernel base class |
| `op_kernel_context.h` | `include/onnxruntime/core/framework/` | OpKernelContext |
| `execution_provider.h` | `include/onnxruntime/core/framework/` | IExecutionProvider |
| `ort_value.h` | `include/onnxruntime/core/framework/` | OrtValue container |
| `data_types.h` | `include/onnxruntime/core/framework/` | Data type definitions |
| `kernel_registry.h` | `include/onnxruntime/core/framework/` | Kernel registry |
| `graph.h` | `include/onnxruntime/core/graph/` | Graph class |
| `graph_viewer.h` | `include/onnxruntime/core/graph/` | Read-only graph view |
| `model_saving_options.h` | `include/onnxruntime/core/graph/` | Model saving options |
| `environment.h` | `include/onnxruntime/core/session/` | Environment class |

---

## 1.4 Core Components

### 1.4.1 Environment (Ort::Env)

The `Environment` is a global singleton that manages:
- Thread pool creation and configuration
- Allocator registration
- Logging system initialization
- Telemetry settings
- Language projection tracking

```cpp
// C++ API
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "my_application");
Ort::Env env_with_custom_logger(ORT_LOGGING_LEVEL_VERBOSE, "app",
    my_logging_function, my_logger_param);
```

```c
// C API
OrtEnv* env;
OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "my_app", &env);
// Or with custom logger
OrtCreateEnvWithCustomLogger(logging_func, param,
    ORT_LOGGING_LEVEL_WARNING, "my_app", &env);
```

### 1.4.2 Session (Ort::Session / InferenceSession)

The `InferenceSession` is the primary entry point for model inference:

**Lifecycle:**
1. **Construction**: Creates session with options
2. **Load()**: Parses ONNX model into Graph IR
3. **Initialize()**: Runs optimizations, assigns EPs, registers kernels
4. **Run()**: Executes inference on input tensors

**Key Methods:**
```cpp
class InferenceSession {
    // Load model
    Status Load(const std::string& model_uri);
    Status Load(const void* model_data, size_t model_size);

    // Initialize (called automatically by constructor with model path)
    Status Initialize();

    // Run inference
    Status Run(const RunOptions& run_options,
               const std::vector<std::string>& feed_names,
               const std::vector<OrtValue>& feeds,
               const std::vector<std::string>& output_names,
               std::vector<OrtValue>* p_fetches);

    // Get model info
    std::vector<std::string> GetInputNames() const;
    std::vector<std::string> GetOutputNames() const;
    ModelMetadata GetModelMetadata() const;
};
```

### 1.4.3 SessionOptions

Configurable parameters for session behavior:

```cpp
Ort::SessionOptions options;

// Thread configuration
options.SetIntraOpNumThreads(4);      // Threads within a single op
options.SetInterOpNumThreads(1);      // Threads for parallel ops
options.SetExecutionMode(ORT_SEQUENTIAL);  // or ORT_PARALLEL

// Optimization
options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

// Memory
options.EnableMemPattern(true);
options.EnableMemReuse(true);

// Execution providers
options.AppendExecutionProvider_CUDA(0);  // device_id = 0

// Custom config entries
options.AddConfigEntry("session.disable_prepacking", "0");
options.AddConfigEntry("session.set_denormal_as_zero", "1");
```

### 1.4.4 Execution Providers

Execution providers are hardware-specific backends. The architecture follows:

```
IExecutionProvider (interface)
├── CPUExecutionProvider        # Default fallback, always available
├── CUDAExecutionProvider       # NVIDIA GPUs
├── TensorrtExecutionProvider   # NVIDIA TensorRT
├── DnnlExecutionProvider       # Intel oneDNN
├── OpenVINOExecutionProvider   # Intel OpenVINO
├── CoreMLExecutionProvider     # Apple CoreML
├── NnapiExecutionProvider      # Android NNAPI
├── DmlExecutionProvider        # Windows DirectML
├── WebGpuExecutionProvider     # WebGPU
├── QNNExecutionProvider        # Qualcomm NPU
├── ACLExecutionProvider        # ARM Compute Library
├── CANNExecutionProvider       # Huawei Ascend
├── VitisAIExecutionProvider    # AMD/Xilinx FPGA
├── XnnpackExecutionProvider    # XNNPACK CPU
├── RknpuExecutionProvider      # Rockchip NPU
├── TvmExecutionProvider        # Apache TVM
├── WebNNExecutionProvider      # Web Neural Network API
└── WinMLExecutionProvider      # Windows ML
```

**EP Registration Flow:**
1. EP is created with specific options
2. EP is appended to SessionOptions
3. During session initialization:
   a. EP declares supported ops via `GetCapability()`
   b. Graph partitioner assigns nodes to EPs
   c. Unsupported nodes fall back to CPU EP
4. Kernel registry maps (op_type, EP) → kernel implementation

### 1.4.5 Graph IR

The Graph IR is the internal representation of ONNX models:

```cpp
// Key classes
class Model {
    Graph& MainGraph();
    const ModelMetaData& MetaData() const;
    Version IrVersion() const;
};

class Graph {
    // Node management
    Node& AddNode(const std::string& name, const std::string& op_type,
                  const std::vector<std::string>& inputs,
                  const std::vector<std::string>& outputs);
    bool RemoveNode(NodeIndex node_index);

    // Access
    const std::vector<const Node*>& Nodes() const;
    GraphViewer CreateGraphViewer() const;

    // Optimization
    Status Resolve();
    void SetGraphResolveNeeded();
};

class Node {
    const std::string& Name() const;
    const std::string& OpType() const;
    const std::string& GetExecutionProviderType() const;
    ConstPointerContainer<std::vector<NodeArg*>> InputDefs() const;
    ConstPointerContainer<std::vector<NodeArg*>> OutputDefs() const;
    const NodeAttributes& GetAttributes() const;
};

class NodeArg {
    const std::string& Name() const;
    const TypeProto* Type() const;
    const TensorShapeProto* Shape() const;
};
```

### 1.4.6 Graph Optimization Levels

```
Level 1 - Basic (always applied):
├── Constant folding
├── Dead code elimination
├── Operator fusion (basic)
└── Redundant node elimination

Level 2 - Extended:
├── All Level 1 optimizations
├── Complex operator fusion (Conv+BN, Gemm+Activation)
├── Layout transformation (NCHW ↔ NHWC)
├── Attention fusion
└── Embedding layer fusion

Level 3 - Layout:
├── All Level 2 optimizations
├── NHWC layout transformation
└── NCHWc layout transformation

Level 4 - Full:
├── All Level 3 optimizations
├── QDQ (QuantizeLinear/DequantizeLinear) fusion
├── MatMulNBits conversion
├── BFloat16 conversion
└── EP-specific optimizations
```

### 1.4.7 Tensor System

```cpp
// Tensor element data types (ONNXTensorElementDataType enum)
ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED    = 0
ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT        = 1   // IEEE 754 float32
ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8        = 2
ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8         = 3
ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16       = 4
ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16        = 5
ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32        = 6
ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64        = 7
ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING       = 8
ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL         = 9
ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16      = 10  // IEEE 754 float16
ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE       = 11
ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32       = 12
ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64       = 13
ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64    = 14
ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128   = 15
ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16     = 16  // Brain float16
ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN = 17
ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ = 18
ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2   = 19
ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ = 20
ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4        = 21  // Packed 4-bit unsigned
ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4         = 22  // Packed 4-bit signed
ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1   = 23  // Packed float4
ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2        = 24  // Packed 2-bit unsigned
ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2         = 25  // Packed 2-bit signed
```

### 1.4.8 Memory Management

```
OrtMemoryInfo
├── OrtAllocatorType: OrtDeviceAllocator | OrtArenaAllocator | OrtReadOnlyAllocator
├── OrtMemType: OrtMemTypeCPU | OrtMemTypeDefault
├── OrtDevice: CPU | GPU | FPGA | NPU
└── Allocator Instance ID

Allocator Hierarchy:
IAllocator (interface)
├── CPUAllocator           # Standard malloc/free
├── BFCArena              # Best-Fit with Coalescing arena
├── StreamAwareBFCArena   # Stream-aware GPU arena
├── TPUArenaAllocator     # TPU-specific arena
└── Custom allocators via OrtAllocator struct

Memory Flow:
1. Session creates allocators per EP and memory type
2. Tensors are allocated via IAllocator::Alloc()
3. Arena allocators pool memory for reuse
4. Pre-packed weights use dedicated allocators
5. IO Binding enables zero-copy for pinned memory
```

---

## 1.5 Build Configurations

### 1.5.1 Major Build Variants

| Variant | Description |
|---------|-------------|
| `onnxruntime` | Full inference build |
| `onnxruntime_training` | Training-enabled build |
| `onnxruntime_minimal` | Minimal build for constrained environments |
| `onnxruntime_extended` | Extended minimal build |
| `onnxruntime_webgpu` | WebGPU-enabled build |
| `onnxruntime_webassembly` | WebAssembly build |

### 1.5.2 Key CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `onnxruntime_ENABLE_CUDA` | OFF | Enable CUDA EP |
| `onnxruntime_ENABLE_TENSORRT` | OFF | Enable TensorRT EP |
| `onnxruntime_ENABLE_DNNL` | OFF | Enable oneDNN EP |
| `onnxruntime_ENABLE_OPENVINO` | OFF | Enable OpenVINO EP |
| `onnxruntime_ENABLE_COREML` | OFF | Enable CoreML EP |
| `onnxruntime_ENABLE_NNAPI` | OFF | Enable NNAPI EP |
| `onnxruntime_ENABLE_WEBGPU` | OFF | Enable WebGPU EP |
| `onnxruntime_ENABLE_DML` | OFF | Enable DirectML EP |
| `onnxruntime_ENABLE_QNN` | OFF | Enable QNN EP |
| `onnxruntime_ENABLE_TRAINING` | OFF | Enable training support |
| `onnxruntime_ENABLE_PYTHON` | OFF | Build Python bindings |
| `onnxruntime_BUILD_UNIT_TESTS` | ON | Build unit tests |
| `onnxruntime_BUILD_SHARED_LIB` | ON | Build shared library |
| `onnxruntime_MINIMAL_BUILD` | OFF | Minimal build |
| `onnxruntime_EXTENDED_MINIMAL_BUILD` | OFF | Extended minimal build |
| `onnxruntime_USE_FLASH_ATTENTION` | OFF | Use Flash Attention |
| `onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION` | OFF | Memory-efficient attention |
| `onnxruntime_ENABLE_LORA` | OFF | Enable LoRA support |

---

## 1.6 Key Design Patterns

### 1.6.1 Error Handling

```cpp
// C++ API: Exceptions
try {
    auto session = Ort::Session(env, model_path, options);
} catch (const Ort::Exception& e) {
    std::cerr << "Error: " << e.what()
              << " Code: " << e.GetOrtErrorCode();
}

// C API: Return status
OrtStatus* status = OrtSessionCreate(env, model_path, options, &session);
if (status != nullptr) {
    const char* msg = OrtGetErrorMessage(status);
    OrtErrorCode code = OrtGetErrorCode(status);
    OrtReleaseStatus(status);
}
```

### 1.6.2 Resource Management (RAII)

All C++ wrapper classes follow RAII:
- `Ort::Env` - owns the environment
- `Ort::Session` - owns the session
- `Ort::SessionOptions` - owns the session options
- `Ort::Value` - owns an OrtValue
- `Ort::MemoryInfo` - owns memory info
- `Ort::IoBinding` - owns IO binding

Copy is disabled; only move semantics are supported. Use `Clone()` for explicit copies.

### 1.6.3 Kernel Registration Pattern

```cpp
// Register a custom kernel
class MyCustomOp : public Ort::CustomOpBase<MyCustomOp, MyCustomKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo& info) const override;
    const char* GetName() const override;
    const char* GetExecutionProviderType() const override;
    std::size_t GetInputTypeCount() const override;
    ONNXTensorElementDataType GetInputType(std::size_t idx) const override;
    std::size_t GetOutputTypeCount() const override;
    ONNXTensorElementDataType GetOutputType(std::size_t idx) const override;
};
```

---

## 1.7 Versioning

- **API Version**: `ORT_API_VERSION = 27` (incremented for breaking changes)
- **ONNX Opset Support**: Up to the latest released ONNX opset
- **ONNX IR Version**: Full compatibility with ONNX specification
- **Runtime Version**: Follows semantic versioning (major.minor.patch)
