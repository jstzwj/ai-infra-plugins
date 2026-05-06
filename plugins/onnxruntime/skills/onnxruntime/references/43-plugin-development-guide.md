# ONNX Runtime Reference - Chapter 43: Plugin EP Development Guide

This chapter provides a comprehensive guide for developing Execution Provider (EP) plugins for ONNX Runtime, covering the plugin architecture, core interfaces, kernel registration, and detailed breakdowns of the CUDA EP plugin and WebGPU EP plugin implementations.

---

## 43.1 Plugin EP Architecture Overview

ONNX Runtime's plugin EP system allows Execution Providers to be developed and distributed as **standalone shared libraries** (`.so` on Linux, `.dll` on Windows, `.dylib` on macOS) that are loaded at runtime without requiring recompilation of the core ONNX Runtime library.

### 43.1.1 Why Plugin EPs?

- **Decoupled release cycle**: EP plugins can be updated independently of ONNX Runtime core
- **Reduced binary size**: Only load the EPs you need
- **Proprietary EP support**: Hardware vendors can ship closed-source EPs
- **Simplified build**: No need to build ONNX Runtime from source with EP-specific flags
- **Dynamic loading**: EPs can be loaded on demand at runtime

### 43.1.2 Plugin EP vs In-Tree EP

| Aspect | In-Tree EP | Plugin EP |
|--------|-----------|-----------|
| Build | Compiled with ORT core | Standalone shared library |
| Registration | Static, at compile time | Dynamic, at runtime |
| Distribution | Part of ORT package | Separate package |
| API surface | Full C++ internal API | C API boundary only |
| Loading | Always available | On-demand via `ort.load()` |
| Update | Requires ORT rebuild | Independent update |

### 43.1.3 Plugin Directory Structure

```
onnxruntime/
└── plugin-ep-<name>/
    ├── CMakeLists.txt              # Build configuration
    ├── src/
    │   ├── <name>_ep.h             # EP class definition
    │   ├── <name>_ep.cc            # EP implementation
    │   ├── <name>_ep_factory.h     # Factory class definition
    │   ├── <name>_ep_factory.cc    # Factory implementation
    │   ├── <name>_allocator.h      # Custom allocator
    │   ├── <name>_allocator.cc
    │   ├── <name>_kernel.h         # Kernel adapter
    │   ├── <name>_kernel.cc
    │   ├── <name>_stream.h         # Stream handling
    │   ├── <name>_stream.cc
    │   └── kernels/
    │       ├── kernel_registry.cc  # Kernel registration
    │       ├── matmul.h            # Individual kernels
    │       ├── matmul.cc
    │       └── ...
    ├── python/                     # Python bindings (optional)
    │   ├── __init__.py
    │   └── binding.cc
    ├── test/
    │   ├── test_ep.cc
    │   └── test_kernels.cc
    └── docs/
        └── README.md
```

---

## 43.2 OrtEpFactory Interface

The `OrtEpFactory` is the entry point for a plugin EP. It is responsible for creating EP instances and advertising capabilities.

### 43.2.1 OrtEpFactory C API

```c
// ort_api_def.h
typedef void (*OrtEpFactory_CreateEpFn)(
    OrtEpFactory* factory,
    const OrtSessionOptions* session_options,
    const OrtLogger* logger,
    OrtEp** ep);

typedef void (*OrtEpFactory_ReleaseEpFn)(
    OrtEpFactory* factory,
    OrtEp* ep);

typedef const char* (*OrtEpFactory_GetNameFn)(
    OrtEpFactory* factory);

typedef const char* (*OrtEpFactory_GetTypeFn)(
    OrtEpFactory* factory);

typedef OrtMemType (*OrtEpFactory_GetDeviceMemTypeFn)(
    OrtEpFactory* factory);

typedef bool (*OrtEpFactory_IsCudaGraphEnabledFn)(
    OrtEpFactory* factory);

typedef void (*OrtEpFactory_GetHardwareDeviceListFn)(
    OrtEpFactory* factory,
    const OrtHardwareDevice** devices,
    size_t num_devices,
    size_t* out_num_devices);
```

### 43.2.2 OrtEpFactory Structure

```c
struct OrtEpFactory {
    OrtEpFactory_CreateEpFn CreateEp;
    OrtEpFactory_ReleaseEpFn ReleaseEp;
    OrtEpFactory_GetNameFn GetName;
    OrtEpFactory_GetTypeFn GetType;
    OrtEpFactory_GetDeviceMemTypeFn GetDeviceMemType;
    OrtEpFactory_IsCudaGraphEnabledFn IsCudaGraphEnabled;
    OrtEpFactory_GetHardwareDeviceListFn GetHardwareDeviceList;
};
```

### 43.2.3 Factory Registration Entry Point

Every plugin shared library must export a `GetEpApiVersion` function and an `OrtSessionOptionsArtifactFactory`:

```c
// Plugin entry point - must be exported from shared library
ORT_API(void, GetEpApiVersion, int* version) {
    *version = ORT_EP_API_VERSION;  // Currently version 1
}

// Factory creation entry point
ORT_API(const OrtEpFactory*, GetEpFactory) {
    static MyEpFactory factory;
    return &factory;
}
```

### 43.2.4 C++ Factory Wrapper

For C++ implementations, ONNX Runtime provides a base class:

```cpp
// include/onnxruntime/core/framework/execution_provider_base.h
class IExecutionProviderFactory {
public:
    virtual ~IExecutionProviderFactory() = default;
    virtual std::unique_ptr<IExecutionProvider> Create(
        const OrtSessionOptions* session_options,
        const OrtLogger& logger) = 0;
};

// Plugin-specific factory example
class MyEpFactory : public OrtEpFactory {
public:
    MyEpFactory() {
        CreateEp = [](OrtEpFactory* factory,
                       const OrtSessionOptions* session_options,
                       const OrtLogger* logger,
                       OrtEp** ep) {
            auto* self = static_cast<MyEpFactory*>(factory);
            *ep = self->CreateEpImpl(session_options, *logger);
        };

        ReleaseEp = [](OrtEpFactory* factory, OrtEp* ep) {
            auto* self = static_cast<MyEpFactory*>(factory);
            self->ReleaseEpImpl(ep);
        };

        GetName = [](OrtEpFactory* factory) -> const char* {
            return "MyEP";
        };

        GetType = [](OrtEpFactory* factory) -> const char* {
            return "MyHardwareDevice";
        };

        GetDeviceMemType = [](OrtEpFactory* factory) -> OrtMemType {
            return OrtMemType::OrtMemType_Default;
        };
    }

private:
    OrtEp* CreateEpImpl(const OrtSessionOptions* session_options,
                         const OrtLogger& logger) {
        // Parse EP-specific options from session_options
        auto provider = std::make_unique<MyExecutionProvider>(logger);
        return reinterpret_cast<OrtEp*>(provider.release());
    }

    void ReleaseEpImpl(OrtEp* ep) {
        std::unique_ptr<MyExecutionProvider> provider(
            reinterpret_cast<MyExecutionProvider*>(ep));
        // Automatic cleanup via unique_ptr
    }
};
```

---

## 43.3 OrtEp Interface

The `OrtEp` interface represents an active EP instance bound to a session.

### 43.3.1 Core OrtEp Methods

```cpp
class IExecutionProvider {
public:
    // Identity
    virtual const char* Name() const = 0;
    virtual const char* Type() const { return Name(); }

    // Device information
    virtual OrtMemType DeviceMemType() const { return OrtMemType_Default; }
    virtual int GetDeviceId() const { return 0; }

    // Graph partitioning
    virtual std::vector<std::unique_ptr<ComputeCapability>>
    GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                  const IKernelLookup& kernel_lookup) const;

    // Kernel compilation
    virtual std::unordered_map<std::string, std::unique_ptr<OpKernel>>
    Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
            const std::vector<const Node*>& non_fused_nodes);

    // Memory management
    virtual std::vector<AllocatorPtr> CreatePreferredAllocators() const;
    virtual AllocatorPtr GetAllocator(int device_id, OrtMemType mem_type) const;

    // Stream handling
    virtual std::unique_ptr<onnxruntime::Stream>
    CreateStream(const OrtDevice& device) const;
    virtual void ReleaseStream(onnxruntime::Stream* stream);
    virtual OrtDevice::DeviceType GetExecutionDeviceType() const;

    // Synchronization
    virtual Status Sync() const;
    virtual Status OnRunStart() override;
    virtual Status OnRunEnd() override;

    // Data transfer
    virtual std::unique_ptr<IDataTransfer>
    GetDataTransfer() const;

    virtual ~IExecutionProvider() = default;
};
```

### 43.3.2 Key Virtual Methods Explained

#### GetCapability

This method determines which nodes in the graph the EP can execute:

```cpp
std::vector<std::unique_ptr<ComputeCapability>>
MyExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph_viewer,
    const IKernelLookup& kernel_lookup) const {

    std::vector<std::unique_ptr<ComputeCapability>> result;

    for (const auto& node : graph_viewer.Nodes()) {
        // Check if we have a kernel for this op type
        const KernelCreateInfo* kernel_info =
            kernel_lookup.LookUp(node.OpType(), node.Domain(),
                                 node.SinceVersion());

        if (kernel_info != nullptr) {
            // This node can be handled by our EP
            std::unique_ptr<ComputeCapability> cc =
                std::make_unique<ComputeCapability>(
                    std::make_unique<IndexedSubGraph>());
            cc->sub_graph->nodes.push_back(node.Index());
            result.push_back(std::move(cc));
        }
    }

    // Optionally fuse compatible nodes into sub-graphs
    result = FuseCompatibleNodes(graph_viewer, result);

    return result;
}
```

#### Compile

Called for fused sub-graphs that the EP should compile:

```cpp
std::unordered_map<std::string, std::unique_ptr<OpKernel>>
MyExecutionProvider::Compile(
    const std::vector<FusedNodeAndGraph>& fused_nodes,
    const std::vector<const Node*>& non_fused_nodes) {

    std::unordered_map<std::string, std::unique_ptr<OpKernel>> kernels;

    for (const auto& [fused_node, graph_viewer] : fused_nodes) {
        // Compile the fused sub-graph into a kernel
        auto kernel = CompileFusedGraph(fused_node, graph_viewer);
        kernels[fused_node->Name()] = std::move(kernel);
    }

    for (const auto* node : non_fused_nodes) {
        // Create kernel for individual nodes
        auto kernel = CreateKernelForNode(node);
        kernels[node->Name()] = std::move(kernel);
    }

    return kernels;
}
```

---

## 43.4 Kernel Registration in Plugins

### 43.4.1 Kernel Registration API

Plugin EPs register their kernels using the `ortkernel_ort_types.h` API:

```cpp
// Kernel definition structure
struct KernelDef {
    std::string op_name;
    std::string domain;
    int since_version;
    int end_version;
    std::vector<MLDataType> input_types;
    std::vector<MLDataType> output_types;
    std::vector<std::string> type_constraints;
    bool variadic_input = false;
    bool variadic_output = false;
    int input_min_count = 1;
    int output_min_count = 1;
};
```

### 43.4.2 Building a Kernel Registry

```cpp
// In the EP constructor or initialization
void MyExecutionProvider::RegisterKernels(KernelRegistry& registry) {
    // Register individual kernels
    static const BuildKernelCreateInfoFn kernel_table[] = {
        BuildKernelCreateInfo<onnxruntime::MyMatMul>,
        BuildKernelCreateInfo<onnxruntime::MyConv>,
        BuildKernelCreateInfo<onnxruntime::MyRelu>,
        BuildKernelCreateInfo<onnxruntime::MyAdd>,
        // ... more kernels
    };

    for (auto& create_fn : kernel_table) {
        auto kernel_create_info = create_fn();
        registry.Register(kernel_create_info);
    }
}
```

### 43.4.3 Kernel Class Implementation

```cpp
class MyMatMulKernel : public OpKernel {
public:
    MyMatMulKernel(const OpKernelInfo& info) : OpKernel(info) {
        // Parse attributes, extract constants, etc.
        transA_ = info.GetAttrOrDefault<int64_t>("transA", 0);
        transB_ = info.GetAttrOrDefault<int64_t>("transB", 0);
        alpha_ = info.GetAttrOrDefault<float>("alpha", 1.0f);
    }

    Status Compute(OpKernelContext* context) const override {
        const Tensor* A = context->Input<Tensor>(0);
        const Tensor* B = context->Input<Tensor>(1);
        Tensor* Y = context->Output(0, A->Shape());

        // Dispatch to hardware-specific implementation
        return MatMulImpl(A, B, Y, transA_, transB_, alpha_, stream_);
    }

    Status ComputeAsync(OpKernelContext* context,
                        Stream* stream) const override {
        // Async version for stream-aware execution
        stream_ = stream;
        return Compute(context);
    }

private:
    int64_t transA_;
    int64_t transB_;
    float alpha_;
    mutable Stream* stream_ = nullptr;
};

// Kernel registration macro
#define REGISTER_MYEP_KERNEL(op_name, kernel_class)                        \
    static Status kernel_class##_RegFn(                                    \
        KernelRegistry* registry) {                                        \
        return registry->Register(                                          \
            KernelCreateInfo(                                               \
                KernelDefBuilder()                                          \
                    .SetName(op_name)                                       \
                    .SetDomain(onnxruntime::kOnnxDomain)                    \
                    .SinceVersion(1)                                        \
                    .Provider("MyEP")                                       \
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), \
                [](const OpKernelInfo& info) -> std::unique_ptr<OpKernel> { \
                    return std::make_unique<kernel_class>(info);             \
                }));                                                        \
    }
```

### 43.4.4 Kernel Registration via ONNX Runtime Macros

```cpp
// Using the standard ONNX Runtime registration macros
ONNX_OPERATOR_KERNEL_EX(
    MyMatMul,                          // Op name
    kOnnxDomain,                        // Domain
    1,                                  // Opset version
    kMyExecutionProvider,               // Provider type
    KernelDefBuilder()                  // Kernel definition
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MyMatMulKernel);                    // Kernel class

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    MyConv,
    kOnnxDomain,
    1, 10,                              // Opset range [1, 10]
    kMyExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MyConvKernel);

ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(
    MyAdd,
    kOnnxDomain,
    1,
    kMyExecutionProvider,
    float, double,                      // Two supported types
    KernelDefBuilder()
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                               DataTypeImpl::GetTensorType<double>()}),
    MyAddKernel);
```

### 43.4.5 Kernel Definition Builder API

```cpp
KernelDefBuilder& SetName(const std::string& name);
KernelDefBuilder& SetDomain(const std::string& domain);
KernelDefBuilder& SinceVersion(int version);
KernelDefBuilder& SinceVersion(int start, int end);
KernelDefBuilder& Provider(const std::string& provider);
KernelDefBuilder& TypeConstraint(const std::string& name,
                                  MLDataType type);
KernelDefBuilder& TypeConstraint(const std::string& name,
                                  const std::vector<MLDataType>& types);
KernelDefBuilder& InputMemoryType(OrtMemType type, int input_index);
KernelDefBuilder& OutputMemoryType(OrtMemType type, int output_index);
KernelDefBuilder& ExecQueueType(ExecQueueType queue_type);
KernelDefBuilder& VariadicInput(bool variadic, int min_count = 1);
KernelDefBuilder& VariadicOutput(bool variadic, int min_count = 1);
KernelDefBuilder& ExternalOutputs(int start, int end);
KernelDefBuilder& MayInplace(int input_index, int output_index);
KernelDefBuilder& AllocateOutputsContiguously();
KernelDefBuilder& Alias(int input_index, int output_index);
```

---

## 43.5 CUDA EP Plugin Architecture (plugin-ep-cuda/)

The CUDA EP plugin is the most comprehensive example of a plugin EP. It provides near-parity with the in-tree CUDA EP.

### 43.5.1 Directory Structure

```
plugin-ep-cuda/
├── CMakeLists.txt
├── cmake/
│   ├── FindCUDAToolkit.cmake
│   └── CUDAEPPluginConfig.cmake
├── src/
│   ├── cuda_ep.h                    # CudaEp class declaration
│   ├── cuda_ep.cc                   # CudaEp implementation
│   ├── cuda_ep_factory.h            # CudaEpFactory declaration
│   ├── cuda_ep_factory.cc           # CudaEpFactory implementation
│   ├── cuda_allocator.h             # CudaArenaAllocator
│   ├── cuda_allocator.cc
│   ├── cuda_stream.h                # CudaSyncStream
│   ├── cuda_stream.cc
│   ├── cuda_kernel_adapter.h        # CudaKernelAdapter
│   ├── cuda_kernel_adapter.cc
│   ├── cuda_call.h                  # CUDA error handling macros
│   ├── cuda_call.cc
│   ├── cudnn_call.h                 # cuDNN error handling
│   ├── cublas_call.h                # cuBLAS error handling
│   ├── cuda_fence.h                 # Fence for async execution
│   ├── cuda_fence.cc
│   ├── cuda_data_transfer.h         # Host<->Device data transfer
│   ├── cuda_data_transfer.cc
│   └── kernels/
│       ├── cuda_kernel_registry.h
│       ├── cuda_kernel_registry.cc
│       ├── core/
│       │   ├── matmul.h
│       │   ├── matmul.cc
│       │   ├── conv.h
│       │   ├── conv.cc
│       │   ├── reduction.h
│       │   ├── reduction.cc
│       │   ├── elementwise_ops.h
│       │   ├── elementwise_ops.cc
│       │   ├── activation_ops.h
│       │   ├── activation_ops.cc
│       │   ├── normalization.h
│       │   ├── normalization.cc
│       │   └── ...
│       ├── contrib/
│       │   ├── attention.h
│       │   ├── attention.cc
│       │   ├── bias_gelu.h
│       │   ├── bias_gelu.cc
│       │   └── ...
│       └── cutlass/
│           ├── gemm_kernel.h
│           ├── gemm_kernel.cc
│           └── ...
├── python/
│   ├── __init__.py
│   ├── cuda_ep.py                   # Python interface
│   └── binding.cc                   # pybind11 bindings
├── test/
│   ├── test_cuda_ep.cc
│   ├── test_cuda_kernels.cc
│   └── test_data/
│       └── *.onnx
└── docs/
    └── README.md
```

### 43.5.2 CudaEp Class

```cpp
// src/cuda_ep.h
class CudaEp : public IExecutionProvider {
public:
    explicit CudaEp(const OrtSessionOptions* session_options,
                    const OrtLogger& logger,
                    int device_id = 0,
                    int gpu_mem_limit = 0,
                    int arena_extend_strategy = 0,
                    int cudnn_conv_algo_search = 0,
                    bool do_copy_in_default_stream = true,
                    bool has_user_compute_stream = false,
                    void* user_compute_stream = nullptr,
                    bool enable_cuda_graph = false,
                    bool enable_skip_layout_transform = false,
                    int cudnn_conv1d_pad_to_pad_to_mp = 0,
                    bool prefer_nhwc = false);

    ~CudaEp() override;

    // IExecutionProvider interface
    std::vector<std::unique_ptr<ComputeCapability>>
    GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                  const IKernelLookup& kernel_lookup) const override;

    std::unordered_map<std::string, std::unique_ptr<OpKernel>>
    Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
            const std::vector<const Node*>& non_fused_nodes) override;

    std::vector<AllocatorPtr> CreatePreferredAllocators() const override;
    std::unique_ptr<onnxruntime::Stream>
    CreateStream(const OrtDevice& device) const override;
    std::unique_ptr<IDataTransfer> GetDataTransfer() const override;

    Status OnRunStart() override;
    Status OnRunEnd() override;
    Status Sync() const override;

    // CUDA-specific methods
    cudaStream_t GetComputeStream() const { return compute_stream_; }
    cudnnHandle_t GetCudnnHandle() const { return cudnn_handle_; }
    cublasHandle_t GetCublasHandle() const { return cublas_handle_; }
    int GetDeviceId() const { return device_id_; }
    bool IsCudaGraphEnabled() const { return enable_cuda_graph_; }

private:
    void InitCudaResources();
    void ReleaseCudaResources();

    int device_id_;
    int gpu_mem_limit_;
    int arena_extend_strategy_;
    int cudnn_conv_algo_search_;
    bool do_copy_in_default_stream_;
    bool enable_cuda_graph_;
    bool enable_skip_layout_transform_;
    bool prefer_nhwc_;

    cudaStream_t compute_stream_;
    cudnnHandle_t cudnn_handle_;
    cublasHandle_t cublas_handle_;

    AllocatorPtr cuda_allocator_;
    AllocatorPtr cuda_pinned_allocator_;

    // CUDA graph state
    cudaGraph_t cuda_graph_ = nullptr;
    cudaGraphExec_t cuda_graph_exec_ = nullptr;
    bool cuda_graph_captured_ = false;
};
```

#### CudaEp Implementation

```cpp
// src/cuda_ep.cc
CudaEp::CudaEp(const OrtSessionOptions* session_options,
               const OrtLogger& logger,
               int device_id, ...)
    : IExecutionProvider("CUDA", logger),
      device_id_(device_id),
      gpu_mem_limit_(gpu_mem_limit),
      arena_extend_strategy_(arena_extend_strategy),
      cudnn_conv_algo_search_(cudnn_conv_algo_search),
      do_copy_in_default_stream_(do_copy_in_default_stream),
      enable_cuda_graph_(enable_cuda_graph),
      enable_skip_layout_transform_(enable_skip_layout_transform),
      prefer_nhwc_(prefer_nhwc) {

    // Set CUDA device
    CUDA_CALL(cudaSetDevice(device_id_));

    // Create CUDA stream
    if (!has_user_compute_stream) {
        CUDA_CALL(cudaStreamCreateWithFlags(&compute_stream_,
                                             cudaStreamNonBlocking));
    } else {
        compute_stream_ = static_cast<cudaStream_t>(user_compute_stream);
    }

    InitCudaResources();
}

void CudaEp::InitCudaResources() {
    // Create cuDNN handle
    CUDNN_CALL(cudnnCreate(&cudnn_handle_));
    CUDNN_CALL(cudnnSetStream(cudnn_handle_, compute_stream_));

    // Create cuBLAS handle
    CUBLAS_CALL(cublasCreate(&cublas_handle_));
    CUBLAS_CALL(cublasSetStream(cublas_handle_, compute_stream_));

    // Create allocators
    auto cuda_allocator = std::make_shared<CudaArenaAllocator>(
        device_id_, gpu_mem_limit_, arena_extend_strategy_);

    auto pinned_allocator = std::make_shared<CudaPinnedAllocator>();

    cuda_allocator_ = cuda_allocator;
    cuda_pinned_allocator_ = pinned_allocator;
}

std::vector<AllocatorPtr> CudaEp::CreatePreferredAllocators() const {
    return {cuda_allocator_, cuda_pinned_allocator_};
}

std::unique_ptr<onnxruntime::Stream>
CudaEp::CreateStream(const OrtDevice& device) const {
    return std::make_unique<CudaSyncStream>(compute_stream_, device);
}

Status CudaEp::OnRunStart() {
    if (enable_cuda_graph_ && !cuda_graph_captured_) {
        // Begin CUDA graph capture
        CUDA_CALL(cudaStreamBeginCapture(compute_stream_,
                                          cudaStreamCaptureModeGlobal));
        cuda_graph_captured_ = true;
    }
    return Status::OK();
}

Status CudaEp::OnRunEnd() {
    if (enable_cuda_graph_) {
        if (cuda_graph_captured_ && !cuda_graph_exec_) {
            // End capture and instantiate graph
            CUDA_CALL(cudaStreamEndCapture(compute_stream_, &cuda_graph_));
            CUDA_CALL(cudaGraphInstantiate(&cuda_graph_exec_, cuda_graph_,
                                            nullptr, nullptr, 0));
        } else if (cuda_graph_exec_) {
            // Launch previously captured graph
            CUDA_CALL(cudaGraphLaunch(cuda_graph_exec_, compute_stream_));
        }
    }
    return Status::OK();
}

Status CudaEp::Sync() const {
    CUDA_CALL(cudaStreamSynchronize(compute_stream_));
    return Status::OK();
}

std::vector<std::unique_ptr<ComputeCapability>>
CudaEp::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                       const IKernelLookup& kernel_lookup) const {
    // Build supported node set
    std::unordered_set<const Node*> supported_nodes;

    for (const auto& node : graph_viewer.Nodes()) {
        if (kernel_lookup.LookUp(node.OpType(), node.Domain(),
                                  node.SinceVersion()) != nullptr) {
            supported_nodes.insert(&node);
        }
    }

    // Fuse contiguous supported nodes
    std::vector<std::unique_ptr<ComputeCapability>> result;
    // ... fusion logic ...

    return result;
}

CudaEp::~CudaEp() {
    ReleaseCudaResources();
}

void CudaEp::ReleaseCudaResources() {
    if (cudnn_handle_) {
        cudnnDestroy(cudnn_handle_);
        cudnn_handle_ = nullptr;
    }
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
        cublas_handle_ = nullptr;
    }
    if (compute_stream_) {
        cudaStreamDestroy(compute_stream_);
        compute_stream_ = nullptr;
    }
    if (cuda_graph_exec_) {
        cudaGraphExecDestroy(cuda_graph_exec_);
    }
    if (cuda_graph_) {
        cudaGraphDestroy(cuda_graph_);
    }
}
```

### 43.5.3 CudaEpFactory

```cpp
// src/cuda_ep_factory.h
class CudaEpFactory {
public:
    static const OrtEpFactory* Get() {
        static CudaEpFactory instance;
        return &instance.factory_;
    }

private:
    CudaEpFactory();

    struct Factory : public OrtEpFactory {
        Factory();
    };

    Factory factory_;
    std::mutex mutex_;
    std::vector<std::unique_ptr<CudaEp>> active_eps_;
};

// src/cuda_ep_factory.cc
CudaEpFactory::Factory::Factory() {
    CreateEp = [](OrtEpFactory* self,
                  const OrtSessionOptions* session_options,
                  const OrtLogger* logger,
                  OrtEp** ep) {
        auto* factory = static_cast<CudaEpFactory*>(
            reinterpret_cast<char*>(self) -
            offsetof(CudaEpFactory, factory_));

        std::lock_guard<std::mutex> lock(factory->mutex_);

        // Parse CUDA options from session options
        int device_id = 0;
        OrtSessionOptionsGetConfigEntry(session_options,
            "ep.cuda.device_id", &device_id);

        int gpu_mem_limit = 0;
        OrtSessionOptionsGetConfigEntry(session_options,
            "ep.cuda.gpu_mem_limit", &gpu_mem_limit);

        auto cuda_ep = std::make_unique<CudaEp>(
            session_options, *logger, device_id, gpu_mem_limit);

        *ep = reinterpret_cast<OrtEp*>(cuda_ep.get());
        factory->active_eps_.push_back(std::move(cuda_ep));
    };

    ReleaseEp = [](OrtEpFactory* self, OrtEp* ep) {
        auto* factory = static_cast<CudaEpFactory*>(
            reinterpret_cast<char*>(self) -
            offsetof(CudaEpFactory, factory_));

        std::lock_guard<std::mutex> lock(factory->mutex_);
        factory->active_eps_.erase(
            std::remove_if(factory->active_eps_.begin(),
                          factory->active_eps_.end(),
                          [ep](const std::unique_ptr<CudaEp>& p) {
                              return reinterpret_cast<OrtEp*>(p.get()) == ep;
                          }),
            factory->active_eps_.end());
    };

    GetName = [](OrtEpFactory*) -> const char* {
        return "CUDA";
    };

    GetType = [](OrtEpFactory*) -> const char* {
        return "CudaGPU";
    };

    GetDeviceMemType = [](OrtEpFactory*) -> OrtMemType {
        return OrtMemType::OrtMemType_Default;
    };

    IsCudaGraphEnabled = [](OrtEpFactory* self) -> bool {
        return true;  // CUDA supports CUDA graphs
    };
}
```

### 43.5.4 CudaSyncStream

```cpp
// src/cuda_stream.h
class CudaSyncStream : public onnxruntime::Stream {
public:
    CudaSyncStream(cudaStream_t stream, const OrtDevice& device)
        : Stream(device), stream_(stream) {}

    ~CudaSyncStream() override = default;

    // Stream interface
    Status Flush() override {
        // CUDA streams auto-flush, nothing to do
        return Status::OK();
    }

    Status CleanUpOnRunEnd() override {
        return Status::OK();
    }

    // Submission semaphores
    void Enqueue(int64_t submit_count = 1) override {
        // No-op for CUDA; stream execution is implicit
    }

    // Synchronization
    void BeforeSubmitting(int64_t submit_count) override {
        // Called before submitting work
    }

    // Get the raw CUDA stream
    cudaStream_t GetCudaStream() const { return stream_; }

    // Get the current stream position (used for profiling)
    uint64_t GetStreamPosition() const override {
        return 0;  // CUDA streams don't have explicit positions
    }

private:
    cudaStream_t stream_;
};

// src/cuda_fence.h
class CudaFence : public IFence {
public:
    explicit CudaFence(cudaStream_t stream)
        : stream_(stream) {
        CUDA_CALL(cudaEventCreateWithFlags(&event_,
                                            cudaEventDisableTiming));
    }

    ~CudaFence() override {
        cudaEventDestroy(event_);
    }

    void BeforeUsingAsInput(onnxruntime::Stream& target_stream,
                             int64_t submit_count) override {
        // Record an event on the source stream
        CUDA_CALL(cudaEventRecord(event_, stream_));

        // Make the target stream wait on this event
        auto& cuda_target = static_cast<CudaSyncStream&>(target_stream);
        CUDA_CALL(cudaStreamWaitEvent(cuda_target.GetCudaStream(),
                                       event_, 0));
    }

    void BeforeUsingAsOutput(onnxruntime::Stream& target_stream,
                              int64_t submit_count) override {
        // Same as input for CUDA
        BeforeUsingAsInput(target_stream, submit_count);
    }

    void AfterUsedAsInput(int64_t submit_count) override {}
    void AfterUsedAsOutput(int64_t submit_count) override {
        CUDA_CALL(cudaEventRecord(event_, stream_));
    }

    bool CanRelease() override {
        // Check if the event has completed
        cudaError_t status = cudaEventQuery(event_);
        return status == cudaSuccess;
    }

private:
    cudaStream_t stream_;
    cudaEvent_t event_;
};
```

### 43.5.5 CudaArenaAllocator

```cpp
// src/cuda_allocator.h
class CudaArenaAllocator : public IAllocator {
public:
    CudaArenaAllocator(int device_id,
                       size_t gpu_mem_limit,
                       int arena_extend_strategy)
        : IAllocator(OrtMemType::OrtMemType_Default,
                     OrtDevice(OrtDevice::GPU,
                               OrtDevice::MemType::DEFAULT,
                               device_id)),
          device_id_(device_id),
          gpu_mem_limit_(gpu_mem_limit) {

        CUDA_CALL(cudaSetDevice(device_id_));

        // Get total GPU memory
        cudaDeviceProp prop;
        CUDA_CALL(cudaGetDeviceProperties(&prop, device_id_));
        total_gpu_memory_ = prop.totalGlobalMem;

        // Set memory limit (0 means unlimited)
        if (gpu_mem_limit_ == 0) {
            gpu_mem_limit_ = total_gpu_memory_;
        }

        // Configure arena
        OrtArenaCfg arena_cfg(
            gpu_mem_limit_,
            arena_extend_strategy,
            -1,    // initial_chunk_size_bytes (-1 = auto)
            -1);   // max_dead_bytes_per_chunk (-1 = auto)

        // Create BFC arena allocator
        bfc_allocator_ = std::make_unique<BFCArena>(
            std::make_unique<CudaRawAllocator>(device_id_),
            arena_cfg);
    }

    void* Alloc(size_t size) override {
        return bfc_allocator_->Alloc(size);
    }

    void Free(void* ptr) override {
        bfc_allocator_->Free(ptr);
    }

    size_t Used() const { return bfc_allocator_->Used(); }
    size_t Allocated() const { return bfc_allocator_->Allocated(); }

    // Get BFC stats
    ArenaStats GetStats() const {
        return bfc_allocator_->GetStats();
    }

    // Shrink arena (release unused memory)
    size_t Shrink() {
        return bfc_allocator_->Shrink();
    }

private:
    int device_id_;
    size_t gpu_mem_limit_;
    size_t total_gpu_memory_;
    std::unique_ptr<BFCArena> bfc_allocator_;
};

// Raw CUDA allocator (used by BFC arena)
class CudaRawAllocator {
public:
    explicit CudaRawAllocator(int device_id) : device_id_(device_id) {}

    void* Alloc(size_t size) {
        void* ptr = nullptr;
        CUDA_CALL(cudaSetDevice(device_id_));
        CUDA_CALL(cudaMalloc(&ptr, size));
        return ptr;
    }

    void Free(void* ptr) {
        CUDA_CALL(cudaSetDevice(device_id_));
        CUDA_CALL(cudaFree(ptr));
    }

    size_t GetTotalMemory() const {
        cudaDeviceProp prop;
        CUDA_CALL(cudaGetDeviceProperties(&prop, device_id_));
        return prop.totalGlobalMem;
    }

private:
    int device_id_;
};

// Pinned host memory allocator
class CudaPinnedAllocator : public IAllocator {
public:
    CudaPinnedAllocator()
        : IAllocator(OrtMemType::OrtMemTypeCPU,
                     OrtDevice(OrtDevice::CPU,
                               OrtDevice::MemType::DEFAULT,
                               0)) {}

    void* Alloc(size_t size) override {
        void* ptr = nullptr;
        CUDA_CALL(cudaMallocHost(&ptr, size));
        return ptr;
    }

    void Free(void* ptr) override {
        CUDA_CALL(cudaFreeHost(ptr));
    }
};
```

### 43.5.6 CudaKernelAdapter

The `CudaKernelAdapter` bridges the generic OpKernel interface with CUDA-specific implementations:

```cpp
// src/cuda_kernel_adapter.h
class CudaKernelAdapter : public OpKernel {
public:
    using CreateFunc = std::function<std::unique_ptr<CudaKernelAdapter>(
        const OpKernelInfo&)>;

    CudaKernelAdapter(const OpKernelInfo& info,
                      const CudaEp& ep)
        : OpKernel(info),
          ep_(ep),
          stream_(ep.GetComputeStream()),
          cudnn_handle_(ep.GetCudnnHandle()),
          cublas_handle_(ep.GetCublasHandle()) {}

    virtual Status ComputeInternal(OpKernelContext* context) const = 0;

    Status Compute(OpKernelContext* context) const override {
        auto status = ComputeInternal(context);
        return status;
    }

protected:
    const CudaEp& ep_;
    cudaStream_t stream_;
    cudnnHandle_t cudnn_handle_;
    cublasHandle_t cublas_handle_;
};

// Example: CUDA MatMul kernel adapter
class CudaMatMulAdapter : public CudaKernelAdapter {
public:
    CudaMatMulAdapter(const OpKernelInfo& info, const CudaEp& ep)
        : CudaKernelAdapter(info, ep) {
        transA_ = info.GetAttrOrDefault<int64_t>("transA", 0);
        transB_ = info.GetAttrOrDefault<int64_t>("transB", 0);
        alpha_ = info.GetAttrOrDefault<float>("alpha", 1.0f);
    }

    Status ComputeInternal(OpKernelContext* context) const override {
        const Tensor* A = context->Input<Tensor>(0);
        const Tensor* B = context->Input<Tensor>(1);
        Tensor* Y = context->Output(0, {});

        auto a_shape = A->Shape();
        auto b_shape = B->Shape();

        // Compute output shape
        TensorShape y_shape = ComputeMatMulOutputShape(a_shape, b_shape,
                                                        transA_, transB_);
        Y = context->Output(0, y_shape);

        // Perform matrix multiplication using cuBLAS
        float alpha = alpha_;
        float beta = 0.0f;

        cublasOperation_t opA = transA_ ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t opB = transB_ ? CUBLAS_OP_T : CUBLAS_OP_N;

        int m = static_cast<int>(a_shape[0]);
        int n = static_cast<int>(b_shape[1]);
        int k = static_cast<int>(a_shape[1]);

        CUBLAS_CALL(cublasGemmEx(
            cublas_handle_,
            opA, opB,
            m, n, k,
            &alpha,
            A->Data<float>(), CUDA_R_32F, (transA_ ? m : k),
            B->Data<float>(), CUDA_R_32F, (transB_ ? k : n),
            &beta,
            Y->MutableData<float>(), CUDA_R_32F, m,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        return Status::OK();
    }

private:
    int64_t transA_;
    int64_t transB_;
    float alpha_;
};
```

### 43.5.7 Build System (CMakeLists.txt)

```cmake
cmake_minimum_required(VERSION 3.24)
project(ort-plugin-ep-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find dependencies
find_package(CUDAToolkit REQUIRED)
find_package(onnxruntime REQUIRED)

# Source files
set(CUDA_EP_SOURCES
    src/cuda_ep.cc
    src/cuda_ep_factory.cc
    src/cuda_allocator.cc
    src/cuda_stream.cc
    src/cuda_fence.cc
    src/cuda_data_transfer.cc
    src/cuda_kernel_adapter.cc
    src/kernels/cuda_kernel_registry.cc
    src/kernels/core/matmul.cc
    src/kernels/core/conv.cc
    src/kernels/core/reduction.cc
    src/kernels/core/elementwise_ops.cc
    src/kernels/core/activation_ops.cc
    src/kernels/core/normalization.cc
    src/kernels/contrib/attention.cc
    src/kernels/contrib/bias_gelu.cc
)

# Shared library
add_library(ort-plugin-ep-cuda SHARED ${CUDA_EP_SOURCES})

target_include_directories(ort-plugin-ep-cuda
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${onnxruntime_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(ort-plugin-ep-cuda
    PRIVATE
        CUDA::cudart
        CUDA::cublas
        CUDA::cudnn
        ${onnxruntime_LIBRARIES}
)

# Strip symbols for smaller binary
set_target_properties(ort-plugin-ep-cuda PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Installation
install(TARGETS ort-plugin-ep-cuda
    LIBRARY DESTINATION lib
)

# Python bindings (optional)
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(_cuda_ep_plugin python/binding.cc)
    target_link_libraries(_cuda_ep_plugin PRIVATE ort-plugin-ep-cuda)
    install(TARGETS _cuda_ep_plugin LIBRARY DESTINATION python)
endif()
```

### 43.5.8 Parity with In-Tree CUDA EP

The plugin CUDA EP aims for near-complete parity with the in-tree CUDA EP:

| Feature | In-Tree | Plugin | Notes |
|---------|---------|--------|-------|
| Core ops (MatMul, Conv, etc.) | Full support | Full support | Same kernel implementations |
| Contrib ops | Full support | Partial | Growing coverage |
| cuDNN integration | Yes | Yes | Same cuDNN calls |
| cuBLAS integration | Yes | Yes | Same cuBLAS calls |
| CUDA Graphs | Yes | Yes | Supported |
| Stream management | Yes | Yes | Same stream model |
| Memory arena (BFC) | Yes | Yes | Same arena implementation |
| Multi-GPU | Yes | Yes | Device selection via config |
| Mixed precision (FP16/BF16) | Yes | Yes | Data type support |
| INT8 quantization | Yes | Partial | QLinear ops supported |
| TensorRT integration | Separate EP | No | TRT is its own EP |
| Flash Attention | Yes | Partial | Contrib kernels |
|_nhwc layout | Yes | Yes | Layout transform support |
| Weight pre-packing | Yes | Yes | Conv weight packing |
| Warmup / algorithm search | Yes | Yes | cuDNN algorithm search |

---

## 43.6 WebGPU EP Plugin Architecture (plugin-ep-webgpu/)

### 43.6.1 Overview

The WebGPU EP plugin enables ONNX Runtime inference on GPUs via the WebGPU API, supporting both browser and native (Dawn/wgpu-native) environments.

```
plugin-ep-webgpu/
├── CMakeLists.txt
├── src/
│   ├── webgpu_ep.h
│   ├── webgpu_ep.cc
│   ├── webgpu_ep_factory.h
│   ├── webgpu_ep_factory.cc
│   ├── webgpu_allocator.h
│   ├── webgpu_allocator.cc
│   ├── webgpu_context.h
│   ├── webgpu_context.cc
│   ├── webgpu_kernel_adapter.h
│   └── kernels/
│       ├── webgpu_kernel_registry.cc
│       ├── core/
│       │   ├── matmul.cc
│       │   ├── elementwise.cc
│       │   ├── reduction.cc
│       │   └── ...
│       └── shaders/
│           ├── matmul.wgsl
│           ├── elementwise.wgsl
│           ├── reduction.wgsl
│           └── ...
├── test/
│   └── test_webgpu_ep.cc
└── docs/
```

### 43.6.2 WebGPU EP Class

```cpp
class WebGpuEp : public IExecutionProvider {
public:
    WebGpuEp(const OrtSessionOptions* session_options,
             const OrtLogger& logger,
             wgpu::Instance instance,
             wgpu::Adapter adapter,
             wgpu::Device device);

    ~WebGpuEp() override;

    std::vector<std::unique_ptr<ComputeCapability>>
    GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                  const IKernelLookup& kernel_lookup) const override;

    std::unordered_map<std::string, std::unique_ptr<OpKernel>>
    Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
            const std::vector<const Node*>& non_fused_nodes) override;

    std::vector<AllocatorPtr> CreatePreferredAllocators() const override;
    std::unique_ptr<IDataTransfer> GetDataTransfer() const override;

    // WebGPU-specific
    const wgpu::Device& GetDevice() const { return device_; }
    const WebGpuContext& GetContext() const { return context_; }

private:
    wgpu::Instance instance_;
    wgpu::Adapter adapter_;
    wgpu::Device device_;
    WebGpuContext context_;
    AllocatorPtr gpu_allocator_;
};
```

### 43.6.3 WebGPU Kernel with WGSL Shaders

```cpp
// Example: WebGPU elementwise Add kernel
class WebGpuAddKernel : public OpKernel {
public:
    explicit WebGpuAddKernel(const OpKernelInfo& info)
        : OpKernel(info) {}

    Status Compute(OpKernelContext* context) const override {
        const Tensor* A = context->Input<Tensor>(0);
        const Tensor* B = context->Input<Tensor>(1);
        Tensor* Y = context->Output(0, A->Shape());

        auto& ep = static_cast<const WebGpuEp&>(Info().GetExecutionProvider());
        auto& device = ep.GetDevice();
        auto& ctx = ep.GetContext();

        size_t element_count = A->Shape().Size();

        // Create compute pipeline
        auto pipeline = ctx.GetOrCreatePipeline("add", kAddShaderCode);

        // Create bind group
        wgpu::BindGroupEntry entries[3] = {};
        entries[0].binding = 0;
        entries[0].buffer = ctx.GetBuffer(A);
        entries[1].binding = 1;
        entries[1].buffer = ctx.GetBuffer(B);
        entries[2].binding = 2;
        entries[2].buffer = ctx.GetBuffer(Y);

        wgpu::BindGroupDescriptor bg_desc = {};
        bg_desc.layout = pipeline.GetBindGroupLayout(0);
        bg_desc.entryCount = 3;
        bg_desc.entries = entries;

        auto bind_group = device.CreateBindGroup(&bg_desc);

        // Dispatch compute
        auto encoder = device.CreateCommandEncoder();
        auto pass = encoder.BeginComputePass();

        uint32_t workgroup_count =
            (element_count + kWorkgroupSize - 1) / kWorkgroupSize;

        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bind_group);
        pass.DispatchWorkgroups(workgroup_count);
        pass.End();

        auto cmd_buffer = encoder.Finish();
        auto queue = device.GetQueue();
        queue.Submit(1, &cmd_buffer);

        return Status::OK();
    }

private:
    static constexpr uint32_t kWorkgroupSize = 64;
    static constexpr const char* kAddShaderCode = R"(
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> y: array<f32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let idx = gid.x;
            if (idx < arrayLength(&a)) {
                y[idx] = a[idx] + b[idx];
            }
        }
    )";
};
```

---

## 43.7 Plugin Loading Mechanism

### 43.7.1 Discovery and Loading Flow

```
1. User calls ort.load("plugin_ep_cuda", library_path)
   ↓
2. PlatformLoadSharedLibrary(library_path) → dlopen/LoadLibrary
   ↓
3. GetEpApiVersion() → verify API version compatibility
   ↓
4. GetEpFactory() → obtain OrtEpFactory pointer
   ↓
5. Register factory with OrtEnv
   ↓
6. When session is created with EP options:
   factory->CreateEp() → creates EP instance
```

### 43.7.2 Loading API

```python
# Python API for loading plugin EPs
import onnxruntime as ort

# Load plugin from shared library
ort.load("cuda_plugin", "/path/to/libort-plugin-ep-cuda.so")

# Or let ONNX Runtime discover it from PATH
ort.load("cuda_plugin")

# Use the plugin EP
options = ort.SessionOptions()
options.add_executable_provider("CUDA", {"device_id": 0})
session = ort.InferenceSession("model.onnx", options)
```

```c
// C API
OrtStatus* OrtLoadPluginEp(
    const OrtApi* api,
    const char* ep_name,
    const char* library_path,
    const OrtEpFactory** out_factory);
```

### 43.7.3 Platform-Specific Loading

```cpp
// Linux/macOS
void* PlatformLoadSharedLibrary(const std::string& path) {
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        throw std::runtime_error(
            std::string("Failed to load plugin: ") + dlerror());
    }
    return handle;
}

void* PlatformGetSymbol(void* handle, const char* symbol) {
    return dlsym(handle, symbol);
}

void PlatformUnloadSharedLibrary(void* handle) {
    dlclose(handle);
}

// Windows
void* PlatformLoadSharedLibrary(const std::string& path) {
    HMODULE handle = LoadLibraryExA(path.c_str(), nullptr,
                                     LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!handle) {
        throw std::runtime_error(
            "Failed to load plugin: " + std::to_string(GetLastError()));
    }
    return handle;
}

void* PlatformGetSymbol(void* handle, const char* symbol) {
    return GetProcAddress(static_cast<HMODULE>(handle), symbol);
}

void PlatformUnloadSharedLibrary(void* handle) {
    FreeLibrary(static_cast<HMODULE>(handle));
}
```

### 43.7.4 Version Checking

```cpp
Status LoadAndRegisterPluginEp(const std::string& library_path) {
    void* handle = PlatformLoadSharedLibrary(library_path);

    // Get API version
    using GetEpApiVersionFn = void(*)(int*);
    auto get_version = reinterpret_cast<GetEpApiVersionFn>(
        PlatformGetSymbol(handle, "GetEpApiVersion"));

    if (!get_version) {
        return ORT_MAKE_STATUS(FAIL, "Plugin missing GetEpApiVersion symbol");
    }

    int version = 0;
    get_version(&version);

    if (version != ORT_EP_API_VERSION) {
        return ORT_MAKE_STATUS(FAIL,
            "Plugin API version mismatch. Expected: ",
            ORT_EP_API_VERSION, ", Got: ", version);
    }

    // Get factory
    using GetEpFactoryFn = const OrtEpFactory*(*)();
    auto get_factory = reinterpret_cast<GetEpFactoryFn>(
        PlatformGetSymbol(handle, "GetEpFactory"));

    if (!get_factory) {
        return ORT_MAKE_STATUS(FAIL, "Plugin missing GetEpFactory symbol");
    }

    const OrtEpFactory* factory = get_factory();

    // Register with environment
    env->RegisterPluginEpFactory(factory, handle);

    return Status::OK();
}
```

---

## 43.8 C API Boundary

### 43.8.1 Plugin C API Contract

Plugin EPs must only interact with ONNX Runtime through the stable C API. This ensures binary compatibility across ORT versions.

```c
// The C API functions available to plugins
typedef struct OrtPluginApi {
    // Memory management
    OrtStatus* (*AllocatorAlloc)(OrtAllocator* allocator, size_t size, void** out);
    OrtStatus* (*AllocatorFree)(OrtAllocator* allocator, void* ptr);

    // Tensor access
    OrtStatus* (*GetTensorTypeAndShape)(const OrtValue* value,
                                         OrtTensorTypeAndShapeInfo** out);
    OrtStatus* (*GetTensorMutableData)(OrtValue* value, void** out);
    OrtStatus* (*GetTensorData)(const OrtValue* value, const void** out);

    // Kernel context
    OrtStatus* (*KernelContext_GetInput)(OrtKernelContext* context,
                                          size_t index, const OrtValue** out);
    OrtStatus* (*KernelContext_GetOutput)(OrtKernelContext* context,
                                           size_t index,
                                           const int64_t* dims,
                                           size_t dim_count,
                                           OrtValue** out);

    // Kernel info (for attribute access)
    OrtStatus* (*KernelInfoGetAttribute_float)(const OrtKernelInfo* info,
                                                const char* name,
                                                float* out);
    OrtStatus* (*KernelInfoGetAttribute_int64)(const OrtKernelInfo* info,
                                                const char* name,
                                                int64_t* out);
    OrtStatus* (*KernelInfoGetAttribute_string)(const OrtKernelInfo* info,
                                                 const char* name,
                                                 char* out,
                                                 size_t* size);

    // Logging
    OrtStatus* (*LogMessage)(const OrtLogger* logger,
                              OrtLoggingLevel level,
                              const char* message,
                              const char* file,
                              int line);

    // Session options
    OrtStatus* (*SessionOptionsGetConfigEntry)(const OrtSessionOptions* options,
                                                const char* key,
                                                char* value,
                                                size_t* size);

    // Graph access (for GetCapability and Compile)
    OrtStatus* (*GraphViewerGetNodes)(const OrtGraphViewer* viewer,
                                       size_t* count,
                                       OrtNodeHandle** nodes);
    OrtStatus* (*GraphViewerGetNodeInfo)(const OrtGraphViewer* viewer,
                                          size_t index,
                                          OrtNodeInfo* info);
} OrtPluginApi;
```

### 43.8.2 C API Usage Patterns

```cpp
// Pattern: Getting tensor data inside a kernel
Status MyKernel::Compute(OpKernelContext* context) const {
    const OrtValue* input_value = nullptr;
    OrtStatus* status = api_->KernelContext_GetInput(
        reinterpret_cast<OrtKernelContext*>(context),
        0, &input_value);

    if (status != nullptr) {
        return Status(static_cast<common::ErrorCode>(status->code),
                      status->msg);
    }

    const void* data = nullptr;
    status = api_->GetTensorData(input_value, &data);

    // Create output tensor
    int64_t dims[] = {batch_size, hidden_dim};
    OrtValue* output_value = nullptr;
    status = api_->KernelContext_GetOutput(
        reinterpret_cast<OrtKernelContext*>(context),
        0, dims, 2, &output_value);

    void* output_data = nullptr;
    status = api_->GetTensorMutableData(output_value, &output_data);

    return Status::OK();
}
```

---

## 43.9 Memory Management in Plugins

### 43.9.1 Memory Ownership Model

```
┌───────────────────────────────────────────────────┐
│                   ONNX Runtime Core               │
│                                                   │
│  Owns: Graph, Session, Thread Pools, etc.         │
│  Provides: Allocator interfaces to plugins        │
├───────────────────────────────────────────────────┤
│                   Plugin EP                       │
│                                                   │
│  Owns: EP-specific allocators, GPU resources      │
│  Allocates: Tensors via own allocators            │
│  Manages: Memory arena, caching, pool             │
└───────────────────────────────────────────────────┘
```

### 43.9.2 Allocator Registration

```cpp
std::vector<AllocatorPtr> MyEp::CreatePreferredAllocators() const {
    // Create device allocator (arena-based)
    auto device_allocator = std::make_shared<MyArenaAllocator>(
        device_id_, max_memory_, arena_config_);

    // Create pinned/host allocator (if needed)
    auto host_allocator = std::make_shared<MyPinnedAllocator>();

    return {device_allocator, host_allocator};
}
```

### 43.9.3 Memory Lifecycle

```cpp
// During session creation:
// 1. EP creates preferred allocators
auto allocators = ep->CreatePreferredAllocators();

// 2. Each allocator is registered with the session
for (auto& allocator : allocators) {
    session->AddAllocator(allocator);
}

// During inference:
// 3. Tensors are allocated via the appropriate allocator
auto allocator = session->GetAllocator(device_id, mem_type);
void* buffer = allocator->Alloc(size);

// 4. After inference, tensors are freed
allocator->Free(buffer);

// During session destruction:
// 5. Allocators are destroyed, releasing all memory
```

### 43.9.4 Custom Memory Pools

```cpp
class PluginMemoryPool {
public:
    explicit PluginMemoryPool(size_t pool_size, int device_id)
        : device_id_(device_id) {
        // Pre-allocate a large buffer
        CUDA_CALL(cudaSetDevice(device_id_));
        CUDA_CALL(cudaMalloc(&pool_base_, pool_size));
        pool_size_ = pool_size;
        free_offset_ = 0;
    }

    void* Allocate(size_t size, size_t alignment = 256) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Align offset
        size_t aligned_offset = (free_offset_ + alignment - 1) & ~(alignment - 1);

        if (aligned_offset + size > pool_size_) {
            // Pool exhausted, fall back to individual allocation
            void* ptr;
            CUDA_CALL(cudaMalloc(&ptr, size));
            overflow_allocations_[ptr] = size;
            return ptr;
        }

        void* ptr = static_cast<char*>(pool_base_) + aligned_offset;
        free_offset_ = aligned_offset + size;
        allocations_[ptr] = {aligned_offset, size};
        return ptr;
    }

    void Deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = allocations_.find(ptr);
        if (it != allocations_.end()) {
            allocations_.erase(it);
            if (allocations_.empty()) {
                free_offset_ = 0;  // Reset pool
            }
            return;
        }

        // Check overflow allocations
        auto oit = overflow_allocations_.find(ptr);
        if (oit != overflow_allocations_.end()) {
            CUDA_CALL(cudaFree(ptr));
            overflow_allocations_.erase(oit);
        }
    }

    ~PluginMemoryPool() {
        CUDA_CALL(cudaSetDevice(device_id_));
        CUDA_CALL(cudaFree(pool_base_));
        for (auto& [ptr, size] : overflow_allocations_) {
            cudaFree(ptr);
        }
    }

private:
    int device_id_;
    void* pool_base_ = nullptr;
    size_t pool_size_ = 0;
    size_t free_offset_ = 0;
    std::mutex mutex_;
    std::unordered_map<void*, std::pair<size_t, size_t>> allocations_;
    std::unordered_map<void*, size_t> overflow_allocations_;
};
```

---

## 43.10 Stream Handling in Plugins

### 43.10.1 Stream Architecture

```
┌──────────────────────────────────────┐
│         ORT Stream Interface         │
│  onnxruntime::Stream                 │
│  - Flush()                           │
│  - CleanUpOnRunEnd()                 │
│  - Enqueue(submit_count)             │
│  - BeforeSubmitting(submit_count)    │
├──────────────────────────────────────┤
│         Plugin Stream Impl           │
│  - Wraps hardware-specific stream    │
│  - Handles synchronization           │
│  - Manages stream-ordered operations │
└──────────────────────────────────────┘
```

### 43.10.2 Stream-Aware Execution

```cpp
// EP creates streams
std::unique_ptr<onnxruntime::Stream>
MyEp::CreateStream(const OrtDevice& device) const {
    return std::make_unique<MyStream>(hardware_queue_, device);
}

// Kernel uses stream for async execution
Status MyAsyncKernel::ComputeAsync(OpKernelContext* context,
                                    Stream* stream) const {
    auto* my_stream = static_cast<MyStream*>(stream);

    // Submit work to the stream
    my_stream->SubmitKernel(kernel_args_);

    return Status::OK();
}

// ORT handles synchronization between streams via fences
```

### 43.10.3 Cross-EP Stream Synchronization

```cpp
// When data flows from EP A to EP B:
// 1. EP A's fence records completion
void MyFence::AfterUsedAsOutput(int64_t submit_count) {
    // Record event on source stream
    cudaEventRecord(event_, source_stream_);
}

// 2. EP B's fence waits for EP A's event
void MyFence::BeforeUsingAsInput(Stream& target_stream,
                                  int64_t submit_count) {
    // Make target stream wait on the recorded event
    auto& target_cuda_stream = static_cast<CudaSyncStream&>(target_stream);
    cudaStreamWaitEvent(target_cuda_stream.GetCudaStream(), event_, 0);
}
```

---

## 43.11 Custom Allocator Integration

### 43.11.1 IAllocator Interface

```cpp
class IAllocator {
public:
    IAllocator(OrtMemType mem_type, OrtDevice device)
        : mem_type_(mem_type), device_(device) {}

    virtual ~IAllocator() = default;

    virtual void* Alloc(size_t size) = 0;
    virtual void Free(void* ptr) = 0;

    // Optional: allocate with alignment
    virtual void* AllocAligned(size_t size, size_t alignment) {
        // Default: ignore alignment
        return Alloc(size);
    }

    // Optional: get allocation statistics
    virtual size_t Used() const { return 0; }
    virtual size_t Allocated() const { return 0; }
    virtual size_t Reserved() const { return 0; }

    // Device information
    OrtMemType MemType() const { return mem_type_; }
    const OrtDevice& GetDevice() const { return device_; }
    int DeviceId() const { return device_.Id(); }

    // Check if this allocator supports the given device
    virtual bool SupportsDevice(const OrtDevice& device) const {
        return device_ == device;
    }

    // Optional: shrink (release unused memory)
    virtual size_t Shrink() { return 0; }

private:
    OrtMemType mem_type_;
    OrtDevice device_;
};
```

### 43.11.2 Registering Custom Allocators

```cpp
// Method 1: Through CreatePreferredAllocators
std::vector<AllocatorPtr> MyEp::CreatePreferredAllocators() const {
    auto alloc = std::make_shared<MyCustomAllocator>(device_id_);
    return {alloc};
}

// Method 2: Through SessionOptions (user-provided)
 Ort::SessionOptions session_options;
 session_options.AddConfigEntry("ep.my_ep.custom_allocator_path",
                                 "/path/to/custom_allocator.so");

// Method 3: Through C API
OrtAllocator* custom_alloc = CreateMyCustomAllocator();
OrtSessionOptionsAppendExecutionProvider_CustomAllocator(
    session_options, custom_alloc);
```

### 43.11.3 Allocator Device Mapping

```cpp
// ORT maps allocators to (device_id, mem_type) pairs
// During tensor allocation:
AllocatorPtr Session::GetAllocator(int device_id, OrtMemType mem_type) const {
    // First, look for EP-specific allocator
    for (auto& [key, allocator] : allocators_) {
        if (allocator->DeviceId() == device_id &&
            allocator->MemType() == mem_type) {
            return allocator;
        }
    }

    // Fall back to default allocator
    return default_allocator_;
}
```

### 43.11.4 Data Transfer Between Allocators

```cpp
// IDataTransfer handles copying between different allocators
class IDataTransfer {
public:
    virtual ~IDataTransfer() = default;

    // Copy tensor from source to destination
    virtual Status CopyTensor(const Tensor& src, Tensor& dst) const = 0;

    // Async copy with stream
    virtual Status CopyTensorAsync(const Tensor& src, Tensor& dst,
                                    Stream& stream) const {
        return CopyTensor(src, dst);
    }
};

// Example: CUDA data transfer
class CudaDataTransfer : public IDataTransfer {
public:
    Status CopyTensor(const Tensor& src, Tensor& dst) const override {
        size_t bytes = src.SizeInBytes();

        if (src.Location().device.Type() == OrtDevice::CPU &&
            dst.Location().device.Type() == OrtDevice::GPU) {
            // Host to device
            CUDA_CALL(cudaMemcpy(dst.MutableDataRaw(),
                                  src.DataRaw(),
                                  bytes,
                                  cudaMemcpyHostToDevice));
        } else if (src.Location().device.Type() == OrtDevice::GPU &&
                   dst.Location().device.Type() == OrtDevice::CPU) {
            // Device to host
            CUDA_CALL(cudaMemcpy(dst.MutableDataRaw(),
                                  src.DataRaw(),
                                  bytes,
                                  cudaMemcpyDeviceToHost));
        } else if (src.Location().device.Type() == OrtDevice::GPU &&
                   dst.Location().device.Type() == OrtDevice::GPU) {
            // Device to device
            CUDA_CALL(cudaMemcpy(dst.MutableDataRaw(),
                                  src.DataRaw(),
                                  bytes,
                                  cudaMemcpyDeviceToDevice));
        }

        return Status::OK();
    }

    Status CopyTensorAsync(const Tensor& src, Tensor& dst,
                            Stream& stream) const override {
        auto& cuda_stream = static_cast<CudaSyncStream&>(stream);
        size_t bytes = src.SizeInBytes();

        cudaMemcpyKind kind;
        if (src.Location().device.Type() == OrtDevice::CPU &&
            dst.Location().device.Type() == OrtDevice::GPU) {
            kind = cudaMemcpyHostToDevice;
        } else if (src.Location().device.Type() == OrtDevice::GPU &&
                   dst.Location().device.Type() == OrtDevice::CPU) {
            kind = cudaMemcpyDeviceToHost;
        } else {
            kind = cudaMemcpyDeviceToDevice;
        }

        CUDA_CALL(cudaMemcpyAsync(dst.MutableDataRaw(),
                                   src.DataRaw(),
                                   bytes,
                                   kind,
                                   cuda_stream.GetCudaStream()));
        return Status::OK();
    }
};
```

---

## 43.12 Error Handling in Plugins

### 43.12.1 Error Handling Strategy

```cpp
// CUDA error checking macro
#define CUDA_CALL(expr)                                               \
    do {                                                              \
        cudaError_t status = (expr);                                  \
        if (status != cudaSuccess) {                                  \
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,                 \
                "CUDA error: ", cudaGetErrorString(status),           \
                " at ", __FILE__, ":", __LINE__);                     \
        }                                                             \
    } while (0)

#define CUDNN_CALL(expr)                                              \
    do {                                                              \
        cudnnStatus_t status = (expr);                                \
        if (status != CUDNN_STATUS_SUCCESS) {                         \
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,                 \
                "cuDNN error: ", cudnnGetErrorString(status),         \
                " at ", __FILE__, ":", __LINE__);                     \
        }                                                             \
    } while (0)

#define CUBLAS_CALL(expr)                                             \
    do {                                                              \
        cublasStatus_t status = (expr);                               \
        if (status != CUBLAS_STATUS_SUCCESS) {                        \
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,                 \
                "cuBLAS error at ", __FILE__, ":", __LINE__);         \
        }                                                             \
    } while (0)
```

### 43.12.2 Graceful Degradation

```cpp
std::vector<std::unique_ptr<ComputeCapability>>
MyEp::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                     const IKernelLookup& kernel_lookup) const {
    std::vector<std::unique_ptr<ComputeCapability>> result;

    for (const auto& node : graph_viewer.Nodes()) {
        try {
            const KernelCreateInfo* info =
                kernel_lookup.LookUp(node.OpType(), node.Domain(),
                                      node.SinceVersion());

            if (info != nullptr) {
                // Try to check if this specific node configuration is supported
                if (IsNodeSupported(node, *info)) {
                    auto cc = std::make_unique<ComputeCapability>(
                        std::make_unique<IndexedSubGraph>());
                    cc->sub_graph->nodes.push_back(node.Index());
                    result.push_back(std::move(cc));
                }
            }
        } catch (const std::exception& e) {
            // Log error but continue with other nodes
            LOGS(logger_, WARNING) << "Error checking capability for node "
                                    << node.Name() << ": " << e.what();
        }
    }

    return result;
}
```

---

## 43.13 Testing Plugin EPs

### 43.13.1 Unit Testing Framework

```cpp
// test/test_cuda_ep.cc
#include "gtest/gtest.h"
#include "cuda_ep.h"
#include "cuda_ep_factory.h"

class CudaEpTest : public ::testing::Test {
protected:
    void SetUp() override {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions opts;
        logger_ = Ort::Logger::GetLogger("test");

        ep_ = std::make_unique<CudaEp>(
            nullptr, *logger_, 0,  // device_id
            0,    // gpu_mem_limit (unlimited)
            0,    // arena_extend_strategy
            0,    // cudnn_conv_algo_search
            true, // do_copy_in_default_stream
            false, 0,  // no user stream
            false, // no cuda graph
            false  // no nhwc
        );
    }

    void TearDown() override {
        ep_.reset();
    }

    std::unique_ptr<CudaEp> ep_;
    const OrtLogger* logger_;
};

TEST_F(CudaEpTest, CreateAllocator) {
    auto allocators = ep_->CreatePreferredAllocators();
    ASSERT_FALSE(allocators.empty());

    auto& alloc = allocators[0];
    void* ptr = alloc->Alloc(1024);
    ASSERT_NE(ptr, nullptr);
    alloc->Free(ptr);
}

TEST_F(CudaEpTest, CreateStream) {
    OrtDevice device(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0);
    auto stream = ep_->CreateStream(device);
    ASSERT_NE(stream, nullptr);

    auto status = stream->Flush();
    ASSERT_TRUE(status.IsOK());
}

TEST_F(CudaEpTest, SyncStream) {
    auto status = ep_->Sync();
    ASSERT_TRUE(status.IsOK());
}
```

### 43.13.2 Integration Testing

```python
# test/test_cuda_ep.py
import onnxruntime as ort
import numpy as np
import onnx
from onnx import helper, TensorProto

def test_cuda_plugin_basic():
    """Test basic inference with CUDA plugin EP."""
    # Create simple ONNX model
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3])

    add_node = helper.make_node('Add', ['X', 'X'], ['Y'])
    graph = helper.make_graph([add_node], 'test_graph', [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7

    # Save model
    onnx.save(model, '/tmp/test_model.onnx')

    # Load plugin
    ort.load("cuda_plugin", "/path/to/libort-plugin-ep-cuda.so")

    # Create session with CUDA EP
    opts = ort.SessionOptions()
    opts.add_executable_provider("CUDA", {"device_id": 0})
    session = ort.InferenceSession('/tmp/test_model.onnx', opts)

    # Run inference
    input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    output = session.run(None, {'X': input_data})

    expected = input_data + input_data
    np.testing.assert_allclose(output[0], expected, rtol=1e-5)

def test_cuda_plugin_matmul():
    """Test MatMul with CUDA plugin EP."""
    # Create MatMul model
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 3])
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [3, 4])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 4])

    matmul_node = helper.make_node('MatMul', ['A', 'B'], ['Y'])
    graph = helper.make_graph([matmul_node], 'test_matmul', [A], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    onnx.save(model, '/tmp/test_matmul.onnx')

    opts = ort.SessionOptions()
    opts.add_executable_provider("CUDA", {"device_id": 0})
    session = ort.InferenceSession('/tmp/test_matmul.onnx', opts)

    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(3, 4).astype(np.float32)
    output = session.run(None, {'A': a, 'B': b})

    expected = np.matmul(a, b)
    np.testing.assert_allclose(output[0], expected, rtol=1e-4, atol=1e-4)
```

---

## 43.14 Summary

| Component | Interface | Purpose |
|-----------|-----------|---------|
| OrtEpFactory | C API | Creates/destroys EP instances |
| OrtEp (IExecutionProvider) | C++ | Core EP functionality |
| OpKernel | C++ | Individual operator implementation |
| IAllocator | C++ | Memory management |
| Stream | C++ | Async execution |
| IDataTransfer | C++ | Cross-device data movement |
| IFence | C++ | Cross-stream synchronization |
| KernelRegistry | C++ | Kernel lookup and registration |

The plugin EP architecture provides a clean separation between ONNX Runtime core and hardware-specific execution providers, enabling independent development, testing, and distribution of EP implementations.
