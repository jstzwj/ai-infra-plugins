# ONNX Runtime Reference - Chapter 3: C++ API Reference

Complete reference for the ONNX Runtime C++ API (`onnxruntime_cxx_api.h`, `onnxruntime_cxx_inline.h`). Header-only RAII wrappers around the C API.

---

## 3.1 Namespace and Error Handling

### 3.1.1 Namespace

All C++ APIs are in the `Ort` namespace.

### 3.1.2 Exception Class

```cpp
namespace Ort {
struct Exception : std::exception {
    Exception(const std::string& string, OrtErrorCode code);
    Exception(std::string&& string, OrtErrorCode code);

    OrtErrorCode GetOrtErrorCode() const;
    const char* what() const noexcept override;
};
}  // namespace Ort
```

- When `ORT_NO_EXCEPTIONS` is defined, errors call `abort()` instead of throwing
- Custom error handler: `#define ORT_CXX_API_THROW(string, code)`

### 3.1.3 API Access

```cpp
// Get reference to the OrtApi
inline const OrtApi& GetApi() noexcept;

// Get ORT version string
std::string GetVersionString();

// Get ORT build info
std::string GetBuildInfoString();

// Manual API initialization
#ifdef ORT_API_MANUAL_INIT
void InitApi() noexcept;
void InitApi(const OrtApi* api) noexcept;
#endif
```

---

## 3.2 Base Classes

### 3.2.1 Base<T> - RAII Wrapper Base

All Ort objects use move-only RAII wrappers:

```cpp
template <typename T>
class Base {
protected:
    T* p_;  // Underlying C object pointer

    Base() = default;
    Base(T* p) : p_(p) {}

    // Move-only
    Base(Base&& other) noexcept;
    Base& operator=(Base&& other) noexcept;
    Base(const Base&) = delete;
    Base& operator=(const Base&) = delete;

    ~Base() { OrtRelease(p_); }  // Automatic cleanup
};
```

### 3.2.2 ConstXXXX Types

Non-owning const wrappers that can be copied:
```cpp
ConstMemoryInfo    // const reference to MemoryInfo
ConstSessionOptions // const reference to SessionOptions
```

### 3.2.3 UnownedXXXX Types

Non-owning mutable wrappers:
```cpp
UnownedMemoryInfo  // mutable reference, doesn't own
UnownedValue       // mutable reference to Value
```

---

## 3.3 Environment (Ort::Env)

```cpp
namespace Ort {

class Env : public Base<OrtEnv> {
public:
    Env() = default;
    explicit Env(OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING,
                 _In_ const char* logid = "");

    Env(OrtLoggingLevel logging_level, const char* logid,
        OrtLoggingFunction logging_function, void* logger_param);

    Env(const OrtThreadingOptions* tp_options,
        OrtLoggingLevel logging_level, const char* logid,
        OrtLoggingFunction logging_function, void* logger_param);

    Env& operator=(Env&& other) noexcept;

    // Telemetry
    Env& EnableTelemetryEvents();
    Env& DisableTelemetryEvents();

    // Language projection
    Env& SetLanguageProjection(OrtLanguageProjection projection);

    // Global thread pools
    Env& SetGlobalThreadPools(const OrtThreadingOptions* tp_options);

    // Create and register allocator
    Env& CreateAndRegisterAllocator(const OrtMemoryInfo* mem_info,
                                     const OrtArenaCfg* arena_cfg);
};

}  // namespace Ort
```

**Usage:**
```cpp
// Simple creation
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "my_app");

// With custom logger
Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "my_app",
             my_logging_func, my_param);

// With global thread pools
Ort::ThreadingOptions tp;
tp.SetGlobalIntraOpNumThreads(4);
tp.SetGlobalInterOpNumThreads(1);
Ort::Env env(&tp, ORT_LOGGING_LEVEL_WARNING, "my_app",
             nullptr, nullptr);
```

---

## 3.4 Session (Ort::Session)

```cpp
namespace Ort {

class Session : public Base<OrtSession> {
public:
    Session() = default;

    // Create from file
    Session(const Env& env, const ORTCHAR_T* model_path,
            const SessionOptions& options);

    // Create from memory
    Session(const Env& env, const void* model_data, size_t model_data_length,
            const SessionOptions& options);

    // Create with pre-packed weights
    Session(const Env& env, const ORTCHAR_T* model_path,
            const SessionOptions& options,
            PrepackedWeightsContainer& prepacked_weights);

    Session& operator=(Session&& other) noexcept;

    // Run inference
    std::vector<Value> Run(const RunOptions& run_options,
                           const char* const* input_names,
                           const Value* input_values, size_t input_count,
                           const char* const* output_names,
                           size_t output_count);

    // Convenience: run with vectors
    std::vector<Value> Run(const RunOptions& run_options,
                           const std::vector<const char*>& input_names,
                           const std::vector<Value>& input_values,
                           const std::vector<const char*>& output_names);

    // Run with IO binding
    void Run(const RunOptions& run_options, IoBinding& io_binding);

    // Input/Output info
    size_t GetInputCount() const;
    size_t GetOutputCount() const;
    size_t GetOverridableInitializerCount() const;

    char* GetInputName(size_t index, OrtAllocator* allocator) const;
    char* GetOutputName(size_t index, OrtAllocator* allocator) const;
    char* GetOverridableInitializerName(size_t index, OrtAllocator* allocator) const;

    TypeInfo GetInputTypeInfo(size_t index) const;
    TypeInfo GetOutputTypeInfo(size_t index) const;
    TypeInfo GetOverridableInitializerTypeInfo(size_t index) const;

    // Model metadata
    ModelMetadata GetModelMetadata() const;

    // Profiling
    char* EndProfiling(OrtAllocator* allocator) const;

    // Session options
    ConstSessionOptions GetSessionOptions() const;

    // Providers
    std::vector<const char*> GetRegisteredProviderNames() const;

    // EP dynamic options
    void SetEpDynamicOptions(const std::vector<const char*>& keys,
                              const std::vector<const char*>& values);

    // LoRA adapters
    void AddLoraAdapter(const LoraAdapter& adapter);
    void RemoveLoraAdapter(const LoraAdapter& adapter);

    // EP graph assignment info
    std::vector<EpAssignedSubgraph> GetEpGraphAssignmentInfo() const;
};

}  // namespace Ort
```

**Usage:**
```cpp
Ort::Env env;
Ort::SessionOptions opts;
opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

// Load from file
Ort::Session session(env, L"model.onnx", opts);

// Load from memory
std::ifstream file("model.onnx", std::ios::binary);
std::vector<char> data((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
Ort::Session session(env, data.data(), data.size(), opts);

// Get I/O info
auto input_count = session.GetInputCount();
for (size_t i = 0; i < input_count; i++) {
    auto name = session.GetInputName(i, allocator);
    auto type_info = session.GetInputTypeInfo(i);
}
```

---

## 3.5 SessionOptions (Ort::SessionOptions)

```cpp
namespace Ort {

class SessionOptions : public Base<OrtSessionOptions> {
public:
    SessionOptions();
    SessionOptions(const SessionOptions& other);  // Deep clone
    SessionOptions& operator=(SessionOptions&& other) noexcept;

    // Clone
    SessionOptions Clone() const;

    // Threading
    SessionOptions& SetIntraOpNumThreads(int intra_op_num_threads);
    SessionOptions& SetInterOpNumThreads(int inter_op_num_threads);
    SessionOptions& SetGlobalIntraOpNumThreads(int intra_op_num_threads);
    SessionOptions& SetGlobalInterOpNumThreads(int inter_op_num_threads);

    // Optimization
    SessionOptions& SetGraphOptimizationLevel(GraphOptimizationLevel level);
    SessionOptions& SetOptimizedModelFilePath(const ORTCHAR_T* model_path);

    // Execution
    SessionOptions& SetExecutionMode(ExecutionMode execution_mode);
    SessionOptions& SetDeterministicCompute(bool value);

    // Memory
    SessionOptions& EnableMemPattern();
    SessionOptions& DisableMemPattern();
    SessionOptions& EnableMemReuse();
    SessionOptions& DisableMemReuse();

    // Profiling
    SessionOptions& EnableProfiling(const ORTCHAR_T* profile_file_prefix);
    SessionOptions& DisableProfiling();

    // Logging
    SessionOptions& SetSessionLogId(const char* logid);
    SessionOptions& SetSessionLogSeverityLevel(int level);
    SessionOptions& SetSessionLogVerbosityLevel(int level);

    // Threading control
    SessionOptions& DisablePerSessionThreads();
    SessionOptions& SetGlobalSpinControl(int allow_spinning);

    // File loading
    SessionOptions& SetFileParallelLoadingThreshold(size_t threshold);

    // Config entries
    SessionOptions& AddConfigEntry(const char* key, const char* value);
    SessionOptions& AddInitializer(const char* name, const OrtValue* value);

    // Custom ops
    SessionOptions& RegisterCustomOpsLibrary(const ORTCHAR_T* library_path,
                                              CustomOpDomain** domain = nullptr);
    SessionOptions& RegisterCustomOpsUsingFunction(const char* registration_func);
    SessionOptions& AddCustomOpDomain(CustomOpDomain domain);

    // Execution providers
    SessionOptions& AppendExecutionProvider_CPU(int use_arena);
    SessionOptions& AppendExecutionProvider_CUDA(const OrtCUDAProviderOptionsV2& options);
    SessionOptions& AppendExecutionProvider_TensorRT(const OrtTensorRTProviderOptionsV2& options);
    SessionOptions& AppendExecutionProvider_CoreML(uint32_t flags);
    SessionOptions& AppendExecutionProvider_Nnapi(uint32_t flags);
    SessionOptions& AppendExecutionProvider_OpenVINO(const char* device_type);
    SessionOptions& AppendExecutionProvider_DML(int device_id);
    SessionOptions& AppendExecutionProvider_WebGPU(OrtArenaCfg* arena_config);
    SessionOptions& AppendExecutionProvider_Xnnpack(const char* const* keys,
                                                     const char* const* values,
                                                     size_t num_keys);

    // Pre-packed weights
    SessionOptions& AddPrepackedWeightsContainer(PrepackedWeightsContainer& container);
};

}  // namespace Ort
```

---

## 3.6 Value (Ort::Value)

```cpp
namespace Ort {

class Value : public Base<OrtValue> {
public:
    Value() = default;
    Value(std::nullptr_t) {}  // Create empty value
    Value& operator=(Value&& other) noexcept;

    // --- Tensor Creation ---

    // Create tensor with allocator
    template <typename T>
    static Value CreateTensor(const OrtMemoryInfo* info,
                               const int64_t* shape, size_t shape_len);

    template <typename T>
    static Value CreateTensor(const OrtMemoryInfo* info,
                               std::vector<int64_t> shape);

    // Create tensor with existing data
    template <typename T>
    static Value CreateTensor(const OrtMemoryInfo* info,
                               T* data, size_t data_len,
                               const int64_t* shape, size_t shape_len);

    template <typename T>
    static Value CreateTensor(const OrtMemoryInfo* info,
                               T* data, size_t data_len,
                               const std::vector<int64_t>& shape);

    // Create tensor with explicit element type
    static Value CreateTensor(const OrtMemoryInfo* info,
                               void* data, size_t data_len,
                               const int64_t* shape, size_t shape_len,
                               ONNXTensorElementDataType type);

    // --- Data Access ---

    template <typename T>
    T* GetTensorMutableData();

    template <typename T>
    const T* GetTensorData() const;

    void* GetTensorMutableRawData();
    const void* GetTensorRawData() const;

    // --- Type Info ---

    TensorTypeAndShapeInfo GetTensorTypeAndShapeInfo() const;
    ONNXTensorElementDataType GetTensorElementType() const;
    size_t GetTensorDimCount() const;
    std::vector<int64_t> GetTensorShape() const;
    size_t GetTensorDataSizeInBytes() const;

    // --- Type checking ---
    bool IsTensor() const;
    ONNXType GetType() const;

    // --- Shape inference ---
    ShapeInferContext GetShapeInferenceContext() const;

    // --- Sparse tensor ---
    static Value CreateSparseTensor(const OrtMemoryInfo* info,
                                     const int64_t* dense_shape, size_t dense_shape_len,
                                     const int64_t* values_shape, size_t values_shape_len,
                                     ONNXTensorElementDataType type);

    OrtSparseFormat GetSparseFormat() const;
    const void* GetSparseTensorValues() const;
    const void* GetSparseTensorIndices(OrtSparseIndicesFormat format) const;

    // --- Map/Sequence ---
    Value GetValue(int index, OrtAllocator* allocator) const;
    size_t GetCount() const;

    // --- String tensor ---
    static Value CreateStringTensor(const OrtMemoryInfo* info,
                                     const char* const* strings,
                                     const int64_t* shape, size_t shape_len);

    // --- Opaque value ---
    static Value CreateOpaqueValue(const char* domain, const char* type_name,
                                    const void* data, size_t data_len);
    void GetOpaqueValue(const char* domain, const char* type_name,
                        void* data, size_t* data_len) const;
};

}  // namespace Ort
```

**Usage:**
```cpp
auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

// Create from existing data
std::vector<float> data(3 * 224 * 224, 1.0f);
std::vector<int64_t> shape = {1, 3, 224, 224};
auto input = Ort::Value::CreateTensor<float>(
    memory_info, data.data(), data.size(), shape.data(), shape.size());

// Create with allocator (empty tensor)
auto output = Ort::Value::CreateTensor<float>(memory_info, shape.data(), shape.size());

// Access output data
float* output_data = output.GetTensorMutableData<float>();
const float* const_data = output.GetTensorData<float>();

// Get shape info
auto type_info = input.GetTensorTypeAndShapeInfo();
auto elem_type = type_info.GetElementType();    // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
auto dims = type_info.GetShape();               // {1, 3, 224, 224}
auto dim_count = type_info.GetDimensionsCount(); // 4
```

---

## 3.7 MemoryInfo (Ort::MemoryInfo)

```cpp
namespace Ort {

class MemoryInfo : public Base<OrtMemoryInfo> {
public:
    MemoryInfo() = default;

    // Factory methods
    static MemoryInfo CreateCpu(OrtAllocatorType type, OrtMemType mem_type);

    static MemoryInfo Create(const char* name, OrtAllocatorType type,
                              int id, OrtMemType mem_type);

    // Accessors
    const char* GetAllocatorName() const;
    OrtAllocatorType GetAllocatorType() const;
    int GetDeviceId() const;
    OrtMemType GetMemoryType() const;

    // Comparison
    bool operator==(const MemoryInfo& other) const;
    bool operator!=(const MemoryInfo& other) const;
};

}  // namespace Ort
```

**Usage:**
```cpp
// Default CPU arena allocator
auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

// Specific device
auto gpu_mem = Ort::MemoryInfo::Create("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);

// CPU pinned memory
auto pinned = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPUOutput);
```

---

## 3.8 TypeInfo (Ort::TypeInfo)

```cpp
namespace Ort {

class TypeInfo : public Base<OrtTypeInfo> {
public:
    TypeInfo() = default;

    ONNXType GetONNXType() const;

    // Cast to tensor type info (returns nullptr if not tensor)
    UnownedTensorTypeAndShapeInfo GetTensorTypeAndShapeInfo() const;

    // Cast to map type info
    UnownedMapTypeInfo GetMapTypeInfo() const;

    // Cast to sequence type info
    UnownedSequenceTypeInfo GetSequenceTypeInfo() const;

    // Cast to optional type info
    UnownedOptionalTypeInfo GetOptionalTypeInfo() const;
};

class TensorTypeAndShapeInfo : public Base<OrtTensorTypeAndShapeInfo> {
public:
    TensorTypeAndShapeInfo() = default;

    ONNXTensorElementDataType GetElementType() const;
    size_t GetDimensionsCount() const;
    std::vector<int64_t> GetShape() const;
    void GetDimensions(int64_t* values, size_t values_count) const;
    std::vector<const char*> GetSymbolicDimensions() const;
    size_t GetElementCount() const;
};

}  // namespace Ort
```

---

## 3.9 IoBinding (Ort::IoBinding)

```cpp
namespace Ort {

class IoBinding : public Base<OrtIoBinding> {
public:
    IoBinding(Session& session);

    // Bind inputs
    void BindInput(const char* name, const Value& value);
    void BindInput(const char* name, const OrtMemoryInfo* mem_info,
                   const void* data, size_t data_len,
                   const int64_t* shape, size_t shape_len,
                   ONNXTensorElementDataType type);

    // Bind outputs
    void BindOutput(const char* name, const Value& value);
    void BindOutput(const char* name, const OrtMemoryInfo* mem_info);

    // Get outputs
    std::vector<Value> GetOutputValues() const;
    std::vector<const char*> GetOutputNames() const;

    // Synchronization
    void SynchronizeInputs();
    void SynchronizeOutputs();

    // Clear
    void ClearBinding();
};

}  // namespace Ort
```

**Usage:**
```cpp
Ort::Session session(env, L"model.onnx", opts);
Ort::IoBinding io_binding(session);

// Bind GPU input
auto gpu_mem = Ort::MemoryInfo::Create("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
std::vector<float> input_data(3 * 224 * 224, 1.0f);
auto input_tensor = Ort::Value::CreateTensor<float>(
    gpu_mem, input_data.data(), input_data.size(), {1, 3, 224, 224});
io_binding.BindInput("input", input_tensor);

// Bind output to GPU
io_binding.BindOutput("output", gpu_mem);

// Run
Ort::RunOptions run_opts;
session.Run(run_opts, io_binding);

// Get results
auto outputs = io_binding.GetOutputValues();
```

---

## 3.10 RunOptions (Ort::RunOptions)

```cpp
namespace Ort {

class RunOptions : public Base<OrtRunOptions> {
public:
    RunOptions();

    // Logging
    RunOptions& SetRunLogSeverityLevel(int level);
    RunOptions& SetRunLogVerbosityLevel(int level);
    RunOptions& SetRunTag(const char* tag);

    // Termination
    RunOptions& SetTerminate();
    RunOptions& UnsetTerminate();

    // Config
    RunOptions& AddConfigEntry(const char* key, const char* value);
};

}  // namespace Ort
```

---

## 3.11 CustomOpDomain (Ort::CustomOpDomain)

```cpp
namespace Ort {

class CustomOpDomain : public Base<OrtCustomOpDomain> {
public:
    CustomOpDomain(const char* domain_name);

    void Add(const OrtCustomOp* op);
};

}  // namespace Ort
```

### CustomOpBase Template

```cpp
template <typename TOp, typename TKernel>
class CustomOpBase {
public:
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo& info) const {
        return new TKernel(api, info);
    }

    void KernelDestroy(void* kernel) {
        delete static_cast<TKernel*>(kernel);
    }

    // Must be implemented by derived class:
    // const char* GetName() const;
    // const char* GetExecutionProviderType() const;
    // size_t GetInputTypeCount() const;
    // ONNXTensorElementDataType GetInputType(size_t idx) const;
    // size_t GetOutputTypeCount() const;
    // ONNXTensorElementDataType GetOutputType(size_t idx) const;

    // Optional overrides:
    OrtMemType GetInputMemoryType(size_t idx) const { return OrtMemTypeDefault; }
    int GetInputCharacteristic(size_t idx) const { return 0; }
    int GetOutputCharacteristic(size_t idx) const { return 0; }
};
```

### Complete Custom Op Example

```cpp
#include <onnxruntime_cxx_api.h>

struct MyKernel {
    MyKernel(const OrtApi& api, const OrtKernelInfo& info)
        : api_(api), info_(info) {
        // Read attributes
        alpha_ = api_.KernelInfoGetAttribute_float(&info, "alpha");
    }

    void Compute(OrtKernelContext* context) {
        // Get inputs
        const OrtValue* input = nullptr;
        api_.KernelContext_GetInput(context, 0, &input);

        // Get input shape
        OrtTensorTypeAndShapeInfo* shape_info = nullptr;
        api_.GetTensorTypeAndShape(input, &shape_info);
        size_t count = 0;
        api_.GetTensorShapeEl(shape_info, 0, reinterpret_cast<int64_t*>(&count));

        // Create output
        OrtValue* output = nullptr;
        int64_t dims[] = {static_cast<int64_t>(count)};
        api_.KernelContext_GetOutput(context, 0, dims, 1, &output);

        // Compute
        const float* in_data = nullptr;
        api_.GetTensorData(input, reinterpret_cast<const void**>(&in_data));
        float* out_data = nullptr;
        api_.GetTensorMutableData(output, reinterpret_cast<void**>(&out_data));

        for (size_t i = 0; i < count; i++) {
            out_data[i] = in_data[i] * alpha_;
        }

        api_.ReleaseTensorTypeAndShapeInfo(shape_info);
    }

private:
    const OrtApi& api_;
    const OrtKernelInfo& info_;
    float alpha_ = 1.0f;
};

struct MyCustomOp : Ort::CustomOpBase<MyCustomOp, MyKernel> {
    const char* GetName() const { return "MyCustomOp"; }
    const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }

    size_t GetInputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetInputType(size_t) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }

    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};

// Registration
Ort::CustomOpDomain domain("my_domain");
MyCustomOp my_op;
domain.Add(&my_op);

Ort::SessionOptions opts;
opts.AddCustomOpDomain(std::move(domain));
```

---

## 3.12 ModelMetadata (Ort::ModelMetadata)

```cpp
namespace Ort {

class ModelMetadata : public Base<OrtModelMetadata> {
public:
    ModelMetadata() = default;

    char* GetProducerName(OrtAllocator* allocator) const;
    char* GetGraphName(OrtAllocator* allocator) const;
    char* GetDomain(OrtAllocator* allocator) const;
    char* GetDescription(OrtAllocator* allocator) const;
    int64_t GetVersion() const;
    char* LookupCustomMetadataMap(const char* key, OrtAllocator* allocator) const;
    std::vector<const char*> GetCustomMetadataMapKeys(OrtAllocator* allocator) const;
};

}  // namespace Ort
```

---

## 3.13 Allocator (Ort::Allocator)

```cpp
namespace Ort {

class Allocator : public Base<OrtAllocator> {
public:
    Allocator(const Session& session, const OrtMemoryInfo* mem_info);
    void* Alloc(size_t size);
    void Free(void* ptr);
    const OrtMemoryInfo* GetInfo() const;
};

class AllocatorWithDefaultOptions {
public:
    AllocatorWithDefaultOptions();
    void* Alloc(size_t size);
    void Free(void* ptr);
    operator OrtAllocator*() const;
};

}  // namespace Ort
```

---

## 3.14 ThreadingOptions (Ort::ThreadingOptions)

```cpp
namespace Ort {

class ThreadingOptions : public Base<OrtThreadingOptions> {
public:
    ThreadingOptions();

    ThreadingOptions& SetGlobalIntraOpNumThreads(int num_threads);
    ThreadingOptions& SetGlobalInterOpNumThreads(int num_threads);
    ThreadingOptions& SetGlobalSpinControl(int allow_spinning);
    ThreadingOptions& SetGlobalDenormalAsZero();
};

}  // namespace Ort
```

---

## 3.15 ArenaCfg (Ort::ArenaCfg)

```cpp
namespace Ort {

class ArenaCfg : public Base<OrtArenaCfg> {
public:
    ArenaCfg(size_t max_memory, int arena_extension_strategy,
             int initial_chunk_size_bytes, int max_dead_bytes_per_chunk);
};

}  // namespace Ort
```

---

## 3.16 PrepackedWeightsContainer

```cpp
namespace Ort {

class PrepackedWeightsContainer : public Base<OrtPrepackedWeightsContainer> {
public:
    PrepackedWeightsContainer();
};

}  // namespace Ort
```

---

## 3.17 Provider Options

### CUDAProviderOptions

```cpp
namespace Ort {

class CUDAProviderOptionsV2 : public Base<OrtCUDAProviderOptionsV2> {
public:
    CUDAProviderOptionsV2();

    void Update(const std::vector<const char*>& keys,
                const std::vector<const char*>& values);

    std::string GetKeysAndValues() const;
};

}  // namespace Ort
```

**Available options:**
| Key | Type | Description |
|-----|------|-------------|
| `device_id` | int | CUDA device ID (default: 0) |
| `gpu_mem_limit` | size_t | GPU memory limit in bytes (default: UINT_MAX) |
| `arena_extend_strategy` | string | "kNextPowerOfTwo" or "kSameAsRequested" |
| `cudnn_conv_algo_search` | string | "EXHAUSTIVE", "HEURISTIC", or "DEFAULT" |
| `do_copy_in_default_stream` | bool | Copy in default stream (default: true) |
| `cudnn_conv_use_max_workspace` | bool | Use max workspace for cuDNN conv (default: false) |
| `enable_cuda_graph` | bool | Enable CUDA graph capture (default: false) |
| `enable_skip_layout_transform` | bool | Skip layout transform (default: false) |
| `enable_cublas_lt_gemm` | bool | Enable cuBLASLt GEMM (default: false) |
| `prefer_nhwc` | bool | Prefer NHWC layout (default: false) |
| `use_cublas_nn_kernel_only` | bool | Use cuBLAS NN kernel only (default: false) |
| `use_tf32` | bool | Use TF32 for matmul (default: true) |
| `use_cudnn_conv` | bool | Use cuDNN for conv (default: true) |
| `use_blockwise_quantization` | bool | Use blockwise quantization (default: true) |
| `user_compute_stream` | void* | External CUDA stream |
| `enable_cuda_tunable_op` | bool | Enable tunable operators (default: false) |

### TensorRTProviderOptions

```cpp
namespace Ort {

class TensorRTProviderOptionsV2 : public Base<OrtTensorRTProviderOptionsV2> {
public:
    TensorRTProviderOptionsV2();

    void Update(const std::vector<const char*>& keys,
                const std::vector<const char*>& values);
};

}  // namespace Ort
```

**Available options:**
| Key | Type | Description |
|-----|------|-------------|
| `device_id` | int | CUDA device ID |
| `trt_max_workspace_size` | size_t | Max workspace size |
| `trt_max_partition_iterations` | int | Max partition iterations |
| `trt_min_subgraph_size` | int | Min subgraph size |
| `trt_fp16_enable` | bool | Enable FP16 |
| `trt_int8_enable` | bool | Enable INT8 |
| `trt_int8_calibration_table` | string | INT8 calibration table path |
| `trt_dla_enable` | bool | Enable DLA |
| `trt_dla_core` | int | DLA core ID |
| `trt_engine_cache_enable` | bool | Enable engine caching |
| `trt_engine_cache_path` | string | Engine cache directory |
| `trt_timing_cache_enable` | bool | Enable timing cache |
| `trt_timing_cache_path` | string | Timing cache path |
| `trt_dump_subgraphs` | bool | Dump subgraphs |
| `trt_builder_opt_level` | int | Builder optimization level (0-5) |
| `trt_extra_plugin_lib_paths` | string | Custom plugin library paths |

---

## 3.18 Complete C++ Inference Example

```cpp
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

int main() {
    // 1. Setup
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 2. Create session
    Ort::Session session(env, L"model.onnx", session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // 3. Print model info
    std::cout << "Inputs: " << session.GetInputCount() << std::endl;
    std::cout << "Outputs: " << session.GetOutputCount() << std::endl;

    for (size_t i = 0; i < session.GetInputCount(); i++) {
        auto name = session.GetInputName(i, allocator);
        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        std::cout << "  Input " << i << ": " << name
                  << " type=" << tensor_info.GetElementType()
                  << " shape=[";
        for (auto d : shape) std::cout << d << " ";
        std::cout << "]" << std::endl;
    }

    // 4. Create input tensor
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    size_t input_size = 1 * 3 * 224 * 224;
    std::vector<float> input_data(input_size, 1.0f);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());

    // 5. Run inference
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1);

    // 6. Process output
    auto& output_tensor = output_tensors[0];
    auto* output_data = output_tensor.GetTensorData<float>();
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape();

    std::cout << "Output shape: [";
    for (auto d : output_shape) std::cout << d << " ";
    std::cout << "]" << std::endl;
    std::cout << "First 5 values: ";
    for (int i = 0; i < 5; i++) std::cout << output_data[i] << " ";
    std::cout << std::endl;

    return 0;
}
```

### C++ GPU Inference Example

```cpp
#include <onnxruntime_cxx_api.h>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "gpu_inference");

    // Configure CUDA EP
    Ort::CUDAProviderOptionsV2 cuda_options;
    std::vector<const char*> keys = {"device_id", "arena_extend_strategy",
                                      "cudnn_conv_algo_search"};
    std::vector<const char*> values = {"0", "kNextPowerOfTwo", "EXHAUSTIVE"};
    cuda_options.Update(keys, values);

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session_options.AppendExecutionProvider_CUDA(cuda_options);

    Ort::Session session(env, L"model.onnx", session_options);

    // Create input on CPU (will be copied to GPU)
    auto cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<float> input_data(3 * 224 * 224, 1.0f);
    std::vector<int64_t> shape = {1, 3, 224, 224};

    auto input = Ort::Value::CreateTensor<float>(
        cpu_mem, input_data.data(), input_data.size(), shape.data(), shape.size());

    // IO Binding for GPU inference
    Ort::IoBinding io_binding(session);
    io_binding.BindInput("input", input);

    auto gpu_mem = Ort::MemoryInfo::Create("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    io_binding.BindOutput("output", gpu_mem);

    Ort::RunOptions run_options;
    session.Run(run_options, io_binding);

    // Copy output back to CPU
    auto outputs = io_binding.GetOutputValues();
    // outputs[0] is on GPU memory

    return 0;
}
```
