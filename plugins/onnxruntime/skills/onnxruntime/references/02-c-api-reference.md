# ONNX Runtime Reference - Chapter 2: C API Reference

Complete reference for the ONNX Runtime C API defined in `onnxruntime_c_api.h`. This is the foundational API that all language bindings wrap.

---

## 2.1 API Version and Initialization

### 2.1.1 Version Constants

```c
#define ORT_API_VERSION 27
```

### 2.1.2 API Discovery

```c
// Get the API base structure
const OrtApiBase* OrtGetApiBase(void);

struct OrtApiBase {
    const OrtApi* (*GetApi)(uint32_t version);  // Returns OrtApi for requested version
    const char* (*GetVersionString)(void);       // Returns version string "major.minor.rev"
};
```

### 2.1.3 Platform Macros

```c
// Windows: wchar_t paths, __stdcall calling convention
#ifdef _WIN32
    #define ORTCHAR_T wchar_t
    #define ORT_TSTR(X) L##X
    #define ORT_API_CALL __stdcall
#else
    #define ORTCHAR_T char
    #define ORT_TSTR(X) X
    #define ORT_API_CALL
#endif
```

---

## 2.2 Runtime Types

### 2.2.1 Opaque Types

```c
ORT_RUNTIME_CLASS(Env);                    // Execution environment
ORT_RUNTIME_CLASS(Status);                 // Error status (nullptr = success)
ORT_RUNTIME_CLASS(MemoryInfo);             // Memory location info
ORT_RUNTIME_CLASS(IoBinding);              // Input/Output binding
ORT_RUNTIME_CLASS(Session);                // Inference session
ORT_RUNTIME_CLASS(Value);                  // Tensor/value container
ORT_RUNTIME_CLASS(RunOptions);             // Per-run options
ORT_RUNTIME_CLASS(TypeInfo);               // Type information
ORT_RUNTIME_CLASS(TensorTypeAndShapeInfo);  // Tensor type + shape
ORT_RUNTIME_CLASS(MapTypeInfo);            // Map type info
ORT_RUNTIME_CLASS(SequenceTypeInfo);        // Sequence type info
ORT_RUNTIME_CLASS(OptionalTypeInfo);        // Optional type info
ORT_RUNTIME_CLASS(SessionOptions);          // Session configuration
ORT_RUNTIME_CLASS(CustomOpDomain);          // Custom operator domain
ORT_RUNTIME_CLASS(ModelMetadata);           // Model metadata
ORT_RUNTIME_CLASS(ThreadPoolParams);        // Thread pool parameters
ORT_RUNTIME_CLASS(ThreadingOptions);        // Threading configuration
ORT_RUNTIME_CLASS(ArenaCfg);               // Arena allocator config
ORT_RUNTIME_CLASS(PrepackedWeightsContainer); // Pre-packed weights
ORT_RUNTIME_CLASS(TensorRTProviderOptionsV2); // TensorRT options
ORT_RUNTIME_CLASS(NvTensorRtRtxProviderOptions); // TensorRT RTX options
ORT_RUNTIME_CLASS(CUDAProviderOptionsV2);   // CUDA options
ORT_RUNTIME_CLASS(CANNProviderOptions);     // CANN options
ORT_RUNTIME_CLASS(DnnlProviderOptions);     // DNNL options
ORT_RUNTIME_CLASS(Op);                      // Operator handle
ORT_RUNTIME_CLASS(OpAttr);                  // Operator attribute
ORT_RUNTIME_CLASS(Logger);                  // Logger instance
ORT_RUNTIME_CLASS(ShapeInferContext);        // Shape inference context
ORT_RUNTIME_CLASS(LoraAdapter);             // LoRA adapter
ORT_RUNTIME_CLASS(ValueInfo);               // Value information
ORT_RUNTIME_CLASS(Node);                    // Graph node
ORT_RUNTIME_CLASS(Graph);                   // Graph instance
ORT_RUNTIME_CLASS(Model);                   // Model instance
ORT_RUNTIME_CLASS(ModelCompilationOptions); // Model compilation config
ORT_RUNTIME_CLASS(HardwareDevice);          // Hardware device
ORT_RUNTIME_CLASS(EpDevice);               // EP device mapping
ORT_RUNTIME_CLASS(KeyValuePairs);           // Key-value string pairs
ORT_RUNTIME_CLASS(SyncStream);             // Synchronization stream
ORT_RUNTIME_CLASS(ExternalInitializerInfo); // External initializer info
ORT_RUNTIME_CLASS(ExternalResourceImporter); // External resource import
ORT_RUNTIME_CLASS(ExternalMemoryHandle);    // External memory handle
ORT_RUNTIME_CLASS(ExternalSemaphoreHandle); // External semaphore handle
ORT_RUNTIME_CLASS(DeviceEpIncompatibilityDetails); // EP incompatibility info
ORT_RUNTIME_CLASS(EpAssignedSubgraph);      // EP-assigned subgraph
ORT_RUNTIME_CLASS(EpAssignedNode);          // EP-assigned node
```

### 2.2.2 Enumerations

#### Tensor Element Data Types

```c
typedef enum ONNXTensorElementDataType {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED    = 0,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT        = 1,   // float (IEEE 754)
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8        = 2,   // uint8_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8         = 3,   // int8_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16       = 4,   // uint16_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16        = 5,   // int16_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32        = 6,   // int32_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64        = 7,   // int64_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING       = 8,   // std::string
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL         = 9,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16      = 10,  // IEEE 754 float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE       = 11,  // double
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32       = 12,  // uint32_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64       = 13,  // uint64_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64    = 14,  // complex64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128   = 15,  // complex128
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16     = 16,  // Brain float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN    = 17,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ  = 18,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2      = 19,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ  = 20,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4       = 21,   // Packed uint4
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4        = 22,   // Packed int4
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1  = 23,   // Packed float4
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2       = 24,   // Packed uint2
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2        = 25,   // Packed int2
} ONNXTensorElementDataType;
```

#### ONNX Type Enum

```c
typedef enum ONNXType {
    ONNX_TYPE_UNKNOWN,
    ONNX_TYPE_TENSOR,
    ONNX_TYPE_SEQUENCE,
    ONNX_TYPE_MAP,
    ONNX_TYPE_OPAQUE,
    ONNX_TYPE_SPARSETENSOR,
    ONNX_TYPE_OPTIONAL
} ONNXType;
```

#### Sparse Format

```c
typedef enum OrtSparseFormat {
    ORT_SPARSE_UNDEFINED    = 0,
    ORT_SPARSE_COO          = 0x1,
    ORT_SPARSE_CSRC         = 0x2,
    ORT_SPARSE_BLOCK_SPARSE = 0x4
} OrtSparseFormat;

enum OrtSparseIndicesFormat {
    ORT_SPARSE_COO_INDICES,
    ORT_SPARSE_CSR_INNER_INDICES,
    ORT_SPARSE_CSR_OUTER_INDICES,
    ORT_SPARSE_BLOCK_SPARSE_INDICES
};
```

#### Logging Levels

```c
typedef enum OrtLoggingLevel {
    ORT_LOGGING_LEVEL_VERBOSE,   // Most verbose
    ORT_LOGGING_LEVEL_INFO,
    ORT_LOGGING_LEVEL_WARNING,
    ORT_LOGGING_LEVEL_ERROR,
    ORT_LOGGING_LEVEL_FATAL      // Least verbose
} OrtLoggingLevel;
```

#### Error Codes

```c
typedef enum OrtErrorCode {
    ORT_OK,
    ORT_FAIL,
    ORT_INVALID_ARGUMENT,
    ORT_NO_SUCHFILE,
    ORT_NO_MODEL,
    ORT_ENGINE_ERROR,
    ORT_RUNTIME_EXCEPTION,
    ORT_INVALID_PROTOBUF,
    ORT_MODEL_LOADED,
    ORT_NOT_IMPLEMENTED,
    ORT_INVALID_GRAPH,
    ORT_EP_FAIL,
    ORT_MODEL_LOAD_CANCELED,
    ORT_MODEL_REQUIRES_COMPILATION,
    ORT_NOT_FOUND,
} OrtErrorCode;
```

#### Operator Attribute Types

```c
typedef enum OrtOpAttrType {
    ORT_OP_ATTR_UNDEFINED = 0,
    ORT_OP_ATTR_INT,
    ORT_OP_ATTR_INTS,
    ORT_OP_ATTR_FLOAT,
    ORT_OP_ATTR_FLOATS,
    ORT_OP_ATTR_STRING,
    ORT_OP_ATTR_STRINGS,
    ORT_OP_ATTR_GRAPH,
    ORT_OP_ATTR_TENSOR,
} OrtOpAttrType;
```

#### Graph Optimization Levels

```c
typedef enum GraphOptimizationLevel {
    ORT_DISABLE_ALL  = 0,    // No optimizations
    ORT_ENABLE_BASIC = 1,    // Level 1: Basic optimizations
    ORT_ENABLE_EXTENDED = 2, // Level 2: Extended optimizations
    ORT_ENABLE_LAYOUT = 3,   // Level 3: Layout optimizations
    ORT_ENABLE_ALL   = 99    // All optimizations (including Level 4)
} GraphOptimizationLevel;
```

#### Execution Modes

```c
typedef enum ExecutionMode {
    ORT_SEQUENTIAL = 0,  // Execute ops sequentially
    ORT_PARALLEL   = 1   // Execute independent ops in parallel
} ExecutionMode;
```

#### Language Projection

```c
typedef enum OrtLanguageProjection {
    ORT_PROJECTION_C        = 0,
    ORT_PROJECTION_CPLUSPLUS = 1,
    ORT_PROJECTION_CSHARP   = 2,
    ORT_PROJECTION_PYTHON   = 3,
    ORT_PROJECTION_JAVA     = 4,
    ORT_PROJECTION_WINML    = 5,
    ORT_PROJECTION_NODEJS   = 6,
} OrtLanguageProjection;
```

#### Allocator Types

```c
typedef enum OrtAllocatorType {
    OrtInvalidAllocator  = -1,
    OrtDeviceAllocator   = 0,   // Direct device allocation
    OrtArenaAllocator    = 1,   // Arena-based allocation (default)
    OrtReadOnlyAllocator = 2,   // Read-only memory
} OrtAllocatorType;
```

#### Memory Types

```c
typedef enum OrtMemType {
    OrtMemTypeCPUInput  = -2,   // CPU memory for non-CPU EP inputs
    OrtMemTypeCPUOutput = -1,   // CPU accessible memory for outputs
    OrtMemTypeCPU       = -1,   // Same as CPUOutput
    OrtMemTypeDefault   = 0,    // Default allocator for EP
} OrtMemType;

typedef enum OrtDeviceMemoryType {
    OrtDeviceMemoryType_DEFAULT        = 0,  // Device memory
    OrtDeviceMemoryType_HOST_ACCESSIBLE = 5, // Shared/pinned memory
} OrtDeviceMemoryType;
```

#### Hardware Device Types

```c
typedef enum OrtHardwareDeviceType {
    OrtHardwareDeviceType_CPU,
    OrtHardwareDeviceType_GPU,
    OrtHardwareDeviceType_NPU
} OrtHardwareDeviceType;

typedef enum OrtMemoryInfoDeviceType {
    OrtMemoryInfoDeviceType_CPU  = 0,
    OrtMemoryInfoDeviceType_GPU  = 1,
    OrtMemoryInfoDeviceType_FPGA = 2,
    OrtMemoryInfoDeviceType_NPU  = 3,
} OrtMemoryInfoDeviceType;
```

#### EP Device Policy

```c
typedef enum OrtExecutionProviderDevicePolicy {
    OrtExecutionProviderDevicePolicy_DEFAULT,
    OrtExecutionProviderDevicePolicy_PREFER_CPU,
    OrtExecutionProviderDevicePolicy_PREFER_NPU,
    OrtExecutionProviderDevicePolicy_PREFER_GPU,
    OrtExecutionProviderDevicePolicy_MAX_PERFORMANCE,
    OrtExecutionProviderDevicePolicy_MAX_EFFICIENCY,
    OrtExecutionProviderDevicePolicy_MIN_OVERALL_POWER,
} OrtExecutionProviderDevicePolicy;
```

#### EP Incompatibility Reasons

```c
typedef enum OrtDeviceEpIncompatibilityReason {
    OrtDeviceEpIncompatibility_NONE               = 0,
    OrtDeviceEpIncompatibility_DRIVER_INCOMPATIBLE = 1 << 0,
    OrtDeviceEpIncompatibility_DEVICE_INCOMPATIBLE = 1 << 1,
    OrtDeviceEpIncompatibility_MISSING_DEPENDENCY  = 1 << 2,
    OrtDeviceEpIncompatibility_UNKNOWN            = 1 << 31
} OrtDeviceEpIncompatibilityReason;
```

---

## 2.3 OrtAllocator Structure

```c
typedef struct OrtAllocator {
    uint32_t version;                          // Must be ORT_API_VERSION

    // Required: Allocate size bytes
    void* (ORT_API_CALL *Alloc)(struct OrtAllocator* this_, size_t size);

    // Required: Free previously allocated memory
    void (ORT_API_CALL *Free)(struct OrtAllocator* this_, void* p);

    // Required: Return memory info
    const struct OrtMemoryInfo* (ORT_API_CALL *Info)(const struct OrtAllocator* this_);

    // Optional: Session initialization allocation (since 1.18)
    void* (ORT_API_CALL *Reserve)(struct OrtAllocator* this_, size_t size);

    // Optional: Get allocator statistics (since 1.23)
    OrtStatus* (ORT_API_CALL *GetStats)(const struct OrtAllocator* this_,
                                         OrtKeyValuePairs** out);

    // Optional: Stream-aware allocation (since 1.23)
    void* (ORT_API_CALL *AllocOnStream)(struct OrtAllocator* this_,
                                         size_t size, OrtSyncStream* stream);

    // Optional: Release unused memory (since 1.25)
    OrtStatus* (ORT_API_CALL *Shrink)(struct OrtAllocator* this_);
} OrtAllocator;

// Statistics keys returned by GetStats:
// "Limit"            - Bytes limit of the allocator (-1 = no limit)
// "InUse"            - Bytes currently in use
// "TotalAllocated"   - Total allocated bytes
// "MaxInUse"         - Maximum bytes in use
// "NumAllocs"        - Number of allocations
// "NumReserves"      - Number of reserve calls
// "NumArenaExtensions"  - Arena extensions
// "NumArenaShrinkages"  - Arena shrinkages
// "MaxAllocSize"     - Max single allocation seen
```

### Logging Callback

```c
typedef void (ORT_API_CALL *OrtLoggingFunction)(
    void* param,              // User parameter
    OrtLoggingLevel severity, // Log severity
    const char* category,     // Log category
    const char* logid,        // Logger ID
    const char* code_location,// Source file:line
    const char* message       // Log message
);
```

### EP Selection Delegate

```c
typedef OrtStatus* (ORT_API_CALL *EpSelectionDelegate)(
    const OrtEpDevice** ep_devices,       // Available devices
    size_t num_devices,                   // Number of devices
    const OrtKeyValuePairs* model_metadata, // Model metadata
    const OrtKeyValuePairs* runtime_metadata, // Runtime metadata (may be null)
    const OrtEpDevice** selected,         // Output: selected devices
    size_t max_selected,                  // Max 8 devices
    size_t* num_selected,                 // Output: count
    void* state                           // User state
);
```

---

## 2.4 OrtApi Structure (Complete Function Reference)

The `OrtApi` structure contains all C API functions. Below is the complete reference:

### 2.4.1 Environment Functions

```c
// Create default environment
OrtStatus* CreateEnv(OrtLoggingLevel default_logging_level,
                     _In_ const char* logid,
                     _Outptr_ OrtEnv** env);

// Create environment with custom logger
OrtStatus* CreateEnvWithCustomLogger(OrtLoggingFunction logging_function,
                                     _In_opt_ void* logger_param,
                                     OrtLoggingLevel default_logging_level,
                                     _In_ const char* logid,
                                     _Outptr_ OrtEnv** env);

// Create environment with custom logger and threading options
OrtStatus* CreateEnvWithCustomLoggerAndGlobalThreadPools(
    OrtLoggingFunction logging_function,
    _In_opt_ void* logger_param,
    OrtLoggingLevel default_logging_level,
    _In_ const char* logid,
    _In_ const OrtThreadingOptions* tp_options,
    _Outptr_ OrtEnv** env);

// Release environment
void ReleaseEnv(_Frees_ptr_opt_ OrtEnv* env);

// Enable telemetry (Windows only)
OrtStatus* EnableTelemetryEvents(_In_ OrtEnv* env);

// Disable telemetry
OrtStatus* DisableTelemetryEvents(_In_ OrtEnv* env);

// Set language projection for telemetry
OrtStatus* SetLanguageProjection(_In_ OrtEnv* env, OrtLanguageProjection projection);

// Get environment thread pool IDs
OrtStatus* GetEnvThreadPool(_In_ OrtEnv* env,
                            _In_ const OrtThreadingOptions* tp_options,
                            _Outptr_ OrtThreadPoolParams** pool_params);
```

### 2.4.2 Status Functions

```c
// Get error message from status
const char* GetErrorMessage(_In_ OrtStatus* status);

// Get error code from status
OrtErrorCode GetErrorCode(_In_ const OrtStatus* status);

// Release status
void ReleaseStatus(_Frees_ptr_opt_ OrtStatus* status);

// Create a status
OrtStatus* CreateStatus(OrtErrorCode code, _In_ const char* msg);
```

### 2.4.3 Memory Info Functions

```c
// Create memory info
OrtStatus* CreateMemoryInfo(_In_ const char* name,
                            OrtAllocatorType type,
                            int id,
                            OrtMemType mem_type,
                            _Outptr_ OrtMemoryInfo** out);

// Create CPU memory info (convenience)
OrtStatus* CreateCpuMemoryInfo(OrtAllocatorType type,
                                OrtMemType mem_type,
                                _Outptr_ OrtMemoryInfo** out);

// Release memory info
void ReleaseMemoryInfo(_Frees_ptr_opt_ OrtMemoryInfo* info);

// Compare two memory info instances
int CompareMemoryInfo(_In_ const OrtMemoryInfo* a, _In_ const OrtMemoryInfo* b);

// Get memory info name
OrtStatus* MemoryInfoGetName(_In_ const OrtMemoryInfo* ptr, _Out_ const char** out);

// Get memory info id
OrtStatus* MemoryInfoGetId(_In_ const OrtMemoryInfo* ptr, _Out_ int* out);

// Get memory info type
OrtStatus* MemoryInfoGetMemType(_In_ const OrtMemoryInfo* ptr, _Out_ OrtMemType* out);

// Get memory info allocator type
OrtStatus* MemoryInfoGetType(_In_ const OrtMemoryInfo* ptr,
                              _Out_ OrtAllocatorType* out);
```

### 2.4.4 Allocator Functions

```c
// Get default allocator
OrtStatus* GetAllocatorWithDefaultOptions(_Outptr_ OrtAllocator** out);

// Create arena allocator config
OrtStatus* CreateArenaCfg(_In_ size_t max_memory,
                          _In_ int arena_extension_strategy,
                          _In_ int initial_chunk_size_bytes,
                          _In_ int max_dead_bytes_per_chunk,
                          _Outptr_ OrtArenaCfg** out);

void ReleaseArenaCfg(_Frees_ptr_opt_ OrtArenaCfg* cfg);

// Create custom allocator
OrtStatus* CreateCustomAllocator(_In_ OrtAllocator* custom_allocator,
                                  _In_ OrtAllocatorType type,
                                  _Outptr_ OrtAllocator** out);

// Register custom allocator
OrtStatus* RegisterCustomAllocator(_In_ OrtAllocator* allocator, _In_ OrtEnv* env);

// Release allocator (for allocators returned by ORT)
void ReleaseAllocator(_Frees_ptr_opt_ OrtAllocator* allocator);
```

### 2.4.5 Session Options Functions

```c
// Create session options
OrtStatus* CreateSessionOptions(_Outptr_ OrtSessionOptions** out);

// Clone session options
OrtStatus* CloneSessionOptions(_In_ const OrtSessionOptions* opts,
                               _Outptr_ OrtSessionOptions** out);

// Release session options
void ReleaseSessionOptions(_Frees_ptr_opt_ OrtSessionOptions* opts);

// Set intra-op thread count
OrtStatus* SetIntraOpNumThreads(_In_ OrtSessionOptions* opts, int intra_op_num_threads);

// Set inter-op thread count
OrtStatus* SetInterOpNumThreads(_In_ OrtSessionOptions* opts, int inter_op_num_threads);

// Set graph optimization level
OrtStatus* SetGraphOptimizationLevel(_In_ OrtSessionOptions* opts,
                                      GraphOptimizationLevel level);

// Set execution mode
OrtStatus* SetExecutionMode(_In_ OrtSessionOptions* opts, ExecutionMode mode);

// Enable/disable memory pattern optimization
OrtStatus* EnableMemPattern(_In_ OrtSessionOptions* opts);

// Disable memory pattern optimization
OrtStatus* DisableMemPattern(_In_ OrtSessionOptions* opts);

// Enable/disable memory reuse
OrtStatus* EnableMemReuse(_In_ OrtSessionOptions* opts);
OrtStatus* DisableMemReuse(_In_ OrtSessionOptions* opts);

// Set optimized model file path (for saving optimized model)
OrtStatus* SetOptimizedModelFilePath(_In_ OrtSessionOptions* opts,
                                      _In_ const ORTCHAR_T* model_path);

// Enable profiling
OrtStatus* EnableProfiling(_In_ OrtSessionOptions* opts, _In_ const ORTCHAR_T* profile_path);
OrtStatus* DisableProfiling(_In_ OrtSessionOptions* opts);

// Set thread spawning strategy
OrtStatus* AddRunConfigEntry(_In_ OrtRunOptions* options,
                              _In_ const char* key,
                              _In_ const char* value);

// Add session config entry
OrtStatus* AddSessionConfigEntry(_In_ OrtSessionOptions* options,
                                  _In_ const char* key,
                                  _In_ const char* value);

// Set session log ID
OrtStatus* SetSessionLogId(_In_ OrtSessionOptions* options, _In_ const char* logid);

// Set session log severity level
OrtStatus* SetSessionLogSeverityLevel(_In_ OrtSessionOptions* options, int level);

// Set session log verbosity
OrtStatus* SetSessionLogVerbosityLevel(_In_ OrtSessionOptions* options, int level);

// Set thread pool options
OrtStatus* DisablePerSessionThreads(_In_ OrtSessionOptions* opts);

// Add custom operator domain
OrtStatus* AddCustomOpDomain(_In_ OrtSessionOptions* options,
                              _In_ OrtCustomOpDomain* domain);

// Register custom ops library
OrtStatus* RegisterCustomOpsLibrary(_In_ OrtSessionOptions* options,
                                     _In_ const ORTCHAR_T* library_path,
                                     _Outptr_ OrtCustomOpDomain** domain);

// Register custom ops using V2 API
OrtStatus* RegisterCustomOpsUsingV2(_In_ OrtSessionOptions* options,
                                     _In_ const char* library_path);

// Set deterministic compute
OrtStatus* SetDeterministicCompute(_In_ OrtSessionOptions* opts, _In_ bool value);

// Set file parallel loading threshold
OrtStatus* SetFileParallelLoadingThreshold(_In_ OrtSessionOptions* opts, size_t threshold);
```

### 2.4.6 Execution Provider Functions

```c
// Append CPU execution provider
OrtStatus* SessionOptionsAppendExecutionProvider_CPU(_In_ OrtSessionOptions* opts, int use_arena);

// Append CUDA execution provider (legacy API)
OrtStatus* SessionOptionsAppendExecutionProvider_CUDA(_In_ OrtSessionOptions* opts, int device_id);

// Create CUDA provider options
OrtStatus* CreateCUDAProviderOptions(_Outptr_ OrtCUDAProviderOptionsV2** out);

// Set CUDA provider options
OrtStatus* UpdateCUDAProviderOptions(_Inout_ OrtCUDAProviderOptionsV2* options,
                                      _In_reads_(num_keys) const char* const* keys,
                                      _In_reads_(num_keys) const char* const* values,
                                      _In_ size_t num_keys);

// Get CUDA provider options
OrtStatus* GetCUDAProviderOptionsAsString(_In_ const OrtCUDAProviderOptionsV2* options,
                                           _Outptr_ OrtAllocator* allocator,
                                           _Outptr_ char** out);

// Append CUDA EP from options
OrtStatus* SessionOptionsAppendExecutionProvider_CUDA_V2(_In_ OrtSessionOptions* opts,
                                                          _In_ const OrtCUDAProviderOptionsV2* cuda_options);

void ReleaseCUDAProviderOptions(_Frees_ptr_opt_ OrtCUDAProviderOptionsV2* opts);

// TensorRT options (same pattern as CUDA)
OrtStatus* CreateTensorRTProviderOptions(_Outptr_ OrtTensorRTProviderOptionsV2** out);
OrtStatus* UpdateTensorRTProviderOptions(_Inout_ OrtTensorRTProviderOptionsV2* options,
                                          _In_ const char* const* keys,
                                          _In_ const char* const* values,
                                          _In_ size_t num_keys);
OrtStatus* SessionOptionsAppendExecutionProvider_TensorRT_V2(
    _In_ OrtSessionOptions* opts,
    _In_ const OrtTensorRTProviderOptionsV2* trt_options);
void ReleaseTensorRTProviderOptions(_Frees_ptr_opt_ OrtTensorRTProviderOptionsV2* opts);

// CoreML EP
OrtStatus* SessionOptionsAppendExecutionProvider_CoreML(_In_ OrtSessionOptions* opts, uint32_t flags);

// NNAPI EP
OrtStatus* SessionOptionsAppendExecutionProvider_Nnapi(_In_ OrtSessionOptions* opts, uint32_t flags);

// OpenVINO EP
OrtStatus* SessionOptionsAppendExecutionProvider_OpenVINO(_In_ OrtSessionOptions* opts,
                                                           _In_ const char* device_type);

// DNNL EP
OrtStatus* CreateDnnlProviderOptions(_Outptr_ OrtDnnlProviderOptions** out);
OrtStatus* UpdateDnnlProviderOptions(_Inout_ OrtDnnlProviderOptions* options,
                                      _In_ const char* const* keys,
                                      _In_ const char* const* values,
                                      _In_ size_t num_keys);
OrtStatus* SessionOptionsAppendExecutionProvider_Dnnl(_In_ OrtSessionOptions* opts,
                                                       _In_ const OrtDnnlProviderOptions* dnnl_options);

// DirectML EP
OrtStatus* SessionOptionsAppendExecutionProvider_DML(_In_ OrtSessionOptions* opts, int device_id);

// WebGPU EP
OrtStatus* SessionOptionsAppendExecutionProvider_WebGPU(_In_ OrtSessionOptions* opts,
                                                          _In_ OrtArenaCfg* arena_config);

// XNNPACK EP
OrtStatus* SessionOptionsAppendExecutionProvider_Xnnpack(_In_ OrtSessionOptions* opts,
                                                           _In_ const char* const* keys,
                                                           _In_ const char* const* values,
                                                           _In_ size_t num_keys);

// Vitis-AI EP
OrtStatus* SessionOptionsAppendExecutionProvider_VitisAI(_In_ OrtSessionOptions* opts,
                                                          _In_ const char* const* keys,
                                                          _In_ const char* const* values,
                                                          _In_ size_t num_keys);

// ACL EP
OrtStatus* SessionOptionsAppendExecutionProvider_ACL(_In_ OrtSessionOptions* opts, const char* use_arena);

// CANN EP
OrtStatus* CreateCANNProviderOptions(_Outptr_ OrtCANNProviderOptions** out);
OrtStatus* SessionOptionsAppendExecutionProvider_CANN(_In_ OrtSessionOptions* opts,
                                                       _In_ const OrtCANNProviderOptions* cann_options);

// RKNPU EP
OrtStatus* SessionOptionsAppendExecutionProvider_Rknpu(_In_ OrtSessionOptions* opts);

// TVM EP
OrtStatus* SessionOptionsAppendExecutionProvider_Tvm(_In_ OrtSessionOptions* opts,
                                                      _In_ const char* const* keys,
                                                      _In_ const char* const* values,
                                                      _In_ size_t num_keys);
```

### 2.4.7 Session Functions

```c
// Create session from file
OrtStatus* CreateSession(_In_ const OrtEnv* env,
                         _In_ const ORTCHAR_T* model_path,
                         _In_ const OrtSessionOptions* opts,
                         _Outptr_ OrtSession** out);

// Create session from memory
OrtStatus* CreateSessionFromArray(_In_ const OrtEnv* env,
                                   _In_ const void* model_data,
                                   size_t model_data_length,
                                   _In_ const OrtSessionOptions* opts,
                                   _Outptr_ OrtSession** out);

// Release session
void ReleaseSession(_Frees_ptr_opt_ OrtSession* session);

// Run inference
OrtStatus* Run(_In_ OrtSession* session,
               _In_ const OrtRunOptions* run_options,
               _In_reads_(input_len) const char* const* input_names,
               _In_reads_(input_len) const OrtValue* const* input_values,
               size_t input_len,
               _In_reads_(output_len) const char* const* output_names,
               size_t output_len,
               _Out_writes_(output_len) OrtValue** output_values);

// Run with IO binding
OrtStatus* RunWithBinding(_In_ OrtSession* session,
                           _In_ const OrtRunOptions* run_options,
                           _In_ const OrtIoBinding* binding);

// Get input count
OrtStatus* SessionGetInputCount(_In_ const OrtSession* session, _Out_ size_t* count);

// Get output count
OrtStatus* SessionGetOutputCount(_In_ const OrtSession* session, _Out_ size_t* count);

// Get overridable initializers count
OrtStatus* SessionGetOverridableInitializerCount(_In_ const OrtSession* session,
                                                   _Out_ size_t* count);

// Get input name
OrtStatus* SessionGetInputName(_In_ const OrtSession* session, size_t index,
                                _Inout_ OrtAllocator* allocator, _Outptr_ char** output);

// Get output name
OrtStatus* SessionGetOutputName(_In_ const OrtSession* session, size_t index,
                                 _Inout_ OrtAllocator* allocator, _Outptr_ char** output);

// Get overridable initializer name
OrtStatus* SessionGetOverridableInitializerName(_In_ const OrtSession* session,
                                                  size_t index,
                                                  _Inout_ OrtAllocator* allocator,
                                                  _Outptr_ char** output);

// Get input type info
OrtStatus* SessionGetInputTypeInfo(_In_ const OrtSession* session, size_t index,
                                    _Outptr_ OrtTypeInfo** type_info);

// Get output type info
OrtStatus* SessionGetOutputTypeInfo(_In_ const OrtSession* session, size_t index,
                                     _Outptr_ OrtTypeInfo** type_info);

// Get model metadata
OrtStatus* SessionGetModelMetadata(_In_ const OrtSession* session,
                                    _Outptr_ OrtModelMetadata** metadata);

// End profiling
OrtStatus* SessionEndProfiling(_In_ OrtSession* session,
                                _Inout_ OrtAllocator* allocator,
                                _Outptr_ char** profile_file);
```

### 2.4.8 OrtValue (Tensor) Functions

```c
// Create tensor from buffer
OrtStatus* CreateTensorWithDataAsOrtValue(_In_ const OrtMemoryInfo* info,
                                           _In_ void* data, size_t data_len,
                                           _In_ const int64_t* shape, size_t shape_len,
                                           ONNXTensorElementDataType type,
                                           _Outptr_ OrtValue** out);

// Create tensor with allocator
OrtStatus* CreateTensorAsOrtValue(_In_ OrtAllocator* allocator,
                                   _In_ const int64_t* shape, size_t shape_len,
                                   ONNXTensorElementDataType type,
                                   _Outptr_ OrtValue** out);

// Get tensor mutable data buffer
OrtStatus* GetTensorMutableData(_In_ OrtValue* value, _Outptr_ void** out);

// Get tensor constant data buffer
OrtStatus* GetTensorData(_In_ const OrtValue* value, _Outptr_ const void** out);

// Get tensor type and shape
OrtStatus* GetTensorTypeAndShape(_In_ const OrtValue* value,
                                  _Outptr_ OrtTensorTypeAndShapeInfo** out);

// Get tensor element type
OrtStatus* GetTensorElementType(_In_ const OrtTensorTypeAndShapeInfo* info,
                                 _Out_ ONNXTensorElementDataType* out);

// Get tensor dimension count
OrtStatus* GetDimensionsCount(_In_ const OrtTensorTypeAndShapeInfo* info,
                               _Out_ size_t* out);

// Get tensor dimensions
OrtStatus* GetDimensions(_In_ const OrtTensorTypeAndShapeInfo* info,
                          _Out_ int64_t* dim_values, size_t dim_values_length);

// Get symbolic dimensions
OrtStatus* GetSymbolicDimensions(_In_ const OrtTensorTypeAndShapeInfo* info,
                                  _Out_ const char* const* dim_values,
                                  size_t dim_values_length);

// Get tensor shape string
OrtStatus* GetTensorShapeEl(_In_ const OrtTensorTypeAndShapeInfo* info,
                             size_t index, _Out_ int64_t* out);

// Release tensor type and shape info
void ReleaseTensorTypeAndShapeInfo(_Frees_ptr_opt_ OrtTensorTypeAndShapeInfo* info);

// Release OrtValue
void ReleaseValue(_Frees_ptr_opt_ OrtValue* value);

// Check if value is tensor
OrtStatus* IsTensor(_In_ const OrtValue* value, _Out_ int* out);

// Get value type
OrtStatus*GetValueType(_In_ const OrtValue* value, _Out_ ONNXType* out);

// Create opaque value (for non-tensor types)
OrtStatus* CreateOpaqueValue(_In_ const char* domain, _In_ const char* type_name,
                              _In_ const void* data, size_t data_len,
                              _Outptr_ OrtValue** out);

// Get opaque value data
OrtStatus* GetOpaqueValue(_In_ const char* domain, _In_ const char* type_name,
                           _In_ OrtValue* value, _Out_ void** data, size_t* data_len);
```

### 2.4.9 Type Info Functions

```c
void ReleaseTypeInfo(_Frees_ptr_opt_ OrtTypeInfo* info);

// Get ONNX type from type info
OrtStatus* GetOnnxTypeFromTypeInfo(_In_ const OrtTypeInfo* typeinfo, _Out_ ONNXType* out);

// Cast type info to tensor type and shape
OrtStatus* CastTypeInfoToTensorInfo(_In_ const OrtTypeInfo* typeinfo,
                                     _Outptr_ const OrtTensorTypeAndShapeInfo** out);

// Get map type info
OrtStatus* GetMapKeyType(_In_ const OrtMapTypeInfo* map_info,
                          _Out_ ONNXTensorElementDataType* out);
OrtStatus* GetMapValueType(_In_ const OrtMapTypeInfo* map_info,
                            _Outptr_ OrtTypeInfo** out);

// Get sequence element type
OrtStatus* GetSequenceElementType(_In_ const OrtSequenceTypeInfo* sequence_info,
                                   _Outptr_ OrtTypeInfo** out);

// Get optional type contained type
OrtStatus* GetOptionalContainedTypeInfo(_In_ const OrtOptionalTypeInfo* optional_info,
                                         _Outptr_ OrtTypeInfo** out);

void ReleaseMapTypeInfo(_Frees_ptr_opt_ OrtMapTypeInfo* info);
void ReleaseSequenceTypeInfo(_Frees_ptr_opt_ OrtSequenceTypeInfo* info);
void ReleaseOptionalTypeInfo(_Frees_ptr_opt_ OrtOptionalTypeInfo* info);
```

### 2.4.10 Run Options Functions

```c
// Create run options
OrtStatus* CreateRunOptions(_Outptr_ OrtRunOptions** out);

// Release run options
void ReleaseRunOptions(_Frees_ptr_opt_ OrtRunOptions* options);

// Set run log severity level
OrtStatus* RunOptionsSetRunLogSeverityLevel(_In_ OrtRunOptions* options, int level);

// Set run log verbosity
OrtStatus* RunOptionsSetRunLogVerbosityLevel(_In_ OrtRunOptions* options, int level);

// Set run tag (for profiling)
OrtStatus* RunOptionsSetRunTag(_In_ OrtRunOptions* options, _In_ const char* tag);

// Set terminate flag (signal to stop)
OrtStatus* RunOptionsSetTerminate(_In_ OrtRunOptions* options);

// Unset terminate flag
OrtStatus* RunOptionsUnsetTerminate(_In_ OrtRunOptions* options);

// Add run config entry
OrtStatus* AddRunConfigEntry(_In_ OrtRunOptions* options,
                              _In_ const char* key,
                              _In_ const char* value);
```

### 2.4.11 IO Binding Functions

```c
// Create IO binding
OrtStatus* CreateIoBinding(_In_ OrtSession* session, _Outptr_ OrtIoBinding** out);

// Release IO binding
void ReleaseIoBinding(_Frees_ptr_opt_ OrtIoBinding* binding);

// Bind input with OrtValue
OrtStatus* BindInput(_In_ OrtIoBinding* binding, _In_ const char* name,
                      _In_ const OrtValue* value);

// Bind input with CPU data
OrtStatus* BindInputToDevice(_In_ OrtIoBinding* binding, _In_ const char* name,
                              _In_ const OrtMemoryInfo* mem_info,
                              _In_ const void* data, size_t data_len,
                              _In_ const int64_t* shape, size_t shape_len,
                              ONNXTensorElementDataType type);

// Bind output to OrtValue
OrtStatus* BindOutput(_In_ OrtIoBinding* binding, _In_ const char* name,
                       _In_ const OrtValue* value);

// Bind output to device (let ORT allocate)
OrtStatus* BindOutputToDevice(_In_ OrtIoBinding* binding, _In_ const char* name,
                               _In_ const OrtMemoryInfo* mem_info);

// Get bound output values
OrtStatus* GetBoundOutputValues(_In_ const OrtIoBinding* binding,
                                 _Inout_ OrtAllocator* allocator,
                                 _Outptr_ OrtValue** output, size_t* output_count);

// Clear binding
void ClearBinding(_In_ OrtIoBinding* binding);

// Synchronize inputs
OrtStatus* SynchronizeInputs(_In_ OrtIoBinding* binding);

// Synchronize outputs
OrtStatus* SynchronizeOutputs(_In_ OrtIoBinding* binding);
```

### 2.4.12 Model Metadata Functions

```c
void ReleaseModelMetadata(_Frees_ptr_opt_ OrtModelMetadata* metadata);

OrtStatus* ModelMetadataGetProducerName(_In_ const OrtModelMetadata* metadata,
                                         _Inout_ OrtAllocator* allocator,
                                         _Outptr_ char** value);

OrtStatus* ModelMetadataGetGraphName(_In_ const OrtModelMetadata* metadata,
                                      _Inout_ OrtAllocator* allocator,
                                      _Outptr_ char** value);

OrtStatus* ModelMetadataGetDomain(_In_ const OrtModelMetadata* metadata,
                                   _Inout_ OrtAllocator* allocator,
                                   _Outptr_ char** value);

OrtStatus* ModelMetadataGetDescription(_In_ const OrtModelMetadata* metadata,
                                        _Inout_ OrtAllocator* allocator,
                                        _Outptr_ char** value);

OrtStatus* ModelMetadataGetVersion(_In_ const OrtModelMetadata* metadata,
                                    _Out_ int64_t* value);

OrtStatus* ModelMetadataGetCustomMetadataMapKeys(_In_ const OrtModelMetadata* metadata,
                                                   _Inout_ OrtAllocator* allocator,
                                                   _Outptr_ char*** keys, size_t* num_keys);

OrtStatus* ModelMetadataLookupCustomMetadataMap(_In_ const OrtModelMetadata* metadata,
                                                  _Inout_ OrtAllocator* allocator,
                                                  _In_ const char* key,
                                                  _Outptr_ char** value);
```

### 2.4.13 Custom Operator Functions

```c
// Create custom op domain
OrtStatus* CreateCustomOpDomain(_In_ const char* domain, _Outptr_ OrtCustomOpDomain** out);

void ReleaseCustomOpDomain(_Frees_ptr_opt_ OrtCustomOpDomain* domain);

// Add custom op to domain
OrtStatus* CustomOpDomain_Add(_In_ OrtCustomOpDomain* domain, _In_ const OrtCustomOp* op);

// Custom operator structure (must be implemented by user)
struct OrtCustomOp {
    uint32_t version;  // Must be ORT_API_VERSION

    void* (ORT_API_CALL *CreateKernel)(const OrtCustomOp* op, const OrtApi* api,
                                        const OrtKernelInfo* info);

    const char* (ORT_API_CALL *GetName)(const OrtCustomOp* op);

    const char* (ORT_API_CALL *GetExecutionProviderType)(const OrtCustomOp* op);

    ONNXTensorElementDataType (ORT_API_CALL *GetInputType)(const OrtCustomOp* op,
                                                            size_t index);

    size_t (ORT_API_CALL *GetInputTypeCount)(const OrtCustomOp* op);

    ONNXTensorElementDataType (ORT_API_CALL *GetOutputType)(const OrtCustomOp* op,
                                                             size_t index);

    size_t (ORT_API_CALL *GetOutputTypeCount)(const OrtCustomOp* op);

    void (ORT_API_CALL *KernelCompute)(void* op_kernel, OrtKernelContext* context);

    void (ORT_API_CALL *KernelDestroy)(void* op_kernel);

    // Optional: Input memory type
    OrtMemType (ORT_API_CALL *GetInputMemoryType)(const OrtCustomOp* op, size_t index);

    // Optional: Input shape inference
    int (ORT_API_CALL *GetInputCharacteristic)(const OrtCustomOp* op, size_t index);

    // Optional: Output shape inference
    int (ORT_API_CALL *GetOutputCharacteristic)(const OrtCustomOp* op, size_t index);

    // Optional: Variadic inputs
    int (ORT_API_CALL *GetVariadicInputCount)(const OrtCustomOp* op);

    // Optional: Variadic outputs
    int (ORT_API_CALL *GetVariadicOutputCount)(const OrtCustomOp* op);
};
```

### 2.4.14 Kernel Context Functions

```c
// Get input count
OrtStatus* KernelContext_GetInputCount(_In_ const OrtKernelContext* context,
                                        _Out_ size_t* out);

// Get input value
OrtStatus* KernelContext_GetInput(_In_ const OrtKernelContext* context,
                                   _In_ size_t index,
                                   _Out_ const OrtValue** out);

// Get output count
OrtStatus* KernelContext_GetOutputCount(_In_ const OrtKernelContext* context,
                                         _Out_ size_t* out);

// Get output value
OrtStatus* KernelContext_GetOutput(_In_ OrtKernelContext* context,
                                   _In_ size_t index,
                                   _In_ const int64_t* dims, size_t dims_len,
                                   _Outptr_ OrtValue** out);

// Get allocator
OrtStatus* KernelContext_GetAllocator(_In_ OrtKernelContext* context,
                                       _In_ const OrtMemoryInfo* mem_info,
                                       _Outptr_ OrtAllocator** out);

// Get GPU stream ID
OrtStatus* KernelContext_GetGPUStreamId(_In_ OrtKernelContext* context,
                                         _Out_ void** stream);

// Get node name and function (for debugging)
OrtStatus* KernelContext_GetNodeName(_In_ const OrtKernelContext* context,
                                      _Out_ const char** out);
```

### 2.4.15 Kernel Info Functions

```c
// Get attribute as int64
OrtStatus* KernelInfoGetAttribute_int64(_In_ const OrtKernelInfo* info,
                                         _In_ const char* name,
                                         _Out_ int64_t* out);

// Get attribute as float
OrtStatus* KernelInfoGetAttribute_float(_In_ const OrtKernelInfo* info,
                                         _In_ const char* name,
                                         _Out_ float* out);

// Get attribute as string
OrtStatus* KernelInfoGetAttribute_string(_In_ const OrtKernelInfo* info,
                                          _In_ const char* name,
                                          _Out_ char* out, size_t* size);

// Get input count
OrtStatus* KernelInfoGetInputCount(_In_ const OrtKernelInfo* info, _Out_ size_t* out);

// Get output count
OrtStatus* KernelInfoGetOutputCount(_In_ const OrtKernelInfo* info, _Out_ size_t* out);

// Get input type info
OrtStatus* KernelInfoGetInputTypeInfo(_In_ const OrtKernelInfo* info, size_t index,
                                       _Outptr_ OrtTypeInfo** out);

// Get output type info
OrtStatus* KernelInfoGetOutputTypeInfo(_In_ const OrtKernelInfo* info, size_t index,
                                        _Outptr_ OrtTypeInfo** out);

// Get input name
OrtStatus* KernelInfoGetInputName(_In_ const OrtKernelInfo* info, size_t index,
                                   _Out_ char* out, size_t* size);

// Get output name
OrtStatus* KernelInfoGetOutputName(_In_ const OrtKernelInfo* info, size_t index,
                                    _Out_ char* out, size_t* size);

// Get node name
OrtStatus* KernelInfoGetNodeName(_In_ const OrtKernelInfo* info, _Out_ const char** out);

// Copy kernel info
OrtStatus* CopyKernelInfo(_In_ const OrtKernelInfo* info, _Outptr_ OrtKernelInfo** out);

void ReleaseKernelInfo(_Frees_ptr_opt_ OrtKernelInfo* info);
```

### 2.4.16 LoRA Adapter Functions

```c
// Create LoRA adapter
OrtStatus* CreateLoraAdapter(_In_ const ORTCHAR_T* adapter_path,
                              _In_ OrtSessionOptions* options,
                              _Outptr_ OrtLoraAdapter** out);

// Release LoRA adapter
void ReleaseLoraAdapter(_Frees_ptr_opt_ OrtLoraAdapter* adapter);
```

### 2.4.17 Pre-packed Weights Functions

```c
// Create pre-packed weights container
OrtStatus* CreatePrepackedWeightsContainer(_Outptr_ OrtPrepackedWeightsContainer** out);

void ReleasePrepackedWeightsContainer(_Frees_ptr_opt_ OrtPrepackedWeightsContainer* container);
```

### 2.4.18 Threading Functions

```c
// Create threading options
OrtStatus* CreateThreadingOptions(_Outptr_ OrtThreadingOptions** out);

void ReleaseThreadingOptions(_Frees_ptr_opt_ OrtThreadingOptions* opts);

// Set global thread pool params
OrtStatus* SetGlobalThreadPools(_In_ OrtEnv* env, _In_ const OrtThreadingOptions* tp_options);

// Set global intra-op threads
OrtStatus* SetGlobalIntraOpNumThreads(_In_ OrtThreadingOptions* tp_options, int num_threads);

// Set global inter-op threads
OrtStatus* SetGlobalInterOpNumThreads(_In_ OrtThreadingOptions* tp_options, int num_threads);

// Set spin control
OrtStatus* SetGlobalSpinControl(_In_ OrtThreadingOptions* tp_options, int allow_spinning);

// Set thread affinity
OrtStatus* SetGlobalDenormalAsZero(_In_ OrtThreadingOptions* tp_options);
```

### 2.4.19 Session Options Accessor Functions

```c
// Get session options (returns const)
OrtStatus* GetSessionOptions(_In_ const OrtSession* session,
                              _Outptr_ const OrtSessionOptions** out);

// Get available providers
OrtStatus* GetAvailableProviders(_Outptr_ char*** providers, size_t* num_providers);

// Release provider list
void ReleaseAvailableProviders(_Frees_ptr_opt_ char** providers, size_t num_providers);

// Set EP dynamic options
OrtStatus* SetEpDynamicOptions(_In_ OrtSession* session,
                                _In_ const char* const* keys,
                                _In_ const char* const* values,
                                size_t num_keys);
```

### 2.4.20 Shape Inference Functions

```c
// Run shape inference on a model
OrtStatus* SessionOptionsSetShapeInferenceFunction(_In_ OrtSessionOptions* opts,
                                                     OrtShapeInferFunc shape_infer_func);
```

### 2.4.21 Sparse Tensor Functions

```c
// Create sparse tensor
OrtStatus* CreateSparseTensorAsOrtValue(_In_ OrtAllocator* allocator,
                                         _In_ const int64_t* dense_shape,
                                         size_t dense_shape_len,
                                         _In_ const int64_t* values_shape,
                                         size_t values_shape_len,
                                         ONNXTensorElementDataType type,
                                         _Outptr_ OrtValue** out);

// Fill sparse tensor with COO format
OrtStatus* FillSparseTensorCoo(_In_ OrtValue* value,
                                _In_ const OrtMemoryInfo* mem_info,
                                _In_ const int64_t* values_shape,
                                size_t values_shape_len,
                                _In_ const void* values,
                                _In_ const int64_t* indices_shape,
                                size_t indices_shape_len,
                                _In_ const void* indices);

// Fill sparse tensor with CSR format
OrtStatus* FillSparseTensorCsr(_In_ OrtValue* value,
                                _In_ const OrtMemoryInfo* mem_info,
                                _In_ const int64_t* values_shape,
                                size_t values_shape_len,
                                _In_ const void* values,
                                _In_ const int64_t* inner_indices_shape,
                                size_t inner_indices_shape_len,
                                _In_ const void* inner_indices,
                                _In_ const int64_t* outer_indices_shape,
                                size_t outer_indices_shape_len,
                                _In_ const void* outer_indices);

// Fill sparse tensor with BlockSparse format
OrtStatus* FillSparseTensorBlockSparse(_In_ OrtValue* value,
                                        _In_ const OrtMemoryInfo* mem_info,
                                        _In_ const int64_t* values_shape,
                                        size_t values_shape_len,
                                        _In_ const void* values,
                                        _In_ const int64_t* indices_shape,
                                        size_t indices_shape_len,
                                        _In_ const void* indices);

// Get sparse tensor format
OrtStatus* GetSparseTensorFormat(_In_ const OrtValue* value, _Out_ OrtSparseFormat* out);

// Get sparse tensor values data
OrtStatus* GetSparseTensorValues(_In_ const OrtValue* value, _Out_ const void** out);

// Get sparse tensor indices
OrtStatus* GetSparseTensorIndices(_In_ const OrtValue* value,
                                   OrtSparseIndicesFormat indices_format,
                                   _Out_ const void** indices_data,
                                   _Out_ size_t* indices_len);
```

### 2.4.22 Model and Graph Functions

```c
// Create model from file
OrtStatus* CreateModel(_In_ const ORTCHAR_T* model_path,
                        _In_ const OrtSessionOptions* options,
                        _Outptr_ OrtModel** out);

// Create model from array
OrtStatus* CreateModelFromArray(_In_ const void* model_data,
                                 size_t model_data_len,
                                 _In_ const OrtSessionOptions* options,
                                 _Outptr_ OrtModel** out);

void ReleaseModel(_Frees_ptr_opt_ OrtModel* model);

// Get model graph
OrtStatus* ModelGetGraph(_Inout_ OrtModel* model, _Outptr_ OrtGraph** out);

// Save model
OrtStatus* ModelSave(_In_ const OrtModel* model, _In_ const ORTCHAR_T* model_path);

void ReleaseGraph(_Frees_ptr_opt_ OrtGraph* graph);
```

### 2.4.23 Hardware Device Functions

```c
// Get all available EP devices
OrtStatus* GetEpDevices(_Outptr_ const OrtEpDevice*** devices, _Out_ size_t* num_devices);

// Release EP device list
void ReleaseEpDevices(_Frees_ptr_opt_ const OrtEpDevice** devices, size_t num_devices);

// Get EP device hardware device
OrtStatus* EpDevice_HardwareDevice(_In_ const OrtEpDevice* ep_device,
                                    _Out_ const OrtHardwareDevice** device);

// Get EP device EP name
OrtStatus* EpDevice_EpName(_In_ const OrtEpDevice* ep_device, _Out_ const char** name);

// Get EP device EP vendor
OrtStatus* EpDevice_EpVendor(_In_ const OrtEpDevice* ep_device, _Out_ const char** vendor);

// Get EP device EP metadata
OrtStatus* EpDevice_EpMetadata(_In_ const OrtEpDevice* ep_device,
                                _Out_ const OrtKeyValuePairs** metadata);

// Get EP device incompatibility details
OrtStatus* EpDevice_GetIncompatibilityDetails(_In_ const OrtEpDevice* ep_device,
                                                _Out_ const OrtDeviceEpIncompatibilityDetails** details);

// Get hardware device type
OrtStatus* HardwareDevice_Type(_In_ const OrtHardwareDevice* device,
                                _Out_ OrtHardwareDeviceType* type);

// Get hardware device vendor ID
OrtStatus* HardwareDevice_VendorId(_In_ const OrtHardwareDevice* device, _Out_ uint32_t* id);

// Get hardware device ID
OrtStatus* HardwareDevice_DeviceId(_In_ const OrtHardwareDevice* device, _Out_ uint32_t* id);

// Get hardware device name
OrtStatus* HardwareDevice_Name(_In_ const OrtHardwareDevice* device, _Out_ const char** name);

// Get hardware device description
OrtStatus* HardwareDevice_Description(_In_ const OrtHardwareDevice* device,
                                       _Out_ const char** description);

// Get hardware device memory size
OrtStatus* HardwareDevice_MemorySize(_In_ const OrtHardwareDevice* device, _Out_ size_t* size);
```

---

## 2.5 API Usage Patterns

### 2.5.1 Complete C Inference Example

```c
#include "onnxruntime_c_api.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // 1. Get API
    const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    // 2. Create environment
    OrtEnv* env;
    ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);

    // 3. Create session options
    OrtSessionOptions* session_options;
    ort->CreateSessionOptions(&session_options);
    ort->SetIntraOpNumThreads(session_options, 1);
    ort->SetGraphOptimizationLevel(session_options, ORT_ENABLE_ALL);

    // 4. Create session
    OrtSession* session;
    ort->CreateSession(env, L"model.onnx", session_options, &session);

    // 5. Create memory info
    OrtMemoryInfo* memory_info;
    ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);

    // 6. Create input tensor
    int64_t input_shape[] = {1, 3, 224, 224};
    size_t input_size = 1 * 3 * 224 * 224;
    float* input_data = (float*)malloc(input_size * sizeof(float));
    for (size_t i = 0; i < input_size; i++) input_data[i] = 1.0f;

    OrtValue* input_tensor;
    ort->CreateTensorWithDataAsOrtValue(
        memory_info, input_data, input_size * sizeof(float),
        input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);

    // 7. Run inference
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    OrtValue* output_tensor = NULL;

    ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1,
             output_names, 1, &output_tensor);

    // 8. Get output data
    float* output_data;
    ort->GetTensorMutableData(output_tensor, (void**)&output_data);

    // 9. Cleanup
    ort->ReleaseValue(output_tensor);
    ort->ReleaseValue(input_tensor);
    ort->ReleaseMemoryInfo(memory_info);
    ort->ReleaseSession(session);
    ort->ReleaseSessionOptions(session_options);
    ort->ReleaseEnv(env);
    free(input_data);

    return 0;
}
```

### 2.5.2 Error Handling Pattern

```c
#define CHECK_ORT(expr) \
    do { \
        OrtStatus* status = (expr); \
        if (status != NULL) { \
            const char* msg = ort->GetErrorMessage(status); \
            OrtErrorCode code = ort->GetErrorCode(status); \
            fprintf(stderr, "ORT Error [%d]: %s\n", code, msg); \
            ort->ReleaseStatus(status); \
            return -1; \
        } \
    } while (0)
```
