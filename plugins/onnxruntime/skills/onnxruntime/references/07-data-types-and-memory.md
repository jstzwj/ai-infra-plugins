# ONNX Runtime Reference - Chapter 7: Data Types and Memory System

Complete reference for tensor element types, memory info, allocators, and memory management.

---

## 7.1 Tensor Element Data Types

### 7.1.1 Complete Type Table

| Enum Value | Name | Size (bytes) | C Type | Numpy Type |
|------------|------|-------------|--------|-----------|
| 0 | UNDEFINED | - | - | - |
| 1 | FLOAT | 4 | float | float32 |
| 2 | UINT8 | 1 | uint8_t | uint8 |
| 3 | INT8 | 1 | int8_t | int8 |
| 4 | UINT16 | 2 | uint16_t | uint16 |
| 5 | INT16 | 2 | int16_t | int16 |
| 6 | INT32 | 4 | int32_t | int32 |
| 7 | INT64 | 8 | int64_t | int64 |
| 8 | STRING | variable | std::string | object |
| 9 | BOOL | 1 | bool | bool |
| 10 | FLOAT16 | 2 | IEEE 754 half | float16 |
| 11 | DOUBLE | 8 | double | float64 |
| 12 | UINT32 | 4 | uint32_t | uint32 |
| 13 | UINT64 | 8 | uint64_t | uint64 |
| 14 | COMPLEX64 | 8 | complex<float> | complex64 |
| 15 | COMPLEX128 | 16 | complex<double> | complex128 |
| 16 | BFLOAT16 | 2 | bfloat16 | - |
| 17 | FLOAT8E4M3FN | 1 | float8_e4m3fn | - |
| 18 | FLOAT8E4M3FNUZ | 1 | float8_e4m3fnuz | - |
| 19 | FLOAT8E5M2 | 1 | float8_e5m2 | - |
| 20 | FLOAT8E5M2FNUZ | 1 | float8_e5m2fnuz | - |
| 21 | UINT4 | 0.5 | packed uint4 | - |
| 22 | INT4 | 0.5 | packed int4 | - |
| 23 | FLOAT4E2M1 | 0.5 | packed float4 | - |
| 24 | UINT2 | 0.25 | packed uint2 | - |
| 25 | INT2 | 0.25 | packed int2 | - |

### 7.1.2 Sub-byte Types

Sub-byte types (UINT4, INT4, FLOAT4E2M1, UINT2, INT2) pack multiple values into a single byte:
- UINT4/INT4: 2 values per byte
- UINT2/INT2: 4 values per byte
- FLOAT4E2M1: 2 values per byte

---

## 7.2 ONNX Type System

```c
typedef enum ONNXType {
    ONNX_TYPE_UNKNOWN,      // Unknown type
    ONNX_TYPE_TENSOR,       // Dense tensor
    ONNX_TYPE_SEQUENCE,     // Sequence of values
    ONNX_TYPE_MAP,          // Key-value map
    ONNX_TYPE_OPAQUE,       // Custom opaque type
    ONNX_TYPE_SPARSETENSOR, // Sparse tensor
    ONNX_TYPE_OPTIONAL      // Optional value
} ONNXType;
```

### 7.2.1 Internal Type System (C++)

```cpp
// MLDataType - type erasure for ORT types
class DataTypeImpl {
    // Factory methods
    template <typename T>
    static MLDataType GetType();

    // Check types
    static MLDataType GetType<Tensor>();
    static MLDataType GetType<SparseTensor>();
    static MLDataType GetType<TensorSeq>();
    static MLDataType GetType<MapStringToString>();
    static MLDataType GetType<MapStringToInt64>();
    static MLDataType GetType<MapStringToFloat>();
    // etc.
};
```

---

## 7.3 Memory Information

### 7.3.1 OrtMemoryInfo

```c
typedef struct OrtMemoryInfo {
    const char* name;           // Allocator name (e.g., "Cpu", "Cuda", "CudaPinned")
    OrtAllocatorType type;      // Allocator type
    int id;                     // Device ID
    OrtMemType mem_type;        // Memory type
} OrtMemoryInfo;
```

### 7.3.2 C++ MemoryInfo

```cpp
// Create CPU memory info
auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
auto mem_info_pinned = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

// Create custom memory info
auto gpu_mem = Ort::MemoryInfo::Create("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
```

### 7.3.3 Python MemoryInfo

```python
import onnxruntime as ort

# CPU memory info
cpu_mem = ort.OrtMemoryInfo.create_cpu(ort.OrtArenaAllocator, ort.OrtMemTypeDefault)
```

### 7.3.4 Memory Type Constants

```c
typedef enum OrtMemType {
    OrtMemTypeCPUInput  = -2,  // CPU memory for non-CPU EP inputs
    OrtMemTypeCPUOutput = -1,  // CPU accessible memory for outputs
    OrtMemTypeCPU       = -1,  // Same as CPUOutput
    OrtMemTypeDefault   = 0,   // Default allocator for EP
} OrtMemType;

typedef enum OrtAllocatorType {
    OrtInvalidAllocator  = -1,
    OrtDeviceAllocator   = 0,  // Direct device allocation
    OrtArenaAllocator    = 1,  // Arena-based allocation
    OrtReadOnlyAllocator = 2,  // Read-only memory
} OrtAllocatorType;
```

---

## 7.4 Device Types

### 7.4.1 OrtDevice

```cpp
class OrtDevice {
    enum DeviceType {
        CPU  = 0,
        GPU  = 1,
        FPGA = 2,
        NPU  = 3,
    };

    enum MemoryType {
        DEFAULT          = 0,  // Device-local memory
        HOST_ACCESSIBLE  = 5,  // Host-visible memory (pinned)
    };

    OrtDevice(DeviceType type, MemoryType mem_type, int device_id);

    DeviceType Type() const;
    MemoryType MemoryType() const;
    int Id() const;
};
```

### 7.4.2 Hardware Device Types

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

---

## 7.5 Allocator System

### 7.5.1 IAllocator Interface (C++)

```cpp
class IAllocator {
public:
    virtual void* Alloc(size_t size) = 0;
    virtual void Free(void* p) = 0;
    virtual const OrtMemoryInfo& Info() const = 0;

    // Optional: stream-aware allocation
    virtual void* AllocOnStream(size_t size, Stream* stream);

    // Optional: release unused memory
    virtual void Shrink();

    // Optional: get statistics
    virtual std::unordered_map<std::string, size_t> GetStats() const;

    // Optional: reservation for session init
    virtual void* Reserve(size_t size);

    bool IsReserve() const;
    void SetTrackAllocator(bool track);
};
```

### 7.5.2 CPUAllocator

```cpp
class CPUAllocator : public IAllocator {
public:
    explicit CPUAllocator(const OrtMemoryInfo& info);

    void* Alloc(size_t size) override;   // Uses malloc/new
    void Free(void* p) override;          // Uses free/delete

    static AllocatorPtr DefaultInstance();
};
```

### 7.5.3 BFCArena (Best-Fit with Coalescing)

```cpp
class BFCArena : public IArena {
public:
    struct ArenaConfig {
        size_t max_memory;                  // Max memory to use
        ArenaExtendStrategy extend_strategy; // How to extend arena
        size_t initial_chunk_size_bytes;    // Initial chunk size
        size_t max_dead_bytes_per_chunk;    // Max dead bytes per chunk
    };

    void* Alloc(size_t size) override;
    void Free(void* p) override;
    void Shrink() override;  // Release unused chunks

    size_t Used() const;
    size_t Available() const;
    size_t Allocated() const;
};
```

**Arena Extend Strategies:**
```python
# kNextPowerOfTwo (default) - extend to next power of 2
"kNextPowerOfTwo"

# kSameAsRequested - extend by exactly the requested size
"kSameAsRequested"
```

### 7.5.4 StreamAwareBFCArena

```cpp
class StreamAwareBFCArena : public BFCArena {
public:
    void* AllocOnStream(size_t size, onnxruntime::Stream* stream);
    // Stream-aware allocation for GPU EPs
};
```

---

## 7.6 Arena Configuration

### 7.6.1 C++ ArenaCfg

```cpp
auto arena_cfg = Ort::ArenaCfg(
    max_memory,                  // Max memory in bytes (0 = unlimited)
    arena_extension_strategy,    // 0 = kNextPowerOfTwo, 1 = kSameAsRequested
    initial_chunk_size_bytes,    // Initial chunk size
    max_dead_bytes_per_chunk     // Max dead bytes per chunk
);
```

### 7.6.2 CUDA Arena Options

```python
providers = [
    ("CUDAExecutionProvider", {
        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB limit
        "arena_extend_strategy": "kNextPowerOfTwo",
    }),
]
```

---

## 7.7 Memory Statistics

### 7.7.1 Allocator Statistics Keys

| Key | Description |
|-----|-------------|
| `Limit` | Memory limit in bytes (-1 = no limit) |
| `InUse` | Currently used bytes |
| `TotalAllocated` | Total allocated bytes |
| `MaxInUse` | Maximum bytes ever in use |
| `NumAllocs` | Number of allocations |
| `NumReserves` | Number of reserve calls |
| `NumArenaExtensions` | Arena extension count |
| `NumArenaShrinkages` | Arena shrinkage count |
| `MaxAllocSize` | Largest single allocation |

---

## 7.8 Memory Patterns

### 7.8.1 Memory Pattern Optimization

```python
# Enable memory patterns (default)
opts = ort.SessionOptions()
opts.enable_mem_pattern = True

# ORT pre-allocates memory based on observed allocation patterns
# This reduces allocation overhead during inference
```

### 7.8.2 Pre-packed Weights

```python
# Pre-pack constant weights for faster kernel execution
opts = ort.SessionOptions()
# Pre-packing is enabled by default
opts.add_session_config_entry("session.disable_prepacking", "0")

# Save pre-packed weights externally
opts.add_session_config_entry(
    "session.save_external_prepacked_constant_initializers", "1")
```

---

## 7.9 Memory Planning

### 7.9.1 Memory Reuse

```python
opts = ort.SessionOptions()
opts.enable_mem_reuse = True  # Default: True

# ORT reuses memory buffers across operations when possible
# Reduces peak memory usage
```

### 7.9.2 Memory-Mapped Models

```python
# Use memory-mapped I/O for ORT format models
opts = ort.SessionOptions()
opts.add_session_config_entry("session.use_memory_mapped_ort_model", "1")
opts.add_session_config_entry("session.use_ort_model_bytes_for_initializers", "1")
```

### 7.9.3 Node Memory Statistics

```python
# Collect per-node memory statistics
opts.add_session_config_entry(
    "session.collect_node_memory_stats_to_file",
    "/path/to/memory_stats.csv")
```
