# ONNX Runtime Reference - Chapter 8: Environment and Session Lifecycle

Complete reference for environment initialization, session creation, and model loading.

---

## 8.1 Environment (OrtEnv)

### 8.1.1 Environment Creation

The environment is a global singleton that must be created before any sessions.

**C++:**
```cpp
// Basic creation
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "my_application");

// With custom logger
Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "my_app",
             my_logging_function, my_logger_param);

// With global thread pools
Ort::ThreadingOptions tp;
tp.SetGlobalIntraOpNumThreads(4);
tp.SetGlobalInterOpNumThreads(1);
Ort::Env env(&tp, ORT_LOGGING_LEVEL_WARNING, "my_app", nullptr, nullptr);
```

**C:**
```c
OrtEnv* env;
ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "my_app", &env);

// With custom logger
ort->CreateEnvWithCustomLogger(logging_func, param,
    ORT_LOGGING_LEVEL_WARNING, "my_app", &env);
```

**Python:**
```python
# Environment is managed automatically by Python bindings
# No explicit Env creation needed
import onnxruntime as ort
```

### 8.1.2 Environment Configuration

```cpp
// Telemetry
env.EnableTelemetryEvents();
env.DisableTelemetryEvents();

// Language projection
env.SetLanguageProjection(ORT_PROJECTION_PYTHON);

// Register custom allocator
env.CreateAndRegisterAllocator(&mem_info, &arena_cfg);
```

---

## 8.2 Session Lifecycle

### 8.2.1 Session States

```
┌──────────────┐
│ Uninitialized │  Session object created but no model loaded
└──────┬───────┘
       │ Load(model_path)
       │ Load(model_bytes, size)
       ▼
┌──────────────┐
│   Loaded      │  Model parsed into Graph IR
└──────┬───────┘
       │ Initialize()
       │ (automatically called by constructor with model path)
       ▼
┌──────────────┐
│  Initialized  │  Graph optimized, EPs assigned, kernels registered
└──────┬───────┘
       │ Run()
       ▼
┌──────────────┐
│    Ready      │  Ready for repeated inference calls
└──────────────┘
```

### 8.2.2 Session Creation (C++)

```cpp
Ort::Env env;
Ort::SessionOptions opts;

// From file
Ort::Session session(env, L"model.onnx", opts);

// From bytes
std::ifstream file("model.onnx", std::ios::binary);
std::vector<char> data((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
Ort::Session session(env, data.data(), data.size(), opts);

// With pre-packed weights
Ort::PrepackedWeightsContainer prepacked;
Ort::Session session(env, L"model.onnx", opts, prepacked);
```

### 8.2.3 Session Creation (Python)

```python
import onnxruntime as ort

# From file
sess = ort.InferenceSession("model.onnx")

# From bytes
with open("model.onnx", "rb") as f:
    sess = ort.InferenceSession(f.read())

# With options and providers
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession("model.onnx", sess_options=opts,
                             providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
```

---

## 8.3 Model Loading Pipeline

### 8.3.1 Loading Stages

```
1. Parse Model
   ├── Read ONNX protobuf (or ORT FlatBuffers)
   ├── Validate model structure
   ├── Extract model metadata
   └── Build initial Graph IR

2. Optimize Graph
   ├── Level 1: Basic optimizations (constant folding, dead code elimination)
   ├── Level 2: Extended optimizations (operator fusion)
   ├── Level 3: Layout transformations (NCHW ↔ NHWC)
   └── Level 4: Full optimizations (QDQ, MatMulNBits)

3. Partition Graph
   ├── Query each EP for supported ops (GetCapability)
   ├── Assign nodes to EPs
   ├── CPU EP fallback for unsupported nodes
   └── Create IndexedSubGraph for each EP partition

4. Register Kernels
   ├── For each (op_type, EP) pair
   ├── Look up kernel in KernelRegistry
   ├── Create kernel instances
   └── Verify all ops have implementations

5. Memory Planning
   ├── Allocate buffers for intermediate tensors
   ├── Pre-pack constant initializers
   ├── Configure memory patterns
   └── Reserve workspace for each kernel
```

### 8.3.2 Model Format Detection

```python
# By file extension
"model.onnx" → ONNX protobuf format
"model.ort"  → ORT FlatBuffers format

# Force format
opts.add_session_config_entry("session.load_model_format", "ORT")
opts.add_session_config_entry("session.load_model_format", "ONNX")
```

### 8.3.3 External Data Files

```python
# Model with external data
# model.onnx + external_data.bin
sess = ort.InferenceSession("model.onnx")

# Specify external data folder
opts.add_session_config_entry(
    "session.model_external_initializers_file_folder_path",
    "/path/to/data/folder")
```

---

## 8.4 Multiple Sessions

### 8.4.1 Sharing Environment

```cpp
// All sessions share the same environment
Ort::Env env;

Ort::Session sess1(env, L"model_a.onnx", opts1);
Ort::Session sess2(env, L"model_b.onnx", opts2);

// Thread-safe: Run() can be called concurrently on different sessions
// Or even on the same session
```

### 8.4.2 Global Thread Pools

```cpp
// Share thread pools across sessions
Ort::ThreadingOptions tp;
tp.SetGlobalIntraOpNumThreads(8);
tp.SetGlobalInterOpNumThreads(2);

Ort::Env env(&tp, ORT_LOGGING_LEVEL_WARNING, "app");

// All sessions will use these global thread pools
Ort::SessionOptions opts;
opts.DisablePerSessionThreads();  // Required for global thread pools
Ort::Session session(env, L"model.onnx", opts);
```

---

## 8.5 Session Options Cloning

```cpp
// Deep clone session options
auto opts1 = Ort::SessionOptions();
opts1.SetIntraOpNumThreads(4);

auto opts2 = opts1.Clone();  // Deep copy
opts2.SetIntraOpNumThreads(8);  // Only affects opts2
```

---

## 8.6 EP Registration Order

```python
# EPs are registered in priority order
# First EP that supports an op gets the node
providers = [
    "CUDAExecutionProvider",       # Highest priority
    "TensorrtExecutionProvider",   # Second priority
    "CPUExecutionProvider",        # Fallback
]

# TensorRT will only get ops that CUDA doesn't support
# CPU gets anything neither supports
```

---

## 8.7 Session Destruction

```cpp
// C++: Automatic via RAII
{
    Ort::Session session(env, L"model.onnx", opts);
    // Use session...
}  // Session automatically released here

// C: Manual release
OrtReleaseSession(session);

// Important: Do NOT release session from DllMain on Windows
// Session owns thread pools that need clean shutdown
```

---

## 8.8 Pre-packed Weights Container

```cpp
// Create container for pre-packed weights
Ort::PrepackedWeightsContainer prepacked;

// First session pre-packs weights
auto session1 = Ort::Session(env, L"model.onnx", opts, prepacked);

// Second session reuses pre-packed weights (faster loading)
auto session2 = Ort::Session(env, L"model.onnx", opts, prepacked);
```

---

## 8.9 Model Metadata

```python
# Get model metadata
metadata = sess.get_modelmeta()

print(metadata.producer_name)        # e.g., "pytorch"
print(metadata.graph_name)           # Graph name
print(metadata.description)          # Model description
print(metadata.domain)               # Model domain
print(metadata.version)              # Model version
print(metadata.custom_metadata_map)  # Dict of custom metadata
```
