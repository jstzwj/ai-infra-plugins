# ONNX Runtime Reference - Chapter 9: Error Handling and Logging

Complete reference for error handling, logging, profiling, and debugging.

---

## 9.1 Error Handling

### 9.1.1 Error Codes

```c
typedef enum OrtErrorCode {
    ORT_OK,                    // Success
    ORT_FAIL,                  // Generic failure
    ORT_INVALID_ARGUMENT,      // Invalid argument passed
    ORT_NO_SUCHFILE,           // File not found
    ORT_NO_MODEL,              // No model loaded
    ORT_ENGINE_ERROR,          // Execution engine error
    ORT_RUNTIME_EXCEPTION,     // Runtime exception
    ORT_INVALID_PROTOBUF,      // Invalid protobuf data
    ORT_MODEL_LOADED,          // Model already loaded
    ORT_NOT_IMPLEMENTED,       // Feature not implemented
    ORT_INVALID_GRAPH,         // Invalid graph structure
    ORT_EP_FAIL,               // Execution provider failure
    ORT_MODEL_LOAD_CANCELED,   // Model loading was canceled
    ORT_MODEL_REQUIRES_COMPILATION, // Model needs compilation
    ORT_NOT_FOUND,             // Resource not found
} OrtErrorCode;
```

### 9.1.2 C API Error Handling

```c
// All functions that can fail return OrtStatus*
// nullptr = success
OrtStatus* status = ort->CreateSession(env, model_path, opts, &session);
if (status != nullptr) {
    OrtErrorCode code = ort->GetErrorCode(status);
    const char* msg = ort->GetErrorMessage(status);
    fprintf(stderr, "Error [%d]: %s\n", code, msg);
    ort->ReleaseStatus(status);
    return -1;
}
```

### 9.1.3 C++ Exception Handling

```cpp
try {
    auto session = Ort::Session(env, L"model.onnx", opts);
    auto outputs = session.Run(run_opts, input_names, &input, 1, output_names, 1);
} catch (const Ort::Exception& e) {
    std::cerr << "ORT Error [" << e.GetOrtErrorCode() << "]: " << e.what() << std::endl;
    // Error codes: ORT_FAIL, ORT_INVALID_ARGUMENT, ORT_NO_SUCHFILE, etc.
}
```

### 9.1.4 Python Error Handling

```python
try:
    sess = ort.InferenceSession("model.onnx")
    results = sess.run(None, {"input": input_data})
except ort.RuntimeException as e:
    print(f"ORT Error: {e}")
except Exception as e:
    print(f"General error: {e}")
```

### 9.1.5 Internal Error Macros (C++)

```cpp
// Status-based error handling
ORT_RETURN_IF_ERROR(expr);      // Return non-OK Status if expr fails
ORT_THROW_IF_ERROR(expr);       // Throw if expr returns non-OK Status

// Conditional returns
ORT_RETURN_IF(condition, message);      // Return FAIL Status if condition is true
ORT_RETURN_IF_NOT(condition, message);  // Return FAIL Status if condition is false

// Assert-like
ORT_ENFORCE(condition, ...);    // Throw OnnxRuntimeException if condition is false

// Create Status
Status s = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid shape: ", shape);
```

---

## 9.2 Logging System

### 9.2.1 Logging Severity Levels

```c
typedef enum OrtLoggingLevel {
    ORT_LOGGING_LEVEL_VERBOSE,  // Most verbose
    ORT_LOGGING_LEVEL_INFO,
    ORT_LOGGING_LEVEL_WARNING,
    ORT_LOGGING_LEVEL_ERROR,
    ORT_LOGGING_LEVEL_FATAL,    // Least verbose
} OrtLoggingLevel;
```

### 9.2.2 C++ Logging

```cpp
// Environment-level logging
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "my_app");

// Session-level logging
Ort::SessionOptions opts;
opts.SetSessionLogId("my_session");
opts.SetSessionLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);
opts.SetSessionLogVerbosityLevel(0);

// Internal C++ logging macros
LOGS(logger, INFO) << "Processing node: " << node_name;
LOGS(logger, WARNING) << "Unsupported op type: " << op_type;
LOGS(logger, ERROR) << "Kernel execution failed";
LOGS_DEFAULT(INFO) << "Default logger message";
```

### 9.2.3 Custom Logger

```cpp
// C++ custom logger
void my_logger(void* param, OrtLoggingLevel severity,
               const char* category, const char* logid,
               const char* code_location, const char* message) {
    std::cout << "[" << logid << "] " << severity
              << " " << code_location << ": " << message << std::endl;
}

Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "my_app", my_logger, nullptr);
```

```c
// C custom logger
void MyLoggingFunction(void* param, OrtLoggingLevel severity,
                       const char* category, const char* logid,
                       const char* code_location, const char* message) {
    fprintf(stderr, "[%s][%d] %s: %s\n", logid, severity, code_location, message);
}

ort->CreateEnvWithCustomLogger(MyLoggingFunction, NULL,
    ORT_LOGGING_LEVEL_WARNING, "my_app", &env);
```

### 9.2.4 Python Logging

```python
import onnxruntime as ort

# Set default logger severity
ort.set_default_logger_severity(2)  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal

# Session-level logging
opts = ort.SessionOptions()
opts.session_log_severity_level = 2
opts.session_log_verbosity_level = 0
```

---

## 9.3 Profiling

### 9.3.1 Enabling Profiling

```python
# Python
opts = ort.SessionOptions()
opts.enable_profiling = True
opts.profile_file_prefix = "ort_profile"

sess = ort.InferenceSession("model.onnx", sess_options=opts)
results = sess.run(None, {"input": input_array})

# End profiling and get file path
profile_file = sess.end_profiling()
print(f"Profile saved to: {profile_file}")
```

```cpp
// C++
Ort::SessionOptions opts;
opts.EnableProfiling(L"ort_profile");

auto session = Ort::Session(env, L"model.onnx", opts);
auto outputs = session.Run(run_opts, input_names, &input, 1, output_names, 1);

auto profile_path = session.EndProfiling(allocator);
```

### 9.3.2 Profile Format

Profiles are saved in Chrome Trace Event format (`.json`):
- Load in Chrome: `chrome://tracing`
- Shows timeline of all operator executions
- Includes kernel execution time, memory allocation, data transfers

### 9.3.3 Run-Level Timing

```python
import time

# Measure inference time
start = time.perf_counter()
results = sess.run(None, {"input": input_array})
elapsed = time.perf_counter() - start
print(f"Inference time: {elapsed * 1000:.2f} ms")

# With warmup
for _ in range(10):
    sess.run(None, {"input": input_array})

# Timed runs
times = []
for _ in range(100):
    start = time.perf_counter()
    sess.run(None, {"input": input_array})
    times.append(time.perf_counter() - start)

import numpy as np
print(f"Mean: {np.mean(times)*1000:.2f} ms")
print(f"P50: {np.percentile(times, 50)*1000:.2f} ms")
print(f"P99: {np.percentile(times, 99)*1000:.2f} ms")
```

---

## 9.4 Thread Safety

### 9.4.1 Thread Safety Guarantees

- **InferenceSession::Run()**: Thread-safe. Multiple threads can call Run() concurrently.
- **SessionOptions**: Not thread-safe. Configure before creating sessions.
- **Environment**: Thread-safe after creation.
- **OrtValue**: Not thread-safe for writes. Read-only access is safe.

### 9.4.2 C API Exception Boundary

```cpp
// C API functions must not propagate C++ exceptions
// Use API_IMPL_BEGIN / API_IMPL_END macros:

API_IMPL_BEGIN()
    // C++ code that might throw
    auto result = session->Run(...);
API_IMPL_END()

// These macros catch all C++ exceptions and convert them to OrtStatus*
```

---

## 9.5 Debugging

### 9.5.1 Debug Node Inputs/Outputs

```cmake
# CMake option
set(onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS ON)
```

### 9.5.2 Tensor Dumping

```cmake
# CMake option
set(onnxruntime_DUMP_TENSOR ON)
```

### 9.5.3 Layout Transform Debugging

```python
# Dump model after layout transformation steps
opts.add_session_config_entry("session.debug_layout_transformation", "1")
```

### 9.5.4 Strict Shape/Type Inference

```python
# Fail on any shape/type inconsistency
opts.add_session_config_entry("session.strict_shape_type_inference", "1")
```

### 9.5.5 Disable Specific Optimizers for Debugging

```python
# Disable named optimizers to isolate issues
opts.add_session_config_entry("optimization.disable_specified_optimizers",
    "ConvBNFusion,GemmActivationFusion")
```

### 9.5.6 Save Optimized Model for Inspection

```python
# Save the optimized model to inspect graph transformations
opts.optimized_model_filepath = "debug_optimized.onnx"

# Save in ORT format
opts.add_session_config_entry("session.save_model_format", "ORT")
opts.optimized_model_filepath = "debug_optimized.ort"
```
