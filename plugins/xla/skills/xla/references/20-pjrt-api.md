# PJRT (Portable Joint Runtime) API

This document provides comprehensive documentation about the PJRT API, which provides a uniform device API for the machine learning ecosystem.

## Table of Contents

- [Overview](#overview)
- [PJRT C API](#pjrt-c-api)
- [PJRT Plugin Mechanism](#pjrt-plugin-mechanism)
- [PJRT Integration Guide](#pjrt-integration-guide)
- [Resources](#resources)

## Overview

### The Problem PJRT Solves

The machine learning ecosystem has a growing number of frameworks (JAX, TensorFlow, PyTorch, and others) and a growing number of hardware accelerators (NVIDIA GPUs, AMD GPUs, Google TPUs, Intel GPUs, and others). Without a standardized interface, each framework must implement and maintain separate support for each hardware device, creating an M x N integration problem.

### Vision

PJRT provides a uniform Device API that solves this problem with a two-part vision:

1. **Frameworks call PJRT**: ML frameworks (JAX, TensorFlow, PyTorch, etc.) target the PJRT API for all device interactions. This means a framework only needs one integration point to support all hardware devices that implement PJRT.

2. **Devices implement PJRT plugins**: Hardware vendors implement PJRT plugins that translate the standard PJRT API calls into device-specific operations. This means a hardware vendor only needs one integration point to be supported by all frameworks that use PJRT.

```
+----------+  +-------------+  +-----------+
|   JAX    |  | TensorFlow  |  |  PyTorch  |  ... (Frameworks)
+----+-----+  +------+------+  +-----+-----+
     |               |               |
     +---------------+---------------+
                     |
              +------+------+
              |   PJRT API  |       (Standard Interface)
              +------+------+
                     |
     +---------------+---------------+
     |               |               |
+----+-----+  +------+------+  +----+-----+
|  NVIDIA  |  |    AMD     |  |   TPU    |  ... (Devices)
|   GPU    |  |    GPU     |  |          |
+----------+  +------------+  +----------+
```

### Key Design Principles

- **C API with stable ABI**: PJRT is defined as a C API to ensure ABI stability across compilers and languages. This allows plugins compiled with different toolchains to interoperate.
- **Minimal surface area**: The API is designed to be as small as possible while remaining expressive enough for all supported use cases.
- **No runtime dependencies**: PJRT plugins do not depend on any particular ML framework at compile time.
- **Plugin architecture**: Hardware support is delivered as dynamically loaded plugins, not compiled into the framework.

## PJRT C API

### Header File Location

The PJRT C API is defined in the XLA repository:

```
xla/pjrt/c/pjrt_c_api.h
```

This header contains all the type definitions, function pointer types, and API entry points that constitute the PJRT interface.

### Core Types

The PJRT C API defines the following core types, all following a consistent naming convention:

```c
// Opaque device type
typedef struct PJRT_Device PJRT_Device;

// Opaque buffer type
typedef struct PJRT_Buffer PJRT_Buffer;

// Opaque client type (represents a connection to a device pool)
typedef struct PJRT_Client PJRT_Client;

// Opaque executable type
typedef struct PJRT_Executable PJRT_Executable;

// Opaque event type (for async operations)
typedef struct PJRT_Event PJRT_Event;

// Opaque loaded executable type
typedef struct PJRT_LoadedExecutable PJRT_LoadedExecutable;
```

### API Function Table

The PJRT API is accessed through a function table (`PJRT_Api`) that contains function pointers for all API operations. This structure is the central dispatch mechanism:

```c
typedef struct PJRT_Api {
  // Plugin information
  PJRT_Plugin_Initialize* plugin_initialize;
  PJRT_Plugin_Attributes* plugin_attributes;

  // Client operations
  PJRT_Client_Create* client_create;
  PJRT_Client_Destroy* client_destroy;
  PJRT_Client_Devices* client_devices;
  PJRT_Client_Addressable_devices* client_addressable_devices;
  PJRT_Client_Default_layout* client_default_layout;
  PJRT_Client_Buffer_from_host_buffer* client_buffer_from_host_buffer;
  PJRT_Client_Buffer_from_host_layout* client_buffer_from_host_layout;
  PJRT_Client_Compile* client_compile;

  // Device operations
  PJRT_Device_Id* device_id;
  PJRT_Device_Get_Attribute* device_get_attribute;
  PJRT_Device_Is_addressable* device_is_addressable;
  PJRT_Device_Local_hardware_id* device_local_hardware_id;

  // Buffer operations
  PJRT_Buffer_Destroy* buffer_destroy;
  PJRT_Buffer_Ready_event* buffer_ready_event;
  PJRT_Buffer_Delete* buffer_delete;
  PJRT_Buffer_Is_deleted* buffer_is_deleted;
  PJRT_Buffer_To_host_buffer* buffer_to_host_buffer;
  PJRT_Buffer_On_device_size_in_bytes* buffer_on_device_size_in_bytes;
  PJRT_Buffer_Unsafe_pointer* buffer_unsafe_pointer;
  PJRT_Buffer_Memory_kind* buffer_memory_kind;
  PJRT_Buffer_Device* buffer_device;
  PJRT_Buffer_ElementType* buffer_element_type;
  PJRT_Buffer_Dimensions* buffer_dimensions;
  PJRT_Buffer_Dynamic_dimensions* buffer_dynamic_dimensions;

  // Executable operations
  PJRT_LoadedExecutable_Destroy* loaded_executable_destroy;
  PJRT_LoadedExecutable_Execute* loaded_executable_execute;
  PJRT_LoadedExecutable_Get_executable* loaded_executable_get_executable;
  PJRT_LoadedExecutable_Addressable_devices* loaded_executable_addressable_devices;
  PJRT_Executable_Destroy* executable_destroy;
  PJRT_Executable_Size_of_generated_code_in_bytes* executable_size_of_generated_code_in_bytes;
  PJRT_Executable_Cost_analysis* executable_cost_analysis;
  PJRT_Executable_Output_dimensions* executable_output_dimensions;

  // Event operations
  PJRT_Event_Destroy* event_destroy;
  PJRT_Event_Is_ready* event_is_ready;
  PJRT_Event_Await* event_await;
  PJRT_Event_On_ready* event_on_ready;

  // Layout operations
  PJRT_Layout_Create* layout_create;
  PJRT_Layout_Destroy* layout_destroy;
  PJRT_Layout_Num_dimensions* layout_num_dimensions;
  PJRT_Layout_Minor_to_major* layout_minor_to_major;
  PJRT_Layout_Dynamic_dimensions* layout_dynamic_dimensions;
  PJRT_Layout_Tiling* layout_tiling;

  // Error handling
  PJRT_Error_Destroy* error_destroy;
  PJRT_Error_Message* error_message;
  PJRT_Error_Get_Code* error_get_code;

  // Named values for configuration
  PJRT_NamedValue* named_value;

  // ... additional function pointers ...
} PJRT_Api;
```

### ABI Versioning

PJRT uses a versioning system to ensure compatibility between frameworks and plugins:

```c
typedef struct PJRT_Api_Version {
  int major_version;
  int minor_version;
} PJRT_Api_Version;
```

The versioning follows these rules:

- **Major version changes**: Indicate breaking API changes. A framework with a different major version than the plugin may not be compatible.
- **Minor version changes**: Indicate backward-compatible additions. A framework can use a plugin with a higher minor version.

The plugin must report its supported version via:

```c
PJRT_Error* PJRT_Plugin_Initialize(PJRT_Api* api);
```

During initialization, the framework and plugin negotiate a compatible version.

### Stability Guarantees

PJRT provides the following stability guarantees:

1. **ABI Stability**: Once a function signature is published in a stable version, it will not change in a way that breaks binary compatibility. New parameters are added through extension mechanisms, not by modifying existing structures.

2. **API Stability**: The semantics of existing API calls will not change in breaking ways. Optional behavior may be refined, but the core contract remains.

3. **Extension Mechanism**: New features are added through:
   - New function pointers appended to the `PJRT_Api` structure.
   - New attribute APIs that allow plugins to expose additional capabilities.
   - New named values for configuration options.

4. **Error Handling**: All API functions return `PJRT_Error*` (NULL on success). Errors must be explicitly destroyed using `PJRT_Error_Destroy`.

## PJRT Plugin Mechanism

### Dynamic Library Loading

PJRT plugins are distributed as shared libraries (`.so` on Linux, `.dylib` on macOS, `.dll` on Windows). The framework discovers and loads plugins at runtime.

The plugin loading process follows these steps:

1. **Discovery**: The framework searches for plugins in standard locations and paths specified by environment variables:
   - `PJRT_PLUGIN_LIBRARY_PATH`: Direct path to the plugin shared library.
   - Framework-specific configuration (e.g., JAX plugin configuration).

2. **Loading**: The framework uses `dlopen` (or equivalent) to load the shared library.

3. **Symbol Resolution**: The framework looks for the initialization symbol:
   ```c
   // The plugin must export this symbol
   PJRT_Error* PJRT_Plugin_Initialize(PJRT_Api* api);
   ```

4. **Initialization**: The framework calls `PJRT_Plugin_Initialize`, passing a `PJRT_Api` structure. The plugin fills in the function pointers.

### Plugin Registration

A minimal plugin registration looks like this:

```c
// my_plugin.cc

#include "xla/pjrt/c/pjrt_c_api.h"

namespace my_plugin {

// Forward declarations of implementations
PJRT_Error* ClientCreate(PJRT_Client_Create_Args* args);
PJRT_Error* ClientDestroy(PJRT_Client_Destroy_Args* args);
// ... other implementations ...

PJRT_Error* PluginInitialize(PJRT_Api* api) {
  // Set version information
  api->pjrt_api_version.major_version = PJRT_API_MAJOR_VERSION;
  api->pjrt_api_version.minor_version = PJRT_API_MINOR_VERSION;

  // Register all implemented functions
  api->client_create = ClientCreate;
  api->client_destroy = ClientDestroy;
  // ... register all function pointers ...

  return nullptr;  // Success
}

}  // namespace my_plugin

// Export the initialization symbol
extern "C" {
  PJRT_Error* PJRT_Plugin_Initialize(PJRT_Api* api) {
    return my_plugin::PluginInitialize(api);
  }
}
```

### Plugin Initialization

The `PJRT_Plugin_Initialize` function is the entry point for the plugin. Its responsibilities include:

1. **Version negotiation**: Check that the framework's API version is compatible with the plugin's implementation.

2. **Function pointer registration**: Fill in all required function pointers in the `PJRT_Api` structure.

3. **Optional feature reporting**: Set any optional function pointers to NULL if the feature is not supported by the plugin.

4. **Global initialization**: Perform any one-time initialization needed by the plugin (e.g., loading device drivers, allocating global resources).

The initialization function receives a pre-allocated `PJRT_Api` structure from the framework and must fill in the function pointers. Any function pointers left as NULL will cause a "not implemented" error if the framework tries to call them.

### C++ Wrapper Library

XLA provides a C++ wrapper library that makes it easier to implement PJRT plugins. This wrapper handles the C-to-C++ bridging:

```cpp
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"

// The wrapper provides C++ abstractions:
// - PJRTClient: wraps PJRT_Client operations
// - PJRTBuffer: wraps PJRT_Buffer operations
// - PJRTLoadedExecutable: wraps PJRT_LoadedExecutable operations
// etc.
```

Using the C++ wrapper is strongly recommended as it handles many of the low-level details of the C API, including memory management, error handling, and type conversions.

## PJRT Integration Guide

### Steps to Integrate PJRT with a New Framework

If you are developing an ML framework and want to use PJRT to support multiple hardware devices:

#### Step 1: Add PJRT as a Dependency

Add the PJRT C API headers to your project. These are available in the XLA repository under `xla/pjrt/c/`.

#### Step 2: Implement Plugin Discovery

Implement a plugin loader that can discover and load PJRT plugins:

```python
# Python example using ctypes
import ctypes
import os

def load_pjrt_plugin(plugin_path):
    """Load a PJRT plugin from a shared library."""
    library = ctypes.CDLL(plugin_path)

    # Get the initialization function
    init_fn = library.PJRT_Plugin_Initialize

    # Create the API structure (or use the framework's pre-built one)
    # ... framework-specific initialization ...

    return library
```

#### Step 3: Implement the Client Interface

Create a client abstraction that uses PJRT's client operations:

```python
class PJRTClient:
    def __init__(self, plugin_api):
        self.api = plugin_api
        # Initialize the PJRT client
        self.api.client_create(...)

    def devices(self):
        """Get available devices."""
        return self.api.client_devices(...)

    def compile(self, program):
        """Compile a program to an executable."""
        return self.api.client_compile(...)

    def buffer_from_host(self, data, shape):
        """Create a device buffer from host data."""
        return self.api.client_buffer_from_host_buffer(...)
```

#### Step 4: Implement Buffer Management

Use PJRT's buffer operations to manage device memory:

```python
class PJRTBuffer:
    def __init__(self, api, buffer_ptr):
        self.api = api
        self.buffer = buffer_ptr

    def to_host(self):
        """Copy buffer data to host."""
        self.api.buffer_to_host_buffer(...)

    def delete(self):
        """Delete the device buffer."""
        self.api.buffer_delete(...)

    def ready(self):
        """Wait for the buffer to be ready."""
        event = self.api.buffer_ready_event(...)
        self.api.event_await(event)
```

#### Step 5: Implement Execution

Use PJRT's execution operations to run compiled programs:

```python
class PJRTLoadedExecutable:
    def __init__(self, api, executable_ptr):
        self.api = api
        self.executable = executable_ptr

    def execute(self, inputs):
        """Execute with the given input buffers."""
        return self.api.loaded_executable_execute(...)

    def destroy(self):
        """Destroy the loaded executable."""
        self.api.loaded_executable_destroy(...)
```

#### Step 6: Handle Errors

Implement error handling using PJRT's error API:

```python
def check_error(api, error_ptr):
    """Check a PJRT error and raise a Python exception if needed."""
    if error_ptr is None:
        return  # Success

    # Get error message
    message = api.error_message(error_ptr)
    code = api.error_get_code(error_ptr)

    # Destroy the error
    api.error_destroy(error_ptr)

    raise RuntimeError(f"PJRT error (code {code}): {message}")
```

### Steps to Implement a PJRT Plugin for New Hardware

If you are a hardware vendor and want your device to be supported by PJRT-using frameworks:

#### Step 1: Set Up the Plugin Project

Create a new project that will compile into a shared library:

```
my_pjrt_plugin/
  BUILD
  my_plugin.cc         # Plugin entry point
  my_client.cc         # Client implementation
  my_buffer.cc         # Buffer implementation
  my_executable.cc     # Executable implementation
  my_device.cc         # Device implementation
```

#### Step 2: Implement Device Abstraction

Create a device representation that implements the PJRT device interface:

```cpp
class MyDevice {
 public:
  int id() const;
  std::string vendor() const;
  std::string device_kind() const;
  bool is_addressable() const;
  int local_hardware_id() const;
};
```

#### Step 3: Implement Client

The client manages the connection to the device pool:

```cpp
class MyClient {
 public:
  static StatusOr<std::unique_ptr<MyClient>> Create(
      const PJRT_Client_Create_Args& args);

  std::vector<MyDevice*> devices();
  std::vector<MyDevice*> addressable_devices();
  StatusOr<std::unique_ptr<MyBuffer>> BufferFromHostBuffer(
      const void* data, PJRT_Buffer_Type type,
      const int64_t* dims, int num_dims,
      PJRT_HostBufferSemantics semantics);
  StatusOr<std::unique_ptr<MyLoadedExecutable>> Compile(
      const PJRT_Client_Compile_Args& args);
};
```

#### Step 4: Implement Buffer

The buffer represents device-resident data:

```cpp
class MyBuffer {
 public:
  PJRT_Buffer_Type element_type() const;
  std::vector<int64_t> dimensions() const;
  int64_t on_device_size_in_bytes() const;
  void* unsafe_pointer() const;
  MyDevice* device() const;

  StatusOr<std::unique_ptr<MyEvent>> ToHostBuffer(void* dst, int64_t size);
  Status Delete();
  bool IsDeleted() const;
};
```

#### Step 5: Implement Executable

The executable wraps a compiled computation:

```cpp
class MyLoadedExecutable {
 public:
  StatusOr<std::vector<std::vector<std::unique_ptr<MyBuffer>>>> Execute(
      absl::Span<std::vector<MyBuffer*>> arguments,
      PJRT_ExecutionOptions* options);

  std::vector<MyDevice*> addressable_devices() const;
};
```

#### Step 6: Implement the C API Bridge

Connect your C++ implementation to the C API:

```cpp
// Each C API function follows this pattern:
PJRT_Error* MyClientCreate(PJRT_Client_Create_Args* args) {
  auto status_or = MyClient::Create(*args);
  if (!status_or.ok()) {
    return CreatePjrtError(status_or.status());
  }
  args->client = WrapClient(std::move(status_or).value());
  return nullptr;
}
```

#### Step 7: Test the Plugin

Use the PJRT C API test suite to validate your implementation:

```bash
# Run PJRT conformance tests
bazel test //xla/pjrt/c:test_pjrt_c_api_client --test_arg=--plugin_path=/path/to/plugin.so
```

#### Step 8: Package and Distribute

Package your plugin as a Python wheel or system package:

```python
# setup.py for a Python wheel
from setuptools import setup, Distribution

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(
    name="my-pjrt-plugin",
    version="0.1.0",
    packages=["my_pjrt_plugin"],
    package_data={
        "my_pjrt_plugin": ["libmy_plugin.so"],
    },
    distclass=BinaryDistribution,
)
```

Users can then install the plugin and configure their framework:

```python
# JAX configuration
import jax
jax.config.update("jax_platforms", "my_plugin")

# Or via environment variable
# JAX_PLATFORMS=my_plugin python my_script.py
```

## Resources

### Official Documentation

- **C API Header**: `xla/pjrt/c/pjrt_c_api.h` - The authoritative API definition.
- **PJRT Changelog**: Documents all API changes across versions.
- **PJRT Integration Guide**: Step-by-step guide for framework integration.
- **PJRT Design Doc**: High-level design rationale and architecture.

### Design Documents

- **PJRT Plugin Mechanism Design Doc**: Describes the dynamic loading, registration, and version negotiation mechanisms.
- **PJRT Tutorial Slides**: Presentation slides covering the basics of PJRT integration.

### Example Implementations

- **CUDA PJRT Plugin**: `xla/pjrt/plugin/` - Reference implementation for NVIDIA GPUs.
- **TPU PJRT Plugin**: Google's TPU implementation (available in libtpu).
- **Intel GPU PJRT Plugin**: Intel's implementation for GPU devices (available in the IPEX project).
- **AMD GPU PJRT Plugin**: AMD's implementation for ROCm GPUs.

### Community

- **OpenXLA Discussion Forum**: For questions and discussions about PJRT.
- **GitHub Issues**: For bug reports and feature requests.
- **OpenXLA Discord**: Real-time community discussion.

### Compatibility Matrix

The following frameworks support PJRT:

| Framework | PJRT Support | Notes |
|-----------|-------------|-------|
| JAX | Full | Primary PJRT consumer |
| TensorFlow | Full | Via tf.function and XLA |
| PyTorch/XLA | Full | Via torch_xla |
| JAX (NumPy) | Full | Via JAX |

The following hardware devices have PJRT plugins:

| Device | Plugin | Status |
|--------|--------|--------|
| NVIDIA GPU | xla/pjrt/plugin | Stable |
| Google TPU | libtpu | Stable |
| AMD GPU | JAX-Plugins | Stable |
| Intel GPU | IPEX-Plugin | Stable |
| AWS Trainium/Inferentia | neuronx-cc | Available |
