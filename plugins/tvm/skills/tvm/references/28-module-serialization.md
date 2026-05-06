# Apache TVM — Chapter 28: Module Serialization and Export

This reference covers module serialization and export in TVM — the process of converting compiled IRModules into deployable artifacts and loading them for execution. TVM supports multiple export formats and deployment scenarios ranging from Python-based development to bare-metal embedded systems.

---

## 28.1 Export Workflow

### Overview

The export workflow transforms a compiled TVM IRModule into a format suitable for deployment:

```
IRModule (after optimization)
        |
        v
   relax.build() or tvm.build()
        |
        v
   Executable / runtime.Module
        |
        v
   export_library()  ---->  Shared library (.so / .dll / .dylib)
        |                        |
        v                        v
   save_param_dict()         Parameter file (.params / .bin)
        |
        v
   Deploy to target device
```

### Complete Build-and-Export Example

```python
import tvm
from tvm import relax
import numpy as np

# Step 1: Obtain or create an IRModule
# (from model import, TVMScript, etc.)
mod = relax.get_pipeline("zero")(imported_mod)

# Step 2: Build the module for a specific target
target = tvm.target.Target("cuda", host="llvm")
exec = relax.build(mod, target=target)

# Step 3: Export the compiled module
exec.export_library("model.so")

# Step 4: Export parameters separately (if needed)
params = {name: tvm.nd.array(arr) for name, arr in model_params.items()}
with open("model.params", "wb") as f:
    f.write(tvm.runtime.save_param_dict(params))

# Step 5: On the deployment side, load and run
loaded_mod = tvm.runtime.load_module("model.so")
loaded_params = tvm.runtime.load_param_dict(open("model.params", "rb").read())
dev = tvm.cuda(0)
vm = relax.VirtualMachine(loaded_mod, dev)

# Set parameters
vm["set_constants"](loaded_params)

# Run inference
input_data = tvm.nd.array(np.random.randn(1, 3, 224, 224).astype("float32"), dev)
output = vm["main"](input_data)
```

---

## 28.2 Export APIs

### exec.export_library(path)

The primary export method that serializes a compiled module into a shared library file.

```python
import tvm
from tvm import relax

# Build
target = tvm.target.Target("llvm")
exec = relax.build(mod, target=target)

# Export as shared library
exec.export_library("model.so")           # Linux
exec.export_library("model.dll")          # Windows
exec.export_library("model.dylib")        # macOS

# The file extension determines the output format:
# .so   -> ELF shared object (Linux)
# .dll  -> PE dynamic library (Windows)
# .dylib -> Mach-O dynamic library (macOS)
```

#### Advanced Export Options

```python
# Export with additional linked libraries
exec.export_library(
    "model.so",
    # Add custom cc options
    cc="gcc",
    # Additional object files to link
    addons=["custom_kernel.o"],
    # Additional libraries to include
    libs=[tvm.get_global_func("runtime.DNNLLibrary")()],
)

# Export for a specific platform (cross-compilation)
exec.export_library(
    "model_arm.so",
    cc="aarch64-linux-gnu-gcc",  # Cross-compiler
)

# Export with debug symbols
exec.export_library(
    "model_debug.so",
    opts=["-g", "-O0"],  # Compiler flags
)
```

#### How export_library Works

The export process involves these steps:

1. **Code generation**: The target codegen (LLVM, CUDA C, etc.) translates the IRModule into machine code.

2. **Object file creation**: Machine code is written to a temporary object file (`.o`).

3. **Device code embedding**: GPU kernels (CUDA, OpenCL, etc.) are embedded into the host object file as data sections.

4. **Linking**: The object file is linked into a shared library using the system compiler (gcc, clang, MSVC).

5. **Metadata embedding**: Function names, type information, and other metadata are included in the library.

### tvm.runtime.save_param_dict(params)

Serializes a dictionary of named parameters (tensors) into a binary byte string:

```python
import tvm
import numpy as np

# Create parameter dictionary
params = {
    "weight1": tvm.nd.array(np.random.randn(128, 64).astype("float32")),
    "bias1": tvm.nd.array(np.random.randn(64).astype("float32")),
    "weight2": tvm.nd.array(np.random.randn(64, 10).astype("float32")),
    "bias2": tvm.nd.array(np.random.randn(10).astype("float32")),
}

# Serialize to binary
binary_data = tvm.runtime.save_param_dict(params)

# Save to file
with open("model.params", "wb") as f:
    f.write(binary_data)

# The binary format includes:
# - Magic number for validation
# - Number of parameters
# - For each parameter:
#   - Name (string)
#   - Shape (list of ints)
#   - Dtype (string)
#   - Data (raw bytes)
```

### tvm.runtime.load_param_dict(path)

Deserializes parameters from a binary byte string:

```python
import tvm

# Load from file
with open("model.params", "rb") as f:
    binary_data = f.read()

# Deserialize
params = tvm.runtime.load_param_dict(binary_data)

# Access parameters
for name, tensor in params.items():
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}")

# Move parameters to a specific device
dev = tvm.cuda(0)
params_on_device = {
    name: tvm.nd.array(tensor.numpy(), dev)
    for name, tensor in params.items()
}
```

---

## 28.3 Loading Compiled Artifacts

### tvm.runtime.load_module(path)

Loads a previously exported shared library:

```python
import tvm

# Load the compiled module
mod = tvm.runtime.load_module("model.so")

# Access exported functions
func = mod["main"]
print(func)  # <tvm.runtime.PackedFunc>

# Get function list
func_names = mod.get_function("get_func_names")()
for name in func_names:
    print(f"Exported function: {name}")
```

#### Module Types

The loaded module type depends on how the module was built:

| Build Method | Module Type | Execution |
|-------------|-------------|-----------|
| `relax.build()` | Relax VM executable | Use `relax.VirtualMachine` |
| `tvm.build()` | Graph executor | Use `GraphModule` |
| `tvm.build()` (single PrimFunc) | Native function | Direct `PackedFunc` call |

### relax.VirtualMachine(mod, device)

Creates a Relax Virtual Machine from a loaded module for execution:

```python
import tvm
from tvm import relax

# Load the compiled module
mod = tvm.runtime.load_module("model.so")

# Create VM with a specific device
dev = tvm.cuda(0)
vm = relax.VirtualMachine(mod, dev)

# The VM provides access to all exported Relax functions
# via dictionary-style access

# Run the main function
input_data = tvm.nd.array(
    np.random.randn(1, 3, 224, 224).astype("float32"), dev
)
output = vm["main"](input_data)
print(output.numpy().shape)

# Run a specific named function
features = vm["extract_features"](input_data)
```

#### VM Configuration

```python
# Configure VM behavior
vm = relax.VirtualMachine(
    mod,
    dev,
    memory_cfg="debug",  # Enable memory allocation debugging
)

# The VM manages:
# - Function dispatch
# - Memory allocation for intermediate tensors
# - Device synchronization
# - Constant pool management
```

### Graph Executor (Legacy)

For models built using the legacy graph executor:

```python
import tvm
from tvm import relay

# Load module
mod = tvm.runtime.load_module("model.so")

# Create graph executor
dev = tvm.cuda(0)
graph_mod = tvm.runtime.GraphModule(mod["default"](dev))

# Set inputs
graph_mod.set_input("data", input_data)

# Set parameters (if loaded separately)
graph_mod.load_params(param_data)

# Run
graph_mod.run()

# Get output
output = graph_mod.get_output(0)
```

---

## 28.4 Serialization Format

### Module Metadata

The compiled module contains metadata that describes its contents:

```python
import tvm

# After building, inspect the module metadata
exec = relax.build(mod, target="llvm")

# The executable contains:
# 1. Function table: maps function names to their implementations
# 2. Constant pool: stores constant tensors (weights, biases)
# 3. Code sections: compiled machine code for each function
# 4. Device code: GPU kernels embedded as data

# Inspect available functions
# (function names depend on the model structure)
```

### Function Table

Each exported function is registered in the function table:

```
Function Table:
+-------------------+----------------+-------------------+
| Function Name     | Type           | Implementation    |
+-------------------+----------------+-------------------+
| "main"            | PackedFunc     | VM bytecode       |
| "fused_kernel_0"  | PackedFunc     | Native code       |
| "fused_kernel_1"  | PackedFunc     | Native code       |
| "set_constants"   | PackedFunc     | Built-in          |
+-------------------+----------------+-------------------+
```

### Constant Pool

Constants (model parameters) can be embedded in the module or stored separately:

```python
# Option 1: Embed constants in the module (bundled)
# All weights are included in the .so file
exec.export_library("model_with_params.so")

# Option 2: Export constants separately
# Weights are stored in a separate .params file
# Reduces the shared library size
params = exec.get_constants()
param_bytes = tvm.runtime.save_param_dict(params)
with open("model.params", "wb") as f:
    f.write(param_bytes)

exec.export_library("model_no_params.so")  # Smaller .so file
```

### Code Sections

The compiled module contains multiple code sections for different purposes:

```
Module Layout:
+----------------------------------+
| Header                           |
|   - Magic number                 |
|   - Version                      |
|   - Section count                |
+----------------------------------+
| Metadata Section                 |
|   - Function table               |
|   - Type information             |
|   - Import table                 |
+----------------------------------+
| Constant Section                 |
|   - Embedded tensors             |
|   - Shape information            |
+----------------------------------+
| Host Code Section                |
|   - Compiled CPU code            |
|   - LLVM-generated native code   |
+----------------------------------+
| Device Code Section              |
|   - CUDA kernels (PTX/CUBIN)    |
|   - OpenCL kernels               |
|   - Vulkan SPIR-V                |
+----------------------------------+
| VM Bytecode Section (if Relax VM)|
|   - VM instructions              |
|   - Instruction constants        |
+----------------------------------+
```

---

## 28.5 Cross-Platform Deployment

### Export on Build Machine, Load on Target Device

```python
# === On Build Machine ===
import tvm
from tvm import relax

# Import model
mod = from_exported_program(model)

# Target-specific compilation
target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu")
mod = relax.get_pipeline("static_shape_tensor")(mod)
exec = relax.build(mod, target=target)

# Export with cross-compiler
exec.export_library(
    "model_arm.so",
    cc="aarch64-linux-gnu-gcc",
)

# Export parameters
params = get_model_params()
with open("model_arm.params", "wb") as f:
    f.write(tvm.runtime.save_param_dict(params))

# Transfer files to target device
# scp model_arm.so model_arm.params user@arm-device:~/deploy/
```

```python
# === On Target Device ===
import tvm
from tvm import relax
import numpy as np

# Load compiled module
mod = tvm.runtime.load_module("model_arm.so")

# Load parameters
with open("model_arm.params", "rb") as f:
    params = tvm.runtime.load_param_dict(f.read())

# Create device and VM
dev = tvm.cpu(0)
vm = relax.VirtualMachine(mod, dev)

# Move parameters to device
params_on_dev = {
    name: tvm.nd.array(tensor.numpy(), dev)
    for name, tensor in params.items()
}

# Run inference
input_data = tvm.nd.array(
    np.random.randn(1, 3, 224, 224).astype("float32"), dev
)
output = vm["main"](input_data)
```

### Device-Specific Considerations

```python
# Android deployment
target_android = tvm.target.Target(
    "opencl",
    host="llvm -mtriple=aarch64-linux-gnu",
)
exec = relax.build(mod, target=target_android)
exec.export_library("model_android.so")

# iOS deployment (requires Xcode for signing)
target_ios = tvm.target.Target(
    "metal",
    host="llvm -mtriple=arm64-apple-darwin",
)
exec = relax.build(mod, target=target_ios)
exec.export_library("model_ios.dylib")

# WebAssembly deployment
target_wasm = tvm.target.Target(
    "llvm -mtriple=wasm32-unknown-unknown -mattr=+simd128",
)
exec = relax.build(mod, target=target_wasm)
exec.export_library("model.wasm")
```

---

## 28.6 GPU Deployment

### Bundling CUDA Kernels

When compiling for NVIDIA GPUs, CUDA kernel code is embedded in the compiled module:

```python
import tvm
from tvm import relax

# Build for CUDA
target = tvm.target.Target("nvidia/nvidia-a100")
mod = relax.get_pipeline("zero")(imported_mod)
exec = relax.build(mod, target=target)

# Export — CUDA kernels are automatically bundled
exec.export_library("model_gpu.so")

# The exported library contains:
# 1. Host code (CPU): function dispatch, memory management
# 2. CUDA code: PTX and/or CUBIN for each GPU kernel
# 3. Metadata: function names, kernel launch parameters
```

### Device Code in Compiled Module

The device code is stored in special sections of the shared library:

```
model_gpu.so:
+----------------------------------+
| .text     (host code)            |
| .data     (host data)            |
| .nv_fatbin (CUDA fat binary)     |  <-- PTX + CUBIN for all SM versions
| .rodata   (metadata, constants)  |
+----------------------------------+
```

At runtime, the TVM CUDA device API extracts the fat binary and loads the appropriate kernel for the detected GPU architecture.

### Multi-GPU Support

```python
# Build once, deploy on multiple GPUs
target = tvm.target.Target("cuda")
exec = relax.build(mod, target=target)
exec.export_library("model_multi_gpu.so")

# Load on multiple devices
mod = tvm.runtime.load_module("model_multi_gpu.so")

# Create separate VMs for each GPU
vms = []
for gpu_id in range(4):
    dev = tvm.cuda(gpu_id)
    vm = relax.VirtualMachine(mod, dev)
    vms.append(vm)

# Run on each GPU independently
for gpu_id, vm in enumerate(vms):
    dev = tvm.cuda(gpu_id)
    input_data = tvm.nd.array(
        np.random.randn(1, 3, 224, 224).astype("float32"), dev
    )
    output = vm["main"](input_data)
    print(f"GPU {gpu_id}: {output.numpy().argmax()}")
```

---

## 28.7 Standalone Deployment

### Minimum Runtime Requirements

For deployment without the full TVM Python package, only the TVM runtime library is needed:

```bash
# Build the minimal TVM runtime
cd tvm
mkdir build && cd build
cmake .. \
    -DRUNTIME_ONLY=ON \
    -DUSE_LLVM=OFF \
    -DUSE_CUDA=ON      # Include if deploying on GPU
make -j$(nproc)

# This produces libtvm_runtime.so — a lightweight runtime library
# Size: typically 5-20 MB (vs. 100+ MB for the full TVM package)
```

### C Runtime API

TVM provides a C API for deploying compiled modules in C/C++ applications:

```c
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Initialize TVM runtime
    TVMModuleHandle mod;
    int status = TVMModLoadFromFile("model.so", "so", &mod);
    if (status != 0) {
        fprintf(stderr, "Failed to load module: %s\n", TVMGetLastError());
        return 1;
    }

    // Get the main function
    TVMFunctionHandle main_func;
    TVMModGetFunction(mod, "main", 0, &main_func);

    // Create input tensor
    int64_t shape[] = {1, 3, 224, 224};
    float input_data[1 * 3 * 224 * 224];
    // ... fill input_data ...

    DLTensor input;
    DLDevice dev = {kDLCPU, 0};
    input.data = input_data;
    input.device = dev;
    input.ndim = 4;
    input.dtype = {kDLFloat, 32, 1};
    input.shape = shape;
    input.strides = NULL;
    input.byte_offset = 0;

    // Allocate output tensor
    int64_t out_shape[] = {1, 1000};
    float output_data[1000];
    DLTensor output;
    output.data = output_data;
    output.device = dev;
    output.ndim = 2;
    output.dtype = {kDLFloat, 32, 1};
    output.shape = out_shape;
    output.strides = NULL;
    output.byte_offset = 0;

    // Call the function
    TVMValue args[2];
    args[0].v_handle = &input;
    args[1].v_handle = &output;
    int type_codes[2] = {kTVMDLTensorHandle, kTVMDLTensorHandle};
    TVMAPISetLastError(NULL);
    status = TVMFuncCall(main_func, args, type_codes, 2, NULL, NULL);
    if (status != 0) {
        fprintf(stderr, "Function call failed: %s\n", TVMGetLastError());
        return 1;
    }

    // Process output
    int predicted_class = 0;
    float max_score = output_data[0];
    for (int i = 1; i < 1000; i++) {
        if (output_data[i] > max_score) {
            max_score = output_data[i];
            predicted_class = i;
        }
    }
    printf("Predicted class: %d (score: %f)\n", predicted_class, max_score);

    // Cleanup
    TVMFuncRelease(main_func);
    TVMModRelease(mod);
    return 0;
}
```

### C++ Deployment API

```cpp
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/ndarray.h>
#include <iostream>

int main() {
    // Load compiled module
    tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile("model.so");

    // Create device
    DLDevice dev = {kDLCPU, 0};

    // Create Relax VM
    tvm::runtime::Module vm_mod = tvm::runtime::Module::LoadFromFile(
        tvm::runtime::Registry::Get("relax.VirtualMachine")->get()
    );

    // Get the VM constructor
    auto vm_ctor = tvm::runtime::Registry::Get("relax.VirtualMachine");
    tvm::runtime::Module vm = (*vm_ctor)(mod, dev);

    // Create input
    auto input = tvm::runtime::NDArray::Empty(
        {1, 3, 224, 224},
        {kDLFloat, 32, 1},
        dev
    );
    // Fill input data...

    // Get main function and call
    auto main_func = vm.GetFunction("main");
    tvm::runtime::TVMRetValue result = main_func(input);

    // Process result
    auto output = result.AsObjectRef<tvm::runtime::NDArray>();
    std::cout << "Output shape: " << output.Shape()[0]
              << "x" << output.Shape()[1] << std::endl;

    return 0;
}
```

### Embedded Deployment

For bare-metal and deeply embedded systems:

```c
// For microcontrollers and embedded systems, TVM provides
// a "CRT" (C Runtime) that does not require dynamic loading

#include <tvm/runtime/crt/api_types.h>
#include <tvm/runtime/crt/module.h>

// The model is compiled to a static byte array
// (no .so file needed — linked directly into the firmware)
extern const unsigned char model_data[];
extern const unsigned int model_data_len;

int deploy_model() {
    // Load module from in-memory data
    TVMModuleHandle mod;
    int ret = TVMModLoadFromMemory(
        model_data, model_data_len, &mod
    );
    if (ret != 0) return ret;

    // Get main function
    TVMFunctionHandle func;
    TVMModGetFunction(mod, "main", 0, &func);

    // Prepare input tensor (from sensor data, etc.)
    float input[1 * 3 * 32 * 32];
    // ... fill from sensor ...

    // Allocate output
    float output[10];

    // Run inference
    // ... call TVMFuncCall with input/output ...

    return 0;
}
```

---

## 28.8 Parameter Management

### Separate vs Bundled Parameters

```python
import tvm
from tvm import relax

# Build the model
exec = relax.build(mod, target="llvm")

# ---- Option A: Bundled Parameters ----
# Parameters are embedded in the shared library
# Simpler deployment (single file)
exec.export_library("model_bundled.so")
# Module size: ~model_params_size + code_size

# ---- Option B: Separate Parameters ----
# Parameters stored in a separate file
# Smaller module, flexible parameter updates
exec.export_library("model_code.so")  # Code only

params = exec.get_constants()
with open("model.params", "wb") as f:
    f.write(tvm.runtime.save_param_dict(params))

# Load with separate parameters
mod = tvm.runtime.load_module("model_code.so")
with open("model.params", "rb") as f:
    params = tvm.runtime.load_param_dict(f.read())
dev = tvm.cpu(0)
vm = relax.VirtualMachine(mod, dev)
# Set parameters on the VM
for name, tensor in params.items():
    vm.set_input(name, tvm.nd.array(tensor.numpy(), dev))
```

### Parameter Sharing Between Models

```python
# When multiple models share weights (e.g., teacher-student),
# parameters can be exported once and shared

# Export shared embedding weights once
shared_params = {"embed_tokens.weight": embedding_weight}
with open("shared_embed.params", "wb") as f:
    f.write(tvm.runtime.save_param_dict(shared_params))

# Export each model's unique parameters
model_a_params = {"lm_head.weight": lm_head_a, ...}
with open("model_a.params", "wb") as f:
    f.write(tvm.runtime.save_param_dict(model_a_params))

model_b_params = {"lm_head.weight": lm_head_b, ...}
with open("model_b.params", "wb") as f:
    f.write(tvm.runtime.save_param_dict(model_b_params))

# At deployment, load shared + model-specific parameters
shared = tvm.runtime.load_param_dict(open("shared_embed.params", "rb").read())
model_a = tvm.runtime.load_param_dict(open("model_a.params", "rb").read())
all_params = {**shared, **model_a}
```

### Dynamic Parameter Loading

```python
class ModelServer:
    """Server that can hot-swap model parameters."""

    def __init__(self, model_path: str, device):
        self.mod = tvm.runtime.load_module(model_path)
        self.device = device
        self.vm = relax.VirtualMachine(self.mod, device)
        self.current_params = None

    def load_params(self, params_path: str):
        """Load new parameters without reloading the module."""
        with open(params_path, "rb") as f:
            self.current_params = tvm.runtime.load_param_dict(f.read())

        # Move to device and set on VM
        for name, tensor in self.current_params.items():
            self.current_params[name] = tvm.nd.array(
                tensor.numpy(), self.device
            )

    def inference(self, input_data):
        """Run inference with current parameters."""
        input_nd = tvm.nd.array(input_data, self.device)
        return self.vm["main"](input_nd).numpy()
```

---

## 28.9 Advanced Export Scenarios

### Exporting with Custom Runtime Modules

```python
import tvm

# Build and include custom runtime modules
# For example, including a custom operator library
custom_op_lib = tvm.runtime.load_module("custom_ops.so")

# Export with the custom library bundled
exec.export_library(
    "model_with_custom.so",
    libs=[custom_op_lib],
)
```

### Export for Multiple Targets (Heterogeneous Execution)

```python
# Build for heterogeneous target (CPU + GPU)
target_cpu = tvm.target.Target("llvm")
target_gpu = tvm.target.Target("cuda")

# TVM can build for a heterogeneous target
target = tvm.target.Target(
    host="llvm",
    devices=[target_gpu, target_cpu],
)

# Parts of the model run on GPU, parts on CPU
exec = relax.build(mod, target=target)
exec.export_library("model_heterogeneous.so")
```

### Export with Metadata

```python
# Add custom metadata to the exported module
import json

metadata = {
    "model_name": "resnet18",
    "input_shape": [1, 3, 224, 224],
    "input_dtype": "float32",
    "output_classes": 1000,
    "version": "1.0.0",
    "framework": "pytorch-2.1",
    "target": "cuda",
}

# TVM modules can carry metadata
# (implementation depends on the module type)
```

### Verifying Exported Modules

```python
import tvm
import numpy as np

def verify_export(model_path, params_path, input_shape, expected_output=None):
    """Verify that an exported module produces correct results."""
    # Load module
    mod = tvm.runtime.load_module(model_path)
    dev = tvm.cpu(0)
    vm = relax.VirtualMachine(mod, dev)

    # Load and set parameters
    if params_path:
        with open(params_path, "rb") as f:
            params = tvm.runtime.load_param_dict(f.read())
        for name, tensor in params.items():
            # Set parameters via VM
            pass

    # Run with test input
    test_input = np.random.randn(*input_shape).astype("float32")
    input_nd = tvm.nd.array(test_input, dev)
    output = vm["main"](input_nd)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output sample: {output.numpy().flatten()[:5]}")

    if expected_output is not None:
        np.testing.assert_allclose(
            output.numpy(), expected_output, rtol=1e-5, atol=1e-5
        )
        print("Verification PASSED")
    else:
        print("Output looks reasonable (no expected output provided)")

# Usage
verify_export("model.so", "model.params", (1, 3, 224, 224))
```

---

## 28.10 Deployment Checklist

### Pre-Deployment Verification

```python
# 1. Verify target compatibility
target = tvm.target.Target("cuda")
print(f"Target: {target}")
print(f"Target kind: {target.kind.name}")
print(f"Target attrs: {target.attrs}")

# 2. Verify module functions
mod = tvm.runtime.load_module("model.so")
# List all exported functions
# (implementation varies by module type)

# 3. Verify parameter shapes
with open("model.params", "rb") as f:
    params = tvm.runtime.load_param_dict(f.read())
for name, tensor in params.items():
    print(f"  {name}: {tensor.shape} {tensor.dtype}")

# 4. Run a smoke test
dev = tvm.cpu(0)
vm = relax.VirtualMachine(mod, dev)
test_input = tvm.nd.array(np.zeros((1, 3, 224, 224), dtype="float32"), dev)
try:
    output = vm["main"](test_input)
    print(f"Smoke test PASSED: output shape {output.shape}")
except Exception as e:
    print(f"Smoke test FAILED: {e}")
```

### Common Export Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `dlopen` error | Incompatible ABI | Rebuild with matching compiler |
| Missing CUDA kernel | Wrong GPU target | Specify correct `nvidia/...` target |
| Parameter mismatch | Shape/dtype mismatch | Verify param export matches build |
| Memory error at load | Insufficient RAM | Reduce model size or use quantization |
| Function not found | Wrong function name | Check exported function names |

---

## 28.11 Summary

| API | Purpose | Output |
|-----|---------|--------|
| `relax.build()` | Compile IRModule | `Executable` |
| `exec.export_library()` | Export to shared library | `.so` / `.dll` / `.dylib` |
| `save_param_dict()` | Serialize parameters | Binary blob |
| `load_param_dict()` | Deserialize parameters | Dict of `NDArray` |
| `load_module()` | Load shared library | `runtime.Module` |
| `VirtualMachine()` | Create execution engine | VM instance |
| `tvm.build()` | Build TIR functions | `runtime.Module` |

The serialization and export system in TVM provides a complete pipeline from compiled IRModule to deployable artifact. The system supports cross-platform deployment, GPU code bundling, standalone C/C++ deployment, and flexible parameter management, making it possible to deploy TVM-compiled models on virtually any target platform.
