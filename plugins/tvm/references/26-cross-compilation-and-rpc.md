# Apache TVM — Chapter 26: Cross-Compilation and RPC

This reference covers TVM's cross-compilation and Remote Procedure Call (RPC) system. TVM decouples compilation from execution, enabling models to be compiled on one machine (the build host) and deployed on a different machine (the target device) with a different architecture.

---

## 26.1 Cross-Compilation Workflow

### Overview

Cross-compilation is the process of building executable code on one platform for execution on a different platform. TVM's cross-compilation workflow follows these steps:

```
Build Host (e.g., x86 server with GPU)         Target Device (e.g., ARM phone)
========================================        ================================
1. Import model                                 4. Start RPC server
2. Optimize with target-specific passes
3. Build for target architecture
5. Export compiled module + parameters  ------> 6. Load module via RPC
                                                7. Execute inference
                                                8. Return results
```

### Key Concepts

- **Target**: The hardware platform for which code is generated (e.g., `llvm -mtriple=aarch64-linux-gnu`, `cuda`, `vulkan`).
- **Build host**: The machine running the TVM compiler. It does not need to match the target architecture.
- **RPC server**: A lightweight server running on the target device that accepts compiled modules and executes them.
- **RPC tracker**: A centralized service that manages connections to multiple RPC servers, enabling device pooling.

### Why Cross-Compilation Matters

- **Resource constraints**: Target devices (phones, IoT, embedded) often lack the memory and compute to run the TVM compiler.
- **Build farm architecture**: Teams compile on powerful servers and deploy to heterogeneous edge devices.
- **Auto-tuning**: MetaSchedule can benchmark candidate schedules on the actual target hardware via RPC without requiring the compiler to run on the device.
- **CI/CD integration**: Compilation happens in CI pipelines; devices are available via RPC for testing.

---

## 26.2 RPC System

### Architecture

The TVM RPC system has three main components:

```
                           RPC Tracker
                          (central coordinator)
                               |
            +------------------+------------------+
            |                  |                  |
      RPC Server 1       RPC Server 2       RPC Server 3
      (ARM phone)        (RISC-V board)     (GPU server)
                               |
                          RPC Client
                        (build machine)
```

### tvm.rpc.Server — Starting an RPC Server

The RPC server runs on the target device and listens for incoming connections from the build host.

```python
# Start a standalone RPC server (on the target device)
python -m tvm.exec.rpc_server --host 0.0.0.0 --port 9090

# Start with a tracker (registers with a central tracker)
python -m tvm.exec.rpc_server \
    --tracker 192.168.1.100:9190 \
    --key jetson-nano \
    --port 9090

# Start with custom configuration
python -m tvm.exec.rpc_server \
    --host 0.0.0.0 \
    --port 9090 \
    --timeout 600 \
    --nofork
```

Parameters:

| Parameter | Description |
|-----------|-------------|
| `--host` | Bind address (default `0.0.0.0`) |
| `--port` | Port number to listen on |
| `--tracker` | Address of RPC tracker to register with (host:port) |
| `--key` | Device key for identification in the tracker |
| `--timeout` | Connection timeout in seconds |
| `--nofork` | Run in foreground (do not fork) |
| `--append-path` | Additional Python path for custom operators |

Programmatic server start:

```python
import tvm
from tvm import rpc

# Start RPC server programmatically
server = rpc.Server(
    host="0.0.0.0",
    port=9090,
    key="my-device",
)
# Server runs in a background thread
# server.host and server.port are available
print(f"RPC server running at {server.host}:{server.port}")
```

### tvm.rpc.connect — Connecting to an RPC Server

The client (build host) connects to the RPC server to upload and execute modules.

```python
import tvm
from tvm import rpc

# Direct connection to a specific RPC server
remote = rpc.connect(
    host="192.168.1.50",   # Target device IP
    port=9090,              # RPC server port
    session_timeout=60,     # Session timeout in seconds
)

# The remote object provides access to the target device's
# runtime environment
print(remote.system_lib())  # Access the remote system library

# Upload a compiled module
remote.upload("model.tar.so")

# Load the uploaded module
mod = remote.load_module("model.tar.so")

# Create a device on the remote
dev = remote.cpu(0)
# or for GPU: dev = remote.cuda(0)

# Execute
result = mod["main"](data)
```

### tvm.rpc.tracker — Tracker for Managing Multiple Devices

The RPC tracker is a centralized service that manages connections to multiple RPC servers. Clients request devices by key, and the tracker assigns available devices.

```python
# Start a tracker (on a central machine)
python -m tvm.exec.rpc_tracker --host 0.0.0.0 --port 9190

# Programmatic tracker start
from tvm import rpc
tracker = rpc.Tracker(
    host="0.0.0.0",
    port=9190,
)
print(f"Tracker running at {tracker.host}:{tracker.port}")
```

Connecting via tracker:

```python
import tvm
from tvm import rpc

# Request a device from the tracker
# The key must match the --key parameter of the RPC server
remote = rpc.connect_tracker(
    host="192.168.1.100",  # Tracker IP
    port=9190,              # Tracker port
).request(
    key="jetson-nano",     # Device key
    session_timeout=60,    # Session timeout
    priority=0,            # Priority (higher = preferred)
)

# Use remote as before
dev = remote.cpu(0)
```

### Remote Session Management

```python
# List available devices from the tracker
from tvm import rpc
tracker_session = rpc.connect_tracker("192.168.1.100", 9190)

# Summary returns a dict of connected devices
summary = tracker_session.summary()
print(summary)
# Output example:
# {
#   "jetson-nano": {"count": 2, "free": 1},
#   "riscv-board": {"count": 1, "free": 1},
# }

# Request with custom configuration
remote = tracker_session.request(
    key="jetson-nano",
    session_timeout=120,
    priority=10,
)

# Upload files to remote
remote.upload("model.so")
remote.upload("params.bin")

# Download files from remote
remote.download("output.bin")

# Clean up: remove uploaded files
remote.remove("model.so")
```

---

## 26.3 Supported Targets for Cross-Compilation

### ARM (Mobile and Embedded)

ARM targets include smartphones, tablets, Raspberry Pi, NVIDIA Jetson, and custom embedded boards.

```python
import tvm
from tvm import relax

# ARM target for Android / Linux ARM
target_arm = tvm.target.Target(
    "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"
)

# ARM Cortex-A with specific CPU features
target_cortex_a = tvm.target.Target(
    "llvm -mtriple=armv8a-linux-gnu -mattr=+neon,+fp16"
)

# ARM Cortex-M (microcontroller)
target_cortex_m = tvm.target.Target(
    "llvm -mtriple=armv7em-none-eabi -mcpu=cortex-m4"
)

# Build for ARM
mod = relax.get_pipeline("zero", target=target_arm)(imported_mod)
exec = relax.build(mod, target=target_arm)
exec.export_library("model_arm.so")
```

For Android specifically, TVM provides JNI integration:

```python
# Build for Android with OpenCL
target_android = tvm.target.Target(
    "opencl -device=mali",
    host="llvm -mtriple=aarch64-linux-gnu",
)
```

### x86 (Server)

x86 targets are used for server-side deployment and testing.

```python
# x86 server with AVX-512
target_x86 = tvm.target.Target(
    "llvm -mcpu=skylake-avx512"
)

# x86 with specific features
target_x86_vnni = tvm.target.Target(
    "llvm -mcpu=cascadelake -mattr=+avx512vnni"
)

# Build and deploy
exec = relax.build(mod, target=target_x86)
exec.export_library("model_x86.so")
```

### RISC-V

TVM supports RISC-V targets with the Vector extension:

```python
# RISC-V with vector extension
target_riscv = tvm.target.Target(
    "llvm -mtriple=riscv64-unknown-linux-gnu "
    "-mcpu=generic-rv64 -mattr=+v,+zfh"
)

# Build for RISC-V
exec = relax.build(mod, target=target_riscv)
exec.export_library("model_riscv.so")
```

### Hexagon (Qualcomm DSP)

TVM can target Qualcomm Hexagon DSP for mobile inference:

```python
# Hexagon DSP target
target_hexagon = tvm.target.Target(
    "hexagon",
    host="llvm -mtriple=hexagon-unknown-linux-gnu",
)

# With specific HVX version
target_hexagon_v68 = tvm.target.Target(
    "hexagon -mcpu=v68",
    host="llvm -mtriple=hexagon-unknown-linux-gnu",
)

# Build for Hexagon
exec = relax.build(mod, target=target_hexagon)
exec.export_library("model_hexagon.so")
```

### WebAssembly / WebGPU

```python
# WebAssembly target
target_wasm = tvm.target.Target(
    "llvm -mtriple=wasm32-unknown-unknown -mattr=+simd128"
)

# WebGPU target
target_webgpu = tvm.target.Target("webgpu")
```

---

## 26.4 Low-Level TensorIR Approach

### Building Individual PrimFunc

The TensorIR approach allows fine-grained control over cross-compilation. You can build individual `PrimFunc` functions and measure their performance on remote devices.

```python
import tvm
from tvm.script import ir as I, tir as T

# Define a TIR kernel
@I.ir_module
class MyModule:
    @T.prim_func
    def vector_add(
        A: T.Buffer((1024,), "float32"),
        B: T.Buffer((1024,), "float32"),
        C: T.Buffer((1024,), "float32"),
    ):
        for i in range(1024):
            with T.sblock("C"):
                vi = T.axis.spatial(1024, i)
                C[vi] = A[vi] + B[vi]

# Build for a specific target
target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu")
mod = MyModule

# Apply scheduling
sch = tvm.s_tir.Schedule(mod)
# ... apply schedule primitives ...

# Build the PrimFunc
rt_mod = tvm.build(sch.mod, target=target)

# Export
rt_mod.export_library("vector_add.so")
```

### Upload and Run on Remote

```python
import tvm
from tvm import rpc
import numpy as np

# Connect to remote device
remote = rpc.connect("192.168.1.50", 9090)

# Upload compiled module
remote.upload("vector_add.so")

# Load module on remote
mod = remote.load_module("vector_add.so")

# Create remote device
dev = remote.cpu(0)

# Allocate tensors on remote device
a = tvm.nd.array(np.random.randn(1024).astype("float32"), dev)
b = tvm.nd.array(np.random.randn(1024).astype("float32"), dev)
c = tvm.nd.array(np.zeros(1024, dtype="float32"), dev)

# Execute on remote
mod["vector_add"](a, b, c)

# Copy result back to local
result = c.numpy()
print(result[:10])
```

### Measure Performance

```python
import time
import tvm
from tvm import rpc

# Connect to remote
remote = rpc.connect("192.168.1.50", 9090)
dev = remote.cpu(0)

# Load module
mod = remote.load_module("vector_add.so")

# Prepare input
a = tvm.nd.array(np.random.randn(1024).astype("float32"), dev)
b = tvm.nd.array(np.random.randn(1024).astype("float32"), dev)
c = tvm.nd.array(np.zeros(1024, dtype="float32"), dev)

# Warm up
for _ in range(10):
    mod["vector_add"](a, b, c)

# Timed run
# Use TVM's built-in timing to exclude network overhead
timer = mod.time_evaluator("vector_add", dev, number=100, repeat=5)
timing_result = timer(a, b, c)

print(f"Mean execution time: {timing_result.mean * 1e6:.2f} us")
print(f"Std deviation: {timing_result.std * 1e6:.2f} us")
```

### Remote Timing APIs

TVM provides `time_evaluator` that measures execution time on the remote device, excluding network transfer overhead:

```python
# time_evaluator runs the function multiple times on the remote
# and returns timing statistics
timing = mod.time_evaluator(
    func_name="main",       # Function to benchmark
    dev=dev,                # Remote device
    number=10,              # Number of runs per repeat
    repeat=3,               # Number of repeats
    min_repeat_ms=100,      # Minimum total time per repeat (ms)
)

result = timing(input1, input2)
print(f"Mean: {result.mean:.6f}s")
print(f"Median: {result.median:.6f}s")
print(f"Std: {result.std:.6f}s")
```

The timing mechanism works by:
1. Transferring inputs to the remote device once.
2. Running the function `number` times in a loop on the remote.
3. Measuring total time locally on the remote device.
4. Returning the per-invocation average.

This ensures that network latency does not contaminate the measurement.

---

## 26.5 High-Level Relax Approach

### Building a Complete Model

The Relax approach compiles an entire model end-to-end and deploys it via RPC:

```python
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
import torch
import torchvision

# Step 1: Import model
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# Create example input
example_input = torch.randn(1, 3, 224, 224)

# Export from PyTorch
with torch.no_grad():
    exported_program = torch.export.export(model, (example_input,))

mod = from_exported_program(exported_program)

# Step 2: Define target for cross-compilation
target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu")

# Step 3: Optimize and build
mod = relax.get_pipeline("static_shape_tensor")(mod)
exec = relax.build(mod, target=target)

# Step 4: Export
exec.export_library("resnet18_arm.so")
```

### Deploy Full Model via RPC

```python
import tvm
from tvm import relax, rpc
import numpy as np
from tvm.runtime import load_param_dict, save_param_dict

# Connect to remote ARM device
remote = rpc.connect("192.168.1.50", 9090)

# Upload compiled model
remote.upload("resnet18_arm.so")

# Upload parameters (if saved separately)
remote.upload("resnet18_params.bin")

# Load module and parameters
mod = remote.load_module("resnet18_arm.so")
params = load_param_dict(open("resnet18_params.bin", "rb").read())

# Create device and VM
dev = remote.cpu(0)
vm = relax.VirtualMachine(mod, dev)

# Prepare input
input_data = tvm.nd.array(
    np.random.randn(1, 3, 224, 224).astype("float32"),
    dev,
)

# Run inference
output = vm["main"](input_data)

# Get result
result = output.numpy()
predicted_class = np.argmax(result)
print(f"Predicted class: {predicted_class}")
```

### End-to-End Execution

A complete workflow that compiles on the build host and runs on the remote:

```python
import tvm
from tvm import relax, rpc, runtime
import numpy as np

def compile_and_deploy(
    model,
    example_input,
    target_str="llvm -mtriple=aarch64-linux-gnu",
    remote_host="192.168.1.50",
    remote_port=9090,
):
    """Complete cross-compilation and deployment workflow."""

    # --- Build Phase (on host) ---
    from tvm.relax.frontend.torch import from_exported_program
    import torch

    with torch.no_grad():
        exported_program = torch.export.export(model, (example_input,))
    mod = from_exported_program(exported_program)

    target = tvm.target.Target(target_str)
    mod = relax.get_pipeline("static_shape_tensor")(mod)
    exec = relax.build(mod, target=target)

    # Export to temporary files
    exec.export_library("/tmp/model.so")

    # --- Deploy Phase (via RPC) ---
    remote = rpc.connect(remote_host, remote_port)
    remote.upload("/tmp/model.so")

    mod = remote.load_module("model.so")
    dev = remote.cpu(0)
    vm = relax.VirtualMachine(mod, dev)

    # Run inference
    input_nd = tvm.nd.array(
        example_input.numpy().astype("float32"), dev
    )
    output = vm["main"](input_nd)

    return output.numpy()

# Usage
import torchvision
model = torchvision.models.mobilenet_v2(pretrained=True).eval()
example = torch.randn(1, 3, 224, 224)
result = compile_and_deploy(model, example)
```

---

## 26.6 Performance Measurement

### Excluding Network Overhead

When benchmarking via RPC, it is critical to exclude network transfer time from measurements. TVM provides two mechanisms:

```python
# Method 1: time_evaluator (recommended)
timer = mod.time_evaluator("main", dev, number=100, repeat=10)
result = timer(input_data)
print(f"Mean: {result.mean * 1000:.3f} ms")

# Method 2: Manual timing on remote device
# Use remote.clock() to get device-side timestamps
import time

# Synchronize device first
dev.sync()

# The evaluator API already handles this correctly
# For custom timing, you can use:
evaluator = mod.time_evaluator(
    "main",
    dev,
    number=50,
    repeat=5,
    min_repeat_ms=100,  # Adaptive: increase number if total < 100ms
)
result = evaluator(input_data)
```

### AutoTuning with RPC

MetaSchedule can use RPC for auto-tuning on actual hardware:

```python
import tvm
from tvm import relax, rpc, meta_schedule as ms

# Configure MetaSchedule to use RPC for tuning
# The tuning database will store the best schedules
database = ms.database.MemoryDatabase()

# Define the target and RPC configuration
target = tvm.target.Target("nvidia/nvidia-a100")
remote = rpc.connect("gpu-server", 9090)

# During MetaSchedule tuning, candidate schedules are:
# 1. Compiled on the build host
# 2. Uploaded to the remote device via RPC
# 3. Benchmarked using time_evaluator
# 4. Results stored in the database

# Apply tuning results during subsequent builds
mod = relax.transform.MetaScheduleApplyDatabase(database)(mod)
```

---

## 26.7 RPC Server Implementations

### Python RPC Server

The standard RPC server is implemented in Python and runs on any platform with Python support:

```bash
# Basic server
python -m tvm.exec.rpc_server --host 0.0.0.0 --port 9090

# With tracker registration
python -m tvm.exec.rpc_server \
    --tracker 192.168.1.100:9190 \
    --key my-device
```

The Python server is the most flexible and supports all TVM features. It requires:
- Python 3.8+
- TVM Python package installed
- Sufficient memory for the models being deployed

### C++ RPC Server

For platforms where Python is not available or too heavy, TVM provides a C++ RPC server:

```bash
# Build the C++ RPC server
cd tvm
mkdir build && cd build
cmake .. -DUSE_RPC=ON
make -j$(nproc)

# The binary is at build/tvm_rpc
./tvm_rpc server --host 0.0.0.0 --port 9090
```

The C++ server links against the TVM runtime library (libtvm_runtime.so) and supports:
- Module loading and execution
- Parameter loading
- Device management (CPU, GPU)
- Time evaluation

### iOS RPC Server

TVM provides an iOS RPC server for testing on iPhones and iPads:

```python
# Build and deploy the iOS RPC app
# Located in tvm/apps/ios_rpc
# Build with Xcode and deploy to device

# Connect from build host
remote = rpc.connect_tracker("ios-device-ip", 9190).request(
    key="iphone",
    session_timeout=60,
)
```

The iOS RPC server is implemented as a native iOS app that embeds the TVM C++ runtime.

### Android RPC Server

Similarly, TVM provides an Android RPC server:

```python
# Located in tvm/apps/android_rpc
# Build with Gradle and deploy to device

# The Android RPC app can be built from the TVM source tree
# It packages the TVM C++ runtime with a Java JNI wrapper

# Connect from build host
remote = rpc.connect_tracker("tracker-ip", 9190).request(
    key="android",
)
```

### MinRPC for Minimal Environments

MinRPC is a lightweight RPC implementation for constrained environments:

```python
# MinRPC is used internally for certain deployment scenarios
# where a full RPC server is not available
# It provides a minimal protocol for:
# - Module upload/download
# - Function invocation
# - Basic device management

# MinRPC is typically used by the build system internally
# and is not usually invoked directly by users
```

---

## 26.8 Setting Up RPC for Different Platforms

### ARM Linux (Raspberry Pi, Jetson)

```bash
# On the ARM device:
# 1. Install TVM runtime
pip install tvm  # or build from source with runtime only

# 2. Build TVM runtime from source (for custom configuration)
git clone https://github.com/apache/tvm.git
cd tvm && mkdir build && cd build
cmake .. -DUSE_LLVM=OFF -DUSE_RPC=ON -DRUNTIME_ONLY=ON
make -j$(nproc)

# 3. Start RPC server
python -m tvm.exec.rpc_server \
    --host 0.0.0.0 \
    --port 9090 \
    --tracker tracker-ip:9190 \
    --key raspberry-pi
```

```python
# On the build host:
import tvm
from tvm import rpc

remote = rpc.connect_tracker("tracker-ip", 9190).request(
    key="raspberry-pi",
)

# Build with ARM target
target = tvm.target.Target("llvm -mtriple=armv7l-linux-gnueabihf")
# ... build and deploy ...
```

### NVIDIA GPU Server

```python
# On the GPU server:
# Start RPC server with CUDA support
# python -m tvm.exec.rpc_server --port 9090 --key gpu-a100

# On the build host:
import tvm
from tvm import rpc, relax

remote = rpc.connect("gpu-server-ip", 9090)
dev = remote.cuda(0)

# Build for specific GPU
target = tvm.target.Target("nvidia/nvidia-a100")
mod = relax.get_pipeline("zero")(imported_mod)
exec = relax.build(mod, target=target)
exec.export_library("model_gpu.so")

# Deploy
remote.upload("model_gpu.so")
mod = remote.load_module("model_gpu.so")
vm = relax.VirtualMachine(mod, dev)

# Run
input_data = tvm.nd.array(np.random.randn(1, 3, 224, 224).astype("float32"), dev)
output = vm["main"](input_data)
```

### Qualcomm Hexagon DSP

```bash
# Set up Hexagon toolchain
export HEXAGON_SDK_ROOT=/path/to/hexagon-sdk
export HEXAGON_TOOLCHAIN=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools

# Build TVM with Hexagon support
cd tvm/build
cmake .. -DUSE_HEXAGON=ON -DUSE_HEXAGON_SDK=$HEXAGON_SDK_ROOT
make -j$(nproc)
```

```python
# On the build host:
import tvm
from tvm import rpc, relax

# Connect to Hexagon device
remote = rpc.connect("hexagon-device-ip", 9090)
dev = remote.hexagon(0)

# Build for Hexagon
target = tvm.target.Target("hexagon -mcpu=v68")
# ... build and deploy ...
```

### Web Browser

```python
# Build for WebAssembly
target = tvm.target.Target("llvm -mtriple=wasm32-unknown-unknown -mattr=+simd128")
exec = relax.build(mod, target=target)
exec.export_library("model.wasm")

# For WebGPU
target = tvm.target.Target("webgpu")
exec = relax.build(mod, target=target)
# Output is JavaScript + WebGPU shaders
```

---

## 26.9 Advanced RPC Patterns

### Batch Processing on Remote

```python
import tvm
from tvm import rpc, relax
import numpy as np

remote = rpc.connect("192.168.1.50", 9090)
mod = remote.load_module("model.so")
dev = remote.cpu(0)
vm = relax.VirtualMachine(mod, dev)

# Process a batch of inputs
batch_size = 16
inputs = [
    tvm.nd.array(np.random.randn(1, 3, 224, 224).astype("float32"), dev)
    for _ in range(batch_size)
]

results = []
for inp in inputs:
    output = vm["main"](inp)
    results.append(output.numpy())

# Or use batched input directly if the model supports it
batch_input = tvm.nd.array(
    np.random.randn(batch_size, 3, 224, 224).astype("float32"),
    dev,
)
batch_output = vm["main"](batch_input)
```

### Multi-Device Deployment

```python
from tvm import rpc

# Connect to multiple devices via tracker
tracker = rpc.connect_tracker("tracker-ip", 9190)

# Request multiple devices
remotes = []
for i in range(4):
    remote = tracker.request(key="gpu-server")
    remotes.append(remote)

# Distribute work across devices
# Each remote connects to a different GPU server
for i, remote in enumerate(remotes):
    mod = remote.load_module("model.so")
    dev = remote.cuda(0)
    vm = relax.VirtualMachine(mod, dev)
    # Process shard i
```

### Dynamic Parameter Loading

```python
from tvm import rpc, runtime

remote = rpc.connect("192.168.1.50", 9090)
mod = remote.load_module("model.so")
dev = remote.cpu(0)

# Load parameters dynamically
params_binary = open("params.bin", "rb").read()
params = runtime.load_param_dict(params_binary)

# Set parameters on the loaded module
# (depends on how the model was exported)
vm = relax.VirtualMachine(mod, dev)

# Some models accept parameters as function arguments
# Others store them in the module's constant pool
```

---

## 26.10 Security Considerations

### RPC Security

The TVM RPC system is designed for development and trusted network environments. For production deployment:

1. **Do not expose RPC servers to the public internet**: Use VPNs, SSH tunnels, or private networks.

2. **Authentication**: The RPC protocol does not include built-in authentication. Use network-level security.

3. **Sandboxing**: RPC servers can execute arbitrary compiled code. Only connect to trusted build hosts.

4. **Tracker isolation**: The tracker manages device allocation but does not inspect the code being executed.

```python
# Use SSH tunnel for secure RPC
import subprocess

# Create SSH tunnel
tunnel = subprocess.Popen(
    ["ssh", "-L", "9090:localhost:9090", "user@remote-device"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Connect through tunnel
remote = rpc.connect("localhost", 9090)
```

---

## 26.11 Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Connection refused | RPC server not running | Start `rpc_server` on target |
| Timeout | Network/firewall issues | Check connectivity, increase timeout |
| Module load error | Target mismatch | Verify target matches device architecture |
| CUDA error on remote | Driver/library mismatch | Update CUDA driver on target |
| Out of memory | Large model on small device | Reduce model size or use quantization |

### Debug Logging

```python
# Enable debug logging for RPC
import logging
logging.basicConfig(level=logging.DEBUG)

# TVM-specific debug
tvm.set_log_level("DEBUG")

# Check remote device capabilities
remote = rpc.connect("192.168.1.50", 9090)
dev = remote.cpu(0)
print(f"Device: {dev}")
print(f"Max shared memory: {dev.max_shared_memory_per_block}")
print(f"Max threads per block: {dev.max_threads_per_block}")
```

---

## 26.12 Summary

| Component | Purpose | API |
|-----------|---------|-----|
| `rpc.Server` | Run on target device | `python -m tvm.exec.rpc_server` |
| `rpc.connect` | Direct connection to server | `rpc.connect(host, port)` |
| `rpc.Tracker` | Central device manager | `python -m tvm.exec.rpc_tracker` |
| `connect_tracker` | Connect via tracker | `rpc.connect_tracker(host, port).request(key)` |
| `time_evaluator` | Measure remote performance | `mod.time_evaluator(name, dev)` |

TVM's RPC system is a powerful tool for cross-compilation workflows, enabling compilation on powerful build hosts with execution and benchmarking on actual target hardware. The combination of fine-grained TensorIR control and high-level Relax model compilation provides flexibility for any deployment scenario.
