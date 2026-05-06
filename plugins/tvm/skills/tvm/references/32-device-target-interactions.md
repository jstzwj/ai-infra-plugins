# 32 — Device and Target Interactions

## Device Abstraction

### tvm.runtime.Device

Represents a compute device with type and index:

```python
dev = tvm.cpu()          # CPU device
dev = tvm.cuda(0)        # NVIDIA GPU 0
dev = tvm.rocm(0)        # AMD GPU 0
dev = tvm.metal(0)       # Apple Metal 0
dev = tvm.opencl(0)      # OpenCL device 0
dev = tvm.vulkan(0)      # Vulkan device 0
dev = tvm.webgpu(0)      # WebGPU device 0
dev = tvm.ext.dev(0)     # External device
```

### Device Properties
```python
dev.device_type    # Device type integer (kDLCPU=1, kDLCUDA=2, ...)
dev.device_id      # Device index
dev.exist          # Whether device is available
```

### Device API
Each device type implements the DeviceAPI interface:
- **Memory allocation**: `dev.alloc_data(size, alignment)`
- **Data transfer**: `dev.copy_to(src, dst)`, `dev.copy_from(src, dst)`
- **Stream management**: `dev.create_stream()`
- **Synchronization**: `dev.sync()`

---

## Target-Device Mapping

### Target to Device Type
| Target Kind | Device Type | Enum |
|-------------|-------------|------|
| `llvm` | CPU | kDLCPU (1) |
| `cuda` | NVIDIA GPU | kDLCUDA (2) |
| `rocm` | AMD GPU | kDLROCM (10) |
| `metal` | Apple Metal | kDLMetal (8) |
| `opencl` | OpenCL | kDLOpenCL (4) |
| `vulkan` | Vulkan | kDLVulkan (7) |
| `webgpu` | WebGPU | kDLWebGPU (12) |
| `hexagon` | Qualcomm DSP | kDLHexagon (11) |

### Multi-device Compilation
TVM automatically handles host-device code split:
```python
# Build for CUDA — TVM generates:
# 1. LLVM host code (CPU)
# 2. CUDA device code (GPU)
exec = relax.build(mod, target="cuda")
```

---

## Target Attributes Affecting Compilation

### Vector Length → Vectorization
```python
target = tvm.target.Target("llvm -mcpu=skylake-avx512")
# AVX-512: 512 bits → 16 x float32 elements
# VectorizeLoop uses this for vector width
```

### Thread Dimensions → GPU Threading
```python
target = tvm.target.Target("cuda")
# max_threads_per_block = 1024
# max_shared_memory_per_block = 49152 bytes
# warp_size = 32
# These limit thread binding and shared memory usage
```

### Memory Hierarchy → Cache/Scheduling
```python
target = tvm.target.Target("nvidia/nvidia-a100")
# l2_cache_size_bytes = 40 * 1024 * 1024  (40 MB)
# Affects cache tiling and DLight scheduling decisions
```

### Register File Size → Register Pressure
```python
# max_num_threads = limits total threads
# registers_per_block = limits register usage per block
```

---

## Target Configuration Flow

1. **Create Target object**
```python
target = tvm.target.Target("nvidia/nvidia-a100")
```

2. **Pass to build or pipeline**
```python
exec = relax.build(mod, target=target)
# or
mod = relax.get_pipeline("zero")(mod)
```

3. **Target attributes influence passes**
- PassContext reads target info
- Individual passes query target for constraints
- DLight/MetaSchedule use target for scheduling decisions

4. **Code generation uses target**
- LLVM target for CPU
- CUDA/OpenCL/etc. for GPU
- External backends via BYOC

---

## Device API Implementations

### CUDADeviceAPI
```python
# Memory: cuMemAlloc, cuMemFree, cuMemAllocManaged
# Compute: cuLaunchKernel
# Stream: cuStreamCreate, cuStreamSynchronize
# Transfer: cuMemcpyAsync, cuMemcpyPeerAsync
```

### ROCmDeviceAPI
```python
# Memory: hipMalloc, hipFree
# Compute: hipLaunchKernel
# Stream: hipStreamCreate, hipStreamSynchronize
```

### MetalDeviceAPI
```python
# Memory: MTLBuffer allocation
# Compute: MTLComputeCommandEncoder
# Transfer: BlitCommandEncoder
```

### OpenCLDeviceAPI
```python
# Memory: clCreateBuffer
# Compute: clEnqueueNDRangeKernel
# Queue: clCreateCommandQueue
```

### VulkanDeviceAPI
```python
# Memory: VkBuffer allocation
# Compute: VkCommandBuffer dispatch
# Transfer: VkCmdCopyBuffer
# Descriptor: VkDescriptorSet management
```

---

## Memory Management Across Devices

### Allocation
```python
# CPU allocation
arr_cpu = tvm.nd.empty((128, 128), device=tvm.cpu())

# GPU allocation
arr_gpu = tvm.nd.empty((128, 128), device=tvm.cuda(0))

# From numpy (copies to device)
arr = tvm.nd.array(np.random.randn(128, 128), device=tvm.cuda(0))
```

### Data Transfer
```python
# CPU → GPU
arr_gpu = arr_cpu.copyto(tvm.cuda(0))

# GPU → CPU
arr_cpu = arr_gpu.copyto(tvm.cpu())

# GPU → GPU (different devices)
arr_gpu1 = arr_gpu0.copyto(tvm.cuda(1))

# To numpy (GPU → CPU → numpy)
np_arr = arr_gpu.numpy()
```

### Zero-copy with DLPack
```python
import torch
import tvm

# TVM → PyTorch (zero copy)
tvm_arr = tvm.nd.empty((128, 128), device=tvm.cuda(0))
torch_tensor = torch.from_dlpack(tvm_arr)

# PyTorch → TVM (zero copy)
torch_tensor = torch.randn(128, 128, device="cuda")
tvm_arr = tvm.nd.from_dlpack(torch_tensor)
```

### Memory Pool
TVM uses a pool allocator for efficient memory management:
- Small allocations served from pools
- Large allocations go to device API directly
- Pool reuse reduces allocation overhead

---

## Stream Management

### Creating Streams
```python
# Create CUDA stream
stream = tvm.cuda(0).create_stream()
```

### Asynchronous Execution
```python
# Execute on specific stream
with tvm.cuda(0).stream(stream):
    result = vm["main"](data)
```

### Synchronization
```python
# Synchronize device
tvm.cuda(0).sync()

# Synchronize specific stream
stream.sync()
```

---

## Multi-Device Scenarios

### Multi-GPU (Same Machine)
```python
# Build once, deploy on multiple GPUs
exec = relax.build(mod, target="cuda")

# Create VMs for each GPU
vm0 = relax.VirtualMachine(exec, tvm.cuda(0))
vm1 = relax.VirtualMachine(exec, tvm.cuda(1))
```

### Tensor Parallelism (via Disco)
```python
from tvm.runtime import disco

session = disco.ThreadedSession(num_workers=4)
dmod = session.import_module(exec)
result = dmod["main"](input_data)
```

### Cross-Compilation
```python
# Build on x86 for ARM
target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu")
exec = relax.build(mod, target=target)

# Export and deploy on ARM device
exec.export_library("model_arm.so")
```

---

## Target Comparison

| Target | Device | Codegen | Vector Support | Memory |
|--------|--------|---------|----------------|--------|
| llvm | CPU | LLVM IR | SSE/AVX/NEON | DDR |
| cuda | NVIDIA GPU | CUDA C | CUDA cores/Tensor | GDDR/HBM |
| rocm | AMD GPU | ROCm | CDNA/RDNA | HBM/GDDR |
| metal | Apple | Metal | Apple GPU | Unified |
| opencl | Cross | OpenCL C | Varies | Varies |
| vulkan | Cross | SPIR-V | Varies | Varies |
| hexagon | Qualcomm | Hexagon | HVX | LPDDR |
