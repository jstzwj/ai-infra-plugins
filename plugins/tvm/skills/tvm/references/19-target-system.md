# Target System - Hardware Target Configuration

This reference covers TVM's target system, which describes compilation targets and drives code generation. The target system provides abstractions for specifying hardware characteristics, configuring compilation pipelines, and selecting appropriate code generation backends.

---

## 19.1 Overview

The `tvm.target.Target` class is the central abstraction for describing compilation targets. A target encapsulates all information needed to generate code for a specific hardware platform: the instruction set, memory hierarchy, threading model, available intrinsics, and code generation backend.

Every compilation operation in TVM requires a target:
- `relax.build(mod, target=...)` -- build a Relax module.
- `tvm.build(mod, target=...)` -- build a TIR module.
- `tvm.tir.transform.LowerOpaqueBlock(target=...)` -- lowering passes.
- MetaSchedule tuning -- target-aware auto-tuning.

---

## 19.2 The Target Class

### 19.2.1 Construction from Tag

TVM pre-registers common hardware targets as tags for convenience:

```python
import tvm
from tvm.target import Target

# Create from registered tag string
target_cuda = Target("cuda")                         # Generic CUDA
target_nvidia_a100 = Target("nvidia/nvidia-a100")   # NVIDIA A100
target_nvidia_v100 = Target("nvidia/nvidia-v100")   # NVIDIA V100
target_nvidia_t4 = Target("nvidia/nvidia-t4")       # NVIDIA T4
target_rocm = Target("rocm")                         # AMD ROCm
target_metal = Target("metal")                       # Apple Metal
target_vulkan = Target("vulkan")                     # Vulkan compute
target_opencl = Target("opencl")                     # OpenCL
target_llvm = Target("llvm")                         # Generic LLVM (CPU)
target_arm_cpu = Target("llvm -device=arm_cpu")     # ARM CPU via LLVM
target_hexagon = Target("hexagon")                   # Qualcomm Hexagon
```

### 19.2.2 Construction from Configuration Dict

For fine-grained control, construct a target from a configuration dictionary:

```python
# CUDA with specific architecture
target_cuda = Target({
    "kind": "cuda",
    "arch": "sm_80",
    "max_threads_per_block": 1024,
    "max_shared_memory_per_block": 49152,
    "thread_warp_size": 32,
})

# x86 CPU with specific features
target_x86 = Target({
    "kind": "llvm",
    "mcpu": "skylake-avx512",
    "mattr": ["+avx512bw", "+avx512vl", "+avx512vnni"],
})

# ARM CPU with specific architecture
target_arm = Target({
    "kind": "llvm",
    "mcpu": "cortex-a78",
    "mattr": ["+v8.4a", "+dotprod"],
})

# Hexagon DSP
target_hexagon = Target({
    "kind": "hexagon",
    "mcpu": "v68",
    "llvm_options": ["-mllvm", "-hexagon-align-loads"],
})
```

### 19.2.3 Construction with Target String

TVM supports a target string format that encodes key-value pairs:

```python
# LLVM target with CPU and features
target = Target("llvm -mcpu=skylake-avx512 -mattr=+avx512vl,+avx512vnni")

# CUDA with architecture
target = Target("cuda -arch=sm_80")

# ARM CPU
target = Target("llvm -device=arm_cpu -mattr=+v8.4a,+dotprod")

# ROCm with architecture
target = Target("rocm -mcpu=gfx908")
```

### 19.2.4 Target with Host

For heterogeneous compilation (GPU + CPU host), specify both device and host targets:

```python
# CUDA device with LLVM host
target = Target(
    "cuda",
    host="llvm",
)

# Or with explicit host target
target = Target(
    {"kind": "cuda", "arch": "sm_80"},
    host=Target({"kind": "llvm", "mcpu": "skylake-avx512"}),
)
```

### 19.2.5 Listing Available Targets

```python
from tvm.target import Target

# List all registered target kinds
kinds = Target.list_kinds()
print(kinds)
# Output: ['cuda', 'rocm', 'metal', 'opencl', 'vulkan', 'webgpu',
#          'llvm', 'hexagon', 'source', 'c', 'ext_dev', ...]

# List registered target tags
tags = Target.list_tags()
print(tags)
# Output: ['nvidia/nvidia-a100', 'nvidia/nvidia-v100', 'nvidia/nvidia-t4', ...]
```

---

## 19.3 Target Attributes

### 19.3.1 Core Attributes

Every target has the following core attributes:

```python
target = Target("nvidia/nvidia-a100")

target.kind          # TargetKind: "cuda"
target.kind_name     # str: "cuda"
target.keys          # list[str]: ["cuda", "gpu"]
target.device_type   # int: DLDeviceType (2 for CUDA)
target.attrs         # dict-like: target-specific attributes
```

### 19.3.2 CUDA Target Attributes

```python
target = Target("nvidia/nvidia-a100")

# Architecture
target.attrs["arch"]                    # "sm_80"
target.attrs["max_threads_per_block"]   # 1024
target.attrs["max_shared_memory_per_block"]  # 49152 (bytes)
target.attrs["thread_warp_size"]        # 32
target.attrs["registers_per_block"]     # 65536
target.attrs["max_smem_per_sm"]         # varies by GPU

# Feature flags
target.attrs.get("supports_int8", False)        # True for sm_75+
target.attrs.get("supports_tensor_core", False) # True for sm_70+
target.attrs.get("supports_bf16", False)        # True for sm_80+
target.attrs.get("supports_tf32", False)        # True for sm_80+
```

### 19.3.3 LLVM Target Attributes

```python
target = Target("llvm -mcpu=skylake-avx512 -mattr=+avx512vnni")

target.attrs["mcpu"]    # "skylake-avx512"
target.attrs["mattr"]   # ["+avx512vnni"]
target.attrs["mtriple"] # target triple (e.g., "x86_64-unknown-linux-gnu")
```

### 19.3.4 Querying Target Features

```python
target = Target("nvidia/nvidia-a100")

# Check if target is GPU
is_gpu = "gpu" in target.keys  # True
is_cuda = "cuda" in target.keys  # True

# Check specific feature
has_tensor_core = target.attrs.get("supports_tensor_core", False)

# Safe attribute access with default
warp_size = target.attrs.get("thread_warp_size", 1)
```

---

## 19.4 Supported Targets

### 19.4.1 CUDA (NVIDIA GPUs)

CUDA is the primary GPU target in TVM, generating CUDA C kernel code compiled by NVCC.

```python
from tvm.target import Target

# Common CUDA configurations
targets = {
    "v100": Target("nvidia/nvidia-v100"),     # SM 7.0, 16GB HBM2
    "t4": Target("nvidia/nvidia-t4"),         # SM 7.5, 16GB GDDR6
    "a100": Target("nvidia/nvidia-a100"),     # SM 8.0, 40/80GB HBM2e
    "a10g": Target("nvidia/nvidia-a10g"),     # SM 8.6, 24GB GDDR6
}
```

**Key configuration options**:

| Option | Description | Default |
|---|---|---|
| `arch` | SM architecture (sm_70, sm_75, sm_80, sm_86, sm_90) | Auto-detected |
| `max_threads_per_block` | Maximum threads per CUDA block | 1024 |
| `max_shared_memory_per_block` | Shared memory per block (bytes) | 49152 |
| `thread_warp_size` | Threads per warp | 32 |
| `registers_per_block` | Max registers per block | 65536 |

**Generated code format**: CUDA C (`.cu` files), compiled via NVCC.

### 19.4.2 ROCm (AMD GPUs)

ROCm targets AMD GPUs using HIP/ROCm compilation.

```python
target_rocm = Target("rocm")

# Specific AMD GPU
target_mi200 = Target({
    "kind": "rocm",
    "mcpu": "gfx90a",        # MI250X
    "max_threads_per_block": 1024,
})

target_mi300 = Target({
    "kind": "rocm",
    "mcpu": "gfx942",        # MI300X
})
```

**Key configuration options**:

| Option | Description | Default |
|---|---|---|
| `mcpu` | GPU architecture (gfx906, gfx908, gfx90a, gfx942) | Auto-detected |
| `max_threads_per_block` | Maximum threads per block | 1024 |
| `thread_warp_size` | Wavefront size | 64 (AMD) or 32 |

**Generated code format**: HIP C (`.cpp` files), compiled via HIP compiler.

### 19.4.3 Metal (Apple)

Metal targets Apple GPUs (M1, M2, M3, M4 series and Apple Silicon).

```python
target_metal = Target("metal")
```

**Generated code format**: Metal Shading Language (`.metal` files), compiled via Metal compiler.

### 19.4.4 OpenCL

OpenCL provides cross-platform GPU compute support.

```python
target_opencl = Target("opencl")

# With specific device
target_opencl = Target({
    "kind": "opencl",
    "device": "adreno",       # Qualcomm Adreno GPU
    "max_threads_per_block": 256,
})
```

**Generated code format**: OpenCL C (`.cl` files).

### 19.4.5 Vulkan

Vulkan compute shaders for cross-platform GPU acceleration.

```python
target_vulkan = Target("vulkan")

# With SPIR-V options
target_vulkan = Target({
    "kind": "vulkan",
    "max_threads_per_block": 256,
    "supports_float16": True,
    "supports_int8": True,
    "supports_16bit_buffer": True,
})
```

**Generated code format**: SPIR-V binary via GLSL/HLSL compilation.

### 19.4.6 WebGPU

WebGPU targets web browser GPU compute via WGSL (WebGPU Shading Language).

```python
target_webgpu = Target("webgpu")
```

**Generated code format**: WGSL (WebGPU Shading Language).

### 19.4.7 LLVM (x86, ARM, RISC-V CPUs)

The LLVM target generates native CPU code via the LLVM backend. It supports a wide range of CPU architectures.

```python
# Generic x86
target_x86 = Target("llvm")

# Specific x86 microarchitectures
target_skylake = Target("llvm -mcpu=skylake-avx512")
target_cascadelake = Target("llvm -mcpu=cascadelake")
target_sapphirerapids = Target("llvm -mcpu=sapphirerapids")

# ARM CPUs
target_cortex_a76 = Target("llvm -device=arm_cpu -mcpu=cortex-a76")
target_cortex_a78 = Target("llvm -device=arm_cpu -mcpu=cortex-a78")
target_neoverse_n1 = Target("llvm -device=arm_cpu -mcpu=neoverse-n1")
target_neoverse_v2 = Target("llvm -device=arm_cpu -mcpu=neoverse-v2")

# RISC-V CPUs
target_rv64 = Target("llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=rocket-rv64")
target_rv64_v = Target("llvm -mtriple=riscv64-unknown-linux-gnu -mattr=+v")

# With specific features
target_avx512_vnni = Target({
    "kind": "llvm",
    "mcpu": "cascadelake",
    "mattr": ["+avx512bw", "+avx512vl", "+avx512vnni", "+avx512bf16"],
})
```

**Key configuration options**:

| Option | Description | Example |
|---|---|---|
| `mcpu` | CPU microarchitecture | `skylake-avx512`, `cortex-a78` |
| `mattr` | CPU feature flags | `["+avx512vnni", "+dotprod"]` |
| `mtriple` | Target triple | `aarch64-linux-gnu`, `riscv64-linux-gnu` |
| `num-cores` | Number of CPU cores | `4`, `8` |
| `llvm-options` | Additional LLVM options | `["-mllvm", "-x86-cmov-converter=false"]` |

**Generated code format**: LLVM IR, compiled to native object files (`.o`).

### 19.4.8 Hexagon (Qualcomm DSP)

Hexagon targets Qualcomm DSP processors with HVX vector extensions.

```python
target_hexagon = Target("hexagon")

# Specific Hexagon version
target_hex_v68 = Target({
    "kind": "hexagon",
    "mcpu": "v68",
    "llvm_options": ["-mllvm", "-hexagon-align-loads"],
})

target_hex_v69 = Target({
    "kind": "hexagon",
    "mcpu": "v69",
})
```

**Key configuration options**:

| Option | Description | Example |
|---|---|---|
| `mcpu` | Hexagon version | `v68`, `v69`, `v73` |
| `llvm_options` | Additional LLVM flags | Various optimization flags |
| `hvx_length` | HVX vector length in bytes | `128` (default) |

**Generated code format**: LLVM IR compiled for Hexagon architecture.

### 19.4.9 Source (C Source Code Generation)

The source backend generates C code that can be compiled externally.

```python
target_c = Target("c")
```

**Generated code format**: Portable C code (`.c` files).

---

## 19.5 Code Generation Backends

### 19.5.1 LLVM Backend

The LLVM backend generates optimized native code via the LLVM compiler infrastructure. It is the primary backend for CPU targets.

**Capabilities**:
- Full LLVM IR generation from TIR.
- Support for SIMD intrinsics (AVX2, AVX-512, NEON, HVX).
- Auto-vectorization and loop optimizations.
- Debug information generation.
- Position-independent code (PIC).

**Pipeline**:
```
TIR PrimFunc
     |
     v
TIR -> LLVM IR lowering (src/target/llvm/llvm.cc)
     |
     v
LLVM IR Module
     |
     v
LLVM optimization passes (O2/O3)
     |
     v
Native object code (.o)
```

**Usage**:
```python
import tvm
from tvm.target import Target

target = Target("llvm -mcpu=skylake-avx512")
rt_mod = tvm.build(mod, target=target)

# The resulting module contains native x86 code
```

### 19.5.2 Source Backend

The source backend generates C-like source code for GPU and other targets.

**Supported source targets**:
- **CUDA C**: For NVIDIA GPUs.
- **OpenCL C**: For OpenCL devices.
- **Metal Shading Language**: For Apple Metal.
- **GLSL/SPIR-V**: For Vulkan.

**Pipeline**:
```
TIR PrimFunc
     |
     v
TIR -> Source code lowering (src/target/source/)
     |
     v
Source code string (.cu, .cl, .metal, .comp)
     |
     v
External compiler invocation (nvcc, clang, metalc, etc.)
     |
     v
Compiled binary
```

### 19.5.3 External Backends

TVM integrates with external libraries and code generators for specific operations.

**CUTLASS** (NVIDIA Tensor Core GEMM):
```python
from tvm.contrib import cutlass

# CUTLASS is dispatched via BYOC/external codegen
target = Target("nvidia/nvidia-a100")
# During compilation, TVM may dispatch matmul operations to CUTLASS
```

**TensorRT** (NVIDIA inference):
```python
from tvm.contrib import tensorrt

# TensorRT integration for NVIDIA GPU inference
# Operations are partitioned and offloaded to TensorRT
```

**cuBLAS** (NVIDIA BLAS):
```python
from tvm.contrib import cublas

# cuBLAS matmul dispatch
```

**cuDNN** (NVIDIA DNN):
```python
from tvm.contrib import cudnn

# cuDNN convolution dispatch
```

**External codegen registration**:
```python
# Register a custom external codegen
@tvm.register_func("target.build.my_custom_codegen")
def my_custom_codegen(mod, target):
    # Generate custom code from the IRModule
    return compiled_module
```

---

## 19.6 Multi-Target Compilation

### 19.6.1 Host + Device Compilation

For GPU targets, TVM compiles host code (CPU) and device code (GPU) separately:

```python
import tvm
from tvm.target import Target

# Define both targets
target = Target("cuda", host="llvm")

# Build with host + device
rt_mod = tvm.build(mod, target=target)

# The resulting module contains:
# - Host code (LLVM compiled)
# - Device code (CUDA C compiled)
rt_mod.import_modules  # List device modules
```

### 19.6.2 Manual Multi-Target Build

```python
from tvm.target import Target

# Create separate targets
target_cuda = Target("cuda")
target_host = Target("llvm -mcpu=skylake-avx512")

# Build with explicit targets
target = Target(target_cuda, host=target_host)
rt_mod = tvm.build(mod, target=target)
```

### 19.6.3 Heterogeneous Compilation

For systems with multiple GPU types:

```python
# Compile for multiple devices
# Note: This requires explicit device assignment in the IR
from tvm.target import Target

target_cuda = Target("nvidia/nvidia-a100")
target_rocm = Target("rocm")

# TVM supports heterogeneous compilation where different
# parts of the graph target different devices
```

---

## 19.7 Target Configuration in Pipelines

### 19.7.1 Relax Pipeline Configuration

```python
import tvm
from tvm import relax
from tvm.target import Target

target = Target("nvidia/nvidia-a100")

# Apply optimization pipeline with target awareness
# The zero pipeline adapts based on target
mod = relax.get_pipeline("zero")(mod)

# Build with target
exec = relax.build(mod, target=target)
```

### 19.7.2 MetaSchedule Target Configuration

```python
from tvm import meta_schedule as ms
from tvm.target import Target

target = Target("nvidia/nvidia-a100")

# MetaSchedule uses the target to:
# 1. Determine available hardware intrinsics
# 2. Set search space bounds (thread limits, shared memory sizes)
# 3. Select schedule rules (tensorize for Tensor Cores, etc.)
database = ms.tune_tir(
    mod=mod,
    target=target,
    max_trials_global=1000,
    work_dir="./tune_logs",
)
```

### 19.7.3 DLight Target Configuration

```python
from tvm import dlight
from tvm.target import Target

target = Target("nvidia/nvidia-a100")

with target:
    # DLight rules are target-aware
    sch = dlight.gpu.Matmul().apply(sch, target)
```

### 19.7.4 PassContext Target Configuration

```python
from tvm import transform
from tvm.target import Target

target = Target("nvidia/nvidia-a100")

with transform.PassContext(opt_level=3):
    with target:
        # Passes that run inside this context can query the target
        mod = tir.transform.LowerOpaqueBlock()(mod)
        mod = tir.transform.VectorizeLoop()(mod)
```

---

## 19.8 Target-Aware Compilation Flow

### 19.8.1 End-to-End Example

```python
import tvm
from tvm import relax
from tvm.target import Target
from tvm.script import ir as I, tirx as T, relax as R

# Define the model
@I.ir_module
class Model:
    @T.prim_func
    def matmul(
        A: T.Buffer((128, 64), "float32"),
        B: T.Buffer((64, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
        for i, j, k in T.grid(128, 128, 64):
            with T.sblock("C"):
                with T.init():
                    C[i, j] = T.float32(0.0)
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] += A[vi, vk] * B[vk, vj]

    @R.function
    def main(
        x: R.Tensor((128, 64), "float32"),
        w: R.Tensor((64, 128), "float32"),
    ) -> R.Tensor((128, 128), "float32"):
        with R.dataflow():
            lv = R.call_tir(
                Model.matmul,
                (x, w),
                out_sinfo=R.Tensor((128, 128), "float32"),
            )
            R.output(lv)
        return lv

# Step 1: Optimize
mod = relax.get_pipeline("zero")(Model)

# Step 2: Build for CUDA
target = Target("nvidia/nvidia-a100")
exec = relax.build(mod, target=target)

# Step 3: Deploy
dev = tvm.cuda(0)
vm = relax.VirtualMachine(exec, dev)
result = vm["main"](x_np, w_np)
```

### 19.8.2 Target Selection Logic

When building for a specific target, TVM performs:

1. **Target validation**: Verify the target is registered and has valid configuration.
2. **Pass selection**: Choose optimization passes based on target capabilities.
3. **Code generation**: Select the appropriate backend (LLVM, Source, External).
4. **Runtime configuration**: Set up the runtime module with correct device handling.

```python
# Inspect what codegen path is used
target = Target("nvidia/nvidia-a100")
print(target.kind_name)  # "cuda"
print(target.keys)       # ["cuda", "gpu"]

# This triggers:
# 1. Source codegen for CUDA kernels
# 2. LLVM codegen for host code
# 3. Runtime module wrapping both
```

---

## 19.9 Custom Target Registration

### 19.9.1 Registering a New Target Kind

```python
from tvm.target import TargetKind, Target

# Define a new target kind
@TargetKind.register("my_custom_target")
def _register_my_target():
    # Set default attributes
    attrs = {
        "max_threads_per_block": 256,
        "thread_warp_size": 8,
        "supports_float16": True,
        "supports_int8": False,
    }
    return attrs

# Use the custom target
target = Target("my_custom_target")
```

### 19.9.2 Registering a Target Tag

```python
from tvm.target import Target

# Register a pre-configured target tag
Target.register_tag(
    "my_device/my_custom_v1",
    Target({
        "kind": "my_custom_target",
        "version": "v1",
        "max_threads_per_block": 512,
    }),
)
```

### 19.9.3 Custom Codegen Integration

```python
import tvm
from tvm.target import Target

# Register a codegen function for the custom target
@tvm.register_func("target.build.my_custom_target")
def build_my_custom_target(mod, target):
    """
    Custom code generation function.

    Parameters
    ----------
    mod : IRModule
        The IRModule to compile.
    target : Target
        The target configuration.

    Returns
    -------
    runtime.Module
        The compiled module.
    """
    # Implement custom code generation here
    # E.g., generate custom ISA, call external compiler, etc.
    from tvm import runtime
    return runtime.Module.empty()
```

### 19.9.4 Custom Target Attributes

```python
from tvm.target import Target, TargetKind

# Add custom attributes to an existing target kind
TargetKind("cuda").add_attr_option(
    "my_custom_attr",
    default_value=0,
    description="Custom attribute for CUDA target",
)

# Use the custom attribute
target = Target({
    "kind": "cuda",
    "arch": "sm_80",
    "my_custom_attr": 42,
})
```

---

## 19.10 Target and Device Interaction

### 19.10.1 Device Type Mapping

Each target kind maps to a DLDeviceType:

| Target Kind | DLDeviceType | Value | Description |
|---|---|---|---|
| `llvm` | `kDLCPU` | 1 | CPU |
| `cuda` | `kDLCUDA` | 2 | NVIDIA GPU |
| `rocm` | `kDLROCM` | 10 | AMD GPU |
| `metal` | `kDLMetal` | 8 | Apple Metal |
| `opencl` | `kDLOpenCL` | 4 | OpenCL |
| `vulkan` | `kDLVulkan` | 7 | Vulkan |
| `webgpu` | `kDLWebGPU` | 19 | WebGPU |
| `hexagon` | `kDLHexagon` | 12 | Qualcomm Hexagon |

### 19.10.2 Querying Device from Target

```python
from tvm.target import Target
import tvm

target = Target("cuda")
dev = tvm.device(target.device_type, 0)
print(dev)  # device(type=cuda, index=0)
```

### 19.10.3 Memory and Threading Constraints

The target affects generated code in several ways:

**Thread configuration**: GPU targets use thread/block dimensions; CPU targets use SIMD widths.

```python
# CUDA: bind loops to threadIdx/blockIdx
# sch.bind(i, "threadIdx.x")
# sch.bind(j, "blockIdx.x")

# LLVM: vectorize loops to SIMD width
# sch.vectorize(i)  # vectorized to AVX512 (64 bytes for float32)
```

**Memory hierarchy**: Different targets have different memory scopes.

```python
# CUDA memory scopes
# "global"     -> device global memory (DRAM)
# "shared"     -> shared memory (on-chip, per-block)
# "local"      -> local memory (per-thread registers)
# "shared.dyn" -> dynamically allocated shared memory

# LLVM memory scopes
# "global"     -> main memory
# "local"      -> stack/registers (auto-promoted)
```

---

## 19.11 Target Comparison Reference

| Feature | CUDA | ROCm | Metal | OpenCL | Vulkan | LLVM | Hexagon |
|---|---|---|---|---|---|---|---|
| **Vendor** | NVIDIA | AMD | Apple | Cross | Cross | LLVM | Qualcomm |
| **Backend** | Source (NVCC) | Source (HIP) | Source (metalc) | Source (clang) | SPIR-V | LLVM IR | LLVM IR |
| **Host Code** | LLVM | LLVM | LLVM | LLVM | LLVM | N/A | LLVM |
| **Thread Model** | blockIdx/threadIdx | blockIdx/threadIdx | threadgroup/thread | workgroup/workitem | workgroup/workitem | OpenMP/SIMD | HVX |
| **Tensor Intrinsics** | Tensor Core (WMMA/MMA) | MFMA | SIMD | Varies | Varies | VNNI/AMX | VRMPY |
| **FP16** | CC 5.3+ | gfx900+ | Apple GPU | Varies | Varies | AVX512_FP16 | v68+ |
| **BF16** | CC 8.0+ | gfx90a+ | No | Varies | Varies | AVX512_BF16 | No |
| **INT8** | CC 6.1+ | gfx900+ | Apple GPU | Varies | Varies | VNNI | HVX |
| **FP8** | CC 8.9+ (Hopper) | gfx942+ | No | No | No | No | No |

---

## 19.12 Source Code Locations

| Component | Path |
|---|---|
| Target Python API | `python/tvm/target/target.py` |
| TargetKind registration | `python/tvm/target/target_kind.py` |
| Target tag definitions | `python/tvm/target/tag.py` |
| CUDA target config | `python/tvm/target/cuda.py` |
| ROCm target config | `python/tvm/target/rocm.py` |
| ARM target config | `python/tvm/target/arm.py` |
| Hexagon target config | `python/tvm/target/hexagon.py` |
| LLVM codegen | `src/target/llvm/llvm.cc` |
| CUDA source codegen | `src/target/source/literal/cuda.cc` |
| OpenCL source codegen | `src/target/source/codegen_opencl.cc` |
| Metal source codegen | `src/target/source/codegen_metal.cc` |
| Vulkan source codegen | `src/target/source/codegen_vulkan.cc` |
| Hexagon codegen | `src/target/hexagon/` |
| Target C++ implementation | `src/target/target.cc` |
| Target kind C++ registry | `src/target/target_kind.cc` |
