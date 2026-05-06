# Apache TVM — Chapter 29: Code Generation

This reference covers TVM's code generation system, which translates optimized TIR (Tensor IR) programs into target-specific executable code. Code generation is the final compilation phase, producing runtime-loadable modules for CPUs, GPUs, and specialized accelerators.

---

## 29.1 Code Generation Overview

### The Role of Code Generation

Code generation bridges the gap between TVM's intermediate representation (TIR) and actual machine code that executes on target hardware. It is responsible for:

- **Translating TIR to target representation**: Converting high-level tensor operations into low-level instructions (LLVM IR, CUDA C, SPIR-V, etc.).
- **Handling memory hierarchy**: Mapping TIR buffer accesses to physical memory types (registers, shared memory, global memory).
- **Managing parallelism**: Translating TIR loop bindings to hardware threading models (CPU threads, CUDA thread blocks, etc.).
- **Generating runtime glue**: Creating PackedFunc wrappers that interface with TVM's runtime system.
- **Optimizing for target**: Applying target-specific optimizations that LLVM or device compilers can perform better.

### Code Generation Pipeline

```
TIR (Tensor IR)
      |
      v
[TIR Lowering]
  ├── FlattenBuffer         -- Multi-dimensional -> 1D buffer access
  ├── LowerIntrin           -- TIR intrinsics -> target instructions
  ├── LowerDeviceStorageAccess  -- Device memory access lowering
  ├── VectorizeLoop         -- Scalar loops -> vector instructions
  ├── Simplify              -- Simplify index expressions
  └── VerifyMemory          -- Verify legal memory accesses
      |
      v
[Target-Specific Code Generation]
  ├── LLVM Backend          -- Generates LLVM IR -> machine code
  ├── CUDA Backend          -- Generates CUDA C -> PTX/CUBIN
  ├── OpenCL Backend        -- Generates OpenCL C -> binary
  ├── Metal Backend         -- Generates Metal Shading Language
  ├── Vulkan Backend        -- Generates SPIR-V compute shaders
  ├── WebGPU Backend        -- Generates WGSL compute shaders
  └── Hexagon Backend       -- Generates Hexagon DSP code
      |
      v
[Compilation to Binary]
  ├── LLVM -> obj/so (clang)
  ├── CUDA C -> PTX/CUBIN (nvcc)
  ├── OpenCL C -> binary (driver)
  └── SPIR-V -> binary (driver)
      |
      v
runtime.Module (ready for loading and execution)
```

### Build API Entry Points

```python
import tvm
from tvm import relax, tir

# Build a TIR PrimFunc to runtime.Module
rt_mod = tvm.build(tir_mod, target="llvm")

# Build a Relax IRModule to executable
exec_mod = relax.build(relax_mod, target="nvidia/nvidia-a100")

# Build with host/device target specification
target = tvm.target.Target("cuda", host="llvm")
rt_mod = tvm.build(mod, target=target)
```

---

## 29.2 LLVM Backend

### 29.2.1 Overview

The LLVM backend is TVM's primary CPU code generation path. It translates TIR to LLVM IR using LLVM's IRBuilder, then uses LLVM's optimization passes and code generation to produce native machine code.

**Supported architectures**: x86 (SSE, AVX, AVX2, AVX-512), ARM (NEON, SVE), RISC-V (Vector Extension), PowerPC, WebAssembly.

### 29.2.2 Target Configuration

```python
import tvm

# x86 with AVX-512 (e.g., Intel Xeon Scalable)
target_avx512 = tvm.target.Target("llvm -mcpu=skylake-avx512")

# x86 with AVX2 (e.g., Intel Core, AMD Zen)
target_avx2 = tvm.target.Target("llvm -mcpu=haswell")

# x86 with SSE4.2 (older Intel/AMD)
target_sse = tvm.target.Target("llvm -mcpu=nehalem")

# ARM Cortex-A (e.g., AWS Graviton)
target_arm = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mattr=+neon")

# ARM with SVE (e.g., Fujitsu A64FX)
target_arm_sve = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mattr=+sve")

# RISC-V with Vector Extension
target_riscv = tvm.target.Target("llvm -mtriple=riscv64-linux-gnu -mattr=+v")

# WebAssembly
target_wasm = tvm.target.Target("llvm -mtriple=wasm32-unknown-unknown")

# Generic LLVM (no specific CPU)
target_generic = tvm.target.Target("llvm")
```

### 29.2.3 LLVM Code Generation Flow

```
TIR PrimFunc
      |
      v
[LLVM IRBuilder]
  1. Create LLVM Function (with correct signature)
  2. Create entry BasicBlock
  3. Walk TIR stmt tree:
     - For loops -> LLVM loop IR
     - If-then-else -> LLVM branch IR
     - Buffer stores -> LLVM store instructions
     - Buffer loads -> LLVM load instructions
     - Arithmetic -> LLVM compute instructions
     - Calls -> LLVM call instructions
  4. Generate PackedFunc wrapper (MakePackedAPI)
      |
      v
[LLVM Optimization Passes]
  - Function inlining
  - Constant folding
  - Loop unrolling
  - Vectorization (SLP, loop vectorization)
  - Dead code elimination
  - Register allocation
      |
      v
[LLVM Code Generation]
  - Select target machine code
  - Emit object file (.o)
      |
      v
[System Linker]
  - Link with TVM runtime
  - Produce shared library (.so)
      |
      v
runtime.Module (LLVMModule)
```

### 29.2.4 Vector Instructions

TVM leverages LLVM's auto-vectorization and also generates explicit vector intrinsics:

```python
# AVX-512 vector operations
# LLVM auto-vectorizes loops like:
#   for i in range(0, 128):
#       C[i] = A[i] + B[i]
# Into:
#   for i in range(0, 128, 16):  # 16 x float32 per AVX-512 register
#       zmm0 = load(A + i)
#       zmm1 = load(B + i)
#       zmm2 = add(zmm0, zmm1)
#       store(C + i, zmm2)

# ARM NEON vectorization
# 4 x float32 per NEON register (128-bit)
target_arm = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mattr=+neon")
```

### 29.2.5 Function Calling Convention

TVM generates PackedFunc wrappers for all compiled functions. The `MakePackedAPI` pass transforms TIR functions to accept TVM's universal calling convention:

```python
# Before MakePackedAPI (TIR function):
@T.prim_func
def add(A: T.Buffer((128,), "float32"),
        B: T.Buffer((128,), "float32")) -> None:
    for i in range(128):
        with T.sblock("add"):
            vi = T.axis.spatial(128, i)
            B[vi] = B[vi] + A[vi]

# After MakePackedAPI (PackedFunc wrapper):
# The function signature is transformed to:
# void add(TVMValue* args, int* type_codes, int num_args,
#           TVMValue* out_ret, int* out_ret_type_code)
#
# The wrapper extracts DLTensor pointers from the packed args
# and calls the inner compute function
```

### 29.2.6 LLVM Debug Information

```python
import tvm

# Enable debug information in LLVM code generation
with tvm.transform.PassContext(config={
    "tir.instrument_bound_checkers": True,
    "tir.disable_assert": False,
}):
    rt_mod = tvm.build(mod, target="llvm")

# The generated LLVM IR includes:
# - Source location mapping (TVM script line numbers)
# - Variable name preservation
# - Debug symbols in the compiled binary
```

### 29.2.7 LLVM JIT Compilation

```python
# TVM supports JIT compilation via LLVM's ORC JIT
# This is useful for development and testing

import tvm

# Build with LLVM backend
rt_mod = tvm.build(mod, target="llvm")

# The module is compiled in-memory
# Functions are available immediately
func = rt_mod["main"]
result = func(input_ndarray)

# No need to write to disk and reload
```

### 29.2.8 LLVM Optimization Level

```python
# Control LLVM optimization level
# Default is -O2 (good balance of compile time and performance)

# Option 1: Through Target string
target_o3 = tvm.target.Target("llvm -O3")  # Aggressive optimization
target_o0 = tvm.target.Target("llvm -O0")  # No optimization (debugging)

# Option 2: Through PassContext
with tvm.transform.PassContext(opt_level=3):
    rt_mod = tvm.build(mod, target="llvm")
```

---

## 29.3 Source Code Backends

Source code backends generate textual source code (e.g., CUDA C, OpenCL C) that is then compiled by external compilers (nvcc, clang, etc.).

### 29.3.1 CUDA Backend

#### Overview

The CUDA backend generates CUDA C kernel source code from TIR, which is compiled to PTX and CUBIN using NVIDIA's nvcc compiler.

#### Target Configuration

```python
import tvm

# NVIDIA A100 (Ampere, SM80)
target_a100 = tvm.target.Target("nvidia/nvidia-a100")

# NVIDIA V100 (Volta, SM70)
target_v100 = tvm.target.Target("nvidia/nvidia-v100")

# NVIDIA T4 (Turing, SM75)
target_t4 = tvm.target.Target("nvidia/nvidia-t4")

# NVIDIA H100 (Hopper, SM90)
target_h100 = tvm.target.Target("nvidia/nvidia-h100")

# Generic CUDA target
target_cuda = tvm.target.Target("cuda")
```

#### CUDA Code Generation Flow

```
TIR PrimFunc
      |
      v
[Lower TIR for CUDA]
  ├── LowerWarpMemory      -- Warp-level memory -> shuffle instructions
  ├── LowerDeviceStorageAccess  -- Shared/global memory access
  ├── LowerIntrin          -- TIR intrinsics -> CUDA builtins
  └── VectorizeLoop        -- (disabled for CUDA; manual vectorization)
      |
      v
[Generate CUDA C Source]
  - Thread index macros: threadIdx.x/y/z, blockIdx.x/y/z
  - Block/grid dimension macros: blockDim.x/y/z, gridDim.x/y/z
  - Shared memory: __shared__ declarations
  - Synchronization: __syncthreads()
  - Math functions: __fmaf_rn(), __expf(), etc.
  - Warp intrinsics: __shfl_sync(), __ballot_sync()
      |
      v
[Compile with nvcc]
  nvcc -arch=sm_80 -O3 kernel.cu -o kernel.cubin
  nvcc -arch=sm_80 -O3 kernel.cu -o kernel.ptx
      |
      v
[CUDAModule]
  - Contains PTX and/or CUBIN binary
  - Loaded via cuModuleLoadData at runtime
  - Kernel launched via cuLaunchKernel
```

#### Generated CUDA C Example

```python
# Input TIR:
@T.prim_func
def matmul(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
) -> None:
    # Simplified for illustration
    for i, j, k in T.grid(128, 128, 128):
        with T.sblock("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float32(0.0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# After scheduling with thread binding, generates CUDA C:
__global__ void matmul_kernel(
    float* __restrict__ A,
    float* __restrict__ B,
    float* __restrict__ C
) {
    int vi = blockIdx.y * blockDim.y + threadIdx.y;
    int vj = blockIdx.x * blockDim.x + threadIdx.x;
    if (vi < 128 && vj < 128) {
        float sum = 0.0f;
        for (int vk = 0; vk < 128; ++vk) {
            sum += A[vi * 128 + vk] * B[vk * 128 + vj];
        }
        C[vi * 128 + vj] = sum;
    }
}
```

#### Thread Binding for CUDA

```python
# In TensorIR schedule, bind loops to CUDA thread hierarchy
sch = tir.Schedule(mod)

# Bind to CUDA threads
sch.bind(loop_i, "blockIdx.y")
sch.bind(loop_j, "blockIdx.x")
sch.bind(loop_ti, "threadIdx.y")
sch.bind(loop_tj, "threadIdx.x")

# Maps to CUDA thread hierarchy:
# blockIdx.x/y/z   -- block index within grid
# threadIdx.x/y/z  -- thread index within block
# blockDim.x/y/z   -- block dimensions
# gridDim.x/y/z    -- grid dimensions
```

#### Shared Memory in CUDA

```python
# In TensorIR schedule, set buffer scope to shared memory
sch = tir.Schedule(mod)

# Move a buffer to shared memory
block = sch.get_block("compute")
sch.set_scope(block, 0, "shared")  # buffer index 0 -> shared memory

# Set storage alignment to avoid bank conflicts
sch.storage_align(block, 0, axis=0, factor=16, offset=0)

# The generated CUDA C will include:
# __shared__ float shared_buf[TILE_SIZE][TILE_SIZE + 1];
# The +1 padding avoids bank conflicts in shared memory
```

#### CUDA-Specific Lowering Passes

```python
from tvm import tir

# CUDA-specific TIR lowering passes:
# 1. LowerWarpMemory: Converts warp-level buffer accesses to shuffle ops
# 2. LowerDeviceStorageAccess: Handles shared/global memory access patterns
# 3. MergesDynamicSharedMemoryAllocations: Combines multiple __shared__
#    allocations into a single dynamic shared memory region
# 4. LowerAsyncDMA: Lowers async memory copy operations (cp.async for Ampere+)
```

### 29.3.2 OpenCL Backend

#### Overview

The OpenCL backend generates OpenCL C kernel source code, providing cross-platform GPU support.

#### Target Configuration

```python
# Generic OpenCL
target_opencl = tvm.target.Target("opencl")

# Intel GPU OpenCL
target_intel = tvm.target.Target("opencl -device=intel_gpu")

# Mali GPU OpenCL
target_mali = tvm.target.Target("opencl -device=mali")
```

#### OpenCL Code Generation

```python
# Generates OpenCL C kernel source:
# __kernel void matmul_kernel(
#     __global float* __restrict__ A,
#     __global float* __restrict__ B,
#     __global float* __restrict__ C
# ) {
#     int vi = get_global_id(1);
#     int vj = get_global_id(0);
#     ...
# }

# Key OpenCL builtins used:
# get_global_id(dim)    -- global thread index
# get_local_id(dim)     -- local thread index within workgroup
# get_global_size(dim)  -- total global size
# get_local_size(dim)   -- workgroup size
# __local               -- shared/local memory
# barrier(CLK_LOCAL_MEM_FENCE) -- synchronization
```

### 29.3.3 Metal Backend

#### Overview

The Metal backend generates Metal Shading Language (MSL) compute shaders for Apple platforms.

#### Target Configuration

```python
# Apple Metal
target_metal = tvm.target.Target("metal")
```

#### Metal Code Generation

```python
# Generates Metal Shading Language:
# kernel void matmul_kernel(
#     device float* A [[buffer(0)]],
#     device float* B [[buffer(1)]],
#     device float* C [[buffer(2)]],
#     uint2 gid [[thread_position_in_grid]]
# ) {
#     int vi = gid.y;
#     int vj = gid.x;
#     ...
# }

# Metal-specific features:
# - threadgroup memory (shared memory equivalent)
# - simdgroup matrix operations (SIMD matrix math for Apple Silicon)
# - thread_position_in_grid, thread_position_in_threadgroup
# - threadgroups_per_grid, threads_per_threadgroup
```

### 29.3.4 Vulkan Backend

#### Overview

The Vulkan backend generates SPIR-V compute shaders for cross-vendor GPU support.

#### Target Configuration

```python
# Generic Vulkan
target_vulkan = tvm.target.Target("vulkan")

# Vulkan with specific extensions
target_vulkan_ext = tvm.target.Target("vulkan -supports_storage_buffer_storage_class=1")
```

#### Vulkan Code Generation

```python
# Generates SPIR-V binary:
# 1. TIR -> SPIR-V IR translation
# 2. SPIR-V optimization passes
# 3. SPIR-V binary output

# Key SPIR-V features:
# - Compute shaders (OpComputeBinary)
# - Storage buffers (SSBO) for tensor data
# - Workgroup shared memory
# - Subgroup operations (on supported hardware)
# - Cooperative matrix operations (NVIDIA, Intel)
```

### 29.3.5 WebGPU Backend

#### Overview

The WebGPU backend generates WGSL (WebGPU Shading Language) compute shaders for web browser GPU access.

#### Target Configuration

```python
target_webgpu = tvm.target.Target("webgpu")
```

#### WebGPU Code Generation

```python
# Generates WGSL compute shader:
# @group(0) @binding(0) var<storage, read> A: array<f32>;
# @group(0) @binding(1) var<storage, read> B: array<f32>;
# @group(0) @binding(2) var<storage, read_write> C: array<f32>;
#
# @compute @workgroup_size(16, 16)
# fn matmul_kernel(
#     @builtin(global_invocation_id) gid: vec3<u32>
# ) {
#     let vi = gid.y;
#     let vj = gid.x;
#     ...
# }

# WebGPU features:
# - Storage buffers for tensor data
# - Workgroup shared memory
# - 32-bit float and integer types
# - Subgroup operations (experimental)
```

---

## 29.4 Hexagon Backend

### 29.4.1 Overview

The Hexagon backend generates code for Qualcomm's Hexagon DSP, targeting the HVX (Hexagon Vector eXtensions) SIMD units.

### 29.4.2 Target Configuration

```python
# Hexagon DSP
target_hexagon = tvm.target.Target("hexagon")

# Hexagon with specific architecture version
target_hexagon_v68 = tvm.target.Target("hexagon -mcpu=hexagonv68")
target_hexagon_v69 = tvm.target.Target("hexagon -mcpu=hexagonv69")
```

### 29.4.3 Hexagon Code Generation Features

```python
# HVX vector instructions:
# - 1024-bit vectors (128 bytes) per HVX instruction
# - 4 HVX threads can execute in parallel
# - Special instructions: vadd, vsub, vmul, vmpy, vlalign, valign

# Hexagon SDK integration:
# 1. TVM generates LLVM IR targeting Hexagon
# 2. Hexagon LLVM compiler produces object files
# 3. Linked with Hexagon SDK runtime libraries
# 4. Loaded on DSP via Hexagon remote procedure calls

# Memory types on Hexagon:
# - TCM (Tightly Coupled Memory): Fast, limited size
# - DDR: Larger, slower
# - HVX can access both TCM and DDR
```

### 29.4.4 Hexagon-Specific TIR Intrinsics

```python
# TVM provides Hexagon-specific intrinsics:
# - hexagon.vadd
# - hexagon.vmpy (vector multiply)
# - hexagon.vrmpy (vector reduce multiply)
# - hexagon.vlut (vector lookup table)
# These are lowered to HVX instructions during code generation
```

---

## 29.5 TIR Lowering Pipeline

### 29.5.1 Lowering Overview

Before code generation, TIR must be lowered from high-level tensor operations to low-level operations that map directly to the target hardware.

### 29.5.2 Standard Lowering Sequence

```python
from tvm import tir

# The standard lowering sequence for CPU targets:
def lower_for_cpu(mod, target):
    """Lower TIR for CPU code generation."""
    # Phase 0: High-level transformations
    mod = tir.transform.FlattenBuffer()(mod)       # Multi-dim -> 1D buffers
    mod = tir.transform.LowerCrossThreadReduction()(mod)
    mod = tir.transform.LowerInitBlock()(mod)

    # Phase 1: Lower control flow
    mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tir.transform.ConvertBlocksToOpaqueBlocks()(mod)
    mod = tir.transform.LowerMatchBufferRegion()(mod)
    mod = tir.transform.CompactBufferAllocation()(mod)

    # Phase 2: Lower arithmetic
    mod = tir.transform.LowerOpaqueBlock()(mod)
    mod = tir.transform.FlattenBuffer()(mod)        # Ensure flat buffers
    mod = tir.transform.NarrowDataType(32)(mod)     # Narrow index types
    mod = tir.transform.Simplify()(mod)             # Simplify expressions
    mod = tir.transform.VectorizeLoop()(mod)        # Vectorize where possible
    mod = tir.transform.InjectVirtualThread()(mod)
    mod = tir.transform.LimitConstIntRange()(mod)

    # Phase 3: Final cleanup
    mod = tir.transform.StorageRewrite()(mod)
    mod = tir.transform.UnifyThreadBinding()(mod)
    mod = tir.transform.LowerThreadAllreduce()(mod)
    mod = tir.transform.LowerIntrin()(mod)          # Target intrinsics

    return mod
```

### 29.5.3 GPU Lowering Sequence

```python
def lower_for_gpu(mod, target):
    """Lower TIR for GPU code generation."""
    # GPU-specific lowering passes
    mod = tir.transform.FlattenBuffer()(mod)
    mod = tir.transform.LowerWarpMemory()(mod)          # Warp shuffles
    mod = tir.transform.LowerDeviceStorageAccess()(mod)  # Shared/global memory
    mod = tir.transform.LowerThreadAllreduce()(mod)      # Cross-thread reduction
    mod = tir.transform.UnifyThreadBinding()(mod)        # Canonicalize bindings
    mod = tir.transform.LowerIntrin()(mod)               # GPU intrinsics

    # GPU-specific: merge dynamic shared memory allocations
    mod = tir.transform.MergesDynamicSharedMemoryAllocations()(mod)

    # For Ampere+: lower async memory copies
    # mod = tir.transform.LowerAsyncDMA()(mod)

    return mod
```

### 29.5.4 FlattenBuffer Pass

The FlattenBuffer pass transforms multi-dimensional buffer accesses into flat 1D accesses:

```python
# Before FlattenBuffer:
@T.prim_func
def example(A: T.Buffer((128, 64), "float32"),
            B: T.Buffer((128, 64), "float32")):
    for i, j in T.grid(128, 64):
        with T.sblock("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0

# After FlattenBuffer:
@T.prim_func
def example(A: T.Buffer((128 * 64,), "float32"),
            B: T.Buffer((128 * 64,), "float32")):
    for i, j in T.grid(128, 64):
        with T.sblock("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            # Multi-dim index flattened: vi * 64 + vj
            B[vi * 64 + vj] = A[vi * 64 + vj] * T.float32(2.0)
```

### 29.5.5 LowerIntrin Pass

The LowerIntrin pass replaces TIR intrinsic calls with target-specific implementations:

```python
# TIR intrinsics that get lowered:
# - tir.exp()       -> target-specific exp implementation
# - tir.log()       -> target-specific log implementation
# - tir.sqrt()      -> target-specific sqrt implementation
# - tir.pow()       -> target-specific pow implementation
# - tir.floor()     -> target-specific floor implementation
# - tir.ceil()      -> target-specific ceil implementation
# - tir.fma()       -> fused multiply-add
# - tir.clz()       -> count leading zeros

# For CUDA target:
# tir.exp(x)  -> __expf(x)  (fast math exp for float32)
# tir.log(x)  -> __logf(x)  (fast math log for float32)
# tir.sqrt(x) -> sqrtf(x)

# For LLVM target:
# tir.exp(x)  -> LLVM intrinsic: llvm.exp.f32
# tir.sqrt(x) -> LLVM intrinsic: llvm.sqrt.f32
```

### 29.5.6 VectorizeLoop Pass

```python
# Before VectorizeLoop:
for i in range(0, 128):
    C[i] = A[i] + B[i]

# After VectorizeLoop (with vector width 4):
for i in range(0, 128, 4):
    C[i:i+4] = A[i:i+4] + B[i:i+4]  # Vector load + add + store

# Vector width is determined by:
# - Target vector register width (AVX-512 = 512 bits = 16 x float32)
# - Data type size (float32 = 4 bytes)
# - Alignment requirements
```

---

## 29.6 Host-Device Code Split

### 29.6.1 Overview

For GPU targets, code generation produces two types of code:
- **Host code**: Runs on the CPU, manages memory, launches kernels, and handles control flow.
- **Device code**: Runs on the GPU, implements the compute kernels.

### 29.6.2 Module Composition

```python
import tvm
from tvm.target import Target

# Build with separate host and device targets
host_target = Target("llvm")
device_target = Target("cuda")

# TVM generates:
# 1. Host module (LLVM) -- contains:
#    - Main entry function (PackedFunc wrapper)
#    - Memory allocation calls
#    - Kernel launch calls
#    - Data transfer calls
#
# 2. Device module (CUDA) -- contains:
#    - GPU kernel functions
#    - Device-side utility functions

# The host module imports the device module
mod = tvm.build(tir_mod, target=device_target)
# mod is an LLVMModule with an imported CUDAModule
```

### 29.6.3 Host Code Structure

```python
# Generated host code (conceptual):
def main_packed(args, type_codes, num_args, ret_val, ret_type_code):
    # 1. Extract input tensors from packed args
    A = args[0]  # DLTensor*
    B = args[1]  # DLTensor*
    C = args[2]  # DLTensor* (output)

    # 2. Set up kernel launch parameters
    grid_dim = (8, 8, 1)
    block_dim = (16, 16, 1)
    stream = get_current_stream(device)

    # 3. Launch GPU kernel
    cuLaunchKernel(
        kernel_function,
        grid_dim[0], grid_dim[1], grid_dim[2],
        block_dim[0], block_dim[1], block_dim[2],
        shared_mem_bytes=0,
        stream=stream,
        kernel_args=[A.data, B.data, C.data],
    )

    # 4. Return
    ret_val[0] = C
```

### 29.6.4 Multiple Device Modules

```python
# When using multiple GPU targets:
# Host module
#   ├── CUDA module (NVIDIA GPU kernels)
#   ├── OpenCL module (Intel GPU kernels)
#   └── External module (cuBLAS, cuDNN calls)

# The host module manages all imported device modules
# and dispatches to the appropriate one based on the target device
```

---

## 29.7 Intrinsic Lowering

### 29.7.1 Math Function Lowering

TVM provides configurable math function lowering for different targets:

```python
# Fast math vs. precise math
# For GPU targets, TVM typically uses fast math implementations:
# exp()  -> __expf()   (CUDA fast math, ~22 cycles)
# log()  -> __logf()   (CUDA fast math)
# sin()  -> __sinf()   (CUDA fast math)
# cos()  -> __cosf()   (CUDA fast math)

# For CPU targets, TVM uses standard math functions:
# exp()  -> expf()     (standard math library)
# These are then optimized by LLVM (may use SVML on Intel)

# Control math function precision:
with tvm.transform.PassContext(config={
    "tir.disable_cse_tir": False,  # Enable common subexpression elimination
}):
    rt_mod = tvm.build(mod, target="llvm")
```

### 29.7.2 Atomic Operations

Atomic operations are lowered to target-specific implementations:

```python
# CUDA atomic operations:
# TIR: T.buffer_atomic_add(buffer, index, value)
# CUDA: atomicAdd(&buffer[index], value)

# Supported CUDA atomics:
# atomicAdd, atomicSub, atomicExch, atomicMin, atomicMax
# atomicAnd, atomicOr, atomicXor, atomicCAS (compare-and-swap)

# For reductions across thread blocks:
# 1. Each block computes partial reduction (in shared memory)
# 2. Thread 0 of each block atomically adds to global result
```

### 29.7.3 Warp-Level Operations

```python
# Warp shuffle operations (CUDA):
# TIR intrinsic -> CUDA instruction
# tir.ptx_shfl_sync(MASK, VAL, SRC_LANE) -> __shfl_sync(MASK, VAL, SRC_LANE)
# tir.ptx_shfl_up_sync(MASK, VAL, DELTA) -> __shfl_up_sync(MASK, VAL, DELTA)
# tir.ptx_shfl_down_sync(MASK, VAL, DELTA) -> __shfl_down_sync(MASK, VAL, DELTA)

# Warp reduction:
# All threads in a warp cooperate to reduce a value
# Uses warp shuffles to exchange data between lanes
```

### 29.7.4 Tensor Core Intrinsics

```python
# Tensor Core operations (NVIDIA):
# TIR intrinsic -> CUDA PTX instruction
# tir.nvidia_mma_sync_884_f16f16f16 -> MMA.884.F16.F16 instruction
# tir.nvidia_mma_sync_1688_f16f16f16 -> MMA.1688.F16.F16 instruction (Ampere)
# tir.nvidia_mma_sync_16816_f16f16f16 -> MMA.16816.F16.F16 instruction (Hopper)

# Tensor Core usage via tensorize:
sch = tir.Schedule(mod)
block = sch.get_block("compute")
sch.tensorize(block, "nvidia_mma_sync_16816_f16f16f16")
# This replaces the compute block with a Tensor Core MMA instruction
```

---

## 29.8 GPU Thread Binding

### 29.8.1 Thread Hierarchy Mapping

TVM maps TIR loop bindings to GPU thread hierarchies:

```python
# TIR schedule with thread binding:
sch = tir.Schedule(mod)
block = sch.get_block("compute")

# Get loops
i, j, k = sch.get_loops(block)

# Tile for CUDA thread hierarchy
# Grid level: blockIdx
# Block level: threadIdx
# Serial loops remain as regular for-loops

# Example tiling:
i0, i1 = sch.split(i, factors=[None, 16])  # i0 -> blockIdx, i1 -> threadIdx
j0, j1 = sch.split(j, factors=[None, 16])  # j0 -> blockIdx, j1 -> threadIdx

sch.bind(i0, "blockIdx.y")
sch.bind(j0, "blockIdx.x")
sch.bind(i1, "threadIdx.y")
sch.bind(j1, "threadIdx.x")
# k remains a serial loop (reduction dimension)
```

### 29.8.2 Shared Memory Management

```python
# Shared memory is used for data reuse within a thread block
sch = tir.Schedule(mod)

# Move a buffer to shared memory
block_read_a = sch.get_block("read_A")
sch.set_scope(block_read_a, 0, "shared")

block_read_b = sch.get_block("read_B")
sch.set_scope(block_read_b, 0, "shared")

# Set storage alignment for bank conflict avoidance
# Padding: factor=16, offset=8 means each row has 16+8=24 elements
# instead of 16, spreading accesses across different banks
sch.storage_align(block_read_a, 0, axis=0, factor=16, offset=8)

# Insert synchronization points
# After reading into shared memory:
sch.syncthreads()  # __syncthreads() in CUDA
```

### 29.8.3 Register-Level Optimizations

```python
# Promote buffers to registers for thread-local data
sch = tir.Schedule(mod)

# Promote a buffer to local (register) scope
block = sch.get_block("compute")
sch.set_scope(block, 0, "local")  # Buffer allocated in registers

# Local scope means:
# - Each thread has its own copy
# - Fastest access (register file)
# - Limited by register count (255 per thread on modern GPUs)
```

### 29.8.4 Warp-Level Matrix Operations

```python
# Use warp-level matrix multiply-accumulate (MMA) operations
# Available on Volta+ (SM70+) architectures

# WMMA (Tensor Core through CUDA C):
# nvcuda::wmma::load_matrix_sync()
# nvcuda::wmma::fill_fragment()
# nvcuda::wmma::mma_sync()
# nvcuda::wmma::store_matrix_sync()

# In TVM, this is accessed via tensorize:
sch = tir.Schedule(mod)
mma_block = sch.get_block("matmul")
sch.tensorize(mma_block, "nvidia_mma_sync_16816_f16f16f16")
```

---

## 29.9 Memory Management in Code Generation

### 29.9.1 Buffer Allocation

```python
# TIR buffer types and their mapping to target memory:

# Global memory (default):
#   - CPU: heap-allocated (malloc)
#   - CUDA: global memory (cudaMalloc)
#   - OpenCL: global memory (CL_MEM_READ_WRITE)

# Shared memory:
#   - CPU: stack-allocated or L1 cache
#   - CUDA: __shared__ memory (48 KB default, configurable)
#   - OpenCL: __local memory

# Local/Register memory:
#   - CPU: registers (compiler-managed)
#   - CUDA: local memory (spills to global if register pressure is high)
#   - OpenCL: __private memory

# In TIR, scope is set via sch.set_scope():
# "global"   -> device global memory
# "shared"   -> shared memory (CUDA) / local memory (OpenCL)
# "local"    -> registers / thread-local
# "warp"     -> warp-level memory (lowered to shuffle ops)
```

### 29.9.2 Dynamic Shared Memory

```python
# CUDA dynamic shared memory allocation
# TVM generates a single dynamic shared memory allocation
# and partitions it among all shared buffers

# Before merging (multiple static shared allocations):
# __shared__ float buf_a[16][16];
# __shared__ float buf_b[16][16];

# After MergesDynamicSharedMemoryAllocations:
# extern __shared__ float dynamic_smem[];
# float* buf_a = dynamic_smem;
# float* buf_b = dynamic_smem + 256;
```

### 29.9.3 Storage Alignment

```python
# Storage alignment helps avoid bank conflicts in shared memory
# CUDA shared memory has 32 banks, each 4 bytes wide

# Without alignment: concurrent accesses to same bank cause conflicts
# buf[i][j] where i is threadIdx.x -> bank = (j * 4 + i) % 32

# With storage_align: pad rows to avoid bank conflicts
sch.storage_align(block, buffer_idx, axis=0, factor=32, offset=1)
# Adds 1 element of padding per row, shifting bank assignments
```

---

## 29.10 Multi-Target Compilation

### 29.10.1 Heterogeneous Target Support

```python
import tvm
from tvm.target import Target

# CPU + GPU compilation
# The host runs on CPU (LLVM), kernels run on GPU (CUDA)
host_target = Target("llvm")
device_target = Target("cuda")

# Build for GPU target (host target is implicit from device target)
exec_mod = tvm.build(mod, target=device_target)

# Explicitly specify both targets
target = Target("cuda", host="llvm")
exec_mod = tvm.build(mod, target=target)
```

### 29.10.2 Target Detection and Selection

```python
import tvm

# Detect available targets
if tvm.cuda().exist:
    target = tvm.target.Target("nvidia/nvidia-a100")
elif tvm.rocm().exist:
    target = tvm.target.Target("rocm")
elif tvm.metal().exist:
    target = tvm.target.Target("metal")
elif tvm.opencl().exist:
    target = tvm.target.Target("opencl")
else:
    target = tvm.target.Target("llvm")

print(f"Using target: {target}")
```

### 29.10.3 Cross-Compilation

```python
# Compile on x86 for ARM target
arm_target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu")
rt_mod = tvm.build(mod, target=arm_target)

# Compile on x86 for Hexagon DSP
hexagon_target = tvm.target.Target("hexagon")
rt_mod = tvm.build(mod, target=hexagon_target)

# Export the compiled module for deployment
rt_mod.export_library("model_for_arm.so")
```

---

## 29.11 Code Generation for Relax

### 29.11.1 Relax Build Process

```python
import tvm
from tvm import relax

# The relax.build() function orchestrates the full compilation:
# 1. Apply Relax-level optimization passes
# 2. Legalize Relax ops to TIR PrimFunc
# 3. Apply TIR-level optimization passes
# 4. Apply MetaSchedule / DLight scheduling
# 5. Lower TIR to target representation
# 6. Generate code for each target
# 7. Compose runtime.Module

exec_mod = relax.build(
    mod,
    target="nvidia/nvidia-a100",
)
```

### 29.11.2 Relax VM Code Generation

```python
from tvm import relax

# Relax generates bytecode for the Relax Virtual Machine
# The VM executable contains:
# 1. Bytecode instructions (Call, Ret, Goto, If, etc.)
# 2. Constant pool (weights, biases)
# 3. Function table (PackedFunc references)
# 4. Compiled TIR kernels

# Build to VM executable
exec_mod = relax.build(mod, target="nvidia/nvidia-a100")

# Save as Relax Object (.ro) format
exec_mod.export_library("model.ro")

# Load and run
vm = relax.VirtualMachine(
    tvm.runtime.load_module("model.ro"),
    tvm.cuda(0),
)
result = vm["main"](input_tensor)
```

### 29.11.3 Relax Pipeline Customization

```python
from tvm import relax

# Use a pre-defined pipeline
exec_mod = relax.build(
    mod,
    target="llvm",
    pipeline=relax.get_pipeline("zero"),
)

# Custom pipeline with specific passes
def custom_pipeline(mod, target):
    mod = relax.transform.FuseOps()(mod)
    mod = relax.transform.FuseOpsByPattern(patterns)(mod)
    mod = relax.transform.LegalizeOps()(mod)
    mod = relax.transform.FuseTIR()(mod)
    mod = relax.transform.DeadCodeElimination()(mod)
    return mod

exec_mod = relax.build(mod, target="llvm", pipeline=custom_pipeline)
```

---

## 29.12 Code Generation Debugging

### 29.12.1 Inspecting Generated Code

```python
import tvm

# Method 1: Get the source code from the module
rt_mod = tvm.build(mod, target="cuda")
source = rt_mod.imported_modules[0].get_source()
print(source)  # CUDA C source code

# Method 2: Get LLVM IR
rt_mod = tvm.build(mod, target="llvm")
llvm_ir = rt_mod.get_source()
print(llvm_ir)  # LLVM IR

# Method 3: Get PTX for CUDA
rt_mod = tvm.build(mod, target="cuda")
ptx = rt_mod.imported_modules[0].get_source(fmt="ptx")
print(ptx)  # PTX assembly
```

### 29.12.2 Lowering Debugging

```python
# Inspect TIR at each lowering stage
from tvm import tir

# Apply lowering passes one at a time
mod_lowered = tir.transform.FlattenBuffer()(mod)
print("After FlattenBuffer:")
print(mod_lowered.script())

mod_lowered = tir.transform.LowerIntrin()(mod_lowered)
print("After LowerIntrin:")
print(mod_lowered.script())

mod_lowered = tir.transform.VectorizeLoop()(mod_lowered)
print("After VectorizeLoop:")
print(mod_lowered.script())
```

### 29.12.3 Verbose Logging

```python
import os

# Enable verbose TVM logging
os.environ["TVM_LOG_DEBUG"] = "1"

# Enable specific debug output
os.environ["TVM_LLVM_DEBUG"] = "1"        # LLVM debug output
os.environ["TVM_CUDA_DEBUG"] = "1"         # CUDA compilation debug

# Use TVM_LOG_DEBUG for targeted debugging
os.environ["TVM_LOG_DEBUG"] = "vta.codegen=2,tir.lower=2"
```

### 29.12.4 Performance Debugging

```python
import tvm
from tvm.runtime import profiling

# Profile the compiled module
dev = tvm.cuda(0)
vm = relax.VirtualMachine(exec_mod, dev)

# Time individual operations
report = vm.profile(input_tensor)
print(report)

# Expected output:
# Name                  Calls   Total Time (ms)  ...
# fused_matmul_add_relu  100      12.5           ...
# fused_conv2d_bn_relu   100      45.3           ...
# fused_reduce_mean      100       3.2           ...
# fused_dense            100       8.1           ...
```

### 29.12.5 Comparing Generated Code Across Targets

```python
import tvm

# Build for multiple targets and compare
targets = ["llvm", "cuda", "opencl"]
sources = {}

for target_str in targets:
    try:
        rt_mod = tvm.build(mod, target=target_str)
        if rt_mod.imported_modules:
            sources[target_str] = rt_mod.imported_modules[0].get_source()
        else:
            sources[target_str] = rt_mod.get_source()
    except Exception as e:
        sources[target_str] = f"Error: {e}"

# Compare code generation output
for target, source in sources.items():
    print(f"\n=== {target} ===")
    print(source[:500])  # First 500 chars
```

---

## 29.13 Register Allocation and Optimization

### 29.13.1 Register Pressure Management

TVM does not perform explicit register allocation -- this is left to downstream compilers (LLVM, nvcc). However, TVM's scheduling decisions directly impact register pressure:

```python
# High register pressure (many live values):
# - Large tile sizes -> more values in flight
# - Unrolled loops -> many concurrent computations
# - Complex fused kernels -> many intermediate results

# Low register pressure:
# - Small tile sizes -> fewer live values
# - Loops kept as loops -> register reuse
# - Separate kernels -> fewer concurrent values

# On GPU, exceeding register limit causes spills to local memory:
# - 255 registers per thread (modern NVIDIA GPUs)
# - Spills to local memory -> global memory accesses -> slow
# - TVM provides control via max_threads_per_block scheduling hint
```

### 29.13.2 Loop Transformations for Code Generation

```python
# Tiling: breaks large loops into smaller tiles
sch = tir.Schedule(mod)
i, j = sch.get_loops(block)
i_outer, i_inner = sch.split(i, factors=[None, 32])
j_outer, j_inner = sch.split(j, factors=[None, 32])

# Unrolling: replicates loop body for small trip counts
sch.unroll(i_inner)

# Fusion: merges loops with the same iteration space
# (done at TIR level via FuseTIR)

# Reordering: changes loop nesting order for better locality
sch.reorder(i_outer, j_outer, i_inner, j_inner)

# These transformations affect:
# - Cache behavior (temporal and spatial locality)
# - Vectorization opportunities
# - Thread utilization on GPUs
```

---

## 29.14 Async and Pipeline Code Generation

### 29.14.1 Asynchronous Memory Operations

```python
# For Ampere+ (SM80+), TVM can generate async memory copies
# using cp.async instructions:

# TIR with async copy:
# T.copy(src, dst, scope="shared")
# Lowered to:
# cp.async.ca.shared.global [dst], [src], 16;

# Pipeline of async copies:
# TIR async pipeline:
# Stage 1: async copy A[0] to shared
# Stage 2: compute with A[0], async copy A[1] to shared
# Stage 3: compute with A[1], async copy A[2] to shared
# ...overlapping compute and memory access

# In TIR schedule:
sch = tir.Schedule(mod)
# Mark memory copies as async
sch.annotate(block, "async_copy", True)
```

### 29.14.2 Software Pipelining

```python
# Software pipelining overlaps prologue, steady-state, and epilogue
# to hide memory latency with computation

# TIR software pipeline annotation:
sch = tir.Schedule(mod)
loop = sch.get_loops(block)[2]  # The pipelineable loop
sch.annotate(loop, "software_pipeline_stage", [0, 1, 2])
sch.annotate(loop, "software_pipeline_order", [0, 1, 2])
sch.annotate(loop, "software_pipeline_async_stages", [0])

# Generated code structure:
# Prologue:    load data for iteration 0
# Steady-state: compute iteration i, load data for iteration i+1
# Epilogue:    compute last iteration
```

---

## 29.15 Specialized Code Generation Features

### 29.15.1 Bfloat16 Support

```python
# Bfloat16 (Brain Float) support
# 16-bit floating point with same exponent range as FP32
# Supported on Intel AVX-512 BF16, NVIDIA Ampere+

target_bf16 = tvm.target.Target("llvm -mcpu=cooperlake")
# Uses AVX-512 BF16 instructions: vdpbf16ps

target_cuda_bf16 = tvm.target.Target("nvidia/nvidia-a100")
# Uses CUDA BF16 types: __nv_bfloat16, __nv_bfloat162
```

### 29.15.2 INT8 Quantization Support

```python
# INT8 code generation for inference
# DP4A instruction: 4-element dot product in a single instruction
# Available on NVIDIA Pascal+ (SM60+)

# TIR intrinsic for INT8 dot product:
# tir.call_packed("tir.ptx_dp4a", x, y)
# Generates: dp4a.rn.s32.u8.u8 %dst, %src1, %src2

# INT8 matmul on CUDA:
# x_q = quantize(x, scale)  # int8
# w_q = quantize(w, scale)  # int8
# result = dequantize(dp4a(x_q, w_q), scale)
```

### 29.15.3 FP8 Support

```python
# FP8 (8-bit floating point) support
# E4M3 (4 exponent, 3 mantissa bits) - for forward pass
# E5M2 (5 exponent, 2 mantissa bits) - for backward pass
# Supported on NVIDIA Hopper (SM90)

target_fp8 = tvm.target.Target("nvidia/nvidia-h100")
# Uses FP8 tensor core instructions
```

### 29.15.4 Sparse Code Generation

```python
# TVM can generate code optimized for sparse operations
# - Block-sparse matrix multiplication
# - Sparse convolution
# - Compressed sparse row/column (CSR/CSC) storage formats

# Sparse scheduling:
sch = tir.Schedule(mod)
block = sch.get_block("sparse_matmul")
# Annotate with sparsity information
sch.annotate(block, "sparse_format", "BSR")
sch.annotate(block, "block_size", [32, 32])
```

---

## 29.16 Build System Integration

### 29.16.1 Export and Save

```python
import tvm

# Export compiled module
rt_mod = tvm.build(mod, target="llvm")

# Save as shared library
rt_mod.export_library("model.so")

# Save as object file (for linking)
rt_mod.save("model.o")

# Save with specific format
rt_mod.export_library("model.tar")
```

### 29.16.2 System Library Mode

```python
# Compile functions into the process address space
# Useful for embedded systems and microcontrollers

rt_mod = tvm.build(mod, target="llvm")
rt_mod.export_library("model.o", system_lib=True)

# In the deployment application:
# #include <tvm/runtime/c_runtime_api.h>
# TVM_DLL_EXPORT_TYPED_FUNC(my_model_func, compiled_function);
# The function is linked at compile time, no dynamic loading needed
```

### 29.16.3 Runtime Module Types

| Target | Module Type | File Format | Loader |
|--------|-------------|-------------|--------|
| LLVM | LLVMModule | .so, .o, .dll | dlopen / system linker |
| CUDA | CUDAModule | .ptx, .cubin | cuModuleLoadData |
| OpenCL | OpenCLModule | .cl binary | clCreateProgramWithBinary |
| Metal | MetalModule | .metal binary | MTLDevice.newLibraryWithSource |
| Vulkan | VulkanModule | SPIR-V binary | vkCreateShaderModule |
| WebGPU | WebGPUModule | WGSL source | GPUDevice.createShaderModule |
| Relax VM | VMExecutable | .ro (flatbuffer) | relax.VirtualMachine |

### 29.16.4 Build Configuration

```python
# Build with custom configuration
import tvm

with tvm.transform.PassContext(
    opt_level=3,
    config={
        "tir.disable_assert": True,        # Disable bounds checking
        "tir.disable_vectorize": False,     # Enable vectorization
        "tir.instrument_bound_checkers": False,
        "relay.backend.use_meta_schedule": True,
        "relax.backend.use_cublas": True,
        "relax.backend.use_cutlass": True,
    },
):
    exec_mod = tvm.build(mod, target="nvidia/nvidia-a100")
```

---

## 29.17 Source Code Locations

| Component | Path |
|-----------|------|
| LLVM codegen | `src/target/source/codegen_llvm.cc` |
| CUDA codegen | `src/target/source/codegen_cuda.cc` |
| OpenCL codegen | `src/target/source/codegen_opencl.cc` |
| Metal codegen | `src/target/source/codegen_metal.cc` |
| Vulkan codegen | `src/target/source/codegen_vulkan.cc` |
| WebGPU codegen | `src/target/source/codegen_webgpu.cc` |
| Hexagon codegen | `src/target/source/codegen_hexagon.cc` |
| SPIR-V codegen | `src/target/spirv/` |
| TIR lowering passes | `src/tir/transforms/` |
| FlattenBuffer | `src/tir/transforms/flatten_buffer.cc` |
| LowerIntrin | `src/tir/transforms/lower_intrin.cc` |
| VectorizeLoop | `src/tir/transforms/vectorize_loop.cc` |
| MakePackedAPI | `src/tir/transforms/make_packed_api.cc` |
| StorageRewrite | `src/tir/transforms/storage_rewrite.cc` |
| Build API | `src/driver/driver_api.cc` |
| Relax build | `src/relax/backend/build.cc` |
| Target registry | `src/target/target_kind.cc` |
| CUDA runtime module | `src/runtime/cuda/` |
| Relax VM | `src/runtime/vm/` |
| Python build bindings | `python/tvm/driver/build_module.py` |
| Python target API | `python/tvm/target/` |

---

## 29.18 Summary

Code generation is the final stage of TVM's compilation pipeline, translating optimized TIR into executable code for diverse hardware targets. Key aspects include:

| Aspect | Description |
|--------|-------------|
| LLVM Backend | Primary CPU codegen path; supports x86, ARM, RISC-V |
| Source Code Backends | Generate CUDA C, OpenCL C, Metal, SPIR-V, WGSL for GPU targets |
| TIR Lowering | FlattenBuffer, LowerIntrin, VectorizeLoop prepare TIR for codegen |
| Host-Device Split | Separate host (orchestration) and device (compute) modules |
| Thread Binding | Maps TIR loops to hardware threading (CUDA threads, CPU SIMD) |
| Intrinsic Lowering | Target-specific implementations for math, atomic, warp operations |
| Memory Management | Global, shared, local, register scope mapping |

The code generation system is designed to be extensible: new targets can be added by implementing the source code generation interface and registering target-specific lowering passes. TVM deliberately defers low-level optimizations (register allocation, instruction scheduling) to downstream compilers (LLVM, nvcc) while focusing on high-level transformations that these compilers cannot perform.
