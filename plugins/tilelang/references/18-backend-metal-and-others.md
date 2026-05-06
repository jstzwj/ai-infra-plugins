# TileLang Metal, CPU, WebGPU, and Other Backends Reference

## 1. Overview

TileLang supports multiple backends beyond CUDA and ROCm, including Metal for Apple
GPUs, a CPU backend with scalar GEMM, and a WebGPU backend with WGSL code generation.
These backends share the same TileLang frontend DSL but produce different target code.
This document covers each backend in detail, along with backend selection logic,
cross-backend portability, and custom backend registration.

---

## 2. Metal Backend

### 2.1 Overview

The Metal backend targets Apple GPUs (Apple Silicon M1/M2/M3/M4 and AMD GPUs in
older Macs). It uses the Metal Shading Language (MSL) for code generation and
leverages the Apple GPU's SIMT execution model.

### 2.2 Architecture

```
TileLang IR (PrimFunc)
        |
        v
+-------------------+
|  Backend Selection |  ---> Target = "metal"
+-------------------+
        |
        v
+-------------------+
|  Transform Passes |  (lower_tile_op, ...)
+-------------------+
        |
        v
+-------------------+
| BuildTileLangMetal |  (rt_mod_metal.cc)
|   -> BuildTileLangCHost with Metal context
+-------------------+
        |
        v
+-------------------+
| Metal Shading Lang |  (.metal)
+-------------------+
        |
        v
Apple GPU Execution
```

### 2.3 Code Generation

The Metal backend reuses the C host code generator with Metal-specific context:

```cpp
// From src/backend/metal/codegen/rt_mod_metal.cc
ffi::Module BuildTileLangMetal(IRModule mod, Target target) {
    return tl::BuildTileLangCHost(mod, target);
}

TVM_FFI_STATIC_INIT_BLOCK() {
    namespace refl = tvm::ffi::reflection;
    refl::GlobalDef().def("target.build.tilelang_metal", BuildTileLangMetal);
}
```

The `BuildTileLangCHost` function has built-in Metal context support via the
`is_in_metal_context` flag. When the IR contains `AttrStmt` with
`attr_key == "metal_context"`, the host codegen emits Metal-specific
`dispatch_sync` / `MTLCommandBuffer` code.

### 2.4 Metal Shading Language Output

The codegen produces Metal Shading Language (MSL) code, which is a C++-based
language with GPU-specific qualifiers:

```metal
// Metal kernel function
kernel void kernel_name(
    device half* A [[buffer(0)]],
    device half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]])
{
    // Kernel body
    uint idx = gid.x;
    // ...
}
```

### 2.5 Metal-Specific Features

#### Thread Groups

Metal uses thread groups instead of CUDA blocks:

| CUDA Concept | Metal Equivalent |
|---|---|
| Grid | Dispatch |
| Block | Thread Group |
| Thread | Thread |
| Shared memory | Thread Group Memory |
| Warp | SIMD Group (32 threads) |

#### Memory Hierarchy

| Scope | Metal | CUDA Equivalent |
|---|---|---|
| Global | `device` memory | Global memory |
| Shared | `threadgroup` memory | Shared memory |
| Local | Thread registers | Registers |
| Constant | `constant` memory | Constant memory |

#### SIMD Operations

Metal provides SIMD-group operations analogous to CUDA warp-level operations:

```metal
// SIMD shuffle
half val = simd_shuffle(value, lane_id);

// SIMD reduction
half sum = simd_sum(value);

// SIMD broadcast
half broadcast = simd_broadcast(value, 0);
```

### 2.6 Copy Operations

From `src/backend/metal/op/copy.cc`:

The Metal backend implements copy operations using the standard SIMT lowering:

```cpp
struct Copy {
    static LayoutMap InferLayout(const CopyNode &op, const LayoutInferArgs &T,
                                  InferLevel level) {
        return op.InferSIMTLayout(T, level);
    }

    static Stmt Lower(const CopyNode &op, const LowerArgs &T,
                       arith::Analyzer *analyzer) {
        return LowerNormalCopy(op, T, analyzer);
    }
};
```

Metal copies use simple thread-parallel memory access patterns without
Tensor Core-specific layout requirements.

### 2.7 Metal Backend Testing

Test files are located at:
- `testing/python/metal/test_metal_codegen.py`
- `testing/python/metal/test_metal_codegen_linux.py`

The Linux test file enables testing Metal codegen on non-Apple platforms by
verifying code generation correctness without actual execution.

### 2.8 Metal Target Configuration

```python
import tilelang as tl

# Metal target for Apple GPU
target = tl.target.Target("metal")
```

### 2.9 Current Limitations

1. **No Tensor Core / Matrix Core:** Apple GPUs do not expose Tensor Core-like
   instructions in the same way as NVIDIA/AMD. Matrix operations are performed
   using standard SIMD operations.

2. **SIMD Group Size:** Metal SIMD groups are typically 32 threads, similar to
   CUDA warps but with different execution semantics.

3. **No async copy:** Metal does not have an equivalent to `cp.async`. All
   memory copies are synchronous from the kernel's perspective.

4. **Thread Group Memory:** Limited to ~32KB per thread group (varies by GPU).

5. **Compilation:** Metal kernels are compiled at runtime by the Metal framework,
   not by NVRTC/hiprtc.

6. **SIMT only:** The Metal backend uses SIMT (Single Instruction Multiple Thread)
   style lowering without hardware-specific matrix instruction support.

---

## 3. CPU Backend

### 3.1 Overview

The CPU backend targets scalar execution on standard processors. It provides a
reference implementation that works on any x86 or ARM CPU.

### 3.2 Architecture

```
TileLang IR (PrimFunc)
        |
        v
+-------------------+
|  Backend Selection |  ---> Target = "c" / "llvm"
+-------------------+
        |
        v
+-------------------+
|  Transform Passes |  (lower_tile_op, ...)
+-------------------+
        |
        v
+-------------------+
| CodeGen C Host     |  (codegen_c_host.cc)
+-------------------+
        |
        v
+-------------------+
| C/C++ Code         |
+-------------------+
```

### 3.3 Code Generation

The CPU backend uses `CodeGenCHost` (defined in `src/target/codegen_c_host.cc`):

```cpp
ffi::Module BuildTileLangCHost(IRModule mod, Target target) {
    // Generate C/C++ host code
    // Can include Metal context when is_in_metal_context is true
}
```

### 3.4 Scalar GEMM Implementation

#### Python Registration

In `tilelang/backend/cpu/gemm.py`:

```python
from tilelang.backend.gemm import register_gemm_impl
from tilelang.tileop.gemm.gemm_scalar import GEMM_INST_SCALAR, GemmScalar

def _match_scalar(target) -> bool:
    return target.kind.name in {"c", "llvm"}

register_gemm_impl("cpu.scalar", GEMM_INST_SCALAR, _match_scalar, GemmScalar)
```

#### C++ Instruction Selection

In `src/backend/cpu/op/gemm.cc`:

```cpp
namespace cpu {

constexpr const char *kCPUScalar = "cpu.scalar";

struct Gemm {
    static String SelectInst(const GemmNode &op, int block_size, Target target) {
        return kCPUScalar;  // Always scalar on CPU
    }

    static std::pair<int, int> ComputeWarpPartition(
        const GemmWarpPolicyNode &policy, int M, int N,
        int block_size, Target target, String gemm_inst) {
        policy.m_warp = 1;
        policy.n_warp = 1;
        return {1, 1};  // Single "warp" on CPU
    }

    static bool ReuseExistingSharedLayout(String gemm_inst) {
        return false;
    }

    static String InstructionKind(String gemm_inst) {
        return "scalar";
    }
};

} // namespace cpu

// Registration
bool MatchCPUGemmTarget(Target target) { return TargetIsCPU(target); }

bool RegisterCPUGemm() {
    RegisterGemmImpl(GemmImpl{
        "cpu.Gemm",
        MatchCPUGemmTarget,
        cpu::Gemm::SelectInst,
        cpu::Gemm::ComputeWarpPartition,
        cpu::Gemm::ReuseExistingSharedLayout,
        cpu::Gemm::InstructionKind,
    });
    return true;
}
```

### 3.5 CPU GEMM Templates

From `src/tl_templates/cpu/gemm.h`:

The CPU GEMM template implements naive matrix multiplication:

```cpp
template <int M, int N, int K, typename A_type, typename B_type, typename C_type>
class GemmScalarOp {
public:
    static void body(A_type *A, B_type *B, C_type *C) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C_type sum = 0;
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
};
```

### 3.6 CPU Copy Operations

From `src/backend/cpu/op/copy.cc`:

CPU copy operations use simple memory copies without any hardware acceleration:

```cpp
struct Copy {
    static LayoutMap InferLayout(const CopyNode &op, const LayoutInferArgs &T,
                                  InferLevel level) {
        return op.InferSIMTLayout(T, level);
    }

    static Stmt Lower(const CopyNode &op, const LowerArgs &T,
                       arith::Analyzer *analyzer) {
        return LowerNormalCopy(op, T, analyzer);
    }
};
```

### 3.7 CPU Target Configuration

```python
import tilelang as tl

# CPU target (C backend)
target = tl.target.Target("c")

# CPU target (LLVM backend)
target = tl.target.Target("llvm")
```

### 3.8 CPU-Specific Code Generation

The CPU codegen produces standard C code:

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Generated kernel function
void kernel(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
```

### 3.9 CPU-Specific Transforms

When targeting CPU:

1. **No shared memory:** All buffers are in main memory
2. **No thread synchronization:** Single-threaded execution
3. **No Tensor Core:** Scalar multiply-accumulate
4. **No async operations:** Synchronous memory access
5. **Standard C loops:** TileLang loops map to for loops

### 3.10 Performance Expectations

The CPU backend is intended for:

- **Functional verification:** Testing kernel correctness without GPU
- **Prototyping:** Quick iteration during development
- **Embedded deployment:** Running on CPU-only systems
- **Reference implementation:** Ground truth for correctness checking

Performance is significantly lower than GPU backends. For high-performance CPU
execution, consider using a dedicated BLAS library instead.

---

## 4. WebGPU Backend

### 4.1 Overview

The WebGPU backend targets web browsers and WebGPU-compatible runtimes. It
generates WGSL (WebGPU Shading Language) code.

### 4.2 Architecture

```
TileLang IR (PrimFunc)
        |
        v
+-------------------+
|  Backend Selection |  ---> Target = "webgpu"
+-------------------+
        |
        v
+-------------------+
|  Transform Passes |  (lower_tile_op, ...)
+-------------------+
        |
        v
+-------------------+
|  WebGPU Copy/GEMM |  (SIMT lowering)
+-------------------+
        |
        v
+-------------------+
|  WGSL Codegen     |
+-------------------+
        |
        v
WebGPU Execution
```

### 4.3 Copy Operations

From `src/backend/webgpu/op/copy.cc`:

```cpp
namespace webgpu {

struct Copy {
    static LayoutMap InferLayout(const CopyNode &op, const LayoutInferArgs &T,
                                  InferLevel level) {
        return op.InferSIMTLayout(T, level);
    }

    static Stmt Lower(const CopyNode &op, const LowerArgs &T,
                       arith::Analyzer *analyzer) {
        return LowerNormalCopy(op, T, analyzer);
    }
};

} // namespace webgpu

// Registration
bool MatchWebGPUCopyTarget(Target target) {
    return target.defined() && target->kind.defined() &&
           target->kind->name == "webgpu";
}

bool RegisterWebGPUCopy() {
    RegisterCopyImpl(CopyImpl{
        "webgpu.Copy",
        MatchWebGPUCopyTarget,
        100,  // priority
        webgpu::Copy::InferLayout,
        webgpu::Copy::Lower,
    });
    return true;
}
```

Key observations:
- Uses SIMT layout inference (same as Metal and CPU)
- Uses normal copy lowering (no hardware-specific optimization)
- Priority is set to 100 (standard priority)

### 4.4 WGSL Code Generation

WebGPU shaders are written in WGSL (WebGPU Shading Language):

```wgsl
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&C)) {
        return;
    }
    C[idx] = A[idx] + B[idx];
}
```

### 4.5 WebGPU Execution Model

| Concept | WebGPU | CUDA Equivalent |
|---|---|---|
| Workgroup | Thread group | Block |
| Invocation | Thread | Thread |
| Dispatch | Command encoder | Kernel launch |
| Storage buffer | Global memory | Global memory |
| Workgroup memory | Shared memory | Shared memory |

### 4.6 WebGPU Limitations

1. **No Tensor Core:** WebGPU does not expose hardware matrix engines
2. **Limited shared memory:** Typically 16KB-32KB per workgroup
3. **No async operations:** Synchronous within workgroup
4. **Subset of data types:** f32, i32, u32, f16 (limited), no fp8/bf16
5. **Browser sandbox:** Limited compute time per dispatch
6. **No recursion or dynamic dispatch**

### 4.7 WebGPU Testing

Test files are located at:
- `testing/python/webgpu/` (if present)

### 4.8 WebGPU Target Configuration

```python
import tilelang as tl

# WebGPU target
target = tl.target.Target("webgpu")
```

---

## 5. Backend Selection Logic

### 5.1 Target Detection

The backend is selected based on the target device type:

```cpp
// From src/target/utils.cc
bool TargetIsCuda(target)   { return target->GetTargetDeviceType() == kDLCUDA; }
bool TargetIsRocm(target)   { return target->GetTargetDeviceType() == kDLROCM; }
bool TargetIsMetal(target)  { return target->GetTargetDeviceType() == kDLMetal; }
bool TargetIsCPU(target)    { return target->GetTargetDeviceType() == kDLCPU; }
```

### 5.2 Target Device Types

| Device Type | Constant | Backend |
|---|---|---|
| CUDA | `kDLCUDA` (2) | CUDA |
| ROCm | `kDLROCM` (10) | HIP/ROCm |
| Metal | `kDLMetal` (8) | Metal |
| CPU | `kDLCPU` (1) | C/LLVM |
| WebGPU | (custom) | WebGPU |

### 5.3 Backend Registration

Each backend registers itself during static initialization:

```cpp
// CUDA: src/backend/cuda/op/gemm.cc
const bool cuda_gemm_registered = RegisterCudaGemm();

// ROCm: src/backend/rocm/op/gemm.cc
const bool rocm_gemm_registered = RegisterROCmGemm();

// CPU: src/backend/cpu/op/gemm.cc
const bool cpu_gemm_registered = RegisterCPUGemm();
```

### 5.4 GEMM Instruction Resolution

```python
# From tilelang/backend/gemm.py
def resolve_gemm_impl(gemm_inst: str, target: Target) -> type:
    """Resolve the registered implementation class for a GEMM instruction key."""
    matches = [entry for entry in _GEMM_IMPLS
               if entry.inst_name == gemm_inst and entry.predicate(target)]
    if not matches:
        raise ValueError(f"No GEMM implementation for {gemm_inst} and {target}")
    if len(matches) > 1:
        raise ValueError(f"Multiple implementations for {gemm_inst}")
    return matches[0].impl_class
```

### 5.5 Copy Instruction Resolution

Similar to GEMM, copy operations are resolved per backend:

```cpp
// Registration pattern (from copy.h)
RegisterCopyImpl(CopyImpl{
    "backend.Copy",
    MatchTarget,
    priority,
    InferLayout,
    Lower,
});
```

### 5.6 Backend Selection Flow

```
1. User specifies target
   target = tl.target.Target("cuda", arch="sm_90")

2. TileLang compiles PrimFunc
   -> IR passes run

3. lower_tile_op pass encounters T.gemm()
   -> Calls Gemm::SelectInst(op, block_size, target)
   -> For CUDA SM90: returns "cuda.wgmma"

4. Python side resolves implementation
   -> resolve_gemm_impl("cuda.wgmma", target)
   -> Returns GemmWGMMA class

5. GemmWGMMA emits TIR macros
   -> WGMMA intrinsic macros

6. CodeGenTileLangCUDA produces CUDA C++
   -> Includes wgmma.h, barrier.h, etc.

7. NVRTC compiles to PTX/CUBIN
   -> Kernel is ready for execution
```

---

## 6. Target String Configuration for Each Backend

### 6.1 CUDA

```python
# Basic CUDA
target = tl.target.Target("cuda")

# Specific architecture
target = tl.target.Target("cuda", arch="sm_80")  # Ampere
target = tl.target.Target("cuda", arch="sm_90")  # Hopper
target = tl.target.Target("cuda", arch="sm_100") # Blackwell

# CuTe DSL target
target = tl.target.Target("cutedsl")
```

### 6.2 ROCm/HIP

```python
# Basic HIP
target = tl.target.Target("hip")

# Specific architecture
target = tl.target.Target("hip", mcpu="gfx90a")  # MI250X (CDNA 2)
target = tl.target.Target("hip", mcpu="gfx942")  # MI300X (CDNA 3)
target = tl.target.Target("hip", mcpu="gfx950")  # gfx950 (CDNA 3+)
target = tl.target.Target("hip", mcpu="gfx1100") # RX 7900 XTX (RDNA 3)
```

### 6.3 Metal

```python
target = tl.target.Target("metal")
```

### 6.4 CPU

```python
# C backend
target = tl.target.Target("c")

# LLVM backend
target = tl.target.Target("llvm")

# With specific CPU features
target = tl.target.Target("llvm", mcpu="skylake-avx512")
target = tl.target.Target("llvm", mattr=["+avx512f", "+avx512vl"])
```

### 6.5 WebGPU

```python
target = tl.target.Target("webgpu")
```

---

## 7. Backend-Specific Transforms

### 7.1 Transform Pass Overview

While most transform passes are backend-agnostic, some are specific to certain
backends:

| Transform | CUDA | ROCm | Metal | CPU | WebGPU |
|---|---|---|---|---|---|
| lower_tile_op | Yes | Yes | Yes | Yes | Yes |
| lower_hopper_intrin | SM90+ | No | No | No | No |
| lower_blackwell_2sm | SM100+ | No | No | No | No |
| lower_ptx_async_copy | SM80+ | No | No | No | No |
| lower_shared_tmem | SM100+ | No | No | No | No |
| inject_tcgen05_fence | SM100+ | No | No | No | No |
| inject_pipeline | Yes | Yes | Yes | Yes | Yes |
| lower_shared_barrier | SM80+ | gfx94+ | No | No | No |
| producer_consumer_ws | SM90+ | No | No | No | No |
| cluster_planning | SM90+ | No | No | No | No |

### 7.2 Metal-Specific Transforms

The Metal backend currently does not have Metal-specific transform passes. All
Metal-specific behavior is handled during code generation.

The `src/backend/metal/` directory structure:

```
src/backend/metal/
    CMakeLists.txt
    codegen/
        rt_mod_metal.cc    # Build entry point
    op/
        copy.cc            # Copy lowering
```

### 7.3 CPU-Specific Transforms

CPU-specific transforms:

```
src/backend/cpu/
    op/
        copy.cc            # Copy lowering
        gemm.cc            # Scalar GEMM selection
```

CPU templates:

```
src/tl_templates/cpu/
    common.h               # Common utilities
    gemm.h                 # Scalar GEMM implementation
```

### 7.4 WebGPU-Specific Transforms

WebGPU-specific transforms:

```
src/backend/webgpu/
    op/
        copy.cc            # Copy lowering
```

---

## 8. Cross-Backend Portability Guidelines

### 8.1 Writing Portable TileLang Code

To write TileLang code that runs on multiple backends:

#### Use Backend-Agnostic Types

```python
# Portable: use standard data types
dtype = "float16"      # Works on CUDA, ROCm, Metal
accum_dtype = "float32"  # Works everywhere

# Backend-specific: use with caution
# FP8 is CUDA-only (SM90+) and ROCm (gfx940+)
# BF16 works on CUDA (SM80+), ROCm (CDNA), but not Metal
```

#### Avoid Backend-Specific Operations

```python
# Portable
T.copy(A, B)           # Works on all backends
T.gemm(A, B, C)        # Dispatches to correct backend
T.barrier_sync()       # Maps to appropriate sync

# Backend-specific (avoid for portable code)
T.cluster_wait()       # CUDA SM90+ only
T.copy_async(A, B)     # CUDA SM80+ or ROCm gfx940+ only
```

#### Use Backend-Agnostic Layouts

```python
# Portable: simple layouts
layout = T.make_linear_layout(buffer)

# Backend-specific: hardware layouts
layout = T.make_wgmma_swizzled_layout(buffer)    # CUDA SM90+ only
layout = T.make_tcgen05mma_swizzled_layout(buffer) # CUDA SM100+ only
```

### 8.2 Feature Availability Matrix

| Feature | CUDA | ROCm | Metal | CPU | WebGPU |
|---|---|---|---|---|---|
| Tensor/Matrix Core | SM75+ | CDNA | No | No | No |
| FP16 | SM53+ | Yes | Yes | Yes | Limited |
| BF16 | SM80+ | CDNA | No | Yes | No |
| FP8 | SM90+ | gfx940+ | No | No | No |
| FP4 | SM100+ | No | No | No | No |
| INT8 | SM75+ | CDNA | No | Yes | No |
| Async copy | SM80+ | gfx940+ | No | No | No |
| TMA/Bulk copy | SM90+ | No | No | No | No |
| Cluster | SM90+ | No | No | No | No |
| Warp specialization | SM90+ | No | No | No | No |
| Pipeline | SM80+ | gfx940+ | No | No | No |
| Shared memory | Yes | Yes (LDS) | Yes (TG mem) | No | Yes |
| Tensor Memory | SM100+ | No | No | No | No |

### 8.3 Conditional Backend Code

```python
import tilelang as tl
from tilelang.utils.target import target_is_cuda, target_is_hip

def create_kernel(M, N, K, target):
    block_M = 128
    block_N = 128
    block_K = 32 if target_is_cuda(target) else 16

    @T.prim_func
    def main(A: T.Buffer((M, K), "float16"),
             B: T.Buffer((K, N), "float16"),
             C: T.Buffer((M, N), "float32")):
        # Backend-agnostic kernel
        ...

    return main
```

### 8.4 Performance Portability

When writing for multiple backends, consider:

1. **Block sizes:** Optimal block sizes differ between architectures
2. **Thread counts:** CUDA uses 128-256 threads, ROCm often uses 256-512
3. **Pipeline depth:** CUDA SM90 supports deep pipelines, others may not
4. **Data layout:** Optimal swizzle patterns differ between architectures

---

## 9. Custom Backend Registration

### 9.1 Overview

TileLang supports registering custom backends through the `register_gemm_impl`
and `RegisterCopyImpl` / `RegisterGemmImpl` (C++) mechanisms.

### 9.2 Python-Side Registration

#### GEMM Backend

```python
from tilelang.backend.gemm import register_gemm_impl

class MyCustomGemm:
    """Custom GEMM implementation."""
    # Must implement required methods
    pass

GEMM_INST_CUSTOM = "custom.instruction"

def _match_custom(target) -> bool:
    return target.kind.name == "my_custom_target"

register_gemm_impl(
    name="my_backend.gemm",
    inst_name=GEMM_INST_CUSTOM,
    predicate=_match_custom,
    impl_class=MyCustomGemm,
)
```

### 9.3 C++ Side Registration

#### GEMM Backend

```cpp
// In src/backend/custom/op/gemm.cc
#include "op/gemm.h"

namespace tvm {
namespace tl {
namespace custom {

struct Gemm {
    static String SelectInst(const GemmNode &op, int block_size, Target target) {
        return "custom.instruction";
    }

    static std::pair<int, int> ComputeWarpPartition(
        const GemmWarpPolicyNode &policy, int M, int N,
        int block_size, Target target, String gemm_inst) {
        // Custom warp partition logic
        policy.m_warp = 1;
        policy.n_warp = 1;
        return {1, 1};
    }

    static bool ReuseExistingSharedLayout(String gemm_inst) {
        return false;
    }

    static String InstructionKind(String gemm_inst) {
        return "custom";
    }
};

} // namespace custom

namespace {
bool MatchCustomGemmTarget(Target target) {
    // Match your custom target
    return false;
}

bool RegisterCustomGemm() {
    RegisterGemmImpl(GemmImpl{
        "custom.Gemm",
        MatchCustomGemmTarget,
        custom::Gemm::SelectInst,
        custom::Gemm::ComputeWarpPartition,
        custom::Gemm::ReuseExistingSharedLayout,
        custom::Gemm::InstructionKind,
    });
    return true;
}

const bool custom_gemm_registered = RegisterCustomGemm();
} // namespace

} // namespace tl
} // namespace tvm
```

#### Copy Backend

```cpp
#include "op/copy.h"

namespace tvm {
namespace tl {
namespace custom {

struct Copy {
    static LayoutMap InferLayout(const CopyNode &op, const LayoutInferArgs &T,
                                  InferLevel level) {
        return op.InferSIMTLayout(T, level);
    }

    static Stmt Lower(const CopyNode &op, const LowerArgs &T,
                       arith::Analyzer *analyzer) {
        return LowerNormalCopy(op, T, analyzer);
    }
};

} // namespace custom

namespace {
bool MatchCustomCopyTarget(Target target) {
    return false; // Match your custom target
}

bool RegisterCustomCopy() {
    RegisterCopyImpl(CopyImpl{
        "custom.Copy",
        MatchCustomCopyTarget,
        100,  // priority
        custom::Copy::InferLayout,
        custom::Copy::Lower,
    });
    return true;
}

const bool custom_copy_registered = RegisterCustomCopy();
} // namespace

} // namespace tl
} // namespace tvm
```

### 9.4 Code Generator Registration

To register a custom code generator:

```cpp
// In src/backend/custom/codegen/my_codegen.cc
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace codegen {

ffi::Module BuildTileLangCustom(IRModule mod, Target target) {
    // Custom code generation logic
    // ...
}

TVM_FFI_STATIC_INIT_BLOCK() {
    namespace refl = tvm::ffi::reflection;
    refl::GlobalDef().def("target.build.tilelang_custom", BuildTileLangCustom);
}

} // namespace codegen
} // namespace tvm
```

### 9.5 Registration Checklist

To add a new backend, you need:

1. **C++ side:**
   - `src/backend/<name>/op/gemm.cc` - GEMM instruction selection
   - `src/backend/<name>/op/copy.cc` - Copy operation lowering
   - `src/backend/<name>/codegen/` - Code generator (optional)
   - `src/tl_templates/<name>/` - C++ templates (optional)

2. **Python side:**
   - `tilelang/backend/<name>/__init__.py` - Backend module
   - `tilelang/backend/<name>/gemm.py` - GEMM implementation registration
   - `tilelang/intrinsics/` - Intrinsic emitters (for Tensor Core backends)

3. **Target detection:**
   - Add target detection in `src/target/utils.cc`
   - Add architecture checks as needed

4. **Build system:**
   - Add CMakeLists.txt in `src/backend/<name>/`
   - Register with parent CMakeLists.txt

---

## 10. Backend Comparison Summary

### 10.1 Feature Comparison

| Feature | CUDA | ROCm | Metal | CPU | WebGPU |
|---|---|---|---|---|---|
| **Codegen** | CUDA C++ | HIP C++ | Metal SL (via CHost) | C/C++ | WGSL |
| **Compiler** | NVRTC/NVCC | HIPRTC/hipcc | Metal framework | GCC/Clang | Browser |
| **GEMM inst** | MMA/WGMMA/TCGEN05 | MFMA/WMMA | Scalar | Scalar | Scalar |
| **Warp size** | 32 | 64/32 | 32 | 1 | 32 |
| **Shared mem** | 48-164 KB | 64 KB | 32 KB | N/A | 16-32 KB |
| **Async copy** | SM80+ | gfx940+ | No | No | No |
| **JIT compile** | Yes | Yes | Yes | No | Yes |

### 10.2 Performance Tier

1. **CUDA (Hopper SM90):** Highest performance, most features
2. **CUDA (Blackwell SM100):** Highest peak throughput
3. **ROCm (CDNA 3):** Competitive with CUDA for supported ops
4. **CUDA (Ampere SM80):** Good performance, wide hardware support
5. **ROCm (CDNA 2):** Good performance for MI250X
6. **Metal:** Moderate performance, Apple ecosystem
7. **CPU:** Lowest performance, useful for testing
8. **WebGPU:** Limited performance, browser sandbox

### 10.3 Maturity Level

| Backend | Maturity | Test Coverage |
|---|---|---|
| CUDA | Production | Extensive |
| ROCm | Production | Good |
| Metal | Experimental | Basic |
| CPU | Development | Basic |
| WebGPU | Experimental | Minimal |

---

## 11. File Organization Summary

### 11.1 Backend Source Structure

```
src/backend/
    cuda/
        CMakeLists.txt
        codegen/
            codegen_cuda.cc / .h
            codegen_cutedsl.cc / .h
            codegen_py.cc / .h
            intrin_rule_cuda.cc
            ptx.cc / .h
            rt_mod_cuda.cc
            rt_mod_cutedsl.cc
            stubs/ ...
        op/
            copy.cc / .h
            copy_analysis.cc
            gemm.cc
    rocm/
        CMakeLists.txt
        codegen/
            codegen_hip.cc / .h
            intrin_rule_hip.cc
            rt_mod_hip.cc
            stubs/ ...
        op/
            copy.cc
            gemm.cc
    metal/
        CMakeLists.txt
        codegen/
            rt_mod_metal.cc
        op/
            copy.cc
    cpu/
        op/
            copy.cc
            gemm.cc
    webgpu/
        op/
            copy.cc
```

### 11.2 Template Source Structure

```
src/tl_templates/
    cpp/       - Common C++ templates
    cpu/       - CPU templates
    cuda/      - CUDA templates (largest)
    hip/       - HIP/ROCm templates
```

### 11.3 Python Backend Structure

```
tilelang/backend/
    __init__.py
    gemm.py              - GEMM registration system
    cuda/
        __init__.py
        gemm.py           - CUDA GEMM registrations
    rocm/
        __init__.py
        gemm.py           - ROCm GEMM registrations
    cpu/
        __init__.py
        gemm.py           - CPU GEMM registration
```
