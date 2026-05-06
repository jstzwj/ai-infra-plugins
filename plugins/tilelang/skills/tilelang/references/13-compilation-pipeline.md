# Compilation Pipeline

This document provides a comprehensive reference for TileLang's compilation pipeline, covering the complete flow from Python DSL to executable GPU kernels, including all compilation phases, target handling, and artifact inspection.

## Table of Contents

- [Overview](#overview)
- [Compilation Flow](#compilation-flow)
- [tilelang.engine.lower Module](#tilelangenginelower-module)
- [Compilation Phases](#compilation-phases)
- [Host Code Generation](#host-code-generation)
- [Device Code Generation](#device-code-generation)
- [Parameter Extraction](#parameter-extraction)
- [Target Canonicalization](#target-canonicalization)
- [CUDA/HIP Compilation Callbacks](#cudahip-compilation-callbacks)
- [External Kernel Name Collection](#external-kernel-name-collection)
- [CompiledArtifact Structure](#compiledartifact-structure)
- [IRModule Manipulation](#irmodule-manipulation)
- [Source Code Generation and Inspection](#source-code-generation-and-inspection)
- [PTX/SASS Generation and Inspection](#ptxsass-generation-and-inspection)

---

## Overview

TileLang's compilation pipeline transforms Python DSL programs into optimized GPU kernels through a multi-stage process:

```
Python DSL  -->  TIR (PrimFunc)  -->  Lowering  -->  Optimization  -->  Codegen  -->  Executable
```

The pipeline leverages TVM's infrastructure (TIR, IRModule, transform passes) while adding TileLang-specific passes for Tensor Core operations, memory layout optimization, and GPU-specific code generation.

---

## Compilation Flow

The complete compilation flow consists of the following stages:

### Stage 1: Python DSL to TIR

The user writes a kernel using TileLang's Python DSL with `@T.prim_func` decorator. The DSL constructs a TVM TIR (Tensor Intermediate Representation) `PrimFunc`:

```python
import tilelang
from tilelang import T

@T.prim_func
def my_kernel(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float32"),
):
    with T.Kernel(T.ceildiv(M, 128), T.ceildiv(N, 128), threads=128) as (bx, by):
        # ... kernel body ...
```

The DSL constructs include `T.Kernel`, `T.alloc_shared`, `T.alloc_fragment`, `T.copy`, `T.gemm`, etc., which are translated into TIR statements during Python execution.

### Stage 2: Pre-Lower Semantic Check

Before any lowering, the module is validated:
- AST printing (if enabled via pass config)
- Nested loop validation
- Fragment access in symbolic parallel loops

### Stage 3: Lower and Legalize

The TIR module is progressively legalized:
- Target binding
- Negative index legalization
- Parallel loop verification
- Warp specialization
- Pipeline planning and injection
- Layout inference
- Tile operation lowering
- Vectorized loop legalization
- Safe memory access legalization
- Access pointer lowering

### Stage 4: Optimize for Target

Target-specific optimizations are applied:
- TMEM lowering
- Buffer allocation placement
- Shared barrier lowering
- Buffer flattening
- Index bitwidth configuration
- Loop vectorization and unrolling
- Storage rewrite
- LDG/STG lowering
- Hopper intrinsic lowering
- Host/device splitting
- Shared memory merging
- Thread synchronization insertion
- Warp group register allocation
- Packed API generation

### Stage 5: Code Generation

The optimized IR is converted to target-specific source code:
- Device code: CUDA C++, HIP, or Metal
- Host code: C/C++ with runtime API calls

### Stage 6: Compilation to Binary

The generated source code is compiled to machine code:
- CUDA: `nvcc` produces cubin
- HIP: `hipcc` produces HSACO
- Metal: Metal shader compiler

---

## tilelang.engine.lower Module

The `tilelang.engine.lower` module is the central compilation entry point.

### lower()

```python
tilelang.engine.lower(
    func_or_mod: tir.PrimFunc | tvm.IRModule,
    target: str | Target = "auto",
    target_host: str | Target | None = None,
    runtime_only: bool = False,
    enable_host_codegen: bool = False,
    enable_device_compile: bool = False,
) -> CompiledArtifact
```

Compiles a TileLang PrimFunc or IRModule into a `CompiledArtifact`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func_or_mod` | `PrimFunc` or `IRModule` | required | The TileLang function or module to compile |
| `target` | `str` or `Target` | `"auto"` | Compilation target. `"auto"` detects GPU type. |
| `target_host` | `str` or `Target` | `None` | Host target for cross-compilation |
| `runtime_only` | `bool` | `False` | If True, skip parameter extraction |
| `enable_host_codegen` | `bool` | `False` | Whether to generate host code (default: False, handled by JIT) |
| `enable_device_compile` | `bool` | `False` | Whether to compile device code to binary (default: False, handled by JIT) |

**When `enable_host_codegen=True` and `enable_device_compile=True`:**
- Both host and device code are fully compiled.
- The returned `CompiledArtifact` includes `rt_mod` (runtime module) ready for execution.
- Used by the `tvm_ffi` execution backend.

**When both are `False` (default):**
- Only source code generation is performed.
- Device code is generated but not compiled to binary.
- Used by `cython`, `nvrtc`, and other backends that handle compilation separately.

**Returns:** `CompiledArtifact` containing the compiled kernel source, parameters, and optionally the runtime module.

**Example:**

```python
import tilelang
from tilelang import T

@T.prim_func
def my_kernel(A: T.Tensor((128, 128), "float16"), B: T.Tensor((128, 128), "float16")):
    with T.Kernel(1, 1, threads=128) as (bx, by):
        # ... kernel body ...

# Compile with host and device codegen
artifact = tilelang.lower(
    my_kernel,
    target="cuda",
    enable_host_codegen=True,
    enable_device_compile=True,
)

# Access the generated source
print(artifact.kernel_source)
```

---

## Compilation Phases

### PreLowerSemanticCheck

```python
PreLowerSemanticCheck(mod: IRModule) -> None
```

Validation-only pass that runs before any lowering. Raises user-friendly errors in Python rather than deep C++ stack traces.

**Checks performed:**
1. **AST Printing** (optional): If `tl.ast_print_enable` is set, prints the TIR AST for debugging.
2. **Nested Loop Checker**: Validates that nested loops follow TileLang's structural rules.
3. **Fragment Loop Checker**: Ensures symbolic parallel loops do not access fragment buffers incorrectly.

**Configuration:**

```python
# Disable pre-lower semantic checks
pass_configs = {"tl.disable_prelower_semantic_check": True}
```

### LowerAndLegalize

```python
LowerAndLegalize(mod: IRModule, target: Target) -> IRModule
```

The main lowering pipeline that transforms frontend Tile IR into TVM-compatible TIR. This phase runs a sequence of passes in a specific order:

**Pass sequence:**

1. **BindTarget(target)**: Binds target device information to the module.
2. **LetInline** (conditional): Force-inlines let bindings when `tl.force_let_inline` is enabled.
3. **AddWrapperForSingleBufStore**: Wraps single buffer store operations for consistent handling.
4. **LegalizeNegativeIndex**: Converts negative indices (e.g., `A[-1]`) to canonical non-negative form.
5. **VerifyParallelLoop** (conditional): Verifies parallel loop correctness, checking for data races.
6. **InjectAssumes**: Injects assume statements for boundary conditions to help TVM's prover.
7. **Simplify**: Simplifies TIR expressions.
8. **LayoutReducer**: Sets layouts for reduction operations.
9. **ProducerConsumerWarpSpecialized** (conditional): Rewrites pipelined tile-op loops into warp-specialized producer/consumer branches. Only enabled for CUDA targets with TMA support.
10. **LowerBlackwell2SM**: Lowers 2SM TCGEN5MMA operations on Blackwell targets.
11. **PipelinePlanning**: Plans pipeline stages for multi-stage shared memory buffers.
12. **InjectSoftwarePipeline**: Injects software pipeline structure.
13. **Simplify**: Second simplification pass.
14. **LayoutInferences**: Infers memory layouts for fragments and shared memory buffers.
15. **LowerTileOp**: Lowers high-level tile operations (T.copy, T.gemm) to low-level operations.
16. **LowerL2Persistent**: Lowers L2 persistence annotations.
17. **DecoupleTypeCast**: Separates type cast vectorization constraints.
18. **LegalizeVectorizedLoop**: Ensures vectorized loops are valid.
19. **LegalizeSafeMemoryAccess**: Inserts safety checks for out-of-bounds memory accesses.
20. **LowerAccessPtr**: Lowers `tl.access_ptr` to `tir.builtin.tvm_access_ptr`.
21. **Simplify**: Third simplification pass to clean up safety check artifacts.
22. **HoistNonRestrictParams**: Hoists non-restrict parameter annotations to PrimFunc attributes.

**Conditions for enabling specific passes:**

| Pass | Condition |
|------|-----------|
| `LetInline` | `tl.force_let_inline == True` |
| `VerifyParallelLoop` | `tl.disable_data_race_check == False` |
| `ProducerConsumerWarpSpecialized` | CUDA target with TMA support AND `tl.disable_warp_specialized == False` |

### OptimizeForTarget

```python
OptimizeForTarget(mod: IRModule, target: Target) -> IRModule
```

Applies target-specific optimizations to the lowered IR. This phase runs after `LowerAndLegalize` and performs the following sequence:

1. **LowerSharedTmem**: Lowers shared TMEM allocations to specific initialization slots.
2. **IfStmtBinding**: Binds if-statement conditions to variables for cleaner codegen.
3. **PlanAndUpdateBufferAllocationLocation**: Plans optimal buffer allocation placement.
4. **LowerSharedBarrier**: Lowers shared memory barrier operations.
5. **FuseMBarrierArriveExpectTx** (conditional): Fuses expect_tx -> TMA -> arrive patterns. Only when TMA is present.
6. **HoistGlobalBufferAllocations**: Moves global buffer allocations to host scope.
7. **LowerOpaqueBlock**: Lowers opaque block constructs.
8. **Simplify**: Simplification pass.
9. **NarrowDataType(32)**: Narrows data types to 32-bit indices.
10. **FlattenBuffer**: Flattens multi-dimensional buffer accesses to 1D.
11. **ConfigIndexBitwidth**: Configures index bitwidth (must run after FlattenBuffer).
12. **Simplify**: Simplification pass.
13. **VectorizeLoop**: Vectorizes loops when enabled.
14. **StorageRewrite**: Rewrites storage allocation for efficiency.
15. **LoopUnswitching**: Hoists loop-invariant if statements.
16. **UnrollLoop**: Unrolls loops based on configuration.
17. **RenormalizeSplitPattern**: Renormalizes split patterns.
18. **Simplify**: Simplification.
19. **RemoveNoOp**: Removes no-op statements.
20. **HoistIfThenElse**: Hoists if-then-else out of loops when possible.
21. **VerifyMemory**: Verifies memory access correctness.
22. **AnnotateEntryFunc**: Annotates the entry function.
23. **InferFragment**: Infers fragment information.
24. **LowerThreadAllreduce**: Lowers thread-level all-reduce operations.
25. **LowerLDGSTG**: Lowers ramp-based global loads/stores to ldg/stg intrinsics.
26. **LowerHopperIntrin**: Lowers Hopper-specific intrinsics.
27. **ThreadSync("global")** (conditional): Inserts global thread synchronization.
28. **AnnotateDeviceRegions**: Annotates device regions.
29. **SplitHostDevice**: Splits host and device functions.
30. **MarkCudaSyncCalls**: Marks CUDA synchronization calls.
31. **AnnotateReadOnlyParams**: Marks read-only parameters for const qualifier.
32. **MergeSharedMemoryAllocations**: Merges shared memory allocations.
33. **InjectFenceProxy**: Injects fence.proxy.async operations.
34. **ThreadSync("shared")**: Inserts shared memory synchronization.
35. **ThreadSync("shared.dyn")**: Inserts dynamic shared memory synchronization.
36. **InjectTcgen05Fence**: Injects TCGEN05 fences for Blackwell.
37. **MergeIfStmt**: Merges consecutive if statements.
38. **AnnotateWarpGroupRegAlloc** (conditional): Injects register allocation for warp specialization.
39. **MakePackedAPI**: Creates packed function API.
40. **Simplify**: Final simplification.
41. **LowerDeviceKernelLaunch**: Lowers device kernel launch constructs.
42. **PersistThreadblock**: Transforms to persistent threadblock scheduling.

---

## Host Code Generation

### host_codegen()

```python
host_codegen(
    host_mod: tvm.IRModule,
    target_host: Target,
    target: Target | None = None,
) -> tvm.IRModule
```

Generates host-side code from the lowered IR module. The host code handles kernel launch, parameter marshaling, and device memory management.

**Internal pass sequence:**

1. **BindTarget(target_host)**: Binds host target.
2. **FP8StorageLegalize**: Legalizes FP8 storage types.
3. **BF16StorageLegalize**: Legalizes BF16 storage types.
4. **LowerTVMBuiltin**: Lowers TVM built-in operations.
5. **LowerCustomDatatypes**: Lowers custom data types.
6. **LowerIntrin**: Lowers TileLang intrinsics.
7. **LowerDeviceStorageAccessInfo**: Lowers device storage access information.
8. **CombineContextCall**: Combines context call operations.
9. **MarkHostMetalContext** (conditional): Applies Metal/MPS synchronization for Metal targets.
10. **Target-specific build**:
    - `llvm`: Uses `target.build.llvm`
    - `c`: Uses `target.build.tilelang_c_host`
    - Others: Raises `ValueError`

**Metal-specific handling:**

When the device target is Metal, `MarkHostMetalContext` is applied so the generated host code contains Metal/MPS synchronization logic.

---

## Device Code Generation

### device_codegen()

```python
device_codegen(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule
```

Generates device-side code and compiles it to binary. The output is a TVM runtime module containing the compiled device code.

**Internal pass sequence:**

1. **LowerDeviceStorageAccessInfo**: Lowers device storage access info.
2. **LowerIntrin**: Lowers TileLang intrinsics.
3. **Simplify**: Simplification.
4. **HoistBroadcastValues**: Hoists broadcast values for efficiency.
5. **Target-specific codegen**:
    - CUDA: `target.build.tilelang_cuda` (or `target.build.tilelang_cutedsl` for CuTeDSL targets)
    - HIP: `target.build.tilelang_hip`
    - Metal: `target.build.metal`
    - Others: Raises `ValueError`

### device_codegen_without_compile()

```python
device_codegen_without_compile(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule
```

Generates device-side source code without compiling to binary. This is used by backends that perform their own compilation (e.g., NVRTC, Cython).

**Same internal pass sequence as `device_codegen()`**, but the final codegen step produces source code rather than compiled binary.

**Supported targets:**

| Target | Codegen Function |
|--------|-----------------|
| `cuda` | `target.build.tilelang_cuda_without_compile` |
| `cuda` (cutedsl) | `target.build.tilelang_cutedsl_without_compile` |
| `hip` | `target.build.tilelang_hip_without_compile` |
| `c` | `target.build.tilelang_c` |
| `llvm` | `target.build.llvm` |
| `webgpu` | `target.build.webgpu` |
| `metal` | `target.build.metal` |

---

## Parameter Extraction

### extrac_params()

```python
extrac_params(func: tir.PrimFunc) -> list[KernelParam]
```

Extracts kernel parameters from a PrimFunc, converting buffer parameters to `KernelParam` objects.

**Parameter types:**

1. **Buffer parameters**: Parameters that are in the function's `buffer_map`. Converted using `KernelParam.from_buffer()`, preserving dtype and shape information.

2. **Scalar parameters**: Parameters not in the `buffer_map`. Converted using `KernelParam.from_var()`, with an empty shape.

```python
# For a function with signature:
# def kernel(A: T.Tensor((M, K), "float16"), alpha: T.float32)
# extrac_params returns:
# [
#   KernelParam(dtype="float16", shape=[M, K]),  # A
#   KernelParam(dtype="float32", shape=[]),       # alpha (scalar)
# ]
```

---

## Target Canonicalization

### canon_target_host()

```python
canon_target_host(
    target: str | Target,
    target_host: str | Target | None,
) -> str
```

Canonicalizes the host target. If no host target is specified, defaults to `"llvm"` if TVM was built with LLVM support, otherwise `"c"`.

### determine_target()

```python
from tilelang.utils.target import determine_target

target = determine_target("auto")  # Auto-detect GPU
target = determine_target("cuda")  # Explicit CUDA
target = determine_target("hip")   # Explicit HIP
```

Resolves the compilation target. The `"auto"` setting auto-detects available GPU hardware.

---

## CUDA/HIP Compilation Callbacks

### tilelang_callback_cuda_compile

```python
@tvm_ffi.register_global_func("tilelang_callback_cuda_compile")
def tilelang_callback_cuda_compile(code, target, pass_config=None) -> bytes
```

Registered as a TVM global function, this callback is invoked during CUDA device code compilation.

**Compilation process:**

1. Determine target architecture from the target's compute version.
2. Build compiler options:
   - `-std=c++17`
   - `-I` for TileLang templates and CUTLASS includes
   - Optional: `--use_fast_math` (when `tl.enable_fast_math` is set)
   - Optional: `--ptxas-options=--register-usage-level=N` (when `tl.ptxas_register_usage_level` is set)
   - Optional: `--ptxas-options=--verbose` (when `tl.enable_ptxas_verbose_output` is set)
   - Additional flags from `tl.device_compile_flags`
3. Compile with `nvcc.compile_cuda()` to produce cubin.

**Pass config keys used:**

| Key | Description |
|-----|-------------|
| `tl.enable_fast_math` | Pass `--use_fast_math` to nvcc |
| `tl.ptxas_register_usage_level` | Set ptxas register usage level |
| `tl.enable_ptxas_verbose_output` | Enable verbose ptxas output |
| `tl.device_compile_flags` | Additional device compiler flags |

### tilelang_callback_cuda_validate

```python
@tvm_ffi.register_global_func("tilelang_callback_cuda_validate")
def tilelang_callback_cuda_validate(device_mod) -> None
```

Validates that CUDA source code kernels are correctly structured. Checks:
- Source-kernel PrimFuncs have the `global_symbol` attribute.
- The `global_symbol` matches the `code_block_entry_name`.
- The source contains at least one `__global__` kernel function.
- The expected kernel name is found in the source.

### tilelang_callback_hip_compile

```python
@tvm_ffi.register_global_func("tilelang_callback_hip_compile")
def tilelang_callback_hip_compile(code, target) -> bytes
```

Compiles HIP device code to HSACO using `hipcc.compile_hip()`.

**Options:**
- `-std=c++17`
- `-I` for TileLang templates
- `-I` for Composable Kernel includes

---

## External Kernel Name Collection

### _collect_external_cuda_kernel_names()

```python
_collect_external_cuda_kernel_names(source: str) -> list[str]
```

Parses CUDA source code to find all `__global__` kernel function names. Uses a regex pattern to match:

```c
extern "C" __global__ void __launch_bounds__(...) kernel_name(...)
__global__ void kernel_name(...)
```

This is used by `tilelang_callback_cuda_validate` to ensure the expected kernel is present in the source.

---

## CompiledArtifact Structure

```python
@dataclass
class CompiledArtifact:
    host_mod: tvm.IRModule                    # Host-side IR module
    device_mod: tvm.IRModule                  # Device-side IR module
    params: list[KernelParam]                 # Kernel parameters
    kernel_source: str                        # Generated kernel source code
    rt_mod: tvm.runtime.Module | None = None  # Runtime module (if compiled)
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `host_mod` | `tvm.IRModule` | Host-side IR module for managing kernel execution |
| `device_mod` | `tvm.IRModule` | Device-side IR module containing the kernel code |
| `params` | `list[KernelParam]` | List of kernel parameters (tensors/scalars) |
| `kernel_source` | `str` | Raw source code of the generated kernel |
| `rt_mod` | `tvm.runtime.Module` or `None` | Runtime module for execution. Available when `enable_host_codegen=True` and `enable_device_compile=True`. |

### KernelParam

```python
@dataclass
class KernelParam:
    dtype: tvm.DataType    # Data type (supports all TVM types including float8, float4)
    shape: list[int | Var] # Dimensions (integers or symbolic variables)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `is_scalar()` | `bool` | True if empty shape (scalar parameter) |
| `is_unsigned()` | `bool` | True if unsigned integer type |
| `is_float8()` | `bool` | True if float8 type (e4m3/e5m2) |
| `is_float4()` | `bool` | True if float4 type |
| `is_boolean()` | `bool` | True if boolean type |
| `torch_dtype()` | `torch.dtype` | Convert to PyTorch dtype |
| `tilelang_dtype()` | `T.dtype` | Convert to TileLang dtype |

---

## IRModule Manipulation

### Creating an IRModule from a PrimFunc

```python
# The lower() function automatically wraps PrimFunc in an IRModule:
mod = tvm.IRModule({func.attrs["global_symbol"]: func})
```

### Filtering Host/Device Functions

The pipeline splits the IRModule into host and device portions:

```python
from tilelang.engine.lower import get_host_call, get_device_call

_is_host_call = get_host_call(is_device_c=is_cpu_device_backend(target))
_is_device_call = get_device_call(is_device_c=is_cpu_device_backend(target))

host_mod = tir.transform.Filter(_is_host_call)(mod)
device_mod = tir.transform.Filter(_is_device_call)(mod)
```

### Function Classification

Functions are classified based on their `calling_conv` attribute:

| Classification | Condition |
|---------------|-----------|
| Device call (GPU) | `calling_conv == DEVICE_KERNEL_LAUNCH` |
| Device call (CPU/C backend) | `target.kind.name == "c"` and not `C_PACKED_FUNC` |
| Host call | Everything else |

---

## Source Code Generation and Inspection

### Inspecting Generated Source

```python
# After lowering:
artifact = tilelang.lower(func, target="cuda")

# Get device kernel source (CUDA C++)
print(artifact.kernel_source)

# Get host module
print(str(artifact.host_mod))

# Get device module
print(str(artifact.device_mod))
```

### IRModule Source Inspection

The `IRModule` object can be inspected for its source:

```python
# inspect_source() returns the generated source code
source = codegen_mod.inspect_source()
```

---

## PTX/SASS Generation and Inspection

### PTX Generation

PTX (Parallel Thread Execution) is the intermediate assembly language for NVIDIA GPUs. It can be generated from the CUDA source:

```python
# Using the nvcc helper
from tilelang.contrib import nvcc as tl_nvcc

code = kernel.get_kernel_source()
ptx = tl_nvcc.get_ptx_from_source(code, compile_flags=None, verbose=False)
```

### SASS Generation

SASS (Source Assembly) is the actual machine code executed by the GPU. It is obtained by disassembling the cubin:

```python
# Using the nvcc helper
sass = tl_nvcc.get_sass_from_source(code, compile_flags=None, verbose=False)
```

### JITKernel PTX/SASS Methods

The `JITKernel` class provides convenient methods:

```python
kernel = tilelang.compile(func, target="cuda")

# Print PTX
kernel.show_ptx()

# Export PTX to file
kernel.export_ptx("/tmp/kernel.ptx")

# Print SASS
kernel.show_sass()

# Export SASS to file
kernel.export_sass("/tmp/kernel.sass")
```

---

## Compilation Configuration

### Pass Configuration Keys

The compilation pipeline behavior can be controlled through pass configuration:

```python
pass_configs = {
    # Fast math
    "tl.enable_fast_math": True,

    # Register control
    "tl.ptxas_register_usage_level": 5,

    # Verbose output
    "tl.enable_ptxas_verbose_output": True,

    # Additional compiler flags
    "tl.device_compile_flags": ["-DMY_FLAG=1", "--ptxas-options=--verbose"],

    # Disable specific optimizations
    "tl.disable_warp_specialized": False,
    "tl.disable_data_race_check": False,
    "tl.disable_vectorize_256": False,

    # Async copy
    "tl.enable_async_copy": True,

    # LDG/STG lowering
    "tl.enable_lower_ldgstg": False,
    "tl.enable_lower_ldgstg_predicated": False,

    # Index bitwidth
    "tl.config_index_bitwidth": 32,

    # Storage rewrite
    "tl.storage_rewrite_detect_inplace": False,

    # Aggressive shared memory merge
    "tl.enable_aggressive_shared_memory_merge": False,

    # Debug IR dump
    "tl.enable_dump_ir": True,
    "tl.dump_ir_path": "./dump_ir",

    # Layout visualization
    "tl.layout_visualization_enable": True,
    "tl.layout_visualization_formats": "txt,png",
}
```

### Using Pass Configuration

```python
import tilelang

kernel = tilelang.compile(
    func,
    target="cuda",
    pass_configs=pass_configs,
)
```

Or within a `PassContext`:

```python
with tilelang.transform.PassContext(opt_level=3, config=pass_configs):
    artifact = tilelang.lower(func, target="cuda")
```

---

## Target-Specific Behavior

### CUDA Targets

For CUDA targets, the pipeline:
- Uses `nvcc` for compilation to cubin
- Supports TMA on SM90+ (Hopper)
- Supports WGMMA on SM90+
- Supports TCGEN05 on SM100+ (Blackwell)
- Can use CuTeDSL codegen when target keys contain "cutedsl"

### HIP Targets

For HIP targets:
- Uses `hipcc` for compilation to HSACO
- Supports MFMA on gfx9+ (CDNA)
- Supports LDS transpose on gfx950

### Metal Targets

For Metal targets:
- Uses Metal shader compiler
- Host code includes Metal/MPS synchronization
- Execution backend is `"torch"`

### CPU (C backend)

For CPU targets:
- Uses C code generator
- Execution backends: `"cython"` or `"tvm_ffi"`
- No GPU-specific intrinsics available

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TILELANG_CACHE_DIR` | Root directory for kernel cache | Platform-specific |
| `TILELANG_TARGET` | Default compilation target | `"auto"` |
| `TILELANG_EXECUTION_BACKEND` | Default execution backend | `"auto"` |
| `TILELANG_VERBOSE` | Enable verbose compilation | `"0"` |

---

## Error Handling

The compilation pipeline includes several layers of error checking:

1. **Pre-lower semantic check**: Catches structural errors before lowering.
2. **VerifyParallelLoop**: Detects data races in parallel loops.
3. **VerifyMemory**: Validates memory access patterns after optimization.
4. **CUDA validation**: Ensures generated CUDA source has correct kernel names.
5. **Target validation**: Checks that the requested target and execution backend are compatible.
