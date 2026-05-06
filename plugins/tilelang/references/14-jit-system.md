# JIT System

This document provides comprehensive reference for TileLang's Just-In-Time (JIT) compilation system, including the compilation APIs, execution backends, kernel caching, and runtime inspection capabilities.

## Table of Contents

- [Overview](#overview)
- [tilelang.compile](#tilelangcompile)
- [tilelang.par_compile](#tilelangpar_compile)
- [tilelang.jit](#tilelangjit)
- [JITImpl Class](#jitimpl-class)
- [JITKernel Class](#jitkernel-class)
- [Execution Backends](#execution-backends)
- [Kernel Caching System](#kernel-caching-system)
- [Compilation Flags](#compilation-flags)
- [Verbose Mode and Debugging](#verbose-mode-and-debugging)

---

## Overview

TileLang's JIT system provides three primary interfaces for compiling and executing GPU kernels:

1. **`tilelang.compile()`**: Compile a single kernel.
2. **`tilelang.par_compile()`**: Compile multiple kernels in parallel.
3. **`tilelang.jit`**: Decorator for automatic JIT compilation with caching.

All three interfaces share the same underlying compilation infrastructure (the `tilelang.engine.lower` pipeline) and kernel caching system. The JIT decorator adds automatic mode detection (lazy vs. eager) and argument-based caching.

---

## tilelang.compile

```python
tilelang.compile(
    func: PrimFunc = None,
    out_idx: list[int] | int | None = None,
    execution_backend: Literal["auto", "dlpack", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] | None = None,
    target: str | Target | None = None,
    target_host: str | Target | None = None,
    verbose: bool | None = None,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | str | None = None,
) -> JITKernel
```

Compiles a TileLang PrimFunc into a `JITKernel` that can be invoked like a regular Python function.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `PrimFunc` | required | The TileLang TIR function to compile |
| `out_idx` | `list[int]`, `int`, or `None` | `None` | Index(es) of output tensors to return |
| `execution_backend` | `str` or `None` | `None` | Backend for kernel execution (see [Execution Backends](#execution-backends)) |
| `target` | `str`, `Target`, or `None` | `None` | Compilation target |
| `target_host` | `str`, `Target`, or `None` | `None` | Host target for cross-compilation |
| `verbose` | `bool` or `None` | `None` | Enable verbose output |
| `pass_configs` | `dict` or `None` | `None` | Compiler pass configuration |
| `compile_flags` | `list[str]`, `str`, or `None` | `None` | Additional compiler flags |

### Output Index (out_idx)

The `out_idx` parameter specifies which function parameters should be treated as outputs and returned from the kernel invocation:

- `None`: No outputs returned.
- `int`: Return a single output tensor at the given parameter index.
- `list[int]`: Return multiple output tensors at the given indices.
- Negative indices count from the end: `-1` means the last parameter.

```python
# Return the last parameter (C in matmul)
kernel = tilelang.compile(func, out_idx=-1)

# Return multiple outputs
kernel = tilelang.compile(func, out_idx=[2, 3])
```

### Function-Level Attributes

If the PrimFunc has `tilelang_out_idx`, `tilelang_pass_configs`, or `tilelang_compile_flags` attributes, they are automatically extracted and merged with the explicit parameters:

```python
# The following attributes can be set on PrimFunc:
# func.attrs["tilelang_out_idx"] = [2]
# func.attrs["tilelang_pass_configs"] = {"tl.enable_fast_math": True}
# func.attrs["tilelang_compile_flags"] = ["--use_fast_math"]
```

### Environment Variable Defaults

When parameters are `None`, the following environment variables are read:

| Variable | Parameter | Default |
|----------|-----------|---------|
| `TILELANG_TARGET` | `target` | `"auto"` |
| `TILELANG_EXECUTION_BACKEND` | `execution_backend` | `"auto"` |
| `TILELANG_VERBOSE` | `verbose` | `False` |

### Examples

#### Basic Compilation

```python
import tilelang
from tilelang import T

@T.prim_func
def vector_add(
    A: T.Tensor((N,), "float32"),
    B: T.Tensor((N,), "float32"),
    C: T.Tensor((N,), "float32"),
):
    with T.Kernel(T.ceildiv(N, 256), threads=256) as pid:
        i = pid * 256 + T.get_lane_idx()
        C[i] = A[i] + B[i]

kernel = tilelang.compile(vector_add, out_idx=-1)

import torch
a = torch.randn(1024, device="cuda", dtype=torch.float32)
b = torch.randn(1024, device="cuda", dtype=torch.float32)
c = kernel(a, b)  # Returns C tensor
```

#### Compilation with Custom Pass Configs

```python
kernel = tilelang.compile(
    func,
    out_idx=-1,
    target="cuda",
    execution_backend="nvrtc",
    pass_configs={
        "tl.enable_fast_math": True,
        "tl.ptxas_register_usage_level": 5,
    },
    verbose=True,
)
```

---

## tilelang.par_compile

```python
tilelang.par_compile(
    funcs: Iterable[PrimFunc],
    out_idx: list[int] | int | None = None,
    execution_backend: str | None = None,
    target: str | Target | None = None,
    target_host: str | Target | None = None,
    verbose: bool | None = None,
    pass_configs: dict | None = None,
    compile_flags: list[str] | str | None = None,
    num_workers: int | None = None,
    ignore_error: bool = False,
) -> list[JITKernel]
```

Compiles multiple TileLang PrimFuncs in parallel using a thread pool.

### Additional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `funcs` | `Iterable[PrimFunc]` | required | The functions to compile |
| `num_workers` | `int` or `None` | `None` | Number of parallel compilation threads |
| `ignore_error` | `bool` | `False` | If True, log errors and return `None` for failed compilations |

### Examples

#### Parallel Compilation of Multiple Variants

```python
import tilelang
from tilelang import T

@T.prim_func
def matmul_128(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"), C: T.Tensor((M, N), "float32")):
    with T.Kernel(T.ceildiv(M, 128), T.ceildiv(N, 128), threads=128):
        # ... variant 1 ...

@T.prim_func
def matmul_256(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"), C: T.Tensor((M, N), "float32")):
    with T.Kernel(T.ceildiv(M, 256), T.ceildiv(N, 256), threads=256):
        # ... variant 2 ...

kernels = tilelang.par_compile(
    [matmul_128, matmul_256],
    out_idx=-1,
    target="cuda",
    num_workers=2,
)
```

#### Error-Tolerant Autotuning Compilation

```python
configs = [
    {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
    {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64},  # May fail on some GPUs
]

funcs = [make_matmul_func(**cfg) for cfg in configs]

kernels = tilelang.par_compile(
    funcs,
    out_idx=-1,
    ignore_error=True,  # Skip failed compilations
    num_workers=4,
)
# kernels[i] is None for failed compilations
```

---

## tilelang.jit

```python
@tilelang.jit(
    out_idx=None,
    target=None,
    target_host=None,
    execution_backend=None,
    verbose=None,
    pass_configs=None,
    debug_root_path=None,
    compile_flags=None,
)
def my_kernel_generator(...):
    ...
```

JIT compiler decorator for TileLang functions. Supports two execution modes that are automatically inferred.

### Decorator Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `out_idx` | `list[int]`, `int`, or `None` | `None` | Output tensor index(es). Only for lazy mode. |
| `target` | `str`, `Target`, or `None` | `None` | Compilation target |
| `target_host` | `str`, `Target`, or `None` | `None` | Host target |
| `execution_backend` | `str` or `None` | `None` | Execution backend |
| `verbose` | `bool` or `None` | `None` | Enable verbose output |
| `pass_configs` | `dict` or `None` | `None` | Pass configuration |
| `debug_root_path` | `str` or `None` | `None` | Directory to save debug source files |
| `compile_flags` | `list[str]`, `str`, or `None` | `None` | Additional compiler flags |

### Execution Modes

#### Lazy Mode

In lazy mode, the decorated function explicitly returns a `PrimFunc`. Calling the JIT wrapper returns a compiled `JITKernel` that can be invoked separately.

```python
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=128) as (bx, by):
            # ... kernel body ...
            pass
    return kernel  # Explicitly return PrimFunc

# Calling with parameters returns a JITKernel
kernel = matmul(1024, 1024, 1024, 128, 128, 32)

# Invoke the kernel
import torch
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = kernel(a, b)
```

#### Eager Mode

In eager mode, the decorated function uses the DSL builder pattern with tensor type annotations. Calling the JIT wrapper compiles and immediately executes the kernel, returning results directly.

```python
@tilelang.jit
def gemm(A, B, C, block_M: int = 64):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]
    C: T.Tensor[[M, N], dtype]
    with T.Kernel(...):
        # ... kernel body ...
        pass

# Calling with tensors executes immediately and returns results
result = gemm(a, b, c)
```

#### Mode Inference

The mode is automatically inferred:

- **Lazy**: Function returns a `PrimFunc` explicitly, or the function is a plain `PrimFunc`.
- **Eager**: Function uses the DSL builder pattern with tensor annotations.

Set `mode` explicitly to override auto-detection:

```python
@tilelang.jit(mode="lazy")  # Force lazy mode
def my_kernel_gen(...):
    ...
```

### Caching Behavior

The JIT decorator caches compiled kernels based on argument values. Repeated calls with the same arguments return the cached kernel:

```python
@tilelang.jit(out_idx=-1)
def matmul(M, N, K, block_M=128, block_N=128, block_K=32):
    # ...
    return kernel

# First call: compiles
k1 = matmul(1024, 1024, 1024)

# Second call with same args: returns cached kernel
k2 = matmul(1024, 1024, 1024)

assert k1 is k2  # Same object
```

### Debug Output

When `debug_root_path` is set, the JIT saves source files for each compiled kernel:

```python
@tilelang.jit(debug_root_path="debug_output")
def my_kernel(...):
    ...

# After compilation, debug_output/ contains:
# tilelang_jit_kernel_my_kernel.c  (CUDA source)
# tilelang_jit_program_my_kernel.py (TIR script)
```

---

## JITImpl Class

```python
@dataclass
class JITImpl(Generic[_P, _KP, _T, _Ret]):
    out_idx: list[int] | int | None
    execution_backend: str | None
    target: str | Target | None
    target_host: str | Target | None
    verbose: bool | None
    pass_configs: dict | None
    debug_root_path: str | None
    compile_flags: list[str] | str | None
    func_source: str
    signature: inspect.Signature
    mode: Literal["auto", "lazy", "eager"]
    func: JITFunc
```

The `JITImpl` class is the internal implementation behind `@tilelang.jit`. It manages mode detection, caching, and compilation.

### Key Methods

#### get_tir()

```python
jit_impl.get_tir(*args, **kwargs) -> PrimFunc
```

Retrieves the TIR PrimFunc for the given arguments. In lazy mode, calls the generator function. In eager mode, builds the TIR from the DSL.

#### compile()

```python
jit_impl.compile(*args, **kwargs) -> JITKernel
```

Compiles the kernel for the given arguments. Always recompiles (does not use cache).

#### par_compile()

```python
jit_impl.par_compile(
    configs: Iterable[dict | tuple],
    num_workers: int = None,
    ignore_error: bool = False,
) -> list[JITKernel]
```

Parallel compilation of multiple configurations. Each config is either a dict of keyword arguments or a tuple of positional arguments.

```python
@tilelang.jit(out_idx=-1)
def matmul(M, N, K, block_M=128, block_N=128):
    ...

kernels = matmul.par_compile([
    {"M": 1024, "N": 1024, "K": 1024, "block_M": 64, "block_N": 64},
    {"M": 1024, "N": 1024, "K": 1024, "block_M": 128, "block_N": 128},
    {"M": 1024, "N": 1024, "K": 1024, "block_M": 128, "block_N": 64},
])
```

#### get_kernel_source()

```python
jit_impl.get_kernel_source(*args, **kwargs) -> str
```

Compiles and returns the generated kernel source code.

#### __call__()

```python
jit_impl(*args, **kwargs) -> JITKernel (lazy) or result (eager)
```

Calls the JIT wrapper. Uses the internal cache to avoid redundant compilation. In lazy mode, returns the compiled `JITKernel`. In eager mode, executes the kernel and returns the result.

### Cache Key Generation

```python
key = jit_impl.parse_cache_key(*args, **kwargs)
```

The cache key is a tuple of:
- Positional arguments
- Sorted keyword arguments
- Sorted tuning parameters (from `__tune_params`)

---

## JITKernel Class

```python
class JITKernel(Generic[_P, _T]):
    prim_func: PrimFunc
    artifact: CompiledArtifact
    adapter: BaseKernelAdapter
    torch_function: Callable
    latency: float
    config: dict
    ref_latency: float
```

The `JITKernel` class wraps a compiled kernel and provides execution, profiling, and inspection capabilities.

### Construction

#### Direct Compilation

```python
kernel = JITKernel(
    func=prim_func,
    out_idx=-1,
    execution_backend="tvm_ffi",
    target="cuda",
    target_host=None,
    verbose=False,
    pass_configs=None,
    compile_flags=None,
)
```

#### From Database

```python
kernel = JITKernel.from_database(
    func=prim_func,
    host_kernel_source=host_src,
    device_kernel_source=device_src,
    kernel_lib_path="/path/to/lib.so",
    params=[...],
    target="cuda",
    target_host=None,
    out_idx=-1,
    execution_backend="tvm_ffi",
    pass_configs=None,
    compile_flags=None,
)
```

### __call__

```python
kernel(*args, **kwargs) -> Any
```

Invokes the compiled kernel with the given arguments. Arguments are PyTorch tensors or scalars.

```python
import torch
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

result = kernel(a, b)
```

The return value depends on `out_idx`:
- `None`: Returns None.
- Single index: Returns a single tensor.
- Multiple indices: Returns a tuple of tensors.

### get_profiler

```python
kernel.get_profiler(tensor_supply_type: TensorSupplyType = TensorSupplyType.Auto) -> Profiler
```

Creates a profiler for benchmarking the compiled kernel.

```python
profiler = kernel.get_profiler()
latency = profiler.run()  # Returns latency in ms
```

### get_kernel_source

```python
kernel.get_kernel_source(kernel_only: bool = True) -> str
```

Returns the generated kernel source code (CUDA/HIP/Metal).

```python
source = kernel.get_kernel_source()
print(source)
```

### get_host_source

```python
kernel.get_host_source() -> str
```

Returns the host-side source code.

### show_source

```python
kernel.show_source(which: Literal["kernel", "host", "both"] = "kernel") -> None
```

Prints generated source code to stdout.

```python
kernel.show_source()            # Print kernel source
kernel.show_source("host")      # Print host source
kernel.show_source("both")      # Print both sources
```

### export_sources

```python
kernel.export_sources(
    kernel_path: str | None = None,
    host_path: str | None = None,
) -> None
```

Exports generated source code to files.

```python
kernel.export_sources(kernel_path="/tmp/kernel.cu")
kernel.export_sources(host_path="/tmp/host.cc")
kernel.export_sources(
    kernel_path="/tmp/kernel.cu",
    host_path="/tmp/host.cc",
)
```

### export_library

```python
kernel.export_library(kernel_file: str) -> None
```

Exports the compiled kernel as a shared library. Requires `tvm_ffi` execution backend.

```python
kernel.export_library("/tmp/my_kernel.so")
```

**Note:** This raises an error if the runtime module is not available (i.e., compiled without `execution_backend="tvm_ffi"`).

### show_ptx / export_ptx

```python
kernel.show_ptx() -> None
kernel.export_ptx(path: str) -> None
```

Prints or exports the PTX (Parallel Thread Execution) assembly for CUDA kernels.

```python
kernel.show_ptx()                       # Print PTX to stdout
kernel.export_ptx("/tmp/kernel.ptx")    # Save PTX to file
```

### show_sass / export_sass

```python
kernel.show_sass() -> None
kernel.export_sass(path: str) -> None
```

Prints or exports the SASS (disassembled machine code) for CUDA kernels.

```python
kernel.show_sass()                       # Print SASS to stdout
kernel.export_sass("/tmp/kernel.sass")   # Save SASS to file
```

### run_once

```python
kernel.run_once(func: Callable | None = None) -> None
```

Executes the kernel once (for warm-up or validation).

### update_tuner_result

```python
kernel.update_tuner_result(
    latency: float,
    config: dict[str, Any],
    ref_latency: float,
) -> JITKernel
```

Updates autotuning results for this kernel. Returns `self` for chaining.

```python
kernel.update_tuner_result(
    latency=0.42,      # ms
    config={"BLOCK_M": 128, "BLOCK_N": 128},
    ref_latency=0.50,   # Reference latency
)
```

### get_tuner_result

```python
kernel.get_tuner_result() -> dict[str, Any]
```

Returns the stored tuning results as a dictionary with keys `latency`, `config`, and `ref_latency`.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `out_idx` | `list[int]` | Output tensor indices |
| `params` | `list[KernelParam]` | Kernel parameters |
| `kernel_source` | `str` | Generated kernel source |
| `host_source` | `str` | Host source code |

---

## Execution Backends

TileLang supports multiple execution backends that control how compiled kernels are loaded and executed:

### Auto Selection

When `execution_backend="auto"` (default), the backend is selected based on the target:

| Target | Default Backend |
|--------|----------------|
| CUDA | `tvm_ffi` |
| HIP | `tvm_ffi` |
| Metal | `tvm_ffi` (with `torch` as fallback) |
| C (CPU) | `cython` |

### Backend Descriptions

#### tvm_ffi

Uses TVM's Foreign Function Interface for kernel execution. Provides the most features:

- Full runtime module with host and device code.
- `export_library()` support.
- PTX/SASS inspection.
- DLPack tensor interop.

**Best for:** Production kernels, library export, full debugging.

#### dlpack

Historical alias for `tvm_ffi`. Maps to `tvm_ffi` internally.

#### cython

Compiles the host wrapper using Cython for direct kernel launch:

- Requires a C++ compiler.
- Lower overhead than `tvm_ffi` for small kernels.
- Source-level debugging of host code.

**Best for:** Performance-critical inference, environments without TVM runtime.

#### nvrtc

Uses NVIDIA's Runtime Compiler (NVRTC) for JIT compilation:

- Compiles CUDA source at runtime using `cuda-python`.
- No `nvcc` dependency at runtime.
- Fast compilation for small kernels.

**Best for:** Development, autotuning, environments without `nvcc`.

#### torch

Uses PyTorch's CUDA integration for Metal targets:

- Metal shader execution through PyTorch's MPS backend.
- Only available for Metal targets.

**Best for:** Apple Silicon development.

#### cutedsl

Uses NVIDIA's CuTe DSL for kernel execution:

- Only available when target keys contain "cutedsl".
- Generates Python executor instead of C code.

**Best for:** CuTe DSL development and testing.

### Backend Compatibility Matrix

| Backend | CUDA | HIP | Metal | CPU (C) | Export Library | PTX/SASS |
|---------|------|-----|-------|---------|----------------|----------|
| `tvm_ffi` | Yes | Yes | Yes | Yes | Yes | Yes (CUDA) |
| `cython` | Yes | Yes | No | Yes | No | No |
| `nvrtc` | Yes | No | No | No | No | Limited |
| `torch` | No | No | Yes | No | No | No |
| `cutedsl` | Yes (CuTeDSL target) | No | No | No | No | No |

---

## Kernel Caching System

TileLang implements a comprehensive kernel caching system through the `KernelCache` singleton class.

### Cache Architecture

The caching system has two levels:

1. **In-memory cache**: A dictionary mapping cache keys to `JITKernel` objects. Lives for the process lifetime.
2. **Disk cache**: Persistent cache stored in the filesystem. Survives across process restarts.

### Cache Flow

```
compile() request
    |
    v
Is caching enabled? --No--> Direct compilation
    |
    Yes
    v
Generate cache key
    |
    v
Check in-memory cache --Hit--> Return cached kernel
    |
    Miss
    v
Check disk cache --Hit--> Load from disk, update memory cache
    |
    Miss
    v
Compile kernel
    |
    v
Save to disk (atomic)
    |
    v
Update memory cache
    |
    v
Return kernel
```

### Cache Key Generation

The cache key is a SHA-256 hash of:

```python
{
    "func": sha256(func.script(show_meta=True).encode()).hexdigest(),
    "out_idx": ...,
    "target": ...,
    "target_host": ...,
    "execution_backend": ...,
    "pass_configs": ...,
    "compile_flags": ...,
    "version": tilelang.__version__,
    "platform": platform.machine(),
    "tilelang_lib": "<library_hash>",  # Content hash of libtilelang.so
}
```

### Disk Cache Structure

```
TILELANG_CACHE_DIR/
  <version>-<platform>/
    kernels/
      <sha256_hash>/
        device_kernel.cu     # Generated CUDA/HIP source
        host_kernel.cu       # Host wrapper source
        kernel_lib.so        # Compiled binary
        params.pkl           # Serialized kernel parameters
    .staging/                # Temporary directory for atomic writes
```

### Cache Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `TILELANG_CACHE_DIR` | Root cache directory | Platform-specific temp dir + `/tilelang_cache` |
| `TILELANG_CACHE_DISABLED` | Disable caching | Not set (caching enabled) |

### Atomic Cache Writes

The cache uses atomic directory rename for safe concurrent access:

1. All files are written to a staging directory under `.staging/`.
2. Once all files are complete, the staging directory is atomically renamed to the final cache path.
3. If the process crashes mid-write, the stale staging directory is cleaned up on next startup.

### Cache Invalidation

The cache is automatically invalidated when:
- The TileLang version changes.
- The `libtilelang.so` library content changes (detected via SHA-256 hash).
- The platform architecture changes.
- Any compilation parameter changes (target, pass configs, compile flags).
- The TIR function body changes.

### Manual Cache Management

```python
from tilelang.cache import KernelCache

# Clear all cached kernels
cache = KernelCache()
cache.clear_path()
```

---

## Compilation Flags

### Pass Configuration Flags

Pass configuration flags control the compilation pipeline behavior:

```python
pass_configs = {
    # Fast math
    "tl.enable_fast_math": True,

    # PTXAS register control
    "tl.ptxas_register_usage_level": 5,     # 0-10 scale
    "tl.enable_ptxas_verbose_output": True,

    # Additional nvcc flags
    "tl.device_compile_flags": [
        "-I/custom/include",
        "-DMY_DEFINE=1",
        "--ptxas-options=--verbose",
    ],

    # Optimization control
    "tl.disable_warp_specialized": False,
    "tl.enable_async_copy": True,
    "tl.enable_lower_ldgstg": True,

    # Debugging
    "tl.enable_dump_ir": True,
    "tl.dump_ir_path": "./dump_ir",
    "tl.ast_print_enable": True,

    # Layout visualization
    "tl.layout_visualization_enable": True,
    "tl.layout_visualization_formats": "txt,png",
}
```

### Compile Flags

The `compile_flags` parameter accepts additional flags passed to the device compiler:

```python
kernel = tilelang.compile(
    func,
    compile_flags=["--use_fast_math", "--ptxas-options=--verbose"],
)
```

These flags are merged with any flags from `pass_configs["tl.device_compile_flags"]`.

### Flag Precedence

1. Function-level attributes (`tilelang_pass_configs`, `tilelang_compile_flags`)
2. Explicit `pass_configs` parameter
3. Explicit `compile_flags` parameter
4. Environment variables

---

## Verbose Mode and Debugging

### Enabling Verbose Output

```python
# Option 1: Environment variable
# TILELANG_VERBOSE=1 python my_script.py

# Option 2: Parameter
kernel = tilelang.compile(func, verbose=True)

# Option 3: JIT decorator
@tilelang.jit(verbose=True)
def my_kernel(...):
    ...
```

### Verbose Output Includes

When verbose mode is enabled:

- Cache key generation information
- Cache hit/miss status
- Kernel compilation start/completion messages
- Source file save paths

### IR Dumping

To dump IR between compilation passes:

```python
pass_configs = {
    "tl.enable_dump_ir": True,
    "tl.dump_ir_path": "./dump_ir",
}
```

This creates a directory with TIR dumps after each transform pass.

### Source Inspection

```python
kernel = tilelang.compile(func, target="cuda")

# View generated CUDA source
kernel.show_source("kernel")

# View host source
kernel.show_source("host")

# View PTX assembly
kernel.show_ptx()

# View SASS disassembly
kernel.show_sass()

# Export all for offline analysis
kernel.export_sources(
    kernel_path="/tmp/kernel.cu",
    host_path="/tmp/host.cc",
)
kernel.export_ptx("/tmp/kernel.ptx")
kernel.export_sass("/tmp/kernel.sass")
```

### Compilation Logging

TileLang uses Python's standard `logging` module. Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

The logger messages include:
- `TileLang begins to compile kernel <name>`
- `TileLang completes to compile kernel <name>`
- Cache hit/miss information
- Compilation errors with context

### Debug Root Path

When using the JIT decorator with `debug_root_path`:

```python
@tilelang.jit(debug_root_path="debug_output")
def my_kernel(M, N, K):
    ...

my_kernel(1024, 1024, 1024)
```

This saves:
- `debug_output/tilelang_jit_kernel_<name>.c` - Generated CUDA source
- `debug_output/tilelang_jit_program_<name>.py` - TIR script
