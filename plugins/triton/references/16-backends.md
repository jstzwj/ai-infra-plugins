# 16. Triton Backend System -- Comprehensive Reference

This document provides an exhaustive reference for the Triton backend system, covering architecture, abstract base classes, concrete implementations (NVIDIA CUDA, AMD HIP), tensor descriptor handling, compilation stages, and the plugin/entry-point discovery mechanism.

---

## Table of Contents

1. [Backend Architecture Overview](#1-backend-architecture-overview)
2. [GPUTarget Dataclass](#2-gputarget-dataclass)
3. [Language Enum](#3-language-enum)
4. [BaseBackend Abstract Base Class](#4-basebackend-abstract-base-class)
5. [DriverBase Abstract Base Class](#5-driverbase-abstract-base-class)
6. [GPUDriver Concrete Class](#6-gpudriver-concrete-class)
7. [Backend Discovery and Registration](#7-backend-discovery-and-registration)
8. [NVIDIA Backend (CUDABackend / CudaDriver)](#8-nvidia-backend-cudabackend--cudardriver)
9. [AMD Backend (HIPBackend / HIPDriver)](#9-amd-backend-hipbackend--hipdriver)
10. [Tensor Descriptor Handling](#10-tensor-descriptor-handling)
11. [Compilation Stages](#11-compilation-stages)
12. [External Plugin Backends (TRITON_PLUGIN_DIRS)](#12-external-plugin-backends-triton_plugin_dirs)
13. [Runtime Driver Selection](#13-runtime-driver-selection)

---

## 1. Backend Architecture Overview

Triton's backend system is the bridge between Triton's frontend/language and the actual GPU hardware. Every hardware target (NVIDIA CUDA, AMD HIP, or a custom accelerator) requires two components:

1. **Compiler backend** (`compiler.py`) -- Translates Triton IR through a pipeline of intermediate representations down to executable device code (e.g., CUDA cubin, AMD HSACO).
2. **Driver backend** (`driver.py`) -- Manages device interaction: querying device properties, loading compiled binaries, launching kernels, and allocating scratch memory.

These are bundled together under a named "backend" (e.g., `"nvidia"`, `"amd"`) and registered at package installation time via Python entry points in the `"triton.backends"` group.

### High-Level Flow

```
Triton Kernel (Python/JIT)
        |
        v
  JITLayer (triton.runtime.jit)
        |
        +--> backend.parse_options(kwargs)
        +--> backend.add_stages(stages, options, language)
        |         |
        |         v
        |    Stages run sequentially:
        |      ttir -> ttgir -> llir -> ptx/amdgcn -> cubin/hsaco
        |
        +--> driver.launch(grid, stream, function, ...)
        |
        v
  Execution on GPU
```

### Source File Layout

```
triton/
  backends/
    __init__.py            # Backend discovery, Backend dataclass
    compiler.py            # BaseBackend ABC, GPUTarget, Language
    driver.py              # DriverBase ABC, GPUDriver, tensor descriptor utilities
  third_party/
    nvidia/backend/
      compiler.py          # CUDABackend, CUDAOptions
      driver.py            # CudaDriver, CudaUtils, CudaLauncher
    amd/backend/
      compiler.py          # HIPBackend, HIPOptions
      driver.py            # HIPDriver, HIPUtils, HIPLauncher
```

---

## 2. GPUTarget Dataclass

**File:** `triton/backends/compiler.py`

```python
@dataclass(frozen=True)
class GPUTarget(object):
    backend: str              # Target backend name, e.g., "cuda", "hip"
    arch: Union[int, str]     # Architecture identifier
    warp_size: int            # Warp/wavefront size
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `backend` | `str` | The backend identifier string. `"cuda"` for NVIDIA, `"hip"` for AMD. Used by `supports_target()` to match backends. |
| `arch` | `Union[int, str]` | Architecture identifier. For NVIDIA this is an integer representing the compute capability (e.g., `80` for SM80, `90` for SM90). For AMD this is a string like `"gfx942"` or `"gfx950"`. |
| `warp_size` | `int` | The warp (NVIDIA) or wavefront (AMD) size in threads. NVIDIA always uses 32. AMD uses 64 for gfx9xx and earlier, 32 for gfx10xx and later. |

The dataclass is frozen (immutable) and is the primary means of identifying the compilation target. It is stored as `self.target` on `BaseBackend` instances and is used throughout the compilation and driver pipelines.

### How it is created

- **NVIDIA:** `CudaDriver.get_current_target()` queries the current device's capability via `torch.cuda.get_device_capability(device)`, computes `capability = major * 10 + minor`, and returns `GPUTarget("cuda", capability, 32)`.
- **AMD:** `HIPDriver.get_current_target()` queries device properties from the HIP runtime, reads the `arch` string from properties (optionally overridden by `knobs.runtime.override_arch`), and returns `GPUTarget("hip", arch, warp_size)`.

---

## 3. Language Enum

**File:** `triton/backends/compiler.py`

```python
class Language(Enum):
    """The input language being compiled by the backend."""
    TRITON = 0
    GLUON = 1
```

The `Language` enum specifies which frontend language produced the IR being compiled:

- **`TRITON`** (0): The standard Triton Python DSL. Compilation starts from Triton IR (TTIR) and proceeds through the standard pipeline.
- **`GLUON`** (1): The Gluon language, a lower-level representation. Compilation skips the TTIR stage and begins directly at the TTGIR level via a dedicated `gluon_to_ttgir` path.

The `language` parameter is passed to `add_stages()` so each backend can register the appropriate compilation stages depending on the source language.

---

## 4. BaseBackend Abstract Base Class

**File:** `triton/backends/compiler.py`

```python
class BaseBackend(metaclass=ABCMeta):
    supports_native_tensor_specialization = True
```

`BaseBackend` is the abstract base class that every compiler backend must subclass. It defines the contract for parsing options, registering compilation stages, loading dialects, and providing module maps.

### Class Attribute

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `supports_native_tensor_specialization` | `bool` | `True` | Whether the backend supports native tensor specialization. Set to `False` by `HIPBackend` because AMD does not yet support TDM (Tensor Descriptor Map) natively. |

### Constructor

```python
def __init__(self, target: GPUTarget) -> None:
    self.target = target
    assert self.supports_target(target)
```

Stores the `GPUTarget` and asserts the backend actually supports it.

### Abstract Methods

#### `supports_target(target: GPUTarget) -> bool` (static, abstract)

Returns whether this backend can compile for the given `GPUTarget`. Each concrete backend checks the `target.backend` field:

- `CUDABackend`: `return target.backend == 'cuda'`
- `HIPBackend`: `return target.backend == 'hip'`

#### `hash() -> str` (abstract)

Returns a unique string identifier for this backend configuration. Used for caching compiled kernels. The hash must incorporate any factor that would change the compiled output (target architecture, toolchain version, etc.).

- `CUDABackend`: Returns `f'{ptxas_version}-{target.arch}'` -- combines the ptxas version string with the SM architecture number.
- `HIPBackend`: Returns `f'{self.target}'` -- uses the string representation of the entire GPUTarget.

#### `parse_options(options: dict) -> object` (abstract)

Converts a raw dictionary of user-provided options into a structured, validated options object. This is where backend-specific heuristics and legality checks live. The returned object is an opaque type from the caller's perspective (though it is always a frozen `@dataclass` in practice: `CUDAOptions` or `HIPOptions`).

**Responsibilities:**
- Fill in default values for missing options.
- Validate option combinations (e.g., `num_ctas > 1` requires SM90+ on NVIDIA).
- Inject architecture-specific defaults (e.g., supported FP8 dtypes vary by compute capability).
- Apply knob overrides (e.g., `knobs.runtime.override_arch`).
- Set up external library paths.

#### `add_stages(stages: dict, options: object, language: Language) -> None` (abstract)

Populates the `stages` dictionary with compilation pipeline stages. Each entry maps:

```
stage_name (str) => Callable[[src: str, metadata: dict], str | bytes]
```

Stages are executed sequentially in insertion order. Each stage receives the output of the previous stage as `src` and a shared `metadata` dictionary. All stages return `str` except the final stage which returns `bytes` (the executable binary).

See [Section 11: Compilation Stages](#11-compilation-stages) for full details.

#### `load_dialects(context)` (abstract)

Loads additional MLIR dialects into the provided MLIR `context`. Called before compilation begins to ensure required dialects are available.

- `CUDABackend`: Calls `nvidia.load_dialects(ctx)`, plus any instrumentation dialects.
- `HIPBackend`: Calls `amd.load_dialects(ctx)`, plus any instrumentation dialects.

#### `get_module_map() -> Dict[str, ModuleType]` (abstract)

Returns a mapping from interface module names to their device-specific implementation modules. This allows the Triton language runtime to dispatch to the correct backend-specific `libdevice` implementation.

- `CUDABackend`: `{"triton.language.extra.libdevice": triton.language.extra.cuda.libdevice}`
- `HIPBackend`: `{"triton.language.extra.libdevice": triton.language.extra.hip.libdevice}`

### Non-Abstract Methods (Overridable)

#### `parse_attr(desc: str) -> list` (static)

Parses an attribute descriptor string and returns a list of `[attribute_name, value]` pairs.

Default implementation checks for `"D"` (divisibility) in `desc`:
```python
ret = []
if "D" in desc:
    ret += [["tt.divisibility", 16]]
return ret
```

`HIPBackend` overrides this to also check for `"S"` (pointer range for buffer operations):
```python
ret = BaseBackend.parse_attr(desc)
if "S" in desc:
    ret += [["tt.pointer_range", 32]]
return ret
```

#### `get_int_specialization(arg, **kwargs) -> str` (static)

Returns a specialization descriptor string for integer arguments. Checks if the argument is divisible by 16 and the `align` kwarg is set, returning `"D"` for divisible or `""` otherwise.

#### `get_tensor_specialization(arg, **kwargs) -> str` (static)

Returns a specialization descriptor string for tensor arguments. Checks if the tensor's data pointer is 16-byte aligned and `align` is set, returning `"D"` or `""`.

`HIPBackend` overrides this to additionally check for pointer range (buffer operations):
```python
ret = BaseBackend.get_tensor_specialization(arg, **kwargs)
if knobs.amd.use_buffer_ops and HIPBackend.is_within_2gb(arg):
    ret += "S"
return ret
```

### Additional Methods on Concrete Backends (Not in ABC)

These methods are called from `add_stages()` and represent individual compilation passes. They are not part of the abstract interface but follow a consistent pattern across backends:

| Method | Description |
|--------|-------------|
| `make_ttir(mod, metadata, options, ...)` | Produces Triton IR from the frontend output. Runs inlining, canonicalization, CSE, loop unrolling. |
| `make_ttgir(mod, metadata, options, ...)` | Converts TTIR to Triton GPU IR. Architecture-specific optimizations (tensor core lowering, pipelining, warp specialization). |
| `gluon_to_ttgir(src, metadata, options, ...)` | Alternative entry point for Gluon language, skipping TTIR. |
| `make_llir(src, metadata, options, ...)` | Converts TTGIR to LLVM IR. Allocates shared memory, lowers to LLVM dialect, links external libraries. |
| `make_ptx(src, metadata, opt, capability)` | (NVIDIA only) Translates LLVM IR to PTX assembly. |
| `make_cubin(src, metadata, opt, capability)` | (NVIDIA only) Assembles PTX to cubin using `ptxas`. |
| `make_amdgcn(src, metadata, options)` | (AMD only) Translates LLVM IR to AMDGCN assembly. |
| `make_hsaco(src, metadata, options)` | (AMD only) Assembles AMDGCN to HSACO binary. |
| `pack_metadata(metadata)` | Packs compilation metadata into a tuple for the launcher. |
| `get_codegen_implementation(options)` | Returns a dict of codegen callbacks (e.g., `min_dot_size`, `convert_custom_types`). |
| `get_target_name(options)` | Returns the target name string (e.g., `"cuda:90"`, `"hip:gfx942"`). |

---

## 5. DriverBase Abstract Base Class

**File:** `triton/backends/driver.py`

```python
class DriverBase(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass
```

`DriverBase` defines the interface for runtime device interaction. Every backend must provide a concrete driver.

### Abstract Methods

#### `is_active() -> bool` (classmethod, abstract)

Returns whether this backend's hardware and software stack is currently available and active. Called during driver selection to determine which backend to use.

- `CudaDriver`: Checks if `libcuda.so.1` can be loaded, `cuInit(0)` succeeds, and at least one CUDA device is present.
- `HIPDriver`: Checks if `torch.cuda.is_available()` and `torch.version.hip is not None`.

#### `map_python_to_cpp_type(ty: str) -> str` (abstract)

Converts a Triton type string to its corresponding C++ type string. Used by the launcher code generation to determine argument types.

Example mappings for NVIDIA:
| Triton Type | C++ Type |
|-------------|----------|
| `*i32` | `CUdeviceptr` |
| `tensordesc` | `CUtensorMap` |
| `i1` | `int8_t` |
| `i32` | `int32_t` |
| `i64` | `int64_t` |
| `fp16` | `double` |
| `fp32` | `double` |
| `fp64` | `double` |
| `nvTmaDesc` | `CUtensorMap` |

Example mappings for AMD:
| Triton Type | C++ Type |
|-------------|----------|
| `*i32` | `hipDeviceptr_t` |
| `tensordesc` | `TDMDescriptor` |
| `i1` | `int8_t` |
| `i32` | `int32_t` |
| `fp16` | `double` |
| `fp32` | `double` |

Note: Pointer types are detected by a leading `'*'` character.

#### `get_current_target() -> GPUTarget` (abstract)

Returns the `GPUTarget` for the currently active GPU device. This target is used to instantiate the correct compiler backend.

#### `get_active_torch_device() -> torch.device` (abstract)

Returns the PyTorch `torch.device` object corresponding to the current GPU. Both NVIDIA and AMD backends return `torch.device("cuda", device_index)` because PyTorch uses the `"cuda"` device string even for HIP devices.

#### `get_benchmarker() -> Benchmarker` (abstract)

Returns the benchmarking function to use for this backend. Both NVIDIA and AMD return `triton.testing.do_bench`.

### Concrete Methods (Overridable)

#### `allocate_default_profile_scratch(size: int, alignment: int, stream) -> torch.Tensor`

Allocates scratch memory for profiling when no explicit profile allocator is installed. Implemented by `GPUDriver` (see below). The allocation respects the given `stream` -- if a stream is provided, the tensor is allocated within that stream's context and `record_stream` is called to prevent premature deallocation.

---

## 6. GPUDriver Concrete Class

**File:** `triton/backends/driver.py`

```python
class GPUDriver(DriverBase):
    def __init__(self):
        import torch
        self.get_device_capability = torch.cuda.get_device_capability
        try:
            from torch._C import _cuda_getCurrentRawStream
            self.get_current_stream = _cuda_getCurrentRawStream
        except ImportError:
            self.get_current_stream = lambda idx: torch.cuda.current_stream(idx).cuda_stream
        self.get_current_device = torch.cuda.current_device
        self.set_current_device = torch.cuda.set_device
```

`GPUDriver` is an intermediate class between `DriverBase` and the concrete `CudaDriver`/`HIPDriver`. It provides common PyTorch-based implementations for device management when PyTorch is available.

### Attributes Set in Constructor

| Attribute | Source | Description |
|-----------|--------|-------------|
| `get_device_capability` | `torch.cuda.get_device_capability` | Returns `(major, minor)` tuple for the given device. |
| `get_current_stream` | `torch._C._cuda_getCurrentRawStream` or fallback | Returns the raw CUDA stream pointer for the current device. |
| `get_current_device` | `torch.cuda.current_device` | Returns the current CUDA device index. |
| `set_current_device` | `torch.cuda.set_device` | Sets the active CUDA device by index. |

### Methods

#### `assemble_tensormap_to_arg(tensormaps_info, args) -> args`

A placeholder method for TMA (Tensor Memory Accelerator) tensor map assembly. Currently returns `args` unchanged. Marked as TODO to be removed once TMA is cleaned up.

#### `allocate_default_profile_scratch(size, alignment, stream) -> torch.Tensor`

Allocates profile scratch memory on the GPU:

1. If `stream` is `None`: Allocates directly with `torch.zeros(size, dtype=torch.int8, device=device)`.
2. If `stream` is `0`: Allocates on the default stream by wrapping it with `device_interface.default_stream()`.
3. If `stream` is a valid stream pointer: Creates an `ExternalStream` from the raw pointer, allocates within that stream context, and calls `scratch.record_stream(launch_stream)` to prevent the tensor from being freed before the kernel completes.

### Benchmark Helpers

Both `CudaDriver` and `HIPDriver` implement:

#### `get_empty_cache_for_benchmark() -> torch.Tensor`

Returns a 256 MB `torch.int` tensor on the CUDA device. Used to flush the L2 cache before benchmarking runs.

#### `clear_cache(cache)`

Zeros out the cache buffer with `cache.zero_()`.

---

## 7. Backend Discovery and Registration

**File:** `triton/backends/__init__.py`

### The Backend Dataclass

```python
@dataclass(frozen=True)
class Backend:
    compiler: Type[BaseBackend]
    driver: Type[DriverBase]
```

A simple frozen dataclass that pairs a compiler backend class with a driver backend class under a single name.

### `_find_concrete_subclasses(module, base_class) -> Type[T]`

A utility function that searches a given Python module for exactly one concrete (non-abstract) subclass of `base_class`:

1. Iterates over all attributes of `module`.
2. Filters for types that are subclasses of `base_class` and are not abstract (`inspect.isabstract()`).
3. Raises `RuntimeError` if zero or more than one concrete subclass is found.
4. Returns the single concrete subclass.

This ensures each backend module (`compiler.py` / `driver.py`) exports exactly one concrete backend class.

### `_discover_backends() -> dict[str, Backend]`

This function is called once at module import time and discovers all available backends. It has two modes:

#### Mode 1: In-Tree Only (Fast Path)

Activated when `TRITON_BACKENDS_IN_TREE=1` is set in the environment.

```python
skip_entrypoints_env = os.environ.get("TRITON_BACKENDS_IN_TREE", "")
if skip_entrypoints_env == "1":
    root = os.path.dirname(__file__)
    for name in os.listdir(root):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        if name.startswith('__'):
            continue
        compiler = importlib.import_module(f"triton.backends.{name}.compiler")
        driver = importlib.import_module(f"triton.backends.{name}.driver")
        backends[name] = Backend(_find_concrete_subclasses(compiler, BaseBackend),
                                 _find_concrete_subclasses(driver, DriverBase))
    return backends
```

Scans the `triton/backends/` directory for subdirectories (skipping `__pycache__` etc.), imports `compiler.py` and `driver.py` from each, and registers the found concrete classes.

#### Mode 2: Entry Point Discovery (Default)

The default path uses Python's `importlib.metadata.entry_points()` to discover backends registered under the `"triton.backends"` entry point group:

```python
for ep in entry_points().select(group="triton.backends"):
    compiler = importlib.import_module(f"{ep.value}.compiler")
    driver = importlib.import_module(f"{ep.value}.driver")
    backends[ep.name] = Backend(
        _find_concrete_subclasses(compiler, BaseBackend),
        _find_concrete_subclasses(driver, DriverBase)
    )
```

Entry points are registered during package installation in `setup.py`:

```python
entry_points["triton.backends"] = [
    f"{b.name} = triton.backends.{b.name}" for b in backends
]
```

For the standard Triton installation, this produces:
- `nvidia = triton.backends.nvidia`
- `amd = triton.backends.amd`

The entry point value (`ep.value`) is the module path (e.g., `"triton.backends.nvidia"`). The entry point name (`ep.name`) is the backend name used at runtime (e.g., `"nvidia"`).

### Module-Level Singleton

```python
backends: dict[str, Backend] = _discover_backends()
```

The discovery result is stored as a module-level `dict[str, Backend]` that is imported throughout the runtime:
- `triton.runtime.driver` imports `backends` for driver selection.
- `triton.tools.compile` imports `triton.backends` for CLI compilation.

---

## 8. NVIDIA Backend (CUDABackend / CudaDriver)

### CUDABackend

**File:** `third_party/nvidia/backend/compiler.py`

```python
class CUDABackend(BaseBackend):
    instrumentation = None
```

#### Class Attributes

| Attribute | Value | Description |
|-----------|-------|-------------|
| `instrumentation` | `None` | Set to an instrumentation object (e.g., GSan) at runtime. If not `None`, its `load_dialects()` and `patch()` methods are called during compilation. |
| `supports_native_tensor_specialization` | `True` (inherited) | NVIDIA supports native tensor specialization via TMA descriptors. |

#### Constructor

```python
def __init__(self, target: GPUTarget) -> None:
    super().__init__(target)
    self.binary_ext = "cubin"
```

Sets `binary_ext` to `"cubin"`, indicating the final binary format is NVIDIA cubin.

#### `supports_target(target)` -> Returns `target.backend == 'cuda'`

#### `parse_options(opts) -> CUDAOptions`

Creates a `CUDAOptions` dataclass from the user-provided dictionary. Performs the following:

1. **Instrumentation handling**: If `instrumentation_mode` contains `"consan"` or `"iisan"`, forces `debug=True` and `sanitize_overflow=False`.
2. **Architecture**: Defaults to `f"sm{self.target.arch}"`, overridden by `knobs.runtime.override_arch`.
3. **num_ctas validation**: `num_ctas > 1` requires SM90+ (compute capability >= 90), raises `ValueError` otherwise.
4. **FP8 dtype support**: Defaults to `("fp8e5", "fp8e4b15")`. Adds `"fp8e4nv"` for SM89+.
5. **Deprecated FP8 dot operand dtypes**: For SM90+, deprecates `"fp8e4b15"`.
6. **FP fusion**: Defaults to `knobs.language.default_fp_fusion`.
7. **max_num_imprecise_acc_default**: Set to `2**30` for SM90, `0` otherwise.

#### CUDAOptions Dataclass

All fields with their defaults:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_warps` | `int` | `4` | Number of warps per CTA. Must be a power of 2. |
| `num_ctas` | `int` | `1` | Number of CTAs (cooperative kernel launch). Requires SM90+. |
| `num_stages` | `int` | `3` | Number of pipeline stages for software pipelining. |
| `warp_size` | `int` | `32` | Warp size (always 32 for NVIDIA). |
| `maxnreg` | `Optional[int]` | `None` | Maximum number of 32-bit registers per thread (maps to PTX `.maxnreg`). |
| `ptx_version` | `int` | `None` | Target PTX version. Auto-detected from ptxas version if not specified. |
| `ptx_options` | `Optional[str]` | `knobs.nvidia.ptxas_options` | Additional ptxas command-line options. |
| `ir_override` | `Optional[str]` | `None` | Filename of user-defined IR to override compilation stage output. |
| `enable_fp_fusion` | `bool` | `True` | Enable fused multiply-add for floating-point operations. |
| `enable_reflect_ftz` | `bool` | `True` | Enable flush-to-zero reflection in libdevice functions. |
| `launch_cooperative_grid` | `bool` | `False` | Use cooperative grid launch. |
| `launch_pdl` | `bool` | `False` | Enable Programmatic Dependent Launch (PDL). |
| `supported_fp8_dtypes` | `Tuple[str]` | `("fp8e5", "fp8e4b15")` | Supported FP8 data types. |
| `deprecated_fp8_dot_operand_dtypes` | `Tuple[str]` | `()` | Deprecated FP8 dot operand types. |
| `default_dot_input_precision` | `str` | `"tf32"` | Default precision for dot product inputs. |
| `allowed_dot_input_precisions` | `Tuple[str]` | `("tf32", "tf32x3", "ieee", "bf16x3", "bf16x6")` | Allowed precisions. |
| `max_num_imprecise_acc_default` | `bool` | `None` | Architecture-dependent default for imprecise accumulation. |
| `extern_libs` | `dict` | `None` | External libraries to link. Defaults to `libdevice.10.bc`. |
| `debug` | `bool` | `False` | Emit debug information. |
| `backend_name` | `str` | `'cuda'` | Backend name identifier. |
| `sanitize_overflow` | `bool` | `True` | Enable overflow sanitization. |
| `arch` | `str` | `None` | Target architecture string (e.g., `"sm90"`). |
| `instrumentation_mode` | `str` | `""` | Instrumentation mode string (e.g., `"gsan"`, `"fpsan"`, `"consan"`). |

The `__post_init__` method:
- Resolves the default `libdevice` path if `extern_libs` does not contain it.
- Injects the GSan runtime library if `"gsan"` is in `instrumentation_mode`.
- Validates `num_warps` is a power of 2.

The `hash()` method creates a SHA-256 hash of all option values (with file paths hashed by content) for caching.

#### Compilation Methods

##### `make_ttir(mod, metadata, opt, capability)` (static)

Runs the Triton IR generation pipeline:

1. `passes.common.add_inliner` -- Inline function calls.
2. `passes.ttir.add_rewrite_tensor_descriptor_to_pointer` -- (SM < 90 only) Lower tensor descriptors to raw pointers.
3. `passes.common.add_canonicalizer` -- Canonicalize IR.
4. `passes.ttir.add_combine` -- Combine redundant operations.
5. `passes.ttir.add_reorder_broadcast` -- Optimize broadcast ordering.
6. `passes.common.add_cse` -- Common subexpression elimination.
7. `passes.common.add_symbol_dce` -- Dead code elimination.
8. `passes.ttir.add_loop_unroll` -- Unroll loops.

##### `make_ttgir(mod, metadata, opt, capability)` (static)

Converts TTIR to Triton GPU IR. This is the most architecture-sensitive pass and varies significantly by capability:

**All architectures:**
- Convert TTIR to TTGIR with target string `f"cuda:{capability}"`.
- Coalesce memory operations.
- Remove layout conversions.
- Optimize thread locality.
- Accelerate matmul (lower to tensor cores).
- Optimize dot operands.

**SM80 (capability // 10 == 8):**
- Fuse nested loops, triton LICM.
- Prefetch optimization.
- Pipeline with `num_stages`.

**SM90 (capability // 10 == 9):**
- Fuse nested loops, triton LICM.
- Hopper warp specialization (`add_hopper_warpspec`).
- Pipeline with `num_stages`.
- TMA lowering.

**SM100+ (capability // 10 >= 10):**
- Fuse nested loops, triton LICM.
- Optimize accumulator init, hoist TMEM allocations.
- Promote LHS to TMEM (tensor memory).
- Warp specialize, pipeline.
- Optimize partition warps.
- Remove TMEM tokens.

**All architectures (tail):**
- Lower MMA operations.
- Reduce data duplication.
- Reorder instructions.
- Fence insertion.
- CSE and symbol DCE.

Also extracts `tensordesc_meta` from the module for TMA descriptor handling.

##### `gluon_to_ttgir(src, metadata, options, capability)`

Dedicated pipeline for Gluon language input:
1. Inliner, infer coalesced encodings, resolve auto encodings.
2. TMA lowering, canonicalizer, SCCP, CSE.
3. Combine tensor select and if.

##### `make_llir(src, metadata, options, capability)`

Converts TTGIR to LLVM IR. This is a complex multi-step process:

1. **GSan**: If enabled, inserts global sanitizer passes before shared memory allocation.
2. **Allocate warp groups**: Sets up warp group structures.
3. **SCF to CF**: Converts structured control flow to control flow dialect.
4. **Shared memory allocation**: NVIDIA-specific shared memory allocation.
5. **Tensor memory allocation**: Allocates tensor memory (for SM100+).
6. **Instrumentation hook**: If `CUDABackend.instrumentation` is set, patches the pipeline.
7. **To LLVM IR**: Converts GPU IR to LLVM IR dialect.
8. **NVVM to LLVM**: Lowers NVVM intrinsics.
9. **Debug info**: Adds line info and variable info if enabled.

After the MLIR pipeline, the LLVM module is created and configured:
- Triple: `nvptx64-nvidia-cuda`
- Processor: `sm_{capability}` (with `"a"` suffix for SM90+)
- Features: `+ptx{version}` (capped at 90 for LLVM compatibility)
- Links external libraries (libdevice, GSan)
- Optimizes at O3 level

Extracts metadata: `num_warps`, `shared`, `tmem_size`, `global_scratch_size/align`, `profile_scratch_size/align`.

##### `make_ptx(src, metadata, opt, capability)`

Translates LLVM IR to PTX assembly using LLVM's backend:
- Target triple: `nvptx64-nvidia-cuda`
- Flags: `["nvptx-mad-wide-opt"]`
- Extracts kernel name from `.visible .entry` directives.
- Adjusts PTX version and target directives in the output.
- Optionally dumps NVPTX if `knobs.nvidia.dump_nvptx` is set.

##### `make_cubin(src, metadata, opt, capability)`

Assembles PTX to cubin using the `ptxas` tool:

1. Writes PTX to a temporary file.
2. Invokes `ptxas` with appropriate flags:
   - Debug info: `-lineinfo` (default), `-g` (if `disable_ptxas_opt`), or `-lineinfo -suppress-debug-info` (if `disable_line_info`).
   - FP fusion: `--fmad=false` if `enable_fp_fusion` is False.
   - Optimization: `--opt-level 0` if `disable_ptxas_opt`.
   - Extra options from `ptx_options`.
   - Register allocation: `--regAllocOptLevel=2` (workaround for ptxas 13.x bug).
   - Target: `--gpu-name=sm_{capability}`.
3. Reads the output `.o` file as raw bytes.
4. Handles errors: raises `PTXASError` with detailed diagnostics on failure.

#### PTX Version Utilities

- `get_ptxas(arch)` -- Returns the `NvidiaTool` for ptxas (Blackwell variant for arch >= 100).
- `get_ptxas_version(arch)` -- Returns the ptxas version string (cached).
- `ptx_get_version(cuda_version)` -- Maps CUDA version string to PTX version number.
- `get_features(options, arch)` -- Returns LLVM feature string (e.g., `"+ptx88"`), capped at PTX 90.

#### `get_codegen_implementation(options)`

Returns a dict with:
- `"convert_custom_types"`: Architecture-dependent FP8 conversion (SM80+ vs SM70).
- `"min_dot_size"`: Architecture-dependent minimum dot size for tensor cores.

#### `pack_metadata(metadata)`

Returns a tuple: `(num_warps, num_ctas, shared)`.

### CudaDriver

**File:** `third_party/nvidia/backend/driver.py`

```python
class CudaDriver(GPUDriver):
    def __init__(self):
        self.utils = CudaUtils()
        self.launcher_cls = CudaLauncher
        if sys.modules.get("torch") is not None:
            super().__init__()
        else:
            self.get_device_capability = self._get_device_capability
            self.get_current_stream = self._get_current_stream
            self.get_current_device = self._get_current_device
            self.set_current_device = self._set_current_device
```

The constructor handles two cases:
1. **PyTorch available**: Calls `super().__init__()` to use PyTorch's device management functions.
2. **PyTorch not available**: Falls back to C++ utility functions from `CudaUtils` for device management.

#### CudaUtils (Singleton)

Compiles a C extension module from `driver.c` at initialization time. Provides:

| Method | Description |
|--------|-------------|
| `load_binary(name, binary, shared_size)` | Loads a cubin binary onto the GPU. |
| `unload_module(module)` | Unloads a GPU module. |
| `get_current_device()` | Gets the current CUDA device index. |
| `set_current_device(device)` | Sets the current CUDA device. |
| `get_default_stream(device)` | Gets the default stream for a device. |
| `get_device_capability(device)` | Returns `(major, minor)` compute capability. |
| `get_device_properties(device)` | Returns device property dict. |
| `cuOccupancyMaxActiveClusters(...)` | Query max active clusters. |
| `set_printf_fifo_size(size)` | Set printf FIFO buffer size. |
| `fill_tma_descriptor_tiled(...)` | Fill a CUtensorMap for tiled TMA access. |
| `fill_tma_descriptor_im2col(...)` | Fill a CUtensorMap for im2col TMA access. |
| `launch(...)` | Launch a CUDA kernel. |
| `build_signature_metadata(sig)` | Build C-side signature metadata. |

Also exposes Python types from the C extension:
- `PyCUtensorMap` -- Python wrapper for CUtensorMap.
- `PyKernelArg` -- Python wrapper for annotated kernel arguments.
- `ARG_CONSTEXPR`, `ARG_KERNEL`, `ARG_TUPLE` -- Argument type constants.

#### CudaLauncher

The kernel launcher for NVIDIA. Manages:

1. **Signature processing**: Flattens nested tuple signatures, removes `constexpr` args, expands tensor descriptor types.
2. **Argument annotation**: Creates `PyKernelArg` objects with type annotations (KERNEL, CONSTEXPR, TUPLE).
3. **Tensor descriptor wrapping**: Wraps the launch function with `wrap_handle_tensordesc` to handle TMA descriptors.
4. **GSan support**: If GSan instrumentation is enabled, appends a per-device state pointer to kernel arguments.
5. **Scratch allocation**: Allocates global scratch and profile scratch per-grid, per-CTA.

The `__call__` method signature:
```python
def __call__(self, gridX, gridY, gridZ, stream, function,
             kernel_metadata, launch_metadata,
             launch_enter_hook, launch_exit_hook, *args)
```

Launches with parameters: grid dimensions, stream, function pointer, cooperative grid flag, PDL flag, metadata, hooks, scratch buffers, and annotated arguments.

#### `libcuda_dirs()` (cached)

Finds `libcuda.so.1` by:
1. Checking `knobs.nvidia.libcuda_path`.
2. Running `/sbin/ldconfig -p` and parsing output.
3. Checking `LD_LIBRARY_PATH`.
4. Asserting the file exists.

#### `_cuda_driver_is_active()`

Determines if the CUDA driver is active by:
1. Loading `libcuda.so.1`.
2. Calling `cuInit(0)`.
3. Checking `cuDeviceGetCount()` returns > 0.

---

## 9. AMD Backend (HIPBackend / HIPDriver)

### HIPBackend

**File:** `third_party/amd/backend/compiler.py`

```python
class HIPBackend(BaseBackend):
    instrumentation = None
    supports_native_tensor_specialization = False
```

#### Key Differences from CUDABackend

| Aspect | CUDABackend | HIPBackend |
|--------|-------------|------------|
| Native tensor specialization | Supported | Not supported (`False`) |
| Binary format | cubin | hsaco |
| Architecture type | Integer (compute capability) | String (e.g., `"gfx942"`) |
| Tensor descriptor lowering | TMA (CUtensorMap) | TDM (TDMDescriptor) or decomposed |
| Buffer operations | N/A | Supported via `pointer_range` attribute |

#### Constructor

```python
def __init__(self, target: GPUTarget) -> None:
    super().__init__(target)
    assert isinstance(target.arch, str)
    self.binary_ext = "hsaco"
```

Asserts that `arch` is a string (unlike NVIDIA where it is an integer).

#### `parse_options(opts) -> HIPOptions`

Similar pattern to CUDA but with AMD-specific options:

1. **ConSan instrumentation**: Forces `debug=True`, `sanitize_overflow=False`.
2. **Multi-CTA validation**: Checks `amd.supports_multi_cta_launch(arch)`.
3. **TF32 for CDNA3**: Adds `"tf32"` to allowed dot input precisions for `gfx942`.
4. **FP8 support**: Defaults include `"fp8e4nv"`, `"fp8e5"`, `"fp8e5b16"`, `"fp8e4b8"`.
5. **Deprecated FP8 on gfx950**: Deprecates `"fp8e5b16"` and `"fp8e4b8"`.

#### HIPOptions Dataclass

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_warps` | `int` | `4` | Number of warps (wavefront groups). |
| `waves_per_eu` | `int` | `0` | Target waves per execution unit. |
| `num_stages` | `int` | `2` | Pipeline stages. |
| `num_ctas` | `int` | `1` | Number of CTAs. |
| `extern_libs` | `dict` | `None` | External libraries (defaults to `ocml.bc`, `ockl.bc`). |
| `debug` | `bool` | `False` | Debug mode. |
| `sanitize_overflow` | `bool` | `True` | Overflow sanitization. |
| `arch` | `str` | `None` | Target architecture (e.g., `"gfx942"`). |
| `supported_fp8_dtypes` | `Tuple[str]` | `("fp8e4nv", "fp8e5", "fp8e5b16", "fp8e4b8")` | Supported FP8 types. |
| `deprecated_fp8_dot_operand_dtypes` | `Tuple[str]` | `()` | Deprecated FP8 types. |
| `default_dot_input_precision` | `str` | `"ieee"` | Default dot precision (IEEE for AMD). |
| `allowed_dot_input_precisions` | `Tuple[str]` | `("ieee", "bf16x3", "bf16x6")` | Allowed precisions. |
| `enable_fp_fusion` | `bool` | `True` | FP fusion. |
| `launch_cooperative_grid` | `bool` | `False` | Cooperative grid launch. |
| `matrix_instr_nonkdim` | `int` | `0` | Non-K dimension for matrix instructions. |
| `kpack` | `int` | `1` | K-pack value for MFMA instructions. Deprecated on gfx950. |
| `allow_flush_denorm` | `bool` | `False` | Allow flushing denormals to zero. |
| `max_num_imprecise_acc_default` | `int` | `0` | Max imprecise accumulation default. |
| `backend_name` | `str` | `'hip'` | Backend name. |
| `instrumentation_mode` | `str` | `""` | Instrumentation mode. |
| `schedule_hint` | `str` | `'none'` | Instruction scheduling hints (`"attention"`, `"memory-bound-attention"`). |

`__post_init__`:
- Computes `warp_size` (32 for gfx10+, 64 for gfx9xx).
- Validates `num_warps` is a power of 2.
- Warns about deprecated `kpack` on gfx950 (overwrites to 1).
- Adds `ocml.bc` and `ockl.bc` to `extern_libs`.

#### `parse_attr(desc)` (static override)

Extends `BaseBackend.parse_attr` to also check for `"S"` (pointer range for buffer operations):

```python
ret = BaseBackend.parse_attr(desc)
if "S" in desc:
    ret += [["tt.pointer_range", 32]]
return ret
```

The `"S"` attribute enables AMD buffer operations for pointers within a 2GB range.

#### `get_tensor_specialization(arg, **kwargs)` (static override)

Extends the base implementation to add pointer range specialization:

```python
ret = BaseBackend.get_tensor_specialization(arg, **kwargs)
if knobs.amd.use_buffer_ops and HIPBackend.is_within_2gb(arg):
    ret += "S"
return ret
```

#### `is_within_2gb(arg)` (static)

Checks whether a tensor's storage is within 2GB (needed for AMD buffer operations):
- If `arg` has `ptr_range()`, checks `arg.ptr_range() <= 2**31 - 1`.
- If `torch` is available and `arg` is a `torch.Tensor`, checks `arg.untyped_storage().size() <= 2**31 - 1`.

#### Compilation Methods

##### `make_ttir(mod, metadata, options)` (static)

Similar to NVIDIA but with AMD-specific differences:
1. Inline, canonicalize, combine, reorder broadcast, CSE, LICM, loop unroll.
2. `add_rewrite_tensor_descriptor_to_pointer` only if TDM is not supported for the arch.

##### `make_ttgir(mod, metadata, options)` (static)

AMD-specific TTGIR pipeline:
1. Convert to TTGIR with `f"hip:{options.arch}"`.
2. Coalesce, F32 dot TC, remove layout conversions.
3. **AMD-specific matmul acceleration**: `add_accelerate_matmul` with arch, `matrix_instr_nonkdim`, `kpack`.
4. Optimize epilogue, optimize dot operands, hoist/sink layout conversions.
5. Fuse nested loops, LICM, canonicalize.
6. **Scheduling**: Ping-pong scheduling for gfx942/gfx950, async copy for gfx950/gfx1250.
7. Pipeline, coalesce async copy, convert to tensor ops.
8. **Instruction scheduling hints**: `"attention"` or `"memory-bound-attention"`.
9. **In-thread transpose**: For gfx942 and gfx120x.
10. **Buffer operations**: If `knobs.amd.use_buffer_ops`, canonicalize pointers and convert to buffer ops.
11. **FP sanitizer**: For supported architectures.

##### `make_llir(src, metadata, options)` (static)

LLVM IR generation for AMD:
1. Update async wait count, warp pipeline conversion.
2. SCF to CF, index to LLVM IR.
3. **ConSan**: Prepare captures and run concurrency sanitizer (gfx1250 only).
4. Allocate shared memory, global scratch memory.
5. Convert to LLVM IR with AMD-specific passes.
6. Warp specialize to LLVM.
7. CF/Arith to LLVM IR.
8. **Instruction scheduling hint lowering**.
9. Debug info.

After MLIR pipeline, configures the LLVM module:
- Triple: `amd.TARGET_TRIPLE` (AMDGPU target triple).
- Features: `+xnack` if ASan enabled, `-real-true16` for gfx11.
- Sets ISA version, ABI version (500).
- Sets control constants: `__oclc_finite_only_opt`, `__oclc_correctly_rounded_sqrt32`, `__oclc_unsafe_math_opt`, `__oclc_wavefrontsize64`.
- Sets kernel attributes: `amdgpu-flat-work-group-size`, `amdgpu-waves-per-eu`, `denormal-fp-math-f32`, `uniform-work-group-size`.
- Links external libraries (ASan runtime, ocml, ockl).
- Optimizes at O3 level with AMD-specific options.

##### `make_amdgcn(src, metadata, options)` (static)

Translates LLVM IR to AMDGCN assembly:
- Extracts kernel name from `define amdgpu_kernel void @...`.
- Translates to MIR and assembly using LLVM.
- Optionally swaps MIR for custom scheduling.
- Dumps AMDGCN if `knobs.amd.dump_amdgcn` is set.

##### `make_hsaco(src, metadata, options)` (static)

Assembles AMDGCN to HSACO binary:
- Assembles AMDGCN with `amd.assemble_amdgcn`.
- Links HSACO with `amd.link_hsaco`.
- Returns the linked binary as raw bytes.

#### `get_codegen_implementation(options)`

Returns `{"min_dot_size": get_min_dot_size(self.target)}`. The AMD min dot size always returns `(1, 1, 1)`, falling back to FMA and casting for unsupported configurations.

#### `pack_metadata(metadata)`

Returns `(num_warps, num_ctas, shared)`.

### HIPDriver

**File:** `third_party/amd/backend/driver.py`

```python
class HIPDriver(GPUDriver):
    def __init__(self):
        super().__init__()
        self.utils = HIPUtils()
        self.launcher_cls = HIPLauncher
```

#### HIPUtils (Singleton)

Similar to `CudaUtils` but for HIP runtime. Compiles `driver.c` with the HIP runtime dynamic library path:

| Method | Description |
|--------|-------------|
| `load_binary` | Load HSACO binary onto the GPU. |
| `unload_module` | Unload a GPU module. |
| `get_device_properties` | Query device properties. |
| `create_tdm_descriptor` | Create a TDM (Tensor Descriptor Map) descriptor. |
| `launch` | Launch a HIP kernel. |
| `build_signature_metadata` | Build C-side signature metadata. |

Exposes Python types:
- `PyTDMDescriptor` -- Python wrapper for TDM descriptors.
- `PyKernelArg` -- Annotated kernel arguments.
- `ARG_CONSTEXPR`, `ARG_KERNEL`, `ARG_TUPLE` -- Argument type constants.

#### `_get_path_to_hip_runtime_dylib()` (cached)

Extensive search for `libamdhip64.so`:
1. `knobs.amd.libhip_path` if set.
2. Already mmapped library via `dl_iterate_phdr` (Linux only).
3. Local backend `lib/` directory.
4. PyTorch's `torch/lib/` directory.
5. `LD_LIBRARY_PATH` directories.
6. `HIP_PATH` environment variable.
7. `hipconfig --path` output.
8. `ROCM_PATH` environment variable.
9. `/sbin/ldconfig -p` output.
10. Common path `/opt/rocm/lib/`.

#### HIPLauncher

Similar to `CudaLauncher` with AMD-specific differences:

1. No GSan support.
2. Cooperative grid launch with device support check (`cooperativeLaunch` property).
3. Scratch allocation without per-CTA multiplier.
4. Tensor descriptor handling uses `TDMDescriptor` instead of `CUtensorMap`.

#### `get_current_target()`

```python
arch = knobs.runtime.override_arch or device_properties['arch']
warp_size = device_properties['warpSize']
return GPUTarget("hip", arch.split(':')[0], warp_size)
```

Note: The arch string from device properties may contain a colon suffix (e.g., `"gfx942:sramecc+:xnack-"`) which is stripped.

#### `get_active_torch_device()`

Returns `torch.device("cuda", device_index)` -- PyTorch uses `"cuda"` as the device string even for HIP devices.

#### `is_active()`

Checks `torch.cuda.is_available() and torch.version.hip is not None`.

---

## 10. Tensor Descriptor Handling

**File:** `triton/backends/driver.py` (base utilities)
**Files:** `third_party/nvidia/backend/driver.py` and `third_party/amd/backend/driver.py` (backend-specific)

Tensor descriptors are a mechanism for passing structured tensor information (base pointer, shape, strides, padding) to kernels. They are used for TMA (Tensor Memory Accelerator) on NVIDIA and TDM (Tensor Descriptor Map) on AMD.

### `decompose_descriptor(arg)`

**File:** `triton/backends/driver.py`

```python
def decompose_descriptor(arg):
    return [arg.base, *arg.shape, *arg.strides,
            arg.padding == "nan",
            arg.round_f32_to_tf32,
            *arg.shape, *arg.strides]
```

Decomposes a tensor descriptor into its constituent scalar arguments:
1. `arg.base` -- Base pointer (data_ptr).
2. `*arg.shape` -- Shape values (one per dimension).
3. `*arg.strides` -- Stride values (one per dimension).
4. `arg.padding == "nan"` -- Boolean flag for NaN padding.
5. `arg.round_f32_to_tf32` -- Boolean flag for TF32 rounding.
6. `*arg.shape` -- Shape values again (duplicated for post-lowering use).
7. `*arg.strides` -- Stride values again (duplicated for post-lowering use).

This is the fallback path when native tensor descriptor support is not available.

### `wrap_handle_tensordesc_impl(launcher, signature, tensordesc_meta, make_tensordesc_arg)`

**File:** `triton/backends/driver.py`

The generic implementation for wrapping a launcher to handle tensor descriptor arguments. It:

1. **Identifies tensor descriptor positions** in the signature using `_is_descriptor()` (checks for `"tensordesc"` prefix).
2. **Builds a path tree** for fast lookup of descriptor positions in nested tuple signatures:
   ```python
   # For signature ('tensordesc', 'i32', ('i32', 'tensordesc'))
   # relevant_paths = {0: {}, 2: {1: {}}}
   ```
3. **Returns a wrapper function** that:
   - Separates base args and kernel args.
   - Calls `make_tensordesc_args` (C++ function) to expand tensor descriptor arguments.
   - Calls the original launcher with expanded arguments.

If no tensor descriptors are found in the signature, returns the launcher unchanged.

### `_parse_descriptor(descriptor) -> (dtype, ndim)`

Parses a tensor descriptor type string like `"tensordesc<fp32[32,32],input_rank=2>"` to extract:
- `dtype`: The element data type (e.g., `"fp32"`).
- `ndim`: The number of dimensions (from block shape or `input_rank=`).

### `_expand_descriptor(descriptor, has_tensordesc_meta, descriptor_type) -> list`

Expands a single descriptor type into its component types for signature building:

**Without tensordesc metadata** (decomposed path):
```
["*dtype", "i64", "i64", ..., "i1", "i1", "i32", ..., "i64", ...]
   base    shape(ndim)  strides(ndim) pad  round shape(ndim) strides(ndim)
```

**With tensordesc metadata** (native descriptor path):
```
[descriptor_type, "i32", ..., "i64", ...]
  descriptor     shape(ndim) strides(ndim)
```

Where `descriptor_type` is backend-specific:
- NVIDIA: `"nvTmaDesc"` (maps to `CUtensorMap`)
- AMD: `"tensordesc"` (maps to `TDMDescriptor`)

### `expand_signature(signature, tensordesc_meta, descriptor_type) -> list`

Recursively expands an entire signature, replacing tensor descriptor types with their expanded component types. Handles nested tuples by recursing into them.

```python
def expand_signature(signature, tensordesc_meta, descriptor_type):
    has_tensordesc_meta = bool(tensordesc_meta)
    result = []
    for s in signature:
        visit(s, result)
    return result
```

### NVIDIA Tensor Descriptor (TMA)

**File:** `third_party/nvidia/backend/driver.py`

#### `make_tensordesc_arg(arg, metadata, _)` for NVIDIA

Creates `CUtensorMap` objects for TMA access:

1. If `metadata` is `None`, falls back to `decompose_descriptor(arg)`.
2. Otherwise, fills a TMA descriptor:
   - **Tiled mode**: Standard 2D/ND tiled access pattern.
   - **Im2col mode**: Convolution-style im2col access pattern with pixel box, element strides.
3. Handles FP4 padding (doubles the last dimension).
4. Maps device TMA dtype enum to host enum (swaps 8, 9, 10).
5. Handles TF32 rounding override.

Returns `[cu_tensor_map, *shape, *strides]`.

### AMD Tensor Descriptor (TDM)

**File:** `third_party/amd/backend/driver.py`

#### `make_tensordesc_arg(arg, tensordesc_metadata, base_args)` for AMD

Creates `TDMDescriptor` objects:

1. If `tensordesc_metadata` is `None`, falls back to `decompose_descriptor(arg)`.
2. Otherwise:
   - Extracts `elem_bits`, `block_size` from metadata.
   - Extracts `interval_padding_pairs` for padding configuration.
   - Gets `num_warps` from kernel metadata.
   - Calls `driver.utils.create_tdm_descriptor(...)` with the parameters.

Returns `[desc, *shape, *strides]`.

---

## 11. Compilation Stages

### Stage Registration via `add_stages()`

Each backend's `add_stages()` method populates the `stages` dict with named compilation passes. The stages run sequentially and each receives the output of the previous stage plus a shared `metadata` dict.

#### NVIDIA Stages (Language == TRITON)

```
Input (Triton frontend IR)
    |
    v
[ttir]  -->  make_ttir()
    |          Inliner, rewrite tensor descriptors (SM<90),
    |          canonicalizer, combine, reorder broadcast, CSE,
    |          symbol DCE, loop unroll
    v
[ttgir] -->  make_ttgir()
    |          Convert to TTGIR, coalesce, F32 dot TC,
    |          architecture-specific optimizations:
    |          SM80: prefetch, pipeline
    |          SM90: hopper warpspec, pipeline, TMA lowering
    |          SM100+: TMEM, warp specialize, pipeline
    |          Lower MMA, reduce data duplication, fence insertion
    v
[llir]  -->  make_llir()
    |          GSan (if enabled), allocate warp groups,
    |          SCF->CF, allocate shared memory, allocate tensor memory,
    |          to LLVM IR, NVVM->LLVM, debug info,
    |          link extern libs, O3 optimize
    v
[ptx]   -->  make_ptx()
    |          LLVM->PTX via LLVM backend,
    |          adjust PTX version and target directives
    v
[cubin] -->  make_cubin()
    |          PTX->cubin via ptxas,
    |          returns bytes
    v
Executable binary (bytes)
```

#### NVIDIA Stages (Language == GLUON)

```
Input (Gluon IR)
    |
    v
[ttgir] -->  gluon_to_ttgir()
    |          Inliner, infer coalesced encodings,
    |          resolve auto encodings, TMA lowering,
    |          canonicalizer, SCCP, CSE
    v
[llir]  -->  make_llir()
    v
[ptx]   -->  make_ptx()
    v
[cubin] -->  make_cubin()
    v
Executable binary (bytes)
```

Note: The Gluon path skips the `ttir` stage entirely.

#### AMD Stages (Language == TRITON)

```
Input (Triton frontend IR)
    |
    v
[ttir]   -->  make_ttir()
    |           Inliner, rewrite tensor descriptors (if no TDM),
    |           canonicalizer, combine, reorder broadcast, CSE,
    |           LICM, symbol DCE, loop unroll
    v
[ttgir]  -->  make_ttgir()
    |           Convert to TTGIR, coalesce, F32 dot TC,
    |           AMD matmul acceleration, optimize epilogue,
    |           optimize dot operands, fuse nested loops,
    |           scheduling (ping-pong, async copy),
    |           pipeline, buffer ops (if enabled),
    |           FP sanitizer (if enabled)
    v
[llir]   -->  make_llir()
    |           Warp pipeline conversion, SCF->CF,
    |           ConSan (gfx1250), allocate shared memory,
    |           allocate global scratch, to LLVM IR,
    |           set kernel attributes, link extern libs, O3 optimize
    v
[amdgcn] -->  make_amdgcn()
    |           LLVM->AMDGCN assembly via LLVM backend,
    |           optional MIR swap for custom scheduling
    v
[hsaco]  -->  make_hsaco()
    |           AMDGCN->HSACO via AMD assembler and linker,
    |           returns bytes
    v
Executable binary (bytes)
```

#### AMD Stages (Language == GLUON)

```
Input (Gluon IR)
    |
    v
[ttgir]  -->  gluon_to_ttgir()
    |           Inliner, resolve auto encodings, SCCP, CSE,
    |           loop unroll, combine tensor select and if,
    |           warp pipeline, allocate warp groups,
    |           FP sanitizer (if enabled)
    v
[llir]   -->  make_llir()
    v
[amdgcn] -->  make_amdgcn()
    v
[hsaco]  -->  make_hsaco()
    v
Executable binary (bytes)
```

### Stage Inspection Hook

Both backends check for `knobs.runtime.add_stages_inspection_hook` at the end of `add_stages()`. If set, this function is called with `(self, stages, options, language, capability)` to allow external code to inspect or modify the compilation stages.

### Metadata Communication Between Stages

Stages communicate through the shared `metadata` dict. Key metadata fields set during compilation:

| Field | Set By | Description |
|-------|--------|-------------|
| `tensordesc_meta` | `make_ttgir` / `gluon_to_ttgir` | Tensor descriptor metadata for TMA/TDM. |
| `name` | `make_ptx` / `make_amdgcn` | Kernel entry point name. |
| `num_warps` | `make_llir` (may override) | Actual number of warps (mutated by warp specialization). |
| `shared` | `make_llir` | Shared memory size in bytes. |
| `tmem_size` | `make_llir` (NVIDIA) | Tensor memory size (SM100+). |
| `global_scratch_size` | `make_llir` | Global scratch memory size. |
| `global_scratch_align` | `make_llir` | Global scratch memory alignment. |
| `profile_scratch_size` | `make_llir` | Profile scratch memory size. |
| `profile_scratch_align` | `make_llir` | Profile scratch memory alignment. |

### Return Types

All stages return `str` (textual IR or assembly) except the final stage which returns `bytes`:
- NVIDIA: `cubin` stage returns raw cubin bytes.
- AMD: `hsaco` stage returns raw HSACO bytes.

---

## 12. External Plugin Backends (TRITON_PLUGIN_DIRS)

### Overview

Triton supports external (out-of-tree) backends through two mechanisms:
1. **TRITON_PLUGIN_DIRS** -- Build-time mechanism for bundling external backends into the Triton package.
2. **Entry points** -- Runtime mechanism for discovering installed backends.

### Build-Time: TRITON_PLUGIN_DIRS

**File:** `setup.py`

The `TRITON_PLUGIN_DIRS` environment variable allows adding external backends at build time. It is a semicolon-separated list of paths to external backend source directories.

#### Directory Structure Required

Each external backend directory must contain:

```
<plugin_dir>/
  backend/
    name.conf         # Contains the backend name (e.g., "my_accelerator")
    compiler.py       # Must contain exactly one concrete subclass of BaseBackend
    driver.py         # Must contain exactly one concrete subclass of DriverBase
    include/          # (optional) C/C++ include directories
    driver.c          # (optional) C driver implementation
  language/           # (optional) Backend-specific language extensions
  tools/              # (optional) Backend-specific tools
```

#### `BackendInstaller.copy_externals()`

```python
@staticmethod
def copy_externals():
    backend_dirs = os.getenv("TRITON_PLUGIN_DIRS")
    if backend_dirs is None:
        return []
    backend_dirs = backend_dirs.strip().split(";")
    backend_names = [
        Path(os.path.join(dir, "backend", "name.conf")).read_text().strip()
        for dir in backend_dirs
    ]
    return [
        BackendInstaller.prepare(backend_name, backend_src_dir=backend_src_dir, is_external=True)
        for backend_name, backend_src_dir in zip(backend_names, backend_dirs)
    ]
```

1. Reads `TRITON_PLUGIN_DIRS` from environment.
2. Splits by semicolons to get individual directory paths.
3. Reads `backend/name.conf` from each directory to get the backend name.
4. Prepares each backend using `BackendInstaller.prepare()` with `is_external=True`.

#### Backend Preparation

`BackendInstaller.prepare(backend_name, backend_src_dir, is_external)`:

1. Validates the backend directory structure.
2. Checks for `compiler.py` and `driver.py` in `backend/`.
3. Checks for optional `language/` and `tools/` directories.
4. Creates a symlink for external backends (not a copy) to the install directory.

#### Package Registration

External backends are integrated into the Triton package:

```python
backends = [*BackendInstaller.copy(["nvidia", "amd"]), *BackendInstaller.copy_externals()]
```

All backends (in-tree and external) are registered as entry points:

```python
entry_points["triton.backends"] = [
    f"{b.name} = triton.backends.{b.name}" for b in backends
]
```

This means an external backend named `"my_accelerator"` would be accessible as `triton.backends.my_accelerator` and discoverable via the `"triton.backends"` entry point group.

### Runtime: Entry Point Discovery

At runtime, `_discover_backends()` uses `importlib.metadata.entry_points()` to find all registered backends in the `"triton.backends"` group. Each entry point maps:

```
name -> module_path
```

For example:
- `nvidia -> triton.backends.nvidia`
- `amd -> triton.backends.amd`
- `my_accelerator -> triton.backends.my_accelerator`

### TRITON_BACKENDS_IN_TREE Optimization

Setting `TRITON_BACKENDS_IN_TREE=1` bypasses entry point discovery (which can be slow) and instead scans the `triton/backends/` directory directly. This is useful for development builds where all backends are installed in-tree.

---

## 13. Runtime Driver Selection

**File:** `triton/runtime/driver.py`

### DriverConfig and Driver Creation

```python
def _create_driver() -> DriverBase:
    selected = os.environ.get("TRITON_DEFAULT_BACKEND", None)
    if selected:
        if selected not in backends:
            raise RuntimeError(f"Unknown backend device '{selected}'. Available backends: {list(backends.keys())}")
        driver = backends[selected].driver
        if not driver.is_active():
            raise RuntimeError(f"Backend device '{selected}' is not active.")
        return driver()
    else:
        active_drivers = [x.driver for x in backends.values() if x.driver.is_active()]
        if len(active_drivers) != 1:
            raise RuntimeError(f"{len(active_drivers)} active drivers ({active_drivers}). There should only be one.")
        return active_drivers[0]()
```

Driver selection follows this logic:

1. **Explicit selection**: If `TRITON_DEFAULT_BACKEND` is set, use that backend's driver. Raises an error if the backend name is unknown or the driver is not active.
2. **Auto-detection**: If no explicit selection, query all backends' `is_active()`. Exactly one must be active, otherwise raises an error.

### `DriverConfig` Class

```python
class DriverConfig:
    def __init__(self):
        self._default: DriverBase | None = None
        self._active: DriverBase | None = None

    @property
    def default(self) -> DriverBase:
        if self._default is None:
            self._default = _create_driver()
        return self._default

    @property
    def active(self) -> DriverBase:
        if self._active is None:
            self._active = self.default
        return self._active

    def set_active(self, driver: DriverBase) -> None:
        self._active = driver

    def reset_active(self) -> None:
        self._active = self.default
```

- `default`: Lazily creates the driver on first access.
- `active`: Defaults to `default`, but can be overridden with `set_active()`.
- `reset_active()`: Resets to the default driver.

### Module-Level Singleton

```python
driver = DriverConfig()
```

Accessed throughout the runtime as `triton.runtime.driver.active` (the currently active driver instance).

### Environment Variables Summary

| Variable | Purpose |
|----------|---------|
| `TRITON_DEFAULT_BACKEND` | Explicitly select the backend at runtime (e.g., `"nvidia"`, `"amd"`). |
| `TRITON_BACKENDS_IN_TREE` | Set to `"1"` to skip entry point discovery and scan `triton/backends/` directly. |
| `TRITON_PLUGIN_DIRS` | Build-time semicolon-separated list of external plugin directories. |

---

## Appendix A: Architecture-Specific Feature Matrix

### NVIDIA

| Feature | SM80 | SM89 | SM90 | SM100+ |
|---------|------|------|------|--------|
| Tensor descriptor rewrite to pointer | Yes | Yes | No | No |
| Prefetch optimization | Yes | Yes | No | No |
| Hopper warp specialization | No | No | Yes | No |
| TMEM (tensor memory) | No | No | No | Yes |
| Promote LHS to TMEM | No | No | No | Yes |
| Cooperative launch (num_ctas > 1) | No | No | Yes | Yes |
| FP8 NV dtype | No | Yes | Yes | Yes |
| TMA lowering | No | No | Yes | Yes |

### AMD

| Feature | gfx942 (MI300X) | gfx950 (MI350) | gfx1100 (RDNA3) | gfx1200+ (RDNA4) | gfx1250 |
|---------|-----------------|----------------|------------------|-------------------|---------|
| Ping-pong scheduling | Yes | Yes (async) | No | No | No |
| Async copy | No | Yes | No | No | Yes |
| In-thread transpose | Yes | No | No | Yes | No |
| Buffer operations | If knob set | If knob set | If knob set | If knob set | If knob set |
| TDM (Tensor Descriptor Map) | No | Yes | No | No | Yes |
| Multi-CTA launch | Yes | Yes | No | No | Yes |
| FP Sanitizer | Yes | Yes | No | No | Yes |
| ConSan | No | No | No | No | Yes |
| TF32 | Yes | No | No | No | No |
| Warp size | 64 | 64 | 32 | 32 | 32 |

---

## Appendix B: Instrumentation Modes

Both backends support various instrumentation modes set via `instrumentation_mode` in options:

| Mode | NVIDIA | AMD | Description |
|------|--------|-----|-------------|
| `gsan` | Yes (SM90+) | No | Global sanitizer. Detects shared memory race conditions. |
| `fpsan` | Yes | Yes (gfx942, gfx950, gfx1250) | Floating-point sanitizer. Tracks NaN/Inf propagation. |
| `consan` | Yes | Yes (gfx1250) | Concurrency sanitizer. Detects data races. |
| `iisan` | Yes | No | Intra-instruction sanitizer. Enables device-side assertions. |

When instrumentation is active:
- Debug mode is forced on.
- Overflow sanitization is disabled.
- Backend-specific IR passes are injected at appropriate points.
- The backend's `instrumentation` class attribute can be set to a plugin object with `load_dialects()` and `patch()` methods.

---

## Appendix C: Complete File Reference

| File | Key Classes/Functions |
|------|-----------------------|
| `triton/backends/__init__.py` | `Backend`, `_find_concrete_subclasses()`, `_discover_backends()`, `backends` dict |
| `triton/backends/compiler.py` | `GPUTarget`, `Language`, `BaseBackend` |
| `triton/backends/driver.py` | `decompose_descriptor()`, `wrap_handle_tensordesc_impl()`, `_parse_descriptor()`, `_expand_descriptor()`, `expand_signature()`, `Benchmarker`, `DriverBase`, `GPUDriver` |
| `third_party/nvidia/backend/compiler.py` | `CUDAOptions`, `CUDABackend`, PTX version utilities |
| `third_party/nvidia/backend/driver.py` | `CudaUtils`, `CudaLauncher`, `CudaDriver`, `ty_to_cpp()`, `make_kernel_signature()`, `annotate_arguments()`, `make_tensordesc_arg()`, `wrap_handle_tensordesc()` |
| `third_party/amd/backend/compiler.py` | `HIPOptions`, `HIPBackend`, arch feature detection functions |
| `third_party/amd/backend/driver.py` | `HIPUtils`, `HIPLauncher`, `HIPDriver`, `ty_to_cpp()`, `make_kernel_signature()`, `annotate_arguments()`, `make_tensordesc_arg()`, `wrap_handle_tensordesc()`, `_get_path_to_hip_runtime_dylib()` |
| `triton/runtime/driver.py` | `_create_driver()`, `DriverConfig`, `driver` singleton |
| `setup.py` | `BackendInstaller`, `get_entry_points()`, `TRITON_PLUGIN_DIRS` handling |
