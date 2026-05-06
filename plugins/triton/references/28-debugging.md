# Triton Debugging and Configuration Reference

This document provides an exhaustive reference for all debugging tools, environment variables, configuration knobs, and diagnostic techniques available in Triton. These tools cover every stage of the compilation pipeline, from frontend AST processing through MLIR lowering to device code generation.

---

## Table of Contents

1. [Environment Variables (Complete Reference)](#1-environment-variables-complete-reference)
2. [Configuration Knobs](#2-configuration-knobs)
3. [Interpreter Mode](#3-interpreter-mode)
4. [IR Dumping](#4-ir-dumping)
5. [Reproducer System](#5-reproducer-system)
6. [Kernel Override](#6-kernel-override)
7. [Floating-Point Sanitizer (FpSan)](#7-floating-point-sanitizer-fpsan)
8. [Debugging Operations](#8-debugging-operations)
9. [Pipeline Inspection Hook](#9-pipeline-inspection-hook)
10. [Address Sanitizer](#10-address-sanitizer)
11. [Concurrency Sanitizer (ConSan)](#11-concurrency-sanitizer-consan)
12. [Third-Party Debugging Tools](#12-third-party-debugging-tools)

---

## 1. Environment Variables (Complete Reference)

Environment variables are the primary mechanism for controlling Triton's debugging and compilation behavior. They are organized into categories: cache-invalidating variables (which change compilation output and invalidate cached binaries), cache-neutral variables (which do not affect compilation output), and backend-specific variables.

### 1.1 MLIR and LLVM Dump Controls

#### `MLIR_ENABLE_DUMP`

Controls whether MLIR dumps IR during the pass pipeline. When enabled, every MLIR pass will print the IR before and after it runs. Output goes to stderr by default, or to the file specified by `MLIR_DUMP_PATH`.

- **Type:** Boolean (accepts `1`, `on`, `true` for enabled; `0`, `off`, `false` for disabled)
- **Default:** Not set (disabled)
- **Cache effect:** Cache-invalidating
- **Example usage:**

```bash
# Dump IR for all passes in all kernels
MLIR_ENABLE_DUMP=1 python my_kernel.py

# Dump IR only for kernels whose name contains "_kernel"
MLIR_ENABLE_DUMP=_kernel python my_kernel.py
```

When set to a string other than a boolean, the string is used as a filter: only passes operating on functions whose name contains the string will dump their IR. This is useful for focusing on a specific kernel in a program with many kernels.

The output includes lines like:
```
// -----// IR Dump Before TritonCombine //----- //
tt.func public @_kernel(...) { ... }
```

You can use this programmatically for test contexts:

```python
import os
from contextlib import contextmanager

@contextmanager
def enable_dump_context(pass_name="1"):
    try:
        os.environ["MLIR_ENABLE_DUMP"] = pass_name
        yield
    finally:
        os.environ["MLIR_ENABLE_DUMP"] = "0"
```

#### `MLIR_DUMP_PATH`

When set, MLIR dump output is written to the specified file path instead of stderr. The file is created (or overwritten) when the first dump occurs. This is useful for capturing large amounts of dump output for later analysis.

- **Type:** String (file path)
- **Default:** Not set (output goes to stderr via `llvm::dbgs()`)
- **Cache effect:** Cache-invalidating
- **Example usage:**

```bash
MLIR_DUMP_PATH=/tmp/triton_ir_dump.mlir MLIR_ENABLE_DUMP=1 python my_kernel.py
```

#### `LLVM_IR_ENABLE_DUMP`

Enables dumping of LLVM IR during the LLVM-level compilation stages (after MLIR-to-LLVM conversion). When enabled, each LLVM pass will print the LLVM IR module before and after transformation.

- **Type:** Boolean
- **Default:** Not set (disabled)
- **Cache effect:** Cache-invalidating
- **Example usage:**

```bash
LLVM_IR_ENABLE_DUMP=1 python my_kernel.py
```

#### `TRITON_ENABLE_LLVM_DEBUG`

Enables LLVM's built-in debug output, which provides verbose information about LLVM internals including pass execution, register allocation, instruction selection, and other low-level compilation details. This produces extremely verbose output and is primarily useful for debugging the Triton compiler itself.

- **Type:** Boolean
- **Default:** Not set (disabled)
- **Cache effect:** Cache-invalidating
- **Example usage:**

```bash
TRITON_ENABLE_LLVM_DEBUG=1 python my_kernel.py
```

#### `TRITON_LLVM_DEBUG_ONLY`

When `TRITON_ENABLE_LLVM_DEBUG` is active, this variable restricts the debug output to only specific LLVM passes or components. The value is a comma-separated list of pass names or debug types.

- **Type:** String (comma-separated pass/component names)
- **Default:** Not set (all debug output shown)
- **Cache effect:** Cache-invalidating
- **Example usage:**

```bash
# Only show debug output for the instruction selection pass
TRITON_ENABLE_LLVM_DEBUG=1 TRITON_LLVM_DEBUG_ONLY=instruction-select python my_kernel.py
```

### 1.2 MLIR Timing and Diagnostics

#### `MLIR_ENABLE_TIMING`

Enables MLIR's timing infrastructure, which measures and reports the time spent in each MLIR pass and pipeline stage. Output is printed to stderr after compilation completes.

- **Type:** Boolean
- **Default:** Not set (disabled)
- **Cache effect:** Cache-invalidating
- **Example usage:**

```bash
MLIR_ENABLE_TIMING=1 python my_kernel.py
```

This produces a hierarchical timing report showing wall time for each pass:
```
===-------------------------------------------------------------------------===
                         ... Pass execution timing report ...
===-------------------------------------------------------------------------===
  Total Execution Time: 0.0234 seconds

  ---User Time---   --System Time--   --User+System--   ---Wall Time---
    0.0012 ( 5.1%)     0.0000 ( 0.0%)     0.0012 ( 4.8%)     0.0013 ( 5.2%)  TritonCombine
    0.0089 (38.0%)     0.0003 (50.0%)     0.0092 (37.4%)     0.0091 (36.4%)  TritonGPUPipeline
    ...
```

#### `LLVM_ENABLE_TIMING`

Similar to `MLIR_ENABLE_TIMING` but for the LLVM-level compilation stages (after MLIR-to-LLVM conversion).

- **Type:** Boolean
- **Default:** Not set (disabled)
- **Cache effect:** Cache-invalidating
- **Example usage:**

```bash
LLVM_ENABLE_TIMING=1 python my_kernel.py
```

#### `MLIR_ENABLE_DIAGNOSTICS`

Enables MLIR diagnostic output during compilation. This controls whether the MLIR context emits diagnostics such as remarks, warnings, and notes from compiler passes. The value can be `remarks` for remark-level diagnostics, or `remarks,operations` to include the full operation in each diagnostic.

- **Type:** String (comma-separated list: `remarks`, `operations`)
- **Default:** Not set (no diagnostics)
- **Cache effect:** Cache-invalidating
- **Example usage:**

```bash
# Show performance remarks (e.g., which MMA version was selected)
MLIR_ENABLE_DIAGNOSTICS=remarks python my_kernel.py

# Show remarks with full operation context
MLIR_ENABLE_DIAGNOSTICS=remarks,operations python my_kernel.py
```

When set to `remarks`, the compiler will emit notes about optimization decisions such as:
- Which MMA version was selected for `tl.dot` operations
- Why certain optimizations were not applied
- Vectorization failure reasons

When `operations` is also included, each diagnostic includes the full MLIR operation that triggered it:
```
remark: MMA version 3 was not selected due to unsupported shapes or data types
note: see current operation: %3 = tt.dot %1, %2 : tensor<32x128xf32> ...
```

### 1.3 Interpreter Mode

#### `TRITON_INTERPRET`

When set to `1`, all `@triton.jit`-decorated functions are executed by a CPU-based interpreter instead of being compiled to GPU code. The interpreter uses NumPy equivalents of Triton operations and processes each program instance sequentially. This is one of the most useful debugging tools because it allows standard Python debugging tools (pdb, print statements, etc.) to be used inside kernel code.

- **Type:** Boolean
- **Default:** Not set (disabled -- kernels are compiled for the GPU)
- **Cache effect:** Cache-neutral (interpreter bypasses compilation entirely)
- **Example usage:**

```bash
# Run all kernels through the interpreter
TRITON_INTERPRET=1 python my_kernel.py

# Use pdb for step-by-step debugging
TRITON_INTERPRET=1 pdb my_kernel.py
# Then in pdb: b my_kernel.py:42
#              r
```

See [Section 3](#3-interpreter-mode) for detailed usage instructions.

### 1.4 Kernel Dump and Override

#### `TRITON_KERNEL_DUMP`

When set to `1`, every compiled kernel's intermediate representations (TTIR, TTGIR, LLIR, PTX/AMDGCN, cubin/HSACO) are dumped to the directory specified by `TRITON_DUMP_DIR` (or the default dump directory). This is useful for inspecting the IR generated for every kernel in a workload.

- **Type:** Boolean
- **Default:** Not set (disabled)
- **Cache effect:** Cache-neutral (does not affect compilation output)
- **Example usage:**

```bash
TRITON_KERNEL_DUMP=1 python my_kernel.py
```

On NVIDIA, when `TRITON_KERNEL_DUMP` is enabled, the SASS disassembly is also dumped alongside the cubin.

#### `TRITON_DUMP_DIR`

Specifies the directory where dumped kernel IR files are written. Defaults to `~/.triton/dump/`.

- **Type:** String (directory path)
- **Default:** `~/.triton/dump/`
- **Example usage:**

```bash
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/triton_dumps python my_kernel.py
```

#### `TRITON_KERNEL_OVERRIDE`

When set to `1`, the compilation pipeline checks the override directory for each compilation stage. If a file matching the expected filename and extension is found, that file is used instead of the compiler's output for that stage. This allows manual editing of IR at any stage and feeding it back into the pipeline.

- **Type:** Boolean
- **Default:** Not set (disabled)
- **Cache effect:** Cache-neutral (does not affect how the compiler generates output)
- **Example usage:**

```bash
TRITON_KERNEL_OVERRIDE=1 python my_kernel.py
```

See [Section 6](#6-kernel-override) for the step-by-step workflow.

#### `TRITON_OVERRIDE_DIR`

Specifies the directory where override IR files are read from. Defaults to `~/.triton/override/`.

- **Type:** String (directory path)
- **Default:** `~/.triton/override/`
- **Example usage:**

```bash
TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=/tmp/my_overrides python my_kernel.py
```

### 1.5 Compilation Control

#### `TRITON_ALWAYS_COMPILE`

When set to `1`, the compilation cache is bypassed and every kernel invocation triggers a full recompilation. This is essential when using dump or reproducer features, because cached results would prevent the compilation pipeline from running.

- **Type:** Boolean
- **Default:** Not set (cached kernels are reused)
- **Cache effect:** Cache-neutral (does not affect what is compiled, only whether compilation runs)
- **Example usage:**

```bash
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 python my_kernel.py
```

#### `USE_IR_LOC`

When set to a specific IR file extension (e.g., `ttir`, `ttgir`), creates location snapshots from the IR at that compilation stage. Location information maps IR operations back to source locations. This is an advanced debugging tool for compiler developers.

- **Type:** String (IR extension: `ttir`, `ttgir`, etc.)
- **Default:** Not set
- **Cache effect:** Cache-invalidating
- **Example usage:**

```bash
USE_IR_LOC=ttir python my_kernel.py
```

#### `DISABLE_LLVM_OPT`

Disables LLVM optimizations during the LLVM IR compilation stage. This produces unoptimized LLVM IR output, which can be useful for understanding what the MLIR-to-LLVM lowering produces before any LLVM-level optimizations are applied.

- **Type:** Boolean
- **Default:** Not set (LLVM optimizations enabled)
- **Cache effect:** Cache-invalidating
- **Example usage:**

```bash
DISABLE_LLVM_OPT=1 python my_kernel.py
```

#### `TRITON_STORE_BINARY_ONLY`

When set to `1`, only the final binary (cubin on NVIDIA, HSACO on AMD) and the JSON metadata are stored in the cache. Intermediate IR files (TTIR, TTGIR, LLIR, PTX/AMDGCN) are not stored. This reduces cache size but prevents later inspection of intermediate IR.

- **Type:** Boolean
- **Default:** Not set (all IR stages are cached)
- **Example usage:**

```bash
TRITON_STORE_BINARY_ONLY=1 python my_kernel.py
```

#### `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS`

By default, Triton requires that global tensor arguments passed to kernels be constexpr. Setting this to `1` relaxes that restriction, allowing non-constexpr globals. This is primarily a development/debugging tool.

- **Type:** Boolean
- **Default:** Not set
- **Cache effect:** Cache-invalidating
- **Example usage:**

```bash
TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 python my_kernel.py
```

#### `LLVM_EXTRACT_DI_LOCAL_VARIABLES`

Enables extraction of debug information (DI) local variable metadata from LLVM IR. This is used in conjunction with debug info generation for improved source-level debugging of generated GPU code.

- **Type:** Boolean
- **Default:** Not set
- **Cache effect:** Cache-invalidating
- **Example usage:**

```bash
LLVM_EXTRACT_DI_LOCAL_VARIABLES=1 python my_kernel.py
```

### 1.6 Frontend Debugging

#### `TRITON_FRONT_END_DEBUGGING`

When set to `1`, the Triton frontend (AST-to-TTIR code generator) preserves full Python tracebacks instead of filtering them. Normally, Triton filters out internal frames from `code_generator.py` and `ast.py` to show only user code in error messages. Enabling this shows the complete traceback including all internal frames.

- **Type:** Boolean
- **Default:** Not set (tracebacks are filtered)
- **Example usage:**

```bash
TRITON_FRONT_END_DEBUGGING=1 python my_kernel.py
```

### 1.7 Line Info and Debug Symbols

#### `TRITON_DISABLE_LINE_INFO`

When set to `1`, debug line information is not emitted in the generated GPU code. This affects both the `-lineinfo` flag passed to `ptxas` (on NVIDIA) and the debug metadata in the generated LLVM IR. Disabling line info can slightly reduce binary size but makes `cuda-gdb` and similar tools unable to map GPU code back to source lines.

- **Type:** Boolean
- **Default:** Not set (line info is emitted)
- **Cache effect:** Cache-invalidating
- **Example usage:**

```bash
TRITON_DISABLE_LINE_INFO=1 python my_kernel.py
```

### 1.8 Runtime Debugging

#### `TRITON_DEBUG`

When set to `1`, enables runtime-level debug checks. Most importantly, `device_assert` operations only execute when `TRITON_DEBUG` is set. Without it, `device_assert` calls are compiled to no-ops, avoiding any performance impact in production code.

- **Type:** Boolean
- **Default:** Not set (device_assert is a no-op)
- **Example usage:**

```bash
TRITON_DEBUG=1 python my_kernel.py
```

#### `TRITON_ENABLE_PYTHON_STACKTRACE`

Enables Python stacktrace attachment in certain error messages from the compiler. This is a cache-neutral variable.

- **Type:** Boolean
- **Default:** Not set
- **Cache effect:** Cache-neutral
- **Example usage:**

```bash
TRITON_ENABLE_PYTHON_STACKTRACE=1 python my_kernel.py
```

### 1.9 Autotuning

#### `TRITON_PRINT_AUTOTUNING`

When set to `1`, Triton prints a message to stdout after autotuning each kernel. The message includes the time spent autotuning and the best configuration that was selected.

- **Type:** Boolean
- **Default:** Not set
- **Example usage:**

```bash
TRITON_PRINT_AUTOTUNING=1 python my_kernel.py
```

Example output:
```
Triton autotuning for function matmul_kernel,
with key as (1024, 1024, 1024),
finished after 2.34s,
best config selected: triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8);
```

#### `TRITON_CACHE_AUTOTUNING`

When set to `1`, autotuning results are cached to disk so that subsequent runs of the same kernel with the same input sizes reuse the previously determined best configuration without re-running the benchmark.

- **Type:** Boolean
- **Default:** Not set (autotuning results are not persisted)
- **Example usage:**

```bash
TRITON_CACHE_AUTOTUNING=1 python my_kernel.py
```

### 1.10 Floating-Point Configuration

#### `TRITON_F32_DEFAULT`

Controls the default floating-point type used when `float32` is specified. This can be used to globally switch all fp32 operations to a different type for testing or compatibility purposes.

- **Type:** String
- **Default:** Not set (fp32 is used as-is)
- **Cache effect:** Cache-invalidating
- **Example usage:**

```bash
TRITON_F32_DEFAULT=TF32 python my_kernel.py
```

#### `TRITON_DEFAULT_FP_FUSION`

Controls whether floating-point fused multiply-add (FMA) operations are enabled by default. When set to `0` (false), FP fusion is disabled and multiply and add are kept as separate operations. When set to `1` (true, the default), the compiler may fuse multiply-add patterns into FMA instructions.

- **Type:** Boolean
- **Default:** `1` (FP fusion enabled)
- **Cache effect:** Cache-invalidating
- **Example usage:**

```bash
# Disable FP fusion for debugging numerics
TRITON_DEFAULT_FP_FUSION=0 python my_kernel.py
```

### 1.11 Reproducer

#### `TRITON_REPRODUCER_PATH`

Enables MLIR crash reproducer generation. When set, the MLIR pass manager generates a reproducer file for each compilation stage. These reproducer files contain the IR state and pass pipeline configuration, allowing the MLIR pass that was running to be replayed for debugging crashes or miscompilations.

The reproducer path is a prefix. For each compilation stage (e.g., `make_ttir`, `make_ttgir`, `make_llir`), a file named `<prefix>.<stage_name>.repro.mlir` is created.

- **Type:** String (file path prefix)
- **Default:** Not set (no reproducer files generated)
- **Cache effect:** Cache-neutral
- **Example usage:**

```bash
TRITON_ALWAYS_COMPILE=1 TRITON_REPRODUCER_PATH=/tmp/repro_prefix python my_kernel.py
```

This generates files like:
- `/tmp/repro_prefix.make_ttir.repro.mlir`
- `/tmp/repro_prefix.make_ttgir.repro.mlir`
- `/tmp/repro_prefix.make_llir.repro.mlir`

Each file contains MLIR reproducer content including the pass pipeline:
```
// configuration: -pass-pipeline="builtin.module(triton-combine, ...)"
module attributes {triton_gpu.num-warps = 4 : i32} {
  ...
}
// pipeline: "triton-combine, triton-gpu-coalesce, ..."
```

### 1.12 Instrumentation

#### `TRITON_INSTRUMENTATION_MODE`

Sets the instrumentation mode for the compiler. Supported values include:
- `fpsan` -- Floating-Point Sanitizer (see [Section 7](#7-floating-point-sanitizer-fpsan))
- `consan` -- Concurrency Sanitizer (see [Section 11](#11-concurrency-sanitizer-consan))
- Empty string (`""`) -- No instrumentation (default)

- **Type:** String
- **Default:** `""` (no instrumentation)
- **Cache effect:** Cache-invalidating (implicitly via mode selection)
- **Example usage:**

```bash
TRITON_INSTRUMENTATION_MODE=fpsan python my_kernel.py
```

Or programmatically:
```python
import triton
triton.knobs.compilation.instrumentation_mode = "fpsan"
# compile and run kernels here
triton.knobs.compilation.instrumentation_mode = ""
```

### 1.13 Architecture Override

#### `TRITON_OVERRIDE_ARCH`

Overrides the GPU architecture detected by the runtime. The value is an architecture string (e.g., `90` for SM90, `gfx942` for AMD). This forces the compiler to generate code for the specified architecture regardless of what GPU is actually present.

- **Type:** String (architecture identifier)
- **Default:** Not set (auto-detected from the GPU)
- **Cache effect:** Cache-invalidating
- **Example usage:**

```bash
# Compile for SM90 even on a different GPU
TRITON_OVERRIDE_ARCH=90 python my_kernel.py
```

### 1.14 Backend Selection

#### `TRITON_DEFAULT_BACKEND`

Selects which Triton backend to use. This overrides the automatic detection of the active GPU driver. Available values correspond to the installed backend names (e.g., `nvidia`, `amd`).

- **Type:** String (backend name)
- **Default:** Not set (auto-detected)
- **Example usage:**

```bash
TRITON_DEFAULT_BACKEND=nvidia python my_kernel.py
```

### 1.15 NVIDIA-Specific Variables

#### `NVPTX_ENABLE_DUMP`

When set to `1`, prints the generated NVPTX code to stdout during compilation. This is a quick way to see the PTX output for a kernel.

- **Type:** Boolean
- **Default:** Not set
- **Cache effect:** Cache-invalidating
- **Example usage:**

```bash
NVPTX_ENABLE_DUMP=1 python my_kernel.py
```

Output:
```
// -----// NVPTX Dump //----- //
.version 8.0
.target sm_90
.address_size 64
...
```

#### `DISABLE_PTXAS_OPT`

Disables ptxas (PTX assembler) optimizations. When set, ptxas is run with `--opt-level 0` and `-g` (full debug info) instead of the default `-lineinfo`. This produces less optimized but more debuggable cubin.

- **Type:** Boolean
- **Default:** Not set (ptxas optimizations enabled)
- **Cache effect:** Cache-invalidating
- **Example usage:**

```bash
DISABLE_PTXAS_OPT=1 python my_kernel.py
```

#### `PTXAS_OPTIONS`

Passes additional options directly to the ptxas assembler. The value is a space-separated string of ptxas command-line flags.

- **Type:** String (space-separated ptxas options)
- **Default:** Not set
- **Cache effect:** Cache-invalidating
- **Example usage:**

```bash
PTXAS_OPTIONS="--verbose --maxnreg 64" python my_kernel.py
```

#### `TRITON_MOCK_PTX_VERSION`

Overrides the PTX version used during compilation. This is primarily a testing tool for verifying compilation behavior against specific PTX versions.

- **Type:** String (PTX version number)
- **Default:** Not set (uses the version from the installed toolkit)
- **Cache effect:** Cache-invalidating
- **Example usage:**

```bash
TRITON_MOCK_PTX_VERSION=8.0 python my_kernel.py
```

#### `TRITON_DUMP_PTXAS_LOG`

When set to `1`, the ptxas log output (which includes register usage and other statistics) is printed to stdout even when ptxas succeeds.

- **Type:** Boolean
- **Default:** Not set (log is only shown on ptxas failure)
- **Example usage:**

```bash
TRITON_DUMP_PTXAS_LOG=1 python my_kernel.py
```

#### `TRITON_LIBDEVICE_PATH`

Specifies the path to the NVIDIA libdevice library (libdevice.10.bc or similar). This overrides the default search path.

- **Type:** String (file path)
- **Default:** Not set (auto-detected)
- **Example usage:**

```bash
TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc python my_kernel.py
```

#### `TRITON_LIBCUDA_PATH`

Specifies the path to the CUDA driver library (libcuda.so). This overrides the default search path.

- **Type:** String (file path)
- **Default:** Not set (auto-detected)

#### `TRITON_PTXAS_BLACKWELL_PATH`

Specifies the path to the ptxas binary for Blackwell GPUs. The default looks for a bundled ptxas-blackwell binary. The general `ptxas` path is controlled by `TRITON_PTAS_PATH`.

#### `TRITON_CUOBJDUMP_PATH`, `TRITON_NVDISASM_PATH`, `TRITON_PTAS_PATH`

Override the paths to the respective NVIDIA tools (cuobjdump, nvdisasm, ptxas). These are automatically set up with bundled binaries but can be overridden if needed.

### 1.16 AMD-Specific Variables

#### `AMDGCN_ENABLE_DUMP`

When set to `1`, prints the generated AMDGCN (AMD GPU assembly) code to stdout during compilation.

- **Type:** Boolean
- **Default:** Not set
- **Cache effect:** Cache-invalidating
- **Example usage:**

```bash
AMDGCN_ENABLE_DUMP=1 python my_kernel.py
```

#### `AMDGCN_USE_BUFFER_OPS`

Controls whether the AMD backend uses buffer operations instead of flat operations for memory accesses. Enabled by default.

- **Type:** Boolean
- **Default:** `1` (enabled)
- **Cache effect:** Cache-invalidating

#### `AMDGCN_USE_BUFFER_ATOMICS`

Controls whether the AMD backend uses buffer atomic operations. Only effective when `AMDGCN_USE_BUFFER_OPS` is also enabled.

- **Type:** Boolean
- **Default:** `1` (enabled)
- **Cache effect:** Cache-invalidating

#### `AMDGCN_ANALYZE_SMALL_TENSOR_RANGE`

Controls whether the AMD backend analyzes small tensor ranges for buffer operations. Only effective when `AMDGCN_USE_BUFFER_OPS` is enabled.

- **Type:** Boolean
- **Default:** `0` (disabled)
- **Cache effect:** Cache-invalidating

#### `TRITON_LIBHIP_PATH`

Specifies the path to the HIP runtime library. Overrides the default search path.

- **Type:** String (file path)
- **Default:** Not set (auto-detected)

#### `TRITON_HIP_USE_BLOCK_PINGPONG`

Controls whether block pingpong optimization is used in the AMD backend.

- **Type:** Boolean (optional)
- **Default:** Not set (backend decides)
- **Cache effect:** Cache-invalidating

#### `TRITON_HIP_USE_IN_THREAD_TRANSPOSE`

Controls whether in-thread transpose is used in the AMD backend.

- **Type:** Boolean (optional)
- **Default:** Not set (backend decides)
- **Cache effect:** Cache-invalidating

#### `TRITON_HIP_USE_ASYNC_COPY`

Controls whether async copy operations are used in the AMD backend.

- **Type:** Boolean (optional)
- **Default:** Not set (backend decides)
- **Cache effect:** Cache-invalidating

#### `AMDGCN_SCALARIZE_PACKED_FOPS`

Controls scalarization of packed floating-point operations in the AMD backend.

- **Type:** Boolean
- **Default:** Not set

#### `TRITON_DUMP_MIR`

Specifies a path to dump MIR (Machine IR) files for AMD backend debugging and analysis.

- **Type:** String (file path)
- **Default:** Not set
- **Cache effect:** Cache-invalidating

#### `TRITON_SWAP_MIR`

Specifies a path to externally-provided MIR files to use instead of generated ones. This enables swapping in hand-edited or pre-compiled MIR for debugging.

- **Type:** String (file path)
- **Default:** Not set
- **Cache effect:** Cache-invalidating

#### `TRITON_SWAP_MIR_ENABLE_MISCHED`

Enables the machine instruction scheduler when using MIR swap mode in the AMD backend.

- **Type:** Boolean
- **Default:** `0` (disabled)
- **Cache effect:** Cache-invalidating

### 1.17 Address Sanitizer

#### `TRITON_ENABLE_ASAN`

Enables AddressSanitizer instrumentation for GPU code generation. Currently supported on AMD GPUs. When enabled, the compiler adds address checking instrumentation to detect out-of-bounds memory accesses.

- **Type:** Boolean
- **Default:** Not set (disabled)
- **Cache effect:** Cache-invalidating
- **Example usage:**

```bash
TRITON_ENABLE_ASAN=1 python my_kernel.py
```

On AMD, this also sets the `+xnack` target feature and links the ASAN runtime library.

### 1.18 Cache Configuration

#### `TRITON_CACHE_DIR`

Specifies the directory where compiled kernel caches are stored. Defaults to `~/.triton/cache/`.

- **Type:** String (directory path)
- **Default:** `~/.triton/cache/`

#### `TRITON_HOME`

Specifies the Triton home directory. Other directory defaults (cache, dump, override) are derived from this as `<TRITON_HOME>/.triton/<subdir>`.

- **Type:** String (directory path)
- **Default:** `~/` (user home directory)

#### `TRITON_CACHE_MANAGER`

Specifies a custom cache manager class. The value must be in `MODULE:CLASS` format (e.g., `my_module:MyCacheManager`). The class must inherit from `CacheManager`.

- **Type:** String (`MODULE:CLASS` format)
- **Default:** Not set (uses `FileCacheManager`)

#### `TRITON_REMOTE_CACHE_BACKEND`

Specifies a remote cache backend class for distributed caching. The value must be in `MODULE:CLASS` format. The class must inherit from `RemoteCacheBackend`.

- **Type:** String (`MODULE:CLASS` format)
- **Default:** Not set

### 1.19 Build Configuration

#### `CC`

Specifies the C compiler to use when building Triton's native extensions.

- **Type:** String (compiler path or name)
- **Default:** Not set (system default)

#### `TRITON_CUDACRT_PATH`

Specifies the path to the CUDA CRT (CUDA runtime) libraries for building.

- **Type:** String (directory path)
- **Default:** Not set

#### `TRITON_CUDART_PATH`

Specifies the path to the CUDA runtime libraries for building.

- **Type:** String (directory path)
- **Default:** Not set

### 1.20 Plugin System

#### `TRITON_PLUGIN_PATHS`

Specifies a list of plugin shared library paths to load at startup. Plugins can extend Triton's compilation pipeline with custom MLIR passes.

- **Type:** String (colon-separated paths, platform-dependent)
- **Default:** Not set
- **Cache effect:** Cache-invalidating

#### `TRITON_PLUGIN_VERSION_CHECK`

Controls version checking for loaded plugins. When set to `true`, the full version string is checked. When set to `false`, version checking is skipped. When unset, a partial release version check is performed.

- **Type:** Boolean (optional)
- **Default:** Not set (partial version check)
- **Cache effect:** Cache-invalidating

#### `LLVM_PASS_PLUGIN_PATH`

Specifies the path to an LLVM pass plugin shared library to load.

- **Type:** String (file path)
- **Default:** Not set
- **Cache effect:** Cache-invalidating

### 1.21 Proton Profiling

#### `TRITON_PROTON_DISABLE`

Disables Triton's Proton profiling system.

- **Type:** Boolean
- **Default:** `0` (Proton is enabled)

#### `TRITON_CUPTI_LIB_PATH`

Specifies the path to the CUPTI library directory for Proton profiling.

- **Type:** String (directory path)
- **Default:** Bundled CUPTI path

#### `TRITON_CUPTI_LIB_BLACKWELL_PATH`

Specifies the path to the CUPTI library directory for Blackwell GPUs.

- **Type:** String (directory path)
- **Default:** Bundled CUPTI Blackwell path

#### `TRITON_PROFILE_BUFFER_SIZE`

Sets the size of the profiling buffer in bytes.

- **Type:** Integer
- **Default:** `67108864` (64 MiB)

#### `TRITON_PROFILE_METRIC_BUFFER_SIZE`

Sets the size of the profiling metric buffer in bytes.

- **Type:** Integer
- **Default:** `67108864` (64 MiB)

#### `TRITON_ENABLE_NVTX`

Enables NVTX (NVIDIA Tools Extension) markers in the Proton profiler.

- **Type:** Boolean
- **Default:** `1` (enabled)

#### `TRITON_ENABLE_HW_TRACE`

Enables hardware trace collection for Blackwell+ GPUs. The profiling session must start after CUDA driver initialization but before CUDA context creation.

- **Type:** Boolean
- **Default:** `0` (disabled)

### 1.22 Warp Specialization Debugging

#### `TRITON_PARTITION_SCHEDULING_ENABLE_DUMP_DOT`

Enables dumping of DOT format graphs for the warp specialization partition scheduling algorithm. Useful for debugging the partition scheduling pass.

- **Type:** Boolean
- **Default:** Not set
- **Cache effect:** Cache-invalidating

#### `TRITON_PARTITION_SCHEDULING_DUMP_DATA_ONLY`

When partition scheduling dump is enabled, restricts the dump to data-related information only.

- **Type:** Boolean
- **Default:** Not set
- **Cache effect:** Cache-invalidating

#### `TRITON_PARTITION_SCHEDULING_DUMP_LOOP_ONLY`

When partition scheduling dump is enabled, restricts the dump to loop-related information only.

- **Type:** Boolean
- **Default:** Not set
- **Cache effect:** Cache-invalidating

### 1.23 Other Debug Variables

#### `DISABLE_MMA_V3`

Disables MMA version 3 (warp-group level matrix multiply-accumulate) selection. Forces the compiler to use an older MMA version.

- **Type:** Boolean
- **Default:** Not set
- **Cache effect:** Cache-invalidating

#### `DISABLE_MMA_V5`

Disables MMA version 5 selection.

- **Type:** Boolean
- **Default:** Not set
- **Cache effect:** Cache-invalidating

#### `ALLOW_LHS_TMEM_LAYOUT_CONVERSION`

Allows layout conversion for LHS (left-hand side) tensors in tensor memory operations. Normally this is disabled to catch potential issues.

- **Type:** Boolean
- **Default:** Not set
- **Cache effect:** Cache-invalidating

#### `TRITON_PREFER_TMEM_16x256_LAYOUT`

Prefers a specific 16x256 layout for tensor memory operations.

- **Type:** Boolean
- **Default:** Not set
- **Cache effect:** Cache-invalidating

#### `MLIR_DISABLE_MULTITHREADING`

Disables multithreading in the MLIR context. Useful for deterministic debugging of MLIR passes.

- **Type:** Boolean
- **Default:** Not set
- **Cache effect:** Cache-invalidating

#### `TRITON_ENABLE_EXPERIMENTAL_CONSAN`

Enables the experimental Concurrency Sanitizer for detecting data races in GPU kernels.

- **Type:** Boolean
- **Default:** Not set
- **Cache effect:** Cache-invalidating

### 1.24 Redis Cache (Distributed Caching)

#### `TRITON_REDIS_HOST`

The hostname for the Redis server used for distributed caching.

- **Type:** String
- **Default:** `localhost`

#### `TRITON_REDIS_PORT`

The port for the Redis server.

- **Type:** Integer
- **Default:** `6379`

#### `TRITON_REDIS_KEY_FORMAT`

The key format string used for Redis cache entries. `{key}` and `{filename}` are replaced with the actual values.

- **Type:** String
- **Default:** `triton:{key}:{filename}`

---

## 2. Configuration Knobs

Triton's configuration knobs provide a Python API for controlling compilation and runtime behavior. Knobs are defined in `triton/knobs.py` and organized into groups. Each knob wraps an environment variable but can also be set programmatically at runtime.

### 2.1 Knob Hierarchy

```
knobs/
  build         -- BuildImpl configuration (CC, CUDA paths)
  redis         -- Redis cache configuration
  cache         -- Cache directories and managers
  compilation   -- Compilation pipeline controls
  autotuning    -- Autotuning behavior
  runtime       -- Runtime behavior (interpreter, debug, hooks)
  language      -- Language-level defaults (fp32, FP fusion)
  nvidia        -- NVIDIA-specific knobs
  amd           -- AMD-specific knobs
  proton        -- Proton profiling knobs
```

### 2.2 Setting Knobs Programmatically

```python
import triton

# Set a knob (also sets the corresponding env var when propagate_env is True)
triton.knobs.runtime.interpret = True
triton.knobs.compilation.dump_ir = True
triton.knobs.compilation.always_compile = True

# Reset a knob to its default (reads from env var or uses default)
del triton.knobs.runtime.interpret

# Get a knob value
if triton.knobs.runtime.debug:
    print("Debug mode is active")
```

### 2.3 Knob Scoping

The `base_knobs.scope()` context manager saves and restores knob state, including environment variables. This is useful for tests or temporary configuration changes:

```python
import triton

with triton.knobs.compilation.scope():
    triton.knobs.compilation.dump_ir = True
    # ... run kernels with dump enabled ...
# knobs are restored to their original values here
```

### 2.4 Compilation Knobs Detail

The `compilation_knobs` class provides these knobs:

| Knob | Env Var | Type | Default | Description |
|------|---------|------|---------|-------------|
| `override` | `TRITON_KERNEL_OVERRIDE` | bool | False | Enable kernel override from override directory |
| `dump_ir` | `TRITON_KERNEL_DUMP` | bool | False | Dump all IR stages to dump directory |
| `dump_ir_extract_di_local_variables` | `LLVM_EXTRACT_DI_LOCAL_VARIABLES` | bool | False | Extract DI local variable info |
| `store_binary_only` | `TRITON_STORE_BINARY_ONLY` | bool | False | Only cache the final binary |
| `always_compile` | `TRITON_ALWAYS_COMPILE` | bool | False | Bypass cache and always recompile |
| `use_ir_loc` | `USE_IR_LOC` | str | None | Create location snapshots at specified IR stage |
| `enable_asan` | `TRITON_ENABLE_ASAN` | bool | False | Enable AddressSanitizer |
| `disable_line_info` | `TRITON_DISABLE_LINE_INFO` | bool | False | Disable debug line info |
| `front_end_debugging` | `TRITON_FRONT_END_DEBUGGING` | bool | False | Show full Python tracebacks |
| `allow_non_constexpr_globals` | `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS` | bool | False | Allow non-constexpr global arguments |
| `instrumentation_mode` | `TRITON_INSTRUMENTATION_MODE` | str | `""` | Instrumentation mode (fpsan, consan) |
| `listener` | N/A | callable | None | CompilationListener callback |

### 2.5 Compilation Listener

You can register a `CompilationListener` callback to receive timing and metadata information for each compilation:

```python
import triton
from triton.knobs import CompileTimes

def my_listener(*, src, metadata, metadata_group, times, cache_hit):
    if cache_hit:
        print(f"Cache hit for {src.name}")
    else:
        print(f"Compiled {src.name} in {times.total}us")
        for stage, duration in times.lowering_stages:
            print(f"  {stage}: {duration}us")

triton.knobs.compilation.listener = my_listener
```

The `CompileTimes` dataclass provides:
- `ir_initialization` -- Time spent in IR initialization (microseconds)
- `lowering_stages` -- List of (stage_name, duration) tuples (microseconds)
- `store_results` -- Time spent storing results (microseconds)
- `total` -- Total compilation time (microseconds)
- `total_lowering` -- Total time spent in all lowering stages (microseconds)

### 2.6 Runtime Knobs Detail

| Knob | Env Var | Type | Default | Description |
|------|---------|------|---------|-------------|
| `interpret` | `TRITON_INTERPRET` | bool | False | Use CPU interpreter instead of GPU |
| `debug` | `TRITON_DEBUG` | bool | False | Enable runtime debug checks (device_assert) |
| `override_arch` | `TRITON_OVERRIDE_ARCH` | str | None | Override GPU architecture |
| `launch_enter_hook` | N/A | HookChain | Empty | Hooks called before kernel launch |
| `launch_exit_hook` | N/A | HookChain | Empty | Hooks called after kernel launch |
| `kernel_load_start_hook` | N/A | HookChain | Empty | Hooks called before binary loading |
| `kernel_load_end_hook` | N/A | HookChain | Empty | Hooks called after binary loading |
| `kernel_unload_hook` | N/A | HookChain | Empty | Hooks called when kernel is freed |
| `jit_cache_hook` | N/A | callable | None | Hook called before JIT compilation |
| `jit_post_compile_hook` | N/A | callable | None | Hook called after JIT compilation |
| `add_stages_inspection_hook` | N/A | callable | None | Hook for inspecting/modifying pipeline stages |

### 2.7 Hook Chains

Runtime hooks use the `HookChain` class, which allows multiple hooks to be registered and called in order:

```python
import triton

def my_launch_hook(metadata):
    print(f"Launching kernel: {metadata['name']}")

triton.knobs.runtime.launch_enter_hook.add(my_launch_hook)

# Later, to remove the hook:
triton.knobs.runtime.launch_enter_hook.remove(my_launch_hook)
```

`launch_enter_hook` calls hooks in forward order (first added, first called).
`launch_exit_hook` calls hooks in reverse order (last added, first called).

### 2.8 Autotuning Knobs Detail

| Knob | Env Var | Type | Default | Description |
|------|---------|------|---------|-------------|
| `cache` | `TRITON_CACHE_AUTOTUNING` | bool | False | Cache autotuning results to disk |
| `print` | `TRITON_PRINT_AUTOTUNING` | bool | False | Print autotuning results |
| `listener` | N/A | callable | None | AutotuneListener callback |

### 2.9 Autotune Listener

```python
import triton

def my_autotune_listener(*, fn, key, best_config, configs_timings, duration, cache_hit):
    print(f"Autotuned {fn.__name__}: best={best_config}, duration={duration:.2f}s")

triton.knobs.autotuning.listener = my_autotune_listener
```

### 2.10 Language Knobs Detail

| Knob | Env Var | Type | Default | Description |
|------|---------|------|---------|-------------|
| `fp32_default` | `TRITON_F32_DEFAULT` | str | None | Override default fp32 type |
| `default_fp_fusion` | `TRITON_DEFAULT_FP_FUSION` | bool | True | Enable FP fusion by default |

### 2.11 NVIDIA Knobs Detail

| Knob | Env Var | Type | Default | Description |
|------|---------|------|---------|-------------|
| `cuobjdump` | `TRITON_CUOBJDUMP_PATH` | NvidiaTool | bundled | cuobjdump binary |
| `nvdisasm` | `TRITON_NVDISASM_PATH` | NvidiaTool | bundled | nvdisasm binary |
| `ptxas` | `TRITON_PTAS_PATH` | NvidiaTool | bundled | ptxas binary |
| `ptxas_blackwell` | `TRITON_PTAS_BLACKWELL_PATH` | NvidiaTool | bundled | ptxas for Blackwell |
| `dump_nvptx` | `NVPTX_ENABLE_DUMP` | bool | False | Print PTX to stdout |
| `disable_ptxas_opt` | `DISABLE_PTXAS_OPT` | bool | False | Disable ptxas optimizations |
| `ptxas_options` | `PTXAS_OPTIONS` | str | None | Extra ptxas options |
| `mock_ptx_version` | `TRITON_MOCK_PTX_VERSION` | str | None | Override PTX version |
| `dump_ptxas_log` | `TRITON_DUMP_PTXAS_LOG` | bool | False | Print ptxas log on success |
| `libdevice_path` | `TRITON_LIBDEVICE_PATH` | str | None | Path to libdevice |
| `libcuda_path` | `TRITON_LIBCUDA_PATH` | str | None | Path to libcuda |

The `NvidiaTool` dataclass provides `path` and `version` attributes.

### 2.12 AMD Knobs Detail

| Knob | Env Var | Type | Default | Description |
|------|---------|------|---------|-------------|
| `use_buffer_ops` | `AMDGCN_USE_BUFFER_OPS` | bool | True | Use buffer operations |
| `use_buffer_atomics` | `AMDGCN_USE_BUFFER_ATOMICS` | bool | True | Use buffer atomics |
| `buffer_ops_analyze_small_tensor_range` | `AMDGCN_ANALYZE_SMALL_TENSOR_RANGE` | bool | False | Analyze small tensor range |
| `dump_amdgcn` | `AMDGCN_ENABLE_DUMP` | bool | False | Print AMDGCN to stdout |
| `libhip_path` | `TRITON_LIBHIP_PATH` | str | None | Path to HIP library |
| `use_block_pingpong` | `TRITON_HIP_USE_BLOCK_PINGPONG` | bool | None | Block pingpong optimization |
| `use_in_thread_transpose` | `TRITON_HIP_USE_IN_THREAD_TRANSPOSE` | bool | None | In-thread transpose |
| `use_async_copy` | `TRITON_HIP_USE_ASYNC_COPY` | bool | None | Async copy operations |
| `scalarize_packed_fops` | `AMDGCN_SCALARIZE_PACKED_FOPS` | bool | False | Scalarize packed fops |
| `dump_mir` | `TRITON_DUMP_MIR` | str | None | Path to dump MIR files |
| `swap_mir` | `TRITON_SWAP_MIR` | str | None | Path to MIR files to swap in |
| `swap_mir_enable_misched` | `TRITON_SWAP_MIR_ENABLE_MISCHED` | bool | False | Enable MISched in MIR swap |

### 2.13 Refreshing Knobs

Some knobs are cached at module load time for performance. The `refresh_knobs()` function re-reads these:

```python
import triton
triton.knobs.refresh_knobs()
```

This refreshes:
- `runtime.debug` from `TRITON_DEBUG`
- `compilation.instrumentation_mode` from `TRITON_INSTRUMENTATION_MODE`

### 2.14 Environment Propagation

The `propagate_env` flag controls whether setting a knob also sets the corresponding environment variable. This is `True` by default but is disabled in some test contexts to prevent knob changes from leaking between tests.

---

## 3. Interpreter Mode

The interpreter is Triton's primary debugging tool for kernel logic errors. It runs Triton kernels on the CPU using NumPy, bypassing GPU compilation entirely.

### 3.1 Enabling the Interpreter

```bash
TRITON_INTERPRET=1 python my_script.py
```

When this is set, every `@triton.jit`-decorated function returns an `InterpretedFunction` instead of a `JITFunction`. The `InterpretedFunction` executes the kernel body using NumPy operations on the CPU.

### 3.2 How the Interpreter Works

1. Each kernel parameter is converted to a `TensorHandle` containing a NumPy array
2. Triton operations (`tl.load`, `tl.store`, `tl.dot`, etc.) are mapped to their NumPy equivalents
3. Each program instance (PID) is executed sequentially
4. Standard Python control flow (`if`, `for`) works normally since the code runs in Python

### 3.3 Debugging with Print

Inside an interpreted kernel, you can use Python's `print` to inspect values:

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Print entire tensor
    print("offsets:", offs)

    # Print individual tensor value
    x = tl.load(x_ptr + offs, mask=offs < N)
    print("x[0]:", x.handle.data[0])  # Access raw NumPy data by index
```

### 3.4 Debugging with pdb

You can use Python's debugger for step-by-step execution:

**From the command line:**

```bash
TRITON_INTERPRET=1 pdb my_script.py
# In pdb:
(pdb) b my_script.py:42
(pdb) r
```

**With inline breakpoints:**

```python
import triton
import triton.language as tl
import pdb

@triton.jit
def my_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    pdb.set_trace()  # Execution will pause here
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs)
    tl.store(y_ptr + offs, x)
```

### 3.5 Interpreter Limitations

The interpreter has known limitations:

1. **No bfloat16 support**: Operations on `bfloat16` tensors are not supported. Convert to `float32` using `tl.cast(tensor)`.

2. **No indirect memory access**: Patterns like the following are not supported:
   ```python
   ptr = tl.load(ptr_ptr)  # Load a pointer from memory
   x = tl.load(ptr)         # Use the loaded pointer -- NOT supported
   ```

3. **Sequential execution**: All program instances run sequentially, so race conditions and synchronization issues cannot be detected.

4. **No GPU-specific features**: Hardware-specific features (tensor memory, warp-group operations, etc.) may not be accurately modeled.

5. **FpSan does not apply**: The floating-point sanitizer is a compiler feature and has no effect in interpreter mode.

---

## 4. IR Dumping

Triton's compilation pipeline goes through several IR stages. Inspecting the IR at each stage is a core debugging technique.

### 4.1 Compilation Stages

On NVIDIA, the stages are:

| Stage | Extension | Description |
|-------|-----------|-------------|
| Source | `.source` | Original Python source (AST) |
| TTIR | `.ttir` | Triton Tensor IR (high-level) |
| TTGIR | `.ttgir` | Triton GPU IR (with GPU-specific info) |
| LLIR | `.llir` | LLVM IR |
| PTX | `.ptx` | NVIDIA PTX assembly |
| CUBIN | `.cubin` | Compiled GPU binary |
| SASS | `.sass` | Disassembled SASS (when kernel dump enabled) |

On AMD, the stages are:

| Stage | Extension | Description |
|-------|-----------|-------------|
| Source | `.source` | Original Python source (AST) |
| TTIR | `.ttir` | Triton Tensor IR |
| TTGIR | `.ttgir` | Triton GPU IR |
| LLIR | `.llir` | LLVM IR |
| AMDGCN | `.amdgcn` | AMD GPU assembly |
| HSACO | `.hsaco` | Compiled AMD binary |

### 4.2 Dumping All Stages

To dump all IR stages for every compiled kernel:

```bash
TRITON_KERNEL_DUMP=1 TRITON_ALWAYS_COMPILE=1 python my_kernel.py
```

Files are written to `~/.triton/dump/` (or `TRITON_DUMP_DIR`). The filenames follow the pattern `<kernel_name>.<extension>`.

To find the dumped files:

```bash
ls ~/.triton/dump/
# Output:
# my_kernel.source  my_kernel.ttir  my_kernel.ttgir  my_kernel.llir
# my_kernel.ptx     my_kernel.cubin my_kernel.sass   my_kernel.json
```

### 4.3 Inspecting IR via the Cache

Even without `TRITON_KERNEL_DUMP`, all IR stages are stored in the Triton cache (unless `TRITON_STORE_BINARY_ONLY=1`). You can find them in `~/.triton/cache/`:

```bash
# Find cached IR files
find ~/.triton/cache/ -name "*.ttir" | head
```

### 4.4 Inspecting IR Programmatically

The `CompiledKernel` object provides access to all assembly stages via the `asm` dict:

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, N, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs, mask=offs < N)
    tl.store(x_ptr + offs, x + 1, mask=offs < N)

# Compile the kernel (without running it)
kernel = triton.compile(
    triton.compiler.ASTSource(
        fn=my_kernel,
        signature={"x_ptr": "*fp32", "N": "i32"},
        constexprs={"BLOCK_SIZE": 1024},
    )
)

# Print each IR stage
print("=== TTIR ===")
print(kernel.asm["ttir"])
print("=== TTGIR ===")
print(kernel.asm["ttgir"])
print("=== LLIR ===")
print(kernel.asm["llir"])
print("=== PTX ===")  # NVIDIA only
print(kernel.asm["ptx"])
print("=== SASS ===")  # NVIDIA only, lazily loaded
print(kernel.asm["sass"])
```

### 4.5 Dumping MLIR Pass IR

For fine-grained IR inspection during individual passes:

```bash
# Dump IR for every pass in every kernel
MLIR_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 python my_kernel.py

# Dump IR only for a specific kernel
MLIR_ENABLE_DUMP=my_kernel TRITON_ALWAYS_COMPILE=1 python my_kernel.py

# Write dump to file instead of stderr
MLIR_DUMP_PATH=/tmp/dump.mlir MLIR_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 python my_kernel.py
```

### 4.6 Dumping LLVM IR

To see LLVM-level IR during LLVM passes:

```bash
LLVM_IR_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 python my_kernel.py
```

### 4.7 Dumping PTX/AMDGCN

For quick inspection of the final assembly:

```bash
# NVIDIA: Print PTX to stdout
NVPTX_ENABLE_DUMP=1 python my_kernel.py

# AMD: Print AMDGCN to stdout
AMDGCN_ENABLE_DUMP=1 python my_kernel.py
```

### 4.8 Unoptimized IR

To see the IR before any optimizations are applied:

```bash
DISABLE_LLVM_OPT=1 python my_kernel.py
```

---

## 5. Reproducer System

The reproducer system generates MLIR crash reproducer files that capture the IR state and pass pipeline at each compilation stage. These files can be used to reproduce and debug compiler crashes or miscompilations.

### 5.1 Enabling Reproducer Generation

```bash
TRITON_ALWAYS_COMPILE=1 TRITON_REPRODUCER_PATH=/tmp/repro_prefix python my_kernel.py
```

The `TRITON_REPRODUCER_PATH` is a prefix. For each compilation stage, a file is created:
- `/tmp/repro_prefix.make_ttir.repro.mlir`
- `/tmp/repro_prefix.make_ttgir.repro.mlir`
- `/tmp/repro_prefix.make_llir.repro.mlir`

### 5.2 Reproducer File Contents

Each reproducer file contains:

1. An MLIR module with the IR at that compilation stage
2. A `pipeline` comment specifying the pass pipeline that was configured

Example content:
```mlir
// configuration: -pass-pipeline="builtin.module(triton-combine, ...)"
module attributes {triton_gpu.num-warps = 4 : i32} {
  tt.func public @my_kernel(%arg0: !tt.ptr<f32>, ...) {
    ...
  }
}
// pipeline: "triton-combine, triton-gpu-coalesce, ..."
```

### 5.3 Using Reproducers

Reproducer files can be fed back to MLIR's standalone tools for debugging:

```bash
# Run the pass pipeline from the reproducer
triton-opt repro_prefix.make_ttgir.repro.mlir -pass-pipeline="builtin.module(triton-combine, ...)"
```

This is primarily useful for compiler developers investigating crashes in specific MLIR passes.

### 5.4 Reproducer with Stages Inspection Hook

The reproducer system works together with the `add_stages_inspection_hook` (see [Section 9](#9-pipeline-inspection-hook)) to provide a complete debugging workflow:

1. Run with reproducer enabled to get baseline reproducer files
2. Set up an inspection hook to modify or inspect specific stages
3. Compare the reproducer files from the hooked run against the baseline

---

## 6. Kernel Override

The kernel override system allows you to manually edit IR at any compilation stage and feed it back into the pipeline. This is useful for experimenting with IR transformations or working around compiler bugs.

### 6.1 Step-by-Step Override Workflow

**Step 1: Dump the kernel IR**

```bash
TRITON_KERNEL_DUMP=1 TRITON_ALWAYS_COMPILE=1 python my_kernel.py
```

**Step 2: Locate the dumped files**

```bash
ls ~/.triton/dump/
# my_kernel.source  my_kernel.ttir  my_kernel.ttgir
# my_kernel.llir    my_kernel.ptx   my_kernel.cubin
```

**Step 3: Copy the IR you want to override to the override directory**

```bash
mkdir -p ~/.triton/override/
cp ~/.triton/dump/my_kernel.ttgir ~/.triton/override/my_kernel.ttgir
```

**Step 4: Edit the IR file**

Open `~/.triton/override/my_kernel.ttgir` in an editor and make your changes. For example, you might change the number of warps, modify an operation, or adjust a layout.

**Step 5: Run with override enabled**

```bash
TRITON_KERNEL_OVERRIDE=1 TRITON_ALWAYS_COMPILE=1 python my_kernel.py
```

When the compiler reaches the TTGIR stage, it will use the file from the override directory instead of the compiler-generated TTGIR. You will see output like:

```
Overriding kernel with file /home/user/.triton/override/my_kernel.ttgir
```

**Step 6: Verify the result**

The remaining pipeline stages (LLIR, PTX, cubin) will be generated from your overridden TTGIR.

### 6.2 Override via Autotune Config

You can also override IR at the autotune config level without using `TRITON_KERNEL_OVERRIDE`. Set the `ir_override` parameter in a `triton.Config`:

```python
import triton
import triton.language as tl

configs = [
    triton.Config(
        {'BLOCK_SIZE': 1024},
        num_warps=4,
        ir_override='/path/to/my_kernel.ttgir'
    ),
]

@triton.autotune(configs=configs, key=['N'])
@triton.jit
def my_kernel(x_ptr, N, BLOCK_SIZE: tl.constexpr):
    ...
```

When `ir_override` is set in a config, the compiler will use the specified IR file for the matching stage extension.

### 6.3 Override Directory Customization

By default, overrides are read from `~/.triton/override/`. You can change this:

```bash
TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=/tmp/my_overrides python my_kernel.py
```

---

## 7. Floating-Point Sanitizer (FpSan)

FpSan is a compiler instrumentation mode that rewrites floating-point operations into deterministic "payload algebra" over integer bit-patterns. It is designed for structural kernel validation rather than IEEE numerical accuracy.

### 7.1 Enabling FpSan

**From the shell:**

```bash
TRITON_INSTRUMENTATION_MODE=fpsan python my_kernel.py
```

**Programmatically:**

```python
import triton

triton.knobs.compilation.instrumentation_mode = "fpsan"
# compile and run kernels here
triton.knobs.compilation.instrumentation_mode = ""
```

### 7.2 How FpSan Works

1. **Embed**: Each floating-point bit-pattern is mapped to an integer payload via `embed(x)`
2. **Rewrite**: Floating-point operations are replaced with integer-domain rewrites that preserve specific algebraic identities
3. **Unembed**: The result is mapped back to a floating-point bit-pattern via `unembed(u)`

Key fixed points:
- `embed(+0.0) = 0`
- `embed(+1.0) = 1`
- `embed(-1.0) = all-ones`

### 7.3 Typical Usage Patterns

FpSan is most effective when comparing two kernels that should produce the same result:

```python
import triton
import triton.language as tl
import torch

triton.knobs.compilation.instrumentation_mode = "fpsan"

# Run reference kernel
result_ref = run_reference_kernel(inputs)

# Run optimized kernel
result_opt = run_optimized_kernel(inputs)

triton.knobs.compilation.instrumentation_mode = ""

# Compare FpSan outputs (NOT against ordinary fp outputs)
assert torch.allclose(result_ref, result_opt)
```

Use cases include:
- Comparing an optimized kernel against a simple reference
- Comparing a fused kernel against an unfused composition
- Comparing schedule variants that should be mathematically equivalent
- Verifying accumulator selection and predication logic

### 7.4 Preserved Operations

**Add, Sub, Mul (ring arithmetic):**
- `x + 0 = x`, `x - 0 = x`, `x - x = 0`, `x * 1 = x`
- Associativity, commutativity, and distributivity are preserved

**Min, Max (signed integer on payloads):**
- Idempotence: `min(x, x) = x`
- Commutativity and associativity

**Division (modular inverse):**
- `x / 1 = x`, `1 / (1 / x) = x`

**FMA:**
- `fma(a, b, c) = a * b + c` in the payload ring

**exp2:**
- `exp2(x + y) = exp2(x) * exp2(y)`, `exp2(0) = 1`

**sin, cos:**
- All standard angle addition identities are preserved
- `cos(x)^2 + sin(x)^2 = 1`

**Tagged unary ops** (log, log2, sqrt, rsqrt, erf, floor, ceil):
- Same-input-same-output determinism
- Different ops produce different tags

**Casts and format conversions:**
- `0`, `+1`, `-1` remain stable across conversions
- Upcast followed by downcast is identity

**MMA and tensor memory operations:**
- Exact matrix-multiply algebra over the payload ring
- Payload preservation across tensor memory loads, stores, copies

### 7.5 Limitations

- FpSan is **not** an IEEE simulator
- Does not preserve real floating-point ordering, rounding, NaN propagation, infinities, or subnormals
- Does not preserve real transcendental semantics for `log`, `sqrt`, `erf`, `floor`, `ceil`, `rsqrt`
- Results should only be compared against other FpSan results, never against ordinary floating-point outputs
- Does not apply in interpreter mode
- On AMD, only supported for `gfx942`, `gfx950`, and `gfx1250`

---

## 8. Debugging Operations

Triton provides four built-in debugging operations: two for compile-time and two for runtime.

### 8.1 `static_print`

Prints values at compile time. Works like Python's `print` but evaluated during JIT compilation. This is useful for inspecting constexpr values and compile-time constants.

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This prints during compilation, not at runtime
    tl.static_print(f"BLOCK_SIZE={BLOCK_SIZE}")
    tl.static_print("Number of warps:", 4, sep=" | ")
```

Parameters are the same as Python's `print`: `*values`, `sep=" "`, `end="\n"`, `file=None`, `flush=False`.

Output appears during the compilation phase:
```
BLOCK_SIZE=1024
Number of warps: | 4
```

### 8.2 `static_assert`

Asserts a condition at compile time. Unlike `device_assert`, it does not require `TRITON_DEBUG` to be set. This is useful for validating constexpr parameters and compile-time invariants.

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, N, BLOCK_SIZE: tl.constexpr):
    tl.static_assert(BLOCK_SIZE == 1024, "BLOCK_SIZE must be 1024")
    tl.static_assert(BLOCK_SIZE <= 4096)  # Optional message
```

If the assertion fails, compilation fails with an error message.

### 8.3 `device_print`

Prints tensor or scalar values at runtime from the GPU. The first argument must be a string prefix; subsequent arguments can be tensors or scalars.

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Print a single value
    tl.device_print("pid", pid)

    # Print a tensor
    x = tl.load(x_ptr + offs, mask=offs < N)
    tl.device_print("x_values", x)

    # Print in hex format
    tl.device_print("x_hex", x, hex=True)

    # Python builtin print maps to device_print
    print("offsets", offs)
```

**Important notes about `device_print`:**

1. CUDA `printf` uses a buffer of limited size (default ~6912 KiB). If you have many program instances printing, some output may be dropped.

2. To increase the printf buffer size:

```python
triton.runtime.driver.active.utils.set_printf_fifo_size(size_bytes)
```

Call this before running any kernel that uses `device_print`. CUDA may raise an error if you try to change the size after running a kernel that uses printf.

3. The `set_printf_fifo_size` call may only affect the current device. For multi-GPU setups, call it for each device.

### 8.4 `device_assert`

Asserts a condition at runtime from the GPU. **Requires** `TRITON_DEBUG=1` to have any effect. Without it, the assertion is compiled to a no-op.

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Assert with a message
    tl.device_assert(pid < 100, "pid out of range")

    # Assert with a mask
    x = tl.load(x_ptr + offs, mask=offs < N)
    tl.device_assert(x >= 0, "x must be non-negative", mask=offs < N)

    # Python builtin assert maps to device_assert (second arg must be string)
    assert pid == 0, "pid != 0"
```

Run with:
```bash
TRITON_DEBUG=1 python my_kernel.py
```

Parameters:
- `cond`: A boolean tensor condition to assert
- `msg`: A string literal message printed if the assertion fails
- `mask`: Optional mask tensor (assertion only checked where mask is true)

### 8.5 `debug_barrier`

Inserts an explicit synchronization barrier for all threads in a block. This is useful for debugging race conditions by forcing synchronization points.

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Write to x
    tl.store(x_ptr + offs, tl.full([BLOCK_SIZE], pid, tl.float32))

    # Synchronize before reading
    tl.debug_barrier()

    # Read from x (guaranteed to see all writes)
    x = tl.load(x_ptr + offs)
```

Note: `debug_barrier` synchronizes threads within a single thread block, not across blocks.

---

## 9. Pipeline Inspection Hook

The `add_stages_inspection_hook` is a runtime hook that allows inspection and modification of the compilation pipeline stages. It is the most powerful hook for compiler debugging and development.

### 9.1 Hook Interface

The hook follows a two-phase protocol:

**Phase 1 (called with no arguments):** Returns a key and hash for cache invalidation.

**Phase 2 (called with stages, options, language, capability):** Can inspect and modify the stages dict.

```python
import triton
import hashlib
import pathlib

def get_key():
    return pathlib.Path(__file__).read_text()

def get_hash():
    return hashlib.sha256(get_key().encode('utf-8')).hexdigest()

def inspect_stages_hook(self=None, stages=None, options=None, language=None, capability=None):
    # Phase 1: Return key and hash for cache invalidation
    if all(arg is None for arg in (stages, options, language, capability)):
        return get_key(), get_hash()

    # Phase 2: Inspect or modify stages
    original_make_ttgir = stages["ttgir"]

    def wrapped_make_ttgir(src, metadata):
        print("Making TTGIR...")
        result = original_make_ttgir(src, metadata)
        print("TTGIR generated successfully")
        return result

    stages["ttgir"] = wrapped_make_ttgir

# Register the hook
triton.knobs.runtime.add_stages_inspection_hook = inspect_stages_hook
```

### 9.2 Modifying Pipeline Stages

You can replace any stage with a custom implementation:

```python
def inspect_stages_hook(self=None, stages=None, options=None, language=None, capability=None):
    if all(arg is None for arg in (stages, options, language, capability)):
        return "", "no-cache-key"

    # Replace the make_ttgir stage with a custom version
    def custom_make_ttgir(src, metadata):
        # Use the backend's original implementation
        result = self.make_ttgir(src, metadata, options, capability)
        # Post-process the result
        print(f"TTGIR for {metadata.get('name', 'unknown')} generated")
        return result

    stages["ttgir"] = custom_make_ttgir
```

### 9.3 Combining with Reproducer

```python
import triton
import hashlib
import pathlib

def get_key():
    return pathlib.Path(__file__).read_text()

def get_hash():
    return hashlib.sha256(get_key().encode('utf-8')).hexdigest()

def inspect_stages_hook(self=None, stages=None, options=None, language=None, capability=None):
    if all(arg is None for arg in (stages, options, language, capability)):
        return get_key(), get_hash()

    # Wrap each stage to log timing
    for stage_name in list(stages.keys()):
        original_stage = stages[stage_name]

        def make_wrapper(name, orig):
            def wrapper(src, metadata):
                import time
                start = time.time()
                result = orig(src, metadata)
                elapsed = time.time() - start
                print(f"Stage {name}: {elapsed:.3f}s")
                return result
            return wrapper

        stages[stage_name] = make_wrapper(stage_name, original_stage)

triton.knobs.runtime.add_stages_inspection_hook = inspect_stages_hook
```

---

## 10. Address Sanitizer

Triton supports GPU AddressSanitizer for detecting out-of-bounds memory accesses on AMD GPUs.

### 10.1 Enabling ASAN

```bash
TRITON_ENABLE_ASAN=1 python my_kernel.py
```

### 10.2 How It Works

When enabled on AMD:

1. The `+xnack` target feature is set on the kernel function
2. The ASAN runtime library (`asanrtl.bc`) is linked alongside `ocml.bc` and `ockl.bc`
3. The ASAN attribute is added to the kernel function
4. Memory accesses are instrumented with bounds checks

On AMD, the compiler:
- Sets the `amdgpu-xnack` target feature
- Links `asanrtl.bc`, `ocml.bc`, and `ockl.bc` from the backend's `lib/` directory
- Adds the ASAN attribute to the kernel function via `add_fn_asan_attr()`

### 10.3 Using with ROCm

For more information on GPU AddressSanitizer, see the [LLVM AddressSanitizer documentation for ROCm](https://rocm.docs.amd.com/projects/llvm-project/en/latest/conceptual/using-gpu-sanitizer.html).

---

## 11. Concurrency Sanitizer (ConSan)

ConSan is an experimental instrumentation mode for detecting data races and concurrency issues in GPU kernels.

### 11.1 Enabling ConSan

```bash
TRITON_ENABLE_EXPERIMENTAL_CONSAN=1 python my_kernel.py
```

Or via the instrumentation mode:
```bash
TRITON_INSTRUMENTATION_MODE=consan python my_kernel.py
```

### 11.2 Implementation

ConSan is implemented as an MLIR pass that instruments memory operations to detect conflicting accesses. It has backend-specific hooks:
- NVIDIA: `ConSanNVIDIA` hooks registered for the `nvidia` backend
- The pass uses `ConSanTargetHooks` to provide backend-specific memory effect information

When ConSan is active, ptxas is invoked with `-Ofc mid` for ConSan code generation.

---

## 12. Third-Party Debugging Tools

### 12.1 NVIDIA Compute Sanitizer

For debugging on NVIDIA GPUs, `compute-sanitizer` checks for data races and memory access issues:

```bash
compute-sanitizer python my_kernel.py
```

This runs the Triton program under NVIDIA's compute sanitizer, which detects:
- Out-of-bounds memory accesses
- Race conditions
- Uninitialized memory reads
- Other memory errors

### 12.2 AMD GPU Sanitizer

For debugging on AMD GPUs, the LLVM AddressSanitizer for ROCm can be used:

```bash
TRITON_ENABLE_ASAN=1 python my_kernel.py
```

### 12.3 triton-viz

For detailed visualization of memory access patterns in Triton programs, [triton-viz](https://github.com/Deep-Learning-Profiling-Tools/triton-viz) provides GPU-agnostic visualization of memory operations.

### 12.4 cuda-gdb / roc-gdb

When line info is enabled (default unless `TRITON_DISABLE_LINE_INFO=1`), you can use GPU debuggers:

```bash
# NVIDIA
cuda-gdb --args python my_kernel.py

# AMD
rocgdb --args python my_kernel.py
```

For best debugging experience with source-level information:
```bash
# Disable ptxas optimizations for cleaner debug info
DISABLE_PTXAS_OPT=1 python my_kernel.py
```

---

## Appendix A: Quick Reference -- Common Debugging Scenarios

### Kernel produces wrong results

```bash
# Step 1: Run in interpreter to check logic
TRITON_INTERPRET=1 python my_kernel.py

# Step 2: Add device_print to check runtime values
# (add tl.device_print calls to kernel)

# Step 3: Check for numerical issues
TRITON_INSTRUMENTATION_MODE=fpsan python my_kernel.py
```

### Kernel crashes during compilation

```bash
# Get reproducers for each stage
TRITON_ALWAYS_COMPILE=1 TRITON_REPRODUCER_PATH=/tmp/repro python my_kernel.py

# Dump IR at each pass
MLIR_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 python my_kernel.py 2>&1 | tee dump.log

# Show full tracebacks
TRITON_FRONT_END_DEBUGGING=1 python my_kernel.py
```

### Kernel crashes at runtime

```bash
# Use compute sanitizer (NVIDIA)
compute-sanitizer python my_kernel.py

# Use ASAN (AMD)
TRITON_ENABLE_ASAN=1 python my_kernel.py

# Add device_assert checks
TRITON_DEBUG=1 python my_kernel.py
```

### Performance investigation

```bash
# See which MMA version was selected
MLIR_ENABLE_DIAGNOSTICS=remarks python my_kernel.py

# Time each compilation stage
MLIR_ENABLE_TIMING=1 python my_kernel.py

# Print autotuning results
TRITON_PRINT_AUTOTUNING=1 python my_kernel.py

# Inspect generated PTX
NVPTX_ENABLE_DUMP=1 python my_kernel.py

# Inspect all IR stages
TRITON_KERNEL_DUMP=1 TRITON_ALWAYS_COMPILE=1 python my_kernel.py
```

### Experimenting with IR

```bash
# Step 1: Dump IR
TRITON_KERNEL_DUMP=1 TRITON_ALWAYS_COMPILE=1 python my_kernel.py

# Step 2: Copy and edit IR
cp ~/.triton/dump/my_kernel.ttgir ~/.triton/override/my_kernel.ttgir
# Edit the file...

# Step 3: Run with override
TRITON_KERNEL_OVERRIDE=1 TRITON_ALWAYS_COMPILE=1 python my_kernel.py
```

### Debugging autotuning

```bash
# Print autotuning decisions
TRITON_PRINT_AUTOTUNING=1 python my_kernel.py

# Cache autotuning results
TRITON_CACHE_AUTOTUNING=1 python my_kernel.py
```

---

## Appendix B: Environment Variable Cache Behavior

Environment variables are classified as either **cache-invalidating** or **cache-neutral**. Changing a cache-invalidating variable changes the compilation output and therefore invalidates any cached binaries. Cache-neutral variables affect runtime behavior or debugging output without changing the compiled code.

**Cache-invalidating variables** (changing these forces recompilation):

```
AMDGCN_ENABLE_DUMP            MLIR_ENABLE_DIAGNOSTICS
AMDGCN_USE_BUFFER_ATOMICS     MLIR_ENABLE_DUMP
AMDGCN_USE_BUFFER_OPS         MLIR_ENABLE_TIMING
ALLOW_LHS_TMEM_LAYOUT_CONV    MLIR_DISABLE_MULTITHREADING
AMDGCN_ANALYZE_SMALL_TENSOR   NVPTX_ENABLE_DUMP
DISABLE_LLVM_OPT              TRITON_DEFAULT_FP_FUSION
DISABLE_MMA_V3                TRITON_DISABLE_LINE_INFO
DISABLE_MMA_V5                TRITON_DUMP_MIR
DISABLE_PTXAS_OPT             TRITON_ENABLE_ASAN
LLVM_EXTRACT_DI_LOCAL_VAR     TRITON_ENABLE_EXPERIMENTAL_CONSAN
LLVM_IR_ENABLE_DUMP           TRITON_ENABLE_LLVM_DEBUG
LLVM_ENABLE_TIMING            TRITON_F32_DEFAULT
LLVM_PASS_PLUGIN_PATH         TRITON_HIP_USE_ASYNC_COPY
TRITON_HIP_USE_BLOCK_PINGPONG TRITON_OVERRIDE_ARCH
TRITON_HIP_USE_IN_THREAD_TR   TRITON_PARTITION_SCHEDULING_*
TRITON_PLUGIN_PATHS           TRITON_PLUGIN_VERSION_CHECK
TRITON_PREFER_TMEM_16x256     USE_IR_LOC
TRITON_LLVM_DEBUG_ONLY
```

**Cache-neutral variables** (changing these does NOT force recompilation):

```
TRITON_REPRODUCER_PATH        TRITON_ENABLE_PYTHON_STACKTRACE
TRITON_ALWAYS_COMPILE         TRITON_KERNEL_DUMP
TRITON_DUMP_DIR               TRITON_KERNEL_OVERRIDE
TRITON_OVERRIDE_DIR           TRITON_CACHE_DIR
TRITON_HOME                   TRITON_PRINT_AUTOTUNING
TRITON_CACHE_AUTOTUNING       TRITON_DEBUG
TRITON_INTERPRET              TRITON_DEFAULT_BACKEND
```
