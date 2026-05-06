# 22. Debugging Tools

TileLang provides a comprehensive suite of debugging tools for diagnosing generation issues,
correctness problems, and performance regressions. This reference covers every debugging facility
available in the TileLang ecosystem.

---

## Table of Contents

1. [Debugging Categories](#debugging-categories)
2. [T.print for Runtime Inspection](#tprint-for-runtime-inspection)
3. [IR Inspection Techniques](#ir-inspection-techniques)
4. [Post-Processing Callbacks](#post-processing-callbacks)
5. [AutoDD (Automatic Delta Debugging)](#autodd-automatic-delta-debugging)
6. [Visual Layout Inference](#visual-layout-inference)
7. [Verbose Mode in JIT Compilation](#verbose-mode-in-jit-compilation)
8. [Source Code Inspection](#source-code-inspection)
9. [PTX and SASS Inspection](#ptx-and-sass-inspection)
10. [Pass Debugging](#pass-debugging)
11. [Data Race Checking](#data-race-checking)
12. [Semantic Checks](#semantic-checks)
13. [Logging Configuration](#logging-configuration)
14. [Common Error Patterns and Solutions](#common-error-patterns-and-solutions)
15. [Out-of-Bounds Access Detection](#out-of-bounds-access-detection)
16. [Tensor Validation](#tensor-validation)

---

## Debugging Categories

TileLang kernel development issues fall into three broad categories:

### 1. Generation Issues

These occur during the TileLang-to-TIR lowering or code generation phase:

- **Syntax errors** in TileLang programs
- **Type mismatches** between tensor dtypes and operations
- **Shape incompatibilities** in tensor operations
- **Invalid scoping** of memory allocations (e.g., shared memory used outside kernel)
- **Unsupported operations** for the target backend

### 2. Correctness Issues

These occur when the kernel compiles but produces incorrect results:

- **Incorrect indexing** in tensor access patterns
- **Race conditions** in shared memory access
- **Incorrect reduction** logic
- **Precision loss** from insufficient accumulation dtypes
- **Boundary condition errors** in masked operations

### 3. Performance Issues

These occur when the kernel is correct but slower than expected:

- **Suboptimal tiling** (block sizes too small or too large)
- **Insufficient pipelining** (num_stages too low)
- **Poor memory access patterns** (non-coalesced reads/writes)
- **Excessive shared memory bank conflicts**
- **Underutilized tensor cores**

---

## T.print for Runtime Inspection

The `T.print` directive allows printing tensor values, indices, and expressions at kernel
runtime. This is the primary tool for inspecting intermediate values during kernel execution.

### Basic Usage

```python
import tilelang.language as T

@T.prim_func
def debug_kernel(A: T.Tensor((128, 128), T.float16), ...):
    with T.Kernel(1, 1, threads=128) as (bx, by):
        A_shared = T.alloc_shared((32, 32), T.float16)

        # Print a simple message
        T.print("Entering kernel block", bx, by)

        # Print tensor values at specific indices
        T.print("A[0,0] =", A[0, 0])

        # Print shared memory values
        T.print("A_shared[0,0] =", A_shared[0, 0])
```

### Printing Expressions

```python
@T.prim_func
def debug_expressions(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
    with T.Kernel(1, threads=128) as (bx):
        tid = T.get_thread_binding()
        T.print("Thread", tid, "A[", tid, "] =", A[tid], "B[", tid, "] =", B[tid])

        # Print computed expressions
        result = A[tid] + B[tid]
        T.print("A[tid] + B[tid] =", result)
```

### Conditional Printing

```python
@T.prim_func
def debug_conditional(A: T.Tensor((1024,), T.float32)):
    with T.Kernel(1, threads=256) as (bx):
        tid = T.get_thread_binding()
        # Only print for specific threads to avoid output flooding
        if tid == 0:
            T.print("First thread processing")
        if A[tid] < 0:
            T.print("Negative value at index", tid, "value =", A[tid])
```

### Important Notes

- `T.print` adds printf calls to the generated CUDA/HIP code
- Excessive printing can significantly slow down kernel execution
- Output may be interleaved across threads; use thread ID checks for clarity
- Some backends may buffer printf output; call `cudaDeviceSynchronize()` to flush

---

## IR Inspection Techniques

Inspecting the intermediate representation (TIR) before and after compiler transforms is
essential for understanding how TileLang processes your kernel.

### Viewing TIR Before Transforms

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def my_kernel(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def gemm(A: T.Tensor((M, K), T.float16), B: T.Tensor((K, N), T.float16),
             C: T.Tensor((M, N), T.float16)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # ... kernel body ...
    return gemm

# Get the PrimFunc (before lowering)
kernel = my_kernel(1024, 1024, 1024, 128, 128, 32)
print(kernel.prim_func)
```

### Viewing TIR After Transforms

```python
# The compiled artifact contains the lowered TIR
artifact = kernel.artifact
if artifact is not None:
    print("Lowered IR:")
    print(artifact)
```

### Using the AST Printer

The `ASTPrinter` pass renders the TileLang AST hierarchy in a visual tree format:

```python
from tilelang.analysis import ASTPrinter

# Enable via pass config
kernel = tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_AST_PRINT_ENABLE: True,
    },
)(my_func)(M, N, K, block_M, block_N, block_K)
```

This produces output like:

```
PrimFunc(params=[A, B, C], ret_type=None, buffer_map={...}, attrs={...})
└── body=
    └── BlockRealize
        ├── iter_values: [bx, by]
        └── block=
            └── Block
                ├── iter_vars: [i, j]
                ├── reads: [A[...], B[...]]
                ├── writes: [C[...]]
                ├── allocations: [A_shared, B_shared, C_local]
                └── body=
                    └── SeqStmt
                        ├── seq0(Stmt): For
                        │   ├── loop_var: k
                        │   └── body= ...
                        └── seq1(Stmt): BufferStore
```

### Comparing IR Before and After Specific Passes

```python
import tvm

# Dump IR for all passes
pass_configs = {
    tilelang.PassConfigKey.TL_ENABLE_DUMP_IR: True,
    tilelang.PassConfigKey.TL_DUMP_IR_DIR: "./dump_ir",
}

with tvm.transform.PassContext(opt_level=3, config=pass_configs):
    mod = tilelang.lower(prim_func, target="cuda")
```

---

## Post-Processing Callbacks

Post-processing callbacks intercept generated source code after TileLang's code generation but
before compilation. They are useful for injecting custom code, modifying kernels, or debugging
generated output.

### register_cuda_postproc_callback

```python
from tilelang.engine import register_cuda_postproc_callback

@register_cuda_postproc_callback
def my_cuda_postproc(code: str, target) -> str:
    """Intercept and modify CUDA kernel source before compilation."""
    # Add custom includes
    code = '#include "my_custom_header.cuh"\n' + code

    # Replace specific patterns
    code = code.replace("// original comment", "// modified comment")

    # Print the generated code for debugging
    print("Generated CUDA code:")
    print(code[:500])  # Print first 500 chars

    return code
```

### register_hip_postproc_callback

```python
from tilelang.engine import register_hip_postproc_callback

@register_hip_postproc_callback
def my_hip_postproc(code: str, target) -> str:
    """Intercept and modify HIP kernel source before compilation."""
    # HIP-specific modifications
    code = code.replace("__umulhi", "__ockl_umulhi")
    return code
```

### register_c_postproc_callback

```python
from tilelang.engine import register_c_postproc_callback

@register_c_postproc_callback
def my_c_postproc(code: str, target) -> str:
    """Intercept and modify C host code before compilation."""
    return code
```

### register_metal_postproc_callback

```python
from tilelang.engine import register_metal_postproc_callback

@register_metal_postproc_callback
def my_metal_postproc(code: str, target) -> str:
    """Intercept and modify Metal shader source before compilation."""
    return code
```

### Functional API (without decorators)

```python
from tilelang.engine import (
    register_cuda_postproc,
    register_hip_postproc,
    register_c_postproc,
    register_metal_postproc,
)

def modify_code(code: str, target) -> str:
    return code

# Register with optional override control
register_cuda_postproc(modify_code, override=True)
```

### Practical Example: Injecting Debug Printfs

```python
@register_cuda_postproc_callback
def inject_debug_printfs(code: str, target) -> str:
    """Add debug printf to kernel entry."""
    # Find the kernel function and add a printf at the beginning
    lines = code.split('\n')
    modified = []
    in_kernel = False
    for line in lines:
        modified.append(line)
        if '__global__' in line and not in_kernel:
            in_kernel = True
        elif in_kernel and '{' in line:
            modified.append('  printf("Kernel launched: block=(%d,%d,%d) thread=(%d,%d,%d)\\n",'
                          ' blockIdx.x, blockIdx.y, blockIdx.z,'
                          ' threadIdx.x, threadIdx.y, threadIdx.z);')
            in_kernel = False
    return '\n'.join(modified)
```

---

## AutoDD (Automatic Delta Debugging)

AutoDD is TileLang's automatic delta debugging tool for minimizing buggy kernel programs.
Given a kernel that exhibits a bug (e.g., compilation error, runtime crash), AutoDD
systematically removes code to produce the smallest possible program that still reproduces
the issue.

### Command-Line Usage

```bash
python -m tilelang.autodd source.py --err-msg "Error message to look for" -o minimized.py
```

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `source` | required | Input Python source file |
| `--err-msg` | required | Error message to look for in output |
| `-o, --output` | required | Output file path for minimized source |
| `--backend` | `"runner"` | Backend: `"runner"` (fast) or `"subproc"` (stable) |
| `--timeout` | `60` | Timeout per task in seconds |
| `-j, --jobs` | `1` | Number of parallel jobs |

### Programmatic Usage

```python
from tilelang.autodd import ASTPDD, LinePDD, Ruff, ParTaskManager
from pathlib import Path
import asyncio

async def minimize(source: str, err_msg: str):
    manager = ParTaskManager(
        err_msg=err_msg,
        text=source,
        output_file=Path("minimized.py"),
        timeout=60,
        num_workers=4,
    )

    # Apply AST-level delta debugging
    task_manager = ASTPDD.from_source(source)
    await manager.run_async(task_manager)
    return manager.text
```

### Freeze Regions

Protect critical code from being removed during minimization:

```python
from tilelang.autodd import __freeze__

# Method 1: Block form with context manager
with __freeze__:
    critical_setup()

# Method 2: Expression form
result = __freeze__(essential_computation())

# Method 3: Comment annotations (converted automatically)
# autodd: freeze-start
important_code_here()
more_important_code()
# autodd: end-freeze

single_statement()  # autodd: freeze
```

### How AutoDD Works

1. **AST Parsing**: The source is parsed into an AST
2. **Rewrite Attachment**: Every AST node is annotated with possible rewrites (removals, simplifications)
3. **Probabilistic Delta Debugging (PDD)**: An iterative algorithm that:
   - Generates candidate subsets of rewrites to apply
   - Applies them to produce a new program
   - Tests whether the bug still reproduces
   - Updates probabilities based on results
4. **Multi-Round Reduction**: Applies increasingly aggressive rewrite strategies:
   - Fast reducers: Statement removal, if-branch elimination
   - Canonicalizers: With-binding simplification, argument attachment
   - Simplifiers: Assignment RHS replacement, binary operation forwarding
   - Slow reducers: Expression removal, keyword removal

---

## Visual Layout Inference

The `LayoutVisual` pass visualizes fragment layouts inferred during compilation, helping
understand how TileLang maps logical tensor indices to physical thread and register locations.

### Enabling Layout Visualization

```python
import tilelang

# Enable with specific format
kernel = tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True,
        tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS: "png",
    },
)(my_func)(...)
```

### Supported Formats

| Format | Value | Best For |
|--------|-------|----------|
| PNG | `"png"` | Quick inspection |
| PDF | `"pdf"` | Documentation |
| SVG | `"svg"` | Web/vector graphics |
| All | `"all"` | Generate all formats |
| Multiple | `"png,svg"` | Comma-separated list |

### Output Description

For each inferred layout, the pass outputs:

```
A_shared inferred layout:
  Shape: (128, 32) -> (128, 32)
  Thread: lambda i, j: i * 4 + j // 8
  Index:  lambda i, j: (i % 4) * 8 + j % 8
  Replicate: 1
```

And generates a color-coded plot showing:
- **Rows**: Thread IDs
- **Columns**: Register indices within each thread
- **Colors**: Mapping from logical (i, j) positions

### Programmatic Usage

```python
from tilelang.analysis import LayoutVisual, print_fragment_format
import tilelang.language as T

# The LayoutVisual pass can be used directly
layout = T.Fragment((128, 32), dtype=T.float16)
print_fragment_format(layout)
```

---

## Verbose Mode in JIT Compilation

Verbose mode provides detailed logging during the compilation process.

### Enabling Verbose Mode

```python
# Method 1: JIT parameter
kernel = tilelang.jit(out_idx=[-1], verbose=True)(my_func)(...)

# Method 2: Environment variable
import os
os.environ["TILELANG_VERBOSE"] = "1"

# Method 3: AutoTuner compile args
autotuner.set_compile_args(verbose=True)
```

### Verbose Output Includes

- Compilation target and backend resolution
- Pass execution order and timing
- Memory allocation details
- Generated code statistics
- Cache hit/miss information

---

## Source Code Inspection

TileLang provides several methods to inspect generated source code at different levels.

### show_source()

Print the generated kernel or host source code to stdout:

```python
kernel = my_jit_func(M, N, K, ...)

# Print device kernel source
kernel.show_source()                # Default: kernel source only
kernel.show_source("host")          # Print host source
kernel.show_source("both")          # Print both kernel and host source
```

### get_kernel_source()

Get the generated device kernel source as a string:

```python
source = kernel.get_kernel_source()
print(source)

# Get without host wrapper
source = kernel.get_kernel_source(kernel_only=True)

# Get including host wrapper
source = kernel.get_kernel_source(kernel_only=False)
```

### get_host_source()

Get the host-side wrapper code:

```python
host_source = kernel.get_host_source()
print(host_source)
```

### export_sources()

Export all generated sources to files:

```python
# The kernel source is also accessible via the adapter
if kernel.adapter is not None:
    print("Kernel source path:", kernel.adapter.libpath)
    print("Kernel source:", kernel.adapter.get_kernel_source())
```

---

## PTX and SASS Inspection

For low-level debugging, TileLang provides access to PTX (intermediate assembly) and SASS
(machine assembly).

### Inspecting PTX

```python
# Get the compiled module and extract PTX
# Note: PTX availability depends on the execution backend
if kernel.execution_backend == "tvm_ffi":
    device_mod = kernel.artifact.device_mod
    if device_mod is not None:
        # Save PTX to file
        ptx_source = device_mod.get_source("ptx")
        print(ptx_source)

# Via pass configuration for verbose PTXAS output
kernel = tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_PTXAS_VERBOSE_OUTPUT: True,
    },
)(my_func)(...)
```

### PTXAS Register Usage Control

```python
# Control register usage level (0-10)
kernel = tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 5,
    },
)(my_func)(...)
```

### Inspecting SASS

Use NVIDIA's `cuobjdump` tool to inspect SASS:

```bash
# After exporting the compiled kernel
cuobjdump -sass kernel.cubin > kernel.sass
```

---

## Pass Debugging

TileLang's compiler uses a sequence of passes to transform and optimize the IR. These passes
can be debugged individually.

### TL_AST_PRINT_ENABLE

Print the AST structure of the TIR:

```python
kernel = tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_AST_PRINT_ENABLE: True,
    },
)(my_func)(...)
```

This prints a tree-structured representation of the TIR AST showing:
- Function parameters and return types
- Block structure with allocations
- Loop nesting and iteration variables
- Buffer access patterns

### TL_ENABLE_DUMP_IR

Dump IR at every pass boundary:

```python
kernel = tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_DUMP_IR: True,
        tilelang.PassConfigKey.TL_DUMP_IR_DIR: "./my_dump_ir",
    },
)(my_func)(...)
```

This creates files in the specified directory, one per pass:

```
my_dump_ir/
  00_input.tir
  01_Simplify.tir
  02_StorageRewrite.tir
  03_LowerTileOp.tir
  04_LowerInitBlock.tir
  ...
```

Each file contains the TIR state after the corresponding pass, allowing you to trace how
transforms affect the IR.

### Pass Execution Order

The typical pass order for CUDA compilation includes:

1. **Simplify**: Arithmetic and algebraic simplifications
2. **StorageRewrite**: Memory allocation optimization
3. **LowerTileOp**: Lower tile operations to explicit loops
4. **LowerInitBlock**: Lower initialization blocks
5. **InjectSoftwarePipeline**: Software pipelining
6. **VectorizePlanner**: Vectorization decisions
7. **LowerVectorize**: Apply vectorization
8. **LowerWarpSpecialized**: Warp specialization
9. **ThreadStorageSync**: Insert synchronization barriers
10. **LegalizeKeyBuffer**: Buffer access legalization

---

## Data Race Checking

TileLang includes data race detection to catch concurrent access issues in shared memory.

### TL_DISABLE_DATA_RACE_CHECK

By default, TileLang checks for data races during compilation. This can be disabled:

```python
# Disable data race checking
kernel = tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)(my_func)(...)
```

### Common Data Race Patterns

```python
# Race condition: Multiple threads writing to the same shared memory location
with T.Kernel(1, 1, threads=256) as (bx, by):
    shared = T.alloc_shared((128,), T.float16)

    # RACE: All threads write to shared[0]
    shared[0] = T.get_thread_binding()

    # Correct: Each thread writes to its own location
    tid = T.get_thread_binding()
    shared[tid] = some_value
```

### What the Data Race Checker Detects

- Unsynchronized writes to the same shared memory location
- Read-write conflicts without barriers
- Missing `T.copy()` synchronization when accessing shared buffers
- Warp-level race conditions in fragment operations

---

## Semantic Checks

TileLang performs pre-lowering semantic checks to catch common programming errors before
expensive compilation.

### TL_DISABLE_PRELOWER_SEMANTIC_CHECK

```python
# Disable pre-lower semantic checks (not recommended)
kernel = tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_PRELOWER_SEMANTIC_CHECK: True,
    },
)(my_func)(...)
```

### What Semantic Checks Validate

1. **Shape compatibility**: Tensor shapes match operations
2. **Dtype consistency**: Operations use compatible data types
3. **Memory scope correctness**: Buffers are accessed in valid scopes
4. **Kernel dimension validity**: Grid and block dimensions are positive
5. **Buffer bound checks**: Access indices are within buffer bounds
6. **Allocation scope**: Shared memory allocated within kernel scope

---

## Logging Configuration

TileLang uses Python's `logging` module for configurable output.

### Setting Log Level

```python
import tilelang

# Set via module API
tilelang.set_log_level("DEBUG")   # Most verbose
tilelang.set_log_level("INFO")    # Normal operation
tilelang.set_log_level("WARNING") # Only warnings and errors
tilelang.set_log_level("ERROR")   # Only errors
```

### TVM_LOG_DEBUG

TVM (which TileLang is built on) provides additional debug logging:

```bash
# Enable TVM debug logging
export TVM_LOG_DEBUG=1

# Or use VLOG for verbose logging
export TVM_LOG_DEBUG="*/tvm/**:DEBUG"
```

### Verbose Logging in Autotuner

The autotuner writes detailed logs to `autotuner.log`:

```python
# The autotuner creates its own log file
# Location: ./autotuner.log (in the current working directory)
# Contains:
#   - Compilation start/end for each configuration
#   - Benchmark results for each configuration
#   - Error messages for failed configurations
#   - Cache hit/miss information
```

---

## Common Error Patterns and Solutions

### 1. "Shared memory allocation exceeds device limit"

**Symptom**: Compilation fails with shared memory allocation error.

**Solution**: Reduce block sizes or pipeline stages:

```python
# Before (too much shared memory)
block_M, block_N, block_K = 256, 256, 64
num_stages = 3
# Total shared: 3 * (256*64 + 64*256) * 2 bytes = 196 KB (exceeds 48KB typical limit)

# After (reduced)
block_M, block_N, block_K = 128, 128, 32
num_stages = 2
# Total shared: 2 * (128*32 + 32*128) * 2 bytes = 32 KB
```

### 2. "Invalid execution backend"

**Symptom**: ValueError when specifying execution backend.

**Solution**: Use a valid backend name:

```python
# Valid backends:
# "auto" - Auto-select based on target
# "tvm_ffi" - TVM FFI with DLPack
# "cython" - Cython wrapper (requires C++ compiler)
# "nvrtc" - NVRTC runtime compilation
# "torch" - PyTorch integration (Metal target)
```

### 3. "No configuration successfully compiled"

**Symptom**: Auto-tuner fails because all configurations had errors.

**Solution**:
- Check that block sizes divide correctly into problem dimensions
- Ensure shared memory requirements are within limits
- Verify that the kernel function signature matches configuration keys
- Enable verbose mode for detailed error messages

### 4. "Results differ significantly"

**Symptom**: Correctness check fails during auto-tuning.

**Solution**:
- Increase tolerance (`rtol`, `atol`)
- Check accumulation dtype (use `T.float32` for FP16/BF16 kernels)
- Verify the reference program matches the kernel's computation
- Use `TensorSupplyType.Randn` instead of `Rand` for more realistic inputs

### 5. "Compilation timeout"

**Symptom**: Auto-tuner times out during compilation.

**Solution**:
- Increase the timeout: `autotuner.run(timeout=120)`
- Reduce the configuration space
- Use the Roller for device-aware configuration pruning
- Check for excessively large kernel code generation

### 6. NaN or Inf in Output

**Symptom**: Kernel produces NaN or infinite values.

**Solution**:
```python
# Use appropriate accumulation dtype
accum_dtype = T.float32  # Always use float32 for accumulation

# Scale attention scores properly
scale = (1.0 / dim) ** 0.5 * 1.44269504  # Use log2(e) for numerical stability

# Use safe softmax
for i, j in T.Parallel(block_M, block_N):
    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
```

---

## Out-of-Bounds Access Detection

TileLang can detect and warn about potential out-of-bounds memory accesses.

### TL_DISABLE_OUT_OF_BOUND_WARNING

```python
# By default, OOB warnings are disabled (default: True means warnings are suppressed)
# To enable OOB warnings:
kernel = tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_OUT_OF_BOUND_WARNING: False,
    },
)(my_func)(...)
```

### Safe Memory Access

TileLang provides safe memory access patterns:

```python
# Use T.if_then_else for boundary checks
for i, j in T.Parallel(block_M, block_N):
    access_h = m % (OH * OW) // OW * S + k // (KW * C) * D - P
    access_w = m % OW * S + k // C % KW * D - P
    in_bound = (access_h >= 0) and (access_w >= 0) and (access_h < H) and (access_w < W)
    data_shared[i, j] = T.if_then_else(in_bound, data[m // (OH * OW), access_h, access_w, k % C], 0)
```

### TL_DISABLE_SAFE_MEMORY_ACCESS

```python
# Disable safe memory access legalization (use with caution)
kernel = tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    },
)(my_func)(...)
```

---

## Tensor Validation

TileLang provides utilities for validating tensor shapes, dtypes, and values.

### Shape Checking

```python
import torch

def validate_tensor(tensor, expected_shape, expected_dtype, name="tensor"):
    """Validate tensor properties."""
    if tensor.shape != expected_shape:
        raise ValueError(
            f"{name}: Expected shape {expected_shape}, got {tensor.shape}"
        )
    if tensor.dtype != expected_dtype:
        raise ValueError(
            f"{name}: Expected dtype {expected_dtype}, got {tensor.dtype}"
        )
    if tensor.device.type != "cuda":
        raise ValueError(
            f"{name}: Expected CUDA tensor, got {tensor.device}"
        )
```

### Dtype Matching

TileLang maps between its dtype system and PyTorch dtypes:

```python
import tilelang.language as T

# Convert TileLang dtype to PyTorch dtype
torch_dtype = T.dtype("float16").as_torch()   # torch.float16
torch_dtype = T.dtype("bfloat16").as_torch()  # torch.bfloat16
torch_dtype = T.dtype("float32").as_torch()   # torch.float32
torch_dtype = T.dtype("int8").as_torch()      # torch.int8
torch_dtype = T.dtype("int32").as_torch()     # torch.int32
```

### torch_assert_close

TileLang provides a utility for asserting tensor closeness:

```python
from tilelang.utils import torch_assert_close

# Use in custom validation
torch_assert_close(
    actual=output_tensor,
    expected=reference_tensor,
    rtol=1e-2,
    atol=1e-2,
)
```

### Profiler Validation

The built-in profiler provides validation methods:

```python
kernel = my_jit_func(M, N, K, ...)
profiler = kernel.get_profiler()

# Assert all close with reference
profiler.assert_allclose(
    ref_program,
    rtol=1e-2,
    atol=1e-2,
    max_mismatched_ratio=0.01,
)

# Manual assertion with custom inputs
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
profiler.assert_allclose(
    ref_program,
    input_tensors=[a, b],
    rtol=1e-2,
    atol=1e-2,
)
```

### Common Validation Patterns

```python
# Check for NaN/Inf
assert not output.isnan().any(), "Output contains NaN values"
assert not output.isinf().any(), "Output contains Inf values"

# Check value range
assert output.abs().max() < 100, "Output values unexpectedly large"

# Compute relative error
ref_output = ref_program(a, b)
rel_error = (output - ref_output).abs() / (ref_output.abs() + 1e-8)
print(f"Max relative error: {rel_error.max().item()}")
print(f"Mean relative error: {rel_error.mean().item()}")
```

---

## Debugging Workflow Summary

1. **Enable verbose mode** for compilation details
2. **Use `T.print`** to inspect runtime values
3. **Check generated source** with `show_source()`
4. **Validate with reference** using `assert_allclose()`
5. **Inspect TIR** with AST printing or IR dumping
6. **Check for data races** using built-in detection
7. **Use AutoDD** to minimize reproduction cases
8. **Inspect PTX/SASS** for low-level code generation issues
9. **Visualize layouts** for tensor core mapping issues
10. **Post-process callbacks** for custom code injection
