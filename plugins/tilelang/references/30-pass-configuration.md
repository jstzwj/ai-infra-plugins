# TileLang Pass Configuration Reference

Pass configuration in TileLang controls the behavior of compilation passes that transform and optimize TIR (Tensor Intermediate Representation) programs. This document covers every pass configuration key, its purpose, default value, effect, and practical usage.

## Table of Contents

1. [Overview](#overview)
2. [PassConfigKey Enum](#passconfigkey-enum)
3. [Simplification Configs](#simplification-configs)
4. [Warp and Vectorization Controls](#warp-and-vectorization-controls)
5. [Memory Safety](#memory-safety)
6. [Compilation Flags](#compilation-flags)
7. [Debug Options](#debug-options)
8. [TIR Configs](#tir-configs)
9. [Pipeline Configs](#pipeline-configs)
10. [Memory Configs](#memory-configs)
11. [Using pass_configs with tilelang.jit and tilelang.compile](#using-pass_configs-with-tilelangjit-and-tilelangcompile)
12. [normalize_pass_configs Function](#normalize_pass_configs-function)
13. [Common Pass Config Recipes](#common-pass-config-recipes)
14. [Architecture-Specific Configurations](#architecture-specific-configurations)

---

## Overview

TileLang's compilation pipeline consists of multiple passes that progressively transform and optimize the TIR program. Each pass can be configured through a dictionary of key-value pairs passed via `pass_configs`.

**Module Location:** `tilelang.transform.pass_config`

**Main Entry Point:**

```python
from tilelang.transform.pass_config import PassConfigKey, normalize_pass_configs
```

---

## PassConfigKey Enum

The `PassConfigKey` enum defines all recognized configuration keys. Each key has a string value that serves as the dictionary key in pass configuration dictionaries.

```python
from tilelang.transform.pass_config import PassConfigKey

# Access a config key
key = PassConfigKey.TL_SIMPLIFY
print(key.value)  # "tl.Simplify"
```

---

## Simplification Configs

### TL_SIMPLIFY

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.Simplify"` |
| **Type** | `dict` |
| **Default** | All sub-options use their individual defaults |

The master configuration for TileLang simplification passes. This is a dictionary that can contain the following sub-options:

**Sub-options:**

| Sub-key | Type | Default | Description |
|---------|------|---------|-------------|
| `transitively_prove_inequalities` | `bool` | `False` | Enable transitive inequality proving during simplification |
| `convert_boolean_to_and_of_ors` | `bool` | `False` | Convert boolean expressions to AND-of-ORs form |
| `apply_constraints_to_boolean_branches` | `bool` | `False` | Apply constraints to simplify boolean branches |
| `propagate_knowns_to_prove_conditional` | `bool` | `False` | Propagate known values to prove conditionals |
| `propagate_knowns_to_simplify_expressions` | `bool` | `False` | Propagate known values to simplify expressions |
| `enable_simplify_let_inline` | `bool` | `True` | Enable inlining of let statements during simplification |

**Usage:**

```python
pass_configs = {
    PassConfigKey.TL_SIMPLIFY: {
        "enable_simplify_let_inline": False,
        "transitively_prove_inequalities": True,
    }
}
```

### TL_SIMPLIFY_TRANSITIVELY_PROVE_INEQUALITIES

| Attribute | Value |
|-----------|-------|
| **Key** | `"transitively_prove_inequalities"` |
| **Type** | `bool` |
| **Default** | `False` |

When enabled, the simplifier attempts to prove inequalities through transitive chains (e.g., if `a < b` and `b < c`, then `a < c`). This can enable more aggressive simplification of conditional expressions but increases compilation time.

### TL_SIMPLIFY_CONVERT_BOOLEAN_TO_AND_OF_ORS

| Attribute | Value |
|-----------|-------|
| **Key** | `"convert_boolean_to_and_of_ors"` |
| **Type** | `bool` |
| **Default** | `False` |

Converts boolean expressions to conjunctive normal form (AND of ORs). This can help with simplification of complex boolean conditions.

### TL_SIMPLIFY_APPLY_CONSTRAINTS_TO_BOOLEAN_BRANCHES

| Attribute | Value |
|-----------|-------|
| **Key** | `"apply_constraints_to_boolean_branches"` |
| **Type** | `bool` |
| **Default** | `False` |

Applies known constraints to simplify the branches of boolean expressions. For example, if `x > 0` is known, then `x > 0 and y > 0` simplifies to `y > 0`.

### TL_SIMPLIFY_PROPAGATE_KNOWNS_TO_PROVE_CONDITIONAL

| Attribute | Value |
|-----------|-------|
| **Key** | `"propagate_knowns_to_prove_conditional"` |
| **Type** | `bool` |
| **Default** | `False` |

Propagates known constant values to prove or disprove conditional expressions, potentially eliminating dead branches.

### TL_SIMPLIFY_PROPAGATE_KNOWNS_TO_SIMPLIFY_EXPRESSIONS

| Attribute | Value |
|-----------|-------|
| **Key** | `"propagate_knowns_to_simplify_expressions"` |
| **Type** | `bool` |
| **Default** | `False` |

Propagates known values to simplify arithmetic and other expressions throughout the TIR.

### TL_SIMPLIFY_ENABLE_LET_INLINE

| Attribute | Value |
|-----------|-------|
| **Key** | `"enable_simplify_let_inline"` |
| **Type** | `bool` |
| **Default** | `True` |

Controls whether let-bound variables are inlined during simplification. Disabling this can preserve variable names for debugging but may reduce optimization opportunities.

---

## Warp and Vectorization Controls

### TL_DISABLE_WARP_SPECIALIZED

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.disable_warp_specialized"` |
| **Type** | `bool` |
| **Default** | `False` |

Disables warp specialization optimization. When enabled (default), TileLang may partition warp groups to perform different tasks (e.g., one group loads data while another computes). Disabling this forces all warps to execute the same code path.

**When to disable:**
- Debugging warp-specific issues
- On architectures where warp specialization causes performance regressions
- When manual warp scheduling is preferred

```python
pass_configs = {
    PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}
```

### TL_DISABLE_VECTORIZE_256

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.disable_vectorize_256"` |
| **Type** | `bool` |
| **Default** | `False` |

Disables the use of LDG/STG 256-bit vectorized load/store instructions. When disabled, TileLang falls back to 128-bit or smaller vector widths.

**When to disable:**
- On architectures that do not support 256-bit memory transactions
- When 256-bit vectorization causes alignment issues
- For debugging memory access patterns

### TL_DISABLE_WGMMA

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.disable_wgmma"` |
| **Type** | `bool` |
| **Default** | `False` |

Disables usage of Hopper WGMMA (Warp Group Matrix Multiply-Accumulate) instructions. When disabled, the compiler falls back to standard MMA instructions.

**When to disable:**
- On non-Hopper architectures
- When WGMMA causes register pressure issues
- For A/B testing against MMA performance

---

## Memory Safety

### TL_DISABLE_DATA_RACE_CHECK

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.disable_data_race_check"` |
| **Type** | `bool` |
| **Default** | `False` |

Disables data race detection in TileLang. By default, TileLang checks for potential data races in shared memory and global memory access patterns. Disabling this check can reduce compilation time but may allow unsafe code to compile.

**When to disable:**
- When the programmer has verified thread safety manually
- When the data race check produces false positives
- For reducing compilation overhead in production builds

```python
pass_configs = {
    PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
}
```

### TL_DISABLE_PRELOWER_SEMANTIC_CHECK

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.disable_prelower_semantic_check"` |
| **Type** | `bool` |
| **Default** | `False` |

Disables Python-side pre-lower semantic checks. These checks validate the TIR program before lowering to device code.

### TL_DISABLE_SAFE_MEMORY_ACCESS

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.disable_safe_memory_legalize"` |
| **Type** | `bool` |
| **Default** | `False` |

Disables safe memory access legalization. When enabled (default), TileLang inserts bounds checks and padding to prevent out-of-bounds memory access. Disabling this can improve performance but may cause undefined behavior on edge cases.

**When to disable:**
- When the programmer guarantees all accesses are in-bounds
- For performance-critical kernels where bounds checks are prohibitive
- When the input shapes are always aligned to tile boundaries

### TL_DISABLE_OUT_OF_BOUND_WARNING

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.disable_out_of_bound_warning"` |
| **Type** | `bool` |
| **Default** | `True` |

Disables out-of-bound access warnings in safe memory access legalization. By default, warnings are suppressed.

### TL_STORAGE_REWRITE_DETECT_INPLACE

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.storage_rewrite_detect_inplace"` |
| **Type** | `bool` |
| **Default** | `False` |

Controls whether `StorageRewrite` can detect and exploit inplace buffer reuse. When `False` (default), distinct temporaries are kept separate. When `True`, the pass may reuse the same memory for read and write buffers when it can prove the update is safely inplace.

**Example:**

```python
# Without inplace detection:
read = T.allocate([1], T.int32, "local.var")
write = T.allocate([1], T.int32, "local.var")
write_buf[0] = read_buf[0] * 2

# With inplace detection (TL_STORAGE_REWRITE_DETECT_INPLACE=True):
read = T.allocate([1], T.int32, "local.var")
# write is aliased to read
read_buf[0] = read_buf[0] * 2
```

---

## Compilation Flags

### TL_PTXAS_REGISTER_USAGE_LEVEL

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.ptxas_register_usage_level"` |
| **Type** | `int or None` |
| **Default** | `None` |
| **Range** | `[0, 10]` |

Controls the PTXAS register usage optimization level. Higher values instruct the PTX assembler to use more aggressive optimizations that may affect register allocation.

**Level Guide:**

| Level | Behavior |
|-------|----------|
| `None` | Default PTXAS behavior |
| `0` | Minimum optimization |
| `5` | Moderate optimization |
| `10` | Maximum optimization (may increase register pressure) |

### TL_ENABLE_PTXAS_VERBOSE_OUTPUT

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.enable_ptxas_verbose_output"` |
| **Type** | `bool` |
| **Default** | `False` |

Enables verbose output from the PTX assembler. Useful for debugging register allocation and memory usage.

### TL_DEVICE_COMPILE_FLAGS

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.device_compile_flags"` |
| **Type** | `str or list[str] or None` |
| **Default** | `None` |

Additional device compiler flags passed to nvcc/NVRTC. Accepts either a string (parsed with shell-like splitting) or a list of strings.

**Common Flags:**

| Flag | Purpose |
|------|---------|
| `-I/path/to/include` | Add include path |
| `-DMY_SWITCH=1` | Define preprocessor macro |
| `--ptxas-options=--verbose` | Verbose PTXAS output |
| `--use_fast_math` | Enable fast math optimizations |
| `-Xptxas -v` | Show register usage |
| `-lineinfo` | Include line information for profiling |

**Usage:**

```python
pass_configs = {
    PassConfigKey.TL_DEVICE_COMPILE_FLAGS: [
        "--use_fast_math",
        "-Xptxas", "-v",
    ]
}
```

### TL_ENABLE_FAST_MATH

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.enable_fast_math"` |
| **Type** | `bool` |
| **Default** | `False` |

When enabled, passes `--use_fast_math` to nvcc. This enables fast but less accurate math approximations for sin, cos, exp, log, etc.

**Effect on Precision:**

| Operation | Standard | Fast Math |
|-----------|----------|-----------|
| Division | IEEE compliant | May not preserve -0.0/NaN |
| Square root | IEEE compliant | Approximate |
| Reciprocal | IEEE compliant | Approximate |
| Transcendentals | High accuracy | ~2 ULP error |

### TL_CONFIG_INDEX_BITWIDTH

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.config_index_bitwidth"` |
| **Type** | `int` |
| **Default** | `32` |

Bitwidth for configuration indices used in the compilation pipeline.

---

## Debug Options

### TL_AST_PRINT_ENABLE

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.ast_print_enable"` |
| **Type** | `bool` |
| **Default** | `False` |

Enables TIR AST printing for debugging. When enabled, the compiler prints the TIR program at various compilation stages.

```python
pass_configs = {
    PassConfigKey.TL_AST_PRINT_ENABLE: True,
}
```

### TL_LAYOUT_VISUALIZATION_ENABLE

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.layout_visualization_enable"` |
| **Type** | `bool` |
| **Default** | `False` |

Enables layout inference visualization. Generates visual representations of memory layout decisions.

### TL_LAYOUT_VISUALIZATION_FORMATS

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.layout_visualization_formats"` |
| **Type** | `str` |
| **Default** | Not set |
| **Options** | `"pdf"`, `"png"`, `"svg"`, `"all"` |

Specifies the output format for layout visualization.

### TL_ENABLE_DUMP_IR

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.enable_dump_ir"` |
| **Type** | `bool` |
| **Default** | `False` |

Enables dumping IR during lowering between passes. Each pass writes its output IR to files in the dump directory.

### TL_DUMP_IR_DIR

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.dump_ir_path"` |
| **Type** | `str` |
| **Default** | `"./dump_ir/"` |

Path to the directory where IR will be dumped when `TL_ENABLE_DUMP_IR` is `True`.

```python
pass_configs = {
    PassConfigKey.TL_ENABLE_DUMP_IR: True,
    PassConfigKey.TL_DUMP_IR_DIR: "./debug/my_kernel_ir/",
}
```

### CUDA_KERNELS_OUTPUT_DIR

| Attribute | Value |
|-----------|-------|
| **Key** | `"cuda.kernels_output_dir"` |
| **Type** | `str` |
| **Default** | `""` (empty) |

Output directory for generated CUDA kernel source files. When set, the compiled CUDA kernels are saved to this directory for inspection.

---

## TIR Configs

### TIR_ENABLE_EQUIV_TERMS_IN_CSE

| Attribute | Value |
|-----------|-------|
| **Key** | `"tir.enable_equiv_terms_in_cse_tir"` |
| **Type** | `bool` |
| **Default** | `True` |

Enables equivalent term detection in TIR Common Subexpression Elimination (CSE). When enabled, structurally different but semantically equivalent expressions are identified and deduplicated.

### TIR_DISABLE_CSE

| Attribute | Value |
|-----------|-------|
| **Key** | `"tir.disable_cse_tir"` |
| **Type** | `bool` |
| **Default** | `False` |

Disables TIR Common Subexpression Elimination entirely. CSE identifies and deduplicates identical subexpressions to reduce computation. Disabling it can increase computation but may preserve intermediate values for debugging.

### TIR_SIMPLIFY

| Attribute | Value |
|-----------|-------|
| **Key** | `"tir.Simplify"` |
| **Type** | `bool` |
| **Default** | `True` |

Enables or disables all TIR simplification passes. This is a coarse-grained control that affects arithmetic simplification, constant folding, dead code elimination, and other simplifications.

### TIR_DISABLE_STORAGE_REWRITE

| Attribute | Value |
|-----------|-------|
| **Key** | `"tir.disable_storage_rewrite"` |
| **Type** | `bool` |
| **Default** | `False` |

Disables storage rewrite optimization. Storage rewrite reuses memory buffers when their lifetimes do not overlap, reducing total memory usage.

### TIR_DISABLE_VECTORIZE

| Attribute | Value |
|-----------|-------|
| **Key** | `"tir.disable_vectorize"` |
| **Type** | `bool` |
| **Default** | `False` |

Disables vectorization of memory access operations. When disabled, all loads and stores use scalar instructions.

**When to disable:**
- Debugging memory access issues
- When vectorization causes incorrect results (alignment issues)
- On architectures with poor vectorized memory performance

### TIR_USE_ASYNC_COPY

| Attribute | Value |
|-----------|-------|
| **Key** | `"tir.use_async_copy"` |
| **Type** | `bool` |
| **Default** | `True` |

Enables asynchronous memory copy operations. When enabled, global-to-shared memory copies may use `cp.async` instructions on Ampere+ GPUs.

### TIR_ENABLE_DEBUG

| Attribute | Value |
|-----------|-------|
| **Key** | `"tir.enable_debug"` |
| **Type** | `bool` |
| **Default** | `False` |

Enables debug information in generated code. This includes source line mappings and additional assertions.

### TIR_MERGE_STATIC_SMEM

| Attribute | Value |
|-----------|-------|
| **Key** | `"tir.merge_static_smem"` |
| **Type** | `bool` |
| **Default** | `True` |

Merges static shared memory allocations into a single allocation to reduce the number of shared memory pointers. This is generally beneficial for reducing register pressure.

### TIR_ADD_LOWER_PASS

| Attribute | Value |
|-----------|-------|
| **Key** | `"tir.add_lower_pass"` |
| **Type** | `None` |
| **Default** | `None` |

Additional lowering passes to be applied during the compilation pipeline. This allows users to inject custom TIR transformations.

### TIR_NOALIAS

| Attribute | Value |
|-----------|-------|
| **Key** | `"tir.noalias"` |
| **Type** | `bool` |
| **Default** | `True` |

Enables pointer non-aliasing assumptions. When enabled, the compiler assumes that different buffer pointers do not alias (point to the same memory). This enables more aggressive optimization but may produce incorrect code if buffers actually alias.

---

## Pipeline Configs

### TL_ENABLE_ASYNC_COPY

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.enable_async_copy"` |
| **Type** | `bool` |
| **Default** | `True` |

Enables lowering eligible global-to-shared copies to PTX `cp.async` instructions.

**Behavior when enabled:**

| Copy Type | Lowered To |
|-----------|-----------|
| `T.copy(global -> shared)` | `cp.async + commit + wait` |
| `T.async_copy(global -> shared)` | `cp.async + commit` (no wait) |
| Manual global->shared stores in `T.Parallel` | `cp.async + commit + wait` |

**Important:** Automatic `cp.async` lowering is gated by the surrounding loop context. It only activates inside software-pipelined loops annotated with `num_stages > 0`. Outside such loops, synchronous copy lowering is preferred even when this flag is `True`.

For local `cp.async` injection on a specific parallel loop, use `T.Parallel(..., prefer_async=True)`.

### TL_ENABLE_LOWER_LDGSTG

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.enable_lower_ldgstg"` |
| **Type** | `bool` |
| **Default** | `False` |

Enables non-predicated LDG/STG lowering for global memory access. When enabled, converts Ramp-based global buffer load/store to ldg/stg intrinsics.

### TL_ENABLE_LOWER_LDGSTG_PREDICATED

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.enable_lower_ldgstg_predicated"` |
| **Type** | `bool` |
| **Default** | `False` |

Enables predicated LDG/STG lowering. When `True`:
- Predicated loads (if_then_else with else=0) are lowered to ldg intrinsics
- Predicated stores (IfThenElse with empty then case) are lowered to stg intrinsics

### TL_ENABLE_VECTORIZE_PLANNER_VERBOSE

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.enable_vectorize_planner_verbose"` |
| **Type** | `bool` |
| **Default** | `False` |

Enables verbose output from the vectorize planner. Prints detailed information about each buffer's inferred vector size and which buffer determines the final vectorization factor.

### TL_DISABLE_TMA_LOWER

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.disable_tma_lower"` |
| **Type** | `bool` |
| **Default** | `False` |
| **Status** | **Deprecated** -- will be removed in v0.1.10 |

Prevents plain `T.copy()` from auto-lowering to TMA store. Use `T.copy(..., disable_tma=True)` per-copy instead.

---

## Memory Configs

### TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.enable_aggressive_shared_memory_merge"` |
| **Type** | `bool` |
| **Default** | `False` |

Enables aggressive merging of shared memory allocations. When enabled, the compiler attempts to merge more shared memory buffers, potentially reducing total shared memory usage at the cost of increased complexity in memory layout.

### TL_DEBUG_MERGE_SHARED_MEMORY_ALLOCATIONS

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.debug_merge_shared_memory_allocations"` |
| **Type** | `bool` |
| **Default** | `False` |

Enables debug output for shared memory allocation merging. Shows which allocations are being merged and why.

### TL_DISABLE_SHUFFLE_ELECT

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.disable_shuffle_elect"` |
| **Type** | `bool` |
| **Default** | `False` |

Disables shuffle election optimization. When enabled (default), the compiler may use warp shuffle instructions for efficient thread elections instead of shared memory-based approaches.

### TL_DISABLE_LOOP_UNSWITCHING

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.disable_loop_unswitching"` |
| **Type** | `bool` |
| **Default** | `False` |

Disables loop unswitching optimization. Loop unswitching moves loop-invariant conditionals outside the loop, creating separate loop versions for each branch.

### TL_LOOP_UNSWITCHING_ALLOW_NON_TRIVIAL_ELSE

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.loop_unswitching_allow_non_trivial_else"` |
| **Type** | `bool` |
| **Default** | `False` |

Allows loop unswitching even when the else-version of the loop body has side effects. This is more aggressive and may increase code size.

### TL_DISABLE_THREAD_STORAGE_SYNC

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.disable_thread_storage_sync"` |
| **Type** | `bool` |
| **Default** | `False` |

Disables automatic insertion of thread synchronization barriers (e.g., `__syncthreads()`) for shared memory access coordination.

**When to disable:**
- When manual synchronization is implemented
- When no shared memory is used
- For performance optimization when synchronization is provably unnecessary

### TL_FORCE_LET_INLINE

| Attribute | Value |
|-----------|-------|
| **Key** | `"tl.force_let_inline"` |
| **Type** | `bool` |
| **Default** | `False` |

Forces TileLang to inline all let bindings during simplification. This can reduce register pressure but may increase code size.

---

## Using pass_configs with tilelang.jit and tilelang.compile

### With tilelang.compile

```python
import tilelang
from tilelang.transform.pass_config import PassConfigKey

# Define the kernel
@tilelang.jit
def my_kernel(A, B):
    M, N = T.const("M, N")
    A: T.Tensor[[M, N], T.float32]
    B: T.Tensor[[M, N], T.float32]
    with T.Kernel(M, N) as (i, j):
        B[i, j] = A[i, j] * 2.0

# Compile with pass configs
program = my_kernel.get_tir(a, b)
kernel = tilelang.compile(
    program,
    pass_configs={
        PassConfigKey.TL_ENABLE_ASYNC_COPY: True,
        PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
        PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ["--use_fast_math"],
    },
)
```

### With tilelang.jit

```python
import tilelang
from tilelang.transform.pass_config import PassConfigKey

@tilelang.jit(
    pass_configs={
        PassConfigKey.TL_ENABLE_FAST_MATH: True,
        PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    }
)
def my_kernel(A, B):
    # ... kernel definition
    pass
```

### Inside Function Body

```python
@tilelang.jit
def my_kernel(A, B):
    T.annotate_pass_configs({
        PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    T.annotate_compile_flags(["--use_fast_math"])
    # ... kernel definition
```

Function-level configs have lower priority than externally provided configs (external configs override).

---

## normalize_pass_configs Function

```python
def normalize_pass_configs(pass_configs: dict[str, Any] | None) -> dict[str, Any]
```

Canonicalizes known pass-config keys and emits compatibility warnings.

**Process:**
1. If `pass_configs` is `None`, returns an empty dictionary
2. For each key in the input dictionary:
   a. Try to match it as a `PassConfigKey` enum value
   b. If matched, use the canonical key
   c. If not matched, keep the key as-is (for custom/external configs)
3. Check for deprecated keys and emit `DeprecationWarning`

**Deprecated Keys:**

| Key | Message |
|-----|---------|
| `tl.disable_tma_lower` | Use `T.copy(..., disable_tma=True)` per-copy instead |

**Usage:**

```python
from tilelang.transform.pass_config import normalize_pass_configs

raw_configs = {
    "tl.Simplify": {"enable_simplify_let_inline": False},
    "tl.enable_fast_math": True,
    "custom_key": "custom_value",
}
normalized = normalize_pass_configs(raw_configs)
# Keys are validated and deprecated keys emit warnings
```

---

## Common Pass Config Recipes

### Maximum Performance (Ampere/Hopper)

```python
pass_configs = {
    PassConfigKey.TL_ENABLE_FAST_MATH: True,
    PassConfigKey.TL_ENABLE_ASYNC_COPY: True,
    PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    PassConfigKey.TL_SIMPLIFY: {
        "enable_simplify_let_inline": True,
        "transitively_prove_inequalities": True,
    },
    PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ["--use_fast_math"],
}
```

### Debug Build

```python
pass_configs = {
    PassConfigKey.TL_AST_PRINT_ENABLE: True,
    PassConfigKey.TL_ENABLE_DUMP_IR: True,
    PassConfigKey.TL_DUMP_IR_DIR: "./debug_ir/",
    PassConfigKey.TL_ENABLE_PTXAS_VERBOSE_OUTPUT: True,
    PassConfigKey.TIR_ENABLE_DEBUG: True,
    PassConfigKey.TL_ENABLE_VECTORIZE_PLANNER_VERBOSE: True,
    PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True,
    PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS: "png",
}
```

### Safe/Conservative Build

```python
pass_configs = {
    PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
    PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: False,
    PassConfigKey.TIR_DISABLE_VECTORIZE: True,
    PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    PassConfigKey.TL_ENABLE_ASYNC_COPY: False,
    PassConfigKey.TIR_DISABLE_CSE: True,
}
```

### Tensor Core Optimized

```python
pass_configs = {
    PassConfigKey.TL_DISABLE_WGMMA: False,
    PassConfigKey.TL_ENABLE_ASYNC_COPY: True,
    PassConfigKey.TL_DISABLE_VECTORIZE_256: False,
    PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True,
    PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 5,
    PassConfigKey.TL_SIMPLIFY: {
        "enable_simplify_let_inline": True,
    },
}
```

### Reduced Register Pressure

```python
pass_configs = {
    PassConfigKey.TL_FORCE_LET_INLINE: True,
    PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
    PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 0,
    PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}
```

---

## Architecture-Specific Configurations

### NVIDIA Volta (SM70)

```python
volta_configs = {
    PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,  # Not supported
    PassConfigKey.TL_ENABLE_ASYNC_COPY: False,          # No cp.async
    PassConfigKey.TL_DISABLE_WGMMA: True,               # No WGMMA
    PassConfigKey.TL_DISABLE_VECTORIZE_256: True,        # Limited vector width
}
```

### NVIDIA Ampere (SM80)

```python
ampere_configs = {
    PassConfigKey.TL_ENABLE_ASYNC_COPY: True,            # cp.async available
    PassConfigKey.TL_DISABLE_WGMMA: True,                # No WGMMA (use MMA)
    PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,     # Optional
    PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ["-arch=sm_80"],
}
```

### NVIDIA Hopper (SM90)

```python
hopper_configs = {
    PassConfigKey.TL_ENABLE_ASYNC_COPY: True,            # cp.async available
    PassConfigKey.TL_DISABLE_WGMMA: False,               # WGMMA available
    PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,     # Recommended
    PassConfigKey.TL_DISABLE_VECTORIZE_256: False,        # Full support
    PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ["-arch=sm_90a"],
}
```

### AMD CDNA

```python
cdna_configs = {
    PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,     # Not CUDA
    PassConfigKey.TL_DISABLE_WGMMA: True,                # Use MFMA instead
    PassConfigKey.TL_ENABLE_ASYNC_COPY: False,           # Different async model
}
```

---

## Complete PassConfigKey Reference Table

| Key | Type | Default | Category |
|-----|------|---------|----------|
| `tl.Simplify` | `dict` | see sub-options | Simplification |
| `transitively_prove_inequalities` | `bool` | `False` | Simplification |
| `convert_boolean_to_and_of_ors` | `bool` | `False` | Simplification |
| `apply_constraints_to_boolean_branches` | `bool` | `False` | Simplification |
| `propagate_knowns_to_prove_conditional` | `bool` | `False` | Simplification |
| `propagate_knowns_to_simplify_expressions` | `bool` | `False` | Simplification |
| `enable_simplify_let_inline` | `bool` | `True` | Simplification |
| `tl.disable_data_race_check` | `bool` | `False` | Memory Safety |
| `tl.disable_prelower_semantic_check` | `bool` | `False` | Memory Safety |
| `tl.disable_warp_specialized` | `bool` | `False` | Warp/Vector |
| `tl.enable_fast_math` | `bool` | `False` | Compilation |
| `tl.ptxas_register_usage_level` | `int` | `None` | Compilation |
| `tl.enable_ptxas_verbose_output` | `bool` | `False` | Compilation |
| `tl.device_compile_flags` | `str/list` | `None` | Compilation |
| `tl.config_index_bitwidth` | `int` | `32` | Compilation |
| `tl.disable_tma_lower` | `bool` | `False` | Pipeline (Deprecated) |
| `tl.disable_safe_memory_legalize` | `bool` | `False` | Memory Safety |
| `tl.disable_vectorize_256` | `bool` | `False` | Warp/Vector |
| `tl.enable_async_copy` | `bool` | `True` | Pipeline |
| `tl.enable_lower_ldgstg` | `bool` | `False` | Pipeline |
| `tl.enable_lower_ldgstg_predicated` | `bool` | `False` | Pipeline |
| `tl.enable_vectorize_planner_verbose` | `bool` | `False` | Debug |
| `tl.disable_wgmma` | `bool` | `False` | Warp/Vector |
| `tl.debug_merge_shared_memory_allocations` | `bool` | `False` | Debug |
| `tl.enable_aggressive_shared_memory_merge` | `bool` | `False` | Memory |
| `tl.disable_shuffle_elect` | `bool` | `False` | Memory |
| `tl.disable_loop_unswitching` | `bool` | `False` | Memory |
| `tl.loop_unswitching_allow_non_trivial_else` | `bool` | `False` | Memory |
| `tl.disable_thread_storage_sync` | `bool` | `False` | Memory |
| `tl.force_let_inline` | `bool` | `False` | Simplification |
| `tl.ast_print_enable` | `bool` | `False` | Debug |
| `tl.layout_visualization_enable` | `bool` | `False` | Debug |
| `tl.layout_visualization_formats` | `str` | N/A | Debug |
| `tl.storage_rewrite_detect_inplace` | `bool` | `False` | Memory |
| `tir.enable_equiv_terms_in_cse_tir` | `bool` | `True` | TIR |
| `tir.disable_cse_tir` | `bool` | `False` | TIR |
| `tir.Simplify` | `bool` | `True` | TIR |
| `tir.disable_storage_rewrite` | `bool` | `False` | TIR |
| `tir.disable_vectorize` | `bool` | `False` | TIR |
| `tir.use_async_copy` | `bool` | `True` | TIR |
| `tir.enable_debug` | `bool` | `False` | TIR |
| `tir.merge_static_smem` | `bool` | `True` | TIR |
| `tir.add_lower_pass` | `None` | `None` | TIR |
| `tir.noalias` | `bool` | `True` | TIR |
| `cuda.kernels_output_dir` | `str` | `""` | Debug |
| `tl.disable_out_of_bound_warning` | `bool` | `True` | Memory Safety |
| `tl.enable_dump_ir` | `bool` | `False` | Debug |
| `tl.dump_ir_path` | `str` | `"./dump_ir/"` | Debug |
