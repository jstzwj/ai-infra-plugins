---
name: triton
description: >
  Comprehensive reference documentation and skill for OpenAI Triton - a language and compiler
  for writing highly efficient custom Deep-Learning primitives on GPUs. Covers Python API
  (triton.language, triton.runtime, triton.compiler), MLIR dialects and passes, backends
  (NVIDIA CUDA, AMD ROCm/HIP), experimental features (Gluon, GSAN, triton_kernels),
  Proton profiler, tutorials, debugging, and build system.
version: 3.7.0
---

# Triton - GPU Programming Language & Compiler

## Overview

Triton is a language and compiler for writing highly efficient custom Deep-Learning primitives. It provides an open-source environment to write fast code at higher productivity than CUDA, but also with higher flexibility than other existing DSLs.

**Supported Hardware:**
- NVIDIA GPUs (Compute Capability 8.0+)
- AMD GPUs (ROCm 6.2+)
- CPUs (under development)

**Supported Platforms:** Linux

**Python Versions:** CPython 3.10-3.14

## Quick Reference

### Installation
```bash
pip install triton
```

### Minimal Kernel Example
```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

## Documentation Structure

### Core API
- [01-overview-and-architecture](references/01-overview-and-architecture.md) - Project architecture, compilation pipeline overview
- [02-getting-started](references/02-getting-started.md) - Installation, building from source, quick start
- [03-language-core](references/03-language-core.md) - `triton.language` core: tensors, pointers, load/store, dtypes, dot, atomic ops
- [04-language-math](references/04-language-math.md) - `triton.language.math`: exp, log, sin, cos, sqrt, etc.
- [05-language-random](references/05-language-random.md) - `triton.language.random`: rand, randn, philox RNG
- [06-language-standard](references/06-language-standard.md) - `triton.language.standard`: reductions, sorting, softmax, cumsum
- [07-language-semantic](references/07-language-semantic.md) - Semantic analysis layer

### Runtime System
- [08-runtime-jit](references/08-runtime-jit.md) - `@triton.jit` decorator, JITFunction, specialization, caching
- [09-runtime-autotuner](references/09-runtime-autotuner.md) - `@triton.autotune`, Config, Heuristics, benchmarking
- [10-runtime-cache](references/10-runtime-cache.md) - Cache management, file cache, remote cache
- [11-runtime-driver](references/11-runtime-driver.md) - GPU driver abstraction, memory allocation
- [12-runtime-interpreter](references/12-runtime-interpreter.md) - `TRITON_INTERPRET=1` interpreter mode

### Compiler
- [13-compiler-pipeline](references/13-compiler-pipeline.md) - Full compilation pipeline: AST → TTIR → TTGIR → LLVMIR → PTX
- [14-compiler-codegen](references/14-compiler-codegen.md) - AST-to-MLIR code generation, CodeGenerator class
- [15-compiler-errors](references/15-compiler-errors.md) - CompilationError, error handling

### Backend System
- [16-backends](references/16-backends.md) - Backend architecture, BaseBackend, GPUTarget, driver/compiler split
- [17-nvidia-backend](references/17-nvidia-backend.md) - NVIDIA CUDA backend specifics
- [18-amd-backend](references/18-amd-backend.md) - AMD ROCm/HIP backend specifics

### MLIR Dialects & Passes
- [19-mlir-dialects](references/19-mlir-dialects.md) - Triton, TritonGPU, Gluon, TritonInstrument dialects
- [20-compilation-passes](references/20-compilation-passes.md) - All compilation passes: coalesce, pipeline, accelerate-matmul, etc.
- [21-memory-layouts](references/21-memory-layouts.md) - LinearLayout, blocked layouts, distributed/shared layouts

### C++ Extension
- [22-cpp-extension](references/22-cpp-extension.md) - Python C++ bindings, pybind11 modules, ir.cc, passes.cc

### Tutorials
- [23-tutorials](references/23-tutorials.md) - All 11 tutorials with detailed explanations

### Advanced / Experimental
- [24-triton-kernels](references/24-triton-kernels.md) - `triton_kernels` library: matmul, reduce, topk, swiglu, etc.
- [25-gluon](references/25-gluon.md) - Gluon experimental language and compiler
- [26-proton-profiler](references/26-proton-profiler.md) - Proton profiling system
- [27-gsan](references/27-gsan.md) - GPU Sanitizer (GSAN) for symmetric memory

### Operations
- [28-debugging](references/28-debugging.md) - Environment variables, knobs, IR dumping, reproducer
- [29-building](references/29-building.md) - Build system, CMake, custom LLVM, cross-compilation
- [30-plugin-system](references/30-plugin-system.md) - Out-of-tree plugin dialects and passes
- [31-testing](references/31-testing.md) - Test framework, lit tests, pytest

## Key Concepts

### Programming Model
Triton kernels execute on a grid of **programs** (thread blocks). Each program:
1. Computes its ID via `tl.program_id(axis)`
2. Computes memory offsets from the program ID
3. Loads data using `tl.load()` with pointers and masks
4. Computes results using tensor operations
5. Stores results using `tl.store()`

### Compilation Pipeline
```
Python AST → TTIR (Triton IR) → TTGIR (Triton GPU IR) → LLVM IR → PTX/AMDGPU → CUBIN/HSACO
```

### Type System
- **Scalar types:** `int1`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `float16`, `bfloat16`, `float32`, `float64`, `float8e4nv`, `float8e5`, etc.
- **Pointer types:** `tl.pointer_type(element_ty, const=False)`
- **Tensor types:** block-level tensors with implicit parallelism
- **constexpr:** Compile-time constants annotated with `tl.constexpr`

### Key Decorators
- `@triton.jit` - Mark a function for JIT compilation
- `@triton.autotune(configs, key)` - Auto-tune kernel configurations
- `@triton.heuristics(values)` - Compute meta-parameters heuristically
- `@triton.experimental.constexpr_function` - Compile-time function evaluation
