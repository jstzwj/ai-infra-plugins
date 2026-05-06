# Chapter 1: Overview and Architecture

## What is Triton?

Triton is a **language and compiler** for writing highly efficient custom Deep-Learning primitives on GPUs. It aims to provide:
- Higher productivity than CUDA (Python-based kernel definitions)
- Higher flexibility than existing DSLs (Turing-complete, arbitrary control flow)
- Performance competitive with hand-written CUDA kernels

The project originated from the paper: [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) (MAPL 2019).

**Current Version:** 3.7.0

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        User Code (Python)                        │
│  @triton.jit decorated functions using triton.language (tl)     │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                    Python Frontend                                │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐  │
│  │ AST Parsing   │  │ Semantic      │  │ Code Generation      │  │
│  │ (code_gen)    │  │ Analysis      │  │ (AST → TTIR MLIR)    │  │
│  └──────────────┘  └───────────────┘  └──────────────────────┘  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                 MLIR Compiler Pipeline                           │
│  ┌─────────┐   ┌──────────┐   ┌─────────┐   ┌───────────────┐  │
│  │  TTIR   │ → │  TTGIR   │ → │ LLVM IR │ → │ PTX / AMDGPU  │  │
│  │ Passes  │   │  Passes  │   │ Passes  │   │ Assembly      │  │
│  └─────────┘   └──────────┘   └─────────┘   └───────────────┘  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                     Runtime / Driver                             │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐  │
│  │ JIT Engine   │  │ Cache Manager │  │ GPU Driver           │  │
│  │              │  │               │  │ (CUDA / HIP)         │  │
│  └──────────────┘  └───────────────┘  └──────────────────────┘  │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐  │
│  │ Autotuner    │  │ Interpreter   │  │ Memory Allocator     │  │
│  └──────────────┘  └───────────────┘  └──────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## Source Code Organization

```
triton/
├── python/                      # Python package
│   ├── triton/                  # Main Python module
│   │   ├── __init__.py          # Top-level exports
│   │   ├── language/            # Kernel language (tl.*)
│   │   │   ├── core.py          # Core ops: load, store, dot, arithmetic
│   │   │   ├── math.py          # Math: exp, log, sin, cos, sqrt
│   │   │   ├── random.py        # RNG: rand, randn, philox
│   │   │   ├── standard.py      # Standard: reductions, sort, softmax
│   │   │   ├── semantic.py      # Semantic analysis layer
│   │   │   ├── target_info.py   # Target-specific info
│   │   │   └── extra/           # Backend-specific language extensions
│   │   │       └── libdevice.py # NVIDIA libdevice functions
│   │   ├── runtime/             # Runtime system
│   │   │   ├── jit.py           # @triton.jit, JITFunction
│   │   │   ├── autotuner.py     # @triton.autotune, Config
│   │   │   ├── cache.py         # Kernel caching
│   │   │   ├── driver.py        # Driver discovery
│   │   │   ├── interpreter.py   # CPU interpreter
│   │   │   ├── build.py         # C/C++ compilation helpers
│   │   │   ├── _allocation.py   # Memory allocation
│   │   │   └── _async_compile.py# Async compilation
│   │   ├── compiler/            # Compiler pipeline
│   │   │   ├── compiler.py      # Main compile() function
│   │   │   ├── code_generator.py# AST → MLIR code gen
│   │   │   └── errors.py        # Compilation errors
│   │   ├── backends/            # Backend interface
│   │   │   ├── __init__.py      # BaseBackend, GPUTarget
│   │   │   ├── compiler.py      # Compiler utilities
│   │   │   └── driver.py        # Backend discovery
│   │   ├── tools/               # CLI tools
│   │   ├── experimental/        # Experimental features
│   │   │   ├── gluon/           # Gluon language
│   │   │   └── gsan/            # GPU sanitizer
│   │   ├── _C/                  # C++ extension (libtriton)
│   │   ├── knobs.py             # Configuration knobs
│   │   ├── testing.py           # Testing utilities
│   │   └── errors.py            # Top-level errors
│   ├── triton_kernels/          # Pre-built kernel library
│   ├── tutorials/               # 11 tutorial scripts
│   ├── test/                    # Test suite
│   └── src/                     # C++ source for Python bindings
│       ├── main.cc              # Module entry point
│       ├── ir.cc / ir.h         # IR builder bindings
│       ├── passes.cc / passes.h # Pass bindings
│       ├── llvm.cc              # LLVM backend bindings
│       ├── interpreter.cc       # Interpreter ops
│       ├── specialize.cc        # Type specialization
│       ├── linear_layout.cc     # Layout utilities
│       └── gluon_ir.cc          # Gluon IR bindings
├── include/triton/              # C++ headers
│   ├── Dialect/                 # MLIR dialect definitions
│   │   ├── Triton/              # Triton dialect
│   │   ├── TritonGPU/           # TritonGPU dialect
│   │   ├── Gluon/               # Gluon dialect
│   │   ├── TritonInstrument/    # Instrumentation dialect
│   │   └── TritonNvidiaGPU/     # NVIDIA-specific dialect
│   ├── Analysis/                # Analysis passes
│   ├── Conversion/              # Conversion passes
│   ├── Target/                  # Target backends
│   └── Tools/                   # Utility tools
├── lib/                         # C++ implementation
│   ├── Dialect/                 # Dialect implementations
│   ├── Analysis/                # Analysis implementations
│   ├── Conversion/              # Conversion pass implementations
│   ├── Target/LLVMIR/           # LLVM IR target
│   └── Tools/                   # Tool implementations
├── third_party/                 # Vendor-specific code
│   ├── nvidia/                  # NVIDIA backend
│   ├── amd/                     # AMD backend
│   └── proton/                  # Proton profiler
├── test/                        # MLIR lit tests
├── unittest/                    # C++ unit tests
├── bin/                         # CLI tools
│   ├── triton-opt.cpp           # MLIR optimizer tool
│   ├── triton-lsp.cpp           # Language server
│   ├── triton-reduce.cpp        # Reducer tool
│   └── triton-tensor-layout.cpp # Layout tool
├── docs/                        # Documentation
├── examples/plugins/            # Plugin examples
└── cmake/                       # CMake modules
```

## Compilation Pipeline Detail

### Stage 1: Python AST → TTIR (Triton IR)

The `CodeGenerator` class in `compiler/code_generator.py` walks the Python AST using the visitor pattern and emits Triton MLIR operations:

1. **Function definition** → `tt.func` operation
2. **Variable assignment** → SSA value binding
3. **`tl.load()`** → `tt.load` operation
4. **`tl.store()`** → `tt.store` operation
5. **`tl.dot()`** → `tt.dot` operation
6. **Arithmetic (`+`, `*`, etc.)** → `tt.add`, `tt.mul` etc.
7. **Control flow (`if`, `for`, `while`)** → `scf.if`, `scf.for`, `scf.while`
8. **`tl.arange()`** → `tt.make_range` operation

### Stage 2: TTIR → TTGIR (Triton GPU IR)

Converts generic Triton operations to GPU-specific operations with memory layout annotations:

- **`TritonToTritonGPU`** - Converts tt operations to ttgpu operations
- **Coalesce** - Optimizes memory access patterns
- **Optimize Thread Locality** - Improves data locality
- **Accelerate Matmul** - Detects and optimizes matmul patterns
- **Pipeline** - Software pipelining for loop optimization
- **Warp Specialization** - Specialized warp-level operations
- **Memory Allocation** - Shared memory management

### Stage 3: TTGIR → LLVM IR

Converts GPU operations to LLVM IR with target-specific intrinsics:

- **`TritonGPUToLLVM`** - Main conversion pass
- Various operation-specific converters (DotOpToLLVM, MemoryOpToLLVM, etc.)

### Stage 4: LLVM IR → Binary

Target-specific code generation:
- **NVIDIA:** LLVM IR → PTX → CUBIN (via ptxas)
- **AMD:** LLVM IR → AMDGPU ISA → HSACO

## Key Design Decisions

1. **Block-level programming model:** Users program at the level of thread blocks (programs), not individual threads. The compiler handles thread-level parallelism automatically.

2. **MLIR-based compiler:** Uses MLIR (Multi-Level Intermediate Representation) for the compilation infrastructure, enabling progressive lowering through multiple IR levels.

3. **Backend abstraction:** Hardware-specific code is isolated in backend modules (`third_party/nvidia/`, `third_party/amd/`), allowing support for multiple GPU architectures.

4. **Python-first:** Kernels are written in Python with `@triton.jit` decoration, making them easy to write, debug, and integrate with PyTorch.

5. **Autotuning:** Built-in support for automatic kernel configuration tuning (block sizes, number of warps, pipeline stages).

## Version Information

- **Triton Version:** 3.7.0
- **MLIR:** Based on LLVM main branch (specific commit in `cmake/llvm-hash.txt`)
- **Python Support:** 3.10 - 3.14
- **License:** MIT
