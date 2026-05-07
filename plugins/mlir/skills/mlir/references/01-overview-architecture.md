# MLIR Overview & Architecture

## What is MLIR

MLIR (Multi-Level Intermediate Representation) is a hybrid compiler intermediate representation from the LLVM project that combines:

1. **Traditional three-address SSA** representations (like LLVM IR or Swift SIL)
2. **Polyhedral loop optimization** as first-class concepts
3. **Extensible dialect system** for domain-specific abstractions

MLIR stands for "Multi-Level IR" (also interpreted as "Multi-dimensional Loop IR", "Machine Learning IR", or "Mid Level IR").

## Design Goals

1. **Represent, analyze, and transform** high-level dataflow graphs and target-specific code for high-performance data parallel systems
2. **Progressive lowering** from domain-specific representations down to machine code through a single continuous design
3. **Extensibility** - arbitrary dialects can be defined with custom operations, types, and attributes
4. **Reusability** - modular and reusable target-independent and target-dependent passes
5. **Multi-target support** - general-purpose multicores, GPUs, neural network accelerators, FPGAs

## Architecture Overview

### Layered Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Frontend Dialects                          │
│   TensorFlow │ PyTorch │ MHLO │ TOSA │ Custom Domain Dialects   │
├──────────────────────────────────────────────────────────────────┤
│                       Transformation Layer                        │
│   Linalg │ Vector │ SCF │ Bufferization │ Transform Dialect      │
├──────────────────────────────────────────────────────────────────┤
│                        Core Dialects                              │
│   Arith │ Math │ MemRef │ Tensor │ Index │ Func │ CF             │
├──────────────────────────────────────────────────────────────────┤
│                        Target Dialects                            │
│   LLVM │ SPIR-V │ GPU │ NVVM │ AMDGPU │ ArmSVE │ X86            │
├──────────────────────────────────────────────────────────────────┤
│                        Core Infrastructure                        │
│   IR Core │ Pass Manager │ Rewriting │ Analysis │ Dialect System │
├──────────────────────────────────────────────────────────────────┤
│                        Bindings & Tools                           │
│   Python Bindings │ C API │ mlir-opt │ mlir-translate │ LSP      │
└──────────────────────────────────────────────────────────────────┘
```

### Core Abstractions

MLIR is built on a small set of core abstractions:

1. **Operations** - The fundamental unit of computation
2. **Values** - SSA values (operation results or block arguments)
3. **Blocks** - Ordered lists of operations with arguments
4. **Regions** - Ordered lists of blocks within operations
5. **Types** - Describe value kinds
6. **Attributes** - Compile-time constant data
7. **Dialects** - Namespaces grouping related constructs

### Operation Structure

An MLIR operation contains:
- A unique string name (e.g., `arith.addi`, `linalg.matmul`)
- Zero or more operands (input SSA values)
- Zero or more results (output SSA values)
- A dictionary of attributes (constant data)
- Zero or more successors (target blocks)
- Zero or more regions (nested blocks)
- Location information (source tracking)

### Dialect System

Dialects are the primary extension mechanism in MLIR:

```
Dialect
├── Operations (e.g., arith.addi, arith.mulf)
├── Types (e.g., memref<MxNxf32>, tensor<?xf32>)
├── Attributes (e.g., #map, #set)
├── Interfaces (e.g., bufferizable, verifiable)
└── Canonicalization patterns
```

## Project Structure

```
mlir/
├── docs/                    # Documentation (Markdown)
│   ├── LangRef.md          # Language reference
│   ├── Rationale/          # Design rationale
│   ├── Dialects/           # Dialect documentation
│   ├── Tutorials/          # Tutorials (Toy, Transform)
│   ├── DefiningDialects/   # Dialect definition guides
│   ├── Tools/              # Tool documentation
│   ├── Traits/             # Trait documentation
│   └── Bindings/           # Binding documentation
├── include/mlir/            # C++ headers
│   ├── IR/                 # Core IR classes
│   ├── Pass/               # Pass infrastructure
│   ├── Dialect/            # Dialect definitions
│   ├── Interfaces/         # Interface definitions
│   ├── Rewriter/           # Rewriting infrastructure
│   ├── Analysis/           # Analysis passes
│   ├── Transforms/         # Transform passes
│   ├── Conversion/         # Dialect conversion
│   └── Target/             # Target backends
├── lib/                     # Implementation
│   ├── IR/                 # Core IR implementation
│   ├── Pass/               # Pass manager
│   ├── Dialect/            # Dialect implementations
│   ├── Rewriter/           # Rewriting
│   ├── Conversion/         # Conversion passes
│   ├── Transforms/         # Transform passes
│   └── Target/             # Target backends
├── python/                  # Python bindings
├── tools/                   # Command-line tools
│   ├── mlir-opt/           # IR optimizer/testing tool
│   ├── mlir-translate/     # Format translator
│   ├── mlir-lsp-server/    # LSP server
│   ├── mlir-tblgen/        # TableGen generator
│   ├── mlir-cpu-runner/    # JIT runner
│   └── mlir-reduce/        # Test case reducer
├── test/                    # Test suite
├── examples/                # Example projects
├── utils/                   # Editor support, scripts
├── benchmark/               # Benchmarks
└── unittests/               # Unit tests
```

## Comparison with LLVM IR

| Feature | LLVM IR | MLIR |
|---------|---------|------|
| Type system | Fixed set of types | Extensible type system |
| Operations | Fixed instruction set | Extensible via dialects |
| Control flow | Basic blocks + PHI nodes | Blocks with arguments + Regions |
| Hierarchical | Flat (functions only) | Nested regions |
| Polyhedral | External (Polly) | Built-in (Affine dialect) |
| Target support | CPU only | CPU, GPU, accelerators |
| Extensibility | Limited | Full dialect system |
| Level | Low-level | Multi-level (high to low) |

## Key Design Decisions

### Block Arguments vs PHI Nodes

MLIR uses block arguments instead of LLVM-style PHI nodes:

1. No need to keep PHI nodes at top of block
2. No separate function Argument node (entry block arguments serve this purpose)
3. No confusing parallel copy semantics of PHI nodes
4. No unordered predecessor lists causing performance issues
5. Values available on specific successor edges

### Signless Integer Types

Integers are signless at the type level (`i32`, not `s32`/`u32`). The interpretation (signed/unsigned) is determined by the operation (e.g., `arith.divsi` vs `arith.divui`).

### Separate Float and Integer Operations

Integer and floating-point operations are split into separate operations (e.g., `arith.addi` vs `arith.addf`) because:
- Floats have NaN behavior, different comparison semantics
- Floats support rounding modes and fast-math
- Integers have overflow behavior concerns

### Extensible Dialect System

Dialects allow MLIR to be extended without modifying core infrastructure:
- Custom operations with ODS (Operation Definition Specification)
- Custom types and attributes
- Dialect interfaces for cross-dialect interaction
- Progressive lowering between dialects

### Multithreaded Compilation

MLIR supports multithreaded pass execution through:
- Extensive uniqued immutable data structures
- Per-operation constant pools (not global uniquing)
- Functions are not SSA values
- Pass instances copied per thread

## Build System (CMake)

### Prerequisites
- C++17 compatible compiler
- CMake >= 3.20
- Ninja (recommended)
- Python 3 (for bindings and tests)

### Building MLIR

```bash
# Clone LLVM project
git clone https://github.com/llvm/llvm-project.git

# Configure
cmake -G Ninja -B build llvm-project/llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON

# Build
cmake --build build --target mlir-opt mlir-translate

# Test
cmake --build build --target check-mlir
```

### Key CMake Variables
- `LLVM_ENABLE_PROJECTS` - Include `mlir` to build MLIR
- `MLIR_ENABLE_BINDINGS_PYTHON` - Enable Python bindings
- `LLVM_TARGETS_TO_BUILD` - Target backends to include
- `MLIR_ENABLE_EXECUTION_ENGINE` - Enable execution engine
- `MLIR_BUILD_MLIR_C_DYLIB` - Build shared C library

## Version Information

MLIR is developed as part of the LLVM project and follows LLVM's version numbering. The current source is based on LLVM/MLIR 19.x development branch.

### Release Notes Highlights (Recent)

Key changes in recent releases:
- Enhanced Transform dialect with new operations
- Improved bufferization infrastructure
- New dialect extensions
- Better Python bindings coverage
- Performance improvements in pass manager
- Enhanced SPIR-V serialization
- Extended GPU dialect support

## Relationship to Other Projects

### TensorFlow/XLA
- TensorFlow graphs lower to MLIR via MHLO dialect
- XLA uses MLIR for optimization passes

### PyTorch
- torch-mlir project lowers PyTorch models to MLIR
- Uses Linalg dialect as primary target

### IREE
- Uses MLIR as core IR for ML model compilation
- Targets Vulkan, Metal, CUDA, and CPU

### JAX/XLA
- JAX models go through XLA → MHLO → MLIR pipeline

### ONNX-MLIR
- Converts ONNX models to MLIR
- Uses TOSA and affine dialects
