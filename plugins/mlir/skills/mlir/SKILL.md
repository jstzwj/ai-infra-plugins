---
name: mlir
description: >
  Comprehensive reference documentation and skill for MLIR (Multi-Level Intermediate Representation) -
  the extensible compiler infrastructure framework from the LLVM project. Covers the MLIR language
  reference, IR core concepts (Operations, Values, Blocks, Regions, Types, Attributes), dialect
  definitions (ODS, TableGen), pass infrastructure, pattern rewriting, dialect conversion, PDLL,
  all standard dialects (Arith, Math, MemRef, Tensor, Vector, SCF, Affine, Linalg, GPU, SPIR-V,
  LLVM, Transform, Bufferization, TOSA, Quant, OpenMP, OpenACC, Async, PDL, IRDL, EmitC, etc.),
  interfaces, traits, Python bindings, C API, tools, and tutorials. Based on MLIR source code analysis.
version: "19.0"
---

# MLIR - Multi-Level Intermediate Representation

## Overview

MLIR (Multi-Level Intermediate Representation) is a compiler infrastructure from the LLVM project designed for easy expression and optimization of computations involving deep loop nests and dense matrices of high dimensionality. It is well-suited to deep learning computations and is general enough to represent arbitrary sequential computation.

MLIR uses ideas drawn from LLVM IR and Swift's SIL for lower-level constructs, combined with ideas from the polyhedral abstraction to represent loop nests, multidimensional data (tensors), and transformations as first-class concepts.

**Key Features:**
- Extensible dialect system for defining custom operations, types, and attributes
- Multi-level IR supporting progressive lowering from high-level to machine code
- Polyhedral compilation support (affine maps, integer sets)
- Built-in pass infrastructure with multithreading support
- Pattern rewriting and dialect conversion frameworks
- Python bindings and C API
- Rich set of standard dialects for CPU, GPU, and accelerator targets

**MLIR Version:** 19.0 (LLVM project)

## Key Architecture Concepts

- **Dialect**: A namespace grouping related operations, types, and attributes (e.g., `arith`, `linalg`, `llvm`)
- **Operation**: The fundamental unit of computation in MLIR (generalized instruction)
- **Value**: SSA value - either an operation result or block argument
- **Block**: A list of operations with arguments (basic block in CFG regions)
- **Region**: An ordered list of blocks contained within an operation
- **Type**: Describes the kind of a value (integers, floats, tensors, memrefs, etc.)
- **Attribute**: Compile-time constant data attached to operations
- **Pass**: A transformation applied to the IR
- **Pattern**: A rewrite rule for transforming operations
- **Interface**: Abstract API for operating on operations generically
- **Trait**: Mixin-like properties for operations

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────┐
│                    High-Level Dialects                         │
│  TOSA │ Linalg │ Transform │ Shape │ Quant │ ...              │
├───────────────────────────────────────────────────────────────┤
│                    Mid-Level Dialects                          │
│  Arith │ Math │ Vector │ Tensor │ MemRef │ SCF │ Affine       │
├───────────────────────────────────────────────────────────────┤
│                    Target Dialects                             │
│  LLVM │ SPIR-V │ GPU │ NVVM │ AMDGPU │ ArmSVE │ X86          │
├───────────────────────────────────────────────────────────────┤
│                    Core Infrastructure                         │
│  IR │ Pass │ Rewriting │ Interfaces │ Dialect │ Analysis      │
├───────────────────────────────────────────────────────────────┤
│                    Bindings & Tools                            │
│  Python │ C API │ mlir-opt │ mlir-translate │ LSP Server      │
└───────────────────────────────────────────────────────────────┘
```

## Compilation Pipeline Example

```
Source Language (Python, C++, etc.)
         │
         ▼
  ┌──────────────┐
  │ Frontend     │  (TOSA, MHLO, etc.)
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ Linalg       │  (Structured ops on tensors)
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ Bufferization│  (tensor → memref conversion)
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ Vector       │  (Vectorization of memref ops)
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ LLVM/SPIR-V  │  (Target-specific lowering)
  └──────┬───────┘
         ▼
  Machine Code / SPIR-V Binary
```

## Quick Reference

### MLIR Assembly Format

```mlir
// A simple MLIR module
module {
  func.func @add(%arg0: i32, %arg1: i32) -> i32 {
    %result = arith.addi %arg0, %arg1 : i32
    return %result : i32
  }
}
```

### Common Operations

```mlir
// Arithmetic
%0 = arith.constant 42 : i32
%1 = arith.addi %a, %b : i32
%2 = arith.mulf %x, %y : f32
%3 = arith.cmpi "eq", %a, %b : i32

// Memory
%m = memref.alloc() : memref<10xf32>
%v = memref.load %m[%idx] : memref<10xf32>
memref.store %val, %m[%idx] : memref<10xf32>

// Control Flow (structured)
%r = scf.if %cond -> (i32) {
  scf.yield %a : i32
} else {
  scf.yield %b : i32
}

// Loop
%sum = scf.for %i = %lb to %ub step %step
        iter_args(%acc = %init) -> (i32) {
  %next = arith.addi %acc, %i : i32
  scf.yield %next : i32
}

// Function call
%result = func.call @my_func(%arg0, %arg1) : (f32, f32) -> f32
```

## Dialects Overview

### Core Dialects
| Dialect | Namespace | Description |
|---------|-----------|-------------|
| Builtin | (builtin) | Core types, attributes, module operation |
| Func | func | Function definitions and calls |
| Arith | arith | Integer and floating-point arithmetic |
| Math | math | Mathematical operations (sin, cos, sqrt, etc.) |
| Index | index | Target-independent index computations |
| Complex | complex | Complex number operations |
| MemRef | memref | Memory buffer operations |
| Tensor | tensor | Tensor type and operations |
| Vector | vector | SIMD vector operations |
| SCF | scf | Structured Control Flow |
| CF | cf | Explicit Control Flow (branch, conditional branch) |
| Affine | affine | Affine/polyhedral operations |

### Transformation Dialects
| Dialect | Namespace | Description |
|---------|-----------|-------------|
| Transform | transform | Composable IR transformations |
| Bufferization | bufferization | Tensor to memref conversion |
| Linalg | linalg | Linear algebra structured operations |
| Shape | shape | Shape inference and computation |
| Shard | shard | Tensor sharding operations |

### Target Dialects
| Dialect | Namespace | Description |
|---------|-----------|-------------|
| LLVM | llvm | LLVM IR representation |
| SPIR-V | spv | SPIR-V shader/compute |
| GPU | gpu | GPU kernel operations |
| NVVM | nvvm | NVIDIA GPU intrinsics |
| NVGPU | nvgpu | NVIDIA GPU-specific operations |
| AMDGPU | amdgpu | AMD GPU operations |

### Parallel Dialects
| Dialect | Namespace | Description |
|---------|-----------|-------------|
| OpenMP | omp | OpenMP parallelism |
| OpenACC | acc | OpenACC parallelism |
| Async | async | Asynchronous execution |

### Specialized Dialects
| Dialect | Namespace | Description |
|---------|-----------|-------------|
| TOSA | tosa | Tensor Operator Set Architecture |
| Quant | quant | Quantization operations |
| EmitC | emitc | C/C++ code emission |
| PDL | pdl | Pattern Descriptor Language |
| IRDL | irdl | IR Dialect Definition |
| SparseTensor | sparse_tensor | Sparse tensor support |
| ArmSME | arm_sme | ARM SME operations |
| ArmSVE | arm_sve | ARM SVE operations |
| X86 | x86 | x86 intrinsics |

## Reference Documentation

### Core Infrastructure
- [01 - Overview & Architecture](references/01-overview-architecture.md) - MLIR design, philosophy, project structure
- [02 - Language Reference](references/02-langref-ir-core.md) - Complete MLIR language reference
- [03 - Types & Attributes](references/03-types-and-attributes.md) - Builtin types and attributes reference
- [04 - Operations, Blocks, Regions](references/04-operations-blocks-regions.md) - IR structure and APIs

### Dialect Definition
- [05 - Dialect Definition](references/05-dialect-definition.md) - Creating and registering dialects
- [06 - Operations (ODS)](references/06-operations-ods.md) - Operation Definition Specification
- [07 - Attributes & Types (ODS)](references/07-attributes-types-ods.md) - Type and attribute definitions in ODS

### Pass Infrastructure & Rewriting
- [08 - Pass Infrastructure](references/08-pass-infrastructure.md) - Pass management and built-in passes
- [09 - Pattern Rewriting](references/09-pattern-rewriting.md) - Pattern matching and rewriting framework
- [10 - Dialect Conversion](references/10-dialect-conversion.md) - Dialect conversion framework
- [11 - Declarative Rewrites](references/11-declarative-rewrites.md) - DRR (Declarative Rewrite Rules)
- [12 - PDLL](references/12-pdll.md) - Pattern Descriptor Language

### Core Dialects
- [13 - Builtin & Func Dialects](references/13-dialect-builtin-func.md) - Module, functions, symbols
- [14 - Arith, Math & Index Dialects](references/14-dialect-arith-math-index.md) - Arithmetic and math operations
- [15 - MemRef & Tensor Dialects](references/15-dialect-memref-tensor.md) - Memory and tensor operations

### Control Flow & Data Structure Dialects
- [16 - SCF & ControlFlow Dialects](references/16-dialect-scf-controlflow.md) - Structured and explicit control flow
- [17 - Affine Dialect](references/17-dialect-affine.md) - Polyhedral/affine operations
- [18 - Vector Dialect](references/18-dialect-vector.md) - SIMD vector operations

### Computation & Transformation Dialects
- [19 - Linalg Dialect](references/19-dialect-linalg.md) - Linear algebra structured operations
- [20 - Transform Dialect](references/20-dialect-transform.md) - Composable transformations
- [21 - Bufferization](references/21-dialect-bufferization.md) - Tensor to memref conversion

### Target & Accelerator Dialects
- [22 - GPU & NVGPU Dialects](references/22-dialect-gpu-nvgpu.md) - GPU kernel operations
- [23 - SPIR-V Dialect](references/23-dialect-spirv.md) - SPIR-V shader/compute
- [24 - LLVM Dialect & Target](references/24-dialect-llvm-target.md) - LLVM IR representation
- [25 - Async, OpenMP & OpenACC](references/25-dialect-async-openmp-openacc.md) - Parallel dialects

### Specialized Dialects
- [26 - TOSA, Quant & Shape Dialects](references/26-dialect-tosa-quant-shape.md) - ML operator dialects
- [27 - PDL & IRDL Dialects](references/27-dialect-pdl-irdl.md) - Pattern and dialect definition
- [28 - Specialized Dialects](references/28-dialect-specialized.md) - EmitC, SparseTensor, ARM, XeGPU, etc.

### Infrastructure
- [29 - Interfaces](references/29-interfaces.md) - Operation, type, attribute, and dialect interfaces
- [30 - Traits](references/30-traits.md) - Operation traits reference
- [31 - Symbols & Data Layout](references/31-symbols-data-layout.md) - Symbol tables and data layout
- [32 - Bytecode & Diagnostics](references/32-bytecode-diagnostics.md) - Serialization, diagnostics, tracing

### Bindings, Tools & Tutorials
- [33 - Python Bindings](references/33-python-bindings.md) - MLIR Python API
- [34 - C API](references/34-c-api.md) - MLIR C interface
- [35 - Tools](references/35-tools.md) - mlir-opt, mlir-translate, LSP, etc.
- [36 - Toy Tutorial](references/36-tutorials-toy.md) - Complete Toy language tutorial
- [37 - Transform Tutorial](references/37-tutorials-transform.md) - Transform dialect tutorial
