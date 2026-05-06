---
name: xla
description: >
  Comprehensive reference for XLA (Accelerated Linear Algebra) compiler - covering architecture,
  operation semantics, HLO IR, compilation pipeline, GPU/CPU/TPU backends, PJRT API, MLIR integration,
  custom calls, autotuning, SPMD partitioning, debugging tools, and build system.
version: "2.0"
---

# XLA (Accelerated Linear Algebra)

XLA is an open-source machine learning (ML) compiler for GPUs, CPUs, and ML accelerators. It takes models from popular ML frameworks such as PyTorch, TensorFlow, and JAX, and optimizes them for high-performance execution across different hardware platforms.

## Key Objectives

- **Improve execution speed**: Compile subgraphs to reduce overhead, fuse pipelined operations, specialize for known tensor shapes
- **Improve memory usage**: Analyze and schedule memory, eliminate intermediate storage buffers
- **Reduce reliance on custom ops**: Fuse low-level ops to match hand-tuned custom op performance
- **Improve portability**: Easy to write new backends for novel hardware

## Supported Frameworks and Hardware

- **Frameworks**: JAX, TensorFlow, PyTorch
- **Hardware**: NVIDIA GPUs (CUDA), AMD GPUs (ROCm), CPUs (x86/ARM), TPUs, custom accelerators
- **Project**: Part of the OpenXLA ecosystem

## Compilation Pipeline Overview

```
ML Framework → StableHLO → HLO → Target-Independent Optimizations → Target-Specific Optimizations → Code Generation
```

1. **StableHLO Input**: ML frameworks produce StableHLO operations
2. **HLO Conversion**: StableHLO converted to internal HLO dialect
3. **Optimizations**: CSE, fusion, buffer analysis, layout assignment
4. **Backend Processing**: Target-specific HLO optimizations and code generation
5. **Code Generation**: LLVM IR → PTX (GPU), native code (CPU), or device-specific binary

## Quick Reference

### Basic Computation Example

```cpp
#include "xla/client/xla_builder.h"

xla::XlaBuilder builder("add_vectors");

// Create parameters
xla::XlaOp x = xla::Parameter(&builder, 0,
    xla::ShapeUtil::MakeShape(xla::F32, {1024}), "x");
xla::XlaOp y = xla::Parameter(&builder, 1,
    xla::ShapeUtil::MakeShape(xla::F32, {1024}), "y");

// Build computation
xla::XlaOp result = xla::Add(x, y);

// Build and compile
auto computation = builder.Build().value();
```

### HLO Text Format Example

```
HloModule matmul_example

ENTRY main {
  %p0 = f32[1024,512]{1,0} parameter(0)
  %p1 = f32[512,2048]{1,0} parameter(1)
  ROOT %dot = f32[1024,2048]{1,0} dot(%p0, %p1),
         lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
```

### Common Operations

```cpp
// Element-wise operations
XlaOp Add(XlaOp lhs, XlaOp rhs);
XlaOp Mul(XlaOp lhs, XlaOp rhs);
XlaOp Sub(XlaOp lhs, XlaOp rhs);
XlaOp Div(XlaOp lhs, XlaOp rhs);

// Data manipulation
XlaOp Reshape(XlaOp operand, ArraySlice<int64> dimensions);
XlaOp Broadcast(XlaOp operand, ArraySlice<int64> broadcast_sizes);
XlaOp Slice(XlaOp operand, ArraySlice<int64> start, ArraySlice<int64> limit, ArraySlice<int64> strides);
XlaOp Transpose(XlaOp operand, ArraySlice<int64> permutation);
XlaOp ConcatInDim(ArraySlice<XlaOp> operands, int64_t dimension);

// Linear algebra
XlaOp Dot(XlaOp lhs, XlaOp rhs);
XlaOp DotGeneral(XlaOp lhs, XlaOp rhs, DotDimensionNumbers dnums);
XlaOp Conv(XlaOp lhs, XlaOp rhs, ArraySlice<int64> strides, Padding padding);

// Collective operations
XlaOp AllReduce(XlaOp operand, XlaComputation computation, ReplicaGroupVector groups);
XlaOp AllGather(XlaOp operand, int64_t dim, int64_t count, ReplicaGroupVector groups);

// Control flow
XlaOp While(XlaComputation condition, XlaComputation body, XlaOp init);
XlaOp Conditional(XlaOp pred, XlaOp true_val, XlaComputation true_comp,
                  XlaOp false_val, XlaComputation false_comp);
```

### Common Tools

```bash
# Dump HLO from JAX
XLA_FLAGS=--xla_dump_to=/tmp/hlo_dump python my_program.py

# Run HLO module
run_hlo_module --platform=CUDA --reference_platform=Interpreter computation.hlo

# Optimize and inspect HLO
hlo-opt --platform=CUDA --stage=hlo input.hlo
hlo-opt --passes=algebraic-simplifier input.hlo

# Deviceless GPU compilation
hlo-opt --platform=CUDA --stage=llvm \
  --xla_gpu_target_config_filename=gpu_specs/a100_pcie_80.txtpb input.hlo
```

## Documentation Structure

### Overview and Architecture
- [01-overview-and-architecture](references/01-overview-and-architecture.md) - XLA overview, objectives, and compiler architecture
- [02-shapes-and-layout](references/02-shapes-and-layout.md) - Shapes, layout, tiling, memory spaces, and indexing
- [03-broadcasting](references/03-broadcasting.md) - Broadcasting semantics, rules, and composition

### Operation Semantics
- [04-operation-semantics-elementwise](references/04-operation-semantics-elementwise.md) - Element-wise unary operations (Abs, Sin, Cos, Exp, etc.)
- [05-operation-semantics-binary](references/05-operation-semantics-binary.md) - Binary operations (Add, Mul, Div, And, Or, etc.)
- [06-operation-semantics-collective](references/06-operation-semantics-collective.md) - Collective operations (AllReduce, AllGather, AllToAll, etc.)
- [07-operation-semantics-control-flow](references/07-operation-semantics-control-flow.md) - Control flow (While, Conditional, Reduce, Sort, etc.)
- [08-operation-semantics-convolution](references/08-operation-semantics-convolution.md) - Convolutions, FFT, and TriangularSolve
- [09-operation-semantics-data-manipulation](references/09-operation-semantics-data-manipulation.md) - Data manipulation (Reshape, Slice, Broadcast, Gather, etc.)
- [10-operation-semantics-linear-algebra](references/10-operation-semantics-linear-algebra.md) - Linear algebra (Dot, Cholesky, BatchNorm)
- [11-operation-semantics-io-and-other](references/11-operation-semantics-io-and-other.md) - Custom calls, I/O, RNG, tokens, and misc operations

### Compiler Infrastructure
- [12-hlo-ir](references/12-hlo-ir.md) - HLO IR: module structure, instruction set, text format, verification
- [13-compilation-pipeline](references/13-compilation-pipeline.md) - Compilation pipeline stages from StableHLO to native code
- [14-hlo-passes](references/14-hlo-passes.md) - HLO optimization and transformation passes
- [15-gpu-backend](references/15-gpu-backend.md) - GPU backend architecture, pipeline, and runtime
- [16-gpu-emitters](references/16-gpu-emitters.md) - GPU code generation: emitters, partitioning, vectorization
- [17-cpu-backend](references/17-cpu-backend.md) - CPU backend architecture and code generation
- [18-tpu-backend](references/18-tpu-backend.md) - TPU backend, memory model, and SparseCore

### Integration and Extension
- [19-developing-new-backend](references/19-developing-new-backend.md) - How to develop a new XLA backend
- [20-pjrt-api](references/20-pjrt-api.md) - PJRT uniform device API and plugin mechanism
- [21-mlir-integration](references/21-mlir-integration.md) - MLIR-HLO dialect integration and TableGen
- [22-custom-calls](references/22-custom-calls.md) - Custom calls and XLA FFI binding
- [23-async-operations](references/23-async-operations.md) - Async HLO instructions and syntax sugar
- [24-autotuning](references/24-autotuning.md) - Autotuning framework and persisted results
- [25-tools](references/25-tools.md) - XLA tools: run_hlo_module, hlo-opt, ptx-opt, isolate_hlo
- [26-build-system](references/26-build-system.md) - Building XLA from source with Bazel
- [27-debugging](references/27-debugging.md) - Debugging, HLO dumps, error codes, determinism
- [28-aliasing](references/28-aliasing.md) - Input/output buffer aliasing and donation
- [29-spmd-partitioner](references/29-spmd-partitioner.md) - SPMD partitioning, sharding, and GSPMD
- [30-symbolic-expression](references/30-symbolic-expression.md) - Symbolic expressions, indexing analysis, and dynamic shapes
