---
name: cutlass
description: NVIDIA CUTLASS CUDA Template Library - comprehensive reference for high-performance matrix multiplication (GEMM), convolution, tensor operations, and CuTe DSL across all GPU architectures (Volta through Blackwell)
version: 3.8
---

# CUTLASS - CUDA Template Library for Linear Algebra

CUTLASS (CUDA Templates for Linear Algebra Subroutines) is NVIDIA's open-source C++ template library for implementing high-performance matrix-matrix multiplication (GEMM) and related computations at all levels and scales within CUDA. It provides near-optimal utilization of peak theoretical throughput on NVIDIA GPUs.

## Overview

CUTLASS provides:
- **High-performance GEMM operations** with near-optimal utilization of peak theoretical throughput
- **Support for multiple NVIDIA architectures**: Volta (SM70), Turing (SM75), Ampere (SM80/SM89), Ada (SM89), Hopper (SM90), and Blackwell (SM100/SM101/SM103/SM120)
- **Extensive data type support**: FP64, FP32, TF32, FP16, BF16, FP8 (e5m2, e4m3), INT8, INT4, INT2, binary 1b, and block-scaled types (NVFP4, MXFP4/6/8)
- **CUTLASS DSLs**: Python native interfaces for writing high-performance CUDA kernels (CuTe DSL)
- **CuTe library**: A modern C++ tensor abstraction for composable GPU micro-kernels

## When to Use This Skill

Use this skill when:
- Writing or optimizing GEMM kernels using CUTLASS
- Implementing convolution operations with CUTLASS
- Using CuTe for tensor operations and layout algebra
- Working with Tensor Core operations on NVIDIA GPUs
- Implementing mixed-precision training or inference kernels
- Building custom epilogue fusion kernels
- Using TMA (Tensor Memory Accelerator) on Hopper/Blackwell
- Implementing fused attention (FMHA) kernels
- Working with sparse or block-scaled GEMM operations
- Profiling and benchmarking CUTLASS kernels

## Quick Reference

### CUTLASS 3.x GEMM (Recommended)
```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

// Define types
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementD = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// Use CollectiveBuilder for convenient kernel assembly
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 8,
    ElementB, LayoutB, 8,
    ElementC,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
    LayoutD, LayoutC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>;

using Kernel = cutlass::gemm::kernel::GemmUniversal<
    cutlass::gemm::collective::CollectiveOp,
    cutlass::epilogue::collective::EpilogueOp
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

// Launch
Gemm gemm_op;
auto args = Gemm::Arguments{
    {M, N, K},
    {ptr_A, stride_A},
    {ptr_B, stride_B},
    {ptr_C, stride_C},
    {ptr_D, stride_D},
    {alpha, beta}
};
gemm_op(args);
```

### CUTLASS 2.x GEMM
```cpp
#include "cutlass/gemm/device/gemm.h"

using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,    // ElementA, LayoutA
    cutlass::half_t, cutlass::layout::ColumnMajor,  // ElementB, LayoutB
    float, cutlass::layout::RowMajor,               // ElementC, LayoutC
    float,                                           // ElementAccumulator
    cutlass::arch::OpClassTensorOp,                  // OpClass
    cutlass::arch::Sm80,                             // ArchTag
    cutlass::gemm::GemmShape<128, 128, 32>,          // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 32>,            // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,             // InstructionShape
    cutlass::epilogue::thread::LinearCombination<
        float, 4, float, float>
>;

Gemm gemm_op;
gemm_op({M, N, K}, alpha, ptr_A, lda, ptr_B, ldb, beta, ptr_C, ldc, ptr_D, ldd);
```

## Architecture Support

| Architecture | SM Version | Key Features |
|---|---|---|
| Volta | SM70 | First Tensor Cores (FP16), WMMA API |
| Turing | SM75 | Tensor Cores (INT8/INT4/INT1), MMA instructions |
| Ampere | SM80/SM89 | TF32, BF16, async copy, cp.async |
| Hopper | SM90 | TMA, GMMA, WGMMA, thread block clusters, warp specialization |
| Blackwell | SM100+ | UMMA, block-scaled types (NVFP4/MXFP), distributed GEMM, green contexts |

## Key Concepts

- **GEMM Hierarchy**: Device > Kernel > Collective > Tiled MMA/Copy > Atom
- **Collective**: Mainloop computation shared across threadblock/warp
- **Epilogue**: Post-GEMM operations (activation, scaling, type conversion)
- **CuTe Layout**: Formalized layout algebra for thread-to-data mapping
- **TensorRef/TensorView**: Lightweight tensor abstractions with layout
- **Pipeline**: Multi-stage data movement with producer-consumer pattern
- **Split-K**: Parallel reduction across K dimension
- **Warp Specialization**: Separate producer and consumer warp groups (SM90+)

## Reference Chapters

Detailed reference documentation is organized in the `references/` directory:

### Fundamentals
- [01-overview-and-architecture.md](references/01-overview-and-architecture.md) - Project overview, design philosophy, and architecture
- [02-quickstart.md](references/02-quickstart.md) - Prerequisites, build instructions, first kernel
- [03-code-organization.md](references/03-code-organization.md) - Directory structure and code layout
- [04-data-types.md](references/04-data-types.md) - Numeric types, conversions, and type traits
- [05-layout-system.md](references/05-layout-system.md) - Matrix/tensor layout definitions (RowMajor, ColumnMajor, etc.)
- [06-tensor-abstractions.md](references/06-tensor-abstractions.md) - TensorRef, TensorView, coordinate systems
- [38-terminology.md](references/38-terminology.md) - Comprehensive glossary of CUTLASS terminology

### GEMM API
- [07-gemm-api-2x.md](references/07-gemm-api-2x.md) - CUTLASS 2.x GEMM API (device, threadblock, warp, thread levels)
- [08-gemm-api-3x.md](references/08-gemm-api-3x.md) - CUTLASS 3.x GEMM API (CollectiveBuilder, GemmUniversal)
- [09-collective-operations.md](references/09-collective-operations.md) - Collective builders, MMA mainloops, dispatch policies

### Post-Processing
- [10-epilogue.md](references/10-epilogue.md) - Epilogue operations, output iterators, scaling
- [26-epilogue-fusion.md](references/26-epilogue-fusion.md) - Epilogue fusion patterns, activation fusion, back-to-back GEMMs

### Data Movement
- [11-transform-operations.md](references/11-transform-operations.md) - Tile iterators, predicated access, transpose
- [13-pipeline.md](references/13-pipeline.md) - Multi-stage pipelines, TMA, async operations

### Operations
- [12-convolution.md](references/12-convolution.md) - Convolution operations (Conv2D, Conv3D, implicit GEMM)
- [20-tensor-core-operations.md](references/20-tensor-core-operations.md) - Tensor Core MMA instructions per architecture
- [27-reduction-operations.md](references/27-reduction-operations.md) - Reduction operations and kernels

### CuTe Library
- [15-cute-overview.md](references/15-cute-overview.md) - CuTe library overview and design philosophy
- [16-cute-layout.md](references/16-cute-layout.md) - CuTe layout algebra, shape, stride, composition
- [17-cute-tensor.md](references/17-cute-tensor.md) - CuTe tensor abstractions and engines
- [18-cute-mma-atoms.md](references/18-cute-mma-atoms.md) - CuTe MMA atoms, tiled MMA, copy atoms
- [19-cute-algorithms.md](references/19-cute-algorithms.md) - CuTe algorithms (copy, gemm, fill, prefetch)

### Advanced Topics
- [14-architecture-support.md](references/14-architecture-support.md) - Architecture-specific features (SM50-SM120)
- [21-mixed-precision.md](references/21-mixed-precision.md) - Mixed precision GEMM, type conversion
- [22-sparse-gemm.md](references/22-sparse-gemm.md) - Sparse tensor operations and 2:4 sparsity
- [23-blockwise-scaling.md](references/23-blockwise-scaling.md) - Block-scaled GEMM (FP8, NVFP4, MX formats)
- [24-distributed-gemm.md](references/24-distributed-gemm.md) - Tensor Parallel GEMM with NVLink
- [25-fused-attention.md](references/25-fused-attention.md) - Fused Multi-Head Attention (FMHA)

### Architecture-Specific
- [31-hopper-features.md](references/31-hopper-features.md) - Hopper (SM90) features: TMA, GMMA, warp specialization
- [32-blackwell-features.md](references/32-blackwell-features.md) - Blackwell (SM100+) features: UMMA, green contexts

### Tools and Utilities
- [28-profiler.md](references/28-profiler.md) - CUTLASS Profiler usage and configuration
- [29-python-interface.md](references/29-python-interface.md) - Python bindings, CuTe DSL, code generation
- [30-examples-catalog.md](references/30-examples-catalog.md) - Complete examples catalog with descriptions
- [33-build-system.md](references/33-build-system.md) - CMake configuration, compilation flags, library builds
- [34-best-practices.md](references/34-best-practices.md) - Performance tuning, optimization tips, common patterns
- [35-functional-operations.md](references/35-functional-operations.md) - Math function objects, activation functions
- [36-coordinate-system.md](references/36-coordinate-system.md) - Coord template, tensor coordinates
- [37-subbyte-and-numeric.md](references/37-subbyte-and-numeric.md) - Subbyte references, numeric conversion
