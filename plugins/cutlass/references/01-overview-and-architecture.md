# CUTLASS: Overview and Architecture

## What is CUTLASS?

CUTLASS (CUDA Templates for Linear Algebra Subroutines and Solvers) is NVIDIA's open-source C++ template library for providing high-performance matrix multiplication (GEMM) and related tensor computation primitives on NVIDIA GPUs. It implements the entire hierarchy of parallel algorithms needed to achieve near-peak performance on NVIDIA GPU Tensor Cores, from thread-level operations to device-level kernel launches.

CUTLASS abstracts the complexity of programming GPU Tensor Cores into a composable, type-safe, and extensible set of C++ template components. Developers can customize GEMM and tensor operation kernels at every level of the GPU execution hierarchy, mixing and matching algorithmic strategies with data types, layouts, and architecture-specific features.

**Key characteristics:**
- Header-only C++ template library (C++17 or later)
- Targets NVIDIA Tensor Cores across all supported architectures
- Provides building blocks for constructing high-performance GEMM kernels
- Supports a wide range of data types including FP64, FP32, FP16, BF16, TF32, FP8, INT8, INT4, and block-scaled formats
- Includes a profiler (`cutlass_profiler`) for benchmarking and validating kernels
- Provides a library generation tool for producing compiled kernel libraries
- BSD-3-Clause open-source license

**GitHub repository:** https://github.com/NVIDIA/cutlass

---

## Design Philosophy

CUTLASS is built on three foundational principles that guide every aspect of its architecture.

### 1. Hierarchical Decomposition

The GPU execution model is inherently hierarchical: threads are grouped into warps, warps into thread blocks (CTAs), and thread blocks are dispatched across streaming multiprocessors (SMs). CUTLASS mirrors this hierarchy in its template design, decomposing GEMM and tensor operations into layers that map directly to the hardware:

```
Device Level     -- Kernel launch, grid configuration, workspace allocation
  Kernel Level   -- Grid-level synchronization, workspace management
    Threadblock  -- CTA-level tiling, shared memory management, pipeline orchestration
      Warp       -- Warp-level matrix multiply-accumulate (MMA), register allocation
        Thread   -- Per-thread memory operations, element-wise computation
```

Each layer is parameterized independently, enabling fine-grained control over data movement and computation at every level of the hierarchy. This decomposition allows CUTLASS to express a vast design space of GEMM algorithms through template composition.

### 2. Explicit Data Movement

CUTLASS makes data movement explicit and controllable. Every stage of data transfer -- from global memory to shared memory, from shared memory to registers, and from registers back to global memory -- is represented as a distinct, customizable operation. This philosophy stems from the observation that data movement, not computation, is typically the bottleneck in dense linear algebra kernels.

Data movement paths in a typical CUTLASS GEMM kernel:

```
Global Memory (DRAM)
    |  <-- Load A, B tiles via memcpy_async or TMA (Hopper+)
    v
Shared Memory (SMEM)
    |  <-- Pipeline with double/triple buffering, prefetch
    v
Register File
    |  <-- Warp-level MMA via Tensor Core instructions
    v
Accumulator (Registers)
    |  <-- Epilogue: apply activation, bias, scale, store
    v
Global Memory (DRAM)  -- Output C
```

### 3. Compute Hierarchy Aligned with Tensor Cores

CUTLASS's compute abstractions directly map to NVIDIA Tensor Core instruction sets:

| CUTLASS Abstraction | Hardware Mapping | Purpose |
|---|---|---|
| `arch::Mma` | Tensor Core MMA instructions | Warp-level matrix multiply-accumulate |
| `gemm::threadblock::Mma` | CTA-level orchestration | Tiling, shared memory, pipelining |
| `gemm::warp::Mma` | Warp-level scheduling | Register allocation, fragment loading |
| `gemm::thread::Mma` | SIMT fallback | Scalar matrix operations on non-Tensor-Core paths |
| `epilogue::Epilogue` | Memory store + post-process | Write results with optional fusion |

---

## Key Abstractions

CUTLASS 3.x introduced a refined hierarchy of abstractions. The following describes the primary concepts:

### Device Layer

The `device::GemmUniversalAdapter` (3.x) or `GemmDevice` (2.x) is the top-level entry point. It manages:
- Kernel launch configuration (grid dimensions, shared memory size)
- Workspace allocation for split-K or stream-K parallelization
- Device-level API that hides kernel details from the caller

```cpp
// CUTLASS 3.x device-level GEMM type alias
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,            // (M, N, K, L) tuple
    CollectiveOp,           // Collective store/load/MMA
    CollectiveEpilogue      // Epilogue operation
>;
using GemmDevice = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

### Kernel Layer

The kernel layer defines the entry point for the CUDA kernel. It coordinates the execution of the collective operation across all threadblocks in the grid:

```cpp
// CUTLASS 3.x kernel structure
namespace cutlass::gemm::kernel {
template <
    class ProblemShape_,
    class CollectiveMainloop_,
    class CollectiveEpilogue_,
    class TileScheduler_ = void
>
struct GemmUniversal {
    // Kernel entry point
    CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
        // Initialize tiled MMA and epilogue
        // Iterate over tile scheduler
        // Call collective mainloop and epilogue
    }
};
}
```

### Collective Layer (CUTLASS 3.x)

The collective layer is a CUTLASS 3.x innovation that encapsulates the mainloop of a GEMM kernel -- loading tiles of A and B from global memory, performing the MMA operation, and managing shared memory. It unifies the previously separate threadblock and warp abstractions into a single, composable component:

```cpp
// Collective MMA for Hopper SM90 using TMA
using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::collective::KernelTmaWarpSpecialized,
    TileShape,                          // (TM, TN, TK)
    ElementA, LayoutA,                  // A matrix type and layout
    ElementB, LayoutB,                  // B matrix type and layout
    TiledMma,                           // Tiled MMA operation
    GmemTiledCopyA, SmemLayoutA,        // A memory copy and SMEM layout
    GmemTiledCopyB, SmemLayoutB         // B memory copy and SMEM layout
>;
```

### TiledMMA and MMA Atoms

At the lowest level of the compute hierarchy are MMA atoms -- indivisible matrix multiply-accumulate operations that map directly to hardware Tensor Core instructions:

```cpp
// MMA Atom for SM90 Tensor Core, FP16, 16x8x16 (m-n-k)
using MmaAtom = cutlass::gemm::AtomMma<
    cutlass::arch::Sm90,                // Architecture
    16, 8, 16,                          // M, N, K dimensions
    ElementA, LayoutA,                  // A type/layout
    ElementB, LayoutB,                  // B type/layout
    ElementAccum, cutlass::layout::RowMajor  // Accumulator type/layout
>;

// TiledMma repeats the atom across the warp group
using TiledMma = cutlass::gemm::collective::TiledMma<
    MmaAtom,
    TileShape                           // Overall tile size
>;
```

---

## CUTLASS 2.x vs 3.x Design Differences

CUTLASS 3.x represents a significant architectural evolution from 2.x. The two versions are not API-compatible, though 2.x APIs remain available for backward compatibility.

### Structural Comparison

| Aspect | CUTLASS 2.x | CUTLASS 3.x |
|---|---|---|
| Core paradigm | Layer-by-layer (thread -> warp -> CTA -> device) | Collective + CuTe-based composable operations |
| Tensor abstraction | Custom layout/stride types | CuTe tensor library (unified layout algebra) |
| Indexing | Manual offset computation | CuTe automatic layout composition |
| Mainloop | Separate `Mma` + `Epilogue` at each scope | Unified `CollectiveMainloop` + `CollectiveEpilogue` |
| Memory copy | Hand-written load/store with predicates | CuTe `copy()` with automatic vectorization |
| Hopper support | Limited / partial | Full TMA, warp-specialized, warp-group MMA |
| Blackwell support | Not available | Full support including block-scaled MMA |
| Kernel launch | `GemmUniversal<...>` | `GemmUniversalAdapter<Kernel>` |
| Policy dispatch | Tag-dispatch classes | Named policy types + compile-time constants |
| Python | Minimal bindings | Full Python DSL with op DSL |

### Migration Path

CUTLASS provides both 2.x and 3.x APIs in the same codebase. Key migration considerations:

1. **Layout types**: 2.x uses `cutlass::layout::RowMajor` / `ColumnMajor`; 3.x uses CuTe layouts
2. **Thread-level GEMM**: 2.x thread-level components map to CuTe `gemm()` operations
3. **Warp-level GEMM**: 2.x `warp::Mma` maps to CuTe `TiledMma`
4. **Epilogue**: 2.x `epilogue::Epilogue` maps to 3.x `collective::Epilogue`

### When to Use Each Version

- **Use 3.x** for: Hopper (SM90) and Blackwell (SM100+) kernels, new development, TMA-based kernels, mixed-precision block-scaled operations, Python-based kernel construction
- **Use 2.x** for: Legacy kernel maintenance, Volta (SM70) and Turing (SM75) optimized paths, SIMT-based kernels, architectures where 2.x tuning is well-established

---

## CuTe Library Integration in 3.x

CuTe is the tensor algebra library at the heart of CUTLASS 3.x. It provides a unified framework for representing and manipulating tensors, layouts, and operations on GPUs.

### Core CuTe Concepts

**Layout**: A mapping from a logical coordinate space to a linear offset space. Defined by a shape (hierarchy of integers) and a stride (hierarchy of integer offsets).

```cpp
// CuTe layout: 128 elements with stride 1 (contiguous)
auto layout = make_layout(make_shape(128), make_stride(1));

// 2D layout: 128 x 64 matrix in row-major order
auto layout_2d = make_layout(
    make_shape(128, 64),          // Shape: 128 rows, 64 cols
    make_stride(64, 1)            // Stride: row stride=64, col stride=1
);
```

**Tensor**: A pairing of a storage pointer with a layout. Tensors can reside in any memory space (global, shared, register).

```cpp
// Create a tensor from a global memory pointer and layout
auto tensor = make_tensor(ptr, layout);
```

**Copy**: CuTe provides `copy()` operations that automatically handle vectorization, predication, and memory space transitions.

```cpp
// Copy from global to shared memory using a tiled copy
copy(tiled_copy, tGxG, tSxS);  // global tensor -> shared tensor
```

**MMA**: CuTe wraps Tensor Core MMA operations as composable operations on tiled tensors.

```cpp
// Perform MMA operation
gemm(mma_tiled, tCrA, tCrB, tCrC);  // A, B -> accumulate into C
```

### CuTe Benefits

1. **Correct by construction**: Layout algebra ensures index computations are always correct
2. **Composable**: Layouts can be composed, sliced, and partitioned without manual index arithmetic
3. **Architecture-portable**: Same CuTe code generates optimal instructions for SM70 through SM100+
4. **Inspectable**: Layouts and tensors can be printed and debugged at compile time or runtime

---

## Tag-Dispatch Policies vs Named Types

### CUTLASS 2.x Tag Dispatch

CUTLASS 2.x uses tag-dispatch to select policies at compile time. Tags are empty structs that carry type information:

```cpp
// Tag dispatch in CUTLASS 2.x
struct GemmShape<128, 128, 32> {};   // Tile size tag
struct OpClassSimt {};                // SIMT operation class tag
struct OpClassTensorOp {};            // Tensor Core operation class tag
struct ArchTag<SM80> {};             // Architecture tag

// Policy selection via tag
template <>
struct DefaultMma<OpClassTensorOp, ArchTag<SM80>,
                  GemmShape<128, 128, 32>,
                  half_t, layout::ColumnMajor,
                  half_t, layout::ColumnMajor,
                  layout::RowMajor> {
    // Specialized for TensorOp on SM80 with specific tile sizes
};
```

### CUTLASS 3.x Named Types

CUTLASS 3.x replaces many tag-dispatch patterns with named types and compile-time constants that are more explicit and composable:

```cpp
// Named types in CUTLASS 3.x
using TileShape = Shape<_128, _128, _32>;    // Cute shape
using ClusterShape = Shape<_1, _1, _1>;       // Cluster shape

// Named operation descriptors
using KernelSchedule = cutlass::gemm::collective::KernelTmaWarpSpecialized;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
```

---

## Performance Goals and Optimization Approach

### Performance Targets

CUTLASS aims to achieve a high fraction of theoretical peak throughput for each target architecture:

| Architecture | Tensor Core Peak (FP16) | CUTLASS Target |
|---|---|---|
| SM70 (Volta V100) | 125 TFLOPS | >80% of peak |
| SM75 (Turing T4) | 65 TFLOPS | >80% of peak |
| SM80 (Ampere A100) | 312 TFLOPS | >85% of peak |
| SM89 (Ada L40) | 181 TFLOPS | >85% of peak |
| SM90 (Hopper H100) | 989 TFLOPS (FP16), 1979 TFLOPS (FP8) | >90% of peak |
| SM100 (Blackwell B200) | 2250 TFLOPS (FP16), 4500 TFLOPS (FP8) | >90% of peak |

### Optimization Techniques

1. **Tiling and double/triple buffering**: Overlap computation with data movement by keeping multiple tiles of data in flight
2. **Shared memory padding and bank-conflict avoidance**: Layout shared memory access patterns to avoid bank conflicts
3. **Warp-specialized pipelines** (SM90+): Dedicate warps to data movement vs. computation for better occupancy
4. **TMA (Tensor Memory Accelerator)** (SM90+): Hardware-accelerated asynchronous bulk copy from global to shared memory
5. **Warp-group MMA** (SM90+): 4-warp collaborative MMA using the `wgmma.mma_async` instruction
6. **Register-pressure-aware scheduling**: Balance register usage between MMA fragments and pipeline depth
7. **Swizzling**: Shared memory access pattern transformation to eliminate bank conflicts
8. **Kernel fusion via epilogue**: Combine GEMM with activation functions, bias addition, scaling, and other element-wise operations in a single kernel

### Roofline Model

CUTLASS kernels are typically memory-bandwidth-bound for small problem sizes and compute-bound for large problem sizes. The library provides the `cutlass_profiler` tool to measure and report achieved throughput relative to theoretical peak.

---

## Supported Operations

### Matrix Multiplication (GEMM)

The primary operation in CUTLASS. Supports all standard GEMM variants:

- **Dense GEMM**: `C = alpha * A * B + beta * C`
- **Batched GEMM**: Multiple independent GEMM operations in a single kernel launch
- **GEMM with split-K/stream-K**: Distribute the K-dimension reduction across multiple threadblocks
- **Mixed-precision GEMM**: Different input/output/accumulator types (e.g., FP16 inputs, FP32 accumulation, FP16 output)
- **GEMM with epilogue fusion**: ReLU, GELU, bias addition, scaling, element-wise operations

### Convolution

CUTLASS implements convolution as implicit GEMM, transforming convolution operations into GEMM form:

- **Forward convolution**: Direct convolution computation
- **Backward data convolution** (deconvolution): Gradient computation w.r.t. input
- **Backward filter convolution**: Gradient computation w.r.t. filter
- **Dgrad/Fprop/Wgrad**: Specific convolution operation variants
- Supports 1D, 2D, and 3D convolutions

### Reduction

Parallel reduction operations including:

- **Tensor reduction**: Reduce along specified dimensions
- **Split-K reduction**: Combine partial GEMM results
- **Array reduction**: Reduce elements in contiguous arrays

### Tensor Operations

Additional tensor operations supported by CUTLASS:

- **Tensor contraction**: Generalized tensor multiplication with arbitrary index contraction
- **Tensor permutation**: Reordering of tensor dimensions
- **Quantization**: Mixed-precision operations with quantized types
- **Block-scaled GEMM** (SM100+): GEMM using block-scaled FP4/FP8 formats with per-block scaling factors

---

## Architecture Support

CUTLASS supports the following NVIDIA GPU architectures. Each architecture has specific optimizations and Tensor Core capabilities:

### Volta (SM70) -- V100

- **Tensor Cores**: First generation (FP16 only)
- **MMA instruction**: `mma.sync` 16x16x4 (FP16 -> FP32 accumulate)
- **Shared memory**: 96 KB per SM (configurable)
- **Key features**: Warp-level matrix operations, cooperative groups
- **CUTLASS support**: Full 2.x API, limited 3.x

### Turing (SM75) -- T4, RTX 2080

- **Tensor Cores**: Second generation (FP16, INT8, INT4, INT1)
- **MMA instruction**: `mma.sync` 16x8x8 (FP16), 8x8x16 (INT8)
- **Shared memory**: 64 KB per SM
- **Key features**: Fragment-based API, independent thread scheduling
- **CUTLASS support**: Full 2.x API, limited 3.x

### Ampere (SM80, SM89) -- A100, L40, RTX 3090

- **Tensor Cores**: Third generation (FP64, FP32 via TF32, FP16, BF16, INT8, INT4, INT1)
- **MMA instruction**: `mma.sync` 16x8x16 (FP16/BF16), 8x8x4 (FP64)
- **Shared memory**: Up to 164 KB per SM (configurable)
- **Key features**: Async copy from global to shared memory, cp.async, L2 cache residency control
- **CUTLASS support**: Full 2.x and 3.x APIs
- **SM89** (Ada Lovelace): Similar to SM80 with additional FP8 support

### Hopper (SM90, SM90a) -- H100

- **Tensor Cores**: Fourth generation (all Ampere types + FP8)
- **MMA instruction**: `wgmma.mma_async` (warp-group MMA, 64xN x K)
- **Shared memory**: Up to 228 KB per SM
- **Key features**:
  - TMA (Tensor Memory Accelerator): Hardware-accelerated bulk async copy
  - Warp-group MMA: 128 threads collaborate on a single large MMA
  - Dynamic shared memory allocation per threadblock
  - Cluster-level cooperative operations (CTAs in same cluster can share SMEM)
  - Warpgroup-specialized execution: separate data-movement and compute warps
- **CUTLASS support**: Full 3.x API with specialized collective operations
- **SM90a**: Enables all Tensor Core features including FP64 Tensor Cores

### Blackwell (SM100, SM100a, SM120, SM120a) -- B200, B100

- **Tensor Cores**: Fifth generation
- **MMA instruction**: `wgmma.mma_async` with block-scaled formats
- **Shared memory**: Up to 228 KB per SM
- **Key features**:
  - Block-scaled FP4/FP8 MMA with per-block scaling factors
  - Second-generation TMA with enhanced capabilities
  - NVFP4 and MX formats for efficient mixed-precision computation
  - Enhanced epilogue fusion capabilities
- **CUTLASS support**: Full 3.x API with block-scaled collective operations

### Architecture Selection Guide

```cpp
// Build for specific architecture
cmake .. -DCUTLASS_NVCC_ARCHS=90a    // Hopper H100
cmake .. -DCUTLASS_NVCC_ARCHS=100a   // Blackwell B200
cmake .. -DCUTLASS_NVCC_ARCHS=80     // Ampere A100
cmake .. -DCUTLASS_NVCC_ARCHS=75     // Turing T4
cmake .. -DCUTLASS_NVCC_ARCHS=70     // Volta V100
```

---

## License

CUTLASS is released under the **BSD-3-Clause** license:

```
Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.
```

This permissive license allows CUTLASS to be used freely in both open-source and proprietary projects.

---

## Version History and Evolution

### CUTLASS 1.x (2017-2018)

- Initial release focusing on Volta (SM70) Tensor Core GEMM
- Basic template design with limited composability
- Supported FP16 only for Tensor Core operations

### CUTLASS 2.x (2019-2022)

- **2.0**: Complete redesign with hierarchical template decomposition
- **2.1**: Added Turing (SM75) support, INT8 Tensor Cores
- **2.2**: Added Ampere (SM80) support, BF16, TF32, async copy, `cp.async`
- **2.3**: Enhanced convolution support, improved epilogue fusion
- **2.4-2.5**: Improved performance, bug fixes, additional examples
- **2.6-2.8**: Stream-K parallelization, additional epilogue fusion, improved profiler
- **2.9-2.11**: Enhanced mixed-precision, Ada (SM89) support, FP8 preliminary support
- **2.12+**: Continued maintenance and bug fixes
- Key innovations: Thread/Warp/CTA/Device hierarchy, tag-dispatch policies, interleaved layouts

### CUTLASS 3.x (2023-present)

- **3.0**: Introduced CuTe tensor library, collective operation model
- **3.1**: Enhanced Hopper support, TMA, warp-specialized kernels
- **3.2**: Improved Python bindings, additional epilogue schedules
- **3.3**: Blackwell SM100 preliminary support, block-scaled types
- **3.4**: Enhanced block-scaled GEMM, NVFP4 support, improved Python DSL
- **3.5+**: Continued Blackwell maturation, performance improvements, expanded examples
- Key innovations: CuTe layout algebra, collective mainloop/epilogue, TMA integration, Python DSL

### Current State

As of 2025, CUTLASS 3.x is the actively developed branch. CUTLASS 2.x APIs remain available but receive primarily maintenance updates. New features and architecture support are developed in the 3.x API. The repository maintains both APIs in a single codebase, and developers can choose which API to use based on their target architecture and requirements.

---

## Summary

CUTLASS provides a comprehensive framework for implementing high-performance dense linear algebra on NVIDIA GPUs. Its hierarchical design, from MMA atoms to device-level adapters, allows developers to construct custom GEMM and tensor operation kernels that achieve near-peak throughput on Tensor Cores. The evolution from 2.x's layer-by-layer approach to 3.x's CuTe-based collective model represents a fundamental improvement in expressiveness, correctness, and composability. With support for Volta through Blackwell architectures and a wide range of data types and operations, CUTLASS is the foundational building block for high-performance linear algebra in modern deep learning and scientific computing workloads.
