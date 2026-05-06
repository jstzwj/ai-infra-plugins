# TileLang Overview and Architecture

## Table of Contents

- [1. What is TileLang](#1-what-is-tilelang)
- [2. Design Philosophy](#2-design-philosophy)
- [3. Three-Level Programming Model](#3-three-level-programming-model)
- [4. Architecture Overview](#4-architecture-overview)
- [5. Core Components](#5-core-components)
- [6. Package Directory Structure](#6-package-directory-structure)
- [7. Compilation Pipeline Deep Dive](#7-compilation-pipeline-deep-dive)
- [8. Comparison with CUDA, Triton, and CUTLASS](#8-comparison-with-cuda-triton-and-cutlass)
- [9. Hardware Support](#9-hardware-support)
- [10. Performance Benchmarks](#10-performance-benchmarks)
- [11. Ecosystem and Tooling](#11-ecosystem-and-tooling)

---

## 1. What is TileLang

TileLang (tile-lang) is a concise domain-specific language (DSL) designed to streamline the development of high-performance GPU and CPU kernels for AI workloads. It is built on top of [Apache TVM](https://tvm.apache.org/), leveraging the Tensor Intermediate Representation (TIR) as its core compiler infrastructure. TileLang enables developers to express complex GPU kernel algorithms using familiar Pythonic syntax while achieving performance that matches or exceeds hand-tuned CUDA/CUTLASS implementations.

### Key Characteristics

- **Pythonic DSL**: Write GPU kernels using Python syntax with TVM's TIR scripting support. The `@T.prim_func` decorator and `T.Kernel` context manager provide a clean, intuitive interface.
- **Performance-First**: Achieves near-peak hardware utilization through automatic dispatching to Tensor Core (MMA, WGMMA, TCGEN05), TMA (Tensor Memory Access), and optimized memory movement patterns.
- **Compiler-Driven**: The TileLang compiler performs layout inference, memory planning, pipeline scheduling, and architecture-specific optimizations automatically.
- **Multi-Backend**: Supports NVIDIA CUDA, AMD ROCm/HIP, Apple Metal, CPU (LLVM), and WebGPU targets from a single source program.

### What TileLang is Used For

TileLang is designed for developing performance-critical GPU kernels such as:

| Category | Examples |
|----------|----------|
| Dense Linear Algebra | GEMM, Batched GEMM, Grouped GEMM |
| Quantized Compute | INT4/INT8 Dequantize-GEMM, FP8 Block-Scaled GEMM |
| Attention Mechanisms | FlashAttention (fwd/bwd), Grouped Query Attention, MLA Decoding |
| Sparse Operations | Block-Sparse Attention, 2:4 Structured Sparsity |
| Convolution | 2D Convolution with Im2Col |
| Elementwise / Reduction | LayerNorm, Softmax, Cast operations |
| Linear Attention | Flash Linear Attention, State-space models (Mamba) |

---

## 2. Design Philosophy

### 2.1 Productivity with Performance

TileLang's primary goal is to let developers express high-performance kernel algorithms without drowning in low-level details. A complete high-performance GEMM kernel can be written in under 30 lines of Python:

```python
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype, accum_dtype):
    @T.prim_func
    def gemm(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype),
             C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return gemm
```

This achieves performance comparable to cuBLAS on H100 and A100 GPUs by automatically:
- Dispatching to WGMMA on Hopper (SM90) and MMA on Ampere (SM80)
- Enabling TMA for global-to-shared copy when available
- Software pipelining with multi-stage buffering
- Layout inference for shared memory and fragment buffers

### 2.2 Portability

A single TileLang kernel source compiles to efficient code across multiple GPU architectures:

- **NVIDIA GPUs**: CUDA codegen with automatic Tensor Core dispatching (SM70 Volta WMMA through SM100 Blackwell TCGEN05)
- **AMD GPUs**: HIP/ROCm codegen with MFMA (Matrix Fused Multiply-Add) support on CDNA architectures
- **Apple Metal**: Metal Shading Language codegen for Apple Silicon GPUs
- **CPU**: LLVM-based codegen for x86 and ARM
- **WebGPU**: WGSL codegen for browser-based GPU compute

### 2.3 Flexibility

TileLang provides three abstraction levels that developers can choose from based on their needs:

1. **Hardware-Unaware Level**: Use high-level tile operations (`T.copy`, `T.gemm`, `T.reduce`) and let the compiler handle all architecture-specific details.
2. **Hardware-Aware with Tile Library**: Use tile-level primitives with explicit memory management and scheduling, selecting specific intrinsic paths.
3. **Hardware-Aware with Thread Primitives**: Drop down to PTX-level intrinsics, manual warp management, and explicit register control for maximum performance.

### 2.4 Optimization Support

The compiler provides rich optimization support:

- **Automatic Layout Inference**: Determines optimal data layouts for shared memory and fragment buffers based on downstream operations (GEMM, copy).
- **Software Pipelining**: Multi-stage pipeline scheduling with automatic barrier management (`T.Pipelined` with `num_stages`).
- **Memory Coalescing**: Automatic coalesced memory access patterns for copies.
- **TMA Utilization**: Automatic dispatching to Tensor Memory Access on SM90+ for bulk global-to-shared transfers.
- **Cache Management**: L2 cache hit ratio hints, eviction policy control, and swizzle patterns.
- **Register Pressure Management**: `T.annotate_min_blocks_per_sm`, `T.inc_max_nreg` / `T.dec_max_nreg` for occupancy tuning.

---

## 3. Three-Level Programming Model

TileLang provides a progressive abstraction model where developers can choose the level of control they need.

### Level 1: Hardware-Unaware (Beginner)

At this level, developers use high-level tile operations without knowledge of specific hardware features. The compiler makes all optimization decisions.

```python
@T.prim_func
def simple_gemm(A: T.Tensor((M, K), "float16"),
                B: T.Tensor((K, N), "float16"),
                C: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
        A_shared = T.alloc_shared((BM, BK), "float16")
        B_shared = T.alloc_shared((BK, BN), "float16")
        C_frag = T.alloc_fragment((BM, BN), "float32")

        T.clear(C_frag)
        for k in T.serial(T.ceildiv(K, BK)):
            T.copy(A[by * BM, k * BK], A_shared)
            T.copy(B[k * BK, bx * BN], B_shared)
            T.gemm(A_shared, B_shared, C_frag)
        T.copy(C_frag, C[by * BM, bx * BN])
```

**Key characteristics of Level 1:**
- Uses `T.copy()` without specifying copy mechanism -- compiler chooses TMA, cp.async, or SIMT
- Uses `T.gemm()` without specifying MMA type -- compiler dispatches to WGMMA, MMA, or WMMA
- No explicit barrier management
- No explicit layout annotations

### Level 2: Hardware-Aware with Tile Library (Developer)

At this level, developers explicitly control scheduling, memory hierarchy, and synchronization while still using the tile-level API.

```python
@T.prim_func
def pipelined_gemm(A: T.Tensor((M, K), "float16"),
                   B: T.Tensor((K, N), "float16"),
                   C: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
        # Multi-stage shared memory buffers for pipelining
        A_shared = T.alloc_shared((2, BM, BK), "float16")  # double-buffered
        B_shared = T.alloc_shared((2, BK, BN), "float16")
        C_frag = T.alloc_fragment((BM, BN), "float32")

        # Explicit barrier for async copy synchronization
        barrier = T.alloc_barrier(128)

        T.clear(C_frag)

        for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
            # Explicit copy with coalescing hint
            T.copy(A[by * BM, k * BK], A_shared[k % 2],
                   coalesced_width=16)
            T.copy(B[k * BK, bx * BN], B_shared[k % 2],
                   coalesced_width=16)
            T.gemm(A_shared[k % 2], B_shared[k % 2], C_frag)

        T.copy(C_frag, C[by * BM, bx * BN])
```

**Key characteristics of Level 2:**
- Uses `T.Pipelined` for software pipelining with explicit stage count
- Uses `T.alloc_barrier` for explicit synchronization
- Provides `coalesced_width` hints for memory access optimization
- May use `T.use_swizzle` for L2 cache optimization

### Level 3: Hardware-Aware with Thread Primitives (Expert)

At this level, developers use low-level PTX intrinsics, manual warp/warpgroup management, and explicit hardware features.

```python
@T.prim_func
def expert_gemm_sm90(A: T.Tensor((M, K), "float16"),
                     B: T.Tensor((K, N), "float16"),
                     C: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
        A_smem = T.alloc_shared((BM, BK), "float16")
        B_smem = T.alloc_shared((BK, BN), "float16")
        C_frag = T.alloc_fragment((BM, BN), "float32")

        # WGMMA descriptor for explicit Hopper Tensor Core usage
        desc_a = T.alloc_wgmma_desc()
        desc_b = T.alloc_wgmma_desc()

        # TMA barrier
        mbar = T.alloc_barrier(1)

        T.clear(C_frag)

        for k in T.serial(T.ceildiv(K, BK)):
            # Explicit TMA copy with user-managed synchronization
            T.mbarrier_expect_tx(mbar[0], BM * BK * 2)
            T.tma_copy(A[by * BM, k * BK], A_smem, barrier=mbar[0])
            T.mbarrier_wait_parity(mbar[0], k % 2)

            # Initialize WGMMA descriptors
            T.initialize_wgmma_descriptor(desc_a, T.address_of(A_smem[0, 0]))
            T.initialize_wgmma_descriptor(desc_b, T.address_of(B_smem[0, 0]))

            # Explicit asynchronous WGMMA without auto-wait
            T.wgmma_gemm(desc_a, desc_b, C_frag)

        # Wait for all WGMMA operations
        T.wait_wgmma(0)

        T.copy(C_frag, C[by * BM, bx * BN])
```

**Key characteristics of Level 3:**
- Uses `T.wgmma_gemm` / `T.tcgen05_gemm` for explicit asynchronous Tensor Core operations
- Uses `T.alloc_wgmma_desc` / `T.alloc_tcgen05_smem_desc` for hardware descriptors
- Uses `T.tma_copy` with manual barrier synchronization
- Uses `T.wait_wgmma` for explicit warpgroup wait
- Uses `T.mbarrier_expect_tx`, `T.mbarrier_wait_parity` for TMA synchronization
- May use `T.alloc_tmem` for Blackwell Tensor Memory

### Level Comparison Table

| Feature | Level 1 | Level 2 | Level 3 |
|---------|---------|---------|---------|
| `T.copy` | Auto-dispatch | With hints | Manual TMA/cp.async |
| `T.gemm` | Auto-dispatch | Auto with policy | Explicit WGMMA/TCGEN05 |
| Pipelining | None / Auto | `T.Pipelined` | Manual multi-stage |
| Barriers | Auto | `T.alloc_barrier` | Manual mbarrier ops |
| Layout | Auto-inferred | Annotated | Manual descriptor init |
| Synchronization | Auto | `T.sync_threads` | `T.mbarrier_wait_parity` |
| Register control | Default | `annotate_min_blocks_per_sm` | `inc_max_nreg` / `dec_max_nreg` |

---

## 4. Architecture Overview

TileLang's architecture follows a classic compiler pipeline:

```
+------------------+     +----------------+     +------------------+
|  Python DSL      | --> |  TIR (Tensor   | --> |  Transform       |
|  (@T.prim_func)  |     |  IR)           |     |  Passes          |
+------------------+     +----------------+     +------------------+
                                                         |
                                                         v
+------------------+     +----------------+     +------------------+
|  Executable      | <-- |  Target-specific| <--|  Optimized       |
|  (CUDA/HIP/Metal)|     |  Codegen       |     |  TIR             |
+------------------+     +----------------+     +------------------+
```

### 4.1 Detailed Pipeline Stages

#### Stage 1: Python DSL to TIR

The TileLang Python DSL is built on top of TVM's TIR scripting framework. When a function decorated with `@T.prim_func` is executed, the TVM IRBuilder captures all operations and constructs a TIR `PrimFunc`:

```
Python Source (@T.prim_func) --> TVM IRBuilder --> tir.PrimFunc (AST)
```

The DSL extends TVM's standard TIR with tile-level operations:
- `T.copy`, `T.async_copy`, `T.tma_copy` -- data movement intrinsics
- `T.gemm`, `T.wgmma_gemm`, `T.tcgen05_gemm` -- matrix multiply intrinsics
- `T.alloc_shared`, `T.alloc_fragment`, `T.alloc_barrier` -- memory allocation
- `T.Parallel`, `T.Pipelined`, `T.Persistent` -- loop constructs

#### Stage 2: TIR Lowering and Transformations

The raw TIR undergoes a series of transformation passes in the TileLang compiler engine (`tilelang/engine/lower.py`):

1. **Pre-Lower Semantic Check** (`PreLowerSemanticCheck`): Validates the TIR program structure.
2. **Lower and Legalize** (`LowerAndLegalize`): Lowers high-level tile operations into explicit TIR statements:
   - `T.copy` is expanded into loop nests with appropriate load/store patterns
   - `T.gemm` is lowered to target-specific MMA intrinsics
   - `T.alloc_shared` is materialized as buffer allocations
3. **Layout Inference** (`LayoutInference`): Determines optimal data layouts for all buffers based on their consumers:
   - Fragment buffers used with GEMM get MMA-compatible layouts
   - Shared memory buffers used with TMA get swizzled layouts
   - Parallel loop layouts are validated and attached
4. **Optimize for Target** (`OptimizeForTarget`): Target-specific optimizations:
   - TMA descriptor creation for eligible copies
   - Async copy (cp.async) injection for pipeline stages
   - WGMMA descriptor initialization
   - Shared memory bank conflict avoidance with swizzling

#### Stage 3: Target-Specific Codegen

The optimized TIR is compiled to target-specific code:

| Target | Codegen | Output |
|--------|---------|--------|
| NVIDIA CUDA | TVM CUDA codegen + PTX intrinsics | CUDA C++ kernel |
| AMD ROCm | TVM HIP codegen + MFMA intrinsics | HIP C++ kernel |
| Apple Metal | TVM Metal codegen | Metal Shading Language |
| CPU | TVM LLVM codegen | Native binary |
| WebGPU | TVM WebGPU codegen | WGSL shader |

The codegen stage also handles:
- Header inclusion (CUTLASS CuTe headers for CUDA, template libraries)
- Kernel wrapper generation (argument parsing, device launch)
- Link-time optimization with pre-compiled template libraries

#### Stage 4: JIT Compilation and Execution

The generated code is compiled at runtime using:

| Backend | Compiler | Notes |
|---------|----------|-------|
| `nvrtc` | NVIDIA NVRTC | Fast compilation, no external compiler needed |
| `cuda` | NVCC | Full optimization, slower compilation |
| `cutedsl` | CUTLASS CuTe DSL | Experimental, uses CuTe abstractions |
| `torch` | LibTorch JIT | PyTorch integration |
| `tvm_ffi` | TVM FFI | Direct TVM runtime execution |
| `dlpack` | DLPack | Generic tensor integration |
| `cython` | Cython | CPU-optimized execution |

---

## 5. Core Components

### 5.1 Language Module (`tilelang/language/`)

The language module provides the DSL surface syntax. It wraps TVM's TIR builder with tile-level abstractions.

**Key sub-modules:**

| File | Purpose |
|------|---------|
| `allocate.py` | Memory allocation: `alloc_shared`, `alloc_fragment`, `alloc_local`, `alloc_global`, `alloc_var`, `alloc_barrier`, `alloc_cluster_barrier`, `alloc_tmem`, `alloc_reducer`, `alloc_descriptor`, `alloc_wgmma_desc`, `alloc_tcgen05_smem_desc`, `empty` |
| `copy_op.py` | Data movement: `copy`, `async_copy`, `tma_copy`, `transpose`, `c2d_im2col` |
| `gemm_op.py` | Matrix multiply: `gemm`, `wgmma_gemm`, `tcgen05_gemm`, `tcgen05_gemm_blockscaled`, `make_blockscaled_gemm_layout` |
| `reduce_op.py` | Reductions: `reduce`, `reduce_max`, `reduce_min`, `reduce_sum`, `reduce_abssum`, `reduce_absmax`, `reduce_bitand`, `reduce_bitor`, `reduce_bitxor`, `cumsum`, `finalize_reducer`, `warp_reduce_sum/max/min` |
| `fill_op.py` | Buffer init: `fill`, `clear` |
| `kernel.py` | Kernel launch: `Kernel`, `CUDASourceCodeKernel`, `KernelLaunchFrame`, `get_thread_binding(s)`, `get_block_binding(s)` |
| `loop.py` | Loop constructs: `Parallel`, `Persistent`, `Pipelined`, `serial`, `unroll`, `vectorized` |
| `builtin.py` | Hardware intrinsics: barriers, TMA, WGMMA, warp shuffle, sync, descriptor init, TMEM ops |
| `annotations.py` | Layout/performance annotations: `use_swizzle`, `annotate_layout`, `annotate_safe_value`, `annotate_l2_hit_ratio`, `annotate_min_blocks_per_sm`, `annotate_restrict_buffers` |
| `proxy.py` | Tensor types: `Tensor`, `Buffer`, `StridedTensor`, `FragmentBuffer`, `SharedBuffer`, `LocalBuffer`, `ptr`, `make_tensor`, `make_tensor_from_addr` |
| `customize.py` | Custom ops: `dp4a`, `clamp`, `reshape`, `view`, `atomic_max/min/add/addx2/addx4`, `atomic_load/store`, `loop_break` |
| `math_intrinsics.py` | Math: `__log`, `__log2`, `__exp`, `__sin`, `__cos`, IEEE-compliant ops, packed x2 ops |
| `logical.py` | Logical: `any_of`, `all_of` |
| `symbolics.py` | Symbolic variables: `dynamic`, `symbolic` (deprecated) |
| `cluster.py` | Cluster ops (SM90+): `cluster_arrive`, `cluster_wait`, `cluster_sync`, `block_rank_in_cluster` |
| `random.py` | Random: `rng_init`, `rng_rand`, `rng_rand_float` |
| `print_op.py` | Debug: `print`, `device_assert` |
| `warpgroup.py` | Warpgroup: `ws` (warpgroup size) |
| `dtypes.py` | Data type definitions |
| `fastmath.py` | Fast math mode control |
| `pdl.py` | Pipeline Description Language: `pdl_trigger`, `pdl_sync` |

### 5.2 Engine Module (`tilelang/engine/`)

The engine orchestrates the compilation pipeline.

| File | Purpose |
|------|---------|
| `lower.py` | Main compilation entry point: `lower()`. Orchestrates semantic checks, TIR lowering, target optimization, and codegen. |
| `phase.py` | Compilation phases: `PreLowerSemanticCheck`, `LowerAndLegalize`, `OptimizeForTarget` |
| `param.py` | Kernel parameter handling and compiled artifact management |
| `callback.py` | Post-processing callbacks: `register_cuda_postproc`, `register_hip_postproc`, `register_c_postproc` |

### 5.3 JIT System (`tilelang/jit/`)

The JIT (Just-In-Time) system provides the `@tilelang.jit` decorator and kernel caching.

**Execution modes:**

1. **Lazy Mode**: The decorated function returns a `@T.prim_func` explicitly. Calling the JIT wrapper returns a compiled `JITKernel` object.

```python
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, BM, BN, BK):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), "float16"), ...):
        ...
    return kernel  # explicit PrimFunc return

kernel = matmul(1024, 1024, 1024, 128, 128, 32)
result = kernel(a, b)  # manual invocation
```

2. **Eager Mode**: The function uses the builder pattern with tensor type annotations. Calling it compiles and executes immediately.

```python
@tilelang.jit
def matmul(A, B, BM=64, BN=64, BK=64):
    M, N, K = T.const('M N K')
    A: T.Tensor[[M, K], "float16"]
    B: T.Tensor[[K, N], "float16"]
    C = T.empty([M, N], "float16")
    with T.Kernel(...):
        ...
    return C

result = matmul(a, b)  # compiles and executes immediately
```

**JIT adapter backends** (`tilelang/jit/adapter/`):

| Backend | Directory | Description |
|---------|-----------|-------------|
| NVRTC | `nvrtc/` | NVIDIA Runtime Compilation -- fastest CUDA compilation |
| CuTeDSL | `cutedsl/` | CUTLASS CuTe DSL backend |
| Cython | `cython/` | CPU-optimized via Cython |
| Torch | `torch/` | PyTorch-native execution |
| TVM FFI | `tvm_ffi.py` | Direct TVM FFI execution |

### 5.4 Backend System (`tilelang/backend/`)

Provides target-specific GEMM implementations and configuration.

| Directory | Target |
|-----------|--------|
| `cuda/` | NVIDIA CUDA with Tensor Core (WMMA, MMA, WGMMA) |
| `rocm/` | AMD ROCm with MFMA |
| `cpu/` | x86/ARM CPU via LLVM |

### 5.5 Transform Passes (`tilelang/transform/`)

Compiler transformation passes that optimize the TIR:

| Pass | Description |
|------|-------------|
| Layout Inference | Determines optimal data layouts for buffers |
| Pipeline Scheduling | Transforms `T.Pipelined` loops into prologue/body/epilogue |
| Flatten Buffer | Converts multi-dimensional buffers to flat memory |
| Decouple Type Cast | Separates type cast operations for optimization |
| Hoist Broadcast Values | Moves invariant values out of loops |
| Simplify | Algebraic simplification of TIR expressions |
| Metal Mark Host Context | Metal-specific host/device splitting |

**Pass Configuration** (`tilelang/transform/pass_config.py`):

The `PassConfigKey` enum defines configurable compiler options that can be passed via `pass_configs`:

```python
kernel = tilelang.compile(func, pass_configs={
    "tl.disable_tma": True,
    "tl.disable_async_copy": False,
})
```

### 5.6 Layout System (`tilelang/layout/`)

The layout system models data arrangement in memory.

| File | Purpose |
|------|---------|
| `layout.py` | `Layout` class: maps logical indices to physical memory positions |
| `fragment.py` | `Fragment` class: extends Layout with metadata for register file layouts |
| `swizzle.py` | Shared memory swizzling patterns for bank conflict avoidance |
| `gemm_sp.py` | Layout for 2:4 sparse GEMM |

**Layout API:**

```python
from tilelang.layout import Layout, Fragment

# Create a custom layout
layout = Layout((128,), lambda i: i // 4 * 32 + i % 4)

# Fragment for register file layout
frag = Fragment((16, 16), ...)

# Annotate a buffer's layout
T.annotate_layout({C_frag: frag})
```

### 5.7 Quantization (`tilelang/quantize/`)

| File | Purpose |
|------|---------|
| `quantization.py` | Quantization/dequantization primitives |
| `lop3.py` | LOP3 (bitwise logic) operations for quantized computation |
| `mxfp.py` | Microscaling floating-point (MXFP) format support |
| `utils.py` | Quantization utility functions |

### 5.8 Carver (`tilelang/carver/`)

The Carver system provides automatic kernel scheduling and optimization. It analyzes the computation pattern and generates optimized kernel configurations.

| Directory | Purpose |
|-----------|---------|
| `arch/` | Architecture models (CUDA, CDNA, CPU, Metal) |
| `roller/` | Configuration space exploration |
| `template/` | Kernel templates (matmul, conv, flashattention, gemv, elementwise, general_reduce) |
| `analysis.py` | Computation pattern analysis |

### 5.9 Intrinsics (`tilelang/intrinsics/`)

Hardware-specific intrinsic generators for Tensor Core operations:

| File | Purpose |
|------|---------|
| `mma_macro_generator.py` | Ampere MMA (SM80) macro generation |
| `wgmma_macro_generator.py` | Hopper WGMMA (SM90) macro generation |
| `tcgen05_macro_generator.py` | Blackwell TCGEN05 (SM100) macro generation |
| `mma_sm70_macro_generator.py` | Volta WMMA (SM70) macro generation |
| `mma_sp_macro_generator.py` | 2:4 Sparse MMA macro generation |
| `mfma_macro_generator.py` | AMD MFMA macro generation |
| `*_layout.py` | Layout definitions for each intrinsic type |

### 5.10 Tile Operations (`tilelang/tileop/`)

Low-level tile operation implementations:

| Directory | Purpose |
|-----------|---------|
| `gemm/` | GEMM implementations: `gemm_mma`, `gemm_wgmma`, `gemm_tcgen05`, `gemm_wmma`, `gemm_mfma`, `gemm_scalar` |
| `gemm_sp/` | Sparse GEMM implementations |
| `base.py` | Base classes and `GemmWarpPolicy` enum |

### 5.11 Profiler (`tilelang/profiler/`)

GPU kernel benchmarking utilities:

```python
profiler = kernel.get_profiler()
latency_ms = profiler.do_bench()                      # CUDA events
latency_ms = profiler.do_bench(backend="cupti")       # CUPTI
latency_ms = profiler.do_bench(backend="cudagraph")   # CUDA graphs
```

### 5.12 AutoTuner (`tilelang/autotuner/`)

Automatic performance tuning for kernel parameters:

```python
from tilelang import autotune

@tilelang.jit(out_idx=[-1])
@autotune(...)
def matmul(M, N, K, BM, BN, BK, ...):
    ...
```

---

## 6. Package Directory Structure

```
tilelang/
+-- __init__.py                  # Package init, version, lazy loading
+-- env.py                       # Environment configuration and flags
+-- dtypes.py                    # Data type utilities
+-- ir.py                        # IR utilities
+-- autodd.py                    # Automatic differential debugging
+-- libinfo.py                   # Library path discovery
+-- _ffi_api.py                  # FFI bridge to C++ backend
+-- _typing.py                   # Type definitions
|
+-- language/                    # DSL surface syntax
|   +-- __init__.py              # Re-exports all T.* operations
|   +-- allocate.py              # Memory allocation functions
|   +-- copy_op.py               # Data movement operations
|   +-- gemm_op.py               # GEMM operations
|   +-- reduce_op.py             # Reduction operations
|   +-- fill_op.py               # Fill/clear operations
|   +-- kernel.py                # Kernel launch management
|   +-- loop.py                  # Loop constructs
|   +-- builtin.py               # Hardware intrinsics
|   +-- annotations.py           # Layout/performance annotations
|   +-- proxy.py                 # Tensor/Buffer proxy types
|   +-- customize.py             # Custom operations
|   +-- math_intrinsics.py       # Math intrinsics
|   +-- logical.py               # Logical operations
|   +-- symbolics.py             # Symbolic variable creation
|   +-- cluster.py               # Thread cluster operations
|   +-- random.py                # Random number generation
|   +-- print_op.py              # Debug printing
|   +-- warpgroup.py             # Warpgroup utilities
|   +-- frame.py                 # Builder frame utilities
|   +-- fastmath.py              # Fast math mode
|   +-- pdl.py                   # Pipeline Description Language
|   +-- dtypes.py                # Data type definitions
|   +-- utils.py                 # Language utilities
|   +-- atomic.py                # Atomic operations
|   +-- eager/                   # Eager execution mode
|   |   +-- builder.py           # Eager builder (AST construction)
|   |   +-- ast.py               # AST utilities
|   |   +-- utils.py             # Eager mode utilities
|   +-- overrides/               # TIR overrides
|   +-- parser/                  # TIR parser extensions
|   +-- tir/                     # TIR entry points
|
+-- engine/                      # Compilation engine
|   +-- lower.py                 # Main lower/compile function
|   +-- phase.py                 # Compilation phases
|   +-- param.py                 # Kernel parameter handling
|   +-- callback.py              # Post-processing callbacks
|
+-- jit/                         # JIT compilation system
|   +-- __init__.py              # jit decorator, compile, par_compile
|   +-- kernel.py                # JITKernel class
|   +-- param.py                 # Kernel parameter types
|   +-- env.py                   # JIT environment
|   +-- exceptions.py            # JIT-specific exceptions
|   +-- execution_backend.py     # Backend selection logic
|   +-- adapter/                 # Execution backends
|       +-- nvrtc/               # NVIDIA NVRTC
|       +-- cutedsl/             # CUTLASS CuTe DSL
|       +-- cython/              # Cython
|       +-- torch/               # PyTorch
|       +-- tvm_ffi.py           # TVM FFI
|
+-- backend/                     # Target backends
|   +-- cuda/                    # NVIDIA CUDA
|   +-- rocm/                    # AMD ROCm
|   +-- cpu/                     # CPU (LLVM)
|
+-- transform/                   # Compiler transformation passes
|   +-- __init__.py              # Pass registration
|   +-- pass_config.py           # PassConfigKey definitions
|   +-- simplify.py              # Simplification pass
|   +-- add_bufstore_wrapper.py  # Buffer store wrapper
|   +-- decouple_type_cast.py    # Type cast decoupling
|   +-- hoist_broadcast_values.py# Broadcast hoisting
|   +-- metal/                   # Metal-specific passes
|
+-- layout/                      # Layout system
|   +-- layout.py                # Layout class
|   +-- fragment.py              # Fragment class
|   +-- swizzle.py               # Swizzle patterns
|   +-- gemm_sp.py               # Sparse GEMM layouts
|
+-- intrinsics/                  # Hardware intrinsic generators
|   +-- mma_macro_generator.py   # Ampere MMA
|   +-- wgmma_macro_generator.py # Hopper WGMMA
|   +-- tcgen05_macro_generator.py# Blackwell TCGEN05
|   +-- mma_sm70_macro_generator.py# Volta WMMA
|   +-- mfma_macro_generator.py  # AMD MFMA
|   +-- mma_sp_macro_generator.py# Sparse MMA
|
+-- tileop/                      # Tile operation implementations
|   +-- base.py                  # Base classes, GemmWarpPolicy
|   +-- gemm/                    # GEMM implementations
|   +-- gemm_sp/                 # Sparse GEMM
|
+-- carver/                      # Automatic scheduling
|   +-- arch/                    # Architecture models
|   +-- roller/                  # Config space exploration
|   +-- template/                # Kernel templates
|
+-- quantize/                    # Quantization support
|   +-- quantization.py          # Quant primitives
|   +-- lop3.py                  # LOP3 operations
|   +-- mxfp.py                  # MXFP formats
|
+-- cache/                       # Kernel caching
|   +-- kernel_cache.py          # Persistent kernel cache
|
+-- profiler/                    # Benchmarking
|   +-- bench.py                 # GPU kernel benchmarking
|
+-- autotuner/                   # Auto-tuning
|   +-- tuner.py                 # Tuning infrastructure
|   +-- capture.py               # Parameter capture
|   +-- param.py                 # Tuning parameters
|
+-- contrib/                     # Third-party integrations
|   +-- nvcc.py                  # NVCC compiler
|   +-- nvrtc.py                 # NVRTC runtime
|   +-- hipcc.py                 # HIP compiler
|   +-- rocm.py                  # ROCm utilities
|   +-- cutedsl/                 # CuTe DSL templates
|
+-- analysis/                    # TIR analysis
|   +-- ast_printer.py           # AST printing
|   +-- fragment_loop_checker.py # Fragment loop validation
|   +-- layout_visual.py         # Layout visualization
|   +-- nested_loop_checker.py   # Nested loop validation
|
+-- math/                        # Math operations
|   +-- __init__.py              # Math function exports
|
+-- tools/                       # Developer tools
|   +-- Analyzer.py              # TIR analyzer
|   +-- plot_layout.py           # Layout plotter
|
+-- testing/                     # Testing utilities
|   +-- perf_regression.py       # Performance regression testing
|
+-- utils/                       # General utilities
    +-- device.py                # Device management
    +-- target.py                # Target detection
    +-- tensor.py                # Tensor utilities
    +-- language.py              # Language utilities
    +-- sparse.py                # Sparse utilities
    +-- version.py               # Version utilities
    +-- deprecated.py            # Deprecation decorators
```

---

## 7. Compilation Pipeline Deep Dive

### 7.1 Entry Points

TileLang provides several ways to trigger compilation:

1. **`@tilelang.jit` decorator**: The primary interface. Wraps a Python function and compiles it on first invocation. Supports both lazy and eager modes.

2. **`tilelang.compile()`**: Explicit compilation from a `PrimFunc` to a `JITKernel`.

3. **`tilelang.par_compile()`**: Parallel compilation of multiple `PrimFunc` instances using a thread pool.

### 7.2 Compilation Flow

```python
# Step 1: Python function with @T.prim_func creates a tir.PrimFunc
@T.prim_func
def kernel(A: T.Tensor((M, K), "float16"), ...):
    with T.Kernel(...) as (bx, by):
        ...

# Step 2: tilelang.compile() or @tilelang.jit triggers the pipeline
kernel = tilelang.compile(func, out_idx=[-1], target="cuda")

# Step 2a: PreLowerSemanticCheck validates the TIR
# Step 2b: LowerAndLegalize transforms tile ops into explicit TIR
# Step 2c: LayoutInference determines buffer layouts
# Step 2d: OptimizeForTarget applies target-specific optimizations

# Step 3: Codegen produces target-specific source
cuda_source = kernel.get_kernel_source()  # CUDA C++ kernel

# Step 4: JIT compilation (nvrtc/nvcc) produces binary
# Step 5: Kernel is wrapped in a callable JITKernel
result = kernel(a_tensor, b_tensor)  # execute on GPU
```

### 7.3 Environment Configuration

TileLang behavior can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TILELANG_TARGET` | `auto` | Compilation target (`cuda`, `hip`, `cpu`, `auto`) |
| `TILELANG_EXECUTION_BACKEND` | `auto` | Execution backend (`auto`, `dlpack`, `tvm_ffi`, `cython`, `nvrtc`, `torch`, `cutedsl`) |
| `TILELANG_VERBOSE` | `false` | Enable verbose compilation output |
| `TILELANG_CACHE_DIR` | `~/.tilelang/cache` | Kernel cache directory |
| `TILELANG_SKIP_LOADING_TILELANG_SO` | `0` | Skip loading the C++ shared library |

---

## 8. Comparison with CUDA, Triton, and CUTLASS

### 8.1 Feature Comparison

| Feature | TileLang | CUDA C++ | Triton | CUTLASS |
|---------|----------|----------|--------|---------|
| **Language** | Python DSL | C++/PTX | Python DSL | C++ |
| **Programming Model** | Tile-level | Thread-level | Block-level | Tile-level (CuTe) |
| **Tensor Core Support** | Auto-dispatch | Manual PTX | Auto-dispatch | Explicit (CuTe) |
| **TMA Support** | Auto/Manual | Manual | Limited | Explicit |
| **WGMMA Support** | Auto/Manual | Manual PTX | No | Explicit |
| **Blackwell TCGEN05** | Yes | Manual PTX | No | Partial |
| **Pipeline Support** | Built-in (`T.Pipelined`) | Manual | `tl.extra_descriptor` | Manual |
| **Multi-Target** | CUDA, HIP, Metal, CPU, WebGPU | CUDA only | CUDA only | CUDA only |
| **Layout Inference** | Automatic | Manual | Automatic | Manual (CuTe) |
| **Memory Management** | Declarative | Manual | Automatic | Manual |
| **JIT Compilation** | Built-in | External (NVRTC) | Built-in | External |
| **Auto-Tuning** | Built-in | External | Built-in | External |
| **PyTorch Integration** | Native | Manual | Native | Manual |

### 8.2 Abstraction Level Comparison

```
Higher Abstraction                               Lower Abstraction
     |                                                |
     v                                                v
+-----------+   +-----------+   +-----------+   +-----------+
|  Triton   |   | TileLang  |   |  CUTLASS  |   |   CUDA    |
| (block-   |   | (tile-    |   | (tile-    |   | (thread-  |
|  level)   |   |  level)   |   |  level)   |   |  level)   |
+-----------+   +-----------+   +-----------+   +-----------+
```

- **Triton**: Operates at the thread-block level. Each program instance maps to one CUDA block. Memory layout and Tensor Core dispatch are largely automatic.
- **TileLang**: Operates at the tile level but provides explicit control over memory hierarchy, synchronization, and scheduling. Supports dropping to PTX level.
- **CUTLASS**: C++ template library using CuTe abstractions for tile-level programming. Requires deep hardware knowledge.
- **CUDA C++**: Full thread-level control. Maximum flexibility but maximum complexity.

### 8.3 Code Size Comparison (FP16 GEMM)

| Framework | Lines of Code | Performance |
|-----------|--------------|-------------|
| TileLang | ~25 lines | ~98% cuBLAS |
| Triton | ~40 lines | ~95% cuBLAS |
| CUTLASS | ~200+ lines | ~99% cuBLAS |
| CUDA C++ | ~500+ lines | ~100% cuBLAS |

---

## 9. Hardware Support

### 9.1 NVIDIA GPUs

| Architecture | Compute Capability | Tensor Core | Key Features |
|-------------|-------------------|-------------|--------------|
| Volta | SM70 (CC 7.0) | V100: WMMA | `T.gemm` dispatches to WMMA |
| Turing | SM75 (CC 7.5) | RTX 20xx: WMMA | WMMA with mixed precision |
| Ampere | SM80 (CC 8.0) | A100: MMA | `T.gemm` dispatches to MMA, cp.async |
| Ada Lovelace | SM89 (CC 8.9) | RTX 4090: MMA | Enhanced MMA |
| Hopper | SM90 (CC 9.0) | H100: WGMMA | WGMMA, TMA, Clusters, `T.Pipelined` auto-TMA |
| Blackwell | SM100 (CC 10.0) | B200: TCGEN05 | TCGEN05 MMA, TMEM, 2CTA, Block-scaled GEMM |

**Tested NVIDIA devices**: H100, A100, V100, RTX 4090, RTX 3090, RTX A6000

### 9.2 AMD GPUs

| Architecture | GPU | Key Features |
|-------------|-----|--------------|
| CDNA2 | MI250 | MFMA, Async Copy |
| CDNA3 | MI300X | Enhanced MFMA, Async DMA |
| CDNA4 | MI350/MI355X (gfx950) | LDS transpose read, enhanced MFMA |

**Key AMD operations:**
- `T.gemm` dispatches to MFMA intrinsics
- `T.async_copy` uses AMD DMA engines
- `T.ds_read_tr16_b64`, `T.ds_read_tr8_b64` for gfx950 LDS transpose

### 9.3 Apple Metal

- Supported on Apple Silicon (M1, M2, M3, M4 series)
- Uses Metal Shading Language codegen
- `T.Kernel` maps to Metal compute pipeline
- Thread group and SIMD group operations

### 9.4 CPU

- LLVM-based codegen for x86 and ARM
- Serial execution model (`is_cpu=True` in `T.Kernel`)
- Vectorized operations via LLVM auto-vectorization

### 9.5 WebGPU

- WGSL (WebGPU Shading Language) codegen
- Browser-based GPU compute
- Workgroup-level parallelism

---

## 10. Performance Benchmarks

### 10.1 GEMM Performance

TileLang achieves near-peak GEMM performance across multiple GPU architectures. Key results (FP16, 1024-8192 dimensions):

| GPU | TileLang vs cuBLAS | Notes |
|-----|-------------------|-------|
| H100 (SM90) | 95-100% | WGMMA + TMA + 3-stage pipeline |
| A100 (SM80) | 95-100% | MMA + cp.async + 2-stage pipeline |
| RTX 4090 (SM89) | 93-98% | MMA with optimized register usage |
| MI300X (CDNA3) | 93-98% | MFMA with async DMA |

### 10.2 Flash Attention Performance

TileLang's FlashAttention implementation achieves competitive performance with FlashAttention-2/3:

- MHA Forward on H100: ~95-100% of FlashAttention-2
- GQA Forward/Backward: Full support with optimized KV-cache access
- MLA Decode: Performance parity with FlashMLA on H100

### 10.3 MLA Decode Performance

TileLang's MLA (Multi-head Latent Attention) Decode kernel, written in ~80 lines of Python, achieves performance on par with FlashMLA on H100 across various batch sizes and sequence lengths.

### 10.4 Dequantize GEMM

INT4/INT8 dequantize GEMM kernels achieve high throughput on A100 and H100, competitive with specialized quantization libraries.

---

## 11. Ecosystem and Tooling

### 11.1 Debugging Tools

- **`T.print()`**: Print variables and buffers from within kernels for debugging
- **`T.device_assert()`**: Runtime assertions on GPU
- **Memory layout plotter** (`tilelang.tools.plot_layout`): Visualize buffer layouts
- **TIR Analyzer** (`tilelang.tools.Analyzer`): Analyze TIR programs
- **Debug root path** (`debug_root_path` in `@tilelang.jit`): Save generated kernel source to disk

### 11.2 Testing Utilities

- **`tilelang.testing`**: Testing framework with performance regression detection
- **`tilelang.testing.perf_regression`**: Automated performance regression testing

### 11.3 AutoTuner

The built-in auto-tuner searches over configuration spaces to find optimal kernel parameters:

```python
from tilelang import autotune

@tilelang.jit(out_idx=[-1])
@autotune(
    configs=[
        {"BM": 64, "BN": 64, "BK": 32},
        {"BM": 128, "BN": 128, "BK": 32},
        {"BM": 128, "BN": 64, "BK": 64},
    ],
    key=["M", "N", "K"],
)
def matmul(M, N, K, BM, BN, BK, ...):
    ...
```

### 11.4 Kernel Cache

TileLang caches compiled kernels to avoid recompilation:

```python
tilelang.enable_cache()   # Enable kernel caching (default)
tilelang.disable_cache()  # Disable caching
tilelang.clear_cache()    # Clear the cache
```

### 11.5 CuTeDSL Backend

Experimental backend that compiles TileLang to CUTLASS CuTe DSL, enabling integration with the CUTLASS ecosystem. Uses CuTe abstractions for layout and tensor operations.

### 11.6 Z3 Integration

TileLang integrates the Z3 theorem prover into TVM's arithmetic analyzer for:
- Symbolic shape validation
- Automatic correctness verification
- Enhanced optimization opportunities

### 11.7 TileLang Puzzles

[TileLang Puzzles](https://github.com/tile-ai/tilelang-puzzles) provides 10 progressively harder interactive puzzles for learning TileLang programming.
