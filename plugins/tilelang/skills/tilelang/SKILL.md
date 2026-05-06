---
name: tilelang
description: >
  Comprehensive reference documentation and skill for TileLang - a concise domain-specific language (DSL)
  for developing high-performance GPU/CPU kernels built on Apache TVM. Covers the complete programming
  model (3-level abstraction), language API (memory management, compute primitives, control flow, data movement),
  compilation pipeline (TIR lowering, optimization passes, codegen), JIT system, backends (CUDA, ROCm, Metal,
  CPU, WebGPU), layout system, quantization, autotuning, carver system, debugging tools, and extensive examples
  (GEMM, FlashAttention, DeepSeek MLA/NSA, sparse/quantized operations). Based on TileLang source code with
  complete API signatures, parameter tables, and code examples for every function.
version: 0.1.0
---

# TileLang - High-Performance GPU/CPU Kernel DSL

## Overview

TileLang is a concise domain-specific language (DSL) designed for developing high-performance GPU/CPU kernels. Built on top of Apache TVM, it enables developers to write optimized kernels for AI workloads with a Pythonic syntax that abstracts away low-level hardware details while still providing fine-grained control when needed.

**Supported Hardware:**
- NVIDIA GPUs: Ampere (SM80), Hopper (SM90), Blackwell (SM100)
- AMD GPUs: MI300X, CDNA architecture
- Apple Metal
- CPU (LLVM backend)
- WebGPU

**Python Versions:** 3.8+

**License:** MIT

## Three-Level Programming Model

### Level 1: Hardware-Unaware (Beginner)
- Pure compute logic without hardware knowledge
- Automatic scheduling and optimization by the compiler
- Currently under development

### Level 2: Hardware-Aware with Tile Library (Developer)
- GPU architecture concepts (shared memory, tiling, thread blocks)
- Predefined operations and patterns
- Layout inference and pipelining handled automatically

### Level 3: Hardware-Aware with Thread Primitives (Expert)
- Full control of thread-level operations
- PTX inline assembly support
- Fine-grained performance optimization

## Quick Reference

### Installation
```bash
pip install tilelang
# Nightly builds
pip install tilelang -f https://tile-ai.github.io/whl/nightly
```

### Minimal GEMM Kernel
```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M=128, block_N=128, block_K=32):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), 'float16'),
        B: T.Tensor((K, N), 'float16'),
        C: T.Tensor((M, N), 'float16'),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), 'float16')
            B_shared = T.alloc_shared((block_K, block_N), 'float16')
            C_local = T.alloc_fragment((block_M, block_N), 'float32')
            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return kernel

# Use with PyTorch
import torch
kernel = matmul(1024, 1024, 1024)
a = torch.randn(1024, 1024).cuda().half()
b = torch.randn(1024, 1024).cuda().half()
c = kernel(a, b)  # Execute kernel
```

### Vector Add Kernel
```python
@tilelang.jit(out_idx=[-1])
def vector_add(N, BLOCK_SIZE=1024):
    @T.prim_func
    def kernel(
        A: T.Tensor((N,), 'float32'),
        B: T.Tensor((N,), 'float32'),
        C: T.Tensor((N,), 'float32'),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx):
            tx = T.get_thread_binding(0)
            idx = bx * BLOCK_SIZE + tx
            if idx < N:
                C[idx] = A[idx] + B[idx]
    return kernel
```

### Flash Attention Forward
```python
@tilelang.jit(out_idx=[-1])
def flash_attn_fwd(B, H, S, D, block_M=64, block_N=64):
    @T.prim_func
    def kernel(
        Q: T.Tensor((B, H, S, D), 'float16'),
        K: T.Tensor((B, H, S, D), 'float16'),
        V: T.Tensor((B, H, S, D), 'float16'),
        O: T.Tensor((B, H, S, D), 'float16'),
    ):
        with T.Kernel(T.ceildiv(S, block_M), threads=128) as (bx):
            # Flash attention algorithm implementation
            # ... (see examples chapter for full code)
            pass
    return kernel
```

## Core Language Constructs

### Kernel Definition
```python
T.Kernel(*blocks, threads=None, cluster_dims=None, is_cpu=False, prelude=None)
T.CUDASourceCodeKernel(*blocks, threads=None, source_code_or_path=..., entry_name="main_kernel")
```

### Memory Allocation
```python
T.alloc_shared(shape, dtype)          # On-chip shared memory
T.alloc_fragment(shape, dtype)        # Register file with layout inference
T.alloc_local(shape, dtype)           # Thread-local memory
T.alloc_global(shape, dtype)          # Global memory buffer
T.alloc_var(dtype, init)              # Scalar variable
T.alloc_barrier(arrive_count)         # Synchronization barrier
T.alloc_cluster_barrier(arrive_count) # Cluster barrier (CC 9.0+)
T.alloc_tmem(shape, dtype)            # Tensor memory (Blackwell)
T.alloc_reducer(shape, dtype, op)     # Reduction buffer
```

### Data Movement
```python
T.copy(src, dst)                      # Synchronous copy
T.async_copy(src, dst)                # Async copy
T.tma_copy(src, dst, barrier=...)     # TMA-based copy
T.transpose(src, dst)                 # Transpose
T.c2d_im2col(img, col, ...)          # Im2Col for convolution
```

### Compute Primitives
```python
T.gemm(A, B, C, policy=...)          # Matrix multiply (tensor cores)
T.wgmma_gemm(A, B, C, ...)           # Hopper WGMMA async GEMM
T.tcgen05_gemm(A, B, C, ...)         # Blackwell TCGEN05 GEMM
T.gemm_sp(A_sparse, E, B, C, ...)    # Sparse 2:4 GEMM
T.clear(buffer)                      # Zero buffer
```

### Reductions
```python
T.reduce_sum(buffer, out, dim)        # Sum reduction
T.reduce_max(buffer, out, dim)        # Max reduction
T.reduce_min(buffer, out, dim)        # Min reduction
T.cumsum(src, dst, dim)               # Cumulative sum
T.warp_reduce_sum(value)              # Warp-level sum
```

### Control Flow
```python
T.Parallel(*extents)                  # Parallel loop
T.Pipelined(extent, num_stages=N)    # Software pipelining
T.serial(start, stop)                 # Serial loop
T.unroll(start, stop)                 # Unrolled loop
T.Persistent(domain, wave_size, ...)  # Persistent kernel
T.Vectorized(start, stop)            # Vectorized loop
```

### Annotations
```python
T.use_swizzle(panel_size, order)      # Memory swizzle
T.annotate_layout(layout_map)         # Custom layout
T.annotate_safe_value(safe_value_map) # Safe value hints
T.annotate_l2_hit_ratio(ratio_map)    # L2 cache hints
T.annotate_restrict_buffers(*bufs)    # No-alias hint
```

## Memory Hierarchy

| Memory | Scope | Speed | Capacity | Allocation |
|--------|-------|-------|----------|------------|
| Global | All threads + host | Slow | Large (GBs) | T.alloc_global |
| Shared | Thread block | Fast | ~48-228 KB | T.alloc_shared |
| Local | Single thread | Medium | Limited | T.alloc_local |
| Fragment | Registers/warp | Fastest | Very limited | T.alloc_fragment |
| TMEM | Thread block cluster | Fast | Limited (Blackwell) | T.alloc_tmem |

## GEMM Warp Policies

| Policy | Description | Use Case |
|--------|-------------|----------|
| `Square` | Balanced M×N warp partition | General purpose |
| `FullRow` | All warps on M dimension | Tall-skinny matrices |
| `FullCol` | All warps on N dimension | Short-wide matrices |

## Compilation Pipeline

```
Python DSL (@T.prim_func)
    ↓
TIR (Tensor Intermediate Representation)
    ↓
PreLowerSemanticCheck → LowerAndLegalize → OptimizeForTarget
    ↓
Layout Inference → Tile Op Lowering → Pipeline Injection
    ↓
Target-specific Codegen (CUDA/HIP/Metal/LLVM)
    ↓
Compiled Kernel (JITKernel / CompiledArtifact)
```

## Execution Backends

| Backend | Description | Use Case |
|---------|-------------|----------|
| `auto` | Automatic selection | Default |
| `tvm_ffi` | TVM FFI execution | Production |
| `cython` | Cython wrapper | Production |
| `nvrtc` | NVRTC runtime compilation | Development |
| `torch` | PyTorch integration | PyTorch workflows |
| `cutedsl` | CuTeDSL backend | Expert users |
| `dlpack` | DLPack tensor interface | Framework interop |

## Key Decorators

```python
@tilelang.jit(out_idx=[-1], target="auto")       # JIT compile kernel
@tilelang.autotune(configs, key)                   # Auto-tune configurations
@T.prim_func                                       # Define primitive function
```

## Target Strings

| Target | Description |
|--------|-------------|
| `"cuda"` | NVIDIA CUDA (auto-detect arch) |
| `"hip"` / `"rocm"` | AMD ROCm/HIP |
| `"metal"` | Apple Metal |
| `"llvm"` | CPU via LLVM |
| `"webgpu"` | WebGPU |

## JITKernel Methods

```python
kernel = tilelang.jit(func)(...args)
kernel(*tensors)                    # Execute kernel
kernel.get_kernel_source()          # Get CUDA/HIP source
kernel.get_host_source()            # Get host code
kernel.show_source("both")          # Display source
kernel.export_sources("k.cu", "h.c") # Export files
kernel.get_profiler()               # Create profiler
kernel.show_ptx()                   # Show PTX assembly
kernel.show_sass()                  # Show SASS assembly
```

## Documentation Structure

### Core Fundamentals
- [01-overview-and-architecture](references/01-overview-and-architecture.md) - Architecture, design philosophy, three-level model, compilation flow, comparison with CUDA/Triton
- [02-getting-started](references/02-getting-started.md) - Installation, first kernel, GEMM walkthrough, PyTorch integration, profiling
- [03-language-basics](references/03-language-basics.md) - @T.prim_func, T.Tensor, T.Kernel, buffer indexing, arithmetic, conditionals, Python compatibility
- [04-memory-management](references/04-memory-management.md) - Memory hierarchy, alloc_shared, alloc_fragment, alloc_local, alloc_global, barriers, descriptors, scope strings
- [05-data-movement](references/05-data-movement.md) - T.copy, T.async_copy, T.tma_copy, T.transpose, T.c2d_im2col, coalescing, eviction policies, TMA descriptors

### Compute and Control
- [06-compute-primitives](references/06-compute-primitives.md) - T.gemm, T.wgmma_gemm, T.tcgen05_gemm, T.tcgen05_gemm_blockscaled, T.gemm_sp, element-wise ops, T.clear
- [07-reduction-operations](references/07-reduction-operations.md) - reduce_sum/max/min, cumsum, finalize_reducer, warp_reduce_*, batch reductions, NaN propagation
- [08-control-flow](references/08-control-flow.md) - T.Parallel, T.Pipelined, T.serial, T.unroll, T.Persistent, T.Vectorized, conditionals, pipeline stages
- [09-kernel-framework](references/09-kernel-framework.md) - T.Kernel, KernelLaunchFrame, CUDASourceCodeKernel, thread/block accessors, grid-stride loops, @tilelang.jit
- [10-type-system](references/10-type-system.md) - All data types: int4/8/16/32/64, float16/32/64, bfloat16, float8_e4m3/e5m2, float4, float6, vector types, type conversion

### System Internals
- [11-annotations-and-optimization-hints](references/11-annotations-and-optimization-hints.md) - use_swizzle, annotate_layout, annotate_safe_value, annotate_l2_hit_ratio, restrict buffers
- [12-builtins-and-intrinsics](references/12-builtins-and-intrinsics.md) - All built-in ops: barriers, warp vote/shuffle, tensor core, PTX async, TMA, LDG/STG, LDS, math intrinsics
- [13-compilation-pipeline](references/13-compilation-pipeline.md) - Lowering pipeline, phases, host/device codegen, parameter extraction, IRModule manipulation
- [14-jit-system](references/14-jit-system.md) - tilelang.compile, par_compile, JITImpl, JITKernel, execution backends, caching, PTX/SASS inspection
- [15-transform-passes](references/15-transform-passes.md) - All 40+ transform passes: LayoutInference, LowerTileOp, PipelinePlanning, ThreadSync, VectorizeLoop, etc.

### Backends and Layout
- [16-backend-cuda](references/16-backend-cuda.md) - CUDA backend: codegen, MMA/WGMMA/TCGEN05, PTX generation, Tensor Core, async copy, TMA, Hopper/Blackwell features
- [17-backend-rocm](references/17-backend-rocm.md) - ROCm/AMD backend: HIP codegen, MFMA, WMMA, LDS, CDNA, MI300X, wavefront model
- [18-backend-metal-and-others](references/18-backend-metal-and-others.md) - Metal, CPU, WebGPU backends, cross-backend portability, custom backend registration
- [19-layout-system](references/19-layout-system.md) - Layout class, swizzle layouts, bank swizzle, fragment layouts, linear layout, layout inference, visualization
- [20-quantization](references/20-quantization.md) - Quantization formats, FP4/FP8/INT4/INT8 conversions, MXFP, LOP3, weight packing, dequantization

### Advanced Topics
- [21-autotuning](references/21-autotuning.md) - @tilelang.autotune, AutoTuner, configuration spaces, BitBLAS roller, parallel compilation, performance regression
- [22-debugging-tools](references/22-debugging-tools.md) - T.print, IR inspection, post-processing callbacks, AutoDD, layout visualization, PTX/SASS, data race check
- [23-examples-gemm](references/23-examples-gemm.md) - All GEMM examples: basic, autotuned, intrinsics, persistent, FP8, dequantized, sparse, block-scaled
- [24-examples-attention](references/24-examples-attention.md) - Flash attention, GQA, block-sparse, linear attention, DeepSeek MLA/NSA, attention sink, varlen, paged
- [25-examples-advanced](references/25-examples-advanced.md) - Convolution, BitNet, element-wise, grouped GEMM, custom CUDA, multi-backend, benchmarks

### Deep Reference
- [26-carver-system](references/26-carver-system.md) - Analysis module, roller, rasterization, templates (Matmul, GEMV, FlashAttention, Conv), architecture configs
- [27-atomic-operations](references/27-atomic-operations.md) - Atomic add/max/min, global/shared memory atomics, lock-free patterns, output accumulation
- [28-eager-mode](references/28-eager-mode.md) - Eager execution, eager AST/builder, JIT mode detection, debugging with eager mode
- [29-experimental-features](references/29-experimental-features.md) - Sparse GEMM, custom intrinsics, random, PDL, cluster, warpgroup, fast math, parser
- [30-pass-configuration](references/30-pass-configuration.md) - All PassConfigKey values, pass_config recipes, architecture-specific configs, normalize_pass_configs

## Performance Benchmarks

### GEMM Performance vs cuBLAS
| Hardware | M=N=K | TileLang vs cuBLAS |
|----------|-------|-------------------|
| RTX 4090 | 8192 | ~1.1x speedup |
| A100 | 8192 | ~0.97x (on par) |
| H100 | 8192 | ~1.0x |
| MI300X | 8192 | ~1.04x |

### vs Triton
| Hardware | GEMM | TileLang vs Triton |
|----------|------|-------------------|
| RTX 4090 | 8192 | 1.08x - 1.25x speedup |
| A100 | 8192 | Competitive |

## Common Patterns

### Grid-Stride Loop
```python
with T.Kernel(T.ceildiv(N, BLOCK), threads=BLOCK) as (bx):
    for i in T.serial(bx * BLOCK, min(bx * BLOCK + BLOCK, N)):
        # process element i
```

### Double Buffering with Pipeline
```python
for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
    T.copy(A[k * BK], A_shared)
    T.gemm(A_shared, B_shared, C_local)
```

### Warp-Level Reduction
```python
val = buffer[i]
reduced = T.warp_reduce_sum(val)
```

### Shared Memory with Swizzle
```python
T.use_swizzle(panel_size=16, order="row")
A_shared = T.alloc_shared((BM, BK), dtype)
```
