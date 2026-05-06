# 23. GEMM Examples

This reference covers all GEMM (General Matrix Multiply) examples in TileLang with complete
source code, detailed explanations, and performance optimization patterns.

---

## Table of Contents

1. [Basic GEMM](#basic-gemm)
2. [Autotuned GEMM](#autotuned-gemm)
3. [Intrinsics GEMM (Tensor Core MMA)](#intrinsics-gemm-tensor-core-mma)
4. [Persistent GEMM](#persistent-gemm)
5. [FP8 GEMM](#fp8-gemm)
6. [Dequantized GEMM (INT4 Weight, INT8 Activation)](#dequantized-gemm-int4-weight-int8-activation)
7. [Sparse GEMM (2:4 Structured Sparsity)](#sparse-gemm-24-structured-sparsity)
8. [Grouped GEMM](#grouped-gemm)
9. [Block-Sparse GEMM](#block-sparse-gemm)
10. [Block-Scaled GEMM SM100](#block-scaled-gemm-sm100)
11. [GEMM with Different Data Types](#gemm-with-different-data-types)
12. [Performance Optimization Patterns](#performance-optimization-patterns)

---

## Basic GEMM

The basic GEMM demonstrates standard tiling with shared memory and software pipelining.
This is the foundational pattern for all GEMM implementations in TileLang.

### Complete Source Code

```python
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
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


def main():
    kernel = matmul(1024, 1024, 1024, 128, 128, 32)

    import torch
    a = torch.randn(1024, 1024).cuda().half()
    b = torch.randn(1024, 1024).cuda().half()

    c = kernel(a, b)
    ref_c = a @ b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")

    # Get CUDA Source
    print("CUDA Source:")
    print(kernel.get_kernel_source())

    # Benchmark
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti")
    print(f"tilelang Latency: {latency}ms")


if __name__ == "__main__":
    main()
```

### Key Concepts Explained

**Kernel Grid Dimensions**:
```python
T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128)
```
- Grid X: Number of tile columns (`ceildiv(N, block_N)`)
- Grid Y: Number of tile rows (`ceildiv(M, block_M)`)
- Each CUDA block computes one output tile of size `(block_M x block_N)`

**Memory Hierarchy**:
- `T.alloc_shared`: Shared memory visible to all threads in a block
- `T.alloc_fragment`: Register-backed local memory (per-thread)
- `T.copy`: Memory copy with automatic vectorization
- `T.gemm`: Matrix multiply-accumulate using tensor cores when available

**Software Pipelining**:
```python
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
```
- `num_stages=3`: Triple-buffered pipeline
- Overlaps memory loads with computation
- Stage 0: Load A[k+2], B[k+2]
- Stage 1: Compute C += A[k] * B[k]
- Stage 2: Store completed results

---

## Autotuned GEMM

This example demonstrates automatic configuration search using the AutoTuner with optional
BitBLAS Roller for device-aware recommendations.

### Complete Source Code

```python
import argparse
import itertools
import tilelang as tl
import tilelang.language as T
from tilelang.autotuner import AutoTuner
from tilelang.carver.template import MatmulTemplate
from tilelang.carver.arch import CUDA, CDNA
from tilelang.carver.roller.rasterization import NoRasterization
import torch


def ref_program(A, B):
    return A @ B.T


def get_configs(M, N, K, with_roller=False, topk=20):
    if with_roller:
        arch = CUDA("cuda") if torch.version.hip is None else CDNA("hip")
        carve_template = MatmulTemplate(
            M=M, N=N, K=K,
            in_dtype=T.float16, out_dtype=T.float16, accum_dtype=T.float32,
        ).with_arch(arch)

        func = carve_template.equivalent_function()
        assert func is not None, "Function is None"
        roller_hints = carve_template.recommend_hints(topk=topk)
        if roller_hints is None:
            raise ValueError("No Roller Hints Found for TensorCore Scheduling")
        configs = []
        for hint in roller_hints:
            config = {}
            block_m, block_n = hint.block
            warp_m, warp_n = hint.warp
            block_rows, block_cols = block_m // warp_m, block_n // warp_n
            config["block_M"] = block_m
            config["block_N"] = block_n
            config["block_K"] = hint.rstep[0]
            config["num_stages"] = hint.pipeline_stage if hint.pipeline_stage > 1 else 0
            config["thread_num"] = block_rows * block_cols * 32
            config["enable_rasteration"] = hint.rasterization_plan is not NoRasterization
            configs.append(config)
    else:
        block_M = [64, 128, 256]
        block_N = [64, 128, 256]
        block_K = [32, 64]
        num_stages = [0, 1, 2, 3]
        thread_num = [128, 256]
        enable_rasterization = [True, False]
        _configs = list(itertools.product(
            block_M, block_N, block_K, num_stages, thread_num, enable_rasterization,
        ))
        configs = [
            {
                "block_M": c[0], "block_N": c[1], "block_K": c[2],
                "num_stages": c[3], "thread_num": c[4], "enable_rasteration": c[5],
            }
            for c in _configs
        ]
    return configs


def get_heuristic_config() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.cuda.current_device()
    sm_major, sm_minor = torch.cuda.get_device_capability(device)
    sm_version = sm_major * 10 + sm_minor
    if sm_version in {80}:
        return {"block_M": 128, "block_N": 256, "block_K": 32,
                "num_stages": 2, "thread_num": 128, "enable_rasteration": True}
    elif sm_version in {90}:
        return {"block_M": 128, "block_N": 256, "block_K": 64,
                "num_stages": 3, "thread_num": 256, "enable_rasteration": True}
    else:
        return {"block_M": 128, "block_N": 256, "block_K": 32,
                "num_stages": 0, "thread_num": 128, "enable_rasteration": True}


@tl.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, num_stages, thread_num,
           enable_rasteration, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm_autotune(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                     threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)
            T.use_swizzle(panel_size=10, enable=enable_rasteration)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return gemm_autotune


def main(M=4096, N=4096, K=4096, use_autotune=False, with_roller=False,
         profile_backend="event"):
    if use_autotune:
        def kernel(block_M, block_N, block_K, num_stages, thread_num, enable_rasteration):
            @T.prim_func
            def main(A, B, C):
                # ... (same as above)
            return main

        autotuner = (
            AutoTuner.from_kernel(kernel=kernel, configs=get_configs(M, N, K, with_roller))
            .set_compile_args(out_idx=[-1], target="auto")
            .set_profile_args(
                supply_type=tl.TensorSupplyType.Integer,
                ref_prog=ref_program,
                backend=profile_backend,
            )
        )
        result = autotuner.run(warmup=3, rep=20)
        print(result.config)
        kernel = result.kernel
    else:
        config = get_heuristic_config()
        kernel = matmul(M, N, K, **config)

    profiler = kernel.get_profiler(tensor_supply_type=tl.TensorSupplyType.Auto)
    tilelang_latency = profiler.do_bench(backend=profile_backend)
    ref_latency = profiler.do_bench(ref_program, backend=profile_backend)
    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)
    print(f"TileLang latency: {tilelang_latency}")
    print(f"Ref latency: {ref_latency}")
    print(f"TileLang TFlops: {2 * M * N * K / tilelang_latency * 1e-9}")
    print(f"Ref TFlops: {2 * M * N * K / ref_latency * 1e-9}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autotuned MatMul Benchmark")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--use_autotune", action="store_true")
    parser.add_argument("--with_roller", action="store_true")
    parser.add_argument("--profile_backend", type=str, default="event")
    args = parser.parse_args()
    main(args.m, args.n, args.k, args.use_autotune, args.with_roller, args.profile_backend)
```

### Key Concepts

- **`T.use_swizzle`**: Enables L2 cache swizzle optimization for improved locality
- **`transpose_B=True`**: Indicates B is stored transposed, enabling optimized memory access
- **Heuristic fallback**: When auto-tuning is too expensive, architecture-specific defaults work well
- **Roller integration**: Reduces search space from hundreds to ~20 device-optimal configurations

---

## Intrinsics GEMM (Tensor Core MMA)

This example uses explicit Tensor Core MMA (Matrix Multiply-Accumulate) instructions via
the `TensorCoreIntrinEmitter` for fine-grained control over tensor core operations.

### Complete Source Code

```python
from tilelang import tvm as tvm
from tvm import DataType
import tilelang
import tilelang.language as T
from tilelang.intrinsics import get_swizzle_layout
from tilelang.intrinsics.mma_macro_generator import TensorCoreIntrinEmitter


def make_swizzle_layout(shared_buf):
    dtype = shared_buf.dtype
    shape = shared_buf.shape
    can_swizzle = shape[-1] * DataType(dtype).bits == 512
    if not can_swizzle:
        return T.Layout(shape, lambda *args: args)

    def transform_func(i, j):
        new_warp_i, new_warp_j = get_swizzle_layout(i, j, shape[-1], dtype)
        return [new_warp_i, new_warp_j]

    return T.Layout(shape, transform_func)


@tilelang.jit(out_idx=[2])
def tl_matmul(M, N, K, in_dtype, out_dtype, accum_dtype):
    assert in_dtype in [T.float16, T.int8]
    assert out_dtype in [T.float16, T.float32, T.int32]

    micro_size_x = micro_size_y = micro_size_k = 16
    if out_dtype == T.int32:
        micro_size_k = 32

    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 64
    warp_col_tiles = 64
    chunk = 32
    shared_scope = "shared.dyn"
    stage = 2

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    A_shape = (M, K)
    B_shape = (N, K)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)
    C_shared_shape = (
        block_M // micro_size_x,
        block_N // micro_size_y,
        micro_size_x,
        micro_size_y,
    )

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size_a = (micro_size_x * micro_size_k) // warp_size
    local_size_b = (micro_size_y * micro_size_k) // warp_size
    local_size_c = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    mma_emitter = TensorCoreIntrinEmitter(
        a_dtype=in_dtype,
        b_dtype=in_dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
    )

    @T.prim_func
    def gemm_intrinsics(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, out_dtype, scope=shared_scope)
            A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)
            B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size_c), accum_dtype)

            T.annotate_layout({
                A_shared: make_swizzle_layout(A_shared),
                B_shared: make_swizzle_layout(B_shared),
            })

            T.use_swizzle(panel_size=10)
            T.clear(C_local)

            for ko in T.Pipelined((K // block_K), num_stages=stage):
                # Load A into shared memory
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                # Load B into shared memory
                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

                for ki in T.serial(0, (block_K // micro_size_k)):
                    mma_emitter.ldmatrix_a(A_local, A_shared, ki)
                    mma_emitter.ldmatrix_b(B_local, B_shared, ki)
                    mma_emitter.mma(A_local, B_local, C_local)

            mma_emitter.stmatrix(C_local, C_shared)

            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = C_shared[
                    i // micro_size_x,
                    j // micro_size_y,
                    i % micro_size_x,
                    j % micro_size_y,
                ]

    return gemm_intrinsics


def main(M=4096, N=4096, K=4096):
    in_dtype, out_dtype, accum_dtype = T.float16, T.float16, T.float32
    kernel = tl_matmul(M, N, K, in_dtype, out_dtype, accum_dtype)
    src_code = kernel.get_kernel_source()
    assert src_code is not None

    profiler = kernel.get_profiler()
    latency = profiler.do_bench(profiler.func, warmup=25)
    assert latency is not None
    profiler.assert_allclose(lambda A, B: A @ B.T, atol=1e-2, rtol=1e-2)
```

### Key Concepts

**TensorCoreIntrinEmitter**: Provides low-level control over tensor core operations:
- `ldmatrix_a` / `ldmatrix_b`: Load data into fragment registers using `ldmatrix` instructions
- `mma`: Execute matrix multiply-accumulate using tensor cores
- `stmatrix`: Store results from fragment to shared memory

**Swizzle Layout**: Bank-conflict-free shared memory access patterns:
```python
T.annotate_layout({
    A_shared: make_swizzle_layout(A_shared),
    B_shared: make_swizzle_layout(B_shared),
})
```

**Warp Tiling**: Each warp computes a `warp_row_tiles x warp_col_tiles` output tile:
- 2 warps along rows (block_row_warps=2)
- 2 warps along columns (block_col_warps=2)
- Total: 4 warps = 128 threads

---

## Persistent GEMM

Persistent kernels keep thread blocks resident on SMs, processing multiple tiles per launch.
This reduces kernel launch overhead and improves SM utilization, especially for large matrices.

### Complete Source Code

```python
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
import argparse


@tilelang.jit(out_idx=[-1])
def matmul_non_persistent(M, N, K, block_M, block_N, block_K, threads,
                          num_stages, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            T.use_swizzle(10)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[bx * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, by * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[bx * block_M, by * block_N])

    return main


@tilelang.jit(out_idx=[-1])
def matmul_persistent(M, N, K, block_M, block_N, block_K, threads, num_stages,
                      dtype=T.float16, accum_dtype=T.float32, use_persistent_primitive=True):
    sm_num = driver.get_num_sms()
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    waves = T.ceildiv(m_blocks * n_blocks, sm_num)
    group_size = 8

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(sm_num, threads=threads) as (block_id):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            for w in T.serial(waves):
                tile_id = sm_num * w + block_id
                bx = (tile_id // group_size) % m_blocks
                by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size

                if bx * block_M < M and by * block_N < N:
                    T.clear(C_local)
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        T.copy(A[bx * block_M, k * block_K], A_shared)
                        T.copy(B[k * block_K, by * block_N], B_shared)
                        T.gemm(A_shared, B_shared, C_local)

                    T.copy(C_local, C_shared)
                    T.copy(C_shared, C[bx * block_M, by * block_N])

    @T.prim_func
    def main_persistent_primitive(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(sm_num, threads=threads) as (block_id):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            for bx, by in T.Persistent(
                [T.ceildiv(M, block_M), T.ceildiv(N, block_N)], sm_num, block_id
            ):
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[bx * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, by * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

                T.copy(C_local, C_shared)
                T.copy(C_shared, C[bx * block_M, by * block_N])

    return main_persistent_primitive if use_persistent_primitive else main


def main(M=8192, N=8192, K=8192):
    total_flops = 2 * M * N * K
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
    threads, num_stages = 256, 3

    persistent_kernel = matmul_persistent(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                                          threads, num_stages)
    persistent_profiler = persistent_kernel.get_profiler(
        tensor_supply_type=tilelang.TensorSupplyType.Randn
    )
    persistent_profiler.assert_allclose(lambda A, B: A @ B, rtol=0.01, atol=0.01)
    persistent_latency = persistent_profiler.do_bench(warmup=500)
    print(f"Persistent GEMM Latency: {persistent_latency} ms")
    print(f"Persistent GEMM TFlops: {total_flops / persistent_latency * 1e-9} TFlops")

    non_persistent_kernel = matmul_non_persistent(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                                                   threads, num_stages)
    non_persistent_profiler = non_persistent_kernel.get_profiler(
        tensor_supply_type=tilelang.TensorSupplyType.Randn
    )
    non_persistent_latency = non_persistent_profiler.do_bench(warmup=500)
    print(f"Non-Persistent GEMM Latency: {non_persistent_latency} ms")
    print(f"Speedup: {non_persistent_latency / persistent_latency}")
```

### Key Concepts

**Persistent Kernel Pattern**:
- Launch exactly `sm_num` blocks (one per SM)
- Each block processes multiple tiles across "waves"
- `T.Persistent` provides built-in tile distribution with swizzle patterns

**Wave Scheduling**:
```python
waves = T.ceildiv(m_blocks * n_blocks, sm_num)
for w in T.serial(waves):
    tile_id = sm_num * w + block_id
```

**Group-Based Swizzle**:
```python
group_size = 8
bx = (tile_id // group_size) % m_blocks
by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size
```
This reorders tile processing to improve L2 cache locality.

---

## FP8 GEMM

FP8 GEMM uses 8-bit floating-point data types (E4M3 and E5M2) for higher throughput on
Hopper and later GPUs.

### Complete Source Code

```python
import torch
import tilelang
import tilelang.language as T
from tilelang.utils import determine_fp8_type


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype, accum_dtype=T.float32):
    @T.prim_func
    def gemm_fp8(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_fp8


def test_gemm_fp8(M, N, K, dtype):
    torch_dtype = T.dtype(dtype).as_torch()
    kernel = matmul(M, N, K, 128, 128, 64, dtype)

    a = torch.randn(M, K, dtype=torch.float16, device="cuda").to(dtype=torch_dtype)
    b = torch.randn(N, K, dtype=torch.float16, device="cuda").to(dtype=torch_dtype)

    c = kernel(a, b)
    ref_c = (a.half() @ b.half().T).to(dtype=torch_dtype)

    diff = calc_diff(c, ref_c)
    print(f"diff: {diff}")
    assert diff < 1e-3


def main():
    test_gemm_fp8(1024, 1024, 1024, determine_fp8_type())       # E4M3
    test_gemm_fp8(1024, 1024, 1024, determine_fp8_type("e5m2")) # E5M2


if __name__ == "__main__":
    main()
```

### FP8 Data Types

| Format | Range | Precision | Use Case |
|--------|-------|-----------|----------|
| E4M3 (FP8) | +/-448 | 3-bit mantissa | Forward pass weights and activations |
| E5M2 (FP8) | +/-57344 | 2-bit mantissa | Backward pass gradients |

```python
from tilelang.utils import determine_fp8_type, determine_torch_fp8_type

# Auto-detect based on hardware
fp8_dtype = determine_fp8_type()          # Returns "e4m3_float8" or "float8"
torch_fp8 = determine_torch_fp8_type()     # Returns torch.float8_e4m3fn
fp8_e5m2 = determine_fp8_type("e5m2")     # Returns "e5m2_float8"
```

---

## Dequantized GEMM (INT4 Weight, INT8 Activation)

This example implements GEMM with INT4 quantized weights and INT8 activations, dequantizing
weights on-the-fly during the GEMM computation.

### Complete Source Code

```python
import tilelang
import tilelang.language as T
from tilelang.autotuner import *
from tvm import tir
import itertools
import torch
import argparse


def _tir_u8_to_i4_to_i8(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
    assert nbit == 4
    assert dtype == T.int8
    assert val.dtype == T.uint8

    mask = tir.const((1 << nbit) - 1, T.uint8)
    i4 = (val >> (pos.astype(T.uint8) * tir.const(nbit, T.uint8))) & mask
    i8_shifted = tir.reinterpret(T.int8, i4 << tir.const(4, T.uint8))
    i8 = i8_shifted >> tir.const(4, T.int8)
    return i8


def get_configs():
    iter_params = dict(
        block_M=[64, 128],
        block_N=[64, 128],
        block_K=[128, 256],
        num_stages=[1, 2],
        threads=[128, 256, 512],
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


def matmul_int8xint4(M, N, K, in_dtype, out_dtype, accum_dtype, num_bits=4, tune=False):
    @tilelang.jit(out_idx=[2])
    def kernel_func(block_M, block_N, block_K, num_stages, threads):
        num_elems_per_byte = 8 // num_bits
        storage_dtype = T.uint8
        A_shape = (M, K)
        B_shape = (N, K // num_elems_per_byte)

        @T.prim_func
        def main(
            A: T.Tensor(A_shape, in_dtype),
            B: T.Tensor(B_shape, storage_dtype),
            Ct: T.Tensor((N, M), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                         threads=threads) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), in_dtype)
                B_shared = T.alloc_shared((block_N, block_K // num_elems_per_byte), storage_dtype)
                B_local = T.alloc_fragment((block_N, block_K // num_elems_per_byte), storage_dtype)
                B_dequantize_local = T.alloc_fragment((block_N, block_K), in_dtype)
                Ct_local = T.alloc_fragment((block_N, block_M), accum_dtype)
                Ct_shared = T.alloc_shared((block_N, block_M), out_dtype)

                T.annotate_layout({
                    B_shared: tilelang.layout.make_swizzled_layout(B_shared),
                })

                T.clear(Ct_local)
                for k in T.Pipelined(K // block_K, num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
                    T.copy(B_shared, B_local)

                    # Dequantize: unpack INT4 from UINT8
                    for i, j in T.Parallel(block_N, block_K):
                        B_dequantize_local[i, j] = _tir_u8_to_i4_to_i8(
                            num_bits, B_local[i, j // num_elems_per_byte],
                            j % num_elems_per_byte, dtype=in_dtype,
                        )

                    T.gemm(B_dequantize_local, A_shared, Ct_local, transpose_B=True)

                T.copy(Ct_local, Ct_shared)
                T.copy(Ct_shared, Ct[bx * block_N:(bx+1)*block_N, by * block_M:(by+1)*block_M])

        return main

    if tune:
        @autotune(configs=get_configs(), warmup=10, rep=10)
        @tilelang.jit(out_idx=[2])
        def kernel(block_M=None, block_N=None, block_K=None,
                   num_stages=None, threads=None):
            return kernel_func(block_M, block_N, block_K, num_stages, threads).prim_func
        return kernel()
    else:
        return kernel_func
```

### Key Concepts

**Weight Dequantization**: INT4 weights packed into UINT8 are unpacked on-the-fly:
```python
def _tir_u8_to_i4_to_i8(nbit, val, pos, dtype):
    # Extract 4-bit value from packed byte
    i4 = (val >> (pos * 4)) & 0xF
    # Sign-extend from 4 bits to 8 bits
    i8_shifted = reinterpret(int8, i4 << 4)
    return i8_shifted >> 4
```

**Transposed Output**: The kernel computes `C^T = B_dequant * A^T` for efficient row-major output.

---

## Sparse GEMM (2:4 Structured Sparsity)

This example implements GEMM with NVIDIA's 2:4 structured sparsity support, which exploits
the sparsity pattern in Ampere+ tensor cores for 2x throughput.

### Complete Source Code

```python
import tilelang
import tilelang.language as T
from tilelang.layout import make_cutlass_metadata_layout
from tilelang.utils.sparse import compress, randn_semi_sparse
from tilelang.contrib import nvcc
from tilelang.profiler import do_bench
import torch

arch = nvcc.get_target_compute_version()

DEFAULT_CONFIG = {
    "h20": {
        T.float: {"block_M": 128, "block_N": 64, "block_K": 128,
                  "num_stages": 3, "thread_num": 128,
                  "policy": T.GemmWarpPolicy.Square, "enable_rasterization": True},
        T.float16: {"block_M": 128, "block_N": 64, "block_K": 128,
                    "num_stages": 3, "thread_num": 128,
                    "policy": T.GemmWarpPolicy.Square, "enable_rasterization": True},
    },
}

ARCH_INFO = {"8.0": (16, "int16"), "8.9": (16, "int16"), "9.0": (8, "uint8")}


@tilelang.jit(out_idx=[-1])
def matmul_sp_fp16(M, N, K, accum_dtype, block_M, block_N, block_K,
                   num_stages, thread_num, policy, enable_rasterization):
    e_factor, e_dtype = ARCH_INFO[arch]

    @T.prim_func
    def gemm_sp_fp16(
        A_sparse: T.Tensor((M, K // 2), T.float16),
        E: T.Tensor((M, K // e_factor), e_dtype),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                     threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K // 2), T.float16)
            E_shared = T.alloc_shared((block_M, block_K // e_factor), e_dtype)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            T.disable_warp_group_reg_alloc()
            T.use_swizzle(panel_size=10, enable=enable_rasterization)
            T.annotate_layout({
                E: make_cutlass_metadata_layout(E, mma_dtype=T.float16,
                                                 block_k=block_K, arch=arch),
                E_shared: make_cutlass_metadata_layout(E_shared, mma_dtype=T.float16,
                                                        block_k=block_K, arch=arch),
            })
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A_sparse[by * block_M, k * block_K // 2], A_shared)
                T.copy(E[by * block_M, k * block_K // e_factor], E_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm_sp(A_shared, E_shared, B_shared, C_local, False, False, policy=policy)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return gemm_sp_fp16


def main(M=1024, N=1024, K=1024, accum_dtype=T.float, cfg="h20"):
    kernel = matmul_sp_fp16(M, N, K, accum_dtype, **DEFAULT_CONFIG[cfg][accum_dtype])

    a = randn_semi_sparse(M, K, device="cuda", dtype=torch.half)
    b = torch.randn(K, N, device="cuda", dtype=torch.half)
    a_sparse, e = compress(a, transposed=False,
                           block_k=DEFAULT_CONFIG[cfg][accum_dtype]["block_K"], arch=arch)
    c = kernel(a_sparse, e, b)

    ref_c = a @ b
    torch.testing.assert_close(c, ref_c.to(c.dtype), rtol=1e-2, atol=1e-2)

    latency = do_bench(lambda: kernel(a_sparse, e, b))
    ref_latency = do_bench(lambda: a @ b)
    total_flops = 2 * M * N * K
    print(f"Sparse TFLOPS: {total_flops / latency / 1e9:.2f}")
    print(f"Reference TFLOPS: {total_flops / ref_latency / 1e9:.2f}")
```

### Key Concepts

- **`T.gemm_sp`**: Sparse GEMM using 2:4 structured sparsity
- **Metadata layout**: `make_cutlass_metadata_layout` creates proper E metadata layout
- **Sparse compression**: `compress()` converts dense matrices to sparse + metadata format
- **Architecture-dependent metadata**: Different SM versions use different element sizes

---

## Grouped GEMM

Grouped GEMM performs multiple GEMM operations with varying sizes in a single kernel launch,
useful for MoE (Mixture of Experts) inference.

### Complete Source Code

```python
import torch
import tilelang
import tilelang.language as T
import math


def torch_gmm(a, b, batch_sizes, batch_offsets_tensor, trans_b=False):
    output = torch.empty((sum(batch_sizes), b.shape[2]), device=a.device, dtype=a.dtype)
    start = 0
    for i, size in enumerate(batch_sizes):
        end = start + size
        part_a = a[start:end]
        part_b = b[i].transpose(0, 1) if trans_b else b[i]
        output[start:end] = torch.mm(part_a, part_b)
        start = end
    return output


@tilelang.jit(out_idx=[2])
def grouped_gemm(batch_sizes_list, K, N, block_M, block_N, block_K,
                 num_stages=2, threads=128, dtype=T.float16):
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    accum_dtype = T.float32
    total_m_blocks = sum(
        (size + block_M - 1) // block_M for size in batch_sizes_list
    )

    @T.prim_func
    def kernel(
        A: T.Tensor([batch_sum, K], dtype),
        B: T.Tensor([batch_count, K, N], dtype),
        C: T.Tensor([batch_sum, N], dtype),
        batch_sizes: T.Tensor([batch_count], T.int32),
        batch_offsets: T.Tensor([batch_count], T.int32),
        batch_padded_offsets: T.Tensor([batch_count], T.int32),
    ):
        with T.Kernel(total_m_blocks, T.ceildiv(N, block_N), threads=threads) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_K], dtype)
            B_shared = T.alloc_shared([block_K, block_N], dtype)
            C_local = T.alloc_fragment([block_M, block_N], accum_dtype)
            cur_batch_idx = T.alloc_var(dtype=T.int32)
            cur_batch_size = T.alloc_var(dtype=T.int32)

            m_start_padded = bx * block_M

            # Find which batch this tile belongs to
            for i in range(batch_count):
                in_cur_batch_idx = m_start_padded >= batch_padded_offsets[i]
                cur_batch_idx = T.if_then_else(in_cur_batch_idx, i, cur_batch_idx)

            cur_batch_size = batch_sizes[cur_batch_idx]
            m_start = (m_start_padded - batch_padded_offsets[cur_batch_idx]
                      + batch_offsets[cur_batch_idx])
            actual_rows = T.max(0, T.min(block_M,
                cur_batch_size + batch_padded_offsets[cur_batch_idx] - m_start_padded))

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[m_start:m_start+block_M, k*block_K:(k+1)*block_K], A_shared)
                T.copy(B[cur_batch_idx, k*block_K:(k+1)*block_K,
                         by*block_N:(by+1)*block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                if i < actual_rows:
                    C[m_start + i, by * block_N + j] = C_local[i, j]

    return kernel
```

### Key Concepts

- **Batch mapping**: Each tile maps to its batch via padded offsets
- **Boundary handling**: `actual_rows` handles non-divisible batch sizes
- **Single kernel launch**: All batches processed in one launch for efficiency

---

## Block-Sparse GEMM

Block-sparse GEMM skips computation for zero blocks, using a 3D mask tensor.

### Complete Source Code

```python
import tilelang
import tilelang.language as T
import torch


@tilelang.autotune(configs=[...])
@tilelang.jit(out_idx=[-1])
def blocksparse_matmul(M, N, K, block_M, block_N, block_K, num_stages,
                       thread_num, enable_rasteration, dtype=T.float16,
                       accum_dtype=T.float32):
    block_mask_shape = (M // block_M, N // block_N, K // block_K)

    @T.prim_func
    def block_sparse_matmul(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        BlockMask: T.Tensor(block_mask_shape, "bool"),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                     threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            T.use_swizzle(panel_size=10, enable=enable_rasteration)
            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if BlockMask[by, bx, k]:  # Only compute if block is non-zero
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return block_sparse_matmul
```

### Key Concepts

- **Block mask**: 3D boolean tensor `(M//block_M, N//block_N, K//block_K)` controls sparsity
- **Conditional computation**: `if BlockMask[by, bx, k]` skips zero blocks entirely
- **Auto-tunable**: Combined with `@tilelang.autotune` for optimal configuration search

---

## Block-Scaled GEMM SM100

Block-scaled GEMM for SM100 (Blackwell) with MXFP8 data types and per-block scale factors.

```python
@tilelang.jit
def mxfp8_blockscaled_gemm(A, B, SFA, SFB, block_M, block_N, block_K,
                           in_dtype, out_dtype, accum_dtype, num_stages,
                           sf_granularity_k=128, transpose_B=False):
    """1D-1D Block-scaled MXFP8 GEMM.

    A:   [M, K] in FP8
    B:   [K, N] or [N, K] in FP8
    SFA: [(K/sf_granularity_k)/4)*M] in uint32 -- scale factors for A
    SFB: [(K/sf_granularity_k)/4)*N] in uint32 -- scale factors for B
    """
    M, N, K = T.const("M, N, K")
    k_iters = T.ceildiv(K, block_K)
    sf_load_period = sf_granularity_k * 4 // block_K
    sf_k_groups = T.ceildiv(T.ceildiv(K, sf_granularity_k), 4)

    # ... tensor and memory declarations ...

    with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=128) as (bx, by):
        # Pipelined shared memory for data and scale factors
        A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((num_stages, block_N, block_K), in_dtype)
        SFA_shared = T.alloc_shared((num_stages, block_M), "uint32")
        SFB_shared = T.alloc_shared((num_stages, block_N), "uint32")

        # Tensor memory for accumulation
        C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
        SFA_tmem = T.alloc_tmem([block_M, block_M // 128 * 4], "uint32")
        SFB_tmem = T.alloc_tmem([block_M, block_N // 128 * 4], "uint32")

        # Barriers for synchronization
        loaded = T.alloc_barrier([32] * num_stages)
        # ... pipeline orchestration ...
```

### Key Concepts

- **`T.alloc_tmem`**: Tensor memory allocation for SM100 block-scaled operations
- **`T.alloc_barrier`**: Explicit barriers for pipeline stage synchronization
- **Scale factors**: Packed as uint32 with 4 E8M0 scale factors per element
- **1D-1D scaling**: Both A and B have independent per-block scale factors

---

## GEMM with Different Data Types

### FP16 GEMM

```python
@tilelang.jit(out_idx=[-1])
def matmul_fp16(M, N, K, block_M=128, block_N=128, block_K=32):
    dtype = T.float16
    accum_dtype = T.float32  # Always use float32 accumulation for FP16

    @T.prim_func
    def gemm(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype),
             C: T.Tensor((M, N), dtype)):
        # ... standard GEMM pattern ...
    return gemm
```

### BF16 GEMM

```python
dtype = T.bfloat16   # Brain floating point (same range as FP32, less precision)
accum_dtype = T.float32
```

### FP32 GEMM

```python
dtype = T.float32
accum_dtype = T.float32
# Note: FP32 GEMM does not use tensor cores on most GPUs
```

### INT8 GEMM

```python
dtype = T.int8
accum_dtype = T.int32  # INT8 GEMM accumulates in INT32
out_dtype = T.int32
```

### FP8 GEMM

```python
from tilelang.utils import determine_fp8_type
dtype = determine_fp8_type()  # "e4m3_float8" or "float8"
accum_dtype = T.float32
```

### Data Type Comparison

| Data Type | Bits/Element | Tensor Core | Accumulation | Use Case |
|-----------|-------------|-------------|-------------|----------|
| FP32 | 32 | No (usually) | FP32 | High-precision training |
| TF32 | 19 | Yes (Ampere+) | FP32 | Training with tensor cores |
| BF16 | 16 | Yes (Ampere+) | FP32 | Training (wider range) |
| FP16 | 16 | Yes (Volta+) | FP32 | Inference, mixed training |
| FP8 E4M3 | 8 | Yes (Hopper+) | FP32 | Inference, memory-bound |
| FP8 E5M2 | 8 | Yes (Hopper+) | FP32 | Backward pass |
| INT8 | 8 | Yes (Volta+) | INT32 | Quantized inference |
| INT4 | 4 | Custom | INT32 | Extreme quantization |

---

## Performance Optimization Patterns

### 1. Software Pipelining

```python
# No pipelining (baseline)
for k in T.serial(T.ceildiv(K, block_K)):
    T.copy(A[...], A_shared)   # Load (stalls)
    T.copy(B[...], B_shared)   # Load (stalls)
    T.gemm(A_shared, B_shared, C_local)  # Compute

# With pipelining (overlapped)
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    T.copy(A[...], A_shared)   # Overlapped with compute
    T.copy(B[...], B_shared)   # Overlapped with compute
    T.gemm(A_shared, B_shared, C_local)
```

### 2. L2 Cache Swizzle

```python
T.use_swizzle(panel_size=10, enable=True)
```

Reorders block execution to improve L2 cache hit rates for large matrices.

### 3. Warp Policy Selection

```python
# For GEMM with square output tiles
T.gemm(A, B, C, policy=T.GemmWarpPolicy.Square)

# For attention score computation (wide tiles)
T.gemm(Q, K, acc_s, policy=T.GemmWarpPolicy.FullRow)

# For attention output accumulation (tall tiles)
T.gemm(acc_s_cast, V, acc_o, policy=T.GemmWarpPolicy.FullCol)
```

### 4. Shared Memory Optimization

```python
# Swizzled layout for bank-conflict-free access
T.annotate_layout({
    A_shared: make_swizzle_layout(A_shared),
    B_shared: make_swizzle_layout(B_shared),
})

# Dynamic shared memory for larger tiles
T.alloc_shared((block_M, block_K), dtype, scope="shared.dyn")
```

### 5. Configuration Parameter Effects

| Parameter | Increase Effect | Decrease Effect |
|-----------|----------------|----------------|
| `block_M/N` | Better compute utilization, more shared memory | Less overhead per tile |
| `block_K` | Fewer pipeline iterations, more shared memory | More pipeline iterations |
| `num_stages` | Better load-compute overlap, more shared memory | Less overhead |
| `threads` | More parallelism per block | Less register pressure |
| `enable_rasterization` | Better L2 locality | Simpler scheduling |
