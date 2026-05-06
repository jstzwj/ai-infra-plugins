# 25. Advanced Examples

This reference covers advanced TileLang examples including convolution, BitNet quantized
inference, element-wise operations, DeepSeek-specific optimizations, reduction, fused
operations, custom CUDA kernel integration, multi-backend examples, benchmark patterns,
and analysis examples.

---

## Table of Contents

1. [Convolution (2D with Im2Col)](#convolution-2d-with-im2col)
2. [BitNet 1.58b (1-Bit Quantized Inference)](#bitnet-158b-1-bit-quantized-inference)
3. [Element-Wise Operations](#element-wise-operations)
4. [DeepSeek-Specific Optimizations](#deepseek-specific-optimizations)
5. [Grouped GEMM Patterns](#grouped-gemm-patterns)
6. [Reduction Examples](#reduction-examples)
7. [Fused Operations](#fused-operations)
8. [Custom CUDA Kernel Integration](#custom-cuda-kernel-integration)
9. [Multi-Backend Examples (CUDA + ROCm)](#multi-backend-examples-cuda--rocm)
10. [Benchmark Patterns and Regression Testing](#benchmark-patterns-and-regression-testing)
11. [Analysis Examples](#analysis-examples)

---

## Convolution (2D with Im2Col)

This example implements 2D convolution using the Im2Col (Image to Column) transformation,
which converts convolution into a matrix multiplication problem for efficient tiled computation.

### Complete Source Code

```python
import torch
import tilelang
import tilelang.language as T
import argparse


def check_hopper():
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    compute_capability = props.major, props.minor
    return compute_capability == (9, 0)


def ref_program(stride, padding, dilation):
    def main(A, B):
        A = A.permute(0, 3, 1, 2)  # N, H, W, C -> N, C, H, W
        B = B.permute(3, 2, 0, 1)  # H, W, C, F -> F, C, H, W
        C = torch.conv2d(A, B, stride=stride, padding=padding, dilation=dilation)
        C = C.permute(0, 2, 3, 1)  # N, C, H, W -> N, H, W, C
        return C
    return main


@tilelang.jit(out_idx=[2])
def convolution(N, C, H, W, F, K, S, D, P, block_M, block_N, block_K,
                num_stages, threads, dtype=T.float16, accum_dtype=T.float32):
    KH, KW = K, K
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    dtype = T.float16
    accum_dtype = T.float32
    is_hopper = check_hopper()

    @T.prim_func
    def main(
        data: T.Tensor((N, H, W, C), dtype),
        kernel: T.Tensor((KH, KW, C, F), dtype),
        out: T.Tensor((N, OH, OW, F), dtype),
    ):
        with T.Kernel(T.ceildiv(F, block_N), T.ceildiv(N * OH * OW, block_M),
                     threads=threads) as (bx, by):
            data_shared = T.alloc_shared((block_M, block_K), dtype)
            kernel_shared = T.alloc_shared((block_K, block_N), dtype)
            out_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            out_shared = T.alloc_shared((block_M, block_N), dtype)

            kernel_flat = T.Tensor((KH * KW * C, F), dtype, kernel.data)
            out_flat = T.Tensor((N * OH * OW, F), dtype, out.data)

            T.clear(out_local)
            for k_iter in T.Pipelined(T.ceildiv(KH * KW * C, block_K),
                                       num_stages=num_stages):
                if is_hopper:
                    # Use hardware-accelerated Im2Col on Hopper
                    T.c2d_im2col(data, data_shared, by, k_iter, KH, S, D, P)
                else:
                    # Software Im2Col transformation
                    for i, j in T.Parallel(block_M, block_K):
                        k = k_iter * block_K + j
                        m = by * block_M + i
                        access_h = m % (OH * OW) // OW * S + k // (KW * C) * D - P
                        access_w = m % OW * S + k // C % KW * D - P
                        in_bound = ((access_h >= 0) and (access_w >= 0) and
                                   (access_h < H) and (access_w < W))
                        data_shared[i, j] = T.if_then_else(
                            in_bound,
                            data[m // (OH * OW), access_h, access_w, k % C],
                            0
                        )
                T.copy(kernel_flat[k_iter * block_K, bx * block_N], kernel_shared)
                T.gemm(data_shared, kernel_shared, out_local)

            T.copy(out_local, out_shared)
            T.copy(out_shared, out_flat[by * block_M, bx * block_N])

    return main


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--c", type=int, default=128)
    parser.add_argument("--h", type=int, default=64)
    parser.add_argument("--w", type=int, default=64)
    parser.add_argument("--f", type=int, default=128)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--s", type=int, default=1)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--p", type=int, default=1)
    args = parser.parse_args(argv)

    N, C, H, W, F, K, S, D, P = args.n, args.c, args.h, args.w, args.f, args.k, args.s, args.d, args.p
    a = torch.randn(N, H, W, C).cuda().half()
    b = torch.randn(K, K, C, F).cuda().half()

    kernel = convolution(N, C, H, W, F, K, S, D, P, 64, 128, 32, 3, 256)
    out_c = kernel(a, b)
    ref_c = ref_program(S, P, D)(a, b)
    torch.testing.assert_close(out_c, ref_c, rtol=1e-2, atol=1e-2)
    print("All checks passed.")


if __name__ == "__main__":
    main()
```

### Key Concepts

**Im2Col Transformation**: Converts the convolution operation into a matrix multiplication:
- The input tensor is reshaped into a 2D matrix where each row contains the receptive field
  elements for one output position
- The filter tensor is reshaped into a 2D matrix
- Convolution is computed as `Output = Im2Col(Input) @ Filter_flat`

**Output Dimension Calculation**:
```python
OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
```
Where H=input height, P=padding, D=dilation, K=kernel size, S=stride.

**Hardware-Accelerated Im2Col**: On Hopper GPUs, `T.c2d_im2col` uses TMA (Tensor Memory
Accelerator) for efficient Im2Col transformation with hardware support.

**Boundary Handling**:
```python
in_bound = (access_h >= 0) and (access_w >= 0) and (access_h < H) and (access_w < W)
data_shared[i, j] = T.if_then_else(in_bound, data[...], 0)
```
Out-of-bounds accesses are replaced with zeros (zero-padding).

---

## BitNet 1.58b (1-Bit Quantized Inference)

BitNet uses 1.58-bit quantization (ternary weights: -1, 0, +1) for extreme model compression
with minimal quality loss.

### Architecture Overview

The BitNet example in `examples/bitnet-1.58b/` demonstrates:
- INT8 activation with INT2 (ternary) weight matrix multiplication
- Full model inference using the HuggingFace transformers API
- Performance comparison with baseline (FP16) inference
- Memory and latency benchmarking

### Key Kernel: INT8 x INT2 Decode

```python
# From: examples/bitnet-1.58b/kernel_benchmark/tilelang_bitnet_158_int8xint2_decode.py
# Implements efficient decode-phase GEMM with ternary weights
# Weight representation: 2 bits per weight element packed into INT8
# Three possible values: -1, 0, +1
```

### Weight Dequantization Pattern

```python
def _tir_unpack_ternary(val, pos):
    """Unpack 2-bit ternary weight from packed byte.
    Values: 0 -> -1, 1 -> 0, 2 -> +1
    """
    mask = tir.const(0x3, T.uint8)
    bits = (val >> (pos.astype(T.uint8) * tir.const(2, T.uint8))) & mask
    # Map: 0 -> -1, 1 -> 0, 2 -> +1
    return bits.astype(T.int8) - 1
```

### Benchmark Components

The BitNet example includes:
- `benchmark_inference_latency.py`: End-to-end inference latency measurement
- `eval_correctness.py`: Output quality verification
- `eval_gpu_memory.py`: GPU memory footprint analysis
- `eval_ppl.py`: Perplexity evaluation on standard benchmarks

---

## Element-Wise Operations

### Vector Addition

```python
import torch
import tilelang
import tilelang.language as T


def ref_program(x, y):
    return x + y


@tilelang.jit(out_idx=[-1])
def elementwise_add(M, N, block_M, block_N, in_dtype, out_dtype, threads):
    @T.prim_func
    def elem_add(
        A: T.Tensor((M, N), in_dtype),
        B: T.Tensor((M, N), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                     threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), in_dtype)
            B_shared = T.alloc_shared((block_M, block_N), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), out_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)

            for local_y, local_x in T.Parallel(block_M, block_N):
                C_local[local_y, local_x] = (
                    A_shared[local_y, local_x] + B_shared[local_y, local_x]
                )

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return elem_add


def main(M=1024, N=1024):
    a = torch.randn(M, N, dtype=torch.float32, device="cuda")
    b = torch.randn(M, N, dtype=torch.float32, device="cuda")

    kernel = elementwise_add(M, N, block_M=32, block_N=32, threads=128,
                             in_dtype=T.float32, out_dtype=T.float32)
    out = kernel(a, b)
    torch.testing.assert_close(out, ref_program(a, b), rtol=1e-2, atol=1e-2)
```

### Activation Functions

```python
@tilelang.jit(out_idx=[-1])
def relu_kernel(M, N, block_M, block_N, in_dtype, threads):
    @T.prim_func
    def relu(
        A: T.Tensor((M, N), in_dtype),
        C: T.Tensor((M, N), in_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                     threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), in_dtype)
            C_shared = T.alloc_shared((block_M, block_N), in_dtype)

            T.copy(A[by * block_M, bx * block_N], A_shared)
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(A_shared[i, j], 0.0)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return relu
```

### Sigmoid and Tanh

```python
# Sigmoid: 1 / (1 + exp(-x))
C_local[i, j] = T.sigmoid(A_shared[i, j])

# Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
C_local[i, j] = T.tanh(A_shared[i, j])
```

---

## DeepSeek-Specific Optimizations

TileLang includes several kernels optimized for DeepSeek model architectures.

### DeepSeek V3 MoE

Located in `examples/deepseek_v3/`:
- `sparse_mla_fwd.py`: Sparse Multi-head Latent Attention forward
- `sparse_mla_bwd.py`: Sparse MLA backward pass
- `topk_selector.py`: Top-K expert selection for MoE routing
- `fp8_lighting_indexer.py`: FP8 quantized Lightning attention indexer

### DeepSeek MHC (Multi-Head Connection)

Located in `examples/deepseek_mhc/`:
- `example_mhc_pre.py`: Pre-processing kernel
- `example_mhc_post.py`: Post-processing kernel
- `example_mhc_bwd.py`: Backward pass kernel

### Sparse MLA Forward

```python
# examples/deepseek_v32/sparse_mla_fwd.py
# Implements sparse MLA with top-K routing:
# 1. Compute query-key scores for expert selection
# 2. Select top-K experts via top-K selection kernel
# 3. Compute attention only for selected KV blocks
# 4. Combine results from multiple experts
```

### Warp-Specialized Flash MLA

```python
# examples/warp_specialize/example_warp_specialize_flashmla.py
# Uses warp specialization to overlap:
# - Warp group 0: Load Q and KV data
# - Warp group 1: Compute attention scores
# This improves latency for decode-phase attention where KV is long
```

---

## Grouped GEMM Patterns

Grouped GEMM handles multiple independent GEMM operations with varying M dimensions in a
single kernel launch. This is essential for MoE (Mixture of Experts) inference.

### Forward Pass Pattern

```python
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

### Backward Pass Pattern

```python
# examples/grouped_gemm/example_grouped_gemm_bwd.py
# Computes dA = dC @ B^T and dB_grouped = A^T @ dC for each group
```

### Pointer-Based Grouped GEMM

```python
# examples/grouped_gemm/example_grouped_gemm_fwd_ptr.py
# Uses pointer arrays for flexible batch specification
# Each batch can have independent A, B, and C pointers
```

---

## Reduction Examples

### Global Reduction

```python
@tilelang.jit(out_idx=[-1])
def sum_reduction(N, block_size, dtype=T.float32):
    @T.prim_func
    def kernel(
        A: T.Tensor((N,), dtype),
        O: T.Tensor((1,), dtype),
    ):
        with T.Kernel(1, threads=block_size) as (bx):
            A_shared = T.alloc_shared((block_size,), dtype)
            local_sum = T.alloc_var(dtype)

            # Load and accumulate
            local_sum = 0
            for i in T.serial(T.ceildiv(N, block_size)):
                idx = i * block_size + T.get_thread_binding()
                if idx < N:
                    local_sum += A[idx]

            A_shared[T.get_thread_binding()] = local_sum
            T.sync_threads()

            # Tree reduction
            stride = block_size // 2
            while stride > 0:
                if T.get_thread_binding() < stride:
                    A_shared[T.get_thread_binding()] += A_shared[T.get_thread_binding() + stride]
                T.sync_threads()
                stride //= 2

            if T.get_thread_binding() == 0:
                O[0] = A_shared[0]

    return kernel
```

### Row/Column Reduction

```python
# reduce_max along dimension 1
T.reduce_max(acc_s, scores_max, dim=1, clear=False)

# reduce_sum along dimension 1
T.reduce_sum(acc_s, scores_sum, dim=1)
```

### RMS Norm (Fused Reduction + Scale)

```python
# examples/norm/rms_norm.py
@tilelang.jit(out_idx=[-1])
def rms_norm(M, N, block_M, block_N, dtype=T.float16, eps=1e-6):
    @T.prim_func
    def kernel(
        X: T.Tensor((M, N), dtype),
        W: T.Tensor((N,), dtype),
        O: T.Tensor((M, N), dtype),
    ):
        # ... load X tile, compute RMS, normalize, scale by W ...
    return kernel
```

---

## Fused Operations

### Fused MoE (Mixture of Experts)

```python
# examples/fusedmoe/example_fusedmoe_tilelang.py
# Fuses top-K gating + expert computation:
# 1. Compute gate scores for all experts
# 2. Select top-K experts
# 3. Perform grouped GEMM for selected experts
# 4. Combine expert outputs with gate weights
```

### Fused Cast to FP8

```python
# examples/cast/example_per_token_cast_to_fp8.py
# Fuses per-token scaling + cast to FP8:
# 1. Compute per-token max absolute value
# 2. Scale to FP8 representable range
# 3. Cast to FP8 with proper rounding
```

### Fused Quantize-GEMM

```python
# examples/dequantize_gemm/example_dequant_gemm_w4a8.py
# Fuses weight dequantization with GEMM:
# 1. Load packed INT4 weights
# 2. Dequantize on-the-fly in shared memory
# 3. Immediately use in GEMM computation
# Avoids round-trip to global memory for dequantized weights
```

### Online Softmax

```python
# examples/online_softmax/online_softmax.py
# Implements the online softmax algorithm:
# Maintains running max and sum without materializing full attention matrix
```

---

## Custom CUDA Kernel Integration

TileLang supports integration with custom CUDA kernels through the `CUDASourceCodeKernel`
mechanism.

### Registering Post-Processing Callbacks

```python
from tilelang.engine import register_cuda_postproc_callback

@register_cuda_postproc_callback
def inject_custom_cuda(code: str, target) -> str:
    """Modify generated CUDA code before compilation."""
    # Add custom includes
    code = '#include "my_custom_kernels.cuh"\n' + code

    # Inject custom kernel calls
    code = code.replace(
        "// kernel end",
        "// Custom post-processing\n"
        "my_custom_postprocess<<<grid, block>>>(output, N);\n"
        "// kernel end"
    )
    return code
```

### Using External CUDA Code

```python
# Integrate custom CUDA code with TileLang's compilation pipeline
@register_cuda_postproc_callback
def add_custom_kernels(code: str, target) -> str:
    custom_code = """
    __global__ void my_custom_kernel(float* data, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            data[idx] = data[idx] * 2.0f;
        }
    }
    """
    return custom_code + "\n" + code
```

---

## Multi-Backend Examples (CUDA + ROCm)

TileLang supports both NVIDIA CUDA and AMD ROCm backends. The same kernel code can target
either platform.

### Target Selection

```python
import torch

# Auto-detect target
if torch.version.hip is None:
    target = "cuda"
    arch = CUDA("cuda")
else:
    target = "hip"
    arch = CDNA("hip")

kernel = tilelang.jit(out_idx=[-1], target=target)(my_func)(M, N, K, ...)
```

### AMD-Specific Examples

Located in `examples/amd/`:
- `example_amd_flash_attn_fwd.py`: Flash attention forward for AMD GPUs
- `example_amd_flash_attn_bwd.py`: Flash attention backward for AMD GPUs

### AMD MFMA Instructions

```python
# TileLang automatically maps T.gemm() to appropriate instructions:
# NVIDIA: MMA (Matrix Multiply-Accumulate) or WGMMA (Warp Group MMA)
# AMD: MFMA (Matrix Fused Multiply-Add)
# The backend is selected automatically based on the target
```

### Backend-Specific Tensor Supply

```python
def supply_tensors_gpu(params):
    """Supply function that creates tensors on GPU for ROCm/HIP."""
    tensors = []
    for param in params:
        if hasattr(param, "shape") and hasattr(param, "dtype"):
            shape = [int(s) for s in param.shape]
            torch_dtype = param.dtype.as_torch()
            tensor = torch.randn(shape, dtype=torch_dtype, device="cuda")
            tensors.append(tensor)
        else:
            tensors.append(param)
    return tensors
```

### Architecture-Specific Roller

```python
from tilelang.carver.arch import CUDA, CDNA

if torch.version.hip is None:
    arch = CUDA("cuda")
else:
    arch = CDNA("hip")

template = MatmulTemplate(M=M, N=N, K=K, ...).with_arch(arch)
hints = template.recommend_hints(topk=20)
```

---

## Benchmark Patterns and Regression Testing

TileLang includes infrastructure for consistent benchmarking and performance regression testing.

### Standard Benchmark Pattern

```python
def main(M=4096, N=4096, K=4096):
    kernel = my_jit_func(M, N, K, block_M=128, block_N=256, block_K=64)
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)

    # Correctness check
    profiler.assert_allclose(ref_program, rtol=1e-2, atol=1e-2)
    print("All checks passed.")

    # Performance measurement
    latency = profiler.do_bench(warmup=500)
    total_flops = 2 * M * N * K
    tflops = total_flops / latency * 1e-9

    print(f"Latency: {latency:.3f} ms")
    print(f"TFlops: {tflops:.2f}")

    # Compare with reference
    ref_latency = profiler.do_bench(ref_program, warmup=500)
    ref_tflops = total_flops / ref_latency * 1e-9
    print(f"Reference TFlops: {ref_tflops:.2f}")
    print(f"Speedup: {ref_latency / latency:.2f}x")
```

### Regression Test Pattern

Each example includes a `run_regression_perf()` function for CI testing:

```python
def run_regression_perf(M=4096, N=4096, K=4096):
    """Standard regression test function used by CI.
    Returns latency in milliseconds for comparison with baseline.
    """
    config = get_heuristic_config()
    kernel = matmul(M, N, K, **config)
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
    return profiler.do_bench(backend="cupti")  # Use cupti for consistent results
```

### Profiler Backends

```python
# CUDA event timing (default)
latency = profiler.do_bench()                    # backend="event"

# CUPTI profiling (most accurate)
latency = profiler.do_bench(backend="cupti")

# CUDA graph timing (for graph-captured workloads)
latency = profiler.do_bench(backend="cudagraph")

# Custom warmup/repeat
latency = profiler.do_bench(warmup=500, rep=1000)
```

### Custom Input Tensors

```python
# Supply specific input tensors for benchmarking
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
latency = profiler.do_bench(input_tensors=[a, b])
```

### TileLang's `do_bench` Utility

```python
from tilelang.profiler import do_bench

# Benchmark any callable
latency = do_bench(lambda: kernel(a, b))
latency = do_bench(lambda: torch.mm(a, b), backend="cupti")
latency = do_bench(lambda: my_function(), warmup=100, rep=500)
```

---

## Analysis Examples

TileLang provides analysis tools for understanding kernel behavior and performance.

### GEMM Analysis

```python
# examples/analyze/example_gemm_analyze.py
# Analyzes GEMM kernel performance characteristics:
# 1. Compute intensity (FLOPs/byte)
# 2. Memory bandwidth utilization
# 3. Tensor core utilization
# 4. Occupancy analysis
```

### Convolution Analysis

```python
# examples/analyze/example_conv_analyze.py
# Analyzes convolution kernel characteristics:
# 1. Im2Col matrix dimensions
# 2. Arithmetic intensity
# 3. Shared memory usage
# 4. Register pressure
```

### Layout Visualization

```python
# examples/visual_layout_inference/visual_layout_inference.py
# Visualizes how TileLang maps logical tensor indices to physical locations:
# - Thread mapping: which thread computes which output element
# - Register mapping: where values are stored in registers
# - Shared memory layout: bank conflict analysis

kernel = tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True,
        tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS: "png",
    },
)(my_func)(...)
# Generates color-coded layout plots
```

### Fragment Layout Plotting

```python
# examples/plot_layout/fragment_mma_load_a.py
# Plots the fragment layout for MMA load A operation
# Shows how matrix elements are distributed across threads and registers

# examples/plot_layout/layout_swizzle.py
# Visualizes swizzled shared memory layouts for bank-conflict-free access

# examples/plot_layout/layout_transform.py
# Shows how TileLang transforms layouts between different memory scopes
```

### Hadamard Transform

```python
# examples/hadamard_transform/example_hadamard.py
# Implements the Hadamard transform (Walsh-Hadamard transform)
# Used in certain attention mechanisms and signal processing
```

### Top-K Selection

```python
# examples/topk/example_topk.py
# Implements efficient top-K selection for MoE routing
# Uses bitonic sort or heap-based approaches
```

### Random Number Generation

```python
# examples/rand/rand_uint.py
# Implements efficient random unsigned integer generation on GPU
```

---

## Summary of Optimization Techniques

### Memory Hierarchy Optimization

| Technique | Pattern | Benefit |
|-----------|---------|---------|
| Tiling | Divide work into shared memory tiles | Data reuse, reduced global memory access |
| Software Pipelining | `T.Pipelined(num_stages=N)` | Overlap memory and compute |
| L2 Swizzle | `T.use_swizzle(panel_size=10)` | Better L2 cache locality |
| Shared Memory Swizzle | `T.annotate_layout({...})` | Bank conflict elimination |

### Compute Optimization

| Technique | Pattern | Benefit |
|-----------|---------|---------|
| Tensor Core MMA | `T.gemm()` (automatic) | 4-16x throughput vs scalar |
| Warp Group MMA | `T.disable_warp_group_reg_alloc()` | Full tensor core utilization |
| Persistent Kernels | `T.Persistent([...], sm_num, block_id)` | Reduced launch overhead |
| Warp Specialization | Separate warp groups for load/compute | Better pipeline utilization |

### Numerical Optimization

| Technique | Pattern | Benefit |
|-----------|---------|---------|
| FP32 Accumulation | `accum_dtype=T.float32` | Avoid precision loss |
| Log-Domain Softmax | `T.exp2(x * scale - max * scale)` | Numerical stability |
| Online Softmax | Running max/sum update | Memory efficiency |
| FP8 Quantization | `determine_fp8_type()` | 2x throughput on Hopper+ |
