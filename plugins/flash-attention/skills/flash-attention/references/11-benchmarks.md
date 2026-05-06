# FlashAttention Benchmarks Reference

This document provides comprehensive reference documentation for the benchmarking utilities and performance results in FlashAttention.

---

## Table of Contents

1. [Overview](#overview)
2. [Benchmark Utilities (utils/benchmark.py)](#benchmark-utilities)
3. [Attention Benchmarks](#attention-benchmarks)
4. [Hopper (SM90) Benchmarks](#hopper-benchmarks)
5. [CuTe DSL Benchmarks](#cute-dsl-benchmarks)
6. [Running Benchmarks](#running-benchmarks)
7. [Performance Metrics](#performance-metrics)
8. [Benchmarking Methodology](#benchmarking-methodology)

---

## Overview

FlashAttention includes comprehensive benchmarks covering:

- **Forward and backward pass timing** for attention operations
- **FLOPs measurement** for model FLOPs utilization (MFU) calculation
- **Memory benchmarks** for peak memory allocation
- **Comparison with baselines** (PyTorch SDPA, cuDNN, Triton attention)
- **Multi-GPU benchmarks** for tensor parallel scaling
- **Architecture-specific benchmarks** for Hopper (H100) and Blackwell GPUs

### Benchmark File Locations

```
flash_attn/utils/benchmark.py              # Core benchmark utilities
benchmarks/
    benchmark_flash_attention.py            # Main attention benchmarks
    benchmark_attn.py                       # Attention comparison benchmarks
    benchmark_causal.py                     # Causal attention benchmarks
    benchmark_alibi.py                      # ALiBi attention benchmarks
    benchmark_gemm.py                       # GEMM microbenchmarks
hopper/
    benchmark_attn.py                       # Hopper SM90 attention benchmarks
    benchmark_flash_attention_fp8.py        # FP8 attention benchmarks
    benchmark_mla_decode.py                 # MLA decode benchmarks
    benchmark_split_kv.py                   # Split-KV benchmarks
flash_attn/cute/
    benchmark.py                            # CuTe DSL benchmark utilities
    benchmark_flash_attention_fp8.py        # CuTe FP8 benchmarks
tests/cute/
    benchmark_block_sparsity.py             # Block sparsity benchmarks
    benchmark_mask_mod.py                   # Mask modifier benchmarks
```

---

## Benchmark Utilities

**File:** `flash_attn/utils/benchmark.py`

Core benchmarking functions using PyTorch's `torch.utils.benchmark`.

### `benchmark_forward`

```python
def benchmark_forward(
    fn, *inputs, repeats=10, desc="", verbose=True,
    amp=False, amp_dtype=torch.float16, **kwinputs
)
```

Benchmarks the forward pass of a function.

**Parameters:**
- `fn` (callable): Function to benchmark
- `*inputs`: Positional arguments to pass to `fn`
- `repeats` (int): Number of timing repetitions (default: 10)
- `desc` (str): Description for logging
- `verbose` (bool): Print results (default: True)
- `amp` (bool): Enable automatic mixed precision
- `amp_dtype` (torch.dtype): AMP dtype (default: float16)
- `**kwinputs`: Keyword arguments to pass to `fn`

**Returns:** Tuple of (timer, measurement) from `torch.utils.benchmark`

### `benchmark_backward`

```python
def benchmark_backward(
    fn, *inputs, grad=None, repeats=10, desc="", verbose=True,
    amp=False, amp_dtype=torch.float16, **kwinputs
)
```

Benchmarks the backward pass. Runs forward first, then times the backward pass.

**Parameters:**
- `grad` (torch.Tensor, optional): Custom gradient for backward. If None, uses `torch.randn_like(output)`
- All other parameters same as `benchmark_forward`

### `benchmark_combined`

```python
def benchmark_combined(
    fn, *inputs, grad=None, repeats=10, desc="", verbose=True,
    amp=False, amp_dtype=torch.float16, **kwinputs
)
```

Benchmarks forward + backward pass together as a single operation.

### `benchmark_fwd_bwd`

```python
def benchmark_fwd_bwd(fn, *inputs, **kwargs)
```

Returns tuple of (forward_result, backward_result) from separate forward and backward benchmarks.

### `benchmark_all`

```python
def benchmark_all(fn, *inputs, **kwargs)
```

Returns tuple of (forward, backward, combined) benchmark results.

### `pytorch_profiler`

```python
def pytorch_profiler(
    fn, *inputs, trace_filename=None, backward=False,
    amp=False, amp_dtype=torch.float16, cpu=False, verbose=True,
    **kwinputs
)
```

Wraps the function with PyTorch Profiler for detailed CUDA kernel analysis.

**Parameters:**
- `trace_filename` (str, optional): Export Chrome trace file for visualization
- `backward` (bool): Profile forward + backward
- `cpu` (bool): Include CPU activities

**Output:** Prints a table of CUDA kernel times sorted by self time.

### `benchmark_memory`

```python
def benchmark_memory(fn, *inputs, desc="", verbose=True, **kwinputs)
```

Measures peak GPU memory usage.

**Returns:** Peak memory in GB

---

## Attention Benchmarks

### Main Flash Attention Benchmark

**File:** `benchmarks/benchmark_flash_attention.py`

Comprehensive benchmark comparing FlashAttention against PyTorch SDPA and other implementations.

**Key benchmark parameters:**
- `--dtype`: Data type (fp16, bf16)
- `--seqlen`: Sequence lengths to benchmark
- `--headdim`: Head dimensions (64, 96, 128, 256)
- `--batch`: Batch sizes
- `--nheads`: Number of query heads
- `--nheads_kv`: Number of KV heads (for GQA)
- `--causal`: Enable causal attention
- `--backward`: Also benchmark backward pass
- `--baseline`: Compare against PyTorch SDPA

### FLOPs Calculation

**File:** `hopper/benchmark_attn.py`

```python
def flops(batch, nheads, seqlen_q, seqlen_k, headdim, headdim_v,
          causal=False, window_size=(-1, -1)):
    """Calculate attention FLOPs."""
    if causal:
        avg_seqlen = (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2
    else:
        if window_size == (-1, -1):
            avg_seqlen = seqlen_k
        else:
            # Sliding window: compute average number of attended positions
            ...
    return batch * nheads * 2 * seqlen_q * avg_seqlen * (headdim + headdim_v)
```

**Formula:**
```
FLOPs = 2 * batch * nheads * seqlen_q * avg_seqlen_k * (headdim_q + headdim_v)
```

The factor of 2 accounts for multiply-add operations.

### Timing Utilities

```python
def time_fwd(func, *args, repeats=30, verbose=True, desc="", **kwargs):
    """Time forward pass using Triton's do_bench."""
    return Timing(do_bench(lambda: func(*args, **kwargs),
                           warmup=3, rep=repeats) * 1e-3)
```

Uses `triton.testing.do_bench` for more accurate GPU timing than PyTorch's built-in benchmark.

### cuDNN Comparison

The benchmark files include cuDNN SDPA setup functions for comparison:

```python
def cudnn_spda_setup(q, k, v, causal=False, window_size_left=-1):
    """Set up cuDNN SDPA graph for benchmarking."""

def cudnn_spda_bwd_setup(q, k, v, o, g, lse, causal=False, window_size_left=-1):
    """Set up cuDNN SDPA backward graph for benchmarking."""
```

---

## Hopper Benchmarks

**Directory:** `hopper/`

Benchmarks specifically targeting NVIDIA Hopper (SM90) architecture features.

### `benchmark_attn.py`

Main Hopper attention benchmark comparing:
- FlashAttention-3 (FA3)
- FlashAttention-2 (FA2)
- cuDNN SDPA
- PyTorch SDPA (`torch.nn.functional.scaled_dot_product_attention`)

**Usage:**
```bash
cd hopper
python benchmark_attn.py \
    --dtype bf16 \
    --seqlen 2048 4096 8192 16384 \
    --headdim 128 \
    --batch 2 \
    --nheads 12 \
    --causal \
    --backward
```

**Output format:**
```
Fwd     Bsz  SeqQ  SeqK  Hdim  Heads  Causal  Time(ms)  TFLOPs  Mem(GB)
FA3     2    8192  8192  128   12     True    0.85      185.2   0.12
FA2     2    8192  8192  128   12     True    1.23      128.1   0.12
cuDNN   2    8192  8192  128   12     True    1.45      108.5   0.12
```

### `benchmark_flash_attention_fp8.py`

FP8 attention benchmarks for Hopper's FP8 hardware support.

**Features:**
- Compares FP8 E4M3 and FP8 E5M2 formats
- Measures numerical accuracy vs. fp16/bf16 baselines
- Benchmarks both forward and backward passes

### `benchmark_mla_decode.py`

Benchmarks for Multi-head Latent Attention (MLA) decode, used in DeepSeek-V2 style models.

### `benchmark_split_kv.py`

Benchmarks for split-KV attention, where the KV cache is split across multiple SMs for better utilization with long sequences.

---

## CuTe DSL Benchmarks

**Directory:** `flash_attn/cute/`

Benchmarks for the next-generation CuTe DSL (FlashAttention-4) implementation.

### `benchmark.py`

CuTe DSL benchmark utilities.

### `benchmark_flash_attention_fp8.py`

FP8 benchmarks using CuTe DSL kernels targeting SM90 and SM100.

---

## Running Benchmarks

### Basic FlashAttention Benchmark

```bash
# Forward only, fp16, sequence length 2048
python benchmarks/benchmark_flash_attention.py \
    --dtype fp16 \
    --seqlen 2048 \
    --headdim 128 \
    --batch 4 \
    --nheads 12

# Forward + backward, causal, bf16
python benchmarks/benchmark_flash_attention.py \
    --dtype bf16 \
    --seqlen 2048 4096 8192 \
    --headdim 128 \
    --batch 2 \
    --nheads 12 \
    --causal \
    --backward

# Compare with baseline
python benchmarks/benchmark_flash_attention.py \
    --dtype bf16 \
    --seqlen 2048 \
    --headdim 128 \
    --baseline
```

### Hopper Benchmarks

```bash
cd hopper

# Full benchmark suite
python benchmark_attn.py --dtype bf16 --seqlen 2048 4096 8192 --causal --backward

# FP8 benchmarks
python benchmark_flash_attention_fp8.py --seqlen 4096 8192 --headdim 128
```

### Memory Benchmarks

```python
from flash_attn.utils.benchmark import benchmark_memory
from flash_attn import flash_attn_func

q = torch.randn(2, 8192, 12, 128, device="cuda", dtype=torch.float16, requires_grad=True)
k = torch.randn(2, 8192, 12, 128, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8192, 12, 128, device="cuda", dtype=torch.float16)

mem = benchmark_memory(flash_attn_func, q, k, v, desc="FlashAttention fwd")
```

### Profiling

```python
from flash_attn.utils.benchmark import pytorch_profiler

pytorch_profiler(
    flash_attn_func, q, k, v,
    trace_filename="flash_attn_trace.json",
    backward=True,
)
```

---

## Performance Metrics

### Model FLOPs Utilization (MFU)

MFU measures how efficiently the implementation uses the GPU's theoretical compute:

```
MFU = (Measured TFLOPs) / (GPU Peak TFLOPs)
```

**GPU Peak FLOPs (tensor core):**
| GPU | FP16/BF16 | FP8 |
|-----|-----------|-----|
| A100 SXM4 80GB | 312 TFLOPs | N/A |
| H100 SXM5 80GB | 990 TFLOPs | 1979 TFLOPs |

**Typical MFU values:**
- FlashAttention-2 on A100: ~50-60% MFU
- FlashAttention-3 on H100: ~50-60% MFU
- Baseline PyTorch SDPA: ~20-35% MFU

### Attention FLOPs Formula

For standard attention:
```
FLOPs = 2 * B * H * S_q * S_k * (D_q + D_v)
```

Where:
- B = batch size
- H = number of heads
- S_q = query sequence length
- S_k = key sequence length
- D_q = query head dimension
- D_v = value head dimension

For causal attention:
```
FLOPs = 2 * B * H * S_q * avg(S_k) * (D_q + D_v)
```

Where `avg(S_k) = (max(0, S_k - S_q) + S_k) / 2`

### Memory Bandwidth

Memory bandwidth is the bottleneck for many attention operations. Key metrics:

- **A100 HBM2e bandwidth**: 2.0 TB/s
- **H100 HBM3 bandwidth**: 3.35 TB/s

FlashAttention achieves memory savings by:
1. Never materializing the full S x S attention matrix
2. Using online softmax to compute row-wise statistics
3. Tiling the computation to keep working set in SRAM

### Memory Usage

FlashAttention memory usage scales as:
```
Memory = O(B * H * S * D)  (linear in sequence length)
```

Compared to standard attention:
```
Memory = O(B * H * S^2)  (quadratic in sequence length)
```

---

## Benchmarking Methodology

### Timing Methodology

1. **Warmup**: 3-5 iterations to warm up GPU clocks and caches
2. **Measurement**: 30+ iterations with median timing
3. **Synchronization**: `torch.cuda.synchronize()` before and after each measurement
4. **Memory**: Fresh allocation for each benchmark point

### Benchmarking Best Practices

1. **Use CUDA graphs** for consistent timing (reduces kernel launch variance)
2. **Pin GPU clocks** for reproducible results (disable GPU boost)
3. **Clear cache** between runs (`torch.cuda.empty_cache()`)
4. **Report median** rather than mean (robust to outliers)
5. **Use triton.testing.do_bench** for accurate GPU timing

### Comparing Implementations

When comparing FlashAttention against baselines:

1. **Same input data**: Use identical random seeds
2. **Same dtype**: Compare fp16 vs fp16, bf16 vs bf16
3. **Same configuration**: Same head dim, causal setting, dropout
4. **End-to-end**: Measure forward + backward, not just forward
5. **Report MFU**: Raw timing is not comparable across GPU generations

### Performance Regression Testing

```bash
# Run all benchmarks and compare against baseline
python benchmarks/benchmark_flash_attention.py \
    --dtype bf16 \
    --seqlen 512 1024 2048 4096 8192 \
    --headdim 64 128 \
    --batch 2 4 \
    --causal \
    --backward \
    2>&1 | tee benchmark_results.txt
```

### Common Benchmark Parameters

| Parameter | Typical Values | Notes |
|-----------|---------------|-------|
| `batch` | 1, 2, 4, 8 | Larger batch = more throughput |
| `seqlen` | 128 - 32768 | Key axis to sweep |
| `headdim` | 64, 96, 128, 256 | 128 is most common |
| `nheads` | 12, 16, 32 | Depends on model size |
| `nheads_kv` | 1, 4, 8, nheads | For GQA/MQA |
| `dtype` | fp16, bf16 | bf16 recommended for training |
| `causal` | True/False | Causal for language modeling |
