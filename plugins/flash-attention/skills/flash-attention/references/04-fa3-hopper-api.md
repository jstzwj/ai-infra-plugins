# FlashAttention-3: Hopper API Reference

## Table of Contents

1. [Overview](#overview)
2. [Installation and Requirements](#installation-and-requirements)
3. [flash_attn_interface Module](#flash_attn_interface-module)
4. [flash_attn_func (FA3)](#flash_attn_func-fa3)
5. [flash_attn_qkvpacked_func (FA3)](#flash_attn_qkvpacked_func-fa3)
6. [flash_attn_varlen_func (FA3)](#flash_attn_varlen_func-fa3)
7. [flash_attn_with_kvcache (FA3)](#flash_attn_with_kvcache-fa3)
8. [flash_attn_combine](#flash_attn_combine)
9. [get_scheduler_metadata](#get_scheduler_metadata)
10. [FP16/BF16 Forward and Backward](#fp16bf16-forward-and-backward)
11. [FP8 Forward](#fp8-forward)
12. [TMA Optimizations](#tma-optimizations)
13. [SplitKV Attention](#splitkv-attention)
14. [Paged KV Cache (FA3)](#paged-kv-cache-fa3)
15. [Autograd Function Classes (FA3)](#autograd-function-classes-fa3)
16. [Comparison with FA2](#comparison-with-fa2)
17. [Benchmark Results on H100](#benchmark-results-on-h100)

---

## Overview

FlashAttention-3 (FA3) is optimized specifically for Hopper GPUs (H100/H800,
SM90). It leverages hardware features introduced in the Hopper architecture:

- **TMA (Tensor Memory Accelerator)**: Async data transfer from HBM to shared
  memory without involving the SM
- **WGMMA (Warp Group Matrix Multiply-Accumulate)**: Async matrix operations
  on the Tensor Core
- **FP8 (E4M3)**: 8-bit floating point for forward pass
- **Producer-consumer overlap**: Pipelining softmax computation with GEMM

FA3 is installed as a separate package (`flash_attn_3`) from the `hopper/`
directory.

### Package Structure

```
hopper/
|-- __init__.py                  # Package init
|-- flash_attn_interface.py      # Public Python API
|-- flash_api.cpp                # C++ binding (older torch)
|-- flash_api_stable.cpp         # Stable ABI binding (torch >= 2.9)
|-- setup.py                     # Build system
|-- flash_fwd_kernel_sm90.h      # SM90 forward kernel
|-- flash_bwd_kernel_sm90.h      # SM90 backward kernel
|-- flash_fwd_kernel_sm80.h      # SM80 fallback forward
|-- flash_bwd_kernel_sm80.h      # SM80 fallback backward
|-- mainloop_fwd_sm90_tma_gmma_ws.hpp  # TMA+WGMMA forward mainloop
|-- mainloop_bwd_sm90_tma_gmma_ws.hpp  # TMA+WGMMA backward mainloop
|-- flash_fwd_combine_kernel.h   # SplitKV combine kernel
|-- tile_scheduler.hpp           # Tile scheduling
|-- instantiations/              # Generated kernel files
|-- test_flash_attn.py           # Tests
|-- benchmark_attn.py            # Benchmarks
```

### Import

```python
# Import the FA3 interface
import flash_attn_interface

# Or import specific functions
from flash_attn_interface import (
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    flash_attn_combine,
    get_scheduler_metadata,
)
```

---

## Installation and Requirements

### Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | H100 (SM90) | H100 SXM 80GB |
| CUDA | 12.3 | 12.8 |
| PyTorch | 2.4 | 2.6+ |
| Python | 3.8 | 3.10+ |
| ninja | Required | Latest |
| packaging | Required | Latest |

### Installation

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

Or with pip:

```bash
cd flash-attention/hopper
pip install . --no-build-isolation
```

### ROCm (Triton Backend)

FA3 also supports AMD GPUs via the Triton backend:

```bash
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pip install . --no-build-isolation
```

### Verification

```bash
export PYTHONPATH=$PWD
pytest -q -s test_flash_attn.py
```

---

## flash_attn_interface Module

The `flash_attn_interface` module provides the public API for FA3. It
automatically selects between the CUDA backend and the Triton (ROCm) backend.

### Backend Selection

```python
# At module load time:
USE_TRITON_ROCM = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"

if USE_TRITON_ROCM:
    # AMD GPU path: uses aiter Triton kernels
    from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_3 as flash_attn_3_gpu
else:
    # NVIDIA GPU path: uses compiled CUDA extension
    import flash_attn_3._C
    flash_attn_3_gpu = torch.ops.flash_attn_3
```

### Internal Functions

The module registers custom PyTorch operations:

| Operation | Mutates | Description |
|-----------|---------|-------------|
| `flash_attn_3::_flash_attn_forward` | None | Unified forward (batch + varlen) |
| `flash_attn_3::_flash_attn_backward` | `dq, dk, dv` | Unified backward (batch + varlen) |

Unlike FA2, FA3 uses a single forward and backward function for both batch
and varlen modes. The mode is determined by whether `cu_seqlens_q` is provided.

---

## flash_attn_func (FA3)

The primary FA3 function for standard attention computation.

### Signature

```python
def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    sm_margin: int = 0,
    return_attn_probs: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
```

### Parameters

| Parameter | Type | Shape | Default | Description |
|-----------|------|-------|---------|-------------|
| `q` | `Tensor` | `(batch, seqlen, nheads, headdim)` | required | Query tensor |
| `k` | `Tensor` | `(batch, seqlen, nheads_k, headdim)` | required | Key tensor |
| `v` | `Tensor` | `(batch, seqlen, nheads_k, headdim_v)` | required | Value tensor |
| `softmax_scale` | `float` | scalar | `None` | Scaling. Default: `1/sqrt(headdim)` |
| `causal` | `bool` | scalar | `False` | Causal mask |
| `qv` | `Tensor` | `(batch, seqlen, nheads, headdim_v)` | `None` | Additional QV tensor (for MLA) |
| `q_descale` | `Tensor` | `(batch, nheads_k)` | `None` | FP8 Q descale factors |
| `k_descale` | `Tensor` | `(batch, nheads_k)` | `None` | FP8 K descale factors |
| `v_descale` | `Tensor` | `(batch, nheads_k)` | `None` | FP8 V descale factors |
| `window_size` | `Tuple[int, int]` | `(left, right)` | `(-1, -1)` | Sliding window |
| `attention_chunk` | `int` | scalar | `0` | Chunk size for attention (0=disabled) |
| `softcap` | `float` | scalar | `0.0` | Softcapping value |
| `num_splits` | `int` | scalar | `1` | SplitKV splits (1=disabled, 0=auto) |
| `pack_gqa` | `bool` | scalar | `None` | Pack GQA heads (None=auto) |
| `deterministic` | `bool` | scalar | `False` | Deterministic backward |
| `sm_margin` | `int` | scalar | `0` | Reserve SMs for other work |
| `return_attn_probs` | `bool` | scalar | `False` | Return LSE |

### Key Differences from FA2

1. **No `dropout_p` parameter**: FA3 does not support dropout
2. **No `alibi_slopes` parameter**: FA3 does not support ALiBi
3. **Adds `qv` parameter**: For MLA weight-absorbed attention
4. **Adds `q/k/v_descale`**: For FP8 quantization support
5. **Adds `attention_chunk`**: For chunked attention computation
6. **Adds `num_splits`**: For SplitKV parallel decoding
7. **Adds `pack_gqa`**: For GQA head packing optimization
8. **Adds `sm_margin`**: For reserving SMs for concurrent operations
9. **Supports different V head dimension** (`headdim_v` != `headdim`)

### Return Value

**Default (`return_attn_probs=False`):**
- `out`: `(batch, seqlen, nheads, headdim_v)`

**With LSE (`return_attn_probs=True`):**
- `out`: `(batch, seqlen, nheads, headdim_v)`
- `softmax_lse`: `(batch, nheads, seqlen)`

### Example

```python
import torch
import flash_attn_interface

batch, seqlen, nheads, dim = 2, 1024, 32, 128

q = torch.randn(batch, seqlen, nheads, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(batch, seqlen, nheads, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(batch, seqlen, nheads, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)

# Basic attention
out = flash_attn_interface.flash_attn_func(q, k, v)

# Causal attention
out = flash_attn_interface.flash_attn_func(q, k, v, causal=True)

# With softcapping
out = flash_attn_interface.flash_attn_func(q, k, v, softcap=50.0)

# GQA with packing
k_gqa = torch.randn(batch, seqlen, 8, dim, device='cuda', dtype=torch.bfloat16)
v_gqa = torch.randn(batch, seqlen, 8, dim, device='cuda', dtype=torch.bfloat16)
out = flash_attn_interface.flash_attn_func(q, k_gqa, v_gqa, pack_gqa=True)

# Backward pass
loss = out.sum()
loss.backward()
```

---

## flash_attn_qkvpacked_func (FA3)

FA3 variant with packed QKV tensor.

### Signature

```python
def flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    deterministic: bool = False,
    num_heads_q: Optional[int] = None,
    sm_margin: int = 0,
    return_attn_probs: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
```

### Parameters

| Parameter | Type | Shape | Default | Description |
|-----------|------|-------|---------|-------------|
| `qkv` | `Tensor` | `(batch, seqlen, 3, nheads, headdim)` or `(batch, seqlen, nheads_q + 2*nheads_k, headdim)` | required | Packed QKV |
| `num_heads_q` | `int` | scalar | `None` | Q heads count for GQA packed format |
| Others | | | | Same as `flash_attn_func` |

### Packed Format

FA3 supports two packed formats:

1. **5D tensor**: `(batch, seqlen, 3, nheads, headdim)` - Standard packed QKV
   where `qkv.unbind(dim=2)` gives Q, K, V

2. **4D tensor (GQA)**: `(batch, seqlen, nheads_q + 2*nheads_k, headdim)` -
   Q, K, V packed along the head dimension. Requires `num_heads_q`.

### Example

```python
import torch
import flash_attn_interface

# Standard 5D packed
qkv = torch.randn(2, 1024, 3, 32, 128, device='cuda', dtype=torch.bfloat16)
out = flash_attn_interface.flash_attn_qkvpacked_func(qkv, causal=True)

# 4D packed GQA (32 Q heads, 8 KV heads)
qkv_4d = torch.randn(2, 1024, 48, 128, device='cuda', dtype=torch.bfloat16)  # 32 + 8 + 8 = 48
out = flash_attn_interface.flash_attn_qkvpacked_func(qkv_4d, num_heads_q=32, causal=True)
```

---

## flash_attn_varlen_func (FA3)

Variable-length attention in FA3. Unlike FA2, FA3 uses `seqused_q` and
`seqused_k` instead of just `cu_seqlens`.

### Signature

```python
def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    sm_margin: int = 0,
    return_attn_probs: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
```

### Parameters

| Parameter | Type | Shape | Default | Description |
|-----------|------|-------|---------|-------------|
| `q` | `Tensor` | `(total_q, nheads, headdim)` | required | Concatenated query tokens |
| `k` | `Tensor` | `(total_k, nheads_k, headdim)` | required | Concatenated key tokens |
| `v` | `Tensor` | `(total_k, nheads_k, headdim_v)` | required | Concatenated value tokens |
| `cu_seqlens_q` | `Tensor` | `(batch_size + 1,)` int32 | `None` | Cumulative query seq lengths |
| `cu_seqlens_k` | `Tensor` | `(batch_size + 1,)` int32 | `None` | Cumulative key seq lengths |
| `max_seqlen_q` | `int` | scalar | required | Max query sequence length |
| `max_seqlen_k` | `int` | scalar | required | Max key sequence length |
| `seqused_q` | `Tensor` | `(batch_size,)` int32 | `None` | Actual query seq length per batch |
| `seqused_k` | `Tensor` | `(batch_size,)` int32 | `None` | Actual key seq length per batch |
| Others | | | | Same as `flash_attn_func` |

### seqused vs cu_seqlens

- `cu_seqlens`: Cumulative offsets. `cu_seqlens[0] = 0`, `cu_seqlens[i+1] = cu_seqlens[i] + seqlen_i`
- `seqused`: Actual used lengths per batch element. Can be used with padded tensors.

---

## flash_attn_with_kvcache (FA3)

FA3 KV cache function for inference. Supports more features than FA2,
including paged KV cache, FP8, and scheduler metadata.

### Signature

```python
def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    qv: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
    return_softmax_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
```

### Key Differences from FA2 KV Cache

1. **`qv` parameter**: For MLA weight-absorbed attention
2. **`cu_seqlens_q`**: Supports variable-length query sequences
3. **`cu_seqlens_k_new`**: For tracking new KV positions in varlen mode
4. **`rotary_seqlens`**: Separate sequence lengths for rotary positions
5. **`scheduler_metadata`**: Pre-computed scheduling for persistent kernels
6. **`q/k/v_descale`**: FP8 quantization descale factors
7. **`attention_chunk`**: Chunked attention computation
8. **`pack_gqa`**: GQA head packing
9. **`sm_margin`**: SM reservation
10. **Paged KV `page_block_size`**: Can be arbitrary (not restricted to 256)

### Paged KV Cache in FA3

FA3 paged KV cache supports arbitrary page sizes (not just multiples of 256):

```
k_cache: (num_blocks, page_block_size, nheads_k, headdim)
v_cache: (num_blocks, page_block_size, nheads_k, headdim_v)
```

### Example

```python
import torch
import flash_attn_interface

batch, max_seqlen, nheads, nheads_kv, dim = 4, 4096, 32, 8, 128

# Pre-allocate KV cache
k_cache = torch.zeros(batch, max_seqlen, nheads_kv, dim, device='cuda', dtype=torch.bfloat16)
v_cache = torch.zeros(batch, max_seqlen, nheads_kv, dim, device='cuda', dtype=torch.bfloat16)

# Prefill
q = torch.randn(batch, 512, nheads, dim, device='cuda', dtype=torch.bfloat16)
k = torch.randn(batch, 512, nheads_kv, dim, device='cuda', dtype=torch.bfloat16)
v = torch.randn(batch, 512, nheads_kv, dim, device='cuda', dtype=torch.bfloat16)
cache_seqlens = torch.zeros(batch, dtype=torch.int32, device='cuda')

out = flash_attn_interface.flash_attn_with_kvcache(
    q, k_cache, v_cache,
    k=k, v=v,
    cache_seqlens=cache_seqlens,
    causal=True,
)
cache_seqlens += 512

# Decode step
q_decode = torch.randn(batch, 1, nheads, dim, device='cuda', dtype=torch.bfloat16)
k_new = torch.randn(batch, 1, nheads_kv, dim, device='cuda', dtype=torch.bfloat16)
v_new = torch.randn(batch, 1, nheads_kv, dim, device='cuda', dtype=torch.bfloat16)

out = flash_attn_interface.flash_attn_with_kvcache(
    q_decode, k_cache, v_cache,
    k=k_new, v=v_new,
    cache_seqlens=cache_seqlens,
    causal=True,
    num_splits=0,  # Auto-select splits for decode
)
cache_seqlens += 1
```

---

## flash_attn_combine

Combines partial outputs from SplitKV attention.

### Signature

```python
def flash_attn_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
```

### Parameters

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `out_partial` | `Tensor` | `(num_splits, batch, seqlen, nheads, headdim)` | Partial outputs from each split |
| `lse_partial` | `Tensor` | `(num_splits, batch, seqlen, nheads)` | Partial log-sum-exp values |
| `out` | `Tensor` | `(batch, seqlen, nheads, headdim)` | Optional pre-allocated output |
| `out_dtype` | `dtype` | scalar | Output dtype override |

### Example

```python
import flash_attn_interface

# After computing partial results with num_splits > 1
out = flash_attn_interface.flash_attn_combine(out_partial, lse_partial)
```

---

## get_scheduler_metadata

Pre-computes scheduling metadata for persistent kernels on Hopper.

### Signature

```python
def get_scheduler_metadata(
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    headdim: int,
    cache_seqlens: torch.Tensor,
    qkv_dtype: torch.dtype = torch.bfloat16,
    headdim_v: Optional[int] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    max_seqlen_k_new: int = 0,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    has_softcap: bool = False,
    num_splits: int = 0,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
) -> torch.Tensor:
```

### Usage

```python
import flash_attn_interface

# Pre-compute scheduler metadata for repeated decode steps
metadata = flash_attn_interface.get_scheduler_metadata(
    batch_size=4,
    max_seqlen_q=1,
    max_seqlen_k=4096,
    num_heads_q=32,
    num_heads_kv=8,
    headdim=128,
    cache_seqlens=cache_seqlens,
)

# Pass metadata to flash_attn_with_kvcache
out = flash_attn_interface.flash_attn_with_kvcache(
    q, k_cache, v_cache,
    cache_seqlens=cache_seqlens,
    scheduler_metadata=metadata,
    causal=True,
)
```

---

## FP16/BF16 Forward and Backward

FA3 supports FP16 and BF16 for both forward and backward passes on SM90.

### Forward Pass

```python
import torch
import flash_attn_interface

q = torch.randn(2, 2048, 32, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(2, 2048, 32, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(2, 2048, 32, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)

# Forward
out = flash_attn_interface.flash_attn_func(q, k, v, causal=True)
print(out.shape)  # (2, 2048, 32, 128)

# Backward
out.sum().backward()
print(q.grad.shape, k.grad.shape, v.grad.shape)
```

### Internal Custom Op

FA3 registers `flash_attn_3::_flash_attn_forward` and
`flash_attn_3::_flash_attn_backward` as custom PyTorch operations with full
`register_fake` support for `torch.compile` and `torch.export`.

The forward op returns:
- `out`: Attention output
- `softmax_lse`: Log-sum-exp values
- `out_accum`: Partial output accumulators (SplitKV, empty if num_splits <= 1)
- `softmax_lse_accum`: Partial LSE accumulators (SplitKV, empty if num_splits <= 1)

---

## FP8 Forward

FA3 supports FP8 E4M3 forward pass on SM90, providing approximately 1.5-2x
speedup over FP16 forward.

### FP8 API

```python
import torch
import flash_attn_interface

# FP8 inputs
q = torch.randn(2, 2048, 32, 128, device='cuda', dtype=torch.float8_e4m3fn)
k = torch.randn(2, 2048, 32, 128, device='cuda', dtype=torch.float8_e4m3fn)
v = torch.randn(2, 2048, 32, 128, device='cuda', dtype=torch.float8_e4m3fn)

# Optional descale factors for quantized inputs
q_descale = torch.ones(2, 32, device='cuda', dtype=torch.float32)
k_descale = torch.ones(2, 32, device='cuda', dtype=torch.float32)
v_descale = torch.ones(2, 32, device='cuda', dtype=torch.float32)

# FP8 forward (output is BF16)
out = flash_attn_interface.flash_attn_func(
    q, k, v,
    q_descale=q_descale,
    k_descale=k_descale,
    v_descale=v_descale,
    causal=True,
)
print(out.dtype)  # torch.bfloat16
```

### FP8 Notes

- **Input**: `torch.float8_e4m3fn`
- **Output**: `torch.bfloat16` (always upcasted)
- **Descale factors**: Optional `(batch, nheads_k)` FP32 tensors for quantization scaling
- **Backward not supported**: FP8 backward is not implemented
- **No gradient tracking**: `q.requires_grad` etc. must be `False` for FP8

### FP8 Descale Behavior

When `q_descale`, `k_descale`, `v_descale` are provided:
```
S_ij = (q_descale * Q) @ (k_descale * K)^T * scale
```

The descale factors are applied per-batch-per-head.

---

## TMA Optimizations

FA3 leverages Hopper's TMA (Tensor Memory Accelerator) for efficient data
transfer:

### TMA Features Used

1. **Bulk async copy**: Transfer tiles from HBM to shared memory without SM
   involvement
2. **TMA descriptors**: Pre-computed memory layout descriptors for efficient
   addressing
3. **Multicast**: TMA can multicast data to multiple SMs in a cluster
4. **Hardware prefetching**: TMA prefetches the next tile while current tile
   is being processed

### Pipelining

FA3 implements a software pipeline that overlaps:
1. **TMA load of Q tile** (stage N+1)
2. **WGMMA computation** (stage N)
3. **Softmax reduction** (stage N)

This achieves much higher utilization of both memory bandwidth and Tensor Cores
compared to FA2.

### WGMMA (Warp Group Matrix Multiply-Accumulate)

FA3 uses Hopper's WGMMA instructions:
- Operates on shared memory operands (no register-to-register copy needed)
- 128x128x16 (FP16) or 128x128x32 (BF16) matrix sizes per warp group
- Asynchronous execution allows overlap with other operations
- Accumulates in FP32 for numerical accuracy

---

## SplitKV Attention

SplitKV splits the K/V sequence into multiple chunks, processes each chunk
independently, then combines the results. This is especially useful for
inference (decoding) where the KV cache is long but the query is short.

### Usage

```python
import flash_attn_interface

# During decoding with long KV cache
out = flash_attn_interface.flash_attn_func(
    q, k, v,
    num_splits=0,  # Auto-select number of splits
    causal=True,
)
```

### SplitKV with KV Cache

```python
out = flash_attn_interface.flash_attn_with_kvcache(
    q, k_cache, v_cache,
    cache_seqlens=cache_seqlens,
    num_splits=0,  # Auto-split for decode
    causal=True,
)
```

### How SplitKV Works

1. K/V sequence is split into `num_splits` chunks
2. Each chunk is processed by a separate thread block
3. Each thread block produces a partial output and partial log-sum-exp
4. A reduction kernel combines the partial results using log-sum-exp
   arithmetic:
   ```
   O_final = sum_i(O_i * exp(lse_i - lse_max)) / sum_i(exp(lse_i - lse_max))
   ```

### num_splits Values

| Value | Behavior |
|-------|----------|
| `0` | Auto-select based on sequence length and SM count |
| `1` | No splitting (standard attention) |
| `>1` | Split into exactly `num_splits` chunks |

---

## Paged KV Cache (FA3)

FA3 supports paged KV cache with arbitrary page sizes (unlike FA2 which
requires multiples of 256).

### Usage

```python
import flash_attn_interface

page_size = 128  # Can be any size in FA3
max_pages = 64
num_blocks = batch * max_pages

k_cache = torch.zeros(num_blocks, page_size, nheads_kv, dim, device='cuda', dtype=torch.bfloat16)
v_cache = torch.zeros(num_blocks, page_size, nheads_kv, dim, device='cuda', dtype=torch.bfloat16)
page_table = torch.zeros(batch, max_pages, dtype=torch.int32, device='cuda')

out = flash_attn_interface.flash_attn_with_kvcache(
    q, k_cache, v_cache,
    k=k_new, v=v_new,
    cache_seqlens=cache_seqlens,
    page_table=page_table,
    causal=True,
)
```

---

## Autograd Function Classes (FA3)

FA3 uses three autograd Function classes:

### FlashAttnFunc

For `flash_attn_func`. Handles:
- Forward: calls `_flash_attn_forward` with optional `qv`, descale, splits, pack_gqa
- Backward: calls `_flash_attn_backward` with same parameters
- Registers autograd via `_flash_attn_forward.register_autograd()`

### FlashAttnQKVPackedFunc

For `flash_attn_qkvpacked_func`. Handles:
- Two packed formats: 5D `(B, S, 3, H, D)` and 4D `(B, S, H_Q + 2*H_KV, D)`
- Forward: unpacks Q, K, V and calls `_flash_attn_forward`
- Backward: asserts `attention_chunk == 0` (not supported in backward)

### FlashAttnVarlenFunc

For `flash_attn_varlen_func`. Handles:
- `cu_seqlens_q`, `cu_seqlens_k` for variable-length sequences
- `seqused_q`, `seqused_k` for actual used lengths
- Saves additional tensors for backward (cu_seqlens, seqused)

---

## Comparison with FA2

### Features

| Feature | FA2 | FA3 |
|---------|-----|-----|
| FP16/BF16 Forward | Yes | Yes |
| FP16/BF16 Backward | Yes | Yes |
| FP8 Forward | No | Yes (E4M3) |
| FP8 Backward | No | No |
| Dropout | Yes | No |
| ALiBi | Yes | No |
| Softcapping | Yes | Yes |
| Sliding Window | Yes | Yes |
| MQA/GQA | Yes | Yes |
| Pack GQA | No | Yes |
| Paged KV Cache | Yes | Yes (arbitrary page size) |
| KV Cache Inference | Yes | Yes (enhanced) |
| Rotary Embedding | Yes (KV cache) | Yes (KV cache) |
| Different V head dim | No | Yes |
| Attention Chunking | No | Yes |
| SM Margin | No | Yes |
| torch.compile | Yes | Yes |
| ROCm Support | Yes (CK+Triton) | Yes (Triton) |

### Performance

FA3 is approximately 1.5-2x faster than FA2 on H100 for FP16/BF16 due to:
1. TMA async data loading
2. WGMMA async matrix multiply
3. Better pipelining and overlap
4. Reduced synchronization overhead

### API Differences

| Aspect | FA2 | FA3 |
|--------|-----|-----|
| Import | `from flash_attn import flash_attn_func` | `import flash_attn_interface; flash_attn_interface.flash_attn_func(...)` |
| Dropout | `dropout_p` parameter | Not supported |
| ALiBi | `alibi_slopes` parameter | Not supported |
| V head dim | Same as Q head dim | Can differ (`headdim_v`) |
| KV cache page size | Must be multiple of 256 | Arbitrary |
| Return with probs | Returns `(out, lse, S_dmask)` | Returns `(out, lse)` |

---

## Benchmark Results on H100

### FP16 Forward (head_dim=128, BF16)

| Sequence Length | FA3 TFLOPS | FA2 TFLOPS | Speedup |
|----------------|------------|------------|---------|
| 512 | ~480 | ~280 | 1.7x |
| 1024 | ~560 | ~330 | 1.7x |
| 2048 | ~600 | ~370 | 1.6x |
| 4096 | ~620 | ~400 | 1.55x |
| 8192 | ~630 | ~420 | 1.5x |

### FP16 Backward (head_dim=128, BF16)

| Sequence Length | FA3 TFLOPS | FA2 TFLOPS | Speedup |
|----------------|------------|------------|---------|
| 512 | ~380 | ~250 | 1.5x |
| 1024 | ~440 | ~300 | 1.47x |
| 2048 | ~480 | ~350 | 1.37x |
| 4096 | ~500 | ~380 | 1.32x |
| 8192 | ~510 | ~400 | 1.28x |

### FP8 Forward (head_dim=128, E4M3)

| Sequence Length | FA3 FP8 TFLOPS | FA3 BF16 TFLOPS | Speedup |
|----------------|----------------|-----------------|---------|
| 512 | ~700 | ~480 | 1.46x |
| 1024 | ~780 | ~560 | 1.39x |
| 2048 | ~820 | ~600 | 1.37x |
| 4096 | ~840 | ~620 | 1.35x |
| 8192 | ~850 | ~630 | 1.35x |

### FA3 vs PyTorch SDPA on H100

FA3 achieves close to the theoretical maximum on H100:
- H100 SXM FP16 theoretical: ~990 TFLOPS
- FA3 FP16 forward peak: ~630 TFLOPS (63% of theoretical for matmul + softmax)
- FA3 FP8 forward peak: ~850 TFLOPS (85% of theoretical for FP8 matmul)
