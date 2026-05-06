# FlashAttention-2: Complete API Reference

## Table of Contents

1. [Overview](#overview)
2. [flash_attn_func](#flash_attn_func)
3. [flash_attn_qkvpacked_func](#flash_attn_qkvpacked_func)
4. [flash_attn_kvpacked_func](#flash_attn_kvpacked_func)
5. [flash_attn_varlen_func](#flash_attn_varlen_func)
6. [flash_attn_varlen_qkvpacked_func](#flash_attn_varlen_qkvpacked_func)
7. [flash_attn_varlen_kvpacked_func](#flash_attn_varlen_kvpacked_func)
8. [flash_attn_with_kvcache](#flash_attn_with_kvcache)
9. [flash_blocksparse_attn_func](#flash_blocksparse_attn_func)
10. [Autograd Function Classes](#autograd-function-classes)
11. [Custom Op Registration](#custom-op-registration)
12. [Block Size Selection Heuristic](#block-size-selection-heuristic)
13. [Causal Mask Behavior](#causal-mask-behavior)
14. [ALiBi Support](#alibi-support)
15. [Softcapping](#softcapping)
16. [Sliding Window Attention](#sliding-window-attention)
17. [MQA/GQA Support](#mqagqa-support)
18. [Paged KV Cache](#paged-kv-cache)
19. [Rotary Embedding](#rotary-embedding)
20. [Error Handling](#error-handling)
21. [Common Patterns and Examples](#common-patterns-and-examples)

---

## Overview

FlashAttention-2 provides the following public API functions, all imported
from the `flash_attn` package:

```python
from flash_attn import (
    flash_attn_func,               # Q, K, V as separate tensors
    flash_attn_qkvpacked_func,     # QKV packed into one tensor
    flash_attn_kvpacked_func,      # KV packed into one tensor
    flash_attn_varlen_func,        # Variable-length sequences, separate Q/K/V
    flash_attn_varlen_qkvpacked_func,  # Variable-length, packed QKV
    flash_attn_varlen_kvpacked_func,   # Variable-length, packed KV
    flash_attn_with_kvcache,       # Inference with KV cache
)
```

All functions support:
- FP16 and BF16 data types
- Head dimensions from 16 to 256 (padded to multiples of 8)
- Causal masking
- Sliding window (local) attention
- ALiBi (attention with linear bias)
- Softcapping
- Dropout
- MQA/GQA (multi-query / grouped-query attention)
- Automatic gradient computation via `torch.autograd`

### Common Conventions

- All tensor inputs must be on CUDA devices
- Last dimension (`headdim`) must be contiguous in memory
- `softmax_scale` defaults to `1 / sqrt(headdim)` when `None`
- `dropout_p` should be `0.0` during evaluation
- `causal=True` aligns the mask to the bottom-right corner (v2.1+ behavior)
- Head dimension is automatically padded to the next multiple of 8

---

## flash_attn_func

The most general function, taking Q, K, V as separate tensors.

### Signature

```python
def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
```

### Parameters

| Parameter | Type | Shape | Default | Description |
|-----------|------|-------|---------|-------------|
| `q` | `Tensor` | `(batch, seqlen_q, nheads, headdim)` | required | Query tensor |
| `k` | `Tensor` | `(batch, seqlen_k, nheads_k, headdim)` | required | Key tensor |
| `v` | `Tensor` | `(batch, seqlen_k, nheads_k, headdim)` | required | Value tensor |
| `dropout_p` | `float` | scalar | `0.0` | Dropout probability. Set to 0.0 during evaluation. |
| `softmax_scale` | `float` | scalar | `None` | Scaling of QK^T before softmax. Default: `1/sqrt(headdim)`. |
| `causal` | `bool` | scalar | `False` | Whether to apply causal attention mask. |
| `window_size` | `Tuple[int, int]` | `(left, right)` | `(-1, -1)` | Sliding window. `-1` means infinite. |
| `softcap` | `float` | scalar | `0.0` | Softcap value. `>0` enables softcapping: `tanh(S/softcap)*softcap`. |
| `alibi_slopes` | `Tensor` | `(nheads,)` or `(batch, nheads)` | `None` | ALiBi bias slopes, FP32. |
| `deterministic` | `bool` | scalar | `False` | Use deterministic backward. Slightly slower, more memory. |
| `return_attn_probs` | `bool` | scalar | `False` | Return attention probabilities (testing only). |

### Return Value

**Default (`return_attn_probs=False`):**
- `out`: `(batch, seqlen_q, nheads, headdim)` - Output tensor

**With attention probs (`return_attn_probs=True`):**
- `out`: `(batch, seqlen_q, nheads, headdim)` - Output tensor
- `softmax_lse`: `(batch, nheads, seqlen_q)` - Log-sum-exp of each row
- `S_dmask`: `(batch, nheads, seqlen_q, seqlen_k)` - Softmax output (possibly with different scaling). Encodes dropout pattern (negative = dropped, nonnegative = kept). **For testing only; not guaranteed correct scaling.**

### MQA/GQA Support

Supports multi-query attention (MQA) and grouped-query attention (GQA) by
passing K, V with fewer heads than Q. The number of heads in Q must be
divisible by the number of heads in KV.

Example: If Q has 6 heads and K, V have 2 heads:
- Q heads 0, 1, 2 attend to K/V head 0
- Q heads 3, 4, 5 attend to K/V head 1

### Example

```python
import torch
from flash_attn import flash_attn_func

batch, seqlen, heads, dim = 4, 1024, 32, 128
q = torch.randn(batch, seqlen, heads, dim, device='cuda', dtype=torch.float16, requires_grad=True)
k = torch.randn(batch, seqlen, heads, dim, device='cuda', dtype=torch.float16, requires_grad=True)
v = torch.randn(batch, seqlen, heads, dim, device='cuda', dtype=torch.float16, requires_grad=True)

# Basic attention
out = flash_attn_func(q, k, v)

# Causal attention
out = flash_attn_func(q, k, v, causal=True)

# With dropout (training only)
out = flash_attn_func(q, k, v, dropout_p=0.1)

# Sliding window attention (look back 512 tokens)
out = flash_attn_func(q, k, v, window_size=(512, 0))

# With ALiBi
alibi_slopes = torch.randn(heads, device='cuda', dtype=torch.float32)
out = flash_attn_func(q, k, v, alibi_slopes=alibi_slopes, causal=True)

# With softcapping (e.g., Gemma-2 uses softcap=50.0)
out = flash_attn_func(q, k, v, softcap=50.0)

# GQA: 32 query heads, 8 KV heads
k_gqa = torch.randn(batch, seqlen, 8, dim, device='cuda', dtype=torch.float16)
v_gqa = torch.randn(batch, seqlen, 8, dim, device='cuda', dtype=torch.float16)
out = flash_attn_func(q, k_gqa, v_gqa)

# Backward pass (automatic via autograd)
loss = out.sum()
loss.backward()
print(q.grad.shape, k.grad.shape, v.grad.shape)
```

---

## flash_attn_qkvpacked_func

Optimized function when Q, K, V are already stacked into a single tensor.
Faster than `flash_attn_func` because the backward pass avoids explicit
concatenation of gradients.

### Signature

```python
def flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
```

### Parameters

| Parameter | Type | Shape | Default | Description |
|-----------|------|-------|---------|-------------|
| `qkv` | `Tensor` | `(batch, seqlen, 3, nheads, headdim)` | required | Packed QKV tensor |
| `dropout_p` | `float` | scalar | `0.0` | Dropout probability |
| `softmax_scale` | `float` | scalar | `None` | Scaling factor. Default: `1/sqrt(headdim)`. |
| `causal` | `bool` | scalar | `False` | Causal mask |
| `window_size` | `Tuple[int, int]` | `(left, right)` | `(-1, -1)` | Sliding window |
| `softcap` | `float` | scalar | `0.0` | Softcap value |
| `alibi_slopes` | `Tensor` | `(nheads,)` or `(batch, nheads)` | `None` | ALiBi slopes, FP32 |
| `deterministic` | `bool` | scalar | `False` | Deterministic backward |
| `return_attn_probs` | `bool` | scalar | `False` | Return attention probs (testing) |

### Return Value

Same as `flash_attn_func`.

### Notes

- For MQA/GQA, use `flash_attn_kvpacked_func` or `flash_attn_func` instead.
- The packed format is `qkv[:, :, 0]` = Q, `qkv[:, :, 1]` = K, `qkv[:, :, 2]` = V.

### Example

```python
import torch
from flash_attn import flash_attn_qkvpacked_func

batch, seqlen, heads, dim = 4, 1024, 32, 128

# Create packed QKV tensor
qkv = torch.randn(batch, seqlen, 3, heads, dim, device='cuda', dtype=torch.float16, requires_grad=True)

out = flash_attn_qkvpacked_func(qkv, causal=True)
```

---

## flash_attn_kvpacked_func

Optimized function when K, V are already stacked. Supports MQA/GQA by having
fewer heads in KV than in Q.

### Signature

```python
def flash_attn_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
```

### Parameters

| Parameter | Type | Shape | Default | Description |
|-----------|------|-------|---------|-------------|
| `q` | `Tensor` | `(batch, seqlen_q, nheads, headdim)` | required | Query tensor |
| `kv` | `Tensor` | `(batch, seqlen, 2, nheads_k, headdim)` | required | Packed KV tensor |
| `dropout_p` | `float` | scalar | `0.0` | Dropout probability |
| `softmax_scale` | `float` | scalar | `None` | Scaling factor |
| `causal` | `bool` | scalar | `False` | Causal mask |
| `window_size` | `Tuple[int, int]` | `(left, right)` | `(-1, -1)` | Sliding window |
| `softcap` | `float` | scalar | `0.0` | Softcap value |
| `alibi_slopes` | `Tensor` | `(nheads,)` or `(batch, nheads)` | `None` | ALiBi slopes |
| `deterministic` | `bool` | scalar | `False` | Deterministic backward |
| `return_attn_probs` | `bool` | scalar | `False` | Return attention probs |

### Return Value

Same as `flash_attn_func`.

### MQA/GQA Example

```python
import torch
from flash_attn import flash_attn_kvpacked_func

batch, seqlen, nheads_q, nheads_kv, dim = 4, 1024, 32, 8, 128

q = torch.randn(batch, seqlen, nheads_q, dim, device='cuda', dtype=torch.float16)
kv = torch.randn(batch, seqlen, 2, nheads_kv, dim, device='cuda', dtype=torch.float16)

out = flash_attn_kvpacked_func(q, kv, causal=True)
```

---

## flash_attn_varlen_func

Variable-length attention for batches with different sequence lengths.
Avoids padding waste by concatenating all tokens and using cumulative sequence
lengths to index into them.

### Signature

```python
def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
```

### Parameters

| Parameter | Type | Shape | Default | Description |
|-----------|------|-------|---------|-------------|
| `q` | `Tensor` | `(total_q, nheads, headdim)` | required | All query tokens concatenated |
| `k` | `Tensor` | `(total_k, nheads_k, headdim)` | required | All key tokens concatenated |
| `v` | `Tensor` | `(total_k, nheads_k, headdim)` | required | All value tokens concatenated |
| `cu_seqlens_q` | `Tensor` | `(batch_size + 1,)` int32 | required | Cumulative query sequence lengths |
| `cu_seqlens_k` | `Tensor` | `(batch_size + 1,)` int32 | required | Cumulative key sequence lengths |
| `max_seqlen_q` | `int` | scalar | required | Maximum query sequence length |
| `max_seqlen_k` | `int` | scalar | required | Maximum key sequence length |
| `dropout_p` | `float` | scalar | `0.0` | Dropout probability |
| `softmax_scale` | `float` | scalar | `None` | Scaling factor |
| `causal` | `bool` | scalar | `False` | Causal mask |
| `window_size` | `Tuple[int, int]` | `(left, right)` | `(-1, -1)` | Sliding window |
| `softcap` | `float` | scalar | `0.0` | Softcap value |
| `alibi_slopes` | `Tensor` | `(nheads,)` or `(batch, nheads)` | `None` | ALiBi slopes |
| `deterministic` | `bool` | scalar | `False` | Deterministic backward |
| `return_attn_probs` | `bool` | scalar | `False` | Return attention probs |
| `block_table` | `Tensor` | `(batch, max_num_blocks)` int32 | `None` | Paged KV cache block table |

### Return Value

**Default:**
- `out`: `(total_q, nheads, headdim)` - All output tokens concatenated

**With attention probs:**
- `out`: `(total_q, nheads, headdim)`
- `softmax_lse`: `(nheads, total_q)` - Log-sum-exp (note: different shape than non-varlen)
- `S_dmask`: `(batch_size, nheads, max_seqlen_q, max_seqlen_k)`

### Key Differences from Non-Varlen

1. **Tensor shapes**: No batch dimension; all tokens concatenated along first axis
2. **cu_seqlens**: `cu_seqlens[0] = 0`, `cu_seqlens[i+1] = cu_seqlens[i] + seqlen_i`
3. **max_seqlen**: Must be provided explicitly (not inferrable from tensor shape)
4. **LSE shape**: `(nheads, total_q)` instead of `(batch, nheads, seqlen)`

### Example

```python
import torch
from flash_attn import flash_attn_varlen_func

# Batch with sequences of different lengths: [3, 5, 2]
cu_seqlens = torch.tensor([0, 3, 8, 10], dtype=torch.int32, device='cuda')
total_tokens = 10
max_seqlen = 5
nheads, dim = 8, 128

q = torch.randn(total_tokens, nheads, dim, device='cuda', dtype=torch.float16)
k = torch.randn(total_tokens, nheads, dim, device='cuda', dtype=torch.float16)
v = torch.randn(total_tokens, nheads, dim, device='cuda', dtype=torch.float16)

out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens, cu_seqlens,
    max_seqlen, max_seqlen,
    causal=True,
)
print(out.shape)  # (10, 8, 128)
```

---

## flash_attn_varlen_qkvpacked_func

Variable-length attention with packed QKV.

### Signature

```python
def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
```

### Parameters

| Parameter | Type | Shape | Default | Description |
|-----------|------|-------|---------|-------------|
| `qkv` | `Tensor` | `(total, 3, nheads, headdim)` | required | Packed QKV for all tokens |
| `cu_seqlens` | `Tensor` | `(batch_size + 1,)` int32 | required | Cumulative sequence lengths |
| `max_seqlen` | `int` | scalar | required | Maximum sequence length |
| Others | | | | Same as `flash_attn_qkvpacked_func` |

### Return Value

Same as `flash_attn_varlen_func`.

---

## flash_attn_varlen_kvpacked_func

Variable-length attention with packed KV. Supports MQA/GQA.

### Signature

```python
def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
```

### Parameters

| Parameter | Type | Shape | Default | Description |
|-----------|------|-------|---------|-------------|
| `q` | `Tensor` | `(total_q, nheads, headdim)` | required | All query tokens |
| `kv` | `Tensor` | `(total_k, 2, nheads_k, headdim)` | required | Packed KV for all tokens |
| `cu_seqlens_q` | `Tensor` | `(batch_size + 1,)` int32 | required | Cumulative query seq lengths |
| `cu_seqlens_k` | `Tensor` | `(batch_size + 1,)` int32 | required | Cumulative key seq lengths |
| `max_seqlen_q` | `int` | scalar | required | Max query sequence length |
| `max_seqlen_k` | `int` | scalar | required | Max key sequence length |
| Others | | | | Same as `flash_attn_kvpacked_func` |

---

## flash_attn_with_kvcache

Inference-optimized function for incremental decoding with KV cache.
Supports in-place KV cache updates and rotary embedding application.
**Does not support backward pass.**

### Signature

```python
def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    alibi_slopes: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    return_softmax_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
```

### Parameters

| Parameter | Type | Shape | Default | Description |
|-----------|------|-------|---------|-------------|
| `q` | `Tensor` | `(batch, seqlen_q, nheads, headdim)` | required | Query tensor (typically 1 token for decoding) |
| `k_cache` | `Tensor` | See below | required | Pre-allocated key cache |
| `v_cache` | `Tensor` | See below | required | Pre-allocated value cache |
| `k` | `Tensor` | `(batch, seqlen_new, nheads_k, headdim)` | `None` | New keys to append to cache |
| `v` | `Tensor` | `(batch, seqlen_new, nheads_k, headdim)` | `None` | New values to append to cache |
| `rotary_cos` | `Tensor` | `(seqlen_ro, rotary_dim/2)` | `None` | Cosine for rotary embedding |
| `rotary_sin` | `Tensor` | `(seqlen_ro, rotary_dim/2)` | `None` | Sine for rotary embedding |
| `cache_seqlens` | `int` or `Tensor` | `(batch,)` int32 | `None` | Current cache sequence lengths |
| `cache_batch_idx` | `Tensor` | `(batch,)` int32 | `None` | Indices into the KV cache batch |
| `cache_leftpad` | `Tensor` | `(batch,)` int32 | `None` | Left padding of KV cache |
| `block_table` | `Tensor` | `(batch, max_num_blocks)` int32 | `None` | Paged KV cache block table |
| `softmax_scale` | `float` | scalar | `None` | Scaling factor |
| `causal` | `bool` | scalar | `False` | Causal mask |
| `window_size` | `Tuple[int, int]` | `(left, right)` | `(-1, -1)` | Sliding window |
| `softcap` | `float` | scalar | `0.0` | Softcap value |
| `rotary_interleaved` | `bool` | scalar | `True` | Interleaved (True) or GPT-NeoX (False) rotary |
| `alibi_slopes` | `Tensor` | `(nheads,)` or `(batch, nheads)` | `None` | ALiBi slopes |
| `num_splits` | `int` | scalar | `0` | KV split count for parallel decoding. `0`=auto, `1`=no split. |
| `return_softmax_lse` | `bool` | scalar | `False` | Return log-sum-exp values |

### KV Cache Shapes

**Without block_table (contiguous cache):**
```
k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim)
v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim)
```

**With block_table (paged KV cache):**
```
k_cache: (num_blocks, page_block_size, nheads_k, headdim)
v_cache: (num_blocks, page_block_size, nheads_k, headdim)
```
Where `page_block_size` must be a multiple of 256 (FA2).

### Rotary Embedding Behavior

When `rotary_cos` and `rotary_sin` are provided (along with `k` and `v`):

- **Causal or local attention**: Query tokens are rotated at positions
  `cache_seqlens`, `cache_seqlens + 1`, etc.
- **Non-causal, non-local**: All query tokens are rotated at position
  `cache_seqlens` only.

Key tokens are always rotated at positions starting from `cache_seqlens`.

`rotary_dim` must be divisible by 16.

### num_splits Heuristic

For `num_splits=0` (auto), the heuristic considers:
- If `seqlen_q` is small (e.g., 1 for token generation), the bottleneck is
  loading KV cache. Splitting KV across thread blocks helps.
- The number of splits is chosen based on sequence lengths and head dimension.

### Return Value

**Default:**
- `out`: `(batch, seqlen_q, nheads, headdim)`

**With `return_softmax_lse=True`:**
- `out`: `(batch, seqlen_q, nheads, headdim)`
- `softmax_lse`: `(batch, nheads, seqlen_q)` - Log-sum-exp values

### Example

```python
import torch
from flash_attn import flash_attn_with_kvcache

# Setup
batch, max_seqlen, nheads, nheads_kv, dim = 4, 2048, 32, 8, 128

# Pre-allocate KV cache
k_cache = torch.zeros(batch, max_seqlen, nheads_kv, dim, device='cuda', dtype=torch.float16)
v_cache = torch.zeros(batch, max_seqlen, nheads_kv, dim, device='cuda', dtype=torch.float16)

# Initial prefill: insert all tokens at once
q_prefill = torch.randn(batch, 512, nheads, dim, device='cuda', dtype=torch.float16)
k_prefill = torch.randn(batch, 512, nheads_kv, dim, device='cuda', dtype=torch.float16)
v_prefill = torch.randn(batch, 512, nheads_kv, dim, device='cuda', dtype=torch.float16)

cache_seqlens = torch.zeros(batch, dtype=torch.int32, device='cuda')
out = flash_attn_with_kvcache(
    q_prefill, k_cache, v_cache,
    k=k_prefill, v=v_prefill,
    cache_seqlens=cache_seqlens,
    causal=True,
)
cache_seqlens += 512

# Incremental decoding: one token at a time
for _ in range(100):
    q_decode = torch.randn(batch, 1, nheads, dim, device='cuda', dtype=torch.float16)
    k_new = torch.randn(batch, 1, nheads_kv, dim, device='cuda', dtype=torch.float16)
    v_new = torch.randn(batch, 1, nheads_kv, dim, device='cuda', dtype=torch.float16)

    out = flash_attn_with_kvcache(
        q_decode, k_cache, v_cache,
        k=k_new, v=v_new,
        cache_seqlens=cache_seqlens,
        causal=True,
    )
    cache_seqlens += 1
```

### Paged KV Cache Example

```python
import torch
from flash_attn import flash_attn_with_kvcache

batch, nheads, nheads_kv, dim = 4, 32, 8, 128
page_size = 256  # Must be multiple of 256 for FA2
max_num_pages_per_seq = 32

# Allocate pages
num_blocks = batch * max_num_pages_per_seq
k_cache = torch.zeros(num_blocks, page_size, nheads_kv, dim, device='cuda', dtype=torch.float16)
v_cache = torch.zeros(num_blocks, page_size, nheads_kv, dim, device='cuda', dtype=torch.float16)

# Block table maps each sequence to its pages
block_table = torch.zeros(batch, max_num_pages_per_seq, dtype=torch.int32, device='cuda')

q = torch.randn(batch, 1, nheads, dim, device='cuda', dtype=torch.float16)
k = torch.randn(batch, 1, nheads_kv, dim, device='cuda', dtype=torch.float16)
v = torch.randn(batch, 1, nheads_kv, dim, device='cuda', dtype=torch.float16)
cache_seqlens = torch.tensor([100, 200, 50, 150], dtype=torch.int32, device='cuda')

out = flash_attn_with_kvcache(
    q, k_cache, v_cache,
    k=k, v=v,
    cache_seqlens=cache_seqlens,
    block_table=block_table,
    causal=True,
)
```

---

## flash_blocksparse_attn_func

Block-sparse attention that skips computing attention for masked-out blocks.
This is useful for long-sequence models where most attention blocks are zero.

### Signature

```python
def flash_blocksparse_attn_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    blockmask: torch.Tensor,
    dropout_p: float,
    max_s: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    return_attn_probs: bool = False,
    convert_mask: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
```

### Parameters

| Parameter | Type | Shape | Default | Description |
|-----------|------|-------|---------|-------------|
| `qkv` | `Tensor` | `(total, 3, nheads, headdim)` | required | Packed QKV (varlen format) |
| `cu_seqlens` | `Tensor` | `(batch_size + 1,)` int32 | required | Cumulative sequence lengths |
| `blockmask` | `Tensor` | `(nblocks_row, nblocks_col)` | required | 0-1 mask (1=compute, 0=skip) |
| `dropout_p` | `float` | scalar | required | Dropout probability |
| `max_s` | `int` | scalar | required | Maximum sequence length |
| `softmax_scale` | `float` | scalar | `None` | Scaling factor |
| `causal` | `bool` | scalar | `False` | Causal mask (must be False for block sparse) |
| `return_attn_probs` | `bool` | scalar | `False` | Return attention probs |
| `convert_mask` | `bool` | scalar | `True` | Auto-convert 0-1 mask to internal format |

### Notes

- `causal` must be `False` for block-sparse attention.
- Block size is 64 (for head_dim 64) or 128 (for head_dim <= 64).
- Uses `flash_attn_cuda` backend (not the same as FA2 main module).

---

## Autograd Function Classes

All public functions delegate to `torch.autograd.Function` subclasses that
handle the forward and backward passes.

### FlashAttnFunc

Wraps `flash_attn_func`. Handles:
- Head dimension padding to multiples of 8
- Forward: calls `_wrapped_flash_attn_forward`
- Backward: allocates dQ, dK, dV and calls `_wrapped_flash_attn_backward`
- Saves Q, K, V, out, softmax_lse, rng_state for backward

### FlashAttnQKVPackedFunc

Wraps `flash_attn_qkvpacked_func`. Handles:
- Unpacking Q, K, V from the packed tensor
- Forward: unpacks and calls forward
- Backward: allocates packed dQKV and writes gradients into it

### FlashAttnKVPackedFunc

Wraps `flash_attn_kvpacked_func`. Handles:
- Unpacking K, V from the packed tensor
- Forward: unpacks K, V and calls forward
- Backward: allocates dQ and packed dKV

### FlashAttnVarlenFunc

Wraps `flash_attn_varlen_func`. Handles:
- Variable-length sequence processing
- Optional block_table for paged KV cache
- Different tensor shapes (no batch dimension)

### FlashAttnVarlenQKVPackedFunc

Wraps `flash_attn_varlen_qkvpacked_func`. Variable-length with packed QKV.

### FlashAttnVarlenKVPackedFunc

Wraps `flash_attn_varlen_kvpacked_func`. Variable-length with packed KV.

### Head Dimension Padding

All autograd classes automatically pad the head dimension to the next multiple
of 8 if it is not already aligned:

```python
head_size_og = q.size(3)  # or q.size(2) for varlen
if head_size_og % 8 != 0:
    q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
    k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
    v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
```

The output is then trimmed back to the original head dimension.

---

## Custom Op Registration

FA2 registers custom PyTorch operations for `torch.compile` support
(requires PyTorch >= 2.4).

### Registered Operations

| Operation | Mutates | Description |
|-----------|---------|-------------|
| `flash_attn::_flash_attn_forward` | None | Forward pass |
| `flash_attn::_flash_attn_backward` | `dq, dk, dv` | Backward pass |
| `flash_attn::_flash_attn_varlen_forward` | None | Varlen forward |
| `flash_attn::_flash_attn_varlen_backward` | `dq, dk, dv` | Varlen backward |

### torch.compile Usage

```python
import torch
from flash_attn import flash_attn_func

@torch.compile
def compiled_attention(q, k, v):
    return flash_attn_func(q, k, v, causal=True)

q = torch.randn(2, 512, 16, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 512, 16, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 512, 16, 64, device='cuda', dtype=torch.float16)

out = compiled_attention(q, k, v)
```

---

## Block Size Selection Heuristic

The function `_get_block_size_n` selects the optimal block size for the K/V
dimension based on GPU architecture, head dimension, dropout, and causal mode.

```python
def _get_block_size_n(device, head_dim, is_dropout, is_causal):
    assert head_dim <= 256
    major, minor = torch.cuda.get_device_capability(device)
    is_sm8x = major == 8 and minor > 0
    is_sm80 = major == 8 and minor == 0
    is_sm90 = major == 9 and minor == 0
```

### Block Size Table

| Head Dim | SM80 (A100) | SM8x (Consumer) | SM90 (Hopper) |
|----------|-------------|------------------|---------------|
| <= 32 | 128 | 128 | 128 |
| <= 64 | 128 / 64(dropout) | 128 / 64(dropout) | 128 / 64(dropout) |
| <= 96 | 64 | 64 | 64 |
| <= 128 | 64 / 32(dropout) | 64 / 32(causal+no dropout) | 64 / 32(dropout) |
| <= 192 | 64 | 64 | 64 |
| <= 256 | 64 | 64 | 64 |

---

## Causal Mask Behavior

Since FA2 v2.1, the causal mask is aligned to the **bottom-right corner** of
the attention matrix.

### Equal Sequence Lengths (seqlen_q == seqlen_k)

Standard lower-triangular mask:
```
1 0 0 0 0
1 1 0 0 0
1 1 1 0 0
1 1 1 1 0
1 1 1 1 1
```

### seqlen_q < seqlen_k (e.g., q=2, k=5)

The causal mask is aligned to the bottom-right:
```
1 1 1 1 0
1 1 1 1 1
```
This means the last `seqlen_q` positions attend causally to all keys.

### seqlen_q > seqlen_k (e.g., q=5, k=2)

The first `seqlen_q - seqlen_k` query positions have an all-zero mask row,
producing zero output:
```
0 0
0 0
0 0
1 0
1 1
```

### Implementation

The causal mask is implemented inside the CUDA kernel by:
1. Computing the position offset: `seqlen_k - seqlen_q`
2. Skipping entire blocks that are fully masked (optimization)
3. Applying the causal mask within partially masked blocks

---

## ALiBi Support

ALiBi (Attention with Linear Bias) adds position-dependent bias to attention
scores:

```
score(i, j) += -alibi_slope * |i - j|
```

### Usage

```python
# Per-head ALiBi slopes
alibi_slopes = torch.tensor([0.5, 0.25, 0.125, ...], dtype=torch.float32, device='cuda')

# Or per-batch-per-head
alibi_slopes = torch.randn(batch_size, nheads, dtype=torch.float32, device='cuda')

out = flash_attn_func(q, k, v, alibi_slopes=alibi_slopes, causal=True)
```

### Shape Requirements

- `(nheads,)` - Same slopes for all batches
- `(batch_size, nheads)` - Different slopes per batch
- Must be FP32
- The bias added is `(-alibi_slope * |i + seqlen_k - seqlen_q - j|)` for
  cross-attention (accounts for position offset)

### Notes

- ALiBi is typically used with causal attention
- The slopes are usually chosen as a geometric sequence: `2^(-8/n) * 2^i` for i=0,...,n-1
- When used with packed KV functions, the bias formula uses the absolute
  position difference

---

## Softcapping

Softcapping bounds the attention logits to prevent numerical instability:

```
S = tanh(S / softcap) * softcap
```

### Usage

```python
# Gemma-2 style softcapping
out = flash_attn_func(q, k, v, softcap=50.0)
```

### Notes

- `softcap > 0` enables softcapping
- `softcap <= 0` or `softcap = 0.0` disables softcapping
- The formula bounds logits to `[-softcap, softcap]`
- Used in Gemma-2 and Grok models
- Supported in both forward and backward passes

---

## Sliding Window Attention

Sliding window (local) attention restricts each query to attend only to a
local window of keys.

### Usage

```python
# Look back 512 tokens, look ahead 0 tokens (causal local)
out = flash_attn_func(q, k, v, window_size=(512, 0))

# Look back 256 tokens, look ahead 64 tokens (bidirectional local)
out = flash_attn_func(q, k, v, window_size=(256, 64))
```

### Window Semantics

For `flash_attn_func` and `flash_attn_kvpacked_func`:
- Query at position `i` attends to keys in range:
  `[i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]]`

For `flash_attn_qkvpacked_func`:
- Query at position `i` attends to keys in range:
  `[i - window_size[0], i + window_size[1]]`

### Window Size Values

| Value | Meaning |
|-------|---------|
| `(-1, -1)` | No window restriction (default) |
| `(W, -1)` | Look back W tokens, no forward limit |
| `(-1, W)` | No backward limit, look ahead W tokens |
| `(W, 0)` | Look back W tokens only (causal local) |

### Notes

- Can be combined with `causal=True`
- Used in Mistral 7B and other models with local attention
- When `window_size_left >= seqlen_k - 1`, it's effectively no restriction
- When the window is empty for a query position, the output is zero

---

## MQA/GQA Support

Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) allow sharing
K/V heads across multiple Q heads.

### Usage

```python
import torch
from flash_attn import flash_attn_func

# GQA: 32 query heads, 8 KV heads (4:1 ratio)
batch, seqlen, nheads_q, nheads_kv, dim = 4, 1024, 32, 8, 128

q = torch.randn(batch, seqlen, nheads_q, dim, device='cuda', dtype=torch.float16)
k = torch.randn(batch, seqlen, nheads_kv, dim, device='cuda', dtype=torch.float16)
v = torch.randn(batch, seqlen, nheads_kv, dim, device='cuda', dtype=torch.float16)

out = flash_attn_func(q, k, v)
# Q heads 0-3 attend to KV head 0
# Q heads 4-7 attend to KV head 1
# ... and so on

# MQA: 32 query heads, 1 KV head
k_mqa = torch.randn(batch, seqlen, 1, dim, device='cuda', dtype=torch.float16)
v_mqa = torch.randn(batch, seqlen, 1, dim, device='cuda', dtype=torch.float16)
out = flash_attn_func(q, k_mqa, v_mqa)
# All 32 Q heads attend to the same KV head
```

### Constraints

- `nheads_q % nheads_kv == 0` (Q heads must be divisible by KV heads)
- The mapping is sequential: Q heads `[i * ratio, ..., (i+1) * ratio - 1]`
  attend to KV head `i`

---

## Paged KV Cache

Paged KV cache (PagedAttention) avoids memory fragmentation by allocating KV
cache in fixed-size blocks (pages).

### Usage with flash_attn_with_kvcache

```python
# Paged KV cache setup
page_size = 256  # Must be multiple of 256 for FA2
max_num_pages_per_seq = 64
num_blocks = batch * max_num_pages_per_seq

k_cache = torch.zeros(num_blocks, page_size, nheads_kv, dim, device='cuda', dtype=torch.float16)
v_cache = torch.zeros(num_blocks, page_size, nheads_kv, dim, device='cuda', dtype=torch.float16)
block_table = torch.zeros(batch, max_num_pages_per_seq, dtype=torch.int32, device='cuda')

out = flash_attn_with_kvcache(
    q, k_cache, v_cache,
    k=k_new, v=v_new,
    cache_seqlens=cache_seqlens,
    block_table=block_table,
    causal=True,
)
```

### Usage with flash_attn_varlen_func

```python
# Paged KV with varlen
out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    block_table=block_table,
)
```

---

## Rotary Embedding

FA2 supports rotary embedding via `flash_attn_with_kvcache`, which applies
rotation in-kernel without additional memory allocation.

### Usage

```python
# Prepare rotary embedding
rotary_dim = 128  # Must be divisible by 16
max_seq = 4096
inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
positions = torch.arange(max_seq, dtype=torch.float32)
freqs = torch.outer(positions, inv_freq)
rotary_cos = torch.cos(freqs).to(device='cuda', dtype=torch.float16)
rotary_sin = torch.sin(freqs).to(device='cuda', dtype=torch.float16)

out = flash_attn_with_kvcache(
    q, k_cache, v_cache,
    k=k_new, v=v_new,
    rotary_cos=rotary_cos,
    rotary_sin=rotary_sin,
    cache_seqlens=cache_seqlens,
    rotary_interleaved=True,  # True for interleaved, False for GPT-NeoX
    causal=True,
)
```

### Rotary Styles

| Style | `rotary_interleaved` | Dimensions combined |
|-------|---------------------|-------------------|
| Interleaved (GPT-J) | `True` | (0,1), (2,3), (4,5), ... |
| Half-rotation (GPT-NeoX) | `False` | (0, d/2), (1, d/2+1), ... |

---

## Error Handling

### Common Errors

**Head dimension not aligned:**
```
RuntimeError: head_dim must be divisible by 8
```
Solution: The kernel automatically pads to multiples of 8.

**Wrong number of heads:**
```
AssertionError: num_head must be divisible by num_head_kv
```
Solution: Ensure `nheads_q % nheads_kv == 0`.

**CUDA device mismatch:**
```
RuntimeError: Expected all tensors to be on the same device
```
Solution: Ensure all input tensors are on the same CUDA device.

**Data type mismatch:**
```
RuntimeError: q, k, v must have the same dtype
```
Solution: Cast all inputs to the same dtype (FP16 or BF16).

**Memory-contiguous requirement:**
```
RuntimeError: k_cache must have contiguous last dimension
```
Solution: Call `.contiguous()` on the tensor or ensure the last dimension
stride is 1.

**Unsupported head dimension:**
```
RuntimeError: FlashAttention only supports head_dim <= 256
```
Solution: Use head dimension <= 256. For larger, project to smaller dim.

### Numerical Precision

FlashAttention's output may differ slightly from standard attention due to:

1. **Online softmax**: Different order of floating-point operations
2. **BF16 vs FP16**: BF16 has less mantissa precision
3. **Softcapping**: Additional nonlinearity

The maximum numerical error should be at most 2x that of a baseline PyTorch
implementation. Tests verify this across all configurations.

---

## Common Patterns and Examples

### Standard Multi-Head Attention Layer

```python
import torch
import torch.nn as nn
from flash_attn import flash_attn_func

class FlashMHA(nn.Module):
    def __init__(self, dim, nheads, causal=False):
        super().__init__()
        self.dim = dim
        self.nheads = nheads
        self.head_dim = dim // nheads
        self.causal = causal

        self.Wqkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.Wqkv(x).reshape(B, S, 3, self.nheads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        out = flash_attn_func(q, k, v, causal=self.causal)
        return self.out_proj(out.reshape(B, S, D))
```

### GQA with Sliding Window

```python
class GQASlidingWindow(nn.Module):
    def __init__(self, dim, nheads_q, nheads_kv, window_size):
        super().__init__()
        self.nheads_q = nheads_q
        self.nheads_kv = nheads_kv
        self.head_dim = dim // nheads_q
        self.window_size = window_size

        self.q_proj = nn.Linear(dim, nheads_q * self.head_dim)
        self.k_proj = nn.Linear(dim, nheads_kv * self.head_dim)
        self.v_proj = nn.Linear(dim, nheads_kv * self.head_dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, S, D = x.shape
        q = self.q_proj(x).reshape(B, S, self.nheads_q, self.head_dim)
        k = self.k_proj(x).reshape(B, S, self.nheads_kv, self.head_dim)
        v = self.v_proj(x).reshape(B, S, self.nheads_kv, self.head_dim)

        out = flash_attn_func(q, k, v,
                              window_size=self.window_size,
                              causal=True)
        return self.out_proj(out.reshape(B, S, D))
```

### BERT-style Attention with Variable Lengths

```python
from flash_attn import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input

class BertFlashAttention(nn.Module):
    def __init__(self, dim, nheads):
        super().__init__()
        self.nheads = nheads
        self.head_dim = dim // nheads
        self.Wqkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, attention_mask):
        # attention_mask: (B, S) with 1 for real tokens, 0 for padding
        B, S, D = x.shape

        # Unpad: remove padding tokens
        qkv, indices, cu_seqlens, max_seqlen = unpad_input(
            self.Wqkv(x).reshape(B, S, 3, self.nheads, self.head_dim),
            attention_mask
        )
        # qkv: (total_real_tokens, 3, nheads, head_dim)

        out = flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, max_seqlen, causal=False
        )

        # Repad: add padding back
        out = pad_input(out, indices, B, S)
        return self.out_proj(out.reshape(B, S, D))
```
