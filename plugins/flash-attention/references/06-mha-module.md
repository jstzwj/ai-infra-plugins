# FlashAttention MHA Module Reference

This document provides an exhaustive reference for the high-level Multi-Head Attention (MHA)
module in FlashAttention. This module provides `nn.Module` wrappers that integrate flash
attention kernels into PyTorch models with support for self-attention, cross-attention,
GQA/MQA, rotary embeddings, ALiBi, KV caching, and tensor parallelism.

## Table of Contents

1. [Module Overview](#1-module-overview)
2. [FlashSelfAttention](#2-flashselfattention)
3. [FlashCrossAttention](#3-flashcrossattention)
4. [SelfAttention (Reference)](#4-selfattention-reference)
5. [CrossAttention (Reference)](#5-crossattention-reference)
6. [MHA](#6-mha)
7. [ParallelMHA](#7-parallelmha)
8. [ALiBi Slopes](#8-alibi-slopes)
9. [KV Cache Management](#9-kv-cache-management)
10. [Rotary Embedding Integration](#10-rotary-embedding-integration)

---

## 1. Module Overview

The MHA module (`flash_attn/modules/mha.py`) provides the following class hierarchy:

```
nn.Module
  |-- FlashSelfAttention    (Flash-optimized self-attention)
  |-- FlashCrossAttention   (Flash-optimized cross-attention)
  |-- SelfAttention         (Reference PyTorch self-attention)
  |-- CrossAttention        (Reference PyTorch cross-attention)
  |-- MHA                   (Full MHA module with projections)
  |-- ParallelMHA           (Tensor-parallel MHA)
```

Each "Flash" variant dispatches to C++/CUDA kernels (`flash_attn_qkvpacked_func`,
`flash_attn_kvpacked_func`, `flash_attn_varlen_*`, `flash_attn_with_kvcache`), while
the non-Flash variants use pure PyTorch operations (useful for debugging/fallback).

---

## 2. FlashSelfAttention

### Class: `FlashSelfAttention(nn.Module)`

Flash-optimized multi-head self-attention with softmax. Wraps the packed QKV flash
attention kernel.

**Constructor Parameters**:
- `causal: bool = False` - Whether to apply causal (autoregressive) masking
- `softmax_scale: float | None = None` - Temperature for softmax (default: `1/sqrt(d)`)
- `attention_dropout: float = 0.0` - Dropout rate for attention weights
- `window_size: Tuple[int, int] = (-1, -1)` - Sliding window `(left, right)` bounds.
  `(-1, -1)` means no window constraint.
- `alibi_slopes: Tensor | None = None` - Per-head ALiBi bias slopes
- `deterministic: bool = False` - Whether to use deterministic backward

**Forward Method**:
```python
def forward(self, qkv, causal=None, cu_seqlens=None, max_seqlen=None)
```

**Input Shapes**:
- Standard: `qkv` is `(B, S, 3, H, D)` where B=batch, S=sequence, H=heads, D=head_dim
- Variable-length (unpadded): `qkv` is `(total, 3, H, D)` where `total = sum(seqlens)`,
  with `cu_seqlens` shape `(B+1,)` and `max_seqlen` as int

**Output Shape**: Same as input minus the QKV dimension: `(B, S, H, D)` or `(total, H, D)`

**Dispatch Logic**:
- If `cu_seqlens` is provided: calls `flash_attn_varlen_qkvpacked_func`
- Otherwise: calls `flash_attn_qkvpacked_func`
- Training mode applies dropout; eval mode disables it

---

## 3. FlashCrossAttention

### Class: `FlashCrossAttention(nn.Module)`

Flash-optimized multi-head cross-attention where Q comes from one sequence and K,V from another.

**Constructor Parameters**: Same as `FlashSelfAttention`.

**Forward Method**:
```python
def forward(self, q, kv, causal=None, cu_seqlens=None, max_seqlen=None,
            cu_seqlens_k=None, max_seqlen_k=None)
```

**Input Shapes**:
- `q`: `(B, Sq, H, D)` - Query tensor
- `kv`: `(B, Sk, 2, Hk, D)` - Key-Value packed tensor (Hk may differ from H for GQA/MQA)

**Variable-length Support**:
- `cu_seqlens`, `max_seqlen`: For Q sequence lengths
- `cu_seqlens_k`, `max_seqlen_k`: For K/V sequence lengths

**Dispatch Logic**:
- Variable-length: `flash_attn_varlen_kvpacked_func`
- Standard: `flash_attn_kvpacked_func`

---

## 4. SelfAttention (Reference)

### Class: `SelfAttention(nn.Module)`

Reference implementation of self-attention using pure PyTorch operations. Useful for
debugging and correctness verification.

**Constructor Parameters**:
- `causal: bool = False`
- `softmax_scale: float | None = None`
- `attention_dropout: float = 0.0`

**Forward Method**:
```python
def forward(self, qkv, causal=None, key_padding_mask=None)
```

**Implementation Details**:
1. Split QKV: `q, k, v = qkv.unbind(dim=2)`
2. Compute scores: `torch.einsum("bthd,bshd->bhts", q, k * scale)`
3. Apply key padding mask by adding -10000 to masked positions
4. Apply causal mask via `torch.triu` with -10000 fill
5. Softmax: `torch.softmax(scores, dim=-1)`
6. Dropout on attention weights
7. Output: `torch.einsum("bhts,bshd->bthd", attention, v)`

**Key Padding Mask**: Boolean tensor `(B, S)` where True means keep, False means mask out.
Implemented by adding -10000 to masked positions (faster than `masked_fill_`).

---

## 5. CrossAttention (Reference)

### Class: `CrossAttention(nn.Module)`

Reference cross-attention implementation with GQA/MQA support.

**Forward Method**:
```python
def forward(self, q, kv, causal=None, key_padding_mask=None)
```

**GQA/MQA Handling**:
If `kv.shape[3] != q.shape[2]` (different number of KV heads), repeats KV heads:
```python
kv = repeat(kv, "... hkv d -> ... (hkv g) d", g=q.shape[2] // kv.shape[3])
```

**Causal Mask for Cross-Attention**:
Handles the case where `seqlen_q != seqlen_k`:
```python
col_idx > row_idx + seqlen_k - seqlen_q
```
Adjusts for key padding when present.

---

## 6. MHA

### Class: `MHA(nn.Module)`

Complete multi-head attention module with linear projections, rotary embeddings, ALiBi,
KV caching, GQA/MQA support, and optional gradient checkpointing.

**Constructor Parameters**:
- `embed_dim: int` - Total embedding dimension
- `num_heads: int` - Number of query attention heads
- `num_heads_kv: int | None` - Number of KV heads (None = same as num_heads). Setting
  `num_heads_kv = 1` enables MQA; `num_heads_kv < num_heads` enables GQA.
- `cross_attn: bool = False` - Whether to use cross-attention
- `qkv_proj_bias: bool = True` - Bias for QKV projection
- `out_proj_bias: bool = True` - Bias for output projection
- `dropout: float = 0.0` - Attention dropout rate
- `softmax_scale: float | None = None` - Softmax temperature
- `causal: bool = False` - Causal masking
- `layer_idx: int | None = None` - Layer index for KV cache management
- `dwconv: bool = False` - Whether to apply depthwise convolution to QKV
- `rotary_emb_dim: int = 0` - Dimension of rotary embedding (0 = disabled)
- `rotary_emb_base: float = 10000.0` - Base for rotary embedding frequency computation
- `rotary_emb_scale_base: float | None = None` - Scale base for xPos rotary variant
- `rotary_emb_interleaved: bool = False` - Whether rotary dimensions are interleaved
- `use_alibi: bool = False` - Whether to use ALiBi positional bias
- `window_size: Tuple[int, int] = (-1, -1)` - Sliding window attention bounds
- `fused_bias_fc: bool = False` - Whether to use fused bias FC (not used in current impl)
- `use_flash_attn: bool = False` - Whether to use FlashAttention kernels
- `return_residual: bool = False` - Whether to return input alongside output
- `checkpointing: bool = False` - Whether to use gradient checkpointing

**Sub-modules Created**:

*Self-attention mode (`cross_attn=False`)*:
- `self.Wqkv: nn.Linear(embed_dim, head_dim * (num_heads + 2 * num_heads_kv))`
  - For MHA (num_heads_kv == num_heads): QKV packed as one projection
  - For GQA/MQA: Q and KV packed but with different head counts
- `self.inner_attn: FlashSelfAttention | SelfAttention`
- `self.inner_cross_attn: FlashCrossAttention | CrossAttention`

*Cross-attention mode (`cross_attn=True`)*:
- `self.Wq: nn.Linear(embed_dim, embed_dim)` - Query projection
- `self.Wkv: nn.Linear(embed_dim, 2 * head_dim * num_heads_kv)` - KV projection
- `self.inner_cross_attn: FlashCrossAttention | CrossAttention`

*Optional*:
- `self.rotary_emb: RotaryEmbedding` - When `rotary_emb_dim > 0`
- `self.dwconv_qkv` or `self.dwconv_q` / `self.dwconv_kv` - When `dwconv=True`

*Always*:
- `self.out_proj: nn.Linear(embed_dim, embed_dim)` - Output projection

**Forward Method**:
```python
def forward(self, x, x_kv=None, key_padding_mask=None, cu_seqlens=None,
            max_seqlen=None, mixer_subset=None, inference_params=None, **kwargs)
```

**Execution Paths**:

The forward method selects between multiple paths based on configuration:

1. **Self-attention MHA (num_heads_kv == num_heads, cross_attn=False)**:
   - Project: `qkv = self.Wqkv(x)` -> rearrange to `(B, S, 3, H, D)`
   - Optional: depthwise conv on QKV
   - Optional: rotary embedding on QKV
   - Attend: `inner_attn(qkv)` or KV cache update path

2. **Self-attention GQA/MQA (num_heads_kv < num_heads)**:
   - Project: `qkv = self.Wqkv(x)`
   - Split: `q = qkv[..., :num_heads * head_dim]`, `kv = qkv[..., num_heads * head_dim:]`
   - Rearrange: `q -> (B, S, H, D)`, `kv -> (B, S, 2, Hkv, D)`
   - Attend via `inner_cross_attn(q, kv)`

3. **Cross-attention (cross_attn=True)**:
   - Project: `q = self.Wq(x)`, `kv = self.Wkv(x_kv or x)`
   - Rearrange and attend via `inner_cross_attn(q, kv)`

**KV Cache (Inference) Paths**:

When `inference_params` is provided, the module switches to inference mode:

- **`_update_kv_cache`**: Writes new KV to the pre-allocated cache tensor and returns
  the full cached KV up to the current position.

- **`_update_kvcache_attention`**: Updates cache then performs attention with the full
  cached KV. Uses `flash_attn_with_kvcache` for the fast path when `seqlen_offset > 0`
  and FlashAttention is available.

- **`_apply_rotary_update_kvcache_attention`**: Fused fast path that combines:
  1. Rotary embedding application (cos/sin from precomputed cache)
  2. KV cache update
  3. Flash attention with KV cache

  This fused path calls `flash_attn_with_kvcache` with `rotary_cos`, `rotary_sin`,
  `cache_seqlens`, and `rotary_interleaved` parameters, delegating rotary computation
  to the CUDA kernel for efficiency.

**Gradient Checkpointing**: When `checkpointing=True`, wraps the attention call in
`torch.utils.checkpoint.checkpoint()` to reduce memory usage during training.

**Return Value**:
- If `return_residual=False`: Returns attention output `(B, S, embed_dim)`
- If `return_residual=True`: Returns `(attention_output, x)` for post-norm residual fusion

---

## 7. ParallelMHA

### Class: `ParallelMHA(nn.Module)`

Tensor-parallel variant of MHA that distributes attention heads across GPUs using
column-parallel and row-parallel linear layers.

**Additional Constructor Parameters**:
- `process_group: ProcessGroup` - Distributed process group
- `sequence_parallel: bool = True` - Whether to use sequence parallelism

**Head Distribution**:
- `num_heads_per_rank = get_dim_for_local_rank(num_heads, world_size, local_rank)`
- `num_heads_kv_per_rank = get_dim_for_local_rank(num_heads_kv, world_size, local_rank)`

**Projections**:
- `self.Wqkv: ColumnParallelLinear` - Column-parallel QKV projection, distributed across
  heads. The `multiple_of` parameter ensures the output dimension is divisible by
  `head_dim * (num_heads/num_heads_kv + 2)` for clean head distribution.
- `self.out_proj: RowParallelLinear` - Row-parallel output projection. Each rank produces
  a partial sum that is all-reduced across ranks.

**Forward Method**:
```python
def forward(self, x, seqlen=None, inference_params=None, **kwargs)
```

When `seqlen` is provided, the input `x` has shape `(batch * seqlen, hidden_dim)` instead
of `(batch, seqlen, hidden_dim)`. This allows splitting the batch*seqlen dimension during
sequence parallelism (important when batch size is small).

**ALiBi in Tensor Parallel**: Each rank computes ALiBi slopes only for its local head range:
```python
alibi_slopes = get_alibi_slopes(num_heads)[local_rank * heads_local : (local_rank + 1) * heads_local]
```

---

## 8. ALiBi Slopes

### Function: `get_albi_slopes`

```python
def get_alibi_slopes(nheads: int) -> List[float]
```

Computes ALiBi (Attention with Linear Biases) slope values for each attention head.
The slopes follow a geometric progression with specific starting points derived from
the formula in the paper "Train Short, Test Long: Attention with Linear Biases Enables
Input Length Extrapolation".

**For power-of-2 head counts**:
```
start = 2^(-(2^(-log2(nheads) - 3)))
slopes = [start * start^i for i in range(nheads)]
```

**For non-power-of-2 head counts**:
Uses slopes from the next power-of-2, taking every other value for the extra heads.

---

## 9. KV Cache Management

### Function: `_update_kv_cache`

```python
def _update_kv_cache(kv, inference_params, layer_idx)
```

Manages the KV cache during autoregressive generation.

**Cache Allocation**:
On first call (layer not in `key_value_memory_dict`), allocates a cache tensor:
```python
kv_cache = torch.empty(
    max_batch_size, max_seqlen, 2, num_heads, head_dim,
    dtype=kv.dtype, device=kv.device
)
```

**Cache Update**:
Writes the new KV values into the cache at the current sequence position:
```python
kv_cache[batch_start:batch_end, sequence_start:sequence_end] = kv
```

Returns the full cached KV up to the current position:
```python
return kv_cache[batch_start:batch_end, :sequence_end]
```

**Batch Offset Support**: Supports `batch_size_offset` and `seqlen_offset` from
`inference_params` for microbatching and continuation.

### Method: `allocate_inference_cache`

Pre-allocates the KV cache tensor with shape `(batch_size, max_seqlen, 2, num_heads_kv, head_dim)`.
For `ParallelMHA`, uses `num_heads_kv_per_rank` instead of `num_heads_kv`.

---

## 10. Rotary Embedding Integration

The MHA module integrates rotary position embeddings (RoPE) through the `RotaryEmbedding`
class from `flash_attn.layers.rotary`.

**Configuration**:
- `rotary_emb_dim` - Number of dimensions to apply rotary embedding to (typically head_dim)
- `rotary_emb_base` - Base frequency (default 10000)
- `rotary_emb_scale_base` - Scale base for xPos variant
- `rotary_emb_interleaved` - Whether dimensions are interleaved (True) or split (False)

**Application Modes**:

1. **Standard path** (`rotary_emb_dim % 16 != 0` or not using flash_attn):
   - Applies rotary embedding on CPU/GPU before the attention call
   - For self-attention: `qkv = self.rotary_emb(qkv, seqlen_offset, max_seqlen)`
   - For cross-attention: `q, kv = self.rotary_emb(q, kv, seqlen_offset, max_seqlen)`

2. **Fused path** (`rotary_emb_dim % 16 == 0` and using flash_attn with KV cache):
   - Passes precomputed `rotary_cos` and `rotary_sin` to `flash_attn_with_kvcache`
   - The CUDA kernel applies rotary embedding during the attention computation
   - Avoids the overhead of separate rotary embedding kernel
   - Requires `rotary_emb.scale is None` (no xPos support in fused path)

**Sequence Offset Handling**:
During inference, `seqlen_offset` is set from `inference_params.lengths_per_sample`
(per-sample lengths for variable-length generation) or `inference_params.seqlen_offset`
(uniform offset). This shifts the rotary position indices to account for previously
generated tokens.
