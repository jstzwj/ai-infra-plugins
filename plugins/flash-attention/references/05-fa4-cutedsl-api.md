# FlashAttention-4: CuTeDSL API Reference

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [flash_attn_func (FA4)](#flash_attn_func-fa4)
4. [flash_attn_varlen_func (FA4)](#flash_attn_varlen_func-fa4)
5. [Score Modifiers (score_mod)](#score-modifiers-score_mod)
6. [Mask Modifiers (mask_mod)](#mask-modifiers-mask_mod)
7. [Block Sparse Attention](#block-sparse-attention)
8. [Pack GQA](#pack-gqa)
9. [Compilation and Caching](#compilation-and-caching)
10. [Environment Variables (FA4)](#environment-variables-fa4)
11. [Advanced Parameters](#advanced-parameters)
12. [Head Dimension Support](#head-dimension-support)
13. [GPU Architecture Targeting](#gpu-architecture-targeting)
14. [SplitKV Attention (FA4)](#splitkv-attention-fa4)
15. [Paged KV Cache (FA4)](#paged-kv-cache-fa4)
16. [MLA Weight-Absorbed Attention](#mla-weight-absorbed-attention)
17. [FP8 Support (FA4)](#fp8-support-fa4)
18. [Autograd Integration](#autograd-integration)
19. [Testing](#testing)
20. [Common Patterns](#common-patterns)

---

## Overview

FlashAttention-4 (FA4) is the latest generation, written entirely in Python
using NVIDIA's CuTeDSL (CUDA Template Engine DSL). Unlike FA2/FA3 which use
hand-written CUDA C++ kernels, FA4 kernels are expressed as Python programs
that are JIT-compiled to PTX/CUBIN at runtime.

### Key Advantages

1. **Python kernel authoring**: Kernels are written in Python, not CUDA C++
2. **User-extensible**: Score modifiers and mask modifiers via `@cute.jit`
3. **Multi-architecture**: SM80, SM90, SM100, SM110, SM120
4. **JIT compilation**: Kernels are compiled on first use and cached
5. **Block sparse**: Native block-sparse attention support
6. **2CTA instructions**: Blackwell two-CTA cluster instructions for head_dim=128/256
7. **MLA absorption**: DeepSeek-style Multi-head Latent Attention
8. **FP8**: FP8 E4M3 and E5M2 support on SM100

### Package Structure

```
flash_attn/cute/
|-- __init__.py              # Exports flash_attn_func, flash_attn_varlen_func
|-- interface.py             # Public API + autograd functions
|-- flash_fwd.py             # SM80 forward kernel
|-- flash_fwd_sm90.py        # SM90 forward kernel
|-- flash_fwd_sm100.py       # SM100 forward (2CTA, SplitKV, paged KV)
|-- flash_fwd_sm120.py       # SM120 forward
|-- flash_fwd_combine.py     # SplitKV combine kernel
|-- flash_fwd_mla_sm100.py   # MLA absorbed forward (SM100)
|-- flash_bwd.py             # SM80 backward kernel
|-- flash_bwd_sm90.py        # SM90 backward kernel
|-- flash_bwd_sm100.py       # SM100 backward (2CTA, block sparse)
|-- flash_bwd_sm120.py       # SM120 backward
|-- flash_bwd_preprocess.py  # Backward preprocessing
|-- flash_bwd_postprocess.py # Backward postprocessing
|-- softmax.py               # Online softmax with score modifiers
|-- mask.py                  # Attention masks
|-- block_info.py            # Tile dimensions
|-- seqlen_info.py           # Sequence length tracking
|-- pipeline.py              # Async data loading pipeline
|-- tile_scheduler.py        # Tile scheduling
|-- pack_gqa.py              # GQA head packing
|-- paged_kv.py              # Paged KV cache
|-- block_sparsity.py        # Block sparse attention
|-- cache_utils.py           # JIT compilation cache
|-- utils.py                 # Utilities
|-- fast_math.py             # Math approximations
|-- copy_utils.py            # Type-converting copies
|-- cute_dsl_utils.py        # CuTeDSL compilation patches
|-- hopper_helpers.py        # SM90 hardware helpers
|-- blackwell_helpers.py     # SM100 hardware helpers
|-- mma_sm100_desc.py        # SM100 MMA descriptors
|-- named_barrier.py         # Warp synchronization
```

### Import

```python
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
```

---

## Installation

### pip Install

```bash
pip install flash-attn-4
```

### With CUDA 13 Extra (Recommended for Blackwell)

```bash
pip install "flash-attn-4[cu13]"
```

### Development Install

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install -e "flash_attn/cute[dev]"
```

### Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| `nvidia-cutlass-dsl` | >= 4.4.1 | CuTeDSL kernel compilation |
| `torch` | >= 2.4 | PyTorch |
| `einops` | Latest | Tensor manipulation |
| `apache-tvm-ffi` | Latest | FFI for kernel execution |
| `quack-kernels` | >= 0.4.0 | Compilation utilities |

---

## flash_attn_func (FA4)

The primary FA4 function for standard batched attention.

### Signature

```python
def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    block_sparse_tensors_bwd: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
```

### Parameters

| Parameter | Type | Shape | Default | Description |
|-----------|------|-------|---------|-------------|
| `q` | `Tensor` | `(batch, seqlen_q, num_heads, head_dim)` | required | Query tensor. Last dim must be contiguous and 16-byte aligned. |
| `k` | `Tensor` | `(batch, seqlen_k, num_heads_kv, head_dim)` | required | Key tensor. |
| `v` | `Tensor` | `(batch, seqlen_k, num_heads_kv, head_dim_v)` | required | Value tensor. Can have different head dim. |
| `qv` | `Tensor` | `(batch, seqlen_q, num_heads, head_dim_v)` | `None` | MLA absorbed QV tensor (SM100 only, head_dim=64, head_dim_v=512). |
| `gather_kv_indices` | `Tensor` | `(batch, seqlen_q, gather_kv_length)` | `None` | TopK gather indices for sparse KV. |
| `softmax_scale` | `float` | scalar | `None` | Scaling factor. Default: `1/sqrt(head_dim)`. |
| `causal` | `bool` | scalar | `False` | Causal attention mask. |
| `window_size` | `Tuple[int, int]` | `(left, right)` | `(None, None)` | Sliding window. `None` means infinite. |
| `learnable_sink` | `Tensor` | `(num_heads,)` | `None` | Learnable attention sink values. BF16. |
| `softcap` | `float` | scalar | `0.0` | Softcapping value. `>0` enables. |
| `num_splits` | `int` | scalar | `1` | SplitKV splits. `1`=disabled. `<1`=auto. |
| `pack_gqa` | `bool` | scalar | `None` | Pack GQA heads. `None`=auto (pack when num_heads > num_heads_kv). |
| `deterministic` | `bool` | scalar | `False` | Deterministic backward. |
| `score_mod` | `Callable` | `@cute.jit` | `None` | User-defined score modifier function. |
| `score_mod_bwd` | `Callable` | `@cute.jit` | `None` | Backward score modifier (if different from forward). |
| `mask_mod` | `Callable` | `@cute.jit` | `None` | User-defined mask modifier function. |
| `aux_tensors` | `list[Tensor]` | varies | `None` | Auxiliary tensors for score/mask mods. |
| `block_sparse_tensors` | `BlockSparseTensorsTorch` | varies | `None` | Block sparse attention configuration (forward). |
| `block_sparse_tensors_bwd` | `BlockSparseTensorsTorch` | varies | `None` | Block sparse config (backward, can differ). |
| `return_lse` | `bool` | scalar | `False` | Return log-sum-exp values. |

### Return Value

**Default (`return_lse=False`):**
- `out`: `(batch, seqlen_q, num_heads, head_dim_v)` - Output tensor

**With LSE (`return_lse=True`):**
- `out`: `(batch, seqlen_q, num_heads, head_dim_v)`
- `lse`: `(batch, num_heads, seqlen_q)` - Log-sum-exp values (FP32)

### Supported Data Types

| Input Type | Output Type | Architectures |
|------------|-------------|---------------|
| `torch.float16` | `torch.float16` | SM80, SM90, SM100, SM110, SM120 |
| `torch.bfloat16` | `torch.bfloat16` | SM80, SM90, SM100, SM110, SM120 |
| `torch.float8_e4m3fn` | `torch.bfloat16` | SM100 only |
| `torch.float8_e5m2` | `torch.bfloat16` | SM100 only |

### Example

```python
import torch
from flash_attn.cute import flash_attn_func

# Basic usage
batch, seqlen, nheads, dim = 2, 2048, 32, 128
q = torch.randn(batch, seqlen, nheads, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(batch, seqlen, nheads, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(batch, seqlen, nheads, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)

# Forward
out = flash_attn_func(q, k, v, causal=True)

# Backward (automatic)
out.sum().backward()

# With sliding window
out = flash_attn_func(q, k, v, window_size=(512, 0))

# With softcapping
out = flash_attn_func(q, k, v, softcap=50.0)

# GQA with packing
k_gqa = torch.randn(batch, seqlen, 8, dim, device='cuda', dtype=torch.bfloat16)
v_gqa = torch.randn(batch, seqlen, 8, dim, device='cuda', dtype=torch.bfloat16)
out = flash_attn_func(q, k_gqa, v_gqa, pack_gqa=True)

# With LSE output
out, lse = flash_attn_func(q, k, v, causal=True, return_lse=True)
```

---

## flash_attn_varlen_func (FA4)

Variable-length attention for batches with different sequence lengths.

### Signature

```python
def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
```

### Parameters

| Parameter | Type | Shape | Default | Description |
|-----------|------|-------|---------|-------------|
| `q` | `Tensor` | `(total_q, num_heads, head_dim)` | required | Concatenated query tokens |
| `k` | `Tensor` | `(total_k, num_heads_kv, head_dim)` or `(num_pages, page_size, num_heads_kv, head_dim)` | required | Key tokens or paged KV |
| `v` | `Tensor` | `(total_k, num_heads_kv, head_dim_v)` or `(num_pages, page_size, num_heads_kv, head_dim_v)` | required | Value tokens or paged KV |
| `cu_seqlens_q` | `Tensor` | `(batch_size + 1,)` int32 | `None` | Cumulative query sequence lengths |
| `cu_seqlens_k` | `Tensor` | `(batch_size + 1,)` int32 | `None` | Cumulative key sequence lengths |
| `max_seqlen_q` | `int` | scalar | `None` | Max query sequence length |
| `max_seqlen_k` | `int` | scalar | `None` | Max key sequence length |
| `min_seqlen_k` | `int` | scalar | `None` | Min KV sequence length (for gather_kv_indices) |
| `seqused_q` | `Tensor` | `(batch_size,)` int32 | `None` | Actual used query lengths |
| `seqused_k` | `Tensor` | `(batch_size,)` int32 | `None` | Actual used key lengths |
| `gather_kv_indices` | `Tensor` | `(total_q, gather_kv_length)` | `None` | TopK gather indices |
| `page_table` | `Tensor` | `(batch_size, max_pages_per_seq)` int32 | `None` | Paged KV block table |
| Others | | | | Same as `flash_attn_func` |

### Example

```python
import torch
from flash_attn.cute import flash_attn_varlen_func

# Variable-length batch: sequences of length [128, 256, 512]
cu_seqlens = torch.tensor([0, 128, 384, 896], dtype=torch.int32, device='cuda')
max_seqlen = 512
total_tokens = 896
nheads, dim = 16, 128

q = torch.randn(total_tokens, nheads, dim, device='cuda', dtype=torch.bfloat16)
k = torch.randn(total_tokens, nheads, dim, device='cuda', dtype=torch.bfloat16)
v = torch.randn(total_tokens, nheads, dim, device='cuda', dtype=torch.bfloat16)

out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_seqlen,
    max_seqlen_k=max_seqlen,
    causal=True,
)
print(out.shape)  # (896, 16, 128)
```

---

## Score Modifiers (score_mod)

FA4 introduces user-defined score modifiers: Python functions decorated with
`@cute.jit` that are injected into the attention kernel at compile time.

### What is a Score Modifier?

A score modifier transforms the attention score matrix `S = QK^T * scale`
before the softmax is applied. It can implement arbitrary transformations
like softcapping, relative position bias, or custom attention patterns.

### Creating a Score Modifier

```python
import cutlass.cute as cute

@cute.jit
def my_score_mod(
    score,           # Current attention score (scalar)
    batch,           # Batch index
    head,            # Head index
    q_idx,           # Query position index
    k_idx,           # Key position index
):
    # Example: Add relative position bias
    rel_pos = q_idx - k_idx
    bias = cute.cast_to[float](rel_pos) * 0.01
    return score + bias
```

### Built-in Softcap Score Modifier

FA4 internally creates a softcap score modifier when `softcap > 0`:

```python
def create_softcap_scoremod(softcap):
    @cute.jit
    def softcap_mod(score, batch, head, q_idx, k_idx):
        return cute.math.tanh(score / softcap) * softcap
    return softcap_mod
```

### Using Score Modifiers

```python
from flash_attn.cute import flash_attn_func

out = flash_attn_func(q, k, v, score_mod=my_score_mod)
```

### Score Modifier with Backward

For backward pass, you can provide a separate backward score modifier:

```python
out = flash_attn_func(
    q, k, v,
    score_mod=my_score_mod,
    score_mod_bwd=my_score_mod_backward,  # Optional: defaults to score_mod
)
```

### Score Modifier with Auxiliary Tensors

Score modifiers can read from auxiliary tensors passed through `aux_tensors`:

```python
@cute.jit
def bias_score_mod(score, batch, head, q_idx, k_idx, bias_tensor):
    # bias_tensor is accessed via aux_tensors
    return score + bias_tensor[batch, head, q_idx, k_idx]

bias = torch.randn(batch, nheads, seqlen, seqlen, device='cuda', dtype=torch.bfloat16)
out = flash_attn_func(q, k, v, score_mod=bias_score_mod, aux_tensors=[bias])
```

### Constraints

- Score modifiers must be decorated with `@cute.jit`
- Not supported on SM80 (Ampere) - raises `NotImplementedError`
- Score modifier and softcap cannot be used together
- The function signature must match the expected pattern

---

## Mask Modifiers (mask_mod)

Mask modifiers define which attention positions should be masked (set to
`-inf` before softmax).

### Creating a Mask Modifier

```python
import cutlass.cute as cute

@cute.jit
def causal_mask(batch, head, q_idx, k_idx):
    # Return True to KEEP the position, False to MASK it
    return q_idx >= k_idx

@cute.jit
def sliding_window_mask(batch, head, q_idx, k_idx):
    window_size = 512
    return (q_idx >= k_idx) and (q_idx - k_idx < window_size)

@cute.jit
def dilated_attention_mask(batch, head, q_idx, k_idx):
    # Only attend to every 4th position
    return (q_idx % 4 == k_idx % 4) and (q_idx >= k_idx)

@cute.jit
def custom_block_mask(batch, head, q_idx, k_idx):
    # Block-diagonal mask: only attend within blocks of size 256
    block_size = 256
    return q_idx // block_size == k_idx // block_size
```

### Using Mask Modifiers

```python
from flash_attn.cute import flash_attn_func

# Custom causal mask
out = flash_attn_func(q, k, v, mask_mod=causal_mask)

# Sliding window via mask modifier
out = flash_attn_func(q, k, v, mask_mod=sliding_window_mask)

# Dilated attention
out = flash_attn_func(q, k, v, mask_mod=dilated_attention_mask)
```

### Mask Modifier with Auxiliary Tensors

```python
@cute.jit
def custom_mask_with_bias(batch, head, q_idx, k_idx, mask_tensor):
    return mask_tensor[batch, head, q_idx, k_idx] > 0

mask = torch.ones(batch, nheads, seqlen, seqlen, device='cuda', dtype=torch.bfloat16)
out = flash_attn_func(q, k, v, mask_mod=custom_mask_with_bias, aux_tensors=[mask])
```

### Constraints

- Mask modifiers must be decorated with `@cute.jit`
- Return `True` to KEEP the position (not masked), `False` to MASK it
- When `mask_mod` is provided, `causal` and `window_size` parameters are ignored
- Not yet supported for varlen sequences (will raise `NotImplementedError`)

---

## Block Sparse Attention

FA4 supports block-sparse attention, which skips computing attention for
entire blocks that are known to be zero.

### BlockSparseTensorsTorch

```python
@dataclass
class BlockSparseTensorsTorch:
    """Container for block sparse attention tensors.

    Attributes:
        mask_block_cnt: (batch, num_heads, num_q_blocks) - Number of non-zero blocks per query block
        mask_block_idx: (batch, num_heads, num_q_blocks, max_blocks_per_q) - Indices of non-zero KV blocks
        full_block_cnt: (batch, num_heads, num_kv_blocks) - Number of query blocks attending to each KV block
        full_block_idx: (batch, num_heads, num_kv_blocks, max_blocks_per_kv) - Indices of query blocks
        dq_write_order: (batch, num_heads, num_kv_blocks, max_blocks_per_kv) - Write order for dQ
        dq_write_order_full: Similar, for full blocks
        block_size: Optional (block_size_q, block_size_k) tuple
        head_dim: Optional, 1 for broadcast across heads
    """
```

### Usage

```python
from flash_attn.cute import flash_attn_func
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

# Create block sparse tensors (example: stride-2 sparse pattern)
# ... construct mask_block_cnt, mask_block_idx, etc.

sparse_tensors = BlockSparseTensorsTorch(
    mask_block_cnt=mask_block_cnt,
    mask_block_idx=mask_block_idx,
    full_block_cnt=full_block_cnt,
    full_block_idx=full_block_idx,
    dq_write_order=dq_write_order,
    dq_write_order_full=dq_write_order_full,
)

out = flash_attn_func(q, k, v, block_sparse_tensors=sparse_tensors)
```

### Constraints

- Block sparsity not supported on SM120
- Block sparsity not supported with SplitKV
- Block sparsity not yet supported for varlen sequences
- Block sparse head dim must be 1 if using pack_gqa

---

## Pack GQA

Pack GQA is an optimization that packs multiple Q heads that share the same
KV head into a single "super-head", increasing the effective tile size for
better Tensor Core utilization.

### Auto-detection

By default, `pack_gqa=None` enables automatic packing when `num_heads > num_heads_kv`:

```python
# Automatic: pack_gqa is True because 32 > 8
out = flash_attn_func(q, k_gqa, v_gqa)  # q: 32 heads, k: 8 heads

# Explicit disable
out = flash_attn_func(q, k_gqa, v_gqa, pack_gqa=False)
```

### When Pack GQA is Disabled Automatically

Pack GQA is automatically disabled when:
1. `num_splits > 1` and not varlen (incompatible with SplitKV)
2. `qv` is provided and `128 % qhead_per_kvhead != 0`
3. Block sparse with non-broadcast head dim

---

## Compilation and Caching

FA4 uses JIT compilation. Kernels are compiled on first use and cached for
subsequent calls with the same configuration.

### Compile Key

The compilation cache key includes:

```python
compile_key = (
    dtype,                          # Data type
    head_dim, head_dim_v,           # Head dimensions
    qhead_per_kvhead,               # GQA ratio
    causal,                         # Causal flag
    score_mod_hash,                 # Score modifier hash
    mask_mod_hash,                  # Mask modifier hash
    use_block_sparsity,             # Block sparse flag
    block_sparse_broadcast_pattern, # Sparse broadcast pattern
    aux_tensor_metadata,            # Auxiliary tensor info
    lse is None,                    # Whether LSE is computed
    cu_seqlens_q is None,           # Varlen Q
    cu_seqlens_k is None,           # Varlen K
    seqused_q is None,              # seqused Q
    seqused_k is None,              # seqused K
    page_table is not None,         # Paged KV
    window_size_left is not None,   # Left window
    window_size_right is not None,  # Right window
    learnable_sink is not None,     # Learnable sink
    q_descale is not None,          # FP8 descale
    k_descale is not None,
    v_descale is not None,
    tile_m, tile_n,                 # Block sizes
    q_stage,                        # Q pipeline stage
    num_threads,                    # Thread count
    is_split_kv,                    # SplitKV flag
    pack_gqa,                       # Pack GQA flag
    arch,                           # GPU architecture
    paged_kv_non_tma,               # Non-TMA paged KV
    use_2cta_instrs,                # 2CTA flag
    q_subtile_factor,               # Block sparse subtile
    mma_pv_is_rs,                   # Register-mapped P*V
    intra_wg_overlap,               # Intra-warp-group overlap
    use_clc_scheduler,              # CLC scheduler
    qv is not None,                 # MLA absorbed
    gather_kv_length,               # TopK gather length
    sparse_kv,                      # Sparse KV flag
    disable_sparse_kv_bitmask,      # Disable sparse bitmask
    log_level,                      # Logging level
)
```

### Cache Levels

1. **In-memory LRU cache**: Per-process Python dictionary. Cleared on exit.
2. **Optional disk cache**: Enabled with `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`.
   Stored at `/tmp/${USER}/flash_attention_cute_dsl_cache/`.

### Cache Utilities

```python
from flash_attn.cute.cache_utils import get_jit_cache

# Get the forward compilation cache
fwd_cache = get_jit_cache("fwd")

# Get the backward preprocessing cache
bwd_pre_cache = get_jit_cache("bwd_pre")
```

---

## Environment Variables (FA4)

### Architecture Selection

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASH_ATTENTION_ARCH` | Auto-detect | Override kernel architecture. E.g., `sm_80`, `sm_90`, `sm_100`. |
| `CUTE_DSL_ARCH` | Auto-detect | Override compilation target architecture. |

### Compilation and Caching

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED` | `0` | Enable persistent disk cache |
| `FLASH_ATTENTION_FAKE_TENSOR` | `0` | Use FakeTensorMode for compilation (no GPU) |

### Debugging and Inspection

| Variable | Default | Description |
|----------|---------|-------------|
| `CUTE_DSL_KEEP_PTX` | None | Set to `1` to keep intermediate PTX files |
| `CUTE_DSL_PTXAS_PATH` | None | Path to custom `ptxas` binary |
| `CUTE_DSL_LINEINFO` | None | Set to `1` to add line info to PTX |
| `CUTE_CUBIN_PATH` | None | Directory to dump CUBIN/SASS files |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `FA_LOG_LEVEL` | `0` | Log level: 0=none, 1=info, 2=debug |

### Scheduling

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASH_ATTENTION_USE_CLC_SCHEDULER` | Varies | Enable CLC persistent scheduling |
| `FLASH_ATTENTION_DISABLE_2CTA` | Varies | Disable 2CTA instructions |

---

## Advanced Parameters

The internal `_flash_attn_fwd` function exposes additional tuning parameters
that are not part of the public API but can be accessed via the internal
interface.

### Tile Sizes

```python
tile_mn: Optional[Tuple[int, int]] = None  # Override (M, N) tile sizes
```

Tile sizes control the block dimensions for the attention kernel:
- `tile_m`: Query block size (rows)
- `tile_n`: Key/Value block size (columns)

Larger tiles use more shared memory but amortize overhead better.

### Thread Configuration

```python
num_threads: int = 384  # SM90/SM100: 384 threads (6 warps)
                         # SM80/SM120: 128 threads (4 warps)
```

### Pipeline Configuration

```python
mma_pv_is_rs: Optional[bool] = None   # Register-mapped P*V accumulation
intra_wg_overlap: Optional[bool] = None  # Overlap within warp group
```

### FwdConfig Parameters

The `FwdConfig` dataclass controls forward kernel behavior:

```python
@dataclass(frozen=True)
class FwdConfig:
    m_block_size: int       # M dimension tile size
    n_block_size: int       # N dimension tile size
    mma_pv_is_rs: bool      # Use register-mapped P*V GEMM
    intra_wg_overlap: bool  # Overlap intra-warp-group operations
```

### BwdConfig Parameters

The `BwdConfig` dataclass controls backward kernel behavior:

```python
@dataclass(frozen=True)
class BwdConfig:
    m_block_size: int        # M dimension tile size
    n_block_size: int        # N dimension tile size
    num_stages_Q: int        # Pipeline stages for Q
    num_stages_dO: int       # Pipeline stages for dO
    num_stages_PdS: int      # Pipeline stages for P*dS
    SdP_swapAB: bool         # Swap A/B for S*dP GEMM
    dKV_swapAB: bool         # Swap A/B for dK/dV GEMM
    dQ_swapAB: bool          # Swap A/B for dQ GEMM
    AtomLayoutMSdP: int      # Atom layout M for S*dP
    AtomLayoutNdKV: int      # Atom layout N for dK/dV
    AtomLayoutMdQ: int       # Atom layout M for dQ
    num_wg: int = 2          # MMA warp groups
    dQ_single_wg: bool = False  # Single warp group for dQ
```

---

## Head Dimension Support

### Supported Head Dimensions by Architecture

| Head Dimension | SM80 | SM90 | SM100/SM110 | SM120 |
|----------------|------|------|-------------|-------|
| 8-64 | Yes | Yes | Yes | Yes |
| 65-96 | Yes | Yes | Yes | No |
| 97-128 | Yes | Yes | Yes | No |
| 129-192 | Yes | Yes | DeepSeek only (192,128) | No |
| 193-256 | Yes | Yes | Yes (2CTA kernel) | No |

### Alignment Requirements

Head dimensions must be divisible by `16 // element_size`:
- FP16/BF16: divisible by 8
- FP8: divisible by 16

### Different V Head Dimension

FA4 supports `head_dim_v != head_dim`:
- SM90: Any combination with both between 8 and 256
- SM100: Standard (8-128, 8-128), or DeepSeek (192, 128), or MLA (64, 512), or hdim=256

---

## GPU Architecture Targeting

### Architecture Detection

FA4 auto-detects the GPU architecture using `torch.cuda.get_device_capability()`.
Override with `FLASH_ATTENTION_ARCH`.

```python
# Auto-detect (default)
major, minor = torch.cuda.get_device_capability()
arch = major * 10 + minor  # e.g., 90 for H100

# Override via environment
FLASH_ATTENTION_ARCH=sm_100 python my_script.py
```

### Kernel Selection by Architecture

| Architecture | Forward Kernel | Backward Kernel | MMA Type |
|-------------|---------------|-----------------|----------|
| SM80 | `FlashAttentionForwardSm80` | `FlashAttentionBackwardSm80` | Ampere HMMA |
| SM86/89 | Same as SM80 | Same as SM80 | Ampere HMMA |
| SM90 | `FlashAttentionForwardSm90` | `FlashAttentionBackwardSm90` | Hopper WGMMA |
| SM100 | `FlashAttentionForwardSm100` | `FlashAttentionBackwardSm100` | Blackwell UMMA |
| SM110 | Same as SM100 | Same as SM100 | Blackwell UMMA |
| SM120 | `FlashAttentionForwardSm120` | `FlashAttentionBackwardSm120` | Ampere HMMA |

### 2CTA Instructions (SM100)

On SM100/SM110, FA4 can use two-CTA cluster instructions when:
- Not causal and not local
- Not using SplitKV
- Not using varlen
- Not using block sparsity
- Paged KV page size is 128 or None
- Head dimension is 128 or 192 (padded to 16-byte alignment)
- V head dimension is 128 (padded)
- Sufficient query blocks exist

The dedicated `BlackwellFusedMultiHeadAttentionForward` kernel is used for
head_dim=256, head_dim_v=256 on SM100.

### Persistent Kernels (SM100)

For non-causal, non-local, non-varlen, non-SplitKV attention, FA4 uses
persistent kernel scheduling (CLC-style). This keeps thread blocks resident
on SMs and assigns work dynamically.

---

## SplitKV Attention (FA4)

FA4 supports SplitKV on SM100/SM110, splitting the K/V sequence across
multiple thread blocks for parallel processing.

### Usage

```python
from flash_attn.cute import flash_attn_func

# Auto SplitKV
out = flash_attn_func(q, k, v, num_splits=0)

# Manual 4 splits
out = flash_attn_func(q, k, v, num_splits=4)
```

### num_splits Heuristic

```python
def num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, max_splits):
    if num_n_blocks <= 4:
        return 1  # No splitting for short sequences
    return min(num_SMs // total_mblocks, max_splits, num_n_blocks)
```

### SplitKV with Different Head Dimensions

When `head_dim != head_dim_v` and `num_splits > 1` on SM100:
- If `num_n_blocks >= 64` and not MLA (head_dim_v != 512): reduce `tile_n` to 64
- Otherwise: disable SplitKV (set `num_splits = 1`)

---

## Paged KV Cache (FA4)

FA4 supports paged KV cache via the `page_table` parameter in
`flash_attn_varlen_func`.

### Usage

```python
from flash_attn.cute import flash_attn_varlen_func

page_size = 128
num_pages = batch * max_pages
k_cache = torch.zeros(num_pages, page_size, nheads_kv, dim, device='cuda', dtype=torch.bfloat16)
v_cache = torch.zeros(num_pages, page_size, nheads_kv, dim, device='cuda', dtype=torch.bfloat16)
page_table = torch.zeros(batch, max_pages, dtype=torch.int32, device='cuda')

out = flash_attn_varlen_func(
    q, k_cache, v_cache,
    cu_seqlens_q=cu_seqlens,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    page_table=page_table,
)
```

### TMA vs Non-TMA Paged KV

When `page_size != tile_n`, FA4 uses a non-TMA path for paged KV loading
(indicated in the compile key). This is slower than the TMA path but
supports arbitrary page sizes.

---

## MLA Weight-Absorbed Attention

FA4 supports DeepSeek-style Multi-head Latent Attention (MLA) via the `qv`
parameter. This is only available on SM100/SM110.

### MLA Absorbed Formula

```
O = softmax(scale * (Q @ K^T + Qv @ V^T)) @ V
```

Where:
- `Q` (q_pe): Position-encoding queries, shape `(batch, seqlen, num_heads, 64)`
- `Qv` (q_nope): Non-position queries, shape `(batch, seqlen, num_heads, 512)`
- `K` (pe_cache): Position-encoding keys
- `V` (kv_cache): Key-value cache

### Constraints

- Only on SM100/SM110
- `head_dim = 64`, `head_dim_v = 512`
- Not supported with local attention, page_table, descale, SplitKV, softcap, or mask_mod

### gather_kv_indices

For topK sparsity with MLA, `gather_kv_indices` selects which KV positions to attend to:

```python
# Shape: (batch, seqlen_q, gather_kv_length) or (total_q, gather_kv_length)
gather_kv_indices = torch.randint(0, seqlen_k, (batch, seqlen_q, 2048), device='cuda', dtype=torch.int32)
# gather_kv_length must be divisible by 256

out = flash_attn_func(
    q, k, v,
    qv=qv,
    gather_kv_indices=gather_kv_indices,
)
```

---

## FP8 Support (FA4)

FA4 supports FP8 data types on SM100/SM110.

### Supported Types

| Type | Output Type | Notes |
|------|-------------|-------|
| `torch.float8_e4m3fn` | `torch.bfloat16` | E4M3 format |
| `torch.float8_e5m2` | `torch.bfloat16` | E5M2 format |

### FP8 with Descale

```python
# FP8 inputs with descale factors
q_fp8 = torch.randn(batch, seqlen, nheads, dim, device='cuda', dtype=torch.float8_e4m3fn)
k_fp8 = torch.randn(batch, seqlen, nheads, dim, device='cuda', dtype=torch.float8_e4m3fn)
v_fp8 = torch.randn(batch, seqlen, nheads, dim, device='cuda', dtype=torch.float8_e4m3fn)

q_descale = torch.ones(batch, nheads_kv, device='cuda', dtype=torch.float32)
k_descale = torch.ones(batch, nheads_kv, device='cuda', dtype=torch.float32)
v_descale = torch.ones(batch, nheads_kv, device='cuda', dtype=torch.float32)

out = flash_attn_func(
    q_fp8, k_fp8, v_fp8,
    # Descale passed via internal interface
)
```

### FP8 Constraints

- SM100 only (not SM90)
- Forward only (backward not supported)
- Cannot mix with gradient computation

---

## Autograd Integration

FA4 provides full autograd support via custom `torch.autograd.Function`
classes.

### FlashAttnFunc (Internal)

Handles the forward/backward for `flash_attn_func`:
- Forward: calls `_flash_attn_fwd` with appropriate kernel selection
- Backward: preprocesses (dP sum, LSE conversion), runs backward kernel,
  postprocesses (accumulation to output dtype)

### FlashAttnVarlenFunc (Internal)

Same as `FlashAttnFunc` but for variable-length sequences:
- Handles `cu_seqlens`, `seqused`, `page_table`
- Different LSE shapes: `(num_heads, total_q)` instead of `(batch, num_heads, seqlen)`

### Gradient Flow

```python
q.requires_grad_(True)
k.requires_grad_(True)
v.requires_grad_(True)

out = flash_attn_func(q, k, v, causal=True)
loss = out.sum()
loss.backward()

assert q.grad is not None
assert k.grad is not None
assert v.grad is not None
```

---

## Testing

### Standard Tests

```bash
# Run all FA4 tests
pytest tests/cute/test_flash_attn.py -x

# Run specific test
pytest tests/cute/test_flash_attn.py -k "test_flash_attn_output" -x

# Run varlen tests
pytest tests/cute/test_flash_attn_varlen.py

# Run mask modifier tests
pytest tests/cute/test_mask_mod.py

# Run score modifier tests
pytest tests/cute/test_score_mod.py

# Run block sparse tests
pytest tests/cute/test_block_sparsity.py
```

### Fast Two-Pass Testing

Separates compilation (parallel, no GPU) from execution:

```bash
# Pass 1: Compile all kernels in parallel using FakeTensorMode
FLASH_ATTENTION_FAKE_TENSOR=1 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
    pytest -n 64 -x tests/cute/test_flash_attn.py

# Pass 2: Run tests using cached kernels
FLASH_ATTENTION_FAKE_TENSOR=0 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
    pytest -x tests/cute/test_flash_attn.py
```

### Test Parametrization

Tests are parametrized over:
- dtype: fp16, bf16
- head dimension: 64, 96, 128
- sequence length: various (512, 1024, 2048, etc.)
- causal/non-causal
- MHA/GQA/MQA

---

## Common Patterns

### Custom Attention with Score Modifier

```python
import cutlass.cute as cute
from flash_attn.cute import flash_attn_func

@cute.jit
def relative_position_bias(score, batch, head, q_idx, k_idx):
    # Add learned relative position bias
    distance = q_idx - k_idx
    # This is a simplified example; in practice, you'd look up from a table
    bias = -cute.abs(cute.cast_to[float](distance)) * 0.001
    return score + bias

out = flash_attn_func(q, k, v, score_mod=relative_position_bias)
```

### Block Sparse Attention Pattern

```python
from flash_attn.cute import flash_attn_func
from flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    normalize_block_sparse_config,
)

# Define your sparse pattern (e.g., local + strided blocks)
# ... create mask_block_cnt, mask_block_idx, etc.

sparse = BlockSparseTensorsTorch(
    mask_block_cnt=cnt,
    mask_block_idx=idx,
    full_block_cnt=full_cnt,
    full_block_idx=full_idx,
    dq_write_order=dq_order,
    dq_write_order_full=dq_order_full,
)

out = flash_attn_func(q, k, v, block_sparse_tensors=sparse)
```

### Mixed Precision with FP8

```python
from flash_attn.cute import flash_attn_func

# FP8 forward, BF16 output
q = some_quantize_fn(q_bf16)  # -> float8_e4m3fn
k = some_quantize_fn(k_bf16)
v = some_quantize_fn(v_bf16)

out = flash_attn_func(q, k, v, causal=True)
# out.dtype == torch.bfloat16
```

### Multi-Architecture Deployment

```python
import torch
from flash_attn.cute import flash_attn_func

# Works on A100, H100, B200, etc.
# Kernel is automatically selected based on GPU architecture
q = torch.randn(2, 1024, 16, 128, device='cuda', dtype=torch.bfloat16)
k = torch.randn(2, 1024, 16, 128, device='cuda', dtype=torch.bfloat16)
v = torch.randn(2, 1024, 16, 128, device='cuda', dtype=torch.bfloat16)

out = flash_attn_func(q, k, v, causal=True)
```
