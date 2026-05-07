# 06 - RoPE with Padded KV-Cache

## Overview

`rope_padded` applies Rotary Position Embeddings (RoPE) and manages KV-cache emplacement for heterogeneous batch inference. It's designed for inference in models like LLaMA that use RoPE for positional encoding.

**Source**: `xformers/ops/rope_padded.py`
**Kernel**: `xformers/ops/_triton/rope_padded_kernels.py`

## Mathematical Definition

RoPE applies position-dependent rotations to query and key vectors:

```
For position m and dimension d:
q_rotated[m, 2d]   = q[m, 2d]   * cos(m * theta_d) - q[m, 2d+1] * sin(m * theta_d)
q_rotated[m, 2d+1] = q[m, 2d+1] * cos(m * theta_d) + q[m, 2d]   * sin(m * theta_d)
```

Where `theta_d = 1 / (base^(2d/dim))` and `base` is typically 10000.

## API Reference

### `rope_padded`

```python
from xformers.ops import rope_padded

out_q = rope_padded(
    xq: torch.Tensor,              # Queries [1, slen, n_heads, dim] or [1, slen, n_groups, n_heads, dim]
    xk: torch.Tensor,              # Keys [1, slen, n_kv_heads, dim]
    xv: torch.Tensor,              # Values [1, slen, n_kv_heads, dim]
    cache_k: torch.Tensor,         # Key cache [1, cache_len, n_kv_heads, dim] - MODIFIED IN PLACE
    cache_v: torch.Tensor,         # Value cache [1, cache_len, n_kv_heads, dim] - MODIFIED IN PLACE
    attn_bias: BlockDiagonalCausalWithOffsetPaddedKeysMask,  # Cache layout
    *,
    theta: float = 10000.0,        # RoPE base frequency
    linear_scale: float = 1.0,     # Sequence ID scaling factor
    use_dynamic_scaling: bool = False,  # Enable YaRN-style dynamic scaling
    dynamic_old_context_len: float = 8192.0,
    dynamic_scale_factor: float = 16.0,
    dynamic_low_freq_factor: float = 1.0,
    dynamic_high_freq_factor: float = 32.0,
    out_q: Optional[torch.Tensor] = None,  # Pre-allocated output
    first_seqpos: Optional[torch.Tensor] = None,  # [logical_bsz] start positions
    seqpos: Optional[torch.Tensor] = None,  # [slen] per-query positions
    adjacents: bool = True,        # Feature layout (True=LLaMA, False=HuggingFace)
    internal_dtype: str = "",      # "f32" or "f64" for precision control
) -> torch.Tensor                   # RoPE'd queries [1, slen, n_heads, dim]
```

## What It Does

This function performs the following in a single fused kernel:

1. **Applies RoPE to queries** (`xq`) and returns the result
2. **Applies RoPE to keys** (`xk`) and writes them into the correct position in `cache_k`
3. **Copies values** (`xv`) into the correct position in `cache_v`

After calling `rope_padded`, you can immediately call `memory_efficient_attention`:

```python
out_q = rope_padded(xq, xk, xv, cache_k, cache_v, attn_bias)
attn_out = xops.memory_efficient_attention(out_q, cache_k, cache_v, attn_bias=attn_bias)
```

## Parameters Explained

### Core Parameters

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `xq` | `[1, slen, n_heads, dim]` | Query tensor (batch dim must be 1) |
| `xk` | `[1, slen, n_kv_heads, dim]` | Key tensor |
| `xv` | `[1, slen, n_kv_heads, dim]` | Value tensor |
| `cache_k` | `[1, cache_len, n_kv_heads, dim]` | Key cache (modified in place) |
| `cache_v` | `[1, cache_len, n_kv_heads, dim]` | Value cache (modified in place) |
| `attn_bias` | `BlockDiagonalCausalWithOffsetPaddedKeysMask` | Describes cache layout |

### RoPE Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `theta` | 10000.0 | Base frequency for RoPE |
| `linear_scale` | 1.0 | Divide all sequence indices by this |
| `adjacents` | True | True = adjacent pairs (LLaMA), False = split halves (HuggingFace) |

### Dynamic Scaling (YaRN)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_dynamic_scaling` | False | Enable YaRN-style scaling |
| `dynamic_old_context_len` | 8192.0 | Original context length |
| `dynamic_scale_factor` | 16.0 | Scale factor for context extension |
| `dynamic_low_freq_factor` | 1.0 | Low frequency cutoff |
| `dynamic_high_freq_factor` | 32.0 | High frequency cutoff |

### Position Control

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `first_seqpos` | `[logical_bsz]` | Start position of each batch element's cache |
| `seqpos` | `[slen]` | Per-query sequence position |
| `out_q` | Same as xq | Pre-allocated output tensor |

### GQA Support

For Grouped-Query Attention with `n_groups > 1`:

```python
# xq shape: [1, slen, n_groups, n_heads, dim]
# xk shape: [1, slen, n_groups, n_kv_heads, dim]
# xv shape: [1, slen, n_groups, n_kv_heads, dim]
# cache_k shape: [1, cache_len, n_groups, n_kv_heads, dim]
# cache_v shape: [1, cache_len, n_groups, n_kv_heads, dim]
```

## Usage Examples

### Basic Inference Step

```python
import torch
from xformers.ops import rope_padded, memory_efficient_attention
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalWithOffsetPaddedKeysMask

# Initialize cache
cache_k = torch.zeros(1, max_seq_len, n_kv_heads, dim, device="cuda", dtype=torch.float16)
cache_v = torch.zeros(1, max_seq_len, n_kv_heads, dim, device="cuda", dtype=torch.float16)

# Create attention bias for heterogeneous batch
attn_bias = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
    q_seqlen=[1, 1, 1],  # one new token per sequence
    kv_seqlen=[100, 200, 50],  # current cache lengths
).to("cuda")

# Apply RoPE and update cache
xq = torch.randn(1, 3, n_heads, dim, device="cuda", dtype=torch.float16)
xk = torch.randn(1, 3, n_kv_heads, dim, device="cuda", dtype=torch.float16)
xv = torch.randn(1, 3, n_kv_heads, dim, device="cuda", dtype=torch.float16)

out_q = rope_padded(xq, xk, xv, cache_k, cache_v, attn_bias)
attn_out = memory_efficient_attention(out_q, cache_k, cache_v, attn_bias=attn_bias)
```

### With YaRN Dynamic Scaling

```python
out_q = rope_padded(
    xq, xk, xv, cache_k, cache_v, attn_bias,
    use_dynamic_scaling=True,
    dynamic_old_context_len=8192,
    dynamic_scale_factor=16.0,
)
```

### With Custom Positions

```python
# Specify starting positions for each batch element
first_seqpos = torch.tensor([0, 100, 200], device="cuda", dtype=torch.int32)
out_q = rope_padded(xq, xk, xv, cache_k, cache_v, attn_bias, first_seqpos=first_seqpos)
```

## Feature Layout

The `adjacents` parameter controls how features are organized:

### Adjacent (LLaMA style, adjacents=True)
```
[x_0, x_1, x_2, x_3, ...] -> pairs: (x_0, x_1), (x_2, x_3), ...
```
Real and imaginary parts of each rotation are adjacent.

### Split (HuggingFace style, adjacents=False)
```
[x_0, x_1, ..., x_d/2, x_d/2+1, ..., x_d]
```
First half are real parts, second half are imaginary parts.

## Implementation Details

1. **Fused Triton Kernel**: All operations (RoPE on Q, RoPE on K, cache emplacement) happen in a single kernel launch.

2. **Heterogeneous batching**: Multiple sequences of different lengths are packed into a single batch. The `attn_bias` parameter describes the layout.

3. **Block size heuristic**: The kernel block size is computed based on the dimension:
   ```python
   MAX_FUSED_SIZE = 65536 // element_size
   BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(dim))
   BLOCK_SIZE = max(BLOCK_SIZE, 128)
   BLOCK_SIZE = min(BLOCK_SIZE, 4096)
   ```

4. **Warp count**: `num_warps = min(max(BLOCK_SIZE // 256, 1), 8)`

## Constraints

1. **No gradient support** - Inference only
2. **Batch size must be 1** - Sequences are concatenated along dim 1
3. **Contiguous heads** - Each head's data must be contiguous (`stride[-1] == 1`)
4. **Same device** - `attn_bias` must be on the same device as tensors
5. **Requires Triton** - GPU compute capability >= 8.0
