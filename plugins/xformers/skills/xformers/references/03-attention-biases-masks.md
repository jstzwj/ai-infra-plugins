# 03 - Attention Biases and Masks

## Overview

xFormers provides a rich set of attention bias classes that control which positions in the attention matrix are attended to. These are passed via the `attn_bias` parameter of `memory_efficient_attention`.

**Source**: `xformers/ops/fmha/attn_bias.py` (re-exported from `mslk.attention.fmha`)

## Base Classes

### `AttentionBias`

Base class for all attention biases. Attention biases are no longer `torch.Tensor` subclasses (changed in 0.0.32).

### `AttentionOpBase`

Base class for attention operators. Subclassed by:
- `AttentionFwOpBase` - Forward-only operators
- `AttentionBwOpBase` - Backward operators
- `AttentionOp` - Tuple of (FwOp, BwOp)

## Bias Types

### `LowerTriangularMask`

Standard causal mask. Each query position can only attend to positions <= its own index.

```python
import xformers.ops as xops

# Causal attention
out = xops.memory_efficient_attention(q, k, v, attn_bias=xops.LowerTriangularMask())
```

**Properties:**
- No GPU memory allocated (created on CPU since 0.0.29)
- Supported by all backends
- Most optimized path

### `LowerTriangularMaskWithTensorBias`

Causal mask combined with an additive bias tensor.

```python
bias = torch.randn(B, H, M, N, device="cuda", dtype=torch.float16)
attn_bias = xops.LowerTriangularMaskWithTensorBias(bias)
out = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias)
```

### `BlockDiagonalMask`

Block-diagonal attention mask for variable-length sequences packed into a single batch.

```python
from xformers.ops import BlockDiagonalMask

# Create from sequence lengths
mask = BlockDiagonalMask.from_seqlens(
    q_seqlen=[4, 3, 5],    # query sequence lengths per block
    kv_seqlen=[6, 4, 8],   # key/value sequence lengths per block
)

# Must be on the same device as other inputs (since 0.0.27)
mask = mask.to("cuda")

# Inputs concatenated along sequence dimension
q = torch.randn(1, 12, H, K, device="cuda", dtype=torch.float16)  # 4+3+5
k = torch.randn(1, 18, H, K, device="cuda", dtype=torch.float16)  # 6+4+8
v = torch.randn(1, 18, H, K, device="cuda", dtype=torch.float16)

out = xops.memory_efficient_attention(q, k, v, attn_bias=mask)
```

**Factory methods:**
- `from_seqlens(q_seqlen, kv_seqlen=None)` - Create from sequence length lists
- `from_tensor_sizes(*sizes)` - Create from tensor dimensions

### `BlockDiagonalCausalMask`

Block-diagonal mask with causal masking within each block.

```python
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

mask = BlockDiagonalCausalMask.from_seqlens(
    q_seqlen=[4, 3, 5],
    kv_seqlen=[6, 4, 8],
)
```

Each block is independently causal - within a block, position i can only attend to positions j <= i.

### `BlockDiagonalCausalWithOffsetPaddedKeysMask`

Advanced mask for heterogeneous batching with padded KV-cache, commonly used for inference with variable-length sequences.

```python
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalWithOffsetPaddedKeysMask

# Used with rope_padded for inference
attn_bias = BlockDiagonalCausalWithOffsetPaddedKeysMask(
    q_seqlen=q_seqinfo,
    k_seqinfo=k_seqinfo,
)
```

This is the mask used by `rope_padded` for heterogeneous batch inference in LLaMA-style models.

**Attributes:**
- `q_seqinfo` - Query sequence info (seqstart, seqlen, max_seqlen)
- `k_seqinfo` - Key sequence info (seqstart, seqlen, max_seqlen)

### `PagedBlockDiagonalPaddedKeysMask`

Paged attention mask for efficient KV-cache management during inference.

```python
from xformers.ops.fmha.attn_bias import PagedBlockDiagonalPaddedKeysMask

paged_bias = PagedBlockDiagonalPaddedKeysMask(
    q_seqlens=q_seqlens,
    kv_seqlens=kv_seqlens,
    page_table=page_table,  # [batch, max_num_pages]
)
```

Used for paged attention where KV-cache is stored in non-contiguous pages. This avoids memory fragmentation and enables efficient memory sharing across sequences (e.g., for beam search).

### `PagedBlockDiagonalGappyKeysMask`

Gappy attention bias for sequences with gaps in the KV-cache. Added in xFormers 0.0.30.

## Sequence Info Structure

Many biases use a sequence info structure that describes the layout of packed sequences:

```python
class SeqLenInfo:
    seqstart: torch.Tensor  # [batch+1] - cumulative sequence start offsets
    seqlen: torch.Tensor    # [batch] - individual sequence lengths
    max_seqlen: int         # maximum sequence length in the batch

    @property
    def seqstart_py(self) -> List[int]:
        """Python list of sequence starts"""
```

## Attention Bias for ROCm

### CK-compatible masks

The Composable Kernel backend supports:
- `LowerTriangularMask` - causal attention
- `BlockDiagonalMask` - block-diagonal attention
- Custom attention patterns via CK-specific implementations

## Usage Patterns

### Variable-Length Sequences

```python
# Pack multiple sequences of different lengths into one batch
seq_lengths = [32, 64, 16, 128]
total_q = sum(seq_lengths)
total_kv = sum([s * 2 for s in seq_lengths])  # assume KV is 2x longer

q = torch.randn(1, total_q, H, K, device="cuda", dtype=torch.float16)
k = torch.randn(1, total_kv, H, K, device="cuda", dtype=torch.float16)
v = torch.randn(1, total_kv, H, K, device="cuda", dtype=torch.float16)

mask = BlockDiagonalMask.from_seqlens(
    q_seqlen=seq_lengths,
    kv_seqlen=[s * 2 for s in seq_lengths],
).to("cuda")

out = xops.memory_efficient_attention(q, k, v, attn_bias=mask)
```

### Inference with KV-Cache

```python
# Initial prefill
prefill_mask = BlockDiagonalCausalMask.from_seqlens(
    q_seqlen=[prompt_len],
    kv_seqlen=[prompt_len],
)
prefill_out = xops.memory_efficient_attention(q_prefill, k_cache, v_cache, attn_bias=prefill_mask)

# Decode step
decode_mask = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
    q_seqlen=[1],
    kv_seqlen=[current_pos + 1],
)
decode_out = xops.memory_efficient_attention(q_step, k_cache, v_cache, attn_bias=decode_mask)
```

## Internal Dispatch

When an attention bias is passed to `memory_efficient_attention`:

1. The dispatch module checks which backends support the specific bias type
2. Each backend's `supports()` method checks:
   - Bias type compatibility
   - Data type support (fp16, bf16, fp32)
   - Hardware capability
   - Head dimension constraints
3. The best compatible backend is selected

## Changes from Previous Versions

- **0.0.32**: Attention biases are no longer `torch.Tensor` subclasses
- **0.0.29**: `LowerTriangularMask` no longer creates a CUDA tensor
- **0.0.27**: Attention biases must be on the same device as other input tensors
- **0.0.27**: Biases constructed on CUDA by default when available
