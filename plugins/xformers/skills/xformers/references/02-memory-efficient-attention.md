# 02 - Memory-Efficient Attention (FMHA)

## Overview

The Flash Memory-Efficient Attention (FMHA) module is the crown jewel of xFormers. It provides exact (not approximated) attention computation that is up to 10x faster than standard PyTorch attention by avoiding materialization of the full attention matrix.

**Source**: `xformers/ops/fmha/__init__.py` (re-exports from `mslk.attention.fmha`)

## Main API

### `memory_efficient_attention`

```python
xformers.ops.memory_efficient_attention(
    query: torch.Tensor,         # [B, M, H, K]
    key: torch.Tensor,           # [B, N, H, K]
    value: torch.Tensor,         # [B, N, H, V]
    attn_bias: Optional[AttentionBias] = None,
    p: float = 0.0,              # dropout probability
    scale: Optional[float] = None,  # override default sqrt(K)
    op: Optional[AttentionOp] = None,  # force specific backend
) -> torch.Tensor:               # [B, M, H, V]
```

Computes `softmax(Q @ K^T / sqrt(d)) @ V` without materializing the full `[B, H, M, N]` attention matrix.

**Key features:**
- Exact computation (not an approximation)
- Supports float16, bfloat16, float32
- Multiple attention patterns via `attn_bias`
- Automatic backend selection
- Supports multi-query attention (MQA) and grouped-query attention (GQA)

### `memory_efficient_attention_forward`

```python
xformers.ops.memory_efficient_attention_forward(
    query, key, value, attn_bias, p=0.0, scale=None, op=None
) -> Tuple[torch.Tensor, torch.Tensor]
```

Forward-only variant, returns `(output, logsumexp)`.

### `memory_efficient_attention_backward`

```python
xformers.ops.memory_efficient_attention_backward(
    grad, query, key, value, lse, output, attn_bias, p=0.0, scale=None, op=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Returns `(grad_query, grad_key, grad_value)`.

### `memory_efficient_attention_forward_requires_grad`

```python
xformers.ops.memory_efficient_attention_forward_requires_grad(
    query, key, value, attn_bias, p=0.0, scale=None, op=None
) -> Tuple[torch.Tensor, torch.Tensor]
```

Returns `(output, logsumexp)`. Used internally for autograd.

### `memory_efficient_attention_partial`

Computes partial attention, used for merge operations.

### `merge_attentions`

```python
xformers.ops.merge_attentions(
    attn_outputs: List[torch.Tensor],  # partial attention outputs
    lses: List[torch.Tensor],          # logsumexp values
) -> torch.Tensor
```

Merges multiple partial attention computations. Useful for gradient checkpointing where attention is computed in chunks.

## Tensor Layouts

The standard layout is `[B, M, H, K]` where:
- `B` = batch size
- `M` = query sequence length
- `N` = key/value sequence length
- `H` = number of attention heads
- `K` = key dimension per head
- `V` = value dimension per head

**Multi-Query Attention (MQA)**: Set `H_kv = 1` for key/value tensors with stride broadcast.
**Grouped-Query Attention (GQA)**: Set `H_kv < H_q` for key/value tensors.

## Backends

### `MemoryEfficientAttentionFlashAttentionOp`

- **Backend**: Flash Attention v2 (external package)
- **Hardware**: A100+ (Ampere and later)
- **Best for**: General attention, training, inference
- **Supports**: causal, local attention, block-diagonal masks, paged attention

### `MemoryEfficientAttentionCutlassOp`

- **Backend**: CUTLASS GEMM kernels
- **Hardware**: A100+
- **Best for**: Attention patterns not supported by Flash Attention
- **Supports**: Full range of attention biases, including arbitrary tensor bias

### `MemoryEfficientAttentionCutlassFwdFlashBwOp`

- Hybrid: CUTLASS forward + Flash backward
- Useful when Flash doesn't support a forward pattern but can handle the backward

### `MemoryEfficientAttentionCutlassBlackwellOp`

- **Backend**: CUTLASS optimized for Blackwell GPUs (B100, B200)
- **Hardware**: Blackwell (compute capability 10.0+)
- **Added in**: xFormers 0.0.33

### `MemoryEfficientAttentionCkOp`

- **Backend**: Composable Kernel (CK) - AMD ROCm
- **Hardware**: AMD GPUs with ROCm 7.1+
- **ROCm only**

### `MemoryEfficientAttentionSplitKCkOp`

- **Backend**: Split-K CK implementation
- **Hardware**: AMD GPUs
- **Optimized for**: Single-query decoding (N=1)

## Backend Dispatch

The dispatch mechanism (`fmha.dispatch`) automatically selects the best backend:

1. **Input validation**: Checks tensor shapes, dtypes, device compatibility
2. **Capability check**: Each backend's `supports()` method validates it can handle the inputs
3. **Priority ordering**: Prefers Flash Attention > CUTLASS > CK
4. **Op override**: The `op` parameter forces a specific backend

```python
# Automatic dispatch
output = xops.memory_efficient_attention(q, k, v)

# Force Flash Attention
output = xops.memory_efficient_attention(q, k, v, op=xops.MemoryEfficientAttentionFlashAttentionOp)

# Force specific FW/BW ops
output = xops.memory_efficient_attention(
    q, k, v,
    op=(xops.MemoryEfficientAttentionCutlassOp.FwOp, xops.MemoryEfficientAttentionFlashAttentionOp.BwOp)
)
```

## Usage Examples

### Basic Attention

```python
import torch
import xformers.ops as xops

B, M, N, H, K = 1, 128, 256, 8, 64
q = torch.randn(B, M, H, K, device="cuda", dtype=torch.float16)
k = torch.randn(B, N, H, K, device="cuda", dtype=torch.float16)
v = torch.randn(B, N, H, K, device="cuda", dtype=torch.float16)

out = xops.memory_efficient_attention(q, k, v)
# Shape: [1, 128, 8, 64]
```

### Causal Attention

```python
out = xops.memory_efficient_attention(q, k, v, attn_bias=xops.LowerTriangularMask())
```

### Multi-Query Attention (MQA)

```python
q = torch.randn(B, M, H, K, device="cuda", dtype=torch.float16)  # [1, 128, 8, 64]
k = torch.randn(B, N, 1, K, device="cuda", dtype=torch.float16)  # [1, 256, 1, 64] - 1 KV head
v = torch.randn(B, N, 1, K, device="cuda", dtype=torch.float16)  # [1, 256, 1, 64]

out = xops.memory_efficient_attention(q, k, v)
```

### Grouped-Query Attention (GQA)

```python
q = torch.randn(B, M, 8, K, device="cuda", dtype=torch.float16)  # 8 query heads
k = torch.randn(B, N, 2, K, device="cuda", dtype=torch.float16)  # 2 KV heads
v = torch.randn(B, N, 2, K, device="cuda", dtype=torch.float16)

out = xops.memory_efficient_attention(q, k, v)
```

### Custom Scale

```python
out = xops.memory_efficient_attention(q, k, v, scale=1.0 / 32.0)
```

### With Block-Diagonal Mask

```python
from xformers.ops import memory_efficient_attention, BlockDiagonalMask

# Create block-diagonal mask for variable-length sequences
block_mask = BlockDiagonalMask.from_seqlens(
    q_seqlen=[4, 3, 5],    # query sequence lengths
    kv_seqlen=[6, 4, 8],   # key/value sequence lengths
)

# Inputs must be concatenated along the sequence dimension
q = torch.randn(1, 12, H, K, device="cuda", dtype=torch.float16)  # 4+3+5=12
k = torch.randn(1, 18, H, K, device="cuda", dtype=torch.float16)  # 6+4+8=18
v = torch.randn(1, 18, H, K, device="cuda", dtype=torch.float16)

out = memory_efficient_attention(q, k, v, attn_bias=block_mask)
```

## Performance Considerations

1. **Head dimension**: K=64 is the most optimized. K=128 is also well-supported.
2. **Sequence length**: Performance gains are most significant for long sequences (>= 512).
3. **Precision**: Use `torch.float16` or `torch.bfloat16` for best performance.
4. **Batch size**: Larger batch sizes amortize kernel launch overhead.
5. **Backend selection**: Flash Attention is fastest for standard patterns; CUTLASS supports more bias types.

## FP8 Support

FMHA supports FP8 tensor-wise quantization for inference:

```python
# Quantize to FP8
q_fp8 = q.to(torch.float8_e4m3fn)
k_fp8 = k.to(torch.float8_e4m3fn)
v_fp8 = v.to(torch.float8_e4m3fn)

# Use with scales
out = memory_efficient_attention(q_fp8, k_fp8, v_fp8, scale_q=scale_q, scale_k=scale_k)
```

## Paged Attention

Supports paged KV-cache for efficient inference:

```python
from xformers.ops.fmha.attn_bias import PagedBlockDiagonalPaddedKeysMask

paged_bias = PagedBlockDiagonalPaddedKeysMask(
    q_seqlens=[4, 3, 5],
    kv_seqlens=[6, 4, 8],
    page_table=page_table_tensor,
)
out = memory_efficient_attention(q, cache_k, cache_v, attn_bias=paged_bias)
```

## torch.compile Support

FMHA supports `torch.compile` with certain limitations:

- Flash Attention backend is fully supported
- Bias support: `LowerTriangularMask`, `LowerTriangularMaskWithTensorBias`, `BlockDiagonalMask`
- Use explicit op specification for best results:

```python
@torch.compile
def forward(self, q, k, v):
    return xops.memory_efficient_attention(
        q, k, v,
        attn_bias=xops.LowerTriangularMask(),
        op=(xops.MemoryEfficientAttentionFlashAttentionOp, None),
    )
```

## Common Issues

1. **"No supported backend found"**: Check GPU compute capability (needs >= 8.0), tensor dtype (needs fp16/bf16), and that CUDA extensions are built.

2. **Shape mismatches**: Ensure K dimension matches between Q and K, and that H dimensions are compatible for MQA/GQA.

3. **Memory errors**: FMHA is designed to reduce memory usage, but very long sequences may still OOM. Consider sequence parallelism.

4. **Triton not available**: Set `XFORMERS_ENABLE_TRITON=1` or ensure GPU has compute capability >= 8.0.
