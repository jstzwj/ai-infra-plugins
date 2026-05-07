# 17 - Triton Kernels

## Overview

xFormers uses Triton for several custom GPU kernels that provide optimized implementations of key operations. These kernels are conditionally loaded based on GPU capability (compute capability >= 8.0, i.e., A100+).

**Source**: `xformers/ops/_triton/`

## Kernel Files

### `rmsnorm_kernels.py`

Triton kernels for RMS normalization.

**Functions:**
- `_rms_norm_forward(x, weight, eps)` - Forward pass
- `_rms_norm_add_forward(x, y, weight, eps)` - Fused add + normalize

**Implementation:**
- Computes sum of squares along the last dimension
- Normalizes by `1 / sqrt(mean(x^2) + eps)`
- Multiplies by learnable weight

### `rope_padded_kernels.py`

Triton kernel for RoPE with padded KV-cache.

**Function:**
- `_rope_padded_kernel` - Main kernel

**Parameters:**
- Input tensors: xq, xk, xv, out_q, cache_k, cache_v
- Layout info: seqstartq, seqstartk, seqlenk
- RoPE config: theta, linear_scale, dynamic scaling params
- Position info: first_seqpos, seqpos
- Dimension info: k_start, v_start, n_groups, dim
- Stride info for all tensors
- Block size and num_warps

### `k_scaled_index_add.py`

Triton kernels for scaled index addition.

**Functions:**
- `scaled_index_add_fwd(output, index, source, scaling, alpha)` - Forward
- `scaled_index_add_bwd(grad_output, grad_source, grad_scaling, source, scaling, index, alpha)` - Backward

### `k_index_select_cat.py`

Triton kernels for index select and concatenation.

**Functions:**
- `index_select_cat_fwd(output, source, index)` - Forward
- `index_select_cat_bwd(grad_source, index, grad_output)` - Backward

### `tiled_matmul_kernels.py`

Triton kernel for tiled matrix multiplication.

**Function:**
- `_launch_triton_matmul(a, b, out, ms, ns, ks)` - Launch the tiled matmul kernel

**Supports up to 3x3 tile grids.**

### `matmul_perf_model.py`

Performance model for matmul operations. Used to estimate whether a Triton kernel would be faster than PyTorch's built-in matmul for a given problem size.

## Triton Availability Check

```python
from xformers import _is_triton_available

if _is_triton_available():
    # Triton kernels are available
    pass
```

**Detection logic:**
1. If `XFORMERS_ENABLE_TRITON=1`: force enable
2. If CUDA is not available: disable
3. If `XFORMERS_FORCE_DISABLE_TRITON=1`: disable
4. If compute capability < 8.0: disable (V100 and older not supported)
5. Try importing triton: if successful, enable

## Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `XFORMERS_ENABLE_TRITON` | `"0"` | Force-enable Triton kernels |
| `XFORMERS_FORCE_DISABLE_TRITON` | `"0"` | Force-disable Triton kernels |
| `XFORMERS_TILED_MATMUL_ENABLE_TRITON` | `"1"` | Enable Triton tiled matmul |

## Kernel Design Patterns

### Block Size Selection

Most kernels use adaptive block sizing:

```python
MAX_FUSED_SIZE = 65536 // element_size  # 64KB limit
BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(dim))
BLOCK_SIZE = max(BLOCK_SIZE, 128)
BLOCK_SIZE = min(BLOCK_SIZE, 4096)
```

### Warp Count

```python
num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
```

### Grid Launch

```python
kernel[grid_size](
    *args,
    BLOCK_SIZE=BLOCK_SIZE,
    num_warps=num_warps,
)
```

## Conditional Import Pattern

The `_triton/__init__.py` uses conditional imports:

```python
from xformers import _is_triton_available

if _is_triton_available():
    from .k_scaled_index_add import scaled_index_add_fwd, scaled_index_add_bwd
    from .k_index_select_cat import index_select_cat_fwd, index_select_cat_bwd
else:
    scaled_index_add_fwd = None
    scaled_index_add_bwd = None
    index_select_cat_fwd = None
    index_select_cat_bwd = None
```

This ensures the module can always be imported, even without Triton.

## Performance Characteristics

### RMSNorm
- ~2-3x faster than PyTorch on A100
- Avoids materializing intermediate tensors

### RoPE Padded
- Fused RoPE + cache emplacement in single kernel
- ~3-5x faster than separate operations

### Tiled Matmul
- ~2.4x faster than separate matmuls for small tiles
- Avoids wave quantization

### Index Operations
- ~1.5-2x faster than PyTorch equivalents
- Better memory access patterns for gather/scatter

## FMHA Triton Kernels

The FMHA module also has Triton kernels in `xformers/ops/fmha/_triton/`:

- `splitk_kernels.py` - Split-K attention implementation for single-query decoding

This provides an alternative to Flash Attention for specific attention patterns.
