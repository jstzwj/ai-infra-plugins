# 11 - Indexing Operations

## Overview

xFormers provides optimized indexing operations with custom Triton kernels for efficient memory access patterns. These are particularly useful for variable-length sequence handling in Transformer models.

**Source**: `xformers/ops/indexing.py`
**Kernels**: `xformers/ops/_triton/k_scaled_index_add.py`, `xformers/ops/_triton/k_index_select_cat.py`

## API Reference

### `scaled_index_add`

```python
from xformers.ops import scaled_index_add

result = scaled_index_add(
    input: torch.Tensor,          # [B, M, D] - MODIFIED IN PLACE
    index: torch.Tensor,          # [Bi] - int64 indices
    source: torch.Tensor,         # [Bi, M, D]
    scaling: Optional[torch.Tensor] = None,  # [D] per-feature scaling
    alpha: float = 1.0,           # Global scaling factor
) -> torch.Tensor                 # Modified input
```

Performs in-place scaled index addition. Equivalent to:

```python
torch.index_add(input, dim=0, source=scaling * source, index=index, alpha=alpha)
```

**Key features:**
- In-place forward pass (input is modified)
- Supports per-feature scaling via `scaling` parameter
- Fully differentiable (with `torch.autograd.Function`)
- Requires Triton

**Constraints:**
- Indices must be unique
- Max index < size of dim 0 of input
- Forward is done in-place

**Gradient computation:**
- `grad_input` = `grad_output` (same as input, since operation is in-place)
- `grad_source` = scaled gradient indexed from `grad_output`
- `grad_scaling` = (if provided) sum of element-wise products

### `index_select_cat`

```python
from xformers.ops import index_select_cat

result = index_select_cat(
    sources: Sequence[torch.Tensor],  # List of [S_i, D_i] tensors
    indices: Sequence[torch.Tensor],  # List of [I_i] int64 index tensors
) -> torch.Tensor                     # [sum(I_i * D_i)] flattened result
```

Concatenates indexed selections from multiple sources. Equivalent to:

```python
torch.cat([s[i.long()].flatten() for s, i in zip(sources, indices)], dim=0)
```

**Example:**
```python
# Given:
sources[0] of shape [100, 64]
indices[0] of shape [10]
sources[1] of shape [200, 32]
indices[1] of shape [5]

# Returns:
result of shape [10 * 64 + 5 * 32] = [800]
```

**Key features:**
- Efficiently concatenates multiple index_select operations
- Fully differentiable (with `torch.autograd.Function`)
- Requires Triton

## Registered Operators

| Operator | Category | Description |
|----------|----------|-------------|
| `ScaledIndexAddFw` | indexing | Forward scaled index add (Triton) |
| `ScaledIndexAddBw` | indexing | Backward scaled index add (Triton) |
| `IndexSelect` | indexing | Forward index select + cat (Triton) |

## Usage Examples

### Scaled Index Add for Sequence Packing

```python
import torch
from xformers.ops import scaled_index_add

# Buffer for packed sequences
buffer = torch.zeros(4, 128, 64, device="cuda", dtype=torch.float16)

# New data for sequences at positions 1 and 3
source = torch.randn(2, 128, 64, device="cuda", dtype=torch.float16)
index = torch.tensor([1, 3], device="cuda", dtype=torch.int64)

# Optional per-feature scaling
scaling = torch.ones(64, device="cuda", dtype=torch.float16)

# Add scaled source to buffer at specified indices
result = scaled_index_add(buffer, index, source, scaling=scaling, alpha=0.5)
```

### Index Select Cat for Variable-Length Sequences

```python
import torch
from xformers.ops import index_select_cat

# Two different-sized embedding tables
embeddings_a = torch.randn(1000, 64, device="cuda", dtype=torch.float16)
embeddings_b = torch.randn(500, 32, device="cuda", dtype=torch.float16)

# Select specific entries from each
indices_a = torch.tensor([0, 5, 10, 15], device="cuda", dtype=torch.int64)
indices_b = torch.tensor([2, 7], device="cuda", dtype=torch.int64)

# Concatenate the selections
result = index_select_cat(
    sources=[embeddings_a, embeddings_b],
    indices=[indices_a, indices_b],
)
# Shape: [4 * 64 + 2 * 32] = [320]
```

## Implementation Details

Both operations use Triton kernels for:

1. **scaled_index_add**: Parallel writes to the output tensor with proper atomic operations for the scaling factor gradient.

2. **index_select_cat**: Parallel reads from source tensors, concatenated into a single output buffer.

The Triton implementations avoid the overhead of launching multiple separate PyTorch operations and enable better memory access patterns.
