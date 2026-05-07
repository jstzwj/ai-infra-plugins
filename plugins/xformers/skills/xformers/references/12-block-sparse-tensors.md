# 12 - Block-Sparse Tensors

## Overview

xFormers provides a `BlockSparseTensor` class for efficient block-sparse matrix operations. This is useful for implementing sparse attention patterns where only certain blocks of the attention matrix are computed.

**Source**: `xformers/sparse/blocksparse_tensor.py`, `xformers/sparse/utils.py`

**Note**: This module is deprecated in favor of PyTorch's native blocksparse ops, but remains available.

## API Reference

### `BlockSparseTensor`

```python
from xformers.sparse import BlockSparseTensor

bst = BlockSparseTensor(
    values: torch.Tensor,  # [B, nnz, block_size, block_size]
    layout: torch.Tensor,  # [heads, h_blocks, w_blocks] binary mask
)
```

Creates a block-sparse tensor where:
- `values` contains the non-zero blocks
- `layout` describes which blocks are non-zero (binary mask)
- `block_size` must be >= 16 (minimum for Triton)

**Shape**: The resulting tensor has logical shape `(B, heads, block_size * h_blocks, block_size * w_blocks)`.

### Supported Operations

#### Matrix Multiplication (bmm)

```python
# Sparse @ Dense -> Dense
result = sparse_tensor @ dense_tensor
# or
result = torch.bmm(sparse_tensor, dense_tensor)
```

Uses `_spmm` for sparse-dense matrix multiplication.

#### Softmax

```python
result = torch.softmax(sparse_tensor, dim=-1)
# or
result = torch.nn.functional.softmax(sparse_tensor, dim=-1)
```

Applied block-wise with proper normalization across sparse blocks.

#### Masked Matmul (SDDMM)

```python
from xformers.ops import masked_matmul

# Dense @ Dense -> Sparse (masked by sparse pattern)
result = masked_matmul(a, b, sparse_mask)
```

Uses `_sddmm` (Sample-Dense-Dense Matrix Multiply) to compute only the non-zero blocks.

#### To Dense

```python
dense = sparse_tensor.to_dense()
```

Converts back to a dense tensor by placing non-zero blocks at their positions.

#### Other Operations

- `torch.dropout(sparse_tensor, p)` - Apply dropout to values
- `sparse_tensor.to(device)` - Move to device
- `sparse_tensor.detach()` - Detach from graph
- `torch.equal(sparse1, sparse2)` - Equality check
- `sparse_tensor.copy_(other)` - Copy from another sparse tensor

## Utility Functions

### Format Conversions (`sparse/utils.py`)

```python
from xformers.sparse.utils import (
    _coo_to_csr,       # COO -> CSR format
    _csr_to_coo,       # CSR -> COO format
    _transpose,        # Transpose sparse matrix
    _dense_to_sparse,  # Dense -> sparse (2D)
    _dense3d_to_sparse, # Dense -> sparse (3D, batched)
)
```

### Layout ↔ Pattern Conversion

```python
from xformers.components.attention.attention_patterns import (
    pattern_to_layout,
    layout_to_pattern,
)

# Pattern (dense boolean mask) -> Layout (block-level mask)
layout = pattern_to_layout(mask, block_size=16)

# Layout -> Pattern
pattern = layout_to_pattern(layout, block_size=16)
```

## Usage Examples

### Sparse Attention

```python
import torch
from xformers.sparse import BlockSparseTensor
from xformers.components.attention.attention_patterns import local_2d_pattern

# Create local attention pattern
H, W = 32, 32
block_size = 16
pattern = local_2d_pattern(H, W, distance=2)

# Convert to block-sparse layout
layout = pattern_to_layout(pattern, block_size)

# Create sparse attention matrix
B = 1
nnz = layout.sum().item()
values = torch.randn(B, nnz, block_size, block_size, device="cuda", dtype=torch.float16)
sparse_attn = BlockSparseTensor(values, layout)

# Apply softmax
sparse_attn = torch.softmax(sparse_attn, dim=-1)

# Multiply with values
values_dense = torch.randn(B, 1, H * block_size, block_size, device="cuda", dtype=torch.float16)
output = sparse_attn @ values_dense
```

### Block-Sparse from Dense

```python
# Create dense tensor
dense = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.float16)

# Define which blocks to keep
layout = torch.zeros(4, 4, 4, dtype=torch.long, device="cuda")
layout[:, 0, 0] = 1  # Keep top-left block of each head
layout[:, 1, 1] = 1  # Keep diagonal block

# Extract non-zero blocks
block_size = 16
nnz = layout.sum().item()
values = torch.empty(1, nnz, block_size, block_size, device="cuda", dtype=torch.float16)
for idx, (h, i, j) in enumerate(zip(*layout.nonzero(as_tuple=True))):
    values[:, idx] = dense[:, h, i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]

sparse = BlockSparseTensor(values, layout)
```

## Implementation Details

### `_spmm` (Sparse Matrix Multiply)

Performs sparse-dense matrix multiplication by:
1. Reshaping dense matrix to block format
2. Computing block-level matrix multiplications
3. Aggregating results using `index_add_`

### `_softmax`

Block-sparse softmax:
1. Computes `logsumexp` within each block
2. Aggregates normalization across blocks sharing the same row
3. Applies exponential normalization

### `_sddmm` (Sampled Dense-Dense Matrix Multiply)

Computes only the blocks specified by the layout:
1. Reshapes both inputs to block format
2. Uses `einsum` for block-level computation
3. Returns only non-zero blocks

## Limitations

1. **Deprecated**: PyTorch now has native blocksparse ops
2. **Block size >= 16**: Minimum block size for Triton compatibility
3. **Square blocks**: Block size must be the same in both dimensions
4. **Limited operations**: Not all torch operations are supported
5. **torch.compile**: Not compatible with torch.compile

## Relationship to FMHA

The block-sparse tensor support is complementary to FMHA's built-in sparsity support. FMHA handles sparse attention patterns through attention biases (like `BlockDiagonalMask`), while `BlockSparseTensor` provides general-purpose sparse matrix operations.
