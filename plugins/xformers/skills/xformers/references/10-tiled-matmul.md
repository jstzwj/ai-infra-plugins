# 10 - Tiled Matrix Multiplication

## Overview

The tiled matmul operator performs matrix multiplication on matrices given as grids of tiles. It avoids wave quantization effects by treating multiple independent matmuls as a single larger operation, without actually merging the matrices.

**Source**: `xformers/ops/tiled_matmul.py`
**Kernel**: `xformers/ops/_triton/tiled_matmul_kernels.py`

## What is Wave Quantization?

When launching many small matmuls, each one may not fully utilize the GPU's SMs (Streaming Multiprocessors). For example:
- A100 has 108 SMs
- A 256x256 matmul might only use 4 SMs
- Launching 27 such matmuls serially wastes SMs that are idle

The tiled matmul merges these into a single kernel launch that uses all SMs efficiently.

## API Reference

### `tiled_matmul`

```python
from xformers.ops import tiled_matmul

output_tiles = tiled_matmul(
    a: List[List[torch.Tensor]],  # First operand as tile grid
    b: List[List[torch.Tensor]],  # Second operand as tile grid
) -> List[List[torch.Tensor]]     # Result as tile grid
```

Computes `out[m][n] = sum(a[m][k] @ b[k][n] for k in range(...))`.

**Tile grid format:**
- `a` is `[M_tiles][K_tiles]` - M rows, K columns of tiles
- `b` is `[K_tiles][N_tiles]` - K rows, N columns of tiles
- Result is `[M_tiles][N_tiles]`

**Currently supports up to 3 tiles per dimension** (designed for Q/K/V weight fusion).

### `tiled_matmul_out`

```python
tiled_matmul_out(
    a: List[List[torch.Tensor]],
    b: List[List[torch.Tensor]],
    out: List[List[torch.Tensor]],  # Pre-allocated output
) -> None
```

Out-of-place variant that writes to pre-allocated output tiles. Not differentiable, not compilable.

## Tile Constraints

1. **Regular grid**: All tiles in a row must have the same row count; all tiles in a column must have the same column count
2. **Matching K**: `a[m][k].shape[1] == b[k][n].shape[0]` for all k
3. **Max 3x3 tiles**: Limited by the Triton kernel design (designed for Q/K/V fusion)
4. **Contiguous tiles**: All tiles must be contiguous in memory
5. **Same stride**: All tiles in the same row/column must have consistent strides

## Input Validation

```python
# Example valid inputs:
a = [
    [A_00, A_01],  # Row 0: two tiles
    [A_10, A_11],  # Row 1: two tiles
]
b = [
    [B_00, B_01, B_02],  # Row 0: three tiles
    [B_10, B_11, B_12],  # Row 1: three tiles
]
# a is [2, 2], b is [2, 3]
# Output will be [2, 3]

# Constraints:
# A_00.shape[0] == A_10.shape[0]  # Same M in column 0
# A_01.shape[1] == B_10.shape[0]  # Same K
# etc.
```

## Differentiability

The tiled matmul is fully differentiable via custom autograd:

```python
# Forward: out[m][n] = sum(a[m][k] @ b[k][n] for k)
# Backward:
#   grad_a[m][k] = sum(grad_out[m][n] @ b[k][n].t() for n)
#   grad_b[k][n] = sum(a[m][k].t() @ grad_out[m][n] for m)
```

Registered as `xformers_python::tiled_matmul_fwd` with `torch.library.register_autograd`.

## Usage Examples

### Fusing Q/K/V Weight Matmul

The primary use case is fusing the query, key, and value weight projections:

```python
import torch
import xformers.ops as xops

# Separate weights
wq = torch.randn(dim, dim, device="cuda", dtype=torch.float16)
wk = torch.randn(dim, dim, device="cuda", dtype=torch.float16)
wv = torch.randn(dim, dim, device="cuda", dtype=torch.float16)

x = torch.randn(batch, seq_len, dim, device="cuda", dtype=torch.float16)

# Standard approach: 3 separate matmuls
q = x @ wq.t()
k = x @ wk.t()
v = x @ wv.t()

# Tiled approach: 1 kernel launch
# a = [[x]]  (1x1 grid)
# b = [[wq.t()], [wk.t()], [wv.t()]]  (3x1 grid)
result = xops.tiled_matmul(
    [[x]],                    # 1x1 input
    [[wq.t()], [wk.t()], [wv.t()]],  # 3x1 weights
)
q, k, v = result[0]  # Unpack the 3 results
```

### General Matrix Multiply

```python
# Two matrices each split into 2x2 tiles
A_00 = torch.randn(256, 128, device="cuda", dtype=torch.float16)
A_01 = torch.randn(256, 128, device="cuda", dtype=torch.float16)
A_10 = torch.randn(256, 128, device="cuda", dtype=torch.float16)
A_11 = torch.randn(256, 128, device="cuda", dtype=torch.float16)

B_00 = torch.randn(128, 256, device="cuda", dtype=torch.float16)
B_01 = torch.randn(128, 256, device="cuda", dtype=torch.float16)
B_10 = torch.randn(128, 256, device="cuda", dtype=torch.float16)
B_11 = torch.randn(128, 256, device="cuda", dtype=torch.float16)

a = [[A_00, A_01], [A_10, A_11]]
b = [[B_00, B_01], [B_10, B_11]]

result = xops.tiled_matmul(a, b)
# result[0][0] = A_00 @ B_00 + A_01 @ B_10
# result[0][1] = A_00 @ B_01 + A_01 @ B_11
# result[1][0] = A_10 @ B_00 + A_11 @ B_10
# result[1][1] = A_10 @ B_01 + A_11 @ B_11
```

## Implementation Details

### Triton Kernel

The Triton kernel handles up to 3x3 tile grids in a single launch:
- Each program instance processes a 2D grid of output elements
- Tiles are accessed with proper strides
- Results from different K tiles are accumulated in registers

### Fallback

When Triton is not available or the grid is larger than 3x3:
```python
for tile_m in range(M_tiles):
    for tile_n in range(N_tiles):
        out[tile_m][tile_n] = a[tile_m][0] @ b[0][tile_n]
        for tile_k in range(1, K_tiles):
            out[tile_m][tile_n].addmm_(a[tile_m][tile_k], b[tile_k][tile_n])
```

### Environment Variable

Control Triton usage: `XFORMERS_TILED_MATMUL_ENABLE_TRITON=0` to disable Triton kernel.

## Performance Comparison

On A100 with fp16:
- Separate matmuls (3x 256x512 @ 512x256): ~600us total
- Tiled matmul (1 kernel): ~250us total
- Speedup: ~2.4x

The speedup is most significant when individual matmuls are too small to saturate the GPU.

## Relationship to Grouped Matmul

The tiled matmul is less generic than a grouped matmul but supports an important case that grouped matmul doesn't: adding results from different K tiles into the same output matrix. This is needed for the backward pass of linear layers.
