# 09 - Model Parallel Linear Layers

## Overview

xFormers provides `ColumnParallelLinear` and `RowParallelLinear` as near-drop-in replacements for FairScale/Megatron model parallel layers. They support fusing communication and computation for sequence parallelism.

**Source**: `xformers/ops/modpar_layers.py`
**Depends on**: `seqpar.py`, `differentiable_collectives.py`

## API Reference

### `ColumnParallelLinear`

```python
from xformers.ops import ColumnParallelLinear

layer = ColumnParallelLinear(
    in_features: int,                          # Input dimension
    out_features: List[int],                   # Output dimensions (multiple outputs)
    *,
    process_group: torch.distributed.ProcessGroup,
    bias: bool = True,                         # Must be False in current impl
    gather_output: bool = True,                # Must be False in current impl
    init_method: Callable = torch.nn.init.xavier_normal_,
    sequence_parallel: bool = False,           # Enable sequence parallelism
    fuse_sequence_parallel: bool = True,       # Fuse comm+compute
)
```

Splits output dimension across ranks. Each rank holds a shard of the weight matrix.

**Key constraints:**
- `out_features` must be a list (supports multiple output projections like Q/K/V)
- `bias` must be `False`
- `gather_output` must be `False`
- All dimensions must be divisible by `process_group.size()`

**Forward:**
```python
outputs = layer(input_)  # Returns List[Tensor]
# With sequence_parallel=False: copy input, matmul each weight
# With sequence_parallel=True: fused all-gather + matmul
```

### `RowParallelLinear`

```python
from xformers.ops import RowParallelLinear

layer = RowParallelLinear(
    in_features: int,                    # Input dimension (global)
    out_features: int,                   # Output dimension
    *,
    process_group: torch.distributed.ProcessGroup,
    bias: bool = True,                   # Must be False
    input_is_parallel: bool = False,     # Must be True
    init_method: Callable = torch.nn.init.xavier_normal_,
    sequence_parallel: bool = False,
    fuse_sequence_parallel: bool = True,
)
```

Splits input dimension across ranks. Each rank holds a shard of the weight matrix.

**Key constraints:**
- `bias` must be `False`
- `input_is_parallel` must be `True`
- `in_features` must be divisible by `process_group.size()`

**Forward:**
```python
output = layer(input_)  # Returns Tensor
# With sequence_parallel=False: matmul + all-reduce
# With sequence_parallel=True: fused matmul + reduce-scatter
```

## Weight Initialization

Both layers use a specific initialization strategy that ensures different ranks get different values:

```python
def _init_2d_weight(weight, init_method, process_group, partition_dim):
    # Create full (unpartitioned) weight
    # Apply init_method to full weight
    # Copy the appropriate slice for this rank
    # This "breaks the symmetry" across ranks
```

This ensures that:
1. Different ranks start with different values (symmetry breaking)
2. The initialization is consistent regardless of model parallelism degree

## Sequence Parallelism Integration

### ColumnParallelLinear with SP

```python
# When sequence_parallel=True:
# Forward:
outputs = sequence_parallel_leading_matmul(
    input_, [w.t() for w in weights],
    fuse=fuse_sequence_parallel,
    process_group=process_group,
)
```

This fuses the all-gather of the input with the matrix multiplication, making the communication effectively free.

### RowParallelLinear with SP

```python
# When sequence_parallel=True:
# Forward:
output = sequence_parallel_trailing_matmul(
    input_, weight.t(),
    fuse=fuse_sequence_parallel,
    process_group=process_group,
)
```

This fuses the matrix multiplication with the reduce-scatter of the output.

## Differentiable Collectives

The model parallel layers depend on differentiable collective operations from `differentiable_collectives.py`:

```python
# All-gather along first dimension
gathered = gather_along_first_dim(tensor, process_group)

# Async all-gather
gathered, handle = gather_along_first_dim_async(tensor, process_group)

# Reduce-scatter along first dimension
scattered = reduce_scatter_along_first_dim(tensor, process_group)

# Async reduce-scatter
scattered, handle = reduce_scatter_along_first_dim_async(tensor, process_group)

# Copy to model parallel region (identity forward, all-reduce backward)
tensor = copy_to_model_parallel_region(tensor, process_group)

# Reduce from model parallel region (all-reduce forward, identity backward)
tensor = reduce_from_model_parallel_region(tensor, process_group)
```

## Sequence Parallel Matmul Operations

### `sequence_parallel_leading_matmul`

```python
from xformers.ops import sequence_parallel_leading_matmul

outputs = sequence_parallel_leading_matmul(
    x: torch.Tensor,              # Scattered input
    ws: List[torch.Tensor],       # Weight matrices
    *,
    fuse: bool,                   # Use fused ops
    process_group: ProcessGroup,
) -> List[torch.Tensor]
```

Registered as `xformers_python::sequence_parallel_leading_matmul_fwd` with custom autograd.

**Backward** computes gradients for both input and all weights, with proper communication:
- Gradient w.r.t. input: fused reduce-scatter + matmul
- Gradient w.r.t. weights: fused all-gather + outer product

### `sequence_parallel_trailing_matmul`

```python
from xformers.ops import sequence_parallel_trailing_matmul

output = sequence_parallel_trailing_matmul(
    x: torch.Tensor,              # Gathered input
    w: torch.Tensor,              # Weight matrix
    *,
    fuse: bool,
    process_group: ProcessGroup,
) -> torch.Tensor
```

Registered as `xformers_python::sequence_parallel_trailing_matmul_fwd` with custom autograd.

## Usage Example: Transformer with Sequence + Model Parallelism

```python
import torch
import torch.distributed as dist
import torch.nn as nn
from xformers.ops import (
    ColumnParallelLinear,
    RowParallelLinear,
    memory_efficient_attention,
)

class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mp_group, sp_group):
        super().__init__()
        self.mp_group = mp_group
        self.sp_group = sp_group
        mp_size = mp_group.size()

        # Column parallel: QKV projection
        self.qkv_proj = ColumnParallelLinear(
            dim,
            [dim // mp_size] * 3,  # 3 outputs: Q, K, V
            process_group=mp_group,
            sequence_parallel=(sp_group is not None),
        )

        # Row parallel: output projection
        self.out_proj = RowParallelLinear(
            dim,
            dim,
            process_group=mp_group,
            input_is_parallel=True,
            sequence_parallel=(sp_group is not None),
        )

    def forward(self, x):
        # Fused all-gather + QKV projection
        q, k, v = self.qkv_proj(x)

        # Attention
        attn_out = memory_efficient_attention(q, k, v)

        # Fused output projection + reduce-scatter
        return self.out_proj(attn_out)
```

## Comparison with FairScale/Megatron

| Feature | xFormers | FairScale/Megatron |
|---------|----------|-------------------|
| Column parallel | Yes | Yes |
| Row parallel | Yes | Yes |
| Sequence parallel | Yes | Yes |
| Fused comm+compute | Yes (NVLink) | No |
| Multiple outputs | Yes (list) | Single output |
| Bias support | No (current) | Yes |
| torch.compile | Partial | No |

## Constraints

1. **bias=False** required for both layers in current implementation
2. **ColumnParallelLinear**: `gather_output=False` required
3. **RowParallelLinear**: `input_is_parallel=True` required
4. **Weight dimensions** must be divisible by model parallel size
5. **NVLink** required for fused fast path (falls back to NCCL otherwise)
