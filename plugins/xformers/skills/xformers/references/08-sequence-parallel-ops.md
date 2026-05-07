# 08 - Sequence Parallel Fused Operations

## Overview

xFormers provides fused communication + computation operators for sequence parallelism that overlap all-gather/reduce-scatter with matrix multiplications over NVLink. These operators make the communication effectively free by hiding it behind computation.

**Source**: `xformers/ops/sequence_parallel_fused_ops.py`
**Inspired by**: https://arxiv.org/abs/2302.05442

## Key Idea

Instead of:
1. All-gather (wait for communication)
2. Matmul (compute)

We do:
1. Start sending our data to neighbor
2. Start computing with local data
3. As remote data arrives, compute with it
4. Overlap continues across all ranks

This achieves the same result but hides communication latency.

## API Reference

### `fused_allgather_and_linear`

```python
from xformers.ops import fused_allgather_and_linear

output = fused_allgather_and_linear(
    scattered_input: torch.Tensor,     # Local shard [shard_batch, ..., in_features]
    weight: Union[torch.Tensor, List[torch.Tensor]],  # Linear weight(s)
    *,
    group: dist.ProcessGroup,
    out: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    timeout_s: int = 60 * 60,
    scale_scattered_input: Optional[torch.Tensor] = None,  # FP8 scale
    scale_weight: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,  # FP8 scale
    out_dtype: Optional[torch.dtype] = None,  # Output dtype for FP8
) -> Union[torch.Tensor, List[torch.Tensor]]
```

Equivalent to:
```python
gathered_input = all_gather(scattered_input)  # [world_size * shard_batch, ..., in_features]
output = F.linear(gathered_input, weight)
```

But with communication overlapped with computation.

**Supports multiple weights** (e.g., QKV fusion):
```python
q, k, v = fused_allgather_and_linear(
    scattered_input,
    weight=[wq, wk, wv],  # Multiple weights
    group=process_group,
)
```

**FP8 support** for tensor-wise quantized operations:
```python
output = fused_allgather_and_linear(
    scattered_input_fp8,
    weight_fp8,
    group=process_group,
    scale_scattered_input=scale_input,
    scale_weight=scale_w,
    out_dtype=torch.float16,
)
```

### `fused_linear_and_reducescatter`

```python
from xformers.ops import fused_linear_and_reducescatter

scattered_output = fused_linear_and_reducescatter(
    gathered_input: torch.Tensor,       # Full input [batch, ..., in_features]
    weight: Union[torch.Tensor, List[torch.Tensor]],
    *,
    group: dist.ProcessGroup,
    out: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    timeout_s: int = 60 * 60,
    scale_gathered_input: Optional[torch.Tensor] = None,
    scale_weight: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]
```

Equivalent to:
```python
gathered_output = F.linear(gathered_input, weight)
scattered_output = reduce_scatter(gathered_output)
```

### `fused_allgather_and_anything`

```python
fused_allgather_and_anything(
    scattered_inputs: List[torch.Tensor],
    my_matmul: Callable[[List[torch.Tensor], int, Callable], None],
    *,
    group: dist.ProcessGroup,
    timeout_s: int = 60 * 60,
) -> None
```

Generic fused all-gather + custom computation. The `my_matmul` callback receives:
- List of input tensors for the current rank
- The rank index
- A stream factory for async execution

### `fused_anything_and_reducescatter`

```python
fused_anything_and_reducescatter(
    my_matmul: Callable[[List[torch.Tensor], int, Callable], None],
    scattered_outputs: List[torch.Tensor],
    *,
    group: dist.ProcessGroup,
    timeout_s: int = 60 * 60,
) -> None
```

Generic custom computation + fused reduce-scatter.

## Internal Architecture

### `_FusedSequenceParallel`

The core class that manages the communication ring:

```python
class _FusedSequenceParallel:
    def __init__(self, device, group):
        self.my_rank = group.rank()
        self.world_size = group.size()
        self.second_stream = torch.cuda.Stream()
        self.memcpy_stream = torch.cuda.Stream(priority=-1)  # Prioritized
        self.compute_wait_stream = torch.cuda.Stream(priority=-1)
        self.memcpy_wait_stream = torch.cuda.Stream(priority=-1)
```

**Stream management:**
- `current_stream` / `second_stream` - Alternating streams for computation
- `memcpy_stream` - High-priority stream for data transfers
- `compute_wait_stream` / `memcpy_wait_stream` - Wait kernel streams

### Communication Protocol

Uses PyTorch's `SymmetricMemory` for NVLink-optimized communication:

1. **Signal-based synchronization**: Uses `put_signal`/`wait_signal` instead of IPC events
2. **Persistent staging buffers**: Buffers persist across calls, sized to the maximum needed
3. **Ring communication**: Data flows in a ring pattern across ranks

### NVLink Detection

```python
def _can_ranks_communicate_all_to_all_over_nvlink(group):
    return group.size() <= 8  # Simplified heuristic
```

Can be disabled with `DISABLE_FUSED_SEQUENCE_PARALLEL=1`.

### Fallback Path

When NVLink is not available (or world_size == 1), falls back to standard NCCL operations:
- `dist.all_gather_into_tensor` for all-gather
- `dist.reduce_scatter_tensor` for reduce-scatter

## Usage Patterns

### Sequence Parallel Attention

```python
import torch
import torch.distributed as dist
from xformers.ops import fused_allgather_and_linear, fused_linear_and_reducescatter

class SequenceParallelAttention(nn.Module):
    def __init__(self, dim, process_group):
        self.wq = nn.Parameter(torch.randn(dim, dim))
        self.wk = nn.Parameter(torch.randn(dim, dim))
        self.wv = nn.Parameter(torch.randn(dim, dim))
        self.wo = nn.Parameter(torch.randn(dim, dim))
        self.process_group = process_group

    def forward(self, x):
        # Fused all-gather + QKV projection
        q, k, v = fused_allgather_and_linear(
            x, [self.wq, self.wk, self.wv], group=self.process_group
        )

        # Attention computation (on gathered sequence)
        attn_out = memory_efficient_attention(q, k, v)

        # Fused output projection + reduce-scatter
        return fused_linear_and_reducescatter(
            attn_out, self.wo, group=self.process_group
        )
```

### FP8 Quantized Communication

```python
# Quantize input to FP8
scale_input = compute_scale(scattered_input)
input_fp8 = quantize_to_fp8(scattered_input, scale_input)

scale_weight = compute_scale(weight)
weight_fp8 = quantize_to_fp8(weight, scale_weight)

output = fused_allgather_and_linear(
    input_fp8,
    weight_fp8,
    group=process_group,
    scale_scattered_input=scale_input,
    scale_weight=scale_weight,
    out_dtype=torch.float16,
)
```

## Performance Considerations

1. **NVLink required**: The fast path only works with NVLink connectivity between all ranks
2. **Buffer sizing**: Staging buffers are sized to the maximum input across calls
3. **Stream priority**: Memory copy stream has high priority to avoid starvation
4. **Wave quantization**: The fused approach also helps avoid wave quantization in the matmul

## Integration with Model Parallel Layers

These fused ops are used internally by `ColumnParallelLinear` and `RowParallelLinear` when `sequence_parallel=True` and `fuse_sequence_parallel=True`.
