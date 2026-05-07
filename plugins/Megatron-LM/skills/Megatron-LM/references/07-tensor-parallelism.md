# Tensor Parallelism Reference

This document provides a detailed technical reference for Megatron-LM's tensor parallelism
implementation, covering the parallel linear layers, communication primitives, sequence
parallelism, and process group initialization.

## Table of Contents

- [Overview](#overview)
- [Process Group Initialization (parallel_state.py)](#process-group-initialization)
- [ColumnParallelLinear](#columnparallellinear)
- [RowParallelLinear](#rowparallellinear)
- [VocabParallelEmbedding](#vocabparallelembedding)
- [Communication Primitives (mappings.py)](#communication-primitives)
- [Sequence Parallelism](#sequence-parallelism)
- [Communication Overlap (tp_comm_overlap)](#communication-overlap)
- [Memory Savings Analysis](#memory-savings-analysis)
- [Performance Implications](#performance-implications)

---

## Overview

Tensor parallelism (TP) in Megatron-LM partitions individual weight tensors across multiple
GPUs within the same node. The key idea is that a linear layer `Y = XA + b` can be
parallelized by splitting matrix `A` along its columns (ColumnParallelLinear) or rows
(RowParallelLinear). The canonical Megatron-LM transformer uses a pair of these layers:

```
[ColumnParallelLinear] -> [Activation] -> [RowParallelLinear]
      (QKV / MLP fc1)                        (Attention out / MLP fc2)
```

This pairing ensures that only two collective communications are needed per transformer
layer block: one AllReduce (or ReduceScatter + AllGather with sequence parallelism).

---

## Process Group Initialization

### `initialize_model_parallel()` in `megatron/core/parallel_state.py`

The `initialize_model_parallel` function creates all necessary process groups. It uses a
`RankGenerator` to compute orthogonal rank groups based on the parallelism dimensions
(tensor, pipeline, data, context, expert) and a user-specified ordering.

```python
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    order: str = "tp-cp-ep-dp-pp",
    ...
) -> None:
```

### Process Groups Created

| Group Variable                      | Description                                         |
|-------------------------------------|-----------------------------------------------------|
| `_TENSOR_MODEL_PARALLEL_GROUP`      | Ranks within a single TP group                      |
| `_PIPELINE_MODEL_PARALLEL_GROUP`    | Ranks within a single PP group                      |
| `_DATA_PARALLEL_GROUP`              | Ranks within a single DP group (no CP)              |
| `_DATA_PARALLEL_GROUP_WITH_CP`      | Ranks within DP+CP combined group                   |
| `_CONTEXT_PARALLEL_GROUP`           | Ranks within a single CP group                      |
| `_MODEL_PARALLEL_GROUP`             | Combined TP+PP group                                |
| `_EMBEDDING_GROUP`                  | First and last pipeline stages for embeddings       |
| `_TENSOR_AND_DATA_PARALLEL_GROUP`   | Combined TP+DP group (for FP8 amax reduction)      |
| `_TENSOR_AND_CONTEXT_PARALLEL_GROUP`| Combined TP+CP group                                |

### RankGenerator and Group Ordering

The `RankGenerator` class uses a configurable `order` parameter (default `"tp-cp-ep-dp-pp"`)
to determine how ranks are mapped to parallel groups. For a given total world size and the
sizes of each parallelism dimension, it generates the list of rank lists for each group type.

```python
# Example: 16 GPUs with TP=2, PP=2, DP=4
# Order: tp-cp-ep-dp-pp
# TP groups:  [g0,g1], [g2,g3], [g4,g5], ..., [g14,g15]
# PP groups:  [g0,g4,g8,g12], [g1,g5,g9,g13], ...
# DP groups:  [g0,g2], [g1,g3], [g4,g6], ...
```

### Accessor Functions

```python
get_tensor_model_parallel_group()       # TP process group
get_tensor_model_parallel_rank()        # Local rank within TP group
get_tensor_model_parallel_world_size()  # TP world size
get_pipeline_model_parallel_group()     # PP process group
get_data_parallel_group(with_context_parallel=False)
get_global_memory_buffer()              # Shared memory buffer for collectives
```

---

## ColumnParallelLinear

**Source:** `megatron/core/tensor_parallel/layers.py`

### Purpose

Splits the weight matrix along the column (output) dimension. Each TP rank holds a
vertical slice of the weight matrix, computing a partial output that is a subset of the
full output features.

### Constructor Parameters

```python
class ColumnParallelLinear(torch.nn.Module):
    def __init__(
        self,
        input_size: int,                # First dimension of weight matrix A
        output_size: int,               # Second dimension of weight matrix A (total)
        *,
        config: ModelParallelConfig,     # Model parallel configuration
        init_method: Callable,           # Weight initialization method
        bias: bool = True,               # Whether to include bias
        gather_output: bool = False,     # All-gather output across TP ranks
        stride: int = 1,                 # Stride for strided initialization
        keep_master_weight_for_test: bool = False,
        skip_bias_add: bool = False,     # Defer bias addition for fusion
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer: Optional[List[torch.Tensor]] = None,
        grad_output_buffer: Optional[List[torch.Tensor]] = None,
        is_expert: bool = False,         # MoE expert layer flag
        tp_comm_buffer_name: Optional[str] = None,
        disable_grad_reduce: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
```

### Weight Splitting

The weight shape is `(output_size_per_partition, input_size)` where
`output_size_per_partition = output_size // tp_world_size`. The partition dimension is
`dim=0` of the weight matrix.

```
Full weight: [output_size, input_size]
   |  Partition 0 (rank 0)  |
   |  Partition 1 (rank 1)  |
   |  ...                   |
   |  Partition N (rank N)  |
```

```python
world_size = get_pg_size(self.tp_group)
self.output_size_per_partition = divide(output_size, world_size)
self.weight = Parameter(
    torch.empty(self.output_size_per_partition, self.input_size, ...)
)
```

### Forward Pass

```python
def forward(self, input_, weight=None, runtime_gather_output=None):
```

1. **Input handling:** If `sequence_parallel` or `allreduce_dgrad` is enabled, the input
   is used directly (already partitioned). Otherwise, the input is copied to the TP
   region via `copy_to_tensor_model_parallel_region`.
2. **Matrix multiply:** `output_parallel = input_parallel @ weight.T + bias`
3. **Output gathering:** If `gather_output=True`, an all-gather across TP ranks produces
   the full output tensor. Otherwise, each rank keeps its local partition.

### Backward Pass (via autograd functions)

The backward pass uses `LinearWithGradAccumulationAndAsyncCommunication`:

1. **Input gradient:** `grad_input = grad_output @ weight`
2. **Weight gradient:** `grad_weight = grad_output.T @ total_input`
3. **Communication overlap:** When `sequence_parallel=True`, the input all-gather and
   gradient reduce-scatter are overlapped with weight gradient computation using
   `async_op=True` and `CUDA_DEVICE_MAX_CONNECTIONS=1`.

### Communication Pattern

| Phase    | Communication                                  |
|----------|------------------------------------------------|
| Forward  | None (or AllGather if `gather_output=True`)    |
| Backward | AllReduce dgrad (or ReduceScatter with SP)     |

### Code Example: Weight Splitting

```python
# For a 4096->12288 linear with TP=4:
# Full weight: torch.Size([12288, 4096])
# Each rank holds: torch.Size([3072, 4096])
# Used for QKV projection, MLP fc1
```

---

## RowParallelLinear

**Source:** `megatron/core/tensor_parallel/layers.py`

### Purpose

Splits the weight matrix along the row (input) dimension. Each TP rank holds a horizontal
slice, and the results from all ranks are summed (AllReduce or ReduceScatter) to produce
the final output.

### Constructor Parameters

```python
class RowParallelLinear(torch.nn.Module):
    def __init__(
        self,
        input_size: int,                # Total input dimension
        output_size: int,               # Output dimension
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,        # Input already partitioned across TP
        skip_bias_add: bool,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: str | None = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
```

### Weight Splitting

The weight shape is `(output_size, input_size_per_partition)` where
`input_size_per_partition = input_size // tp_world_size`. The partition dimension is
`dim=1`.

```
Full weight: [output_size, input_size]
   Split horizontally:
   Rank 0: [output_size, input_size/N]
   Rank 1: [output_size, input_size/N]
   ...
```

```python
world_size = get_pg_size(self.tp_group)
self.input_size_per_partition = divide(input_size, world_size)
self.weight = Parameter(
    torch.empty(self.output_size, self.input_size_per_partition, ...)
)
```

### Forward Pass

```python
def forward(self, input_):
```

1. **Input handling:** If `input_is_parallel=True`, input is used directly. Otherwise,
   the input is scattered via `scatter_to_tensor_model_parallel_region`.
2. **Matrix multiply:** `output_parallel = input_partition @ weight.T` (local matmul).
3. **Output reduction:** If `sequence_parallel=True`, uses `reduce_scatter_to_sequence_parallel_region`.
   Otherwise, uses `reduce_from_tensor_model_parallel_region` (AllReduce).

### Communication Pattern

| Phase    | Communication                                                |
|----------|--------------------------------------------------------------|
| Forward  | AllReduce or ReduceScatter (sum partial outputs from ranks)  |
| Backward | None (or AllGather if input was scattered)                   |

### Code Example: Weight Splitting

```python
# For a 12288->4096 linear with TP=4:
# Full weight: torch.Size([4096, 12288])
# Each rank holds: torch.Size([4096, 3072])
# Used for attention output projection, MLP fc2
```

---

## VocabParallelEmbedding

**Source:** `megatron/core/tensor_parallel/layers.py`

The vocabulary table is partitioned across TP ranks along the vocabulary dimension. Each
rank holds `num_embeddings_per_partition = num_embeddings // tp_world_size` rows.

```python
class VocabParallelEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, *, init_method, config, ...):
        self.vocab_start_index, self.vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(...)
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
        self.weight = Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim, ...))
```

**Forward:**
1. Mask input tokens outside this rank's vocab range.
2. Lookup embeddings (out-of-range tokens produce zero vectors).
3. AllReduce across TP ranks to sum contributions.

---

## Communication Primitives

**Source:** `megatron/core/tensor_parallel/mappings.py`

### Core Operations

| Function                                       | Forward       | Backward          |
|------------------------------------------------|---------------|-------------------|
| `copy_to_tensor_model_parallel_region`         | Identity      | AllReduce         |
| `reduce_from_tensor_model_parallel_region`     | AllReduce     | Identity          |
| `scatter_to_tensor_model_parallel_region`       | Split (last)  | AllGather (last)  |
| `gather_from_tensor_model_parallel_region`      | AllGather (last) | Split (last)   |
| `scatter_to_sequence_parallel_region`           | Split (first) | AllGather (first) |
| `gather_from_sequence_parallel_region`          | AllGather (first) | ReduceScatter (first) |
| `reduce_scatter_to_sequence_parallel_region`    | ReduceScatter (first) | AllGather (first) |

### Autograd Function Implementations

Each communication primitive is implemented as a custom `torch.autograd.Function` that
inserts the appropriate collective in the forward and backward passes. This ensures
gradients flow correctly through the communication operations.

```python
class _CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return input_                     # No-op in forward

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, ctx.group), None  # AllReduce in backward
```

### Low-Level Collective Wrappers

```python
def _reduce(input_, group):
    """All-reduce the input tensor across the model parallel group."""
    torch.distributed.all_reduce(input_.contiguous(), group=group)

def _gather_along_first_dim(input_, group, output_split_sizes=None, use_global_buffer=False):
    """Gather tensors and concatenate along the first dimension."""
    dist_all_gather_func(output, input_.contiguous(), group=group)

def _reduce_scatter_along_first_dim(input_, group, input_split_sizes=None, use_global_buffer=False):
    """Reduce-scatter the input tensor across the model parallel group."""
    dist_reduce_scatter_func(output, input_.contiguous(), group=group)

def _gather_along_last_dim(input_, group):
    """Gather tensors and concatenate along the last dimension."""
    dist_all_gather_func(output, input_.contiguous(), group=group)
    output = torch.cat(tensor_list, dim=-1).contiguous()
```

### AllToAll Operations

The `all_to_all_sp2hp` and `all_to_all_hp2sp` functions convert between sequence-parallel
and hidden-parallel tensor layouts using AllToAll communication:

```python
def all_to_all_sp2hp(input_, group=None):
    """Transform [num_tokens/TP, H] -> [num_tokens, H/TP]"""

def all_to_all_hp2sp(input_, group=None):
    """Transform [num_tokens, H/TP] -> [num_tokens/TP, H]"""
```

---

## Sequence Parallelism

Sequence parallelism extends tensor parallelism by partitioning activations along the
sequence dimension for operations that do not involve matrix multiplication (LayerNorm,
Dropout, etc.). This reduces memory footprint and eliminates unnecessary AllReduce
operations.

### How It Works

Without sequence parallelism, each TP rank holds a full copy of the activation tensor
for non-TP regions (LayerNorm, Dropout). With sequence parallelism, the sequence length
dimension is partitioned across TP ranks.

```
Standard TP:     All ranks hold [seq_len, batch, hidden] for LayerNorm/Dropout
Sequence Parallel: Each rank holds [seq_len/TP, batch, hidden]
```

### Integration with Transformer Layer

In the transformer layer, the flow is:

```
1. [SP region] Input: [seq_len/TP, batch, hidden]
2. AllGather -> [seq_len, batch, hidden]
3. [TP region] ColumnParallelLinear (QKV projection)
4. [TP region] Attention computation
5. [TP region] RowParallelLinear (output projection)
6. ReduceScatter -> [seq_len/TP, batch, hidden]
7. [SP region] LayerNorm + Residual + Dropout
```

### Configuration

Sequence parallelism is enabled via `ModelParallelConfig.sequence_parallel = True`:

```python
# In ColumnParallelLinear.__init__:
self.sequence_parallel = config.sequence_parallel
if self.sequence_parallel and world_size <= 1:
    warnings.warn("sequence_parallel set True but TP=1; disabling")
    self.sequence_parallel = False

# In RowParallelLinear.__init__:
if self.sequence_parallel and not self.input_is_parallel:
    raise RuntimeError("input_is_parallel must be True for sequence_parallel")
```

### Communication Changes

| Component              | Without SP              | With SP                            |
|------------------------|-------------------------|------------------------------------|
| ColumnParallel fwd     | Copy (identity)         | AllGather along seq dim            |
| ColumnParallel bwd     | AllReduce               | ReduceScatter along seq dim        |
| RowParallel fwd        | AllReduce               | ReduceScatter along seq dim        |
| RowParallel bwd        | Copy (identity)         | AllGather along seq dim            |
| LayerNorm/Dropout      | Full activation         | Partitioned activation             |

---

## Communication Overlap

### Gradient Communication Overlap

The `LinearWithGradAccumulationAndAsyncCommunication` class overlaps communication with
computation in the backward pass. This requires `CUDA_DEVICE_MAX_CONNECTIONS=1`.

**Sequence parallel backward overlap:**
1. Launch async AllGather of input activations.
2. Compute `grad_input = grad_output @ weight` (overlapped with AllGather).
3. Wait for AllGather to complete.
4. Launch async ReduceScatter of `grad_input`.
5. Compute `grad_weight = grad_output.T @ total_input` (overlapped with ReduceScatter).
6. Wait for ReduceScatter to complete.

```python
# From LinearWithGradAccumulationAndAsyncCommunication.backward:
if ctx.sequence_parallel:
    handle = dist_all_gather_func(all_gather_buffer, input, group=tp_group, async_op=True)
    # grad_input computed while all_gather runs
    grad_input = grad_output.matmul(weight)
    handle.wait()
    # Now launch reduce_scatter while wgrad is computed
    handle = dist_reduce_scatter_func(sub_grad_input, grad_input, group=tp_group, async_op=True)
    # wgrad computed while reduce_scatter runs
```

### TP Communication Overlap with TransformerEngine

When using TransformerEngine (TE), the `tp_comm_overlap` feature overlaps GEMM computation
with tensor-parallel communication. This is configured outside the core layers via TE's
module-level overlap settings.

---

## Memory Savings Analysis

### Weight Memory per Rank

For a weight matrix of shape `[M, K]`:

| Parallelism       | Memory per Rank        | Total Memory (N ranks) |
|-------------------|------------------------|------------------------|
| No parallelism    | M * K elements         | N * M * K              |
| ColumnParallel    | (M/N) * K elements     | M * K                  |
| RowParallel       | M * (K/N) elements     | M * K                  |

### Activation Memory with Sequence Parallelism

For activation shape `[S, B, H]` (sequence, batch, hidden):

| Mode              | Activation per Rank    |
|-------------------|------------------------|
| Without SP        | S * B * H              |
| With SP (TP=N)    | (S/N) * B * H          |

### Concrete Example

For a 7B parameter model with TP=8:
- Model parameters: ~7B * 2 bytes (bf16) = 14 GB total
- Per-rank parameters: ~14 GB / 8 = 1.75 GB
- Activation savings with SP: ~8x reduction for non-TP regions

---

## Performance Implications

### Communication Volume

| Layer Type            | Bytes Transferred (bf16)               |
|-----------------------|----------------------------------------|
| ColumnParallel fwd    | 0 (unless gather_output)               |
| ColumnParallel bwd    | 2 * S * B * H (AllReduce)              |
| RowParallel fwd       | 2 * S * B * H (AllReduce)              |
| RowParallel bwd       | 0                                      |

With sequence parallelism, AllReduce is replaced by ReduceScatter + AllGather, which
maintains the same total volume but reduces peak memory:

| Layer Type            | With SP                                 |
|-----------------------|-----------------------------------------|
| ColumnParallel fwd    | AllGather: S * B * H / TP per rank      |
| ColumnParallel bwd    | ReduceScatter: S * B * H / TP per rank  |
| RowParallel fwd       | ReduceScatter: S * B * H / TP per rank  |
| RowParallel bwd       | AllGather: S * B * H / TP per rank      |

### Overlapping Communication with Computation

The async communication overlap in backward pass is critical for performance:
- Without overlap: backward compute + communication (serialized)
- With overlap: backward compute overlapped with communication
- Requires `CUDA_DEVICE_MAX_CONNECTIONS=1` to ensure kernel scheduling order

### Bucket Size and Gradient Buffering

The `DistributedDataParallelConfig.bucket_size` parameter controls the size of gradient
buckets for overlapped all-reduce/reduce-scatter. Default is
`max(40000000, 1000000 * dp_size)` parameters.

### Best Practices

1. Keep TP within a single node (NVLink domain) for minimal latency.
2. Use `gather_output=False` for ColumnParallel when feeding into RowParallel (avoids
   unnecessary AllGather followed by scatter).
3. Enable `sequence_parallel=True` when TP > 1 for activation memory savings.
4. Set `CUDA_DEVICE_MAX_CONNECTIONS=1` for communication-computation overlap.
5. Use `gradient_accumulation_fusion=True` with APEX for fused wgrad GEMM.
