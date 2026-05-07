# Data Parallelism Reference

This document provides a detailed technical reference for Megatron-LM's data parallelism
implementations, covering DDP, Megatron-FSDP, the distributed optimizer, and
communication overlap strategies.

## Table of Contents

- [Overview](#overview)
- [DistributedDataParallelConfig](#distributeddataparallelconfig)
- [DDP (DistributedDataParallel)](#ddp-distributeddataparallel)
- [Megatron-FSDP (Fully Sharded Data Parallel)](#megatron-fsdp)
- [Sharding Strategies (ZeRO Equivalents)](#sharding-strategies)
- [Distributed Optimizer](#distributed-optimizer)
- [Communication Overlap](#communication-overlap)
- [Mixed Precision with FSDP](#mixed-precision-with-fsdp)
- [Integration with TP/PP/CP](#integration-with-tp-pp-cp)
- [Memory Savings Calculations](#memory-savings-calculations)
- [Performance Implications](#performance-implications)

---

## Overview

Megatron-LM provides two data parallelism implementations:

1. **DistributedDataParallel (DDP):** Traditional data parallelism with optional
   distributed optimizer (ZeRO-1). Gradients are stored in contiguous buffers and
   synchronized via all-reduce or reduce-scatter.

2. **Megatron-FSDP:** Fully sharded data parallelism supporting ZeRO-1 through ZeRO-3
   style sharding. Parameters, gradients, and optimizer states can be sharded across
   data-parallel ranks.

Both implementations support overlapping communication with computation, gradient
accumulation, and integration with tensor, pipeline, context, and expert parallelism.

Key source files:
- `megatron/core/distributed/distributed_data_parallel.py` -- DDP implementation
- `megatron/core/distributed/distributed_data_parallel_config.py` -- shared config
- `megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py` -- FSDP implementation
- `megatron/core/distributed/param_and_grad_buffer.py` -- contiguous gradient buffers

---

## DistributedDataParallelConfig

**Source:** `megatron/core/distributed/distributed_data_parallel_config.py`

This dataclass configures both DDP and FSDP behavior:

```python
@dataclass
class DistributedDataParallelConfig:
    grad_reduce_in_fp32: bool = False
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    align_param_gather: bool = False
    use_distributed_optimizer: bool = False
    num_distributed_optimizer_instances: int = 1
    check_for_nan_in_grad: bool = False
    check_for_large_grads: bool = False
    bucket_size: Optional[int] = None
    pad_buckets_for_high_nccl_busbw: bool = False
    reduce_scatter_with_fp32_accumulation: bool = False
    average_in_collective: bool = False
    fp8_param_gather: bool = False
    fp4_param_gather: bool = False
    use_megatron_fsdp: bool = False
    data_parallel_sharding_strategy: str = 'no_shard'
    gradient_reduce_div_fusion: bool = True
    nccl_ub: bool = False
    fsdp_double_buffer: bool = False
    delay_wgrad_compute: bool = False
    megatron_fsdp_main_params_dtype: Optional[torch.dtype] = torch.float32
    megatron_fsdp_main_grads_dtype: Optional[torch.dtype] = None
    megatron_fsdp_grad_comm_dtype: Optional[torch.dtype] = None
    outer_dp_sharding_strategy: str = 'no_shard'
```

### Key Configuration Options

| Option                          | Description                                                     |
|---------------------------------|-----------------------------------------------------------------|
| `grad_reduce_in_fp32`           | Accumulate and reduce gradients in FP32 for numerical stability |
| `overlap_grad_reduce`           | Overlap gradient all-reduce/RS with backward compute             |
| `overlap_param_gather`          | Overlap parameter all-gather with forward compute                |
| `use_distributed_optimizer`     | Enable distributed optimizer (ZeRO-1)                            |
| `use_megatron_fsdp`             | Use Megatron-FSDP instead of DDP                                |
| `data_parallel_sharding_strategy`| FSDP sharding level: `no_shard`, `optim`, `optim_grads`, `optim_grads_params` |
| `bucket_size`                   | Max parameters per gradient bucket                               |
| `nccl_ub`                       | Use NCCL userbuffer for low-SM communication                     |
| `average_in_collective`         | Average gradients within the collective operation                 |

---

## DDP (DistributedDataParallel)

**Source:** `megatron/core/distributed/distributed_data_parallel.py`

### Class Overview

```python
class DistributedDataParallel(_BaseDataParallel):
    def __init__(self, config, ddp_config, module, disable_bucketing=False,
                 pg_collection=None, full_param_layout=None):
```

The DDP wrapper stores gradients in contiguous buffers organized by dtype and expert
parallelism. This enables efficient bucket-based gradient reduction.

### Gradient Buffer Architecture

Parameters are grouped by `(param_dtype, grad_dtype, is_expert_parallel)` into buffer
groups. Each group allocates a contiguous `_ParamAndGradBuffer` that holds all gradients:

```python
# Group parameters by dtype and expert status
buffer_groups = group_params_for_buffers(all_params, ddp_config.grad_reduce_in_fp32)

# Allocate contiguous buffers for each group
for buffer_key, (params, param_indices) in buffer_groups.items():
    buffer = _ParamAndGradBuffer(
        ddp_config, param_dtype, grad_dtype, params_with_names,
        data_parallel_group, bucket_size, ...
    )
```

### Gradient Synchronization

#### Without Distributed Optimizer

Uses all-reduce to aggregate gradients:

```python
def start_grad_sync(self, *unused):
    for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
        bucket_group.start_grad_sync()
    # Internally calls torch.distributed.all_reduce on each bucket
```

#### With Distributed Optimizer

Uses reduce-scatter to shard gradients across DP ranks:

```python
# When use_distributed_optimizer=True:
# Uses reduce-scatter instead of all-reduce
# Each rank keeps only 1/DP_size of the gradient
```

### Bucketing

Gradients are divided into buckets for overlapped communication. Bucket size defaults
to `max(40000000, 1000000 * dp_size)` parameters:

```python
if ddp_config.bucket_size is None:
    ddp_config.bucket_size = max(40000000, 1000000 * dp_group.size())
if not ddp_config.overlap_grad_reduce:
    ddp_config.bucket_size = None  # Single bucket when no overlap
```

### no_sync Context Manager

```python
@contextmanager
def no_sync(self):
    """Disable gradient synchronization for gradient accumulation."""
    for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
        bucket_group.is_last_microbatch = False
    try:
        yield
    finally:
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.is_last_microbatch = True
```

### Backward Hook

Each parameter registers a backward post-hook that accumulates gradients into the
contiguous buffer and triggers bucket-level communication:

```python
def _make_backward_post_hook(self, param):
    def hook(*unused):
        if param.grad is not None and not param.grad_added_to_main_grad:
            param.main_grad.add_(param.grad.data)
        param.grad = None
        if self.ddp_config.overlap_grad_reduce:
            self.param_to_bucket_group[param].register_grad_ready(param, self.force_all_reduce)
    return hook
```

### Key Methods

| Method                | Description                                                 |
|-----------------------|-------------------------------------------------------------|
| `start_grad_sync()`   | Initiate gradient all-reduce or reduce-scatter              |
| `finish_grad_sync()`  | Wait for async gradient operations to complete              |
| `start_param_sync()`  | Initiate parameter all-gather (for distributed optimizer)   |
| `zero_grad_buffer()`  | Zero out gradient buffers for next iteration                 |
| `scale_gradients()`   | Scale all gradients by a factor                              |
| `broadcast_params()`  | Broadcast parameters across DP ranks                         |
| `no_sync()`           | Context manager for gradient accumulation                    |

---

## Megatron-FSDP

**Source:** `megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py`

### Class Overview

```python
class MegatronFSDP(torch.nn.Module):
    def __init__(self, module, dist_index, ddp_config=None,
                 mixed_precision_policy=MixedPrecisionPolicy(),
                 fsdp_unit_modules=None, disable_bucketing=False,
                 device=None, init_model_with_meta_device=False,
                 sync_model_each_microbatch=False, nccl_ub=False, ...):
```

Megatron-FSDP is a fully-featured FSDP implementation that shards parameters, gradients,
and optimizer states across data-parallel workers.

### Training State Machine

```python
class TrainingState(Enum):
    FORWARD = auto()        # Parameters unsharded for forward
    PRE_BACKWARD = auto()   # Before backward, parameters unsharded
    POST_BACKWARD = auto()  # After backward, gradients re-sharded
    IDLE = auto()           # No sharding activity
```

### FSDP Unit Modules

FSDP unit modules define the granularity of parameter sharding. Only parameters within
an FSDP unit are released together after forward/backward passes:

```python
fsdp_unit_modules = [TransformerLayer, LanguageModelEmbedding]
```

### Hooks

Megatron-FSDP registers the following hooks on the model:

| Hook Type         | Purpose                                              |
|-------------------|------------------------------------------------------|
| Pre-forward       | Unshard (all-gather) parameters before forward       |
| Post-forward      | Reshard (release) parameters after forward           |
| Pre-backward      | Unshard parameters before backward                   |
| Post-backward     | Reshard parameters and reduce-scatter gradients      |
| Grad accumulator  | Accumulate gradients into main_grad buffer           |

```python
def _pre_forward_param_unshard(module, *unused):
    # All-gather parameters before forward
    self.all_gather_and_wait_parameters_ready(
        params=param_list, prefetch=True, prefetch_order=PrefetchOrder.FORWARD_PASS_ORDER
    )

def _post_forward(module, input, output):
    # Release parameters after forward
    release_module_parameters(module, bwd=False, lazy=lazy_release)

def _pre_backward_param_unshard(module, *unused):
    # All-gather parameters before backward
    self.all_gather_and_wait_parameters_ready(
        param_list, prefetch_order=PrefetchOrder.BACKWARD_PASS_ORDER, bwd=True
    )

def _post_backward_release_module(module, *unused):
    # Release parameters after backward
    release_module_parameters(module, bwd=True)
    release_module_parameters(module, bwd=False)
```

### Parameter All-Gather Pipeline

The `AllGatherPipeline` manages asynchronous parameter all-gathering with prefetching:

```python
self.all_gather_pipeline = AllGatherPipeline(
    self.param_and_grad_buffer, ag_stream=self.side_stream_for_param_gather
)
```

Prefetching fetches parameters for the next FSDP unit while the current one is computing:
```python
suggested_communication_unit_size = total_param_elements // total_fsdp_module * 2
self.suggested_AG_prefetch_size = suggested_communication_unit_size // 2
```

### Gradient Reduce-Scatter Pipeline

The `GradReducePipeline` manages asynchronous gradient reduce-scatter:

```python
self.grad_reduce_pipeline = GradReducePipeline(
    self.param_and_grad_buffer, rs_stream=self.side_stream_for_buffer_copy_and_grad_accum
)
```

### Key Methods

| Method                         | Description                                            |
|--------------------------------|--------------------------------------------------------|
| `start_param_sync()`           | Initiate parameter all-gather                          |
| `start_grad_sync()`            | Initiate gradient all-reduce or reduce-scatter         |
| `finish_grad_sync()`           | Wait for gradient sync, update optimizer params        |
| `zero_grad_buffer()`           | Zero gradient buffers                                  |
| `install_optimized_model_weights()` | Copy optimized weights to model buffers           |
| `broadcast_params()`           | Broadcast parameters across DP ranks                   |
| `no_sync()`                    | Context manager for gradient accumulation              |
| `sync()`                       | Context manager for per-step synchronization           |
| `set_model_auto_sync(bool)`    | Control automatic gradient/param synchronization      |

---

## Sharding Strategies

### Strategy Overview

| Strategy                | ZeRO Equivalent | Parameters | Gradients | Optimizer States |
|-------------------------|-----------------|------------|-----------|------------------|
| `no_shard`              | None (pure DP)  | Replicated | Replicated | Replicated       |
| `optim`                 | ZeRO-1          | Replicated | Replicated | Sharded          |
| `optim_grads`           | ZeRO-2          | Replicated | Sharded    | Sharded          |
| `optim_grads_params`    | ZeRO-3          | Sharded    | Sharded    | Sharded          |

### Behavior Per Strategy

**`no_shard`:**
- Parameters are replicated across all DP ranks.
- Gradients are accumulated and all-reduced (or averaged) each optimization cycle.
- No parameter all-gather needed.
- Compatible with basic DDP training.

**`optim`:**
- Parameters replicated, gradients replicated, optimizer states sharded.
- Uses `use_distributed_optimizer=True` in DDP mode.
- Reduce-scatter shards gradient for optimizer step.
- Main weight buffer maintained in FP32 for mixed precision.

**`optim_grads`:**
- Parameters replicated, gradients sharded, optimizer states sharded.
- `overlap_grad_reduce` is automatically enabled.
- Reduce-scatter happens every backward pass.
- No need to accumulate full gradients across microbatches.

**`optim_grads_params`:**
- All three are sharded.
- `overlap_param_gather` and `overlap_grad_reduce` are automatically enabled.
- Parameters are all-gathered before each forward/backward pass and released after.
- Maximum memory savings but highest communication overhead.

### Hybrid FSDP (HSDP)

Megatron-FSDP supports hybrid sharding via `outer_dp_sharding_strategy`:

```python
data_parallel_sharding_strategy = "optim_grads_params"  # Inner FSDP group
outer_dp_sharding_strategy = "optim"                     # Outer DP-Replica group
```

This creates a two-level hierarchy:
1. **Inner group (DP-Shard):** Full ZeRO-3 sharding across a subset of DP ranks.
2. **Outer group (DP-Outer):** ZeRO-1 style optimizer state sharding across inner groups.

---

## Distributed Optimizer

The distributed optimizer shards optimizer states across data-parallel ranks. It is
enabled via `DistributedDataParallelConfig.use_distributed_optimizer = True`.

### How It Works

1. **Gradient reduce-scatter:** Instead of all-reduce, gradients are reduce-scattered so
   each rank receives only its shard of the gradient.
2. **Optimizer state sharding:** Each rank maintains optimizer states (momentum, variance
   in Adam) only for its shard of parameters.
3. **Parameter all-gather:** Before the forward pass, parameters are all-gathered from
   the sharded main weight buffers.

### Partial Distributed Optimizer

With `num_distributed_optimizer_instances > 1`, the DP domain is split into sub-groups,
each maintaining its own optimizer instance:

```python
intra_partial_dp_size = (data_parallel_size * context_parallel_size) // num_distributed_optimizer_instances
```

This enables HSDP-style hybrid sharding where optimizer states are sharded within
sub-groups and replicated across sub-groups.

---

## Communication Overlap

### Gradient Reduction Overlap (`overlap_grad_reduce`)

When enabled, gradient all-reduce or reduce-scatter is overlapped with backward computation:

1. Gradients are bucketed by parameter group.
2. As each bucket completes backward, its communication is launched asynchronously.
3. The backward hook `_make_backward_post_hook` triggers bucket-level reduction.

```python
# DDP backward hook
def hook(*unused):
    if param.grad is not None and not param.grad_added_to_main_grad:
        param.main_grad.add_(param.grad.data)
    param.grad = None
    if self.ddp_config.overlap_grad_reduce:
        self.param_to_bucket_group[param].register_grad_ready(param, ...)
```

### Parameter Gather Overlap (`overlap_param_gather`)

When enabled, parameter all-gather is overlapped with forward computation:

1. Before forward, the first bucket's all-gather is initiated.
2. A forward pre-hook waits for each parameter's bucket to be ready before use.
3. After use, the bucket is released and the next bucket's all-gather is dispatched.

```python
def _make_forward_pre_hook(self):
    def hook(module, *unused):
        for param in module.parameters(recurse=False):
            self.param_to_bucket_group[param].finish_param_sync(
                skip_next_bucket_dispatch=skip_next_bucket_dispatch
            )
    return hook
```

### NCCL Userbuffer (`nccl_ub`)

When `nccl_ub=True`, NCCL userbuffers are used for communication, which reduces SM
usage:

| Communication Domain | use_sharp | SM Usage (AG/RS) |
|----------------------|-----------|-------------------|
| NVLink               | N/A       | 4 / 5             |
| NVLink + IB          | False     | 16 / 16           |
| NVLink + IB          | True      | 6 / 6             |
| IB                   | False     | 1 / 4             |
| IB                   | True      | 1 / 1             |

This requires `fsdp_double_buffer=True` (automatically set).

---

## Mixed Precision with FSDP

### MixedPrecisionPolicy

```python
class MixedPrecisionPolicy:
    main_params_dtype: torch.dtype    # Main weight buffer dtype (default: FP32)
    main_grads_dtype: torch.dtype     # Main gradient buffer dtype
    grad_comm_dtype: torch.dtype      # Communication buffer dtype
```

### Mixed Precision Flow

```
Model params (BF16) --> Main weights (FP32) --> Gather (BF16) --> Compute
                                                        |
Compute grads (BF16) --> Reduce-scatter --> Main grads (FP32) --> Optimizer step
```

### FP8 Support

Megatron-FSDP supports FP8 parameters with `fp8_param_gather=True`:

```python
# Keep compute param in FP8
fp8_param_gather: bool = False
# Keep FP8 transpose cache for performance
keep_fp8_transpose_cache: bool = False
```

### Gradient Communication Dtype

`megatron_fsdp_grad_comm_dtype` allows using lower precision for communication while
maintaining higher precision for accumulation:

```python
megatron_fsdp_grad_comm_dtype = torch.bfloat16  # Communicate in BF16
megatron_fsdp_main_grads_dtype = torch.float32   # Accumulate in FP32
```

---

## Integration with TP/PP/CP

### Process Group Hierarchy

Megatron-LM's data parallelism integrates with all other parallelism dimensions. The
`initialize_model_parallel` function creates all necessary process groups:

```python
# Default order: tp-cp-ep-dp-pp
# This means:
# - TP groups are formed from adjacent ranks
# - DP groups span ranks differing only in the DP dimension
# - PP groups span ranks differing only in the PP dimension
```

### DP + TP Integration

- Data-parallel communication is independent of TP.
- The `_TENSOR_AND_DATA_PARALLEL_GROUP` is used for FP8 amax reduction.
- Sequence parallelism piggybacks on DP groups for weight gradient reduction.

### DP + PP Integration

- Only the first pipeline stage uses fine-grained gradient bucketing:
  ```python
  if disable_bucketing or pp_rank > 0:
      self.bucket_size = None  # Single bucket on non-first PP stages
  ```
- The `align_param_gather` option synchronizes param all-gather across PP stages.

### DP + CP Integration

- Context parallelism is folded into the DP group for gradient reduction:
  `_DATA_PARALLEL_GROUP_WITH_CP` combines DP and CP ranks.
- Gradient scaling accounts for both DP and CP sizes.

### DP + Expert Parallelism

Expert parameters use separate expert DP groups for gradient reduction:

```python
for buffer_key, (params, param_indices) in buffer_groups.items():
    if buffer_key.is_expert_parallel:
        data_parallel_group = self.intra_expt_dp_group
    else:
        data_parallel_group = self.intra_dp_cp_group
```

---

## Memory Savings Calculations

### Per-Rank Memory for a Model with P Parameters

Assuming FP32 optimizer states (Adam: 2 states per parameter), BF16 parameters, and
DP=DP_size:

| Component        | No Sharding   | ZeRO-1 (`optim`) | ZeRO-2 (`optim_grads`) | ZeRO-3 (`optim_grads_params`) |
|------------------|---------------|-------------------|------------------------|-------------------------------|
| Parameters       | 2P bytes      | 2P bytes          | 2P bytes               | 2P/DP bytes                   |
| Gradients        | 2P bytes      | 2P bytes          | 2P/DP bytes            | 2P/DP bytes                   |
| Optimizer states | 8P bytes      | 8P/DP bytes       | 8P/DP bytes            | 8P/DP bytes                   |
| Main weights     | 0             | 4P bytes          | 4P bytes               | 4P/DP bytes                   |
| **Total**        | **12P bytes** | **2P + 12P/DP**   | **2P + 14P/DP**        | **16P/DP**                    |

### Concrete Example: 70B Parameter Model with DP=64

| Strategy          | Memory per Rank             |
|-------------------|-----------------------------|
| No sharding       | 12 * 70B = 840 GB           |
| ZeRO-1            | 140 GB + 13.1 GB = 153.1 GB |
| ZeRO-2            | 140 GB + 15.3 GB = 155.3 GB |
| ZeRO-3            | 16 * 70B / 64 = 17.5 GB     |

### FSDP Double Buffer Overhead

When `fsdp_double_buffer=True`, additional memory is allocated for communication buffers:

```
Overhead = total_param_elements * sizeof(param_dtype)
```

### NCCL Userbuffer Overhead

When `nccl_ub=True`, additional memory is allocated for NCCL userbuffer registration:

```
Overhead = total_param_elements * sizeof(param_dtype) * 2  (double buffer)
```

---

## Performance Implications

### Communication Volume per Step

| Strategy         | Gradient Comm             | Parameter Comm              |
|------------------|---------------------------|-----------------------------|
| No sharding      | AllReduce: 2P per rank    | None                        |
| ZeRO-1           | ReduceScatter: 2P/DP      | AllGather: 2P/DP per bucket |
| ZeRO-2           | ReduceScatter: 2P/DP      | None                        |
| ZeRO-3           | ReduceScatter: 2P/DP      | AllGather: 2P/DP per layer  |

### Overlap Effectiveness

The effectiveness of communication-computation overlap depends on:

1. **Compute-to-communication ratio:** Larger models have more computation per layer,
   providing more opportunity to hide communication.
2. **Bucket size:** Larger buckets amortize communication startup overhead but reduce
   overlap granularity.
3. **Network bandwidth:** NVLink (600+ GB/s) enables effective overlap within a node;
   IB (100-400 Gb/s) may become the bottleneck for cross-node DP.

### Gradient Accumulation

With `no_sync()`, gradients are accumulated across microbatches without communication:

```python
# Gradient accumulation across K microbatches
with ddp_model.no_sync():
    for i in range(K - 1):
        output = ddp_model(batch)
        loss = criterion(output)
        loss.backward()
# Last microbatch triggers communication
output = ddp_model(batch)
loss = criterion(output)
loss.backward()
```

### Recommended Configurations

| Scenario                          | Recommended Config                               |
|-----------------------------------|--------------------------------------------------|
| Single-node, small model          | DDP, `no_shard`, `overlap_grad_reduce=True`      |
| Single-node, large model          | FSDP, `optim_grads`, overlap both                |
| Multi-node, large model           | FSDP, `optim_grads_params`, NCCL UB, HSDP       |
| Maximum throughput, enough memory | DDP, `optim`, `overlap_grad_reduce=True`         |
| Minimum memory                    | FSDP, `optim_grads_params`, double buffer        |

### Best Practices

1. Use `overlap_grad_reduce=True` on the first pipeline stage for overlapped gradient
   reduction.
2. Use `overlap_param_gather=True` with distributed optimizer for overlapped parameter
   gathering.
3. Set `bucket_size` appropriately: larger for high-DP counts, smaller for low-latency
   networks.
4. Use `average_in_collective=True` to offload gradient averaging to the NCCL kernel.
5. Enable `nccl_ub=True` for SM-efficient communication with large DP sizes.
6. Use `gradient_reduce_div_fusion=True` (default) to fuse gradient scaling with
   reduction.
7. For mixed precision training, use `grad_reduce_in_fp32=True` for numerical stability
   when training large models.
