# Pipeline Parallelism Reference

This document provides a detailed technical reference for Megatron-LM's pipeline
parallelism implementation, covering scheduling algorithms, point-to-point communication,
and pipeline-stage layout customization.

## Table of Contents

- [Overview](#overview)
- [Schedule Selection](#schedule-selection)
- [Non-Interleaved 1F1B Schedule](#non-interleaved-1f1b-schedule)
- [Interleaved 1F1B Schedule (Virtual Pipeline Stages)](#interleaved-1f1b-schedule)
- [P2P Communication](#p2p-communication)
- [Pipeline Stage Utilities](#pipeline-stage-utilities)
- [Warmup and Cooldown Phases](#warmup-and-cooldown-phases)
- [Pipeline Bubble Analysis and Optimization](#pipeline-bubble-analysis)
- [num_microbatches Calculation](#num-microbatches-calculation)
- [Pipeline Parallel Layout Customization](#pipeline-parallel-layout-customization)
- [Uneven Pipeline Stage Sizes](#uneven-pipeline-stage-sizes)
- [Defer Embedding Wgrad During Pipeline Flush](#defer-embedding-wgrad)
- [Multi-Module Pipeline Parallelism](#multi-module-pipeline-parallelism)
- [Performance Implications](#performance-implications)

---

## Overview

Pipeline parallelism (PP) partitions the transformer model layers across multiple GPUs.
Each GPU holds a contiguous subset of layers called a "pipeline stage." Microbatches flow
through the stages sequentially: forward activations pass from stage 0 to the last stage,
and backward gradients flow in the reverse direction.

The key files are:
- `megatron/core/pipeline_parallel/schedules.py` -- schedule implementations
- `megatron/core/pipeline_parallel/p2p_communication.py` -- P2P communication layer
- `megatron/core/pipeline_parallel/utils.py` -- utility functions and schedule nodes

---

## Schedule Selection

The `get_forward_backward_func` function selects the appropriate schedule based on
`pipeline_model_parallel_size` (PP) and `virtual_pipeline_model_parallel_size` (VP):

```python
def get_forward_backward_func(pp_size=None, vp_size=None):
    if pp_size > 1:
        if vp_size is not None:
            return forward_backward_pipelining_with_interleaving
        else:
            return forward_backward_pipelining_without_interleaving
    else:
        return forward_backward_no_pipelining
```

| Schedule                                           | PP Size | VP Size |
|----------------------------------------------------|---------|---------|
| `forward_backward_no_pipelining`                   | 1       | None    |
| `forward_backward_pipelining_without_interleaving` | > 1     | None    |
| `forward_backward_pipelining_with_interleaving`    | > 1     | > 1     |

---

## Non-Interleaved 1F1B Schedule

**Function:** `forward_backward_pipelining_without_interleaving`

This is the standard one-forward-one-backward schedule. Each pipeline stage processes
microbatches in the following order:

### Schedule Phases

```
Phase 1: Warmup (forward only)
  Stage 0: F0 F1 F2 F3 ...
  Stage 1:    F0 F1 F2 ...
  Stage 2:       F0 F1 ...
  Stage 3:          F0 ...

Phase 2: Steady-state 1F1B (one forward, one backward)
  Stage 0: F4 B0 F5 B1 ...
  Stage 1: F3 B0 F4 B1 ...
  Stage 2: F2 B0 F3 B1 ...
  Stage 3: F1 B0 F2 B1 ...

Phase 3: Cooldown (backward only)
  Stage 0:          B4 B5 B6 B7
  Stage 1:       B4 B5 B6 B7
  Stage 2:    B4 B5 B6 B7
  Stage 3: B4 B5 B6 B7
```

### Implementation Flow

```python
def forward_backward_pipelining_without_interleaving(
    *, forward_step_func, data_iterator, model, num_microbatches,
    seq_length, micro_batch_size, forward_only=False, ...
):
    # 1. Compute number of warmup microbatches
    num_warmup_microbatches = total_stages - current_stage - 1
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # 2. Warmup: forward passes only
    for i in range(num_warmup_microbatches):
        input_tensor = recv_forward(tensor_shapes, is_first_stage)
        output_tensor = forward_step(...)
        send_forward(output_tensor, is_last_stage)

    # 3. Steady state: 1F1B
    if num_microbatches_remaining > 0:
        input_tensor = recv_forward(...)
    for i in range(num_microbatches_remaining):
        output_tensor = forward_step(...)
        output_tensor_grad = send_forward_recv_backward(output_tensor, ...)
        input_tensor_grad = backward_step(...)
        if not last_iteration:
            input_tensor = send_backward_recv_forward(input_tensor_grad, ...)
        else:
            send_backward(input_tensor_grad, ...)

    # 4. Cooldown: backward passes only
    for i in range(num_warmup_microbatches):
        output_tensor_grad = recv_backward(...)
        input_tensor_grad = backward_step(...)
        send_backward(input_tensor_grad, ...)
```

### No Pipelining Schedule

When PP=1, `forward_backward_no_pipelining` runs all microbatches sequentially with
optional gradient accumulation via `no_sync_func`:

```python
with no_sync_func():
    for i in range(num_microbatches - 1):
        output_tensor = forward_step(...)
        if not forward_only:
            backward_step(...)
# Last microbatch outside no_sync to trigger grad sync
output_tensor = forward_step(...)
if not forward_only:
    backward_step(...)
```

---

## Interleaved 1F1B Schedule

**Function:** `forward_backward_pipelining_with_interleaving`

### Virtual Pipeline Stages

Interleaved pipeline parallelism assigns multiple "virtual stages" (model chunks) to each
GPU. For example, with PP=4 and VP=2 on a 16-layer model:

```
GPU 0: layers [1,2]  + layers [9,10]    (model chunks 0, 1)
GPU 1: layers [3,4]  + layers [11,12]
GPU 2: layers [5,6]  + layers [13,14]
GPU 3: layers [7,8]  + layers [15,16]
```

This reduces the pipeline bubble because each GPU can begin backward passes sooner by
having smaller, interleaved model chunks.

### Schedule Table

The `get_schedule_table` function creates a lookup table that maps `virtual_microbatch_id`
to `(microbatch_id, model_chunk_id)`:

```python
def get_schedule_table(num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage):
    # Example: PP=2, num_microbatches=5, VP=2
    # virtual_id | 0 1 2 3 4 5 6 7 8 9
    # microbatch | 0 1 2 0 1 2 3 4 3 4
    # model_chunk| 0 0 0 1 1 1 0 0 1 1
```

### Microbatch Group Size

The `microbatch_group_size_per_vp_stage` config controls how many contiguous microbatches
are processed for a virtual stage before switching. This must satisfy:

```
PP <= microbatch_group_size_per_vp_stage <= num_microbatches
```

### Number of Warmup Microbatches

```python
num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
num_warmup_microbatches += (num_model_chunks - 1) * microbatch_group_size_per_vp_stage
```

### P2P Communication Overlap

The interleaved schedule supports overlapping P2P communication with computation via
`config.overlap_p2p_comm`:

```python
if config.overlap_p2p_comm:
    # Forward send/recv and backward send/recv are overlapped
    # using async P2P operations
    output_tensor, input_tensor_grad = forward_backward_helper_wrapper(
        f_virtual_microbatch_id=forward_k,
        b_virtual_microbatch_id=backward_k,
        pre_forward=pp_pre_forward,
        pre_backward=pp_pre_backward,
        post_forward=pp_post_forward,
        post_backward=pp_post_backward,
    )
```

### Warmup Flush Overlap

When `config.overlap_p2p_comm_warmup_flush` is enabled, receive operations are prefetched
during the warmup phase to overlap with forward computation.

---

## P2P Communication

**Source:** `megatron/core/pipeline_parallel/p2p_communication.py`

### P2PCommunicator Class

The `P2PCommunicator` class encapsulates all P2P communication for pipeline parallelism:

```python
class P2PCommunicator:
    def __init__(self, pp_group, config):
        self.pp_group = pp_group
        self.config = config
        self.next_rank = ...  # Global rank of next stage
        self.prev_rank = ...  # Global rank of previous stage
```

### Core Methods

| Method                                 | Description                                       |
|----------------------------------------|---------------------------------------------------|
| `recv_forward(tensor_shapes, is_first)` | Receive activation from previous stage            |
| `send_forward(output_tensors, is_last)` | Send activation to next stage                     |
| `recv_backward(tensor_shapes, is_last)` | Receive gradient from next stage                  |
| `send_backward(input_grads, is_first)`  | Send gradient to previous stage                   |
| `send_forward_recv_backward(...)`        | Combined forward send + backward recv             |
| `send_backward_recv_forward(...)`        | Combined backward send + forward recv             |
| `send_forward_recv_forward(...)`         | Combined forward send + forward recv (overlap)    |
| `send_backward_recv_backward(...)`       | Combined backward send + backward recv (overlap)  |

### Communication Backends

Two P2P communication modes are supported:

1. **Batched P2P** (`config.batch_p2p_comm=True`): Uses `torch.distributed.batch_isend_irecv`
   to group all send/recv operations into a single batch call.

```python
def _batched_p2p_ops(*, tensor_send_prev, tensor_recv_prev, tensor_send_next,
                     tensor_recv_next, group, prev_pipeline_rank, next_pipeline_rank):
    ops = []
    if tensor_send_prev is not None:
        ops.append(torch.distributed.P2POp(torch.distributed.isend, tensor_send_prev, prev_pipeline_rank, group))
    # ... collect all ops
    reqs = torch.distributed.batch_isend_irecv(ops)
    return reqs
```

2. **Non-batched P2P** (`config.batch_p2p_comm=False`): Issues individual `isend`/`irecv`
   calls with rank-based ordering to avoid deadlocks:

```python
def _p2p_ops(...):
    # Even ranks send first, odd ranks receive first
    # For PP=2, uses WORLD group for one direction to enable overlap
    if group.rank() % 2 == 0:
        send_next_req = torch.distributed.isend(tensor=tensor_send_next, dst=next_rank, ...)
        recv_prev_req = torch.distributed.irecv(tensor=tensor_recv_prev, src=prev_rank, ...)
        # ...
    else:
        recv_prev_req = torch.distributed.irecv(...)
        send_next_req = torch.distributed.isend(...)
```

3. **Ring Exchange** (`config.use_ring_exchange_p2p=True`): Uses `torch.distributed.ring_exchange`
   for simultaneous send/recv in both directions.

### Shape Communication

When `config.variable_seq_lengths=True` or `config.mtp_standalone=True`, tensor shapes
are communicated dynamically between stages before data transfer:

```python
def _communicate_shapes(self, tensor_send_next, tensor_send_prev, recv_prev, recv_next):
    # Send/receive shape tensors (3-element int64)
    # Then allocate receive buffers with the correct shape
```

### Pipeline Communication Backend

The PP group can use either NCCL or UCC backend (`pipeline_model_parallel_comm_backend`):

- **NCCL**: Default, high bandwidth utilization within NVLink domains.
- **UCC**: Better IB bandwidth utilization, zero SM resource usage. Requires
  `CUDA_DEVICE_MAX_CONNECTIONS > 1`.

---

## Pipeline Stage Utilities

**Source:** `megatron/core/pipeline_parallel/utils.py`

### Stage Identification

```python
def is_pp_first_stage(pp_group):   # Returns True for rank 0 in PP group
def is_pp_last_stage(pp_group):    # Returns True for last rank in PP group
def is_vp_first_stage(vp_stage, vp_size):  # Virtual stage checks
def is_vp_last_stage(vp_stage, vp_size):
def get_pp_first_rank(pp_group):   # Global rank of first stage
def get_pp_last_rank(pp_group):    # Global rank of last stage
def get_pp_next_rank(pp_group):    # Global rank of next stage (or None)
def get_pp_prev_rank(pp_group):    # Global rank of previous stage (or None)
```

### ScheduleNode

The `ScheduleNode` class is used for fine-grained scheduling within a pipeline stage,
particularly for MoE expert parallelism and combined 1F1B schedules:

```python
class ScheduleNode:
    def __init__(self, forward_func, stream, event, backward_func=None, free_input=False, name=""):
        # Represents a single computational node (e.g., attention, expert)
```

### Output Tensor Deallocation

```python
def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    """Pseudo-deallocate output tensor by replacing data with scalar tensor.
    The output is only needed for its .grad_fn after being sent."""
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype)
```

---

## Warmup and Cooldown Phases

### Warmup Phase

During warmup, each pipeline stage runs only forward passes to fill the pipeline. The
number of warmup microbatches depends on the stage rank:

**Non-interleaved:**
```python
num_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank - 1
num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
```

- Stage 0 (first): runs `PP - 1` warmup forward passes
- Stage 1: runs `PP - 2` warmup forward passes
- Last stage: runs 0 warmup forward passes

**Interleaved:**
```python
num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
num_warmup_microbatches += (num_model_chunks - 1) * microbatch_group_size_per_vp_stage
```

### Cooldown Phase

During cooldown, each stage runs only backward passes to flush remaining activations from
the pipeline. The cooldown phase processes the remaining `num_warmup_microbatches`
backward passes.

### Gradient Sync Control

Gradient synchronization is disabled during warmup and most of steady-state, and enabled
only at the end:

```python
# Disable grad sync during warmup and steady-state
disable_grad_sync()
# Enable at last microbatch or during cooldown
enable_grad_sync()
```

---

## Pipeline Bubble Analysis

### Bubble Size

The pipeline bubble is the fraction of time when GPUs are idle waiting for data from
other stages.

**Non-interleaved 1F1B:**
```
Bubble fraction = (PP - 1) / M
where M = total number of microbatches
```

**Interleaved 1F1B (virtual pipeline stages):**
```
Bubble fraction = (PP - 1) / (M * VP)
where VP = virtual pipeline model parallel size
```

The interleaved schedule reduces the bubble by a factor of VP, at the cost of increased
P2P communication (2x more send/recv operations per microbatch).

### Forward-Only Mode

When `forward_only=True`, all microbatches are run in the warmup phase (no backward
passes). This is used for inference and validation:

```python
if forward_only:
    num_warmup_microbatches = total_num_microbatches
```

### Activation Checkpointing

Partial activation checkpointing can reduce memory during warmup:

```python
if config.num_microbatches_with_partial_activation_checkpoints is not None:
    max_outstanding_backprops = num_warmup_microbatches + 1
    # Selectively checkpoint activations based on microbatch ID
    checkpoint_activations_microbatch = (
        k % max_outstanding_backprops >= config.num_microbatches_with_partial_activation_checkpoints
    )
```

---

## num_microbatches Calculation

The number of microbatches is typically calculated as:

```python
num_microbatches = global_batch_size // (micro_batch_size * data_parallel_size)
```

Constraints:
- Must be divisible by `pipeline_model_parallel_size` for even pipeline utilization.
- For interleaved schedule, must be divisible by `microbatch_group_size_per_vp_stage`.

---

## Pipeline Parallel Layout Customization

### Pipeline Stage Assignment

The pipeline stage for a given global rank is determined by the `order` parameter in
`initialize_model_parallel`. With the default order `"tp-cp-ep-dp-pp"`:

```
global_rank = tp_rank + cp_rank * TP + ep_rank * TP * CP + dp_rank * TP * CP * EP
```

The PP group contains ranks that differ only in the PP dimension.

### Custom Embedding Ranks

By default, embeddings are placed on the first and last pipeline stages:

```python
def default_embedding_ranks(pp_ranks):
    if len(pp_ranks) == 1:
        return [pp_ranks[0]]
    return [pp_ranks[0], pp_ranks[-1]]
```

Custom embedding placement can be specified via `get_embedding_ranks` callback:

```python
def custom_embedding_ranks(pp_ranks, rank_offset=None):
    return [pp_ranks[0], pp_ranks[2], pp_ranks[-1]]  # Custom placement

initialize_model_parallel(..., get_embedding_ranks=custom_embedding_ranks)
```

---

## Uneven Pipeline Stage Sizes

Megatron-LM supports assigning different numbers of transformer layers to different
pipeline stages. This is configured when building the transformer model by specifying
`num_layers_in_first_pipeline_stage` or similar parameters.

The pipeline schedule itself does not change with uneven stage sizes -- each stage still
processes microbatches in the same 1F1B order. However, the compute time per microbatch
will differ across stages, which can increase the pipeline bubble.

### Balancing Considerations

- Memory-bound models may benefit from assigning fewer layers to stages that hold
  embeddings or other large tensors.
- Compute-bound models benefit from equal layer counts per stage.
- The UCC backend (`pipeline_model_parallel_comm_backend="ucc"`) can help mitigate
  performance impact from uneven stage sizes due to zero SM usage for communication.

---

## Defer Embedding Wgrad

During the pipeline flush (cooldown phase), the embedding weight gradient computation
can be deferred to overlap with other operations.

### Configuration

```python
config.defer_embedding_wgrad_compute = True
config.wgrad_deferral_limit = 0  # 0 = defer all microbatches
```

### Implementation

```python
def clear_embedding_activation_buffer(config, model, is_last_stage):
    """Clear buffers on last pipeline stage at the start of forward-backward."""
    if is_last_stage and config.defer_embedding_wgrad_compute:
        embedding_module.embedding_activation_buffer.clear()
        return embedding_module

def finish_embedding_wgrad_compute(config, embedding_module, is_last_stage, tp_group):
    """Compute deferred embedding wgrads after pipeline flush."""
    if is_last_stage and config.defer_embedding_wgrad_compute:
        drain_embedding_wgrad_compute(
            config, embedding_activation_buffer, grad_output_buffer, weight, tp_group
        )
```

In `ColumnParallelLinear.forward`, activations and grad_outputs are buffered:

```python
if self.config.defer_embedding_wgrad_compute:
    if self.config.wgrad_deferral_limit == 0 or \
       len(self.embedding_activation_buffer) < self.config.wgrad_deferral_limit:
        self.embedding_activation_buffer.append(input_parallel)
```

---

## Multi-Module Pipeline Parallelism

The pipeline parallel infrastructure supports multi-module models (e.g., vision-language
models with separate encoder and decoder) via `MultiModulePipelineCommunicator` and
`MultiModuleProcessGroupCollection`.

In this mode, tensors are organized as dictionaries with module names as keys:

```python
def backward_step_multimodule(input_tensor, output_tensor, output_tensor_grad, config,
                              language_model_module_name):
    # Each module's backward pass is performed independently
    for module_name in output_tensor.keys():
        torch.autograd.backward(output_tensor[module_name],
                                grad_tensors=output_tensor_grad[module_name])
```

---

## Performance Implications

### Communication Volume

Each P2P transfer sends an activation tensor of shape `[seq_len, micro_batch_size, hidden_size]`:

```
Per microbatch per stage: 2 * S * B * H bytes (bf16)
Total per pipeline step: 2 * S * B * H * (PP - 1) * M * 2 bytes
```

### Overlapping P2P with Computation

- `overlap_p2p_comm`: Overlaps P2P communication with forward/backward computation in
  the interleaved schedule.
- `overlap_p2p_comm_warmup_flush`: Prefetches receive operations during warmup.
- Both require `CUDA_DEVICE_MAX_CONNECTIONS=1` for kernel ordering.

### Batch P2P vs Non-Batch

- `batch_p2p_comm=True`: Groups all operations into `batch_isend_irecv`. Lower overhead
  but less opportunity for overlap.
- `batch_p2p_comm=False`: Individual `isend`/`irecv` calls with rank-based ordering.
  More overlap opportunity.

### Pipeline Bubble Reduction

| Strategy                     | Bubble Fraction              |
|------------------------------|------------------------------|
| GPipe (all-forward first)    | (PP - 1) / M                 |
| 1F1B non-interleaved         | (PP - 1) / M                 |
| 1F1B interleaved (VP=V)      | (PP - 1) / (M * V)           |

### Best Practices

1. Ensure `num_microbatches >= pipeline_model_parallel_size` for optimal pipeline utilization.
2. Use interleaved schedule (VP > 1) for smaller pipeline bubbles when memory permits.
3. Set `deallocate_pipeline_outputs=True` to free activation tensors after P2P send.
4. Use `overlap_p2p_comm=True` with the interleaved schedule for communication hiding.
5. Consider UCC backend for long-distance (IB) pipeline stages.
6. Align embedding layers to pipeline stages that minimize communication.
