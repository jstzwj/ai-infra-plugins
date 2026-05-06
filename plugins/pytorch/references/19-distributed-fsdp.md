# PyTorch FullyShardedDataParallel (FSDP) - Comprehensive Reference

This chapter covers `torch.distributed.fsdp.FullyShardedDataParallel` (FSDP), PyTorch's native solution for training large models with parameter sharding.

## Table of Contents

1. [FSDP Overview](#fsdp-overview)
2. [FSDP Constructor](#fsdp-constructor)
3. [Sharding Strategies](#sharding-strategies)
4. [Mixed Precision](#mixed-precision)
5. [CPU Offloading](#cpu-offloading)
6. [Auto Wrap Policies](#auto-wrap-policies)
7. [State Dict Management](#state-dict-management)
8. [FSDP + torch.compile](#fsdp--torchcompile)
9. [FSDP + Activation Checkpointing](#fsdp--activation-checkpointing)
10. [Memory Optimization Strategies](#memory-optimization-strategies)
11. [Scaling to Thousands of GPUs](#scaling-to-thousands-of-gpus)
12. [Unshard/Reshard Operations](#unshardreshard-operations)
13. [Complete Setup and Training Example](#complete-setup-and-training-example)

---

## FSDP Overview

FullyShardedDataParallel (FSDP) shards model parameters, gradients, and optimizer state across data parallel workers. Unlike DDP, which replicates the entire model on each GPU, FSDP distributes the model state, enabling training of models that are too large to fit on a single GPU.

### How FSDP Works

1. **Sharding**: Model parameters are partitioned (sharded) across all ranks.
2. **Unsharding (Forward)**: Before the forward pass of an FSDP unit, the full parameters for that unit are gathered from all ranks (all-gather).
3. **Computation**: The forward pass executes with the full parameters.
4. **Resharding**: After the forward pass, the gathered parameters are freed (returned to sharded state).
5. **Gradient Sharding (Backward)**: During backward, gradients are computed with full parameters, then reduce-scattered so each rank holds only its gradient shard.
6. **Optimizer Step**: Each rank updates only its parameter shard using its gradient shard and optimizer state shard.

### FSDP vs DDP

| Aspect | DDP | FSDP |
|--------|-----|------|
| Model size | Must fit on 1 GPU | Can exceed single GPU |
| Parameter replication | Full replica per GPU | Sharded across GPUs |
| Communication | AllReduce gradients | AllGather params + ReduceScatter grads |
| Memory per GPU | Full model + optimizer | Sharded model + optimizer |
| Setup complexity | Simple | Medium |
| Best for | Small-medium models | Large models |

---

## FSDP Constructor

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    module,
    process_group=None,
    sharding_strategy=None,
    cpu_offload=None,
    auto_wrap_policy=None,
    backward_prefetch=None,
    forward_prefetch=False,
    mixed_precision=None,
    use_orig_params=False,
    sync_module_states=False,
    device_id=None,
    ignored_modules=None,
    param_init_fn=None,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | nn.Module | required | The module to wrap with FSDP. |
| `process_group` | ProcessGroup | None | The process group for sharding. Default: the default global group. Can be a tuple of (inter-node group, intra-node group) for hybrid sharding. |
| `sharding_strategy` | ShardingStrategy | FULL_SHARD | The sharding strategy. |
| `cpu_offload` | CPUOffload | None | CPU offloading configuration. |
| `auto_wrap_policy` | Callable or None | None | Policy for automatically wrapping submodules. |
| `backward_prefetch` | BackwardPrefetch | None | Controls when to prefetch parameters for the backward pass. |
| `forward_prefetch` | bool | False | If True, prefetches the next FSDP unit's parameters during the current forward pass. |
| `mixed_precision` | MixedPrecision | None | Mixed precision configuration. |
| `use_orig_params` | bool | False | If True, preserves the original parameter objects. Required for `torch.compile` compatibility and some optimizers. |
| `sync_module_states` | bool | False | If True, synchronizes module parameters and buffers across ranks before wrapping. Rank 0's state is broadcast to all ranks. |
| `device_id` | int or torch.device | None | The device for this FSDP unit's computation. If None, uses the device of the first parameter. |
| `ignored_modules` | Iterable[nn.Module] | None | Modules whose parameters should not be sharded. |
| `param_init_fn` | Callable | None | Function to initialize module parameters. Called on each rank with the module. |

---

## Sharding Strategies

```python
from torch.distributed.fsdp import ShardingStrategy
```

### FULL_SHARD

Full sharding of parameters, gradients, and optimizer state. Each rank stores only 1/N of the total model state.

```python
model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
```

**Characteristics:**
- Maximum memory savings
- Highest communication volume (AllGather for parameters + ReduceScatter for gradients)
- Best for models that don't fit on a single GPU

**Memory per rank:** `(params + grads + optimizer_state) / world_size + activations`

### SHARD_GRAD_OP

Shards gradients and optimizer state, but keeps full parameters replicated.

```python
model = FSDP(model, sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)
```

**Characteristics:**
- Similar to DDP for parameters (full replica)
- Gradients and optimizer state are sharded
- No AllGather needed for forward/backward (saves communication)
- More memory than FULL_SHARD for parameters, same for optimizer state
- Useful when parameter memory is not the bottleneck

**Memory per rank:** `params + (grads + optimizer_state) / world_size + activations`

### NO_SHARD

No sharding at all. Equivalent to DDP but with FSDP's wrapping and mixed precision features.

```python
model = FSDP(model, sharding_strategy=ShardingStrategy.NO_SHARD)
```

**Characteristics:**
- Full parameter, gradient, and optimizer state replication
- No communication for parameters (same as DDP)
- Only gradient AllReduce (same as DDP)
- Useful for debugging or as a baseline

### HYBRID_SHARD

Combines intra-node sharding (FULL_SHARD within a node) with inter-node replication.

```python
from torch.distributed.fsdp import ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    process_group=(inter_node_group, intra_node_group)
)
```

**Characteristics:**
- Parameters are fully sharded within each node
- Each node has a complete replica of the model
- Reduces inter-node communication (only AllReduce for gradients)
- Good for multi-node setups where intra-node bandwidth (NVLink) is much higher than inter-node (InfiniBand)

**Memory per node:** `(params + grads + optimizer_state) / local_world_size + activations`

### _HYBRID_SHARD_ZERO2

Same as HYBRID_SHARD but uses SHARD_GRAD_OP within each node instead of FULL_SHARD.

```python
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy._HYBRID_SHARD_ZERO2,
    process_group=(inter_node_group, intra_node_group)
)
```

### Strategy Selection Guide

```
Model fits on 1 GPU?
  Yes -> DDP or FSDP(NO_SHARD)
  No
    -> Model fits on 1 node with FULL_SHARD?
        Yes -> FSDP(FULL_SHARD)
        No
          -> Need multi-node?
              Yes -> FSDP(HYBRID_SHARD) or 3D parallelism
              No -> Need more memory optimization (CPU offload, activation checkpointing)
```

---

## Mixed Precision

FSDP supports mixed precision training with fine-grained control over parameter, reduction, and buffer dtypes.

### MixedPrecision Configuration

```python
from torch.distributed.fsdp import MixedPrecision

mixed_precision = MixedPrecision(
    param_dtype=torch.float32,       # Dtype for parameters during computation
    reduce_dtype=torch.float32,       # Dtype for gradient reduction
    buffer_dtype=torch.float32,       # Dtype for buffers
)

model = FSDP(model, mixed_precision=mixed_precision)
```

### Common Configurations

```python
# BF16 mixed precision (recommended for Ampere+ GPUs)
bf16_mp = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)
model = FSDP(model, mixed_precision=bf16_mp)

# FP16 mixed precision
fp16_mp = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)
model = FSDP(model, mixed_precision=fp16_mp)

# BF16 computation with FP32 reduction (more precise gradients)
bf16_mp_fp32_reduce = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.bfloat16,
)
model = FSDP(model, mixed_precision=bf16_mp_fp32_reduce)

# FP32 parameters with FP16 reduction (memory-efficient reduction)
fp32_mp_fp16_reduce = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float32,
)
model = FSDP(model, mixed_precision=fp32_mp_fp16_reduce)
```

### MixedPrecision Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `param_dtype` | torch.dtype | None | Dtype for parameters during forward/backward computation. Parameters are cast to this dtype after unsharding. |
| `reduce_dtype` | torch.dtype | None | Dtype for gradient reduction (ReduceScatter). Gradients are cast to this dtype before communication. |
| `buffer_dtype` | torch.dtype | None | Dtype for buffers (e.g., BatchNorm running stats). |

---

## CPU Offloading

FSDP can offload parameters and/or gradients to CPU to reduce GPU memory usage.

### CPUOffload Configuration

```python
from torch.distributed.fsdp import CPUOffload

# Offload parameters to CPU when not in use
cpu_offload = CPUOffload(offload_params=True)
model = FSDP(model, cpu_offload=cpu_offload)

# No CPU offloading (default)
cpu_offload = CPUOffload(offload_params=False)
model = FSDP(model, cpu_offload=cpu_offload)
```

### How CPU Offloading Works

```
1. Parameters are stored on CPU in sharded form
2. Before forward/backward: Parameters are gathered to GPU via H2D transfer
3. After computation: GPU parameters are freed
4. During backward: Gradients may be computed on GPU and moved to CPU
5. Optimizer step happens on CPU (or GPU, depending on configuration)
```

### Performance Impact

CPU offloading significantly reduces GPU memory usage but adds CPU-GPU transfer overhead:

```python
# With CPU offloading
model = FSDP(model, cpu_offload=CPUOffload(offload_params=True))
# GPU memory: activations + small working set of parameters
# Performance: ~2-5x slower due to CPU-GPU transfer overhead

# With CPU offloading + mixed precision (smaller transfers)
mp = MixedPrecision(param_dtype=torch.bfloat16)
model = FSDP(model, cpu_offload=CPUOffload(offload_params=True), mixed_precision=mp)
# BF16 halves the CPU-GPU transfer size
```

### When to Use CPU Offloading

- GPU memory is the bottleneck (model doesn't fit even with sharding).
- Training throughput is less critical than model size.
- Combined with NVMe offloading for very large models.

---

## Auto Wrap Policies

Auto wrap policies determine how submodules are wrapped into FSDP units. The granularity of wrapping affects the trade-off between memory efficiency and communication overhead.

### size_based_auto_wrap_policy

Wraps submodules when their parameter count exceeds a threshold.

```python
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# Wrap when a submodule has >= 1M parameters
model = FSDP(
    model,
    auto_wrap_policy=functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=1_000_000
    )
)
```

### transformer_auto_wrap_policy

Specifically designed for transformer models. Wraps at the transformer layer granularity.

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# Assuming your model has a TransformerBlock class
class TransformerBlock(nn.Module):
    ...

model = FSDP(
    model,
    auto_wrap_policy=functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock}
    )
)
```

### lambda_auto_wrap_policy

Wraps submodules based on a custom lambda function.

```python
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

# Wrap submodules whose class name contains 'Block'
model = FSDP(
    model,
    auto_wrap_policy=functools.partial(
        lambda_auto_wrap_policy,
        lambda_fn=lambda module: 'Block' in type(module).__name__
    )
)
```

### ModuleWrapPolicy

Wraps specified module classes.

```python
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

# Wrap all instances of TransformerBlock and MLP
model = FSDP(
    model,
    auto_wrap_policy=ModuleWrapPolicy({TransformerBlock, MLP})
)
```

### Choosing a Wrap Policy

**Guidelines:**
1. **Coarse wrapping** (entire model as one FSDP unit): Simplest, but no overlapping of communication and computation.
2. **Fine wrapping** (every layer as an FSDP unit): Maximum overlap, but more AllGather operations.
3. **Block-level wrapping** (transformer blocks): Good balance for transformer models.
4. **Size-based wrapping**: Good default for unknown architectures.

```python
# Coarse (simple but less efficient)
model = FSDP(model)  # No auto_wrap_policy

# Block-level (recommended for transformers)
model = FSDP(model, auto_wrap_policy=functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock}
))

# Size-based (good default)
model = FSDP(model, auto_wrap_policy=functools.partial(
    size_based_auto_wrap_policy,
    min_num_params=1_000_000
))
```

---

## State Dict Management

FSDP provides multiple options for saving and loading model state.

### Full State Dict

```python
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

# Configure for full state dict (unsharded, for saving)
save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
    state_dict = model.state_dict()
    if dist.get_rank() == 0:
        torch.save(state_dict, 'model.pt')
```

### Local State Dict

```python
from torch.distributed.fsdp import LocalStateDictConfig, StateDictType

# Save local (sharded) state dict - more memory efficient
local_policy = LocalStateDictConfig(offload_to_cpu=True)
with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT, local_policy):
    local_state_dict = model.state_dict()
    torch.save(local_state_dict, f'model_shard_{dist.get_rank()}.pt')
```

### Sharded State Dict

```python
from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType

# Save sharded state dict (one file per rank)
sharded_policy = ShardedStateDictConfig(offload_to_cpu=True)
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_policy):
    state_dict = model.state_dict()
```

### Loading State Dict

```python
# Load full state dict
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
    state_dict = torch.load('model.pt', map_location='cpu')
    model.load_state_dict(state_dict)

# Load local state dict
with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
    local_state_dict = torch.load(f'model_shard_{dist.get_rank()}.pt')
    model.load_state_dict(local_state_dict)
```

### Complete Checkpoint/Load Example

```python
import os
import functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

def save_checkpoint(model, optimizer, epoch, rank):
    """Save a checkpoint that can be loaded on any number of GPUs."""
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()

    if rank == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': cpu_state,
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'checkpoint.pt')
    dist.barrier()

def load_checkpoint(model, optimizer, rank):
    """Load a checkpoint."""
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        checkpoint = torch.load('checkpoint.pt', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']
```

---

## FSDP + torch.compile

FSDP is compatible with `torch.compile` when using `use_orig_params=True`.

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Must use use_orig_params=True for torch.compile compatibility
model = FSDP(
    model,
    use_orig_params=True,
    auto_wrap_policy=auto_wrap_policy,
)

# Compile the FSDP model
model = torch.compile(model)

# Or compile the underlying module before FSDP wrapping
model = torch.compile(model)
model = FSDP(model, use_orig_params=True)
```

### Known Limitations

- `use_orig_params=True` is required for `torch.compile` compatibility.
- Some FSDP features may have limited compatibility with `torch.compile`.
- Graph breaks may occur if the model has dynamic control flow.

---

## FSDP + Activation Checkpointing

Activation checkpointing (gradient checkpointing) reduces memory usage by recomputing activations during the backward pass instead of storing them.

### Using Activation Checkpointing with FSDP

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

# Option 1: Wrap specific modules
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = TransformerBlock(768, 12, 3072)
        self.block2 = TransformerBlock(768, 12, 3072)
        self.block3 = TransformerBlock(768, 12, 3072)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

model = MyModel()

# Apply activation checkpointing to transformer blocks
check_fn = lambda submodule: isinstance(submodule, TransformerBlock)
apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)

# Then wrap with FSDP
model = FSDP(model, auto_wrap_policy=auto_wrap_policy)
```

### Selective Activation Checkpointing

```python
# Checkpoint only every other layer (trade memory for compute)
def selective_checkpoint_check(submodule):
    # Only checkpoint transformer blocks at even indices
    if isinstance(submodule, TransformerBlock):
        return submodule.layer_id % 2 == 0
    return False

apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    ),
    check_fn=selective_checkpoint_check,
)
```

---

## Memory Optimization Strategies

### Strategy 1: Sharding Strategy Selection

```python
from torch.distributed.fsdp import ShardingStrategy

# Most memory efficient: FULL_SHARD
model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)

# Less memory efficient but less communication: SHARD_GRAD_OP
model = FSDP(model, sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)
```

### Strategy 2: Mixed Precision

```python
from torch.distributed.fsdp import MixedPrecision

# BF16 halves parameter memory during computation
mp = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)
model = FSDP(model, mixed_precision=mp)
```

### Strategy 3: CPU Offloading

```python
from torch.distributed.fsdp import CPUOffload

# Offload parameters to CPU
model = FSDP(model, cpu_offload=CPUOffload(offload_params=True))
```

### Strategy 4: Activation Checkpointing

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing, checkpoint_wrapper,
)

# Recompute activations during backward (saves activation memory)
apply_activation_checkpointing(model, check_fn=lambda m: isinstance(m, TransformerBlock))
```

### Strategy 5: Gradient Accumulation

```python
# Accumulate gradients over multiple steps before optimizer step
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    output = model(batch)
    loss = output.sum() / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Strategy 6: Backward Prefetch

```python
from torch.distributed.fsdp import BackwardPrefetch

# Prefetch parameters for backward pass to overlap communication
model = FSDP(
    model,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Prefetch before needed
    # or BackwardPrefetch.BACKWARD_POST  # Prefetch after current backward
)
```

---

## Scaling to Thousands of GPUs

### Hybrid Sharding for Multi-Node

```python
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

# Create intra-node and inter-node process groups
world_size = dist.get_world_size()
rank = dist.get_rank()
local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])

# Number of nodes
num_nodes = world_size // local_world_size

# Node-local group (ranks on the same node)
node_id = rank // local_world_size
local_ranks = list(range(node_id * local_world_size, (node_id + 1) * local_world_size))
intra_node_group = dist.new_group(ranks=local_ranks)

# Inter-node group (one rank per node)
inter_node_ranks = list(range(0, world_size, local_world_size))
inter_node_group = dist.new_group(ranks=inter_node_ranks)

# Hybrid sharding
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    process_group=(inter_node_group, intra_node_group),
    auto_wrap_policy=auto_wrap_policy,
)
```

### NCCL Tuning for Large Scale

```bash
# For large-scale FSDP training
export NCCL_SOCKET_IFNAME=ib0          # Use InfiniBand
export NCCL_IB_DISABLE=0                # Enable IB
export NCCL_IB_HCA=mlx5_0              # IB HCA
export NCCL_NET_GDR_LEVEL=5             # GPU Direct RDMA
export NCCL_MIN_NCHANNELS=8             # More channels for bandwidth
export NCCL_MAX_NCHANNELS=32
export NCCL_BUFFSIZE=16777216           # 16MB buffer
export NCCL_ALGO=Ring                   # Ring algorithm for AllGather
export NCCL_PROTO=LL                   # Low-latency protocol
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

---

## Unshard/Reshard Operations

FSDP internally manages unsharding (gathering full parameters) and resharding (releasing them) during forward and backward passes.

### Manual Unshard/Reshard

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Unshard: gather full parameters for all FSDP units
with FSDP.summon_full_params(model, writeback=False, recurse=True):
    # All parameters are fully materialized here
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

# Parameters are automatically resharded when exiting the context

# Writeback=True allows modifying parameters
with FSDP.summon_full_params(model, writeback=True, recurse=True):
    for param in model.parameters():
        param.data.fill_(0)  # Zero all parameters
    # Changes are written back to sharded form
```

### summon_full_params Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | FSDP | required | The FSDP model |
| `writeback` | bool | True | If True, writes modified parameters back to sharded form |
| `recurse` | bool | True | If True, applies to nested FSDP units |
| `offload_to_cpu` | bool | False | If True, offloads unsharded params to CPU |

### FSDP.validate_state_dict

```python
# Validate that all ranks have consistent model state
FSDP.validate_state_dict(model)
```

---

## Complete Setup and Training Example

```python
import os
import functools
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing, checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

def setup():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def main():
    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create model
    model = GPTModel(
        vocab_size=50000,
        d_model=2048,
        n_heads=16,
        d_ff=8192,
        n_layers=24,
    )

    # Apply activation checkpointing before FSDP
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=checkpoint_wrapper,
        check_fn=lambda m: isinstance(m, TransformerBlock),
    )

    # Configure mixed precision
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Wrap with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerBlock}
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        device_id=local_rank,
        sync_module_states=True,  # Ensure all ranks start with same weights
        use_orig_params=True,     # For torch.compile compatibility
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Data setup
    dataset = MyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler,
                            num_workers=4, pin_memory=True)

    # Training loop
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        model.train()

        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(local_rank)
            labels = labels.to(local_rank)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0 and rank == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss={loss.item():.4f}")

        # Save checkpoint
        if rank == 0:
            save_checkpoint(model, optimizer, epoch)

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```

### Launching

```bash
# Single-node, 8 GPUs
torchrun --nproc_per_node=8 train_fsdp.py

# Multi-node, 4 nodes with 8 GPUs each
torchrun --nnodes=4 --nproc_per_node=8 \
    --rdzv_id=job1 --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_IP:29500 \
    train_fsdp.py
```
