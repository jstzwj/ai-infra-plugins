# FullyShardedDataParallel (FSDP)

FSDP shards model parameters, gradients, and optimizer state across data-parallel workers.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, CPUOffload
```

---

## FSDP Constructor

```python
model = FSDP(
    module: nn.Module,
    process_group=None,              # ProcessGroup or (inter, intra) tuple
    sharding_strategy=None,          # ShardingStrategy enum
    cpu_offload=None,                # CPUOffload config
    auto_wrap_policy=None,           # wrap policy callable
    backward_prefetch=None,          # BackwardPrefetch enum
    forward_prefetch: bool = False,
    mixed_precision=None,            # MixedPrecision config
    use_orig_params: bool = False,   # needed for torch.compile
    sync_module_states: bool = False,# broadcast rank 0 state
    device_id=None,                  # device for computation
    ignored_modules=None,            # modules to exclude from sharding
    param_init_fn=None,              # custom param init callback
)
```

---

## Sharding Strategies

```python
from torch.distributed.fsdp import ShardingStrategy
```

| Strategy | Parameters | Gradients | Optimizer | Communication |
|----------|-----------|-----------|-----------|---------------|
| `FULL_SHARD` | Sharded | Sharded | Sharded | AllGather + ReduceScatter |
| `SHARD_GRAD_OP` | Replicated | Sharded | Sharded | ReduceScatter only |
| `NO_SHARD` | Replicated | Replicated | Replicated | AllReduce (like DDP) |
| `HYBRID_SHARD` | Sharded intra-node, replicated inter-node | | | |

```python
# Full sharding (most memory efficient)
model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)

# Hybrid for multi-node
model = FSDP(model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    process_group=(inter_node_group, intra_node_group))
```

Memory per rank: `(params + grads + opt) / world_size + activations` (FULL_SHARD)

---

## MixedPrecision

```python
from torch.distributed.fsdp import MixedPrecision

mp = MixedPrecision(
    param_dtype=torch.bfloat16,     # dtype for computation parameters
    reduce_dtype=torch.bfloat16,     # dtype for gradient reduction
    buffer_dtype=torch.bfloat16,     # dtype for buffers (BatchNorm stats)
)
model = FSDP(model, mixed_precision=mp)

# BF16 compute with FP32 reduction (more precise gradients)
mp = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.bfloat16,
)
```

---

## CPUOffload

```python
from torch.distributed.fsdp import CPUOffload

model = FSDP(model, cpu_offload=CPUOffload(offload_params=True))
# Parameters stored on CPU, gathered to GPU for compute
# Reduces GPU memory ~2-5x slower due to CPU-GPU transfer
```

---

## Auto Wrap Policies

```python
import functools
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
    lambda_auto_wrap_policy,
    ModuleWrapPolicy,
)

# Size-based: wrap when submodule has >= 1M params
model = FSDP(model, auto_wrap_policy=functools.partial(
    size_based_auto_wrap_policy, min_num_params=1_000_000))

# Transformer: wrap at transformer block granularity
model = FSDP(model, auto_wrap_policy=functools.partial(
    transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock}))

# Module class-based
model = FSDP(model, auto_wrap_policy=ModuleWrapPolicy({TransformerBlock, MLP}))
```

Guidelines: coarse wrapping = less communication overlap; fine wrapping = more overlap but more AllGather ops. Block-level wrapping is a good balance for transformers.

---

## State Dict Management

```python
from torch.distributed.fsdp import FullStateDictConfig, LocalStateDictConfig, StateDictType

# Save full (unsharded) checkpoint - rank 0 only
save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_cfg):
    state = model.state_dict()
    if dist.get_rank() == 0:
        torch.save({"model": state, "epoch": epoch}, "ckpt.pt")

# Load full checkpoint
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
    model.load_state_dict(torch.load("ckpt.pt", map_location="cpu")["model"])

# Save local (sharded) - one file per rank
with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
    torch.save(model.state_dict(), f"shard_{dist.get_rank()}.pt")
```

---

## Backward Prefetch

```python
from torch.distributed.fsdp import BackwardPrefetch

model = FSDP(model, backward_prefetch=BackwardPrefetch.BACKWARD_PRE)
# BACKWARD_PRE: prefetch before needed (more overlap, higher memory)
# BACKWARD_POST: prefetch after current backward (less memory)
```

---

## FSDP + torch.compile

```python
# use_orig_params=True is required for torch.compile
model = FSDP(model, use_orig_params=True, auto_wrap_policy=policy)
model = torch.compile(model)
```

---

## FSDP + Activation Checkpointing

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, apply_activation_checkpointing,
)

# Apply before FSDP wrapping
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=checkpoint_wrapper,
    check_fn=lambda m: isinstance(m, TransformerBlock),
)
model = FSDP(model, auto_wrap_policy=policy)
```

---

## summon_full_params (Manual Unshard)

```python
# Temporarily materialize full parameters
with FSDP.summon_full_params(model, writeback=False, recurse=True):
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
# Auto-resharded on exit

# With writeback for modifications
with FSDP.summon_full_params(model, writeback=True):
    for p in model.parameters():
        p.data.fill_(0)
```

---

## Complete Training Example

```python
import os, functools, torch, torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FSDP, ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

class Block(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.ln2 = nn.LayerNorm(d)
    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.ln1(x + a)
        return self.ln2(x + self.ff(x))

class GPT(nn.Module):
    def __init__(self, v, d, h, ff, n):
        super().__init__()
        self.embed = nn.Embedding(v, d)
        self.blocks = nn.ModuleList([Block(d, h, ff) for _ in range(n)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, v)
    def forward(self, x):
        x = self.embed(x)
        for b in self.blocks: x = b(x)
        return self.head(self.ln(x))

def main():
    dist.init_process_group("nccl")
    lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)

    model = GPT(50000, 2048, 16, 8192, 24)
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    model = FSDP(model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={Block}),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp, device_id=lr,
        sync_module_states=True, use_orig_params=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sampler = DistributedSampler(dataset, rank=dist.get_rank(),
                                  num_replicas=dist.get_world_size())
    loader = DataLoader(dataset, batch_size=8, sampler=sampler)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for ids, lbls in loader:
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(
                model(ids.to(lr)).view(-1, 50000), lbls.to(lr).view(-1))
            loss.backward()
            opt.step()

if __name__ == "__main__":
    main()
```

Launch: `torchrun --nproc_per_node=8 train.py`
