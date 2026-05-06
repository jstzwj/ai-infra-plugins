# PyTorch DistributedDataParallel (DDP) - Comprehensive Reference

This chapter covers `torch.nn.parallel.DistributedDataParallel` (DDP), PyTorch's primary data parallelism module.

## Table of Contents

1. [DDP Overview](#ddp-overview)
2. [DDP Constructor](#ddp-constructor)
3. [How DDP Works](#how-ddp-works)
4. [DDP Communication Hooks](#ddp-communication-hooks)
5. [DDP Logging and Debugging](#ddp-logging-and-debugging)
6. [Process Group Integration](#process-group-integration)
7. [Gradient Bucketing Optimization](#gradient-bucketing-optimization)
8. [find_unused_parameters](#find_unused_parameters)
9. [SyncBatchNorm with DDP](#syncbatchnorm-with-ddp)
10. [DDP + torch.compile](#ddp--torchcompile)
11. [Forward/Backward/Gradient Timeline](#forwardbackwardgradient-timeline)
12. [Performance Tips](#performance-tips)
13. [Multi-Node DDP Setup](#multi-node-ddp-setup)
14. [Fault Tolerance and Elastic Training](#fault-tolerance-and-elastic-training)

---

## DDP Overview

DistributedDataParallel (DDP) implements data parallelism at the module level. It uses communication collectives (typically AllReduce via NCCL) to synchronize gradients across processes during the backward pass.

Key characteristics:
- Each process has a complete replica of the model.
- Each process processes a different subset of the data.
- Gradients are synchronized (averaged) across all processes during backward.
- The synchronized gradients are identical across all processes, ensuring consistent parameter updates.

### When to Use DDP

- Your model fits entirely on a single GPU.
- You want to scale training across multiple GPUs or nodes.
- You want near-linear speedup with increasing GPU count.
- You want simple, reliable distributed training.

### When NOT to Use DDP

- Your model is too large for a single GPU (use FSDP or pipeline parallelism instead).
- You need sub-module-level parallelism (use tensor parallelism).
- Your model has dynamic computation graphs that vary across ranks (use `find_unused_parameters=True`).

---

## DDP Constructor

```python
torch.nn.parallel.DistributedDataParallel(
    module,
    device_ids=None,
    output_device=None,
    broadcast_buffers=True,
    bucket_cap_mb=25,
    find_unused_parameters=False,
    gradient_as_bucket_view=False,
    static_graph=False,
    delay_allreduce_named_params=None
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | Module | required | The module to be parallelized. DDP wraps this module. |
| `device_ids` | list[int] | None | CUDA device IDs for this module. Should be `[local_rank]` for single-GPU per process. If None, all visible CUDA devices are used. |
| `output_device` | int or torch.device | None | Device for module output. Default: device_ids[0] or None. |
| `broadcast_buffers` | bool | True | Whether to broadcast buffers (e.g., BatchNorm statistics) from rank 0 to all other ranks at the start of each forward pass. |
| `bucket_cap_mb` | int | 25 | The bucket size in megabytes for gradient AllReduce. DDP buckets parameters into groups and reduces each bucket together. |
| `find_unused_parameters` | bool | False | If True, DDP analyzes the forward pass output to find which parameters were used. Unused parameters' gradients are not synchronized. Required for models with dynamic computation graphs. |
| `gradient_as_bucket_view` | bool | False | If True, gradients are stored as views of the AllReduce communication buffers. This reduces memory usage but means gradients cannot be accessed after `optimizer.step()` until a new backward pass. |
| `static_graph` | bool | False | If True, indicates the computation graph is static (same parameters are used in every forward pass). Enables optimizations by avoiding graph analysis. |
| `delay_allreduce_named_params` | dict | None | Experimental. Delays AllReduce for specific named parameters. |

### Basic Usage

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def main():
    local_rank = setup()
    rank = dist.get_rank()

    # Create model and wrap with DDP
    model = MyModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for data, target in dataloader:
            data, target = data.to(local_rank), target.to(local_rank)

            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    dist.destroy_process_group()
```

---

## How DDP Works

### Gradient Synchronization

DDP uses AllReduce to average gradients across all processes. This happens during the backward pass, overlapped with gradient computation.

**Mechanism:**
1. DDP registers backward hooks on each parameter.
2. When a gradient is computed for a parameter, the hook is triggered.
3. DDP batches parameters into "buckets" based on the reverse order of the forward pass.
4. When all gradients in a bucket are ready, DDP initiates an async AllReduce on that bucket.
5. The AllReduce averages the gradients across all processes.
6. After AllReduce completes, the averaged gradients are written back to the `.grad` attributes.

### Model Parameter Broadcast

When DDP is constructed, it broadcasts model parameters from rank 0 to all other ranks. This ensures all ranks start with identical model states.

```python
# This happens internally during DDP construction:
# 1. All parameters are broadcast from rank 0
# 2. All buffers are broadcast from rank 0
# 3. Bucket assignment is determined

model = DDP(model, device_ids=[local_rank])
# At this point, all ranks have identical model parameters
```

### Bucketing

Parameters are grouped into buckets for efficient communication. Each bucket is reduced as a single operation, reducing the number of small communications.

```
Bucket assignment (reverse order of forward pass):
  Bucket 0: last_layer.weight, last_layer.bias  (reduced first)
  Bucket 1: mid_layer.weight, mid_layer.bias
  Bucket 2: first_layer.weight, first_layer.bias (reduced last)
```

The bucket order is determined by the reverse order of parameter usage in the forward pass. This allows overlap: while gradients for later layers are being reduced, gradients for earlier layers are still being computed.

---

## DDP Communication Hooks

Communication hooks allow customizing how gradients are communicated between processes. They can be used to compress gradients, implement gradient quantization, or apply custom AllReduce algorithms.

### register_comm_hook

```python
ddp_model.register_comm_hook(state=None, hook=hook_fn)
```

**Parameters:**
- `state` (Any): State object passed to the hook. Can be None or any object for maintaining hook-specific state.
- `hook` (callable): The hook function. Signature: `hook(state, bucket) -> torch.futures.Future`.

### Default Hook (AllReduce)

The default hook simply performs AllReduce on gradients without modification.

```python
# Default behavior (no hook registered is equivalent to this)
def default_hook(state, bucket):
    futures = []
    for tensor in bucket.get_tensors():
        fut = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
        futures.append(fut)
    return torch.futures.Future().set_result(futures)

# Reset to default behavior
ddp_model.register_comm_hook(state=None, hook=default_hook)
```

### FP16 Compress Hook

Compresses gradients to FP16 before communication, reducing communication volume by half.

```python
import torch.distributed.algorithms.ddp_comm_hooks as comm_hooks

# FP16 compression
ddp_model.register_comm_hook(
    state=None,
    hook=comm_hooks.default_hooks.fp16_compress_hook
)

# With error feedback (recommended for better accuracy)
state = comm_hooks.powerSGD_hook.PowerSGDState(process_group=None)
ddp_model.register_comm_hook(
    state=state,
    hook=comm_hooks.default_hooks.fp16_compress_hook
)
```

### PowerSGD Hook

Uses the PowerSGD algorithm for low-rank gradient compression. Significantly reduces communication volume for large gradient tensors.

```python
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook

state = powerSGD_hook.PowerSGDState(
    process_group=None,
    matrix_approximation_rank=1,       # Rank for approximation (lower = more compression)
    start_powerSGD_iter=1000,          # Iteration to start PowerSGD (use AllReduce before this)
    use_error_feedback=True,           # Use error feedback for better accuracy
    warm_start=False,                  # Warm start for low-rank matrices
    random_seed=0,                     # Random seed for initialization
    compression_stats_logging_frequency=1000,
)

ddp_model.register_comm_hook(state=state, hook=powerSGD_hook.powerSGD_hook)
```

**PowerSGD Parameters:**
- `matrix_approximation_rank` (int): The rank of the low-rank approximation. Lower values mean more compression but less accuracy. Default: 1.
- `start_powerSGD_iter` (int): Start PowerSGD after this many iterations. Before this, standard AllReduce is used. Default: 1000.
- `use_error_feedback` (bool): Whether to use error feedback to compensate for compression errors. Default: True.
- `warm_start` (bool): Whether to reuse low-rank matrices from the previous iteration. Default: False.

### BF16 Compress Hook

```python
ddp_model.register_comm_hook(
    state=None,
    hook=comm_hooks.default_hooks.bf16_compress_hook
)
```

### Custom Communication Hook

```python
def custom_quantization_hook(state, bucket):
    """Quantize gradients to int8 before communication."""
    tensors = bucket.get_tensors()
    quantized_tensors = []

    for tensor in tensors:
        scale = tensor.abs().max() / 127.0
        quantized = (tensor / scale).to(torch.int8)
        quantized_tensors.append((quantized, scale))

    # AllReduce quantized tensors
    # ... custom communication logic ...

    # Dequantize
    # ... return dequantized gradients ...

    return future

ddp_model.register_comm_hook(state=None, hook=custom_quantization_hook)
```

---

## DDP Logging and Debugging

### get_ddp_logging_data

Get internal DDP logging and performance data.

```python
# After a forward-backward pass
logging_data = ddp_model.get_ddp_logging_data()
print(f"Num iterations: {logging_data.num_iterations}")
print(f"Bucket size: {logging_data.bucket_size_mb}")
```

### TORCH_DISTRIBUTED_DEBUG

```bash
# Detailed DDP debugging
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=2 train.py
```

This enables:
- Logging of each AllReduce operation
- Detection of unused parameters
- Verification of gradient synchronization
- Timing of communication operations

### Common Debugging Scenarios

```python
# Debug gradient mismatch between ranks
def check_gradient_sync(model, rank):
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Gather gradients from all ranks
            gathered = [torch.zeros_like(param.grad) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, param.grad)
            for i, g in enumerate(gathered):
                if not torch.allclose(g, gathered[0], atol=1e-6):
                    print(f"Rank {rank}: Gradient mismatch for {name} with rank {i}")

# Debug parameter sync
def check_parameter_sync(model, rank):
    for name, param in model.named_parameters():
        gathered = [torch.zeros_like(param) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, param.data)
        for i, p in enumerate(gathered):
            if not torch.allclose(p, gathered[0], atol=1e-6):
                print(f"Rank {rank}: Parameter mismatch for {name} with rank {i}")
```

---

## Process Group Integration

DDP uses the default process group for communication. You can specify a custom process group.

```python
import torch.distributed as dist

# Create a custom process group for DDP
group = dist.new_group(ranks=[0, 1, 2, 3])

# Use the custom group with DDP
model = DDP(model, device_ids=[local_rank], process_group=group)
```

### Multi-Backend Setup

```python
# Use NCCL for GPU communication, Gloo for CPU operations
dist.init_process_group(backend='nccl')

# For a model that needs both GPU and CPU communication
nccl_group = dist.new_group(backend='nccl')
gloo_group = dist.new_group(backend='gloo')

model = DDP(model, device_ids=[local_rank], process_group=nccl_group)
```

---

## Gradient Bucketing Optimization

### How Bucketing Works

DDP organizes parameters into buckets for efficient AllReduce:

1. Parameters are assigned to buckets in reverse order of their creation (approximately reverse forward pass order).
2. Each bucket has a maximum size controlled by `bucket_cap_mb`.
3. When all gradients in a bucket are computed, the AllReduce for that bucket starts.
4. Bucket AllReduce and gradient computation are overlapped.

### Tuning bucket_cap_mb

```python
# Default: 25 MB
model = DDP(model, device_ids=[local_rank], bucket_cap_mb=25)

# Smaller buckets: more communication overlap, more AllReduce overhead
model = DDP(model, device_ids=[local_rank], bucket_cap_mb=1)

# Larger buckets: fewer AllReduce calls, less overlap
model = DDP(model, device_ids=[local_rank], bucket_cap_mb=100)

# Maximum bucket (all parameters in one bucket)
model = DDP(model, device_ids=[local_rank], bucket_cap_mb=float('inf'))
```

**Guidelines:**
- Small models (< 100M params): Use small `bucket_cap_mb` (1-10) for fine-grained overlap.
- Medium models (100M-1B params): Default `bucket_cap_mb=25` works well.
- Large models (> 1B params): Consider larger `bucket_cap_mb` (50-100) to reduce AllReduce count.
- Profile different values to find the optimum for your specific model and hardware.

---

## find_unused_parameters

### When to Use

Set `find_unused_parameters=True` when:
- Your model has conditional branches (e.g., different paths for different inputs).
- Some parameters may not receive gradients in every forward pass.
- You use dropout that varies between ranks (rare).

### Performance Impact

```python
# Without find_unused_parameters (fastest)
model = DDP(model, device_ids=[local_rank])

# With find_unused_parameters (slower, ~10-30% overhead)
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
```

**Why it's slower:**
- DDP must traverse the autograd graph after each forward pass to determine which parameters were used.
- This adds CPU overhead proportional to the model size.
- The graph traversal is sequential and cannot be easily parallelized.

### Alternatives

```python
# Use static_graph=True if the same set of parameters are used every iteration
# (even if some parameters don't receive gradients)
model = DDP(model, device_ids=[local_rank], static_graph=True)

# Manually handle unused parameters by zeroing their gradients
model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

for data, target in dataloader:
    optimizer.zero_grad(set_to_none=True)
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    # Zero out gradients for unused parameters manually
    for name, param in model.named_parameters():
        if param.grad is None:
            param.grad = torch.zeros_like(param)

    optimizer.step()
```

---

## SyncBatchNorm with DDP

When using BatchNorm with DDP, each GPU computes statistics (mean, variance) only on its local data. SyncBatchNorm synchronizes these statistics across all GPUs for more accurate normalization.

### Converting BatchNorm to SyncBatchNorm

```python
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Method 1: Convert before wrapping with DDP
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DDP(model, device_ids=[local_rank])

# Method 2: Convert specific layers
model = MyModel()
# Convert only specific BatchNorm layers
for name, module in model.named_modules():
    if isinstance(module, nn.BatchNorm2d):
        setattr(model, name, nn.SyncBatchNorm.convert_sync_batchnorm(module))

model = DDP(model, device_ids=[local_rank])
```

### SyncBatchNorm Implementation Details

```python
# SyncBatchNorm computes:
# 1. Local mean and count on each GPU
# 2. AllGather to get all local means and counts
# 3. Compute global mean from gathered statistics
# 4. Compute local (x - global_mean)^2
# 5. AllGather to get all local variances
# 6. Compute global variance
# 7. Normalize using global statistics
```

### Performance Considerations

SyncBatchNorm adds communication overhead (two AllGather operations per layer). Consider:
- Use SyncBatchNorm only for small batch sizes per GPU where local statistics are unreliable.
- For large batch sizes per GPU (> 16), regular BatchNorm may suffice.
- Group Normalization or Layer Normalization may be better alternatives that don't require synchronization.

---

## DDP + torch.compile

DDP is compatible with `torch.compile` for additional performance optimization.

```python
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# Compile the model before wrapping with DDP
model = torch.compile(MyModel())
model = DDP(model, device_ids=[local_rank])

# Or wrap with DDP first, then compile the inner module
model = DDP(MyModel(), device_ids=[local_rank])
model.module = torch.compile(model.module)
```

### Best Practices for DDP + torch.compile

```python
# Recommended: compile first, then DDP
model = torch.compile(MyModel(), mode='default')
model = DDP(model, device_ids=[local_rank])

# For graph capture with CUDA graphs
model = torch.compile(MyModel(), mode='reduce-overhead')
model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)
```

---

## Forward/Backward/Gradient Timeline

### Timeline of a Single Training Step

```
Time ──────────────────────────────────────────────────────────>

Forward Pass:
├── Layer 1 forward
├── Layer 2 forward
├── Layer 3 forward
└── Loss computation

Backward Pass + Gradient Synchronization (Overlapped):
├── Layer 3 backward ──> Bucket 0 ready ──> AllReduce Bucket 0 ──>
├── Layer 2 backward ──> Bucket 1 ready ──> AllReduce Bucket 1 ──>
└── Layer 1 backward ──> Bucket 2 ready ──> AllReduce Bucket 2 ──>

Optimizer Step (after all gradients synchronized):
└── optimizer.step()
```

### Detailed Breakdown

1. **Forward Pass**: Each process independently computes the forward pass on its local data. Model parameters are identical across all processes (guaranteed by DDP construction).

2. **Backward Pass**: Gradient computation and communication are overlapped.
   - Gradients are computed layer by layer (reverse order).
   - When all gradients in a bucket are ready, AllReduce is initiated asynchronously.
   - Computation and communication proceed in parallel.

3. **Gradient Finalization**: After all backward hooks have fired, DDP waits for all pending AllReduce operations to complete. Gradients are now synchronized across all ranks.

4. **Optimizer Step**: Each process independently updates its parameters using the synchronized gradients. Since gradients are identical and parameters were identical, parameters remain identical after the update.

---

## Performance Tips

### 1. Set Device Before Creating DDP

```python
# Correct
torch.cuda.set_device(local_rank)
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Also correct (device_ids implies set_device)
model = DDP(model, device_ids=[local_rank])
```

### 2. Use gradient_as_bucket_view for Memory Savings

```python
# Saves memory by reusing bucket buffers for gradients
model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)
```

This eliminates the need for separate gradient buffers, reducing memory usage by the total size of model parameters. However, gradients will be invalidated after `optimizer.step()`.

### 3. Use set_to_none=True

```python
# More memory efficient than zeroing gradients
optimizer.zero_grad(set_to_none=True)
```

### 4. Tune bucket_cap_mb

```python
# Benchmark different bucket sizes
for bucket_size in [1, 5, 10, 25, 50, 100]:
    model = DDP(model, device_ids=[local_rank], bucket_cap_mb=bucket_size)
    # ... benchmark ...
```

### 5. Use DistributedSampler

```python
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=dist.get_world_size(),
    rank=dist.get_rank(),
    shuffle=True,
    drop_last=True
)

dataloader = DataLoader(
    dataset,
    batch_size=per_gpu_batch_size,
    sampler=sampler,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

# Important: set epoch for proper shuffling
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in dataloader:
        train_step(batch)
```

### 6. Use pinned memory and multiple workers

```python
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=8,          # Multiple workers for data loading
    pin_memory=True,        # Pin memory for faster GPU transfer
    persistent_workers=True, # Keep workers alive between epochs
    prefetch_factor=2       # Prefetch batches
)
```

### 7. Overlap data loading with computation

```python
# Use CUDA streams for data transfer
data_stream = torch.cuda.Stream()

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    data_iter = iter(dataloader)

    # Prefetch first batch
    with torch.cuda.stream(data_stream):
        batch = next(data_iter)
        data, target = batch[0].to(local_rank, non_blocking=True), \
                       batch[1].to(local_rank, non_blocking=True)

    for i, batch in enumerate(data_iter):
        # Wait for data transfer to complete
        torch.cuda.current_stream().wait_stream(data_stream)

        # Start prefetching next batch
        with torch.cuda.stream(data_stream):
            next_data, next_target = batch[0].to(local_rank, non_blocking=True), \
                                     batch[1].to(local_rank, non_blocking=True)

        # Train on current batch
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        data, target = next_data, next_target
```

---

## Multi-Node DDP Setup

### Launch Command

```bash
# On node 0 (master):
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=job1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$HOST_IP:29500 \
    train.py

# On node 1 (worker):
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=job1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$HOST_IP:29500 \
    train.py
```

### Network Configuration

```bash
# Set the network interface for NCCL
export NCCL_SOCKET_IFNAME=eth0    # or ib0 for InfiniBand

# For InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0

# For RoCE
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
```

### Bandwidth Requirements

Approximate bandwidth needed for DDP gradient synchronization:

```
Bandwidth_needed = model_size * 2 * world_size / step_time

Example:
- Model: 1B parameters (4 GB in FP32)
- 8 GPUs, 100ms per step
- Bandwidth: 4GB * 2 * 8 / 0.1s = 640 GB/s (needs NVLink or InfiniBand)
```

### Multi-Node Training Script

```python
import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    return parser.parse_args()

def setup():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def main():
    args = parse_args()
    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model = MyModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank], bucket_cap_mb=50)

    dataset = MyDataset()
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr * world_size,  # Scale lr with world_size
        weight_decay=0.01
    )

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()

        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(local_rank, non_blocking=True)
            target = target.to(local_rank, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch}/{args.epochs} complete")
            save_checkpoint(model, optimizer, epoch)

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```

---

## Fault Tolerance and Elastic Training

### Checkpoint/Restart Pattern

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

CHECKPOINT_PATH = 'checkpoint.pt'

def save_checkpoint(model, optimizer, epoch, rank):
    if rank == 0:
        # Save only the underlying model, not the DDP wrapper
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)
    dist.barrier()

def load_checkpoint(model, optimizer):
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    return 0

def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    model = DDP(MyModel().to(local_rank), device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    start_epoch = load_checkpoint(model, optimizer)

    for epoch in range(start_epoch, num_epochs):
        train_one_epoch(model, optimizer, epoch)
        save_checkpoint(model, optimizer, epoch, rank)

    dist.destroy_process_group()
```

### Elastic Training with torchrun

```bash
# Enable restarts for fault tolerance
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --max_restarts=3 \
    --rdzv_id=job1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$HOST_IP:29500 \
    train.py
```

### Handling Membership Changes

```python
from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    # Initialize with potentially changing world_size
    dist.init_process_group(backend='nccl')

    # Store rank and world_size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Load checkpoint (adjusts to new world_size)
    model, optimizer, start_epoch = load_or_init_model(rank)

    # Wrap with DDP
    model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])

    # Resume training
    for epoch in range(start_epoch, num_epochs):
        sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, epoch)
        save_checkpoint(model, optimizer, epoch, rank)
```

### Common DDP Errors and Solutions

**Error: "Gradient sync failed"**
```python
# Solution: Check NCCL connectivity
export NCCL_DEBUG=INFO
# Check firewall, network interface, etc.
```

**Error: "Expected to have finished reduction in the prior iteration before starting a new one"**
```python
# Solution: Ensure every parameter participates in every forward pass
# Option 1: Use find_unused_parameters=True
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
# Option 2: Use static_graph=True (if graph is truly static)
model = DDP(model, device_ids=[local_rank], static_graph=True)
```

**Error: "NCCL error: unhandled system error"**
```python
# Solution: Set NCCL environment variables
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=eth0  # Correct network interface
```
