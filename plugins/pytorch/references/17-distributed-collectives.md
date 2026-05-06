# PyTorch Distributed Collective Operations - Comprehensive Reference

This chapter covers all collective and point-to-point communication operations in `torch.distributed`.

## Table of Contents

1. [Point-to-Point Communication](#point-to-point-communication)
2. [AllReduce](#allreduce)
3. [AllGather](#allgather)
4. [ReduceScatter](#reducescatter)
5. [Broadcast](#broadcast)
6. [Reduce](#reduce)
7. [Scatter](#scatter)
8. [Gather](#gather)
9. [All-to-All](#all-to-all)
10. [Barrier](#barrier)
11. [Object Collectives](#object-collectives)
12. [Async Operations](#async-operations)
13. [Group Operations](#group-operations)
14. [Backend-Specific Options](#backend-specific-options)
15. [Timeout Handling](#timeout-handling)
16. [Backend-Specific Notes](#backend-specific-notes)

---

## Point-to-Point Communication

Point-to-point operations send data between exactly two processes.

### send

Sends a tensor to a destination process.

```python
torch.distributed.send(tensor, dst, group=None, tag=0)
```

**Parameters:**
- `tensor` (Tensor): The tensor to send.
- `dst` (int): Destination rank.
- `group` (ProcessGroup, optional): The process group to use.
- `tag` (int, optional): Tag for matching send/recv pairs.

```python
import torch.distributed as dist

rank = dist.get_rank()

if rank == 0:
    tensor = torch.randn(100)
    dist.send(tensor, dst=1)
elif rank == 1:
    tensor = torch.zeros(100)
    dist.recv(tensor, src=0)
```

### recv

Receives a tensor from a source process.

```python
torch.distributed.recv(tensor, src=None, group=None, tag=0)
```

**Parameters:**
- `tensor` (Tensor): The tensor to fill with received data. Must be pre-allocated with the correct shape.
- `src` (int, optional): Source rank. If None, receives from any source.
- `group` (ProcessGroup, optional): The process group to use.
- `tag` (int, optional): Tag for matching send/recv pairs.

**Returns:** The source rank of the received data.

```python
# Receive from any source
source_rank = dist.recv(tensor)

# Receive from a specific source
dist.recv(tensor, src=0)
```

### isend

Non-blocking (asynchronous) send.

```python
torch.distributed.isend(tensor, dst, group=None, tag=0)
```

**Returns:** A distributed `Work` object.

```python
# Non-blocking send
req = dist.isend(tensor, dst=1)

# Do other computation while send is in progress...
compute_something()

# Wait for send to complete
req.wait()
```

### irecv

Non-blocking (asynchronous) receive.

```python
torch.distributed.irecv(tensor, src=None, group=None, tag=0)
```

**Returns:** A distributed `Work` object.

```python
# Non-blocking receive
req = dist.irecv(tensor, src=0)

# Do other computation...
compute_something()

# Wait for receive to complete
req.wait()
```

### Overlapping Send/Recv with Computation

```python
rank = dist.get_rank()
world_size = dist.get_world_size()

# Ring communication pattern
send_rank = (rank + 1) % world_size
recv_rank = (rank - 1) % world_size

tensor = torch.randn(1000).cuda()

for step in range(world_size):
    # Start async send and recv
    send_req = dist.isend(tensor, dst=send_rank)
    recv_buf = torch.zeros_like(tensor)
    recv_req = dist.irecv(recv_buf, src=recv_rank)

    # Wait for both to complete
    send_req.wait()
    recv_req.wait()

    # Process received data
    tensor = recv_buf + rank * 0.1
```

---

## AllReduce

Reduces data across all processes and makes the result available on all processes.

```python
torch.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor` | Tensor | required | Input/output tensor. The result is written in-place. |
| `op` | ReduceOp | SUM | The reduction operation. |
| `group` | ProcessGroup | None | Process group. Default: the default group. |
| `async_op` | bool | False | If True, returns a Work handle for async operation. |

### ReduceOp Values

```python
from torch.distributed import ReduceOp

ReduceOp.SUM      # Element-wise sum
ReduceOp.PRODUCT  # Element-wise product
ReduceOp.MAX      # Element-wise maximum
ReduceOp.MIN      # Element-wise minimum
ReduceOp.BAND     # Bitwise AND (integer tensors only)
ReduceOp.BOR      # Bitwise OR (integer tensors only)
ReduceOp.BXOR     # Bitwise XOR (integer tensors only)
ReduceOp.AVG      # Element-wise average (divides by world_size)
ReduceOp.PREMUL_SUM  # Multiply by a scalar before summing
```

### Examples

```python
import torch
import torch.distributed as dist

rank = dist.get_rank()
world_size = dist.get_world_size()

# SUM: Sum all tensors across ranks
tensor = torch.tensor([rank * 1.0])  # [0.0], [1.0], [2.0], [3.0]
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
# Result on all ranks: [6.0] (0+1+2+3)

# AVG: Average all tensors
tensor = torch.tensor([rank * 1.0])
dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
# Result on all ranks: [1.5] (6.0/4)

# MAX: Element-wise maximum
tensor = torch.tensor([rank * 1.0])
dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
# Result on all ranks: [3.0]

# MIN: Element-wise minimum
tensor = torch.tensor([rank * 1.0])
dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
# Result on all ranks: [0.0]

# Async AllReduce
tensor = torch.randn(1000).cuda()
work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
# Do other work...
work.wait()

# AllReduce on a subgroup
group = dist.new_group(ranks=[0, 1])
if rank in [0, 1]:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
```

### Common Use Cases

```python
# Average gradients across GPUs (DDP does this internally)
for param in model.parameters():
    if param.grad is not None:
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

# Aggregate metrics across all ranks
total_loss = torch.tensor([local_loss]).cuda()
dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
avg_loss = total_loss.item() / dist.get_world_size()

# Find global maximum
local_max = torch.tensor([local_value]).cuda()
dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
```

---

## AllGather

Gathers tensors from all processes and distributes the combined result to all processes.

### all_gather

```python
torch.distributed.all_gather(tensor_list, tensor, group=None, async_op=False)
```

**Parameters:**
- `tensor_list` (list[Tensor]): Output list. Must be pre-allocated with the correct size and dtype. Each element corresponds to a rank.
- `tensor` (Tensor): The tensor to gather from this rank.
- `group` (ProcessGroup): Process group.
- `async_op` (bool): If True, returns async handle.

```python
rank = dist.get_rank()
world_size = dist.get_world_size()

# Each rank contributes a tensor
tensor = torch.tensor([rank * 10.0]).cuda()

# Pre-allocate output list
gathered = [torch.zeros(1).cuda() for _ in range(world_size)]

dist.all_gather(gathered, tensor)
# Rank 0: gathered = [tensor(0.), tensor(10.), tensor(20.), tensor(30.)]
# Rank 1: gathered = [tensor(0.), tensor(10.), tensor(20.), tensor(30.)]
# All ranks get the same result
```

### all_gather_into_tensor

A more efficient version that gathers into a single contiguous tensor.

```python
torch.distributed.all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False)
```

**Parameters:**
- `output_tensor` (Tensor): Output tensor. Must be pre-allocated with size `input_tensor.numel() * world_size`.
- `input_tensor` (Tensor): The tensor to gather from this rank.
- `group` (ProcessGroup): Process group.
- `async_op` (bool): Async operation flag.

```python
rank = dist.get_rank()
world_size = dist.get_world_size()

input_tensor = torch.tensor([rank, rank + 0.5]).cuda()
output_tensor = torch.zeros(2 * world_size).cuda()

dist.all_gather_into_tensor(output_tensor, input_tensor)
# output_tensor = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5] (for 4 ranks)
```

### Multi-Dimensional AllGather

```python
# Gather tensors of shape [batch_size, features]
rank = dist.get_rank()
world_size = dist.get_world_size()

features = 128
local_batch = 32
local_data = torch.randn(local_batch, features).cuda()

# Method 1: Using list
gathered = [torch.zeros(local_batch, features).cuda() for _ in range(world_size)]
dist.all_gather(gathered, local_data)
all_data = torch.cat(gathered, dim=0)  # Shape: [world_size * local_batch, features]

# Method 2: Using into_tensor (more efficient)
output = torch.zeros(world_size * local_batch * features).cuda()
dist.all_gather_into_tensor(output, local_data.flatten())
all_data = output.view(world_size * local_batch, features)
```

---

## ReduceScatter

Reduces data across all processes and scatters the result across processes.

### reduce_scatter

```python
torch.distributed.reduce_scatter(output, input_list, op=ReduceOp.SUM, group=None, async_op=False)
```

**Parameters:**
- `output` (Tensor): Output tensor. Receives this rank's portion of the reduced data.
- `input_list` (list[Tensor]): List of tensors to reduce-scatter. Each element corresponds to a destination rank.
- `op` (ReduceOp): Reduction operation. Default: SUM.
- `group` (ProcessGroup): Process group.
- `async_op` (bool): Async flag.

```python
rank = dist.get_rank()
world_size = dist.get_world_size()

# Each rank has data for all ranks
input_list = [torch.tensor([rank * 1.0]) for _ in range(world_size)]
output = torch.zeros(1)

dist.reduce_scatter(output, input_list)
# Rank 0 gets sum of input_list[0] from all ranks: 0+1+2+3 = 6
# Rank 1 gets sum of input_list[1] from all ranks: 0+1+2+3 = 6
```

### reduce_scatter_tensor

A more efficient version that operates on a single tensor.

```python
torch.distributed.reduce_scatter_tensor(output, input, op=ReduceOp.SUM, group=None, async_op=False)
```

**Parameters:**
- `output` (Tensor): Output tensor. Size must be `input.numel() / world_size`.
- `input` (Tensor): Input tensor. Must be divisible by world_size.
- `op` (ReduceOp): Reduction operation.
- `group` (ProcessGroup): Process group.
- `async_op` (bool): Async flag.

```python
rank = dist.get_rank()
world_size = dist.get_world_size()

# Each rank has a contiguous tensor with world_size chunks
input_tensor = torch.arange(world_size).float() * (rank + 1)
output_tensor = torch.zeros(1)

dist.reduce_scatter_tensor(output_tensor, input_tensor)
# Each rank gets its corresponding chunk, reduced across all ranks
```

### Use Case: FSDP Gradient Reduction

```python
# FSDP uses reduce_scatter to shard gradients
# Each rank computes the full gradient but only keeps its shard
full_grad = compute_full_gradient()
grad_shard = torch.zeros(full_grad.numel() // world_size)
dist.reduce_scatter_tensor(grad_shard, full_grad)
```

---

## Broadcast

Broadcasts a tensor from one process to all other processes.

```python
torch.distributed.broadcast(tensor, src, group=None, async_op=False)
```

**Parameters:**
- `tensor` (Tensor): The tensor to broadcast. On the source rank, this is the data to send. On other ranks, this is the buffer to receive.
- `src` (int): The source rank.
- `group` (ProcessGroup): Process group.
- `async_op` (bool): Async flag.

```python
rank = dist.get_rank()

# Broadcast from rank 0
if rank == 0:
    data = torch.tensor([1.0, 2.0, 3.0])
else:
    data = torch.zeros(3)

dist.broadcast(data, src=0)
# All ranks now have [1.0, 2.0, 3.0]

# Broadcast model parameters from rank 0
for param in model.parameters():
    dist.broadcast(param.data, src=0)

# Broadcast a configuration dict
if rank == 0:
    config_tensor = torch.tensor([learning_rate, weight_decay])
else:
    config_tensor = torch.zeros(2)
dist.broadcast(config_tensor, src=0)
```

---

## Reduce

Reduces data across all processes and sends the result to a single destination process.

```python
torch.distributed.reduce(tensor, dst, op=ReduceOp.SUM, group=None, async_op=False)
```

**Parameters:**
- `tensor` (Tensor): Input tensor. On the destination rank, contains the result after the operation.
- `dst` (int): Destination rank that receives the result.
- `op` (ReduceOp): Reduction operation.
- `group` (ProcessGroup): Process group.
- `async_op` (bool): Async flag.

```python
rank = dist.get_rank()

# Reduce to rank 0
data = torch.tensor([float(rank)])
dist.reduce(data, dst=0, op=dist.ReduceOp.SUM)
# Only rank 0 has the sum: [6.0] (for 4 ranks)
# Other ranks' data is undefined after reduce
```

### Difference from AllReduce

- `reduce`: Only the destination rank has the result.
- `all_reduce`: All ranks have the result.
- Use `reduce` when only one rank needs the result (e.g., logging, checkpointing).

---

## Scatter

Scatters a list of tensors from a source process to all processes.

```python
torch.distributed.scatter(tensor, scatter_list=None, src=0, group=None, async_op=False)
```

**Parameters:**
- `tensor` (Tensor): Output tensor. Receives this rank's scattered data.
- `scatter_list` (list[Tensor]): List of tensors to scatter. Only significant on the source rank.
- `src` (int): Source rank.
- `group` (ProcessGroup): Process group.
- `async_op` (bool): Async flag.

```python
rank = dist.get_rank()
world_size = dist.get_world_size()

if rank == 0:
    scatter_list = [torch.tensor([i * 1.0]) for i in range(world_size)]
else:
    scatter_list = None

tensor = torch.zeros(1)
dist.scatter(tensor, scatter_list=scatter_list, src=0)
# Rank 0: tensor = [0.0]
# Rank 1: tensor = [1.0]
# Rank 2: tensor = [2.0]
# Rank 3: tensor = [3.0]
```

---

## Gather

Gathers tensors from all processes to a single destination process.

```python
torch.distributed.gather(tensor, gather_list=None, dst=0, group=None, async_op=False)
```

**Parameters:**
- `tensor` (Tensor): The tensor to send from this rank.
- `gather_list` (list[Tensor]): List of tensors to receive. Only significant on the destination rank.
- `dst` (int): Destination rank.
- `group` (ProcessGroup): Process group.
- `async_op` (bool): Async flag.

```python
rank = dist.get_rank()
world_size = dist.get_world_size()

tensor = torch.tensor([rank * 1.0])

if rank == 0:
    gather_list = [torch.zeros(1) for _ in range(world_size)]
else:
    gather_list = None

dist.gather(tensor, gather_list=gather_list, dst=0)
# Rank 0: gather_list = [tensor(0.), tensor(1.), tensor(2.), tensor(3.)]
# Other ranks: gather_list is unchanged
```

---

## All-to-All

Each process sends distinct data to every other process and receives distinct data from every other process.

### all_to_all

```python
torch.distributed.all_to_all(output_list, input_list, group=None, async_op=False)
```

**Parameters:**
- `output_list` (list[Tensor]): List of tensors to receive. Must be pre-allocated.
- `input_list` (list[Tensor]): List of tensors to send. `input_list[i]` is sent to rank `i`.
- `group` (ProcessGroup): Process group.
- `async_op` (bool): Async flag.

```python
rank = dist.get_rank()
world_size = dist.get_world_size()

# Each rank sends a different tensor to each other rank
input_list = [torch.tensor([rank * 10 + i]) for i in range(world_size)]
output_list = [torch.zeros(1) for _ in range(world_size)]

dist.all_to_all(output_list, input_list)
# Rank 0: output_list = [tensor(0.), tensor(10.), tensor(20.), tensor(30.)]
# Rank 1: output_list = [tensor(1.), tensor(11.), tensor(21.), tensor(31.)]
```

### all_to_all_single

A more efficient version that operates on a single contiguous tensor.

```python
torch.distributed.all_to_all_single(output, input, output_split_sizes=None,
                                      input_split_sizes=None, group=None, async_op=False)
```

**Parameters:**
- `output` (Tensor): Output tensor.
- `input` (Tensor): Input tensor.
- `output_split_sizes` (list[int]): Sizes of each chunk to receive.
- `input_split_sizes` (list[int]): Sizes of each chunk to send.
- `group` (ProcessGroup): Process group.
- `async_op` (bool): Async flag.

```python
rank = dist.get_rank()
world_size = dist.get_world_size()

# Even split (each chunk is the same size)
chunk_size = 4
input_tensor = torch.arange(world_size * chunk_size).float() + rank * 100
output_tensor = torch.zeros(world_size * chunk_size)

dist.all_to_all_single(output_tensor, input_tensor)

# Uneven split
input_splits = [2, 3, 1, 4]  # Total must match input size
output_splits = [2, 3, 1, 4]
input_tensor = torch.randn(sum(input_splits))
output_tensor = torch.randn(sum(output_splits))

dist.all_to_all_single(output_tensor, input_tensor,
                        output_split_sizes=output_splits,
                        input_split_sizes=input_splits)
```

---

## Barrier

Synchronizes all processes. All processes block until every process has reached the barrier.

```python
torch.distributed.barrier(group=None, async_op=False, device_ids=None)
```

```python
# Simple barrier
dist.barrier()

# Async barrier
work = dist.barrier(async_op=True)
# Do other work...
work.wait()

# Barrier within a subgroup
group = dist.new_group(ranks=[0, 1, 2])
dist.barrier(group=group)
```

---

## Object Collectives

Operations for communicating arbitrary Python objects (not just tensors).

### broadcast_object_list

Broadcasts a list of Python objects from a source rank to all other ranks.

```python
torch.distributed.broadcast_object_list(object_list, src=0, group=None, device=None)
```

**Parameters:**
- `object_list` (list): List of Python objects. Modified in-place on all non-source ranks.
- `src` (int): Source rank.
- `group` (ProcessGroup): Process group.
- `device` (torch.device): Device for tensor conversion. Default: CPU.

```python
rank = dist.get_rank()

if rank == 0:
    config = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'model_name': 'resnet50'
    }
    object_list = [config, [1, 2, 3], "hello"]
else:
    object_list = [None, None, None]

dist.broadcast_object_list(object_list, src=0)
# All ranks now have the same config, list, and string
print(f"Rank {rank}: {object_list[0]}")
```

### all_gather_object

Gathers Python objects from all ranks to all ranks.

```python
torch.distributed.all_gather_object(object_list, obj, group=None)
```

**Parameters:**
- `object_list` (list): Output list. Will be filled with objects from all ranks.
- `obj` (Any): The Python object to gather from this rank.
- `group` (ProcessGroup): Process group.

```python
rank = dist.get_rank()
world_size = dist.get_world_size()

# Each rank has a different result
local_metrics = {
    'rank': rank,
    'loss': random.random(),
    'accuracy': random.random()
}

# Gather metrics from all ranks
all_metrics = [None] * world_size
dist.all_gather_object(all_metrics, local_metrics)

# All ranks have all metrics
for metrics in all_metrics:
    print(f"Rank {metrics['rank']}: loss={metrics['loss']:.4f}")
```

### Performance Notes

Object collectives use pickle serialization under the hood, so they are much slower than tensor collectives. Use them only for small control data (configuration, metrics, small lists), not for model parameters or large data.

---

## Async Operations

Most collective operations support `async_op=True`, which returns a `Work` handle for non-blocking execution.

### Work Handle

```python
# Start async operation
work = dist.all_reduce(tensor, async_op=True)

# Check if completed (non-blocking)
is_done = work.is_completed()

# Wait for completion (blocking)
work.wait()

# Wait with timeout
work.wait(timeout=datetime.timedelta(seconds=10))
```

### Multiple Async Operations

```python
# Start multiple async operations
works = []
for i in range(10):
    t = tensors[i]
    w = dist.all_reduce(t, async_op=True)
    works.append(w)

# Wait for all to complete
for w in works:
    w.wait()
```

### Overlapping Communication and Computation

```python
# Overlap gradient computation with gradient synchronization
# This is essentially what DDP does internally

# Compute gradients for later layers first
loss = model(inputs)
loss.backward(retain_graph=True)

# Start reducing gradients for later layers while computing earlier layer gradients
for name, param in reversed(list(model.named_parameters())):
    if param.grad is not None:
        work = dist.all_reduce(param.grad, async_op=True)
        # Continue backward computation for earlier layers...
        work.wait()
```

### Work Handle Methods

```python
work = dist.all_reduce(tensor, async_op=True)

# Check completion status
work.is_completed()  # Returns bool

# Wait for completion
work.wait()          # Blocks until done

# Get the source rank (for recv operations)
work.source_rank()   # Returns int

# Get result (for some operations)
work.result()        # Returns the result

# Get a future
fut = work.get_future()
```

---

## Group Operations

### new_group

Creates a new process group containing a subset of ranks.

```python
group = dist.new_group(ranks=[0, 1, 2], backend='nccl')
```

### new_subgroups

Splits the default group into subgroups of approximately equal size.

```python
# Split 8 ranks into groups of 2
subgroup, _ = dist.new_subgroups(group_size=2)
# subgroup for rank 0: contains ranks [0, 1]
# subgroup for rank 2: contains ranks [2, 3]
# etc.
```

### Group-Level Collectives

```python
# All collective operations accept a group parameter
group = dist.new_group(ranks=[0, 1, 2])

# Only ranks 0, 1, 2 participate
dist.all_reduce(tensor, group=group)
dist.broadcast(tensor, src=0, group=group)
dist.barrier(group=group)
```

### Group Rank Mapping

```python
# Get group rank from global rank
group = dist.new_group(ranks=[2, 3, 4])
global_rank = dist.get_rank()
group_rank = dist.get_group_rank(group, global_rank)
# For rank 2: group_rank = 0
# For rank 3: group_rank = 1
# For rank 4: group_rank = 2

# Get global rank from group rank
global_rank = dist.get_global_rank(group, group_rank)
```

### Monitored Barrier

```python
# Barrier that reports stuck ranks (for debugging)
dist.monitored_barrier(
    group=group,
    timeout=datetime.timedelta(seconds=30),
    wait_all_ranks=False
)
```

---

## Backend-Specific Options

### ProcessGroupNCCL Options

```python
from torch.distributed import ProcessGroupNCCL

options = ProcessGroupNCCL.Options()
options.is_high_priority_stream = False

# Backend options for init_process_group
dist.init_process_group(
    backend='nccl',
    pg_options=options
)
```

**NCCL-specific environment variables:**
```bash
# Communication tuning
NCCL_ALGO=Ring                    # Algorithm: Ring, Tree, Collnet
NCCL_PROTO=Simple                 # Protocol: Simple, LL (Low Latency)
NCCL_MIN_NCHANNELS=4              # Minimum channels
NCCL_MAX_NCHANNELS=16             # Maximum channels
NCCL_BUFFSIZE=8388608             # Buffer size in bytes

# Network configuration
NCCL_SOCKET_IFNAME=eth0           # Network interface for sockets
NCCL_IB_DISABLE=1                 # Disable InfiniBand
NCCL_IB_HCA=mlx5_0               # InfiniBand HCA
NCCL_NET_GDR_LEVEL=5              # GPU Direct RDMA level

# Topology
NCCL_P2P_LEVEL=SYS                # P2P level: SYS, NODE, PHB, PXB, PIX
NCCL_SHM_DISABLE=0                # Disable shared memory

# Debug
NCCL_DEBUG=INFO                   # Debug level: VERSION, WARN, INFO, TRACE
NCCL_DEBUG_SUBSYS=ALL             # Debug subsystems: ALL, INIT, COLL, P2P, etc.
NCCL_DEBUG_FILE=/tmp/nccl_debug.log  # Debug output file

# Performance
NCCL_MAX_NRINGS=8                 # Maximum number of rings
NCCL_SINGLE_RING_THRESHOLD=256k   # Threshold for single ring
NCCL_MIN_CTAS=1                   # Minimum CTAs
NCCL_MAX_CTAS=4                   # Maximum CTAs
```

### ProcessGroupGloo Options

```python
from torch.distributed import ProcessGroupGloo

options = ProcessGroupGloo.Options()
options._timeout = 300  # seconds

dist.init_process_group(
    backend='gloo',
    pg_options=options
)
```

---

## Timeout Handling

All collective operations respect the timeout set during `init_process_group`.

### Setting Timeout

```python
import datetime

# Global timeout for all operations
dist.init_process_group(
    backend='nccl',
    timeout=datetime.timedelta(minutes=30)
)

# Per-group timeout
group = dist.new_group(
    ranks=[0, 1, 2],
    timeout=datetime.timedelta(minutes=10)
)
```

### Handling Timeouts

```python
import datetime

try:
    dist.all_reduce(tensor)
except RuntimeError as e:
    if "Timeout" in str(e):
        print(f"Rank {dist.get_rank()}: Operation timed out")
        # Handle timeout (retry, checkpoint, abort)
    else:
        raise
```

### NCCL Async Error Handling

```python
# Enable async error handling (recommended for production)
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'

# This causes NCCL errors to be raised as Python exceptions
# rather than silently corrupting training
```

---

## Backend-Specific Notes

### NCCL Notes

1. **GPU Tensors Only**: NCCL operations require all tensors to be on CUDA devices.
2. **Same Device**: All tensors in a single operation should be on the same device.
3. **CUDA Stream**: NCCL operations respect CUDA streams. Use `torch.cuda.current_stream()` for proper synchronization.
4. **Error Recovery**: NCCL errors are typically fatal. Enable `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` for better error reporting.

```python
# NCCL requires CUDA tensors
tensor = torch.randn(100).cuda()  # Must be on CUDA
dist.all_reduce(tensor)

# This will fail with NCCL:
tensor_cpu = torch.randn(100)  # CPU tensor
dist.all_reduce(tensor_cpu)  # RuntimeError with NCCL backend
```

### Gloo Notes

1. **CPU and GPU Support**: Gloo supports both CPU and GPU tensors.
2. **Slower for GPU**: Gloo is significantly slower than NCCL for GPU communication.
3. **Better Error Messages**: Gloo provides more descriptive error messages for debugging.
4. **Useful for CPU Workloads**: Gloo is the recommended backend for CPU-only distributed training.

```python
# Gloo works with both CPU and GPU tensors
tensor_cpu = torch.randn(100)          # CPU tensor
dist.all_reduce(tensor_cpu)            # Works with Gloo

tensor_gpu = torch.randn(100).cuda()   # GPU tensor
dist.all_reduce(tensor_gpu)            # Also works with Gloo (slower than NCCL)
```

### Mixed Backend Usage

```python
# Use NCCL for GPU operations and Gloo for CPU operations
dist.init_process_group(backend='nccl')

# Create a Gloo group for CPU operations
cpu_group = dist.new_group(backend='gloo')

# GPU collective (uses NCCL, the default)
gpu_tensor = torch.randn(100).cuda()
dist.all_reduce(gpu_tensor)

# CPU collective (uses Gloo via the custom group)
cpu_tensor = torch.randn(100)
dist.all_reduce(cpu_tensor, group=cpu_group)
```

### Performance Comparison

```
AllReduce performance for 1M float tensor on 8 GPUs:

NCCL (NVLink):     ~10-20 us    (fastest, GPU only)
NCCL (PCIe):       ~50-100 us   (GPU only)
Gloo (GPU):        ~200-500 us  (GPU, much slower)
Gloo (CPU):        ~100-300 us  (CPU-only option)
```

---

## Complete Example: Custom Training with Collectives

```python
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F

def train_with_custom_collectives(model, dataloader, optimizer):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])

    model = model.to(local_rank)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for data, target in dataloader:
            data = data.to(local_rank)
            target = target.to(local_rank)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()

            # Manually all-reduce gradients (like DDP does internally)
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

            optimizer.step()

            # Track metrics locally
            total_loss += loss.item() * data.size(0)
            total_correct += (output.argmax(1) == target).sum().item()
            total_samples += data.size(0)

        # Aggregate metrics across all ranks
        metrics = torch.tensor([total_loss, total_correct, total_samples]).cuda()
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        global_loss = metrics[0].item() / metrics[2].item()
        global_acc = metrics[1].item() / metrics[2].item()

        if rank == 0:
            print(f"Epoch {epoch}: Loss={global_loss:.4f}, Acc={global_acc:.4f}")

        # Broadcast model from rank 0 (ensure consistency)
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

    return model
```
