# Communication Primitives

## Overview

The DeepSpeed communication module (`deepspeed/comm/`) provides a unified, hardware-agnostic communication layer that abstracts distributed collective and point-to-point operations across multiple backend implementations. It serves as the foundational communication infrastructure for all distributed training and inference workflows in DeepSpeed, including ZeRO optimization, pipeline parallelism, tensor parallelism, and MoE expert parallelism.

The module is designed to:

- Offer a single API surface for all communication primitives regardless of the underlying backend
- Automatically select the optimal backend based on hardware and runtime environment
- Provide detailed profiling and logging of all communication operations
- Support advanced patterns such as coalesced collectives for overlapping computation and communication
- Manage process group lifecycle and world information

## Source Code Structure

```
deepspeed/comm/
    __init__.py               # Module entry point, exports public API
    comm.py                   # Core communication API implementation
    torch.py                  # PyTorch distributed backend adapter
    ccl.py                    # Intel oneCCL backend adapter
    mpi.py                    # MPI backend adapter
    backend.py                # Abstract backend interface
    utils.py                  # Utility functions
    coalesced_collectives.py  # Overlapping communication with computation
```

## Backend Architecture

### Supported Backends

DeepSpeed supports four primary communication backends:

| Backend | Hardware Platform | Library | Configuration Value |
|---------|-------------------|---------|---------------------|
| **NCCL** | NVIDIA GPUs | libnccl.so | `"nccl"` |
| **CCL** | Intel CPUs/Xeons, XPU | liboneccl.so | `"ccl"` |
| **MPI** | Any (via MPI implementation) | libmpi.so | `"mpi"` |
| **HCCL** | Huawei Ascend NPUs | libhccl.so | `"hccl"` |

### Backend Selection Priority

When `configure()` is called, DeepSpeed selects the backend using the following priority:

1. **Explicit backend** specified in the configuration (`communication_config.backend`)
2. **NCCL** if `torch.distributed` is already initialized with NCCL
3. **CCL** if running on Intel hardware with oneCCL available
4. **MPI** if MPI environment variables are detected
5. **Default**: PyTorch native distributed (which itself defaults to NCCL on CUDA)

### Backend Adapter Pattern

Each backend adapter follows the same interface pattern:

```python
# deepspeed/comm/torch.py - PyTorch/NCCL backend
class TorchBackend:
    def allreduce(self, tensor, op=None, group=None, async_op=False):
        return torch.distributed.all_reduce(
            tensor, op=op, group=group, async_op=async_op
        )

# deepspeed/comm/ccl.py - Intel CCL backend
class CCLBackend:
    def allreduce(self, tensor, op=None, group=None, async_op=False):
        return torch_ccl.all_reduce(
            tensor, op=op, group=group, async_op=async_op
        )
```

## Initialization: configure()

The `configure()` function is the primary entry point for initializing the communication module. It must be called before any collective operations are performed.

### Function Signature

```python
deepspeed.comm.configure(
    deepspeed_config=None,
    enabled=True,
    prof_all=False,
    prof_ops=None,
    verbose=False,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `deepspeed_config` | `dict` or `None` | `None` | DeepSpeed configuration dictionary. Contains communication-specific settings under the `"communication"` key. |
| `enabled` | `bool` | `True` | Whether to enable the communication module. If `False`, operations are no-ops. |
| `prof_all` | `bool` | `False` | Profile all communication operations automatically. |
| `prof_ops` | `list[str]` or `None` | `None` | Specific operations to profile (e.g., `["allreduce", "allgather"]`). |
| `verbose` | `bool` | `False` | Enable verbose logging of communication events. |

### Configuration-Driven Initialization

When DeepSpeed is initialized via `deepspeed.initialize()`, the communication module is automatically configured from the `ds_config` dictionary:

```json
{
    "communication": {
        "enabled": true,
        "backend": "nccl",
        "prof_all": false,
        "prof_ops": ["allreduce"],
        "comms_logger": {
            "enabled": true,
            "verbose": false,
            "prof_all": false,
            "debug": false
        },
        "communication_data_type": "auto"
    }
}
```

### Initialization Flow

```
deepspeed.initialize()
    |
    v
ds.engine = DeepSpeedEngine(...)
    |
    v
self._configure_comms()
    |
    v
deepspeed.comm.configure(deepspeed_config=config)
    |
    +-- Read communication config from ds_config
    +-- Determine backend (NCCL/CCL/MPI/HCCL)
    +-- Initialize backend adapter
    +-- Set up CommsLogger if configured
    +-- Configure communication data type
    +-- Store caller function tracking
```

## Communication Primitives

### All-Reduce

Performs a reduction operation (sum, product, min, max) across all ranks and distributes the result to all ranks.

```python
deepspeed.comm.allreduce(
    tensor,
    op=deepspeed.comm.ReduceOp.SUM,  # SUM, PRODUCT, MIN, MAX, BAND, BOR, BXOR
    group=None,                        # Process group (default: global group)
    async_op=False                     # Asynchronous operation
)
```

**Parameters:**
- `tensor` (`torch.Tensor`): Input tensor. Modified in-place with the result.
- `op` (`ReduceOp`): Reduction operation. Default is `SUM`.
- `group` (`ProcessGroup` or `None`): The process group to operate on. `None` uses the default global group.
- `async_op` (`bool`): If `True`, returns a distributed handle for waiting later.

**Returns:** A distributed work handle if `async_op=True`, otherwise `None`.

**Behavior:**
```
Before allreduce (op=SUM):
  Rank 0: [1, 2, 3]    Rank 1: [4, 5, 6]    Rank 2: [7, 8, 9]

After allreduce:
  Rank 0: [12, 15, 18]  Rank 1: [12, 15, 18]  Rank 2: [12, 15, 18]
```

**Usage in DeepSpeed:** Used extensively for gradient averaging in data-parallel training, ZeRO stage 0/1 gradient synchronization, and tensor parallel communication.

```python
# Gradient averaging in DDP
for param in model.parameters():
    if param.grad is not None:
        deepspeed.comm.allreduce(param.grad.data, op=deepspeed.comm.ReduceOp.SUM)
        param.grad.data.div_(deepspeed.comm.get_world_size())
```

### All-Gather

Gathers tensors from all ranks and concatenates them along the first dimension, distributing the full result to all ranks.

```python
deepspeed.comm.allgather(
    tensor_list,    # Output: list of tensors (one per rank)
    tensor,         # Input: tensor from this rank
    group=None,
    async_op=False
)
```

**Parameters:**
- `tensor_list` (`list[torch.Tensor]`): Pre-allocated list of tensors to receive gathered results. Must have length equal to the group size.
- `tensor` (`torch.Tensor`): The tensor to send from this rank.
- `group` (`ProcessGroup` or `None`): Process group.
- `async_op` (`bool`): Asynchronous operation flag.

**Behavior:**
```
Before allgather:
  Rank 0: [1, 2]    Rank 1: [3, 4]    Rank 2: [5, 6]

After allgather:
  Rank 0: [1, 2, 3, 4, 5, 6]
  Rank 1: [1, 2, 3, 4, 5, 6]
  Rank 2: [1, 2, 3, 4, 5, 6]
```

**Usage in DeepSpeed:** ZeRO stage 3 parameter gathering before forward/backward passes, tensor parallel sequence parallelism, MoE expert output aggregation.

```python
# ZeRO-3 parameter gathering
param_partition = get_my_partition(param)
gathered_tensors = [torch.empty_like(param_partition) for _ in range(world_size)]
deepspeed.comm.allgather(gathered_tensors, param_partition)
full_param = torch.cat(gathered_tensors, dim=0)
```

### Broadcast

Broadcasts a tensor from one rank (the source) to all other ranks.

```python
deepspeed.comm.broadcast(
    tensor,
    src=0,           # Source rank
    group=None,
    async_op=False
)
```

**Parameters:**
- `tensor` (`torch.Tensor`): Data to broadcast (if `src` rank) or receive (if non-src rank). Modified in-place on non-src ranks.
- `src` (`int`): The source rank that sends the data.
- `group` (`ProcessGroup` or `None`): Process group.
- `async_op` (`bool`): Asynchronous operation flag.

**Behavior:**
```
Before broadcast (src=0):
  Rank 0: [1, 2, 3]    Rank 1: [0, 0, 0]    Rank 2: [0, 0, 0]

After broadcast:
  Rank 0: [1, 2, 3]    Rank 1: [1, 2, 3]    Rank 2: [1, 2, 3]
```

**Usage in DeepSpeed:** Model parameter synchronization at initialization, broadcasting optimizer states in ZeRO stage 1, model weight synchronization for inference.

### Reduce

Reduces data across all ranks and sends the result to a specified destination rank.

```python
deepspeed.comm.reduce(
    tensor,
    dst=0,           # Destination rank
    op=deepspeed.comm.ReduceOp.SUM,
    group=None,
    async_op=False
)
```

**Parameters:**
- `tensor` (`torch.Tensor`): Input tensor. Modified in-place on the destination rank.
- `dst` (`int`): The destination rank that receives the result.
- `op` (`ReduceOp`): Reduction operation.
- `group` (`ProcessGroup` or `None`): Process group.
- `async_op` (`bool`): Asynchronous operation flag.

**Behavior:**
```
Before reduce (dst=0, op=SUM):
  Rank 0: [1, 2, 3]    Rank 1: [4, 5, 6]    Rank 2: [7, 8, 9]

After reduce:
  Rank 0: [12, 15, 18]  Rank 1: [4, 5, 6]    Rank 2: [7, 8, 9]
```

**Usage in DeepSpeed:** Aggregating metrics to a master rank for logging, collecting loss values for checkpointing.

### Reduce-Scatter

Reduces data across all ranks, then scatters the reduced result in equal chunks to each rank.

```python
deepspeed.comm.reduce_scatter(
    output,          # Output tensor (smaller, rank-sized chunk)
    input_list,      # Input: list of tensors or single tensor
    op=deepspeed.comm.ReduceOp.SUM,
    group=None,
    async_op=False
)
```

**Parameters:**
- `output` (`torch.Tensor`): Pre-allocated output tensor for this rank's chunk.
- `input_list` (`list[torch.Tensor]` or `torch.Tensor`): Input tensors to reduce-scatter. If a list, must have length equal to world size.
- `op` (`ReduceOp`): Reduction operation.
- `group` (`ProcessGroup` or `None`): Process group.
- `async_op` (`bool`): Asynchronous operation flag.

**Behavior:**
```
Before reduce_scatter (world_size=3, op=SUM):
  Rank 0: [1,2,  3,4,  5,6]    Rank 1: [7,8,  9,10,  11,12]    Rank 2: [13,14, 15,16, 17,18]

After reduce_scatter:
  Rank 0: [21, 24]    (1+7+13, 2+8+14)
  Rank 1: [27, 30]    (3+9+15, 4+10+16)
  Rank 2: [33, 36]    (5+11+17, 6+12+18)
```

**Usage in DeepSpeed:** ZeRO stage 2/3 gradient partitioning after backward pass, MoE gradient reduction, tensor parallel gradient aggregation.

```python
# ZeRO-2 gradient partitioning
grad_flat = torch.cat([p.grad.flatten() for p in model.parameters()])
my_grad_partition = torch.empty(grad_flat.numel() // world_size, device=grad_flat.device)
deepspeed.comm.reduce_scatter(my_grad_partition, [grad_flat] * world_size)
```

### Send

Sends a tensor to a specific destination rank (point-to-point communication).

```python
deepspeed.comm.send(
    tensor,
    dst,             # Destination rank
    group=None,
    tag=0            # Message tag for matching
)
```

**Parameters:**
- `tensor` (`torch.Tensor`): Tensor to send.
- `dst` (`int`): Destination rank.
- `group` (`ProcessGroup` or `None`): Process group.
- `tag` (`int`): Tag for matching send/recv pairs.

**Usage in DeepSpeed:** Pipeline parallelism stage-to-stage tensor transfer, MoE expert token routing to remote GPUs.

### Recv

Receives a tensor from a specific source rank (point-to-point communication).

```python
deepspeed.comm.recv(
    tensor,
    src=None,        # Source rank (None = any source)
    group=None,
    tag=0            # Message tag for matching
)
```

**Parameters:**
- `tensor` (`torch.Tensor`): Pre-allocated tensor to receive data into.
- `src` (`int` or `None`): Source rank. `None` receives from any source.
- `group` (`ProcessGroup` or `None`): Process group.
- `tag` (`int`): Tag for matching send/recv pairs.

**Usage in DeepSpeed:** Pipeline parallelism receiving activations from previous stage, MoE receiving tokens from other GPUs.

### Scatter

Scatters a list of tensors from the source rank to all ranks (each rank receives one tensor).

```python
deepspeed.comm.scatter(
    tensor,          # Output: receives one chunk
    scatter_list,    # Input: list of tensors (only used on src rank)
    src=0,
    group=None,
    async_op=False
)
```

**Parameters:**
- `tensor` (`torch.Tensor`): Output tensor to receive scattered data.
- `scatter_list` (`list[torch.Tensor]`): List of tensors to scatter (only significant on `src` rank).
- `src` (`int`): Source rank.
- `group` (`ProcessGroup` or `None`): Process group.
- `async_op` (`bool`): Asynchronous operation flag.

**Behavior:**
```
Before scatter (src=0):
  Rank 0 scatter_list: [[1,2], [3,4], [5,6]]
  Rank 1 tensor: [0, 0]
  Rank 2 tensor: [0, 0]

After scatter:
  Rank 0: [1, 2]    Rank 1: [3, 4]    Rank 2: [5, 6]
```

### Gather

Gathers tensors from all ranks to the destination rank.

```python
deepspeed.comm.gather(
    tensor,
    gather_list,     # Output: list of tensors (only used on dst rank)
    dst=0,
    group=None,
    async_op=False
)
```

**Parameters:**
- `tensor` (`torch.Tensor`): Tensor to send from this rank.
- `gather_list` (`list[torch.Tensor]` or `None`): Pre-allocated list to receive gathered tensors (only significant on `dst` rank). Must be `None` on non-dst ranks.
- `dst` (`int`): Destination rank.
- `group` (`ProcessGroup` or `None`): Process group.
- `async_op` (`bool`): Asynchronous operation flag.

**Behavior:**
```
Before gather (dst=0):
  Rank 0: [1, 2]    Rank 1: [3, 4]    Rank 2: [5, 6]

After gather (on Rank 0):
  gather_list: [[1,2], [3,4], [5,6]]
```

### Barrier

Synchronizes all ranks. Each rank blocks until all ranks have reached the barrier.

```python
deepspeed.comm.barrier(
    group=None,
    device_ids=None    # Device IDs for barrier (NCCL-specific)
)
```

**Parameters:**
- `group` (`ProcessGroup` or `None`): Process group.
- `device_ids** (`list[int]` or `None`): List of device IDs for NCCL barrier. Required for NCCL backend in some cases.

**Usage in DeepSpeed:** Ensuring all ranks have completed initialization before training begins, synchronizing before checkpoint save/load, coordinating between pipeline stages.

## Communication Data Type Selection

### Overview

DeepSpeed can automatically convert tensor data types for communication operations to reduce bandwidth and improve performance. This is controlled by the `communication_data_type` configuration parameter.

### Configuration

```json
{
    "communication": {
        "communication_data_type": "auto"
    }
}
```

### Supported Values

| Value | Description |
|-------|-------------|
| `"auto"` | Automatically selects based on model data type. If model is FP16/BF16, uses the same type. If model is FP32, uses FP32. |
| `"fp32"` | Forces all communication in FP32 regardless of model type. Ensures numerical precision but uses more bandwidth. |
| `"fp16"` | Forces all communication in FP16. Reduces bandwidth by 2x compared to FP32. |
| `"bf16"` | Forces all communication in BF16. Reduces bandwidth by 2x with better dynamic range than FP16. |
| `None` | No data type conversion; uses the tensor's native data type. |

### Implementation Details

When a communication data type is specified, the communication module:

1. Casts the input tensor(s) to the specified type before the collective operation
2. Performs the collective operation in the reduced precision
3. Casts the result back to the original tensor dtype (for in-place operations)

```python
# Internal implementation (simplified)
def allreduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    original_dtype = tensor.dtype
    if comm_dtype is not None and tensor.dtype != comm_dtype:
        tensor_to_comm = tensor.to(comm_dtype)
        result = backend.allreduce(tensor_to_comm, op=op, group=group, async_op=async_op)
        if not async_op:
            tensor.copy_(tensor_to_comm.to(original_dtype))
        return result
    return backend.allreduce(tensor, op=op, group=group, async_op=async_op)
```

### Performance Impact

| Data Type | Bandwidth Reduction | Precision Loss Risk | Recommended Use Case |
|-----------|--------------------|--------------------|----------------------|
| FP32 | Baseline (1x) | None | Debugging, high-precision requirements |
| FP16 | 2x | Moderate | FP16 training, gradient communication |
| BF16 | 2x | Low (wider exponent) | BF16 training, modern GPUs |
| Auto | Varies | Minimal | Default recommendation |

## Coalesced Collectives

### Overview

The coalesced collectives module (`deepspeed/comm/coalesced_collectives.py`) provides optimized collective operations that batch multiple small tensors into a single larger communication, reducing the number of collective calls and their associated latency overhead. This is particularly important for ZeRO-3 where many small parameter tensors must be gathered/scattered.

### Core Functions

#### allgather_coalesced

Gathers multiple tensors from all ranks in a single coalesced operation.

```python
deepspeed.comm.allgather_coalesced(
    output_tensors_list,   # List of lists of output tensors (one list per rank)
    input_tensors_list,    # List of input tensors from this rank
    world_size,            # Total number of ranks
    group=None
)
```

**Parameters:**
- `output_tensors_list` (`list[list[torch.Tensor]]`): Pre-allocated nested list. `output_tensors_list[rank][tensor_idx]` receives the tensor from `rank`.
- `input_tensors_list` (`list[torch.Tensor]`): The tensors to send from this rank.
- `world_size` (`int`): Number of ranks participating.
- `group` (`ProcessGroup` or `None`): Process group.

**Implementation Strategy:**

1. **Flatten and Concatenate**: All input tensors are flattened and concatenated into a single contiguous buffer.
2. **Single All-Gather**: One `allgather()` call is performed on the concatenated buffer.
3. **Split and Reshape**: The gathered buffer is split back into individual tensors with their original shapes.

This approach reduces N all-gather calls to a single call, where N is the number of tensors.

```python
# Without coalescing (N collective calls)
for tensor in tensors:
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    deepspeed.comm.allgather(gathered, tensor)

# With coalescing (1 collective call)
output_list = [[torch.empty_like(t) for t in tensors] for _ in range(world_size)]
deepspeed.comm.allgather_coalesced(output_list, tensors, world_size)
```

#### reduce_scatter_coalesced

Reduces and scatters multiple tensors in a single coalesced operation.

```python
deepspeed.comm.reduce_scatter_coalesced(
    output_tensors,        # List of output tensors (one per rank partition)
    input_tensors,         # List of input tensors
    world_size,
    group=None
)
```

**Usage in DeepSpeed:** ZeRO-3 gradient reduce-scatter, where gradients from many parameters must be partitioned across ranks.

### Performance Benefits

| Scenario | Without Coalescing | With Coalescing | Speedup |
|----------|-------------------|-----------------|---------|
| 100 small tensors (ZeRO-3) | 100 all-gather calls | 1 all-gather call | 10-50x |
| 10 medium tensors | 10 all-gather calls | 1 all-gather call | 3-8x |
| 1 large tensor | 1 all-gather call | 1 all-gather call | ~1x |

## CommsLogger: Communication Profiling

### Overview

The `CommsLogger` class provides comprehensive profiling of all communication operations, collecting timing, volume, and call-count statistics. It integrates with the DeepSpeed timer system for accurate measurement.

### Configuration

```json
{
    "comms_logger": {
        "enabled": true,
        "verbose": false,
        "prof_all": true,
        "debug": false
    }
}
```

### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable/disable communication logging. |
| `verbose` | `bool` | `false` | Print detailed information for each communication call. |
| `prof_all` | `bool` | `false` | Profile all operations by default. |
| `debug` | `bool` | `false` | Enable debug-level output with call stacks. |

### Logged Metrics

For each communication operation, CommsLogger records:

| Metric | Description |
|--------|-------------|
| `count` | Number of times the operation was called |
| `total_time_ms` | Total time spent in the operation across all calls (milliseconds) |
| `avg_time_ms` | Average time per call (milliseconds) |
| `min_time_ms` | Minimum time observed for a single call |
| `max_time_ms` | Maximum time observed for a single call |
| `total_size_MB` | Total data volume transferred (megabytes) |
| `avg_size_MB` | Average data volume per call |

### Programmatic Usage

```python
import deepspeed.comm as dist

# Enable profiling
dist.configure(prof_all=True)

# Perform communication operations
for step in range(100):
    dist.allreduce(gradient_tensor)

# Retrieve profiling results
from deepspeed.comm.comm import CommsLogger
logger = CommsLogger()

# Print summary
# Output format:
# Op            Count  Total(ms)  Avg(ms)  Min(ms)  Max(ms)  Total(MB)  Avg(MB)
# allreduce     100    1234.5     12.35    10.2     15.7     5120.0     51.2
# allgather     50     456.7      9.13     8.1      11.3     2048.0     40.96
# broadcast     10     23.4       2.34     2.1      2.8      256.0      25.6
```

### Verbose Mode

When verbose mode is enabled, each communication call prints:

```
[Rank 0] allreduce: shape=[4096, 4096], dtype=torch.float16, op=SUM, time=12.35ms, size=32.0MB
[Rank 0] allgather: shape=[1024, 1024], dtype=torch.float16, time=9.13ms, size=2.0MB
```

### Debug Mode

In debug mode, additional information is logged:

- Full call stack trace showing which function triggered the communication
- Process group information
- Tensor device and memory layout
- Synchronization barriers

## Process Group Management

### Default Process Group

When DeepSpeed initializes communication, it creates or uses the default global process group. All collective operations without an explicit `group` parameter use this default group.

```python
# The default group encompasses all ranks
global_group = deepspeed.comm.group.WORLD  # Same as group=None
```

### Creating Sub-Groups

DeepSpeed supports creating sub-groups of ranks for hierarchical communication patterns:

```python
# Create a sub-group for intra-node communication
ranks_per_node = 8
local_ranks = list(range(node_id * ranks_per_node, (node_id + 1) * ranks_per_node))
intra_node_group = deepspeed.comm.new_group(ranks=local_ranks)

# Create a sub-group for inter-node communication (one rank per node)
inter_node_ranks = list(range(0, world_size, ranks_per_node))
inter_node_group = deepspeed.comm.new_group(ranks=inter_node_ranks)
```

### Process Group Functions

```python
# Create a new process group
group = deepspeed.comm.new_group(ranks=[0, 1, 2, 3])

# Get the rank within a specific group
rank_in_group = deepspeed.comm.get_rank(group=group)

# Get the size of a specific group
group_size = deepspeed.comm.get_world_size(group=group)

# Get the global rank from a group rank
global_rank = deepspeed.comm.get_global_rank(group, group_rank)
```

### Hierarchical Communication

DeepSpeed uses hierarchical process groups for optimized communication in multi-node settings:

```
Global Group (all 16 ranks)
  |
  +-- Node 0 Intra-Node Group (ranks 0-7)
  +-- Node 1 Intra-Node Group (ranks 8-15)
  +-- Inter-Node Group (ranks 0, 8)  -- one representative per node
```

This enables:
- **Intra-node allreduce** using fast NVLink connections
- **Inter-node allreduce** across nodes using PCIe/NVLink uplinks
- **Hierarchical allreduce**: Reduce within nodes, then across nodes, then broadcast back

## World Info Management

### Global Information Functions

```python
# Get the rank of the current process
rank = deepspeed.comm.get_rank()
# Returns: int (0-indexed global rank)

# Get the total number of processes
world_size = deepspeed.comm.get_world_size()
# Returns: int (total number of distributed processes)

# Get the local rank within the current node
local_rank = deepspeed.comm.get_local_rank()
# Returns: int (0-indexed rank within the node)
```

### Determining the Local Rank

DeepSpeed determines the local rank through several mechanisms:

1. **Environment variable**: `LOCAL_RANK` or `OMPI_COMM_WORLD_LOCAL_RANK`
2. **CUDA device mapping**: If `torch.cuda.device_count()` equals the per-node rank count
3. **Explicit configuration**: From `deepspeed_config["local_rank"]`

```python
# Internal logic (simplified)
def get_local_rank():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        return int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    # Fallback: compute from global rank and GPUs per node
    return get_rank() % torch.cuda.device_count()
```

### World Info Utility

```python
# Get all world info at once
world_info = {
    "rank": deepspeed.comm.get_rank(),
    "local_rank": deepspeed.comm.get_local_rank(),
    "world_size": deepspeed.comm.get_world_size(),
    "local_world_size": torch.cuda.device_count(),
    "master_addr": os.environ.get("MASTER_ADDR", "localhost"),
    "master_port": os.environ.get("MASTER_PORT", "29500"),
}
```

## Timer Integration

### Communication Timing

DeepSpeed integrates its communication module with the timer system (`deepspeed.utils.timer`) for precise measurement of communication overhead.

```python
# Timer integration in communication operations
class TimedCommunication:
    def __init__(self):
        self.timer = SynchronizedWallClockTimer()

    def timed_allreduce(self, tensor, op=ReduceOp.SUM):
        self.timer.start("allreduce")
        result = deepspeed.comm.allreduce(tensor, op=op)
        self.timer.stop("allreduce")
        return result
```

### Wall Clock Timer

The `SynchronizedWallClockTimer` provides synchronized timing across ranks:

```python
from deepspeed.utils.timer import SynchronizedWallClockTimer

timer = SynchronizedWallClockTimer()

# Start timing a phase
timer.start("forward_comm")

# ... communication operations ...

# Stop timing
timer.stop("forward_comm")

# Retrieve timing summary
summary = timer.get_mean()
# {"forward_comm": {"total_time_ms": 123.4, "avg_time_ms": 12.34, "count": 10}}
```

## Caller Function Tracking

### Overview

DeepSpeed can track which function initiated each communication operation, providing detailed attribution for profiling and debugging. This is controlled by the `prof_ops` and `prof_all` configuration options.

### Configuration

```json
{
    "communication": {
        "prof_ops": ["allreduce"],
        "comms_logger": {
            "enabled": true,
            "debug": true
        }
    }
}
```

### Tracking Output

When caller tracking is enabled, each communication operation logs:

```
[Rank 0] allreduce called from:
  File "deepspeed/runtime/zero/stage3.py", line 1234, in _reduce_scatter_gradients
    deepspeed.comm.reduce_scatter_coalesced(...)
  File "deepspeed/runtime/zero/stage3.py", line 567, in backward
    self._reduce_scatter_gradients(buffer=grad_buffer)
  File "train.py", line 89, in training_step
    model_engine.backward(loss)
```

### Usage for Performance Analysis

```python
# Identify which functions are the top communicators
from deepspeed.comm.comm import get_comm_call_tracker

tracker = get_comm_call_tracker()
summary = tracker.get_summary()

# Output:
# Top communication callers:
#   deepspeed.zero3._reduce_scatter_gradients: 45.2% of comm time
#   deepspeed.zero3._allgather_params:        32.1% of comm time
#   deepspeed.pipe._send_activations:         12.3% of comm time
#   deepspeed.moe._dispatch_tokens:           10.4% of comm time
```

## Asynchronous Operations

### Using async_op

Most collective operations support the `async_op=True` flag, which returns a handle for later synchronization. This enables overlapping communication with computation.

```python
# Synchronous (blocking)
deepspeed.comm.allreduce(tensor)  # Blocks until complete

# Asynchronous (non-blocking)
handle = deepspeed.comm.allreduce(tensor, async_op=True)

# Do useful computation while communication is in flight
compute_result = some_computation(other_tensor)

# Wait for communication to complete
handle.wait()
```

### Wait Sets and Batching

```python
# Launch multiple async operations
handles = []
for grad in gradients:
    h = deepspeed.comm.allreduce(grad, async_op=True)
    handles.append(h)

# Wait for all to complete
for h in handles:
    h.wait()

# Alternative: batch wait (more efficient)
deepspeed.comm.barrier()  # Implicitly waits for all pending operations
```

## Integration with DeepSpeed Engine

### Automatic Initialization

When `deepspeed.initialize()` is called, the communication module is automatically configured:

```python
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config_params=ds_config,  # Contains communication config
)
# Communication is now fully initialized and ready to use
```

### ZeRO Communication Patterns

```
ZeRO Stage 0 (DDP):
  Forward:  no communication
  Backward: allreduce(gradients)

ZeRO Stage 1:
  Forward:  no communication
  Backward: reduce_scatter(gradients)

ZeRO Stage 2:
  Forward:  no communication
  Backward: reduce_scatter(gradients)  # gradient partitioning

ZeRO Stage 3:
  Forward:  allgather(parameters)      # per-layer parameter gathering
  Backward: reduce_scatter(gradients)   # gradient partitioning
            allgather(parameters)       # per-layer for backward
```

### Pipeline Parallelism Communication

```
Stage 0 -> Stage 1 -> Stage 2 -> Stage 3

Each stage boundary uses:
  send(output_tensor, dst=next_stage)
  recv(input_tensor, src=previous_stage)
```

### Tensor Parallelism Communication

```
All-Reduce Linear (column parallel):
  Forward:  allreduce(output)     # after column-parallel linear
  Backward: no communication      # gradients naturally aligned

All-Reduce Linear (row parallel):
  Forward:  no communication      # output already reduced
  Backward: allreduce(grad_input) # after row-parallel linear

Sequence Parallel:
  Forward:  allgather(sequence_tokens)  / reduce_scatter(sequence_tokens)
  Backward: reduce_scatter(sequence_tokens) / allgather(sequence_tokens)
```

## Configuration Examples

### Minimal Configuration (NCCL)

```json
{
    "train_batch_size": 32,
    "zero_optimization": {
        "stage": 2
    },
    "communication": {
        "enabled": true
    }
}
```

### Communication Profiling Enabled

```json
{
    "train_batch_size": 32,
    "zero_optimization": {
        "stage": 3
    },
    "communication": {
        "enabled": true,
        "backend": "nccl",
        "communication_data_type": "fp16",
        "prof_all": true,
        "comms_logger": {
            "enabled": true,
            "verbose": true,
            "prof_all": true,
            "debug": false
        }
    }
}
```

### Intel XPU with CCL Backend

```json
{
    "train_batch_size": 32,
    "zero_optimization": {
        "stage": 2
    },
    "communication": {
        "enabled": true,
        "backend": "ccl",
        "communication_data_type": "bf16"
    }
}
```

### Huawei Ascend NPU with HCCL

```json
{
    "train_batch_size": 32,
    "zero_optimization": {
        "stage": 2
    },
    "communication": {
        "enabled": true,
        "backend": "hccl",
        "communication_data_type": "fp16"
    }
}
```

### Complete Communication Configuration

```json
{
    "train_batch_size": 256,
    "gradient_accumulation_steps": 4,
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "bf16": {
        "enabled": true
    },
    "communication": {
        "enabled": true,
        "backend": "nccl",
        "communication_data_type": "bf16",
        "prof_all": false,
        "prof_ops": ["allreduce", "allgather", "reduce_scatter"],
        "comms_logger": {
            "enabled": true,
            "verbose": false,
            "prof_all": true,
            "debug": false
        }
    },
    "gradient_accumulation_steps": 4
}
```

### Programmatic API Usage

```python
import deepspeed
import deepspeed.comm as dist

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    args=args,
    config_params=ds_config,
)

# Use communication primitives directly
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = dist.get_local_rank()

# All-reduce gradients
dist.allreduce(gradient_tensor, op=dist.ReduceOp.SUM)

# All-gather parameters (ZeRO-3 pattern)
params = [torch.empty(shape) for _ in range(world_size)]
dist.allgather(params, my_param_partition)

# Reduce-scatter gradients (ZeRO-2/3 pattern)
my_grad = torch.empty(grad_size)
dist.reduce_scatter(my_grad, all_grads)

# Barrier for synchronization
dist.barrier()

# Point-to-point for pipeline parallelism
if dist.get_rank() > 0:
    dist.recv(activation, src=dist.get_rank() - 1)
if dist.get_rank() < world_size - 1:
    dist.send(output, dst=dist.get_rank() + 1)

# Broadcast model from rank 0
for param in model.parameters():
    dist.broadcast(param.data, src=0)
```

## Best Practices

### Reducing Communication Overhead

1. **Use `communication_data_type`** to reduce bandwidth. For FP16/BF16 training, use the same type for communication.
2. **Enable `overlap_comm`** in ZeRO configuration to overlap communication with backward computation.
3. **Use coalesced collectives** when gathering/scattering many small tensors.
4. **Use `async_op=True`** when you have computation to overlap with communication.
5. **Use `contiguous_gradients: true`** in ZeRO to ensure memory-contiguous gradient buffers.

### Debugging Communication Issues

1. **Enable `comms_logger`** with `verbose: true` to see every communication call.
2. **Enable `debug: true`** to get call stack traces for each operation.
3. **Use `prof_all: true`** to profile all operations and identify bottlenecks.
4. **Check `NCCL_DEBUG=INFO`** environment variable for NCCL-specific debugging.
5. **Verify network connectivity** with `ds_ssh` before multi-node training.

### Backend-Specific Notes

**NCCL:**
- Requires CUDA-capable GPUs with proper driver installation
- Set `NCCL_SOCKET_IFNAME` for multi-node to specify network interface
- Set `NCCL_IB_DISABLE=1` to disable InfiniBand if not available
- Set `NCCL_DEBUG=WARN` for troubleshooting without excessive logging

**CCL:**
- Requires Intel oneCCL installation
- Optimized for Intel CPUs and XPUs
- Supports Intel-specific optimizations like AVX-512

**MPI:**
- Requires MPI implementation (OpenMPI, MPICH, Intel MPI)
- Useful for CPU-only distributed training
- May require additional compilation flags

**HCCL:**
- Requires Huawei Ascend NPU with HCCL library
- Used automatically when running on NPU hardware
- Supports Ascend-specific optimizations
