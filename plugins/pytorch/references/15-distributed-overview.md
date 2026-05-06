# PyTorch Distributed Training Overview - Comprehensive Reference

This chapter covers the fundamentals of PyTorch's distributed training infrastructure, including backends, initialization, process groups, and launch mechanisms.

## Table of Contents

1. [torch.distributed Overview](#torchdistributed-overview)
2. [Backends](#backends)
3. [Initialization](#initialization)
4. [Store](#store)
5. [Process Group Management](#process-group-management)
6. [Rank and World Size](#rank-and-world-size)
7. [Barrier Synchronization](#barrier-synchronization)
8. [Launching Distributed Training](#launching-distributed-training)
9. [Elastic Training](#elastic-training)
10. [Environment Variables](#environment-variables)
11. [Debugging Distributed](#debugging-distributed)
12. [Parallelism Strategies Comparison](#parallelism-strategies-comparison)

---

## torch.distributed Overview

`torch.distributed` (often imported as `dist`) provides a communication primitive interface for multi-process parallelism across multiple computation nodes. It supports several backends for different hardware and communication patterns.

### Architecture

PyTorch distributed training is built on these core concepts:

1. **Process Groups**: Each training process is a member of a process group. The default group includes all processes. Custom groups can be created for subset communication.

2. **Backends**: Communication backends that implement collective and point-to-point operations (NCCL for GPU, Gloo for CPU/GPU, MPI).

3. **Store**: A key-value store used for coordination and rendezvous between processes during initialization.

4. **Collective Operations**: Communication primitives like AllReduce, AllGather, Broadcast, etc.

### Basic Structure

```python
import torch
import torch.distributed as dist
import os

def setup(rank, world_size):
    """Initialize the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )

def cleanup():
    """Clean up the distributed process group."""
    dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)
    # ... training code ...
    cleanup()
```

---

## Backends

PyTorch supports multiple communication backends. The choice of backend depends on hardware, use case, and operation types.

### Backend Comparison

| Feature | NCCL | Gloo | MPI |
|---------|------|------|-----|
| **Primary Use** | GPU communication | CPU communication | General HPC |
| **GPU Operations** | Best performance | Supported but slower | Limited |
| **CPU Operations** | Not supported | Full support | Full support |
| **Infiniband** | Yes (native) | Yes | Yes |
| **RoCE** | Yes | Limited | Yes |
| **Point-to-Point** | GPU only | CPU and GPU | CPU and GPU |
| **Collectives** | GPU optimized | CPU optimized | CPU optimized |
| **Recommended For** | Multi-GPU training | Multi-CPU or debugging | HPC clusters |
| **Build Requirement** | Bundled | Bundled | Requires MPI installation |

### NCCL Backend

The NVIDIA Collective Communications Library (NCCL) backend is the recommended choice for GPU training.

```python
dist.init_process_group(backend='nccl')
```

**Features:**
- Optimized for NVIDIA GPUs
- Supports all collective operations on CUDA tensors
- Implements ring-based and tree-based communication algorithms
- Auto-detects topology for optimal communication paths
- Supports NVLink, PCIe, and InfiniBand interconnects
- Bundled with PyTorch (no separate installation needed)

**NCCL Environment Variables:**
```bash
export NCCL_DEBUG=INFO              # Debug output level
export NCCL_DEBUG_SUBSYS=ALL        # Debug subsystems
export NCCL_SOCKET_IFNAME=eth0      # Network interface
export NCCL_IB_DISABLE=0            # Enable InfiniBand
export NCCL_IB_HCA=mlx5            # InfiniBand HCA
export NCCL_NET_GDR_LEVEL=5         # GPU Direct RDMA level
export NCCL_BUFFSIZE=2097152        # Buffer size
export NCCL_NTHREADS=4              # Number of threads
export NCCL_MIN_NCHANNELS=1         # Min channels
export NCCL_MAX_NCHANNELS=4         # Max channels
export NCCL_P2P_DISABLE=0           # P2P support
export NCCL_SHM_DISABLE=0           # Shared memory
export NCCL_SOCKET_NTHREADS=1       # Socket threads
export NCCL_ALGO=Ring               # Algorithm (Ring, Tree, Collnet)
export NCCL_PROTO=Simple            # Protocol (Simple, LL)
```

**NCCL Timeout Configuration:**
```python
import datetime

dist.init_process_group(
    backend='nccl',
    timeout=datetime.timedelta(seconds=3600)  # 1 hour timeout
)
```

### Gloo Backend

The Gloo backend supports both CPU and GPU communication but is primarily used for CPU-based operations.

```python
dist.init_process_group(backend='gloo')
```

**Features:**
- Developed by Facebook (now Meta)
- Supports CPU and GPU tensors
- Good for CPU-heavy workloads
- Useful for debugging (more descriptive error messages)
- Supports point-to-point operations on CPU
- Bundled with PyTorch

**Gloo Environment Variables:**
```bash
export GLOO_SOCKET_IFNAME=eth0     # Network interface for Gloo
export GLOO_DEVICE_TRANSPORT=PCI   # Device transport
```

### MPI Backend

The MPI (Message Passing Interface) backend requires a working MPI installation.

```python
dist.init_process_group(backend='mpi')
```

**Requirements:**
- PyTorch must be built from source with MPI support (`USE_MPI=1`)
- An MPI implementation (OpenMPI, MPICH, Intel MPI) must be installed

**When to use:**
- When running on HPC clusters with existing MPI infrastructure
- When MPI provides optimized communication for your specific hardware

### XCCL Backend

XCCL is a backend for communication on non-NVIDIA accelerators.

```python
# For specific accelerator support
dist.init_process_group(backend='xccl')
```

### Backend Selection Criteria

```python
import torch

# Automatic backend selection based on hardware
if torch.cuda.is_available():
    backend = 'nccl'   # Best for GPU training
else:
    backend = 'gloo'   # Best for CPU training

dist.init_process_group(backend=backend)
```

**Decision tree:**
1. Training on NVIDIA GPUs? Use **NCCL**
2. Training on CPUs only? Use **Gloo**
3. Running on HPC with MPI infrastructure? Use **MPI**
4. Debugging distributed issues? Use **Gloo** (better error messages)
5. Mixed CPU/GPU communication needed? Use **Gloo** for CPU ops, **NCCL** for GPU ops with separate process groups

---

## Initialization

### init_process_group

Initializes the default distributed process group.

```python
torch.distributed.init_process_group(backend=None, init_method=None,
                                      timeout=None, rank=None,
                                      world_size=None, store=None,
                                      group_name='', pg_options=None)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | str or Backend | None | The backend to use. One of 'nccl', 'gloo', 'mpi'. Should be set based on the training hardware. |
| `init_method` | str | 'env://' | URL specifying how to initialize the process group. |
| `timeout` | timedelta | 30 min | Timeout for operations executed against the process group. |
| `rank` | int | None | Rank of the current process. Required if not using 'env://' init_method. |
| `world_size` | int | None | Number of processes participating in the job. Required if not using 'env://' init_method. |
| `store` | Store | None | Key-value store for rendezvous. If specified, `init_method` must be None. |
| `group_name` | str | '' | Group name. Deprecated. |
| `pg_options` | ProcessGroupOptions | None | Backend-specific options. |

### init_method Options

#### Environment Variable ('env://')

The default method. Reads rank, world_size, and master address from environment variables.

```python
dist.init_process_group(backend='nccl', init_method='env://')
```

Required environment variables:
- `MASTER_ADDR`: IP address or hostname of the master node (rank 0)
- `MASTER_PORT`: Free port on the master node
- `RANK`: Rank of the current process
- `WORLD_SIZE`: Total number of processes

#### TCP ('tcp://')

Direct TCP connection for rendezvous.

```python
dist.init_process_group(
    backend='nccl',
    init_method='tcp://10.1.1.20:23456',
    rank=rank,
    world_size=world_size
)
```

#### File ('file://')

Uses a shared file on a network filesystem for rendezvous.

```python
dist.init_process_group(
    backend='nccl',
    init_method='file:///mnt/nfs/sharedfile',
    rank=rank,
    world_size=world_size
)
```

**Warning:** The file must be on a shared filesystem accessible by all processes. Remove the file between runs to avoid reinitialization errors.

#### Store-Based

Uses a Store object directly for rendezvous.

```python
store = dist.TCPStore('10.1.1.20', 23456, world_size, rank == 0)
dist.init_process_group(
    backend='nccl',
    store=store,
    rank=rank,
    world_size=world_size
)
```

### Backend Enum

```python
import torch.distributed as dist

# Use string or Backend enum
dist.init_process_group(backend='nccl')
dist.init_process_group(backend=dist.Backend.NCCL)

# Check available backends
print(dist.is_backend_available('nccl'))  # True if NCCL is available
print(dist.is_backend_available('gloo'))  # True if Gloo is available

# List all available backends
print(dist.Backend.backend_list)  # List of available backends
```

### Checking Initialization

```python
# Check if process group is initialized
if dist.is_initialized():
    print("Process group is initialized")
else:
    print("Process group is NOT initialized")
```

### Destroying the Process Group

```python
# Clean up at the end of training
dist.destroy_process_group()
```

This should be called by all processes. It frees distributed resources and ensures clean shutdown.

---

## Store

Stores provide a distributed key-value store for coordination between processes.

### TCPStore

A TCP-based key-value store. One process (typically rank 0) runs the server, and all other processes connect as clients.

```python
torch.distributed.TCPStore(host_name, port, world_size, is_master=False,
                            multi_tenant=False, main_address=None, main_port=None,
                            timeout=datetime.timedelta(seconds=300),
                            use_libuv=False)
```

**Parameters:**
- `host_name` (str): The hostname or IP address of the store.
- `port` (int): The port the store should listen on.
- `world_size` (int): The total number of processes.
- `is_master` (bool): Whether this process is the master (server). Typically True for rank 0.
- `multi_tenant` (bool): If True, multiple processes on the same host can share the store.
- `main_address` (str): Address of the master store (for worker processes).
- `main_port` (int): Port of the master store (for worker processes).
- `timeout` (timedelta): Timeout for store operations.
- `use_libuv` (bool): Use libuv for the store server.

**Example:**
```python
import torch.distributed as dist

# Rank 0 starts the server
store = dist.TCPStore(
    host_name='127.0.0.1',
    port=29500,
    world_size=4,
    is_master=(rank == 0)
)

# Use for coordination
store.set('key', 'value')
value = store.get('key')
```

### FileStore

A file-based key-value store for coordination.

```python
store = dist.FileStore('/tmp/shared_store', world_size)
```

**Note:** Requires a shared filesystem accessible by all processes.

### HashStore

An in-memory store for single-process use (mainly for testing).

```python
store = dist.HashStore()
```

### PrefixStore

Wraps another store and prefixes all keys. Useful for isolating different training runs.

```python
base_store = dist.TCPStore('127.0.0.1', 29500, world_size, True)
store = dist.PrefixStore('my_experiment', base_store)

store.set('key', 'value')  # Actually stores 'my_experiment/key'
```

### Store Operations

```python
# Basic operations
store.set('key', 'value')             # Set a key-value pair
value = store.get('key')              # Get value by key
store.add('counter', 1)               # Atomic add to a key
store.delete_key('key')               # Delete a key
store.compare_set('key', 'expected', 'new')  # Compare and set

# Wait for keys to be set by other processes
store.wait(['key1', 'key2'])          # Block until all keys are set
store.wait(['key1'], timeout=10.0)    # With timeout

# Number of keys
num_keys = len(store)
```

---

## Process Group Management

### Default Process Group

When `init_process_group` is called, a default process group is created that includes all processes.

### new_group

Creates a new process group containing a subset of the processes.

```python
torch.distributed.new_group(ranks=None, timeout=datetime.timedelta(seconds=1800),
                             backend=None, pg_options=None)
```

**Parameters:**
- `ranks` (list[int]): List of ranks to include in the new group. If None, includes all ranks (same as default group).
- `timeout` (timedelta): Timeout for operations on this group.
- `backend` (str): Backend to use for this group. If None, uses the default backend.
- `pg_options` (ProcessGroupOptions): Backend-specific options.

**Returns:** A `ProcessGroup` object to use in collective operations.

```python
import torch.distributed as dist

# Initialize default group
dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

# Create a subgroup for ranks 0, 1, 2
group_012 = dist.new_group(ranks=[0, 1, 2])

# Create another subgroup for ranks 2, 3
group_23 = dist.new_group(ranks=[2, 3])

# Use in collective operations
if rank in [0, 1, 2]:
    tensor = torch.randn(10).cuda()
    dist.all_reduce(tensor, group=group_012)
```

### new_subgroups

Creates multiple subgroups of approximately equal size.

```python
torch.distributed.new_subgroups(group_size=None, pg=None)
```

```python
# Split 8 ranks into groups of 2
subgroup, _ = dist.new_subgroups(group_size=2)
# Creates 4 subgroups: [0,1], [2,3], [4,5], [6,7]
```

### Backend-Specific Options

#### ProcessGroupNCCL Options

```python
import torch.distributed as dist

# NCCL options
pg_options = dist.ProcessGroupNCCL.Options()
pg_options.is_high_priority_stream = False
pg_options.config.blocking = False

group = dist.new_group(ranks=[0, 1], pg_options=pg_options)
```

#### ProcessGroupGloo Options

```python
pg_options = dist.ProcessGroupGloo.Options()
pg_options._timeout = 300  # seconds

group = dist.new_group(ranks=[0, 1], backend='gloo', pg_options=pg_options)
```

---

## Rank and World Size

### Concepts

- **World Size**: Total number of processes participating in distributed training.
- **Rank**: A unique identifier for each process in the range `[0, world_size - 1]`.
- **Local Rank**: The rank of a process within its node. Used for device assignment.

### Querying Rank and World Size

```python
# Get global rank
rank = dist.get_rank()

# Get world size
world_size = dist.get_world_size()

# Get local rank (from environment variable set by torchrun)
local_rank = int(os.environ.get('LOCAL_RANK', 0))

# Get the process group of the current process
pg = dist.group.WORLD  # Default process group
```

### Common Patterns

```python
import os
import torch
import torch.distributed as dist

def setup():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])

    # Each process uses its corresponding GPU
    torch.cuda.set_device(local_rank)

    print(f"Rank {rank}/{world_size} on {os.uname().nodename}, "
          f"using GPU {local_rank}")

def get_device():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    return torch.device(f'cuda:{local_rank}')
```

---

## Barrier Synchronization

### barrier

Blocks until all processes in the group reach this point.

```python
torch.distributed.barrier(group=None, async_op=False, device_ids=None)
```

**Parameters:**
- `group` (ProcessGroup): The process group to use. Default: the default group.
- `async_op` (bool): If True, returns a distributed work handle that can be waited on.
- `device_ids` (list[int]): Device IDs for barrier synchronization.

```python
# Simple barrier - all processes wait here
dist.barrier()

# Barrier in a subgroup
dist.barrier(group=my_subgroup)

# Barrier with async operation
work = dist.barrier(async_op=True)
# Do other work...
work.wait()
```

### Monitored Barrier

A barrier that reports stuck processes for debugging.

```python
torch.distributed.monitored_barrier(group=None, timeout=None, wait_all_ranks=False)
```

**Parameters:**
- `group` (ProcessGroup): Process group. Default: default group.
- `timeout` (timedelta): Timeout. Default: from the process group.
- `wait_all_ranks` (bool): If True, waits for all ranks to respond. If False, returns after a majority responds.

```python
# Use monitored barrier for debugging
dist.monitored_barrier(timeout=datetime.timedelta(seconds=30))

# In debug mode, this will print which ranks have/haven't reached the barrier
```

---

## Launching Distributed Training

### torchrun (Recommended)

`torchrun` is the recommended launcher for PyTorch distributed training. It is part of PyTorch and provides robust, fault-tolerant training launching.

```bash
# Single-node, multi-GPU
torchrun --nproc_per_node=4 train.py

# Single-node with specific GPUs
torchrun --nproc_per_node=2 train.py --batch_size 32

# Multi-node
# On node 0 (master):
torchrun --nnodes=2 --nproc_per_node=4 \
    --rdzv_id=job1 --rdzv_backend=c10d \
    --rdzv_endpoint=MASTER_IP:29500 \
    train.py

# On node 1 (worker):
torchrun --nnodes=2 --nproc_per_node=4 \
    --rdzv_id=job1 --rdzv_backend=c10d \
    --rdzv_endpoint=MASTER_IP:29500 \
    train.py
```

### torchrun Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--nnodes` | 1 | Number of nodes participating in the job |
| `--nproc_per_node` | 1 | Number of processes per node (typically = number of GPUs) |
| `--rdzv_id` | none | A unique id for the rendezvous (used for multi-node) |
| `--rdzv_backend` | c10d | Rendezvous backend (c10d, etcd, etcd-v2) |
| `--rdzv_endpoint` | none | Rendezvous endpoint (host:port) |
| `--max_restarts` | 0 | Maximum number of restarts for a worker group |
| `--monitor_interval` | 5 | Interval (seconds) to monitor the state of workers |
| `--start_method` | spawn | Multiprocessing start method (spawn, fork, forkserver) |
| `--role` | default | User-defined role for the workers |
| `--log_dir` | none | Directory for torchrun logs |
| `--redirects` | none | Redirect stdout/stderr to files |
| `--tee` | none | Tee stdout/stderr to files and console |
| `--master_addr` | 127.0.0.1 | Master node address |
| `--master_port` | 29500 | Master node port |
| `--node_rank` | 0 | Rank of the node (for multi-node without rendezvous) |

### torchrun in Python Script

```python
# train.py - designed to be launched with torchrun
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def setup():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

def cleanup():
    dist.destroy_process_group()

def main():
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])

    # Create model and move to current GPU
    model = MyModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Create data sampler for distributed training
    dataset = MyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Important for proper shuffling
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(batch)
            loss = compute_loss(output)
            loss.backward()
            optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch} complete")

    cleanup()

if __name__ == '__main__':
    main()
```

### torch.distributed.launch (Deprecated)

The legacy launcher. Use `torchrun` instead.

```bash
# DEPRECATED - use torchrun instead
python -m torch.distributed.launch --nproc_per_node=4 train.py
```

### mp.spawn

Programmatic launching using multiprocessing.

```python
import torch.multiprocessing as mp

def train_fn(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # ... training loop ...

    dist.destroy_process_group()

world_size = torch.cuda.device_count()
mp.spawn(train_fn, args=(world_size,), nprocs=world_size, join=True)
```

### SLURM Integration

```python
import os
import torch.distributed as dist

def setup_slurm():
    # SLURM sets these environment variables
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    master_addr = os.environ['MASTERADDR']

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(local_rank)
```

```bash
# SLURM submission script
#!/bin/bash
#SBATCH --job-name=pytorch_ddp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

srun torchrun --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 train.py
```

---

## Elastic Training

PyTorch Elastic Training (torchelastic/torchrun) provides fault-tolerant, elastic distributed training.

### Key Concepts

1. **Rendezvous**: A mechanism for processes to discover each other and agree on membership.
2. **Elastic Agent**: Manages workers on each node, handles restarts.
3. **State**: Checkpointable training state for recovery.

### Rendezvous Backends

#### c10d (Default)

Uses the built-in C10d TCPStore for rendezvous.

```bash
torchrun --rdzv_backend=c10d --rdzv_endpoint=MASTER_IP:29500 train.py
```

#### etcd

Uses etcd for distributed coordination (better for large-scale).

```bash
torchrun --rdzv_backend=etcd --rdzv_endpoint=ETCD_IP:2379 train.py
```

### Elastic Training Pattern

```python
import torch.distributed.elastic as elastic

def main():
    # elastic.multiprocessing handles rank/world_size automatically
    dist.init_process_group(backend='nccl')

    # Load from checkpoint if available
    start_epoch = load_checkpoint_if_available()

    for epoch in range(start_epoch, num_epochs):
        train_one_epoch()
        if dist.get_rank() == 0:
            save_checkpoint(epoch)

    dist.destroy_process_group()
```

### torch.distributed.elastic.multiprocessing

```python
from torch.distributed.elastic.multiprocessing.errors import record

@record  # Captures and propagates errors across ranks
def main():
    setup()
    train()
    cleanup()
```

### Fault Tolerance with State

```python
from torch.distributed.elastic.multiprocessing import StdRedirectionType
from torch.distributed.elastic.multiprocessing.redirects import redirect_stderr, redirect_stdout

# Save state for recovery
def save_checkpoint(epoch, model, optimizer, scheduler):
    if dist.get_rank() == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'checkpoint.pt')
    dist.barrier()

# Recover state
def load_checkpoint(model, optimizer):
    if os.path.exists('checkpoint.pt'):
        checkpoint = torch.load('checkpoint.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    return 0
```

---

## Environment Variables

### Essential Variables

```bash
# Set by torchrun automatically
MASTER_ADDR=10.1.1.20        # Master node IP
MASTER_PORT=29500             # Master node port
WORLD_SIZE=8                  # Total number of processes
RANK=3                        # Global rank of this process
LOCAL_RANK=1                  # Local rank within this node
LOCAL_WORLD_SIZE=4            # Number of processes on this node
GROUP_WORLD_SIZE=1            # Size of the node group
ROLE=default                  # Role of this worker
ROLE_INDEX=3                  # Index within the role
```

### NCCL Variables

```bash
NCCL_DEBUG=INFO               # Enable NCCL debug logging
NCCL_DEBUG_SUBSYS=ALL         # All subsystems
NCCL_SOCKET_IFNAME=ib0        # Network interface
NCCL_IB_DISABLE=0             # Enable InfiniBand
NCCL_NET_GDR_LEVEL=5          # GPU Direct RDMA
NCCL_P2P_LEVEL=SYS            # P2P level (SYS, NODE, PHB, PXB, PIX)
NCCL_SHM_DISABLE=0            # Enable shared memory
NCCL_MIN_NCHANNELS=4          # Min channels
NCCL_MAX_NCHANNELS=16         # Max channels
NCCL_BUFFSIZE=8388608         # Buffer size (8MB)
NCCL_ALGO=Ring                # Algorithm
NCCL_PROTO=LL                 # Protocol
NCCL_NRINGS=2                 # Number of rings
NCCL_MAX_NRINGS=8             # Max rings
NCCL_CHECKS_DISABLE=0         # Disable checks
NCCL_CHECK_POINTERS=1         # Check pointers
```

### PyTorch Distributed Variables

```bash
TORCH_DISTRIBUTED_DEBUG=OFF   # Debug mode (OFF, INFO, DETAIL)
TORCH_DISTRIBUTED_GATE_TIMEOUT=600  # Gate timeout in seconds
TORCH_NCCL_ASYNC_ERROR_HANDLING=1   # Enable async error handling
TORCH_NCCL_BLOCKING_WAIT=0          # Blocking wait mode
TORCH_DISTRIBUTED_FILE_SYSTEM=0     # Use filesystem for rendezvous
TORCH_DIST_INIT_BARRIER=1           # Barrier after init
```

---

## Debugging Distributed

### TORCH_DISTRIBUTED_DEBUG

Set the `TORCH_DISTRIBUTED_DEBUG` environment variable for detailed logging:

```bash
# OFF - No debug output
TORCH_DISTRIBUTED_DEBUG=OFF torchrun --nproc_per_node=2 train.py

# INFO - Basic debug information
TORCH_DISTRIBUTED_DEBUG=INFO torchrun --nproc_per_node=2 train.py

# DETAIL - Very detailed logging of all collective operations
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=2 train.py
```

### NCCL Debug

```bash
# Enable NCCL debug logging
NCCL_DEBUG=INFO torchrun --nproc_per_node=2 train.py

# Debug specific subsystems
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL torchrun --nproc_per_node=2 train.py
```

### Common Debugging Patterns

```python
import torch.distributed as dist

# Check if distributed is available
print(f"Distributed available: {dist.is_available()}")

# Check backend availability
print(f"NCCL available: {dist.is_backend_available('nccl')}")

# Get process info
print(f"Rank: {dist.get_rank()}")
print(f"World size: {dist.get_world_size()}")

# Use monitored barrier to detect stuck processes
dist.monitored_barrier(timeout=datetime.timedelta(seconds=30))

# Check for NCCL errors
try:
    dist.all_reduce(torch.zeros(1).cuda())
except RuntimeError as e:
    print(f"NCCL error: {e}")
```

### Debugging DDP Issues

```python
# Enable DDP debug mode
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

# Check unused parameters
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

# Monitor gradient synchronization
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if dist.get_rank() == 0:
            print(f"{name}: grad_norm={grad_norm}")
```

---

## Parallelism Strategies Comparison

### Data Parallelism (DDP)

Replicates the entire model on each GPU. Each GPU processes a different data subset.

```python
# Simple to implement
model = DDP(model, device_ids=[local_rank])
```

**Pros:** Easy to use, linear scaling for small models
**Cons:** Model must fit on a single GPU, communication overhead grows with model size

### Model Parallelism

Splits the model across multiple GPUs. Each GPU holds a portion of the model.

```python
# Manual model parallelism
class ModelParallelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.part1 = nn.Linear(1000, 1000).to('cuda:0')
        self.part2 = nn.Linear(1000, 10).to('cuda:1')

    def forward(self, x):
        x = self.part1(x.to('cuda:0'))
        x = self.part2(x.to('cuda:1'))
        return x
```

**Pros:** Can train models larger than a single GPU
**Cons:** Sequential execution, poor GPU utilization, manual placement

### Pipeline Parallelism

Splits the model into stages, with each stage on a different GPU. Uses micro-batching for overlap.

**Pros:** Good for sequential models, better utilization than naive model parallelism
**Cons:** Pipeline bubbles, complex to implement

### Tensor/ZeRO Parallelism (FSDP)

Shards model parameters, gradients, and optimizer state across GPUs.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model, auto_wrap_policy=..., sharding_strategy=...)
```

**Pros:** Can train very large models, memory efficient
**Cons:** More communication overhead, more complex setup

### Choosing a Strategy

| Strategy | Model Size | Communication | Memory Efficiency | Ease of Use |
|----------|-----------|---------------|-------------------|-------------|
| DDP | Fits on 1 GPU | Medium | Low (full replica) | Easy |
| FSDP (FULL_SHARD) | Larger than 1 GPU | High | High | Medium |
| Pipeline | Very large sequential | Low-Medium | Medium | Hard |
| Tensor Parallel | Very large layers | Very High | High | Hard |
| Hybrid (FSDP+TP) | Extremely large | High | Very High | Very Hard |
