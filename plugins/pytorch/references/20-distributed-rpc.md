# PyTorch Distributed RPC - Comprehensive Reference

This chapter covers `torch.distributed.rpc`, PyTorch's framework for remote procedure calls, remote references, distributed autograd, and distributed optimization.

## Table of Contents

1. [RPC Overview](#rpc-overview)
2. [RPC Initialization](#rpc-initialization)
3. [Remote Execution](#remote-execution)
4. [RRef (Remote Reference)](#rref-remote-reference)
5. [Distributed Autograd](#distributed-autograd)
6. [Distributed Optimizer](#distributed-optimizer)
7. [RPC Profiling](#rpc-profiling)
8. [Pipeline Parallel with RPC](#pipeline-parallel-with-rpc)
9. [RemoteModule](#remotemodule)
10. [Fault Tolerance](#fault-tolerance)
11. [RPC Benchmarking](#rpc-benchmarking)
12. [Parameter Server Example](#parameter-server-example)

---

## RPC Overview

`torch.distributed.rpc` provides a framework for distributed training that goes beyond data parallelism. It enables:

1. **Remote function execution**: Run functions on remote workers.
2. **Remote references (RRef)**: Reference objects on remote workers without copying data.
3. **Distributed autograd**: Automatic differentiation across process boundaries.
4. **Distributed optimization**: Optimizer that works across process boundaries.

### When to Use RPC

- **Parameter server training**: Centralized parameter management with distributed workers.
- **Pipeline parallelism**: Model split across workers with RPC-based communication.
- **Reinforcement learning**: Central policy with distributed environment workers.
- **Model serving**: Distributed inference with model partitioning.
- **Custom distributed patterns**: When DDP/FSDP don't fit your use case.

### Architecture

```
Worker 0 (Master)     Worker 1 (Parameter Server)     Worker 2 (Worker)
     |                         |                              |
     |--- rpc_sync() -------->|                              |
     |<-- result --------------|                              |
     |                                                        |
     |--- rpc_async() ------>|                               |
     |   (returns Future)     |                               |
     |                                                        |
     |--- remote() ---------->|                              |
     |<-- RRef ---------------|                              |
     |                                                        |
     |--- rpc_to_here() ----> via RRef                        |
```

---

## RPC Initialization

### init_rpc

Initializes the RPC framework.

```python
torch.distributed.rpc.init_rpc(
    name=None,
    rank=None,
    world_size=None,
    rpc_backend_options=None,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | None | A unique name for this worker. If None, uses `"worker_{rank}"`. |
| `rank` | int | None | The global rank of this worker. If None, uses `dist.get_rank()` if already initialized. |
| `world_size` | int | None | The total number of workers. If None, uses `dist.get_world_size()`. |
| `rpc_backend_options` | RpcBackendOptions | None | Backend-specific options. |

### RpcBackendOptions

```python
import torch.distributed.rpc as rpc
from torch.distributed.rpc.backend import TensorPipeRpcBackendOptions

# Default options
options = TensorPipeRpcBackendOptions(
    num_worker_threads=8,           # Number of threads for RPC execution
    rpc_timeout=60.0,               # Timeout in seconds for RPC calls
    init_method='tcp://localhost:29501',  # URL for rendezvous
    device_maps=None,               # Device mapping between workers
)
```

### TensorPipeRpcBackendOptions Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_worker_threads` | int | 8 | Number of threads in the thread pool for executing RPC callbacks. |
| `rpc_timeout` | float | 60.0 | Timeout in seconds for all RPCs. |
| `init_method` | str | 'tcp://...' | URL for rendezvous initialization. |
| `device_maps` | dict | None | Dict mapping worker names to device mappings for cross-device RPC. |

### Basic Initialization

```python
import os
import torch.distributed.rpc as rpc

def setup_rpc(rank, world_size):
    rpc.init_rpc(
        name=f"worker_{rank}",
        rank=rank,
        world_size=world_size,
    )

def cleanup_rpc():
    rpc.shutdown()

# Usage
if __name__ == '__main__':
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    setup_rpc(rank, world_size)

    # ... RPC operations ...

    cleanup_rpc()
```

### shutdown

Shuts down the RPC framework. Should be called by all workers.

```python
rpc.shutdown()
# or with graceful shutdown
rpc.shutdown(graceful=True)   # Wait for all pending RPCs to complete
rpc.shutdown(graceful=False)  # Immediately terminate
```

### is_available

```python
print(rpc.is_available())  # True if RPC is available
```

---

## Remote Execution

### rpc_sync

Executes a function on a remote worker synchronously. Blocks until the result is returned.

```python
torch.distributed.rpc.rpc_sync(to, func, args=None, kwargs=None, timeout=-1.0)
```

**Parameters:**
- `to` (str or int): Name or rank of the destination worker.
- `func` (callable): The function to execute remotely.
- `args` (tuple): Positional arguments for the function.
- `kwargs` (dict): Keyword arguments for the function.
- `timeout` (float): Timeout in seconds. -1 means use the default timeout.

**Returns:** The return value of `func`.

```python
import torch
import torch.distributed.rpc as rpc

def add(x, y):
    return x + y

# Execute add() on worker 1
result = rpc.rpc_sync("worker_1", add, args=(torch.tensor(1.0), torch.tensor(2.0)))
# result = tensor(3.0)

# Using rank instead of name
result = rpc.rpc_sync(1, add, args=(1, 2))
# result = 3
```

### rpc_async

Executes a function on a remote worker asynchronously. Returns a `Future` immediately.

```python
torch.distributed.rpc.rpc_async(to, func, args=None, kwargs=None, timeout=-1.0)
```

**Returns:** A `torch.futures.Future` object.

```python
import torch
import torch.distributed.rpc as rpc

def heavy_computation(x):
    return x * 2

# Execute asynchronously
future = rpc.rpc_async("worker_1", heavy_computation, args=(torch.tensor(5.0),))

# Do other work while waiting...
local_result = some_local_work()

# Wait for the result
remote_result = future.wait()
# remote_result = tensor(10.0)
```

### Multiple Async RPCs

```python
# Fan-out: send work to multiple workers
futures = []
for worker_rank in range(1, world_size):
    f = rpc.rpc_async(f"worker_{worker_rank}", process_data, args=(data_chunk,))
    futures.append(f)

# Collect results
results = [f.wait() for f in futures]
```

### remote

Creates a remote reference (RRef) to an object on a remote worker.

```python
torch.distributed.rpc.remote(to, func, args=None, kwargs=None, timeout=-1.0)
```

**Returns:** An `RRef` (Remote Reference) to the remotely created object.

```python
import torch
import torch.distributed.rpc as rpc

class ParameterManager:
    def __init__(self, size):
        self.params = torch.randn(size)

    def get_params(self):
        return self.params

    def update(self, grad, lr=0.01):
        self.params -= lr * grad

# Create a ParameterManager on worker 1
param_rref = rpc.remote("worker_1", ParameterManager, args=(1000,))

# The object lives on worker 1, we only have a reference
print(type(param_rref))  # <class 'torch.distributed.rpc.RRef'>
```

### RPC with Remote Modules

```python
def create_and_use_model(worker_name):
    """Create a model on a remote worker and execute it."""
    # Create model on remote worker
    model_rref = rpc.remote(worker_name, MyModel)

    # Execute forward pass on remote worker
    def forward_on_remote(model_rref, input_tensor):
        return model_rref.local_value()(input_tensor)

    result = rpc.rpc_sync(
        worker_name,
        forward_on_remote,
        args=(model_rref, torch.randn(10))
    )
    return result
```

---

## RRef (Remote Reference)

RRef (Remote Reference) is a reference to an object on a remote worker. It allows accessing and manipulating remote objects without moving the entire object.

### Creating RRefs

```python
# Method 1: Using rpc.remote()
rref = rpc.remote("worker_1", torch.randn, args=(100,))

# Method 2: Inside a remote function
class MyClass:
    def __init__(self):
        self.data = torch.randn(100)

def create_on_remote():
    obj = MyClass()
    return rpc.RRef(obj)  # Return an RRef to this object
```

### RRef Methods

#### `to_here(timeout=-1.0)`

Copies the value from the remote worker to the local worker.

```python
rref = rpc.remote("worker_1", torch.randn, args=(100,))
local_tensor = rref.to_here()
# local_tensor is a copy of the remote tensor
```

#### `local_value()`

Returns the local value. Can only be called on the owner worker.

```python
# Only works on the worker that owns the RRef
if rref.owner() == rpc.get_worker_info().id:
    value = rref.local_value()
```

#### `owner()`

Returns the worker ID (rank) of the RRef's owner.

```python
owner_rank = rref.owner()
```

#### `owner_name()`

Returns the worker name of the RRef's owner.

```python
owner_name = rref.owner_name()
```

#### `rpc_async(func, args=None, kwargs=None, timeout=-1.0)`

Executes a function on the owner worker using the RRef's value as the first argument.

```python
# Create remote object
rref = rpc.remote("worker_1", MyClass, args=())

# Call method on remote object
future = rref.rpc_async(lambda obj: obj.compute())
result = future.wait()
```

#### `rpc_sync(func, args=None, kwargs=None, timeout=-1.0)`

Synchronous version of `rpc_async`.

```python
result = rref.rpc_sync(lambda obj: obj.get_data())
```

### RRef Example

```python
import torch
import torch.distributed.rpc as rpc

class RemoteModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

    def get_gradients(self):
        return [(n, p.grad) for n, p in self.named_parameters() if p.grad is not None]

def parameter_server():
    """Run on the parameter server worker."""
    # Create model and return RRef
    model_rref = rpc.RRef(RemoteModel(784, 256, 10))
    print(f"Parameter server ready: {model_rref}")

    # Serve forever (or until training is done)
    while True:
        # Process updates from workers
        pass

def worker(ps_rref):
    """Run on a worker."""
    # Get model output via RPC
    data = torch.randn(32, 784)

    # Execute forward pass on parameter server
    output = rpc.rpc_sync(
        ps_rref.owner_name(),
        lambda model: model.forward(torch.randn(32, 784)),
        args=(ps_rref.local_value(),)
    )
```

---

## Distributed Autograd

Distributed autograd extends PyTorch's autograd to work across process boundaries via RPC.

### dist_autograd.context()

Creates a context for distributed autograd. All RPCs within this context are tracked for gradient computation.

```python
import torch.distributed.autograd as dist_autograd

with dist_autograd.context() as context_id:
    # All RPCs within this block are tracked
    result = rpc.rpc_sync("worker_1", some_function, args=(input_tensor,))

    # Compute loss from the result
    loss = compute_loss(result)

    # Backpropagate through the distributed computation graph
    dist_autograd.backward(context_id, [loss])

    # Get gradients
    grads = dist_autograd.get_gradients(context_id)
```

### How Distributed Autograd Works

1. **Context**: A `dist_autograd.context()` creates a unique context ID distributed to all participating workers.
2. **Forward**: During forward, all RPC calls are recorded with the context ID. Each worker stores the local autograd graph.
3. **Backward**: `dist_autograd.backward()` triggers backward propagation across all involved workers. Each worker computes local gradients and propagates them via RPC.
4. **Gradients**: `dist_autograd.get_gradients()` retrieves the accumulated gradients for the context.

### Example: Distributed Forward + Backward

```python
import torch
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd

def remote_forward(model_rref, input):
    """Execute forward pass on remote worker."""
    return model_rref.local_value().forward(input)

def distributed_training_loop(model_rref, dataloader):
    """Training loop using distributed autograd."""
    optimizer = torch.optim.SGD(model_rref.local_value().parameters(), lr=0.01)

    for data, target in dataloader:
        with dist_autograd.context() as ctx_id:
            # Forward pass on remote worker
            output = rpc.rpc_sync(
                model_rref.owner_name(),
                remote_forward,
                args=(model_rref, data)
            )

            # Compute loss locally
            loss = torch.nn.functional.cross_entropy(output, target)

            # Distributed backward
            dist_autograd.backward(ctx_id, [loss])

            # Get distributed gradients
            grads = dist_autograd.get_gradients(ctx_id)

            # Apply gradients
            for param, grad in zip(model_rref.local_value().parameters(), grads.values()):
                param.grad = grad
            optimizer.step()
            optimizer.zero_grad()
```

### get_gradients

Retrieves gradients accumulated in a distributed autograd context.

```python
grads = dist_autograd.get_gradients(context_id)
# Returns a dict mapping tensor -> gradient tensor
```

---

## Distributed Optimizer

The distributed optimizer applies gradients across workers using distributed autograd.

### DistributedOptimizer

```python
from torch.distributed.optim import DistributedOptimizer

dist_optimizer = DistributedOptimizer(
    optimizer_class,           # e.g., torch.optim.SGD
    parameter_rrefs,           # List of RRefs to parameters
    args=None,                 # Args for the optimizer constructor
    kwargs=None,               # Kwargs for the optimizer constructor
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer_class` | type | required | The optimizer class to instantiate (e.g., `torch.optim.SGD`). |
| `parameter_rrefs` | list[RRef] | required | List of RRefs to the parameters to optimize. |
| `args` | tuple | None | Positional arguments for the optimizer constructor. |
| `kwargs` | dict | None | Keyword arguments for the optimizer constructor. |

### Usage

```python
import torch
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer

def get_param_rrefs(model_rref):
    """Get RRefs to all parameters of a remote model."""
    param_rrefs = rpc.rpc_sync(
        model_rref.owner_name(),
        lambda m: [rpc.RRef(p) for p in m.parameters()],
        args=(model_rref.local_value(),)
    )
    return param_rrefs

def train_with_dist_optimizer(model_rref, dataloader):
    # Get parameter RRefs
    param_rrefs = get_param_rrefs(model_rref)

    # Create distributed optimizer
    dist_optimizer = DistributedOptimizer(
        torch.optim.SGD,
        parameter_rrefs=param_rrefs,
        lr=0.01,
    )

    for epoch in range(num_epochs):
        for data, target in dataloader:
            with dist_autograd.context() as ctx_id:
                # Forward
                output = rpc.rpc_sync(
                    model_rref.owner_name(),
                    lambda m, x: m(x),
                    args=(model_rref.local_value(), data)
                )
                loss = torch.nn.functional.cross_entropy(output, target)

                # Distributed backward
                dist_autograd.backward(ctx_id, [loss])

                # Distributed optimizer step
                dist_optimizer.step(ctx_id)
```

---

## RPC Profiling

### Using torch.profiler with RPC

```python
import torch.profiler as profiler

def profile_rpc_training():
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler('./log_dir'),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for i, (data, target) in enumerate(dataloader):
            with dist_autograd.context() as ctx_id:
                output = rpc.rpc_sync("worker_1", forward_fn, args=(data,))
                loss = criterion(output, target)
                dist_autograd.backward(ctx_id, [loss])
                dist_optimizer.step(ctx_id)
            prof.step()
```

### RPC-Specific Profiling

```python
# Enable RPC profiling
import torch.distributed.rpc as rpc

# RPC operations are automatically profiled when using torch.profiler
# The profile traces will include RPC call overhead and remote execution time
```

---

## Pipeline Parallel with RPC

RPC enables pipeline parallelism by executing different model stages on different workers.

### Basic Pipeline with RPC

```python
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc

class Stage(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x_rref):
        x = x_rref.to_here()
        return self.layer(x)

def pipeline_forward(stages, input_tensor):
    """Forward pass through pipeline stages on different workers."""
    current = input_tensor

    for i, stage_rref in enumerate(stages):
        # Execute stage on its worker
        current = rpc.rpc_sync(
            stage_rref.owner_name(),
            lambda stage, x: stage.forward(x),
            args=(stage_rref.local_value(), current)
        )

    return current

# Setup
def setup_pipeline(rank, world_size):
    rpc.init_rpc(f"worker_{rank}", rank=rank, world_size=world_size)

    # Create stages on different workers
    stages = []
    if rank == 0:
        stage0 = Stage(nn.Linear(784, 512))
        stages.append(rpc.RRef(stage0))
    elif rank == 1:
        stage1 = Stage(nn.Linear(512, 256))
        stages.append(rpc.RRef(stage1))
    elif rank == 2:
        stage2 = Stage(nn.Linear(256, 10))
        stages.append(rpc.RRef(stage2))

    rpc.barrier()

    return stages
```

### Micro-Batch Pipeline with RPC

```python
def pipeline_forward_micro_batch(stages, input_batch, chunks=4):
    """Forward pass with micro-batching for pipeline parallelism."""
    micro_batches = input_batch.chunk(chunks)

    # Forward all micro-batches through each stage
    for stage_rref in stages:
        futures = []
        for mb in micro_batches:
            f = rpc.rpc_async(
                stage_rref.owner_name(),
                lambda stage, x: stage.forward(x),
                args=(stage_rref.local_value(), mb)
            )
            futures.append(f)
        micro_batches = [f.wait() for f in futures]

    return torch.cat(micro_batches)
```

---

## RemoteModule

`RemoteModule` creates a module replica on a remote worker and provides a convenient interface for calling it.

### Creating a RemoteModule

```python
from torch.distributed.nn.api.remote_module import RemoteModule

# Create a remote module on worker 1
remote_linear = RemoteModule(
    on="worker_1",
    module_cls=torch.nn.Linear,
    args=(784, 256),  # Args for nn.Linear constructor
)

# Use it like a local module
output = remote_linear(torch.randn(32, 784))
```

### RemoteModule Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `on` | str or int | required | The worker name or rank where the module should be created. |
| `module_cls` | type | required | The module class to instantiate (e.g., `nn.Linear`). |
| `args` | tuple | None | Positional arguments for the module constructor. |
| `kwargs` | dict | None | Keyword arguments for the module constructor. |

### RemoteModule Methods

```python
# Forward pass (executed on remote worker)
output = remote_module(input_tensor)

# Get a remote reference to the module's parameters
for param_rref in remote_module.remote_parameters():
    param = param_rref.to_here()
    print(f"Parameter shape: {param.shape}")
```

### Multi-Stage Pipeline with RemoteModule

```python
from torch.distributed.nn.api.remote_module import RemoteModule

class PipelineModel:
    def __init__(self, worker_names):
        self.stages = []

        # Stage 0: Input processing
        self.stages.append(RemoteModule(
            on=worker_names[0],
            module_cls=nn.Sequential,
            args=(nn.Linear(784, 512), nn.ReLU()),
        ))

        # Stage 1: Hidden layers
        self.stages.append(RemoteModule(
            on=worker_names[1],
            module_cls=nn.Sequential,
            args=(nn.Linear(512, 256), nn.ReLU()),
        ))

        # Stage 2: Output layer
        self.stages.append(RemoteModule(
            on=worker_names[2],
            module_cls=nn.Linear,
            args=(256, 10),
        ))

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x
```

---

## Fault Tolerance

### TensorPipeRpcBackendOptions for Fault Tolerance

```python
from torch.distributed.rpc.backend import TensorPipeRpcBackendOptions

options = TensorPipeRpcBackendOptions(
    num_worker_threads=16,
    rpc_timeout=120.0,
    init_method='tcp://localhost:29501',
)

rpc.init_rpc(
    name=f"worker_{rank}",
    rank=rank,
    world_size=world_size,
    rpc_backend_options=options,
)
```

### Handling RPC Timeouts

```python
import torch.distributed.rpc as rpc

try:
    result = rpc.rpc_sync(
        "worker_1",
        some_function,
        args=(data,),
        timeout=30.0
    )
except RuntimeError as e:
    if "RPC timed out" in str(e):
        print("RPC call timed out, retrying...")
        result = rpc.rpc_sync(
            "worker_1",
            some_function,
            args=(data,),
            timeout=60.0
        )
    else:
        raise
```

### Retry Logic

```python
def rpc_with_retry(to, func, args=None, kwargs=None, max_retries=3, timeout=30.0):
    """Execute an RPC with retry logic."""
    for attempt in range(max_retries):
        try:
            return rpc.rpc_sync(to, func, args=args, kwargs=kwargs, timeout=timeout)
        except RuntimeError as e:
            if attempt < max_retries - 1:
                print(f"RPC failed (attempt {attempt + 1}), retrying...")
                import time
                time.sleep(1)
            else:
                raise
```

---

## RPC Benchmarking

### Measuring RPC Latency

```python
import time
import torch.distributed.rpc as rpc

def ping_pong(x):
    """Simple function for latency measurement."""
    return x

def benchmark_rpc_latency(worker_name, num_iterations=100):
    """Measure round-trip RPC latency."""
    latencies = []

    # Warmup
    for _ in range(10):
        rpc.rpc_sync(worker_name, ping_pong, args=(torch.tensor(1.0),))

    # Benchmark
    for _ in range(num_iterations):
        start = time.perf_counter()
        rpc.rpc_sync(worker_name, ping_pong, args=(torch.tensor(1.0),))
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    avg = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]

    print(f"RPC Latency: avg={avg:.2f}ms, p50={p50:.2f}ms, p99={p99:.2f}ms")
```

### Measuring RPC Throughput

```python
def benchmark_rpc_throughput(worker_name, tensor_size=1000000, num_iterations=100):
    """Measure RPC throughput with tensor transfer."""
    tensor = torch.randn(tensor_size)

    # Warmup
    for _ in range(10):
        rpc.rpc_sync(worker_name, ping_pong, args=(tensor,))

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        rpc.rpc_sync(worker_name, ping_pong, args=(tensor,))
    end = time.perf_counter()

    total_bytes = tensor_size * 4 * 2 * num_iterations  # Send + receive
    elapsed = end - start
    throughput = total_bytes / elapsed / 1e9  # GB/s

    print(f"RPC Throughput: {throughput:.2f} GB/s ({elapsed/num_iterations*1000:.2f} ms/iter)")
```

### Comparing Sync vs Async RPC

```python
def benchmark_sync_vs_async(worker_name, num_calls=100):
    """Compare sync and async RPC performance."""
    data = torch.randn(1000)

    # Sync benchmark
    start = time.perf_counter()
    for _ in range(num_calls):
        rpc.rpc_sync(worker_name, ping_pong, args=(data,))
    sync_time = time.perf_counter() - start

    # Async benchmark
    start = time.perf_counter()
    futures = [rpc.rpc_async(worker_name, ping_pong, args=(data,))
               for _ in range(num_calls)]
    for f in futures:
        f.wait()
    async_time = time.perf_counter() - start

    print(f"Sync:  {sync_time:.3f}s ({sync_time/num_calls*1000:.2f} ms/call)")
    print(f"Async: {async_time:.3f}s ({async_time/num_calls*1000:.2f} ms/call)")
    print(f"Async speedup: {sync_time/async_time:.2f}x")
```

---

## Parameter Server Example

A complete parameter server training example using RPC.

### Architecture

```
Worker 0 (Parameter Server)
  - Stores model parameters
  - Applies gradients
  - Returns updated parameters

Workers 1-N (Trainers)
  - Pull parameters from PS
  - Compute forward/backward
  - Push gradients to PS
```

### Implementation

```python
import os
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer

# ---- Parameter Server ----

class ParameterServer(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.model(x)

    def get_param_rrefs(self):
        return [rpc.RRef(p) for p in self.model.parameters()]

param_server_rref = None

def init_parameter_server():
    global param_server_rref
    param_server_rref = rpc.RRef(ParameterServer())
    return param_server_rref

def get_param_server():
    global param_server_rref
    return param_server_rref

# ---- Trainer ----

def train_batch(ps_rref, data, target):
    """Execute one training step on the trainer."""
    def forward_fn(model, x):
        return model(x)

    with dist_autograd.context() as ctx_id:
        # Forward pass on parameter server
        output = rpc.rpc_sync(
            ps_rref.owner_name(),
            forward_fn,
            args=(ps_rref.local_value(), data)
        )

        # Compute loss locally
        loss = nn.functional.cross_entropy(output, target)

        # Distributed backward
        dist_autograd.backward(ctx_id, [loss])

        return loss, ctx_id

def run_trainer(rank, world_size, dataloader, num_epochs):
    """Main trainer loop."""
    # Get parameter server RRef
    ps_rref = rpc.rpc_sync(
        "worker_0",
        get_param_server,
    )

    # Get parameter RRefs for distributed optimizer
    param_rrefs = rpc.rpc_sync(
        "worker_0",
        lambda ps: ps.get_param_rrefs(),
        args=(ps_rref.local_value(),)
    )

    # Create distributed optimizer
    optimizer = DistributedOptimizer(
        torch.optim.SGD,
        parameter_rrefs=param_rrefs,
        lr=0.01,
    )

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for data, target in dataloader:
            loss, ctx_id = train_batch(ps_rref, data, target)

            # Step the distributed optimizer
            optimizer.step(ctx_id)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Trainer {rank}, Epoch {epoch}: avg_loss={avg_loss:.4f}")

# ---- Main ----

def main():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    rpc.init_rpc(f"worker_{rank}", rank=rank, world_size=world_size)

    if rank == 0:
        # Parameter server
        init_parameter_server()
        print("Parameter server initialized")
        # Keep alive until training is done
        rpc.barrier()
    else:
        # Trainers
        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(
            torch.randn(1000, 784),
            torch.randint(0, 10, (1000,))
        )
        dataloader = DataLoader(dataset, batch_size=32)

        run_trainer(rank, world_size, dataloader, num_epochs=10)
        rpc.barrier()

    rpc.shutdown()

if __name__ == '__main__':
    main()
```

### Launching

```bash
# Launch with torchrun
torchrun --nproc_per_node=4 rpc_train.py

# Or with mp.spawn
# python rpc_train.py
```

### Asynchronous Parameter Server

For higher throughput, use async RPC:

```python
def async_trainer(rank, world_size, dataloader, num_epochs):
    ps_rref = rpc.rpc_sync("worker_0", get_param_server)

    param_rrefs = rpc.rpc_sync(
        "worker_0",
        lambda ps: ps.get_param_rrefs(),
        args=(ps_rref.local_value(),)
    )

    optimizer = DistributedOptimizer(
        torch.optim.Adam,
        parameter_rrefs=param_rrefs,
        lr=0.001,
    )

    for epoch in range(num_epochs):
        futures = []

        for data, target in dataloader:
            with dist_autograd.context() as ctx_id:
                output = rpc.rpc_sync(
                    ps_rref.owner_name(),
                    lambda m, x: m(x),
                    args=(ps_rref.local_value(), data)
                )
                loss = nn.functional.cross_entropy(output, target)
                dist_autograd.backward(ctx_id, [loss])
                optimizer.step(ctx_id)

        print(f"Trainer {rank}, Epoch {epoch} complete")
```

---

## RPC Best Practices

### 1. Use Async RPC for High Throughput

```python
# Instead of sequential sync calls:
for worker in workers:
    rpc.rpc_sync(worker, process, args=(data,))

# Use async calls in parallel:
futures = [rpc.rpc_async(worker, process, args=(data,)) for worker in workers]
results = [f.wait() for f in futures]
```

### 2. Minimize Data Transfer

```python
# Bad: Send entire model every time
rpc.rpc_sync(worker, train, args=(model, data))

# Good: Send only the data, keep model on the worker
rpc.rpc_sync(worker, train_with_local_model, args=(data,))
```

### 3. Use RRefs for Large Objects

```python
# Bad: Copy large tensors back and forth
params = rpc.rpc_sync("ps", get_params)  # Copies all params
result = rpc.rpc_sync("ps", compute, args=(params,))  # Sends params back

# Good: Use RRef to reference remote object
param_rref = rpc.remote("ps", create_params)
result = param_rref.rpc_sync(lambda p: p.compute())
```

### 4. Set Appropriate Timeouts

```python
options = TensorPipeRpcBackendOptions(
    rpc_timeout=300.0,  # 5 minutes for long operations
)
```

### 5. Batch RPC Calls

```python
# Instead of many small RPCs:
for item in items:
    rpc.rpc_sync(worker, process_one, args=(item,))

# Batch into fewer RPCs:
rpc.rpc_sync(worker, process_batch, args=(items,))
```
