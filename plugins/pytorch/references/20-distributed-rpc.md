# Distributed RPC

`torch.distributed.rpc` provides remote procedure calls, remote references, distributed autograd, and distributed optimization.

```python
import torch.distributed.rpc as rpc
```

---

## Initialization

```python
rpc.init_rpc(
    name: str = None,               # unique worker name, default "worker_{rank}"
    rank: int = None,               # global rank
    world_size: int = None,         # total workers
    rpc_backend_options=None,       # backend config
)

rpc.shutdown(graceful: bool = True)
rpc.is_available() -> bool
rpc.get_worker_info() -> WorkerInfo
rpc.get_worker_info(name_or_rank) -> WorkerInfo
```

### Backend Options

```python
from torch.distributed.rpc.backend import TensorPipeRpcBackendOptions

options = TensorPipeRpcBackendOptions(
    num_worker_threads=8,
    rpc_timeout=60.0,
    init_method="tcp://localhost:29501",
    device_maps=None,              # cross-device mapping
)
rpc.init_rpc(f"worker_{rank}", rank=rank, world_size=ws, rpc_backend_options=options)
```

---

## rpc_sync

Execute function on remote worker synchronously. Blocks until result returned.

```python
result = rpc.rpc_sync(
    to: Union[str, int],           # worker name or rank
    func: Callable,
    args: tuple = None,
    kwargs: dict = None,
    timeout: float = -1.0,
)
```

```python
def add(x, y): return x + y
result = rpc.rpc_sync("worker_1", add, args=(torch.tensor(1.), torch.tensor(2.)))
# result = tensor(3.0)
result = rpc.rpc_sync(1, add, args=(1, 2))  # using rank
```

---

## rpc_async

Execute function asynchronously. Returns `Future` immediately.

```python
future = rpc.rpc_async(to, func, args=None, kwargs=None, timeout=-1.0)
result = future.wait()
```

```python
# Fan-out to multiple workers
futures = [rpc.rpc_async(f"worker_{w}", process, args=(data,))
           for w in range(1, world_size)]
results = [f.wait() for f in futures]
```

---

## remote

Create a remote reference (RRef) to an object on a remote worker.

```python
rref = rpc.remote(to, func, args=None, kwargs=None, timeout=-1.0)
# Returns RRef pointing to the remote object
```

---

## RRef (Remote Reference)

Reference to an object living on a remote worker.

```python
# Create via rpc.remote
rref = rpc.remote("worker_1", MyClass, args=(100,))

rref.to_here(timeout=-1.0)       # copy value to local worker
rref.local_value()               # get value (owner only)
rref.owner()                     # worker ID of owner
rref.owner_name()                # worker name of owner

# Execute on owner using RRef value
rref.rpc_sync(lambda obj: obj.method())     # sync
rref.rpc_async(lambda obj: obj.method())    # async, returns Future
```

---

## Distributed Autograd

Automatic differentiation across process boundaries.

```python
import torch.distributed.autograd as dist_autograd

with dist_autograd.context() as ctx_id:
    # All RPCs within this context are tracked for gradients
    result = rpc.rpc_sync("worker_1", forward_fn, args=(input_tensor,))
    loss = compute_loss(result)
    dist_autograd.backward(ctx_id, [loss])
    grads = dist_autograd.get_gradients(ctx_id)
    # grads: dict mapping tensor -> gradient tensor
```

---

## DistributedOptimizer

Applies gradients across workers using distributed autograd.

```python
from torch.distributed.optim import DistributedOptimizer

dist_opt = DistributedOptimizer(
    optimizer_class,                # e.g., torch.optim.SGD
    parameter_rrefs: List[RRef],   # RRefs to parameters
    args=None, kwargs=None,        # optimizer constructor args
    lr=0.01,                       # optimizer kwargs
)
```

---

## RemoteModule

Create a module replica on a remote worker with a convenient interface.

```python
from torch.distributed.nn.api.remote_module import RemoteModule

remote_linear = RemoteModule(
    on="worker_1",                 # target worker
    module_cls=torch.nn.Linear,    # module class
    args=(784, 256),               # constructor args
)
output = remote_linear(torch.randn(32, 784))  # forward on remote worker

# Get remote parameter references
for prref in remote_linear.remote_parameters():
    param = prref.to_here()
```

---

## Pipeline Parallel with RPC

```python
def pipeline_forward(stages, input_tensor, chunks=4):
    """Forward through pipeline stages on different workers."""
    micro_batches = input_tensor.chunk(chunks)
    for stage_rref in stages:
        futures = [rpc.rpc_async(
            stage_rref.owner_name(),
            lambda s, x: s.forward(x),
            args=(stage_rref.local_value(), mb))
            for mb in micro_batches]
        micro_batches = [f.wait() for f in futures]
    return torch.cat(micro_batches)
```

---

## Parameter Server Example

```python
import torch, torch.nn as nn
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer

class ParameterServer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    def forward(self, x): return self.model(x)
    def get_param_rrefs(self): return [rpc.RRef(p) for p in self.model.parameters()]

def run_trainer(ps_rref, dataloader):
    param_rrefs = rpc.rpc_sync("worker_0",
        lambda ps: ps.get_param_rrefs(), args=(ps_rref.local_value(),))
    optimizer = DistributedOptimizer(torch.optim.SGD, param_rrefs, lr=0.01)

    for data, target in dataloader:
        with dist_autograd.context() as ctx_id:
            output = rpc.rpc_sync(ps_rref.owner_name(),
                lambda m, x: m(x), args=(ps_rref.local_value(), data))
            loss = nn.functional.cross_entropy(output, target)
            dist_autograd.backward(ctx_id, [loss])
            optimizer.step(ctx_id)
```

---

## Best Practices

1. **Use async RPC** for fan-out: `rpc_async` to multiple workers in parallel.
2. **Minimize data transfer**: Keep large objects on remote workers via RRefs.
3. **Batch RPC calls**: Send batches, not individual items.
4. **Set appropriate timeouts**: `TensorPipeRpcBackendOptions(rpc_timeout=300.0)`.
5. **Use RRef** for large objects instead of copying back and forth.
