# 09 — Relax Distributed — Disco Runtime

## Overview

Disco is TVM's distributed runtime for executing models across multiple devices. When a model is too large to fit on a single GPU, the `relax.distributed` module annotates how tensors should be partitioned and placed across a mesh of devices at compile time. Disco manages the runtime execution across workers.

---

## Device Mesh

### Concept
A device mesh defines the physical arrangement of devices used for distributed computation.

```python
from tvm.relax.distributed import DeviceMesh

# Define mesh of 4 GPUs
mesh = DeviceMesh(shape=(4,))
# 2D mesh: 2 nodes x 2 GPUs
mesh_2d = DeviceMesh(shape=(2, 2))
```

### Tensor Partitioning
Specify how tensors are distributed across the mesh:

```python
from tvm.relax.distributed import ShardDim, ShardSpec

# Shard tensor along dimension 0 across 4 devices
shard_spec = ShardSpec(mesh, [ShardDim(axis=0, num_shards=4)])
```

---

## Compile-Time Annotations

The `relax.distributed` module provides compile-time annotations for tensor distribution:

```python
from tvm.script import relax as R, distributed as Dist

@R.function
def distributed_matmul(
    x: R.Tensor(("n", "d"), "float32"),
    w: R.Tensor(("d", "m"), "float32"),
) -> R.Tensor(("n", "m"), "float32"):
    # Annotations specify how tensors are distributed
    ...
```

---

## Session Abstraction

### Overview
The central abstraction is the **Session**, which owns a group of workers and exposes a SPMD-style programming interface.

```python
from tvm.runtime import disco

# Create session
session = disco.ThreadedSession(num_workers=4)
```

### SPMD Programming Model
- Every object that lives on workers is represented by a **DRef** (distributed reference)
- When the controller invokes a `DPackedFunc`, all workers execute the same PackedFunc call synchronously
- Each worker operates on its own local shard

---

## DRef (Distributed Reference)

A `DRef` maps to a concrete value on each worker:

```python
# Create distributed tensor
data_dref = session.empty((128, 128), dtype="float32", device=tvm.cuda(0))

# Each worker gets its own local shard
# DRef provides the same interface as local NDArray
```

---

## DPackedFunc

Distributed PackedFunc calls execute on all workers simultaneously:

```python
# Call a function on all workers
result_dref = session.call_packed("my_func", data_dref)
```

---

## DModule

Compiled VM modules loaded into a session:

```python
# Load compiled module
dmod = session.import_module(exec)

# Call function
result = dmod["main"](input_data)
```

---

## Collective Operations

Disco provides collective operations backed by NCCL or RCCL for inter-worker communication.

### AllReduce
```python
# Sum across all workers
result = session.allreduce(data_dref, op="sum")
```

### AllGather
```python
# Gather from all workers
result = session.allgather(data_dref)
```

### Broadcast
```python
# Broadcast from worker 0 to all
result = session.broadcast(data_dref, src=0)
```

### Scatter
```python
# Scatter data to workers
result = session.scatter(data_dref, src=0)
```

### ReduceScatter
```python
# Reduce then scatter
result = session.reduce_scatter(data_dref, op="sum")
```

### Send/Recv
```python
# Point-to-point communication
session.send(data_dref, dst=1)
received = session.recv(src=1)
```

---

## Session Backends

### ThreadedSession
Workers as threads within a single process. Most common for multi-GPU inference.

```python
from tvm.runtime import disco

session = disco.ThreadedSession(num_workers=4)
# Workers share the same process
# Best for: single machine, multiple GPUs
```

### ProcessSession
Workers as separate OS processes connected by pipes.

```python
session = disco.ProcessSession(num_workers=4)
# Workers in separate processes
# Better isolation between workers
# Best for: isolation requirements, debugging
```

### SocketSession
Multi-node clusters connected via TCP sockets.

```python
session = disco.SocketSession(
    host="localhost",
    port=9000,
    num_workers=4,
)
# Workers across different machines
# Best for: multi-node inference
```

---

## NCCL / RCCL Integration

### NCCL Backend (NVIDIA)
```python
# NCCL is automatically used for collective operations on CUDA devices
session = disco.ThreadedSession(num_workers=4)
# Collective ops use NCCL under the hood
```

### RCCL Backend (AMD)
```python
# RCCL for AMD GPUs
session = disco.ThreadedSession(num_workers=4)
# Uses RCCL on ROCm devices
```

---

## Complete Example: Distributed Inference

```python
import tvm
from tvm import relax
from tvm.runtime import disco
import numpy as np

# 1. Build model
mod = relax.get_pipeline("zero")(mod)
exec = relax.build(mod, target="cuda")

# 2. Create distributed session
num_gpus = 4
session = disco.ThreadedSession(num_workers=num_gpus)

# 3. Load module into session
dmod = session.import_module(exec)

# 4. Prepare input data
data = np.random.randn(1, 784).astype("float32")
input_dref = session.copy_from_numpy(data)

# 5. Run distributed inference
result_dref = dmod["main"](input_dref)

# 6. Get results
result = session.copy_to_numpy(result_dref)
print(f"Output: {result}")
```

---

## Tensor Parallelism

For large models, tensor parallelism partitions individual operations across devices:

```python
# Linear layer partitioned across 4 GPUs
# Weight: (768, 3072) -> each GPU holds (768, 768)
# Each GPU computes partial result, then allreduce to combine

# Compile-time annotations specify the partitioning strategy
# Disco handles runtime coordination
```

---

## Debugging Distributed Execution

### Check Worker State
```python
# Inspect individual worker state
for i in range(num_workers):
    worker_state = session.get_worker_state(i)
    print(f"Worker {i}: {worker_state}")
```

### Logging
```python
import os
os.environ["TVM_DISCO_DEBUG"] = "1"  # Enable Disco debug logging
```

### Common Issues
- **NCCL errors**: Check CUDA/NCCL version compatibility
- **Timeout**: Increase socket timeout for large models
- **Memory**: Ensure each GPU has enough memory for its shard
