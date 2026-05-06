# Distributed Collective Operations

All collective and point-to-point operations in `torch.distributed`.

```python
import torch.distributed as dist
from torch.distributed import ReduceOp
```

---

## ReduceOp Enum

```python
ReduceOp.SUM          # element-wise sum
ReduceOp.PRODUCT      # element-wise product
ReduceOp.MAX          # element-wise maximum
ReduceOp.MIN          # element-wise minimum
ReduceOp.AVG          # element-wise average (divides by world_size)
ReduceOp.BAND         # bitwise AND (integer tensors)
ReduceOp.BOR          # bitwise OR
ReduceOp.BXOR         # bitwise XOR
ReduceOp.PREMUL_SUM   # multiply by scalar before summing
```

---

## all_reduce

Reduces tensor data across all ranks; result written in-place to all ranks.

```python
dist.all_reduce(
    tensor: Tensor,               # [in-place] input and output
    op: ReduceOp = ReduceOp.SUM,
    group: ProcessGroup = None,
    async_op: bool = False,
) -> Optional[dist.Work]
```

```python
tensor = torch.ones(4, device="cuda") * dist.get_rank()
dist.all_reduce(tensor, op=ReduceOp.SUM)
# All ranks: [0+1+2+3, ...] = [6,6,6,6] for 4 ranks

# Async
work = dist.all_reduce(tensor, async_op=True)
work.wait()
```

---

## all_gather / all_gather_into_tensor

Gathers tensors from all ranks; all ranks receive all data.

```python
dist.all_gather(
    tensor_list: List[Tensor],    # pre-allocated output list
    tensor: Tensor,               # input from this rank
    group: ProcessGroup = None,
    async_op: bool = False,
) -> Optional[dist.Work]

dist.all_gather_into_tensor(
    output_tensor: Tensor,        # contiguous output (world_size * input_size)
    input_tensor: Tensor,
    group: ProcessGroup = None,
    async_op: bool = False,
) -> Optional[dist.Work]
```

```python
ws = dist.get_world_size()
tensor = torch.ones(4, device="cuda") * dist.get_rank()
gathered = [torch.zeros(4, device="cuda") for _ in range(ws)]
dist.all_gather(gathered, tensor)

# Efficient contiguous version
output = torch.zeros(4 * ws, device="cuda")
dist.all_gather_into_tensor(output, tensor)
```

---

## broadcast

Sends tensor from source rank to all other ranks.

```python
dist.broadcast(
    tensor: Tensor,               # data on src; overwritten on others
    src: int,                     # source rank
    group: ProcessGroup = None,
    async_op: bool = False,
) -> Optional[dist.Work]
```

```python
tensor = torch.zeros(5, device="cuda")
if dist.get_rank() == 0:
    tensor = torch.tensor([1., 2., 3., 4., 5.], device="cuda")
dist.broadcast(tensor, src=0)
```

---

## reduce

Reduces data to a single destination rank.

```python
dist.reduce(
    tensor: Tensor,               # in-place; result on dst only
    dst: int,
    op: ReduceOp = ReduceOp.SUM,
    group: ProcessGroup = None,
    async_op: bool = False,
) -> Optional[dist.Work]
```

---

## reduce_scatter / reduce_scatter_tensor

Reduces then scatters result; each rank gets one chunk.

```python
dist.reduce_scatter(
    output: Tensor,               # output (1/world_size of input)
    input_list: List[Tensor],     # list of tensors to reduce-scatter
    op: ReduceOp = ReduceOp.SUM,
    group: ProcessGroup = None,
    async_op: bool = False,
) -> Optional[dist.Work]

dist.reduce_scatter_tensor(
    output: Tensor,               # output tensor
    input: Tensor,                # contiguous (world_size * output_size)
    op: ReduceOp = ReduceOp.SUM,
    group: ProcessGroup = None,
    async_op: bool = False,
) -> Optional[dist.Work]
```

```python
ws = dist.get_world_size()
input_list = [torch.ones(3, device="cuda") * (rank * ws + i) for i in range(ws)]
output = torch.zeros(3, device="cuda")
dist.reduce_scatter(output, input_list)

# Contiguous version
input_tensor = torch.arange(ws * 3, dtype=torch.float32, device="cuda")
output = torch.zeros(3, device="cuda")
dist.reduce_scatter_tensor(output, input_tensor)
```

---

## scatter

Scatters list of tensors from source rank to all ranks.

```python
dist.scatter(
    tensor: Tensor,               # [in-place] output
    scatter_list: List[Tensor] = None,  # required on src only
    src: int = 0,
    group: ProcessGroup = None,
    async_op: bool = False,
) -> Optional[dist.Work]
```

---

## gather

Gathers tensors to a single destination rank.

```python
dist.gather(
    tensor: Tensor,               # input
    gather_list: List[Tensor] = None,  # required on dst only
    dst: int = 0,
    group: ProcessGroup = None,
    async_op: bool = False,
) -> Optional[dist.Work]
```

---

## send / recv (Point-to-Point)

```python
dist.send(tensor: Tensor, dst: int, tag: int = 0, group=None) -> Optional[dist.Work]
dist.recv(tensor: Tensor, src: int = None, tag: int = 0, group=None) -> Optional[dist.Work]

# Async versions
dist.isend(tensor, dst, tag=0, group=None) -> dist.Work
dist.irecv(tensor, src=None, tag=0, group=None) -> dist.Work
```

```python
rank = dist.get_rank()
tensor = torch.zeros(5, device="cuda")
if rank == 0:
    tensor = torch.tensor([1., 2., 3., 4., 5.], device="cuda")
    dist.send(tensor, dst=1)
elif rank == 1:
    dist.recv(tensor, src=0)

# Async ring pattern
send_rank = (rank + 1) % ws
recv_rank = (rank - 1) % ws
send_req = dist.isend(tensor, dst=send_rank)
recv_req = dist.irecv(recv_buf, src=recv_rank)
send_req.wait()
recv_req.wait()
```

---

## all_to_all

Each rank sends distinct data to and receives from every other rank.

```python
dist.all_to_all(
    output_tensor_list: List[Tensor],
    input_tensor_list: List[Tensor],
    group: ProcessGroup = None,
    async_op: bool = False,
) -> Optional[dist.Work]

dist.all_to_all_single(
    output: Tensor,
    input: Tensor,
    output_split_sizes: List[int] = None,
    input_split_sizes: List[int] = None,
    group: ProcessGroup = None,
    async_op: bool = False,
) -> Optional[dist.Work]
```

---

## barrier

Synchronizes all processes.

```python
dist.barrier(
    group: ProcessGroup = None,
    device_ids: List[int] = None,
    async_op: bool = False,
) -> Optional[dist.Work]

dist.monitored_barrier(group=None, timeout=None, wait_all_ranks=False)
```

---

## Object Collectives

Communicate arbitrary Python objects via pickle serialization.

### broadcast_object_list

```python
dist.broadcast_object_list(
    object_list: List[Any],       # [in-place] src provides, others receive
    src: int = 0,
    group: ProcessGroup = None,
    device: Union[str, torch.device] = "cpu",
) -> None
```

```python
config = None
if dist.get_rank() == 0:
    config = {"lr": 0.001, "epochs": 100}
obj_list = [config]
dist.broadcast_object_list(obj_list, src=0)
config = obj_list[0]
```

### all_gather_object

```python
dist.all_gather_object(
    object_list: List[Any],       # output (populated on all ranks)
    obj: Any,                     # object from this rank
    group: ProcessGroup = None,
) -> None
```

```python
local_loss = compute_loss()
losses = [None] * dist.get_world_size()
dist.all_gather_object(losses, local_loss)
```

---

## Process Groups

```python
# Create subgroup
group = dist.new_group(ranks=[0, 1, 2], backend="nccl")

# Split into subgroups
subgroup, _ = dist.new_subgroups(group_size=2)

# Rank mapping
group_rank = dist.get_group_rank(group, global_rank)
global_rank = dist.get_global_rank(group, group_rank)
```

---

## Work Handle (Async Operations)

```python
work = dist.all_reduce(tensor, async_op=True)
work.is_completed()        # bool
work.wait()                # blocks until done
work.wait(timeout=datetime.timedelta(seconds=10))
work.get_future()          # returns Future
```

### Overlapping Communication and Computation

```python
works = []
for param in model.parameters():
    if param.grad is not None:
        w = dist.all_reduce(param.grad, async_op=True)
        works.append(w)
# overlap with other work
for w in works:
    w.wait()
```

---

## DeviceMesh (torch.distributed.device_mesh)

```python
from torch.distributed.device_mesh import DeviceMesh

mesh_1d = DeviceMesh("cuda", list(range(world_size)))
mesh_2d = DeviceMesh("cuda", [[0, 1, 2, 3], [4, 5, 6, 7]])
```

---

## Backend Notes

| Backend | Tensors | send/recv | Speed (GPU) |
|---------|---------|-----------|-------------|
| NCCL    | CUDA only | No     | Fastest     |
| Gloo    | CPU + GPU | Yes    | Slower      |
| MPI     | CPU + GPU | Yes    | Varies      |

NCCL environment variables: `NCCL_DEBUG=INFO`, `NCCL_ALGO=Ring`, `NCCL_SOCKET_IFNAME=eth0`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`

---

## Complete Training Example

```python
import os, torch, torch.nn.functional as F
import torch.distributed as dist

def train(model, dataloader, optimizer):
    rank, ws = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ["LOCAL_RANK"])
    model = model.to(lr)

    for epoch in range(num_epochs):
        for data, target in dataloader:
            data, target = data.to(lr), target.to(lr)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()

            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            optimizer.step()

        # Aggregate metrics
        metrics = torch.tensor([loss.item(), 0.0], device="cuda")
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        if rank == 0:
            print(f"Epoch {epoch}: Loss={metrics[0].item()/ws:.4f}")
```
