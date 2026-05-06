# Distributed Pipeline Parallelism

Pipeline parallelism splits a model into sequential stages across GPUs, processing micro-batches concurrently.

```python
import torch
import torch.nn as nn
```

---

## torch.distributed.pipeline.sync.Pipe

```python
from torch.distributed.pipeline.sync import Pipe

pipe_model = Pipe(
    module: nn.Sequential,          # must be nn.Sequential
    balance: List[int] = None,      # layers per device, e.g. [4, 4]
    chunks: int = 1,                # number of micro-batches
    checkpoint: str = "except_last", # "always" | "except_last" | "never"
)
```

| Parameter | Description |
|-----------|-------------|
| `module` | Model as `nn.Sequential`. Layers partitioned by `balance`. |
| `balance` | List of layer counts per stage. Length = number of stages. |
| `chunks` | Micro-batch count. Higher = less bubble, more communication. |
| `checkpoint` | Activation recomputation strategy for memory savings. |

### Basic Example

```python
model = nn.Sequential(
    nn.Linear(1024, 1024), nn.ReLU(),     # Stage 0
    nn.Linear(1024, 1024), nn.ReLU(),     # Stage 0
    nn.Linear(1024, 512),  nn.ReLU(),     # Stage 1
    nn.Linear(512, 10),                   # Stage 1
)

pipe_model = Pipe(model, balance=[4, 4], chunks=8)
output = pipe_model(input)
loss = criterion(output, target)
loss.backward()
```

---

## Micro-Batching (Chunking)

Input batch is split into `chunks` micro-batches flowing through the pipeline concurrently.

```
chunks=4, 2 stages:
Time:   t0    t1    t2    t3    t4    t5    t6    t7
Stage0: [MB0] [MB1] [MB2] [MB3]
Stage1:       [MB0] [MB1] [MB2] [MB3]

Pipeline bubble fraction = (p - 1) / m    (p=stages, m=chunks)
4 stages,  8 chunks => bubble = 3/8  = 37.5%
4 stages, 32 chunks => bubble = 3/32 =  9.4%
```

---

## GPipe Schedule (Default)

All forward micro-batches complete, then all backward micro-batches.

```
Time:    t0   t1   t2   t3   t4   t5   t6   t7
Stage 0: F(0) F(1) F(2) F(3)                 B(3) B(2) B(1) B(0)
Stage 1:            F(0) F(1) F(2) F(3) B(3) B(2) B(1) B(0)
```

- High memory: all forward activations stored before backward.
- Simple implementation.

---

## 1F1B (One Forward One Backward) Schedule

Alternates forward and backward passes to reduce peak memory.

```
Time:    t0   t1   t2   t3   t4   t5   t6   t7
Stage 0: F(0) F(1) F(2) B(0) F(3) B(1) B(2) B(3)
Stage 1:            F(0) F(1) B(0) F(2) B(1) F(3) B(2) B(3)
```

- Peak memory: O(p * batch_size) vs O(m * batch_size) for GPipe.
- Same pipeline bubble fraction.
- Available via `torch.distributed.pipelining` API.

---

## Virtual Stages

Subdivide a single GPU stage into multiple virtual stages for finer scheduling.

```
2 GPUs, 4 virtual stages:
GPU 0: Virtual Stage 0, Virtual Stage 1
GPU 1: Virtual Stage 2, Virtual Stage 3
```

Reduces bubbles without requiring more physical devices.

---

## Activation Checkpointing

```python
# Never checkpoint (most memory, fastest)
Pipe(model, balance=[4,4], checkpoint="never")

# Checkpoint all micro-batches (least memory, recomputes all)
Pipe(model, balance=[4,4], checkpoint="always")

# Default: checkpoint all except last micro-batch
Pipe(model, balance=[4,4], checkpoint="except_last")
```

---

## Skip Connections

```python
from torch.distributed.pipeline.sync.skip import PipelineStageSkipped

class ConditionalStage(nn.Module):
    def forward(self, x):
        if should_skip(x):
            raise PipelineStageSkipped("Skip this stage")
        return process(x)
```

---

## Pipeline + DDP (2D Parallelism)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.pipeline.sync import Pipe

# Assume 4 GPUs: 2 pipeline stages x 2 data-parallel replicas
rank = dist.get_rank()
pipeline_rank = rank // 2
dp_rank = rank % 2

dp_group = dist.new_group(ranks=[pipeline_rank * 2, pipeline_rank * 2 + 1])

# Create stage
if pipeline_rank == 0:
    stage = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU())
else:
    stage = nn.Sequential(nn.Linear(2048, 10))

stage = DDP(stage.cuda(), device_ids=[rank % 2], process_group=dp_group)
# Use Pipe or manual orchestration for inter-stage communication
```

---

## torch.distributed.pipelining (Newer API)

```python
from torch.distributed.pipelining import pipeline, PipelineStage, ScheduleGPipe

# Define stage
class MyStage(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    def forward(self, x):
        return self.layer(x)

# Create pipeline stage
stage = PipelineStage(
    MyStage(nn.Linear(1024, 1024)),
    stage_index=rank,
    num_stages=world_size,
    device=f"cuda:{local_rank}",
)

# Schedule
schedule = ScheduleGPipe(stage, chunks=8)
# Run forward + backward through schedule
```

---

## Balance Selection Guidelines

1. **Balance compute, not layer count**: Profile FLOPs per stage.
2. **Minimize boundary activation size**: Split where tensors are smallest.
3. **Tune chunks**: Start with 8, adjust based on memory vs throughput.
4. **Use checkpoint="except_last"**: Good default balance.

---

## Complete Transformer Example

```python
import torch, torch.nn as nn
from torch.distributed.pipeline.sync import Pipe

class TransformerBlock(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + a)
        return self.norm2(x + self.ff(x))

layers = []
for _ in range(12):
    layers.append(TransformerBlock(768, 12, 3072))
model = nn.Sequential(*layers)

pipe = Pipe(model, balance=[4, 4, 4], chunks=8, checkpoint="except_last")
optimizer = torch.optim.AdamW(pipe.parameters(), lr=1e-4)

for input_ids, labels in dataloader:
    optimizer.zero_grad()
    loss = F.cross_entropy(pipe(input_ids), labels)
    loss.backward()
    optimizer.step()
```
