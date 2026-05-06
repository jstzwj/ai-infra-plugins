# PyTorch Distributed Pipeline Parallelism - Comprehensive Reference

This chapter covers pipeline parallelism in PyTorch, including concepts, APIs, and implementation details.

## Table of Contents

1. [Pipeline Parallelism Concepts](#pipeline-parallelism-concepts)
2. [torch.distributed.pipeline API](#torchdistributedpipeline-api)
3. [GPipe-Style Pipeline Parallelism](#gpipe-style-pipeline-parallelism)
4. [Micro-Batching and Chunking](#micro-batching-and-chunking)
5. [Schedules](#schedules)
6. [Skip Connections](#skip-connections)
7. [Device Assignment](#device-assignment)
8. [Pipeline Parallel with DDP](#pipeline-parallel-with-ddp)
9. [Virtual Stages](#virtual-stages)
10. [Checkpointing Within Pipeline](#checkpointing-within-pipeline)
11. [Forward and Backward Mechanics](#forward-and-backward-mechanics)
12. [Complete Example](#complete-example)

---

## Pipeline Parallelism Concepts

Pipeline parallelism splits a model into sequential stages, with each stage assigned to a different GPU. Data flows through the stages sequentially, and multiple micro-batches are processed concurrently to overlap computation across stages.

### Why Pipeline Parallelism?

1. **Large models**: When a model is too large to fit on a single GPU, pipeline parallelism allows distributing it across multiple GPUs.
2. **Memory efficiency**: Each GPU only needs to store its stage's parameters and activations.
3. **Communication efficiency**: Only activations at stage boundaries need to be communicated (vs. all parameters with DDP).
4. **Composability**: Can be combined with data parallelism for multi-dimensional scaling.

### Key Concepts

- **Stage**: A contiguous subset of model layers assigned to a specific GPU.
- **Micro-batch (chunk)**: A subdivision of the input batch. Multiple micro-batches are processed through the pipeline to enable parallelism.
- **Schedule**: The order in which micro-batches are processed across stages (e.g., GPipe, 1F1B).
- **Pipeline bubble**: Idle time when some stages are waiting for input from other stages.
- **Skip connection**: Connections that skip stages (e.g., residual connections across stages).

### Pipeline Parallelism vs Other Strategies

| Aspect | Pipeline Parallel | Data Parallel (DDP) | Tensor Parallel |
|--------|------------------|--------------------|-----------------| 
| Model size | Larger than 1 GPU | Fits on 1 GPU | Very large layers |
| Communication | Small (activations) | Large (gradients) | Large (activations) |
| Memory per GPU | Model shard + activations | Full model | Layer shard + activations |
| Bubble overhead | Yes (pipeline bubble) | No | No |
| Ease of use | Medium-Hard | Easy | Hard |

---

## torch.distributed.pipeline API

### PipelineStage

Represents a single stage in the pipeline.

```python
from torch.distributed.pipeline.sync import Pipe
```

### Pipe

The main class for pipeline parallelism in PyTorch.

```python
from torch.distributed.pipeline.sync import Pipe

pipe_model = Pipe(
    module,           # An nn.Sequential representing the full model
    balance=None,     # List of layer counts per device
    chunks=1,         # Number of micro-batches
    checkpoint='except_last'  # Checkpointing strategy
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | nn.Sequential | required | The model to pipeline. Must be an `nn.Sequential`. |
| `balance` | list[int] | None | A list of integers specifying how many layers to assign to each device. Length determines the number of pipeline stages. If None, uses even distribution. |
| `chunks` | int | 1 | Number of micro-batches. Higher values reduce pipeline bubbles but increase communication overhead. |
| `checkpoint` | str | 'except_last' | Checkpointing strategy for activation recomputation. One of 'always', 'except_last', or 'never'. |

### PipelineStageSkipped

Exception raised when a stage should be skipped (used with conditional computation).

```python
from torch.distributed.pipeline.sync.skip import PipelineStageSkipped
```

---

## GPipe-Style Pipeline Parallelism

### Basic Setup

```python
import torch
import torch.nn as nn
from torch.distributed.pipeline.sync import Pipe

# Create a model as nn.Sequential
model = nn.Sequential(
    nn.Linear(1024, 1024),   # Stage 0
    nn.ReLU(),
    nn.Linear(1024, 1024),   # Stage 0
    nn.ReLU(),
    nn.Linear(1024, 512),    # Stage 1
    nn.ReLU(),
    nn.Linear(512, 10),      # Stage 1
)

# Partition into 2 stages with 4 layers and 4 layers
balance = [4, 4]  # 4 layers per stage, 2 stages

# Create pipeline model
pipe_model = Pipe(model, balance=balance, chunks=8)

# Move stages to different devices
# Stage 0 goes to cuda:0, Stage 1 goes to cuda:1
# Pipe automatically moves partitions to consecutive CUDA devices

# Use the pipeline model like a regular model
output = pipe_model(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

### Balance Parameter

The `balance` parameter controls how layers are distributed across devices.

```python
# Unequal distribution
model = nn.Sequential(
    nn.Conv2d(3, 64, 3),    # Layer 0
    nn.ReLU(),               # Layer 1
    nn.Conv2d(64, 128, 3),  # Layer 2
    nn.ReLU(),               # Layer 3
    nn.MaxPool2d(2),         # Layer 4
    nn.Flatten(),            # Layer 5
    nn.Linear(128*13*13, 256), # Layer 6
    nn.ReLU(),               # Layer 7
    nn.Linear(256, 10),      # Layer 8
)

# 3 stages: layers 0-3 on GPU 0, layers 4-5 on GPU 1, layers 6-8 on GPU 2
pipe_model = Pipe(model, balance=[4, 2, 3], chunks=4)

# Equal distribution across 3 GPUs (3 layers each for 9 layers)
pipe_model = Pipe(model, balance=[3, 3, 3], chunks=4)
```

### Choosing Balance

Guidelines for choosing the balance parameter:
1. **Equal compute**: Try to balance FLOPs per stage, not just layer count.
2. **Memory balance**: Ensure each stage has similar memory usage.
3. **Communication**: Minimize data transfer at stage boundaries (e.g., put boundary where activation tensor is smallest).

---

## Micro-Batching and Chunking

### How Chunking Works

When `chunks > 1`, the input batch is split into multiple micro-batches that flow through the pipeline concurrently.

```
With chunks=4, 2 stages:

Time:   t0    t1    t2    t3    t4    t5    t6    t7
Stage0: [MB0] [MB1] [MB2] [MB3]
Stage1:       [MB0] [MB1] [MB2] [MB3]

With chunks=4, the pipeline bubble is reduced compared to chunks=1:
Time:   t0    t1    t2    t3    t4
Stage0: [FULL]
Stage1:       [FULL]
```

### Choosing the Number of Chunks

```python
# Few chunks (1-4): Less communication overhead, more pipeline bubble
pipe_model = Pipe(model, balance=balance, chunks=1)

# Many chunks (8-64): Less pipeline bubble, more communication overhead
pipe_model = Pipe(model, balance=balance, chunks=16)

# Trade-off:
# - Pipeline bubble fraction ≈ (p-1) / m where p = stages, m = chunks
# - With 4 stages and 8 chunks: bubble ≈ 3/8 = 37.5%
# - With 4 stages and 32 chunks: bubble ≈ 3/32 = 9.4%
```

### Chunking Implementation

```python
# Internal chunking process:
# 1. Input is split into 'chunks' micro-batches
# 2. Each micro-batch flows through the pipeline independently
# 3. Outputs are collected and concatenated

input_batch = torch.randn(128, 1024)  # Batch size 128
pipe_model = Pipe(model, balance=balance, chunks=4)
# Each micro-batch has size 32

output = pipe_model(input_batch)
# Output is the concatenation of all micro-batch outputs
# Shape: [128, 10] (same as if processed without chunking)
```

---

## Schedules

### GPipe Schedule

The GPipe schedule processes all micro-batches in the forward pass first, then all micro-batches in the backward pass.

```
GPipe Schedule (2 stages, 4 micro-batches):

Time:    t0    t1    t2    t3    t4    t5    t6    t7
Stage 0: F(0)  F(1)  F(2)  F(3)                    B(3)  B(2)  B(1)  B(0)
Stage 1:             F(0)  F(1)  F(2)  F(3)  B(3)  B(2)  B(1)  B(0)

F(n) = Forward micro-batch n
B(n) = Backward micro-batch n
```

This is the default schedule in `torch.distributed.pipeline.sync.Pipe`.

**Characteristics:**
- Simple to implement.
- High memory usage: All forward activations are stored before backward begins.
- Pipeline bubble: `(p-1) / m` where p is stages and m is micro-batches.

### 1F1B (One Forward One Backward) Schedule

The 1F1B schedule alternates between forward and backward passes to reduce peak memory.

```
1F1B Schedule (2 stages, 4 micro-batches):

Time:    t0    t1    t2    t3    t4    t5    t6    t7
Stage 0: F(0)  F(1)  F(2)  B(0)  F(3)  B(1)  B(2)  B(3)
Stage 1:             F(0)  F(1)  B(0)  F(2)  B(1)  F(3)  B(2)  B(3)

F(n) = Forward micro-batch n
B(n) = Backward micro-batch n
```

**Characteristics:**
- Lower peak memory than GPipe (releases activations earlier).
- Same pipeline bubble as GPipe.
- More complex to implement.

```python
# Note: torch.distributed.pipeline.sync.Pipe uses GPipe schedule
# For 1F1B, consider using the PipeDream-style implementation or
# the newer torch.distributed.pipelining API
```

### Schedule Comparison

| Aspect | GPipe | 1F1B |
|--------|-------|------|
| Peak activation memory | O(m * batch_size) | O(p * batch_size) |
| Pipeline bubble | (p-1)/m | (p-1)/m |
| Implementation complexity | Simple | Complex |
| Communication | Same | Same |
| Recommended for | Memory-rich setups | Memory-constrained |

Where `p` = number of pipeline stages, `m` = number of micro-batches.

---

## Skip Connections

### SkipPotEnumerate

Handles skip connections (e.g., residual connections) that cross pipeline stage boundaries.

```python
from torch.distributed.pipeline.sync.skip import SkipPotEnumerate
```

### Handling Residual Connections Across Stages

When a model has skip connections that span multiple stages, special handling is needed to pass the skip tensor through intermediate stages.

```python
import torch.nn as nn
from torch.distributed.pipeline.sync import Pipe

# Model with skip connections
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x + residual  # Skip connection

# For pipeline parallelism with skip connections,
# use torch.distributed.pipeline.sync.skip utilities
# to handle tensors that skip stages
```

### PipelineStageSkipped

Used to skip a stage in conditional computation scenarios.

```python
from torch.distributed.pipeline.sync.skip import PipelineStageSkipped

class ConditionalStage(nn.Module):
    def forward(self, x):
        if some_condition(x):
            raise PipelineStageSkipped("Skip this stage")
        return process(x)
```

---

## Device Assignment

### Automatic Device Assignment

By default, Pipe assigns stages to consecutive CUDA devices starting from cuda:0.

```python
# Stage 0 -> cuda:0, Stage 1 -> cuda:1, etc.
pipe_model = Pipe(model, balance=[4, 4], chunks=8)
```

### Manual Device Assignment

For more control, move layers to specific devices before creating the Pipe.

```python
import torch.nn as nn

# Manually assign layers to devices
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage0 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
        ).to('cuda:0')

        self.stage1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        ).to('cuda:1')

    def forward(self, x):
        x = self.stage0(x.to('cuda:0'))
        x = self.stage1(x.to('cuda:1'))
        return x
```

### Multi-Node Device Assignment

For pipeline parallelism across nodes, each node's stages use its local GPUs.

```python
# Node 0: stages 0-1 on local GPUs 0-1
# Node 1: stages 2-3 on local GPUs 0-1

# Setup requires process group configuration:
# 4 processes, each responsible for one stage
import torch.distributed as dist

rank = dist.get_rank()
local_rank = rank % 2  # Local GPU index on this node
torch.cuda.set_device(local_rank)
```

---

## Pipeline Parallel with DDP

Pipeline parallelism can be combined with data parallelism for 2D parallelism.

### 2D Parallelism Pattern

```
Model stages:     Stage 0     Stage 1
                  (Layers 1-4) (Layers 5-8)
                  ----------   ----------
DDP Group 0:  GPU 0, GPU 1   GPU 2, GPU 3
DDP Group 1:  GPU 4, GPU 5   GPU 6, GPU 7

Each column is a pipeline (2 stages).
Each row within a stage uses DDP for data parallelism.
```

### Implementation

```python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.pipeline.sync import Pipe
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_2d_parallel(pipeline_rank, dp_rank, pipeline_group, dp_group):
    # Create the model stage
    if pipeline_rank == 0:
        stage = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
        )
    else:
        stage = nn.Sequential(
            nn.Linear(2048, 10),
        )

    local_rank = int(os.environ['LOCAL_RANK'])
    stage = stage.to(local_rank)

    # Apply DDP within the data parallel group
    stage = DDP(stage, device_ids=[local_rank], process_group=dp_group)

    return stage

# Create process groups for 2D parallelism
world_size = dist.get_world_size()
rank = dist.get_rank()
pipeline_stages = 2
dp_size = world_size // pipeline_stages

pipeline_rank = rank // dp_size
dp_rank = rank % dp_size

# Create pipeline group (all ranks in the same pipeline)
pipeline_ranks = list(range(pipeline_rank * dp_size, (pipeline_rank + 1) * dp_size))
pipeline_group = dist.new_group(ranks=pipeline_ranks)

# Create DP group (ranks at the same pipeline stage across pipelines)
dp_ranks = [pipeline_rank * dp_size + i for i in range(dp_size)]
dp_group = dist.new_group(ranks=dp_ranks)
```

---

## Virtual Stages

Virtual stages allow subdividing a single model stage into multiple virtual stages for more fine-grained pipeline parallelism without adding more physical devices.

### Concept

Instead of assigning one stage per GPU, you can assign multiple virtual stages per GPU. This increases the effective number of pipeline stages without requiring additional hardware.

```
2 GPUs, 4 virtual stages:
GPU 0: Virtual Stage 0, Virtual Stage 1
GPU 1: Virtual Stage 2, Virtual Stage 3

Pipeline: VS0 -> VS1 -> VS2 -> VS3
```

This reduces pipeline bubbles by increasing the granularity of micro-batch scheduling.

---

## Checkpointing Within Pipeline

### Checkpoint Strategies

The `checkpoint` parameter in Pipe controls activation checkpointing (also called gradient checkpointing or activation recomputation).

```python
# Never checkpoint: Store all activations (most memory, fastest)
pipe_model = Pipe(model, balance=balance, chunks=8, checkpoint='never')

# Checkpoint everything: Recompute activations during backward
pipe_model = Pipe(model, balance=balance, chunks=8, checkpoint='always')

# Checkpoint except last micro-batch (default, good balance)
pipe_model = Pipe(model, balance=balance, chunks=8, checkpoint='except_last')
```

### How Checkpointing Works

```
Without checkpointing:
  Forward:  Compute activation, store in memory
  Backward: Use stored activation for gradient computation

With checkpointing:
  Forward:  Compute activation, store only input
  Backward: Recompute activation from input, then compute gradient
```

### Memory vs Compute Trade-off

```python
# Memory usage comparison (assuming 4 chunks):
# checkpoint='never':     4x activation memory
# checkpoint='except_last': 3x activation memory (recomputes 3, stores 1)
# checkpoint='always':    1x activation memory (recomputes all 4)
```

---

## Forward and Backward Mechanics

### Forward Pass

```
1. Input batch is split into 'chunks' micro-batches
2. Each micro-batch flows through all stages sequentially
3. At each stage boundary, the activation tensor is transferred to the next device
4. Outputs from all micro-batches are collected and concatenated

Stage 0 (cuda:0)  ->  Stage 1 (cuda:1)  ->  Stage 2 (cuda:2)
   MB0                   MB0                   MB0
   MB1                   MB1                   MB1
   MB2                   MB2                   MB2
   MB3                   MB3                   MB3
```

### Backward Pass

```
1. Loss is computed on the full output
2. Loss is backpropagated through the concatenated output
3. Each micro-batch's gradient flows backward through stages
4. At each stage boundary, gradients are transferred back

Stage 2 (cuda:2)  ->  Stage 1 (cuda:1)  ->  Stage 0 (cuda:0)
   MB3                   MB3                   MB3
   MB2                   MB2                   MB2
   MB1                   MB1                   MB1
   MB0                   MB0                   MB0
```

### Activation Handling

During the forward pass, activations at stage boundaries must be stored for the backward pass. The memory required depends on the checkpoint strategy and number of chunks.

```python
# Approximate activation memory per stage:
# activation_memory = chunks * micro_batch_size * boundary_activation_size

# With checkpoint='except_last':
# activation_memory = 1 * micro_batch_size * boundary_activation_size
# (only the last micro-batch's activations are stored)
```

---

## Complete Example

### Splitting a Transformer Model Across GPUs

```python
import torch
import torch.nn as nn
from torch.distributed.pipeline.sync import Pipe

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class PipelineTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Create transformer blocks
        blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]

        # Create a sequential model for pipeline
        layers = [self.embedding]
        for block in blocks:
            layers.append(block)
        layers.append(nn.Linear(d_model, n_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Create the model
model = PipelineTransformer(
    vocab_size=30000,
    d_model=768,
    n_heads=12,
    d_ff=3072,
    n_layers=12,
    n_classes=1000
)

# Split into 3 pipeline stages
# Embedding + blocks 0-3 | blocks 4-7 | blocks 8-11 + head
balance = [5, 4, 4]  # 5 layers, 4 layers, 4 layers

# Create pipeline model
pipe_model = Pipe(
    model.model,  # Must be nn.Sequential
    balance=balance,
    chunks=8,     # 8 micro-batches
    checkpoint='except_last'
)

# Training
optimizer = torch.optim.AdamW(pipe_model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for input_ids, labels in dataloader:
        optimizer.zero_grad()
        output = pipe_model(input_ids)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
```

### Memory Estimation for Pipeline Parallelism

```python
def estimate_pipeline_memory(model, balance, chunks, batch_size, seq_len, d_model):
    """Estimate memory usage per stage for pipeline parallelism."""
    num_stages = len(balance)

    for stage_id in range(num_stages):
        # Parameter memory
        layers_in_stage = balance[stage_id]
        # Approximate: each transformer layer has ~4 * d_model^2 parameters
        param_count = layers_in_stage * 4 * d_model ** 2
        param_memory = param_count * 4  # FP32

        # Activation memory (per micro-batch)
        micro_batch_size = batch_size // chunks
        activation_size = micro_batch_size * seq_len * d_model

        # Total activation memory depends on checkpoint strategy
        # With 'except_last': store activations for 1 micro-batch
        activation_memory = activation_size * 2 * layers_in_stage  # forward + backward

        total_memory = param_memory + activation_memory
        print(f"Stage {stage_id}: "
              f"Params={param_memory/1e9:.2f} GB, "
              f"Activations={activation_memory/1e9:.2f} GB, "
              f"Total={total_memory/1e9:.2f} GB")

# Example
estimate_pipeline_memory(model, balance=[5, 4, 4], chunks=8,
                          batch_size=64, seq_len=512, d_model=768)
```

### Best Practices

1. **Balance compute, not layers**: Use profiling to ensure each stage has similar FLOPs.
2. **Minimize boundary activation size**: Place boundaries where activation tensors are smallest.
3. **Tune chunks**: Start with chunks=8 and adjust based on memory and throughput.
4. **Use checkpoint='except_last'**: Good default balance of memory and compute.
5. **Combine with DDP**: For models that fit in pipeline-parallel form factor, add DDP for further scaling.
6. **Profile pipeline bubble**: Measure the fraction of idle time and adjust chunks accordingly.
