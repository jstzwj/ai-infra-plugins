# Pipeline Parallelism

## Overview

DeepSpeed provides a comprehensive pipeline parallelism implementation that partitions a model across multiple GPUs, with each GPU hosting a contiguous subset of model layers. Pipeline parallelism complements data parallelism and ZeRO by adding a model-parallel dimension that scales across the depth of the network, enabling training of extremely deep models.

The pipeline parallelism system consists of three main components:
- **PipelineModule**: Model definition and layer partitioning (`deepspeed/pipe/module.py`)
- **Topology and Grid**: Process group management (`deepspeed/pipe/topology.py`)
- **PipelineEngine**: Runtime execution with scheduling (`deepspeed/pipe/engine.py`)

### Architecture Overview

```
Model Layers:   [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11]

With 4 pipeline stages (P=4):
  Stage 0 (GPU 0):  L0, L1, L2
  Stage 1 (GPU 1):  L3, L4, L5
  Stage 2 (GPU 2):  L6, L7, L8
  Stage 3 (GPU 3):  L9, L10, L11

Forward pass:  Stage 0 -> Stage 1 -> Stage 2 -> Stage 3
Backward pass: Stage 3 -> Stage 2 -> Stage 1 -> Stage 0

Combined with data parallelism (D=2, P=4, total=8 GPUs):
  DP Group 0: GPU 0-3 (pipeline stages 0-3)
  DP Group 1: GPU 4-7 (replica of pipeline stages 0-3)

Inter-stage: point-to-point (p2p) communication between adjacent stages
Intra-stage: all-reduce across data-parallel replicas
```

## PipelineModule

`PipelineModule` in `deepspeed/pipe/module.py` is the core class for defining pipeline-parallel models. It replaces `torch.nn.Sequential` and provides automatic layer partitioning across pipeline stages.

### Class Definition

```python
# deepspeed/pipe/module.py
class PipelineModule(nn.Module):
    """Module wrapper for pipeline-parallel models.
    
    Encapsulates a sequence of layers that are partitioned across
    pipeline stages. Handles:
    - Layer specification and building
    - Automatic partitioning across stages
    - Activation checkpointing
    - Tied weight management
    - Inter-stage communication setup
    """
    
    def __init__(self,
                 layers,            # List[LayerSpec] or nn.Sequential
                 num_stages=None,   # Number of pipeline stages
                 topology=None,     # ProcessTopology
                 loss_fn=None,      # Loss function
                 seed_layers=False, # Random seed per layer
                 seed_fn=None,      # Custom seed function
                 base_seed=1234,    # Base random seed
                 partition_method='uniform',  # Partitioning strategy
                 activation_checkpoint_interval=0,  # Activation checkpointing
                 activation_checkpoint_func=CheckpointFunction,  # Checkpoint impl
                 checkpointable_layers=None):
        super().__init__()
        ...
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layers` | `List[LayerSpec]` or `nn.Sequential` | required | Sequence of layer specifications defining the model architecture |
| `num_stages` | int | None | Number of pipeline stages. If None, determined from topology |
| `topology` | ProcessTopology | None | Process topology for distributed mapping. If None, uses all available GPUs as pipeline stages |
| `loss_fn` | callable | None | Loss function applied at the final stage. Signature: `loss_fn(outputs, labels) -> scalar` |
| `seed_layers` | bool | False | Set a different random seed for each layer. Useful for reproducibility with dropout |
| `seed_fn` | callable | None | Custom function to generate per-layer seeds. Default uses `base_seed + layer_index` |
| `base_seed` | int | 1234 | Base random seed for layer seeding |
| `partition_method` | str | `'uniform'` | Method for partitioning layers across stages: `'uniform'` (equal layer count) or `'parameters'` (equal parameter count) |
| `activation_checkpoint_interval` | int | 0 | Number of layers between activation checkpoints. 0 = no checkpointing |
| `activation_checkpoint_func` | callable | CheckpointFunction | Activation checkpointing implementation |
| `checkpointable_layers` | list | None | List of layer types that support activation checkpointing |

### Creating a PipelineModule from LayerSpec

```python
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec

model = PipelineModule(
    layers=[
        # Embedding layer
        LayerSpec(nn.Embedding, num_embeddings=50000, embedding_dim=1024),
        
        # Transformer layers
        LayerSpec(MyTransformerLayer, hidden_size=1024, num_heads=16),
        LayerSpec(MyTransformerLayer, hidden_size=1024, num_heads=16),
        LayerSpec(MyTransformerLayer, hidden_size=1024, num_heads=16),
        LayerSpec(MyTransformerLayer, hidden_size=1024, num_heads=16),
        
        # Output head
        LayerSpec(nn.Linear, in_features=1024, out_features=50000),
    ],
    num_stages=2,
    loss_fn=cross_entropy_loss,
    partition_method='parameters',
    activation_checkpoint_interval=1
)
```

### Partition Methods

#### Uniform Partitioning

```python
# partition_method='uniform'
# Equal number of layers per stage

# 12 layers, 4 stages:
#   Stage 0: layers 0-2  (3 layers)
#   Stage 1: layers 3-5  (3 layers)
#   Stage 2: layers 6-8  (3 layers)
#   Stage 3: layers 9-11 (3 layers)
```

#### Parameter-Balanced Partitioning

```python
# partition_method='parameters'
# Equal parameter count per stage

# If embedding has 50M params and each transformer layer has 20M:
#   Stage 0: Embedding + Layer1 + Layer2  (90M params)
#   Stage 1: Layer3 + Layer4 + Layer5      (60M params)
#   Stage 2: Layer6 + Layer7 + Layer8      (60M params)
#   Stage 3: Layer9 + Layer10 + Linear     (90M params)
# 
# More balanced than uniform when layers have different sizes
```

### num_stages and topology

```python
# Option 1: Specify num_stages (simple pipeline only, no data parallelism)
model = PipelineModule(
    layers=my_layers,
    num_stages=4,  # 4 pipeline stages, 4 GPUs total
)

# Option 2: Specify topology (pipeline + data parallelism)
from deepspeed.pipe import ProcessTopology

topo = ProcessTopology(
    dims=['pipe', 'data'],
    shape=[4, 2]  # 4 pipeline stages, 2 data-parallel replicas = 8 GPUs
)

model = PipelineModule(
    layers=my_layers,
    topology=topo,
)
```

### Activation Checkpointing

```python
model = PipelineModule(
    layers=my_layers,
    num_stages=4,
    activation_checkpoint_interval=2,  # Checkpoint every 2 layers
)

# Memory behavior:
#   interval=0: No checkpointing, all activations stored (fastest, most memory)
#   interval=1: All layers checkpointed (slowest, least memory)
#   interval=2: Every other layer checkpointed (balanced)
#   interval=N: Every N-th boundary is checkpointed
```

Activation checkpointing trades computation for memory:
- **Without checkpointing**: All intermediate activations are stored for backward pass
- **With checkpointing**: Only layer boundaries are stored; intermediate activations are recomputed during backward pass
- **Memory savings**: ~50% with `interval=1`, ~25% with `interval=2`

## LayerSpec

`LayerSpec` defines a single layer in the pipeline module. It is a declarative specification that defers layer instantiation until the partitioning phase.

```python
# deepspeed/pipe/module.py
class LayerSpec:
    """Declarative specification for a layer in a pipeline module.
    
    Stores the layer class and initialization arguments. The actual
    layer is instantiated later during module setup, after determining
    which pipeline stage owns this layer.
    
    Args:
        typename: Layer class (e.g., nn.Linear, MyTransformerLayer)
        *args: Positional arguments for the layer constructor
        **kwargs: Keyword arguments for the layer constructor
    """
    
    def __init__(self, typename, *args, **kwargs):
        self.typename = typename
        self.args = list(args)
        self.kwargs = kwargs
    
    def build(self, log=False):
        """Instantiate the layer.
        
        Returns:
            Instance of typename(*args, **kwargs)
        """
        if log:
            logger.info(f'Building {self.typename.__name__} layer')
        return self.typename(*self.args, **self.kwargs)
```

### Usage Examples

```python
from deepspeed.pipe import LayerSpec
import torch.nn as nn

# Standard PyTorch modules
spec1 = LayerSpec(nn.Linear, in_features=1024, out_features=4096)
spec2 = LayerSpec(nn.LayerNorm, normalized_shape=1024)
spec3 = LayerSpec(nn.Dropout, p=0.1)

# Custom modules
spec4 = LayerSpec(MyTransformerBlock,
                  hidden_size=1024,
                  num_attention_heads=16,
                  intermediate_size=4096,
                  attention_dropout=0.1)

# Build a layer instance
layer = spec1.build()
# layer is now nn.Linear(1024, 4096)
```

### LayerSpec as a Building Block

LayerSpec enables several powerful patterns:

```python
# Pattern 1: Dynamic layer construction
def make_transformer_layers(num_layers, hidden_size, num_heads):
    return [
        LayerSpec(TransformerLayer, hidden_size=hidden_size, num_heads=num_heads)
        for _ in range(num_layers)
    ]

model = PipelineModule(
    layers=[
        LayerSpec(nn.Embedding, 50000, hidden_size),
        *make_transformer_layers(24, hidden_size, 16),
        LayerSpec(nn.Linear, hidden_size, 50000),
    ],
    num_stages=4,
)

# Pattern 2: Conditional architecture
if use_pre_norm:
    layers.append(LayerSpec(nn.LayerNorm, hidden_size))
layers.append(LayerSpec(TransformerLayer, hidden_size, num_heads))
```

## TiedLayerSpec

`TiedLayerSpec` enables weight sharing across pipeline stages. Layers specified with `TiedLayerSpec` share the same parameter tensor even when placed on different pipeline stages.

```python
# deepspeed/pipe/module.py
class TiedLayerSpec(LayerSpec):
    """Layer specification with weight tying across pipeline stages.
    
    Used for layers that should share parameters across stages.
    Common use cases:
    - Shared embedding and output projection in language models
    - Shared attention patterns across transformer layers
    - Tied autoencoder weights
    
    Args:
        name: Unique identifier for the tied weight group
        typename: Layer class
        *args: Positional arguments for the layer constructor
        **kwargs: Keyword arguments for the layer constructor
    """
    
    def __init__(self, name, typename, *args, **kwargs):
        super().__init__(typename, *args, **kwargs)
        self.name = name  # Tied weight group identifier
```

### Usage: Shared Embedding and Output Projection

```python
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec

model = PipelineModule(
    layers=[
        # Shared embedding (first layer)
        TiedLayerSpec('shared_embed', nn.Embedding, 
                      num_embeddings=50000, embedding_dim=1024),
        
        # Transformer layers
        *[LayerSpec(TransformerLayer, hidden_size=1024, num_heads=16)
          for _ in range(12)],
        
        # Output projection (last layer) shares weights with embedding
        TiedLayerSpec('shared_embed', nn.Linear,
                      in_features=1024, out_features=50000, bias=False),
    ],
    num_stages=4,
    loss_fn=cross_entropy_loss,
)
```

### How Tied Weights Work

```
Stage 0: Embedding layer (TiedLayerSpec 'shared_embed')
         |  Embedding weight: [50000, 1024]
         |  weight_id = id(shared_embed_weight)
         
Stage 3: Output projection (TiedLayerSpec 'shared_embed')
         |  Linear weight: [50000, 1024] (SAME tensor as embedding)
         |  weight_id = id(shared_embed_weight) (same ID)

During forward/backward:
  - Both layers use the same parameter tensor
  - Gradients from both layers are accumulated on the same parameter
  - Only one copy of the weight is stored (saves memory)
  
Synchronization:
  - Before Stage 0 forward: weight is up-to-date (updated by both stages)
  - Before Stage 3 forward: weight is synchronized from Stage 0
```

## PipelineEngine

`PipelineEngine` in `deepspeed/pipe/engine.py` manages the runtime execution of pipeline-parallel training, including forward/backward scheduling, inter-stage communication, and micro-batch management.

### Class Definition

```python
# deepspeed/pipe/engine.py
class PipelineEngine(DeepSpeedEngine):
    """Pipeline-parallel training engine.
    
    Extends DeepSpeedEngine with pipeline-specific execution:
    - Micro-batch scheduling across pipeline stages
    - Point-to-point communication between adjacent stages
    - Gradient accumulation across micro-batches
    - Pipeline schedule execution (GPipe, 1F1B, interleaved)
    """
    
    def __init__(self, *super_args, **kwargs):
        super().__init__(*super_args, **kwargs)
        
        # Pipeline-specific state
        self.pipeline_model = self.module  # PipelineModule
        self.grid = self.pipeline_model.grid  # PipelineParallelGrid
        self.num_stages = self.grid.pipe_parallel_size
        self.stage_id = self.grid.get_pipe_parallel_rank()
        
        # Micro-batch management
        self.num_microbatches = None  # Set per training step
        self.microbatch_id = 0
        
        # Inter-stage communication buffers
        self._setup_stage_buffers()
```

### Forward/Backward with Pipeline Scheduling

```python
def forward(self, *args, **kwargs):
    """Execute forward pass through the pipeline stage.
    
    For non-first stages: receives input from previous stage via p2p
    For first stage: uses provided input data
    """
    if self.is_first_stage():
        # Use input data directly
        input_tensor = args[0]
    else:
        # Receive from previous stage
        input_tensor = self._recv_from_prev_stage()
    
    # Forward through local layers
    output_tensor = self.module.forward(input_tensor)
    
    if not self.is_last_stage():
        # Send to next stage
        self._send_to_next_stage(output_tensor)
    
    return output_tensor

def backward(self, loss):
    """Execute backward pass through the pipeline stage.
    
    For non-last stages: receives gradient from next stage via p2p
    For last stage: uses gradient of loss
    """
    if self.is_last_stage():
        # Compute gradient from loss
        loss.backward()
    else:
        # Receive gradient from next stage
        output_grad = self._recv_grad_from_next_stage()
        
        # Backward through local layers
        self.module.backward(output_grad)
    
    if not self.is_first_stage():
        # Send gradient to previous stage
        self._send_grad_to_prev_stage()
```

### Pipeline Schedule Execution

The engine executes micro-batches according to a schedule that determines when each stage processes each micro-batch:

```python
def _exec_schedule(self, pipeline_schedule):
    """Execute the given pipeline schedule.
    
    Args:
        pipeline_schedule: List of instruction tuples for each clock cycle.
            Each instruction is (microbatch_id, stage_action) where
            stage_action is 'forward' or 'backward'.
    """
    for clock_cycle in pipeline_schedule:
        for microbatch_id, action in clock_cycle:
            if action == 'forward':
                self._exec_forward(microbatch_id)
            elif action == 'backward':
                self._exec_backward(microbatch_id)
            elif action == 'idle':
                pass  # Bubble in the pipeline
```

## ProcessTopology

`ProcessTopology` in `deepspeed/pipe/topology.py` manages the mapping of global process ranks to a multi-dimensional Cartesian coordinate grid. It provides the foundation for organizing processes into pipeline-parallel and data-parallel groups.

### Class Definition

```python
# deepspeed/pipe/topology.py
class ProcessTopology:
    """Manages mapping of process ranks to a multi-dimensional Cartesian grid.
    
    Supports arbitrary dimensional grids with named dimensions.
    Common dimensions:
    - 'pipe': Pipeline parallel dimension
    - 'data': Data parallel dimension
    - 'model': Model parallel dimension (tensor parallel)
    
    Args:
        dims: List of dimension names (e.g., ['pipe', 'data'])
        shape: Shape of the grid (e.g., [4, 2] = 4 pipeline x 2 data)
    """
    
    def __init__(self, dims, shape):
        self.dims = dims
        self.shape = shape
        self.world_size = 1
        for s in shape:
            self.world_size *= s
        
        assert dist.get_world_size() == self.world_size, \
            f"Grid world size {self.world_size} != dist world size {dist.get_world_size()}"
        
        self.rank = dist.get_rank()
        
        # Build coordinate mapping
        self._build_mapping()
```

### Coordinate Mapping

```python
# Example: dims=['pipe', 'data'], shape=[4, 2]
# 8 GPUs organized as 4 pipeline stages x 2 data-parallel replicas
#
# Grid layout:
#           data=0    data=1
# pipe=0    rank 0    rank 4
# pipe=1    rank 1    rank 5
# pipe=2    rank 2    rank 6
# pipe=3    rank 3    rank 7
#
# Pipeline group 0: ranks [0, 1, 2, 3]
# Pipeline group 1: ranks [4, 5, 6, 7]
# Data-parallel group at stage 0: ranks [0, 4]
# Data-parallel group at stage 1: ranks [1, 5]
# ...

topo = ProcessTopology(dims=['pipe', 'data'], shape=[4, 2])

# Query functions
topo.get_rank(coordinate={'pipe': 2, 'data': 0})  # Returns 2
topo.get_rank(coordinate={'pipe': 2, 'data': 1})  # Returns 6
topo.get_coord(rank=5)  # Returns {'pipe': 1, 'data': 1}
topo.get_pipe_parallel_rank(rank=3)  # Returns 3
topo.get_data_parallel_rank(rank=5)  # Returns 1
```

### Key Methods

| Method | Description |
|--------|-------------|
| `get_rank(coordinate)` | Get global rank from dimension coordinates |
| `get_coord(rank)` | Get dimension coordinates from global rank |
| `get_dim_rank(dim, rank)` | Get rank along a specific dimension |
| `get_group(dim)` | Get all groups along a dimension |
| `filter_match(**kwargs)` | Get ranks matching specific coordinate values |
| `get_pipe_parallel_rank()` | Get this process's pipeline rank |
| `get_data_parallel_rank()` | Get this process's data-parallel rank |

### PipeDataParallelTopology

A convenience class for the common case of pipeline + data parallelism:

```python
class PipeDataParallelTopology(ProcessTopology):
    """Convenience topology for pipeline + data parallelism.
    
    Args:
        num_pp: Number of pipeline stages
        num_dp: Number of data-parallel replicas
    """
    
    def __init__(self, num_pp, num_dp):
        super().__init__(dims=['pipe', 'data'], shape=[num_pp, num_dp])
```

## PipelineParallelGrid

`PipelineParallelGrid` extends the topology to provide a 2D process grid specifically designed for pipeline parallelism with data parallelism:

```python
# deepspeed/pipe/topology.py
class PipelineParallelGrid:
    """2D grid for pipeline parallelism.
    
    Manages both pipeline-parallel and data-parallel process groups.
    Provides convenient access to pipeline stage information and
    inter-stage communication groups.
    """
    
    def __init__(self, topology=None):
        if topology is None:
            # Default: all GPUs form a single pipeline
            self.pipe_parallel_size = dist.get_world_size()
            self.data_parallel_size = 1
        else:
            self.topology = topology
            self.pipe_parallel_size = topology.get_dim_size('pipe')
            self.data_parallel_size = topology.get_dim_size('data')
        
        self.stage_id = self._get_stage_id()
        
        # Create process groups
        self._build_process_groups()
    
    def _build_process_groups(self):
        """Create torch.distributed process groups for pipeline and data parallelism."""
        # Pipeline groups: processes with same data-parallel rank
        for dp_rank in range(self.data_parallel_size):
            pipe_ranks = self.topology.filter_match(data=dp_rank)
            pipe_group = dist.new_group(ranks=pipe_ranks)
            if dp_rank == self.dp_rank:
                self.pipe_group = pipe_group
        
        # Data-parallel groups: processes with same pipeline stage
        for pp_rank in range(self.pipe_parallel_size):
            dp_ranks = self.topology.filter_match(pipe=pp_rank)
            dp_group = dist.new_group(ranks=dp_ranks)
            if pp_rank == self.stage_id:
                self.dp_group = dp_group
    
    def is_first_stage(self):
        return self.stage_id == 0
    
    def is_last_stage(self):
        return self.stage_id == self.pipe_parallel_size - 1
    
    def prev_stage(self):
        return self.stage_id - 1
    
    def next_stage(self):
        return self.stage_id + 1
```

## Inter-Stage Communication (P2P)

Pipeline stages communicate through point-to-point (p2p) operations between adjacent stages. DeepSpeed implements p2p communication in `deepspeed/runtime/pipe/`:

### Communication Pattern

```python
# deepspeed/runtime/pipe/p2p.py (conceptual)

def recv_from_prev_stage(tensor, pipe_group, stage_id):
    """Receive tensor from the previous pipeline stage.
    
    For stage 0: returns None (uses input data)
    For stage > 0: blocking receive from stage_id - 1
    """
    if stage_id == 0:
        return None
    tensor = torch.empty_like(tensor)
    dist.recv(tensor=tensor, src=stage_id - 1, group=pipe_group)
    return tensor

def send_to_next_stage(tensor, pipe_group, stage_id, num_stages):
    """Send tensor to the next pipeline stage.
    
    For last stage: no-op (output goes to loss computation)
    For other stages: blocking send to stage_id + 1
    """
    if stage_id == num_stages - 1:
        return
    dist.send(tensor=tensor, dst=stage_id + 1, group=pipe_group)
```

### Asynchronous P2P Communication

For better overlap, DeepSpeed supports asynchronous p2p:

```python
def async_recv_from_prev_stage(tensor, pipe_group, stage_id):
    """Asynchronously receive tensor from previous stage."""
    if stage_id == 0:
        return None, None
    tensor = torch.empty_like(tensor)
    handle = dist.irecv(tensor=tensor, src=stage_id - 1, group=pipe_group)
    return tensor, handle

def async_send_to_next_stage(tensor, pipe_group, stage_id, num_stages):
    """Asynchronously send tensor to next stage."""
    if stage_id == num_stages - 1:
        return None
    handle = dist.isend(tensor=tensor, dst=stage_id + 1, group=pipe_group)
    return handle
```

### Tensor Metadata Communication

DeepSpeed communicates tensor shape and dtype metadata alongside data:

```python
def send_tensor_metadata(tensor, dst, group):
    """Send shape and dtype metadata before sending tensor data."""
    metadata = torch.tensor([
        len(tensor.shape),
        *tensor.shape,
        # dtype encoded as int
    ], dtype=torch.long)
    dist.send(metadata, dst=dst, group=group)

def recv_tensor_metadata(src, group):
    """Receive tensor metadata."""
    metadata = torch.zeros(10, dtype=torch.long)  # Max 8 dims
    dist.recv(metadata, src=src, group=group)
    ndim = metadata[0].item()
    shape = tuple(metadata[1:ndim+1].tolist())
    return shape
```

## Pipeline Scheduling Strategies

### GPipe Schedule

The simplest schedule: forward all micro-batches, then backward all micro-batches.

```
Stage:   0         1         2         3
Time:  |----|----|----|----|----|----|----|----|
       | F0 | F1 | F2 | F3 |    |    |    |    |  Stage 0
       |    | F0 | F1 | F2 | F3 |    |    |    |  Stage 1
       |    |    | F0 | F1 | F2 | F3 |    |    |  Stage 2
       |    |    |    | F0 | F1 | F2 | F3 |    |  Stage 3
       |    |    |    |    |    | B3 | B2 | B1 | B0 |  Stage 3
       |    |    |    |    | B3 | B2 | B1 | B0 |    |  Stage 2
       |    |    |    | B3 | B2 | B1 | B0 |    |    |  Stage 1
       |    |    | B3 | B2 | B1 | B0 |    |    |    |  Stage 0

F = Forward, B = Backward, subscript = microbatch ID
Notable: Large pipeline bubble between last forward and first backward
```

**Characteristics**:
- Simple to implement
- High memory usage: all micro-batch activations stored simultaneously
- Large pipeline bubble: $\frac{P-1}{M}$ fraction of time is idle (P = stages, M = microbatches)

### 1F1B Schedule (One Forward One Backward)

The interleaved schedule that alternates forward and backward passes to reduce memory and bubble:

```
Phase 1 (Warmup): Forward-only to fill the pipeline
Phase 2 (Steady state): One forward + one backward per step
Phase 3 (Cooldown): Backward-only to drain the pipeline

Stage 0: F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
Stage 1:    F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 B6 B7
Stage 2:       F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 B7
Stage 3:          F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7

Memory: Only (P - stage_id) activations in flight at once
```

**Characteristics**:
- Reduced peak memory: only `P - stage_id` micro-batch activations stored
- Smaller bubble: warmup + cooldown overhead
- Most commonly used schedule in DeepSpeed

### Interleaved 1F1B Schedule

An optimized 1F1B schedule where each device hosts multiple non-contiguous stage chunks, reducing the pipeline bubble further:

```
With 4 stages and 2 chunks per device (V=2):
Stage assignment:
  Device 0: chunks [0, 4]
  Device 1: chunks [1, 5]
  Device 2: chunks [2, 6]
  Device 3: chunks [3, 7]

Pipeline bubble reduced by factor of V (number of chunks per device)
Communication volume increases by V (more inter-device transfers)
```

**Characteristics**:
- Smallest pipeline bubble: $\frac{P-1}{M \cdot V}$ (V = chunks per device)
- Increased communication: V times more p2p transfers
- Best for communication-heavy models with many micro-batches

### Schedule Comparison

| Schedule | Bubble Fraction | Peak Memory | Communication | Complexity |
|----------|----------------|-------------|---------------|------------|
| GPipe | $\frac{P-1}{M}$ | $M$ activations | $2(P-1)$ transfers | Simple |
| 1F1B | $\frac{P-1}{M}$ | $P$ activations | $2(P-1)$ transfers | Moderate |
| Interleaved 1F1B | $\frac{P-1}{M \cdot V}$ | $P$ activations | $2V(P-1)$ transfers | Complex |

Where P = pipeline stages, M = micro-batches, V = chunks per device.

## Configuration Example

### Basic Pipeline Parallelism

```json
{
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0
    },
    "pipeline": {
        "enabled": true,
        "parallel_size": 4,
        "micro_batches": 4,
        "activation_checkpoint_interval": 1
    },
    "zero_optimization": {
        "stage": 0
    }
}
```

### Pipeline with Data Parallelism

```json
{
    "train_batch_size": 128,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
    },
    "bf16": {
        "enabled": true
    },
    "pipeline": {
        "enabled": true,
        "parallel_size": 4
    },
    "zero_optimization": {
        "stage": 0
    }
}
```

With `parallel_size=4` and 16 total GPUs, there are 4 data-parallel replicas of the pipeline.

### Python API Usage

```python
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
import torch.nn as nn

def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

# Define model as layer specs
layers = [
    LayerSpec(nn.Embedding, num_embeddings=50000, embedding_dim=1024),
    LayerSpec(nn.Dropout, p=0.1),
]
for _ in range(24):
    layers.append(LayerSpec(TransformerLayer, hidden_size=1024, num_heads=16))
layers.append(nn.LayerNorm(1024))
layers.append(LayerSpec(nn.Linear, 1024, 50000, bias=False))

# Create pipeline module
net = PipelineModule(
    layers=layers,
    num_stages=4,
    loss_fn=loss_fn,
    partition_method='parameters',
    activation_checkpoint_interval=2,
)

# Initialize DeepSpeed
ds_engine = deepspeed.initialize(
    model=net,
    optimizer=optimizer,
    config=ds_config,
)[0]

# Training loop
for batch in dataloader:
    inputs, labels = batch
    loss = ds_engine.train_batch()  # Handles pipeline scheduling internally
```

## Integration with ZeRO

Pipeline parallelism can be combined with ZeRO for additional memory savings:

### Pipeline + ZeRO Stage 1

```json
{
    "pipeline": {
        "enabled": true,
        "parallel_size": 4
    },
    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": 5e8
    }
}
```

**Effect**: Each pipeline stage uses ZeRO Stage 1 within its data-parallel group. Optimizer states are partitioned across data-parallel replicas of each stage.

### Pipeline + ZeRO Stage 2

```json
{
    "pipeline": {
        "enabled": true,
        "parallel_size": 4
    },
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": true,
        "overlap_comm": true
    }
}
```

**Effect**: Each pipeline stage uses ZeRO Stage 2. Gradients and optimizer states are partitioned across data-parallel replicas.

### Pipeline + ZeRO Stage 3 (Not Recommended)

```json
{
    "pipeline": {
        "enabled": true,
        "parallel_size": 4
    },
    "zero_optimization": {
        "stage": 3
    }
}
```

**Warning**: ZeRO Stage 3 partitions parameters, which conflicts with pipeline parallelism's requirement that each stage has its parameters readily available for forward/backward. This combination is generally not recommended and may lead to issues with parameter gathering during pipeline execution.

### Recommended Combinations

| Combination | Memory Savings | Throughput | Recommendation |
|------------|---------------|-----------|----------------|
| Pipeline only | Moderate | High | Best for throughput-critical training |
| Pipeline + ZeRO-1 | Good | High | Recommended for most use cases |
| Pipeline + ZeRO-2 | Very good | Medium-High | When GPU memory is limited |
| Pipeline + ZeRO-3 | Not recommended | Low | Conflicts between parameter partitioning and pipeline scheduling |

### 3D Parallelism (Pipeline + Data + ZeRO)

The most common large-scale training configuration combines pipeline parallelism, data parallelism, and ZeRO:

```
Total GPUs = 64
Pipeline parallel size (P) = 4
Data parallel size (D) = 16
ZeRO Stage = 1

Layout:
  4 pipeline stages, each replicated 16 times
  Within each stage, ZeRO-1 partitions optimizer states across 16 GPUs

Per-GPU memory:
  Parameters: full stage parameters (1/P of model)
  Gradients: full stage gradients (1/P of model)
  Optimizer states: 1/D of stage optimizer states (1/(P*D) of model)

Example with 175B model, P=4, D=16:
  Per-stage params: 175B/4 = 43.75B
  Per-GPU optimizer states: 12 * 43.75B / 16 = 32.8 GB
  Per-GPU total: ~2*43.75B + 32.8GB ≈ 120 GB (with FP16 + FP32 optimizer)
```

## Key Source Files

| File | Description |
|------|-------------|
| `deepspeed/pipe/module.py` | PipelineModule, LayerSpec, TiedLayerSpec |
| `deepspeed/pipe/topology.py` | ProcessTopology, PipeDataParallelTopology, PipelineParallelGrid |
| `deepspeed/pipe/engine.py` | PipelineEngine, pipeline scheduling, forward/backward execution |
| `deepspeed/runtime/pipe/schedule.py` | Pipeline schedule definitions (GPipe, 1F1B, interleaved) |
| `deepspeed/runtime/pipe/p2p.py` | Point-to-point communication between pipeline stages |
| `deepspeed/runtime/pipe/amp.py` | Mixed precision support for pipeline stages |
