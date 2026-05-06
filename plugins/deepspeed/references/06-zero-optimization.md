# ZeRO (Zero Redundancy Optimizer)

## Overview

ZeRO (Zero Redundancy Optimizer) is DeepSpeed's flagship memory optimization technology that eliminates memory redundancies across data-parallel processes by partitioning the three components of model training state: optimizer states, gradients, and parameters. Developed by Microsoft Research, ZeRO progressively reduces per-GPU memory consumption through three stages of optimization, enabling the training of models with hundreds of billions to trillions of parameters.

### The Memory Problem in Data Parallelism

In standard Distributed Data Parallel (DDP), every GPU holds a complete replica of:
- **Optimizer states**: For Adam, this includes momentum (first moment) and variance (second moment) tensors, each the size of the model parameters
- **Gradients**: One gradient tensor per parameter
- **Parameters**: The model weights themselves

For a model with $\Psi$ parameters (e.g., 7B parameters), using Adam with mixed precision (FP16 parameters + FP32 optimizer states):
- FP16 parameters: $2\Psi$ bytes
- FP16 gradients: $2\Psi$ bytes
- FP32 optimizer states (master weights + momentum + variance): $12\Psi$ bytes
- **Total per GPU**: $16\Psi$ bytes = 112 GB for a 7B parameter model

With 64 GPUs, total memory across all GPUs = $64 \times 16\Psi = 1024\Psi$ bytes, but only $16\Psi$ bytes are unique. ZeRO eliminates this redundancy.

## ZeroStageEnum

```python
# deepspeed/runtime/zero/stage_enum.py
class ZeroStageEnum(IntEnum):
    disabled = 0       # Standard DDP, no memory optimization
    optimizer_states = 1  # Stage 1: Partition optimizer states
    gradients = 2         # Stage 2: + Partition gradients
    weights = 3           # Stage 3: + Partition parameters (weights)
```

| Stage | Constant | Value | Partitioned Components | Memory Reduction (vs DDP) |
|-------|----------|-------|----------------------|---------------------------|
| 0 | `disabled` | 0 | None (standard DDP) | 1x (baseline) |
| 1 | `optimizer_states` | 1 | Optimizer states | 4x |
| 2 | `gradients` | 2 | Optimizer states + Gradients | 8x |
| 3 | `weights` | 3 | Optimizer states + Gradients + Parameters | $N_d$x (linear with GPU count) |

## Stage 0: Standard DDP

Stage 0 disables all ZeRO optimizations and falls back to standard PyTorch `DistributedDataParallel`. Every rank holds a full copy of all model states.

### Configuration

```json
{
    "zero_optimization": {
        "stage": 0
    }
}
```

### Behavior
- Full model replica on each GPU
- Standard all-reduce for gradient synchronization
- No memory savings relative to baseline DDP
- Useful as a baseline for benchmarking or when model fits comfortably in GPU memory

### When to Use Stage 0
- Model size is small relative to GPU memory
- Debugging distributed training issues
- Establishing baseline performance metrics
- Compatibility testing with other features

## Stage 1: Optimizer State Partitioning

Stage 1 partitions optimizer states across data-parallel ranks. Each rank stores only $\frac{1}{N_d}$ of the optimizer states (where $N_d$ is the data-parallel degree), while maintaining full copies of parameters and gradients.

### DeepSpeedZeroOptimizer Class

The core implementation resides in `deepspeed/runtime/zero/stage1and2.py`:

```python
class DeepSpeedZeroOptimizer(object):
    """DeepSpeedZeroOptimizer for Stage 1 and Stage 2.
    
    Acts as a wrapper around a user-provided optimizer to provide
    ZeRO memory optimization.
    """
    
    def __init__(self,
                 init_optimizer,
                 timers,
                 static_loss_scale,
                 dynamic_loss_args,
                 verbose,
                 contiguous_gradients,
                 reduce_bucket_size,
                 use_prefix_autocast,
                 allgather_bucket_size):
        ...
```

### How Stage 1 Works

1. **Parameter Grouping**: Parameters are grouped into flat buffers (one per parameter group) for efficient reduction operations.

2. **Optimizer State Partitioning**: After the first optimizer step, DeepSpeed partitions optimizer states:
   - Each rank $i$ owns optimizer states for parameters in partition $[i \cdot \text{partition_size}, (i+1) \cdot \text{partition_size})$
   - For Adam: momentum and variance tensors are partitioned
   - Master weights (FP32 copy) are partitioned

3. **Gradient Reduction**: After backward pass:
   - Standard all-reduce averages gradients across all ranks
   - Each rank updates its local partition of optimizer states

4. **Parameter Update**: After optimizer step:
   - Each rank updates its partition of FP32 master weights
   - Updated FP16 parameters are broadcast from the owning rank to all other ranks

### Memory Savings Calculation (Stage 1)

For a model with $\Psi$ parameters and $N_d$ data-parallel GPUs:

| Component | Per-GPU Memory (DDP) | Per-GPU Memory (Stage 1) |
|-----------|---------------------|--------------------------|
| FP16 Parameters | $2\Psi$ | $2\Psi$ |
| FP16 Gradients | $2\Psi$ | $2\Psi$ |
| FP32 Optimizer States | $12\Psi$ | $\frac{12\Psi}{N_d}$ |
| **Total** | $16\Psi$ | $4\Psi + \frac{12\Psi}{N_d}$ |

With $N_d = 64$: $4\Psi + \frac{12\Psi}{64} \approx 4.19\Psi$ bytes (3.8x savings).

For Adam specifically, the optimizer states are 3x the parameter count (master weights + momentum + variance), so partitioning them yields the 4x memory reduction.

### DeepSpeedZeroConfig Fields (Stage 1)

```json
{
    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
        "overlap_comm": false,
        "contiguous_gradients": true,
        "round_robin_gradients": false
    }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `stage` | int | - | Must be 1 |
| `reduce_bucket_size` | int | $5 \times 10^8$ | Number of elements reduced in one all-reduce operation |
| `allgather_bucket_size` | int | $5 \times 10^8$ | Number of elements gathered in one all-gather operation |
| `overlap_comm` | bool | false | Overlap gradient reduction with backward computation |
| `contiguous_gradients` | bool | true | Store gradients in contiguous memory buffers |
| `round_robin_gradients` | bool | false | Assign gradient buckets to ranks in round-robin fashion |

### Communication Pattern (Stage 1)

```
Backward Pass:
  For each bucket of gradients:
    1. all-reduce(gradients)       # Average across all ranks
    2. Local optimizer step on owned partition
    3. broadcast(updated_params)   # Each rank broadcasts its updated partition

Total communication volume: 2 * Psi * sizeof(fp16) * N_dp
```

## Stage 2: Gradient Partitioning

Stage 2 adds gradient partitioning on top of optimizer state partitioning. Instead of all-reducing full gradients, Stage 2 uses reduce-scatter so each rank only retains the gradient partition it needs.

### Key Mechanism

1. **Reduce-Scatter for Gradients**: After backward computation, gradients are reduce-scattered across ranks. Each rank receives only the averaged gradient for its parameter partition.

2. **Contiguous Gradient Buffers**: Gradients are stored in pre-allocated contiguous memory buffers, organized into reduction buckets for efficient communication.

3. **Bucket-Based Reduction**: Parameters are divided into buckets (controlled by `reduce_bucket_size`). Each bucket's gradients are reduce-scattered as a unit.

### DeepSpeedZeroOptimizer (Stage 2 Mode)

When `stage=2`, the same `DeepSpeedZeroOptimizer` class enables gradient partitioning:

```python
# In stage 2, after reduce-scatter, only the local gradient partition is kept
# Other gradient memory is freed immediately

def _reduce_scatter_gradients(self):
    """Reduce-scatter gradients across data-parallel ranks."""
    for bucket in self.gradient_buckets:
        # reduce-scatter: each rank gets 1/Nd of the averaged gradients
        handle = dist.reduce_scatter(
            tensor=bucket.output_buffer,
            scatter_list=bucket.gradient_shards,
            group=self.dp_process_group
        )
        # Free non-owned gradient memory immediately
        for i, shard in enumerate(bucket.gradient_shards):
            if i != self.dp_rank:
                shard.data = None  # Release memory
```

### Configuration

```json
{
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
        "use_multi_rank_bucket_allreduce": true,
        "round_robin_gradients": true,
        "zero_quantized_weights": false,
        "zero_quantized_gradients": false,
        "zero_hpz_partition_size": 0
    }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `contiguous_gradients` | bool | true | Allocate contiguous memory for gradient buffers to reduce memory fragmentation |
| `overlap_comm` | bool | false | Overlap gradient reduction with backward computation using separate streams |
| `reduce_bucket_size` | int | $5 \times 10^8$ | Size (in elements) of gradient reduction buckets. Smaller values increase communication overhead; larger values increase peak memory |
| `allgather_bucket_size` | int | $5 \times 10^8$ | Size (in elements) for all-gather operations during parameter gathering |
| `use_multi_rank_bucket_allreduce` | bool | false | Enable multi-rank bucket all-reduce for improved efficiency at scale |
| `round_robin_gradients` | bool | false | Distribute gradient buckets to ranks in round-robin to balance memory |
| `zero_quantized_weights` | bool | false | Quantize weights for reduced communication (ZeRO++) |
| `zero_quantized_gradients` | bool | false | Quantize gradients for reduced communication (ZeRO++) |
| `zero_hpz_partition_size` | int | 0 | Hierarchical partitioning group size (ZeRO++) |

### Memory Savings Calculation (Stage 2)

| Component | Per-GPU Memory (DDP) | Per-GPU Memory (Stage 2) |
|-----------|---------------------|--------------------------|
| FP16 Parameters | $2\Psi$ | $2\Psi$ |
| FP16 Gradients | $2\Psi$ | $\frac{2\Psi}{N_d}$ |
| FP32 Optimizer States | $12\Psi$ | $\frac{12\Psi}{N_d}$ |
| **Total** | $16\Psi$ | $2\Psi + \frac{14\Psi}{N_d}$ |

With $N_d = 64$: $2\Psi + \frac{14\Psi}{64} \approx 2.22\Psi$ bytes (7.2x savings).

### overlap_comm Detail

When `overlap_comm=true`, DeepSpeed uses a separate CUDA stream for communication operations, allowing gradient reduction to overlap with backward computation:

```python
# Conceptual flow with overlap_comm=True:
# Stream 1 (compute): backward pass computing gradients
# Stream 2 (comm):    reduce-scatter of completed gradient buckets

def backward_with_overlap(self, loss):
    # Register backward hooks on parameters
    for param in model.parameters():
        param.register_hook(lambda grad: self._async_reduce_bucket(grad))
    
    loss.backward()  # Gradient buckets are reduced as they become ready
```

**Performance Impact**:
- Reduces the effective communication time by hiding it behind computation
- Requires `reduce_bucket_size` tuning: smaller buckets enable more overlap but increase overhead
- Recommended for multi-node training where communication cost is significant

### reduce_bucket_size Tuning

The `reduce_bucket_size` controls how many parameter elements' gradients are grouped together for a single reduction operation.

| Scenario | Recommended Value | Rationale |
|----------|-------------------|-----------|
| Small model (< 1B) | $1 \times 10^8$ | Low communication volume, smaller buckets reduce peak memory |
| Medium model (1B-10B) | $5 \times 10^8$ (default) | Good balance of throughput and memory |
| Large model (10B+) | $5 \times 10^8 - 2 \times 10^9$ | Larger buckets amortize latency |
| Multi-node, overlap_comm | $2 \times 10^8 - 5 \times 10^8$ | Smaller buckets enable finer-grained overlap |
| Limited GPU memory | $1 \times 10^7 - 5 \times 10^7$ | Reduce peak memory from gradient buffers |

### Communication Pattern (Stage 2)

```
Backward Pass (with overlap_comm=False):
  For each bucket:
    1. Compute gradients for bucket parameters
    2. reduce_scatter(gradients)  -> each rank keeps 1/Nd of averaged gradients
    3. Free non-local gradient memory immediately

Optimizer Step:
  4. Update local optimizer states with local gradient partition
  5. Update local FP32 master weights
  6. Convert to FP16 parameters
  
After Optimizer:
  7. all_gather(updated FP16 parameters) -> all ranks get full updated model

Total communication volume: 2 * Psi * sizeof(fp16) * N_dp
(but peak gradient memory is reduced by Nd)
```

## Stage 3: Parameter Partitioning

Stage 3 (also called ZeRO-3 or FwDP - Fully Sharded Data Parallel) completes the partitioning by also sharding model parameters across data-parallel ranks. Each rank owns only $\frac{1}{N_d}$ of all model states. Parameters are gathered on-demand during forward and backward computation and immediately discarded after use.

### DeepSpeedZeroOptimizer_Stage3

The Stage 3 optimizer is implemented in `deepspeed/runtime/zero/stage3.py`:

```python
class DeepSpeedZeroOptimizer_Stage3(object):
    """ZeRO Stage 3 optimizer with full parameter, gradient, and optimizer state partitioning."""
    
    def __init__(self,
                 module,
                 init_optimizer,
                 timers,
                 ds_config,
                 static_loss_scale,
                 dynamic_loss_args,
                 verbose,
                 contiguous_gradients,
                 reduce_bucket_size,
                 allgather_bucket_size,
                 dp_process_group,
                 reduce_scatter,
                 overlap_comm,
                 cpu_offload,
                 mpu=None):
        ...
```

### partition_parameters.py - Init Context Manager

The `PartitionParameters` context manager in `deepspeed/runtime/zero/partition_parameters.py` intercepts parameter creation to immediately partition parameters:

```python
class Init:
    """Context manager for parameter partitioning in ZeRO Stage 3.
    
    Replaces torch.nn.Module.__init__ to intercept parameter registration
    and immediately partition parameters across data-parallel ranks.
    """
    
    def __enter__(self):
        # Save original __init__
        self._original_init = torch.nn.Module.__init__
        # Replace with partitioning __init__
        torch.nn.Module.__init__ = self._partitioned_init
        return self
    
    def __exit__(self, *args):
        # Restore original __init__
        torch.nn.Module.__init__ = self._original_init
    
    def _partitioned_init(self, module):
        self._original_init(module)
        # Immediately partition all parameters of the module
        for name, param in list(module.named_parameters(recurse=False)):
            self._partition_param(param, module, name)
```

Usage:

```python
import deepspeed

with deepspeed.zero.Init(config_dict=ds_config):
    # All parameters created within this block are automatically partitioned
    model = MyLargeModel(...)
    # Parameters are distributed across GPUs; each GPU holds only 1/Nd
```

### GatheredParameters Context

The `GatheredParameters` context manager temporarily gathers all (or specified) parameters for operations requiring full parameters (e.g., model initialization, checkpoint loading):

```python
class GatheredParameters(object):
    """Context manager to temporarily gather partitioned parameters.
    
    Parameters are gathered at entry and re-partitioned at exit.
    """
    
    def __init__(self, params, modifier_rank=None, fwd_module=None, enabled=True):
        self.params = [p for p in params if hasattr(p, 'ds_id')]
        self.modifier_rank = modifier_rank
        self.enabled = enabled
    
    def __enter__(self):
        if self.enabled:
            for param in self.params:
                param.ds_active_sub_modules += 1
                # Gather full parameter from all ranks
                self._gather_param(param)
        return self
    
    def __exit__(self, *args):
        if self.enabled:
            for param in self.params:
                param.ds_active_sub_modules -= 1
                if param.ds_active_sub_modules == 0:
                    # Re-partition: discard non-local chunks
                    self._discard_nonlocal(param)
```

Usage:

```python
with deepspeed.zero.GatheredParameters(model.parameters()):
    # All parameters are now fully available on all ranks
    torch.save(model.state_dict(), "checkpoint.pt")
# Parameters are re-partitioned upon exit
```

### Parameter Lifecycle

Each parameter in Stage 3 has a lifecycle managed by the `ParamInferenceState` enum:

```
NOT_AVAILABLE -> INFLIGHT -> AVAILABLE -> (discard) -> NOT_AVAILABLE
```

| State | Description |
|-------|-------------|
| `NOT_AVAILABLE` | Parameter data is not on this GPU; only the local partition (1/Nd) is stored |
| `INFLIGHT` | Parameter is being gathered from all ranks via all-gather operation |
| `AVAILABLE` | Full parameter is available on this GPU for computation |
| Discard | After computation, non-local portions of the parameter are freed |

```python
# deepspeed/runtime/zero/partition_parameters.py
class ParamInferenceState(Enum):
    NOT_AVAILABLE = 0   # Parameter not present on this rank
    INFLIGHT = 1         # All-gather in progress
    AVAILABLE = 2        # Full parameter available for use
```

The parameter lifecycle is managed through forward and backward hooks:

```python
# Forward pre-hook: gather parameter before layer computation
def _pre_forward_hook(self, module, *args):
    for param in module.parameters():
        if param.ds_status == ParamInferenceState.NOT_AVAILABLE:
            param.ds_status = ParamInferenceState.INFLIGHT
            self._gather_param(param)  # Initiate all-gather
            param.ds_status = ParamInferenceState.AVAILABLE

# Forward post-hook: release parameter after layer computation
def _post_forward_hook(self, module, *args):
    for param in module.parameters():
        if param.ds_status == ParamInferenceState.AVAILABLE:
            self._release_param(param)  # Discard non-local chunks
            param.ds_status = ParamInferenceState.NOT_AVAILABLE
```

### Prefetching

Stage 3 supports parameter prefetching to overlap communication with computation. While layer $L$ is computing, parameters for layer $L+1$ (or further) are being gathered.

```python
# Prefetch mechanism
def _prefetch_params(self, upcoming_modules):
    """Start gathering parameters for upcoming modules while current module computes."""
    for module in upcoming_modules:
        for param in module.parameters():
            if param.ds_status == ParamInferenceState.NOT_AVAILABLE:
                param.ds_status = ParamInferenceState.INFLIGHT
                # Initiate async all-gather
                self._async_gather_param(param)
```

### Configuration

```json
{
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
        "prefetch_bucket_size": 5e7,
        "max_live_parameters": 1e9,
        "max_reuse_distance": 1e9,
        "param_persistence_threshold": 1e5,
        "memory_efficient_linear": true,
        "module_granularity_threshold": 0,
        "allgather_sequential": false,
        "sub_group_size": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_max_live_parameters": 1e9,
        "stage3_param_persistence_threshold": 1e5
    }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prefetch_bucket_size` | int | $5 \times 10^7$ | Maximum number of elements to prefetch in a single all-gather. Controls communication/computation overlap granularity |
| `max_live_parameters` | int | $1 \times 10^9$ | Upper bound on the number of parameters resident in GPU memory simultaneously. Limits peak memory at the cost of more frequent gathering |
| `max_reuse_distance` | int | $1 \times 10^9$ | Maximum parameter reuse distance (in number of parameters traversed) before eviction. Parameters reused within this window are kept in memory |
| `param_persistence_threshold` | int | $1 \times 10^5$ | Do not partition parameters with fewer elements than this threshold. Small parameters (biases, norms) are kept locally to avoid communication overhead |
| `memory_efficient_linear` | bool | true | Use memory-efficient implementation of linear layers that avoids gathering full weight matrices |
| `module_granularity_threshold` | int | 0 | Minimum number of parameters a module must have to be considered for partitioning. 0 means partition all modules |
| `allgather_sequential` | bool | false | Perform all-gather operations sequentially instead of in parallel. Useful for debugging or when NCCL bandwidth is limited |
| `sub_group_size` | int | $1 \times 10^9$ | Number of parameters in each sub-group for optimizer step processing |

### Memory Savings Calculation (Stage 3)

| Component | Per-GPU Memory (DDP) | Per-GPU Memory (Stage 3) |
|-----------|---------------------|--------------------------|
| FP16 Parameters | $2\Psi$ | $\frac{2\Psi}{N_d}$ |
| FP16 Gradients | $2\Psi$ | $\frac{2\Psi}{N_d}$ |
| FP32 Optimizer States | $12\Psi$ | $\frac{12\Psi}{N_d}$ |
| Temp buffers | 0 | $\frac{2\Psi}{N_d}$ (prefetch + live) |
| **Total** | $16\Psi$ | $\approx \frac{16\Psi}{N_d}$ + temp buffers |

With $N_d = 64$: $\approx 0.25\Psi$ bytes per GPU. A 7B parameter model needs only ~1.75 GB per GPU for model states (vs 112 GB with DDP).

### Communication Pattern (Stage 3)

```
Forward Pass:
  For each layer L:
    1. all_gather(params[L])           # Gather this layer's parameters
    2. Prefetch: all_gather(params[L+1..L+k])  # Overlap with computation
    3. Compute: output = layer_L(input)
    4. Discard params[L] non-local chunks

Backward Pass (reverse order):
  For each layer L:
    1. all_gather(params[L])           # Re-gather for backward computation
    2. Compute gradients for layer L
    3. reduce_scatter(grads[L])        # Each rank keeps 1/Nd of gradients
    4. Discard params[L] non-local chunks

Optimizer Step:
  5. Update local optimizer states and FP32 master weights
  6. Convert updated master weights to FP16
  7. Parameters remain partitioned; no all-gather needed

Total communication:
  Forward:  2 * Psi * sizeof(fp16)   (all-gather all params)
  Backward: 2 * Psi * sizeof(fp16)   (all-gather params + reduce-scatter grads)
  Total:    4 * Psi * sizeof(fp16)   (per micro-batch)
```

### Performance Tuning Tips for Stage 3

1. **Prefetch bucket size**: Set `prefetch_bucket_size` to match `allgather_bucket_size` for consistent behavior. Increase for better compute/comm overlap on large clusters.

2. **Max live parameters**: Reduce `max_live_parameters` if experiencing OOM. The default allows ~2 GB of FP16 parameters in flight.

3. **Parameter persistence threshold**: Increase `param_persistence_threshold` (e.g., to $5 \times 10^5$) to keep more small parameters local and reduce all-gather frequency.

4. **Enable overlap_comm**: Always set to `true` for multi-node Stage 3 training.

5. **Gradient accumulation**: Use micro-batching with gradient accumulation to amortize communication overhead:
   ```json
   {
       "gradient_accumulation_steps": 4,
       "train_micro_batch_size_per_gpu": 2
   }
   ```

6. **Contiguous memory**: Keep `contiguous_gradients=true` to reduce memory fragmentation.

## MiCS (Multi-Instance Communication Sharding)

MiCS extends ZeRO-3 with hierarchical communication patterns to reduce the communication volume at large scale. Instead of involving all data-parallel ranks in every all-gather and reduce-scatter, MiCS restricts communication to smaller sharding groups.

### Configuration

```json
{
    "zero_optimization": {
        "stage": 3,
        "mics_shard_size": 8,
        "mics_hierarchical_params_gather": true
    }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mics_shard_size` | int | -1 (disabled) | Number of ranks in each MiCS sharding group. Parameters are partitioned within this group only. E.g., with 64 ranks and `mics_shard_size=8`, parameters are partitioned across 8 ranks, and there are 8 such groups |
| `mics_hierarchical_params_gather` | bool | false | Use hierarchical parameter gathering. Instead of a flat all-gather across the shard group, use a two-level gather: intra-node first, then inter-node |

### How MiCS Works

Without MiCS, ZeRO-3 with 64 GPUs partitions parameters across all 64 GPUs. Every all-gather involves all 64 GPUs in a single collective operation.

With MiCS (`mics_shard_size=8`):
1. 64 GPUs are divided into 8 groups of 8 GPUs each
2. Parameters are partitioned within each group (each GPU holds $\frac{1}{8}$ of parameters)
3. All-gather operations are within-group only (8 GPUs, not 64)
4. Across groups, parameters are replicated (redundancy trade-off for communication reduction)

```
Without MiCS (Nd=64):
  Rank 0: owns params[0/64 .. 1/64]
  all_gather involves 64 ranks, 64-way communication

With MiCS (mics_shard_size=8):
  Group 0: Ranks [0..7],   each owns 1/8 of params
  Group 1: Ranks [8..15],  each owns 1/8 of params
  ...
  Group 7: Ranks [56..63], each owns 1/8 of params
  all_gather involves 8 ranks, 8-way communication (8x less traffic)
  
  Memory per GPU: 1/8 of model (vs 1/64 without MiCS)
  Communication: 8-way all-gather (vs 64-way)
```

### When to Use MiCS

- Large clusters (> 32 GPUs) where all-gather latency dominates
- Models that fit in $\frac{\text{GPU memory}}{\text{mics_shard_size}}$ with acceptable overhead
- Multi-node training with high inter-node latency

## Tiled Linear Layers

Tiled linear layers split large linear operations into smaller tiles, reducing peak memory during parameter gathering in Stage 3.

### TiledLinear

```python
# deepspeed/runtime/zero/linear.py
class TiledLinear(torch.nn.Module):
    """Tiled linear layer that processes weight matrix in tiles.
    
    Instead of gathering the full weight matrix W of shape [out, in],
    processes it in tiles of shape [tile_size, in], gathering one tile
    at a time to reduce peak memory.
    """
    
    def __init__(self, in_features, out_features, bias=True,
                 tile_factor=1, split_dim=0, ...):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tile_factor = tile_factor
        
        # Create tiled weight parameters
        self.weight_tiles = nn.ParameterList()
        for i in range(tile_factor):
            tile_out = out_features // tile_factor
            self.weight_tiles.append(
                nn.Parameter(torch.empty(tile_out, in_features))
            )
        
        if bias:
            self.bias_tiles = nn.ParameterList()
            for i in range(tile_factor):
                tile_out = out_features // tile_factor
                self.bias_tiles.append(
                    nn.Parameter(torch.empty(tile_out))
                )
    
    def forward(self, input):
        output_tiles = []
        for weight_tile in self.weight_tiles:
            # Only gather one tile at a time in Stage 3
            output_tile = F.linear(input, weight_tile)
            output_tiles.append(output_tile)
        return torch.cat(output_tiles, dim=-1)
```

### TiledLinearReturnBias

```python
class TiledLinearReturnBias(TiledLinear):
    """Tiled linear that returns bias as a separate tensor for fusion."""
    
    def forward(self, input):
        # Same tiling as TiledLinear but returns (output, bias) tuple
        ...
```

### Configuration

Tiled linear layers are enabled automatically by Stage 3 when a linear layer exceeds the tiling threshold. The tiling is transparent to the user model code.

## ZeRO++ Optimizations

ZeRO++ builds on top of ZeRO-3 with additional communication optimizations through quantization and hierarchical partitioning.

### zero_hpz_partition_size (Hierarchical Partitioning ZeRO)

```json
{
    "zero_optimization": {
        "stage": 3,
        "zero_hpz_partition_size": 4
    }
}
```

Creates two-level parameter partitioning:
- **Intra-node partitioning**: Parameters partitioned within a node (e.g., across 4 GPUs on same node)
- **Inter-node replication**: Different nodes hold replicas of the same parameter partitions
- Reduces inter-node communication by performing most all-gather operations intra-node

### zero_quantized_weights

```json
{
    "zero_optimization": {
        "stage": 3,
        "zero_quantized_weights": true
    }
}
```

When enabled, parameters are quantized (to INT8 or FP8) during all-gather operations, reducing communication volume by 2x for FP16 parameters.

### zero_quantized_gradients

```json
{
    "zero_optimization": {
        "stage": 3,
        "zero_quantized_gradients": true
    }
}
```

Quantizes gradients during reduce-scatter operations, reducing backward-pass communication volume.

### zeropp_loco_param

```json
{
    "zero_optimization": {
        "stage": 3,
        "zeropp_loco_param": true
    }
}
```

Enables LOCO (Low-Cost Communication) optimization that uses learned compression for parameter communication.

### Complete ZeRO++ Configuration

```json
{
    "zero_optimization": {
        "stage": 3,
        "zero_hpz_partition_size": 4,
        "zero_quantized_weights": true,
        "zero_quantized_gradients": true,
        "zeropp_loco_param": true,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    }
}
```

## Complete Configuration Examples

### Stage 1 - Single Node, 8 GPUs

```json
{
    "train_batch_size": 160,
    "train_micro_batch_size_per_gpu": 20,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    }
}
```

### Stage 2 - Multi-Node, 32 GPUs

```json
{
    "train_batch_size": 640,
    "train_micro_batch_size_per_gpu": 10,
    "gradient_accumulation_steps": 2,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0
    },
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
        "round_robin_gradients": true
    }
}
```

### Stage 3 - Large Model, 64 GPUs

```json
{
    "train_batch_size": 1280,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 10,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
        "prefetch_bucket_size": 5e7,
        "max_live_parameters": 1e9,
        "max_reuse_distance": 1e9,
        "param_persistence_threshold": 1e5,
        "memory_efficient_linear": true
    }
}
```

### Stage 3 with MiCS and ZeRO++

```json
{
    "train_batch_size": 2560,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 20,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
        "mics_shard_size": 8,
        "mics_hierarchical_params_gather": true,
        "zero_hpz_partition_size": 4,
        "zero_quantized_weights": true,
        "zero_quantized_gradients": true
    }
}
```

## Stage Selection Guide

```
                Model fits on single GPU?
                       /         \
                     Yes           No
                      |             |
               Use Stage 0    Model fits with
               (standard      optimizer states
                DDP)          offloaded?
                                   /         \
                                 Yes           No
                                  |             |
                           Use Stage 1     Model fits with
                           (optimizer       optimizer + gradient
                            partitioning)   offloaded?
                                               /         \
                                             Yes           No
                                              |             |
                                       Use Stage 2     Use Stage 3
                                       (optimizer +    (full partitioning
                                        gradient         + offload if
                                        partitioning)     needed)
```

### Quick Selection Table

| Model Size | GPU Count | GPU Memory | Recommended Stage |
|------------|-----------|------------|-------------------|
| < 1B | Any | >= 16 GB | Stage 0 or 1 |
| 1B - 7B | 8+ | 16-32 GB | Stage 2 |
| 7B - 13B | 8+ | 40 GB | Stage 2 or 3 |
| 13B - 30B | 16+ | 40-80 GB | Stage 3 |
| 30B - 70B | 32+ | 40-80 GB | Stage 3 + Offload |
| 70B+ | 64+ | 80 GB | Stage 3 + ZeRO++ + MiCS |

## Communication Volume Comparison

| Operation | Stage 0 | Stage 1 | Stage 2 | Stage 3 |
|-----------|---------|---------|---------|---------|
| Backward (grad sync) | $2\Psi$ all-reduce | $2\Psi$ all-reduce | $2\Psi$ reduce-scatter | $2\Psi$ reduce-scatter |
| Parameter update | None | $2\Psi$ broadcast | $2\Psi$ all-gather | $2\Psi$ all-gather (fwd) |
| Forward gather | None | None | None | $2\Psi$ all-gather |
| **Total per step** | $2\Psi$ | $4\Psi$ | $4\Psi$ | $6\Psi$ |

Note: All volumes are in bytes (FP16). Stage 3 has 3x the communication of Stage 0, but the trade-off is $\frac{1}{N_d}$ memory consumption.

## Key Source Files

| File | Description |
|------|-------------|
| `deepspeed/runtime/zero/stage_enum.py` | ZeroStageEnum definition |
| `deepspeed/runtime/zero/stage1and2.py` | DeepSpeedZeroOptimizer (Stages 1 & 2) |
| `deepspeed/runtime/zero/stage3.py` | DeepSpeedZeroOptimizer_Stage3 |
| `deepspeed/runtime/zero/partition_parameters.py` | Init context manager, GatheredParameters, parameter partitioning |
| `deepspeed/runtime/zero/linear.py` | TiledLinear, TiledLinearReturnBias |
| `deepspeed/runtime/zero/config.py` | DeepSpeedZeroConfig, DeepSpeedZeroOffloadParamConfig, DeepSpeedZeroOffloadOptimizerConfig |
| `deepspeed/runtime/zero/mics.py` | MiCS optimization |
| `deepspeed/runtime/zero/zeropp.py` | ZeRO++ optimizations |
