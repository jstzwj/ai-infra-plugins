# Distributed Checkpointing Reference

## Overview

Megatron-LM provides a distributed checkpointing system (`megatron.core.dist_checkpointing`) that
stores model state across multiple files, with each GPU saving only its local shard. This
enables efficient checkpointing for models with billions to trillions of parameters distributed
across hundreds or thousands of GPUs.

Key capabilities:
- Save and load checkpoints across arbitrary TP/PP/DP/EP configurations
- Rescale checkpoints to different parallelism configurations (change TP, PP, DP)
- Async checkpoint saving to overlap I/O with training
- Integrity verification with SHA-256 manifests
- Support for MoE models, FP8 weights, and distributed optimizers

## Checkpoint Format

### Directory Structure

A distributed checkpoint is stored as a directory containing:

```
checkpoint_dir/
  metadata.json                    # Backend and version metadata
  common.pt                        # Non-sharded data (args, rng state, etc.)
  [tp_rank]_[pp_rank]/             # Per-rank shard directories
    sharded_tensor_files.pt        # Sharded tensor data
```

For MoE models with expert parallelism:

```
checkpoint_dir/
  metadata.json
  common.pt
  [tp_rank]_[pp_rank]_[ep_rank]/  # Includes expert parallel rank
    sharded_tensor_files.pt
```

### Metadata Format

The `metadata.json` file contains the checkpoint format configuration:

```json
{
  "sharded_backend": "torch_dist",
  "sharded_backend_version": 1,
  "common_backend": "torch",
  "common_backend_version": 1
}
```

### Checking if a Checkpoint is Distributed

```python
from megatron.core.dist_checkpointing import check_is_distributed_checkpoint

is_dist = check_is_distributed_checkpoint("/path/to/checkpoint")
```

This checks if `metadata.json` exists and contains a valid `CheckpointingConfig`.

## Core Data Types

### ShardedTensor

`ShardedTensor` represents a mapping between a local tensor shard and the global tensor it
belongs to:

```python
from megatron.core.dist_checkpointing.mapping import ShardedTensor

# Create a ShardedTensor for TP-sharded weight
sharded_weight = ShardedTensor(
    key="model.layers.0.self_attention.linear_qkv.weight",
    data=local_weight_tensor,           # Local shard [hidden/TP, hidden]
    dtype=torch.bfloat16,
    local_shape=(hidden_size // tp_size, hidden_size),
    global_shape=(hidden_size, hidden_size),
    global_offset=(tp_rank * hidden_size // tp_size, 0),
    axis_fragmentations=(tp_size, 1),
    replica_id=0,
)
```

#### from_rank_offsets Constructor

The recommended way to create ShardedTensors using rank-based offsets:

```python
sharded_tensor = ShardedTensor.from_rank_offsets(
    key="model.weight",
    data=local_tensor,
    # (axis, rank_offset, fragmentation) tuples:
    (0, tp_rank, tp_size),      # Axis 0 sharded by TP
    (1, dp_rank, dp_size),      # Axis 1 sharded by DP
    replica_id=0,
    prepend_axis_num=0,
)
```

Each tuple `(axis, rank_offset, fragmentation)` specifies that the global tensor is divided
into `fragmentation` parts along `axis`, and the local data corresponds to the `rank_offset` chunk.

### ShardedObject

`ShardedObject` represents non-tensor objects distributed across ranks:

```python
from megatron.core.dist_checkpointing.mapping import ShardedObject

sharded_obj = ShardedObject(
    key="optimizer.state",
    data=local_optimizer_state,
    global_shape=(dp_size,),
    global_offset=(dp_rank,),
    replica_id=0,
)
```

### LocalNonpersistentObject

Objects that should not be stored in the checkpoint but restored locally:

```python
from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject

# During saving: this object will be skipped
# During loading: a local version is placed in the state dict
state_dict["temp_buffer"] = LocalNonpersistentObject(local_buffer)
```

### ShardedTensorFactory

Allows applying transformations to tensors before/after serialization:

```python
from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory

factory = ShardedTensorFactory(
    key="param",
    data=original_tensor,
    build_fn=lambda key, data, replica_id, flattened_range: {
        "param": ShardedTensor.from_rank_offsets(key, data, ...)
    },
    merge_fn=lambda state_dict: state_dict["param"],
    replica_id=0,
)
```

## Save and Load API

### Saving Checkpoints

```python
from megatron.core.dist_checkpointing import save
from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy

# Save a distributed checkpoint
save(
    sharded_state_dict=state_dict,
    checkpoint_dir="/path/to/checkpoint",
    validate_access_integrity=True,
)
```

#### Save with Async

```python
# Async save returns an AsyncRequest that must be scheduled
async_request = save(
    sharded_state_dict=state_dict,
    checkpoint_dir="/path/to/checkpoint",
    async_sharded_save=True,
    async_strategy="nvrx",  # or "mcore"
)

# Schedule the async save
async_calls_queue.schedule_async_request(async_request)

# Later, check completion and finalize
async_calls_queue.maybe_finalize_async_calls(blocking=True)
```

#### Save Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sharded_state_dict` | ShardedStateDict | State dict with ShardedTensors |
| `checkpoint_dir` | str | Directory to save checkpoint |
| `sharded_strategy` | TorchDistSaveShardedStrategy | Save backend configuration |
| `validate_access_integrity` | bool | Check each shard accessed exactly once |
| `async_sharded_save` | bool | Enable async saving |
| `preprocess_common_before_consistancy_check` | Callable | Preprocess common state dict |
| `content_metadata` | dict | Custom metadata to store |
| `verify_integrity` | bool | Compute SHA-256 manifest |
| `async_strategy` | str | Async backend: "nvrx" or "mcore" |

### Loading Checkpoints

```python
from megatron.core.dist_checkpointing import load
from megatron.core.dist_checkpointing.strategies.torch import TorchDistLoadShardedStrategy

# Load a distributed checkpoint
state_dict = load(
    sharded_state_dict=model_state_dict,
    checkpoint_dir="/path/to/checkpoint",
    validate_access_integrity=True,
    strict="assume_ok_unexpected",  # or "return_all", "log_unexpected"
)
```

#### Strict Loading Options

```python
from megatron.core.dist_checkpointing.validation import StrictHandling

# Don't check for mismatches (fastest, default)
state_dict = load(sharded_state_dict, ckpt_dir,
                  strict=StrictHandling.ASSUME_OK_UNEXPECTED)

# Log unexpected keys but don't fail
state_dict = load(sharded_state_dict, ckpt_dir,
                  strict=StrictHandling.LOG_UNEXPECTED)

# Return missing and unexpected keys
state_dict, missing, unexpected = load(sharded_state_dict, ckpt_dir,
                                        strict=StrictHandling.RETURN_ALL)
```

#### Load Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sharded_state_dict` | ShardedStateDict | Template state dict for loading |
| `checkpoint_dir` | str | Directory to load from |
| `sharded_strategy` | TorchDistLoadShardedStrategy | Load backend configuration |
| `validate_access_integrity` | bool | Validate shard access patterns |
| `strict` | StrictHandling | Handling of key mismatches |
| `verify_integrity` | bool | Verify SHA-256 manifest |

### Load Metadata Only

```python
from megatron.core.dist_checkpointing import (
    load_common_state_dict,
    load_tensors_metadata,
    load_content_metadata,
)

# Load only non-sharded data (args, iteration, rng state)
common = load_common_state_dict("/path/to/checkpoint")

# Load tensor metadata (shapes, dtypes) without data
metadata = load_tensors_metadata("/path/to/checkpoint")

# Load custom content metadata
content_meta = load_content_metadata("/path/to/checkpoint")

# Load plain tensors (all shards, no sharding)
plain = load_plain_tensors("/path/to/checkpoint")
```

## Checkpoint Rescaling

One of the most powerful features of distributed checkpointing is the ability to load a
checkpoint saved with one parallelism configuration into a model with a different configuration.

### Changing Tensor Parallelism

Load a checkpoint saved with TP=4 into a model with TP=8:

```bash
# Saved with: --tensor-model-parallel-size 4
# Load with:  --tensor-model-parallel-size 8
```

The ShardedTensor system automatically handles the mapping. During loading, the model's
sharded state dict specifies the new sharding, and the loading infrastructure reads the
appropriate shards from the old checkpoint.

### Changing Pipeline Parallelism

```bash
# Saved with: --pipeline-model-parallel-size 2
# Load with:  --pipeline-model-parallel-size 4
```

Each PP rank loads only the layers it owns in the new configuration.

### Changing Expert Parallelism

For MoE models, EP rescaling allows changing how experts are distributed:

```bash
# Saved with: --expert-model-parallel-size 4 --num-experts 8
# Load with:  --expert-model-parallel-size 8 --num-experts 8
```

The expert sharded state dict maps local experts to global experts, enabling rescaling:

```python
# From experts.py SequentialMLP.sharded_state_dict()
expert_sharded_offsets = (
    *sharded_offsets,
    (len(sharded_offsets), expert_global_idx, num_global_experts),
)
```

### Changing Data Parallelism with Distributed Optimizer

```bash
# Saved with: --use-distributed-optimizer --tensor-model-parallel-size 4
# Load with:  --use-distributed-optimizer --tensor-model-parallel-size 8
```

The distributed optimizer states are resharded automatically during loading.

### How Rescaling Works

1. The model creates a sharded state dict with the **new** parallelism configuration
2. The loading infrastructure reads shards from the checkpoint saved with the **old** config
3. Each rank determines which global tensor regions it needs
4. The appropriate shards are loaded and assembled

The key is that `ShardedTensor` tracks `global_shape`, `global_offset`, and
`axis_fragmentations`, enabling the loading code to map between configurations.

## Checkpoint Conversion Utilities

Megatron-LM provides tools in `tools/checkpoint/` for converting between checkpoint formats.

### Converting Checkpoint Formats

```bash
# Convert from Megatron-LM legacy to distributed format
python tools/checkpoint/convert.py \
    --model-type GPT \
    --load-dir /path/to/legacy_checkpoint \
    --save-dir /path/to/distributed_checkpoint \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 4

# Convert from distributed to HuggingFace format
python tools/checkpoint/saver_hf.py \
    --load-dir /path/to/megatron_checkpoint \
    --save-dir /path/to/hf_checkpoint
```

### Upcycling Dense Models to MoE

Convert a dense model checkpoint to an MoE model:

```bash
--moe-use-upcycling
--load /path/to/dense_checkpoint
--save /path/to/moe_checkpoint
--num-experts 8
```

The dense MLP is duplicated to create initial expert weights. Granular upcycling creates
smaller experts:

```bash
--moe-use-upcycling
--moe-upcycling-granularity 4  # Expert hidden size = dense FFN / 4
```

## Async Checkpoint Saving

Async checkpoint saving writes checkpoints in a background process, overlapping I/O with
training computation to minimize training interruption.

### Configuration

```bash
--async-save                    # Enable async saving
--ckpt-format torch_dist        # Required for async save
--auto-detect-ckpt-format       # Auto-detect checkpoint format
```

### Async Architecture

The async saving system uses `mp.Process` (multiprocessing) to offload checkpoint writing:

```
Training Process                    Async Writer Process
    |                                      |
    | -- save() called -->                 |
    |    GPU->CPU tensor staging           |
    |    Fork async process ------------>  |
    |    Continue training                 |
    |                                      | -- Write to storage
    |                                      | -- Signal completion
    | -- Check completion <--              |
    |    Finalize (metadata, barrier)      |
```

### AsyncRequest

The `AsyncRequest` class encapsulates an async checkpoint operation:

```python
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest

async_request = AsyncRequest(
    async_fn=save_function,            # Function to call in background
    async_fn_args=(args,),             # Positional arguments
    finalize_fns=[metadata_finalize],  # Functions to call after all ranks finish
    async_fn_kwargs={},                # Keyword arguments
    preload_fn=preload_to_cpu,         # GPU->CPU staging function
)
```

### AsyncCaller Types

**TemporalAsyncCaller:** Spawns a new process for each checkpoint save.

```python
from megatron.core.dist_checkpointing.strategies.async_utils import (
    TemporalAsyncCaller,
    AsyncCallsQueue,
)

queue = AsyncCallsQueue(persistent=False)
queue.schedule_async_request(async_request)
# ... continue training ...
finalized = queue.maybe_finalize_async_calls(blocking=False)
```

**PersistentAsyncCaller:** Reuses a single background process across saves.

```python
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue

queue = AsyncCallsQueue(persistent=True)

# Warmup the persistent worker at training start
AsyncCallsQueue.warmup_persistent_caller(
    rank=torch.distributed.get_rank(),
    mp_mode='spawn',
    cpu_priority=10,     # Lower priority than training
    io_priority=3,       # Idle I/O class
)

# Schedule saves
queue.schedule_async_request(async_request)
```

### QoS (Quality of Service)

The persistent async writer sets lower CPU and I/O priority to avoid interfering with training:

```python
# From async_utils.py
_set_process_qos(
    cpu_priority=10,   # Nice value 0-19 (higher = lower priority)
    io_priority=None,  # I/O scheduling class
)
```

### Checking Completion

```python
# Non-blocking check
finalized = queue.maybe_finalize_async_calls(blocking=False)

# Blocking wait for all pending saves
finalized = queue.maybe_finalize_async_calls(blocking=True)

# Check number of unfinalized saves
count = queue.get_num_unfinalized_calls()
```

### Cleanup

```python
# Clean shutdown (waits for pending saves)
queue.close(abort=False)

# Emergency shutdown
queue.close(abort=True)
```

## High-Level Training Checkpoint API

The training script uses a higher-level checkpoint API in
`megatron/training/checkpointing.py`.

### Save Checkpoint

```python
from megatron.training.checkpointing import save_checkpoint

save_checkpoint(
    iteration,          # Current training iteration
    model,              # Model (or list of models for virtual pipeline)
    optimizer,          # Optimizer
    opt_param_scheduler,# Learning rate scheduler
    num_floating_point_operations_so_far,  # FLOPs tracker
    force_async_save=False,  # Force async save
    checkpointing_context=None,  # Context for async saves
)
```

### Load Checkpoint

```python
from megatron.training.checkpointing import load_checkpoint

iteration, state_dict = load_checkpoint(
    model,              # Model to load into
    optimizer,          # Optimizer to load into
    opt_param_scheduler,# LR scheduler
    load_arg="load",    # Argument name for checkpoint path
    strict=True,        # Strict state dict loading
)
```

### Find Latest Checkpoint

```python
from megatron.training.checkpointing import (
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    checkpoint_exists,
)

# Check if any checkpoint exists
if checkpoint_exists("/path/to/checkpoints"):
    # Read tracker file for latest iteration
    tracker_file = get_checkpoint_tracker_filename("/path/to/checkpoints")
    iteration, release = read_metadata(tracker_file)

    # Get checkpoint directory for specific iteration
    ckpt_dir = get_checkpoint_name(
        "/path/to/checkpoints",
        iteration,
        return_base_dir=True,  # Return directory, not file
    )
```

## Checkpointing for MoE Models

MoE models require special checkpoint handling because expert parameters are distributed across
EP ranks.

### Expert State Dict

```python
# From experts.py SequentialMLP.sharded_state_dict()
def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
    sharded_state_dict = {}
    num_global_experts = self.ep_group.size() * self.num_local_experts
    local_expert_indices_offset = self.ep_group.rank() * self.num_local_experts

    for expert_local_idx, expert in enumerate(self.local_experts):
        expert_global_idx = local_expert_indices_offset + expert_local_idx
        expert_sharded_offsets = (
            *sharded_offsets,
            (len(sharded_offsets), expert_global_idx, num_global_experts),
        )
        expert_state_dict = expert.sharded_state_dict(
            f'{prefix}local_experts.{expert_local_idx}.',
            expert_sharded_offsets,
            metadata,
        )
        sharded_state_dict.update(expert_state_dict)
    return sharded_state_dict
```

The expert axis is added as an additional sharding dimension, enabling resharding when EP changes.

### Shared Expert Checkpointing

```python
# From shared_experts.py SharedExpertMLP.sharded_state_dict()
def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
    sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
    if self.use_shared_expert_gate:
        sub_sd = {
            f'{prefix}gate_weight': make_sharded_tensor_for_checkpoint(
                state_dict['gate_weight'],
                f'{prefix}gate_weight',
                prepend_offsets=sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=metadata['dp_cp_group'],
            )
        }
        sharded_state_dict.update(sub_sd)
    return sharded_state_dict
```

## Integrity Verification

### Saving with Integrity Check

```python
save(
    sharded_state_dict=state_dict,
    checkpoint_dir="/path/to/checkpoint",
    verify_integrity=True,  # Compute SHA-256 manifest
)
```

This computes SHA-256 hashes for every file in the checkpoint directory after all data has been
written. The manifest is stored alongside the checkpoint data.

### Loading with Integrity Verification

```python
load(
    sharded_state_dict=state_dict,
    checkpoint_dir="/path/to/checkpoint",
    verify_integrity=True,  # Verify SHA-256 manifest
)
```

This re-hashes every checkpoint file and compares against the stored SHA-256 manifest. Raises
`CheckpointingException` on any mismatch.

**Performance note:** Integrity verification adds I/O overhead proportional to total checkpoint
size (one extra read pass over all files on rank 0).

## Best Practices for Large Models

### 1. Use Distributed Checkpoints Exclusively

```bash
--ckpt-format torch_dist
--auto-detect-ckpt-format
```

Distributed checkpoints are required for models that cannot fit in a single GPU's memory and
enable checkpoint rescaling.

### 2. Enable Async Saving

```bash
--async-save
--ckpt-format torch_dist
```

Async saving overlaps checkpoint I/O with training, minimizing throughput impact. The typical
overhead reduction is 70-90% compared to synchronous saving.

### 3. Use Distributed Optimizer

```bash
--use-distributed-optimizer
```

The distributed optimizer shards optimizer states across DP ranks, reducing checkpoint size per
rank and enabling optimizer state rescaling when DP changes.

### 4. Save Less Frequently for Large Models

```bash
--save-interval 5000     # Save every 5000 iterations
--eval-interval 1000     # Evaluate more frequently than saving
```

### 5. Clean Up Old Checkpoints

```bash
# Keep only the last N checkpoints
--keep-last-n-checkpoints 3
```

### 6. Use Non-Persistent Checkpoints for Frequent Saves

For rollback protection without full checkpoint overhead, use non-persistent checkpoints:

```python
# Non-persistent checkpoints are saved to a subdirectory
# and deleted on the next save
checkpointing_context = {
    'non_persistent_ckpt': True,
}
save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                num_floating_point_operations_so_far,
                checkpointing_context=checkpointing_context)
```

### 7. Handle FP8 Tensors Correctly

The distributed checkpointing system automatically handles FP8 tensor dequantization during
loading:

```python
# From serialization.py load()
force_all_tensors_to_non_fp8(sharded_state_dict)
```

This ensures FP8 tensors are converted to high-precision tensors before loading, preventing
issues with quantization state and delayed scaling.

### 8. Use Fully Parallel Save for Large Clusters

For large-scale training (1000+ GPUs), use fully parallel save to distribute the save I/O:

```python
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelSaveStrategyWrapper,
)

save_strategy = FullyParallelSaveStrategyWrapper(
    TorchDistSaveShardedStrategy(),
    parallel_group=dp_group,
)
```

### 9. Memory-Efficient Loading

When loading large checkpoints, minimize peak memory:

```python
# Load sharded metadata first to plan memory allocation
metadata = load_tensors_metadata(checkpoint_dir)

# Load with minimal validation for faster loading
state_dict = load(
    sharded_state_dict,
    checkpoint_dir,
    validate_access_integrity=False,  # Skip for faster loading
)
```

### 10. Checkpoint Conversion Between Frameworks

```bash
# Megatron -> HuggingFace
python tools/checkpoint/saver_hf.py

# HuggingFace -> Megatron
python tools/checkpoint/loader_hf.py

# Megatron legacy -> Megatron distributed
python tools/checkpoint/convert.py --model-type GPT
```

## Configuration Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--ckpt-format` | Format: torch, torch_dist, zarr | torch |
| `--auto-detect-ckpt-format` | Auto-detect format on load | False |
| `--async-save` | Enable async checkpoint saving | False |
| `--save-interval` | Iterations between saves | 1000 |
| `--keep-last-n-checkpoints` | Number of checkpoints to keep | None (all) |
| `--use-distributed-optimizer` | Shard optimizer states | False |
| `--save` | Directory to save checkpoints | None |
| `--load` | Directory to load checkpoints | None |
| `--no-save-optim` | Don't save optimizer state | False |
| `--no-load-optim` | Don't load optimizer state | False |
| `--finetune` | Load model but not iteration | False |
| `--ckpt-step` | Specific checkpoint step to load | None |
