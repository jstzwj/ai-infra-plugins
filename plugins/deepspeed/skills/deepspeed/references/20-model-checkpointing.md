# DeepSpeed Model Checkpointing

## Overview

DeepSpeed provides a comprehensive model checkpointing system for saving and loading training state. The system supports multiple checkpoint formats, elastic training (changing GPU counts), universal checkpoint conversion, asynchronous checkpointing, and integration with external checkpoint storage systems (Nebula, DataStates). The checkpointing infrastructure is designed to handle models ranging from millions to trillions of parameters across hundreds or thousands of GPUs.

---

## Module Architecture

```
deepspeed/checkpoint/
    __init__.py
    deepspeed_checkpoint.py       # Main checkpoint manager class
    universal_checkpoint.py       # Universal checkpoint format handler
    ds_to_universal.py            # Conversion tool: DeepSpeed -> Universal
    zero_checkpoint.py            # ZeRO-specific checkpoint utilities
    constants.py                  # Checkpoint constants and keys

deepspeed/runtime/
    model_checkpointing/
        __init__.py
        config.py                 # CheckpointConfig class
        model_checkpointing.py    # Core save/load implementation
```

---

## CheckpointConfig (runtime/model_checkpointing/config.py)

The `CheckpointConfig` class defines the checkpointing behavior through the DeepSpeed JSON configuration.

### Configuration Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tag_validation` | str | `"Ignore"` | How to handle checkpoint tag mismatches. `"Ignore"`: proceed without warning. `"Warn"`: log a warning but continue. `"Fail"`: raise an error and stop loading. |
| `load_universal` | bool | `false` | When `true`, loads checkpoints in universal format instead of DeepSpeed-native format. |
| `use_node_local_storage` | bool | `false` | When `true`, each node saves checkpoints to its local storage instead of shared filesystem. Faster for large checkpoints but requires manual aggregation. |
| `parallel_write` | dict or bool | `false` | Configuration for parallel checkpoint writing. Can include `pipeline_stage` for pipelined writes. |
| `save_latest` | bool | `true` | Whether to maintain a `latest` file pointing to the most recent checkpoint. |

### Configuration Example

```json
{
    "checkpoint": {
        "tag_validation": "Warn",
        "load_universal": false,
        "use_node_local_storage": false,
        "parallel_write": {
            "pipeline_stage": true
        }
    }
}
```

---

## save_checkpoint()

The `save_checkpoint()` method saves the complete training state to disk.

### Method Signature

```python
def save_checkpoint(self, save_dir, tag=None, client_state=None, save_latest=True):
    """Save the complete training state to disk.

    Args:
        save_dir (str): Directory to save the checkpoint. The checkpoint
                        will be stored at save_dir/tag/ (or save_dir/global_stepXXXX/).
        tag (str, optional): A checkpoint tag (e.g., "global_step1000").
                            If None, uses the current global step count.
        client_state (dict, optional): Additional state to save alongside the
                                       DeepSpeed state. Typically includes:
                                       - epoch_num
                                       - global_step
                                       - client_weights
                                       - any user-defined state
        save_latest (bool): Whether to update the 'latest' pointer file.
                           Defaults to True.

    Returns:
        None

    Raises:
        RuntimeError: If checkpoint saving fails.
    """
```

### What Is Saved

A DeepSpeed checkpoint contains:

1. **Model parameters**: The model's weight tensors (partitioned or gathered depending on ZeRO stage).
2. **Optimizer states**: First and second moment estimates, step counts for all parameter groups.
3. **Learning rate scheduler state**: Current step, learning rate, and any scheduler-internal state.
4. **Random number generator states**: Python, CPU, and CUDA RNG states for reproducibility.
5. **Training progress**: Global step, consumed samples, gradient accumulation state.
6. **DeepSpeed internal state**: ZeRO state, FP16 loss scaler state, sparse tensor state.
7. **Client state**: User-specified additional state (via `client_state` parameter).
8. **Checkpoint tag**: A string identifying the checkpoint (e.g., `"global_step10000"`).

### Checkpoint Directory Structure

```
save_dir/
    global_step10000/
        mp_rank_00/                    # Model parallel rank 0
            model_states.pt            # Model parameters and optimizer states
        zero_pp_rank_0_mp_rank_00/     # ZeRO pipeline-parallel rank 0
            model_states.pt            # Partitioned model states
            optimizer_states.pt        # Partitioned optimizer states
        zero_pp_rank_1_mp_rank_00/
            model_states.pt
            optimizer_states.pt
        ...
        latest                         # Points to this checkpoint
        zero_checkpoint.json           # ZeRO checkpoint metadata
    latest                             # Points to the latest checkpoint tag
```

### Usage Examples

#### Basic Save

```python
# Save at the current global step
model_engine.save_checkpoint(save_dir="/checkpoints/my_model")
```

#### Save with Custom Tag

```python
# Save with a custom tag
model_engine.save_checkpoint(
    save_dir="/checkpoints/my_model",
    tag="epoch_5"
)
```

#### Save with Client State

```python
# Save with additional training state
client_state = {
    "epoch": 5,
    "global_step": 10000,
    "best_eval_loss": 0.123,
    "tokenizer_state": tokenizer.save_state(),
}
model_engine.save_checkpoint(
    save_dir="/checkpoints/my_model",
    client_state=client_state
)
```

#### Save Without Updating Latest

```python
# Save a temporary checkpoint without updating 'latest'
model_engine.save_checkpoint(
    save_dir="/checkpoints/my_model",
    tag="temp_recovery",
    save_latest=False
)
```

### ZeRO Stage-Specific Save Behavior

#### ZeRO Stage 1 (No Partitioning)

All parameters and optimizer states are saved in full (unpartitioned). Each data-parallel rank saves an identical copy.

```
save_dir/global_stepXXXX/
    mp_rank_00_model_states.pt     # Full model + optimizer
```

#### ZeRO Stage 2 (Optimizer State Partitioning)

Optimizer states are partitioned across data-parallel ranks. Each rank saves only its partition. Model parameters are saved in full.

```
save_dir/global_stepXXXX/
    mp_rank_00_model_states.pt           # Full model parameters
    zero_pp_rank_X_mp_rank_XX_optim_states.pt  # Partitioned optimizer states
```

#### ZeRO Stage 3 (Full Partitioning)

Parameters, gradients, and optimizer states are all partitioned. Each rank saves only its partitions.

```
save_dir/global_stepXXXX/
    zero_pp_rank_X_mp_rank_XX_model_states.pt   # Partitioned model parameters
    zero_pp_rank_X_mp_rank_XX_optim_states.pt   # Partitioned optimizer states
```

---

## load_checkpoint()

The `load_checkpoint()` method restores training state from a saved checkpoint.

### Method Signature

```python
def load_checkpoint(self, load_dir, tag=None, load_module_strict=True,
                    load_optimizer_states=True, load_lr_scheduler_states=True,
                    load_module_only=False):
    """Load training state from a checkpoint.

    Args:
        load_dir (str): Directory containing the checkpoint.
        tag (str, optional): Specific checkpoint tag to load. If None,
                            loads from the 'latest' checkpoint.
        load_module_strict (bool): If True, requires that the checkpoint's
                                   model state dict keys exactly match the
                                   current model. If False, allows missing
                                   or extra keys.
        load_optimizer_states (bool): If True, loads optimizer states.
                                      Set to False when fine-tuning with
                                      a different optimizer configuration.
        load_lr_scheduler_states (bool): If True, loads LR scheduler states.
        load_module_only (bool): If True, loads only model parameters,
                                  skipping optimizer and scheduler states.

    Returns:
        tuple: (load_path, client_state) where load_path is the path to the
               loaded checkpoint and client_state is any user-defined state
               saved with the checkpoint.
    """
```

### Usage Examples

#### Basic Load (Resume Training)

```python
# Load the latest checkpoint for resumption
load_path, client_state = model_engine.load_checkpoint(
    load_dir="/checkpoints/my_model"
)
print(f"Loaded from: {load_path}")
print(f"Previous global step: {client_state.get('global_step')}")
```

#### Load Specific Checkpoint

```python
# Load a specific checkpoint by tag
load_path, client_state = model_engine.load_checkpoint(
    load_dir="/checkpoints/my_model",
    tag="global_step5000"
)
```

#### Load for Fine-Tuning (No Optimizer States)

```python
# Load model weights only, discard optimizer states
load_path, client_state = model_engine.load_checkpoint(
    load_dir="/checkpoints/pretrained_model",
    load_optimizer_states=False,
    load_lr_scheduler_states=False
)
```

#### Load with Non-Strict Matching

```python
# Allow missing or extra keys (useful when model architecture changed)
load_path, client_state = model_engine.load_checkpoint(
    load_dir="/checkpoints/my_model",
    load_module_strict=False
)
```

#### Load Module Only

```python
# Load only the model weights (no optimizer, no scheduler)
load_path, client_state = model_engine.load_checkpoint(
    load_dir="/checkpoints/my_model",
    load_module_only=True
)
```

---

## Universal Checkpoint Format

The Universal Checkpoint format is a standardized checkpoint format designed for cross-GPU-count loading. It allows loading a checkpoint saved with any number of GPUs onto a different number of GPUs.

### Why Universal Checkpoints Are Needed

DeepSpeed's native checkpoint format stores partitioned state dicts where each GPU's data is in a separate file. The partitioning is specific to the number of GPUs used during saving. If you save with 8 GPUs and try to load with 16 GPUs, the partition boundaries don't match.

The Universal Checkpoint format solves this by storing each parameter's data in a flat, non-partitioned format with metadata describing how to re-partition for any GPU count.

### Universal Checkpoint Structure

```
save_dir/global_stepXXXX/
    zero_checkpoint.json              # Metadata (parameter shapes, dtypes)
    mp_rank_00/
        model_states.pt               # Universal format model states
    mp_rank_00_model_states.json      # Per-parameter metadata
        # For each parameter:
        # {
        #     "param_name": "transformer.h.0.attn.c_attn.weight",
        #     "shape": [4096, 4096],
        #     "dtype": "torch.float16",
        #     "offset": 0,
        #     "length": 16777216
        # }
```

### ds_to_universal.py -- Conversion Tool

The `ds_to_universal.py` script converts DeepSpeed's native checkpoint format to the Universal format:

```bash
# Convert a single checkpoint
python -m deepspeed.checkpoint.ds_to_universal \
    --input_dir /checkpoints/my_model/global_step10000 \
    --output_dir /checkpoints/my_model_universal/global_step10000 \
    --max_samples 1000

# Convert the latest checkpoint
python -m deepspeed.checkpoint.ds_to_universal \
    --input_dir /checkpoints/my_model \
    --output_dir /checkpoints/my_model_universal
```

### Conversion Parameters

| Parameter | Description |
|---|---|
| `--input_dir` | Path to the DeepSpeed checkpoint directory |
| `--output_dir` | Path to write the Universal checkpoint |
| `--max_samples` | Maximum number of samples (for sampling during conversion) |
| `--num_extract_workers` | Number of parallel workers for extraction |
| `--num_merge_workers` | Number of parallel workers for merging |
| `--extract_only` | Only extract (do not merge) |
| `--merge_only` | Only merge (assume extraction is done) |

### Loading Universal Checkpoints

To load a Universal checkpoint, set `load_universal: true` in the checkpoint config:

```json
{
    "checkpoint": {
        "load_universal": true
    }
}
```

Then load normally:

```python
load_path, client_state = model_engine.load_checkpoint(
    load_dir="/checkpoints/my_model_universal"
)
```

### Cross-GPU-Count Loading

The primary use case for Universal checkpoints:

```bash
# Step 1: Train and save on 8 GPUs
deepspeed --num_gpus 8 train.py --save_steps 10000

# Step 2: Convert to Universal format
python -m deepspeed.checkpoint.ds_to_universal \
    --input_dir /checkpoints/model/global_step10000 \
    --output_dir /checkpoints/model_universal/global_step10000

# Step 3: Load on 32 GPUs for continued training
deepspeed --num_gpus 32 train.py \
    --load_checkpoint /checkpoints/model_universal/global_step10000
```

---

## ZeRO Checkpoint Format

### zero_checkpoint.py

The `zero_checkpoint.py` module provides utilities for working with ZeRO-specific checkpoint formats, including:

- **Reading ZeRO checkpoint metadata**: `zero_checkpoint.json` files that describe the partition layout.
- **Merging partitioned checkpoints**: Combining partitions from multiple GPUs into a single complete state dict.
- **Splitting checkpoints**: Dividing a complete state dict into partitions for a specific number of GPUs.

### ZeRO Checkpoint Metadata (zero_checkpoint.json)

```json
{
    "iteration": 10000,
    "version": 2.0,
    "num_steps": 10000,
    "global_batch_size": 4096,
    "consumed_train_samples": 40960000,
    "consumed_valid_samples": 0,
    "zero_stage": 3,
    "dp_world_size": 8,
    "mp_world_size": 1,
    "pp_world_size": 1,
    "param_shapes": {
        "transformer.h.0.attn.c_attn.weight": [4096, 4096],
        "transformer.h.0.attn.c_proj.weight": [4096, 4096],
        ...
    },
    "optimizer_name": "FusedAdam",
    "optimizer_params": {
        "lr": 6e-5,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.1
    }
}
```

### Merging ZeRO Checkpoints

```python
from deepspeed.checkpoint.zero_checkpoint import merge_pipeline_parallel_checkpoint

# Merge partitioned checkpoint into a single complete state dict
merged_state = merge_pipeline_parallel_checkpoint(
    checkpoint_dir="/checkpoints/model/global_step10000",
    output_dir="/checkpoints/model_merged"
)
```

---

## save_16bit_model()

The `save_16bit_model()` method saves the model in a standard 16-bit (FP16 or BF16) format that can be loaded by any PyTorch-based framework without DeepSpeed.

### Method Signature

```python
def save_16bit_model(self, save_dir, save_filename, client_state=None):
    """Save the model in standard FP16/BF16 format.

    Args:
        save_dir (str): Directory to save the model.
        save_filename (str): Filename for the saved model.
        client_state (dict, optional): Additional state to include.

    Returns:
        None

    Note:
        For ZeRO Stage 3, this requires gathering all parameters
        from across GPUs, which requires the
        stage3_gather_16bit_weights_on_model_save config option.
    """
```

### stage3_gather_16bit_weights_on_model_save

For ZeRO Stage 3, parameters are distributed across GPUs. To save a complete 16-bit model, all parameters must be gathered first. This is controlled by the `stage3_gather_16bit_weights_on_model_save` flag:

```json
{
    "zero_optimization": {
        "stage": 3,
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

**Warning**: Enabling this flag increases peak memory usage during checkpoint saving because all parameters are temporarily gathered on each GPU. For very large models, this may cause OOM.

### Usage Example

```python
# Save 16-bit model for deployment
model_engine.save_16bit_model(
    save_dir="/models/my_model",
    save_filename="model.pt"
)

# The resulting file is a standard PyTorch state dict
# that can be loaded without DeepSpeed:
import torch
state_dict = torch.load("/models/my_model/model.pt")
model.load_state_dict(state_dict)
```

### Saving for HuggingFace Transformers

```python
# Save model and tokenizer for HuggingFace
model_engine.save_16bit_model(
    save_dir="/models/hf_model",
    save_filename="pytorch_model.bin"
)
tokenizer.save_pretrained("/models/hf_model")

# Now the model can be loaded with:
# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained("/models/hf_model")
```

---

## Loading with Different GPU Counts

DeepSpeed supports loading checkpoints saved with one GPU count onto a different GPU count, with varying levels of support depending on the approach.

### Approach 1: Universal Checkpoint (Recommended)

The recommended approach for cross-GPU-count loading:

1. Convert the checkpoint to Universal format (see above).
2. Load with `load_universal: true`.
3. DeepSpeed automatically re-partitions the state for the new GPU count.

### Approach 2: Elastic Checkpoint Support

DeepSpeed provides elastic checkpoint support that allows loading checkpoints with mismatched GPU counts without conversion:

```json
{
    "elastic_checkpoint": {
        "enabled": true
    }
}
```

With elastic checkpoints:
- If the new GPU count has fewer GPUs than the original, multiple partitions are merged.
- If the new GPU count has more GPUs, partitions are split.
- This works for ZeRO Stage 2 and 3 checkpoints.

### Approach 3: Manual Resharding

For maximum control, you can manually reshard checkpoints:

```python
from deepspeed.checkpoint.zero_checkpoint import reshard_checkpoint

# Reshard a checkpoint from 8 GPUs to 16 GPUs
reshard_checkpoint(
    input_dir="/checkpoints/model_8gpu/global_step10000",
    output_dir="/checkpoints/model_16gpu/global_step10000",
    original_dp_world_size=8,
    new_dp_world_size=16
)
```

### Limitations

- **Pipeline parallelism**: Changing the pipeline-parallel degree requires re-partitioning the model across pipeline stages, which may not be supported for all model architectures.
- **Tensor parallelism**: Changing the tensor-parallel degree changes how individual layers are split and requires model-specific handling.
- **Data parallelism**: Changing the data-parallel degree (via ZeRO) is fully supported through Universal checkpoints or elastic checkpoints.

---

## Elastic Checkpoint Support

Elastic checkpointing enables training to continue with a different number of GPUs than what was used for saving. This is critical for:

- **Spot instance recovery**: When spot instances are preempted and replaced with a different number of GPUs.
- **Elastic scaling**: Dynamically adding or removing GPUs based on cluster availability.
- **Cost optimization**: Training with fewer GPUs during peak pricing and more GPUs during off-peak.

### Configuration

```json
{
    "elastic_checkpoint": {
        "enabled": true
    }
}
```

### How Elastic Checkpoints Work

1. **Save**: Checkpoint is saved normally with the current GPU count.
2. **Metadata**: The checkpoint includes metadata about the original GPU count and partition layout.
3. **Load**: When loading with a different GPU count:
   - DeepSpeed reads the original partition layout.
   - It re-maps the partitions to the new GPU count.
   - Missing partitions are gathered from other GPUs.
   - Extra partitions are split across the new GPUs.

### Example: Save with 8 GPUs, Load with 16 GPUs

```python
# Save on 8 GPUs
# deepspeed --num_gpus 8 train.py --save_steps 1000

# Load on 16 GPUs (elastic checkpoint handles the conversion)
# deepspeed --num_gpus 16 train.py --load_checkpoint /checkpoints/model

# In the training script:
load_path, client_state = model_engine.load_checkpoint(
    load_dir="/checkpoints/model"
)
```

---

## DataStates Async Checkpointing

DataStates is an asynchronous checkpointing system that overlaps checkpoint I/O with training computation, significantly reducing the wall-clock time spent on checkpointing.

### How Async Checkpointing Works

Without async checkpointing:
```
Training -> [BLOCK: Save checkpoint] -> Training -> ...
```

With async checkpointing:
```
Training -> [Start async save] -> Training -> [Training overlaps with save] -> ...
```

The save operation runs in a background thread while training continues.

### Configuration

```json
{
    "checkpoint": {
        "use_data_states": true,
        "data_states_config": {
            "backend": "local",
            "pin_memory": true
        }
    }
}
```

### Supported Backends

| Backend | Description |
|---|---|
| `"local"` | Saves to local storage (SSD/HDD). Fastest for single-node training. |
| `"shared"` | Saves to shared filesystem (NFS, Lustre, GPFS). Required for multi-node. |
| `"s3"` | Saves to Amazon S3 or compatible object storage. |

### Usage

```python
# Enable async checkpointing in config
ds_config = {
    "checkpoint": {
        "use_data_states": True,
        "data_states_config": {
            "backend": "local",
            "pin_memory": True
        }
    }
}

# Save checkpoint asynchronously
model_engine.save_checkpoint(
    save_dir="/checkpoints/my_model"
)
# Returns immediately; saving happens in the background

# The next save_checkpoint call will wait for the previous one to complete
# before starting a new one
```

### DataStates Implementation Details

```python
class DataStatesCheckpointManager:
    """Manages asynchronous checkpoint saving."""

    def __init__(self, backend, pin_memory=True):
        self.backend = backend
        self.pin_memory = pin_memory
        self.save_thread = None
        self.pending_save = None

    def save_async(self, state_dict, save_path):
        """Start an asynchronous save."""
        # Pin memory for faster CPU operations
        if self.pin_memory:
            pinned_state = {k: v.cpu().pin_memory() for k, v in state_dict.items()}
        else:
            pinned_state = {k: v.cpu() for k, v in state_dict.items()}

        # Start background save
        self.save_thread = threading.Thread(
            target=self._save_to_backend,
            args=(pinned_state, save_path)
        )
        self.save_thread.start()

    def wait_for_save(self):
        """Wait for the current save to complete."""
        if self.save_thread is not None:
            self.save_thread.join()
            self.save_thread = None

    def _save_to_backend(self, state_dict, save_path):
        """Save state dict to the backend."""
        torch.save(state_dict, save_path)
```

---

## Nebula Checkpoint System

Nebula is a high-performance, asynchronous checkpointing system integrated with DeepSpeed for use in cloud and distributed environments.

### Key Features

- **Asynchronous I/O**: Checkpoint saving and loading overlap with training.
- **Distributed storage**: Supports saving to distributed storage backends (S3, GCS, Azure Blob).
- **Checkpoint versioning**: Maintains multiple checkpoint versions with automatic cleanup.
- **Fault tolerance**: Detects and recovers from corrupted checkpoints.

### Configuration

```json
{
    "nebula": {
        "enabled": true,
        "persistent_storage_path": "/nebula/checkpoints",
        "persistent_time_interval": 100,
        "persistent_timeout": 300,
        "num_of_version_retention": 3,
        "enable_nebula_load": true
    }
}
```

### Nebula Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Enable Nebula checkpoint system |
| `persistent_storage_path` | str | - | Base path for persistent storage |
| `persistent_time_interval` | int | `100` | Minimum seconds between checkpoint saves |
| `persistent_timeout` | int | `300` | Maximum seconds to wait for checkpoint save |
| `num_of_version_retention` | int | `2` | Number of checkpoint versions to keep |
| `enable_nebula_load` | bool | `false` | Enable loading from Nebula storage |

### Usage

```python
# Nebula is configured via the DeepSpeed config
ds_config = {
    "nebula": {
        "enabled": True,
        "persistent_storage_path": "/shared/checkpoints",
        "num_of_version_retention": 5
    }
}

# Save checkpoints normally - Nebula handles async and storage
model_engine.save_checkpoint(save_dir="/checkpoints/my_model")

# Load from Nebula storage
load_path, client_state = model_engine.load_checkpoint(
    load_dir="/checkpoints/my_model"
)
```

---

## Configuration Examples

### Example 1: Basic Checkpointing (Save Every N Steps)

```python
# In training loop
if global_step % save_steps == 0:
    model_engine.save_checkpoint(save_dir=args.output_dir)
```

```json
{
    "checkpoint": {
        "tag_validation": "Ignore"
    }
}
```

### Example 2: Checkpointing with Client State

```python
# Save with full training state
client_state = {
    "epoch": epoch,
    "global_step": global_step,
    "consumed_samples": global_step * train_batch_size,
    "best_eval_loss": best_eval_loss,
    "random_states": {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state()
    }
}
model_engine.save_checkpoint(
    save_dir=args.output_dir,
    client_state=client_state
)

# Load and restore state
load_path, client_state = model_engine.load_checkpoint(
    load_dir=args.output_dir
)
start_epoch = client_state.get("epoch", 0) + 1
best_eval_loss = client_state.get("best_eval_loss", float("inf"))
```

### Example 3: ZeRO-3 Checkpointing with 16-bit Model Export

```json
{
    "zero_optimization": {
        "stage": 3,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "checkpoint": {
        "tag_validation": "Warn"
    }
}
```

```python
# During training
if global_step % save_steps == 0:
    # Save full DeepSpeed checkpoint
    model_engine.save_checkpoint(save_dir=args.output_dir)

# At the end of training
# Export 16-bit model for deployment
model_engine.save_16bit_model(
    save_dir=args.output_dir,
    save_filename="pytorch_model.bin"
)
```

### Example 4: Universal Checkpoint for Cross-GPU Loading

```json
{
    "checkpoint": {
        "load_universal": true
    }
}
```

```bash
# Convert checkpoint to universal format
python -m deepspeed.checkpoint.ds_to_universal \
    --input_dir /checkpoints/model/global_step10000 \
    --output_dir /checkpoints/model_universal/global_step10000

# Load on different GPU count
deepspeed --num_gpus 16 train.py \
    --load_checkpoint /checkpoints/model_universal
```

### Example 5: Elastic Checkpointing

```json
{
    "elastic_checkpoint": {
        "enabled": true
    }
}
```

```python
# Training can be resumed with any number of GPUs
load_path, client_state = model_engine.load_checkpoint(
    load_dir=args.load_checkpoint
)
```

### Example 6: Node-Local Storage for Fast Checkpointing

```json
{
    "checkpoint": {
        "use_node_local_storage": true
    }
}
```

```python
# Each node saves to its local SSD
model_engine.save_checkpoint(save_dir="/local_ssd/checkpoints")

# For loading, ensure all nodes have the checkpoint data
# (typically via a shared setup script)
```

### Example 7: Async Checkpointing with DataStates

```json
{
    "checkpoint": {
        "use_data_states": true,
        "data_states_config": {
            "backend": "local",
            "pin_memory": true
        }
    }
}
```

### Example 8: Full Production Configuration

```json
{
    "train_batch_size": 4096,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 16,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 6e-5,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 3,
        "stage3_gather_16bit_weights_on_model_save": true,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    },
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": true,
        "contiguous_memory_optimization": true,
        "number_checkpoints": 32
    },
    "checkpoint": {
        "tag_validation": "Warn",
        "load_universal": false
    },
    "elastic_checkpoint": {
        "enabled": true
    }
}
```

```python
# Training script
import deepspeed

def main():
    # Initialize
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config
    )

    # Load checkpoint if resuming
    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(
            load_dir=args.checkpoint_dir
        )
        global_step = client_state.get("global_step", 0)
    else:
        global_step = 0

    # Training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            outputs = model_engine(batch)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()
            global_step += 1

            # Periodic checkpointing
            if global_step % args.save_steps == 0:
                client_state = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "consumed_samples": global_step * ds_config["train_batch_size"],
                    "best_eval_loss": best_eval_loss,
                }
                model_engine.save_checkpoint(
                    save_dir=args.checkpoint_dir,
                    client_state=client_state
                )

    # Export final 16-bit model
    model_engine.save_16bit_model(
        save_dir=args.output_dir,
        save_filename="pytorch_model.bin"
    )
```

---

## Checkpoint Utilities

### Listing Available Checkpoints

```python
import os

def list_checkpoints(checkpoint_dir):
    """List all available checkpoint tags in a directory."""
    checkpoints = []
    for entry in os.listdir(checkpoint_dir):
        path = os.path.join(checkpoint_dir, entry)
        if os.path.isdir(path) and entry.startswith("global_step"):
            checkpoints.append(entry)
    return sorted(checkpoints, key=lambda x: int(x.split("global_step")[1]))

# Usage
checkpoints = list_checkpoints("/checkpoints/my_model")
print(f"Available checkpoints: {checkpoints}")
```

### Getting the Latest Checkpoint

```python
def get_latest_checkpoint(checkpoint_dir):
    """Get the path to the latest checkpoint."""
    latest_file = os.path.join(checkpoint_dir, "latest")
    if os.path.exists(latest_file):
        with open(latest_file, "r") as f:
            tag = f.read().strip()
        return os.path.join(checkpoint_dir, tag)
    return None

# Usage
latest = get_latest_checkpoint("/checkpoints/my_model")
print(f"Latest checkpoint: {latest}")
```

### Deleting Old Checkpoints

```python
import shutil

def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=3):
    """Remove old checkpoints, keeping only the last N."""
    checkpoints = list_checkpoints(checkpoint_dir)
    if len(checkpoints) > keep_last_n:
        for checkpoint in checkpoints[:-keep_last_n]:
            path = os.path.join(checkpoint_dir, checkpoint)
            shutil.rmtree(path)
            print(f"Deleted: {path}")

# Usage
cleanup_old_checkpoints("/checkpoints/my_model", keep_last_n=3)
```

---

## Best Practices

1. **Save checkpoints frequently**: For long training runs, save every 1000-5000 steps. The overhead is small (especially with async checkpointing) compared to the risk of losing hours of training.

2. **Use Universal checkpoints for flexibility**: Convert checkpoints to Universal format to enable loading with any GPU count. This is essential for elastic training and spot instance recovery.

3. **Enable elastic checkpoints for cloud training**: When training on spot instances, elastic checkpoints ensure you can always resume regardless of how many GPUs are available after preemption.

4. **Use `stage3_gather_16bit_weights_on_model_save` for final exports**: Only enable this when you need to export a deployable model, not for every intermediate checkpoint. The gathering operation increases peak memory.

5. **Save client state comprehensively**: Include all information needed to resume training exactly: RNG states, epoch number, global step, consumed samples, and any best-metric tracking.

6. **Use node-local storage for large models**: When checkpointing models with billions of parameters, writing to local SSD is much faster than shared filesystem. Aggregate to shared storage asynchronously.

7. **Validate checkpoints after saving**: After saving, verify the checkpoint can be loaded by reading the metadata file. This catches corruption early.

8. **Keep multiple checkpoint versions**: Maintain at least 2-3 recent checkpoints in case the latest one is corrupted. Use `num_of_version_retention` with Nebula or implement manual cleanup.

9. **Test checkpoint loading separately**: Before relying on checkpoints for a long training run, test that loading works correctly by saving and loading a single step.

10. **Monitor checkpoint I/O performance**: Checkpoint saving should not become a bottleneck. If it does, consider async checkpointing, node-local storage, or reducing checkpoint frequency.
