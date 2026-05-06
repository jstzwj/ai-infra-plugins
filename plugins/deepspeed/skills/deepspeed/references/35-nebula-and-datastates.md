# Nebula and DataStates

## Overview

DeepSpeed provides two advanced checkpoint management systems -- Nebula and DataStates -- that address the critical challenge of saving and loading massive model checkpoints efficiently. As model sizes grow from billions to trillions of parameters, checkpoint files can reach hundreds of gigabytes or terabytes, and saving them using traditional synchronous methods can consume a significant fraction of total training time (often 10-30% for large models). Both Nebula and DataStates provide asynchronous checkpointing solutions that overlap checkpoint I/O with training computation, but they differ in architecture and implementation.

**Nebula** is a lightweight, user-space asynchronous checkpointing library that provides transparent checkpoint persistence to external storage systems. It manages checkpoint versioning, retention policies, and automatic failover. Nebula is designed as a drop-in replacement for DeepSpeed's standard checkpoint saving, requiring minimal code changes.

**DataStates** is a more recent approach that provides asynchronous checkpointing through a different architectural pattern, focusing on data state management and multi-tier checkpoint storage.

---

## Source Code Organization

```
deepspeed/nebula/
    __init__.py
    config.py                     # DeepSpeedNebulaConfig class
    constants.py                  # Nebula configuration constants

deepspeed/datastates/
    __init__.py
    config.py                     # DeepSpeedDataStatesConfig class
```

---

## Nebula

### Overview

Nebula is DeepSpeed's primary asynchronous checkpointing solution. It decouples the checkpoint save operation from the training loop by maintaining a persistent background process that handles checkpoint writing to durable storage. This allows training to continue immediately after the checkpoint state is captured in memory, while the actual I/O to persistent storage happens asynchronously in the background.

Key features:
- **Asynchronous persistence**: Checkpoint data is first written to local storage (fast), then asynchronously persisted to the configured storage path.
- **Version management**: Maintains multiple checkpoint versions with configurable retention policies.
- **Automatic cleanup**: Old checkpoints are automatically cleaned up based on retention settings.
- **Transparent integration**: Works as a drop-in enhancement to DeepSpeed's standard `save_checkpoint()` and `load_checkpoint()` methods.
- **Failover support**: If a checkpoint write fails, Nebula maintains the previous valid checkpoint and can load from it.

### DeepSpeedNebulaConfig (config.py)

```python
# deepspeed/nebula/config.py

class DeepSpeedNebulaConfig:
    """Configuration for DeepSpeed Nebula asynchronous checkpointing.

    Attributes:
        enabled (bool): Whether Nebula checkpointing is enabled.
        persistent_storage_path (str): Path to persistent storage for
            checkpoint persistence. This should be a durable storage
            location (e.g., network filesystem, cloud storage mount).
        persistent_time_interval (int): Time interval in seconds between
            periodic persistence attempts. Nebula will attempt to persist
            checkpoints at this interval.
        num_of_version_in_retention (int): Number of checkpoint versions
            to retain in persistent storage. Older versions are automatically
            cleaned up.
        enable_nebula_load (bool): Whether to use Nebula for loading
            checkpoints (in addition to saving).
        load_path (str): Path to load checkpoints from. Can differ from
            persistent_storage_path (e.g., loading from a different
            storage location).
    """

    def __init__(self, nebula_config_dict):
        """Initialize Nebula configuration from a dictionary.

        Args:
            nebula_config_dict (dict): Nebula configuration dictionary
                from the DeepSpeed JSON config.
        """
        self.enabled = nebula_config_dict.get(NEBULA_ENABLED, False)
        self.persistent_storage_path = nebula_config_dict.get(
            NEBULA_PERSISTENT_STORAGE_PATH, ""
        )
        self.persistent_time_interval = nebula_config_dict.get(
            NEBULA_PERSISTENT_TIME_INTERVAL, 100
        )
        self.num_of_version_in_retention = nebula_config_dict.get(
            NEBULA_NUM_OF_VERSION_IN_RETENTION, 2
        )
        self.enable_nebula_load = nebula_config_dict.get(
            NEBULA_ENABLE_NEBULA_LOAD, True
        )
        self.load_path = nebula_config_dict.get(NEBULA_LOAD_PATH, "")

    def validated(self):
        """Validate the Nebula configuration.

        Returns:
            bool: True if the configuration is valid.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        if not self.enabled:
            return True
        if not self.persistent_storage_path:
            raise ValueError(
                "Nebula requires 'persistent_storage_path' when enabled."
            )
        return True
```

### Nebula Constants (constants.py)

```python
# deepspeed/nebula/constants.py

# Top-level configuration key
NEBULA = "nebula"

# Individual parameter keys
NEBULA_ENABLED = "enabled"
NEBULA_PERSISTENT_STORAGE_PATH = "persistent_storage_path"
NEBULA_PERSISTENT_TIME_INTERVAL = "persistent_time_interval"
NEBULA_NUM_OF_VERSION_IN_RETENTION = "num_of_version_in_retention"
NEBULA_ENABLE_NEBULA_LOAD = "enable_nebula_load"
NEBULA_LOAD_PATH = "load_path"
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Whether to enable Nebula asynchronous checkpointing. When `false`, standard synchronous checkpointing is used. |
| `persistent_storage_path` | str | `""` | **Required when enabled**. Path to the persistent storage directory where checkpoints will be asynchronously persisted. This should be a durable storage location such as a network filesystem mount (NFS, Lustre, GPFS), cloud storage mount (S3 FUSE, GCS FUSE), or a replicated local path. The directory must exist and be writable by the training process. |
| `persistent_time_interval` | int | `100` | Time interval in seconds between periodic checkpoint persistence attempts. Nebula checks if there are unpersisted checkpoints at this interval and writes them to `persistent_storage_path`. Lower values provide faster persistence but increase I/O load. Higher values reduce I/O load but increase the risk of data loss on failure. |
| `num_of_version_in_retention` | int | `2` | Number of checkpoint versions to retain in persistent storage. When a new checkpoint is persisted, Nebula automatically deletes the oldest version if the count exceeds this value. Set to a higher value (e.g., 5-10) for more rollback options at the cost of storage space. |
| `enable_nebula_load` | bool | `true` | Whether to use Nebula for loading checkpoints. When `true`, `load_checkpoint()` will look for checkpoints in the Nebula-managed storage path. When `false`, standard loading is used (useful for loading from a non-Nebula checkpoint). |
| `load_path` | str | `""` | Explicit path to load checkpoints from. If specified, this overrides `persistent_storage_path` for loading. Useful when loading from a different storage location than where new checkpoints are being saved (e.g., loading a baseline checkpoint from shared storage while saving to local NVMe). |

### Nebula Configuration Example

```json
{
    "nebula": {
        "enabled": true,
        "persistent_storage_path": "/shared/checkpoints/nebula",
        "persistent_time_interval": 100,
        "num_of_version_in_retention": 3,
        "enable_nebula_load": true,
        "load_path": "/shared/checkpoints/baseline"
    }
}
```

### How Nebula Works

The Nebula checkpointing flow operates in three stages:

**Stage 1: Fast Local Save (Synchronous)**
When `save_checkpoint()` is called, Nebula writes the checkpoint to local storage (typically local NVMe or RAM disk) using the standard DeepSpeed checkpoint format. This is fast because it uses local I/O and completes quickly.

```
Training Step N
  -> save_checkpoint() called
    -> Write to local NVMe: /local_nvme/checkpoints/global_step1000/
    -> Return immediately (training continues)
```

**Stage 2: Asynchronous Persistence (Background)**
A background process periodically checks for unpersisted checkpoints in local storage and copies them to the `persistent_storage_path`. This happens concurrently with training.

```
Background Thread (every persistent_time_interval seconds):
  -> Check /local_nvme/checkpoints/ for unpersisted checkpoints
  -> Copy /local_nvme/checkpoints/global_step1000/ -> /shared/checkpoints/nebula/global_step1000/
  -> Mark as persisted
```

**Stage 3: Version Cleanup (Background)**
After successful persistence, Nebula checks the number of retained versions and deletes the oldest if it exceeds `num_of_version_in_retention`.

```
Background Thread (after persistence):
  -> List checkpoints in /shared/checkpoints/nebula/
  -> Found: global_step500, global_step750, global_step1000
  -> num_of_version_in_retention = 2
  -> Delete: global_step500 (oldest)
  -> Retained: global_step750, global_step1000
```

### Using Nebula with DeepSpeed

```python
import deepspeed

# DeepSpeed configuration with Nebula
ds_config = {
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999]
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "nebula": {
        "enabled": true,
        "persistent_storage_path": "/shared/nfs/checkpoints/nebula",
        "persistent_time_interval": 60,
        "num_of_version_in_retention": 5,
        "enable_nebula_load": true,
        "load_path": ""
    }
}

# Initialize DeepSpeed
model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
    model_parameters=model.parameters(),
)

# Training loop
for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        loss = model_engine(batch)
        model_engine.backward(loss)
        model_engine.step()

        # Save checkpoint (fast local save + async persistence)
        if step % save_interval == 0:
            # This returns quickly because Nebula writes locally first
            model_engine.save_checkpoint(
                save_dir="/local_nvme/checkpoints",
                tag=f"global_step{step}",
                client_state={"epoch": epoch, "step": step}
            )
```

### Loading from Nebula Checkpoints

```python
# Load from Nebula-managed checkpoint
# If enable_nebula_load is true, load_checkpoint automatically
# looks in the Nebula persistent storage path
_, client_state = model_engine.load_checkpoint(
    load_dir="/local_nvme/checkpoints",
    tag="global_step5000",
)

print(f"Resumed from epoch {client_state['epoch']}, step {client_state['step']}")
```

### Nebula with Multiple Checkpoint Strategies

```python
# Combine Nebula with node-local storage for maximum performance
ds_config = {
    "checkpoint": {
        "use_node_local_storage": true  # Save to local NVMe on each node
    },
    "nebula": {
        "enabled": true,
        "persistent_storage_path": "/shared/nfs/checkpoints",  # Async to NFS
        "persistent_time_interval": 30,
        "num_of_version_in_retention": 3
    }
}

# The checkpoint flow becomes:
# 1. save_checkpoint() -> local NVMe on each node (fast, synchronous)
# 2. Nebula -> each node persists to shared NFS (async, background)
# 3. Training continues immediately after step 1
```

---

## DataStates

### Overview

DataStates is DeepSpeed's newer asynchronous checkpointing system that provides an alternative architecture to Nebula. While Nebula focuses on background persistence of locally-saved checkpoints, DataStates takes a more integrated approach to data state management, handling the capture, serialization, and storage of training state as a unified operation.

### DeepSpeedDataStatesConfig (config.py)

```python
# deepspeed/datastates/config.py

class DeepSpeedDataStatesConfig:
    """Configuration for DeepSpeed DataStates asynchronous checkpointing.

    Attributes:
        enabled (bool): Whether DataStates checkpointing is enabled.
        config (dict): Additional configuration options for DataStates.
    """

    def __init__(self, datastates_config_dict):
        """Initialize DataStates configuration.

        Args:
            datastates_config_dict (dict): DataStates configuration
                dictionary from the DeepSpeed JSON config.
        """
        self.enabled = datastates_config_dict.get("enabled", False)
        self.config = datastates_config_dict.get("config", {})

    def validated(self):
        """Validate the DataStates configuration.

        Returns:
            bool: True if configuration is valid.
        """
        if not self.enabled:
            return True
        # Additional validation as needed
        return True
```

### Configuration

```json
{
    "datastates": {
        "enabled": true,
        "config": {
            "save_path": "/shared/checkpoints/datastates",
            "async_save": true,
            "pin_memory": true
        }
    }
}
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Whether DataStates checkpointing is enabled. |
| `config` | dict | `{}` | Additional configuration options passed to the DataStates backend. |
| `config.save_path` | str | - | Path for saving DataStates checkpoints. |
| `config.async_save` | bool | `true` | Whether to use asynchronous saving. |
| `config.pin_memory` | bool | `true` | Whether to use pinned memory for checkpoint buffers. |

### Using DataStates

```python
import deepspeed

ds_config = {
    "train_batch_size": 512,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999]
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "datastates": {
        "enabled": true,
        "config": {
            "save_path": "/shared/checkpoints/datastates",
            "async_save": true,
            "pin_memory": true
        }
    }
}

model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
    model_parameters=model.parameters(),
)

# Training loop
for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        loss = model_engine(batch)
        model_engine.backward(loss)
        model_engine.step()

        if step % save_interval == 0:
            model_engine.save_checkpoint(
                save_dir="/local_nvme/checkpoints",
                tag=f"global_step{step}",
            )
```

---

## Asynchronous Checkpointing Comparison

### Nebula vs DataStates

| Feature | Nebula | DataStates |
|---------|--------|------------|
| **Architecture** | Background thread persists locally-saved checkpoints | Integrated async save with pinned buffers |
| **Checkpoint format** | Standard DeepSpeed format | Standard DeepSpeed format |
| **Version management** | Built-in (num_of_version_in_retention) | Manual or external |
| **Load support** | Built-in (enable_nebula_load) | Standard load |
| **Storage backend** | Filesystem (any mount point) | Filesystem |
| **Memory overhead** | Minimal (reads from local disk) | Higher (pinned buffers for async) |
| **Maturity** | Production-ready, widely used | Newer, fewer deployments |
| **Best for** | Long-running training with frequent checkpointing | Scenarios requiring minimal checkpoint latency |

### When to Use Each

**Use Nebula when:**
- Training runs are long (hours to days) and checkpoints are frequent
- You need automatic version management and retention
- You want a battle-tested, production-ready solution
- You need to persist to network storage (NFS, Lustre) without blocking training

**Use DataStates when:**
- You need the absolute lowest checkpoint latency
- Your checkpoint sizes are very large and you want to use pinned memory for fast serialization
- You prefer an integrated async save without a separate persistence thread

---

## Advanced Configuration Examples

### Example 1: Nebula with ZeRO-3 and NVMe Offloading

```json
{
    "train_batch_size": 512,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 16,
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
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_gather_16bit_weights_on_model_save": true,
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme/params"
        },
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme/optimizer"
        }
    },
    "checkpoint": {
        "use_node_local_storage": true
    },
    "nebula": {
        "enabled": true,
        "persistent_storage_path": "/shared/nfs/model_checkpoints",
        "persistent_time_interval": 60,
        "num_of_version_in_retention": 5,
        "enable_nebula_load": true,
        "load_path": "/shared/nfs/model_checkpoints"
    }
}
```

### Example 2: Nebula with Pipeline Parallelism

```json
{
    "train_batch_size": 1024,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-5,
            "betas": [0.9, 0.999]
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 1
    },
    "pipeline": {
        "enabled": true,
        "parallel_size": 4,
        "micro_batches": 8
    },
    "nebula": {
        "enabled": true,
        "persistent_storage_path": "/lustre/checkpoints/nebula",
        "persistent_time_interval": 120,
        "num_of_version_in_retention": 3,
        "enable_nebula_load": true
    }
}
```

### Example 3: DataStates for Fast Checkpointing

```json
{
    "train_batch_size": 2048,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 6e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 0.1
        }
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        },
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "datastates": {
        "enabled": true,
        "config": {
            "save_path": "/shared/checkpoints/datastates",
            "async_save": true,
            "pin_memory": true
        }
    }
}
```

### Example 4: Nebula with Tensor and Pipeline Parallelism

```json
{
    "train_batch_size": 2048,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 16,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1.5e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 1,
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "tensor_pipeline": {
        "enabled": true,
        "tp_size": 4,
        "pp_size": 2
    },
    "nebula": {
        "enabled": true,
        "persistent_storage_path": "/gpfs/checkpoints/nebula",
        "persistent_time_interval": 90,
        "num_of_version_in_retention": 4,
        "enable_nebula_load": true,
        "load_path": "/gpfs/checkpoints/nebula"
    },
    "gradient_clipping": 1.0
}
```

---

## Data State Management

### What Gets Checkpointed

Both Nebula and DataStates capture the complete training state:

1. **Model parameters**: The model weights (FP16 and/or FP32 master copies).
2. **Optimizer states**: Momentum, variance, and any other optimizer-specific state.
3. **Learning rate scheduler state**: Current learning rate, step count, warmup progress.
4. **Random number generator states**: Python random, NumPy random, PyTorch RNG (CPU and CUDA).
5. **Data loader state**: Current position in the dataset, sampler state.
6. **Gradient accumulation state**: Accumulated gradients for micro-batches.
7. **User-defined state**: Any additional state passed via `client_state` in `save_checkpoint()`.

### Checkpoint State Categories

```python
# DeepSpeed checkpoint state structure
checkpoint_state = {
    # Model state
    "module": model_state_dict,           # All model parameters
    "fp32_flat_params": fp32_params,      # FP32 master weights (if FP16)

    # Optimizer state
    "optimizer_state_dict": {
        "step": global_step,
        "param_groups": optimizer.param_groups,
        "state": optimizer_state,         # momentum, variance per param
    },

    # LR scheduler state
    "lr_scheduler": scheduler.state_dict(),

    # RNG states
    "random_rng_state": python_rng_state,
    "numpy_rng_state": numpy_rng_state,
    "torch_rng_state": torch_rng_state,
    "cuda_rng_state": cuda_rng_states,

    # DeepSpeed internal state
    "global_steps": global_step,
    "global_samples": global_samples,
    "ds_version": deepspeed.__version__,
    "ds_config": ds_config_dict,

    # User state (passed via client_state)
    "client_state": {
        "epoch": current_epoch,
        "step": current_step,
        "best_accuracy": best_accuracy,
        # Any user-defined state
    },
}
```

### State Restoration on Load

```python
# Loading restores all state categories
_, client_state = model_engine.load_checkpoint(load_dir, tag=None)

# The following are automatically restored:
# - Model parameters (weights)
# - Optimizer state (momentum, variance, step)
# - Learning rate scheduler state
# - RNG states (for reproducibility)
# - Gradient accumulation buffers
# - DeepSpeed engine state (global step, etc.)

# User state is returned in client_state
epoch = client_state.get("epoch", 0)
step = client_state.get("step", 0)
best_acc = client_state.get("best_accuracy", 0.0)
```

---

## Performance Analysis

### Checkpointing Overhead Comparison

For a 175B parameter model with ZeRO Stage 3 on 64 GPUs:

| Method | Checkpoint Size (per GPU) | Save Time | Blocking Time | Total Overhead |
|--------|--------------------------|-----------|---------------|----------------|
| Standard synchronous | ~10 GB | 120s | 120s | 100% blocking |
| + node-local storage | ~10 GB | 45s | 45s | 37.5% blocking |
| + Nebula (async) | ~10 GB | 45s (local) + async (NFS) | 5s | 4.2% blocking |
| + DataStates (async) | ~10 GB | 30s (async) | 3s | 2.5% blocking |

*Assumptions: NVMe write speed = 3 GB/s, NFS write speed = 500 MB/s, 1 hour between checkpoints.*

### Nebula Persistence Timing

```
Timeline (save_interval = 3600s, persistent_time_interval = 60s):

T=0s:     save_checkpoint() -> local NVMe write (45s) -> return
T=45s:    Training resumes
T=60s:    Nebula background: check for unpersisted -> found 1 -> start copy to NFS
T=300s:   Nebula background: copy complete -> checkpoint persisted
T=3600s:  save_checkpoint() -> local NVMe write (45s) -> return
T=3650s:  Training resumes
T=3720s:  Nebula background: persist new checkpoint, clean up oldest

Effective training overhead: 45s / 3600s = 1.25% (vs 100% blocking for synchronous NFS)
```

### Storage Requirements

| num_of_version_in_retention | Model Size | Storage per GPU | Total Storage (64 GPUs) |
|----------------------------|------------|----------------|------------------------|
| 2 | 175B | ~20 GB | ~1.3 TB |
| 3 | 175B | ~30 GB | ~1.9 TB |
| 5 | 175B | ~50 GB | ~3.2 TB |
| 2 | 530B | ~60 GB | ~3.8 TB |
| 3 | 530B | ~90 GB | ~5.8 TB |

---

## Troubleshooting

### Common Nebula Issues

1. **"Nebula persistent_storage_path not found"**: Ensure the directory exists and is writable. Create it before starting training: `mkdir -p /shared/checkpoints/nebula && chmod 777 /shared/checkpoints/nebula`.

2. **Checkpoint persistence falling behind**: If checkpoints are being created faster than Nebula can persist them, increase `persistent_time_interval` to reduce I/O contention, or use faster persistent storage.

3. **Nebula checkpoint version mismatch**: If you change model architecture between runs, old Nebula checkpoints may not load correctly. Set `enable_nebula_load: false` and use `load_path` to specify the correct checkpoint.

4. **Disk space exhaustion**: Reduce `num_of_version_in_retention` to limit storage usage. Monitor disk usage with `du -sh /shared/checkpoints/nebula/`.

5. **Nebula not cleaning up old checkpoints**: Ensure the persistent storage path is writable and that the training process has permission to delete files in the directory.

### Common DataStates Issues

1. **High memory usage with pinned buffers**: DataStates uses pinned host memory for async checkpointing. Reduce batch size or disable `pin_memory` if host memory is limited.

2. **Checkpoint not fully written on crash**: DataStates async writes may be in progress when a crash occurs. Use the `latest` file to identify the last complete checkpoint.

3. **Loading a DataStates checkpoint without DataStates**: If DataStates is disabled in the load config, the checkpoint can still be loaded using standard DeepSpeed loading, as the on-disk format is the same.

### Debugging Tips

```bash
# Check Nebula checkpoint status
ls -la /shared/checkpoints/nebula/
# Should show: global_stepXXXX/ directories

# Check if checkpoint is fully persisted
# Nebula writes a .persisted marker file when persistence is complete
find /shared/checkpoints/nebula/ -name ".persisted" -exec cat {} \;

# Monitor Nebula persistence progress
watch -n 5 'du -sh /shared/checkpoints/nebula/*'

# Check local checkpoint directory
ls -la /local_nvme/checkpoints/

# Verify checkpoint integrity
python -c "
import torch
import os
checkpoint_path = '/shared/checkpoints/nebula/global_step1000'
files = os.listdir(checkpoint_path)
print(f'Files in checkpoint: {files}')
for f in files:
    if f.endswith('.pt') or f.endswith('.sd'):
        data = torch.load(os.path.join(checkpoint_path, f), map_location='cpu')
        print(f'{f}: keys={list(data.keys()) if isinstance(data, dict) else type(data)}')
"
```

### Environment Variables

```bash
# Enable Nebula debug logging
export DS_NEBULA_DEBUG=1

# Disable Nebula (override config)
export DS_DISABLE_NEBULA=1

# Disable DataStates (override config)
export DS_DISABLE_DATASTATES=1

# Set Nebula log level
export DS_NEBULA_LOG_LEVEL=DEBUG
```

---

## Programmatic API

### Nebula-Specific Operations

```python
import deepspeed
from deepspeed.nebula.config import DeepSpeedNebulaConfig

# Check Nebula configuration programmatically
nebula_config = DeepSpeedNebulaConfig({
    "enabled": True,
    "persistent_storage_path": "/shared/checkpoints",
    "persistent_time_interval": 60,
    "num_of_version_in_retention": 3,
})

print(f"Nebula enabled: {nebula_config.enabled}")
print(f"Storage path: {nebula_config.persistent_storage_path}")
print(f"Time interval: {nebula_config.persistent_time_interval}s")
print(f"Versions retained: {nebula_config.num_of_version_in_retention}")

# Validate configuration
nebula_config.validated()  # Raises ValueError if invalid
```

### DataStates-Specific Operations

```python
from deepspeed.datastates.config import DeepSpeedDataStatesConfig

# Check DataStates configuration programmatically
datastates_config = DeepSpeedDataStatesConfig({
    "enabled": True,
    "config": {
        "save_path": "/shared/checkpoints/datastates",
        "async_save": True,
        "pin_memory": True
    }
})

print(f"DataStates enabled: {datastates_config.enabled}")
print(f"DataStates config: {datastates_config.config}")

# Validate configuration
datastates_config.validated()
```

### Checking Active Checkpoint Backend

```python
import deepspeed

def get_active_checkpoint_backend(ds_config):
    """Determine which checkpoint backend is active."""
    if ds_config.get("nebula", {}).get("enabled", False):
        return "nebula"
    elif ds_config.get("datastates", {}).get("enabled", False):
        return "datastates"
    else:
        return "standard_synchronous"

# Usage
ds_config = {
    "nebula": {
        "enabled": True,
        "persistent_storage_path": "/shared/checkpoints"
    }
}
backend = get_active_checkpoint_backend(ds_config)
print(f"Active checkpoint backend: {backend}")  # "nebula"
```

---

## Best Practices

### Checkpoint Frequency

| Training Duration | Recommended Frequency | Nebula Retention | Rationale |
|-------------------|-----------------------|------------------|-----------|
| < 1 hour | Every epoch or at end | 2 | Short runs, minimal risk |
| 1-6 hours | Every 30-60 minutes | 3 | Moderate risk, manageable storage |
| 6-24 hours | Every 30 minutes | 5 | Significant investment, need rollback |
| > 24 hours | Every 15-30 minutes | 5-10 | Large investment, frequent rollback may be needed |

### Storage Path Selection

- **Local NVMe**: Best for fast local saves. Pair with Nebula for async persistence to shared storage.
- **Shared NFS**: Good for persistence but slow for direct saves. Always use with Nebula or DataStates async.
- **Lustre/GPFS**: High-throughput shared filesystem. Can handle direct saves but still benefits from async.
- **Cloud storage (S3, GCS)**: Use via FUSE mount with Nebula. High latency makes async essential.

### Fault Tolerance Strategy

```python
# Robust checkpoint loading with fallback
def load_checkpoint_robustly(model_engine, primary_path, fallback_path=None):
    """Try loading from primary path, fall back to secondary."""
    try:
        _, client_state = model_engine.load_checkpoint(primary_path)
        print(f"Loaded checkpoint from {primary_path}")
        return client_state
    except Exception as e:
        print(f"Failed to load from {primary_path}: {e}")
        if fallback_path:
            try:
                _, client_state = model_engine.load_checkpoint(fallback_path)
                print(f"Loaded fallback checkpoint from {fallback_path}")
                return client_state
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
        print("Starting from scratch.")
        return None
```

---

## Summary

DeepSpeed provides two complementary asynchronous checkpointing systems. **Nebula** is the more mature and feature-rich option, providing background persistence of locally-saved checkpoints with automatic version management, retention policies, and transparent integration with `save_checkpoint()` and `load_checkpoint()`. It is configured through the `nebula` section of the DeepSpeed JSON config, with key parameters including `persistent_storage_path` (where checkpoints are asynchronously copied), `persistent_time_interval` (how often persistence is attempted), and `num_of_version_in_retention` (how many old checkpoints to keep). **DataStates** is a newer alternative that uses pinned memory buffers for integrated async saving, offering potentially lower checkpoint latency at the cost of higher memory usage. Both systems dramatically reduce the training overhead of checkpointing -- from 10-30% blocking time with synchronous saves to under 5% with asynchronous persistence -- enabling more efficient utilization of expensive GPU resources during long training runs.
