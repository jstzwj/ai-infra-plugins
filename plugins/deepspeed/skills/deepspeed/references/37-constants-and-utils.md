# Constants and Utility Functions

## Overview

DeepSpeed organizes its constants and utility functions across multiple modules. This reference covers all constants from `deepspeed/constants.py` and `deepspeed/runtime/constants.py`, the optimizer registry, the DtypeEnum system, and all utility functions exported from `deepspeed/utils/`.

---

## Top-Level Constants (deepspeed/constants.py)

These constants are defined at the package level and control fundamental distributed training behavior.

### Network and Communication Constants

| Constant | Value | Description |
|---|---|---|
| `TORCH_DISTRIBUTED_DEFAULT_PORT` | `29500` | Default port for PyTorch distributed communication. Used when no port is specified via environment variables or command-line arguments. |
| `default_pg_timeout` | `timedelta(minutes=30)` | Default process group timeout. Can be overridden via the `DEEPSPEED_TIMEOUT` environment variable (in minutes). |

#### Timeout Configuration

```python
import os
from datetime import timedelta

# The timeout is configurable via environment variable
default_pg_timeout = timedelta(minutes=int(os.getenv("DEEPSPEED_TIMEOUT", default=30)))
```

To increase the timeout for large-scale training:
```bash
export DEEPSPEED_TIMEOUT=120  # 2 hours
deepspeed train.py --deepspeed_config ds_config.json
```

### Inference Mode Constants

| Constant | Value | Description |
|---|---|---|
| `INFERENCE_GENERIC_MODE` | `'generic'` | Generic inference mode. No kernel replacements; uses standard PyTorch operations. Suitable for models without specialized kernel support. |
| `INFERENCE_SPECIALIZED_MODE` | `'specialized'` | Specialized inference mode. Enables DeepSpeed kernel replacements (e.g., fused attention, fused MLP) for supported model architectures. |

### Distributed Rank Constants

| Constant | Value | Description |
|---|---|---|
| `CROSS_RANK` | `"CROSS_RANK"` | Environment variable key for cross-node rank in expert/tensor parallel groups. |
| `CROSS_SIZE` | `"CROSS_SIZE"` | Environment variable key for cross-node world size in expert/tensor parallel groups. |
| `LOCAL_RANK` | `"LOCAL_RANK"` | Environment variable key for local (within-node) rank identifier. Used to assign GPUs to processes on multi-GPU nodes. |

#### Usage Example

```python
import os
import deepspeed.constants as const

local_rank = int(os.getenv(const.LOCAL_RANK, 0))
```

---

## Runtime Constants (deepspeed/runtime/constants.py)

These constants govern the training loop, optimizer configuration, mixed precision settings, checkpointing, and pipeline parallelism.

### Route Constants

| Constant | Value | Description |
|---|---|---|
| `ROUTE_TRAIN` | `"train"` | Route identifier for training mode |
| `ROUTE_EVAL` | `"eval"` | Route identifier for evaluation mode |
| `ROUTE_PREDICT` | `"predict"` | Route identifier for prediction mode |
| `ROUTE_ENCODE` | `"encode"` | Route identifier for encoding mode |

### Batch Size Constants

| Constant | Value | Description |
|---|---|---|
| `TRAIN_BATCH_SIZE` | `"train_batch_size"` | Config key for total training batch size |
| `TRAIN_BATCH_SIZE_DEFAULT` | `32` | Default training batch size if not specified |

### Sparse Attention Constants

| Constant | Value | Description |
|---|---|---|
| `SPARSE_ATTENTION` | `"sparse_attention"` | Config key for sparse attention module |
| `SPARSE_MODE` | `"mode"` | Sparse attention mode key |
| `SPARSE_BLOCK` | `"block"` | Sparse attention block size key |
| `SPARSE_DIFFERENT_LAYERS` | `"different_layout_per_head"` | Enable per-head sparse layout |

### Optimizer Configuration Constants

| Constant | Value | Description |
|---|---|---|
| `OPTIMIZER` | `"optimizer"` | Top-level optimizer config key |
| `OPTIMIZER_TYPE` | `"type"` | Optimizer type selection key (e.g., `"AdamW"`, `"Muon"`) |
| `OPTIMIZER_PARAMS` | `"params"` | Optimizer hyperparameters key |
| `SCHEDULER` | `"scheduler"` | Learning rate scheduler config key |

### FP16 Mixed Precision Constants

| Constant | Value | Description |
|---|---|---|
| `FP16` | `"fp16"` | Top-level FP16 config key |
| `FP16_ENABLED` | `"enabled"` | FP16 enable/disable key |
| `FP16_LOSS_SCALE` | `"loss_scale"` | Static loss scale value. Default `0` means dynamic scaling. |
| `FP16_AUTO_CAST` | `"auto_cast"` | Enable PyTorch autocast for FP16 |
| `FP16_INITIAL_SCALE_POWER` | `"initial_scale_power"` | Initial loss scale as `2^power`. Default `16` (scale = 65536). |
| `FP16_LOSS_SCALE_WINDOW` | `"loss_scale_window"` | Number of steps between loss scale adjustments. Default `1000`. |
| `FP16_HYSTERESIS` | `"hysteresis"` | Loss scale hysteresis factor. Default `2`. |
| `FP16_CONSECUTIVE_HYSTERESIS` | `"consecutive_hysteresis"` | Enable consecutive hysteresis for loss scaling. |
| `FP16_MIN_LOSS_SCALE` | `"min_loss_scale"` | Minimum loss scale value. Default `1`. |
| `FP16_MASTER_WEIGHTS_AND_GRADS` | `"fp16_master_weights_and_grads"` | Store master weights and gradients in FP16 (vs FP32). |

#### FP16 Default Values

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

### BF16 Mixed Precision Constants

| Constant | Value | Description |
|---|---|---|
| `BFLOAT16` | `"bf16"` | Top-level BF16 config key |
| `BFLOAT16_ENABLED` | `"enabled"` | BF16 enable/disable key |
| `BFLOAT16_CHECK_OVERFLOW` | `"check_overflow"` | Enable gradient overflow checking for BF16. Default `false` since BF16 has wider dynamic range. |
| `BFLOAT16_IMMEDIATE_GRAD_UPDATE` | `"immediate_grad_update"` | Immediately apply gradient updates in BF16 mode. |
| `BFLOAT16_MASTER_WEIGHTS_AND_GRADS` | `"bf16_master_weights_and_grads"` | Store master weights and gradients in BF16. |
| `BFLOAT16_OPTIMIZER_STATES` | `"bf16_optimizer_states"` | Store optimizer states in BF16. |
| `BFLOAT16_DDP_BFLOAT16` | `"ddp_bfloat16"` | Enable BF16 for DDP gradient synchronization. |

#### BF16 Default Configuration

```json
{
    "bf16": {
        "enabled": true,
        "check_overflow": false
    }
}
```

### Gradient Constants

| Constant | Value | Description |
|---|---|---|
| `GRADIENT_CLIPPING` | `"gradient_clipping"` | Gradient clipping config key. Value `0` (default) means no clipping. |
| `GRAD_ACCUM_DTYPE` | `"grad_accum_dtype"` | Data type for gradient accumulation. Can be `"fp16"`, `"bf16"`, or `"fp32"`. |
| `WALL_CLOCK_BREAKDOWN` | `"wall_clock_breakdown"` | Enable wall-clock timing breakdown of training steps. |
| `MEMORY_BREAKDOWN` | `"memory_breakdown"` | Enable detailed memory usage breakdown. |

### Checkpoint Constants

| Constant | Value | Description |
|---|---|---|
| `CHECKPOINT` | `"checkpoint"` | Top-level checkpoint config key |
| `CHECKPOINT_OUTPUT_DIR` | `"checkpoint_dir"` | Checkpoint output directory path |
| `CHECKPOINT_TAG_INTERVAL` | `"tag_interval"` | Interval for checkpoint tagging |
| `CHECKPOINT_SIZE_LIMIT` | `"size_limit"` | Maximum checkpoint size in bytes |
| `CHECKPOINT_PARALLEL_SAVE` | `"parallel_save"` | Enable parallel checkpoint saving (multiple ranks write simultaneously). |

### Validation Constants

#### `ValidationMode` Enum

| Value | Name | Description |
|---|---|---|
| `WARN` | `"warn"` | Log validation warnings but continue. Default behavior for non-critical config issues. |
| `IGNORE` | `"ignore"` | Skip validation entirely. Use at your own risk. |
| `FAIL` | `"fail"` | Raise an exception on any validation error. Recommended for production training. |

```python
from deepspeed.runtime.constants import ValidationMode

# In config JSON
{
    "validation_mode": "fail"
}
```

### Data Type Constants

| Constant | Value | Description |
|---|---|---|
| `DATA_TYPES` | `["fp16", "bf16", "fp32"]` | List of supported training data types |
| `DATALOADER_DROP_LAST` | `"dataloader_drop_last"` | Whether to drop the last incomplete batch |

### Pipeline Parallelism Constants

| Constant | Value | Description |
|---|---|---|
| `PIPE_REPLICATED` | `"pipe_replicated"` | Parameters replicated across pipeline stages |

### Data Parallel Constants

| Constant | Value | Description |
|---|---|---|
| `DATA_PARALLEL_GROUP` | `"data_parallel_group"` | Config key for explicit data parallel group specification |
| `USE_DATA_BEFORE_EXPERT_PARALLEL` | `"use_data_before_expert_parallel"` | Enable data parallelism before expert parallelism in MoE models |

---

## Optimizer Name Constants (deepspeed/runtime/config.py)

All optimizer name constants used for registration and lookup:

| Constant | Value | Description |
|---|---|---|
| `ADAM_OPTIMIZER` | `"adam"` | FusedAdam optimizer (GPU) |
| `ADAMW_OPTIMIZER` | `"adamw"` | FusedAdam in AdamW mode |
| `LAMB_OPTIMIZER` | `"lamb"` | FusedLamb optimizer |
| `ONEBIT_ADAM_OPTIMIZER` | `"onebitadam"` | 1-bit Adam (communication-efficient) |
| `ZEROONE_ADAM_OPTIMIZER` | `"zerooneadam"` | 0/1 Adam (1-bit + momentum) |
| `ONEBIT_LAMB_OPTIMIZER` | `"onebitlamb"` | 1-bit LAMB |
| `MUADAM_OPTIMIZER` | `"muadam"` | MuAdam optimizer |
| `MUADAMW_OPTIMIZER` | `"muadamw"` | MuAdamW optimizer |
| `MUSGD_OPTIMIZER` | `"musgd"` | MuSGD optimizer |
| `LION_OPTIMIZER` | `"lion"` | FusedLion optimizer |
| `MUON_OPTIMIZER` | `"muon"` | Muon optimizer (Newton-Schulz) |
| `ADAGRAD_OPTIMIZER` | `"adagrad"` | CPUAdagrad optimizer |

### `DEEPSPEED_OPTIMIZERS` List

The complete registry of all DeepSpeed-provided optimizers:

```python
from deepspeed.runtime.config import DEEPSPEED_OPTIMIZERS

DEEPSPEED_OPTIMIZERS = [
    ADAM_OPTIMIZER,        # "adam"
    ADAMW_OPTIMIZER,       # "adamw"
    LAMB_OPTIMIZER,        # "lamb"
    ONEBIT_ADAM_OPTIMIZER, # "onebitadam"
    ZEROONE_ADAM_OPTIMIZER,# "zerooneadam"
    ONEBIT_LAMB_OPTIMIZER, # "onebitlamb"
    MUADAM_OPTIMIZER,      # "muadam"
    MUADAMW_OPTIMIZER,     # "muadamw"
    MUSGD_OPTIMIZER,       # "musgd"
    LION_OPTIMIZER,        # "lion"
    MUON_OPTIMIZER,        # "muon"
    ADAGRAD_OPTIMIZER,     # "adagrad"
]
```

---

## DtypeEnum (deepspeed/runtime/config.py)

The `DtypeEnum` class provides a flexible data type specification system with multi-alias support. It converts string representations to PyTorch dtypes.

### Supported Values and Aliases

| Dtype | Aliases | PyTorch Type |
|---|---|---|
| `torch.float16` | `"fp16"`, `"float16"`, `"half"` | `torch.float16` |
| `torch.bfloat16` | `"bf16"`, `"bfloat16"` | `torch.bfloat16` |
| `torch.float32` | `"fp32"`, `"float32"`, `"float"` | `torch.float32` |
| `torch.int8` | `"int8"` | `torch.int8` |
| `torch.int16` | `"int16"` | `torch.int16` |
| `torch.int32` | `"int32"`, `"int"` | `torch.int32` |
| `torch.int64` | `"int64"`, `"long"` | `torch.int64` |
| `torch.bool` | `"bool"` | `torch.bool` |

### Usage

```python
from deepspeed.runtime.config import DtypeEnum

# All of these resolve to torch.float16
dtype = DtypeEnum("fp16").value        # torch.float16
dtype = DtypeEnum("float16").value     # torch.float16
dtype = DtypeEnum("half").value        # torch.float16

# BF16
dtype = DtypeEnum("bf16").value        # torch.bfloat16
dtype = DtypeEnum("bfloat16").value    # torch.bfloat16
```

### Internal Implementation

`DtypeEnum` is a string-based enum that maps string aliases to the canonical type string, then resolves to the actual `torch.dtype`:

```python
# Conceptual mapping
_ALIAS_MAP = {
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp32": "fp32",
    "float32": "fp32",
    "float": "fp32",
    # ... etc
}

_STR_TO_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    # ... etc
}
```

---

## Utility Functions (deepspeed/utils/)

### Logging Utilities

#### `logger`

The standard Python logger instance for DeepSpeed. All DeepSpeed modules use this logger.

```python
from deepspeed.utils import logger

logger.info("Training started")
logger.warning("Gradient overflow detected")
logger.error("Checkpoint save failed")
```

#### `log_dist(message, ranks=None, level=logging.INFO)`

Log a message from specific ranks only. Prevents duplicate log messages in multi-process training.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `message` | str | required | Message to log |
| `ranks` | list[int] or None | `None` | Ranks that should log. If None, logs from rank 0 only. |
| `level` | int | `logging.INFO` | Python logging level |

```python
from deepspeed.utils import log_dist

# Only log from rank 0 (default)
log_dist("Global step completed", ranks=[0])

# Log from all ranks
log_dist(f"Local rank {local_rank} ready", ranks=None)
```

#### `log_dist_once(message, ranks=None, level=logging.INFO)`

Log a message exactly once across all ranks, even when called from multiple processes. Ensures a message is printed only once regardless of how many ranks execute the code path.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `message` | str | required | Message to log |
| `ranks` | list[int] or None | `None` | Ranks that may log. Only the first caller logs. |
| `level` | int | `logging.INFO` | Python logging level |

#### `set_log_level_from_string(level_str)`

Set the DeepSpeed logger level from a string name.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `level_str` | str | required | One of: `"debug"`, `"info"`, `"warning"`, `"error"`, `"critical"` |

```python
from deepspeed.utils import set_log_level_from_string

set_log_level_from_string("debug")  # Enable debug logging
set_log_level_from_string("warning")  # Only warnings and above
```

---

### Device Utilities

#### `OnDevice` Context Manager

A context manager that creates and operates on tensors on a specified device. Useful for performing temporary computations on a specific device without moving model parameters.

```python
from deepspeed.utils import OnDevice

with OnDevice(dtype=torch.float32, device="cuda:0"):
    # Tensors created here will be on cuda:0
    temp = torch.randn(10, 10)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dtype` | torch.dtype | `torch.float32` | Data type for tensors created in context |
| `device` | str or torch.device | required | Target device |

---

### Distributed Group Utilities

The `groups` module provides utilities for managing distributed process groups.

#### `groups.initialize(model_parallel_size)`

Initialize all distributed groups (data parallel, model parallel, etc.).

```python
from deepspeed.utils import groups

# Initialize with model parallel size
groups.initialize(model_parallel_size=2)
```

#### Key Group Attributes

| Attribute | Type | Description |
|---|---|---|
| `groups.mpu` | object | Model parallel utility object |
| `groups.data_parallel_group` | ProcessGroup | Data parallel process group |
| `groups.model_parallel_group` | ProcessGroup | Model parallel process group |

---

### Profiling Utilities

#### `instrument_w_nvtx(func)`

Decorator that wraps a function with NVTX (NVIDIA Tools Extension) range markers for profiling with NVIDIA Nsight Systems.

```python
from deepspeed.utils import instrument_w_nvtx

@instrument_w_nvtx
def forward_pass(x):
    return model(x)

# The function call will appear as a labeled range in Nsight Systems
result = forward_pass(inputs)
```

Can also be used without decorator syntax:

```python
instrumented_fn = instrument_w_nvtx(my_function)
result = instrumented_fn(args)
```

---

### Tensor Fragment Utilities

The `tensor_fragment` module provides utilities for managing fragments of flat parameter tensors in ZeRO optimization.

#### Key Functions

| Function | Description |
|---|---|
| `get_tensor_fragment(tensor, partition_start, partition_end)` | Extract a contiguous fragment from a flat tensor given start/end indices |
| `get_full_fragment(tensor, partition_size, world_size, rank)` | Reconstruct the full tensor from a partition by gathering across ranks |

---

### ZeRO-3 Leaf Module Utilities

The `z3_leaf_module` module provides utilities for marking modules as "leaf" modules in ZeRO-3, preventing parameter splitting for specific submodules.

#### `z3_leaf_module(module)`

Mark a module as a ZeRO-3 leaf, preventing its parameters from being split across ranks.

```python
from deepspeed.utils import z3_leaf_module

# Prevent specific module's params from being sharded in ZeRO-3
z3_leaf_module(model.special_layer)
```

---

### Parameter Utilities

#### `link_hp_params(lp_param, hp_param, lp_optimizer_state, group_id)`

Link a low-precision (LP) parameter to its high-precision (HP) counterpart. Used internally by ZeRO to maintain the relationship between sharded parameters and their full-precision copies.

| Parameter | Type | Description |
|---|---|---|
| `lp_param` | nn.Parameter | Low-precision parameter |
| `hp_param` | nn.Parameter | High-precision parameter |
| `lp_optimizer_state` | dict | Optimizer state for the LP parameter |
| `group_id` | int | Parameter group identifier |

---

### DataLoader Utilities

#### `RepeatingLoader(data_loader)`

A wrapper around any PyTorch DataLoader that repeats indefinitely. When the underlying DataLoader is exhausted, it is restarted from the beginning.

```python
from deepspeed.utils import RepeatingLoader
from torch.utils.data import DataLoader

base_loader = DataLoader(dataset, batch_size=32)
infinite_loader = RepeatingLoader(base_loader)

# Never raises StopIteration
for batch in infinite_loader:
    train_step(batch)
    if done:
        break
```

This is particularly useful for training loops that need to continue until a convergence criterion is met, rather than for a fixed number of epochs.

| Parameter | Type | Description |
|---|---|---|
| `data_loader` | DataLoader | Any PyTorch DataLoader to wrap |

---

### System Utilities

#### `get_numactl_cmd(numa_node)`

Returns the `numactl` command prefix for binding to a specific NUMA node. Used for CPU affinity optimization in multi-socket systems.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `numa_node` | int | required | NUMA node ID to bind to |

```python
from deepspeed.utils import get_numactl_cmd

cmd_prefix = get_numactl_cmd(0)
# Returns: ["numactl", "--cpunodebind=0", "--membind=0"]

full_cmd = cmd_prefix + ["python", "train.py"]
subprocess.run(full_cmd)
```

---

## Quick Reference: Config Key Mapping

### Complete Config JSON Key Reference

```json
{
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 8,
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
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "bf16": {
        "enabled": false,
        "check_overflow": false
    },

    "gradient_clipping": 1.0,
    "grad_accum_dtype": "fp32",

    "wall_clock_breakdown": false,
    "memory_breakdown": false,

    "checkpoint": {
        "checkpoint_dir": "/path/to/checkpoints",
        "tag_interval": 100,
        "size_limit": 0,
        "parallel_save": false
    },

    "validation_mode": "warn",
    "dataloader_drop_last": true,

    "sparse_attention": {
        "mode": "fixed",
        "block": 16,
        "different_layout_per_head": false
    },

    "zero_optimization": {
        "stage": 2,
        "data_parallel_group": null,
        "use_data_before_expert_parallel": false
    }
}
```

---

## Source Files

| File | Description |
|---|---|
| `deepspeed/constants.py` | Top-level package constants (port, timeout, inference modes, rank keys) |
| `deepspeed/runtime/constants.py` | Runtime training constants (optimizer keys, FP16/BF16 keys, checkpoint keys) |
| `deepspeed/runtime/config.py` | Optimizer name constants, DEEPSPEED_OPTIMIZERS, DtypeEnum, DeepSpeedConfig |
| `deepspeed/utils/__init__.py` | Utility function exports |
| `deepspeed/utils/logging.py` | Logger, log_dist, log_dist_once |
| `deepspeed/utils/tensor_fragment.py` | ZeRO tensor fragment utilities |
| `deepspeed/utils/z3_leaf_module.py` | ZeRO-3 leaf module marking |
