# DeepSpeed Reference - Chapter 3: Configuration Reference

This chapter provides the complete reference for the DeepSpeed configuration file (`ds_config.json`). Every configuration field is documented with its name, type, description, default value, valid options, and usage examples.

---

## 3.1 Configuration Overview

### 3.1.1 Configuration Format

DeepSpeed configuration is defined as a JSON file (`ds_config.json`) or a Python dictionary. The configuration drives all aspects of training behavior.

**JSON file format:**

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
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2
    }
}
```

**Python dictionary format:**

```python
ds_config = {
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": 2},
}
```

### 3.1.2 Configuration Categories

| Category | Top-Level Key | Required |
|----------|--------------|----------|
| Batch Size | `train_batch_size`, `train_micro_batch_size_per_gpu`, `gradient_accumulation_steps` | At least one |
| Optimizer | `optimizer` | Recommended |
| Scheduler | `scheduler` | Optional |
| FP16 Training | `fp16` | Optional |
| BF16 Training | `bf16` | Optional |
| AMP Training | `amp` | Optional |
| Torch Autocast | `torch_autocast` | Optional |
| ZeRO Optimization | `zero_optimization` | Optional |
| Gradient Clipping | `gradient_clipping` | Optional |
| Communication | Various | Optional |
| Checkpoint | `checkpoint` | Optional |
| Tensor Parallel | `tensor_parallel` | Optional |
| Pipeline | `pipeline` | Optional |
| Logging | Various | Optional |
| Autotuning | `autotuning` | Optional |
| Flops Profiler | `flops_profiler` | Optional |
| Activation Checkpointing | `activation_checkpointing` | Optional |
| Sparse Attention | `sparse_attention` | Optional |
| Curriculum Learning | `curriculum_learning` | Optional |
| Monitor | Various | Optional |
| Compression | `compression` | Optional |
| Elasticity | `elasticity` | Optional |
| Communication Logging | `communication_logging` | Optional |
| DeepCompile | `deepcompile` | Optional |
| Nebula | `nebula` | Optional |
| DataStates | `data_states` | Optional |
| ZenFlow | `zenflow` | Optional |

---

## 3.2 Batch Size Parameters

DeepSpeed decouples the effective training batch size from the per-GPU micro-batch size through gradient accumulation. The relationship is:

```
train_batch_size = train_micro_batch_size_per_gpu * gradient_accumulation_steps * world_size
```

### 3.2.1 `train_batch_size`

| Field | Value |
|-------|-------|
| **Name** | `train_batch_size` |
| **Type** | `int` |
| **Default** | Computed from micro_batch_size * grad_accum * world_size |
| **Required** | No (can be auto-computed) |
| **Valid Range** | Positive integer |
| **Description** | The effective total training batch size across all GPUs and gradient accumulation steps. If specified along with `train_micro_batch_size_per_gpu`, `gradient_accumulation_steps` is automatically computed. |

**Example:**

```json
{
    "train_batch_size": 1024
}
```

### 3.2.2 `train_micro_batch_size_per_gpu`

| Field | Value |
|-------|-------|
| **Name** | `train_micro_batch_size_per_gpu` |
| **Type** | `int` |
| **Default** | Computed from train_batch_size / (grad_accum * world_size) |
| **Required** | No |
| **Valid Range** | Positive integer |
| **Description** | The batch size processed in a single forward+backward pass on each GPU. This determines the peak GPU memory usage during training. Larger values use more memory but may improve throughput. |

**Example:**

```json
{
    "train_micro_batch_size_per_gpu": 8
}
```

### 3.2.3 `gradient_accumulation_steps`

| Field | Value |
|-------|-------|
| **Name** | `gradient_accumulation_steps` |
| **Type** | `int` |
| **Default** | Computed from train_batch_size / (micro_batch_size * world_size) |
| **Required** | No |
| **Valid Range** | Positive integer |
| **Description** | Number of forward+backward passes to accumulate gradients before performing an optimizer step. Increasing this allows a larger effective batch size without increasing GPU memory usage. |

**Example:**

```json
{
    "gradient_accumulation_steps": 16
}
```

### 3.2.4 Batch Size Configuration Examples

**Example 1: Specify everything explicitly**

```json
{
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8
}
```
This is valid for `256 / (4 * 8) = 8` GPUs.

**Example 2: Specify train_batch_size only (auto-compute)**

```json
{
    "train_batch_size": 256
}
```
With 8 GPUs, this gives `micro_batch_size_per_gpu = 1`, `gradient_accumulation_steps = 32`.

**Example 3: Specify micro_batch_size and grad_accum only**

```json
{
    "train_micro_batch_size_per_gpu": 16,
    "gradient_accumulation_steps": 4
}
```
With 8 GPUs, `train_batch_size = 16 * 4 * 8 = 512`.

---

## 3.3 Optimizer Configuration

### 3.3.1 `optimizer`

| Field | Value |
|-------|-------|
| **Name** | `optimizer` |
| **Type** | `object` |
| **Default** | None (use user-provided optimizer or error) |
| **Required** | No (if optimizer passed to `initialize()`) |
| **Description** | Defines the optimizer type and its hyperparameters. DeepSpeed can create the optimizer internally or wrap an existing user-provided optimizer. |

**Structure:**

```json
{
    "optimizer": {
        "type": "<optimizer_name>",
        "params": {
            "<param1>": <value1>,
            "<param2>": <value2>
        }
    }
}
```

### 3.3.2 Supported Optimizers

#### Adam

| Field | Value |
|-------|-------|
| **type** | `"Adam"` |
| **Description** | Standard Adam optimizer. Uses the fused CUDA kernel when available, otherwise falls back to PyTorch native Adam. |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | `float` | Required | Learning rate |
| `betas` | `[float, float]` | `[0.9, 0.999]` | Coefficients for running averages of gradient and its square |
| `eps` | `float` | `1e-8` | Term added for numerical stability |
| `weight_decay` | `float` | `0` | Weight decay (L2 penalty) |
| `amsgrad` | `bool` | `false` | Use AMSGrad variant |
| `max_grad_norm` | `float` | `0` | Max gradient norm for clipping (0 = no clipping) |

**Example:**

```json
{
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    }
}
```

#### AdamW

| Field | Value |
|-------|-------|
| **type** | `"AdamW"` |
| **Description** | Adam with decoupled weight decay. This is the recommended optimizer for most transformer training. Uses the fused CUDA kernel when available. |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | `float` | Required | Learning rate |
| `betas` | `[float, float]` | `[0.9, 0.999]` | Coefficients for running averages |
| `eps` | `float` | `1e-8` | Numerical stability term |
| `weight_decay` | `float` | `0.01` | Decoupled weight decay |
| `amsgrad` | `bool` | `false` | Use AMSGrad variant |
| `max_grad_norm` | `float` | `0` | Max gradient norm for clipping |

**Example:**

```json
{
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    }
}
```

#### Lamb

| Field | Value |
|-------|-------|
| **type** | `"Lamb"` |
| **Description** | Layer-wise Adaptive Moments optimizer for Batch training. Provides better scaling to large batch sizes than Adam. |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | `float` | Required | Learning rate |
| `betas` | `[float, float]` | `[0.9, 0.999]` | Coefficients for running averages |
| `eps` | `float` | `1e-6` | Numerical stability term |
| `weight_decay` | `float` | `0.01` | Weight decay |
| `max_grad_norm` | `float` | `0` | Max gradient norm |
| `max_coeff` | `float` | `10.0` | Maximum trust ratio coefficient |
| `min_coeff` | `float` | `0.01` | Minimum trust ratio coefficient |

**Example:**

```json
{
    "optimizer": {
        "type": "Lamb",
        "params": {
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-6,
            "weight_decay": 0.01
        }
    }
}
```

#### OneBitAdam

| Field | Value |
|-------|-------|
| **type** | `"OneBitAdam"` |
| **Description** | 1-bit compressed Adam for communication-efficient distributed training. After a warmup phase, gradients are compressed to 1-bit representation, reducing communication volume by ~26x. |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | `float` | Required | Learning rate |
| `betas` | `[float, float]` | `[0.9, 0.999]` | Coefficients for running averages |
| `eps` | `float` | `1e-8` | Numerical stability term |
| `weight_decay` | `float` | `0` | Weight decay |
| `freeze_step` | `int` | `100000` | Step at which to switch from full-precision to 1-bit compression |
| `cuda_aware` | `bool` | `false` | Enable CUDA-aware communication |
| `comm_backend_name` | `str` | `"nccl"` | Communication backend |

**Example:**

```json
{
    "optimizer": {
        "type": "OneBitAdam",
        "params": {
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "freeze_step": 1000
        }
    }
}
```

#### OneBitLamb

| Field | Value |
|-------|-------|
| **type** | `"OneBitLamb"` |
| **Description** | 1-bit compressed Lamb optimizer. Combines Lamb's layer-wise scaling with 1-bit gradient compression. |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | `float` | Required | Learning rate |
| `betas` | `[float, float]` | `[0.9, 0.999]` | Coefficients |
| `eps` | `float` | `1e-6` | Numerical stability term |
| `weight_decay` | `float` | `0.01` | Weight decay |
| `freeze_step` | `int` | `100000` | Step to switch to 1-bit |
| `max_coeff` | `float` | `10.0` | Max trust ratio |
| `min_coeff` | `float` | `0.01` | Min trust ratio |
| `cuda_aware` | `bool` | `false` | CUDA-aware communication |

**Example:**

```json
{
    "optimizer": {
        "type": "OneBitLamb",
        "params": {
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "freeze_step": 5000
        }
    }
}
```

#### ZeroOneAdam

| Field | Value |
|-------|-------|
| **type** | `"ZeroOneAdam"` |
| **Description** | Hybrid 0/1 compressed Adam. Alternates between full-precision and 1-bit compression for optimal accuracy-efficiency tradeoff. |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | `float` | Required | Learning rate |
| `betas` | `[float, float]` | `[0.9, 0.999]` | Coefficients |
| `eps` | `float` | `1e-8` | Numerical stability |
| `weight_decay` | `float` | `0` | Weight decay |
| `freeze_step` | `int` | `100000` | Step to switch to 0/1 mode |
| `cuda_aware` | `bool` | `false` | CUDA-aware communication |
| `comm_backend_name` | `str` | `"nccl"` | Communication backend |
| `offset` | `float` | `0.0` | Error feedback offset |
| `a clr` | `float` | `0` | Adaptive learning rate |

**Example:**

```json
{
    "optimizer": {
        "type": "ZeroOneAdam",
        "params": {
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "freeze_step": 2000
        }
    }
}
```

#### Lion

| Field | Value |
|-------|-------|
| **type** | `"Lion"` |
| **Description** | EvoLved Sign Momentum optimizer. Uses sign of the momentum for updates, achieving similar or better performance than Adam with lower memory. |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | `float` | Required | Learning rate (typically 3-10x smaller than Adam) |
| `betas` | `[float, float]` | `[0.9, 0.99]` | Momentum decay rates |
| `weight_decay` | `float` | `0.0` | Weight decay |

**Example:**

```json
{
    "optimizer": {
        "type": "Lion",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.99],
            "weight_decay": 0.01
        }
    }
}
```

#### Muon

| Field | Value |
|-------|-------|
| **type** | `"Muon"` |
| **Description** | Momentum orthogonalized by Newton-Schulz iteration. An optimizer that applies orthogonalization to momentum updates for better conditioning. |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | `float` | Required | Learning rate |
| `momentum` | `float` | `0.95` | Momentum factor |
| `nesterov` | `bool` | `true` | Use Nesterov momentum |
| `weight_decay` | `float` | `0.0` | Weight decay |
| `ns_steps` | `int` | `5` | Number of Newton-Schulz iterations |

**Example:**

```json
{
    "optimizer": {
        "type": "Muon",
        "params": {
            "lr": 2e-3,
            "momentum": 0.95,
            "weight_decay": 0.01
        }
    }
}
```

#### MuAdam / MuAdamW

| Field | Value |
|-------|-------|
| **type** | `"MuAdam"` or `"MuAdamW"` |
| **Description** | Mu-optimizer variants of Adam and AdamW, which apply Mueller-iteration-based orthogonalization. |

**Parameters (MuAdam):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | `float` | Required | Learning rate |
| `betas` | `[float, float]` | `[0.9, 0.999]` | Adam beta coefficients |
| `eps` | `float` | `1e-8` | Numerical stability |
| `weight_decay` | `float` | `0.0` | Weight decay |

**Parameters (MuAdamW):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | `float` | Required | Learning rate |
| `betas` | `[float, float]` | `[0.9, 0.999]` | Adam beta coefficients |
| `eps` | `float` | `1e-8` | Numerical stability |
| `weight_decay` | `float` | `0.01` | Decoupled weight decay |

**Example:**

```json
{
    "optimizer": {
        "type": "MuAdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    }
}
```

#### MuSGD

| Field | Value |
|-------|-------|
| **type** | `"MuSGD"` |
| **Description** | Mu-optimizer variant of SGD with Mueller-iteration-based orthogonalization. |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | `float` | Required | Learning rate |
| `momentum` | `float` | `0.0` | Momentum factor |
| `weight_decay` | `float` | `0.0` | Weight decay |
| `nesterov` | `bool` | `false` | Use Nesterov momentum |

**Example:**

```json
{
    "optimizer": {
        "type": "MuSGD",
        "params": {
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 1e-4
        }
    }
}
```

#### Adagrad

| Field | Value |
|-------|-------|
| **type** | `"Adagrad"` |
| **Description** | Adaptive gradient optimizer. Particularly effective for sparse data. DeepSpeed provides a CPU-optimized version for offloading. |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | `float` | Required | Learning rate |
| `eps` | `float` | `1e-10` | Numerical stability |
| `weight_decay` | `float` | `0` | Weight decay |
| `initial_accumulator_value` | `float` | `0` | Initial value for sum of squares |

**Example:**

```json
{
    "optimizer": {
        "type": "Adagrad",
        "params": {
            "lr": 1e-2,
            "eps": 1e-10,
            "weight_decay": 0.01
        }
    }
}
```

---

## 3.4 Scheduler Configuration

### 3.4.1 `scheduler`

| Field | Value |
|-------|-------|
| **Name** | `scheduler` |
| **Type** | `object` |
| **Default** | None |
| **Required** | No |
| **Description** | Defines the learning rate scheduler. DeepSpeed provides several built-in schedulers. Alternatively, a user-provided scheduler can be passed to `initialize()`. |

**Structure:**

```json
{
    "scheduler": {
        "type": "<scheduler_name>",
        "params": {
            "<param1>": <value1>
        }
    }
}
```

### 3.4.2 Supported Schedulers

#### WarmupLR

| Field | Value |
|-------|-------|
| **type** | `"WarmupLR"` |
| **Description** | Linear warmup followed by linear decay. During warmup, the LR linearly increases from 0 to the base LR over `warmup_min_lr` to `warmup_max_lr`. After warmup, LR linearly decays to `warmup_max_lr * total_num_steps / warmup_num_steps`. |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `warmup_min_lr` | `float` | `0` | Starting learning rate for warmup |
| `warmup_max_lr` | `float` | Required | Maximum (target) learning rate |
| `warmup_num_steps` | `int` | Required | Number of warmup steps |
| `total_num_steps` | `int` | Required | Total number of training steps |

**Example:**

```json
{
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": 1000,
            "total_num_steps": 100000
        }
    }
}
```

#### OneCycle

| Field | Value |
|-------|-------|
| **type** | `"OneCycle"` |
| **Description** | 1cycle learning rate policy. Increases LR from `lr_min` to `lr_max`, then decreases back to `lr_min`, with an optional final decrease to a much lower value. |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr_min` | `float` | Required | Minimum learning rate |
| `lr_max` | `float` | Required | Maximum learning rate |
| `cycle_first_step_size` | `int` | Required | Steps in first half of cycle |
| `cycle_first_stair_count` | `int` | `500` | Stair count for first phase |
| `cycle_second_step_size` | `int` | `None` | Steps in second half (default = first_step_size) |
| `cycle_second_stair_count` | `int` | `500` | Stair count for second phase |
| `decay_step_size` | `int` | `None` | Step size for final decay |
| `cycle_momentum` | `bool` | `true` | Cycle momentum inversely with LR |
| `momentum_min` | `float` | `0.85` | Minimum momentum |
| `momentum_max` | `float` | `0.95` | Maximum momentum |
| `momentum_decay` | `float` | `0.01` | Momentum decay |

**Example:**

```json
{
    "scheduler": {
        "type": "OneCycle",
        "params": {
            "lr_min": 1e-6,
            "lr_max": 1e-3,
            "cycle_first_step_size": 5000,
            "cycle_first_stair_count": 500
        }
    }
}
```

#### LRRangeTest

| Field | Value |
|-------|-------|
| **type** | `"LRRangeTest"` |
| **Description** | Linear LR range test. Increases LR linearly from `lr_min` to `lr_max` over `num_steps`. Used for finding the optimal learning rate. |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr_min` | `float` | Required | Starting learning rate |
| `lr_max` | `float` | Required | Ending learning rate |
| `num_steps` | `int` | Required | Total number of test steps |
| `staircase` | `bool` | `false` | Use staircase (discrete) increases |

**Example:**

```json
{
    "scheduler": {
        "type": "LRRangeTest",
        "params": {
            "lr_min": 1e-6,
            "lr_max": 10.0,
            "num_steps": 2000
        }
    }
}
```

#### WarmupDecayLR

| Field | Value |
|-------|-------|
| **type** | `"WarmupDecayLR"` |
| **Description** | Warmup followed by cosine decay. Warms up linearly then decays using a cosine schedule. |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `warmup_min_lr` | `float` | `0` | Starting LR |
| `warmup_max_lr` | `float` | Required | Peak LR |
| `warmup_num_steps` | `int` | Required | Warmup steps |
| `total_num_steps` | `int` | Required | Total training steps |

**Example:**

```json
{
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 1e-7,
            "warmup_max_lr": 5e-5,
            "warmup_num_steps": 2000,
            "total_num_steps": 500000
        }
    }
}
```

---

## 3.5 Communication Options

### 3.5.1 `communication_data_type`

| Field | Value |
|-------|-------|
| **Name** | `communication_data_type` |
| **Type** | `str` |
| **Default** | Same as training data type |
| **Required** | No |
| **Valid Options** | `"fp32"`, `"fp16"`, `"bf16"` |
| **Description** | Data type used for inter-GPU gradient communication. Can be set independently of the training data type to reduce communication volume. |

**Example:**

```json
{
    "communication_data_type": "fp16"
}
```

### 3.5.2 `prescale_gradients`

| Field | Value |
|-------|-------|
| **Name** | `prescale_gradients` |
| **Type** | `bool` |
| **Default** | `false` |
| **Required** | No |
| **Description** | Scale gradients before the all-reduce communication. This can prevent underflow in FP16 training by scaling up before averaging across GPUs. |

**Example:**

```json
{
    "prescale_gradients": true
}
```

### 3.5.3 `gradient_predivide_factor`

| Field | Value |
|-------|-------|
| **Name** | `gradient_predivide_factor` |
| **Type** | `float` |
| **Default** | `1.0` |
| **Required** | No |
| **Description** | Divides gradients by this factor before all-reduce, then multiplies back after. Useful for preventing overflow in large-scale training (e.g., set to `world_size` to pre-divide before all-reduce). |

**Example:**

```json
{
    "gradient_predivide_factor": 8.0
}
```

### 3.5.4 `sparse_gradients`

| Field | Value |
|-------|-------|
| **Name** | `sparse_gradients` |
| **Type** | `bool` |
| **Default** | `false` |
| **Required** | No |
| **Description** | Enable sparse gradient all-reduce for embedding layers. When enabled, only non-zero gradients are communicated, significantly reducing communication for sparse models. |

**Example:**

```json
{
    "sparse_gradients": true
}
```

---

## 3.6 FP16 Training

### 3.6.1 `fp16`

| Field | Value |
|-------|-------|
| **Name** | `fp16` |
| **Type** | `object` |
| **Default** | `{"enabled": false}` |
| **Required** | No |
| **Description** | Configuration for FP16 (IEEE 754 half precision) mixed precision training. When enabled, the forward pass and gradients are computed in FP16, while the optimizer maintains FP32 master weights and optimizer states. |

#### `fp16.enabled`

| Field | Value |
|-------|-------|
| **Name** | `fp16.enabled` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable FP16 mixed precision training. Mutually exclusive with `bf16.enabled` and `amp.enabled`. |

#### `fp16.auto_cast`

| Field | Value |
|-------|-------|
| **Name** | `fp16.auto_cast` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Use `torch.autocast` for automatic mixed precision instead of manual FP16 casting. This lets PyTorch decide which operations to run in FP16 vs FP32. |

#### `fp16.loss_scale`

| Field | Value |
|-------|-------|
| **Name** | `fp16.loss_scale` |
| **Type** | `float` or `int` |
| **Default** | `0` |
| **Valid Values** | `0` (dynamic), positive float (static) |
| **Description** | Loss scaling factor to prevent gradient underflow in FP16. Set to 0 for dynamic loss scaling (recommended), or a specific value for static scaling. |

#### `fp16.initial_scale_power`

| Field | Value |
|-------|-------|
| **Name** | `fp16.initial_scale_power` |
| **Type** | `int` |
| **Default** | `32` |
| **Valid Range** | 1 to 64 |
| **Description** | Initial loss scale as `2^initial_scale_power`. For dynamic loss scaling, this is the starting scale. |

#### `fp16.loss_scale_window`

| Field | Value |
|-------|-------|
| **Name** | `fp16.loss_scale_window` |
| **Type** | `int` |
| **Default** | `1000` |
| **Description** | Number of successful steps before increasing the loss scale (for dynamic scaling). |

#### `fp16.hysteresis`

| Field | Value |
|-------|-------|
| **Name** | `fp16.hysteresis` |
| **Type** | `int` |
| **Default** | `2` |
| **Description** | Number of consecutive overflow steps before reducing the loss scale. Higher values make the loss scale more tolerant of occasional overflows. |

#### `fp16.consecutive_hysteresis`

| Field | Value |
|-------|-------|
| **Name** | `fp16.consecutive_hysteresis` |
| **Type** | `bool` |
| **Default** | `true` |
| **Description** | Whether to use consecutive overflow detection for hysteresis. When enabled, the overflow counter resets after a successful step. |

#### `fp16.min_loss_scale`

| Field | Value |
|-------|-------|
| **Name** | `fp16.min_loss_scale` |
| **Type** | `float` |
| **Default** | `1` |
| **Description** | Minimum loss scale value. The dynamic loss scale will not decrease below this value. |

#### `fp16.fp16_master_weights_and_grads`

| Field | Value |
|-------|-------|
| **Name** | `fp16.fp16_master_weights_and_grads` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Keep master weights and gradients in FP16 instead of FP32. This reduces memory usage but may harm training stability. Only recommended for inference or when memory is extremely constrained. |

**Full FP16 example:**

```json
{
    "fp16": {
        "enabled": true,
        "auto_cast": false,
        "loss_scale": 0,
        "initial_scale_power": 32,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "consecutive_hysteresis": true,
        "min_loss_scale": 1,
        "fp16_master_weights_and_grads": false
    }
}
```

---

## 3.7 BF16 Training

### 3.7.1 `bf16`

| Field | Value |
|-------|-------|
| **Name** | `bf16` |
| **Type** | `object` |
| **Default** | `{"enabled": false}` |
| **Required** | No |
| **Description** | Configuration for BF16 (Brain Float 16) mixed precision training. BF16 has the same dynamic range as FP32 (8 exponent bits) but reduced mantissa precision (7 bits), making it more robust to overflow/underflow than FP16 while still providing memory savings. |

#### `bf16.enabled`

| Field | Value |
|-------|-------|
| **Name** | `bf16.enabled` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable BF16 mixed precision training. Mutually exclusive with `fp16.enabled` and `amp.enabled`. BF16 does not require loss scaling. |

#### `bf16.bf16_master_weights_and_grads`

| Field | Value |
|-------|-------|
| **Name** | `bf16.bf16_master_weights_and_grads` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Keep master weights and gradients in BF16. Reduces memory but may affect training quality. |

#### `bf16.bf16_optimizer_states`

| Field | Value |
|-------|-------|
| **Name** | `bf16.bf16_optimizer_states` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Store optimizer states in BF16 instead of FP32. Provides additional memory savings at the cost of optimizer state precision. |

**Full BF16 example:**

```json
{
    "bf16": {
        "enabled": true,
        "bf16_master_weights_and_grads": false,
        "bf16_optimizer_states": false
    }
}
```

---

## 3.8 AMP Training

### 3.8.1 `amp`

| Field | Value |
|-------|-------|
| **Name** | `amp` |
| **Type** | `object` |
| **Default** | `{"enabled": false}` |
| **Required** | No |
| **Description** | Configuration for PyTorch native Automatic Mixed Precision (AMP). Uses `torch.cuda.amp` for mixed precision. Mutually exclusive with `fp16` and `bf16`. |

#### `amp.enabled`

| Field | Value |
|-------|-------|
| **Name** | `amp.enabled` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable PyTorch native AMP. |

#### `amp.opt_level`

| Field | Value |
|-------|-------|
| **Name** | `amp.opt_level` |
| **Type** | `str` |
| **Default** | `"O1"` |
| **Valid Options** | `"O1"`, `"O2"` |
| **Description** | AMP optimization level. O1 uses automatic casting, O2 uses FP16 for most operations with FP32 master weights. |

**Example:**

```json
{
    "amp": {
        "enabled": true,
        "opt_level": "O2"
    }
}
```

---

## 3.9 Torch Autocast

### 3.9.1 `torch_autocast`

| Field | Value |
|-------|-------|
| **Name** | `torch_autocast` |
| **Type** | `object` |
| **Default** | `{"enabled": false}` |
| **Required** | No |
| **Description** | Configuration for using `torch.autocast` for mixed precision. This provides fine-grained control over autocast behavior and allows specifying which modules are safe for lower precision. |

#### `torch_autocast.enabled`

| Field | Value |
|-------|-------|
| **Name** | `torch_autocast.enabled` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable `torch.autocast` for automatic mixed precision. |

#### `torch_autocast.dtype`

| Field | Value |
|-------|-------|
| **Name** | `torch_autocast.dtype` |
| **Type** | `str` |
| **Default** | `"float16"` |
| **Valid Options** | `"float16"`, `"bfloat16"` |
| **Description** | The dtype used by autocast. |

#### `torch_autocast.lower_precision_safe_modules`

| Field | Value |
|-------|-------|
| **Name** | `torch_autocast.lower_precision_safe_modules` |
| **Type** | `list[str]` |
| **Default** | `["torch.nn.modules.linear.Linear", "torch.nn.modules.activation.GELU"]` |
| **Description** | List of module class paths that are safe to run in lower precision. Modules not in this list will run in FP32. |

**Example:**

```json
{
    "torch_autocast": {
        "enabled": true,
        "dtype": "bfloat16",
        "lower_precision_safe_modules": [
            "torch.nn.modules.linear.Linear",
            "torch.nn.modules.activation.GELU",
            "torch.nn.modules.normalization.LayerNorm"
        ]
    }
}
```

---

## 3.10 Gradient Clipping

### 3.10.1 `gradient_clipping`

| Field | Value |
|-------|-------|
| **Name** | `gradient_clipping` |
| **Type** | `float` |
| **Default** | `0.0` (no clipping) |
| **Required** | No |
| **Valid Range** | `0.0` or positive float |
| **Description** | Maximum gradient norm for gradient clipping. Applied before the optimizer step. A value of 0.0 disables gradient clipping. Common values for transformer training: 1.0. |

**Example:**

```json
{
    "gradient_clipping": 1.0
}
```

---

## 3.11 ZeRO Optimization

### 3.11.1 `zero_optimization`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization` |
| **Type** | `object` |
| **Default** | `{"stage": 0}` (ZeRO disabled) |
| **Required** | No |
| **Description** | Configuration for ZeRO (Zero Redundancy Optimizer). ZeRO progressively eliminates memory redundancy across data-parallel processes. |

#### `zero_optimization.stage`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.stage` |
| **Type** | `int` |
| **Default** | `0` |
| **Valid Options** | `0` (disabled), `1`, `2`, `3` |
| **Description** | ZeRO optimization stage. Stage 1 partitions optimizer states, Stage 2 adds gradient partitioning, Stage 3 adds parameter partitioning. Higher stages provide more memory savings but may increase communication overhead. |

**Stage Comparison:**

| Stage | What is Partitioned | Memory Savings | Communication Overhead |
|-------|--------------------|--------------------|----------------------|
| 0 | Nothing (standard DP) | 1x | Baseline |
| 1 | Optimizer states | ~4x | Same as DP |
| 2 | Optimizer states + Gradients | ~8x | Same as DP |
| 3 | Optimizer states + Gradients + Parameters | ~N_gpu x | +50% (allgather for forward/backward) |

#### `zero_optimization.allgather_partitions`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.allgather_partitions` |
| **Type** | `bool` |
| **Default** | `true` |
| **Description** | For Stage 3. After each optimizer step, gather updated parameters from all partitions using all-gather instead of all-reduce. All-gather is more efficient for parameter partitioning. |

#### `zero_optimization.allgather_bucket_size`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.allgather_bucket_size` |
| **Type** | `int` |
| **Default** | `500000000` (500M elements ~ 1 GB FP16) |
| **Description** | Bucket size for all-gather operations in Stage 3. Parameters are gathered in buckets of this size. Larger buckets reduce the number of communication rounds but increase peak memory. |

#### `zero_optimization.overlap_comm`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.overlap_comm` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Overlap communication with computation. When enabled, gradient reduction (Stage 2) or parameter all-gather (Stage 3) is overlapped with the backward pass computation, hiding communication latency. |

#### `zero_optimization.reduce_scatter`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.reduce_scatter` |
| **Type** | `bool` |
| **Default** | `true` |
| **Description** | Use reduce-scatter instead of all-reduce for gradient reduction. Reduce-scatter is more efficient for Stage 2 because each process only needs its partition of the gradient. |

#### `zero_optimization.reduce_bucket_size`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.reduce_bucket_size` |
| **Type** | `int` |
| **Default** | `500000000` |
| **Description** | Bucket size for gradient reduction operations. Gradients are bucketed and reduced in chunks of this size. Larger buckets improve communication efficiency but increase memory usage. |

#### `zero_optimization.contiguous_gradients`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.contiguous_gradients` |
| **Type** | `bool` |
| **Default** | `true` |
| **Description** | Store gradients in a contiguous memory buffer instead of individual tensor buffers. This reduces memory fragmentation and improves the efficiency of communication operations. |

#### `zero_optimization.load_from_fp32_weights`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.load_from_fp32_weights` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | For Stage 3. When loading a checkpoint, load FP32 weights and convert them to the appropriate format. This is useful when loading from a non-ZeRO checkpoint into a ZeRO Stage 3 training run. |

#### `zero_optimization.grad_hooks`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.grad_hooks` |
| **Type** | `bool` |
| **Default** | `true` |
| **Description** | Enable gradient hooks for reducing gradients during the backward pass. When enabled, gradient reduction happens as soon as each layer's gradients are computed, enabling overlap with backward computation. |

#### `zero_optimization.round_robin_gradients`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.round_robin_gradients` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | For Stage 2. Assign gradient reduction destinations in a round-robin fashion across processes, rather than each process reducing its own partition. This can improve load balancing. |

### 3.11.2 ZeRO Stage 3 Specific Parameters

#### `zero_optimization.stage3_max_live_parameters`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.stage3_max_live_parameters` |
| **Type** | `int` |
| **Default** | `1000000000` (1 billion) |
| **Description** | Maximum number of parameters that can be resident in GPU memory simultaneously. This controls the parameter cache size. When exceeded, least-recently-used parameters are evicted. |

#### `zero_optimization.stage3_max_reuse_distance`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.stage3_max_reuse_distance` |
| **Type** | `int` |
| **Default** | `1000000000` |
| **Description** | Maximum reuse distance (in number of parameters) for keeping a parameter in the cache. If a parameter is not reused within this distance, it is a candidate for eviction. |

#### `zero_optimization.stage3_prefetch_bucket_size`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.stage3_prefetch_bucket_size` |
| **Type** | `int` |
| **Default** | `50000000` (50M elements) |
| **Description** | Size of prefetch buckets for parameter prefetching in Stage 3. Parameters needed for upcoming layers are prefetched in the background while the current layer is computing. |

#### `zero_optimization.stage3_param_persistence_threshold`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.stage3_param_persistence_threshold` |
| **Type** | `int` |
| **Default** | `100000` (100K elements) |
| **Description** | Parameters with fewer elements than this threshold are not partitioned and remain on all GPUs. This avoids the overhead of partitioning and gathering small parameters (e.g., bias vectors, layer norm parameters). |

#### `zero_optimization.sub_group_size`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.sub_group_size` |
| **Type** | `int` |
| **Default** | `1000000000000` (1 trillion) |
| **Description** | Size of parameter sub-groups for ZeRO Stage 3 parameter fetching. Parameters are grouped into sub-groups for efficient all-gather operations. |

#### `zero_optimization.elastic_checkpoint`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.elastic_checkpoint` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable elastic checkpoints that can be loaded with a different number of GPUs than they were saved with. Useful for elastic training where GPU count may change. |

#### `zero_optimization.stage3_gather_16bit_weights_on_model_save`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.stage3_gather_16bit_weights_on_model_save` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | When saving the model (not checkpoint), gather all 16-bit weights from all GPUs. This produces a complete model state dict suitable for inference. Without this, only the local partition is saved. |

#### `zero_optimization.ignore_unused_parameters`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.ignore_unused_parameters` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Ignore parameters that do not contribute gradients during backward. Set to `true` if your model has parameters that are not used in every forward pass (e.g., tied weights, conditional computation). |

#### `zero_optimization.zero_hpz_partition_size`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.zero_hpz_partition_size` |
| **Type** | `int` |
| **Default** | `0` (disabled) |
| **Description** | ZeRO++ (ZeRO-Infinity Plus) hierarchical parameter partitioning group size. When set to a value > 0, enables hierarchical partitioning within sub-groups of GPUs for reduced communication. |

#### `zero_optimization.zero_quantized_weights`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.zero_quantized_weights` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | ZeRO++ feature. Quantize weights to INT4/INT8 during all-gather communication in Stage 3, reducing communication volume. |

#### `zero_optimization.zero_quantized_gradients`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.zero_quantized_gradients` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | ZeRO++ feature. Quantize gradients during reduce-scatter communication, reducing communication volume for gradient exchange. |

#### `zero_optimization.log_trace_cache_warnings`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.log_trace_cache_warnings` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Log detailed trace information when parameter cache warnings are emitted. Useful for debugging Stage 3 parameter fetching performance. |

#### `zero_optimization.memory_efficient_linear`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.memory_efficient_linear` |
| **Type** | `bool` |
| **Default** | `true` |
| **Description** | Use memory-efficient linear layer implementation for Stage 3. The parameters are gathered just-in-time for the GEMM operation and freed immediately after, reducing peak memory usage. |

#### `zero_optimization.stage3_module_granularity_threshold`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.stage3_module_granularity_threshold` |
| **Type** | `int` |
| **Default** | `0` |
| **Description** | Minimum number of parameters in a module for it to be considered for module-level parameter fetching in Stage 3. Modules smaller than this threshold have their parameters fetched at a coarser granularity. |

#### `zero_optimization.stage3_use_all_reduce_for_fetch_params`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.stage3_use_all_reduce_for_fetch_params` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Use all-reduce instead of all-gather for parameter fetching in Stage 3. All-reduce may be more efficient on some hardware/interconnect configurations. |

#### `zero_optimization.stage3_allgather_sequential`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.stage3_allgather_sequential` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Use sequential (ring-based) all-gather instead of standard all-gather for Stage 3. Sequential all-gather has lower peak memory usage but may be slower. |

#### `zero_optimization.zeropp_loco_param`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.zeropp_loco_param` |
| **Type** | `int` |
| **Default** | `0` |
| **Description** | ZeRO++ LOCO (Low-Communication) parameter. Controls the number of groups for hierarchical communication in ZeRO++. A value of 0 disables LOCO. |

**Full ZeRO Stage 3 example:**

```json
{
    "zero_optimization": {
        "stage": 3,
        "allgather_partitions": true,
        "allgather_bucket_size": 500000000,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 500000000,
        "contiguous_gradients": true,
        "stage3_max_live_parameters": 1000000000,
        "stage3_max_reuse_distance": 1000000000,
        "stage3_prefetch_bucket_size": 50000000,
        "stage3_param_persistence_threshold": 100000,
        "stage3_gather_16bit_weights_on_model_save": true,
        "ignore_unused_parameters": true,
        "memory_efficient_linear": true
    }
}
```

---

## 3.12 Offload Configuration

### 3.12.1 `zero_optimization.offload_param`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.offload_param` |
| **Type** | `object` |
| **Default** | `{"device": "none"}` |
| **Description** | Configuration for offloading model parameters to CPU or NVMe. This enables training models that are larger than GPU memory. |

#### `offload_param.device`

| Field | Value |
|-------|-------|
| **Name** | `offload_param.device` |
| **Type** | `str` |
| **Default** | `"none"` |
| **Valid Options** | `"none"`, `"cpu"`, `"nvme"` |
| **Description** | Target device for parameter offloading. `"cpu"` offloads to system RAM. `"nvme"` offloads to NVMe SSD for models too large even for CPU RAM. |

#### `offload_param.nvme_path`

| Field | Value |
|-------|-------|
| **Name** | `offload_param.nvme_path` |
| **Type** | `str` |
| **Default** | `"/local_nvme"` |
| **Description** | Path to the NVMe mount point for parameter offloading. Required when `device` is `"nvme"`. |

#### `offload_param.pin_memory`

| Field | Value |
|-------|-------|
| **Name** | `offload_param.pin_memory` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Pin offloaded CPU memory for faster GPU-CPU transfers. Pinned memory enables DMA (direct memory access) transfers, which are faster but reduce available system RAM. |

#### `offload_param.buffer_count`

| Field | Value |
|-------|-------|
| **Name** | `offload_param.buffer_count` |
| **Type** | `int` |
| **Default** | `5` |
| **Description** | Number of buffers for parameter swapping. More buffers enable more pipelining but use more memory. |

#### `offload_param.buffer_size`

| Field | Value |
|-------|-------|
| **Name** | `offload_param.buffer_size` |
| **Type** | `int` |
| **Default** | `1000000000` (1 GB) |
| **Description** | Size of each parameter buffer in bytes. Controls the granularity of parameter transfers. |

#### `offload_param.max_in_cpu`

| Field | Value |
|-------|-------|
| **Name** | `offload_param.max_in_cpu` |
| **Type** | `int` |
| **Default** | `1000000000` |
| **Description** | Maximum number of elements to keep in CPU memory when using NVMe offloading. Acts as a CPU cache for NVMe-stored parameters. |

**Example - CPU offload:**

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

**Example - NVMe offload:**

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 5,
            "buffer_size": 1000000000,
            "max_in_cpu": 1000000000
        }
    }
}
```

### 3.12.2 `zero_optimization.offload_optimizer`

| Field | Value |
|-------|-------|
| **Name** | `zero_optimization.offload_optimizer` |
| **Type** | `object` |
| **Default** | `{"device": "none"}` |
| **Description** | Configuration for offloading optimizer states to CPU or NVMe. Optimizer states (first/second moments in Adam) consume 2x the parameter memory and are good candidates for offloading. |

#### `offload_optimizer.device`

| Field | Value |
|-------|-------|
| **Name** | `offload_optimizer.device` |
| **Type** | `str` |
| **Default** | `"none"` |
| **Valid Options** | `"none"`, `"cpu"`, `"nvme"` |
| **Description** | Target device for optimizer state offloading. |

#### `offload_optimizer.nvme_path`

| Field | Value |
|-------|-------|
| **Name** | `offload_optimizer.nvme_path` |
| **Type** | `str` |
| **Default** | `"/local_nvme"` |
| **Description** | NVMe path for optimizer state offloading. |

#### `offload_optimizer.pin_memory`

| Field | Value |
|-------|-------|
| **Name** | `offload_optimizer.pin_memory` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Pin CPU memory for optimizer states. |

#### `offload_optimizer.ratio`

| Field | Value |
|-------|-------|
| **Name** | `offload_optimizer.ratio` |
| **Type** | `float` |
| **Default** | `1.0` |
| **Valid Range** | `0.0` to `1.0` |
| **Description** | Fraction of optimizer states to offload. A ratio of 1.0 offloads everything, while 0.5 offloads half. Useful for partial offloading when CPU RAM is limited. |

#### `offload_optimizer.buffer_count`

| Field | Value |
|-------|-------|
| **Name** | `offload_optimizer.buffer_count` |
| **Type** | `int` |
| **Default** | `4` |
| **Description** | Number of buffers for optimizer state swapping. |

#### `offload_optimizer.fast_init`

| Field | Value |
|-------|-------|
| **Name** | `offload_optimizer.fast_init` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable fast optimizer initialization by skipping initialization of optimizer states. States are lazily initialized during the first optimizer step. |

#### `offload_optimizer.pipeline_read`

| Field | Value |
|-------|-------|
| **Name** | `offload_optimizer.pipeline_read` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Pipeline reading optimizer states from NVMe. Overlaps reading of the next batch of states with computation on the current batch. |

#### `offload_optimizer.pipeline_write`

| Field | Value |
|-------|-------|
| **Name** | `offload_optimizer.pipeline_write` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Pipeline writing optimizer states to NVMe. Overlaps writing updated states with the next forward pass. |

#### `offload_optimizer.super_offload`

| Field | Value |
|-------|-------|
| **Name** | `offload_optimizer.super_offload` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable aggressive optimizer state offloading that minimizes GPU memory usage at the cost of additional CPU computation. |

#### `offload_optimizer.cpuadam_cores_perc`

| Field | Value |
|-------|-------|
| **Name** | `offload_optimizer.cpuadam_cores_perc` |
| **Type** | `float` |
| **Default** | `1.0` |
| **Valid Range** | `0.1` to `1.0` |
| **Description** | Percentage of CPU cores to use for the CPU Adam optimizer. Set to a lower value to leave cores available for other tasks. |

**Example - CPU optimizer offload with ZeRO Stage 2:**

```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true,
            "ratio": 1.0
        }
    }
}
```

**Example - Full NVMe offload (ZeRO-Infinity):**

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 5,
            "buffer_size": 1000000000,
            "max_in_cpu": 1000000000
        },
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "pipeline_read": true,
            "pipeline_write": true,
            "ratio": 1.0
        }
    }
}
```

---

## 3.13 Tensor Parallelism

### 3.13.1 `tensor_parallel`

| Field | Value |
|-------|-------|
| **Name** | `tensor_parallel` |
| **Type** | `object` |
| **Default** | None |
| **Description** | Configuration for tensor parallelism (TP). Tensor parallelism splits individual model layers across GPUs for training models that are too large for a single GPU even with ZeRO. |

#### `tensor_parallel.autotp_size`

| Field | Value |
|-------|-------|
| **Name** | `tensor_parallel.autotp_size` |
| **Type** | `int` |
| **Default** | `1` |
| **Valid Options** | Positive integer, power of 2 recommended |
| **Description** | Automatic tensor parallelism size. When set > 1, DeepSpeed automatically partitions supported model layers across this many GPUs. |

#### `tensor_parallel.preset_model`

| Field | Value |
|-------|-------|
| **Name** | `tensor_parallel.preset_model` |
| **Type** | `str` |
| **Default** | None |
| **Valid Options** | `"llama"`, `"bloom"`, `"chatglm"`, `"mixtral"`, `"deepseek_v2"`, `"qwen2"`, `"phi3"` |
| **Description** | Pre-configured tensor parallelism layout for a specific model architecture. When specified, DeepSpeed uses the known optimal partitioning strategy for the model. |

#### `tensor_parallel.tp_overlap_comm`

| Field | Value |
|-------|-------|
| **Name** | `tensor_parallel.tp_overlap_comm` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Overlap tensor parallelism communication with computation for improved throughput. |

#### `tensor_parallel.partition_config`

| Field | Value |
|-------|-------|
| **Name** | `tensor_parallel.partition_config` |
| **Type** | `object` |
| **Description** | Custom tensor parallelism partition configuration for fine-grained control. |

**Structure:**

```json
{
    "tensor_parallel": {
        "autotp_size": 4,
        "preset_model": "llama",
        "tp_overlap_comm": true,
        "partition_config": {
            "layer_specs": [
                {
                    "name": "self_attention.query_key_value",
                    "partition_type": "column",
                    "partition_dim": 0,
                    "shape": [4096, 4096]
                }
            ],
            "patterns": ["column", "row"],
            "default_partition_type": "column",
            "default_partition_dim": 0
        }
    }
}
```

**Example - LLaMA TP:**

```json
{
    "tensor_parallel": {
        "autotp_size": 4,
        "preset_model": "llama"
    }
}
```

---

## 3.14 AIO Configuration (Async I/O for NVMe)

### 3.14.1 `aio`

| Field | Value |
|-------|-------|
| **Name** | `aio` |
| **Type** | `object` |
| **Default** | Default values shown below |
| **Description** | Configuration for asynchronous I/O operations used in NVMe offloading. These parameters control the libaio-based async I/O engine. |

#### `aio.block_size`

| Field | Value |
|-------|-------|
| **Name** | `aio.block_size` |
| **Type** | `int` |
| **Default** | `1048576` (1 MB) |
| **Description** | Block size for AIO operations in bytes. Larger block sizes improve sequential throughput but increase latency for small transfers. |

#### `aio.queue_depth`

| Field | Value |
|-------|-------|
| **Name** | `aio.queue_depth` |
| **Type** | `int` |
| **Default** | `8` |
| **Description** | Number of outstanding AIO requests (queue depth). Higher values enable more pipelining but increase memory usage. |

#### `aio.thread_count`

| Field | Value |
|-------|-------|
| **Name** | `aio.thread_count` |
| **Type** | `int` |
| **Default** | `4` |
| **Description** | Number of threads for AIO operations. Multiple threads can improve throughput for concurrent read/write operations. |

#### `aio.single_submit`

| Field | Value |
|-------|-------|
| **Name** | `aio.single_submit` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Submit AIO requests one at a time instead of batching. May improve latency at the cost of throughput. |

#### `aio.overlap_events`

| Field | Value |
|-------|-------|
| **Name** | `aio.overlap_events` |
| **Type** | `bool` |
| **Default** | `true` |
| **Description** | Overlap AIO events for improved pipelining. When enabled, multiple AIO operations can be in flight simultaneously. |

**Example:**

```json
{
    "aio": {
        "block_size": 1048576,
        "queue_depth": 16,
        "thread_count": 8,
        "single_submit": false,
        "overlap_events": true
    }
}
```

---

## 3.15 Checkpoint Options

### 3.15.1 `checkpoint`

| Field | Value |
|-------|-------|
| **Name** | `checkpoint` |
| **Type** | `object` |
| **Default** | Default values shown below |
| **Description** | Configuration for checkpoint save/load behavior. |

#### `checkpoint.tag_validation`

| Field | Value |
|-------|-------|
| **Name** | `checkpoint.tag_validation` |
| **Type** | `str` |
| **Default** | `"warn"` |
| **Valid Options** | `"ignore"`, `"warn"`, `"error"` |
| **Description** | Behavior when checkpoint tag (step number) validation fails. `"warn"` logs a warning, `"error"` raises an exception, `"ignore"` skips validation. |

#### `checkpoint.load_universal`

| Field | Value |
|-------|-------|
| **Name** | `checkpoint.load_universal` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Load checkpoint in universal format. Universal checkpoints can be loaded with different GPU counts or ZeRO configurations than they were saved with. |

#### `checkpoint.use_node_local_storage`

| Field | Value |
|-------|-------|
| **Name** | `checkpoint.use_node_local_storage` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Save checkpoints to node-local storage instead of shared storage. Each node saves its own checkpoint independently. This avoids the bottleneck of writing to shared storage. |

#### `checkpoint.parallel_write`

| Field | Value |
|-------|-------|
| **Name** | `checkpoint.parallel_write` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable parallel checkpoint writing where multiple processes write their portions of the checkpoint simultaneously. |

**Example:**

```json
{
    "checkpoint": {
        "tag_validation": "warn",
        "load_universal": false,
        "use_node_local_storage": true,
        "parallel_write": true
    }
}
```

---

## 3.16 Data Types

### 3.16.1 `grad_accum_dtype`

| Field | Value |
|-------|-------|
| **Name** | `grad_accum_dtype` |
| **Type** | `str` |
| **Default** | Same as training dtype |
| **Valid Options** | `"fp32"`, `"fp16"`, `"bf16"` |
| **Description** | Data type for gradient accumulation buffers. By default, accumulated gradients match the training dtype. Setting to `"fp32"` improves numerical accuracy when accumulating many micro-batches. |

**Example:**

```json
{
    "grad_accum_dtype": "fp32"
}
```

---

## 3.17 Logging Configuration

### 3.17.1 `steps_per_print`

| Field | Value |
|-------|-------|
| **Name** | `steps_per_print` |
| **Type** | `int` |
| **Default** | `1` |
| **Description** | Number of training steps between progress log messages. Logs include loss, learning rate, throughput, and other metrics. |

### 3.17.2 `wall_clock_breakdown`

| Field | Value |
|-------|-------|
| **Name** | `wall_clock_breakdown` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable detailed wall-clock timing of forward pass, backward pass, gradient reduction, optimizer step, and other operations. Useful for performance analysis. |

### 3.17.3 `dump_state`

| Field | Value |
|-------|-------|
| **Name** | `dump_state` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Dump the internal state of the DeepSpeed engine (memory usage, parameter partitioning, etc.) during initialization. Useful for debugging. |

**Example:**

```json
{
    "steps_per_print": 10,
    "wall_clock_breakdown": true,
    "dump_state": false
}
```

---

## 3.18 Autotuning Configuration

### 3.18.1 `autotuning`

| Field | Value |
|-------|-------|
| **Name** | `autotuning` |
| **Type** | `object` |
| **Default** | `{"enabled": false}` |
| **Description** | Configuration for automatic hyperparameter tuning. DeepSpeed can automatically search for optimal batch sizes, ZeRO stages, and other parameters. |

#### `autotuning.enabled`

| Field | Value |
|-------|-------|
| **Name** | `autotuning.enabled` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable autotuning. |

#### `autotuning.start_step`

| Field | Value |
|-------|-------|
| **Name** | `autotuning.start_step` |
| **Type** | `int` |
| **Default** | `10` |
| **Description** | Step at which to start autotuning measurements. Allows skipping warmup steps. |

#### `autotuning.end_step`

| Field | Value |
|-------|-------|
| **Name** | `autotuning.end_step` |
| **Type** | `int` |
| **Default** | `100` |
| **Description** | Step at which to stop autotuning measurements. |

#### `autotuning.metric`

| Field | Value |
|-------|-------|
| **Name** | `autotuning.metric` |
| **Type** | `str` |
| **Default** | `"throughput"` |
| **Valid Options** | `"throughput"`, `"latency"` |
| **Description** | Metric to optimize during autotuning. |

#### `autotuning.metric_path`

| Field | Value |
|-------|-------|
| **Name** | `autotuning.metric_path` |
| **Type** | `str` |
| **Default** | `None` |
| **Description** | Path to a custom metric function for autotuning. |

#### `autotuning.tuner_type`

| Field | Value |
|-------|-------|
| **Name** | `autotuning.tuner_type` |
| **Type** | `str` |
| **Default** | `"modelsize"` |
| **Valid Options** | `"modelsize"`, `"gridsearch"`, `"random"` |
| **Description** | Tuning strategy. `"modelsize"` uses heuristics based on model size, `"gridsearch"` tries all combinations, `"random"` samples randomly. |

#### `autotuning.results_dir`

| Field | Value |
|-------|-------|
| **Name** | `autotuning.results_dir` |
| **Type** | `str` |
| **Default** | `"autotuning_results"` |
| **Description** | Directory to store autotuning results. |

#### `autotuning.exps_dir`

| Field | Value |
|-------|-------|
| **Name** | `autotuning.exps_dir` |
| **Type** | `str` |
| **Default** | `"autotuning_exps"` |
| **Description** | Directory for autotuning experiment outputs. |

**Example:**

```json
{
    "autotuning": {
        "enabled": true,
        "start_step": 10,
        "end_step": 50,
        "metric": "throughput",
        "tuner_type": "modelsize",
        "results_dir": "./autotuning_results"
    }
}
```

---

## 3.19 Flops Profiler Configuration

### 3.19.1 `flops_profiler`

| Field | Value |
|-------|-------|
| **Name** | `flops_profiler` |
| **Type** | `object` |
| **Default** | `{"enabled": false}` |
| **Description** | Configuration for the FLOPS profiler, which measures the floating-point operations per second of the model. |

#### `flops_profiler.enabled`

| Field | Value |
|-------|-------|
| **Name** | `flops_profiler.enabled` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable FLOPS profiling. |

#### `flops_profiler.profile_step`

| Field | Value |
|-------|-------|
| **Name** | `flops_profiler.profile_step` |
| **Type** | `int` |
| **Default** | `1` |
| **Description** | Training step at which to measure FLOPS. |

#### `flops_profiler.module_depth`

| Field | Value |
|-------|-------|
| **Name** | `flops_profiler.module_depth` |
| **Type** | `int` |
| **Default** | `-1` (all depths) |
| **Description** | Maximum depth for module-level FLOPS profiling. `-1` profiles all modules. |

#### `flops_profiler.top_modules`

| Field | Value |
|-------|-------|
| **Name** | `flops_profiler.top_modules` |
| **Type** | `int` |
| **Default** | `3` |
| **Description** | Number of top modules to display in the FLOPS report. |

#### `flops_profiler.detailed`

| Field | Value |
|-------|-------|
| **Name** | `flops_profiler.detailed` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Print detailed FLOPS report for each module. |

#### `flops_profiler.output_file`

| Field | Value |
|-------|-------|
| **Name** | `flops_profiler.output_file` |
| **Type** | `str` |
| **Default** | `None` |
| **Description** | File to write FLOPS profile results. If None, prints to stdout. |

**Example:**

```json
{
    "flops_profiler": {
        "enabled": true,
        "profile_step": 10,
        "module_depth": -1,
        "top_modules": 5,
        "detailed": true,
        "output_file": "flops_profile.txt"
    }
}
```

---

## 3.20 Activation Checkpointing

### 3.20.1 `activation_checkpointing`

| Field | Value |
|-------|-------|
| **Name** | `activation_checkpointing` |
| **Type** | `object` |
| **Default** | `{"partition_activations": false}` |
| **Description** | Configuration for activation checkpointing (also called gradient checkpointing). Activation checkpointing reduces memory by recomputing activations during the backward pass instead of storing them. |

#### `activation_checkpointing.partition_activations`

| Field | Value |
|-------|-------|
| **Name** | `activation_checkpointing.partition_activations` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable activation partitioning. When enabled, activations are partitioned across GPUs and only gathered when needed for backward computation. |

#### `activation_checkpointing.cpu_checkpointing`

| Field | Value |
|-------|-------|
| **Name** | `activation_checkpointing.cpu_checkpointing` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Offload activation checkpoints to CPU memory. Reduces GPU memory at the cost of CPU-GPU transfer time. |

#### `activation_checkpointing.contiguous_memory_optimization`

| Field | Value |
|-------|-------|
| **Name** | `activation_checkpointing.contiguous_memory_optimization` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Store activation checkpoints in contiguous memory buffers to reduce fragmentation. |

#### `activation_checkpointing.number_checkpoints`

| Field | Value |
|-------|-------|
| **Name** | `activation_checkpointing.number_checkpoints` |
| **Type** | `int` |
| **Default** | `None` |
| **Description** | Total number of activation checkpoints across the model. If not specified, all transformer layers are checkpointed. |

#### `activation_checkpointing.checkpoint_in_cpu`

| Field | Value |
|-------|-------|
| **Name** | `activation_checkpointing.checkpoint_in_cpu` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Store activation checkpoints in CPU memory. Same as `cpu_checkpointing`. |

#### `activation_checkpointing.synchronize_checkpoint_boundary`

| Field | Value |
|-------|-------|
| **Name** | `activation_checkpointing.synchronize_checkpoint_boundary` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Synchronize processes at checkpoint boundaries. Ensures all processes checkpoint at the same layers. |

#### `activation_checkpointing.profile`

| Field | Value |
|-------|-------|
| **Name** | `activation_checkpointing.profile` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Profile activation checkpointing operations for performance analysis. |

**Example:**

```json
{
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": true,
        "contiguous_memory_optimization": true,
        "number_checkpoints": 24,
        "synchronize_checkpoint_boundary": true
    }
}
```

---

## 3.21 Sparse Attention Configuration

### 3.21.1 `sparse_attention`

| Field | Value |
|-------|-------|
| **Name** | `sparse_attention` |
| **Type** | `object` |
| **Default** | `{"enabled": false}` |
| **Description** | Configuration for DeepSpeed sparse attention. Sparse attention reduces the memory and compute cost of attention by using block-sparse patterns instead of full attention. |

#### `sparse_attention.enabled`

| Field | Value |
|-------|-------|
| **Name** | `sparse_attention.enabled` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable sparse attention. |

#### `sparse_attention.mode`

| Field | Value |
|-------|-------|
| **Name** | `sparse_attention.mode` |
| **Type** | `str` |
| **Default** | `"fixed"` |
| **Valid Options** | `"fixed"`, `"variable"`, `"bigbird"`, `"longformer"` |
| **Description** | Sparse attention pattern mode. `"fixed"` uses fixed block-sparse patterns, `"variable"` supports variable-length sequences. |

#### `sparse_attention.block`

| Field | Value |
|-------|-------|
| **Name** | `sparse_attention.block` |
| **Type** | `int` |
| **Default** | `16` |
| **Description** | Block size for sparse attention computation. |

**Example:**

```json
{
    "sparse_attention": {
        "enabled": true,
        "mode": "fixed",
        "block": 16
    }
}
```

---

## 3.22 Curriculum Learning

### 3.22.1 `curriculum_learning`

| Field | Value |
|-------|-------|
| **Name** | `curriculum_learning` |
| **Type** | `object` |
| **Default** | `{"enabled": false}` |
| **Description** | Configuration for curriculum learning, which gradually increases sequence length during training for faster convergence. |

#### `curriculum_learning.enabled`

| Field | Value |
|-------|-------|
| **Name** | `curriculum_learning.enabled` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable curriculum learning. |

#### `curriculum_learning.curriculum_type`

| Field | Value |
|-------|-------|
| **Name** | `curriculum_learning.curriculum_type` |
| **Type** | `str` |
| **Default** | `"seqlen"` |
| **Valid Options** | `"seqlen"` |
| **Description** | Type of curriculum. `"seqlen"` gradually increases sequence length. |

#### `curriculum_learning.min_difficulty`

| Field | Value |
|-------|-------|
| **Name** | `curriculum_learning.min_difficulty` |
| **Type** | `int` |
| **Default** | `1` |
| **Description** | Starting difficulty level (e.g., minimum sequence length). |

#### `curriculum_learning.max_difficulty`

| Field | Value |
|-------|-------|
| **Name** | `curriculum_learning.max_difficulty` |
| **Type** | `int` |
| **Default** | `None` (maximum sequence length) |
| **Description** | Maximum difficulty level (e.g., maximum sequence length). |

#### `curriculum_learning.schedule_type`

| Field | Value |
|-------|-------|
| **Name** | `curriculum_learning.schedule_type` |
| **Type** | `str` |
| **Default** | `"fixed_linear"` |
| **Valid Options** | `"fixed_linear"`, `"fixed_discrete"`, `"custom"` |
| **Description** | Schedule type for difficulty progression. |

#### `curriculum_learning.schedule_config`

| Field | Value |
|-------|-------|
| **Name** | `curriculum_learning.schedule_config` |
| **Type** | `object` |
| **Default** | `{}` |
| **Description** | Schedule-specific configuration parameters. |

**Example:**

```json
{
    "curriculum_learning": {
        "enabled": true,
        "curriculum_type": "seqlen",
        "min_difficulty": 128,
        "max_difficulty": 2048,
        "schedule_type": "fixed_linear",
        "schedule_config": {
            "total_curriculum_step": 10000
        }
    }
}
```

---

## 3.23 Monitor Configuration

### 3.23.1 TensorBoard

| Field | Value |
|-------|-------|
| **Name** | `tensorboard` |
| **Type** | `object` |
| **Default** | `{"enabled": false}` |

#### `tensorboard.enabled`

| Field | Value |
|-------|-------|
| **Name** | `tensorboard.enabled` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable TensorBoard logging. |

#### `tensorboard.output_path`

| Field | Value |
|-------|-------|
| **Name** | `tensorboard.output_path` |
| **Type** | `str` |
| **Default** | `""` |
| **Description** | Output directory for TensorBoard logs. |

#### `tensorboard.job_name`

| Field | Value |
|-------|-------|
| **Name** | `tensorboard.job_name` |
| **Type** | `str` |
| **Default** | `"DeepSpeedJob"` |
| **Description** | Job name for TensorBoard run identification. |

**Example:**

```json
{
    "tensorboard": {
        "enabled": true,
        "output_path": "./tensorboard_logs",
        "job_name": "llama-finetune"
    }
}
```

### 3.23.2 Weights & Biases

| Field | Value |
|-------|-------|
| **Name** | `wandb` |
| **Type** | `object` |
| **Default** | `{"enabled": false}` |

#### `wandb.enabled`

| Field | Value |
|-------|-------|
| **Name** | `wandb.enabled` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable Weights & Biases logging. |

#### `wandb.project`

| Field | Value |
|-------|-------|
| **Name** | `wandb.project` |
| **Type** | `str` |
| **Default** | `"deepspeed"` |
| **Description** | W&B project name. |

#### `wandb.team`

| Field | Value |
|-------|-------|
| **Name** | `wandb.team` |
| **Type** | `str` |
| **Default** | `None` |
| **Description** | W&B team/entity name. |

#### `wandb.group`

| Field | Value |
|-------|-------|
| **Name** | `wandb.group` |
| **Type** | `str` |
| **Default** | `None` |
| **Description** | W&B group name for grouping related runs. |

#### `wandb.name`

| Field | Value |
|-------|-------|
| **Name** | `wandb.name` |
| **Type** | `str` |
| **Default** | `None` |
| **Description** | W&B run name. |

**Example:**

```json
{
    "wandb": {
        "enabled": true,
        "project": "llm-training",
        "team": "my-team",
        "group": "llama-experiments",
        "name": "zero3-bf16-run1"
    }
}
```

### 3.23.3 Comet ML

| Field | Value |
|-------|-------|
| **Name** | `comet` |
| **Type** | `object` |
| **Default** | `{"enabled": false}` |

#### `comet.enabled`

| Field | Value |
|-------|-------|
| **Name** | `comet.enabled` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable Comet ML logging. |

#### `comet.project_name`

| Field | Value |
|-------|-------|
| **Name** | `comet.project_name` |
| **Type** | `str` |
| **Default** | `None` |
| **Description** | Comet project name. |

#### `comet.experiment_name`

| Field | Value |
|-------|-------|
| **Name** | `comet.experiment_name` |
| **Type** | `str` |
| **Default** | `None` |
| **Description** | Comet experiment name. |

#### `comet.workspace`

| Field | Value |
|-------|-------|
| **Name** | `comet.workspace` |
| **Type** | `str` |
| **Default** | `None` |
| **Description** | Comet workspace name. |

**Example:**

```json
{
    "comet": {
        "enabled": true,
        "project_name": "deepspeed-training",
        "experiment_name": "zero2-experiment-1"
    }
}
```

### 3.23.4 CSV Monitor

| Field | Value |
|-------|-------|
| **Name** | `csv_monitor` |
| **Type** | `object` |
| **Default** | `{"enabled": false}` |

#### `csv_monitor.enabled`

| Field | Value |
|-------|-------|
| **Name** | `csv_monitor.enabled` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable CSV file logging of training metrics. |

#### `csv_monitor.output_path`

| Field | Value |
|-------|-------|
| **Name** | `csv_monitor.output_path` |
| **Type** | `str` |
| **Default** | `"."` |
| **Description** | Directory for CSV output files. |

#### `csv_monitor.job_name`

| Field | Value |
|-------|-------|
| **Name** | `csv_monitor.job_name` |
| **Type** | `str` |
| **Default** | `"DeepSpeedJob"` |
| **Description** | Job name for CSV file naming. |

**Example:**

```json
{
    "csv_monitor": {
        "enabled": true,
        "output_path": "./training_logs",
        "job_name": "llama-pretrain"
    }
}
```

---

## 3.24 Compression Configuration

### 3.24.1 `compression`

| Field | Value |
|-------|-------|
| **Name** | `compression` |
| **Type** | `object` |
| **Default** | `{"enabled": false}` |
| **Description** | Configuration for model compression during training, including quantization, pruning, and knowledge distillation. |

#### `compression.enabled`

| Field | Value |
|-------|-------|
| **Name** | `compression.enabled` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable compression-aware training. |

**Example:**

```json
{
    "compression": {
        "enabled": true,
        "weight_quantization": {
            "shared_parameters": {
                "quantizer_kernel": "ds_quantize",
                "schedule_offset": 0,
                "quantize_groups": 64,
                "quantize_verbose": false,
                "quantization_type": "symmetric",
                "rounding": "nearest",
                "num_bits": 8
            },
            "different_groups": {
                "wq": {
                    "params": {
                        "start_bits": 8,
                        "target_bits": 4
                    },
                    "modules": ["attention.self", "attention.output", "intermediate", "output"]
                }
            }
        },
        "activation_quantization": {
            "shared_parameters": {
                "quantizer_kernel": "ds_quantize",
                "schedule_offset": 100,
                "quantize_groups": 64,
                "quantize_verbose": false,
                "quantization_type": "symmetric",
                "rounding": "nearest",
                "num_bits": 8
            },
            "different_groups": {
                "aq": {
                    "params": {
                        "start_bits": 8,
                        "target_bits": 8
                    },
                    "modules": ["attention.self", "attention.output", "intermediate", "output"]
                }
            }
        },
        "pruning": {
            "shared_parameters": {
                "pruner_kernel": "l1_norm",
                "schedule_offset": 0,
                "pruner_verbose": false
            },
            "different_groups": {
                "sp": {
                    "params": {
                        "start_density": 1.0,
                        "target_density": 0.5
                    },
                    "modules": ["attention.self", "attention.output", "intermediate", "output"]
                }
            }
        }
    }
}
```

---

## 3.25 Elasticity Configuration

### 3.25.1 `elasticity`

| Field | Value |
|-------|-------|
| **Name** | `elasticity` |
| **Type** | `object` |
| **Default** | `{"enabled": false}` |
| **Description** | Configuration for elastic training, which allows training to adapt to changing GPU counts dynamically. |

#### `elasticity.enabled`

| Field | Value |
|-------|-------|
| **Name** | `elasticity.enabled` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable elastic training. |

#### `elasticity.max_acceptable_batch_size`

| Field | Value |
|-------|-------|
| **Name** | `elasticity.max_acceptable_batch_size` |
| **Type** | `int` |
| **Default** | `None` |
| **Description** | Maximum batch size that is acceptable for training quality. |

#### `elasticity.min_acceptable_batch_size`

| Field | Value |
|-------|-------|
| **Name** | `elasticity.min_acceptable_batch_size` |
| **Type** | `int` |
| **Default** | `1` |
| **Description** | Minimum batch size that is acceptable for training quality. |

#### `elasticity.micro_batch_sizes`

| Field | Value |
|-------|-------|
| **Name** | `elasticity.micro_batch_sizes` |
| **Type** | `list[int]` |
| **Default** | `[]` |
| **Description** | List of acceptable micro-batch sizes to search over. |

#### `elasticity.num_gpus_per_node`

| Field | Value |
|-------|-------|
| **Name** | `elasticity.num_gpus_per_node` |
| **Type** | `int` |
| **Default** | `None` |
| **Description** | Number of GPUs per node for elastic training calculations. |

**Example:**

```json
{
    "elasticity": {
        "enabled": true,
        "max_acceptable_batch_size": 2048,
        "min_acceptable_batch_size": 128,
        "micro_batch_sizes": [2, 4, 8, 16],
        "num_gpus_per_node": 8
    }
}
```

---

## 3.26 Communication Logging

### 3.26.1 `communication_logging`

| Field | Value |
|-------|-------|
| **Name** | `communication_logging` |
| **Type** | `object` |
| **Default** | `{"enabled": false}` |
| **Description** | Configuration for logging communication operations and their timing. |

#### `communication_logging.enabled`

| Field | Value |
|-------|-------|
| **Name** | `communication_logging.enabled` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable communication logging. |

**Example:**

```json
{
    "communication_logging": {
        "enabled": true
    }
}
```

---

## 3.27 DeepCompile Configuration

### 3.27.1 `deepcompile`

| Field | Value |
|-------|-------|
| **Name** | `deepcompile` |
| **Type** | `object` |
| **Default** | `{"enabled": false}` |
| **Description** | Configuration for DeepCompile, which provides compiler-based optimization of the training computation graph. |

#### `deepcompile.enabled`

| Field | Value |
|-------|-------|
| **Name** | `deepcompile.enabled` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable DeepCompile. |

#### `deepcompile.free_activation`

| Field | Value |
|-------|-------|
| **Name** | `deepcompile.free_activation` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable the free activation pass that aggressively frees activation memory after it is consumed. |

#### `deepcompile.offload_activation`

| Field | Value |
|-------|-------|
| **Name** | `deepcompile.offload_activation` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable the activation offload pass that moves activations to CPU memory during forward and brings them back during backward. |

#### `deepcompile.offload_opt_states`

| Field | Value |
|-------|-------|
| **Name** | `deepcompile.offload_opt_states` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Offload optimizer states to CPU memory when using DeepCompile. |

#### `deepcompile.double_buffer`

| Field | Value |
|-------|-------|
| **Name** | `deepcompile.double_buffer` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable double buffering for overlapping computation and memory transfers. |

#### `deepcompile.symmetric_memory`

| Field | Value |
|-------|-------|
| **Name** | `deepcompile.symmetric_memory` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable symmetric memory optimization for improved memory layout. |

#### `deepcompile.passes`

| Field | Value |
|-------|-------|
| **Name** | `deepcompile.passes` |
| **Type** | `list[str]` |
| **Default** | `[]` |
| **Description** | List of custom compilation passes to apply. |

**Example:**

```json
{
    "deepcompile": {
        "enabled": true,
        "free_activation": true,
        "offload_activation": true,
        "offload_opt_states": false,
        "double_buffer": true,
        "symmetric_memory": true,
        "passes": []
    }
}
```

---

## 3.28 Nebula Configuration

### 3.28.1 `nebula`

| Field | Value |
|-------|-------|
| **Name** | `nebula` |
| **Type** | `object` |
| **Default** | `{"enabled": false}` |
| **Description** | Configuration for Nebula asynchronous checkpointing, which reduces checkpoint I/O overhead by writing checkpoints in the background. |

#### `nebula.enabled`

| Field | Value |
|-------|-------|
| **Name** | `nebula.enabled` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable Nebula async checkpointing. |

#### `nebula.persistent_storage_path`

| Field | Value |
|-------|-------|
| **Name** | `nebula.persistent_storage_path` |
| **Type** | `str` |
| **Default** | `None` |
| **Description** | Path for persistent checkpoint storage. |

#### `nebula.persistent_timer_interval`

| Field | Value |
|-------|-------|
| **Name** | `nebula.persistent_timer_interval` |
| **Type** | `int` |
| **Default** | `30` |
| **Description** | Timer interval in seconds for persistent storage sync. |

**Example:**

```json
{
    "nebula": {
        "enabled": true,
        "persistent_storage_path": "/shared/checkpoints",
        "persistent_timer_interval": 30
    }
}
```

---

## 3.29 DataStates Configuration

### 3.29.1 `data_states`

| Field | Value |
|-------|-------|
| **Name** | `data_states` |
| **Type** | `object` |
| **Default** | `{"enabled": false}` |
| **Description** | Configuration for DataStates distributed checkpointing. |

#### `data_states.enabled`

| Field | Value |
|-------|-------|
| **Name** | `data_states.enabled` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable DataStates checkpointing. |

#### `data_states.path`

| Field | Value |
|-------|-------|
| **Name** | `data_states.path` |
| **Type** | `str` |
| **Default** | `None` |
| **Description** | Path for DataStates checkpoint storage. |

**Example:**

```json
{
    "data_states": {
        "enabled": true,
        "path": "/shared/datastates"
    }
}
```

---

## 3.30 ZenFlow Configuration

### 3.30.1 `zenflow`

| Field | Value |
|-------|-------|
| **Name** | `zenflow` |
| **Type** | `object` |
| **Default** | `{"enabled": false}` |
| **Description** | Configuration for ZenFlow, an optimized training pipeline that improves throughput through advanced scheduling and memory management. |

#### `zenflow.enabled`

| Field | Value |
|-------|-------|
| **Name** | `zenflow.enabled` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable ZenFlow optimizations. |

**Example:**

```json
{
    "zenflow": {
        "enabled": true
    }
}
```

---

## 3.31 Pipeline Parallelism Configuration

### 3.31.1 `pipeline`

| Field | Value |
|-------|-------|
| **Name** | `pipeline` |
| **Type** | `object` |
| **Default** | `{"enabled": false}` |
| **Description** | Configuration for pipeline parallelism. When enabled, the model is split into stages that are placed on different GPUs. |

#### `pipeline.enabled`

| Field | Value |
|-------|-------|
| **Name** | `pipeline.enabled` |
| **Type** | `bool` |
| **Default** | `false` |
| **Description** | Enable pipeline parallelism. Requires the model to be wrapped as a `PipelineModule`. |

#### `pipeline.parallel_size`

| Field | Value |
|-------|-------|
| **Name** | `pipeline.parallel_size` |
| **Type** | `int` |
| **Default** | `1` |
| **Description** | Number of pipeline stages (devices to split the model across). |

#### `pipeline.micro_batches`

| Field | Value |
|-------|-------|
| **Name** | `pipeline.micro_batches` |
| **Type** | `int` |
| **Default** | `gradient_accumulation_steps` |
| **Description** | Number of micro-batches for pipeline scheduling. |

#### `pipeline.micro_batch_size`

| Field | Value |
|-------|-------|
| **Name** | `pipeline.micro_batch_size` |
| **Type** | `int` |
| **Default** | `train_micro_batch_size_per_gpu` |
| **Description** | Size of each micro-batch in the pipeline. |

#### `pipeline.scale_loss`

| Field | Value |
|-------|-------|
| **Name** | `pipeline.scale_loss` |
| **Type** | `float` |
| **Default** | `1.0` |
| **Description** | Scaling factor for pipeline parallel loss. |

#### `pipeline.gradient_accumulation_dtype`

| Field | Value |
|-------|-------|
| **Name** | `pipeline.gradient_accumulation_dtype` |
| **Type** | `str` |
| **Default** | `"fp32"` |
| **Description** | Data type for gradient accumulation in pipeline parallel. |

**Example:**

```json
{
    "pipeline": {
        "enabled": true,
        "parallel_size": 4,
        "micro_batches": 8,
        "micro_batch_size": 4,
        "scale_loss": 1.0
    }
}
```

---

## 3.32 Complete Configuration Examples

### 3.32.1 Minimal ZeRO Stage 2 Configuration

```json
{
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2
    }
}
```

### 3.32.2 ZeRO Stage 3 with CPU Offload

```json
{
    "train_batch_size": 128,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "wall_clock_breakdown": true
}
```

### 3.32.3 ZeRO-Infinity (Full NVMe Offload)

```json
{
    "train_batch_size": 64,
    "gradient_accumulation_steps": 16,
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
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 5,
            "buffer_size": 1000000000,
            "max_in_cpu": 1000000000
        },
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "pipeline_read": true,
            "pipeline_write": true,
            "ratio": 1.0
        },
        "aio": {
            "block_size": 1048576,
            "queue_depth": 16,
            "thread_count": 8,
            "single_submit": false,
            "overlap_events": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "wall_clock_breakdown": true
}
```

### 3.32.4 HuggingFace Trainer Integration

```json
{
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

> Note: The `"auto"` value is specific to HuggingFace Trainer integration. DeepSpeed fills in these values from the HuggingFace `TrainingArguments`.

### 3.32.5 Inference Configuration

```json
{
    "tensor_parallel": {
        "tp_size": 4
    },
    "dtype": "bf16",
    "replace_with_kernel_inject": true,
    "max_out_tokens": 2048
}
```
