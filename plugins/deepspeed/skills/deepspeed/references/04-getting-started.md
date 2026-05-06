# DeepSpeed Reference - Chapter 4: Getting Started

This chapter walks through the fundamentals of using DeepSpeed, from writing your first training script to launching distributed training, understanding the training loop, selecting ZeRO stages, and integrating with popular frameworks like HuggingFace.

---

## 4.1 First Training Script

### 4.1.1 Minimal Example

The following is a complete, minimal DeepSpeed training script:

```python
# train.py
import torch
import deepspeed

# 1. Define the model
class SimpleModel(torch.nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * 4),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size * 4, hidden_size),
            torch.nn.LayerNorm(hidden_size),
        )

    def forward(self, x):
        return self.net(x)

# 2. Create model and dataset
model = SimpleModel(hidden_size=1024)

# Simple random dataset
class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, size=10000, hidden_size=1024):
        self.size = size
        self.hidden_size = hidden_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randn(self.hidden_size), torch.randn(self.hidden_size)

dataset = RandomDataset()

# 3. DeepSpeed configuration
ds_config = {
    "train_batch_size": 64,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        }
    },
    "fp16": {
        "enabled": True,
    },
    "zero_optimization": {
        "stage": 2,
    },
}

# 4. Initialize DeepSpeed
model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    training_data=dataset,
    config_params=ds_config,
)

# 5. Training loop
for step, (inputs, targets) in enumerate(train_dataloader):
    inputs = inputs.cuda()
    targets = targets.cuda()

    # Forward pass
    outputs = model_engine(inputs)
    loss = torch.nn.functional.mse_loss(outputs, targets)

    # Backward pass
    model_engine.backward(loss)

    # Optimizer step
    model_engine.step()

    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.6f}")

    if step >= 100:
        break
```

**Launch the script:**

```bash
# Single GPU
deepspeed --num_gpus=1 train.py

# Multi-GPU (4 GPUs)
deepspeed --num_gpus=4 train.py
```

### 4.1.2 What Happens During `deepspeed.initialize()`

When `deepspeed.initialize()` is called, the following happens:

1. **Configuration parsing**: The `ds_config` dictionary is parsed into a `DeepSpeedConfig` object.
2. **Distributed initialization**: The distributed process group is initialized (if not already done).
3. **Engine selection**: Based on the configuration, the appropriate engine is created:
   - `PipelineEngine` if pipeline parallelism is enabled
   - `DeepSpeedEngine` for standard training
4. **ZeRO initialization**: If ZeRO is configured, the appropriate ZeRO stage handler is initialized:
   - Stage 1: Wraps optimizer with `DeepSpeedZeroOptimizer` (partitions optimizer states)
   - Stage 2: Adds gradient partitioning
   - Stage 3: Adds parameter partitioning via `DeepSpeedZeroEngine`
5. **Mixed precision setup**: FP16/BF16/AMP handlers are configured.
6. **DataLoader creation**: If `training_data` is provided, a DeepSpeed-aware DataLoader is created with proper distributed sampling.
7. **Optimizer creation**: If no optimizer is provided, DeepSpeed creates one based on the config.
8. **Learning rate scheduler setup**: If a scheduler is configured, it is created and attached.
9. **Memory reporting**: Memory usage statistics are logged if `dump_state` is enabled.

---

## 4.2 `deepspeed.initialize()` API Reference

### 4.2.1 Function Signature

```python
def initialize(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    args: Optional[argparse.Namespace] = None,
    config_params: Optional[Union[dict, str]] = None,
    dist_init_backend: str = 'nccl',
    model_parameters: Optional[Iterable] = None,
    training_data: Optional[Union[Dataset, DataLoader]] = None,
    lr_scheduler: Optional[object] = None,
    mpu: Optional[object] = None,
    dist_init_required: Optional[bool] = None,
    config: Optional[str] = None,
    collate_fn: Optional[callable] = None,
) -> Tuple:
```

### 4.2.2 Parameter Details

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | `torch.nn.Module` | Yes | The PyTorch model to train. Must be on the correct device before initialization. |
| `optimizer` | `torch.optim.Optimizer` | No | A pre-created optimizer. If None, DeepSpeed creates one from config. |
| `args` | `argparse.Namespace` | No | Command-line arguments. Used by the launcher for distributed setup. |
| `config_params` | `dict` or `str` | No | DeepSpeed config as a dict or path to JSON file. Alternative to `config`. |
| `dist_init_backend` | `str` | No | Backend for `torch.distributed.init_process_group`. Default: 'nccl'. |
| `model_parameters` | `Iterable[Parameter]` | No | Model parameters for optimizer creation. Required if `optimizer` is None. |
| `training_data` | `Dataset` or `DataLoader` | No | Training dataset. DeepSpeed wraps it in a distributed DataLoader. |
| `lr_scheduler` | `object` | No | Learning rate scheduler with a `step()` method. |
| `mpu` | `object` | No | Model parallelism utility (for Megatron integration). Must have methods: `get_model_parallel_rank/group/world_size`. |
| `dist_init_required` | `bool` | No | Whether to call `torch.distributed.init_process_group`. Default: auto-detect. |
| `config` | `str` | No | Path to `ds_config.json`. Alternative to `config_params`. |
| `collate_fn` | `callable` | No | Collate function for the DataLoader. |

### 4.2.3 Return Value

Returns a 4-tuple:

```
(engine, optimizer, training_dataloader, lr_scheduler)
```

| Return Value | Type | Description |
|-------------|------|-------------|
| `engine` | `DeepSpeedEngine` (or subclass) | The DeepSpeed engine wrapping the model. This replaces the original model. |
| `optimizer` | `DeepSpeedOptimizer` | The wrapped optimizer. Use `engine.step()` instead of `optimizer.step()`. |
| `training_dataloader` | `DataLoader` | The distributed DataLoader (if `training_data` was provided). |
| `lr_scheduler` | `object` | The configured LR scheduler (if provided or configured). |

### 4.2.4 Important Notes

- After `deepspeed.initialize()`, use `engine` instead of `model` for forward/backward/step operations.
- Do not call `optimizer.step()` directly. Use `engine.step()` instead, which handles gradient accumulation, optimizer steps, and LR scheduling.
- The `model` object is modified in place. Do not use the original model after initialization.
- DeepSpeed handles `torch.cuda.set_device()` automatically based on `LOCAL_RANK`.

---

## 4.3 ds_config.json Creation

### 4.3.1 Creating the Configuration File

You can create the configuration as a Python dictionary or a JSON file:

**Python dictionary (recommended for scripts):**

```python
ds_config = {
    "train_batch_size": 64,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        }
    },
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": 2},
}
```

**JSON file (recommended for HuggingFace or reproducibility):**

```json
{
    "train_batch_size": 64,
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

### 4.3.2 Configuration Best Practices

1. **Start simple**: Begin with ZeRO Stage 2 and FP16. Add complexity as needed.
2. **Set `gradient_accumulation_steps` explicitly**: This gives you fine control over memory vs throughput.
3. **Use `"auto"` for batch size with HuggingFace**: Let HuggingFace calculate batch sizes from `TrainingArguments`.
4. **Enable `overlap_comm` for Stage 2/3**: Overlapping communication with computation is almost always beneficial.
5. **Set `contiguous_gradients: true`**: Reduces memory fragmentation.
6. **For ZeRO Stage 3, enable `ignore_unused_parameters`**: If your model has tied weights or conditional parameters.

---

## 4.4 Launching Training

### 4.4.1 DeepSpeed Launcher

The `deepspeed` command is the recommended way to launch distributed training:

```bash
# Single GPU
deepspeed train.py

# Specific number of GPUs on a single node
deepspeed --num_gpus=4 train.py

# Specific GPUs by ID
deepspeed --include localhost:0,1,2,3 train.py

# Exclude specific GPUs
deepspeed --exclude localhost:4,5,6,7 train.py

# Multi-node training
deepspeed --num_gpus=8 --num_nodes=2 \
    --hostfile hostfile \
    --master_addr <master_ip> \
    --master_port 29500 \
    train.py

# With extra arguments passed to the training script
deepspeed --num_gpus=4 train.py --epochs 10 --lr 1e-4

# Using a specific DeepSpeed config file
deepspeed --num_gpus=4 train.py --deepspeed ds_config.json
```

### 4.4.2 Hostfile Format

For multi-node training, create a hostfile:

```
worker-0 slots=8
worker-1 slots=8
worker-2 slots=8
```

Each line specifies a hostname and the number of available GPU slots.

### 4.4.3 Launcher Options

| Option | Description |
|--------|-------------|
| `--num_gpus INT` | Number of GPUs per node |
| `--num_nodes INT` | Number of nodes |
| `--hostfile PATH` | Path to hostfile |
| `--include STR` | Specific devices to include (e.g., `localhost:0,1`) |
| `--exclude STR` | Specific devices to exclude |
| `--master_addr STR` | Master node IP address |
| `--master_port INT` | Master port (default: 29500) |
| `--launcher {pdsh,openmpi,mvapich,slurm}` | Multi-node launcher backend |
| `--launcher_args STR` | Additional launcher arguments |
| `--module STR` | Run a Python module instead of a script |
| `--force_multi` | Force multi-node mode |
| `--save_pid` | Save the launcher PID to a file |
| `--autotuning [run|tune]` | Enable autotuning |
| `--elastic_training` | Enable elastic training |
| `--bind_cores_to_rank` | Bind CPU cores to each rank |
| `--bind_core_list STR` | List of CPU cores to bind |

### 4.4.4 Using `torchrun` as Alternative

You can also use PyTorch's `torchrun` launcher with DeepSpeed:

```bash
torchrun --nproc_per_node=4 \
    --nnodes=1 \
    --rdzv_id=100 \
    --rdzv_endpoint=localhost:29500 \
    train.py --deepspeed ds_config.json
```

When using `torchrun`, DeepSpeed detects that the process group is already initialized and skips its own initialization.

### 4.4.5 SLURM Integration

```bash
#!/bin/bash
#SBATCH --job-name=deepspeed-train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00

module load cuda/12.1
module load nccl

# Get master address
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500

srun deepspeed --num_gpus=8 --num_nodes=$SLURM_JOB_NUM_NODES \
    train.py --deepspeed ds_config.json
```

---

## 4.5 Single-GPU vs Multi-GPU

### 4.5.1 Single-GPU Training

DeepSpeed can be used on a single GPU for memory optimization (ZeRO offloading) and mixed precision:

```bash
deepspeed --num_gpus=1 train.py
```

```python
ds_config = {
    "train_batch_size": 16,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
        },
    },
}
```

**When to use single-GPU DeepSpeed:**
- Model doesn't fit in GPU memory (use ZeRO-3 + CPU offload)
- Want mixed precision training with automatic loss scaling
- Want to use DeepSpeed optimizers (e.g., CPU Adam for offloading)

### 4.5.2 Multi-GPU Training (Single Node)

```bash
deepspeed --num_gpus=8 train.py
```

The number of GPUs determines `world_size`. DeepSpeed automatically:
- Assigns each process to a GPU
- Creates a distributed process group
- Partitions data across GPUs via `DistributedSampler`

### 4.5.3 Multi-Node Training

```bash
# On master node (or using a hostfile)
deepspeed --num_gpus=8 --num_nodes=4 \
    --hostfile hostfile \
    train.py
```

For multi-node training, ensure:
- All nodes have the same code and data accessible
- Network allows communication on the master port (default: 29500)
- NCCL is properly configured for your interconnect

---

## 4.6 Basic Training Loop

### 4.6.1 Standard Training Loop

```python
import deepspeed
import torch

# Initialize
model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config_params=ds_config,
    training_data=train_dataset,
)

# Training loop
model_engine.train()
for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        # Move batch to GPU
        inputs = batch[0].cuda()
        labels = batch[1].cuda()

        # Forward pass
        outputs = model_engine(inputs)
        loss = loss_fn(outputs, labels)

        # Backward pass
        model_engine.backward(loss)

        # Optimizer step (handles gradient accumulation internally)
        model_engine.step()

        # Learning rate scheduler step is called automatically by engine.step()
```

### 4.6.2 Understanding the Training Loop

The DeepSpeed training loop differs from standard PyTorch in several ways:

**1. Forward pass: `model_engine(inputs)`**

- With FP16: Input is automatically cast to FP16 if the engine uses FP16 mode
- With ZeRO Stage 3: Parameters are gathered from other GPUs before the forward pass
- Returns the model's output (no need to call `.cuda()` on the model)

**2. Backward pass: `model_engine.backward(loss)`**

- Computes gradients in the configured precision (FP16/BF16/FP32)
- With ZeRO Stage 2: Gradients are reduced and partitioned during backward
- With ZeRO Stage 3: Parameter gradients are computed and immediately reduced
- Handles gradient accumulation internally

**3. Optimizer step: `model_engine.step()`**

This single call handles multiple operations:
```
if (step_count % gradient_accumulation_steps == 0):
    1. Gradient unscaling (if FP16)
    2. Gradient clipping (if configured)
    3. Optimizer step
    4. Zero gradients
    5. LR scheduler step (if configured)
    6. Loss scale update (if FP16 dynamic scaling)
```

**Important**: Do not manually call `optimizer.step()`, `optimizer.zero_grad()`, or `lr_scheduler.step()`. The engine handles all of these.

### 4.6.3 Evaluation Loop

```python
model_engine.eval()
with torch.no_grad():
    for batch in eval_dataloader:
        inputs = batch[0].cuda()
        labels = batch[1].cuda()
        outputs = model_engine(inputs)
        # Compute metrics
        ...

model_engine.train()
```

For ZeRO Stage 3, parameter gathering is handled automatically during `eval()`. However, if you need all parameters on GPU for inference:

```python
# For ZeRO Stage 3, gather all parameters
model_engine.module.eval()  # Access the underlying model

# Or use the DeepSpeed context manager
with deepspeed.zero.GatheredParameters(model_engine.parameters()):
    # All parameters are now gathered on all GPUs
    outputs = model_engine(inputs)
    # Parameters are freed when exiting the context
```

### 4.6.4 Accessing Loss and Metrics

```python
for step, batch in enumerate(train_dataloader):
    inputs = batch[0].cuda()
    labels = batch[1].cuda()

    outputs = model_engine(inputs)
    loss = loss_fn(outputs, labels)

    # Print loss before backward (loss is still valid)
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.6f}")

    model_engine.backward(loss)
    model_engine.step()

    # Access the learning rate
    current_lr = model_engine.get_lr()
    print(f"Current LR: {current_lr}")
```

---

## 4.7 Gradient Accumulation

### 4.7.1 How Gradient Accumulation Works

Gradient accumulation allows simulating a larger batch size by accumulating gradients over multiple micro-batches:

```
Effective Batch Size = micro_batch_size * gradient_accumulation_steps * num_gpus
```

DeepSpeed handles gradient accumulation automatically. The user only needs to set `gradient_accumulation_steps` in the config:

```json
{
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8
}
```

With 4 GPUs, this gives an effective batch size of `4 * 8 * 4 = 128`.

### 4.7.2 Internal Flow

```
Micro-batch 1:  forward -> backward (gradients accumulated)
Micro-batch 2:  forward -> backward (gradients accumulated)
...
Micro-batch 8:  forward -> backward (gradients accumulated)
                                                  |
                                        optimizer step (gradient average)
                                        zero gradients
                                        LR scheduler step
```

### 4.7.3 Manual Gradient Accumulation (Advanced)

Normally, DeepSpeed handles gradient accumulation automatically. However, if you need manual control:

```python
# Manual gradient accumulation control
for step, batch in enumerate(dataloader):
    inputs, labels = batch[0].cuda(), batch[1].cuda()

    outputs = model_engine(inputs)
    loss = loss_fn(outputs, labels)

    # Scale loss by accumulation steps
    # DeepSpeed handles this automatically, so this is typically NOT needed
    model_engine.backward(loss)

    # Check if we should step
    if (step + 1) % gradient_accumulation_steps == 0:
        model_engine.step()
```

### 4.7.4 Gradient Accumulation Data Type

By default, accumulated gradients use the same dtype as training. For better numerical accuracy with many accumulation steps:

```json
{
    "gradient_accumulation_steps": 64,
    "grad_accum_dtype": "fp32"
}
```

---

## 4.8 Mixed Precision Basics

### 4.8.1 FP16 Training

FP16 reduces memory usage by 2x for weights and activations. DeepSpeed handles loss scaling automatically:

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 32,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

**How FP16 loss scaling works:**

1. Forward pass is computed in FP16
2. Loss is multiplied by a scale factor (initially `2^32`)
3. Scaled loss goes through backward pass in FP16
4. Gradients are unscaled before the optimizer step
5. If overflow is detected, the step is skipped and the scale is reduced
6. If no overflow for `loss_scale_window` steps, the scale is increased

**When to use FP16:**
- GPU with good FP16 performance (Ampere, Hopper)
- Standard training where BF16 is not available
- Need dynamic loss scaling

### 4.8.2 BF16 Training

BF16 has the same dynamic range as FP32 (8 exponent bits) but reduced mantissa (7 bits). It does not require loss scaling:

```json
{
    "bf16": {
        "enabled": true
    }
}
```

**When to use BF16:**
- A100, H100, or newer GPUs (good BF16 hardware support)
- Training is more stable without loss scaling
- Recommended for most transformer training

### 4.8.3 Choosing Between FP16 and BF16

| Feature | FP16 | BF16 |
|---------|------|------|
| Memory savings | 2x | 2x |
| Dynamic range | Small (may overflow) | Same as FP32 |
| Loss scaling required | Yes | No |
| Hardware support | All CUDA GPUs | Ampere+ (A100, H100) |
| Numerical precision | Higher mantissa | Lower mantissa |
| Recommended for | Legacy GPUs, mixed workloads | Most modern training |

---

## 4.9 ZeRO Stage Selection Guide

### 4.9.1 Decision Matrix

```
                    Does the model fit on a single GPU?
                           |
                    +------+------+
                    |             |
                   YES            NO
                    |             |
              Use ZeRO 0     What doesn't fit?
              (standard       |
               DP)       +----+----+
                        |         |
                   Weights+      Only optimizer
                   grads         states + grads
                   don't fit     don't fit
                        |         |
                   ZeRO Stage 3  ZeRO Stage 2
                                  (or Stage 1
                                   if only
                                   optimizer
                                   states)
```

### 4.9.2 Stage Comparison

| Aspect | Stage 1 | Stage 2 | Stage 3 |
|--------|---------|---------|---------|
| **What is partitioned** | Optimizer states | + Gradients | + Parameters |
| **Memory savings** | ~4x | ~8x | ~N_gpu x |
| **Communication cost** | Same as DP | Same as DP | +50% over DP |
| **Code changes needed** | None | None | Minimal |
| **Forward pass** | Standard | Standard | Allgather parameters |
| **Backward pass** | Standard | Reduce-scatter gradients | Reduce-scatter + allgather |
| **Checkpoint format** | Compatible | Compatible | DeepSpeed-specific |
| **Best for** | Large optimizer states (e.g., Adam) | Large model + large optimizer | Very large models |

### 4.9.3 Practical Guidelines

**Use Stage 1 when:**
- Using an optimizer with large states (Adam has 2x parameter memory for moments)
- Model and gradients fit on GPU but optimizer states push you over
- Minimal performance impact is desired

**Use Stage 2 when:**
- Model fits on GPU but total training state (model + optimizer + gradients) does not
- Training medium-sized models (1B-7B parameters) on limited GPUs
- Want best throughput-to-memory ratio

**Use Stage 3 when:**
- Model parameters alone exceed GPU memory
- Training very large models (10B+ parameters)
- Combined with CPU/NVMe offload for extreme scale

### 4.9.4 Stage 3 Special Considerations

When using ZeRO Stage 3, be aware of these differences:

**1. Parameter access:**

```python
# Stage 3 parameters are partitioned. Direct access requires gathering:
with deepspeed.zero.GatheredParameters(model.parameters()):
    # Parameters are now fully available on all GPUs
    param_sum = sum(p.sum().item() for p in model.parameters())

# Alternative: use module_scoped_gather
from deepspeed.utils import get_global_norm
norm = get_global_norm(model.parameters())
```

**2. Creating optimizer:**

```python
# With Stage 3, you must use `deepspeed.zero.Init()` context manager
# when creating the model:
with deepspeed.zero.Init(config_dict_or_path=ds_config):
    model = MyLargeModel()

# Or use the zero.Init during model creation
import deepspeed
ds_engine = deepspeed.initialize(model=model, config_params=ds_config)
```

**3. Saving and loading:**

```python
# Save a complete model (not just checkpoint)
# Requires stage3_gather_16bit_weights_on_model_save: true in config
model_engine.save_checkpoint(save_dir="checkpoints", tag=f"step-{step}")

# Load checkpoint
model_engine.load_checkpoint(load_dir="checkpoints")
```

**4. Inference with Stage 3 model:**

```python
# Gather parameters for inference
with deepspeed.zero.GatheredParameters(model_engine.parameters()):
    model_engine.eval()
    outputs = model_engine(inputs)
```

---

## 4.10 Checkpoint Save and Load

### 4.10.1 Saving Checkpoints

```python
# Basic checkpoint save
model_engine.save_checkpoint(save_dir="./checkpoints")

# Save with a custom tag (e.g., step number)
model_engine.save_checkpoint(
    save_dir="./checkpoints",
    tag=f"step-{global_step}",
    client_state={"epoch": epoch, "global_step": global_step}
)
```

**Checkpoint directory structure:**

```
checkpoints/
|-- step-100/
|   |-- zero_pp_rank_0_mp_rank_00_model_states.pt
|   |-- zero_pp_rank_1_mp_rank_00_model_states.pt
|   |-- zero_pp_rank_0_mp_rank_00_optim_states.pt
|   |-- zero_pp_rank_1_mp_rank_00_optim_states.pt
|   |-- latest
|   |-- zero_checkpoint.json
```

### 4.10.2 Loading Checkpoints

```python
# Load the latest checkpoint
load_path, client_state = model_engine.load_checkpoint(load_dir="./checkpoints")

if load_path is None:
    print("No checkpoint found, starting from scratch")
else:
    print(f"Loaded checkpoint from: {load_path}")
    global_step = client_state.get("global_step", 0)
    epoch = client_state.get("epoch", 0)
```

### 4.10.3 Loading with Different GPU Count

For loading a checkpoint saved with a different number of GPUs:

```python
# Method 1: Enable elastic checkpoint in config
# {
#     "zero_optimization": {
#         "stage": 2,
#         "elastic_checkpoint": true
#     }
# }
model_engine.load_checkpoint(load_dir="./checkpoints")

# Method 2: Use universal checkpoint format
# {
#     "checkpoint": {
#         "load_universal": true
#     }
# }
model_engine.load_checkpoint(load_dir="./checkpoints")
```

### 4.10.4 Converting Checkpoints to Standard Format

For ZeRO Stage 3 checkpoints, convert to a standard PyTorch state dict for inference:

```bash
# Convert zero checkpoint to a single file
python -m deepspeed.utils.zero_to_fp32 \
    ./checkpoints/step-100 \
    ./checkpoints/step-100/pytorch_model.bin

# Or use the script
deepspeed --num_gpus=1 zero_to_fp32.py \
    ./checkpoints/step-100 \
    ./checkpoints/step-100/pytorch_model.bin
```

### 4.10.5 Saving Model for Inference

```python
# For ZeRO Stage 3, ensure weights are gathered before saving
# Add to config: "stage3_gather_16bit_weights_on_model_save": true

# Save the full model (not a DeepSpeed checkpoint)
if model_engine.zero_optimization_partition_weights():
    # ZeRO Stage 3: Save gathered model
    model_engine.save_checkpoint(save_dir="./model_output")
else:
    # ZeRO Stage 0/1/2: Save normally
    torch.save(model_engine.module.state_dict(), "model.pt")
```

---

## 4.11 Basic Inference Setup

### 4.11.1 Simple Inference

```python
import deepspeed
import torch

# Load model
model = MyModel()
model.load_state_dict(torch.load("model.pt"))
model.eval()

# Initialize DeepSpeed inference
ds_model = deepspeed.init_inference(
    model=model,
    mp_size=1,
    dtype=torch.float16,
    replace_with_kernel_inject=True,
)

# Run inference
with torch.no_grad():
    inputs = torch.randn(1, 1024).cuda()
    outputs = ds_model(inputs)
```

### 4.11.2 Transformer Model Inference

```python
import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
)

# Initialize DeepSpeed inference with kernel injection
ds_model = deepspeed.init_inference(
    model=model,
    dtype=torch.float16,
    replace_with_kernel_inject=True,
    tensor_parallel={"tp_size": 1},
    max_out_tokens=1024,
)

# Generate text
prompt = "DeepSpeed is a"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = ds_model.generate(**inputs, max_new_tokens=100)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### 4.11.3 Launching Inference

```bash
# Single-GPU inference
deepspeed --num_gpus=1 inference.py

# Multi-GPU tensor parallel inference
deepspeed --num_gpus=4 inference.py
```

---

## 4.12 Integration with HuggingFace

### 4.12.1 Using DeepSpeed with HuggingFace Trainer

HuggingFace Trainer has built-in DeepSpeed support. Simply pass the DeepSpeed config path:

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./output",
    deepspeed="ds_config.json",  # Path to DeepSpeed config
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=500,
    bf16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

Launch with:

```bash
deepspeed --num_gpus=4 train.py --deepspeed ds_config.json
```

### 4.12.2 HuggingFace Accelerate Integration

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)
```

For DeepSpeed-specific features, use the Trainer integration instead.

### 4.12.3 HuggingFace DeepSpeed Config with "auto" Values

HuggingFace Trainer supports `"auto"` values in the DeepSpeed config, which are filled in from `TrainingArguments`:

```json
{
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

When `"auto"` is used:
- `train_batch_size` is computed from `per_device_train_batch_size * num_processes * gradient_accumulation_steps`
- `gradient_accumulation_steps` uses `TrainingArguments.gradient_accumulation_steps`
- `gradient_clipping` uses `TrainingArguments.max_grad_norm`
- `bf16.enabled` uses `TrainingArguments.bf16`

---

## 4.13 Common Pitfalls and Troubleshooting

### 4.13.1 Common Errors and Solutions

**Error 1: `RuntimeError: Expected all tensors to be on the same device`**

```python
# Problem: Input data is not on GPU
# Solution: Move inputs to CUDA
inputs = inputs.cuda()
# Or use the device from the engine
inputs = inputs.to(model_engine.device)
```

**Error 2: `AssertionError: ZeRO-3 model was not initialized with zero.Init()`**

```python
# Problem: Stage 3 requires special initialization
# Solution: Wrap model creation in zero.Init()
with deepspeed.zero.Init(config_dict_or_path=ds_config):
    model = MyLargeModel()
```

**Error 3: `TypeError: can't pickle ...`**

```python
# Problem: DeepSpeed engine cannot be pickled
# Solution: Save model state dict, not the engine
state_dict = model_engine.module.state_dict()
torch.save(state_dict, "model.pt")
```

**Error 4: Checkpoint loading fails with different GPU count**

```python
# Problem: Checkpoint was saved with different world_size
# Solution: Enable elastic checkpoints
# In config:
# "zero_optimization": {"stage": 2, "elastic_checkpoint": true}
```

**Error 5: OOM (Out of Memory) during forward pass**

```python
# Problem: Model + activations don't fit in GPU memory
# Solutions:
# 1. Reduce micro_batch_size_per_gpu
# 2. Enable activation checkpointing
# 3. Use ZeRO Stage 3
# 4. Enable CPU offload
```

**Error 6: `ImportError: No module named 'deepspeed'`**

```bash
# Problem: DeepSpeed not installed in the correct environment
# Solution: Verify installation
pip show deepspeed
ds_report
```

**Error 7: Slow training with ZeRO Stage 3**

```python
# Problem: Stage 3 communication overhead is high
# Solutions:
# 1. Enable overlap_comm: true
# 2. Tune bucket sizes (allgather_bucket_size, reduce_bucket_size)
# 3. Enable prefetching (stage3_prefetch_bucket_size)
# 4. Use contiguous_gradients: true
# 5. Increase stage3_param_persistence_threshold for small parameters
```

**Error 8: Loss is NaN with FP16**

```python
# Problem: FP16 loss scaling is too aggressive
# Solutions:
# 1. Lower initial_scale_power (e.g., 16 instead of 32)
# 2. Increase loss_scale_window (e.g., 2000)
# 3. Increase min_loss_scale (e.g., 1e-4)
# 4. Switch to BF16 if hardware supports it
```

### 4.13.2 Performance Tips

1. **Batch size tuning**: Start with a small micro-batch size and increase until you hit OOM. Then set `gradient_accumulation_steps` to achieve the desired effective batch size.

2. **ZeRO Stage selection**:
   - If model fits in GPU memory: Stage 0 or 1
   - If only optimizer states cause OOM: Stage 1
   - If model + optimizer cause OOM: Stage 2
   - If model itself doesn't fit: Stage 3

3. **Communication optimization**:
   - Always enable `overlap_comm: true` for Stage 2/3
   - Tune `reduce_bucket_size` and `allgather_bucket_size` for your interconnect
   - Consider `reduce_scatter: true` for Stage 2

4. **Memory vs throughput tradeoff**:
   - Larger micro-batch size = better throughput but more memory
   - More gradient accumulation = same effective batch size but slower
   - Activation checkpointing = less memory but slower (recomputation)

5. **Monitoring training**:
   - Enable `wall_clock_breakdown: true` to identify bottlenecks
   - Use `flops_profiler` to verify compute efficiency
   - Monitor GPU utilization with `nvidia-smi dmon -s puc`

---

## 4.14 Complete Example: Training a Transformer Model

### 4.14.1 Full Training Script

```python
#!/usr/bin/env python
# train_gpt.py - Complete GPT training example with DeepSpeed

import argparse
import deepspeed
import torch
from torch.utils.data import Dataset, DataLoader

# Argument parser
def add_argument():
    parser = argparse.ArgumentParser(description="GPT Training with DeepSpeed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default=None)
    return parser.parse_args()

# Dataset
class GPTDataset(Dataset):
    def __init__(self, vocab_size=50257, seq_len=1024, size=10000):
        self.data = torch.randint(0, vocab_size, (size, seq_len))
        self.labels = torch.randint(0, vocab_size, (size, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx], "labels": self.labels[idx]}

# Model
class SimpleGPT(torch.nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, nhead=12, num_layers=12):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
        return {"loss": loss, "logits": logits}

# Main
def main():
    args = add_argument()

    # Create model
    model = SimpleGPT()

    # DeepSpeed config
    ds_config = {
        "train_batch_size": 64,
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 4,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 3e-4,
                "warmup_num_steps": 1000,
                "total_num_steps": 100000,
            },
        },
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 500000000,
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "wall_clock_breakdown": True,
    }

    # Create dataset
    train_dataset = GPTDataset()

    # Initialize DeepSpeed
    model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        config_params=ds_config,
        args=args,
    )

    # Training loop
    global_step = 0
    for epoch in range(args.epochs):
        model_engine.train()
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()

            outputs = model_engine(input_ids, labels=labels)
            loss = outputs["loss"]

            model_engine.backward(loss)
            model_engine.step()

            global_step += 1

            if global_step % 100 == 0:
                # Save checkpoint
                model_engine.save_checkpoint(
                    save_dir="./checkpoints",
                    tag=f"global_step_{global_step}",
                    client_state={"epoch": epoch, "global_step": global_step},
                )

    # Final save
    model_engine.save_checkpoint(
        save_dir="./checkpoints",
        tag="final",
        client_state={"epoch": args.epochs, "global_step": global_step},
    )

if __name__ == "__main__":
    main()
```

### 4.14.2 Running the Example

```bash
# Single GPU
deepspeed --num_gpus=1 train_gpt.py

# 4 GPUs
deepspeed --num_gpus=4 train_gpt.py

# 8 GPUs with custom config file
deepspeed --num_gpus=8 train_gpt.py --deepspeed ds_config.json
```

---

## 4.15 Quick Reference Card

| Task | Code |
|------|------|
| **Initialize training** | `model_engine, optimizer, dataloader, scheduler = deepspeed.initialize(model=model, config_params=ds_config)` |
| **Forward pass** | `outputs = model_engine(inputs)` |
| **Backward pass** | `model_engine.backward(loss)` |
| **Optimizer step** | `model_engine.step()` |
| **Save checkpoint** | `model_engine.save_checkpoint(save_dir="./ckpt")` |
| **Load checkpoint** | `path, state = model_engine.load_checkpoint(load_dir="./ckpt")` |
| **Set train mode** | `model_engine.train()` |
| **Set eval mode** | `model_engine.eval()` |
| **Get learning rate** | `model_engine.get_lr()` |
| **Get batch size** | `model_engine.train_batch_size` |
| **Get ZeRO stage** | `model_engine.zero_optimization_stage` |
| **Check FP16** | `model_engine.fp16_enabled` |
| **Check BF16** | `model_engine.bf16_enabled` |
| **Initialize inference** | `ds_model = deepspeed.init_inference(model=model, dtype=torch.float16)` |
| **Launch single GPU** | `deepspeed --num_gpus=1 train.py` |
| **Launch multi GPU** | `deepspeed --num_gpus=4 train.py` |
| **Environment report** | `ds_report` |
| **Convert ZeRO checkpoint** | `python -m deepspeed.utils.zero_to_fp32 ./ckpt ./model.bin` |
