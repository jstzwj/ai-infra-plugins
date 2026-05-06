# DeepSpeed Reference - Chapter 5: DeepSpeed Engine

This chapter provides a comprehensive reference for the DeepSpeed engine classes, including `DeepSpeedEngine`, `PipelineEngine`, and `DeepSpeedHybridEngine`. It covers initialization parameters, core methods, properties, internal mechanisms, and advanced usage patterns.

---

## 5.1 DeepSpeedEngine Class

**Module:** `deepspeed.runtime.engine`

**File:** `deepspeed/runtime/engine.py`

`DeepSpeedEngine` is the central class of DeepSpeed. It wraps a PyTorch model and orchestrates distributed training with ZeRO optimization, mixed precision, gradient accumulation, checkpointing, and more.

### 5.1.1 Class Definition

```python
class DeepSpeedEngine:
    def __init__(
        self,
        args: argparse.Namespace,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        model_parameters: Optional[Iterable[torch.nn.Parameter]] = None,
        lr_scheduler: Optional[object] = None,
        mpu: Optional[object] = None,
        dist_init_required: Optional[bool] = None,
        collate_fn: Optional[callable] = None,
        training_data: Optional[Union[Dataset, DataLoader]] = None,
        config_params: Optional[Union[dict, DeepSpeedConfig]] = None,
        config: Optional[str] = None,
        dont_change_device: bool = False,
    ):
```

### 5.1.2 `__init__` Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `args` | `argparse.Namespace` | Yes (usually from launcher) | Command-line arguments. Must contain `local_rank` if using DeepSpeed launcher. |
| `model` | `torch.nn.Module` | Yes | The PyTorch model to wrap. For ZeRO Stage 3, the model should be created within `deepspeed.zero.Init()` context. |
| `optimizer` | `torch.optim.Optimizer` | No | Pre-created optimizer. If None, DeepSpeed creates one from `ds_config`. |
| `model_parameters` | `Iterable[Parameter]` | No | Model parameters for optimizer creation. Required if `optimizer` is None and optimizer is configured in `ds_config`. |
| `lr_scheduler` | `object` | No | Learning rate scheduler. Must implement `step()`, `get_last_lr()` or `get_lr()`. |
| `mpu` | `object` | No | Model parallelism utility for Megatron-style tensor parallelism. Must provide `get_model_parallel_rank/group/world_size` methods. |
| `dist_init_required` | `bool` | No | Whether to call `torch.distributed.init_process_group()`. Default: auto-detect (True if not already initialized). |
| `collate_fn` | `callable` | No | Collate function for DataLoader creation. |
| `training_data` | `Dataset` or `DataLoader` | No | Training dataset or DataLoader. If a Dataset, DeepSpeed wraps it in a distributed DataLoader. |
| `config_params` | `dict` or `DeepSpeedConfig` | No | DeepSpeed configuration as a dictionary or `DeepSpeedConfig` object. |
| `config` | `str` | No | Path to `ds_config.json` file. |
| `dont_change_device` | `bool` | No | If True, do not move the model to the GPU. Useful for models already on the correct device. |

### 5.1.3 Initialization Sequence

When `DeepSpeedEngine.__init__()` is called, the following sequence of operations occurs:

```
1. Store arguments and configuration
2. Initialize distributed backend (if dist_init_required)
   - Set device based on local_rank
   - Call torch.distributed.init_process_group()
3. Parse configuration (DeepSpeedConfig)
4. Apply accelerator abstraction
5. Configure mixed precision
   - FP16: Create FP16Optimizer wrapper
   - BF16: Create BF16Optimizer wrapper
   - AMP: Configure torch.cuda.amp
6. Configure ZeRO optimization
   - Stage 1/2: Create DeepSpeedZeroOptimizer
   - Stage 3: Create DeepSpeedZeroEngine (replaces module with partitioned version)
7. Configure optimizer
   - Create from config if not provided
   - Wrap with appropriate ZeRO/mixed precision optimizer
8. Configure offloading (if enabled)
   - CPU offload for parameters/optimizer states
   - NVMe offload for parameters/optimizer states
9. Create training DataLoader (if training_data provided)
10. Configure learning rate scheduler
11. Configure activation checkpointing (if enabled)
12. Configure monitoring (TensorBoard, wandb, etc.)
13. Configure communication logging (if enabled)
14. Log initialization summary and memory report
```

---

## 5.2 Core Methods

### 5.2.1 `forward()`

```python
def forward(self, *inputs, **kwargs) -> Any:
```

Performs the forward pass of the model. Handles:
- Automatic precision casting (FP16/BF16)
- ZeRO Stage 3 parameter gathering
- Autocast integration
- Loss computation (if configured)

**Parameters:**
- `*inputs`: Positional arguments passed to the model's `forward()` method
- `**kwargs`: Keyword arguments passed to the model's `forward()` method

**Returns:** The output of the model's `forward()` method.

**Behavior by Configuration:**

| Configuration | Behavior |
|---------------|----------|
| No mixed precision | Forward in FP32 |
| FP16 enabled | Model parameters cast to FP16, forward in FP16 |
| BF16 enabled | Model parameters cast to BF16, forward in BF16 |
| AMP enabled | `torch.autocast` wraps the forward pass |
| ZeRO Stage 3 | Parameters are gathered from partitions before forward |
| DeepCompile | Forward is traced and compiled |

**Example:**

```python
# Standard usage
outputs = model_engine(input_ids, attention_mask=mask)

# The output depends on your model
# For HuggingFace models:
# outputs = model_engine(input_ids, labels=labels)
# loss = outputs.loss
```

### 5.2.2 `backward()`

```python
def backward(self, loss: torch.Tensor, allreduce: bool = True) -> None:
```

Performs the backward pass. Handles:
- Mixed precision gradient computation
- Gradient reduction across GPUs
- ZeRO gradient partitioning
- Gradient accumulation

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss` | `torch.Tensor` | Required | The loss tensor to backpropagate. Must be a scalar. |
| `allreduce` | `bool` | `True` | Whether to perform gradient all-reduce. Set to `False` for pipeline parallelism where reduction is handled separately. |

**Behavior by Configuration:**

| Configuration | Behavior |
|---------------|----------|
| FP16 | Gradients computed in FP16, then unscaled |
| BF16 | Gradients computed in BF16 |
| ZeRO Stage 1 | Gradient all-reduce, optimizer states partitioned |
| ZeRO Stage 2 | Gradient reduce-scatter, each GPU keeps only its partition |
| ZeRO Stage 3 | Gradient reduce-scatter + parameter partition updates |
| Gradient accumulation | Gradients are accumulated over micro-batches |

**Example:**

```python
outputs = model_engine(inputs, labels=labels)
loss = outputs.loss

model_engine.backward(loss)
# Gradients are now computed and reduced
```

### 5.2.3 `step()`

```python
def step(self, lr_kwargs: Optional[dict] = None) -> bool:
```

Performs the optimizer step. This is the main training step method that handles gradient accumulation, optimizer updates, learning rate scheduling, and gradient zeroing.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr_kwargs` | `dict` | `None` | Additional keyword arguments for the learning rate scheduler's `step()` method. |

**Returns:** `True` if the optimizer step was performed, `False` if gradients are still being accumulated.

**Internal Flow:**

```
1. Increment gradient_accumulation counter
2. Check if gradient_accumulation_steps reached:
   a. If NOT reached: return False (skip optimizer step)
   b. If reached:
      i.   Unscale gradients (if FP16)
      ii.  Gradient clipping (if configured)
      iii. Optimizer step
      iv.  Update FP16 loss scale (if dynamic loss scaling)
      v.   Zero gradients
      vi.  LR scheduler step
      vii. Reset gradient accumulation counter
      viii. Return True
```

**Example:**

```python
for step, batch in enumerate(dataloader):
    outputs = model_engine(batch)
    loss = outputs.loss
    model_engine.backward(loss)

    stepped = model_engine.step()
    if stepped:
        print(f"Optimizer step performed at micro-step {step}")
```

### 5.2.4 `train()` and `eval()`

```python
def train(self) -> None:
def eval(self) -> None:
```

Set the engine and underlying model to training or evaluation mode.

```python
# Training mode
model_engine.train()
# Model is in training mode, dropout is active, batch norm updates

# Evaluation mode
model_engine.eval()
# Model is in eval mode, dropout is disabled, batch norm uses running stats
```

For ZeRO Stage 3, `eval()` triggers parameter gathering if needed for inference.

### 5.2.5 `save_checkpoint()`

```python
def save_checkpoint(
    self,
    save_dir: str,
    tag: Optional[str] = None,
    client_state: Optional[dict] = None,
    save_latest: bool = True,
    exclude_frozen_parameters: bool = False,
) -> None:
```

Saves a training checkpoint including model weights, optimizer states, LR scheduler state, and custom client state.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_dir` | `str` | Required | Directory to save the checkpoint. Created if it doesn't exist. |
| `tag` | `str` | `None` | Custom tag for the checkpoint (e.g., "step-1000"). If None, a default tag is generated. |
| `client_state` | `dict` | `None` | User-defined state to save (e.g., epoch, global_step, best_metric). |
| `save_latest` | `bool` | `True` | Save a `latest` file pointing to the most recent checkpoint. |
| `exclude_frozen_parameters` | `bool` | `False` | Exclude frozen (non-trainable) parameters from the checkpoint. |

**Saved State:**

| State | Description |
|-------|-------------|
| Model parameters | FP16/BF16 weights (partitioned for ZeRO Stage 3) |
| Optimizer states | Adam moments, step count (partitioned for ZeRO) |
| LR scheduler state | Scheduler internal state |
| Random state | RNG states for reproducibility |
| Client state | User-provided state dict |

**Checkpoint Directory Structure:**

```
save_dir/
|-- tag/                            # e.g., "step_1000"
|   |-- zero_pp_rank_0_mp_rank_00_model_states.pt
|   |-- zero_pp_rank_0_mp_rank_00_optim_states.pt
|   |-- zero_pp_rank_1_mp_rank_00_model_states.pt
|   |-- zero_pp_rank_1_mp_rank_00_optim_states.pt
|   |-- ...
|   |-- latest                      # Points to this tag
|   |-- zero_checkpoint.json        # Metadata
```

**Example:**

```python
# Save checkpoint with custom state
model_engine.save_checkpoint(
    save_dir="./checkpoints",
    tag=f"epoch-{epoch}",
    client_state={
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
    },
)
```

### 5.2.6 `load_checkpoint()`

```python
def load_checkpoint(
    self,
    load_dir: str,
    tag: Optional[str] = None,
    load_optimizer_states: bool = True,
    load_lr_scheduler_states: bool = True,
    client_state: Optional[dict] = None,
    load_module_strict: bool = False,
    load_dp_weights: bool = True,
) -> Tuple[str, dict]:
```

Loads a training checkpoint.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `load_dir` | `str` | Required | Directory containing the checkpoint. |
| `tag` | `str` | `None` | Specific checkpoint tag to load. If None, loads the latest. |
| `load_optimizer_states` | `bool` | `True` | Load optimizer states (Adam moments, etc.). Set to `False` for fine-tuning with a fresh optimizer. |
| `load_lr_scheduler_states` | `bool` | `True` | Load LR scheduler state. Set to `False` for a new schedule. |
| `client_state` | `dict` | `None` | Dict to populate with the saved client state. |
| `load_module_strict` | `bool` | `False` | Strict mode for loading model state dict. If `True`, keys must match exactly. |
| `load_dp_weights` | `bool` | `True` | Load data-parallel weights. Set to `False` for pipeline parallel loading. |

**Returns:** A tuple of `(load_path, client_state)` where `load_path` is the path to the loaded checkpoint directory, and `client_state` is the user-provided state dict. Returns `(None, None)` if no checkpoint is found.

**Example:**

```python
# Load latest checkpoint
load_path, client_state = model_engine.load_checkpoint(
    load_dir="./checkpoints",
)

if load_path is not None:
    global_step = client_state.get("global_step", 0)
    epoch = client_state.get("epoch", 0)
    print(f"Resumed from {load_path}, step {global_step}, epoch {epoch}")
else:
    print("No checkpoint found, starting from scratch")

# Load specific checkpoint
load_path, client_state = model_engine.load_checkpoint(
    load_dir="./checkpoints",
    tag="epoch-5",
    load_optimizer_states=False,  # Fresh optimizer
)

# Load with strict key matching
load_path, client_state = model_engine.load_checkpoint(
    load_dir="./checkpoints",
    load_module_strict=True,  # All keys must match
)
```

---

## 5.3 Properties

### 5.3.1 Batch Size Properties

```python
@property
def train_batch_size(self) -> int:
    """The effective total training batch size across all GPUs."""
    return self._train_batch_size

@property
def train_micro_batch_size_per_gpu(self) -> int:
    """The micro-batch size per GPU for each forward+backward pass."""
    return self._train_micro_batch_size_per_gpu

@property
def gradient_accumulation_steps(self) -> int:
    """Number of forward+backward passes per optimizer step."""
    return self._gradient_accumulation_steps

@property
def micro_steps(self) -> int:
    """Current number of micro-steps within the current gradient accumulation window."""
    return self._micro_steps
```

### 5.3.2 Optimization Properties

```python
@property
def zero_optimization_stage(self) -> int:
    """Current ZeRO optimization stage (0, 1, 2, or 3)."""
    return self._zero_optimization_stage

@property
def fp16_enabled(self) -> bool:
    """Whether FP16 mixed precision training is enabled."""
    return self._fp16_enabled

@property
def bf16_enabled(self) -> bool:
    """Whether BF16 mixed precision training is enabled."""
    return self._bf16_enabled

@property
def amp_enabled(self) -> bool:
    """Whether PyTorch native AMP is enabled."""
    return self._amp_enabled

@property
def gradient_accumulation_enabled(self) -> bool:
    """Whether gradient accumulation is active."""
    return self._gradient_accumulation_steps > 1
```

### 5.3.3 Model and Optimizer Properties

```python
@property
def module(self) -> torch.nn.Module:
    """Access the underlying PyTorch model."""
    return self._module

@property
def optimizer(self) -> torch.optim.Optimizer:
    """Access the wrapped optimizer."""
    return self._optimizer

@property
def lr_scheduler(self) -> object:
    """Access the learning rate scheduler."""
    return self._lr_scheduler

@property
def device(self) -> torch.device:
    """The device the model is on."""
    return self._device

@property
def global_rank(self) -> int:
    """The global rank of this process."""
    return self._global_rank

@property
def local_rank(self) -> int:
    """The local rank of this process on the current node."""
    return self._local_rank

@property
def world_size(self) -> int:
    """The total number of processes."""
    return self._world_size

@property
def dp_world_size(self) -> int:
    """The data-parallel world size."""
    return self._dp_world_size
```

### 5.3.4 Memory Properties

```python
@property
def total_parameters(self) -> int:
    """Total number of parameters in the model (across all GPUs)."""
    return self._total_parameters

@property
def trainable_parameters(self) -> int:
    """Number of trainable parameters."""
    return self._trainable_parameters
```

---

## 5.4 Memory Management Methods

### 5.4.1 `empty_partition_cache()`

```python
def empty_partition_cache(self) -> None:
```

Empty the parameter and gradient caches in ZeRO Stage 3. Forces release of cached parameter partitions to free GPU memory.

```python
# Free cached parameters
model_engine.empty_partition_cache()
```

### 5.4.2 `memory_usage()`

```python
def memory_usage(self) -> dict:
```

Returns a dictionary with current memory usage statistics.

```python
mem = model_engine.memory_usage()
# Returns:
# {
#     "gpu_allocated": float,  # GB currently allocated on GPU
#     "gpu_reserved": float,   # GB reserved by CUDA allocator
#     "gpu_max_allocated": float,  # GB peak allocation
#     "cpu_allocated": float,  # GB allocated on CPU (for offloading)
# }
```

### 5.4.3 `print_memory_usage()`

```python
def print_memory_usage(self, tag: str = "") -> None:
```

Print current memory usage to the log.

```python
model_engine.print_memory_usage(tag="After forward pass")
```

---

## 5.5 Gradient Handling

### 5.5.1 `get_grad_norm()`

```python
def get_grad_norm(self) -> float:
```

Get the total gradient norm across all parameters.

```python
grad_norm = model_engine.get_grad_norm()
print(f"Gradient norm: {grad_norm:.4f}")
```

### 5.5.2 `clip_gradients()`

```python
def clip_gradients(self, max_norm: float) -> float:
```

Clip gradients by total norm. Returns the gradient norm before clipping.

```python
grad_norm = model_engine.clip_gradients(max_norm=1.0)
```

> Note: Gradient clipping is also handled automatically by `engine.step()` if `gradient_clipping` is configured.

### 5.5.3 Gradient Accumulation Details

DeepSpeed handles gradient accumulation transparently. The internal counter tracks how many micro-batches have been processed:

```python
# Check current micro-step count
current_micro_step = model_engine.micro_steps

# Check if this step will trigger an optimizer update
will_step = (model_engine.micro_steps + 1) % model_engine.gradient_accumulation_steps == 0
```

### 5.5.4 Gradient Accumulation with Mixed Precision

When using FP16 with gradient accumulation, DeepSpeed uses the following approach:

1. The loss for each micro-batch is divided by `gradient_accumulation_steps`
2. Gradients are accumulated in the precision specified by `grad_accum_dtype` (default: same as training dtype)
3. After accumulation, gradients are unscaled (if FP16) and the optimizer step is performed

```python
# For better numerical accuracy with FP16:
ds_config = {
    "fp16": {"enabled": True},
    "gradient_accumulation_steps": 16,
    "grad_accum_dtype": "fp32",  # Accumulate in FP32
}
```

---

## 5.6 Distributed Training Integration

### 5.6.1 Process Group Management

```python
# Access distributed information
rank = model_engine.global_rank
local_rank = model_engine.local_rank
size = model_engine.world_size
dp_size = model_engine.dp_world_size

# Check if this is the main process
is_main = model_engine.global_rank == 0
```

### 5.6.2 Barrier Synchronization

```python
def barrier(self, name: Optional[str] = None) -> None:
```

Synchronize all processes. Useful for ensuring all ranks reach the same point before proceeding.

```python
model_engine.barrier("before_checkpoint")

# Only rank 0 saves metadata
if model_engine.global_rank == 0:
    save_metadata()

model_engine.barrier("after_metadata_save")
```

### 5.6.3 Distributed Metrics

```python
def all_reduce_tensor(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
```

Perform an all-reduce operation on a tensor across all processes.

```python
# Average a metric across all GPUs
total_loss = model_engine.all_reduce_tensor(loss_tensor, op="sum")
avg_loss = total_loss / model_engine.world_size
```

### 5.6.4 Model Parallel Utility (MPU)

For models using Megatron-style tensor parallelism, the MPU object provides model-parallel process groups:

```python
# MPU provides:
mpu.get_model_parallel_rank()       # Rank within model-parallel group
mpu.get_model_parallel_world_size() # Size of model-parallel group
mpu.get_model_parallel_group()      # torch.distributed process group
mpu.get_data_parallel_rank()        # Rank within data-parallel group
mpu.get_data_parallel_world_size()  # Size of data-parallel group
mpu.get_data_parallel_group()       # Data-parallel process group
```

---

## 5.7 Learning Rate and Optimizer Access

### 5.7.1 `get_lr()`

```python
def get_lr(self) -> float:
```

Get the current learning rate.

```python
current_lr = model_engine.get_lr()
```

### 5.7.2 Optimizer State Access

```python
# Access the underlying optimizer
opt = model_engine.optimizer

# For ZeRO Stage 1/2, the optimizer is wrapped in DeepSpeedZeroOptimizer
# Access parameter groups
for group in opt.param_groups:
    print(f"LR: {group['lr']}")
    print(f"Params: {len(group['params'])}")

# For ZeRO Stage 3, access is more restricted
# Use engine methods for checkpoint save/load
```

### 5.7.3 `get_optimizer_param_groups()`

```python
def get_optimizer_param_groups(self) -> list:
```

Returns the optimizer parameter groups with LR and other hyperparameters.

```python
param_groups = model_engine.get_optimizer_param_groups()
for i, group in enumerate(param_groups):
    print(f"Group {i}: lr={group['lr']}, params={len(group['params'])}")
```

---

## 5.8 Engine State Management

### 5.8.1 `is_gradient_accumulation_boundary()`

```python
def is_gradient_accumulation_boundary(self) -> bool:
```

Check if the current step is at a gradient accumulation boundary (i.e., the next `step()` will trigger an optimizer update).

```python
if model_engine.is_gradient_accumulation_boundary():
    print("Next step will update weights")
```

### 5.8.2 `get_global_grad_norm()`

```python
def get_global_grad_norm(self) -> float:
```

Get the global gradient norm across all parameters and all GPUs.

```python
global_norm = model_engine.get_global_grad_norm()
```

### 5.8.3 `zero_grad()`

```python
def zero_grad(self) -> None:
```

Zero out all gradients. Normally handled automatically by `step()`, but can be called manually.

```python
model_engine.zero_grad()
```

### 5.8.4 `set_train_batch_size()`

```python
def set_train_batch_size(self, batch_size: int) -> None:
```

Dynamically adjust the training batch size. Useful for curriculum learning or adaptive batch sizing.

```python
model_engine.set_train_batch_size(128)
```

---

## 5.9 PipelineEngine Class

**Module:** `deepspeed.runtime.pipe.engine`

**File:** `deepspeed/runtime/pipe/engine.py`

`PipelineEngine` extends `DeepSpeedEngine` with pipeline parallelism support. The model is split into stages, each placed on a different GPU. Micro-batches flow through the pipeline using scheduling algorithms.

### 5.9.1 Class Definition

```python
class PipelineEngine(DeepSpeedEngine):
    """DeepSpeed engine for pipeline parallelism."""
```

### 5.9.2 Pipeline-Specific Methods

#### `forward()`

```python
def forward(self, *inputs, **kwargs) -> Any:
```

In pipeline mode, `forward()` processes the micro-batch through the local pipeline stage only. Input from the previous stage is received via point-to-point communication.

#### `backward()`

```python
def backward(self, loss: torch.Tensor, allreduce: bool = True) -> None:
```

In pipeline mode, `backward()` computes gradients for the local stage and sends gradients to the previous stage.

#### `step()`

```python
def step(self) -> bool:
```

In pipeline mode, `step()` executes the full pipeline schedule (multiple micro-batch forward and backward passes) before performing the optimizer step.

#### Pipeline Stage Methods

```python
def is_first_stage(self) -> bool:
    """Check if this process holds the first pipeline stage."""
    return self._stage_id == 0

def is_last_stage(self) -> bool:
    """Check if this process holds the last pipeline stage."""
    return self._stage_id == self.num_stages - 1

def set_has_optimizer_model(self, flag: bool) -> None:
    """Configure whether this pipeline stage has optimizer states."""
    self._has_optimizer_model = flag
```

### 5.9.3 Pipeline Schedule Types

| Schedule | Description | Pros | Cons |
|----------|-------------|------|------|
| GPipe | All forwards, then all backwards | Simple, good utilization | High memory (all activations stored) |
| 1F1B | Interleave 1 forward, 1 backward | Low steady-state memory | Slightly lower utilization |
| Interleaved 1F1B | Multiple stages per device | Better utilization | More complex setup |

### 5.9.4 Pipeline Configuration Example

```json
{
    "pipeline": {
        "enabled": true,
        "parallel_size": 4,
        "micro_batches": 8,
        "micro_batch_size": 4
    },
    "train_batch_size": 128,
    "fp16": {"enabled": true},
    "zero_optimization": {"stage": 0}
}
```

> Note: Pipeline parallelism is typically combined with ZeRO Stage 0 or 1. Stage 2/3 is generally not used with pipeline parallelism because each stage only needs a subset of parameters.

### 5.9.5 Creating a Pipeline Module

```python
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec

class GPTLayer(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(hidden_size, num_heads)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * 4),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size * 4, hidden_size),
        )
        self.ln1 = torch.nn.LayerNorm(hidden_size)
        self.ln2 = torch.nn.LayerNorm(hidden_size)

    def forward(self, x, attention_mask=None):
        residual = x
        x = self.ln1(x)
        x, _ = self.attention(x, x, x, attn_mask=attention_mask)
        x = residual + x
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        return x

# Define the pipeline model
class GPTPipeline(PipelineModule):
    def __init__(self, num_layers, hidden_size, num_heads, num_stages=4):
        layers = [
            LayerSpec(GPTLayer, hidden_size=hidden_size, num_heads=num_heads)
            for _ in range(num_layers)
        ]
        super().__init__(
            layers=layers,
            num_stages=num_stages,
            loss_fn=torch.nn.CrossEntropyLoss(),
        )
```

---

## 5.10 DeepSpeedHybridEngine Class

**Module:** `deepspeed.runtime.hybrid_engine`

**File:** `deepspeed/runtime/hybrid_engine.py`

`DeepSpeedHybridEngine` extends `PipelineEngine` to support both training and inference in a single engine. It is designed for RLHF (Reinforcement Learning from Human Feedback) workloads where the model alternates between training (PPO updates) and inference (generating responses).

### 5.10.1 Class Definition

```python
class DeepSpeedHybridEngine(PipelineEngine):
    """Hybrid engine supporting both training and inference."""
```

### 5.10.2 Hybrid Engine Methods

#### `inference_forward()`

```python
def inference_forward(self, *inputs, **kwargs) -> Any:
```

Perform an inference-optimized forward pass. Uses kernel injection and other inference optimizations while maintaining ZeRO partitioning.

```python
# Switch to inference mode
model_engine.eval()

# Inference forward
with torch.no_grad():
    outputs = model_engine.inference_forward(input_ids)
```

#### `inference()`

```python
def inference(self, *inputs, **kwargs) -> Any:
```

Full inference pipeline including parameter gathering and kernel injection.

#### `generate()`

```python
def generate(self, *inputs, **kwargs) -> Any:
```

Text generation for autoregressive models. Supports the same interface as HuggingFace's `generate()`.

```python
# Generate text
outputs = model_engine.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
)
```

### 5.10.3 Hybrid Engine Configuration

```json
{
    "hybrid_engine": {
        "enabled": true,
        "max_out_tokens": 512,
        "inference_tp_size": 1,
        "release_inference_cache": false,
        "pin_parameters": true,
        "tp_gather_partition_size": 8
    }
}
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable the hybrid engine |
| `max_out_tokens` | `int` | `1024` | Maximum number of output tokens for generation |
| `inference_tp_size` | `int` | `1` | Tensor parallel size for inference |
| `release_inference_cache` | `bool` | `false` | Release inference parameter cache after generation |
| `pin_parameters` | `bool` | `true` | Keep inference parameters pinned in GPU memory |
| `tp_gather_partition_size` | `int` | `8` | Partition size for tensor parallel gathering |

### 5.10.4 RLHF Training Pattern

```python
# RLHF training pattern with hybrid engine
for epoch in range(num_epochs):
    # Phase 1: Generate responses (inference)
    model_engine.eval()
    with torch.no_grad():
        prompts = get_prompts()
        responses = model_engine.generate(
            input_ids=prompts,
            max_new_tokens=256,
        )

    # Phase 2: Compute rewards
    rewards = reward_model(responses)
    advantages = compute_advantages(rewards)

    # Phase 3: PPO update (training)
    model_engine.train()
    for ppo_epoch in range(num_ppo_epochs):
        for batch in make_ppo_batches(prompts, responses, advantages):
            outputs = model_engine(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
            )
            loss = compute_ppo_loss(outputs, batch["advantages"])
            model_engine.backward(loss)
            model_engine.step()
```

---

## 5.11 Internal Mechanisms

### 5.11.1 ZeRO Stage Implementation

**ZeRO Stage 1 (Optimizer State Partitioning):**

```
Each GPU stores:
- Full model parameters
- Full gradients (temporarily)
- 1/N of optimizer states (Adam first/second moments)

During step():
1. All-reduce gradients across all GPUs
2. Each GPU updates its optimizer state partition
3. Each GPU updates its parameter partition
4. All-gather updated parameters (or use reduce-scatter)
```

**ZeRO Stage 2 (Gradient + Optimizer State Partitioning):**

```
Each GPU stores:
- Full model parameters
- 1/N of gradients
- 1/N of optimizer states

During backward():
1. Reduce-scatter gradients (each GPU gets 1/N)
2. Free non-owned gradient partitions

During step():
1. Update optimizer states (local partition only)
2. Update parameters (local partition)
3. All-gather updated parameters
```

**ZeRO Stage 3 (Parameter + Gradient + Optimizer State Partitioning):**

```
Each GPU stores:
- 1/N of parameters
- 1/N of gradients
- 1/N of optimizer states

During forward():
1. All-gather parameters layer by layer (just-in-time)
2. Run forward for current layer
3. Free gathered parameters

During backward():
1. All-gather parameters for current layer
2. Compute gradients
3. Reduce-scatter gradients
4. Free gathered parameters

During step():
1. Update optimizer states (local partition)
2. Update parameters (local partition)
```

### 5.11.2 Mixed Precision Implementation

**FP16 Training Flow:**

```
Initialization:
- Create FP16 copy of model parameters
- Create FP32 master copy of parameters
- Configure loss scaler

Forward:
1. Use FP16 parameters for forward pass
2. Compute loss
3. Scale loss by loss_scale

Backward:
1. Compute gradients in FP16
2. Unscale gradients (divide by loss_scale)
3. Check for inf/nan in gradients
4. If overflow: skip optimizer step, reduce loss_scale
5. If no overflow: FP16 gradients -> FP32 gradient copy

Step:
1. Update FP32 master parameters with FP32 gradients
2. Copy FP32 master parameters -> FP16 parameters
3. Update loss scale (increase if no recent overflows)
```

**BF16 Training Flow:**

```
Initialization:
- Create BF16 copy of model parameters
- Create FP32 master copy of parameters

Forward:
1. Use BF16 parameters for forward pass
2. Compute loss

Backward:
1. Compute gradients in BF16
2. Copy to FP32 for optimizer update

Step:
1. Update FP32 master parameters
2. Copy FP32 master -> BF16 parameters

Note: No loss scaling needed for BF16 due to wider dynamic range
```

### 5.11.3 Communication Overlap

When `overlap_comm` is enabled:

```
Timeline (without overlap):
[Forward] -> [Backward Layer N] -> [Backward Layer N-1] -> ... -> [Reduce Gradients] -> [Step]

Timeline (with overlap):
[Forward] -> [Backward Layer N + Reduce Layer N] -> [Backward Layer N-1 + Reduce Layer N-1] -> ... -> [Step]
```

The overlap is achieved by registering gradient hooks on each parameter. When a parameter's gradient is computed during backward, the hook immediately initiates the reduce-scatter for that gradient bucket while the next layer's backward computation proceeds.

### 5.11.4 Bucket-Based Gradient Reduction

DeepSpeed groups parameters into buckets for efficient gradient reduction:

```
Model Parameters (ordered by first-touch during backward):
  Layer 12 output.weight -> [Bucket 3]
  Layer 12 output.bias   -> [Bucket 3]
  Layer 11 output.weight -> [Bucket 3]
  ...
  Layer 6 output.weight  -> [Bucket 2]
  ...
  Layer 1 output.weight  -> [Bucket 1]
  Embedding weight       -> [Bucket 0]

When Bucket 3 is full (reaches reduce_bucket_size):
  -> Trigger reduce-scatter for Bucket 3
  -> Continue backward computation for Bucket 2 parameters
```

### 5.11.5 Parameter Prefetching (ZeRO Stage 3)

For ZeRO Stage 3, parameters are prefetched to hide communication latency:

```
Forward Pass Timeline:
  [Gather Layer 1 params] -> [Compute Layer 1] -> [Gather Layer 2 params] -> [Compute Layer 2] -> ...
                                                        ^
                                                        |
                                          Prefetch starts during Layer 1 compute
```

The prefetch is controlled by `stage3_prefetch_bucket_size`. Parameters needed for upcoming layers are fetched while the current layer is being computed.

---

## 5.12 Advanced Usage Patterns

### 5.12.1 Manual Forward/Backward with Gradient Accumulation

```python
# Manual gradient accumulation control
model_engine.train()

accumulation_steps = 4
for step, batch in enumerate(dataloader):
    inputs, labels = batch[0].cuda(), batch[1].cuda()

    outputs = model_engine(inputs)
    loss = loss_fn(outputs, labels)

    # Divide loss by accumulation steps
    loss = loss / accumulation_steps

    model_engine.backward(loss)

    # Only step every N micro-batches
    if (step + 1) % accumulation_steps == 0:
        model_engine.step()
```

### 5.12.2 Multi-Model Training

```python
# Training multiple models (e.g., GAN, teacher-student)
with deepspeed.zero.Init(config_dict_or_path=ds_config):
    generator = Generator()
    discriminator = Discriminator()

# Initialize separate engines
gen_engine, gen_optim, _, _ = deepspeed.initialize(
    model=generator,
    optimizer=gen_optimizer,
    config_params=ds_config,
)

disc_engine, disc_optim, _, _ = deepspeed.initialize(
    model=discriminator,
    optimizer=disc_optimizer,
    config_params=ds_config,
)

# Training loop
for real_data in dataloader:
    real_data = real_data.cuda()
    fake_data = gen_engine(noise)

    # Train discriminator
    disc_loss = discriminator_loss(disc_engine, real_data, fake_data.detach())
    disc_engine.backward(disc_loss)
    disc_engine.step()

    # Train generator
    gen_loss = generator_loss(disc_engine, fake_data)
    gen_engine.backward(gen_loss)
    gen_engine.step()
```

### 5.12.3 Accessing Model Parameters in ZeRO Stage 3

```python
import deepspeed

# Method 1: Context manager for temporary gathering
with deepspeed.zero.GatheredParameters(model_engine.parameters()):
    # All parameters are gathered on all GPUs
    total_params = sum(p.numel() for p in model_engine.parameters())
    print(f"Total parameters: {total_params}")

# Parameters are freed when exiting the context

# Method 2: Gather specific parameters
with deepspeed.zero.GatheredParameters([model_engine.module.lm_head.weight]):
    weight = model_engine.module.lm_head.weight
    print(f"LM head weight shape: {weight.shape}")

# Method 3: For read-only access (more memory efficient)
@deepspeed.zero.register_external_parameter
def get_embedding_weight(engine):
    return engine.module.embedding.weight
```

### 5.12.4 Custom Training Loop with Engine

```python
# Custom training loop with logging, evaluation, and checkpointing
import time
import deepspeed

model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    training_data=train_dataset,
    config_params=ds_config,
)

best_loss = float("inf")
global_step = 0

for epoch in range(num_epochs):
    model_engine.train()
    epoch_loss = 0.0
    num_batches = 0

    for batch in train_dataloader:
        inputs = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()

        # Forward
        outputs = model_engine(inputs, labels=labels)
        loss = outputs.loss

        # Backward
        model_engine.backward(loss)

        # Step
        stepped = model_engine.step()

        if stepped:
            global_step += 1
            epoch_loss += loss.item()
            num_batches += 1

            # Log every N steps
            if global_step % 100 == 0:
                avg_loss = epoch_loss / max(num_batches, 1)
                lr = model_engine.get_lr()
                mem = model_engine.memory_usage()
                print(
                    f"Epoch {epoch}, Step {global_step}: "
                    f"Loss={avg_loss:.4f}, LR={lr:.2e}, "
                    f"GPU Mem={mem['gpu_allocated']:.2f} GB"
                )

            # Evaluate every N steps
            if global_step % 1000 == 0:
                eval_loss = evaluate(model_engine, eval_dataloader)
                print(f"Eval Loss: {eval_loss:.4f}")

                # Save best model
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    model_engine.save_checkpoint(
                        save_dir="./checkpoints",
                        tag=f"best-step-{global_step}",
                        client_state={
                            "global_step": global_step,
                            "epoch": epoch,
                            "best_loss": best_loss,
                        },
                    )

            # Save periodic checkpoint
            if global_step % 5000 == 0:
                model_engine.save_checkpoint(
                    save_dir="./checkpoints",
                    tag=f"step-{global_step}",
                    client_state={
                        "global_step": global_step,
                        "epoch": epoch,
                    },
                )
```

### 5.12.5 ZeRO Stage 3 Model Initialization

```python
import deepspeed

# For large models that don't fit in GPU memory,
# use zero.Init() to create the model with parameter partitioning

ds_config = {
    "zero_optimization": {"stage": 3},
    "train_batch_size": 64,
    "fp16": {"enabled": True},
}

# Method 1: Context manager
with deepspeed.zero.Init(config_dict_or_path=ds_config):
    # Model parameters are immediately partitioned across GPUs
    model = LargeModel()  # No GPU OOM even for very large models

# Method 2: Decorator
@deepspeed.zero.Init(config_dict_or_path=ds_config)
class LargeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(4096, 4096) for _ in range(100)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Initialize DeepSpeed engine
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config_params=ds_config,
)
```

### 5.12.6 Sharing Parameters in ZeRO Stage 3

```python
import deepspeed

# For models with shared (tied) parameters
with deepspeed.zero.Init(config_dict_or_path=ds_config):
    model = ModelWithSharedWeights()

# Register shared parameters with DeepSpeed
# DeepSpeed needs to know about tied parameters for correct partitioning
class ModelWithSharedWeights(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_embedding = torch.nn.Embedding(50000, 4096)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4096, 4096)])
        self.lm_head = torch.nn.Linear(4096, 50000, bias=False)

    def tie_weights(self):
        # Tie embedding and LM head weights
        self.lm_head.weight = self.shared_embedding.weight

# DeepSpeed handles tied weights automatically when they share
# the same parameter object
```

### 5.12.7 Progressive Layer Drop

```python
# Enable progressive layer dropping during training
# This randomly drops transformer layers during training
# for faster iteration with regularization effect

ds_config = {
    "progressive_layer_drop": {
        "enabled": True,
        "theta": 0.5,         # Layer drop probability
        "gamma": 0.001,       # Minimum layer drop rate
    },
    # ... other config
}
```

### 5.12.8 Using DeepSpeed with torch.compile

```python
import deepspeed
import torch

# Create model
model = MyModel()

# Compile the model with torch.compile (PyTorch 2.0+)
compiled_model = torch.compile(model)

# Initialize DeepSpeed with the compiled model
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=compiled_model,
    config_params=ds_config,
)

# Training loop works as usual
for batch in dataloader:
    outputs = model_engine(batch)
    loss = outputs.loss
    model_engine.backward(loss)
    model_engine.step()
```

---

## 5.13 Engine Event Hooks

### 5.13.1 Available Hooks

DeepSpeed provides event hooks for custom behavior during training:

```python
# Example: Custom callback after each optimizer step
class TrainingCallback:
    def on_step_end(self, engine):
        """Called after each optimizer step."""
        pass

    def on_epoch_end(self, engine):
        """Called after each epoch."""
        pass

    def on_checkpoint_save(self, engine, save_dir):
        """Called after checkpoint save."""
        pass

    def on_checkpoint_load(self, engine, load_dir):
        """Called after checkpoint load."""
        pass
```

### 5.13.2 Wall Clock Breakdown

When `wall_clock_breakdown: true` is configured, DeepSpeed tracks detailed timing:

```python
# Access timing breakdown
if model_engine.wall_clock_breakdown():
    timing = model_engine.timer_names()
    for name in timing:
        print(f"{name}: {model_engine.timer(name):.4f}s")

# Typical timing categories:
# forward       - Forward pass time
# backward      - Backward pass time
# backward_inner - Backward compute time
# backward_allreduce - Gradient reduction time
# step          - Optimizer step time
# allgather     - Parameter gathering (Stage 3)
# reduce_scatter - Gradient partitioning (Stage 2/3)
```

---

## 5.14 Comparison: Engine Types

| Feature | DeepSpeedEngine | PipelineEngine | HybridEngine |
|---------|----------------|---------------|-------------|
| **Use case** | Standard training | Pipeline parallel training | RLHF (train + inference) |
| **Parallelism** | Data + ZeRO | Data + Pipeline + ZeRO | Data + Pipeline + ZeRO + TP inference |
| **Forward** | Full model forward | Per-stage forward | Full or per-stage |
| **Backward** | Full model backward | Per-stage backward | Full or per-stage |
| **Step** | Single optimizer step | Multiple micro-batch steps | Configurable |
| **Inference** | No special support | No special support | Kernel injection, generate() |
| **ZeRO Stage** | 0, 1, 2, 3 | 0, 1 | 0, 1, 2, 3 |
| **Tensor parallel** | Via config | Limited | Full support |
| **Memory** | ZeRO-managed | ZeRO + activation partitioning | ZeRO + inference cache |
| **Best for** | Most training workloads | Very deep models | RLHF, PPO training |

---

## 5.15 Error Handling and Debugging

### 5.15.1 Common Engine Errors

**Error: `RuntimeError: Expected to have finished reduction in the prior call before starting a new one`**

This occurs when communication operations overlap incorrectly:

```python
# Solution: Ensure allreduce completes before next backward
# In config:
# "zero_optimization": {"overlap_comm": false}  # Disable overlap if needed
```

**Error: `AssertionError: attempted to re-partition an already partitioned parameter`**

This occurs when initializing ZeRO Stage 3 without `zero.Init()`:

```python
# Solution: Wrap model creation
with deepspeed.zero.Init(config_dict_or_path=ds_config):
    model = MyModel()
```

**Error: `RuntimeError: CUDA out of memory`**

```python
# Debug memory usage
model_engine.print_memory_usage("before forward")
outputs = model_engine(inputs)
model_engine.print_memory_usage("after forward")
model_engine.backward(loss)
model_engine.print_memory_usage("after backward")
```

### 5.15.2 Debugging Tools

```python
# Enable verbose logging
import os
os.environ["DS_DEBUG"] = "1"

# Enable communication blocking for debugging
os.environ["DS_COMM_BLOCKING"] = "1"

# Enable wall clock breakdown
ds_config = {
    "wall_clock_breakdown": True,
    "dump_state": True,  # Dump engine state on initialization
}

# Profile communication
ds_config = {
    "communication_logging": {"enabled": True},
}
```

### 5.15.3 State Inspection

```python
# Inspect engine state
print(f"ZeRO Stage: {model_engine.zero_optimization_stage}")
print(f"FP16: {model_engine.fp16_enabled}")
print(f"BF16: {model_engine.bf16_enabled}")
print(f"Batch size: {model_engine.train_batch_size}")
print(f"Micro batch size: {model_engine.train_micro_batch_size_per_gpu}")
print(f"Grad accum steps: {model_engine.gradient_accumulation_steps}")
print(f"World size: {model_engine.world_size}")
print(f"Device: {model_engine.device}")

# Check parameter count (works for all ZeRO stages)
with deepspeed.zero.GatheredParameters(model_engine.parameters()):
    total = sum(p.numel() for p in model_engine.parameters())
    trainable = sum(p.numel() for p in model_engine.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
```

---

## 5.16 Engine API Quick Reference

### 5.16.1 Core Training API

| Method | Description |
|--------|-------------|
| `forward(*inputs, **kwargs)` | Forward pass |
| `backward(loss)` | Backward pass |
| `step()` | Optimizer step (with gradient accumulation) |
| `train()` | Set training mode |
| `eval()` | Set evaluation mode |
| `zero_grad()` | Zero gradients |

### 5.16.2 Checkpoint API

| Method | Description |
|--------|-------------|
| `save_checkpoint(save_dir, tag, client_state)` | Save checkpoint |
| `load_checkpoint(load_dir, tag, ...)` | Load checkpoint |

### 5.16.3 Properties

| Property | Type | Description |
|----------|------|-------------|
| `train_batch_size` | `int` | Effective total batch size |
| `train_micro_batch_size_per_gpu` | `int` | Per-GPU micro-batch size |
| `gradient_accumulation_steps` | `int` | Gradient accumulation count |
| `zero_optimization_stage` | `int` | ZeRO stage (0-3) |
| `fp16_enabled` | `bool` | FP16 mode |
| `bf16_enabled` | `bool` | BF16 mode |
| `module` | `Module` | Underlying PyTorch model |
| `optimizer` | `Optimizer` | Wrapped optimizer |
| `device` | `device` | Current device |
| `global_rank` | `int` | Global process rank |
| `local_rank` | `int` | Local GPU rank |
| `world_size` | `int` | Total process count |

### 5.16.4 Utility Methods

| Method | Description |
|--------|-------------|
| `get_lr()` | Get current learning rate |
| `get_grad_norm()` | Get gradient norm |
| `clip_gradients(max_norm)` | Clip gradients |
| `memory_usage()` | Get memory statistics |
| `print_memory_usage(tag)` | Print memory usage |
| `barrier(name)` | Synchronize processes |
| `all_reduce_tensor(tensor, op)` | All-reduce a tensor |
| `is_gradient_accumulation_boundary()` | Check if at accumulation boundary |
| `empty_partition_cache()` | Free ZeRO Stage 3 cache |
| `set_train_batch_size(size)` | Dynamically adjust batch size |
