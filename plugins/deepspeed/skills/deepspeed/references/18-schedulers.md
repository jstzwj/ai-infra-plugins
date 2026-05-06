# DeepSpeed Learning Rate Schedulers

## Overview

DeepSpeed provides built-in learning rate schedulers that are tightly integrated with the training engine. When configured, the scheduler is automatically stepped during `model_engine.step()`, ensuring the learning rate is updated at the correct point in the training loop. DeepSpeed also supports using any PyTorch learning rate scheduler as a custom scheduler.

---

## Built-in Schedulers

DeepSpeed includes four built-in schedulers, all registered in the `DEEPSPEED_SCHEDULERS` list:

```python
DEEPSPEED_SCHEDULERS = [
    "WarmupLR",
    "WarmupDecayLR",
    "OneCycle",
    "LRRangeTest",
]
```

Each scheduler is selected via the `"type"` field in the scheduler configuration:

```json
{
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": 1000
        }
    }
}
```

---

## WarmupLR

`WarmupLR` implements a linear warmup schedule. The learning rate linearly increases from `warmup_min_lr` to `warmup_max_lr` over `warmup_num_steps`, then remains constant at `warmup_max_lr` for the rest of training.

This is the most commonly used scheduler for transformer pre-training. It is often combined with an external decay schedule (e.g., cosine decay applied by modifying the optimizer's learning rate after warmup).

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `warmup_min_lr` | float | `0` | Starting learning rate at the beginning of warmup |
| `warmup_max_lr` | float | `0` | Peak learning rate after warmup is complete |
| `warmup_num_steps` | int | `0` | Number of training steps for the warmup phase |
| `warmup_type` | str | `"linear"` | Warmup curve type. Currently supports `"linear"`. |

### Algorithm

```
if current_step < warmup_num_steps:
    lr = warmup_min_lr + (warmup_max_lr - warmup_min_lr) * current_step / warmup_num_steps
else:
    lr = warmup_max_lr
```

### Learning Rate Curve

```
LR
^
|                    _______________  warmup_max_lr
|                   /
|                  /
|                 /
|                /
|               /
|______________/                        --> Steps
0           warmup_num_steps
```

### Configuration Example

```json
{
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 5e-5,
            "warmup_num_steps": 2000
        }
    }
}
```

### Usage with Gradient Accumulation

When using gradient accumulation, the warmup steps refer to optimizer steps (not micro-batch steps):

```json
{
    "train_batch_size": 4096,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 8,
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 6e-5,
            "warmup_num_steps": 500
        }
    }
}
```

In this example, warmup takes 500 optimizer steps = 500 * 8 = 4000 micro-batch steps.

### Warmup Num Steps Calculation

For a typical GPT pretraining scenario:

```python
# Calculate warmup steps
total_training_samples = 300_000_000_000  # 300B tokens
global_batch_size = 4096
total_steps = total_training_samples // global_batch_size  # ~73M steps

# 1-2% of total steps for warmup
warmup_steps = int(0.01 * total_steps)  # ~730K steps

# In config
config = {
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 6e-5,
            "warmup_num_steps": warmup_steps
        }
    }
}
```

---

## WarmupDecayLR

`WarmupDecayLR` extends `WarmupLR` with a decay phase after the warmup. After the learning rate reaches `warmup_max_lr`, it decays to a minimum value over the remaining training steps.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `warmup_min_lr` | float | `0` | Starting learning rate |
| `warmup_max_lr` | float | `0` | Peak learning rate after warmup |
| `warmup_num_steps` | int | `0` | Number of warmup steps |
| `total_num_steps` | int | `0` | Total number of training steps (warmup + decay) |
| `warmup_type` | str | `"linear"` | Warmup curve type |

### Algorithm

```
if current_step < warmup_num_steps:
    # Linear warmup
    lr = warmup_min_lr + (warmup_max_lr - warmup_min_lr) * current_step / warmup_num_steps
else:
    # Linear decay to 0 (or warmup_min_lr)
    decay_steps = total_num_steps - warmup_num_steps
    remaining = current_step - warmup_num_steps
    lr = warmup_max_lr * (1 - remaining / decay_steps)
    lr = max(lr, warmup_min_lr)
```

### Learning Rate Curve

```
LR
^
|          ____
|         /    \
|        /      \
|       /        \
|      /          \
|     /            \
|____/              \____             --> Steps
0   warmup_num_steps   total_num_steps
```

### Configuration Example

```json
{
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 5e-5,
            "warmup_num_steps": 2000,
            "total_num_steps": 100000
        }
    }
}
```

---

## OneCycle

`OneCycle` implements the 1Cycle learning rate policy (Smith & Topin, 2018). It consists of two phases:

1. **Increase phase**: Learning rate increases from `lr_min` to `lr_max` over `cycle_first_step_size` steps.
2. **Decrease phase**: Learning rate decreases from `lr_max` back to `lr_min` over `cycle_first_step_size` steps (or a different number if `cycle_second_step_size` is specified).

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cycle_min_lr` | float | `0.01` | Minimum learning rate at start and end of cycle |
| `cycle_max_lr` | float | `1.0` | Peak learning rate at the middle of the cycle |
| `decay_lr_rate` | float | `0` | Rate at which to decay the learning rate after the cycle |
| `cycle_first_step_size` | int | `1000` | Number of steps for the increase phase |
| `cycle_second_step_size` | int | `1000` | Number of steps for the decrease phase |
| `cycle_first_stair_count` | int | `500` | Number of stairs in the increase phase (for staircase scheduling) |
| `cycle_second_stair_count` | int | `500` | Number of stairs in the decrease phase |
| `decay_step_size` | int | `1000` | Step size for post-cycle decay |
| `cycle_momentum` | bool | `true` | Whether to cycle momentum inversely to learning rate |
| `cycle_momentum_cfg` | dict | - | Momentum cycle configuration |

### Algorithm

```
# Phase 1: Increase
if step <= cycle_first_step_size:
    pct = step / cycle_first_step_size
    lr = cycle_min_lr + (cycle_max_lr - cycle_min_lr) * pct

# Phase 2: Decrease
elif step <= cycle_first_step_size + cycle_second_step_size:
    pct = (step - cycle_first_step_size) / cycle_second_step_size
    lr = cycle_max_lr - (cycle_max_lr - cycle_min_lr) * pct

# Phase 3: Post-cycle decay
else:
    lr = cycle_min_lr * (1 - decay_lr_rate) ^ ((step - cycle_first_step_size - cycle_second_step_size) / decay_step_size)
```

### Learning Rate Curve

```
LR
^
|       /\
|      /  \
|     /    \
|    /      \
|   /        \____
|  /              \____
|_/                     \___         --> Steps
0   cycle_first   cycle_second
    step_size      step_size
```

### Configuration Example

```json
{
    "scheduler": {
        "type": "OneCycle",
        "params": {
            "cycle_min_lr": 1e-6,
            "cycle_max_lr": 1e-3,
            "cycle_first_step_size": 5000,
            "cycle_second_step_size": 5000,
            "cycle_first_stair_count": 5000,
            "cycle_second_stair_count": 5000,
            "decay_lr_rate": 0.01,
            "decay_step_size": 1000
        }
    }
}
```

### Staircase Mode

When `cycle_first_stair_count` < `cycle_first_step_size`, the scheduler operates in staircase mode where the learning rate stays constant for multiple steps before jumping to the next level:

```json
{
    "scheduler": {
        "type": "OneCycle",
        "params": {
            "cycle_min_lr": 1e-6,
            "cycle_max_lr": 1e-3,
            "cycle_first_step_size": 10000,
            "cycle_second_step_size": 10000,
            "cycle_first_stair_count": 10,
            "cycle_second_stair_count": 10
        }
    }
}
```

This creates a staircase with 10 levels over 10000 steps (each stair lasts 1000 steps).

---

## LRRangeTest

`LRRangeTest` implements a learning rate range test (also known as an LR finder). It linearly increases the learning rate from `lr_min` to `lr_max` over `num_steps` steps. This is used to determine the optimal learning rate for a new model or dataset.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lr_min` | float | `1e-6` | Starting learning rate |
| `lr_max` | float | `1.0` | Ending learning rate |
| `num_steps` | int | `1000` | Total number of steps for the range test |
| `step_mode` | str | `"linear"` | How to increase the LR. `"linear"` or `"exponential"`. |

### Algorithm

```
if step_mode == "linear":
    pct = step / num_steps
    lr = lr_min + (lr_max - lr_min) * pct
elif step_mode == "exponential":
    pct = step / num_steps
    lr = lr_min * (lr_max / lr_min) ^ pct
```

### Learning Rate Curve

```
LR (linear)          LR (exponential)
^                    ^
|            /       |                /
|           /        |              /
|          /         |            /
|         /          |          /
|        /           |        /
|       /            |      /
|      /             |    /
|_____/              |__/
|                    |
+----------> Steps   +----------> Steps
0  num_steps         0  num_steps
```

### Configuration Example

```json
{
    "scheduler": {
        "type": "LRRangeTest",
        "params": {
            "lr_min": 1e-7,
            "lr_max": 1.0,
            "num_steps": 500,
            "step_mode": "linear"
        }
    }
}
```

### How to Use the LR Range Test

1. **Run the test**: Train the model with `LRRangeTest` for `num_steps` steps.
2. **Plot loss vs. learning rate**: Record the loss at each step and plot against the learning rate.
3. **Find the optimal LR**: The optimal learning rate is typically at the point where the loss decreases most steeply, just before it starts to increase.

```python
import deepspeed
import matplotlib.pyplot as plt

# Record losses
losses = []
lrs = []

for batch in dataloader:
    outputs = model_engine(batch)
    loss = outputs.loss
    model_engine.backward(loss)
    model_engine.step()

    losses.append(loss.item())
    lrs.append(model_engine.get_lr())

# Plot
plt.plot(lrs, losses)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('LR Range Test')
plt.savefig('lr_range_test.png')
```

---

## Scheduler Integration with model_engine.step()

When a scheduler is configured in the DeepSpeed config, it is automatically created and integrated into the training engine. The scheduler is stepped inside `model_engine.step()` after each optimizer update.

### Automatic Stepping

```python
import deepspeed

ds_config = {
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 5e-5,
            "warmup_num_steps": 1000
        }
    }
}

model_engine, optimizer, scheduler, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config
)

# The scheduler is returned as the third return value.
# It is automatically stepped by model_engine.step().

for batch in dataloader:
    outputs = model_engine(batch)
    loss = outputs.loss
    model_engine.backward(loss)
    model_engine.step()  # Scheduler is stepped here automatically
```

### Accessing the Current Learning Rate

```python
# Get the current learning rate
current_lr = model_engine.get_lr()
print(f"Current LR: {current_lr}")

# Or from the scheduler directly
current_lr = scheduler.get_lr()
```

### Scheduler Step Counting

The scheduler step count is based on optimizer steps, not micro-batch steps. With gradient accumulation:

```
1 optimizer step = gradient_accumulation_steps micro-batch steps
```

For example, with `gradient_accumulation_steps=4`:
- 4 forward/backward passes occur before 1 optimizer step
- The scheduler is stepped once per optimizer step
- `warmup_num_steps=1000` means 1000 optimizer steps = 4000 micro-batch steps

---

## Custom Scheduler Support

DeepSpeed supports using any PyTorch `torch.optim.lr_scheduler._LRScheduler` or custom scheduler.

### Method 1: Passing a PyTorch Scheduler

```python
import torch
import deepspeed

model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Create a PyTorch scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=10000,
    eta_min=1e-6
)

ds_config = {
    "train_batch_size": 256,
    "fp16": {"enabled": True}
}

model_engine, optimizer, scheduler, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    lr_scheduler=scheduler,  # Pass custom scheduler
    config=ds_config
)

# Training loop - scheduler is NOT auto-stepped when custom
for batch in dataloader:
    outputs = model_engine(batch)
    loss = outputs.loss
    model_engine.backward(loss)
    model_engine.step()

    # Must manually step the scheduler if desired
    scheduler.step()
```

### Method 2: Using a Custom Scheduler Class

```python
class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine schedule with warmup."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        step = self._step_count
        if step < self.warmup_steps:
            # Linear warmup
            scale = step / self.warmup_steps
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * progress))

        return [max(self.min_lr, base_lr * scale) for base_lr in self.base_lrs]

# Usage
scheduler = CosineWarmupScheduler(
    optimizer,
    warmup_steps=2000,
    total_steps=100000,
    min_lr=1e-6
)

model_engine, _, scheduler, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    lr_scheduler=scheduler,
    config=ds_config
)
```

### Method 3: Using HuggingFace Schedulers

```python
from transformers import get_cosine_schedule_with_warmup

# Create HF scheduler
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,
    num_training_steps=100000
)

model_engine, _, scheduler, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    lr_scheduler=scheduler,
    config=ds_config
)

# HF schedulers need manual stepping
for batch in dataloader:
    model_engine.backward(loss)
    model_engine.step()
    scheduler.step()
```

### Important Notes on Custom Schedulers

1. **Auto-stepping**: DeepSpeed auto-steps only its built-in schedulers. Custom schedulers must be manually stepped.
2. **Learning rate reading**: When using a custom scheduler, `model_engine.get_lr()` reads from the scheduler if available.
3. **Checkpointing**: Custom schedulers should implement `state_dict()` and `load_state_dict()` for checkpoint compatibility.
4. **Warmup integration**: If your custom scheduler includes warmup, make sure it accounts for gradient accumulation steps correctly.

---

## Common Scheduler Patterns

### Pattern 1: Warmup + Cosine Decay (Most Common for LLMs)

DeepSpeed does not have a built-in cosine decay scheduler, so use a PyTorch or custom scheduler:

```python
from torch.optim.lr_scheduler import LambdaLR
import math

def get_cosine_warmup_schedule(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Cosine schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)

scheduler = get_cosine_warmup_schedule(
    optimizer,
    warmup_steps=2000,
    total_steps=100000,
    min_lr_ratio=0.1
)
```

### Pattern 2: Warmup + Linear Decay

Use `WarmupDecayLR` for this pattern:

```json
{
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 5e-5,
            "warmup_num_steps": 2000,
            "total_num_steps": 100000
        }
    }
}
```

### Pattern 3: Warmup + Constant

Use `WarmupLR` for this pattern:

```json
{
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 5e-5,
            "warmup_num_steps": 2000
        }
    }
}
```

### Pattern 4: 1Cycle for Fast Convergence

```json
{
    "scheduler": {
        "type": "OneCycle",
        "params": {
            "cycle_min_lr": 1e-6,
            "cycle_max_lr": 1e-3,
            "cycle_first_step_size": 5000,
            "cycle_second_step_size": 5000,
            "cycle_first_stair_count": 5000,
            "cycle_second_stair_count": 5000,
            "decay_lr_rate": 0.0,
            "decay_step_size": 1000
        }
    }
}
```

---

## Full Configuration Examples

### Example 1: GPT Pretraining with Warmup + Constant

```json
{
    "train_batch_size": 4096,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 6e-5,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 6e-5,
            "warmup_num_steps": 2000
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

### Example 2: BERT Fine-Tuning with Warmup + Decay

```json
{
    "train_batch_size": 128,
    "train_micro_batch_size_per_gpu": 32,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 2e-5,
            "warmup_num_steps": 100,
            "total_num_steps": 10000
        }
    },
    "fp16": {
        "enabled": true
    }
}
```

### Example 3: Vision Model with OneCycle

```json
{
    "train_batch_size": 2048,
    "train_micro_batch_size_per_gpu": 32,
    "gradient_accumulation_steps": 2,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.05
        }
    },
    "scheduler": {
        "type": "OneCycle",
        "params": {
            "cycle_min_lr": 1e-6,
            "cycle_max_lr": 1e-3,
            "cycle_first_step_size": 5000,
            "cycle_second_step_size": 45000,
            "cycle_first_stair_count": 5000,
            "cycle_second_stair_count": 45000,
            "decay_lr_rate": 0.01,
            "decay_step_size": 1000
        }
    },
    "fp16": {
        "enabled": true
    }
}
```

### Example 4: LR Range Test

```json
{
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 32,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0
        }
    },
    "scheduler": {
        "type": "LRRangeTest",
        "params": {
            "lr_min": 1e-7,
            "lr_max": 10.0,
            "num_steps": 500,
            "step_mode": "exponential"
        }
    },
    "fp16": {
        "enabled": true
    }
}
```

---

## Best Practices

1. **Always warm up**: For transformer models, always use a warmup phase (typically 1-2% of total steps). Starting with a high learning rate can destabilize training.

2. **Use cosine or linear decay after warmup**: Constant learning rate after warmup is acceptable for fine-tuning but not for pre-training. Use cosine or linear decay for pre-training.

3. **Scale warmup with batch size**: Larger batch sizes typically require longer warmup. A good rule of thumb is `warmup_steps = max(500, total_steps * 0.01)`.

4. **Consider 1Cycle for smaller models**: The 1Cycle policy often provides faster convergence for models with fewer than 1 billion parameters.

5. **Use LRRangeTest for new architectures**: When training a new architecture or using a new dataset, an LR range test helps identify the optimal learning rate range quickly.

6. **Account for gradient accumulation**: Remember that scheduler steps correspond to optimizer steps, not micro-batch steps. Adjust `warmup_num_steps` and `total_num_steps` accordingly.

7. **Monitor learning rate**: Log the learning rate at each step to verify the schedule is correct:
```python
if global_step % 100 == 0:
    current_lr = model_engine.get_lr()
    print(f"Step {global_step}: LR = {current_lr}")
    wandb.log({"train/lr": current_lr}, step=global_step)
```
