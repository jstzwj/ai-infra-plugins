# Mixed Precision Training

## Overview

DeepSpeed provides comprehensive support for mixed precision training, enabling the use of lower-precision numeric formats (FP16, BF16) to accelerate training while maintaining model accuracy. Mixed precision training stores master weights in a higher-precision format (FP32) while performing forward and backward computation in a lower-precision format (FP16 or BF16), achieving 2-3x throughput improvement on modern GPUs.

### Numeric Format Reference

| Format | Bits | Exponent | Mantissa | Range | Precision | Bytes/Element |
|--------|------|----------|----------|-------|-----------|---------------|
| FP32 | 32 | 8 | 23 | $\pm 3.4 \times 10^{38}$ | ~7 decimal digits | 4 |
| FP16 | 16 | 5 | 10 | $\pm 6.5 \times 10^{4}$ | ~3 decimal digits | 2 |
| BF16 | 16 | 8 | 7 | $\pm 3.4 \times 10^{38}$ | ~2 decimal digits | 2 |

```
FP32:  [S][EEEE EEEE][MMM MMMM MMMM MMMM MMMM MMMM]   (1-8-23)
FP16:  [S][EEEE E][MMM MMMM MMMM]                       (1-5-10)
BF16:  [S][EEEE EEEE][MMMM MMM]                          (1-8-7)
```

### Why Mixed Precision Works

1. **Memory savings**: FP16/BF16 use 2 bytes per element vs 4 bytes for FP32 -- a 2x reduction in memory for parameters, gradients, and activations.
2. **Throughput**: Tensor cores (NVIDIA) and matrix units (AMD) provide 2-8x higher throughput for FP16/BF16 matrix operations compared to FP32.
3. **Bandwidth**: Halved data size doubles effective memory bandwidth utilization.
4. **Accuracy**: Master weights in FP32 and gradient scaling prevent accuracy loss.

## FP16 Training

FP16 training is the most widely used mixed precision mode in DeepSpeed. It uses the FP16_Optimizer to manage loss scaling and FP32 master weights.

### FP16_Optimizer

```python
# deepspeed/runtime/fp16/fused_optimizer.py
class FP16_Optimizer(object):
    """FP16 optimizer that manages master weights in FP32.
    
    Handles:
    - Maintaining FP32 master weights alongside FP16 model parameters
    - Dynamic loss scaling to prevent gradient underflow
    - Copying gradients from FP16 to FP32 for the optimizer step
    - Copying updated FP32 weights back to FP16 model parameters
    """
    
    def __init__(self,
                 init_optimizer,
                 static_loss_scale=1.0,
                 dynamic_loss_args=None,
                 verbose=True,
                 mpu=None):
        ...
```

### FP16_UnfusedOptimizer

For cases where a fused optimizer is not available:

```python
# deepspeed/runtime/fp16/unfused_optimizer.py
class FP16_UnfusedOptimizer(object):
    """FP16 optimizer without fused CUDA kernels.
    
    Uses standard PyTorch operations instead of custom CUDA kernels
    for gradient copying and weight updates. Slower but more portable.
    """
    
    def __init__(self,
                 init_optimizer,
                 static_loss_scale=1.0,
                 dynamic_loss_args=None,
                 verbose=True,
                 mpu=None):
        ...
```

### Dynamic Loss Scaling (LossScaler)

FP16 has a limited dynamic range ($6 \times 10^{-8}$ to $6.5 \times 10^{4}$). Small gradients can underflow to zero in FP16, causing training to stall. Loss scaling addresses this by multiplying the loss by a large constant before backward, effectively scaling up all gradients. After backward, gradients are divided by the same constant before the optimizer step.

```python
# deepspeed/runtime/fp16/loss_scaler.py
class LossScaler:
    """Dynamic loss scaler for FP16 training.
    
    Automatically adjusts the loss scale factor:
    - If gradients contain inf/nan: reduce loss scale
    - If N consecutive steps have valid gradients: increase loss scale
    
    This ensures gradients remain in FP16 representable range.
    """
    
    def __init__(self,
                 init_scale=2**16,
                 scale_window=1000,
                 min_loss_scale=1,
                 delayed_shift=2,
                 consecutive_hysteresis=False,
                 raise_error_at_min_scale=True):
        self.cur_scale = init_scale
        self.cur_hysteresis = delayed_shift
        self.consecutive_hysteresis = consecutive_hysteresis
        self.raise_error_at_min_scale = raise_error_at_min_scale
        self.scale_window = scale_window
        self.min_loss_scale = min_loss_scale
        
        # Track successful steps for scale increase
        self._num_good_steps = 0
    
    def update_scale(self, overflow):
        """Update loss scale based on whether overflow occurred.
        
        Args:
            overflow: True if inf/nan was detected in gradients
        """
        if overflow:
            # Reduce scale: divide by 2
            self.cur_scale = max(self.cur_scale / 2.0, self.min_loss_scale)
            self._num_good_steps = 0
        else:
            # Track successful step
            self._num_good_steps += 1
            if self._num_good_steps >= self.scale_window:
                # Increase scale: multiply by 2
                self.cur_scale *= 2.0
                self._num_good_steps = 0
```

### Loss Scaling Parameters

| Parameter | Config Key | Default | Description |
|-----------|-----------|---------|-------------|
| Initial scale | `initial_scale_power` | 16 | Initial loss scale = $2^{\text{initial\_scale\_power}}$. Default: $2^{16} = 65536$ |
| Scale window | `loss_scale_window` | 1000 | Number of consecutive successful (no overflow) steps before increasing the scale |
| Hysteresis | `hysteresis` | 2 | Number of consecutive overflow events before reducing the scale. Prevents overly aggressive scale reduction from a single outlier |
| Minimum scale | `min_loss_scale` | 1 | Minimum loss scale. If scale reaches this and overflow persists, training fails |
| Loss scale | `loss_scale` | 0 | Static loss scale. 0 = dynamic (recommended), >0 = fixed scale |

### FP16 Configuration

```json
{
    "fp16": {
        "enabled": true,
        "auto_cast": false,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1,
        "fp16_master_weights_and_grads": false
    }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | false | Enable FP16 training |
| `auto_cast` | bool | false | Use PyTorch's automatic mixed precision (autocast) instead of DeepSpeed's manual FP16 management |
| `loss_scale` | int | 0 | Static loss scale. 0 = dynamic scaling (recommended) |
| `initial_scale_power` | int | 16 | Initial scale = $2^{16} = 65536$. Increase for models with small gradients |
| `loss_scale_window` | int | 1000 | Steps of successful training before scale increase. Larger = more stable scale |
| `hysteresis` | int | 2 | Number of overflows before scale reduction. Higher = more tolerant of occasional overflow |
| `min_loss_scale` | int | 1 | Minimum allowed scale. Set >1 to fail fast if gradients consistently overflow |
| `fp16_master_weights_and_grads` | bool | false | Keep master weights and gradients in FP16 instead of FP32. Saves memory but may hurt convergence |

### Loss Scaling Flow

```
1. Forward Pass:
   loss = model(input)  # Computed in FP16

2. Scale Loss:
   scaled_loss = loss * cur_scale  # Typically 65536

3. Backward Pass:
   scaled_loss.backward()
   # All gradients are scaled by cur_scale

4. Unscale Gradients:
   for param in model.parameters():
       param.grad = param.grad / cur_scale

5. Check for Overflow:
   overflow = contains_inf_or_nan(gradients)

6. Update Scale:
   if overflow:
       cur_scale = max(cur_scale / 2, min_loss_scale)
       skip optimizer step  # Don't update weights with inf/nan gradients
   else:
       # Proceed with optimizer step
       optimizer.step()

7. After scale_window consecutive good steps:
   cur_scale *= 2  # Increase scale
```

### auto_cast Mode

When `auto_cast=true`, DeepSpeed delegates mixed precision to PyTorch's `torch.cuda.amp.autocast`:

```python
# DeepSpeed internally does:
with torch.cuda.amp.autocast():
    loss = model(inputs)

# Loss scaling is still handled by DeepSpeed's LossScaler
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**When to use auto_cast**:
- Working with custom ops that are not FP16-safe
- Need fine-grained control over which ops run in FP16 vs FP32
- Debugging precision issues

**Note**: `auto_cast` mode is not compatible with ZeRO Stage 3 parameter partitioning.

### fp16_master_weights_and_grads

When `true`, master weights and gradients are kept in FP16 instead of FP32:

```python
# Default (fp16_master_weights_and_grads=False):
#   Model params: FP16
#   Master weights: FP32  (separate copy)
#   Gradients: FP32 (after unscaling)
#   Optimizer states: FP32

# With fp16_master_weights_and_grads=True:
#   Model params: FP16
#   Master weights: FP16  (same as model params)
#   Gradients: FP16
#   Optimizer states: FP32
```

**Trade-off**: Saves ~6 bytes per parameter (FP32 master weights + FP32 gradients -> FP16), but may degrade convergence for models sensitive to gradient precision. Not recommended for production training.

## BF16 Training

BF16 (BFloat16) uses the same exponent width as FP32 (8 bits) but reduces mantissa to 7 bits. This gives BF16 the same dynamic range as FP32 ($\pm 3.4 \times 10^{38}$) but lower precision (~2 decimal digits vs ~7 for FP32).

### Key Advantage: No Loss Scaling Required

Because BF16 has the same dynamic range as FP32, gradient underflow is not a concern. This eliminates the need for loss scaling entirely, simplifying training and avoiding the performance overhead of dynamic scale adjustment.

```python
# FP16 training requires loss scaling:
#   scaled_loss = loss * scale  (must manage scale)
# BF16 training does NOT require loss scaling:
#   loss = model(input)  (no scaling needed)
```

### Hardware Requirements

| GPU Architecture | BF16 Support | Tensor Core BF16 |
|-----------------|-------------|-----------------|
| NVIDIA A100 (Ampere) | Yes | Yes (312 TFLOPS) |
| NVIDIA H100 (Hopper) | Yes | Yes (990 TFLOPS) |
| NVIDIA V100 (Volta) | No | No |
| NVIDIA T4 (Turing) | No | No |
| NVIDIA RTX 3090 (Ampere) | Yes | Yes |
| NVIDIA RTX 4090 (Ada) | Yes | Yes |
| AMD MI250 (CDNA2) | Yes | Yes |
| AMD MI300 (CDNA3) | Yes | Yes |

**Minimum requirement**: NVIDIA Ampere (A100, A10, A30, A40) or later, AMD MI200+ series, or Intel GPU with BF16 support.

### BF16 Configuration

```json
{
    "bf16": {
        "enabled": true
    }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | false | Enable BF16 training |
| `bf16_master_weights_and_grads` | bool | false | Keep master weights and gradients in BF16 instead of FP32 |
| `bf16_optimizer_states` | bool | false | Keep optimizer states in BF16 instead of FP32. Further memory savings but may impact convergence |

### bf16_master_weights_and_grads

```json
{
    "bf16": {
        "enabled": true,
        "bf16_master_weights_and_grads": true
    }
}
```

When enabled:
- Model parameters: BF16
- Master weights: BF16 (same as model params, no separate FP32 copy)
- Gradients: BF16
- Optimizer states: FP32 (momentum, variance)

**Memory savings**: Eliminates the FP32 master weight copy, saving $2\Psi$ bytes per parameter.

### bf16_optimizer_states

```json
{
    "bf16": {
        "enabled": true,
        "bf16_optimizer_states": true
    }
}
```

When enabled:
- Optimizer states (momentum, variance) are stored in BF16 instead of FP32
- Further reduces memory usage by $4\Psi$ bytes (2 state tensors x 2 bytes saved each)
- **Warning**: BF16 optimizer states can significantly impact convergence for some models. Use with caution and monitor training loss closely.

### BF16 vs FP16: Decision Matrix

| Factor | FP16 | BF16 |
|--------|------|------|
| Loss scaling required | Yes | No |
| Dynamic range | Limited ($\pm 6.5 \times 10^4$) | Full ($\pm 3.4 \times 10^{38}$) |
| Precision | Higher (10-bit mantissa) | Lower (7-bit mantissa) |
| Hardware requirement | V100+ | A100+ |
| Training stability | Requires careful loss scale tuning | Generally stable |
| Throughput (same hardware) | Same | Same |
| Recommended for | V100 GPUs, older hardware | A100+ GPUs, production training |

## AMP (Apex AMP)

DeepSpeed supports NVIDIA Apex AMP (Automatic Mixed Precision) for backward compatibility:

```json
{
    "amp": {
        "enabled": true,
        "opt_level": "O2"
    }
}
```

### Opt Levels

| Level | Description | Behavior |
|-------|-------------|----------|
| O0 | FP32 training | No mixed precision, baseline |
| O1 | Mixed precision (conservative) | Only whitelist ops in FP16, rest in FP32. Automatic casting |
| O2 | "Almost FP16" | Model parameters in FP16, with FP32 master weights. Batch norm in FP32 |
| O3 | Full FP16 | All ops in FP16 including batch norm. Fastest but may lose accuracy |

### Limitations

- **Not compatible with ZeRO**: Apex AMP cannot be used with ZeRO Stage 1, 2, or 3. DeepSpeed's native FP16/BF16 modes should be used instead.
- **Deprecated**: NVIDIA has deprecated Apex AMP in favor of PyTorch native AMP.
- **Use only when**: You need backward compatibility with existing Apex-based training scripts.

## torch.autocast (Native PyTorch AMP)

DeepSpeed supports PyTorch's native automatic mixed precision through `torch.autocast`:

```json
{
    "fp16": {
        "enabled": true,
        "auto_cast": true
    }
}
```

### How autocast Works

```python
# DeepSpeed wraps forward pass with autocast:
with torch.autocast(device_type='cuda', dtype=torch.float16):
    loss = model(inputs)

# For BF16:
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    loss = model(inputs)
```

PyTorch's autocast automatically determines which operations should run in FP16 and which should remain in FP32 based on safety considerations.

### lower_precision_safe_modules

DeepSpeed allows configuring which module types are safe to run in lower precision:

```json
{
    "fp16": {
        "enabled": true,
        "auto_cast": true,
        "lower_precision_safe_modules": [
            "torch.nn.Linear",
            "torch.nn.Conv1d",
            "torch.nn.Conv2d",
            "torch.nn.Conv3d",
            "torch.nn.GRU",
            "torch.nn.LSTM",
            "torch.nn.RNN"
        ]
    }
}
```

Operations in these modules will be automatically cast to FP16 during the forward pass. Operations not in this list remain in FP32.

### Autocast Compatibility Notes

- Works with ZeRO Stage 1 and Stage 2
- Limited compatibility with Stage 3 (parameter gathering may conflict with autocast context)
- For Stage 3, prefer DeepSpeed's native FP16 mode (auto_cast=false)

## Gradient Accumulation Data Types (grad_accum_dtype)

When using gradient accumulation (accumulating gradients across multiple micro-batches), the data type used for accumulation affects both memory and numerical accuracy:

```json
{
    "gradient_accumulation_dtype": "fp32",
    "gradient_accumulation_steps": 4
}
```

| Value | Description | Memory per Parameter | Numerical Accuracy |
|-------|-------------|---------------------|-------------------|
| `"fp32"` | Accumulate in FP32 | 4 bytes | Full precision accumulation |
| `"fp16"` | Accumulate in FP16 | 2 bytes | May lose small gradients |
| `"bf16"` | Accumulate in BF16 | 2 bytes | Full range, lower precision |
| `null` (default) | Same as training dtype | 2 bytes (FP16/BF16) | Matches training precision |

### When to Use FP32 Accumulation

```json
{
    "fp16": {
        "enabled": true
    },
    "gradient_accumulation_dtype": "fp32",
    "gradient_accumulation_steps": 8
}
```

Use FP32 gradient accumulation when:
- Large number of accumulation steps (> 4), where small gradient contributions may underflow in FP16
- Training with learning rate warmup where early gradients are very small
- Observing training instability during accumulation phases
- Working with models that have large parameter counts and correspondingly small per-GPU gradients

**Memory cost**: FP32 accumulation uses 2x more memory for the gradient accumulator buffer. For a 7B model, this is an additional ~14 GB.

## Loss Scaling Internals

### Static vs Dynamic Scaling

```json
// Static loss scale (fixed value)
{
    "fp16": {
        "enabled": true,
        "loss_scale": 65536
    }
}

// Dynamic loss scale (recommended)
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

### Dynamic Scale Behavior

```
Step  | Scale    | Event
------|---------|------------------------------------------
1     | 65536   | initial_scale = 2^16
2     | 65536   | good step (no overflow)
...   | 65536   | good steps
1000  | 65536   | scale_window reached, increase scale
1001  | 131072  | scale = 2 * 65536
...   | 131072  | good steps
1050  | 131072  | OVERFLOW detected
1051  | 65536   | scale = 131072 / 2
1052  | 65536   | OVERFLOW detected again
1053  | 32768   | scale = 65536 / 2
...   | 32768   | good steps (hysteresis resets)
2050  | 32768   | scale_window reached
2051  | 65536   | scale = 2 * 32768
```

### Overflow Detection

DeepSpeed uses fused CUDA kernels for efficient overflow detection:

```python
# deepspeed/runtime/fp16/fused_optimizer.py
def _check_overflow(self, partition_gradients):
    """Check for inf/nan in gradients using fused kernel."""
    overflow = self.has_overflow(partition_gradients)
    if overflow:
        self._overflow_buf.fill_(0)  # Reset overflow flag
    return overflow

@torch.no_grad()
def has_overflow(self, grads):
    """Check for gradient overflow using a single CUDA kernel.
    
    Scans all gradient tensors for inf/nan values.
    Much faster than checking each tensor individually.
    """
    # Fused overflow check across all gradient partitions
    return has_overflow_cuda(grads, self._overflow_buf)
```

## Precision Handling in ZeRO

### ZeRO Stage 1 and 2 Precision

In Stages 1 and 2, parameters remain in FP16 on all GPUs:

```
FP16 parameters (full on each GPU)
FP16 gradients (full on each GPU, Stage 1; partitioned, Stage 2)
FP32 optimizer states (partitioned):
  - FP32 master weights
  - FP32 momentum (Adam)
  - FP32 variance (Adam)

Forward/Backward: FP16 computation
Optimizer step: FP32 computation on partitioned states
```

### ZeRO Stage 3 Precision

Stage 3 adds complexity because parameters are partitioned:

```
FP16 parameters (partitioned, 1/Nd per GPU)
FP16 gradients (partitioned, 1/Nd per GPU)
FP32 optimizer states (partitioned):
  - FP32 master weights (partitioned)
  - FP32 momentum (partitioned)
  - FP32 variance (partitioned)

Forward: all_gather FP16 params -> compute in FP16 -> discard
Backward: all_gather FP16 params -> compute FP16 grads -> reduce_scatter -> discard
Optimizer: FP32 computation on local partition
```

### FP16 with ZeRO Configuration

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true
    }
}
```

### BF16 with ZeRO Configuration

```json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true
    }
}
```

**Note**: BF16 with ZeRO is simpler because no loss scaling is needed. The loss_scale configuration is ignored when BF16 is enabled.

### Precision Matrix with ZeRO

| Config | Parameters | Gradients | Optimizer States | Loss Scaling |
|--------|-----------|-----------|-----------------|-------------|
| FP16 + Stage 1 | FP16 (full) | FP16 (full) | FP32 (part) | Required |
| FP16 + Stage 2 | FP16 (full) | FP16 (part) | FP32 (part) | Required |
| FP16 + Stage 3 | FP16 (part) | FP16 (part) | FP32 (part) | Required |
| BF16 + Stage 1 | BF16 (full) | BF16 (full) | FP32 (part) | Not needed |
| BF16 + Stage 2 | BF16 (full) | BF16 (part) | FP32 (part) | Not needed |
| BF16 + Stage 3 | BF16 (part) | BF16 (part) | FP32 (part) | Not needed |

## Configuration Examples

### Example 1: FP16 Training (Standard)

```json
{
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 1,
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
    }
}
```

### Example 2: BF16 Training with ZeRO Stage 3

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
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "prefetch_bucket_size": 5e7,
        "max_live_parameters": 1e9,
        "param_persistence_threshold": 1e5
    }
}
```

### Example 3: FP16 with Gradient Accumulation in FP32

```json
{
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 16,
    "gradient_accumulation_dtype": "fp32",
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
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": true,
        "overlap_comm": true
    }
}
```

### Example 4: FP16 with Apex AMP (Legacy)

```json
{
    "train_batch_size": 64,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-3
        }
    },
    "amp": {
        "enabled": true,
        "opt_level": "O2"
    }
}
```

**Note**: This configuration is not compatible with ZeRO. Use native FP16 instead.

### Example 5: BF16 with Memory-Saving Options

```json
{
    "train_batch_size": 512,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 64,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
    },
    "bf16": {
        "enabled": true,
        "bf16_master_weights_and_grads": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "prefetch_bucket_size": 5e7,
        "max_live_parameters": 5e8,
        "param_persistence_threshold": 1e5
    }
}
```

## Performance Comparison

### Throughput (A100-80GB, 8 GPUs, GPT-2 Medium ~350M params)

| Precision | TFLOPS/GPU | Samples/sec | Memory/GPU | Convergence |
|-----------|-----------|-------------|------------|-------------|
| FP32 | 78 | 420 | 28 GB | Baseline |
| FP16 (dynamic scale) | 156 | 840 | 16 GB | Match FP32 |
| FP16 (static scale) | 158 | 850 | 16 GB | Match FP32 (if scale correct) |
| BF16 | 156 | 840 | 16 GB | Match FP32 |
| AMP O2 (Apex) | 155 | 835 | 16 GB | Match FP32 |

### Memory Usage by Precision Mode

| Configuration | Parameters | Gradients | Master Weights | Optimizer States | Total (per param) |
|---------------|-----------|-----------|----------------|-----------------|-------------------|
| FP32 | 4B | 4B | N/A | 8B | 16B |
| FP16 (default) | 2B | 2B | 4B | 8B | 16B |
| FP16 (master+grad FP16) | 2B | 2B | 0B (shared) | 8B | 12B |
| BF16 (default) | 2B | 2B | 4B | 8B | 16B |
| BF16 (master+grad BF16) | 2B | 2B | 0B (shared) | 8B | 12B |
| BF16 (all BF16) | 2B | 2B | 0B (shared) | 4B | 8B |

**B** = bytes, values per parameter element.

### Loss Scaling Overhead

| Component | Overhead | Impact |
|-----------|---------|--------|
| Scale multiplication | Negligible | Single scalar multiply on loss |
| Gradient unscaling | ~0.1% step time | Division of each gradient tensor |
| Overflow check | ~0.5% step time | Fused scan across all gradients |
| Scale adjustment | Negligible | Conditional scalar operation |
| **Total FP16 overhead** | **~0.6% step time** | Negligible compared to 2x throughput gain |

## Troubleshooting

### Common Issues

**Loss scale drops to minimum frequently**:
- Reduce `initial_scale_power` (e.g., from 16 to 12 or 10)
- Increase `loss_scale_window` (e.g., from 1000 to 2000)
- Check for data issues causing extreme gradient values
- Consider switching to BF16 if hardware supports it

**Training diverges with FP16**:
- Try increasing `initial_scale_power` to 20
- Use `gradient_accumulation_dtype: "fp32"`
- Ensure the optimizer epsilon is appropriate (1e-7 for FP16)
- Consider BF16 for more stable training

**BF16 convergence is worse than FP16**:
- BF16 has lower precision (7-bit mantissa vs 10-bit)
- Try using a slightly lower learning rate
- Ensure optimizer states remain in FP32 (avoid `bf16_optimizer_states`)

**Overflow check reports many overflows**:
- Reduce `initial_scale_power`
- Increase `hysteresis` to tolerate more overflows before reducing scale
- Check for NaN inputs or corrupt data

## Key Source Files

| File | Description |
|------|-------------|
| `deepspeed/runtime/fp16/fused_optimizer.py` | FP16_Optimizer, fused FP16 optimizer implementation |
| `deepspeed/runtime/fp16/unfused_optimizer.py` | FP16_UnfusedOptimizer, non-fused variant |
| `deepspeed/runtime/fp16/loss_scaler.py` | LossScaler, dynamic loss scaling logic |
| `deepspeed/runtime/bf16_optimizer.py` | BF16 optimizer support |
| `deepspeed/runtime/engine.py` | DeepSpeedEngine, mixed precision mode selection |
| `deepspeed/runtime/zero/stage1and2.py` | ZeRO Stages 1 & 2 mixed precision handling |
| `deepspeed/runtime/zero/stage3.py` | ZeRO Stage 3 mixed precision handling |
